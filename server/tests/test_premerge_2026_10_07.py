"""Gates for the pre-merge review of 2026-10-07 (audit-fixes -> main: the reader-P3 batches).

Thirteen findings, all in code the branch wrote, none a blocker. The ones with a Python half
are driven here; the C++ halves are console legs (`--browser-listing`, `--dock-view-keys`,
`--worker-outlives`, `--glow-truth`) and the source pins at the foot of this file. Every test
that asserts behaviour was run on the unfixed tree first. The record is
docs/audits/full-2026-10-02/readers/P3-fixes.md (Pre-merge review, 2026-10-07).
"""
from __future__ import annotations

import errno
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from mix_engine import dsp
from mix_engine import models as engine_models
from mix_engine.session import Session
from orchestrator import daw_bridge
from orchestrator.device_auth import (MSG_KEYCHAIN_SILENT, DeviceAuthError, DeviceLogin,
                                      MemoryKeychain)

SR = 48000
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "daw" / "Source"
NOW = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)


def _write(path: Path, audio: np.ndarray, sr: int = SR) -> None:
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sr, subtype="FLOAT")


def _tone(hz: float, seconds: float = 2.0, level: float = 0.2) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return level * np.sin(2.0 * np.pi * hz * t)


def _project(folder: Path, stems: dict[str, np.ndarray]) -> Path:
    folder.mkdir(parents=True)
    for name, audio in stems.items():
        _write(folder / f"{name}.wav", audio)
    return folder


# ---------------------------------------------------------------------------------------------
# 2 - a volume that cannot sync a FILE does not fail the save (it already forgave the folder)
# ---------------------------------------------------------------------------------------------

def _no_sync(monkeypatch) -> list[int]:
    """`os.fsync` as a volume that does not support it answers: ENOTSUP, every time."""
    calls: list[int] = []

    def refuse(fd):
        calls.append(fd)
        raise OSError(errno.ENOTSUP, "Operation not supported")

    monkeypatch.setattr(os, "fsync", refuse)
    return calls


def test_a_session_still_saves_on_a_volume_that_cannot_sync(tmp_path, monkeypatch):
    """Py7 put an fsync in front of the session's rename. The folder's sync was forgiven on a
    volume that cannot do one; the file's was not, so on such a volume (some network shares)
    every save of the session raised - and before the batch every one of them had landed."""
    proj = _project(tmp_path / "proj", {"vox": _tone(330.0), "bed": _tone(110.0)})
    session = Session(proj)
    calls = _no_sync(monkeypatch)
    session.user_edit("vox", {"gain_db": -3.0})
    assert calls, "the control: the save did ask for a sync"
    on_disk = json.loads((proj / "session.json").read_text())
    assert on_disk["stems"]["vox"]["gain_db"] == -3.0
    assert not list(proj.glob("*.tmp")), "the temp file was left behind"


def test_the_brains_small_files_still_land_on_a_volume_that_cannot_sync(tmp_path, monkeypatch):
    target = tmp_path / "runs" / "session_digest.jsonl"
    target.parent.mkdir()
    target.write_text("old\n")
    calls = _no_sync(monkeypatch)
    daw_bridge.write_atomic(target, "new\n")
    assert calls, "the control: the write did ask for a sync"
    assert target.read_text() == "new\n"
    assert [p.name for p in target.parent.iterdir()] == ["session_digest.jsonl"]


def test_a_write_that_really_fails_is_still_an_error(tmp_path, monkeypatch):
    """The control for both: forgiving a sync is not forgiving the write."""
    target = tmp_path / "gone" / "digest.jsonl"           # no such folder
    with pytest.raises(OSError):
        daw_bridge.write_atomic(target, "new\n")


# ---------------------------------------------------------------------------------------------
# 4 - a padded stem is named by EVERY render, with the stem cache on (the gateway's path)
# ---------------------------------------------------------------------------------------------

def test_a_short_stem_is_named_by_every_render_with_the_stem_cache_on(tmp_path):
    """A stem of a few samples is padded and named (`short_stems`). The name was written only
    where the file is read, and the padded render went into the persistent stem cache: with
    the cache on, the second render of the project hit it and said nothing. A mended
    (non-finite) stem was already kept out of the cache for exactly this reason."""
    proj = _project(tmp_path / "proj", {"pad": _tone(330.0)})
    _write(proj / "blip.wav", np.full((3, 2), 0.25))
    Session(proj)                                           # the folder door takes it as a stem
    dsp.configure_render_cache(64)
    try:
        said = []
        for _ in range(3):
            report: dict = {}
            dsp.render_master(Session(proj, read_only=True), report=report)
            said.append(report.get("short_stems"))
        assert said == [["blip"]] * 3, said
        # the control: an ordinary stem IS served from the cache the second time. (The master
        # memo beside the stem cache - Phase 54 - keeps the whole render WITH its report, which
        # is why the three renders above still name the stem; it is emptied here so the stems
        # are asked for again.)
        dsp.MASTER_MEMO.clear()
        before = dsp.RENDER_COUNTS["stem"]
        report = {}
        dsp.render_master(Session(proj, read_only=True), report=report)
        assert dsp.RENDER_COUNTS["stem"] - before == 1, "only the short stem is rendered again"
        assert report.get("short_stems") == ["blip"]
    finally:
        dsp.configure_render_cache(0)
        dsp.RENDER_CACHE.clear()


# ---------------------------------------------------------------------------------------------
# 7 - a refresh token the keychain would not take: the code is spent, and nothing is left behind
# ---------------------------------------------------------------------------------------------

class _Response:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.content = b"{}" if payload is not None else b""

    def json(self):
        return self._payload


class _Http:
    def __init__(self, **by_path):
        self.by_path = {k: list(v) for k, v in by_path.items()}
        self.calls: list[tuple[str, dict]] = []

    def post(self, url, json=None, **kwargs):
        self.calls.append((url, json or {}))
        for fragment, queue in self.by_path.items():
            if fragment in url:
                return queue.pop(0) if len(queue) > 1 else queue[0]
        raise AssertionError(f"unexpected POST to {url}")

    def posted(self, fragment: str) -> list[dict]:
        return [body for url, body in self.calls if fragment in url]


def _refresh(n: int) -> str:
    return "refresh-token-value-" + str(n)       # assembled: no token-shaped literal in the tree


class _LockedKeychain(MemoryKeychain):
    """The login keychain while its unlock prompt stands: it reads, and a write is not answered."""

    def set(self, secret: str) -> None:
        raise DeviceAuthError(MSG_KEYCHAIN_SILENT)


def _approved_http():
    return _Http(
        code=[_Response(200, {"device_code": "dev-secret", "user_code": "T5G3-SS9Q",
                              "verification_uri": "https://zammoth.test/link",
                              "expires_in": 600, "interval": 5})],
        token=[_Response(200, {"access_token": "access-2", "expires_in": 3600,
                               "refresh_token": _refresh(2)})],
        revoke=[_Response(200, {"revoked": True})])


def test_a_token_the_keychain_would_not_store_ends_the_session_it_belongs_to():
    """The service has claimed the code and made a session by the time the token is stored.
    A store that failed left the code armed (the next poll presented a CLAIMED code, which
    ends the session and reads "its reply did not reach this computer"), the new session's
    access token in memory, and - when the app took the refusal as final - a session nobody
    held the token of, alive for its ninety days."""
    http = _approved_http()
    login = DeviceLogin("https://api.test", keychain=_LockedKeychain(), http=http, now=lambda: NOW)
    login.begin()
    with pytest.raises(DeviceAuthError, match="keychain"):
        login.poll_once()
    assert http.posted("/v1/auth/device/revoke") == [{"refresh_token": _refresh(2)}], \
        "the session the code made was not ended"
    assert login._access_token is None, "the unstored session's access token is still held"
    with pytest.raises(DeviceAuthError, match="no login in progress"):
        login.poll_once()
    assert len(http.posted("/v1/auth/device/token")) == 1, "the spent code was presented again"


def test_a_relink_the_keychain_refused_leaves_the_old_link_as_it_was():
    http = _approved_http()
    keychain = _LockedKeychain()
    MemoryKeychain.set(keychain, _refresh(1))               # this machine is already linked
    login = DeviceLogin("https://api.test", keychain=keychain, http=http, now=lambda: NOW)
    login.begin()
    with pytest.raises(DeviceAuthError, match="keychain"):
        login.poll_once()
    assert keychain.get() == _refresh(1)
    assert http.posted("/v1/auth/device/revoke") == [{"refresh_token": _refresh(2)}], \
        "only the session that could not be stored is ended - never the one still stored"


def test_a_store_that_fails_is_still_said_when_the_revoke_cannot_be_sent():
    http = _approved_http()
    del http.by_path["revoke"]                              # a POST there raises
    login = DeviceLogin("https://api.test", keychain=_LockedKeychain(), http=http, now=lambda: NOW)
    login.begin()
    with pytest.raises(DeviceAuthError, match="keychain"):
        login.poll_once()


def test_a_link_the_keychain_takes_is_linked_as_before():
    """The control."""
    http = _approved_http()
    keychain = MemoryKeychain()
    login = DeviceLogin("https://api.test", keychain=keychain, http=http, now=lambda: NOW)
    login.begin()
    assert login.poll_once() == "linked"
    assert keychain.get() == _refresh(2) and login._access_token == "access-2"
    assert http.posted("/v1/auth/device/revoke") == []


# ---------------------------------------------------------------------------------------------
# 11 - `get_chains` / `get_strip` are allowed in an unwritten folder, so they answer in one
# ---------------------------------------------------------------------------------------------

def test_the_two_reads_the_app_may_make_of_an_unwritten_folder_answer(tmp_path, monkeypatch):
    """Both are in `_LEAVES_FOLDER_ALONE` (they make nothing a project) and both read
    session.json - which an unwritten folder does not have: FileNotFoundError, as an error
    reply, to the re-read the app makes after flushing the user's queued moves."""
    monkeypatch.setenv("MIX_AGENT_DATA_DIR", str(tmp_path / "appdata"))
    from orchestrator import daw_server

    proj = _project(tmp_path / "proj", {"drums": _tone(110.0, 0.6), "bass": _tone(55.0, 0.7)})
    before = sorted(p.name for p in proj.iterdir())
    brain = daw_server.Brain(proj)
    brain.open_session(daw=True, stems_from_folder=True, unwritten=True)
    assert brain.unwritten is True
    assert {"get_chains", "get_strip"} <= daw_server._LEAVES_FOLDER_ALONE

    chains = brain.get_chains()
    assert sorted(k for k in chains if k != "master" and ":" not in k) == ["bass", "drums"]
    strip = brain.get_strip()
    assert sorted(strip) == ["bass", "drums"]
    assert sorted(p.name for p in proj.iterdir()) == before, "a read wrote into the folder"

    # the control: once it is a project they read the file, as they always did
    brain.become_project()
    assert sorted(brain.get_strip()) == ["bass", "drums"]
    assert (proj / "session.json").exists()


# ---------------------------------------------------------------------------------------------
# 12 - the analysis version's note says the harmonics the code leaves out
# ---------------------------------------------------------------------------------------------

def test_the_analysis_version_note_names_the_harmonics_the_code_uses():
    from mix_engine import analysis

    text = Path(engine_models.__file__).read_text()
    above = text[:text.index("ANALYSIS_VERSION = ")]        # the notes sit above the constant
    said = re.findall(r"harmonic \(2\.\.(\d+)\)", above)
    assert said == [str(analysis.RESONANCE_HARMONIC_MAX)], said


# ---------------------------------------------------------------------------------------------
# The C++ halves that can be read: daw/Source
# ---------------------------------------------------------------------------------------------

def _between(text: str, start: str, end: str) -> str:
    a = text.index(start)
    return text[a:text.index(end, a)]


def test_every_item_of_the_two_pad_menus_acts_through_the_one_function():
    """Finding 5: `slotMenuInvoke` (the scene found when the item is PICKED) was called by its
    leg alone - the two menus carried thirteen inline copies of its body, so the leg proved a
    function no menu item ran."""
    text = (SOURCE / "LauncherPanel.h").read_text()
    menus = _between(text, "void showSlotMenu (", "void showQuantiseMenu()")
    assert "safe->invoke (" not in menus, "a menu item still invokes on its own"
    assert menus.count("slotMenuSceneOrSay (target) : -1") == 0
    # thirteen items and helpers: twelve through the callback's door, the file chooser's directly
    assert menus.count("slotMenuPick (safe") == 12 and menus.count("slotMenuInvoke (") == 1
    assert "safe->slotMenuInvoke (id, target" in _between(text, "static void slotMenuPick (", "\n    }\n")
    assert "void showChannelLoopMenu (" in menus


def test_the_length_cache_never_asks_the_disk_for_a_files_time():
    """Finding 3: with no time handed in, `audioSecondsCached` asked the file for one - on the
    message thread, for every favorite, recent and search hit, at every rebuild."""
    text = (SOURCE / "FileBrowserPanel.h").read_text()
    body = _between(text, "double audioSecondsCached (", "\n    }\n")
    assert "getLastModificationTime" not in body and "fileTimeMs" not in body
    # ...and every read of a file's time in the panel is the one counted function's
    counted = _between(text, "static juce::int64 fileTimeMs (", "\n    }\n")
    assert "++diskAsksForTest" in counted and counted.count("getLastModificationTime()") == 1
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("//"))
    assert code.count("getLastModificationTime()") == 1, "a file's time is read outside fileTimeMs"


def test_a_length_is_filed_under_the_time_its_row_asked_with():
    """Found on the way: the length worker read the file's time again to file its answer, so a
    file changed since its row's time was read was filed under a key no row asks with - and
    asked for again at every rebuild the answer caused, without end (`--browser-listing` T d, T e)."""
    text = (SOURCE / "FileBrowserPanel.h").read_text()
    worker = _between(text, "void fillDurationsAsync()", "void rebuildTree()")
    assert "juce::String (ask.mtimeMs)" in worker
    code = "\n".join(line for line in worker.splitlines() if not line.lstrip().startswith("//"))
    assert "fileTimeMs (" not in code and "getLastModificationTime" not in code
    assert text.count("durationMisses.add ({ f, mtimeMs })") == 3, "a row asks for its length without its time"


def test_the_bank_power_memory_joins_the_three_doors_the_folds_use():
    """Finding 9 (CLAUDE.md, M19 rule 6): view state that names a track by NAME follows it."""
    text = (SOURCE / "DeviceChainView.h").read_text()
    for door in ("void foldsFollowRemoval (", "void foldsFollowRename (", "void forgetFolds ("):
        body = _between(text, door, "\n    }\n")
        assert "bankOnMemory" in body or "bankMemory" in body, f"{door.strip()} leaves the EQ box's memory behind"


def test_the_red_switch_of_this_round_is_gone():
    """The old path of each C++ fix was kept behind one temporary switch for the red run."""
    left = [p.name for p in sorted(SOURCE.glob("*")) if p.is_file() and "pm7red" in p.read_text(errors="ignore")]
    assert left == [], f"the temporary red switch is still in: {left}"
