"""P3 batch T2 - both sides of the wire (full audit 2026-10-02, readers' P3 findings).

The brain's halves. Each test was written against the unfixed tree and seen red there; the
record is docs/audits/full-2026-10-02/readers/P3-fixes.md, batch T2.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import licence_signing
import numpy as np
import pytest
import soundfile as sf

from orchestrator import api_key, license_state
from orchestrator.device_auth import DeviceLogin, MemoryKeychain

REPO = Path(__file__).resolve().parents[2]
GOOD = "sk-ant-api03-" + "a" * 40
NOW = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)


class _Response:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.content = b"{}" if payload is not None else b""

    def json(self):
        return self._payload


class _Http:
    """Answers the control plane's device endpoints from a queued script, and keeps the calls."""

    def __init__(self, **by_path):
        self.by_path = {k: list(v) for k, v in by_path.items()}
        self.calls: list[tuple[str, dict]] = []
        self.fail: set[str] = set()

    def post(self, url, json=None, **kwargs):
        self.calls.append((url, json or {}))
        for fragment, queue in self.by_path.items():
            if fragment in url:
                if fragment in self.fail:
                    raise ConnectionError("offline")
                return queue.pop(0) if len(queue) > 1 else queue[0]
        raise AssertionError(f"unexpected POST to {url}")


class _OkClient:
    class models:  # noqa: N801 - the SDK's attribute name
        @staticmethod
        def retrieve(_model):
            return object()


@pytest.fixture
def signed_in(monkeypatch, tmp_path):
    """A shipped install's shape: a key in the (memory) keychain, a linked machine, a scratch
    data dir for the licence cache. Yields the login's fake HTTP."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("MIX_AGENT_LLM_BACKEND", raising=False)
    monkeypatch.delenv("MIX_AGENT_LICENSE", raising=False)
    monkeypatch.setenv("MIX_AGENT_DATA_DIR", str(tmp_path / "appdata"))
    monkeypatch.setenv("MIX_AGENT_CLOUD_URL", "https://meter.example.test")
    api_key.use_store(api_key.memory_store())
    license_state.reset()
    from orchestrator import device_auth
    device_auth.reset_shared()
    http = _Http(revoke=[_Response(200, {"revoked": True})])
    login = DeviceLogin("https://meter.example.test", keychain=MemoryKeychain(), http=http,
                        now=lambda: NOW)
    login._keychain.set("r" * 40)
    device_auth._shared = login
    api_key.set_api_key(GOOD, client_factory=lambda k: _OkClient())
    yield http
    api_key.use_store(None)
    license_state.reset()
    device_auth.reset_shared()


# ---------------------------------------------------------------------------------------------
# S8·R8 - signing out
# ---------------------------------------------------------------------------------------------

def test_signing_out_ends_the_cached_licence_on_this_machine_at_once(signed_in, demo_project):
    """`auth_sign_out` ended with `license_state.reset()`, which forgets the copy in memory and
    leaves license.json - and a signed-out machine believes a `licensed` cache for thirty days.
    An explicit sign-out is not a lapsed session (James, 2026-10-06: it forgets the cache)."""
    from orchestrator.daw_server import Brain

    license_state._write_cache(licence_signing.cache({"status": "licensed", "checked_at": time.time() - 60}))
    assert license_state._cache_path().exists()
    res = Brain(demo_project).auth_sign_out()
    assert res["ok"] is True
    assert not license_state._cache_path().exists(), "the licence cache outlived the sign-out"
    st = license_state.state()
    assert st["status"] == "none" and st["gated"] is True
    assert license_state.refusal(), "the engineer still runs after a sign-out"
    # the reply says what the app's gate should now show, so it need not ask again
    assert res["license"]["status"] == "none"


def test_a_lapsed_sign_in_still_keeps_its_licence(signed_in):
    """The control: a session that merely LAPSED (the token is gone, nobody signed out) keeps
    the confirmed licence for the offline grace - `forget` is the sign-out's alone."""
    from orchestrator import device_auth

    license_state._write_cache(licence_signing.cache({"status": "licensed", "checked_at": time.time() - 60}))
    device_auth._shared._keychain.delete()          # lapsed: no token, no sign-out
    license_state.reset()
    st = license_state.state()
    assert st["status"] == "licensed" and st["message"] == license_state.MSG_RESIGN_IN
    assert license_state._cache_path().exists()


def test_a_sign_out_the_keychain_refused_keeps_the_licence_cache(signed_in, demo_project,
                                                                 monkeypatch):
    """"Signed out" was not true (the token is still stored): nothing is forgotten."""
    from orchestrator import device_auth
    from orchestrator.daw_server import Brain

    license_state._write_cache(licence_signing.cache({"status": "licensed", "checked_at": time.time() - 60}))

    def refuse():
        raise device_auth.DeviceAuthError("the keychain did not answer")
    monkeypatch.setattr(device_auth._shared._keychain, "delete", refuse)
    res = Brain(demo_project).auth_sign_out()
    assert res["ok"] is False and "keychain" in res["message"]
    assert license_state._cache_path().exists()


def _link(keychain, http):
    login = DeviceLogin("https://api.test", keychain=keychain, http=http, now=lambda: NOW)
    login.begin()
    assert login.poll_once() == "linked"
    return login


def _link_http(**extra):
    return _Http(
        code=[_Response(200, {"device_code": "dev-secret", "user_code": "T5G3-SS9Q",
                              "verification_uri": "https://zammoth.test/link",
                              "expires_in": 600, "interval": 5})],
        token=[_Response(200, {"access_token": "access-2", "expires_in": 3600,
                               "refresh_token": "refresh-token-value-2"})],
        **extra)


def test_a_relink_revokes_the_session_it_replaces():
    """Re-link wrote the new refresh token over the old one and left the old server session
    alive for its ninety days, with no way to end it from this machine."""
    keychain = MemoryKeychain()
    keychain.set("refresh-token-value-1")
    http = _link_http(revoke=[_Response(200, {"revoked": True})])
    _link(keychain, http)
    assert keychain.get() == "refresh-token-value-2"
    revokes = [body for url, body in http.calls if url.endswith("/v1/auth/device/revoke")]
    assert revokes == [{"refresh_token": "refresh-token-value-1"}]


def test_a_first_link_revokes_nothing():
    http = _link_http()                 # no revoke endpoint: a call to it would raise
    keychain = MemoryKeychain()
    _link(keychain, http)
    assert keychain.get() == "refresh-token-value-2"


def test_a_relink_still_lands_when_the_old_session_cannot_be_revoked():
    keychain = MemoryKeychain()
    keychain.set("refresh-token-value-1")
    http = _link_http(revoke=[_Response(200, {"revoked": True})])
    http.fail.add("revoke")
    _link(keychain, http)
    assert keychain.get() == "refresh-token-value-2"


# ---------------------------------------------------------------------------------------------
# S3·R10 - Stop on a stem split (James, 2026-10-06: the split runs in a child process)
# ---------------------------------------------------------------------------------------------

def _song(folder: Path, name: str = "song.wav", seconds: float = 1.0) -> Path:
    sr = 44100
    t = np.arange(int(sr * seconds)) / sr
    audio = np.stack([0.3 * np.sin(2 * np.pi * 220 * t), 0.2 * np.sin(2 * np.pi * 330 * t)], axis=1)
    path = folder / name
    sf.write(str(path), audio.astype(np.float32), sr, subtype="FLOAT")
    return path


def _standin_worker(monkeypatch, tmp_path, body: str) -> Path:
    """A stand-in for the split's worker: `body` is Python that runs with `src`, `out` (a Path)
    and `stems` bound, in a process of its own. Returns the file the stand-in writes its pid to."""
    from mix_engine import separation

    pidfile = tmp_path / "worker.pid"
    script = tmp_path / "standin_worker.py"
    script.write_text(
        "import json, os, sys, time\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "src, out, stems = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])\n"
        f"Path({str(pidfile)!r}).write_text(str(os.getpid()))\n"
        + body)
    monkeypatch.setattr(separation, "worker_argv",
                        lambda: ([sys.executable, str(script)], dict(os.environ)))
    return pidfile


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    # a killed child we have reaped is gone; one nobody reaped would still answer signal 0
    return True


def test_a_split_in_a_process_of_its_own_gives_the_stems_the_in_process_split_gives(
        tmp_path, monkeypatch):
    from mix_engine.separation import FakeSeparator
    from orchestrator.daw_bridge import separate_track

    monkeypatch.setenv("MIX_AGENT_SEPARATOR", "fake")
    proj = tmp_path / "proj"
    proj.mkdir()
    src = _song(proj)
    stages: list[str] = []
    result = separate_track(proj, str(src), stems=4, in_child=True,
                            on_stage=lambda e: stages.append(e["stage"]))
    assert result["ok"] is True, result
    assert [s["name"] for s in result["stems"]] == ["vocals", "drums", "bass", "other"]
    audio, sr = sf.read(str(src), dtype="float32", always_2d=True)
    want = FakeSeparator(4).separate(audio, sr)
    for stem in result["stems"]:
        have, have_sr = sf.read(str(proj / stem["wav"]), dtype="float32", always_2d=True)
        assert have_sr == sr
        assert np.allclose(have, np.asarray(want[stem["name"]]).reshape(have.shape), atol=1e-6)
    assert stages.index("loading") < stages.index("separating") < stages.index("writing")


def test_stop_ends_a_split_at_once_whatever_the_model_is_doing(tmp_path, monkeypatch):
    """The inference was one uninterruptible call in the brain's own process, so a Stop could
    only land between stages - minutes away - and the app's 45 s grace then killed the brain
    every open project shares. In a child, Stop is a kill."""
    from orchestrator.daw_bridge import separate_track

    pidfile = _standin_worker(monkeypatch, tmp_path, "time.sleep(25)\n")
    proj = tmp_path / "proj"
    proj.mkdir()
    src = _song(proj)
    stop = threading.Event()
    threading.Timer(0.6, stop.set).start()
    t0 = time.monotonic()
    result = separate_track(proj, str(src), stems=4, in_child=True, should_cancel=stop.is_set)
    took = time.monotonic() - t0
    assert result == {"ok": False, "error": "cancelled"}
    assert took < 5.0, f"the stop took {took:.1f} s"
    pid = int(pidfile.read_text())
    assert not _alive(pid), "the split's process outlived the stop"
    assert sorted(p.name for p in proj.iterdir()) == ["song.wav"], "a stopped split left files"


def test_a_worker_that_dies_is_an_answer_not_a_hang(tmp_path, monkeypatch):
    from orchestrator.daw_bridge import separate_track

    _standin_worker(monkeypatch, tmp_path, "sys.exit(7)\n")
    proj = tmp_path / "proj"
    proj.mkdir()
    result = separate_track(proj, str(_song(proj)), stems=4, in_child=True)
    assert result["ok"] is False and "stopped" in result["error"] and "7" in result["error"]


def test_a_missing_model_in_the_worker_is_still_the_clean_no_model_error(tmp_path, monkeypatch):
    from mix_engine.separation import ModelNotAvailable
    from orchestrator.daw_bridge import separate_track

    _standin_worker(monkeypatch, tmp_path,
                    "(out / 'done.json').write_text(json.dumps("
                    "{'ok': False, 'kind': 'no_model', 'message': 'no weights here'}))\n")
    proj = tmp_path / "proj"
    proj.mkdir()
    with pytest.raises(ModelNotAvailable, match="no weights here"):
        separate_track(proj, str(_song(proj)), stems=4, in_child=True)


def test_the_download_is_its_own_stage_in_the_child_too(tmp_path, monkeypatch):
    import mix_engine.separation as sep_mod
    from orchestrator.daw_bridge import separate_track

    class NotCached:
        stems, names, in_child = 4, sep_mod.FOUR_STEM, True
        def models_cached(self):  # noqa: D102, E301
            return False
        def download_detail(self):  # noqa: D102, E301
            return "AI model htdemucs - about 166 MB, first use only"

    monkeypatch.setattr(sep_mod, "get_separator", lambda stems, kind=None: NotCached())
    _standin_worker(
        monkeypatch, tmp_path,
        "time.sleep(0.4)\n"
        "(out / 'stage.separating').touch()\n"
        "time.sleep(0.4)\n"
        "n = 44100\n"
        "for name in ('vocals', 'drums', 'bass', 'other'):\n"
        "    np.save(out / (name + '.npy'), np.zeros((n, 2), dtype=np.float32))\n"
        "(out / 'done.json').write_text(json.dumps({'ok': True, 'sample_rate': 44100}))\n")
    proj = tmp_path / "proj"
    proj.mkdir()
    seen: list[tuple[str, float]] = []
    t0 = time.monotonic()
    result = separate_track(proj, str(_song(proj)), stems=4,
                            on_stage=lambda e: seen.append((e["stage"], time.monotonic() - t0)))
    assert result["ok"] is True, result
    names = [s for s, _ in seen]
    assert names.index("downloading") < names.index("separating") < names.index("writing")
    when = dict(seen)
    assert when["separating"] - when["downloading"] > 0.2, "the stage did not wait for the download"


def test_the_shipping_separators_run_in_a_child_and_the_stand_in_does_not():
    from mix_engine.separation import (DemucsOnnxSeparator, FakeSeparator,
                                       SpleeterOnnxSeparator)

    assert DemucsOnnxSeparator.in_child is True and SpleeterOnnxSeparator.in_child is True
    assert getattr(FakeSeparator, "in_child", False) is False


def test_the_real_worker_answers_through_its_own_entry(tmp_path, monkeypatch):
    """`python -m mix_engine.separation_worker` - what a source brain starts."""
    from mix_engine import separation

    monkeypatch.setenv("MIX_AGENT_SEPARATOR", "fake")
    argv, env = separation.worker_argv()
    assert argv[:1] == [sys.executable] and argv[1:] == ["-m", "mix_engine.separation_worker"]
    out = tmp_path / "out"
    out.mkdir()
    src = _song(tmp_path)
    done = subprocess.run([*argv, str(src), str(out), "2"], env=env, cwd=str(tmp_path),
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert json.loads((out / "done.json").read_text()) == {"ok": True, "sample_rate": 44100}
    assert sorted(p.name for p in out.glob("*.npy")) == ["accompaniment.npy", "vocals.npy"]
    assert (out / "stage.separating").exists()


def test_a_worker_whose_brain_has_gone_does_not_run_on(tmp_path):
    """A brain that is killed mid-split (the app's force-stop, a crash) left minutes of
    inference running with nobody to hand the stems to."""
    server = REPO / "server"
    parent = (
        "import subprocess, sys\n"
        "child = subprocess.Popen([sys.executable, '-c', "
        "'from mix_engine.separation_worker import exit_when_orphaned; import time; "
        "exit_when_orphaned(0.1); time.sleep(600)'])\n"
        "print(child.pid, flush=True)\n")
    env = {**os.environ, "PYTHONPATH": str(server)}
    done = subprocess.run([sys.executable, "-c", parent], env=env, capture_output=True,
                          text=True, timeout=60)
    pid = int(done.stdout.strip())
    deadline = time.monotonic() + 10.0
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    try:
        assert not _alive(pid), "the worker outlived its brain"
    finally:
        if _alive(pid):
            os.kill(pid, 9)


def test_the_frozen_brain_starts_its_worker_before_it_imports_the_server():
    entry = (REPO / "daw" / "sidecar" / "brain_entry.py").read_text()
    assert '"--separation-worker"' in entry
    assert entry.index('"--separation-worker"') < entry.index("from orchestrator.daw_server import")
    from mix_engine import separation
    src = Path(separation.__file__).read_text()
    assert '"--separation-worker"' in src, "the frozen brain's worker flag is not what its entry reads"


# ---------------------------------------------------------------------------------------------
# S1·R25 b - an unreadable session.json is set aside, as the edit file is
# ---------------------------------------------------------------------------------------------

def _stems_folder(folder: Path, names=("drums", "bass")) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names):
        _song(folder, f"{name}.wav", seconds=0.5 + 0.1 * i)
    return folder


def _asides(proj: Path) -> list[Path]:
    return sorted(proj.glob("session.json.unreadable-*"))


def test_an_unreadable_session_is_set_aside_and_the_last_save_is_opened(tmp_path):
    """It was reported in chat ("the file needs repairing") and left where it was: every
    co-edit, every run and every open failed on it until somebody removed it by hand - with
    the checkpoint of the last save sitting beside it."""
    from mix_engine.session import SAVED_SESSION_FILE, Session
    from orchestrator.daw_bridge import ensure_saved_checkpoint
    from orchestrator.daw_server import Brain

    proj = _stems_folder(tmp_path / "proj")
    s = Session(proj)
    s.user_edit("drums", {"gain_db": -4.0})
    assert ensure_saved_checkpoint(proj) is True              # "the last save"
    saved = json.loads((proj / SAVED_SESSION_FILE).read_text())
    whole = (proj / "session.json").read_text()
    (proj / "session.json").write_text(whole[: len(whole) // 2])   # what a full disk leaves

    brain = Brain(proj)
    assert brain.session_problem()                            # the control: it IS unreadable
    brain.open_session(daw=True)
    assert brain.session_problem() == "", "the session is still unreadable after the open"
    assert Session(proj, read_only=True)._state["stems"]["drums"]["gain_db"] == -4.0
    assert json.loads((proj / "session.json").read_text())["stems"].keys() == saved["stems"].keys()
    aside = _asides(proj)
    assert len(aside) == 1 and aside[0].read_text() == whole[: len(whole) // 2]
    notes = brain.take_open_notices()
    assert len(notes) == 1 and aside[0].name in notes[0] and "last save" in notes[0]
    assert brain.take_open_notices() == [], "the notice is said once"


def test_with_no_checkpoint_the_mix_starts_again_from_the_audio_and_says_so(tmp_path):
    from mix_engine.session import Session
    from orchestrator.daw_server import Brain

    proj = _stems_folder(tmp_path / "proj")
    (proj / "session.json").write_text("")                    # zero bytes: an interrupted write
    brain = Brain(proj)
    brain.open_session(daw=True)
    assert brain.session_problem() == ""
    assert sorted(Session(proj, read_only=True)._state["stems"]) == ["bass", "drums"]
    assert len(_asides(proj)) == 1
    notes = brain.take_open_notices()
    assert len(notes) == 1 and "from the audio" in notes[0]


def test_a_readable_session_is_left_alone(tmp_path):
    from mix_engine.session import Session
    from orchestrator.daw_server import Brain

    proj = _stems_folder(tmp_path / "proj")
    Session(proj).user_edit("drums", {"gain_db": -3.0})
    before = (proj / "session.json").read_bytes()
    brain = Brain(proj)
    brain.open_session(daw=True)
    assert _asides(proj) == [] and brain.take_open_notices() == []
    assert json.loads(before)["stems"]["drums"]["gain_db"] == -3.0
    assert Session(proj, read_only=True)._state["stems"]["drums"]["gain_db"] == -3.0


def test_a_session_that_parses_and_is_not_a_session_is_set_aside_too(tmp_path):
    from orchestrator.daw_server import Brain

    proj = _stems_folder(tmp_path / "proj")
    (proj / "session.json").write_text("[1, 2, 3]")
    brain = Brain(proj)
    brain.open_session(daw=True)
    assert len(_asides(proj)) == 1 and brain.session_problem() == ""


# ---------------------------------------------------------------------------------------------
# S1·R25 d - opening a folder writes nothing into it (James, 2026-10-06)
# ---------------------------------------------------------------------------------------------

def _listing(folder: Path) -> list[str]:
    return sorted(str(p.relative_to(folder)) for p in folder.rglob("*"))


def test_a_held_folder_is_read_and_not_written(tmp_path):
    from mix_engine.session import Session, hold_writes, writes_held

    proj = _stems_folder(tmp_path / "proj")
    before = _listing(proj)
    hold_writes(proj, True)
    try:
        assert writes_held(proj)
        s = Session(proj)
        assert sorted(s._state["stems"]) == ["bass", "drums"]     # the folder is still read
        s.user_edit("drums", {"gain_db": -2.0})
        assert _listing(proj) == before, "a held folder was written"
    finally:
        hold_writes(proj, False)
    assert not writes_held(proj)
    Session(proj)
    assert "session.json" in _listing(proj)                       # the control


def test_an_unwritten_open_leaves_the_folder_as_it_found_it(tmp_path, monkeypatch):
    monkeypatch.setenv("MIX_AGENT_DATA_DIR", str(tmp_path / "appdata"))
    from orchestrator.daw_server import Brain

    proj = _stems_folder(tmp_path / "proj")
    before = _listing(proj)
    brain = Brain(proj)
    brain.open_session(daw=True, stems_from_folder=True, unwritten=True)
    assert brain.unwritten is True
    assert sorted(brain.stems()) == ["bass", "drums"], "the hello names no stems"
    sync = brain.boot_sync()
    assert sorted(sync["chains"]["stems"] if "stems" in sync.get("chains", {}) else
                  [k for k in sync["chains"] if k != "master"]) == ["bass", "drums"]
    brain.analyze()
    brain.recover_inflight()
    assert _listing(proj) == before, f"the open wrote: {sorted(set(_listing(proj)) - set(before))}"
    # ...and the first thing that writes makes it a project, with the folder's stems
    brain.become_project()
    assert brain.unwritten is False
    after = _listing(proj)
    assert "session.json" in after and "session.saved.json" in after
    assert sorted(brain.stems()) == ["bass", "drums"]


def test_a_written_open_still_makes_the_project_at_once(tmp_path):
    """The control: without `unwritten` an open mints the session as it always did."""
    from orchestrator.daw_server import Brain

    proj = _stems_folder(tmp_path / "proj")
    brain = Brain(proj)
    brain.open_session(daw=True, stems_from_folder=True)
    assert brain.unwritten is False and "session.json" in _listing(proj)


class _Wire:
    """A real brain, started the way the DAW starts it, and one authenticated connection."""

    def __init__(self, tmp_path: Path, project: Path):
        import socket

        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("ANTHROPIC_", "MIX_AGENT_"))}
        env["MIX_AGENT_DATA_DIR"] = str(tmp_path / "data")
        self.out = tmp_path / "brain.out"
        self.err = tmp_path / "brain.err"
        with open(self.out, "w") as so, open(self.err, "w") as se:
            self.proc = subprocess.Popen(      # ON the folder itself: the first project's brain
                [sys.executable, str(REPO / "orchestrator" / "daw_server.py"), str(project)],
                cwd=tmp_path, stdin=subprocess.DEVNULL, stdout=so, stderr=se, env=env)
        deadline = time.monotonic() + 60
        text = ""
        while time.monotonic() < deadline:
            text = self.out.read_text()
            if "MIXAGENT_LISTENING port=" in text or self.proc.poll() is not None:
                break
            time.sleep(0.05)
        assert "MIXAGENT_LISTENING port=" in text, self.err.read_text()[-1500:]
        line = next(l for l in text.splitlines() if l.startswith("MIXAGENT_LISTENING"))
        fields = dict(part.split("=", 1) for part in line.split()[1:])
        self.sock = socket.create_connection(("127.0.0.1", int(fields["port"])), timeout=60)
        self.file = self.sock.makefile("rwb")
        self.n = 0
        assert self.ask("auth", {"token": fields["token"]}).get("ok") is True

    def ask(self, method: str, params: dict | None = None) -> dict:
        self.n += 1
        self.file.write((json.dumps({"id": self.n, "method": method,
                                     "params": params or {}}) + "\n").encode())
        self.file.flush()
        while True:
            line = self.file.readline()
            assert line, f"the brain hung up on {method}: {self.err.read_text()[-1500:]}"
            msg = json.loads(line)
            if msg.get("id") == self.n:
                assert "error" not in msg, f"{method}: {msg['error']}"
                return msg.get("result") or {}

    def close(self) -> None:
        try:
            self.sock.close()
        finally:
            self.proc.kill()
            self.proc.wait(timeout=10)


def test_over_the_wire_an_unwritten_hello_and_the_boot_write_nothing(tmp_path):
    """The whole door, a real brain: hello {unwritten}, the boot's reads, then the first
    request that changes the mix - which makes the folder a project by itself."""
    proj = _stems_folder(tmp_path / "proj")
    before = _listing(proj)
    wire = _Wire(tmp_path, proj)
    try:
        hello = wire.ask("hello", {"role": "daw", "project": str(proj),
                                   "stems_from_folder": True, "unwritten": True})
        assert sorted(hello["stems"]) == ["bass", "drums"] and hello.get("unwritten") is True
        wire.ask("recover_inflight")
        wire.ask("boot_sync")
        wire.ask("analyze")
        wire.ask("observations")
        wire.ask("license_state")
        assert _listing(proj) == before, \
            f"the open wrote: {sorted(set(_listing(proj)) - set(before))}"
        wire.ask("coedit", {"urls": ["/api/set_stem?stem=drums&gain_db=-5"]})
        after = _listing(proj)
        assert "session.json" in after, "a change to the mix was answered and kept nowhere"
        state = json.loads((proj / "session.json").read_text())
        assert state["stems"]["drums"]["gain_db"] == -5.0
    finally:
        wire.close()


def test_a_second_hello_that_no_longer_says_unwritten_makes_the_project(tmp_path):
    proj = _stems_folder(tmp_path / "proj")
    wire = _Wire(tmp_path, proj)
    try:
        wire.ask("hello", {"role": "daw", "project": str(proj),
                           "stems_from_folder": True, "unwritten": True})
        assert "session.json" not in _listing(proj)
        hello = wire.ask("hello", {"role": "daw", "project": str(proj),
                                   "stems_from_folder": True})
        assert not hello.get("unwritten") and "session.json" in _listing(proj)
    finally:
        wire.close()


# ---------------------------------------------------------------------------------------------
# S4·R27 - a send the DAW could not place (the track holds its sixteen plug-ins)
# ---------------------------------------------------------------------------------------------

def _routing_brain(tmp_path):
    from mix_engine.session import Session
    from orchestrator.daw_server import Brain

    proj = _stems_folder(tmp_path / "proj")
    s = Session(proj)
    rid = s.add_return("reverb")["return_id"]
    return Brain(proj), proj, rid


def test_a_send_the_daw_could_not_place_is_refused_and_leaves_the_session(tmp_path):
    """`send.set` to a track at the plug-in cap was recorded in session.json and answered ok:
    the agent was told a send exists that the graph does not hold, and every offline
    measurement afterwards counted its wet."""
    from mix_engine.session import Session
    from orchestrator import daw_bridge as B

    brain, proj, verb = _routing_brain(tmp_path)

    def daw(command, args):
        assert command == "routing.apply"
        return {"ok": True,
                "sends_not_placed": [{"source": "drums", "return": verb}],
                "coedits": [f"/api/set_send?source=drums&return={verb}&gain_db=-60"]}

    def mirror(urls):
        B.apply_coedits(proj, urls)

    out = B.perform_routing("send.set", {"source": "drums", "return": verb, "gain_db": -12.0},
                            brain.routing_ops(), daw, mirror)
    assert out["ok"] is False, out
    assert "drums" in out["error"] and verb in out["error"] and "sixteen" in out["error"]
    sends = Session(proj, read_only=True)._state["stems"]["drums"].get("sends") or {}
    assert verb not in sends, f"the session kept a send the graph does not hold: {sends}"


def test_a_send_the_daw_placed_is_answered_as_before(tmp_path):
    from mix_engine.session import Session
    from orchestrator import daw_bridge as B

    brain, proj, verb = _routing_brain(tmp_path)
    out = B.perform_routing("send.set", {"source": "drums", "return": verb, "gain_db": -12.0},
                            brain.routing_ops(), lambda command, args: {"ok": True},
                            lambda urls: None)
    assert out["ok"] is True and "sends_not_placed" not in out
    assert Session(proj, read_only=True)._state["stems"]["drums"]["sends"][verb] == -12.0


def test_another_routing_command_names_the_sends_it_left_unplaced(tmp_path):
    """A group or a return made in the same breath is still made: the reply says which sends
    are not there, and stays ok."""
    from orchestrator import daw_bridge as B

    brain, proj, verb = _routing_brain(tmp_path)

    def daw(command, args):
        return {"ok": True, "sends_not_placed": [{"source": "bass", "return": verb}],
                "coedits": []}

    out = B.perform_routing("return.add", {"kind": "delay"}, brain.routing_ops(), daw,
                            lambda urls: None)
    assert out["ok"] is True
    assert out["sends_not_placed"] == [{"source": "bass", "return": verb}]
    assert "sixteen" in out["note"]


# ---------------------------------------------------------------------------------------------
# The app's halves, where a test can read them (the console legs drive them: AuditReaderP3LegsT2.h)
# ---------------------------------------------------------------------------------------------

SOURCE = REPO / "daw" / "Source"


def _product_sources() -> list[Path]:
    """The app's own sources: not the console legs, not the console harness."""
    out = []
    for path in sorted(SOURCE.glob("*.h")) + sorted(SOURCE.glob("*.cpp")) + sorted(SOURCE.glob("*.mm")):
        name = path.name
        if "Leg" in name or name in ("Main.cpp", "CommandReplayLeg.h"):
            continue
        out.append(path)
    return out


def _code_lines(path: Path):
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        code = line.split("//", 1)[0]
        if code.strip():
            yield n, code


def test_no_small_store_is_written_with_the_call_that_ignores_its_own_write():
    """S1·R25 a: `juce::File::replaceWithText` / `replaceWithData` rename the temp whatever it
    holds. Every store goes through wholefile:: (WholeFile.h); a new bare call is refused here.
    (One leg that lives inside App.cpp writes a fixture of its own with it.)"""
    bare = []
    for path in _product_sources():
        if path.name == "WholeFile.h":
            continue
        for n, code in _code_lines(path):
            if ".replaceWithText (" in code or ".replaceWithData (" in code:
                if path.name == "App.cpp" and "f.deleteFile(); f.replaceWithText (text);" in code:
                    continue
                bare.append(f"{path.name}:{n}")
    assert bare == [], f"written with a call that never checks its write: {bare}"
    whole = (SOURCE / "WholeFile.h").read_text()
    assert whole.count("getStatus().failed()") == 2 and whole.count("overwriteTargetFileWithTemporary") >= 2


def test_the_red_switch_of_this_batch_is_gone():
    """The old path of each fix was kept behind one temporary switch for the red run."""
    left = [p.name for p in sorted(SOURCE.glob("*")) if p.is_file() and "t2red" in p.read_text(errors="ignore")]
    assert left == [], f"the temporary red switch is still in: {left}"


def _catalog_keys() -> set[str]:
    import re
    header = (SOURCE / "DawCommands.h").read_text()
    return set(re.findall(r'^\s*\{\s*\w+,\s*"([a-z_0-9.]+)"', header, re.M))


def test_every_command_named_as_looking_only_is_a_command():
    """S1·R25 d: `dawcmd::looksOnly` decides which commands leave a folder that is not a project
    untouched. A key it names that is not in the table protects nothing; a family it names must
    exist."""
    import re
    header = (SOURCE / "DawCommands.h").read_text()
    body = header[header.index("inline bool looksOnly (const juce::String& key)"):]
    body = body[: body.index("\n}\n")]
    keys = _catalog_keys()
    named = set(re.findall(r'"([a-z_]+\.[a-z_0-9.]+)"', body))
    families = set(re.findall(r'startsWith \("([a-z_]+)\."\)', body))
    assert named and families
    assert named <= keys, f"not commands: {sorted(named - keys)}"
    for family in families:
        assert any(k.startswith(family + ".") for k in keys), f"no command in the family {family}."
    # the ones that plainly change or write must not be in it
    for writer in ("transport.record", "session.save", "session.export", "track.set_gain",
                   "clip.delete", "agent.run", "library.add", "edit.new"):
        looks = writer in named or any(writer.startswith(f + ".") for f in families)
        assert not looks, f"{writer} is listed as a command that only looks"


def test_a_folder_becomes_a_project_at_each_door_that_writes():
    app = (SOURCE / "App.cpp").read_text()
    for door in ('becomeProject ("a save")', 'becomeProject ("the errand " + action)',
                 'becomeProject ("the command " + juce::String (s.key))',
                 'if (marksEdit && unwritten.load()) becomeProject ("a co-edit")',
                 'becomeProject ("the edit changed")', 'becomeProject ("a new conversation")'):
        assert door in app, f"no door: {door}"
    # the hello says so, and the brain's method list knows which requests may leave the folder alone
    assert 'o->setProperty ("unwritten", true);' in app
    from orchestrator import daw_server
    assert {"hello", "boot_sync", "analyze", "observations"} <= daw_server._LEAVES_FOLDER_ALONE
    for writer in ("coedit", "run", "chat", "chat_async", "auto_mix", "import", "separate_track",
                   "route_stem", "group_stems", "transpose", "undo", "open_edit"):
        assert writer not in daw_server._LEAVES_FOLDER_ALONE, writer


def test_file_sign_out_is_a_menu_item_wired_end_to_end():
    """S8·R8: `auth_sign_out` had no caller anywhere in the app."""
    app = (SOURCE / "App.cpp").read_text()
    assert "m.addCommandItem (&appCommands, cmdSignOut);" in app
    assert "c.add (cmdSignOut);" in app
    assert "case cmdSignOut:     if (auto* d = currentDaw()) d->signOut();" in app
    assert 'brain.request ("auth_sign_out"' in app
    assert 'if (action == "sign_out") { performSignOut(); return; }' in app
    manual = (REPO / "docs" / "17-user-manual.md").read_text()
    assert "Sign Out" in manual, "the manual does not mention File > Sign Out"


def test_the_agent_is_told_a_pad_sampler_has_a_power_switch_and_a_reader():
    """D6·R12: the grammar is what the agent reads."""
    header = (SOURCE / "DawCommands.h").read_text()
    line = next(l for l in header.splitlines() if l.lstrip().startswith('{ "sampler.set",'))
    assert 'param \\"enabled\\" (value: true|false)' in line
    line = next(l for l in header.splitlines() if l.lstrip().startswith('{ "view.sampler_editor",'))
    assert "sampler:" in line and "enabled" in line
    session = (SOURCE / "DawSession.h").read_text()
    assert "reads back through view.sampler_editor)" not in session, "the comment that named a reader that read nothing"


def test_the_manual_says_what_an_open_writes_and_what_stop_does_to_a_split():
    manual = (REPO / "docs" / "17-user-manual.md").read_text()
    assert "nothing is written into" in manual or "writes nothing into" in manual
    assert "not a project" in manual
    split = manual[manual.index("Split into stems"):]
    assert "Stop" in split[:6000]
