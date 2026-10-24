""""Don't Save" gives the mix back too (reader S1·R3, 2026-10-03).

session.json is written at every mix move; the edit file at ⌘S; and at the next open the brain's
state is laid over the edit. So a declined save threw away the clips, automation, MIDI and tempo
and KEPT every fader, device, send, group, mute and exclusion — and a file dropped on a new
track stayed a stem the brain rendered and the DAW never played.

The DAW now copies session.json to ``session.saved.json`` at each save that reaches the edit
file (with the user's moves it had not yet handed over), and leaves ``session.discard`` when the
user declines. These tests are the brain's half — what it does with those files, on the real
`Session` and the real socket. The DAW's half is the console leg ``--dont-save``.
"""
from __future__ import annotations

import asyncio
import json
import re
import shutil
from pathlib import Path

import pytest

from mix_engine import session as session_mod
from mix_engine.session import (DISCARD_MARKER_FILE, FROZEN_LIVE, OWED_COEDITS_FILE,
                                SAVED_COEDITS_FILE, SAVED_SESSION_FILE, SESSION_FILE, Session)
from orchestrator import daw_bridge
from orchestrator import daw_server as server

ROOT = Path(__file__).resolve().parents[2]


def _state(project: Path) -> dict:
    return json.loads((project / SESSION_FILE).read_text())


def _daw_saves(project: Path, queued: list[str] | None = None) -> None:
    """What `dawsession::markSessionSaved` does: a byte copy, plus the moves still queued."""
    (project / DISCARD_MARKER_FILE).unlink(missing_ok=True)
    shutil.copyfile(project / SESSION_FILE, project / SAVED_SESSION_FILE)
    (project / SAVED_COEDITS_FILE).write_text(json.dumps(queued or []))


def _daw_declines(project: Path) -> None:
    """…and `dawsession::declineUnsavedSession`."""
    assert (project / SAVED_SESSION_FILE).exists()
    (project / DISCARD_MARKER_FILE).write_text("declined\n")


def _the_declined_session(project: Path) -> None:
    """The reader's four cases, through the doors the DAW uses: a fader, a mute, a deleted track
    (excluded), and a file dropped on a new track (copied into the folder and registered)."""
    res = daw_bridge.apply_coedits(project, [
        "/api/set_stem?stem=stem_a&gain_db=-9.00",
        "/api/set_stem?stem=stem_a&mute=1",
        "/api/set_stem?stem=stem_b&excluded=1"])
    assert res["failed"] == 0, res
    daw_bridge.import_stem(project, str(project / "stem_a.wav"), "loop")
    state = _state(project)
    assert state["stems"]["stem_a"]["gain_db"] == -9.0
    assert state["stems"]["stem_b"]["excluded"] is True
    assert "loop" in state["stems"] and (project / "loop.wav").exists()


def test_the_three_file_names_are_the_daws_too():
    header = (ROOT / "daw" / "Source" / "DawSession.h").read_text(encoding="utf-8")
    for const, value in (("kSavedSessionFile", SAVED_SESSION_FILE),
                         ("kSavedCoeditsFile", SAVED_COEDITS_FILE),
                         ("kDiscardMarkerFile", DISCARD_MARKER_FILE),
                         ("kOwedCoeditsFile", OWED_COEDITS_FILE)):      # batch T: the moves a teardown could not hand over
        found = re.search(const + r'\s*=\s*"([^"]+)"', header)
        assert found and found.group(1) == value, const


def test_the_frozen_facts_live_value_is_the_daws_too():
    """Batch T (reader S2·R12): the DAW reports WHICH frozen track holds a stem, and `:live` for
    none - a query value cannot be empty. Both ends spell it the same way, and both ends know
    the field is a report of the engine's facts and not a move."""
    app = (ROOT / "daw" / "Source" / "App.cpp").read_text(encoding="utf-8")
    assert f'juce::String ("{FROZEN_LIVE}")' in app
    fact_names = app[app.index("static bool coeditIsFactReport"):][:900]
    assert 'name != "frozen"' in fact_names
    assert "frozen" in session_mod.ENGINE_FACT_FIELDS
    assert session_mod.SESSION_FILE == "session.json"


def test_a_declined_session_goes_back_to_its_last_save(demo_project):
    daw_bridge.apply_coedits(demo_project, ["/api/set_stem?stem=stem_a&gain_db=-3.00"])
    _daw_saves(demo_project)
    saved = _state(demo_project)
    _the_declined_session(demo_project)
    _daw_declines(demo_project)

    done = daw_bridge.honour_discard(demo_project)

    assert done == {"restored": True, "retired": ["loop.wav"], "owed": 0, "refused": []}
    assert not (demo_project / DISCARD_MARKER_FILE).exists()
    session = Session(demo_project)                      # …and discovery runs over the result
    assert sorted(session.stems) == ["stem_a", "stem_b"]             # no stem the edit never plays
    assert session.stems["stem_a"]["gain_db"] == -3.0                # the saved fader
    assert not session.stems["stem_a"].get("mute")
    assert not session.stems["stem_b"].get("excluded")               # the deleted track is back whole
    assert (demo_project / "loop.wav").exists()                      # the audio is the user's: kept
    after = _state(demo_project)
    assert after["retired_wavs"] == ["loop.wav"]
    assert {k: v for k, v in after["stems"].items()} == saved["stems"]
    assert json.loads((demo_project / SAVED_SESSION_FILE).read_text()) == saved   # the checkpoint is not spent


def test_without_the_marker_nothing_is_put_back(demo_project):
    """The control: a checkpoint alone changes nothing — every ordinary open has one."""
    Session(demo_project)
    _daw_saves(demo_project)
    _the_declined_session(demo_project)
    before = _state(demo_project)
    assert daw_bridge.honour_discard(demo_project) is None
    assert _state(demo_project) == before


def test_the_moves_queued_at_the_save_are_applied_over_the_checkpoint(demo_project):
    """⌘S half a second after a fader move: the edit file holds the move, session.json had not
    heard of it. Declining LATER work must not take that saved move back."""
    Session(demo_project)
    _daw_saves(demo_project, queued=["/api/set_stem?stem=stem_b&gain_db=-4.00"])
    daw_bridge.apply_coedits(demo_project, ["/api/set_stem?stem=stem_b&gain_db=-4.00",   # the flush
                                            "/api/set_stem?stem=stem_b&gain_db=-11.00"])  # later, unsaved
    _daw_declines(demo_project)
    done = daw_bridge.honour_discard(demo_project)
    assert done["restored"] and done["owed"] == 1 and done["refused"] == []
    assert _state(demo_project)["stems"]["stem_b"]["gain_db"] == -4.0


def test_a_marker_with_no_checkpoint_keeps_the_session(demo_project):
    Session(demo_project)
    daw_bridge.apply_coedits(demo_project, ["/api/set_stem?stem=stem_a&gain_db=-5.00"])
    before = _state(demo_project)
    (demo_project / DISCARD_MARKER_FILE).write_text("declined\n")
    done = daw_bridge.honour_discard(demo_project)
    assert done["restored"] is False
    assert _state(demo_project) == before
    assert not (demo_project / DISCARD_MARKER_FILE).exists()         # asked once, answered once
    (demo_project / SAVED_SESSION_FILE).write_text("{ not json")
    (demo_project / DISCARD_MARKER_FILE).write_text("declined\n")
    assert daw_bridge.honour_discard(demo_project)["restored"] is False
    assert _state(demo_project) == before


def test_an_open_that_finds_no_checkpoint_makes_one_and_never_replaces_it(demo_project):
    assert daw_bridge.ensure_saved_checkpoint(demo_project) is False     # no session yet
    Session(demo_project)
    assert daw_bridge.ensure_saved_checkpoint(demo_project) is True
    first = (demo_project / SAVED_SESSION_FILE).read_bytes()
    assert first == (demo_project / SESSION_FILE).read_bytes()
    daw_bridge.apply_coedits(demo_project, ["/api/set_stem?stem=stem_a&gain_db=-7.00"])
    assert daw_bridge.ensure_saved_checkpoint(demo_project) is False     # a save's job, not an open's
    assert (demo_project / SAVED_SESSION_FILE).read_bytes() == first


def test_boot_sync_honours_a_late_marker_and_never_makes_a_checkpoint(demo_project):
    """boot_sync is also the mid-session re-read: a checkpoint made THERE would hold the very
    moves a later "Don't Save" is meant to decline. The first checkpoint is the open's."""
    Session(demo_project)
    brain = server.Brain(demo_project)
    daw_bridge.apply_coedits(demo_project, ["/api/set_stem?stem=stem_a&gain_db=-7.00"])   # unsaved
    brain.boot_sync()
    assert not (demo_project / SAVED_SESSION_FILE).exists()
    _daw_saves(demo_project)
    _the_declined_session(demo_project)
    brain._run_marks.append(3)
    brain._run_gestures.append("g")
    brain._run_daw_ops.append(1)
    _daw_declines(demo_project)
    chains = brain.boot_sync()
    assert "loop" not in chains["chains"] and "stem_b" in chains["chains"]
    assert brain._run_marks == [] and brain._run_gestures == [] and brain._run_daw_ops == []


def test_the_open_of_a_project_the_brain_has_never_seen_checkpoints_its_untouched_state(demo_project):
    """A brand-new project has no session.json. Its last saved state is "the stems in the
    folder, untouched", and that is what a "Don't Save" before the first ⌘S goes back to."""
    assert not (demo_project / SESSION_FILE).exists()
    brain = server.Brain(demo_project)
    brain.open_session()
    assert (demo_project / SAVED_SESSION_FILE).exists()
    pristine = json.loads((demo_project / SAVED_SESSION_FILE).read_text())
    assert sorted(pristine["stems"]) == ["stem_a", "stem_b"]
    _the_declined_session(demo_project)                   # an hour of mixing, never saved
    _daw_declines(demo_project)
    brain.open_session()                                  # the next open
    after = Session(demo_project)
    assert sorted(after.stems) == ["stem_a", "stem_b"]
    assert after.stems["stem_a"]["gain_db"] == 0.0 and not after.stems["stem_b"].get("excluded")
    first = (demo_project / SAVED_SESSION_FILE).read_bytes()
    brain.open_session()                                  # …and an open never replaces a checkpoint
    assert (demo_project / SAVED_SESSION_FILE).read_bytes() == first


def test_open_edit_honours_a_discard_answered_for_that_edit(demo_project, tmp_path):
    other = tmp_path / "edit2"
    shutil.copytree(demo_project, other)
    Session(other)
    _daw_saves(other)
    _the_declined_session(other)
    _daw_declines(other)
    brain = server.Brain(demo_project)
    out = brain.open_edit(str(other))
    assert sorted(out["stems"]) == ["stem_a", "stem_b"]
    assert not (other / DISCARD_MARKER_FILE).exists()


def test_hello_puts_the_checkpoint_back_before_it_reads_the_stems(demo_project, monkeypatch, tmp_path):
    """The real door: the marker is honoured on the connection's first exchange, so no later
    request of that connection can read (or write over) the declined session."""
    monkeypatch.setenv("MIX_AGENT_DATA_DIR", str(tmp_path / "data"))
    Session(demo_project)
    _daw_saves(demo_project)
    _the_declined_session(demo_project)
    _daw_declines(demo_project)

    async def scenario():
        hub = server.Hub(demo_project)
        srv = await asyncio.start_server(lambda r, w: server._serve_client(r, w, hub), "127.0.0.1", 0)
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", srv.sockets[0].getsockname()[1])

            async def ask(obj):
                writer.write((json.dumps(obj) + "\n").encode())
                await writer.drain()
                while True:
                    msg = json.loads(await asyncio.wait_for(reader.readline(), 20))
                    if msg.get("id") == obj["id"]:
                        return msg

            await ask({"id": 0, "method": "auth", "params": {"token": hub.token}})
            hello = await ask({"id": 1, "method": "hello",
                               "params": {"project": str(demo_project), "role": "daw"}})
            writer.close()
            return hello["result"]
        finally:
            srv.close()
            await srv.wait_closed()

    result = asyncio.run(scenario())
    state = _state(demo_project)
    assert "loop" not in state["stems"] and state["retired_wavs"] == ["loop.wav"]
    assert not state["stems"]["stem_b"].get("excluded")
    assert not (demo_project / DISCARD_MARKER_FILE).exists()
    assert sorted(result["stems"]) == ["stem_a", "stem_b"]
    assert (demo_project / SAVED_SESSION_FILE).exists()


# ---- pre-merge review 2026-10-04: whose wav is it? -------------------------------------------
# "Don't Save" retires the wavs the declined session copied in. It used to retire every wav the
# checkpoint did not claim - including a file the user had put in the folder themselves.

def _finder_copies_in(project: Path, name: str, source: str = "stem_a.wav") -> None:
    """A file arriving in the folder outside the app: no import, no brain request."""
    shutil.copyfile(project / source, project / name)


def test_a_wav_the_user_put_in_the_folder_survives_dont_save(demo_project):
    brain = server.Brain(demo_project)
    brain.open_session()                                  # session 1: the first checkpoint
    assert sorted(Session(demo_project).stems) == ["stem_a", "stem_b"]
    _finder_copies_in(demo_project, "vox.wav")            # …between sessions, in Finder
    brain.open_session()                                  # session 2 opens on it
    assert json.loads((demo_project / SAVED_SESSION_FILE).read_text())["found_wavs"] == ["vox.wav"]
    assert "vox" in Session(demo_project).stems           # discovered, as it always was
    _the_declined_session(demo_project)                   # …and the session copies one in itself
    _daw_declines(demo_project)

    done = daw_bridge.honour_discard(demo_project)

    assert done["restored"] and done["retired"] == ["loop.wav"]      # the session's copy, only
    after = Session(demo_project)
    assert sorted(after.stems) == ["stem_a", "stem_b", "vox"]
    assert "found_wavs" not in _state(demo_project)       # the checkpoint's note, not the session's
    brain.open_session()                                  # the next open keeps the note…
    assert json.loads((demo_project / SAVED_SESSION_FILE).read_text())["found_wavs"] == ["vox.wav"]
    _daw_saves(demo_project)                              # …until a save claims the file
    brain.open_session()
    assert "found_wavs" not in json.loads((demo_project / SAVED_SESSION_FILE).read_text())


def test_a_reconnect_is_not_an_open(demo_project):
    """A hello also follows every reconnect. A bounce whose registration is still queued is on
    disk and in no session.json then; it arrived AFTER the DAW took the project (its lock), so
    it is the session's, and a declined session takes it along."""
    import os
    import time

    app = (ROOT / "daw" / "Source" / "App.cpp").read_text(encoding="utf-8")
    assert f'getChildFile ("{daw_bridge.DAW_LOCK_FILE}")' in app
    brain = server.Brain(demo_project)
    brain.open_session()
    lock = demo_project / daw_bridge.DAW_LOCK_FILE
    lock.write_text("4242")
    opened = time.time() - 60.0                           # the DAW opened the project a minute ago
    os.utime(lock, (opened, opened))
    _finder_copies_in(demo_project, "bounce.wav")         # …written since, not yet registered
    brain.open_session()                                  # the reconnect's hello
    assert "found_wavs" not in json.loads((demo_project / SAVED_SESSION_FILE).read_text())
    Session(demo_project)                                 # the queued registration lands
    _daw_declines(demo_project)
    assert daw_bridge.honour_discard(demo_project)["retired"] == ["bounce.wav"]
    # the control: the same file, there BEFORE the DAW opened the project, is the folder's
    later = time.time() + 60.0
    os.utime(lock, (later, later))
    _finder_copies_in(demo_project, "vox.wav")
    brain.open_session()
    assert json.loads((demo_project / SAVED_SESSION_FILE).read_text())["found_wavs"] == ["vox.wav"]


def test_a_session_that_ended_without_an_answer_is_not_mistaken_for_the_folder(demo_project):
    """The crash case: the brain's hello comes BEFORE "Discard" is answered. What the crashed
    session copied in is in its session.json, so the open does not list it as found."""
    brain = server.Brain(demo_project)
    brain.open_session()
    _the_declined_session(demo_project)                   # …and the app dies: no marker
    brain.open_session()                                  # the relaunch's hello, under the prompt
    assert "found_wavs" not in json.loads((demo_project / SAVED_SESSION_FILE).read_text())
    _daw_declines(demo_project)                           # "Discard"
    assert daw_bridge.honour_discard(demo_project)["retired"] == ["loop.wav"]
    assert sorted(Session(demo_project).stems) == ["stem_a", "stem_b"]


def test_a_wav_that_arrives_after_the_session_was_declined_is_not_its_work(demo_project):
    import os
    import time

    brain = server.Brain(demo_project)
    brain.open_session()
    _the_declined_session(demo_project)
    _daw_declines(demo_project)
    marker = demo_project / DISCARD_MARKER_FILE
    past = time.time() - 30.0                             # the decline, half a minute ago
    os.utime(marker, (past, past))
    _finder_copies_in(demo_project, "late.wav")           # …and then a copy, before the reopen
    done = daw_bridge.honour_discard(demo_project)
    assert "late.wav" not in done["retired"]
    assert "late" in Session(demo_project).stems


def test_importing_a_retired_file_registers_it(demo_project):
    """The other half: a retired `<stem>.wav` dragged back in answered `ok` and registered
    nothing - the same-bytes shortcut kept the file, and discovery skips a retired one."""
    Session(demo_project)
    _daw_saves(demo_project)
    _the_declined_session(demo_project)
    _daw_declines(demo_project)
    assert daw_bridge.honour_discard(demo_project)["retired"] == ["loop.wav"]
    assert "loop" not in Session(demo_project).stems

    res = daw_bridge.import_stem(demo_project, str(demo_project / "loop.wav"), "loop")

    assert res["ok"] and res["stem_id"] == "loop" and res["wav"] == "loop.wav", res
    session = Session(demo_project)
    assert session.stems["loop"]["path"] == "loop.wav" and not session.stems["loop"].get("excluded")
    assert "loop.wav" not in (_state(demo_project).get("retired_wavs") or [])
    assert sorted(p.name for p in demo_project.glob("*.wav")) == ["loop.wav", "stem_a.wav", "stem_b.wav"]


def test_importing_the_audio_another_stem_holds_under_its_old_name_registers(demo_project):
    """Same shortcut, same silence: `stem_a` renamed to `lead` keeps stem_a.wav, and the file
    dropped again as a new `stem_a` was "the audio this stem already plays"."""
    Session(demo_project).rename_stem("stem_a", "lead")
    res = daw_bridge.import_stem(demo_project, str(demo_project / "stem_a.wav"), "stem_a")
    assert res["ok"], res
    session = Session(demo_project)
    assert session.stems["lead"]["path"] == "stem_a.wav"
    assert session.stems["stem_a"]["path"] == res["wav"] != "stem_a.wav"
