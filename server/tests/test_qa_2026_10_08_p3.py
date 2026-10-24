"""User-style QA of 2026-10-08, the P3 batch - the brain's half.

D6   Asked to "export ... and call it smoke", the engineer passed an ABSOLUTE path at the top of
     the project folder - beside the stems, where a wav is read as a stem. The rule that sends a
     bare export name to `bounces/` never saw it.
D20  The line under a mix pass read "BPM 169.9 -> BPM 171.1" where a loudness line belongs:
     tempo is a readout, not something a mix move is measured by.
D22  The notice after a Stop said "press Undo to take it back" (Cmd-Z does not take a level
     back; Undo Agent Move does), and Auto Mix's reason line showed its internal profile names.

Record: docs/audits/user-qa-2026-10-08/QA-log.md.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator import automix, chat, command_door, daw_bridge

PROJ = Path("/Users/someone/Music/Zammoth/song")


# ---- D6 ------------------------------------------------------------------------------------

@pytest.mark.parametrize("given", [str(PROJ / "smoke.wav"), "~/Music/Zammoth/song/smoke.wav"])
def test_an_export_named_at_the_top_of_the_project_goes_where_exports_go(given, monkeypatch):
    monkeypatch.setenv("HOME", "/Users/someone")
    out = command_door.resolve_path_args(PROJ, "session.export", {"format": "wav", "file": given})
    assert out["file"] == str(PROJ / "bounces" / "smoke.wav")


@pytest.mark.parametrize("given", [str(PROJ / "deliver" / "smoke.wav"),          # a folder it names itself
                                   str(PROJ / "bounces" / "smoke.wav"),          # already there
                                   "/Users/someone/Desktop/smoke.wav"])          # outside the project
def test_an_export_that_names_its_own_folder_is_taken_as_written(given):
    args = {"format": "wav", "file": given}
    assert command_door.resolve_path_args(PROJ, "session.export", args) is args


def test_only_an_export_is_moved_out_of_the_project_top():
    """`library.add` of a file at the top of the project is that file: it is not an export."""
    args = {"path": str(PROJ / "vocals.wav")}
    assert command_door.resolve_path_args(PROJ, "library.add", args) is args


def test_the_redirect_is_said():
    before = {"format": "wav", "file": str(PROJ / "smoke.wav")}
    after = command_door.resolve_path_args(PROJ, "session.export", before)
    note = command_door.export_redirect_note("session.export", before, after)
    assert "bounces" in note and "smoke.wav" in note
    assert command_door.export_redirect_note("session.export", before, before) == ""
    bare = {"format": "wav", "file": "smoke.wav"}
    assert command_door.export_redirect_note(
        "session.export", bare, command_door.resolve_path_args(PROJ, "session.export", bare)) == "", \
        "a bare name was always sent there: nothing to say"


# ---- D20 -----------------------------------------------------------------------------------

def test_tempo_is_never_the_measured_line():
    before = {"lufs_i": -14.0, "tempo_bpm": 169.9, "tempo_confidence": 0.4}
    after = {"lufs_i": -13.5, "tempo_bpm": 171.1, "tempo_confidence": 0.9}
    delta = daw_bridge.turn_delta(before, after)
    assert delta["changed"] is True
    for field in ("was", "now", "text"):
        assert "BPM" not in delta[field] and "TEMPO" not in delta[field].upper(), delta[field]
    assert "LUFS -14.0" in delta["was"] and "LUFS -13.5" in delta["now"]


def test_a_tempo_reading_that_moved_alone_is_no_change():
    delta = daw_bridge.turn_delta({"lufs_i": -14.0, "tempo_bpm": 85.0}, {"lufs_i": -14.0, "tempo_bpm": 170.0})
    assert delta == {"changed": False, "was": "", "now": "", "text": ""}


# ---- D22 -----------------------------------------------------------------------------------

def test_a_stop_names_the_undo_that_takes_a_run_back():
    for sentence in (chat.STOPPED_NOTICE,):
        assert "Undo Agent Move" in sentence and "⌥⌘Z" in sentence, sentence
        assert "press Undo to" not in sentence


def test_no_sentence_of_the_brain_promises_plain_undo_for_a_run():
    """Cmd-Z takes back timeline edits; a level the agent set is put back by Undo Agent Move."""
    for module in ("chat.py", "daw_server.py"):
        text = (Path(chat.__file__).parent / module).read_text(encoding="utf-8")
        for stale in ("press Undo to", "Undo takes them back", "Undo takes it back"):
            assert stale not in text, f"{module}: {stale!r}"


def test_auto_mix_gives_its_reasons_in_words():
    for intent in automix.STEM_DYNAMICS_PROFILES:
        words = automix.dynamics_reason(intent)
        assert words and "_" not in words and words != intent, (intent, words)
    assert automix.dynamics_reason("something_new") == "something new"
