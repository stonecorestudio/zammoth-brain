"""Gates on what batch M2 of the reader P2 fixes left open on the brain's side (2026-10-04;
docs/audits/full-2026-10-02/readers/P2-fixes.md, "Left open, on purpose").

D4·R8 / D1·R2  a stem is held where its ONE clip plays (session `clip`): a split stem, a file
               dropped later in the song. Any other arrangement is NAMED to the agent.
D1·R5          the prompts say what `brain_take_gone` means.
S1·R19         in a project the DAW holds, a wav that turns up at the top of the folder is
               not a stem; a registration, a first open and a folder the DAW built its lanes
               from still mint.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from mix_engine import dsp
from mix_engine.session import Session, normalize_clip_window, stems_by_registration
from orchestrator import daw_bridge
from orchestrator.daw_bridge import apply_coedits, import_stem, render_blind_spots


# ---- D4·R8 / D1·R2: where the stem's file plays -------------------------------------------------
def test_a_stem_is_rendered_where_its_one_clip_plays(demo_project: Path):
    whole, rate = dsp.render_stem(Session(demo_project), "stem_a")
    key_before = dsp._stem_cache_key(Session(demo_project), "stem_a", False)

    out = apply_coedits(demo_project, ["/api/set_stem?stem=stem_a&clip_at_s=2.0&clip_from_s=0.5&clip_for_s=1.0"])
    assert out["failed"] == 0, out
    session = Session(demo_project)
    assert session.stems["stem_a"]["clip"] == {"at_s": 2.0, "from_s": 0.5, "for_s": 1.0}
    placed, placed_rate = dsp.render_stem(session, "stem_a")
    assert placed_rate == rate
    assert placed.shape[0] == 3 * rate                      # 2 s of lead + the 1 s window
    assert not np.any(placed[: 2 * rate])                   # nothing plays before the clip
    np.testing.assert_array_equal(placed[2 * rate:], whole[rate // 2: rate // 2 + rate])
    assert dsp._stem_cache_key(session, "stem_a", False) != key_before   # never served the old render

    # the brain says what it holds, so the DAW sends nothing when they agree
    row = daw_bridge.strip_state(session._state)["stem_a"]
    assert (row["clip_at_s"], row["clip_from_s"], row["clip_for_s"]) == (2.0, 0.5, 1.0)
    assert daw_bridge.strip_state(session._state)["stem_b"]["clip_at_s"] == 0.0

    # the other stem is untouched, and the mix is as long as its longest part
    other, _ = dsp.render_stem(session, "stem_b")
    assert other.shape[0] == whole.shape[0]

    # the clip went back to the top of the file: the record holds nothing, the render is the file
    assert apply_coedits(demo_project, ["/api/set_stem?stem=stem_a&clip_at_s=0&clip_from_s=0&clip_for_s=0"])["failed"] == 0
    again = Session(demo_project)
    assert "clip" not in again.stems["stem_a"]
    np.testing.assert_array_equal(dsp.render_stem(again, "stem_a")[0], whole)


def test_a_clip_window_is_bounded_and_never_empty(demo_project: Path):
    assert normalize_clip_window(0, 0, 0) is None
    assert normalize_clip_window(float("nan"), "x", None) is None
    assert normalize_clip_window(1e9, 0, 0) == {"at_s": 3600.0, "from_s": 0.0, "for_s": 0.0}
    assert normalize_clip_window(-4, 2, 0) == {"at_s": 0.0, "from_s": 2.0, "for_s": 0.0}
    # a number that is present and not finite is refused at the co-edit door, like every other
    assert apply_coedits(demo_project, ["/api/set_stem?stem=stem_a&clip_at_s=nan"])["failed"] == 1
    assert "clip" not in Session(demo_project).stems["stem_a"]
    # a window that starts past the file's end is one frame of silence, not an empty array
    audio = np.ones((100, 2), dtype=np.float32)
    past = dsp._placed(audio, 10, {"at_s": 0.0, "from_s": 60.0, "for_s": 1.0})
    assert past.shape == (1, 2) and not past.any()
    assert dsp._placed(audio, 10, None) is audio             # no window: the same array


def _track(name: str, clips: list, **more) -> dict:
    return {"name": name, "track_kind": "audio", "clips": clips, **more}


def test_the_agent_is_told_which_tracks_are_measured_as_their_file():
    one = {"index": 0, "start_s": 16.0, "offset_s": 1.5, "length_s": 3.0, "source": "a.wav"}
    split_in_place = [{"start_s": 0.0, "offset_s": 0.0, "source": "a.wav"},
                      {"start_s": 4.0, "offset_s": 4.0, "source": "a.wav"}]
    moved = [{"start_s": 0.0, "offset_s": 0.0, "source": "a.wav"},
             {"start_s": 9.0, "offset_s": 4.0, "source": "a.wav"}]
    two_files = [{"start_s": 0.0, "offset_s": 0.0, "source": "a.wav"},
                 {"start_s": 4.0, "offset_s": 4.0, "source": "b.wav"}]
    edit = {"tracks": [_track("One", [one]), _track("Split", split_in_place),
                       _track("Moved", moved), _track("Two", two_files),
                       _track("Warped", [{**one, "warp": True}]),
                       _track("Muted", moved, muted=True),
                       {"name": "Keys", "track_kind": "midi", "muted": True,
                        "clips": [{"kind": "midi"}, {"kind": "midi"}]}]}
    spots = {s["what"]: s for s in render_blind_spots(edit)}
    assert spots["arranged_audio"]["tracks"] == ["Moved", "Two", "Warped"]
    assert "track.bounce" in spots["arranged_audio"]["to_measure"]

    stale = {"tracks": [_track("Vox", [], brain_take_gone="Vox_1.wav"),
                        _track("Gtr", [one], brain_take_gone="Gtr_1.wav", brain_take_playing="Gtr_2.wav"),
                        _track("Deck", [], brain_audio_stale=True, brain_audio_measured="a.wav",
                               brain_audio_playing="b.wav"),
                        _track("Fine", [one])]}
    said = {s["what"]: s for s in render_blind_spots(stale)}
    assert said["stale_audio"]["tracks"] == ["Vox: measured Vox_1.wav, plays nothing",
                                             "Gtr: measured Gtr_1.wav, plays Gtr_2.wav",
                                             "Deck: measured a.wav, plays b.wav"]
    assert "arranged_audio" not in said


# ---- D1·R5: the prompts explain the flag --------------------------------------------------------
def test_the_prompts_say_what_a_take_that_left_its_track_means():
    from orchestrator import agents, chat
    chat_source = open(chat.__file__, encoding="utf-8").read()
    for needle in ("`brain_take_gone`", "`brain_take_playing`", "`arranged_audio`", "`stale_audio`"):
        assert needle in chat_source, needle
    assert "brain_take_gone" in agents.SYSTEM_PROMPT and "brain_take_playing" in agents.SYSTEM_PROMPT


def test_the_prompts_say_what_audio_that_went_away_means():
    """`audio_lost` (a drive unplugged under an open project) and `plugin_files_lost` are
    published by the DAW's snapshot (EditSnapshot.h); a field the agent is sent and never told
    about is noise - it went on measuring a track nobody could hear."""
    from orchestrator import agents, chat
    chat_source = open(chat.__file__, encoding="utf-8").read()
    for needle in ("`audio_lost`", "`plugin_files_lost`", "`files_lost`", "`files_lost_shared_code`"):
        assert needle in chat_source, needle
    assert "audio_lost" in agents.SYSTEM_PROMPT and "plugin_files_lost" in agents.SYSTEM_PROMPT
    assert " files_lost " in agents.SYSTEM_PROMPT      # ...and WHICH plug-in: its own row says (2026-10-04)
    snapshot_source = (Path(chat.__file__).resolve().parents[1] / "daw" / "Source" / "EditSnapshot.h").read_text(encoding="utf-8")
    assert '"audio_lost"' in snapshot_source and '"plugin_files_lost"' in snapshot_source
    assert '"files_lost"' in snapshot_source and '"files_lost_shared_code"' in snapshot_source


# ---- S1·R19: which wavs are stems ---------------------------------------------------------------
@pytest.fixture()
def held_by_the_daw(demo_project: Path):
    from orchestrator.daw_server import Brain
    brain = Brain(demo_project)
    brain.open_session(daw=True)                 # a first open: the folder IS the track list
    yield brain
    stems_by_registration(demo_project, False)   # the rule is per process: leave none behind


def test_a_wav_that_turns_up_in_a_daw_project_is_not_a_stem(held_by_the_daw, demo_project: Path):
    assert sorted(Session(demo_project).stems) == ["stem_a", "stem_b"]

    shutil.copy(demo_project / "stem_a.wav", demo_project / "master export.wav")
    session = Session(demo_project)
    assert sorted(session.stems) == ["stem_a", "stem_b"]
    assert sorted(session.refresh_stems()) == ["stem_a", "stem_b"]
    assert "master export.wav" not in (session._state.get("retired_wavs") or [])   # not a track; not buried either

    # a registration names its file, and only that one is minted
    shutil.copy(demo_project / "stem_b.wav", demo_project / "second stray.wav")
    made = import_stem(demo_project, str(demo_project / "master export.wav"), "master export")
    assert made["ok"] and made["wav"] == "master export.wav", made
    assert sorted(Session(demo_project).stems) == ["master export", "stem_a", "stem_b"]

    # a file brought in from outside the project registers as it always did
    outside = demo_project.parent / "gtr take.wav"
    shutil.copy(demo_project / "stem_b.wav", outside)
    assert import_stem(demo_project, str(outside), "gtr take")["ok"]
    assert sorted(Session(demo_project).stems) == ["gtr take", "master export", "stem_a", "stem_b"]


def test_a_stray_named_after_a_stem_does_not_take_its_record(held_by_the_daw, demo_project: Path):
    """A file named after a stem whose record points at ANOTHER file is discovery's re-drop rule:
    the record follows it. In a DAW-held project only a registration may do that."""
    session = Session(demo_project)
    (demo_project / "stem_a.wav").rename(demo_project / "stem_a take.wav")
    session.claim_stem_file("stem_a", "stem_a take.wav")
    assert Session(demo_project).stems["stem_a"]["path"] == "stem_a take.wav"

    shutil.copy(demo_project / "stem_b.wav", demo_project / "stem_a.wav")        # copied in by hand
    after = Session(demo_project)
    assert after.stems["stem_a"]["path"] == "stem_a take.wav"
    assert "stem_a take.wav" not in (after._state.get("retired_wavs") or [])

    # ...and the same file through a registration is the stem's new audio, as before
    (demo_project / "stem_a.wav").unlink()
    outside = demo_project.parent / "new take.wav"
    shutil.copy(demo_project / "stem_b.wav", outside)
    made = import_stem(demo_project, str(outside), "stem_a")
    assert made["ok"], made
    assert Session(demo_project).stems["stem_a"]["path"] == made["wav"]


def test_the_next_open_decides_again(held_by_the_daw, demo_project: Path):
    from orchestrator.daw_server import Brain
    shutil.copy(demo_project / "stem_a.wav", demo_project / "copied in.wav")

    stems_by_registration(demo_project, False)   # a new brain process knows nothing of the last one
    Brain(demo_project).open_session(daw=True)   # the project has an edit: the DAW made no lane for the file
    assert sorted(Session(demo_project).stems) == ["stem_a", "stem_b"]

    stems_by_registration(demo_project, False)
    Brain(demo_project).open_session(daw=True, stems_from_folder=True)   # no edit file: the DAW built a lane from it
    assert sorted(Session(demo_project).stems) == ["copied in", "stem_a", "stem_b"]

    shutil.copy(demo_project / "stem_a.wav", demo_project / "later.wav")   # ...and from then on, by registration
    assert "later" not in Session(demo_project).stems


def test_a_folder_nobody_holds_is_read_as_before(demo_project: Path):
    """The control: the offline engine's contract, and a client of the brain that is not the DAW."""
    from orchestrator.daw_server import Brain
    Brain(demo_project).open_session()           # a hello with no role
    shutil.copy(demo_project / "stem_a.wav", demo_project / "third.wav")
    assert sorted(Session(demo_project).stems) == ["stem_a", "stem_b", "third"]
