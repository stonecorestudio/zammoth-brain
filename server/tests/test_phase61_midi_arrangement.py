"""Phase 61: a song the engineer composed (instrument tracks, MIDI clips) is arranged in ONE
``arrangement.build``, by the brain's own planner as by the model.

The native half - what a MIDI track's part is, how it is written out - is the console leg
``--arrangement-midi``. This holds the planner, the reply's summary and the words both prompts
and the catalog use, since a model can only use ``loop_s`` and a MIDI part if it is told of them.
"""
from __future__ import annotations

import re
from pathlib import Path

from orchestrator.arrangement import (arrangement_summary, build_arrangement_plan,
                                      _arrangeable_tracks)

REPO = Path(__file__).resolve().parents[2]

# The snapshot's shape for a composed song: instrument tracks whose clips are MIDI (no `source`).
MIDI_EDIT = {
    "tempo": {"bpm": 85.0, "time_sigs": [{"numerator": 4, "denominator": 4}]},
    "tracks": [
        {"name": "Drums", "kind": "stem", "track_kind": "midi",
         "clips": [{"index": 0, "kind": "midi", "start_s": 0.0, "length_s": 11.29, "note_count": 24}]},
        {"name": "Bass", "kind": "stem", "track_kind": "midi",
         "clips": [{"index": 0, "kind": "midi", "start_s": 0.0, "length_s": 11.29, "note_count": 8}]},
        {"name": "Keys", "kind": "stem", "track_kind": "midi",
         "clips": [{"index": 0, "kind": "midi", "start_s": 0.0, "length_s": 11.29, "note_count": 12}]},
        {"name": "Lead melody", "kind": "stem", "track_kind": "midi",
         "clips": [{"index": 0, "kind": "midi", "start_s": 0.0, "length_s": 11.29, "note_count": 16}]},
        # a step clip is not a MIDI part: the build leaves such a track alone
        {"name": "Steps", "kind": "stem", "track_kind": "midi",
         "clips": [{"index": 0, "kind": "step", "start_s": 0.0, "length_s": 2.0}]},
        {"name": "bus:drums", "kind": "bus", "clips": []},
    ],
}


def test_a_composed_song_is_arrangeable():
    tracks = _arrangeable_tracks(MIDI_EDIT)
    assert [(t["name"], t["midi"]) for t in tracks] == [
        ("Drums", True), ("Bass", True), ("Keys", True), ("Lead melody", True)]


def test_an_audio_clip_makes_a_track_an_audio_source():
    edit = {"tracks": [{"name": "Gtr", "kind": "stem", "clips": [
        {"kind": "midi", "length_s": 4.0}, {"source": "gtr.wav", "length_s": 4.0}]}]}
    assert [(t["name"], t["midi"]) for t in _arrangeable_tracks(edit)] == [("Gtr", False)]


def test_the_lofi_brief_over_a_midi_only_project_builds_its_52_bars():
    plan = build_arrangement_plan("arrange this into a lofi song", MIDI_EDIT)
    assert plan["style"] == "lo-fi" and plan["total_bars"] == 52 and plan["bpm"] == 85.0
    named = {t for s in plan["sections"] for t in s["tracks"]}
    assert named == {"Drums", "Bass", "Keys", "Lead melody"}
    assert "Steps" not in named
    # roles come from the names: the intro is the music, the hook is everything
    assert plan["sections"][0]["tracks"] == ["Keys"]
    hook = next(s for s in plan["sections"] if s["name"] == "Hook")
    assert set(hook["tracks"]) == named


def test_the_summary_says_what_was_placed_and_what_was_cleared():
    plan = build_arrangement_plan("arrange this into a lofi song", MIDI_EDIT)
    said = arrangement_summary(plan, {"ok": True, "clips": 21, "midi_clips": 21,
                                      "left_out": [{"track": "Keys", "notes": 3}]})
    assert "21 MIDI clips placed" in said and "audio" not in said
    assert "Keys (3)" in said and said.endswith("One Undo reverts the whole arrangement.")
    mixed = arrangement_summary(plan, {"ok": True, "clips": 5, "midi_clips": 2})
    assert "3 audio clips and 2 MIDI clips placed" in mixed
    assert "cleared" not in arrangement_summary(plan, {"ok": True, "clips": 1, "midi_clips": 1,
                                                       "left_out": [{"track": "Keys", "notes": 3}]},
                                                preview=True)


def _catalog_line() -> str:
    text = (REPO / "daw/Source/DawCommands.h").read_text(encoding="utf-8")
    m = re.search(r'\{ "arrangement\.build", (.*?)" \},\n', text, re.S)
    assert m, "the catalog has an arrangement.build grammar line"
    return m.group(1)


def test_the_catalog_names_loop_s_midi_parts_and_left_out():
    line = _catalog_line()
    for word in ("loop_s?", "MIDI", "PART", "BEATS", "left_out", "midi_clips", "frozen"):
        assert word in line, word


def test_both_daw_prompts_tell_the_engineer_about_midi_parts():
    from orchestrator import chat, prompts
    for text in (chat.daw_prompt(), prompts.DAW_PROMPT_WHOLE):
        para = text[text.index("**A whole arrangement in one command**"):]
        para = para[:para.index("  * **", 10)]
        for word in ("audio or MIDI", "`loop_s`", "PART", "left_out", "composed"):
            assert word in para, word


def test_the_builder_reads_a_midi_part_and_answers_left_out():
    src = (REPO / "daw/Source/DawSession.h").read_text(encoding="utf-8")
    body = src[src.index("inline ArrangementBuildResult buildArrangement"):]
    body = body[:body.index("\ninline void seedSlotRegionFromClipWindow")]
    assert "readMidiPart (edit, *track)" in body
    assert "frozenRefusal (*t, \"a song arrangement\")" in body
    assert '"loop_s"' in body
    app = (REPO / "daw/Source/App.cpp").read_text(encoding="utf-8")
    handler = app[app.index("case dawcmd::arrangementBuild:"):]
    handler = handler[:handler.index("case dawcmd::sceneAdd:")]
    assert '"midi_clips"' in handler and '"left_out"' in handler


def _arrange_checks(monkeypatch, tmp_path, commands, failed=()):
    from orchestrator import lofi_showcase as ls
    turns = [{"user": st.prompt,
              "commands": list(commands) if st.key == "arrange" else [],
              "failed": list(failed) if st.key == "arrange" else []}
             for st in ls.STAGES]
    monkeypatch.setattr(ls, "feed_turns", lambda project, session="": turns)
    checks, _degraded, _turns = ls.grade_plan(tmp_path)
    return {c.name.split(": ", 1)[1]: c.ok for c in checks if ": " in c.name and "Arrangement" in c.name}


def test_the_lofi_showcase_credits_one_arrangement_build(monkeypatch, tmp_path):
    got = _arrange_checks(monkeypatch, tmp_path, ["arrangement.build", "scene.from_sections"])
    assert got["labelled sections"] and got["the song grew"]
    # a build that was refused is not credited, and four section.adds are still short of five
    refused = _arrange_checks(monkeypatch, tmp_path, ["arrangement.build", "section.add"],
                              failed=["arrangement.build"])
    assert not refused["labelled sections"] and not refused["the song grew"]
    four = _arrange_checks(monkeypatch, tmp_path, ["section.add"] * 4 + ["time.duplicate"])
    assert not four["labelled sections"] and four["the song grew"]
