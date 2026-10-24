"""The second mashup showcase's rehearsal (2026-10-12): two things the agent could not see.

Seven typed prompts drove the real app from two full songs to a mastered mashup. The agent's
bar lines for one song were a BEAT early - its drums come in one beat after the line it had
drawn - and nothing it could read said so: `analyze_activity` answers to the bar, and a bar
with three beats of drums in it is `in`. And it named the wrong home note for the other
song's bass, because `analyze_notes` lists the eight notes furthest out of tune, not the notes
the part plays: the bass sat on C# for most of the song and the list held none of them.

Both fixtures are synthesised here, and each test holds its own control: the old view of the
same audio, which must show the blind spot or the test proves nothing.
"""
from __future__ import annotations

import numpy as np
import pytest

from mix_engine import analysis

SR = 44100


def _kick(n: int) -> np.ndarray:
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * (50 + 70 * np.exp(-t / 0.02)) * t) * np.exp(-t / 0.08)


def _drums(bpm: float, first_hit_s: float, seconds: float) -> np.ndarray:
    """Four on the floor from `first_hit_s`, silence before it."""
    out = np.zeros(int(seconds * SR))
    beat = 60.0 / bpm
    hit = _kick(int(0.2 * SR))
    t = first_hit_s
    while t < seconds - 0.25:
        i = int(round(t * SR))
        out[i:i + hit.size] += hit[:out.size - i]
        t += beat
    return (0.7 * out / np.max(np.abs(out))).astype(np.float32)


def _first_entry(result, stem_id="drums"):
    stem = next(s for s in result.stems if s.stem_id == stem_id)
    return next(span for span in stem.spans if span.state != "out")


def test_a_part_that_enters_a_beat_after_the_bar_line_says_so():
    """The rehearsal's case: bar 1 drawn a beat early, so the drums' first hit is on beat 2."""
    bpm, beat = 86.0, 60.0 / 86.0
    bar = 4 * beat
    true_bar1 = 0.880
    first_hit = true_bar1 + 16 * bar                          # the drums come in on bar 17
    audio = _drums(bpm, first_hit, first_hit + 12 * bar)

    early = analysis.analyze_activity({"drums": audio}, SR, bpm=bpm, bar1_s=true_bar1 - beat)
    entry = _first_entry(early)
    # the control: to the bar, the wrong grid reads exactly like the right one
    assert (entry.state, entry.from_bar) == ("in", 17)
    assert entry.enters_beat == pytest.approx(2.0, abs=0.05), entry
    assert entry.enters_s == pytest.approx(first_hit, abs=0.012), entry

    right = analysis.analyze_activity({"drums": audio}, SR, bpm=bpm, bar1_s=true_bar1)
    entry = _first_entry(right)
    assert (entry.state, entry.from_bar) == ("in", 17)
    assert entry.enters_beat == pytest.approx(1.0, abs=0.05), entry


def test_an_entry_is_said_once_per_entry_and_never_for_silence():
    bpm, bar = 120.0, 2.0
    audio = _drums(bpm, 8 * bar, 16 * bar)
    audio[int(12 * bar * SR):int(14 * bar * SR)] = 0.0        # a two-bar break, then back in
    result = analysis.analyze_activity({"drums": audio}, SR, bpm=bpm, bar1_s=0.0)
    spans = next(s for s in result.stems if s.stem_id == "drums").spans
    assert [s.state for s in spans] == ["out", "in", "out", "in"], spans
    assert [s.enters_beat is not None for s in spans] == [False, True, False, True]
    assert spans[3].enters_s == pytest.approx(14 * bar, abs=0.012)
    assert spans[3].enters_beat == pytest.approx(1.0, abs=0.05)


def _note(freq_hz: float, seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    tone = sum(np.sin(2 * np.pi * freq_hz * k * t) / k for k in (1, 2, 3, 4))
    fade = np.minimum(1.0, np.minimum(t, seconds - t) / 0.02)
    return tone * fade


def _hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def _bass_line() -> np.ndarray:
    """Sits on C#2 in tune; visits G#2 and E2, each a little out of tune - so the notes
    furthest out of tune are exactly the notes it plays least."""
    parts = []
    for _ in range(4):
        parts += [_note(_hz(37), 2.4), np.zeros(int(0.1 * SR)),              # C#2, in tune
                  _note(_hz(44 + 0.30), 0.5), np.zeros(int(0.1 * SR)),       # G#2, 30 cents sharp
                  _note(_hz(37), 2.0), np.zeros(int(0.1 * SR)),
                  _note(_hz(40 - 0.25), 0.5), np.zeros(int(0.1 * SR))]       # E2, 25 cents flat
    line = np.concatenate(parts)
    return (0.5 * line / np.max(np.abs(line))).astype(np.float32)


def test_the_notes_a_part_plays_are_listed_by_how_long_it_holds_them():
    audio = _bass_line()
    read = analysis.analyze_notes(audio, SR, "bass")
    assert read.notes_found >= 12, read.reason

    # the control: the list the agent had - the worst-tuned notes - holds no C# at all
    worst = {note.note_name[:-1] for note in read.notes}
    assert "C#" not in worst and worst <= {"G#", "E"}, worst

    played = [(row.note, row.share) for row in read.played]
    assert played[0][0] == "C#" and played[0][1] > 0.6, played
    assert {name for name, _ in played} == {"C#", "G#", "E"}, played
    assert sum(share for _, share in played) == pytest.approx(1.0, abs=0.02)
    assert read.played[0].seconds == pytest.approx(4 * 4.4, rel=0.15)


def test_a_part_with_no_notes_lists_none():
    rng = np.random.default_rng(3)
    noise = (0.2 * rng.standard_normal(SR * 4)).astype(np.float32)
    read = analysis.analyze_notes(noise, SR, "noise")
    assert read.played == [] and read.reason


def test_the_agent_is_told_what_the_two_fields_are():
    """A field nobody explains is a field nobody reads: the chat's tool text and the prompt
    name both, and the planner's copy of the tools carries the same words."""
    from orchestrator import agents, chat

    activity = next(d for d in chat._READ_TOOL_DEFS if d["name"] == "analyze_activity")["description"]
    assert "enters_beat" in activity and "bar1_s" in activity
    notes = next(d for d in chat._READ_TOOL_DEFS if d["name"] == "analyze_notes")["description"]
    assert "played" in notes and "home note" in notes
    for name in ("analyze_activity", "analyze_notes"):
        planner = next(d for d in agents.PLANNER_TOOL_DEFS if d["name"] == name)["description"]
        chat_text = next(d for d in chat._READ_TOOL_DEFS if d["name"] == name)["description"]
        assert ("enters_beat" in planner) == ("enters_beat" in chat_text)
        assert ("`played`" in planner) == ("`played`" in chat_text)
    assert "enters_beat" in chat.DAW_SYSTEM_PROMPT and "`played`" in chat.DAW_SYSTEM_PROMPT
