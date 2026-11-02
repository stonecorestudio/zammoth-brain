"""The final QA of 2026-10-30, fixed.

A bass under C2 is read note by note (the agent used to transpose the user's clip and print
it, bar by bar, to learn its notes), and the notes come back in the order they are played.
"""
import numpy as np

from mix_engine import analysis

SR = 44100


def _bass(notes, sr=SR):
    """A plain bass line: (midi note, seconds) in turn, a little second harmonic, soft ends."""
    parts = []
    for midi, dur in notes:
        n = int(dur * sr)
        t = np.arange(n) / sr
        hz = 440.0 * 2 ** ((midi - 69) / 12)
        x = 0.6 * np.sin(2 * np.pi * hz * t) + 0.15 * np.sin(4 * np.pi * hz * t)
        ramp = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.015)
        parts.append(x * ramp)
    x = np.concatenate(parts)
    return np.stack([x, x], axis=1)


LINE = [(33, 2.0), (35, 1.4), (33, 1.4), (38, 1.4), (31, 1.4)]   # A1 B1 A1 D2 G1


def test_a_bass_under_the_old_floor_is_read_note_by_note():
    out = analysis.analyze_notes(_bass(LINE), SR, "bass")
    assert out.reason is None
    names = [s.note_name for s in out.sequence]
    assert names == ["A1", "B1", "A1", "D2", "G1"]
    assert out.played[0].note == "A"
    # the frame is long down here: a note's edges are still its own, within a frame's reach
    starts = [s.start_s for s in out.sequence]
    want = np.cumsum([0.0] + [d for _, d in LINE[:-1]])
    assert np.max(np.abs(np.array(starts) - want)) < 0.15


def test_a_note_under_the_floor_is_in_tune_by_its_own_figure():
    out = analysis.analyze_notes(_bass([(33, 3.0)]), SR, "bass")
    assert [n.note_name for n in out.notes] == ["A1"]
    assert abs(out.notes[0].cents_off) < 5.0


def test_a_part_above_the_floor_reads_as_it_did_and_gains_its_order():
    out = analysis.analyze_notes(_bass([(48, 1.0), (52, 1.0), (55, 1.0)]), SR, "lead")
    assert [s.note_name for s in out.sequence] == ["C3", "E3", "G3"]
    assert out.notes_found == 3


def test_the_order_is_capped_and_the_count_says_so():
    line = [(45 + (i % 5), 0.25) for i in range(analysis.NOTE_SEQUENCE_MAX + 20)]
    out = analysis.analyze_notes(_bass(line), SR, "long")
    assert len(out.sequence) == analysis.NOTE_SEQUENCE_MAX
    assert out.notes_found > analysis.NOTE_SEQUENCE_MAX
