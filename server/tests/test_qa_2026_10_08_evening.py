"""User-style QA, 2026-10-08 evening (docs/audits/user-qa-2026-10-08/QA-log.md, D37-D57).

The brain's half of that batch is one finding: D51.
"""
from __future__ import annotations

import numpy as np

from mix_engine import analysis

SR = 48000


def _tone(f0: float, seconds: float, level: float = 0.3) -> np.ndarray:
    n = int(seconds * SR)
    t = np.arange(n) / SR
    fade = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.01)
    return fade * sum(level / k * np.sin(2.0 * np.pi * k * f0 * t) for k in range(1, 7))


def _low_bass_with_a_blip() -> np.ndarray:
    """A bass loop on A1 (55 Hz) and C1 (32.7 Hz) - the QA's - with one short note above the
    tracking floor between them: 50 ms at 98 Hz, too short to be a note, long enough for a few
    frames to track."""
    parts = [_tone(55.0, 0.6), _tone(32.7, 0.6), _tone(98.0, 0.05), _tone(55.0, 0.6), _tone(32.7, 0.6)]
    return np.concatenate(parts)[:, np.newaxis]


def test_a_low_bass_with_a_few_tracked_frames_is_told_the_floor():
    """D51. Tune Notes on a bass loop at 55 Hz and 32.7 Hz answered "no note held long enough to
    measure - the source may be polyphonic, percussive, or heavily slurred". The sentence
    written for exactly that part ("this part plays below the 65 Hz tracking floor ... nothing
    here says the source is polyphonic") was reached only when NOTHING at all tracked - and a
    real bass always has a few frames that do."""
    audio = _low_bass_with_a_blip()
    f0, _, _ = analysis._f0_track(audio.mean(axis=1), SR)
    assert np.any(f0 > 0), "the fixture must have frames that track, or it is the older case"
    result = analysis.analyze_notes(audio, SR, "bass")
    assert result.notes == [] and result.reason
    assert "65 Hz" in result.reason, result.reason
    assert "polyphonic, percussive" not in result.reason, result.reason


def test_a_part_above_the_floor_with_no_held_note_is_still_told_so():
    """Control: short notes above the floor (40 ms each) hold nothing long enough - and are NOT
    told they play below the floor."""
    audio = np.concatenate([_tone(f, 0.04) for f in (220.0, 330.0, 262.0, 392.0) * 6])[:, np.newaxis]
    result = analysis.analyze_notes(audio, SR, "lead")
    if result.notes:          # a detector that holds one of them: nothing to say here
        return
    assert "65 Hz" not in (result.reason or ""), result.reason
