"""The offline duck honours the key stem's capture offset.

`Session.set_stem_capture_start`'s docstring promised that a sidechain duck
reads the key stem's envelope at the right place even when the key and the
ducked track were captured separately (the VST captures one track at a time).
Nothing in `dsp.py` read the field: `_duck` aligned the key at frame 0, so a
bass captured 8 s after the kick was ducked 8 s early in the render and the
export while the plugin's live duck was host-aligned (2026-09-01 sweep, P1).
"""
from __future__ import annotations

import numpy as np

from mix_engine.dsp import _duck

SR = 48000
ENTRY = {"type": "duck", "threshold_db": -30.0, "depth_db": 12.0,
         "attack_ms": 5.0, "release_ms": 40.0}


def _key_burst_at(t_s: float, dur_s: float = 0.5, total_s: float = 6.0) -> np.ndarray:
    key = np.zeros((int(total_s * SR), 2))
    a, b = int(t_s * SR), int((t_s + dur_s) * SR)
    key[a:b] = 0.5
    return key


def _gain_db_at(out: np.ndarray, t_s: float) -> float:
    i = int(t_s * SR)
    return 20 * np.log10(np.abs(out[i:i + 480, 0]).mean() / 0.25)


def test_a_key_captured_later_ducks_later_by_exactly_the_offset():
    audio = np.full((6 * SR, 2), 0.25)
    key = _key_burst_at(2.0)                         # key content at its OWN t = 2.0 s
    aligned = _duck(audio, SR, key, ENTRY)           # same capture start: duck at 2 s
    assert _gain_db_at(aligned, 2.3) < -9.0
    assert _gain_db_at(aligned, 3.5) > -0.5
    # the key was captured 1.0 s AFTER the ducked stem: its t = 2.0 is our t = 3.0
    later = _duck(audio, SR, key, ENTRY, key_offset_s=1.0)
    assert _gain_db_at(later, 2.3) > -0.5, "duck engaged where the old code put it"
    assert _gain_db_at(later, 3.3) < -9.0
    # …and 1.0 s BEFORE: its t = 2.0 is our t = 1.0
    earlier = _duck(audio, SR, key, ENTRY, key_offset_s=-1.0)
    assert _gain_db_at(earlier, 1.3) < -9.0
    assert _gain_db_at(earlier, 2.6) > -0.5


def test_zero_offset_is_bit_identical_to_the_old_call():
    audio = np.full((6 * SR, 2), 0.25)
    key = _key_burst_at(2.0)
    assert np.array_equal(_duck(audio, SR, key, ENTRY), _duck(audio, SR, key, ENTRY, key_offset_s=0.0))
