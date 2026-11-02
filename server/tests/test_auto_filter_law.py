"""The Auto Filter's law (2026-10-27): fitted to a reference filter's clean style - `auto_filter_law`.

Held here:
  * the reference's MEASURED behaviour, which the filter before this did not have (the red half): the lift
    at the cutoff by resonance, -6 dB at the cutoff at 24 dB/oct, the band-pass's and the notch's own
    widths, the envelope's 60 dB window and its knobs as the time to 90 %;
  * the offline engines run the same law, AutoFilterLaw.h is the file this module writes.

The native chain is held to the same law by the Auto Filter's own parity fixture
(`test_auto_filter_native_parity.py`, `data/auto_filter_parity.json`) and the console leg `--auto-filter`.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))

from mix_engine import dsp, reference_monitor as rm  # noqa: E402

SR = 48000


def _gain_db(e: dict, f: float) -> float:
    """A steady sine through the filter at a fixed cutoff, its level change in dB."""
    t = np.arange(SR) / SR
    x = np.stack([0.1 * np.sin(2 * math.pi * f * t)] * 2, 1)
    y = dsp._auto_filter(x, SR, {"lfo_amount_oct": 0.0, **e})
    a, b = x[SR // 2:, 0], y[SR // 2:, 0]
    return 10 * math.log10(np.mean(b ** 2) / np.mean(a ** 2))


# ------------------------------------------------ the reference's behaviour (the red half)
# |H| at the cutoff (1 kHz) of its clean style, a 12 dB low-pass, by resonance.
MEASURED_LP12_AT_FC = {0.0: -3.0, 0.3: 0.0, 0.5: 2.9, 0.7: 7.1, 0.9: 15.8}


def test_the_lift_at_the_cutoff_is_the_references():
    for r, want in MEASURED_LP12_AT_FC.items():
        got = _gain_db({"filter_type": 0, "slope_db_oct": 12, "cutoff_hz": 1000.0, "resonance": r}, 1000.0)
        assert abs(got - want) < 0.3, (r, got, want)


def test_24_db_is_minus_6_at_the_cutoff():
    got = _gain_db({"filter_type": 0, "slope_db_oct": 24, "cutoff_hz": 1000.0, "resonance": 0.0}, 1000.0)
    assert abs(got - -6.1) < 0.2, got
    # and at resonance 0.5 the pair lifts it 1.2 dB, the resonant section the 12 dB law's
    got = _gain_db({"filter_type": 0, "slope_db_oct": 24, "cutoff_hz": 1000.0, "resonance": 0.5}, 1000.0)
    assert abs(got - 1.2) < 0.3, got


def test_the_band_pass_is_the_references():
    # 0 dB at its centre; three octaves under it -18.0 dB at 0 and -20.5 at 0.5 (Q 1 and 1.33)
    for r, want in ((0.0, -18.0), (0.5, -20.5)):
        e = {"filter_type": 2, "slope_db_oct": 12, "cutoff_hz": 1000.0, "resonance": r}
        assert abs(_gain_db(e, 1000.0)) < 0.2
        assert abs(_gain_db(e, 125.0) - want) < 0.3, (r, _gain_db(e, 125.0))


def test_the_notch_is_the_references():
    # half an octave under its centre: -9.5 dB at 0 (Q 0.5), -5.5 at 0.5
    for r, want in ((0.0, -9.5), (0.5, -5.5)):
        e = {"filter_type": 3, "slope_db_oct": 12, "cutoff_hz": 1000.0, "resonance": r}
        assert abs(_gain_db(e, 707.1) - want) < 0.3, (r, _gain_db(e, 707.1))


def _cutoff_octaves(level_db_steps, e, seconds):
    """The cutoff the law asks for, in octaves over the knob, for a square wave (crest 0 dB) whose level
    steps through `level_db_steps` (each `seconds` long)."""
    n = int(seconds * SR)
    g = np.concatenate([np.full(n, 10 ** (d / 20)) for d in level_db_steps])
    sq = np.sign(np.sin(2 * math.pi * 997 * np.arange(g.size) / SR))
    x = np.stack([sq * g] * 2, 1)
    v = rm.auto_filter_values({"lfo_amount_oct": 0.0, "cutoff_hz": 100.0, **e})
    fc = rm.auto_filter_modulation(x, SR, v, follower=dsp._rising_follower)
    return np.log2(fc[:, 0] / 100.0)


def test_the_envelope_reads_60_db():
    o = _cutoff_octaves([-30.0], {"env_amount_oct": 2.0, "env_attack_ms": 1.0, "env_release_ms": 50.0}, 0.5)
    assert abs(o[-1] - 1.0) < 0.01, o[-1]           # -30 dBFS is half way up the window: 1 of 2 octaves
    o = _cutoff_octaves([-70.0], {"env_amount_oct": 2.0, "env_attack_ms": 1.0, "env_release_ms": 50.0}, 0.5)
    assert abs(o[-1]) < 1e-9, o[-1]                  # under the floor: nothing


def test_the_envelope_knobs_are_the_time_to_90_percent():
    e = {"env_amount_oct": 2.0, "env_attack_ms": 100.0, "env_release_ms": 500.0}
    o = _cutoff_octaves([-60.0, -12.0, -60.0], e, 2.0)
    lo, hi = o[2 * SR - 1], 2.0 * (48.0 / 60.0)
    rise = np.argmax(o[2 * SR:] > lo + 0.9 * (hi - lo)) / SR * 1000.0
    fall = np.argmax(o[4 * SR:] < lo + 0.1 * (hi - lo)) / SR * 1000.0
    assert abs(rise - 100.0) < 2.0 and abs(fall - 500.0) < 5.0, (rise, fall)


# ------------------------------------------------ the law, one in both offline engines
def test_the_reference_monitor_is_the_same_law():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((SR // 2, 2)) * 0.2
    for e in ({"filter_type": 0, "slope_db_oct": 24, "resonance": 0.7, "env_amount_oct": 3.0},
              {"filter_type": 2, "slope_db_oct": 12, "resonance": 0.4, "lfo_amount_oct": 2.0},
              {"filter_type": 3, "slope_db_oct": 24, "resonance": 0.9, "env_amount_oct": -2.0}):
        a = dsp._auto_filter(x, SR, e)
        b = rm._auto_filter(x, float(SR), e)
        assert np.max(np.abs(a - b)) < 1e-9, e


def test_the_header_is_the_one_this_module_writes():
    from mix_engine import auto_filter_law as afl
    assert (REPO / "native/Source/AutoFilterLaw.h").read_text() == afl.header_text()
