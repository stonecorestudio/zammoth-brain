"""OTT's audit, 2026-10-17 (the device-by-device audit) - the offline engine.

The native half is `--ott-audit`. Here:
  * the offline OTT IS the DAW's law: the real 3-way LR4 split (the native split by the complement
    `mid = dry - low - high`, so levelling one band moved the tone in another, and offline split by
    a tapered FFT partition). Since 2026-10-28 the law is `ott_law` (a reference OTT, measured):
    `dsp._ott` is `ott_law.process`, and the native half is held to it by the `--ott-law` leg on
    `data/ott_law_parity.json` (test_ott_law.py) - the Python transcription of the old native law
    that stood here is gone with that law;
  * the reference monitor's OTT is the same function (it was a third copy of the old law);
  * a gain stored past the table's +-12 dB plays at 12 in both engines.
"""
from __future__ import annotations


import numpy as np

from mix_engine import dsp, reference_monitor

SR = 48000


def _hits(seconds: float = 0.25) -> np.ndarray:
    """A kick (60 Hz, decaying) every 120 ms over quiet hats - a band to press down, and a
    quiet band to lift."""
    n = int(SR * seconds)
    t = np.arange(n) / SR
    kick = np.zeros(n)
    for start in np.arange(0.0, seconds, 0.12):
        k = t >= start
        kick[k] += 0.7 * np.sin(2 * np.pi * 60.0 * (t[k] - start)) * np.exp(-(t[k] - start) / 0.05)
    rng = np.random.default_rng(7)
    hats = np.diff(rng.standard_normal(n + 1)) * 0.004
    return np.stack([kick + hats, kick + np.roll(hats, 3)]).astype(np.float32)


ENTRY = {"type": "ott", "depth": 0.6, "time": 1.3, "in_gain_db": 2.0, "out_gain_db": -1.0,
         "low": {"up": 0.7, "down": 1.0, "gain_db": 1.5}, "mid": {},
         "high": {"up": 1.0, "down": 0.4}, "crossover_lo_hz": 120.0, "crossover_hi_hz": 3000.0}


def test_the_offline_ott_is_the_daws_law_sample_for_sample():
    from mix_engine import ott_law
    x = _hits()
    offline = dsp._ott(x, SR, ENTRY).astype(np.float64)
    law = ott_law.process(x.astype(np.float64), SR, ENTRY)
    assert np.max(np.abs(law)) > 0.1
    assert np.max(np.abs(offline - law)) < 1e-5, float(np.max(np.abs(offline - law)))


def test_the_reference_monitors_ott_is_the_same_function():
    x = _hits(0.1)
    offline = dsp._ott(x, SR, ENTRY).astype(np.float64)
    monitor = reference_monitor._ott(x.T.astype(np.float64), SR, ENTRY).T
    assert np.max(np.abs(offline - monitor)) < 1e-6


def test_a_gain_past_the_table_plays_at_its_end():
    x = _hits(0.1)
    flat = {"up": 0.0, "down": 0.0}
    at_12 = dsp._ott(x, SR, {"depth": 0.0, "in_gain_db": 12.0, "low": flat, "mid": flat, "high": flat})
    at_30 = dsp._ott(x, SR, {"depth": 0.0, "in_gain_db": 30.0, "low": flat, "mid": flat, "high": flat})
    assert np.max(np.abs(at_30 - at_12)) < 1e-6
