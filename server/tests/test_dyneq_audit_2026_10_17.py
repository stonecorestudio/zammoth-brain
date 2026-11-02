"""The dynamic EQ's audit, 2026-10-17 (the device-by-device audit) - the offline engine.

The native half is `--dyneq-audit`. (The offline-vs-native law sample for sample is
test_dyneq_law.py and `--dyneq-law` since 2026-10-27, when the law was fitted to a reference.) Here:
  * a band's threshold does not move with its frequency: a tone at the same level is cut the same
    at 200 Hz and at 6 kHz (the native's dB-domain follower read a 6 kHz tone 6.5 dB low);
  * the reference monitor's dynamic EQ is the same function.
"""
from __future__ import annotations

import math

import numpy as np

from mix_engine import dsp, reference_monitor

SR = 48000


def _bursts(seconds: float = 0.4) -> np.ndarray:
    n = int(SR * seconds)
    t = np.arange(n) / SR
    bed = 0.05 * np.sin(2 * np.pi * 300 * t)
    hiss = np.zeros(n)
    for s0 in np.arange(0.02, seconds, 0.1):
        m = (t >= s0) & (t < s0 + 0.04)
        hiss[m] += 0.3 * np.sin(2 * np.pi * 6000 * t[m])
    return np.stack([bed + hiss, bed + 0.8 * hiss]).astype(np.float32)


ENTRY = {"freq_hz": 6000.0, "q": 2.0, "threshold_db": -30.0,
         "attack_ms": 3.0, "release_ms": 80.0, "gain_db": -9.0}


def test_a_bands_threshold_does_not_move_with_its_frequency():
    t = np.arange(SR) / SR
    cuts = []
    for hz in (200.0, 6000.0):
        x = np.stack([0.1 * np.sin(2 * np.pi * hz * t)] * 2).astype(np.float32)   # -20 dBFS peak
        y = dsp._dynamic_eq(x, SR, {**ENTRY, "freq_hz": hz, "threshold_db": -24.0})
        half = slice(SR // 2, SR)
        cuts.append(20 * np.log10(np.sqrt(np.mean(y[0, half] ** 2)) / np.sqrt(np.mean(x[0, half] ** 2))))
    assert abs(cuts[0] - cuts[1]) < 0.5, cuts
    assert cuts[0] < -2.0, cuts                          # and the tone over the threshold is cut


def test_the_reference_monitors_dynamic_eq_is_the_same_function():
    x = _bursts(0.2)
    offline = dsp._dynamic_eq(x, SR, ENTRY).astype(np.float64)
    monitor = reference_monitor._dynamic_eq(x.T.astype(np.float64), SR, ENTRY["freq_hz"], ENTRY["q"],
                                            ENTRY["threshold_db"], ENTRY["attack_ms"],
                                            ENTRY["release_ms"], ENTRY["gain_db"]).T
    assert np.max(np.abs(offline - monitor)) < 1e-6
