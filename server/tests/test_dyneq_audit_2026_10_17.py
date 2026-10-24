"""The dynamic EQ's audit, 2026-10-17 (the device-by-device audit) - the offline engine.

The native half is `--dyneq-audit`. Here:
  * the offline dynamic EQ IS the DAW's law, sample for sample: the native's band-pass in float64,
    the band's linked per-sample peak followed LINEARLY from rest, the move held to gain_db (it was
    a zero-phase Gaussian FFT band and a 5 ms block RMS in dB);
  * a band's threshold does not move with its frequency: a tone at the same level is cut the same
    at 200 Hz and at 6 kHz (the native's dB-domain follower read a 6 kHz tone 6.5 dB low);
  * the reference monitor's dynamic EQ is the same function.
"""
from __future__ import annotations

import math

import numpy as np

from mix_engine import dsp, reference_monitor
from mix_engine.reference_monitor import biquad_coeffs

SR = 48000


def _native(x: np.ndarray, sr: int, e: dict) -> np.ndarray:
    """`MonitorChain::applyDynamicEq`, one band, transcribed per sample. x is (channels, frames)."""
    b, a = biquad_coeffs("bandpass", sr, e["freq_hz"], e["q"], 0.0)
    nch, n = x.shape
    z = np.zeros((nch, 2))
    a_att = math.exp(-1 / (sr * e["attack_ms"] / 1000))
    a_rel = math.exp(-1 / (sr * e["release_ms"] / 1000))
    env = 0.0
    out = x.astype(np.float64).copy()
    for i in range(n):
        band = np.empty(nch)
        for ch in range(nch):
            xv = float(x[ch, i])
            y = b[0] * xv + z[ch, 0]
            z[ch, 0] = b[1] * xv - a[1] * y + z[ch, 1]
            z[ch, 1] = b[2] * xv - a[2] * y
            band[ch] = np.float32(y)
        pk = float(np.max(np.abs(band)))
        c = a_att if pk > env else a_rel
        env = c * env + (1 - c) * pk
        lvl = 20 * math.log10(max(env, 1e-6))
        move = max(lvl - e["threshold_db"], 0.0) * (1 - 1 / e["ratio"])
        g = e["gain_db"]
        applied = min(max(math.copysign(move, g), min(0.0, g)), max(0.0, g))
        out[:, i] += band * (10 ** (applied / 20) - 1)
    return out


def _bursts(seconds: float = 0.4) -> np.ndarray:
    n = int(SR * seconds)
    t = np.arange(n) / SR
    bed = 0.05 * np.sin(2 * np.pi * 300 * t)
    hiss = np.zeros(n)
    for s0 in np.arange(0.02, seconds, 0.1):
        m = (t >= s0) & (t < s0 + 0.04)
        hiss[m] += 0.3 * np.sin(2 * np.pi * 6000 * t[m])
    return np.stack([bed + hiss, bed + 0.8 * hiss]).astype(np.float32)


ENTRY = {"freq_hz": 6000.0, "q": 2.0, "threshold_db": -30.0, "ratio": 4.0,
         "attack_ms": 3.0, "release_ms": 80.0, "gain_db": -9.0}


def test_the_offline_dynamic_eq_is_the_daws_law_sample_for_sample():
    x = _bursts()
    offline = dsp._dynamic_eq(x, SR, ENTRY).astype(np.float64)
    native = _native(x, SR, ENTRY)
    assert np.max(np.abs(offline - x)) > 0.02          # it cut something
    assert np.max(np.abs(offline - native)) < 1e-4, float(np.max(np.abs(offline - native)))


def test_a_bands_threshold_does_not_move_with_its_frequency():
    t = np.arange(SR) / SR
    cuts = []
    for hz in (200.0, 6000.0):
        x = np.stack([0.1 * np.sin(2 * np.pi * hz * t)] * 2).astype(np.float32)   # -20 dBFS peak
        y = dsp._dynamic_eq(x, SR, {**ENTRY, "freq_hz": hz, "threshold_db": -24.0, "ratio": 8.0})
        half = slice(SR // 2, SR)
        cuts.append(20 * np.log10(np.sqrt(np.mean(y[0, half] ** 2)) / np.sqrt(np.mean(x[0, half] ** 2))))
    assert abs(cuts[0] - cuts[1]) < 0.5, cuts
    assert cuts[0] < -2.0, cuts                          # and the tone over the threshold is cut


def test_the_reference_monitors_dynamic_eq_is_the_same_function():
    x = _bursts(0.2)
    offline = dsp._dynamic_eq(x, SR, ENTRY).astype(np.float64)
    monitor = reference_monitor._dynamic_eq(x.T.astype(np.float64), SR, ENTRY["freq_hz"], ENTRY["q"],
                                            ENTRY["threshold_db"], ENTRY["ratio"], ENTRY["attack_ms"],
                                            ENTRY["release_ms"], ENTRY["gain_db"]).T
    assert np.max(np.abs(offline - monitor)) < 1e-6
