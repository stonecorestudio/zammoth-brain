"""The gate's audit, 2026-10-17 (the device-by-device audit) - the offline engine.

`dsp._gate` is `MonitorChain::applyGate` (v3) - the one law, `dsp.gate_gain_db`, vectorised -
held here to a literal per-sample loop of the native code: a peak-follower detector (the raw |x|
the native read fell to nothing at every zero crossing), a Schmitt trigger, the target floored at
range, HOLD as the running maximum of the target, the one-pole envelope, from REST (shut, -80 dB).
The native half is `--gate-audit`; the export oracle's `nonlinear` profile carries a gate.
"""
from __future__ import annotations

import math

import numpy as np

from mix_engine import dsp

SR = 48000


def _native_loop(peak: np.ndarray, p: dict) -> np.ndarray:
    a_att = math.exp(-1.0 / (SR * p["attack_ms"] / 1000.0))
    a_rel = math.exp(-1.0 / (SR * p["release_ms"] / 1000.0))
    a_det = math.exp(-1.0 / (SR * 15.0 / 1000.0))
    hold = int(SR * p["hold_ms"] / 1000.0)
    det, is_open, env, hist = 0.0, False, -80.0, []
    out = np.empty(peak.shape[0])
    for i, x in enumerate(peak):
        det = max(x, det * a_det)
        lv = 20.0 * math.log10(max(det, 1e-12))
        thr, hyst = p["threshold_db"], p["hysteresis_db"]
        is_open = lv >= thr - hyst if is_open else lv >= thr
        eff = thr - hyst if is_open else thr
        target = min(max((lv - eff) * (p["ratio"] - 1.0), p["range_db"]), 0.0)
        hist.append(target)
        target = max(hist[-(hold + 1):])
        coef = a_att if target > env else a_rel
        env = coef * env + (1.0 - coef) * target
        out[i] = env
    return out


def _drums(seconds: float = 0.6) -> np.ndarray:
    rng = np.random.default_rng(3)
    t = np.arange(int(SR * seconds)) / SR
    x = 0.3 * np.exp(-((t % 0.25) * 30.0)) * rng.standard_normal(t.shape[0]) \
        + 0.001 * rng.standard_normal(t.shape[0])
    return np.stack([x, 0.8 * x]).astype(np.float32)


def test_the_offline_gate_is_the_native_law():
    audio = _drums()
    for p in ({"threshold_db": -30.0, "ratio": 4.0, "attack_ms": 1.0, "release_ms": 80.0,
               "hold_ms": 20.0, "range_db": -20.0, "hysteresis_db": 6.0},
              {"threshold_db": -26.0, "ratio": 20.0, "attack_ms": 20.0, "release_ms": 5.0,
               "hold_ms": 0.0, "range_db": -80.0, "hysteresis_db": 0.0}):
        expected_db = _native_loop(np.max(np.abs(audio), axis=0).astype(np.float64), p)
        out = dsp._gate(audio, SR, {"type": "gate", **p})
        ours_db = 20.0 * np.log10(np.maximum(np.abs(out[0]), 1e-12) / np.maximum(np.abs(audio[0]), 1e-12))
        loud = np.abs(audio[0]) > 1e-4
        assert np.max(np.abs(ours_db[loud] - expected_db[loud])) < 0.01, p


def test_a_tone_over_the_threshold_passes_clean():
    t = np.arange(SR) / SR
    tone = (10 ** (-16 / 20)) * np.sin(2 * np.pi * 100 * t)
    out = dsp._gate(np.stack([tone, tone]).astype(np.float32), SR,
                    {"type": "gate", "threshold_db": -26.0, "ratio": 8.0, "attack_ms": 1.0,
                     "release_ms": 5.0, "hold_ms": 0.0, "range_db": -80.0, "hysteresis_db": 0.0})
    half = SR // 2
    assert abs(20 * np.log10(np.sqrt(np.mean(out[0][half:] ** 2)) / np.sqrt(np.mean(tone[half:] ** 2)))) < 0.01
