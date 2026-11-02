"""The gate's audit, 2026-10-17 (the device-by-device audit) - the offline engine.

`dsp._gate` is `MonitorChain::applyGate` - the one law, `dsp.gate_gain_db`, vectorised. Since
2026-10-27 that law is the four characters of `gate_characters`, and test_gate_characters.py holds
it to a per-sample transcription of the native code (the v3 transcription that stood here is gone
with v3); what stays here is the audit's own finding: a peak-follower detector (the raw |x| the
native read fell to nothing at every zero crossing).
The native half is `--gate-audit`; the export oracle's `nonlinear` profile carries a gate.
"""
from __future__ import annotations

import numpy as np

from mix_engine import dsp

SR = 48000


def test_a_tone_over_the_threshold_passes_clean():
    t = np.arange(SR) / SR
    tone = (10 ** (-16 / 20)) * np.sin(2 * np.pi * 100 * t)
    out = dsp._gate(np.stack([tone, tone]).astype(np.float32), SR,
                    {"type": "gate", "threshold_db": -26.0, "ratio": 8.0, "attack_ms": 1.0,
                     "release_ms": 5.0, "hold_ms": 0.0, "range_db": -80.0, "hysteresis_db": 0.0})
    half = SR // 2
    assert abs(20 * np.log10(np.sqrt(np.mean(out[0][half:] ** 2)) / np.sqrt(np.mean(tone[half:] ** 2)))) < 0.01
