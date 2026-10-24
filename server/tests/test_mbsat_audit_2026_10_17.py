"""The multiband saturation's audit, 2026-10-17 (the device-by-device audit) - the offline engine.

The native half is `--mbsat-audit`. Here:
  * the offline multiband saturation splits as the DAW does - the real 3-way LR4 (`_lr4_three_way`),
    against a native that split by the COMPLEMENT and an offline FFT partition (audit 05 F10): with the
    low band driven 24 dB and the others dry, an 80 Hz tone comes out -12.3 dB in both engines (the
    native's complement put it +2.2 dB OVER the input);
  * the reference monitor renders the bands, not a flat saturator (audit 05 F9).
"""
from __future__ import annotations

import numpy as np

from mix_engine import dsp, reference_monitor

SR = 48000


def _entry(driven: str) -> dict:
    e = {"type": "saturation", "drive_db": 3.0, "curve": 0, "mix": 1.0,
         "crossover_lo_hz": 150.0, "crossover_hi_hz": 2000.0}
    for b in ("low", "mid", "high"):
        e[b] = {"drive_db": 24.0, "output_db": 0.0, "mix": 1.0} if b == driven else \
               {"drive_db": 0.0, "output_db": 0.0, "mix": 0.0}
    return e


def _tone(hz: float) -> np.ndarray:
    t = np.arange(SR * 4) / SR
    return np.stack([0.5 * np.sin(2 * np.pi * hz * t)] * 2, axis=1)


def _level_db(x: np.ndarray, y: np.ndarray) -> float:
    h = slice(len(x) // 2, len(x))
    return float(10 * np.log10(np.sum(np.asarray(y)[h, 0] ** 2) / np.sum(x[h, 0] ** 2)))


def test_driving_the_low_band_takes_a_bass_tone_down_as_the_daw_does():
    x = _tone(80.0)
    assert abs(_level_db(x, dsp._multiband_saturate(x, _entry("low"), SR)) - (-12.32)) < 0.05


def test_driving_the_mid_band_leaves_a_bass_tone_alone():
    x = _tone(80.0)
    assert _level_db(x, dsp._multiband_saturate(x, _entry("mid"), SR)) > -0.5


def test_the_reference_monitor_renders_the_bands():
    x = _tone(80.0)
    e = _entry("low")
    want = dsp._multiband_saturate(x, e, SR)
    got = reference_monitor.render_monitor(x, SR, {"entries": [e]})
    assert np.max(np.abs(np.asarray(got) - want)) < 1e-6
