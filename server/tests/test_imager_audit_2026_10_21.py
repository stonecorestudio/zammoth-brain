"""The Stereo Imager audit (2026-10-21, the device-by-device audit's twenty-fourth device).

ONE LAW IN BOTH ENGINES (James's picks: LR8, and always split): the band width's real 3-way
48 dB/oct Linkwitz-Riley split at the crossovers, each band's side scaled, the bands summed - at
all widths 1 too (the split's allpass). The offline engine ran a non-causal whole-file FFT mask
(the low side gone at width_low 0) while the DAW's export ran LR4 in float (-23.6 dB), and the DAW
skipped the recombination at all widths 1, so a ride on a fresh imager was never heard and leaving
1.00 clicked. The crossovers resolve one way for the engines, the face and the guardrail.
The native half is `--imager-audit` (daw/Source/ImagerAuditLegs.h), whose pins this file reads.
"""
from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest

from mix_engine import dsp, guardrails
from mix_engine import reference_monitor as rm
from mix_engine.models import imager_crossovers

ROOT = pathlib.Path(__file__).resolve().parents[2]
LEG = ROOT / "daw" / "Source" / "ImagerAuditLegs.h"


def _pins():
    text = LEG.read_text(encoding="utf-8")
    block = re.search(r"kImgPins\[\]\s*=\s*\{(.*?)\n\s*\};", text, re.S)
    rows = re.findall(r"\{\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*(\d+),\s*(\d+),\s*"
                      r"([-+\d.eE]+)f,\s*([-+\d.eE]+)f\s*\}", block.group(1))
    assert len(rows) >= 16, rows
    return rows


def _pin_signal(sr: int, n: int) -> np.ndarray:
    i = np.arange(n)
    left = 0.4 * np.sin(2 * np.pi * 47 * i / sr) + 0.3 * np.sin(2 * np.pi * 3100 * i / sr)
    right = 0.4 * np.sin(2 * np.pi * 47 * i / sr + 1.0) + 0.3 * np.sin(2 * np.pi * 3100 * i / sr - 0.7)
    return np.stack([left, right], axis=1).astype(np.float32).astype(np.float64)


def test_the_offline_imager_is_the_native_one():
    for wl, wm, wh, lo, hi, sr, idx, left, right in _pins():
        entry = {"width_low": float(wl), "width_mid": float(wm), "width_high": float(wh),
                 "crossover_lo_hz": float(lo), "crossover_hi_hz": float(hi)}
        y = dsp._imager(_pin_signal(int(sr), 2048), int(sr), entry)
        assert y[int(idx), 0] == pytest.approx(float(left), rel=1e-6, abs=1e-7)
        assert y[int(idx), 1] == pytest.approx(float(right), rel=1e-6, abs=1e-7)


def test_the_reference_monitor_runs_the_same_split():
    x = np.random.default_rng(3).standard_normal((48000, 2)) * 0.3
    for entry in ({"width_low": 0.0}, {"width_high": 2.0, "crossover_hi_hz": 3000.0}, {},
                  {"width_low": 3.0, "crossover_lo_hz": 3000.0, "crossover_hi_hz": 2000.0}):
        assert np.array_equal(dsp._imager(x, 48000, entry), rm._imager(x, 48000.0, entry))


def test_it_is_causal_and_keeps_every_level_at_unity():
    x = np.zeros((48000, 2))
    x[30000:30100, 0] = 0.8
    assert np.all(dsp._imager(x, 48000, {"width_low": 0.0, "width_high": 1.6})[:30000] == 0.0)
    n = np.random.default_rng(7).standard_normal((48000 * 2, 2)) * 0.3
    y = dsp._imager(n, 48000, {})   # all widths 1: the split's allpass

    def side_db(v, lo, hi):
        s = 0.5 * (v[:, 0] - v[:, 1])
        p = np.abs(np.fft.rfft(s)) ** 2
        f = np.fft.rfftfreq(len(s), 1 / 48000)
        return 10 * np.log10(p[(f > lo) & (f < hi)].mean())
    for lo, hi in ((20, 150), (150, 2000), (2000, 20000)):
        assert abs(side_db(y, lo, hi) - side_db(n, lo, hi)) < 0.05
    low = dsp._imager(n, 48000, {"width_low": 0.0})
    assert side_db(low, 20, 100) - side_db(n, 20, 100) < -30.0    # LR8: about -40 below 100 Hz
    assert abs(side_db(low, 300, 1000) - side_db(n, 300, 1000)) < 0.3


def test_one_resolution_of_the_crossovers_for_the_engines_and_the_guardrail():
    assert imager_crossovers(3000.0, 2000.0) == (3000.0, 4500.0)
    assert imager_crossovers(10.0, 50000.0) == (30.0, 16000.0)
    assert imager_crossovers(float("nan"), None) == (150.0, 2000.0)
    pieces = guardrails._side_pieces({"type": "imager", "width_low": 0.0, "width_mid": 1.0,
                                      "width_high": 1.0, "crossover_lo_hz": 3000.0,
                                      "crossover_hi_hz": 2000.0})
    assert pieces[0][1] == 3000.0 and pieces[1] == (3000.0, 4500.0, 1.0)


def test_a_stored_width_is_held_to_the_table_offline():
    x = np.random.default_rng(2).standard_normal((4800, 2))
    assert np.array_equal(dsp._imager(x, 48000, {"width_low": 3.0}), dsp._imager(x, 48000, {"width_low": 2.0}))
