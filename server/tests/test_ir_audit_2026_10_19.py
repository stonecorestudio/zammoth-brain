"""The IR Reverb audit (2026-10-19, the device-by-device audit's twenty-first device).

One impulse response, made the same way by the three engines that make it - the native chain
(`MonitorChain::proceduralIr`), the offline render (`dsp._procedural_ir`) and the reference monitor
(`reference_monitor._procedural_ir`):
  - the same noise: a counter hash both languages compute exactly (it was numpy's normal noise
    against juce::Random's uniform noise - two different filters, per band up to 3.9 dB apart);
  - normalised BEFORE the tone filter, so TONE is a shelf on the tail and not a level knob (Tone 0
    lifted the lows by 10 dB, and the level moved with the sample rate);
  - a room's three early reflections there to be heard (they were a single sample under the noise);
  - every parameter held to its table offline too (a stored 1000 ms pre-delay rendered 1 s here
    and 250 ms in the DAW), and the pre-delay rounded as the native rounds it.
The native half - the pinned samples, the tail kept across a change, the fade-in, the face, the
per-device taps - is `--ir-audit` (daw/Source/IrAuditLegs.h), whose pins this file reads.
"""
from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest

from mix_engine import dsp
from mix_engine import reference_monitor as rm

ROOT = pathlib.Path(__file__).resolve().parents[2]
LEG = ROOT / "daw" / "Source" / "IrAuditLegs.h"


def _pins():
    text = LEG.read_text(encoding="utf-8")
    block = re.search(r"kIrPins\[\]\s*=\s*\{(.*?)\n\s*\};", text, re.S)
    assert block, "the native pins are not in the leg"
    rows = re.findall(r'\{\s*"(\w+)",\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*(\d+),\s*(\d+),\s*'
                      r'([-+\d.eE]+)f?\s*\}', block.group(1))
    assert len(rows) >= 12, rows
    return [(m, float(d), float(t), float(p), int(sr), int(i), float(v))
            for m, d, t, p, sr, i, v in rows]


def _band_db(ir: np.ndarray, sr: int, lo: float, hi: float) -> float:
    n = 1 << int(np.ceil(np.log2(len(ir))) + 1)
    power = np.abs(np.fft.rfft(ir, n)) ** 2
    f = np.fft.rfftfreq(n, 1.0 / sr)
    m = (f > lo) & (f < hi)
    return float(10.0 * np.log10(power[m].mean()))


def _pink_db(ir: np.ndarray, sr: int) -> float:
    n = 1 << int(np.ceil(np.log2(len(ir))) + 1)
    power = np.abs(np.fft.rfft(ir, n)) ** 2
    f = np.fft.rfftfreq(n, 1.0 / sr)
    m = (f > 20.0) & (f < 16000.0)
    w = 1.0 / f[m]
    return float(10.0 * np.log10((power[m] * w).sum() / w.sum()))


def test_the_offline_impulse_is_the_native_one_sample_for_sample():
    """The native leg checks MonitorChain::proceduralIr against the same rows."""
    for mode, decay, tone, pre, sr, i, value in _pins():
        pre_n = int(np.floor(pre * sr / 1000.0 + 0.5))
        ir = dsp._procedural_ir(mode, decay, tone, sr)
        got = 0.0 if i < pre_n else float(ir[i - pre_n])
        assert got == pytest.approx(value, rel=2e-4, abs=2e-7), (mode, decay, tone, pre, sr, i)


def test_the_reference_monitor_makes_the_same_impulse():
    for mode in ("room", "hall", "plate"):
        for sr in (44100, 48000, 96000):
            a = dsp._procedural_ir(mode, 1.3, 0.37, sr)
            b = rm._procedural_ir(mode, 1.3, 0.37, sr)
            assert a.shape == b.shape
            assert np.allclose(a, b, rtol=0, atol=1e-12), (mode, sr)


def test_tone_is_a_shelf_not_a_level():
    """The lows hold at every Tone and every rate (they moved 10 dB with Tone, 2 dB with the rate)."""
    per_rate = []
    for sr in (44100, 48000, 96000):
        lows = [_band_db(dsp._procedural_ir("hall", 2.0, tone, sr), sr, 90.0, 180.0)
                for tone in (0.0, 0.5, 1.0)]
        assert max(lows) - min(lows) < 0.3, (sr, [round(v, 2) for v in lows])
        per_rate.append(lows[1])
    # across rates the noise is another draw, so the band's own scatter is in this (+-0.5 dB)
    assert max(per_rate) - min(per_rate) < 1.0, [round(v, 2) for v in per_rate]
    # ...and it IS a shelf: Tone 0 takes the top down
    dark = dsp._procedural_ir("hall", 2.0, 0.0, 48000)
    bright = dsp._procedural_ir("hall", 2.0, 1.0, 48000)
    assert _band_db(bright, 48000, 8000, 12000) - _band_db(dark, 48000, 8000, 12000) > 12.0


def test_the_default_space_is_as_loud_as_it_was():
    """Tone 0.5, Decay 2, 48 kHz: the music-weighted level under the old law, per space."""
    old = {"room": -9.37, "hall": -9.87, "plate": -9.39}
    for mode, was in old.items():
        now = _pink_db(dsp._procedural_ir(mode, 2.0, 0.5, 48000), 48000)
        assert abs(now - was) < 0.6, (mode, round(now, 2), was)


def test_a_rooms_early_reflections_stand_out():
    """Each spike's energy against one millisecond of the tail around it (they were 0.4-0.6 of a
    single noise sample: nothing anyone could hear)."""
    for sr in (44100, 48000, 96000):
        ir = dsp._procedural_ir("room", 2.0, 1.0, sr)
        ms = sr // 1000
        for when, at_least_db in ((7.0, 2.5), (11.0, 0.0)):
            k = int(np.floor(when * sr / 1000.0 + 0.5))
            spike = float(np.sum(ir[k - 2: k + 3] ** 2))
            around = np.concatenate([ir[k - ms: k - 3], ir[k + 4: k + ms]])
            diffuse = float(np.mean(around ** 2)) * ms
            assert 10.0 * np.log10(spike / diffuse) > at_least_db, (sr, when)


def test_every_parameter_is_held_to_its_table_offline():
    sr = 48000
    hit = np.zeros((sr * 10, 2))
    hit[0] = 1.0

    def render(**kw):
        entry = {"type": "return_convolution", "ir": "hall", "decay_s": 2.0, "pre_delay_ms": 10.0,
                 "tone": 0.5, "mix": 1.0}
        entry.update(kw)
        return dsp._return_convolution(hit, sr, entry), rm._return_convolution(hit, float(sr), entry)

    for wild, held in (({"pre_delay_ms": 1000.0}, {"pre_delay_ms": 250.0}),
                       ({"pre_delay_ms": -40.0}, {"pre_delay_ms": 0.0}),
                       ({"decay_s": 20.0}, {"decay_s": 8.0}),
                       ({"decay_s": 0.01}, {"decay_s": 0.2}),
                       ({"tone": 3.0}, {"tone": 1.0}),
                       ({"tone": -1.0}, {"tone": 0.0})):
        a, ra = render(**wild)
        b, rb = render(**held)
        assert np.array_equal(a, b), wild
        assert np.array_equal(ra, rb), wild


def test_the_pre_delay_rounds_as_the_native_rounds():
    sr = 48000
    hit = np.zeros((4000, 1))
    hit[0] = 1.0
    entry = {"type": "return_convolution", "ir": "plate", "decay_s": 0.5,
             "pre_delay_ms": 10.0125, "tone": 1.0, "mix": 1.0}   # 480.6 samples: the native's 481
    for out in (dsp._return_convolution(hit, sr, entry), rm._return_convolution(hit, float(sr), entry)):
        assert np.all(np.abs(out[:481, 0]) < 1e-12)   # fftconvolve's own rounding, no more
        assert abs(out[481, 0]) > 1e-6
