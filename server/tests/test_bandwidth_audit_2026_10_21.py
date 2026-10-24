"""The Band Width audit (2026-10-21, the device-by-device audit's twenty-third device).

ONE LAW IN BOTH ENGINES (James's pick, LR8): the band is the middle of a real 3-way Linkwitz-Riley
split at its edges, 48 dB/oct, and only its side is scaled. Until now the offline engine cut the
band out of the whole file with an FFT mask (gone at width 0, and non-causal: a hit leaked into the
silence before it) while the DAW's export ran one 2nd-order band-pass (6.9 dB down at width 0).
The native half - the pins, the step per entry, the memory, the rides, the glide, the face - is
`--bandwidth-audit` (daw/Source/BandWidthAuditLegs.h), whose pins this file reads; the stereo
oracle profile now carries a band width.
"""
from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest

from mix_engine import dsp
from mix_engine import reference_monitor as rm

ROOT = pathlib.Path(__file__).resolve().parents[2]
LEG = ROOT / "daw" / "Source" / "BandWidthAuditLegs.h"


def _pins():
    text = LEG.read_text(encoding="utf-8")
    block = re.search(r"kBwPins\[\]\s*=\s*\{(.*?)\n\s*\};", text, re.S)
    rows = re.findall(r'\{\s*"(\w*)",\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*(\d+),\s*(\d+),\s*'
                      r'([-+\d.eE]+)f,\s*([-+\d.eE]+)f\s*\}', block.group(1))
    assert len(rows) >= 12, rows
    return rows


def _pin_signal(sr: int, n: int) -> np.ndarray:
    i = np.arange(n)
    left = 0.4 * np.sin(2 * np.pi * 47 * i / sr) + 0.3 * np.sin(2 * np.pi * 3100 * i / sr)
    right = 0.4 * np.sin(2 * np.pi * 47 * i / sr + 1.0) + 0.3 * np.sin(2 * np.pi * 3100 * i / sr - 0.7)
    return np.stack([left, right], axis=1).astype(np.float32).astype(np.float64)   # the leg feeds floats


def test_the_offline_band_width_is_the_native_one():
    for band, lo, hi, w, sr, idx, left, right in _pins():
        entry = {"width": float(w)}
        if band:
            entry["band"] = band
        else:
            entry.update(lo_hz=float(lo), hi_hz=float(hi))
        y = dsp._band_width_scale(_pin_signal(int(sr), 2048), int(sr), entry)
        assert y[int(idx), 0] == pytest.approx(float(left), rel=1e-6, abs=1e-7)
        assert y[int(idx), 1] == pytest.approx(float(right), rel=1e-6, abs=1e-7)


def test_the_reference_monitor_runs_the_same_split():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((48000, 2)) * 0.3
    for entry in ({"band": "sub", "width": 0.0}, {"band": "air", "width": 2.0},
                  {"lo_hz": 100.0, "hi_hz": 4000.0, "width": 0.5}, {"freq_hz": 2828.4, "q": 1.4142, "width": 0.0}):
        assert np.array_equal(dsp._band_width_scale(x, 48000, entry), rm._band_width(x, 48000.0, entry))


def test_it_is_causal():
    """The FFT mask leaked a hit into the silence before it (and wrapped it to the file's start)."""
    x = np.zeros((48000, 2))
    x[30000:30100, 0] = 0.8
    y = dsp._band_width_scale(x, 48000, {"band": "sub", "width": 0.0})
    assert np.all(y[:30000] == 0.0)


def test_width_0_takes_the_band_down_and_leaves_the_rest():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((48000 * 4, 2)) * 0.3
    y = dsp._band_width_scale(x, 48000, {"band": "bass", "width": 0.0})

    def side_db(v, lo, hi):
        s = 0.5 * (v[:, 0] - v[:, 1])
        power = np.abs(np.fft.rfft(s)) ** 2
        f = np.fft.rfftfreq(len(s), 1 / 48000)
        return 10 * np.log10(power[(f > lo) & (f < hi)].mean())

    assert side_db(y, 60, 150) - side_db(x, 60, 150) < -11.0       # the band (LR8: not to silence)
    assert side_db(y, 85, 106) - side_db(x, 85, 106) < -18.0       # its centre
    assert abs(side_db(y, 600, 1200) - side_db(x, 600, 1200)) < 0.1   # two octaves up: untouched
    # the mono sum keeps its magnitude (it passes the split's allpass, so not its samples)
    mono_x = np.abs(np.fft.rfft(x[:, 0] + x[:, 1])) ** 2
    mono_y = np.abs(np.fft.rfft(y[:, 0] + y[:, 1])) ** 2
    assert abs(10 * np.log10(mono_y.sum() / mono_x.sum())) < 0.05


def test_the_band_resolves_and_holds_as_the_native_does():
    sr = 48000
    assert dsp._band_width_edges({"band": "sub", "freq_hz": 2828.4, "q": 1.4142}, sr) == (20.0, 60.0)   # the label wins
    lo, hi = dsp._band_width_edges({"freq_hz": 2828.4, "q": 1.4142}, sr)
    assert lo == pytest.approx(2000.0, rel=1e-3) and hi == pytest.approx(4000.0, rel=1e-3)
    assert dsp._band_width_edges({"lo_hz": 10.0, "hi_hz": 50000.0}, sr) == (20.0, 2000.0)   # 20 Hz, 100:1
    assert dsp._band_width_edges({"band": "air"}, 44100)[1] == pytest.approx(0.45 * 44100)
    rng = np.random.default_rng(2)
    x = rng.standard_normal((4800, 2))
    assert np.array_equal(dsp._band_width_scale(x, sr, {"band": "sub", "width": 5.0}),
                          dsp._band_width_scale(x, sr, {"band": "sub", "width": 2.0}))
    assert np.array_equal(dsp._band_width_scale(x, sr, {"band": "sub", "width": -1.0}),
                          dsp._band_width_scale(x, sr, {"band": "sub", "width": 0.0}))


def test_the_native_side_of_the_law_is_in_the_source():
    chain = (ROOT / "native" / "Source" / "MonitorChain.cpp").read_text(encoding="utf-8")
    block = chain[chain.index('const String pre = "band_width"'):]
    block = block[:block.index("}")]
    assert "freq_hz" not in block and '"width"' in block   # the band is not a ride target
    assert "&& tag != LinStep::BandWidth" in chain      # a step per band width
