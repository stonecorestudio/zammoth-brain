"""What "mono this band" actually does, measured - ONE law in both engines since 2026-10-21.

2026-09-08 device audit, 07-F4: the offline render cut the band out with an exact rFFT mask (gone at
width 0, and non-causal) while the native monitor - and so the DAW's bounce, the deliverable - ran
one 2nd-order band-pass and removed about 6.5 dB of it. This file used to pin that gap.

The band width audit of 2026-10-21 (James's pick, LR8) put both engines on the structure this file
named as the one that would genuinely improve it: an allpass-aligned three-way split (below /
in-band / above), each crossover a 48 dB/oct Linkwitz-Riley, summed back so every path carries the
same phase - and only the in-band SIDE is scaled. `dsp._band_width_scale` and
`reference_monitor._band_width` (the native's transcription, MonitorChain::applyBandWidth) are the
same numbers; `--bandwidth-audit` pins the native to them, and the stereo oracle profile compares
the two engines' exports on the side axis.

What is pinned now is how much a one-octave band's side really goes at width 0 - 11 to 16 dB, not to
silence: a causal split keeps half of each edge - with a window on both sides, so the device can
neither weaken nor change topology silently. The yardstick stays a brickwall, deliberately: it must
not be either engine's own band filter.
"""
from __future__ import annotations

import numpy as np
import pytest

from mix_engine.dsp import _band_width_scale
from mix_engine.models import BAND_EDGES_HZ, BAND_LABELS
from mix_engine.reference_monitor import _band_width

SR = 48000

# Measured 2026-10-21 on the source below (the LR8 split, both engines), in dB of the band's side
# removed at width 0. The window is 1.5 dB under each and 6 dB over.
_LR8_REDUCTION_DB = {
    "sub": 15.1, "bass": 12.1, "low_mid": 12.1, "mid": 12.9,
    "high_mid": 11.2, "presence": 11.4, "brilliance": 11.8, "air": 16.4,
}


def _source() -> np.ndarray:
    rng = np.random.default_rng(5)
    n = SR
    return np.stack([rng.standard_normal(n) * 0.2, rng.standard_normal(n) * 0.2], axis=1)


def _inband_side_db(a: np.ndarray, lo: float, hi: float) -> float:
    """The side signal's RMS INSIDE [lo, hi) — the quantity "mono this band" is about.
    Measured with a brickwall, deliberately: the yardstick must not be either engine's
    own band filter."""
    s = a[:, 0] - a[:, 1]
    spec = np.fft.rfft(s)
    freqs = np.fft.rfftfreq(len(s), 1.0 / SR)
    spec[(freqs < lo) | (freqs >= hi)] = 0.0
    return 20.0 * np.log10(float(np.sqrt(np.mean(np.fft.irfft(spec, len(s)) ** 2))) + 1e-18)


def _edges(band: str) -> tuple[float, float]:
    i = BAND_LABELS.index(band)
    return float(BAND_EDGES_HZ[i]), float(BAND_EDGES_HZ[i + 1])


@pytest.mark.parametrize("band", BAND_LABELS)
def test_both_engines_take_the_band_s_side_down_by_the_measured_amount(band):
    """Bounded on BOTH sides: too little and the device stopped working, too much and the topology
    changed under us and every number in this file needs re-measuring."""
    lo, hi = _edges(band)
    x = _source()
    for out in (_band_width_scale(x.copy(), SR, {"band": band, "width": 0.0}),
                _band_width(x, float(SR), {"band": band, "width": 0.0})):
        got = -(_inband_side_db(out, lo, hi) - _inband_side_db(x, lo, hi))
        assert _LR8_REDUCTION_DB[band] - 1.5 <= got <= _LR8_REDUCTION_DB[band] + 6.0, (band, round(got, 2))


@pytest.mark.parametrize("band", ["sub", "mid", "air"])
def test_width_one_keeps_every_level(band):
    """NEGATIVE CONTROL. At width 1.0 the device is the split's allpass: every band's level, the
    side's and the mid's, is the input's (no longer bit-identical - James's pick, the price of a
    real split). Without this a device that did nothing at all would satisfy nothing above."""
    x = _source()
    out = _band_width_scale(x.copy(), SR, {"band": band, "width": 1.0})
    for lo, hi in ((20.0, 200.0), (200.0, 2000.0), (2000.0, 20000.0)):
        assert abs(_inband_side_db(out, lo, hi) - _inband_side_db(x, lo, hi)) < 0.05
    mono = lambda a: np.abs(np.fft.rfft(a[:, 0] + a[:, 1])) ** 2
    assert abs(10 * np.log10(mono(out).sum() / mono(x).sum())) < 0.05


def test_the_engines_are_one_law():
    """The gap this file used to pin (~300 dB: a brickwall mask against a causal filter) is gone."""
    x = _source()
    for band in BAND_LABELS:
        np.testing.assert_array_equal(_band_width_scale(x.copy(), SR, {"band": band, "width": 0.0}),
                                      _band_width(x, float(SR), {"band": band, "width": 0.0}))
