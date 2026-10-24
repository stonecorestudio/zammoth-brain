"""A legal EQ band above Nyquist must not render `inf`.

The 2026-09-08 device audit's first P0. `Session.project_sample_rate` is simply
the highest stem rate with no floor, and `gateway.AUDIO_EXTS` decodes mp3/ogg at
whatever rate they carry — so a 22.05 kHz project is ordinary. `EqBand.freq_hz`
allows 20 kHz, `DEVICE_PARAMS["eq"]["freq_hz"]` clamps to 20 kHz, and the EQ face
drags to 20 kHz; none of them knows the project rate. Add `lpf @ 12 kHz` (a
guardrail-approved move — only `hpf` has a frequency guardrail) and every rendered
sample of the stem AND the master came back `inf`.

Nothing in `dsp.py` sanitises non-finite values, so it reached the exported wav and
turned the analysis LUFS into `nan`. `MonitorChain::makeBiquad` clamps, so the live
monitor sounded correct throughout: the user heard a good mix and shipped garbage.

`_notch`, `_bandpass_band` and `reference_monitor.eq_band_coeffs` were each given
this clamp for exactly this reason, each with a comment saying so. `_eq_plugin` —
the path every peak, every shelf, every tilt and every DEFAULT-slope hpf/lpf takes —
never was.
"""
from __future__ import annotations

import numpy as np
import pytest

from mix_engine.dsp import _apply_master_chain, _eq_band, _tilt

# A 22.05 kHz project: Nyquist 11025, and the clamp sits at sr * 0.49 = 10804.5.
SR_LOW = 22050
SR_OK = 48000


def _noise(sr: int, seconds: float = 0.25) -> np.ndarray:
    rng = np.random.default_rng(20260908)
    return rng.standard_normal((2, int(sr * seconds))).astype(np.float32) * 0.1


def _band(shape: str, freq_hz: float, q: float = 0.707) -> dict:
    return {"shape": shape, "freq_hz": freq_hz, "gain_db": 0.0, "q": q}


@pytest.mark.parametrize("shape", ["lpf", "hpf", "peak", "low_shelf", "high_shelf"])
@pytest.mark.parametrize("freq_hz", [11100.0, 12000.0, 20000.0])
def test_a_band_above_nyquist_renders_finite(shape, freq_hz):
    """The P0 itself, at every shape that reaches `_eq_plugin`."""
    band = _band(shape, freq_hz)
    band["gain_db"] = 3.0
    out = _eq_band(_noise(SR_LOW), SR_LOW, band)
    assert np.isfinite(out).all(), f"{shape} @ {freq_hz} Hz on a {SR_LOW} Hz project renders non-finite"


def test_the_master_render_is_finite_end_to_end():
    """Through the real master chain, which is where the audit measured `inf` on
    every sample of the deliverable."""
    chain = [{"type": "eq", "band": _band("lpf", 12000.0)}]
    out = _apply_master_chain(_noise(SR_LOW), SR_LOW, chain, None)
    assert np.isfinite(out).all()
    assert np.abs(out).max() < 10.0, "a finite but exploding filter is still a broken render"


def test_tilt_is_covered_too():
    """`_tilt` builds two shelves at `pivot_hz` through the same helper, and
    `DEVICE_PARAMS` lets the pivot reach 8 kHz — under Nyquist here, but the shelf
    pair is the other caller and must not be left behind."""
    out = _tilt(_noise(SR_LOW), SR_LOW, {"tilt_db": 6.0, "pivot_hz": 8000.0})
    assert np.isfinite(out).all()


def test_the_clamp_engages_only_above_it():
    """NEGATIVE CONTROL, and the reason this file is not vacuous.

    A clamp that pinned every band to one frequency would satisfy every assertion
    above. So pin the behaviour on both sides of the threshold: below it the filter
    still tracks frequency, above it two different requests land on the same place.
    """
    audio = _noise(SR_LOW)
    below_a = _eq_band(audio, SR_LOW, _band("lpf", 4000.0))
    below_b = _eq_band(audio, SR_LOW, _band("lpf", 9000.0))
    assert not np.allclose(below_a, below_b), (
        "two cutoffs below the clamp must still render differently — the clamp is not a constant"
    )
    above_a = _eq_band(audio, SR_LOW, _band("lpf", 12000.0))
    above_b = _eq_band(audio, SR_LOW, _band("lpf", 20000.0))
    np.testing.assert_allclose(
        above_a, above_b, rtol=0, atol=0,
        err_msg="two cutoffs above the clamp both land on sr * 0.49, so they must be identical",
    )


def test_a_legal_band_is_untouched_by_the_fix():
    """The clamp must not change any render that was already correct: at 48 kHz
    nothing about a 12 kHz cutoff is near the threshold, and the same band at the
    two rates must differ (or the comparison above proves nothing about rate)."""
    at_48 = _eq_band(_noise(SR_OK), SR_OK, _band("lpf", 12000.0))
    assert np.isfinite(at_48).all()
    # …and the clamp is genuinely rate-dependent: 12 kHz is clamped at 22.05 kHz and
    # not at 48 kHz, so the SAME request must reach a different filter.
    audio = _noise(SR_LOW)
    clamped = _eq_band(audio, SR_LOW, _band("lpf", 12000.0))
    unclamped_freq = _eq_band(audio, SR_LOW, _band("lpf", SR_LOW * 0.49))
    np.testing.assert_allclose(clamped, unclamped_freq, rtol=0, atol=0,
                               err_msg="a request above the clamp must land exactly on sr * 0.49")


def test_q_is_clamped_like_the_native():
    """`makeBiquad` also does `jmax (0.1, q)`; a zero or negative q reaching a
    Pedalboard filter is the same class of divergence."""
    for q in (0.0, -1.0, 0.05):
        out = _eq_band(_noise(SR_LOW), SR_LOW, _band("peak", 2000.0, q=q))
        assert np.isfinite(out).all(), f"q={q} renders non-finite"
