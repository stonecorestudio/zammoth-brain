"""The export oracle's verdict must see a truncated bounce.

2026-09-01 sweep, P0: `daw_export_oracle` judged the native export against the
Python render on integrated LUFS + an octave-band spectral shape — both chosen
to be timing-invariant, which makes them LENGTH-invariant. Measured on the demo
master, an export missing 75 % of its tail scored dLUFS 0.02 dB / band corr
0.999 and PASSED; the two lengths were only printed. A bounce that stops at the
last clip, drops a reverb tail, or has a render-length off-by-one is the most
likely export-path regression, and the invariant's only gate was blind to it.

Free: `compare()` is pure numpy over synthetic audio; no DAW binary, no render.
"""
from __future__ import annotations

import numpy as np
import pytest

from orchestrator.daw_export_oracle import LEN_TOL_S, compare


@pytest.fixture
def reference():
    rng = np.random.default_rng(7)
    sr = 48000
    # 6 s of quiet noise: loud enough for the LUFS gate, long enough that a
    # 15 % cut is well outside LEN_TOL_S
    return rng.standard_normal(sr * 6).astype(np.float64) * 0.1, sr


def test_an_identical_render_passes(reference):
    mono, sr = reference
    ok, lines = compare(mono, sr, mono.copy(), sr)
    assert ok, lines


@pytest.mark.parametrize("keep", [0.25, 0.5, 0.85])
def test_a_truncated_bounce_fails_on_length_not_on_luck(reference, keep):
    mono, sr = reference
    ok, lines = compare(mono, sr, mono[: int(len(mono) * keep)], sr)
    assert not ok, (keep, lines)
    assert any("length" in line for line in lines), lines


def test_a_rate_mismatch_fails(reference):
    mono, sr = reference
    ok, lines = compare(mono, sr, mono, 44100)
    assert not ok
    assert any("rate" in line for line in lines), lines


def test_a_block_of_padding_is_tolerated(reference):
    """PDC / a lookahead tail may pad the bounce by a block; that is not a
    truncation and must not trip the length gate."""
    mono, sr = reference
    padded = np.concatenate([mono, np.zeros(2048)])
    assert 2048 / sr < LEN_TOL_S
    ok, lines = compare(mono, sr, padded, sr)
    assert ok, lines


# ---- the SIDE axis (2026-09-08) --------------------------------------------------------------
# The 2026-09-08 device audit found `compare` blind to the entire stereo device family: every
# metric above is computed on the MONO downmix, and width / band_width / imager / utility are
# mono-sum-preserving to 0.000000000 dB by construction, so no divergence in any of them could
# register. The audit measured `band_width` at width 0 removing 6.89 dB of a band's side natively
# where the offline reference removes all of it, with no gate in the repo able to see it — and
# `_controlled_chain` already carried a comment recording that a width stage had been REMOVED from
# the linear profile in 2026-09-01 because nothing could measure it. These pin the new axis, and
# especially the two ways it must decline to judge: silence and a mono reference both "agree"
# perfectly, and neither is coverage.

@pytest.fixture
def stereo_pair():
    """A mid/side pair with real, independent side content — the shape the `stereo` profile
    generates, and the only shape on which the side metrics mean anything."""
    rng = np.random.default_rng(20260908)
    sr = 48000
    n = sr * 6
    mono = rng.standard_normal(n).astype(np.float64) * 0.1
    side = rng.standard_normal(n).astype(np.float64) * 0.05
    return mono, side, sr


def test_identical_stereo_renders_pass(stereo_pair):
    mono, side, sr = stereo_pair
    ok, lines = compare(mono, sr, mono, sr, "stereo", side, side, ref_stereo=True)
    assert ok, lines
    assert any("side LUFS" in ln for ln in lines), "the side axis must actually have been measured"


def test_a_side_only_divergence_is_caught(stereo_pair):
    """THE POINT. The mono sum is untouched, so every pre-existing metric passes; only the side
    axis can see this. A width device rendering at the wrong amount looks exactly like it."""
    mono, side, sr = stereo_pair
    ok, lines = compare(mono, sr, mono, sr, "stereo", side, side * 0.5, ref_stereo=True)
    assert not ok, lines
    # …and it failed on the SIDE, not incidentally on something else.
    assert any("side LUFS" in ln for ln in lines)
    mono_ok, _ = compare(mono, sr, mono, sr, "stereo")
    assert mono_ok, "the mono metrics must be blind to this — that is why the axis was needed"


def test_a_collapsed_stereo_image_is_caught(stereo_pair):
    mono, side, sr = stereo_pair
    ok, lines = compare(mono, sr, mono, sr, "stereo", side, np.zeros_like(side), ref_stereo=True)
    assert not ok
    assert any("stereo image" in ln for ln in lines), lines


def test_silence_is_not_agreement(stereo_pair):
    """Two silent sides correlate perfectly. Reporting that as a pass would be inventing
    coverage, so the axis declines to judge and says so."""
    mono, side, sr = stereo_pair
    z = np.zeros_like(side)
    ok, lines = compare(mono, sr, mono, sr, "stereo", z, z, ref_stereo=True)
    assert ok
    assert any("n/a — no side content" in ln for ln in lines), lines


def test_a_mono_reference_declines_to_judge(stereo_pair):
    """The offline render of an all-mono project is 1-CHANNEL and cannot carry a side, while the
    DAW's dual-mono path can — on the nonlinear profile the chorus/flanger `stereo_deg` spread
    puts ~-37 dBFS RMS there. Failing on that would be failing for a channel-LAYOUT difference
    rather than an export-path fault, so it is reported and not asserted. The number still has to
    appear in the output: a skipped axis that says nothing reads as coverage."""
    mono, side, sr = stereo_pair
    ok, lines = compare(mono, sr, mono, sr, "stereo", np.zeros_like(side), side, ref_stereo=False)
    assert ok, lines
    assert any("1-channel (mono project)" in ln for ln in lines), lines
    assert any("dBFS RMS of side" in ln for ln in lines), "the unjudgeable side level must be printed"
