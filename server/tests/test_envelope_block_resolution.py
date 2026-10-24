"""Time constants faster than the envelope block must actually do something.

`_smooth_env_db` used to clamp its coefficients to the 5 ms block
(`max(attack_ms, block_ms)`), so EVERY `attack_ms` in [0.1, 5.0] rendered
bit-identically offline — the whole fast-gate region, the de-esser's 1 ms
default, and the transient shaper's `speed` — while the native monitor, which
detects per sample, honoured all of them. The knob moved what you heard live
and nothing in the deliverable.

`_env_block_s` now picks a finer block for a faster constant, and keeps the
5 ms block for anything at or above it, so slower renders are unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest

from mix_engine import dsp


def _burst(sample_rate: int = 48000) -> np.ndarray:
    """(channels, frames) — a quiet floor then a loud tone, so a gate has to open."""
    n = sample_rate
    t = np.arange(n) / sample_rate
    sig = np.zeros((1, n), dtype=np.float32)
    sig[0, : n // 2] = 0.0005 * np.sin(2 * np.pi * 300 * t[: n // 2])
    sig[0, n // 2:] = 0.5 * np.sin(2 * np.pi * 300 * t[n // 2:])
    return sig


GATE = dict(type="gate", threshold_db=-30.0, ratio=4.0, release_ms=100.0,
            hold_ms=0.0, range_db=-80.0, hysteresis_db=0.0,
            sc_hpf_hz=0.0, sc_lpf_hz=20000.0)


def test_block_follows_the_fastest_constant():
    assert dsp._env_block_s(10.0) == dsp.ENV_BLOCK_S      # slow: unchanged
    assert dsp._env_block_s(5.0) == dsp.ENV_BLOCK_S       # at the block: unchanged
    assert dsp._env_block_s(1.0) < dsp.ENV_BLOCK_S        # faster: finer
    assert dsp._env_block_s(0.1) >= dsp.ENV_BLOCK_MIN_S   # floored, never 0


@pytest.mark.parametrize("attack_ms", [0.1, 2.0, 5.0])
def test_gate_attack_below_the_block_is_audible(attack_ms):
    sig = _burst()
    base = dsp._gate(np.ascontiguousarray(sig), 48000, dict(GATE, attack_ms=1.0))
    out = dsp._gate(np.ascontiguousarray(sig), 48000, dict(GATE, attack_ms=attack_ms))
    assert not np.array_equal(out, base), (
        f"gate attack_ms={attack_ms} renders identically to the 1 ms default — "
        "the block envelope has swallowed the knob again")


def _ess_burst(sample_rate: int = 48000) -> np.ndarray:
    """(channels, frames) — silence, then a burst of 8 kHz sibilance.

    Must be real high-frequency content: since the split became Linkwitz-Riley 4th
    order the low leg rejects 1 kHz at -64 dB (it was -16 dB), so a bass tone no
    longer trips a de-esser — which is the point of the change, and exactly what a
    de-esser should do.
    """
    n = sample_rate
    t = np.arange(n) / sample_rate
    sig = np.zeros((1, n), dtype=np.float32)
    sig[0, n // 2:] = 0.5 * np.sin(2 * np.pi * 8000 * t[n // 2:])
    return sig


def test_deesser_default_attack_is_not_silently_5ms():
    """The shipped default is 1 ms; it used to render as a 5 ms de-esser."""
    sig = _ess_burst()
    de = dict(type="deess", threshold_db=-40.0, freq_hz=6000.0, ratio=8.0,
              release_ms=50.0, range_db=-80.0)
    fast = dsp._deess(np.ascontiguousarray(sig), 48000, dict(de, attack_ms=1.0))
    slow = dsp._deess(np.ascontiguousarray(sig), 48000, dict(de, attack_ms=5.0))
    assert not np.array_equal(fast, slow)


def test_deesser_rejects_low_content_and_reaches_real_depth():
    """The 1st-order split leaked the ess band back through the low leg, so the
    device saturated at ~5 dB however hard `ratio`/`range_db` pushed."""
    sr = 48000
    ess = _ess_burst(sr)
    de = dict(type="deess", threshold_db=-60.0, freq_hz=6000.0, ratio=20.0,
              attack_ms=1.0, release_ms=50.0, range_db=-80.0)
    out = dsp._deess(np.ascontiguousarray(ess), sr, de)
    half = sr // 2
    depth = 20 * np.log10(np.sqrt(np.mean(out[0, half:] ** 2))
                          / np.sqrt(np.mean(ess[0, half:] ** 2)))
    assert depth < -10.0, f"de-esser only reached {depth:.1f} dB on an 8 kHz ess"

    # ...and a 300 Hz tone must NOT trigger it at all
    low = _burst(sr)
    untouched = dsp._deess(np.ascontiguousarray(low), sr, de)
    moved = 20 * np.log10(np.sqrt(np.mean(untouched[0, half:] ** 2))
                          / np.sqrt(np.mean(low[0, half:] ** 2)))
    assert moved > -1.0, f"bass content is being de-essed ({moved:.1f} dB)"


def test_transient_speed_changes_the_render():
    """`speed` divides the detector constants; above ~1 it went sub-block and died."""
    sig = np.ascontiguousarray(_burst().T.astype(np.float64))
    tr = dict(type="transient", attack_db=6.0, sustain_db=0.0, output_db=0.0)
    one = dsp._transient(sig.copy(), 48000, dict(tr, speed=1.0))
    four = dsp._transient(sig.copy(), 48000, dict(tr, speed=4.0))
    assert not np.array_equal(one, four)


def test_duck_attack_below_the_block_is_audible():
    sr = 48000
    key = np.zeros((sr, 2))
    key[sr // 2:, :] = 0.5
    victim = np.ones((sr, 2)) * 0.2
    e = dict(type="duck", key_stem_id="k", threshold_db=-30.0, depth_db=6.0,
             release_ms=150.0)
    fast = dsp._duck(victim.copy(), sr, key, dict(e, attack_ms=1.0))
    slow = dsp._duck(victim.copy(), sr, key, dict(e, attack_ms=5.0))
    assert not np.allclose(fast, slow)


def test_slow_constants_stay_on_the_5ms_block():
    """Renders whose constants are 5 ms or slower must be untouched by all this."""
    sig = _burst()
    a = dsp._gate(np.ascontiguousarray(sig), 48000, dict(GATE, attack_ms=10.0))
    b = dsp._gate(np.ascontiguousarray(sig), 48000, dict(GATE, attack_ms=10.0))
    assert np.array_equal(a, b)                        # deterministic
    assert dsp._env_block_s(10.0, 100.0) == dsp.ENV_BLOCK_S


def test_gate_opening_has_no_step_discontinuity():
    """The gain curve is interpolated between block centres, not held: the output's
    own sample-to-sample slew must not exceed the source's."""
    sr = 48000
    sig = _burst(sr)
    out = dsp._gate(np.ascontiguousarray(sig), sr, dict(GATE, attack_ms=1.0))
    lo, hi = sr // 2 - 960, sr // 2 + 960          # +/-20 ms around the opening
    src_slew = float(np.abs(np.diff(sig[0, sr // 2: sr // 2 + 960])).max())
    out_slew = float(np.abs(np.diff(out[0, lo:hi])).max())
    assert out_slew <= src_slew * 1.05, (
        f"gate opening injects a step ({out_slew:.5f} vs source {src_slew:.5f}) — "
        "the block gain is being zero-order held again")
