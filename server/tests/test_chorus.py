"""The chorus in the session, the agent's bound and the JUCE stencil it reads with (its LAW -
the two triangle-swept lines, the tone, the dry / wet curve - is test_chorus_law.py's; 2026-10-28).

What is tested here guards the ways the device could be wrong while still "working": the monitor
contract carrying every knob (a device missing from it is silent live), the agent's guardrail
biting, a proposal landing and rendering, MODE snapped on write, the chunked gather being the
unchunked one, and the Lagrange stencil being JUCE's at a fractional delay.
"""
import pathlib

import numpy as np
import pytest

from mix_engine import chorus_law, dsp
from mix_engine.guardrails import LIMITS
from mix_engine.models import ChorusAction, ProposalRequest
from mix_engine.monitor_chain import monitor_chain_params
from mix_engine.session import Session

SR = 48_000


def test_monitor_contract_emits_every_param():
    """The DAW's live monitor silently DROPS a device missing from the contract, so a chorus that
    renders offline but is inaudible live is the failure mode here - and an old entry reaches the
    native side already read through the law."""
    out = monitor_chain_params([{"type": "chorus", "stem_id": "a", "mode": 2, "stereo": 0.5, "mix": 0.4,
                                 "output_db": -2.0}])
    assert out["entries"] == [{"type": "chorus", "mode": 2, "stereo": 0.5, "mix": 0.4, "output_db": -2.0,
                               "enabled": True}]
    assert out["print_only"] == []
    old = monitor_chain_params([{"type": "chorus", "stem_id": "a", "rate_hz": 2.0, "voices": 1, "mix": 0.3}])
    assert old["entries"][0]["mode"] == 1 and "rate_hz" not in old["entries"][0]


# ------------------------------------------------------------------ the agent's bound

def test_guardrail_caps_the_agents_wet_blend(demo_project):
    session = Session(demo_project)
    cap = LIMITS["chorus_mix_max"]
    ok = session.propose(ProposalRequest(
        rationale="thicken the pad", predicted_impact=[],
        actions=[{"type": "chorus", "stem_id": "stem_a", "mix": cap}]))
    assert ok.accepted
    swirl = session.propose(ProposalRequest(
        rationale="drown it", predicted_impact=[],
        actions=[{"type": "chorus", "stem_id": "stem_a", "mix": cap + 0.3}]))
    assert not swirl.accepted
    assert any(v.rule == "chorus_mix_max" for v in swirl.violations)


def test_the_user_knob_goes_further_than_the_agent_may():
    """Guardrails bind the AGENT only - a fully wet chorus is a legitimate user choice."""
    assert ChorusAction(stem_id="a", mix=1.0).mix == 1.0
    assert LIMITS["chorus_mix_max"] < 1.0


def test_it_lands_in_the_chain_and_renders(demo_project):
    """The registration layers: proposable, applied, and audible in a real render."""
    session = Session(demo_project)
    before, _ = dsp.render_stem(session, "stem_a")
    res = session.propose(ProposalRequest(
        rationale="widen the synth", predicted_impact=[],
        actions=[{"type": "chorus", "stem_id": "stem_a", "mix": 0.3, "mode": "I+II", "output_db": -3.0}]))
    assert res.accepted, res.violations
    session.apply(res.proposal_id)
    entries = [e for e in session.applied_chain("stem_a") if e["type"] == "chorus"]
    assert len(entries) == 1 and entries[0]["mode"] == 2
    after, _ = dsp.render_stem(session, "stem_a")
    n = min(len(before), len(after))
    assert not np.allclose(before[:n], after[:n]), "the chorus made no difference to the render"


# ---------------------------------------------------------------------------------------
# The parity oracle: JUCE's own delay line, transcribed.
#
# Every other test here is blind to the ONE thing that decides whether the native monitor and
# this renderer are the same filter — the Lagrange3rd stencil shift in
# `juce::dsp::DelayLine::updateInternalVariables()`. They are blind to it because they read at
# a delay whose fractional part is zero (`test_zero_depth_is_a_plain_delay`) or because they
# only assert a statistical property, and the shifted and unshifted stencils agree EXACTLY at
# f == 0. So the check has to be a null test against the vendored source, at a delay whose
# fractional part is NOT zero.
#
# This transcribes `juce_DelayLine.h`/`.cpp` as written: a circular buffer whose write and read
# positions DECREMENT (so index+1 is the older sample), `setDelay` splitting into
# delayInt/delayFrac, `updateInternalVariables()`, and the four-tap cubic.
# ---------------------------------------------------------------------------------------

class _JuceLagrange3rdDelayLine:
    """A faithful transcription of juce::dsp::DelayLine<float, Lagrange3rd>, single channel."""

    def __init__(self, maximum_delay_in_samples: int):
        self.total = int(maximum_delay_in_samples) + 1
        self.buf = np.zeros(self.total, dtype=np.float64)
        self.write_pos = 0
        self.read_pos = 0
        self.delay_int = 0
        self.delay_frac = 0.0

    def set_delay(self, d: float) -> None:
        # juce_DelayLine.cpp setDelay(): clamp, then split into integer + fractional parts.
        d = float(np.clip(d, 0.0, self.total - 1))
        self.delay_int = int(np.floor(d))
        self.delay_frac = d - self.delay_int
        # updateInternalVariables(), the Lagrange3rd branch, verbatim.
        if self.delay_frac < 2.0 and self.delay_int >= 1:
            self.delay_frac += 1.0
            self.delay_int -= 1

    def push(self, sample: float) -> None:
        self.buf[self.write_pos] = sample
        self.write_pos = (self.write_pos + self.total - 1) % self.total

    def pop(self) -> float:
        i1 = self.read_pos + self.delay_int
        i2, i3, i4 = i1 + 1, i1 + 2, i1 + 3
        if i4 >= self.total:
            i1 %= self.total
            i2 %= self.total
            i3 %= self.total
            i4 %= self.total
        v1, v2, v3, v4 = self.buf[i1], self.buf[i2], self.buf[i3], self.buf[i4]
        f = self.delay_frac
        d1, d2, d3 = f - 1.0, f - 2.0, f - 3.0
        c1 = -d1 * d2 * d3 / 6.0
        c2 = d2 * d3 * 0.5
        c3 = -d1 * d3 * 0.5
        c4 = d1 * d2 / 6.0
        out = v1 * c1 + f * (v2 * c2 + v3 * c3 + v4 * c4)
        self.read_pos = (self.read_pos + self.total - 1) % self.total
        return out


def _juce_read(x: np.ndarray, delay: np.ndarray) -> np.ndarray:
    """Run `x` through the transcribed line, setting the delay per sample as applyChorus does."""
    line = _JuceLagrange3rdDelayLine(int(np.max(delay)) + 8)
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        line.push(x[i])
        line.set_delay(float(delay[i]))
        out[i] = line.pop()
    return out


@pytest.mark.parametrize("hz", [440.0, 3000.0, 9000.0])
def test_the_stencil_is_juces_stencil_at_a_fractional_delay(hz):
    """`_lagrange3_taps` must equal JUCE's Lagrange3rd delay line SAMPLE FOR SAMPLE.

    The delay here sweeps, so its fractional part is non-zero almost everywhere — which is the
    only regime that can tell the shifted stencil from the unshifted one. 9 kHz is the case
    that hurts: before the shift was mirrored, the two differed by -22.9 dB relative to the
    wet, while the depth-0 test that was supposed to cover this saw 0.0.
    """
    n = 4096
    t = np.arange(n, dtype=np.float64) / SR
    x = np.sin(2.0 * np.pi * hz * t) * 0.5
    # A real sweep: 18 ms centre, depth 0.3, 1 Hz — the export oracle's own chorus settings.
    d = 18.0 * (1.0 + 0.3 * np.sin(2.0 * np.pi * 1.0 * t)) / 1000.0 * SR

    ours = dsp._lagrange3_taps(x, d)
    juce = _juce_read(x, d)

    # Skip the first samples, where our "unfilled line reads silence" and JUCE's cleared
    # buffer agree only up to the ramp-in the transcription models differently.
    lead = int(np.max(d)) + 8
    err = np.max(np.abs(ours[lead:] - juce[lead:]))
    ref = np.max(np.abs(juce[lead:]))
    assert ref > 0.1, f"vacuous: the oracle produced no signal at {hz} Hz"
    assert err < 1e-9 * max(1.0, ref), (
        f"{hz} Hz: our stencil is not JUCE's — max error {err:.3e} "
        f"({20 * np.log10(max(err, 1e-300) / ref):.1f} dB relative to the wet)"
    )


def test_the_fractional_part_the_parity_test_rides_is_really_fractional():
    """Guard the guard: the test above only means something if `frac` is non-zero.

    A delay that happened to land on integers would make the test pass on either stencil —
    exactly how the depth-0 test came to certify a filter it could not distinguish.
    """
    n = 4096
    t = np.arange(n, dtype=np.float64) / SR
    d = 18.0 * (1.0 + 0.3 * np.sin(2.0 * np.pi * 1.0 * t)) / 1000.0 * SR
    frac = d - np.floor(d)
    near_integer = np.mean((frac < 1e-6) | (frac > 1 - 1e-6))
    assert near_integer < 0.01, (
        f"{near_integer:.1%} of the sweep sits on integer delays — the parity test would be "
        "blind to the stencil shift"
    )


def test_the_chunked_gather_is_the_unchunked_one():
    """chorus_law.apply renders in chunks to bound its temporaries. There is no feedback (the two
    first-order filters run after, over the whole column), so that must be sample-for-sample the
    same render - a chunk boundary that dropped or repeated the timeline would click every ~11 s."""
    rng = np.random.default_rng(11)
    x = rng.standard_normal((70_000, 2)) * 0.1
    e = {"type": "chorus", "mode": 2, "stereo": 0.7, "mix": 1.0}
    whole = chorus_law.apply(x, SR, e)
    src = pathlib.Path(chorus_law.__file__).read_text()
    fn = src[src.index("def apply("):src.index("#: what an old chorus's ride")]
    ns = dict(chorus_law.__dict__)
    exec(compile(fn.replace("chunk = 1 << 19", "chunk = 4096"), "chunked", "exec"), ns)
    diced = ns["apply"](x, SR, e)         # ~17 chunks incl. a ragged tail
    assert np.array_equal(whole, diced), (
        f"chunking changed the render (max {np.max(np.abs(whole - diced)):.3e})"
    )


@pytest.mark.parametrize("asked,want", [(0, 0), (1.4, 1), (1.6, 2), (2.5, 2), (-1.0, 0), (7.0, 2)])
def test_mode_is_snapped_on_write(demo_project, asked, want):
    """MODE is an index (I / II / I+II): a fractional value is rounded as chorus_law.mode_index and the
    native parse round it (half to even), never persisted between two modes."""
    s = Session(demo_project)
    stem = next(iter(s._state["stems"]))
    s.add_stem_device(stem, "chorus")
    got = s.set_master_device_param("chorus", 0, "mode", float(asked), target=stem)
    assert got["value"] == float(want), f"mode {asked} persisted as {got['value']}, want {want}"


@pytest.mark.parametrize("asked,want", [(1, 1), (2, 2), (3, 2), (5, 4), (6, 4), (7, 4), (8, 8), (9, 8)])
def test_oversample_snaps_DOWN_like_both_native_paths(demo_project, asked, want):
    """Both native paths floor (`>=8?8:>=4?4:>=2?2:1`). Snapping to NEAREST here meant a knob
    dragged to 7 read 7 on the dock, ran 4x in the monitor and the DAW bounce, and persisted 8
    for the offline re-render and the export oracle — three answers to one gesture."""
    s = Session(demo_project)
    stem = next(iter(s._state["stems"]))
    s.add_stem_device(stem, "saturation")
    got = s.set_master_device_param("saturation", 0, "oversample", float(asked), target=stem)
    assert got["value"] == float(want), f"oversample {asked} persisted as {got['value']}, want {want}"
