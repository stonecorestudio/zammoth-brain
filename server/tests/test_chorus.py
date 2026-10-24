"""Chorus / ensemble — the modulation family's first device (2026-09-06).

The claims worth testing are NOT "it changes the signal". A chorus is a delay swept by an
LFO, and the two things that decide whether it is a chorus rather than a wobble are:

  * the SWEEP is real (the wet is pitch-modulated, so a steady tone comes back with its
    frequency moving), and
  * the LFO's phase is a function of the HOST TIMELINE, so the same sample of the same take
    always sees the same point of the sweep. That is what makes a render reproducible and
    what lets the native monitor land on the same sweep as this renderer.

Everything else here guards a specific way the device could be wrong while still "working":
the stereo offset actually decorrelating, voices changing texture and not level, mix 0 being
a true bypass, and the agent's guardrail biting.
"""
import pathlib

import numpy as np
import pytest

from mix_engine import dsp, reference_monitor
from mix_engine.guardrails import LIMITS
from mix_engine.models import ChorusAction, ProposalRequest
from mix_engine.monitor_chain import monitor_chain_params
from mix_engine.session import Session

SR = 48_000
N = SR // 2


def _tone(hz: float = 440.0, amp: float = 0.5, stereo: bool = True) -> np.ndarray:
    t = np.arange(N, dtype=np.float64) / SR
    x = amp * np.sin(2.0 * np.pi * hz * t)
    return np.repeat(x[:, np.newaxis], 2 if stereo else 1, axis=1)


def _entry(**kw) -> dict:
    base = {"type": "chorus", "stem_id": "a", "rate_hz": 1.0, "depth": 0.3,
            "delay_ms": 20.0, "mix": 1.0, "voices": 1, "stereo_deg": 0.0}
    base.update(kw)
    return base


def _zero_crossing_rate(x: np.ndarray) -> float:
    """A phase-robust proxy for instantaneous frequency."""
    return float(np.mean(np.abs(np.diff(np.signbit(x).astype(int)))))


# ------------------------------------------------------------------ the sweep is real

def test_the_delay_is_actually_swept():
    """The wet must be pitch-modulated. A delay that did not move would return the tone
    unchanged in frequency — which is the failure this catches, and the one a plain
    'output differs from input' assertion would sail past."""
    wet = dsp._chorus(_tone(), SR, _entry(depth=0.6, rate_hz=2.0))[:, 0]
    # instantaneous frequency, early vs late in the sweep, well clear of the fill-in
    seg = SR // 20
    rates = [_zero_crossing_rate(wet[o:o + seg]) for o in range(SR // 4, SR // 4 + 5 * seg, seg)]
    assert max(rates) > min(rates) * 1.02, (
        f"the tone came back at a constant pitch ({rates}) - the delay is not sweeping")


def test_zero_depth_is_a_plain_delay():
    """depth 0 must be an EXACT delay of delay_ms - the property that makes the
    interpolation trustworthy at every other depth."""
    x = _tone()
    wet = dsp._chorus(x, SR, _entry(depth=0.0, delay_ms=20.0, mix=1.0))
    d = int(round(20.0 / 1000.0 * SR))     # 960 samples
    assert np.max(np.abs(wet[d:, 0] - x[:N - d, 0])) < 1e-12
    assert np.max(np.abs(wet[:d - 4, 0])) == 0.0, "the line should start empty, not repeat"


def test_the_lfo_rides_the_host_timeline_not_a_free_accumulator():
    """Rendering the same input twice must give the same samples, and rendering a LATER
    slice must give what the full render gave at that point. A free-running accumulator
    passes the first and fails the second - and it is the second that makes an export
    reproducible and lets the native monitor agree."""
    x = _tone()
    a = dsp._chorus(x, SR, _entry())
    b = dsp._chorus(x, SR, _entry())
    assert np.array_equal(a, b), "not deterministic"

    # THE PROPERTY IN THE DOCSTRING, actually tested (2026-09-06 review). The old body asserted
    # `abs(a[quarter, 0]) >= 0.0` — a tautology, conceded in its own comment as "a touch" — and
    # never rendered a slice at all, so the one claim that separates a host-timeline LFO from a
    # free-running accumulator went unmeasured. `_lagrange3_taps`/`_lfo_phase` take an absolute
    # `offset`, which is exactly the seam that makes the property expressible: render a later
    # window as its own chunk and it must equal the full render's samples there.
    quarter = SR // 4
    from mix_engine.dsp import _lfo_phase
    for start in (0, quarter, SR // 3):
        n = 2048
        ph_full = _lfo_phase(SR, SR, 0.8, 0.0)[start:start + n]
        ph_slice = _lfo_phase(n, SR, 0.8, 0.0, offset=start)
        assert np.array_equal(ph_full, ph_slice), (
            f"the LFO at sample {start} depends on where the render STARTED — that is a free "
            "accumulator, and it makes an export unreproducible"
        )

    slow = dsp._chorus(x, SR, _entry(rate_hz=0.05))
    assert not np.allclose(a[quarter:], slow[quarter:]), "rate_hz did nothing"


# ------------------------------------------------------------------ it is WIDE, not just detuned

def test_stereo_offset_decorrelates_a_mono_source():
    """The whole point on a mono synth. With stereo_deg 0 the channels stay identical;
    with 180 they must genuinely differ."""
    x = _tone()
    narrow = dsp._chorus(x, SR, _entry(stereo_deg=0.0))
    wide = dsp._chorus(x, SR, _entry(stereo_deg=180.0))
    assert np.max(np.abs(narrow[:, 0] - narrow[:, 1])) < 1e-12, "0 deg should stay mono"
    tail = slice(SR // 4, None)
    corr = float(np.corrcoef(wide[tail, 0], wide[tail, 1])[0, 1])
    assert corr < 0.9, f"stereo_deg 180 left the channels correlated ({corr:.3f})"


def test_voices_change_texture_not_level():
    """voices is an ensemble control. If it multiplied level instead, every agent that
    reached for 3 voices would also be making a gain move it never declared."""
    x = _tone()
    rms = [float(np.sqrt(np.mean(dsp._chorus(x, SR, _entry(voices=v))[SR // 4:, 0] ** 2)))
           for v in (1, 2, 3)]
    assert max(rms) < min(rms) * 2.0, f"voices moved the level: {rms}"
    one = dsp._chorus(x, SR, _entry(voices=1))
    three = dsp._chorus(x, SR, _entry(voices=3))
    assert not np.allclose(one[SR // 4:], three[SR // 4:]), "voices did nothing"


def test_mix_zero_is_a_true_bypass():
    x = _tone()
    assert np.array_equal(dsp._chorus(x, SR, _entry(mix=0.0)), x)


# ------------------------------------------------------------------ the parity triangle

@pytest.mark.parametrize("kw", [
    {}, {"voices": 3, "stereo_deg": 120.0}, {"depth": 0.9, "rate_hz": 5.0},
    {"delay_ms": 1.0, "mix": 0.3}, {"delay_ms": 50.0, "depth": 1.0},
])
def test_reference_monitor_matches_the_engine(kw):
    """reference_monitor is the export oracle and re-implements dsp on purpose; it has to
    track this device, including at the range edges where the delay clamps."""
    e = _entry(**kw)
    a = dsp._chorus(_tone(), SR, e)
    b = reference_monitor._chorus(_tone(), SR, e["rate_hz"], e["depth"], e["delay_ms"],
                                  e["mix"], e["voices"], e["stereo_deg"])
    assert np.allclose(a, b, atol=1e-9), f"oracle diverged for {kw}"


def test_monitor_contract_emits_every_param():
    """The DAW's live monitor silently DROPS a device missing from the contract, so a
    chorus that renders offline but is inaudible live is the failure mode here."""
    out = monitor_chain_params([_entry(voices=2, stereo_deg=45.0)])
    assert out["entries"] == [{
        "type": "chorus", "rate_hz": 1.0, "depth": 0.3, "delay_ms": 20.0,
        "mix": 1.0, "voices": 2, "stereo_deg": 45.0, "enabled": True}]
    assert out["print_only"] == []


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
        actions=[{"type": "chorus", "stem_id": "stem_a", "mix": 0.4,
                  "voices": 2, "stereo_deg": 120.0}]))
    assert res.accepted, res.violations
    session.apply(res.proposal_id)
    entries = [e for e in session.applied_chain("stem_a") if e["type"] == "chorus"]
    assert len(entries) == 1 and entries[0]["voices"] == 2
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


@pytest.mark.parametrize("voices", [1, 2, 3])
@pytest.mark.parametrize("deg", [60.0, 90.0, 120.0, 180.0])
def test_the_width_knob_never_collapses_to_mono(voices, deg):
    """Every (stereo_deg > 0, voices) pair must decorrelate the channels.

    The voices sit on a uniform lattice of spacing 1/voices, and a uniform lattice is INVARIANT
    under a shift of its own spacing — so a whole-cycle channel offset that landed on a multiple
    of 1/voices gave both channels the same set of phases and the sum came out bit-identical
    mono. Measured 0.000 decorrelation at (180, 2) — the TOP of the Width knob on the default
    voice count — and at (120, 3). The old `test_stereo_offset_decorrelates_a_mono_source` could
    not see it: it only ever ran voices = 1, the one count with no lattice to alias against.
    """
    x = _tone(stereo=True)
    x[:, 1] = x[:, 0]                       # a strictly mono source
    y = dsp._chorus(x, SR, _entry(depth=0.3, mix=1.0, voices=voices, stereo_deg=deg))
    l, r = y[:, 0], y[:, 1]
    decorr = 1.0 - float(np.corrcoef(l, r)[0, 1])
    assert decorr > 0.02, (
        f"{deg}deg x {voices} voices came back mono (decorrelation {decorr:.4f}) — the channel "
        "offset landed on a multiple of the voice spacing"
    )


def test_stereo_zero_is_still_mono():
    """The other end must NOT decorrelate — a guard on the fix above, which would otherwise be
    satisfied by an offset that ignores stereo_deg entirely."""
    x = _tone(stereo=True)
    x[:, 1] = x[:, 0]
    y = dsp._chorus(x, SR, _entry(depth=0.3, mix=1.0, voices=2, stereo_deg=0.0))
    assert np.allclose(y[:, 0], y[:, 1]), "stereo_deg 0 must leave a mono source mono"


def test_the_chunked_gather_is_the_unchunked_one():
    """_chorus renders in chunks to bound its temporaries (2894 -> 699 MB on a 5-minute stem).
    There is no feedback, so that must be sample-for-sample the same render — a chunk boundary
    that dropped or repeated the LFO's timeline would be a click at every ~11 s."""
    rng = np.random.default_rng(11)
    x = rng.standard_normal((70_000, 2)) * 0.1
    e = _entry(depth=0.4, mix=1.0, voices=3, stereo_deg=120.0)
    whole = dsp._chorus(x, SR, e)
    src = pathlib.Path(dsp.__file__).read_text()
    fn = src[src.index("def _chorus("):src.index("def _overdrive(")]
    ns = dict(dsp.__dict__)
    exec(compile(fn.replace("chunk = 1 << 19", "chunk = 4096"), "chunked", "exec"), ns)
    diced = ns["_chorus"](x, SR, e)         # ~17 chunks incl. a ragged tail
    assert np.array_equal(whole, diced), (
        f"chunking changed the render (max {np.max(np.abs(whole - diced)):.3e})"
    )


@pytest.mark.parametrize("asked,want", [(1, 1), (2.6, 2), (3.9, 3), (0.4, 1), (7.0, 3)])
def test_voices_is_snapped_on_write(demo_project, asked, want):
    """DeviceSpecs.h says "Voices is discrete, snapped on write" and nothing snapped it, so a
    fractional 2.6 persisted and the three engines disagreed: dsp and the native contract parse
    TRUNCATE to 2 while the native live-ride path ROUNDED to 3 — a sqrt(3)/sqrt(2) = 0.9 dB wet
    step between the monitor and the bounce."""
    s = Session(demo_project)
    stem = next(iter(s._state["stems"]))
    s.add_stem_device(stem, "chorus")
    got = s.set_master_device_param("chorus", 0, "voices", float(asked), target=stem)
    assert got["value"] == float(want), f"voices {asked} persisted as {got['value']}, want {want}"


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
