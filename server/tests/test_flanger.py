"""Flanger — the modulation family's second device (2026-09-06).

A flanger is a chorus with a RETURN PATH, and every test here is about that return. It is
what turns the comb's gentle dips into resonant teeth, and it is also what makes this the
first device in the engine whose output is not a function of its input alone: with feedback
the renderer has to walk a recursion, in blocks, where `_chorus` was one vectorised gather.

So the claims worth testing are:

  * the recursion is RIGHT — an impulse comes back as an echo train at multiples of the
    delay whose amplitudes are the feedback's powers, sign included;
  * the BLOCKING that makes that recursion affordable in numpy changes no sample (the
    oracle walks the same recurrence in fixed blocks of the global minimum delay; the engine
    walks it in blocks that track the sweep, and the two must null);
  * the LFO's phase is a function of the HOST TIMELINE, as it is for the whole family;
  * and the one behavioural trap that sits on the device's own defaults — dry + wet with
    NEGATIVE feedback is nearly a Schroeder allpass, so a half-and-half blend is inaudible.
    The face's TEETH readout exists to say so, which makes it a claim under test here.
"""
import numpy as np
import pytest

from mix_engine import dsp, reference_monitor
from mix_engine.guardrails import LIMITS
from mix_engine.models import FlangerAction, ProposalRequest
from mix_engine.monitor_chain import monitor_chain_params
from mix_engine.session import Session

SR = 48_000
N = SR // 2


def _tone(hz: float = 440.0, amp: float = 0.5, stereo: bool = True) -> np.ndarray:
    t = np.arange(N, dtype=np.float64) / SR
    x = amp * np.sin(2.0 * np.pi * hz * t)
    return np.repeat(x[:, np.newaxis], 2 if stereo else 1, axis=1)


def _entry(**kw) -> dict:
    base = {"type": "flanger", "stem_id": "a", "rate_hz": 0.5, "depth": 0.6,
            "delay_ms": 3.0, "feedback": 0.0, "mix": 1.0, "stereo_deg": 0.0}
    base.update(kw)
    return base


# ------------------------------------------------------------------ the return path

def test_zero_depth_zero_feedback_is_a_plain_delay():
    """3 ms at 48 k is 144 samples exactly, and the line starts EMPTY. The arithmetic leg:
    an off-by-one in the read lands here and nowhere else (it agrees at a zero fraction)."""
    x = _tone()
    out = dsp._flanger(x, SR, _entry(depth=0.0, feedback=0.0, mix=1.0))
    d = 144
    assert np.abs(out[d:, 0] - x[:-d, 0]).max() < 1e-9
    assert np.abs(out[: d - 4, 0]).max() < 1e-12, "the delay line did not start empty"


@pytest.mark.parametrize("fb", [0.6, -0.6, 0.9])
def test_the_return_path_is_an_echo_train_of_the_feedbacks_powers(fb):
    """The leg a device that merely added a second dry copy could not pass: with depth 0 the
    flanger is a static feedback comb, so an impulse must come back at every multiple of the
    delay, each one `feedback` times the last — carrying the SIGN."""
    x = np.zeros((N, 1))
    x[0, 0] = 1.0
    out = dsp._flanger(x, SR, _entry(depth=0.0, feedback=fb, mix=1.0))
    for k in range(1, 5):
        assert out[144 * k, 0] == pytest.approx(fb ** (k - 1), abs=1e-9), f"echo {k}"


def test_feedback_zero_leaves_no_echoes():
    """The discrimination guard for the leg above."""
    x = np.zeros((N, 1))
    x[0, 0] = 1.0
    out = dsp._flanger(x, SR, _entry(depth=0.0, feedback=0.0, mix=1.0))
    assert out[144, 0] == pytest.approx(1.0, abs=1e-9)
    assert abs(out[288, 0]) < 1e-12 and abs(out[432, 0]) < 1e-12


# ------------------------------------------------------------------ the blocking

@pytest.mark.parametrize("kw", [
    dict(rate_hz=0.5, depth=0.6, delay_ms=3.0, feedback=0.0, mix=0.5, stereo_deg=0.0),
    dict(rate_hz=0.37, depth=0.8, delay_ms=2.4, feedback=0.62, mix=0.5, stereo_deg=90.0),
    dict(rate_hz=1.3, depth=0.9, delay_ms=0.7, feedback=-0.55, mix=1.0, stereo_deg=45.0),
    dict(rate_hz=0.11, depth=0.0, delay_ms=5.0, feedback=0.4, mix=0.3, stereo_deg=0.0),
    dict(rate_hz=2.0, depth=1.0, delay_ms=20.0, feedback=0.9, mix=0.7, stereo_deg=180.0),
])
def test_reference_monitor_matches_the_engine(kw):
    """THE test for this device. The oracle walks the recurrence in fixed blocks of the
    sweep's global minimum delay; the engine walks it in blocks that track the sweep, with the
    tap indices precomputed and the range mask folded into the coefficients. Two different
    derivations of one recurrence — if they null, the engine's blocking is what is proven."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal((N, 2)) * 0.2
    a = dsp._flanger(x, SR, dict(kw, type="flanger", stem_id="a"))
    b = reference_monitor._flanger(x, SR, **kw)
    assert np.abs(a - b).max() < 1e-9


def test_the_blocks_do_not_depend_on_how_much_audio_is_handed_over():
    """Nothing in this device looks ahead, so the first M samples of an N-sample render must
    be the first M samples of an M-sample one — including across the engine's OUTER block
    boundary, which is where a mis-carried delay line would show."""
    rng = np.random.default_rng(11)
    x = rng.standard_normal(((1 << 19) + 5000, 1)) * 0.2
    kw = _entry(depth=0.7, feedback=0.7, delay_ms=2.0, mix=1.0)
    full = dsp._flanger(x, SR, kw)
    part = dsp._flanger(x[: (1 << 19) - 100], SR, kw)
    assert np.abs(full[: len(part)] - part).max() < 1e-12


# ------------------------------------------------------------------ the sweep

def test_the_lfo_rides_the_host_timeline():
    """Offsetting the audio in TIME must change the output — the family's contract, and what
    lets the native monitor land on the same sweep as this renderer."""
    x = _tone(stereo=False)
    at_zero = dsp._flanger(x, SR, _entry(depth=0.7, feedback=0.5))
    # the same audio, half a second later on the timeline: prepend silence, drop it after
    shifted = dsp._flanger(np.vstack([np.zeros((SR // 2, 1)), x]), SR,
                           _entry(depth=0.7, feedback=0.5))[SR // 2:]
    assert np.abs(at_zero - shifted).max() > 0.01


def test_stereo_offset_decorrelates_a_mono_source_and_zero_leaves_it_mono():
    x = _tone()
    wide = dsp._flanger(x, SR, _entry(depth=0.5, feedback=0.4, stereo_deg=180.0))
    mono = dsp._flanger(x, SR, _entry(depth=0.5, feedback=0.4, stereo_deg=0.0))
    assert np.abs(wide[2000:, 0] - wide[2000:, 1]).max() > 0.01
    assert np.abs(mono[:, 0] - mono[:, 1]).max() < 1e-12


def test_mix_zero_is_a_true_bypass():
    x = _tone()
    assert np.abs(dsp._flanger(x, SR, _entry(mix=0.0, feedback=0.7)) - x).max() < 1e-12


# ------------------------------------------------------------------ the comb itself

def _teeth_db(feedback: float, mix: float, delay_ms: float = 2.4) -> float:
    """Peak-to-trough of the rendered magnitude response, measured one sine at a time. With
    depth 0 the device is LTI, so this is the same number the face's TEETH cell computes
    analytically — measuring it through the RENDERER is what ties the two together."""
    levels = []
    for hz in (210.0, 280.0, 350.0, 420.0, 490.0, 560.0, 630.0, 700.0):
        x = _tone(hz, 0.4, stereo=False)
        y = dsp._flanger(x, SR, _entry(rate_hz=0.01, depth=0.0, delay_ms=delay_ms,
                                       feedback=feedback, mix=mix))
        rms = np.sqrt(np.mean(y[8000:, 0] ** 2)) / (0.4 / np.sqrt(2.0))
        levels.append(20.0 * np.log10(max(1e-6, rms)))
    return max(levels) - min(levels)


def test_positive_feedback_is_a_deep_comb_and_negative_at_half_mix_is_nearly_an_allpass():
    """The device's one real trap, and it sits on the defaults. Dry + wet with NEGATIVE
    feedback approaches a Schroeder allpass, whose magnitude is flat: at mix 0.5 the comb all
    but disappears while every knob still reads like it is working. Taking MIX up brings it
    back. If this test ever goes green in both directions, the face's "FLAT - RAISE MIX"
    warning has become a lie."""
    assert _teeth_db(0.62, 0.5) > 8.0
    assert _teeth_db(-0.62, 0.5) < 2.0
    assert _teeth_db(-0.62, 1.0) > 8.0


def test_no_feedback_at_half_mix_is_the_textbook_cancellation():
    assert _teeth_db(0.0, 0.5) > 30.0


# ------------------------------------------------------------------ registration + bounds

def test_monitor_contract_emits_every_param():
    """A device missing from the contract is silently dropped by the DAW's live monitor."""
    out = monitor_chain_params([_entry(rate_hz=0.4, depth=0.8, delay_ms=2.0,
                                       feedback=-0.5, mix=0.45, stereo_deg=90.0)])
    assert out["entries"] == [{
        "type": "flanger", "rate_hz": 0.4, "depth": 0.8, "delay_ms": 2.0,
        "feedback": -0.5, "mix": 0.45, "stereo_deg": 90.0, "enabled": True}]


def test_guardrails_cap_the_agents_blend_and_its_return_path(demo_project):
    session = Session(demo_project)
    ok = session.propose(ProposalRequest(
        rationale="move the guitar", predicted_impact=[],
        actions=[{"type": "flanger", "stem_id": "stem_a",
                  "mix": LIMITS["flanger_mix_max"],
                  "feedback": LIMITS["flanger_feedback_max"]}]))
    assert ok.accepted, ok.violations
    wet = session.propose(ProposalRequest(
        rationale="drown it", predicted_impact=[],
        actions=[{"type": "flanger", "stem_id": "stem_a",
                  "mix": LIMITS["flanger_mix_max"] + 0.3}]))
    assert not wet.accepted and any(v.rule == "flanger_mix_max" for v in wet.violations)
    # …and the one the chorus has no equivalent of: a comb on the edge of self-oscillation
    for fb in (LIMITS["flanger_feedback_max"] + 0.2, -(LIMITS["flanger_feedback_max"] + 0.2)):
        ring = session.propose(ProposalRequest(
            rationale="ring it", predicted_impact=[],
            actions=[{"type": "flanger", "stem_id": "stem_a", "feedback": fb, "mix": 0.4}]))
        assert not ring.accepted
        assert any(v.rule == "flanger_feedback_max" for v in ring.violations)


def test_the_user_knob_goes_further_than_the_agent_may():
    """Guardrails bind the AGENT only — a fully wet, hard-resonant flanger is a real choice."""
    assert FlangerAction(stem_id="a", mix=1.0, feedback=0.95).feedback == 0.95
    assert LIMITS["flanger_mix_max"] < 1.0 and LIMITS["flanger_feedback_max"] < 0.95


def test_it_lands_in_the_chain_and_renders(demo_project):
    """The registration layers: proposable, applied, and audible in a real render."""
    session = Session(demo_project)
    before, _ = dsp.render_stem(session, "stem_a")
    res = session.propose(ProposalRequest(
        rationale="give the guitar movement", predicted_impact=[],
        actions=[{"type": "flanger", "stem_id": "stem_a", "mix": 0.4,
                  "feedback": 0.5, "delay_ms": 2.5}]))
    assert res.accepted, res.violations
    session.apply(res.proposal_id)
    entries = [e for e in session.applied_chain("stem_a") if e["type"] == "flanger"]
    assert len(entries) == 1 and entries[0]["feedback"] == 0.5
    after, _ = dsp.render_stem(session, "stem_a")
    n = min(len(before), len(after))
    assert not np.allclose(before[:n], after[:n]), "the flanger made no difference to the render"
