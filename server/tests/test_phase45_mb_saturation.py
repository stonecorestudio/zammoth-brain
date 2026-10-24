"""Phase 45: multiband saturation (Saturn-style) — 3 bands at movable crossovers, per-band
drive/output/mix(/curve), global dry/wet, + saturation-curve models. Engine + agent;
single-band saturation stays bit-exact (backward compatible)."""
import numpy as np
import pytest
from pydantic import ValidationError

from mix_engine import dsp
from mix_engine.models import ProposalRequest, SaturationAction
from mix_engine.session import Session


def _apply(session: Session, actions: list[dict]):
    result = session.propose(ProposalRequest(
        rationale="phase45", predicted_impact=[], actions=actions))
    assert result.accepted, result.violations
    assert session.apply(result.proposal_id).ok
    return result


def _mb_sat(stem_id: str, low_drive: float = 12.0, **over) -> dict:
    a = {"type": "saturation", "stem_id": stem_id, "curve": "warm",
         "low": {"drive_db": low_drive}, "mid": {"drive_db": 4.0, "mix": 0.5},
         "high": {"drive_db": 6.0, "curve": "hard"},
         "crossover_lo_hz": 120.0, "crossover_hi_hz": 2500.0, "mix": 0.8}
    a.update(over)
    return a


def test_single_band_saturation_bit_exact():
    sr = 48000
    a = (np.random.default_rng(0).standard_normal((sr, 2)) * 0.3)
    old = np.tanh(10 ** (3 / 20) * a) / 10 ** (3 / 20)
    new = dsp._saturate(a.copy(), {"drive_db": 3.0, "output_db": 0.0}, sr)
    assert np.allclose(old, new)


def test_single_band_saturation_dry_wet():
    """Single-band `mix` is a dry/wet blend (parallel saturation); 1.0 = fully wet, 0.0 = bypass."""
    sr = 48000
    a = (np.random.default_rng(3).standard_normal((sr, 2)) * 0.3)
    wet = dsp._saturate(a.copy(), {"drive_db": 9.0, "output_db": 0.0}, sr)            # default mix 1.0
    half = dsp._saturate(a.copy(), {"drive_db": 9.0, "output_db": 0.0, "mix": 0.5}, sr)
    assert np.allclose(half, wet * 0.5 + a * 0.5)
    dry = dsp._saturate(a.copy(), {"drive_db": 9.0, "output_db": 0.0, "mix": 0.0}, sr)
    assert np.allclose(dry, a)                                                        # mix 0 = bypass


def test_saturation_curves_differ():
    a = np.linspace(-1, 1, 512)[:, None]
    outs = {t: dsp._waveshape(a, 6.0, t) for t in dsp.SAT_TYPES}
    types = list(dsp.SAT_TYPES)
    for i in range(len(types)):                       # every model is a distinct curve
        for j in range(i + 1, len(types)):
            assert not np.allclose(outs[types[i]], outs[types[j]]), (types[i], types[j])


def test_mb_saturation_applies(demo_project):
    session = Session(demo_project)
    _apply(session, [_mb_sat("stem_a")])
    sat = [e for e in session.applied_chain("stem_a") if e["type"] == "saturation"][0]
    assert sat["low"]["drive_db"] == 12.0 and sat["high"]["curve"] == 3   # "hard" -> 3
    assert sat["crossover_lo_hz"] == 120.0 and sat["mix"] == 0.8


def test_mb_saturation_needs_all_three_bands():
    with pytest.raises(ValidationError):
        SaturationAction(stem_id="stem_a", low={"drive_db": 6.0}, mid={"drive_db": 6.0})


def test_mb_saturation_band_drive_capped(demo_project):
    session = Session(demo_project)
    result = session.propose(ProposalRequest(
        rationale="too hot", predicted_impact=[], actions=[_mb_sat("stem_a", low_drive=20.0)]))
    assert not result.accepted
    assert result.violations[0].rule == "saturation_drive_max_db"


def test_mb_saturation_renders():
    sr = 48000
    a = (np.random.default_rng(1).standard_normal((sr, 2)) * 0.3)
    entry = {"type": "saturation", "drive_db": 3.0, "curve": "tanh",
             "low": {"drive_db": 18.0, "output_db": 0.0, "mix": 1.0},
             "mid": {"drive_db": 0.0, "output_db": 0.0, "mix": 1.0},
             "high": {"drive_db": 0.0, "output_db": 0.0, "mix": 1.0},
             "crossover_lo_hz": 200.0, "crossover_hi_hz": 2000.0, "mix": 1.0}
    out = dsp._saturate(a.copy(), entry, sr)
    assert out.shape == a.shape and np.isfinite(out).all()
    assert not np.allclose(out, a)            # the saturator colored the signal


def test_mb_saturation_per_band_curve():
    """Each band can carry its OWN curve; a band without one inherits the global curve."""
    sr = 48000
    a = (np.random.default_rng(2).standard_normal((sr, 2)) * 0.3)
    base = {"type": "saturation", "drive_db": 3.0, "curve": "tanh",
            "low": {"drive_db": 18.0}, "mid": {"drive_db": 18.0}, "high": {"drive_db": 18.0},
            "crossover_lo_hz": 200.0, "crossover_hi_hz": 2000.0, "mix": 1.0}
    out_inherit = dsp._saturate(a.copy(), base, sr)                       # low inherits global tanh
    hard_low = {**base, "low": {"drive_db": 18.0, "curve": "hard"}}       # low overridden to hard
    out_hard = dsp._saturate(a.copy(), hard_low, sr)
    assert not np.allclose(out_inherit, out_hard)                         # the per-band curve changed the shaping


def test_sat_multiband_toggle(demo_project):
    """The native MB toggle (set_sat_multiband) seeds/clears the bands server-side."""
    session = Session(demo_project)
    _apply(session, [{"type": "saturation", "stem_id": "stem_a", "drive_db": 5.0}])
    sat = lambda: [e for e in session.applied_chain("stem_a") if e["type"] == "saturation"][0]
    assert "low" not in sat()
    session.set_sat_multiband(0, True, "stem_a")
    s = sat()
    assert s["low"]["drive_db"] == 5.0 and "crossover_lo_hz" in s   # seeded from flat drive
    session.set_sat_multiband(0, False, "stem_a")
    assert "low" not in sat() and "crossover_lo_hz" not in sat()    # cleared
