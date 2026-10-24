"""Phase 44: the agent can propose a TRUE M/S compressor (independent mid + side)
and the advanced limiter ballistics (release / lookahead / channel link / style).
Validates the model + guardrails + that the entries round-trip into the chain and
render (offline parity with the native engine)."""
import numpy as np

from mix_engine import dsp
from mix_engine.models import ProposalRequest
from mix_engine.session import Session


def _apply(session: Session, actions: list[dict]):
    result = session.propose(ProposalRequest(
        rationale="phase44", predicted_impact=[], actions=actions))
    assert result.accepted, result.violations
    assert session.apply(result.proposal_id).ok
    return result


def _ms_comp(stem_id: str, mid_makeup: float = 0.0) -> dict:
    return {"type": "comp", "stem_id": stem_id,
            "threshold_db": -18.0, "ratio": 2.0, "attack_ms": 10.0, "release_ms": 120.0,
            "mid":  {"threshold_db": -24.0, "ratio": 3.0, "attack_ms": 8.0,
                     "release_ms": 120.0, "makeup_db": mid_makeup},
            "side": {"threshold_db": -18.0, "ratio": 1.0, "attack_ms": 10.0,
                     "release_ms": 120.0}}


def test_ms_comp_applies_with_both_blocks(demo_project):
    session = Session(demo_project)
    _apply(session, [_ms_comp("stem_a")])
    comps = [e for e in session.applied_chain("stem_a") if e["type"] == "comp"]
    assert len(comps) == 1
    c = comps[0]
    assert c["channel"] == "mid"                  # auto-marked M/S (engine keys off channel)
    assert c["mid"]["ratio"] == 3.0 and c["side"]["ratio"] == 1.0


def test_ms_comp_requires_both_blocks(demo_project):
    """A `mid` block without a `side` (or vice versa) is a schema error."""
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ProposalRequest(rationale="x", predicted_impact=[], actions=[
            {"type": "comp", "stem_id": "stem_a", "threshold_db": -18.0, "ratio": 2.0,
             "attack_ms": 10.0, "release_ms": 120.0,
             "mid": {"threshold_db": -24.0, "ratio": 3.0, "attack_ms": 8.0,
                     "release_ms": 120.0}}])


def test_ms_comp_makeup_on_master_rejected(demo_project):
    """M/S makeup on the master must be caught (only the limiter pushes loudness)."""
    session = Session(demo_project)
    result = session.propose(ProposalRequest(
        rationale="bad", predicted_impact=[], actions=[_ms_comp("master", mid_makeup=4.0)]))
    assert not result.accepted
    assert result.violations[0].rule == "master_comp_makeup_max_db"


def test_advanced_limiter_round_trips(demo_project):
    session = Session(demo_project)
    # explicit release/lookahead win over the style preset
    _apply(session, [{"type": "limiter", "stem_id": "master", "ceiling_dbtp": -1.0,
                      "target_lufs_i": -10.0, "release_ms": 200.0, "lookahead_ms": 1.5,
                      "channel_link": 0.5, "style": "transparent"}])
    lim = [e for e in session.master_chain if e["type"] == "limiter"][0]
    assert lim["release_ms"] == 200.0 and lim["lookahead_ms"] == 1.5
    assert lim["channel_link"] == 0.5
    assert lim["style"] == 1                      # "transparent" name -> index 1


def test_limiter_style_is_a_knob_preset(demo_project):
    """Style with no explicit knobs fills release_ms / lookahead_ms from the preset."""
    session = Session(demo_project)
    _apply(session, [{"type": "limiter", "stem_id": "master",
                      "ceiling_dbtp": -1.0, "style": "punchy"}])
    lim = [e for e in session.master_chain if e["type"] == "limiter"][0]
    assert lim["release_ms"] == 20.0 and lim["lookahead_ms"] == 1.5   # punchy preset
    assert lim["style"] == 2


def test_ms_comp_renders_independently():
    """Offline render parity: the mid is compressed, the side passes through."""
    sr = 48000
    rng = np.random.default_rng(0)
    a = np.ascontiguousarray(np.stack([
        rng.standard_normal(sr).astype(np.float32) * 0.5,
        rng.standard_normal(sr).astype(np.float32) * 0.5]))
    entry = {"type": "comp", "channel": "mid",
             "threshold_db": -18.0, "ratio": 2.0, "attack_ms": 10.0,
             "release_ms": 120.0, "makeup_db": 0.0, "mix": 1.0,
             "mid":  {"threshold_db": -40.0, "ratio": 8.0, "attack_ms": 8.0,
                      "release_ms": 120.0, "makeup_db": 0.0, "mix": 1.0},
             "side": {"threshold_db": -18.0, "ratio": 1.0, "attack_ms": 10.0,
                      "release_ms": 120.0, "makeup_db": 0.0, "mix": 1.0}}
    out = dsp._apply_comp(a, sr, entry)

    def side_rms(x):
        return float(np.sqrt(np.mean(((x[0] - x[1]) * 0.5) ** 2)))

    def mid_rms(x):
        return float(np.sqrt(np.mean(((x[0] + x[1]) * 0.5) ** 2)))

    assert mid_rms(out) < mid_rms(a) * 0.5        # mid heavily compressed
    assert abs(side_rms(out) - side_rms(a)) < side_rms(a) * 0.02   # side untouched


# ---------- multiband comp: movable crossovers + per-band upward mode ----------

def _mb(stem_id: str, up: bool = False, makeup: float = 0.0) -> dict:
    band = {"threshold_db": -24.0, "ratio": 2.0, "makeup_db": makeup, "up": up}
    return {"type": "mb_comp", "stem_id": stem_id,
            "low": dict(band), "mid": dict(band), "high": dict(band),
            "attack_ms": 10.0, "release_ms": 150.0,
            "crossover_lo_hz": 120.0, "crossover_hi_hz": 2500.0}


def test_mb_comp_crossovers_apply(demo_project):
    session = Session(demo_project)
    _apply(session, [_mb("stem_a")])
    mb = [e for e in session.applied_chain("stem_a") if e["type"] == "mb_comp"][0]
    assert mb["crossover_lo_hz"] == 120.0 and mb["crossover_hi_hz"] == 2500.0
    assert mb["low"]["up"] is False


def test_mb_comp_up_on_a_stem_is_the_users_knob(demo_project):
    """An upward band lifts a quiet band by up to MB_MAX_BOOST_DB (12) at ANY ratio > 1 — past
    the agent's 6 dB/pass budget by itself (full audit 2026-10-02, D9·R2: ratio 2 used to be
    charged 6, accepted, and lifted 12). The `up` flag still rides the model for the user's
    co-edit; an agent proposal carrying it is refused, and says why."""
    session = Session(demo_project)
    result = session.propose(ProposalRequest(
        rationale="lift the quiet detail", predicted_impact=[], actions=[_mb("stem_a", up=True)]))
    assert not result.accepted
    hit = [v for v in result.violations if v.rule == "gain_per_stem_per_pass_db"]
    assert hit and hit[0].value == 12.0 and "upward" in hit[0].message


def test_mb_comp_up_on_master_rejected(demo_project):
    session = Session(demo_project)
    result = session.propose(ProposalRequest(
        rationale="bad", predicted_impact=[], actions=[_mb("master", up=True)]))
    assert not result.accepted
    assert any(v.rule == "master_comp_makeup_max_db" for v in result.violations)


def test_mb_comp_crossover_order_validated():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ProposalRequest(rationale="x", predicted_impact=[], actions=[{
            "type": "mb_comp", "stem_id": "stem_a",
            "low": {"threshold_db": -24.0, "ratio": 2.0},
            "mid": {"threshold_db": -24.0, "ratio": 2.0},
            "high": {"threshold_db": -24.0, "ratio": 2.0},
            "crossover_lo_hz": 3000.0, "crossover_hi_hz": 1000.0}])  # lo >= hi


def test_spectral_denoise_is_proposable(demo_project):
    session = Session(demo_project)
    _apply(session, [{"type": "spectral_denoise", "stem_id": "stem_a",
                      "reduction_db": 8.0, "sensitivity_db": 6.0}])
    assert any(e["type"] == "spectral_denoise"
               for e in session.applied_chain("stem_a"))
