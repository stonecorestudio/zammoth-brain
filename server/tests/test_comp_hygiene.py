"""2026-06 review fixes: compressor hygiene.

One whole-file comp per stem/bus (a new one replaces the old, like the
limiter); time-ranged comps coexist only on non-overlapping ranges; comp and
multiband makeup draw from the same gain budgets as gain moves; a squashed
render raises the `over_compression` flag; deep-threshold comps are never
"conservative" for auto-apply gating.
"""
import pytest

from mix_engine import analysis
from mix_engine.models import ProposalRequest
from mix_engine.session import Session

from orchestrator.loop import _is_conservative

from test_phase1_acceptance import call
from test_phase10_efficiency import in_memory_engine


def comp_request(stem_id: str, threshold_db: float = -20.0,
                 makeup_db: float = 0.0, time_range_s=None) -> ProposalRequest:
    action = {"type": "comp", "stem_id": stem_id,
              "threshold_db": threshold_db, "ratio": 2.0,
              "attack_ms": 10.0, "release_ms": 150.0, "makeup_db": makeup_db}
    if time_range_s is not None:
        action["time_range_s"] = time_range_s
    return ProposalRequest(rationale="comp move", predicted_impact=[],
                           actions=[action])


def gain_request(stem_id: str, delta_db: float) -> ProposalRequest:
    return ProposalRequest(
        rationale=f"{delta_db:+.1f} dB on {stem_id}", predicted_impact=[],
        actions=[{"type": "gain", "stem_id": stem_id, "delta_db": delta_db}])


def apply(session: Session, request: ProposalRequest) -> str:
    result = session.propose(request)
    assert result.accepted, result.violations
    assert session.apply(result.proposal_id).ok
    return result.proposal_id


def stem_comps(session: Session, stem_id: str) -> list[dict]:
    return [e for e in session.applied_chain(stem_id) if e["type"] == "comp"]


def test_whole_file_comp_replaces_existing(demo_project):
    session = Session(demo_project)
    apply(session, comp_request("stem_a", threshold_db=-20.0))
    session.advance_pass()
    apply(session, comp_request("stem_a", threshold_db=-30.0))

    (comp,) = stem_comps(session, "stem_a")
    assert comp["threshold_db"] == -30.0  # adjusted, not stacked


def test_ranged_comp_over_whole_file_comp_rejected(demo_project):
    session = Session(demo_project)
    apply(session, comp_request("stem_a"))
    session.advance_pass()

    result = session.propose(
        comp_request("stem_a", time_range_s=(0.0, 2.0)))
    assert not result.accepted
    assert result.violations[0].rule == "comp_stacking"


def test_non_overlapping_ranged_comps_coexist(demo_project):
    session = Session(demo_project)
    apply(session, comp_request("stem_a", time_range_s=(0.0, 2.0)))
    apply(session, comp_request("stem_a", time_range_s=(2.0, 4.0)))
    assert len(stem_comps(session, "stem_a")) == 2

    overlapping = session.propose(
        comp_request("stem_a", time_range_s=(1.0, 3.0)))
    assert not overlapping.accepted
    assert overlapping.violations[0].rule == "comp_stacking"

    whole_file = session.propose(comp_request("stem_a"))
    assert not whole_file.accepted
    assert whole_file.violations[0].rule == "comp_stacking"


def test_overlap_within_one_proposal_rejected(demo_project):
    session = Session(demo_project)
    request = ProposalRequest(
        rationale="two comps at once", predicted_impact=[],
        actions=[comp_request("stem_a", time_range_s=(0.0, 2.0)).actions[0],
                 comp_request("stem_a", time_range_s=(1.0, 3.0)).actions[0]])
    result = session.propose(request)
    assert not result.accepted
    assert result.violations[0].rule == "comp_stacking"
    assert result.violations[0].action_index == 1


def test_mb_comp_replaces_existing(demo_project):
    session = Session(demo_project)
    band = {"threshold_db": -20.0, "ratio": 2.0, "makeup_db": 0.0}

    def mb_request(attack_ms: float) -> ProposalRequest:
        return ProposalRequest(
            rationale="mb comp", predicted_impact=[],
            actions=[{"type": "mb_comp", "stem_id": "stem_a",
                      "low": band, "mid": band, "high": band,
                      "attack_ms": attack_ms, "release_ms": 150.0}])

    apply(session, mb_request(attack_ms=10.0))
    session.advance_pass()
    apply(session, mb_request(attack_ms=20.0))

    mb = [e for e in session.applied_chain("stem_a") if e["type"] == "mb_comp"]
    assert len(mb) == 1 and mb[0]["attack_ms"] == 20.0


def test_master_bus_comp_replaces_existing(demo_project):
    session = Session(demo_project)
    apply(session, comp_request("master", threshold_db=-12.0))
    session.advance_pass()
    apply(session, comp_request("master", threshold_db=-18.0))

    comps = [e for e in session.master_chain if e["type"] == "comp"]
    assert len(comps) == 1 and comps[0]["threshold_db"] == -18.0


def test_comp_makeup_charges_pass_gain_budget(demo_project):
    session = Session(demo_project)
    request = ProposalRequest(
        rationale="gain + makeup in one move", predicted_impact=[],
        actions=[{"type": "gain", "stem_id": "stem_a", "delta_db": 4.0},
                 comp_request("stem_a", makeup_db=4.0).actions[0]])
    result = session.propose(request)
    assert not result.accepted
    assert result.violations[0].rule == "gain_per_stem_per_pass_db"
    assert result.violations[0].action_index == 1


def test_comp_makeup_charges_session_gain_budget(demo_project):
    session = Session(demo_project)
    session.user_edit("stem_a", {"gain_db": -12.0})  # the fader starts low: these moves test the gain WALLET, and a fader stops at +6 (test_fader_ceiling)
    apply(session, gain_request("stem_a", 6.0))
    session.advance_pass()
    apply(session, gain_request("stem_a", 6.0))  # session budget now full
    session.advance_pass()

    over = session.propose(comp_request("stem_a", makeup_db=1.0))
    assert not over.accepted
    assert over.violations[0].rule == "gain_per_stem_per_session_db"

    # without makeup the comp itself is still fine
    assert session.propose(comp_request("stem_a")).accepted


def test_replacing_comp_refunds_its_makeup(demo_project):
    session = Session(demo_project)
    apply(session, comp_request("stem_a", makeup_db=6.0))
    session.advance_pass()
    # the replacement's makeup costs the *difference*, not a second 6 dB
    apply(session, comp_request("stem_a", threshold_db=-25.0, makeup_db=6.0))
    session.advance_pass()

    (comp,) = stem_comps(session, "stem_a")
    assert comp["makeup_db"] == 6.0
    # chain makeup (6) + this gain (6) lands exactly on the 12 dB session cap
    apply(session, gain_request("stem_a", 6.0))
    session.advance_pass()
    over = session.propose(gain_request("stem_a", 0.1))
    assert not over.accepted
    assert over.violations[0].rule == "gain_per_stem_per_session_db"


def test_undo_refunds_makeup_budget(demo_project):
    session = Session(demo_project)
    session.user_edit("stem_a", {"gain_db": -12.0})  # the fader starts low: these moves test the gain WALLET, and a fader stops at +6 (test_fader_ceiling)
    proposal_id = apply(session, comp_request("stem_a", makeup_db=6.0))
    session.advance_pass()
    assert session.undo(proposal_id).ok

    apply(session, gain_request("stem_a", 6.0))
    session.advance_pass()
    assert session.propose(gain_request("stem_a", 6.0)).accepted


# ---------- the over-compression "ears" (render flag + gate) ----------

def test_over_compressed_predicate():
    heavy = {"processor": "comp", "gr_mean_db": 7.2,
             "gr_max_db": 10.0, "gr_active_pct": 60.0}
    pinned = {"processor": "comp", "gr_mean_db": 3.5,
              "gr_max_db": 5.0, "gr_active_pct": 97.0}
    gentle_glue = {"processor": "comp", "gr_mean_db": 1.2,
                   "gr_max_db": 4.0, "gr_active_pct": 85.0}
    busy_limiter = {"processor": "limiter", "gr_mean_db": 8.0,
                    "gr_max_db": 12.0, "gr_active_pct": 100.0}
    assert analysis.over_compressed([heavy])
    assert analysis.over_compressed(None, {"stem_a": [pinned]})
    assert not analysis.over_compressed([busy_limiter, gentle_glue])
    assert not analysis.over_compressed(None, None)


def test_over_limited_predicate():
    slammed_limiter = {"processor": "limiter", "gr_mean_db": 4.2,
                       "gr_max_db": 8.0, "gr_active_pct": 80.0}
    peak_limiter = {"processor": "limiter", "gr_mean_db": 1.0,
                    "gr_max_db": 11.0, "gr_active_pct": 12.0}
    busy_but_open = {"processor": "limiter", "gr_mean_db": 5.0,
                     "gr_max_db": 12.0, "gr_active_pct": 100.0}
    assert analysis.over_limited([slammed_limiter], psr_db=5.8, plr_db=9.0)
    assert analysis.over_limited([peak_limiter], psr_db=None, plr_db=7.5)
    assert not analysis.over_limited([busy_but_open], psr_db=8.1, plr_db=12.0)
    assert not analysis.over_limited(None, psr_db=4.0, plr_db=6.0)


@pytest.mark.anyio
async def test_brutal_stem_comp_flags_the_render(demo_project):
    async with in_memory_engine(demo_project) as client:
        clean = await call(client, "render_and_analyze")
        assert "over_compression" not in clean["flags"]

        proposal = await call(client, "propose_actions", {"req": {
            "rationale": "crush it", "predicted_impact": [],
            "actions": [{"type": "comp", "stem_id": "stem_a",
                         "threshold_db": -60.0, "ratio": 10.0,
                         "attack_ms": 1.0, "release_ms": 50.0,
                         "makeup_db": 0.0}]}})
        await call(client, "apply_proposal",
                   {"proposal_id": proposal["proposal_id"]})
        squashed = await call(client, "render_and_analyze")
        assert "over_compression" in squashed["flags"]


def test_deep_threshold_comp_is_not_conservative():
    comp = {"type": "comp", "stem_id": "stem_a", "threshold_db": -20.0,
            "ratio": 2.0, "attack_ms": 10.0, "release_ms": 150.0,
            "makeup_db": 0.0}
    assert _is_conservative(comp)
    assert not _is_conservative({**comp, "threshold_db": -50.0})
    assert not _is_conservative({**comp, "ratio": 8.0})


def test_gain_is_billed_to_the_pass_it_lands_in(demo_project):
    """A proposal held across advance_pass spends the pass it APPLIES in.

    `pass_number` was stamped once at propose time and never moved, but its
    only two readers are the per-pass wallets (_cumulative_gain and
    _accepted_proposals_this_pass). So a card left pending — ordinary in Ask
    mode — and applied a pass later was billed to a pass that was already
    over, and its gain counted against nobody: propose +5 dB in pass 1, apply
    in pass 2, and pass 2's wallet read empty, so a second +5 dB on the same
    stem sailed past the 6 dB/pass cap (2026-08-16 review).
    """
    session = Session(demo_project)
    held = session.propose(gain_request("stem_a", 5.0))
    assert held.accepted, held.violations
    session.advance_pass()
    assert session.apply(held.proposal_id).ok          # lands in pass 2

    assert session._cumulative_gain_this_pass().get("stem_a") == pytest.approx(5.0), (
        "the pass the gain landed in must carry its spend")
    second = session.propose(gain_request("stem_a", 5.0))
    assert not second.accepted, "10 dB on one stem in one pass beat the 6 dB cap"
