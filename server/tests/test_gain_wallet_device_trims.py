"""Device output trims spend the same gain wallets as gain moves — across proposals.

2026-09-01 sweep, P1: guardrails charged eq / tilt / saturation / clip / transient /
overdrive / ott / utility trims to the per-proposal running total, but the wallets
that seed it (`Session._cumulative_gain`) counted only `gain` actions. So each
proposal was charged in isolation and three proposals of eight EQ bands at +0.75 dB
landed +18 dB of applied trim in ONE pass against a 6 dB/pass cap (and +108 dB over
six passes). The 2026-07-17 audit memo's "per-PASS cap is intact" was wrong.
Separately, `OttAction.in_gain_db` was charged to nothing: a free +12 dB.
"""
from __future__ import annotations

from mix_engine.guardrails import LIMITS, validate_proposal
from mix_engine.models import EqAction, EqBand, OttAction, ProposalRequest
from mix_engine.session import Session


def _eq_trims(stem: str, n: int = 8, output_db: float = 0.75) -> ProposalRequest:
    return ProposalRequest(rationale="lift", predicted_impact=[], actions=[
        EqAction(stem_id=stem, output_db=output_db,
                 band=EqBand(shape="peak", freq_hz=400.0 + 250.0 * i, gain_db=0.0, q=1.0))
        for i in range(n)])


def test_eq_output_trims_accumulate_across_proposals_in_one_pass(demo_project):
    session = Session(demo_project)
    stem = sorted(session.stems)[0]
    cap = LIMITS["gain_per_stem_per_pass_db"]
    assert 8 * 0.75 == cap, "fixture: one proposal spends exactly the whole wallet"

    first = session.propose(_eq_trims(stem))
    assert first.accepted, first.violations
    assert session.apply(first.proposal_id).ok

    second = session.propose(_eq_trims(stem))
    assert not second.accepted, "the second +6 dB of applied trim must exceed the per-pass wallet"
    assert "gain_per_stem_per_pass_db" in {v.rule for v in second.violations}


def test_applied_trims_are_refunded_by_undo(demo_project):
    session = Session(demo_project)
    stem = sorted(session.stems)[0]
    first = session.propose(_eq_trims(stem))
    session.apply(first.proposal_id)
    assert session.undo(first.proposal_id).ok
    again = session.propose(_eq_trims(stem))
    assert again.accepted, again.violations


def test_ott_input_gain_is_a_level_move():
    req = ProposalRequest(rationale="x", predicted_impact=[],
                          actions=[OttAction(stem_id="stem_a", depth=0.0, in_gain_db=12.0)])
    rules = {v.rule for v in validate_proposal(req, {})}
    assert "gain_per_stem_per_pass_db" in rules
