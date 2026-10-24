"""2026-07-19 grouping/duplication audit fixes (engine side).

- One id namespace, for real: a group bus may not take a return's id (existing
  or created earlier in the same proposal), and add_return mints ids clear of
  buses, stems and 'master' — a duplicate id across namespaces made set_send's
  source resolution ambiguous.
- set_send resolves PREFIXED sources ("bus:x"/"return:x" — the native track
  names) in their own namespace, so a legacy id collision can't silently write
  the send onto the wrong holder.
- reset_stem: a soft-deleted (excluded) stem re-registered under a reused track
  name starts fresh instead of resurrecting the dead track's chain + flag.
"""
import pytest

from mix_engine.models import ProposalRequest
from mix_engine.session import Session


def proposal(actions: list[dict], rationale: str = "test") -> ProposalRequest:
    return ProposalRequest(rationale=rationale, predicted_impact=[],
                           actions=actions)


def test_group_bus_id_may_not_collide_with_a_return(demo_project):
    session = Session(demo_project)
    assert session.add_return("blank")["return_id"] == "fx"

    result = session.propose(proposal(
        [{"type": "group", "bus_id": "fx", "members": ["stem_a"]}]))
    assert not result.accepted
    assert any(v.rule == "bus_name_taken" for v in result.violations)


def test_group_bus_id_may_not_collide_with_a_pending_return(demo_project):
    # the return is created EARLIER IN THE SAME proposal — still one namespace
    session = Session(demo_project)
    result = session.propose(proposal([
        {"type": "return_reverb", "return_id": "wash",
         "decay_s": 1.2, "pre_delay_ms": 10.0, "tone": 0.5},
        {"type": "group", "bus_id": "wash", "members": ["stem_a"]},
    ]))
    assert not result.accepted
    assert any(v.rule == "bus_name_taken" for v in result.violations)


def test_add_return_mints_ids_clear_of_buses(demo_project):
    session = Session(demo_project)
    result = session.propose(proposal(
        [{"type": "group", "bus_id": "fx", "members": ["stem_a"]}]))
    assert result.accepted, result.violations
    assert session.apply(result.proposal_id).ok

    r = session.add_return("blank")           # bare prefix "fx" is taken by the bus
    assert r["ok"] is True and r["return_id"] == "fx2"


def test_set_send_prefixed_source_resolves_in_its_own_namespace(demo_project):
    session = Session(demo_project)
    assert session.add_return("blank")["return_id"] == "fx"
    # simulate a LEGACY collision (pre-fix state): a bus with the return's id
    session._state["buses"]["fx"] = {"gain_db": 0.0, "chain": [], "sends": {}}
    session._persist()
    assert session.add_return("blank")["return_id"] == "fx2"

    # the native track names pin the namespace — no silent cross-write
    session.set_send("return:fx", "fx2", -6.0)
    assert session.returns["fx"]["sends"] == {"fx2": -6.0}
    assert session.buses["fx"]["sends"] == {}

    session.set_send("bus:fx", "fx2", -12.0)
    assert session.buses["fx"]["sends"] == {"fx2": -12.0}
    assert session.returns["fx"]["sends"] == {"fx2": -6.0}

    # a bare id keeps the legacy priority order (bus first)
    session.set_send("fx", "fx2", -18.0)
    assert session.buses["fx"]["sends"] == {"fx2": -18.0}

    with pytest.raises(ValueError):
        session.set_send("bus:nope", "fx2", -6.0)


def test_reset_stem_only_if_excluded(demo_project):
    session = Session(demo_project)
    session.stems["stem_a"]["chain"] = [{"type": "gain", "delta_db": -3.0}]
    session.stems["stem_a"]["excluded"] = True
    session._persist()

    assert session.reset_stem("stem_a", only_if_excluded=True) == {
        "ok": True, "reset": True}
    assert session.stems["stem_a"]["chain"] == []
    assert session.stems["stem_a"]["excluded"] is False
    assert session.stems["stem_a"]["path"]        # the wav reference survives

    # a live (non-excluded) stem keeps its chain — the guard protects the
    # legitimate re-register of the same logical track
    session.stems["stem_b"]["chain"] = [{"type": "gain", "delta_db": 1.0}]
    session._persist()
    assert session.reset_stem("stem_b", only_if_excluded=True) == {
        "ok": True, "reset": False}
    # (an entry also carries its identity, `uid`, since reader E2·R16 - not what this test is about)
    assert [{k: v for k, v in e.items() if k != "uid"}
            for e in session.stems["stem_b"]["chain"]] == [{"type": "gain", "delta_db": 1.0}]

    assert session.reset_stem("nope", only_if_excluded=True)["ok"] is False
