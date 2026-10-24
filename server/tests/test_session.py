"""Session-layer tests: persistence across restarts, guardrails, undo ordering."""
import pydantic
import pytest

from mix_engine.models import ProposalRequest
from mix_engine.session import Session


def gain_request(stem_id: str, delta_db: float) -> ProposalRequest:
    return ProposalRequest(
        rationale=f"{delta_db:+.1f} dB on {stem_id}",
        predicted_impact=[{"metric": f"stem({stem_id}).lufs_s",
                           "predicted_delta": delta_db, "unit": "LUFS"}],
        actions=[{"type": "gain", "stem_id": stem_id, "delta_db": delta_db}],
    )


def test_state_survives_restart(demo_project):
    session = Session(demo_project)
    result = session.propose(gain_request("stem_a", 2.0))
    assert result.accepted
    assert session.apply(result.proposal_id).ok

    reloaded = Session(demo_project)
    assert reloaded.stems["stem_a"]["gain_db"] == 2.0
    (proposal,) = reloaded.get_proposals(status="applied")
    assert proposal.proposal_id == result.proposal_id


def test_gain_clamp_single_move(demo_project):
    session = Session(demo_project)
    session.user_edit("stem_a", {"gain_db": -12.0})  # the fader starts low: these moves test the gain WALLET, and a fader stops at +6 (test_fader_ceiling)
    result = session.propose(gain_request("stem_a", 7.0))
    assert not result.accepted
    (violation,) = result.violations
    assert violation.rule == "gain_per_stem_per_pass_db"
    assert violation.limit == 6.0
    assert violation.action_index == 0


def test_gain_clamp_is_cumulative_per_pass(demo_project):
    session = Session(demo_project)
    session.user_edit("stem_a", {"gain_db": -12.0})  # the fader starts low: these moves test the gain WALLET, and a fader stops at +6 (test_fader_ceiling)
    first = session.propose(gain_request("stem_a", 4.0))
    assert session.apply(first.proposal_id).ok

    second = session.propose(gain_request("stem_a", 4.0))
    assert not second.accepted
    assert second.violations[0].rule == "gain_per_stem_per_pass_db"

    # ...but the budget resets on the next pass.
    session.advance_pass()
    third = session.propose(gain_request("stem_a", 4.0))
    assert third.accepted


def test_apply_revalidates_against_current_state(demo_project):
    session = Session(demo_project)
    first = session.propose(gain_request("stem_a", 4.0))
    second = session.propose(gain_request("stem_a", 4.0))  # both fine at propose time
    assert first.accepted and second.accepted

    assert session.apply(first.proposal_id).ok
    blocked = session.apply(second.proposal_id)
    assert not blocked.ok
    assert "stem_a" in blocked.message


def test_eq_action_accepted_and_chained(demo_project):
    session = Session(demo_project)
    request = ProposalRequest(
        rationale="presence boost",
        predicted_impact=[],
        actions=[{"type": "eq", "stem_id": "stem_a",
                  "band": {"freq_hz": 3000, "gain_db": 2.0, "q": 1.0, "shape": "peak"}}],
    )
    result = session.propose(request)
    assert result.accepted
    assert session.apply(result.proposal_id).ok
    chain = session.applied_chain("stem_a")
    assert chain[1]["type"] == "eq" and chain[1]["band"]["freq_hz"] == 3000


def test_non_gain_action_on_master_rejected(demo_project):
    session = Session(demo_project)
    request = ProposalRequest(
        rationale="pan the master??",
        predicted_impact=[],
        actions=[{"type": "pan", "stem_id": "master", "pan": 0.5}],
    )
    result = session.propose(request)
    assert not result.accepted
    # Phase 6: master accepts gain/eq/comp/limiter; pan & co stay stems-only.
    assert result.violations[0].rule == "master_action_type"


def test_unknown_stem_rejected(demo_project):
    session = Session(demo_project)
    result = session.propose(gain_request("kazoo", 1.0))
    assert not result.accepted
    assert result.violations[0].rule == "unknown_stem"


def test_excluded_stem_refuses_everything_but_an_exclude(demo_project):
    """A stem OUT of the mix (a deleted DAW track's soft-excluded record, an agent
    exclude, a missing file) is a ghost: a move on it renders nowhere and the next
    registration of its name resets the record, taking the move with it (2026-09-04:
    the agent gated a deleted namesake's ghost and told the user it was done). Only
    an exclude action — the way back in — may target it."""
    session = Session(demo_project)
    session.user_edit("stem_a", {"excluded": True})
    result = Session(demo_project).propose(gain_request("stem_a", 1.0))
    assert not result.accepted
    assert result.violations[0].rule == "excluded_stem"
    assert "stem_a" in result.violations[0].message
    # a duck may not be keyed to it either
    duck = ProposalRequest(rationale="duck", predicted_impact=[], actions=[{
        "type": "duck", "stem_id": "stem_b", "key_stem_id": "stem_a",
        "threshold_db": -30.0, "depth_db": 6.0, "attack_ms": 5.0, "release_ms": 150.0}])
    result = Session(demo_project).propose(duck)
    assert not result.accepted and result.violations[0].rule == "duck_key_stem"
    # the way back in is allowed, and afterwards the stem takes moves again
    back = ProposalRequest(rationale="bring it back", predicted_impact=[],
                           actions=[{"type": "exclude", "stem_id": "stem_a", "excluded": False}])
    s2 = Session(demo_project)
    result = s2.propose(back)
    assert result.accepted, result.violations
    assert s2.apply(result.proposal_id).ok
    assert Session(demo_project).propose(gain_request("stem_a", 1.0)).accepted


def test_master_gain_action(demo_project):
    session = Session(demo_project)
    result = session.propose(gain_request("master", -1.5))
    assert result.accepted
    assert session.apply(result.proposal_id).ok
    assert session.master_gain_db == -1.5
    assert session.undo(result.proposal_id).ok
    assert session.master_gain_db == 0.0


def test_out_of_order_undo_restricted_per_stem(demo_project):
    session = Session(demo_project)
    first = session.propose(gain_request("stem_a", 1.0))
    second = session.propose(gain_request("stem_a", 1.0))
    other_stem = session.propose(gain_request("stem_b", 1.0))
    for proposal in (first, second, other_stem):
        assert session.apply(proposal.proposal_id).ok

    blocked = session.undo(first.proposal_id)
    assert not blocked.ok and second.proposal_id in blocked.message

    # A later proposal on a *different* stem doesn't block this stem's most recent.
    assert session.undo(second.proposal_id).ok
    assert session.undo(first.proposal_id).ok
    assert session.stems["stem_a"]["gain_db"] == 0.0


def test_insert_wet_proposals_do_not_share_a_fake_return_dependency(demo_project):
    """Two independent inserts may be undone independently.

    Insert-form wet actions carry ``return_id=""`` for schema symmetry. That
    empty value must never be recorded as a created return or a shared
    ``return:`` proposal target.
    """
    session = Session(demo_project)
    first = session.propose(ProposalRequest(
        rationale="vocal room", predicted_impact=[],
        actions=[{"type": "return_reverb", "stem_id": "stem_a",
                  "decay_s": 1.2, "pre_delay_ms": 0.0, "tone": 0.5, "mix": 0.3}]))
    second = session.propose(ProposalRequest(
        rationale="guitar echo", predicted_impact=[],
        actions=[{"type": "return_delay", "stem_id": "stem_b",
                  "delay_ms": 120.0, "feedback": 0.2, "tone": 0.5, "mix": 0.3}]))
    assert first.accepted and second.accepted
    assert session.apply(first.proposal_id).ok
    assert session.apply(second.proposal_id).ok

    records = {p["proposal_id"]: p for p in session._state["proposals"]}
    assert records[first.proposal_id]["created_returns"] == []
    assert records[second.proposal_id]["created_returns"] == []
    assert session.undo(first.proposal_id).ok
    assert not any(e["type"] == "return_reverb" for e in session.stems["stem_a"]["chain"])
    assert any(e["type"] == "return_delay" for e in session.stems["stem_b"]["chain"])


def test_proposals_per_pass_limit(demo_project):
    session = Session(demo_project)
    for stem_id in ("stem_a", "stem_b", "master"):
        assert session.propose(gain_request(stem_id, 1.0)).accepted
    fourth = session.propose(gain_request("stem_a", 1.0))
    assert not fourth.accepted
    assert fourth.violations[0].rule == "proposals_per_pass"


def test_actions_per_proposal_capped_at_schema(demo_project):
    with pytest.raises(pydantic.ValidationError):
        ProposalRequest(
            rationale="too many moves",
            predicted_impact=[],
            actions=[{"type": "gain", "stem_id": "stem_a", "delta_db": 0.1}] * 9,
        )


def test_apply_rolls_back_on_mid_proposal_failure(demo_project, monkeypatch):
    """A _mutate exception on action >=2 must restore every already-mutated
    target — a partial apply may never linger in memory (it would silently
    diverge from the persisted/undo record until the next _persist bakes it in)."""
    session = Session(demo_project)
    request = ProposalRequest(
        rationale="two gains, second one blows up", predicted_impact=[],
        actions=[{"type": "gain", "stem_id": "stem_a", "delta_db": 2.0},
                 {"type": "gain", "stem_id": "stem_b", "delta_db": 2.0}])
    result = session.propose(request)
    assert result.accepted

    real_mutate = Session._mutate
    calls = {"n": 0}

    def exploding_mutate(self, action):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return real_mutate(self, action)

    monkeypatch.setattr(Session, "_mutate", exploding_mutate)
    applied = session.apply(result.proposal_id)
    assert not applied.ok
    assert "rolled back" in applied.message

    # stem_a's first-action mutation was reverted; nothing persisted.
    assert session.stems["stem_a"]["gain_db"] == 0.0
    assert session.stems["stem_b"]["gain_db"] == 0.0
    reloaded = Session(demo_project)
    assert reloaded.stems["stem_a"]["gain_db"] == 0.0

    # The proposal survives and applies cleanly once _mutate behaves again.
    monkeypatch.setattr(Session, "_mutate", real_mutate)
    ok = session.apply(result.proposal_id)
    assert ok.ok, ok.message
    assert session.stems["stem_a"]["gain_db"] == 2.0
    assert session.stems["stem_b"]["gain_db"] == 2.0


def routing_request(actions: list[dict], rationale: str = "routing") -> ProposalRequest:
    return ProposalRequest(rationale=rationale, predicted_impact=[],
                           actions=actions)


def test_apply_rejects_stale_route_destination(demo_project):
    """2026-07-19 routing audit: a route's destination bus removed between
    propose and apply must fail cleanly — never land as a dangling reference
    the render silently drops."""
    session = Session(demo_project)
    grouped = session.propose(routing_request(
        [{"type": "group", "bus_id": "drums", "members": ["stem_a"]}]))
    assert grouped.accepted and session.apply(grouped.proposal_id).ok

    stale = session.propose(routing_request(
        [{"type": "route", "stem_id": "stem_b", "bus_id": "drums"}]))
    assert stale.accepted  # valid at propose time

    session.remove_bus("drums")  # direct user edit (shared control)
    blocked = session.apply(stale.proposal_id)
    assert not blocked.ok
    assert "drums" in blocked.message
    assert session.stems["stem_b"]["bus"] is None  # nothing dangling


def test_apply_rejects_stale_send_destination(demo_project):
    session = Session(demo_project)
    rid = session.add_return("reverb")["return_id"]

    stale = session.propose(routing_request(
        [{"type": "send", "stem_id": "stem_a", "return_id": rid,
          "gain_db": -12.0}]))
    assert stale.accepted

    session.remove_return(rid)  # direct user edit (shared control)
    blocked = session.apply(stale.proposal_id)
    assert not blocked.ok
    assert rid in blocked.message
    assert rid not in (session.stems["stem_a"].get("sends") or {})


def test_proposal_route_away_drops_emptied_bus_and_undo_restores(demo_project):
    """The proposal path must tidy emptied busses like route_stem/group_stems
    do (no orphan realizeBuses track) — and undo must restore the bus, not
    leave member stems pointing at a deleted one."""
    session = Session(demo_project)
    grouped = session.propose(routing_request(
        [{"type": "group", "bus_id": "drums",
          "members": ["stem_a", "stem_b"]}]))
    assert grouped.accepted and session.apply(grouped.proposal_id).ok

    away = session.propose(routing_request(
        [{"type": "route", "stem_id": "stem_a", "bus_id": None},
         {"type": "route", "stem_id": "stem_b", "bus_id": None}]))
    assert away.accepted and session.apply(away.proposal_id).ok
    assert "drums" not in session.buses  # emptied bus dropped

    assert session.undo(away.proposal_id).ok
    assert "drums" in session.buses  # bus restored with the stems
    assert session.stems["stem_a"]["bus"] == "drums"
    assert session.stems["stem_b"]["bus"] == "drums"


def test_bus_created_and_emptied_in_one_proposal_leaves_no_trace(demo_project):
    session = Session(demo_project)
    result = session.propose(routing_request(
        [{"type": "group", "bus_id": "ghost", "members": ["stem_a"]},
         {"type": "route", "stem_id": "stem_a", "bus_id": None}]))
    assert result.accepted and session.apply(result.proposal_id).ok
    assert "ghost" not in session.buses

    assert session.undo(result.proposal_id).ok  # nothing to restore, no crash
    assert "ghost" not in session.buses
    assert session.stems["stem_a"]["bus"] is None


def test_return_cap_applies_to_proposal_path(demo_project):
    """MAX_RETURNS (12, Ableton parity) must bind the agent's return_* proposal
    path the same as the direct add_return channel."""
    from mix_engine.session import MAX_RETURNS

    session = Session(demo_project)
    for _ in range(MAX_RETURNS):
        assert session.add_return("blank")["ok"]
    assert not session.add_return("blank")["ok"]  # direct path capped

    capped = session.propose(routing_request(
        [{"type": "return_reverb", "return_id": "shiny_new"}]))
    assert not capped.accepted
    assert capped.violations[0].rule == "return_limit"

    # Editing an EXISTING return stays allowed at the cap.
    existing = next(iter(session.returns))
    edit = session.propose(routing_request(
        [{"type": "return_reverb", "return_id": existing}]))
    assert edit.accepted


def test_missing_stem_wav_degrades_instead_of_bricking(demo_project):
    """A hand-deleted (or unmounted) stem wav used to raise FileNotFoundError on
    EVERY Session construction — one missing file silently bricked the whole
    project brain-side (2026-08-09 §2 audit P0). It must degrade: the stem sits
    out (excluded + missing), the rest keeps working, and the file coming back
    (remount / relink re-register) revives it."""
    seed = Session(demo_project)                       # persist a session that REFERENCES stem_a
    applied = seed.propose(gain_request("stem_a", 1.0))
    assert seed.apply(applied.proposal_id).ok

    wav = demo_project / "stem_a.wav"
    moved = demo_project / "stem_a.wav.gone"
    wav.rename(moved)

    session = Session(demo_project)                    # constructs — no raise
    assert session.stems["stem_a"]["excluded"] is True
    assert session.stems["stem_a"]["missing"] is True
    assert session.stems["stem_b"].get("excluded", False) is False   # the rest untouched

    moved.rename(wav)                                  # the file comes back
    revived = Session(demo_project)
    assert revived.stems["stem_a"].get("missing") is None or not revived.stems["stem_a"]["missing"]
    assert revived.stems["stem_a"]["excluded"] is False


def test_proposal_retention_trim(demo_project):
    """Scale/soak audit (2026-08-09): the proposals list keeps the last
    _MAX_PROPOSAL_RECORDS; ids never collide across a trim (monotonic counter,
    not list length), and trimmed APPLIED gain still counts toward the session
    gain budget via the trimmed_gain_totals carry."""
    from mix_engine.session import _MAX_PROPOSAL_RECORDS

    session = Session(demo_project)
    for _ in range(2):  # spend real budget first — these become the OLDEST records
        r = session.propose(gain_request("stem_a", 2.0))
        assert r.accepted and session.apply(r.proposal_id).ok
        session.advance_pass()
    assert session._cumulative_gain(None)["stem_a"] == pytest.approx(4.0)

    # Flood: guardrails reject most (proposal cap per pass) but every record appends.
    for _ in range(_MAX_PROPOSAL_RECORDS + 80):
        session.propose(gain_request("stem_b", 4.0))

    records = session._state["proposals"]
    # Trims run in batches (cap + 50 hysteresis), so the steady-state bound is the window top.
    assert len(records) <= _MAX_PROPOSAL_RECORDS + 50
    assert len(records) < _MAX_PROPOSAL_RECORDS + 82, "a trim must actually have run"
    ids = [r["proposal_id"] for r in records]
    assert len(set(ids)) == len(ids), "trim must not recycle proposal ids"

    reloaded = Session(demo_project)  # counter + carry survive persistence
    reloaded.advance_pass()           # the flood filled this pass's proposal cap
    fresh = reloaded.propose(gain_request("stem_b", 1.0))
    assert fresh.accepted
    ids2 = [r["proposal_id"] for r in reloaded._state["proposals"]]
    assert len(set(ids2)) == len(ids2)
    # The two applied stem_a records were trimmed away — their spend must persist.
    assert reloaded._cumulative_gain(None).get("stem_a", 0.0) == pytest.approx(4.0)
