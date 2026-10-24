"""The composed static-EQ rail (design call 2026-09-11, docs/10-daw.md).

The per-band rail (`eq_boost_max_db`) saw one band at a time, so eight legal +6 dB
bells at 2.5 kHz landed +21 dB (2026-09-10 agent audit, 05·F13). The bound now binds
the agent's COMPOSED eq / tilt / match_eq response per target: +6 / -12 dB per pass,
+12 / -24 dB per session, from the ledger, refunded by undo.
"""
from __future__ import annotations

from mix_engine.guardrails import validate_proposal
from mix_engine.models import EqAction, EqBand, MatchEqAction, ProposalRequest
from mix_engine.session import Session


def _bell(target, gain_db, freq_hz=2500.0, q=1.0, shape="peak", **kw):
    return EqAction(stem_id=target, band=EqBand(shape=shape, freq_hz=freq_hz,
                                                gain_db=gain_db, q=q, **kw))


def _req(*actions):
    return ProposalRequest(rationale="x", predicted_impact=[], actions=list(actions))


def _rules(req, ledger=None):
    return {v.rule for v in validate_proposal(req, {}, eq_ledger=ledger)}


EMPTY = {"pass": {}, "session": {}, "trimmed": {}}


def test_eight_stacked_bells_in_one_proposal_are_refused():
    req = _req(*[_bell("vox", 6.0) for _ in range(8)])
    assert "eq_boost_pass_db" in _rules(req, EMPTY)
    # the per-band rail alone passed it: that is the hole this closes
    assert "eq_boost_pass_db" not in _rules(req, None)


def test_one_legal_bell_passes_and_two_do_not():
    assert not {"eq_boost_pass_db", "eq_boost_session_db"} & _rules(_req(_bell("vox", 6.0)), EMPTY)
    assert "eq_boost_pass_db" in _rules(_req(_bell("vox", 6.0), _bell("vox", 6.0)), EMPTY)


def test_bells_far_apart_do_not_compose():
    req = _req(_bell("vox", 6.0, freq_hz=100.0, q=2.0), _bell("vox", 6.0, freq_hz=8000.0, q=2.0))
    assert "eq_boost_pass_db" not in _rules(req, EMPTY)


def test_stacked_cuts_are_bounded_but_filters_are_not_cuts():
    assert "eq_cut_pass_db" in _rules(_req(_bell("vox", -8.0), _bell("vox", -8.0)), EMPTY)
    hpfs = _req(*[_bell("vox", 0.0, freq_hz=100.0, q=0.707, shape="hpf", slope_db_oct=24)
                  for _ in range(3)])
    assert not {"eq_cut_pass_db", "eq_cut_session_db"} & _rules(hpfs, EMPTY)


def test_a_corrective_move_on_a_target_already_over_is_allowed():
    over = [_bell("vox", 6.0).model_dump(), _bell("vox", 6.0).model_dump()]
    ledger = {"pass": {"vox": over}, "session": {"vox": over}, "trimmed": {}}
    assert not {"eq_boost_pass_db", "eq_boost_session_db"} & _rules(_req(_bell("vox", -3.0)), ledger)
    assert "eq_boost_pass_db" in _rules(_req(_bell("vox", 2.0)), ledger)


def test_match_eq_composes_with_a_bell():
    m = MatchEqAction(stem_id="master", band_gains_db={"high_mid": 6.0}).model_dump()
    ledger = {"pass": {"master": [m]}, "session": {"master": [m]}, "trimmed": {}}
    assert "eq_boost_pass_db" in _rules(_req(_bell("master", 4.0, freq_hz=1414.0)), ledger)


def _stem(project):
    return next(iter(Session(project).stems))


def _propose_apply(project, req):
    s = Session(project)
    res = s.propose(req)
    if not res.accepted:
        return None, {v.rule for v in res.violations}
    assert s.apply(res.proposal_id).ok
    return res.proposal_id, set()


def test_the_session_cap_binds_across_passes_and_undo_refunds(demo_project):
    sid = _stem(demo_project)
    first, _ = _propose_apply(demo_project, _req(_bell(sid, 6.0)))
    assert first
    _, rules = _propose_apply(demo_project, _req(_bell(sid, 6.0)))
    assert "eq_boost_pass_db" in rules, "a second +6 in the same pass"
    Session(demo_project).advance_pass()
    second, _ = _propose_apply(demo_project, _req(_bell(sid, 6.0)))
    assert second, "a new pass may add +6 more, up to the +12 session rail"
    Session(demo_project).advance_pass()
    _, rules = _propose_apply(demo_project, _req(_bell(sid, 6.0)))
    assert "eq_boost_session_db" in rules
    assert Session(demo_project).undo(second).ok   # undo is newest-first per target
    again, _ = _propose_apply(demo_project, _req(_bell(sid, 6.0)))
    assert again, "undo refunds the session budget"


def test_trimmed_records_keep_counting(demo_project):
    sid = _stem(demo_project)
    s = Session(demo_project)
    s._state["trimmed_eq_curves_db"] = {sid: {"boost": [10.0] * 512, "cut": [0.0] * 512}}
    s._persist()
    _, rules = _propose_apply(demo_project, _req(_bell(sid, 4.0)))
    assert "eq_boost_session_db" in rules


def test_get_state_reports_the_room_the_rail_leaves(demo_project):
    sid = _stem(demo_project)
    assert _propose_apply(demo_project, _req(_bell(sid, 6.0)))[0]
    room = Session(demo_project).get_state().eq_headroom_db[sid]
    assert room["presence"]["boost"] < 0.5, "a +6 bell at 2.5 kHz spent this pass's presence room"
    assert room["sub"]["boost"] >= 5.9 and room["sub"]["cut"] <= -11.9   # rounded toward zero
    # a move sized to the reported room clears the rail
    Session(demo_project).advance_pass()
    room = Session(demo_project).get_state().eq_headroom_db[sid]
    assert _propose_apply(demo_project, _req(_bell(sid, room["presence"]["boost"])))[0]


# ------------------------------------------------ skirt slack (audit follow-ups 2026-09-11, item 2)

def _session_ledger(target, *actions):
    return {"pass": {}, "session": {target: [a.model_dump() for a in actions]}, "trimmed": {}}


def test_a_distant_bells_skirt_is_not_refused_for_a_band_at_the_rail():
    """A +6 dB q=0.707 bell at 8 kHz leaks 0.16 dB onto 1 kHz. With 1 kHz already at the +12
    session rail, the whole move was refused for a tail nobody can hear."""
    full_at_1k = _session_ledger("vox", _bell("vox", 6.0, freq_hz=1000.0),
                                 _bell("vox", 6.0, freq_hz=1000.0))
    air = _req(_bell("vox", 6.0, freq_hz=8000.0, q=0.707))
    assert "eq_boost_session_db" not in _rules(air, full_at_1k)


def test_small_pushes_creep_at_most_a_decibel_past_the_rail():
    at_200 = [_bell("vox", 6.0, freq_hz=200.0, q=2.0)] * 2          # +12 at 200 Hz
    small = _bell("vox", 0.4, freq_hz=200.0, q=2.0)
    assert "eq_boost_session_db" not in _rules(_req(small), _session_ledger("vox", *at_200))
    assert "eq_boost_session_db" not in _rules(_req(small), _session_ledger("vox", *at_200, small))
    assert "eq_boost_session_db" in _rules(_req(small),
                                           _session_ledger("vox", *at_200, small, small)), \
        "+13.2 dB: the slack is one decibel, not a hole"


def test_a_material_push_still_meets_the_exact_rail():
    at_200 = [_bell("vox", 6.0, freq_hz=200.0, q=2.0)] * 2
    assert "eq_boost_session_db" in _rules(_req(_bell("vox", 0.6, freq_hz=200.0, q=2.0)),
                                           _session_ledger("vox", *at_200))
    cut_floor = [_bell("vox", -12.0, freq_hz=200.0, q=2.0)] * 2       # -24 at 200 Hz
    assert "eq_cut_session_db" in _rules(_req(_bell("vox", -0.6, freq_hz=200.0, q=2.0)),
                                         _session_ledger("vox", *cut_floor))


def test_cut_skirts_get_the_same_slack():
    cut_floor = [_bell("vox", -12.0, freq_hz=200.0, q=2.0)] * 2       # -24 at 200 Hz
    assert "eq_cut_session_db" not in _rules(_req(_bell("vox", -0.4, freq_hz=200.0, q=2.0)),
                                             _session_ledger("vox", *cut_floor))
