"""Phase 65 step 3 (2026-10-24): a mix proposal says WHICH device it changes.

A channel may carry as many compressors (gates, utilities, ...) as the user likes (step 2). A
whole-file proposal of such a type REPLACES the device, so on a channel that holds two it could
not say which (rule instance_which). Here a proposal names one: `instance: N` is that type's N-th
entry on the target (the numbering of `comp#N` everywhere else), replaced where it stands, and
`instance: "new"` adds one beside the others. The ledger charges each instance by the entry it
made, so changing one of two utilities is charged that one's delta and nothing for the other.
"""
from __future__ import annotations

from mix_engine.models import ProposalRequest
from mix_engine.session import Session


def _req(*actions: dict) -> ProposalRequest:
    return ProposalRequest.model_validate({
        "rationale": "instance", "actions": list(actions),
        "predicted_impact": [{"metric": "stem.crest_db", "predicted_delta": -1.0, "unit": "dB"}]})


def _comp(stem: str, **kw) -> dict:
    return {"type": "comp", "stem_id": stem, "threshold_db": -20.0, "ratio": 3.0,
            "attack_ms": 10.0, "release_ms": 120.0, "makeup_db": 0.0, **kw}


def _util(stem: str, gain_db: float, **kw) -> dict:
    return {"type": "utility", "stem_id": stem, "gain_db": gain_db, **kw}


def _of(chain: list[dict], kind: str) -> list[dict]:
    return [e for e in chain if e["type"] == kind]


def _apply(s: Session, *actions: dict):
    res = s.propose(_req(*actions))
    assert res.accepted, res.violations
    assert s.apply(res.proposal_id).ok
    return res


def _stem(s: Session) -> str:
    return next(iter(s._state["stems"]))


def test_a_proposal_names_the_second_compressor_and_changes_it_alone(demo_project):
    s = Session(demo_project)
    stem = _stem(s)
    s.add_stem_device(stem, "comp")
    s.add_stem_device(stem, "comp")
    first, second = _of(s._state["stems"][stem]["chain"], "comp")
    first_before = dict(first)
    uid = second["uid"]
    _apply(s, _comp(stem, threshold_db=-9.0, instance=1))
    a, b = _of(s._state["stems"][stem]["chain"], "comp")
    assert a == first_before                       # the other one is untouched
    assert b["threshold_db"] == -9.0 and b["uid"] == uid
    assert "instance" not in b                     # the address is not a setting


def test_new_adds_a_gate_beside_the_one_there(demo_project):
    # (a compressor beside a compressor is refused since 2026-11-02: test_stacked_levelling.py)
    s = Session(demo_project)
    stem = _stem(s)
    s.add_stem_device(stem, "gate")
    _apply(s, {"type": "gate", "stem_id": stem, "threshold_db": -60.0, "instance": "new"})
    gates = _of(s._state["stems"][stem]["chain"], "gate")
    assert len(gates) == 2 and gates[1]["threshold_db"] == -60.0


def test_an_instance_the_channel_does_not_have_is_refused(demo_project):
    s = Session(demo_project)
    stem = _stem(s)
    s.add_stem_device(stem, "comp")
    res = s.propose(_req(_comp(stem, instance=3)))
    assert not res.accepted
    assert any(v.rule == "instance_missing" and "comp#3" in v.message for v in res.violations)


def test_an_instance_on_a_time_ranged_move_is_refused(demo_project):
    s = Session(demo_project)
    stem = _stem(s)
    s.add_stem_device(stem, "comp")
    res = s.propose(_req(_comp(stem, instance=0, time_range_s=[1.0, 2.0])))
    assert not res.accepted
    assert any(v.rule == "instance_ranged" for v in res.violations)


def test_the_refusal_for_two_names_the_way_to_say_which(demo_project):
    s = Session(demo_project)
    stem = _stem(s)
    s.add_stem_device(stem, "comp")
    s.add_stem_device(stem, "comp")
    res = s.propose(_req(_comp(stem)))
    which = [v for v in res.violations if v.rule == "instance_which"]
    assert which and "instance" in which[0].message


def test_changing_one_of_two_utilities_is_charged_its_own_delta(demo_project):
    s = Session(demo_project)
    stem = _stem(s)
    _apply(s, _util(stem, 3.0, instance="new"))
    _apply(s, _util(stem, 3.0, instance="new"))      # +6 this pass: the rail
    assert len(_of(s._state["stems"][stem]["chain"], "utility")) == 2
    # The first one to +6 lifts the channel 3 dB more: +9 this pass.
    res = s.propose(_req(_util(stem, 6.0, instance=0)))
    assert not res.accepted
    assert any(v.rule == "gain_per_stem_per_pass_db" for v in res.violations), res.violations
    # ...and the same value again moves nothing.
    res = s.propose(_req(_util(stem, 3.0, instance=1)))
    assert res.accepted, res.violations


def test_changing_one_of_two_compressors_keeps_the_others_makeup_in_the_session(demo_project):
    s = Session(demo_project)
    stem = _stem(s)
    s.add_stem_device(stem, "comp")                 # the user's two (the engineer may not stack them)
    s.add_stem_device(stem, "comp")
    _apply(s, _comp(stem, makeup_db=3.0, instance=0))
    _apply(s, _comp(stem, makeup_db=3.0, instance=1))
    s.advance_pass()
    _apply(s, _util(stem, 6.0, instance="new"))      # 12 dB spent this session
    s.advance_pass()
    # comp#0's makeup 3 -> 4 is a +1: 13 dB in the session, over its 12.
    res = s.propose(_req(_comp(stem, makeup_db=4.0, instance=0)))
    assert not res.accepted
    assert any(v.rule == "gain_per_stem_per_session_db" for v in res.violations), res.violations


def test_control_one_utility_still_replaces_without_an_instance(demo_project):
    s = Session(demo_project)
    stem = _stem(s)
    _apply(s, _util(stem, 2.0))
    _apply(s, _util(stem, 5.0))
    utils = _of(s._state["stems"][stem]["chain"], "utility")
    assert len(utils) == 1 and utils[0]["gain_db"] == 5.0


def test_the_composed_eq_rail_holds_each_overdrives_tone(demo_project):
    # Two bright overdrives (+6 dB of shelf each at tone 1, under the pass rail only together
    # at tone 0.7), then the second one changed: the first one's shelf still stands.
    s = Session(demo_project)
    stem = _stem(s)
    od = {"type": "overdrive", "stem_id": stem, "drive_db": 6.0, "tone": 0.7}
    _apply(s, {**od, "instance": "new"})
    _apply(s, {**od, "instance": "new"})
    slots = s._eq_ledger()["singletons"][stem]
    assert len(slots) == 2 and all(k.startswith("overdrive@") for k in slots)
    # Re-tuning the second to the same tone composes nothing new: accepted.
    res = s.propose(_req({**od, "instance": 1}))
    assert res.accepted, res.violations
    # Making the second fully bright adds on top of the first's shelf, still standing.
    from mix_engine.guardrails import _ledger_curves
    led = s._eq_ledger()
    (_, _), (boost, _) = _ledger_curves(led, stem, {next(iter(slots))})
    assert boost.max() > 1.0          # the other instance's shelf is still counted


def test_every_stackable_type_is_addressed_by_its_occurrence_at_the_param_door(demo_project):
    # `<type>#N.<field>` in the brain: the co-edit door reaches the N-th entry and no other.
    from mix_engine.guardrails import INSTANCE_KINDS
    s = Session(demo_project)
    stem = _stem(s)
    for kind in INSTANCE_KINDS:
        s.add_stem_device(stem, kind)
        s.add_stem_device(stem, kind)
        param, (path, lo, hi) = next(iter(Session.DEVICE_PARAMS[kind].items()))
        first, second = _of(s._state["stems"][stem]["chain"], kind)[:2]
        before, second_before = json_copy(first), json_copy(second)
        value = lo + 0.37 * (hi - lo)
        s.set_master_device_param(kind, 1, param, value, target=stem)
        first, second = _of(s._state["stems"][stem]["chain"], kind)[:2]
        assert first == before, kind
        assert second != second_before, (kind, param)


def json_copy(d):
    import json
    return json.loads(json.dumps(d))
