"""Phase 65 step 2 (2026-10-24; James: "it should be like ableton where you can add many"): the
compressor is the first device the brain holds as many of as the user likes.

The native chain runs every comp entry as its own instance (MonitorChain::kMaxComps = 128,
`--stacked-devices`). Here: the brain takes a second compressor at every add door, a chain stops at
128 devices, and the agent's whole-file comp proposal - which REPLACES the compressor - is refused
on a channel that carries two, rather than replacing both and charging one.
"""
from __future__ import annotations

import pytest

from mix_engine.models import ProposalRequest
from mix_engine.session import CHAIN_MAX_ENTRIES, MASTER, Session, _NATIVE_SINGLETON_KINDS


def _comps(chain: list[dict]) -> int:
    return sum(1 for e in chain if e["type"] == "comp")


def _comp_req(stem: str, **kw) -> ProposalRequest:
    action = {"type": "comp", "stem_id": stem, "threshold_db": -20.0, "ratio": 3.0,
              "attack_ms": 10.0, "release_ms": 120.0, "makeup_db": 0.0, **kw}
    return ProposalRequest.model_validate({
        "rationale": "comp", "actions": [action],
        "predicted_impact": [{"metric": "stem.crest_db", "predicted_delta": -1.0, "unit": "dB"}]})


def test_comp_is_no_longer_one_per_chain():
    assert "comp" not in _NATIVE_SINGLETON_KINDS


def test_the_brain_takes_a_second_compressor_on_a_track_and_on_the_master(demo_project):
    s = Session(demo_project)
    stem = next(iter(s._state["stems"]))
    s.add_stem_device(stem, "comp")
    s.add_stem_device(stem, "comp")
    assert _comps(s._state["stems"][stem]["chain"]) == 2
    s.add_master_device("comp")
    s.add_master_device("comp")
    assert _comps(s._state["master_chain"]) == 2


def test_a_chain_stops_at_its_device_count(demo_project):
    s = Session(demo_project)
    stem = next(iter(s._state["stems"]))
    while len(s._state["stems"][stem]["chain"]) < CHAIN_MAX_ENTRIES:
        s.add_stem_device(stem, "comp")
    with pytest.raises(ValueError, match="128 devices"):
        s.add_stem_device(stem, "comp")
    assert len(s._state["stems"][stem]["chain"]) == CHAIN_MAX_ENTRIES


def test_a_comp_proposal_on_a_channel_with_two_is_refused_and_says_how(demo_project):
    s = Session(demo_project)
    stem = next(iter(s._state["stems"]))
    s.add_stem_device(stem, "comp")
    s.add_stem_device(stem, "comp")
    res = s.propose(_comp_req(stem))
    assert not res.accepted
    which = [v for v in res.violations if v.rule == "instance_which"]
    assert which and "device.set" in which[0].message and "comp#1" in which[0].message
    assert _comps(s._state["stems"][stem]["chain"]) == 2


def test_control_a_comp_proposal_on_a_channel_with_one_still_replaces_it(demo_project):
    s = Session(demo_project)
    stem = next(iter(s._state["stems"]))
    s.add_stem_device(stem, "comp")
    res = s.propose(_comp_req(stem, threshold_db=-14.0))
    assert res.accepted, res.violations
    assert s.apply(res.proposal_id).ok
    chain = s._state["stems"][stem]["chain"]
    assert _comps(chain) == 1
    assert next(e for e in chain if e["type"] == "comp")["threshold_db"] == -14.0


def test_a_project_with_two_compressors_keeps_both_when_it_opens(demo_project):
    s = Session(demo_project)
    stem = next(iter(s._state["stems"]))
    s.add_stem_device(stem, "comp")
    s.add_stem_device(stem, "comp")
    again = Session(demo_project)
    assert _comps(again._state["stems"][stem]["chain"]) == 2


# ---- the second family: the gate, the transient shaper and the duck -------------------------

def test_gate_transient_and_duck_are_no_longer_one_per_chain():
    for kind in ("gate", "transient", "duck"):
        assert kind not in _NATIVE_SINGLETON_KINDS


def test_the_brain_takes_a_second_gate_transient_and_duck(demo_project):
    s = Session(demo_project)
    stem = next(iter(s._state["stems"]))
    for kind in ("gate", "transient", "duck"):
        s.add_stem_device(stem, kind)
        s.add_stem_device(stem, kind)
        assert sum(1 for e in s._state["stems"][stem]["chain"] if e["type"] == kind) == 2, kind


def test_a_trigger_pick_keys_every_duck_and_keeps_them_all(demo_project):
    s = Session(demo_project)
    stems = list(s._state["stems"])
    stem, key = stems[0], stems[1]
    s.add_stem_device(stem, "duck")
    s.add_stem_device(stem, "duck")
    ducks = [e for e in s._state["stems"][stem]["chain"] if e["type"] == "duck"]
    ducks[1]["mix"] = 0.45
    s.set_duck(stem, key)
    ducks = [e for e in s._state["stems"][stem]["chain"] if e["type"] == "duck"]
    assert len(ducks) == 2
    assert all(e["key_stem_id"] == key for e in ducks)
    assert ducks[1]["mix"] == 0.45               # its own tuning, kept
    s.set_duck(stem, "none")
    ducks = [e for e in s._state["stems"][stem]["chain"] if e["type"] == "duck"]
    assert len(ducks) == 2 and all(not e["key_stem_id"] for e in ducks)


def test_a_new_duck_listens_to_the_trigger_the_track_already_has(demo_project):
    s = Session(demo_project)
    stems = list(s._state["stems"])
    stem, key = stems[0], stems[1]
    s.add_stem_device(stem, "duck")
    s.set_duck(stem, key)
    s.add_stem_device(stem, "duck")
    ducks = [e for e in s._state["stems"][stem]["chain"] if e["type"] == "duck"]
    assert len(ducks) == 2 and ducks[1].get("key_stem_id") == key


def test_a_gate_proposal_on_a_channel_with_two_gates_is_refused(demo_project):
    s = Session(demo_project)
    stem = next(iter(s._state["stems"]))
    s.add_stem_device(stem, "gate")
    s.add_stem_device(stem, "gate")
    req = ProposalRequest.model_validate({
        "rationale": "gate", "actions": [{"type": "gate", "stem_id": stem, "threshold_db": -45.0}],
        "predicted_impact": [{"metric": "stem.crest_db", "predicted_delta": 1.0, "unit": "dB"}]})
    res = s.propose(req)
    assert not res.accepted
    which = [v for v in res.violations if v.rule == "instance_which"]
    assert which and "gate#1" in which[0].message


# ---- the third family: the limiter, the multiband compressor, OTT and the clipper ------------

def test_limiter_multiband_ott_and_clipper_are_no_longer_one_per_chain():
    for kind in ("limiter", "mb_comp", "ott", "clip"):
        assert kind not in _NATIVE_SINGLETON_KINDS


def test_the_brain_takes_a_second_of_the_third_family(demo_project):
    s = Session(demo_project)
    stem = next(iter(s._state["stems"]))
    for kind in ("limiter", "mb_comp", "ott", "clip"):
        s.add_stem_device(stem, kind)
        s.add_stem_device(stem, kind)
        assert sum(1 for e in s._state["stems"][stem]["chain"] if e["type"] == kind) == 2, kind


def test_the_master_takes_a_second_limiter(demo_project):
    s = Session(demo_project)
    before = sum(1 for e in s._state["master_chain"] if e["type"] == "limiter")
    s.add_master_device("limiter")
    s.add_master_device("limiter")
    assert sum(1 for e in s._state["master_chain"] if e["type"] == "limiter") == before + 2


def test_a_limiter_proposal_on_a_channel_with_two_limiters_is_refused(demo_project):
    s = Session(demo_project)
    stem = next(iter(s._state["stems"]))
    s.add_stem_device(stem, "limiter")
    s.add_stem_device(stem, "limiter")
    req = ProposalRequest.model_validate({
        "rationale": "limit", "actions": [{"type": "limiter", "stem_id": stem, "ceiling_dbtp": -3.0}],
        "predicted_impact": [{"metric": "stem.true_peak_dbtp", "predicted_delta": -1.0, "unit": "dB"}]})
    res = s.propose(req)
    assert not res.accepted
    which = [v for v in res.violations if v.rule == "instance_which"]
    assert which and "limiter#1" in which[0].message


# ---- the last family: saturation, overdrive, chorus, flanger, Auto Filter, utility ----

LAST_FAMILY = ("saturation", "overdrive", "chorus", "flanger", "auto_filter", "utility")


def test_the_last_family_is_no_longer_one_per_chain_and_width_stays():
    for kind in LAST_FAMILY:
        assert kind not in _NATIVE_SINGLETON_KINDS, kind
    # a track's Width device IS its strip's width: a second would be a second face on one number
    assert _NATIVE_SINGLETON_KINDS == ("width",)


def test_the_brain_takes_a_second_of_the_last_family(demo_project):
    s = Session(demo_project)
    stem = next(iter(s._state["stems"]))
    for kind in LAST_FAMILY:
        s.add_stem_device(stem, kind)
        s.add_stem_device(stem, kind)
        assert sum(1 for e in s._state["stems"][stem]["chain"] if e["type"] == kind) == 2, kind
    for kind in ("saturation", "utility"):
        before = sum(1 for e in s._state["master_chain"] if e["type"] == kind)
        s.add_master_device(kind)
        s.add_master_device(kind)
        assert sum(1 for e in s._state["master_chain"] if e["type"] == kind) == before + 2, kind


def test_a_saturation_proposal_on_a_channel_with_two_saturators_is_refused(demo_project):
    s = Session(demo_project)
    stem = next(iter(s._state["stems"]))
    s.add_stem_device(stem, "saturation")
    s.add_stem_device(stem, "saturation")
    req = ProposalRequest.model_validate({
        "rationale": "warmth", "actions": [{"type": "saturation", "stem_id": stem, "drive_db": 3.0}],
        "predicted_impact": [{"metric": "stem.crest_db", "predicted_delta": -0.5, "unit": "dB"}]})
    res = s.propose(req)
    assert not res.accepted
    which = [v for v in res.violations if v.rule == "instance_which"]
    assert which and "saturation#1" in which[0].message
