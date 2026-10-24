"""Gates for batch Py1 of the reader P2s (full audit 2026-10-02): guardrails and the session ledger.

Eight findings - D9·R5 to D9·R10, D8·R2, D8·R4. Each test drives the door the product uses
(`Session.propose` / `apply`, `daw_bridge.apply_coedits` / `rename_stem` / `undo_last`) and carries
its control: the same shape inside the bound is accepted, the same excess sent plainly is refused.
Every one was run on the unfixed tree first. The record is
docs/audits/full-2026-10-02/readers/P2-fixes.md (Py1).
"""
from __future__ import annotations

import math
import shutil

import pytest
from pydantic import ValidationError

from mix_engine.guardrails import LIMITS, level_charge_db
from mix_engine.models import ProposalRequest
from mix_engine.session import Session

_U = "http://127.0.0.1:1/api/"


def _propose(session: Session, actions: list[dict]):
    return session.propose(ProposalRequest(rationale="py1", predicted_impact=[], actions=actions))


def _apply(session: Session, actions: list[dict]) -> None:
    result = _propose(session, actions)
    assert result.accepted, result.violations
    assert session.apply(result.proposal_id).ok


def _lands(session: Session, actions: list[dict]) -> bool:
    result = _propose(session, actions)
    return bool(result.accepted and session.apply(result.proposal_id).ok)


def _rules(result) -> set[str]:
    return {v.rule for v in result.violations}


def _bell(stem: str, gain: float, freq: float = 440.0) -> dict:
    return {"type": "eq", "stem_id": stem,
            "band": {"shape": "peak", "freq_hz": freq, "gain_db": gain, "q": 1.0}}


def _gain(stem: str, db: float) -> dict:
    return {"type": "gain", "stem_id": stem, "delta_db": db}


def _low_faders(session: Session, *targets: str) -> None:
    """These tests are about the WALLETS; a fader stops at +6 (test_fader_ceiling), so start low."""
    for target in targets:
        session.user_edit(target, {"gain_db": -30.0})


def _four_stems(tmp_path):
    import numpy as np
    import soundfile as sf

    proj = tmp_path / "four"
    proj.mkdir(parents=True)
    t = np.arange(2 * 44100) / 44100
    for name, hz in (("drums", 110), ("bassline", 55), ("lead", 440), ("pads", 220)):
        sf.write(proj / f"{name}.wav", (0.2 * np.sin(2 * np.pi * hz * t)).astype("float32"), 44100)
    Session(proj)
    return proj


# ---------- D9·R5: a cut the user took out is not credit ----------

def _bells_that_land(session: Session, tries: int = 4) -> int:
    landed = 0
    for _ in range(tries):
        landed += _lands(session, [_bell("stem_a", 6.0)])
        session.advance_pass()
    return landed


@pytest.mark.parametrize("how", ["removed", "zeroed", "bypassed"])
def test_an_agent_cut_the_user_took_out_is_not_credit_for_a_boost(demo_project, how):
    """The agent cuts 440 Hz by 12 dB; the user deletes the band (or drags it to 0, or switches
    it off). Four +6 dB bells then landed, one a pass: +24 dB of agent EQ against a +12 rail."""
    session = Session(demo_project)
    _apply(session, [_bell("stem_a", -12.0)])
    session.advance_pass()
    if how == "removed":
        session.remove_stem_device("stem_a", "eq", 0)
    elif how == "zeroed":
        session.set_master_device_param("eq", 0, "gain_db", 0.0, target="stem_a")
    else:
        session.set_master_device_enabled("eq", 0, False, target="stem_a")
    assert _bells_that_land(session) == 2


def test_an_agent_cut_still_in_the_chain_is_still_a_cut(demo_project):
    """Control: the cut is still there, so the response it composes with is real - more bells
    land than on a track with no cut (whole-file moves net, as the D9·R4 fix left them)."""
    session = Session(demo_project)
    _apply(session, [_bell("stem_a", -12.0)])
    session.advance_pass()
    assert _bells_that_land(session) > 2


def test_an_agent_cut_put_back_by_its_undo_counts_again(demo_project):
    """The entry is found by identity, so a later proposal's undo - which restores the chain
    from a snapshot - still leaves the cut where the ledger looks for it."""
    session = Session(demo_project)
    _apply(session, [_bell("stem_a", -12.0)])
    session.advance_pass()
    _apply(session, [{"type": "saturation", "stem_id": "stem_a", "drive_db": 3.0}])
    assert session.undo(session._state["proposals"][-1]["proposal_id"]).ok
    session.advance_pass()
    assert _bells_that_land(session) > 2


def test_a_utility_cut_the_user_removed_is_not_credit_for_gain(demo_project):
    """The same hole in the gain wallet, where there is an entry to ask: the agent's Utility at
    -6 dB, removed by the user, went on reading as -6 dB of room."""
    for removed in (True, False):
        proj = demo_project.parent / f"util_{removed}"
        shutil.copytree(demo_project, proj)
        session = Session(proj)
        _low_faders(session, "stem_a")
        _apply(session, [{"type": "utility", "stem_id": "stem_a", "gain_db": -6.0}])
        session.advance_pass()
        if removed:
            session.remove_stem_device("stem_a", "utility", 0)
        landed = 0
        for _ in range(3):
            landed += _lands(session, [_gain("stem_a", 6.0)])
            session.advance_pass()
        assert landed == (2 if removed else 3), removed


# ---------- D9·R6: a transient's lift is level; OTT's upward side fits the pass rail ----------

def test_a_transient_lift_is_charged_to_the_gain_wallet(demo_project):
    """attack +6 and sustain +6 is a flat +6 dB, and nothing charged it: a gain +6 in the same
    proposal put the stem +12 dB up in one pass."""
    session = Session(demo_project)
    _low_faders(session, "stem_a", "stem_b")
    both = {"type": "transient", "stem_id": "stem_a", "attack_db": 6.0, "sustain_db": 6.0}
    refused = _propose(session, [both, _gain("stem_a", 6.0)])
    assert "gain_per_stem_per_pass_db" in _rules(refused)
    # ...and the ledger remembers it for the next proposal of the pass
    _apply(session, [both])
    assert "gain_per_stem_per_pass_db" in _rules(_propose(session, [_gain("stem_a", 1.0)]))
    # control: the same shape inside the budget
    _apply(session, [{"type": "transient", "stem_id": "stem_b", "attack_db": 3.0, "sustain_db": 1.0},
                     _gain("stem_b", 3.0)])


def test_the_transient_charge_is_its_larger_lift_plus_its_output():
    charge = lambda **kw: level_charge_db({"type": "transient", **kw})  # noqa: E731
    assert charge(attack_db=6.0, sustain_db=6.0) == 6.0
    assert charge(attack_db=4.0, sustain_db=-3.0, output_db=2.0) == 6.0
    assert charge(attack_db=-6.0, sustain_db=-6.0) == 0.0            # a cut is free, never credit
    assert charge(attack_db=-6.0, sustain_db=-6.0, output_db=3.0) == 3.0


def test_the_agents_ott_depth_keeps_its_upward_side_inside_the_pass_rail(demo_project):
    """OTT lifts quiet material by up to OTT_MAX_BOOST_DB x depth and no wallet charges it. The
    run's depth cap is set so that worst case IS the per-pass rail (James, 2026-10-05)."""
    from mix_engine.dsp import OTT_MAX_BOOST_DB

    assert OTT_MAX_BOOST_DB * LIMITS["ott_depth_max"] <= LIMITS["gain_per_stem_per_pass_db"]
    session = Session(demo_project)
    deep = _propose(session, [{"type": "ott", "stem_id": "stem_a", "depth": 0.3}])
    assert "ott_depth_max" in _rules(deep)
    _apply(session, [{"type": "ott", "stem_id": "stem_a", "depth": LIMITS["ott_depth_max"]}])


def test_ott_at_the_agents_cap_lifts_a_quiet_stem_no_more_than_the_rail():
    """Measured, not derived: a -45 dB stem through `dsp._ott` at the cap."""
    import numpy as np

    from mix_engine import dsp

    sr = 44100
    rng = np.random.default_rng(7)
    quiet = (rng.standard_normal((2, 4 * sr)) * 10 ** (-45 / 20)).astype(np.float32)   # (channels, frames)

    def lift_db(depth: float) -> float:
        out = dsp._ott(quiet.copy(), sr, {"type": "ott", "depth": depth, "time": 1.0})
        rms = lambda x: float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64)[:, sr:] ** 2)))  # noqa: E731
        return 20 * math.log10(rms(out) / rms(quiet))

    assert lift_db(LIMITS["ott_depth_max"]) <= LIMITS["gain_per_stem_per_pass_db"] + 0.1
    # control: the old cap was past it. Since 2026-10-17 (the OTT audit) the band level is the
    # DAW's per-sample PEAK, which reads noise ~10 dB over its RMS, so the lift at 0.8 is 6.3 dB
    # (it was over 7 on the old 5 ms RMS): still past the rail, by less.
    assert lift_db(0.8) > LIMITS["gain_per_stem_per_pass_db"]


# ---------- D9·R7: a group is on its members' signal path ----------

def test_a_one_member_group_is_not_a_second_gain_wallet(demo_project):
    """`group solo = [stem_a]`, gain stem_a +6, gain solo +6: one proposal, the stem +12 dB."""
    session = Session(demo_project)
    _low_faders(session, "stem_a")
    alias = _propose(session, [{"type": "group", "bus_id": "solo", "members": ["stem_a"]},
                               _gain("stem_a", 6.0), _gain("solo", 6.0)])
    assert "gain_per_stem_per_pass_db" in _rules(alias)
    # control: the same three actions inside one budget
    _apply(session, [{"type": "group", "bus_id": "solo", "members": ["stem_a"]},
                     _gain("stem_a", 3.0), _gain("solo", 3.0)])


def test_a_one_member_group_is_not_a_second_eq_rail(demo_project):
    session = Session(demo_project)
    alias = _propose(session, [{"type": "group", "bus_id": "solo", "members": ["stem_a"]},
                               _bell("stem_a", 6.0), _bell("solo", 6.0)])
    assert "eq_boost_pass_db" in _rules(alias)
    _apply(session, [{"type": "group", "bus_id": "solo", "members": ["stem_a"]},
                     _bell("stem_a", 3.0), _bell("solo", 3.0)])


def test_a_groups_gain_counts_on_every_member_across_proposals(demo_project):
    """The two-member case (the same alias at half strength): the group +6, then a member +6."""
    session = Session(demo_project)
    session.group_stems(["stem_a", "stem_b"])
    session = Session(demo_project)
    _low_faders(session, "stem_a", "stem_b", "bus1")
    _apply(session, [_gain("bus1", 6.0)])
    refused = _propose(session, [_gain("stem_a", 6.0)])
    assert "gain_per_stem_per_pass_db" in _rules(refused)
    assert "bus1" in refused.violations[0].message          # the refusal says why
    # ...and the other way round, on the session rail: a member +6 a pass for two passes, then the group
    session.advance_pass()
    _apply(session, [_gain("stem_b", 6.0)])                  # stem_b's path: 6 (group) + 6 = 12, the session rail
    session.advance_pass()
    assert "gain_per_stem_per_session_db" in _rules(_propose(session, [_gain("bus1", 1.0)]))
    # control: a track that is not in the group has its own budget
    session.route_stem("stem_a", "master")
    session = Session(demo_project)
    _apply(session, [_gain("stem_a", 6.0)])


def test_a_groups_eq_counts_on_every_member_across_proposals(demo_project):
    session = Session(demo_project)
    session.group_stems(["stem_a", "stem_b"])
    session = Session(demo_project)
    _apply(session, [_bell("bus1", 6.0)])
    assert "eq_boost_pass_db" in _rules(_propose(session, [_bell("stem_a", 6.0)]))
    _apply(session, [_bell("stem_a", 6.0, freq=6000.0)])    # control: another part of the spectrum


def test_routing_a_lifted_track_into_a_lifted_group_is_refused(demo_project, tmp_path):
    """The three-action way round: gain the track, gain a group it is not in, route it there."""
    proj = _four_stems(tmp_path)
    Session(proj).group_stems(["drums", "bassline"])
    session = Session(proj)
    _low_faders(session, "lead", "bus1")
    _apply(session, [_gain("lead", 6.0), _gain("bus1", 6.0)])
    moved = _propose(session, [{"type": "route", "stem_id": "lead", "bus_id": "bus1"}])
    assert "gain_per_stem_per_pass_db" in _rules(moved)
    # control: a track with nothing spent may join
    _apply(session, [{"type": "route", "stem_id": "pads", "bus_id": "bus1"}])


def test_a_path_already_past_the_rail_still_takes_a_cut(demo_project, tmp_path):
    """Checked only where the proposal pushes: the user routes a lifted track into a lifted
    group (their move, not the agent's); the agent may still turn either one DOWN."""
    proj = _four_stems(tmp_path)
    Session(proj).group_stems(["drums", "bassline"])
    session = Session(proj)
    _low_faders(session, "lead", "bus1")
    _apply(session, [_gain("lead", 6.0), _gain("bus1", 6.0)])
    Session(proj).route_stem("lead", "bus1")
    session = Session(proj)
    assert not _propose(session, [_gain("lead", 0.5)]).accepted
    _apply(session, [_gain("lead", -3.0)])
    _apply(session, [_gain("bus1", -2.0)])


# ---------- D9·R8: the agent's chorus and flanger replace ----------

_SINGLETON_SHAPES = {
    "comp": {"threshold_db": -20.0, "ratio": 2.0, "attack_ms": 10.0, "release_ms": 120.0},
    "mb_comp": {b: {"threshold_db": -20.0, "ratio": 2.0} for b in ("low", "mid", "high")},
    "ott": {"depth": 0.2}, "saturation": {"drive_db": 3.0}, "overdrive": {},
    "chorus": {"mix": 0.3}, "flanger": {"mix": 0.3}, "auto_filter": {"cutoff_hz": 800.0}, "clip": {},
    "deess": {"threshold_db": -30.0, "freq_hz": 6000.0}, "gate": {},
    "transient": {"attack_db": 2.0}, "imager": {}, "utility": {"gain_db": 1.0},
    "duck": {"key_stem_id": "stem_b", "threshold_db": -30.0, "depth_db": 3.0},
}


@pytest.mark.parametrize("kind", ["chorus", "flanger", "auto_filter"])
@pytest.mark.parametrize("target", ["stem_a", "bus1"])
def test_a_re_proposed_modulation_device_replaces_the_one_there(demo_project, kind, target):
    """Pass 1 `chorus mix 0.3`, pass 2 "a bit more", `mix 0.4`: the brain rendered and measured
    two in series while the DAW played one at 0.4."""
    session = Session(demo_project)
    if target == "bus1":
        session.group_stems(["stem_a", "stem_b"])
        session = Session(demo_project)
    chain = lambda: (session.buses if target == "bus1" else session.stems)[target]["chain"]  # noqa: E731
    _apply(session, [{"type": kind, "stem_id": target, "mix": 0.3}])
    first = [e for e in chain() if e["type"] == kind][0]["uid"]
    session.advance_pass()
    _apply(session, [{"type": kind, "stem_id": target, "mix": 0.4}])
    held = [e for e in chain() if e["type"] == kind]
    assert [e["mix"] for e in held] == [0.4]
    assert held[0]["uid"] == first                      # the same device in the dock


def test_every_native_singleton_the_agent_may_propose_replaces(demo_project):
    """The list the native chain holds ONE of (`_NATIVE_SINGLETON_KINDS`) and the list the
    agent's re-proposal replaces were two lists; chorus and flanger were in the first only."""
    from mix_engine.session import BUS_ACTION_TYPES, _NATIVE_SINGLETON_KINDS

    stacked = []
    for where, allowed in (("stem_a", None), ("bus1", BUS_ACTION_TYPES)):
        for kind in _NATIVE_SINGLETON_KINDS:
            if kind in ("limiter", "width"):            # master-only / a strip scalar on a stem
                continue
            if allowed is not None and kind not in allowed:
                continue
            proj = demo_project.parent / f"single_{where}_{kind}"
            shutil.copytree(demo_project, proj)
            session = Session(proj)
            if where == "bus1":
                session.group_stems(["stem_a", "stem_b"])
                session = Session(proj)
            action = {"type": kind, "stem_id": where, **_SINGLETON_SHAPES[kind]}
            for _ in range(2):
                _apply(session, [action])
                session.advance_pass()
            chain = (session.buses if where == "bus1" else session.stems)[where]["chain"]
            if [e["type"] for e in chain].count(kind) != 1:
                stacked.append((where, kind))
    assert stacked == []


# ---------- D9·R9: a time range is finite and inside the track ----------

@pytest.mark.parametrize("rng", [[0.0, float("inf")], [float("nan"), 2.0], [0.0, float("nan")],
                                 [float("-inf"), 2.0]])
def test_a_time_range_that_is_not_finite_is_refused_at_the_model(rng):
    """`[0, 1e999]` is valid JSON and reads as infinity; it was accepted, applied, written to
    session.json as `Infinity`, and every later render raised until the proposal was undone."""
    shapes = [
        {"type": "gain", "stem_id": "stem_a", "delta_db": 2.0},
        _bell("stem_a", 2.0),
        {"type": "comp", "stem_id": "stem_a", "threshold_db": -20.0, "ratio": 2.0,
         "attack_ms": 10.0, "release_ms": 120.0},
        {"type": "deess", "stem_id": "stem_a", "threshold_db": -30.0, "freq_hz": 6000.0},
    ]
    for shape in shapes:
        with pytest.raises(ValidationError):
            ProposalRequest(rationale="py1", predicted_impact=[],
                            actions=[{**shape, "time_range_s": rng}])
        ProposalRequest(rationale="py1", predicted_impact=[],          # control
                        actions=[{**shape, "time_range_s": [0.0, 2.0]}])


def test_the_json_number_1e999_is_refused_too():
    import json

    raw = json.loads('{"type": "gain", "stem_id": "stem_a", "delta_db": 2.0, "time_range_s": [0, 1e999]}')
    with pytest.raises(ValidationError):
        ProposalRequest(rationale="py1", predicted_impact=[], actions=[raw])


def test_a_time_range_past_the_end_of_the_track_is_refused(demo_project):
    """`[5000, 6000]` on an 8 s stem rendered as nothing, and still took a comp slot and charged
    its make-up."""
    session = Session(demo_project)
    comp = {"type": "comp", "stem_id": "stem_a", "threshold_db": -20.0, "ratio": 2.0,
            "attack_ms": 10.0, "release_ms": 120.0, "makeup_db": 3.0}
    for shape in (comp, _gain("stem_a", 2.0), _bell("stem_a", 2.0)):
        past = _propose(session, [{**shape, "time_range_s": [5000.0, 6000.0]}])
        assert "time_range_past_end" in _rules(past), shape["type"]
    # controls: inside the track, and one that starts inside and runs off its end
    _apply(session, [{**_gain("stem_a", 2.0), "time_range_s": [6.0, 7.5]}])
    _apply(session, [{**_gain("stem_b", 2.0), "time_range_s": [7.0, 60.0]}])


def test_a_range_is_judged_where_the_tracks_clip_plays(demo_project):
    """A track whose one clip sits at 0:30 plays its 8 s from 30 s to 38 s."""
    session = Session(demo_project)
    session.user_edit("stem_a", {"clip_at_s": 30.0, "clip_from_s": 0.0, "clip_for_s": 0.0})
    _apply(session, [{**_gain("stem_a", 2.0), "time_range_s": [31.0, 33.0]}])
    past = _propose(session, [{**_gain("stem_a", 2.0), "time_range_s": [40.0, 42.0]}])
    assert "time_range_past_end" in _rules(past)


# ---------- D9·R10: the master's width floor is on the product ----------

def _master(kind: str, **fields) -> dict:
    return {"type": kind, "stem_id": "master", **fields}


def test_four_legal_narrowings_of_the_master_are_one_illegal_one(demo_project):
    """width 0.5 x utility 0.5 x imager 0.5 x band_width 0.5 = 0.0625 of the side in the mids:
    one accepted proposal, master side/mid -23.6 dB."""
    session = Session(demo_project)
    four = [_master("width", width=0.5), _master("utility", width=0.5),
            _master("imager", width_mid=0.5, width_high=0.5),
            _master("band_width", band="mid", width=0.5)]
    assert "master_width_min" in _rules(_propose(session, four))
    # two that are each legal and compose under the floor
    assert "master_width_min" in _rules(_propose(session, [_master("width", width=0.7),
                                                           _master("utility", width=0.7)]))
    # controls: each alone at the floor, and two that compose above it
    for one in four:
        assert _propose(session, [one]).accepted, one
        session.advance_pass()                  # three open proposals a pass is the other limit
    _apply(session, [_master("width", width=0.8), _master("utility", width=0.8)])


def test_the_floor_counts_what_the_master_already_carries(demo_project):
    session = Session(demo_project)
    _apply(session, [_master("utility", width=0.6)])
    session.advance_pass()
    assert "master_width_min" in _rules(_propose(session, [_master("width", width=0.6)]))
    # a singleton REPLACES its own factor: re-tuning the utility is judged as the new value
    _apply(session, [_master("utility", width=0.7)])
    _apply(session, [_master("width", width=0.8)])               # 0.7 x 0.8 = 0.56


def test_the_floor_binds_only_where_the_proposal_narrows(demo_project):
    """The user's own narrowing is not the agent's to be refused for: with the user's Utility
    at 0.3 the agent may still EQ the master, or widen it - but not narrow it further."""
    session = Session(demo_project)
    session.add_master_device("utility")
    session.set_master_device_param("utility", 0, "width", 0.3)
    session = Session(demo_project)
    _apply(session, [_bell("master", 2.0)])
    _apply(session, [_master("imager", width_high=1.2)])
    assert "master_width_min" in _rules(_propose(session, [_master("width", width=0.9)]))
    # a device the user switched off narrows nothing
    session.set_master_device_enabled("utility", 0, False)
    session = Session(demo_project)
    _apply(session, [_master("width", width=0.9)])


def test_folding_the_lows_stays_free_under_the_composed_floor(demo_project):
    session = Session(demo_project)
    _apply(session, [_master("utility", width=0.5), _master("imager", width_low=0.0),
                     _master("band_width", band="bass", width=0.0)])


# ---------- D8·R2: the journal follows a rename ----------

def test_rekey_reads_a_url_that_carries_a_hash():
    from mix_engine.session import _rekey_coedit_url

    url = _U + "device_order?order=eq#0,clip#0,eq#1&stem=vox"
    moved = _rekey_coedit_url(url, "vox", "Lead")
    assert moved != url and "stem=Lead" in moved
    from orchestrator.daw_bridge import _coedit_target

    class _S:                                           # _coedit_target only asks for .buses
        buses: dict = {}
    assert _coedit_target(_S(), moved) == "Lead"
    plain = _rekey_coedit_url(_U + "device_order?order=eq,clip&stem=vox", "vox", "Lead")
    assert "stem=Lead" in plain                         # control: the form that always worked


def test_a_reorder_made_before_a_rename_survives_the_undo(tmp_path):
    """The agent EQs "lead"; the user adds a comp and a saturator and drags the saturator first;
    renames the track; Undo Agent Move. The chain came back comp, saturation: the drag was gone."""
    from orchestrator import daw_bridge

    for rename in (True, False):
        proj = _four_stems(tmp_path / f"order_{rename}")
        _apply(Session(proj), [_bell("lead", -2.0)])
        assert daw_bridge.apply_coedits(proj, [
            _U + "add_device?device_type=comp&stem=lead",
            _U + "add_device?device_type=saturation&stem=lead",
            _U + "device_order?order=eq#0,saturation#0,comp#0&stem=lead"])["failed"] == 0
        name = "lead"
        if rename:
            assert daw_bridge.rename_stem(proj, "lead", "Lead Vox")["renamed"]
            name = "Lead Vox"
        assert [e["type"] for e in Session(proj, read_only=True).stems[name]["chain"]] == \
            ["eq", "saturation", "comp"]
        assert daw_bridge.undo_last(proj) is not None
        assert [e["type"] for e in Session(proj, read_only=True).stems[name]["chain"]] == \
            ["saturation", "comp"], rename


def test_a_duck_keyed_to_a_renamed_track_survives_the_undo(tmp_path):
    """The agent EQs the bassline; the user picks Duck > Trigger: drums; renames drums to
    "Kick"; Undo Agent Move. The bassline had no Duck at all: the replay named a track that no
    longer existed."""
    from orchestrator import daw_bridge

    for rename in (True, False):
        proj = _four_stems(tmp_path / f"duck_{rename}")
        _apply(Session(proj), [_bell("bassline", -2.0)])
        assert daw_bridge.apply_coedits(proj, [_U + "set_duck?stem=bassline&key=drums"])["failed"] == 0
        key = "drums"
        if rename:
            assert daw_bridge.rename_stem(proj, "drums", "Kick")["renamed"]
            key = "Kick"
        assert daw_bridge.undo_last(proj) is not None
        ducks = [e.get("key_stem_id") for e in Session(proj, read_only=True).stems["bassline"]["chain"]
                 if e["type"] == "duck"]
        assert ducks == [key], rename


def test_an_audio_to_target_in_the_journal_follows_the_rename(tmp_path):
    from orchestrator import daw_bridge

    proj = _four_stems(tmp_path)
    _apply(Session(proj), [_bell("pads", -2.0)])
    assert daw_bridge.apply_coedits(proj, [_U + "set_stem?stem=pads&output=track:lead"])["failed"] == 0
    assert daw_bridge.rename_stem(proj, "lead", "Vox")["renamed"]
    urls = [e["url"] for e in Session(proj, read_only=True)._state.get("coedit_log", [])]
    assert any("track%3AVox" in u or "track:Vox" in u for u in urls), urls
    assert not any("lead" in u for u in urls), urls


# ---------- D8·R4: the retention trim keeps what was spent ----------

def _past_the_horizon(session: Session, stem: str = "stem_a") -> None:
    for _ in range(60):                 # refused attempts are records too; the trim runs at +50
        _propose(session, [_gain(stem, 12.0)])


def test_a_trimmed_device_trim_is_still_spent(demo_project, monkeypatch):
    """The agent's Utility at +6 dB is still in the chain; once its record was trimmed the
    wallet read 0, and +12 dB of fader gain landed on top."""
    import mix_engine.session as session_mod

    for trim in (True, False):
        proj = demo_project.parent / f"trim_{trim}"
        shutil.copytree(demo_project, proj)
        monkeypatch.setattr(session_mod, "_MAX_PROPOSAL_RECORDS", 2 if trim else 400)
        session = Session(proj)
        _low_faders(session, "stem_b")
        _apply(session, [{"type": "utility", "stem_id": "stem_b", "gain_db": 6.0}])
        session.advance_pass()
        _past_the_horizon(session)
        assert (len(session._state["proposals"]) <= 52) == trim
        assert session._session_gain_baseline([]).get("stem_b") == 6.0, trim
        landed = 0
        for _ in range(3):
            landed += _lands(session, [_gain("stem_b", 6.0)])
            session.advance_pass()
        assert landed == 1, trim


def test_a_trimmed_eq_output_trim_is_still_spent(demo_project, monkeypatch):
    import mix_engine.session as session_mod

    monkeypatch.setattr(session_mod, "_MAX_PROPOSAL_RECORDS", 2)
    session = Session(demo_project)
    _low_faders(session, "stem_b")
    _apply(session, [{**_bell("stem_b", -2.0), "output_db": 6.0}])
    session.advance_pass()
    _past_the_horizon(session)
    assert session._session_gain_baseline([]).get("stem_b") == 6.0


def test_a_singleton_re_proposed_after_its_record_was_trimmed_costs_its_delta(demo_project, monkeypatch):
    """The carry holds a singleton's LAST value, as the live ledger does: adjusting the trimmed
    Utility from +6 to +4 is not +10 dB of spend."""
    import mix_engine.session as session_mod

    monkeypatch.setattr(session_mod, "_MAX_PROPOSAL_RECORDS", 2)
    session = Session(demo_project)
    _low_faders(session, "stem_b")
    _apply(session, [{"type": "utility", "stem_id": "stem_b", "gain_db": 6.0}])
    session.advance_pass()
    _past_the_horizon(session)
    _apply(session, [{"type": "utility", "stem_id": "stem_b", "gain_db": 4.0}])
    assert session._session_gain_baseline([]).get("stem_b") == 4.0


def test_the_trimmed_carry_follows_a_track_through_a_rename(demo_project, monkeypatch):
    """+10 dB spent on "stem_b" and trimmed; the user renames it "Vox"; the wallet of "Vox" read 0."""
    import mix_engine.session as session_mod
    from orchestrator import daw_bridge

    for rename in (True, False):
        proj = demo_project.parent / f"rename_{rename}"
        shutil.copytree(demo_project, proj)
        monkeypatch.setattr(session_mod, "_MAX_PROPOSAL_RECORDS", 2)
        session = Session(proj)
        _low_faders(session, "stem_b")
        for _ in range(2):
            _apply(session, [_gain("stem_b", 5.0), _bell("stem_b", 6.0),
                             {**_gain("stem_b", 1.0), "time_range_s": [1.0, 2.0]}])
            session.advance_pass()
        _past_the_horizon(session)
        name = "stem_b"
        if rename:
            assert daw_bridge.rename_stem(proj, "stem_b", "Vox")["renamed"]
            name = "Vox"
        session = Session(proj)
        assert session._cumulative_gain(None).get(name) == 10.0, rename
        assert session._ranged_gain(None).get(name) == (2.0, 0.0), rename
        assert not _propose(session, [_gain(name, 5.0)]).accepted, rename
        assert "eq_boost_session_db" in _rules(_propose(session, [_bell(name, 6.0)])), rename
        assert "stem_b" not in session._state.get("trimmed_gain_totals", {}) or not rename


def test_the_trimmed_carry_follows_a_group_and_a_return_through_a_rename(tmp_path, monkeypatch):
    import mix_engine.session as session_mod

    monkeypatch.setattr(session_mod, "_MAX_PROPOSAL_RECORDS", 2)
    proj = _four_stems(tmp_path)
    Session(proj).group_stems(["drums", "bassline"])
    session = Session(proj)
    _low_faders(session, "bus1")
    _apply(session, [_gain("bus1", 5.0)])
    session.advance_pass()
    _past_the_horizon(session, "lead")
    assert session._cumulative_gain(None).get("bus1") == 5.0
    session.rename_bus("bus1", "Rhythm")
    session = Session(proj)
    assert session._cumulative_gain(None).get("Rhythm") == 5.0
    assert "bus1" not in session._state.get("trimmed_gain_totals", {})
    # a return's carry is keyed `return:<id>` and moves the same way
    session._state.setdefault("trimmed_gain_totals", {})["return:rv"] = 3.0
    session._state["returns"]["rv"] = session._new_routing_record("return", "rv2")
    session._state["trimmed_gain_totals"]["return:rv"] = 3.0
    session.rename_return("rv", "Plate")
    assert session._state["trimmed_gain_totals"].get("return:Plate") == 3.0
    assert "return:rv" not in session._state["trimmed_gain_totals"]


# ---------- the parts of the fixes the tests above do not reach ----------

def test_the_agent_is_told_the_room_a_groups_eq_left_its_members(demo_project):
    """`eq_headroom_db` is what the agent sizes its moves by: with the group lifted 6 dB at
    440 Hz, a member has no boost left there this pass - and is listed, though untouched."""
    session = Session(demo_project)
    session.group_stems(["stem_a", "stem_b"])
    session = Session(demo_project)
    assert "stem_a" not in session.get_state().eq_headroom_db          # control: nothing spent yet
    _apply(session, [_bell("bus1", 6.0)])
    room = session.get_state().eq_headroom_db
    assert room["stem_a"]["mid"]["boost"] == 0.0                       # 440 Hz is in `mid`
    assert room["stem_a"]["air"]["boost"] > 5.0                        # the bell is not up there


def test_a_track_fed_by_another_is_on_that_tracks_path(tmp_path):
    """Audio To: "pads" plays into "lead"'s input, so lead's fader and devices are on its path."""
    proj = _four_stems(tmp_path)
    session = Session(proj)
    _low_faders(session, "lead", "pads")
    session.user_edit("pads", {"output": "track:lead"})
    session = Session(proj)
    assert "gain_per_stem_per_pass_db" in _rules(_propose(session, [_gain("pads", 6.0), _gain("lead", 6.0)]))
    _apply(session, [_gain("pads", 3.0), _gain("lead", 3.0)])
    _apply(session, [_gain("drums", 6.0)])                             # control: not on that path


def test_routing_a_boosted_track_into_a_boosted_group_is_refused_for_its_eq(tmp_path):
    proj = _four_stems(tmp_path)
    Session(proj).group_stems(["drums", "bassline"])
    session = Session(proj)
    _apply(session, [_bell("lead", 6.0), _bell("bus1", 6.0)])
    moved = _propose(session, [{"type": "route", "stem_id": "lead", "bus_id": "bus1"}])
    assert "eq_boost_pass_db" in _rules(moved)
    _apply(session, [{"type": "route", "stem_id": "pads", "bus_id": "bus1"}])


def test_a_range_past_a_tracks_own_end_is_fine_while_a_track_feeding_it_plays(tmp_path):
    import numpy as np
    import soundfile as sf

    proj = _four_stems(tmp_path)                                       # every stem is 2 s
    t = np.arange(6 * 44100) / 44100
    sf.write(proj / "long.wav", (0.2 * np.sin(2 * np.pi * 330 * t)).astype("float32"), 44100)
    session = Session(proj)
    assert "time_range_past_end" in _rules(_propose(session, [{**_gain("lead", 2.0), "time_range_s": [3.0, 4.0]}]))
    session.user_edit("long", {"output": "track:lead"})
    session = Session(proj)
    _apply(session, [{**_gain("lead", 2.0), "time_range_s": [3.0, 4.0]}])


def test_a_cut_the_user_took_out_is_not_credit_once_its_record_is_trimmed(demo_project, monkeypatch):
    import mix_engine.session as session_mod

    for removed in (True, False):
        proj = demo_project.parent / f"trimcut_{removed}"
        shutil.copytree(demo_project, proj)
        monkeypatch.setattr(session_mod, "_MAX_PROPOSAL_RECORDS", 2)
        session = Session(proj)
        _apply(session, [_bell("stem_a", -12.0)])
        session.advance_pass()
        if removed:
            session.remove_stem_device("stem_a", "eq", 0)
        _past_the_horizon(session, "stem_b")
        assert not any(r["status"] == "applied" for r in session._state["proposals"])   # it IS trimmed
        landed = _bells_that_land(session)
        assert (landed == 2) if removed else (landed > 2), (removed, landed)


def test_re_tuning_a_trimmed_singleton_costs_the_pass_its_delta(demo_project, monkeypatch):
    """In the scope of a PASS a replaced singleton is charged against the value it replaced;
    the carry is where that value lives once its record is trimmed."""
    import mix_engine.session as session_mod

    monkeypatch.setattr(session_mod, "_MAX_PROPOSAL_RECORDS", 2)
    session = Session(demo_project)
    _low_faders(session, "stem_b")
    _apply(session, [{"type": "utility", "stem_id": "stem_b", "gain_db": 6.0}])
    session.advance_pass()
    _past_the_horizon(session)
    # the same Utility again is no move at all, so a whole pass of gain still fits beside it
    _apply(session, [{"type": "utility", "stem_id": "stem_b", "gain_db": 6.0}, _gain("stem_b", 6.0)])
