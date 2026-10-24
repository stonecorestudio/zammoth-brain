"""A channel's fader stops at +6 dB - at every door (reader P2 E1.R4; James, 2026-10-05).

The fader, the track head's cell and the group drag in the DAW run -60..+6. Three other doors
did not: `track.set_gain` and the brain's `USER_EDIT_FIELDS` took +12, a gain lane or
`device.set gain.gain_db` took +12 (the table row in native/Source/DeviceSpecs.h), and a mix
run's gain move ADDED to the stored level with no ceiling at all. A channel above +6 showed
6.0 on its fader, lost the excess at the first touch, and kept both surfaces repainting.

One number now: `guardrails.FADER_DB_RANGE`. Level past the fader's top is a Utility's job.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from mix_engine import guardrails
from mix_engine.models import GainAction, ProposalRequest
from mix_engine.session import MASTER, Session

REPO = Path(__file__).resolve().parents[2]
TOP = 6.0


def _propose(session: Session, *actions):
    return session.propose(ProposalRequest(rationale="fader ceiling", predicted_impact=[], actions=list(actions)))


def _stem(session: Session) -> str:
    return sorted(session.stems)[0]


def test_one_range_and_it_stops_at_plus_six():
    assert getattr(guardrails, "FADER_DB_RANGE", None) == (-60.0, TOP)
    assert Session.USER_EDIT_FIELDS["gain_db"] == guardrails.FADER_DB_RANGE


def test_the_users_own_setters_stop_at_plus_six(demo_project):
    s = Session(demo_project)
    stem = _stem(s)
    assert s.user_edit(stem, {"gain_db": 9.0})["gain_db"] == TOP
    assert s.user_edit(MASTER, {"gain_db": 11.0})["gain_db"] == TOP
    bus = s.group_stems([stem])["bus_id"]
    s.user_edit(bus, {"gain_db": 12.0})
    assert s.buses[bus]["gain_db"] == TOP
    assert s.set_master_param("gain", 0, 9.0, target=stem)["value"] == TOP


@pytest.mark.parametrize("target", ["stem", "bus", "master"])
def test_a_mix_runs_gain_move_cannot_take_a_fader_past_its_top(demo_project, target):
    s = Session(demo_project)
    stem = _stem(s)
    who = {"stem": stem, "master": MASTER}.get(target) or s.group_stems([stem])["bus_id"]
    s.user_edit(who, {"gain_db": 4.0})

    over = _propose(s, GainAction(stem_id=who, delta_db=3.0))
    assert not over.accepted
    hit = [v for v in over.violations if v.rule == "fader_max_db"]
    assert hit and hit[0].limit == TOP and hit[0].value == pytest.approx(7.0), over.violations
    assert "utility" in hit[0].message.lower(), "the refusal says where more level comes from"

    fits = _propose(s, GainAction(stem_id=who, delta_db=2.0))
    assert fits.accepted, fits.violations
    s.apply(fits.proposal_id)
    level = s.master_gain_db if who == MASTER else (s.buses.get(who) or s.stems[who])["gain_db"]
    assert level == pytest.approx(TOP)


def test_two_moves_in_one_proposal_are_counted_together(demo_project):
    s = Session(demo_project)
    stem = _stem(s)
    s.user_edit(stem, {"gain_db": 2.0})
    res = _propose(s, GainAction(stem_id=stem, delta_db=3.0), GainAction(stem_id=stem, delta_db=3.0))
    assert not res.accepted and any(v.rule == "fader_max_db" for v in res.violations), res.violations


def test_a_cut_and_a_ranged_ride_are_not_the_faders_business(demo_project):
    s = Session(demo_project)
    stem = _stem(s)
    s.user_edit(stem, {"gain_db": TOP})
    cut = _propose(s, GainAction(stem_id=stem, delta_db=-3.0))
    assert cut.accepted, cut.violations
    # a time-ranged move is a ride on top of the fader, not the fader's own level
    ride = _propose(s, GainAction(stem_id=stem, delta_db=2.0, time_range_s=(0.0, 1.0)))
    assert not any(v.rule == "fader_max_db" for v in ride.violations), ride.violations


def test_a_stored_level_past_the_top_opens_at_the_top(demo_project):
    s = Session(demo_project)
    stem = _stem(s)
    bus = s.group_stems([stem])["bus_id"]
    path = Path(demo_project) / "session.json"
    state = json.loads(path.read_text())
    state["stems"][stem]["gain_db"] = 11.0
    state["buses"][bus]["gain_db"] = 9.5
    state["master_gain_db"] = 12.0
    path.write_text(json.dumps(state))

    again = Session(demo_project)
    assert again.stems[stem]["gain_db"] == TOP
    assert again.buses[bus]["gain_db"] == TOP
    assert again.master_gain_db == TOP


def test_the_daws_two_copies_say_the_same_number():
    cmds = (REPO / "daw" / "Source" / "DawCommands.h").read_text(encoding="utf-8")
    assert float(re.search(r"kMaxLevelDb\s*=\s*(-?[\d.]+)", cmds).group(1)) == TOP
    specs = (REPO / "native" / "Source" / "DeviceSpecs.h").read_text(encoding="utf-8")
    row = re.search(r'\{\s*"gain",\s*"gain_db",\s*(-?[\d.]+)f,\s*(-?[\d.]+)f', specs)
    assert row and (float(row.group(1)), float(row.group(2))) == guardrails.FADER_DB_RANGE
    knob = re.search(r'\{\s*"Gain",\s*"gain_db",\s*(-?[\d.]+),\s*(-?[\d.]+)', specs)
    assert knob and float(knob.group(2)) == TOP, "the Gain tile's knob spans the fader's range"
