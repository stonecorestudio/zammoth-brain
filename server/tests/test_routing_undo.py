"""A removed group or return can come back, and no undo leaves a reference to one that is gone.

Full audit 2026-10-02, readers S4·R25 (a), C7·R4, C7·R5 and D8·R1 — "routing is in neither
undo". A group's and a return's devices, level, sends and members lived only in the brain's
session; the DAW realizes a removal as an undoable track delete. So ⌘Z brought the TRACK back,
the brain had nothing to bring back, and the app deleted the track again through the undo
manager: every further ⌘Z re-added and re-deleted it, nothing older could be reached and redo
was gone. And a proposal's undo wrote back whole prior records without looking at the
structure as it is now, leaving stems routed to buses that do not exist.

Both ends are undoable now. This file is the brain's half: `Session._bury` /
`restore_routing` / `remove_routing`, the co-edits the DAW's undo sends, and the reference
reconcile after an undo. The DAW's half is the console leg ``--routing-undo``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mix_engine.models import ProposalRequest
from mix_engine.session import ROUTING_TOMBSTONES_MAX, Session
from orchestrator import daw_bridge
from orchestrator import daw_server as server


def _apply(session: Session, actions: list[dict]) -> str:
    result = session.propose(ProposalRequest(rationale="routing undo", predicted_impact=[],
                                             actions=actions))
    assert result.accepted, result.violations
    assert session.apply(result.proposal_id).ok
    return result.proposal_id


def _dangling(session: Session) -> list[str]:
    """The invariant D8·R1 names: nothing points at a bus or return that is not there."""
    buses, returns, out = session.buses, session.returns, []
    for sid, stem in session.stems.items():
        if stem.get("bus") is not None and stem["bus"] not in buses:
            out.append(f"stem {sid} -> bus {stem['bus']}")
        out += [f"stem {sid} sends to {rid}" for rid in (stem.get("sends") or {}) if rid not in returns]
    for bid, bus in buses.items():
        if bus.get("output") is not None and bus["output"] not in buses:
            out.append(f"bus {bid} -> bus {bus['output']}")
        out += [f"bus {bid} sends to {rid}" for rid in (bus.get("sends") or {}) if rid not in returns]
    for rid, ret in returns.items():
        out += [f"return {rid} sends to {r}" for r in (ret.get("sends") or {}) if r not in returns]
    return out


@pytest.fixture()
def rig(demo_project) -> Session:
    """A plate with a tuned reverb and a vocal send; a delay the plate feeds; a drum group with
    a compressor that feeds the plate; and a group nested into it."""
    s = Session(demo_project)
    assert s.add_return("reverb")["return_id"] == "rv"
    assert s.add_return("delay")["return_id"] == "dly"
    s.returns["rv"]["chain"][0]["decay_s"] = 3.3
    s.set_return_level("rv", -4.0)
    s.set_send("stem_a", "rv", -9.0)
    s.set_send("return:rv", "dly", -15.0)           # a cascade: rv feeds the LATER return
    assert s.group_stems(["stem_a", "stem_b"])["bus_id"] == "bus1"
    s.add_stem_device("bus1", "comp")
    s.set_send("bus:bus1", "rv", -20.0)
    return Session(demo_project)


def test_a_removed_return_comes_back_whole(rig):
    before = json.loads(json.dumps({"entry": rig.returns["rv"], "order": list(rig.returns),
                                    "stem": rig.stems["stem_a"]["sends"],
                                    "bus": rig.buses["bus1"]["sends"]}))
    rig.remove_return("rv")
    assert "rv" not in rig.returns and not _dangling(rig)
    assert rig.stems["stem_a"]["sends"] == {} and rig.buses["bus1"]["sends"] == {}

    done = rig.restore_routing("return", "rv", before["entry"]["uid"])

    assert done["restored"] is True and done["adopted"] is False
    again = Session(rig.project_dir)
    assert list(again.returns) == before["order"]            # its POSITION: the cascade rule is an index
    assert again.returns["rv"] == before["entry"]            # devices, level, tap, its own send to dly
    assert again.stems["stem_a"]["sends"] == before["stem"]
    assert again.buses["bus1"]["sends"] == before["bus"]
    assert not _dangling(again)


def test_a_removed_group_comes_back_with_its_members_and_the_bus_that_fed_it(rig):
    assert rig.group_stems(["ghost_track"])["bus_id"] == "bus2"     # an empty track: a pending member
    rig.route_bus("bus2", "bus1")
    entry = json.loads(json.dumps(rig.buses["bus1"]))
    rig.remove_bus("bus1")
    assert rig.stems["stem_a"]["bus"] is None and rig.buses["bus2"]["output"] is None

    rig.restore_routing("bus", "bus1", entry["uid"])

    again = Session(rig.project_dir)
    assert again.buses["bus1"] == entry                              # the compressor, the send, the fader
    assert again.stems["stem_a"]["bus"] == "bus1" and again.stems["stem_b"]["bus"] == "bus1"
    assert again.buses["bus2"]["output"] == "bus1"
    assert list(again.buses) == ["bus1", "bus2"]
    assert not _dangling(again)


def test_a_member_the_user_has_since_routed_elsewhere_stays_there(rig):
    uid = rig.buses["bus1"]["uid"]
    rig.remove_bus("bus1")
    rig.route_stem("stem_b", "new")                                  # → bus1 again? no: a NEW record
    other = rig.stems["stem_b"]["bus"]
    rig.restore_routing("bus", other, uid) if other == "bus1" else rig.restore_routing("bus", "bus1", uid)
    again = Session(rig.project_dir)
    assert again.stems["stem_a"]["bus"] == "bus1"
    assert not _dangling(again)


def test_an_id_that_was_reused_restores_the_record_its_track_names(rig):
    """Remove `rv`, add a reverb: the new return is `rv` again. ⌘Z twice walks back over BOTH,
    and a newest-first stack would hand the second press the wrong one."""
    old_uid = rig.returns["rv"]["uid"]
    rig.remove_return("rv")
    assert rig.add_return("reverb")["return_id"] == "rv"
    new_uid = rig.returns["rv"]["uid"]
    assert new_uid != old_uid
    rig.remove_routing("return", "rv", new_uid)          # ⌘Z #1: the new return's track is gone
    rig.restore_routing("return", "rv", old_uid)         # ⌘Z #2: the old return's track is back
    assert rig.returns["rv"]["uid"] == old_uid
    assert rig.returns["rv"]["chain"][0]["decay_s"] == 3.3 and rig.returns["rv"]["gain_db"] == -4.0
    assert rig.stems["stem_a"]["sends"] == {"rv": -9.0}
    rig.remove_routing("return", "rv", old_uid)          # ⇧⌘Z #1
    rig.restore_routing("return", "rv", new_uid)         # ⇧⌘Z #2
    assert rig.returns["rv"]["uid"] == new_uid and rig.stems["stem_a"]["sends"] == {}
    assert not _dangling(Session(rig.project_dir))


def test_a_live_record_is_swapped_out_only_for_the_exact_tombstone(rig):
    """A track stamped with an identity the brain has no tombstone for must not cost the live
    record of the same name its devices."""
    live = json.loads(json.dumps(rig.returns["rv"]))
    done = rig.restore_routing("return", "rv", "rv@999")
    assert done["restored"] is False and done["kept"] == live["uid"]
    assert rig.returns["rv"] == live
    # …and the swap, when the tombstone IS there, keeps the one that stepped aside
    rig.remove_return("rv")
    rig.add_return("reverb")
    newer = rig.returns["rv"]["uid"]
    rig.restore_routing("return", "rv", live["uid"])
    assert rig.returns["rv"]["uid"] == live["uid"]
    assert any(t["uid"] == newer for t in rig._state["routing_tombstones"])


def test_with_no_tombstone_the_record_returns_at_its_defaults(rig):
    done = rig.restore_routing("bus", "old_group", "old_group@7")
    assert done["restored"] is True and done["adopted"] is True
    assert rig.buses["old_group"]["uid"] == "old_group@7" and rig.buses["old_group"]["chain"] == []
    with pytest.raises(ValueError):
        rig.restore_routing("bus", "stem_a")             # a stem has that name: never a second holder


def test_remove_routing_states_what_the_engine_holds(rig):
    uid = rig.buses["bus1"]["uid"]
    assert rig.remove_routing("bus", "bus1", "bus1@999")["removed"] is False     # a later record: not that one
    assert "bus1" in rig.buses
    assert rig.remove_routing("bus", "bus1", uid)["removed"] is True
    assert rig.remove_routing("bus", "bus1", uid)["removed"] is False            # already gone: no error
    assert rig.stems["stem_a"]["bus"] is None and not _dangling(rig)


def test_a_group_the_tidy_dropped_is_kept_too(rig):
    """Routing the last member away drops the group — and ⌘Z of that route brings its track
    back as surely as ⌘Z of a removal does."""
    entry = json.loads(json.dumps(rig.buses["bus1"]))
    rig.route_stem("stem_a", None)
    rig.route_stem("stem_b", None)
    assert "bus1" not in rig.buses
    rig.restore_routing("bus", "bus1", entry["uid"])
    assert rig.buses["bus1"]["chain"] == entry["chain"]


def test_the_tombstones_are_bounded(rig):
    for n in range(ROUTING_TOMBSTONES_MAX + 5):
        rig.group_stems([f"empty {n}"])
        rig.remove_bus(sorted(b for b in rig.buses if b != "bus1")[0])
    assert len(rig._state["routing_tombstones"]) == ROUTING_TOMBSTONES_MAX


def test_every_new_record_gets_an_identity_and_an_old_one_derives_the_same_one_twice(rig):
    uids = [r["uid"] for r in (*rig.buses.values(), *rig.returns.values())]
    assert len(set(uids)) == len(uids) and all("@" in u for u in uids)
    state = json.loads((rig.project_dir / "session.json").read_text())
    for record in (*state["buses"].values(), *state["returns"].values()):
        record.pop("uid")                                # a session from before identities
    (rig.project_dir / "session.json").write_text(json.dumps(state))
    first = daw_bridge.routing_uids(Session(rig.project_dir, read_only=True)._state)
    second = daw_bridge.routing_uids(Session(rig.project_dir, read_only=True)._state)
    assert first == second == {"bus:bus1": "bus1@0", "return:rv": "rv@0", "return:dly": "dly@0"}
    renamed = Session(rig.project_dir)
    renamed.rename_bus("bus1", "Drums")
    assert daw_bridge.routing_uids(renamed._state)["bus:Drums"] == "bus1@0"      # the record moved: so did it
    assert "routing_uids" in daw_bridge.chains_with_readout(rig.project_dir, light=True)


def test_the_daws_undo_co_edits_follow_the_engine_and_are_never_journaled(rig):
    rv, bus = rig.returns["rv"]["uid"], rig.buses["bus1"]["uid"]
    project = rig.project_dir
    journal = list(Session(project)._state.get("coedit_log", []))
    res = daw_bridge.apply_coedits(project, [
        f"/api/remove_routing?kind=return&id=rv&uid={rv}",
        f"/api/remove_routing?kind=bus&id=bus1&uid={bus}"])
    assert res["failed"] == 0
    gone = Session(project)
    assert "rv" not in gone.returns and "bus1" not in gone.buses and not _dangling(gone)
    res = daw_bridge.apply_coedits(project, [
        f"/api/restore_routing?kind=bus&id=bus1&uid={bus}",
        f"/api/restore_routing?kind=return&id=rv&uid={rv}",
        "/api/route_stem?stem=stem_b&bus=",                      # the engine has stem_b on the master
        "/api/route_stem?stem=stem_a&bus=bus%3Abus1",            # …and stem_a where it already is
        "/api/set_send?source=stem_a&return=rv&gain_db=-6.00"])
    assert res["failed"] == 0, res
    back = Session(project)
    assert back.stems["stem_a"]["bus"] == "bus1" and back.stems["stem_b"]["bus"] is None
    assert back.returns["rv"]["chain"][0]["decay_s"] == 3.3
    assert back.stems["stem_a"]["sends"]["rv"] == -6.0
    logged = [e["url"] for e in back._state.get("coedit_log", [])][len(journal):]
    assert logged == ["/api/set_send?source=stem_a&return=rv&gain_db=-6.00"]     # the control; not the structure


# ---- D8·R1: an undo leaves nothing pointing at what is not there ---------------------------

def test_undo_after_a_rename_keeps_the_member_in_the_group_the_user_renamed(demo_project):
    s = Session(demo_project)
    s.group_stems(["stem_a", "stem_b"])
    pid = _apply(s, [{"type": "eq", "stem_id": "stem_a",
                      "band": {"shape": "peak", "freq_hz": 3000.0, "gain_db": -2.0, "q": 1.0}}])
    s.rename_bus("bus1", "Rhythm")
    assert s.undo(pid).ok
    assert s.stems["stem_a"]["bus"] == "Rhythm"          # it was written back as "bus1": out of the group
    assert not _dangling(s)


def test_undo_after_the_user_removed_the_group_or_return_leaves_no_reference(rig):
    pid = _apply(rig, [{"type": "gain", "stem_id": "stem_a", "delta_db": -2.0}])
    rig.remove_bus("bus1")
    rig.remove_return("rv")
    assert rig.undo(pid).ok                              # the snapshot predates both removals
    assert not _dangling(rig), _dangling(rig)
    assert rig.group_stems(["stem_b"])["bus_id"] == "bus1"
    assert rig.stems["stem_a"]["bus"] is None            # …and the next ⌘G does not find stem_a inside it
    rid = rig.add_return("reverb")["return_id"]
    assert rid not in (rig.stems["stem_a"].get("sends") or {})


def test_undo_of_a_proposal_that_created_a_return_clears_the_send_the_user_made_into_it(demo_project):
    s = Session(demo_project)
    pid = _apply(s, [{"type": "return_reverb", "return_id": "plate"},
                     {"type": "send", "stem_id": "stem_a", "return_id": "plate", "gain_db": -14.0}])
    s.set_send("stem_b", "plate", -12.0)                 # the user turns another send knob up
    assert s.undo(pid).ok
    assert "plate" not in s.returns and not _dangling(s), _dangling(s)


def test_undo_of_a_proposal_that_created_a_group_clears_the_route_the_user_made_into_it(demo_project):
    s = Session(demo_project)
    pid = _apply(s, [{"type": "group", "bus_id": "rhythm", "members": ["stem_a"]}])
    s.route_stem("stem_b", "rhythm")
    assert s.undo(pid).ok
    assert "rhythm" not in s.buses and s.stems["stem_b"]["bus"] is None and not _dangling(s)


# ---- Undo Agent Move carries the DAW's reconcile home -------------------------------------

def test_undo_agent_move_takes_back_the_return_the_agent_added(demo_project):
    """"add a plate and send the vocal to it", then ⌥⌘Z. The brain asks the DAW for edit.undo;
    the DAW's undo takes the return's track away and says so in the reply's co-edits; the brain
    follows. It used to answer "undid the run (0 moves) · its timeline edit was left alone"
    with the return and the send still there."""
    s = Session(demo_project)
    rid = s.add_return("reverb")["return_id"]
    s.set_send("stem_a", rid, -12.0)
    uid = Session(demo_project).returns[rid]["uid"]
    brain = server.Brain(demo_project)
    asked: list[str] = []

    def daw(command, args=None, *a, **k):
        asked.append(command)
        return {"ok": True, "coedits": [f"/api/remove_routing?kind=return&id={rid}&uid={uid}"]}

    brain.daw_invoke = daw
    brain._run_marks.append(server.max_applied_seq(demo_project))
    brain._run_gestures.append("g1")
    brain._run_daw_ops.append(2)
    out = brain.undo("")
    assert asked == ["edit.undo"]
    assert out["daw_ops_undone"] == 2 and not out["daw_ops_pending"]
    after = Session(demo_project)
    assert rid not in after.returns and after.stems["stem_a"]["sends"] == {} and not _dangling(after)
    assert all(r["id"] != rid for r in out["send_returns"]["returns"])
