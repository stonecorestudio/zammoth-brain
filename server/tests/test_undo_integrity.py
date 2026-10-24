"""Undo integrity — the brain half of the 2026-09-10 agent audit's P1-1 to P1-5.

Each test here failed against the code before the fix:

- P1-1: ⌥⌘Z replayed EVERY journaled co-edit after its floor, onto every target. add_device
  appends, so each undo stacked another copy of a device the user had added.
- P1-2: a run, chat turn or approved plan that raised after landing moves pushed no undo
  mark, so the next ⌥⌘Z also took back the gesture before it.
- P1-3: picking A on the agent's A/B told the model "put the move back with edit.undo",
  which never reaches a proposal.
- P1-4: "undo run" carried no identity; Undo popped the newest mark whatever the verb sat
  under.
- P1-5: the DAW swaps its co-edit queue into every command reply, and only chat mirrored
  it — an approved plan and the run planner dropped the fader moves, the user's included,
  and a plan's return.add never reached session.json.
"""
import json
from pathlib import Path

import pytest

from mix_engine.models import ProposalRequest
from mix_engine.session import Session
from orchestrator import daw_bridge, daw_server
from orchestrator.daw_bridge import apply_coedits, max_applied_seq, undo_run
from orchestrator.daw_server import Brain

URL = "http://127.0.0.1:1/api/"


def _gain(project: Path, stem: str, db: float) -> str:
    session = Session(project)
    result = session.propose(ProposalRequest(
        rationale=f"{stem} {db:+}",
        predicted_impact=[{"metric": f"stem({stem}).lufs_s", "predicted_delta": db,
                           "unit": "LUFS"}],
        actions=[{"type": "gain", "stem_id": stem, "delta_db": db}]))
    assert result.accepted, result.violations
    assert session.apply(result.proposal_id).ok
    return result.proposal_id


def _state(project: Path) -> dict:
    return json.loads((project / "session.json").read_text())


def _eqs(project: Path, stem: str) -> int:
    return sum(1 for e in _state(project)["stems"][stem].get("chain", [])
               if e.get("type") == "eq")


def _gesture(brain: Brain, stem: str, db: float) -> str:
    before = max_applied_seq(brain.project_dir)
    _gain(brain.project_dir, stem, db)
    brain._close_gesture(before)
    return brain.last_gesture


# ------------------------------------------------------------------ P1-1: co-edit replay

def test_undo_does_not_duplicate_devices_the_user_added_between_gestures(demo_project):
    _gain(demo_project, "stem_a", -1.0)                    # gesture 1
    floor = max_applied_seq(demo_project)
    assert apply_coedits(demo_project, [                   # the user's hands, in between
        URL + "add_device?stem=stem_a&device_type=eq",
        URL + "add_device?stem=stem_b&device_type=eq"])["applied"] == 2
    _gain(demo_project, "stem_a", -1.0)                    # gesture 2, on stem_a again
    assert (_eqs(demo_project, "stem_a"), _eqs(demo_project, "stem_b")) == (1, 1)

    assert undo_run(demo_project, floor) == 1
    # stem_a's restored snapshot already holds the EQ; stem_b was never restored at all
    assert (_eqs(demo_project, "stem_a"), _eqs(demo_project, "stem_b")) == (1, 1)
    assert undo_run(demo_project, 0) == 1
    assert (_eqs(demo_project, "stem_a"), _eqs(demo_project, "stem_b")) == (1, 1)
    assert _state(demo_project)["stems"]["stem_a"]["gain_db"] == 0.0


def test_a_device_added_after_the_gesture_survives_its_undo_exactly_once(demo_project):
    floor = max_applied_seq(demo_project)
    _gain(demo_project, "stem_a", -2.0)
    assert apply_coedits(demo_project, [
        URL + "add_device?stem=stem_a&device_type=eq"])["applied"] == 1
    assert undo_run(demo_project, floor) == 1
    assert _eqs(demo_project, "stem_a") == 1, "the restore erased it; the replay puts it back once"
    assert _state(demo_project)["stems"]["stem_a"]["gain_db"] == 0.0


def test_the_agents_own_undo_proposal_keeps_the_users_later_tweak(demo_project):
    pid = _gain(demo_project, "stem_a", 2.0)
    assert apply_coedits(demo_project, [URL + "set_stem?stem=stem_a&pan=-0.4"])["applied"] == 1
    assert daw_bridge._undo_one(demo_project, pid) == (pid, "")
    stem = _state(demo_project)["stems"]["stem_a"]
    assert stem["gain_db"] == 0.0 and stem["pan"] == -0.4


# ------------------------------------------------------------------ P1-2: failed gestures

@pytest.mark.parametrize("path", ["run", "chat", "apply_plan"])
def test_a_gesture_that_fails_after_landing_moves_is_still_one_undo(path, demo_project,
                                                                   monkeypatch):
    brain = Brain(demo_project)
    _gesture(brain, "stem_b", -1.0)                        # an earlier, finished gesture

    def fail(project_dir, *a, **kw):
        _gain(Path(project_dir), "stem_a", -1.0)
        raise RuntimeError("usage limit reached")
    monkeypatch.setattr(daw_server, {"run": "run_agent_pass", "chat": "chat_turn",
                                     "apply_plan": "apply_plan"}[path], fail)
    call = {"run": lambda: brain.run("make it punchier", use_llm=False),
            "chat": lambda: brain.chat("the vocal down 1 dB", use_llm=True),
            "apply_plan": lambda: brain.apply_plan({"steps": []})}[path]
    with pytest.raises(RuntimeError, match=r"1 move\(s\) had already applied"):
        call()
    assert len(brain._run_marks) == 2, "the failed gesture closes as its own mark"

    assert brain.undo()["undone"]["proposals"] == 1
    stems = _state(demo_project)["stems"]
    assert stems["stem_a"]["gain_db"] == 0.0
    assert stems["stem_b"]["gain_db"] == -1.0, "the gesture before it stands"


# ------------------------------------------------------------------ P1-3: A/B routing

def test_choosing_A_names_the_undo_that_actually_reaches_the_move(demo_project):
    from orchestrator.chat import PREFERENCE_OPTIONS
    from orchestrator.daw_bridge import (offer_preference, preference_note,
                                         record_preference_answer)

    # ONE move per A/B (reader E6·R6): a proposal is put back by its id, a command by ⌘Z
    # only when it was its turn's only command.
    pid = _gain(demo_project, "stem_a", 1.0)
    offer_preference(demo_project, about="the brighter vocal", move={"proposal_id": pid})
    note = preference_note(record_preference_answer(demo_project, PREFERENCE_OPTIONS[0]))
    assert f"undo_proposal {pid}" in note and "track.set_gain" not in note

    offer_preference(demo_project, about="the cut",
                     move={"command": "clip.split", "alone": True})
    note = preference_note(record_preference_answer(demo_project, PREFERENCE_OPTIONS[0]))
    assert "edit.undo for the DAW command clip.split" in note

    # ...and never for a LEVEL (pre-merge review 2026-10-06): a fader is not an undo step, so
    # edit.undo would take back whatever timeline edit is on top instead.
    offer_preference(demo_project, about="the fader",
                     move={"command": "track.set_gain", "alone": True})
    note = preference_note(record_preference_answer(demo_project, PREFERENCE_OPTIONS[0]))
    assert "edit.undo for the DAW command" not in note and "NOT edit.undo" in note

    offer_preference(demo_project, about="x", move={"proposal_id": pid})
    assert "keep it" in preference_note(
        record_preference_answer(demo_project, PREFERENCE_OPTIONS[1]))


# ------------------------------------------------------------------ P1-4: gesture identity

def test_undo_run_takes_back_only_the_gesture_it_names(demo_project):
    brain = Brain(demo_project)
    g1 = _gesture(brain, "stem_a", -1.0)
    g2 = _gesture(brain, "stem_b", -1.0)
    assert g1 and g2 and g1 != g2

    refused = brain.undo(g1)                               # the OLDER run's verb
    assert refused["ok"] is False and "no longer the newest" in refused["refused"]
    stems = _state(demo_project)["stems"]
    assert (stems["stem_a"]["gain_db"], stems["stem_b"]["gain_db"]) == (-1.0, -1.0)
    assert len(brain._run_marks) == 2, "a refusal pops nothing"

    assert brain.undo(g2)["undone"]["gesture"] == g2
    assert brain.undo(g1)["undone"]["gesture"] == g1       # now it IS the newest
    stems = _state(demo_project)["stems"]
    assert (stems["stem_a"]["gain_db"], stems["stem_b"]["gain_db"]) == (0.0, 0.0)


@pytest.mark.parametrize("path", ["run", "chat", "apply_plan", "auto_mix"])
def test_every_gesture_result_names_its_undo_mark(path, demo_project, monkeypatch):
    """The DAW's ledger shows "undo run" only when the result names a gesture, so a path
    that forgets it silently loses its undo verb."""
    Session(demo_project)
    brain = Brain(demo_project)

    def lands(project_dir, *a, **kw):
        _gain(Path(project_dir), "stem_a", -1.0)
        return {"run": {"stop_reason": "done", "passes": 1, "cost_usd": 0.0,
                        "budget_usd": None},
                "chat": {"reply": "done"}, "apply_plan": {"ok": True, "results": []},
                "auto_mix": None}[path]
    monkeypatch.setattr(daw_server, {"run": "run_agent_pass", "chat": "chat_turn",
                                     "apply_plan": "apply_plan",
                                     "auto_mix": "auto_mix_pass"}[path], lands)
    result = {"run": lambda: brain.run("make it punchier", use_llm=False),
              "chat": lambda: brain.chat("the vocal down 1 dB", use_llm=True),
              "apply_plan": lambda: brain.apply_plan({"steps": []}),
              "auto_mix": lambda: brain.auto_mix("")}[path]()
    assert result.get("gesture") and result["gesture"] == brain._run_gestures[-1]


def test_a_command_only_turn_makes_the_older_verb_stale(demo_project):
    """The audit's repro: a run lands, then "turn the kick down" (a daw_command turn — no
    ledger, so the panel never retired the run's verb), then the run's "undo run"."""
    brain = Brain(demo_project)
    g_run = _gesture(brain, "stem_a", -1.0)
    brain._turn_daw_ops = 1                                # the daw_command turn
    brain._close_gesture(max_applied_seq(demo_project))
    assert brain.undo(g_run)["ok"] is False
    assert _state(demo_project)["stems"]["stem_a"]["gain_db"] == -1.0


# ------------------------------------------------------------------ P1-5: reply co-edits

class FakeDaw:
    """The App.cpp reply contract: every command reply carries the co-edit queue, swapped out."""

    def __init__(self, coedits=()):
        self.calls: list[str] = []
        self.queue = list(coedits)

    def __call__(self, command, args=None, *a, **kw):
        self.calls.append(command)
        out = {"ok": True}
        if self.queue:
            out["coedits"], self.queue = self.queue, []
        return out


def test_an_approved_plan_mirrors_the_coedits_its_command_replies_carry(demo_project):
    brain = Brain(demo_project)
    brain.daw_invoke = FakeDaw([URL + "set_stem?stem=stem_a&gain_db=-3.0",
                                URL + "set_stem?stem=stem_b&gain_db=-4.0"])   # + the user's move
    brain.apply_plan({"steps": [{"kind": "command", "payload": {
        "command": "track.set_gain", "args": {"track": "stem_a", "gain_db": -3.0}}}]})
    stems = _state(demo_project)["stems"]
    assert stems["stem_a"]["gain_db"] == -3.0
    assert stems["stem_b"]["gain_db"] == -4.0


def test_an_approved_plan_performs_routing_brain_side(demo_project):
    brain = Brain(demo_project)
    daw = FakeDaw()
    brain.daw_invoke = daw
    brain.apply_plan({"steps": [{"kind": "command", "payload": {
        "command": "return.add", "args": {"kind": "blank"}}}]})
    assert _state(demo_project).get("returns"), "the plan's return exists in session.json"
    # `coedits.take` first since 2026-10-03 (reader S4·R3): the push that follows is laid over
    # every node, so the user's queued moves are taken before it is resolved.
    assert daw.calls == ["coedits.take", "routing.apply"], \
        "the DAW realizes it; it never queues a return.add"


def test_the_run_planners_commands_mirror_their_coedits(demo_project, monkeypatch):
    Session(demo_project)                                  # the stems exist, so run() runs
    brain = Brain(demo_project)
    brain.daw_invoke = FakeDaw([URL + "set_stem?stem=stem_b&gain_db=-4.0"])
    seen = {}

    def fake_pass(project_dir, brief, *, daw_invoke=None, **kw):
        seen["reply"] = daw_invoke("track.set_gain", {"track": "stem_b", "gain_db": -4.0})
        _gain(Path(project_dir), "stem_a", -1.0)
        return {"stop_reason": "done", "passes": 1, "cost_usd": 0.0, "budget_usd": None}
    monkeypatch.setattr(daw_server, "run_agent_pass", fake_pass)

    result = brain.run("make it punchier", use_llm=False)
    assert _state(demo_project)["stems"]["stem_b"]["gain_db"] == -4.0
    assert "coedits" not in seen["reply"], "the model never sees raw co-edit URLs"
    assert result["gesture"] == brain._run_gestures[-1]


def test_a_routing_apply_reply_mirrors_its_coedits_too(demo_project):
    """Brain-performed routing asks the DAW to realize it with `routing.apply`, and that
    reply carries the co-edit queue like any other — chat's path dropped it."""
    from orchestrator.daw_bridge import perform_routing

    queued = URL + "set_stem?stem=stem_b&gain_db=-4.0"
    daw, mirrored = FakeDaw([queued]), []
    out = perform_routing("return.add", {}, {"return.add": lambda a: {"return_id": "r1"}},
                          daw, mirrored.extend)
    assert out["ok"] and out["return_id"] == "r1" and daw.calls == ["routing.apply"]
    assert mirrored == [queued]
