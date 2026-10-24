"""Gates for the pre-merge review of 2026-10-06 (audit-fixes -> main): the Python halves.

Thirty-nine findings in code the branch itself had written; the C++ halves are console legs
(daw/Source/PremergeReviewLegs.h). Each test drives the door the product uses and carries its
control, and every one was run on the unfixed tree first. The record is
docs/audits/full-2026-10-02/readers/P2-fixes.md (Pre-merge review, 2026-10-06).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from mix_engine.models import ProposalRequest
from mix_engine.session import Session
from orchestrator import daw_bridge as B
from orchestrator import daw_server as S

REPO = Path(__file__).resolve().parents[2]


def _state(project: Path) -> dict:
    return json.loads((project / "session.json").read_text())


def _propose(session: Session, actions: list[dict]):
    return session.propose(ProposalRequest(rationale="premerge", predicted_impact=[], actions=actions))


def _apply(session: Session, actions: list[dict]) -> None:
    result = _propose(session, actions)
    assert result.accepted, result.violations
    assert session.apply(result.proposal_id).ok


def _four_stems(tmp_path: Path) -> Path:
    import numpy as np
    import soundfile as sf

    proj = tmp_path / "four"
    proj.mkdir(parents=True)
    t = np.arange(2 * 44100) / 44100
    for name, hz in (("drums", 110), ("bassline", 55), ("lead", 440), ("pads", 220)):
        sf.write(proj / f"{name}.wav", (0.2 * np.sin(2 * np.pi * hz * t)).astype("float32"), 44100)
    Session(proj)
    return proj


def _agent_turn(brain, urls: list[str], floor: int | None = None, undo_reply: dict | None = None):
    """One agent turn whose commands queue `urls` as their own co-edits, mirrored as chat does."""
    floor = B.max_applied_seq(brain.project_dir) if floor is None else floor
    it = iter(urls)
    brain.daw_invoke = lambda cmd, args=None, *a, **k: (
        dict(undo_reply or {"ok": True, "timeline": "none"}) if cmd == "edit.undo"
        else (lambda u: {"ok": True, "coedits": [u], "coedits_own": [u]})(next(it)))
    brain.begin_agent_turn()
    invoke = brain.counting_invoke()
    for _ in urls:
        brain._mirror_coedits(list(invoke("level", {})["coedits"]))
    brain._close_gesture(floor)


# ====================================================================================
# Alt-Cmd-Z and the levels a turn set (batch T's level undo)
# ====================================================================================

def test_the_master_fader_has_an_inverse_at_both_of_its_doors(demo_project):
    """`track.set_gain {track: master}` queues set_stem, the dock's Gain and `device.set` queue
    chain_param: the first had no inverse at all and the second's was always 0.0 dB - the
    stored master chain has no `gain` entry, the fader is `master_gain_db`."""
    Session(demo_project).user_edit("master", {"gain_db": -4.0})
    by_stem = B.level_inverses(demo_project, ["/api/set_stem?stem=master&gain_db=-1.00"])
    by_chain = B.level_inverses(
        demo_project, ["/api/chain_param?stem=master&device_type=gain&occurrence=0&value=-1.0000"])
    assert len(by_stem) == 1 and "gain_db=-4.0" in by_stem[0][1], by_stem
    assert len(by_chain) == 1 and "value=-4.0" in by_chain[0][1], by_chain
    # control: a stem's fader, as before
    assert "gain_db=0.0" in B.level_inverses(demo_project, ["/api/set_stem?stem=stem_a&gain_db=-1.0"])[0][1]


def test_a_returns_strip_has_an_inverse(demo_project):
    rid = Session(demo_project).add_return("blank")["return_id"]
    Session(demo_project).user_edit(f"return:{rid}", {"pan": -0.25})
    back = B.level_inverses(demo_project, [f"/api/set_stem?stem=return%3A{rid}&pan=0.5&muted=true"])
    urls = sorted(url for _control, url in back)
    assert len(urls) == 2 and "muted=false" in urls[0] and "pan=-0.25" in urls[1], urls


@pytest.mark.parametrize("url", ["/api/chain_param?stem=master&device_type=gain&occurrence=0&value=-1.0000",
                                 "/api/set_stem?stem=master&gain_db=-1.00"])
def test_alt_cmd_z_puts_the_master_fader_back_where_it_was(demo_project, url):
    Session(demo_project).user_edit("master", {"gain_db": -12.0})
    brain = S.Brain(demo_project)
    _agent_turn(brain, [url])
    assert _state(demo_project)["master_gain_db"] == -1.0
    brain.undo()
    assert _state(demo_project)["master_gain_db"] == -12.0, "not 0.0, and not left at the agent's"


def test_a_proposal_and_then_a_command_on_one_fader_undo_to_before_the_turn(demo_project):
    """The inverse was read at the command's first touch - after the turn's own proposal had
    moved the fader - and written back after the proposal was undone: the fader ended on the
    proposal's value with the ledger saying the proposal was gone."""
    Session(demo_project)
    brain = S.Brain(demo_project)
    url = "/api/set_stem?stem=stem_a&gain_db=-3.00"
    brain.daw_invoke = lambda cmd, args=None, *a, **k: (
        {"ok": True, "timeline": "none"} if cmd == "edit.undo"
        else {"ok": True, "coedits": [url], "coedits_own": [url]})
    brain.begin_agent_turn()
    floor = B.max_applied_seq(demo_project)
    _apply(Session(demo_project), [{"type": "gain", "stem_id": "stem_a", "delta_db": 2.0}])
    brain._mirror_coedits(list(brain.counting_invoke()("track.set_gain", {})["coedits"]))
    brain._close_gesture(floor)
    assert _state(demo_project)["stems"]["stem_a"]["gain_db"] == -3.0
    brain.undo()
    assert _state(demo_project)["stems"]["stem_a"].get("gain_db", 0.0) == 0.0


def test_alt_cmd_z_leaves_a_level_the_user_moved_since(demo_project):
    """James, 2026-10-06 ("fix em all", the review's open decision taken as the proposal half's
    rule): a control the user has moved since the turn is the user's."""
    Session(demo_project).user_edit("stem_a", {"gain_db": -2.0})
    Session(demo_project).user_edit("stem_b", {"gain_db": -2.0})
    brain = S.Brain(demo_project)
    _agent_turn(brain, ["/api/set_stem?stem=stem_a&gain_db=-9.00", "/api/set_stem?stem=stem_b&gain_db=-9.00"])
    brain.coedit(["/api/set_stem?stem=stem_a&gain_db=-5.00"])          # the user's own, afterwards
    out = brain.undo()
    stems = _state(demo_project)["stems"]
    assert stems["stem_a"]["gain_db"] == -5.0, "the user's later move was written over"
    assert stems["stem_b"]["gain_db"] == -2.0, "control: a fader nobody touched since goes back"
    assert out["undone"]["levels"] == 1 and out["undone"].get("levels_left") == 1


def test_a_command_that_timed_out_still_leaves_an_undo_mark(demo_project):
    """`the DAW gave no answer in 30 s ... it may already have run` is not a refusal: counted
    as one, a turn whose bounce did land left no mark, and the next Alt-Cmd-Z popped the turn
    before it."""
    Session(demo_project)
    brain = S.Brain(demo_project)
    marks = len(brain._run_marks)

    def turn(reply):
        brain.daw_invoke = lambda cmd, args=None, *a, **k: dict(reply)
        brain.begin_agent_turn()
        floor = B.max_applied_seq(demo_project)
        brain.counting_invoke()("track.bounce", {"track": "stem_a"})
        brain._close_gesture(floor)

    turn({"ok": False, "error": "the DAW gave no answer in 30 s", "may_have_run": True})
    assert len(brain._run_marks) == marks + 1
    turn({"ok": False, "error": "no track nope"})                       # control: a refusal changed nothing
    assert len(brain._run_marks) == marks + 1


def test_the_timeout_reply_says_it_may_have_run():
    source = (REPO / "orchestrator" / "daw_server.py").read_text()
    at = source.index("the DAW gave no answer in")
    assert '"may_have_run": True' in source[at - 200:at + 400], \
        "daw_invoke's own timeout must be told apart from a refusal"


# ====================================================================================
# The A/B answer for a level
# ====================================================================================

def test_an_a_on_a_lone_level_move_is_not_put_back_with_edit_undo():
    """A fader, a pan, a mute, a send are not steps of the timeline's undo: `edit.undo` took
    back whatever was on top of it and left the level standing."""
    for command in ("track.set_gain", "track.set_pan", "track.set_mute", "send.set", "return.set_level"):
        note = B.preference_note({"choice": "A", "about": "the level", "commands": [command],
                                  "command_alone": True})
        assert "NOT edit.undo" in note and "same command" in note, (command, note)
    # control: a timeline command that was its turn's only one is exactly one undo
    note = B.preference_note({"choice": "A", "about": "the cut", "commands": ["clip.split"],
                              "command_alone": True})
    assert "edit.undo" in note and "NOT edit.undo" not in note


# ====================================================================================
# Guardrails and the ledger (batch Py1)
# ====================================================================================

def test_a_cut_that_replaced_a_utility_is_not_credit_once_the_user_removed_it(demo_project):
    """D9.R5 on the REPLACE path: the entry an action made was found by `id()` of the chain's
    dicts, and the dict a replace frees hands its address to the new one - so the record did
    not know its own entry and the removed cut stayed credit for good."""
    session = Session(demo_project)
    session.user_edit("stem_a", {"gain_db": -30.0})
    _apply(session, [{"type": "utility", "stem_id": "stem_a", "gain_db": 0.0, "width": 0.9}])
    session.advance_pass()
    _apply(session, [{"type": "utility", "stem_id": "stem_a", "gain_db": -6.0}])
    session.advance_pass()
    assert isinstance(_state(demo_project)["proposals"][-1]["entry_uids"][0], int)
    B.apply_coedits(demo_project, ["/api/remove_device?stem=stem_a&device_type=utility&occurrence=0"])
    session = Session(demo_project)
    landed = 0
    for _ in range(4):
        result = _propose(session, [{"type": "gain", "stem_id": "stem_a", "delta_db": 6.0}])
        landed += bool(result.accepted and session.apply(result.proposal_id).ok)
        session.advance_pass()
    assert landed == 2, "the session rail is +12: a third +6 landed on the removed cut's credit"


def test_a_utility_turned_down_lands_on_a_path_already_past_the_rail(tmp_path):
    """`before` was the wallet with the replaced singleton taken out, so any positive
    replacement read as a push: +6 -> +3 on the track was refused as "+9 with the group"."""
    proj = _four_stems(tmp_path)
    Session(proj).group_stems(["drums", "bassline"])
    session = Session(proj)
    for target in ("lead", "bus1"):
        session.user_edit(target, {"gain_db": -30.0})
    _apply(session, [{"type": "utility", "stem_id": "lead", "gain_db": 6.0},
                     {"type": "gain", "stem_id": "bus1", "delta_db": 6.0}])
    Session(proj).route_stem("lead", "bus1")
    session = Session(proj)
    assert _propose(session, [{"type": "utility", "stem_id": "lead", "gain_db": 3.0}]).accepted
    # control: the same replacement turned UP is still the push it is
    assert not _propose(session, [{"type": "utility", "stem_id": "lead", "gain_db": 7.0}]).accepted


def test_a_master_width_opened_up_is_not_refused_as_narrowing(demo_project):
    session = Session(demo_project)
    _apply(session, [{"type": "width", "stem_id": "master", "width": 0.5}])
    B.apply_coedits(demo_project, ["/api/add_device?stem=master&device_type=utility"])
    B.apply_coedits(demo_project, ["/api/device_param?stem=master&device_type=utility&occurrence=0&param=width&value=0.9"])
    session = Session(demo_project)
    assert any(e.get("type") == "utility" and abs(e.get("width", 1.0) - 0.9) < 1e-6
               for e in session._state["master_chain"]), "setup: the user's utility is on the master"
    opened = _propose(session, [{"type": "width", "stem_id": "master", "width": 0.55}])
    assert "master_width_min" not in [v.rule for v in opened.violations], opened.violations
    closed = _propose(session, [{"type": "width", "stem_id": "master", "width": 0.45}])   # control
    assert "master_width_min" in [v.rule for v in closed.violations]


def test_a_pan_or_a_mute_on_a_frozen_stem_is_the_agents(demo_project):
    """The DAW keeps mute, solo and pan live on a frozen track (M16); the brain refused both
    with "nothing that is heard", which is false for them."""
    B.apply_coedits(demo_project, ["/api/set_stem?stem=stem_a&frozen=stem_a"])
    session = Session(demo_project)
    for action in ({"type": "pan", "stem_id": "stem_a", "pan": 0.3},
                   {"type": "mute", "stem_id": "stem_a", "muted": True}):
        rules = [v.rule for v in _propose(session, [action]).violations]
        assert "frozen_track" not in rules, (action["type"], rules)
    rules = [v.rule for v in _propose(session, [{"type": "gain", "stem_id": "stem_a", "delta_db": -1.0}]).violations]
    assert rules == ["frozen_track"]                                     # control: its level is locked


# ====================================================================================
# Owed co-edits
# ====================================================================================

ADD = "/api/add_device?stem=stem_a&device_type=return_delay"


def test_a_batch_the_daw_gave_up_on_is_not_applied_behind_its_back(demo_project):
    """The 1.5 s flush at a quit times out behind a render, the DAW parks the batch for the
    next open and goes - and the brain, when the lock came free, applied it anyway: twice."""
    Session(demo_project)
    brain = S.Brain(demo_project)
    skipped = S.coedit_unless_gone(brain, [ADD], gone=lambda: True)
    assert skipped.get("ok") is False and "gone" in str(skipped.get("skipped"))
    assert not [e for e in _state(demo_project)["stems"]["stem_a"]["chain"] if e["type"] == "return_delay"]
    applied = S.coedit_unless_gone(brain, [ADD], gone=lambda: False)     # control
    assert applied.get("applied") == 1
    assert len([e for e in _state(demo_project)["stems"]["stem_a"]["chain"] if e["type"] == "return_delay"]) == 1


def test_owed_moves_wait_for_a_turn_that_is_still_winding_up(demo_project):
    from mix_engine.session import OWED_COEDITS_FILE

    Session(demo_project)
    brain = S.Brain(demo_project)
    owed = demo_project / OWED_COEDITS_FILE
    owed.write_text(json.dumps(["/api/set_stem?stem=stem_a&gain_db=-7.00"]))
    brain.run_active = True
    brain.apply_owed()
    assert owed.exists() and _state(demo_project)["stems"]["stem_a"].get("gain_db", 0.0) == 0.0
    brain.run_active = False
    brain.apply_owed()                                                   # control: a quiet open applies it, once
    assert not owed.exists() and _state(demo_project)["stems"]["stem_a"]["gain_db"] == -7.0


# ---- a flush the DAW got no answer for is sent again under the SAME id ----

def _delays(project: Path) -> int:
    return len([e for e in _state(project)["stems"]["stem_a"]["chain"] if e["type"] == "return_delay"])


def test_a_batch_sent_twice_is_applied_once(demo_project):
    """The 30 s flush times out behind a long render with the DAW still connected; the brain
    applies the first send when the lock comes free, and the retry applied it again."""
    Session(demo_project)
    brain = S.Brain(demo_project)
    first = S.coedit_unless_gone(brain, [ADD], lambda: False, "batch-1")
    again = S.coedit_unless_gone(brain, [ADD], lambda: False, "batch-1")
    assert first.get("applied") == 1 and again.get("already") is True and again.get("applied") == 0
    assert _delays(demo_project) == 1
    other = S.coedit_unless_gone(brain, [ADD], lambda: False, "batch-2")     # control: another batch is applied
    assert other.get("applied") == 1 and _delays(demo_project) == 2


def test_the_retry_rides_the_next_flush_as_a_sent_group(demo_project):
    """What the DAW sends after the timeout: the unanswered batch as a group under its own
    id, ahead of the moves made since. The group is dropped when its batch landed."""
    Session(demo_project)
    brain = S.Brain(demo_project)
    brain.coedit([ADD], "batch-1")                                           # the late first send
    res = brain.coedit([{"batch": "batch-1", "urls": [ADD]},
                        "/api/set_stem?stem=stem_a&gain_db=-4.00"], "batch-2")
    assert res.get("applied") == 1 and _delays(demo_project) == 1
    assert _state(demo_project)["stems"]["stem_a"]["gain_db"] == -4.0
    # ...and in the other order - the retry gets there first, the first send comes in late.
    brain.coedit([{"batch": "batch-9", "urls": [ADD]}], "batch-10")
    late = brain.coedit([ADD], "batch-9")
    assert late.get("already") is True and _delays(demo_project) == 2


def test_a_group_whose_batch_never_landed_is_applied(demo_project):
    Session(demo_project)
    brain = S.Brain(demo_project)
    res = brain.coedit([{"batch": "lost", "urls": [ADD]}], "next")
    assert res.get("applied") == 1 and _delays(demo_project) == 1


def test_a_new_brain_knows_the_batches_the_last_one_applied(demo_project):
    """The record is in session.json: the brain that applied the batch can be gone."""
    Session(demo_project)
    S.Brain(demo_project).coedit([ADD], "batch-1")
    assert "batch-1" in _state(demo_project)["coedit_batches"]
    fresh = S.Brain(demo_project)
    assert fresh.coedit([ADD], "batch-1").get("already") is True
    assert fresh.coedit([{"batch": "batch-1", "urls": [ADD]}], "batch-2").get("applied") == 0
    assert _delays(demo_project) == 1


def test_a_replys_queue_is_resolved_before_anything_mirrors_it(demo_project):
    Session(demo_project)
    brain = S.Brain(demo_project)
    brain.coedit([ADD], "batch-1")
    urls = brain.resolve_sent([{"batch": "batch-1", "urls": [ADD]},
                               {"batch": "batch-2", "urls": ["/api/set_stem?stem=stem_a&pan=0.200"]},
                               "/api/set_stem?stem=stem_a&gain_db=-2.00"])
    assert urls == ["/api/set_stem?stem=stem_a&pan=0.200", "/api/set_stem?stem=stem_a&gain_db=-2.00"]
    assert brain.coedit(["/api/set_stem?stem=stem_a&pan=0.200"], "batch-2").get("already") is True
    src = (REPO / "orchestrator" / "daw_server.py").read_text()
    body = src[src.index("    def daw_invoke(command: str"):src.index("    # AUTH FIRST, or nothing at all")]
    assert "conn_brain.resolve_sent(res[\"coedits\"])" in body      # the one place a DAW reply enters


def test_owed_and_checkpointed_moves_carry_their_groups(demo_project):
    from mix_engine.session import (DISCARD_MARKER_FILE, OWED_COEDITS_FILE, SAVED_COEDITS_FILE,
                                    SAVED_SESSION_FILE)

    Session(demo_project)
    brain = S.Brain(demo_project)
    brain.coedit([ADD], "batch-1")
    # parked at a quit: the batch that landed is dropped, the one that did not is applied
    (demo_project / OWED_COEDITS_FILE).write_text(json.dumps(
        [{"batch": "batch-1", "urls": [ADD]}, {"batch": "batch-2", "urls": [ADD]},
         "/api/set_stem?stem=stem_a&gain_db=-7.00"]))
    done = B.apply_owed_coedits(demo_project)
    assert done["applied"] == 2 and _delays(demo_project) == 2
    assert _state(demo_project)["stems"]["stem_a"]["gain_db"] == -7.0
    # "Don't Save": the checkpoint is judged by the batches IT holds, not by what came after
    (demo_project / SAVED_SESSION_FILE).write_text((demo_project / "session.json").read_text())
    (demo_project / SAVED_COEDITS_FILE).write_text(json.dumps(
        [{"batch": "batch-2", "urls": [ADD]}, {"batch": "batch-3", "urls": [ADD]}]))
    brain.coedit([ADD], "batch-3")                    # lands AFTER the save: it is declined work
    assert _delays(demo_project) == 3
    (demo_project / DISCARD_MARKER_FILE).write_text("declined\n")
    brain.honour_discard()
    assert _delays(demo_project) == 3                 # the checkpoint's two + batch-3 from the save's queue
    assert brain.coedit([ADD], "batch-3").get("already") is True


# ====================================================================================
# Auto Mix says why it did less, at every door
# ====================================================================================

def test_every_auto_mix_door_passes_the_notes_and_what_was_left_out():
    """Only the button's door (Brain.auto_mix) handed `notes=` and `left_out=`; the chat tool
    and an approved plan's step dropped "nothing here was split" and "the de-noise was left
    out" on the floor."""
    calls = []
    # (the chat tool is `turn_host.LiveTurn.do_auto_mix` since Phase 55)
    for name in ("daw_bridge.py", "daw_server.py", "turn_host.py"):
        source = (REPO / "orchestrator" / name).read_text()
        for m in re.finditer(r"functools\.partial\((?:B\.)?auto_mix_pass,|auto_mix_pass\(self\.project_dir", source):
            calls.append((name, source[m.start():m.start() + 420]))
    assert len(calls) == 3, [c[0] for c in calls]
    for name, call in calls:
        assert "notes=" in call and "left_out=" in call, (name, call[:200])


def test_what_auto_mix_says_carries_the_reason():
    said = B.auto_mix_said(0, refused=[], notes=["nothing in this project was split"],
                           left_out=["vocals"], stopped=False)
    assert "nothing in this project was split" in said and "vocals" in said
    assert B.auto_mix_said(2, refused=[], notes=[], left_out=[], stopped=False) \
        == "Auto Mix applied 2 bundles (undoable)"


# ====================================================================================
# The engine clone's check, on a machine whose git has opinions about diffs
# ====================================================================================

@pytest.mark.parametrize("key,value", [("diff.context", "5"), ("diff.algorithm", "histogram"),
                                       ("diff.interHunkContext", "4")])
def test_the_engine_check_does_not_depend_on_this_machines_git_config(key, value):
    script = REPO / "daw" / "scripts" / "fetch_engine.sh"
    clone = REPO / "daw" / "third_party" / "tracktion_engine"
    if not (clone / ".git").exists():
        pytest.skip("no engine clone on this machine")
    env = {**os.environ, "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": key, "GIT_CONFIG_VALUE_0": value}
    done = subprocess.run([str(script), "--check"], cwd=REPO, env=env, capture_output=True, text=True)
    assert done.returncode == 0 and "is the recorded engine" in done.stdout + done.stderr, \
        (done.stdout + done.stderr)[-600:]


# ====================================================================================
# The Dynamic EQ's six bands, at the brain's own door for a user's add
# ====================================================================================

def test_the_brain_refuses_a_seventh_dynamic_eq_band_like_the_dock(demo_project):
    Session(demo_project)
    add = "/api/add_device?stem=master&device_type=dynamic_eq"
    for _ in range(6):
        assert B.apply_coedits(demo_project, [add]).get("applied") == 1
    seventh = B.apply_coedits(demo_project, [add])
    assert seventh.get("applied") == 0 and seventh.get("failed") == 1, seventh
    assert len([e for e in _state(demo_project)["master_chain"] if e["type"] == "dynamic_eq"]) == 6
