"""Agent Undo in the DAW: one Undo takes back exactly one agent gesture.

Found by the 2026-08-05 audit. The undo boundary was recorded only when a gesture
applied a PROPOSAL, but the agent's ordinary move in the DAW is a `daw_command`
("delete the last 8 bars"), which applies none. Those gestures recorded nothing,
so the next Undo popped an empty stack and fell through to `floor = -1` — which
`undo_run` reads as "revert ALL applied proposals". Asking the agent to delete a
few bars and then pressing Undo reverted every mix decision in the project to
flat, left the bars deleted, and wrote the whole history into the calibration
store as a negative signal.

Two invariants here: a gesture always records a boundary, and Undo never reverts
more than one gesture.
"""
import json

import pytest

from orchestrator.daw_bridge import max_applied_seq
from orchestrator.daw_server import Brain


@pytest.fixture()
def brain(demo_project):
    return Brain(demo_project)


def _applied(project_dir):
    state = json.loads((project_dir / "session.json").read_text())
    return [p for p in state.get("proposals", []) if p.get("status") == "applied"]


def _apply_gain(brain, stem, delta_db):
    """Land one real proposal through the engine, as a run would."""
    from mix_engine.models import ProposalRequest
    from mix_engine.session import Session

    session = Session(brain.project_dir)
    result = session.propose(ProposalRequest(**{
        "rationale": f"{stem} {delta_db:+} dB",
        "predicted_impact": [{"metric": "stem.lufs_s", "predicted_delta": delta_db,
                              "unit": "LUFS"}],
        "actions": [{"type": "gain", "stem_id": stem, "delta_db": delta_db}]}))
    assert result.accepted, result.violations
    assert session.apply(result.proposal_id).ok
    return result.proposal_id


def test_a_command_only_gesture_records_a_boundary(brain):
    """The regression: a gesture that applies no proposal still has to be a
    gesture, or the NEXT Undo has nothing to pop and reaches for everything."""
    invoked = []
    brain.daw_invoke = lambda cmd, args=None, *a, **kw: (
        invoked.append(cmd) or {"ok": True})

    before = len(brain._run_marks)
    brain.begin_agent_turn()
    brain.counting_invoke()("clip.delete", {"clip": 3})
    brain._close_gesture(0)

    assert len(brain._run_marks) == before + 1
    assert brain._run_daw_ops[-1] == 1, "the DAW command must be counted for Undo"


def test_undo_after_a_command_only_gesture_keeps_the_mix(brain):
    """The exact reported failure: mix moves land, then a command-only gesture,
    then Undo. The mix must survive."""
    _apply_gain(brain, "stem_a", -2.0)
    _apply_gain(brain, "stem_b", -1.0)
    assert len(_applied(brain.project_dir)) == 2

    brain.daw_invoke = lambda cmd, args=None, *a, **kw: {"ok": True}
    brain.begin_agent_turn()
    floor = __import__("orchestrator.daw_bridge", fromlist=["max_applied_seq"]) \
        .max_applied_seq(brain.project_dir)
    brain.counting_invoke()("clip.delete", {"clip": 3})
    brain._close_gesture(floor)

    brain.undo()

    assert len(_applied(brain.project_dir)) == 2, (
        "Undo of a command-only gesture must not revert the mix to flat")


def test_undo_reverts_one_gesture_at_a_time(brain):
    """Two mix gestures, two Undos — never both at once."""
    from orchestrator.daw_bridge import max_applied_seq

    floor_a = max_applied_seq(brain.project_dir)
    _apply_gain(brain, "stem_a", -2.0)
    brain._close_gesture(floor_a)

    floor_b = max_applied_seq(brain.project_dir)
    _apply_gain(brain, "stem_b", -1.0)
    brain._close_gesture(floor_b)

    assert len(_applied(brain.project_dir)) == 2
    brain.undo()
    assert len(_applied(brain.project_dir)) == 1, "one Undo = one gesture"
    brain.undo()
    assert len(_applied(brain.project_dir)) == 0


def test_undo_with_no_recorded_boundary_takes_back_only_the_last_move(brain):
    """The boundary stack is in-memory, so a brain restart loses it. That must
    degrade to ordinary single-step undo, never to a full revert."""
    _apply_gain(brain, "stem_a", -2.0)
    _apply_gain(brain, "stem_b", -1.0)
    _apply_gain(brain, "stem_a", -1.0)
    assert len(_applied(brain.project_dir)) == 3

    brain._run_marks.clear()      # as if the brain had just restarted
    brain._run_daw_ops.clear()

    brain.undo()

    assert len(_applied(brain.project_dir)) == 2, (
        "no boundary must mean 'undo the last move', not 'revert everything'")


def test_switching_edit_drops_the_previous_edits_boundaries(brain, tmp_path):
    """`applied_seq` floors belong to one edit's session.json; carried across a
    switch they revert the wrong edit's proposals."""
    brain._run_marks.append(7)
    brain._run_daw_ops.append(2)
    brain._pending_taste = [{"proposal_id": "p1"}]

    brain.open_edit(str(tmp_path / ".edits" / "verse"))

    assert brain._run_marks == []
    assert brain._run_daw_ops == []
    assert brain._pending_taste == []


def test_the_boundary_stacks_stay_aligned_and_bounded(brain):
    """They are popped as a pair; if they ever drift, Undo reverts one gesture's
    mix with another's DAW ops."""
    from orchestrator.daw_server import _MAX_UNDO_DEPTH

    brain.daw_invoke = lambda cmd, args=None, *a, **kw: {"ok": True}
    for _ in range(_MAX_UNDO_DEPTH + 25):
        brain.begin_agent_turn()
        brain.counting_invoke()("clip.split", {"clip": 1})   # a real gesture each time
        brain._close_gesture(0)

    assert len(brain._run_marks) == len(brain._run_daw_ops) == _MAX_UNDO_DEPTH


def test_a_turn_that_changed_nothing_records_no_boundary(brain):
    """Chat is the DAW's front door, so most turns are questions. A no-op mark makes
    the NEXT Undo a no-op too — the user presses it, a full render happens, nothing
    moves, and Undo reads as broken until enough presses reach a real gesture."""
    _apply_gain(brain, "stem_a", -2.0)
    brain.begin_agent_turn()
    brain._close_gesture(0)               # the gesture that applied it
    assert len(brain._run_marks) == 1

    brain.daw_invoke = lambda cmd, args=None, *a, **kw: {"ok": True}
    for _ in range(3):                    # "what's the LUFS?" x3 — no proposal, no DAW op
        brain.begin_agent_turn()
        assert brain._close_gesture(max_applied_seq(brain.project_dir)) is False

    assert len(brain._run_marks) == len(brain._run_daw_ops) == 1, (
        "a question is not a gesture")

    brain.undo()
    assert len(_applied(brain.project_dir)) == 0, (
        "one Undo after three questions must still reach the last real gesture")


def test_close_gesture_reports_whether_the_MIX_moved(brain):
    """Its return value is what the callers re-resolve chains on: a command-only
    gesture records a boundary but must NOT claim the mix changed, or every clip edit
    would trigger a pointless re-render and a stale-cache clear."""
    brain.daw_invoke = lambda cmd, args=None, *a, **kw: {"ok": True}
    brain.begin_agent_turn()
    brain.counting_invoke()("clip.delete", {"clip": 1})
    floor = max_applied_seq(brain.project_dir)
    assert brain._close_gesture(floor) is False, "no proposal landed"
    assert len(brain._run_marks) == 1, "...but it is still one gesture"

    brain.begin_agent_turn()
    floor = max_applied_seq(brain.project_dir)
    _apply_gain(brain, "stem_a", -2.0)
    assert brain._close_gesture(floor) is True
