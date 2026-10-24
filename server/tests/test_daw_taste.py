"""Taste learning under shared control (2026-07-30).

The offline product learned from approve/reject clicks. The DAW removed the
approval gate — and with it, silently, the entire signal: it runs
`gating="auto"`, so `loop.py`'s recorder never fired and `user_taste_decisions`
was permanently empty *while both system prompts instructed the agent to
respect it*. The agent was told to remember your taste and was structurally
incapable of it.

Shared control changed the signal's shape, not its existence:
  undone — the user heard it and took it back (strong negative)
  kept   — still standing when they asked for the next thing (weak positive)
"""
import json

import pytest

from orchestrator import daw_bridge
from orchestrator.calibration import load_decisions, taste_summary


def _proposal(stem="stem_a", kind="eq"):
    action = {"type": kind, "stem_id": stem}
    if kind == "eq":
        action["band"] = {"freq_hz": 3000, "gain_db": -2.0, "q": 1.0,
                          "shape": "peak"}
    else:
        action.update({"threshold_db": -18.0, "ratio": 2.5, "attack_ms": 8.0,
                       "release_ms": 120.0, "makeup_db": 0.0})
    return {"rationale": "test move", "predicted_impact": [], "actions": [action]}


# ---------- the recorder ----------

def test_record_taste_writes_a_decision(demo_project):
    n = daw_bridge.record_taste(demo_project, "undone", [_proposal()])
    assert n == 1
    decisions = load_decisions(demo_project / "runs")
    assert len(decisions) == 1
    assert decisions[0]["decision"] == "undone"
    assert decisions[0]["source"] == "daw"


def test_a_proposal_with_no_actions_records_nothing(demo_project):
    assert daw_bridge.record_taste(demo_project, "kept", [{"rationale": "x"}]) == 0
    assert load_decisions(demo_project / "runs") == []


def test_kept_and_undone_are_counted_separately(demo_project):
    runs = demo_project / "runs"
    for _ in range(2):
        daw_bridge.record_taste(demo_project, "kept", [_proposal()])
    daw_bridge.record_taste(demo_project, "undone", [_proposal()])
    row = taste_summary(runs)[0]
    assert row["kept"] == 2
    assert row["undone"] == 1
    assert row["approved"] == 0, "passive acceptance must not masquerade as approval"


def test_undo_outranks_kept_in_the_ordering(demo_project):
    """The summary is sorted worst-first so the agent sees what it should stop
    doing; `kept` must not offset that."""
    runs = demo_project / "runs"
    for _ in range(3):
        daw_bridge.record_taste(demo_project, "kept", [_proposal(kind="eq")])
    for _ in range(3):
        daw_bridge.record_taste(demo_project, "undone", [_proposal(kind="comp")])
    rows = taste_summary(runs)
    assert rows[0]["undone"] == 3, "the rejected kind must sort first"


# ---------- the Brain wiring: the loop actually closes ----------

@pytest.fixture()
def brain(demo_project, monkeypatch):
    import orchestrator.daw_server as ds
    from mix_engine.session import Session

    Session(demo_project)   # discover stems
    return ds.Brain(demo_project)


def _apply_a_proposal(project):
    """Apply one proposal the way a gesture does — through the same plan path the
    ask-mode "do it" uses, rather than hand-rolling the MCP call shape."""
    result = daw_bridge.apply_plan(
        project, {"steps": [{"kind": "proposal", "payload": _proposal()}]},
        daw_invoke=lambda c, a: {"ok": True})
    assert result["ok"], result


def test_undo_records_the_negative_signal(brain, demo_project):
    before = brain._pending_taste
    assert before == []
    _apply_a_proposal(demo_project)
    brain._run_marks.append(-1)
    brain._arm_kept_taste(-1)
    assert brain._pending_taste, "the gesture's moves should await a verdict"

    brain.undo()

    decisions = load_decisions(demo_project / "runs")
    assert [d["decision"] for d in decisions] == ["undone"]
    assert brain._pending_taste == [], \
        "an undone gesture must never later be banked as kept"


def test_the_next_gesture_banks_the_previous_one_as_kept(brain, demo_project):
    _apply_a_proposal(demo_project)
    brain._arm_kept_taste(-1)
    assert brain._pending_taste

    brain.begin_agent_turn()   # the user asks for the next thing
    # Banked when THAT turn ends, not when it begins (2026-09-10 audit P2, 05·F11): the next
    # turn may be the one that takes the move back ("no, put that back").
    assert load_decisions(demo_project / "runs") == []
    brain._close_gesture(10 ** 9)

    decisions = load_decisions(demo_project / "runs")
    assert [d["decision"] for d in decisions] == ["kept"]
    assert brain._pending_taste == [] and brain._taste_carry == [], \
        "banked once, not on every later turn"


def test_a_gesture_that_applied_nothing_records_nothing(brain, demo_project):
    brain.begin_agent_turn()
    brain.begin_agent_turn()
    assert load_decisions(demo_project / "runs") == []


def test_learning_never_breaks_a_turn(brain, demo_project, monkeypatch):
    """Taste is a nice-to-have; a failure in it must not cost the user their turn."""
    import orchestrator.daw_server as ds

    def boom(*_a, **_k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(ds, "record_taste", boom)
    brain._pending_taste = [_proposal()]
    brain.begin_agent_turn()          # must not raise
    brain._close_gesture(10 ** 9)     # …nor where the verdict is banked
    assert brain._pending_taste == []


# ---------- the OTHER undo authority ----------

def test_undo_also_reverts_the_daw_side_of_a_mixed_turn(brain, demo_project):
    """A chat turn can move the mix AND edit the arrangement. Reverting only the
    mix leaves the clip edits standing — the half-revert this fixes."""
    invoked = []
    brain.daw_invoke = lambda c, a=None, *x, **k: (invoked.append(c) or {"ok": True})

    _apply_a_proposal(demo_project)
    brain._turn_daw_ops = 3            # the gesture also issued 3 DAW commands
    brain._run_marks.append(-1)
    brain._arm_kept_taste(-1)
    brain._run_daw_ops.append(brain._turn_daw_ops)

    result = brain.undo()

    assert invoked == ["edit.undo"], "one edit.undo reverts the whole grouped gesture"
    assert result["daw_ops_undone"] == 3
    assert result["daw_ops_pending"] == 0


def test_a_mix_only_turn_does_not_touch_the_daw(brain, demo_project):
    invoked = []
    brain.daw_invoke = lambda c, a=None, *x, **k: (invoked.append(c) or {"ok": True})
    _apply_a_proposal(demo_project)
    brain._run_marks.append(-1)
    brain._run_daw_ops.append(0)       # no DAW commands in this gesture

    brain.undo()
    assert invoked == [], "a mix-only gesture must not fire a Tracktion undo"


def test_a_refused_daw_undo_is_reported_not_swallowed(brain, demo_project):
    brain.daw_invoke = lambda c, a=None, *x, **k: {"ok": False, "error": "nothing to undo"}
    _apply_a_proposal(demo_project)
    brain._run_marks.append(-1)
    brain._run_daw_ops.append(2)

    result = brain.undo()
    assert result["daw_ops_undone"] == 0
    assert result["daw_ops_pending"] == 2, "the user must not be told it fully reverted"


def test_counting_invoke_tracks_the_gestures_daw_commands(brain):
    brain.daw_invoke = lambda c, a=None, *x, **k: {"ok": True}
    invoke = brain.counting_invoke()
    brain.begin_agent_turn()
    invoke("clip.split", {})
    invoke("clip.move", {})
    assert brain._turn_daw_ops == 2
    brain.begin_agent_turn()
    assert brain._turn_daw_ops == 0, "the count is per gesture"


def test_counting_invoke_is_none_without_a_daw(brain):
    brain.daw_invoke = None
    assert brain.counting_invoke() is None


# ---------- taste + lessons reach the DAW agent, not just the store (2026-08-05) ----------

def test_the_daw_agent_actually_sees_the_taste_it_records(demo_project):
    """The DAW records taste diligently and loop.py writes lessons, but neither was
    ever read back into the chat context — while the prompt instructs at length on
    respecting both. Written, never read: the personalization loop was dead on the
    flagship path exactly the way the RECORDING side was before 2026-07-30."""
    from orchestrator.calibration import append_lesson
    from orchestrator.daw_bridge import _chat_context_stable, record_taste

    runs = demo_project / "runs"
    runs.mkdir(exist_ok=True)
    # TASTE_MIN_DECISIONS: a kind only reaches the agent once there is enough
    # evidence to be worth acting on.
    for i in range(3):
        record_taste(demo_project, "undone",
                     [{"proposal_id": f"p{i}", "actions": [{"type": "eq"}]}])
    append_lesson(runs, "make it bright", ["this mix needs less presence, not more"])

    ctx = _chat_context_stable(demo_project, "agent")

    assert "user_taste_decisions" in ctx, "the agent must see how moves were received"
    assert "learned_lessons" in ctx, "the agent must see what past runs concluded"
    assert "less presence" in ctx


def test_no_taste_yet_is_simply_absent(demo_project):
    """A fresh project must not carry empty scaffolding into every prompt."""
    from orchestrator.daw_bridge import _chat_context_stable

    ctx = _chat_context_stable(demo_project, "agent")
    assert "user_taste_decisions" not in ctx
    assert "learned_lessons" not in ctx


# ---------- the brain's edit.undo names the transaction it means (2026-09-02 audit) ----------

def test_undo_asks_the_daw_for_the_agents_own_transaction(brain, demo_project):
    """A plain ⌘Z can already have taken the agent's timeline edit back. The brain's undo
    therefore says WHICH transaction it means (expect_agent), and the DAW refuses rather
    than undoing whatever sits on top of its stack — the user's own previous edit."""
    calls = []
    brain.daw_invoke = lambda c, a=None, *x, **k: (calls.append((c, a)) or {"ok": True})
    _apply_a_proposal(demo_project)
    brain._run_marks.append(-1)
    brain._run_daw_ops.append(1)

    brain.undo()
    assert calls == [("edit.undo", {"expect_agent": True})]


# ---------- the brain's rename_stem answers with re-keyed chains, no render (2026-09-02) ----------

def test_brain_rename_stem_returns_chains_under_the_new_id(brain, demo_project):
    out = brain.rename_stem("stem_a", "lead a")
    assert out["renamed"] and "lead a" in out["chains"] and "stem_a" not in out["chains"]
    assert brain.rename_stem("nothing here", "x")["renamed"] is False
