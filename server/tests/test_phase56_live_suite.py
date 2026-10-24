"""Phase 56: the live suite - tasks in the real app, graded on the END STATE.

The suite runs through the built console (`python -m orchestrator.live_suite`, and `--null`),
which is not pytest's to start. What pytest holds is everything about it that needs no app:
its shape, that no prompt hands the model the command, and that no task's check passes for an
agent that DID NOTHING and said "Done." - read here against the project exactly as it opens.
"""
from __future__ import annotations

import re

import pytest

from orchestrator import chat_rig as R
from orchestrator import live_suite as L


def _untouched(reply: str = "Done.", **turn) -> R.Outcome:
    """The project as it opens (test_projects/suno_demo), and a turn that changed nothing."""
    tracks = [{"name": n, "muted": False, "solo": False, "pan": 0.0, "gain_db": 0.0, "sends": {},
               "devices": [], "output": "master", "automation": [],
               "clips": [{"index": 0, "start_s": 0.0, "length_s": 8.0, "fade_in_s": 0.0,
                          "fade_out_s": 0.0, "gain_db": 0.0}]} for n in L.STEMS]
    snapshot = {"tracks": tracks, "markers": [], "returns": [], "groups": [],
                "tempo": {"bpm": 120.0, "numerator": 4, "denominator": 4},
                "transport": {"loop_on": False, "loop_start_s": 0.0, "loop_end_s": 0.0},
                "master": {"devices": [{"type": "limiter", "enabled": True}]}}
    feed = [{"k": "user", "t": "the request"}, {"k": "prose", "t": reply}]
    return R.Outcome(turns=[{"snapshot": snapshot, "feed": feed, "first_block": 0, "ended": True,
                             "tree_writes": [], **turn}],
                     rig={}, timing=[], requests=[], violations=[], unscripted=0, left=0,
                     fake_s=[], log="", project=None)


def test_the_suite_is_twenty_six_tasks_of_the_kinds_nothing_else_grades():
    kinds = L.by_kind(L.TASKS)
    # twenty-five since Phase 56; arrange_a_midi_loop is the twenty-sixth (Phase 61)
    assert len(L.TASKS) == 26 and len({t.name for t in L.TASKS}) == 26
    assert kinds["ten commands or more"] >= 2 and kinds["an index that moved"] >= 2
    assert kinds["an Ask plan"] >= 2 and kinds["a refusal to own"] >= 5


@pytest.mark.parametrize("task", L.TASKS, ids=lambda t: t.name)
def test_no_task_is_done_by_an_agent_that_did_nothing_and_said_done(task):
    """The class the gate tolerates none of: "done" over a project that is not as asked."""
    assert task.check(_untouched()) != [], f"{task.name} passes with nothing done"


def test_a_refusal_is_owned_by_words_that_say_it_did_not_happen():
    for said in ("I couldn't split it - the track is frozen.", "There is no guitar track. Which one?",
                 "That one is yours to turn off - only you can.", "The export was declined."):
        assert L.owns_it(_untouched(said)) == [], said
    for said in ("Done.", "Split at 4 seconds, as asked.", "All set - no errors."):
        assert L.owns_it(_untouched(said)) != [], said
    assert L.owns_it(_untouched("")) == ["no reply in the feed"]


def test_a_task_of_ten_commands_scripts_at_least_ten():
    for task in L.TASKS:
        if task.kind != "ten commands or more":
            continue
        calls = [b for step in task.tape for b in step.get("blocks") or [] if b[0] == "tool_use"]
        assert len(calls) >= 10, (task.name, len(calls))


def test_no_prompt_hands_the_model_its_command():
    """A task says what the user wants, in a user's words: never a catalog key, never an
    argument's name."""
    keys = set(R.sweep_plan())
    for task in L.TASKS:
        words = set(re.findall(r"[a-z_]+\.[a-z_]+", task.prompt))
        assert not words & keys, (task.name, words & keys)
        assert not re.search(r"\b(gain_db|position_s|at_s|fade_in_s|stem_id|start_s|end_s)\b", task.prompt)


def test_an_index_task_cannot_be_done_with_the_indexes_it_started_with():
    """Its tape's SECOND call names an index that is right only after the first one landed."""
    task = next(t for t in L.TASKS if t.name == "index_after_a_delete")
    calls = [b[3] for step in task.tape for b in step.get("blocks") or [] if b[0] == "tool_use"]
    assert calls[0]["command"] == "clip.delete" and calls[1]["args"]["clip"] == 2
    assert len([s for s in task.before if s.get("key") == "clip.split"]) == 3     # four clips to begin with
    task = next(t for t in L.TASKS if t.name == "markers_after_a_remove")
    calls = [b[3] for step in task.tape for b in step.get("blocks") or [] if b[0] == "tool_use"]
    assert [c["args"]["index"] for c in calls] == [0, 1]                          # C is 1 once A has gone


def test_a_staged_state_is_the_users_commands_before_the_prompt():
    staged = [t for t in L.TASKS if t.before]
    assert {t.name for t in staged} >= {"frozen_track", "index_after_a_delete",
                                        "markers_after_a_remove", "return_removal_declined"}
    for task in staged:
        leg = L.scenario(task).turns[0].leg
        assert leg["before"] == task.before
        assert L.scenario(task, live=True).turns[0].tape == []                    # a live run has no tape
        assert L.scenario(task, live=True).turns[0].leg["before"] == task.before  # ...and the same start


def test_the_null_run_reads_a_failing_check_as_right_and_a_passing_one_as_wrong():
    task = L.TASKS[0]
    out = _untouched()
    lines, wrong = L.report([task], {task.name: (task.check(out), out)}, null=True)
    assert wrong == 0 and lines[0].startswith("ok") and "as it must" in lines[0]
    lines, wrong = L.report([task], {task.name: ([], out)}, null=True)
    assert wrong == 1 and "did nothing and said Done" in lines[0]
    lines, wrong = L.report([task], {task.name: (["bass.muted is False, not True"], out)}, null=False)
    assert wrong == 1 and lines[0].startswith("XX")


def test_a_live_run_is_refused_unless_it_is_called_billed(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(L, "BINARY", tmp_path / "there")
    (tmp_path / "there").write_text("x")
    monkeypatch.setattr(L, "run", lambda *a, **k: pytest.fail("a live run started without --billed"))
    assert L.main(["--live", "api"]) == 1
    assert "BILLED" in capsys.readouterr().out
    assert L.main(["--list"]) == 0
