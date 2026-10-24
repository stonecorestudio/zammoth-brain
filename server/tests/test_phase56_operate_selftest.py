"""Phase 56, step 6: the operate eval proves its own grader, free.

`technique_eval --self-test` has shown since Phase 48 that its graders pass a golden and
fail an untouched project. The operate eval - 378 cases, the gate the agent improvement
loop runs on - had nothing of the kind: a case nobody can pass, or a grader that passes an
agent that did nothing, was found only by paying for a sweep and reading failures.

`operate_eval.self_test()` plays four agents through `grade` for every case, with no model:
the ORACLE (the case's own expected reach) must be `correct`; the one that DOES NOTHING and
says nothing must not be; the one that reaches right and then removes something nobody asked
for must not be; the one that reaches right and names none of the values must not be.
"""
from __future__ import annotations

import json

import pytest

from orchestrator import operate_eval as E


def _case(**over) -> E.OpCase:
    return E.OpCase(**{"text": "a request", "expect": (), **over})


def test_every_case_is_passable_and_no_case_passes_an_agent_that_did_nothing():
    problems = E.self_test()
    assert problems == [], "\n".join(problems[:20])


def test_the_self_test_can_fail(monkeypatch):
    """Controls: an unpassable case and a grader that passes anything are each reported."""
    unpassable = _case(text="mute the bass", expect=("track.set_mute",),
                       forbid=("track.set_mute",))          # the right reach is forbidden
    assert any("oracle" in p for p in E.self_test([unpassable]))
    monkeypatch.setattr(E, "grade", lambda case, steps, reply="": ("correct", "anything", []))
    lenient = E.self_test([_case(text="mute the bass", expect=("track.set_mute",),
                                 args_contain=("bass",))])
    assert any("did nothing" in p for p in lenient) and any("removed" in p for p in lenient)


def test_the_main_flag_runs_it_and_exits_by_it(capsys):
    assert E.main(["--self-test"]) == 0
    assert f"{len(E.CORPUS)} cases" in capsys.readouterr().out


# ---------------------------------------------------------------- three leniencies it found

def test_a_question_answered_with_nothing_is_not_answered():
    """`wants_talk` passed on "no command was called" alone, so a turn that said nothing at
    all was `correct`."""
    case = _case(text="what does a compressor's ratio do?", wants_talk=True)
    assert E.grade(case, [], "")[0] == "none"
    assert E.grade(case, [], "   ")[0] == "none"
    assert E.grade(case, [], "It sets how much a signal over the threshold is reduced.")[0] == "correct"


def _cmd(key: str, **args) -> dict:
    return {"kind": "command", "payload": {"command": key, "args": args}}


def test_a_run_beside_something_removed_is_not_the_right_escalation():
    case = _case(text="mix this for streaming", wants_run=True)
    run = {"kind": "run", "payload": {"brief": "mix this for streaming"}}
    assert E.grade(case, [run], "Running.")[0] == "correct"
    verdict, detail, _ = E.grade(case, [run, _cmd("track.delete", track="gtr_r")], "Running.")
    assert verdict == "wrong" and "track.delete" in detail


def test_a_proposal_beside_something_removed_is_not_the_technique():
    case = _case(text="de-ess the vocal", wants_propose=("deess",), propose_stem="lead_vox")
    proposal = {"kind": "proposal",
                "payload": {"actions": [{"type": "deess", "stem_id": "lead_vox"}]}}
    assert E.grade(case, [proposal], "Proposed.")[0] == "correct"
    verdict, detail, _ = E.grade(case, [proposal, _cmd("clip.delete", track="lead_vox", clip=0)],
                                 "Proposed.")
    assert verdict == "wrong" and "clip.delete" in detail
    # a READ beside the proposal is still fine
    read = next(iter(E.CANNED_READ_REPLIES))
    assert E.grade(case, [_cmd(read), proposal], "Proposed.")[0] == "correct"


def test_a_refused_reach_beside_something_removed_is_not_honest_reporting():
    """The fourth: after a refusal "retries and a sibling key are legitimate, so over-reach
    is not judged" - and neither was an unasked removal."""
    case = next(c for c in E.CORPUS if c.fail_with)
    steps, reply = E.oracle(case)
    assert E.grade(case, steps, reply)[0] == "correct"
    verdict, detail, _ = E.grade(case, [*steps, _cmd("automation.clear", track="gtr_r")], reply)
    assert verdict == "wrong" and "automation.clear" in detail


# ---------------------------------------------------------------- trajectories, the served model

def test_a_cases_whole_trajectory_can_be_kept(tmp_path, monkeypatch):
    """A surprising verdict could not be traced without paying for the case again: the
    checkpoint keeps the verdict and the command keys, never what was said or sent."""
    case = _case(text="mute the bass / now", expect=("track.set_mute",), args_contain=("bass",))
    steps = [_cmd("track.set_mute", track="bass", muted=True)]
    monkeypatch.setenv(E.TRAJECTORIES_ENV, str(tmp_path))
    path = E.keep_trajectory(case, steps, "Bass muted.", ("correct", "track.set_mute", ["track.set_mute"]),
                             served="claude-opus-5-5", asked="claude-opus-5-5", run=2)
    kept = json.loads(path.read_text())
    assert kept["text"] == case.text and kept["verdict"] == "correct" and kept["run"] == 2
    assert kept["steps"] == steps and kept["reply"] == "Bass muted."
    assert kept["served_model"] == "claude-opus-5-5"
    assert path.parent == tmp_path and "/" not in path.name
    monkeypatch.delenv(E.TRAJECTORIES_ENV)
    assert E.keep_trajectory(case, steps, "x", ("correct", "", [])) is None, "off unless asked for"


@pytest.mark.parametrize("asked, served, problem", [
    ("claude-opus-5-5", "claude-opus-5-5", ""),
    ("claude-opus-5-5", "", ""),                       # a transport that does not say: not judged
    (None, "claude-opus-5-5", ""),                      # no model asked for by name
    ("claude-opus-5-5", "claude-opus-5", "claude-opus-5"),
])
def test_a_case_served_by_another_model_is_not_a_score_of_the_one_asked(asked, served, problem):
    said = E.served_by_another(asked, served)
    assert (problem in said and bool(said)) if problem else said == ""
