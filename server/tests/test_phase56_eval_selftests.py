"""Phase 56: the Q&A eval and the routing eval prove their own graders, free.

The operate eval got this first (`test_phase56_operate_selftest.py`). The other two billed
gates had the same hole: a case nobody can pass, or a rubric that passes a reply which says
nothing, was found only by paying for a sweep and reading the failures.

`qa_eval.self_test()` and `understanding.self_test()` play scripted agents through the
grader for every case, with no model. Each has an ORACLE (the case's own right answer, built
from its fields) that must pass, and agents that must not: one that says or does nothing,
one that says what the case forbids, one that answers from memory, one that invents a
figure, one that starts a mix run nobody asked for, one that moves another track.
"""
from __future__ import annotations

import json
import types

import pytest

from orchestrator import operate_eval as O
from orchestrator import qa_eval as Q
from orchestrator import understanding as U


# ---------------------------------------------------------------- Q&A

def test_every_question_is_answerable_and_no_rubric_passes_an_empty_answer():
    problems = Q.self_test()
    assert problems == [], "\n".join(problems[:20])


def test_the_qa_self_test_can_fail(monkeypatch):
    """Controls: each agent's problem is reported when the corpus or the grader lets it by."""
    unanswerable = Q.QACase(text="is it a c/e?", needs=("c/e",), never=("c/e",))
    assert any("oracle" in p and "never" in p for p in Q.self_test([unanswerable]))

    too_long = Q.QACase(text="write my essay", needs=("decline",), honesty_only=True, max_chars=3)
    assert any("longer than its own bound" in p for p in Q.self_test([too_long]))

    # the rubric the self-test found in the corpus: `=i` is the pronoun
    pronoun = Q.QACase(text="what key is it in?", needs=("minor|major|=i",))
    assert any("said nothing" in p for p in Q.self_test([pronoun]))

    # figures that the words `needs` already asks for are not a check that the tool was quoted
    same = Q.QACase(text="is the drummer rushing?", needs=("11 ms",), quotes=("11 ms",),
                    expect_tool="analyze_timing")
    assert any("quotes" in p and "already hold" in p for p in Q.self_test([same]))

    measured = Q.QACase(text="is the drummer rushing?", needs=("ahead|behind",),
                        quotes=("11.4|7.5",), unit=r"ms\b", never=("perfectly tight",),
                        expect_tool="analyze_timing")
    assert Q.self_test([measured]) == []
    monkeypatch.setattr(Q, "grade", lambda case, reply, tools: Q.Outcome(case, "answered", "anything", tools))
    lenient = Q.self_test([measured])
    for agent in ("said nothing", "never", "ungrounded", "quotes", "invented"):
        assert any(p.startswith(agent) for p in lenient), (agent, lenient)


def test_the_qa_oracle_names_a_measuring_tool_where_the_figures_come_from_one():
    case = next(c for c in Q.CORPUS if c.quotes)
    reply, tools = Q.oracle(case)
    assert any(t in Q._MEASURED for t in tools)
    assert Q.grade(case, reply, tools).verdict == "answered"
    assert Q.grade(case, reply, ()).verdict == "ungrounded"


def test_the_qa_main_flag_runs_it_and_exits_by_it(capsys):
    assert Q.main(["--self-test"]) == 0
    assert f"{len(Q.CORPUS)} cases" in capsys.readouterr().out


# ---------------------------------------------------------------- routing

def test_every_routing_label_resolves_and_no_wrong_route_does():
    problems = U.self_test()
    assert problems == [], "\n".join(problems[:20])


def test_the_routing_self_test_can_fail(monkeypatch):
    mislabelled = U.Case(text="turn the tuba up", scope=U.TRACK, stem="tuba")
    assert any(p.startswith("label") for p in U.self_test([mislabelled]))

    track = U.Case(text="turn the vocal up", scope=U.TRACK, stem="lead_vox")
    talk = U.Case(text="what is a compressor?", scope=U.TALK)
    assert U.self_test([track, talk]) == []

    monkeypatch.setattr(U, "_over_reach", lambda case, steps, actions: None)
    # (a run beside the move and a move on the master are caught by the ROUTE, before this)
    assert [p.split(":")[0] for p in U.self_test([track])] == ["over-reach"]
    monkeypatch.undo()

    monkeypatch.setattr(U, "_stem_ok", lambda case, stem: True)
    assert any(p.startswith("wrong track") for p in U.self_test([track]))
    monkeypatch.undo()

    monkeypatch.setattr(U, "grade_plan", lambda case, steps: U.Outcome(case, "resolved"))
    lenient = U.self_test([track, talk])
    for agent in ("did nothing", "escalated", "acted", "master", "over-reach", "wrong track"):
        assert any(p.startswith(agent) for p in lenient), (agent, lenient)


def test_the_live_routing_grader_is_the_one_the_self_test_drives():
    """`evaluate_llm` ends in `grade_plan`: what the self-test proves is what a sweep uses."""
    import inspect
    assert "grade_plan(case, recorder.steps)" in inspect.getsource(U.evaluate_llm)
    track = U.Case(text="turn the vocal up", scope=U.TRACK, stem="lead_vox")
    assert U.grade_plan(track, U.oracle(track)).verdict == "resolved"
    assert U.grade_plan(track, [U._RUN_STEP]).verdict == "mis-scoped"


def test_the_routing_main_flag_runs_it_and_exits_by_it(capsys):
    assert U.main(["--self-test"]) == 0
    assert f"{len(U.CORPUS)} cases" in capsys.readouterr().out


# ---------------------------------------------------------------- trajectories, the served model

class _Session:
    """Stands in for ChatSession: says one thing, calls nothing, names who answered."""
    served = "claude-opus-5-5"
    says = "Here is the answer."

    def __init__(self, **_kw):
        self.usage = {}
        self.timing = types.SimpleNamespace(served_model=self.served)

    async def turn(self, text, *, tools, context, live_context, emit):
        await emit({"type": "chat_message", "text": self.says})


@pytest.fixture
def scripted(monkeypatch):
    from orchestrator import chat
    monkeypatch.setattr(chat, "ChatSession", _Session)
    return _Session


def test_a_question_and_a_routing_case_keep_their_whole_trajectory(tmp_path, monkeypatch, scripted):
    monkeypatch.setenv(O.TRAJECTORIES_ENV, str(tmp_path))
    question = Q.CORPUS[0]
    got = Q.evaluate(question)
    assert got.verdict != "error", got.detail
    request = next(c for c in U.CORPUS if c.scope == U.TALK)
    assert U.evaluate_llm(request).verdict == "resolved"
    kept = {json.loads(p.read_text())["eval"]: json.loads(p.read_text()) for p in tmp_path.iterdir()}
    assert set(kept) == {"qa", "routing"}
    assert kept["qa"]["text"] == question.text and kept["qa"]["reply"] == _Session.says
    assert kept["qa"]["verdict"] == got.verdict and kept["qa"]["served_model"] == "claude-opus-5-5"
    assert kept["routing"]["text"] == request.text and kept["routing"]["steps"] == []


def test_nothing_is_kept_unless_asked_for(tmp_path, monkeypatch, scripted):
    monkeypatch.delenv(O.TRAJECTORIES_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    Q.evaluate(Q.CORPUS[0])
    assert list(tmp_path.iterdir()) == []


def test_a_case_answered_by_another_model_is_not_a_score(scripted, monkeypatch):
    """A fallback model's answer graded as the asked model's: not measured, in both evals."""
    question = next(c for c in Q.CORPUS if c.honesty_only)
    request = next(c for c in U.CORPUS if c.scope == U.TALK)
    assert Q.evaluate(question, model="claude-opus-5-5").verdict != "error"
    assert U.evaluate_llm(request, model="claude-opus-5-5").verdict == "resolved"
    for outcome in (Q.evaluate(question, model="claude-sonnet-5-5"),
                    U.evaluate_llm(request, model="claude-sonnet-5-5")):
        assert outcome.verdict == "error" and "served by claude-opus-5-5" in outcome.detail
