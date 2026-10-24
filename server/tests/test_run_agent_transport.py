"""Mix runs on the production `api` transport, driven through the REAL SDK (2026-09-22 audit B).

The 2026-09-10 audit tested runs only on the subscription backend; the api path's run code had
never been exercised against the wire. These drive the real `ClaudeAgent` and a real
`anthropic.AsyncAnthropic` against a local fake Messages API (tests/fake_anthropic.py — no key,
no network, nothing billed) and pin what audit B found:

- F4: the planner branched on nothing — a refusal's partial or a max_tokens cut could carry a
  tool call with truncated input (a proposal cut after its first action was applied as the whole
  move), and a thinking-only cut read as "nothing to propose", the words for a fine mix;
- F1: one planner call was a blocking create with a 180 s whole-response timeout, retried up to
  four times, that nothing could interrupt — the DAW killed the whole brain after 45 s of Stop;
- F5: the structured turns (brief compile, critic, reflection) assumed a parsed reply existed;
- F8: guardrail revisions ran past the run's spend limit;
- F11: two submit_proposals in one reply left the first unanswered and the next request 400'd.
"""
from __future__ import annotations

import time

import pytest

from orchestrator import llm_models
from orchestrator.agents import ClaudeAgent, LLMTurnEnded, PassContext
from tests.fake_anthropic import serve

VALID = {"rationale": "lift the vocal", "predicted_impact": [
    {"metric": "lufs_i", "predicted_delta": 1.0, "unit": "LUFS"}],
    "actions": [{"type": "gain", "stem_id": "vox", "delta_db": 1.0}]}


@pytest.fixture
def api(monkeypatch):
    fake, shutdown = serve(monkeypatch)
    try:
        yield fake
    finally:
        shutdown()


def _ctx(ran: list, cancel=None) -> PassContext:
    def engine_call(name, args):
        ran.append(name)
        return {"ok": True}
    return PassContext(pass_number=1, metrics={}, targets=[], stem_chains={},
                       engine_call=engine_call, should_cancel=cancel)


def _submit(tid: str, proposals: list) -> tuple:
    return ("tool_use", tid, "submit_proposals", {"proposals": proposals})


# ---------------------------------------------------------------- F4: stop reasons first

def test_a_refused_round_runs_nothing_and_says_why(api):
    ran: list = []
    api.script = [{"blocks": [("text", "Sure, I'll"), ("tool_use", "toolu_1", "probe_actions",
                                                         {"actions": []}),
                              _submit("toolu_2", [VALID])],
                   "stop": "refusal", "stop_details": {"type": "refusal", "category": "cyber"}}]
    agent = ClaudeAgent(model="claude-opus-5-5", effort="high")
    assert agent.propose(_ctx(ran)) == []
    assert ran == [], "a declined reply's tool calls must never run"
    assert agent.last_empty_reason == "declined" and agent.last_empty_detail == "cyber"
    assert agent.usage["turns"] == 1


def test_a_cut_off_submit_is_never_applied(api):
    """max_tokens mid-submit: the tool input may be truncated, so none of it is applied."""
    api.script = [{"blocks": [("thinking", ""), _submit("toolu_1", [VALID])],
                   "stop": "max_tokens"}]
    agent = ClaudeAgent(model="claude-opus-5-5", effort="max")
    assert agent.propose(_ctx([])) == []
    assert agent.last_empty_reason == "cut off"


def test_the_control_a_clean_submit_is_applied(api):
    api.script = [{"blocks": [("thinking", ""), _submit("toolu_1", [VALID])]}]
    agent = ClaudeAgent(model="claude-opus-5-5", effort="high")
    out = agent.propose(_ctx([]))
    assert len(out) == 1 and out[0]["actions"][0]["stem_id"] == "vox"
    assert agent.last_empty_reason is None


def test_two_submits_both_get_a_result_so_the_repair_request_is_valid(api):
    bad = {"rationale": "x", "actions": [{"type": "not-an-action"}]}
    api.script = [{"blocks": [_submit("toolu_a", [bad]), _submit("toolu_b", [bad])]},
                  {"blocks": [_submit("toolu_c", [VALID])]}]
    agent = ClaudeAgent(model="claude-opus-5-5", effort="high")
    out = agent.propose(_ctx([]))
    assert len(out) == 1
    repair = api.requests[1]["messages"][-1]["content"]
    answered = {b["tool_use_id"] for b in repair if b.get("type") == "tool_result"}
    assert answered == {"toolu_a", "toolu_b"}, "every tool_use needs its tool_result"


# ---------------------------------------------------------------- F1: Stop reaches a call

def test_stop_interrupts_a_planner_call_in_flight(api):
    api.script = [{"blocks": [_submit("toolu_1", [VALID])], "delay_s": 3.0}]
    agent = ClaudeAgent(model="claude-opus-5-5", effort="high")
    t0 = time.monotonic()
    stop_at = t0 + 0.4
    out = agent.propose(_ctx([], cancel=lambda: time.monotonic() >= stop_at))
    elapsed = time.monotonic() - t0
    assert out == []
    assert elapsed < 1.5, f"Stop took {elapsed:.1f} s to land (the call ran 3 s)"


def test_the_control_without_stop_the_slow_call_completes(api):
    api.script = [{"blocks": [_submit("toolu_1", [VALID])], "delay_s": 1.0}]
    agent = ClaudeAgent(model="claude-opus-5-5", effort="high")
    assert len(agent.propose(_ctx([], cancel=lambda: False))) == 1


# ---------------------------------------------------------------- F5: structured turns

def test_a_refused_brief_compile_raises_the_honest_reason(api):
    api.script = [{"blocks": [], "stop": "refusal",
                   "stop_details": {"type": "refusal", "category": "bio"}}]
    agent = ClaudeAgent(model="claude-opus-5-5", effort="high")
    with pytest.raises(LLMTurnEnded) as err:
        agent.compile_brief("make it pop", {"stems": {}})
    assert "declined" in str(err.value) and "bio" in str(err.value)
    assert agent.usage["turns"] == 1


def test_a_truncated_verdict_is_no_verdict_and_is_still_metered(api):
    api.script = [{"blocks": [("text", '{"targets_met": true, "conc')], "stop": "max_tokens"}]
    agent = ClaudeAgent(model="claude-opus-5-5", effort="high")
    from orchestrator.agents import ReviewContext
    assert agent.review(ReviewContext(brief="b", metrics={}, targets=[], applied=[])) is None
    assert agent.usage["turns"] == 1 and agent.usage["estimated_cost_usd"] > 0


def test_the_control_a_clean_structured_reply_parses(api):
    api.script = [{"blocks": [("text", '{"lessons": ["keep the kick dry"]}')]}]
    agent = ClaudeAgent(model="claude-opus-5-5", effort="high")
    assert agent.reflect("b", {}, 1, "targets met") == ["keep the kick dry"]
    body = api.requests[0]
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["output_config"]["effort"] == "low", "reflection runs at low effort"


# ---------------------------------------------------------------- F8: spend limit covers revise

def test_no_revision_is_billed_past_the_spend_limit(api):
    agent = ClaudeAgent(model="claude-opus-5-5", effort="high")
    agent.usage["estimated_cost_usd"] = agent.budget_usd   # already spent
    assert agent.revise(VALID, [{"rule": "gain"}], _ctx([])) is None
    assert api.requests == []


# ---------------------------------------------------------------- per-model request shapes

@pytest.mark.parametrize("model", [m.id for m in llm_models.MODELS])
@pytest.mark.parametrize("effort", ["low", "xhigh", "max"])
def test_every_model_gets_a_planner_request_it_accepts(api, model, effort):
    api.script = [{"blocks": [_submit("toolu_1", [VALID])]}]
    ClaudeAgent(model=model, effort=effort).propose(_ctx([]))
    body = api.requests[0]
    spec = llm_models.spec(model)
    if not spec.efforts:
        assert "thinking" not in body and "output_config" not in body
    else:
        assert body["thinking"] == {"type": "adaptive"}
        assert body["output_config"]["effort"] in spec.efforts
    assert body["max_tokens"] == llm_models.max_tokens_for(model, effort)
    assert body["max_tokens"] <= spec.max_output_tokens
    for forbidden in ("temperature", "top_p", "top_k", "tool_choice"):
        assert forbidden not in body


# ---------------------------------------------------------------- the brain log names the pick

def test_the_brain_log_names_a_runs_model_and_the_effort_it_gets(api, capsys, monkeypatch,
                                                                 odd_ladders):
    """Nothing on screen names the model a run went out on, so the brain log does (10.9)."""
    from orchestrator.daw_bridge import _llm_agent
    monkeypatch.delenv("MIX_AGENT_MODEL", raising=False)     # the fallback below is the default
    monkeypatch.delenv("MIX_AGENT_EFFORT", raising=False)
    no_xhigh, no_effort = odd_ladders
    agent = _llm_agent({"model": no_xhigh, "reasoning": "xhigh"}, "mix run")
    assert agent.pick == (no_xhigh, "high")                   # the level below the one it lacks
    assert f"[daw_bridge] mix run on {no_xhigh} · effort high" in capsys.readouterr().err
    _llm_agent({"model": no_effort, "reasoning": "max"}, "mid-run answer")
    assert f"mid-run answer on {no_effort} · effort none" in capsys.readouterr().err
    agent = _llm_agent({"model": "claude-haiku-5-5", "reasoning": "xhigh"}, "mix run")
    assert agent.pick == ("claude-haiku-5-5", "xhigh")        # every picker model takes it
    assert "mix run on claude-haiku-5-5 · effort xhigh" in capsys.readouterr().err
    _llm_agent({"model": "not-a-model", "reasoning": "high"}, "mix run")   # ignored, not fatal
    assert (f"mix run on {llm_models.DEFAULT_MODEL} · effort {llm_models.DEFAULT_EFFORT}"
            in capsys.readouterr().err)                      # the default, whatever it is
