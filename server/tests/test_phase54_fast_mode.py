"""Phase 54, step 4: the Fast output switch.

Anthropic's fast mode is a research preview: `speed: "fast"` on a request, up to 2.5 times the
output tokens a second from the same model, at twice the price, for accounts they have switched
it on for. Opus 5.5 is the one picker model that has it. The switch is the user's, off by
default, and these pin what it sends and what happens when it is turned away - against a local
fake Messages API through the real SDK (no key, no network, nothing billed):

- a fast request goes out as documented (top-level `speed`, the beta named, the SDK's beta
  door), and is priced by what the RESPONSE says it ran at;
- nothing is sent for a model without it, on the subscription transport, or with the switch off;
- fast mode's own rate limit carries the turn on at standard speed and says so once;
- an account without the preview is told once and is not asked again by this brain;
- a fallback inside a turn never breaks the conversation's preserved thinking.
"""
from __future__ import annotations

import anyio
import pytest

import orchestrator.chat as chatmod
from orchestrator import daw_bridge, daw_server, llm_models
from orchestrator.agents import ClaudeAgent, PassContext
from orchestrator.chat import (BETA_DISPLAY_UPDATES, CATALOG_LABEL, FAST_LIMIT_NOTICE,
                               FAST_UNAVAILABLE_NOTICE, ChatSession, ChatTools)
from orchestrator.llm_models import BETA_FAST_MODE
from tests.fake_anthropic import serve

OPUS = "claude-opus-5-5"


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(chatmod, "_DISABLED_BETAS", set())
    monkeypatch.setattr(llm_models, "_FAST_UNAVAILABLE", {"why": ""})
    fake, shutdown = serve(monkeypatch)
    try:
        yield fake
    finally:
        shutdown()


TOOLS = [{"name": "daw_command", "description": "operate the DAW",
          "input_schema": {"type": "object", "properties": {
              "command": {"type": "string"}, "args": {"type": "object"}}}}]
CONTEXT = ("Workspace mode: agent (your moves apply immediately).\n\n"
           + CATALOG_LABEL + "): [{\"key\": \"track.set_gain\"}]")


class _Tools(ChatTools):
    def __init__(self):
        pass

    async def daw_command(self, command, args):
        return {"ok": True}


def _turn(session: ChatSession, text: str = "pull the vocal down") -> list[dict]:
    events: list[dict] = []

    async def emit(event):
        events.append(event)
    anyio.run(lambda: session.turn(text, tools=_Tools(), context=CONTEXT, emit=emit))
    return events


def _session(model: str = OPUS, fast: bool = True) -> ChatSession:
    session = ChatSession(model=model, reasoning="medium", tool_defs_for=lambda during_run: TOOLS)
    session.set_fast(fast)
    return session


def _betas(headers: dict) -> set[str]:
    return {b.strip() for b in (headers.get("anthropic-beta") or "").split(",") if b.strip()}


TOOL_ROUND = {"blocks": [("thinking", ""), ("tool_use", "toolu_1", "daw_command",
                                             {"command": "track.set_gain", "args": {}})]}
TEXT_ROUND = {"blocks": [("thinking", ""), ("text", "done")]}
STANDARD_ROUND_USD = llm_models.cost_usd(OPUS, 10, 5)


# ---------------------------------------------------------------- who may ask

def test_only_opus_5_5_on_the_direct_api_may_ask_for_it(monkeypatch):
    monkeypatch.setattr(llm_models, "_FAST_UNAVAILABLE", {"why": ""})
    assert [m.id for m in llm_models.MODELS if m.fast] == [OPUS]
    assert llm_models.fast_kwargs(OPUS, True, "api") == {"speed": "fast", "betas": [BETA_FAST_MODE]}
    assert llm_models.fast_kwargs(OPUS, False, "api") == {}
    for backend in ("subscription", "cloud"):
        assert llm_models.fast_kwargs(OPUS, True, backend) == {}
    for other in ("claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5", "not-a-model"):
        assert llm_models.fast_kwargs(other, True, "api") == {}
    llm_models.note_fast_unavailable("not in the preview")
    assert llm_models.fast_kwargs(OPUS, True, "api") == {}
    assert {m["id"]: m["fast"] for m in llm_models.picker_rows()}[OPUS] is True


def test_a_fast_response_is_twice_the_price_and_only_for_a_model_that_has_it():
    plain = llm_models.cost_usd(OPUS, 1000, 2000, 5000, 300)
    assert llm_models.cost_usd(OPUS, 1000, 2000, 5000, 300, fast=True) == pytest.approx(2 * plain)
    sonnet = llm_models.cost_usd("claude-sonnet-5-5", 1000, 2000)
    assert llm_models.cost_usd("claude-sonnet-5-5", 1000, 2000, fast=True) == sonnet


def test_a_refusal_is_read_by_its_status_and_its_words():
    class _Err(Exception):
        def __init__(self, status, message):
            super().__init__(message)
            self.status_code = status
            self.body = {"error": {"message": message}}
    assert llm_models.fast_refusal(_Err(429, "rate limited")) == "limit"
    assert llm_models.fast_refusal(_Err(400, "speed: fast mode is not enabled")) == "unavailable"
    assert llm_models.fast_refusal(_Err(403, "Fast mode requires access")) == "unavailable"
    assert llm_models.fast_refusal(_Err(400, "messages.3: tool_use ids must be unique")) == ""
    assert llm_models.fast_refusal(_Err(529, "overloaded")) == ""


# ---------------------------------------------------------------- the chat

def test_a_fast_turn_asks_for_it_as_documented_and_is_priced_as_it_ran(api):
    session = _session()
    api.script = [dict(TOOL_ROUND), dict(TEXT_ROUND)]
    events = _turn(session)

    assert len(api.requests) == 2
    for body, headers, path in zip(api.requests, api.headers, api.paths):
        assert body["speed"] == "fast", "a top-level field of the request"
        assert "betas" not in body, "the SDK turns `betas` into the header"
        assert _betas(headers) == {BETA_FAST_MODE, BETA_DISPLAY_UPDATES}
        assert "beta=true" in path, "the SDK's beta door"
        assert body["thinking"] == {"type": "adaptive", "display": "updates"}
        assert body["output_config"]["effort"] == "medium"
    assert session.fast_requests == 2 and session.fast_notice == ""
    assert session.cost_usd == pytest.approx(2 * 2 * STANDARD_ROUND_USD)
    assert not [e for e in events if e["type"] == "chat_notice"]


def test_the_control_with_the_switch_off_nothing_is_sent(api):
    session = _session(fast=False)
    api.script = [dict(TOOL_ROUND), dict(TEXT_ROUND)]
    _turn(session)
    for body, headers, path in zip(api.requests, api.headers, api.paths):
        assert "speed" not in body and "beta=true" not in path
        assert _betas(headers) == {BETA_DISPLAY_UPDATES}
    assert session.fast_requests == 0
    assert session.cost_usd == pytest.approx(2 * STANDARD_ROUND_USD)


@pytest.mark.parametrize("model", ["claude-sonnet-5-5", "claude-haiku-5-5"])
def test_a_model_without_fast_mode_is_never_sent_it(api, model):
    session = _session(model=model)
    api.script = [{"blocks": [("text", "done")]}]
    _turn(session)
    assert "speed" not in api.requests[0] and BETA_FAST_MODE not in _betas(api.headers[0])


def test_a_response_that_ran_at_standard_speed_is_billed_at_standard_price(api):
    """Priced by what the response says, never by what was asked for."""
    session = _session()
    api.script = [{**TEXT_ROUND, "speed": "standard"}]
    _turn(session)
    assert api.requests[0]["speed"] == "fast"
    assert session.fast_requests == 0
    assert session.cost_usd == pytest.approx(STANDARD_ROUND_USD)


def test_fast_modes_own_rate_limit_carries_the_turn_on_at_standard_speed_and_says_so_once(api):
    session = _session()
    limited = {"error": (429, "rate limit for fast mode exceeded")}
    # the first round runs fast; the second is limited (the SDK re-sends it once), and the
    # rest of the turn is standard - it never goes back to fast inside the turn
    api.script = [dict(TOOL_ROUND), limited, limited, dict(TOOL_ROUND), dict(TEXT_ROUND)]
    events = _turn(session)

    speeds = [body.get("speed") for body in api.requests]
    assert speeds == ["fast", "fast", "fast", None, None]
    notices = [e["text"] for e in events if e["type"] == "chat_notice"]
    assert notices == [FAST_LIMIT_NOTICE] and session.fast_notice == FAST_LIMIT_NOTICE
    assert llm_models.fast_unavailable() == "", "a rate limit is not 'this account has no fast mode'"
    assert api.rejected == [], "the fallback kept the conversation's thinking valid"
    assert session.cost_usd == pytest.approx((2 + 1 + 1) * STANDARD_ROUND_USD)
    assert [e for e in events if e["type"] == "chat_message"][-1]["text"] == "done"

    api.script = [dict(TEXT_ROUND)]
    api.requests.clear()
    _turn(session, "and the next turn?")
    assert api.requests[0]["speed"] == "fast", "the next turn asks again"
    assert api.rejected == []


def test_an_account_without_the_preview_is_told_once_and_not_asked_again(api):
    session = _session()
    api.script = [{"error": (400, "speed: fast mode is not enabled for this organization")},
                  dict(TEXT_ROUND)]
    events = _turn(session)
    assert [body.get("speed") for body in api.requests] == ["fast", None]
    assert [e["text"] for e in events if e["type"] == "chat_notice"] == [FAST_UNAVAILABLE_NOTICE]
    assert llm_models.fast_unavailable()

    api.script = [dict(TEXT_ROUND)]
    api.requests.clear()
    events = _turn(session, "again")
    assert "speed" not in api.requests[0], "not asked again by this brain"
    assert not [e for e in events if e["type"] == "chat_notice"], "and not said again"
    assert ClaudeAgent(model=OPUS, effort="medium", fast=True)._fast is True   # the switch stays
    assert llm_models.fast_kwargs(OPUS, True, "api") == {}                    # the request does not


def test_another_400_on_a_fast_request_is_still_that_error(api):
    session = _session()
    api.script = [{"error": (400, "messages.1: tool_use ids must be unique")}]
    with pytest.raises(Exception) as caught:
        _turn(session)
    assert "tool_use ids" in str(caught.value)
    assert llm_models.fast_unavailable() == ""


# ---------------------------------------------------------------- the run planner

VALID = {"rationale": "lift the vocal", "predicted_impact": [
    {"metric": "lufs_i", "predicted_delta": 1.0, "unit": "LUFS"}],
    "actions": [{"type": "gain", "stem_id": "vox", "delta_db": 1.0}]}
SUBMIT = {"blocks": [("thinking", ""),
                     ("tool_use", "toolu_1", "submit_proposals", {"proposals": [VALID]})]}


def _ctx() -> PassContext:
    return PassContext(pass_number=1, metrics={}, targets=[], stem_chains={},
                       engine_call=lambda name, args: {"ok": True}, should_cancel=None)


def test_a_fast_run_asks_for_it_and_is_priced_as_it_ran(api):
    api.script = [dict(SUBMIT)]
    agent = ClaudeAgent(model=OPUS, effort="medium", fast=True)
    assert len(agent.propose(_ctx())) == 1
    assert api.requests[0]["speed"] == "fast" and BETA_FAST_MODE in _betas(api.headers[0])
    assert "beta=true" in api.paths[0]
    assert agent.usage["estimated_cost_usd"] == pytest.approx(2 * STANDARD_ROUND_USD, abs=1e-4)
    assert agent.fast_notice == ""


def test_the_control_a_run_with_the_switch_off_sends_nothing(api):
    api.script = [dict(SUBMIT)]
    agent = ClaudeAgent(model=OPUS, effort="medium")
    agent.propose(_ctx())
    assert "speed" not in api.requests[0] and BETA_FAST_MODE not in _betas(api.headers[0])
    assert agent.usage["estimated_cost_usd"] == pytest.approx(STANDARD_ROUND_USD, abs=1e-4)


def test_a_limited_run_carries_on_at_standard_speed_and_says_so(api):
    limited = {"error": (429, "rate limit for fast mode exceeded")}
    api.script = [limited, limited, dict(SUBMIT)]
    agent = ClaudeAgent(model=OPUS, effort="medium", fast=True)
    assert len(agent.propose(_ctx())) == 1
    assert [body.get("speed") for body in api.requests] == ["fast", "fast", None]
    assert "standard speed" in agent.fast_notice and agent._fast is False


# ---------------------------------------------------------------- the wire from the app

def test_the_switch_rides_every_request_beside_the_model_and_the_effort():
    assert daw_server._request_llm({"model": OPUS, "effort": "medium", "fast": True}) == {
        "model": OPUS, "reasoning": "medium", "fast": True}
    for not_true in ("true", 1, None, False):
        assert daw_server._request_llm({"model": OPUS, "effort": "medium",
                                        "fast": not_true})["fast"] is False
    assert daw_server._request_llm({"model": OPUS, "effort": "medium"})["fast"] is False

    session = ChatSession(model=OPUS, reasoning="medium", tool_defs_for=lambda during_run: TOOLS)
    daw_bridge._apply_turn_llm(session, {"model": OPUS, "reasoning": "high", "fast": True})
    assert session._fast is True and session.settings == {"model": OPUS, "reasoning": "high"}
    daw_bridge._apply_turn_llm(session, {"model": OPUS, "reasoning": "high"})
    assert session._fast is False, "a request that does not carry it switches it off"

    assert daw_bridge.agent_llm({"model": OPUS, "reasoning": "medium", "fast": True}) == {
        "model": OPUS, "effort": "medium", "fast": True}
    assert daw_bridge.agent_llm({"model": OPUS, "reasoning": "medium"})["fast"] is False
