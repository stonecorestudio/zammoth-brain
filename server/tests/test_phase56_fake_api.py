"""Phase 56, step 2: the fake Anthropic API runs outside pytest, fails the way a network
fails, and can refuse what the real API refuses.

Until 2026-10-12 the fake answered every request in one write and needed pytest's monkeypatch
to start. So three things the shipping `api` loop does were proven only on a hand-built
exception or a class NAME: a connection that drops mid-response, a rate limit the SDK waits
out, and the spend cap. And nothing checked a request's SHAPE at all - the one thing a real
key would refuse first.

The strict rules are a transcription of Anthropic's documents (`fake_anthropic.violation`),
not the API: a request they accept is not thereby valid. What they are for is the other
direction - a change that makes the product send something the documents forbid goes red
here, free, before a key is spent on it.
"""
from __future__ import annotations

import json
import threading
import time

import anthropic
import anyio
import pytest

import orchestrator.chat as chatmod
from orchestrator import llm_backend, llm_models
from orchestrator.chat import CATALOG_LABEL, ChatSession, ChatTools
from tests import fake_anthropic as F

MODEL = llm_models.DEFAULT_MODEL


# ---------------------------------------------------------------- outside pytest

def test_the_fake_starts_with_no_test_framework_and_says_what_a_client_needs(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    fake, port, shutdown = F.start()
    try:
        import os
        assert "ANTHROPIC_BASE_URL" not in os.environ, "start() touches no environment"
        env = F.env(port)
        assert env == {"ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
                       "ANTHROPIC_API_KEY": F.PLACEHOLDER_KEY, "MIX_AGENT_LLM_BACKEND": "api"}
        client = anthropic.Anthropic(base_url=env["ANTHROPIC_BASE_URL"], api_key=F.PLACEHOLDER_KEY)
        fake.script = [{"blocks": [("text", "hello")]}]
        said = client.messages.create(model=MODEL, max_tokens=64,
                                      messages=[{"role": "user", "content": "hi"}], stream=False)
        assert said is not None and len(fake.requests) == 1
    finally:
        shutdown()


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(chatmod, "_DISABLED_BETAS", set())
    monkeypatch.setattr(chatmod, "ROUND_RESEND_WAITS_S", (0.02, 0.02))
    fake, shutdown = F.serve(monkeypatch)
    fake.require_script = True
    try:
        yield fake
    finally:
        shutdown()


def _client(api) -> anthropic.Anthropic:
    import os
    return anthropic.Anthropic(base_url=os.environ["ANTHROPIC_BASE_URL"],
                               api_key=F.PLACEHOLDER_KEY, max_retries=0)


TOOLS = [{"name": "daw_command", "description": "operate the DAW",
          "input_schema": {"type": "object", "properties": {
              "command": {"type": "string"}, "args": {"type": "object"}}}}]
CONTEXT = ("Workspace mode: agent (your moves apply immediately).\n\n"
           + CATALOG_LABEL + "): [{\"key\": \"track.set_gain\"}]")
CALL = {"blocks": [("thinking", ""), ("tool_use", "toolu_1", "daw_command",
                                      {"command": "track.set_gain",
                                       "args": {"track": "vox", "gain_db": -2}})]}
DONE = {"blocks": [("thinking", ""), ("text", "Vocal down 2 dB.")]}


class _Tools(ChatTools):
    def __init__(self):
        self.calls: list = []

    async def daw_command(self, command, args):
        self.calls.append((command, args))
        return {"ok": True}


def _turn(api, script, should_cancel=None, tool_defs=None, fast=False):
    """One chat turn through the real SDK: (events, tools, error or None, seconds)."""
    api.script = list(script)
    session = ChatSession(reasoning="medium",
                          tool_defs_for=tool_defs or (lambda during_run: TOOLS))
    if fast:
        session.set_fast(True)
    events: list[dict] = []
    tools = _Tools()
    began = time.monotonic()

    async def emit(event):
        events.append({**event, "_at": time.monotonic() - began})
    error = None
    try:
        anyio.run(lambda: session.turn("pull the vocal down", tools=tools, context=CONTEXT,
                                       emit=emit, should_cancel=should_cancel))
    except Exception as exc:  # noqa: BLE001 - the test reads what the turn raised
        error = exc
    return events, tools, error, time.monotonic() - began, session


def _kinds(events) -> list[str]:
    return [e["type"] for e in events]


# ---------------------------------------------------------------- a network, not one write

def test_a_paced_response_arrives_over_time(api):
    """One write per response made a 60-second stream and a 6-millisecond one the same test."""
    events, _, error, _, _ = _turn(api, [{"blocks": [("text", "one "), ("text", "two "),
                                                     ("text", "three")], "pace_s": 0.08}])
    assert error is None
    deltas = [e["_at"] for e in events if e["type"] == "chat_delta"]
    assert len(deltas) == 3 and deltas[-1] - deltas[0] > 0.2, deltas


def test_a_held_response_is_a_moment_a_test_can_meet_and_stop_ends_it(api):
    """A Stop pressed while the model is mid-sentence: the hold is where "mid-sentence" is,
    with no sleep to guess it."""
    stop = threading.Event()

    def press_stop_once_held():
        assert api.wait_for_hold("mid"), "the response never reached its hold"
        stop.set()
    presser = threading.Thread(target=press_stop_once_held, daemon=True)
    presser.start()
    events, tools, error, seconds, _ = _turn(
        api, [{"blocks": [("text", "Pulling the vocal"), ("text", " down now")],
               "hold": "mid", "hold_after": 3, "hold_max_s": 20}],
        should_cancel=stop.is_set)
    presser.join(5)
    assert error is None and seconds < 5.0, (error, seconds)
    assert _kinds(events)[-1] == "chat_message" and chatmod.STOPPED_NOTICE in events[-1]["text"]
    assert "Pulling the vocal" in "".join(e.get("text", "") for e in events
                                         if e["type"] == "chat_delta")
    assert tools.calls == [] and len(api.requests) == 1
    for _ in range(100):                      # the fake sees the client go
        if api.dropped:
            break
        time.sleep(0.05)
    assert api.dropped == 1


def test_a_held_response_goes_on_when_it_is_released(api):
    threading.Thread(target=lambda: (api.wait_for_hold("gate"), api.release("gate")),
                     daemon=True).start()
    events, _, error, _, _ = _turn(api, [{**DONE, "hold": "gate", "hold_after": 1}])
    assert error is None and events[-1]["text"] == "Vocal down 2 dB."


# ---------------------------------------------------------------- the faults, on a real socket

def test_a_connection_dropped_before_anything_was_shown_is_asked_again(api):
    """`transient_stream_failure` knew a dropped connection by its class NAME, on an exception
    a test built by hand. This is a socket that closes under the SDK."""
    events, tools, error, _, _ = _turn(api, [{**CALL, "cut_after": 1}, CALL, DONE])
    assert error is None, error
    assert events[0]["type"] == "chat_notice" and events[0]["text"] == chatmod.ROUND_RESENT_NOTICE
    assert tools.calls == [("track.set_gain", {"track": "vox", "gain_db": -2})], \
        "the call of the round that died never ran; the round asked again ran once"
    assert len(api.requests) == 3 and events[-1]["text"] == "Vocal down 2 dB."


def test_a_connection_dropped_inside_a_tool_call_runs_nothing_of_it(api):
    half = {"blocks": [("tool_use", "toolu_9", "daw_command", {"command": "track.delete"})],
            "cut_after": 3}                  # the call's block had begun; its input had not ended
    events, tools, error, _, _ = _turn(api, [half, CALL, DONE])
    assert error is None and [c[0] for c in tools.calls] == ["track.set_gain"]


def test_a_connection_dropped_after_words_were_shown_ends_the_turn_and_says_so(api):
    events, tools, error, _, _ = _turn(api, [{"blocks": [("text", "Working on it"), ("text", "…")],
                                              "cut_after": 4, "pace_s": 0.02}, DONE])
    assert isinstance(error, RuntimeError) and "connection to Anthropic dropped" in str(error)
    assert len(api.requests) == 1, "words the user has read are not sent for again"
    assert not isinstance(error, llm_backend.NoRetry), "trying again can work"


def test_a_rate_limit_is_waited_out_by_its_own_header(api):
    events, _, error, seconds, _ = _turn(api, [
        {"error": [429, "This request would exceed your organization's rate limit of 400,000 "
                        "output tokens per minute.", "rate_limit_error"],
         "headers": {"retry-after": "1"}}, DONE])
    assert error is None and len(api.requests) == 2
    assert 0.9 <= seconds < 4.0, f"waited {seconds:.2f} s for a retry-after of 1"


#: Anthropic's own bodies (the rate-limits page, "Reaching your spend cap" and "Setting your
#: own spend limit"), not a paraphrase.
SPEND_CAP = {"error": [429, "You have reached your API usage limits: your organization has "
                            "crossed its monthly API usage threshold, set based on your "
                            "organization's API tier. You will regain access on 2026-11-01 at "
                            "00:00 UTC.", "rate_limit_error"],
             "details": {"error_code": "enforced_spend_limit_reached"}}
OWN_LIMIT = {"error": [400, "You have reached your specified API usage limits. You will regain "
                            "access on 2026-11-01 at 00:00 UTC.", "invalid_request_error"]}


def test_the_spend_cap_is_a_sentence_with_no_retry_whatever_the_sdk_tried(api, monkeypatch):
    monkeypatch.setattr(llm_backend, "API_MAX_RETRIES", 1)      # the SDK's backoff, once
    events, _, error, _, _ = _turn(api, [SPEND_CAP, SPEND_CAP])
    assert isinstance(error, llm_backend.NoRetry) and "monthly spend cap" in str(error)
    assert len(api.requests) == 2, "the SDK asks again (no retry-after): the count is the cost"


def test_a_limit_the_user_set_is_its_own_sentence_at_once(api):
    events, _, error, seconds, _ = _turn(api, [OWN_LIMIT])
    assert isinstance(error, llm_backend.NoRetry) and "the spend limit you set" in str(error)
    assert len(api.requests) == 1 and seconds < 2.0


def test_a_rejected_key_is_not_retried(api):
    events, _, error, _, _ = _turn(api, [{"error": [401, "invalid x-api-key",
                                                    "authentication_error"]}])
    assert isinstance(error, llm_backend.NoRetry) and "API key was rejected" in str(error)
    assert len(api.requests) == 1


def test_a_request_the_script_has_no_step_for_is_an_error_not_an_ok(api):
    """The default reply, "ok", hid a turn that asked more often than its script expected."""
    events, _, error, _, _ = _turn(api, [CALL])          # the round after the tool has no step
    assert error is not None and len(api.unscripted) == 1
    api.require_script = False                           # ...and the old default still exists
    events, _, error, _, _ = _turn(api, [CALL])
    assert error is None and events[-1]["text"] == "ok"


# ---------------------------------------------------------------- what the model can get wrong

def test_a_tool_the_model_invented_is_refused_by_name_and_the_turn_goes_on(api):
    events, tools, error, _, _ = _turn(api, [
        {"blocks": [("tool_use", "toolu_1", "delete_everything", {"sure": True})]}, DONE])
    assert error is None and tools.calls == []
    done = next(e for e in events if e["type"] == "chat_tool_done")
    assert done["ok"] is False and done["error"] == "unknown tool: delete_everything"
    answered = api.requests[1]["messages"][-1]["content"][0]
    assert answered["type"] == "tool_result" and answered.get("is_error") is True


def test_a_call_that_left_out_a_required_argument_is_told_which(api):
    """It read `error: 'command'` - a KeyError's repr, which names nothing to the model."""
    events, tools, error, _, _ = _turn(api, [
        {"blocks": [("tool_use", "toolu_1", "daw_command", {"cmd": "track.set_gain"})]}, DONE])
    assert error is None and tools.calls == []
    done = next(e for e in events if e["type"] == "chat_tool_done")
    assert done["ok"] is False
    assert "daw_command" in done["error"] and "`command`" in done["error"], done["error"]


def test_a_call_cut_off_by_max_tokens_runs_nothing(api):
    cut = {"blocks": [("tool_use_raw", "toolu_1", "daw_command",
                       '{"command": "track.delete", "args": {"track": "vo')],
           "stop": "max_tokens"}
    events, tools, error, _, _ = _turn(api, [cut])
    assert error is None and tools.calls == [], "half a call is not a call"
    assert "nothing was applied" in events[-1]["text"]


# ---------------------------------------------------------------- the strict rules

def _body(**over) -> dict:
    return {"model": MODEL, "max_tokens": 1024,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}], **over}


CALLED = [{"role": "user", "content": [{"type": "text", "text": "mute it"}]},
          {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "daw_command",
                                             "input": {}}]}]
RESULT = {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}
MARK = {"cache_control": {"type": "ephemeral"}}
HOUR = {"cache_control": {"type": "ephemeral", "ttl": "1h"}}


@pytest.mark.parametrize("body, headers, path, says", [
    (_body(model="claude-opus-9"), {}, "", "model: claude-opus-9"),
    ({k: v for k, v in _body().items() if k != "max_tokens"}, {}, "", "max_tokens: Field required"),
    (_body(max_tokens=200_000), {}, "", "maximum allowed number of output tokens"),
    (_body(messages=[{"role": "assistant", "content": "hi"}]), {}, "", 'must use the "user" role'),
    (_body(messages=_body()["messages"] + [{"role": "assistant", "content": "{"}]), {}, "",
     "does not support assistant message prefill"),
    (_body(messages=CALLED + [{"role": "user", "content": [{"type": "text", "text": "and?"}]}]),
     {}, "", "`tool_use` ids were found without `tool_result` blocks immediately after: t1"),
    (_body(messages=CALLED + [{"role": "user", "content": [
        {"type": "text", "text": "note"}, RESULT]}]), {}, "", "must come first"),
    (_body(messages=_body()["messages"][:1] + [
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        {"role": "user", "content": [RESULT]}]), {}, "", "unexpected `tool_use_id`"),
    (_body(tool_choice={"type": "any"}), {}, "", 'type "tool" and "any" are not supported'),
    (_body(thinking={"type": "enabled", "budget_tokens": 4000}), {}, "", "budget_tokens"),
    (_body(thinking={"type": "disabled"}), {}, "", "`disabled` is not supported"),
    (_body(temperature=0.2), {}, "", "temperature"),
    (_body(system=[{"type": "text", "text": str(i), **MARK} for i in range(5)]), {}, "",
     "A maximum of 4 blocks with cache_control may be provided. Found 5."),
    (_body(tools=[{"name": "a", "input_schema": {}, **MARK}],
           system=[{"type": "text", "text": "s", **HOUR}]), {}, "", "must not come after"),
    (_body(speed="fast"), {}, "", "speed: Extra inputs are not permitted"),
    (_body(thinking={"type": "adaptive", "display": "updates"}), {}, "",
     "thinking.display: Extra inputs are not permitted"),
    (_body(output_config={"effort": "extreme"}), {}, "", "output_config.effort"),
    (_body(messages=[{"role": "user", "content": [{"type": "text", "text": ""}]}]), {}, "",
     "text content blocks must be non-empty"),
    (_body(tools=[{"name": "a", "input_schema": {}}, {"name": "a", "input_schema": {}}]), {}, "",
     "Tool names must be unique"),
])
def test_each_rule_refuses_its_request(body, headers, path, says):
    assert says in (F.violation(body, headers, path) or ""), F.violation(body, headers, path)


def test_what_the_rules_allow():
    """Controls: each is the allowed neighbour of a request a rule refuses."""
    fine = [
        _body(),
        _body(messages=CALLED + [{"role": "user", "content": [RESULT, {"type": "text", "text": "measured"}]}]),
        _body(tool_choice={"type": "auto"}), _body(thinking={"type": "adaptive"}),
        _body(system=[{"type": "text", "text": str(i), **MARK} for i in range(4)]),
        _body(tools=[{"name": "a", "input_schema": {}, **HOUR}],
              system=[{"type": "text", "text": "s", **MARK}]),
        _body(output_config={"effort": "medium"}),
        _body(model="claude-haiku-5-5", thinking={"type": "disabled"}),
    ]
    for body in fine:
        assert F.violation(body, {}, "") is None, (body, F.violation(body, {}, ""))
    assert F.violation(_body(speed="fast"), {"anthropic-beta": "fast-mode-2026-02-01"},
                       "/v1/messages?beta=true") is None
    assert F.violation(_body(thinking={"type": "adaptive", "display": "updates"}),
                       {"anthropic-beta": "thinking-display-updates-2026-08-18"}, "") is None


def test_a_strict_fake_answers_a_bad_request_as_the_api_does(api):
    api.strict = True
    with pytest.raises(anthropic.BadRequestError) as refused:
        _client(api).messages.create(model=MODEL, max_tokens=64, tool_choice={"type": "any"},
                                     tools=TOOLS, messages=[{"role": "user", "content": "hi"}])
    assert "are not supported for this model" in str(refused.value)
    assert api.violations and api.script == [], "a refused request spends no step of the script"
    with pytest.raises(anthropic.NotFoundError):
        _client(api).messages.create(model="claude-opus-9", max_tokens=64,
                                     messages=[{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------- the product, held to them

def test_the_chat_loops_own_requests_pass_the_strict_rules(api):
    """Two rounds of the shipping loop - a tool call, its result with the turn's notes, the
    reply - with the DAW's real tool set. Zero refusals, and no request beyond the script."""
    api.strict = True
    events, tools, error, _, session = _turn(api, [CALL, DONE],
                                             tool_defs=lambda during_run: chatmod.daw_tool_defs(False))
    assert error is None and api.violations == [] and api.unscripted == []
    assert len(api.requests) == 2 and tools.calls
    assert len(F._breakpoints(api.requests[1])) <= F.MAX_CACHE_BREAKPOINTS
    # a second turn of the same conversation: the history it replays is still well formed
    api.script = [DONE]
    anyio.run(lambda: session.turn("a bit more", tools=_Tools(), context=CONTEXT,
                                   emit=_discard))
    assert api.violations == [] and len(api.requests) == 3


async def _discard(_event):
    pass


def test_a_fast_request_passes_them_too(api):
    api.strict = True
    events, _, error, _, _ = _turn(api, [DONE], fast=True)
    assert error is None and api.violations == []
    assert api.requests[0].get("speed") == "fast", "the control: this turn did ask for fast"


def test_the_accounts_limits_reach_the_timing_record(api):
    """Step 1 reads the API's rate-limit headers; this is a response that carries them."""
    api.limits = {"anthropic-ratelimit-requests-limit": "1000",
                  "anthropic-ratelimit-input-tokens-limit": "2000000",
                  "anthropic-ratelimit-output-tokens-limit": "400000"}
    events, _, error, _, session = _turn(api, [DONE])
    assert error is None
    assert session.timing.summary()["rate_limits"] == {"rpm": 1000, "itpm": 2_000_000,
                                                       "otpm": 400_000}
