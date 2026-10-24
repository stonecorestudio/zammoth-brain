"""Phase 55, step 3: the request, for the transport customers run.

Each of these was read against Anthropic's own pages on 2026-10-12, not measured on a model:
the output ceiling, the two account limits that are not rate limits, a round that dies inside
an answered request, the history's budget, the conversation cache's lifetime, how long the
brain waits for a command that renders, and a ceiling on what one turn may spend.
"""
from __future__ import annotations

import anyio
import pytest

import orchestrator.chat as chatmod
from orchestrator import daw_server as S
from orchestrator import llm_models
from orchestrator.chat import CATALOG_LABEL, ChatSession, ChatTools
from orchestrator.llm_backend import NoRetry, raise_api_friendly, spend_limit
from tests.fake_anthropic import serve

PICKER = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5")


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(chatmod, "_DISABLED_BETAS", set())
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
        self.ran: list[str] = []

    async def daw_command(self, command, args):
        self.ran.append(command)
        return {"ok": True}


def _drive(session, text="go", tools=None, events=None, should_cancel=None):
    async def emit(event):
        if events is not None:
            events.append(event)
    anyio.run(lambda: session.turn(text, tools=tools or _Tools(), context=CONTEXT, emit=emit,
                                   should_cancel=should_cancel))


def _session(**kw) -> ChatSession:
    return ChatSession(reasoning="medium", tool_defs_for=lambda during_run: TOOLS, **kw)


# ---------------------------------------------------------------- 3a: the output ceiling

def test_a_round_may_generate_64000_tokens_at_every_effort(api):
    for model in PICKER:
        assert [llm_models.max_tokens_for(model, e) for e in llm_models.EFFORT_LEVELS] == [64_000] * 5
        assert llm_models.max_tokens_for(model, "medium", base=8_000) == 64_000
    session = _session()
    api.script = [{"blocks": [("thinking", ""), ("text", "done")]}]
    _drive(session)
    assert api.requests[0]["max_tokens"] == 64_000 and api.requests[0]["stream"] is True


def test_an_id_off_the_table_keeps_its_own_ceiling():
    """No effort and no thinking config are sent for it, and its maximum may be lower."""
    assert llm_models.max_tokens_for("claude-opus-5", "high") == 16_000
    assert llm_models.max_tokens_for("some-other-model", "max", base=8_000) == 8_000


# ---------------------------------------------------------------- 3b: the account's limits

class _Err(Exception):
    def __init__(self, status_code, message="", details=None, kind="rate_limit_error"):
        super().__init__(message)
        self.status_code = status_code
        self.body = {"type": "error", "error": {"type": kind, "message": message,
                                                **({"details": details} if details else {})}}


CAP = _Err(429, "You have reached your API usage limits: your organization has crossed its "
                "monthly API usage threshold, set based on your organization's API tier. You "
                "will regain access on 2026-11-01 at 00:00 UTC.",
           details={"error_code": "enforced_spend_limit_reached"})
OWN = _Err(400, "You have reached your specified API usage limits. You will regain access on "
                "2026-11-01 at 00:00 UTC.", kind="invalid_request_error")
OWN_WORKSPACE = _Err(400, "You have reached your specified workspace API usage limits.",
                     kind="invalid_request_error")
RATE = _Err(429, "This request would exceed your organization's rate limit of 400,000 output "
                 "tokens per minute.")


def test_the_spend_cap_and_the_users_own_limit_are_told_from_a_rate_limit():
    assert spend_limit(CAP) == "cap" and spend_limit(OWN) == "own"
    assert spend_limit(OWN_WORKSPACE) == "own"
    assert spend_limit(RATE) == "" and spend_limit(_Err(400, "messages.0: bad shape")) == ""
    assert spend_limit(_Err(429, CAP.body["error"]["message"])) == "cap", "by its words too"


def test_each_gets_its_own_sentence_and_no_retry_verb():
    with pytest.raises(NoRetry, match="monthly spend cap"):
        raise_api_friendly(CAP)
    for exc in (OWN, OWN_WORKSPACE):
        with pytest.raises(NoRetry, match="the spend limit you set"):
            raise_api_friendly(exc)
    with pytest.raises(RuntimeError, match="rate limit") as waited:      # control
        raise_api_friendly(RATE)
    assert not isinstance(waited.value, NoRetry), "a rate limit is worth another try"


def test_a_spend_cap_on_a_fast_request_is_not_fast_modes_own_limit():
    assert llm_models.fast_refusal(CAP) == ""
    assert llm_models.fast_refusal(RATE) == "limit"                       # control


# ---------------------------------------------------------------- 3f: a command that renders

def test_a_command_that_renders_in_the_app_is_waited_for():
    for command in ("track.bounce", "track.freeze", "session.export", "clip.consolidate"):
        assert S.reply_wait_s(command, False) == S.RENDER_REPLY_TIMEOUT_S >= 300.0, command
    assert S.reply_wait_s("track.set_gain", False) == S.DAW_REPLY_TIMEOUT_S == 30.0
    assert S.reply_wait_s("track.set_gain", True) == S.CONFIRM_WAIT_S
    assert S.reply_wait_s("track.bounce", False, timeout=5.0) == 5.0, "a caller's own bound holds"
    import inspect
    assert "reply_wait_s(command" in inspect.getsource(S._serve_client), "daw_invoke asks it"


# ---------------------------------------------------------------- 3c: a round that died

def _tool_round(n: int, **step) -> dict:
    return {"blocks": [("thinking", ""), ("tool_use", f"toolu_{n}", "daw_command",
                                          {"command": "track.set_gain", "args": {"n": n}})], **step}


FINAL = {"blocks": [("thinking", ""), ("text", "done")]}


def test_a_round_that_died_before_showing_anything_is_asked_again(api, monkeypatch):
    monkeypatch.setattr(chatmod, "ROUND_RESEND_WAITS_S", (0.0, 0.0))
    session, tools, events = _session(), _Tools(), []
    api.script = [_tool_round(1), {"stream_error": ("overloaded_error", "Overloaded")},
                  {"stream_error": ("api_error", "Internal server error")}, FINAL]
    _drive(session, tools=tools, events=events)
    assert tools.ran == ["track.set_gain"], "the command of the round before ran once"
    assert len(api.requests) == 4 and api.rejected == []
    assert api.requests[1]["messages"] == api.requests[2]["messages"] == api.requests[3]["messages"], \
        "the same request, asked again: nothing of a dead round is stored"
    assert [e["text"] for e in events if e["type"] == "chat_notice"] == [chatmod.ROUND_RESENT_NOTICE] * 2
    assert [e["text"] for e in events if e["type"] == "chat_message"] == ["done"]
    s = session.timing.summary()
    assert s["failed_rounds"] == 2 and s["rounds"] == 4


def test_it_is_asked_twice_and_no_more(api, monkeypatch):
    monkeypatch.setattr(chatmod, "ROUND_RESEND_WAITS_S", (0.0, 0.0))
    session = _session()
    api.script = [{"stream_error": ("overloaded_error", "Overloaded")}] * 3 + [FINAL]
    with pytest.raises(RuntimeError, match="overloaded"):
        _drive(session)
    assert len(api.requests) == 3


def test_a_round_that_had_begun_to_speak_is_not_asked_again(api, monkeypatch):
    """Its words are in the feed already; a second copy beside them would be the worse fault."""
    monkeypatch.setattr(chatmod, "ROUND_RESEND_WAITS_S", (0.0, 0.0))
    session = _session()
    api.script = [{"blocks": [("text", "Here is the pl")],
                   "stream_error": ("overloaded_error", "Overloaded")}, FINAL]
    with pytest.raises(RuntimeError, match="overloaded"):
        _drive(session)
    assert len(api.requests) == 1


def test_a_failure_the_sdk_already_retried_is_not_asked_again(api, monkeypatch):
    from orchestrator import llm_backend
    monkeypatch.setattr(llm_backend, "API_MAX_RETRIES", 0)
    monkeypatch.setattr(chatmod, "ROUND_RESEND_WAITS_S", (0.0, 0.0))
    session = _session()
    api.script = [{"error": (529, "Overloaded", "overloaded_error")}, FINAL]
    with pytest.raises(RuntimeError, match="overloaded"):
        _drive(session)
    assert len(api.requests) == 1


def test_which_dead_streams_are_worth_asking_again():
    from orchestrator.llm_backend import transient_stream_failure as blip

    class _In(Exception):
        def __init__(self, kind, status=200):
            super().__init__(kind)
            self.status_code, self.body = status, {"error": {"type": kind, "message": kind}}
    assert blip(_In("overloaded_error")) == "overloaded" and blip(_In("api_error")) == "server"
    assert blip(_In("rate_limit_error")) == "" and blip(_In("invalid_request_error")) == ""
    assert blip(_In("overloaded_error", status=529)) == "", "an HTTP status: the SDK retried it"
    APITimeoutError = type("APITimeoutError", (Exception,), {})
    RemoteProtocolError = type("RemoteProtocolError", (Exception,), {})
    assert blip(APITimeoutError()) == "timeout" and blip(RemoteProtocolError()) == "dropped"
    assert blip(ValueError("a bug")) == ""


# ---------------------------------------------------------------- 3d: history by size

def test_the_budget_is_in_tokens_and_a_model_priced_by_length_stays_under_its_line():
    assert llm_models.history_budget("claude-opus-5-5") == (200_000, 120_000)
    assert llm_models.history_budget("claude-sonnet-5-5") == (200_000, 120_000)
    trigger, floor = llm_models.history_budget("claude-haiku-5-5")
    line = llm_models.spec("claude-haiku-5-5").long_prompt_over
    assert floor < trigger < line and trigger == int(0.7 * line)
    assert llm_models.history_budget("some-other-model") == (120_000, 80_000)


def _weighing(prompt: int, **step) -> dict:
    """A final round whose request the API says weighed `prompt` tokens."""
    return {**FINAL, "usage": {"input_tokens": 100, "cache_read_input_tokens": prompt - 100,
                               "cache_creation_input_tokens": 0}, **step}


def _opening(message: dict) -> str:
    content = message["content"]
    return content if isinstance(content, str) else content[0]["text"]


def test_a_long_conversation_is_trimmed_by_what_it_weighs(api):
    session = _session()
    for n, prompt in enumerate((60_000, 110_000, 160_000, 210_000), 1):
        api.script = [_weighing(prompt)]
        _drive(session, text=f"turn {n}")
    assert len(session.history) == 8, "vacuity: nothing trimmed below the trigger"
    api.script = [_weighing(115_000)]
    _drive(session, text="turn 5")
    # 210k is past 200k: the oldest whole turns go until it is under 120k. The marks say turns
    # 1 and 2 weigh 100k together (160k - 60k), so turn 3 is the first kept.
    assert [_opening(m) for m in session.history if m["role"] == "user"] == ["turn 3", "turn 4", "turn 5"]
    assert api.rejected == [], "a trim strips the thinking it would otherwise invalidate"


def test_many_small_messages_are_not_a_reason_to_trim(api):
    """The old cap was 60 MESSAGES: one long build tripped it at the next turn, whatever it
    weighed - dropping the earlier reasoning and rewriting the cached conversation."""
    session = _session()
    for n in range(1, 5):                       # four builds of nine rounds: 20 messages each
        api.script = [_tool_round(10 * n + k) for k in range(9)] + [_weighing(30_000 + 5_000 * n)]
        _drive(session, text=f"build {n}")
    assert len(session.history) == 80, "past sixty messages, and 50,000 tokens"
    api.script = [_weighing(52_000)]
    _drive(session, text="and then")
    assert _opening(session.history[0]) == "build 1" and len(session.history) == 82


def test_a_trim_never_cuts_into_the_newest_finished_turn(api):
    session = _session()
    api.script = [_tool_round(1), _weighing(500_000)]
    _drive(session, text="one enormous turn")
    api.script = [_weighing(500_000)]
    _drive(session, text="next")
    assert _opening(session.history[0]) == "one enormous turn", "the turn before was cut into"


def test_haiku_is_trimmed_before_its_dearer_tier(api):
    session = ChatSession(model="claude-haiku-5-5", reasoning="medium",
                          tool_defs_for=lambda during_run: TOOLS)
    for n, prompt in enumerate((30_000, 55_000, 80_000), 1):
        api.script = [_weighing(prompt)]
        _drive(session, text=f"turn {n}")
    api.script = [_weighing(50_000)]
    _drive(session, text="turn 4")
    kept = [_opening(m) for m in session.history if m["role"] == "user"]
    assert kept[0] != "turn 1" and kept[-1] == "turn 4", kept


# ---------------------------------------------------------------- 3e: the conversation's cache

def test_the_conversations_breakpoints_live_an_hour(api):
    session = _session()
    api.script = [_tool_round(1), FINAL]
    _drive(session)
    for body in api.requests:
        marks = [b["cache_control"] for m in body["messages"] if isinstance(m["content"], list)
                 for b in m["content"] if isinstance(b, dict) and "cache_control" in b]
        assert marks and all(mark == {"type": "ephemeral", "ttl": "1h"} for mark in marks)
        system = [b["cache_control"] for b in body["system"] if "cache_control" in b]
        assert len(system) + len(marks) <= 4, "the API takes four breakpoints"


# ---------------------------------------------------------------- 3g: what one turn may spend

def _costly(n: int) -> dict:
    """A tool round that generated 300,000 tokens: $6.00 of Opus 5.5 output."""
    return _tool_round(n, out=300_000)


def test_a_turn_that_has_spent_five_dollars_stops_and_asks(api, monkeypatch):
    monkeypatch.delenv(chatmod.TURN_SPEND_PAUSE_ENV, raising=False)
    session, tools, events = _session(), _Tools(), []
    api.script = [_costly(1), _tool_round(2), FINAL]
    _drive(session, tools=tools, events=events)
    assert tools.ran == ["track.set_gain"], "what the costly round asked for ran; nothing after"
    assert len(api.requests) == 1
    asked = [e for e in events if e["type"] == "chat_question"]
    assert len(asked) == 1 and asked[0]["options"] == ["Keep going", "Stop here"]
    assert "$6.0" in asked[0]["question"] and "is applied" in asked[0]["question"]
    assert session.history[-1]["role"] == "assistant"
    assert "paused this turn" in session.history[-1]["content"][0]["text"]
    # the click is the next message, and the turn it starts has its own five dollars
    api.script = [_tool_round(2), FINAL]
    events.clear()
    _drive(session, text="Keep going", tools=tools, events=events)
    assert len(tools.ran) == 2 and api.rejected == []
    assert not [e for e in events if e["type"] == "chat_question"]


def test_the_pause_can_be_moved_or_switched_off(api, monkeypatch):
    monkeypatch.setenv(chatmod.TURN_SPEND_PAUSE_ENV, "0")
    session, tools, events = _session(), _Tools(), []
    api.script = [_costly(1), FINAL]
    _drive(session, tools=tools, events=events)
    assert len(api.requests) == 2 and not [e for e in events if e["type"] == "chat_question"]
    monkeypatch.setenv(chatmod.TURN_SPEND_PAUSE_ENV, "20")
    assert chatmod.turn_spend_pause_usd() == 20.0
    monkeypatch.setenv(chatmod.TURN_SPEND_PAUSE_ENV, "not a number")
    assert chatmod.turn_spend_pause_usd() == chatmod.TURN_SPEND_PAUSE_USD == 5.0
