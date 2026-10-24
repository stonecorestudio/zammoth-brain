"""Phase 55, step 1: the turn record tells the truth.

The Phase 54 recorder said where a turn's seconds went and got the tokens wrong. On the
subscription transport - the one every timed session so far ran on - `out_tokens` was each
response's count at its START (a seven-round turn recorded 65), and on neither transport was
there an input figure, a cache figure, a stop reason or a price. A turn that failed recorded
nothing at all. "Is the model thinking or writing" could not be answered from the file.
"""
from __future__ import annotations

import sys
import types

import anyio
import pytest

import orchestrator.chat as chatmod
from orchestrator import turn_timing
from orchestrator.chat import CATALOG_LABEL, ChatSession, ChatTools
from orchestrator.llm_subscription import SubscriptionLLM, _Usage
from orchestrator.turn_timing import TurnTimer
from tests.fake_anthropic import serve


class _Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def _round(timer: TurnTimer, clock: _Clock, **kw) -> None:
    timer.begin_round(); clock.t += 1; timer.stream_started(); clock.t += 1
    timer.end_round(**kw)


def test_a_round_keeps_what_it_read_wrote_and_how_it_stopped():
    clock = _Clock()
    timer = TurnTimer(clock)
    _round(timer, clock, output_tokens=400, stop_reason="tool_use",
           usage={"input_tokens": 200, "cache_read_input_tokens": 40_000,
                  "cache_creation_input_tokens": 3_000,
                  "cache_creation": {"ephemeral_1h_input_tokens": 1_000},
                  "output_tokens_details": {"thinking_tokens": 250}})
    assert timer.rounds[0] == {"wait_s": 1.0, "stream_s": 1.0, "out": 400, "calls": 0,
                               "in": 200, "read": 40_000, "write": 3_000, "write_1h": 1_000,
                               "think": 250, "stop": "tool_use"}
    s = timer.summary()
    assert (s["in_tokens"], s["cache_read_tokens"], s["cache_write_tokens"]) == (200, 40_000, 3_000)
    assert s["cache_write_1h_tokens"] == 1_000 and s["thinking_tokens"] == 250
    assert s["cache_share"] == round(40_000 / 43_200, 3) and s["stops"] == {"tool_use": 1}
    assert "cache_cold_rounds" not in s


def test_a_round_nobody_metered_adds_no_token_figures():
    """A zero that was never measured is not a measurement: the host that reports no usage
    (a scripted chat, an old fake) records exactly what it always did."""
    clock = _Clock()
    timer = TurnTimer(clock)
    _round(timer, clock, output_tokens=12)
    assert timer.rounds[0] == {"wait_s": 1.0, "stream_s": 1.0, "out": 12, "calls": 0}
    s = timer.summary()
    assert not {"in_tokens", "cache_share", "thinking_tokens", "stops"} & set(s)


def test_a_cold_round_after_the_first_is_flagged_and_a_cold_first_round_is_not():
    """The first request of a turn may write the whole conversation (a pause past the cache's
    lifetime). A LATER round that reads under half its prompt paid for it again."""
    warm = {"input_tokens": 100, "cache_read_input_tokens": 50_000, "cache_creation_input_tokens": 900}
    cold = {"input_tokens": 100, "cache_read_input_tokens": 9_000, "cache_creation_input_tokens": 41_000}
    clock = _Clock()
    first_cold = TurnTimer(clock)
    _round(first_cold, clock, usage=cold); _round(first_cold, clock, usage=warm)
    assert "cache_cold_rounds" not in first_cold.summary()
    later_cold = TurnTimer(clock)
    _round(later_cold, clock, usage=warm); _round(later_cold, clock, usage=cold)
    s = later_cold.summary()
    assert s["cache_cold_rounds"] == 1
    assert "1 COLD round(s)" in turn_timing.line("t0002", s)


def test_a_call_that_comes_back_while_its_response_still_streams_is_filed_on_that_response():
    """The subscription transport's CLI starts a tool as soon as its block is complete. The
    call used to be credited to the round BEFORE it (or to none, in a turn's first round)."""
    clock = _Clock()
    timer = TurnTimer(clock)
    timer.stream_started()
    timer.announce_call(); timer.announce_call()
    assert not timer.last_call_back
    timer.tool("daw_command", 0.5, "track.set_gain")      # call 1, mid-stream
    assert not timer.last_call_back, "its response is still open"
    timer.end_round(30)
    assert not timer.last_call_back, "call 2 has not come back"
    timer.tool("daw_command", 0.5, "track.set_gain")      # call 2, after the response ended
    assert timer.last_call_back
    assert [r["calls"] for r in timer.rounds] == [2]
    timer.stream_started()                                # the next response: a clean count
    assert not timer.last_call_back
    timer.end_round(5)
    assert timer.last_call_back and [r["calls"] for r in timer.rounds] == [2, 0]


def test_a_request_that_died_keeps_its_seconds_and_is_counted_as_failed():
    clock = _Clock()
    timer = TurnTimer(clock)
    timer.begin_round(); clock.t += 4.0
    timer.fail_round("overloaded_error")
    _round(timer, clock, output_tokens=9)
    s = timer.summary()
    assert s["rounds"] == 2 and s["failed_rounds"] == 1 and s["model_s"] == 6.0
    assert timer.rounds[0]["failed"] == "overloaded_error"


# ---------------------------------------------------------------- the subscription consumer

def _sdk(monkeypatch):
    m = types.ModuleType("claude_agent_sdk")

    class TextBlock:
        def __init__(self, text):
            self.text = text

    class AssistantMessage:
        def __init__(self, content, message_id=None, usage=None):
            self.content, self.message_id, self.usage = content, message_id, usage

    class ResultMessage:
        def __init__(self, usage=None):
            self.usage = usage

    class StreamEvent:
        def __init__(self, event):
            self.event = event

    m.TextBlock, m.AssistantMessage = TextBlock, AssistantMessage
    m.ResultMessage, m.StreamEvent = ResultMessage, StreamEvent
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", m)
    return m


def _consume(messages: list, timer: TurnTimer) -> _Usage:
    usage = _Usage()

    async def run() -> None:
        async def stream():
            for message in messages:
                yield message
        await SubscriptionLLM._consume(stream(), [], usage, None, True, timer=timer)
    anyio.run(run)
    return usage


def test_the_subscription_round_records_the_tokens_the_response_ended_with(monkeypatch):
    """`message_start` says 3 output tokens (the count as the response begins) and
    `message_delta` 900 (what it generated). The record held the 3."""
    m = _sdk(monkeypatch)
    start = {"type": "message_start", "message": {"id": "msg_1", "usage": {
        "input_tokens": 120, "cache_read_input_tokens": 44_000,
        "cache_creation_input_tokens": 2_000, "output_tokens": 3}}}
    events = [
        m.StreamEvent(start),
        m.StreamEvent({"type": "content_block_start", "index": 0,
                       "content_block": {"type": "tool_use", "id": "t1", "name": "daw_command"}}),
        # the CLI hands the message over block by block, with the usage it STARTED with
        m.AssistantMessage([], message_id="msg_1", usage={"input_tokens": 120, "output_tokens": 3}),
        m.StreamEvent({"type": "message_delta", "delta": {"stop_reason": "tool_use"},
                       "usage": {"output_tokens": 900}}),
        m.StreamEvent({"type": "message_stop"}),
        m.ResultMessage(usage={"input_tokens": 120, "output_tokens": 900}),
    ]
    timer = TurnTimer()
    _consume(events, timer)
    assert len(timer.rounds) == 1
    r = timer.rounds[0]
    assert r["out"] == 900, f"the record holds the response's opening count: {r}"
    assert (r["in"], r["read"], r["write"], r["stop"]) == (120, 44_000, 2_000, "tool_use")
    assert timer.summary()["out_tokens"] == 900


def test_a_stream_that_carries_no_final_count_falls_back_to_the_messages_own(monkeypatch):
    """Whether every CLI build's `message_delta` carries usage is not established: without
    it the round keeps what the message reported, once."""
    m = _sdk(monkeypatch)
    events = [
        m.StreamEvent({"type": "message_start", "message": {"id": "msg_1", "usage": {}}}),
        m.AssistantMessage([], message_id="msg_1", usage={"input_tokens": 5, "output_tokens": 41}),
        m.StreamEvent({"type": "message_delta", "delta": {"stop_reason": "end_turn"}}),
        m.StreamEvent({"type": "message_stop"}),
        m.AssistantMessage([], message_id="msg_1", usage={"input_tokens": 5, "output_tokens": 41}),
    ]
    timer = TurnTimer()
    _consume(events, timer)
    assert [r["out"] for r in timer.rounds] == [41]


def test_the_consumer_announces_each_tool_call_of_the_response(monkeypatch):
    m = _sdk(monkeypatch)

    def block(i: int) -> dict:
        return {"type": "content_block_start", "index": i,
                "content_block": {"type": "tool_use", "id": f"t{i}", "name": "daw_command"}}
    timer = TurnTimer()
    _consume([m.StreamEvent({"type": "message_start", "message": {"id": "m", "usage": {}}}),
              m.StreamEvent(block(0)), m.StreamEvent(block(1)),
              m.StreamEvent({"type": "content_block_start", "index": 2,
                             "content_block": {"type": "text", "text": ""}}),
              m.StreamEvent({"type": "message_stop"})], timer)
    assert timer._announced == 2 and not timer.last_call_back
    timer.tool("daw_command", 0.1); timer.tool("daw_command", 0.1)
    assert timer.last_call_back


# ---------------------------------------------------------------- the production API loop

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
        pass

    async def daw_command(self, command, args):
        return {"ok": True}


def test_the_api_loop_records_each_rounds_usage_and_stop_reason(api):
    session = ChatSession(reasoning="medium", tool_defs_for=lambda during_run: TOOLS)
    api.script = [
        {"blocks": [("thinking", ""), ("tool_use", "toolu_1", "daw_command",
                                       {"command": "track.set_gain", "args": {}})],
         "usage": {"input_tokens": 300, "cache_read_input_tokens": 0,
                   "cache_creation_input_tokens": 45_000}, "out": 700},
        {"blocks": [("thinking", ""), ("text", "done")],
         "usage": {"input_tokens": 50, "cache_read_input_tokens": 45_300,
                   "cache_creation_input_tokens": 800}, "out": 120}]

    async def emit(_event):
        pass
    anyio.run(lambda: session.turn("pull the vocal down", tools=_Tools(), context=CONTEXT, emit=emit))
    rounds = session.timing.rounds
    assert [(r["in"], r["read"], r["write"], r["out"], r["stop"]) for r in rounds] == [
        (300, 0, 45_000, 700, "tool_use"), (50, 45_300, 800, 120, "end_turn")]
    s = session.timing.summary()
    assert s["out_tokens"] == 820 and s["stops"] == {"tool_use": 1, "end_turn": 1}
    assert s["cache_share"] == round(45_300 / 91_450, 3) and "cache_cold_rounds" not in s


# ---------------------------------------------------------------- the file and the table

def test_the_table_prints_the_prompt_the_cache_share_the_price_and_a_failed_turn(tmp_path):
    base = {"session_id": "s1", "rounds": 2, "model_s": 5.0, "tools_s": 2.0, "tool_calls": 1,
            "out_tokens": 820, "reply_words": 30, "marks_s": {"prep": 0.5}, "mark_counts": {},
            "renders": {"master": 1, "stem": 0}, "user": "Mix it."}
    turn_timing.record(tmp_path, {**base, "turn": "t0001", "wall_s": 12.0, "in_tokens": 350,
                                  "cache_read_tokens": 45_300, "cache_write_tokens": 45_800,
                                  "cache_share": 0.495, "cost_usd": 0.41})
    turn_timing.record(tmp_path, {**base, "turn": "t0002", "wall_s": 9.0, "in_tokens": 100,
                                  "cache_read_tokens": 9_000, "cache_write_tokens": 41_000,
                                  "cache_share": 0.18, "cache_cold_rounds": 1, "cost_usd": 0.2,
                                  "failed": "RuntimeError"})
    out = turn_timing.table(tmp_path)
    row1 = next(x for x in out.splitlines() if x.startswith("t0001"))
    row2 = next(x for x in out.splitlines() if x.startswith("t0002"))
    assert "91450" in row1 and "50%" in row1 and "0.41" in row1
    assert "18%!" in row2 and "FAILED Mix it." in row2
    assert "$0.61" in out and "the brain's own share" in out and "5 s" in out
