"""The Anthropic history contract: every `tool_use` block must be answered by a
`tool_result` with the same id, or the API rejects EVERY later request in that
conversation.

Found by the 2026-08-05 audit. The turn loop returned early when a turn stopped
for any reason other than `tool_use` — but by then the assistant message with its
tool_use blocks was already in history, and the early return skipped the
result-appending block (and its `finally` backfill). One turn truncated by
`max_tokens` mid-tool-use therefore bricked that project's chat permanently:
`_CHAT_SESSIONS` lives for the process and `reset_chat_session` is never called,
so the only cure was restarting the brain.

A turn cut off by max_tokens can also carry a half-written tool_use input, so the
fix must close those blocks WITHOUT running them.
"""
import anyio
import pytest

from orchestrator.chat import ChatSession, ChatTools


class Block:
    def __init__(self, type, **kw):
        self.type = type
        for k, v in kw.items():
            setattr(self, k, v)


class FakeMessage:
    def __init__(self, content, stop_reason):
        self.content = content
        self.stop_reason = stop_reason


class FakeStream:
    def __init__(self, message):
        self._message = message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        async def _empty():
            return
            yield  # pragma: no cover — streamed deltas aren't what this pins
        return _empty()

    async def get_final_message(self):
        return self._message


class FakeMessages:
    def __init__(self, messages):
        self._messages = list(messages)
        self.calls = 0

    def stream(self, **kwargs):
        self.calls += 1
        return FakeStream(self._messages.pop(0))


class FakeClient:
    """Stands in for AsyncAnthropic (ChatSession takes an injected client)."""

    def __init__(self, *messages):
        self.messages = FakeMessages(messages)
        self.api_key = None


class RecordingTools(ChatTools):
    """Every tool call is a failure for these cases — nothing should run."""

    def __init__(self):
        self.ran = []


def _tool_defs(_during_run):
    return [{"name": "propose", "description": "x",
             "input_schema": {"type": "object"}}]


def run_turn(session, tools):
    events = []

    async def emit(event):
        events.append(event)

    async def main():
        await session.turn("make the vocal sit", tools=tools, context="",
                           emit=emit)

    anyio.run(main)
    return events


def _tool_use_ids(history):
    return {b.id for m in history if m["role"] == "assistant"
            for b in m["content"] if getattr(b, "type", None) == "tool_use"}


def _tool_result_ids(history):
    return {b["tool_use_id"] for m in history if m["role"] == "user"
            and isinstance(m["content"], list) for b in m["content"]
            if b.get("type") == "tool_result"}


@pytest.mark.parametrize("stop_reason", ["max_tokens", "refusal", "end_turn"])
def test_a_truncated_tool_use_is_still_answered(stop_reason):
    """The whole point: history stays valid whatever the turn stopped for."""
    truncated = FakeMessage(
        [Block("text", text="one moment"),
         Block("tool_use", id="tu_1", name="propose", input={"stem": "voc"})],
        stop_reason=stop_reason)
    session = ChatSession(client=FakeClient(truncated), tool_defs_for=_tool_defs)

    run_turn(session, RecordingTools())

    assert _tool_use_ids(session.history) == _tool_result_ids(session.history), (
        "a tool_use with no matching tool_result 400s every later turn")


def test_a_truncated_tool_use_is_not_executed():
    """max_tokens can cut a tool_use's input JSON in half — running it would
    apply a half-parsed move."""
    truncated = FakeMessage(
        [Block("tool_use", id="tu_1", name="propose", input={"stem": "voc"})],
        stop_reason="max_tokens")
    tools = RecordingTools()
    session = ChatSession(client=FakeClient(truncated), tool_defs_for=_tool_defs)

    run_turn(session, tools)

    assert tools.ran == []
    # the tool results alone: on the default model the user's own message is a list too (its
    # text, then the notes the chat stores beside it - the 5.5 family's append-only layout)
    results = [b for m in session.history if m["role"] == "user"
               and isinstance(m["content"], list) for b in m["content"]
               if b.get("type") == "tool_result"]
    assert results and all(b.get("is_error") for b in results)


def test_the_user_is_told_rather_than_left_guessing():
    truncated = FakeMessage(
        [Block("tool_use", id="tu_1", name="propose", input={})],
        stop_reason="max_tokens")
    session = ChatSession(client=FakeClient(truncated), tool_defs_for=_tool_defs)

    events = run_turn(session, RecordingTools())

    said = " ".join(e.get("text", "") for e in events
                    if e["type"] == "chat_message")
    assert "nothing was applied" in said


def test_the_session_survives_for_the_next_turn():
    """The regression that mattered: the NEXT message must still work. A dangling
    id would make every later turn 400 for the life of the process."""
    truncated = FakeMessage(
        [Block("tool_use", id="tu_1", name="propose", input={})],
        stop_reason="max_tokens")
    ordinary = FakeMessage([Block("text", text="done")], stop_reason="end_turn")
    session = ChatSession(client=FakeClient(truncated, ordinary),
                          tool_defs_for=_tool_defs)

    run_turn(session, RecordingTools())
    run_turn(session, RecordingTools())

    assert session._client.messages.calls == 2
    assert _tool_use_ids(session.history) == _tool_result_ids(session.history)


def test_a_clean_turn_with_no_tools_appends_nothing_extra():
    clean = FakeMessage([Block("text", text="the vocal sits fine")],
                        stop_reason="end_turn")
    session = ChatSession(client=FakeClient(clean), tool_defs_for=_tool_defs)

    run_turn(session, RecordingTools())

    assert _tool_result_ids(session.history) == set()
    assert session.history[-1]["role"] == "assistant"
