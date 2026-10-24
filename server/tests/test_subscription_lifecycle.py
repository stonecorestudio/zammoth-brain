"""Stop and failure on the subscription chat path — the 2026-09-10 agent audit's P1-11
to P1-13, driven through the real ChatSession + SubscriptionLLM against a scripted
stand-in for the Claude Agent SDK (the real one is dev-only, and would bill).

- P1-11: Stop never reached `_turn_subscription`; a build kept applying commands until
  the DAW killed the brain 45 s later.
- P1-12: the idle ceiling re-armed only on SDK messages, and a running tool sends none,
  so a long chat-started run was cut off while it kept applying, orphaned.
- P1-13: a turn that failed after tools ran left only the user's request in the text
  history, so the next turn (or the feed's retry) did it again.

The fake SDK calls tool handlers the way the SDK does — through the in-process tool
functions `_chat_tools` registers — so every test runs the shipping code between them.
"""
from __future__ import annotations

import asyncio
import inspect
import sys
import time
import types
from pathlib import Path

import pytest

from orchestrator import chat as chatmod
from orchestrator import llm_subscription as subm
from orchestrator.llm_subscription import render_history

REPO = Path(__file__).resolve().parents[2]


def _fake_sdk(steps: list[tuple]) -> types.ModuleType:
    """A `claude_agent_sdk` whose `query` plays `steps`:
    ("text", s) · ("tool", name, args) — the SDK runs the handler, then streams on ·
    ("bg_tool", name, args) — a handler still running when the request dies ·
    ("sleep", s) — the model thinking · ("raise", msg) — the request dies."""
    m = types.ModuleType("claude_agent_sdk")

    class TextBlock:
        def __init__(self, text):
            self.text = text

    class AssistantMessage:
        def __init__(self, content):
            self.content = content

    class ResultMessage:
        usage = None

    class StreamEvent:
        event = None

    m.TextBlock, m.AssistantMessage = TextBlock, AssistantMessage
    m.ResultMessage, m.StreamEvent = ResultMessage, StreamEvent
    m.ClaudeAgentOptions = lambda **kw: types.SimpleNamespace(**kw)
    m.tool = lambda name, desc, schema: (lambda fn: types.SimpleNamespace(name=name, handler=fn))
    m.create_sdk_mcp_server = lambda name, version, tools: {"tools": {t.name: t for t in tools}}
    m.handler_tasks = []

    async def query(prompt, options):
        tools = options.mcp_servers["mixagent"]["tools"]
        for kind, *rest in steps:
            if kind == "text":
                yield AssistantMessage([TextBlock(rest[0])])
            elif kind == "tool":
                await tools[rest[0]].handler(rest[1])
                yield AssistantMessage([])
            elif kind == "bg_tool":
                m.handler_tasks.append(asyncio.ensure_future(tools[rest[0]].handler(rest[1])))
                await asyncio.sleep(0)
            elif kind == "sleep":
                await asyncio.sleep(rest[0])
            elif kind == "raise":
                raise RuntimeError(rest[0])
    m.query = query
    return m


def _cmd(name: str) -> tuple:
    return ("tool", "daw_command", {"command": name, "args": {}})


def _tools(applied: list, delay: float = 0.0) -> chatmod.ChatTools:
    async def unsupported(*_a, **_k):
        return "unsupported"

    async def daw_command(command, args):
        if delay:   # a worker thread, the way a chat-started run executes
            await asyncio.to_thread(time.sleep, delay)
        applied.append(command)
        return {"ok": True}

    return chatmod.ChatTools(
        engine_call=unsupported, propose=unsupported, start_run=unsupported,
        stop_run=unsupported, auto_mix=unsupported, file_instruction=unsupported,
        save_note=unsupported, workspace_command=unsupported, list_files=unsupported,
        import_sample=unsupported, undo_proposal=unsupported, render=unsupported,
        daw_command=daw_command)


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setenv("MIX_AGENT_LLM_BACKEND", "subscription")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "test-token-not-real")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return chatmod.ChatSession(client=object(), tool_defs_for=chatmod.daw_tool_defs)


def _turn(session, text, tools, should_cancel=None) -> list[dict]:
    events: list[dict] = []

    async def emit(event):
        events.append(event)

    asyncio.run(session.turn(text, tools=tools, context="ctx", emit=emit,
                             should_cancel=should_cancel))
    return events


# ------------------------------------------------------------------ P1-11: Stop

def test_stop_ends_a_subscription_turn_between_tools(session, monkeypatch):
    applied: list[str] = []
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", _fake_sdk(
        [_cmd("a"), ("sleep", 0.6), _cmd("b"), _cmd("c"), ("text", "all three done")]))
    events = _turn(session, "build the rack", _tools(applied),
                   should_cancel=lambda: bool(applied))   # Stop, once the first lands
    assert applied == ["a"], "nothing after Stop runs"
    assert any(e.get("text") == chatmod.STOPPED_NOTICE for e in events)
    note = session.history[-1]["content"]
    assert "pressed Stop" in note and "daw_command" in note


def test_a_tool_call_after_stop_is_refused_not_run(session, monkeypatch):
    applied: list[str] = []
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", _fake_sdk(
        [_cmd("a"), _cmd("b"), _cmd("c"), ("text", "stopped after a")]))
    _turn(session, "build the rack", _tools(applied), should_cancel=lambda: bool(applied))
    assert applied == ["a"]


# ------------------------------------------------------------------ P1-12: idle ceiling

def test_a_long_tool_does_not_trip_the_idle_ceiling(session, monkeypatch):
    monkeypatch.setattr(subm, "SDK_REQUEST_TIMEOUT_S", 0.3)
    applied: list[str] = []
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", _fake_sdk(
        [_cmd("start_run"), ("text", "mixed")]))
    _turn(session, "mix it", _tools(applied, delay=1.0))
    assert applied == ["start_run"]
    assert session.history[-1] == {"role": "assistant", "content": "mixed"}


def test_a_silent_model_still_trips_it(session, monkeypatch):
    """The control: the guard times the MODEL's silence, and still does."""
    monkeypatch.setattr(subm, "SDK_REQUEST_TIMEOUT_S", 0.3)
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", _fake_sdk(
        [("sleep", 1.0), ("text", "too late")]))
    with pytest.raises(RuntimeError, match="went silent"):
        _turn(session, "hello", _tools([]))


def test_a_dying_request_never_orphans_a_running_tool(session, monkeypatch):
    """The turn must not end — releasing the pass slot and the cancel handle — while a
    tool it started is still running in a worker thread."""
    applied: list[str] = []
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", _fake_sdk(
        [("bg_tool", "daw_command", {"command": "start_run", "args": {}}),
         ("sleep", 0.1), ("raise", "Claude Code returned an error result: success")]))
    with pytest.raises(chatmod.TurnCutOff):
        _turn(session, "mix it", _tools(applied, delay=0.6))
    assert applied == ["start_run"], "the turn returned before its tool finished"


# ------------------------------------------------------------------ P1-13: history + retry

def test_a_failed_turn_leaves_what_already_ran_in_history(session, monkeypatch):
    applied: list[str] = []
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", _fake_sdk(
        [_cmd("edit.delete_time"), ("raise", "Claude Code returned an error result: success")]))
    with pytest.raises(chatmod.TurnCutOff, match="already applied"):
        _turn(session, "delete the last 8 bars", _tools(applied))
    assert applied == ["edit.delete_time"]
    prompt = render_history(session.history
                            + [{"role": "user", "content": "now bring the vocal up 1 dB"}])
    assert "edit.delete_time" in prompt and "do not repeat them" in prompt


def test_a_failure_before_any_write_is_not_dressed_up(session, monkeypatch):
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", _fake_sdk(
        [("raise", "usage limit reached")]))
    with pytest.raises(RuntimeError) as err:
        _turn(session, "hello", _tools([]))
    assert not isinstance(err.value, chatmod.TurnCutOff)
    assert session.history[-1] == {"role": "user", "content": "hello"}


def test_an_ask_user_question_rides_the_text_history(session, monkeypatch):
    """P1-17: the question lived only in the tool call, and the tool tells the agent to end
    its turn after asking — so the next prompt held a bare "Tighter" answering nothing."""
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", _fake_sdk(
        [("tool", "ask_user", {"question": "Should the kick be tighter or boomier?",
                               "options": ["Tighter", "Boomier"]}),
         ("text", "")]))
    events = _turn(session, "fix the kick", _tools([]))
    assert any(e.get("type") == "chat_question" for e in events)
    prompt = render_history(session.history + [{"role": "user", "content": "Tighter"}])
    assert "Should the kick be tighter or boomier?" in prompt
    assert "Tighter | Boomier" in prompt


def test_the_daw_withholds_retry_when_moves_already_landed():
    """The seam, both ends. It was a PHRASE ("already applied") the app searched the failure
    for; since the 2026-10-04 pre-merge review the brain says it with a type - `NoRetry`, sent
    as `retryable: false` - and the app goes by that, never by the words."""
    from orchestrator import daw_server, llm_backend

    cut = chatmod.TurnCutOff(RuntimeError("x"), 1)
    assert "already applied" in str(cut)                       # the user still reads why
    assert not llm_backend.retry_helps(cut)
    assert daw_server._failure(cut)["retryable"] is False
    assert "raise NoRetry(" in inspect.getsource(daw_server.Brain._raise_if_landed)
    src = (REPO / "daw/Source/App.cpp").read_text()
    block = src[src.index("void setStatus (juce::String s, bool retryable"):][:5000]
    assert "const bool canRetry = retryable && c->runBrief.isNotEmpty();" in block
    assert 's.contains ("already applied")' not in block


def test_a_cancelled_caller_does_not_abandon_the_tool_it_started(session, monkeypatch):
    """D3 (2026-09-11 pre-merge review): the shield above has its own gate, and it was blind.
    `test_a_dying_request_never_orphans_a_running_tool` passes with `asyncio.shield` deleted,
    because a request that DIES raises — and settle() waits for the tool either way. The shield
    exists for the other half: the caller being CANCELLED, which propagates INTO the tool and
    abandons it half-way in a worker thread. So cancel the handler while the tool is in flight."""
    applied: list[str] = []
    fake = _fake_sdk([("bg_tool", "daw_command", {"command": "start_run", "args": {}}),
                      ("sleep", 0.6), ("raise", "Claude Code returned an error result: success")])
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake)
    events: list[dict] = []

    async def emit(event):
        events.append(event)

    async def cancel_the_caller():
        while not fake.handler_tasks:      # …once the tool is actually running
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        for task in fake.handler_tasks:
            task.cancel()

    async def drive():
        killer = asyncio.ensure_future(cancel_the_caller())
        with pytest.raises(chatmod.TurnCutOff):
            await session.turn("mix it", tools=_tools(applied, delay=0.3), context="ctx",
                               emit=emit)
        await killer

    asyncio.run(drive())
    assert applied == ["start_run"], "a cancelled caller took its running tool down with it"
