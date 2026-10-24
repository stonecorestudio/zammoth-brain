"""Phase 55, step 2a: the mix is measured once per response, and only what the delta reads.

The timed run-through spent 105 of its 627 seconds on this: a fader, a mute, a send or a knob
re-rendered and re-measured the whole mix when its own reply came back - ten mutes in one
response were ten measures, 63 s - though the model reads a response's results together and
cannot act between two of them. And each measure computed the song's tempo, key and band
energy, which the delta then skipped.

A command now marks the measure OWED; `ChatTools.after_calls` takes it once, when the
response's calls are all in, and reads loudness, true peak and tilt alone.
"""
from __future__ import annotations

import asyncio
import json
import sys
import types

import anyio
import pytest

import orchestrator.chat as chatmod
import orchestrator.daw_bridge as B
from mix_engine import analysis, dsp
from orchestrator.chat import AFTER_CALLS_NOTE, CATALOG_LABEL, LIVE_NOTE, ChatSession, ChatTools
from orchestrator.turn_timing import TurnTimer
from tests.fake_anthropic import serve
from tests.test_phase54_render_once import FACT, MOVE, _masters, _project, _turn, kept  # noqa: F401


class _Responses:
    """Stands in for ChatSession: each RESPONSE is a list of tool calls run through the real
    `execute_tool`, and - as the real loop does - the host's after-calls note is asked for
    once when a response's calls are all in."""

    def __init__(self, responses):
        self.responses, self.history, self.usage, self._model = list(responses), [], {}, ""
        self.log: list[tuple] = []
        self.notes: list[str] = []
        self.timing = TurnTimer()

    def set_settings(self, model, reasoning):
        pass

    async def turn(self, text, *, tools, context, emit, should_cancel=None,
                   live_context=None, during_run=False):
        self.history.append({"role": "user", "content": text})
        for calls in self.responses:
            for name, args in calls:
                out = await chatmod.execute_tool(tools, name, args)
                self.log.append((name, out.text, out.error))
            hook = getattr(tools, "after_calls", None)
            self.notes.append(await hook() if hook is not None else "")
        await emit({"type": "chat_message", "text": "ok"})


def _run(proj, responses, coedits, monkeypatch, brain=None):
    """`_turn` with a host that speaks in responses. Returns (session, out, masters)."""
    import tests.test_phase54_render_once as base
    monkeypatch.setattr(base, "_Scripted", lambda script: _Responses(script))
    session, out, masters, _ = _turn(proj, responses, coedits, monkeypatch, brain=brain)
    return session, out, masters


def _gain(track="bass"):
    return ("daw_command", {"command": "track.set_gain", "args": {"track": track}})


def _trim():
    return ("daw_command", {"command": "clip.trim", "args": {"track": "bass", "clip": 0}})


def test_three_level_commands_in_one_response_are_measured_once(tmp_path, monkeypatch, kept):
    proj = _project(tmp_path)
    B.session_readout(proj, {})
    session, out, masters = _run(proj, [[_gain(), _gain(), _gain()]],
                                 {"track.set_gain": [MOVE]}, monkeypatch)
    assert out["timing"]["mark_counts"].get("measure") == 1, "one response, one measurement"
    assert masters == 1, "and one master render"
    assert all("measured" not in json.loads(text) for _, text, _ in session.log)
    assert len(session.notes) == 1 and session.notes[0].startswith("measured after these commands: ")
    assert "loudness" in session.notes[0] or "true peak" in session.notes[0]


def test_two_responses_that_each_move_a_level_are_two_measurements(tmp_path, monkeypatch, kept):
    proj = _project(tmp_path)
    B.session_readout(proj, {})
    coedits = {"track.set_gain": [MOVE], "track.set_mute": ["/api/set_stem?stem=bass&muted=1"]}
    mute = ("daw_command", {"command": "track.set_mute", "args": {"track": "bass"}})
    session, out, _ = _run(proj, [[_gain()], [mute]], coedits, monkeypatch)
    assert out["timing"]["mark_counts"].get("measure") == 2
    assert all(note.startswith("measured after these commands") for note in session.notes)


def test_a_level_command_after_a_clip_edit_is_still_measured(tmp_path, monkeypatch, kept):
    """The baseline was read from the brain's render cache, which the brain empties at every
    mirror: once a clip edit's report had been mirrored on its own, no level command after it
    in that turn was measured at all. It is measured now, as plain figures - the clip edit
    moved the mix unmeasured, so there is no honest before-and-after."""
    proj = _project(tmp_path)
    B.session_readout(proj, {})
    session, out, _ = _run(proj, [[_trim()], [_gain()]],
                           {"clip.trim": [FACT], "track.set_gain": [MOVE]}, monkeypatch)
    assert out["timing"]["mark_counts"].get("measure") == 1
    assert session.notes[0] == "", "a fact report owes no measurement"
    said = session.notes[1]
    assert "not a before-and-after" in said and "LUFS" in said and "->" not in said


def test_a_response_of_reads_and_fact_reports_owes_nothing(tmp_path, monkeypatch, kept):
    proj = _project(tmp_path)
    session, out, _ = _run(proj, [[_trim(), _trim()]], {"clip.trim": [FACT]}, monkeypatch)
    assert session.notes == [""] and "measure" not in out["timing"]["marks_s"]


def test_the_measure_reads_loudness_peak_and_tilt_and_not_the_songs_tempo_or_key(
        tmp_path, monkeypatch, kept):
    proj = _project(tmp_path)
    brain = B_brain(proj)
    B.session_readout(proj, brain.render_cache)       # the turn's own context is then kept
    asked: list[str] = []
    tempo, key = analysis._tempo, analysis._key
    monkeypatch.setattr(analysis, "_tempo", lambda *a, **k: asked.append("tempo") or tempo(*a, **k))
    monkeypatch.setattr(analysis, "_key", lambda *a, **k: asked.append("key") or key(*a, **k))
    session, out, _ = _run(proj, [[_gain()]], {"track.set_gain": [MOVE]}, monkeypatch, brain=brain)
    assert out["timing"]["mark_counts"].get("measure") == 1
    assert asked == [], f"a move's measurement computed {asked}"
    levels = B.level_readout(proj)
    assert set(levels) == {"lufs_i", "true_peak_dbtp", "spectral_tilt_db_per_oct", "silent"}
    full = B.session_readout(proj, {})
    assert {k: full[k] for k in levels} == levels, "the same three figures the readout gives"


def B_brain(proj):
    import orchestrator.daw_server as S
    return S.Brain(proj)


def test_a_host_that_never_asks_leaves_the_measure_to_the_turns_end(tmp_path, monkeypatch, kept):
    """The evals' hosts and a turn ended by Stop never call the hook. Nothing is lost: the
    turn's delta is settled when it ends, and its feed row is not suppressed."""
    proj = _project(tmp_path)
    B.session_readout(proj, {})
    import tests.test_phase54_render_once as base
    monkeypatch.setattr(base._Scripted, "asks_after_calls", False)   # tool calls and no hook
    session, out, masters, _ = _turn(proj, [_gain()], {"track.set_gain": [MOVE]}, monkeypatch)
    assert masters == 0 and "measure" not in out["timing"]["marks_s"]
    pending = out["turn_delta_pending"]
    assert pending["settle"] == "chat_done" and pending["measured_midturn"] is False


def test_the_record_holds_the_price_and_a_failed_turn_is_recorded(tmp_path, monkeypatch, kept):
    proj = _project(tmp_path)
    _, out, _ = _run(proj, [[_trim()]], {"clip.trim": [FACT]}, monkeypatch)
    assert out["timing"]["cost_usd"] == out["cost_usd"] and out["timing"]["v"] == 3

    class _Dies(_Responses):
        async def turn(self, *a, **k):
            raise RuntimeError("the request died")
    import tests.test_phase54_render_once as base
    monkeypatch.setattr(base, "_Scripted", lambda script: _Dies(script))
    with pytest.raises(Exception) as caught:      # the turn's task group wraps what it raised
        _turn(proj, [], {}, monkeypatch)
    timing = caught.value.timing
    assert timing["failed"] and "wall_s" in timing and "cost_usd" in timing


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


class _Host(ChatTools):
    def __init__(self, notes):
        self.ran: list[str] = []
        self.asks: list[int] = []      # a list: the session works on a shallow copy of the host
        self._notes = list(notes)

    @property
    def asked(self) -> int:
        return len(self.asks)

    async def daw_command(self, command, args):
        self.ran.append(command)
        return {"ok": True}

    async def after_calls(self):
        self.asks.append(1)
        return self._notes.pop(0) if self._notes else ""


def _three_calls() -> dict:
    return {"blocks": [("thinking", "")] + [
        ("tool_use", f"toolu_{n}", "daw_command", {"command": "track.set_gain", "args": {"n": n}})
        for n in range(3)]}


def _drive(session, host, text="pull the vocals down", live=None, should_cancel=None):
    async def emit(_event):
        pass
    anyio.run(lambda: session.turn(text, tools=host, context=CONTEXT, emit=emit,
                                   live_context=live, should_cancel=should_cancel))


def test_the_api_loop_asks_once_per_response_and_puts_the_note_after_the_results(api):
    session = ChatSession(reasoning="medium", tool_defs_for=lambda during_run: TOOLS)
    api.script = [_three_calls(), {"blocks": [("thinking", ""), ("text", "done")]}]
    host = _Host(["measured after these commands: loudness -14.0 -> -15.2"])
    live = iter(["Live DAW edit (daw_edit_state): {\"seq\": 1}",
                 "Live DAW edit (daw_edit_state): {\"seq\": 2}",
                 "Live DAW edit (daw_edit_state): {\"seq\": 2}"])
    _drive(session, host, live=lambda: next(live))
    assert host.ran == ["track.set_gain"] * 3 and host.asked == 1
    results = session.history[2]["content"]
    kinds = [b["type"] for b in results]
    assert kinds[:4] == ["tool_result", "tool_result", "tool_result", "text"]
    assert results[3]["text"] == (AFTER_CALLS_NOTE
                                  + "measured after these commands: loudness -14.0 -> -15.2")
    assert results[4]["text"].startswith(LIVE_NOTE), "the live edit's refresh comes after it"
    assert api.rejected == [], "the note is stored with the results: the history stayed append-only"
    sent = api.requests[1]["messages"][2]["content"]
    assert [b["type"] for b in sent][:4] == kinds[:4]


def test_a_blank_note_adds_no_block_and_a_failing_hook_loses_no_result(api):
    session = ChatSession(reasoning="medium", tool_defs_for=lambda during_run: TOOLS)
    api.script = [_three_calls(), _three_calls(), {"blocks": [("thinking", ""), ("text", "done")]}]

    class _Fails(_Host):
        async def after_calls(self):
            self.asks.append(1)
            if self.asked == 2:
                raise RuntimeError("the render died")
            return "   "
    host = _Fails([])
    _drive(session, host)
    assert host.asked == 2 and len(host.ran) == 6
    for message in (session.history[2], session.history[4]):
        assert [b["type"] for b in message["content"]] == ["tool_result"] * 3
    assert api.rejected == []


def test_the_note_is_not_asked_for_after_stop(api):
    session = ChatSession(reasoning="medium", tool_defs_for=lambda during_run: TOOLS)
    api.script = [_three_calls()]
    stop = {"now": False}

    class _StopsAfterOne(_Host):
        async def daw_command(self, command, args):
            stop["now"] = True
            return await super().daw_command(command, args)
    host = _StopsAfterOne(["never said"])
    _drive(session, host, should_cancel=lambda: stop["now"])
    assert host.ran == ["track.set_gain"] and host.asked == 0


# ---------------------------------------------------------------- the subscription loop

def _stream_sdk(plan):
    """A `claude_agent_sdk` whose `query` plays `plan(tools, m)`: an async generator that
    yields the SDK's messages and may start a tool handler itself."""
    m = types.ModuleType("claude_agent_sdk")

    class TextBlock:
        def __init__(self, text):
            self.text = text

    class AssistantMessage:
        def __init__(self, content, message_id=None, usage=None):
            self.content, self.message_id, self.usage = content, message_id, usage

    class ResultMessage:
        usage = None

    class StreamEvent:
        def __init__(self, event):
            self.event = event

    m.TextBlock, m.AssistantMessage = TextBlock, AssistantMessage
    m.ResultMessage, m.StreamEvent = ResultMessage, StreamEvent
    m.ClaudeAgentOptions = lambda **kw: types.SimpleNamespace(**kw)
    m.tool = lambda name, desc, schema: (lambda fn: types.SimpleNamespace(name=name, handler=fn))
    m.create_sdk_mcp_server = lambda name, version, tools: {"tools": {t.name: t for t in tools}}

    async def query(prompt, options):
        async for message in plan(options.mcp_servers["mixagent"]["tools"], m):
            yield message
    m.query = query
    return m


def _block(m, i):
    return m.StreamEvent({"type": "content_block_start", "index": i,
                          "content_block": {"type": "tool_use", "id": f"t{i}", "name": "daw_command"}})


def test_the_subscription_loop_measures_on_a_responses_last_call_only(monkeypatch):
    """The CLI starts call 1 while the model is still streaming call 2. "Measure when the
    response has not been seen to close" measured after call 1 AND after call 2."""
    monkeypatch.setenv("MIX_AGENT_LLM_BACKEND", "subscription")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "test-token-not-real")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(chatmod, "STOP_POLL_S", 0.01)
    results: list[str] = []

    async def plan(tools, m):
        handler = tools["daw_command"].handler
        args = {"command": "track.set_gain", "args": {}}
        yield m.StreamEvent({"type": "message_start", "message": {"id": "m1", "usage": {}}})
        yield _block(m, 0)
        first = asyncio.ensure_future(handler(args))     # call 1 starts mid-stream
        await asyncio.sleep(0.05)                        # ...and its command has come back
        yield _block(m, 1)                               # only now is call 2 announced
        yield m.StreamEvent({"type": "message_delta", "delta": {"stop_reason": "tool_use"},
                             "usage": {"output_tokens": 40}})
        yield m.StreamEvent({"type": "message_stop"})
        results.append((await first)["content"][0]["text"])
        results.append((await handler(args))["content"][0]["text"])
        yield m.StreamEvent({"type": "message_start", "message": {"id": "m2", "usage": {}}})
        yield m.StreamEvent({"type": "message_stop"})
        yield m.AssistantMessage([m.TextBlock("done")], message_id="m2")
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", _stream_sdk(plan))

    session = chatmod.ChatSession(client=object(), tool_defs_for=lambda during_run: TOOLS)
    host = _Host(["measured after these commands: loudness -14.0 -> -15.2", "a second measure"])
    _drive(session, host)
    assert len(host.ran) == 2
    assert host.asked == 1, "one response, one measurement"
    assert AFTER_CALLS_NOTE not in results[0]
    assert results[1].endswith(AFTER_CALLS_NOTE
                               + "measured after these commands: loudness -14.0 -> -15.2")
    assert session.timing.rounds[0]["calls"] == 2
