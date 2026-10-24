"""Phase 55, step 4: the engineer's code, restructured with no change in behaviour.

These pin the SHAPE, so it does not grow back: one implementation of what a tool call does,
used by the live turn and by an approved plan; a turn that is a class with its state as
fields, not one function of closures. What the code DOES is held by every test that was
there before the move (1,879 across the files that touch the bridge) and by steps 1-3's.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import anyio

import orchestrator.daw_bridge as B
from orchestrator import turn_host

REPO = Path(__file__).resolve().parents[2]


def _lines(obj) -> int:
    return len(inspect.getsource(obj).splitlines())


def test_a_turn_is_a_class_and_chat_turn_is_its_entry():
    assert _lines(B.chat_turn) < 60, "chat_turn grew back: it was 600 lines of closures"
    assert "LiveTurn(" in inspect.getsource(B.chat_turn)
    longest = max((_lines(fn), name) for name, fn in vars(turn_host.LiveTurn).items()
                  if inspect.isfunction(fn))
    assert longest[0] <= 110, f"{longest[1]} is {longest[0]} lines"
    assert "nonlocal" not in inspect.getsource(turn_host)


def test_what_a_tool_call_does_has_one_implementation():
    plan, live = inspect.getsource(B.apply_plan), inspect.getsource(turn_host.LiveTurn)
    for shared in ("checked_command(", "registering_command(", "propose_and_apply("):
        assert shared in plan and shared in live, shared
    # ...and neither door carries its own copy of what those do
    for own in ('"propose_actions"', '"apply_proposal"', "_duplicate_track_step(",
                "brain_only_refusal(", "resolve_path_args("):
        assert own not in plan, f"apply_plan has its own {own}"
    assert plan.count("_import_sample_step(") == 1, "only the import step's own call"
    source = inspect.getsource(turn_host.LiveTurn.do_daw_command)
    assert "_duplicate_track_step(" not in source and "brain_only_refusal(" not in source


def test_a_command_is_refused_before_its_paths_are_read(tmp_path):
    """One order at both doors: the plan's copy resolved the paths first and refused second."""
    key = sorted(B.BRAIN_ONLY_COMMANDS)[0]
    refused, args = turn_host.checked_command(tmp_path, key, {"file": "mix.wav"}, None)
    assert refused is not None and refused.get("ok") is False
    assert args == {"file": "mix.wav"}, "a refused command's arguments are left as they came"
    refused, args = turn_host.checked_command(tmp_path, "session.export", {"file": "mix.wav"}, None)
    assert refused is None and Path(args["file"]).is_absolute()


def test_a_command_that_makes_a_stem_is_one_function_and_the_rest_are_not_its_business(tmp_path):
    seen: list = []

    def daw_invoke(command, args=None, *a, **k):
        seen.append(command)
        return {"ok": True}

    async def ask(command, args):
        return await turn_host.registering_command(tmp_path, command, args, daw_invoke=daw_invoke,
                                                   register_stem=None, mirror_coedits=None)
    assert anyio.run(ask, "track.set_gain", {"track": "bass"}) is None
    assert anyio.run(ask, "track.add", {}) is None, "an empty track is no stem"
    assert anyio.run(ask, "library.import", {"track": "drums", "note": 36}) is None
    assert seen == []


def test_the_turn_reaches_the_bridge_through_the_module():
    """A name bound at import would not see what a test or an eval patches on daw_bridge."""
    source = (REPO / "orchestrator" / "turn_host.py").read_text()
    assert "from .daw_bridge import" not in source and "from . import daw_bridge as B" in source


# ---------------------------------------------------------------- 4b: the chat session

def test_a_tool_call_is_run_by_one_wrapper_on_both_transports():
    from orchestrator.chat import ChatSession
    api, sub = (inspect.getsource(getattr(ChatSession, name))
                for name in ("_turn_api_rounds", "_turn_subscription"))
    for loop in (api, sub):
        assert "self._call_tool(" in loop
        assert "execute_tool(" not in loop and "_tool_done_event(" not in loop
    assert inspect.getsource(ChatSession._call_tool).count("await emit(") == 2


def test_the_api_loop_has_one_layout_and_one_stop_ending():
    from orchestrator import chat
    loop = inspect.getsource(chat.ChatSession._turn_api_rounds)
    assert not hasattr(chat.ChatSession, "_preserved")
    assert list(inspect.signature(chat._request_messages).parameters) == ["history"]
    assert loop.count("await self._stopped(turn_id, emit)") == 2 and "_STOPPED_HISTORY" not in loop


# ---------------------------------------------------------------- 4c: what was removed

def test_the_brain_does_not_import_the_eval_harness():
    """`in_memory_engine` lived in orchestrator/benchmark.py, so the shipped brain imported an
    eval module to open the engine for a turn."""
    for name in ("daw_server.py", "daw_bridge.py", "turn_host.py", "chat.py", "loop.py", "agents.py"):
        source = (REPO / "orchestrator" / name).read_text()
        assert "from .benchmark import" not in source and "orchestrator.benchmark" not in source, name
    from orchestrator import benchmark, engine_client
    assert benchmark.in_memory_engine is engine_client.in_memory_engine
