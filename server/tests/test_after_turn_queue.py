"""Commands that wait for the agent's own turn to end (readers C4·R6 = C2·R2, 2026-10-03).

An agent's command only ever arrives while its own turn holds the DAW's RunThread, and eleven
handlers refused whenever that thread was running: `track.separate`, `agent.reference`,
`agent.advise`, `agent.undo`, every cross-project `project.*`, `edit.open`. From the agent they
could never succeed — five of them after the user had clicked "Do it" — while the catalog
offered them and the operate eval graded reaching for them against a stub that said ok.

The DAW now QUEUES those commands and runs them the moment the turn ends (`--after-turn` drives
that on the real handlers). This file holds the brain's half, and the three texts that have to
agree: the DAW's reply note, the eval's canned reply, and the prompt.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from orchestrator import daw_bridge, operate_eval
from orchestrator import daw_server as server
from orchestrator.chat import DAW_SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parents[2]
APP = (ROOT / "daw" / "Source" / "App.cpp").read_text(encoding="utf-8")
COMMANDS_H = (ROOT / "daw" / "Source" / "DawCommands.h").read_text(encoding="utf-8")


def _cpp_string(function: str) -> str:
    """The adjacent string literals a `return "…" "…";` in `function` concatenates."""
    body = COMMANDS_H.split(f"inline juce::String {function}()", 1)[1].split("}", 1)[0]
    return "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', body))


def _enum_for(key: str) -> str:
    row = re.search(r'\{\s*(\w+),\s*"' + re.escape(key) + r'"', COMMANDS_H)
    assert row, f"{key} is not in the command table"
    return row.group(1)


def _handler(key: str) -> str:
    """The text of `key`'s case in invokeCommand, up to the next case."""
    marker = f"case dawcmd::{_enum_for(key)}:"
    start = APP.index(marker, APP.index("juce::var invokeCommand (int id"))
    nxt = re.search(r"\n\s*case dawcmd::\w+:", APP[start + len(marker):])
    return APP[start: start + len(marker) + (nxt.start() if nxt else 4000)]


def test_the_note_is_one_wording_in_the_daw_and_in_the_eval():
    assert _cpp_string("afterTurnNote") == operate_eval.AFTER_TURN_NOTE


@pytest.mark.parametrize("key", operate_eval.AFTER_TURN_KEYS)
def test_each_command_that_waits_queues_in_its_own_handler(key):
    body = _handler(key)
    assert "queueAfterTurn (" in body, f"{key} does not queue behind the turn"
    # ...ahead of any refusal for a busy thread: the queue must win for the agent's own turn
    refusal = body.find("runThread.isThreadRunning()) return err")
    assert refusal < 0 or body.index("queueAfterTurn (") < refusal, key


def test_no_handler_refuses_a_busy_thread_without_asking_whose_turn_it_is():
    """The regression this class of bug comes back through: a new brain-backed command that
    writes `if (runThread.isThreadRunning()) return err (busy)` and is thereby dead to the
    agent. Every such line sits right under the own-turn check."""
    lines = APP.splitlines()
    start = next(i for i, l in enumerate(lines) if "juce::var invokeCommand (int id" in l)
    bad = []
    for i in range(start, len(lines)):
        if "runThread.isThreadRunning()) return err" not in lines[i]:
            continue
        above = "\n".join(lines[max(i - 14, 0): i + 1])
        if "inOwnTurn()" not in above and "agentTurnActive()" not in above:
            bad.append(f"App.cpp:{i + 1}: {lines[i].strip()[:90]}")
    assert not bad, "a busy refusal the agent can never get past:\n" + "\n".join(bad)


def test_the_agents_catalog_reads_the_two_run_keys_disabled():
    enabled = APP.split("bool commandEnabled (int id)", 1)[1].split("case dawcmd::undoAgent:", 1)[0]
    assert "case dawcmd::runAgent: case dawcmd::autoMix:" in enabled
    assert "return catalogForAgent ? false : ! busy;" in enabled
    fixture = {e["key"]: e["enabled"] for e in operate_eval.OPERATE_EDIT["commands"]}
    assert sorted(k for k, on in fixture.items() if not on) == sorted(operate_eval.AGENT_DISABLED_KEYS)
    for key in operate_eval.AGENT_DISABLED_KEYS:
        assert "ownTurnRunRefusal" in _handler(key), key


def test_the_prompt_names_every_command_that_waits_and_what_can_be_read_now():
    for key in operate_eval.AFTER_TURN_KEYS:
        assert f"`{key}`" in DAW_SYSTEM_PROMPT, key
    assert '"queued": true' in DAW_SYSTEM_PROMPT
    assert "`project.peek {dir}`" in DAW_SYSTEM_PROMPT
    assert "busy while your own turn runs" not in DAW_SYSTEM_PROMPT   # the line that sent it nowhere


def test_the_eval_answers_those_keys_the_way_the_daw_does():
    for key in operate_eval.AFTER_TURN_KEYS:
        reply = operate_eval.CANNED_WRITE_REPLIES[key]({})
        assert reply == {"queued": True, "note": operate_eval.AFTER_TURN_NOTE}, key
    peek = operate_eval.CANNED_READ_REPLIES["project.peek"]
    assert peek["tracks"] and "started" not in peek


def test_a_queued_command_is_not_an_edit_of_the_turn(demo_project):
    """`_turn_daw_ops` decides whether the turn closes an undo gesture. Counting the REQUEST for
    agent.undo made the turn a gesture of its own, so the queued undo found the newest gesture
    was the turn that had asked for it and took back nothing."""
    brain = server.Brain(demo_project)
    replies = {"agent.undo": {"ok": True, "queued": True, "note": operate_eval.AFTER_TURN_NOTE},
               "clip.split": {"ok": True}}
    brain.daw_invoke = lambda command, args=None, *a, **k: dict(replies[command])
    brain._turn_daw_ops = 0
    invoke = brain.counting_invoke()
    assert invoke("agent.undo", {})["queued"] is True
    assert brain._turn_daw_ops == 0
    invoke("clip.split", {"track": "stem_a", "clip": 0, "at_s": 1.0})
    assert brain._turn_daw_ops == 1                       # control: a real edit still counts


class _Scripted:
    def __init__(self, calls):
        self._calls, self.results = calls, []

    async def turn(self, message, *, tools, context, emit, during_run=False,
                   should_cancel=None, live_context=None):
        from orchestrator.chat import execute_tool
        for name, args in self._calls:
            self.results.append(await execute_tool(tools, name, args))
        await emit({"type": "chat_message", "text": "done"})


def test_project_peek_is_read_inside_the_turn(demo_project, tmp_path):
    """The agent's `project.peek` never reaches the DAW: the brain reads the folder and the
    listing IS the tool result."""
    import shutil

    other = tmp_path / "older_song"
    shutil.copytree(demo_project, other)
    asked = []
    turn = _Scripted([("daw_command", {"command": "project.peek", "args": {"dir": str(other)}}),
                      ("daw_command", {"command": "project.peek", "args": {"dir": "older_song"}}),
                      ("daw_command", {"command": "project.peek", "args": {"dir": "no_such_song"}})])
    daw_bridge._CHAT_SESSIONS[str(demo_project)] = turn
    try:
        daw_bridge.chat_turn(demo_project, "what's inside older_song?", mode="agent",
                             daw_invoke=lambda c, a: asked.append(c) or {"ok": True})
    finally:
        daw_bridge._CHAT_SESSIONS.clear()
    assert "project.peek" not in asked
    listing = json.loads(turn.results[0].text)
    assert listing["ok"] is True and listing["name"] == "older_song"
    assert sorted(t["stem_id"] for t in listing["tracks"]) == ["stem_a", "stem_b"]
    assert all("colour" not in t for t in listing["tracks"])
    # A NAME is read where projects live (reader E7·R2, 2026-10-06): the brain resolves a
    # relative `dir` itself. This used to be refused for not being an absolute path - and
    # "older_song" is the user's own word for it.
    by_name = json.loads(turn.results[1].text)
    assert by_name["ok"] is True and by_name["name"] == "older_song"
    assert Path(by_name["dir"]) == other
    refused = json.loads(turn.results[2].text)
    assert refused["ok"] is False and "not a folder" in refused["error"]
    assert str(demo_project / "no_such_song") in refused["error"]
    assert turn.results[1].error is False and turn.results[2].error is True
