"""Phase 54, step 0: a chat turn records where its seconds went.

Until 2026-10-12 nothing logged a round's timing. That a 23-minute arrange turn was 40 rounds
of one clip command each, and that most of each round was the brain re-measuring the mix,
had to be rebuilt by hand from one-second timestamps in brain.log. `turn_timing.TurnTimer`
records it as the turn runs, on both transports; one JSON line a turn lands in
`runs/turn_timing.jsonl` and `python -m orchestrator.turn_timing <project>` prints the table.
"""
from __future__ import annotations

import json

import anyio
import pytest

import orchestrator.chat as chatmod
from orchestrator import turn_timing
from orchestrator.chat import CATALOG_LABEL, ChatSession, ChatTools
from orchestrator.turn_timing import TurnTimer
from tests.fake_anthropic import serve


class _Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def test_a_round_is_split_into_the_wait_and_the_stream():
    clock = _Clock()
    timer = TurnTimer(clock)
    timer.begin_round()
    clock.t += 3.0
    timer.stream_started()
    clock.t += 5.0
    timer.stream_started()          # a later event of the same response moves nothing
    timer.end_round(120)
    s = timer.summary()
    assert s["rounds"] == 1 and s["out_tokens"] == 120
    assert timer.rounds[0] == {"wait_s": 3.0, "stream_s": 5.0, "out": 120, "calls": 0}
    assert s["model_s"] == 8.0 and s["first_byte_median_s"] == 3.0


def test_a_tool_call_belongs_to_the_round_that_asked_for_it():
    clock = _Clock()
    timer = TurnTimer(clock)
    timer.begin_round(); clock.t += 1; timer.stream_started(); clock.t += 1; timer.end_round(10)
    for key in ("track.set_gain", "track.set_gain", "clip.trim"):
        clock.t += 2.0
        timer.tool("daw_command", 2.0, key)
    timer.begin_round(); clock.t += 1; timer.stream_started(); clock.t += 1; timer.end_round(5)
    s = timer.summary()
    assert [r["calls"] for r in timer.rounds] == [3, 0]
    assert s["rounds_with_several_calls"] == 1 and s["tool_calls"] == 3 and s["tools_s"] == 6.0
    assert s["slowest_tools"][0] == {"tool": "track.set_gain", "n": 2, "s": 4.0}


def test_a_round_with_no_begin_is_dated_from_when_the_model_was_handed_the_turn():
    """The subscription path: the SDK sends the request, so a round opens at its first
    stream event and its wait runs from the end of the last tool."""
    clock = _Clock()
    timer = TurnTimer(clock)
    clock.t += 4.0                   # the request is out; nothing has come back
    timer.stream_started()
    clock.t += 2.0
    timer.end_round()
    timer.add_output(40)             # usage that arrives after the response's last event
    timer.tool("analyze_session", 6.0)
    clock.t += 9.0
    timer.stream_started()
    timer.add_output(7)              # ...and usage that arrives before it
    clock.t += 1.0
    assert timer.rounds[0] == {"wait_s": 4.0, "stream_s": 2.0, "out": 40, "calls": 1}
    s = timer.summary()              # the open round is closed, never lost
    assert s["rounds"] == 2 and timer.rounds[1]["wait_s"] == 9.0 and timer.rounds[1]["out"] == 7


def test_the_brains_own_work_is_summed_and_counted():
    timer = TurnTimer(_Clock())
    timer.mark("measure", 4.5)
    timer.mark("measure", 5.5)
    timer.mark("prep", 1.25)
    s = timer.summary()
    assert s["marks_s"] == {"measure": 10.0, "prep": 1.25}
    assert s["mark_counts"] == {"measure": 2, "prep": 1}
    said = turn_timing.line("t0005", {**s, "renders": {"master": 2, "stem": 9}, "reply_words": 41})
    assert "t0005 done" in said and "measure 10.0 s x2" in said
    assert "renders master 2 stem 9" in said and "reply 41 words" in said


# ---------------------------------------------------------------- the production loop

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


def _tool_round(n: int, command: str) -> dict:
    return {"blocks": [("thinking", ""), ("tool_use", f"toolu_{n}", "daw_command",
                                          {"command": command, "args": {}})]}


def test_the_api_loop_records_every_round_and_every_tool(api):
    session = ChatSession(reasoning="high", tool_defs_for=lambda during_run: TOOLS)
    api.script = [_tool_round(1, "track.set_gain"), _tool_round(2, "clip.trim"),
                  {"blocks": [("thinking", ""), ("text", "done")]}]
    tools = _Tools()

    async def emit(_event):
        pass
    anyio.run(lambda: session.turn("pull the vocal down", tools=tools, context=CONTEXT, emit=emit))

    assert tools.ran == ["track.set_gain", "clip.trim"]
    s = session.timing.summary()
    assert s["rounds"] == 3 and s["tool_calls"] == 2
    assert [r["calls"] for r in session.timing.rounds] == [1, 1, 0]
    assert sorted(t["tool"] for t in s["slowest_tools"]) == ["clip.trim", "track.set_gain"]
    assert s["out_tokens"] > 0, "a round's usage is its tokens"
    assert session.turn_id == "t0001"

    api.script = [{"blocks": [("thinking", ""), ("text", "nothing to do")]}]
    anyio.run(lambda: session.turn("and now?", tools=tools, context=CONTEXT, emit=emit))
    assert session.timing.summary()["rounds"] == 1, "a new turn starts a new record"
    assert session.turn_id == "t0002"


# ---------------------------------------------------------------- the file and the table

def test_a_turn_is_one_line_in_the_projects_runs_folder_and_the_file_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(turn_timing, "TURN_TIMING_KEEP", 3)
    for n in range(5):
        turn_timing.record(tmp_path, {"turn": f"t{n:04d}", "session_id": "s1", "wall_s": 10.0 + n,
                                      "rounds": n, "model_s": 5.0, "tools_s": 2.0, "tool_calls": 1,
                                      "out_tokens": 100, "reply_words": 30 + n,
                                      "marks_s": {"prep": 0.5, "measure": 4.0, "tail": 0.1},
                                      "mark_counts": {"measure": 2},
                                      "renders": {"master": 1, "stem": 2}, "user": "Mix it."})
    lines = (tmp_path / "runs" / turn_timing.TURN_TIMING_FILE).read_text().splitlines()
    assert [json.loads(x)["turn"] for x in lines] == ["t0002", "t0003", "t0004"]
    assert all("ts" in json.loads(x) for x in lines)
    out = turn_timing.table(tmp_path)
    assert "t0004" in out and "4x2" in out and "1/2" in out and "Mix it." in out
    assert "3 turn(s)" in out


def test_a_project_from_before_the_recorder_still_reports_its_reply_lengths(tmp_path):
    chats = tmp_path / "chats"
    chats.mkdir()
    (chats / "index.json").write_text(json.dumps({"active": "s9"}))
    (chats / "s9.json").write_text(json.dumps([
        {"k": "user", "t": "Split both songs into stems."},
        {"k": "tool", "key": "track.separate"},
        {"k": "prose", "t": "a note between tools"},
        {"k": "tool", "key": "track.separate"},
        {"k": "prose", "t": "one two three four five"}]))
    turns = turn_timing.feed_replies(tmp_path)
    assert turns[0]["tools"] == 2 and turns[0]["reply_words"] == 5 and turns[0]["note_words"] == 4
    out = turn_timing.table(tmp_path)
    assert "no recorded turns" in out and "reply    5 words" in out
