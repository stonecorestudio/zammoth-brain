"""Phase 56, step 1: the timing record keeps what it used to sum away (record v3).

The second showcase's run-through (2026-10-12) was read from a record that could not answer
four plain questions. How long before the user saw anything? Which tool is slow, call by
call, rather than "these eight had the largest totals"? Of a 191-second `session.export`, how
much was the app working and how much a person reading an approval box (187 s of it)? And why
do a turn's parts add up to 14 s more than the turn (the subscription transport runs a call
while its response still streams, so those seconds were in both columns)?

Record v3 is additive: every v2 field means what it did, and `TurnTimer.rounds` /
`TurnTimer.tools` keep their shape (two older test files compare them to exact dicts).
"""
from __future__ import annotations

import asyncio
import json
import sys

import anyio
import pytest

import orchestrator.chat as chatmod
import orchestrator.daw_bridge as B
from orchestrator import llm_models, turn_timing
from orchestrator.chat import CATALOG_LABEL, ChatSession, ChatTools
from orchestrator.turn_timing import TurnTimer
from tests.fake_anthropic import serve


class _Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def _round(timer: TurnTimer, clock: _Clock, wait: float = 1.0, stream: float = 1.0, **kw) -> None:
    timer.begin_round(); clock.t += wait; timer.stream_started(); clock.t += stream
    timer.end_round(**kw)


# ---------------------------------------------------------------- the recorder

def test_every_round_is_kept_as_a_row_with_when_it_began():
    clock = _Clock()
    timer = TurnTimer(clock)
    clock.t += 2.0                                   # the context was built first
    _round(timer, clock, wait=1.5, stream=3.0, output_tokens=90, stop_reason="tool_use")
    clock.t += 4.0                                   # a tool ran
    _round(timer, clock, wait=0.5, stream=1.0, output_tokens=20, stop_reason="end_turn")
    s = timer.summary()
    assert s["rounds"] == 2, "v2's count is still a count"
    assert s["round_rows"] == [
        {"at_s": 2.0, "wait_s": 1.5, "stream_s": 3.0, "out": 90, "calls": 0, "stop": "tool_use"},
        {"at_s": 10.5, "wait_s": 0.5, "stream_s": 1.0, "out": 20, "calls": 0, "stop": "end_turn"}]
    assert "at_s" not in timer.rounds[0], "the list the turn reads keeps its shape"


def test_every_tool_is_kept_not_the_eight_with_the_largest_totals():
    clock = _Clock()
    timer = TurnTimer(clock)
    for i in range(11):                              # eleven different tools: v2 kept eight
        clock.t += 1.0
        timer.tool("daw_command", 1.0, f"key.{i:02d}")
    clock.t += 5.0
    timer.tool("daw_command", 5.0, "key.00")
    s = timer.summary()
    assert len(s["slowest_tools"]) == 8 and len(s["tools"]) == 11
    assert s["tools"]["key.00"] == {"n": 2, "total_s": 6.0, "max_s": 5.0}
    assert s["tools"]["key.10"] == {"n": 1, "total_s": 1.0, "max_s": 1.0}
    assert len(s["tool_rows"]) == 12 and s["tool_rows"][-1] == ["key.00", 5.0]


def test_seconds_that_are_in_the_model_and_in_a_tool_are_said():
    """The subscription transport: the CLI starts a call while its response is still
    arriving. Those seconds were in `model_s` AND in `tools_s`, and a turn's parts added up
    to more than the turn."""
    clock = _Clock()
    timer = TurnTimer(clock)
    timer.stream_started()                           # the response opens at 100
    timer.announce_call()
    clock.t += 7.0                                   # ...and at 107 a call that took 3 s is back
    timer.tool("analyze_stem", 3.0)
    clock.t += 3.0                                   # the response ends at 110
    timer.end_round(50)
    clock.t += 2.0
    timer.tool("render", 2.0)                        # a call after the response: no overlap
    s = timer.summary()
    assert s["model_s"] == 10.0 and s["tools_s"] == 5.0
    assert s["overlap_s"] == 3.0
    assert s["model_s"] + s["tools_s"] - s["overlap_s"] == 12.0 == clock.t - 100.0


def test_a_turn_whose_calls_follow_their_response_overlaps_nothing():
    clock = _Clock()
    timer = TurnTimer(clock)
    _round(timer, clock, output_tokens=10)
    clock.t += 2.0
    timer.tool("daw_command", 2.0, "track.set_gain")
    _round(timer, clock, output_tokens=5)
    assert timer.summary()["overlap_s"] == 0.0


def test_the_first_thing_the_user_sees_is_dated_once():
    clock = _Clock()
    timer = TurnTimer(clock)
    assert timer.first_activity_at is None
    clock.t += 1.25
    timer.activity()
    clock.t += 9.0
    timer.activity()                                 # a later word moves nothing
    assert timer.first_activity_at == 101.25


def test_the_model_that_answered_and_the_accounts_limits_are_kept():
    timer = TurnTimer(_Clock())
    timer.note_served("claude-opus-5-5", {
        "anthropic-ratelimit-requests-limit": "1000",
        "anthropic-ratelimit-input-tokens-limit": "2000000",
        "anthropic-ratelimit-output-tokens-limit": "400000",
        "anthropic-ratelimit-input-tokens-remaining": "1912000",
        "retry-after": "3"})
    timer.note_served("claude-opus-5-5", {"anthropic-ratelimit-input-tokens-remaining": "1700000",
                                          "anthropic-ratelimit-input-tokens-limit": "not a number"})
    timer.note_served(None, None)                    # a response with neither changes nothing
    s = timer.summary()
    assert s["served_model"] == "claude-opus-5-5"
    assert s["rate_limits"] == {"rpm": 1000, "itpm": 2_000_000, "otpm": 400_000,
                                "itpm_left": 1_700_000}, "what is left is kept at its lowest"


def test_a_turn_nobody_served_says_nothing_about_a_model_or_a_limit():
    s = TurnTimer(_Clock()).summary()
    assert "served_model" not in s and "rate_limits" not in s


# ---------------------------------------------------------------- the wire's view

def test_the_apps_time_and_a_persons_click_are_told_apart():
    timing = {"tools_s": 200.0, "marks_s": {"prep": 4.0, "measure": 3.0}, "v": 3}
    turn_timing.merge_commands(timing, [
        {"key": "session.export", "wait_s": 190.9, "app_s": 190.7, "confirm_s": 187.0},
        {"key": "track.bounce", "wait_s": 4.0, "app_s": 3.9},
        {"key": "track.bounce", "wait_s": 5.0, "app_s": 4.8},
        {"key": "track.set_mute", "wait_s": 0.02}])              # an app that stamped nothing
    assert timing["commands"]["track.bounce"] == {"n": 2, "wait_s": 9.0, "app_s": 8.7}
    assert timing["commands"]["session.export"]["confirm_s"] == 187.0
    assert "app_s" not in timing["commands"]["track.set_mute"]
    assert timing["confirm_wait_s"] == 187.0
    assert turn_timing.brain_share_s(timing) == 20.0, \
        "a person reading an approval box is nobody's work"


def test_a_turn_that_asked_the_app_nothing_adds_nothing():
    timing = {"tools_s": 1.0}
    turn_timing.merge_commands(timing, [])
    assert timing == {"tools_s": 1.0}


# ---------------------------------------------------------------- the tables and the budget

def _session(tmp_path, walls=(10.0, 30.0), **extra):
    for i, wall in enumerate(walls, 1):
        turn_timing.record(tmp_path, {
            "v": 3, "session_id": "s1", "turn": f"t{i:04d}", "wall_s": wall, "rounds": 2,
            "model_s": wall * 0.5, "tools_s": wall * 0.3, "tool_calls": 2, "out_tokens": 100,
            "marks_s": {"prep": 1.0, "measure": 2.0}, "mark_counts": {"measure": 1},
            "renders": {"master": 1, "stem": 2}, "first_activity_s": 1.0 + i, "overlap_s": 0.5,
            "tool_rows": [["track.bounce", wall * 0.2], ["analyze_stem", wall * 0.1]],
            "round_rows": [{"at_s": 1.0, "wait_s": 0.8, "stream_s": 2.0, "out": 60, "calls": 2},
                           {"at_s": 9.0, "wait_s": 1.2, "stream_s": 1.0, "out": 40, "calls": 0}],
            "commands": {"track.bounce": {"n": 1, "wait_s": wall * 0.2, "app_s": wall * 0.19}},
            "served_model": "claude-opus-5-5", "rate_limits": {"itpm": 2_000_000},
            "user": "Mix it.", **extra})
    return tmp_path


def test_the_detail_prints_the_rounds_every_tools_spread_and_the_apps_own_time(tmp_path):
    out = turn_timing.detail(_session(tmp_path))
    assert "4 round(s): first byte p50 0.80 s, p95 1.20 s" in out
    assert "200 output tokens in 6 s of streaming (33 a second)" in out
    bounce = next(x for x in out.splitlines() if x.startswith("track.bounce") and " 2 " in x)
    assert "8.0" in bounce and "6.00" in bounce, bounce          # total 2 + 6, the longest 6
    asked = [x for x in out.splitlines() if x.startswith("track.bounce")][-1]
    assert "7.60" in asked and "0.40" in asked, asked            # in the app 7.6 of 8.0 waited
    first = next(x for x in out.splitlines() if x.startswith("t0002"))
    assert " 3.0 " in first and "claude-opus-5-5 itpm 2,000,000" in first


def test_a_session_over_its_budget_fails_and_one_inside_it_passes(tmp_path):
    fast, slow = _session(tmp_path / "fast"), _session(tmp_path / "slow", walls=(10.0, 60.0))
    budget = tmp_path / "budget.json"
    written = turn_timing.write_budget(fast, budget, slack=1.25)
    assert written["agent_time_s"] == 50.0 and written["rounds"] == 5, written
    ok, said = turn_timing.check(fast, budget)
    assert ok and "OVER" not in said
    ok, said = turn_timing.check(slow, budget)
    assert not ok and "OVER  agent_time_s: 70.0 (budget 50.0)" in said
    assert turn_timing.main([str(slow), "--check", str(budget)]) == 1
    assert turn_timing.main([str(fast), "--check", str(budget)]) == 0


def test_a_shorter_session_is_not_under_budget_for_nothing(tmp_path):
    """Three of six turns are under every limit of a six-turn budget. That is not a pass."""
    full, short = _session(tmp_path / "full"), _session(tmp_path / "short", walls=(10.0,))
    budget = tmp_path / "budget.json"
    turn_timing.write_budget(full, budget, slack=1.25)
    ok, said = turn_timing.check(short, budget)
    assert not ok and "the budget is for 2 turn(s), this session has 1" in said


def test_a_number_the_budget_names_and_the_session_lacks_is_a_failure(tmp_path):
    full = _session(tmp_path / "full")
    budget = tmp_path / "budget.json"
    turn_timing.write_budget(full, budget, slack=1.25)
    old = tmp_path / "old"                           # a v2 session: nothing dated the first word
    for i in (1, 2):
        turn_timing.record(old, {"v": 2, "session_id": "s1", "turn": f"t{i:04d}", "wall_s": 5.0,
                                 "rounds": 1, "model_s": 2.0, "tools_s": 1.0, "tool_calls": 1,
                                 "out_tokens": 10, "marks_s": {}, "renders": {}})
    ok, said = turn_timing.check(old, budget)
    assert not ok and "first_activity_max_s: not measured in this session" in said


def test_two_sessions_are_compared_number_by_number_and_tool_by_tool(tmp_path):
    out = turn_timing.compare(_session(tmp_path / "a", walls=(10.0, 60.0)),
                              _session(tmp_path / "b"))
    row = next(x for x in out.splitlines() if x.startswith("agent_time_s"))
    assert "70.0" in row and "40.0" in row and "+75%" in row
    assert next(x for x in out.splitlines() if x.startswith("track.bounce")).split()[-2:] == ["14.0", "8.0"]


def test_the_nearest_rank_percentile_of_five_readings_is_a_reading():
    assert turn_timing.percentile([5.0, 1.0, 3.0, 2.0, 4.0], 0.5) == 3.0
    assert turn_timing.percentile([5.0, 1.0, 3.0, 2.0, 4.0], 0.95) == 5.0
    assert turn_timing.percentile([], 0.5) == 0.0


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
        pass

    async def daw_command(self, command, args):
        return {"ok": True}


def test_the_api_loop_records_the_model_that_served_each_turn(api):
    """An eval's score is of the model that ANSWERED. Nothing recorded which one did."""
    session = ChatSession(reasoning="medium", tool_defs_for=lambda during_run: TOOLS)
    api.script = [{"blocks": [("thinking", ""), ("text", "done")],
                   "usage": {"input_tokens": 50, "cache_read_input_tokens": 0,
                             "cache_creation_input_tokens": 0}, "out": 12}]

    async def emit(_event):
        pass
    anyio.run(lambda: session.turn("how loud is it?", tools=_Tools(), context=CONTEXT, emit=emit))
    assert session.timing.summary()["served_model"] == llm_models.DEFAULT_MODEL


def test_a_turn_over_the_socket_records_first_activity_the_apps_time_and_the_click(tmp_path, monkeypatch):
    """The whole path: a chat over the brain's socket, two commands answered by a stand-in
    app - one stamped with the app's own milliseconds, one behind an approval box - and the
    line the brain writes to `runs/turn_timing.jsonl` before it says the turn is done."""
    from tests import test_agent_audit_p2 as P

    proj = P._project(tmp_path)
    P._sub_env(monkeypatch)
    monkeypatch.setenv("MIX_AGENT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(B, "_CHAT_SESSIONS", {})

    async def script(tools, m):
        m.results.append(await tools["daw_command"].handler(
            {"command": "track.set_gain", "args": {"track": "bass", "gain_db": -7}}))
        yield m.AssistantMessage([])
        m.results.append(await tools["daw_command"].handler(
            {"command": "track.set_mute", "args": {"track": "bass", "on": True}}))
        yield m.AssistantMessage([])
        yield m.AssistantMessage([m.TextBlock("Bass down seven and muted.")])
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", P._fake_sdk(script))

    async def scenario(hub, port):
        reader, writer = await P._authed(port, hub.token, proj, "daw")
        try:
            writer.write(P._line({"id": 5, "method": "chat", "params": {
                "message": "bass down 7 and mute it", "mode": "agent", "agent": "claude"}}))
            await writer.drain()
            while True:
                msg = json.loads(await asyncio.wait_for(reader.readline(), 60))
                if "daw_request" in msg and msg["command"] == "track.set_gain":
                    writer.write(P._line({"daw_reply": msg["daw_request"], "app_ms": 12.5,
                                          "result": {"ok": True, "coedits": [P.SET_BASS]}}))
                elif "daw_request" in msg:           # a person reads a box for 0.6 s
                    writer.write(P._line({"daw_ack": msg["daw_request"]}))
                    await writer.drain()
                    await asyncio.sleep(0.6)
                    writer.write(P._line({"daw_reply": msg["daw_request"], "app_ms": 640,
                                          "result": {"ok": True}}))
                await writer.drain()
                ev = str(msg.get("event", ""))
                if ev == "chat_done" or "fail" in ev:
                    return msg
        finally:
            writer.close()
    done = P._serve(proj, scenario)
    assert done.get("event") == "chat_done" and done.get("ok"), done
    record = json.loads((proj / "runs" / "turn_timing.jsonl").read_text().splitlines()[-1])
    assert record["v"] == 3
    assert 0.0 <= record["first_activity_s"] <= record["wall_s"]
    assert [row[0] for row in record["tool_rows"]] == ["track.set_gain", "track.set_mute"]
    asked = record["commands"]
    assert asked["track.set_gain"]["n"] == 1 and asked["track.set_gain"]["app_s"] == 0.013
    assert asked["track.set_mute"]["app_s"] == 0.64
    assert 0.5 <= asked["track.set_mute"]["confirm_s"] <= 2.0
    assert "confirm_s" not in asked["track.set_gain"]
    assert record["confirm_wait_s"] == round(asked["track.set_mute"]["confirm_s"], 1)
    # ...and what the model was handed is the result alone
    assert "app_ms" not in json.dumps(done)


# ---------------------------------------------------------------- the watcher

def _watcher():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / ".claude" / "skills" / "drive-app" / "turn_watch.py"
    spec = importlib.util.spec_from_file_location("turn_watch", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _feed(project, *asked):
    chats = project / "chats"
    chats.mkdir(parents=True, exist_ok=True)
    (chats / "index.json").write_text(json.dumps({"active": "s1"}))
    (chats / "s1.json").write_text(json.dumps([{"k": "user", "t": a} for a in asked]))


def test_a_failed_turn_ends_for_the_watcher(tmp_path, capsys):
    """A turn that died on the API wrote no digest line, so the watcher (and the showcase
    driver behind it) said STILL RUNNING until its timeout, every time."""
    watch = _watcher()
    B.record_session_digest(tmp_path, session_id="s1", user="Import the songs.", reply="Done.",
                            commands=["track.add"])
    _feed(tmp_path, "Import the songs.", "Mix it. Keep each vocal clear.")
    turn_timing.record(tmp_path, {"turn": "t0002", "user": "Mix it. Keep each vocal clear.",
                                  "failed": "RateLimitError", "wall_s": 4.0})
    assert watch.main([str(tmp_path), "2", "0.5"]) == 3
    assert "TURN FAILED (RateLimitError)" in capsys.readouterr().out


def test_an_old_failure_does_not_end_the_wait_for_a_new_request(tmp_path, capsys):
    watch = _watcher()
    turn_timing.record(tmp_path, {"turn": "t0001", "user": "Mix it.", "failed": "RateLimitError"})
    _feed(tmp_path, "Mix it.", "Now master it for streaming.")       # a new request is running
    assert watch.main([str(tmp_path), "1", "0.2"]) == 1
    assert "STILL RUNNING" in capsys.readouterr().out
    # ...and a failure a later turn has finished since is history
    B.record_session_digest(tmp_path, session_id="s1", user="Mix it.", reply="Mixed.", commands=[])
    assert watch.failed_turn(tmp_path, "Mix it.") == ""
    assert watch.main([str(tmp_path), "1", "0.2"]) == 0
