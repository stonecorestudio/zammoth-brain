"""The two P1s of the user-style QA of 2026-10-08
(docs/audits/user-qa-2026-10-08/QA-log.md: D21, D24).

D21 - a mix run stopped on its first pass and nothing said why. `run_passes` kept the
exception as one sentence in the report; the brain log had `stop=error` and no traceback,
so the cause of a failed run could not be read anywhere after the fact.

With the traceback in the log the cause was one run away: a run started FROM CHAT is handed
the live edit as a getter (`edit_state=lambda: self.latest_edit`, so a turn's later rounds see
what its own commands left), and `do_start_run` passed that getter straight to the run. The
planner's context builder read it as the snapshot: `AttributeError: 'function' object has no
attribute 'items'`, on the first pass of every chat-started mix run. The chat tests replace
`run_agent_pass` whole and the planner tests hand `_context_block` a dict, so the seam between
them had no test.
"""
import json
from pathlib import Path

import anyio
import numpy as np
import pytest
import soundfile as sf
from mcp.shared.memory import create_connected_server_and_client_session

from mix_engine.mcp_server import create_server
from orchestrator import daw_bridge
from orchestrator.agents import ClaudeAgent, PassContext
from orchestrator.loop import EngineClient, policy_gate, run_passes


@pytest.fixture
def quiet_project(tmp_path: Path) -> Path:
    sr = 48000
    t = np.arange(sr * 2) / sr
    for name, f in (("stem_a", 220.0), ("stem_b", 440.0)):
        sf.write(tmp_path / f"{name}.wav", 0.1 * np.sin(2 * np.pi * f * t), sr)
    return tmp_path


class AgentThatBreaksWhileThinking:
    """The failure of 2026-10-08 was inside the agent's turn, before any proposal existed."""

    def propose(self, ctx: PassContext) -> list[dict]:
        return self._read_the_session(ctx)

    def _read_the_session(self, ctx: PassContext) -> list[dict]:
        raise KeyError("no such field in the live edit: 'clips'")

    def revise(self, proposal, violations, ctx):
        return None

    def review(self, ctx):
        return None


def _run(project: Path, agent):
    events: list[dict] = []

    async def main():
        async with create_connected_server_and_client_session(
                create_server(project)) as session:
            async def emit(event: dict) -> None:
                events.append(event)

            return await run_passes(EngineClient(session), project, "stem A clarity",
                                    agent, max_passes=1, gate=policy_gate("auto"),
                                    emit=emit)

    return anyio.run(main), events


def test_a_failed_pass_leaves_its_reason_in_the_brain_log(quiet_project, capfd):
    """stderr IS the brain log. A pass that raised used to leave `stop=error` there and
    nothing else: the exception's type, its text and where it was raised were gone."""
    report, events = _run(quiet_project, AgentThatBreaksWhileThinking())

    assert report.stop_reason == "error"
    err = capfd.readouterr().err
    assert "pass 1 failed" in err, "the log line that says which pass, and that it failed"
    assert "KeyError" in err, "the exception's TYPE - str(exc) of a KeyError is only the key"
    assert "no such field in the live edit" in err
    assert "Traceback" in err and "_read_the_session" in err, \
        "where it was raised: the one thing nobody can reconstruct afterwards"


def test_the_reason_the_user_is_given_names_the_kind_of_failure(quiet_project):
    """`pass 1 failed after 0 applied move(s): 'clips'` is what a KeyError used to read as."""
    report, events = _run(quiet_project, AgentThatBreaksWhileThinking())

    assert report.warnings and "KeyError" in report.warnings[0]
    said = [e["message"] for e in events if e["type"] == "error"]
    assert said and "KeyError" in said[0]


# ---------- D21, the cause: the live edit reaches the planner as itself ----------

SNAPSHOT = {"tempo_bpm": 85.0,
            "tracks": [{"name": "stem_a", "kind": "audio",
                        "clips": [{"start_s": 0.0, "length_s": 2.0, "source": "stem_a.wav"}]}]}


class PlannerStandIn:
    """Proposes nothing, but builds its context with the PLANNER'S OWN builder - the line
    that raised - and keeps what the edit looked like in it."""
    usage = {"estimated_cost_usd": 0.0}
    budget_usd = None

    def __init__(self):
        self.edits_seen: list = []

    def propose(self, ctx: PassContext) -> list[dict]:
        self.edits_seen.append(json.loads(ClaudeAgent._context_block(ctx)).get("daw_edit_state"))
        return []

    def revise(self, proposal, violations, ctx):
        return None

    def review(self, ctx):
        return None


class ScriptedTurn:
    """The chat model's stand-in (as in test_daw_chat): calls the tools it was told to."""

    def __init__(self, calls):
        self._calls = calls
        self.results: list = []

    async def turn(self, message, *, tools, context, emit, during_run=False,
                   should_cancel=None, live_context=None):
        from orchestrator.chat import execute_tool
        for name, args in self._calls:
            self.results.append(await execute_tool(tools, name, args))
        await emit({"type": "chat_message", "text": "done"})


def test_a_run_started_from_chat_reads_the_live_edit(demo_project, monkeypatch):
    """The app's path, end to end below the model: chat_turn -> start_run -> run_agent_pass
    -> run_passes -> the planner's context. The edit arrives as the getter the brain passes."""
    planner = PlannerStandIn()
    monkeypatch.setattr(daw_bridge, "_llm_agent", lambda llm, what: planner)
    daw_bridge._CHAT_SESSIONS[str(demo_project)] = ScriptedTurn(
        [("start_run", {"brief": "master this for streaming", "gating": "", "max_passes": 1})])
    try:
        out = daw_bridge.chat_turn(demo_project, "master this for streaming", mode="agent",
                                   use_llm=True, edit_state=lambda: SNAPSHOT)
    finally:
        daw_bridge._CHAT_SESSIONS.clear()

    assert out.get("stop_reason") != "error", "the run died reading the live edit"
    assert planner.edits_seen, "the planner never built its context"
    assert planner.edits_seen[0]["tracks"][0]["name"] == "stem_a"


def test_every_pass_reads_the_edit_as_it_is_then(quiet_project):
    """A getter is handed over so that it is ASKED: the second pass sees what the DAW held
    after the first one's moves, not the snapshot the run started with."""
    planner = PlannerStandIn()
    pushes = iter([{"tempo_bpm": 85.0, "tracks": [{"name": "before"}]},
                   {"tempo_bpm": 85.0, "tracks": [{"name": "after"}]}])
    latest: dict = {}

    def live():
        nonlocal latest
        latest = next(pushes, latest)
        return latest

    class TwoPasses(PlannerStandIn):
        def propose(self, ctx):
            super().propose(ctx)
            # one harmless move keeps the loop going to a second pass
            return ([{"rationale": "nudge", "predicted_impact": [],
                      "actions": [{"type": "gain", "stem_id": "stem_a", "delta_db": 0.5}]}]
                    if ctx.pass_number == 1 else [])

    planner = TwoPasses()

    async def main():
        async with create_connected_server_and_client_session(
                create_server(quiet_project)) as session:
            async def emit(event: dict) -> None:
                pass
            # a loudness target two quiet sines are nowhere near: the run goes on to pass 2
            return await run_passes(EngineClient(session), quiet_project,
                                    "master for streaming", planner, max_passes=2,
                                    gate=policy_gate("auto"), emit=emit, edit_state=live)

    report = anyio.run(main)
    assert report.stop_reason != "error"
    names = [e["tracks"][0]["name"] for e in planner.edits_seen]
    assert names[:2] == ["before", "after"], names


def test_the_planner_is_never_handed_a_getter():
    """The pin under both: whatever a caller passes as `edit_state`, what the context builder
    is given is the snapshot. A callable here is the D21 crash."""
    ctx = PassContext(pass_number=1, metrics={}, targets=[], stem_chains={},
                      edit_state=lambda: SNAPSHOT)
    with pytest.raises(AttributeError):
        ClaudeAgent._context_block(ctx)   # the builder's contract: a dict - the LOOP resolves
