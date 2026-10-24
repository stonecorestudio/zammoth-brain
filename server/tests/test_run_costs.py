"""2026-06 cost-control work: a run must never spend LLM credits invisibly
or pointlessly. The loop stops on the agent's spend ceiling, skips renders
when a pass changed nothing, streams agent_working progress while paid
rounds run, and reports usage on the way out. ClaudeAgent's cost estimator
is pure and tested without the SDK."""
from pathlib import Path

import anyio
import pytest
import soundfile as sf
from mcp.shared.memory import create_connected_server_and_client_session

from mix_engine.mcp_server import create_server
from orchestrator.agents import PassContext, estimate_cost_usd
from orchestrator.loop import EngineClient, policy_gate, run_passes


def in_memory_engine(project_dir):
    return create_connected_server_and_client_session(create_server(project_dir))


@pytest.fixture()
def masked_project(demo_project: Path) -> Path:
    """Demo project with the 'vocal' pulled down 30 dB so the clarity
    target is genuinely unmet (same setup as test_phase2_loop)."""
    data, sample_rate = sf.read(demo_project / "stem_a.wav", dtype="float32")
    sf.write(demo_project / "stem_a.wav", data * 10 ** (-30 / 20), sample_rate)
    return demo_project


class CountingEngine(EngineClient):
    def __init__(self, session):
        super().__init__(session)
        self.renders = 0

    async def render_and_analyze(self) -> dict:
        self.renders += 1
        return await super().render_and_analyze()


def run_with_events(project_dir, agent, max_passes=5):
    events: list[dict] = []

    async def main():
        async with in_memory_engine(project_dir) as session:
            engine = CountingEngine(session)

            async def emit(event: dict) -> None:
                events.append(event)

            report = await run_passes(engine, project_dir, "stem A clarity",
                                      agent, max_passes=max_passes,
                                      gate=policy_gate("auto"), emit=emit)
            return report, engine.renders

    report, renders = anyio.run(main)
    return report, events, renders


class SpentAgent:
    """An agent whose meter says the run budget is already gone."""

    usage = {"turns": 3, "input_tokens": 200_000, "output_tokens": 40_000,
             "cache_read_tokens": 0, "cache_write_tokens": 0,
             "estimated_cost_usd": 9.99}
    budget_usd = 5.0

    def budget_exceeded(self) -> bool:
        return True

    def propose(self, ctx: PassContext) -> list[dict]:
        return []

    def revise(self, proposal, violations, ctx):
        return None


class ProgressAgent:
    """Proposes nothing, but reports investigation progress first — the
    bridge from the agent's worker thread to the event stream."""

    def propose(self, ctx: PassContext) -> list[dict]:
        assert ctx.progress is not None
        ctx.progress({"status": "thinking", "round": 1, "rounds_max": 6})
        ctx.progress({"status": "measuring", "round": 1, "rounds_max": 6,
                      "tools": ["analyze_masking"]})
        return []

    def revise(self, proposal, violations, ctx):
        return None


def test_spend_limit_stops_the_run(masked_project):
    report, events, renders = run_with_events(masked_project, SpentAgent())
    assert report.stop_reason == "spend limit reached"
    assert report.passes_run == 1
    # the meter reached the UI: llm_usage events carry spend + ceiling
    usage_events = [e for e in events if e["type"] == "llm_usage"]
    assert usage_events
    assert usage_events[-1]["estimated_cost_usd"] == 9.99
    assert usage_events[-1]["budget_usd"] == 5.0
    # run_finished carries the final meter for the feed's closing note
    finished = next(e for e in events if e["type"] == "run_finished")
    assert finished["usage"]["estimated_cost_usd"] == 9.99


def test_progress_events_reach_the_stream(masked_project):
    report, events, renders = run_with_events(masked_project, ProgressAgent())
    working = [e for e in events if e["type"] == "agent_working"]
    assert [e["status"] for e in working] == ["thinking", "measuring"]
    assert working[0]["pass_number"] == 1
    assert working[1]["tools"] == ["analyze_masking"]
    assert report.stop_reason == "nothing to propose"


def test_empty_pass_skips_the_render(masked_project):
    # Only the baseline render happens — an applied-nothing pass renders a
    # bit-identical bounce, so the loop must not pay for it.
    report, events, renders = run_with_events(masked_project, ProgressAgent())
    assert renders == 1
    assert len([e for e in events if e["type"] == "metrics"]) == 1


def test_estimate_cost_usd_price_table():
    # 1M in + 1M out on opus = $5 + $25
    assert estimate_cost_usd("claude-opus-4-8", 1_000_000, 1_000_000) == 30.0
    # cache reads bill at a tenth of input, writes at 1.25x
    assert estimate_cost_usd("claude-opus-4-8", 0, 0,
                             cache_read_tokens=1_000_000) == 0.5
    assert estimate_cost_usd("claude-opus-4-8", 0, 0,
                             cache_write_tokens=1_000_000) == 6.25
    assert estimate_cost_usd("claude-haiku-4-5", 1_000_000, 0) == 1.0
    # unknown models fall back to the most expensive row — never undercount (Fable: $10 in)
    assert estimate_cost_usd("future-model", 1_000_000, 0) == 10.0
    # Fable has its own row: it used to match nothing and fall back to Opus's $5/$25 — a run
    # budget that counted half of what the user's key was billed
    assert estimate_cost_usd("claude-fable-5-1", 1_000_000, 1_000_000) == 60.0
    # a dated snapshot prices as its alias; a different model never does
    assert estimate_cost_usd("claude-haiku-4-5-20251001", 1_000_000, 0) == 1.0
    assert estimate_cost_usd("claude-opus-5-6", 1_000_000, 0) == 10.0
