"""The guardrail revision path (docs/04): a rejected proposal gets exactly one
structured revision attempt, then it is dropped.

Found by the 2026-08-05 audit: the revised proposal was read as if it carried
the fields the loop had attached to the ORIGINAL. `ClaudeAgent.revise` returns a
bare `ProposalRequest` dump (rationale / predicted_impact / actions), so
`proposal["confidence"]` raised KeyError and killed the run mid-pass — with that
pass's earlier proposals already applied and no records written. Every LLM run
was one guardrail rejection away from it.

It survived because every revise stub in the suite returned `None` or a deep copy
of the original (which does carry the loop's fields). These agents revise the way
the real one does: by handing back a fresh proposal.
"""
from pathlib import Path

import anyio
import pytest
import soundfile as sf
from mcp.shared.memory import create_connected_server_and_client_session

from mix_engine.mcp_server import create_server
from orchestrator.agents import PassContext
from orchestrator.loop import EngineClient, policy_gate, run_passes

# guardrails.LIMITS["gain_per_stem_per_pass_db"]
OVER_THE_LIMIT_DB = 9.0
WITHIN_THE_LIMIT_DB = 3.0


def _proposal(delta_db: float) -> dict:
    """Exactly the shape `ProposalRequest.model_dump()` produces — no
    `confidence`, no calibration fields. This is the point of the test."""
    return {"rationale": f"lift stem_a by {delta_db} dB",
            "predicted_impact": [{"metric": "stem.lufs_s",
                                  "predicted_delta": delta_db, "unit": "LUFS"}],
            "actions": [{"type": "gain", "stem_id": "stem_a",
                         "delta_db": delta_db}]}


class RevisingAgent:
    """Proposes past the per-pass gain clamp, then revises within it — the
    ordinary rejected-then-accepted path, returning a FRESH dict each time."""

    def __init__(self):
        self.revisions = 0

    def propose(self, ctx: PassContext) -> list[dict]:
        return [_proposal(OVER_THE_LIMIT_DB)] if ctx.pass_number == 1 else []

    def revise(self, proposal, violations, ctx):
        self.revisions += 1
        assert violations, "the engine must say what it rejected"
        return _proposal(WITHIN_THE_LIMIT_DB)

    def review(self, ctx):
        return None


class UnrepentantAgent(RevisingAgent):
    """Revises to something still over the limit — one attempt, then dropped."""

    def revise(self, proposal, violations, ctx):
        self.revisions += 1
        return _proposal(OVER_THE_LIMIT_DB)


@pytest.fixture()
def quiet_project(demo_project: Path) -> Path:
    data, sample_rate = sf.read(demo_project / "stem_a.wav", dtype="float32")
    sf.write(demo_project / "stem_a.wav", data * 10 ** (-30 / 20), sample_rate)
    return demo_project


def run_with_events(project_dir, agent):
    events: list[dict] = []

    async def main():
        async with create_connected_server_and_client_session(
                create_server(project_dir)) as session:
            async def emit(event: dict) -> None:
                events.append(event)

            return await run_passes(EngineClient(session), project_dir,
                                    "stem A clarity", agent, max_passes=1,
                                    gate=policy_gate("auto"), emit=emit)

    return anyio.run(main), events


def test_an_accepted_revision_does_not_kill_the_run(quiet_project):
    agent = RevisingAgent()
    report, events = run_with_events(quiet_project, agent)

    assert agent.revisions == 1
    assert [e for e in events if e["type"] == "proposal_rejected"]
    applied = [e for e in events if e["type"] == "proposal_status"
               and e["status"] == "applied"]
    assert applied, "the revised proposal should have been applied"
    assert report.stop_reason  # the run reached a real ending, not an exception


def test_the_revised_proposal_is_scored_not_the_original(quiet_project):
    """The emitted proposal must describe the move that actually landed, and
    carry the confidence the loop attaches — reading it used to raise."""
    _report, events = run_with_events(quiet_project, RevisingAgent())

    accepted = [e for e in events if e["type"] == "proposal"]
    assert len(accepted) == 1
    assert "confidence" in accepted[0]
    assert accepted[0]["actions"][0]["delta_db"] == WITHIN_THE_LIMIT_DB
    # and the prediction recorded for calibration is the revision's own, not
    # the rejected original's — otherwise the table learns a bias for a move
    # that never happened.
    predicted = accepted[0]["predicted_impact"][0]["predicted_delta"]
    assert abs(predicted - WITHIN_THE_LIMIT_DB) < abs(predicted - OVER_THE_LIMIT_DB)


def test_one_revision_attempt_then_the_proposal_is_dropped(quiet_project):
    agent = UnrepentantAgent()
    _report, events = run_with_events(quiet_project, agent)

    assert agent.revisions == 1, "docs/04: exactly one revision attempt"
    assert len([e for e in events if e["type"] == "proposal_rejected"]) == 2
    assert not [e for e in events if e["type"] == "proposal_status"
                and e["status"] == "applied"]


# ---------- a failing pass returns a report, not an exception (2026-08-05 audit) ----------

class ExplodingEngine(EngineClient):
    """Fails apply_proposal after the FIRST one lands — the realistic shape: a
    transient engine/tool error part-way through a pass."""

    def __init__(self, session):
        super().__init__(session)
        self.applied = 0

    async def call(self, name, args=None):
        if name == "apply_proposal":
            self.applied += 1
            if self.applied > 1:
                raise RuntimeError("engine tool 'apply_proposal' failed: disk full")
        return await super().call(name, args)


class TwoProposalAgent:
    """Two independent moves in one pass, so the second can fail after the first
    has already been applied."""

    def propose(self, ctx: PassContext) -> list[dict]:
        if ctx.pass_number != 1:
            return []
        return [_proposal(WITHIN_THE_LIMIT_DB), _proposal(WITHIN_THE_LIMIT_DB)]

    def revise(self, proposal, violations, ctx):
        return None

    def review(self, ctx):
        return None


def test_a_pass_that_fails_part_way_still_returns_a_report(quiet_project):
    """Nothing in run_one_pass was wrapped, so an engine error escaped run_passes
    entirely: the caller got an exception instead of a SessionReport, the UI never
    saw run_finished, and the records for what HAD been applied were never written
    (they are written after the render, which never happened)."""
    events: list[dict] = []

    async def main():
        async with create_connected_server_and_client_session(
                create_server(quiet_project)) as session:
            async def emit(event: dict) -> None:
                events.append(event)

            return await run_passes(ExplodingEngine(session), quiet_project,
                                    "stem A clarity", TwoProposalAgent(),
                                    max_passes=1, gate=policy_gate("auto"),
                                    emit=emit)

    report = anyio.run(main)   # must NOT raise

    assert report.stop_reason == "error"
    assert report.warnings and "disk full" in report.warnings[0]
    assert any(e["type"] == "error" for e in events), "the failure must reach the UI"


def test_what_landed_before_the_failure_is_still_reported(quiet_project):
    """The ids lived in a local, so a mid-pass raise lost them — the caller could
    neither undo what had been applied nor account for it."""
    async def main():
        async with create_connected_server_and_client_session(
                create_server(quiet_project)) as session:
            async def emit(_event: dict) -> None:
                return None

            return await run_passes(ExplodingEngine(session), quiet_project,
                                    "stem A clarity", TwoProposalAgent(),
                                    max_passes=1, gate=policy_gate("auto"),
                                    emit=emit)

    report = anyio.run(main)
    assert len(report.applied_proposal_ids) == 1, \
        "the move that DID land must be in the report so it can be undone"
