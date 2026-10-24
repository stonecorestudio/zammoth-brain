"""Phase 2 acceptance (docs/05-roadmap.md): the pass-control loop on the
two-stem demo with target "stem A clarity" terminates within 5 passes via
targets-met, nothing-to-propose, or diminishing-returns, never by crash;
every applied proposal has measured impact recorded."""
import json
from pathlib import Path

import pytest
import soundfile as sf
from mcp.shared.memory import create_connected_server_and_client_session

from mix_engine.mcp_server import create_server
from orchestrator.agents import PassContext, RuleAgent
from orchestrator.loop import run_session
from orchestrator.targets import Target, targets_from_brief


def in_memory_engine(project_dir):
    return create_connected_server_and_client_session(create_server(project_dir))


@pytest.fixture()
def masked_project(demo_project: Path) -> Path:
    """Demo project with the 'vocal' pulled down 30 dB so the noise bed
    genuinely masks it (initial worst-band excess ~ +6 dB, target <= 3 dB)."""
    data, sample_rate = sf.read(demo_project / "stem_a.wav", dtype="float32")
    sf.write(demo_project / "stem_a.wav", data * 10 ** (-30 / 20), sample_rate)
    return demo_project


class NullAgent:
    def propose(self, ctx: PassContext) -> list[dict]:
        return []

    def revise(self, proposal, violations, ctx):
        return None


def test_loop_reaches_clarity_target(masked_project):
    report = run_session(masked_project, "stem A clarity",
                         gating="auto_conservative", agent=RuleAgent(),
                         connect=in_memory_engine)
    assert report.stop_reason == "targets met"
    assert report.passes_run <= 5
    assert report.applied_proposal_ids  # it actually did something

    # prediction accountability: one runs/ line per applied proposal,
    # each with the engine-measured impact next to the prediction
    # (Phase 8 also appends a critic-verdict line — proposal records only here)
    lines = [json.loads(line) for line in
             Path(report.runs_path).read_text().splitlines()]
    proposals = [l for l in lines if "proposal_id" in l]
    assert {l["proposal_id"] for l in proposals} == set(report.applied_proposal_ids)
    for line in proposals:
        assert line["measured_impact"], line
        assert line["predicted_impact"]


def test_loop_stops_when_agent_proposes_nothing(masked_project):
    # An empty pass means "targets met or nothing sensible remains" — the
    # loop must stop right there, not grind out two more stalled passes
    # (2026-06 cost-control: each LLM pass costs real money).
    report = run_session(masked_project, "stem A clarity",
                         gating="auto", agent=NullAgent(),
                         connect=in_memory_engine)
    assert report.stop_reason == "nothing to propose"
    assert report.passes_run == 1
    assert report.applied_proposal_ids == []


def test_cancel_mid_pass_reads_as_cancelled(masked_project):
    """Stop arriving DURING a pass (2026-08-09): the planner sees it via
    ctx.should_cancel and returns early, and the run must report 'cancelled' —
    not fall through to the empty-pass 'nothing to propose' stop."""
    flag = {"set": False}

    class CancelMidPassAgent(NullAgent):
        def propose(self, ctx):
            assert ctx.should_cancel is not None, \
                "the loop must hand Stop to the planner (PassContext.should_cancel)"
            flag["set"] = True   # Stop pressed while the planner was thinking
            return []

    report = run_session(masked_project, "stem A clarity",
                         gating="auto", agent=CancelMidPassAgent(),
                         connect=in_memory_engine,
                         should_cancel=lambda: flag["set"])
    assert report.stop_reason == "cancelled"
    assert report.applied_proposal_ids == []


def test_cancel_before_apply_skips_the_pass_proposals(masked_project):
    """Stop landing between the planner and the apply loop: nothing further is
    proposed/probed/applied, and the run still reads as cancelled."""
    flag = {"set": False}

    class CancelAfterProposeAgent(NullAgent):
        def propose(self, ctx):
            flag["set"] = True
            return [{"rationale": "too late — Stop already pressed",
                     "predicted_impact": [],
                     "actions": [{"type": "gain", "stem_id": "stem_a",
                                  "delta_db": 1.0}]}]

    report = run_session(masked_project, "stem A clarity",
                         gating="auto", agent=CancelAfterProposeAgent(),
                         connect=in_memory_engine,
                         should_cancel=lambda: flag["set"])
    assert report.stop_reason == "cancelled"
    assert report.applied_proposal_ids == []


def test_manual_gating_blocks_apply(masked_project, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *args: "n")
    report = run_session(masked_project, "stem A clarity",
                         gating="manual", agent=RuleAgent(),
                         connect=in_memory_engine)
    assert report.applied_proposal_ids == []
    assert report.stop_reason in ("diminishing returns", "pass budget exhausted")


def test_targets_from_brief():
    targets = targets_from_brief("master for streaming", ["stem_a", "stem_b"])
    # every loudness ask carries the true-peak ceiling and (2026-06 review)
    # the PLR punch floor
    assert {t.metric for t in targets} == {"lufs_i", "true_peak_dbtp",
                                           "plr_db"}

    clarity = targets_from_brief("stem A clarity", ["stem_a", "stem_b"])
    assert [t.metric for t in clarity] == ["masking(stem_a,stem_b).worst_band_excess_db"]

    with pytest.raises(ValueError):
        targets_from_brief("make it slap", ["stem_a"])


def test_target_distance():
    ceiling = Target(metric="m", unit="dB", max_value=3.0)
    assert ceiling.distance(6.1) == pytest.approx(3.1)
    assert ceiling.distance(2.0) == 0.0
    goal = Target(metric="lufs_i", unit="LUFS", goal=-11.0, tolerance=0.5)
    assert goal.distance(-11.4) == 0.0
    assert goal.distance(-14.0) == pytest.approx(2.5)
    assert goal.distance(None) == float("inf")
    floor = Target(metric="plr_db", unit="dB", min_value=4.0)  # 2026-06
    assert floor.distance(13.0) == 0.0
    assert floor.distance(2.5) == pytest.approx(1.5)
