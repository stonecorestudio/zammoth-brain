"""Phase 12 (docs/05-roadmap.md): investigating planner + sandbox probing.

Covers: probe → apply parity (bit-consistent on the same range), probes
leaving zero trace, probe target validation, the planner bridge whitelist
(mutations stay behind the proposal lifecycle), the ProbingRuleAgent beating
the Phase 11 baseline on the benchmark, per-action attribution in runs/, and
the mix-plan artifact.
"""
import json
import shutil

import pytest

from mix_engine import dsp
from orchestrator import benchmark
from orchestrator.agents import ProbingRuleAgent, RuleAgent, derive_plan, TargetState
from orchestrator.loop import run_session

from test_phase1_acceptance import call
from test_phase10_efficiency import in_memory_engine, reset_render_counts

GAIN_PLUS_2 = {"type": "gain", "stem_id": "stem_a", "delta_db": 2.0}
EQ_CUT = {"type": "eq", "stem_id": "stem_b",
          "band": {"freq_hz": 300, "gain_db": -4.0, "q": 1.0, "shape": "peak"}}


# ---------- probe correctness ----------

@pytest.mark.anyio
@pytest.mark.parametrize("range_s", [None, [2.0, 6.0]])
async def test_probe_apply_parity(demo_project, range_s):
    """Acceptance: a probe's `after` equals the post-apply measurement of
    the same actions on the same range — exactly, not approximately."""
    metrics = ["lufs_i", "stem(stem_a).lufs_s", "band_energy.mid"]
    async with in_memory_engine(demo_project) as client:
        probe = await call(client, "probe_actions", {"req": {
            "actions": [GAIN_PLUS_2, EQ_CUT], "metrics": metrics,
            "range_s": range_s}})
        probed = {m["metric"]: m["after"] for m in probe["measurements"]}

        proposal = await call(client, "propose_actions", {"req": {
            "rationale": "parity", "predicted_impact": [],
            "actions": [GAIN_PLUS_2, EQ_CUT]}})
        assert (await call(client, "apply_proposal",
                           {"proposal_id": proposal["proposal_id"]}))["ok"]
        # Measure the now-applied state through the same range-aware
        # resolver: a zero-delta probe's `before` values are exactly that.
        applied = await call(client, "probe_actions", {"req": {
            "actions": [{"type": "gain", "stem_id": "stem_b", "delta_db": 0.0}],
            "metrics": metrics, "range_s": range_s}})
        measured = {m["metric"]: m["before"] for m in applied["measurements"]}

    for metric in metrics:
        assert probed[metric] == measured[metric], metric


@pytest.mark.anyio
async def test_probe_leaves_no_trace(demo_project):
    async with in_memory_engine(demo_project) as client:
        baseline = await call(client, "render_and_analyze")
        bounce = demo_project / baseline["bounce_path"]
        bounce_bytes = bounce.read_bytes()
        state_before = (demo_project / "session.json").read_bytes()

        await call(client, "probe_actions", {"req": {
            "actions": [GAIN_PLUS_2, EQ_CUT],
            "metrics": ["lufs_i"]}})

        assert (demo_project / "session.json").read_bytes() == state_before
        assert await call(client, "get_proposals") == []
        state = await call(client, "get_state")
        assert all(s["gain_db"] == 0.0 for s in state["stems"])
        await call(client, "render_and_analyze")
        assert bounce.read_bytes() == bounce_bytes  # render unchanged


@pytest.mark.anyio
async def test_probe_rejects_unknown_targets(demo_project):
    async with in_memory_engine(demo_project) as client:
        result = await client.call_tool("probe_actions", {"req": {
            "actions": [{"type": "gain", "stem_id": "ghost", "delta_db": 2.0}],
            "metrics": ["lufs_i"]}})
        assert result.isError
        assert "ghost" in str(result.content)


# ---------- planner bridge whitelist ----------

class MutinousAgent(RuleAgent):
    """Tries to mutate through the investigation bridge; must be refused."""

    review = None  # no critic pass: propose runs exactly once

    def __init__(self):
        self.bridge_errors: list[str] = []

    def propose(self, ctx):
        for name in ("apply_proposal", "set_stem", "advance_pass"):
            try:
                ctx.engine_call(name, {})
            except ValueError as error:
                self.bridge_errors.append(str(error))
        return []


def test_bridge_refuses_mutations(demo_project):
    """The whitelist is enforced in the loop, not in prompts: every
    mutation tool raises before reaching the engine."""
    agent = MutinousAgent()
    run_session(demo_project, "master for streaming", gating="auto",
                agent=agent, max_passes=1, connect=in_memory_engine)
    assert len(agent.bridge_errors) == 3
    assert all("not a planner tool" in e for e in agent.bridge_errors)


# ---------- acceptance: the investigating planner beats the baseline ----------

def test_probing_agent_beats_baseline_on_benchmark(tmp_path):
    """Acceptance: closure no worse, never more passes, strictly fewer
    passes overall, and no regression flagged by the Phase 11 gate."""
    baseline = json.loads(benchmark.BASELINE_PATH.read_text())
    report = benchmark.run_benchmark(agent_factory=ProbingRuleAgent,
                                     work_dir=tmp_path)
    assert benchmark.compare(baseline, report) == []
    by_key = {(c["project"], c["brief"]): c for c in baseline["cases"]}
    for case in report["cases"]:
        base = by_key[(case["project"], case["brief"])]
        # met everywhere the rule baseline is met (Phase 20's "pop master"
        # is deliberately out of reach for both — docs/05 Phase 20)
        assert case["targets_met"] or not base["targets_met"], case["brief"]
        assert case["passes_run"] <= base["passes_run"], case["brief"]
        if case["mean_closure"] is not None and base["mean_closure"] is not None:
            assert (case["mean_closure"]
                    >= base["mean_closure"] - benchmark.CLOSURE_TOLERANCE)
    assert (sum(c["passes_run"] for c in report["cases"])
            < sum(c["passes_run"] for c in baseline["cases"]))
    # probe-measured predictions: error collapses vs the heuristic baseline
    assert (report["summary"]["mean_abs_prediction_error"]
            < baseline["summary"]["mean_abs_prediction_error"])


# ---------- attribution + mix plan ----------

class BundlingAgent(RuleAgent):
    """Emits one two-action bundle (Phase 15 made the reference proposal a
    single match-EQ action, so attribution needs a deliberate bundle)."""

    attribution = True
    review = None

    def propose(self, ctx):
        if ctx.pass_number > 1:
            return []
        return [{
            "rationale": "two-band master move",
            "predicted_impact": [
                {"metric": "band_energy.mid", "predicted_delta": -1.0,
                 "unit": "dB"},
                {"metric": "band_energy.presence", "predicted_delta": -1.0,
                 "unit": "dB"}],
            "actions": [
                {"type": "eq", "stem_id": "master",
                 "band": {"freq_hz": 600, "gain_db": -3.0, "q": 1.0,
                          "shape": "peak"}},
                {"type": "eq", "stem_id": "master",
                 "band": {"freq_hz": 2800, "gain_db": -3.0, "q": 1.0,
                          "shape": "peak"}}],
        }]


def test_attribution_records_per_action_deltas(demo_project):
    run_session(demo_project, "master for streaming", gating="auto",
                agent=BundlingAgent(), max_passes=1,
                connect=in_memory_engine)
    runs = sorted((demo_project / "runs").glob("*.jsonl"))
    records = [json.loads(line) for line in runs[0].read_text().splitlines()]
    attributed = [r for r in records if "attribution" in r]
    assert attributed, "no per-action attribution recorded"
    entry = attributed[0]
    assert len(entry["attribution"]) == len(entry["actions"]) == 2
    assert all("deltas" in a for a in entry["attribution"])
    # each cut owns its own band: the 600 Hz action moves mid, not presence
    first = entry["attribution"][0]["deltas"]
    assert first["band_energy.mid"] < first["band_energy.presence"]


def test_plan_artifact_lands_with_stages_checked(tmp_path):
    case = next(c for c in benchmark.BENCHMARK_CASES
                if "match" in c.brief)
    report = benchmark.run_case(case, ProbingRuleAgent, tmp_path)
    assert report["targets_met"]
    project = tmp_path / case.project
    plans = sorted((project / "runs").glob("*.plan.json"))
    assert plans
    plan = json.loads(plans[0].read_text())
    assert any(s["stage"] == "match tonal balance" for s in plan)
    assert all(s["done"] for s in plan)  # targets met => stages checked off


def test_derive_plan_orders_stages():
    states = [
        TargetState("lufs_i", "LUFS", -18.0, 4.0, goal=-14.0),
        TargetState("masking(a,b).worst_band_excess_db", "dB", 7.0, 4.0,
                    max_value=3.0),
        TargetState("band_energy.mid", "dB", -10.0, 2.0, goal=-8.0),
        TargetState("stereo_correlation", "", 0.4, 0.0, goal=0.4),
    ]
    plan = derive_plan(states)
    assert [s["stage"] for s in plan] == [
        "carve space", "match tonal balance", "hit loudness and ceiling",
        "remaining targets"]
    assert plan[0]["metrics"] == ["masking(a,b).worst_band_excess_db"]
    assert plan[3]["metrics"] == ["stereo_correlation"]


# ---------- probing agent improves the clarity case directly ----------

def test_probing_meets_clarity_faster_than_rule(tmp_path):
    """Same staged project, both agents: the prober finds the one-move fix
    the fixed-step heuristic needs three passes for."""
    case = next(c for c in benchmark.BENCHMARK_CASES if "clarity" in c.brief)
    rule = benchmark.run_case(case, RuleAgent, tmp_path / "rule")
    probing = benchmark.run_case(case, ProbingRuleAgent, tmp_path / "probing")
    assert rule["targets_met"] and probing["targets_met"]
    assert probing["passes_run"] < rule["passes_run"]
    assert (probing["proposals"]["applied"] < rule["proposals"]["applied"])
