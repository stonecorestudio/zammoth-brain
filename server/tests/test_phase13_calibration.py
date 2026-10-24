"""Phase 13 (docs/05-roadmap.md): calibration v2 + earned autonomy.

Covers: the v1 feedback loop fixed (bias learned from raw predictions),
shrinkage, the cross-project store with project-over-global precedence,
honest run records (raw_predicted_impact), `auto_confident` earned autonomy
end to end with seeded history, and reflection lessons + experience replay
reaching the next run's context.
"""
import json
import shutil

import pytest

from orchestrator import benchmark
from orchestrator.agents import ProbingRuleAgent, RuleAgent
from orchestrator.calibration import (CONFIDENT_MAE_MAX, bias_for,
                                      calibrate_proposal, calibration_table,
                                      confidence_for, exemplars,
                                      load_global_rows, load_lessons,
                                      load_records, update_global_store)
from orchestrator.loop import policy_gate, run_session

from test_phase10_efficiency import in_memory_engine


def gain_record(predicted, measured, raw=None, tags=None):
    record = {
        "actions": [{"type": "gain", "stem_id": "vox", "delta_db": predicted}],
        "predicted_impact": [{"metric": "lufs_i",
                              "predicted_delta": predicted, "unit": "LUFS"}],
        "measured_impact": [{"metric": "lufs_i",
                             "measured_delta": measured, "unit": "LUFS"}],
    }
    if raw is not None:
        record["raw_predicted_impact"] = [{"metric": "lufs_i",
                                           "predicted_delta": raw,
                                           "unit": "LUFS"}]
    if tags is not None:
        record["stem_tags"] = {"vox": tags}
    return record


# ---------- the v1 feedback loop is fixed ----------

def test_bias_is_learned_from_raw_predictions():
    """v1 recorded the calibrated prediction and re-learned from it, so the
    table chased its own corrections. v2: with raw_predicted_impact present,
    the learned bias is stable across calibration generations."""
    # generation 1: the agent predicts 1.0 raw, reality says 2.0
    gen1 = [gain_record(predicted=1.0, measured=2.0, raw=1.0)] * 3
    (row,) = calibration_table(gen1)
    assert row["bias"] == pytest.approx(1.0)

    # generation 2: predictions were calibrated (+0.6 shift recorded in
    # predicted_impact) but raw stays raw; reality unchanged
    gen2 = [gain_record(predicted=1.6, measured=2.0, raw=1.0)] * 3
    (row,) = calibration_table(gen1 + gen2)
    assert row["bias"] == pytest.approx(1.0)  # stable — no feedback
    # v1 semantics would have averaged in gen2's apparent 0.4 error:
    legacy = [gain_record(predicted=1.6, measured=2.0)] * 3  # no raw field
    (drifted,) = calibration_table(gen1 + legacy)
    assert drifted["bias"] < 1.0  # the failure mode v2 removes


def test_shrinkage_nudges_then_commits():
    one = calibration_table([gain_record(2.0, 3.0, raw=2.0)])
    actions = [{"type": "gain", "stem_id": "vox", "delta_db": 2.0}]
    assert bias_for(one, actions, "lufs_i") == pytest.approx(1.0 / 3)  # n=1
    ten = calibration_table([gain_record(2.0, 3.0, raw=2.0)] * 10)
    assert bias_for(ten, actions, "lufs_i") == pytest.approx(10.0 / 12)  # n=10


# ---------- cross-project store ----------

def test_global_store_roundtrip_and_precedence():
    records = [gain_record(1.0, 2.0, raw=1.0, tags=["vocal"])] * 4
    update_global_store(records)
    rows = load_global_rows()
    (row,) = [r for r in rows if r["tag"] == "vocal"]
    assert row["source"] == "global"
    assert row["count"] == 4 and row["bias"] == pytest.approx(1.0)

    actions = [{"type": "gain", "stem_id": "vox", "delta_db": 1.0}]
    # a fresh project with no local records falls back to global experience
    table = calibration_table([], load_global_rows())
    assert bias_for(table, actions, "lufs_i",
                    stem_tags={"vox": ["vocal"]}) == pytest.approx(4 / 6)
    # ...via the untagged bucket too when the tag doesn't match
    update_global_store([gain_record(1.0, 1.5, raw=1.0)] * 2)  # tag ""
    table = calibration_table([], load_global_rows())
    assert bias_for(table, actions, "lufs_i") == pytest.approx(0.5 * 2 / 4)
    # ...and project-local rows win over global ones
    local = [gain_record(1.0, 1.2, raw=1.0, tags=["vocal"])] * 8
    table = calibration_table(local, load_global_rows())
    assert bias_for(table, actions, "lufs_i",
                    stem_tags={"vox": ["vocal"]}) == pytest.approx(0.2 * 0.8)

    # merging is associative: a second fold accumulates counts
    update_global_store(records)
    (row,) = [r for r in load_global_rows() if r["tag"] == "vocal"]
    assert row["count"] == 8


# ---------- honest run records ----------

def test_runs_record_raw_predictions_and_tags(tmp_path):
    case = next(c for c in benchmark.BENCHMARK_CASES if "clarity" in c.brief)
    benchmark.run_case(case, RuleAgent, tmp_path)
    runs = sorted((tmp_path / case.project / "runs").glob("*.jsonl"))
    records = load_records(runs[0].parent)
    assert records
    for record in records:
        assert "raw_predicted_impact" in record
        assert "stem_tags" in record
        # no calibration history on a fresh project: raw == recorded
        assert record["raw_predicted_impact"] == record["predicted_impact"]


# ---------- earned autonomy ----------

def confident_table():
    return calibration_table([gain_record(2.0, 2.0, raw=2.0)] * 5)


def test_confidence_verdicts():
    proposal = {"actions": [{"type": "gain", "stem_id": "vox",
                             "delta_db": 2.0}],
                "predicted_impact": [{"metric": "lufs_i",
                                      "predicted_delta": 2.0, "unit": "LUFS"}]}
    good = confidence_for(proposal, confident_table())
    assert good["confident"] is True
    assert good["per_metric"][0]["count"] == 5
    assert good["per_metric"][0]["mean_abs_error"] == 0.0

    # no history -> not confident
    assert confidence_for(proposal, [])["confident"] is False
    # sloppy history -> not confident
    sloppy = calibration_table(
        [gain_record(2.0, 2.0 + 2 * CONFIDENT_MAE_MAX, raw=2.0)] * 5)
    assert confidence_for(proposal, sloppy)["confident"] is False
    # one observation -> not confident (CONFIDENT_MIN_SAMPLES)
    single = calibration_table([gain_record(2.0, 2.0, raw=2.0)])
    assert confidence_for(proposal, single)["confident"] is False
    # no predictions at all -> never confident
    assert confidence_for({"actions": [], "predicted_impact": []},
                          [])["confident"] is False


@pytest.mark.anyio
async def test_auto_confident_gate(monkeypatch):
    prompts = []
    monkeypatch.setattr("builtins.input",
                        lambda *_: prompts.append(1) or "n")
    gate = policy_gate("auto_confident")
    conservative = {"rationale": "small move",
                    "actions": [{"type": "gain", "stem_id": "vox",
                                 "delta_db": 2.0}],
                    "confidence": {"confident": True, "per_metric": []}}
    assert await gate("p1", conservative) is True
    assert prompts == []  # earned autonomy: no human in the loop

    bold = dict(conservative,
                actions=[{"type": "gain", "stem_id": "vox", "delta_db": 5.0}])
    assert await gate("p2", bold) is False  # confident but not conservative
    unproven = dict(conservative,
                    confidence={"confident": False, "per_metric": []})
    assert await gate("p3", unproven) is False  # conservative but unproven
    assert len(prompts) == 2  # both queued for the human


def test_auto_confident_end_to_end(tmp_path, monkeypatch):
    """Seeded history earns gain moves their autonomy: generation 1 (auto)
    builds the record, generation 2 under auto_confident applies the same
    kind of moves without a single prompt."""
    case = next(c for c in benchmark.BENCHMARK_CASES if "clarity" in c.brief)
    gen1 = benchmark.run_case(case, RuleAgent, tmp_path / "gen1")
    assert gen1["targets_met"]

    # fresh copy of the same problem, seeded with generation 1's records
    source = tmp_path / "gen1" / case.project
    project = tmp_path / "gen2" / case.project
    project.mkdir(parents=True)
    for wav in source.glob("*.wav"):
        shutil.copy(wav, project / wav.name)
    (project / "runs").mkdir()
    for log in (source / "runs").glob("*.jsonl"):
        shutil.copy(log, project / "runs" / f"prior_{log.name}")

    prompts = []
    monkeypatch.setattr("builtins.input",
                        lambda *_: prompts.append(1) or "n")
    report = run_session(project, case.brief, gating="auto_confident",
                         agent=RuleAgent(), connect=in_memory_engine)
    assert report.applied_proposal_ids        # the agent acted...
    assert prompts == []                      # ...without asking once
    assert report.stop_reason.startswith("targets met")


# ---------- reflection lessons + experience replay ----------

class RecordingAgent(ProbingRuleAgent):
    """Captures what the loop hands the agent (Phase 13 memory fields)."""

    def __init__(self):
        self.seen_lessons = None
        self.seen_exemplars = None

    def propose(self, ctx):
        self.seen_lessons = ctx.lessons
        self.seen_exemplars = ctx.exemplars
        return super().propose(ctx)


def test_lessons_and_exemplars_reach_the_next_run(tmp_path):
    case = next(c for c in benchmark.BENCHMARK_CASES if "clarity" in c.brief)
    gen1 = benchmark.run_case(case, RuleAgent, tmp_path / "gen1")
    assert gen1["targets_met"]
    source = tmp_path / "gen1" / case.project

    lessons = load_lessons(source / "runs")
    assert lessons and lessons[-1]["lessons"]  # reflection memo written
    assert case.brief in lessons[-1]["brief"]

    records = load_records(source / "runs")
    examples = exemplars(records, ["masking(stem_a,masker).worst_band_excess_db"])
    assert examples and "measured_impact" in examples[0]
    assert exemplars(records, ["band_energy.air"]) == []  # kind mismatch

    # a fresh copy of the same problem, seeded with gen 1's runs dir (the
    # mixed gen-1 project has nothing left to do, so propose never runs):
    # memory rides in the pass context
    project = tmp_path / "gen2" / case.project
    project.mkdir(parents=True)
    for wav in source.glob("*.wav"):
        shutil.copy(wav, project / wav.name)
    shutil.copytree(source / "runs", project / "runs")
    agent = RecordingAgent()
    run_session(project, case.brief, gating="auto", agent=agent,
                max_passes=1, connect=in_memory_engine)
    assert agent.seen_lessons, "lessons did not reach the context"
    assert agent.seen_exemplars, "exemplars did not reach the context"


def test_calibrate_proposal_uses_shrunk_bias():
    table = confident_table()  # 5 perfect observations, bias 0
    proposal = {"actions": [{"type": "gain", "stem_id": "vox",
                             "delta_db": 2.0}],
                "predicted_impact": [{"metric": "lufs_i",
                                      "predicted_delta": 2.0, "unit": "LUFS"}]}
    calibrated = calibrate_proposal(json.loads(json.dumps(proposal)), table)
    assert calibrated["predicted_impact"][0]["predicted_delta"] == 2.0
