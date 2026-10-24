"""Phase 8 (docs/05-roadmap.md): self-correction — calibration + critic.

Acceptance: on a scripted multi-run benchmark, calibrated runs beat
uncalibrated on mean |predicted − measured|; the critic's verdict is present
and structured in runs/ for every completed loop, including a case where it
correctly flags an unmet target (and fixes it with its one corrective pass).
"""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from mcp.shared.memory import create_connected_server_and_client_session
from scipy.signal import butter, sosfilt

from mix_engine.mcp_server import create_server
from orchestrator.agents import RuleAgent
from orchestrator.calibration import (action_key, calibrate_proposal,
                                      calibration_table, load_records,
                                      metric_kind)
from orchestrator.loop import run_session


def in_memory_engine(project_dir):
    return create_connected_server_and_client_session(create_server(project_dir))


# ---------- calibration math (pure) ----------

def test_metric_kind_and_action_key():
    assert metric_kind("masking(a,b).worst_band_excess_db") == "masking.worst_band_excess_db"
    assert metric_kind("stem(vox).lufs_s") == "stem.lufs_s"
    assert metric_kind("band_energy.presence") == "band_energy.presence"
    assert metric_kind("lufs_i") == "lufs_i"
    assert action_key([{"type": "eq"}, {"type": "gain"}, {"type": "eq"}]) == "eq+gain"
    assert action_key([]) == "unknown"


def test_calibration_table_bias_and_mae():
    records = [
        {"actions": [{"type": "gain"}],
         "predicted_impact": [{"metric": "lufs_i", "predicted_delta": 2.0, "unit": "LUFS"}],
         "measured_impact": [{"metric": "lufs_i", "measured_delta": 3.0, "unit": "LUFS"}]},
        {"actions": [{"type": "gain"}],
         "predicted_impact": [{"metric": "lufs_i", "predicted_delta": -1.0, "unit": "LUFS"}],
         "measured_impact": [{"metric": "lufs_i", "measured_delta": 0.5, "unit": "LUFS"}]},
    ]
    (row,) = calibration_table(records)
    assert row["action_key"] == "gain" and row["metric_kind"] == "lufs_i"
    assert row["count"] == 2
    assert row["bias"] == pytest.approx(1.25)            # predictions run cold
    assert row["mean_abs_error"] == pytest.approx(1.25)

    proposal = {"actions": [{"type": "gain"}],
                "predicted_impact": [{"metric": "lufs_i", "predicted_delta": 2.0,
                                      "unit": "LUFS"}]}
    calibrated = calibrate_proposal(proposal, calibration_table(records))
    # v2 (Phase 13): the bias applies with shrinkage n/(n+2) — two
    # observations commit half the correction: 2.0 + 1.25 * 2/4 = 2.625,
    # rounded to 2 decimals in the stored prediction.
    assert calibrated["predicted_impact"][0]["predicted_delta"] == pytest.approx(2.62, abs=0.011)


# ---------- benchmark + critic over real loops ----------

def add_reference(project: Path) -> None:
    """Synthesized reference in refs/ (tilt + level offset; see Phase 7)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mix_engine import dsp
    from mix_engine.session import Session

    mix, sample_rate = dsp.render_master(Session(project))
    mono = mix.mean(axis=1)
    sos = butter(2, 2000.0, btype="highpass", fs=sample_rate, output="sos")
    tilted = (mono + 1.0 * sosfilt(sos, mono)) * 10 ** (3.0 / 20.0)
    tilted /= max(1.0, np.abs(tilted).max() / 0.9)
    (project / "refs").mkdir(exist_ok=True)
    sf.write(project / "refs" / "reference.wav", tilted.astype(np.float32),
             sample_rate)


def mean_abs_prediction_error(paths: list[Path]) -> float:
    errors = []
    for path in paths:
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if not record.get("predicted_impact") or not record.get("measured_impact"):
                continue
            measured = {m["metric"]: m["measured_delta"]
                        for m in record["measured_impact"]}
            for prediction in record["predicted_impact"]:
                if prediction["metric"] in measured:
                    errors.append(abs(measured[prediction["metric"]]
                                      - prediction["predicted_delta"]))
    assert errors, "no prediction/measurement pairs recorded"
    return sum(errors) / len(errors)


def test_phase8_acceptance_calibration_benchmark(demo_project, tmp_path):
    """Run 1 (no history) vs run 2 (seeded with run 1's records): the
    calibrated run must beat the uncalibrated one on mean |pred − measured|."""
    import shutil

    add_reference(demo_project)
    report_1 = run_session(demo_project, "match refs/reference.wav",
                           gating="auto", agent=RuleAgent(),
                           connect=in_memory_engine)
    assert report_1.stop_reason.startswith("targets met")
    error_uncalibrated = mean_abs_prediction_error(
        list((demo_project / "runs").glob("*.jsonl")))

    # fresh copy of the same project, seeded with run 1's records
    project_2 = tmp_path / "calibrated_copy"
    project_2.mkdir()
    for wav in demo_project.glob("*.wav"):
        shutil.copy(wav, project_2 / wav.name)
    add_reference(project_2)
    (project_2 / "runs").mkdir()
    for log in (demo_project / "runs").glob("*.jsonl"):
        shutil.copy(log, project_2 / "runs" / f"prior_{log.name}")

    report_2 = run_session(project_2, "match refs/reference.wav",
                           gating="auto", agent=RuleAgent(),
                           connect=in_memory_engine)
    assert report_2.stop_reason.startswith("targets met")
    error_calibrated = mean_abs_prediction_error(
        [p for p in (project_2 / "runs").glob("*.jsonl")
         if not p.name.startswith("prior_")])

    assert error_calibrated < error_uncalibrated, (
        f"calibration did not help: {error_calibrated} vs {error_uncalibrated}")


def critic_lines(runs_path: str) -> list[dict]:
    return [json.loads(line) for line in Path(runs_path).read_text().splitlines()
            if "critic" in json.loads(line)]


def test_critic_verdict_present_when_targets_met(demo_project):
    report = run_session(demo_project, "master for streaming", gating="auto",
                         agent=RuleAgent(), connect=in_memory_engine)
    assert report.stop_reason == "targets met"
    (verdict_line,) = critic_lines(report.runs_path)
    verdict = verdict_line["critic"]
    assert verdict["targets_met"] is True
    assert verdict["concerns"] == []
    assert verdict["suggested_followup"] is None
    assert verdict_line["corrective_pass"] is False


def test_critic_flags_unmet_target_and_corrects(demo_project):
    """Pass budget of 1 leaves the clarity target unmet; the critic flags it
    with a structured concern and its single corrective pass finishes the job."""
    data, sample_rate = sf.read(demo_project / "stem_a.wav", dtype="float32")
    sf.write(demo_project / "stem_a.wav", data * 10 ** (-30 / 20), sample_rate)

    report = run_session(demo_project, "stem A clarity", gating="auto",
                         agent=RuleAgent(), connect=in_memory_engine,
                         max_passes=1)
    lines = [json.loads(line) for line in Path(report.runs_path).read_text().splitlines()]
    (verdict_line,) = [l for l in lines if "critic" in l]
    verdict = verdict_line["critic"]
    assert verdict["targets_met"] is False
    assert any("masking(stem_a,stem_b)" in c for c in verdict["concerns"])
    assert verdict_line["corrective_pass"] is True

    (followup,) = [l["critic_followup"] for l in lines if "critic_followup" in l]
    assert followup["applied_proposal_ids"]      # the corrective pass acted
    assert report.passes_run == 2                # exactly one extra pass
    assert report.stop_reason in ("targets met (critic pass)",
                                  "pass budget exhausted")
