"""Benchmark harness (Phase 11 — docs/05-roadmap.md).

Replays a fixed brief set across the test projects and reports, per case:
target-closure %, passes-to-termination, mean |predicted − measured|, and
guardrail rejection rate. The checked-in baseline (`benchmark_baseline.json`)
is the regression gate — standing rule from Phase 11 on: agent-quality
changes (prompts, calibration, models, planner shape) must show a
non-regression here before merge. A case that DISAPPEARS from
`BENCHMARK_CASES` (deleted, or re-worded, which changes its key) counts as a
regression too: the baseline is the contract, so dropping a case from it goes
through `--update-baseline`, never through a quietly clean `--check`.

Run from the repo root (engine + RuleAgent are deterministic, so the rule
baseline is stable):

    python -m orchestrator.benchmark                    # report to stdout
    python -m orchestrator.benchmark --check            # diff vs baseline
    python -m orchestrator.benchmark --update-baseline  # rewrite baseline
    python -m orchestrator.benchmark --agent claude     # LLM agent (no gate)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import anyio
import numpy as np
import soundfile as sf

from .agents import Agent, ProbingRuleAgent, RuleAgent
from .calibration import load_records
from .engine_models import LIMITS  # noqa: F401  (bootstraps server/ onto sys.path)
from .loop import EngineClient, run_session
from .targets import (Target, reference_path_from_brief, reference_targets,
                      resolve_stem, targets_from_brief)

from .engine_client import in_memory_engine  # noqa: F401  (its home; the evals import it here)

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECTS_DIR = REPO_ROOT / "test_projects"
BASELINE_PATH = Path(__file__).resolve().parent / "benchmark_baseline.json"

# Regression tolerances: the pipeline is deterministic, but floats can drift
# across platforms/library versions — a regression must exceed these.
CLOSURE_TOLERANCE = 0.02
ERROR_TOLERANCE = 0.05


def _synthesize_reference(project: Path) -> None:
    """The project's own master with a known tilt (+~6 dB above 2 kHz) and
    +3 dB level as refs/reference.wav — outside the project root, where it
    would be discovered as a stem (docs/03). The tilt is what makes the
    per-band targets start unmet; without it only loudness would bite."""
    from scipy.signal import butter, sosfilt

    from mix_engine import dsp
    from mix_engine.session import Session

    mix, sample_rate = dsp.render_master(Session(project))
    mono = mix.mean(axis=1)
    sos = butter(2, 2000.0, btype="highpass", fs=sample_rate, output="sos")
    tilted = (mono + 1.0 * sosfilt(sos, mono)) * 10 ** (3.0 / 20.0)
    tilted /= max(1.0, np.abs(tilted).max() / 0.9)
    (project / "refs").mkdir(exist_ok=True)
    sf.write(project / "refs" / "reference.wav",
             tilted.astype(np.float32), sample_rate)


def _add_masker(project: Path) -> None:
    """Start the clarity case unmet: a deterministic mid-band tone cluster
    written as a third stem, sitting ~7 dB over stem_a in its worst band
    (the demo's own two stems barely overlap spectrally, so no fader move
    can create real masking between them)."""
    info = sf.info(str(project / "stem_a.wav"))
    t = np.arange(info.frames) / info.samplerate
    masker = 0.2 * sum(np.sin(2 * np.pi * f * t) for f in (500.0, 650.0, 800.0))
    sf.write(project / "masker.wav", masker.astype(np.float32),
             info.samplerate)


def _add_tone_stem(project: Path, name: str, freqs_hz: tuple[float, ...],
                   amplitude: float) -> None:
    """A sustained tone-cluster stem (Phase 20 composite cases): constant
    level, so the targeted region reads loud AND squashed — exactly what
    the harshness/mud composites measure."""
    info = sf.info(str(project / "stem_a.wav"))
    t = np.arange(info.frames) / info.samplerate
    tones = amplitude * sum(np.sin(2 * np.pi * f * t) for f in freqs_hz)
    sf.write(project / f"{name}.wav", tones.astype(np.float32),
             info.samplerate)


# NOTE the stem NAMES here. They must not be a word the brief uses: `resolve_stem`
# matches a stem id as a WHOLE WORD of the brief, so a stem literally called "harsh"
# resolves out of "less harsh" and the composite rule scopes to that stem instead of the
# session — silently changing what these cases measure. (Found 2026-08-05, when the
# composite path learned to scope to a named stem.) The match is whole-word, not
# substring, so "mud" would NOT have caught "less muddy" — but a plural would
# ("mud"/"muds"), and the next fixture may not be so lucky. Keep the names distinct.
def _add_mud(project: Path) -> None:
    """Start "less muddy" unmet: a constant 200–320 Hz cluster pumps the
    low_mid region (mud_db target = current − 2 dB)."""
    _add_tone_stem(project, "lowmids", (200.0, 250.0, 320.0), amplitude=0.15)


def _add_harsh(project: Path) -> None:
    """Start "less harsh" unmet: a constant 2.8–4.2 kHz cluster pumps the
    presence/brilliance region (harshness_db target = current − 2 dB)."""
    _add_tone_stem(project, "fizz", (2800.0, 3400.0, 4200.0), amplitude=0.12)


def _add_piano(project: Path) -> None:
    """Start "the piano is muddy" unmet, and CLOSABLE. band_energy is a share of
    that stem's own full-band power, so a stem made only of low-mid tone can't move
    its own low_mid share at all — cutting it scales the numerator and denominator
    together. The fixture is therefore broadband (a tone per band) with the low_mid
    pair pushed hot: now an EQ cut at 245 Hz genuinely shifts the share."""
    info = sf.info(str(project / "stem_a.wav"))
    t = np.arange(info.frames) / info.samplerate
    hot = 0.20 * sum(np.sin(2 * np.pi * f * t) for f in (200.0, 300.0))
    rest = 0.05 * sum(np.sin(2 * np.pi * f * t)
                      for f in (45.0, 90.0, 600.0, 1400.0, 3000.0, 6000.0, 11000.0))
    sf.write(project / "piano.wav", (hot + rest).astype(np.float32),
             info.samplerate)


@dataclass(frozen=True)
class Case:
    project: str
    brief: str
    setup: Optional[Callable[[Path], None]] = None


# The fixed brief set: one case per target family the loop knows (masking,
# destination loudness, reference matching, explicit LUFS). Add cases when a
# phase adds a target family; never edit existing ones in place — the
# baseline diff is the point.
BENCHMARK_CASES: tuple[Case, ...] = (
    Case("two_stem_demo", "stem_a clarity", setup=_add_masker),
    Case("two_stem_demo", "master for streaming"),
    Case("two_stem_demo", "match refs/reference.wav", setup=_synthesize_reference),
    Case("three_section_demo", "master to -16 LUFS"),
    # Phase 20: the composite ("less muddy"/"less harsh") and genre-curve
    # target families.
    Case("two_stem_demo", "less muddy", setup=_add_mud),
    Case("two_stem_demo", "less harsh", setup=_add_harsh),
    Case("two_stem_demo", "pop master"),
    # 2026-08-05: a brief that NAMES an instrument must compile to targets on that
    # stem, not on the whole session. Before the track-scope fix this case measured
    # session mud_db and "fixed" it with a master EQ cut — the piano untouched.
    Case("two_stem_demo", "the piano is muddy", setup=_add_piano),
)




def _stage_project(name: str, work_dir: Path) -> Path:
    """A throwaway copy of a test project (stems generated on first use)."""
    source = PROJECTS_DIR / name
    if not list(source.glob("*.wav")):
        subprocess.run([sys.executable, str(source / "generate_stems.py")],
                       check=True)
    project = work_dir / name
    project.mkdir(parents=True)
    for wav in source.glob("*.wav"):
        shutil.copy(wav, project / wav.name)
    return project


async def _targets_and_values(project: Path, brief: str,
                              targets: Optional[list[Target]] = None,
                              ) -> tuple[list[Target], dict]:
    """Compute the brief's targets (first call) and measure their current
    values. Reference targets are computed once and passed back in for the
    final measurement so goals stay fixed."""
    async with in_memory_engine(project) as client:
        engine = EngineClient(client)
        if targets is None:
            if ref_path := reference_path_from_brief(brief):
                reference = await engine.call("analyze_reference",
                                              {"path": ref_path})
                targets = reference_targets(reference)
            else:
                analysis = await engine.call("analyze_session")
                # Same resolve-then-fetch as loop.py: a brief that names a stem
                # compiles to targets on that stem's own bands, which need its
                # analysis as their anchor.
                stem_analysis = None
                if named := resolve_stem(brief, analysis["stems"]):
                    stem_analysis = await engine.call("analyze_stem",
                                                      {"stem_id": named})
                targets = targets_from_brief(brief, analysis["stems"], analysis,
                                             None, stem_analysis)
        values = await engine.measure_all([t.metric for t in targets])
    return targets, values


def _prediction_errors(runs_dir: Path) -> list[float]:
    errors = []
    for record in load_records(runs_dir):
        measured = {m["metric"]: m["measured_delta"]
                    for m in record["measured_impact"]}
        for prediction in record["predicted_impact"]:
            if prediction["metric"] in measured:
                errors.append(abs(measured[prediction["metric"]]
                                  - prediction["predicted_delta"]))
    return errors


def run_case(case: Case, agent_factory: Callable[[], Agent],
             work_dir: Path) -> dict:
    project = _stage_project(case.project, work_dir)
    if case.setup is not None:
        case.setup(project)

    # Hermetic per-case experience (Phase 13): the cross-project store must
    # not leak between cases, or the baseline becomes order-dependent.
    saved_data_dir = os.environ.get("MIX_AGENT_DATA_DIR")
    os.environ["MIX_AGENT_DATA_DIR"] = str(work_dir / "appdata")
    try:
        targets, initial = anyio.run(_targets_and_values, project, case.brief)
        report = run_session(project, case.brief, gating="auto",
                             agent=agent_factory(), connect=in_memory_engine)
        _, final = anyio.run(_targets_and_values, project, case.brief, targets)
    finally:
        if saved_data_dir is None:
            os.environ.pop("MIX_AGENT_DATA_DIR", None)
        else:
            os.environ["MIX_AGENT_DATA_DIR"] = saved_data_dir

    per_target, closures = [], []
    for target in targets:
        d0 = target.distance(initial[target.metric])
        d1 = target.distance(final[target.metric])
        entry = {"metric": target.metric,
                 "initial_distance": round(d0, 3),
                 "final_distance": round(d1, 3)}
        if math.isfinite(d0) and d0 > 0:
            entry["closure"] = round((d0 - d1) / d0, 3)
            closures.append((d0 - d1) / d0)
        per_target.append(entry)

    errors = _prediction_errors(project / "runs")
    proposals = json.loads((project / "session.json").read_text())["proposals"]
    statuses = [p["status"] for p in proposals]

    return {
        "project": case.project,
        "brief": case.brief,
        "stop_reason": report.stop_reason,
        "passes_run": report.passes_run,
        "targets_met": all(t["final_distance"] <= 0 for t in per_target),
        "mean_closure": (round(sum(closures) / len(closures), 3)
                         if closures else None),
        "mean_abs_prediction_error": (round(sum(errors) / len(errors), 3)
                                      if errors else None),
        "proposals": {
            "total": len(statuses),
            "applied": statuses.count("applied"),
            "rejected": statuses.count("rejected"),
        },
        "per_target": per_target,
    }


def run_benchmark(agent_factory: Callable[[], Agent] = RuleAgent,
                  cases: tuple[Case, ...] = BENCHMARK_CASES,
                  work_dir: Optional[Path] = None) -> dict:
    work = Path(work_dir) if work_dir is not None else Path(
        tempfile.mkdtemp(prefix="mix-agent-benchmark-"))
    case_reports = [run_case(case, agent_factory, work / f"case_{i}")
                    for i, case in enumerate(cases)]

    closures = [c["mean_closure"] for c in case_reports
                if c["mean_closure"] is not None]
    errors = [c["mean_abs_prediction_error"] for c in case_reports
              if c["mean_abs_prediction_error"] is not None]
    totals = sum(c["proposals"]["total"] for c in case_reports)
    rejected = sum(c["proposals"]["rejected"] for c in case_reports)
    return {
        "agent": getattr(agent_factory, "__name__", repr(agent_factory)),
        "cases": case_reports,
        "summary": {
            "cases": len(case_reports),
            "targets_met_rate": round(
                sum(c["targets_met"] for c in case_reports)
                / len(case_reports), 3),
            "mean_closure": (round(sum(closures) / len(closures), 3)
                             if closures else None),
            "mean_abs_prediction_error": (round(sum(errors) / len(errors), 3)
                                          if errors else None),
            "rejection_rate": round(rejected / totals, 3) if totals else None,
        },
    }


def compare(baseline: dict, current: dict) -> list[str]:
    """Regressions of `current` against `baseline`; empty list = pass.
    A case regresses when targets_met flips false, mean closure drops more
    than CLOSURE_TOLERANCE, prediction error rises more than
    ERROR_TOLERANCE, or it needs 2+ extra passes — or when it DISAPPEARS."""
    problems: list[str] = []
    baseline_cases = {(c["project"], c["brief"]): c for c in baseline["cases"]}
    # A case that is gone cannot regress, and that was the hole: this loop only
    # ever walked `current`, so deleting a failing case from BENCHMARK_CASES (or
    # re-wording its brief, which changes the key) made --check print "no
    # regressions" and exit 0 — the gate erasing its own evidence. The baseline
    # is the contract; dropping a case from it is a deliberate act that goes
    # through --update-baseline, never through a silent pass (2026-08-16 review).
    current_keys = {(c["project"], c["brief"]) for c in current["cases"]}
    for project, brief in sorted(baseline_cases.keys() - current_keys):
        problems.append(
            f"{project} / {brief!r}: in the baseline but MISSING from this run "
            f"(deleted or re-worded) — run --update-baseline if deliberate")
    for case in current["cases"]:
        base = baseline_cases.get((case["project"], case["brief"]))
        if base is None:
            continue  # new case: no baseline yet, nothing to regress
        label = f"{case['project']} / {case['brief']!r}"
        if base["targets_met"] and not case["targets_met"]:
            problems.append(f"{label}: targets_met regressed (was met)")
        if (base["mean_closure"] is not None
                and case["mean_closure"] is not None
                and case["mean_closure"]
                < base["mean_closure"] - CLOSURE_TOLERANCE):
            problems.append(
                f"{label}: closure {case['mean_closure']} < baseline "
                f"{base['mean_closure']} - {CLOSURE_TOLERANCE}")
        if (base["mean_abs_prediction_error"] is not None
                and case["mean_abs_prediction_error"] is not None
                and case["mean_abs_prediction_error"]
                > base["mean_abs_prediction_error"] + ERROR_TOLERANCE):
            problems.append(
                f"{label}: prediction error "
                f"{case['mean_abs_prediction_error']} > baseline "
                f"{base['mean_abs_prediction_error']} + {ERROR_TOLERANCE}")
        if case["passes_run"] > base["passes_run"] + 1:
            problems.append(f"{label}: passes_run {case['passes_run']} vs "
                            f"baseline {base['passes_run']}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mix-agent benchmark (Phase 11)")
    parser.add_argument("--agent", default="rule",
                        choices=["rule", "probing", "claude"])
    parser.add_argument("--out", type=Path, default=None,
                        help="write the JSON report here")
    parser.add_argument("--check", action="store_true",
                        help="compare against the checked-in baseline; "
                             "exit 1 on regression")
    parser.add_argument("--update-baseline", action="store_true",
                        help="rewrite the checked-in baseline (rule agent only)")
    args = parser.parse_args()

    if args.agent == "claude":
        from .agents import ClaudeAgent
        factory: Callable[[], Agent] = ClaudeAgent
    elif args.agent == "probing":
        factory = ProbingRuleAgent  # Phase 12: deterministic investigator
    else:
        factory = RuleAgent

    report = run_benchmark(agent_factory=factory)
    text = json.dumps(report, indent=2)
    if args.out is not None:
        args.out.write_text(text + "\n")
    print(text)

    if args.update_baseline:
        if args.agent != "rule":
            sys.exit("the baseline is the deterministic rule-agent run only")
        BASELINE_PATH.write_text(text + "\n")
        print(f"\nbaseline updated: {BASELINE_PATH}")
        return
    if args.check:
        baseline = json.loads(BASELINE_PATH.read_text())
        problems = compare(baseline, report)
        if problems:
            print("\nREGRESSIONS:\n" + "\n".join(f"- {p}" for p in problems))
            sys.exit(1)
        print("\nno regressions against the baseline")
        # Target CLOSURE is only half the gate: closing the wrong target is
        # still a failure the user feels. The understanding eval covers the
        # step before — was this the right target, at the right scope
        # (2026-07-30, orchestrator/understanding.py). Deterministic, so it
        # belongs on the same gate rather than behind an LLM run.
        from .understanding import main as understanding_main
        print()
        if understanding_main(["--check"]) != 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
