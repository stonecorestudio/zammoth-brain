"""Phase 11 (docs/05-roadmap.md): eval harness + brief compiler.

The benchmark replays a fixed brief set and must reproduce the checked-in
baseline exactly (the regression gate later phases diff against); compare()
flags genuine regressions; free-form briefs compile into validated targets
and drive a terminating run; garbage compiler output is rejected by the
metric-grammar validation, and the no-LLM path still raises cleanly.
"""
import json

import pytest

from orchestrator import benchmark
from orchestrator.agents import RuleAgent
from orchestrator.loop import run_session
from orchestrator.targets import Target, compiled_targets

from test_phase10_efficiency import in_memory_engine


# ---------- benchmark harness ----------

def test_benchmark_matches_checked_in_baseline(tmp_path):
    """One fresh full run reproduces the baseline within compare()'s
    tolerances — this is both the determinism check and proof the gate
    works end to end."""
    report = benchmark.run_benchmark(agent_factory=RuleAgent,
                                     work_dir=tmp_path)
    assert report["summary"]["cases"] == len(benchmark.BENCHMARK_CASES)
    baseline = json.loads(benchmark.BASELINE_PATH.read_text())
    expected_met = {(c["project"], c["brief"]) for c in baseline["cases"]
                    if c["targets_met"]}
    # every case does real work; cases the baseline closes terminate by
    # measurement, not budget ("pop master" is deliberately out of the rule
    # agent's reach — docs/05 Phase 20)
    for case in report["cases"]:
        assert case["proposals"]["applied"] >= 1, case
        if (case["project"], case["brief"]) in expected_met:
            assert case["stop_reason"].startswith("targets met"), case
    assert benchmark.compare(baseline, report) == []


def test_compare_flags_regressions():
    baseline = {"cases": [{
        "project": "p", "brief": "b", "targets_met": True,
        "mean_closure": 0.9, "mean_abs_prediction_error": 0.2,
        "passes_run": 2,
    }]}
    regressed = {"cases": [{
        "project": "p", "brief": "b", "targets_met": False,
        "mean_closure": 0.5, "mean_abs_prediction_error": 0.9,
        "passes_run": 5,
    }]}
    problems = benchmark.compare(baseline, regressed)
    assert len(problems) == 4  # met flip, closure drop, error rise, passes
    assert benchmark.compare(baseline, baseline) == []
    # drift inside tolerance is not a regression
    drift = {"cases": [dict(baseline["cases"][0],
                            mean_closure=0.89,
                            mean_abs_prediction_error=0.24,
                            passes_run=3)]}
    assert benchmark.compare(baseline, drift) == []


def test_compare_flags_a_case_that_vanished():
    """Deleting an inconvenient case must not be a way to pass the gate.

    compare() only ever walked `current`, so a case dropped from
    BENCHMARK_CASES — or re-worded, which changes the (project, brief) key —
    simply stopped being compared, and --check exited 0 reporting no
    regressions (2026-08-16 review). Losing a case is a regression.
    """
    baseline = {"cases": [
        {"project": "p", "brief": "kept", "targets_met": True,
         "mean_closure": 0.9, "mean_abs_prediction_error": 0.2,
         "passes_run": 2},
        {"project": "p", "brief": "dropped", "targets_met": True,
         "mean_closure": 0.9, "mean_abs_prediction_error": 0.2,
         "passes_run": 2},
    ]}
    deleted = {"cases": [baseline["cases"][0]]}
    problems = benchmark.compare(baseline, deleted)
    assert len(problems) == 1 and "dropped" in problems[0]

    # re-wording the brief is the same evasion with a different spelling
    reworded = {"cases": [baseline["cases"][0],
                          dict(baseline["cases"][1], brief="reworded")]}
    assert any("dropped" in p for p in benchmark.compare(baseline, reworded))

    # a NEW case still isn't a regression, and a matched run stays green
    added = {"cases": baseline["cases"] + [
        dict(baseline["cases"][0], brief="brand new")]}
    assert benchmark.compare(baseline, added) == []
    assert benchmark.compare(baseline, baseline) == []


# ---------- compiled-target validation ----------

STEMS = ["stem_a", "stem_b"]


def test_compiled_targets_accepts_the_grammar():
    targets = compiled_targets([
        {"metric": "band_energy.low_mid", "goal": -10.0, "tolerance": 1.0},
        {"metric": "true_peak_dbtp", "max_value": -1.0},
        {"metric": "masking(stem_a,stem_b).worst_band_excess_db",
         "max_value": 3.0},
    ], STEMS)
    assert [t.metric for t in targets] == [
        "band_energy.low_mid", "true_peak_dbtp",
        "masking(stem_a,stem_b).worst_band_excess_db"]
    assert targets[0].goal == -10.0 and targets[0].tolerance == 1.0
    assert targets[1].max_value == -1.0


def test_compiled_targets_accepts_per_stem_goals():
    """2026-07-30: stem(<id>).<field> became measurable (measure_all resolves it
    via analyze_stem). Before this the compiler could only aim at session-level
    numbers, so a brief about ONE instrument had to be flattened into a
    master-wide target — understood correctly, then applied to the whole mix."""
    targets = compiled_targets([
        {"metric": "stem(stem_a).transient_energy_ratio", "goal": 0.35,
         "tolerance": 0.05, "unit": "ratio"},
        {"metric": "stem(stem_b).band_energy.presence", "goal": -9.0},
    ], STEMS)
    assert [t.metric for t in targets] == [
        "stem(stem_a).transient_energy_ratio", "stem(stem_b).band_energy.presence"]


@pytest.mark.parametrize("entry", [
    {"metric": "band_energy.nope", "goal": -10.0},        # unknown band
    {"metric": "stem(ghost).lufs_s", "goal": -20.0},      # unknown stem
    {"metric": "stem(stem_a).vibe", "goal": 1.0},         # unknown stem field
    {"metric": "stem(stem_a).band_energy.nope", "goal": -9.0},  # unknown band
    {"metric": "masking(stem_a,ghost).worst_band_excess_db",
     "max_value": 3.0},                                   # unknown stem
    {"metric": "masking(stem_a,stem_a).worst_band_excess_db",
     "max_value": 3.0},                                   # self-masking
    {"metric": "lufs_i"},                                 # no goal, no max
    {"metric": "lufs_i", "goal": -14.0, "max_value": -1.0},  # both
])
def test_compiled_targets_rejects_garbage(entry):
    with pytest.raises(ValueError):
        compiled_targets([entry], STEMS)


def test_compiled_targets_rejects_empty():
    with pytest.raises(ValueError):
        compiled_targets([], STEMS)


# ---------- free-form brief end to end ----------

class CompilerRuleAgent(RuleAgent):
    """RuleAgent + a deterministic stand-in for ClaudeAgent.compile_brief:
    same call shape (brief + analysis in, raw target dicts out), exercising
    the loop's fallback path and the grammar validation without an LLM."""

    def compile_brief(self, brief, analysis, notes=None, taste=None):
        bands = analysis["band_energy"]
        return [
            {"metric": "band_energy.low_mid", "unit": "dB",
             "goal": bands["low_mid"] + 2.0, "tolerance": 1.0},
            {"metric": "band_energy.high_mid", "unit": "dB",
             "goal": bands["high_mid"] - 2.0, "tolerance": 1.0},
        ]


def test_free_form_brief_compiles_and_terminates(demo_project):
    """Acceptance: a brief the regex table can't parse produces valid
    targets via the compiler and a terminating run that moves toward them."""
    brief = "warm up the low mids and tame the harshness around 3k"
    report = run_session(demo_project, brief, gating="auto",
                         agent=CompilerRuleAgent(),
                         connect=in_memory_engine)
    assert report.stop_reason in ("targets met", "targets met (critic pass)",
                                  "diminishing returns",
                                  "pass budget exhausted")
    assert report.passes_run >= 1
    assert report.applied_proposal_ids  # it acted on the compiled targets


def test_free_form_brief_without_compiler_raises(demo_project):
    """The no-LLM path is unchanged: RuleAgent has no compiler, so an
    unparseable brief still fails with the known-patterns error (wrapped in
    ExceptionGroups by the in-memory transport's task groups)."""
    def leaves(error):
        if isinstance(error, BaseExceptionGroup):
            return [leaf for sub in error.exceptions for leaf in leaves(sub)]
        return [error]

    with pytest.raises(BaseException) as exc_info:
        run_session(demo_project, "make it sparkle and vibe", gating="auto",
                    agent=RuleAgent(), connect=in_memory_engine)
    assert any(isinstance(e, ValueError) and "known patterns" in str(e)
               for e in leaves(exc_info.value)), exc_info.value


def test_compiled_targets_drops_malformed_entries_instead_of_dying():
    """Phase 51 find (2026-08-19): ONE malformed LLM-compiled target — a
    metric with neither goal nor max_value — raised and killed the whole
    run. A bad entry is model output, not a caller bug: it is dropped with
    a warning and the valid entries survive. Only ZERO usable targets
    still raises (there is genuinely nothing to mix toward)."""
    import pytest

    from orchestrator.targets import compiled_targets

    raw = [
        {"metric": "lufs_i", "goal": -14.0, "unit": "LUFS"},
        # the real-world failure: neither goal nor max_value
        {"metric": "stem(drums).transient_energy_ratio", "unit": "ratio"},
        # unknown metric — also dropped, not fatal
        {"metric": "stem(drums).vibe_quotient", "goal": 1.0},
    ]
    warnings: list[str] = []
    targets = compiled_targets(raw, ["drums"], warnings=warnings)
    assert [t.metric for t in targets] == ["lufs_i"]
    assert len(warnings) == 2
    assert "exactly one of goal / max_value" in warnings[0]
    assert "vibe_quotient" in warnings[1]

    with pytest.raises(ValueError, match="no usable targets"):
        compiled_targets([{"metric": "stem(drums).vibe_quotient"}], ["drums"])
