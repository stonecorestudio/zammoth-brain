"""Run speed & cost baseline (Phase 51 — docs/05-roadmap.md).

Measures what a mix run actually costs — wall-clock and dollars — BEFORE any
optimization lands, so every later change is judged against a committed
artifact instead of a feeling. Thirteen runs per backend: the ten Phase 48
technique briefs on `technique_demo` plus "master for streaming" on three
wild projects (the floor, the heavy-audio one, the heavy-state one).

Per run it records what the loop already emits, adding nothing to the hot
path: total wall-clock, passes, the per-pass `pass_timing` breakdown
(agent / apply / render seconds), and the final `llm_usage` (tokens in/out,
cache reads/writes, estimated cost at API rates). Summary = median + p95
wall-clock and cost across the thirteen — the exact numbers the phase's
−40 % / −30 % targets bind to.

Backends:
  --backend rule           free, deterministic — the no-LLM floor (how much
                           of a run is renders/analysis vs model latency)
  --backend subscription   the dev transport (billed to the Claude plan)
  --backend api            the production-shaped transport — needs
                           ANTHROPIC_API_KEY and bills it; refuses without
                           --allow-api (same guard as technique_eval)

    python -m orchestrator.speed_baseline --backend rule
    python -m orchestrator.speed_baseline --backend subscription \
        --out orchestrator/speed_baseline_2026-08-19_subscription.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "server"))

WILD_RUNS = ("tiny_two_30s", "hires_96k_f32", "many_stems_30")


def _cases() -> list[dict]:
    from .technique_eval import CASES as TECH_CASES
    cases = [{"key": c.key, "brief": c.brief, "project": "technique_demo"}
             for c in TECH_CASES]
    cases += [{"key": f"wild_{name}", "brief": "master for streaming",
               "project": name} for name in WILD_RUNS]
    return cases


def _stage(project: str, work: Path) -> Path:
    # Both fixtures are synthesised into the WORK folder (2026-10-07, readers E7·R12 /
    # E7·R14): this used to rewrite test_projects/technique_demo and rmtree + rewrite
    # test_projects/wild/<name> in the repo, on every staged run.
    if project == "technique_demo":
        from .technique_eval import fresh_fixture, stage
        return stage(fresh_fixture(work), work / "staged")
    from .wild_fixture import generate
    generate(work / "_fixture", only={project})
    return work / "_fixture" / project


def measure_run(case: dict, work: Path, backend: str, model: str,
                max_passes: int) -> dict:
    from .benchmark import in_memory_engine
    from .loop import run_session

    proj = _stage(case["project"], work)
    events: list[dict] = []
    t0 = time.monotonic()

    async def emit(event: dict) -> None:
        if event.get("type") in ("pass_timing", "llm_usage", "pass_started"):
            events.append({**event, "at_s": round(time.monotonic() - t0, 2)})

    if backend == "rule":
        from .agents import RuleAgent
        agent = RuleAgent()
    else:
        from .agents import ClaudeAgent
        agent = ClaudeAgent()

    error = None
    try:
        report = run_session(proj, case["brief"], gating="auto",
                             agent=agent, max_passes=max_passes,
                             connect=in_memory_engine, emit=emit)
    except BaseException as exc:  # noqa: BLE001 — a failed run is a data point
        def leaves(e):
            if hasattr(e, "exceptions"):
                for sub in e.exceptions:
                    yield from leaves(sub)
            else:
                yield e
        leaf = next(leaves(exc), exc)
        error = f"{type(leaf).__name__}: {str(leaf)[:200]}"
        report = None
    # A pass that RAISED does not raise here: `loop.run_passes` swallows it into
    # stop_reason "error" and run_session returns. That run was a fast, free "success" in
    # the medians the Phase 51 targets bind to (reader E7·R15) - a usage limit mid-sweep
    # LOWERED the reported wall and cost and the command exited 0. technique_eval learned
    # the same thing on 2026-08-18.
    if error is None and getattr(report, "stop_reason", None) == "error":
        why = "; ".join(str(w) for w in (getattr(report, "warnings", None) or [])) or "unknown"
        error = f"run stopped in error: {why[:200]}"

    wall_s = round(time.monotonic() - t0, 1)
    usage = getattr(agent, "usage", None) or {}
    pass_timings = [{k: v for k, v in e.items() if k != "type"}
                    for e in events if e.get("type") == "pass_timing"]
    return {
        "key": case["key"], "project": case["project"], "brief": case["brief"],
        "backend": backend, "wall_s": wall_s,
        "passes": getattr(report, "passes_run", None),
        "stop_reason": getattr(report, "stop_reason", None),
        "applied": len(getattr(report, "applied_proposal_ids", []) or []),
        "pass_timings_s": pass_timings,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "cache_read_tokens": usage.get("cache_read_tokens", 0),
        "cache_write_tokens": usage.get("cache_write_tokens", 0),
        "cost_usd": round(usage.get("estimated_cost_usd", 0.0), 4),
        "error": error,
    }


def _pctl(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))
    return xs[int(idx)]


def summarize(runs: list[dict]) -> dict:
    ok = [r for r in runs if not r["error"]]
    walls = [r["wall_s"] for r in ok]
    costs = [r["cost_usd"] for r in ok]
    return {
        "runs": len(runs), "errored": len(runs) - len(ok),
        "wall_s_median": round(statistics.median(walls), 1) if walls else None,
        "wall_s_p95": round(_pctl(walls, 0.95), 1) if walls else None,
        "wall_s_total": round(sum(walls), 1),
        "cost_usd_median": round(statistics.median(costs), 4) if costs else None,
        "cost_usd_p95": round(_pctl(costs, 0.95), 4) if costs else None,
        "cost_usd_total": round(sum(costs), 4),
        "tokens_in_total": sum(r["input_tokens"] for r in ok),
        "tokens_out_total": sum(r["output_tokens"] for r in ok),
    }


def main(argv=None) -> int:
    import tempfile

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", choices=("rule", "subscription", "api"),
                    default="rule")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--max-passes", type=int, default=3)
    ap.add_argument("--case", action="append",
                    help="run only this case key (repeatable)")
    ap.add_argument("--allow-api", action="store_true")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)

    cases = _cases()
    if args.case:
        keys = set(args.case)
        cases = [c for c in cases if c["key"] in keys]
    if args.list:
        for c in cases:
            print(f"  {c['key']:28s} [{c['project']}] {c['brief'][:60]!r}")
        print(f"\n{len(cases)} runs per backend")
        return 0

    if args.backend == "api":
        if not args.allow_api:
            print("the api backend bills ANTHROPIC_API_KEY — pass --allow-api "
                  "to confirm the spend", file=sys.stderr)
            return 2
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("no ANTHROPIC_API_KEY in this shell", file=sys.stderr)
            return 2
        os.environ["MIX_AGENT_LLM_BACKEND"] = "api"
    elif args.backend == "subscription":
        os.environ["MIX_AGENT_LLM_BACKEND"] = "subscription"
        os.environ.pop("ANTHROPIC_API_KEY", None)
    os.environ["MIX_AGENT_MODEL"] = args.model

    print(f"[baseline] {len(cases)} runs on backend={args.backend} "
          f"model={args.model if args.backend != 'rule' else '-'}")
    runs = []
    with tempfile.TemporaryDirectory(prefix="mix-agent-speed-") as td:
        for i, case in enumerate(cases, 1):
            work = Path(td) / f"run_{i:02d}"
            work.mkdir()
            r = measure_run(case, work, args.backend, args.model,
                            args.max_passes)
            runs.append(r)
            print(f"  [{i:2d}/{len(cases)}] {r['key']:28s} "
                  f"{r['wall_s']:6.1f}s  {r['passes'] or 0} passes  "
                  f"${r['cost_usd']:.3f}"
                  + (f"  ERROR {r['error']}" if r["error"] else ""))

    artifact = {"backend": args.backend,
                "model": args.model if args.backend != "rule" else None,
                "max_passes": args.max_passes,
                "summary": summarize(runs), "runs": runs}
    text = json.dumps(artifact, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
        print(f"\nartifact: {args.out}")
    s = artifact["summary"]
    print(f"\nwall-clock median {s['wall_s_median']}s  p95 {s['wall_s_p95']}s  "
          f"total {s['wall_s_total']}s")
    print(f"cost      median ${s['cost_usd_median']}  p95 ${s['cost_usd_p95']}  "
          f"total ${s['cost_usd_total']}")
    return 1 if s["errored"] else 0


if __name__ == "__main__":
    sys.exit(main())
