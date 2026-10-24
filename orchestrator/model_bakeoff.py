"""Model bake-off — which model should drive the DAW agent? (2026-08-15)

James asked for one agent model instead of four. This runs the three labelled
tracks under each candidate and prints them side by side, so the choice is made
on measurements rather than on which one sounds more impressive:

* **route**   (`understanding --agent claude`) — a mixing brief must land on the
  right scope. Mis-scoped is the damaging one: a request about one track becomes
  a confident session-wide change.
* **operate** (`operate_eval`) — the DAW half. The right command, on the right
  thing, without escalating a one-command request into a billed mix run.
* **answer**  (`qa_eval`) — the reply text. Ungrounded and bluffed are scored
  separately because they fail differently: one is worthless, one destroys trust.

It also records wall time and tokens per track, because "better" that costs 3x
and takes 40 s a turn is a different product.

    python -m orchestrator.model_bakeoff --tracks operate            # quick
    python -m orchestrator.model_bakeoff                             # all three
    python -m orchestrator.model_bakeoff --models claude-opus-5-5,claude-sonnet-5-5,claude-haiku-5-5

**Backend caveat, and it is not small.** With `MIX_AGENT_LLM_BACKEND=subscription`
the Claude Agent SDK owns the tool loop, so this measures each model *through
that loop*, not through the product's own API loop (`chat.py`'s other body). The
comparison stays internally fair — same harness, same prompts, same grading, only
the model changes, and the SDK does honour the model id — but absolute numbers
will not match an `api` run. Re-run with an ANTHROPIC_API_KEY before pinning a
number to anything outside this comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

DEFAULT_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5")   # Haiku 5.5 by --models: a third billed sweep
TRACKS = ("route", "operate", "answer")


def _run_route(model: str, jobs: int) -> dict:
    from .understanding import run_llm

    result = run_llm(model, jobs)
    c = result["counts"]
    errors = int(result.get("errors", 0) or 0)
    return {"total": result["total"], "counts": c, "errors": errors,
            "score": round(100.0 * c["resolved"] / _graded(result["total"], errors), 1),
            "headline": "resolved", "harm": ["mis-scoped", "over-reach"]}


def _run_operate(model: str, jobs: int) -> dict:
    from .operate_eval import run

    result = run(model, jobs, quiet=True)
    return {"total": result["total"], "counts": result["counts"],
            "errors": int(result.get("errors", 0) or 0),
            "score": result["score"], "seconds": result["seconds"],
            "headline": "correct", "harm": ["wrong", "escalated", "over-reach"]}


def _run_answer(model: str, jobs: int) -> dict:
    from .qa_eval import run

    result = run(model, jobs)
    c = result["counts"]
    errors = int(result.get("errors", 0) or 0)
    return {"total": result["total"], "counts": c, "errors": errors,
            "score": round(100.0 * c.get("answered", 0) / _graded(result["total"], errors), 1),
            "headline": "answered", "harm": ["bluffed", "ungrounded", "missed"]}


def _graded(total: int, errors: int) -> int:
    """The denominator of a score: the turns that reached the model. Every track leaves its
    errors out (operate always did) and REPORTS them - a score over 2 of 32 turns is a
    different number from the same score over 32, and the table used to print them alike
    (reader E7·R15)."""
    return max(1, int(total) - int(errors))


RUNNERS = {"route": _run_route, "operate": _run_operate, "answer": _run_answer}


def bakeoff(models: tuple[str, ...], tracks: tuple[str, ...],
            jobs: int, out: Optional[Path] = None) -> dict:
    report: dict = {"backend": os.environ.get("MIX_AGENT_LLM_BACKEND", "api"),
                    "models": {}}
    for model in models:
        report["models"][model] = {}
        for track in tracks:
            print(f"\n=== {model} · {track} ===", file=sys.stderr)
            started = time.monotonic()
            try:
                result = RUNNERS[track](model, jobs)
            except Exception as exc:  # noqa: BLE001 — one dead track must not
                print(f"  track failed: {type(exc).__name__}: {exc}",
                      file=sys.stderr)          # lose the tracks already run
                result = {"error": f"{type(exc).__name__}: {exc}"}
            result["wall_s"] = round(time.monotonic() - started, 1)
            report["models"][model][track] = result
            if out is not None:   # write as we go: a long sweep must survive a kill
                out.write_text(json.dumps(report, indent=2))
    return report


def _table(report: dict) -> str:
    models = list(report["models"])
    lines = [f"\nbackend: {report['backend']}", ""]
    head = f"{'track':10} {'metric':12}" + "".join(f"{m:>18}" for m in models)
    lines += [head, "-" * len(head)]
    for track in TRACKS:
        rows = {m: report["models"].get(m, {}).get(track) for m in models}
        if not any(rows.values()):
            continue
        # ERROR is decided per CELL. It was decided by the first model's track, so one dead
        # track replaced the row for every model - and hid the result the others did get.
        dead = {m: bool(rows[m]) and "error" in rows[m] for m in models}
        first = next((r for m, r in rows.items() if r and not dead[m]), None)

        def cell(m: str, value, suffix: str = "") -> str:
            if not rows[m]:
                return f"{'-':>18}"
            if dead[m]:
                return f"{'ERROR':>18}"
            return f"{str(value(rows[m])) + suffix:>18}"

        if first is None:
            lines.append(f"{track:10} {'ERROR':12}"
                         + "".join(f"{(rows[m] or {}).get('error', '-')[:18]:>18}"
                                   for m in models))
            lines.append("")
            continue
        lines.append(f"{track:10} {first['headline'] + ' %':12}"
                     + "".join(cell(m, lambda r: r.get("score", 0), "%") for m in models))
        for harm in first["harm"]:
            lines.append(f"{'':10} {harm:12}"
                         + "".join(cell(m, lambda r, h=harm: r.get("counts", {}).get(h, 0))
                                   for m in models))
        # the turns that never reached the model: out of the score above, and said
        lines.append(f"{'':10} {'errors':12}"
                     + "".join(cell(m, lambda r: f"{r.get('errors', 0)}/{r.get('total', 0)}")
                               for m in models))
        lines.append(f"{'':10} {'wall s':12}"
                     + "".join(cell(m, lambda r: r.get("wall_s", 0)) for m in models))
        for m in models:
            if dead[m]:
                lines.append(f"{'':10} {m}: {rows[m]['error'][:90]}")
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--tracks", default=",".join(TRACKS))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--out", type=Path,
                        default=Path("model_bakeoff.json"))
    args = parser.parse_args(argv)

    models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    tracks = tuple(t.strip() for t in args.tracks.split(",") if t.strip())
    bad = [t for t in tracks if t not in RUNNERS]
    if bad:
        print(f"unknown track(s): {bad}; choose from {list(RUNNERS)}",
              file=sys.stderr)
        return 2

    report = bakeoff(models, tracks, args.jobs, args.out)
    print(_table(report))
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
