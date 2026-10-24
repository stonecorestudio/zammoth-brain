"""What the brain's own work costs as a project grows (Phase 56).

The chat rig's ladder says the brain and the app add about half a second to a turn that does
not render - on four stems of thirty seconds. This is the other half of that sentence: the
same work on twelve and forty stems, on three and six minutes, with no model and no app.

    python -m orchestrator.size_sweep            # five sizes: a cross through 12 stems x 3 min
    python -m orchestrator.size_sweep --full     # all nine
    python -m orchestrator.size_sweep --over 5   # exit 1 when any figure passes five seconds

Free. Each project is generated in a temp folder (seeded noise and tones, mono 16-bit: what
is timed is the engine's work per stem and per second, not the music) and removed.

What is timed, as the DAW's brain runs it (the render cache and the master memo ON, as
`daw_server.main` switches them): the first readout of a project (its first whole render),
the same readout again, the measure after ONE fader moved (what follows a level command),
and the read tools a chat turn reaches for.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from .engine_client import in_memory_engine

CROSS = [(4, 180), (12, 30), (12, 180), (12, 360), (40, 180)]
FULL = [(n, d) for n in (4, 12, 40) for d in (30, 180, 360)]
RATE = 44_100


def make_project(folder: Path, stems: int, seconds: int, seed: int = 7) -> Path:
    """`stems` mono 16-bit files of `seconds`: a tone and shaped noise each, never clipping."""
    import numpy as np
    import soundfile as sf

    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    t = np.arange(seconds * RATE) / RATE
    for i in range(stems):
        tone = 0.12 * np.sin(2 * np.pi * (55.0 * (1 + i % 12)) * t)
        noise = 0.05 * rng.standard_normal(t.size) * (0.5 + 0.5 * np.sin(2 * np.pi * (0.25 + 0.05 * i) * t))
        sf.write(folder / f"stem{i:02d}.wav", (tone + noise).astype("float32"), RATE, subtype="PCM_16")
    return folder


async def _tools(project: Path, names: list[str], seconds: int) -> dict[str, float]:
    out: dict[str, float] = {}
    async with in_memory_engine(project) as engine:
        calls = [("analyze_session", {}), ("analyze_stem", {"stem_id": names[0]}),
                 ("describe_region", {"stem_id": "master", "start_s": 0.0, "end_s": min(16.0, seconds)}),
                 ("analyze_masking", {"stem_a": names[0], "stem_b": names[1]}),
                 ("analyze_sections", {}), ("analyze_activity", {}),
                 ("analyze_phase", {})]
        for name, args in calls:
            began = time.monotonic()
            result = await engine.call_tool(name, args)
            out[name] = round(time.monotonic() - began, 2)
            if result.isError:
                out[name] = -1.0          # a refusal is not a time
    return out


def measure(project: Path, seconds: int) -> dict[str, float]:
    from mix_engine.session import Session

    from . import daw_bridge as B

    row: dict[str, float] = {}
    began = time.monotonic()
    cache: dict = {}
    B.session_readout(project, cache)
    row["first_readout"] = round(time.monotonic() - began, 2)
    began = time.monotonic()
    B.session_readout(project, cache)
    row["readout_again"] = round(time.monotonic() - began, 2)
    names = sorted(Session(project, read_only=True).stems)
    Session(project).user_edit(names[0], {"gain_db": -3.0})
    began = time.monotonic()
    B.level_readout(project)
    row["measure_after_a_fader"] = round(time.monotonic() - began, 2)
    row.update(asyncio.run(_tools(project, names, seconds)))
    began = time.monotonic()
    B.session_observations(project, wait=True)
    row["observations"] = round(time.monotonic() - began, 2)
    return row


def sweep(sizes: list[tuple[int, int]], work: Optional[Path] = None) -> dict[str, dict[str, float]]:
    from mix_engine import dsp

    from .daw_server import _keep_renders
    _keep_renders()                        # as the DAW's brain runs
    root = work or Path(tempfile.mkdtemp(prefix="size-sweep-"))
    table: dict[str, dict[str, float]] = {}
    try:
        for stems, seconds in sizes:
            project = make_project(root / f"s{stems}_d{seconds}", stems, seconds)
            getattr(dsp, "MASTER_MEMO", {}).clear() if hasattr(dsp, "MASTER_MEMO") else None
            table[f"{stems} stems x {seconds // 60}:{seconds % 60:02d}"] = measure(project, seconds)
            shutil.rmtree(project, ignore_errors=True)
    finally:
        if work is None:
            shutil.rmtree(root, ignore_errors=True)
    return table


def report(table: dict[str, dict[str, float]]) -> str:
    cols = list(next(iter(table.values())))
    head = f"{'project':18} " + " ".join(f"{c[:13]:>13}" for c in cols)
    lines = [head, "-" * len(head)]
    for name, row in table.items():
        lines.append(f"{name:18} " + " ".join(f"{row[c]:13.2f}" for c in cols))
    lines.append("seconds; -1 = the tool refused. first_readout is a project's first whole render.")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--full", action="store_true", help="all nine sizes")
    parser.add_argument("--over", type=float, metavar="SECONDS",
                        help="exit 1 when anything but a first readout takes longer")
    parser.add_argument("--out", type=Path, help="write the table as JSON")
    args = parser.parse_args(argv)
    table = sweep(FULL if args.full else CROSS)
    print(report(table))
    if args.out:
        args.out.write_text(json.dumps(table, indent=1) + "\n")
    if args.over is not None:
        slow = [f"{name}: {col} {s} s" for name, row in table.items() for col, s in row.items()
                if col != "first_readout" and s > args.over]
        for line in slow:
            print("OVER  " + line)
        return 1 if slow else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
