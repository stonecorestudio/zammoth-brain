"""Perception-eval harness (Phase 20 — docs/05-roadmap.md).

The composite metrics (harshness_db, mud_db, clarity_index) are formulas
with hand-picked constants (mix-analysis skill). This harness measures how
well they track human judgment: point it at a CSV of rated clips and it
reports the Spearman rank correlation per metric. That makes the constants
tunable against perception instead of by feel — a change to the composite
formulas is an improvement when these correlations rise.

CSV format (header required; ratings on any consistent scale, higher =
more of that quality; blank cells skip that clip for that metric; paths
relative to the CSV file):

    path,harshness,mud,clarity
    clips/mix_a.wav,7,3,
    clips/mix_b.wav,2,6,4

Run:

    python -m orchestrator.perception_eval ratings.csv [--json out.json]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

from .engine_models import LIMITS  # noqa: F401  (bootstraps server/ onto sys.path)

from mix_engine import analysis

# CSV column -> the SessionAnalysis composite it claims to rate.
RATING_METRICS = {
    "harshness": "harshness_db",
    "mud": "mud_db",
    "clarity": "clarity_index",
}
MIN_CLIPS = 3  # a rank correlation over fewer pairs is noise


def _measure(clip_path: Path) -> dict[str, Optional[float]]:
    import soundfile as sf
    audio, sample_rate = sf.read(clip_path, dtype="float32", always_2d=True)
    result = analysis.analyze_master(audio, sample_rate, stems=[],
                                     pass_number=0)
    return {column: getattr(result, metric)
            for column, metric in RATING_METRICS.items()}


def _spearman(ratings: list[float], measured: list[float]) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(ratings, measured)[0])


def evaluate(csv_path: Path) -> dict:
    """Per composite: number of rated clips and the Spearman rank
    correlation of the metric against the human ratings (None under
    MIN_CLIPS). Clips are measured once each."""
    csv_path = Path(csv_path)
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"no rated clips in {csv_path}")

    pairs: dict[str, list[tuple[float, float]]] = {
        column: [] for column in RATING_METRICS}
    for row in rows:
        clip = csv_path.parent / row["path"]
        measured = _measure(clip)
        for column in RATING_METRICS:
            rating = (row.get(column) or "").strip()
            if not rating or measured[column] is None:
                continue
            pairs[column].append((float(rating), measured[column]))

    report = {}
    for column, metric in RATING_METRICS.items():
        n = len(pairs[column])
        report[metric] = {
            "n": n,
            "spearman": (round(_spearman([r for r, _ in pairs[column]],
                                         [m for _, m in pairs[column]]), 3)
                         if n >= MIN_CLIPS else None),
        }
    return {"csv": str(csv_path), "clips": len(rows), "metrics": report}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Composite-metric perception eval (Phase 20)")
    parser.add_argument("ratings_csv", type=Path,
                        help="CSV of clip paths + human ratings (see module doc)")
    parser.add_argument("--json", type=Path, default=None,
                        help="also write the report here")
    args = parser.parse_args()
    report = evaluate(args.ratings_csv)
    text = json.dumps(report, indent=2)
    if args.json is not None:
        args.json.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
