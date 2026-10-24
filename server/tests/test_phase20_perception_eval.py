"""Phase 20 — perception-eval harness: the composite metrics must rank a
constructed clip set the way its built-in ratings do (docs/05 Phase 20)."""
from pathlib import Path

import numpy as np
import soundfile as sf

from orchestrator.perception_eval import evaluate

SR = 48000


def _clip(harsh_gain: float, mud_gain: float, seed: int = 11) -> np.ndarray:
    """A noise bed plus scaled harshness (3 kHz region) and mud (250 Hz
    region) tone clusters — more gain = more of the quality."""
    t = np.arange(SR * 4) / SR
    rng = np.random.default_rng(seed)
    bed = 0.05 * np.convolve(rng.standard_normal(t.shape),
                             np.ones(8) / 8.0, mode="same")
    harsh = harsh_gain * sum(np.sin(2 * np.pi * f * t)
                             for f in (2800.0, 3400.0, 4200.0))
    mud = mud_gain * sum(np.sin(2 * np.pi * f * t) for f in (200.0, 250.0))
    return (bed + harsh + mud).astype(np.float32)


def test_composites_rank_a_constructed_set(tmp_path: Path):
    levels = [0.005, 0.02, 0.05, 0.1, 0.2]
    lines = ["path,harshness,mud,clarity"]
    for i, level in enumerate(levels, start=1):
        # clip i: harshness rises with i, mud falls with i
        sf.write(tmp_path / f"clip_{i}.wav",
                 _clip(harsh_gain=level, mud_gain=levels[-i]), SR)
        lines.append(f"clip_{i}.wav,{i},{len(levels) + 1 - i},")
    csv_path = tmp_path / "ratings.csv"
    csv_path.write_text("\n".join(lines) + "\n")

    report = evaluate(csv_path)
    harsh = report["metrics"]["harshness_db"]
    mud = report["metrics"]["mud_db"]
    assert harsh["n"] == len(levels) and mud["n"] == len(levels)
    assert harsh["spearman"] >= 0.9   # acceptance bar from the roadmap
    assert mud["spearman"] >= 0.9
    # clarity column was left blank on every row
    assert report["metrics"]["clarity_index"]["spearman"] is None
