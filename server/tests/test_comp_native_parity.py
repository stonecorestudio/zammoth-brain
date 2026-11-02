"""The compressor's CLEAN law, held between Python and the native chain (2026-10-25).

ONE fixture, `server/tests/data/comp_clean_parity.json`, carries the inputs and PYTHON's answers:
fifteen compressor settings (CLEAN in stereo at medium, fast and slow knobs with makeup and mix, a
side-only comp and a true M/S comp; then each other character, GLUE on a true M/S comp, and four with a knee) rendered by `dsp._apply_comp` on a signal both engines make from the same
formula (three partials under short bursts and a held over) - the output at 800 sample indices per
channel.

The console leg `--comp-clean` reads the same file and asserts `MonitorChain` renders those samples.
THIS test asserts the file is still what Python says, so neither side can move without the other
being made to agree. A law change on purpose: regenerate with

    python server/tests/test_comp_native_parity.py --write

and then run the leg.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mix_engine import dsp  # noqa: E402

FIXTURE = HERE / "data" / "comp_clean_parity.json"
SR = 48000
N = 96000
INDICES = list(range(0, N, 120))
ENTRIES = [
    {"type": "comp", "threshold_db": -20.0, "ratio": 4.0, "attack_ms": 10.0, "release_ms": 100.0},
    {"type": "comp", "threshold_db": -26.0, "ratio": 8.0, "attack_ms": 0.5, "release_ms": 30.0,
     "makeup_db": 4.0},
    {"type": "comp", "threshold_db": -18.0, "ratio": 2.5, "attack_ms": 40.0, "release_ms": 600.0,
     "mix": 0.6},
    {"type": "comp", "threshold_db": -30.0, "ratio": 4.0, "attack_ms": 5.0, "release_ms": 80.0,
     "channel": "side"},
    {"type": "comp", "threshold_db": -20.0, "ratio": 3.0, "attack_ms": 10.0, "release_ms": 150.0,
     "channel": "mid",
     "mid": {"threshold_db": -20.0, "ratio": 3.0, "attack_ms": 10.0, "release_ms": 150.0},
     "side": {"threshold_db": -28.0, "ratio": 6.0, "attack_ms": 2.0, "release_ms": 60.0, "makeup_db": 2.0}},
    # the characters (2026-10-25): each at knobs of its own, GLUE on a true M/S comp
    {"type": "comp", "threshold_db": -22.0, "ratio": 6.0, "attack_ms": 3.0, "release_ms": 80.0, "style": 1},
    {"type": "comp", "threshold_db": -24.0, "ratio": 3.0, "attack_ms": 15.0, "release_ms": 200.0, "style": 2},
    {"type": "comp", "threshold_db": -25.0, "ratio": 4.0, "attack_ms": 20.0, "release_ms": 400.0, "style": 3,
     "makeup_db": 3.0, "mix": 0.8},
    {"type": "comp", "threshold_db": -30.0, "ratio": 4.0, "attack_ms": 5.0, "release_ms": 120.0, "style": 4},
    {"type": "comp", "threshold_db": -18.0, "ratio": 2.0, "attack_ms": 30.0, "release_ms": 300.0, "style": 5},
    {"type": "comp", "threshold_db": -20.0, "ratio": 3.0, "attack_ms": 10.0, "release_ms": 150.0,
     "channel": "mid", "style": 3,
     "mid": {"threshold_db": -20.0, "ratio": 3.0, "attack_ms": 10.0, "release_ms": 150.0},
     "side": {"threshold_db": -30.0, "ratio": 4.0, "attack_ms": 5.0, "release_ms": 80.0}},
    # the knee: CLEAN's measured one, the quadratic (PUNCH), the table axes (SMOOTH, GLUE on a half)
    {"type": "comp", "threshold_db": -24.0, "ratio": 4.0, "attack_ms": 10.0, "release_ms": 100.0, "knee_db": 12.0},
    {"type": "comp", "threshold_db": -22.0, "ratio": 6.0, "attack_ms": 3.0, "release_ms": 80.0, "style": 1,
     "knee_db": 9.0},
    {"type": "comp", "threshold_db": -24.0, "ratio": 3.0, "attack_ms": 15.0, "release_ms": 200.0, "style": 2,
     "knee_db": 30.0},
    {"type": "comp", "threshold_db": -20.0, "ratio": 3.0, "attack_ms": 10.0, "release_ms": 150.0,
     "channel": "mid", "style": 3,
     "mid": {"threshold_db": -20.0, "ratio": 3.0, "attack_ms": 10.0, "release_ms": 150.0, "knee_db": 40.0},
     "side": {"threshold_db": -30.0, "ratio": 4.0, "attack_ms": 5.0, "release_ms": 80.0}},
]


def signal(n: int = N) -> np.ndarray:
    """(2, n) float32. The leg's `ccl::signal` makes the same numbers: a 10 ms burst at 0.9 every
    0.1 s over a bed at 0.25 for the first second, then a held over at 0.7 for half a second and
    the bed again - three partials, the right channel's phases shifted."""
    i = np.arange(n, dtype=np.float64)
    amp = np.where(i < SR, np.where((i % 4800) < 480, 0.9, 0.25),
                   np.where(i < SR + SR // 2, 0.7, 0.25))
    tau = 2.0 * math.pi
    left = amp * (0.6 * np.sin(tau * 110.0 * i / SR) + 0.3 * np.sin(tau * 1370.0 * i / SR + 0.5)
                  + 0.2 * np.sin(tau * 5300.0 * i / SR + 1.1))
    right = amp * (0.6 * np.sin(tau * 110.0 * i / SR + 0.7) + 0.3 * np.sin(tau * 1370.0 * i / SR + 1.9)
                   + 0.2 * np.sin(tau * 5300.0 * i / SR + 2.3))
    return np.ascontiguousarray(np.stack([left, right]).astype(np.float32))


def answers() -> dict:
    x = signal()
    cases = []
    for e in ENTRIES:
        y = dsp._apply_comp(x, SR, e).astype(np.float64)
        cases.append({"entry": e, "left": [float(y[0, k]) for k in INDICES],
                      "right": [float(y[1, k]) for k in INDICES]})
    return {"sr": SR, "n": N, "indices": INDICES, "cases": cases}


def test_the_fixture_is_what_python_renders():
    have = json.loads(FIXTURE.read_text())
    want = answers()
    assert have["indices"] == want["indices"] and have["n"] == want["n"]
    assert len(have["cases"]) == len(want["cases"])
    for h, w in zip(have["cases"], want["cases"]):
        assert h["entry"] == w["entry"]
        assert np.max(np.abs(np.array(h["left"]) - np.array(w["left"]))) < 1e-9
        assert np.max(np.abs(np.array(h["right"]) - np.array(w["right"]))) < 1e-9


def test_the_fixture_compresses():
    """Every case moves the signal by more than the leg's tolerance somewhere - the comparison is
    not two untouched signals."""
    x = signal().astype(np.float64)
    for case in answers()["cases"]:
        y = np.array([case["left"], case["right"]])
        assert np.max(np.abs(y - x[:, INDICES])) > 0.05, case["entry"]


if __name__ == "__main__" and "--write" in sys.argv:
    FIXTURE.write_text(json.dumps(answers(), indent=1) + "\n")
    print("wrote", FIXTURE)
