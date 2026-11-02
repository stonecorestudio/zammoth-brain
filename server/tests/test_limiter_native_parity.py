"""The two-stage limiter, held between Python and the native chain (2026-10-25).

ONE fixture, `server/tests/data/limiter_two_stage_parity.json`, carries the inputs and PYTHON's
answers: three limiter settings (CLEAN at its own knobs; LOUD; GLUE half-linked at a -2 dB ceiling
with its own release and a long lookahead) rendered by `dsp._brickwall` on a signal both engines
make from the same formula (three partials under a burst envelope - short hits and held overs) -
the output at 400 sample indices per channel.

The console leg `--limiter-two-stage` reads the same file and asserts `MonitorChain` renders those
samples (block by block, its fixed delay taken off). THIS test asserts the file is still what
Python says, so neither side can move without the other being made to agree. A law change on
purpose: regenerate with

    python server/tests/test_limiter_native_parity.py --write

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

FIXTURE = HERE / "data" / "limiter_two_stage_parity.json"
SR = 48000
N = 24000
INDICES = list(range(0, N, 60))
ENTRIES = [
    {"type": "limiter", "ceiling_dbtp": -1.0},
    {"type": "limiter", "ceiling_dbtp": -1.0, "style": 2},
    {"type": "limiter", "ceiling_dbtp": -2.0, "style": 1, "channel_link": 0.5,
     "release_ms": 120.0, "lookahead_ms": 2.0},
    {"type": "limiter", "ceiling_dbtp": -1.0, "gain_db": 6.0},                # the GAIN knob
]


def signal(n: int = N) -> np.ndarray:
    """(n, 2) float32. The leg's `ltsSignal` makes the same numbers: a burst every 0.1 s (10 ms at
    1.9, the rest at 0.8) over three partials, the right channel's phases shifted."""
    i = np.arange(n, dtype=np.float64)
    amp = np.where((i % 4800) < 480, 1.9, 0.8)
    tau = 2.0 * math.pi
    left = amp * (0.6 * np.sin(tau * 110.0 * i / SR) + 0.3 * np.sin(tau * 1370.0 * i / SR + 0.5)
                  + 0.2 * np.sin(tau * 5300.0 * i / SR + 1.1))
    right = amp * (0.6 * np.sin(tau * 110.0 * i / SR + 0.7) + 0.3 * np.sin(tau * 1370.0 * i / SR + 1.9)
                   + 0.2 * np.sin(tau * 5300.0 * i / SR + 2.3))
    return np.stack([left, right], axis=1).astype(np.float32)


def answers() -> dict:
    x = signal().astype(np.float64)
    cases = []
    for e in ENTRIES:
        y = dsp._brickwall(x, SR, e)
        cases.append({"entry": e, "left": [float(y[k, 0]) for k in INDICES],
                      "right": [float(y[k, 1]) for k in INDICES]})
    return {"sr": SR, "n": N, "indices": INDICES, "cases": cases}


def test_the_fixture_is_what_python_renders():
    have = json.loads(FIXTURE.read_text())
    want = answers()
    assert have["indices"] == want["indices"] and have["n"] == want["n"]
    for h, w in zip(have["cases"], want["cases"]):
        assert h["entry"] == w["entry"]
        assert np.max(np.abs(np.array(h["left"]) - np.array(w["left"]))) < 1e-9
        assert np.max(np.abs(np.array(h["right"]) - np.array(w["right"]))) < 1e-9


def test_the_fixture_limits_hard():
    """The signal must make the law work: every case reduces by several dB somewhere."""
    x = signal().astype(np.float64)
    for e in ENTRIES:
        g, _, _ = dsp.limiter_stages(x, SR, e)
        assert 20 * math.log10(float(g.min())) < -6.0


if __name__ == "__main__" and "--write" in sys.argv:
    FIXTURE.write_text(json.dumps(answers(), indent=1) + "\n")
    print("wrote", FIXTURE)
