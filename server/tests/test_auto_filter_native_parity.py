"""The Auto Filter and the tempo map, held between Python and the native chain (Phase 63).

ONE fixture, `server/tests/data/auto_filter_parity.json`, carries the inputs and PYTHON's answers:

* six Auto Filter settings (every type, both slopes, every LFO shape, Hz and synced, the envelope
  on its own input and on a trigger) rendered by `dsp._auto_filter` on noise both engines can make
  from a counter hash - the output at 200 sample indices per channel;
* four tempo maps (a step, a linear ramp, two Bezier curves, a 6/8 change, a key change) read as
  quarter notes by `tempo_map.TempoMap` at 37 times each.

The console leg `--auto-filter` reads the same file and asserts the native chain renders those
samples (MonitorChain, block by block, playing from 0) and Tracktion's own `tempo::Sequence` reads
those quarters. THIS test asserts the file is still what Python says - so neither side can move
without the other being made to agree. A law change on purpose: regenerate with

    python server/tests/test_auto_filter_native_parity.py --write

and then run the leg.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mix_engine import dsp  # noqa: E402
from mix_engine.tempo_map import TempoMap  # noqa: E402

FIXTURE = HERE / "data" / "auto_filter_parity.json"
SR = 48000
N = 12000
BLOCK = 512
AMP = 0.25
SEEDS = {"left": 11, "right": 29, "key": 47}
INDICES = list(range(0, N, 60))


def noise(seed: int, n: int = N) -> np.ndarray:
    """lowbias32 of the index + seed, the top 24 bits, as AMP x (-1..1), in float32 - the leg's
    `afNoise` makes the same numbers."""
    i = np.arange(n, dtype=np.uint32)
    with np.errstate(over="ignore"):
        h = i * np.uint32(0x9E3779B1) + np.uint32(seed)
        h ^= h >> np.uint32(16)
        h *= np.uint32(0x7FEB352D)
        h ^= h >> np.uint32(15)
        h *= np.uint32(0x846CA68B)
        h ^= h >> np.uint32(16)
    u = (h >> np.uint32(8)).astype(np.float64) * (1.0 / 16777216.0)
    return (AMP * (2.0 * u - 1.0)).astype(np.float32)


def key_signal() -> np.ndarray:
    """The trigger: noise in bursts (3000 samples on, 3000 at 5 %), so an envelope moves."""
    k = noise(SEEDS["key"]).astype(np.float64)
    gate = np.where((np.arange(N) // 3000) % 2 == 0, 1.0, 0.05)
    return (k * gate).astype(np.float32)


CASES = [
    {"name": "lp12 sine Hz, stereo", "keyed": False,
     "entry": {"filter_type": 0, "slope_db_oct": 12, "cutoff_hz": 900.0, "resonance": 0.5,
               "lfo_amount_oct": 2.0, "lfo_shape": 0, "lfo_rate_hz": 3.0, "lfo_stereo_deg": 90.0}},
    {"name": "hp24 square Hz + own envelope", "keyed": False,
     "entry": {"filter_type": 1, "slope_db_oct": 24, "cutoff_hz": 1500.0, "resonance": 0.8,
               "lfo_amount_oct": 1.5, "lfo_shape": 4, "lfo_rate_hz": 7.0, "env_amount_oct": 2.0,
               "env_attack_ms": 2.0, "env_release_ms": 80.0}},
    {"name": "bp24 random synced 1/16, mix 0.6", "keyed": False,
     "entry": {"filter_type": 2, "slope_db_oct": 24, "cutoff_hz": 600.0, "resonance": 0.3,
               "lfo_amount_oct": 2.0, "lfo_shape": 5, "lfo_sync": 1, "lfo_rate_note": 2, "mix": 0.6}},
    {"name": "notch12 saw down, phase 90", "keyed": False,
     "entry": {"filter_type": 3, "slope_db_oct": 12, "cutoff_hz": 3000.0, "resonance": 1.0,
               "lfo_amount_oct": 1.0, "lfo_shape": 3, "lfo_rate_hz": 11.0, "lfo_phase_deg": 90.0}},
    {"name": "lp24 keyed envelope", "keyed": True,
     "entry": {"filter_type": 0, "slope_db_oct": 24, "cutoff_hz": 300.0, "resonance": 0.6,
               "lfo_amount_oct": 0.0, "env_amount_oct": 3.0, "env_attack_ms": 1.0,
               "env_release_ms": 120.0, "key_stem_id": "trigger"}},
    {"name": "hp12 triangle synced 1/8T, closing envelope", "keyed": False,
     "entry": {"filter_type": 1, "slope_db_oct": 12, "cutoff_hz": 800.0, "resonance": 0.2,
               "lfo_amount_oct": 1.0, "lfo_shape": 1, "lfo_sync": 1, "lfo_rate_note": 3,
               "lfo_stereo_deg": 180.0, "env_amount_oct": -2.0}},
]

TEMPO_CASES = [
    {"name": "120 in 4/4", "tempos": [[0, 120, 0]], "time_sigs": [[0, 4, 4]], "key_beats": []},
    {"name": "a step to 90 at beat 8", "tempos": [[0, 120, 1.0], [8, 90, 1.0]],
     "time_sigs": [[0, 4, 4]], "key_beats": []},
    {"name": "a straight ramp 120 to 60 over 16 beats", "tempos": [[0, 120, 0.0], [16, 60, 0.0]],
     "time_sigs": [[0, 4, 4]], "key_beats": []},
    {"name": "curves, a 6/8 bar and a key change", "tempos": [[0, 100, 0.3], [12, 140, -0.8], [24, 110, 0.0]],
     "time_sigs": [[0, 4, 4], [12, 6, 8]], "key_beats": [6]},
]
TIMES = [round(0.8 * k, 3) for k in range(37)]


def python_answers() -> dict:
    left, right = noise(SEEDS["left"]), noise(SEEDS["right"])
    x = np.stack([left, right], axis=1).astype(np.float64)
    key = key_signal().astype(np.float64)
    out_cases = []
    for case in CASES:
        e = {"type": "auto_filter", **case["entry"]}
        y = dsp._auto_filter(x, SR, e, key_level=np.abs(key) if case["keyed"] else None)
        out_cases.append({**case, "out_l": [round(float(y[i, 0]), 9) for i in INDICES],
                          "out_r": [round(float(y[i, 1]), 9) for i in INDICES]})
    tempo = []
    for case in TEMPO_CASES:
        tm = TempoMap(case["tempos"], case["time_sigs"], case["key_beats"])
        tempo.append({**case, "times": TIMES,
                      "ppq": [round(float(q), 12) for q in tm.ppq_at(np.array(TIMES))]})
    return {"sample_rate": SR, "n": N, "block": BLOCK, "amp": AMP, "seeds": SEEDS,
            "key_bursts": 3000, "indices": INDICES, "cases": out_cases, "tempo": tempo}


def test_the_fixture_is_what_python_says():
    """The leg holds the native chain to these numbers; these numbers are held to Python here."""
    held = json.loads(FIXTURE.read_text())
    now = python_answers()
    assert held["indices"] == now["indices"] and held["seeds"] == now["seeds"]
    for a, b in zip(held["cases"], now["cases"]):
        assert a["entry"] == b["entry"], a["name"]
        assert np.max(np.abs(np.array(a["out_l"]) - np.array(b["out_l"]))) < 1e-8, a["name"]
        assert np.max(np.abs(np.array(a["out_r"]) - np.array(b["out_r"]))) < 1e-8, a["name"]
    for a, b in zip(held["tempo"], now["tempo"]):
        assert np.max(np.abs(np.array(a["ppq"]) - np.array(b["ppq"]))) < 1e-10, a["name"]


def test_the_fixture_is_not_vacuous():
    """Every case filters: its output is not its input and not silence (a null proves nothing on
    a device that did nothing); every tempo map reads different quarters from 120 in 4/4."""
    held = json.loads(FIXTURE.read_text())
    left = noise(SEEDS["left"]).astype(np.float64)[INDICES]
    for case in held["cases"]:
        out = np.array(case["out_l"])
        assert np.max(np.abs(out)) > 0.02, case["name"]
        assert np.max(np.abs(out - left)) > 0.02, case["name"]
    flat = np.array(held["tempo"][0]["ppq"])
    for case in held["tempo"][1:]:
        assert np.max(np.abs(np.array(case["ppq"]) - flat)) > 0.5, case["name"]


def test_the_leg_reads_this_fixture():
    leg = (HERE.parents[1] / "daw" / "Source" / "AutoFilterLegs.h").read_text()
    assert "server/tests/data/auto_filter_parity.json" in leg
    assert "0x9E3779B1u" in leg and "0x7FEB352Du" in leg and "0x846CA68Bu" in leg


if __name__ == "__main__" and "--write" in sys.argv:
    FIXTURE.write_text(json.dumps(python_answers(), indent=1) + "\n")
    print(f"wrote {FIXTURE}")
