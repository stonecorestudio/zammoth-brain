"""The gate's characters (2026-10-27): four laws fitted to a reference gate - `gate_characters`.

What is held here:
  * the native header is the table (`GateCharacters.h` is written from it), and the names agree
    across the model, the table and the DAW's named values;
  * the vectorised law is a literal per-sample loop of `MonitorChain::applyGate` (transcribed below);
  * each character's MEASURED static curve - where it starts cutting under its threshold, and its
    built-in hysteresis - which the gate before this did not have (the red half of this file);
  * a low tone over the threshold passes untouched, as the reference passes it;
  * a ratio of 1 and a range of 0 are off, bit for bit; the agent may name a character.

ONE fixture, `server/tests/data/gate_characters_parity.json`, carries Python's answers on a signal
both engines make from one formula; the console leg `--gate-characters` asserts MonitorChain renders
them. A law change on purpose: `python server/tests/test_gate_characters.py --write`, then the leg.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))

from mix_engine import dsp, gate_characters as gc, models  # noqa: E402

SR = 48000
FIXTURE = HERE / "data" / "gate_characters_parity.json"


def _entry(**kw) -> dict:
    e = {"type": "gate", "threshold_db": -30.0, "ratio": 4.0, "attack_ms": 1.0, "release_ms": 100.0,
         "hold_ms": 0.0, "range_db": -80.0, "hysteresis_db": 0.0}
    e.update(kw)
    return e


def _tone(peak_db: float, seconds: float = 1.0, freq: float = 1000.0, pre_db: float | None = None) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    x = 10.0 ** (peak_db / 20.0) * np.sin(2.0 * math.pi * freq * t)
    if pre_db is not None:
        x[: len(x) // 2] *= 10.0 ** ((pre_db - peak_db) / 20.0)
    return np.stack([x, x]).astype(np.float32)


def _settled_gain_db(entry: dict, x: np.ndarray) -> float:
    y = dsp._gate(x, SR, entry).astype(np.float64)
    tail = slice(x.shape[1] * 7 // 8, None)
    return 20.0 * math.log10(np.sqrt(np.mean(y[0, tail] ** 2)) / np.sqrt(np.mean(x[0, tail].astype(np.float64) ** 2)))


# --------------------------------------------------------------------------------- the tables
def test_the_native_header_is_this_table():
    assert (REPO / "native/Source/GateCharacters.h").read_text() == gc.header_text()


def test_the_names_agree():
    assert models.GATE_CHARACTERS == gc.NAMES
    specs = (REPO / "native/Source/DeviceSpecs.h").read_text()
    assert 'gateCharacters { "classic", "clean", "vocal", "guitar" }' in specs
    assert '{ "gate",               "character",         0.0f,     3.0f,     Snap::index }' in specs


def test_the_agent_names_a_character():
    a = models.GateAction(stem_id="kick", threshold_db=-30.0, character="vocal")
    assert a.character == 2
    with pytest.raises(Exception):
        models.GateAction(stem_id="kick", threshold_db=-30.0, character="shouty")


# ------------------------------------------------------------------- the law, sample by sample
def _native_loop(peak: np.ndarray, p: dict, c: int) -> np.ndarray:
    """MonitorChain::applyGate, transcribed (after the detector EQ)."""
    C = gc.CHARACTERS[c]
    def coef(ms): return math.exp(-1000.0 / (SR * ms)) if ms > 0.0 else 0.0
    a_det = coef(C["det_ms"])
    lv_ms = C["det_attack_k"] * p["attack_ms"] + C["det_attack_ms"]
    lv_up, lv_dn = coef(lv_ms), coef(C["det_release_ms"])
    slow = lv_ms > 0.0 or C["det_release_ms"] > 0.0
    up = coef(C["attack_k"] * p["attack_ms"] + C["attack_ms"])
    down = coef(C["release_k"] * p["release_ms"] ** C["release_p"])
    fall_c = coef(C["fall_ms"] + C["fall_k"] * p["release_ms"])
    floor = min(0.0, p["range_db"])
    thr, hyst = p["threshold_db"], C["hyst_db"] + p["hysteresis_db"]
    hold = int(SR * p["hold_ms"] * C["hold_k"] / 1000.0)
    rest = 10.0 ** (floor / 20.0) if C["lin"] else floor
    det, lvl, is_open, s, fall, hist = 0.0, gc.FLOOR_DB, False, rest, rest, []
    out = np.empty(peak.shape[0])
    for i, x in enumerate(peak):
        det = max(x, det * a_det)
        level = max(gc.FLOOR_DB, 20.0 * math.log10(max(det, 1e-12)))
        if slow:
            lvl = level + (lvl - level) * (lv_up if level > lvl else lv_dn)
            level = lvl
        is_open = level >= thr - hyst if is_open else level >= thr
        target = C["open_db"] if is_open else min(C["open_db"], (p["ratio"] - 1.0) * (level - thr) + C["offset_db"])
        target = max(target, floor)
        hist.append(target)
        target = max(hist[-(hold + 1):])
        t = 10.0 ** (target / 20.0) if C["lin"] else target
        s = t + (s - t) * (up if t > s else down)
        fall = s if s > fall else s + (fall - s) * fall_c
        o = 20.0 * math.log10(max(fall, 1e-9)) if C["lin"] else fall
        out[i] = min(0.0, max(floor, o))
    return out


def _drums(seconds: float = 0.8) -> np.ndarray:
    rng = np.random.default_rng(3)
    t = np.arange(int(SR * seconds)) / SR
    x = 0.3 * np.exp(-((t % 0.25) * 30.0)) * (rng.standard_normal(t.shape[0]) + np.sin(2 * np.pi * 80 * t)) \
        + 0.001 * rng.standard_normal(t.shape[0])
    return np.stack([x, 0.8 * x]).astype(np.float32)


@pytest.mark.parametrize("c", range(len(gc.NAMES)))
def test_the_offline_gate_is_the_native_law(c):
    audio = _drums()
    det = np.max(np.abs(gc.detector_eq(audio.astype(np.float64), SR, c, axis=1)), axis=0)
    for p in (dict(threshold_db=-30.0, ratio=4.0, attack_ms=1.0, release_ms=80.0, hold_ms=20.0,
                   range_db=-40.0, hysteresis_db=3.0),
              dict(threshold_db=-26.0, ratio=20.0, attack_ms=20.0, release_ms=300.0, hold_ms=0.0,
                   range_db=-60.0, hysteresis_db=0.0)):
        want = _native_loop(det, p, c)
        got = gc.gain_db(det, SR, p["threshold_db"], p["ratio"], p["attack_ms"], p["release_ms"],
                         p["hold_ms"], p["range_db"], p["hysteresis_db"], c)
        # the offline followers run in float32 (dsp._gr_ballistics_db) and the native in double: on
        # a slow follower of the LINEAR gain (clean, vocal) float32 leaves ~2e-4 of gain - 0.13 dB at
        # -40 dB, nothing near 0 dB. So: within 1e-3 of linear gain everywhere, 0.05 dB where the gate
        # passes more than -20 dB.
        assert np.max(np.abs(10.0 ** (got / 20.0) - 10.0 ** (want / 20.0))) < 1.0e-3, (gc.NAMES[c], p)
        near = want > -20.0
        assert np.max(np.abs(got - want)[near]) < 0.05, (gc.NAMES[c], p)


# ------------------------------------------------------------------- the measured static curve
# (peak level dBFS of a 1 kHz tone from silence, the reference's settled gain at 4:1, threshold -30)
MEASURED_RISING = {
    "classic": {-34.0: -4.0, -36.0: -10.3, -40.0: -22.0},
    "clean":   {-32.0: -2.0, -34.0: -8.0, -38.0: -20.0},
    "vocal":   {-32.0: -5.0, -34.0: -11.0, -38.0: -23.0},
    "guitar":  {-32.0: -6.0, -34.0: -12.0, -38.0: -24.0},
}


@pytest.mark.parametrize("name", gc.NAMES)
def test_each_character_cuts_where_the_reference_cuts(name):
    c = gc.NAMES.index(name)
    for level, want in MEASURED_RISING[name].items():
        got = _settled_gain_db(_entry(character=c, range_db=-80.0), _tone(level))
        assert abs(got - want) < 1.0, (name, level, got, want)


# (a tone falling from -10 dBFS: the level the reference stayed open at, and one it shut at - read at
# 100:1; ours is at 20:1, our ratio's top, so guitar's shut level is 3 dB under its threshold, not 1)
MEASURED_FALLING = {"classic": (-36.0, -38.0), "clean": (-38.0, -40.0), "vocal": (-36.0, -38.0),
                    "guitar": (-29.0, -33.0)}


@pytest.mark.parametrize("name", gc.NAMES)
def test_each_character_has_the_reference_hysteresis(name):
    c = gc.NAMES.index(name)
    still_open, shut = MEASURED_FALLING[name]
    e = _entry(character=c, ratio=20.0, release_ms=20.0)
    assert _settled_gain_db(e, _tone(still_open, 2.0, pre_db=-10.0)) > -1.0, name
    assert _settled_gain_db(e, _tone(shut, 2.0, pre_db=-10.0)) < -40.0, name


@pytest.mark.parametrize("name", gc.NAMES)
def test_a_low_tone_over_the_threshold_passes_clean(name):
    c = gc.NAMES.index(name)
    for over in (3.0, 10.0):
        x = _tone(-26.0 + over, 1.0, 100.0)
        got = _settled_gain_db(_entry(character=c, threshold_db=-26.0, ratio=8.0, release_ms=5.0), x)
        assert abs(got) < 0.1, (name, over, got)


def test_off_is_bit_identical():
    x = _drums()
    for c in range(len(gc.NAMES)):
        assert np.array_equal(dsp._gate(x, SR, _entry(character=c, ratio=1.0)), x)
        assert np.array_equal(dsp._gate(x, SR, _entry(character=c, range_db=0.0)).astype(np.float32), x)


# ------------------------------------------------------------------- the native fixture
N = 96000
INDICES = list(range(0, N, 120))
ENTRIES = [
    _entry(character=0, threshold_db=-24.0, ratio=4.0, attack_ms=1.0, release_ms=100.0, range_db=-40.0),
    _entry(character=0, threshold_db=-20.0, ratio=20.0, attack_ms=0.2, release_ms=30.0, hold_ms=15.0),
    _entry(character=1, threshold_db=-24.0, ratio=4.0, attack_ms=2.0, release_ms=150.0, range_db=-40.0),
    _entry(character=1, threshold_db=-20.0, ratio=10.0, attack_ms=5.0, release_ms=400.0, hold_ms=30.0,
           range_db=-60.0, hysteresis_db=2.0),
    _entry(character=2, threshold_db=-24.0, ratio=4.0, attack_ms=1.0, release_ms=100.0, range_db=-40.0),
    _entry(character=2, threshold_db=-30.0, ratio=8.0, attack_ms=10.0, release_ms=250.0, hold_ms=20.0),
    _entry(character=3, threshold_db=-24.0, ratio=4.0, attack_ms=1.0, release_ms=100.0, range_db=-40.0),
    _entry(character=3, threshold_db=-20.0, ratio=20.0, attack_ms=5.0, release_ms=300.0, hold_ms=20.0,
           range_db=-60.0, sc_hpf_hz=120.0),
]


def signal(n: int = N, sr: float = SR) -> np.ndarray:
    """(2, n) float32; the leg's `gcl::signal` makes the same numbers: bursts that decay every
    0.125 s (a low, a mid and a high partial) over a quiet bed, the right channel's phases shifted."""
    i = np.arange(n, dtype=np.float64)
    env = 0.02 + 0.8 * np.exp(-(i % 6000) / 900.0)
    tau = 2.0 * math.pi
    left = env * (0.5 * np.sin(tau * 90.0 * i / sr) + 0.3 * np.sin(tau * 1100.0 * i / sr + 0.5)
                  + 0.2 * np.sin(tau * 6200.0 * i / sr + 1.1))
    right = env * (0.5 * np.sin(tau * 90.0 * i / sr + 0.7) + 0.3 * np.sin(tau * 1100.0 * i / sr + 1.9)
                   + 0.2 * np.sin(tau * 6200.0 * i / sr + 2.3))
    return np.ascontiguousarray(np.stack([left, right]).astype(np.float32))


def answers() -> dict:
    x = signal()
    cases = []
    for e in ENTRIES:
        y = dsp._gate(x, SR, e).astype(np.float64)
        cases.append({"entry": e, "left": [float(y[0, k]) for k in INDICES],
                      "right": [float(y[1, k]) for k in INDICES]})
    # ...and at 44.1 kHz (the review of 2026-10-28: every case was 48 kHz, so nothing held the law's
    # dependence on the rate) - each such case names its rate
    x441 = signal(sr=44100)
    for e in (ENTRIES[2], ENTRIES[4]):
        y = dsp._gate(x441, 44100, e).astype(np.float64)
        cases.append({"entry": e, "sr": 44100, "left": [float(y[0, k]) for k in INDICES],
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


def test_the_fixture_gates():
    """Every case moves the signal well past the leg's tolerance somewhere."""
    for case in json.loads(FIXTURE.read_text())["cases"]:
        x = signal(sr=case.get("sr", SR)).astype(np.float64)
        moved = np.max(np.abs(np.array(case["left"]) - x[0, INDICES]))
        assert moved > 0.01, case["entry"]


if __name__ == "__main__":
    if "--write" in sys.argv:
        FIXTURE.write_text(json.dumps(answers()))
        print(f"wrote {FIXTURE}")
