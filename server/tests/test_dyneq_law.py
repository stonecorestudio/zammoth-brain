"""The dynamic EQ's law (2026-10-27): fitted to a reference EQ's dynamic band - `dyneq_law`.

Held here:
  * the reference's MEASURED behaviour, which the dynamic EQ before this did not have (the red half):
    its static cut (no ratio: dB for dB over the threshold through a range-wide soft knee, stopped at
    the range), how wide its detector listens, its release and its attack;
  * the knob set is the reference's: no ratio anywhere (the model, the table, the native table, the
    prompts); RANGE is the old gain_db, +-30 for a hand on it;
  * a range of 0 is off, bit for bit; the two offline engines run the same law.

ONE fixture, `data/dyneq_law_parity.json`, carries Python's answers on a signal both engines make from
one formula; the console leg `--dyneq-law` asserts MonitorChain renders them. A law change on purpose:
`python server/tests/test_dyneq_law.py --write`, then the leg.
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

from mix_engine import dsp, dyneq_law as dl, models, reference_monitor  # noqa: E402

SR = 48000
FIXTURE = HERE / "data" / "dyneq_law_parity.json"
ATK50, REL50 = 3.41, 245.6          # the reference's 50 % / 50 %, as this law's milliseconds


def _entry(**kw) -> dict:
    e = {"type": "dynamic_eq", "freq_hz": 1000.0, "q": 1.0, "threshold_db": -40.0, "gain_db": -30.0,
         "attack_ms": ATK50, "release_ms": REL50}
    e.update(kw)
    return e


def _tone(f: float, db: float, secs: float = 1.2) -> np.ndarray:
    t = np.arange(int(SR * secs)) / SR
    s = 10.0 ** (db / 20.0) * np.sin(2 * math.pi * f * t)
    return np.stack([s, s]).astype(np.float32)


def _settled_cut(entry: dict, x: np.ndarray) -> float:
    y = dsp._dynamic_eq(x, SR, entry).astype(np.float64)
    tail = slice(x.shape[1] - SR // 2, None)
    return -20.0 * math.log10(np.sqrt(np.mean(y[0, tail] ** 2)) / np.sqrt(np.mean(x[0, tail].astype(np.float64) ** 2)))


def _probe_cut(entry: dict, drive_hz: float, drive_db: float, secs: float = 1.2) -> float:
    """The cut at the band's frequency, read by lock-in on a quiet probe on the LEFT while the RIGHT
    carries the drive (the detector listens to the louder side), as the reference was measured."""
    t = np.arange(int(SR * secs)) / SR
    probe = 1e-3 * np.sin(2 * math.pi * entry["freq_hz"] * t)
    drive = 10.0 ** (drive_db / 20.0) * np.sin(2 * math.pi * drive_hz * t)
    y = dsp._dynamic_eq(np.stack([probe, drive]).astype(np.float32), SR, entry).astype(np.float64)
    m = slice(SR // 2, None)
    ref = np.exp(-2j * math.pi * entry["freq_hz"] * t[m])
    return -20.0 * math.log10(abs(np.sum(y[0, m] * ref) / np.sum(probe[m] * ref)))


# ------------------------------------------------ the reference's behaviour (the red half)
# Its cut on a sine at the band's frequency, threshold -40, by how far over it the sine's peak is,
# read off the reference at two ranges.
MEASURED_STATIC = {30.0: {-10: 1.847, 0: 5.766, 10: 11.863, 20: 20.137, 36: 30.0},
                   12.0: {-4: 0.094, 0: 1.36, 4: 4.126, 12: 12.0}}


def test_the_static_cut_is_the_references():
    for R, rows in MEASURED_STATIC.items():
        e = _entry(gain_db=-R)
        for d, want in rows.items():
            got = _settled_cut(e, _tone(1000.0, -40.0 + d))
            assert abs(got - want) < 0.3, (R, d, got, want)


# Its detector: a drive away from the band, the cut read at the band (threshold -40, range 30).
MEASURED_DETECT = [(1.0, 2000.0, 15.28), (1.0, 4000.0, 10.0), (4.0, 1414.0, 11.74), (4.0, 2000.0, 7.67)]


def test_the_detector_listens_as_wide_as_the_references():
    for q, f, want in MEASURED_DETECT:
        got = _probe_cut(_entry(q=q, attack_ms=1.12, release_ms=110.6), f, -20.0)
        assert abs(got - want) < 0.6, (q, f, got, want)


def _trace(entry: dict, secs: float = 3.0, win_ms: float = 1.0):
    n = int(SR * secs)
    t = np.arange(n) / SR
    a = np.where((t >= 0.5) & (t < 1.5), 10.0 ** (-15.0 / 20.0), 0.0)
    probe = 1e-3 * np.sin(2 * math.pi * 1000.0 * t)
    y = dsp._dynamic_eq(np.stack([probe, a * np.sin(2 * math.pi * 1000.0 * t)]).astype(np.float32), SR, entry)
    w = int(SR * win_ms / 1000)
    ref = np.exp(-2j * math.pi * 1000.0 * t)
    ts, g = [], []
    for i in range(0, n - w, w // 2):
        h = np.sum(y[0, i:i + w].astype(np.float64) * ref[i:i + w]) / np.sum(probe[i:i + w] * ref[i:i + w])
        ts.append((i + w / 2) / SR)
        g.append(20 * math.log10(abs(h)))
    return np.array(ts), np.array(g)


def test_the_release_is_the_references():
    """Release 50 %: the reference took 308 ms to give back 63 % of a 25 dB cut, 632 ms for 90 %."""
    ts, g = _trace(_entry(attack_ms=1.12), win_ms=2.0)
    m = ts > 1.5
    st = np.median(g[(ts > 1.3) & (ts < 1.48)])
    t63 = (ts[m][np.argmax(g[m] >= st * 0.368)] - 1.5) * 1000
    t90 = (ts[m][np.argmax(g[m] >= st * 0.1)] - 1.5) * 1000
    assert abs(st + 25.0) < 0.3 and abs(t63 - 308) < 30 and abs(t90 - 632) < 60, (st, t63, t90)


def test_the_attack_is_the_references():
    """Attack 75 %: the reference took 13.5 ms to reach 63 % of its cut, 30.5 ms for 90 %."""
    ts, g = _trace(_entry(attack_ms=12.85), secs=1.0)
    m = ts > 0.5
    fin = np.median(g[(ts > 0.9) & (ts < 0.98)])
    t63 = (ts[m][np.argmax(g[m] <= 0.632 * fin)] - 0.5) * 1000
    t90 = (ts[m][np.argmax(g[m] <= 0.9 * fin)] - 0.5) * 1000
    assert abs(t63 - 13.5) < 2.0 and abs(t90 - 30.5) < 4.0, (t63, t90)


# ------------------------------------------------ the knob set
def test_no_ratio_anywhere():
    from mix_engine.session import Session
    assert "ratio" not in models.DynamicEqAction.model_fields
    assert "ratio" not in Session.DEVICE_PARAMS["dynamic_eq"]
    assert Session.DEVICE_PARAMS["dynamic_eq"]["gain_db"] == ("gain_db", -30.0, 30.0)
    specs = (REPO / "native/Source/DeviceSpecs.h").read_text()
    assert '{ "dynamic_eq",         "ratio"' not in specs
    for p in ("orchestrator/prompts.py", "orchestrator/chat.py", "orchestrator/agents.py"):
        text = (REPO / p).read_text()
        i = text.index("dynamic_eq: {type")
        assert "ratio" not in text[i:i + 220], p
    with pytest.raises(Exception, match="ratio"):
        models.ProposalRequest.model_validate(
            {"rationale": "x", "actions": [{"type": "dynamic_eq", "stem_id": "vox", "freq_hz": 3000.0, "gain_db": -4.0,
                                            "q": 2.0, "threshold_db": -20.0, "ratio": 4.0}]})


def test_the_native_header_is_this_table():
    assert (REPO / "native/Source/DynEqLaw.h").read_text() == dl.header_text()


def test_off_is_bit_identical_and_the_engines_agree():
    x = signal()
    assert np.array_equal(dsp._dynamic_eq(x, SR, _entry(gain_db=0.0)), x)
    e = _entry(freq_hz=3000.0, q=2.0, threshold_db=-24.0, gain_db=-8.0, attack_ms=2.0, release_ms=150.0)
    ours = dsp._dynamic_eq(x, SR, e).astype(np.float64)
    ref = reference_monitor._dynamic_eq(x.T, SR, e["freq_hz"], e["q"], e["threshold_db"], e["attack_ms"],
                                        e["release_ms"], e["gain_db"]).T
    assert np.max(np.abs(ours - ref)) < 1e-6


# ------------------------------------------------ the native fixture
N = 72000
INDICES = list(range(0, N - 100, 97))
ENTRIES = [
    _entry(freq_hz=300.0, q=1.4, threshold_db=-30.0, gain_db=-6.0, attack_ms=3.41, release_ms=245.6),
    _entry(freq_hz=3000.0, q=2.0, threshold_db=-36.0, gain_db=-12.0, attack_ms=1.26, release_ms=146.9),
    _entry(freq_hz=8000.0, q=1.0, threshold_db=-40.0, gain_db=6.0, attack_ms=12.85, release_ms=379.3),
    _entry(freq_hz=1200.0, q=6.0, threshold_db=-24.0, gain_db=-30.0, attack_ms=1.0, release_ms=40.0),
]


def signal(n: int = N, sr: float = SR) -> np.ndarray:
    """(2, n) float32; the leg's `dql::signal` makes the same numbers: a low pad, a 3 kHz burst every
    6000 samples and an 8 kHz shimmer that swells, the right channel's phases shifted."""
    i = np.arange(n, dtype=np.float64)
    tau = 2.0 * math.pi
    out = []
    for ph in (0.0, 0.7):
        pad = 0.25 * np.sin(tau * 300.0 * i / sr + ph) + 0.1 * np.sin(tau * 1200.0 * i / sr + 2 * ph)
        burst = 0.35 * np.exp(-(i % 6000) / 900.0) * np.sin(tau * 3000.0 * i / sr + ph)
        swell = 0.12 * (0.5 - 0.5 * np.cos(tau * i / 24000.0)) * np.sin(tau * 8000.0 * i / sr + 0.3 + ph)
        out.append(pad + burst + swell)
    return np.ascontiguousarray(np.stack(out).astype(np.float32))


def answers() -> dict:
    x = signal()
    cases = []
    for e in ENTRIES:
        y = dsp._dynamic_eq(x, SR, e).astype(np.float64)
        cases.append({"entry": e, "left": [float(y[0, k]) for k in INDICES], "right": [float(y[1, k]) for k in INDICES]})
    # ...and at 44.1 kHz (the review of 2026-10-28: every case was 48 kHz, so nothing held the law's
    # dependence on the rate) - each such case names its rate
    x441 = signal(sr=44100)
    for e in (ENTRIES[1], ENTRIES[2]):
        y = dsp._dynamic_eq(x441, 44100, e).astype(np.float64)
        cases.append({"entry": e, "sr": 44100, "left": [float(y[0, k]) for k in INDICES],
                      "right": [float(y[1, k]) for k in INDICES]})
    return {"sr": SR, "n": N, "indices": INDICES, "cases": cases}


def test_the_fixture_is_what_python_renders():
    have, want = json.loads(FIXTURE.read_text()), answers()
    assert have["indices"] == want["indices"] and len(have["cases"]) == len(want["cases"])
    for h, w in zip(have["cases"], want["cases"]):
        assert h["entry"] == w["entry"]
        assert np.max(np.abs(np.array(h["left"]) - np.array(w["left"]))) < 1e-9
        assert np.max(np.abs(np.array(h["right"]) - np.array(w["right"]))) < 1e-9


def test_the_fixture_moves_the_band():
    for case in json.loads(FIXTURE.read_text())["cases"]:
        x = signal(sr=case.get("sr", SR)).astype(np.float64)
        assert np.max(np.abs(np.array(case["left"]) - x[0, INDICES])) > 0.01, case["entry"]


if __name__ == "__main__":
    if "--write" in sys.argv:
        FIXTURE.write_text(json.dumps(answers()))
        print(f"wrote {FIXTURE}")
