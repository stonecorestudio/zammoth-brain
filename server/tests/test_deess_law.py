"""The de-esser's law (2026-10-27): fitted to a reference de-esser - `deess_law`.

What is held here:
  * the native header is the table (`DeessLaw.h` is written from it), and the names agree across the
    model, the table and the DAW's named values; the knob set is the reference's (no ratio, attack
    or release - a request that names one is refused by name; a stored range deeper than 24 is held);
  * the reference's MEASURED behaviour, which the de-esser before this did not have (the red half):
    its static curve (a range reached at 0 dBFS on the detector), its split band (a linear-phase
    complementary shelf), its lookahead (the cut is under way when the ess arrives), and VOCAL (an
    ess over a loud vowel is left alone, an ess alone is cut);
  * a range of 0 is off, bit for bit; the two offline engines run the same law.

ONE fixture, `server/tests/data/deess_law_parity.json`, carries Python's answers on a signal both
engines make from one formula; the console leg `--deess-law` asserts MonitorChain renders them, its
output later by exactly the lookahead it reports. A law change on purpose:
`python server/tests/test_deess_law.py --write`, then the leg.
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

from mix_engine import deess_law as dl, dsp, models, reference_monitor  # noqa: E402

SR = 48000
FIXTURE = HERE / "data" / "deess_law_parity.json"


def _entry(**kw) -> dict:
    e = {"type": "deess", "threshold_db": -30.0, "freq_hz": 6000.0, "top_hz": 20000.0, "range_db": -12.0,
         "mode": 0, "wide": False}
    e.update(kw)
    return e


def _tones(parts, seconds: float = 1.0) -> np.ndarray:
    """(2, n) float32: a sum of (freq, peak dB) sines."""
    t = np.arange(int(SR * seconds)) / SR
    x = sum(10.0 ** (d / 20.0) * np.sin(2.0 * math.pi * f * t + 0.3) for f, d in parts)
    return np.stack([x, x]).astype(np.float32)


def _settled_gain_db(entry: dict, x: np.ndarray) -> float:
    y = dsp._deess(x, SR, entry).astype(np.float64)
    tail = slice(x.shape[1] // 2, None)
    return 20.0 * math.log10(np.sqrt(np.mean(y[0, tail] ** 2)) / np.sqrt(np.mean(x[0, tail].astype(np.float64) ** 2)))


# --------------------------------------------------------------------------------- the tables
def test_the_native_header_is_this_table():
    assert (REPO / "native/Source/DeessLaw.h").read_text() == dl.header_text()


def test_the_names_and_the_knob_set_agree():
    assert models.DEESS_MODES == dl.MODES
    specs = (REPO / "native/Source/DeviceSpecs.h").read_text()
    assert 'deessModes { "allround", "vocal" }' in specs
    assert '{ "deess",              "mode",              0.0f,     1.0f,     Snap::index }' in specs
    assert '{ "deess",              "range_db",          -24.0f,   0.0f,     Snap::none }' in specs
    for gone in ("ratio", "attack_ms", "release_ms"):
        assert f'{{ "deess",              "{gone}"' not in specs
        assert gone not in models.DeessAction.model_fields


def test_the_agent_names_a_mode_and_cannot_send_a_ratio():
    a = models.DeessAction(stem_id="vox", threshold_db=-30.0, freq_hz=6000.0, mode="vocal")
    assert a.mode == 1 and a.range_db == -6.0 and a.top_hz == 20000.0
    with pytest.raises(Exception):
        models.DeessAction(stem_id="vox", threshold_db=-30.0, freq_hz=6000.0, mode="shouty")
    with pytest.raises(Exception, match="ratio"):
        models.ProposalRequest.model_validate(
            {"rationale": "esses", "actions": [{"type": "deess", "stem_id": "vox", "threshold_db": -30.0,
                                                "freq_hz": 6000.0, "ratio": 4.0}]})


def test_a_stored_deeper_range_is_held_to_24():
    a = models.DeessAction.model_validate(
        {"type": "deess", "stem_id": "vox", "threshold_db": -30.0, "freq_hz": 6000.0, "range_db": -80.0,
         "ratio": 4.0, "attack_ms": 1.0}, context={"stored": True})
    assert a.range_db == -24.0
    with pytest.raises(Exception):
        models.DeessAction(stem_id="vox", threshold_db=-30.0, freq_hz=6000.0, range_db=-40.0)


# ------------------------------------------------------- the reference's behaviour (the red half)
# Its static curve on a 9 kHz tone (thr -20, range 24, band 7-14 kHz), read off the reference: the cut
# grows 1.2 dB per dB - the range over the threshold - from 5.5 dB under it.
MEASURED_STATIC = {-24.0: -1.86, -20.0: -6.65, -16.0: -11.43, -12.0: -16.21, -8.0: -20.99}


def test_the_static_curve_is_the_references():
    e = _entry(threshold_db=-20.0, range_db=-24.0, freq_hz=7000.0, top_hz=14000.0, wide=True)
    for level, want in MEASURED_STATIC.items():
        got = _settled_gain_db(e, _tones([(9000.0, level)]))
        assert abs(got - want) < 0.6, (level, got, want)


# Its split band at a held 12 dB cut (band 7-14 kHz, the cut held by a loud 9.9 kHz tone): what it
# does to a quiet probe at each frequency - one linear-phase shelf, all the way down to 500 Hz.
MEASURED_SPLIT = {500.0: -0.49, 1000.0: -1.72, 2000.0: -4.69, 4000.0: -8.48}


def test_the_split_band_is_the_references_shelf():
    e = _entry(threshold_db=-60.0, range_db=-12.0, freq_hz=7000.0, top_hz=14000.0)
    t = np.arange(int(SR * 1.5)) / SR
    hold = 10.0 ** (-6.0 / 20.0) * np.sin(2.0 * math.pi * 9899.0 * t)
    for f, want in MEASURED_SPLIT.items():
        probe = 10.0 ** (-70.0 / 20.0) * np.sin(2.0 * math.pi * f * t)
        x = np.stack([hold + probe, hold + probe]).astype(np.float32)
        y = dsp._deess(x, SR, e).astype(np.float64)
        ref = np.exp(-2j * math.pi * f * t)[SR // 2:]
        h = np.mean(y[0, SR // 2:] * ref) / np.mean(x[0, SR // 2:].astype(np.float64) * ref)
        assert abs(20.0 * math.log10(abs(h)) - want) < 0.35, (f, 20.0 * math.log10(abs(h)), want)
        assert abs(math.degrees(np.angle(h))) < 2.0, (f, math.degrees(np.angle(h)))   # zero phase


def test_the_cut_is_under_way_when_the_ess_arrives():
    """The lookahead: at the onset of a burst the reference had 73 % of its steady cut in place."""
    n = SR
    t = np.arange(n) / SR
    a = np.where((t >= 0.3) & (t < 0.7), 10.0 ** (-10.0 / 20.0), 10.0 ** (-60.0 / 20.0))
    x = np.stack([a * np.sin(2.0 * math.pi * 9000.0 * t)] * 2).astype(np.float32)
    g = dl.gain_db(x, SR, -30.0, -24.0, 7000.0, 14000.0, 0)
    on = int(0.3 * SR)
    assert g[on] < 0.5 * g[on + int(0.1 * SR)] < 0.0, (g[on], g[on + int(0.1 * SR)])


def test_vocal_leaves_an_ess_over_a_loud_vowel_alone():
    e = _entry(threshold_db=-40.0, range_db=-24.0, freq_hz=7000.0, top_hz=14000.0, mode=1, wide=True)
    assert _settled_gain_db(e, _tones([(9000.0, -20.0)])) < -6.0               # an ess alone: cut
    assert abs(_settled_gain_db(e, _tones([(9000.0, -20.0), (400.0, -23.0)]))) < 0.1   # over a vowel: left
    allround = dict(e, mode=0)
    assert _settled_gain_db(allround, _tones([(9000.0, -20.0), (400.0, -23.0)])) < -6.0   # allround cuts it


def test_off_is_bit_identical_and_the_engines_agree():
    x = signal()
    assert np.array_equal(dsp._deess(x, SR, _entry(range_db=0.0)), x)
    e = _entry(mode=1, top_hz=12000.0, range_db=-18.0)
    ours = dsp._deess(x, SR, e).astype(np.float64)
    ref = reference_monitor._deess(x.T, SR, e).T
    assert np.max(np.abs(ours - ref)) < 1e-6


# ------------------------------------------------------------------- the native fixture
N = 96000
INDICES = list(range(0, N - 2000, 113))
ENTRIES = [
    _entry(threshold_db=-30.0, range_db=-12.0, freq_hz=6000.0),
    _entry(threshold_db=-24.0, range_db=-6.0, freq_hz=7000.0, top_hz=14000.0, wide=True),
    _entry(threshold_db=-20.0, range_db=-24.0, freq_hz=4000.0, top_hz=9000.0),
    _entry(threshold_db=-36.0, range_db=-12.0, freq_hz=5000.0, mode=1),
    _entry(threshold_db=-30.0, range_db=-24.0, freq_hz=6000.0, top_hz=12000.0, mode=1, wide=True),
]


def signal(n: int = N, sr: float = SR) -> np.ndarray:
    """(2, n) float32; the leg's `dll::signal` makes the same numbers: an 'ess' burst (three high
    partials) every 3000 samples over a vowel (two low partials) that is there in every other
    6000-sample stretch, the right channel's phases shifted."""
    i = np.arange(n, dtype=np.float64)
    tau = 2.0 * math.pi
    vowel_on = ((i // 6000) % 2 == 0).astype(np.float64)
    ess_env = np.exp(-(i % 3000) / 600.0)
    out = []
    for ph in (0.0, 0.9):
        vowel = vowel_on * (0.4 * np.sin(tau * 300.0 * i / sr + ph) + 0.2 * np.sin(tau * 800.0 * i / sr + 2.0 * ph))
        ess = 0.3 * ess_env * (np.sin(tau * 7300.0 * i / sr + ph) + 0.7 * np.sin(tau * 9100.0 * i / sr + 0.4 + ph)
                               + 0.5 * np.sin(tau * 11500.0 * i / sr + 1.3 + ph))
        out.append(vowel + ess)
    return np.ascontiguousarray(np.stack(out).astype(np.float32))


def answers() -> dict:
    x = signal()
    cases = []
    for e in ENTRIES:
        y = dsp._deess(x, SR, e).astype(np.float64)
        cases.append({"entry": e, "left": [float(y[0, k]) for k in INDICES],
                      "right": [float(y[1, k]) for k in INDICES]})
    # ...and at 44.1 kHz (the review of 2026-10-28: every case was 48 kHz, so nothing held the law's
    # dependence on the rate) - each such case names its rate
    x441 = signal(sr=44100)
    for e in (ENTRIES[0], ENTRIES[3]):
        y = dsp._deess(x441, 44100, e).astype(np.float64)
        cases.append({"entry": e, "sr": 44100, "lookahead": dl.lookahead_samples(44100), "left": [float(y[0, k]) for k in INDICES],
                      "right": [float(y[1, k]) for k in INDICES]})
    return {"sr": SR, "n": N, "indices": INDICES, "lookahead": dl.lookahead_samples(SR), "cases": cases}


def test_the_fixture_is_what_python_renders():
    have = json.loads(FIXTURE.read_text())
    want = answers()
    assert have["indices"] == want["indices"] and have["n"] == want["n"] and have["lookahead"] == want["lookahead"]
    assert len(have["cases"]) == len(want["cases"])
    for h, w in zip(have["cases"], want["cases"]):
        assert h["entry"] == w["entry"]
        assert np.max(np.abs(np.array(h["left"]) - np.array(w["left"]))) < 1e-9
        assert np.max(np.abs(np.array(h["right"]) - np.array(w["right"]))) < 1e-9


def test_the_fixture_de_esses():
    """Every case moves the signal well past the leg's tolerance somewhere."""
    for case in json.loads(FIXTURE.read_text())["cases"]:
        x = signal(sr=case.get("sr", SR)).astype(np.float64)
        moved = np.max(np.abs(np.array(case["left"]) - x[0, INDICES]))
        assert moved > 0.01, case["entry"]


if __name__ == "__main__":
    if "--write" in sys.argv:
        FIXTURE.write_text(json.dumps(answers()))
        print(f"wrote {FIXTURE}")
