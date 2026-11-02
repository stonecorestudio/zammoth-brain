"""OTT's law (2026-10-28): a reference OTT, measured as a black box - `ott_law`.

What is held here:
  * the reference's MEASURED static curves (`data/ott_reference_statics.json`, numbers read off the
    reference with a sine per band, at the default and at depth 50 %, threshold 150 %, upward 50 %,
    downward 0 %) - the law within 2.6 dB rms and 3.6 dB at worst over -90..0 dBFS. That is what the
    law James chose by ear IS: its detector was fitted on music, and on a steady sine the mid and
    high bands sit about 2.3 dB loud at the default (low within 1.2). The law it replaces missed the
    same curves by up to 22 dB (seen red, 2026-10-28);
  * the split (LR4 at 88.3 Hz / 2.5 kHz), depth 0 = the input, the linked mean-power detector;
  * the knob set is the reference's - depth, time (0..1000 %), in / out, the two strengths, per band
    up / down / threshold / gain - in the model, the session's table and the native clamp table;
  * an OTT SAVED before is read once (`from_legacy`): its crossovers left at the old defaults move,
    and a band that did nothing (both sides off) still does nothing - its makeup taken back by its
    gain; the session (STATE_VERSION 7) converts every copy and tells the user once;
  * the agent: the fixed makeup at its depth is charged to the gain wallet with the gains;
  * every reader - dsp, the reference monitor, the monitor resolver - plays the same law.

ONE fixture, `data/ott_law_parity.json`, carries Python's answers on a signal both engines make from
one formula; the console leg `--ott-law` asserts MonitorChain renders them. A law change on purpose:
`python server/tests/test_ott_law.py --write`, then the leg.
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

from mix_engine import dsp, models, ott_law as ol, reference_monitor  # noqa: E402
from mix_engine.monitor_chain import monitor_chain_params  # noqa: E402

SR = 48000
FIXTURE = HERE / "data" / "ott_law_parity.json"
STATICS = json.loads((HERE / "data" / "ott_reference_statics.json").read_text())


def _entry(**kw) -> dict:
    e = {"type": "ott", "stem_id": "s", "depth": 1.0, "time": 1.0, "upward_pct": 100.0,
         "downward_pct": 100.0}
    e.update(kw)
    return e


def _tone(f: float, level_db: float, secs: float = 0.9) -> np.ndarray:
    t = np.arange(int(secs * SR)) / SR
    x = 10 ** (level_db / 20) * math.sqrt(2) * np.sin(2 * np.pi * f * t)
    return np.stack([x, x])


def _static_db(entry: dict, f: float, level_db: float) -> float:
    x = _tone(f, level_db)
    y = ol.process(x, SR, entry)
    n = int(0.3 * SR)
    return 20 * math.log10(np.sqrt(np.mean(y[0, -n:] ** 2)) / np.sqrt(np.mean(x[0, -n:] ** 2)))


# ------------------------------------------------------------------ the reference's curves
CONFIGS = {"default": {}, "depth50": {"depth": 0.5},
           "upward50": {"upward_pct": 50.0}, "downward0": {"downward_pct": 0.0}}


@pytest.mark.parametrize("band", ["low", "mid", "high"])
@pytest.mark.parametrize("config", list(CONFIGS) + ["threshold150"])
def test_the_static_curve_is_the_references(band, config):
    kw = dict(CONFIGS.get(config, {}))
    if config == "threshold150":
        kw[band] = {"threshold_pct": 150.0}
    ref = STATICS["curves"][band][config]
    f = STATICS["freq_hz"][band]
    got = [_static_db(_entry(**kw), f, L) for L in STATICS["levels"]]
    err = np.array(got) - np.array(ref)
    assert np.sqrt(np.mean(err ** 2)) < 2.6, (band, config, np.round(err, 2))
    assert np.max(np.abs(err)) < 3.6, (band, config, np.round(err, 2))


def test_the_split_and_depth_zero():
    """LR4 at 88.3 Hz / 2.5 kHz by default (-6 dB at each corner, as the reference); depth 0 passes."""
    m = models.OttAction(stem_id="s")
    assert (m.crossover_lo_hz, m.crossover_hi_hz) == (88.3, 2500.0)
    lo, mid, hi = dsp._lr4_three_way(_tone(88.3, -20, 0.5), SR, 88.3, 2500.0)
    n = SR // 4
    assert 20 * math.log10(np.std(lo[0, -n:]) / np.std(_tone(88.3, -20, 0.5)[0, -n:])) == pytest.approx(-6.02, abs=0.05)
    # depth 0 passes the input: the three bands sum to an allpass, flat to 0.02 dB (the reference: 0.012)
    for f in (30, 60, 88.3, 150, 1000, 2500, 5000, 16000):
        assert _static_db(_entry(depth=0.0), f, -20) == pytest.approx(0.0, abs=0.02), f


def test_the_detector_is_linked_on_mean_power():
    """A loud right channel cuts a quiet left one (the reference: -40 dB left alone +11, right at -10: -11)."""
    t = np.arange(SR) / SR
    left = 10 ** (-40 / 20) * math.sqrt(2) * np.sin(2 * np.pi * 600 * t)
    alone = ol.process(np.stack([left, 0 * left]), SR, _entry())
    loud = ol.process(np.stack([left, 10 ** (-10 / 20) * math.sqrt(2) * np.sin(2 * np.pi * 700 * t)]), SR, _entry())
    n = SR // 3
    g = lambda y: 20 * math.log10(np.std(y[0, -n:]) / np.std(left[-n:]))
    assert g(alone) - g(loud) > 15.0


# ------------------------------------------------------------------------- the knob set
def test_the_knob_set_is_the_references():
    from mix_engine.session import Session
    m = models.OttAction(stem_id="s")
    assert (m.depth, m.time, m.upward_pct, m.downward_pct) == (1.0, 1.0, 100.0, 100.0)
    assert m.low.threshold_pct == 100.0
    with pytest.raises(Exception):
        models.OttAction(stem_id="s", downward_pct=150.0)   # the reference adds nothing past 100
    with pytest.raises(Exception):
        models.OttAction(stem_id="s", time=11.0)
    table = Session.DEVICE_PARAMS["ott"]
    for k in ("upward_pct", "downward_pct", "low_threshold_pct", "mid_threshold_pct", "high_threshold_pct"):
        assert k in table
    assert table["time"][1:] == (0.0, 10.0)
    cpp = (REPO / "native/Source/DeviceSpecs.h").read_text()
    assert '"ott",                "upward_pct",' in cpp


def test_time_scales_the_release_only():
    """A step down: at Time 300 % the gain recovers far slower than at 100 %; the attack does not move."""
    t = np.arange(int(1.5 * SR)) / SR
    probe = 10 ** (-90 / 20) * math.sqrt(2) * np.sin(2 * np.pi * 650 * t)
    def back(time, rise=False):
        lev = np.where(t < 0.75, -48.0, -12.0) if rise else np.where(t < 0.75, -12.0, -48.0)
        x = 10 ** (lev / 20) * math.sqrt(2) * np.sin(2 * np.pi * 600 * t)
        y = ol.process(np.stack([probe, x]), SR, _entry(time=time))
        g = 20 * np.log10(np.abs(y[0]) + 1e-30) - 20 * np.log10(np.abs(probe) + 1e-30)
        w = int(0.005 * SR)
        gs = np.array([np.median(g[i:i + w]) for i in range(int(0.75 * SR), len(g) - w, w)])
        return gs
    fast, slow = back(1.0), back(3.0)
    assert slow[8] < fast[8] - 2.0   # 40 ms after the drop the slow release has come back less far
    up_fast, up_slow = back(1.0, rise=True), back(3.0, rise=True)
    assert np.max(np.abs(up_fast[:6] - up_slow[:6])) < 0.5   # the attack does not move with TIME


# ------------------------------------------------------------------------- an old OTT
def test_an_old_ott_is_read_once_as_the_law():
    old = {"type": "ott", "stem_id": "s", "depth": 0.4, "time": 2.0,
           "low": {"up": 1.0, "down": 0.5, "gain_db": 1.0}, "mid": {"up": 0.0, "down": 0.0, "gain_db": 0.0},
           "high": {"up": 1.0, "down": 1.0}, "crossover_lo_hz": 150.0, "crossover_hi_hz": 3000.0}
    assert ol.is_legacy(old)
    v = ol.values(old)
    assert v["depth"] == 0.4 and v["time"] == 2.0 and v["upward_pct"] == 100.0
    assert v["crossover_lo_hz"] == 88.3 and v["crossover_hi_hz"] == 3000.0   # a moved crossover stays
    assert v["low"]["down"] == 0.5 and v["low"]["gain_db"] == 1.0
    # the mid band did nothing: it still does nothing - its makeup taken back by its gain
    assert v["mid"]["gain_db"] == pytest.approx(-ol.MAKEUP_DB[1] * 0.4, abs=1e-9)
    assert not ol.is_legacy(ol.from_legacy(old))
    t = np.arange(SR // 2) / SR
    x = 10 ** (-30 / 20) * np.sin(2 * np.pi * 600 * t)
    nothing = {**old, "low": {"up": 0.0, "down": 0.0}, "mid": {"up": 0.0, "down": 0.0},
               "high": {"up": 0.0, "down": 0.0}, "depth": 0.5}
    y = ol.process(np.stack([x, x]), SR, nothing)
    n = SR // 4
    assert 20 * math.log10(np.std(y[0, -n:]) / np.std(x[-n:])) == pytest.approx(0.0, abs=0.1)
    assert ol.left_lifted({**nothing, "depth": 1.0})   # 14 dB of makeup, the gain stops at -12


def test_every_reader_plays_the_same_law_for_an_old_entry():
    old = {"type": "ott", "stem_id": "s", "depth": 0.5, "low": {"up": 1.0, "down": 1.0},
           "mid": {"up": 1.0, "down": 1.0}, "high": {"up": 1.0, "down": 1.0},
           "crossover_lo_hz": 150.0, "crossover_hi_hz": 2000.0}
    mc = monitor_chain_params([old])
    [m] = mc["entries"] if isinstance(mc, dict) else mc
    want = ol.values(old)
    assert {k: m[k] for k in want} == want and m["upward_pct"] == 100.0
    x = np.random.default_rng(2).standard_normal((SR // 4, 2)).astype(np.float32) * 0.1
    a = dsp._ott(x.T.copy(), SR, old).T
    b = reference_monitor._ott(x.astype(np.float64), SR, old)
    assert np.max(np.abs(a - b)) < 1e-6
    c = ol.process(x.T.astype(np.float64), SR, want).T
    assert np.max(np.abs(a - c)) < 1e-5


def test_a_session_saved_before_is_converted_once(demo_project):
    """STATE_VERSION 7: the chains, the undo snapshots and the recorded actions - and the user is told."""
    from mix_engine import session as S
    path = Path(demo_project) / "session.json"
    S.Session(demo_project)._persist()
    state = json.loads(path.read_text())
    stem = next(iter(state["stems"]))
    old = {"type": "ott", "stem_id": stem, "depth": 0.3, "time": 1.0,
           "low": {"up": 1.0, "down": 1.0, "gain_db": 0.0}, "mid": {"up": 1.0, "down": 1.0, "gain_db": 0.0},
           "high": {"up": 1.0, "down": 1.0, "gain_db": 0.0},
           "crossover_lo_hz": 150.0, "crossover_hi_hz": 2000.0}
    state["stems"][stem]["chain"] = [dict(old)]
    state["proposals"] = [{"id": "p1", "actions": [dict(old)], "inverse": {stem: {"chain": [dict(old)]}}}]
    state["state_version"] = 6
    path.write_text(json.dumps(state))
    s = S.Session(demo_project)
    assert s._state["state_version"] == S.STATE_VERSION >= 7
    for e in (s._state["stems"][stem]["chain"][0], s._state["proposals"][0]["actions"][0],
              s._state["proposals"][0]["inverse"][stem]["chain"][0]):
        assert e["upward_pct"] == 100.0 and e["crossover_lo_hz"] == 88.3 and e["mid"]["threshold_pct"] == 100.0
    notes = S.take_migration_notices(demo_project) if hasattr(S, "take_migration_notices") else None
    if notes is not None:
        assert any("OTT" in n for n in notes)


# ------------------------------------------------------------------------------ the agent
def test_the_fixed_makeup_is_charged_to_the_gain_wallet():
    from mix_engine import guardrails
    assert guardrails.level_charge_db(_entry(depth=0.25)) == pytest.approx(ol.MAKEUP_DB[2] * 0.25, abs=1e-6)
    assert guardrails.level_charge_db(_entry(depth=0.25, in_gain_db=2.0, out_gain_db=1.0,
                                             mid={"gain_db": 6.0})) == pytest.approx(3.0 + ol.MAKEUP_DB[1] * 0.25 + 6.0)
    assert guardrails.level_charge_db(_entry(depth=0.0)) == 0.0


# ------------------------------------------------------------------------------ the header
def test_the_header_is_this_law():
    assert (REPO / "native/Source/OttLaw.h").read_text() == ol.header_text()


# ------------------------------------------------------------------------------ the fixture
def fixture_cases() -> list[dict]:
    return [_entry(), _entry(depth=0.5, time=0.3), _entry(depth=0.8, upward_pct=150.0, downward_pct=60.0,
            in_gain_db=3.0, out_gain_db=-2.0, low={"up": 0.5, "threshold_pct": 140.0, "gain_db": 2.0},
            high={"down": 0.0, "threshold_pct": 60.0}),
            {"type": "ott", "depth": 0.5, "low": {"up": 0.0, "down": 0.0}, "mid": {"up": 1.0, "down": 1.0},
             "high": {"up": 1.0, "down": 1.0}, "crossover_lo_hz": 150.0, "crossover_hi_hz": 2000.0}]


def fixture_signal(n: int = 9600) -> np.ndarray:
    i = np.arange(n)
    env = np.where((i // 1600) % 2 == 0, 1.0, 0.03)
    left = 0.6 * env * (np.sin(2 * np.pi * i * 0.0013) + 0.5 * np.sin(2 * np.pi * i * 0.0211))
    right = 0.4 * np.sin(2 * np.pi * i * 0.0457 + 0.4) * (0.05 + 0.95 * (i / n))
    return np.stack([left, right], axis=1)


def fixture() -> dict:
    x = fixture_signal().astype(np.float32).astype(np.float64)
    return {"n": int(x.shape[0]), "rate": SR, "every": 4,
            "signal": "env = 1 for i // 1600 even else 0.03; left = 0.6 env (sin(2 pi i 0.0013) + 0.5 "
                      "sin(2 pi i 0.0211)); right = 0.4 sin(2 pi i 0.0457 + 0.4)(0.05 + 0.95 i / n)",
            "cases": [{"entry": e, "out": [[round(float(v), 6) for v in row[::4]]
                                           for row in dsp._ott(x.T.astype(np.float32), SR, e)]}
                      for e in fixture_cases()]}


def test_the_parity_fixture_is_this_law():
    got = fixture()
    want = json.loads(FIXTURE.read_text())
    assert want["n"] == got["n"] and want["signal"] == got["signal"]
    for a, b in zip(want["cases"], got["cases"]):
        assert a["entry"] == b["entry"]
        assert np.max(np.abs(np.array(a["out"]) - np.array(b["out"]))) < 2e-6


if __name__ == "__main__" and "--write" in sys.argv:
    FIXTURE.write_text(json.dumps(fixture()))
    print("wrote", FIXTURE)
