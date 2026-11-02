"""OTT's law (2026-10-28): a three-band upward + downward compressor fitted to a reference OTT
measured as a black box (James's pick, B) - our DSP, the reference's knob set.

Measured (headless through Pedalboard; the session's scratchpad has the probes and the fits):

* SPLIT: a Linkwitz-Riley 4th-order three-way at 88.3 Hz and 2.5 kHz, -6 dB at each corner, the
  slopes to 0.1 dB - the DAW's own split (`dsp._lr4_three_way`). Depth 0 is the input to 0.01 dB.
* DETECTOR, per band: the MEAN POWER of the channels (a loud right channel cuts the left), averaged
  over a short window (WINDOW_MS, a boxcar), then followed on power (attack ATTACK_MS, release
  FLOOR_MS), then held by a second follower that rises at once and falls with RELEASE_K x the TIME
  knob (in %; 100 % = 1.0 here). Its level D is 10 log10 of that. A sine, a two-tone and band noise
  of one RMS get the same gain within 0.4 dB: it reads RMS, not peaks.
* CURVE, per band, in dB: MAKEUP x depth (a fixed lift: about +14 / +9 / +14 dB at full depth), plus
  UPWARD = UP_SLOPE x softplus(UP_THR - D; UP_KNEE) below the upward threshold, capped at UP_CAP, minus
  DOWNWARD = DOWN_SLOPE x softplus(D - DOWN_THR; DOWN_KNEE), capped at DOWN_CAP. The curves were
  fitted on sines first (0.7-0.8 dB rms over -90..0 dBFS); the detector was then fitted on MUSIC,
  and the sines moved: the shipped law reads the reference's static curves within 0.4-2.5 dB rms
  (3.4 dB at worst: the mid and high bands sit about 2.3 dB loud at the default on a steady sine).
* DEPTH scales the makeup in proportion and each side's slope by depth ^ UP_EXP / ^ DOWN_EXP (the
  upward side a little more than in proportion); the caps stay where they are.
* THRESHOLD (per band, %): moves the upward threshold by UP_THR_K x 20 log10(t / 100) dB and the
  downward one by DOWN_THR_K x the same - about 9-11 dB per 50 %.
* UPWARD / DOWNWARD STRENGTH (%): scale the slopes. Downward strength past 100 % adds nothing in the
  reference, so ours stops at 100. Each band's `up` / `down` (0..1) scale that band's side too.
* IN and OUT gain are a pure gain before and after; a band's gain_db is a gain after its band (the
  reference's band gain is not quite a post gain - +6 dB adds 4 to 6 - and that is not copied).

On a released pop master (0:03-0:24) this nulls 26 / 29 / 29 dB under what the reference changed at
depth 100 / 50 / 25 (the law it replaces: 14 / 9 / 7), the level within 1 dB, and 30-31 dB at Time
20 / 300 / 1000 %. The sine step traces fit to 1.5-2 dB: one detector shape could not match both
sines and music, and this one is the music's (James chose it by ear). At full depth its peaks run
about 2 dB over the reference's.

`OttLaw.h` is written from this file (`python -m mix_engine.ott_law --write-header`).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HEADER = Path(__file__).resolve().parents[2] / "native/Source/OttLaw.h"

BANDS = ("low", "mid", "high")
CROSSOVER_LO_HZ = 88.3
CROSSOVER_HI_HZ = 2500.0
LEGACY_CROSSOVERS_HZ = (150.0, 2000.0)   # the law before: an old OTT left there is moved to these

# Per band (low, mid, high) - the fit (dB, dB/dB, ms).
MAKEUP_DB = (13.962, 9.351, 13.975)
UP_THR_DB = (-44.964, -45.080, -43.973)
UP_SLOPE = (0.739, 0.746, 0.739)
UP_CAP_DB = (35.969, 35.983, 35.960)
UP_KNEE_DB = (0.973, 0.679, 1.082)
DOWN_THR_DB = (-37.833, -33.314, -38.622)
DOWN_SLOPE = (1.030, 1.010, 1.014)
DOWN_CAP_DB = (41.092, 39.788, 40.170)
DOWN_KNEE_DB = (0.955, 0.834, 0.154)
UP_EXP = (0.806, 0.910, 0.802)
DOWN_EXP = (1.278, 1.158, 1.251)
UP_THR_K = (2.654, 2.538, 2.533)
DOWN_THR_K = (2.585, 2.659, 2.567)
WINDOW_MS = (0.921, 5.285, 6.188)
ATTACK_MS = (3.809, 0.469, 0.017)
FLOOR_MS = (6.329, 9.709, 5.813)
RELEASE_K = (28.6, 31.0, 16.3)   # ms of release per 1.0 of TIME (100 %)

HOLD_ATTACK_MS = 0.01   # the hold's rise (see detector_db), the same coefficient natively
RING_MAX = 2048   # the native window's ring (samples): 6.19 ms at 192 kHz is 1188

# The knobs, their ranges and defaults (the reference's defaults).
DEPTH = (0.0, 1.0, 1.0)
TIME = (0.0, 10.0, 1.0)
GAIN_DB = (-12.0, 12.0, 0.0)
UPWARD_PCT = (0.0, 200.0, 100.0)
DOWNWARD_PCT = (0.0, 100.0, 100.0)
THRESHOLD_PCT = (0.0, 200.0, 100.0)
SIDE = (0.0, 1.0, 1.0)


def _clip(v, rng) -> float:
    lo, hi, dflt = rng
    try:
        v = float(v)
    except (TypeError, ValueError):
        return dflt
    if not math.isfinite(v):
        return dflt
    return min(hi, max(lo, v))


def is_legacy(entry: dict) -> bool:
    """An OTT saved before this law: it has no `upward_pct` (the stamp - every entry written since
    carries it)."""
    return isinstance(entry, dict) and entry.get("type") == "ott" and "upward_pct" not in entry


def from_legacy(entry: dict) -> dict:
    """An old OTT read as this law's knobs (once): depth, time, the gains and each band's up /
    down / gain keep their meaning; the strengths and thresholds start at 100 %; crossovers left at
    the old defaults move to this law's, a moved crossover stays where it was put."""
    e = dict(entry)
    e["upward_pct"] = UPWARD_PCT[2]
    e["downward_pct"] = DOWNWARD_PCT[2]
    depth = _clip(e.get("depth", 0.5), DEPTH)   # the old model's default depth
    e["depth"] = depth
    for i, b in enumerate(BANDS):
        band = dict(e.get(b) or {})
        band.setdefault("threshold_pct", THRESHOLD_PCT[2])
        # A band with both sides off did nothing; this law still lifts it by its makeup, so the
        # band's gain takes the makeup back (as far as the gain's range goes - see `left_lifted`).
        if _clip(band.get("up", 1.0), SIDE) == 0.0 and _clip(band.get("down", 1.0), SIDE) == 0.0:
            band["gain_db"] = _clip(float(band.get("gain_db", 0.0) or 0.0) - MAKEUP_DB[i] * depth, GAIN_DB)
        e[b] = band
    lo, hi = e.get("crossover_lo_hz"), e.get("crossover_hi_hz")
    if lo is None or float(lo) == LEGACY_CROSSOVERS_HZ[0]:
        e["crossover_lo_hz"] = CROSSOVER_LO_HZ
    if hi is None or float(hi) == LEGACY_CROSSOVERS_HZ[1]:
        e["crossover_hi_hz"] = CROSSOVER_HI_HZ
    if "time" in e:
        e["time"] = _clip(e["time"], TIME)
    return e


def left_lifted(entry: dict) -> bool:
    """An old OTT whose band did nothing and that this law still lifts after `from_legacy` (the
    band's gain could not take the whole makeup back)."""
    if not is_legacy(entry):
        return False
    e = from_legacy(entry)
    for i, b in enumerate(BANDS):
        band = e[b]
        if band["up"] == 0.0 and band["down"] == 0.0 and MAKEUP_DB[i] * e["depth"] + band["gain_db"] > 0.05:
            return True
    return False


def values(entry: dict) -> dict:
    """The knobs as the law plays them: clamped, defaulted, an old entry read once."""
    e = from_legacy(entry) if is_legacy(entry) else dict(entry or {})
    out = {"depth": _clip(e.get("depth", DEPTH[2]), DEPTH),
           "time": _clip(e.get("time", TIME[2]), TIME),
           "in_gain_db": _clip(e.get("in_gain_db", 0.0), GAIN_DB),
           "out_gain_db": _clip(e.get("out_gain_db", 0.0), GAIN_DB),
           "upward_pct": _clip(e.get("upward_pct", UPWARD_PCT[2]), UPWARD_PCT),
           "downward_pct": _clip(e.get("downward_pct", DOWNWARD_PCT[2]), DOWNWARD_PCT)}
    for b in BANDS:
        band = e.get(b) or {}
        out[b] = {"up": _clip(band.get("up", 1.0), SIDE),
                  "down": _clip(band.get("down", 1.0), SIDE),
                  "gain_db": _clip(band.get("gain_db", 0.0), GAIN_DB),
                  "threshold_pct": _clip(band.get("threshold_pct", THRESHOLD_PCT[2]), THRESHOLD_PCT)}
    lo = _clip(e.get("crossover_lo_hz", CROSSOVER_LO_HZ), (30.0, 4000.0, CROSSOVER_LO_HZ))
    hi = _clip(e.get("crossover_hi_hz", CROSSOVER_HI_HZ), (200.0, 16000.0, CROSSOVER_HI_HZ))
    if hi <= lo:   # keep the mid band non-empty (the UI clamps too)
        hi = lo * 1.5
    out["crossover_lo_hz"], out["crossover_hi_hz"] = lo, hi
    return out


def convert_legacy(state: dict) -> tuple[int, int]:
    """Every OTT in a session's state - the chains, undo snapshots, recorded actions - read once as
    this law's knobs. Returns (how many were converted, how many still lift a band that did nothing)."""
    n = lifted = 0

    def walk(node):
        nonlocal n, lifted
        if isinstance(node, dict):
            if is_legacy(node):
                lifted += left_lifted(node)
                node.update(from_legacy(node))
                n += 1
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(state)
    return n, lifted


def band_coeffs(i: int, v: dict) -> dict:
    """Band i's curve at these knobs (dB, dB/dB): the makeup, both thresholds, slopes, knees, caps."""
    b = v[BANDS[i]]
    d = v["depth"]
    shift = 20.0 * math.log10(max(b["threshold_pct"], 1.0) / 100.0)
    su = (v["upward_pct"] / 100.0) * b["up"] * (d ** UP_EXP[i] if d > 0 else 0.0)
    sd = min(v["downward_pct"] / 100.0, 1.0) * b["down"] * (d ** DOWN_EXP[i] if d > 0 else 0.0)
    return {"makeup": MAKEUP_DB[i] * d,
            "up_thr": UP_THR_DB[i] + UP_THR_K[i] * shift, "up_slope": UP_SLOPE[i] * su,
            "up_knee": UP_KNEE_DB[i], "up_cap": UP_CAP_DB[i],
            "down_thr": DOWN_THR_DB[i] + DOWN_THR_K[i] * shift, "down_slope": DOWN_SLOPE[i] * sd,
            "down_knee": DOWN_KNEE_DB[i], "down_cap": DOWN_CAP_DB[i]}


def _softplus(x: np.ndarray, k: float) -> np.ndarray:
    if k <= 1e-6:
        return np.maximum(x, 0.0)
    return k * np.logaddexp(0.0, x / k)


def curve_db(i: int, v: dict, level_db: np.ndarray) -> np.ndarray:
    """Band i's static gain (dB) at detector level `level_db`, the band gain not included."""
    c = band_coeffs(i, v)
    up = np.minimum(c["up_cap"], c["up_slope"] * _softplus(c["up_thr"] - level_db, c["up_knee"]))
    dn = np.minimum(c["down_cap"], c["down_slope"] * _softplus(level_db - c["down_thr"], c["down_knee"]))
    return c["makeup"] + up - dn


def window_samples(i: int, sample_rate: float) -> int:
    return max(1, min(RING_MAX, int(WINDOW_MS[i] * sample_rate / 1000.0)))


def detector_db(band: np.ndarray, i: int, sample_rate: float, time: float) -> np.ndarray:
    """Band i's detector level (dB) per sample. `band` is (channels, frames)."""
    from .dsp import _rising_follower   # dsp imports this module: a late import
    p = np.mean(np.asarray(band, dtype=np.float64) ** 2, axis=0)
    nw = window_samples(i, sample_rate)
    if nw > 1:
        c = np.cumsum(np.concatenate([np.zeros(nw), p]))
        p = np.maximum((c[nw:] - c[:-nw]) / nw, 0.0)
    e1 = _rising_follower(p, sample_rate, ATTACK_MS[i], FLOOR_MS[i])
    rel = RELEASE_K[i] * time
    if rel <= HOLD_ATTACK_MS:
        return 10.0 * np.log10(e1 + 1e-30)
    # the hold: it rises within a sample (an attack of HOLD_ATTACK_MS - a coefficient of exactly 0
    # is a division by zero in the follower's closed form, and a smaller one makes its blocks short)
    e2 = _rising_follower(e1, sample_rate, HOLD_ATTACK_MS, rel)
    return 10.0 * np.log10(e2 + 1e-30)


def band_gain_db(band: np.ndarray, i: int, sample_rate: float, v: dict) -> np.ndarray:
    """Band i's levelling gain (dB) per sample - the curve on the detector, the band gain not included."""
    return curve_db(i, v, detector_db(band, i, sample_rate, v["time"]))


def process(audio: np.ndarray, sample_rate: float, entry: dict, telemetry_pre: list | None = None) -> np.ndarray:
    """(channels, frames) in and out, float64. `telemetry_pre`, when a list, is handed the band sum
    before the band gains (the levelling alone)."""
    from .dsp import _lr4_three_way   # dsp imports this module: a late import
    v = values(entry)
    x = np.asarray(audio, dtype=np.float64)
    if v["in_gain_db"]:
        x = x * 10.0 ** (v["in_gain_db"] / 20.0)
    out = np.zeros_like(x)
    pre = np.zeros_like(x) if telemetry_pre is not None else None
    for i, band in enumerate(_lr4_three_way(x, sample_rate, v["crossover_lo_hz"], v["crossover_hi_hz"])):
        lev = band * (10.0 ** (band_gain_db(band, i, sample_rate, v) / 20.0))[np.newaxis, :]
        if pre is not None:
            pre += lev
        g = v[BANDS[i]]["gain_db"]
        out += lev * 10.0 ** (g / 20.0) if g else lev
    if v["out_gain_db"]:
        out = out * 10.0 ** (v["out_gain_db"] / 20.0)
    if telemetry_pre is not None:
        telemetry_pre.append(pre)
    return out


def makeup_lift_db(entry: dict) -> float:
    """The largest fixed lift a band gets from the makeup at these knobs (dB), its band gain added -
    what the gain wallet charges with the in and out gains (guardrails.level_charge_db)."""
    v = values(entry)
    return max(MAKEUP_DB[i] * v["depth"] + v[b]["gain_db"] for i, b in enumerate(BANDS))


def header_text() -> str:
    def row(name, vals):
        return f"    static constexpr double {name}[3] = {{ " + ", ".join(repr(float(x)) for x in vals) + " };"
    L = [
        "#pragma once",
        "// GENERATED by `python -m mix_engine.ott_law --write-header` from server/mix_engine/ott_law.py -",
        "// OTT's law (that file says how it was measured and what each constant does). Do not edit:",
        "// change the Python table and write this again; test_ott_law.py holds the two equal.",
        "",
        "namespace ottlaw",
        "{",
        "    // per band: low, mid, high",
        row("kMakeupDb", MAKEUP_DB), row("kUpThrDb", UP_THR_DB), row("kUpSlope", UP_SLOPE),
        row("kUpCapDb", UP_CAP_DB), row("kUpKneeDb", UP_KNEE_DB), row("kDownThrDb", DOWN_THR_DB),
        row("kDownSlope", DOWN_SLOPE), row("kDownCapDb", DOWN_CAP_DB), row("kDownKneeDb", DOWN_KNEE_DB),
        row("kUpExp", UP_EXP), row("kDownExp", DOWN_EXP), row("kUpThrK", UP_THR_K), row("kDownThrK", DOWN_THR_K),
        row("kWindowMs", WINDOW_MS), row("kAttackMs", ATTACK_MS), row("kFloorMs", FLOOR_MS),
        row("kReleaseK", RELEASE_K),
        f"    static constexpr int kRingMax = {RING_MAX};",
        f"    static constexpr double kHoldAttackMs = {HOLD_ATTACK_MS!r};",
        f"    static constexpr double kCrossoverLoHz = {CROSSOVER_LO_HZ!r}, kCrossoverHiHz = {CROSSOVER_HI_HZ!r};",
        f"    static constexpr double kLegacyCrossoverLoHz = {LEGACY_CROSSOVERS_HZ[0]!r}, kLegacyCrossoverHiHz = {LEGACY_CROSSOVERS_HZ[1]!r};",
        f"    static constexpr double kTimeMax = {TIME[1]!r}, kUpwardPctMax = {UPWARD_PCT[1]!r}, kDownwardPctMax = {DOWNWARD_PCT[1]!r}, kThresholdPctMax = {THRESHOLD_PCT[1]!r};",
        "}",
        "",
    ]
    return "\n".join(L)


if __name__ == "__main__":
    if "--write-header" in sys.argv:
        HEADER.write_text(header_text())
        print(f"wrote {HEADER}")
