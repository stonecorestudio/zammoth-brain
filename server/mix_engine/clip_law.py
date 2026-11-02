"""The clipper's law (2026-10-28): a reference clipper, measured as a black box.

James's pick after hearing A (ours) / REF / B (this law) on a released pop master at three
settings: B, with the reference's knob set. The clipper has no threshold, hardness or output trim
any more: GAIN (0..12 dB, before the clip), CLIP (the ceiling, 0..100 % of full scale), SOFTNESS
(0..100 %) and OVERSAMPLE (off / on).

Measured on a slow sine swept past full scale (every knob at many settings), on a hard-clipped tone
(its aliases), on an impulse (its oversampling filter) and on the song:

    gain      a plain gain before the shaper; nothing after it makes the level back up
    shape     memoryless, each channel on its own. With c = CLIP and s = SOFTNESS:
                  |x| <= c(1 - s)            y = x                       (untouched)
                  c(1 - s) < |x| < c(1 + s)  y = x - (|x| - c(1 - s))^2 / (4 s c), signed
                  |x| >= c(1 + s)            y = c, signed
              a QUADRATIC knee in linear amplitude, centred on the ceiling, 2 s c wide, joining
              both straight parts with their slopes. On the song this nulls 108-145 dB under the
              reference; the tanh blend it replaces nulled 15-22 dB at soft settings.
    oversample  on = the shaper at 2x, then a hard clamp at 0 dBFS after coming back down -
              whatever CLIP is (below full scale the resampling filter's ringing passes). The
              reference's own 2x filter is short and gentle (-2 dB at 20 kHz at 48 kHz, the 5th
              harmonic of a 5 kHz tone folds back at -24 dB); ours is the engine's own resampler,
              flatter and with far less aliasing - the one part of the reference not copied.

A clipper saved before (`threshold_db`, `hardness`, `output_db`, `oversample` 1/2/4/8) is read once
as this law by `from_legacy`: CLIP at its threshold, SOFTNESS = 1 - hardness, its oversampling on
when it had any, and a positive output trim moved in front as GAIN with the ceiling raised by the
same amount (when that ceiling still fits under full scale) - the same output wherever the old
knee was hard. Any other trim has nowhere to go (there is no output knob) and is dropped: that
clipper plays |trim| louder or quieter than it did.

`native/Source/ClipLaw.h` is the same law in C++, held to this file by test_clip_law.py.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

GAIN_MAX_DB = 12.0
CLIP_MAX_PCT = 100.0
OVERSAMPLE_ON = 2            # the reference's oversampling is 2x
DEFAULTS = {"gain_db": 0.0, "clip_pct": 100.0, "softness": 0.0, "oversample": 1}
LEGACY_FIELDS = ("threshold_db", "hardness", "output_db")


def is_legacy(entry: dict) -> bool:
    """A clipper saved before this law: it names an old knob and none of the new ones."""
    return (any(k in entry for k in LEGACY_FIELDS)
            and not any(k in entry for k in ("gain_db", "clip_pct", "softness")))


def from_legacy(entry: dict) -> dict:
    """An old clipper's knobs as this law's (see the module text)."""
    thr = min(max(float(entry.get("threshold_db", -3.0)), -12.0), 0.0)
    hard = min(max(float(entry.get("hardness", 0.5)), 0.0), 1.0)
    out = min(max(float(entry.get("output_db", 0.0) or 0.0), -12.0), 12.0)
    t = 10.0 ** (thr / 20.0)
    gain = 0.0
    if out > 0.0 and t * 10.0 ** (out / 20.0) <= 1.0:
        gain = out
        t *= 10.0 ** (out / 20.0)
    os_old = int(entry.get("oversample", 1) or 1)
    return {"gain_db": gain, "clip_pct": 100.0 * t,
            "softness": 1.0 - hard, "oversample": OVERSAMPLE_ON if os_old > 1 else 1}


def values(entry: dict) -> dict:
    """The four knobs of a stored entry, held to their ranges - an old one read through
    `from_legacy`. Every reader of a clipper entry (the render, the monitor resolver, the reference
    monitor, the guardrails) goes through here."""
    v = from_legacy(entry) if is_legacy(entry) else entry
    os_n = int(v.get("oversample", 1) or 1)
    return {"gain_db": min(max(float(v.get("gain_db", 0.0) or 0.0), 0.0), GAIN_MAX_DB),
            "clip_pct": min(max(float(v.get("clip_pct", CLIP_MAX_PCT)), 0.0), CLIP_MAX_PCT),
            "softness": min(max(float(v.get("softness", 0.0) or 0.0), 0.0), 1.0),
            "oversample": OVERSAMPLE_ON if os_n > 1 else 1}


def shape(x: np.ndarray, c: float, s: float) -> np.ndarray:
    """The knee, elementwise (float64 in, float64 out)."""
    x = np.asarray(x, dtype=np.float64)
    a = np.abs(x)
    lo, hi = c * (1.0 - s), c * (1.0 + s)
    if s <= 0.0:
        y = np.minimum(a, c)
    else:
        y = np.where(a <= lo, a, np.where(a >= hi, c, a - (a - lo) ** 2 / (4.0 * s * c)))
    return np.copysign(y, x)


def shape_one(x: float, c: float, s: float) -> float:
    """The knee on one sample (the face's curve, the tests)."""
    return float(shape(np.array([x]), c, s)[0])


def apply(audio: np.ndarray, entry: dict) -> np.ndarray:
    """The clipper on (frames, channels): gain, the knee at 1x or 2x, and at 2x the 0 dBFS clamp."""
    v = values(entry)
    g = 10.0 ** (v["gain_db"] / 20.0)
    c = v["clip_pct"] / 100.0
    x = np.asarray(audio, dtype=np.float64) * g
    if v["oversample"] > 1:
        from scipy.signal import resample_poly
        n = x.shape[0]
        y = shape(resample_poly(x, OVERSAMPLE_ON, 1, axis=0), c, v["softness"])
        return np.clip(resample_poly(y, 1, OVERSAMPLE_ON, axis=0)[:n], -1.0, 1.0)
    return shape(x, c, v["softness"])


def lift_db(entry: dict) -> float:
    """What the clipper adds in level (the gain wallet's charge): its GAIN - what is under the knee
    comes out that much louder. The ceiling only ever takes away."""
    return values(entry)["gain_db"]


def crush_db(entry_or_values: Any) -> float:
    """How far under the input's full scale the ceiling sits, in dB (the agent's floor reads this):
    CLIP in dB minus GAIN. -inf at CLIP 0."""
    v = values(entry_or_values) if isinstance(entry_or_values, dict) else entry_or_values
    c = v["clip_pct"] / 100.0
    return (20.0 * math.log10(c) if c > 0.0 else float("-inf")) - v["gain_db"]


def dropped_trim_db(entry: dict) -> float:
    """What of an old clipper's output trim has no knob in this law (`from_legacy`): a negative
    trim, or a positive one that would carry the ceiling past full scale."""
    out = min(max(float(entry.get("output_db", 0.0) or 0.0), -12.0), 12.0)
    return out - from_legacy(entry)["gain_db"]


#: The keys under which a list is a device CHAIN (a trim kept as a Utility is inserted there; a
#: list of recorded actions is a record and keeps no device of its own).
CHAIN_KEYS = ("chain", "master_chain", "entries", "return_entries")


def convert_legacy(obj, _chain: bool = False) -> int:
    """Walk anything read from an old file and convert every clipper in it in place (a dict of type
    "clip" holding the old knobs): the session's chains, its undo snapshots, its recorded actions.
    A trim the law has no knob for is KEPT, in a chain: a Utility right after the clipper carries it,
    so the project plays as it did (the pre-merge review of 2026-10-28: a -3 dB trim played 3 dB
    hotter into the limiter). Returns how many it converted (session STATE_VERSION 4)."""
    n = 0
    if isinstance(obj, dict):
        if obj.get("type") == "clip" and is_legacy(obj):
            conv = from_legacy(obj)
            for k in LEGACY_FIELDS:
                obj.pop(k, None)
            obj.update(conv)
            n += 1
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                n += convert_legacy(v, _chain=k in CHAIN_KEYS)
    elif isinstance(obj, list):
        i = 0
        while i < len(obj):
            v = obj[i]
            trim = (dropped_trim_db(v) if _chain and isinstance(v, dict) and v.get("type") == "clip"
                    and is_legacy(v) else 0.0)
            n += convert_legacy(v)
            if abs(trim) > 1e-9:
                obj.insert(i + 1, {"type": "utility", "gain_db": round(trim, 4),
                                   "enabled": bool(v.get("enabled", True))})
                i += 1
            i += 1
    return n


def trims_kept(obj, _chain: bool = False) -> int:
    """How many old clippers in `obj` have a trim `convert_legacy` keeps as a Utility (asked before
    converting: the open's notice counts them)."""
    n = 0
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                n += trims_kept(v, _chain=k in CHAIN_KEYS)
    elif isinstance(obj, list):
        for v in obj:
            if _chain and isinstance(v, dict) and v.get("type") == "clip" and is_legacy(v) \
                    and abs(dropped_trim_db(v)) > 1e-9:
                n += 1
            n += trims_kept(v)
    return n
