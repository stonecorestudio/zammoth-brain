"""The gate's characters (2026-10-27): four laws fitted to a reference gate, measured as a black box.

James's pick: all four of a reference gate's styles, as CHARACTERS of our gate - 0 CLASSIC (the
default), 1 CLEAN, 2 VOCAL, 3 GUITAR. Each was measured on the same signals as ours: its static
curve rising and falling at five ratios, its open and close levels, its detector's response across
frequency, its gain against time at five attacks, five releases, two ranges and two holds, a low
tone just over the threshold - and then a full song (a released pop master, 21 s) at four settings.
The law below is ours to run; its curve offsets are the reference's, read off its static curve, and
the fit holds its gain on the part of the song it was not fitted on to 0.09-0.24 dB (CLASSIC),
0.14-0.24 (CLEAN), 0.6-1.4 (VOCAL) and 0.2-1.05 (GUITAR; 3.45 at a long release with a hold), where
the gate before this was 1.2-4.4 dB from every one of them. The record is docs/10-daw.md.

One law, every stage a fixed-coefficient branching one-pole (so Python runs it at compiled speed
through `dsp._gr_ballistics_db`, and `MonitorChain::applyGate` per sample):

    detector  the linked |x| after the user's sidechain filters and the character's own EQ (GUITAR
              listens with the lows down 3 dB, VOCAL with the top up 26 dB - consonants open it),
              a peak follower (instant up, released over det_ms), in dB; then (where a character has
              one) a slow follower of that level in dB, attacking over det_attack_k x attack +
              det_attack_ms and releasing over det_release_ms
    trigger   open at the threshold, shut under threshold - (hyst_db + the user's hysteresis_db)
    target    open: open_db; shut: (ratio - 1)(level - threshold) + offset_db, no higher than
              open_db; floored at range_db. The OFFSET is the reference's: a CLASSIC gate at 4:1
              starts cutting 8/3 dB under its threshold, not at it.
    hold      the largest target of the last hold_k x hold_ms
    smooth    a one-pole toward the target - attack_k x attack + attack_ms up, release_k x
              release^release_p down - in dB, or (LIN) on the linear gain
    fall      a one-pole that follows it up at once and down over fall_ms + fall_k x release (in
              the same domain)
    gain      that, no higher than 0 dB and no lower than range_db

A target above 0 dB (open_db) is how the reference's attack and its delay before a release look:
the state runs up past 0 dB and has to come back to it before the gain moves.

Everything starts at rest: shut, at range_db. A ratio of 1 is off in every character.
The native header (`native/Source/GateCharacters.h`) is written by
`python -m mix_engine.gate_characters --write-header` and held to this table by test_gate_characters.py.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

NAMES = ("classic", "clean", "vocal", "guitar")       # the stored words, by character index
FACE_WORDS = ("CLASSIC", "CLEAN", "VOCAL", "GUITAR")
DEFAULT = 0

#: index -> the law's constants (see the module text). sc: the detector's own EQ, (shape, Hz, q, dB).
CHARACTERS = (
    dict(lin=0, offset_db=8.0, hyst_db=7.266, open_db=4.599, attack_k=0.3849, attack_ms=0.0,
         det_ms=2.906, fall_ms=51.06, fall_k=0.09366, release_k=0.6319, release_p=0.8985, hold_k=0.5219,
         det_attack_k=0.0, det_attack_ms=0.0, det_release_ms=0.0, sc=()),
    dict(lin=1, offset_db=4.36, hyst_db=11.75, open_db=0.7737, attack_k=0.3101, attack_ms=0.3644,
         det_ms=1.939, fall_ms=83.64, fall_k=0.2023, release_k=1.195, release_p=0.4507, hold_k=1.435,
         det_attack_k=0.0, det_attack_ms=0.0, det_release_ms=0.0, sc=()),
    dict(lin=1, offset_db=0.31, hyst_db=5.936, open_db=0.0, attack_k=0.04135, attack_ms=54.68,
         det_ms=5.833, fall_ms=61.52, fall_k=0.2119, release_k=0.02132, release_p=0.3, hold_k=2.325,
         det_attack_k=0.0, det_attack_ms=1.245, det_release_ms=14.75,
         sc=(("high_shelf", 3631.17, 0.59, 12.92), ("high_shelf", 4124.35, 0.94, 12.92))),
    dict(lin=0, offset_db=0.1, hyst_db=1.074, open_db=17.81, attack_k=0.8922, attack_ms=0.9958,
         det_ms=0.4357, fall_ms=51.71, fall_k=0.2399, release_k=3.204, release_p=0.5035, hold_k=0.7756,
         det_attack_k=0.3548, det_attack_ms=0.1005, det_release_ms=115.7, sc=(("low_shelf", 169.68, 0.66, -2.97),)),
)
NUM_SC = 2            # the most EQ sections a character's detector has
FLOOR_DB = -120.0     # the detector's level is never read under this


def character_index(c) -> int:
    """A stored gate entry's character (absent = CLASSIC), held to the table."""
    if isinstance(c, str):
        c = c.strip().lower()
        return NAMES.index(c) if c in NAMES else DEFAULT
    try:
        return min(len(NAMES) - 1, max(0, int(round(float(c)))))
    except (TypeError, ValueError):
        return DEFAULT


def detector_eq(det: np.ndarray, sample_rate: float, c: int, axis: int) -> np.ndarray:
    """The character's own detector EQ (its sections, in order) - the audio path is untouched."""
    from scipy.signal import lfilter
    from .reference_monitor import biquad_coeffs
    for shape, hz, q, db in CHARACTERS[c]["sc"]:
        b, a = biquad_coeffs(shape, sample_rate, min(hz, 0.45 * sample_rate), q, db)
        det = lfilter(b, a, det, axis=axis)
    return det


def _branching(y: np.ndarray, start: float, sample_rate: float, up_ms: float, down_ms: float) -> np.ndarray:
    """A one-pole that moves toward `y` over up_ms when y is above it and over down_ms when below,
    from `start` (0 ms = at once) - `dsp._gr_ballistics_db` mirrored: it attacks DOWNWARD, so it is
    fed -y, shifted under 0."""
    from .dsp import _gr_ballistics_db
    k = 1.0 - float(min(np.min(y), start))   # every fed value is -1 or lower: the follower's law
    lead_ms = max(up_ms, 0.01)
    lead = int(math.ceil(18.0 * sample_rate * lead_ms / 1000.0)) + 16
    fed = np.concatenate([np.full(lead, -start - k), -np.asarray(y, dtype=np.float64) - k])
    z = _gr_ballistics_db(fed, sample_rate, up_ms, down_ms)[lead:]
    return -(z + k)


def gain_db(peak: np.ndarray, sample_rate: float, threshold_db: float, ratio: float,
            attack_ms: float, release_ms: float, hold_ms: float = 0.0, range_db: float = -80.0,
            hysteresis_db: float = 0.0, character: int = DEFAULT) -> np.ndarray:
    """The gate's gain (dB, range_db..0) for a detector `peak` - the linked |x| after the user's
    sidechain filters and `detector_eq`. MonitorChain::applyGate is this, per sample."""
    peak = np.asarray(peak, dtype=np.float64)
    n = peak.shape[0]
    floor = min(0.0, float(range_db))
    if n == 0 or ratio <= 1.0 or floor >= 0.0:
        return np.zeros(n)
    C = CHARACTERS[character]
    sr = float(sample_rate)
    # the peak follower: det = max(|x|, det * a), in closed form (from rest)
    lx = 20.0 * np.log10(np.maximum(peak, 1.0e-12))
    step = 20.0 * math.log10(math.exp(-1000.0 / (sr * C["det_ms"])))
    j = np.arange(n, dtype=np.float64)
    level = np.maximum(j * step + np.maximum.accumulate(lx - j * step), FLOOR_DB)
    da, dr = C["det_attack_k"] * float(attack_ms) + C["det_attack_ms"], C["det_release_ms"]
    if da > 0.0 or dr > 0.0:
        level = _branching(level, FLOOR_DB, sr, da, dr)
    thr = float(threshold_db)
    hyst = C["hyst_db"] + max(0.0, float(hysteresis_db))
    event = np.where(level >= thr, 1, np.where(level < thr - hyst, 0, -1))
    last = np.maximum.accumulate(np.where(event >= 0, j.astype(np.int64), -1))
    is_open = np.where(last >= 0, event[np.maximum(last, 0)] == 1, False)   # starts shut
    cap = C["open_db"]
    target = np.where(is_open, cap, np.minimum((float(ratio) - 1.0) * (level - thr) + C["offset_db"], cap))
    target = np.maximum(target, floor)
    hold = int(sr * float(hold_ms) * C["hold_k"] / 1000.0)
    if hold > 0:
        from scipy.ndimage import maximum_filter1d
        padded = np.concatenate([np.full(hold, floor), target])
        target = maximum_filter1d(padded, size=hold + 1, origin=hold // 2)[hold:]   # [i - hold, i]
    up = C["attack_k"] * float(attack_ms) + C["attack_ms"]
    down = C["release_k"] * float(release_ms) ** C["release_p"]
    fall = C["fall_ms"] + C["fall_k"] * float(release_ms)
    if C["lin"]:
        s = _branching(10.0 ** (target / 20.0), 10.0 ** (floor / 20.0), sr, up, down)
        o = _branching(s, 10.0 ** (floor / 20.0), sr, 0.0, fall)
        o = 20.0 * np.log10(np.maximum(o, 1.0e-9))
    else:
        s = _branching(target, floor, sr, up, down)
        o = _branching(s, floor, sr, 0.0, fall)
    return np.clip(o, floor, 0.0)


# ------------------------------------------------------------------------------------- the header
_SHAPE = {"low_shelf": "low_shelf", "high_shelf": "high_shelf"}


def header_text() -> str:
    def f(v):
        t = f"{float(v):.7g}"
        return (t if ("." in t or "e" in t) else t + ".0") + "f"
    rows = []
    for i, C in enumerate(CHARACTERS):
        sc = list(C["sc"]) + [("none", 0.0, 0.0, 0.0)] * (NUM_SC - len(C["sc"]))
        sct = ", ".join(f"{{ Shape::{s}, {f(hz)}, {f(q)}, {f(db)} }}" for s, hz, q, db in sc)
        rows.append(
            f"        {{   // {i} = {NAMES[i]} ({FACE_WORDS[i]})\n"
            f"            {'true' if C['lin'] else 'false'}, {f(C['offset_db'])}, {f(C['hyst_db'])}, {f(C['open_db'])},\n"
            f"            {f(C['attack_k'])}, {f(C['attack_ms'])}, {f(C['det_ms'])}, {f(C['fall_ms'])}, {f(C['fall_k'])},\n"
            f"            {f(C['release_k'])}, {f(C['release_p'])}, {f(C['hold_k'])}, {f(C['det_attack_k'])}, {f(C['det_attack_ms'])}, {f(C['det_release_ms'])},\n"
            f"            {len(C['sc'])}, {{ {sct} }} }}")
    body = ",\n".join(rows)
    return f"""#pragma once
// GENERATED by `python -m mix_engine.gate_characters --write-header` from
// server/mix_engine/gate_characters.py - the gate's characters (that file says how they were
// measured and what each constant does). Do not edit: change the Python table and write this
// again; test_gate_characters.py holds the two equal.

namespace gatechar
{{
    enum class Shape {{ none, low_shelf, high_shelf }};
    struct Section {{ Shape shape; float hz, q, db; }};
    struct Character {{ bool lin; float offsetDb, hystDb, openDb, attackK, attackMs, detMs, fallMs, fallK,
                       releaseK, releaseP, holdK, detAttackK, detAttackMs, detReleaseMs; int numSc; Section sc[{NUM_SC}]; }};
    static constexpr int kNumCharacters = {len(CHARACTERS)}, kNumSc = {NUM_SC};
    static constexpr float kFloorDb = {f(FLOOR_DB)};
    static constexpr const char* kFaceWord[kNumCharacters] = {{ {", ".join(f'"{w}"' for w in FACE_WORDS)} }};
    static constexpr Character kCharacters[kNumCharacters] = {{
{body}
    }};
}}
"""


if __name__ == "__main__":
    if "--write-header" in sys.argv:
        out = Path(__file__).resolve().parents[2] / "native/Source/GateCharacters.h"
        out.write_text(header_text())
        print(f"wrote {out}")
