"""The saturation characters (2026-10-26): four curves fitted to a reference saturator.

The curve slots keep their names and indexes - `tanh` 0, `warm` 1, `tube` 2, `hard` 3 - and since
2026-10-26 each is a character measured as a black box on a reference saturator's styles (James's
pick: a gentle saturation, a warm tape, a warm tube and a heavy saturation), on the same signals as
ours: its static curve sample by sample on a 20 Hz sine, its gain, H2 and H3 against level at 1 kHz,
its small-signal response on noise. The record is docs/10-daw.md.

Every character is one curve family, `base(u) = u / (1 + |u|^p)^(1/p)` (unity slope at 0, bounded by
1), driven by `s = 10^((drive_db + s0_db)/20)`, divided back by `s` and lifted by the character's
LEVEL LAW (below):

  - SOFT (`tanh`): p 3.83, s0 +0.94 dB - the reference's gentle curve to 0.11 dB RMS over its range.
  - TAPE (`warm`): the soft curve, then a fixed tone (TONES) - the reference's tape is its gentle
    curve with a tone after it (its harmonics against frequency say the tone is after the shaper).
  - TUBE (`tube`): p 2.83, the input warped by `(1 - e^(-2 beta u)) / (2 beta)` with beta 0.32 (a
    second harmonic that grows with drive, as the reference's does: its H2 against level matches to
    ~2 dB from -40 to -6 dBFS at both drives measured), its positive side clipping at 1.34, then its
    own tone (which also takes the curve's DC out).
  - HEAVY (`hard`): p 1.06, s0 +2.51 dB - the reference's heavy curve to 0.16 dB RMS: dense odd
    harmonics even at low level, quiet detail lifted against the peaks.

THE LEVEL LAW (2026-10-26, measured on the reference's multiband form, James's pick C): the reference
takes the drive back out only up to a ceiling, and adds a fixed gain of its own, so a driven band keeps
its level instead of falling away from the others. Measured on a dry vocal as the reference's level
over ours at the drive that nulls it, it is a static gain of `drive_db` alone (the same 12 dB down):
`level_db(d) = m0 + d - smoothmin(d, ceiling; k)` (LEVEL; 0.03-0.13 dB RMS per character). Without it
a tube / tube / heavy split read 4 dB darker above 4 kHz than the reference; with it every octave
from 250 Hz up is within 0.3 dB. It is a gain AFTER the curve (`shape` returns `y * 10^(level/20) / s`),
so a character is no longer unity: drive 0 is +0.8..+1.0 dB (SOFT / TAPE / TUBE) or +4.0 (HEAVY) at
low level. The guardrail charges it as the device's lift (`entry_lift_db`, guardrails.level_charge_db).

What runs it: `dsp._waveshape` + `dsp._saturate` / `_multiband_saturate` (the export),
`reference_monitor._saturate` (the oracle) and `MonitorChain` + the faces (`satCharacter`,
native/Source/SatCharacters.h - this table, written by `python -m mix_engine.sat_characters
--write-header` and held to it by test_sat_characters.py: change the numbers here, never there).
A tone is linear and runs at the rate its shaper's output is at: the base rate after a single-band
shaper's oversampler comes back down, the oversampled rate inside a multiband band.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

NAMES = ("tanh", "warm", "tube", "hard")      # the stored words, by curve index (models.SAT_CURVES)
FACE_WORDS = ("SOFT", "TAPE", "TUBE", "HEAVY")

# index -> (p, s0_db, beta, positive ceiling, tone index or -1)
CURVES = (
    (3.83, 0.94, 0.0, 1.0, -1),
    (3.83, 0.94, 0.0, 1.0, 0),
    (2.83, 0.0, 0.32, 1.34, 1),
    (1.06, 2.51, 0.0, 1.0, -1),
)
# tone index -> a first-order high-pass (Hz), then three peaks (Hz, dB, Q). Fitted on the
# reference's small-signal response at 48 and 96 kHz below 16.5 kHz, to 0.18 / 0.22 dB RMS.
TONES = (
    (31.2, ((33.8, 4.15, 0.42), (1196.0, 0.98, 1.16), (10880.0, 0.77, 0.54))),   # tape
    (31.3, ((33.6, 3.79, 0.54), (1204.0, 1.15, 2.54), (11210.0, 1.20, 0.98))),   # tube
)
# index -> the level law (m0 dB, ceiling dB, knee k per dB): the reference's level over the bare curve
LEVEL = (
    (0.97, 16.9, 0.498),
    (0.99, 16.8, 0.522),
    (0.79, 11.8, 0.440),
    (3.99, 29.3, 0.124),
)
# how an entry's drives are read (dsp._saturate / _multiband_saturate and the guardrail's charge)
DRIVE_DEFAULT_DB = 3.0
DRIVE_RANGE_DB = (1.0, 12.0)        # the single-band form
BAND_DRIVE_RANGE_DB = (0.0, 24.0)   # a band of the multiband form
WARP_LIMIT = 60.0   # the tube warp's exponent is held here: e^60 is finite, and base() is 1 long before
U_CLAMP = 1.0e6     # the curve's input is held here (pre-merge review 2026-10-26): the warp reaches -1.8e26,
                    # where |u|^p overflows float and the native curve read -0 - a hole on a hot negative peak


def curve_index(curve) -> int:
    """A curve as its index 0..3; the stored word or the index (anything else is 0)."""
    if isinstance(curve, str):
        return NAMES.index(curve) if curve in NAMES else 0
    try:
        return max(0, min(3, int(curve)))
    except (TypeError, ValueError):
        return 0


def _base(u: np.ndarray, p: float) -> np.ndarray:
    return u / (1.0 + np.abs(u) ** p) ** (1.0 / p)


def level_db(drive_db: float, curve=0) -> float:
    """The character's level law at `drive_db`: m0 + d - smoothmin(d, ceiling; k), in dB."""
    m0, ceiling, k = LEVEL[curve_index(curve)]
    d = float(drive_db)
    lo = min(d, ceiling)
    return m0 + d - (lo - np.log1p(np.exp(-k * abs(d - ceiling))) / k)


def entry_lift_db(entry: dict) -> float:
    """The most a saturation entry lifts its target, in dB, before its output trims: the level law at
    each drive the entry runs (its bands in parallel take their max). Reads the entry as the render
    does - the single form's drive held to DRIVE_RANGE_DB, a band's to BAND_DRIVE_RANGE_DB, a missing
    band inheriting the flat drive, a band's curve else the global one."""
    gcurve = entry.get("curve", 0)
    if isinstance(entry.get("low"), dict):
        flat = float(entry.get("drive_db", DRIVE_DEFAULT_DB))
        out = []
        for name in ("low", "mid", "high"):
            b = entry.get(name) or {}
            d = float(np.clip(b.get("drive_db", flat), *BAND_DRIVE_RANGE_DB))
            c = gcurve if b.get("curve") is None else b.get("curve")
            out.append(level_db(d, c) + float(b.get("output_db") or 0.0))
        return max(out)
    d = float(np.clip(entry.get("drive_db", DRIVE_DEFAULT_DB), *DRIVE_RANGE_DB))
    return level_db(d, gcurve)


def shape(x, drive_db: float, curve=0) -> np.ndarray:
    """One character's curve at `drive_db`: memoryless, bounded, its level law applied after it."""
    p, s0, beta, ceil, _ = CURVES[curve_index(curve)]
    s = 10.0 ** ((float(drive_db) + s0) / 20.0)
    u = s * np.asarray(x, dtype=np.float64)
    if beta > 0.0:
        u = -np.expm1(np.minimum(-2.0 * beta * u, WARP_LIMIT)) / (2.0 * beta)
    u = np.clip(u, -U_CLAMP, U_CLAMP)
    if ceil != 1.0:
        y = np.where(u >= 0.0, ceil * _base(u / ceil, p), _base(u, p))
    else:
        y = _base(u, p)
    return y * (10.0 ** (level_db(drive_db, curve) / 20.0) / s)


def tone_sections(curve, sample_rate: float) -> list:
    """The character's tone as (shape, Hz, Q, dB) sections at `sample_rate` - none for SOFT / HEAVY."""
    t = CURVES[curve_index(curve)][4]
    if t < 0:
        return []
    hp, peaks = TONES[t]
    top = 0.49 * float(sample_rate)   # MonitorChain::makeBiquadD holds every frequency under this
    return [("hpf", min(hp, top), 0.707, 0.0)] + [("peak", min(f, top), q, g) for f, g, q in peaks]


def apply_tone(y: np.ndarray, curve, sample_rate: float, axis: int = 0) -> np.ndarray:
    """The character's tone over `y` (linear; nothing for SOFT / HEAVY), from rest."""
    secs = tone_sections(curve, sample_rate)
    if not secs:
        return y
    from scipy.signal import lfilter

    from .reference_monitor import biquad_coeffs
    for kind, f, q, g in secs:
        y = lfilter(*biquad_coeffs(kind, sample_rate, f, q, g), y, axis=axis)
    return y


# ---- the C++ header ---------------------------------------------------------------------------

def header_text() -> str:
    f = lambda v: f"{v:.4f}f"
    rows = "\n".join(f"        {{ {f(p)}, {f(s0)}, {f(b)}, {f(c)}, {t} }},   // {NAMES[i]} = {FACE_WORDS[i]}"
                      for i, (p, s0, b, c, t) in enumerate(CURVES))
    levels = "\n".join(f"        {{ {f(m0)}, {f(c)}, {f(k)} }},   // {NAMES[i]}" for i, (m0, c, k) in enumerate(LEVEL))
    tones = ",\n".join("        { " + f(hp) + ", { " + ", ".join(f"{{ {f(fr)}, {f(g)}, {f(q)} }}" for fr, g, q in pk) + " } }"
                       for hp, pk in TONES)
    return f"""#pragma once
// GENERATED by `python -m mix_engine.sat_characters --write-header` from
// server/mix_engine/sat_characters.py - the saturation characters (that file says how they were
// measured). Do not edit: change the Python table and write this again; test_sat_characters.py
// holds the two equal.

#include <cmath>

namespace satchar
{{
    struct Curve {{ float p, s0Db, beta, posCeil; int tone; }};
    struct Peak {{ float hz, db, q; }};
    struct Tone {{ float hpHz; Peak peaks[3]; }};
    struct Level {{ float m0Db, ceilingDb, k; }};
    static constexpr int kNumCurves = 4, kNumToneSections = 4;
    static constexpr float kWarpLimit = {f(WARP_LIMIT)}, kUClamp = {f(U_CLAMP)};
    // [curve: 0 tanh, 1 warm, 2 tube, 3 hard] - p, s0 (dB), the tube warp's beta, the positive ceiling, tone or -1
    static constexpr Curve kCurve[kNumCurves] = {{
{rows}
    }};
    // [tone: 0 tape, 1 tube] - a first-order high-pass, then three peaks
    static constexpr Tone kTone[2] = {{
{tones}
    }};
    // [curve] - the level law: m0 + d - smoothmin(d, ceiling; k), in dB (sat_characters.level_db)
    static constexpr Level kLevel[kNumCurves] = {{
{levels}
    }};
    static constexpr const char* kFaceWord[kNumCurves] = {{ {", ".join(f'"{w}"' for w in FACE_WORDS)} }};

    /** The drive's scale for a curve: s = 10^((drive + s0) / 20). */
    inline float driveScale (float driveDb, int curve) noexcept
    {{
        return std::pow (10.0f, (driveDb + kCurve[curve].s0Db) / 20.0f);
    }}

    /** The character's level law at this drive, in dB (sat_characters.level_db). */
    inline float levelDb (float driveDb, int curve) noexcept
    {{
        const Level& l = kLevel[curve];
        const float lo = std::fmin (driveDb, l.ceilingDb);
        return l.m0Db + driveDb - (lo - std::log1p (std::exp (-l.k * std::abs (driveDb - l.ceilingDb))) / l.k);
    }}

    /** What the curve's output is multiplied by: the level law over s - the `invS` shape takes. */
    inline float outScale (float driveDb, int curve) noexcept
    {{
        return std::pow (10.0f, levelDb (driveDb, curve) / 20.0f) / driveScale (driveDb, curve);
    }}

    /** One sample of a character's curve (sat_characters.shape), given s and its out scale
        (outScale: the level law over s). */
    inline float shape (float x, float s, float invS, int curve) noexcept
    {{
        const Curve& c = kCurve[curve];
        float u = s * x;
        if (c.beta > 0.0f)
            u = -std::expm1 (std::fmin (-2.0f * c.beta * u, kWarpLimit)) / (2.0f * c.beta);
        u = std::fmax (-kUClamp, std::fmin (kUClamp, u));
        auto base = [p = c.p] (float v) noexcept {{ return v / std::pow (1.0f + std::pow (std::abs (v), p), 1.0f / p); }};
        const float y = (c.posCeil != 1.0f && u >= 0.0f) ? c.posCeil * base (u / c.posCeil) : base (u);
        return y * invS;
    }}
}}
"""


if __name__ == "__main__":
    if "--write-header" in sys.argv:
        out = Path(__file__).resolve().parents[2] / "native/Source/SatCharacters.h"
        out.write_text(header_text())
        print(f"wrote {out}")
