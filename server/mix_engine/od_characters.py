"""The overdrive's characters (2026-10-26): the pedal it always was, and three amps fitted to a reference.

`character` 0 PEDAL is the overdrive as it was (dsp._overdrive's voicing: the low-shelf emphasis, the
biased exponential clipper, the 6.5 kHz roll-off) - a project saved before this plays as it did. 1 SMOOTH,
2 CRUNCHY and 3 POWER are James's pick of a reference saturator's amp styles, measured as a black box
on the same signals as ours: each one's harmonics against level at four frequencies and five drives,
its two-tone intermodulation, its small-signal impulse response. No amp levels over time (a level step
settles inside 10 ms), and none is one filter-curve-filter exactly (the harmonics at different
frequencies do not line up by a level shift: there is more than one stage), so the model is the
nearest one that is ours to run:

    pre (high-pass, peak, high shelf) -> s = 10^((drive_db + s0)/20) -> curve(s x)/s at `oversample` x
    -> post (two high-passes, six peaks, two low-passes) -> a cabinet of a few reflections -> trim

with the curve the saturation's family made lopsided: `base(u) = u/(1+|u|^p)^(1/p)`, the input warped
by `(1 - e^(-2 beta u))/(2 beta)`, the positive side clipping at `ceil` times the negative. The fit
holds the reference's tone within ~2 dB from 250 Hz to 4 kHz on a vocal, its harmonics within 2-4 dB
RMS, its full small-signal response 25 Hz - 20 kHz to 1.4 / 1.5 / 1.6 dB RMS; the waveform nulls only -7 to
-14 dB below 4 kHz - a match of tone and grit, not a copy.
The record is docs/10-daw.md.

`trim_db` makes each amp's small-signal path 0 dB on average over 200 Hz - 4 kHz (computed here, at
48 kHz), so drive 0 is about as loud as the pedal at drive 0. A cabinet tap is a delay in ms with a
gain, read with linear interpolation; the longest is 10 ms.

What runs it: `dsp._overdrive` (the export), `reference_monitor` (through dsp), and `MonitorChain`
(`native/Source/OdCharacters.h`, written by `python -m mix_engine.od_characters --write-header` and held
to this table by test_od_characters.py: change the numbers here, never there).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

NAMES = ("pedal", "smooth", "crunchy", "power")       # the stored words, by character index
FACE_WORDS = ("PEDAL", "SMOOTH", "CRUNCH", "POWER")
DEFAULT = 0                                           # an overdrive nobody set is the pedal
CAB_MAX_MS = 12.0                                     # every tap is under this

# index -> the amp (None for the pedal, which is dsp._overdrive's own voicing)
AMPS = (
    None,
    dict(pre=(135.0, 2.094, 5677.0, 24.0, 0.5327, 4077.0, -14.75),
         curve=(1.984, 0.0, 4.849), s0_db=-8.916,
         post=(("hpf2", 23.27, 2.0, 0.0), ("hpf2", 36.0, 1.831, 0.0), ("peak", 88.54, 0.8607, 17.71), ("peak", 298.8, 3.649, 2.92), ("peak", 626.6, 2.285, 2.32), ("peak", 5843.0, 0.3, 21.63), ("peak", 7199.0, 0.4967, 24.0), ("peak", 11020.0, 0.4495, 23.61), ("lpf2", 1500.0, 0.4, 0.0), ("lpf2", 1842.0, 0.6711, 0.0)),
         taps=((1.063, -0.1653), (2.199, -0.0875), (6.045, -0.1554), (7.942, -0.0893))),
    dict(pre=(136.1, 0.9098, 1559.0, 8.83, 0.2, 1672.0, 7.237),
         curve=(1.468, 0.007757, 1.515), s0_db=9.181,
         post=(("hpf2", 63.07, 1.272, 0.0), ("hpf2", 26.29, 2.0, 0.0), ("peak", 9080.0, 0.6987, 24.0), ("peak", 7231.0, 0.5968, 24.0), ("peak", 6365.0, 0.5648, 24.0), ("peak", 16000.0, 0.3, -18.89), ("peak", 4392.0, 0.3533, 24.0), ("peak", 12720.0, 0.6259, 22.44), ("lpf2", 800.0, 0.4, 0.0), ("lpf2", 800.0, 0.4498, 0.0)),
         taps=((1.062, -0.1759), (2.159, -0.1231), (5.929, -0.1681), (7.934, -0.0586))),
    dict(pre=(102.2, 0.6269, 2664.0, 24.0, 3.101, 2318.0, -19.96),
         curve=(1.875, 0.01447, 0.6251), s0_db=-1.451,
         post=(("hpf2", 26.03, 1.691, 0.0), ("hpf2", 21.72, 2.0, 0.0), ("peak", 45.02, 0.407, 24.0), ("peak", 467.9, 0.3727, 23.17), ("peak", 1219.0, 4.103, 15.77), ("peak", 2597.0, 6.0, -9.052), ("peak", 8227.0, 0.3981, 16.86), ("peak", 4796.0, 0.6575, 21.31), ("lpf2", 3527.0, 2.0, 0.0), ("lpf2", 1501.0, 0.4, 0.0)),
         taps=((1.271, 0.1263), (2.414, -0.1094), (3.202, -0.0935), (4.246, -0.1279), (7.948, 0.0847), (10.05, -0.1969))),
)
WARP_LIMIT = 60.0
U_CLAMP = 1.0e6      # the curve's input is held here: past it the curve is its bound, and |u|^p overflows float


def character_index(c) -> int:
    """A character as its index 0..3; the stored word or the index (anything else is the pedal)."""
    if isinstance(c, str):
        return NAMES.index(c) if c in NAMES else DEFAULT
    try:
        return max(0, min(3, int(c)))
    except (TypeError, ValueError):
        return DEFAULT


def _base(u, p):
    return u / (1.0 + np.abs(u) ** p) ** (1.0 / p)


def amp_curve(x, s: float, c: int) -> np.ndarray:
    """An amp's curve at scale `s` (memoryless, unity slope at 0 bar the warp's own, divided back by s)."""
    p, beta, ceil = AMPS[c]["curve"]
    u = s * np.asarray(x, dtype=np.float64)
    if beta > 0.0:
        u = -np.expm1(np.minimum(-2.0 * beta * u, WARP_LIMIT)) / (2.0 * beta)
    u = np.clip(u, -U_CLAMP, U_CLAMP)
    y = np.where(u >= 0.0, ceil * _base(u / ceil, p), _base(u, p))
    return y / s


def _top(sr: float) -> float:
    return 0.49 * float(sr)   # MonitorChain::makeBiquadD holds every frequency under this


#: The rates an amp's filters have a table at (48 kHz is AMPS itself; od_rate_sections the rest).
TABLE_RATES = (44100, 48000, 88200, 96000, 176400, 192000)


def table_rate(sr: float) -> int:
    """The table rate an amp's filters are built from at `sr`: the nearest by ratio."""
    return min(TABLE_RATES, key=lambda r: abs(np.log(float(sr) / r)))


def _sections_48k(c: int, which: str) -> list:
    if which == "post":
        return [tuple(s) for s in AMPS[c]["post"]]
    hp, hq, pf, pg, pq, sf, sg = AMPS[c]["pre"]
    return [("hpf2", hp, hq, 0.0), ("peak", pf, pq, pg), ("high_shelf", sf, 0.7, sg)]


def _sections(c: int, which: str, sr: float) -> list:
    """An amp's `which` ("pre" / "post") sections at `sr` - its 48 kHz fit, refitted per table
    rate (od_rate_sections, 2026-10-26) so the voicing does not move with the project's rate."""
    from .od_rate_sections import RATE_SECTIONS
    r = table_rate(sr)
    secs = _sections_48k(c, which) if r == 48000 else [tuple(x) for x in RATE_SECTIONS[(c, which, r)]]
    t = _top(sr)
    return [(k, min(f, t), q, g) for k, f, q, g in secs]


def pre_sections(c: int, sr: float) -> list:
    return _sections(c, "pre", sr)


def post_sections(c: int, sr: float) -> list:
    return _sections(c, "post", sr)


#: THE LEVEL LAW (2026-10-26, the pre-merge review): the reference's amps hold their level as the
#: drive rises; ours divided the whole drive back out (curve(s x)/s) and fell 3-7 dB under the
#: reference at drive 0, 4-9 at 12 and 13-26 at 36, measured at the fit's drive mapping on a full
#: mix and a dry vocal at -1 and -13 dBFS (the same within ~1 dB: a static gain of the drive alone).
#: The saturation's form fits it: level_db(d) = m0 + d - smoothmin(d, ceiling; k), 0.4-0.6 dB RMS.
#: index -> (m0 dB, ceiling dB, k)
LEVEL = (None, (3.30, 30.3, 0.109), (6.50, 17.2, 0.157), (4.03, 22.9, 0.192))


def level_db(drive_db: float, c: int) -> float:
    """An amp's level law at `drive_db` (0 for the pedal), in dB, applied after its curve."""
    if c <= 0:
        return 0.0
    m0, ceiling, k = LEVEL[c]
    d = float(drive_db)
    lo = min(d, ceiling)
    return m0 + d - (lo - np.log1p(np.exp(-k * abs(d - ceiling))) / k)


def fit_rate_sections(rates=(44100, 88200, 96000, 176400, 192000), floor_db: float = 45.0) -> dict:
    """Refit every amp's pre and post sections at each rate to its 48 kHz response (what wrote
    od_rate_sections.py): magnitude in dB, 20 Hz to min(20 kHz, 0.45 sr), floored `floor_db`
    under the peak, from the 48 kHz numbers prewarped to the new rate."""
    from scipy.optimize import least_squares

    def resp_db(secs, sr, f):
        return 20 * np.log10(np.abs(_cascade_response(secs, sr, f)) + 1e-12)

    out = {}
    for c in (1, 2, 3):
        for which in ("pre", "post"):
            secs48 = _sections_48k(c, which)
            for R in rates:
                f = np.geomspace(20.0, min(20000.0, 0.45 * R), 300)
                want0 = resp_db(secs48, 48000, f)
                top = want0.max()
                want = np.maximum(want0, top - floor_db)
                shapes = [x[0] for x in secs48]
                x0, lo, hi = [], [], []
                for _, fr, q, g in secs48:
                    fa = 48000 / np.pi * np.tan(np.pi * min(fr, 23900) / 48000)
                    fd = R / np.pi * np.arctan(np.pi * fa / R)
                    x0 += [np.log(min(fd, 0.48 * R)), np.log(q), g]
                    lo += [np.log(5.0), np.log(0.05), -40.0]
                    hi += [np.log(0.49 * R), np.log(30.0), 40.0]

                def build(x, shapes=shapes):
                    return [(shapes[i], float(np.exp(x[3 * i])), float(np.exp(x[3 * i + 1])),
                             float(x[3 * i + 2]) if shapes[i] in ("peak", "high_shelf") else 0.0)
                            for i in range(len(shapes))]

                def err(x, R=R, f=f, top=top, want=want, build=build):
                    return np.maximum(resp_db(build(x), R, f), top - floor_db) - want

                r = least_squares(err, np.clip(x0, lo, hi), bounds=(lo, hi), x_scale="jac", max_nfev=4000)
                out[(c, which, R)] = tuple(build(r.x))
    return out


def _cascade_response(secs, sr, freqs):
    from scipy.signal import freqz

    from .reference_monitor import biquad_coeffs
    h = np.ones(len(freqs), complex)
    for k, f, q, g in secs:
        b, a = biquad_coeffs(k, sr, f, q, g)
        _, hh = freqz(b, a, worN=np.asarray(freqs, float), fs=sr)
        h *= hh
    return h


def _taps_response(c, freqs):
    w = 2j * np.pi * np.asarray(freqs, float) / 1000.0
    return 1.0 + sum(g * np.exp(-w * ms) for ms, g in AMPS[c]["taps"])


def trim_db(c: int) -> float:
    """The level that makes the amp's small-signal path 0 dB on average over 200 Hz - 4 kHz (at 48 kHz)."""
    f = np.geomspace(200.0, 4000.0, 64)
    h = _cascade_response(pre_sections(c, 48000) + post_sections(c, 48000), 48000, f) * _taps_response(c, f)
    return float(-10.0 * np.log10(np.mean(np.abs(h) ** 2)))


TRIMS_DB = (0.0, -15.281, -10.892, -22.472)   # trim_db(c), rounded (test_od_characters.py recomputes them)


def cab(y: np.ndarray, c: int, sr: float, axis: int = 0) -> np.ndarray:
    """The amp's reflections over `y` (the direct sound plus each tap, linear interpolation), from rest."""
    y = np.moveaxis(np.asarray(y, dtype=np.float64), axis, 0)
    out = y.copy()
    n = y.shape[0]
    for ms, g in AMPS[c]["taps"]:
        d = ms * float(sr) / 1000.0
        i = int(np.floor(d))
        fr = d - i
        if i < n:
            out[i:] += g * (1.0 - fr) * y[: n - i]
        if i + 1 < n:
            out[i + 1:] += g * fr * y[: n - i - 1]
    return np.moveaxis(out, 0, axis)


# ---- the C++ header ---------------------------------------------------------------------------

_SHAPE = {"hpf2": "hpf2", "lpf2": "lpf2", "peak": "peak", "high_shelf": "high_shelf"}


def header_text() -> str:
    def f(v):
        t = f"{float(v):.6g}"
        return (t if ("." in t or "e" in t) else t + ".0") + "f"
    amps = []
    for c in (1, 2, 3):
        A = AMPS[c]
        hp, hq, pf, pg, pq, sf, sg = A["pre"]
        p, beta, ceil = A["curve"]
        post = ", ".join(f"{{ Shape::{_SHAPE[k]}, {f(fr)}, {f(q)}, {f(g)} }}" for k, fr, q, g in A["post"])
        taps = A["taps"] + ((0.0, 0.0),) * (6 - len(A["taps"]))
        tp = ", ".join(f"{{ {f(ms)}, {f(g)} }}" for ms, g in taps)
        amps.append(f"""        {{   // {c} = {NAMES[c]} ({FACE_WORDS[c]})
            {{ {f(hp)}, {f(hq)}, {f(pf)}, {f(pg)}, {f(pq)}, {f(sf)}, {f(sg)} }},
            {f(p)}, {f(beta)}, {f(ceil)}, {f(A["s0_db"])}, {f(TRIMS_DB[c])},
            {{ {post} }},
            {len(A["taps"])}, {{ {tp} }},
            {f(LEVEL[c][0])}, {f(LEVEL[c][1])}, {f(LEVEL[c][2])} }}""")
    body = ",\n".join(amps)
    def secs_text(secs):
        return ", ".join(f"{{ Shape::{_SHAPE[k]}, {f(fr)}, {f(q)}, {f(g)} }}" for k, fr, q, g in secs)
    rate_rows = []
    for c in (1, 2, 3):
        sets = ",\n".join(f"            {{ {f(r)}, {{ {secs_text(pre_sections(c, r))} }},\n              {{ {secs_text(post_sections(c, r))} }} }}"
                          for r in TABLE_RATES)
        rate_rows.append(f"        {{   // {NAMES[c]}\n{sets} }}")
    rates_body = ",\n".join(rate_rows)
    return f"""#pragma once
// GENERATED by `python -m mix_engine.od_characters --write-header` from
// server/mix_engine/od_characters.py - the overdrive's amp characters (that file says how they were
// measured). Do not edit: change the Python table and write this again; test_od_characters.py
// holds the two equal.

#include <cmath>

namespace odchar
{{
    enum class Shape {{ hpf2, lpf2, peak, high_shelf }};
    struct Section {{ Shape shape; float hz, q, db; }};
    struct Pre {{ float hpHz, hpQ, pkHz, pkDb, pkQ, shelfHz, shelfDb; }};
    struct Tap {{ float ms, gain; }};
    struct Amp {{ Pre pre; float p, beta, posCeil, s0Db, trimDb; Section post[10]; int numTaps; Tap taps[6];
                 float levelM0, levelCeil, levelK; }};
    static constexpr int kNumCharacters = 4, kNumPostSections = 10, kMaxTaps = 6, kNumTableRates = {len(TABLE_RATES)};
    // the amps' filters per table rate (od_characters.pre_sections / post_sections): [amp][rate]
    struct RateSet {{ float rate; Section pre[3]; Section post[10]; }};
    static constexpr float kCabMaxMs = {f(CAB_MAX_MS)}, kWarpLimit = {f(WARP_LIMIT)}, kUClamp = {f(U_CLAMP)};
    static constexpr const char* kFaceWord[kNumCharacters] = {{ {", ".join(f'"{w}"' for w in FACE_WORDS)} }};
    // [amp: character - 1] (character 0 is the pedal, MonitorChain's own voicing)
    static constexpr Amp kAmp[3] = {{
{body}
    }};

    static constexpr RateSet kAmpRate[3][kNumTableRates] = {{
{rates_body}
    }};

    /** The table an amp's filters are built from at `sr` (od_characters.table_rate: nearest by ratio). */
    inline const RateSet& rateSet (int amp, double sr) noexcept
    {{
        int best = 0;
        for (int r = 1; r < kNumTableRates; ++r)
            if (std::abs (std::log (sr / (double) kAmpRate[amp][r].rate)) < std::abs (std::log (sr / (double) kAmpRate[amp][best].rate)))
                best = r;
        return kAmpRate[amp][best];
    }}

    /** An amp's level law at `driveDb` (od_characters.level_db), in dB. */
    inline float levelDb (float driveDb, const Amp& a) noexcept
    {{
        const float lo = std::fmin (driveDb, a.levelCeil);
        return a.levelM0 + driveDb - (lo - std::log1p (std::exp (-a.levelK * std::abs (driveDb - a.levelCeil))) / a.levelK);
    }}

    /** What the curve's output is scaled by: the level law over s (od_characters: curve / s x level). */
    inline float outScale (float driveDb, const Amp& a, float s) noexcept {{ return std::pow (10.0f, levelDb (driveDb, a) / 20.0f) / s; }}

    /** The drive's scale for an amp: s = 10^((drive + s0) / 20). */
    inline float driveScale (float driveDb, const Amp& a) noexcept {{ return std::pow (10.0f, (driveDb + a.s0Db) / 20.0f); }}

    /** One sample of an amp's curve (od_characters.amp_curve), given s and 1/s. */
    inline float curve (float x, float s, float invS, const Amp& a) noexcept
    {{
        float u = s * x;
        if (a.beta > 0.0f)
            u = -std::expm1 (std::fmin (-2.0f * a.beta * u, kWarpLimit)) / (2.0f * a.beta);
        u = std::fmax (-kUClamp, std::fmin (kUClamp, u));   // |u|^p overflows float past ~1e13 (2026-10-26)
        auto base = [p = a.p] (float v) noexcept {{ return v / std::pow (1.0f + std::pow (std::abs (v), p), 1.0f / p); }};
        const float y = u >= 0.0f ? a.posCeil * base (u / a.posCeil) : base (u);
        return y * invS;
    }}
}}
"""


if __name__ == "__main__":
    if "--fit-rates" in sys.argv:
        fits = fit_rate_sections()
        rows = "\n".join(f"    {k!r}: {tuple((a, float(f'{b:.6g}'), float(f'{q:.6g}'), float(f'{g:.6g}')) for a, b, q, g in v)!r},"
                         for k, v in sorted(fits.items()))
        print("RATE_SECTIONS = {\n" + rows + "\n}")
    if "--write-header" in sys.argv:
        out = Path(__file__).resolve().parents[2] / "native/Source/OdCharacters.h"
        out.write_text(header_text())
        print(f"wrote {out}")
