"""The de-esser's law (2026-10-27): a reference de-esser, measured as a black box and fitted.

James's pick after hearing A (ours) / REF / B (this law) on a released pop master, both modes, split
and wide band: "B sounds right in both modes. Port it". The reference has no ratio, attack or
release - so neither has this device any more (his pick of the knob set, the same day): THRESHOLD,
RANGE, FREQ (the band's bottom), TOP (its top), MODE (allround / vocal), WIDE, and a fixed 12 ms
lookahead. A project saved before opens its de-essers as ALLROUND with their range held to 24 dB.

Measured on tones (its static curve at many thresholds and ranges, its band at ten settings, its
timing on bursts, its split-band shelf at five depths with its phase, its vocal-mode decision against
a vowel band) and fitted on a released song (21 s at five settings) and, for VOCAL, a dry vocal too:

    detector  the MID ((L + R) / 2) through a Butterworth 4th-order high-pass at FREQ and (TOP on) a
              Butterworth 4th-order low-pass at TOP, plus a bell at the band's centre that grows as
              the band narrows (BAND_BELL: none at two octaves, +6.4 dB at one); the largest |x| of
              the last HOLD_MS, in dB
    law       reduction = RANGE x (level - THRESHOLD) / (0 - THRESHOLD), 0..RANGE: it starts at the
              threshold and reaches the full range when the detector reads 0 dBFS. No ratio: the
              slope is the range over the threshold.
    VOCAL     the cut is let through only while its own sibilance detector (the high-pass at FREQ,
              a low-pass at SIB_TOP_K x FREQ, held SIB_HOLD_MS, smoothed SIB_SMOOTH_MS) is over
              THRESHOLD + SIB_GATE_DB AND beats a vowel band (VOWEL_HP_HZ .. VOWEL_TOP_K x FREQ, a
              dB average of |x| over VOWEL_SMOOTH_MS) by VOWEL_MARGIN_DB. Fitted on music: on a dry
              vocal it nulls 23 dB under what the reference removes. On SINE tests it disagrees
              (the reference reads a noisy vowel well under a pure one) - a named limit.
    smooth    four branching one-poles on the reduction in dB (STAGES: the first quicker), each
              `dsp._gr_ballistics_db`
    lookahead the gain a sample gets is the one computed LOOKAHEAD_MS after it. Offline the render is
              aligned; natively the audio is delayed by it and the chain reports the latency.
    apply     WIDE: the whole signal x gain. SPLIT: x - (1 - gain) x (x - LPz(x)), LPz a ZERO-PHASE
              one-pole low-pass at FREQ / 4 run forward and back - a symmetric FIR of r^|k| - so the
              cut is a gentle, linear-phase high shelf whose depth is the reduction.

The native header (`native/Source/DeessLaw.h`) is written by
`python -m mix_engine.deess_law --write-header` and held to this table by test_deess_law.py.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

MODES = ("allround", "vocal")          # the stored words, by mode index
FACE_WORDS = ("ALLROUND", "VOCAL")
DEFAULT_MODE = 0

RANGE_MAX_DB = 24.0                    # the reference's range stops at 24 dB
TOP_OFF_HZ = 20000.0                   # TOP at 20 kHz = no low-pass (the band runs to the top)
THRESHOLD_MAX_DB = -0.5                # the slope is range / -threshold: never divide by 0
BW4_Q = (0.5412, 1.3066)               # a Butterworth 4th order, two sections
BAND_MIN_OCT = 0.32                    # TOP is held at least this far over FREQ
#: the detector's centre bell, by the band's width in octaves: (octaves, gain dB, Q)
BAND_BELL = ((0.32, 17.7, 5.57), (0.58, 12.5, 2.85), (1.0, 6.4, 0.78), (1.42, 2.1, 0.28), (2.0, 0.0, 0.28))
HOLD_MS = 5.312
#: the four smoothing stages, (attack ms, release ms) - toward more reduction is the attack
STAGES = ((0.611, 2.496), (2.439, 2.753), (2.439, 2.753), (2.439, 2.753))
LOOKAHEAD_MS = 12.0
SPLIT_FC_K = 0.25                      # the split's low-pass corner, x FREQ
SPLIT_FIR_MAX_MS = 3.0                 # its symmetric FIR's half-length at most
SPLIT_FIR_FLOOR = 1.0e-5               # ...and no longer than r^k reaching this
# VOCAL
SIB_TOP_K = 2.823
SIB_HOLD_MS = 4.142
SIB_SMOOTH_MS = 2.512
SIB_GATE_DB = -1.223
VOWEL_HP_HZ, VOWEL_HP_Q = 142.0, 0.79
VOWEL_TOP_K, VOWEL_TOP_Q = 0.327, 0.82
VOWEL_SMOOTH_MS = 1.805
VOWEL_MARGIN_DB = 19.176
LEVEL_FLOOR = 1.0e-9                   # |x| floor before the dB of the vowel average


def mode_index(v) -> int:
    if isinstance(v, str):
        w = v.strip().lower()
        return MODES.index(w) if w in MODES else DEFAULT_MODE
    try:
        return min(max(int(round(float(v))), 0), len(MODES) - 1)
    except (TypeError, ValueError):
        return DEFAULT_MODE


def band_bell(hp_hz: float, top_hz: float) -> tuple[float, float]:
    """(gain dB, Q) of the detector's centre bell for a band FREQ..TOP (none with TOP off)."""
    if top_hz >= TOP_OFF_HZ:
        return 0.0, 0.0
    octs = math.log2(top_hz / hp_hz)
    o = [b[0] for b in BAND_BELL]
    return float(np.interp(octs, o, [b[1] for b in BAND_BELL])), float(np.interp(octs, o, [b[2] for b in BAND_BELL]))


def band_top(hp_hz: float, top_hz: float) -> float:
    """TOP as the law uses it: off at 20 kHz, else at least BAND_MIN_OCT over FREQ."""
    if top_hz >= TOP_OFF_HZ:
        return TOP_OFF_HZ
    return max(float(top_hz), float(hp_hz) * 2.0 ** BAND_MIN_OCT)


def split_fir(hp_hz: float, sr: float) -> np.ndarray:
    """The zero-phase low-pass: h[k] = r^|k|, normalised, |k| <= N."""
    r = math.exp(-2.0 * math.pi * SPLIT_FC_K * hp_hz / sr)
    n = split_fir_half(hp_hz, sr)
    k = np.arange(-n, n + 1)
    h = r ** np.abs(k)
    return h / h.sum()


def split_fir_half(hp_hz: float, sr: float) -> int:
    r = math.exp(-2.0 * math.pi * SPLIT_FC_K * hp_hz / sr)
    return int(min(math.ceil(math.log(SPLIT_FIR_FLOOR) / math.log(r)), round(SPLIT_FIR_MAX_MS * sr / 1000.0)))


def lookahead_samples(sr: float) -> int:
    return int(round(LOOKAHEAD_MS * sr / 1000.0))


def _filters(sr: float, hp: float, top: float):
    """(main detector sections, sibilance sections, vowel sections) as engine biquads (b, a)."""
    from .reference_monitor import biquad_coeffs
    hp = min(float(hp), 0.45 * sr)
    hp4 = [biquad_coeffs("hpf2", sr, hp, q, 0.0) for q in BW4_Q]
    main = list(hp4)
    top = band_top(hp, top)
    if top < TOP_OFF_HZ and top < 0.45 * sr:
        main += [biquad_coeffs("lpf2", sr, top, q, 0.0) for q in BW4_Q]
        g, q = band_bell(hp, top)
        if g > 0.0:
            main.append(biquad_coeffs("peak", sr, min(math.sqrt(hp * top), 0.45 * sr), q, g))
    sib = list(hp4) + [biquad_coeffs("lpf2", sr, min(SIB_TOP_K * hp, 0.45 * sr), 0.7071, 0.0)]
    vow = [biquad_coeffs("hpf2", sr, VOWEL_HP_HZ, VOWEL_HP_Q, 0.0),
           biquad_coeffs("lpf2", sr, min(VOWEL_TOP_K * hp, 0.45 * sr), VOWEL_TOP_Q, 0.0)]
    return main, sib, vow


def _cascade(x: np.ndarray, sections) -> np.ndarray:
    from scipy.signal import lfilter
    y = x
    for b, a in sections:
        y = lfilter(b, a, y)
    return y


def _running_max(a: np.ndarray, n: int) -> np.ndarray:
    """max of a[i - n .. i] (zeros before the start)."""
    from scipy.ndimage import maximum_filter1d
    if n <= 0:
        return a
    pad = np.concatenate([np.zeros(n), a])
    return maximum_filter1d(pad, size=n + 1, origin=n // 2)[n:]


def _one_pole(x: np.ndarray, sr: float, ms: float, start: float) -> np.ndarray:
    from scipy.signal import lfilter
    k = math.exp(-1000.0 / (sr * ms))
    y, _ = lfilter([1.0 - k], [1.0, -k], x, zi=[k * start])
    return y


def gain_db(audio: np.ndarray, sr: float, threshold_db: float, range_db: float, freq_hz: float,
            top_hz: float = TOP_OFF_HZ, mode=DEFAULT_MODE) -> np.ndarray:
    """The gain (dB, <= 0) each sample of `audio` (channels, frames) gets - lookahead included."""
    from .dsp import _gr_ballistics_db
    x = np.asarray(audio, dtype=np.float64)
    mid = x.mean(axis=0) if x.ndim == 2 else x
    n = mid.shape[0]
    la = lookahead_samples(sr)
    mid = np.concatenate([mid, np.zeros(la)])       # the cut the last samples see comes from after them
    thr = min(float(threshold_db), THRESHOLD_MAX_DB)
    rng = min(max(-float(range_db), 0.0), RANGE_MAX_DB)
    if rng <= 0.0:
        return np.zeros(n)
    main, sib, vow = _filters(sr, float(freq_hz), float(top_hz))
    det = _running_max(np.abs(_cascade(mid, main)), int(HOLD_MS * sr / 1000.0))
    level = 20.0 * np.log10(np.maximum(det, 1.0e-12))
    red = np.clip(rng * (level - thr) / -thr, 0.0, rng)
    if mode_index(mode) == 1:
        s = _running_max(np.abs(_cascade(mid, sib)), int(SIB_HOLD_MS * sr / 1000.0))
        S = _one_pole(20.0 * np.log10(s + LEVEL_FLOOR), sr, SIB_SMOOTH_MS, -120.0)
        V = _one_pole(20.0 * np.log10(np.abs(_cascade(mid, vow)) + LEVEL_FLOOR), sr, VOWEL_SMOOTH_MS, -120.0)
        red = np.where((S - V >= VOWEL_MARGIN_DB) & (S >= thr + SIB_GATE_DB), red, 0.0)
    g = -red
    for att, rel in STAGES:
        g = _gr_ballistics_db(g, sr, att, rel)
    return g[la:la + n]


def apply(audio: np.ndarray, sr: float, entry: dict) -> np.ndarray:
    """(channels, frames) float32 through the de-esser `entry`."""
    x = np.asarray(audio, dtype=np.float32)
    hp = float(entry.get("freq_hz", 6000.0))
    g = gain_db(x, sr, float(entry.get("threshold_db", -20.0)), float(entry.get("range_db", -6.0)), hp,
                float(entry.get("top_hz", TOP_OFF_HZ)), entry.get("mode", DEFAULT_MODE))
    if not np.any(g < 0.0):
        return x
    gain = (10.0 ** (g / 20.0)).astype(np.float64)
    if entry.get("wide"):
        return (x * gain[np.newaxis, :]).astype(np.float32)
    from scipy.signal import fftconvolve
    h = split_fir(hp, sr)
    xd = x.astype(np.float64)
    lpz = np.stack([fftconvolve(c, h, mode="same") for c in xd])
    return (xd - (1.0 - gain)[np.newaxis, :] * (xd - lpz)).astype(np.float32)


# ---- the native header ----------------------------------------------------------------------------
HEADER = Path(__file__).resolve().parents[2] / "native" / "Source" / "DeessLaw.h"


def header_text() -> str:
    def f(v):
        t = f"{float(v):.6g}"
        return (t if ("." in t or "e" in t) else t + ".0") + "f"
    bell = ", ".join("{ " + ", ".join(f(v) for v in b) + " }" for b in BAND_BELL)
    stages = ", ".join("{ " + f(a) + ", " + f(r) + " }" for a, r in STAGES)
    return f"""#pragma once
// GENERATED by `python -m mix_engine.deess_law --write-header` from server/mix_engine/deess_law.py -
// the de-esser's law (that file says how it was measured and what each constant does). Do not edit:
// change the Python table and write this again; test_deess_law.py holds the two equal.

namespace deesslaw
{{
    static constexpr int kNumModes = {len(MODES)};
    static constexpr const char* kFaceWord[kNumModes] = {{ {", ".join('"' + w + '"' for w in FACE_WORDS)} }};
    static constexpr float kRangeMaxDb = {f(RANGE_MAX_DB)}, kTopOffHz = {f(TOP_OFF_HZ)}, kThresholdMaxDb = {f(THRESHOLD_MAX_DB)};
    static constexpr float kBw4Q[2] = {{ {f(BW4_Q[0])}, {f(BW4_Q[1])} }};
    static constexpr float kBandMinOct = {f(BAND_MIN_OCT)};
    static constexpr int kNumBell = {len(BAND_BELL)};
    static constexpr float kBandBell[kNumBell][3] = {{ {bell} }};   // octaves, gain dB, Q
    static constexpr float kHoldMs = {f(HOLD_MS)};
    static constexpr int kNumStages = {len(STAGES)};
    static constexpr float kStages[kNumStages][2] = {{ {stages} }};   // attack ms, release ms
    static constexpr float kLookaheadMs = {f(LOOKAHEAD_MS)};
    static constexpr float kSplitFcK = {f(SPLIT_FC_K)}, kSplitFirMaxMs = {f(SPLIT_FIR_MAX_MS)}, kSplitFirFloor = {f(SPLIT_FIR_FLOOR)};
    static constexpr float kSibTopK = {f(SIB_TOP_K)}, kSibHoldMs = {f(SIB_HOLD_MS)}, kSibSmoothMs = {f(SIB_SMOOTH_MS)}, kSibGateDb = {f(SIB_GATE_DB)};
    static constexpr float kVowelHpHz = {f(VOWEL_HP_HZ)}, kVowelHpQ = {f(VOWEL_HP_Q)}, kVowelTopK = {f(VOWEL_TOP_K)}, kVowelTopQ = {f(VOWEL_TOP_Q)};
    static constexpr float kVowelSmoothMs = {f(VOWEL_SMOOTH_MS)}, kVowelMarginDb = {f(VOWEL_MARGIN_DB)}, kLevelFloor = {f(LEVEL_FLOOR)};
}}
"""


if __name__ == "__main__":
    if "--write-header" in sys.argv:
        HEADER.write_text(header_text())
        print(f"wrote {HEADER}")


#: What a de-esser held before this law (it has no ratio, attack or release).
LEGACY_FIELDS = ("ratio", "attack_ms", "release_ms")


def convert_legacy(obj, _chain: bool = False) -> tuple[int, int]:
    """A de-esser saved before this law, converted once, in place (session STATE_VERSION 6): its old
    knobs dropped, its range held to this law's (the old default was -80), and - in a chain - one at
    ratio 1 or less, which was exactly transparent, bypassed: this law has no ratio and would cut it.
    Returns (converted, bypassed)."""
    n = off = 0
    if isinstance(obj, dict):
        if obj.get("type") == "deess" and "ratio" in obj:
            try:
                ratio = float(obj.get("ratio") or 1.0)
            except (TypeError, ValueError):
                ratio = 1.0
            for k in LEGACY_FIELDS:
                obj.pop(k, None)
            r = obj.get("range_db")
            if isinstance(r, (int, float)) and not isinstance(r, bool):
                obj["range_db"] = min(0.0, max(-RANGE_MAX_DB, float(r)))
            if _chain and ratio <= 1.0 and obj.get("enabled", True):
                obj["enabled"] = False
                off += 1
            n += 1
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                a, b = convert_legacy(v, _chain=k in ("chain", "master_chain"))
                n, off = n + a, off + b
    elif isinstance(obj, list):
        for v in obj:
            a, b = convert_legacy(v, _chain=_chain)
            n, off = n + a, off + b
    return n, off
