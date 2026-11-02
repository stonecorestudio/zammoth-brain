"""The dynamic EQ's law (2026-10-27): fitted to a reference EQ's dynamic band, measured as a black box.

What the reference does, measured (a bell band; tones at its frequency, a quiet probe on one channel
read by lock-in while the other carries the drive - the detector is linked, so the probe reads the
gain without triggering it):
  * the band rests at 0 dB and moves by its RANGE (the old `gain_db`; up to +-30 dB) - there is NO
    ratio: over the threshold the cut follows the level dB for dB, through a soft knee whose
    half-width grows with the range, K = KNEE_A + KNEE_B * range^2 (one formula held every range from
    3 to 30 dB to ~0.05 dB), and stops at the range;
  * what it listens to: a band-pass at the band's frequency with Q = DETECT_Q_K x q, the louder of the
    two channels;
  * its timing: a peak follower that takes a rise at once and lets go over FAST_RELEASE_K x RELEASE,
    then a smoother on the cut in dB - ATTACK going in, RELEASE coming back (the reference's 0-100 %
    attack is 1.1-38 ms of this smoother, its release 110-640 ms; fitted to <= 0.13 dB on 25 dB swings);
  * the band it moves is the EQ's own bell (`eq_law`), at the band's Q, at the moving gain.

The moving bell is realised as the EQ law's sections at gain steps of TABLE_STEP_DB, chosen every
GRID samples (from the cut at the first sample of each grid block) and run as one biquad whose state
carries across a change of coefficients. The native chain runs the same (`DynEqLaw.h` is written from
this file: `python -m mix_engine.dyneq_law --write-header`); `--dyneq-law` and test_dyneq_law.py hold
the two together.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

DETECT_Q_K = 1.1
FAST_RELEASE_K = 0.13
DETECT_ATTACK_MS = 0.01   # the reference's detector takes a rise at once; a near-instant attack (88 % of a step in a sample) keeps the follower's recursion finite
KNEE_A, KNEE_B = 2.08, 0.02336
TABLE_STEP_DB = 0.1
GRID = 32
RANGE_MAX_DB = 30.0


def knee_half_width(range_db: float) -> float:
    r = abs(float(range_db))
    return KNEE_A + KNEE_B * r * r


def static_cut_db(over_db: np.ndarray, range_db: float) -> np.ndarray:
    """The cut (>= 0 dB) for a detector level `over_db` above the threshold."""
    R = min(abs(float(range_db)), RANGE_MAX_DB)
    K = knee_half_width(R)
    d = np.asarray(over_db, dtype=np.float64)
    r = np.where(d <= -K, 0.0, np.where(d < K, (d + K) ** 2 / (4.0 * K), d))
    return np.minimum(r, R)


def cut_db(x: np.ndarray, sample_rate: float, freq_hz: float, q: float, threshold_db: float,
           range_db: float, attack_ms: float, release_ms: float) -> np.ndarray:
    """The band's move in dB (>= 0) for every sample of (channels, frames) audio."""
    from scipy.signal import lfilter
    from .dsp import _gr_ballistics_db, _rising_follower
    from .reference_monitor import biquad_coeffs
    x = np.asarray(x, dtype=np.float64)
    b, a = biquad_coeffs("bandpass", sample_rate, float(freq_hz), DETECT_Q_K * float(q), 0.0)
    band = lfilter(b, a, x, axis=-1)
    s = np.max(np.abs(band), axis=0)
    env = _rising_follower(s, sample_rate, DETECT_ATTACK_MS, FAST_RELEASE_K * float(release_ms))
    level = 20.0 * np.log10(np.maximum(env, 1.0e-12))
    target = static_cut_db(level - float(threshold_db), range_db)
    return -_gr_ballistics_db(-target, sample_rate, float(attack_ms), float(release_ms))


def bell_table(sample_rate: float, freq_hz: float, q: float, range_db: float) -> np.ndarray:
    """[k] = (b0, b1, b2, a1, a2) of the EQ law's bell at sign(range) x k x TABLE_STEP_DB."""
    from .eq_law import band_sections
    R = min(abs(float(range_db)), RANGE_MAX_DB)
    sign = 1.0 if range_db >= 0 else -1.0
    n = int(math.floor(R / TABLE_STEP_DB + 1e-9)) + 1
    out = np.zeros((n, 5))
    for k in range(n):
        (b, a), = band_sections("peak", sample_rate, freq_hz, q, sign * k * TABLE_STEP_DB, 12)
        out[k] = (b[0], b[1], b[2], a[1], a[2])
    return out


def table_index(cut: float, range_db: float) -> int:
    R = min(abs(float(range_db)), RANGE_MAX_DB)
    n = int(math.floor(R / TABLE_STEP_DB + 1e-9))
    return max(0, min(n, int(math.floor(cut / TABLE_STEP_DB + 0.5))))


def apply(x: np.ndarray, sample_rate: float, entry: dict) -> np.ndarray:
    """One dynamic EQ entry on (channels, frames) audio (float32 out). Range 0 is off, bit for bit."""
    from scipy.signal import lfilter
    rng = float(entry.get("gain_db", 0.0))
    if x.shape[-1] == 0 or rng == 0.0:
        return x
    f, q = float(entry["freq_hz"]), float(entry["q"])
    g = cut_db(x, sample_rate, f, q, float(entry["threshold_db"]), rng,
               float(entry["attack_ms"]), float(entry["release_ms"]))
    table = bell_table(sample_rate, f, q, rng)
    n = x.shape[-1]
    idx = np.array([table_index(g[i], rng) for i in range(0, n, GRID)])
    xd = np.asarray(x, dtype=np.float64)
    y = np.empty_like(xd)
    zi = np.zeros((xd.shape[0], 2))
    start = 0
    while start < len(idx):            # runs of grid blocks sharing one set of coefficients
        stop = start
        while stop < len(idx) and idx[stop] == idx[start]:
            stop += 1
        b0, b1, b2, a1, a2 = table[idx[start]]
        seg = slice(start * GRID, min(n, stop * GRID))
        y[:, seg], zi = lfilter([b0, b1, b2], [1.0, a1, a2], xd[:, seg], axis=-1, zi=zi)
        start = stop
    return y.astype(np.float32)


def header_text() -> str:
    return f"""#pragma once
// GENERATED by server/mix_engine/dyneq_law.py (python -m mix_engine.dyneq_law --write-header) - the
// dynamic EQ's law, fitted to a reference EQ's dynamic band (2026-10-27). Do not edit by hand.
namespace dyneqlaw
{{
    inline constexpr double kDetectQK      = {DETECT_Q_K!r};
    inline constexpr double kFastReleaseK  = {FAST_RELEASE_K!r};
    inline constexpr double kDetectAttackMs = {DETECT_ATTACK_MS!r};
    inline constexpr double kKneeA         = {KNEE_A!r};
    inline constexpr double kKneeB         = {KNEE_B!r};
    inline constexpr double kTableStepDb   = {TABLE_STEP_DB!r};
    inline constexpr int    kGrid          = {GRID};
    inline constexpr double kRangeMaxDb    = {RANGE_MAX_DB!r};
    inline constexpr int    kTableMax      = {int(round(RANGE_MAX_DB / TABLE_STEP_DB)) + 1};
}}
"""


if __name__ == "__main__":
    if "--write-header" in sys.argv:
        p = Path(__file__).resolve().parents[2] / "native" / "Source" / "DynEqLaw.h"
        p.write_text(header_text())
        print(f"wrote {p}")


def convert_legacy(obj, _chain: bool = False) -> tuple[int, int]:
    """A dynamic EQ saved before this law (it holds a `ratio`), converted once, in place - in the
    chains, the undo snapshots and the recorded actions (session STATE_VERSION 6): its bell's Q
    into the EQ law's meaning (the cookbook bell at q is the EQ law's at q x sqrt 2, eq_law.
    legacy_band), and - in a chain - a band at ratio 1 or less, which cut nothing, bypassed: this
    law has no ratio and would cut it to its range. Returns (converted, bypassed)."""
    import math
    n = off = 0
    if isinstance(obj, dict):
        if obj.get("type") == "dynamic_eq" and "ratio" in obj:
            try:
                ratio = float(obj.get("ratio") or 1.0)
            except (TypeError, ValueError):
                ratio = 1.0
            q = obj.get("q")
            if isinstance(q, (int, float)) and not isinstance(q, bool) and math.isfinite(float(q)):
                obj["q"] = round(min(12.0, max(0.3, float(q) * math.sqrt(2.0))), 4)
            obj.pop("ratio", None)
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
