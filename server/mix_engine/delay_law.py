"""The delay's law (2026-10-27): fitted to a reference delay, measured as a black box.

What the reference does (its default patch, its modulation measured apart from its core):
  * echoes land on time; FEEDBACK is the linear gain of a pass;
  * every pass goes through the same two filters - a 12 dB/oct Butterworth high-pass at HP_HZ (-3 dB
    there) and a 24 dB/oct Linkwitz-Riley low-pass (-6 dB at its corner; two Butterworth sections) -
    so each repeat is thinner and darker than the one before; TONE sets the low-pass corner,
    tone_hz (5 kHz at 0.5, 1.25 kHz at 0, 20 kHz at 1);
  * a slow wow on the time (WOW_MS either side at WOW_HZ), the right side RIGHT_SCALE longer, and the
    first echo OUT_DB under the input (at a full wet mix).
  Its envelope-follower ducking is not part of this law.

The network: a ring per side holding r = x + feedback x e; each side reads its ring at its delay (the
right's RIGHT_SCALE longer) plus the wow, on a 4-point cubic; e = the loop's filter of that read; the
wet is OUT_DB x e. PING-PONG: the left's ring takes the input's mid and the right's echo, the right's
ring the left's echo, both at the plain delay. The native chain runs the same (`DelayLaw.h` is written
from this file: `python -m mix_engine.delay_law --write-header`); `--delay-law` and test_delay_law.py
hold the two together.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HP_HZ = 200.0
BUTTERWORTH_Q = 1.0 / math.sqrt(2.0)
TONE_MID_HZ, TONE_SPAN = 5000.0, 4.0           # tone_hz = TONE_MID_HZ x TONE_SPAN^(2 tone - 1)
WOW_MS, WOW_HZ = 1.4, 0.14
RIGHT_SCALE = 1.01
OUT_DB = -3.1
DELAY_MIN_MS, DELAY_MAX_MS, FEEDBACK_MAX = 20.0, 2000.0, 0.85


def tone_hz(tone: float) -> float:
    return TONE_MID_HZ * TONE_SPAN ** (2.0 * min(1.0, max(0.0, float(tone))) - 1.0)


def biquad(kind: str, sample_rate: float, fc: float, q: float = BUTTERWORTH_Q) -> tuple[float, float, float, float, float]:
    """The bilinear (cookbook) low- or high-pass, (b0, b1, b2, a1, a2); the corner held under 0.45 x the rate."""
    w = 2.0 * math.pi * min(fc, 0.45 * sample_rate) / sample_rate
    c, s = math.cos(w), math.sin(w)
    al = s / (2.0 * q)
    a0 = 1.0 + al
    if kind == "lp":
        b0, b1, b2 = (1.0 - c) / 2.0, 1.0 - c, (1.0 - c) / 2.0
    else:
        b0, b1, b2 = (1.0 + c) / 2.0, -(1.0 + c), (1.0 + c) / 2.0
    return b0 / a0, b1 / a0, b2 / a0, (-2.0 * c) / a0, (1.0 - al) / a0


def loop_sections(sample_rate: float, tone: float) -> list[tuple[float, float, float, float, float]]:
    """The loop's filter, in order: the high-pass, then the two low-pass sections of the Linkwitz-Riley."""
    lp = biquad("lp", sample_rate, tone_hz(tone))
    return [biquad("hp", sample_rate, HP_HZ), lp, lp]


def held(entry: dict) -> tuple[float, float, float, bool]:
    delay_ms = float(np.clip(float(entry["delay_ms"]), DELAY_MIN_MS, DELAY_MAX_MS))
    fb = float(np.clip(float(entry["feedback"]), 0.0, FEEDBACK_MAX))
    tone = float(np.clip(float(entry.get("tone", 0.5)), 0.0, 1.0))
    return delay_ms, fb, tone, float(entry.get("ping_pong", 0.0)) >= 0.5


def _cubic(ring: np.ndarray, k: np.ndarray, fr: np.ndarray) -> np.ndarray:
    def at(i):
        return np.where(i >= 0, ring[np.clip(i, 0, None)], 0.0)
    xm1, x0, x1, x2 = at(k + 1), at(k), at(k - 1), at(k - 2)
    return x0 + fr * ((x1 - xm1) * 0.5 + fr * ((xm1 - 2.5 * x0 + 2.0 * x1 - 0.5 * x2)
                                               + fr * (1.5 * (x0 - x1) + 0.5 * (x2 - xm1))))


def apply(audio: np.ndarray, sample_rate: float, entry: dict) -> np.ndarray:
    """The wet of one delay on (frames, channels) audio. A mono input gives a mono output."""
    from scipy.signal import lfilter
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        return apply(x[:, np.newaxis], sample_rate, entry)[:, 0]
    n, channels = x.shape
    sr = float(sample_rate)
    delay_ms, fb, tone, pp = held(entry)
    out = np.zeros_like(x)
    if n == 0:
        return out
    D = delay_ms * sr / 1000.0
    wow = WOW_MS * sr / 1000.0
    secs = loop_sections(sr, tone)
    g = 10.0 ** (OUT_DB / 20.0)
    stereo = channels >= 2
    sides = 2 if stereo else 1
    pp = pp and stereo
    scale = [1.0, 1.0 if pp else RIGHT_SCALE][:sides]
    rings = np.zeros((sides, n))
    e = np.zeros((sides, n))
    zi = [[np.zeros(2) for _ in secs] for _ in range(sides)]
    mid = 0.5 * (x[:, 0] + x[:, 1]) if stereo else x[:, 0]
    block = max(1, int(D - wow) - 3)    # every read is from a sample written before the block
    for s0 in range(0, n, block):
        e0 = min(n, s0 + block)
        idx = np.arange(s0, e0)
        m = wow * np.sin(2.0 * math.pi * WOW_HZ * idx / sr)
        for c in range(sides):
            dl = D * scale[c] + m
            ip = np.floor(dl).astype(np.int64)
            u = _cubic(rings[c], idx - ip, dl - ip)
            for j, (b0, b1, b2, a1, a2) in enumerate(secs):
                u, zi[c][j] = lfilter([b0, b1, b2], [1.0, a1, a2], u, zi=zi[c][j])
            e[c, s0:e0] = u
        if pp:
            rings[0, s0:e0] = mid[s0:e0] + fb * e[1, s0:e0]
            rings[1, s0:e0] = fb * e[0, s0:e0]
        else:
            for c in range(sides):
                rings[c, s0:e0] = x[s0:e0, c] + fb * e[c, s0:e0]
    out[:, :sides] = (g * e).T
    return out


def pass_db(f: float, tone: float, sample_rate: float) -> float:
    """What one pass through the loop's filter does at f, in dB (the face draws the repeats with it)."""
    w = 2.0 * math.pi * f / sample_rate
    z1 = complex(math.cos(w), -math.sin(w))
    tot = 0.0
    for b0, b1, b2, a1, a2 in loop_sections(sample_rate, tone):
        h = (b0 + b1 * z1 + b2 * z1 * z1) / (1.0 + a1 * z1 + a2 * z1 * z1)
        tot += 20.0 * math.log10(max(abs(h), 1e-30))
    return tot


def header_text() -> str:
    return f"""#pragma once
// GENERATED by server/mix_engine/delay_law.py (python -m mix_engine.delay_law --write-header) - the
// delay's law, fitted to a reference delay (2026-10-27). Do not edit by hand.
namespace delaylaw
{{
    inline constexpr double kHpHz = {HP_HZ!r};
    inline constexpr double kButterworthQ = {BUTTERWORTH_Q!r};
    inline constexpr double kToneMidHz = {TONE_MID_HZ!r}, kToneSpan = {TONE_SPAN!r};
    inline constexpr double kWowMs = {WOW_MS!r}, kWowHz = {WOW_HZ!r};
    inline constexpr double kRightScale = {RIGHT_SCALE!r};
    inline constexpr double kOutDb = {OUT_DB!r};
    inline constexpr double kDelayMinMs = {DELAY_MIN_MS!r}, kDelayMaxMs = {DELAY_MAX_MS!r}, kFeedbackMax = {FEEDBACK_MAX!r};
}}
"""


if __name__ == "__main__":
    if "--write-header" in sys.argv:
        p = Path(__file__).resolve().parents[2] / "native" / "Source" / "DelayLaw.h"
        p.write_text(header_text())
        print(f"wrote {p}")
