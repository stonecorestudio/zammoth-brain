"""The algorithmic reverb's law (2026-10-27): fitted to a reference algorithmic reverb, measured as a black box.

What the reference does (its default "modern" space, measured on impulses and on a song):
  * the knob is the decay time near 1 kHz; the lows ring longer and the highs shorter, by ratios
    that move with the decay (TS_* below, a measured table - one formula failed it);
  * it starts ONSET_MS after the input (plus the pre-delay), and builds without a gap;
  * its tail has a colour of its own (OUT_EQ, fitted by third-octave) and no audible modes;
  * TONE is its brightness: a high shelf that turns down into a lowpass and up into a lift, a
    low shelf that thins the lows at the top of the range (TONE_TABLE), and the highs' decay
    shortened or lengthened with it (high_decay_scale).

The network: four allpasses per channel, eight early taps of the diffused input, then sixteen
delay lines mixed by a 16x16 Hadamard on every pass. Each line is read MOD_MS either side of its
length on a slow sine of its own (a 4-point cubic read - a linear one is a lowpass inside the loop;
the cubic still takes ~1.5 dB a pass at 16 kHz, so the top octave decays ~25 % sooner than the
reference's - a named limit, heard and kept: the reverb James picked had it),
scaled by its loop gain and filtered by a first-order low shelf and high shelf whose gains make
every line fall 60 dB in the band's own decay. The lines' lengths are whole samples. The output is
coloured by OUT_EQ and the tone's two shelves (eq_law sections).

The native chain runs the same (`ReverbLaw.h` is written from this file:
`python -m mix_engine.reverb_law --write-header`); `--reverb-law` and test_reverb_law.py hold the
two together.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

LINE_MS = (9.1, 11.3, 13.7, 16.1, 18.9, 22.3, 25.7, 29.9, 34.3, 39.1, 44.9, 50.3, 56.9, 63.7, 71.3, 79.9)
AP_MS = ((1.53, 2.31, 3.97, 5.61), (1.67, 2.53, 3.71, 5.89))   # left, right
AP_G = 0.6
IN_SIGNS = ((1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1),
            (-1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1))
OUT_SIGNS = ((-1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1),
             (-1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1))
IN_GAIN = 0.25
OUT_GAIN = 0.25
ONSET_MS = 6.5
EARLY_MS = (0.4, 3.1, 6.7, 9.9, 14.3, 18.1, 23.9, 29.3)
EARLY_SIGNS = ((1, 1, 1, -1, -1, -1, 1, -1), (1, -1, 1, -1, 1, -1, -1, 1))
EARLY_G = 0.08
EARLY_TAU_MS = 25.0
MOD_MS = 0.18
MOD_RATE_LO_HZ, MOD_RATE_HI_HZ = 0.31, 0.97
LOW_SHELF_HZ, HIGH_SHELF_HZ = 400.0, 3500.0
# T60 / decay, by band, against the decay (measured)
TS_S = (0.6, 1.2, 2.5, 5.0, 10.0)
TS_LOW = (1.03, 1.13, 1.08, 1.06, 1.09)
TS_MID = (0.82, 0.87, 0.87, 0.85, 0.85)
TS_HI = (0.70, 0.72, 0.70, 0.61, 0.46)
# the tail's colour (shape, Hz, dB, q, slope)
OUT_EQ = (("low_shelf", 202.1, 4.36, 0.908, 12), ("high_shelf", 1171.7, 1.78, 2.184, 12),
          ("lpf", 6541.5, 0.0, 0.937, 12), ("hpf", 20.0, 0.0, 0.7, 12))
# TONE: (tone, high shelf Hz, dB, q, 400 Hz low shelf dB); tone 0.5 is flat
TONE_TABLE = ((0.0, 2293.8, -32.39, 0.34, 2.14), (0.125, 2874.5, -24.0, 0.43, 1.65),
              (0.25, 3582.3, -15.83, 0.50, 1.10), (0.375, 4435.8, -7.71, 0.64, 0.41),
              (0.5, 6000.0, 0.0, 0.66, 0.0), (0.625, 9806.4, 10.0, 0.69, -0.56),
              (0.75, 9806.4, 14.67, 0.69, -2.08), (0.875, 10014.6, 14.89, 0.68, -3.98),
              (1.0, 9936.0, 14.81, 0.68, -6.44))
TONE_LOW_SHELF_HZ = 400.0
WET_DB = -1.26
DECAY_MIN_S, DECAY_MAX_S, PRE_MAX_MS = 0.2, 12.0, 250.0


def ms_samples(ms: float, sample_rate: float) -> int:
    return int(math.floor(ms * sample_rate / 1000.0 + 0.5))


def line_samples(sample_rate: float) -> list[int]:
    return [max(1, ms_samples(ms, sample_rate)) for ms in LINE_MS]


def mod_rates() -> list[float]:
    return [MOD_RATE_LO_HZ + (MOD_RATE_HI_HZ - MOD_RATE_LO_HZ) * j / 15.0 for j in range(16)]


def _interp_log(tab, s: float) -> float:
    x = math.log(max(s, 1e-6))
    xs = [math.log(v) for v in TS_S]
    if x <= xs[0]:
        return tab[0]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return tab[i - 1] + t * (tab[i] - tab[i - 1])
    return tab[-1]


def high_decay_scale(tone: float) -> float:
    b = 2.0 * min(1.0, max(0.0, tone)) - 1.0
    return 1.0 + 0.23 * b if b < 0.0 else 1.0 + 0.13 * min(2.0 * b, 1.0)


def decays(decay_s: float, tone: float) -> tuple[float, float, float]:
    """(low, mid, high) T60 seconds."""
    s = min(DECAY_MAX_S, max(DECAY_MIN_S, float(decay_s)))
    return (s * _interp_log(TS_LOW, s), s * _interp_log(TS_MID, s), s * _interp_log(TS_HI, s) * high_decay_scale(tone))


def shelf1(sample_rate: float, fc: float, g_db: float, high: bool) -> tuple[float, float, float]:
    """A first-order shelf (bilinear), gain g_db at DC (low) or Nyquist (high): (b0, b1, a1)."""
    K = math.tan(math.pi * fc / sample_rate)
    V = 10.0 ** (g_db / 20.0)
    if not high:
        if V >= 1.0:
            b0, b1, a0, a1 = 1.0 + V * K, V * K - 1.0, 1.0 + K, K - 1.0
        else:
            b0, b1, a0, a1 = 1.0 + K, K - 1.0, 1.0 + K / V, K / V - 1.0
    else:
        if V >= 1.0:
            b0, b1, a0, a1 = V + K, K - V, 1.0 + K, K - 1.0
        else:
            b0, b1, a0, a1 = 1.0 + K, K - 1.0, 1.0 / V + K, K - 1.0 / V
    return b0 / a0, b1 / a0, a1 / a0


def loop_filters(sample_rate: float, decay_s: float, tone: float) -> list[tuple[float, float, float, float, float]]:
    """Each line's loop filter as one biquad (b0, b1, b2, a1, a2): its gain at 1 kHz's decay times the
    low and the high shelf that set the other bands' decays."""
    tl, tm, th = decays(decay_s, tone)
    out = []
    for n in line_samples(sample_rate):
        sec = n / sample_rate
        g = 10.0 ** (-3.0 * sec / tm)
        lb0, lb1, la1 = shelf1(sample_rate, LOW_SHELF_HZ, -60.0 * sec * (1.0 / tl - 1.0 / tm), False)
        hb0, hb1, ha1 = shelf1(sample_rate, HIGH_SHELF_HZ, -60.0 * sec * (1.0 / th - 1.0 / tm), True)
        out.append((g * lb0 * hb0, g * (lb0 * hb1 + lb1 * hb0), g * lb1 * hb1, la1 + ha1, la1 * ha1))
    return out


def tone_shelves(tone: float) -> tuple[float, float, float, float]:
    """(high shelf Hz, dB, q, low shelf dB) at a tone, linear between the table's rows."""
    t = min(1.0, max(0.0, float(tone)))
    rows = TONE_TABLE
    for i in range(1, len(rows)):
        if t <= rows[i][0] or i == len(rows) - 1:
            a, b = rows[i - 1], rows[i]
            u = (t - a[0]) / (b[0] - a[0])
            return tuple(a[k] + u * (b[k] - a[k]) for k in range(1, 5))   # type: ignore[return-value]
    return rows[-1][1:]   # pragma: no cover


def output_sections(sample_rate: float, tone: float) -> list[tuple[float, float, float, float, float]]:
    """The tail's colour and the tone, as eq_law biquads (b0, b1, b2, a1, a2), in order."""
    from .eq_law import band_sections
    secs = []
    for shape, f, g, q, slope in OUT_EQ:
        secs += band_sections(shape, sample_rate, f, q, g, slope)
    hf, hg, hq, lg = tone_shelves(tone)
    secs += band_sections("high_shelf", sample_rate, hf, hq, hg, 12)
    secs += band_sections("low_shelf", sample_rate, TONE_LOW_SHELF_HZ, 0.7, lg, 12)
    return [(b[0], b[1], b[2], a[1], a[2]) for (b, a) in secs]


def _allpass(x: np.ndarray, m: int, g: float) -> np.ndarray:
    """Schroeder allpass v[n] = x[n] + g·v[n-m], y[n] = v[n-m] - g·v[n], one m-sized block at a time."""
    n = x.shape[0]
    v = np.zeros(n + m)
    for s0 in range(0, n, m):
        e0 = min(n, s0 + m)
        v[m + s0:m + e0] = x[s0:e0] + g * v[s0:e0]
    return v[:n] - g * v[m:]


def _hadamard16() -> np.ndarray:
    h = np.array([[1.0]])
    for _ in range(4):
        h = np.block([[h, h], [h, -h]])
    return h / 4.0


def apply(audio: np.ndarray, sample_rate: float, entry: dict) -> np.ndarray:
    """The wet of one reverb on (frames, channels) audio. A mono input gives a mono output (the left side)."""
    from scipy.signal import lfilter
    x_in = np.asarray(audio, dtype=np.float64)
    n_frames = x_in.shape[0]
    if n_frames == 0:
        return np.zeros_like(x_in)
    sr = float(sample_rate)
    mono = x_in.shape[1] == 1
    x2 = np.repeat(x_in[:, :1], 2, axis=1) if mono else x_in[:, :2]
    pre_ms = min(PRE_MAX_MS, max(0.0, float(entry.get("pre_delay_ms", 20.0))))
    pre = int(pre_ms * sr / 1000.0) + ms_samples(ONSET_MS, sr)
    decay_s = float(entry.get("decay_s", 1.8))
    tone = float(entry.get("tone", 0.5))
    fed = np.zeros_like(x2)
    if pre < n_frames:
        fed[pre:] = x2[: n_frames - pre]
    dif = np.zeros_like(fed)
    for ch in (0, 1):
        v = fed[:, ch]
        for ms in AP_MS[ch]:
            v = _allpass(v, max(1, ms_samples(ms, sr)), AP_G)
        dif[:, ch] = v
    early = np.zeros_like(dif)
    for k, ms in enumerate(EARLY_MS):
        d = ms_samples(ms, sr)
        g = EARLY_G * math.exp(-ms / EARLY_TAU_MS)
        if d < n_frames:
            for ch in (0, 1):
                early[d:, ch] += g * EARLY_SIGNS[ch][k] * dif[: n_frames - d, ch]
    lines = np.array(line_samples(sr))
    depth = MOD_MS * sr / 1000.0
    rates = np.array(mod_rates())
    phase0 = np.arange(16, dtype=np.float64)
    filt = loop_filters(sr, decay_s, tone)
    h = _hadamard16()
    zi = np.zeros((16, 2))
    block = int(lines.min()) - 1          # every read is from a sample written before the block
    cols = np.arange(16)
    # The lines' history is a RING as long as the longest read, not the song (the pre-merge review
    # of 2026-10-28: three whole-song (frames, 16) arrays were 1.45 GB a minute of audio, and the
    # brain is one process). A slot not yet written holds 0 - what a read before the start reads.
    max_read = int(math.ceil(float(lines.max()) + 2.0 * depth)) + 4
    ring_len = 1 << int(math.ceil(math.log2(max_read + block + 8)))
    mask = ring_len - 1
    ring = np.zeros((ring_len, 16))
    out_signs = np.array(OUT_SIGNS, dtype=np.float64)          # (2, 16)
    wet = np.zeros((n_frames, 2))
    for s0 in range(0, n_frames, block):
        e0 = min(n_frames, s0 + block)
        idx = np.arange(s0, e0, dtype=np.float64)
        dl = lines[None, :] + depth * (1.0 + np.sin(2.0 * math.pi * rates[None, :] * idx[:, None] / sr + phase0[None, :]))
        ip = np.floor(dl).astype(np.int64)
        fr = dl - ip
        w = np.arange(s0, e0)[:, None]

        def tap(off):
            return ring[(w - ip + off) & mask, cols[None, :]]
        xm1, x0, x1, x2_ = tap(1), tap(0), tap(-1), tap(-2)
        rd = x0 + fr * ((x1 - xm1) * 0.5 + fr * ((xm1 - 2.5 * x0 + 2.0 * x1 - 0.5 * x2_)
                                                 + fr * (1.5 * (x0 - x1) + 0.5 * (x2_ - xm1))))
        for j in range(16):
            b0, b1, b2, a1, a2 = filt[j]
            rd[:, j], zi[j] = lfilter([b0, b1, b2], [1.0, a1, a2], rd[:, j], zi=zi[j])
        wet[s0:e0, 0] = rd @ out_signs[0]
        wet[s0:e0, 1] = rd @ out_signs[1]
        inj = (np.outer(dif[s0:e0, 0], IN_SIGNS[0]) + np.outer(dif[s0:e0, 1], IN_SIGNS[1])) * IN_GAIN
        ring[np.arange(s0, e0) & mask] = rd @ h.T + inj
    wet = wet * OUT_GAIN
    wet = wet + early
    for b0, b1, b2, a1, a2 in output_sections(sr, tone):
        wet = lfilter([b0, b1, b2], [1.0, a1, a2], wet, axis=0)
    wet *= 10.0 ** (WET_DB / 20.0)
    return wet[:, :1] if mono else wet


def tail_seconds(decay_s: float, tone: float, pre_delay_ms: float) -> float:
    """How long the wet rings after its input stops: the pre-delay and onset, then the longest band's T60."""
    pre = min(PRE_MAX_MS, max(0.0, pre_delay_ms)) + ONSET_MS
    return pre / 1000.0 + max(decays(decay_s, tone))


def _arr(v) -> str:
    return "{ " + ", ".join(repr(float(x)) for x in v) + " }"


def header_text() -> str:
    return f"""#pragma once
// GENERATED by server/mix_engine/reverb_law.py (python -m mix_engine.reverb_law --write-header) - the
// algorithmic reverb's law, fitted to a reference reverb (2026-10-27). Do not edit by hand.
namespace reverblaw
{{
    inline constexpr int    kLines = 16, kAps = 4, kEarly = 8, kOutSections = {len(OUT_EQ) + 2};
    inline constexpr double kLineMs[kLines] = {_arr(LINE_MS)};
    inline constexpr double kApMs[2][kAps] = {{ {_arr(AP_MS[0])}, {_arr(AP_MS[1])} }};
    inline constexpr double kApG = {AP_G!r};
    inline constexpr double kInSigns[2][kLines] = {{ {_arr(IN_SIGNS[0])}, {_arr(IN_SIGNS[1])} }};
    inline constexpr double kOutSigns[2][kLines] = {{ {_arr(OUT_SIGNS[0])}, {_arr(OUT_SIGNS[1])} }};
    inline constexpr double kInGain = {IN_GAIN!r}, kOutGain = {OUT_GAIN!r};
    inline constexpr double kOnsetMs = {ONSET_MS!r};
    inline constexpr double kEarlyMs[kEarly] = {_arr(EARLY_MS)};
    inline constexpr double kEarlySigns[2][kEarly] = {{ {_arr(EARLY_SIGNS[0])}, {_arr(EARLY_SIGNS[1])} }};
    inline constexpr double kEarlyG = {EARLY_G!r}, kEarlyTauMs = {EARLY_TAU_MS!r};
    inline constexpr double kModMs = {MOD_MS!r}, kModRateLoHz = {MOD_RATE_LO_HZ!r}, kModRateHiHz = {MOD_RATE_HI_HZ!r};
    inline constexpr double kLowShelfHz = {LOW_SHELF_HZ!r}, kHighShelfHz = {HIGH_SHELF_HZ!r};
    inline constexpr int    kTsRows = {len(TS_S)};
    inline constexpr double kTsS[kTsRows] = {_arr(TS_S)};
    inline constexpr double kTsLow[kTsRows] = {_arr(TS_LOW)};
    inline constexpr double kTsMid[kTsRows] = {_arr(TS_MID)};
    inline constexpr double kTsHi[kTsRows] = {_arr(TS_HI)};
    // the tail's colour: shape (0 low_shelf, 1 high_shelf, 2 lpf, 3 hpf), Hz, dB, q, slope
    inline constexpr double kOutEq[{len(OUT_EQ)}][5] = {{ {", ".join(_arr(({"low_shelf": 0, "high_shelf": 1, "lpf": 2, "hpf": 3}[s], f, g, q, sl)) for s, f, g, q, sl in OUT_EQ)} }};
    inline constexpr int    kToneRows = {len(TONE_TABLE)};
    inline constexpr double kTone[kToneRows][5] = {{ {", ".join(_arr(r) for r in TONE_TABLE)} }};
    inline constexpr double kToneLowShelfHz = {TONE_LOW_SHELF_HZ!r};
    inline constexpr double kWetDb = {WET_DB!r};

    inline constexpr double kDecayMinS = {DECAY_MIN_S!r}, kDecayMaxS = {DECAY_MAX_S!r}, kPreMaxMs = {PRE_MAX_MS!r};
}}
"""


if __name__ == "__main__":
    if "--write-header" in sys.argv:
        p = Path(__file__).resolve().parents[2] / "native" / "Source" / "ReverbLaw.h"
        p.write_text(header_text())
        print(f"wrote {p}")
