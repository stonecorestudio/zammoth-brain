"""The EQ's law (2026-10-27): fitted to a reference EQ, measured as a black box.

What the reference does, measured (its zero-latency mode, which is what this is):
  * every shape is an ANALOG prototype, and its magnitude holds to the top of the band - no
    cramping toward Nyquist (the cookbook bilinear filters this replaces lost 7 dB at 20 kHz on a
    +12 dB bell at 15 kHz);
  * its Q is not the cookbook's: a bell, a notch and a band-pass run Q / sqrt 2; a shelf (and the
    tilt shelf) runs SHELF_Q_SCALE x Q^SHELF_Q_POWER; a 12 dB/oct cut runs Q / sqrt 2 (Q 1 is
    Butterworth); a steeper cut is Butterworth with ONLY its most resonant section scaled by Q;
    a 6 dB/oct cut is first order and has no Q;
  * the tilt shelf swings +-gain (a high shelf of 2 x gain, the whole curve down by gain).

The prototypes are realised as magnitude-matched biquads: poles by the matched z-transform, the
numerator by matching |H|^2 at DC, at Nyquist and at one point between (the best of a few points,
and of the zeros by the matched z-transform too, on a fixed grid). The native engine runs the
same arithmetic (`native/Source/EqLaw.h`); `server/tests/data/eq_law_parity.json` holds the two
together. Old projects stored Q in the cookbook's meaning: `legacy_band` converts one band, once,
at load (session STATE_VERSION 3; the DAW's chain stamp `eq_law`).
"""
from __future__ import annotations

import cmath
import math

import numpy as np

LAW = 2
Q_MIN, Q_MAX = 0.1, 40.0
SECTION_Q_MAX = 60.0
INV_SQRT2 = 1.0 / math.sqrt(2.0)
SHELF_Q_SCALE = 0.7064
SHELF_Q_POWER = 0.47
GRID_POINTS = 24
GRID_LOW_HZ = 20.0
TIE_DB = 1e-9         # a later candidate wins only by more than this: no float tie decides the law
FLOOR_DB = -60.0      # a fit is judged above this: below it, the stopband's depth is no one's ear
SLOPES = (6, 12, 24, 48, 96)


def butterworth_qs(order: int) -> list[float]:
    """Section Qs of an `order`-th order Butterworth, ascending (the most resonant last)."""
    return sorted(1.0 / (2.0 * math.cos((2 * k + 1) * math.pi / (2 * order))) for k in range(order // 2))


def shelf_q(q: float) -> float:
    return SHELF_Q_SCALE * q ** SHELF_Q_POWER


def analog_sections(shape: str, gain_db: float, q: float, slope: int = 12) -> list:
    """[(num, den)] with num/den = [s^2, s, 1] coefficients, s normalised to the band's frequency."""
    q = min(max(float(q), Q_MIN), Q_MAX)
    g = float(gain_db)
    if shape == "peak":
        A, Q = 10.0 ** (g / 40.0), q * INV_SQRT2
        return [([1.0, A / Q, 1.0], [1.0, 1.0 / (A * Q), 1.0])]
    if shape in ("low_shelf", "high_shelf", "tilt_shelf"):
        Q = shelf_q(q)
        if shape == "tilt_shelf":
            A, k = 10.0 ** (2.0 * g / 40.0), 10.0 ** (-g / 20.0)
            rA = math.sqrt(A)
            return [([k * A * A, k * A * rA / Q, k * A], [1.0, rA / Q, A])]
        A = 10.0 ** (g / 40.0)
        rA = math.sqrt(A)
        if shape == "low_shelf":
            return [([A, A * rA / Q, A * A], [A, rA / Q, 1.0])]
        return [([A * A, A * rA / Q, A], [1.0, rA / Q, A])]
    if shape == "notch":
        return [([1.0, 0.0, 1.0], [1.0, 1.0 / (q * INV_SQRT2), 1.0])]
    if shape == "bandpass":
        Q = q * INV_SQRT2
        return [([0.0, 1.0 / Q, 0.0], [1.0, 1.0 / Q, 1.0])]
    if shape in ("hpf", "lpf"):
        slope = int(slope or 6)
        if slope <= 6:
            return [([0.0, 1.0, 0.0] if shape == "hpf" else [0.0, 0.0, 1.0], [0.0, 1.0, 1.0])]
        qs = butterworth_qs(slope // 6)
        qs[-1] = min(qs[-1] * q, SECTION_Q_MAX)
        num = [1.0, 0.0, 0.0] if shape == "hpf" else [0.0, 0.0, 1.0]
        return [(num, [1.0, 1.0 / Qk, 1.0]) for Qk in qs]
    raise ValueError(f"unknown EQ shape: {shape!r}")


def _mag2(num, den, W: float) -> float:
    s = 1j * W
    n = num[0] * s * s + num[1] * s + num[2]
    d = den[0] * s * s + den[1] * s + den[2]
    return abs(n / d) ** 2


def _quad_roots(a: float, b: float, c: float):
    """Both roots of a s^2 + b s + c (a != 0), as complex."""
    disc = cmath.sqrt(b * b - 4.0 * a * c)
    return (-b + disc) / (2.0 * a), (-b - disc) / (2.0 * a)


def _phis(w: float):
    p1 = math.sin(0.5 * w) ** 2
    p0 = 1.0 - p1
    return p0, p1, 4.0 * p0 * p1


def _lin_of(b0, b1, b2, a1, a2, w: float) -> float:
    z = cmath.exp(-1j * w)
    return abs((b0 + b1 * z + b2 * z * z) / (1.0 + a1 * z + a2 * z * z))


def _db_of(b0, b1, b2, a1, a2, w: float) -> float:
    z = cmath.exp(-1j * w)
    return 20.0 * math.log10(abs((b0 + b1 * z + b2 * z * z) / (1.0 + a1 * z + a2 * z * z)) + 1e-15)


def matched_section(num, den, f0: float, fs: float) -> list[float]:
    """One analog section -> [b0, b1, b2, a1, a2] (a0 = 1)."""
    T = 1.0 / fs
    w0 = 2.0 * math.pi * f0
    if den[0] == 0.0:                                             # first order
        a1 = -math.exp(-den[2] / den[1] * w0 * T)
        Wn = math.pi * fs / w0
        s0 = math.sqrt(_mag2(num, den, 1e-9)) * (1.0 + a1)
        s1 = math.sqrt(_mag2(num, den, Wn)) * (1.0 - a1)
        return [0.5 * (s0 + s1), 0.5 * (s0 - s1), 0.0, a1, 0.0]
    r1, r2 = _quad_roots(den[0], den[1], den[2])
    z1, z2 = cmath.exp(r1 * w0 * T), cmath.exp(r2 * w0 * T)
    a1, a2 = -(z1 + z2).real, (z1 * z2).real
    A0, A1, A2 = (1.0 + a1 + a2) ** 2, (1.0 - a1 + a2) ** 2, -4.0 * a2

    def W(w):   # digital radians -> normalised analog frequency
        return w * fs / w0

    def Aw(w):
        p0, p1, p2 = _phis(w)
        return A0 * p0 + A1 * p1 + A2 * p2

    B0 = _mag2(num, den, 1e-9) * A0
    B1 = _mag2(num, den, W(math.pi)) * A1
    s0, s1 = math.sqrt(max(B0, 0.0)), math.sqrt(max(B1, 0.0))
    half = 0.5 * (s0 + s1)
    wb = 2.0 * math.pi * f0 / fs
    lo = 2.0 * math.pi * GRID_LOW_HZ / fs
    hi = 0.98 * math.pi
    grid = [lo * (hi / lo) ** (i / (GRID_POINTS - 1)) for i in range(GRID_POINTS)] + [min(wb, hi)]
    target = [10.0 * math.log10(_mag2(num, den, W(w)) + 1e-30) for w in grid]

    best = None
    for wm in (wb, 0.5 * math.pi, 0.5 * wb, min(1.5 * wb, 0.95 * math.pi), 0.25 * math.pi,
               0.75 * math.pi, min(wb, 0.9 * math.pi)):
        if not (1e-6 < wm < 0.97 * math.pi):
            continue
        p0, p1, p2 = _phis(wm)
        B2 = (_mag2(num, den, W(wm)) * Aw(wm) - B0 * p0 - B1 * p1) / p2
        b0 = 0.5 * (half + math.sqrt(max(half * half + B2, 0.0)))
        b1 = 0.5 * (s0 - s1)
        b2 = half - b0
        e = max(abs(max(_db_of(b0, b1, b2, a1, a2, w), FLOOR_DB) - max(t, FLOOR_DB)) for w, t in zip(grid, target))
        if best is None or e < best[0] - TIE_DB:
            best = (e, b0, b1, b2)
    if num[0] != 0.0:                                             # the zeros by the matched z-transform
        q1, q2 = _quad_roots(num[0], num[1], num[2])
        y1, y2 = cmath.exp(q1 * w0 * T), cmath.exp(q2 * w0 * T)
        c0, c1, c2 = 1.0, -(y1 + y2).real, (y1 * y2).real
        shape = [_db_of(c0, c1, c2, a1, a2, w) for w in grid]
        for wref in (lo, hi, min(wb, hi)):
            here = _db_of(c0, c1, c2, a1, a2, wref)
            want = 10.0 * math.log10(_mag2(num, den, W(wref)) + 1e-30)
            if here < -200.0 or want < -200.0:
                continue
            k = 10.0 ** ((want - here) / 20.0)
            e = max(abs(max(sh + 20.0 * math.log10(k), FLOOR_DB) - max(t, FLOOR_DB)) for sh, t in zip(shape, target))
            if e < best[0] - TIE_DB:
                best = (e, k * c0, k * c1, k * c2)
    _, b0, b1, b2 = best
    if num[0] == 1.0 and num[1] == 0.0 and num[2] == 1.0:          # a notch: its zeros on the circle
        c = math.cos(wb)
        g0 = math.sqrt(_mag2(num, den, 1e-9)) * (1.0 + a1 + a2) / (2.0 - 2.0 * c)
        b0, b1, b2 = g0, -2.0 * c * g0, g0
    elif num[0] == 0.0 and num[2] == 0.0:                          # a band-pass: zeros at DC and Nyquist,
        k = 1.0 / max(abs(_lin_of(1.0, 0.0, -1.0, a1, a2, wb)), 1e-30)   # its centre at 0 dB (measured)
        b0, b1, b2 = k, 0.0, -k
    return [b0, b1, b2, a1, a2]


def band_sections(shape: str, sample_rate: float, freq_hz: float, q: float, gain_db: float,
                  slope_db_oct: int = 6) -> list[tuple[np.ndarray, np.ndarray]]:
    """One EQ band as [(b, a)] for lfilter / freqz, at the native's clamps."""
    fs = float(sample_rate)
    f0 = min(max(float(freq_hz), 2.0), fs * 0.49)
    out = []
    for num, den in analog_sections(shape, gain_db, q, int(slope_db_oct or 6)):
        b0, b1, b2, a1, a2 = matched_section(num, den, f0, fs)
        out.append((np.array([b0, b1, b2]), np.array([1.0, a1, a2])))
    return out


def legacy_band(shape: str, q: float, gain_db: float, slope_db_oct) -> tuple[float, float]:
    """(q, gain_db) under this law for a band stored under the cookbook law, sounding the same:
    exact for a bell, notch, band-pass, tilt gain and every cut; a shelf's Q through the measured
    curve, held to Q_MAX (a cookbook shelf over ~4 has no equal here)."""
    q, g = float(q), float(gain_db)
    slope = int(slope_db_oct or 6)
    if shape in ("peak", "notch", "bandpass"):
        q = q * math.sqrt(2.0)
    elif shape in ("low_shelf", "high_shelf", "tilt_shelf"):
        q = (max(q, 1e-6) / SHELF_Q_SCALE) ** (1.0 / SHELF_Q_POWER)
        if shape == "tilt_shelf":
            g = g / 2.0
    elif shape in ("hpf", "lpf"):
        if slope == 12:
            q = q * math.sqrt(2.0)
        elif slope > 12:
            q = 1.0                     # the old law ran a steep cut Butterworth whatever its q
    return min(max(q, Q_MIN), Q_MAX), g


def convert_legacy(obj) -> int:
    """Walk anything read from an old file and convert every eq band in it in place: a dict of
    type "eq" holding a `band` (the brain's entry and action) or flat fields (the DAW's entry).
    Returns how many bands it converted."""
    n = 0
    if isinstance(obj, dict):
        if obj.get("type") == "eq":
            band = obj.get("band") if isinstance(obj.get("band"), dict) else obj
            if "shape" in band and "q" in band:
                band["q"], band["gain_db"] = legacy_band(band["shape"], band["q"],
                                                         band.get("gain_db", 0.0),
                                                         band.get("slope_db_oct", 6))
                n += 1
        for v in obj.values():
            if isinstance(v, (dict, list)):
                n += convert_legacy(v)
    elif isinstance(obj, list):
        for v in obj:
            n += convert_legacy(v)
    return n
