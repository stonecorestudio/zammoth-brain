"""OTT's audit, 2026-10-17 (the device-by-device audit) - the offline engine.

The native half is `--ott-audit`. Here:
  * the offline OTT IS the DAW's law, sample for sample: the real 3-way LR4 split (the native
    split by the complement `mid = dry - low - high`, so levelling one band moved the tone in
    another, and offline split by a tapered FFT partition), and each band's linked per-sample
    peak followed LINEARLY from rest (offline it was a 5 ms block RMS smoothed in dB);
  * the reference monitor's OTT is the same function (it was a third copy of the old law);
  * a gain stored past the table's +-12 dB plays at 12 in both engines.
"""
from __future__ import annotations

import math

import numpy as np

from mix_engine import dsp, reference_monitor
from mix_engine.reference_monitor import biquad_coeffs

SR = 48000


def _native(x: np.ndarray, sr: int, e: dict) -> np.ndarray:
    """`MonitorChain::applyOtt`, transcribed per sample (float64)."""
    nch, n = x.shape
    lo_x, hi_x = e.get("crossover_lo_hz", 150.0), e.get("crossover_hi_hz", 2000.0)
    co = {}
    for key, shape, f in (("lpLo", "lpf2", lo_x), ("hpLo", "hpf2", lo_x),
                          ("lpHi", "lpf2", hi_x), ("hpHi", "hpf2", hi_x)):
        b, a = biquad_coeffs(shape, sr, f, 0.0, 0.0)
        co[key] = (np.asarray(b) / a[0], np.asarray(a) / a[0])
    st = {k: np.zeros((nch, 2, 2)) for k in ("lpLo", "hpLo", "lpHi", "hpHi", "apLp", "apHp")}

    def lr4(xv, key, sk, ch):
        b, a = co[key]
        z = st[sk][ch]
        y0 = b[0] * xv + z[0][0]
        z[0][0] = b[1] * xv - a[1] * y0 + z[0][1]
        z[0][1] = b[2] * xv - a[2] * y0
        y1 = b[0] * y0 + z[1][0]
        z[1][0] = b[1] * y0 - a[1] * y1 + z[1][1]
        z[1][1] = b[2] * y0 - a[2] * y1
        return y1

    t = e.get("time", 1.0)
    a_att = math.exp(-1 / (sr * dsp.OTT_BASE_ATTACK_MS * t / 1000))
    a_rel = math.exp(-1 / (sr * dsp.OTT_BASE_RELEASE_MS * t / 1000))
    x = x * 10 ** (min(12.0, max(-12.0, e.get("in_gain_db", 0.0))) / 20)
    env = [0.0] * 3
    out = np.zeros_like(x)
    for i in range(n):
        bands = np.zeros((3, nch))
        for ch in range(nch):
            xv = x[ch, i]
            low_raw = lr4(xv, "lpLo", "lpLo", ch)
            rest = lr4(xv, "hpLo", "hpLo", ch)
            bands[1, ch] = lr4(rest, "lpHi", "lpHi", ch)
            bands[2, ch] = lr4(rest, "hpHi", "hpHi", ch)
            bands[0, ch] = lr4(low_raw, "lpHi", "apLp", ch) + lr4(low_raw, "hpHi", "apHp", ch)
        g = []
        for b, name in enumerate(("low", "mid", "high")):
            p = e.get(name) or {}
            pk = float(np.max(np.abs(bands[b])))
            c = a_att if pk > env[b] else a_rel
            env[b] = c * env[b] + (1 - c) * pk
            d = dsp.OTT_TARGET_DB - 20 * math.log10(max(env[b], 1e-6))
            boost = min(max(d, 0.0) * dsp.OTT_UP_SLOPE, dsp.OTT_MAX_BOOST_DB) * p.get("up", 1.0)
            cut = max(min(d, 0.0) * dsp.OTT_DOWN_SLOPE, -dsp.OTT_MAX_CUT_DB) * p.get("down", 1.0)
            g.append(10 ** (e.get("depth", 0.5) * (boost + cut) / 20) * 10 ** (p.get("gain_db", 0.0) / 20))
        out[:, i] = (bands[0] * g[0] + bands[1] * g[1] + bands[2] * g[2]) * 10 ** (e.get("out_gain_db", 0.0) / 20)
    return out


def _hits(seconds: float = 0.25) -> np.ndarray:
    """A kick (60 Hz, decaying) every 120 ms over quiet hats - a band to press down, and a
    quiet band to lift."""
    n = int(SR * seconds)
    t = np.arange(n) / SR
    kick = np.zeros(n)
    for start in np.arange(0.0, seconds, 0.12):
        k = t >= start
        kick[k] += 0.7 * np.sin(2 * np.pi * 60.0 * (t[k] - start)) * np.exp(-(t[k] - start) / 0.05)
    rng = np.random.default_rng(7)
    hats = np.diff(rng.standard_normal(n + 1)) * 0.004
    return np.stack([kick + hats, kick + np.roll(hats, 3)]).astype(np.float32)


ENTRY = {"type": "ott", "depth": 0.6, "time": 1.3, "in_gain_db": 2.0, "out_gain_db": -1.0,
         "low": {"up": 0.7, "down": 1.0, "gain_db": 1.5}, "mid": {},
         "high": {"up": 1.0, "down": 0.4}, "crossover_lo_hz": 120.0, "crossover_hi_hz": 3000.0}


def test_the_offline_ott_is_the_daws_law_sample_for_sample():
    x = _hits()
    offline = dsp._ott(x, SR, ENTRY).astype(np.float64)
    native = _native(x.astype(np.float64), SR, ENTRY)
    assert np.max(np.abs(native)) > 0.1
    assert np.max(np.abs(offline - native)) < 1e-4, float(np.max(np.abs(offline - native)))


def test_the_reference_monitors_ott_is_the_same_function():
    x = _hits(0.1)
    offline = dsp._ott(x, SR, ENTRY).astype(np.float64)
    monitor = reference_monitor._ott(
        x.T.astype(np.float64), SR, ENTRY["depth"], ENTRY["time"], ENTRY["in_gain_db"],
        ENTRY["out_gain_db"], ENTRY["low"], ENTRY["mid"], ENTRY["high"],
        ENTRY["crossover_lo_hz"], ENTRY["crossover_hi_hz"]).T
    assert np.max(np.abs(offline - monitor)) < 1e-6


def test_a_gain_past_the_table_plays_at_its_end():
    x = _hits(0.1)
    flat = {"up": 0.0, "down": 0.0}
    at_12 = dsp._ott(x, SR, {"depth": 0.0, "in_gain_db": 12.0, "low": flat, "mid": flat, "high": flat})
    at_30 = dsp._ott(x, SR, {"depth": 0.0, "in_gain_db": 30.0, "low": flat, "mid": flat, "high": flat})
    assert np.max(np.abs(at_30 - at_12)) < 1e-6
