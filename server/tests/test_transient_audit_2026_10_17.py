"""The transient shaper's audit, 2026-10-17 (the device-by-device audit) - the offline engine.

The native half is `--transient-audit`. Here:
  * the offline shaper IS the DAW's law, sample for sample: a fast and a slow follower of the
    linked per-sample peak, LINEAR from rest, compared in dB (offline it read a 5 ms block RMS
    smoothed in dB; the native smoothed the per-sample level in dB, so every zero crossing put
    -240 dB into both followers);
  * a steady tone is not a transient: its gain moves less than 1 dB at full attack;
  * the reference monitor's shaper is the same function;
  * a value stored past the table plays at the table's end in both engines.
"""
from __future__ import annotations

import math

import numpy as np

from mix_engine import dsp, reference_monitor

SR = 48000


def _native(x: np.ndarray, sr: int, e: dict) -> np.ndarray:
    """`MonitorChain::applyTransient`, transcribed per sample (float64). x is (frames, ch)."""
    sp = e.get("speed", 1.0)
    fa, fr, sa, sl = (math.exp(-1 / (sr * ms / sp / 1000))
                      for ms in (dsp.TRANSIENT_FAST_ATTACK_MS, dsp.TRANSIENT_FAST_RELEASE_MS,
                                 dsp.TRANSIENT_SLOW_ATTACK_MS, dsp.TRANSIENT_SLOW_RELEASE_MS))
    f = s = 0.0
    hold = int(sr * dsp.TRANSIENT_HOLD_MS / 1000.0)
    raw = np.max(np.abs(x), axis=1)
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        pk = float(np.max(raw[max(0, i - hold):i + 1]))     # the peak hold: [i - hold, i]
        c = fa if pk > f else fr
        f = c * f + (1 - c) * pk
        c = sa if pk > s else sl
        s = c * s + (1 - c) * pk
        d = 20 * math.log10(max(f, 1e-6)) - 20 * math.log10(max(s, 1e-6))
        tns = min(1.0, max(0.0, (d - dsp.TRANSIENT_DEADBAND_DB)
                           / (dsp.TRANSIENT_FULL_DB - dsp.TRANSIENT_DEADBAND_DB)))
        g = e["attack_db"] * tns + e["sustain_db"] * (1 - tns) + e.get("output_db", 0.0)
        out[i] = x[i] * 10 ** (g / 20)
    return out


def _kicks(seconds: float = 0.6) -> np.ndarray:
    n = int(SR * seconds)
    t = np.arange(n) / SR
    k = np.zeros(n)
    for start in np.arange(0.05, seconds, 0.25):
        m = t >= start
        k[m] += 0.7 * np.sin(2 * np.pi * 60.0 * (t[m] - start)) * np.exp(-(t[m] - start) / 0.08)
    rng = np.random.default_rng(3)
    snare = np.zeros(n)
    for start in np.arange(0.17, seconds, 0.25):
        i0 = int(start * SR)
        burst = rng.standard_normal(min(4800, n - i0)) * 0.3 * np.exp(-np.arange(min(4800, n - i0)) / 1200)
        snare[i0:i0 + len(burst)] += burst
    return np.stack([k + snare, k + 0.8 * snare], axis=1)


ENTRY = {"attack_db": 5.0, "sustain_db": -3.0, "speed": 1.6, "output_db": 1.0}


def test_the_offline_shaper_is_the_daws_law_sample_for_sample():
    x = _kicks()
    offline = np.asarray(dsp._transient(x, SR, ENTRY), dtype=np.float64)
    native = _native(x, SR, ENTRY)
    assert np.max(np.abs(native)) > 0.3
    assert np.max(np.abs(offline - native)) < 1e-5, float(np.max(np.abs(offline - native)))


def _steady_lift_db(signal: np.ndarray) -> float:
    """The largest gain full attack gives a steady signal past its first half second."""
    x = np.stack([signal, signal], axis=1)
    out = np.asarray(dsp._transient(x, SR, {"attack_db": 6.0, "sustain_db": 0.0}))
    half = slice(SR // 2, len(signal))
    big = np.abs(x[half, 0]) > 0.1 * np.max(np.abs(signal))
    return float(np.max(20 * np.log10(np.abs(out[half, 0])[big] / np.abs(x[half, 0])[big])))


def test_a_held_note_is_not_a_transient():
    """James's pick (2026-10-17): the detector's peak hold + deadband. Without them a 60 Hz tone
    gained 0.8 dB at full attack, a 1 kHz tone 0.4, a two-note bass chord 2.6, noise ~2; with
    them the three read 0."""
    t = np.arange(2 * SR) / SR
    rng = np.random.default_rng(1)
    for name, sig in (("60 Hz", 0.5 * np.sin(2 * np.pi * 60 * t)),
                      ("1 kHz", 0.5 * np.sin(2 * np.pi * 1000 * t)),
                      ("a bass chord", 0.3 * np.sin(2 * np.pi * 55 * t) + 0.3 * np.sin(2 * np.pi * 82.4 * t))):
        lift = _steady_lift_db(sig)
        assert lift < 0.05, (name, lift)
    # noise has peaks of its own: its rarest are still lifted a little (0.7 dB), its level is not
    noise = rng.standard_normal(2 * SR) * 0.2
    assert _steady_lift_db(noise) < 1.0
    x = np.stack([noise, noise], axis=1)
    out = np.asarray(dsp._transient(x, SR, {"attack_db": 6.0, "sustain_db": 0.0}))
    half = slice(SR // 2, 2 * SR)
    assert 10 * np.log10(np.sum(out[half, 0] ** 2) / np.sum(x[half, 0] ** 2)) < 0.05


def test_a_kick_still_gets_its_attack():
    """The price, measured and taken: a kick's first 20 ms at +6 attack gets ~+4.2 dB (+5.6
    before the hold) - and a hit is still a hit."""
    t = np.arange(SR) / SR
    kick = np.zeros(SR)
    starts = np.arange(0.1, 1.0, 0.25)
    for s0 in starts:
        k = t >= s0
        kick[k] += 0.7 * np.sin(2 * np.pi * 60 * (t[k] - s0)) * np.exp(-(t[k] - s0) / 0.08)
    x = np.stack([kick, kick], axis=1)
    out = np.asarray(dsp._transient(x, SR, {"attack_db": 6.0, "sustain_db": 0.0}))
    lifts = []
    for s0 in starts:
        a = int(s0 * SR)
        lifts.append(10 * np.log10(np.sum(out[a:a + 960, 0] ** 2) / np.sum(x[a:a + 960, 0] ** 2)))
    assert 3.5 < float(np.mean(lifts)) < 5.0, lifts


def test_the_reference_monitors_shaper_is_the_same_function():
    x = _kicks(0.3)
    offline = np.asarray(dsp._transient(x, SR, ENTRY), dtype=np.float64)
    monitor = reference_monitor._transient(x, SR, ENTRY["attack_db"], ENTRY["sustain_db"],
                                           ENTRY["speed"], ENTRY["output_db"])
    assert np.max(np.abs(offline - monitor)) < 1e-9


def test_a_value_past_the_table_plays_at_its_end():
    x = _kicks(0.3)
    at_end = dsp._transient(x, SR, {"attack_db": 6.0, "sustain_db": -6.0, "speed": 4.0, "output_db": 12.0})
    past = dsp._transient(x, SR, {"attack_db": 30.0, "sustain_db": -30.0, "speed": 40.0, "output_db": 30.0})
    assert np.max(np.abs(np.asarray(past) - np.asarray(at_end))) < 1e-9


def test_every_live_reading_a_device_row_carries_is_a_meter_field():
    """A field of the device row that moves while audio plays is not an edit: the brain leaves
    `edit_state.METER_FIELDS` out of the delta. The multiband's and OTT's `gr_bands_db` (this
    audit's day) were not in it, so every snapshot read as a change to those devices."""
    import re
    from pathlib import Path

    from orchestrator.edit_state import METER_FIELDS

    src = (Path(__file__).resolve().parents[2] / "daw" / "Source" / "MonitorChainPlugin.h").read_text()
    for name in ("gr_bands_db", "transient", "shape_db", "move_db", "band_level_db", "band_levels_db"):
        assert re.search(r'setProperty \("%s"' % name, src), name
        assert name in METER_FIELDS, name
