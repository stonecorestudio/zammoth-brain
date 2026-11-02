"""The two-stage limiter (2026-10-25; James picked CLEAN / LOUD / GLUE and face A).

The limiter's gain is the lower of two envelopes over the same lookahead window:

* the FAST stage catches every peak - attack across the lookahead (six constants), release in
  milliseconds (the character's fast release);
* the SLOW stage carries the held level - it follows the window's minimum held for the
  character's hold, with the character's attack and `release_ms` as its release.

`_native_loop` is `MonitorChain::applyLookaheadLimiter` transcribed step by step (the fixed 3 ms
delay, the target window [leaving, leaving + L], the two one-poles, the slow stage's hold, the gain
floor, the ceiling clamp); `dsp._brickwall` must be that law. The rest is what the redesign is for:
a short hit is let go far sooner than a held over, the lookahead knob changes the sound, and the
three characters differ. The native half is `--limiter-two-stage`.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import resample

from mix_engine import dsp
from mix_engine.models import (LIMITER_CHARACTERS, LIMITER_STYLE_PRESETS, LIMITER_STYLES,
                               LimiterAction, limiter_knobs)
from mix_engine.tp_detector import TP_CENTRE, intersample_peak

SR = 48000
ROOT = Path(__file__).resolve().parents[2]


def _native_loop(x: np.ndarray, e: dict) -> np.ndarray:
    n, nch = x.shape
    c = 10 ** (e["ceiling_dbtp"] / 20)
    rel, look, style = limiter_knobs(e)
    fast_rel, slow_att, hold_ms = LIMITER_CHARACTERS[style]
    ring = max(1, int(SR * 0.003))
    h = 2                                                   # kLimSmoothHalf
    L = min(max(1, round(SR * look / 1000)), ring - TP_CENTRE - 2 * h)
    H = round(SR * hold_ms / 1000)
    link = e.get("channel_link", 1.0)
    ac = 1 - math.exp(-6 / L)
    fc = 1 - math.exp(-1 / (SR * fast_rel / 1000))
    sa = 1 - math.exp(-1 / (SR * slow_att / 1000))
    sr_ = 1 - math.exp(-1 / (SR * max(1.0, rel) / 1000))
    pk = intersample_peak(x)
    target = np.where(pk > c, c / np.maximum(pk, 1e-30), 1.0)
    loud = np.max(pk, axis=1, keepdims=True)
    target = target * (1 - link) + np.where(loud > c, c / np.maximum(loud, 1e-30), 1.0) * link

    def t_at(m, ch):
        return target[m, ch] if 0 <= m < n else 1.0
    out = np.zeros_like(x)
    fast = [1.0] * nch
    slow = [1.0] * nch
    seen = [[] for _ in range(nch)]
    raw = [{} for _ in range(nch)]                          # sample -> its raw gain
    for s in range(n + ring):
        m = s - ring                                        # the sample leaving the delay
        j = m + 2 * h                                       # the sample whose raw gain is made now
        for ch in range(nch):
            fwd = min(t_at(k, ch) for k in range(j, j + L + 1))
            fast[ch] += (fwd - fast[ch]) * (ac if fwd < fast[ch] else fc)
            seen[ch].append(fwd)
            held = min(seen[ch][-(H + 1):])
            slow[ch] += (held - slow[ch]) * (sa if held < slow[ch] else sr_)
            raw[ch][j] = min(fast[ch], slow[ch], min(t_at(k, ch) for k in range(j - 2, j + 3)))
            if 0 <= m < n:
                r = lambda k: raw[ch].get(k, 1.0)
                mf = [min(r(i) for i in range(k - h, k + h + 1)) for k in range(m - h, m + h + 1)]
                g = sum(mf) / len(mf)
                out[m, ch] = max(-c, min(c, x[m, ch] * g))
    return out


def _loud(n: int = 4000) -> np.ndarray:
    rng = np.random.default_rng(1)
    t = np.arange(n) / SR
    left = np.sin(2 * np.pi * 220 * t) * (0.5 + 0.8 * (t % 0.05 < 0.01))
    right = 0.7 * np.sin(2 * np.pi * 330 * t + 1) + 0.3 * rng.standard_normal(n)
    return np.stack([left, right], axis=1) * 1.6


@pytest.mark.parametrize("entry", [
    {"ceiling_dbtp": -1.0},                                                          # CLEAN, its own knobs
    {"ceiling_dbtp": -1.0, "style": 2},                                              # LOUD (no hold)
    {"ceiling_dbtp": -2.0, "style": 1, "channel_link": 0.4},                         # GLUE
    {"ceiling_dbtp": -3.0, "style": 0, "release_ms": 40.0, "lookahead_ms": 2.9},     # past the window: clamped
])
def test_the_offline_limiter_is_the_native_two_stage_law(entry):
    x = _loud()
    assert np.max(np.abs(dsp._brickwall(x, SR, entry) - _native_loop(x, entry))) < 2e-4


def _true_peak_db(y: np.ndarray) -> float:
    """The EXACT peak between samples: the ideal band-limited reconstruction (an FFT, 16 points a
    sample), 20 ms off each end. A windowed resampler shares the detector's blind spot and read
    -0.988 where this reads -0.993 (CLAUDE.md: an exact claim needs the FFT meter)."""
    up = resample(y, len(y) * 16, axis=0)
    e = 16 * 960
    return 20 * math.log10(float(np.max(np.abs(up[e:-e]))))


def test_the_meter_reads_a_tone_between_the_samples():
    t = np.arange(SR) / SR
    tone = 0.5 * np.sin(2 * np.pi * 11997 * t + 0.3)
    assert abs(_true_peak_db(np.stack([tone, tone], axis=1)) - 20 * math.log10(0.5)) < 1e-3


def _burst(dur_s: float, over_db: float = 9.0) -> tuple[np.ndarray, int]:
    pre, post = 0.3, 1.5
    t = np.arange(int((pre + dur_s + post) * SR)) / SR
    a = np.where((t >= pre) & (t < pre + dur_s), 10 ** (over_db / 20), 10 ** (-40 / 20))
    y = a * np.sin(2 * np.pi * 1000 * t)
    return np.stack([y, y], axis=1), int((pre + dur_s) * SR)


def _recovery_ms(y: np.ndarray, x: np.ndarray, end: int, to_db: float = -1.0) -> float:
    """Time after the burst ends until the gain is back within `to_db` of none (5 ms RMS)."""
    w = int(0.005 * SR)
    k = np.ones(w) / w
    gx = np.sqrt(np.convolve(x[:, 0] ** 2, k, "same"))
    gy = np.sqrt(np.convolve(y[:, 0] ** 2, k, "same"))
    gain = 20 * np.log10(np.maximum(gy, 1e-9) / np.maximum(gx, 1e-9)) + 1.0   # +1: the ceiling's own -1
    after = gain[end + w:]
    i = int(np.argmax(after > to_db))
    return (i + w) / SR * 1000


def test_a_short_hit_is_let_go_long_before_a_held_over():
    """The point of the redesign. One release for every material ducked a mix for 95 ms after each
    drum hit and let a held over go too soon; the reference limiter we measured lets a 5 ms hit go
    in about 3 ms and a held over in 300-1000 ms."""
    e = {"ceiling_dbtp": -1.0, "style": 0}
    xs, end_s = _burst(0.005)
    xl, end_l = _burst(1.0)
    short = _recovery_ms(dsp._brickwall(xs, SR, e), xs, end_s)
    held = _recovery_ms(dsp._brickwall(xl, SR, e), xl, end_l)
    assert short < 30.0, short
    assert held > 300.0, held


@pytest.mark.parametrize("style", [0, 1, 2])
def test_every_character_holds_the_ceiling(style):
    # band-limited under 16 kHz: energy above 20 kHz is the detector's known residual (CLAUDE.md,
    # "the limiter's residual is energy above 20 kHz"), which no character is asked to hold
    from scipy.signal import butter, sosfiltfilt
    x = sosfiltfilt(butter(8, 16000, fs=SR, output="sos"), _loud(24000), axis=0) * 1.8
    y = dsp._brickwall(x, SR, {"ceiling_dbtp": -1.0, "style": style})
    assert _true_peak_db(y) <= -0.99


def test_the_lookahead_knob_changes_the_sound():
    """It did nothing: the reduction always began the whole 3 ms delay ahead of a peak. A shorter
    window starts later, so less of the music around a peak is turned down - louder."""
    x = _loud(24000) * 2.5
    rms = lambda y: float(np.sqrt(np.mean(y ** 2)))
    short = dsp._brickwall(x, SR, {"ceiling_dbtp": -1.0, "lookahead_ms": 0.5})
    long_ = dsp._brickwall(x, SR, {"ceiling_dbtp": -1.0, "lookahead_ms": 2.5})
    assert rms(short) > rms(long_) * 1.01


def test_the_three_characters_sound_different():
    x = _loud(24000) * 2.5
    outs = [dsp._brickwall(x, SR, {"ceiling_dbtp": -1.0, "style": s}) for s in range(3)]
    for a in range(3):
        for b in range(a + 1, 3):
            assert np.max(np.abs(outs[a] - outs[b])) > 1e-3, (a, b)


def test_a_style_name_is_a_character_and_fills_its_knobs():
    act = LimiterAction(stem_id="master", style="glue")
    assert act.style == 1 and (act.release_ms, act.lookahead_ms) == LIMITER_STYLE_PRESETS[1]
    # the order keeps a saved project's meaning: Modern (0) -> clean, Transparent (1) -> glue,
    # Punchy (2) -> loud
    assert LIMITER_STYLES == ("clean", "glue", "loud")
    with pytest.raises(ValueError):
        LimiterAction(stem_id="master", style="modern")


def test_the_native_tables_are_the_models():
    hdr = (ROOT / "native/Source/MonitorChain.h").read_text()
    row = re.search(r"kLimCharacters\[3\]\[3\]\s*=\s*\{(.*?)\};", hdr, re.S).group(1)
    triples = [tuple(float(v.rstrip("f")) for v in p.split(",")) for p in re.findall(r"\{\s*([^{}]+?)\s*\}", row)]
    assert triples == [LIMITER_CHARACTERS[i] for i in range(3)]
    specs = (ROOT / "native/Source/DeviceSpecs.h").read_text()
    assert '{ "clean", "glue", "loud" }' in specs


def test_the_native_fallbacks_are_the_characters():
    """A stored limiter with no release / lookahead plays its character's (limiter_knobs), at both
    ends: the native's getProperty fallbacks read the preset table, never a literal."""
    src = (ROOT / "native/Source/MonitorChain.cpp").read_text()
    block = src[src.index('t == "limiter" && limSlot >= 0'):][:1400]
    assert "kLimStylePresets" in block
    assert 'getProperty ("release_ms", 50.0)' not in block


def test_the_gain_knob_drives_the_limiter():
    """GAIN (James: "where is the gain knob like pro l has?") is the drive into the limiter, before
    its detector, on every limiter: the same as turning the input up by that much."""
    x = _loud(8000) * 0.4
    e = {"ceiling_dbtp": -1.0}
    driven = dsp._brickwall(x, SR, {**e, "gain_db": 9.0})
    assert np.max(np.abs(driven - dsp._brickwall(x * 10 ** (9.0 / 20), SR, e))) < 1e-12
    assert np.max(np.abs(dsp._brickwall(x, SR, {**e, "gain_db": 0.0}) - dsp._brickwall(x, SR, e))) == 0.0


def test_a_new_limiter_starts_on_cleans_knobs_at_every_door():
    """The dock's new limiter wrote the OLD knobs (release 50, lookahead 3) - so a limiter added in
    the app played CLEAN's character at the old release. (The master's safety limiter, a second door
    that wrote them, was removed 2026-10-25.)"""
    specs = (ROOT / "native/Source/DeviceSpecs.h").read_text()
    row = specs[specs.index('else if (type == "limiter")    {'):][:300]
    assert 'p ("release_ms", 500)' in row and 'p ("lookahead_ms", 1.5)' in row and 'p ("gain_db", 0)' in row
    assert LIMITER_STYLE_PRESETS[0] == (500.0, 1.5)
    # each character sets a lookahead of its own, so the switch moves both knobs
    assert len({look for _, look in LIMITER_STYLE_PRESETS.values()}) == 3
