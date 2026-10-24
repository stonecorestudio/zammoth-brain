"""The offline compressor runs the NATIVE law (reader E4·R1, 2026-10-03).

Until this change `dsp._apply_comp` (and the multiband comp's downward bands)
was Pedalboard's compressor: `attack_ms` a time constant of ms/2π on the level,
per channel. The DAW's (`MonitorChain::processOneComp`) is a linked peak
detector with the ballistics on the gain reduction in dB and `attack_ms` a
literal time constant. Same numbers on the face, ~6x apart in time, and on
drums up to 4.7 dB apart in level — the agent measured one compressor and the
user heard and exported another.

`reference_monitor._compress` is the Python model of the native law (a plain
loop); these tests hold the offline engine to it, and hold the fast path
(`_gr_ballistics_db`, which runs the recursion inside Pedalboard's follower) to
the loop it replaces. Every test here but the two passthrough ones fails on the
Pedalboard law.
"""
from __future__ import annotations

import numpy as np
import pytest

from mix_engine import dsp, reference_monitor

SR = 48000


def _db(x: float) -> float:
    return 20.0 * np.log10(max(float(x), 1e-12))


def _drum_bursts(seconds: float = 2.0, sr: int = SR, seed: int = 7) -> np.ndarray:
    """Decaying noise bursts over a quiet bed, stereo with the hits off-centre —
    transient material, which is what a compressor is for. (channels, frames)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    left = 0.02 * rng.standard_normal(n)
    right = 0.02 * rng.standard_normal(n)
    for k, start in enumerate(np.arange(0.1, seconds - 0.2, 0.23)):
        i = int(start * sr)
        length = int(0.12 * sr)
        hit = rng.standard_normal(length) * np.exp(-np.arange(length) / (0.02 * sr))
        hit += 0.8 * np.sin(2 * np.pi * 60.0 * t[:length]) * np.exp(-np.arange(length) / (0.05 * sr))
        pan = 0.3 + 0.4 * (k % 3) / 2.0
        left[i:i + length] += 0.7 * hit * (1.0 - pan)
        right[i:i + length] += 0.7 * hit * pan
    return np.ascontiguousarray(np.stack([left, right]).astype(np.float32))


def _entry(**over) -> dict:
    entry = {"type": "comp", "threshold_db": -24.0, "ratio": 4.0, "attack_ms": 10.0,
             "release_ms": 100.0, "makeup_db": 0.0, "mix": 1.0}
    entry.update(over)
    return entry


@pytest.mark.parametrize("sr", [44100, 48000, 96000])
@pytest.mark.parametrize("attack_ms,release_ms",
                         [(10.0, 100.0), (0.1, 10.0), (1.0, 1000.0), (100.0, 1000.0), (30.0, 250.0)])
def test_the_fast_ballistics_are_the_loop(sr, attack_ms, release_ms):
    """The pin on Pedalboard's internals: if its compressor's follower ever stops
    being `y = x + cte·(y − x)` at `exp(-2π·1000/(sr·ms))`, this goes red."""
    rng = np.random.default_rng(3)
    n = sr // 2
    t = np.arange(n) / sr
    level = np.abs(np.sin(2 * np.pi * 110.0 * t) * (0.2 + 0.8 * (np.sin(2 * np.pi * 6.0 * t) > 0))
                   + 0.2 * rng.standard_normal(n))
    over = 20.0 * np.log10(np.maximum(level, 1e-12)) + 24.0
    target = np.where(over > 0.0, -over * 0.75, 0.0)
    fast = dsp._gr_ballistics_db(target, sr, attack_ms, release_ms)
    loop = dsp._gr_ballistics_reference(target, sr, attack_ms, release_ms)
    assert fast.shape == loop.shape
    assert float(loop.min()) < -6.0                       # the material does compress
    assert float(np.max(np.abs(fast - loop))) < 0.02      # float32 rounding, as in the native


def test_the_fast_ballistics_start_from_no_reduction_and_never_boost():
    target = np.concatenate([np.zeros(500), np.full(4000, -12.0), np.zeros(4000)])
    env = dsp._gr_ballistics_db(target, SR, 10.0, 100.0)
    assert np.max(np.abs(env[:500])) < 1e-3               # the lead-in left nothing behind
    assert float(env.max()) < 1e-3 and float(env.min()) > -12.0


@pytest.mark.parametrize("params", [
    dict(threshold_db=-24.0, ratio=4.0, attack_ms=10.0, release_ms=100.0),
    dict(threshold_db=-18.0, ratio=3.0, attack_ms=10.0, release_ms=120.0),
    dict(threshold_db=-30.0, ratio=6.0, attack_ms=5.0, release_ms=60.0),
    dict(threshold_db=-20.0, ratio=4.0, attack_ms=30.0, release_ms=250.0, makeup_db=3.0, mix=0.6),
])
def test_the_offline_comp_is_the_native_law_on_drums(params):
    audio = _drum_bursts()
    offline = dsp._apply_comp(audio, SR, _entry(**params))
    native = reference_monitor._compress(
        audio.T.astype(np.float64), SR, params["threshold_db"], params["ratio"],
        params["attack_ms"], params["release_ms"],
        makeup_db=params.get("makeup_db", 0.0), mix=params.get("mix", 1.0)).T
    rms_offline = float(np.sqrt(np.mean(offline.astype(np.float64) ** 2)))
    rms_native = float(np.sqrt(np.mean(native ** 2)))
    assert abs(_db(rms_offline) - _db(rms_native)) < 0.01   # it was +0.96 … +4.67 dB
    assert float(np.max(np.abs(offline - native))) < 2e-3
    # ...and it did compress: the comparison is not two untouched signals.
    assert _db(float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))) - _db(rms_offline) \
        > (0.5 if params.get("mix", 1.0) == 1.0 and not params.get("makeup_db") else -10.0)


def test_attack_and_release_are_literal_time_constants():
    """A step from −40 to −6 dBFS: the reduction is 63 % of the way in after
    `attack_ms`, and 63 % of the way out `release_ms` after the step ends. On
    the Pedalboard law those were 0.96 ms and 21.7 ms."""
    n = SR
    step_on, step_off = int(0.2 * SR), int(0.6 * SR)
    level = np.full(n, 10.0 ** (-40.0 / 20.0))
    level[step_on:step_off] = 10.0 ** (-6.0 / 20.0)
    audio = np.ascontiguousarray(np.stack([level, level]).astype(np.float32))   # DC: the level IS the signal
    out = dsp._apply_comp(audio, SR, _entry())
    gr_db = 20.0 * np.log10(out[0].astype(np.float64) / audio[0].astype(np.float64))
    full = (-6.0 + 24.0) * 0.75                                # 13.5 dB of reduction
    assert gr_db[step_off - 1] == pytest.approx(-full, abs=0.05)
    attack_at = int(np.argmax(gr_db[step_on:] <= -full * (1.0 - np.exp(-1.0)))) / SR * 1000.0
    release_at = int(np.argmax(gr_db[step_off:] >= -full * np.exp(-1.0))) / SR * 1000.0
    assert attack_at == pytest.approx(10.0, abs=0.3)
    assert release_at == pytest.approx(100.0, abs=1.5)


def test_the_detector_is_linked_across_the_channels():
    """A loud left and a quiet right: both come down by the SAME amount, so the
    image holds still. Per-channel (Pedalboard) left the right untouched."""
    t = np.arange(SR) / SR
    left = 10.0 ** (-6.0 / 20.0) * np.sin(2 * np.pi * 220.0 * t)
    right = 10.0 ** (-30.0 / 20.0) * np.sin(2 * np.pi * 220.0 * t)
    audio = np.ascontiguousarray(np.stack([left, right]).astype(np.float32))
    out = dsp._apply_comp(audio, SR, _entry(attack_ms=1.0, release_ms=200.0))
    tail = slice(SR // 2, SR)
    drop = [_db(np.sqrt(np.mean(audio[c, tail].astype(np.float64) ** 2)))
            - _db(np.sqrt(np.mean(out[c, tail].astype(np.float64) ** 2))) for c in (0, 1)]
    assert drop[0] > 10.0
    assert drop[1] == pytest.approx(drop[0], abs=0.05)


def test_a_side_comp_leaves_the_mid_alone_and_matches_the_native_ms_law():
    audio = _drum_bursts(seed=11)
    out = dsp._apply_comp(audio, SR, _entry(channel="side", threshold_db=-40.0))
    mid_in, mid_out = (audio[0] + audio[1]) * 0.5, (out[0] + out[1]) * 0.5
    side_in, side_out = (audio[0] - audio[1]) * 0.5, (out[0] - out[1]) * 0.5
    assert float(np.max(np.abs(mid_out - mid_in))) < 1e-6
    native_side = reference_monitor._compress(
        side_in.astype(np.float64)[:, np.newaxis], SR, -40.0, 4.0, 10.0, 100.0)[:, 0]
    assert float(np.max(np.abs(side_out - native_side))) < 2e-3
    assert float(np.sqrt(np.mean(side_out.astype(np.float64) ** 2))) \
        < 0.9 * float(np.sqrt(np.mean(side_in.astype(np.float64) ** 2)))


def test_under_the_threshold_and_at_ratio_one_nothing_is_touched():
    audio = _drum_bursts()
    assert dsp._apply_comp(audio, SR, _entry(ratio=1.0)) is audio
    quiet = (audio * np.float32(1e-4)).astype(np.float32)
    assert np.array_equal(dsp._apply_comp(quiet, SR, _entry()), quiet)


def test_an_empty_buffer_is_an_empty_buffer():
    empty = np.zeros((2, 0), dtype=np.float32)
    assert dsp._apply_comp(empty, SR, _entry()).shape == (2, 0)
    assert dsp._gr_ballistics_db(np.zeros(0), SR, 10.0, 100.0).shape == (0,)


def test_a_multiband_band_compresses_by_the_native_law_too():
    """The downward bands were the same Pedalboard compressor. A loud left and a
    quiet right inside one band come down together now."""
    t = np.arange(SR) / SR
    left = 10.0 ** (-6.0 / 20.0) * np.sin(2 * np.pi * 1000.0 * t)
    right = 10.0 ** (-30.0 / 20.0) * np.sin(2 * np.pi * 1000.0 * t)
    audio = np.ascontiguousarray(np.stack([left, right]).astype(np.float32))
    band = {"threshold_db": -24.0, "ratio": 4.0, "makeup_db": 0.0}
    still = {"threshold_db": 0.0, "ratio": 1.0, "makeup_db": 0.0}
    entry = {"type": "mb_comp", "attack_ms": 1.0, "release_ms": 200.0,
             "low": dict(still), "mid": dict(band), "high": dict(still)}
    out = dsp._multiband_comp(audio, SR, entry)
    tail = slice(SR // 2, SR - SR // 10)
    drop = [_db(np.sqrt(np.mean(audio[c, tail].astype(np.float64) ** 2)))
            - _db(np.sqrt(np.mean(out[c, tail].astype(np.float64) ** 2))) for c in (0, 1)]
    assert drop[0] > 10.0
    assert drop[1] == pytest.approx(drop[0], abs=0.3)
