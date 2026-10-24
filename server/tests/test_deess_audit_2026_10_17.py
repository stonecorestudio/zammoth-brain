"""The de-esser's audit, 2026-10-17 (the device-by-device audit) - the offline engine.

`dsp._deess` is the DAW's law, transcribed (`MonitorChain::applyDeess`): held here against a
literal per-sample loop of the native code, on noise-like sibilance (where the old block-peak
detector read several dB more reduction than the DAW plays). The native half is `--deess-audit`.
"""
from __future__ import annotations

import math

import numpy as np

from mix_engine import dsp

SR = 48000


def _native_loop(audio: np.ndarray, sr: int, e: dict) -> np.ndarray:
    """MonitorChain::applyDeess, line for line, in float32 (the LR4 split is dsp's own)."""
    low, high = dsp._lr4_split(audio, sr, e["freq_hz"])
    low, high = low.astype(np.float32), high.astype(np.float32)
    ratio = np.float32(max(1.0, e["ratio"]))
    a_att = np.float32(math.exp(-1.0 / (sr * e["attack_ms"] / 1000.0)))
    a_rel = np.float32(math.exp(-1.0 / (sr * e["release_ms"] / 1000.0)))
    thr, rng = np.float32(e["threshold_db"]), np.float32(e["range_db"])
    out = np.empty_like(low)
    env = np.float32(0.0)
    for i in range(low.shape[1]):
        peak = np.float32(max(abs(high[0, i]), abs(high[1, i])))
        level = np.float32(20.0) * np.float32(math.log10(max(float(peak), 1e-12)))
        over = level - thr
        target = np.float32(max(-over * (np.float32(1.0) - np.float32(1.0) / ratio) if over > 0 else 0.0, rng))
        coef = a_att if target < env else a_rel
        env = coef * env + (np.float32(1.0) - coef) * target
        g = np.float32(10.0 ** (float(env) / 20.0))
        out[:, i] = low[:, i] + high[:, i] * g
    return out


def _sibilance(seconds: float = 0.6) -> np.ndarray:
    rng = np.random.default_rng(7)
    n = int(SR * seconds)
    t = np.arange(n) / SR
    noise = rng.standard_normal(n)
    # a high-passed noise burst every 120 ms over a low tone: esses on a voice
    hp = np.diff(np.concatenate([[0.0], noise])) * 0.15
    bursts = (np.sin(2 * np.pi * t / 0.12) > 0.3).astype(float)
    x = 0.2 * np.sin(2 * np.pi * 180 * t) + hp * bursts
    return np.stack([x, 0.9 * x]).astype(np.float32)


ENTRY = {"type": "deess", "freq_hz": 6000.0, "threshold_db": -36.0, "ratio": 4.0,
         "release_ms": 50.0, "range_db": -12.0}


def test_the_offline_de_esser_is_the_native_law_on_sibilance():
    audio = _sibilance()
    for attack in (0.1, 1.0, 20.0):
        e = {**ENTRY, "attack_ms": attack}
        ours = dsp._deess(audio, SR, e)
        native = _native_loop(audio, SR, e)
        assert np.max(np.abs(ours - native)) < 2e-3, attack


def test_attack_below_a_quarter_millisecond_moves_the_offline_render():
    audio = _sibilance()
    fast = dsp._deess(audio, SR, {**ENTRY, "attack_ms": 0.1})
    quarter = dsp._deess(audio, SR, {**ENTRY, "attack_ms": 0.25})
    assert np.max(np.abs(fast - quarter)) > 1e-4    # it was bit-identical (audit 02 F4)


def test_a_track_takes_four_de_essers_and_refuses_a_fifth(demo_project):
    """James, 2026-10-17: up to four de-essers per track (two-band de-essing) - it was one."""
    import pytest

    from mix_engine.session import Session
    session = Session(demo_project)
    for _ in range(4):
        session.add_stem_device("stem_b", "deess")
    assert sum(e["type"] == "deess" for e in session.stems["stem_b"]["chain"]) == 4
    with pytest.raises(ValueError, match="4 deess"):
        session.add_stem_device("stem_b", "deess")


def test_the_agents_whole_file_de_esser_adjusts_the_last_one_and_leaves_the_others(demo_project):
    from mix_engine.models import DeessAction
    from mix_engine.session import Session
    session = Session(demo_project)
    session.add_stem_device("stem_b", "deess")
    session.add_stem_device("stem_b", "deess")
    session.stems["stem_b"]["chain"][-1]["freq_hz"] = 4000.0        # the user's second one, at the harshness
    session._mutate(DeessAction(stem_id="stem_b", threshold_db=-33.0, freq_hz=8000.0))
    deess = [e for e in session.stems["stem_b"]["chain"] if e["type"] == "deess"]
    assert len(deess) == 2                                           # it took both away before
    assert sorted(e["freq_hz"] for e in deess) != [8000.0, 8000.0]
    assert any(e["freq_hz"] == 8000.0 and e["threshold_db"] == -33.0 for e in deess)
