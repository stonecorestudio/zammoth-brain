"""The limiter's audit, 2026-10-17 (the device-by-device audit) - the offline engine and the brain.

`dsp._brickwall` is `MonitorChain::applyLookaheadLimiter` (v2) transcribed - held here to a literal
per-sample loop of the native law: the 4x detector, the target blended by link, HOLD as the
running minimum over the lookahead, the one-pole envelope (attack over lookahead/6, release),
the gain floor and the ceiling clamp. It was a windowed law of its own, and at release 300 the
two engines read ~3 dB apart; the export oracle's `nonlinear` profile now runs a limiter that
works hard at release 300. The native half is `--limiter-audit`.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

from mix_engine import dsp
from mix_engine.models import LIMITER_STYLE_PRESETS
from mix_engine.session import Session
from mix_engine.tp_detector import TP_CENTRE, intersample_peak

SR = 48000
ROOT = Path(__file__).resolve().parents[2]


def _native_loop(x: np.ndarray, e: dict) -> np.ndarray:
    n, nch = x.shape
    c = 10 ** (e["ceiling_dbtp"] / 20)
    ring = max(1, int(SR * 0.003))
    atk = min(max(e.get("lookahead_ms", 3.0), 0.1), ring * 1000 / SR)
    rel = max(1.0, e.get("release_ms", 50.0))
    link = e.get("channel_link", 1.0)
    ac, rc = 1 - math.exp(-6 / (SR * atk / 1000)), 1 - math.exp(-1 / (SR * rel / 1000))
    pk = intersample_peak(x)
    target = np.where(pk > c, c / np.maximum(pk, 1e-30), 1.0)
    loud = np.max(pk, axis=1, keepdims=True)
    target = target * (1 - link) + np.where(loud > c, c / np.maximum(loud, 1e-30), 1.0) * link

    def t_at(m, ch):
        return target[m, ch] if 0 <= m < n else 1.0
    out = np.zeros_like(x)
    env = [1.0] * nch
    for s in range(n + ring):
        for ch in range(nch):
            hold = min(t_at(m, ch) for m in range(s - TP_CENTRE - ring, s - TP_CENTRE + 1))
            env[ch] += (hold - env[ch]) * (ac if hold < env[ch] else rc)
            m = s - ring
            if 0 <= m < n:
                g = min(env[ch], min(t_at(k, ch) for k in range(m - 2, m + 3)))
                out[m, ch] = max(-c, min(c, x[m, ch] * g))
    return out


def _loud(n: int = 4000) -> np.ndarray:
    rng = np.random.default_rng(1)
    t = np.arange(n) / SR
    left = np.sin(2 * np.pi * 220 * t) * (0.5 + 0.8 * (t % 0.05 < 0.01))
    right = 0.7 * np.sin(2 * np.pi * 330 * t + 1) + 0.3 * rng.standard_normal(n)
    return np.stack([left, right], axis=1) * 1.6


@pytest.mark.parametrize("entry", [
    {"ceiling_dbtp": -1.0, "release_ms": 50.0, "lookahead_ms": 3.0, "channel_link": 1.0},
    {"ceiling_dbtp": -3.0, "release_ms": 300.0, "lookahead_ms": 1.0, "channel_link": 0.3},
])
def test_the_offline_limiter_is_the_native_law(entry):
    x = _loud()
    assert np.max(np.abs(dsp._brickwall(x, SR, entry) - _native_loop(x, entry))) < 2e-4


def test_a_style_write_applies_its_preset(demo_project):
    s = Session(demo_project)
    s.add_master_device("limiter")
    s.set_master_device_param("limiter", 0, "style", 1.4)
    lim = next(e for e in s._state["master_chain"] if e["type"] == "limiter")
    assert lim["style"] == 1.0
    assert (lim["release_ms"], lim["lookahead_ms"]) == LIMITER_STYLE_PRESETS[1]


def test_the_native_style_table_is_the_models():
    src = (ROOT / "native/Source/DeviceSpecs.h").read_text()
    row = re.search(r"kLimiterStylePresets\[3\]\[2\] = \{(.*?)\};", src).group(1)
    pairs = [tuple(float(v) for v in p.split(",")) for p in re.findall(r"\{\s*([^{}]+?)\s*\}", row)]
    assert pairs == [LIMITER_STYLE_PRESETS[i] for i in range(3)]


def test_a_loudness_target_off_the_master_is_refused(demo_project):
    s = Session(demo_project)
    s.add_stem_device("stem_b", "limiter")
    with pytest.raises(ValueError, match="master"):
        s.set_master_device_param("limiter", 0, "target_lufs_i", -10.0, target="stem_b")


def test_an_unheard_mix_says_the_loudness_target_can_be_missed(demo_project):
    from orchestrator import daw_bridge
    s = Session(demo_project)
    s.add_master_device("limiter")
    lim = next(e for e in s._state["master_chain"] if e["type"] == "limiter")
    lim["target_lufs_i"] = -12.0
    s._persist()
    edit = {"tracks": [{"name": "Keys", "track_kind": "midi", "clips": [{}]}]}
    out = daw_bridge.with_blind_spots("analyze_session", {}, {"lufs_i": -14.0}, edit, demo_project)
    assert any(b["what"] == "loudness_target" for b in out["not_heard"])
