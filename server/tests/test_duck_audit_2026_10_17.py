"""The duck's audit, 2026-10-17 (the device-by-device audit) - the offline engine and the brain.

The native half is `--duck-audit`. Here:
  * `dsp._duck` is `MonitorChain::applyDuck` transcribed (per-sample key level, the LOUDER
    channel, the comp's recursion) - held to a literal loop of the native code;
  * a deleted (excluded) trigger ducks nothing - the DAW has no such track to key from;
  * the DAW may key a duck from a track the brain holds no stem for (an instrument track):
    the brain keeps the key (it refused, and reverted the pick at its next chain push), renders
    the duck unkeyed, and the measurement says it cannot hear it;
  * two stems cannot trigger each other.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from mix_engine import dsp
from mix_engine.session import Session

SR = 48000


def _native_loop(audio: np.ndarray, key: np.ndarray, e: dict) -> np.ndarray:
    """MonitorChain::applyDuck's sidechain branch, line for line, in float32."""
    a_att = np.float32(math.exp(-1.0 / (SR * e["attack_ms"] / 1000.0)))
    a_rel = np.float32(math.exp(-1.0 / (SR * e["release_ms"] / 1000.0)))
    red = np.float32(0.0)
    out = audio.astype(np.float64).copy()
    for i in range(audio.shape[0]):
        k = np.float32(max(abs(np.float32(key[i, 0])), abs(np.float32(key[i, 1]))))
        key_db = np.float32(20.0) * np.float32(math.log10(max(float(k), 1e-12)))
        target = np.float32(e["depth_db"]) if key_db > e["threshold_db"] else np.float32(0.0)
        coef = a_att if target > red else a_rel
        red = coef * red + (np.float32(1.0) - coef) * target
        out[i] *= float(np.float32(10.0 ** (-float(red) / 20.0)))
    return out


ENTRY = {"type": "duck", "threshold_db": -24.0, "depth_db": 6.0, "attack_ms": 5.0,
         "release_ms": 150.0, "key_stem_id": "k"}


def _signals():
    t = np.arange(int(0.5 * SR)) / SR
    audio = np.stack([0.3 * np.sin(2 * np.pi * 220 * t)] * 2, axis=1)
    hits = (np.sin(2 * np.pi * 4 * t) > 0.6).astype(float)
    tonal = 0.07 * np.sin(2 * np.pi * 100 * t)            # within ~3 dB of the threshold
    key = np.stack([tonal + 0.2 * hits * np.sin(2 * np.pi * 60 * t), 0.2 * tonal], axis=1)
    return audio, key


def test_the_offline_duck_is_the_native_law():
    audio, key = _signals()
    for thr in (-24.0, -26.0, -30.0):
        e = {**ENTRY, "threshold_db": thr}
        ours = dsp._duck(audio, SR, key, e)
        assert np.max(np.abs(ours - _native_loop(audio, key, e))) < 2e-3, thr


def test_a_key_on_one_channel_ducks():
    audio, _ = _signals()
    t = np.arange(audio.shape[0]) / SR
    key = np.stack([0.1 * np.sin(2 * np.pi * 60 * t), np.zeros_like(t)], axis=1)   # -20 dBFS, left only
    out = dsp._duck(audio, SR, key, ENTRY)
    half = audio.shape[0] // 2
    assert 20 * np.log10(np.sqrt(np.mean(out[half:] ** 2)) / np.sqrt(np.mean(audio[half:] ** 2))) < -5.0


def _keyed(demo_project, key: str = "stem_a") -> Session:
    s = Session(demo_project)
    s.set_duck("stem_b", key, threshold_db=-60.0, depth_db=12.0)
    return s


def test_a_deleted_trigger_ducks_nothing(demo_project):
    s = _keyed(demo_project)
    ducked, _ = dsp.render_stem(s, "stem_b")
    s.user_edit("stem_a", {"excluded": True})
    after, _ = dsp.render_stem(s, "stem_b")
    s.set_duck("stem_b", "none")
    dry, _ = dsp.render_stem(s, "stem_b")
    assert not np.allclose(ducked, dry)          # the control: the duck ducks while its key is there
    assert np.allclose(after, dry)


def test_the_daw_may_key_a_duck_from_a_track_with_no_stem(demo_project):
    s = Session(demo_project)
    with pytest.raises(ValueError, match="unknown key stem"):
        s.set_duck("stem_b", "Kick")                       # a typo from a model is still refused
    s.set_duck("stem_b", "Kick", key_may_be_track=True)    # the DAW checked the track exists
    duck = [e for e in s.stems["stem_b"]["chain"] if e["type"] == "duck"]
    assert duck and duck[0]["key_stem_id"] == "Kick"
    rendered, _ = dsp.render_stem(s, "stem_b")             # unkeyed here: no error
    assert np.isfinite(rendered).all()


def test_the_measurement_says_a_duck_under_an_instrument_is_not_heard():
    from orchestrator import daw_bridge
    edit = {"tracks": [{"name": "Bass", "track_kind": "audio", "sidechain_key": "Kick"},
                       {"name": "Kick", "track_kind": "midi"}]}
    spots = daw_bridge.render_blind_spots(edit)
    assert any(s["what"] == "ducks_from_instruments" and "Bass" in s["tracks"][0] for s in spots)


def test_two_stems_cannot_trigger_each_other(demo_project):
    s = _keyed(demo_project)
    with pytest.raises(ValueError, match="each other"):
        s.set_duck("stem_a", "stem_b")
