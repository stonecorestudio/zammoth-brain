"""The duck's audit, 2026-10-17 (the device-by-device audit) - the offline engine and the brain,
re-pointed at the curve law of 2026-10-28 (duck_law; its own gates are test_duck_law.py).

The native half is `--duck-audit`. Here:
  * a trigger on ONE channel fires the duck (the detector reads the louder channel);
  * a deleted (excluded) trigger ducks nothing - the DAW has no such track to key from;
  * the DAW may key a duck from a track the brain holds no stem for (an instrument track):
    the brain keeps the key (it refused, and reverted the pick at its next chain push), renders
    the duck unkeyed, and the measurement says it cannot hear it;
  * two stems cannot trigger each other.
"""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from mix_engine import dsp
from mix_engine.session import Session

SR = 48000
ENTRY = {"type": "duck", "trigger": 1, "curve": 1, "length": 1, "mix": 1.0, "key_stem_id": "k"}


def _kicks(n: int, every_s: float = 0.25) -> np.ndarray:
    k = np.zeros(n)
    tt = np.arange(int(0.2 * SR)) / SR
    hit = 0.9 * np.sin(2 * np.pi * np.cumsum(50 + 100 * np.exp(-tt / 0.03)) / SR) * np.exp(-tt / 0.12)
    for at in np.arange(0.0, n / SR, every_s):
        i = int(at * SR); m = min(len(hit), n - i)
        k[i:i + m] += hit[:m]
    return k


def test_a_key_on_one_channel_ducks():
    n = SR
    audio = np.stack([0.3 * np.sin(2 * np.pi * 220 * np.arange(n) / SR)] * 2, axis=1)
    key = np.stack([_kicks(n), np.zeros(n)], axis=1)          # left only
    out = dsp._duck(audio, SR, ENTRY, key_audio=key)
    assert 20 * np.log10(np.sqrt(np.mean(out ** 2)) / np.sqrt(np.mean(audio ** 2))) < -3.0


def _project(tmp_path):
    project = tmp_path / "duck-audit"
    project.mkdir()
    t = np.arange(SR) / SR
    sf.write(project / "stem_b.wav", (0.3 * np.sin(2 * np.pi * 80 * t)).astype(np.float32), SR)
    sf.write(project / "stem_a.wav", _kicks(SR).astype(np.float32), SR)
    return project


def _keyed(project, key: str = "stem_a") -> Session:
    s = Session(project)
    s.set_duck("stem_b", key)
    return s


def test_a_deleted_trigger_ducks_nothing(tmp_path):
    s = _keyed(_project(tmp_path))
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
