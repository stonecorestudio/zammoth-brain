"""Sidechain ducking shared control (DAW Tier-1, docs/10-daw.md).

Session.set_duck is a DIRECT co-edit (no proposal lifecycle): the user keys a
stem's sidechain duck off ANOTHER stem (the kick→bass classic), the same routing
the agent sets via its `duck` action. The duck lives as a chain entry carrying
`key_stem_id`; the offline render (dsp.render_master) renders the key duck-free and
applies _duck, and in the DAW the native graph routes the key track live (App.cpp
wireDuckSidechains). This proves the brain half end-to-end.
"""
import numpy as np
import pytest
import soundfile as sf

from mix_engine import dsp
from mix_engine.session import Session

SR = 48000


def _write(project, name: str, audio: np.ndarray) -> None:
    sf.write(project / f"{name}.wav", audio.astype(np.float32), SR)


def test_set_duck_shared_control_round_trip(tmp_path):
    project = tmp_path / "sc-duck"
    project.mkdir()
    t = np.arange(int(1.0 * SR)) / SR
    _write(project, "bass", 0.30 * np.sin(2 * np.pi * 80.0 * t))    # the ducked stem
    _write(project, "kick", 0.50 * np.sin(2 * np.pi * 60.0 * t))    # the key (loud, > -30 dBFS)

    assert set(Session(project).stems) == {"bass", "kick"}
    dry, _ = dsp.render_master(Session(project))                     # no duck yet

    # the user (or agent) keys bass's duck off kick — direct, no proposal lifecycle
    res = Session(project).set_duck("bass", "kick")
    assert res["stem"] == "bass" and res["key_stem_id"] == "kick"
    ducks = [e for e in Session(project).stems["bass"]["chain"] if e["type"] == "duck"]
    assert len(ducks) == 1 and ducks[0]["key_stem_id"] == "kick"     # one duck per stem, keyed by kick

    wet, _ = dsp.render_master(Session(project))
    assert not np.array_equal(wet, dry)                              # kick now ducks bass

    # re-keying replaces (still one duck), not stacks
    Session(project).set_duck("bass", "kick", threshold_db=-40.0, depth_db=9.0)
    again = [e for e in Session(project).stems["bass"]["chain"] if e["type"] == "duck"]
    assert len(again) == 1 and again[0]["depth_db"] == 9.0

    # "none" DISARMS the duck (the entry stays, unkeyed, with its tuning — 2026-09-04 duck
    # audit: the device box must not vanish when the user picks "Trigger: none") → an unkeyed
    # duck is a render no-op, so the master is bit-identical to the dry render
    cleared = Session(project).set_duck("bass", "none")
    assert cleared["key_stem_id"] is None
    left = [e for e in Session(project).stems["bass"]["chain"] if e["type"] == "duck"]
    assert len(left) == 1 and left[0]["key_stem_id"] == "" and left[0]["depth_db"] == 9.0
    cut, _ = dsp.render_master(Session(project))
    assert np.array_equal(cut, dry)

    # guards: unknown stem / unknown key / self-key are rejected (never a silent no-op)
    with pytest.raises(ValueError):
        Session(project).set_duck("ghost", "kick")
    with pytest.raises(ValueError):
        Session(project).set_duck("bass", "ghost")
    with pytest.raises(ValueError):
        Session(project).set_duck("bass", "bass")
