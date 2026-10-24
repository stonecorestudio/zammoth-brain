"""Convolution (IR) reverb — Tier-1 #3 (DAW capability map; docs/10-daw.md).

A new wet-only RETURN type `return_convolution`: the summed sends are convolved with a
procedural room/hall/plate impulse response (scipy.fftconvolve offline; juce::dsp::Convolution
native, oracle-validated within tolerance). This covers the OFFLINE deliverable + the agent/user
plumbing (the native + live-monitor half is increment 2). Proves: the procedural IRs are
deterministic + distinct per space + scale with decay; the return renders a real convolved tail;
and the user co-edit (Session.add_return("convolution")) wires it end to end.
"""
import numpy as np
import soundfile as sf

from mix_engine import dsp
from mix_engine.session import Session

SR = 48000


def _write(project, name: str, audio: np.ndarray) -> None:
    sf.write(project / f"{name}.wav", audio.astype(np.float32), SR)


def test_procedural_ir_deterministic_distinct_and_scales():
    room = dsp._procedural_ir("room", 2.0, 0.5, SR)
    hall = dsp._procedural_ir("hall", 2.0, 0.5, SR)
    plate = dsp._procedural_ir("plate", 2.0, 0.5, SR)

    # deterministic: same params → byte-identical IR (so renders are reproducible)
    assert np.array_equal(room, dsp._procedural_ir("room", 2.0, 0.5, SR))
    # each space is a distinct impulse
    assert not np.array_equal(room, hall)
    assert not np.array_equal(hall, plate)
    # decay scales the IR length (longer RT60 → longer tail)
    assert dsp._procedural_ir("hall", 4.0, 0.5, SR).shape[0] > dsp._procedural_ir("hall", 1.0, 0.5, SR).shape[0]
    # finite + non-trivial energy after normalization
    assert np.isfinite(room).all()
    assert np.sqrt(np.mean(room ** 2)) > 0.0


def test_convolution_reverb_renders_a_wet_tail(tmp_path):
    project = tmp_path / "conv"
    project.mkdir()
    # a 0.1 s burst then silence — the convolution should ring on past the burst
    x = np.zeros(int(1.5 * SR), dtype=np.float32)
    burst = int(0.1 * SR)
    x[:burst] = 0.4 * np.sin(2 * np.pi * 440.0 * np.arange(burst) / SR)
    _write(project, "vox", x)

    s = Session(project)
    rid = s.add_return("convolution")["return_id"]
    assert rid.startswith("cv")
    assert s.returns[rid]["chain"][-1]["type"] == "return_convolution"
    s.set_send("vox", rid, -3.0)

    wet, sr = dsp.render_return(Session(project), rid)
    assert sr == SR
    assert np.max(np.abs(wet)) > 1e-4                                   # the return is audible
    tail = wet[int(0.4 * SR):]                                          # well past the 0.1 s burst
    assert np.sqrt(np.mean(tail ** 2)) > 1e-5                           # the IR rings into the silence


def test_convolution_reverb_in_master_and_modes_differ(tmp_path):
    project = tmp_path / "conv-master"
    project.mkdir()
    t = np.arange(int(1.0 * SR)) / SR
    _write(project, "vox", 0.3 * np.sin(2 * np.pi * 440.0 * t))

    dry, _ = dsp.render_master(Session(project))

    s = Session(project)
    rid = s.add_return("convolution")["return_id"]
    s.set_send("vox", rid, -6.0)
    hall, _ = dsp.render_master(Session(project))
    assert not np.array_equal(hall, dry)                                # the IR reverb is in the mix

    # switching the space changes the deliverable (room ≠ hall)
    s2 = Session(project)
    s2.returns[rid]["chain"][-1]["ir"] = "room"
    s2._persist()
    room, _ = dsp.render_master(Session(project))
    assert not np.array_equal(room, hall)
