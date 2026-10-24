"""Phase 55, step 2b: "things the engine noticed" - right, cheap, never in front of a request.

The block the engineer may raise unasked was computed in front of the model request whenever
a stem file or a fader had changed: 21-26 s before a turn's first byte after new stems
landed. It measured the masking of every ordered pair by rebuilding both stems' tables each
time and again per channel (336 block-FFT passes for eight stereo stems where 8 are
distinct), read every file once per stale half, and analysed the stems of DELETED tracks,
which also took places in its eight that live stems then did not get.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import orchestrator.daw_bridge as B
from mix_engine import analysis
from mix_engine.session import Session

SR = 44100


def _tone(freq: float, seconds: float = 3.0, gain: float = 0.4, stereo: bool = True,
          start_s: float = 0.0) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    x = (gain * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    x[: int(SR * start_s)] = 0.0
    return np.stack([x, x * 0.7], axis=1) if stereo else x[:, None]


def _noise(seconds: float = 3.0, gain: float = 0.3, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (gain * rng.standard_normal((int(SR * seconds), 2))).astype(np.float32)


def _project(tmp_path: Path, stems: dict) -> Path:
    proj = tmp_path / "proj"
    proj.mkdir()
    for name, audio in stems.items():
        sf.write(str(proj / f"{name}.wav"), audio, SR)
    Session(proj)
    return proj


def _edit(proj: Path, stem: str, **fields) -> None:
    state = json.loads((proj / "session.json").read_text())
    state["stems"][stem].update(fields)
    (proj / "session.json").write_text(json.dumps(state))


@pytest.fixture(autouse=True)
def _fresh_store():
    B._OBSERVATIONS.clear()
    yield
    for worker in list(B._OBS_WORKERS.values()):
        worker.join(30)
    B._OBSERVATIONS.clear()
    B._OBS_WORKERS.clear()


# ---------------------------------------------------------------- the shared core

CASES = {
    "tone": _tone(220.0),
    "noise": _noise(),
    "quiet_tone": _tone(3000.0, gain=0.01),
    "mono": _tone(440.0, stereo=False),
    "late": _tone(880.0, start_s=2.0),
    "short": _tone(300.0, seconds=0.2),             # under one 400 ms block
    "silent": np.zeros((SR * 3, 2), dtype=np.float32),
    "with_nan": np.where(np.arange(SR * 3)[:, None] == 5000, np.nan, _noise(seed=3)).astype(np.float32),
}


def test_the_ranking_figures_equal_analyze_maskings_for_every_ordered_pair():
    """`masking_worst` on two stems' tables is `analyze_masking`'s worst band and excess -
    and it refuses the pairs `analyze_masking` refuses."""
    tables: dict = {}
    for name, audio in CASES.items():
        try:
            tables[name] = analysis.masking_table(audio, SR)
        except ValueError:
            tables[name] = None
    compared = refused = 0
    for a in CASES:
        for b in CASES:
            if a == b:
                continue
            try:
                full = analysis.analyze_masking(CASES[a], CASES[b], SR, a, b)
            except ValueError:
                full = None
            if tables[a] is None or tables[b] is None:
                light = None
            else:
                try:
                    light = analysis.masking_worst(tables[a], tables[b], a)
                except ValueError:
                    light = None
            if full is None:
                assert light is None, f"{a} under {b}: analyze_masking refuses, the core answered"
                refused += 1
            else:
                assert light == (full.worst_band, full.worst_band_excess_db), f"{a} under {b}"
                compared += 1
    assert compared >= 20 and refused >= 10, "vacuity: both outcomes must be exercised"


def test_eight_stereo_stems_build_eight_tables(tmp_path, monkeypatch):
    stems = {f"s{i}": (_tone(110.0 * (i + 1)) + _noise(seed=i, gain=0.05)) for i in range(8)}
    proj = _project(tmp_path, stems)
    built = [0]
    real = analysis._block_band_power

    def counting(*a, **k):
        built[0] += 1
        return real(*a, **k)
    monkeypatch.setattr(analysis, "_block_band_power", counting)
    B.session_observations(proj)
    assert built[0] <= 8, f"{built[0]} block-FFT passes for eight stems (336 before)"


def test_each_file_is_read_once_when_both_halves_are_stale(tmp_path, monkeypatch):
    proj = _project(tmp_path, {"a": _tone(220.0), "b": _noise()})
    opened: list[str] = []
    real = sf.SoundFile

    class Counting(real):
        def __init__(self, path, *a, **k):
            opened.append(Path(path).name)
            super().__init__(path, *a, **k)
    monkeypatch.setattr(sf, "SoundFile", Counting)
    monkeypatch.setattr(sf, "read", lambda *a, **k: pytest.fail("a whole-file read"))
    B.session_observations(proj)
    assert sorted(opened) == ["a.wav", "b.wav"]


# ---------------------------------------------------------------- only what is in the mix

def _crowded(tmp_path: Path) -> Path:
    """A vocal-ish tone, a quiet bed, and a loud noise that buries the tone."""
    return _project(tmp_path, {"bed": _tone(110.0, gain=0.05), "noise": _noise(gain=0.6),
                               "vocal": _tone(1000.0, gain=0.01)})    # 23 dB under the noise


def test_a_deleted_tracks_stem_is_never_named(tmp_path):
    proj = _crowded(tmp_path)
    said = " | ".join(o["text"] for o in B.session_observations(proj, cap=8))
    assert "noise is crowding" in said, f"vacuity: the masker must be found while it is live: {said}"
    _edit(proj, "noise", excluded=True)               # the track was deleted
    said = " | ".join(o["text"] for o in B.session_observations(proj, cap=8))
    assert "noise" not in said, f"a deleted track was reported: {said}"


def test_deleted_stems_do_not_take_the_places_of_live_ones(tmp_path):
    stems = {f"a{i}": _tone(100.0 + 40 * i) for i in range(6)}            # sort first
    stems.update({f"z{i}": _tone(900.0 + 40 * i) for i in range(4)})
    proj = _project(tmp_path, stems)
    for i in range(6):
        _edit(proj, f"a{i}", excluded=True)
    state = json.loads((proj / "session.json").read_text())["stems"]
    assert B._observed_stems(state) == ["z0", "z1", "z2", "z3"]


def test_a_muted_stem_is_not_in_the_sum_the_resonance_rule_reads(tmp_path, monkeypatch):
    proj = _project(tmp_path, {"a": _tone(220.0), "ring": _tone(3100.0, gain=0.5)})
    peaks: list[float] = []
    real = analysis._resonances

    def spy(mix, sr):
        peaks.append(float(np.max(np.abs(mix))))
        return real(mix, sr)
    monkeypatch.setattr(analysis, "_resonances", spy)
    B.session_observations(proj)
    _edit(proj, "ring", muted=True)
    B.session_observations(proj)
    assert len(peaks) == 2, "a mute is a change of the faders half"
    assert peaks[1] < peaks[0] - 0.2, "the muted stem was still summed"


# ---------------------------------------------------------------- never in front of a request

def test_a_turns_context_does_not_wait_for_the_findings(tmp_path, monkeypatch):
    proj = _project(tmp_path, {"a": _tone(220.0), "b": _noise()})
    release, entered = threading.Event(), threading.Event()
    real = B._compute_halves

    def slow(project_dir, halves):
        entered.set()
        assert release.wait(20), "the test never released the computation"
        return real(project_dir, halves)
    monkeypatch.setattr(B, "_compute_halves", slow)
    monkeypatch.setattr(B, "session_readout", lambda *a, **k: {"lufs_i": -14.0})
    try:
        context = B._chat_context_stable(proj, "agent", {}, None)   # returns while it computes
        assert entered.wait(10), "nothing started the computation"
        assert "Things the engine noticed" not in context
        assert "Workspace mode: agent" in context
    finally:
        release.set()
    assert B.wait_for_observations(proj, 30)
    monkeypatch.setattr(B, "_compute_halves", lambda *a, **k: pytest.fail("computed again"))
    assert B.session_observations(proj, wait=False) == B.session_observations(proj)


def test_one_worker_a_project_and_a_caller_that_waits_joins_it(tmp_path, monkeypatch):
    proj = _project(tmp_path, {"a": _tone(220.0), "b": _noise()})
    runs = [0]
    release = threading.Event()
    real = B._compute_halves

    def slow(project_dir, halves):
        runs[0] += 1
        release.wait(20)
        return real(project_dir, halves)
    monkeypatch.setattr(B, "_compute_halves", slow)
    for _ in range(5):
        B.warm_observations(proj)
    threading.Timer(0.3, release.set).start()
    found = B.session_observations(proj)                # waits for the worker, computes nothing
    assert runs[0] == 1
    assert found == B.session_observations(proj, wait=False)


def test_a_computation_that_failed_is_not_kept_as_no_findings(tmp_path, monkeypatch):
    proj = _crowded(tmp_path)
    real = analysis.masking_table
    monkeypatch.setattr(analysis, "masking_table", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    assert all("crowding" not in o["text"] for o in B.session_observations(proj, cap=8))
    monkeypatch.setattr(analysis, "masking_table", real)
    said = " | ".join(o["text"] for o in B.session_observations(proj, cap=8))
    assert "noise is crowding" in said, "the failure was served as a finding of nothing"
