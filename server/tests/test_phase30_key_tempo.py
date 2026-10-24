"""Phase 30 — musical key + song-level tempo on SessionAnalysis
(docs/05-roadmap.md Phase 30; definitions in the mix-analysis skill)."""
from pathlib import Path

import numpy as np

from mix_engine import analysis, dsp
from mix_engine.models import KeyEstimate
from mix_engine.session import Session

SR = 48000


def _mono(x: np.ndarray) -> np.ndarray:
    return x[:, np.newaxis]


def _note(freq: float, dur: float = 3.0, sr: int = SR) -> np.ndarray:
    """A tone with two overtones — closer to a real instrument's chroma than a
    bare sine, so the key correlation reflects musical material."""
    t = np.arange(int(sr * dur)) / sr
    return (np.sin(2 * np.pi * freq * t)
            + 0.5 * np.sin(2 * np.pi * 2 * freq * t)
            + 0.3 * np.sin(2 * np.pi * 3 * freq * t))


# ---------- 1. key detection ----------

def test_key_detects_c_major():
    # C major triad with the root doubled in the bass (C3 + C4/E4/G4):
    # energy on C/E/G, C emphasized → unambiguously C major.
    chord = (_note(130.81) + _note(261.63) + _note(329.63) + _note(392.00))
    key = analysis._key(chord, SR)
    assert key is not None
    assert key.tonic == "C"
    assert key.mode == "maj"
    assert key.confidence >= analysis.KEY_MIN_CONFIDENCE


def test_key_detects_a_minor():
    # A minor triad with the root doubled (A2 + A3/C4/E4) → A minor, the
    # relative minor of C major: confirms the detector resolves mode, not just
    # the diatonic set.
    chord = (_note(110.00) + _note(220.00) + _note(261.63) + _note(329.63))
    key = analysis._key(chord, SR)
    assert key is not None
    assert key.tonic == "A"
    assert key.mode == "min"


def test_key_none_on_noise():
    noise = 0.1 * np.random.default_rng(7).standard_normal(SR * 4)
    assert analysis._key(noise, SR) is None


def test_key_deterministic():
    chord = _note(261.63) + _note(329.63) + _note(392.00)
    assert analysis._key(chord, SR) == analysis._key(chord, SR)


# ---------- 2. session-level wiring ----------

def test_analyze_master_carries_key_and_tempo(demo_project: Path):
    session = Session(demo_project)
    audio, sr = dsp.render_master(session)
    result = analysis.analyze_master(audio, sr, list(session.stems), 0)
    # Values depend on the demo audio; assert the contract shape.
    assert result.tempo_bpm is None or isinstance(result.tempo_bpm, float)
    assert result.tempo_confidence is None or isinstance(result.tempo_confidence, float)
    assert result.key is None or isinstance(result.key, KeyEstimate)
    if result.key is not None:
        assert result.key.tonic in analysis.PITCH_CLASSES
        assert result.key.mode in ("maj", "min")
    # Determinism: re-analyzing the same render gives the same readout.
    again = analysis.analyze_master(audio, sr, list(session.stems), 0)
    assert again.tempo_bpm == result.tempo_bpm
    assert again.key == result.key


def test_analyze_master_silent_has_no_key_or_tempo():
    silent = np.zeros((SR * 2, 2))
    result = analysis.analyze_master(silent, SR, ["a"], 0)
    assert result.silent is True
    assert result.tempo_bpm is None
    assert result.tempo_confidence is None
    assert result.key is None
