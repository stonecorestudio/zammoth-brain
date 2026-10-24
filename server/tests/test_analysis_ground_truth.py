"""Ground truth for the descriptive detectors (2026-09-02 eval audit).

The agent reads numbers, not audio, so its judgement is only as good as the
analysis layer — and the audit found that layer proven at single points: key
detection on C major + A minor only, tempo on one BPM with no half/double-time
case, tuning at two offsets, and nothing at all on whether the same music reads
the same at 44.1k, 48k and 96k. Every test here synthesizes material with a
KNOWN answer and asserts the detector recovers it; a range-only "returns a
float" test proves nothing about accuracy.

Fixtures are deliberately musical rather than bare sines: three-partial tones,
a I-IV-V-I (major) or i-iv-V-i with the raised leading tone (minor) progression
with the root doubled below, so the Krumhansl profile has a real tonal centre to
correlate with — and a relative key (C major vs A minor) differs in more than
one pitch class.
"""
from __future__ import annotations

import numpy as np
import pytest

from mix_engine import analysis

SR = 48000
PITCH_CLASSES = analysis.PITCH_CLASSES
_A4 = 440.0


def _hz(pitch_class: str, octave: int = 4) -> float:
    semis = PITCH_CLASSES.index(pitch_class) - 9 + 12 * (octave - 4)
    return _A4 * 2.0 ** (semis / 12.0)


def _tone(freq: float, dur: float, sr: int = SR, level: float = 0.2) -> np.ndarray:
    """Three partials + a 10 ms fade at both ends (no clicks to fake onsets)."""
    t = np.arange(int(sr * dur)) / sr
    x = (np.sin(2 * np.pi * freq * t)
         + 0.5 * np.sin(2 * np.pi * 2 * freq * t)
         + 0.3 * np.sin(2 * np.pi * 3 * freq * t))
    ramp = min(int(0.010 * sr), x.size // 2)
    env = np.ones_like(x)
    env[:ramp] = np.linspace(0.0, 1.0, ramp)
    env[-ramp:] = np.linspace(1.0, 0.0, ramp)
    return level * x * env


def _chord(root: str, quality: str, dur: float, sr: int = SR) -> np.ndarray:
    third = 4 if quality == "maj" else 3
    pcs = [PITCH_CLASSES[(PITCH_CLASSES.index(root) + i) % 12] for i in (0, third, 7)]
    audio = _tone(_hz(root, 3), dur, sr)              # root doubled an octave below
    for pc in pcs:
        audio = audio + _tone(_hz(pc, 4), dur, sr)
    return audio


def _progression(tonic: str, mode: str, chord_s: float = 2.0, sr: int = SR) -> np.ndarray:
    """I-IV-V-I in a major key; i-iv-V-i (harmonic minor's major dominant) in a
    minor key. Degrees as semitone offsets from the tonic."""
    idx = PITCH_CLASSES.index(tonic)
    degree = lambda semis: PITCH_CLASSES[(idx + semis) % 12]  # noqa: E731
    if mode == "maj":
        chords = [(degree(0), "maj"), (degree(5), "maj"), (degree(7), "maj"), (degree(0), "maj")]
    else:
        chords = [(degree(0), "min"), (degree(5), "min"), (degree(7), "maj"), (degree(0), "min")]
    return np.concatenate([_chord(r, q, chord_s, sr) for r, q in chords])


def _click_track(bpm: float, duration_s: float, sr: int = SR,
                 offset_s: float = 0.0, level: float = 0.5) -> np.ndarray:
    audio = np.zeros(int(sr * duration_s))
    blip_t = np.arange(int(0.010 * sr)) / sr
    blip = level * np.sin(2 * np.pi * 1000.0 * blip_t) * np.hanning(blip_t.size)
    beat = offset_s
    while beat < duration_s:
        start = int(beat * sr)
        end = min(start + blip.size, audio.size)
        audio[start:end] += blip[: end - start]
        beat += 60.0 / bpm
    return audio


# ---------------------------------------------------------------- key ----

@pytest.mark.parametrize("mode", ["maj", "min"])
@pytest.mark.parametrize("tonic", PITCH_CLASSES)
def test_key_recovers_every_tonic_and_mode(tonic: str, mode: str):
    """All 24 tonic/mode pairs, not two. The relative-key trap is the one that
    matters: a minor progression must not read as its relative major."""
    key = analysis._key(_progression(tonic, mode), SR)
    assert key is not None, f"{tonic} {mode}: no key at all"
    assert (key.tonic, key.mode) == (tonic, mode), \
        f"{tonic} {mode} read as {key.tonic} {key.mode} (conf {key.confidence})"


def test_key_survives_a_transposition_of_the_same_material():
    """The same progression a semitone up must move the tonic and nothing else."""
    base = analysis._key(_progression("F", "maj"), SR)
    up = analysis._key(_progression("F#", "maj"), SR)
    assert base is not None and up is not None
    assert (base.tonic, base.mode, up.tonic, up.mode) == ("F", "maj", "F#", "maj")


# -------------------------------------------------------------- tempo ----

@pytest.mark.parametrize("bpm", [72.0, 96.0, 120.0, 140.0, 174.0])
def test_tempo_recovers_a_click_across_the_range(bpm: float):
    tempo = analysis._tempo(_click_track(bpm, 12.0), SR)
    assert tempo is not None
    got, confidence, _offset = tempo
    assert abs(got - bpm) <= 2.0, f"{bpm} BPM read as {got}"
    assert confidence >= analysis.TEMPO_MIN_CONFIDENCE


def _backbeat(bpm: float, duration_s: float, sr: int = SR) -> np.ndarray:
    """A drum-shaped pattern: loud kick on 1 and 3, loud snare on 2 and 4,
    quiet hats on every eighth. The eighth-note pulse is the half/double-time
    trap — a musician says `bpm`, a naive peak-picker says 2x."""
    beat = 60.0 / bpm
    audio = np.zeros(int(sr * duration_s))

    def hit(at_s: float, freq: float, level: float, dur: float) -> None:
        t = np.arange(int(dur * sr)) / sr
        x = level * np.sin(2 * np.pi * freq * t) * np.exp(-t / (dur / 4))
        start = int(at_s * sr)
        end = min(start + x.size, audio.size)
        audio[start:end] += x[: end - start]

    n = 0
    while n * beat < duration_s:
        hit(n * beat, 60.0 if n % 2 == 0 else 200.0, 0.8, 0.12)     # kick / snare
        hit(n * beat, 6000.0, 0.15, 0.03)                            # hat on the beat
        hit((n + 0.5) * beat, 6000.0, 0.15, 0.03)                    # hat off the beat
        n += 1
    return audio


def test_tempo_does_not_double_a_backbeat():
    """85 BPM with eighth-note hats must read as 85, not the 170 eighth pulse."""
    tempo = analysis._tempo(_backbeat(85.0, 16.0), SR)
    assert tempo is not None
    got = tempo[0]
    assert abs(got - 85.0) <= 2.0, f"backbeat at 85 read as {got} (double-time trap)"


def test_tempo_does_not_halve_a_fast_click():
    """160 BPM is inside the 60-200 window and so is its half; the click is
    unambiguous, so 80 would be the halving error."""
    tempo = analysis._tempo(_click_track(160.0, 12.0), SR)
    assert tempo is not None
    assert abs(tempo[0] - 160.0) <= 2.0, f"160 BPM click read as {tempo[0]}"


# ------------------------------------------------------------- tuning ----

@pytest.mark.parametrize("cents", [-45.0, -12.0, 12.0, 45.0])
def test_tuning_recovers_offsets_out_to_the_edge_of_the_range(cents: float):
    """±45 sits next to the ±50 wrap where a sign error would hide."""
    ratio = 2.0 ** (cents / 1200.0)
    chord = sum(_tone(_hz(pc, 4) * ratio, 3.0) for pc in ("C", "E", "G"))
    chord = chord + _tone(_hz("C", 3) * ratio, 3.0)
    tuning = analysis._tuning(chord, SR)
    assert tuning is not None
    got, _coherence = tuning
    assert abs(got - cents) <= 5.0, f"{cents:+.0f} cents read as {got:+.1f}"


# ------------------------------------------------- sample-rate stability ----

def _song(sr: int, click_level: float = 0.8, music_level: float = 0.5) -> np.ndarray:
    """The same sixteen seconds of music synthesized natively at `sr`: a G major
    progression, twice, under a 120 BPM click, stereo."""
    one = _progression("G", "maj", chord_s=2.0, sr=sr) * music_level
    music = np.concatenate([one, one])
    click = _click_track(120.0, music.size / sr, sr=sr, level=click_level)[: music.size]
    mono = music + click
    return np.stack([mono, mono * 0.9], axis=1).astype(np.float32)


@pytest.mark.parametrize("sr", [44100, 96000])
def test_the_same_music_reads_the_same_at_every_sample_rate(sr: int):
    """Key, tempo, loudness and tuning must agree with the 48 kHz reading —
    a project renders at its HIGHEST stem rate (the engine resamples up, never
    down), the DAW runs at the device rate, and a readout that moved with the
    rate would move the agent's decisions with it. The 24-key sweep at 88.2 /
    96 / 192 kHz is in test_audit_p1_fixes.py (reader E5.R3)."""
    ref = analysis.analyze_master(_song(48000), 48000, ["a"], 0)
    other = analysis.analyze_master(_song(sr), sr, ["a"], 0)
    assert ref.key is not None and other.key is not None
    assert (ref.key.tonic, ref.key.mode) == (other.key.tonic, other.key.mode) == ("G", "maj")
    assert ref.tempo_bpm is not None and other.tempo_bpm is not None
    assert abs(ref.tempo_bpm - other.tempo_bpm) <= 1.0
    assert abs(ref.lufs_i - other.lufs_i) <= 0.3
    assert abs(ref.true_peak_dbtp - other.true_peak_dbtp) <= 0.5
    if ref.tuning_offset_cents is not None and other.tuning_offset_cents is not None:
        assert abs(ref.tuning_offset_cents - other.tuning_offset_cents) <= 3.0


# Recorded 2026-09-02 as a strict xfail: `_spectral_flux` ran a fixed 1024/512-SAMPLE STFT, so at
# 96 kHz every frame spanned half the time and this 120 BPM read None. The window is rate-proportional
# since the 2026-10-02 full audit (reader E5.R3, `analysis._rate_factor`), with every other
# sample-counted window in the module; 44.1 / 48 kHz readings are bit-for-bit what they were.
def test_tempo_confidence_survives_96k_on_chord_heavy_material():
    hard = lambda sr: _song(sr, click_level=0.5, music_level=1.0).mean(axis=1)  # noqa: E731
    at_48k = analysis._tempo(hard(48000), 48000)
    at_96k = analysis._tempo(hard(96000), 96000)
    assert at_48k is not None and abs(at_48k[0] - 120.0) <= 1.5
    assert at_96k is not None, "96 kHz lost the grid 48 kHz found"
    assert abs(at_96k[0] - 120.0) <= 1.5
    assert at_96k[1] >= 0.6 * at_48k[1], "confidence halved with the sample rate"
