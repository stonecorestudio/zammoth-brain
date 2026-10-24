"""The Phase 48 technique-eval fixture: four deterministic stems whose defects
are TIME-VARYING by construction (docs/05-roadmap.md, Phase 48).

Single source of truth: the synthesis AND the window/frequency constants the
`technique_eval` graders read both live here, so the fixture and its assertions
cannot drift apart. `test_projects/technique_demo/generate_stems.py` is a thin
shim that calls `write_fixture()`.

Why time-varying: a static defect rewards the static cheat. The bass boom
swells only in known windows while its 220/440 Hz top stays constant, so a
fixed EQ cut either leaves the swells boomy or guts the steady low end — only
band-selective dynamics (multiband comp / dynamic EQ) pass both assertions.
Harshness rides only the loud lead phrases while the quiet phrases keep a
small 3.2 kHz presence that must survive, so a static presence cut fails where
a threshold does not. The lead's phrase/gap grid gives ducking a measurable
depth; the pads' slightly-loose low end gives "keep the lows mono" something
real to fix; the bass being pure sines makes added saturation harmonics an
exact FFT-bin measurement.

Pure numpy/scipy/soundfile — importable standalone (no engine, no LLM deps).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

SR = 48000
DUR_S = 16.0

# --- lead phrasing -----------------------------------------------------------
# Four phrases; the 2nd and 4th are LEAD_LOUD_DB hotter and carry the harsh
# 3.2 kHz component at full strength. Gaps are where ducks must release and
# where a reverb tail is measured (TAIL_WINDOWS start right at a phrase end).
LEAD_PHRASES = ((1.0, 3.0), (5.0, 7.0), (9.0, 11.0), (13.0, 15.0))
LEAD_LOUD_IDX = (1, 3)
LEAD_GAPS = ((3.4, 4.6), (7.4, 8.6), (11.4, 12.6))
TAIL_WINDOWS = ((3.05, 3.35), (7.05, 7.35), (11.05, 11.35))
# Esses sit on each phrase's RELEASE (a consonant ends a note — it does not
# ride a full-amplitude sustain), ending before TAIL_WINDOWS open. Placed
# mid-sustain (the 2026-08-17 layout) they were masked by the tone + the
# harshness bin and analyze_phrases found ZERO sibilance_bursts, so the
# planner had nothing to measure — a fixture-realism defect, not a model one
# (2026-08-18). SIB_AMP is set so the four bursts register at ~5.5-5.7 kHz.
# Loud phrases fade out ~70 ms later than quiet ones, so their esses sit
# at +0.92 (not +0.85) to clear the sustain; every ess ends before its
# TAIL_WINDOW opens (2.97 < 3.05, 7.04 < 7.05).
SIBILANT_S = (2.85, 6.92, 10.85, 14.92)
SIB_DUR_S = 0.12
SIB_AMP = 0.16
SIB_BAND_HZ = (5000.0, 9000.0)

LEAD_F0_HZ = 660.0
HARSH_HZ = 3200.0
# A small constant "air" partial in EVERY phrase, sitting between the harshness bin and the
# sibilance band. It gives the de-ess case something a STATIC high cut destroys and a real
# de-esser (threshold-gated, ess-only) leaves alone: before it existed the esses were the
# lead's only HF content, so a plain high-shelf cut passed the outcome grader and the
# fixture's "static cheats measurably fail" claim was untested for de-essing (2026-09-01
# sweep, B18). Outside BAND_SIB (4.5 kHz+) and BAND_MELODY (<= 2.5 kHz); not the 3.2 kHz bin.
LEAD_AIR_HZ = 4000.0
LEAD_AIR_AMP = 0.012
LEAD_LOUD_DB = 8.0
HARSH_QUIET_AMP = 0.006   # small musical presence — must SURVIVE a proper fix
HARSH_LOUD_AMP = 0.10     # the actual harshness

# --- bass --------------------------------------------------------------------
BASS_F0_HZ = 55.0
BASS_TOP_HZ = (220.0, 440.0)          # constant "top" — must not pump
BASS_H3_HZ = 165.0                     # 3rd harmonic bin the saturation case reads
BASS_SWELLS = ((2.0, 3.0), (6.0, 7.0), (10.0, 11.0), (14.0, 15.0))
BASS_STEADY = ((0.2, 1.8), (4.2, 5.8), (8.2, 9.8), (12.2, 13.8))
BASS_SWELL_DB = 7.0

# --- pads --------------------------------------------------------------------
PAD_LOW_HZ = 110.0        # mostly-mono low, plus independent low noise per
                          # channel: the low end starts a little LOOSE
                          # (correlation ≈ 0.9) so "keep the lows mono" is a
                          # real move, and naive full-band widening loosens it
                          # further — measurably.

# Measurement bands (Hz) shared with the graders.
BAND_BOOM = (40.0, 90.0)
BAND_BASS_TOP = (180.0, 500.0)
BAND_MELODY = (400.0, 2500.0)
BAND_TAIL = (500.0, 2500.0)
BAND_SIB = (4500.0, 9500.0)
BAND_PAD_LOW = (40.0, 150.0)
BAND_PAD_MID = (150.0, 500.0)
BAND_PAD_HIGH = (4000.0, 10000.0)

STEM_NAMES = ("drums", "bassline", "lead", "pads")


def _time() -> np.ndarray:
    return np.arange(int(SR * DUR_S)) / SR


def _gate(t: np.ndarray, windows, fade_s: float = 0.05) -> np.ndarray:
    """0..1 envelope: 1 inside each window, smoothstep edges of `fade_s`."""
    env = np.zeros_like(t)
    for start, end in windows:
        x = np.clip(np.minimum(t - start, end - t) / fade_s, 0.0, 1.0)
        env = np.maximum(env, x * x * (3.0 - 2.0 * x))
    return env


def _band_noise(rng: np.random.Generator, n: int, lo_hz: float,
                hi_hz: float) -> np.ndarray:
    noise = rng.standard_normal(n)
    sos = butter(4, [lo_hz, hi_hz], btype="bandpass", fs=SR, output="sos")
    out = sosfilt(sos, noise)
    return out / max(1e-9, np.abs(out).max())


def _low_noise(rng: np.random.Generator, n: int, hi_hz: float) -> np.ndarray:
    noise = rng.standard_normal(n)
    sos = butter(4, hi_hz, btype="lowpass", fs=SR, output="sos")
    out = sosfilt(sos, noise)
    return out / max(1e-9, np.abs(out).max())


def _add_at(audio: np.ndarray, start_s: float, burst: np.ndarray) -> None:
    i0 = int(start_s * SR)
    if i0 >= len(audio):
        return
    seg = burst[: len(audio) - i0]
    audio[i0:i0 + len(seg)] += seg


def _drums(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Kick every 0.5 s, a 150–400 Hz tom every 2 s, hats every 0.25 s.
    Deliberately sparse in 0.5–2.5 kHz so BAND_TAIL stays clear for the
    reverb-tail measurement."""
    audio = np.zeros_like(t)
    kick_len = int(0.15 * SR)
    seg = np.arange(kick_len) / SR
    kick = np.sin(2 * np.pi * 60.0 * seg) * np.exp(-seg / 0.05)
    click = _band_noise(rng, int(0.004 * SR), 3000.0, 9000.0) * 0.25
    tom_len = int(0.12 * SR)
    tom = (_band_noise(rng, tom_len, 150.0, 400.0)
           * np.exp(-np.arange(tom_len) / SR / 0.04) * 0.5)
    hat_len = int(0.025 * SR)
    hat = (_band_noise(rng, hat_len, 6000.0, 12000.0)
           * np.exp(-np.arange(hat_len) / SR / 0.01) * 0.35)
    for k in np.arange(0.0, DUR_S, 0.5):
        _add_at(audio, k, kick)
        _add_at(audio, k, click)
    for k in np.arange(1.0, DUR_S, 2.0):
        _add_at(audio, k, tom)
    for k in np.arange(0.125, DUR_S, 0.25):
        _add_at(audio, k, hat)
    return 0.55 * audio


def _bassline(t: np.ndarray) -> np.ndarray:
    """55 Hz fundamental that SWELLS +7 dB in known windows; constant 220/440
    top. Pure sines: no energy at 165 Hz until something nonlinear adds it."""
    swell = 1.0 + (10 ** (BASS_SWELL_DB / 20) - 1.0) * _gate(t, BASS_SWELLS,
                                                             fade_s=0.15)
    fund = 0.10 * swell * np.sin(2 * np.pi * BASS_F0_HZ * t)
    top = (0.045 * np.sin(2 * np.pi * BASS_TOP_HZ[0] * t)
           + 0.028 * np.sin(2 * np.pi * BASS_TOP_HZ[1] * t))
    return fund + top


def _lead(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Vibrato melody at 660 Hz (+harmonics), phrased. Loud phrases are +8 dB
    and carry the 3.2 kHz harsh tone at HARSH_LOUD_AMP; quiet phrases keep it
    at HARSH_QUIET_AMP (the presence a proper fix must not remove). Sibilance
    bursts (5–9 kHz noise) at known times."""
    quiet = tuple(w for i, w in enumerate(LEAD_PHRASES)
                  if i not in LEAD_LOUD_IDX)
    loud = tuple(LEAD_PHRASES[i] for i in LEAD_LOUD_IDX)
    env_q = _gate(t, quiet)
    env_l = _gate(t, loud)
    amp = 0.05 * (env_q + 10 ** (LEAD_LOUD_DB / 20) * env_l)
    tv = t + 0.0006 * np.sin(2 * np.pi * 5.0 * t)     # gentle vibrato
    tone = (np.sin(2 * np.pi * LEAD_F0_HZ * tv)
            + 0.4 * np.sin(2 * np.pi * 2 * LEAD_F0_HZ * tv)
            + 0.2 * np.sin(2 * np.pi * 3 * LEAD_F0_HZ * tv))
    harsh = ((HARSH_QUIET_AMP * env_q + HARSH_LOUD_AMP * env_l)
             * np.sin(2 * np.pi * HARSH_HZ * t))       # no vibrato: exact bin
    sib_windows = tuple((s, s + SIB_DUR_S) for s in SIBILANT_S)
    sib = (SIB_AMP * _gate(t, sib_windows, fade_s=0.01)
           * _band_noise(rng, len(t), *SIB_BAND_HZ))
    air = LEAD_AIR_AMP * (env_q + env_l) * np.sin(2 * np.pi * LEAD_AIR_HZ * t)   # no vibrato: exact bin
    return amp * tone + harsh + sib + air


def _pads(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Constant stereo bed (constant on purpose: ducking depth is measured on
    it). Mono-ish 110 Hz low + independent per-channel low noise (low end
    starts ≈0.9 correlated), detuned 330/332 Hz mids (real side content), and
    independent 5–8 kHz shimmer per channel. Nothing in 0.5–2.5 kHz."""
    n = len(t)
    low = 0.05 * np.sin(2 * np.pi * PAD_LOW_HZ * t)
    mid_common = 0.04 * np.sin(2 * np.pi * 220.0 * t)
    left = (low + 0.07 * _low_noise(rng, n, 200.0)
            + mid_common + 0.032 * np.sin(2 * np.pi * 330.0 * t)
            + 0.008 * _band_noise(rng, n, 5000.0, 8000.0))
    right = (low + 0.07 * _low_noise(rng, n, 200.0)
             + mid_common + 0.032 * np.sin(2 * np.pi * 332.0 * t)
             + 0.008 * _band_noise(rng, n, 5000.0, 8000.0))
    return np.stack([left, right], axis=1)


def write_fixture(out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t = _time()
    rng = np.random.default_rng(seed=48)   # Phase 48; determinism matters
    stems = {
        "drums": _drums(t, rng),
        "bassline": _bassline(t),
        "lead": _lead(t, rng),
        "pads": _pads(t, rng),
    }
    for name, audio in stems.items():
        # Global −6 dB: the untouched master must sit well BELOW the
        # streaming target (≈ −21 LUFS), or the mastering case has no teeth.
        sf.write(out_dir / f"{name}.wav", (0.5 * audio).astype(np.float32),
                 SR)


if __name__ == "__main__":
    write_fixture(Path(__file__).resolve().parent.parent
                  / "test_projects" / "technique_demo")
