"""What ONE SOUND is: a one-shot or a loop, measured (Phase 58).

Every other reading in this engine is of a mix or of a part of one: integrated loudness is
`None` under 0.4 s, key and tuning need 0.17 s of steady partials, tempo needs a pulse. A
kick is 150 ms and none of them answers. `sample_profile` reads the things a sound designer
asks of a single sound - how fast it starts, how long it rings, where its energy and its
pitch are, whether its pitch falls, whether it is a tone or noise, whether it will click when
it is cut or looped - on any length from a few milliseconds up.

Pure: the caller reads the file and passes the audio in (the analysis module's rule). The
definitions are in `.claude/skills/mix-analysis/SKILL.md` ▸ *One sound*, and are contractual
in the same way: change one and `SAMPLE_PROFILE_VERSION` moves.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft

from . import analysis
from .models import (BAND_EDGES_HZ, BAND_LABELS, SAMPLE_PROFILE_VERSION, SampleLoop,
                     SampleProfile)

# ---- where it starts, rises, falls and ends -------------------------------------------
# All four are read on the SAMPLES (the largest channel's magnitude), never on a smoothed or
# analytic envelope: an envelope with any memory - a moving average, the Hilbert transform's
# 1 / t kernel - puts energy in front of a sound that starts from digital silence, and
# "leading silence" and "attack" are exactly the readings that would see it.
ONSET_BELOW_PEAK_DB = 40.0       # the sound starts at the first sample within this of its peak
END_BELOW_PEAK_DB = 60.0         # ...and ends at the last one within this
# attack: the first sample at 10 % of the sound's LEVEL -> the first at 90 % of it. The level is
# the 95th percentile of the sample magnitudes while it sounds, not the one largest sample: on a
# tone they are the same, on steady noise the largest sample is a 4-sigma accident that takes a
# random few hundred milliseconds to come round ("white noise has a 480 ms attack").
ATTACK_FROM, ATTACK_TO = 0.1, 0.9
ATTACK_LEVEL_PERCENTILE = 95.0
DECAY_DB = 20.0                  # decay: from the peak until no later sample comes within this of it
END_WINDOW_S = 0.005             # the file's last 5 ms: what a cut leaves on the output
LEVEL_FLOOR_DB = -120.0

# ---- pitch -----------------------------------------------------------------------------
PITCH_LO_HZ = 28.0               # A0 and a little: an 808's bottom
PITCH_HI_HZ = 2000.0
PITCH_MIN_PERIODICITY = 0.5      # the autocorrelation peak a pitch needs (analyze_notes' own)
PITCH_OCTAVE_TOLERANCE = 0.85    # the first peak within this of the best wins (octave defence)
PITCH_BODY_MAX_S = 0.5           # the body's pitch is read from at most this much of it
PITCH_BODY_FLOOR_DB = 30.0       # ...and only where the envelope is within this of the peak
PITCH_TRACK_HOP_S = 0.002        # the walk back from the body to the onset
PITCH_TRACK_PERIODS = 2.5        # a frame of the walk holds this many periods of the pitch it follows
PITCH_TRACK_MIN_S, PITCH_TRACK_MAX_S = 0.004, 0.08
PITCH_DROP_MIN_SEMITONES = 1.0   # under this the pitch did not move: no start pitch is reported
PITCH_SETTLE_SEMITONES = 1.0

MAX_SECONDS = 120.0              # one sound: a longer file is a song's, and analyze_reference's

# ---- a loop's seam -----------------------------------------------------------------------
SEAM_WINDOW_S = 0.02
FULL_SCALE = 0.9999              # a fixed-point file's ceiling, as a float

_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _db(value: float, floor: float = LEVEL_FLOOR_DB) -> float:
    return analysis._round_db(max(20.0 * np.log10(max(float(value), 1e-12)), floor))


def _magnitude(frames: np.ndarray) -> np.ndarray:
    """Each sample's size: the largest channel's (a mono sum cancels an anti-phase pair)."""
    return np.max(np.abs(frames), axis=1)


def _first_at_or_over(values: np.ndarray, level: float) -> Optional[int]:
    hit = np.flatnonzero(values >= level)
    return int(hit[0]) if hit.size else None


def _acf_pitch(frame: np.ndarray, sample_rate: float, lo_hz: float, hi_hz: float,
               refine: bool = False) -> tuple[Optional[float], float]:
    """(f0 or None, periodicity 0..1) of one frame by normalised autocorrelation - by lag 0
    and by the overlap at each lag, as `analyze_notes` does - taking the FIRST peak within
    PITCH_OCTAVE_TOLERANCE of the best one (the lag at twice the period correlates as well)."""
    n = frame.shape[0]
    if n < 16:
        return None, 0.0
    x = frame - frame.mean()
    spectrum = rfft(x, next_fast_len(2 * n))
    acf = irfft(np.abs(spectrum) ** 2)[:n]
    if acf[0] <= 0.0:
        return None, 0.0
    lag_min = max(2, int(np.floor(sample_rate / hi_hz)))
    lag_max = min(n // 2, int(np.ceil(sample_rate / lo_hz)))
    if lag_max - lag_min < 3:
        return None, 0.0
    lags = np.arange(n)
    norm = acf / acf[0] * (n / np.maximum(n - lags, 1))
    seg = norm[lag_min - 1:lag_max + 2]
    peaks = np.flatnonzero((seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:])) + 1
    if peaks.size == 0:
        return None, 0.0
    heights = seg[peaks]
    best = float(heights.max())
    if best <= 0.0:
        return None, 0.0
    first = int(peaks[np.flatnonzero(heights >= PITCH_OCTAVE_TOLERANCE * best)[0]])

    def vertex(at: int) -> float:                 # the peak's place between the lags, by parabola
        a, b, c = norm[at - 1], norm[at], norm[at + 1]
        denom = a - 2.0 * b + c
        return at + float(np.clip(0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0, -0.5, 0.5))

    at = (lag_min - 1) + first
    lag = vertex(at)
    periodicity = float(np.clip(norm[at], 0.0, 1.0))
    if refine:
        # A steady pitch repeats at every multiple of its period, and a lag m periods long says
        # the period m times finer (a 1 kHz tone is 44 samples a period: 4 cents a quarter
        # sample). The longest multiple that still correlates as well is used.
        m = 2
        while True:
            near = int(round(m * lag))
            if near + 3 >= n // 2:
                break
            lo_i = near - 2
            local = lo_i + int(np.argmax(norm[lo_i:near + 3]))
            if norm[local] < PITCH_OCTAVE_TOLERANCE * norm[at] or not (norm[local] >= norm[local - 1]
                                                                    and norm[local] >= norm[local + 1]):
                break
            lag = vertex(local) / m
            m *= 2
    return float(sample_rate / lag), periodicity


def _note_of(hz: float) -> tuple[str, int, float]:
    midi = 69.0 + 12.0 * np.log2(hz / 440.0)
    nearest = int(np.round(midi))
    return (f"{_NOTE_NAMES[nearest % 12]}{nearest // 12 - 1}", nearest,
            round(float((midi - nearest) * 100.0), 1))


def _pitch(mono: np.ndarray, sample_rate: float, onset: int, end: int) -> dict:
    """The body's pitch and how sure it is; then, walking back from the body to the onset in
    frames that hold a few periods of the pitch they follow, where the pitch STARTED."""
    out: dict = {"pitch_hz": None, "pitch_note": None, "pitch_midi": None, "pitch_cents": None,
                 "pitch_confidence": None, "pitch_start_hz": None,
                 "pitch_drop_semitones": None, "pitch_settle_ms": None, "noisiness": None}
    size = np.abs(mono)
    peak = float(size[onset:end + 1].max()) if end >= onset else 0.0
    if peak <= 0.0:
        return out
    loud = np.flatnonzero(size[onset:end + 1] >= peak * 10.0 ** (-PITCH_BODY_FLOOR_DB / 20.0))
    risen = np.flatnonzero(size[onset:end + 1] >= ATTACK_TO * peak)
    if loud.size == 0 or risen.size == 0:
        return out
    loud_end = onset + int(loud[-1])
    attack_end = onset + int(risen[0])
    # the body: past the attack - a quarter of the way from there to where it is still loud
    body_from = min(attack_end + (loud_end - attack_end) // 4, max(onset, loud_end - 32))
    body_to = min(loud_end, body_from + int(PITCH_BODY_MAX_S * sample_rate))
    body = mono[body_from:body_to + 1]
    # a body shorter than two periods of the floor cannot hold the floor: read what it can
    lo = max(PITCH_LO_HZ, 2.2 * sample_rate / max(1, body.shape[0]))
    f0, periodicity = _acf_pitch(body, sample_rate, lo, PITCH_HI_HZ, refine=True)
    out["noisiness"] = round(1.0 - periodicity, 2)
    if f0 is None or periodicity < PITCH_MIN_PERIODICITY:
        return out
    name, number, cents = _note_of(f0)
    out.update(pitch_hz=round(f0, 1), pitch_note=name, pitch_midi=number, pitch_cents=cents,
               pitch_confidence=round(periodicity, 2))

    # the walk back: (time, f0) from the body's start to the onset
    hop = max(1, int(round(PITCH_TRACK_HOP_S * sample_rate)))
    track: list[tuple[int, float]] = []
    following = f0
    centre = body_from
    while centre > onset:
        centre -= hop
        half = int(np.clip(PITCH_TRACK_PERIODS / following, PITCH_TRACK_MIN_S, PITCH_TRACK_MAX_S)
                   * sample_rate / 2)
        lo_i, hi_i = centre - half, centre + half
        if lo_i < max(0, onset - half // 2) or hi_i >= mono.shape[0]:
            break
        got, sure = _acf_pitch(mono[lo_i:hi_i + 1], sample_rate,
                               max(PITCH_LO_HZ, following / 1.25), min(PITCH_HI_HZ * 2.0, following * 1.6))
        if got is None or sure < PITCH_MIN_PERIODICITY:
            break
        track.append((centre, got))
        following = got
    if not track:
        return out
    earliest = float(np.median([f for _, f in track[-3:]]))
    drop = 12.0 * np.log2(earliest / f0)
    if abs(drop) < PITCH_DROP_MIN_SEMITONES:
        return out
    out["pitch_start_hz"] = round(earliest, 1)
    out["pitch_drop_semitones"] = round(float(drop), 1)
    settled = body_from
    for at, f in track:                       # from the body backwards: the last place still settled
        if abs(12.0 * np.log2(f / f0)) > PITCH_SETTLE_SEMITONES:
            break
        settled = at
    out["pitch_settle_ms"] = round((settled - onset) * 1000.0 / sample_rate, 1)
    return out


def _spectrum(frames: np.ndarray, sample_rate: float) -> tuple[Optional[float], Optional[dict]]:
    """(centroid Hz, each band's share of the power in per cent) over 20 Hz - 20 kHz, from the
    channels' Welch spectra ADDED (a mono sum loses an anti-phase pair)."""
    nperseg = analysis._welch_nperseg(4096, frames.shape[0], sample_rate)
    if nperseg < 16:
        return None, None
    freqs, total = None, None
    for ch in range(frames.shape[1]):
        freqs, psd = analysis._welch_psd(frames[:, ch], sample_rate, nperseg)
        total = psd if total is None else total + psd
    inside = (freqs >= BAND_EDGES_HZ[0]) & (freqs < BAND_EDGES_HZ[-1])
    power = float(total[inside].sum())
    if power <= 0.0:
        return None, None
    centroid = float((freqs[inside] * total[inside]).sum() / power)
    centroid = round(centroid / 10.0) * 10.0 if centroid >= 1000.0 else round(centroid)
    shares = {label: round(100.0 * float(total[(freqs >= lo) & (freqs < hi)].sum()) / power, 1)
              for label, lo, hi in zip(BAND_LABELS, BAND_EDGES_HZ[:-1], BAND_EDGES_HZ[1:])}
    return float(centroid), shares


def _loudness(frames: np.ndarray, sample_rate: int, first: int, last: int) -> Optional[float]:
    """K-weighted loudness of the sounding span, ungated: BS.1770's weighting and channel
    sum over the mean square of what sounds. On a long steady signal it is the integrated
    loudness; under 0.4 s - where the gated meter has no block - it still answers."""
    weighted = analysis._k_weighted(analysis._lufs_arg(frames), sample_rate)[first:last + 1]
    if weighted.shape[0] == 0:
        return None
    mean_square = float(np.mean(weighted ** 2, axis=0).sum())
    if mean_square <= 0.0:
        return None
    return analysis._round_db(-0.691 + 10.0 * np.log10(mean_square))


def _loop(frames: np.ndarray, sample_rate: float, bpm: float, beats: float, peak: float) -> SampleLoop:
    n = frames.shape[0]
    expected_s = beats * 60.0 / bpm
    error_samples = n - int(round(expected_s * sample_rate))
    step = float(np.max(np.abs(frames[0] - frames[-1])))
    own = np.abs(np.diff(frames, axis=0))
    typical = float(np.percentile(own, 99.9)) if own.size else 0.0
    width = max(1, min(n // 2, int(round(SEAM_WINDOW_S * sample_rate))))
    head = float(np.sqrt(np.mean(frames[:width] ** 2)))
    tail = float(np.sqrt(np.mean(frames[-width:] ** 2)))
    return SampleLoop(
        bpm=round(float(bpm), 3), beats=round(float(beats), 3), expected_s=round(expected_s, 4),
        length_error_ms=round(1000.0 * error_samples / sample_rate, 2),
        length_error_samples=int(error_samples),
        seam_step_db=_db(step / peak) if step > 0.0 else None,
        seam_excess_db=(analysis._round_db(20.0 * np.log10(step / typical))
                        if step > 0.0 and typical > 0.0 else None),
        seam_level_db=(analysis._round_db(20.0 * np.log10(tail / head))
                       if head > 1e-9 and tail > 1e-9 else None))


def sample_profile(audio: np.ndarray, sample_rate: int, *, kind: str = "one_shot",
                   bpm: Optional[float] = None, beats: Optional[float] = None,
                   path: Optional[str] = None) -> SampleProfile:
    """One sound, measured. `kind` is "one_shot" or "loop"; a loop given its `bpm` and `beats`
    is also held to its length and read at its seam."""
    if kind not in ("one_shot", "loop"):
        raise ValueError('kind is "one_shot" or "loop"')
    if (bpm is None) != (beats is None):
        raise ValueError("a loop's length is read against bpm AND beats: give both, or neither")
    frames = analysis._frames_channels(audio)
    n, channels = frames.shape
    base = dict(path=path, kind=kind, sample_rate_hz=int(sample_rate), channels=int(channels),
                duration_s=round(n / float(sample_rate), 4))
    peak = float(np.max(np.abs(frames))) if n else 0.0
    if n < 8 or peak < analysis.SILENCE_PEAK:
        return SampleProfile(**base, silent=True)

    size = _magnitude(frames)
    onset = _first_at_or_over(size, peak * 10.0 ** (-ONSET_BELOW_PEAK_DB / 20.0)) or 0
    end = int(np.flatnonzero(size >= peak * 10.0 ** (-END_BELOW_PEAK_DB / 20.0))[-1])
    peak_at = int(np.argmax(size))
    level = float(np.percentile(size[onset:end + 1], ATTACK_LEVEL_PERCENTILE))
    a10 = _first_at_or_over(size, ATTACK_FROM * level)
    a90 = _first_at_or_over(size, ATTACK_TO * level)
    # what is still to come at each sample (the largest later one): past the peak it only falls
    to_come = np.maximum.accumulate(size[::-1])[::-1]
    fell = np.flatnonzero(to_come[peak_at:] < peak * 10.0 ** (-DECAY_DB / 20.0))
    # ...and a sound the FILE cuts off while it is still that loud has not decayed: it was cut
    end_width = max(1, min(n, int(round(END_WINDOW_S * sample_rate))))
    if fell.size and to_come[n - end_width] >= peak * 10.0 ** (-DECAY_DB / 20.0):
        fell = fell[:0]
    sounding = frames[onset:end + 1]
    rms = float(np.sqrt(np.mean(sounding ** 2)))
    end_rms = float(np.sqrt(np.mean(frames[-end_width:] ** 2)))
    centroid, shares = _spectrum(sounding, sample_rate)

    width_db = correlation = None
    if channels >= 2:
        mid = 0.5 * (frames[:, 0] + frames[:, 1])
        side = 0.5 * (frames[:, 0] - frames[:, 1])
        p_mid, p_side = float(np.mean(mid ** 2)), float(np.mean(side ** 2))
        if p_mid > 1e-18 or p_side > 1e-18:
            width_db = analysis._round_db(float(np.clip(
                10.0 * np.log10(max(p_side, 1e-18) / max(p_mid, 1e-18)), -40.0, 6.0)))
        correlation = analysis._stereo_correlation(frames)

    pitch: dict = {}
    if kind == "one_shot":
        mono = frames.mean(axis=1) if channels > 1 else frames[:, 0]
        if channels > 1 and float(np.max(np.abs(mono))) < 0.05 * peak:
            mono = frames[:, 0]                      # an anti-phase pair: read one side
        mono, pitch_rate = analysis._at_base_rate(mono, sample_rate)
        scale = pitch_rate / float(sample_rate)
        pitch = _pitch(mono, pitch_rate, int(onset * scale), min(mono.shape[0] - 1, int(end * scale)))

    loop = None
    if bpm is not None and beats is not None:
        if not (np.isfinite(bpm) and np.isfinite(beats) and bpm > 0.0 and beats > 0.0):
            raise ValueError("bpm and beats are positive numbers")
        loop = _loop(frames, sample_rate, float(bpm), float(beats), peak)

    return SampleProfile(
        **base, silent=False,
        leading_silence_ms=round(1000.0 * onset / sample_rate, 1),
        sounding_s=round((end - onset + 1) / float(sample_rate), 4),
        attack_ms=(round(1000.0 * max(0, a90 - a10) / sample_rate, 1)
                   if a10 is not None and a90 is not None else None),
        decay_ms=round(1000.0 * int(fell[0]) / sample_rate, 1) if fell.size else None,
        end_level_dbfs=_db(end_rms),
        sample_peak_dbfs=_db(peak),
        true_peak_dbtp=analysis._round_db(max(analysis._true_peak_dbtp(frames), LEVEL_FLOOR_DB)),
        rms_dbfs=_db(rms),
        loudness_lufs=_loudness(frames, int(sample_rate), onset, end),
        crest_db=analysis._round_db(20.0 * np.log10(peak / rms)) if rms > 0.0 else None,
        centroid_hz=centroid, band_share_pct=shares,
        dc_offset_pct=round(100.0 * float(np.max(np.abs(frames.mean(axis=0)))), 2),
        full_scale_samples=int(np.count_nonzero(np.max(np.abs(frames), axis=1) >= FULL_SCALE)),
        over_full_scale=bool(peak > 1.0),
        width_db=width_db, stereo_correlation=correlation, loop=loop, **pitch)
