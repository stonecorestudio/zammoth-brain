"""The tempo of a FULL-LENGTH song (2026-10-11, the mashup showcase's pre-flight).

Two released songs made to a click at 124.000 BPM read 123.725 and 123.833 on their masters,
and 123.808 and 123.784 on their drum stems after a split - the last at confidence 0.91, and
exactly 167 of the tempo envelope's 128-sample hops. The lag grid quantises the period to one
hop (0.6 % at 124 BPM), and the least-squares fit that was meant to mend it could not, for two
reasons that both grow with the take:

* it scores a grid by how near the onsets fall to BEAT lines, so on anything with eighths or
  sixteenths in it - half the onsets sit half a beat from a line whatever the period - no fit
  ever scored better than the coarse one, and the coarse one was kept;
* the coarse period's drift passes half a beat over a few minutes, so the beat indices the fit
  assigns are wrong before it starts.

Every fixture the detector had been sized on was 8 to 24 seconds long. Here the take is three
minutes, the answer is known to the sample, and the song is busy the way a song is: a kick on
every beat, a clap on two and four, hats on the sixteenths, a syncopated bass, an intro with no
drums and stray hits off the grid.
"""
from __future__ import annotations

import numpy as np
import pytest

from mix_engine import analysis


def _song(bpm: float, dur_s: float, sr: int = 44100, seed: int = 1, intro_s: float = 10.0,
          strays: float = 0.6, swing: float = 0.5, jitter_s: float = 0.0, triplets: bool = False,
          jump_at_s: float | None = None, step_to: float | None = None,
          drum_spans: tuple[tuple[float, float], ...] | None = None) -> np.ndarray:
    """A four-on-the-floor song whose beat b falls at exactly b x 60 / bpm seconds.

    `swing` places the off-eighth (0.5 = straight); `jitter_s` is per-hit timing noise;
    `triplets` plays the hats on the triplet; `jump_at_s` moves everything after it a third of
    a beat (an edit that is not on the grid); `step_to` is the tempo of the second half;
    `drum_spans` are the only stretches the drums play in."""
    rng = np.random.default_rng(seed)
    n = int(dur_s * sr)
    x = np.zeros(n)

    def add(at_s: float, hit: np.ndarray, gain: float = 1.0) -> None:
        if jitter_s:
            at_s = at_s + rng.normal(0.0, jitter_s)
        start = int(round(at_s * sr))
        if 0 <= start < n:
            end = min(n, start + len(hit))
            x[start:end] += gain * hit[: end - start]

    t = np.arange(int(0.3 * sr)) / sr
    kick = 0.9 * np.sin(2 * np.pi * (50 + 90 * np.exp(-t / 0.03)) * t) * np.exp(-t / 0.15)
    kick[: int(0.004 * sr)] += 0.5 * rng.standard_normal(int(0.004 * sr))
    t = np.arange(int(0.2 * sr)) / sr
    clap = 0.35 * rng.standard_normal(len(t)) * np.exp(-t / 0.05)
    t = np.arange(int(0.04 * sr)) / sr
    hat = 0.12 * np.diff(np.concatenate([[0.0], rng.standard_normal(len(t))])) * np.exp(-t / 0.008)
    t = np.arange(int(0.18 * sr)) / sr

    def pluck(hz: float) -> np.ndarray:
        return 0.3 * np.sin(2 * np.pi * hz * t) * np.exp(-t / 0.06) * np.minimum(t / 0.003, 1.0)

    def beat_time(b: int) -> float:
        if step_to is not None:
            first = int(dur_s / 2.0 / (60.0 / bpm))
            at = b * 60.0 / bpm if b <= first else first * 60.0 / bpm + (b - first) * 60.0 / step_to
        else:
            at = b * 60.0 / bpm
        if jump_at_s is not None and at >= jump_at_s:
            at += 60.0 / bpm / 3.0
        return at

    b = 0
    while beat_time(b) < dur_s:
        at, beat = beat_time(b), beat_time(b + 1) - beat_time(b)
        if at >= intro_s and (drum_spans is None or any(lo <= at < hi for lo, hi in drum_spans)):
            add(at, kick)
            if b % 2 == 1:
                add(at, clap)
            if triplets:
                add(at + beat / 3.0, hat, 0.7)
                add(at + 2.0 * beat / 3.0, hat, 1.4)
            else:
                add(at + swing * beat, hat, 1.6)
                for frac in (0.0, 0.25, 0.75):
                    add(at + frac * beat, hat, 0.7)
                if b % 4 in (0, 2):
                    add(at + 0.75 * beat, pluck(98.0))
                if b % 4 == 3:
                    add(at + 0.25 * beat, pluck(123.5))
        b += 1
    for at in rng.uniform(0.3, dur_s - 1.0, int(strays * dur_s)):     # a "vocal": off the grid
        add(at, pluck(float(rng.choice([440.0, 494.0, 587.0, 659.0]))), 0.8)
    return x


def _from_the_kick_ms(offset_s: float, bpm: float) -> float:
    """How far the reported beat line is from the kick's (the kicks fall at b x 60 / bpm)."""
    period = 60.0 / bpm
    return ((offset_s + period / 2.0) % period - period / 2.0) * 1000.0


def _fold(audio: np.ndarray, sr: int):
    """`_fold_grid` on the inputs `_tempo` hands it."""
    energy, _flux, hop_s = analysis._tempo_onset_envelopes(audio, sr)
    lag, _confidence = analysis._tempo_period(energy, hop_s)
    frame_s = analysis.TEMPO_FRAME * analysis._rate_factor(sr) / sr
    return analysis._fold_grid(analysis._onset_times_precise(audio, sr), energy, hop_s,
                               frame_s, lag * hop_s)


# ---------------------------------------------------------------------------------------------
# The defect: three minutes of a song made to a click
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("bpm, sr", [(124.0, 44100), (128.0, 44100), (97.3, 48000), (140.0, 44100)])
def test_a_full_length_song_reads_its_tempo_and_its_beat(bpm: float, sr: int):
    """Red on the detector as it was: 123.687, 128.465 and 139.588 - a third to half a BPM
    off at confidence 0.93 and up (a tenth of a BPM is a 50 ms drift over two minutes) - and
    97.3 at 48 k read right with its beat line 75 ms from the kick."""
    got = analysis._tempo(_song(bpm, 180.0, sr), sr)
    assert got is not None
    tempo, confidence, offset_s = got
    assert abs(tempo - bpm) <= 0.01, f"{bpm} BPM over three minutes read as {tempo}"
    assert confidence >= analysis.TEMPO_AUTO_ALIGN_CONFIDENCE
    assert abs(_from_the_kick_ms(offset_s, bpm)) <= 3.0, (
        f"the beat line is {_from_the_kick_ms(offset_s, bpm):+.1f} ms from the kick")


@pytest.mark.parametrize("how", [dict(swing=0.62), dict(jitter_s=0.012), dict(triplets=True)])
def test_a_feel_does_not_move_the_tempo(how: dict):
    """Swung off-eighths, a loose player (12 ms) and a triplet feel: the period is the same
    period. (Swing was -0.398 off, the loose take +0.529.)"""
    got = analysis._tempo(_song(124.0, 180.0, **how), 44100)
    assert got is not None
    assert abs(got[0] - 124.0) <= 0.02, f"{how}: read as {got[0]}"
    assert abs(_from_the_kick_ms(got[2], 124.0)) <= 5.0


def test_ten_minutes_cost_no_more_precision_than_three():
    got = analysis._tempo(_song(124.0, 600.0), 44100)
    assert got is not None and abs(got[0] - 124.0) <= 0.005, got


# ---------------------------------------------------------------------------------------------
# What it must NOT claim
# ---------------------------------------------------------------------------------------------

def test_a_take_with_two_grids_is_not_given_one_to_three_decimals():
    """An edit that is not on the grid, and a tempo that steps a third of a BPM at the half:
    folded, such a take piles up at two periods nearly as well as at one. The fold says nothing
    and the reading is whatever it was before - coarse, and not dressed as exact."""
    assert _fold(_song(124.0, 180.0, jump_at_s=90.0), 44100) is None
    assert _fold(_song(124.0, 180.0, step_to=124.4), 44100) is None


def test_two_islands_of_drums_far_apart_are_left_to_the_coarse_reading():
    """Fifteen seconds of drums, two minutes of nothing rhythmic, fifteen more: the two islands
    fringe (two slits) and a wrong period scores 0.93 of the right one. Declined, not guessed."""
    assert _fold(_song(124.0, 180.0, strays=0.0, drum_spans=((10.0, 25.0), (150.0, 165.0))),
                 44100) is None


@pytest.mark.parametrize("bpm, dur_s, sr, was", [
    (124.0, 12.0, 44100, 122.872),
    (140.0, 12.0, 48000, 140.06),
    (128.0, 45.0, 44100, 127.821),
])
def test_a_short_busy_loop_reads_its_tempo_too(bpm: float, dur_s: float, sr: int, was: float):
    """The fit against beat lines was no better on a few bars than on a song: 190 synthetic
    loops of 6-32 s read up to 3.2 BPM off (`was` is what each of these read, on that code).
    Folded, the same loops read to 0.008."""
    got = analysis._tempo(_song(bpm, dur_s, sr, intro_s=0.0), sr)
    assert got is not None
    assert abs(got[0] - bpm) <= 0.02, f"{bpm} BPM over {dur_s:.0f} s read as {got[0]} (was {was})"
    assert abs(_from_the_kick_ms(got[2], bpm)) <= 3.0


@pytest.mark.parametrize("bpm, dur_s, sr, was", [
    (124.0, 12.0, 44100, (122.872, 0.94, 0.2198)),
    (100.0, 10.0, 48000, (100.0, 0.92, 0.288)),
])
def test_where_the_fold_says_nothing_the_old_fit_answers_as_it_did(monkeypatch, bpm, dur_s, sr, was):
    """The least-squares fit under the fold is not touched: with the fold silenced, these read
    what they read on the detector before it (recorded on that code, 2026-10-11)."""
    monkeypatch.setattr(analysis, "_fold_grid", lambda *args: None)
    assert analysis._tempo(_song(bpm, dur_s, sr, intro_s=0.0), sr) == was


def test_too_few_onsets_to_fold_is_left_alone():
    onsets = np.arange(8) * 0.5
    assert analysis._fold_grid(onsets, np.ones(4096), 128 / 44100, 512 / 44100, 0.5) is None
