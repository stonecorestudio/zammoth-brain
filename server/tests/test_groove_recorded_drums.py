"""Extract Groove on recorded drums (2026-10-09; James, of the one item the hand-driven QA left
open: "let's take care of this").

On a commercial pack of 79 real drum loops, 6 extracted and none of the twenty full drum loops.
Two causes, both measured before anything was changed (the tables are in
docs/audits/user-qa-2026-10-08/QA-log.md, *Extract Groove on recorded drums*):

1. **The onset detector reported a kick twice.** Its window is 11.6 ms and a kick's fundamental
   has a 21 ms period, so the magnitude of a REAL signal's short-time spectrum ripples at twice
   the tone's frequency (the positive- and negative-frequency images beating in one bin) - and
   the tallest ripple past the 50 ms refractory was a second "onset", 50 to 85 ms after the
   first, and often the only one (25 ms late). The flux is now taken on the ANALYTIC signal,
   whose spectrum has no mirror image, with a term that keeps a click-less sub kick as visible
   as a noise burst of the same energy, and a hit masks a much weaker one just behind it.
2. **The grid rule called every deviation a lateness.** A 32nd pickup forced the whole part onto
   a 32nd grid ("too sparse"), and - with the pairs gone - a syncopation read as a note half a
   cell early. A deviation is now a VOTE only if it is a feel: a small push or drag, or a swing
   (late, on an off cell). Anything else is an ornament of a finer grid: it is left out, and too
   many of them send the part to that finer grid.

The fixtures are synthesised (a swept sub kick with no click, noise snares and hats); the real
pack is one folder in ~/Downloads and is not in the repo.
"""
from __future__ import annotations

import numpy as np
import pytest

from mix_engine import analysis
from mix_engine.models import ANALYSIS_VERSION

SR = 44100
BPM = 100.0
BEAT = 60.0 / BPM
CELL = BEAT / 4.0                                   # a sixteenth
LEAD = 0.10                                         # the loop starts here: no hit at sample 0


def _kick(sr: int = SR) -> np.ndarray:
    """A sub kick with NO click: 110 -> 48 Hz, 8 ms rise, a tail a 512-sample window ripples on."""
    n = int(0.40 * sr)
    t = np.arange(n) / sr
    phase = 2.0 * np.pi * np.cumsum(48.0 + 62.0 * np.exp(-t / 0.03)) / sr
    return 0.8 * np.sin(phase) * np.minimum(1.0, t / 0.008) * np.exp(-t / 0.20)


def _noise_hit(rng: np.random.Generator, level: float, decay_s: float, sr: int = SR) -> np.ndarray:
    n = int(6.0 * decay_s * sr)
    return level * rng.standard_normal(n) * np.exp(-np.arange(n) / (decay_s * sr))


def _loop(hits: list[tuple[float, str]], bars: int = 4, seed: int = 3, sr: int = SR
          ) -> np.ndarray:
    """`hits` = (time in sixteenths within ONE bar, sound). Repeated for `bars` bars."""
    rng = np.random.default_rng(seed)
    out = np.zeros(int((LEAD + bars * 16 * CELL) * sr) + sr)
    for bar in range(bars):
        for cells, sound in hits:
            start = int(round((LEAD + (bar * 16 + cells) * CELL) * sr))
            hit = (_kick(sr) if sound == "kick"
                   else _noise_hit(rng, 0.5, 0.030, sr) if sound == "snare"
                   else _noise_hit(rng, 0.25, 0.006, sr) if sound == "hat"
                   else _noise_hit(rng, 0.12, 0.004, sr))                # "ghost"
            out[start:start + hit.size] += hit[: out.size - start]
    return out


def _onsets(audio: np.ndarray) -> np.ndarray:
    return analysis._onset_times_precise(audio, SR)


# ---------------------------------------------------------------- the detector ----

def test_a_sub_kick_is_one_onset_at_its_attack():
    """The shipped detector answered two for each kick, or one 25 ms late."""
    audio = _loop([(0.0, "kick"), (6.0, "kick"), (10.0, "kick")])
    truth = np.array([LEAD + (bar * 16 + c) * CELL for bar in range(4) for c in (0.0, 6.0, 10.0)])
    found = _onsets(audio)
    assert found.size == truth.size, f"{found.size} onsets for {truth.size} kicks"
    late_ms = (found - truth) * 1000.0
    assert np.all(np.abs(late_ms) < 12.0), np.round(late_ms, 1)
    assert float(np.ptp(late_ms)) < 4.0, "the same kick is timed the same way every time"


def test_a_click_less_sub_kick_is_found_beside_hats_louder_than_it():
    hits = [(0.0, "kick"), (8.0, "kick")] + [(c, "snare") for c in (2.0, 4.0, 6.0, 10.0, 12.0, 14.0)]
    audio = _loop(hits)
    found = _onsets(audio)
    kicks = np.array([LEAD + (bar * 16 + c) * CELL for bar in range(4) for c in (0.0, 8.0)])
    nearest = np.array([np.min(np.abs(found - t)) for t in kicks])
    assert np.all(nearest < 0.015), np.round(nearest * 1000.0, 1)
    assert found.size == 4 * len(hits)


def test_a_hat_is_timed_as_it_was():
    """The control: nothing below 300 Hz, nothing for the new terms to move."""
    audio = _loop([(float(c), "hat") for c in range(0, 16, 2)])
    truth = np.array([LEAD + (bar * 16 + c) * CELL for bar in range(4) for c in range(0, 16, 2)])
    found = _onsets(audio)
    assert found.size == truth.size
    assert np.all(np.abs(found - truth) < 0.002)


def test_a_much_weaker_peak_just_behind_a_hit_is_its_tail_not_a_note():
    rng = np.random.default_rng(5)
    audio = np.zeros(SR * 2)
    for k in range(6):
        start = int((0.2 + 0.25 * k) * SR)
        loud = _noise_hit(rng, 0.6, 0.010)
        audio[start:start + loud.size] += loud
        tail = _noise_hit(rng, 0.6 * 0.12, 0.004)             # 18 dB down, 70 ms behind
        audio[start + int(0.070 * SR): start + int(0.070 * SR) + tail.size] += tail
    assert _onsets(audio).size == 6


def test_two_hits_of_like_strength_a_32nd_apart_are_two():
    """The mask is for a tail: a real pickup is not 10 dB under the hit before it."""
    audio = _loop([(0.0, "hat"), (3.5, "hat"), (4.0, "hat"), (8.0, "hat"), (11.5, "hat"), (12.0, "hat")])
    assert _onsets(audio).size == 4 * 6


@pytest.mark.parametrize("rate", [44100, 48000, 88200, 96000])
def test_the_detector_hears_the_same_loop_the_same_way_at_every_rate(rate: int):
    """The window was 512 SAMPLES at every rate: half as long in time at 96 k, where a sub
    kick's 8 ms rise is spread over twice the hops and was not found at all (8 of 8 missed)."""
    hits = [(0.0, "kick"), (10.0, "kick"), (4.0, "snare"), (12.0, "snare")]
    hits += [(float(c), "hat") for c in range(16) if c not in (0, 4, 10, 12)]
    audio = _loop(hits, sr=rate)
    found = analysis._onset_times_precise(audio, rate)
    assert found.size == 4 * len(hits), f"{found.size} onsets for {4 * len(hits)} hits at {rate} Hz"
    kicks = np.array([LEAD + (bar * 16 + c) * CELL for bar in range(4) for c in (0.0, 10.0)])
    late_ms = np.array([found[np.argmin(np.abs(found - t))] - t for t in kicks]) * 1000.0
    assert np.all(np.abs(late_ms) < 12.0), np.round(late_ms, 1)
    hats = np.array([LEAD + (bar * 16 + c) * CELL for bar in range(4) for c in (2.0, 6.0, 8.0, 14.0)])
    hat_ms = np.array([found[np.argmin(np.abs(found - t))] - t for t in hats]) * 1000.0
    assert np.all(np.abs(hat_ms) < 2.0), np.round(hat_ms, 1)


def test_the_analytic_signal_in_blocks_is_the_whole_one(monkeypatch):
    rng = np.random.default_rng(2)
    t = np.arange(SR * 3) / SR
    x = 0.5 * np.sin(2 * np.pi * 47.0 * t) + 0.1 * rng.standard_normal(t.size)
    whole = analysis._analytic_signal(x)
    monkeypatch.setattr(analysis, "ANALYTIC_BLOCK", 1 << 15)
    monkeypatch.setattr(analysis, "ANALYTIC_MARGIN", 1 << 13)
    blocks = analysis._analytic_signal(x)
    inner = slice(1 << 13, x.size - (1 << 13))                # the two ends see different padding
    assert float(np.max(np.abs(whole[inner] - blocks[inner]))) < 1e-2      # margins a quarter of the real ones
    assert whole.dtype == np.complex64


def test_a_low_tone_makes_no_onsets_once_it_has_started():
    """What the ripple was: a steady 48 Hz tone, and an onset every few periods of it."""
    t = np.arange(SR * 2) / SR
    tone = 0.7 * np.sin(2 * np.pi * 48.0 * t) * np.minimum(1.0, t / 0.01)
    flux = analysis._onset_flux(tone, SR)
    hop_s = analysis.TIMING_HOP / SR
    steady = flux[int(0.2 / hop_s): int(1.8 / hop_s)]
    assert float(steady.max()) < 0.02 * float(flux.max()), float(steady.max() / flux.max())


# ---------------------------------------------------------------- the grid ----

SWING = 0.24                                        # an off sixteenth, a quarter of a cell late: 62 %


def _swung_loop(swing: float = SWING, pickups: bool = False) -> np.ndarray:
    hits: list[tuple[float, str]] = [(0.0, "kick"), (10.0, "kick"), (4.0, "snare"), (12.0, "snare")]
    hits += [(float(c) + (swing if c % 2 else 0.0), "hat") for c in range(16)
             if c not in (0, 4, 10, 12)]
    if pickups:
        hits += [(3.5, "ghost"), (11.5, "ghost")]              # a 32nd before each snare
    return _loop(hits)


def _extract(audio: np.ndarray, bpm: float = BPM):
    return analysis.extract_groove(audio[:, None], SR, "loop", bpm=bpm, beat_offset_s=LEAD)


def test_swung_sixteenths_over_sub_kicks_come_out_as_their_swing():
    groove = _extract(_swung_loop())
    assert groove.reason is None, groove.reason
    assert groove.notes_per_beat == 4
    lateness = np.array(groove.latenesses)
    off, on = lateness[1::2], lateness[0::2]
    assert np.all(np.abs(off - 2.0 * SWING) < 0.08), np.round(off, 2)      # Tracktion's unit: 2 x cells
    assert np.all(np.abs(on) < 0.12), np.round(on, 2)


def test_a_32nd_pickup_is_an_ornament_not_a_sixteenth_played_early():
    groove = _extract(_swung_loop(pickups=True))
    assert groove.reason is None, groove.reason
    assert groove.notes_per_beat == 4, "two pickups a bar sent the whole part to a 32nd grid"
    lateness = np.array(groove.latenesses)
    assert float(lateness.min()) > -0.15, np.round(lateness, 2)
    assert np.all(np.abs(lateness[1::2] - 2.0 * SWING) < 0.08)


def test_a_syncopation_is_not_read_as_a_beat_played_early():
    """Kicks on 1 and the sixteenth BEFORE 3, snares on 2 and 4: on a quarter grid the second
    kick is 'a quarter of a beat early'. That is the rhythm, not a feel to lend."""
    groove = _extract(_loop([(0.0, "kick"), (7.0, "kick"), (4.0, "snare"), (12.0, "snare"),
                             (2.0, "hat"), (6.0, "hat"), (10.0, "hat"), (14.0, "hat")]))
    if groove.reason is None:
        assert float(np.min(groove.latenesses)) > -0.2, groove.latenesses
        assert float(np.max(groove.latenesses)) < 0.2, groove.latenesses
    else:
        assert "dead on the grid" in groove.reason or "sparse" in groove.reason, groove.reason


def test_swung_eighths_are_a_lateness_on_the_eighth_grid():
    third = 2.0 / 3.0                               # the off eighth on the last triplet: a third of ITS cell late
    hits = [(0.0, "kick"), (8.0, "kick"), (4.0, "snare"), (12.0, "snare")]
    hits += [(c + third, "hat") for c in (2.0, 6.0, 10.0, 14.0)]
    groove = _extract(_loop(hits))
    assert groove.reason is None, groove.reason
    assert groove.notes_per_beat == 2
    lateness = np.array(groove.latenesses)
    assert np.all(np.abs(lateness[1::2] - 2.0 / 3.0) < 0.08), np.round(lateness, 2)
    assert np.all(np.abs(lateness[0::2]) < 0.12)


def test_a_part_that_is_late_everywhere_keeps_its_lean_in_the_template():
    """An existing rule, held: a uniformly late take has a feel worth borrowing
    (test_agent_mastery: 'keeps the lean but does not call it swing')."""
    groove = _extract(_loop([(float(c) + 0.11, "hat") for c in range(16)]))     # 16.5 ms late
    assert groove.reason is None, groove.reason
    assert np.all(np.abs(np.array(groove.latenesses) - 0.22) < 0.05), groove.latenesses
    assert 13.0 < groove.bias_ms < 20.0


def test_a_part_on_the_grid_is_still_told_so():
    groove = _extract(_loop([(float(c), "hat") for c in range(16)]))
    assert groove.reason is not None and "dead on the grid" in groove.reason, groove.reason


def test_what_votes_is_a_push_or_a_swing_and_nothing_else():
    onsets = np.array([0.0, 1.24, 2.0, 2.9, 3.52, 4.0, 5.24, 6.0, 7.24]) * CELL
    chosen = analysis._choose_groove_grid(onsets, BPM, 0.0)
    assert chosen is not None
    subdivision, deviations, index = chosen
    assert subdivision == 4
    assert list(index) == [0, 1, 2, 3, 4, 5, 6, 7]              # 3.52 is the ornament: 4 has its own hit
    assert abs(float(deviations[1]) * 4 - 0.24) < 1e-9          # late, on an off cell: a swing
    assert abs(float(deviations[3]) * 4 + 0.10) < 1e-9          # a tenth of a cell early: a push


def test_recorded_drums_moved_the_analysis_version():
    assert ANALYSIS_VERSION >= 11
