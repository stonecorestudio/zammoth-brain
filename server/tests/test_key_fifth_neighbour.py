"""The key a fifth away (analysis v15, 2026-11-14).

A reel in E major that leans on its fifth read "B maj 0.78" over "E maj 0.72", and the DAW
aligned the grid to B. The chroma below is that song's own (the finished mix, measured);
the rule is tested on it, on its mirror image, and on two songs it must leave alone.
"""
from __future__ import annotations

import numpy as np
import pytest

from mix_engine import analysis

# C, C#, D, D#, E, F, F#, G, G#, A, A#, B - measured on the three finished mixes of the session
REEL_E_MAJOR = [0.0562, 0.0435, 0.0514, 0.0604, 0.1393, 0.0566, 0.1088, 0.0512, 0.0586, 0.0883, 0.0604, 0.2254]
MENU_D_MAJOR = [0.0317, 0.0746, 0.2302, 0.0198, 0.1119, 0.0287, 0.1019, 0.0685, 0.0395, 0.1507, 0.033, 0.1094]
COURT_G_MINOR = [0.0863, 0.0416, 0.1705, 0.0575, 0.0541, 0.0481, 0.0737, 0.146, 0.0446, 0.1272, 0.1029, 0.0475]


def _read(monkeypatch: pytest.MonkeyPatch, chroma: list[float]):
    monkeypatch.setattr(analysis, "_chroma", lambda mono, sr: np.asarray(chroma, dtype=float))
    monkeypatch.setattr(analysis, "_sounding_pitch_classes", lambda mono, sr: 7)
    monkeypatch.setattr(analysis, "_bass_pitch_classes", lambda mono, sr: None)
    return analysis._key(np.zeros(48000), 48000)


def test_a_song_that_leans_on_its_fifth_is_read_in_its_own_key(monkeypatch):
    key = _read(monkeypatch, REEL_E_MAJOR)
    assert (key.tonic, key.mode) == ("E", "maj")          # it read B maj
    assert key.confidence == 0.72                         # the named key's own correlation


def test_the_same_song_with_the_leading_note_instead_stays_in_the_upper_key(monkeypatch):
    swapped = list(REEL_E_MAJOR)
    swapped[9], swapped[10] = swapped[10], swapped[9]     # A# sounds, A does not: that is B major
    key = _read(monkeypatch, swapped)
    assert (key.tonic, key.mode) == ("B", "maj")


def test_two_notes_that_neither_lead_settle_nothing(monkeypatch):
    even = list(REEL_E_MAJOR)
    even[9] = even[10] = (even[9] + even[10]) / 2
    key = _read(monkeypatch, even)
    assert (key.tonic, key.mode) == ("B", "maj")          # the profiles' own answer, as before


@pytest.mark.parametrize("chroma, tonic, mode, confidence", [
    (MENU_D_MAJOR, "D", "maj", 0.95),
    (COURT_G_MINOR, "G", "min", 0.77),
])
def test_a_key_with_no_near_tie_reads_what_it_read(monkeypatch, chroma, tonic, mode, confidence):
    key = _read(monkeypatch, chroma)
    assert (key.tonic, key.mode, key.confidence) == (tonic, mode, confidence)


def test_the_note_pair_is_the_one_the_two_keys_do_not_share():
    major = {0, 2, 4, 5, 7, 9, 11}
    minor = {0, 2, 3, 5, 7, 8, 10}
    for mode, scale in (("maj", major), ("min", minor)):
        for lower in range(12):
            low = {(lower + d) % 12 for d in scale}
            up = {(lower + 7 + d) % 12 for d in scale}
            only_low, only_up = analysis._fifth_pair_notes(lower, mode)
            assert low - up == {only_low} and up - low == {only_up}
