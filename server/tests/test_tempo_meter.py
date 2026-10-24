"""A dotted figure is not the beat (analysis v13, 2026-10-11).

The second mashup showcase's pre-flight: a released song at 172 BPM (felt at 86) read 114.66 -
two thirds of it - at confidence 0.69 on its own drum stem. Kicks three sixteenths apart
correlate at that lag as well as the beat does, and the prior, centred on 120, prefers it. The
tempo stage now asks which candidate's BARS the pattern repeats on (`TEMPO_METER_*`).

Every fixture is synthesised here. The control for the fixtures that reproduce the misreading
is the same audio read with the rule switched off (a margin nothing clears): it must read at
4/3 of its tempo, or the fixture proves nothing. The grooves built to BREAK the rule - real
triplet feels, 6/8, a waltz, a tresillo, straight beats - must read exactly what they read
without it.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from mix_engine import analysis


def _groove(step_s: float, steps_per_bar: int, bars: list[dict], seconds: float = 60.0,
            sample_rate: int = 44100) -> np.ndarray:
    """Kick (`k`), snare (`s`) and hats (`h`; `hg` = quiet ones) on a grid of `step_s`, the
    bars of `bars` in turn. The same every run."""
    rng = np.random.default_rng(1)
    n = int(seconds * sample_rate)
    out = np.zeros(n)

    def put(t: float, kind: str, amp: float) -> None:
        i = int(round(t * sample_rate))
        if kind == "k":
            m = int(0.18 * sample_rate)
            tt = np.arange(m) / sample_rate
            hit = np.sin(2 * np.pi * (48 + 60 * np.exp(-tt / 0.02)) * tt) * np.exp(-tt / 0.07)
        elif kind == "s":
            m = int(0.12 * sample_rate)
            tt = np.arange(m) / sample_rate
            band = signal.butter(2, [300, 6000], btype="band", fs=sample_rate, output="sos")
            hit = signal.sosfilt(band, rng.standard_normal(m)) * np.exp(-tt / 0.035) * 0.8
        else:
            m = int(0.03 * sample_rate)
            tt = np.arange(m) / sample_rate
            high = signal.butter(2, 7000, btype="high", fs=sample_rate, output="sos")
            hit = signal.sosfilt(high, rng.standard_normal(m)) * np.exp(-tt / 0.008) * 0.35
        m = min(m, n - i)
        if m > 0:
            out[i:i + m] += amp * hit[:m]

    bar, t0 = 0, 0.31
    while t0 < seconds:
        for kind, steps in bars[bar % len(bars)].items():
            for step in steps:
                put(t0 + step * step_s, kind[0], 0.6 if kind.endswith("g") else 1.0)
        t0 += steps_per_bar * step_s
        bar += 1
    return (out / np.max(np.abs(out)) * 0.7).astype(np.float32)


def _without_the_rule(monkeypatch, mono: np.ndarray, sample_rate: int):
    with monkeypatch.context() as patch:
        patch.setattr(analysis, "TEMPO_METER_MARGIN", 9.0)      # nothing repeats that much better
        return analysis._tempo(mono, sample_rate)


# Straight sixteenths with a dotted kick figure: what a trap or hip-hop beat at 80-90 is.
_QUIET_HATS = list(range(16))
DOTTED = {
    "kicks 0 3 6 12, half-time clap": [
        {"k": [0, 3, 6, 12], "s": [12], "hg": _QUIET_HATS},
        {"k": [0, 3, 6, 10, 12], "s": [12], "hg": _QUIET_HATS}],
    "a kick every three sixteenths, backbeat": [
        {"k": [0, 3, 6, 9, 12, 15], "s": [4, 12], "hg": _QUIET_HATS}],
    "kicks 0 3 6 10 13, clap on three": [
        {"k": [0, 3, 6, 10, 13], "s": [8], "hg": list(range(0, 16, 2))}],
}


@pytest.mark.parametrize("name", list(DOTTED))
@pytest.mark.parametrize("feel_bpm,sample_rate", [(80.0, 44100), (86.0, 44100), (86.0, 48000), (90.0, 44100)])
def test_a_dotted_kick_figure_is_not_the_beat(monkeypatch, name, feel_bpm, sample_rate):
    mono = _groove(60.0 / feel_bpm / 4.0, 16, DOTTED[name], sample_rate=sample_rate)

    was = _without_the_rule(monkeypatch, mono, sample_rate)
    assert was is not None and abs(was[0] / feel_bpm - 4.0 / 3.0) < 0.03, (
        f"the fixture must reproduce the misreading (4/3 of {feel_bpm}), or it proves nothing: {was}")

    read = analysis._tempo(mono, sample_rate)
    assert read is not None
    assert min(abs(read[0] - feel_bpm), abs(read[0] - 2 * feel_bpm)) < 0.5, (
        f"{name} at {feel_bpm} BPM read {read[0]:.2f} (was {was[0]:.2f})")


def test_the_misreading_was_a_confident_one(monkeypatch):
    """Not a reading the 0.45 cap already marked as undecided: 0.72, and wrong."""
    mono = _groove(60.0 / 86.0 / 4.0, 16, DOTTED["a kick every three sixteenths, backbeat"])
    was = _without_the_rule(monkeypatch, mono, 44100)
    assert was is not None and was[1] > analysis.TEMPO_AUTO_ALIGN_CONFIDENCE, was


# Built to break the rule: where three really is the beat's own division, or its bar.
_SHUFFLE_HATS = [0, 2, 3, 5, 6, 8, 9, 11]
SAFE = [
    ("shuffle 12/8", 3, 12, [{"k": [0, 6], "s": [3, 9], "h": _SHUFFLE_HATS}], (72.0, 93.0, 120.0)),
    ("shuffle, busy kick", 3, 12, [{"k": [0, 5, 6], "s": [3, 9], "h": _SHUFFLE_HATS},
                                   {"k": [0, 6, 8], "s": [3, 9], "h": list(range(12))}], (80.0, 105.0)),
    ("six-eight", 3, 6, [{"k": [0], "s": [3], "h": list(range(6))}], (72.0, 100.0)),
    ("waltz", 2, 6, [{"k": [0], "s": [2, 4], "h": list(range(6))}], (90.0, 150.0)),
    ("tresillo 3-3-2", 2, 8, [{"k": [0, 3, 6], "s": [4], "h": list(range(8))}], (100.0, 126.0)),
    ("straight rock", 4, 16, [{"k": [0, 8], "s": [4, 12], "h": list(range(0, 16, 2))}], (100.0, 140.0)),
    ("four on the floor", 4, 16, [{"k": [0, 4, 8, 12], "s": [4, 12], "h": [2, 6, 10, 14]}], (120.0,)),
]


@pytest.mark.parametrize("name,steps_per_beat,steps_per_bar,bars,tempi", SAFE, ids=[row[0] for row in SAFE])
def test_what_read_right_does_not_move(monkeypatch, name, steps_per_beat, steps_per_bar, bars, tempi):
    for bpm in tempi:
        mono = _groove(60.0 / bpm / steps_per_beat, steps_per_bar, bars)
        was = _without_the_rule(monkeypatch, mono, 44100)
        read = analysis._tempo(mono, 44100)
        assert read == was, f"{name} at {bpm}: {was} -> {read}"
        assert read is not None and abs(read[0] - bpm) < 0.6, f"{name} at {bpm} reads {read}"
