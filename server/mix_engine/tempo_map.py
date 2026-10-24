"""The project's tempo map, read the way the DAW's engine reads it (Phase 63).

A tempo-synced LFO (the Auto Filter's) is a function of the song position in QUARTER NOTES, and
the native chain reads that position from Tracktion's `tempo::Sequence`. For the offline render
to land on the same point of the same cycle, this module builds the SAME sections from the same
points and reads quarter notes from them the same way - a transcription of
`tracktion_core/utilities/tracktion_Tempo.h` (`Sequence::Sequence`, `calcCurveBpm`,
`Position::getPPQTime`) and of `tracktion_Bezier.h`, quirks included:

* a ramped tempo (a curve that is not -1 or +1) is cut into `clamp(4 x beats to the next tempo
  point, 1, 100)` sections of constant tempo, each at the curve's bpm at its start;
* the FIRST section after any change point takes the curve's bpm at the TEMPO POINT's beat, not
  at the change's - a time-signature or key change inside a ramp holds the ramp's opening tempo
  until the next subdivision (Tracktion's own behaviour, kept, not corrected);
* `getBezierEnds` rounds two of its values to float, so this does too;
* PPQ = the section's PPQ at its start + 4 x beats since / the section's denominator.

The map reaches the brain as an engine fact (`session.json` `tempo_map`): tempo points
`[beat, bpm, curve]`, time signatures `[beat, numerator, denominator]` and the beats of the key
changes (they cut sections too). A leg holds this module to the engine's own readings
(`--tempo-map-parity` -> `server/tests/data/tempo_map_parity.json`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

DEFAULT_BPM = 120.0


def _f32(v: float) -> float:
    return float(np.float32(v))


def _bezier_point(x1: float, y1: float, x2: float, y2: float, c: float) -> tuple[float, float]:
    if y2 > y1:
        run, rise = x2 - x1, y2 - y1
        xc, yc = x1 + run / 2, y1 + rise / 2
        return xc - run / 2 * -c, yc + rise / 2 * -c
    run, rise = x2 - x1, y1 - y2
    xc, yc = x1 + run / 2, y2 + rise / 2
    return xc - run / 2 * -c, yc - rise / 2 * -c


def _bezier_ends(x1: float, y1: float, x2: float, y2: float, c: float):
    minic = (abs(c) - 0.5) * 2.0
    run = minic * (x2 - x1)
    rise = minic * ((y2 - y1) if y2 > y1 else (y1 - y2))
    if c > 0:
        return (x1 + run, _f32(y1), x2, _f32((y2 - rise) if y1 < y2 else (y2 + rise)))
    return (x1, _f32((y1 + rise) if y1 < y2 else (y1 - rise)), x2 - run, _f32(y2))


def _bezier_y_from_x(x: float, x1: float, y1: float, xb: float, yb: float,
                     x2: float, y2: float) -> float:
    if x1 == x2 or y1 == y2:
        return y1
    if x <= x1:
        return y1
    if x >= x2:
        return y2
    a = x1 - 2 * xb + x2
    b = -2 * x1 + 2 * xb
    c = x1 - x
    if a == 0:
        t = -c / b
    else:
        t = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
        if t < 0.0 or t > 1.0:
            t = (-b - math.sqrt(b * b - 4 * a * c)) / (2 * a)
    return (1 - t) ** 2 * y1 + 2 * t * (1 - t) * yb + t ** 2 * y2


def _curve_bpm(beat: float, t1: tuple, t2: tuple) -> float:
    """`Sequence::calcCurveBpm`: the tempo at `beat` on the curve from point t1 to t2."""
    b1, bpm1, c = t1[0], t1[1], t1[2]
    b2, bpm2 = t2[0], t2[1]
    x, y = _bezier_point(b1, bpm1, b2, bpm2, c)
    if -0.5 <= c <= 0.5:
        return _bezier_y_from_x(beat, b1, bpm1, x, y, b2, bpm2)
    x1e, y1e, x2e, y2e = _bezier_ends(b1, bpm1, b2, bpm2, c)
    if b1 <= beat <= x1e:
        return y1e
    if x2e <= beat <= b2:
        return y2e
    return _bezier_y_from_x(beat, x1e, y1e, x, y, x2e, y2e)


@dataclass(frozen=True)
class Section:
    start_s: float
    start_beat: float
    beats_per_second: float
    denominator: int
    ppq_at_start: float
    bpm: float


class TempoMap:
    """Built from the engine fact; reads quarter notes (and beats) at times in seconds."""

    def __init__(self, tempos: Sequence, time_sigs: Sequence, key_beats: Sequence = (),
                 beat_follows_time_sig: bool = True):
        tp = sorted(((float(b), float(bpm), _f32(float(c))) for b, bpm, c in tempos),
                    key=lambda t: t[0]) or [(0.0, DEFAULT_BPM, 0.0)]
        ts = sorted(((float(b), int(n), int(d)) for b, n, d in time_sigs),
                    key=lambda t: t[0]) or [(0.0, 4, 4)]
        keys = sorted({0.0, *(float(b) for b in key_beats)})
        beats = sorted({*(t[0] for t in tp), *(t[0] for t in ts), *keys})
        sections: list[Section] = []
        time = 0.0
        beat_num = 0.0
        ppq = 0.0
        ti, si = 0, 0
        cur_t = tp[ti]; ti += 1
        cur_s = ts[si]; si += 1
        for i, current in enumerate(beats):
            while ti < len(tp) and tp[ti][0] == current:
                cur_t = tp[ti]; ti += 1
            if si < len(ts) and ts[si][0] == current:
                cur_s = ts[si]; si += 1
            next_valid = ti < len(tp)
            bpm = _curve_bpm(cur_t[0], cur_t, tp[ti]) if next_valid else cur_t[1]
            subdivisions = 1
            if next_valid and cur_t[2] != -1.0 and cur_t[2] != 1.0:
                subdivisions = int(min(max(4.0 * (tp[ti][0] - current), 1.0), 100.0))
            num_beats = ((beats[i + 1] - current) / subdivisions) if i < len(beats) - 1 else 1.0e6
            den = cur_s[2]
            for _ in range(subdivisions):
                spb = (240.0 / (bpm * den)) if beat_follows_time_sig else (60.0 / bpm)
                sections.append(Section(time, beat_num, 1.0 / spb, den, ppq, bpm))
                ppq += 4 * num_beats / den
                time = time + num_beats * spb
                beat_num = beat_num + num_beats
                bpm = _curve_bpm(beat_num, cur_t, tp[ti]) if next_valid else cur_t[1]
        self.sections = sections
        self._starts = np.array([s.start_s for s in sections], dtype=np.float64)

    @classmethod
    def from_fact(cls, fact: Optional[dict]) -> "TempoMap":
        """The engine fact as `session.json` keeps it; a project without one is 4/4 at 120."""
        fact = fact or {}
        return cls(fact.get("tempos") or [[0.0, DEFAULT_BPM, 0.0]],
                   fact.get("time_sigs") or [[0.0, 4, 4]],
                   fact.get("key_beats") or ())

    def _index(self, seconds: np.ndarray) -> np.ndarray:
        # `details::toBeats` walks back from the last section to the first whose start is at or
        # before the time; section 0 takes everything before it.
        return np.clip(np.searchsorted(self._starts, seconds, side="right") - 1, 0, None)

    def ppq_at(self, seconds) -> np.ndarray:
        """Quarter notes at each time (`Position::getPPQTime`)."""
        t = np.asarray(seconds, dtype=np.float64)
        idx = self._index(t)
        start = self._starts[idx]
        bps = np.array([s.beats_per_second for s in self.sections])[idx]
        ppq0 = np.array([s.ppq_at_start for s in self.sections])[idx]
        den = np.array([s.denominator for s in self.sections], dtype=np.float64)[idx]
        return ppq0 + 4.0 * ((t - start) * bps) / den

    def beats_at(self, seconds) -> np.ndarray:
        """Beats at each time (`Sequence::toBeats`)."""
        t = np.asarray(seconds, dtype=np.float64)
        idx = self._index(t)
        start = self._starts[idx]
        bps = np.array([s.beats_per_second for s in self.sections])[idx]
        b0 = np.array([s.start_beat for s in self.sections])[idx]
        return b0 + (t - start) * bps

    def quarters_per_second_at(self, seconds) -> np.ndarray:
        t = np.asarray(seconds, dtype=np.float64)
        idx = self._index(t)
        bps = np.array([s.beats_per_second for s in self.sections])[idx]
        den = np.array([s.denominator for s in self.sections], dtype=np.float64)[idx]
        return 4.0 * bps / den
