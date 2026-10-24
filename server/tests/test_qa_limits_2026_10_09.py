"""The limits the user QA of 2026-10-08 left, taken care of on 2026-10-09 - the brain's half.

James, of the list "known limits, left on purpose": "we need to take care of these". Two of the
six are the brain's:

* **A key read as its relative** (O12). C major and A minor are the same seven notes; where the
  two fits are close the bass settles it (analysis v10, `analysis._bass_pitch_classes`).
* **The chain behind D37.** After an undo the app reports the routes the engine holds and the
  brain follows. A report that left a group with nothing in it used to BURY the group - and the
  re-read then removed its track and cleared the undo history. One wrong route after a Cmd-Z
  (D37 was one cause; nothing guarded against a second) destroyed a group for good. The report
  of routes is not a removal: the group stays while its track stands.

The DAW's halves are the console legs ``--qa-routing-guard`` and the rest of
``QaLimitsLegs.h``.
"""
from __future__ import annotations

import numpy as np
import pytest

from mix_engine import analysis
from mix_engine.models import ANALYSIS_VERSION
from mix_engine.session import Session
from orchestrator import daw_bridge

SR = 48000
PITCH_CLASSES = analysis.PITCH_CLASSES


# ---------------------------------------------------------------- the relative key ----

def _hz(pitch_class: str, octave: int) -> float:
    return 440.0 * 2.0 ** ((PITCH_CLASSES.index(pitch_class) - 9 + 12 * (octave - 4)) / 12.0)


def _tone(freq: float, dur: float, level: float = 0.2,
          partials: tuple[float, ...] = (1.0, 0.5, 0.3)) -> np.ndarray:
    t = np.arange(int(SR * dur)) / SR
    x = sum(a * np.sin(2 * np.pi * (i + 1) * freq * t) for i, a in enumerate(partials))
    ramp = int(0.010 * SR)
    env = np.ones_like(x)
    env[:ramp] = np.linspace(0.0, 1.0, ramp)
    env[-ramp:] = np.linspace(1.0, 0.0, ramp)
    return level * x * env


def _triad(root: str, quality: str, dur: float) -> np.ndarray:
    idx = PITCH_CLASSES.index(root)
    return sum(_tone(_hz(PITCH_CLASSES[(idx + s) % 12], 4), dur)
               for s in (0, 4 if quality == "maj" else 3, 7))


def _chords(chords: list[tuple[str, str]], chord_s: float = 2.0) -> np.ndarray:
    return np.concatenate([_triad(root, quality, chord_s) for root, quality in chords])


def _pedal(pitch_class: str, octave: int, seconds: float) -> np.ndarray:
    """A bass note with no partials: a sine, so nothing of it reaches the chroma's band from
    below 65 Hz - what the bass register says is then said by the bass register alone."""
    return _tone(_hz(pitch_class, octave), seconds, level=0.5, partials=(1.0,))


#: Am - F - C - G: the four chords of half the songs ever written, and the same seven notes
#: whether the song is in A minor or in C major.
AMBIGUOUS = [("A", "min"), ("F", "maj"), ("C", "maj"), ("G", "maj")]


def _named(audio: np.ndarray) -> tuple[str, str]:
    key = analysis._key(audio, SR)
    assert key is not None
    return key.tonic, key.mode


def test_the_four_chords_alone_read_as_they_did():
    """The control, and the reason the rule exists: the chords alone lean on C."""
    assert _named(_chords(AMBIGUOUS)) == ("C", "maj")


def test_a_bass_line_on_the_relative_minors_tonic_settles_a_near_tie():
    upper = _chords(AMBIGUOUS)
    audio = upper + _pedal("A", 1, upper.size / SR)            # A1, 55 Hz: under the chroma's band
    assert _named(audio) == ("A", "min")


def test_a_bass_line_on_the_named_keys_tonic_changes_nothing():
    upper = _chords(AMBIGUOUS)
    assert _named(upper + _pedal("C", 2, upper.size / SR)) == ("C", "maj")


def test_the_same_rule_settles_it_the_other_way():
    """A minor named by a little, the bass on C: C major."""
    upper = _chords([("A", "min"), ("F", "maj"), ("C", "maj"), ("F", "maj")])
    alone = analysis._key(upper, SR)
    assert alone is not None and (alone.tonic, alone.mode) == ("A", "min"), alone
    chroma = analysis._chroma(upper, SR)
    lead = alone.confidence - float(np.corrcoef(chroma, analysis.KRUMHANSL_MAJOR)[0, 1])
    assert lead < analysis.KEY_RELATIVE_MARGIN, lead        # the fixture is a near-tie
    assert _named(upper + _pedal("C", 2, upper.size / SR)) == ("C", "maj")


def test_a_relative_key_that_does_not_fit_is_not_named_whatever_the_bass_does():
    """Am - F - E - C: the E major chord's G sharp is A harmonic minor's, and C major does not
    clear the confidence floor. A C in the bass does not make it C major."""
    upper = _chords([("A", "min"), ("F", "maj"), ("E", "maj"), ("C", "maj")])
    assert _named(upper) == ("A", "min")
    assert _named(upper + _pedal("C", 2, upper.size / SR)) == ("A", "min")


def test_a_key_that_is_not_a_near_tie_is_not_moved_by_the_bass():
    """I-IV-V-I in C major leads A minor by far more than the margin: an A in the bass - a
    pedal on the sixth - is a colour, not a key."""
    upper = _chords([("C", "maj"), ("F", "maj"), ("G", "maj"), ("C", "maj")])
    chroma = analysis._chroma(upper, SR)
    lead = (float(np.corrcoef(chroma, analysis.KRUMHANSL_MAJOR)[0, 1])
            - float(np.corrcoef(chroma, np.roll(analysis.KRUMHANSL_MINOR, 9))[0, 1]))
    assert lead >= analysis.KEY_RELATIVE_MARGIN, lead       # the fixture is what it claims
    assert _named(upper + _pedal("A", 1, upper.size / SR)) == ("C", "maj")


def test_a_bass_that_is_spread_over_both_tonics_settles_nothing():
    """Half the bass on A, half on C: neither holds twice the other."""
    upper = _chords(AMBIGUOUS)
    half = upper.size // 2
    bass = np.concatenate([_pedal("A", 1, half / SR), _pedal("C", 2, (upper.size - half) / SR)])
    assert _named(upper + bass[:upper.size]) == ("C", "maj")


def test_rumble_under_the_bass_register_is_not_a_bass_line():
    """31 Hz reads as B0 at this resolution; the register starts at 36 Hz, because on recorded
    loops what sits under it is rumble."""
    upper = _chords(AMBIGUOUS)
    rumble = _tone(31.0, upper.size / SR, level=0.5, partials=(1.0,))
    assert analysis._bass_pitch_classes(upper + rumble, SR) is None
    assert _named(upper + rumble) == ("C", "maj")


def test_the_bass_register_reads_the_note_it_holds():
    upper = _chords(AMBIGUOUS)
    shares = analysis._bass_pitch_classes(upper + _pedal("A", 1, upper.size / SR), SR)
    assert shares is not None and shares[PITCH_CLASSES.index("A")] > 0.9
    assert abs(float(shares.sum()) - 1.0) < 1e-6


def test_the_relative_key_rule_is_a_new_analysis_version():
    assert ANALYSIS_VERSION >= 10


# ------------------------------------------------- the undo's report of routes (D37's chain) ----

@pytest.fixture()
def grouped(demo_project) -> Session:
    s = Session(demo_project)
    assert s.group_stems(["stem_a", "stem_b"])["bus_id"] == "bus1"
    s.add_stem_device("bus1", "comp")
    return Session(demo_project)


def test_the_undos_report_of_routes_does_not_remove_the_group_it_empties(grouped):
    """What D37 did through the app: after a Cmd-Z the engine read every member as playing
    somewhere else, and the app said so. The group must still be there."""
    project = grouped.project_dir
    uid = grouped.buses["bus1"]["uid"]
    res = daw_bridge.apply_coedits(project, ["/api/route_stem?stem=stem_a&bus=",
                                             "/api/route_stem?stem=stem_b&bus="])
    assert res["failed"] == 0, res
    after = Session(project)
    assert "bus1" in after.buses, "the group was buried by a report of routes"
    assert after.buses["bus1"]["uid"] == uid
    assert [e["type"] for e in after.buses["bus1"]["chain"]] == ["comp"]      # ...with its devices
    assert after.stems["stem_a"]["bus"] is None and after.stems["stem_b"]["bus"] is None


def test_the_next_report_puts_the_members_back_in_it(grouped):
    """The way back the user has: the redo. The engine reads right again, the app says so, and
    the group is as it was - nothing was buried, so nothing has to be dug up."""
    project = grouped.project_dir
    daw_bridge.apply_coedits(project, ["/api/route_stem?stem=stem_a&bus=",
                                       "/api/route_stem?stem=stem_b&bus="])
    res = daw_bridge.apply_coedits(project, ["/api/route_stem?stem=stem_a&bus=bus%3Abus1",
                                             "/api/route_stem?stem=stem_b&bus=bus%3Abus1"])
    assert res["failed"] == 0, res
    back = Session(project)
    assert back.stems["stem_a"]["bus"] == "bus1" and back.stems["stem_b"]["bus"] == "bus1"
    assert not back._state.get("routing_tombstones")


def test_an_emptied_group_still_renders(grouped):
    project = grouped.project_dir
    daw_bridge.apply_coedits(project, ["/api/route_stem?stem=stem_a&bus=",
                                       "/api/route_stem?stem=stem_b&bus="])
    chains = daw_bridge.chains_with_readout(project, light=True)
    assert "chains" in chains or chains                        # it resolves
    assert "bus1" in Session(project).buses


def test_the_groups_track_going_is_what_removes_it(grouped):
    """The removal has its own sentence, and it still works on an emptied group."""
    project = grouped.project_dir
    uid = grouped.buses["bus1"]["uid"]
    daw_bridge.apply_coedits(project, ["/api/route_stem?stem=stem_a&bus=",
                                       "/api/route_stem?stem=stem_b&bus="])
    daw_bridge.apply_coedits(project, [f"/api/remove_routing?kind=bus&id=bus1&uid={uid}"])
    assert "bus1" not in Session(project).buses


def test_an_emptied_group_is_tidied_by_the_next_route_somebody_makes(grouped):
    project = grouped.project_dir
    daw_bridge.apply_coedits(project, ["/api/route_stem?stem=stem_a&bus=",
                                       "/api/route_stem?stem=stem_b&bus="])
    s = Session(project)
    s.route_stem("stem_a", "new")                              # a person's route: the tidy runs
    after = Session(project)
    assert "bus1" not in after.buses and after.stems["stem_a"]["bus"] in after.buses


def test_a_route_a_person_makes_still_drops_the_group_it_empties(grouped):
    """The control: only the undo's report keeps an emptied group."""
    s = Session(grouped.project_dir)
    s.route_stem("stem_a", None)
    s.route_stem("stem_b", None)
    assert "bus1" not in Session(grouped.project_dir).buses
