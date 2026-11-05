"""What the engineer's processing did against the audio it started from (2026-11-02).

The site's before-and-after samples, played at matched loudness, were heard as better BEFORE: a
sample-pack vocal that arrived compressed and bright was given three compressors in series,
saturation, +2.5 dB at 3.5 kHz and +4 dB of air, and every reading the engineer had was of the
track as it then stood. `against_raw` reads a track against its own file and the master against
the raw tracks at their levels; three gentle compressors are levelling added together.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from mix_engine import against_raw, analysis
from mix_engine.session import Session

SR = 48000


def _vocal(seconds: float = 8.0) -> np.ndarray:
    """A sung line's shape: a harmonic tone in syllables, its phrases at different levels, with
    a little breath noise on each onset."""
    rng = np.random.default_rng(7)
    t = np.arange(int(SR * seconds)) / SR
    tone = sum((0.86 ** k) * np.sin(2 * np.pi * 220.0 * (k + 1) * t) for k in range(30))
    syllable = 0.5 - 0.5 * np.cos(2 * np.pi * 2.5 * t)            # 2.5 syllables a second
    phrase = np.repeat(10 ** (np.array([-4, -16, -8, -20]) / 20.0), len(t) // 4 + 1)[:len(t)]
    breath = rng.normal(0.0, 0.02, len(t)) * (syllable < 0.2)
    mono = 0.25 * phrase * (tone * syllable + breath)
    return np.c_[mono, mono]


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "vocal_only"
    project.mkdir()
    sf.write(project / "vox.wav", _vocal(), SR, subtype="FLOAT")
    return project


def _comp(threshold_db: float = -30.0, ratio: float = 3.0, makeup_db: float = 0.0) -> dict:
    return {"type": "comp", "threshold_db": threshold_db, "ratio": ratio, "attack_ms": 5.0,
            "release_ms": 80.0, "makeup_db": makeup_db, "mix": 1.0, "channel": "stereo",
            "knee_db": 0.0, "owner": "agent"}


def _shelf(freq_hz: float, gain_db: float, shape: str = "high_shelf") -> dict:
    return {"type": "eq", "band": {"freq_hz": freq_hz, "gain_db": gain_db, "q": 0.7,
                                   "shape": shape, "slope_db_oct": 12},
            "channel": "stereo", "output_db": 0.0, "owner": "agent"}


def _chain(session: Session, chain: list[dict]) -> None:
    session.stems["vox"]["chain"] = [{**entry, "uid": i + 1} for i, entry in enumerate(chain)]


def test_three_gentle_compressors_in_series_are_heavy_levelling_together():
    gentle = {"processor": "comp", "gr_mean_db": 2.5, "gr_max_db": 6.0, "gr_active_pct": 60.0}
    # each alone is under the one-compressor line; the chain takes 7.5 dB
    assert not analysis.over_compressed(None, {"vox": [gentle]})
    assert analysis.over_compressed(None, {"vox": [gentle, gentle, gentle]})
    # a de-esser, a duck and a limiter are not levelling: they do not add up with it
    others = [{**gentle, "processor": p} for p in ("deess", "duck", "limiter")]
    assert not analysis.over_compressed(None, {"vox": [gentle] + others})
    # two tracks' compressors are two chains, not one
    assert not analysis.over_compressed(None, {"vox": [gentle, gentle], "harm": [gentle]})


def test_the_stacked_reduction_counts_only_levelling_processors():
    tel = [{"processor": "comp", "gr_mean_db": 3.0}, {"processor": "mb_comp", "gr_mean_db": 2.0},
           {"processor": "deess", "gr_mean_db": 4.0}, {"processor": "limiter", "gr_mean_db": 5.0}]
    assert against_raw.stacked_gr_db(tel) == (5.0, 2)


def test_a_vocal_made_bright_and_flat_by_its_chain_is_said_against_its_raw_file(tmp_path):
    session = Session(_project(tmp_path))
    _chain(session, [_comp(-30, 3, 4), _comp(-30, 3, 4), _comp(-28, 4, 4),
                     _shelf(3500, 6.0, "peak"), _shelf(8000, 6.0)])
    reading = against_raw.stem_against_raw(session, "vox")
    assert reading is not None
    said = " ".join(reading["lines"])
    assert "brighter by" in said and "over 2 kHz" in said, said
    assert "level moves" in said, said
    assert "3 compressors take" in said, said
    text = against_raw.note(reading)
    assert text.startswith("At matched loudness, vox, against its raw file: ")
    assert "undo or ease" in text


def test_a_light_touch_says_nothing(tmp_path):
    session = Session(_project(tmp_path))
    _chain(session, [_shelf(1000.0, 1.0, "peak"), _comp(-12, 1.5)])
    reading = against_raw.stem_against_raw(session, "vox")
    assert reading is not None and reading["lines"] == [], reading
    assert against_raw.note(reading) is None


def test_a_track_with_no_chain_has_nothing_to_compare(tmp_path):
    session = Session(_project(tmp_path))
    assert against_raw.stem_against_raw(session, "vox") is None
    assert against_raw.stem_against_raw(session, "nobody") is None


def test_the_raw_copy_leaves_the_session_alone(tmp_path):
    session = Session(_project(tmp_path))
    _chain(session, [_comp()])
    session.stems["vox"]["gain_db"] = -6.0
    raw = against_raw.raw_copy(session)
    assert raw.stems["vox"]["chain"] == [] and raw.stems["vox"]["gain_db"] == -6.0   # a level is kept
    assert len(session.stems["vox"]["chain"]) == 1 and session.stems["vox"]["gain_db"] == -6.0


def test_the_master_darkened_against_the_raw_tracks_is_said(tmp_path):
    session = Session(_project(tmp_path))
    session._state["master_chain"] = [{**_shelf(2000.0, -6.0), "uid": 9}]
    reading = against_raw.master_against_raw(session)
    assert reading is not None
    assert any(line.startswith("darker by") for line in reading["lines"]), reading["lines"]
    assert against_raw.note(reading).startswith(
        "At matched loudness, the master, against the raw tracks at their levels: ")


def test_a_move_that_overworks_a_track_comes_back_with_the_note(tmp_path):
    from orchestrator.turn_host import against_raw_note, master_against_raw_note
    project = _project(tmp_path)
    session = Session(project)
    _chain(session, [_comp(-30, 3, 4), _comp(-30, 3, 4), _comp(-28, 4, 4),
                     _shelf(3500, 6.0, "peak"), _shelf(8000, 6.0)])
    session._persist()
    req = {"actions": [{"type": "eq", "stem_id": "vox"}, {"type": "gain", "stem_id": "vox"}]}
    said = against_raw_note(project, req)
    assert said and said.startswith("At matched loudness, vox, against its raw file: "), said
    # a level, pan or send move does not shape the track: it asks for no reading
    assert against_raw_note(project, {"actions": [{"type": "gain", "stem_id": "vox"},
                                                   {"type": "pan", "stem_id": "vox"}]}) is None
    # the whole mix is the one track here, so the master reads the same change
    assert "the master, against the raw tracks at their levels" in (master_against_raw_note(project) or "")


def test_a_band_with_next_to_no_energy_is_no_tone_change():
    raw = {"top_db": -40.0, "low_db": -45.0, "spread_db": 10.0, "crest_db": 12.0}
    now = {"top_db": -30.0, "low_db": -55.0, "spread_db": 10.0, "crest_db": 12.0}
    assert against_raw._tone_lines(raw, now, against_raw.STEM_SHARE_DB) == []
    now["top_db"] = -14.0                       # 4 % of the energy: a top the track now has
    assert against_raw._tone_lines(raw, now, against_raw.STEM_SHARE_DB)[0].startswith("brighter")


def test_a_runs_planner_reads_the_notes_each_pass(tmp_path):
    from orchestrator.agents import ClaudeAgent, PassContext
    from orchestrator.loop import _against_raw_notes
    project = _project(tmp_path)
    assert _against_raw_notes(project) == []          # nothing processed: nothing to say
    session = Session(project)
    _chain(session, [_comp(-30, 3, 4), _comp(-30, 3, 4), _comp(-28, 4, 4),
                     _shelf(3500, 6.0, "peak"), _shelf(8000, 6.0)])
    session._persist()
    notes = _against_raw_notes(project)
    assert [n.split(":")[0] for n in notes] == [
        "At matched loudness, vox, against its raw file"], notes   # no master: the pass budget
    block = ClaudeAgent._context_block(PassContext(pass_number=2, metrics={}, targets=[], stem_chains={},
                                                   against_raw=notes))
    assert "against its raw file" in block
    assert _against_raw_notes(tmp_path / "nowhere") == []   # a reading that fails says nothing


def test_a_quiet_part_brought_up_is_balance_not_flattening(tmp_path):
    """The hip hop re-mix's master read "its level moves 16.3 dB where the raw file moved 25.6":
    most of it was a piano at -56 dBFS, alone in the intro and outro, lifted 16 dB by a trim. The
    raw reference keeps the faders and the trims, so the reading is of the processing alone."""
    project = tmp_path / "intro"
    project.mkdir()
    n = SR * 8
    t = np.arange(n) / SR
    loud = 0.3 * np.sin(2 * np.pi * 110 * t) * (t >= 4.0)            # the drop, second half
    quiet = 0.003 * np.sin(2 * np.pi * 440 * t)                     # a piano at -50 dBFS, all song
    sf.write(project / "drums.wav", np.c_[loud, loud], SR, subtype="FLOAT")
    sf.write(project / "piano.wav", np.c_[quiet, quiet], SR, subtype="FLOAT")
    session = Session(project)
    session.stems["piano"]["chain"] = [{"type": "utility", "gain_db": 24.0, "width": 1.0,
                                        "owner": "user", "uid": 1}]
    session.stems["piano"]["gain_db"] = 4.0
    reading = against_raw.master_against_raw(session)
    assert reading is not None and reading["lines"] == [], reading
    # ...and squashing the drop still reads as flattening against that reference
    session._state["master_chain"] = [{**_comp(-40, 20, 0), "uid": 2}]
    said = " ".join(against_raw.master_against_raw(session)["lines"])
    assert "level moves" in said, said
