"""Agent-mastery track (docs/15-agent-mastery-handoff.md): the measurements and
tool wiring the resident engineer needs to answer a user's questions from THIS
session's numbers instead of from generalities.

Metric definitions under test are contractual: .claude/skills/mix-analysis/SKILL.md.
"""
from pathlib import Path

import numpy as np
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from mix_engine import analysis
from mix_engine.mcp_server import create_server

from test_phase1_acceptance import call

SR = 44100


def _stereo(seconds: float, width: float, *, freq: float = 220.0,
            seed: int = 0) -> np.ndarray:
    """A tone with `width` of decorrelated side content (0 = mono)."""
    t = np.arange(int(SR * seconds)) / SR
    tone = np.sin(2 * np.pi * freq * t) * 0.3
    side = np.random.default_rng(seed).normal(0, width, size=t.shape)
    return np.stack([tone + side, tone - side], axis=1)


# ---------- per-section stereo picture (item 2a) ----------

def test_width_db_is_none_for_mono_and_floors_for_dual_mono():
    mono = _stereo(1.0, width=0.0)[:, :1]
    assert analysis._width_db(mono) is None
    assert analysis._width_db(_stereo(1.0, width=0.0)) == analysis.BAND_WIDTH_FLOOR_DB


def test_width_db_rises_with_side_content():
    narrow = analysis._width_db(_stereo(1.0, width=0.02))
    wide = analysis._width_db(_stereo(1.0, width=0.25))
    assert narrow is not None and wide is not None
    assert wide > narrow


def test_sections_carry_width_and_correlation_and_discriminate():
    """"Does the chorus open up against the verse?" — the question that had no
    number before. A mono verse followed by a wide chorus must read as such."""
    master = np.concatenate([_stereo(8.0, width=0.0),
                             _stereo(8.0, width=0.25)], axis=0)
    result = analysis.analyze_sections(master, SR, {"stem_a": master})
    assert len(result.sections) == 2
    verse, chorus = result.sections
    assert chorus.width_db > verse.width_db
    # A dual-mono verse is perfectly correlated; hard-decorrelated sides are not.
    assert verse.stereo_correlation == pytest.approx(1.0, abs=0.01)
    assert chorus.stereo_correlation < verse.stereo_correlation


def test_silent_section_reports_no_stereo_picture():
    """Silence has no width to report — None, never a fabricated 0."""
    quiet = np.zeros((SR * 8, 2), dtype=float)
    master = np.concatenate([_stereo(8.0, width=0.25), quiet], axis=0)
    result = analysis.analyze_sections(master, SR, {"stem_a": master})
    silent = [s for s in result.sections if s.lufs_i is None]
    assert silent, "expected the silent tail to be its own section"
    assert all(s.width_db is None and s.stereo_correlation is None
               for s in silent)


@pytest.mark.anyio
async def test_sections_expose_stereo_picture_through_mcp(demo_project):
    """The fields survive the MCP boundary — the agent reads them, not just us."""
    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        result = await call(client, "analyze_sections")
        for section in result["sections"]:
            assert "width_db" in section
            assert "stereo_correlation" in section


# ---------- per-phrase intonation (item 2b) ----------

def _sung(freq: float, seconds: float = 2.0, cents: float = 0.0) -> np.ndarray:
    """A held note with a few harmonics and soft edges — enough spectral peaks
    for the tuning estimator, without transient smear at the boundaries."""
    t = np.arange(int(SR * seconds)) / SR
    f = freq * 2.0 ** (cents / 1200.0)
    sig = sum(np.sin(2 * np.pi * f * h * t) / h for h in (1, 2, 3)) * 0.25
    return sig * np.minimum(1.0, np.minimum(t * 20, (seconds - t) * 20))


def _rest(seconds: float) -> np.ndarray:
    return np.zeros(int(SR * seconds))


def test_phrase_spans_split_on_rests_but_not_breaths():
    take = np.concatenate([_sung(220, 2.0), _rest(0.6), _sung(220, 2.0)])
    assert len(analysis._phrase_spans(take, SR)) == 2
    breathed = np.concatenate([_sung(220, 1.5), _rest(0.12), _sung(220, 1.5)])
    assert len(analysis._phrase_spans(breathed, SR)) == 1


def test_phrase_pitch_finds_the_one_flat_line():
    """"Which line is most out of tune?" — the third phrase is 35 cents flat
    against three solid ones, and must rank first by deviation."""
    take = np.concatenate([
        _sung(220), _rest(0.6), _sung(261.63), _rest(0.6),
        _sung(196, cents=-35), _rest(0.6), _sung(246.94)])
    audio = np.stack([take, take], axis=1)
    result = analysis.analyze_phrases(audio, SR, "vox")

    assert len(result.phrase_pitch) == 4
    worst = result.phrase_pitch[0]
    assert worst.deviation_cents < -25          # the flat line, not one of the others
    assert worst.start_s == pytest.approx(4.9, abs=0.5)
    assert all(abs(p.deviation_cents) < 5 for p in result.phrase_pitch[1:])


def test_phrase_pitch_deviation_is_relative_to_the_takes_own_centre():
    """A take that is uniformly flat is ONE stem-level move, not N bad
    phrases — deviation must stay ~0 while the stem centre carries the offset."""
    flat = np.concatenate([_sung(220, cents=-30), _rest(0.6),
                           _sung(261.63, cents=-30), _rest(0.6),
                           _sung(196, cents=-30)])
    result = analysis.analyze_phrases(np.stack([flat, flat], axis=1), SR, "vox")

    assert result.stem_tuning_offset_cents == pytest.approx(-30, abs=6)
    assert all(abs(p.deviation_cents) < 5 for p in result.phrase_pitch)


def test_phrase_pitch_is_empty_on_unpitched_material():
    """Absence means "not measurable here", never "in tune" — drums and noise
    must report nothing rather than a fabricated offset."""
    noise = np.random.default_rng(0).normal(0, 0.2, SR * 4)
    result = analysis.analyze_phrases(np.stack([noise, noise], axis=1), SR, "drums")
    assert result.phrase_pitch == []


@pytest.mark.anyio
async def test_phrase_pitch_survives_the_mcp_boundary(demo_project):
    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        result = await call(client, "analyze_phrases", {"stem_id": "stem_a"})
        assert "phrase_pitch" in result
        assert "stem_tuning_offset_cents" in result


# ---------- onset timing against the grid (item 2c) ----------

BPM = 120.0
BEAT_S = 60.0 / BPM


def _click(seconds: float = 0.06) -> np.ndarray:
    """A percussive hit that decays AND fades to zero. The fade matters: a
    truncated decay is a second transient, and the detector rightly finds it."""
    t = np.arange(int(SR * seconds)) / SR
    env = np.exp(-t * 120) * np.minimum(1.0, (seconds - t) * 60)
    return np.sin(2 * np.pi * 1200 * t) * env


def _hits(times_s, total_s: float = 18.0) -> np.ndarray:
    buf = np.zeros(int(SR * total_s))
    click = _click()
    for t in times_s:
        i = int(t * SR)
        if 0 <= i and i + len(click) < len(buf):
            buf[i:i + len(click)] += click
    return np.stack([buf, buf], axis=1)


def _eighths(n: int = 32, *, offset_ms: float = 0.0, jitter_ms: float = 0.0,
             swing: float = 0.5, seed: int = 0, start_s: float = 1.0):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        pos = start_s + (k // 2) * BEAT_S + (0.0 if k % 2 == 0 else swing * BEAT_S)
        out.append(pos + offset_ms / 1000.0 + rng.normal(0, jitter_ms / 1000.0))
    return out


def _timing(times, **kw):
    return analysis.analyze_timing(_hits(times), SR, "drums",
                                   bpm=BPM, beat_offset_s=1.0, **kw)


def test_onset_detection_survives_sparse_material():
    """The percentile threshold DEGENERATES when most frames carry no flux:
    median and p90 both collapse to 0, so every numerical ripple scored as an
    onset and six hits detected as twelve. A sparse percussion take is not an
    edge case."""
    six = _hits([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], total_s=5.0)
    assert len(analysis._onset_times_precise(six[:, 0], SR)) == 6


def test_onset_times_are_accurate_to_a_millisecond():
    """The metric measures 5-30 ms leans, so the detector's own error has to be
    far below that — the shared 512-sample hop (11.6 ms) is not good enough,
    which is why this path uses a finer hop plus parabolic refinement."""
    expected = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    got = analysis._onset_times_precise(_hits(expected, total_s=5.0)[:, 0], SR)
    errors_ms = [(float(g) - e) * 1000.0 for g, e in zip(got, expected)]
    assert max(abs(e) for e in errors_ms) < 3.0


def test_timing_reads_zero_on_a_machine_perfect_take():
    result = _timing(_eighths())
    assert result.subdivision == 2
    assert abs(result.bias_ms) < 3.0
    assert result.spread_ms < 3.0
    assert result.swing_percent == pytest.approx(50.0, abs=2.0)


def test_timing_signs_rushing_negative_and_dragging_positive():
    """The convention the whole readout rests on: negative = early."""
    assert _timing(_eighths(offset_ms=-20)).bias_ms < -15
    assert _timing(_eighths(offset_ms=20)).bias_ms > 15


def test_timing_separates_a_lean_from_scatter():
    """The distinction that makes the metric worth having: a consistent lean is
    a pocket, scatter is sloppiness, and they must not read the same."""
    leaning = _timing(_eighths(offset_ms=20))
    scattered = _timing(_eighths(jitter_ms=25, seed=3))

    assert leaning.bias_ms > 15 and leaning.spread_ms < 5      # deliberate
    assert abs(scattered.bias_ms) < 8 and scattered.spread_ms > 12   # loose


def test_timing_does_not_call_swing_dragging():
    """On a swung part the off-beats are SUPPOSED to be late. Reporting that as
    drag would tell a user to fix the thing that makes their groove work."""
    light = _timing(_eighths(swing=0.56))
    assert light.swing_percent == pytest.approx(56.0, abs=2.0)
    assert abs(light.bias_ms) < 5                    # not reported as dragging

    # Heavy swing IS a triplet grid — describing it as sloppy 16ths is wrong.
    triplet = _timing(_eighths(swing=2.0 / 3.0))
    assert triplet.subdivision == 3
    assert abs(triplet.bias_ms) < 5


def test_timing_reports_why_when_it_cannot_measure():
    """Absence of a number must say 'not measurable', never imply 'in time'."""
    no_grid = analysis.analyze_timing(_hits(_eighths()), SR, "drums", bpm=None)
    assert no_grid.bias_ms is None and no_grid.reason

    too_few = _timing([1.0, 2.0, 3.0])
    assert too_few.bias_ms is None and "too few" in too_few.reason

    silent = analysis.analyze_timing(np.zeros((SR * 4, 2)), SR, "pad", bpm=BPM)
    assert silent.bias_ms is None and silent.reason == "silent"


@pytest.mark.anyio
async def test_analyze_timing_round_trips_through_mcp(demo_project):
    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        result = await call(client, "analyze_timing", {"stem_id": "stem_a"})
        assert result["stem_id"] == "stem_a"
        assert "bias_ms" in result and "spread_ms" in result


def test_analyze_timing_is_reachable_from_chat():
    from orchestrator.chat import READ_TOOLS, _READ_TOOL_DEFS, daw_tool_defs

    assert "analyze_timing" in READ_TOOLS
    assert any(d["name"] == "analyze_timing" for d in _READ_TOOL_DEFS)
    assert any(d["name"] == "analyze_timing"
               for d in daw_tool_defs(during_run=False))


# ---------- per-note pitch detection (2026-08-08) ----------

def _sung_note(midi: int, seconds: float, *, cents: float = 0.0,
               vibrato_cents: float = 0.0) -> np.ndarray:
    """A held note with harmonics, optionally detuned and/or with vibrato."""
    t = np.arange(int(SR * seconds)) / SR
    freq = 440.0 * 2.0 ** ((midi - 69 + cents / 100.0) / 12.0) * np.ones_like(t)
    if vibrato_cents:
        freq = freq * 2.0 ** (vibrato_cents * np.sin(2 * np.pi * 5.5 * t) / 1200.0)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    partials = sum(np.sin(phase * h) / (h ** 1.3) for h in (1, 2, 3, 4, 5))
    return partials * np.minimum(1.0, np.minimum(t * 30, (seconds - t) * 30)) * 0.25


def _melody(parts) -> np.ndarray:
    mono = np.concatenate(parts)
    return np.stack([mono, mono], axis=1)


def test_notes_name_the_one_flat_note():
    """C4-E4-G4-E4 with the G4 45 cents flat: the note must be named, measured,
    and ranked first."""
    silence = np.zeros(int(SR * 0.15))
    result = analysis.analyze_notes(_melody([
        _sung_note(60, 0.6), silence, _sung_note(64, 0.6), silence,
        _sung_note(67, 0.6, cents=-45), silence, _sung_note(64, 0.6)]),
        SR, "vox")

    assert result.notes_found == 4
    assert [n.note_name for n in sorted(result.notes, key=lambda n: n.start_s)] \
        == ["C4", "E4", "G4", "E4"]
    worst = result.notes[0]                      # ranked worst-deviation-first
    assert worst.note_name == "G4"
    assert worst.deviation_cents == pytest.approx(-45, abs=8)


def test_in_tune_notes_measure_in_tune():
    result = analysis.analyze_notes(
        _melody([_sung_note(60, 0.8), np.zeros(int(SR * 0.2)),
                 _sung_note(64, 0.8)]), SR, "vox")
    assert all(abs(n.deviation_cents) < 10 for n in result.notes)


def test_vibrato_is_movement_not_a_wrong_note():
    """A median holds the intended pitch through symmetric vibrato, and the
    movement shows up as spread — otherwise every expressive singer would be
    reported as out of tune."""
    result = analysis.analyze_notes(
        _melody([_sung_note(62, 1.2, vibrato_cents=50)]), SR, "vox")

    (note,) = result.notes
    assert abs(note.cents_off) < 15          # the pitch itself is right
    assert note.spread_cents > 50            # …and the vibrato is visible


def test_low_notes_do_not_octave_error():
    """Autocorrelation's signature failure: the lag at twice the true period
    correlates just as well, so 'highest peak' reports a note an octave down.
    An octave error is a wrong note, not a small one."""
    result = analysis.analyze_notes(_melody([_sung_note(41, 0.8)]), SR, "bass")
    assert [n.midi_note for n in result.notes] == [41]


def test_notes_say_why_on_unsuitable_material():
    """Polyphonic and percussive sources must decline, not report the root of a
    chord as 'the note'."""
    noise = np.random.default_rng(0).normal(0, 0.3, SR * 3)
    percussive = analysis.analyze_notes(np.stack([noise, noise], axis=1),
                                        SR, "drums")
    assert percussive.notes == [] and percussive.reason

    silent = analysis.analyze_notes(np.zeros((SR * 2, 2)), SR, "pad")
    assert silent.notes == [] and silent.reason == "silent"


@pytest.mark.parametrize("name,midis", [
    ("major triad",  (60, 64, 67)),
    ("power fifth",  (52, 59)),
    ("five-note pad", (48, 55, 60, 64, 67)),
    ("bare third",   (60, 64)),
])
def test_polyphony_declines_instead_of_naming_a_virtual_fundamental(name, midis):
    """The case the test above only CLAIMED to cover (2026-08-08 review).

    Autocorrelation answers a chord with the common sub-period — a note nobody
    played, one or two octaves below the real ones. A C4-E4-G4 triad was reported
    as a confident "C2, 3.8 cents sharp" with reason=None, which is worse than
    declining because the agent cannot tell it is wrong.
    """
    chord = np.zeros(int(SR * 1.5))
    for midi in midis:
        voice = _sung_note(midi, 1.5)
        chord[:len(voice)] += voice
    result = analysis.analyze_notes(np.stack([chord, chord], axis=1), SR, "keys")
    lowest = min(midis)
    assert result.notes == [], (
        f"{name}: named {[n.note_name for n in result.notes]} for a chord whose "
        f"lowest real note is MIDI {lowest}")
    assert result.reason and "POLYPHONIC" in result.reason


def test_a_monophonic_take_with_a_weak_fundamental_still_measures():
    """The gate must fire on an ABSENT fundamental, not a quiet one — a real
    vowel can put more energy in its 2nd partial than its 1st."""
    t = np.arange(int(SR * 1.5)) / SR
    freq = 440.0 * 2.0 ** ((57 - 69) / 12.0)
    voice = 0.2 * np.sin(2 * np.pi * freq * t)          # fundamental at 20 %
    for h in (2, 3, 4, 5, 6):
        voice += (1.0 / h ** 1.2) * np.sin(2 * np.pi * freq * h * t)
    voice *= np.minimum(1.0, np.minimum(t * 30, (1.5 - t) * 30)) * 0.2
    result = analysis.analyze_notes(np.stack([voice, voice], axis=1), SR, "vox")
    assert result.notes_found >= 1 and result.reason is None


@pytest.mark.anyio
async def test_analyze_notes_round_trips_through_mcp(demo_project):
    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        result = await call(client, "analyze_notes", {"stem_id": "stem_a"})
        assert result["stem_id"] == "stem_a"
        assert "notes" in result


def test_analyze_notes_is_reachable_from_chat():
    from orchestrator.chat import READ_TOOLS, _READ_TOOL_DEFS, daw_tool_defs

    assert "analyze_notes" in READ_TOOLS
    assert any(d["name"] == "analyze_notes" for d in _READ_TOOL_DEFS)
    assert any(d["name"] == "analyze_notes"
               for d in daw_tool_defs(during_run=False))


# ---------- chord detection (2026-08-08) ----------

def _tone(midi: int, seconds: float) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    freq = 440.0 * 2.0 ** ((midi - 69) / 12.0)
    partials = sum(np.sin(2 * np.pi * freq * h * t) / (h ** 1.4) for h in (1, 2, 3, 4))
    return partials * np.minimum(1.0, np.minimum(t * 30, (seconds - t) * 30)) * 0.2


def _chord(root: int, quality: str, beats: int = 2, octave: int = 4,
           extra=()) -> np.ndarray:
    intervals = analysis.CHORD_TEMPLATES[quality]
    seconds = beats * BEAT_S
    base = 12 * (octave + 1) + root
    out = np.zeros(int(SR * seconds))
    for step in tuple(intervals) + tuple(extra):
        voice = _tone(base + step, seconds)
        out[:len(voice)] += voice
    return out


def _progression(chords, key=None, noise: float = 0.0):
    audio = np.concatenate([_chord(*c) for c in chords])
    if noise:
        audio = audio + np.random.default_rng(0).normal(0, noise, len(audio))
    stereo = np.stack([audio, audio], axis=1)
    return analysis.analyze_chords(stereo, SR, "keys", bpm=BPM,
                                   beat_offset_s=0.0, key=key)


def _labels(result):
    return [f"{s.root}{s.quality}" for s in result.spans]


def _key(tonic: str, mode: str):
    from mix_engine.models import KeyEstimate
    return KeyEstimate(tonic=tonic, mode=mode, confidence=0.9)


def test_chords_read_a_pop_progression():
    """I-V-vi-IV in C: the most common progression in recorded music."""
    result = _progression([(0, "maj"), (7, "maj"), (9, "min"), (5, "maj")],
                          key=_key("C", "maj"))
    assert _labels(result) == ["Cmaj", "Gmaj", "Amin", "Fmaj"]
    assert all(s.confidence > 0.8 for s in result.spans)


def test_chords_do_not_depend_on_the_key_prior():
    """The prior breaks ties; it must not be doing the work. Same audio, no
    key, same answer."""
    chords = [(0, "maj"), (7, "maj"), (9, "min"), (5, "maj")]
    assert _labels(_progression(chords)) == _labels(
        _progression(chords, key=_key("C", "maj")))


def test_chords_read_a_minor_progression():
    result = _progression([(9, "min"), (5, "maj"), (0, "maj"), (7, "maj")],
                          key=_key("A", "min"))
    assert _labels(result) == ["Amin", "Fmaj", "Cmaj", "Gmaj"]


def test_a_major_seventh_keeps_its_own_root():
    """The regression this file exists for. With triad templates only, Cmaj7
    (C E G B) shares three notes with Em (E G B), the vectors are near
    identical, and it resolved to Emin — the WRONG ROOT, the one error a
    musician cannot work around. Seventh templates are what fixed it."""
    result = _progression([(0, "maj7"), (5, "maj7")], key=_key("C", "maj"))
    assert [s.root for s in result.spans] == ["C", "F"]


def test_chords_read_a_two_five_one():
    result = _progression([(2, "min7"), (7, "dom7"), (0, "maj7")],
                          key=_key("C", "maj"))
    assert _labels(result) == ["Dmin7", "Gdom7", "Cmaj7"]


def test_plain_triads_are_not_over_labelled_as_sevenths():
    result = _progression([(0, "maj"), (5, "maj")], key=_key("C", "maj"))
    assert _labels(result) == ["Cmaj", "Fmaj"]


def test_chords_survive_broadband_noise():
    result = _progression([(0, "maj"), (7, "maj"), (9, "min"), (5, "maj")],
                          key=_key("C", "maj"), noise=0.05)
    assert _labels(result) == ["Cmaj", "Gmaj", "Amin", "Fmaj"]


def _rich_progression(chords, octave, harmonics, roll):
    """The same progression at a chosen REGISTER and TIMBRE.

    `_tone` above uses octave 4 with harmonics 1-4, which is the one timbre whose
    partials barely reach other pitch classes — and the timbre under which the
    2026-08-08 review found binary templates reporting the wrong root for
    everything lower and brighter.
    """
    def tone(midi, seconds):
        t = np.arange(int(SR * seconds)) / SR
        freq = 440.0 * 2.0 ** ((midi - 69) / 12.0)
        partials = sum(np.sin(2 * np.pi * freq * h * t) / (h ** roll)
                       for h in harmonics)
        return partials * np.minimum(1.0, np.minimum(t * 30, (seconds - t) * 30)) * 0.2

    audio = []
    for root, quality in chords:
        seconds = 2 * BEAT_S
        base = 12 * (octave + 1) + root
        block = np.zeros(int(SR * seconds))
        for step in analysis.CHORD_TEMPLATES[quality]:
            voice = tone(base + step, seconds)
            block[:len(voice)] += voice
        audio.append(block)
    mono = np.concatenate(audio)
    stereo = np.stack([mono, mono], axis=1)
    return analysis.analyze_chords(stereo, SR, "keys", bpm=BPM, beat_offset_s=0.0)


@pytest.mark.parametrize("octave,harmonics,roll", [
    (4, tuple(range(1, 5)), 1.4),    # the original test timbre
    (4, tuple(range(1, 9)), 1.4),
    (2, tuple(range(1, 9)), 1.4),    # low + rich: returned ['Amin','Amin'] before
    (2, tuple(range(1, 13)), 1.0),   # low + bright: returned the WRONG ROOTS before
])
def test_chords_hold_their_root_across_register_and_timbre(octave, harmonics, roll):
    """A chord's own partials are other pitch classes — a low C's series lands on
    C, G, E and Bb. Against binary templates that made a rich low-register C
    major correlate better with Emin7 than with Cmaj. Harmonic-weighted
    templates put that energy where the right template already expects it."""
    result = _rich_progression([(0, "maj"), (7, "maj"), (9, "min"), (5, "maj")],
                               octave, harmonics, roll)
    assert _labels(result) == ["Cmaj", "Gmaj", "Amin", "Fmaj"]


def test_chord_spans_merge_at_any_tempo():
    """Contiguity is a beat-index question, not a float one. Comparing a rounded
    end_s against an unrounded start_s only ever matched at tempi whose beat
    lands on a 10 ms boundary — identical audio merged into 4 spans at 120 BPM
    and fragmented into dozens at 128."""
    for bpm in (120.0, 128.0, 90.0, 174.0, 96.0):
        beat_s = 60.0 / bpm
        chords = [(0, "maj"), (7, "maj"), (9, "min"), (5, "maj")]
        mono = np.concatenate([
            _chord(root, quality, beats=2) for root, quality in chords])
        # _chord builds at the module BPM; re-grid by telling the analyzer the
        # tempo the blocks actually imply, so each chord still spans 2 beats.
        implied = 60.0 / (len(mono) / SR / (2 * len(chords)))
        stereo = np.stack([mono, mono], axis=1)
        result = analysis.analyze_chords(stereo, SR, "keys", bpm=implied,
                                         beat_offset_s=0.0)
        assert _labels(result) == ["Cmaj", "Gmaj", "Amin", "Fmaj"], (
            f"fragmented at {bpm} BPM (implied {implied:.3f}): {_labels(result)}")
        assert beat_s > 0


def test_chord_spans_are_capped():
    assert analysis.CHORD_MAX_SPANS > 0
    long_prog = [(r % 12, "maj" if r % 2 else "min") for r in range(200)]
    mono = np.concatenate([_chord(root, quality, beats=1)
                           for root, quality in long_prog])
    stereo = np.stack([mono, mono], axis=1)
    result = analysis.analyze_chords(stereo, SR, "keys", bpm=BPM, beat_offset_s=0.0)
    assert len(result.spans) <= analysis.CHORD_MAX_SPANS
    if len(result.spans) == analysis.CHORD_MAX_SPANS:
        assert result.reason and "of" in result.reason


def test_inversions_keep_the_root_even_though_the_bass_is_unheard():
    """Chroma is inversion-blind, so C/E must still report C — naming the
    chord right while being unable to hear the bass note."""
    result = _progression([(0, "maj", 2, 4, (-8,)), (5, "maj", 2, 4, (-5,))],
                          key=_key("C", "maj"))
    assert [s.root for s in result.spans] == ["C", "F"]


def test_chords_say_why_when_undetectable():
    noise = np.random.default_rng(1).normal(0, 0.3, SR * 8)
    percussive = np.stack([noise, noise], axis=1)

    atonal = analysis.analyze_chords(percussive, SR, "drums", bpm=BPM)
    assert atonal.spans == [] and atonal.reason

    no_grid = analysis.analyze_chords(percussive, SR, "drums", bpm=None)
    assert no_grid.spans == [] and "tempo" in no_grid.reason

    silent = analysis.analyze_chords(np.zeros((SR * 4, 2)), SR, "pad", bpm=BPM)
    assert silent.spans == [] and silent.reason == "silent"


@pytest.mark.anyio
async def test_analyze_chords_round_trips_through_mcp(demo_project):
    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        result = await call(client, "analyze_chords", {"stem_id": "stem_a"})
        assert result["stem_id"] == "stem_a"
        assert "spans" in result


def test_analyze_chords_is_reachable_from_chat():
    from orchestrator.chat import READ_TOOLS, _READ_TOOL_DEFS, daw_tool_defs

    assert "analyze_chords" in READ_TOOLS
    assert any(d["name"] == "analyze_chords" for d in _READ_TOOL_DEFS)
    assert any(d["name"] == "analyze_chords"
               for d in daw_tool_defs(during_run=False))


# ---------- the orphaned read tool (item 1) ----------

def test_analyze_separation_is_reachable_from_chat():
    """It existed server-side but was absent from the chat agent's tool set, so
    "is this split clean?" was unanswerable — and the system prompt already told
    the agent to size spectral_denoise from it."""
    from orchestrator.chat import READ_TOOLS, _READ_TOOL_DEFS, tool_defs, daw_tool_defs

    assert "analyze_separation" in READ_TOOLS
    assert any(d["name"] == "analyze_separation" for d in _READ_TOOL_DEFS)
    # Both shells: the gateway workspace and the native DAW.
    for defs in (tool_defs(during_run=False), daw_tool_defs(during_run=False)):
        assert any(d["name"] == "analyze_separation" for d in defs)


# ---------- tiered knowledge (item 3) ----------

def test_every_topic_renders_and_the_enum_matches_the_module():
    """The advertised enum and the module's keys must not drift — an advertised
    topic that does not resolve is a tool call the agent cannot recover from."""
    from orchestrator import knowledge_deep
    from orchestrator.chat import _READ_TOOL_DEFS

    assert knowledge_deep.TOPICS, "no topics registered"
    for topic in knowledge_deep.TOPICS:
        body = knowledge_deep.lookup(topic)
        assert len(body) > 500, f"{topic} is too thin to be worth a round trip"

    # The lookup def is built PER CALL now (the book may arrive post-sign-in).
    from orchestrator.chat import daw_tool_defs
    (tool,) = [d for d in daw_tool_defs(False) if d["name"] == "lookup_knowledge"]
    assert (tuple(tool["input_schema"]["properties"]["topic"]["enum"])
            == knowledge_deep.TOPICS)
    for topic in knowledge_deep.TOPICS:           # the index advertises them all
        assert topic in tool["description"]


def test_unknown_topic_names_the_real_ones():
    from orchestrator import knowledge_deep

    reply = knowledge_deep.lookup("music")
    assert "no such topic" in reply
    assert "harmony" in reply                      # recoverable, not a dead end


def test_deep_knowledge_stays_out_of_the_always_on_block():
    """The whole point of the split: depth must NOT ride in every prompt. The
    index may, the bodies may not."""
    from orchestrator import knowledge_deep
    from orchestrator.knowledge import KNOWLEDGE_BLOCK

    for topic in knowledge_deep.TOPICS:
        body = knowledge_deep.lookup(topic)
        assert body not in KNOWLEDGE_BLOCK
    assert len(knowledge_deep.topic_index()) < 1500   # always-on cost, keep terse


@pytest.mark.anyio
async def test_lookup_knowledge_needs_no_engine():
    """It is local text — it must work with no session, no engine and mid-run,
    which is what makes it free to reach for."""
    from orchestrator.chat import ChatTools, execute_tool

    async def boom(*_args, **_kwargs):
        raise AssertionError("lookup_knowledge must not touch the engine")

    tools = ChatTools(
        engine_call=boom, propose=boom, start_run=boom, stop_run=boom,
        auto_mix=boom, file_instruction=boom, save_note=boom,
        workspace_command=boom, list_files=boom, import_sample=boom,
        undo_proposal=boom, render=boom)

    outcome = await execute_tool(tools, "lookup_knowledge", {"topic": "harmony"})
    assert not outcome.error
    assert "voice leading" in outcome.text.lower()

    bad = await execute_tool(tools, "lookup_knowledge", {"topic": "nope"})
    assert "no such topic" in bad.text


def test_lookup_knowledge_is_available_in_every_mode():
    from orchestrator.chat import daw_tool_defs, tool_defs

    for defs in (tool_defs(during_run=False), tool_defs(during_run=True),
                 daw_tool_defs(during_run=False), daw_tool_defs(during_run=True)):
        assert any(d["name"] == "lookup_knowledge" for d in defs)


# ---------- the Q&A eval harness (item 4) ----------

def test_qa_corpus_is_well_formed():
    """A `needs=("x")` without the trailing comma is a STRING, and the grader
    would then require every CHARACTER of it — a rubric that can never pass.
    Cheap to write by accident, invisible until a billed sweep."""
    from orchestrator import qa_eval

    assert len(qa_eval.CORPUS) >= 15
    for case in qa_eval.CORPUS:
        assert isinstance(case.needs, tuple), f"{case.text!r}: needs is not a tuple"
        assert isinstance(case.never, tuple), f"{case.text!r}: never is not a tuple"
        assert case.why, f"{case.text!r} has no rationale"
        for group in case.needs + case.never:
            assert isinstance(group, str) and group.strip()
            # Substring matcher, not regex — a bracket expression silently
            # never matches and quietly weakens the rubric.
            assert "[" not in group, f"{case.text!r}: {group!r} looks like regex"
        if case.expect_tool and case.expect_tool != "lookup_knowledge":
            assert case.expect_tool in qa_eval._MEASURED


def test_qa_grader_separates_bluffing_from_declining():
    """The harmful class: inventing a chord the app cannot detect must score
    `bluffed`, while declining the same question scores `answered`."""
    from orchestrator import qa_eval

    (case,) = [c for c in qa_eval.CORPUS if "inversion" in c.text]
    bluff = qa_eval.grade(case, "Yes, that is a C/E — the E is in the bass.", ())
    honest = qa_eval.grade(
        case, "I read it as C major, but I can't hear inversions: chord "
              "detection is chroma-based, so C/E and C look identical.", ())
    assert bluff.verdict == "bluffed"
    assert honest.verdict == "answered"


def test_qa_rubric_does_not_match_inside_ordinary_words():
    """Bare substring matching graded content-free replies as ANSWERED
    (2026-08-08 review): `ms` lives inside "dru**ms**", `cent` inside
    "re**cent**ly", `not` inside "**Not**hing". Those are the cases guarding
    analyze_timing / analyze_notes / the inversion-bluff invariant, and every
    false hit inflates the baseline the gate is measured against."""
    from orchestrator import qa_eval

    cases = {c.text: c for c in qa_eval.CORPUS}
    for text, reply in [
        ("is my drummer rushing the beat?", "Your drums sound fine to me."),
        ("which notes in the vocal are flat?",
         "I looked at it recently; nothing stands out."),
        ("is the guitar playing a C major or a C/E inversion there?",
         "Nothing to report."),
    ]:
        outcome = qa_eval.grade(cases[text], reply, ("analyze_session",))
        assert outcome.verdict != "answered", f"{reply!r} graded as answered"

    # …while the real answers still hit, including a stem (`invers` -> inversion)
    # and a unit glued to its number.
    assert qa_eval._hit("the kick sits 12ms late", "ahead|behind|early|late|ms")
    assert qa_eval._hit("about 12 ms behind", "ahead|behind|early|late|ms")
    assert qa_eval._hit("it is 30 cents flat", "cent|flat|sharp")
    assert qa_eval._hit("that could be an inversion", "invers|chroma")
    assert qa_eval._hit("it moves I-V-vi-IV", "minor|major|=i|=iv|=v|=vi")
    assert not qa_eval._hit("a vinyl sheen", "minor|major|=i|=iv|=v|=vi")


def test_qa_check_does_not_spend_before_finding_no_baseline(tmp_path, monkeypatch):
    """--check used to run all 26 billed turns and only THEN discover there was
    no baseline to compare against — and none has ever been committed."""
    from orchestrator import qa_eval

    monkeypatch.setattr(qa_eval, "BASELINE_PATH", tmp_path / "absent.json")

    def _boom(*a, **k):  # pragma: no cover - must never be reached
        raise AssertionError("run() was called before the baseline check")

    monkeypatch.setattr(qa_eval, "run", _boom)
    assert qa_eval.main(["--check"]) == 2


def test_qa_grader_separates_grounded_from_generic():
    """The generic-answer failure: correct-sounding, measured nothing."""
    from orchestrator import qa_eval

    (case,) = [c for c in qa_eval.CORPUS if c.text.startswith("is my mix too quiet")]
    generic = qa_eval.grade(case, "Mixes usually sit around -14 LUFS.", ())
    measured = qa_eval.grade(case, "Your mix reads -16.2 LUFS integrated.",
                             ("analyze_session",))
    thin = qa_eval.grade(case, "Sounds fine to me.", ("analyze_session",))

    assert generic.verdict == "ungrounded"
    assert measured.verdict == "answered"
    assert thin.verdict == "missed"


def test_qa_lookup_counts_as_grounding_for_theory():
    from orchestrator import qa_eval

    (case,) = [c for c in qa_eval.CORPUS if "ii-V-I" in c.text]
    reply = "The V chord builds tension that resolves to the tonic."
    assert qa_eval.grade(case, reply, ("lookup_knowledge",)).verdict == "answered"
    assert qa_eval.grade(case, reply, ()).verdict == "ungrounded"


def test_qa_regression_gate_catches_the_classes_that_matter():
    from orchestrator import qa_eval

    base = {"total": 10, "counts": {"answered": 8, "ungrounded": 1,
                                    "missed": 1, "bluffed": 0}}
    worse = {"total": 10, "counts": {"answered": 7, "ungrounded": 1,
                                     "missed": 1, "bluffed": 1}}
    problems = qa_eval.compare(base, worse)
    assert any("bluffed" in p for p in problems)
    assert any("answered shrank" in p for p in problems)
    # Deleting cases to make the numbers look good is itself a regression.
    assert qa_eval.compare(base, {"total": 4, "counts": base["counts"]})


def test_qa_eval_is_not_wired_into_the_free_gate():
    """It is billed and non-deterministic — `benchmark --check` must stay free
    and deterministic, so this must never be imported into that path."""
    from pathlib import Path

    benchmark = Path(__file__).resolve().parents[2] / "orchestrator" / "benchmark.py"
    assert "qa_eval" not in benchmark.read_text()


def test_read_tools_are_all_defined_and_read_only():
    """READ_TOOLS' contract is "none of these mutate session state" — keep it
    honest. auto_tag_stems writes tags, which is why it is deliberately absent."""
    from orchestrator.chat import READ_TOOLS, _READ_TOOL_DEFS

    defined = {d["name"] for d in _READ_TOOL_DEFS}
    assert READ_TOOLS <= defined, f"advertised but undefined: {READ_TOOLS - defined}"
    assert "auto_tag_stems" not in READ_TOOLS


# --------------------------------------------------------------------------
# Extract Groove (2026-08-08) — the timing SHAPE, not the timing summary.
# --------------------------------------------------------------------------

def _groove(times, **kw):
    return analysis.extract_groove(_hits(times), SR, "drums",
                                   bpm=BPM, beat_offset_s=1.0, **kw)


def test_extract_groove_puts_swing_in_the_LATENESSES_not_the_grid():
    """The one thing this must not reuse is `_choose_subdivision`.

    That picks the grid which best DESCRIBES a part, and for a swung part the
    honest description is a triplet grid — on which the swung notes sit
    on-grid, the latenesses come out ~0, and the template is flat. The feel
    would have been absorbed by the grid instead of recorded. A groove template
    exists to express swing as lateness on a STRAIGHT grid.
    """
    g = _groove(_eighths(swing=0.65))
    assert g.notes_per_beat == 2, "a swung 8th part must extract on 8ths, not triplets"
    assert len(g.latenesses) == 8
    odd = [g.latenesses[i] for i in (1, 3, 5, 7)]
    even = [g.latenesses[i] for i in (0, 2, 4, 6)]
    assert all(v > 0.5 for v in odd), f"the swing should live here: {odd}"
    assert all(abs(v) < 0.05 for v in even), f"downbeats should be straight: {even}"


def test_extract_groove_refuses_a_quantized_part():
    """Onset detection is accurate to ~1 ms, which is a lateness of ~0.01 —
    real, and musically nothing. Without a floor a dead-straight loop yields a
    template of measurement noise that reports success and does nothing."""
    g = _groove(_eighths(swing=0.5))
    assert g.latenesses == []
    assert "already dead on the grid" in (g.reason or "")


def test_extract_groove_keeps_the_lean_but_does_not_call_it_swing():
    """A take that is uniformly late has a feel worth borrowing, so the lean
    stays IN the template. `bias_ms` reports it from the STRUCTURAL cells only
    — a median over every cell would be dominated by the swing and report a
    large drag on a part that is not dragging at all."""
    late = [t + 0.025 for t in _eighths(swing=0.5)]
    g = _groove(late)
    assert g.notes_per_beat == 2
    assert all(v > 0.05 for v in g.latenesses), "the lean belongs in the groove"
    assert g.bias_ms is not None and 18.0 < g.bias_ms < 32.0

    both = [t + 0.025 for t in _eighths(swing=0.65)]
    gb = _groove(both)
    # swing on the off-beats, lean on everything, and the two read separately
    assert gb.bias_ms is not None and 15.0 < gb.bias_ms < 35.0
    assert gb.latenesses[1] > gb.latenesses[0] + 0.4


def test_extract_groove_is_honest_when_it_cannot_measure():
    assert _groove([1.0, 1.5, 2.0]).reason is not None          # too few onsets
    assert analysis.extract_groove(np.zeros((SR * 4, 2)), SR, "pad",
                                   bpm=BPM).reason == "silent"
    assert _groove(_eighths(swing=0.65), ).grid_bpm == BPM
    no_grid = analysis.extract_groove(_hits(_eighths()), SR, "drums", bpm=None)
    assert no_grid.latenesses == [] and no_grid.reason is not None


def test_extract_groove_latenesses_are_in_tracktions_units():
    """The DAW hands these straight to `GrooveTemplate::setLatenessProportion`,
    where the audible shift is HALF the value times the cell. A deviation of
    `d` beats on a `subdivision` grid is therefore a lateness of
    2*d*subdivision — get this wrong and every extracted groove is half or
    double the feel that was measured."""
    swing = 0.65                       # off-beats at 0.65 of the beat
    g = _groove(_eighths(swing=swing))
    dev_beats = swing - 0.5            # how late they actually are
    expected = 2.0 * dev_beats * g.notes_per_beat
    assert abs(g.latenesses[1] - expected) < 0.05


# ---------- real audio (2026-08-08 review) ----------
# Every behavioural assertion above runs on synthetic audio the tests generate;
# the only real-project tests asserted key presence, which pydantic guarantees —
# measured, all of them passed on the DECLINE path. These pin the detectors'
# actual behaviour on the repo's musical fixture, so a regression to
# decline-on-everything (a threshold slip, tempo plumbing breaking) fails loudly
# instead of leaving 1100 tests green. Values pinned from a verified-good run;
# the code and audio are deterministic, so drift here means behaviour changed.

_SUNO = Path(__file__).resolve().parents[2] / "test_projects" / "suno_demo"


def _suno(stem: str):
    import soundfile as sf
    data, sr = sf.read(_SUNO / f"{stem}.wav", dtype="float32", always_2d=True)
    return data, sr


@pytest.fixture(scope="module")
def suno_grid():
    parts = [_suno(s)[0] for s in ("vocals", "drums", "bass", "other")]
    sr = _suno("vocals")[1]
    n = min(p.shape[0] for p in parts)
    master = sum(p[:n] for p in parts)
    grid = analysis.song_tempo(master, sr)
    assert grid is not None, "the demo master lost its tempo grid"
    return grid, sr


def test_real_audio_tempo_grid_holds(suno_grid):
    (bpm, confidence, beat_offset_s), _ = suno_grid
    assert 118.0 < bpm < 122.0
    assert confidence > 0.5
    assert 0.0 <= beat_offset_s < 60.0 / bpm


def test_real_audio_chords_measure_and_merge(suno_grid):
    """The harmonic stem must yield chords — and MERGED ones: before the
    beat-index contiguity fix this exact file returned 13 consecutive Gmaj
    fragments at its 119.7 BPM grid."""
    (bpm, _, off), sr = suno_grid
    data, _ = _suno("other")
    result = analysis.analyze_chords(data, sr, "other", bpm=bpm, beat_offset_s=off)
    assert result.spans, f"declined on real harmonic material: {result.reason}"
    assert len(result.spans) <= 4, (
        f"adjacent equal chords no longer merge: {len(result.spans)} spans")
    assert result.spans[0].root == "G"


def test_real_audio_timing_measures_the_drums(suno_grid):
    (bpm, _, off), sr = suno_grid
    data, _ = _suno("drums")
    result = analysis.analyze_timing(data, sr, "drums", bpm=bpm, beat_offset_s=off)
    assert result.subdivision is not None, (
        f"declined on the demo drums: {result.reason}")
    assert result.onsets_total >= 8
    assert result.onsets_on_grid * 4 >= result.onsets_total * 3   # >= 75 % on grid
    assert result.bias_ms is not None and abs(result.bias_ms) < 30.0


def test_real_audio_notes_measure_the_vocal(suno_grid):
    _, sr = suno_grid
    data, _ = _suno("vocals")
    result = analysis.analyze_notes(data, sr, "vocals")
    assert result.notes_found >= 1, (
        f"the demo vocal (monophonic) was declined: {result.reason}")


def test_width_reads_antiphase_as_wide_not_mono():
    """L = -R is the WIDEST possible input; it used to fall through the
    mid_power<=0 branch and report None ('mono') beside correlation -1.0."""
    sig = np.sin(2 * np.pi * 220.0 * np.arange(SR) / SR) * 0.3
    assert analysis._width_db(np.stack([sig, -sig], axis=1)) == analysis.BAND_WIDTH_CAP_DB
    assert analysis._width_db(np.zeros((SR, 2))) is None            # silence stays None
    assert analysis._width_db(np.stack([sig, sig], axis=1)) == analysis.BAND_WIDTH_FLOOR_DB


# ---------- the knowledge book is served, not shipped (2026-08-08) ----------

def test_knowledge_loads_from_the_private_module_here():
    """In the private repo (and every dev/eval context) the content module is
    present, so the loader serves the real book with no network."""
    from orchestrator import knowledge

    assert "Audio engineering + music theory reference" in knowledge.get_block()
    topics = knowledge.get_topics()
    assert topics and "arrangement" in topics


def test_knowledge_falls_back_when_the_private_module_is_absent(monkeypatch):
    """The shipped sidecar has no content module and boots before sign-in: the
    loader must degrade to the fallback block (and zero topics) rather than
    crash — and must NOT cache the failure, so the first signed-in turn heals."""
    from orchestrator import knowledge

    monkeypatch.setattr(knowledge, "_cached", None)
    monkeypatch.setattr(knowledge, "_retry_after", 0.0)   # and no negative cache from an earlier test
    monkeypatch.setattr(knowledge, "_backoff_s", 60.0)
    monkeypatch.setattr(knowledge, "_from_env_file", lambda: None)
    monkeypatch.setattr(knowledge, "_from_private_module", lambda: None)
    monkeypatch.setattr(knowledge, "_from_cloud", lambda: None)
    assert knowledge.get_block() == knowledge.fallback_block()
    assert knowledge.get_topics() == {}
    assert knowledge._cached is None            # failure was not cached

    # A failure IS backed off (2026-09-01 sweep: every chat round paid a 6 s connect
    # timeout retrying an unreachable book) — so the same call again is a cheap None…
    calls = []
    monkeypatch.setattr(knowledge, "_from_cloud", lambda: calls.append(1))
    assert knowledge.get_block() == knowledge.fallback_block()
    assert calls == [], "the negative cache must skip the sources"

    # …and the moment a source appears (the device sign-in LINKS -> daw_server calls
    # reset_backoff), the book is live on the very next turn.
    knowledge.reset_backoff()
    monkeypatch.setattr(knowledge, "_from_cloud",
                        lambda: {"block": "# The book", "topics": {"music": ("s", "b")}})
    assert knowledge.get_block() == "# The book"
    assert knowledge.get_topics() == {"music": ("s", "b")}


def test_lookup_tool_vanishes_without_topics(monkeypatch):
    """No topics loaded -> no lookup_knowledge tool, rather than an empty enum
    the API would reject."""
    from orchestrator import chat, knowledge

    names = [d["name"] for d in chat.daw_tool_defs(False)]
    assert "lookup_knowledge" in names          # private module present here

    monkeypatch.setattr(knowledge, "_cached", None)
    monkeypatch.setattr(knowledge, "_retry_after", 0.0)   # and no negative cache from an earlier test
    monkeypatch.setattr(knowledge, "_backoff_s", 60.0)
    monkeypatch.setattr(knowledge, "_from_env_file", lambda: None)
    monkeypatch.setattr(knowledge, "_from_private_module", lambda: None)
    monkeypatch.setattr(knowledge, "_from_cloud", lambda: None)
    bare = [d["name"] for d in chat.daw_tool_defs(False)]
    assert "lookup_knowledge" not in bare


def test_env_file_override_wins(tmp_path, monkeypatch):
    """MIX_AGENT_KNOWLEDGE_FILE is the self-host/test override, first in line."""
    import json

    from orchestrator import knowledge

    override = tmp_path / "book.json"
    override.write_text(json.dumps(
        {"block": "# Override book", "topics": {"only": ["s", "b"]}}))
    monkeypatch.setenv("MIX_AGENT_KNOWLEDGE_FILE", str(override))
    monkeypatch.setattr(knowledge, "_cached", None)
    monkeypatch.setattr(knowledge, "_retry_after", 0.0)   # and no negative cache from an earlier test
    monkeypatch.setattr(knowledge, "_backoff_s", 60.0)
    assert knowledge.get_block() == "# Override book"
    assert knowledge.get_topics() == {"only": ("s", "b")}
    monkeypatch.setattr(knowledge, "_cached", None)   # do not poison later tests


def test_export_knowledge_round_trips():
    from orchestrator.export_knowledge import payload

    data = payload()
    assert data["version"] and len(data["block"]) > 5000
    assert all(len(v) == 2 for v in data["topics"].values())
