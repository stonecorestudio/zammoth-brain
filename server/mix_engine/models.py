"""Tool-surface models. This file is the executable twin of docs/03-mcp-tools.md.

If you change a schema here, update that doc in the same commit (and vice versa).
"""
from __future__ import annotations

import math
from typing import Any, Literal, Optional, Union, get_args

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

# v3: Phase 5 — sections, lufs_range, band_crest_db, transient_energy_ratio,
#     sibilance_density, masking worst_windows (definitions: mix-analysis skill)
# v4: Phase 10 — lufs_s_max / stem lufs_s / lufs_range windows are ungated
#     K-weighted short-term loudness (was gated-integrated per window); same
#     window/hop/gating shapes, ~window-count× faster on real-length audio
# v5: Phase 14 — composites (harshness_db, mud_db, clarity_index), mastering
#     QC (plr_db, psr_db, band_correlation, mono-compat flag), and the
#     deterministic stem classifier (definitions: mix-analysis skill)
# v6: Phase 20 — resonances (narrow PSD peaks with freq/prominence/q),
#     dynamics gain-reduction telemetry (stem `dynamics`, session
#     `master_dynamics`), tempo on sections (tempo_bpm/confidence/
#     beat_offset_s + bar annotations), spatial masking
#     (per_channel_worst_excess_db, spatial_release_db)
# Phase 30 — musical key (KeyEstimate) + song-level tempo on SessionAnalysis
#     (tempo_bpm/tempo_confidence/key). Purely additive Optional fields — no
#     existing metric's meaning changed — so ANALYSIS_VERSION is NOT bumped
#     (cf. the Phase 29 separation metrics, also additive).
# Phase 31 — tuning readout (tuning_offset_cents/tuning_coherence) on both
#     SessionAnalysis and StemAnalysis, plus the TuneAction proposal. Additive
#     Optional fields again — ANALYSIS_VERSION stays 6.
# Phase 34 — limiter report v2 fields on MasterReport. Additive render
#     telemetry/readout only — no existing metric meaning changed, so
#     ANALYSIS_VERSION stays 6.
# Phase 36 — reference-set analysis and section-pair metadata. Additive read
#     surface for matching multiple references; single-reference metrics keep
#     their Phase 7 meaning, so ANALYSIS_VERSION stays 6.
# v7: 2026-10-02 (full audit E5·R1) — a 1-channel signal's loudness is measured AS PLAYED,
#     dual-mono: every loudness of mono material (lufs_i, lufs_s, lufs_s_max, the loudness
#     envelopes, limiter make-up) reads +3.01 dB against v6, and equals what the same audio
#     reads once anything makes it stereo. LRA and every non-loudness metric are unchanged.
# v8: 2026-10-06 (full audit E5·R5; James: "octave-weighted, white / pink / brown still
#     0 / -3 / -6") — spectral_tilt_db_per_oct (and describe_region's tilt + tilt trend) is the
#     slope of the mean power density of each OCTAVE, not of every Welch bin: through v7 half
#     the bins sat above 10 kHz, so the figure was the slope of the top octave. A straight
#     spectrum reads what it read; real material reads shallower, and the low end now moves it.
#     Same version, gates only (E5·R6): `key` is None for fewer than three sounding pitch
#     classes or a fit that falls without its loudest one; the tuning pair is None when the
#     peaks are bursts (a kick, a click), not steady partials. A reported value is unchanged.
# v9: 2026-10-07 (full audit E5·R11; James: "redefine: the agent acts on it") — `resonances`
#     leaves out the part's own NOTES. On material a tuning can be read from, a peak centred
#     within 30 cents of a semitone of that tuning is a played note, and so is one within 25
#     cents of a harmonic (2..64) of such a note; neither is reported. Through v8 the three
#     slots of a pitched stem held the sung or played pitches (439 Hz = the A4). Unpitched
#     material (drums, noise: no tuning) reads exactly what it read. The cost is said in the
#     mix-analysis skill: a real resonance that sits ON a played pitch is not reported.
# v10: 2026-10-09 (user QA O12; James: "we need to take care of these") — `key`: a NEAR-TIE with
#     the relative key is settled by the bass. Where the named key's correlation leads its
#     relative's by less than 0.10, and a quarter or more of the lowest register's power
#     (36-165 Hz, read at 0.7 Hz a bin) sits on the relative key's tonic with at least twice
#     what the named key's tonic holds there, the relative key is reported, with ITS
#     correlation as the confidence. Material with no bass register, and every key that is not
#     a near-tie, reads exactly what it read. Sized on labelled loops (mix-analysis skill).
# v11: 2026-10-09 (James, of Extract Groove on recorded drums: "let's take care of this") — the
#     precise onset detector (`_onset_times_precise`: the tempo grid's refinement,
#     `analyze_timing`, `extract_groove`). Its flux is taken on the ANALYTIC signal, with the low
#     band's envelope added back and a hit masking a much weaker peak just behind it: a recorded
#     kick is ONE onset at its attack, where it was two, or one 25 ms late (a 512-sample window
#     ripples on a tone whose period is longer than it). Hats, snares and anything with nothing
#     under 300 Hz are timed as they were. `analysis._onset_flux` has the measurements.
# v12: 2026-10-11 (James, of the mashup showcase's pre-flight: "yes fix the tempo reader first")
#     — the tempo grid's refinement. The period is read off the precise onsets FOLDED at trial
#     periods within one envelope hop of the lag grid's (`analysis._fold_grid`), where it was a
#     least-squares fit scored against beat lines, which a part with eighths or sixteenths in it
#     could never improve and a long take mis-indexed: two released songs at 124.000 BPM read
#     123.725 and 123.833, their drum stems 123.808 and 123.784, and read 124.001 and 123.999
#     now. `beat_offset_s` is then the class of onsets the envelope piles up on most (the kick's),
#     at the median of its precise onsets, where it was the class nearest the take's FIRST onset.
#     Where the fold does not single out one grid the old fit decides, bit for bit, and
#     `tempo_confidence` is unchanged everywhere. The mix-analysis skill has the sizing.
#     Same day (the showcase's rehearsal): a part with fewer than eight precise onsets has no
#     tempo (a plain sine read 172 BPM at confidence 0.45), and two additive reads -
#     TempoAnalysis (`analyze_tempo`: one stem's own tempo) and ActivityAnalysis
#     (`analyze_activity`: where each part plays, to the bar).
# 13: `tempo_bpm` again (2026-10-11, the second showcase's pre-flight): a tempo in a 3:2 relation
#     to the lag the prior prefers - a dotted kick figure, the commonest shape in hip-hop and trap -
#     is told from the beat by the BAR: the pattern repeats every 3, 4, 6 or 8 beats of the tempo
#     it is in. A released song at 172 (86) read 114.66 at confidence 0.69 on its own drum stem,
#     and reads 86.0; on 344 labelled files five readings move, each to its tempo or the octave,
#     and none that read right. Where the bars do not decide it the reading is as it was.
ANALYSIS_VERSION = 13

BAND_LABELS = ["sub", "bass", "low_mid", "mid", "high_mid", "presence", "brilliance", "air"]
BAND_EDGES_HZ = [20, 60, 150, 400, 1000, 2000, 4000, 8000, 20000]


def imager_crossovers(lo_hz, hi_hz) -> tuple[float, float]:
    """The imager's crossovers as every reader resolves them (= imagerspec::crossovers,
    DeviceSpecs.h; the imager audit 2026-10-21, IM9): each held to its range (30..4000,
    200..16000 Hz), a top at or under the bottom moved to 1.5x it. A crossed pair used to be
    played one way, drawn another and judged by the guardrail a third."""
    import math
    lo = float(lo_hz) if lo_hz is not None and math.isfinite(float(lo_hz)) else 150.0
    hi = float(hi_hz) if hi_hz is not None and math.isfinite(float(hi_hz)) else 2000.0
    lo = max(30.0, min(4000.0, lo))
    hi = max(200.0, min(16000.0, hi))
    if hi <= lo:
        hi = lo * 1.5
    return lo, hi


# ---------- analysis (read) outputs ----------

class Resonance(BaseModel):
    """One narrow spectral peak (Phase 20, mix-analysis skill v6): the
    input a surgical EQ cut needs — exact center frequency, how far it
    stands above its neighborhood, and the Q of its −3 dB bandwidth."""
    freq_hz: int
    prominence_db: float
    q: float


class KeyEstimate(BaseModel):
    """Detected musical key (Phase 30, mix-analysis skill): the song's tonic
    and mode from a chroma vs Krumhansl-Schmuckler key-profile correlation.
    `confidence` is that best correlation (0..1); the detector returns None
    below KEY_MIN_CONFIDENCE, so this model only exists for a confident call."""
    tonic: str   # pitch class with sharps: C, C#, D, ... B
    mode: str    # "maj" | "min"
    confidence: float


class DynamicsTelemetry(BaseModel):
    """What one dynamics processor actually did on the last render
    (Phase 20, mix-analysis skill v6): gain reduction measured dry-vs-wet
    on 5 ms block envelopes, over blocks where the dry signal is active."""
    processor: str        # comp | mb_comp | deess | duck | limiter
    gr_mean_db: float     # mean reduction over active blocks (>= 0)
    gr_max_db: float      # worst-case block reduction (>= 0)
    gr_active_pct: float  # % of active blocks with > 0.5 dB reduction


class SessionAnalysis(BaseModel):
    analysis_version: int = ANALYSIS_VERSION
    # Phase 25: True when measured pre-master (master chain bypassed) for
    # mix-scope reference matching — so records are unambiguous.
    pre_master: bool = False
    lufs_i: Optional[float]
    lufs_s_max: Optional[float]
    lufs_range: Optional[float] = None  # LRA in LU (EBU-style; mix-analysis skill)
    true_peak_dbtp: Optional[float]
    plr_db: Optional[float] = None  # Phase 14 QC: true peak − lufs_i
    psr_db: Optional[float] = None  # Phase 14 QC: true peak − lufs_s_max
    stereo_correlation: Optional[float]
    band_correlation: Optional[dict[str, float]] = None  # Phase 14: per-band L/R
    spectral_tilt_db_per_oct: Optional[float]
    crest_factor_db: Optional[float]
    band_crest_db: Optional[dict[str, float]] = None  # per-band dynamics
    band_energy: Optional[dict[str, float]]  # keys = BAND_LABELS, dB rel. full-band RMS
    # Phase 14 composites (mix-analysis skill v5): the words users actually
    # use, as measurable session scalars — targetable like any metric.
    harshness_db: Optional[float] = None
    mud_db: Optional[float] = None
    clarity_index: Optional[float] = None
    # Phase 20 (v6): narrow spectral peaks worth a surgical cut, and what
    # the master-chain dynamics did on the render that produced this.
    resonances: Optional[list[Resonance]] = None
    master_dynamics: Optional[list[DynamicsTelemetry]] = None
    # Phase 30: song-level musical key + tempo — a descriptive readout, not a
    # mix target. All None on atonal/unrhythmic material. tempo reuses the
    # Phase 20 section detector; key is chroma vs K-S profiles (mix-analysis skill).
    tempo_bpm: Optional[float] = None
    tempo_confidence: Optional[float] = None
    key: Optional[KeyEstimate] = None
    # Phase 31: song-level tuning readout (descriptive, not a target).
    # tuning_offset_cents = global deviation from equal temperament / A440,
    # in cents (±50); tuning_coherence = 0..1 intonation tightness. Both None
    # on unpitched/percussive material (mix-analysis skill).
    tuning_offset_cents: Optional[float] = None
    tuning_coherence: Optional[float] = None
    stems: list[str]
    pass_number: int
    silent: bool = False


class StemAnalysis(BaseModel):
    analysis_version: int = ANALYSIS_VERSION
    stem_id: str
    lufs_s: Optional[float]
    peak_dbfs: Optional[float]
    band_energy: Optional[dict[str, float]]
    transient_density: Optional[float]
    transient_energy_ratio: Optional[float] = None  # 0..1, energy near onsets
    sibilance_density: Optional[float] = None       # 0..1, 5-10 kHz dominant frames
    dynamic_spread_db: Optional[float]
    # Phase 20 (v6): narrow spectral peaks + per-processor gain reduction
    # from this stem's chain on the render measured here.
    resonances: Optional[list[Resonance]] = None
    dynamics: Optional[list[DynamicsTelemetry]] = None
    # Phase 31: this stem's tuning — offset from equal temperament in cents
    # (±50) and intonation coherence (0..1). None on unpitched stems. The
    # actionable readout: a non-zero offset is what TuneAction corrects.
    tuning_offset_cents: Optional[float] = None
    tuning_coherence: Optional[float] = None
    applied_chain: list[dict]  # ordered processors with params, from session state
    silent: bool = False


class PhraseLevelMove(BaseModel):
    """Phase 35: a local level outlier suitable for an existing ranged
    GainAction. `delta_db` is the suggested move toward the active median."""
    start_s: float
    end_s: float
    delta_db: float
    level_dbfs: float
    reason: Literal["too_quiet", "too_loud"]


class PhrasePitch(BaseModel):
    """Intonation of one sung/played phrase (2026-08-08). Two different
    complaints hide behind "out of tune", and they have different fixes:

    - `deviation_cents` — how far this phrase sits from the STEM's own tuning
      centre. A phrase that drifts while the rest of the take is solid is a
      performance problem; the whole take sitting flat is the stem-level
      `tuning_offset_cents`, which one whole-file `tune` action fixes.
    - `tuning_coherence` — 0..1, how tightly the phrase's pitches cluster.
      Low means smeared/unstable intonation (wobble, scooping, a doubled take
      drifting apart). NO tool the agent has fixes this: a whole-file shift
      moves it without tightening it, and per-note correction is parked. Say
      so instead of promising a fix.
    """
    start_s: float
    end_s: float
    tuning_offset_cents: float   # vs A440 equal temperament, ±50
    deviation_cents: float       # vs this stem's own centre — the ranking key
    tuning_coherence: float


class SibilanceBurst(BaseModel):
    """Phase 35: a local 5-10 kHz burst suitable for an existing ranged
    DeessAction."""
    start_s: float
    end_s: float
    density: float
    freq_hz: int
    threshold_db: float


class PhraseAnalysis(BaseModel):
    """Phase 35 phrase-aware readout: compact local level and sibilance
    candidates. Read-only; repair still lands through normal proposals."""
    analysis_version: int = ANALYSIS_VERSION
    stem_id: str
    level_moves: list[PhraseLevelMove] = []
    sibilance_bursts: list[SibilanceBurst] = []
    # Per-phrase intonation (2026-08-08), worst deviation first. Empty on
    # unpitched material (drums, noise) — absence means "not measurable here",
    # never "in tune".
    phrase_pitch: list[PhrasePitch] = []
    # This stem's own tuning centre, for reading `deviation_cents` against.
    stem_tuning_offset_cents: Optional[float] = None
    silent: bool = False


class NoteEvent(BaseModel):
    """One sung or played note (2026-08-08). MONOPHONIC sources only.

    `cents_off` is vs equal temperament at A440; `deviation_cents` is vs the
    STEM's own tuning centre — the same distinction `PhrasePitch` draws, and for
    the same reason: a take that is uniformly flat is one whole-file `tune`, not
    a list of bad notes.

    `spread_cents` is how much the pitch moved WITHIN the note. Vibrato lives
    here too, so a wide value on a sustained note is often expression rather
    than a fault — `cents_off` uses the median and is unmoved by symmetric
    vibrato, which is why the two are reported separately.
    """
    start_s: float
    end_s: float
    midi_note: int
    note_name: str            # e.g. "C#4"
    cents_off: float          # vs A440 equal temperament
    deviation_cents: float    # vs this stem's own centre
    spread_cents: float       # pitch movement within the note (incl. vibrato)


class NotePlayed(BaseModel):
    """One pitch class and how long the part holds it (`NoteAnalysis.played`)."""
    note: str                  # "C#" - the octave is left out: a home note has none
    seconds: float
    share: float               # of all the time the part's notes are held


class NoteAnalysis(BaseModel):
    """Per-note intonation for a MONOPHONIC stem — a lead vocal, a bass, a
    single-line lead. On polyphonic material (piano, guitar chords, a full mix)
    pitch tracking locks onto one voice or the root and the answer is
    meaningless, so this reports a `reason` instead of guessing; use
    `analyze_chords` for harmony.

    `notes` is ordered worst-deviation-first and capped. Empty + `reason` means
    not measurable, never "in tune".

    `played` is the other question - WHICH notes the part plays: every pitch class it holds,
    longest first, over ALL its notes and not only the capped list. `notes` holds the notes
    furthest out of tune, which are as often the ones a part plays least; a bass's home note
    is read from `played` (2026-10-12). Notes under the tracking floor (about C2) are not in it.
    """
    analysis_version: int = ANALYSIS_VERSION
    stem_id: str
    stem_tuning_offset_cents: Optional[float] = None
    notes: list[NoteEvent] = []
    notes_found: int = 0
    played: list[NotePlayed] = []
    reason: Optional[str] = None


class ChordSpan(BaseModel):
    """One detected chord, held for a contiguous run of beats."""
    start_s: float
    end_s: float
    start_bar: Optional[float] = None
    root: str                 # pitch class, e.g. "C", "F#"
    quality: str              # "maj" | "min" | "maj7" | "min7" | "dom7"
    confidence: float         # 0..1 — chroma correlation with the template


class ChordAnalysis(BaseModel):
    """Chords DETECTED from audio (2026-08-08) — deliberately distinct from the
    DAW's `chord_progression`, which is AUTHORED by the user or the agent and is
    not analysis of a recording. Never present one as the other.

    Honest scope: **major/minor triads and major, minor and dominant sevenths**,
    per beat, on tonal material, merged into spans. What it does not do, and
    what the agent must say rather than fake: diminished, augmented and
    suspended chords (dropped below the confidence gate, or forced to the
    nearest modelled quality), 9ths and above, and **inversions** — chroma
    cannot see which note is in the bass, so C/E and C are the same vector and
    the root is reported without the slash. A triad played over a passing
    seventh may read as a seventh chord; the root stays right, which is what
    matters. Accuracy is much better on an isolated harmonic stem than on a
    full mix, where drums and vocals smear the chroma.

    Empty `spans` with a `reason` means not detectable, never "no chords".
    """
    analysis_version: int = ANALYSIS_VERSION
    stem_id: str
    key: Optional[KeyEstimate] = None    # the prior the detection was scored with
    grid_bpm: Optional[float] = None
    spans: list[ChordSpan] = []
    reason: Optional[str] = None


class GrooveExtraction(BaseModel):
    """A groove template measured off real audio — Ableton's Extract Groove
    (2026-08-08). Where `TimingAnalysis` answers "is this take tight?" with
    aggregates, this keeps the *shape*: the per-grid-cell lean that makes the
    part feel the way it does, ready to apply to something else.

    `latenesses` is one value per cell of a 4/4 bar at `notes_per_beat`
    resolution, in Tracktion's units: a proportion of the cell where the
    audible shift is **half** the value times the cell length, positive = late.
    Cells no onset landed on are 0.0 (straight) — `cells_measured` says how
    many carried real evidence, so a sparse part cannot pass itself off as a
    fully-specified groove.

    The take's overall lean is **kept in** the template rather than centred
    out: a part that sits 20 ms behind the beat feels that way, and that lean
    is exactly what someone extracting its groove is trying to borrow.
    `bias_ms` reports it separately so the caller can see what it is taking.

    Cell 0 is the grid line at the analysed tempo's `beat_offset_s`. Applying
    the groove anchors cell 0 to edit beat 0, so a clip whose downbeat is not
    on the song's grid gets a rotated feel — true of Live too, and the reason
    Extract Groove is a downbeat-aligned-loop gesture.
    """
    source: str
    grid_bpm: Optional[float] = None
    notes_per_beat: Optional[int] = None   # straight grids only: 1/2/4/8 (never a triplet
                                           # grid — that is the whole point of _choose_groove_grid)
    latenesses: list[float] = []           # per cell, −1..1, + = late
    cells_measured: int = 0                # cells an onset actually landed on
    bias_ms: Optional[float] = None        # the take's overall lean, kept IN the groove
    onsets_used: int = 0
    reason: Optional[str] = None           # why nothing was extracted


class TimingAnalysis(BaseModel):
    """Onset timing against the song's grid (2026-08-08) — "is this take
    rushing, and is it tight?"

    Sign convention, everywhere: **negative = EARLY** (ahead of the beat,
    rushing), **positive = LATE** (behind the beat, dragging).

    `bias_ms` and `spread_ms` are different questions and must not be conflated:
    a player can be consistently 20 ms behind (large bias, small spread — a
    deliberate, good-sounding lean) or centred but scattered (no bias, large
    spread — actually sloppy). Only the second is a fault.

    Swing is separated from drag deliberately. On a swung part the off-beats are
    SUPPOSED to be late; counting that as drag would call a good groove sloppy.
    `bias_ms` is therefore measured on the structural beats only, and the lean
    of the off-beats is reported as `swing_percent`.

    All measurement fields are None when the part is not on a grid at all (free
    time, unrhythmic material, or a tempo estimate that does not hold) — that is
    "not measurable", never "in time". `reason` says which.
    """
    stem_id: str
    grid_bpm: Optional[float] = None
    # 1 = quarters, 2 = eighths, 4 = sixteenths, 3/6 = triplet grids.
    subdivision: Optional[int] = None
    bias_ms: Optional[float] = None      # −early / +late, structural beats
    spread_ms: Optional[float] = None    # jitter once bias AND swing are removed
    # 50 = straight, ~66-67 = triplet swing. None on triplet grids (the grid
    # already expresses it) or when either parity class is too sparse.
    swing_percent: Optional[float] = None
    onsets_total: int = 0
    onsets_on_grid: int = 0
    reason: Optional[str] = None         # why nothing was measured


class TempoAnalysis(BaseModel):
    """ONE part's own tempo (`analyze_tempo`, 2026-10-11): the same estimator as the song's,
    run on that stem alone as it plays (through its clips), so `beat_offset_s` is a time on the
    timeline. Every other grid the engine reads is the master's - with two songs in a project
    that is neither's. All three are None with a `reason` when the part has no steady pulse:
    that is "not measurable", never "free time". Which beat is bar 1 is not said."""
    stem_id: str
    tempo_bpm: Optional[float] = None
    tempo_confidence: Optional[float] = None   # <= 0.45: the octave (or a 3:2) was not decided
    beat_offset_s: Optional[float] = None      # where a beat falls, within the first beat
    reason: Optional[str] = None


class ActivitySpan(BaseModel):
    """A run of whole bars in which a part is in one state (`analyze_activity`)."""
    state: str                 # "in" | "low" | "out"
    from_bar: int              # 1-based, inclusive
    to_bar: int                # inclusive
    level_db: Optional[float] = None   # the run's median bar level, dBFS RMS; None when out
    # Only on a run the part ENTERS with (the one before it is quieter, or there is none):
    # the moment its sound starts, on the timeline, and where in the run's first bar that is,
    # in beats (1.0 = on the bar line, 2.0 = one beat after it). To the bar, a grid drawn a
    # beat early reads exactly like the right one (2026-10-12, the second showcase's rehearsal).
    enters_s: Optional[float] = None
    enters_beat: Optional[float] = None


class StemActivity(BaseModel):
    stem_id: str
    spans: list[ActivitySpan] = Field(default_factory=list)


class ActivityAnalysis(BaseModel):
    """Where each part comes in and drops out, to the BAR (`analyze_activity`, 2026-10-11).

    A bar is `in` when its level is within 12 dB of the part's own loud bars, `low` within 30
    (a hat fill under a break, a tail), `out` below that or under the silence floor. The grid
    is the caller's (`bpm`, `bar1_s` = where bar 1 starts on the timeline), else the master's
    detected tempo with bar 1 at its first beat - and then which beat is the downbeat is not
    known. Empty `stems` with a `reason` means no grid to count bars on."""
    bpm: Optional[float] = None
    bar1_s: Optional[float] = None
    beats_per_bar: int = 4
    bars: int = 0
    stems: list[StemActivity] = Field(default_factory=list)
    reason: Optional[str] = None


class StemArtifacts(BaseModel):
    """Phase 29 separation-quality metrics for one stem (analyze_separation).
    All 0..1, higher = more of that artifact; None when the stem is too
    short/silent to measure. Definitions are contractual: mix-analysis skill."""
    stem_id: str
    lufs_s: Optional[float] = None
    hf_shimmer: Optional[float] = None           # HF musical-noise in quiet frames
    transient_smear: Optional[float] = None       # pre-echo energy around onsets
    spectral_hole_ratio: Optional[float] = None   # carved notches in content bands
    leakage: Optional[float] = None               # bleed tracking another stem
    silent: bool = False


class SeparationReport(BaseModel):
    """Phase 29 — the 'double check after splitting': per-stem artifact metrics
    plus the cross-stem loudness spread. Read-only; the agent reads these to
    decide repair (the AI-cleanup preset). Separation itself is a user import
    action, not an MCP tool."""
    analysis_version: int = ANALYSIS_VERSION
    per_stem: list[StemArtifacts]
    inter_stem_lufs_spread_db: Optional[float] = None
    # Active stems the app did NOT split, so they were not measured (reader E5·R7): these
    # metrics read sibilance as shimmer, sustain as smear and playing in time as leakage on
    # ordinary clean material. A stem split elsewhere and imported is listed here too.
    not_separated: list[str] = []


class MaskingWindow(BaseModel):
    start_s: float
    end_s: float
    excess_db: float


class MaskingAnalysis(BaseModel):
    analysis_version: int = ANALYSIS_VERSION
    stem_a: str
    stem_b: str
    per_band_excess_db: dict[str, float]  # positive = stem_b overpowers stem_a
    worst_band: str
    worst_band_excess_db: float
    worst_windows: list[MaskingWindow] = []  # up to 3 worst runs in worst_band
    # Phase 20 (v6): spatial picture of the worst band — excess per channel
    # and how much clearer the best channel is than the mono sum. Pan
    # separation finally registers; both None when both stems are mono.
    per_channel_worst_excess_db: Optional[dict[str, float]] = None  # left/right
    spatial_release_db: Optional[float] = None


class PhaseBand(BaseModel):
    """One band's stereo phase (`analyze_phase`, 2026-10-13; mix-analysis skill ▸ *Phase*).
    `corr` is the band's L/R correlation over the whole song (+1 mono, 0 two unrelated channels,
    -1 anti-phase); `below_zero_pct` the share of the band's SOUNDING ~0.3 s windows whose own
    correlation is below 0 - the stretches that cancel when the song is folded to mono, and what
    a device's PHASE lamp lights on; `worst` the 5th-percentile window; `mono_db` how much quieter
    the band is folded to mono than it plays in stereo (0 = mono, -3 = two unrelated channels,
    deeper = cancelling). `file_corr` (stems): the same band of the stem's FILE, before its
    chain - so a chain that made a band cancel is told from a file that arrives that way."""
    corr: float
    below_zero_pct: float
    worst: float
    mono_db: float
    file_corr: Optional[float] = None


class PhaseTarget(BaseModel):
    """One channel's phase reading: a stem, a group or the master. `bands` holds only the bands
    that carry the channel's sound (within 30 dB of its total). For a stem an issue names (its
    own, or one of the master's `from_stems`): `width`, its strip width now, and
    `mono_safe_width`, the widest strip width at which every band of it reads +0.2 or more over
    the song and below 0 less than 20 % of the time - and its bands carry `file_corr`."""
    target: str
    kind: Literal["stem", "bus", "master"]
    corr: Optional[float] = None
    bands: dict[str, PhaseBand] = Field(default_factory=dict)
    width: Optional[float] = None
    mono_safe_width: Optional[float] = None


class PhaseIssue(BaseModel):
    """Something that cancels when the song is folded to mono (mix-analysis skill ▸ *Phase*).
    `inverted_channel`: one side plays upside down (the whole channel; `band` None).
    `channel_delay`: the two sides are the same sound `lag_ms` apart (a Haas widener, a delay
    on one side) - it combs when folded to mono (`band` None). `cancels_in_mono`: on the MASTER,
    a band below 0 over the song or below 0 for `below_zero_pct` >= 20 % of its sounding time;
    on a stem or group, a band that is anti-phase as a whole (corr <= -0.2: it folds to mono
    quieter than two unrelated channels would). A wide stem on its own is not an issue - the
    master is the judge, and its issue names the stems it comes from (`from_stems`, widest
    side first). `cause`: `file` (the audio arrives that way), `chain` (the channel's own
    processing - its input does not), `master_chain` (the master's devices; the mix going in
    does not), `mix` (the mix going into the master already does). `fix`: the move, in a
    sentence."""
    target: str
    band: Optional[str] = None
    kind: Literal["inverted_channel", "channel_delay", "cancels_in_mono"]
    cause: Literal["file", "chain", "master_chain", "mix"]
    corr: float
    below_zero_pct: Optional[float] = None
    mono_db: Optional[float] = None
    lag_ms: Optional[float] = None
    from_stems: list[str] = Field(default_factory=list)
    fix: str


class PhaseConflict(BaseModel):
    """Two stems whose sum cancels in a low band (sub / bass / low_mid): measured on their
    CENTRE (mid) signals, at their mix levels, only in windows where both sound within 10 dB of
    each other. `corr` is their zero-lag correlation there (below -0.3 is a conflict), `sum_db`
    how much their sum falls short of their power sum (0 = independent, negative = cancelling),
    `overlap_pct` the share of the song both sound in the band. When the two line up at a small
    offset instead (a timing problem, not a polarity one), `lag_ms` says by how much
    `stems[1]` arrives EARLY relative to `stems[0]` (negative = late) and `corr_at_lag` what
    the correlation is there; both None when flipping one stem's polarity is the fix."""
    stems: list[str]
    band: str
    corr: float
    sum_db: float
    overlap_pct: float
    lag_ms: Optional[float] = None
    corr_at_lag: Optional[float] = None
    fix: str


class PhaseAnalysis(BaseModel):
    """The mix's stereo phase in one read (`analyze_phase`, 2026-10-13): the master, every stem
    (and group) in the mix, what cancels in mono and why, and pairs of stems that cancel each
    other in the lows. `clean` is True when there is no issue and no conflict - the PHASE lamps
    of the DAW's imager, width, utility and band-width devices light on the same rule (a band's
    ~0.3 s correlation below 0). `window_s` is the window the per-window figures are measured
    on. Definitions are contractual: mix-analysis skill ▸ *Phase*."""
    analysis_version: int = ANALYSIS_VERSION
    window_s: float
    clean: bool
    master: Optional[PhaseTarget] = None
    targets: list[PhaseTarget] = Field(default_factory=list)
    issues: list[PhaseIssue] = Field(default_factory=list)
    conflicts: list[PhaseConflict] = Field(default_factory=list)
    note: Optional[str] = None


class RegionEnvelopePoint(BaseModel):
    t_s: float                 # window start, absolute song time
    lufs_s: Optional[float]    # short-term loudness of that window


class RegionMasker(BaseModel):
    stem_id: str
    worst_band: str
    excess_db: float           # positive: that stem overpowers this one in the band


class RegionDescription(BaseModel):
    """`describe_region` (2026-09-03): a compact perceptual read of ONE stem
    over a time range — the agent's way to look at 'this part' instead of the
    whole file. Every number is measured on the stem's PROCESSED audio
    (after its applied chain), sliced to [start_s, end_s). Silent regions
    carry `silent=True` and no metrics."""
    analysis_version: int = ANALYSIS_VERSION
    stem_id: str
    start_s: float
    end_s: float
    silent: bool = False
    # The window's own INTEGRATED loudness (2026-10-11, the mashup showcase's mastering turn): the
    # figure a section is reported by. `stem_id` may be "master" - a bar range of the mix.
    lufs_i: Optional[float] = None
    lufs_s: Optional[float] = None            # loudest short-term window
    lufs_envelope: list[RegionEnvelopePoint] = []   # <= 8 points across the region
    peak_dbfs: Optional[float] = None
    true_peak_dbtp: Optional[float] = None    # between the samples (4x), over the window
    crest_db: Optional[float] = None
    dynamic_spread_db: Optional[float] = None
    band_energy: Optional[dict[str, float]] = None
    spectral_tilt_db_per_oct: Optional[float] = None
    tilt_trend_db_per_oct: Optional[float] = None   # second half minus first half
    transient_density: Optional[float] = None       # onsets per second
    transient_energy_ratio: Optional[float] = None
    harshness_db: Optional[float] = None
    mud_db: Optional[float] = None
    clarity_index: Optional[float] = None
    masked_by: list[RegionMasker] = []   # other stems overpowering this one here
    masks: list[RegionMasker] = []       # other stems this one overpowers here


class Section(BaseModel):
    label: str   # similarity letters: sections that sound alike share a letter
    start_s: float
    end_s: float
    lufs_i: Optional[float]
    band_energy: Optional[dict[str, float]]
    stem_activity: dict[str, float]  # stem_id -> fraction of active 400 ms blocks
    # Phase 20 (v6): musical position (4/4 against the detected grid),
    # present only when tempo confidence clears the floor.
    start_bar: Optional[float] = None
    end_bar: Optional[float] = None
    # Stereo picture per section — the numbers behind "does the chorus open up
    # against the verse". width_db is side-vs-mid power (0 = side as strong as
    # mid, more negative = narrower); correlation is zero-lag L/R Pearson,
    # where a negative value on a section is a mono-compatibility warning.
    # Both None for mono or silent sections.
    width_db: Optional[float] = None
    stereo_correlation: Optional[float] = None


class SectionAnalysis(BaseModel):
    analysis_version: int = ANALYSIS_VERSION
    duration_s: float
    sections: list[Section]
    # Phase 20 (v6): spectral-flux autocorrelation tempo estimate. All None
    # when no rhythmic grid clears the confidence floor (mix-analysis skill).
    tempo_bpm: Optional[float] = None
    tempo_confidence: Optional[float] = None
    beat_offset_s: Optional[float] = None


class ReferenceAnalysis(BaseModel):
    """Profile of a reference track (Phase 7). The agent steers the *delta*
    between the mix and these values toward zero — band_energy is relative to
    each render's own power, so the per-band difference is the tonal gap."""
    analysis_version: int = ANALYSIS_VERSION
    path: str
    duration_s: float
    lufs_i: Optional[float]
    lufs_range: Optional[float]
    crest_factor_db: Optional[float]
    band_energy: Optional[dict[str, float]]
    band_width_db: Optional[dict[str, float]]  # side/mid power per band; None if mono
    silent: bool = False
    # The file's peaks (2026-10-11): the tool is also how an exported file is measured, and the
    # agent could say its loudness and not its true peak.
    sample_peak_dbfs: Optional[float] = None
    true_peak_dbtp: Optional[float] = None
    # Set when the profile is of ONE STRETCH of the file (`start_s` / `end_s`): one section of
    # an exported song, which holds what the DAW played - automation included.
    window_s: Optional[tuple[float, float]] = None


# ---------- one sound (Phase 58) ----------
# A one-shot or a loop, measured by `sample_profile.sample_profile` - what `analyze_sample`
# returns and what the brain attaches to each sound of a `sound.render`. Definitions:
# mix-analysis skill, *One sound*. A change to what a field MEANS moves this number.
SAMPLE_PROFILE_VERSION = 1


class SampleLoop(BaseModel):
    """A loop held to the length its tempo says, and read at its seam."""
    bpm: float
    beats: float
    expected_s: float
    length_error_ms: float        # the file against bpm x beats; + = long
    length_error_samples: int
    seam_step_db: Optional[float]     # last sample -> first sample, dB re the file's peak (None: no step)
    seam_excess_db: Optional[float]   # that step over the file's own largest sample-to-sample steps; > 0 clicks
    seam_level_db: Optional[float]    # the last 20 ms against the first 20 ms


class SampleProfile(BaseModel):
    profile_version: int = SAMPLE_PROFILE_VERSION
    path: Optional[str] = None
    kind: str                         # "one_shot" | "loop"
    sample_rate_hz: int
    channels: int
    duration_s: float                 # the file
    silent: bool = False
    leading_silence_ms: Optional[float] = None   # before the envelope first stands within 40 dB of its peak
    sounding_s: Optional[float] = None           # from there to where it last stands within 60 dB
    attack_ms: Optional[float] = None            # 10 % -> 90 % of the peak envelope
    decay_ms: Optional[float] = None             # peak -> 20 dB under it; None = it never falls that far
    end_level_dbfs: Optional[float] = None       # the file's last 5 ms: what a cut leaves
    sample_peak_dbfs: Optional[float] = None
    true_peak_dbtp: Optional[float] = None
    rms_dbfs: Optional[float] = None             # over what sounds
    loudness_lufs: Optional[float] = None        # K-weighted, ungated, over what sounds - at any length
    crest_db: Optional[float] = None
    centroid_hz: Optional[float] = None
    band_share_pct: Optional[dict[str, float]] = None   # keys = BAND_LABELS, per cent of the power
    pitch_hz: Optional[float] = None             # the body's pitch; None = unpitched (or a loop)
    pitch_note: Optional[str] = None             # scientific names, as analyze_notes gives them: A4 = 440 Hz
    pitch_midi: Optional[int] = None             # the nearest MIDI note number (60 = middle C): what a note's `pitch` is
    pitch_cents: Optional[float] = None
    pitch_confidence: Optional[float] = None
    pitch_start_hz: Optional[float] = None       # where the pitch started, when it moved a semitone or more
    pitch_drop_semitones: Optional[float] = None # + = it FALLS to pitch_hz
    pitch_settle_ms: Optional[float] = None      # from the onset until it is within a semitone of pitch_hz
    noisiness: Optional[float] = None            # 1 - the body's periodicity: a tone 0, noise 1
    dc_offset_pct: Optional[float] = None        # of full scale
    full_scale_samples: int = 0
    over_full_scale: bool = False                # a float file past 0 dBFS: it clips when written fixed-point
    width_db: Optional[float] = None             # side / mid power, -40..+6; None = one channel
    stereo_correlation: Optional[float] = None
    loop: Optional[SampleLoop] = None
    # ---- only when the caller said what the sound is meant to be (Phase 60, sound_classes)
    sound_kind: Optional[str] = None             # "kick", "pad", ...: the kind it was read against
    issues: Optional[list[str]] = None           # where it sits outside what that kind usually measures as; [] = nowhere
    issues_are: Optional[str] = None             # the caveat: a guide, not a verdict


ReferenceRole = Literal["tonality", "width", "loudness"]


class ReferenceSpec(BaseModel):
    """One member of a Phase 36 reference set. Roles decide which aggregate
    target families this reference contributes to; they do not change the
    underlying single-reference analysis."""
    path: str
    weight: float = Field(default=1.0, gt=0.0)
    roles: list[ReferenceRole] = Field(
        default_factory=lambda: ["tonality", "width", "loudness"])

    @model_validator(mode="after")
    def _check_roles(self) -> "ReferenceSpec":
        if not self.roles:
            raise ValueError("reference spec needs at least one role")
        self.roles = list(dict.fromkeys(self.roles))
        return self


class ReferenceSetMember(BaseModel):
    spec: ReferenceSpec
    analysis: ReferenceAnalysis
    sections: Optional[SectionAnalysis] = None


class ReferenceSectionContribution(BaseModel):
    path: str
    weight: float
    section_index: int
    label: str
    start_s: float
    end_s: float


class ReferenceSectionTarget(BaseModel):
    """A section-aware tonal target: compare this mix section to the weighted
    reference section blend, then still apply ordinary match-EQ actions."""
    mix_index: int  # loudness rank (0 = loudest), the cross-render-stable key
    mix_label: str
    mix_start_s: float
    mix_end_s: float
    band_energy: dict[str, float]
    confidence: float
    references: list[ReferenceSectionContribution]


class ReferenceSetAnalysis(BaseModel):
    """Weighted aggregate of multiple references (Phase 36). The top-level
    fields intentionally mirror ReferenceAnalysis where possible so existing
    target code can consume the set as a richer reference."""
    analysis_version: int = ANALYSIS_VERSION
    path: str
    references: list[ReferenceSetMember]
    lufs_i: Optional[float]
    lufs_range: Optional[float]
    crest_factor_db: Optional[float]
    band_energy: Optional[dict[str, float]]
    band_width_db: Optional[dict[str, float]]
    section_targets: list[ReferenceSectionTarget] = Field(default_factory=list)
    section_fallback: bool = True
    silent: bool = False


# ---------- actions ----------

class EqBand(BaseModel):
    freq_hz: float = Field(ge=20, le=20000)
    # User co-edit (native EQ-8) may boost to +12 for a symmetric scale; the agent
    # stays capped at +6 via the eq_boost_max guardrail (guardrails bind the agent,
    # not the user). Cuts to -12.
    gain_db: float = Field(ge=-12, le=12)
    q: float = Field(ge=0.3, le=8.0)
    # Phase 46 (Pro-Q-class EQ, Tranche 1): `bandpass` (a single RBJ bandpass biquad,
    # gain_db ignored, q sets width) and `tilt_shelf` (a matched low-shelf -g/2 +
    # high-shelf +g/2 pair at freq_hz — gain_db is the total tilt) join the existing
    # shapes.
    shape: Literal["peak", "low_shelf", "high_shelf", "hpf", "lpf", "notch",
                  "bandpass", "tilt_shelf"]
    # Phase 46: variable-slope cuts, hpf/lpf only. A Butterworth cascade of
    # `slope_db_oct // 12` RBJ 2nd-order biquad sections (6 = today's 1st-order
    # Pedalboard filter — legacy behavior, bit-identical; the default/unset value,
    # so existing sessions render unchanged). Ignored for every other shape.
    slope_db_oct: Literal[6, 12, 24, 48, 96] = 6


def _check_time_range(time_range_s) -> None:
    """Shared validation for time-confined actions (Phases 5 + 15)."""
    if time_range_s is None:
        return
    start_s, end_s = time_range_s
    # Not finite first (reader D9·R9): both comparisons below are False for NaN, and the valid
    # JSON number 1e999 reads as infinity - the one unbounded numeric pair in every ranged
    # action. An applied one was written to session.json as `Infinity` and made every later
    # render raise (`int(start_s * sample_rate)`) until the proposal was undone.
    if not (math.isfinite(start_s) and math.isfinite(end_s)):
        raise ValueError("time_range_s must be two finite numbers of seconds")
    if start_s < 0:
        raise ValueError("time_range_s start must be >= 0")
    if end_s - start_s < 0.5:
        raise ValueError("time_range_s must span at least 0.5 s")


def _check_wet_target(action) -> None:
    """Shared validation for the wet space devices (reverb / delay / IR reverb), which
    since 2026-07-20 have TWO placements: an FX RETURN (`return_id`) or a plain INSERT on
    a track / bus / master (`stem_id`). Exactly one — an action naming both is ambiguous
    (which chain owns it?), and one naming neither has no target at all."""
    if bool(action.return_id) == bool(action.stem_id):
        raise ValueError(f"{action.type} needs exactly one of return_id (FX return) "
                         "or stem_id (insert on a track/bus/master)")


class EqAction(BaseModel):
    type: Literal["eq"] = "eq"
    stem_id: str
    band: EqBand
    # Phase 41: M/S — apply the band to only the mid or side of a stereo signal
    # ("stereo" = both, the default). Encoded/decoded around the filter; no-op on mono.
    # Phase 46: `left`/`right` process only that stereo channel directly (index 0/1,
    # no M/S encode/decode) — the other channel renders bit-identically.
    channel: Literal["stereo", "mid", "side", "left", "right"] = "stereo"
    # 2026-07-12 gain staging: the EQ device's OUTPUT trim (EQ-Eight's Output knob) —
    # a plain FULL-signal post-bank gain, so a boost/cut can be level-compensated on
    # the device instead of a Utility after it. Entry-level; the DEVICE's audible trim
    # is the SUM across all eq entries on the node (exact — the bank is linear). The
    # UI writes it on the first band; the agent should set it on one entry only.
    output_db: float = Field(default=0.0, ge=-12.0, le=12.0)
    # Phase 15: confine the band to [start, end] seconds — rendered as a
    # crossfaded wet/dry blend with the ride-envelope fades. Stems only.
    time_range_s: Optional[tuple[float, float]] = None

    @model_validator(mode="after")
    def _check_range(self) -> "EqAction":
        _check_time_range(self.time_range_s)
        return self


class CompChannelParams(BaseModel):
    """One channel's compressor settings for a true M/S comp (Phase 44): the mid and
    the side each get their own block, processed independently and summed."""
    threshold_db: float = Field(ge=-60, le=0)
    ratio: float = Field(ge=1.0, le=10.0)
    attack_ms: float = Field(ge=0.1, le=100)
    release_ms: float = Field(ge=10, le=1000)
    makeup_db: float = Field(default=0.0, ge=0, le=12)
    mix: float = Field(default=1.0, ge=0.0, le=1.0)


class CompAction(BaseModel):
    type: Literal["comp"] = "comp"
    stem_id: str
    threshold_db: float = Field(ge=-60, le=0)
    ratio: float = Field(ge=1.0, le=10.0)
    attack_ms: float = Field(ge=0.1, le=100)
    release_ms: float = Field(ge=10, le=1000)
    makeup_db: float = Field(default=0.0, ge=0, le=12)
    # Phase 23: parallel (NY) compression — wet/dry blend post-makeup.
    # 1.0 = fully wet (classic insert, the pre-Phase-23 behavior);
    # 0.0 renders bit-exactly dry.
    mix: float = Field(default=1.0, ge=0.0, le=1.0)
    # Phase 41: M/S — compress only the mid or side of a stereo signal ("stereo" =
    # both, the default). e.g. side-only comp to tame a wide, unstable stereo image.
    channel: Literal["stereo", "mid", "side"] = "stereo"
    # Phase 44: TRUE M/S — independent compressors on the mid AND the side at once.
    # Provide BOTH blocks; the flat params above are ignored for the M/S channels.
    mid: Optional[CompChannelParams] = None
    side: Optional[CompChannelParams] = None
    time_range_s: Optional[tuple[float, float]] = None  # Phase 15, stems only

    @model_validator(mode="after")
    def _check_range(self) -> "CompAction":
        _check_time_range(self.time_range_s)
        if (self.mid is None) != (self.side is None):
            raise ValueError("a true M/S comp needs BOTH `mid` and `side` blocks")
        if self.mid is not None and self.channel == "stereo":
            self.channel = "mid"   # mark the entry as M/S (the engine keys M/S off channel)
        return self


class GainAction(BaseModel):
    type: Literal["gain"] = "gain"
    stem_id: str  # a stem id or "master"
    delta_db: float = Field(ge=-24, le=12)
    # Phase 5 gain ride: confine the move to [start, end] seconds. Renders as a
    # deterministic envelope with 50 ms raised-cosine fades (audio-dsp skill).
    # Stems only — session rejects rides on "master".
    time_range_s: Optional[tuple[float, float]] = None

    @model_validator(mode="after")
    def _check_range(self) -> "GainAction":
        _check_time_range(self.time_range_s)
        return self


class DeessAction(BaseModel):
    """Split-band de-esser (v2 2026-07-05): compress the band above `freq_hz` when
    it exceeds `threshold_db`. v2 frees the old fixed 4:1 / 1 ms / 50 ms dynamics
    (`ratio`/`attack_ms`/`release_ms`), caps the reduction depth with `range_db`
    (−6..−12 keeps esses natural; the −80 default is effectively uncapped), and
    adds `wide=True` — apply the computed gain to the WHOLE signal (classic
    wide-band de-essing, dips the full level on ess hits — often more transparent)
    while the detector stays the band above `freq_hz`."""
    type: Literal["deess"] = "deess"
    stem_id: str
    threshold_db: float = Field(ge=-60, le=0)
    freq_hz: float = Field(ge=2000, le=12000)
    ratio: float = Field(default=4.0, ge=1.0, le=20.0)
    attack_ms: float = Field(default=1.0, ge=0.1, le=20.0)
    release_ms: float = Field(default=50.0, ge=10.0, le=500.0)
    range_db: float = Field(default=-80.0, ge=-80.0, le=0.0)
    wide: bool = False
    time_range_s: Optional[tuple[float, float]] = None  # Phase 15, stems only

    @model_validator(mode="after")
    def _check_range(self) -> "DeessAction":
        _check_time_range(self.time_range_s)
        return self


class MatchEqAction(BaseModel):
    """Match-EQ (Phase 15): per-band corrections rendered as one
    delay-compensated linear-phase FIR (audio-dsp skill). The corrections
    are computed at propose time from the mix-vs-reference band gaps —
    8 numbers in session state, never a path to external audio, so renders
    stay deterministic from state alone."""
    type: Literal["match_eq"] = "match_eq"
    stem_id: str  # a stem id or "master"
    band_gains_db: dict[str, float]  # subset of BAND_LABELS -> dB

    @model_validator(mode="after")
    def _check_bands(self) -> "MatchEqAction":
        if not self.band_gains_db:
            raise ValueError("match_eq needs at least one band gain")
        unknown = set(self.band_gains_db) - set(BAND_LABELS)
        if unknown:
            raise ValueError(f"unknown bands: {sorted(unknown)} "
                             f"(bands: {BAND_LABELS})")
        for band, gain_db in self.band_gains_db.items():
            if not -12.0 <= gain_db <= 6.0:
                raise ValueError(f"band {band}: gain {gain_db:+.1f} dB outside "
                                 "[-12, +6]")
        return self


class DynamicEqAction(BaseModel):
    """Dynamic spectral EQ cut (Phase 38): isolate a narrow band and attenuate it
    only while that band's envelope crosses threshold. This is the surgical tool
    for resonances/harshness/mud bursts that are not constant enough for static
    EQ. Downward only; use ordinary EQ for broad/static boosts."""
    type: Literal["dynamic_eq"] = "dynamic_eq"
    stem_id: str  # a stem id, bus id, or "master"
    freq_hz: float = Field(ge=40, le=16000)
    gain_db: float = Field(ge=-12.0, le=0.0)
    q: float = Field(ge=0.3, le=12.0)
    threshold_db: float = Field(ge=-60.0, le=0.0)
    ratio: float = Field(ge=1.0, le=10.0)
    attack_ms: float = Field(default=10.0, ge=1.0, le=100.0)
    release_ms: float = Field(default=120.0, ge=20.0, le=1000.0)
    time_range_s: Optional[tuple[float, float]] = None

    @model_validator(mode="after")
    def _check_range(self) -> "DynamicEqAction":
        _check_time_range(self.time_range_s)
        if self.gain_db == 0.0:
            raise ValueError("dynamic_eq gain_db must be a cut, not 0")
        return self


class MbBand(BaseModel):
    """One band of the multiband compressor (Phase 15). `up` flips it to UPWARD
    compression (boost the band when it falls BELOW threshold — lift quiet detail
    instead of taming peaks). Upward mode is rejected on the master (it pushes
    loudness; only the limiter may)."""
    threshold_db: float = Field(ge=-60, le=0)
    ratio: float = Field(ge=1.0, le=10.0)
    makeup_db: float = Field(default=0.0, ge=0, le=12)
    up: bool = False


class MultibandCompAction(BaseModel):
    """Three-band compressor (Phase 15, un-parked from Phase 7): exactly-complementary
    FFT band split, shared timing. Crossovers default to 150 Hz / 2 kHz but are movable
    (`crossover_lo_hz` / `crossover_hi_hz`). Master allowed (per-band makeup must be 0
    there — only the limiter pushes loudness)."""
    type: Literal["mb_comp"] = "mb_comp"
    stem_id: str  # a stem id or "master"
    low: MbBand
    mid: MbBand
    high: MbBand
    attack_ms: float = Field(default=10.0, ge=0.1, le=100)
    release_ms: float = Field(default=150.0, ge=10, le=1000)
    # Movable band edges (Phase 44 — exposed to the agent, like the native co-edit).
    crossover_lo_hz: float = Field(default=150.0, ge=30.0, le=4000.0)
    crossover_hi_hz: float = Field(default=2000.0, ge=200.0, le=16000.0)

    @model_validator(mode="after")
    def _check_crossovers(self) -> "MultibandCompAction":
        if self.crossover_lo_hz >= self.crossover_hi_hz:
            raise ValueError("crossover_lo_hz must be below crossover_hi_hz")
        return self


class OttBand(BaseModel):
    """One band of the OTT upward/downward compressor (2026-07-07): how hard the
    band leans into each side of the levelling (`up` boosts quiet material,
    `down` tames loud material; 0 = that side off, 1 = full), plus a post gain."""
    up: float = Field(default=1.0, ge=0.0, le=1.0)
    down: float = Field(default=1.0, ge=0.0, le=1.0)
    gain_db: float = Field(default=0.0, ge=-12.0, le=12.0)


class OttAction(BaseModel):
    """OTT-style 3-band upward+downward compressor (2026-07-07): each band is
    levelled toward a fixed internal target — quiet material is boosted (upward)
    and loud material is cut (downward) — the aggressive, dense "OTT" sound
    (modern pop/EDM vocals, synths, drum busses). `depth` is the one knob that
    matters (0 = off, 1 = full squash); `time` scales the attack/release
    ballistics (< 1 = snappier); per-band `up`/`down` rebalance the two sides;
    `in_gain_db` drives more (or less) material into the target, `out_gain_db`
    trims the result. Three real LR4 bands, the same in the render and the live
    monitor. Stems + busses for the agent; master is a user-add
    (upward compression pushes loudness — guardrails reject agent OTT there)."""
    type: Literal["ott"] = "ott"
    stem_id: str  # a stem id or a bus id ("master" = user-add only)
    depth: float = Field(default=0.5, ge=0.0, le=1.0)
    time: float = Field(default=1.0, ge=0.1, le=4.0)
    in_gain_db: float = Field(default=0.0, ge=-12.0, le=12.0)
    out_gain_db: float = Field(default=0.0, ge=-12.0, le=12.0)
    low: OttBand = Field(default_factory=OttBand)
    mid: OttBand = Field(default_factory=OttBand)
    high: OttBand = Field(default_factory=OttBand)
    crossover_lo_hz: float = Field(default=150.0, ge=30.0, le=4000.0)
    crossover_hi_hz: float = Field(default=2000.0, ge=200.0, le=16000.0)

    @model_validator(mode="after")
    def _check_crossovers(self) -> "OttAction":
        if self.crossover_lo_hz >= self.crossover_hi_hz:
            raise ValueError("crossover_lo_hz must be below crossover_hi_hz")
        return self


# The oversampling factors a memoryless shaper may run at. 1 keeps the original
# elementwise path (bit-identical to every render made before the option existed);
# above 1 the shaper runs at N x the sample rate, so the harmonics it generates above
# Nyquist stop folding back into the audible band as inharmonic aliasing. Shared by
# clip / saturation / overdrive — the three hard nonlinearities in the palette.
def _validate_oversample(v: int) -> int:
    if v not in (1, 2, 4, 8):
        raise ValueError("oversample must be 1, 2, 4 or 8")
    return v


class OverdriveAction(BaseModel):
    """Overdrive distortion (2026-07-07): the classic drive-pedal topology — an
    optional resonant band-pass pre-filter (`filter_on` + `freq_hz`/`q`: focus
    the drive on a band, Tube-Screamer style), an exponential soft-clipper at
    `drive_db` (memoryless, unity small-signal — grit without a loudness cheat),
    a `tone` tilt on the shaped signal (0 dark / 0.5 neutral / 1 bright), a post
    output trim, then dry/wet `mix`. Harder and dirtier than `saturation` (whose
    curves stay polite) — reach for it for aggressive color on a stem or bus.
    `oversample` (1/2/4/8) runs the soft-clipper at N x the sample rate — the grit
    stays harmonic instead of folding back as aliasing.
    The tone is a high shelf, corner near 3 kHz: tone 1 lifts the top up to +6 dB, and that
    lift is charged to the EQ boost budget. Deterministic; the same law in the native monitor
    (at oversample 1 sample-identical, above it within the oversamplers' difference)."""
    type: Literal["overdrive"] = "overdrive"
    stem_id: str
    drive_db: float = Field(default=12.0, ge=0.0, le=36.0)
    filter_on: bool = False
    freq_hz: float = Field(default=800.0, ge=40.0, le=12000.0)
    q: float = Field(default=0.7, ge=0.3, le=8.0)
    tone: float = Field(default=0.5, ge=0.0, le=1.0)
    output_db: float = Field(default=0.0, ge=-24.0, le=12.0)
    mix: float = Field(default=1.0, ge=0.0, le=1.0)
    # Overdrive is the dirtiest curve in the palette, so it aliases the most: at heavy
    # drive a bright source folds harmonics back as inharmonic grit. 4 is the useful
    # setting; 1 (default) keeps older renders bit-identical.
    oversample: int = Field(default=1)

    @field_validator("oversample")
    @classmethod
    def _os_pow2(cls, v: int) -> int:
        return _validate_oversample(v)


class BandWidthAction(BaseModel):
    """Per-band stereo width (Phase 15): the band's side scaled, the band split out at its edges
    by a 48 dB/oct Linkwitz-Riley crossover pair (one law in both engines since 2026-10-21) - width
    0 takes a one-octave band's side about 11-17 dB down, deeper for a wider band. Stems only;
    no-op on mono. For multiband width work prefer the imager (movable
    crossovers, all three bands in one device). 2026-07-19: optional
    ``lo_hz``/``hi_hz`` place the band freely (both or neither, lo < hi) —
    they override the label, mirroring the DAW's free band scrub."""
    type: Literal["band_width"] = "band_width"
    stem_id: str
    band: Literal["sub", "bass", "low_mid", "mid", "high_mid", "presence",
                  "brilliance", "air"]
    width: float = Field(ge=0.0, le=2.0)
    lo_hz: Optional[float] = Field(default=None, ge=20.0, le=20000.0)
    hi_hz: Optional[float] = Field(default=None, ge=20.0, le=20000.0)

    @model_validator(mode="after")
    def _custom_edges(self) -> "BandWidthAction":
        if (self.lo_hz is None) != (self.hi_hz is None):
            raise ValueError("band_width: lo_hz and hi_hz come together")
        if self.lo_hz is not None and self.hi_hz is not None \
                and self.lo_hz >= self.hi_hz:
            raise ValueError("band_width: lo_hz must be below hi_hz")
        return self


class ImagerAction(BaseModel):
    """Multiband stereo imager (2026-07-05 — the band_width upgrade, Ozone-Imager
    class): the signal splits into three bands at movable crossovers (a real
    48 dB/oct Linkwitz-Riley split, one law in both engines since 2026-10-21 -
    width_low 0 takes the low side ~40 dB down below 2/3 of the crossover, 6 dB at
    it), each band's stereo width is M/S-scaled independently (0 = mono the band,
    1 = untouched, 2 = double the side), and the bands sum back. The classic
    mastering moves: mono the lows (width_low 0 for a tight, club-safe bottom),
    widen the highs (width_high 1.2-1.4 for air) — without touching the mids.
    Stems, busses and master; no-op on mono; one per chain (a new one replaces
    it). All-1.0 widths = the split's allpass: every level the input's."""
    type: Literal["imager"] = "imager"
    stem_id: str
    width_low: float = Field(default=1.0, ge=0.0, le=2.0)
    width_mid: float = Field(default=1.0, ge=0.0, le=2.0)
    width_high: float = Field(default=1.0, ge=0.0, le=2.0)
    crossover_lo_hz: float = Field(default=150.0, ge=30.0, le=4000.0)
    crossover_hi_hz: float = Field(default=2000.0, ge=200.0, le=16000.0)

    @model_validator(mode="after")
    def _check_crossovers(self) -> "ImagerAction":
        if self.crossover_lo_hz >= self.crossover_hi_hz:
            raise ValueError("crossover_lo_hz must be below crossover_hi_hz")
        return self


# Limiter Style presets — (release_ms, lookahead_ms). A Style is a knob preset, NOT a
# hidden DSP scaler; keep in sync with the native editor button (NativeEditor limStyleBtn).
LIMITER_STYLES = ("modern", "transparent", "punchy")   # style index 0..2 -> name


def _index_of_name(value: str, names: tuple, what: str) -> int:
    """The index of a NAME the agent gave, in any case. A name that is not one of them is
    refused with the ones that are (reader D9·R11): `.get(name, 0)` read "loud" as Modern and
    "fuzz" as tanh, and the move was applied and reported as what was asked."""
    name = value.strip().lower()
    if name not in names:
        raise ValueError(f"unknown {what} {value!r} - it is one of: {', '.join(names)} "
                         f"(or its index 0..{len(names) - 1})")
    return names.index(name)


LIMITER_STYLE_PRESETS = {0: (50.0, 3.0),    # Modern — balanced default
                         1: (120.0, 3.0),   # Transparent — slow release, clean
                         2: (20.0, 1.5)}     # Punchy — fast release, tight transients


class LimiterAction(BaseModel):
    """Master-bus true-peak limiter (Phase 6). The only processor allowed to
    push loudness: optional LUFS makeup toward target_lufs_i, then a brickwall
    at ceiling_dbtp. Master-only (session rejects it on stems); one limiter per
    bus — a new limiter action replaces the existing one."""
    type: Literal["limiter"] = "limiter"
    stem_id: str  # must be "master"
    ceiling_dbtp: float = Field(default=-1.0, ge=-6.0, le=-1.0)
    target_lufs_i: Optional[float] = Field(default=None, ge=-24.0, le=-6.0)
    # Phase 44 advanced ballistics (character only — the ceiling is still guaranteed):
    # release_ms = recovery speed (pumping vs loudness); lookahead_ms = peak-catch window;
    # channel_link = 1 stereo-linked / 0 independent L-R; style = modern/transparent/punchy.
    release_ms: Optional[float] = Field(default=None, ge=1.0, le=1000.0)
    lookahead_ms: Optional[float] = Field(default=None, ge=0.5, le=3.0)
    channel_link: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    style: Optional[int] = Field(default=None, ge=0, le=2)  # 0 modern / 1 transparent / 2 punchy

    @field_validator("style", mode="before")
    @classmethod
    def _style_name(cls, v: Any) -> Any:
        if isinstance(v, str):   # accept the names too (the agent speaks words)
            return _index_of_name(v, LIMITER_STYLES, "limiter style")
        return v

    @model_validator(mode="after")
    def _apply_style_preset(self) -> "LimiterAction":
        # Style is a PRESET that fills release_ms / lookahead_ms (like the editor button
        # moving the knobs), not a hidden DSP scaler. Explicit values win.
        if self.style is not None:
            rel, look = LIMITER_STYLE_PRESETS[self.style]
            if self.release_ms is None:
                self.release_ms = rel
            if self.lookahead_ms is None:
                self.lookahead_ms = look
        return self


SAT_CURVES = ("tanh", "warm", "tube", "hard")   # curve index 0..3 -> name


def _curve_to_index(v: Any) -> Any:
    """Accept a saturation curve as a name (tanh/warm/tube/hard) or an int 0..3."""
    if isinstance(v, str):
        return _index_of_name(v, SAT_CURVES, "saturation curve")
    return v


class SatBand(BaseModel):
    """One band of the multiband saturator (Phase 45): its own drive / output / dry-wet
    mix and optionally its own curve index 0..3 (None = the device's global curve)."""
    drive_db: float = Field(default=0.0, ge=0.0, le=24.0)
    output_db: float = Field(default=0.0, ge=-12.0, le=12.0)
    mix: float = Field(default=1.0, ge=0.0, le=1.0)
    curve: Optional[int] = Field(default=None, ge=0, le=3)

    @field_validator("curve", mode="before")
    @classmethod
    def _band_curve(cls, v: Any) -> Any:
        return _curve_to_index(v) if v is not None else None


class SaturationAction(BaseModel):
    """Deterministic waveshaper saturator (Phase 7; Phase 45 multiband). Single-band:
    y = waveshape(k·x)/k, k = drive — unity for small signals, peaks compressed; `curve`
    picks the model (tanh / warm / tube / hard); `output_db` is a post-shaper output trim.
    Multiband (Saturn-style): give a `bands` dict (low/mid/high), each with its own
    drive/output/mix(/curve), split at movable crossovers and summed, then a global `mix`.
    The flat drive_db/output_db are ignored when `bands` is set. Memoryless + deterministic.
    `oversample` (1/2/4/8, single-band only) runs the curve at N x the sample rate so its
    harmonics stop folding back as aliasing; 1 (default) is bit-identical to older renders."""
    type: Literal["saturation"] = "saturation"
    stem_id: str
    drive_db: float = Field(default=3.0, ge=1.0, le=12.0)   # single-band; ignored when bands are set
    output_db: float = Field(default=0.0, ge=-12.0, le=12.0)
    curve: int = Field(default=0, ge=0, le=3)   # global saturation model (0 tanh/1 warm/2 tube/3 hard)

    @field_validator("curve", mode="before")
    @classmethod
    def _global_curve(cls, v: Any) -> Any:
        return _curve_to_index(v)
    # Multiband (Phase 45): independent low/mid/high bands (top-level, like the MB comp).
    # Give all three for a Saturn-style 3-band saturator; the flat params above are then
    # ignored. Per-band drive/output/mix(/curve) + movable crossovers + global dry/wet.
    low: Optional[SatBand] = None
    mid: Optional[SatBand] = None
    high: Optional[SatBand] = None
    crossover_lo_hz: float = Field(default=150.0, ge=30.0, le=4000.0)
    crossover_hi_hz: float = Field(default=2000.0, ge=200.0, le=16000.0)
    mix: float = Field(default=1.0, ge=0.0, le=1.0)           # global dry/wet
    # SINGLE-BAND only: the multiband path splits with an FFT and shapes each band
    # separately, and is not oversampled (2026-09-06) — set it on the flat saturator.
    oversample: int = Field(default=1)

    @field_validator("oversample")
    @classmethod
    def _os_pow2(cls, v: int) -> int:
        return _validate_oversample(v)

    @model_validator(mode="after")
    def _check_bands(self) -> "SaturationAction":
        present = [b for b in (self.low, self.mid, self.high) if b is not None]
        if present and len(present) != 3:
            raise ValueError("multiband saturation needs all of low / mid / high")
        if present and self.crossover_lo_hz >= self.crossover_hi_hz:
            raise ValueError("crossover_lo_hz must be below crossover_hi_hz")
        return self


class ChorusAction(BaseModel):
    """Chorus / ensemble (2026-09-06) — the modulation family's first device.

    One to three voices read the signal through a delay swept around `delay_ms` by `depth`
    (`delay_ms * (1 + depth*sin)`), spread evenly around the LFO cycle, with the right channel
    offset by `stereo_deg` — that offset is what makes it WIDE rather than merely detuned, and
    it is the reason chorus is the standard fix for a thin mono synth, a lifeless doubled
    vocal, or a guitar that needs to sit wider without EQ.

    There is deliberately **no feedback**: feedback through a short swept delay is a FLANGER,
    a separate device. Its absence also means every voice reads the dry input, so the LFO's
    phase is a pure function of the host timeline and the render is reproducible.

    Subtle is the point: `mix` around 0.2-0.4 thickens, 1.0 is an effect. Mono sources gain
    the most. Not a tone or loudness tool — it will comb-filter a source if pushed."""
    type: Literal["chorus"] = "chorus"
    stem_id: str
    rate_hz: float = Field(default=0.8, ge=0.01, le=10.0)     # sweep speed
    depth: float = Field(default=0.25, ge=0.0, le=1.0)        # fraction of delay_ms swung
    delay_ms: float = Field(default=20.0, ge=1.0, le=50.0)    # the delay swept AROUND
    mix: float = Field(default=0.5, ge=0.0, le=1.0)           # dry/wet
    voices: int = Field(default=1, ge=1, le=3)                # 2-3 = ensemble
    stereo_deg: float = Field(default=90.0, ge=0.0, le=180.0)  # L/R LFO offset = the width


class FlangerAction(BaseModel):
    """Flanger (2026-09-06) — the modulation family's second device.

    One voice reads the signal through a delay swept around `delay_ms` by `depth`, with the
    right channel offset by `stereo_deg` — so far, a chorus. What makes it a flanger is
    `feedback`: the line recirculates its own output, which turns the comb's gentle dips into
    resonant teeth and produces the jet-plane sweep. The sign matters and is not cosmetic —
    POSITIVE resonates peaks at multiples of 1/delay_ms and puts the first NOTCH between them
    at 1/(2*delay_ms); NEGATIVE puts the notches on the multiples themselves, so the first one
    lands an octave HIGHER and the whole picture is the other half of the comb - but only with
    mix above (1-fb^2)/(1-fb^2-fb), about 0.5; below that the notches sit between the
    multiples, as for positive.

    THE ONE TRAP, and it sits on the defaults: dry + wet with NEGATIVE feedback approaches a
    Schroeder allpass, whose magnitude response is FLAT. At mix 0.5 with feedback -0.62 the
    comb is 0.1 dB deep — the device is inaudible while every knob reads like it is working.
    Negative feedback is deep either side of that mix: at 0.85-1.0 (notches on the multiples;
    25.6 dB of teeth at -0.9, mix 1) or at 0.2-0.45 (notches between; 19.9 dB at -0.62, mix 0.3)
    - the agent's mix cap of 0.5 leaves it only the second. POSITIVE feedback is deepest around
    mix 0.5-0.7. With no feedback at all, mix 0.5 is the
    total cancellation everyone means by "comb filter".

    Where it earns its place: a flat DI guitar or synth that needs movement without a new
    part, a drum loop that needs a hook on the last bar, an effect-return that wants to be
    the interesting thing. It is NOT a widener (that is the chorus, or `band_width`) and it
    is NOT a tone tool — a comb is a comb, and at `mix` 0.5 it removes energy at every notch,
    which on a full mix reads as phasiness. Keep it on a stem.

    `delay_ms` is much shorter than the chorus's: 0.5-5 ms is the flanger's range (its notches
    then sit in the mids and above, where they sweep audibly). Above ~10 ms the teeth crowd
    into the low end and it stops sounding like a flanger and starts sounding broken."""
    type: Literal["flanger"] = "flanger"
    stem_id: str
    rate_hz: float = Field(default=0.5, ge=0.01, le=10.0)      # sweep speed
    depth: float = Field(default=0.6, ge=0.0, le=1.0)          # fraction of delay_ms swung
    delay_ms: float = Field(default=3.0, ge=0.5, le=20.0)      # the delay swept AROUND
    feedback: float = Field(default=0.0, ge=-0.95, le=0.95)    # the return path; sign = polarity
    mix: float = Field(default=0.5, ge=0.0, le=1.0)            # dry/wet
    stereo_deg: float = Field(default=0.0, ge=0.0, le=180.0)   # L/R LFO offset = the width


# The Auto Filter's word values (Phase 63). Each is stored as its INDEX (deviceValueNames on the
# native side reads the same names); the agent may write the name or the index.
AUTO_FILTER_TYPES = ("lp", "hp", "bp", "notch")
AUTO_FILTER_SHAPES = ("sine", "triangle", "saw_up", "saw_down", "square", "random")
# A synced LFO's period, as the note value it lasts, in QUARTER NOTES (so a 6/8 bar or the
# engine's beat convention cannot change what "1/4" means). Ordered by length: the knob walks it.
AUTO_FILTER_NOTES = (("1/32", 0.125), ("1/16T", 1.0 / 6.0), ("1/16", 0.25), ("1/8T", 1.0 / 3.0),
                     ("1/16D", 0.375), ("1/8", 0.5), ("1/4T", 2.0 / 3.0), ("1/8D", 0.75),
                     ("1/4", 1.0), ("1/2T", 4.0 / 3.0), ("1/4D", 1.5), ("1/2", 2.0),
                     ("1/2D", 3.0), ("1/1", 4.0), ("2/1", 8.0), ("4/1", 16.0), ("8/1", 32.0))
AUTO_FILTER_NOTE_NAMES = tuple(name for name, _ in AUTO_FILTER_NOTES)
AUTO_FILTER_SLOPES = (12, 24)


class AutoFilterAction(BaseModel):
    """Auto Filter (Phase 63) — the modulation family's third device: a resonant filter whose
    cutoff MOVES.

    `filter_type` is low-pass, high-pass, band-pass (its peak at 0 dB) or notch, at 12 or 24
    dB/oct (`slope_db_oct`; 24 is a Butterworth section and then the resonant one, so the
    resonance is one peak, not two stacked). `resonance` 0..1 is the peak's sharpness - at 0
    the response is flat up to the cutoff; it rises toward a whistle at 1.

    Three things push the cutoff, in OCTAVES around `cutoff_hz`, and they add:
    * the LFO, `lfo_amount_oct` either side, in `lfo_shape` (sine, triangle, saw_up, saw_down,
      square, random - the last a new random level each cycle), at `lfo_rate_hz` or, with
      `lfo_sync` on, once per `lfo_rate_note` locked to the song's tempo map ("1/4", "1/8T",
      "1/16D", "1/1" = a bar of 4/4, up to "8/1"); `lfo_phase_deg` turns where the cycle starts
      and `lfo_stereo_deg` offsets the right channel, which is what makes a sweep move across
      the stereo field;
    * the envelope, `env_amount_oct` (negative closes the filter on a hit), following the
      level of the device's own input - or, when `key_stem_id` names another track, THAT
      track's level (a sidechain: the hi-hats opening the pad's filter). A track has ONE
      trigger: a Duck on it and this device share it.

    Where it earns its place: movement on a static part (a pad that breathes with the beat, a
    synth that opens over a bar), a filtered intro or breakdown, a wah on a guitar, a band-pass
    "telephone" on a vocal. It is a CUT and a moving one - not a tone fix (that is the EQ), and a
    deep LFO on a full mix reads as a mistake. Keep it on a part or a group."""
    type: Literal["auto_filter"] = "auto_filter"
    stem_id: str
    filter_type: int = Field(default=0, ge=0, le=3)            # lp / hp / bp / notch
    slope_db_oct: int = Field(default=12, ge=12, le=24)        # 12 or 24
    cutoff_hz: float = Field(default=1000.0, ge=20.0, le=20000.0)
    resonance: float = Field(default=0.3, ge=0.0, le=1.0)
    lfo_amount_oct: float = Field(default=1.0, ge=0.0, le=6.0)
    lfo_shape: int = Field(default=0, ge=0, le=5)
    lfo_sync: int = Field(default=0, ge=0, le=1)
    lfo_rate_hz: float = Field(default=0.5, ge=0.01, le=20.0)
    lfo_rate_note: int = Field(default=8, ge=0, le=16)         # "1/4"
    lfo_phase_deg: float = Field(default=0.0, ge=0.0, le=360.0)
    lfo_stereo_deg: float = Field(default=0.0, ge=0.0, le=180.0)
    env_amount_oct: float = Field(default=0.0, ge=-6.0, le=6.0)
    env_attack_ms: float = Field(default=10.0, ge=0.1, le=500.0)
    env_release_ms: float = Field(default=200.0, ge=10.0, le=3000.0)
    key_stem_id: str = ""                                      # "" = the device's own input
    mix: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("filter_type", mode="before")
    @classmethod
    def _type_name(cls, v: Any) -> Any:
        return _index_of_name(v, AUTO_FILTER_TYPES, "filter type") if isinstance(v, str) else v

    @field_validator("lfo_shape", mode="before")
    @classmethod
    def _shape_name(cls, v: Any) -> Any:
        return _index_of_name(v, AUTO_FILTER_SHAPES, "LFO shape") if isinstance(v, str) else v

    @field_validator("lfo_rate_note", mode="before")
    @classmethod
    def _note_name(cls, v: Any) -> Any:
        if isinstance(v, str):
            name = v.strip().upper().replace(" ", "")
            names = tuple(n.upper() for n in AUTO_FILTER_NOTE_NAMES)
            if name not in names:
                raise ValueError(f"unknown note value {v!r} - it is one of: "
                                 f"{', '.join(AUTO_FILTER_NOTE_NAMES)} (or its index 0..16)")
            return names.index(name)
        return v

    @field_validator("lfo_sync", mode="before")
    @classmethod
    def _sync_word(cls, v: Any) -> Any:
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, str):
            return _index_of_name(v, ("off", "on"), "sync")
        return v

    @field_validator("slope_db_oct", mode="before")
    @classmethod
    def _slope(cls, v: Any) -> Any:
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v not in AUTO_FILTER_SLOPES:
            raise ValueError(f"slope_db_oct is 12 or 24, not {v}")
        return v


class ClipAction(BaseModel):
    """Pre-limiter clipper (Phase 41; v2 2026-07-05 adds oversampling): shave transient
    peaks before the limiter so the master gets louder and cleaner (the modern loudness
    chain). Deterministic waveshaper from a soft tanh knee (`hardness` 0) to a hard
    brickwall (`hardness` 1) at `threshold_db`. `oversample` (1/2/4/8) runs the shaper
    at N× the sample rate — far less inter-sample overshoot and aliasing (a short
    linear-phase filter), though not a ceiling of its own: the filter rings past the
    threshold, and the master's limiter is what holds the true peak; 1 (default) is the
    plain memoryless path. Legal on the MASTER, a STEM (Phase 43 device freedom) or a
    group BUS (2026-09-08); on the master it runs just before the limiter. Unlike saturation it
    flattens peaks, not warms them."""
    type: Literal["clip"] = "clip"
    stem_id: str  # "master", a stem, or a group bus
    threshold_db: float = Field(default=-3.0, ge=-12.0, le=0.0)
    hardness: float = Field(default=0.5, ge=0.0, le=1.0)
    # Post-clip output trim (level makeup), applied last as a plain gain. Negative
    # backs the clipped signal off the limiter; positive drives it harder (the true
    # peak still bounded by the limiter ceiling downstream).
    output_db: float = Field(default=0.0, ge=-12.0, le=12.0)
    oversample: int = Field(default=1)  # 1 = off (bit-exact legacy path)

    @field_validator("oversample")
    @classmethod
    def _oversample_pow2(cls, v: int) -> int:
        return _validate_oversample(v)


class TiltAction(BaseModel):
    """One-knob tilt EQ (Phase 41): tilt the whole spectrum around a pivot — positive
    `tilt_db` brightens (lows down, highs up), negative warms (lows up, highs down).
    Implemented as two gentle shelves at `pivot_hz` (low shelf −tilt, high shelf +tilt).
    A broad tonal-balance staple; master / stem / bus. Cleaner + smoother than asking
    for two coordinated eq shelves."""
    type: Literal["tilt"] = "tilt"
    stem_id: str
    tilt_db: float = Field(default=0.0, ge=-12.0, le=12.0)
    pivot_hz: float = Field(default=1000.0, ge=200.0, le=8000.0)
    # 2026-07-12 gain staging: post-tilt output trim (a tilted spectrum often lands
    # louder/quieter) — level-compensate on the device, not with a Utility after it.
    output_db: float = Field(default=0.0, ge=-12.0, le=12.0)


class GateAction(BaseModel):
    """Downward gate / expander (Phase 41; v2 adds hold/range/hysteresis + detector
    filters): attenuate the signal when it falls BELOW `threshold_db` — silence
    noise/bleed between hits, tighten drums, control breaths. `ratio` sets the
    expansion steepness (1 = none, high = a hard gate); attack opens fast, release
    closes slowly. `hold_ms` keeps the gate open after the level drops (no chatter
    on decaying hits); `range_db` is the closed-gate floor (−80 ≈ full mute; −12
    "ducks" the bleed instead of chopping it); `hysteresis_db` drops the CLOSE
    threshold below the open threshold (no flutter at the threshold). The sidechain
    filters shape only the DETECTOR (the audio path is untouched): `sc_hpf_hz` > 0
    keys the gate off highs (e.g. ignore kick bleed in a snare mic), `sc_lpf_hz`
    < 20000 keys it off lows. Stems, busses and the master (gating a full mix is rarely wanted).
    Distinct from spectral_denoise, which is broadband artifact repair, not dynamics."""
    type: Literal["gate"] = "gate"
    stem_id: str
    threshold_db: float = Field(default=-40.0, ge=-80.0, le=0.0)
    ratio: float = Field(default=4.0, ge=1.0, le=20.0)
    attack_ms: float = Field(default=1.0, ge=0.1, le=100.0)
    release_ms: float = Field(default=100.0, ge=5.0, le=2000.0)
    hold_ms: float = Field(default=0.0, ge=0.0, le=500.0)
    range_db: float = Field(default=-80.0, ge=-80.0, le=0.0)
    hysteresis_db: float = Field(default=0.0, ge=0.0, le=24.0)
    sc_hpf_hz: float = Field(default=0.0, ge=0.0, le=2000.0)      # 0 = off
    sc_lpf_hz: float = Field(default=20000.0, ge=200.0, le=20000.0)  # 20000 = off


class TransientAction(BaseModel):
    """Attack/sustain shaper (Phase 7; v2 2026-07-05 adds speed + output trim):
    how far a fast level follower sits above a slow one (the transientness)
    blends between the attack and sustain gains. At least one must be non-zero.
    `speed` scales the DETECTOR time constants — > 1 = faster envelopes (a
    tighter transient window; fast percussion), < 1 = slower (a longer window;
    slower material like plucked bass or piano); 1.0 (default) is the original
    detector.
    `output_db` is a plain post-shaper level trim (positive draws from the same
    gain budget as gain moves)."""
    type: Literal["transient"] = "transient"
    stem_id: str
    attack_db: float = Field(default=0.0, ge=-6.0, le=6.0)
    sustain_db: float = Field(default=0.0, ge=-6.0, le=6.0)
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    output_db: float = Field(default=0.0, ge=-12.0, le=12.0)

    @model_validator(mode="after")
    def _check_not_noop(self) -> "TransientAction":
        if self.attack_db == 0.0 and self.sustain_db == 0.0:
            raise ValueError("transient shaper with attack_db=sustain_db=0 is a no-op")
        return self


class SpectralDenoiseAction(BaseModel):
    """Spectral-gate de-noise / de-bleed (Phase 29) — the repair processor for
    separation artifacts (musical noise, inter-stem leakage). Per-frequency
    soft gate: bins within `sensitivity_db` of their own estimated noise floor
    are attenuated by up to `reduction_db`; phase is taken from the input.
    Stems only (the session refuses it on the master/busses — de-noising a full
    mix is destructive); one per stem (a new one replaces the old)."""
    type: Literal["spectral_denoise"] = "spectral_denoise"
    stem_id: str
    reduction_db: float = Field(ge=1.0, le=24.0)
    sensitivity_db: float = Field(default=6.0, ge=0.0, le=24.0)


class TuneAction(BaseModel):
    """Uniform tuning correction (Phase 31): one deterministic pitch shift
    (Pedalboard PitchShift / Rubber Band) of the whole stem by `cents`, to pull
    a flat or sharp source onto concert pitch (A440 / equal temperament). Stems
    only (the session refuses it on the master/busses — globally retuning a full
    mix is a transpose, not a fix); one per stem (a new one replaces the old);
    whole-file only. Bounded to ±100 cents (a semitone): this *tunes* a source
    that drifted off pitch, it does not transpose. Per-note correction is parked
    — read `stem(<id>).tuning_offset_cents` to size the move."""
    type: Literal["tune"] = "tune"
    stem_id: str
    cents: float = Field(ge=-100, le=100)


class TransposeAction(BaseModel):
    """Whole-song key change (Phase 32): transpose the master mix by an integer
    number of semitones — the interval that moves the song's detected tonic onto
    a target. Master only (`stem_id` must be "master"); one per master (a new one
    replaces the old); whole-file; first in the master chain so the limiter still
    guarantees the ceiling on the transposed audio. Preserves harmony + mode (a
    uniform shift can't turn major into minor — that's per-note autotune, parked).
    Deterministic via the same Rubber Band PitchShift as Phase 31's `tune`; this
    is the macro transpose, not the ±100¢ correction."""
    type: Literal["transpose"] = "transpose"
    stem_id: str          # must be "master"
    semitones: int = Field(ge=-12, le=12)

    @model_validator(mode="after")
    def _check_not_noop(self) -> "TransposeAction":
        if self.semitones == 0:
            raise ValueError("transpose of 0 semitones is a no-op")
        return self


class SidechainDuckAction(BaseModel):
    """Sidechain duck (Phase 7): the key stem's level pushes this stem down —
    the kick→bass classic. Key renders without its own ducks (no cycles)."""
    type: Literal["duck"] = "duck"
    stem_id: str       # the stem being ducked
    key_stem_id: str   # the stem whose level drives the duck
    threshold_db: float = Field(ge=-60.0, le=0.0)
    depth_db: float = Field(ge=1.0, le=12.0)  # max gain reduction
    attack_ms: float = Field(default=5.0, ge=1.0, le=100.0)
    release_ms: float = Field(default=150.0, ge=20.0, le=1000.0)


class ReturnReverbAction(BaseModel):
    """Reverb. TWO placements, one processor (2026-07-20 — "no restrictions"):
    as a RETURN (`return_id`, the Phase 38 form — sources feed it with SendAction,
    parallel to the dry mix) or as a plain INSERT on a track/bus/master (`stem_id`),
    where `mix` sets the dry/wet balance like Live's Reverb. Deterministic
    Schroeder-style tail in dsp.py. Set exactly one of return_id / stem_id."""
    type: Literal["return_reverb"] = "return_reverb"
    return_id: str = ""
    stem_id: str = ""     # insert placement: stem / bus id / "master" (mutually exclusive with return_id)
    decay_s: float = Field(default=1.8, ge=0.2, le=12.0)
    pre_delay_ms: float = Field(default=20.0, ge=0.0, le=250.0)
    tone: float = Field(default=0.5, ge=0.0, le=1.0)
    # Dry/wet. 1.0 = wet only — what a RETURN needs (the dry lives on the source tracks)
    # and the pre-2026-07-20 behaviour, so every existing session renders bit-identically.
    # An insert wants a blend; add_stem_device seeds INSERT_WET_MIX.
    mix: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_target(self) -> "ReturnReverbAction":
        _check_wet_target(self)
        return self


class ReturnConvolutionAction(BaseModel):
    """Create/update an agent-owned CONVOLUTION (IR) reverb return (Tier-1): the summed sends
    convolved with a procedural impulse response (decaying noise; a room has three early
    reflections, a hall a 20 ms onset, a plate is dense at once) - denser and smoother than the
    algorithmic ``return_reverb``. The true RT60 is ``decay_s`` x 0.55 room / 1.0 hall / 0.75
    plate; ``tone`` is a shelf on the whole tail. Wet-only, parallel to the dry
    mix; sources feed it with SendAction. Offline = scipy.fftconvolve; native = juce::dsp::
    Convolution on the same IR, made sample for sample alike by both engines.
    Placeable as a return (`return_id`) or as an insert (`stem_id`) — see ReturnReverbAction."""
    type: Literal["return_convolution"] = "return_convolution"
    return_id: str = ""
    stem_id: str = ""     # insert placement (mutually exclusive with return_id)
    ir: Literal["room", "hall", "plate"] = "hall"
    decay_s: float = Field(default=2.0, ge=0.2, le=8.0)
    pre_delay_ms: float = Field(default=10.0, ge=0.0, le=250.0)
    tone: float = Field(default=0.5, ge=0.0, le=1.0)  # a shelf on the whole tail: 0 dark .. 1 bright (the lows hold)
    mix: float = Field(default=1.0, ge=0.0, le=1.0)   # dry/wet; 1.0 = return form (unchanged)

    @model_validator(mode="after")
    def _check_target(self) -> "ReturnConvolutionAction":
        _check_wet_target(self)
        return self


class ReturnDelayAction(BaseModel):
    """Delay — deterministic, bounded feedback delay (Phase 38). Placeable as a return
    (`return_id`) or as an insert (`stem_id`) with a dry/wet `mix` — see ReturnReverbAction."""
    type: Literal["return_delay"] = "return_delay"
    return_id: str = ""
    stem_id: str = ""     # insert placement (mutually exclusive with return_id)
    delay_ms: float = Field(default=250.0, ge=20.0, le=2000.0)
    feedback: float = Field(default=0.25, ge=0.0, le=0.85)
    tone: float = Field(default=0.5, ge=0.0, le=1.0)
    mix: float = Field(default=1.0, ge=0.0, le=1.0)   # dry/wet; 1.0 = return form (unchanged)

    @model_validator(mode="after")
    def _check_target(self) -> "ReturnDelayAction":
        _check_wet_target(self)
        return self


class SendAction(BaseModel):
    """Route a stem or group bus to an agent-owned return (Phase 38). Absolute
    send level in dB, post source processing, parallel to the dry path."""
    type: Literal["send"] = "send"
    stem_id: str  # stem id or group bus id; never master/return
    return_id: str
    gain_db: float = Field(ge=-60.0, le=0.0)


class MuteAction(BaseModel):
    """Take a stem out of the mix (Phase 9). State, not monitoring: muted
    stems don't render into the master until un-muted. Undoable like any
    other proposal."""
    type: Literal["mute"] = "mute"
    stem_id: str
    muted: bool


class ExcludeAction(BaseModel):
    """Remove a stem from the project mix entirely (Phase 9) — the UI's
    "delete". The file stays on disk and the move is undoable; excluded stems
    render nowhere and the UI hides them."""
    type: Literal["exclude"] = "exclude"
    stem_id: str
    excluded: bool


class PanAction(BaseModel):
    type: Literal["pan"] = "pan"
    stem_id: str
    pan: float = Field(ge=-1.0, le=1.0)


class WidthAction(BaseModel):
    type: Literal["width"] = "width"
    stem_id: str
    width: float = Field(ge=0.0, le=2.0)


class SpectrogramAction(BaseModel):
    """The Spectrogram analyzer device (DAW, 2026-07-06): a pure ANALYZER chain entry —
    audio passes through untouched (the offline render and the native MonitorChain both
    treat it as a no-op step; the waterfall itself reads the track's LiveMeters history
    in-process). It exists as a chain entry so it persists, reorders, and undoes like
    every other device, and so the agent SEES it in the chain (total-agent-control)."""
    type: Literal["spectrogram"] = "spectrogram"
    stem_id: str  # a stem id, a bus id, "master", or (via return_id-less reuse) a return


class UtilityAction(BaseModel):
    """Ableton-style Utility (device freedom; v2 2026-07-05 = Live parity): a
    freely-placeable output/trim insert — DC filter, gain trim, per-channel
    polarity, stereo width (M/S), mono fold-down, bass mono, and balance — one
    chain entry on a stem / bus / master. `invert_left`/`invert_right` flip one
    channel (unset = inherit the legacy both-channel `invert`); `bass_mono`
    folds the band below `bass_mono_hz` to mono through a Linkwitz-Riley
    48 dB/oct split (the Live workflow: keep the lows centered, leave the top
    wide; an octave below the knob the side is -48 dB, at the knob -6); `dc_filter` is a gentle one-pole 5 Hz
    high-pass; `balance` is the output balance law (−1 hard left .. +1 hard
    right). Distinct from the channel-strip fader; all-default is a no-op. One
    per chain (the native MonitorChain hosts a single Utility step)."""
    type: Literal["utility"] = "utility"
    stem_id: str  # a stem id, a bus id, or "master"
    gain_db: float = Field(default=0.0, ge=-24.0, le=24.0)
    width: float = Field(default=1.0, ge=0.0, le=2.0)
    mono: bool = False
    invert: bool = False
    invert_left: Optional[bool] = None    # unset = inherit `invert`
    invert_right: Optional[bool] = None   # unset = inherit `invert`
    bass_mono: bool = False
    bass_mono_hz: float = Field(default=120.0, ge=50.0, le=500.0)
    dc_filter: bool = False
    balance: float = Field(default=0.0, ge=-1.0, le=1.0)


class GroupAction(BaseModel):
    """Phase 28 — the agent's routing primitive. Create a group bus (if it
    doesn't already exist) and route the named member stems into it, so the
    agent can glue related stems with one bus move instead of repeating the
    same processing per stem. Atomic + undoable like any proposal action.
    Routing is the agent's now — the user's manual channel is only
    volume/pan/mute/solo. `bus_id` shares the stem/'master' namespace;
    `members` must be real stems."""
    type: Literal["group"] = "group"
    bus_id: str
    members: list[str] = Field(min_length=1)


class RouteAction(BaseModel):
    """Phase 28 — re-home a single stem's output: into a group bus, or
    `bus_id=None` to send it straight to the master (the default). The
    companion to GroupAction for fine routing control."""
    type: Literal["route"] = "route"
    stem_id: str
    bus_id: Optional[str] = None


Action = Union[EqAction, CompAction, GainAction, DeessAction, PanAction,
               WidthAction, UtilityAction, SpectrogramAction, LimiterAction, SaturationAction, ClipAction,
               GateAction, TiltAction, TransientAction, SpectralDenoiseAction,
               TuneAction, TransposeAction,
               SidechainDuckAction, DynamicEqAction, ReturnReverbAction,
               ReturnConvolutionAction, ReturnDelayAction, SendAction, MuteAction,
               ExcludeAction, MatchEqAction, MultibandCompAction, OttAction,
               OverdriveAction, ChorusAction, FlangerAction, AutoFilterAction, BandWidthAction,
               ImagerAction, GroupAction,
               RouteAction]


class StemUserState(BaseModel):
    """Echo of a user edit (set_stem, Phase 9): the fields the user owns.
    Master edits return gain_db plus the session-global automation switch
    (Phase 27)."""
    stem_id: str
    gain_db: float
    pan: Optional[float] = None
    width: Optional[float] = None
    muted: Optional[bool] = None
    excluded: Optional[bool] = None
    # Phase 27: Ableton-style automation bypass — on a stem, that stem's
    # section moves; on "master", the session-global switch.
    automation_enabled: Optional[bool] = None


class StemCaptureState(BaseModel):
    """Echo of a capture-bookkeeping write (Phase 41): a stem's recorded host-
    timeline offset (seconds), set by the gateway at import so a sidechain duck
    aligns to its key stem even across separate captures."""
    stem_id: str
    capture_start_s: float


class MasterDeviceState(BaseModel):
    """Echo of a user co-edit (Phase 40, native VST): the master-chain device's
    new primary value after a direct (shared-control) edit. `value` is None for
    add/remove (no single value), set for a param edit."""
    type: str
    occurrence: int
    value: Optional[float] = None


class StemState(BaseModel):
    """One stem's render-relevant state (get_state, Phase 10). A cheap read
    straight from session state — no rendering, no analysis."""
    stem_id: str
    gain_db: float
    pan: float
    width: float
    muted: bool
    excluded: bool
    tags: list[str]
    rides: list[dict]
    applied_chain: list[dict]  # ordered processors, gain is entry 0
    bus: Optional[str] = None  # Phase 23: group bus this stem sums into
    # Phase 27: this stem's automation switch. Section moves (rides +
    # time-ranged entries) render only when this AND the session-global
    # switch are on; the entries themselves stay in state and in the chain.
    automation_enabled: bool = True


class BusState(BaseModel):
    """One group bus (Phase 23): member stems sum into it, its chain runs
    on the group, the result sums into the master. Render order between
    busses is sorted bus_id (determinism)."""
    bus_id: str
    gain_db: float
    # The group strip's own scalars (2026-09-05), read explicitly like a stem's rather
    # than only as the trailing entries of applied_chain — the agent must be able to READ
    # every control it can move (docs/10-daw.md, total agent control). Defaulted, so a
    # session written before they existed reads back as the neutral values it rendered at.
    pan: float = 0.0
    width: float = 1.0
    muted: bool = False
    members: list[str]  # sorted stem ids routed here (active or not)
    applied_chain: list[dict]  # ordered processors, gain is entry 0
    sends: dict[str, float] = {}  # Phase 38: return_id -> send gain dB


class ReturnState(BaseModel):
    """Send/return bus (Phase 38): sources feed it in parallel, then its wet
    chain sums into the master. Shared control — user + agent both dial sends."""
    return_id: str
    gain_db: float
    sources: dict[str, float]  # source id (stem/bus/"return:<id>") -> send gain dB
    applied_chain: list[dict]  # ordered processors, gain is entry 0
    pre_fader: bool = False    # send tap: Post (scales w/ fader) | Pre (post-devices, pre-fader)
    sends: dict[str, float] = {}  # cascading: this return -> a LATER return (forward-only)
    pan: float = 0.0           # the return strip's pan (2026-10-03): the wet, panned into the mix
    muted: bool = False        # a muted return is silent in the mixdown, like a muted stem


class SessionState(BaseModel):
    """get_state output (Phase 10): everything the planning loop needs
    between renders without paying for analysis."""
    stems: list[StemState]
    master_gain_db: float
    master_chain: list[dict]
    pass_number: int
    buses: list[BusState] = []  # Phase 23
    returns: list[ReturnState] = []  # Phase 38
    automation_enabled: bool = True  # Phase 27: the session-global switch
    # The composed-EQ rail's remaining room (design call 2026-09-11): target -> band ->
    # {boost >= 0, cut <= 0}, for every target the agent has EQ'd. A move past it is refused.
    eq_headroom_db: dict[str, dict[str, dict[str, float]]] = {}


# ---------- the agent's door: an action says exactly what it is (reader D9·R11) ----------
#
# The action models take what they know and DROP the rest, which is right for a stored chain
# entry (it carries `uid`, `enabled`, `applied_seq` and whatever a later build adds) and wrong
# for a request: `time_range` for `time_range_s` turned a section move into a whole-song one,
# `ceiling_db` set the limiter to its default ceiling, and an action with no `type` was taken
# for the first model whose fields all default - a Utility that does nothing. Each was
# accepted, applied and reported as what was asked.
#
# So the REQUEST models (`ProposalRequest`, `ProbeRequest`) check the raw actions before they
# are parsed. Not `extra="forbid"` on the action models: `Session._validated_chain` validates
# stored entries through the same union, and every copied device would be dropped; and it
# would put `additionalProperties: false` into the tool schema the model is sent. A record
# read back from session.json says so (`context={"stored": True}`) and is not checked.

def _action_models() -> dict[str, type[BaseModel]]:
    out: dict[str, type[BaseModel]] = {}
    for model in get_args(Action):
        for name in get_args(model.model_fields["type"].annotation):
            out[name] = model
    return out


def _nested_model(annotation: Any) -> Optional[type[BaseModel]]:
    """The model a field holds (`EqBand`, `Optional[SatBand]`), or None."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    for arg in get_args(annotation):
        if isinstance(arg, type) and issubclass(arg, BaseModel):
            return arg
    return None


def _unknown_fields(model: type[BaseModel], raw: dict, where: str) -> list[str]:
    import difflib

    problems: list[str] = []
    for key, value in raw.items():
        field = model.model_fields.get(key)
        if field is None:
            close = difflib.get_close_matches(str(key), list(model.model_fields), n=1, cutoff=0.6)
            hint = f" - did you mean '{close[0]}'?" if close else ""
            problems.append(f"unknown field {key!r} on {where}{hint} "
                            f"(its fields: {', '.join(model.model_fields)})")
        elif isinstance(value, dict) and (inner := _nested_model(field.annotation)) is not None:
            problems += _unknown_fields(inner, value, f"{where}, in '{key}'")
    return problems


# How many actions one proposal - or one probe - may hold. guardrails.LIMITS says the same number
# (`actions_per_proposal`; a test pins the two). It is said HERE, in a sentence, before the
# field's own bound can: the agent mixing the showcase mashup sent eleven and was shown "List
# should have at most 8 items after validation, not 11 [type=too_long, input_value=[...".
MAX_ACTIONS_PER_REQUEST = 8


def _refuse_loose_actions(data: Any, info: ValidationInfo) -> Any:
    if (info.context or {}).get("stored") or not isinstance(data, dict):
        return data
    actions = data.get("actions")
    if not isinstance(actions, list):
        return data
    if len(actions) > MAX_ACTIONS_PER_REQUEST:
        raise ValueError(f"at most {MAX_ACTIONS_PER_REQUEST} actions go in one request and this "
                         f"one has {len(actions)}: split it into two, each a move that makes "
                         "sense on its own")
    models = _action_models()
    problems: list[str] = []
    for index, raw in enumerate(actions):
        if not isinstance(raw, dict):
            continue                                    # an Action built in code: already exact
        kind = raw.get("type")
        if not isinstance(kind, str) or kind not in models:
            problems.append(f"action {index} needs a `type` that is one of: "
                            f"{', '.join(sorted(models))}" + (f" (got {kind!r})" if kind else ""))
            continue
        problems += _unknown_fields(models[kind], raw, f"action {index} ({kind})")
    if problems:
        raise ValueError("; ".join(problems))
    return data


# ---------- probes (Phase 12) ----------

class ProbeRequest(BaseModel):
    """probe_actions input: a what-if. Actions apply to a scratch copy of
    session state, named metrics are measured before/after over an optional
    time range, and everything is discarded. No proposal record and no
    guardrails — nothing persists. Deterministic: a probe's `after` equals
    the post-apply measurement of the same actions on the same range."""
    actions: list[Action] = Field(min_length=1, max_length=MAX_ACTIONS_PER_REQUEST)
    metrics: list[str] = Field(min_length=1, max_length=8)
    range_s: Optional[tuple[float, float]] = None  # None = whole file

    _exact_actions = model_validator(mode="before")(_refuse_loose_actions)

    @field_validator("range_s")
    @classmethod
    def _range_is_sane(cls, v):
        # Same rule as an action's time_range_s (_check_time_range): a negative start
        # used to become a NEGATIVE Python index — the probe measured the wrong region
        # silently — and an inverted or empty range produced a zero-length buffer whose
        # analysis raised and was swallowed into all-None probe deltas (2026-09-01 sweep).
        if v is None:
            return v
        start_s, end_s = v
        if start_s < 0.0 or end_s <= start_s:
            raise ValueError(f"range_s must be [start >= 0, end > start], got {v}")
        return v


class ProbeMeasurement(BaseModel):
    metric: str
    before: Optional[float]
    after: Optional[float]
    delta: Optional[float]  # after - before; None if either is unmeasurable


class ProbeResult(BaseModel):
    measurements: list[ProbeMeasurement]
    render_ms: int


# ---------- proposal lifecycle ----------

class MetricPrediction(BaseModel):
    metric: str            # e.g. "masking(vocals,synths).presence"
    predicted_delta: float
    unit: str              # "dB", "index", "LUFS"


class ProposalRequest(BaseModel):
    rationale: str
    predicted_impact: list[MetricPrediction]
    actions: list[Action] = Field(min_length=1, max_length=MAX_ACTIONS_PER_REQUEST)

    _exact_actions = model_validator(mode="before")(_refuse_loose_actions)


class Violation(BaseModel):
    action_index: int
    rule: str
    message: str
    limit: float
    value: float


class ProposalResult(BaseModel):
    accepted: bool
    proposal_id: Optional[str] = None
    violations: list[Violation] = []


class MetricMeasurement(BaseModel):
    metric: str
    measured_delta: float
    unit: str


class ProposalSummary(BaseModel):
    proposal_id: str
    status: Literal["proposed", "applied", "undone", "rejected"]
    rationale: str
    actions: list[Action]
    predicted_impact: list[MetricPrediction]
    measured_impact: Optional[list[MetricMeasurement]] = None  # filled in Phase 2
    pass_number: int
    # Phase 27: the engine's apply order (None until applied; kept after
    # undo). The UI ranks "what changed last" off this, not feed timing.
    applied_seq: Optional[int] = None


class ApplyResult(BaseModel):
    ok: bool
    proposal_id: str
    message: str = ""


class MasterReport(BaseModel):
    """What the master limiter did on this render: the UI's mastering
    readout. Legacy fields stay stable; Phase 34 adds direct limiter input /
    output loudness, GR telemetry and true-peak headroom."""
    pre_limiter_lufs_i: Optional[float] = None
    limiter_input_lufs_i: Optional[float] = None  # after LUFS makeup, before limiting
    post_limiter_lufs_i: Optional[float] = None   # after brickwall + TP trim
    limiter_makeup_db: float = 0.0  # gain applied toward target_lufs_i
    limiter_trim_db: float = 0.0    # extra reduction to honor the ceiling (<= 0)
    limiter_gr_mean_db: float = 0.0
    limiter_gr_max_db: float = 0.0
    limiter_gr_active_pct: float = 0.0
    true_peak_overshoot_db: float = 0.0  # TP excess after brickwall, before trim
    true_peak_margin_db: Optional[float] = None  # ceiling - final TP; >=0 is safe


class RenderResult(BaseModel):
    bounce_path: str
    render_ms: int
    analysis: SessionAnalysis
    master: Optional[MasterReport] = None  # present when the master chain ran
    flags: list[str] = []  # post-render guardrail flags, e.g. "master_true_peak_exceeded"
    # Stems whose FILE held NaN / inf samples: the render dropped them (as the DAW's node
    # does) and the `non_finite_input` flag is set (reader E5·R8).
    non_finite_stems: list[str] = []


class ExportRequest(BaseModel):
    """Phase 24, user channel: print deliverables into a folder the user
    picked. Never targets the project folder (exported audio would be
    re-imported as stems by the next refresh). flac/mp3 added in the
    Phase 26 format expansion."""
    out_dir: str
    format: Literal["wav24", "wav16", "wav32f", "flac", "mp3"] = "wav24"
    include_stems: bool = False  # also print active stems through their chains


class ExportedFile(BaseModel):
    path: str  # absolute
    kind: Literal["master", "stem"]
    stem_id: Optional[str] = None  # set when kind == "stem"
    duration_s: float
    peak_dbfs: Optional[float]  # sample peak of the float render; None = silence
    clipped: bool  # peak hit integer full scale before the PCM clip


class ExportResult(BaseModel):
    out_dir: str  # absolute, resolved
    format: Literal["wav24", "wav16", "wav32f", "flac", "mp3"]
    sample_rate_hz: int
    render_ms: int
    files: list[ExportedFile]
