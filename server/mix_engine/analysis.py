"""Metric computation. Definitions are contractual: .claude/skills/mix-analysis/SKILL.md.

If you change what a metric means, bump ANALYSIS_VERSION in models.py and note it
in the skill file. Keep this module pure (no caching, no I/O beyond reading audio
passed in by the caller).
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pyloudnorm as pyln
from scipy.signal import resample_poly, welch
from scipy.stats import rankdata

from . import instruments
from .models import (  # noqa: F401
    ActivityAnalysis,
    ActivitySpan,
    BAND_EDGES_HZ,
    BAND_LABELS,
    ChordAnalysis,
    ChordSpan,
    GrooveExtraction,
    KeyEstimate,
    MaskingAnalysis,
    MaskingWindow,
    NoteAnalysis,
    NotePlayed,
    NoteEvent,
    PhraseAnalysis,
    PhraseLevelMove,
    PhrasePitch,
    TimingAnalysis,
    ReferenceAnalysis,
    ReferenceSectionContribution,
    ReferenceSectionTarget,
    ReferenceSetAnalysis,
    ReferenceSetMember,
    Resonance,
    Section,
    SectionAnalysis,
    SeparationReport,
    SibilanceBurst,
    SessionAnalysis,
    StemActivity,
    StemAnalysis,
    StemArtifacts,
    TempoAnalysis,
                     RegionDescription, RegionEnvelopePoint, RegionMasker)

SHORT_TERM_WINDOW_S = 3.0
SHORT_TERM_HOP_S = 0.1
BAND_ENERGY_FLOOR_DB = -80.0
SILENCE_PEAK = 1e-9
MASKING_BLOCK_S = 0.4
MASKING_GATE_DBFS = -60.0
MASKING_EXCESS_CAP_DB = 40.0
MASKING_BAND_FLOOR_DB = -25.0  # ignore bands holding < this share of stem_a's power
MASKING_WORST_WINDOWS = 3
RMS_FRAME_S = 0.4
RMS_HOP_S = 0.1
# LRA per EBU R128 / Tech 3342 shape (exact parameters in mix-analysis skill)
LRA_WINDOW_S = 3.0
LRA_HOP_S = 1.0
LRA_ABS_GATE_LUFS = -70.0
LRA_REL_GATE_LU = -20.0
# Transient/sustain split + sibilance (mix-analysis skill)
TRANSIENT_WINDOW_S = 0.05
SIBILANCE_LO_HZ = 5000
SIBILANCE_HI_HZ = 10000
SIBILANCE_SHARE = 0.25
SIBILANCE_FRAME_S = 0.1
# Phrase-aware readout (Phase 35): compact local candidates for existing
# ranged gain and de-ess actions.
PHRASE_LEVEL_TOLERANCE_DB = 3.0
PHRASE_MAX_GAIN_DB = 3.0
PHRASE_MIN_SPAN_S = 0.5
PHRASE_MAX_MOVES = 4
# Timing / groove (2026-08-08). The onset hop is deliberately finer than the
# shared one: at 44.1 k, 128 samples = 2.9 ms, an order below the 5-30 ms leans
# this measures. 512 (the shared hop) would quantise the answer into noise.
TIMING_NPERSEG = 512
TIMING_HOP = 128
TIMING_MIN_ONSETS = 8           # below this a median is not a measurement
TIMING_PEAK_FLOOR = 0.10        # an onset is never a tenth of the loudest one
# The onset function on recorded drums (2026-10-09; _onset_flux has the story). Below
# ONSET_LOW_BAND_HZ a drum's energy is ONE tone in one or two bins, where a noise burst of the
# same energy is spread over hundreds: summed over bins, the tone counts for a tenth of the
# burst. The low band's own envelope is added back at this weight - sized on a click-less sub
# kick beside hats from 6 dB over it to 30 dB under (2 finds all of both; 0 loses the kick under
# loud hats, 4 loses the quiet hats) and on 79 recorded loops, where it adds 2 close pairs to 10.
ONSET_LOW_BAND_HZ = 300.0
ONSET_LOW_BAND_WEIGHT = 2.0
# A hit masks what is much weaker and just behind it: the bumps in a kick's own tail (14-20 % of
# its attack, 50-90 ms on) are not notes. A real pickup a 32nd before a beat is of like strength.
ONSET_MASK_S = 0.12
ONSET_MASK_RATIO = 0.3
ANALYTIC_BLOCK = 1 << 20        # the analytic signal is made a block at a time...
ANALYTIC_MARGIN = 1 << 15       # ...with this much real audio either side (0.7 s at 48 k)
# Candidates coarse-to-fine, including triplet grids: a heavily swung part IS a
# triplet grid, not sloppy 16ths.
TIMING_SUBDIVISIONS = (1, 2, 3, 4, 6, 8)
TIMING_FIT_FRACTION = 0.25      # "on this grid" = within 1/4 of its interval
TIMING_FIT_MIN_RATIO = 0.8      # …for at least this share of onsets
TIMING_SWING_MIN_ONSETS = 4     # per parity class, before quoting a swing figure
# Extract Groove (2026-08-08): the template spans ONE 4/4 bar of the detected
# subdivision, so a part's habit at each position is what survives the fold.
GROOVE_BEATS_PER_BAR = 4
GROOVE_SUBDIVISIONS = (1, 2, 4, 8)   # STRAIGHT only — see _choose_groove_grid
GROOVE_MIN_CELL_ONSETS = 1      # a cell needs a hit before it may claim a lean
GROOVE_MIN_CELL_COVERAGE = 0.5  # ...and half the bar must be played, or it is not a groove
# What may VOTE on a cell's lateness (2026-10-09; _choose_groove_grid): a feel, and nothing else.
GROOVE_PUSH_BEATS = 0.06        # a push or a drag: within this of the line (36 ms at 100 BPM)...
GROOVE_PUSH_CELLS = 0.25        # ...and never more than a quarter of the cell
GROOVE_SWING_CELLS = 0.42       # a swing: LATE by up to this, on an OFF cell (71 % swing)
GROOVE_MAX_EXTRAS = 0.25        # more onsets than this left out = the grid is too coarse
# Onset detection lands within ~1 ms, which is a lateness of ~0.01 — real, and
# musically nothing. Below this floor a part is quantized, and the honest answer
# is 'no groove here', not a template of measurement noise.
GROOVE_MIN_LATENESS = 0.05      # ~6 ms of shift on 1/8 cells at 120 BPM
# Repetition test (2026-08-08): median within-cell scatter, as a fraction of the
# scatter across the whole bar. A groove repeats bar to bar, so its cells agree
# with themselves; free playing does not. Deliberately PERMISSIVE — measured
# within/across is 0.10 for a machine-tight swung loop but 0.60 for the same
# loop with 20 ms of human jitter, which is a groove we must not reject. It is a
# coarse backstop, not a precise classifier: the grid adapts to loose material
# by choosing a finer subdivision, which absorbs jitter and flattens this ratio,
# so the primary guard against free playing remains the grid-uniqueness test in
# _choose_groove_grid.
GROOVE_MAX_CELL_SCATTER = 0.75

PHRASE_GAP_S = 0.3      # silence shorter than this is a breath, not a phrase end
PHRASE_TUNING_MIN_S = 1.0   # shorter phrases can't support a stable pitch estimate
PHRASE_MAX_PITCH = 6    # cap the intonation readout, worst deviation first
SIBILANCE_BURST_SHARE = 0.35
SIBILANCE_MAX_BURSTS = 4
# Separation artifact detection (Phase 29, mix-analysis skill)
HF_SHIMMER_LO_HZ = 4000.0           # brilliance+air: where separation musical noise sits
SHIMMER_QUIET_QUANTILE = 0.25       # bottom-quartile-RMS frames = the source's gaps
TRANSIENT_SMEAR_WINDOW_S = 0.03     # ±30 ms pre/post-onset windows
SPECTRAL_HOLE_CONTENT_FLOOR_DB = -20.0  # a band must hold ≥ this share to be judged
SPECTRAL_HOLE_DROP_DB = 12.0        # power this far under the band median = a hole
LEAKAGE_MIN_BLOCKS = 4
# Section detection (mix-analysis skill)
SECTION_MIN_LEN_S = 4.0
SECTION_NOVELTY_HALF_S = 2.0
SECTION_SAME_LABEL_DB = 6.0
# Reference-set matching (Phase 36): use the loudest detected section as the
# stable "energetic section" comparison point, then fall back to full-song
# aggregates when either side is uniform or too uncertain.
REFERENCE_SECTION_TARGETS_MAX = 1
# Composite perceptual metrics (Phase 14, mix-analysis skill v5)
HARSH_BANDS = ("presence", "brilliance")   # the 2–8 kHz aggression region
MUD_BAND = "low_mid"                       # the 150–400 Hz buildup region
CLARITY_BANDS = ("mid", "high_mid", "presence")
COMPOSITE_CREST_WEIGHT = 0.5  # dB of crest that offsets 1 dB of band energy
CLARITY_CREST_NORM_DB = 12.0
# Stem classification (Phase 14, mix-analysis skill v5)
TAG_LOW_SHARE = 0.5
TAG_HIGH_SHARE = 0.4
TAG_VOCAL_MID_SHARE = 0.5
TAG_VOCAL_SIBILANCE = 0.05
TAG_TRANSIENT_DENSITY = 0.5
TAG_TRANSIENT_RATIO = 0.25
TAG_PAD_RATIO = 0.15
TAG_PAD_DENSITY = 2.0  # slow movement/beating still counts as sustained
TAG_FULL_MIX_BANDS = 6
TAG_FULL_MIX_FLOOR_DB = -20.0
# A finished SONG is low-dominated too (a hip-hop master puts 75-90 % of its power under 250 Hz),
# and was read as `kick` - the whole song then went into Auto Mix's drums group (user QA
# 2026-10-22). It is told from a drum part by its attack: on a labelled corpus every released song
# read a transient-energy ratio of 0.17-0.38 and every drum loop or drum stem 0.42-0.97, while
# kicks and basses have no mid or top-end energy to speak of (mids <= 0.06 and top 0.000).
TAG_MIX_ATTACK_MAX = 0.40
TAG_MIX_MID_SHARE = 0.05
TAG_MIX_HIGH_SHARE = 0.004
TAG_MIX_BANDS = 5
# Over-compression (2026-06 review, mix-analysis skill: dynamics telemetry):
# a chain dynamics processor reducing this hard on average — or pinned (near-
# always active AND materially reducing) — sets the `over_compression` flag
# on render_and_analyze. The limiter is exempt: its QC is psr_db + the
# MasterReport readout, and a loud master legitimately keeps it busy.
OVER_COMPRESSION_GR_MEAN_DB = 6.0
OVER_COMPRESSION_PINNED_ACTIVE_PCT = 95.0
OVER_COMPRESSION_PINNED_MEAN_DB = 3.0
# Over-limiting (Phase 34): the limiter is allowed to be busy on loud masters,
# but low PSR/PLR plus sustained limiter GR means the master is likely being
# flattened rather than merely caught. Thresholds are intentionally conservative
# and feed a warning flag, not a guardrail.
OVER_LIMITING_PSR_DB = 8.0
OVER_LIMITING_PLR_DB = 8.5
OVER_LIMITING_GR_MEAN_DB = 0.8
OVER_LIMITING_PINNED_ACTIVE_PCT = 95.0
OVER_LIMITING_PINNED_MEAN_DB = 2.5
OVER_LIMITING_GR_MAX_DB = 4.0
# Resonance detection (Phase 20, mix-analysis skill v6)
RESONANCE_NPERSEG = 8192            # finer grid than band energy's 4096
RESONANCE_LO_HZ = 60.0
RESONANCE_HI_HZ = 16000.0
RESONANCE_BASELINE_HALF_OCT = 1.0 / 3.0  # median window half-width
RESONANCE_MIN_PROMINENCE_DB = 8.0
RESONANCE_PROMINENCE_CAP_DB = 40.0
RESONANCE_MAX_REPORTED = 3
RESONANCE_MIN_SPACING_OCT = 1.0 / 3.0    # reported peaks at least this apart
RESONANCE_Q_CAP = 50.0
# v9 (reader E5·R11): a peak that is a NOTE THE PART PLAYS is not a resonance. On material a
# tuning can be read from, a peak whose centre is within RESONANCE_NOTE_CENTS of a semitone
# of that tuning is a played note, and so is one within RESONANCE_HARMONIC_CENTS of a
# harmonic (2..RESONANCE_HARMONIC_MAX) of such a note - the 7th, 11th and 13th partials of a
# held tone sit between the pitches. The centre is the power centroid over
# +-RESONANCE_CENTRE_CENTS: a bin is 90 cents wide at 110 Hz, and a note sung with vibrato
# peaks at the two ends of its swing.
RESONANCE_NOTE_CENTS = 30.0
RESONANCE_HARMONIC_CENTS = 25.0
RESONANCE_HARMONIC_MAX = 64
RESONANCE_CENTRE_CENTS = 50.0
RESONANCE_CENTRE_RISE_DB = 3.0
_TUNING_UNREAD = object()
# Tempo estimation (Phase 20; the detector rewritten in v7 — mix-analysis skill)
TEMPO_MIN_BPM = 60.0
TEMPO_MAX_BPM = 200.0
TEMPO_MIN_CONFIDENCE = 0.25  # below this the grid is noise: report None
# The onset envelope the tempo is read from (v7): band energies every TEMPO_HOP samples
# (2.7 ms at 48 k — a beat is almost never a whole number of 512-sample hops, and a
# half-hop miss cost a beat lag most of its correlation), each band compressed against
# ITS OWN peak so that a kick, a snare and a hat are onsets on one scale.
TEMPO_FRAME = 512
TEMPO_HOP = 128
TEMPO_BAND_EDGES_HZ = (0.0, 250.0, 1000.0, 4000.0, 16000.0)
TEMPO_MU = 100.0             # mu-law compression of each band's energy
TEMPO_ENV_SMOOTH_S = 0.02    # so a lag between two hops still correlates
TEMPO_DETREND_S = 1.5        # what stands above its own neighbourhood is an onset
TEMPO_PRIOR_BPM = 120.0      # the octave nobody can measure is decided by a prior…
TEMPO_PRIOR_OCTAVES = 1.0    # …a log-normal this wide, centred here
# A candidate at 1/2, 3/2 or 2/3 of the winner's period scoring within this of it means
# the octave (or the 3:2) is NOT decided: the tempo is still reported, with its confidence
# capped under the DAW's auto-align gate (App.cpp: `detectedConf >= 0.5f`), so it is shown
# and never written into the edit unasked. test_audit_p1_fixes.py pins the two together.
# The DOUBLE period is deliberately not a rival: alternate beats differ in nearly all music
# (kick, snare), so a strong two-beat period is what a backbeat IS and says nothing about
# the octave — with it in the list the repo's own 120 BPM demo master, and every rock loop
# from 137 BPM up, read right and were never aligned.
TEMPO_RIVAL_RATIO = 0.8
TEMPO_RIVAL_RELATIONS = (0.5, 1.5, 2.0 / 3.0)        # rival period / winner period
TEMPO_AMBIGUOUS_CONFIDENCE = 0.45
TEMPO_AUTO_ALIGN_CONFIDENCE = 0.5
# A pulse with NO accent structure at the longer period — the half-period candidate
# correlates as well as the winner — is read at its own period, up to this tempo (a uniform
# click at 174 BPM is 174, not the 87 the prior prefers); above it, it is a subdivision.
TEMPO_UNIFORM_RATIO = 0.97
TEMPO_UNIFORM_MAX_BPM = 180.0
# The flux fallback (see _tempo) must clear a higher bar than the detector itself: on eight
# seconds of white noise it reads a "period" at 0.26-0.28 in one seed out of four.
TEMPO_FALLBACK_MIN_CONFIDENCE = 0.35
# THE BEAT, AND THE FIGURE PLAYED ACROSS IT (analysis v13, 2026-10-11; `_tempo_period`). A dotted
# figure - kicks three sixteenths apart, the commonest shape in hip-hop and trap - correlates at
# 2/3 or 3/2 of the beat as well as the beat does, and the prior, centred on 120, prefers it for
# anything felt at 80-95 BPM: a released song at 172 (86) read 114.66 at confidence 0.69 on its
# own drum stem, and thirteen of 273 tagged hip-hop loops read in the same family. What tells
# the two apart is the
# BAR: the pattern comes round every three, four, six or eight beats of the tempo it is in, and
# does not come round on the other's. A candidate in the 3:2 family of the winner (its period
# 2/3, 3/2, 4/3 or 3/4 of the winner's) takes its place when the envelope repeats at one of
# those multiples of ITS period by this much more than at any of the winner's.
# Sized on 344 labelled files (those loops, 63 vocal takes, seven songs and stems): margins
# 0.10 to 0.20 move the same five - four loops and that drum stem, each to its tempo or the
# octave - and nothing that read right; 0.05 breaks three, 0.30 stops fixing. And on synthetic
# shuffles, 6/8, waltzes, tresillos and straight beats built to break it: none moves at any of
# those margins. Eleven loops of that family still read as they did (their bars do not decide
# it), and so does the full mix of that song, at the capped 0.45. A multiple is searched within 2 % of its
# place - the winner's own lag is only a peak near the beat, and four of it can sit a twentieth
# of a second from the bar - and counted only inside half the take.
TEMPO_METER_MULTIPLES = (3, 4, 6, 8)
TEMPO_METER_RELATIONS = (2.0 / 3.0, 1.5, 4.0 / 3.0, 0.75)     # rival period / winner period
TEMPO_METER_MARGIN = 0.15
TEMPO_METER_TOLERANCE = 0.02
TEMPO_METER_MIN_SHARE = 0.5      # a rival must itself correlate at half the winner's level
# The grid of a take from its onsets FOLDED (analysis v12, 2026-10-11; `_fold_grid`). The lag
# grid quantises the period to one envelope hop - 0.6 % at 124 BPM - and the least-squares fit
# that was to mend it scores a grid by how near the onsets fall to BEAT lines: with eighths or
# sixteenths in the part, half the onsets sit half a beat from a line whatever the period, so
# no fit ever scored better than the coarse one. Two released songs made to a click at 124.000
# BPM read 123.725 and 123.833 on their masters, and 123.784 on a drum stem at confidence 0.91
# (exactly 167 hops); a 12 s synthetic loop at 124 read 122.872.
# Folding is the pulsar-timing statistic: onset times taken modulo a trial period pile up - at
# the beat and at every subdivision of it - only when the period is right. Z is the power of
# that pile-up over TEMPO_FOLD_HARMONICS harmonics, in units of what as many onsets at random
# times would give.
TEMPO_FOLD_HARMONICS = 8        # down to a thirty-second: the hats sharpen the peak the kick has
TEMPO_FOLD_OVERSAMPLE = 4       # trial periods per width of the highest harmonic's peak...
TEMPO_FOLD_MIN_STEPS = 16       # ...and never fewer than this across the span (a short take)
TEMPO_FOLD_MIN_ONSETS = 16
TEMPO_FOLD_MAX_S = 600.0        # ten minutes of onsets place the period to 1e-5; more is cost
# Random onsets give Z of 1-3, and the two stems the detector misreads 1.4 and 2.3. Sized on
# 153 recorded loops labelled 85 BPM, 190 synthetic loops of 6-32 s and fifty synthetic takes
# of 20 s to ten minutes (the mix-analysis skill has the table).
TEMPO_FOLD_MIN_Z = 8.0
# A second period, outside the peak's own lobe, standing this high beside the first: the take
# does not have ONE grid - an edit off the grid (0.75), a tempo that steps a third of a BPM at
# the half (0.80), two islands of drums minutes apart, which fringe like two slits (0.54-0.93).
# One steady song: 0.01-0.22.
TEMPO_FOLD_MAX_RIVAL = 0.5
TEMPO_FOLD_PHASE_BINS = 96      # the folded envelope: 5 ms a bin at 124 BPM
# A tempo is a spacing of ONSETS, and a part with none has none (2026-10-11). The tempo envelope
# is frame energies every 128 samples, and a steady tone beats against that hop: a plain 220 Hz
# sine read 172 BPM, a 440 Hz one 181, each at confidence 0.45 - under the DAW's align gate,
# but said to the agent, and `analyze_tempo` asks this of every pad and bass. Such material has
# 0 or 1 precise onsets; the sparsest rhythmic fixture (a click at 72 BPM for 12 s) has 14, and
# 8 is what `analyze_timing` already asks before it will measure a part.
TEMPO_MIN_ONSETS = 8
TEMPO_FOLD_CLASS_S = 0.025      # an onset this near the chosen phase is the beat's own
# Musical key (Phase 30, mix-analysis skill): chroma vs Krumhansl-Schmuckler
# key profiles. PITCH_CLASSES indexes 0=C .. 11=B (sharps); C0_HZ anchors the
# pitch-class fold.
PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
C0_HZ = 16.351597831287414
KEY_NPERSEG = 8192           # ~5.9 Hz bins at 48k — resolves pitch classes ≥ ~65 Hz
KEY_LO_HZ = 65.0             # ~C2; below this a semitone is narrower than a bin
KEY_HI_HZ = 2000.0          # most key-defining energy sits below here
KEY_FRAME_GATE_DB = -60.0   # skip STFT frames > 60 dB below the loudest one
KEY_MIN_CONFIDENCE = 0.6    # below this the tonal center is ambiguous: report None
# A key is several notes (reader E5·R6). ONE pitch class alone correlates 0.68 with a
# Krumhansl profile - above the floor - and a note's own harmonic series spells a major triad,
# so one held note read "A maj 0.79" and a sine kick under band-noise toms "C# min 0.66".
KEY_MIN_PITCH_CLASSES = 3   # pitch classes that each hold >= KEY_PITCH_CLASS_SHARE of the power
KEY_PITCH_CLASS_SHARE = 0.05
# ...and the fit must survive without its loudest pitch class: the same correlation over the
# other eleven. Measured: the 24 cadences 0.77-0.96, scales and triads 0.72-0.89, chords under
# a drum loop 6 dB over them 0.73; drum stems 0.27-0.29, one sine -0.31, one note + noise 0.0.
KEY_MIN_REST_CORRELATION = 0.5
# THE RELATIVE KEY (user QA 2026-10-08, O12; analysis v10). C major and A minor are the same
# seven notes, and the profiles tell them apart only by which notes are leaned on - so a part in
# A minor that leans on C reads as C major. Where the two fits are CLOSE the bass settles it: the
# pitch class the lowest register sits on is the tonal centre a listener hears. Only a near-tie
# is settled this way (KEY_RELATIVE_MARGIN, in correlation), and only by a bass line that is
# clearly on one of the two tonics: KEY_BASS_SHARE of the low register's power on the relative
# key's tonic and KEY_BASS_RATIO times what the named key's tonic holds. Sized on 74 labelled
# loops and 38 bass + keys mixes made from them (mix-analysis skill): the mixes go 16 -> 17
# right and 4 -> 3 named as the relative, nothing that was right changes, and a part with no
# bass register reads exactly what it read.
KEY_RELATIVE_MARGIN = 0.10
KEY_BASS_SHARE = 0.25
KEY_BASS_RATIO = 2.0
KEY_BASS_LO_HZ = 36.0        # D1. Not lower: on recorded loops 30-35 Hz holds rumble, and it read as B0 / C1
KEY_BASS_HI_HZ = 165.0       # E3: above this the "bass" is the bottom of a chord
KEY_BASS_DECIMATE = 16       # 48 kHz -> 3 kHz: 0.73 Hz a bin at 4096, a semitone at 36 Hz is 2.1
KEY_BASS_NPERSEG = 4096
# Krumhansl-Kessler (1982) major/minor key profiles, rooted at C.
KRUMHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                            2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KRUMHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                            2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
# Tuning detection (Phase 31, mix-analysis skill): global offset from equal
# temperament (A440) + intonation coherence, reusing the Phase 30 pitch STFT.
TUNING_PEAK_REL_DB = -40.0      # per frame: keep spectral peaks within this of its max
TUNING_HIST_BINS = 100          # 1-cent resolution histogram over [-50, 50] cents
TUNING_COHERENCE_CENTS = 15.0   # weighted mass within ±this of the mode = "in tune"
TUNING_MIN_PEAKS = 16           # too few spectral peaks ⇒ unpitched: report None
TUNING_MIN_PEAKINESS = 3.0      # mode bin must stand this × above the mean bin ⇒ tonal
# A tuning is read from STEADY partials (reader E5·R6): a peak whose neighbours
# TUNING_STEADY_BINS away stand at most TUNING_STEADY_RATIO of it - the main lobe of a tone
# held through the frame. A kick's decaying sine and a click are bursts: their lobe is many
# bins wide and its centre is not a pitch to the cent ("the drums sit 45 cents sharp").
TUNING_STEADY_BINS = 3
TUNING_STEADY_RATIO = 0.5
TUNING_MIN_STEADY_SHARE = 0.25  # of all peak magnitude; chords under loud drums 0.61, drums <= 0.14
TUNING_FLAG_CENTS = 15.0        # |offset| ≥ this flags out_of_tune (mcp_server)
TUNING_COHERENCE_FLOOR = 0.5    # coherence below this also flags out_of_tune
# Spatial masking (Phase 20): channel keys for per_channel_worst_excess_db
CHANNEL_LABELS = ("left", "right")


def _frames_channels(audio: np.ndarray) -> np.ndarray:
    """(frames, channels) float64 - and FINITE (reader E5·R8). Every public entry point comes
    through here. One NaN or inf made every sum and comparison after it NaN (`nan > -1.0` is
    False: no true-peak flag; `nan < floor` is False: a "confident" tempo), a measured delta of
    NaN went into the cross-project calibration store, and nothing there ever recovers. A
    non-finite sample is dropped, as the DAW's node and the offline render's read drop one;
    the caller's array is never written."""
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    if not np.isfinite(audio).all():
        audio = np.where(np.isfinite(audio), audio, 0.0)
    return audio


def _round_db(value: float) -> float:
    return round(float(value), 1)


def _lufs_arg(audio: np.ndarray) -> np.ndarray:
    """What pyloudnorm measures: the signal AS IT IS PLAYED. A 1-channel signal plays from both
    speakers — the DAW's graph runs a mono clip dual-mono at unity, and its export is a stereo
    file — so it is measured as two identical channels (analysis v7; full audit 2026-10-02,
    E5·R1). Measured as ONE channel, BS.1770 read every all-mono, all-centred project 3.01 dB
    under what the DAW plays: the readout said −14.0 for a master that exports at −11.0, the
    limiter's loudness make-up was 3 dB hot, and panning one stem by 1 % moved every number
    3 dB with nothing audible changing."""
    return np.repeat(audio, 2, axis=1) if audio.shape[1] == 1 else audio


def _integrated_lufs(meter: pyln.Meter, audio: np.ndarray) -> Optional[float]:
    if audio.shape[0] < int(0.4 * meter.rate):  # below one BS.1770 gating block
        return None
    lufs = meter.integrated_loudness(_lufs_arg(audio))
    return None if not np.isfinite(lufs) else float(lufs)


def _k_weighted(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Audio through the BS.1770 K-weighting pre-filters (high shelf + high
    pass), once, all channels. Coefficients come from pyloudnorm's own Meter
    so windowed measurements use exactly the weighting the integrated meter
    uses; the Phase 10 parity test guards this against internals changing."""
    from scipy.signal import lfilter
    out = audio
    for stage in pyln.Meter(sample_rate)._filters.values():
        out = stage.passband_gain * lfilter(stage.b, stage.a, out, axis=0)
    return out


def _short_term_lufs_series(audio: np.ndarray, sample_rate: int,
                            window_s: float, hop_s: float) -> np.ndarray:
    """Ungated short-term loudness per sliding window (BS.1770 / EBU R128
    short-term, v4): K-weight once, per-window mean square via cumulative
    sums — O(n) regardless of hop. Empty array if audio < one window."""
    window = int(window_s * sample_rate)
    hop = max(1, int(hop_s * sample_rate))
    if window <= 0 or audio.shape[0] < window:
        return np.empty(0)
    weighted = _k_weighted(audio, sample_rate)
    csum = np.vstack([np.zeros((1, audio.shape[1])),
                      np.cumsum(weighted ** 2, axis=0)])
    starts = np.arange(0, audio.shape[0] - window + 1, hop)
    mean_squares = (csum[starts + window] - csum[starts]) / window
    weights = np.ones(audio.shape[1])
    weights[3:] = 1.41  # BS.1770 surround weights; stereo unaffected
    if audio.shape[1] == 1:
        weights = weights * 2.0   # mono plays from both speakers (_lufs_arg, analysis v7)
    power = mean_squares @ weights
    return -0.691 + 10.0 * np.log10(np.maximum(power, SILENCE_PEAK ** 2))


def _short_term_max_lufs(meter: pyln.Meter, audio: np.ndarray,
                         sample_rate: int) -> Optional[float]:
    """Max short-term loudness over a sliding 3 s window, 100 ms hop (v4:
    ungated K-weighted short-term per window — was gated-integrated per
    window; see mix-analysis skill). Whole-file integrated when audio ≤ 3 s;
    None when even the loudest window sits under the −70 LUFS absolute gate."""
    if audio.shape[0] <= int(SHORT_TERM_WINDOW_S * sample_rate):
        return _integrated_lufs(meter, audio)
    series = _short_term_lufs_series(audio, sample_rate,
                                     SHORT_TERM_WINDOW_S, SHORT_TERM_HOP_S)
    if series.size == 0:
        return None
    loudest = float(series.max())
    return loudest if loudest > LRA_ABS_GATE_LUFS else None


# ---------- one window size in HERTZ, whatever the sample rate (analysis v7) ----------
#
# Every spectral and pitch window below was written as a sample count for 44.1 / 48 kHz
# material (`KEY_NPERSEG = 8192  # ~5.9 Hz bins at 48k`), and the project renders at its
# HIGHEST stem rate (`Session.project_sample_rate`, "never down"). One 96 kHz stem therefore
# halved every window: an 8192-sample pitch STFT is 11.7 Hz a bin at 96 k and 23.4 Hz at 192 k
# — wider than a semitone below ~200 / ~400 Hz, so the pitch-class fold and the tuning
# histogram lost the bass and the chord roots — a 4096-sample Welch left one or two bins in
# `sub`, and a 2048-sample f0 frame held too few periods of a bass note (full audit 2026-10-02,
# reader E5.R3: 24-key cadence sweep 23/24 right at 44.1 k, 24/24 at 48 k, 14/24 at 88.2 k,
# 17/24 at 96 k, 3/24 at 192 k, the wrong answers at confidence 0.6-0.88; a 96 k copy of the
# very master in the mix "differed" from it by 1.4 dB in the bass).
#
# `_rate_factor` is how many times the base rate the audio runs at, as a power of two: 1 up to
# 50 kHz, 2 for 88.2 / 96, 4 for 176.4 / 192. The Welch windows are that many times longer
# (the same Hz per bin, nothing resampled); the pitch front ends, which only look below
# ~5 kHz, are DECIMATED by it instead, so they run on exactly the path the 44.1 / 48 k tests
# exercise — every frame count below them (NOTE_MIN_FRAMES, the hop) keeps its meaning — and
# their STFT memory does not double with the rate. At 44.1 / 48 kHz the factor is 1 and
# every number is bit-for-bit what it was.
ANALYSIS_BASE_RATE_HZ = 50_000.0


def _rate_factor(sample_rate: float) -> int:
    factor = 1
    while sample_rate / factor > ANALYSIS_BASE_RATE_HZ:
        factor *= 2
    return factor


def _welch_nperseg(base: int, n_samples: int, sample_rate: float) -> int:
    """A Welch segment of `base` samples at the base rate — the same span in seconds, and so
    the same Hz per bin, at any rate."""
    return min(base * _rate_factor(sample_rate), n_samples)


def _at_base_rate(mono: np.ndarray, sample_rate: float) -> tuple[np.ndarray, float]:
    """`mono` decimated to the base rate (44.1 / 48 kHz for the standard rates), with the rate
    it then runs at. Polyphase, anti-aliased, deterministic."""
    factor = _rate_factor(sample_rate)
    if factor == 1:
        return mono, sample_rate
    return resample_poly(mono, 1, factor), sample_rate / factor


TRUE_PEAK_BLOCK = 1 << 18        # frames oversampled at once (5.5 s at 48 kHz)
TRUE_PEAK_OVERLAP = 64           # frames of context each side: the 4x filter spans ~10


# ---------- whole-song transforms, a piece at a time (reader E5·R4) ----------
#
# A Welch PSD and an STFT of the whole song each build every segment's spectrum at once: 0.7 GB
# for one Welch of six minutes, 1.4 GB for the hop-128 timing STFT - and the readout ran four
# of them on every brain reply. The two helpers below give the SAME numbers (the same window,
# padding, scaling and FFT; a test compares them with scipy's on a signal that crosses many
# seams) while holding one run of segments. Every whole-length Welch and STFT in this file
# goes through them; a new one does too.
WELCH_CHUNK_SEGMENTS = 256
STFT_CHUNK_SAMPLES = 1 << 21        # windowed samples transformed at once (16 MB of float64)


def _welch_psd(x: np.ndarray, sample_rate: float, nperseg: int,
               y: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    """`scipy.signal.welch(x, fs, nperseg=nperseg)` - Hann, 50 % overlap, the mean of the
    segments' periodograms - computed WELCH_CHUNK_SEGMENTS segments at a time. Each run is cut
    on segment boundaries, so it holds exactly the segments the whole-file call would, and the
    runs are averaged by how many they hold. With `y`, the cross-spectrum
    `scipy.signal.csd(x, y, fs, nperseg=nperseg)` the same way."""
    from scipy.signal import csd

    def spectrum(lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
        if y is None:
            return welch(x[lo:hi], fs=sample_rate, nperseg=nperseg)
        return csd(x[lo:hi], y[lo:hi], fs=sample_rate, nperseg=nperseg)

    hop = nperseg - nperseg // 2
    segments = 1 + (x.shape[0] - nperseg) // hop if x.shape[0] >= nperseg else 0
    if segments <= WELCH_CHUNK_SEGMENTS:
        return spectrum(0, x.shape[0])
    freqs, total = None, None
    for first in range(0, segments, WELCH_CHUNK_SEGMENTS):
        last = min(segments, first + WELCH_CHUNK_SEGMENTS)
        freqs, psd = spectrum(first * hop, (last - 1) * hop + nperseg)
        total = psd * (last - first) if total is None else total + psd * (last - first)
    return freqs, total / segments


def _stft_magnitudes(mono: np.ndarray, nperseg: int, hop: int):
    """Yield `|scipy.signal.stft(mono, nperseg=nperseg, noverlap=nperseg - hop)|` as
    (frames, bins) blocks of STFT_CHUNK_SAMPLES // nperseg frames, in order: the periodic Hann window, half
    a window of zeros in front and behind, zero-padded to a whole frame, 'spectrum' scaling.
    One padded copy of the signal and one block of spectra are alive at a time."""
    from scipy.fft import rfft
    from scipy.signal import get_window
    window = get_window("hann", nperseg)
    scale = 1.0 / window.sum()
    half = nperseg // 2
    extra = (-(mono.shape[0] + 2 * half - nperseg) % hop) % nperseg
    padded = np.concatenate([np.zeros(half), mono, np.zeros(half + extra)])
    frames = np.lib.stride_tricks.sliding_window_view(padded, nperseg)[::hop]
    block = max(1, STFT_CHUNK_SAMPLES // nperseg)
    for start in range(0, frames.shape[0], block):
        yield np.abs(rfft(frames[start:start + block] * window, axis=1) * scale)


def _stft_flux(mono: np.ndarray, nperseg: int, hop: int) -> np.ndarray:
    """Half-wave-rectified magnitude flux per hop: for each frame after the first, the sum over
    bins of how far each bin's magnitude rose. `np.maximum(np.diff(|stft|, axis=1), 0).sum(0)`."""
    parts: list[np.ndarray] = []
    previous = None
    for magnitude in _stft_magnitudes(mono, nperseg, hop):
        rows = magnitude if previous is None else np.vstack([previous[np.newaxis, :], magnitude])
        parts.append(np.maximum(np.diff(rows, axis=0), 0.0).sum(axis=1))
        previous = magnitude[-1]
    return np.concatenate(parts) if parts else np.empty(0)


def _analytic_signal(mono: np.ndarray) -> np.ndarray:
    """`mono` with its negative frequencies removed (complex64), a block at a time: each block
    is transformed with ANALYTIC_MARGIN samples of real audio on either side and only its own
    span is kept, so a long file never needs one FFT of its whole length."""
    from scipy.fft import next_fast_len
    from scipy.signal import hilbert
    frames = mono.shape[0]
    out = np.empty(frames, dtype=np.complex64)
    if frames == 0:
        return out
    centred = mono - float(np.mean(mono))       # DC has no pair: left in, it beats with a low tone
    if frames <= ANALYTIC_BLOCK + 2 * ANALYTIC_MARGIN:
        out[:] = hilbert(centred, N=next_fast_len(frames))[:frames]
        return out
    for start in range(0, frames, ANALYTIC_BLOCK):
        stop = min(frames, start + ANALYTIC_BLOCK)
        lead = min(ANALYTIC_MARGIN, start)
        tail = min(ANALYTIC_MARGIN, frames - stop)
        segment = centred[start - lead: stop + tail]
        out[start:stop] = hilbert(segment, N=next_fast_len(segment.shape[0]))[lead: lead + stop - start]
    return out


def _onset_flux(mono: np.ndarray, sample_rate: int) -> np.ndarray:
    """The onset function behind `_onset_times_precise`: how fast energy ARRIVES, per hop.

    Until 2026-10-09 this was `_stft_flux` on the real signal, and on recorded drums it was
    wrong in a way no synthetic click showed. The window is TIMING_NPERSEG samples - 11.6 ms -
    and a kick's fundamental has a period of about 21 ms. The spectrum of a REAL signal holds
    every tone twice, at +f and at -f; in a window shorter than the tone's period the two images
    land in the same bins and beat, so the magnitude there rises and falls at twice the tone's
    frequency for as long as the tone lasts. Rectified, that ripple is a train of "onsets": on a
    pack of 79 recorded loops a kick was reported twice (its attack, and the tallest ripple past
    the 50 ms refractory, 50-85 ms on - at whole multiples of the kick's half-period), or once
    and 25 ms late, and Extract Groove refused 73 of them.

    The ANALYTIC signal has no -f image, so a steady tone's short-time magnitude is steady
    whatever the window: the flux is taken on that. Two more things, each sized on the same pack
    and on synthesised kicks (the constants say how):

    * summed over bins, a tone in two bins counts for far less than a noise burst of the same
      energy in two hundred - and with the ripple gone a click-less sub kick beside loud hats
      would not be seen at all. The low band's own envelope (one number: the root of its bins'
      summed squares) is added back at ONSET_LOW_BAND_WEIGHT.
    * magnitudes are halved, so everything above the low band reads exactly as `_stft_flux`
      read it: a hat or a snare is timed as it always was.

    Checked against a time-domain attack meter (the first block of 1 ms to pass 30 % of the
    hit's rise; itself checked on synthesised hits): recorded kicks sit +1.7 ms from the
    physical attack at the median where they sat +25.5, full drum loops +1.7 (p75 +2.9) where
    p75 was +28.6, hats +0.8 as before.
    """
    from scipy.fft import fft
    from scipy.signal import get_window
    # The window is a length of TIME (11.6 ms): TIMING_NPERSEG at 44.1 / 48 k, twice that at
    # 88.2 / 96, four times at 176.4 / 192. Left in samples it was half as long at 96 k, where a
    # sub kick's 8 ms rise is spread over twice the hops - and was not found at all.
    factor = _rate_factor(sample_rate)
    nperseg, hop = TIMING_NPERSEG * factor, TIMING_HOP * factor
    analytic = _analytic_signal(mono)
    window = get_window("hann", nperseg).astype(np.float32)
    scale = 0.5 / float(window.sum())
    half = nperseg // 2
    extra = (-(analytic.shape[0] + 2 * half - nperseg) % hop) % nperseg
    padded = np.concatenate([np.zeros(half, np.complex64), analytic,
                             np.zeros(half + extra, np.complex64)])
    frames = np.lib.stride_tricks.sliding_window_view(padded, nperseg)[::hop]
    low_bins = max(1, int(np.ceil(ONSET_LOW_BAND_HZ * nperseg / sample_rate)))
    block = max(1, STFT_CHUNK_SAMPLES // nperseg)
    parts: list[np.ndarray] = []
    previous = previous_low = None
    for start in range(0, frames.shape[0], block):
        magnitude = np.abs(fft(frames[start:start + block] * window, axis=1)[:, :half + 1]) * scale
        low = np.sqrt(np.square(magnitude[:, :low_bins]).sum(axis=1))
        rows = magnitude if previous is None else np.vstack([previous[np.newaxis, :], magnitude])
        lows = low if previous_low is None else np.concatenate([[previous_low], low])
        parts.append(np.maximum(np.diff(rows, axis=0), 0.0).sum(axis=1)
                     + ONSET_LOW_BAND_WEIGHT * np.maximum(np.diff(lows), 0.0))
        previous, previous_low = magnitude[-1], low[-1]
    return np.concatenate(parts).astype(np.float64) if parts else np.empty(0)


def _true_peak_dbtp(audio: np.ndarray) -> float:
    """4x-oversampled peak, a block at a time (reader E5·R4). The whole-file version built a 4x
    float64 copy of the master and its `np.abs` - 2.2 GB at six minutes, ~22 GB for an hour.
    Each block is oversampled with TRUE_PEAK_OVERLAP frames of real audio on either side and
    only its own span is read, so every output sample is the one the whole-file filter gives."""
    frames = audio.shape[0]
    if frames <= TRUE_PEAK_BLOCK + 2 * TRUE_PEAK_OVERLAP:
        peak = float(np.max(np.abs(resample_poly(audio, 4, 1, axis=0)))) if frames else 0.0
        return 20.0 * np.log10(max(peak, SILENCE_PEAK))
    peak = 0.0
    for start in range(0, frames, TRUE_PEAK_BLOCK):
        stop = min(frames, start + TRUE_PEAK_BLOCK)
        lead = min(TRUE_PEAK_OVERLAP, start)
        tail = min(TRUE_PEAK_OVERLAP, frames - stop)
        over = resample_poly(audio[start - lead: stop + tail], 4, 1, axis=0)
        own = over[4 * lead: over.shape[0] - 4 * tail]
        if own.size:
            peak = max(peak, float(np.max(np.abs(own))))
    return 20.0 * np.log10(max(peak, SILENCE_PEAK))


def _peak_dbfs(audio: np.ndarray) -> float:
    return 20.0 * np.log10(max(np.max(np.abs(audio)), SILENCE_PEAK))


def _band_energy(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Per-band power in dB relative to full-band power (Welch PSD, mono sum)."""
    mono = audio.mean(axis=1)
    nperseg = _welch_nperseg(4096, mono.shape[0], sample_rate)
    freqs, psd = _welch_psd(mono, sample_rate, nperseg)
    total = psd.sum()
    bands: dict[str, float] = {}
    for label, lo_hz, hi_hz in zip(BAND_LABELS, BAND_EDGES_HZ[:-1], BAND_EDGES_HZ[1:]):
        power = psd[(freqs >= lo_hz) & (freqs < hi_hz)].sum()
        if total <= 0 or power <= 0:
            bands[label] = BAND_ENERGY_FLOOR_DB
        else:
            bands[label] = max(_round_db(10.0 * np.log10(power / total)),
                               BAND_ENERGY_FLOOR_DB)
    return bands


def _stereo_correlation(audio: np.ndarray) -> Optional[float]:
    """Zero-lag Pearson correlation of L/R. None for mono or zero-variance channels."""
    if audio.shape[1] < 2:
        return None
    left, right = audio[:, 0], audio[:, 1]
    if np.std(left) < SILENCE_PEAK or np.std(right) < SILENCE_PEAK:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _width_db(audio: np.ndarray) -> Optional[float]:
    """Broadband stereo width: side-vs-mid power in dB. 0 dB = side as strong
    as mid; more negative = narrower, toward mono. Clamped with the same floor
    and cap as the per-band width used for references, so the two numbers read
    on one scale. None for mono or silence; pure anti-phase reads the cap."""
    if audio.shape[1] < 2:
        return None
    mid = (audio[:, 0] + audio[:, 1]) / 2.0
    side = (audio[:, 0] - audio[:, 1]) / 2.0
    mid_power = float(np.mean(np.square(mid)))
    side_power = float(np.mean(np.square(side)))
    if mid_power <= 0.0:
        # No mid at all: silence is unmeasurable — but PERFECT ANTI-PHASE (L = -R) is the
        # widest possible input and used to fall through this None too, reporting "mono"
        # alongside stereo_correlation = -1.0 (2026-08-08 review). All side, no mid = the cap.
        return None if side_power <= 0.0 else BAND_WIDTH_CAP_DB
    if side_power <= 0.0:
        return BAND_WIDTH_FLOOR_DB
    ratio_db = float(10.0 * np.log10(side_power / mid_power))
    return _round_db(max(BAND_WIDTH_FLOOR_DB, min(BAND_WIDTH_CAP_DB, ratio_db)))


TILT_LO_HZ = 20.0
TILT_OCTAVES = 10                # 20 Hz .. 20.48 kHz, cut at 20 kHz / Nyquist
TILT_HI_HZ = 20000.0
TILT_NPERSEG = 16384             # ~2.9 Hz bins at 48 k: seven bins in the lowest octave
TILT_FLOOR_DB = 80.0             # an octave this far under the loudest holds no slope
TILT_MIN_OCTAVES = 4


def _spectral_tilt_db_per_oct(audio: np.ndarray, sample_rate: int) -> Optional[float]:
    """Slope, in dB per octave, of the mean power density of each OCTAVE from 20 Hz to 20 kHz
    (analysis v8; reader E5·R5, James 2026-10-05: "octave-weighted, white / pink / brown still
    0 / -3 / -6"). One point per octave, a least-squares line through them.

    Through v7 the line was fitted to every Welch bin, and bins are evenly spaced in Hz: half
    of them sit above 10 kHz and under 1 % below 200 Hz, so the "tilt" was the slope of the top
    octave - +6 dB of bass moved it 0.06 dB/oct, a 16 kHz low-pass (a 128 kbps reference) 3.3.
    Every octave now weighs the same. The mean is of POWER, so a hole narrower than an octave
    costs that octave a few dB instead of dragging a point to the floor; an octave more than
    TILT_FLOOR_DB under the loudest is held there. A straight spectrum reads what it read."""
    mono = audio.mean(axis=1)
    nperseg = _welch_nperseg(TILT_NPERSEG, mono.shape[0], sample_rate)
    freqs, psd = _welch_psd(mono, sample_rate, nperseg)
    centres: list[float] = []
    levels: list[float] = []
    for octave in range(TILT_OCTAVES):
        lo_hz = TILT_LO_HZ * 2.0 ** octave
        hi_hz = min(lo_hz * 2.0, TILT_HI_HZ, sample_rate / 2.0)
        band = (freqs >= lo_hz) & (freqs < hi_hz)
        if hi_hz <= lo_hz or not band.any():
            continue
        centres.append(0.5 * (np.log2(lo_hz) + np.log2(hi_hz)))
        levels.append(float(psd[band].mean()))
    if len(levels) < TILT_MIN_OCTAVES or max(levels) <= 0.0:
        return None
    floor = max(levels) * 10.0 ** (-TILT_FLOOR_DB / 10.0)
    level_db = 10.0 * np.log10(np.maximum(levels, floor))
    slope, _ = np.polyfit(np.asarray(centres), level_db, 1)
    return float(slope)


def _crest_factor_db(audio: np.ndarray) -> float:
    rms = np.sqrt(np.mean(audio ** 2))
    return _peak_dbfs(audio) - 20.0 * np.log10(max(rms, SILENCE_PEAK))


def _rms_frames_db(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """RMS per 400 ms frame, 100 ms hop, in dBFS (mono sum)."""
    mono = audio.mean(axis=1)
    frame, hop = int(RMS_FRAME_S * sample_rate), int(RMS_HOP_S * sample_rate)
    if mono.shape[0] < frame:
        frame = mono.shape[0]
    starts = range(0, mono.shape[0] - frame + 1, hop)
    rms = np.array([np.sqrt(np.mean(mono[s:s + frame] ** 2)) for s in starts])
    return 20.0 * np.log10(np.maximum(rms, SILENCE_PEAK))


def _dynamic_spread_db(audio: np.ndarray, sample_rate: int) -> Optional[float]:
    """p95 - p10 of the 400 ms RMS frame levels: how much the level moves."""
    frames_db = _rms_frames_db(audio, sample_rate)
    active = frames_db[frames_db > MASKING_GATE_DBFS]
    if active.size < 3:
        return None
    return float(np.percentile(active, 95) - np.percentile(active, 10))


def _spectral_flux(mono: np.ndarray, sample_rate: int
                   ) -> Optional[tuple[np.ndarray, float]]:
    """Half-wave-rectified spectral flux per STFT hop (1024/512 at the base
    rate, the same span in seconds above it — `_rate_factor`) — the onset
    strength signal shared by onset detection and tempo estimation. Returns
    (flux, hop_s); None when the audio is too short for one STFT frame."""
    factor = _rate_factor(sample_rate)
    if mono.shape[0] < 2048 * factor:
        return None
    flux = _stft_flux(mono, 1024 * factor, 512 * factor)
    return flux, 512 * factor / sample_rate


def _onset_peaks(mono: np.ndarray, sample_rate: int) -> Optional[tuple[np.ndarray, float]]:
    """Onset frames via spectral-flux peaks (deterministic; thresholds
    defined in the mix-analysis skill). Returns (peak_frames, hop_s)."""
    from scipy.signal import find_peaks
    flux_hop = _spectral_flux(mono, sample_rate)
    if flux_hop is None:
        return None
    flux, hop_s = flux_hop
    if flux.max() <= 0:
        return np.array([], dtype=int), hop_s
    min_gap = max(1, int(0.05 / hop_s))  # onsets at least 50 ms apart
    height = np.median(flux) + 1.5 * (np.percentile(flux, 90) - np.median(flux))
    peaks, _ = find_peaks(flux, height=height, distance=min_gap)
    return peaks, hop_s


def _onset_times_precise(mono: np.ndarray, sample_rate: int) -> np.ndarray:
    """Onset times in SECONDS, refined below the STFT hop.

    The shared `_onset_peaks` runs on a 512-sample hop — 11.6 ms at 44.1 k,
    which is the same order as the timing differences this is meant to measure
    (a drummer's lean is 5-30 ms), so its peak indices are mostly quantisation
    noise for this purpose. Two changes: a 4x finer hop (128 = 2.9 ms), and
    parabolic interpolation of each flux peak against its neighbours — the same
    sub-bin refinement `_tuning` uses on spectral peaks. Together they put the
    resolution well under the effect size.
    """
    from scipy.signal import find_peaks
    factor = _rate_factor(sample_rate)
    if mono.shape[0] < TIMING_NPERSEG * factor * 2:
        return np.array([])
    hop = TIMING_HOP * factor                 # the hop _onset_flux runs at this rate
    flux = _onset_flux(mono, sample_rate)
    if flux.size < 3 or flux.max() <= 0:
        return np.array([])
    hop_s = hop / sample_rate
    # The percentile threshold alone DEGENERATES on sparse material: when most
    # frames carry no flux, the median and the 90th percentile are both zero,
    # the threshold collapses to zero, and every numerical ripple becomes an
    # "onset" (a synthetic 6-hit bar detected 12). A sparse percussion take is
    # not a synthetic edge case, so floor the threshold against the loudest
    # onset as well — a real hit is never a tenth of the biggest one.
    spread = np.median(flux) + 1.5 * (np.percentile(flux, 90) - np.median(flux))
    height = max(float(spread), float(flux.max()) * TIMING_PEAK_FLOOR)
    peaks, _ = find_peaks(flux, height=height,
                          distance=max(1, int(0.05 / hop_s)))
    if peaks.size == 0:
        return np.array([])
    # A hit masks what is much weaker and just behind it (ONSET_MASK_*): the bumps in a kick's
    # own tail are peaks of this function too, at 14-20 % of the attack's.
    kept: list[int] = []
    for peak in peaks:
        if not any((peak - earlier) * hop_s < ONSET_MASK_S
                   and flux[peak] < ONSET_MASK_RATIO * flux[earlier] for earlier in kept[-4:]):
            kept.append(int(peak))
    peaks = np.asarray(kept, dtype=int)
    # Parabolic vertex through (peak-1, peak, peak+1); interior peaks only.
    interior = peaks[(peaks > 0) & (peaks < flux.size - 1)]
    left, centre, right = flux[interior - 1], flux[interior], flux[interior + 1]
    denom = left - 2.0 * centre + right
    shift = np.where(denom != 0.0, 0.5 * (left - right) / denom, 0.0)
    shift = np.clip(shift, -0.5, 0.5)
    # flux[j] = mag[j+1] − mag[j], so a peak at flux index j reports the energy
    # arriving in FRAME j+1. This +1 is not cosmetic and does NOT wash out: a
    # constant offset shifts every deviation by the same amount, so it lands
    # whole in `bias_ms` (the median of those deviations) and would read as the
    # player rushing by one hop. Measured residual after the correction is well
    # under a millisecond on synthetic onsets.
    return (interior + 1 + shift) * hop_s


def _grid_deviations(onsets_s: np.ndarray, bpm: float, offset_s: float,
                     subdivision: int) -> tuple[np.ndarray, np.ndarray]:
    """Signed distance from each onset to its nearest line of a `subdivision`
    grid. Returns (deviation in beats, grid index). Positive = LATE."""
    beat_s = 60.0 / bpm
    pos_beats = (onsets_s - offset_s) / beat_s
    index = np.round(pos_beats * subdivision)
    return pos_beats - index / subdivision, index.astype(int)


def _choose_subdivision(onsets_s: np.ndarray, bpm: float, offset_s: float
                        ) -> Optional[tuple[int, np.ndarray, np.ndarray]]:
    """The COARSEST grid the part actually plays on.

    A finer grid always fits at least as well (every 8th-note position is also a
    16th-note position), so "best fit" would always choose the finest candidate
    and every part would look like 32nds. Testing coarse-to-fine and stopping at
    the first grid that fits is what makes the answer meaningful.

    Triplet grids are candidates in their own right: a heavily swung part IS a
    triplet grid, and describing it as sloppy 16ths would be wrong.
    """
    for subdivision in TIMING_SUBDIVISIONS:
        dev_beats, index = _grid_deviations(onsets_s, bpm, offset_s, subdivision)
        interval_beats = 1.0 / subdivision
        fit = np.abs(dev_beats) <= TIMING_FIT_FRACTION * interval_beats
        if fit.mean() >= TIMING_FIT_MIN_RATIO:
            return subdivision, dev_beats, index
    return None


def _tempo_onset_envelopes(mono: np.ndarray, sample_rate: int
                           ) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
    """The two onset envelopes the tempo is read from, and their hop in seconds (v7).

    ENERGY (the detector's own): per frame, the power in each of TEMPO_BAND_EDGES_HZ's bands;
    each band mu-law compressed against its own peak, differenced, half-wave rectified; the
    bands summed. FLUX (the fallback): the half-wave-rectified magnitude flux of each band,
    each scaled to its own 99.5th percentile; the bands summed. Both smoothed over 20 ms.

    Why not the summed linear spectral flux `_spectral_flux` returns (full audit 2026-10-02,
    reader E5.R2): a kick moves a handful of low bins and a snare hundreds, so on that
    envelope the kick-to-kick lag — two beats — always outscored the beat, and every backbeat
    read half-time (120 -> 60 at confidence 0.89); and its 512-sample hop put most beat lags
    between two frames. One pass of chunked FFTs over a strided view: the cost is linear in
    the length and the memory is one chunk."""
    factor = _rate_factor(sample_rate)
    frame, hop = TEMPO_FRAME * factor, TEMPO_HOP * factor
    if mono.shape[0] < frame * 4:
        return None
    n = 1 + (mono.shape[0] - frame) // hop
    window = np.hanning(frame)
    freqs = np.fft.rfftfreq(frame, 1.0 / sample_rate)
    masks = [(freqs >= lo) & (freqs < hi)
             for lo, hi in zip(TEMPO_BAND_EDGES_HZ[:-1], TEMPO_BAND_EDGES_HZ[1:])]
    band_power = np.zeros((len(masks), n))
    band_flux = np.zeros((len(masks), n))
    frames = np.lib.stride_tricks.sliding_window_view(mono, frame)[::hop]
    previous = None
    chunk = 4096
    for start in range(0, n, chunk):
        magnitude = np.abs(np.fft.rfft(frames[start:start + chunk] * window, axis=1))
        rows = magnitude if previous is None else np.vstack([previous[np.newaxis, :], magnitude])
        rise = np.maximum(np.diff(rows, axis=0), 0.0)
        at = start + 1 if previous is None else start
        power = magnitude ** 2
        for b, mask in enumerate(masks):
            band_power[b, start:start + power.shape[0]] = power[:, mask].sum(axis=1)
            band_flux[b, at:at + rise.shape[0]] = rise[:, mask].sum(axis=1)
        previous = magnitude[-1]
    total_power, total_flux = float(band_power.sum()), float(band_flux.sum())
    if total_power <= 0:
        return None
    energy_onsets = np.zeros(n)
    flux_onsets = np.zeros(n)
    for power, flux in zip(band_power, band_flux):
        if power.sum() >= total_power * 1e-6:    # else nothing lives here: its peak is noise
            compressed = np.log1p(TEMPO_MU * power / power.max()) / np.log1p(TEMPO_MU)
            energy_onsets[1:] += np.maximum(np.diff(compressed), 0.0)
        if total_flux > 0 and flux.sum() >= total_flux * 1e-3:
            ceiling = float(np.percentile(flux, 99.5)) or float(flux.max())
            flux_onsets += np.minimum(flux / ceiling, 1.0)
    hop_s = hop / sample_rate
    taps = max(1, int(round(TEMPO_ENV_SMOOTH_S / hop_s)))
    if taps > 1:
        kernel = np.hanning(taps + 2)[1:-1]
        kernel = kernel / kernel.sum()
        energy_onsets = np.convolve(energy_onsets, kernel, mode="same")
        flux_onsets = np.convolve(flux_onsets, kernel, mode="same")
    return energy_onsets, flux_onsets, hop_s


def _tempo_period(onsets: np.ndarray, hop_s: float) -> Optional[tuple[int, float]]:
    """The beat period of an onset envelope as (lag in hops, confidence), or None.

    - DETRENDED: only what stands above its own 1.5 s neighbourhood counts. A mean-removed
      envelope with one level change in it (a chorus, a crescendo, a 12 dB step in pink
      noise) correlates at EVERY lag and the argmax landed on the shortest one — ~200 BPM at
      confidence 0.88 on material with no rhythm at all.
    - The autocorrelation is unbiased (a long lag has fewer products) and a candidate must be
      a local maximum INSIDE 60–200 BPM, never the edge of the range.
    - Candidates are weighed by a log-normal prior around 120 BPM: a period and its double
      are both periods of the same music, and which one is "the beat" is not in the signal.
    - A candidate in the 3:2 family of the winner takes its place when the pattern repeats
      at ITS bars and not at the winner's (TEMPO_METER_*, v13): a dotted kick figure is not
      the beat.
    - A candidate at 1/2, 3/2 or 2/3 of the winner's period (TEMPO_RIVAL_RELATIONS) scoring
      within TEMPO_RIVAL_RATIO of it caps the confidence at TEMPO_AMBIGUOUS_CONFIDENCE.
    - A pulse with no accent structure is read at its own period (TEMPO_UNIFORM_*), capped
      the same way: that, too, is an octave nobody measured."""
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks
    if onsets.max() <= 0:
        return None
    span = max(3, int(round(TEMPO_DETREND_S / hop_s)) | 1)
    env = np.maximum(onsets - uniform_filter1d(onsets, size=span, mode="nearest"), 0.0)
    if env.max() <= 0:
        return None
    env = env - env.mean()
    n = env.size
    # ceil / floor, not round: every candidate lag is a tempo INSIDE the contract range
    min_lag = max(1, int(np.ceil(60.0 / TEMPO_MAX_BPM / hop_s)))
    max_lag = int(np.floor(60.0 / TEMPO_MIN_BPM / hop_s))
    if max_lag + 1 >= n or min_lag >= max_lag:
        return None  # too short to hold even one slowest-tempo period
    spectrum = np.fft.rfft(env, n=2 * n)
    autocorr = np.fft.irfft(spectrum * np.conj(spectrum))[:n]
    if autocorr[0] <= 0:
        return None
    normalized = (autocorr / autocorr[0]) * (n / np.maximum(n - np.arange(n), 1))
    head = normalized[: max_lag + 2]
    candidates, _ = find_peaks(head)
    candidates = candidates[(candidates >= min_lag) & (candidates <= max_lag)]
    if candidates.size == 0:
        return None
    prior = np.exp(-0.5 * (np.log2(candidates * hop_s / (60.0 / TEMPO_PRIOR_BPM))
                           / TEMPO_PRIOR_OCTAVES) ** 2)
    score = head[candidates] * prior
    best = int(np.argmax(score))

    def bar_support(lag_hops: int) -> Optional[float]:
        """How well the envelope repeats at 3, 4, 6 or 8 of this period (the best of them)."""
        found = []
        for multiple in TEMPO_METER_MULTIPLES:
            at = multiple * lag_hops
            reach = max(multiple // 2 + 1, int(round(TEMPO_METER_TOLERANCE * at)))
            if at + reach >= n // 2:
                continue
            found.append(float(normalized[at - reach: at + reach + 1].max()))
        return max(found) if found else None

    own = bar_support(int(candidates[best]))
    if own is not None:
        ratio = candidates / candidates[best]
        family = np.zeros(candidates.size, dtype=bool)
        for relation in TEMPO_METER_RELATIONS:
            family |= np.abs(ratio / relation - 1.0) < 0.03
        family &= head[candidates] >= max(TEMPO_MIN_CONFIDENCE,
                                          TEMPO_METER_MIN_SHARE * float(head[candidates[best]]))
        better = [int(j) for j in np.where(family)[0]
                  if (bar_support(int(candidates[j])) or -1.0) >= own + TEMPO_METER_MARGIN]
        if better:
            best = max(better, key=lambda j: float(score[j]))
    lag = int(candidates[best])
    confidence = float(min(head[lag], 1.0))
    if confidence < TEMPO_MIN_CONFIDENCE:
        return None
    ratio = candidates / lag
    related = np.zeros(candidates.size, dtype=bool)
    for relation in TEMPO_RIVAL_RELATIONS:
        related |= np.abs(ratio / relation - 1.0) < 0.03
    if related.any() and float(score[related].max()) >= TEMPO_RIVAL_RATIO * float(score[best]):
        confidence = min(confidence, TEMPO_AMBIGUOUS_CONFIDENCE)
    halves = np.where((np.abs(ratio / 0.5 - 1.0) < 0.03)
                      & (60.0 / (candidates * hop_s) <= TEMPO_UNIFORM_MAX_BPM))[0]
    if halves.size:
        half = int(halves[np.argmax(head[candidates[halves]])])
        if head[candidates[half]] >= TEMPO_UNIFORM_RATIO * head[lag]:
            lag = int(candidates[half])
            confidence = min(float(head[lag]), TEMPO_AMBIGUOUS_CONFIDENCE)
    return lag, confidence


def _fold_power(onsets_s: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """Onset times folded at each trial period: the summed power of the first
    TEMPO_FOLD_HARMONICS harmonics of the folded train (Z-squared). A run of periods at a
    time - the memory is 32 rows of one phasor an onset."""
    power = np.zeros(periods.size)
    for start in range(0, periods.size, 32):
        chunk = periods[start:start + 32]
        unit = np.exp(2j * np.pi * onsets_s[np.newaxis, :] / chunk[:, np.newaxis])
        harmonic = unit.copy()
        for _ in range(TEMPO_FOLD_HARMONICS):
            power[start:start + chunk.size] += np.abs(harmonic.sum(axis=1)) ** 2
            harmonic *= unit
    return power


def _fold_grid(onsets_s: np.ndarray, envelope: np.ndarray, hop_s: float, frame_s: float,
               coarse_period: float) -> Optional[tuple[float, float]]:
    """(beat offset, beat period) from the precise onsets FOLDED at trial periods within one
    envelope hop of the lag grid's, or None where folding does not single out one grid - the
    least-squares fit in `_tempo` then decides, as it did before this existed.

    The period is where the fold's power peaks (TEMPO_FOLD_* say what it must clear): every
    onset on any subdivision of the beat speaks for it, and none is assigned a beat index, so
    neither a busy part nor a long take can mislead it. WHICH of the folded classes is the beat
    is where the onset ENVELOPE piles up most - a kick and a hat together outweigh a hat alone -
    and the line itself is the median of that class's precise onsets, so a vocal or a bass note
    a few milliseconds off the kick does not pull it. A least-squares fit over that class alone,
    every index now right, polishes the period; it is kept where it agrees with the fold.

    Measured: three-minute synthetic songs at 97.3-174 BPM read to 0.001 BPM and 0.3 ms where
    they read 0.2-0.5 BPM off; the two released songs read 124.001 and 123.999, mix and drum
    stem agreeing on the beat line to 1.2 ms. A song whose tempo CHANGES by more than the span
    (124 to 126 at the half) is still read at one of its tempi: one global BPM cannot say it."""
    if onsets_s.size < TEMPO_FOLD_MIN_ONSETS or coarse_period <= 0:
        return None
    onsets_s = onsets_s[onsets_s < onsets_s[0] + TEMPO_FOLD_MAX_S]
    span_s = float(onsets_s[-1] - onsets_s[0])
    lo = max(coarse_period - hop_s, 60.0 / TEMPO_MAX_BPM)
    hi = min(coarse_period + hop_s, 60.0 / TEMPO_MIN_BPM)
    if onsets_s.size < TEMPO_FOLD_MIN_ONSETS or span_s <= 0 or hi <= lo:
        return None
    step = min(coarse_period ** 2 / (TEMPO_FOLD_OVERSAMPLE * TEMPO_FOLD_HARMONICS * span_s),
               (hi - lo) / TEMPO_FOLD_MIN_STEPS)
    periods = np.linspace(lo, hi, int(np.ceil((hi - lo) / step)) + 1)
    power = _fold_power(onsets_s, periods)
    best = int(np.argmax(power))
    if best in (0, periods.size - 1):       # the peak may lie outside the span: not ours to say
        return None
    if power[best] < TEMPO_FOLD_MIN_Z * onsets_s.size * TEMPO_FOLD_HARMONICS:
        return None
    lobe_lo = lobe_hi = best                # the peak's own lobe: down to its first minimum
    while lobe_lo > 0 and power[lobe_lo - 1] < power[lobe_lo]:
        lobe_lo -= 1
    while lobe_hi < periods.size - 1 and power[lobe_hi + 1] < power[lobe_hi]:
        lobe_hi += 1
    outside = np.concatenate([power[:lobe_lo], power[lobe_hi + 1:]])
    if outside.size and float(outside.max()) > TEMPO_FOLD_MAX_RIVAL * power[best]:
        return None
    left, centre, right = power[best - 1], power[best], power[best + 1]
    denom = left - 2.0 * centre + right
    shift = 0.5 * (left - right) / denom if denom != 0.0 else 0.0
    grid_step = float(periods[1] - periods[0])
    period = float(periods[best]) + float(np.clip(shift, -0.5, 0.5)) * grid_step

    def from_line(offset_s: float, period_s: float) -> np.ndarray:
        return (onsets_s - offset_s + period_s / 2.0) % period_s - period_s / 2.0

    def on_its_class(offset_s: float, period_s: float) -> float:
        """The line moved onto the median of the precise onsets within the class's gate."""
        residual = from_line(offset_s, period_s)
        near = np.abs(residual) < min(TEMPO_FOLD_CLASS_S, period_s / 8.0)
        return offset_s + (float(np.median(residual[near])) if int(near.sum()) >= 4 else 0.0)

    # The envelope's own timing is a frame wide: it says which class, the onsets say where.
    env = envelope[: int(np.ceil((onsets_s[0] + TEMPO_FOLD_MAX_S) / hop_s))]
    times = np.arange(env.size) * hop_s + 0.5 * frame_s
    bins = TEMPO_FOLD_PHASE_BINS
    profile = np.bincount(np.floor(times % period / period * bins).astype(int) % bins,
                          weights=env, minlength=bins)
    profile = profile + np.roll(profile, 1) + np.roll(profile, -1)
    offset = on_its_class((int(np.argmax(profile)) + 0.5) / bins * period, period)

    fit_offset, fit_period = offset, period
    for _ in range(2):
        index = np.round((onsets_s - fit_offset) / fit_period)
        residual = onsets_s - (fit_offset + index * fit_period)
        keep = np.abs(residual) < min(TEMPO_FOLD_CLASS_S, fit_period / 8.0)
        if int(keep.sum()) < 8:
            break
        design = np.stack([np.ones(int(keep.sum())), index[keep]], axis=1)
        solution, *_ = np.linalg.lstsq(design, onsets_s[keep], rcond=None)
        fit_offset, fit_period = float(solution[0]), float(solution[1])
    if (abs(fit_period - period) <= grid_step
            and 60.0 / TEMPO_MAX_BPM <= fit_period <= 60.0 / TEMPO_MIN_BPM):
        period, offset = fit_period, on_its_class(fit_offset, fit_period)
    return offset % period, period


def _tempo(mono: np.ndarray, sample_rate: int
           ) -> Optional[tuple[float, float, float]]:
    """Tempo from the autocorrelation of an onset envelope (mix-analysis skill, v7; the
    detector was rewritten for reader E5.R2 — `_tempo_onset_envelopes`, `_tempo_period`).
    Returns (bpm, confidence, beat_offset_s), or None when no period in 60–200 BPM clears
    TEMPO_MIN_CONFIDENCE — unrhythmic material has no grid worth reporting.

    The band-ENERGY envelope decides. Where it finds no grid at all — a quiet click under
    loud sustained chords: every band's energy is the chords' — the band-FLUX envelope is
    asked, and its answer is always capped at TEMPO_AMBIGUOUS_CONFIDENCE (it halves a
    backbeat far more often: 9 of 33 of the reader's cases against 0).

    Measured on 184 synthetic grooves at 62–197 BPM (rock, four-on-the-floor, accented
    clicks, a kit): confidently wrong 66 -> 8 at 48 kHz and 79 -> 10 at 44.1 kHz — above
    190 BPM (half-time), and a four-on-the-floor under ~75 BPM whose off-beat hat is as
    strong as its kick (double); and 12 dB steps, crescendos and A-B-A noise read None.

    The lag grid decides the OCTAVE and the confidence; the period and the beat line inside one
    envelope hop of it are `_fold_grid`'s (analysis v12), or - where folding does not single
    out one grid - the least-squares fit's, as before."""
    front = _tempo_onset_envelopes(mono, sample_rate)
    if front is None:
        return None
    energy_onsets, flux_onsets, hop_s = front
    envelope = energy_onsets
    picked = _tempo_period(energy_onsets, hop_s)
    if picked is None:
        envelope = flux_onsets
        picked = _tempo_period(flux_onsets, hop_s)
        if picked is not None and picked[1] < TEMPO_FALLBACK_MIN_CONFIDENCE:
            picked = None
        if picked is not None:
            picked = (picked[0], min(picked[1], TEMPO_AMBIGUOUS_CONFIDENCE))
    if picked is None:
        return None
    lag, confidence = picked
    coarse_period = lag * hop_s
    beat_period = coarse_period
    peaks = _onset_peaks(mono, sample_rate)      # (frames, ITS hop) — not this envelope's
    coarse_offset_s = (float(peaks[0][0]) * peaks[1]
                       if peaks is not None and peaks[0].size else 0.0)
    beat_offset_s = coarse_offset_s

    # Refine the grid against the PRECISE onsets before anyone measures
    # milliseconds against it (2026-08-08 review). The lag grid quantises the
    # period in ~2.8 BPM steps near 120, and the offset came from the 512-hop
    # peak picker — errors of ~10 ms in phase and, worse, a period error that
    # accumulates linearly with time. Measured on a machine-perfect 120 BPM
    # click: bias_ms +21 at 15 s, +59 at 60 s, and by 120 s analyze_timing
    # reported "free time" because nothing lined up any more.
    #
    # One least-squares fit of onset_time = offset + index * period over the
    # onsets that sit near a beat (inliers only, so off-beat playing does not
    # drag the grid), iterated a few times. Off-beat-heavy material simply keeps
    # the coarse estimate.
    def _misfit(offset: float, period: float) -> float:
        """Median distance from an onset to its nearest grid line, in beats."""
        if period <= 0:
            return float("inf")
        pos = (precise - offset) / period
        return float(np.median(np.abs(pos - np.round(pos))))

    precise = _onset_times_precise(mono, sample_rate)
    if precise.size < TEMPO_MIN_ONSETS:
        return None                 # a period in the envelope, and nothing was ever struck
    # First the fold (v12): it reads the period off EVERY onset, on any subdivision, and needs
    # no beat index - so a busy part or a long take cannot mislead it as they did the fit
    # below, which scores a grid against beat lines alone. Where the fold does not single out
    # one grid, that fit decides, exactly as it did.
    folded = _fold_grid(precise, envelope, hop_s,
                        TEMPO_FRAME * _rate_factor(sample_rate) / sample_rate, coarse_period)
    if folded is not None:
        beat_offset_s, beat_period = folded
    elif precise.size >= 4 and beat_period > 0:
        best_offset, best_period = beat_offset_s, beat_period
        best_misfit = _misfit(best_offset, best_period)
        offset, period = beat_offset_s, beat_period
        for _ in range(4):
            index = np.round((precise - offset) / period)
            residual = precise - (offset + index * period)
            keep = np.abs(residual) < 0.25 * period
            if int(keep.sum()) < 4:
                break
            design = np.stack([np.ones(int(keep.sum())), index[keep]], axis=1)
            solution, *_ = np.linalg.lstsq(design, precise[keep], rcond=None)
            offset, period = float(solution[0]), float(solution[1])
            # Reject a fit that has wandered onto a different grid entirely.
            if not (0.5 * beat_period < period < 2.0 * beat_period):
                break
            # …and only KEEP a refinement that genuinely fits better. Over a long
            # take a single mis-assigned index at one end drags the regression
            # onto a slightly wrong period, which then fits worse everywhere; the
            # coarse estimate is the safer answer in that case.
            misfit = _misfit(offset, period)
            if misfit < best_misfit:
                best_offset, best_period, best_misfit = offset, period, misfit
        beat_offset_s, beat_period = best_offset, best_period
        # The refined period is a tempo too: it stays inside the contract range (the reader
        # saw 59.842, 200.152 and 203.587 BPM returned) or the coarse one is kept.
        if not (60.0 / TEMPO_MAX_BPM <= beat_period <= 60.0 / TEMPO_MIN_BPM):
            beat_offset_s, beat_period = coarse_offset_s, coarse_period
        if beat_period > 0:
            beat_offset_s = float(beat_offset_s % beat_period)

    # Keep more precision than the display needs: 0.1 BPM is a 50 ms drift over
    # two minutes, which is larger than the leans this grid is used to measure.
    return (round(60.0 / beat_period, 3), round(confidence, 2),
            round(beat_offset_s, 4))


def _pitch_stft(mono: np.ndarray, sample_rate: int
                ) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Shared front-end for the pitch analyzers (Phase 30 key, Phase 31 tuning):
    scipy STFT magnitude restricted to KEY_LO..KEY_HI, keeping only the frames
    within KEY_FRAME_GATE_DB of the loudest. Returns (band_freqs, mag) where
    mag is (n_band_freq, n_kept_frame), or None when too short/quiet."""
    mono, sample_rate = _at_base_rate(mono, sample_rate)   # 8192 samples mean ~5.9 Hz a bin
    if mono.shape[0] < KEY_NPERSEG:
        return None
    freqs = np.fft.rfftfreq(KEY_NPERSEG, 1.0 / sample_rate)
    band = (freqs >= KEY_LO_HZ) & (freqs <= KEY_HI_HZ)
    if not band.any():
        return None
    # only the band's rows are kept from each block of spectra (E5·R4): (n_band_freq, n_frame)
    mag = np.ascontiguousarray(np.concatenate(
        [block[:, band] for block in _stft_magnitudes(mono, KEY_NPERSEG, KEY_NPERSEG // 2)],
        axis=0).T)
    frame_energy = mag.sum(axis=0)
    if frame_energy.max() <= 0:
        return None
    keep = frame_energy > frame_energy.max() * 10 ** (KEY_FRAME_GATE_DB / 20.0)
    return freqs[band], mag[:, keep]


def _chroma(mono: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
    """12-bin pitch-class profile (Phase 30, mix-analysis skill): scipy STFT
    magnitude over KEY_LO..KEY_HI folded into pitch classes, summed over
    energy-bearing frames and sum-normalized. None when too short/quiet."""
    front = _pitch_stft(mono, sample_rate)
    if front is None:
        return None
    band_freqs, mag = front
    pc = np.mod(np.round(12.0 * np.log2(band_freqs / C0_HZ)).astype(int), 12)
    weight = np.log1p(mag)                          # log-compress; tame transients
    chroma = np.array([weight[pc == k].sum() for k in range(12)])
    total = chroma.sum()
    if total <= 0:
        return None
    return chroma / total


def _tuning(mono: np.ndarray, sample_rate: int
            ) -> Optional[tuple[float, float]]:
    """Global tuning offset + intonation coherence (Phase 31, mix-analysis
    skill). Reuses the pitch STFT front-end: magnitude-weighted spectral peaks
    (parabolically interpolated in frequency) contribute their cents-deviation
    from the nearest equal-tempered note to a histogram; the histogram mode is
    the global offset from A440 (±50¢), the weighted mass within
    ±TUNING_COHERENCE_CENTS of it is the coherence (0..1). Returns
    (offset_cents, coherence), or None on unpitched material (too few peaks or
    no tonal concentration). Deterministic, pure numpy/scipy (no librosa)."""
    front = _pitch_stft(mono, sample_rate)
    if front is None:
        return None
    band_freqs, mag = front
    if band_freqs.shape[0] < 3:                     # need a neighbor each side
        return None
    # Per-frame spectral peaks: interior local maxima above a per-frame floor.
    interior = mag[1:-1, :]
    thresh = mag.max(axis=0) * 10 ** (TUNING_PEAK_REL_DB / 20.0)
    is_peak = ((interior > mag[:-2, :]) & (interior >= mag[2:, :])
               & (interior > thresh[np.newaxis, :]))
    fi, ti = np.where(is_peak)                      # fi indexes `interior` rows
    if fi.size < TUNING_MIN_PEAKS:
        return None
    if not _has_steady_partials(mag, is_peak):      # bursts: a kick, a click (E5·R6)
        return None
    center = mag[fi + 1, ti]
    left, right = mag[fi, ti], mag[fi + 2, ti]
    denom = left - 2.0 * center + right
    shift = np.clip(np.where(denom != 0.0, 0.5 * (left - right) / denom, 0.0),
                    -0.5, 0.5)
    bin_hz = band_freqs[1] - band_freqs[0]          # STFT bins are linear in Hz
    peak_freq = band_freqs[fi + 1] + shift * bin_hz
    peak_semi = 12.0 * np.log2(peak_freq / C0_HZ)   # semitones above C0
    cents = ((peak_semi + 0.5) % 1.0 - 0.5) * 100.0  # nearest-note residual, [-50,50]
    hist, edges = np.histogram(cents, bins=TUNING_HIST_BINS, range=(-50.0, 50.0),
                               weights=center)
    total = hist.sum()
    if total <= 0 or hist.max() < TUNING_MIN_PEAKINESS * (total / TUNING_HIST_BINS):
        return None                                 # flat distribution ⇒ no tuning center
    centers = (edges[:-1] + edges[1:]) / 2.0
    peak = int(np.argmax(hist))
    offset = float(centers[peak])
    if 0 < peak < TUNING_HIST_BINS - 1:             # parabolic sub-bin refine of the mode
        l, c, r = hist[peak - 1], hist[peak], hist[peak + 1]
        d = l - 2.0 * c + r
        if d != 0:
            offset += float(np.clip(0.5 * (l - r) / d, -0.5, 0.5)
                            * (centers[1] - centers[0]))
    # coherence: weighted mass within ±window of the mode, circular over 100¢
    diff = np.abs(cents - offset)
    near = np.minimum(diff, 100.0 - diff) <= TUNING_COHERENCE_CENTS
    coherence = float(center[near].sum() / total)
    return round(offset, 1), round(coherence, 2)


def _has_steady_partials(mag: np.ndarray, is_peak: np.ndarray) -> bool:
    """Whether enough of the spectral peaks are STEADY partials for a tuning to be read from
    them (E5·R6): at least TUNING_MIN_PEAKS of them, holding TUNING_MIN_STEADY_SHARE of all
    peak magnitude. `mag` is `_pitch_stft`'s (bins, frames); `is_peak` marks its interior
    rows. A gate only - the histogram is still built from every peak, so a tuning that is
    reported is the number it was."""
    rows = mag.shape[0]
    reach = TUNING_STEADY_BINS
    padded = np.pad(mag, ((reach, reach), (0, 0)), mode="edge")
    interior = mag[1:-1, :]
    below = padded[:rows][1:-1]                     # mag[bin - reach]
    above = padded[2 * reach:][1:-1]                # mag[bin + reach]
    steady = (is_peak & (below <= TUNING_STEADY_RATIO * interior)
              & (above <= TUNING_STEADY_RATIO * interior))
    if int(steady.sum()) < TUNING_MIN_PEAKS:
        return False
    total = float((interior * is_peak).sum())
    return total > 0.0 and float((interior * steady).sum()) / total >= TUNING_MIN_STEADY_SHARE


def _sounding_pitch_classes(mono: np.ndarray, sample_rate: int) -> int:
    """How many pitch classes each hold KEY_PITCH_CLASS_SHARE of the band's POWER. The chroma
    `_key` correlates is log-compressed, which is right for a profile and wrong for counting:
    there a cadence shows three or four classes over 10 % and so does one note."""
    front = _pitch_stft(mono, sample_rate)
    if front is None:
        return 0
    band_freqs, mag = front
    pc = np.mod(np.round(12.0 * np.log2(band_freqs / C0_HZ)).astype(int), 12)
    power = np.square(mag).sum(axis=1)
    by_class = np.array([power[pc == k].sum() for k in range(12)])
    total = by_class.sum()
    return int((by_class >= KEY_PITCH_CLASS_SHARE * total).sum()) if total > 0 else 0


def _bass_pitch_classes(mono: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
    """The share of the lowest register's power on each pitch class (KEY_BASS_LO_HZ..HI_HZ, the
    fundamentals of a bass line), read at a resolution that tells semitones apart down there:
    the key STFT's 5.9 Hz bins cannot, below about 100 Hz. None when there is no such register
    (a part too short, or one whose band there holds nothing)."""
    mono, sample_rate = _at_base_rate(mono, sample_rate)
    low = resample_poly(mono, 1, KEY_BASS_DECIMATE)
    n = KEY_BASS_NPERSEG
    if low.shape[0] < n:
        return None
    rate = sample_rate / KEY_BASS_DECIMATE
    window = np.hanning(n)
    power = np.zeros(n // 2 + 1)
    for start in range(0, low.shape[0] - n + 1, n // 2):
        power += np.square(np.abs(np.fft.rfft(low[start:start + n] * window)))
    freqs = np.fft.rfftfreq(n, 1.0 / rate)
    band = (freqs >= KEY_BASS_LO_HZ) & (freqs <= KEY_BASS_HI_HZ)
    total = float(power[1:].sum())
    in_band = float(power[band].sum())
    # a register that is there: a hundredth of what the decimated part holds (it reaches 1.5 kHz)
    if total <= 0.0 or in_band < 0.01 * total:
        return None
    pc = np.mod(np.round(12.0 * np.log2(freqs[band] / C0_HZ)).astype(int), 12)
    by_class = np.array([power[band][pc == k].sum() for k in range(12)])
    return by_class / in_band


def _relative_key(tonic: int, mode: str) -> tuple[int, str]:
    return ((tonic + 9) % 12, "min") if mode == "maj" else ((tonic + 3) % 12, "maj")


def _key(mono: np.ndarray, sample_rate: int) -> Optional[KeyEstimate]:
    """Musical key via chroma vs Krumhansl-Schmuckler profiles (Phase 30,
    mix-analysis skill): the best Pearson correlation across all 12 rotations
    of the major and minor profiles names (tonic, mode). None below
    KEY_MIN_CONFIDENCE — atonal/percussive material has no key worth reporting —
    and None for a guess (reader E5·R6): fewer than KEY_MIN_PITCH_CLASSES sounding pitch
    classes, or a fit that does not hold without its loudest class
    (KEY_MIN_REST_CORRELATION). A reported key's tonic, mode and confidence are unchanged."""
    chroma = _chroma(mono, sample_rate)
    if chroma is None or chroma.std() == 0:
        return None
    best_corr, best = -2.0, None
    for mode, profile in (("maj", KRUMHANSL_MAJOR), ("min", KRUMHANSL_MINOR)):
        for tonic in range(12):
            corr = float(np.corrcoef(chroma, np.roll(profile, tonic))[0, 1])
            if corr > best_corr:
                best_corr, best = corr, (tonic, mode)
    if best is None or best_corr < KEY_MIN_CONFIDENCE:
        return None
    rest = np.ones(12, dtype=bool)
    rest[int(np.argmax(chroma))] = False
    profile = np.roll(KRUMHANSL_MAJOR if best[1] == "maj" else KRUMHANSL_MINOR, best[0])
    if chroma[rest].std() == 0:
        return None
    if float(np.corrcoef(chroma[rest], profile[rest])[0, 1]) < KEY_MIN_REST_CORRELATION:
        return None
    if _sounding_pitch_classes(mono, sample_rate) < KEY_MIN_PITCH_CLASSES:
        return None
    tonic, mode = best
    # A near-tie with the relative key is settled by the bass (see KEY_RELATIVE_MARGIN).
    other_tonic, other_mode = _relative_key(tonic, mode)
    other_corr = float(np.corrcoef(
        chroma, np.roll(KRUMHANSL_MAJOR if other_mode == "maj" else KRUMHANSL_MINOR,
                        other_tonic))[0, 1])
    if best_corr - other_corr < KEY_RELATIVE_MARGIN and other_corr >= KEY_MIN_CONFIDENCE:
        bass = _bass_pitch_classes(mono, sample_rate)
        if (bass is not None and bass[other_tonic] >= KEY_BASS_SHARE
                and bass[other_tonic] >= KEY_BASS_RATIO * bass[tonic]):
            tonic, mode, best_corr = other_tonic, other_mode, other_corr
    return KeyEstimate(tonic=PITCH_CLASSES[tonic], mode=mode,
                       confidence=round(best_corr, 2))


def _transient_density(audio: np.ndarray, sample_rate: int) -> Optional[float]:
    """Onsets per second (see _onset_peaks)."""
    mono = audio.mean(axis=1)
    onsets = _onset_peaks(mono, sample_rate)
    if onsets is None:
        return None
    peaks, _ = onsets
    return round(len(peaks) / (mono.shape[0] / sample_rate), 2)


def _transient_energy_ratio(audio: np.ndarray, sample_rate: int) -> Optional[float]:
    """Share of total energy inside the 50 ms windows following onsets:
    1.0 = all attack, 0.0 = all sustain (or no onsets)."""
    mono = audio.mean(axis=1)
    onsets = _onset_peaks(mono, sample_rate)
    if onsets is None:
        return None
    peaks, hop_s = onsets
    total = float(np.sum(mono ** 2))
    if total <= 0:
        return None
    if peaks.size == 0:
        return 0.0
    window = int(TRANSIENT_WINDOW_S * sample_rate)
    mask = np.zeros(mono.shape[0], dtype=bool)
    for frame in peaks:
        start = int(frame * hop_s * sample_rate)
        mask[start:start + window] = True
    return round(float(np.sum(mono[mask] ** 2) / total), 2)


def _sibilance_density(audio: np.ndarray, sample_rate: int) -> Optional[float]:
    """Fraction of active 100 ms frames where the 5-10 kHz band holds more than
    25 % of the frame's power. Meaningful on vocal-ish sources; just a number
    elsewhere (mix-analysis skill)."""
    mono = audio.mean(axis=1)
    frame = int(SIBILANCE_FRAME_S * sample_rate)
    n_frames = mono.shape[0] // frame
    if n_frames < 3:
        return None
    frames = mono[: n_frames * frame].reshape(n_frames, frame)
    rms_db = 20.0 * np.log10(
        np.maximum(np.sqrt(np.mean(frames ** 2, axis=1)), SILENCE_PEAK))
    active = frames[rms_db > MASKING_GATE_DBFS]
    if active.shape[0] < 3:
        return None
    power = np.abs(np.fft.rfft(active, axis=1)) ** 2
    freqs = np.fft.rfftfreq(frame, d=1.0 / sample_rate)
    sib_band = (freqs >= SIBILANCE_LO_HZ) & (freqs < SIBILANCE_HI_HZ)
    share = power[:, sib_band].sum(axis=1) / np.maximum(power.sum(axis=1), SILENCE_PEAK)
    return round(float(np.mean(share > SIBILANCE_SHARE)), 2)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive/exclusive index runs where mask is true."""
    runs: list[tuple[int, int]] = []
    start: Optional[int] = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, mask.size))
    return runs


def _pad_min_span(start_s: float, end_s: float, duration_s: float,
                  min_span_s: float = PHRASE_MIN_SPAN_S) -> tuple[float, float]:
    """Pad a range to the engine's minimum ranged-action span, shifting inward
    at file edges so the final span is valid whenever the file is long enough."""
    if end_s - start_s >= min_span_s:
        return start_s, end_s
    if duration_s <= min_span_s:
        return 0.0, duration_s
    pad = (min_span_s - (end_s - start_s)) / 2.0
    start_s = max(0.0, start_s - pad)
    end_s = min(duration_s, end_s + pad)
    if end_s - start_s < min_span_s:
        if start_s <= 0.0:
            end_s = min(duration_s, min_span_s)
        elif end_s >= duration_s:
            start_s = max(0.0, duration_s - min_span_s)
    return start_s, end_s


def _phrase_level_moves(audio: np.ndarray,
                        sample_rate: int) -> tuple[list[PhraseLevelMove], bool]:
    """Local level outliers over the same 400 ms / 100 ms frame grid as
    dynamic_spread_db. Returns suggested ranged gain moves toward the active
    median. These are ears, not mutations."""
    mono = audio.mean(axis=1)
    frame = max(1, int(RMS_FRAME_S * sample_rate))
    hop = max(1, int(RMS_HOP_S * sample_rate))
    if mono.shape[0] < frame:
        frame = mono.shape[0]
    if frame == 0:
        return [], True
    starts = np.arange(0, mono.shape[0] - frame + 1, hop)
    if starts.size == 0:
        return [], True
    levels = np.array([
        20.0 * np.log10(max(
            float(np.sqrt(np.mean(mono[s:s + frame] ** 2))), SILENCE_PEAK))
        for s in starts
    ])
    active = levels > MASKING_GATE_DBFS
    if active.sum() < 3:
        return [], True
    median = float(np.median(levels[active]))
    candidates: list[PhraseLevelMove] = []
    for reason, mask in (
        ("too_quiet", active & (levels < median - PHRASE_LEVEL_TOLERANCE_DB)),
        ("too_loud", active & (levels > median + PHRASE_LEVEL_TOLERANCE_DB)),
    ):
        for start_i, end_i in _runs(mask):
            start_s = starts[start_i] / sample_rate
            end_s = (starts[end_i - 1] + frame) / sample_rate
            start_s, end_s = _pad_min_span(
                start_s, end_s, mono.shape[0] / sample_rate)
            level = float(np.mean(levels[start_i:end_i]))
            delta = float(np.clip(median - level, -PHRASE_MAX_GAIN_DB,
                                  PHRASE_MAX_GAIN_DB))
            if abs(delta) < 0.5:
                continue
            candidates.append(PhraseLevelMove(
                start_s=round(start_s, 2),
                end_s=round(end_s, 2),
                delta_db=round(delta, 1),
                level_dbfs=round(level, 1),
                reason=reason))
    candidates.sort(key=lambda move: abs(move.delta_db), reverse=True)
    return candidates[:PHRASE_MAX_MOVES], False


def _sibilance_bursts(audio: np.ndarray,
                      sample_rate: int) -> tuple[list[SibilanceBurst], bool]:
    """Local 5-10 kHz bursts over 100 ms frames. The suggested threshold is a
    conservative de-ess threshold relative to the burst RMS."""
    mono = audio.mean(axis=1)
    frame = max(1, int(SIBILANCE_FRAME_S * sample_rate))
    n_frames = mono.shape[0] // frame
    if n_frames < 3:
        return [], True
    frames = mono[: n_frames * frame].reshape(n_frames, frame)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    rms_db = 20.0 * np.log10(np.maximum(rms, SILENCE_PEAK))
    active = rms_db > MASKING_GATE_DBFS
    if active.sum() < 3:
        return [], True
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(frame, d=1.0 / sample_rate)
    sib_band = (freqs >= SIBILANCE_LO_HZ) & (freqs < SIBILANCE_HI_HZ)
    sib_power = power[:, sib_band].sum(axis=1)
    total_power = np.maximum(power.sum(axis=1), SILENCE_PEAK)
    share = sib_power / total_power
    mask = active & (share > SIBILANCE_BURST_SHARE)
    bursts: list[SibilanceBurst] = []
    for start_i, end_i in _runs(mask):
        start_s = start_i * SIBILANCE_FRAME_S
        end_s = end_i * SIBILANCE_FRAME_S
        start_s, end_s = _pad_min_span(
            start_s, end_s, mono.shape[0] / sample_rate)
        region_power = power[start_i:end_i][:, sib_band].sum(axis=0)
        if region_power.sum() <= 0:
            freq_hz = 7000
        else:
            centroid = float(np.average(freqs[sib_band],
                                         weights=region_power))
            # DeessAction.freq_hz is the split point, not the whistling peak
            # itself. Put the split below the centroid so the burst lands in
            # the compressed high band.
            freq_hz = int(round(float(np.clip(centroid - 1500.0,
                                              SIBILANCE_LO_HZ, 9000.0))
                                / 100.0) * 100)
        threshold = float(np.clip(np.mean(rms_db[start_i:end_i]) - 6.0,
                                  -50.0, -12.0))
        bursts.append(SibilanceBurst(
            start_s=round(start_s, 2),
            end_s=round(end_s, 2),
            density=round(float(np.mean(share[start_i:end_i])), 2),
            freq_hz=freq_hz,
            threshold_db=round(threshold, 1)))
    bursts.sort(key=lambda burst: burst.density, reverse=True)
    return bursts[:SIBILANCE_MAX_BURSTS], False


# ---------- per-note pitch (2026-08-08) ----------

NOTE_FRAME = 2048            # ~46 ms at 44.1 k — three periods of the lowest f0
NOTE_HOP = 441               # 10 ms
NOTE_F0_MIN_HZ = 65.0        # ~C2; below this the frame holds too few periods
NOTE_F0_MAX_HZ = 1200.0
NOTE_PERIODICITY = 0.5       # normalised autocorrelation below this = unvoiced
NOTE_OCTAVE_TOLERANCE = 0.85  # accept the FIRST peak within this of the best
NOTE_MIN_FRAMES = 8          # 80 ms — shorter runs are transitions, not notes
NOTE_MAX_REPORTED = 8
NOTE_PLAYED_MAX = 7          # pitch classes listed in `played`: a scale's worth
# The monophony gate (2026-08-08). Autocorrelation answers a CHORD with the
# virtual fundamental, so polyphony used to be reported as a confident note one
# or two octaves below anything played, with reason=None. A lone voice's energy
# sits on the harmonic series of one f0; a chord's does not.
NOTE_HARMONIC_PARTIALS = 8   # partials the presence test looks across
NOTE_HARMONIC_TOL = 0.03     # ±3 % around each partial (≈ ±50 cents)
# Deliberately low. Measured: real monophony 1.00, polyphony 0.003-0.009. A real
# vowel can put more energy in its 2nd partial than its 1st, so the gate fires
# only when the fundamental is essentially ABSENT (the virtual-fundamental
# signature), never merely weak.
NOTE_MIN_FUNDAMENTAL = 0.05
# Below NOTE_F0_MIN_HZ a part is not tracked - and is told THAT, not that it is polyphonic
# (reader E5·R9): frames periodic at a lag between the floor's and this one's are a pitch
# under the floor. ~31 Hz is B0, the low string of a five-string bass.
NOTE_SUBFLOOR_MIN_HZ = 31.0
NOTE_SUBFLOOR_SHARE = 0.5    # of the frames that carry signal


def _f0_track(mono: np.ndarray, sample_rate: int
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frame-wise f0 in Hz via normalised autocorrelation (0 = unvoiced), the
    frame start times, and the harmonic SHARE of each frame. Pure numpy/scipy —
    no librosa, deterministic.

    The share is the energy at the f0's OWN first partial relative to its
    strongest partial — i.e. "is this fundamental actually there?". A C4-E4-G4
    triad has no energy at C2, yet autocorrelation answers it with exactly that
    VIRTUAL fundamental (the common sub-period), which is how a triad came to be
    reported as a C2 "3.8 cents sharp" with reason=None (2026-08-08 review).

    Note the discarded first attempt: measuring how much total energy sits on
    the f0's harmonic SERIES does not separate them at all (mono 0.77-0.90 vs
    poly 0.71-0.85, overlapping), because a consonant chord's virtual
    fundamental has a series that covers the chord tones — which is precisely
    why the autocorrelation locks onto it. Presence of the fundamental itself
    separates cleanly (mono 1.00 vs poly 0.003-0.009). It costs nothing: the FFT
    it needs is the one already taken for the autocorrelation.

    Octave errors are autocorrelation's signature failure: the lag at TWICE the
    true period correlates just as well, so "highest peak" regularly reports a
    note an octave down. Taking the FIRST peak that comes within
    NOTE_OCTAVE_TOLERANCE of the best is the standard defence, and it matters
    here because an octave error is not a small error — it is a wrong note.
    """
    from scipy.signal import find_peaks
    # NOTE_FRAME / NOTE_HOP are sample counts at the base rate: at 96 k a 2048-sample frame
    # held 1.4 periods of a 65 Hz note and at 192 k found no note at all (reader E5.R3).
    mono, sample_rate = _at_base_rate(mono, sample_rate)
    if mono.shape[0] < NOTE_FRAME * 2:
        return np.array([]), np.array([]), np.array([])
    starts = np.arange(0, mono.shape[0] - NOTE_FRAME + 1, NOTE_HOP)
    min_lag = max(2, int(sample_rate / NOTE_F0_MAX_HZ))
    max_lag = min(NOTE_FRAME - 1, int(sample_rate / NOTE_F0_MIN_HZ))
    if min_lag >= max_lag:
        return np.array([]), np.array([]), np.array([])

    bin_freqs = np.fft.rfftfreq(2 * NOTE_FRAME, 1.0 / sample_rate)
    bin_hz = float(sample_rate) / (2 * NOTE_FRAME)
    overlap = NOTE_FRAME / (NOTE_FRAME - np.arange(NOTE_FRAME, dtype=np.float64))
    f0 = np.zeros(starts.size)
    share = np.zeros(starts.size)
    for i, start in enumerate(starts):
        frame = mono[start:start + NOTE_FRAME]
        frame = frame - frame.mean()
        energy = float(np.dot(frame, frame))
        if energy <= 0 or np.sqrt(energy / NOTE_FRAME) < SILENCE_PEAK:
            continue
        # Autocorrelation through the FFT, normalised by lag 0 AND by how many samples
        # overlap at each lag. Without the second (reader E5·R9) the correlation fell with
        # the lag, which pulled every peak toward a shorter one - by more the longer the
        # period: E2-B2 read ~4 cents sharp, a note 20 cents flat read 16.
        spectrum = np.fft.rfft(frame, n=2 * NOTE_FRAME)
        corr = np.fft.irfft(spectrum * np.conj(spectrum))[:NOTE_FRAME]
        if corr[0] <= 0:
            continue
        norm = corr / corr[0] * overlap
        window = norm[min_lag:max_lag + 1]
        peaks, _ = find_peaks(window)
        if peaks.size == 0:
            continue
        best = float(window[peaks].max())
        if best < NOTE_PERIODICITY:
            continue
        lag = int(peaks[np.argmax(window[peaks] >= NOTE_OCTAVE_TOLERANCE * best)])
        lag += min_lag
        if not (min_lag < lag < max_lag):
            f0[i] = sample_rate / lag
            continue
        # Parabolic vertex for sub-sample lag → sub-cent pitch resolution.
        left, centre, right = norm[lag - 1], norm[lag], norm[lag + 1]
        denom = left - 2.0 * centre + right
        shift = 0.0 if denom == 0 else float(np.clip(0.5 * (left - right) / denom,
                                                     -0.5, 0.5))
        f0[i] = sample_rate / (lag + shift)

        # Is this f0 REAL, or the virtual fundamental of a chord? (see docstring)
        power = np.abs(spectrum) ** 2
        amps = []
        for k in range(1, NOTE_HARMONIC_PARTIALS + 1):
            f = f0[i] * k
            if f >= bin_freqs[-1]:
                break
            # At least one bin wide, so a low f0 on coarse bins is not
            # penalised for the FFT's resolution rather than its content.
            tol = max(1.5 * bin_hz, f * NOTE_HARMONIC_TOL)
            amps.append(float(power[np.abs(bin_freqs - f) <= tol].sum()))
        if len(amps) >= 3:
            peak = max(amps)
            share[i] = amps[0] / peak if peak > 0 else 0.0
    return f0, starts / sample_rate, share


def _plays_below_the_note_floor(mono: np.ndarray, sample_rate: int) -> bool:
    """Whether a part nothing tracked is PERIODIC under NOTE_F0_MIN_HZ: at least
    NOTE_SUBFLOOR_SHARE of its sounding frames correlate (NOTE_PERIODICITY, as a tracked
    frame must) at a lag between the floor's period and NOTE_SUBFLOOR_MIN_HZ's. Every fourth
    frame is enough for a yes or no."""
    from scipy.signal import find_peaks
    mono, sample_rate = _at_base_rate(mono, sample_rate)
    if mono.shape[0] < NOTE_FRAME * 2:
        return False
    first = int(sample_rate / NOTE_F0_MIN_HZ) + 1
    last = min(NOTE_FRAME - NOTE_FRAME // 4, int(sample_rate / NOTE_SUBFLOOR_MIN_HZ))
    if first >= last:
        return False
    overlap = NOTE_FRAME / (NOTE_FRAME - np.arange(NOTE_FRAME, dtype=np.float64))
    sounding = periodic = 0
    for start in range(0, mono.shape[0] - NOTE_FRAME + 1, NOTE_HOP * 4):
        frame = mono[start:start + NOTE_FRAME]
        frame = frame - frame.mean()
        energy = float(np.dot(frame, frame))
        if energy <= 0 or np.sqrt(energy / NOTE_FRAME) < SILENCE_PEAK:
            continue
        sounding += 1
        spectrum = np.fft.rfft(frame, n=2 * NOTE_FRAME)
        corr = np.fft.irfft(spectrum * np.conj(spectrum))[:NOTE_FRAME]
        window = (corr / corr[0] * overlap)[first:last + 1]
        peaks, _ = find_peaks(window)
        if peaks.size and float(window[peaks].max()) >= NOTE_PERIODICITY:
            periodic += 1
    return sounding > 0 and periodic >= NOTE_SUBFLOOR_SHARE * sounding


def analyze_notes(audio, sample_rate: int, stem_id: str,
                  max_notes: Optional[int] = None) -> NoteAnalysis:
    """Per-note intonation for a monophonic stem (contract: mix-analysis skill).

    `max_notes` overrides the NOTE_MAX_REPORTED payload cap — the CORRECTION
    path (clip.pitch_correct, 2026-08-08) needs every off note on the clip,
    not the eight worst; the chat readout keeps the default.
    """
    audio = _frames_channels(audio)
    if np.max(np.abs(audio)) < SILENCE_PEAK:
        return NoteAnalysis(stem_id=stem_id, reason="silent")

    mono = audio.mean(axis=1)
    f0, times, fundamental = _f0_track(mono, sample_rate)
    below_floor = (f"this part plays below the {NOTE_F0_MIN_HZ:.0f} Hz tracking floor "
                   "(about C2) — a bass's lowest octave cannot be measured note by "
                   "note. Nothing here says the source is polyphonic or out of tune")
    if f0.size == 0 or not np.any(f0 > 0):
        if _plays_below_the_note_floor(mono, sample_rate):
            return NoteAnalysis(stem_id=stem_id, reason=below_floor)
        return NoteAnalysis(
            stem_id=stem_id,
            reason="no stable pitch to track — this needs a MONOPHONIC source "
                   "(lead vocal, bass, single-line lead); use analyze_chords "
                   "for harmony")

    voiced = f0 > 0
    # Decline polyphony instead of naming its virtual fundamental. Reporting a
    # confident "C2, 3.8 cents sharp" for a C major triad is worse than saying
    # nothing, and the contract already promises a reason here.
    if float(np.median(fundamental[voiced])) < NOTE_MIN_FUNDAMENTAL:
        return NoteAnalysis(
            stem_id=stem_id,
            reason="this stem sounds POLYPHONIC — the pitch that tracks is a "
                   "virtual fundamental of more than one note, not a note "
                   "anyone played. Per-note intonation needs a monophonic "
                   "source; use analyze_chords for harmony")
    midi = np.zeros_like(f0)
    midi[voiced] = 69.0 + 12.0 * np.log2(f0[voiced] / 440.0)
    semitone = np.where(voiced, np.round(midi), np.nan)

    centre = _tuning(mono, sample_rate)
    centre_cents = centre[0] if centre is not None else 0.0

    events: list[NoteEvent] = []
    i = 0
    while i < semitone.size:
        if not voiced[i]:
            i += 1
            continue
        j = i
        while (j + 1 < semitone.size and voiced[j + 1]
               and semitone[j + 1] == semitone[i]):
            j += 1
        if (j - i + 1) >= NOTE_MIN_FRAMES:
            cents = (midi[i:j + 1] - semitone[i]) * 100.0
            # Median, so symmetric vibrato does not move the reported pitch;
            # the movement itself is reported separately as spread.
            cents_off = float(np.median(cents))
            note = int(semitone[i])
            events.append(NoteEvent(
                start_s=round(float(times[i]), 2),
                end_s=round(float(times[j])
                            + NOTE_FRAME * _rate_factor(sample_rate) / sample_rate, 2),
                midi_note=note,
                note_name=f"{PITCH_CLASSES[note % 12]}{note // 12 - 1}",
                cents_off=round(cents_off, 1),
                deviation_cents=round(cents_off - centre_cents, 1),
                spread_cents=round(float(np.percentile(cents, 90)
                                         - np.percentile(cents, 10)), 1)))
        i = j + 1

    if not events:
        # A part under the floor is told the floor here too (user QA 2026-10-08, D51). The
        # check above is reached only when NOTHING tracked, and a real bass always has a few
        # frames that do - a passing note above C2, an octave error - so a 55 Hz bass loop was
        # told it "may be polyphonic, percussive, or heavily slurred".
        if _plays_below_the_note_floor(mono, sample_rate):
            return NoteAnalysis(stem_id=stem_id, reason=below_floor)
        return NoteAnalysis(
            stem_id=stem_id,
            stem_tuning_offset_cents=(round(centre_cents, 1)
                                      if centre is not None else None),
            reason="no note held long enough to measure — the source may be "
                   "polyphonic, percussive, or heavily slurred")

    ranked = sorted(events, key=lambda n: abs(n.deviation_cents), reverse=True)
    return NoteAnalysis(
        stem_id=stem_id,
        stem_tuning_offset_cents=(round(centre_cents, 1)
                                  if centre is not None else None),
        notes=ranked[:(max_notes if max_notes is not None else NOTE_MAX_REPORTED)],
        notes_found=len(events),
        played=_notes_played(events))


def _notes_played(events: list[NoteEvent]) -> list[NotePlayed]:
    """Each pitch class the part holds and for how long, longest first - over every note, not
    the capped worst-tuned list. What a bass sits on is its song's home note far more often
    than a key name read off the whole mix says."""
    held: dict[int, float] = {}
    for note in events:
        held[note.midi_note % 12] = held.get(note.midi_note % 12, 0.0) + max(0.0, note.end_s - note.start_s)
    total = sum(held.values())
    if total <= 0.0:
        return []
    ranked = sorted(held.items(), key=lambda kv: (-kv[1], kv[0]))[:NOTE_PLAYED_MAX]
    return [NotePlayed(note=PITCH_CLASSES[pc], seconds=round(seconds, 1),
                       share=round(seconds / total, 2)) for pc, seconds in ranked]


# ---------- chord detection (2026-08-08) ----------

# Triads AND sevenths. Starting with triads alone was worse, not simpler: a
# Cmaj7 (C E G B) has three of its four notes in common with Em (E G B), the
# vectors are near-identical, and the tie broke toward Em — reporting the WRONG
# ROOT, which is the one error a musician cannot work around. With the seventh
# templates present the four-note chord matches itself. The cost is the mild,
# opposite error (a triad with a passing seventh reads as a seventh chord),
# which keeps the root right. Still not modelled, and declared as limits:
# diminished/augmented, suspensions, 9ths and up, and inversions — chroma
# cannot see which note is in the bass, so C/E and C are the same vector.
CHORD_TEMPLATES = {"maj": (0, 4, 7), "min": (0, 3, 7),
                   "maj7": (0, 4, 7, 11), "min7": (0, 3, 7, 10),
                   "dom7": (0, 4, 7, 10)}
# (scale step, triad quality, seventh quality) per degree, for the key prior.
_DIATONIC = {"maj": ((0, "maj", "maj7"), (2, "min", "min7"), (4, "min", "min7"),
                     (5, "maj", "maj7"), (7, "maj", "dom7"), (9, "min", "min7")),
             "min": ((0, "min", "min7"), (3, "maj", "maj7"), (5, "min", "min7"),
                     (7, "min", "min7"), (8, "maj", "maj7"), (10, "maj", "dom7"))}
CHORD_KEY_BONUS = 0.05      # nudge toward the key; never enough to override evidence
CHORD_MIN_CONCENTRATION = 0.42  # top-3 pitch-class share: chord ~0.7+, noise ~0.31
CHORD_MIN_CORR = 0.5        # below this the chroma does not look like a triad
CHORD_MIN_MARGIN = 0.05     # …and it must beat the best OTHER-ROOT candidate by this much
CHORD_MAX_SPANS = 64        # payload ceiling, like NOTE_MAX_REPORTED (compact metrics)
# Each chord tone is modelled with its first CHORD_PARTIALS partials, decaying
# geometrically. Six is deliberate: partials 1-6 fold to pitch-class offsets
# 0, 0, +7, 0, +4, +7 — root, fifth and major third only. Partial 7 folds to +10,
# a minor seventh, which would put a b7 into every MAJOR template and erode the
# maj/dom7 distinction the vocabulary exists for.
CHORD_PARTIALS = 6
CHORD_PARTIAL_DECAY = 0.6


def _chord_templates() -> tuple[list[tuple[int, str]], np.ndarray]:
    """(root, quality) labels and their 12-bin HARMONIC templates.

    Not binary (2026-08-08 review). Real instruments put strong energy on their
    partials, and folded into 12 bins those partials ARE other pitch classes: a
    low C's series alone lands on C, G, E and Bb. Against binary templates that
    made a rich low-register C major correlate better with Emin7 than with Cmaj
    — measured, a clean I-V-vi-IV at octave 2 returned the WRONG ROOTS, which is
    the one error a musician cannot work around. Modelling each chord tone's
    series puts that energy where the right template already expects it.

    Weights accumulate rather than overwrite: partials from different chord
    tones land on the same pitch class and must stack. Linear amplitude is the
    correct domain even though `_chroma` applies `log1p` first — scipy's `stft`
    normalises by the window sum, so the magnitudes reaching it are ~0.1 and
    `log1p(x) ≈ x` there. `_chroma` is left untouched, which also leaves `_key`,
    the concentration gate and the Phase 30 contract alone.
    """
    labels: list[tuple[int, str]] = []
    rows: list[np.ndarray] = []
    for quality, intervals in CHORD_TEMPLATES.items():
        for root in range(12):
            vec = np.zeros(12)
            for interval in intervals:
                for h in range(1, CHORD_PARTIALS + 1):
                    pc = (root + interval + int(round(12.0 * np.log2(h)))) % 12
                    vec[pc] += CHORD_PARTIAL_DECAY ** (h - 1)
            labels.append((root, quality))
            rows.append(vec)
    return labels, np.array(rows)


def _diatonic_set(key: Optional[KeyEstimate]) -> set[tuple[int, str]]:
    if key is None or key.mode not in _DIATONIC:
        return set()
    tonic = PITCH_CLASSES.index(key.tonic)
    return {((tonic + step) % 12, quality)
            for step, triad, seventh in _DIATONIC[key.mode]
            for quality in (triad, seventh)}


def _beat_chord(chroma: np.ndarray, labels, templates: np.ndarray,
                diatonic: set) -> Optional[tuple[int, str, float, float]]:
    """Best (root, quality, correlation, margin) for one beat's chroma, or None
    when nothing looks enough like a triad."""
    if chroma is None or chroma.std() == 0:
        return None
    # Tonality gate before scoring. A correlation can clear the confidence bar
    # on material that is not harmonic at all: broadband noise scored a
    # 0.51-confidence A#maj7 on one beat. A chord concentrates its energy in a
    # few pitch classes; noise spreads it evenly (top-3 share ~0.31 measured,
    # against 0.68-0.83 for real triads and sevenths), so the share is what
    # separates "this is a chord" from "this correlates with one".
    if float(np.sort(chroma)[-3:].sum()) < CHORD_MIN_CONCENTRATION:
        return None
    scores = np.array([float(np.corrcoef(chroma, t)[0, 1]) for t in templates])
    scores = np.nan_to_num(scores, nan=-1.0)
    # The key prior breaks ties toward chords that belong; it is deliberately
    # smaller than any real difference in fit, so it cannot manufacture a chord.
    biased = scores + np.array([CHORD_KEY_BONUS if lab in diatonic else 0.0
                                for lab in labels])
    order = np.argsort(biased)[::-1]
    best = int(order[0])
    root, quality = labels[best]
    # The margin exists to ask "is this the right ROOT?" — so score against the
    # best rival with a DIFFERENT root, not the overall runner-up (2026-08-08
    # review). A major triad's strongest rival is its own maj7, because the
    # fifth's 5th partial IS the major seventh; scoring against it collapsed the
    # margin below CHORD_MIN_MARGIN and threw the chord away over a distinction
    # that does not change the root. Measured effect: a clean I-V-vi-IV in C on
    # low, harmonically rich material returned ['Amin','Amin'] — 6 of 8 chords
    # discarded, and the survivors systematically minor, because a minor triad's
    # min7 rival sits much further away than a major triad's maj7 one.
    runner = next((int(i) for i in order[1:] if labels[int(i)][0] != root), None)
    margin = (float(biased[best] - biased[runner]) if runner is not None
              else float("inf"))
    return root, quality, float(scores[best]), margin


def analyze_chords(audio, sample_rate: int, stem_id: str, *,
                   bpm: Optional[float], beat_offset_s: float = 0.0,
                   key: Optional[KeyEstimate] = None) -> ChordAnalysis:
    """Detect major/minor triads per beat and merge them into spans.

    Beats, not fixed windows: chords change on the grid, so beat-aligned
    segments put each chord's evidence in one bucket instead of smearing two
    chords together. Requires the song's grid — without one there is no honest
    place to cut, and a wrong segmentation reads as wrong chords.
    """
    audio = _frames_channels(audio)
    if bpm is None or bpm <= 0:
        return ChordAnalysis(stem_id=stem_id, key=key,
                             reason="no confident tempo grid — chords are "
                                    "detected per beat, so there is nowhere to cut")
    if np.max(np.abs(audio)) < SILENCE_PEAK:
        return ChordAnalysis(stem_id=stem_id, key=key, grid_bpm=bpm,
                             reason="silent")

    mono = audio.mean(axis=1)
    beat_s = 60.0 / bpm
    beat_samples = int(round(beat_s * sample_rate))
    if beat_samples < KEY_NPERSEG * _rate_factor(sample_rate):
        return ChordAnalysis(stem_id=stem_id, key=key, grid_bpm=bpm,
                             reason="tempo too fast for a per-beat pitch window")

    labels, templates = _chord_templates()
    diatonic = _diatonic_set(key)
    start = int(round(beat_offset_s * sample_rate))
    per_beat: list[Optional[tuple[int, str, float, float]]] = []
    starts: list[int] = []
    while start + beat_samples <= mono.shape[0]:
        chroma = _chroma(mono[start:start + beat_samples], sample_rate)
        per_beat.append(_beat_chord(chroma, labels, templates, diatonic)
                        if chroma is not None else None)
        starts.append(start)
        start += beat_samples

    if not any(b is not None for b in per_beat):
        return ChordAnalysis(stem_id=stem_id, key=key, grid_bpm=bpm,
                             reason="no tonal content on this stem — chord "
                                    "detection needs harmonic material, not drums")

    # Despeckle: a single beat disagreeing with two identical neighbours is far
    # more often a passing note than a one-beat chord change.
    for i in range(1, len(per_beat) - 1):
        prev, here, nxt = per_beat[i - 1], per_beat[i], per_beat[i + 1]
        if prev is None or here is None or nxt is None:
            continue
        if prev[:2] == nxt[:2] and here[:2] != prev[:2] and here[3] < 2 * CHORD_MIN_MARGIN:
            per_beat[i] = (prev[0], prev[1], here[2], here[3])

    spans: list[ChordSpan] = []
    prev_beat = -2      # index of the beat the current span ends on
    tonal = kept = 0
    for i, entry in enumerate(per_beat):
        if entry is None:
            continue
        tonal += 1
        root, quality, corr, margin = entry
        if corr < CHORD_MIN_CORR or margin < CHORD_MIN_MARGIN:
            continue
        kept += 1
        start_s = starts[i] / sample_rate
        end_s = (starts[i] + beat_samples) / sample_rate
        # Contiguity is a BEAT-INDEX question, not a float one. Comparing the
        # stored (rounded) end_s against an unrounded start_s only ever matched
        # at tempi whose beat lands on a 10 ms boundary, so identical audio
        # merged into 8 spans at 120 BPM and fragmented into 31 at 128 BPM
        # (2026-08-08 review).
        if (spans and prev_beat == i - 1
                and spans[-1].root == PITCH_CLASSES[root]
                and spans[-1].quality == quality):
            spans[-1].end_s = round(end_s, 2)
            spans[-1].confidence = round(max(spans[-1].confidence, corr), 2)
            prev_beat = i
            continue
        spans.append(ChordSpan(
            start_s=round(start_s, 2), end_s=round(end_s, 2),
            start_bar=round((start_s - beat_offset_s) / (4.0 * beat_s) + 1.0, 1),
            root=PITCH_CLASSES[root], quality=quality,
            confidence=round(corr, 2)))
        prev_beat = i

    if not spans:
        return ChordAnalysis(stem_id=stem_id, key=key, grid_bpm=bpm,
                             reason="nothing scored confidently as a chord — "
                                    "the harmony may be denser than the "
                                    "triad/seventh vocabulary, or the stem too "
                                    "busy to read")
    # Bound the payload like every sibling readout (notes -> 8, worst_windows ->
    # 3). One span per beat with no ceiling sent ~450 spans / 42 KB for a
    # 4-minute track with fast harmonic rhythm, into history that is re-sent
    # every round — against the compact-metrics invariant.
    note: Optional[str] = None
    if len(spans) > CHORD_MAX_SPANS:
        note = (f"showing the first {CHORD_MAX_SPANS} of {len(spans)} chord "
                f"spans (the harmony changes faster than this readout reports)")
        spans = spans[:CHORD_MAX_SPANS]
    # Say so when most of the tonal material was too ambiguous to name, rather
    # than returning a confident-looking partial progression with reason=None.
    elif tonal and kept * 2 < tonal:
        note = (f"only {kept} of {tonal} tonal beats scored confidently — treat "
                "this progression as partial")
    return ChordAnalysis(stem_id=stem_id, key=key, grid_bpm=bpm, spans=spans,
                         reason=note)


def song_key(audio, sample_rate: int) -> Optional[KeyEstimate]:
    """The song's key from a master render — the prior chord detection scores
    against. Same estimator `analyze_session` reports."""
    return _key(_frames_channels(audio).mean(axis=1), sample_rate)


def song_tempo(audio, sample_rate: int) -> Optional[tuple[float, float, float]]:
    """The song's grid — (bpm, confidence, beat_offset_s) — from a master
    render. The same estimator `analyze_sections` reports, exposed so callers
    can hand one part's timing the grid it should be measured against."""
    return _tempo(_frames_channels(audio).mean(axis=1), sample_rate)


def stem_tempo(audio, sample_rate: int, stem_id: str) -> TempoAnalysis:
    """ONE part's own tempo and where its beat falls (`analyze_tempo`, 2026-10-11): `_tempo`
    on that part alone. Until this existed the only grid anyone could ask for was the
    master's, and with two songs in a project the agent muted each in turn to read it."""
    audio = _frames_channels(audio)
    if audio.shape[0] == 0 or np.max(np.abs(audio)) < SILENCE_PEAK:
        return TempoAnalysis(stem_id=stem_id, reason="the part is silent")
    tempo = _tempo(audio.mean(axis=1), sample_rate)
    if tempo is None:
        return TempoAnalysis(stem_id=stem_id,
                             reason="no steady pulse in this part (unrhythmic, free time, "
                                    "or too short to hold a tempo)")
    return TempoAnalysis(stem_id=stem_id, tempo_bpm=tempo[0], tempo_confidence=tempo[1],
                         beat_offset_s=tempo[2])


# Where each part comes in and drops out, to the bar (`analyze_activity`, 2026-10-11). A bar is
# judged against the PART's own loud bars (the 90th percentile of its bars above the silence
# floor): a hat fill under a break is 18-20 dB under a kit's full bars, a kick out and its
# reverb tail more.
ACTIVITY_IN_DB = 12.0
ACTIVITY_LOW_DB = 30.0
ACTIVITY_SILENCE_DBFS = -60.0
ACTIVITY_MAX_BARS = 2000             # a twelve-hour timeline is not a song: the reply is bounded
ACTIVITY_MAX_SPANS = 48              # ...and so is one part's list


ACTIVITY_ENTRY_WINDOW_S = 0.010       # the entry is found on a level read this often
ACTIVITY_ENTRY_BELOW_DB = 15.0        # ...where the part first comes within this of its run's level


def _entry_frame(totals: np.ndarray, first: int, last: int, sample_rate: int,
                 run_db: float, before_db: Optional[float]) -> Optional[int]:
    """The frame, inside a run's first bar (`first`..`last`), where the part starts sounding:
    the first short window within `ACTIVITY_ENTRY_BELOW_DB` of the run's own level - and, after
    a quieter run rather than silence, at least half-way up from that run's level, so what was
    already playing is not the entry. `totals` is the cumulative squared signal. None when the
    bar never gets there (a slow swell is no entry to point at)."""
    hop = max(1, int(round(ACTIVITY_ENTRY_WINDOW_S * sample_rate)))
    if last - first < hop:
        return None
    starts = np.arange(first, last - hop + 1, hop)
    window_db = 10.0 * np.log10(np.maximum((totals[starts + hop] - totals[starts]) / hop,
                                           SILENCE_PEAK ** 2))
    floor = run_db - ACTIVITY_ENTRY_BELOW_DB
    if before_db is not None:
        floor = max(floor, 0.5 * (before_db + run_db))
    reached = np.flatnonzero(window_db >= floor)
    return int(starts[reached[0]]) if reached.size else None


def analyze_activity(stems: dict, sample_rate: int, *, bpm: Optional[float],
                     bar1_s: float = 0.0, beats_per_bar: int = 4) -> ActivityAnalysis:
    """Per part, runs of whole bars that are `in`, `low` or `out` (contract: mix-analysis
    skill). `stems` maps a stem id to its audio AS IT PLAYS; `bpm` and `bar1_s` are the grid
    the bars are counted on - the caller's. Audio before bar 1 belongs to no bar."""
    if not bpm or bpm <= 0 or not np.isfinite(bpm):
        return ActivityAnalysis(reason="no tempo to count bars with: give `bpm` (and `bar1_s`, "
                                       "where bar 1 starts), or use it on material with a "
                                       "steady pulse")
    beats_per_bar = int(beats_per_bar) if beats_per_bar and beats_per_bar > 0 else 4
    bar_s = beats_per_bar * 60.0 / float(bpm)
    bar1_s = float(bar1_s) if np.isfinite(bar1_s) and bar1_s > 0 else 0.0
    prepared = {stem_id: _frames_channels(audio) for stem_id, audio in stems.items()}
    longest = max((a.shape[0] for a in prepared.values()), default=0)
    # a last stretch under half a bar is the timeline's end (a tail), not a bar of its own
    bars = int(np.floor(max(0.0, longest / sample_rate - bar1_s) / bar_s + 0.5))
    if bars <= 0:
        return ActivityAnalysis(bpm=round(float(bpm), 3), bar1_s=round(bar1_s, 4),
                                beats_per_bar=beats_per_bar, reason="no audio at or after bar 1")
    reason = None
    if bars > ACTIVITY_MAX_BARS:
        reason = f"only the first {ACTIVITY_MAX_BARS} bars of {bars} are listed"
        bars = ACTIVITY_MAX_BARS
    edges = np.round((bar1_s + np.arange(bars + 1) * bar_s) * sample_rate).astype(np.int64)
    out: list[StemActivity] = []
    for stem_id, audio in prepared.items():
        mono_sq = (audio.astype(np.float64) ** 2).mean(axis=1)
        totals = np.concatenate([[0.0], np.cumsum(mono_sq)])
        lo = np.minimum(edges[:-1], mono_sq.shape[0])
        hi = np.minimum(edges[1:], mono_sq.shape[0])
        frames = np.maximum(edges[1:] - edges[:-1], 1)       # a bar past the end is silence
        level = 10.0 * np.log10(np.maximum((totals[hi] - totals[lo]) / frames, SILENCE_PEAK ** 2))
        sounding = level[level > ACTIVITY_SILENCE_DBFS]
        loud = float(np.percentile(sounding, 90)) if sounding.size else None
        states = ["out" if loud is None or v <= ACTIVITY_SILENCE_DBFS or v < loud - ACTIVITY_LOW_DB
                  else "in" if v >= loud - ACTIVITY_IN_DB else "low" for v in level]
        spans: list[ActivitySpan] = []
        start = 0
        rank = {"out": 0, "low": 1, "in": 2}
        for bar in range(1, bars + 1):
            if bar == bars or states[bar] != states[start]:
                run = level[start:bar]
                span = ActivitySpan(
                    state=states[start], from_bar=start + 1, to_bar=bar,
                    level_db=None if states[start] == "out" else _round_db(float(np.median(run))))
                before = spans[-1] if spans else None
                if span.state != "out" and (before is None or rank[before.state] < rank[span.state]):
                    entered = _entry_frame(totals, int(lo[start]), int(hi[start]), sample_rate,
                                           float(np.median(run)),
                                           None if before is None or before.level_db is None
                                           else float(before.level_db))
                    if entered is not None:
                        span.enters_s = round(entered / sample_rate, 3)
                        span.enters_beat = round(
                            1.0 + (entered - int(edges[start])) / sample_rate / (bar_s / beats_per_bar), 2)
                spans.append(span)
                start = bar
        if len(spans) > ACTIVITY_MAX_SPANS:
            reason = reason or (f"a part changes state more than {ACTIVITY_MAX_SPANS} times: "
                                "its list is cut short")
            spans = spans[:ACTIVITY_MAX_SPANS]
        out.append(StemActivity(stem_id=stem_id, spans=spans))
    return ActivityAnalysis(bpm=round(float(bpm), 3), bar1_s=round(bar1_s, 4),
                            beats_per_bar=beats_per_bar, bars=bars, stems=out, reason=reason)


def analyze_timing(audio, sample_rate: int, stem_id: str, *,
                   bpm: Optional[float], beat_offset_s: float = 0.0,
                   ) -> TimingAnalysis:
    """Onset timing against the song's grid (contract: mix-analysis skill).

    `bpm`/`beat_offset_s` are the SONG's grid, not this stem's — the question is
    whether the part sits with the record, so a stem that cannot hold its own
    tempo estimate (a sparse pad, a single fill) is still measurable against the
    master's. Callers pass the session tempo.
    """
    audio = _frames_channels(audio)
    if bpm is None or bpm <= 0:
        return TimingAnalysis(stem_id=stem_id,
                              reason="no confident tempo grid for this song")
    if np.max(np.abs(audio)) < SILENCE_PEAK:
        return TimingAnalysis(stem_id=stem_id, grid_bpm=bpm, reason="silent")

    onsets = _onset_times_precise(audio.mean(axis=1), sample_rate)
    if onsets.size < TIMING_MIN_ONSETS:
        return TimingAnalysis(
            stem_id=stem_id, grid_bpm=bpm, onsets_total=int(onsets.size),
            reason=f"only {onsets.size} onsets — too few to time")

    chosen = _choose_subdivision(onsets, bpm, beat_offset_s)
    if chosen is None:
        return TimingAnalysis(
            stem_id=stem_id, grid_bpm=bpm, onsets_total=int(onsets.size),
            reason="onsets do not line up with any subdivision of this grid — "
                   "free time, or the tempo estimate does not hold here")
    subdivision, dev_beats, index = chosen
    ms_per_beat = 60_000.0 / bpm
    dev_ms = dev_beats * ms_per_beat
    on_grid = np.abs(dev_beats) <= TIMING_FIT_FRACTION / subdivision

    # Structural beats (even grid index) vs the positions between them (odd).
    # Swing moves ONLY the odd class, so the even class is the honest place to
    # read rush/drag, and the odd class's own lean is the swing figure.
    even, odd = (index % 2 == 0) & on_grid, (index % 2 == 1) & on_grid
    pair_ok = (subdivision in (2, 4, 8)
               and even.sum() >= TIMING_SWING_MIN_ONSETS
               and odd.sum() >= TIMING_SWING_MIN_ONSETS)
    # Which class is STRUCTURAL? `beat_offset_s` is only "where a beat was
    # found", so a track opening on a pickup or an off-beat hat anchors the grid
    # half a cell out and inverts the parity — the same 62 %-swung part read as
    # 38 % (2026-08-08 review). Swing moves the off-beats LATE; off-beats
    # arriving EARLY is not something players do, so a sub-50 reading means the
    # two classes are swapped, and swapping them back also puts `bias_ms` on the
    # structural beats where it belongs.
    if pair_ok:
        lean = (float(np.median(dev_beats[odd]))
                - float(np.median(dev_beats[even])))
        if 50.0 + 100.0 * lean / (2.0 / subdivision) < 50.0:
            even, odd = odd, even

    only_one_class = (even.sum() < TIMING_SWING_MIN_ONSETS) != (odd.sum() < TIMING_SWING_MIN_ONSETS)
    structural = dev_ms[even] if even.sum() >= TIMING_SWING_MIN_ONSETS else dev_ms[on_grid]
    bias_ms = float(np.median(structural))

    swing_percent = None
    if pair_ok:
        # The odd class nominally sits half a subdivision-pair after the even
        # one; its lean beyond that, as a percentage of the pair, IS the swing.
        lean_beats = float(np.median(dev_beats[odd])) - float(np.median(dev_beats[even]))
        pair_beats = 2.0 / subdivision
        swing_percent = round(50.0 + 100.0 * lean_beats / pair_beats, 1)

    # Tightness = what is left once the deliberate parts (a consistent lean and
    # the swing) are removed. Centre each parity class on its own median so
    # neither bias nor swing inflates it.
    jitter = dev_ms[on_grid].copy()
    if even.sum() and odd.sum():
        centred = np.concatenate([dev_ms[even] - np.median(dev_ms[even]),
                                  dev_ms[odd] - np.median(dev_ms[odd])])
    else:
        centred = jitter - np.median(jitter)
    # Robust spread: the interquartile range reads tightness without letting one
    # flam or missed hit dominate the way a standard deviation would.
    spread_ms = float(np.percentile(centred, 75) - np.percentile(centred, 25))

    return TimingAnalysis(
        stem_id=stem_id,
        grid_bpm=bpm,
        subdivision=subdivision,
        bias_ms=round(bias_ms, 1),
        spread_ms=round(spread_ms, 1),
        swing_percent=swing_percent,
        onsets_total=int(onsets.size),
        onsets_on_grid=int(on_grid.sum()),
        # Say when the lean is NOT read on structural beats. A part that plays
        # only off-beats (a reggae skank, an off-beat hat) has one parity class,
        # so bias falls back to every onset and its swing reads as drag — a
        # 62 %-swung skank reported "drags by 60 ms" with reason=None.
        reason=("this part plays only one side of the grid, so the lean is "
                "measured across every onset rather than on structural beats — "
                "a swung off-beat part will read as drag here"
                if only_one_class else None))


def _groove_votes(pos_beats: np.ndarray, subdivision: int
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Which onsets may say how late a cell of this grid is: (their indices into `pos_beats`,
    every onset's nearest cell, every onset's deviation in CELLS).

    A deviation is a feel, or it is a note of a finer grid, and only a feel is a lateness:

    * a PUSH or a DRAG - within GROOVE_PUSH_BEATS of the line, and a quarter of the cell;
    * a SWING - late by up to GROOVE_SWING_CELLS, on an OFF cell. A swing is never early and
      never on the beat: a note a sixteenth BEFORE beat 3 is a syncopation, not "beat 3, played
      a quarter of a beat early", and a 32nd pickup is not "a sixteenth, half a cell early".

    Where two onsets claim one cell the nearer keeps it (a flam, a kick's own second bump)."""
    cells = pos_beats * subdivision
    index = np.floor(cells + 0.5).astype(int)
    deviation = cells - index
    push = np.abs(deviation) <= min(GROOVE_PUSH_CELLS, GROOVE_PUSH_BEATS * subdivision)
    swing = ((subdivision >= 2) & (np.mod(index, 2) == 1)
             & (deviation > 0.0) & (deviation <= GROOVE_SWING_CELLS))
    votes: list[int] = []
    taken: set[int] = set()
    for i in np.argsort(np.abs(deviation), kind="stable"):
        if (push[i] or swing[i]) and int(index[i]) not in taken:
            taken.add(int(index[i]))
            votes.append(int(i))
    return np.asarray(sorted(votes), dtype=int), index, deviation


def _choose_groove_grid(onsets_s: np.ndarray, bpm: float, offset_s: float
                        ) -> Optional[tuple[int, np.ndarray, np.ndarray]]:
    """The grid a groove template can be APPLIED on — a different question from
    `_choose_subdivision`, and it must not reuse it.

    `_choose_subdivision` picks the grid that best *describes* a part, and for a
    swung part the honest description is a triplet grid. That is exactly wrong
    here: on a triplet grid the swing sits on-grid, so the extracted latenesses
    come out near zero and the template is flat — the feel we were trying to
    capture has been absorbed into the grid instead of recorded. Worse, applying
    a triplet-grid template to straight material interpolates between cells the
    target's notes never land on.

    A groove template's whole job is to express swing as LATENESS on a straight
    grid, which is why Live's grooves look like "16ths, off-beats +60%". So:
    straight subdivisions only, and the coarsest one on which the part's onsets
    are FEELS — pushes, drags and swings (`_groove_votes`) — with at most
    GROOVE_MAX_EXTRAS of them left over. Returns the voting onsets alone:
    (subdivision, their deviations in beats, their cells).

    Until 2026-10-09 the test was "no two onsets round to the same line", and any
    deviation up to half a cell was a lateness. On recorded drums that failed
    both ways: one 32nd pickup a bar sent the whole part to a 32nd grid, where
    it was "too sparse"; and a kick a sixteenth before the beat came out as a
    cell 0.95 early — a template that re-writes the rhythm of whatever borrows
    it. A note that is not a feel on this grid is an ornament of a finer one: it
    does not vote, and too many of them mean this grid is too coarse.
    """
    beat_s = 60.0 / bpm
    pos_beats = (onsets_s - offset_s) / beat_s
    for subdivision in GROOVE_SUBDIVISIONS:
        votes, index, deviation = _groove_votes(pos_beats, subdivision)
        if onsets_s.size - votes.size <= GROOVE_MAX_EXTRAS * onsets_s.size:
            return subdivision, deviation[votes] / subdivision, index[votes]
    return None


def extract_groove(audio, sample_rate: int, source: str, *,
                   bpm: Optional[float], beat_offset_s: float = 0.0,
                   ) -> GrooveExtraction:
    """Extract Groove: the per-cell timing shape of a take (contract: models).

    Shares its whole front end with `analyze_timing` — same precise onsets, same
    coarse-to-fine subdivision choice — and differs only in what it does with
    the per-onset deviations. `analyze_timing` collapses them to medians (is it
    rushing, is it tight); this one KEEPS the shape, folded onto one bar of the
    detected grid, which is the thing you can lend to another part.

    Folding onto one bar is what makes it a template rather than a recording of
    the take: every bar's hit at the same position votes on that cell, so the
    median is the part's habit there, not one performance of it.
    """
    audio = _frames_channels(audio)
    if bpm is None or bpm <= 0:
        return GrooveExtraction(source=source,
                                reason="no confident tempo grid for this song")
    if np.max(np.abs(audio)) < SILENCE_PEAK:
        return GrooveExtraction(source=source, grid_bpm=bpm, reason="silent")

    onsets = _onset_times_precise(audio.mean(axis=1), sample_rate)
    if onsets.size < TIMING_MIN_ONSETS:
        return GrooveExtraction(
            source=source, grid_bpm=bpm, onsets_used=int(onsets.size),
            reason=f"only {onsets.size} onsets — too few to extract a groove")

    chosen = _choose_groove_grid(onsets, bpm, beat_offset_s)
    if chosen is None:
        return GrooveExtraction(
            source=source, grid_bpm=bpm, onsets_used=int(onsets.size),
            reason="onsets are too dense or too free to sit on a straight grid — "
                   "no groove to extract from this part")
    subdivision, dev_beats, index = chosen

    cells = subdivision * GROOVE_BEATS_PER_BAR
    cell_of = np.mod(index, cells)
    latenesses = [0.0] * cells
    measured = 0
    for c in range(cells):
        sel = (cell_of == c)
        if int(sel.sum()) < GROOVE_MIN_CELL_ONSETS:
            continue   # no evidence here: straight, and not counted as measured
        # Tracktion's unit: the audible shift is HALF the lateness times the
        # cell, and a cell is 1/subdivision of a beat — so a deviation of
        # `d` beats is a lateness of 2*d*subdivision.
        shift = 2.0 * float(np.median(dev_beats[sel])) * subdivision
        latenesses[c] = round(float(np.clip(shift, -1.0, 1.0)), 4)
        measured += 1

    if measured < max(2, int(round(cells * GROOVE_MIN_CELL_COVERAGE))):
        return GrooveExtraction(
            source=source, grid_bpm=bpm, notes_per_beat=subdivision,
            onsets_used=int(onsets.size), cells_measured=measured,
            reason=f"only {measured} of {cells} grid positions were played — "
                   "too sparse to be a groove")
    if max(abs(v) for v in latenesses) < GROOVE_MIN_LATENESS:
        return GrooveExtraction(
            source=source, grid_bpm=bpm, notes_per_beat=subdivision,
            onsets_used=int(onsets.size), cells_measured=measured,
            reason="this part is already dead on the grid — no groove to extract")

    # …and does the part actually SIT on this grid? Coverage and lateness alone
    # accept anything: the repo's own three_section_demo/stem_b, which
    # analyze_timing calls free time (0 of 29 onsets on the grid), yielded a
    # confident 16-cell template, and so did white noise (2026-08-08 review).
    #
    # A groove is PERIODIC — the same cell is late by the same amount every bar,
    # which is exactly what makes it lendable. Random material is not. So
    # compare the spread of deviations WITHIN a repeated cell against the spread
    # across the bar: a real groove's cells agree with themselves and differ
    # from each other; noise scatters equally either way. Cells seen once carry
    # no evidence of repetition and are excluded.
    within: list[float] = []
    for c in range(cells):
        sel = (cell_of == c)
        if int(sel.sum()) < 2:
            continue
        d = dev_beats[sel]
        within.append(float(np.median(np.abs(d - np.median(d)))))
    if within:
        across = float(np.median(np.abs(dev_beats - np.median(dev_beats))))
        # Scale-free: a swung part's cells are tight about their own habit even
        # though the bar as a whole is spread.
        if across > 0 and float(np.median(within)) > across * GROOVE_MAX_CELL_SCATTER:
            return GrooveExtraction(
                source=source, grid_bpm=bpm, notes_per_beat=subdivision,
                onsets_used=int(onsets.size), cells_measured=measured,
                reason="the timing does not repeat from bar to bar — this is "
                       "free or unquantised playing, not a groove that can be "
                       "lent to another part")

    # The take's LEAN, measured the way analyze_timing measures it: on the
    # structural cells only. A median over every cell would be dominated by the
    # swing on a swung part and report a 74 ms drag on a take that is not
    # dragging at all — bias and swing are different things and must not merge.
    ms_per_beat = 60_000.0 / bpm
    structural = dev_beats[np.mod(index, 2) == 0]
    lean = float(np.median(structural)) if structural.size else float(np.median(dev_beats))
    return GrooveExtraction(
        source=source,
        grid_bpm=bpm,
        notes_per_beat=subdivision,
        latenesses=latenesses,
        cells_measured=measured,
        bias_ms=round(lean * ms_per_beat, 1),
        onsets_used=int(onsets.size))


def _phrase_spans(mono: np.ndarray, sample_rate: int) -> list[tuple[int, int]]:
    """Breath-separated phrases: runs of active audio on the same 400 ms /
    100 ms grid as the level moves, merged across gaps shorter than
    PHRASE_GAP_S (a breath does not end a phrase, a rest does). Returns sample
    index pairs, dropping spans too short to support a pitch estimate."""
    frame = max(1, min(int(RMS_FRAME_S * sample_rate), mono.shape[0]))
    hop = max(1, int(RMS_HOP_S * sample_rate))
    starts = np.arange(0, mono.shape[0] - frame + 1, hop)
    if starts.size == 0:
        return []
    levels = np.array([
        20.0 * np.log10(max(
            float(np.sqrt(np.mean(mono[s:s + frame] ** 2))), SILENCE_PEAK))
        for s in starts
    ])
    active = levels > MASKING_GATE_DBFS
    if not active.any():
        return []
    # Merge in FRAME space, not sample space: each window spans RMS_FRAME_S, so
    # a run's sample bounds overshoot the audio by up to a frame on each side
    # and two phrases a real rest apart can end up overlapping. The count of
    # wholly-silent frames between runs is the honest measure of the gap.
    gap_frames = max(1, int(round(PHRASE_GAP_S / RMS_HOP_S)))
    merged: list[tuple[int, int]] = []
    for start_i, end_i in _runs(active):
        if merged and start_i - merged[-1][1] < gap_frames:
            merged[-1] = (merged[-1][0], end_i)   # same phrase, just a breath
        else:
            merged.append((start_i, end_i))
    spans = [(int(starts[a]), int(starts[b - 1] + frame)) for a, b in merged]
    min_samples = max(KEY_NPERSEG * _rate_factor(sample_rate),
                      int(PHRASE_TUNING_MIN_S * sample_rate))
    return [(a, b) for a, b in spans if b - a >= min_samples]


def _phrase_pitch(mono: np.ndarray, sample_rate: int
                  ) -> tuple[list[PhrasePitch], Optional[float]]:
    """Per-phrase intonation (2026-08-08): the Phase 31 `_tuning` estimator run
    over each detected phrase, reported as a deviation from the stem's own
    tuning centre so a uniformly flat take doesn't read as N bad phrases.
    Returns (phrases worst-deviation-first, the stem's centre). Empty list on
    unpitched material — `_tuning` returns None there and we report nothing
    rather than guessing."""
    stem_tuning = _tuning(mono, sample_rate)
    if stem_tuning is None:
        return [], None
    centre_cents = stem_tuning[0]
    measured: list[PhrasePitch] = []
    for start, end in _phrase_spans(mono, sample_rate):
        phrase = _tuning(mono[start:end], sample_rate)
        if phrase is None:                       # unpitched phrase — skip it
            continue
        offset_cents, coherence = phrase
        measured.append(PhrasePitch(
            start_s=round(start / sample_rate, 2),
            end_s=round(end / sample_rate, 2),
            tuning_offset_cents=round(offset_cents, 1),
            deviation_cents=round(offset_cents - centre_cents, 1),
            tuning_coherence=round(coherence, 2)))
    measured.sort(key=lambda p: abs(p.deviation_cents), reverse=True)
    return measured[:PHRASE_MAX_PITCH], round(centre_cents, 1)


def analyze_phrases(audio, sample_rate: int, stem_id: str) -> PhraseAnalysis:
    """Phase 35 phrase-aware readout for existing ranged gain/de-ess moves.
    The agent still mutates only by proposal; this returns compact numbers."""
    audio = _frames_channels(audio)
    if np.max(np.abs(audio)) < SILENCE_PEAK:
        return PhraseAnalysis(stem_id=stem_id, silent=True)
    level_moves, level_silent = _phrase_level_moves(audio, sample_rate)
    bursts, sibilance_silent = _sibilance_bursts(audio, sample_rate)
    pitch, centre_cents = _phrase_pitch(audio.mean(axis=1), sample_rate)
    return PhraseAnalysis(
        stem_id=stem_id,
        level_moves=level_moves,
        sibilance_bursts=bursts,
        phrase_pitch=pitch,
        stem_tuning_offset_cents=centre_cents,
        silent=level_silent and sibilance_silent)


def _lufs_range(meter: pyln.Meter, audio: np.ndarray,
                sample_rate: int) -> Optional[float]:
    """LRA in LU, EBU R128-shaped: short-term (3 s window, 1 s hop) loudness
    distribution, absolute gate −70 LUFS, relative gate −20 LU below the
    power-mean of gated values, LRA = p95 − p10. None if too short. v4: the
    per-window values are ungated K-weighted short-term loudness (EBU's own
    short-term definition) instead of gated-integrated per window."""
    window = int(LRA_WINDOW_S * sample_rate)
    if audio.shape[0] <= window:
        return None
    values = _short_term_lufs_series(audio, sample_rate, LRA_WINDOW_S, LRA_HOP_S)
    values = values[values > LRA_ABS_GATE_LUFS]
    if values.size < 3:
        return None
    mean_lu = 10.0 * np.log10(np.mean(10.0 ** (values / 10.0)))
    gated = values[values >= mean_lu + LRA_REL_GATE_LU]
    if gated.size < 2:
        return None
    return float(np.percentile(gated, 95) - np.percentile(gated, 10))


def _band_crest_db(audio: np.ndarray, sample_rate: int) -> Optional[dict[str, float]]:
    """Per-band dynamics: p95 vs mean of 400 ms-block band power over active
    blocks, in dB. Low values flag a squashed band (mix-analysis skill)."""
    try:
        band_power, rms_db = _block_band_power(audio, sample_rate)
    except ValueError:
        return None
    active = band_power[rms_db > MASKING_GATE_DBFS]
    if active.shape[0] < 3:
        return None
    crest: dict[str, float] = {}
    for label, column in zip(BAND_LABELS, active.T):
        mean = column.mean()
        if mean <= 0:
            crest[label] = 0.0
        else:
            ratio = max(np.percentile(column, 95), SILENCE_PEAK) / mean
            crest[label] = max(_round_db(10.0 * np.log10(ratio)), 0.0)
    return crest


def _band_correlation(audio: np.ndarray, sample_rate: int) -> Optional[dict[str, float]]:
    """Per-band L/R correlation from Welch auto/cross spectra:
    Re(Sxy)/sqrt(Sxx·Syy) summed over each band's bins. −1 = anti-phase
    (cancels on mono fold-down), +1 = mono. None for mono sources."""
    if audio.shape[1] < 2:
        return None
    nperseg = _welch_nperseg(4096, audio.shape[0], sample_rate)
    freqs, sxx = _welch_psd(audio[:, 0], sample_rate, nperseg)
    _, syy = _welch_psd(audio[:, 1], sample_rate, nperseg)
    _, sxy = _welch_psd(audio[:, 0], sample_rate, nperseg, y=audio[:, 1])
    correlation: dict[str, float] = {}
    for label, lo_hz, hi_hz in zip(BAND_LABELS, BAND_EDGES_HZ[:-1], BAND_EDGES_HZ[1:]):
        in_band = (freqs >= lo_hz) & (freqs < hi_hz)
        denom = np.sqrt(sxx[in_band].sum() * syy[in_band].sum())
        if denom <= SILENCE_PEAK:
            continue  # band holds no energy: correlation is meaningless
        correlation[label] = round(float(np.clip(
            np.real(sxy[in_band].sum()) / denom, -1.0, 1.0)), 2)
    return correlation or None


def _played_notes(peak_hz: np.ndarray, tuning_offset_cents: float) -> np.ndarray:
    """Which of these peak centres are PLAYED NOTES on a part tuned `tuning_offset_cents` from
    A440 (mix-analysis skill v9): on a semitone of that tuning, or on a harmonic of a peak
    that is. A boolean per peak."""
    semis = 12.0 * np.log2(peak_hz / C0_HZ) - tuning_offset_cents / 100.0
    off_grid = ((semis + 0.5) % 1.0 - 0.5) * 100.0
    played = np.abs(off_grid) <= RESONANCE_NOTE_CENTS
    notes = peak_hz[played]
    if notes.size:
        ratio = peak_hz[:, np.newaxis] / notes[np.newaxis, :]
        k = np.round(ratio)
        cents = np.abs(1200.0 * np.log2(ratio / np.maximum(k, 1.0)))
        harmonic = (k >= 2) & (k <= RESONANCE_HARMONIC_MAX) & (cents <= RESONANCE_HARMONIC_CENTS)
        played = played | harmonic.any(axis=1)
    return played


def _resonances(audio: np.ndarray, sample_rate: int,
                tuning: Any = _TUNING_UNREAD) -> Optional[list[Resonance]]:
    """Narrow spectral peaks that are not the part's own notes (mix-analysis skill v9;
    Phase 20's detector, redefined for reader E5·R11): Welch-PSD bins standing >=
    RESONANCE_MIN_PROMINENCE_DB above a ±1/3-octave median baseline, 60 Hz–16 kHz - less
    the played notes and their harmonics when the material has a tuning (`_played_notes`).
    Top 3 by prominence, at least 1/3 octave apart, each with the Q of its −3
    dB-of-prominence bandwidth. None when nothing qualifies — most material has no resonance
    worth a surgical cut.

    A long-term peak 8 dB over its neighbourhood is exactly what a held note and each of its
    partials look like, so on pitched material the three slots used to fill with the song's
    notes: a "resonance" at 439 Hz was the sung A4, the agent was told to notch it, the user
    was offered the notch, and the rule agent capped a match-EQ boost for it. `tuning` is
    `_tuning`'s answer when the caller already has it (None = unpitched)."""
    from scipy.signal import find_peaks
    mono = audio.mean(axis=1)
    nperseg = _welch_nperseg(RESONANCE_NPERSEG, mono.shape[0], sample_rate)
    freqs, psd = _welch_psd(mono, sample_rate, nperseg)
    in_range = ((freqs >= RESONANCE_LO_HZ)
                & (freqs <= min(RESONANCE_HI_HZ, sample_rate / 2.0))
                & (psd > 0))
    if in_range.sum() < 16:
        return None
    freqs, power = freqs[in_range], psd[in_range]
    level_db = 10.0 * np.log10(power)

    # Median baseline over a ±1/3-octave window per bin (two-pointer walk —
    # the frequency axis is sorted, so the window only ever moves forward).
    log_f = np.log2(freqs)
    baseline = np.empty_like(level_db)
    lo = hi = 0
    for i, center in enumerate(log_f):
        while log_f[lo] < center - RESONANCE_BASELINE_HALF_OCT:
            lo += 1
        while hi < log_f.size and log_f[hi] <= center + RESONANCE_BASELINE_HALF_OCT:
            hi += 1
        baseline[i] = np.median(level_db[lo:hi])
    prominence = np.minimum(level_db - baseline, RESONANCE_PROMINENCE_CAP_DB)

    peak_idx, _ = find_peaks(prominence, height=RESONANCE_MIN_PROMINENCE_DB)
    if peak_idx.size == 0:
        return None
    if tuning is _TUNING_UNREAD:
        tuning = _tuning(mono, sample_rate)
    played: set[int] = set()
    if tuning is not None:                          # pitched: its notes are not resonances
        centres = np.array([_peak_centre_hz(freqs, power, log_f, level_db, int(p))
                            for p in peak_idx])
        played = {int(p) for p, note in zip(peak_idx, _played_notes(centres, tuning[0])) if note}
    found: list[Resonance] = []
    kept_log_f: list[float] = []
    for p in sorted(peak_idx, key=lambda i: prominence[i], reverse=True):
        if int(p) in played:
            continue  # a note the part plays, or a partial of one
        if any(abs(log_f[p] - kept) < RESONANCE_MIN_SPACING_OCT
               for kept in kept_log_f):
            continue  # a shoulder of an already-reported peak
        # Q from the −3 dB-of-prominence bandwidth around the peak.
        drop = prominence[p] - 3.0
        left = p
        while left > 0 and prominence[left - 1] > drop:
            left -= 1
        right = p
        while right < prominence.size - 1 and prominence[right + 1] > drop:
            right += 1
        bandwidth_hz = max(freqs[right] - freqs[left], freqs[1] - freqs[0])
        q = min(float(freqs[p] / bandwidth_hz), RESONANCE_Q_CAP)
        freq = float(freqs[p])
        found.append(Resonance(
            freq_hz=int(round(freq)) if freq < 1000 else int(round(freq / 10) * 10),
            prominence_db=_round_db(prominence[p]),
            q=round(q, 1)))
        kept_log_f.append(float(log_f[p]))
        if len(found) == RESONANCE_MAX_REPORTED:
            break
    return found or None


def _peak_centre_hz(freqs: np.ndarray, power: np.ndarray, log_f: np.ndarray,
                    level_db: np.ndarray, p: int) -> float:
    """Where a spectral peak is CENTRED, for the played-note test: the power centroid (in log
    frequency) of the peak's own stretch when that holds at least five bins, else the
    parabolic vertex of the three bins around it. The stretch runs out to
    ±RESONANCE_CENTRE_CENTS, and where it runs into something more than
    RESONANCE_CENTRE_RISE_DB LOUDER it is cut back to the valley in between: a ring a
    quarter-tone from a played note has that note's line at the edge of its span, 30 dB over
    it, and was otherwise weighed as the note - by the line itself, or by the flank of it.
    (The two ends of a vibrato'd note's swing are within that rise of each other, and stay
    one stretch.)"""
    half = RESONANCE_CENTRE_CENTS / 1200.0
    ceiling = level_db[p] + RESONANCE_CENTRE_RISE_DB
    lo = p
    while lo > 0 and log_f[p] - log_f[lo - 1] <= half and level_db[lo - 1] <= ceiling:
        lo -= 1
    if lo > 0 and level_db[lo - 1] > ceiling:           # a louder neighbour below: not ours
        lo += int(np.argmin(level_db[lo:p + 1]))
    hi = p + 1
    while hi < level_db.size and log_f[hi] - log_f[p] <= half and level_db[hi] <= ceiling:
        hi += 1
    if hi < level_db.size and level_db[hi] > ceiling:   # ...and above
        hi = p + int(np.argmin(level_db[p:hi])) + 1
    if hi - lo >= 5:
        weight = power[lo:hi]
        return float(2.0 ** ((weight * log_f[lo:hi]).sum() / weight.sum()))
    if 0 < p < level_db.size - 1:
        left, centre, right = level_db[p - 1], level_db[p], level_db[p + 1]
        denom = left - 2.0 * centre + right
        shift = 0.0 if denom == 0 else float(np.clip(0.5 * (left - right) / denom, -0.5, 0.5))
        return float(freqs[p] + shift * (freqs[1] - freqs[0]))
    return float(freqs[p])


def _composites(band_energy: dict[str, float],
                band_crest: Optional[dict[str, float]],
                transient_ratio: Optional[float],
                ) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """The Phase 14 composite indices (contract: mix-analysis skill v5).
    harshness_db / mud_db: region energy share minus half its crest — loud
    AND squashed reads as harsh/muddy; dynamics offset energy. clarity_index:
    master attack share plus normalized mid-region crest (higher = clearer)."""
    harshness = mud = clarity = None
    if band_crest is not None:
        harsh_energy = sum(band_energy[b] for b in HARSH_BANDS) / len(HARSH_BANDS)
        harsh_crest = sum(band_crest[b] for b in HARSH_BANDS) / len(HARSH_BANDS)
        harshness = _round_db(harsh_energy - COMPOSITE_CREST_WEIGHT * harsh_crest)
        mud = _round_db(band_energy[MUD_BAND]
                        - COMPOSITE_CREST_WEIGHT * band_crest[MUD_BAND])
        if transient_ratio is not None:
            clarity_crest = (sum(band_crest[b] for b in CLARITY_BANDS)
                             / len(CLARITY_BANDS))
            clarity = round(transient_ratio
                            + clarity_crest / CLARITY_CREST_NORM_DB, 2)
    return harshness, mud, clarity


def over_compressed(master_dynamics: Optional[list],
                    stem_dynamics: Optional[dict[str, list]] = None) -> bool:
    """The `over_compression` render-flag predicate (2026-06 review):
    True when any non-limiter dynamics entry in the render's telemetry shows
    gr_mean_db >= OVER_COMPRESSION_GR_MEAN_DB, or is pinned — active on >=
    OVER_COMPRESSION_PINNED_ACTIVE_PCT of blocks while reducing >=
    OVER_COMPRESSION_PINNED_MEAN_DB on average. Entries are the telemetry
    dicts dsp produces (processor / gr_mean_db / gr_max_db / gr_active_pct)."""
    entries = list(master_dynamics or [])
    for per_stem in (stem_dynamics or {}).values():
        entries += per_stem
    return any(
        entry["processor"] != "limiter"
        and (entry["gr_mean_db"] >= OVER_COMPRESSION_GR_MEAN_DB
             or (entry["gr_active_pct"] >= OVER_COMPRESSION_PINNED_ACTIVE_PCT
                 and entry["gr_mean_db"] >= OVER_COMPRESSION_PINNED_MEAN_DB))
        for entry in entries)


def over_limited(master_dynamics: Optional[list], psr_db: Optional[float],
                 plr_db: Optional[float]) -> bool:
    """The `over_limiting` render-flag predicate (Phase 34): true when a
    limiter is reducing hard and the resulting master has low short-term or
    integrated loudness margin. Busy limiter GR alone is not enough — loud
    targets can legitimately engage the limiter."""
    low_loudness_margin = (
        (psr_db is not None and psr_db <= OVER_LIMITING_PSR_DB)
        or (plr_db is not None and plr_db <= OVER_LIMITING_PLR_DB)
    )
    if not low_loudness_margin:
        return False
    limiters = [entry for entry in (master_dynamics or [])
                if entry["processor"] == "limiter"]
    return any(
        entry["gr_mean_db"] >= OVER_LIMITING_GR_MEAN_DB
        or entry["gr_max_db"] >= OVER_LIMITING_GR_MAX_DB
        or (entry["gr_active_pct"] >= OVER_LIMITING_PINNED_ACTIVE_PCT
            and entry["gr_mean_db"] >= OVER_LIMITING_PINNED_MEAN_DB)
        for entry in limiters)


def analyze_master(audio, sample_rate: int, stems: list[str],
                   pass_number: int,
                   master_dynamics: Optional[list[dict]] = None,
                   ) -> SessionAnalysis:
    """`master_dynamics` is the render's gain-reduction telemetry (Phase 20,
    dsp.render_master telemetry param) — analysis passes it through; it is
    measured during the render, not derivable from the bounce."""
    audio = _frames_channels(audio)
    if np.max(np.abs(audio)) < SILENCE_PEAK:
        return SessionAnalysis(
            lufs_i=None, lufs_s_max=None, true_peak_dbtp=None,
            stereo_correlation=None, spectral_tilt_db_per_oct=None,
            crest_factor_db=None, band_energy=None,
            stems=stems, pass_number=pass_number, silent=True)

    meter = pyln.Meter(sample_rate)
    lufs_i = _integrated_lufs(meter, audio)
    lufs_s_max = _short_term_max_lufs(meter, audio, sample_rate)
    lufs_range = _lufs_range(meter, audio, sample_rate)
    correlation = _stereo_correlation(audio)
    tilt = _spectral_tilt_db_per_oct(audio, sample_rate)
    true_peak = _round_db(_true_peak_dbtp(audio))
    band_energy = _band_energy(audio, sample_rate)
    band_crest = _band_crest_db(audio, sample_rate)
    harshness, mud, clarity = _composites(
        band_energy, band_crest, _transient_energy_ratio(audio, sample_rate))
    # Phase 30: song-level key + tempo on the master mono (descriptive readout).
    mono = audio.mean(axis=1)
    tempo = _tempo(mono, sample_rate)
    key = _key(mono, sample_rate)
    tuning = _tuning(mono, sample_rate)  # Phase 31: global tuning readout
    return SessionAnalysis(
        lufs_i=_round_db(lufs_i) if lufs_i is not None else None,
        lufs_s_max=_round_db(lufs_s_max) if lufs_s_max is not None else None,
        lufs_range=_round_db(lufs_range) if lufs_range is not None else None,
        true_peak_dbtp=true_peak,
        # Phase 14 mastering QC: crest of the loudness, not the samples —
        # how much headroom the limiter left over integrated / short-term.
        plr_db=(_round_db(true_peak - lufs_i) if lufs_i is not None else None),
        psr_db=(_round_db(true_peak - lufs_s_max)
                if lufs_s_max is not None else None),
        stereo_correlation=round(correlation, 2) if correlation is not None else None,
        band_correlation=_band_correlation(audio, sample_rate),
        spectral_tilt_db_per_oct=_round_db(tilt) if tilt is not None else None,
        crest_factor_db=_round_db(_crest_factor_db(audio)),
        band_crest_db=band_crest,
        band_energy=band_energy,
        harshness_db=harshness,
        mud_db=mud,
        clarity_index=clarity,
        resonances=_resonances(audio, sample_rate, tuning),
        master_dynamics=master_dynamics,
        tempo_bpm=tempo[0] if tempo is not None else None,
        tempo_confidence=tempo[1] if tempo is not None else None,
        key=key,
        tuning_offset_cents=tuning[0] if tuning is not None else None,
        tuning_coherence=tuning[1] if tuning is not None else None,
        stems=stems, pass_number=pass_number)


def level_metrics(audio, sample_rate: int) -> dict:
    """What a MOVE is measured by: loudness, true peak and tilt - the three figures a turn's
    delta compares (`daw_bridge.turn_delta`). `readout_metrics` gives the same three, by these
    calls, with the song's tempo, key and band energy beside them; a measurement taken after
    a fader move paid for all six and the delta read three (Phase 55)."""
    audio = _frames_channels(audio)
    if np.max(np.abs(audio)) < SILENCE_PEAK:
        return {"lufs_i": None, "true_peak_dbtp": None, "spectral_tilt_db_per_oct": None,
                "silent": True}
    lufs_i = _integrated_lufs(pyln.Meter(sample_rate), audio)
    tilt = _spectral_tilt_db_per_oct(audio, sample_rate)
    return {"lufs_i": _round_db(lufs_i) if lufs_i is not None else None,
            "true_peak_dbtp": _round_db(_true_peak_dbtp(audio)),
            "spectral_tilt_db_per_oct": _round_db(tilt) if tilt is not None else None,
            "silent": False}


def readout_metrics(audio, sample_rate: int) -> dict:
    """What the DAW's top bar shows of a master, and nothing else (reader E5·R4): loudness,
    true peak, tempo, key, band energy, tilt. Each number is the one `analyze_master` gives -
    the same functions, the same rounding - but the short-term series, the loudness range,
    band correlation, band crest, resonances and the tuning it also computes (most of its
    time) are not paid for by every brain reply only to be dropped."""
    audio = _frames_channels(audio)
    if np.max(np.abs(audio)) < SILENCE_PEAK:
        return {"lufs_i": None, "true_peak_dbtp": None, "tempo_bpm": None,
                "tempo_confidence": None, "key": None, "band_energy": None,
                "spectral_tilt_db_per_oct": None, "silent": True}
    levels = level_metrics(audio, sample_rate)
    mono = audio.mean(axis=1)
    tempo = _tempo(mono, sample_rate)
    key = _key(mono, sample_rate)
    return {
        "lufs_i": levels["lufs_i"],
        "true_peak_dbtp": levels["true_peak_dbtp"],
        "tempo_bpm": tempo[0] if tempo is not None else None,
        "tempo_confidence": tempo[1] if tempo is not None else None,
        "key": ({"tonic": key.tonic, "mode": key.mode, "confidence": round(key.confidence, 2)}
                if key is not None else None),
        "band_energy": _band_energy(audio, sample_rate),
        "spectral_tilt_db_per_oct": levels["spectral_tilt_db_per_oct"],
        "silent": False,
    }


def analyze_stem(audio, sample_rate: int, stem_id: str,
                 applied_chain: list[dict],
                 dynamics: Optional[list[dict]] = None) -> StemAnalysis:
    """`dynamics` is the stem render's gain-reduction telemetry (Phase 20,
    dsp.render_stem telemetry param), passed through like master_dynamics."""
    audio = _frames_channels(audio)
    if np.max(np.abs(audio)) < SILENCE_PEAK:
        return StemAnalysis(
            stem_id=stem_id, lufs_s=None, peak_dbfs=None, band_energy=None,
            transient_density=None, dynamic_spread_db=None,
            applied_chain=applied_chain, silent=True)

    meter = pyln.Meter(sample_rate)
    lufs_s = _short_term_max_lufs(meter, audio, sample_rate)
    spread = _dynamic_spread_db(audio, sample_rate)
    tuning = _tuning(audio.mean(axis=1), sample_rate)  # Phase 31: stem tuning
    return StemAnalysis(
        stem_id=stem_id,
        lufs_s=_round_db(lufs_s) if lufs_s is not None else None,
        peak_dbfs=_round_db(_peak_dbfs(audio)),
        band_energy=_band_energy(audio, sample_rate),
        transient_density=_transient_density(audio, sample_rate),
        transient_energy_ratio=_transient_energy_ratio(audio, sample_rate),
        sibilance_density=_sibilance_density(audio, sample_rate),
        dynamic_spread_db=_round_db(spread) if spread is not None else None,
        resonances=_resonances(audio, sample_rate, tuning),
        dynamics=dynamics,
        tuning_offset_cents=tuning[0] if tuning is not None else None,
        tuning_coherence=tuning[1] if tuning is not None else None,
        applied_chain=applied_chain)


# --------------------------------------------------------------------------- #
# Region description (2026-09-03) — see mix-analysis skill                     #
# --------------------------------------------------------------------------- #

#: A masker has to overpower the stem by this much in its worst band before it
#: is named — below it the two are sharing the band, not fighting over it.
REGION_MASK_FLOOR_DB = 3.0
#: Envelope resolution: the region is split into at most this many windows.
REGION_ENVELOPE_POINTS = 8


def _slice_region(audio: np.ndarray, sample_rate: int,
                  start_s: float, end_s: float) -> np.ndarray:
    a = max(0, int(round(start_s * sample_rate)))
    b = min(audio.shape[0], int(round(end_s * sample_rate)))
    return audio[a:b]


def _region_envelope(seg: np.ndarray, sample_rate: int, start_s: float,
                     ) -> list[RegionEnvelopePoint]:
    """<= REGION_ENVELOPE_POINTS short-term loudness readings across the
    region: window = region / points (floored at 400 ms so a reading means
    something), hop = window, so the points tile the region."""
    duration = seg.shape[0] / sample_rate
    window_s = max(0.4, duration / REGION_ENVELOPE_POINTS)
    series = _short_term_lufs_series(seg, sample_rate, window_s, window_s)
    points = []
    for i, value in enumerate(series[:REGION_ENVELOPE_POINTS]):
        lufs = float(value)
        points.append(RegionEnvelopePoint(
            t_s=round(start_s + i * window_s, 2),
            lufs_s=_round_db(lufs) if lufs > LRA_ABS_GATE_LUFS else None))
    return points


def _region_maskers(seg: np.ndarray, sample_rate: int, stem_id: str,
                    others: dict[str, np.ndarray],
                    ) -> tuple[list[RegionMasker], list[RegionMasker]]:
    """Both directions of masking against every other active stem in the
    same window, worst band each, above the floor. Sorted worst first."""
    masked_by: list[RegionMasker] = []
    masks: list[RegionMasker] = []
    for other_id, other in sorted(others.items()):
        other = _frames_channels(other)
        n = min(seg.shape[0], other.shape[0])
        if n < int(0.4 * sample_rate) or np.max(np.abs(other[:n])) < SILENCE_PEAK:
            continue
        a, b = seg[:n], other[:n]
        # Each side's table once for both directions (Phase 55): `analyze_masking` per
        # direction built both tables and both channels' again, twelve block-FFT passes a
        # pair, for the two figures `masking_worst` reads off the tables.
        try:
            table_a, table_b = masking_table(a, sample_rate), masking_table(b, sample_rate)
        except ValueError:
            continue
        # A pair `analyze_masking` cannot judge is a pair with no masking to report (reader
        # E5·R10): a region that holds only DC has no band above the content floor, and its
        # ValueError took the whole description down.
        for first, second, first_id, into in ((table_a, table_b, stem_id, masked_by),
                                              (table_b, table_a, other_id, masks)):
            try:
                worst_band, excess = masking_worst(first, second, first_id)
            except ValueError:
                continue
            if excess >= REGION_MASK_FLOOR_DB:
                into.append(RegionMasker(stem_id=other_id, worst_band=worst_band,
                                         excess_db=_round_db(excess)))
    key = lambda m: -m.excess_db  # noqa: E731
    return sorted(masked_by, key=key), sorted(masks, key=key)


def describe_region(audio, sample_rate: int, stem_id: str,
                    start_s: float, end_s: float,
                    others: Optional[dict[str, np.ndarray]] = None,
                    ) -> RegionDescription:
    """One stem, one time range, every perceptual number the agent needs to
    talk about 'this part': loudness + its envelope, dynamics, spectrum and
    how it moves, transients, the composites, and who is masking whom among
    the other stems (`others`: stem_id -> that stem's processed audio, same
    sample rate, whole-file; sliced here). Deterministic; no file access."""
    audio = _frames_channels(audio)
    seg = _slice_region(audio, sample_rate, start_s, end_s)
    base = dict(stem_id=stem_id, start_s=round(float(start_s), 2),
                end_s=round(float(end_s), 2))
    if seg.shape[0] < int(0.4 * sample_rate) or np.max(np.abs(seg)) < SILENCE_PEAK:
        return RegionDescription(silent=True, **base)

    meter = pyln.Meter(sample_rate)
    lufs_i = _integrated_lufs(meter, seg)
    lufs_s = _short_term_max_lufs(meter, seg, sample_rate)
    spread = _dynamic_spread_db(seg, sample_rate)
    band = _band_energy(seg, sample_rate)
    half = seg.shape[0] // 2
    tilt = _spectral_tilt_db_per_oct(seg, sample_rate)
    trend = None
    if half >= int(0.4 * sample_rate):
        first = _spectral_tilt_db_per_oct(seg[:half], sample_rate)
        second = _spectral_tilt_db_per_oct(seg[half:], sample_rate)
        if first is not None and second is not None:
            trend = round(second - first, 2)
    transient_ratio = _transient_energy_ratio(seg, sample_rate)
    harsh, mud, clarity = _composites(band, _band_crest_db(seg, sample_rate),
                                      transient_ratio)
    sliced_others = {sid: _slice_region(_frames_channels(a), sample_rate, start_s, end_s)
                     for sid, a in (others or {}).items() if sid != stem_id}
    masked_by, masks = _region_maskers(seg, sample_rate, stem_id, sliced_others)
    return RegionDescription(
        lufs_i=_round_db(lufs_i) if lufs_i is not None else None,
        lufs_s=_round_db(lufs_s) if lufs_s is not None else None,
        lufs_envelope=_region_envelope(seg, sample_rate, start_s),
        peak_dbfs=_round_db(_peak_dbfs(seg)),
        true_peak_dbtp=_round_db(_true_peak_dbtp(seg)),
        crest_db=_round_db(_crest_factor_db(seg)),
        dynamic_spread_db=_round_db(spread) if spread is not None else None,
        band_energy=band,
        spectral_tilt_db_per_oct=round(tilt, 2) if tilt is not None else None,
        tilt_trend_db_per_oct=trend,
        transient_density=_transient_density(seg, sample_rate),
        transient_energy_ratio=transient_ratio,
        harshness_db=harsh, mud_db=mud, clarity_index=clarity,
        masked_by=masked_by, masks=masks, **base)


# --------------------------------------------------------------------------- #
# Separation artifact detection (Phase 29) — see mix-analysis skill            #
# --------------------------------------------------------------------------- #

def _hf_shimmer(audio: np.ndarray, sample_rate: int) -> Optional[float]:
    """Share of 4–20 kHz energy that lands in the stem's quiet frames —
    separation 'musical noise' flickers in the gaps where the source is silent.
    100 ms frames; quiet = bottom-quartile broadband RMS among active frames.
    None if too few active frames."""
    mono = audio.mean(axis=1)
    frame = int(SIBILANCE_FRAME_S * sample_rate)
    n = mono.shape[0] // frame
    if n < 4:
        return None
    frames = mono[: n * frame].reshape(n, frame)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    rms_db = 20.0 * np.log10(np.maximum(rms, SILENCE_PEAK))
    active = rms_db > MASKING_GATE_DBFS
    if active.sum() < 4:
        return None
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(frame, d=1.0 / sample_rate)
    hf = power[:, freqs >= HF_SHIMMER_LO_HZ].sum(axis=1)[active]
    total = float(hf.sum())
    if total <= 0.0:
        return 0.0
    rms_active = rms[active]
    quiet = rms_active <= np.quantile(rms_active, SHIMMER_QUIET_QUANTILE)
    return round(float(hf[quiet].sum() / total), 2)


def _transient_smear(audio: np.ndarray, sample_rate: int) -> Optional[float]:
    """Pre-echo smear at onsets: mean pre/(pre+post) energy in ±30 ms windows
    around each detected onset. 0 = clean attacks, →0.5 = energy smeared ahead
    of the transient (a transform/separation artifact). None if no onsets."""
    mono = audio.mean(axis=1)
    onsets = _onset_peaks(mono, sample_rate)
    if onsets is None:
        return None
    peaks, hop_s = onsets
    w = int(TRANSIENT_SMEAR_WINDOW_S * sample_rate)
    if peaks.size == 0 or w < 1:
        return None
    ratios = []
    for frame in peaks:
        c = int(frame * hop_s * sample_rate)
        pre = float(np.sum(mono[max(0, c - w):c] ** 2))
        post = float(np.sum(mono[c:c + w] ** 2))
        if pre + post > 0.0:
            ratios.append(pre / (pre + post))
    if not ratios:
        return None
    return round(float(np.mean(ratios)), 2)


def _spectral_hole_ratio(audio: np.ndarray, sample_rate: int) -> Optional[float]:
    """Carved-notch indicator: among the stem's content bands (≥ −20 dB of its
    band power), the fraction of active 400 ms blocks where the band drops more
    than 12 dB below that band's own median power. None if too few active
    blocks."""
    try:
        band_power, rms_db = _block_band_power(audio, sample_rate)
    except ValueError:
        return None
    active = rms_db > MASKING_GATE_DBFS
    if active.sum() < 4:
        return None
    bp = band_power[active]
    total = float(bp.sum())
    if total <= 0.0:
        return None
    share = bp.sum(axis=0) / total
    content = np.where(share >= 10.0 ** (SPECTRAL_HOLE_CONTENT_FLOOR_DB / 10.0))[0]
    if content.size == 0:
        return None
    hole_fracs = []
    for b in content:
        col = bp[:, b]
        positive = col[col > 0]
        if positive.size == 0:
            continue
        med = float(np.median(positive))
        if med <= 0:
            continue
        drop_db = 10.0 * np.log10(np.maximum(col, SILENCE_PEAK ** 2) / med)
        hole_fracs.append(float(np.mean(drop_db < -SPECTRAL_HOLE_DROP_DB)))
    if not hole_fracs:
        return None
    return round(float(np.mean(hole_fracs)), 2)


def _leakage(env_target: np.ndarray, others: list[np.ndarray],
             active: np.ndarray) -> Optional[float]:
    """Bleed indicator for one stem: the strongest rank (Spearman) correlation
    between this stem's level envelope and another stem's, over the blocks where
    this stem is quiet (≤ its own median) but active — where a clean stem should
    be silent, a bleeding one tracks the other source. Rank correlation keeps a
    single transition block (a loud tail landing in the quiet set) from
    dominating. 0..1; None if too few quiet blocks or no variation."""
    if active.sum() < LEAKAGE_MIN_BLOCKS:
        return None
    x = env_target[active]
    quiet = x <= np.median(x)
    if quiet.sum() < LEAKAGE_MIN_BLOCKS or np.std(x[quiet]) < SILENCE_PEAK:
        return None
    rx = rankdata(x[quiet])
    if np.std(rx) < 1e-9:
        return None
    best = 0.0
    for env_y in others:
        y = env_y[active][quiet]
        if np.std(y) < SILENCE_PEAK:
            continue
        ry = rankdata(y)
        if np.std(ry) < 1e-9:
            continue
        r = float(np.corrcoef(rx, ry)[0, 1])
        best = max(best, r)
    return round(min(max(best, 0.0), 1.0), 2)


def analyze_separation(stems: dict, sample_rate: int) -> SeparationReport:
    """Phase 29 'double check after splitting': per-stem artifact metrics
    (hf_shimmer, transient_smear, spectral_hole_ratio, leakage, lufs_s) plus the
    cross-stem loudness spread. Pure — callers pass {stem_id: audio} in.
    Definitions are contractual: mix-analysis skill."""
    meter = pyln.Meter(sample_rate)
    per_stem: list[StemArtifacts] = []
    envelopes: dict[str, np.ndarray] = {}
    actives: dict[str, np.ndarray] = {}
    lufs_values: list[float] = []
    for stem_id in sorted(stems):
        audio = _frames_channels(stems[stem_id])
        if np.max(np.abs(audio)) < SILENCE_PEAK:
            per_stem.append(StemArtifacts(stem_id=stem_id, silent=True))
            continue
        lufs_s = _short_term_max_lufs(meter, audio, sample_rate)
        if lufs_s is not None:
            lufs_values.append(lufs_s)
        per_stem.append(StemArtifacts(
            stem_id=stem_id,
            lufs_s=_round_db(lufs_s) if lufs_s is not None else None,
            hf_shimmer=_hf_shimmer(audio, sample_rate),
            transient_smear=_transient_smear(audio, sample_rate),
            spectral_hole_ratio=_spectral_hole_ratio(audio, sample_rate)))
        try:
            _, rms_db = _block_band_power(audio, sample_rate)
            envelopes[stem_id] = rms_db
            actives[stem_id] = rms_db > MASKING_GATE_DBFS
        except ValueError:
            pass
    # Leakage is cross-stem: align level envelopes to a common block count and
    # correlate each stem's quiet spans against the others.
    if len(envelopes) >= 2:
        n = min(len(e) for e in envelopes.values())
        linear = {sid: 10.0 ** (e[:n] / 20.0) for sid, e in envelopes.items()}
        for art in per_stem:
            if art.stem_id not in linear:
                continue
            others = [linear[o] for o in linear if o != art.stem_id]
            art.leakage = _leakage(linear[art.stem_id], others,
                                   actives[art.stem_id][:n])
    spread = (round(max(lufs_values) - min(lufs_values), 1)
              if len(lufs_values) >= 2 else None)
    return SeparationReport(per_stem=per_stem, inter_stem_lufs_spread_db=spread)


def _block_band_power(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """(blocks, bands) band power per non-overlapping 400 ms block, plus the
    per-block full-band RMS in dBFS used for gating."""
    mono = audio.mean(axis=1)
    block = int(MASKING_BLOCK_S * sample_rate)
    n_blocks = mono.shape[0] // block
    if n_blocks == 0:
        raise ValueError("audio shorter than one 400 ms masking block")
    blocks = mono[: n_blocks * block].reshape(n_blocks, block)
    spectrum = np.fft.rfft(blocks, axis=1)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(block, d=1.0 / sample_rate)
    band_power = np.stack(
        [power[:, (freqs >= lo) & (freqs < hi)].sum(axis=1)
         for lo, hi in zip(BAND_EDGES_HZ[:-1], BAND_EDGES_HZ[1:])], axis=1)
    rms_db = 20.0 * np.log10(np.maximum(np.sqrt(np.mean(blocks ** 2, axis=1)),
                                        SILENCE_PEAK))
    return band_power, rms_db


def masking_table(audio, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """What one stem brings to a masking measurement: its band power per 400 ms block and each
    block's level. A function of that stem alone - so anything that measures many pairs (the
    console's observations: every ordered pair of eight stems) builds each stem's table ONCE.
    Raises ValueError for audio shorter than one block."""
    return _block_band_power(_frames_channels(audio), sample_rate)


def _masking_core(table_a: tuple, table_b: tuple, stem_a: str,
                  ) -> tuple[dict[str, float], int, np.ndarray]:
    """The masking of stem_a by stem_b from their tables: (per-band excess, the blocks the
    pair shares, where stem_a is active in them). The one implementation `analyze_masking` and
    `masking_worst` read; it raises the ValueErrors `analyze_masking` always raised."""
    power_a, rms_a_db = table_a
    power_b, _ = table_b
    n_blocks = min(power_a.shape[0], power_b.shape[0])
    active = rms_a_db[:n_blocks] > MASKING_GATE_DBFS
    if not active.any():
        raise ValueError(f"stem '{stem_a}' is never active above {MASKING_GATE_DBFS} dBFS")

    mean_a = power_a[:n_blocks][active].mean(axis=0)
    mean_b = power_b[:n_blocks][active].mean(axis=0)
    # Only bands where stem_a actually holds energy can be "masked" — a band
    # with no stem_a content would report a huge, meaningless excess.
    band_share = mean_a / max(mean_a.sum(), SILENCE_PEAK)
    considered = band_share >= 10.0 ** (MASKING_BAND_FLOOR_DB / 10.0)
    cap = MASKING_EXCESS_CAP_DB
    per_band = {}
    for label, pa, pb, use in zip(BAND_LABELS, mean_a, mean_b, considered):
        if not use:
            continue
        if pb <= 0:
            excess = -cap
        else:
            excess = float(np.clip(10.0 * np.log10(pb / pa), -cap, cap))
        per_band[label] = _round_db(excess)
    if not per_band:
        # No band cleared the stem_a content floor — there is nothing that can be
        # masked. Raise the same well-formed error the callers already handle,
        # not a cryptic "max() arg is an empty sequence".
        raise ValueError(f"stem '{stem_a}' has no band above the masking content floor")
    return per_band, n_blocks, active


def masking_worst(table_a: tuple, table_b: tuple, stem_a: str) -> tuple[str, float]:
    """(worst band, its excess in dB) of stem_a under stem_b, from `masking_table`s: the two
    figures of `analyze_masking` a ranking of pairs needs, without its worst windows or its
    per-channel pass. Same numbers, same ValueErrors."""
    per_band, _, _ = _masking_core(table_a, table_b, stem_a)
    worst = max(per_band, key=per_band.get)
    return worst, per_band[worst]


def analyze_masking(audio_a, audio_b, sample_rate: int, stem_a: str,
                    stem_b: str) -> MaskingAnalysis:
    """Per-band excess of stem_b's energy over stem_a's, over the 400 ms blocks
    where stem_a is active (block RMS > -60 dBFS). Positive = stem_a is masked.
    Definition is contractual: .claude/skills/mix-analysis/SKILL.md."""
    audio_a, audio_b = _frames_channels(audio_a), _frames_channels(audio_b)
    table_a = _block_band_power(audio_a, sample_rate)
    table_b = _block_band_power(audio_b, sample_rate)
    per_band, n_blocks, active = _masking_core(table_a, table_b, stem_a)
    power_a, power_b = table_a[0], table_b[0]
    worst_band = max(per_band, key=per_band.get)
    worst_windows = _worst_masking_windows(
        power_a[:n_blocks], power_b[:n_blocks], active,
        BAND_LABELS.index(worst_band))
    per_channel, release = _spatial_masking(
        audio_a, audio_b, sample_rate, n_blocks, active,
        BAND_LABELS.index(worst_band), per_band[worst_band])
    return MaskingAnalysis(
        stem_a=stem_a, stem_b=stem_b, per_band_excess_db=per_band,
        worst_band=worst_band, worst_band_excess_db=per_band[worst_band],
        worst_windows=worst_windows,
        per_channel_worst_excess_db=per_channel,
        spatial_release_db=release)


def _spatial_masking(audio_a: np.ndarray, audio_b: np.ndarray,
                     sample_rate: int, n_blocks: int, active: np.ndarray,
                     band_idx: int, mono_excess_db: float,
                     ) -> tuple[Optional[dict[str, float]], Optional[float]]:
    """The worst band's excess per channel (Phase 20, mix-analysis skill
    v6): same blocks, same gating, but L and R measured separately — pan
    separation between the stems shows up here when the mono sum can't see
    it. Channels where stem_a holds no energy are skipped (it can't be
    masked where it isn't). spatial_release = mono excess − best channel,
    floored at 0. (None, None) when both stems are mono."""
    if audio_a.shape[1] < 2 and audio_b.shape[1] < 2:
        return None, None
    cap = MASKING_EXCESS_CAP_DB
    per_channel: dict[str, float] = {}
    for ch, label in enumerate(CHANNEL_LABELS):
        ch_a = audio_a[:, [min(ch, audio_a.shape[1] - 1)]]
        ch_b = audio_b[:, [min(ch, audio_b.shape[1] - 1)]]
        power_a_ch, _ = _block_band_power(ch_a, sample_rate)
        power_b_ch, _ = _block_band_power(ch_b, sample_rate)
        mean_a = power_a_ch[:n_blocks][active][:, band_idx].mean()
        mean_b = power_b_ch[:n_blocks][active][:, band_idx].mean()
        if mean_a <= 0:
            continue  # stem_a doesn't live in this channel
        if mean_b <= 0:
            excess = -cap
        else:
            excess = float(np.clip(10.0 * np.log10(mean_b / mean_a), -cap, cap))
        per_channel[label] = _round_db(excess)
    if not per_channel:
        return None, None
    release = _round_db(max(0.0, mono_excess_db - min(per_channel.values())))
    return per_channel, release


def _worst_masking_windows(power_a: np.ndarray, power_b: np.ndarray,
                           active: np.ndarray, band_idx: int) -> list[MaskingWindow]:
    """The worst moments of masking in the worst band: per-active-block excess,
    top blocks merged into runs when adjacent, at most MASKING_WORST_WINDOWS,
    sorted worst-first (mix-analysis skill)."""
    cap = MASKING_EXCESS_CAP_DB
    excess_by_block: dict[int, float] = {}
    for i in np.flatnonzero(active):
        pa, pb = power_a[i, band_idx], power_b[i, band_idx]
        if pa <= 0:
            continue
        excess = -cap if pb <= 0 else float(np.clip(10.0 * np.log10(pb / pa), -cap, cap))
        excess_by_block[i] = excess
    top = sorted(excess_by_block, key=excess_by_block.get,
                 reverse=True)[:MASKING_WORST_WINDOWS]
    runs: list[list[int]] = []
    for block in sorted(top):
        if runs and block == runs[-1][-1] + 1:
            runs[-1].append(block)
        else:
            runs.append([block])
    windows = [MaskingWindow(
        start_s=round(run[0] * MASKING_BLOCK_S, 2),
        end_s=round((run[-1] + 1) * MASKING_BLOCK_S, 2),
        excess_db=_round_db(max(excess_by_block[b] for b in run)),
    ) for run in runs]
    return sorted(windows, key=lambda w: w.excess_db, reverse=True)


# ---------- stem classification (Phase 14) ----------

def _band_shares(band_energy: dict[str, float]) -> dict[str, float]:
    powers = {band: 10.0 ** (db / 10.0) for band, db in band_energy.items()}
    total = max(sum(powers.values()), 1e-12)
    return {band: p / total for band, p in powers.items()}


def classify_stem(stem: StemAnalysis) -> Optional[str]:
    """Deterministic stem class from the stem's NAME, else its already-computed
    metrics (contract: mix-analysis skill v6). First matching rule wins; None =
    leave untagged. Drives context-dependent guardrails (a "bass" tag tightens
    the HPF limit) and the agent's reasoning — never overwrites user tags."""
    if stem.silent or stem.band_energy is None:
        return None
    # The NAME first (2026-08-21). The metric rules below read the acoustics only,
    # and a plucked or synth bass clears the kick rule trivially (low-dominated,
    # and 30 notes a minute is already transient_density >= 0.5) — so a stem
    # plainly called "bass" was tagged `kick` and walked straight into Auto Mix's
    # `drums` bus (TAG_BUS_FAMILY), which is what James watched happen. Naming a
    # track IS a user signal about the instrument, the same kind as a user tag.
    named = instruments.tag_from_name(stem.stem_id)
    if named is not None:
        return named
    shares = _band_shares(stem.band_energy)
    low = shares["sub"] + shares["bass"]
    high = shares["brilliance"] + shares["air"]
    mids = shares["mid"] + shares["high_mid"] + shares["presence"]
    density = stem.transient_density or 0.0
    attack = stem.transient_energy_ratio or 0.0

    broad = sum(1 for share in shares.values()
                if 10.0 * np.log10(max(share, 1e-12)) > TAG_FULL_MIX_FLOOR_DB)
    if low > TAG_LOW_SHARE:
        # a whole song: soft attack, and mids and top a kick or a bass never has
        if (attack < TAG_MIX_ATTACK_MAX and mids >= TAG_MIX_MID_SHARE
                and high >= TAG_MIX_HIGH_SHARE and broad >= TAG_MIX_BANDS):
            return "full_mix"
        # low-dominated: punchy = kick, sustained = bass
        if density >= TAG_TRANSIENT_DENSITY and attack >= TAG_TRANSIENT_RATIO:
            return "kick"
        return "bass"
    if high > TAG_HIGH_SHARE and density >= TAG_TRANSIENT_DENSITY:
        return "cymbal"
    if (mids > TAG_VOCAL_MID_SHARE
            and (stem.sibilance_density or 0.0) >= TAG_VOCAL_SIBILANCE):
        return "vocal"
    if density < TAG_PAD_DENSITY and attack < TAG_PAD_RATIO:
        # sustained: little energy near onsets, even if slow movement or
        # chord beating registers a few flux peaks
        return "pad"
    if broad >= TAG_FULL_MIX_BANDS:
        return "full_mix"
    return None


# ---------- section detection (Phase 5) ----------

def _block_features(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-400 ms-block feature vectors for novelty/labeling: 8 band-power dims
    in dB (floored) plus full-band RMS dB. Returns (features, rms_db)."""
    band_power, rms_db = _block_band_power(audio, sample_rate)
    band_db = 10.0 * np.log10(np.maximum(band_power, SILENCE_PEAK))
    return np.concatenate([band_db, rms_db[:, np.newaxis]], axis=1), rms_db


def _section_boundaries(features: np.ndarray) -> list[int]:
    """Block indices where the mix changes character: peaks of the novelty
    curve (distance between mean features of the 2 s before vs after), at
    least SECTION_MIN_LEN_S apart and away from the edges."""
    from scipy.signal import find_peaks
    n_blocks = features.shape[0]
    half = int(SECTION_NOVELTY_HALF_S / MASKING_BLOCK_S)
    min_len = int(SECTION_MIN_LEN_S / MASKING_BLOCK_S)
    if n_blocks < 2 * min_len or half < 1:
        return [0, n_blocks]
    novelty = np.zeros(n_blocks)
    for i in range(half, n_blocks - half + 1):
        pre = features[i - half:i].mean(axis=0)
        post = features[i:i + half].mean(axis=0)
        novelty[i] = float(np.linalg.norm(post - pre))
    height = novelty.mean() + novelty.std()
    peaks, _ = find_peaks(novelty, height=height, distance=min_len)
    inner = [int(p) for p in peaks if min_len <= p <= n_blocks - min_len]
    return [0] + inner + [n_blocks]


def _stem_active_blocks(audio: np.ndarray, sample_rate: int,
                        n_blocks: int) -> np.ndarray:
    """Boolean active-above-gate per 400 ms block, padded/truncated to n_blocks."""
    mono = _frames_channels(audio).mean(axis=1)
    block = int(MASKING_BLOCK_S * sample_rate)
    padded = np.zeros(n_blocks * block)
    padded[: min(mono.shape[0], padded.shape[0])] = mono[: n_blocks * block]
    rms = np.sqrt(np.mean(padded.reshape(n_blocks, block) ** 2, axis=1))
    return 20.0 * np.log10(np.maximum(rms, SILENCE_PEAK)) > MASKING_GATE_DBFS


def analyze_sections(master_audio, sample_rate: int,
                     stem_audio: dict[str, np.ndarray]) -> SectionAnalysis:
    """Deterministic section detection on the master render (contract:
    .claude/skills/mix-analysis/SKILL.md). Sections that sound alike share a
    letter label; a uniform track is one section spanning the file. v6 adds
    the tempo estimate and, when it is confident, 4/4 bar annotations."""
    audio = _frames_channels(master_audio)
    duration_s = round(audio.shape[0] / sample_rate, 2)
    if np.max(np.abs(audio)) < SILENCE_PEAK:
        return SectionAnalysis(duration_s=duration_s, sections=[Section(
            label="A", start_s=0.0, end_s=duration_s, lufs_i=None,
            band_energy=None, stem_activity={s: 0.0 for s in stem_audio})])
    tempo = _tempo(audio.mean(axis=1), sample_rate)

    features, _ = _block_features(audio, sample_rate)
    n_blocks = features.shape[0]
    boundaries = _section_boundaries(features)
    block = int(MASKING_BLOCK_S * sample_rate)
    stem_active = {stem_id: _stem_active_blocks(stem, sample_rate, n_blocks)
                   for stem_id, stem in stem_audio.items()}

    meter = pyln.Meter(sample_rate)
    sections: list[Section] = []
    centroids: list[np.ndarray] = []  # per assigned label, mean band-db features
    labels: list[str] = []
    for b0, b1 in zip(boundaries[:-1], boundaries[1:]):
        start = b0 * block
        end = audio.shape[0] if b1 == n_blocks else b1 * block  # last gets the tail
        chunk = audio[start:end]
        silent = np.max(np.abs(chunk)) < SILENCE_PEAK
        lufs_i = None if silent else _integrated_lufs(meter, chunk)

        centroid = features[b0:b1].mean(axis=0)
        for known, label in zip(centroids, labels):
            if np.mean(np.abs(centroid - known)) <= SECTION_SAME_LABEL_DB:
                break
        else:
            label = chr(ord("A") + len(centroids))
            centroids.append(centroid)
            labels.append(label)

        correlation = None if silent else _stereo_correlation(chunk)
        start_s, end_s = round(start / sample_rate, 2), round(end / sample_rate, 2)
        bars: tuple[Optional[float], Optional[float]] = (None, None)
        if tempo is not None:
            # 4/4 bars against the detected grid, 1-based at the beat offset
            bpm, _, offset_s = tempo
            bar_s = 4.0 * 60.0 / bpm
            bars = (round((start_s - offset_s) / bar_s + 1.0, 1),
                    round((end_s - offset_s) / bar_s + 1.0, 1))
        sections.append(Section(
            label=label,
            start_s=start_s,
            end_s=end_s,
            lufs_i=_round_db(lufs_i) if lufs_i is not None else None,
            band_energy=None if silent else _band_energy(chunk, sample_rate),
            stem_activity={
                stem_id: round(float(active[b0:b1].mean()), 2)
                for stem_id, active in stem_active.items()
            },
            start_bar=bars[0], end_bar=bars[1],
            width_db=None if silent else _width_db(chunk),
            stereo_correlation=(round(correlation, 2)
                                if correlation is not None else None)))
    return SectionAnalysis(
        duration_s=duration_s, sections=sections,
        tempo_bpm=tempo[0] if tempo is not None else None,
        tempo_confidence=tempo[1] if tempo is not None else None,
        beat_offset_s=tempo[2] if tempo is not None else None)


# ---------- reference profiling (Phase 7) ----------

BAND_WIDTH_FLOOR_DB = -40.0
BAND_WIDTH_CAP_DB = 6.0


def _band_width_db(audio: np.ndarray, sample_rate: int) -> Optional[dict[str, float]]:
    """Per-band stereo width: side-vs-mid power in dB (Welch PSD per channel
    combination). 0 dB = side as strong as mid; floored/capped so dead bands
    don't explode. None for mono sources."""
    if audio.shape[1] < 2:
        return None
    mid = (audio[:, 0] + audio[:, 1]) / 2.0
    side = (audio[:, 0] - audio[:, 1]) / 2.0
    nperseg = _welch_nperseg(4096, mid.shape[0], sample_rate)
    freqs, psd_mid = _welch_psd(mid, sample_rate, nperseg)
    _, psd_side = _welch_psd(side, sample_rate, nperseg)
    width: dict[str, float] = {}
    for label, lo_hz, hi_hz in zip(BAND_LABELS, BAND_EDGES_HZ[:-1], BAND_EDGES_HZ[1:]):
        in_band = (freqs >= lo_hz) & (freqs < hi_hz)
        power_mid = psd_mid[in_band].sum()
        power_side = psd_side[in_band].sum()
        if power_mid <= SILENCE_PEAK and power_side <= SILENCE_PEAK:
            # dead band (v5 fix): flooring only the side against a mid that
            # is even smaller used to read empty bands as +6 "wide"
            width[label] = BAND_WIDTH_FLOOR_DB
        else:
            ratio_db = 10.0 * np.log10(max(power_side, SILENCE_PEAK)
                                       / max(power_mid, SILENCE_PEAK))
            width[label] = _round_db(np.clip(ratio_db, BAND_WIDTH_FLOOR_DB,
                                             BAND_WIDTH_CAP_DB))
    return width


def analyze_reference(audio, sample_rate: int, path: str) -> ReferenceAnalysis:
    """Profile a reference track (contract: mix-analysis skill). Pure — the
    caller reads the file; the reference never enters the mix path."""
    audio = _frames_channels(audio)
    duration_s = round(audio.shape[0] / sample_rate, 2)
    if np.max(np.abs(audio)) < SILENCE_PEAK:
        return ReferenceAnalysis(path=path, duration_s=duration_s, lufs_i=None,
                                 lufs_range=None, crest_factor_db=None,
                                 band_energy=None, band_width_db=None, silent=True)
    meter = pyln.Meter(sample_rate)
    lufs_i = _integrated_lufs(meter, audio)
    lufs_range = _lufs_range(meter, audio, sample_rate)
    return ReferenceAnalysis(
        path=path,
        duration_s=duration_s,
        lufs_i=_round_db(lufs_i) if lufs_i is not None else None,
        lufs_range=_round_db(lufs_range) if lufs_range is not None else None,
        crest_factor_db=_round_db(_crest_factor_db(audio)),
        band_energy=_band_energy(audio, sample_rate),
        band_width_db=_band_width_db(audio, sample_rate),
        sample_peak_dbfs=_round_db(_peak_dbfs(audio)),
        true_peak_dbtp=_round_db(_true_peak_dbtp(audio)))


def _weighted_mean(values: list[tuple[float, float]]) -> Optional[float]:
    total = sum(weight for weight, _ in values)
    if total <= 0.0:
        return None
    return sum(weight * value for weight, value in values) / total


def _weighted_scalar(members: list[ReferenceSetMember], role: str,
                     field: str) -> Optional[float]:
    values = [(member.spec.weight, value)
              for member in members
              if role in member.spec.roles
              if (value := getattr(member.analysis, field)) is not None]
    mean = _weighted_mean(values)
    return _round_db(mean) if mean is not None else None


def _weighted_dict(members: list[ReferenceSetMember], role: str,
                   field: str) -> Optional[dict[str, float]]:
    result: dict[str, float] = {}
    for band in BAND_LABELS:
        values = []
        for member in members:
            if role not in member.spec.roles:
                continue
            source = getattr(member.analysis, field) or {}
            if band in source:
                values.append((member.spec.weight, source[band]))
        mean = _weighted_mean(values)
        if mean is not None:
            result[band] = _round_db(mean)
    return result or None


def _section_sort_key(section: Section) -> float:
    return section.lufs_i if section.lufs_i is not None else -120.0


def _weighted_section_band_energy(
        members: list[ReferenceSetMember],
        mix_sections: Optional[SectionAnalysis]) -> list[ReferenceSectionTarget]:
    if mix_sections is None or len(mix_sections.sections) < 2:
        return []
    tonal_members = [member for member in members
                     if "tonality" in member.spec.roles
                     and member.sections is not None
                     and len(member.sections.sections) >= 2]
    if not tonal_members:
        return []

    mix_ranked = sorted(enumerate(mix_sections.sections),
                        key=lambda item: _section_sort_key(item[1]),
                        reverse=True)
    targets: list[ReferenceSectionTarget] = []
    for rank, (mix_index, mix_section) in enumerate(
            mix_ranked[:REFERENCE_SECTION_TARGETS_MAX]):
        band_values: dict[str, list[tuple[float, float]]] = {
            band: [] for band in BAND_LABELS}
        contributions: list[ReferenceSectionContribution] = []
        for member in tonal_members:
            ref_ranked = sorted(enumerate(member.sections.sections),
                                key=lambda item: _section_sort_key(item[1]),
                                reverse=True)
            ref_index, ref_section = ref_ranked[min(rank, len(ref_ranked) - 1)]
            if not ref_section.band_energy:
                continue
            contributions.append(ReferenceSectionContribution(
                path=member.spec.path,
                weight=member.spec.weight,
                section_index=ref_index,
                label=ref_section.label,
                start_s=ref_section.start_s,
                end_s=ref_section.end_s))
            for band, value in ref_section.band_energy.items():
                if band in band_values:
                    band_values[band].append((member.spec.weight, value))
        band_energy = {
            band: _round_db(mean)
            for band, values in band_values.items()
            if (mean := _weighted_mean(values)) is not None
        }
        if band_energy and contributions:
            targets.append(ReferenceSectionTarget(
                # The metric key is the loudness *rank* (0 = loudest), not the
                # chronological index: section boundaries/order drift as the
                # mix changes pass to pass, but "the loudest section" stays a
                # stable comparison point. The label/times below are the actual
                # section, for display. (chronological index = `mix_index`.)
                mix_index=rank,
                mix_label=mix_section.label,
                mix_start_s=mix_section.start_s,
                mix_end_s=mix_section.end_s,
                band_energy=band_energy,
                confidence=round(len(contributions) / len(tonal_members), 2),
                references=contributions))
    return targets


def analyze_reference_set(
        members: list[ReferenceSetMember],
        mix_sections: Optional[SectionAnalysis] = None) -> ReferenceSetAnalysis:
    """Weighted Phase 36 reference set. Pure combiner: callers do file I/O and
    cache individual ReferenceAnalysis records. Roles decide what contributes:
    tonality -> band_energy, loudness -> lufs/crest/LRA, width -> band_width."""
    if not members:
        raise ValueError("reference set needs at least one reference")
    band_energy = _weighted_dict(members, "tonality", "band_energy")
    band_width = _weighted_dict(members, "width", "band_width_db")
    lufs_i = _weighted_scalar(members, "loudness", "lufs_i")
    lufs_range = _weighted_scalar(members, "loudness", "lufs_range")
    crest = _weighted_scalar(members, "loudness", "crest_factor_db")
    section_targets = _weighted_section_band_energy(members, mix_sections)
    path = (members[0].spec.path if len(members) == 1
            else f"reference set ({len(members)})")
    return ReferenceSetAnalysis(
        path=path,
        references=members,
        lufs_i=lufs_i,
        lufs_range=lufs_range,
        crest_factor_db=crest,
        band_energy=band_energy,
        band_width_db=band_width,
        section_targets=section_targets,
        section_fallback=not bool(section_targets),
        silent=not any((band_energy, band_width, lufs_i, lufs_range, crest)))
