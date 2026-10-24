"""Session state and the proposal lifecycle. Single source of truth for audio state.

Responsibilities (see docs/02-architecture.md):
- Load a project directory: stems/*.wav + session.json (created if absent).
- Proposal lifecycle: propose (validate via guardrails) -> apply (atomic) -> undo.
- Track cumulative per-stem gain within the current pass for the guardrail clamp.
- Persist state after every mutation; state must survive process restart.

NOT responsible for: DSP (dsp.py), metrics (analysis.py), transport (mcp_server.py).
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Optional

from .guardrails import FADER_DB_RANGE, LIMITS, validate_proposal
from .reference_monitor import UTILITY_TOGGLES   # the Utility's switches: snapped to 0 / 1
from .models import (
    BAND_LABELS,
    ApplyResult,
    BandWidthAction,
    BusState,
    ClipAction,
    CompAction,
    DeessAction,
    DynamicEqAction,
    EqAction,
    EqBand,
    GateAction,
    ImagerAction,
    LimiterAction,
    MbBand,
    MetricMeasurement,
    MultibandCompAction,
    OttAction,
    OverdriveAction,
    ChorusAction,
    FlangerAction,
    AutoFilterAction,
    ProposalRequest,
    ProposalResult,
    ProposalSummary,
    ReturnConvolutionAction,
    ReturnDelayAction,
    ReturnReverbAction,
    SpectrogramAction,
    ReturnState,
    SaturationAction,
    SessionState,
    SidechainDuckAction,
    StemState,
    TiltAction,
    TransientAction,
    UtilityAction,
    Violation,
    WidthAction,
)

SESSION_FILE = "session.json"
# The saved checkpoint (reader S1·R3, 2026-10-03). session.json is written at every mix move and
# the edit file only at ⌘S, so "Don't Save" kept the mix it had just declined. The DAW copies
# session.json here each time a save reaches the edit file (with the user's moves it had not yet
# handed over, which that edit file contains), and leaves the marker when the user declines the
# unsaved work; `daw_bridge.honour_discard` puts the checkpoint back before anything reads the
# session. The three names are the DAW's too (DawSession.h; test_saved_checkpoint.py pins them).
SAVED_SESSION_FILE = "session.saved.json"
SAVED_COEDITS_FILE = "session.saved.coedits.json"
DISCARD_MARKER_FILE = "session.discard"
# The user's moves the DAW could not hand over before it went (P2 batch T, readers S1·R12 /
# S3·R2): the brain was down, or still winding up a cancelled turn. The edit file has them and
# this session has not, so the next open applies them (`daw_bridge.apply_owed_coedits`) before
# its chains are laid over the edit - once, and never after a "Don't Save". The DAW's name too
# (DawSession.h kOwedCoeditsFile; test_reader_p2_t.py pins the pair).
OWED_COEDITS_FILE = "session.owed.coedits.json"
BOUNCE_DIRNAME = "bounces"
STATE_VERSION = 2  # v2: the chain LIST is the render order (Ableton-style device freedom) — see _migrate_to_list_order
MASTER = "master"
MAX_RETURNS = 12  # Ableton parity: at most 12 send/return tracks (audio/MIDI tracks themselves are unlimited)
# How many removed groups / returns are kept so the DAW's undo can ask for one back
# (Session._bury / restore_routing). Undo is LIFO and its history dies with the app, so the
# oldest are the ones nothing can still reach.
ROUTING_TOMBSTONES_MAX = 32

# A session.json written by a NEWER build is read as far as this one understands it and written
# back at this build's shape (reader D8·R10). Nothing refuses it - a refusal would brick the
# project in the older app - but brain.log says so, once per file and version.
_NEWER_FILES_SAID: set[tuple[str, int]] = set()
# Send / return LEVEL bounds. Named because they are a CONTRACT, not an implementation detail:
# `send#i.send_db` and `.return_db` are live-ride targets the native engine applies straight to
# the audio thread, so its clamp table (native/Source/DeviceSpecs.h) has to carry the same
# numbers, and test_device_clamp_parity pins the two together against THESE (2026-09-10 — the
# native side had no bound at all, and a modifier could put ~1.0e5 linear on a send bus).
SEND_DB_RANGE = (-60.0, 0.0)
RETURN_DB_RANGE = (-60.0, 6.0)
# Proposal-record retention (scale/soak audit 2026-08-09): the last N stay undoable/queryable;
# older records trim away with their applied gain folded into trimmed_gain_totals (the session
# gain-budget guardrail keeps counting them). Comfortably above the agent-side undo depth (200).
_MAX_PROPOSAL_RECORDS = 400

# Action types the engine can render (all of docs/03-mcp-tools.md as of Phase 15).
# Phase 28 adds the agent's routing primitives: group (create bus + route
# members) and route (re-home one stem). Routing is the agent's now.
SUPPORTED_ACTION_TYPES = {"gain", "eq", "tilt", "dynamic_eq", "comp", "deess", "pan", "width", "limiter",
                          "saturation", "clip", "gate", "transient", "duck", "mute", "exclude",
                          "utility",  # device freedom: Ableton-style Utility (gain/width/mono/phase)
                          "imager",  # 2026-07-05: 3-band stereo imager (the band_width upgrade)
                          "ott",  # 2026-07-07: 3-band upward+downward comp (OTT-style)
                          "overdrive",  # 2026-07-07: drive-pedal distortion
                          "chorus",  # 2026-09-06: modulation family (chorus/ensemble)
                          "flanger",  # 2026-09-06: modulation family (swept comb + feedback)
                          "auto_filter",  # Phase 63: modulation family (swept resonant filter)
                          "match_eq", "mb_comp", "band_width", "group", "route",
                          "return_reverb", "return_convolution", "return_delay", "send",
                          "spectral_denoise",  # Phase 29 repair de-noise
                          "tune",  # Phase 31 tuning correction (stems only)
                          "transpose",  # Phase 32 whole-song key change (master only)
                          "spectrogram"}  # analyzer (render no-op) — operable, not just visible

# What each action type may target: the master bus takes gain and its own
# chain (eq -> match_eq -> bus comp -> mb comp -> limiter); the limiter is
# master-only (docs/03).
MASTER_ACTION_TYPES = {"gain", "eq", "tilt", "match_eq", "dynamic_eq", "comp", "mb_comp", "limiter",
                       "clip",  # Phase 41: pre-limiter peak clipper (also stems +
                                # busses — see _STEM_KINDS / BUS_ACTION_TYPES)
                       "transpose",  # Phase 32: whole-song key change, master only
                       # Phase 43 device freedom: the AGENT can also propose the color/dynamics devices
                       # on the master (real mastering moves) — the user already hand-adds them (Stage 2).
                       "gate", "deess", "transient", "band_width", "imager",
                       "utility",  # Ableton-style Utility (gain trim charged to the gain budget)
                       # `width` was missing while the USER could add a master Width from the
                       # browser (DeviceChainView::addableTypesFor) — the agent had to reach the
                       # same result through utility.width. Total-agent-control: same surface.
                       "width",
                       "spectrogram",  # analyzer (render no-op)
                       # 2026-07-20: the space devices are free inserts, master included
                       # (a touch of IR reverb on a master is a real mastering move).
                       "return_reverb", "return_delay", "return_convolution"}

# Group busses (Phase 23) take the stem palette minus the stem-bound moves:
# no duck (key rendering is stem machinery), no rides — and always whole-file (no
# time_range_s). NB this is the AGENT PROPOSAL palette; pan / width / muted became
# real bus STATE on 2026-09-05 (the group strip's own scalars, via user_edit), which
# this comment used to deny.
BUS_ACTION_TYPES = {"gain", "eq", "tilt", "match_eq", "dynamic_eq", "comp", "deess", "saturation",
                    "gate", "transient", "mb_comp", "ott", "overdrive", "chorus", "flanger",
                    "auto_filter",   # Phase 63 - on a group its envelope follows the group (no trigger)
                    "band_width",
                    # `clip` since 2026-09-08 (device audit 05-F3): it was already in _BUS_KINDS
                    # and the bus renderer, and the user can add one from the dock — so a bus
                    # clipper rendered perfectly well and ONLY the agent's proposal path refused
                    # it. Refusing what the engine plainly does is a rule the agent cannot learn
                    # from the audio.
                    "clip",
                    "imager", "utility", "spectrogram",
                    # 2026-07-20: space devices as inserts (a reverb ON the drum bus, not
                    # only a send off it) — the return form targets return_id instead.
                    "return_reverb", "return_delay", "return_convolution"}

STEM_DEFAULTS = {"gain_db": 0.0, "pan": 0.0, "width": 1.0, "tags": [],
                 "chain": [], "rides": [], "muted": False, "excluded": False,
                 "bus": None, "automation_enabled": True, "sends": {},
                 # Phase 41: the host-timeline offset (s) of this stem's capture's
                 # first frame, so a duck keys off the RIGHT part of its key stem
                 # even when tracks were captured in separate play-throughs.
                 "capture_start_s": 0.0}

# pending: DAW track names grouped into this bus BEFORE they carry audio (an empty track has
# no brain stem yet — 2026-07-19 empty-track grouping). They hold the bus alive + are listed in
# bus_topology so the DAW draws the group; _discover_stems adopts each one as a real member the
# moment a stem of that name registers.
# pan/width/muted (2026-09-05): a group bus carries the same channel-strip scalars a stem
# does. The DAW's bus strip always offered the pan knob, the W field and [M] — they moved the
# live native graph, but the brain had nowhere to keep them, so every one came back refused and
# the next chains push snapped the control home. They are the STRIP's scalars here too, not
# chain devices: effective_bus_chain appends them (like a stem's), so the render, the native
# node and the export oracle read one truth.
BUS_DEFAULTS = {"gain_db": 0.0, "pan": 0.0, "width": 1.0, "muted": False,
                "chain": [], "sends": {}, "pending": []}
# ── What only the DAW's engine knows (full audit 2026-10-02, readers D2·R3 / D2·R4) ───────────
# Three facts about a track live in the Tracktion edit and nowhere in session.json: the VCA
# its fader is under, the Track Delay its node RUNS (the dial summed along its path, plus the
# compensation a negative dial forces on everyone), and its Audio To target when that is not a
# bus or the master. The brain's render sets the live master limiter's make-up and is every
# number the agent reads, and it rendered a mix with every VCA at 0 dB, every Sends Only track
# dry in the master, every track-to-track route summed straight into its bus, and no delay.
# The DAW sends them as `set_stem` co-edits whenever they differ from what the brain holds
# (App.cpp noteEngineFacts); they are stored on the record, absent = the default, and
# dsp.render_* honours them. They are the ENGINE's: the brain never changes one, except that a
# bus route clears `output` — a track's output has ONE owner (see route_stem).
VCA_DB_RANGE = (-60.0, 6.0)          # == dawcmd::kVcaMinDb / kVcaMaxDb
REALIZED_DELAY_MS_RANGE = (0.0, 800.0)   # == dawcmd::kTrackDelayMaxRealizedMs
ENGINE_FACT_FIELDS = ("vca_db", "delay_ms", "output", "clip", "clips", "frozen")

#: Top-level session state the offline RENDER never reads (Phase 54): the ledger, the counters,
#: the co-edit journal, the retention trim's carries, the lists of files a split made or an
#: undo retired. `Session.render_fingerprint` leaves these out and takes EVERYTHING ELSE, so a
#: state key added later is fingerprinted until someone lists it here - a kept render can then
#: only be thrown away too often, never served for audio it does not describe. A key goes in
#: this set only when no `dsp.render_*` path reads it (`test_phase54_render_once.py` holds the
#: set against a fresh session's keys).
RENDER_BOOKKEEPING = frozenset({
    "state_version", "proposals", "proposal_counter", "pass_number", "applied_seq",
    "entry_serial", "routing_serial", "coedit_log", "coedit_batches",
    "separated_wavs", "retired_wavs", "routing_tombstones",
    "trimmed_gain_totals", "trimmed_ranged_gain", "trimmed_eq_curves_db",
    "trimmed_singleton_charges", "trimmed_eq_singletons",
    "native_loudness",
})
# A FIFTH fact (P2 batch T, reader S2·R12; James, 2026-10-05: a frozen track is LOCKED until it
# is unfrozen, one rule at every door): `frozen` = the name of the frozen track that holds this
# stem - itself, or the group it plays into (`bus:<id>`); absent = live. A frozen track plays
# its freeze FILE, so a move on it is exported and not heard. The DAW refuses its own doors;
# this is what lets the brain refuse a PROPOSAL on one (validate_action_targets), whichever of
# chat, a run, Auto Mix or an approved plan made it.
FROZEN_LIVE = ":live"                # the fact's value for a track that is not frozen (DawSession.h too)
# A FOURTH fact (2026-10-04, readers D4·R8 / D1·R2): where the stem's file PLAYS. A stem is one
# wav held from 0:00, and the track that plays it may hold it as one clip put anywhere - a file
# dropped at bar 9, a split stem that sits where its source clip sat (start 16 s, 1.5 s into the
# file, 3 s long). The brain then rendered, measured and placed every ranged move on audio the
# song does not play there. For a track that is ONE plain clip the DAW reports the clip's
# window (`clip_at_s`, `clip_from_s`, `clip_for_s` on a `set_stem`), stored as `clip`
# {at_s, from_s, for_s}; dsp.render_stem reads the file through it. Absent = the whole file at
# 0:00, which is every session written before this and every track of several clips (named to
# the agent by daw_bridge.render_blind_spots). A recorded take is not reported this way: its
# copy is written with the punch-in's silence in it (import_stem `start_s`).
CLIP_AT_MAX_S = 3600.0               # an hour is a song (import_stem's own bound on a lead)


def normalize_clip_window(at_s, from_s, for_s) -> Optional[dict]:
    """A clip window as the record stores it, or None for "the whole file at 0:00".

    ``at_s`` = where the window's first sample plays, ``from_s`` = how far into the file it
    starts, ``for_s`` = how long it is (0 = to the end of the file). A value that is not a
    finite number reads as 0; all three are bounded, because the lead is rendered as silence."""
    import math

    def number(value) -> float:
        try:
            x = float(value)
        except (TypeError, ValueError):
            return 0.0
        return x if math.isfinite(x) and x > 0.0 else 0.0

    at = min(CLIP_AT_MAX_S, number(at_s))
    start = min(CLIP_AT_MAX_S, number(from_s))
    length = min(CLIP_AT_MAX_S, number(for_s))
    if at < 1e-4 and start < 1e-4 and length < 1e-4:
        return None
    return {"at_s": round(at, 6), "from_s": round(start, 6), "for_s": round(length, 6)}


# SEVERAL clips (2026-10-11, the mashup showcase's rehearsal). One window covered a track that
# is one clip; a track of several was still "the file, whole, from 0:00" - and an arranged song
# is nothing else. Every lane of the mashup was several clips of its stem's own file, so the
# agent's loudness, peak and per-part numbers were of eight full-length stems summed, and after
# it moved a clip its measurement "had not updated". For a track whose clips are ALL plain
# pieces of the stem's own file the DAW reports them as a list (`clips` on a `set_stem`):
#   at:from:for:gain_db:fade_in_s:fade_out_s ; ...   ("-" = no list)
# stored as `clips`, in timeline order; `clip` and `clips` are never both kept. Anything else
# on the lane - another file, a warped, looped, reversed, re-sped or pitched clip, clips that
# overlap, a fade that is not linear - and the DAW reports no list: the file from 0:00, named
# to the agent as before.
CLIP_WINDOWS_MAX = 512               # an hour of sixteenth-bar edits is not an arrangement
CLIP_GAIN_DB_RANGE = (-60.0, 24.0)   # == clip.gain's own bounds (App.cpp)


def normalize_clip_windows(value) -> Optional[list]:
    """The windows of a track of several clips as the record stores them, or None for "no
    list". Takes the DAW's wire text or a stored list. A report that is not six finite numbers
    a clip, holds a clip of no length, or is longer than CLIP_WINDOWS_MAX is not a report:
    nothing of it is kept (half an arrangement is a wrong one)."""
    import math

    if value is None or value == "" or value == "-":
        return None
    if isinstance(value, str):
        rows = [part.split(":") for part in value.split(";") if part.strip()]
    elif isinstance(value, list):
        rows = [[w.get(k) for k in ("at_s", "from_s", "for_s", "gain_db", "fade_in_s", "fade_out_s")]
                if isinstance(w, dict) else None for w in value]
    else:
        return None
    if not rows or len(rows) > CLIP_WINDOWS_MAX:
        return None
    out = []
    for row in rows:
        if row is None or len(row) != 6:
            return None
        try:
            at, start, length, gain, fade_in, fade_out = (float(v) for v in row)
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(v) for v in (at, start, length, gain, fade_in, fade_out)):
            return None
        if length <= 0.0 or at < 0.0 or start < 0.0 or at > CLIP_AT_MAX_S or start > CLIP_AT_MAX_S:
            return None
        length = min(CLIP_AT_MAX_S, length)
        fade_in, fade_out = max(0.0, fade_in), max(0.0, fade_out)
        if fade_in + fade_out > length:                 # both fades fit inside the clip
            scale = length / (fade_in + fade_out)
            fade_in, fade_out = fade_in * scale, fade_out * scale
        out.append({"at_s": round(at, 4), "from_s": round(start, 4), "for_s": round(length, 4),
                    "gain_db": round(min(CLIP_GAIN_DB_RANGE[1], max(CLIP_GAIN_DB_RANGE[0], gain)), 2),
                    "fade_in_s": round(fade_in, 4), "fade_out_s": round(fade_out, 4)})
    return sorted(out, key=lambda w: w["at_s"])


def format_clip_windows(windows) -> str:
    """The stored windows as the DAW sends them - the same text, so the DAW can compare what
    the brain holds with what it would say. "" for no list."""
    if not windows:
        return ""
    return ";".join(f"{w['at_s']:.4f}:{w['from_s']:.4f}:{w['for_s']:.4f}:{w['gain_db']:.2f}"
                    f":{w['fade_in_s']:.4f}:{w['fade_out_s']:.4f}" for w in windows)


# ── Which top-level wavs become stems (2026-10-04, reader S1·R19) ────────────────────────────
# Discovery mints a stem from every top-level wav no record claims. That is the offline
# engine's contract - a project is a folder of stems - and it is wrong for a project the DAW
# holds open: there a stem is a TRACK, and a wav that lands at the top of the folder without
# one (a master exported there, a file copied in from Finder) was rendered into every
# measurement from 0:00 and listed to the agent, while no track played it. The DAW's brain
# (daw_server.Brain.open_session) switches a project to "by registration" once its open has
# read the folder: from then on discovery mints only a file a registration names
# (`minting`, which daw_bridge.import_stem holds around its refresh). Per process and per
# folder, never persisted: the gateway, the benchmarks and every test read folders as before,
# and each open decides again.
_BY_REGISTRATION: dict[str, set] = {}


def stems_by_registration(project_dir, on: bool = True) -> None:
    """Switch a project folder to (or back from) "a stem is minted only by a registration"."""
    key = str(Path(project_dir).resolve())
    if on:
        _BY_REGISTRATION.setdefault(key, set())
    else:
        _BY_REGISTRATION.pop(key, None)


# A folder the DAW has opened and not yet made a project (P3 batch T2, reader S1·R25 d; James,
# 2026-10-06: "write nothing until the first save or the first agent turn"). Every Session of
# such a folder is read-only - the folder is still READ, so the boot's topology and readout are
# what they would be - until the brain releases it (daw_server.Brain.become_project). Per
# process and per folder, never persisted, like the registry above.
_WRITES_HELD: set = set()


def hold_writes(project_dir, on: bool = True) -> None:
    key = str(Path(project_dir).resolve())
    if on:
        _WRITES_HELD.add(key)
    else:
        _WRITES_HELD.discard(key)


def writes_held(project_dir) -> bool:
    return str(Path(project_dir).resolve()) in _WRITES_HELD


class minting:
    """`with minting(project_dir, "gtr.wav"):` - discovery may mint that one file meanwhile.
    A no-op for a folder that is not by-registration."""

    def __init__(self, project_dir, filename: str):
        self._key, self._name = str(Path(project_dir).resolve()), str(filename)
        self._added = False

    def __enter__(self):
        allowed = _BY_REGISTRATION.get(self._key)
        if allowed is not None and self._name not in allowed:
            allowed.add(self._name)
            self._added = True
        return self

    def __exit__(self, *exc):
        if self._added:
            (_BY_REGISTRATION.get(self._key) or set()).discard(self._name)
        return False


class without_minting:
    """`with without_minting(project_dir):` - a Session opened meanwhile mints no stem from the
    folder (the folder is by-registration for the length of the block, whatever it was). For a
    door that must WRITE session state about files it has just put in the folder and must not
    register them: `daw_bridge.separate_track` records which wavs a split made, and the DAW's
    one import path registers each."""

    def __init__(self, project_dir):
        self._key = str(Path(project_dir).resolve())
        self._was = None

    def __enter__(self):
        self._was = _BY_REGISTRATION.get(self._key)
        _BY_REGISTRATION[self._key] = set()
        return self

    def __exit__(self, *exc):
        if self._was is None:
            _BY_REGISTRATION.pop(self._key, None)
        else:
            _BY_REGISTRATION[self._key] = self._was
        return False


def normalize_output_target(value) -> str:
    """A track's Audio To target as the brain stores it: "" (its bus, or the master),
    "sends_only", "ext" (a hardware output — out of the master either way; the device id is
    the DAW's business) or "track:<name>". Anything else reads as ""."""
    text = str(value or "").strip()
    if text in ("", "master"):
        return ""
    if text == "sends_only":
        return "sends_only"
    if text.startswith(("ext:", "device:")) or text == "ext":
        return "ext"
    if text.startswith("track:") and text[6:].strip():
        return "track:" + text[6:].strip()
    return ""


# pre_fader: send tap point. Pre = the source AFTER its devices, BEFORE its fader and pan
# (dsp._source_pre_fader; natively the send before the pan stage, the fader taken back out).
# sends: a return may itself send to a LATER return (cascading; forward-only, feedback-safe).
RETURN_DEFAULTS = {"gain_db": 0.0, "chain": [], "pre_fader": False, "sends": {}}


def project_wavs(project_dir) -> list[str]:
    """The wav files at the top of a project folder, by the names they have on disk, sorted -
    the ONE listing every brain-side reader of the folder uses (reader S1·R17). The extension is
    matched without regard to case, as the DAW's own listing matches it (JUCE's
    `findChildFiles (…, "*.wav")`): `Path.glob("*.wav")` is case-sensitive, so a folder from a
    hardware recorder (`TRACK01.WAV`) was a row of playing lanes in the app and "no audio to
    mix" to the engineer. The name is returned as it is; a record's ``path`` keeps it."""
    try:
        return sorted(p.name for p in Path(project_dir).iterdir()
                      if p.name.lower().endswith(".wav"))
    except OSError:
        return []


def _one_file(a: Path, b: Path) -> bool:
    """True when two paths are one file ON THIS VOLUME (a case-insensitive one says so for names
    that differ only by case; a case-sensitive one does not)."""
    try:
        return a.samefile(b)
    except OSError:
        return False


def reserved_stem_name(name: str) -> bool:
    """A name no stem may carry — the ONE rule both ends apply (full audit 2026-10-02, reader
    S4·R10). ``:`` is the routing lanes' mark (``bus:<id>``, ``return:<id>``, ``sink:null``) and
    ``master`` — in any case: on a case-insensitive disk ``Master.wav`` IS ``master.wav`` — is the
    master bus. The DAW's twin is ``dawsession::isReservedTrackName`` (DawSession.h); the two are
    pinned against one table of names, ``server/tests/data/stem_names.json``, by
    ``test_stem_name_parity.py`` here and the console leg ``--stem-names`` there. Before this the
    app refused three prefixes and the brain every ``:``, so "Vox: lead" was a track with no
    stem."""
    name = (name or "").strip()
    return not name or ":" in name or name.lower() == MASTER


def _return_target(return_id: str) -> str:
    return f"return:{return_id}"


def _rekey_coedit_url(url: str, old_id: str, new_id: str,
                      keys: tuple[str, ...] = ("stem", "source")) -> str:
    """A journaled co-edit URL (``coedit_log``) re-pointed from a renamed target to its new id —
    so an undo that replays the journal still lands on the track (rename_stem / rename_bus /
    rename_return). `keys` names the query parameters that carry the id: a chain target rides as
    ``stem``/``source`` (a stem name, or the decorated ``bus:<id>`` / ``return:<id>``), while the
    send + return-level endpoints name a return BARE in ``return`` — the same rename has to move
    both spellings or a replayed send lands on a return that no longer exists."""
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

    # A co-edit URL never carries a fragment: the dock's order tokens (`eq#0,comp#0`) put a bare
    # `#` in the QUERY, urlsplit read everything after the first one as the fragment, `&stem=`
    # was never seen and the entry stayed under the old name (reader D8·R2) - daw_bridge's
    # `_hash_safe` rule, which this function never applied. urlencode writes it back as %23,
    # which every reader of the journal accepts.
    parts = urlsplit(url.replace("#", "%23"))
    pairs = parse_qsl(parts.query, keep_blank_values=True)
    changed = False
    out = []
    for key, value in pairs:
        if key in keys and value == old_id:
            value, changed = new_id, True
        out.append((key, value))
    return urlunsplit(parts._replace(query=urlencode(out))) if changed else url


def effective_chain(stem_state: dict) -> list[dict]:
    """The processor list in actual render order (see audio-dsp skill):
    gain -> tune -> spectral de-noise -> eq (incl. hpf/lpf) -> match-eq -> de-essers -> compressors ->
    multiband comps -> saturation -> transient -> duck -> gain rides ->
    pan -> band width -> width. Within a type, entries keep their
    application order. Pan/width appear only when non-default. dsp.py builds
    from exactly this list. Rides sit post-dynamics like a console fader
    (Phase 5). Tuning runs first (Phase 31): correct the source's pitch before
    de-noising and shaping it."""
    chain = stem_state["chain"]
    ordered = [{"type": "gain", "gain_db": round(stem_state["gain_db"], 1)}]
    # Phase 43 (device freedom): render the chain in LIST order — the user's
    # arrangement — not grouped by type. band_width is a normal in-list insert now
    # (was pinned post-pan). gain stays first; rides/pan/width remain trailing
    # channel-strip stages (the mixer's scalars, not chain inserts).
    ordered += [e for e in chain if e["type"] in set(_STEM_KINDS)]   # band_width included
    ordered += [{"type": "gain_ride", **ride} for ride in stem_state["rides"]]
    if stem_state["pan"] != 0.0:
        ordered.append({"type": "pan", "pan": stem_state["pan"]})
    if stem_state["width"] != 1.0:
        # ...with its on/off (`width_on`, the stereo width audit SW1: a bypassed strip Width keeps
        # its value; the native box and the offline render both read `enabled`)
        entry = {"type": "width", "width": stem_state["width"]}
        if not stem_state.get("width_on", True):
            entry["enabled"] = False
        ordered.append(entry)
    return ordered


# The SPACE devices — reverb / delay / IR reverb. Since 2026-07-20 they are ORDINARY
# INSERTS on any node (the Live model: a Reverb belongs on a track as readily as on a
# return) as well as the wet engine of an FX return; the only difference is the dry/wet
# `mix` (a return runs 1.0 = wet only, because its dry lives on the source tracks).
# Canonical slot: after the spatial stages, BEFORE the peak ceiling — a tail should be
# limited, not smeared after limiting. The native engine reserves kMaxWetInserts
# of these per chain (see _WET_INSERT_MAX), independently of parallel sends.
_WET_KINDS = ("return_reverb", "return_delay", "return_convolution")
# The Auto Filter's discrete fields (Phase 63): snapped on a co-edit, not clamped.
AUTO_FILTER_SNAPPED = ("slope_db_oct", "lfo_sync", "filter_type", "lfo_shape", "lfo_rate_note")
# The devices that listen to a TRIGGER track (`key_stem_id`). The DAW gives a track ONE sidechain
# input (Tracktion keys a node from one source), so every keyed device on a track names the same
# trigger (Phase 63, James: "one trigger per track"); `trigger_conflict` refuses a second.
KEYED_TYPES = ("duck", "auto_filter")

# Canonical stem render order (the re-sortable middle of effective_chain — gain is
# synthesized first; rides/pan/band_width/width follow after). The user may re-sort it
# via the native DAW drag-to-reorder / device add, persisted as stem_device_order.
_STEM_KINDS = ("tune", "spectral_denoise", "eq", "tilt", "match_eq", "dynamic_eq",
               "gate", "deess", "comp", "mb_comp", "ott", "saturation", "overdrive",
               "chorus", "flanger",   # 2026-09-06: modulation after the colour, before the spatial stages
               "auto_filter",         # Phase 63: the third, beside them
               "transient", "duck",
               "utility",   # device freedom: Ableton Utility (output trim), before the ceiling
               # band_width sits here (same slot as _BUS_KINDS / _MASTER_KINDS). It used to be
               # absent — effective_chain re-added it by hand, so it RENDERED, but _canonical_chain
               # treats an unknown type as unplaceable: a user's drag-to-reorder of Band Width on a
               # TRACK was silently discarded and the device forced to the end of the chain, while
               # the identical drag on a bus was honoured. Chain order is audible and the native
               # monitor honours the list, so the monitor and the export disagreed after any reorder.
               "band_width",
               "imager",    # 2026-07-05: 3-band stereo imager (spatial, before the ceiling)
               ) + _WET_KINDS + (   # 2026-07-20: space devices as free inserts
               "clip", "limiter",   # Phase 43 device freedom: per-stem clip + peak ceiling
               "spectrogram")   # 2026-07-06: pure analyzer, render no-op (kinds membership
                                # only — the effective-chain filters DROP unknown types)

# Canonical master render order (audio-dsp skill). gain is synthesized first; the limiter's
# canonical slot is last so the true-peak ceiling holds by default. The user may re-sort the
# middle via the native drag-to-reorder, and may move the limiter itself — see
# _canonical_chain, which takes `type` or `type#occurrence` tokens.
_MASTER_KINDS = ("transpose", "eq", "tilt", "match_eq", "dynamic_eq", "gate", "deess", "comp",
                 "mb_comp", "ott", "saturation", "overdrive", "chorus", "flanger", "auto_filter",
                 "transient", "utility",
                 "band_width", "imager", "width",
                 ) + _WET_KINDS + (   # 2026-07-20: space devices as free inserts
                 "clip", "limiter", "spectrogram")   # spectrogram = analyzer no-op (2026-07-06)

# Canonical group-bus render order (effective_bus_chain): the stem order minus the
# stem-bound stages. Named here so the device-freedom migration can normalize buses too.
_BUS_KINDS = ("eq", "tilt", "match_eq", "dynamic_eq", "gate", "deess", "comp",
              "mb_comp", "ott", "saturation", "overdrive", "chorus", "flanger", "auto_filter",
              "transient", "utility",
              "band_width", "imager",
              ) + _WET_KINDS + (   # 2026-07-20: space devices as free inserts
              "clip", "limiter", "spectrogram")   # spectrogram = analyzer no-op (2026-07-06)

# FX-return device set (Ableton-style blank returns — a return is a bus fed by sends). Since
# 2026-07-20 this is exactly the bus palette: the wet reverb/delay/convolution are in _BUS_KINDS
# (and _STEM_KINDS / _MASTER_KINDS) too, so a return is no longer the only place they can live.
_RETURN_KINDS = _BUS_KINDS

# How many in-chain space devices ONE chain may host. Hard native ceiling:
# MonitorChain::kMaxWetInserts. Parallel sends have their own twelve-slot budget;
# the combined native storage covers both maxima. Refuse a fourth insert honestly.
_WET_INSERT_MAX = 3
#: Devices the native chain holds a fixed number of slots for, beside the space devices above:
#: MonitorChain::kMaxBandWidth. (An EQ's bank is counted in bands - guardrails eq_bands_max.)
_NATIVE_SLOT_KINDS = {"band_width": 8, "dynamic_eq": 6,   # (dynamic_eq: MonitorChain::kMaxDynEq - pre-merge review 2026-10-06)
                      "deess": 4,   # MonitorChain::kMaxDeess: four per chain at the user's door (James, 2026-10-17)
                      "tilt": 4,    # MonitorChain::kMaxTilt: a fifth played unrideable (2026-10-17, the tilt audit)
                      "spectrogram": 4}   # MonitorChain::kMaxSpectrograms: each reads at its place through a tap of its own (2026-10-21)

# Canonical IR-reverb spaces, in picker order (dsp._IR_SEEDS / _IR_DECAY_SCALE).
IR_MODES = ("room", "hall", "plate")

# Device types the native MonitorChain holds exactly ONE parameter set for (MonitorChain.h
# ChainState: comp/lim/mb/ott/sat/od/clip/deess/gate/trans/duck/img/wid/util). Its step walk
# de-duplicates by type, so a second entry of one of these does not run — `fill` simply
# overwrites the first's params and the LAST one wins. The offline engine renders all of
# them, so a chain with two compressors monitored one way and EXPORTED another (measured:
# 3.98 dB). Refuse the second honestly, the same call `_WET_INSERT_MAX` makes for a fourth
# space device, rather than accepting an edit one of the two engines silently discards.
# NOT singletons: eq (a 32-band bank), dynamic_eq (6), band_width (8), the wet sends.
_NATIVE_SINGLETON_KINDS = ("comp", "limiter", "mb_comp", "ott", "saturation", "overdrive", "chorus",
                           "flanger", "auto_filter",
                           "clip", "gate", "transient", "duck", "imager",
                           "width", "utility")


# Phase 43 device freedom (Stage 3): the limiter is NOT pinned last — the user places it
# freely and the ceiling comes from wherever it sits. An unplaced limiter still lands last via
# its canonical _MASTER_KINDS slot, so the default order is unchanged.
#
# The old justification for unpinning it was "the export's existing hard-clip-at-full-scale
# prevents wrapping if it's moved/removed". That net does not exist at every depth: a 32-bit
# FLOAT export writes through unclamped (JUCE's Float32 converter has no jlimit, unlike Int16 /
# Int24 / Int32), and 32-bit float is a depth the export dialog offers by name. 16- and 24-bit
# deliverables do clamp. Passing overs through is correct FLOAT behaviour and should not be
# "fixed" in the writer — but it is not a safety net, so do not lean on it as one
# (dock audit 2026-09-10).


def _last_of_each_singleton(chain: list[dict], exempt: tuple = ()) -> tuple[list[dict], list[str]]:
    """`chain` with at most ONE whole-file entry of each `_NATIVE_SINGLETON_KINDS` type, and
    what was left out. The LAST is kept, where it stands: that is the one the DAW's node
    plays (it fills one parameter set per type, in order). A time-ranged entry is its own
    slot and is never counted."""
    last: dict[str, int] = {}
    for index, entry in enumerate(chain):
        kind = entry.get("type") if isinstance(entry, dict) else None
        if (kind in _NATIVE_SINGLETON_KINDS and kind not in exempt
                and entry.get("time_range_s") is None):
            last[kind] = index
    kept: list[dict] = []
    dropped: list[str] = []
    for index, entry in enumerate(chain):
        kind = entry.get("type") if isinstance(entry, dict) else None
        if (kind in last and last[kind] != index and entry.get("time_range_s") is None):
            dropped.append(f"{kind} (an earlier one: the engine plays one per chain, the last)")
        else:
            kept.append(entry)
    return kept, dropped


def _canonical_chain(chain: list[dict], kinds: tuple, order) -> list[dict]:
    """Reorder a chain's entries into render order.

    `order` is a list of tokens, each either a bare device TYPE or a
    `type#occurrence` pair:

      ["eq", "comp"]            every eq, then every comp   (the pre-2026-09-10 meaning)
      ["eq#0", "clip#0", "eq#1"] those three entries, exactly there

    Occurrence tokens exist because the type-only channel could not express a chain
    that interleaves two entries of one type, and both ends of it silently regrouped:
    `eq, clip, eq` came back as `eq, eq, clip`, moving the clipper in front of the
    second EQ and changing the render by 3.95 dB peak. The dock reached that with three
    ordinary clicks, because it re-posts the order on every device ADD and REMOVE, not
    just on a drag (dock audit 2026-09-10, P0).

    Bare types still mean what they always did, so stored orders and any caller that
    only knows types keep working — including this function's own callers at load and
    migration time, which pass None and get pure canonical grouping.

    Entries the caller did not place fall back to canonical `kinds` order, and entries
    whose type is not in `kinds` at all (a stem's band_width, a master transpose) keep
    their relative order at the end. Stable within a type.
    """
    by_key: dict[tuple[str, int], dict] = {}
    counts: dict[str, int] = {}
    for entry in chain:
        etype = entry.get("type")
        n = counts.get(etype, 0)
        counts[etype] = n + 1
        by_key[(etype, n)] = entry

    if not order:
        # No caller order: pure canonical grouping. This is the load/migration path, and its
        # result must not move — a re-render before and after must be bit-identical.
        out: list[dict] = []
        for kind in kinds:
            out += [e for e in chain if e.get("type") == kind]
        out += [e for e in chain if e.get("type") not in set(kinds)]
        return out

    placed: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()

    def take(key):
        if key in by_key and key not in seen:
            placed.append(key)
            seen.add(key)

    for token in order:
        etype, _, occ = str(token).partition("#")
        if occ.isdigit():
            take((etype, int(occ)))
        else:                                    # a bare type: every entry of it, here
            for n in range(counts.get(etype, 0)):
                take((etype, n))

    # A PARTIAL order still falls back to canonical for whatever it did not name — an unplaced
    # limiter sits canonical-last, which callers rely on and test_master_device_order_user_reorder
    # pins. The fix for a caller that CANNOT name everything is to make it name everything: the
    # dock now posts one token per entry, box or no box, so a master `transpose` is no longer
    # absent from its own chain's order (which is how it walked past the limiter and put the
    # deliverable 0.634 dB over its stated ceiling).
    for kind in kinds:
        for n in range(counts.get(kind, 0)):
            take((kind, n))
    for key in by_key:                           # types outside `kinds` keep chain order, at the end
        take(key)
    return [by_key[k] for k in placed]


def _dock_order(chain: list[dict], kinds: tuple, order) -> list[dict]:
    """`_canonical_chain` for an order that came from the DOCK (reader E2·R4).

    The dock builds its order from the chain its node holds, and the node never holds a device
    the brain pushes as `print_only` - a master `transpose`, a stem's `tune`, its
    `spectral_denoise` (`monitor_chain.dock_can_name`). So the order cannot name them, and
    `_canonical_chain` puts whatever an order did not name after everything it did: every dock
    add, remove or drag rewrote session.json with the key change AFTER the limiter (the 09-10
    audit measured that shape 0.634 dB over its stated ceiling) and a stem's de-noise after its
    compressor. The producer cannot name what it does not hold, so the consumer keeps it: a
    device the dock could not have named, and did not, stays directly behind the device it was
    behind (at the head if it led the chain) while the others are reordered around it. An order
    that DOES name one (an old journal can) places it as asked, and a nameable device a partial
    order leaves out still falls to its canonical slot - both as before."""
    from .monitor_chain import dock_can_name

    named = {str(token).partition("#")[0] for token in (order or [])}
    held: list[tuple[Optional[dict], dict]] = []       # (the entry it follows, the entry)
    anchor: Optional[dict] = None
    for entry in chain:
        etype = entry.get("type")
        if order and not dock_can_name(etype) and etype not in named:
            held.append((anchor, entry))
        else:
            anchor = entry
    if not held:
        return _canonical_chain(chain, kinds, order)
    kept = {id(entry) for _, entry in held}
    out = _canonical_chain([e for e in chain if id(e) not in kept], kinds, order)
    for follows, entry in reversed(held):              # reversed: a run keeps its own order
        at = 0 if follows is None else next(i for i, e in enumerate(out) if e is follows) + 1
        out.insert(at, entry)
    return out


def _insert_canonical(chain: list[dict], entry: dict, kinds: tuple) -> None:
    """Insert a device entry at its canonical position among the existing entries,
    preserving the order of everything else (stable). The AGENT builds chains this way
    so its mixes render in the intended order — identical to the pre-device-freedom
    grouping — while the USER appends (add_stem_device) and reorders (set_*_device_order)
    freely. For an already-canonical chain this yields canonical order regardless of the
    add order; for a user-reordered chain it respects the existing order and slots the new
    device in by rank."""
    rank = {k: i for i, k in enumerate(kinds)}
    r = rank.get(entry.get("type"), len(kinds))
    for i, e in enumerate(chain):
        if rank.get(e.get("type"), len(kinds)) > r:
            chain.insert(i, entry)
            return
    chain.append(entry)


def effective_master_chain(state: dict) -> list[dict]:
    """Master-bus processors in actual render order (Phases 6 + 15; Phase 32
    adds `transpose` first; Phase 40 adds master-bus `saturation`/`width`; Phase 41
    adds the pre-limiter `clip`; Phase 42 lets the user re-sort via the native
    drag-to-reorder, `master_device_order`): gain -> transpose -> eq -> match-eq ->
    dynamic-eq -> bus comp -> multiband comp -> saturation -> width -> clip ->
    limiter by default. gain stays first, the limiter stays last (ceiling). dsp.py
    renders from exactly this list."""
    chain = state["master_chain"]
    ordered = [{"type": "gain", "gain_db": round(state["master_gain_db"], 1)}]
    # Phase 43 device freedom (Stage 3): LIST order — the limiter renders at its chain position too,
    # no longer pinned last. The user freely places / moves / removes it (the limiter caps the ceiling
    # wherever it sits; the export hard-clip prevents wrapping if it's removed — shared control).
    ordered += [e for e in chain if e["type"] in set(_MASTER_KINDS)]
    return ordered


def effective_bus_chain(bus_state: dict) -> list[dict]:
    """Group-bus processors in actual render order (Phase 23): gain ->
    eq -> match-eq -> dynamic-eq -> de-essers -> compressors -> multiband comps ->
    saturation -> transient -> band width -> pan -> width — the stem order minus
    the stem-bound stages (rides/duck). Within a type, entries keep
    application order. Pan/width appear only when non-default and are the bus
    strip's trailing scalars, exactly as on a stem (2026-09-05). dsp.py renders
    from exactly this list."""
    chain = bus_state["chain"]
    ordered = [{"type": "gain", "gain_db": round(bus_state["gain_db"], 1)}]
    ordered += [e for e in chain if e["type"] in set(_BUS_KINDS)]  # Phase 43: LIST order
    # WIDTH then PAN, the order the deliverable runs (see dsp._render_stem's note): the native
    # graph is MonitorChain-with-width first, Tracktion's VolumeAndPan last, and the two stages do
    # not commute. `pan` is print-only for the native node — the DAW pans the group TRACK, with the
    # balance law daw/patches/0007 adds — so this order is what makes the offline render of a
    # panned bus match its bounce (2026-09-05 review).
    if bus_state.get("width", 1.0) != 1.0:
        entry = {"type": "width", "width": bus_state["width"]}
        if not bus_state.get("width_on", True):
            entry["enabled"] = False
        ordered.append(entry)
    if bus_state.get("pan", 0.0) != 0.0:
        ordered.append({"type": "pan", "pan": bus_state["pan"]})
    return ordered


def effective_return_chain(return_state: dict) -> list[dict]:
    """FX-return processors in actual render order — a blank return is a bus fed by sends, so its
    chain is a FREE device list (Ableton-style): input gain, then every device in LIST order
    (eq/comp/sat/… AND reverb/delay/convolution, freely placeable). dsp.py renders from exactly
    this list serially; the wet reverb/delay/conv replace the running signal at their position."""
    chain = return_state["chain"]
    ordered = [{"type": "gain", "gain_db": round(return_state["gain_db"], 1)}]
    ordered += [e for e in chain if e["type"] in set(_RETURN_KINDS)]   # LIST order (device freedom)
    return ordered


def send_return_entries(state: dict, source: dict) -> list[dict]:
    """The synthetic ``send_return`` monitor entries for one source's sends (Phase 38 reverb/
    delay returns) — shared by the gateway (offline/plugin path) and the DAW brain so both
    resolve returns the SAME oracle-faithful way. Each entry carries the send level and its
    target return's resolved wet chain; the native monitor adds the parallel wet locally.
    Because the return chain is linear, the DAW-summed per-instance wet equals the offline
    shared return. Placed after the source's serial chain (the send taps the post-chain signal).
    Empty when the source has no sends (so a chainless stem is unaffected)."""
    returns = state.get("returns", {})
    out: list[dict] = []
    for return_id, gain_db in sorted(source.get("sends", {}).items()):
        if return_id in returns:
            out.append({"type": "send_return", "send_db": round(float(gain_db), 1),
                        "return": effective_return_chain(returns[return_id])})
    return out


class Session:
    def __init__(self, project_dir: Path, read_only: bool = False):
        """`read_only=True` is for renderers peeking at the current state
        (e.g. the gateway's solo-audio endpoint): never writes session.json,
        so it can't race the engine's own persistence."""
        self.project_dir = Path(project_dir).resolve()
        if not self.project_dir.is_dir():
            raise FileNotFoundError(f"project dir not found: {self.project_dir}")
        self._read_only = read_only or writes_held(self.project_dir)   # not a project yet (S1·R25 d)
        self._session_path = self.project_dir / SESSION_FILE
        # Only a Session built HERE follows session.json (sync_from_disk). A probe copy
        # (probe_copy, built via __new__) is scratch state and must never be reloaded.
        self._detached = False
        self._load_state()
        self._persist()

    def _disk_stamp(self) -> Optional[tuple]:
        """(inode, mtime_ns, size) of session.json, or None when it does not exist. Every
        writer replaces the file (tmp + rename, `_persist`), so another Session's write
        shows up here as a new inode whatever the clock resolution."""
        try:
            st = self._session_path.stat()
        except FileNotFoundError:
            return None
        return (st.st_ino, st.st_mtime_ns, st.st_size)

    def _load_state(self) -> None:
        """Read session.json (or start a fresh state) and normalise it: everything
        construction does before its first persist, shared with sync_from_disk."""
        stamp = self._disk_stamp()   # BEFORE the read: a write racing it only re-syncs later
        if self._session_path.exists():
            state = json.loads(self._session_path.read_text())
        else:
            state = {
                "state_version": STATE_VERSION,
                "stems": {},
                "master_gain_db": 0.0,
                "master_chain": [],
                "proposals": [],
                "pass_number": 1,
                "applied_seq": 0,
            }
        self._state = state
        self._state.setdefault("master_chain", [])  # sessions from < Phase 6
        self._state.setdefault("buses", {})  # sessions from < Phase 23
        self._state.setdefault("returns", {})  # sessions from < Phase 38
        version = self._state.get("state_version", 1)
        if isinstance(version, int) and version > STATE_VERSION:
            said = (str(self._session_path), version)
            if said not in _NEWER_FILES_SAID:
                _NEWER_FILES_SAID.add(said)
                print(f"[session] {self._session_path} was written by a newer build "
                      f"(state_version {version}, this build writes {STATE_VERSION}): fields "
                      "this build does not know are kept as they are and not acted on",
                      file=sys.stderr, flush=True)
        if self._state.get("state_version", 1) < STATE_VERSION:
            self._migrate_to_list_order()
            self._state["state_version"] = STATE_VERSION
        self._one_of_each_singleton()
        self._discover_stems()
        self._stamp_entries()
        self._hold_faders_to_range()
        self._sample_rate_cache = None
        self._stamp = stamp

    def sync_from_disk(self) -> bool:
        """Re-read session.json if another Session wrote it since this one last read or
        wrote it; True when it did (2026-09-10 agent audit P0-1).

        Several Session objects write the same file. A chat turn's engine lives for the
        whole turn, while brain-performed routing, the mid-turn co-edit mirror, in-turn
        stem registration, undo_proposal and a nested run or Auto Mix each write through a
        fresh one. The long-lived one never re-read, so its next persist (any propose, even
        a rejected one) wrote its stale copy over all of them: "add a return, send the
        vocal to it, brighten the vocal" lost the return. The engine calls this before
        every tool (mcp_server.create_server). A file that cannot be read is left for the
        next persist to overwrite, exactly as before."""
        if getattr(self, "_detached", True):
            return False
        stamp = self._disk_stamp()
        if stamp is None or stamp == self._stamp:
            return False
        try:
            self._load_state()
        except (OSError, ValueError):   # json.JSONDecodeError is a ValueError
            return False
        return True

    def _migrate_to_list_order(self) -> None:
        """STATE_VERSION 2 (device freedom): the chain LIST becomes the render order
        (Ableton-style). Legacy chains stored insertion order plus a separate
        device_order type-permutation; normalize each into canonical list order so it
        renders identically, then retire device_order. Idempotent and behaviour-
        preserving — effective_chain still groups by type until the list-order flip,
        so a re-render before/after this migration is bit-identical."""
        for stem in self._state.get("stems", {}).values():
            stem["chain"] = _canonical_chain(
                stem.get("chain", []), _STEM_KINDS, stem.pop("stem_device_order", None))
        self._state["master_chain"] = _canonical_chain(
            self._state.get("master_chain", []), _MASTER_KINDS,
            self._state.pop("master_device_order", None))
        for bus in self._state.get("buses", {}).values():
            bus["chain"] = _canonical_chain(
                bus.get("chain", []), _BUS_KINDS, bus.pop("bus_device_order", None))
        # ...and every snapshot an undo would write back (reader D8·R10): `_restore_targets`
        # puts a snapshot back as it is, so undoing a move made before the migration restored
        # the v1 INSERTION order into a file whose list is now the render order - the comp
        # ran in front of the eq it used to follow.
        for record in self._state.get("proposals", []):
            for target, prior in (record.get("inverse") or {}).items():
                if not isinstance(prior, dict):
                    continue                              # a pre-Phase-6 master snapshot: gain only
                if target == MASTER:
                    prior["master_chain"] = _canonical_chain(
                        prior.get("master_chain", []), _MASTER_KINDS,
                        prior.pop("master_device_order", None))
                elif target.startswith("return:") or "chain" not in prior:
                    continue
                elif "path" in prior:                     # a stem's record names its file
                    prior["chain"] = _canonical_chain(
                        prior["chain"], _STEM_KINDS, prior.pop("stem_device_order", None))
                else:
                    prior["chain"] = _canonical_chain(
                        prior["chain"], _BUS_KINDS, prior.pop("bus_device_order", None))

    def _one_of_each_singleton(self) -> None:
        """A chain from before the one-per-type rule (or a hand-made file) may hold two
        whole-file entries of a `_NATIVE_SINGLETON_KINDS` type: the offline render ran both,
        the DAW's node holds one parameter set per type and the last wins (reader D8·R8).
        Every door refuses a second now; a file that already holds one is brought to what the
        node plays - once, at load, said in brain.log. Idempotent; no STATE_VERSION bump."""
        def tidy(chain: list[dict], where: str, exempt: tuple = ()) -> list[dict]:
            kept, dropped = _last_of_each_singleton(chain, exempt)
            if dropped and not self._read_only:
                print(f"[session] {where}: dropped {', '.join(dropped)} - the engine plays one "
                      "of each per chain, the last", file=sys.stderr, flush=True)
            return kept
        for stem_id, stem in self._state.get("stems", {}).items():
            if isinstance(stem.get("chain"), list):
                stem["chain"] = tidy(stem["chain"], stem_id)
        for bus_id, bus in self._state.get("buses", {}).items():
            if isinstance(bus.get("chain"), list):
                bus["chain"] = tidy(bus["chain"], f"bus {bus_id}")
        # the master may hold more than one limiter (set_master_param's own rule)
        self._state["master_chain"] = tidy(self._state.get("master_chain", []), "master",
                                           exempt=("limiter",))

    # ---------- state access ----------

    @property
    def stems(self) -> dict[str, dict[str, Any]]:
        return self._state["stems"]

    @property
    def master_gain_db(self) -> float:
        return self._state["master_gain_db"]

    @property
    def master_chain(self) -> list[dict]:
        """Master-bus processors in render order (gain entry included)."""
        return effective_master_chain(self._state)

    @property
    def buses(self) -> dict[str, dict[str, Any]]:
        """Group busses (Phase 23): bus_id -> {gain_db, chain}."""
        return self._state["buses"]

    @property
    def returns(self) -> dict[str, dict[str, Any]]:
        """Agent-owned send/return busses (Phase 38)."""
        return self._state["returns"]

    @property
    def tempo_map(self) -> Optional[dict]:
        """The song's tempo map as the DAW's engine holds it (Phase 63) - `{tempos: [[beat, bpm,
        curve]...], time_sigs: [[beat, numerator, denominator]...], key_beats: [...]}`, an engine
        fact. A synced Auto Filter reads quarter notes from it (`tempo_map.TempoMap`); None = the
        DAW has not sent one (4/4 at 120)."""
        return self._state.get("tempo_map")

    def set_tempo_map(self, fact: dict) -> dict:
        """The DAW's engine fact (`/api/set_tempo_map`): the map, checked and kept. Writes only
        when it differs, so the fact's resend at every boot costs nothing."""
        def rows(key, width, cast):
            out = []
            for row in (fact or {}).get(key) or []:
                if not isinstance(row, (list, tuple)) or len(row) < width:
                    raise ValueError(f"set_tempo_map: a {key} row is [{width} numbers], got {row!r}")
                vals = [cast[i](row[i]) for i in range(width)]
                if not all(math.isfinite(float(v)) for v in vals):
                    raise ValueError(f"set_tempo_map: a {key} row is not finite: {row!r}")
                out.append(vals)
            return out
        tempos = rows("tempos", 3, (float, float, float))
        sigs = rows("time_sigs", 3, (float, int, int))
        keys = [float(b) for b in (fact or {}).get("key_beats") or [] if math.isfinite(float(b))]
        if not tempos or tempos[0][0] != 0.0 or any(t[1] <= 0.0 for t in tempos):
            raise ValueError("set_tempo_map: the first tempo is at beat 0 and every bpm is above 0")
        if not sigs or sigs[0][0] != 0.0 or any(n < 1 or d < 1 for _, n, d in sigs):
            raise ValueError("set_tempo_map: the first time signature is at beat 0")
        new = {"tempos": tempos, "time_sigs": sigs, "key_beats": keys}
        if self._state.get("tempo_map") != new:
            self._state["tempo_map"] = new
            self._persist()
        return {"tempo_map": new}

    def set_native_loudness(self, pre_limiter_lufs_i: float) -> dict:
        """The DAW's measure of the mix at the master limiter's input (`/api/set_master_loudness`,
        user QA 2026-10-22, Q14): what the APP played, instrument tracks and all - the brain's own
        render cannot hear those. `daw_bridge.measured_pre_limiter` makes the limiter's make-up from
        it while that render is blind. Not fingerprinted (no render reads it). Writes only when it
        differs."""
        v = float(pre_limiter_lufs_i)
        if not math.isfinite(v) or v < -100.0 or v > 30.0:
            raise ValueError(f"set_master_loudness: pre_limiter_lufs_i is a loudness, got {pre_limiter_lufs_i!r}")
        new = {"pre_limiter_lufs_i": round(v, 2)}
        if self._state.get("native_loudness") != new:
            self._state["native_loudness"] = new
            self._persist()
        return {"native_loudness": new}

    @property
    def automation_enabled(self) -> bool:
        """Phase 27: the session-global automation switch (Ableton's
        re-enable button). Off = every stem's section moves (rides +
        time-ranged entries) render bypassed; per-stem switches gate
        underneath it. Sessions from before Phase 27 default to on."""
        return self._state.get("automation_enabled", True)

    def automation_active(self, stem_id: str) -> bool:
        """Whether this stem's section moves render (global AND stem)."""
        return (self.automation_enabled
                and self.stems[stem_id].get("automation_enabled", True))

    def bus_members(self, bus_id: str) -> list[str]:
        """Sorted stem ids routed to this bus (active or not)."""
        return sorted(stem_id for stem_id, stem in self.stems.items()
                      if self.stem_bus(stem_id) == bus_id)

    def stem_bus(self, stem_id: str) -> Optional[str]:
        """The group a stem's signal is summed into, or None (the master, or its own Audio To
        target — see stem_output: a `track:bus:<id>` target is that group)."""
        kind, dest = self.stem_output(stem_id)
        if kind == "bus":
            return dest
        if kind:
            return None
        bus = self.stems[stem_id].get("bus")
        return bus if bus in self.buses else None

    def return_sources(self, return_id: str) -> dict[str, float]:
        """Source ids feeding a return, with send gains in dB."""
        sources: dict[str, float] = {}
        for stem_id, stem in sorted(self.stems.items()):
            if return_id in stem.get("sends", {}):
                sources[stem_id] = stem["sends"][return_id]
        for bus_id, bus in sorted(self.buses.items()):
            if return_id in bus.get("sends", {}):
                sources[bus_id] = bus["sends"][return_id]
        for rid, ret in self.returns.items():   # cascading: an upstream return feeding this one
            if return_id in (ret.get("sends") or {}):
                sources[f"return:{rid}"] = ret["sends"][return_id]
        return sources

    def applied_bus_chain(self, bus_id: str) -> list[dict]:
        """Ordered processors on a bus. Gain is processor 0."""
        return effective_bus_chain(self._require_bus(bus_id))

    @property
    def pass_number(self) -> int:
        return self._state["pass_number"]

    def applied_chain(self, stem_id: str) -> list[dict]:
        """Ordered processors on a stem as shown to the agent. Gain is processor 0."""
        return effective_chain(self._require_stem(stem_id))

    def get_state(self) -> SessionState:
        """Full session state for the loop (Phase 10): a cheap read — no
        rendering, no analysis. Chains come pre-ordered via effective_chain."""
        return SessionState(
            stems=[StemState(
                stem_id=stem_id,
                gain_db=round(stem["gain_db"], 1),
                pan=stem["pan"],
                width=stem["width"],
                muted=stem["muted"],
                excluded=stem["excluded"],
                tags=list(stem["tags"]),
                rides=list(stem["rides"]),
                applied_chain=effective_chain(stem),
                bus=stem.get("bus"),
                automation_enabled=stem.get("automation_enabled", True),
            ) for stem_id, stem in sorted(self.stems.items())],
            master_gain_db=round(self.master_gain_db, 1),
            master_chain=effective_master_chain(self._state),
            pass_number=self.pass_number,
            automation_enabled=self.automation_enabled,
            buses=[BusState(
                bus_id=bus_id,
                gain_db=round(bus["gain_db"], 1),
                pan=round(bus.get("pan", 0.0), 3),
                width=round(bus.get("width", 1.0), 3),
                muted=bool(bus.get("muted", False)),
                members=self.bus_members(bus_id),
                applied_chain=effective_bus_chain(bus),
                sends={rid: round(gain, 1)
                       for rid, gain in sorted(bus.get("sends", {}).items())},
            ) for bus_id, bus in sorted(self.buses.items())],
            returns=[ReturnState(
                return_id=return_id,
                gain_db=round(ret["gain_db"], 1),
                sources={source: round(gain, 1)
                         for source, gain in self.return_sources(return_id).items()},
                applied_chain=effective_return_chain(ret),
                pre_fader=bool(ret.get("pre_fader", False)),
                sends={rid: round(gain, 1)
                       for rid, gain in sorted((ret.get("sends") or {}).items())},
                pan=round(float(ret.get("pan", 0.0) or 0.0), 3),
                muted=bool(ret.get("muted", False)),
            ) for return_id, ret in sorted(self.returns.items())],
            eq_headroom_db=self._eq_headroom(),
        )

    def _eq_headroom(self) -> dict:
        """eq_headroom_db for every target the agent's EQ ledger touches (untouched targets have
        the full rail, so they are left out to keep get_state small)."""
        from .guardrails import LIMITS, eq_bands_in, eq_headroom_db
        ledger = self._eq_ledger()
        # A FULL BANK has no room either (2026-10-02). The native EQ bank holds eq_bands_max
        # bands per chain and guardrails refuses the next one, so "room" there is a number no
        # move can spend: four match_eqs fill the master's bank, the dB rail still showed a few
        # tenths in one band, and the loop's "EQ rail reached" stop (loop.rail_spent) let the
        # probing agent spend a whole pass on four candidates the bank refused.
        full = {t for t, chain in self._chains_for_guardrails().items()
                if sum(eq_bands_in(e) for e in chain) >= LIMITS["eq_bands_max"]}
        targets = set(ledger["session"]) | set(ledger["trimmed"]) | set(ledger["singletons"]) | full
        # A group's EQ is on every member's path, and a member's on the group's (D9·R7): the
        # room of each is the least of its own and of every path it is on.
        from .guardrails import _signal_path
        parents = self._signal_parents()
        paths = [p for p in (_signal_path(parents, node) for node in parents) if len(p) > 1]
        touched = set(targets)
        for path in paths:
            if touched & set(path):
                targets |= set(path)
        room = {t: eq_headroom_db(ledger, t, [p for p in paths if t in p])
                for t in sorted(targets)}
        for t in full:
            room[t] = {band: {"boost": 0.0, "cut": 0.0} for band in room[t]}
        return room

    def refresh_stems(self, separated: Optional[list[str]] = None) -> list[str]:
        """Re-run stem discovery against the project dir (Phase 10): picks up
        wavs added after open without reopening the project — the gateway's
        in-place import path, so chat/feed state survives. Returns stem ids.
        ``separated`` names the wavs a stem split just wrote (``note_separated``)."""
        if separated:
            self.note_separated(separated)
        self._discover_stems()
        self._sample_rate_cache = None  # a new stem may change the project rate
        self._persist()
        return sorted(self.stems)

    def reset_stem(self, stem_id: str, only_if_excluded: bool = False) -> dict:
        """Restore a stem's record to a fresh registration (STEM_DEFAULTS, path kept).
        The import boundary calls this with ``only_if_excluded=True`` when a stem id is
        re-registered: a deleted DAW track only soft-excludes its stem (the wav must stay
        on disk for the edit's clips + undo), so a NEW track that reuses the name would
        otherwise silently inherit ``excluded: true`` plus the dead track's device chain
        via ``_discover_stems``'s setdefault. A live re-register (same logical track) is
        never excluded, so the guard keeps its chain untouched. Returns {ok, reset}."""
        stem = self.stems.get(stem_id)
        if stem is None:
            return {"ok": False, "reset": False, "error": f"unknown stem {stem_id!r}"}
        if only_if_excluded and not stem.get("excluded"):
            return {"ok": True, "reset": False}
        path = stem["path"]
        # A NEW entity under the old id (D8·R6): the proposals, snapshots and journal entries
        # that name the track this record used to be no longer reach it, or charge it.
        self.stems[stem_id] = {"path": path,
                               **json.loads(json.dumps(STEM_DEFAULTS)),
                               "uid": self._mint_uid(stem_id)}
        self._retire_trimmed_charges(stem_id)
        self._drop_empty_buses()   # the record that was reset may have been a group's last member (D8·R7)
        self._persist()
        return {"ok": True, "reset": True}

    # ---- which stems came out of "Split into stems" (reader E5·R7) ----
    def note_separated(self, filenames: list[str]) -> list[str]:
        """Record that these project wavs were WRITTEN BY A STEM SPLIT. Kept by file name, as
        `retired_wavs` is: a stem is its record's ``path`` whatever it is renamed to, and the
        split's files are on disk before any stem is registered for them. The separation
        "double check" and the AI-cleanup preset consider only these - on ordinary clean
        material their metrics read sibilance as shimmer, sustain as smear and playing in
        time as leakage, and the preset repaired all of it. Returns the list as it stands."""
        kept = self._state.setdefault("separated_wavs", [])
        for name in filenames or []:
            name = Path(str(name)).name
            if name and name not in kept:
                kept.append(name)
        self._persist()
        return list(kept)

    # ---- which co-edit batches this session already holds (pre-merge review 2026-10-06) ----
    COEDIT_BATCHES_KEPT = 64

    def coedit_batches(self) -> list[str]:
        """The ids of the DAW's co-edit batches whose moves are in this session, oldest first."""
        return [str(b) for b in (self._state.get("coedit_batches") or [])]

    def note_coedit_batches(self, batches: list[str]) -> None:
        """Record that these batches are being applied. Written INTO session.json, so the
        record and the moves are one file: a checkpoint copy, a restore and a brain restart
        each see a session and the batches it holds together. A batch the DAW got no answer
        for is sent again under the same id, and one that is recorded here is not applied a
        second time (an `add_device` in it would land twice)."""
        kept = self._state.setdefault("coedit_batches", [])
        fresh = [str(b) for b in (batches or []) if b and str(b) not in kept]
        if not fresh:
            return
        kept.extend(fresh)
        del kept[:-self.COEDIT_BATCHES_KEPT]
        self._persist()

    def separated_stems(self) -> list[str]:
        """The active stems whose file a stem split wrote (sorted)."""
        split = set(self._state.get("separated_wavs") or [])
        return sorted(stem_id for stem_id, stem in self.stems.items()
                      if not stem.get("excluded") and str(stem.get("path", "")) in split)

    def stem_file_owner(self, filename: str) -> Optional[str]:
        """The stem whose record's ``path`` is ``filename`` (discovery's claim rule), or None."""
        for sid, stem in self.stems.items():
            if str(stem.get("path", "")) == filename:
                return sid
        return None

    def import_filename(self, stem_id: str, *, keep_default: bool = False) -> str:
        """The project file an import for ``stem_id`` may write (full audit 2026-10-02, S4·R8).

        ``<stem_id>.wav`` when it is free or is this stem's own LIVE file (the caller writes it
        only when it holds the same audio - see ``keep_default``), or its own file and gone
        from disk (a relink brings it back). Otherwise
        the next free ``<stem_id> N.wav``: the default name can hold
        another stem's audio (a renamed track keeps its original file), a soft-deleted track's
        audio (its clips + the undo of the delete still play it), or a retired file — and the
        import used to copy straight over it. A free name is on no record, not on disk, not
        retired and not another stem's default name. Pure: nothing is written.

        ``keep_default=True`` (reader D4·R1): the default file is on disk and holds OTHER audio
        than the import brings, so it is not written whoever's it is. "This stem's own live
        file" was not enough: a clip stacked on another lane keeps its copy at
        ``<project>/<file name>``, discovery mints that copy as a stem of the same name, and a
        later drop of a different file with that name was then "a re-import of its own file" -
        the stacked clip played the new audio. And a real re-import replaced the file under
        the clips, and the undo history, still playing the old one."""
        default = f"{stem_id}.wav"
        retired = set(self._state.get("retired_wavs", []))
        owner = self.stem_file_owner(default)
        own = owner == stem_id and (not self.stems[stem_id].get("excluded")
                                    or not (self.project_dir / default).exists())
        if not keep_default and (own or (owner is None and default not in retired)):
            return default
        n = 2
        while True:
            name = f"{stem_id} {n}"
            cand = f"{name}.wav"
            if (self.stem_file_owner(cand) is None and cand not in retired
                    and name not in self.stems and not (self.project_dir / cand).exists()):
                return cand
            n += 1

    def claim_stem_file(self, stem_id: str, filename: str) -> dict:
        """Point ``stem_id``'s record at ``filename`` BEFORE discovery sees the file (S4·R8):
        the record is created with STEM_DEFAULTS when there is none, else its ``path`` moves.
        The file it held before is RETIRED, never minted as a ghost stem and never deleted (the
        edit's clips + undo may still play it). A retired file that is claimed is retired no
        longer: a stem plays it. Returns {ok, stem_id, path, retired}."""
        if self.stem_file_owner(filename) not in (None, stem_id):
            raise ValueError(f"claim_stem_file: {filename!r} belongs to another stem")
        if filename in (self._state.get("retired_wavs") or []):
            self._state["retired_wavs"] = [n for n in self._state["retired_wavs"] if n != filename]
        stem = self.stems.get(stem_id)
        retired = None
        if stem is None:
            self.stems[stem_id] = {"path": filename, **json.loads(json.dumps(STEM_DEFAULTS))}
        else:
            old = str(stem.get("path", ""))
            if old and old != filename and (self.project_dir / old).exists():
                bucket = self._state.setdefault("retired_wavs", [])
                if old not in bucket:
                    bucket.append(old)
                retired = old
            stem["path"] = filename
        self._persist()
        return {"ok": True, "stem_id": stem_id, "path": filename, "retired": retired}

    # Everything in session.json that names a stem by id. The record itself moves; these are the
    # references INTO it from elsewhere (2026-09-02 agent-workspace audit — rename endpoint).
    @staticmethod
    def _rekey_chain_refs(chain, old_id: str, new_id: str) -> None:
        for entry in chain or []:
            if isinstance(entry, dict) and entry.get("key_stem_id") == old_id:   # a duck keyed to it
                entry["key_stem_id"] = new_id

    def rename_stem(self, old_id: str, new_id: str) -> dict:
        """A track rename follows into the brain (2026-09-02 agent-workspace audit). The stem id IS
        the track name everywhere the brain speaks about it — chains reply keys, co-edit targets,
        duck keys, group pending lists, sends, the proposal ledger's inverse snapshots — so before
        this endpoint a renamed audio track kept its old id: every later co-edit on it was refused
        ("no such stem") and the agent's resolved chains never landed on its node again. The
        record moves under the new id; its wav does NOT move (the DAW's clips reference the file,
        and a mapped wav must never change under the audio thread) — discovery claims files by
        the record's ``path``, not the id. Returns ``{ok, renamed, stem}``; a name the brain has
        no stem for (an instrument or an empty track) is ``renamed: False``, not an error — the
        DAW's local rename is the whole rename there."""
        new = (new_id or "").strip()
        if not new:
            raise ValueError("rename_stem: the new name is empty")
        if old_id == new:
            return {"ok": True, "renamed": False, "stem": old_id}
        if old_id not in self.stems:
            return {"ok": True, "renamed": False, "stem": old_id, "reason": "no brain stem"}
        if new in self.stems or self.name_taken(new, "stem"):
            raise ValueError(f"rename_stem: '{new}' is taken or reserved "
                             f"(master, an existing stem/bus/return, or a 'bus:'/'return:' prefix)")
        self.stems[new] = self.stems.pop(old_id)
        for bus in self.buses.values():
            bus["pending"] = [new if n == old_id else n for n in bus.get("pending", [])]
        for other in self.stems.values():   # an Audio To target that names this track (D2·R4)
            if other.get("output") == f"track:{old_id}":
                other["output"] = f"track:{new}"
        for holder in (*self.stems.values(), *self.buses.values(), *self.returns.values()):
            self._rekey_chain_refs(holder.get("chain"), old_id, new)
        self._rekey_chain_refs(self._state["master_chain"], old_id, new)
        for record in self._state["proposals"]:
            for action in record.get("actions") or []:
                for field in ("stem_id", "key_stem_id"):
                    if action.get(field) == old_id:
                        action[field] = new
                if isinstance(action.get("members"), list):
                    action["members"] = [new if m == old_id else m for m in action["members"]]
            inverse = record.get("inverse") or {}
            if old_id in inverse:
                inverse[new] = inverse.pop(old_id)
            uids = record.get("target_uids") or {}
            if old_id in uids:
                uids[new] = uids.pop(old_id)
            for prior in inverse.values():
                if isinstance(prior, dict):
                    self._rekey_chain_refs(prior.get("chain"), old_id, new)
                    self._rekey_chain_refs(prior.get("master_chain"), old_id, new)
        self._rekey_trimmed_charges(old_id, new)
        for entry in self._state.get("coedit_log", []):
            # `key` is a Duck's trigger track (set_duck) and `output` an Audio To target
            # (`track:<stem>`): a journal entry that names this track either way has to move
            # with it, or the replay after an undo is refused ("unknown key stem") and the
            # user's Duck is simply gone (reader D8·R2).
            url = _rekey_coedit_url(str(entry.get("url", "")), old_id, new,
                                    keys=("stem", "source", "key"))
            entry["url"] = _rekey_coedit_url(url, f"track:{old_id}", f"track:{new}",
                                             keys=("output",))
        self._persist()
        return {"ok": True, "renamed": True, "stem": new, "was": old_id}

    def _validated_chain(self, src_chain, dest_stem_id: str) -> tuple[list[dict], list[str]]:
        """Check a device chain copied from another stem (or another project) BEFORE it lands
        here, returning (kept, dropped-descriptions). Copied state is untrusted input — a
        foreign session.json can be malformed and even our own can be stale — and one bad
        entry would raise on every later render of the destination. A duck keyed to a track
        this project doesn't have is a dangling cross-track dependency and drops the same way."""
        from pydantic import TypeAdapter

        from .models import Action

        adapter = TypeAdapter(Action)
        chain: list[dict] = []
        dropped: list[str] = []
        for entry in src_chain or []:
            kind = entry.get("type", "?") if isinstance(entry, dict) else "?"
            # An UNKEYED duck ("" key) is a legal, deliberate state, not a broken one: since
            # 2026-09-04 picking Trigger: none DISARMS the duck but keeps the device so its
            # tuning survives the next pick, and both engines render it as a no-op (dsp.py's
            # `key_stem_id in session.stems` guard; MonitorChain's `duckKey.isEmpty()` branch).
            # Dropping it here cost the user the whole device on the next duplicate/paste - and
            # blamed a missing key track, which was not the reason (2026-09-05).
            duck_key = entry.get("key_stem_id") if isinstance(entry, dict) else None
            if kind == "duck" and duck_key and duck_key not in self.stems:
                dropped.append("duck (its key track isn't in this project)")
                continue
            # ...and one keyed to the track it is landing ON (reader D8·R8): a duplicate of the
            # bass onto the kick it ducks under made the kick duck under itself, which
            # `set_duck` and the proposal validators both refuse.
            if kind == "duck" and duck_key and duck_key == dest_stem_id:
                dropped.append("duck (it was keyed to this track itself)")
                continue
            # An Auto Filter is a filter first: one whose trigger cannot come with it keeps the
            # device and follows its own input (Phase 63), and says so.
            if kind == "auto_filter" and duck_key and (duck_key not in self.stems
                                                     or duck_key == dest_stem_id):
                entry = {**entry, "key_stem_id": ""}
                dropped.append("auto_filter's trigger (it follows its own input now)")
            try:                                         # stored entries omit stem_id — inject to validate
                adapter.validate_python({**entry, "stem_id": dest_stem_id})
                chain.append(entry)                      # keep the ORIGINAL storage shape
            except Exception:
                dropped.append(str(kind))
        # One of each `_NATIVE_SINGLETON_KINDS` type, as every other door holds a chain to
        # (D8·R8): each entry was checked alone, so a foreign chain with two saturations landed
        # whole - rendered twice offline, played once by the node.
        chain, twice = _last_of_each_singleton(chain)
        return chain, dropped + twice

    def copy_stem_state(self, src_id: str, dest_id: str) -> dict:
        """Duplicate a stem's MIX STATE onto another stem in the SAME project — the DAW's
        ⌘D / track paste. Everything carries VERBATIM: the device chain, channel scalars,
        tags, rides, sends AND group membership. That is the opposite of the cross-project
        ``adopt_stem_state`` policy (which strips routing and filters sends) and deliberately
        so: this project has the very same buses and returns, so keeping them is what
        "duplicate this track" means. The source may be EXCLUDED (a cut DAW track whose stem
        record survives) — its state is still the thing being copied.
        Returns {ok, devices, devices_dropped, sends, bus}."""
        src = self.stems.get(src_id)
        dest = self.stems.get(dest_id)
        if src is None:
            return {"ok": False, "error": f"copy state: no track {src_id!r} in this project"}
        if dest is None:
            return {"ok": False, "error": f"copy state: no track {dest_id!r} in this project"}
        if src_id == dest_id:
            return {"ok": False, "error": "copy state: source and destination are the same track"}
        s = json.loads(json.dumps(src))                  # deep copy — never alias the source
        for key in ("gain_db", "pan", "width", "width_on", "muted", "tags", "rides", "automation_enabled"):
            if key in s:
                dest[key] = s[key]
        chain, dropped = self._validated_chain(s.get("chain"), dest_id)
        dest["chain"] = _canonical_chain(chain, _STEM_KINDS, None)
        # Same project → the source's returns and bus exist here by construction; a send to a
        # return that has since been deleted is stale, so it drops rather than dangling.
        dest["sends"] = {rid: db for rid, db in (s.get("sends") or {}).items()
                         if rid in self._state["returns"]}
        dest["bus"] = s.get("bus")
        self._drop_empty_buses()   # the copy may have taken a group's last member out of it (D8·R7)
        self._persist()
        return {"ok": True, "devices": [e.get("type") for e in dest["chain"]],
                "devices_dropped": dropped, "sends": sorted(dest["sends"]),
                "bus": dest["bus"]}

    def adopt_stem_state(self, stem_id: str, src_state: dict,
                         chain_only: bool = False) -> dict:
        """Cross-project import (multi-project workspace Inc 1): overwrite a freshly
        registered stem's mix state with a copy of a SOURCE project's stem state, under
        the locked dependency policy — the device chain, channel scalars (gain/pan/width/
        mute), tags, rides and automation flag all carry; BUS ROUTING STRIPS to master
        (the destination has no such group); a send carries only when the destination has
        a return of the same kind (matched by id, else by id prefix — fx/rv/dly/cv, the
        kind encoding add_return mints). ``excluded`` and ``capture_start_s`` never carry
        (the destination registration is live, and the capture offset is a source-timeline
        fact). Every chain entry is VALIDATED against its Action model before it lands —
        a foreign session.json is untrusted input, and one malformed entry would otherwise
        poison this session (every later render raises); a duck whose key stem doesn't
        exist here is a cross-track dependency and drops like a dangling send. Returns
        {ok, sends_kept, sends_dropped, devices_dropped}.

        ``chain_only=True`` (Inc 3 — "grab the chain, not the track"): adopt ONLY the
        validated device chain onto an EXISTING stem — scalars, tags, rides, routing and
        sends all stay this stem's own. Same untrusted-input validation."""
        stem = self.stems.get(stem_id)
        if stem is None:
            return {"ok": False, "error": f"unknown stem {stem_id!r}"}
        src = json.loads(json.dumps(src_state))          # deep copy — never alias source state
        if not chain_only:
            for key in ("gain_db", "pan", "width", "width_on", "muted", "tags", "rides",
                        "automation_enabled"):
                if key in src:
                    stem[key] = src[key]
        chain, devices_dropped = self._validated_chain(src.get("chain"), stem_id)
        stem["chain"] = _canonical_chain(chain, _STEM_KINDS, None)
        if chain_only:                                   # nothing but the devices moves
            self._persist()
            return {"ok": True, "sends_kept": [], "sends_dropped": [],
                    "devices_dropped": devices_dropped}
        stem["bus"] = None                               # policy: routing strips to master
        prefixes = {rid: rid.rstrip("0123456789") for rid in self._state["returns"]}
        kept: dict[str, float] = {}
        dropped: list[str] = []
        for rid, gain_db in (src.get("sends") or {}).items():
            match = rid if rid in prefixes else next(
                (dest for dest, p in prefixes.items() if p == rid.rstrip("0123456789")), None)
            if match is not None:
                kept[match] = gain_db
            else:
                dropped.append(rid)
        stem["sends"] = kept
        self._persist()
        return {"ok": True, "sends_kept": sorted(kept), "sends_dropped": sorted(dropped),
                "devices_dropped": devices_dropped}

    def project_sample_rate(self) -> int:
        """The rate every render happens at (Phase 15): the highest stem
        sample rate — mixed-SR stems resample *up* at read time, never down."""
        if getattr(self, "_sample_rate_cache", None) is None:
            import soundfile as sf
            # A project with NO stems has no rate to take the max of, and `max()` of an empty
            # iterable raises. Every caller until 2026-09-08 ran on a rendered project so none
            # could reach it; the DAW bridge's empty-project path (a ⌘G group made before any
            # audio landed) can, so the nominal default keeps that path answering instead of
            # throwing. Nothing renders at it — there is nothing to render.
            self._sample_rate_cache = max(
                (sf.info(str(self.project_dir / stem["path"])).samplerate
                 for stem in self.stems.values()), default=48000)
        return self._sample_rate_cache

    def render_fingerprint(self) -> str:
        """Everything a render of this session reads, as one hash (Phase 54).

        Two sessions with the same fingerprint render the same audio, so a kept master render
        (`dsp.MASTER_MEMO`) and anything measured on it can be served instead of rendered
        again: the DAW's brain used to re-render the whole mix before every chat message and
        after every reply whether or not anything had changed. It is the state as this object
        holds it (not the file: a read-only session's discovery is in it), minus
        `RENDER_BOOKKEEPING`, plus each stem file's size and time on disk and the project
        rate. Default INCLUDE: see `RENDER_BOOKKEEPING`. Not in it: a file a chain entry only
        names (an impulse response) - as the stem cache's own key leaves it out."""
        import hashlib

        state = {k: v for k, v in self._state.items() if k not in RENDER_BOOKKEEPING}
        files: dict[str, Any] = {}
        stems = self._state.get("stems")
        for stem_id, stem in (stems.items() if isinstance(stems, dict) else ()):
            try:
                stat = (self.project_dir / stem.get("path", f"{stem_id}.wav")).stat()
                files[stem_id] = [stat.st_mtime_ns, stat.st_size]
            except (OSError, AttributeError, TypeError):
                files[stem_id] = None
        try:
            rate = self.project_sample_rate()
        except Exception:  # noqa: BLE001 - a file that cannot be read has no rate to add
            rate = None
        blob = json.dumps({"project": str(self.project_dir), "state": state, "files": files,
                           "rate": rate}, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def set_stem_tags(self, stem_id: str, tags: list[str]) -> None:
        """Tags ("bass", ...) steer context-dependent guardrails. Set by the user
        or UI at project setup, not by the agent."""
        self._require_stem(stem_id)["tags"] = list(tags)
        self._persist()

    def set_stem_capture_start(self, stem_id: str, capture_start_s: float) -> float:
        """Record a stem's capture timeline offset (Phase 41). Capture bookkeeping
        set by the gateway at import time (the plugin reports the host-timeline
        offset of each per-track capture), NOT an audio mutation — so a sidechain
        duck reads the key stem's envelope at the right place even when the key and
        the ducked track were captured separately. Persists across sidecar restarts
        (a single shared offset would lose this on a non-simultaneous capture)."""
        self._require_stem(stem_id)["capture_start_s"] = float(capture_start_s)
        self._persist()
        return float(capture_start_s)

    # User-editable stem fields and their absolute ranges (docs/03, Phase 9).
    USER_EDIT_FIELDS = {
        "gain_db": FADER_DB_RANGE,   # -60..+6: the fader's own range, at every door (2026-10-05)
        "pan": (-1.0, 1.0),
        "width": (0.0, 2.0),
        "muted": None,
        "excluded": None,
        "automation_enabled": None,  # Phase 27: this stem's section moves
        # the strip Width's on/off (2026-10-21, the stereo width audit SW1): a bypassed Width box
        # keeps its value. A width set with no width_on switches it on (a hand on it is heard).
        "width_on": None,
    }

    # Phase 40 (native VST co-edit): the user's direct channel for master-chain
    # device controls — each editable device type -> (param path, lo, hi). "gain"
    # is master_gain_db, handled specially. Ranges mirror the action models.
    MASTER_PARAM = {
        "eq":         ("band.gain_db", -12.0, 12.0),   # == DEVICE_PARAMS["eq"]["gain_db"]: the +6 was the AGENT'S guardrail leaking into the user's primary knob (2026-09-01)
        "dynamic_eq": ("gain_db",      -12.0, 12.0),   # == DEVICE_PARAMS["dynamic_eq"]["gain_db"]: 0 was the AGENT'S downward-only rule leaking into the user's knob (2026-10-17)
        "comp":       ("ratio",          1.0, 10.0),
        "limiter":    ("ceiling_dbtp",  -6.0, -1.0),
        "saturation": ("drive_db",       1.0, 12.0),   # Phase 40 master coloration
        "width":      ("width",          0.0,  2.0),
        "utility":    ("gain_db",      -24.0, 24.0),   # device freedom: Utility primary knob = gain trim
        "clip":       ("threshold_db", -12.0,  0.0),   # Phase 41 pre-limiter clipper
        "tilt":       ("tilt_db",      -12.0, 12.0),   # Phase 41 one-knob tonal tilt
        "mb_comp":    ("mid.ratio",      1.0, 10.0),   # primary knob = mid-band ratio
        "ott":        ("depth",          0.0,  1.0),   # OTT: the one knob that matters
        "overdrive":  ("drive_db",       0.0, 36.0),   # overdrive: primary = drive
        "chorus":     ("mix",            0.0, 1.0),    # chorus: primary = the wet blend
        "flanger":    ("feedback",      -0.95, 0.95),  # flanger: primary = the return path
        "auto_filter": ("cutoff_hz",    20.0, 20000.0),  # Phase 63: primary = where the filter sits
        "transient":  ("attack_db",     -6.0,  6.0),   # Phase 40 stem color (per-track)
        "deess":      ("threshold_db", -60.0,  0.0),
        "band_width": ("width",          0.0,  2.0),
        "imager":     ("width_mid",      0.0,  2.0),   # 3-band imager (primary = mid width)
        "gate":       ("threshold_db", -80.0,  0.0),   # Phase 41 gate/expander (primary = threshold)
        # Space devices (2026-07-20): the primary knob is DRY/WET — the one you reach for on
        # an insert, and the knob that makes the device usable off a return at all.
        "return_reverb":      ("mix", 0.0, 1.0),
        "return_delay":       ("mix", 0.0, 1.0),
        "return_convolution": ("mix", 0.0, 1.0),
        # match_eq has no scalar primary knob (per-band gains) -> tile is display-only.
    }

    def _coedit_chain(self, target: str) -> list[dict]:
        """The chain a USER co-edit targets (Phase 40 + per-track instances): the
        master chain, a stem's chain, or a bus's chain. A track/bus plugin insert
        edits its OWN stem; "master"/"" is the master bus."""
        if target in ("", "master"):
            return self._state["master_chain"]
        if target in self._state["stems"]:
            return self._state["stems"][target]["chain"]
        if target in self._state.get("buses", {}):
            return self._state["buses"][target]["chain"]
        if target.startswith("return:"):   # an FX-return track's reverb/delay chain (user co-edit)
            rid = target.split(":", 1)[1]
            if rid in self._state.get("returns", {}):
                return self._state["returns"][rid]["chain"]
        raise ValueError(f"unknown co-edit target {target!r}")

    def set_master_param(self, device_type: str, occurrence: int,
                         value: float, target: str = "master") -> dict:
        """USER co-edit channel (Phase 40, native VST): set a chain device's primary
        control directly — immediate, no proposal, no guardrails (the user's hands),
        clamped to the control's valid range. Shared-control model: the user and the
        agent operate the same devices. `occurrence` selects the Nth entry of
        `device_type` in the chain (ignored for gain). `target` is master (default),
        a stem id, or a bus id — a per-track insert edits its own stem."""
        chain = self._coedit_chain(target)
        if device_type == "gain":
            # The SAME bounds user_edit/set_stem clamp gain_db to. This hardcoded
            # -24..12 while USER_EDIT_FIELDS allows -60..12, so the identical field
            # had a different floor depending on which affordance the user touched:
            # the channel-strip fader took -40 dB, the device-dock Gain tile silently
            # made it -24 (2026-08-16 review).
            lo, hi = self.USER_EDIT_FIELDS["gain_db"]
            v = float(min(hi, max(lo, value)))
            if target in ("", "master"):
                self._state["master_gain_db"] = v
            elif target in self._state["stems"]:
                self._state["stems"][target]["gain_db"] = v
            elif target in self._state["buses"]:
                self._state["buses"][target]["gain_db"] = v
            else:
                # An FX return (`_coedit_chain` above refused anything else): its level, held
                # to a return's own range. This fell through to `buses[target]` - a KeyError
                # for the one kind of target the door's own lookup accepts.
                return {"type": "gain", "occurrence": 0, "value": self.set_return_level(
                    target.split(":", 1)[1], value)["gain_db"]}
            self._persist()
            return {"type": "gain", "occurrence": 0, "value": round(v, 1)}
        if device_type not in self.MASTER_PARAM:
            raise ValueError(f"set_master_param: unsupported device {device_type!r}")
        path, lo, hi = self.MASTER_PARAM[device_type]
        matches = [e for e in chain if e["type"] == device_type]
        if not 0 <= occurrence < len(matches):
            raise ValueError(f"no {device_type} #{occurrence} on the {target} chain")
        v = float(min(hi, max(lo, value)))
        entry = matches[occurrence]
        if "." in path:                       # nested, e.g. eq -> band.gain_db
            outer, inner = path.split(".", 1)
            entry[outer][inner] = v
        else:
            entry[path] = v
        self._persist()
        return {"type": device_type, "occurrence": occurrence, "value": round(v, 1)}

    # every editable param per master device (the native detail popup) -> (path, lo,
    # hi). Ranges mirror the Action models; eq params live under band.*.
    DEVICE_PARAMS = {
        "eq": {
            "freq_hz": ("band.freq_hz", 20.0, 20000.0),
            "gain_db": ("band.gain_db", -12.0, 12.0),   # user co-edit: symmetric ±12 (agent capped +6 by guardrails)
            "q":       ("band.q", 0.3, 8.0),
            # Phase 46: discrete slope, hpf/lpf only — snapped to EQ_SLOPES in
            # set_master_device_param (not a plain min/max clamp).
            "slope_db_oct": ("band.slope_db_oct", 6.0, 96.0),
            # 2026-07-12 gain staging: the EQ device's Output trim (entry-level; the
            # device's audible trim = the SUM across eq entries — the UI writes occ 0).
            "output_db": ("output_db", -12.0, 12.0),
        },
        # SPACE devices — the user adjusts them wherever they sit (an FX return's wet engine
        # OR a plain insert on a track/bus/master since 2026-07-20). `mix` is the dry/wet:
        # 1.0 = wet only (what a return runs at), lower = blended, the insert's main knob.
        "return_reverb": {
            "decay_s":      ("decay_s", 0.2, 12.0),
            "pre_delay_ms": ("pre_delay_ms", 0.0, 250.0),
            "tone":         ("tone", 0.0, 1.0),
            "mix":          ("mix", 0.0, 1.0),
        },
        "return_delay": {
            "delay_ms": ("delay_ms", 20.0, 2000.0),
            "feedback": ("feedback", 0.0, 0.85),
            "tone":     ("tone", 0.0, 1.0),
            "mix":      ("mix", 0.0, 1.0),
        },
        "return_convolution": {
            "decay_s":      ("decay_s", 0.2, 8.0),
            "pre_delay_ms": ("pre_delay_ms", 0.0, 250.0),
            "tone":         ("tone", 0.0, 1.0),
            "mix":          ("mix", 0.0, 1.0),
            # The SPACE (room/hall/plate) as an index, so the co-edit channel — which
            # carries floats — can reach it. It is the device's headline parameter and
            # had NO edit path in either shell: every IR Reverb a user added was
            # permanently a hall, and only an agent proposal could change it.
            "ir":           ("ir", 0.0, float(len(IR_MODES) - 1)),
        },
        "dynamic_eq": {
            "freq_hz":      ("freq_hz", 20.0, 20000.0),   # user co-edit: full audible range (agent action stays 40–16k)
            "q":            ("q", 0.3, 12.0),
            "threshold_db": ("threshold_db", -60.0, 0.0),
            "ratio":        ("ratio", 1.0, 10.0),
            "gain_db":      ("gain_db", -12.0, 12.0),   # user co-edit: - cut / + boost (agent action stays a cut)
            "attack_ms":    ("attack_ms", 1.0, 100.0),
            "release_ms":   ("release_ms", 20.0, 1000.0),
        },
        "comp": {
            "threshold_db": ("threshold_db", -60.0, 0.0),
            "ratio":        ("ratio", 1.0, 10.0),
            "attack_ms":    ("attack_ms", 0.1, 100.0),
            "release_ms":   ("release_ms", 10.0, 1000.0),
            "makeup_db":    ("makeup_db", 0.0, 12.0),
            "mix":          ("mix", 0.0, 1.0),
            # True M/S comp (native VST): independent mid/side param sets, both compress.
            "mid_threshold_db":  ("mid.threshold_db", -60.0, 0.0),
            "mid_ratio":         ("mid.ratio", 1.0, 10.0),
            "mid_attack_ms":     ("mid.attack_ms", 0.1, 100.0),
            "mid_release_ms":    ("mid.release_ms", 10.0, 1000.0),
            "mid_makeup_db":     ("mid.makeup_db", 0.0, 12.0),
            "mid_mix":           ("mid.mix", 0.0, 1.0),
            "side_threshold_db": ("side.threshold_db", -60.0, 0.0),
            "side_ratio":        ("side.ratio", 1.0, 10.0),
            "side_attack_ms":    ("side.attack_ms", 0.1, 100.0),
            "side_release_ms":   ("side.release_ms", 10.0, 1000.0),
            "side_makeup_db":    ("side.makeup_db", 0.0, 12.0),
            "side_mix":          ("side.mix", 0.0, 1.0),
        },
        "limiter": {
            "ceiling_dbtp": ("ceiling_dbtp", -6.0, -1.0),
            # The limiter's loudness/gain control: drive toward this integrated-LUFS
            # target (the offline render measures + applies the makeup; the live
            # monitor reflects it after the next render). The maximizer's "gain".
            "target_lufs_i": ("target_lufs_i", -24.0, -6.0),
            # Advanced (native VST): recovery speed, lookahead ramp window, stereo-link
            # amount (1 linked / 0 independent), and style (0 Modern/1 Transparent/2 Punchy).
            "release_ms":    ("release_ms", 1.0, 1000.0),
            "lookahead_ms":  ("lookahead_ms", 0.5, 3.0),
            "channel_link":  ("channel_link", 0.0, 1.0),
            "style":         ("style", 0.0, 2.0),
        },
        "saturation": {
            "drive_db":  ("drive_db", 1.0, 12.0),
            "output_db": ("output_db", -12.0, 12.0),   # post-shaper output trim (level makeup)
            "curve":     ("curve", 0.0, 3.0),          # 0 tanh / 1 warm / 2 tube / 3 hard
            # 2026-09-06: run the curve at N x so its harmonics stop aliasing. SINGLE-BAND
            # only — the multiband path below splits with an FFT and is not oversampled.
            "oversample": ("oversample", 1.0, 8.0),
            # Multiband (Phase 45): per-band drive/output/mix + movable crossovers + global mix.
            "low_drive_db":  ("low.drive_db", 0.0, 24.0),
            "low_output_db": ("low.output_db", -12.0, 12.0),
            "low_mix":       ("low.mix", 0.0, 1.0),
            "low_curve":     ("low.curve", 0.0, 3.0),    # per-band waveshape model (else the global one)
            "mid_drive_db":  ("mid.drive_db", 0.0, 24.0),
            "mid_output_db": ("mid.output_db", -12.0, 12.0),
            "mid_mix":       ("mid.mix", 0.0, 1.0),
            "mid_curve":     ("mid.curve", 0.0, 3.0),
            "high_drive_db":  ("high.drive_db", 0.0, 24.0),
            "high_output_db": ("high.output_db", -12.0, 12.0),
            "high_mix":       ("high.mix", 0.0, 1.0),
            "high_curve":     ("high.curve", 0.0, 3.0),
            "crossover_lo_hz": ("crossover_lo_hz", 30.0, 4000.0),
            "crossover_hi_hz": ("crossover_hi_hz", 200.0, 16000.0),
            "mix":             ("mix", 0.0, 1.0),
        },
        "width": {
            "width": ("width", 0.0, 2.0),
        },
        # Device freedom: Ableton-style Utility — gain trim + stereo width + mono fold-down
        # + polarity (phase) invert. mono/invert are 0/1 toggles (stored as 0.0/1.0; the
        # native + offline read them as truthy). The fader stays the separate internal gain.
        # v2 (2026-07-05, Live parity): per-channel polarity + bass mono + DC + balance.
        "utility": {
            "gain_db":      ("gain_db", -24.0, 24.0),
            "width":        ("width", 0.0, 2.0),
            "mono":         ("mono", 0.0, 1.0),
            "invert":       ("invert", 0.0, 1.0),
            "invert_left":  ("invert_left", 0.0, 1.0),
            "invert_right": ("invert_right", 0.0, 1.0),
            "bass_mono":    ("bass_mono", 0.0, 1.0),
            "bass_mono_hz": ("bass_mono_hz", 50.0, 500.0),
            "dc_filter":    ("dc_filter", 0.0, 1.0),
            "balance":      ("balance", -1.0, 1.0),
        },
        "mb_comp": {
            "low_threshold_db":  ("low.threshold_db", -60.0, 0.0),
            "low_ratio":         ("low.ratio", 1.0, 10.0),
            "low_makeup_db":     ("low.makeup_db", 0.0, 12.0),     # per-band output/makeup gain
            "mid_threshold_db":  ("mid.threshold_db", -60.0, 0.0),
            "mid_ratio":         ("mid.ratio", 1.0, 10.0),
            "mid_makeup_db":     ("mid.makeup_db", 0.0, 12.0),
            "high_threshold_db": ("high.threshold_db", -60.0, 0.0),
            "high_ratio":        ("high.ratio", 1.0, 10.0),
            "high_makeup_db":    ("high.makeup_db", 0.0, 12.0),
            "attack_ms":         ("attack_ms", 0.1, 100.0),
            "release_ms":        ("release_ms", 10.0, 1000.0),
            "crossover_lo_hz":   ("crossover_lo_hz", 30.0, 4000.0),    # user-movable band edges
            "crossover_hi_hz":   ("crossover_hi_hz", 200.0, 16000.0),
            "low_up":            ("low.up", 0.0, 1.0),     # per-band upward-compression mode
            "mid_up":            ("mid.up", 0.0, 1.0),     # (0 = compress / boost-quiet when 1)
            "high_up":           ("high.up", 0.0, 1.0),
        },
        # 2026-07-07: OTT — 3-band upward+downward comp. depth/time are the device;
        # per-band up/down rebalance the two sides; gains bracket it.
        "ott": {
            "depth":           ("depth", 0.0, 1.0),
            "time":            ("time", 0.1, 4.0),
            "in_gain_db":      ("in_gain_db", -12.0, 12.0),
            "out_gain_db":     ("out_gain_db", -12.0, 12.0),
            "low_up":          ("low.up", 0.0, 1.0),
            "low_down":        ("low.down", 0.0, 1.0),
            "low_gain_db":     ("low.gain_db", -12.0, 12.0),
            "mid_up":          ("mid.up", 0.0, 1.0),
            "mid_down":        ("mid.down", 0.0, 1.0),
            "mid_gain_db":     ("mid.gain_db", -12.0, 12.0),
            "high_up":         ("high.up", 0.0, 1.0),
            "high_down":       ("high.down", 0.0, 1.0),
            "high_gain_db":    ("high.gain_db", -12.0, 12.0),
            "crossover_lo_hz": ("crossover_lo_hz", 30.0, 4000.0),
            "crossover_hi_hz": ("crossover_hi_hz", 200.0, 16000.0),
        },
        # 2026-07-07: overdrive — drive-pedal distortion (band-pass focus + tone).
        # filter_on is a 0/1 toggle (stored 0.0/1.0, read as truthy).
        "overdrive": {
            "drive_db":  ("drive_db", 0.0, 36.0),
            "filter_on": ("filter_on", 0.0, 1.0),
            "freq_hz":   ("freq_hz", 40.0, 12000.0),
            "q":         ("q", 0.3, 8.0),
            "tone":      ("tone", 0.0, 1.0),
            "output_db": ("output_db", -24.0, 12.0),
            "mix":       ("mix", 0.0, 1.0),
            # 2026-09-06: run the soft-clipper at N x so its harmonics stop aliasing.
            "oversample": ("oversample", 1.0, 8.0),
        },
        # 2026-09-06: chorus / ensemble — the modulation family's first device.
        "chorus": {
            "rate_hz":    ("rate_hz", 0.01, 10.0),
            "depth":      ("depth", 0.0, 1.0),
            "delay_ms":   ("delay_ms", 1.0, 50.0),
            "mix":        ("mix", 0.0, 1.0),
            "voices":     ("voices", 1.0, 3.0),
            "stereo_deg": ("stereo_deg", 0.0, 180.0),
        },
        # 2026-09-06: flanger — the same sweep with a return path. `feedback` is the
        # primary because it is the knob that makes it a flanger at all.
        "flanger": {
            "rate_hz":    ("rate_hz", 0.01, 10.0),
            "depth":      ("depth", 0.0, 1.0),
            "delay_ms":   ("delay_ms", 0.5, 20.0),
            "feedback":   ("feedback", -0.95, 0.95),
            "mix":        ("mix", 0.0, 1.0),
            "stereo_deg": ("stereo_deg", 0.0, 180.0),
        },
        # Phase 63: the Auto Filter. The word fields are stored as their INDEX (filter_type,
        # lfo_shape, lfo_rate_note - models.AUTO_FILTER_*), lfo_sync as 0 / 1 and the slope as
        # 12 / 24, each snapped in set_master_device_param. The trigger (key_stem_id) is a NAME
        # and has its own door (set_key), like the Duck's.
        "auto_filter": {
            "filter_type":    ("filter_type", 0.0, 3.0),
            "slope_db_oct":   ("slope_db_oct", 12.0, 24.0),
            "cutoff_hz":      ("cutoff_hz", 20.0, 20000.0),
            "resonance":      ("resonance", 0.0, 1.0),
            "lfo_amount_oct": ("lfo_amount_oct", 0.0, 6.0),
            "lfo_shape":      ("lfo_shape", 0.0, 5.0),
            "lfo_sync":       ("lfo_sync", 0.0, 1.0),
            "lfo_rate_hz":    ("lfo_rate_hz", 0.01, 20.0),
            "lfo_rate_note":  ("lfo_rate_note", 0.0, 16.0),
            "lfo_phase_deg":  ("lfo_phase_deg", 0.0, 360.0),
            "lfo_stereo_deg": ("lfo_stereo_deg", 0.0, 180.0),
            "env_amount_oct": ("env_amount_oct", -6.0, 6.0),
            "env_attack_ms":  ("env_attack_ms", 0.1, 500.0),
            "env_release_ms": ("env_release_ms", 10.0, 3000.0),
            "mix":            ("mix", 0.0, 1.0),
        },
        # v2 (2026-07-05): detector speed scale + post-shaper output trim.
        "transient": {
            "attack_db":  ("attack_db", -6.0, 6.0),
            "sustain_db": ("sustain_db", -6.0, 6.0),
            "speed":      ("speed", 0.25, 4.0),
            "output_db":  ("output_db", -12.0, 12.0),
        },
        # v2 (2026-07-05): freed dynamics + range floor + wide-band mode (wide 0/1).
        "deess": {
            "threshold_db": ("threshold_db", -60.0, 0.0),
            "freq_hz":      ("freq_hz", 2000.0, 12000.0),
            "ratio":        ("ratio", 1.0, 20.0),
            "attack_ms":    ("attack_ms", 0.1, 20.0),
            "release_ms":   ("release_ms", 10.0, 500.0),
            "range_db":     ("range_db", -80.0, 0.0),
            "wide":         ("wide", 0.0, 1.0),
        },
        "band_width": {
            "width": ("width", 0.0, 2.0),
        },
        # 2026-07-05: 3-band stereo imager — per-band width + movable crossovers.
        # (width_low/... are FLAT keys by design: the low_/mid_/high_ prefixes would
        # nest into sub-objects via writeParam/isNestedParam.)
        "imager": {
            "width_low":       ("width_low", 0.0, 2.0),
            "width_mid":       ("width_mid", 0.0, 2.0),
            "width_high":      ("width_high", 0.0, 2.0),
            "crossover_lo_hz": ("crossover_lo_hz", 30.0, 4000.0),
            "crossover_hi_hz": ("crossover_hi_hz", 200.0, 16000.0),
        },
        # Phase 41: pre-limiter clipper (master) — clip level + soft/hard knee.
        # v2 (2026-07-05): oversample 1/2/4/8 (snapped) = true-peak-safe shaping.
        "clip": {
            "threshold_db": ("threshold_db", -12.0, 0.0),
            "hardness":     ("hardness", 0.0, 1.0),
            "output_db":    ("output_db", -12.0, 12.0),   # post-clip output trim (level makeup)
            "oversample":   ("oversample", 1.0, 8.0),
        },
        # Phase 41: one-knob tilt EQ — tonal tilt + the pivot it tilts around.
        "tilt": {
            "tilt_db":  ("tilt_db", -12.0, 12.0),
            "pivot_hz": ("pivot_hz", 200.0, 8000.0),
            "output_db": ("output_db", -12.0, 12.0),   # 2026-07-12 post trim (gain staging)
        },
        # Phase 41: downward gate/expander (stem/bus) — threshold, steepness, timing.
        # v2 (2026-07-05): hold / range floor / hysteresis + detector-only SC filters
        # (sc_hpf_hz 0 = off, sc_lpf_hz 20000 = off).
        "gate": {
            "threshold_db":  ("threshold_db", -80.0, 0.0),
            "ratio":         ("ratio", 1.0, 20.0),
            "attack_ms":     ("attack_ms", 0.1, 100.0),
            "release_ms":    ("release_ms", 5.0, 2000.0),
            "hold_ms":       ("hold_ms", 0.0, 500.0),
            "range_db":      ("range_db", -80.0, 0.0),
            "hysteresis_db": ("hysteresis_db", 0.0, 24.0),
            "sc_hpf_hz":     ("sc_hpf_hz", 0.0, 2000.0),
            "sc_lpf_hz":     ("sc_lpf_hz", 200.0, 20000.0),
        },
        # Phase 41: the user can dial the agent's sidechain duck directly (native VST
        # co-edit) — on a per-track insert this targets that stem's duck.
        "duck": {
            "threshold_db": ("threshold_db", -60.0, 0.0),
            "depth_db":     ("depth_db", 1.0, 12.0),
            "attack_ms":    ("attack_ms", 1.0, 100.0),
            "release_ms":   ("release_ms", 20.0, 1000.0),
        },
    }

    def set_device_time_range(self, device_type: str, occurrence: int,
                             start_s, end_s, target: str = "master") -> dict:
        """USER co-edit channel: a device's Phase 33 time range (`time_range_s`), or None to
        clear it and make the device whole-file again.

        There was no channel for this at all, so the DAW's cross-track device MOVE could not
        carry it: `entryPersistUrls` posts numeric params only, and a moved phrase compressor
        arrived brain-side as a WHOLE-FILE compressor — a silent change to what the offline
        render and the export produce (dock audit 2026-09-10, P2). The live entry always cloned
        intact; only the brain's copy lost it.
        """
        matches = [e for e in self._coedit_chain(target)
                   if e["type"] == device_type]
        if not 0 <= occurrence < len(matches):
            raise ValueError(f"no {device_type} #{occurrence} on the {target} chain")
        entry = matches[occurrence]
        if start_s is None or end_s is None:
            entry.pop("time_range_s", None)
            self._persist()
            return {"type": device_type, "occurrence": occurrence, "time_range_s": None}
        a, b = float(start_s), float(end_s)
        if b <= a:
            raise ValueError("set_device_time_range: end_s must be after start_s")
        entry["time_range_s"] = [round(a, 3), round(b, 3)]
        self._persist()
        return {"type": device_type, "occurrence": occurrence,
                "time_range_s": entry["time_range_s"]}

    def set_master_device_param(self, device_type: str, occurrence: int,
                                param: str, value: float,
                                target: str = "master") -> dict:
        """USER co-edit channel (Phase 40, native VST): set ANY editable param on a
        device (the device detail popup) — not just the primary knob. Immediate, no
        proposal, clamped. `target` = master / a stem / a bus (per-track insert)."""
        specs = self.DEVICE_PARAMS.get(device_type)
        if specs is None or param not in specs:
            raise ValueError(
                f"set_master_device_param: {device_type!r} has no param {param!r}")
        path, lo, hi = specs[param]
        if device_type == "limiter" and param == "target_lufs_i" and target != MASTER:
            # A loudness target is the MASTER limiter's: off it neither engine reads it (2026-10-17,
            # the limiter audit) - it was stored, accepted and echoed for a stem limiter.
            raise ValueError("a limiter's loudness target (target_lufs_i) is the master's - on "
                             f"{target!r} the limiter is a ceiling only")
        matches = [e for e in self._coedit_chain(target)
                   if e["type"] == device_type]
        if not 0 <= occurrence < len(matches):
            raise ValueError(f"no {device_type} #{occurrence} on the {target} chain")
        if path == "ir":
            # index -> the stored NAME (the engine + native both key off the string)
            entry = matches[occurrence]
            name = IR_MODES[int(min(len(IR_MODES) - 1, max(0, round(value))))]
            entry["ir"] = name
            self._persist()
            return {"type": device_type, "occurrence": occurrence,
                    "param": param, "value": name}
        if device_type == "auto_filter" and path in AUTO_FILTER_SNAPPED:
            # Phase 63: the slope is 12 or 24, the sync a switch, the word fields an index -
            # snapped as every engine reads them (auto_filter_values, the native parse).
            if path == "slope_db_oct":
                v = 24.0 if value >= 18.0 else 12.0
            elif path == "lfo_sync":
                v = 1.0 if value > 0.5 else 0.0
            else:
                v = float(min(hi, max(lo, round(value))))
        elif path.endswith("slope_db_oct"):
            # Phase 46: a discrete set, not a continuous range — snap to nearest.
            v = float(min(self.EQ_SLOPES, key=lambda s: abs(s - value)))
        elif path.endswith("oversample"):
            # clip / saturation / overdrive: power-of-2 factors only. Snap DOWN, not to nearest —
            # BOTH native paths floor (`osReq >= 8 ? 8 : >= 4 ? 4 : >= 2 ? 2 : 1`), so snapping to
            # nearest here meant a knob dragged to 7 read 7 on the dock, ran 4x in the native
            # monitor and the DAW bounce, and persisted 8 for the offline re-render and the export
            # oracle — three different answers to one gesture (2026-09-06 review).
            v = float(8 if value >= 8 else 4 if value >= 4 else 2 if value >= 2 else 1)
        elif device_type == "limiter" and path == "style":
            # A STYLE is a preset of the release and the lookahead (models.LIMITER_STYLE_PRESETS) -
            # it stored a float nothing read, and `device.set limiter.style` answered ok for a no-op
            # (2026-10-17, the limiter audit; 01 F11). Snapped to its index; applied below.
            v = float(min(2, max(0, round(value))))
        elif device_type == "utility" and path in UTILITY_TOGGLES:
            # A Utility's switch is ON or OFF (the Utility audit, 2026-10-21, U5): a stored 0.3 played
            # mono in one engine and not in another. Snapped, as every engine reads it (> 0.5).
            v = 1.0 if value > 0.5 else 0.0
        elif path.endswith("voices"):
            # DeviceSpecs.h claimed "Voices is discrete, snapped on write" and nothing snapped it,
            # so a fractional 2.6 persisted and the three engines disagreed about what it meant
            # (2026-09-06 review). Truncate, matching dsp.py's int() and the native contract cast.
            v = float(min(3, max(1, int(value))))
        else:
            v = float(min(hi, max(lo, value)))
        entry = matches[occurrence]
        if "." in path:
            outer, inner = path.split(".", 1)
            # A saturation entry becomes MULTIBAND the moment a `low` sub-object exists
            # (dsp._saturate dispatches on it), so writing one band without its siblings used
            # to leave a half-formed entry the offline renderer could not read. Seed all three
            # from the flat drive first — the same seeding set_sat_multiband does — so the
            # entry is always either wholly single-band or wholly 3-band.
            if (device_type == "saturation" and outer in ("low", "mid", "high")
                    and outer not in entry):
                drv = float(min(24.0, max(0.0, entry.get("drive_db", 3.0))))
                for b in ("low", "mid", "high"):
                    entry.setdefault(b, {"drive_db": drv, "output_db": 0.0, "mix": 1.0})
                entry.setdefault("crossover_lo_hz", 150.0)
                entry.setdefault("crossover_hi_hz", 2000.0)
            entry.setdefault(outer, {})[inner] = v   # nested (mb_comp bands / M/S comp)
        else:
            entry[path] = v
            # A utility's `invert` IS both channels' flip (2026-10-13). The per-channel flags win
            # over it when they are there (dsp._utility, MonitorChain's parse), and a utility the
            # dock added carries them as false - so `invert` alone flipped nothing that renders,
            # while the live knob flipped both. James's phase test found it: the polarity flip
            # the engineer reached for was silent.
            if device_type == "utility" and path == "invert":
                entry["invert_left"] = entry["invert_right"] = v
            if device_type == "limiter" and path == "style":
                from .models import LIMITER_STYLE_PRESETS
                entry["release_ms"], entry["lookahead_ms"] = LIMITER_STYLE_PRESETS[int(v)]
        self._persist()
        return {"type": device_type, "occurrence": occurrence,
                "param": param, "value": round(v, 2)}

    EQ_SHAPES = ("peak", "low_shelf", "high_shelf", "hpf", "lpf", "notch",
                "bandpass", "tilt_shelf")
    EQ_SLOPES = (6, 12, 24, 48, 96)

    def set_eq_shape(self, occurrence: int, shape: str,
                     target: str = "master") -> dict:
        """USER co-edit channel (Phase 40, native VST): change an EQ band's filter
        shape (peak / shelves / hpf / lpf / notch / bandpass / tilt_shelf — see
        EQ_SHAPES) from the device detail popup. `target` = master / a stem / a
        bus (per-track insert)."""
        if shape not in self.EQ_SHAPES:
            raise ValueError(f"set_eq_shape: unknown shape {shape!r}")
        matches = [e for e in self._coedit_chain(target) if e["type"] == "eq"]
        if not 0 <= occurrence < len(matches):
            raise ValueError(f"no eq #{occurrence} on the {target} chain")
        matches[occurrence]["band"]["shape"] = shape
        self._persist()
        return {"type": "eq", "occurrence": occurrence, "shape": shape}

    def set_device_channel(self, device_type: str, occurrence: int, channel: str,
                           target: str = "master", seed: bool = False) -> dict:
        """USER co-edit channel (Phase 41, native VST): set an eq/comp device's M/S
        mode (stereo / mid / side) from the device UI. Phase 46 adds left / right,
        eq only (a direct per-channel process, no M/S encode/decode). `target` =
        master / stem / bus."""
        if device_type not in ("eq", "comp"):
            raise ValueError(f"set_device_channel: {device_type!r} has no M/S channel")
        valid_channels = (("stereo", "mid", "side", "left", "right") if device_type == "eq"
                          else ("stereo", "mid", "side"))
        if channel not in valid_channels:
            raise ValueError(f"set_device_channel: unknown channel {channel!r}")
        matches = [e for e in self._coedit_chain(target) if e["type"] == device_type]
        if not 0 <= occurrence < len(matches):
            raise ValueError(f"no {device_type} #{occurrence} on the {target} chain")
        entry = matches[occurrence]
        # True M/S comp: first time mid/side is selected, seed both param sets from the
        # flat params — the clicked channel keeps its ratio (compresses), the other starts
        # at ratio 1 (off) so the sides aren't surprise-compressed. Mirrors native seedCompMs.
        # `seed`: the DAW seeded its own entry's halves on this flip (its entry had none - an undo
        # took the last flip back), so the halves held here are stale and are written again from
        # the knobs as they stand (audit 2026-10-17, the compressor: the brain kept the old ones).
        if device_type == "comp" and channel in ("mid", "side") and ("mid" not in entry or seed):
            flat = {k: float(entry.get(k, d)) for k, d in (
                ("threshold_db", -18.0), ("ratio", 2.0), ("attack_ms", 10.0),
                ("release_ms", 120.0), ("makeup_db", 0.0), ("mix", 1.0))}
            off = {"ratio": 1.0, "makeup_db": 0.0}   # the half not chosen passes through (dsp._apply_comp)
            entry["mid"] = {**flat, **({} if channel == "mid" else off)}
            entry["side"] = {**flat, **({} if channel == "side" else off)}
        entry["channel"] = channel
        self._persist()
        return {"type": device_type, "occurrence": occurrence, "channel": channel}

    def set_device_band(self, occurrence: int, band: str = "",
                        lo_hz: float | None = None, hi_hz: float | None = None,
                        target: str = "master") -> dict:
        """USER co-edit channel (2026-07-05): pick a band_width device's band LABEL
        (one of the fixed Phase 15 bands) from the device UI — immediate, no
        proposal. The label is the session-side truth; monitor_chain_params
        resolves it to the native bandpass center/Q. `target` = master / a stem /
        a bus (per-track insert).

        2026-07-19 (free band scrub): pass explicit ``lo_hz``/``hi_hz`` instead of a
        label to place the band anywhere — the custom edges are stored on the entry
        and OVERRIDE the label everywhere (offline render, native monitor); picking
        a preset label again clears them."""
        matches = [e for e in self._coedit_chain(target)
                   if e["type"] == "band_width"]
        if not 0 <= occurrence < len(matches):
            raise ValueError(f"no band_width #{occurrence} on the {target} chain")
        entry = matches[occurrence]
        if lo_hz is not None and hi_hz is not None:
            lo, hi = float(lo_hz), float(hi_hz)
            if not (20.0 <= lo < hi <= 20000.0):
                raise ValueError(f"set_device_band: bad custom band {lo!r}..{hi!r}")
            entry["lo_hz"], entry["hi_hz"] = round(lo, 1), round(hi, 1)
        elif band in BAND_LABELS:
            entry["band"] = band
            entry.pop("lo_hz", None)   # a preset pick clears the custom edges
            entry.pop("hi_hz", None)
        else:
            raise ValueError(f"set_device_band: unknown band {band!r}")
        self._persist()
        return {"type": "band_width", "occurrence": occurrence,
                "band": entry.get("band"),
                "lo_hz": entry.get("lo_hz"), "hi_hz": entry.get("hi_hz")}

    def holds_send_source(self, source_id: str) -> bool:
        """Is `source_id` a track whose sends this session keeps - a stem, a group bus
        (``bus:<id>`` or bare) or a return (``return:<id>`` or bare)? An instrument track and an
        audio lane with no stem yet are not: their sends live in the DAW's edit, and the DAW
        dials them itself (full audit 2026-10-02, reader S1·R5 = S4·R21). The same resolution
        `set_send` makes, without the write."""
        if source_id.startswith("return:"):
            return source_id.split(":", 1)[1] in self._state["returns"]
        if source_id.startswith("bus:"):
            return source_id.split(":", 1)[1] in self.buses
        return (source_id in self.buses or source_id in self.stems
                or source_id in self._state["returns"])

    def set_send(self, source_id: str, return_id: str, gain_db: float) -> dict:
        """USER co-edit channel (sends/returns shared control): set a source's absolute send
        level (dB, ≤ 0) into an agent-owned return, live — the dry/wet balance the user owns
        alongside the agent. Mirrors SendAction but direct (no proposal lifecycle), like the
        channel-strip set_stem move. A level ≤ −60 dB disconnects the send. The source may be a
        stem, a bus, OR another return (cascading) — a bus/return source arrives as the native track
        name (``bus:<id>`` / ``return:<id>``) or the bare id (both stripped here); a return may only
        send to a LATER return (insertion order) so it can't feed back. The agent CREATES returns;
        user + agent both dial sends."""
        if return_id not in self._state["returns"]:
            raise ValueError(f"set_send: unknown return {return_id!r}")
        returns = self._state["returns"]
        # A prefixed source names its namespace explicitly (it IS the native track name) —
        # resolve it there and only there: ids are unique across namespaces going forward,
        # but a legacy collision (a bus and a return both called 'fx') must not silently
        # write the send onto the wrong holder.
        explicit = ("return" if source_id.startswith("return:")
                    else "bus" if source_id.startswith("bus:") else None)
        src = source_id.split(":", 1)[1] if explicit else source_id
        if explicit == "bus" or (explicit is None and src in self.buses):
            if src not in self.buses:
                raise ValueError(f"set_send: unknown bus {source_id!r}")
            holder = self.buses[src]
        elif explicit is None and src in self.stems:
            holder = self.stems[src]
        elif src in returns:
            # cascading: forward-only (a return may only feed a return created AFTER it) — no loop
            order = list(returns)
            if order.index(src) >= order.index(return_id):
                raise ValueError(f"set_send: a return may only send to a later return "
                                 f"({src!r} -> {return_id!r} would feed back)")
            holder = returns[src]
        else:
            raise ValueError(f"set_send: unknown source {source_id!r}")
        gain_db = float(max(SEND_DB_RANGE[0], min(SEND_DB_RANGE[1], gain_db)))
        sends = holder.setdefault("sends", {})
        if gain_db <= -60.0:
            sends.pop(return_id, None)          # a full cut disconnects the send
        else:
            sends[return_id] = round(gain_db, 1)
        self._persist()
        return {"source": src, "return": return_id,
                "gain_db": sends.get(return_id, -60.0)}

    def add_return(self, kind: str = "blank") -> dict:
        """USER co-edit (sends/returns shared control): create a BLANK FX return (Ableton-style —
        a return is a bus fed by sends, an empty slate you add any device onto via add_stem_device
        on the return:<id> target). Direct (no proposal lifecycle), like set_send. `kind` defaults to
        "blank"; "reverb"/"delay"/"convolution" pre-add that one wet device as a convenience (the agent
        / quick-add path). Returns {return_id, kind}; the user then dials sends into it + adds devices."""
        k = str(kind or "blank").lower()
        if len(self._state["returns"]) >= MAX_RETURNS:   # Ableton parity: 12 send/return tracks max (tracks are unlimited)
            return {"ok": False, "error": f"return limit reached — {MAX_RETURNS} max (Ableton-style)"}
        prefix = ("dly" if k.startswith("d") else "cv" if (k.startswith("c") or k.startswith("i"))
                  else "rv" if k.startswith("r") else "fx")   # blank → "fx"
        # unique id (fx, fx2, … / rv, rv2, …) across EVERY namespace — stems, busses,
        # returns and 'master' share one (a return 'fx' next to a bus 'fx' would make
        # set_send's source resolution ambiguous)
        def _taken(candidate: str) -> bool:
            return (candidate in self._state["returns"]
                    or self.name_taken(candidate, "return") is not None)
        rid, n = prefix, 1
        while _taken(rid):
            n += 1
            rid = f"{prefix}{n}"
        ret = self._state["returns"][rid] = self._new_routing_record("return", rid)
        atype = ("return_delay" if prefix == "dly" else "return_convolution" if prefix == "cv"
                 else "return_reverb" if prefix == "rv" else None)   # None → blank
        if atype is not None:                         # convenience: pre-add the one wet device
            action = (ReturnDelayAction(return_id=rid) if atype == "return_delay"
                      else ReturnConvolutionAction(return_id=rid) if atype == "return_convolution"
                      else ReturnReverbAction(return_id=rid))
            ret["chain"] = [e for e in ret["chain"] if e["type"] != atype]
            ret["chain"].append(action.model_dump(
                exclude={"return_id", "stem_id"}, exclude_none=True))
        self._persist()
        return {"ok": True, "return_id": rid, "kind": atype or "blank"}

    def remove_return(self, return_id: str) -> dict:
        """USER co-edit (sends/returns shared control): delete an FX return — the inverse of add_return.
        Drops it from state AND clears any sends to it (from every stem + bus), so no dangling sends
        remain; the DAW's realizeReturns then drops the orphaned return:<id> track on the next re-resolve."""
        if return_id not in self._state["returns"]:
            raise ValueError(f"remove_return: unknown return {return_id!r}")
        self._bury("return", return_id)   # …so the DAW's undo can ask for it back (restore_routing)
        self._persist()
        return {"ok": True, "return_id": return_id}

    def rename_return(self, old_id: str, new_id: str) -> dict:
        """USER co-edit (2026-09-07): rename an FX return. Like a group bus (``rename_bus``), a
        return's id IS its display name — the DAW paints ``rv`` on the return strip, the pinned
        head card and every send knob's chip — so "rename the return" can only mean moving the id,
        and the id is a reference held all over the session: every sender's ``sends`` dict keys it
        (stems, buses and earlier returns alike), the proposal ledger names it as ``return_id``,
        a device the agent put on the return names it as ``stem_id`` = ``return:<id>``, the
        inverse snapshots are keyed by that same decorated form, and journaled co-edit URLs carry
        it BOTH ways (``stem``/``source`` decorated, ``return`` bare). Every one of those moves
        here, for the reason ``rename_stem`` moves a stem's: a reference left behind is not a stale
        label, it is a send that silently stops feeding the return you just named.

        The return keeps its POSITION in the dict, which is load-bearing rather than cosmetic: the
        cascade rule ("a return may only send to a LATER return") is an index comparison over the
        insertion order, so popping and re-adding a renamed return would move it to the end and
        silently legalise a feedback path that was refused a moment ago. Its gain, pan, pre/post
        tap, chain and sends are untouched — only the key changes. Returns ``{ok, renamed,
        return_id, was}``."""
        old = (old_id or "").strip()
        new = (new_id or "").strip()
        if old.startswith("return:"):     # the DAW may hand either spelling; the id is the bare one
            old = old.split(":", 1)[1]
        if new.startswith("return:"):
            new = new.split(":", 1)[1]
        if not new:
            raise ValueError("rename_return: the new name is empty")
        if old not in self._state["returns"]:
            raise ValueError(f"rename_return: unknown return {old!r}")
        if old == new:
            return {"ok": True, "renamed": False, "return_id": old}
        # One namespace across stems, buses, returns and the master (a chain target is matched by
        # name, and set_send resolves its source in it), and ":" is the prefix separator itself.
        if new in self._state["returns"] or self.name_taken(new, "return"):
            raise ValueError(f"rename_return: '{new}' is taken or reserved "
                             f"(master, an existing stem/bus/return, or a name containing ':')")
        self._state["returns"] = {(new if rid == old else rid): state
                                  for rid, state in self._state["returns"].items()}
        # every sender's send to this return — stems, buses AND earlier returns (cascades)
        for holder in (*self._state["stems"].values(), *self._state["buses"].values(),
                       *self._state["returns"].values()):
            sends = holder.get("sends") or {}
            if old in sends:
                holder["sends"] = {(new if rid == old else rid): db
                                   for rid, db in sends.items()}
        old_target, new_target = _return_target(old), _return_target(new)
        for record in self._state["proposals"]:
            for action in record.get("actions") or []:
                if action.get("return_id") == old:        # send + the wet RETURN forms
                    action["return_id"] = new
                if action.get("stem_id") == old_target:   # a device/gain move ON the return chain
                    action["stem_id"] = new_target
            record["created_returns"] = [new if r == old else r
                                         for r in record.get("created_returns") or []]
            inverse = record.get("inverse") or {}
            if old_target in inverse:                     # inverse snapshots use the decorated id
                inverse[new_target] = inverse.pop(old_target)
            # …and INSIDE each snapshot too. An inverse entry is a deep copy of a holder, so it
            # carries that holder's `sends` dict — still keyed by the OLD return id. Re-keying only
            # the outer key left every snapshot pointing at a return that no longer exists, and
            # undoing such a proposal restored it: `dsp.add_sends` skips an unknown return id
            # WITHOUT error, so the send simply went silent. rename_stem walks `inverse.values()`
            # for exactly this reason (2026-09-08 pre-merge review).
            uids = record.get("target_uids") or {}
            if old_target in uids:
                uids[new_target] = uids.pop(old_target)
            for prior in inverse.values():
                if not isinstance(prior, dict):
                    continue
                prior_sends = prior.get("sends")
                if isinstance(prior_sends, dict) and old in prior_sends:
                    prior["sends"] = {(new if rid == old else rid): db
                                      for rid, db in prior_sends.items()}
        self._rekey_trimmed_charges(old_target, new_target)
        for entry in self._state.get("coedit_log", []):
            url = _rekey_coedit_url(str(entry.get("url", "")), old_target, new_target)
            entry["url"] = _rekey_coedit_url(url, old, new, keys=("return",))
        self._persist()
        return {"ok": True, "renamed": True, "return_id": new, "was": old}

    def set_return_prefader(self, return_id: str, pre_fader: bool) -> dict:
        """USER/agent co-edit (sends/returns shared control): a return's send tap point. Post
        (default) = sends scale with the source's fader; Pre = sends tap the source after its
        devices and before its fader (and pan), so the send keeps its level whatever the fader
        does. A routing flag
        (not a level), so the DAW re-resolves to move the AuxSend insert point."""
        if return_id not in self._state["returns"]:
            raise ValueError(f"set_return_prefader: unknown return {return_id!r}")
        self._state["returns"][return_id]["pre_fader"] = bool(pre_fader)
        self._persist()
        return {"return": return_id, "pre_fader": bool(pre_fader)}

    def set_return_level(self, return_id: str, gain_db: float) -> dict:
        """USER/agent co-edit (sends/returns shared control): a return's output level (the return
        chain's leading gain). Lives in state so the live return-strip fader is oracle-faithful —
        the native return MonitorChain's leading gain mirrors it, exactly as a stem fader ↔
        set_stem. Clamped to [-60, 6] like a channel fader."""
        if return_id not in self._state["returns"]:
            raise ValueError(f"set_return_level: unknown return {return_id!r}")
        gain_db = float(max(RETURN_DB_RANGE[0], min(RETURN_DB_RANGE[1], gain_db)))
        self._state["returns"][return_id]["gain_db"] = round(gain_db, 1)
        self._persist()
        return {"return": return_id, "gain_db": round(gain_db, 1)}

    def remove_bus(self, bus_id: str) -> dict:
        """USER co-edit (routing shared control): dissolve a group bus — re-home every member stem to
        the master, re-home any OTHER bus routed into this one back to the master too (Phase
        2026-07-03 nested bus routing — the inverse of route_bus), clear any sends from the bus, and
        drop it from state. The inverse of routing stems into a bus; mirrors remove_return. The DAW's
        realizeBuses then drops the orphaned bus:<id> track + its strip on the next re-resolve (its
        members were just re-homed, so nothing outputs to it)."""
        if bus_id not in self._state["buses"]:
            raise ValueError(f"remove_bus: unknown bus {bus_id!r}")
        self._bury("bus", bus_id)   # …so the DAW's undo can ask for it back (restore_routing)
        self._persist()
        return {"ok": True, "bus_id": bus_id}

    # ---- A removed group or return can come back (full audit 2026-10-02, readers S4·R25 / C7·R4)
    # A group's or a return's devices, level, sends and members lived only here, and removing
    # one dropped them for good. The DAW realizes the removal as an undoable track delete, so
    # ⌘Z brought the TRACK back; with nothing here to bring back, the app deleted it again -
    # through the undo manager, on top of the history - and every further ⌘Z re-added and
    # re-deleted the same track: nothing older could be reached, and redo was gone. Both ends
    # are undoable now. A removal BURIES the record with everything that pointed at it; the
    # DAW's undo asks for it back by the identity its track carries (`uid`: an id is reused -
    # remove `rv`, add a reverb, and the new return is `rv` again), and the brain follows the
    # engine instead of the engine being made to follow the brain.

    def _mint_uid(self, record_id: str) -> str:
        """A new identity for a stem / bus / return record: never reissued in this session."""
        serial = int(self._state.get("routing_serial", 0)) + 1
        self._state["routing_serial"] = serial
        return f"{record_id}@{serial}"

    def _new_routing_record(self, kind: str, record_id: str) -> dict:
        """A fresh bus / return record, with the identity that outlives its id."""
        record = json.loads(json.dumps(BUS_DEFAULTS if kind == "bus" else RETURN_DEFAULTS))
        record["uid"] = self._mint_uid(record_id)
        # A NEW entity under an id an earlier one held: what the old one's trimmed proposals
        # spent is not this one's debt (D8·R6). The live proposals are told apart by identity.
        self._retire_trimmed_charges(record_id if kind == "bus" else _return_target(record_id))
        return record

    #: What the retention trim folded out of the records, each keyed by TARGET id (propose).
    _TRIM_CARRIES = ("trimmed_gain_totals", "trimmed_ranged_gain", "trimmed_eq_curves_db",
                     "trimmed_singleton_charges", "trimmed_eq_singletons")

    def _retire_trimmed_charges(self, target: str) -> None:
        for key in self._TRIM_CARRIES:
            (self._state.get(key) or {}).pop(target, None)

    def _rekey_trimmed_charges(self, old_target: str, new_target: str) -> None:
        """A renamed track, group or return is the same entity: what its trimmed records spent
        moves with it (reader D8·R4). Left under the old id, the renamed track's session wallet
        and composed-EQ budget started from zero - +10 dB spent on "lead", renamed "Lead Vox",
        and another +12 was accepted."""
        for key in self._TRIM_CARRIES:
            carry = self._state.get(key) or {}
            if old_target in carry:
                carry[new_target] = carry.pop(old_target)

    #== An undo, a replay and a ledger charge are addressed to an ENTITY, not an id (D8·R6) ==
    # A stem record survives its track's deletion and a new track may take its name
    # (`reset_stem`); a dissolved group's id and a removed return's id are minted again by the
    # next ⌘G or "add return". The proposal ledger, its inverse snapshots and the co-edit journal
    # are keyed by id - so undoing the agent's EQ on a deleted "pads" brought its fader, its
    # saturator and its EXCLUDED flag onto the new "pads"; undoing a glue on a dissolved `bus1`
    # gave the vocal group that took the id the drum group's sends; and that new group started
    # with the old one's gain wallet spent. Every record has an identity (`uid`); a proposal
    # keeps the identities it acted on (`target_uids`), and whatever reads the ledger asks
    # whether the id still names that entity.

    def _entity_uid(self, target: str) -> Optional[str]:
        """The identity of whatever `target` names NOW (stems first, as every dispatch does)."""
        if target in self.stems:
            return self.stems[target].get("uid")
        if target in self.buses:
            return self.buses[target].get("uid")
        if target.startswith("return:"):
            record = self.returns.get(target.split(":", 1)[1])
            return record.get("uid") if record else None
        return None

    def _same_entity(self, record: dict, target: str) -> bool:
        """False when the proposal acted on ANOTHER entity that held this id then. True for the
        master, for a record from before identities, and for a target that is gone (an undo
        re-creates it from its snapshot, as it always has)."""
        then = (record.get("target_uids") or {}).get(target)
        if not then:
            return True
        now = self._entity_uid(target)
        return now is None or now == then

    def _bury(self, kind: str, record_id: str) -> None:
        """Take a bus / return out of the state, keeping what it was and what pointed at it.
        Does not persist. The one way a routing record leaves: `remove_bus`, `remove_return`,
        the empty-bus tidy and `restore_routing`'s swap all come through here."""
        table = self._state["buses" if kind == "bus" else "returns"]
        if record_id not in table:
            return
        tomb: dict = {"kind": kind, "id": record_id, "uid": str(table[record_id].get("uid") or ""),
                      "index": list(table).index(record_id)}
        if kind == "bus":
            members = [sid for sid, stem in self._state["stems"].items()
                       if stem.get("bus") == record_id]
            outputs = [bid for bid, bus in self._state["buses"].items()
                       if bid != record_id and bus.get("output") == record_id]
            for sid in members:
                self._state["stems"][sid]["bus"] = None       # re-homed to the master
            for bid in outputs:
                self._state["buses"][bid]["output"] = None
            tomb.update(members=members, outputs=outputs)
        else:
            sends: dict[str, float] = {}
            for prefix, holders in (("stem", self._state["stems"]), ("bus", self._state["buses"]),
                                    ("return", self._state["returns"])):
                for hid, holder in holders.items():
                    level = (holder.get("sends") or {}).pop(record_id, None)   # no dangling send
                    if level is not None:
                        sends[f"{prefix}:{hid}"] = level
            tomb["sends"] = sends
        tomb["entry"] = table.pop(record_id)
        graves = self._state.setdefault("routing_tombstones", [])
        graves.append(tomb)
        del graves[:-ROUTING_TOMBSTONES_MAX]

    def remove_routing(self, kind: str, record_id: str, uid: str = "") -> dict:
        """The DAW's undo took a group's / return's TRACK away (⌘Z of its creation, ⇧⌘Z of its
        removal): the brain follows. Unlike remove_bus / remove_return this is a statement of
        what the engine now holds, so it is idempotent - one that is already gone, or that is a
        LATER record under the same id (`uid` says which the DAW meant), is left alone."""
        if kind not in ("bus", "return"):
            raise ValueError(f"remove_routing: unknown kind {kind!r}")
        table = self._state["buses" if kind == "bus" else "returns"]
        record = table.get(record_id)
        if record is None or (uid and str(record.get("uid") or "") != uid):
            return {"ok": True, "removed": False, "kind": kind, "id": record_id}
        self._bury(kind, record_id)
        self._persist()
        return {"ok": True, "removed": True, "kind": kind, "id": record_id}

    def restore_routing(self, kind: str, record_id: str, uid: str = "") -> dict:
        """The DAW's undo brought a group's / return's TRACK back (⌘Z of its removal, ⇧⌘Z of
        its creation): the record comes back with it - devices, level, position, the sends into
        a return, a group's members and the buses that fed it. A reference that has since been
        pointed somewhere else stays where it is (the DAW tells the engine's routing separately).

        `uid` names WHICH record: the one its track was stamped with. A live record of the same
        id and another uid is swapped out only when the exact tombstone is here to replace it -
        never destroyed for a guess. With no tombstone at all (evicted, or removed before they
        were kept) the record returns at its defaults, so the track the DAW holds is real."""
        if kind not in ("bus", "return"):
            raise ValueError(f"restore_routing: unknown kind {kind!r}")
        key = "buses" if kind == "bus" else "returns"
        table = self._state[key]
        graves = self._state.setdefault("routing_tombstones", [])
        live = table.get(record_id)
        if live is not None and (not uid or str(live.get("uid") or "") == uid):
            return {"ok": True, "restored": False, "kind": kind, "id": record_id}   # it is there
        at = next((i for i in range(len(graves) - 1, -1, -1)
                   if graves[i].get("kind") == kind and graves[i].get("id") == record_id
                   and (not uid or graves[i].get("uid") == uid)), None)
        if live is not None and at is None:
            return {"ok": True, "restored": False, "kind": kind, "id": record_id,
                    "kept": str(live.get("uid") or "")}
        others = (self._state["returns"] if kind == "bus" else self._state["buses"])
        if record_id == MASTER or record_id in self._state["stems"] or record_id in others:
            raise ValueError(f"restore {kind}: '{record_id}' is another track's name now")
        tomb = graves.pop(at) if at is not None else None
        if live is not None:
            self._bury(kind, record_id)                       # the later one steps aside, kept
        if kind == "return" and len(table) >= MAX_RETURNS:
            if tomb is not None:
                graves.append(tomb)
            raise ValueError(f"restore return: the return limit ({MAX_RETURNS}) is reached")
        if tomb is None:
            entry = self._new_routing_record(kind, record_id)
            if uid:
                entry["uid"] = uid
            tomb = {"index": len(table)}
        else:
            entry = tomb["entry"]
        items = list(table.items())
        items.insert(max(0, min(int(tomb.get("index", len(items))), len(items))), (record_id, entry))
        table = self._state[key] = dict(items)
        returns = self._state["returns"]
        entry["sends"] = {rid: db for rid, db in (entry.get("sends") or {}).items()
                          if rid in returns and rid != record_id}
        if kind == "bus":
            if entry.get("output") is not None and (entry["output"] not in table
                                                    or entry["output"] == record_id):
                entry["output"] = None        # the group it fed is gone: it plays to the master
            for sid in tomb.get("members") or []:
                stem = self._state["stems"].get(sid)
                if stem is not None and stem.get("bus") is None:
                    stem["bus"] = record_id
            for bid in tomb.get("outputs") or []:
                other = table.get(bid)
                if other is not None and bid != record_id and other.get("output") is None:
                    other["output"] = record_id
        else:
            for holder_key, level in (tomb.get("sends") or {}).items():
                prefix, _, hid = holder_key.partition(":")
                holder = self._state[{"stem": "stems", "bus": "buses",
                                      "return": "returns"}[prefix]].get(hid)
                if holder is not None and hid != record_id \
                        and record_id not in (holder.get("sends") or {}):
                    holder.setdefault("sends", {})[record_id] = level
        self._persist()
        return {"ok": True, "restored": True, "kind": kind, "id": record_id,
                "adopted": "entry" not in tomb, "uid": str(entry.get("uid") or "")}

    def _reconcile_routing_refs(self) -> None:
        """No stem routes to a bus that is not there and nothing sends to a return that is not
        there (reader D8·R1). An undo writes back whole prior records - a stem's `bus` and
        `sends`, a bus's `output` - without looking at the structure as it is NOW, and deletes
        the buses / returns its proposal created without clearing what the user has since
        pointed at them. The render quietly fell back, the stem left its group with no message,
        and the dangling id was not inert: the next ⌘G minted `bus1` again and the stale member
        was in it; the next "add reverb" minted `rv` and the stale send fed it at its old level."""
        buses, returns = self._state["buses"], self._state["returns"]
        for stem in self._state["stems"].values():
            if stem.get("bus") is not None and stem.get("bus") not in buses:
                stem["bus"] = None
        for bid, bus in buses.items():
            if bus.get("output") is not None and (bus.get("output") not in buses
                                                  or bus.get("output") == bid):
                bus["output"] = None
        for holders in (self._state["stems"], buses, returns):
            for holder in holders.values():
                sends = holder.get("sends")
                if sends and any(rid not in returns for rid in sends):
                    holder["sends"] = {rid: db for rid, db in sends.items() if rid in returns}
        # ...and a return sends only to a LATER return (reader D8·R5). Neither engine plays a
        # cascade that points backward - the offline render runs the returns in order, the DAW
        # refuses to build the send - so one left in the state was a send the agent's picture
        # listed and nothing played, and its presence made the reverse send look like a loop.
        order = {rid: i for i, rid in enumerate(returns)}
        for rid, ret in returns.items():
            sends = ret.get("sends")
            if sends and any(order[to] <= order[rid] for to in sends):
                ret["sends"] = {to: db for to, db in sends.items() if order[to] > order[rid]}

    def rename_bus(self, old_id: str, new_id: str) -> dict:
        """USER co-edit (2026-09-06): rename a group bus. The bus id IS its display name — the DAW
        paints ``bus1`` on the strip, the head card, the rail and both routing menus — so "rename the
        bus" can only mean moving the id, and the id is a reference: members point at it
        (``stem["bus"]``), a nested bus points at it (``bus["output"]``), the proposal ledger's group
        and route actions name it, its inverse snapshots are keyed by it, and journaled co-edit URLs
        carry it as ``bus:<id>``. Every one of those moves here, for the same reason ``rename_stem``
        moves a stem's: a reference left behind is not a stale label, it is a member that silently
        stops belonging to the group you just named. The bus keeps its gain, pan, chain, sends and
        pending list — only the key changes. Returns ``{ok, bus_id, was}``."""
        new = (new_id or "").strip()
        old = (old_id or "").strip()
        if old.startswith("bus:"):
            old = old.split(":", 1)[1]
        if new.startswith("bus:"):        # the DAW may hand either spelling; the id is the bare one
            new = new.split(":", 1)[1]
        if not new:
            raise ValueError("rename_bus: the new name is empty")
        if old not in self._state["buses"]:
            raise ValueError(f"rename_bus: unknown bus {old!r}")
        if old == new:
            return {"ok": True, "renamed": False, "bus_id": old}
        # The bus id shares ONE namespace with stems, returns and the master (a chain target is
        # matched by name), and ":" is the prefix separator itself — "bus:a:b" has no meaning.
        if new in self._state["buses"] or self.name_taken(new, "bus"):
            raise ValueError(f"rename_bus: '{new}' is taken or reserved "
                             f"(master, an existing stem/bus/return, or a name containing ':')")
        # Keep the bus's POSITION in the dict: buses are rendered and listed in insertion order, so
        # popping and re-adding would send a renamed group to the end of the mixer.
        self._state["buses"] = {(new if bid == old else bid): state
                                for bid, state in self._state["buses"].items()}
        aimed_old, aimed_new = f"track:bus:{old}", f"track:bus:{new}"   # an Audio To at the group's track
        for stem in self.stems.values():
            if stem.get("bus") == old:
                stem["bus"] = new
            if stem.get("output") == aimed_old:
                stem["output"] = aimed_new
        for bus in self._state["buses"].values():
            if bus.get("output") == old:
                bus["output"] = new
        for record in self._state["proposals"]:
            for action in record.get("actions") or []:
                # BOTH fields name a bus, and both have to move. `bus_id` is only on a group /
                # route action; a DEVICE or gain move on a bus targets it through `stem_id` (a
                # chain target is matched by name — see the `action.stem_id in self.buses` arm of
                # validate). Leaving stem_id behind is not a stale label: `_cumulative_gain` keys
                # the agent's per-pass and session gain budget by it, so a rename would hand the
                # agent a FRESH +-6 dB on the bus the user just named, and the spend already
                # applied would sit under an id no bus has. rename_stem moves stem_id for the
                # same reason; this one used to move only bus_id.
                if action.get("stem_id") == old:
                    action["stem_id"] = new
                if action.get("bus_id") == old:      # group + route actions name the bus
                    action["bus_id"] = new
            record["created_buses"] = [new if b == old else b
                                       for b in record.get("created_buses") or []]
            inverse = record.get("inverse") or {}
            if old in inverse:                        # inverse snapshots are keyed by the bare id
                inverse[new] = inverse.pop(old)
            uids = record.get("target_uids") or {}
            if old in uids:                           # ...and so are the identities it acted on
                uids[new] = uids.pop(old)
            # …and INSIDE each snapshot (reader D8·R1): a member stem's snapshot says
            # `"bus": <old id>` and a nested bus's says `"output": <old id>`. Left behind, an
            # undo of that proposal wrote the old id back and the stem left the group the user
            # had only renamed. rename_return does this walk for `sends`.
            for prior in inverse.values():
                if not isinstance(prior, dict):
                    continue
                if prior.get("bus") == old:
                    prior["bus"] = new
                if prior.get("output") == old:
                    prior["output"] = new
                elif prior.get("output") == aimed_old:
                    prior["output"] = aimed_new
        self._rekey_trimmed_charges(old, new)     # a bus's carry is keyed by its bare id
        for entry in self._state.get("coedit_log", []):
            entry["url"] = _rekey_coedit_url(str(entry.get("url", "")),
                                             f"bus:{old}", f"bus:{new}")
            entry["url"] = _rekey_coedit_url(entry["url"], aimed_old, aimed_new, keys=("output",))
        self._persist()
        return {"ok": True, "renamed": True, "bus_id": new, "was": old}

    def route_bus(self, bus_id: str, target: str | None) -> dict:
        """USER co-edit (routing shared control, 2026-07-03): route a group bus's OWN output — a
        SEPARATE surface from route_stem (which routes a STEM into a bus). `target` = another bus id
        (nests bus_id's processed sum into it, before the target's own chain) or the master
        (target None/""/"master", the default when absent). Cycle-safe: refuses a target whose own
        output chain already reaches back to bus_id (would loop the graph). Direct, like route_stem —
        stored as buses[bus_id]['output']; absent/None means master. The DAW's realizeBuses wires this
        as ordinary Tracktion track-output routing (bus_id's track -> target's track), and
        dsp.render_master sums nested busses in a topological (children-before-parents) order so the
        offline render matches. Returns {bus, output} (output None = master)."""
        if bus_id not in self._state["buses"]:
            raise ValueError(f"route_bus: unknown bus {bus_id!r}")
        tgt = (target or "").strip()
        tgt = tgt[len("bus:"):] if tgt.startswith("bus:") else tgt
        if tgt in ("", "master"):
            tgt = None
        else:
            if tgt not in self._state["buses"]:
                raise ValueError(f"route_bus: unknown target bus {tgt!r}")
            if tgt == bus_id:
                raise ValueError(f"route_bus: a bus cannot route to itself ({bus_id!r})")
            cur = tgt
            while cur is not None:
                if cur == bus_id:
                    raise ValueError(
                        f"route_bus: {bus_id!r} -> {target!r} would create a routing cycle")
                cur = self._state["buses"][cur].get("output")
        self._state["buses"][bus_id]["output"] = tgt
        self._persist()
        return {"bus": bus_id, "output": tgt}

    def _drop_empty_buses(self) -> None:
        """Tidy: drop any bus with no active reason to exist — no stem member AND nothing routing
        its output INTO it (a bus with a downstream child bus, even if it has no direct stem members
        of its own, is a legitimate submix collector and must survive — Phase 2026-07-03 nested
        routing). Called after route_stem/group_stems reshuffle membership so realizeBuses never
        renders an orphan bus track."""
        # To a fixpoint (reader D8·R7): a parent whose only reason to exist was the child this
        # pass drops is unused only once the child is gone.
        while (unused := self._unused_buses()):
            for bid in unused:
                self._bury("bus", bid)   # kept: ⌘Z of the route that emptied it brings its track back

    def _unused_buses(self) -> list[str]:
        """Busses with no active reason to exist — no stem member, no PENDING member (an
        empty DAW track grouped before its audio landed), and no child bus routing its
        output into them. Shared by _drop_empty_buses and the proposal apply() tidy
        (which must snapshot before dropping)."""
        # Membership is `stem_bus`, never the raw field (pre-merge review 2026-10-04): a track
        # whose Audio To names the group's own track (`track:bus:<id>`) has `bus` None and is
        # a member all the same - the render sums it there. Read raw, the group was buried the
        # moment its last ⌘G member left, with that track still playing into it.
        in_use = {self.stem_bus(sid) for sid in self._state["stems"]}
        in_use |= {b.get("output") for b in self._state["buses"].values() if b.get("output")}
        return [b for b, bus in self._state["buses"].items()
                if b not in in_use and not bus.get("pending")]

    def name_taken(self, name: str, own: str) -> Optional[str]:
        """Why ``name`` cannot name a ``own`` ("stem" | "bus" | "return") - in words - or None.

        Stems, groups, returns and the master share ONE namespace: a chain target, a send's
        source and a fader's co-edit are all matched by bare name, groups first. Every door that
        gives something a name asks here (readers D2·R14 / D8·R3): `route_stem` did not, so the
        agent's `route.stem {bus: "drums"}` - naming the drums TRACK - minted a group called
        `drums` beside the stem, and from then on the user's fader on the track was written to
        the group. The rule: not the master (in any case), no ``:`` (the routing lanes' mark -
        `reserved_stem_name`), and not a name one of the OTHER two tables holds. The caller's own
        table is the caller's to ask ("is there such a group already" is a different question
        for a mint, a rename and a route), and a name already in it is not held to the reserved
        rule - a record from before the rule stays reachable."""
        name = (name or "").strip()
        tables = {"stem": self._state["stems"], "bus": self._state["buses"],
                  "return": self._state["returns"]}
        if name == MASTER or (name not in tables[own] and reserved_stem_name(name)):
            return "reserved (the master's name, or one with ':' in it)"
        for kind, word in (("stem", "a track"), ("bus", "a group"), ("return", "a return")):
            if kind != own and name in tables[kind]:
                return f"{word}'s name"
        return None

    def _mint_bus_id(self) -> str:
        """Fresh user-bus id: bus1, bus2, … (2026-07-19 naming: the ⌘G entity is a BUS
        everywhere the user reads it — the id shows in heads/menus/chips, so it must say
        bus, not group; old sessions keep their groupN ids untouched). Skips ids taken by
        ANY namespace — buses, stems, returns — since a stem or return of the same name
        would collide with the bus's native track / prefixed-source resolution."""
        n = 1
        while f"bus{n}" in self._state["buses"] or self.name_taken(f"bus{n}", "bus"):
            n += 1
        return f"bus{n}"

    def _unpend(self, name: str) -> None:
        """Remove a track name from every bus's pending list (it was regrouped, routed
        elsewhere, or just registered as a real stem)."""
        for bus in self._state["buses"].values():
            pend = bus.get("pending")
            if pend and name in pend:
                pend.remove(name)

    def route_stem(self, stem_id: str, bus_id: str | None, *, keep_emptied: bool = False) -> dict:
        """USER co-edit (routing shared control): route a stem to a group bus (created if absent) or
        to the master (bus_id None/""/"master"). Direct, like set_send — the user owns routing
        alongside the agent (the offline 'agent owns routing' rule is reversed in the DAW). bus_id
        "new" mints a fresh bus id (bus1, bus2, …). Empty busses are dropped. An UNREGISTERED
        stem_id (an empty DAW track — no audio landed yet, so no brain stem) is routed by NAME as a
        pending member (see BUS_DEFAULTS); reserved names (master / bus:* / return:*) still refuse.
        Returns {stem, bus}.

        `keep_emptied` is the DAW's undo speaking (the `route_stem` co-edit, user QA 2026-10-08,
        D37's chain): after an undo or a redo the engine is the truth for the routing
        STRUCTURE, and a group whose track still stands there is a group - with nothing in it,
        for the moment. Its track going is said by its own `remove_routing`. Dropping it here
        because a report of routes left it empty is what turned one wrong route after a Cmd-Z
        into a destroyed group: the brain buried it, the re-read removed its track, and the
        undo history was cleared so nothing could bring it back. It is tidied by the next route
        a person or the engineer makes."""
        name = (stem_id or "").strip()
        # An EXCLUDED stem is a soft-deleted track — a live empty track may reuse its name,
        # so route the NAME as a pending member, never the dead record (mirrors group_stems).
        known = (name in self._state["stems"]
                 and not self._state["stems"][name].get("excluded"))
        if not known and (not name or name == MASTER
                          or name.startswith(("bus:", "return:"))):
            raise ValueError(f"route_stem: unknown stem {stem_id!r}")
        bus = (bus_id or "").strip()
        if bus in ("", "master"):
            bus = None
        elif bus == "new":
            bus = self._mint_bus_id()
        elif bus not in self._state["buses"] and bus.lower() == MASTER:
            bus = None                 # "Master" is the master: nothing else can carry the name
        if bus is not None and bus not in self._state["buses"]:
            # A group is made under a FREE name only (readers D2·R14 / D8·R3 - see name_taken).
            why = self.name_taken(bus, "bus")
            if why:
                raise ValueError(f"route_stem: '{bus}' is {why}, not a group - route into an "
                                 "existing group, or \"new\" for a new one")
            self._state["buses"][bus] = self._new_routing_record("bus", bus)
        self._unpend(name)   # leaves any previous group's pending list (or none)
        if known:
            self._state["stems"][name]["bus"] = bus
            self._clear_output_target(name)
        elif bus is not None:
            # An EMPTY DAW track (no brain stem yet — no audio has landed): remember its
            # NAME as a pending member so the group holds + draws; _discover_stems adopts
            # it as a real member when its audio registers.
            self._state["buses"][bus]["pending"].append(name)
        if not keep_emptied:
            self._drop_empty_buses()
        self._persist()
        return {"stem": name, "bus": bus}

    def _clear_output_target(self, stem_id: str) -> None:
        """A bus (or master) route is the track's output now (D2·R3 — one owner). The stored
        Audio To target goes, and `output_cleared` tells the DAW to drop ITS stored target at
        the next reply: until it did, `realizeBuses` skipped the member, the route answered
        ok, and the track went on playing to its old target. The flag stays until the DAW
        reports the track's output again (user_edit)."""
        stem = self._state["stems"].get(stem_id)
        if stem is not None and stem.pop("output", None):
            stem["output_cleared"] = True

    def stem_output(self, stem_id: str) -> tuple[str, Optional[str]]:
        """Where a stem's signal goes when not to its bus / the master: ``("sends_only", None)``
        (nothing dry — a hardware output reads the same from the master's side),
        ``("track", <stem id>)`` (into that stem's input, in front of its devices),
        ``("bus", <bus id>)`` (a track: target that names a group), or ``("", None)``. A target
        the brain cannot render — a track it holds no stem for, itself, an excluded stem —
        reads as none: the signal is summed where its bus says, the closest true thing."""
        target = self.stems.get(stem_id, {}).get("output") or ""
        if target in ("sends_only", "ext"):
            return "sends_only", None
        if target.startswith("track:"):
            dest = target[6:]
            if dest.startswith("bus:") and dest[4:] in self.buses:
                return "bus", dest[4:]
            if (dest != stem_id and dest in self.stems
                    and not self.stems[dest].get("excluded")):
                return "track", dest
        return "", None

    def stem_feeders(self, stem_id: str) -> list[str]:
        """The active stems whose Audio To is this stem (sorted): their output is its input."""
        return sorted(sid for sid, stem in self.stems.items()
                      if not stem.get("muted") and not stem.get("excluded")
                      and self.stem_output(sid) == ("track", stem_id))

    def group_stems(self, stem_ids: list[str]) -> dict:
        """USER co-edit (⌘G — "Create Bus"): route several stems into ONE new bus in a
        single action — mints a fresh bus (bus1, bus2, …) and routes every given stem into it.
        The one-shot form of route_stem(..., "new"), which would otherwise mint a separate bus per
        call. A name with NO registered stem (an empty DAW track — no audio landed yet) joins as a
        PENDING member: it holds the bus alive, bus_topology lists it so the DAW draws + routes the
        group, and _discover_stems adopts it as a real member when its audio registers (empty tracks
        are groupable, like Live). A regrouped stem may empty its previous bus; empty busses are
        dropped (so realizeBuses leaves no orphan bus track). Returns {bus_id, stems, pending};
        chains/topology are added by the bridge."""
        names = [s.strip() for s in (stem_ids or []) if s and s.strip()
                 and s.strip() != MASTER and not s.strip().startswith(("bus:", "return:"))]
        if not names:
            raise ValueError("create bus: no tracks given")   # shown verbatim in the DAW status
        # An EXCLUDED stem is a soft-deleted track — a live empty track may reuse its
        # name, so group the NAME as pending, never the dead record (whose excluded flag
        # would hide the member from bus_topology: the invisible-group bug again).
        known = [s for s in names if s in self._state["stems"]
                 and not self._state["stems"][s].get("excluded")]
        pending = [s for s in names if s not in known]
        bus = self._mint_bus_id()
        self._state["buses"][bus] = self._new_routing_record("bus", bus)
        for name in names:      # a regrouped EMPTY track leaves its old group's pending list
            self._unpend(name)
        for stem_id in known:
            self._state["stems"][stem_id]["bus"] = bus
            self._clear_output_target(stem_id)       # one owner of an output (D2·R3)
        self._state["buses"][bus]["pending"] = pending
        self._drop_empty_buses()  # a regrouped stem may have emptied its old bus
        self._persist()
        return {"bus_id": bus, "stems": known, "pending": pending}

    def key_cycles(self, stem_id: str, key: str) -> bool:
        """Would `stem_id` listening to `key` close a loop of triggers (`key`'s own trigger
        leading back to `stem_id`, directly or along a chain)? The DAW's graph drops one
        direction of a sidechain cycle - one of the devices gets no key there - while the
        offline render keys a key from its keyed-device-free render, so both keyed here
        (2026-10-17, the duck audit). Since Phase 63 a trigger is any keyed device's."""
        seen, at = {stem_id}, key
        for _ in range(64):
            if at in seen:
                return at == stem_id
            seen.add(at)
            nxt = self.track_trigger(at)
            if not nxt:
                return False
            at = nxt
        return False

    duck_key_cycles = key_cycles   # the Duck's name for it (callers and tests keep it)

    def track_trigger(self, stem_id: str, skip_type: Optional[str] = None) -> str:
        """The trigger track a stem's keyed devices listen to ("" = none) - one per track
        (KEYED_TYPES). `skip_type` leaves one device type out: the one about to be re-keyed."""
        rec = self._state["stems"].get(stem_id)
        for e in (rec or {}).get("chain", []):
            if e.get("type") in KEYED_TYPES and e.get("type") != skip_type and e.get("key_stem_id"):
                return str(e["key_stem_id"])
        return ""

    def trigger_conflict(self, stem_id: str, key: str, device_type: str) -> Optional[str]:
        """The sentence that refuses `device_type` on `stem_id` listening to `key` while another
        keyed device on the track listens to a DIFFERENT trigger - or None. A track has one
        sidechain input in the DAW (Phase 63, James: "one trigger per track")."""
        if not key:
            return None
        other = self.track_trigger(stem_id, skip_type=device_type)
        if not other or other == key:
            return None
        holder = next(e.get("type") for e in self._state["stems"][stem_id]["chain"]
                      if e.get("type") in KEYED_TYPES and e.get("type") != device_type
                      and e.get("key_stem_id") == other)
        label = {"duck": "Duck", "auto_filter": "Auto Filter"}.get(holder, holder)
        return (f"{stem_id!r} already listens to {other!r} (its {label}) - a track has one "
                f"trigger, so this one listens to {other!r} too, or the {label}'s changes first")

    def set_key(self, stem_id: str, device_type: str, key_stem_id: Optional[str],
                key_may_be_track: bool = False) -> dict:
        """USER co-edit (`/api/set_key`, Phase 63): the trigger a keyed device listens to. The
        Duck's door is `set_duck` (it may create the device); the Auto Filter's needs the
        device there and keeps every other setting - "" / none / off = its own input."""
        if device_type == "duck":
            return self.set_duck(stem_id, key_stem_id, key_may_be_track=key_may_be_track)
        if device_type != "auto_filter":
            raise ValueError(f"set_key: {device_type!r} does not listen to a trigger "
                             f"(the ones that do: {', '.join(KEYED_TYPES)})")
        if stem_id not in self._state["stems"]:
            raise ValueError(f"set_key: unknown stem {stem_id!r}")
        entry = next((e for e in self._state["stems"][stem_id]["chain"]
                      if e.get("type") == "auto_filter"), None)
        if entry is None:
            raise ValueError(f"set_key: {stem_id!r} has no Auto Filter")
        key = (key_stem_id or "").strip()
        if key.lower() in ("", "none", "off", "master", "self", "own"):
            key = ""
        if key:
            if key not in self._state["stems"] and not key_may_be_track:
                raise ValueError(f"set_key: unknown trigger track {key!r}")
            if key == stem_id:
                raise ValueError("set_key: a track cannot be its own trigger - leave the trigger "
                                 "empty and the Auto Filter follows its own input")
            if self.key_cycles(stem_id, key):
                raise ValueError(f"set_key: {key!r} already listens to {stem_id!r} (directly or "
                                 f"along a chain) - two tracks cannot trigger each other")
            if (why := self.trigger_conflict(stem_id, key, "auto_filter")) is not None:
                raise ValueError(f"set_key: {why}")
        entry["key_stem_id"] = key
        self._persist()
        return {"stem": stem_id, "type": "auto_filter", "key_stem_id": key or None}

    def set_duck(self, stem_id: str, key_stem_id: str | None,
                 threshold_db: float | None = None,
                 depth_db: float | None = None,
                 key_may_be_track: bool = False) -> dict:
        """USER co-edit (Tier-1 sidechain shared control): key a stem's sidechain DUCK off another
        stem (the kick→bass classic), live — the routing the agent already sets via its `duck` action,
        now in the user's hands too. Direct, like set_send / route_stem (no proposal lifecycle).
        `key_stem_id` None/""/"none"/"off" clears the duck. One duck per stem (the native engine keys a
        single sidechain), so re-keying replaces it. Params default neutral-ish (the user/agent dials
        them via set_master_device_param's "duck" entry). Returns the new duck state."""
        if stem_id not in self._state["stems"]:
            raise ValueError(f"set_duck: unknown stem {stem_id!r}")
        chain = self._state["stems"][stem_id]["chain"]
        # Re-keying keeps the duck's TUNING (2026-09-04 duck audit): this used to drop the whole
        # entry and append a fresh one at −30 / 6 / 5 / 150, so every trigger pick — the user's
        # combo, the agent's duck.set_key, a duck moved to another track — silently reset the
        # threshold / depth / attack / release the user (or SC AUTO) had just dialled. The
        # existing entry's numbers carry over; explicit threshold/depth arguments still win.
        # Keep the duck WHERE IT IS. Filtering it out and appending moved it to the end of the
        # chain, which the DAW never adopts: MonitorChainPlugin::setDuckKeyStemId edits the entry
        # in place and apply_coedits returns no chains, so after one trigger pick the brain held
        # [eq, duck] while the live graph held [duck, eq] — every Python number and the export
        # oracle ducked after the EQ, and the next applyChainsAndReadout pushed the brain's order
        # back and made the user's Duck jump to the last dock slot (2026-09-05).
        slot = next((i for i, e in enumerate(chain) if e.get("type") == "duck"), None)
        prev = chain[slot] if slot is not None else None
        # VALIDATE, THEN TOUCH THE CHAIN (full audit 2026-10-02, reader S4·R6). The duck used to
        # be stripped out first and the key checked after: a refused key (an instrument track
        # the brain holds no stem for, picked in the dock's trigger list) raised with the chain
        # already missing its duck, and the next co-edit of the same flush persisted that.
        key = (key_stem_id or "").strip()
        clearing = key.lower() in ("", "none", "off", "master")
        if not clearing:
            # `key_may_be_track`: the DAW keyed this duck from a track the session holds no stem
            # for - an instrument (MIDI) track, the commonest trigger there is. The DAW checked
            # the track exists; the brain keeps the key so its next chain push does not take the
            # pick back, and renders the duck unkeyed (`render_blind_spots` says so). It used to
            # refuse, keep the OLD key, and revert the user's pick at the next push (2026-10-17).
            if key not in self._state["stems"] and not key_may_be_track:
                raise ValueError(f"set_duck: unknown key stem {key!r}")
            if key == stem_id:
                raise ValueError("set_duck: a stem cannot key its own duck")
            if self.duck_key_cycles(stem_id, key):
                raise ValueError(f"set_duck: {key!r} already ducks under {stem_id!r} (directly or "
                                 f"along a chain) - two tracks cannot trigger each other")
            if (why := self.trigger_conflict(stem_id, key, "duck")) is not None:
                raise ValueError(f"set_duck: {why}")

        def _strip() -> None:
            chain[:] = [e for e in chain if e.get("type") != "duck"]   # one duck per stem — replace

        def _reseat(entry_: dict) -> None:
            # `slot` indexes the ORIGINAL list, but nothing before the FIRST duck is a duck, so
            # it is also the count of surviving entries ahead of it.
            chain.append(entry_) if slot is None else chain.insert(slot, entry_)
        if clearing:
            # Clearing the trigger DISARMS the duck but keeps the device (unkeyed = a render
            # no-op, see _ADD_STEM_DEVICE["duck"]) — the box, its knobs and its tuning stay so
            # the next pick doesn't start over. A duck that never existed is not created.
            if prev is not None:
                prev = dict(prev)
                prev["key_stem_id"] = ""
                _strip()
                _reseat(prev)
            self._persist()
            return {"stem": stem_id, "key_stem_id": None}
        keep = prev or {}
        action = SidechainDuckAction(
            stem_id=stem_id, key_stem_id=key,
            threshold_db=float(threshold_db) if threshold_db is not None
                         else float(keep.get("threshold_db", -30.0)),
            depth_db=float(depth_db) if depth_db is not None
                     else float(keep.get("depth_db", 6.0)),
            attack_ms=float(keep.get("attack_ms", 5.0)),
            release_ms=float(keep.get("release_ms", 150.0)))
        entry = action.model_dump(exclude={"stem_id"})   # stem_id is implied by the chain owner
        if "enabled" in keep:
            entry["enabled"] = keep["enabled"]           # a bypassed duck stays bypassed across a re-key
        _strip()                                         # only now: nothing above can raise any more
        _reseat(entry)
        self._persist()
        return {"stem": stem_id, "key_stem_id": key,
                "threshold_db": action.threshold_db, "depth_db": action.depth_db}

    def set_sat_multiband(self, occurrence: int, on: bool,
                          target: str = "master") -> dict:
        """USER co-edit (Phase 45 native VST): toggle a saturation device between single-band
        and the 3-band (Saturn) form — seed low/mid/high from the flat drive (mirrors the
        native MB button), or clear them back to single-band. `target` = master / stem / bus."""
        matches = [e for e in self._coedit_chain(target) if e["type"] == "saturation"]
        if not 0 <= occurrence < len(matches):
            raise ValueError(f"no saturation #{occurrence} on the {target} chain")
        entry = matches[occurrence]
        if on:
            drv = float(min(24.0, max(0.0, entry.get("drive_db", 3.0))))
            for b in ("low", "mid", "high"):
                entry[b] = {"drive_db": drv, "output_db": 0.0, "mix": 1.0}
            entry.setdefault("crossover_lo_hz", 150.0)
            entry.setdefault("crossover_hi_hz", 2000.0)
            entry.setdefault("mix", 1.0)
        else:
            for k in ("low", "mid", "high", "crossover_lo_hz", "crossover_hi_hz", "mix"):
                entry.pop(k, None)
        self._persist()
        return {"type": "saturation", "occurrence": occurrence, "multiband": on}

    def set_master_device_enabled(self, device_type: str, occurrence: int,
                                  enabled: bool, target: str = "master") -> dict:
        """USER co-edit channel (Phase 40, native VST): bypass / re-enable a device
        (the tile on/off switch). The flag rides on the chain entry; both the render
        and the live monitor skip a disabled device, so the export stays honest, and
        a re-enabled device renders bit-identically (params untouched). `target` =
        master / a stem / a bus (per-track insert)."""
        matches = [e for e in self._coedit_chain(target)
                   if e["type"] == device_type]
        if not 0 <= occurrence < len(matches):
            raise ValueError(f"no {device_type} #{occurrence} on the {target} chain")
        matches[occurrence]["enabled"] = bool(enabled)
        self._persist()
        return {"type": device_type, "occurrence": occurrence,
                "enabled": bool(enabled)}

    # user-addable master devices -> a default entry (Phase 40 native VST co-edit).
    # Defaults come from the Action models so the dict shape always matches what the
    # renderer expects; start neutral (the user/agent dials them in).
    _ADD_MASTER_DEVICE = {
        "eq":         lambda: EqAction(stem_id=MASTER, band=EqBand(
                          freq_hz=1000.0, gain_db=0.0, q=1.0, shape="peak")),
        "comp":       lambda: CompAction(stem_id=MASTER, threshold_db=-18.0,
                          ratio=2.0, attack_ms=10.0, release_ms=120.0),
        # target_lufs_i matches defaultDeviceEntry("limiter") (DeviceSpecs.h). It used to be
        # left unset here, so the in-process monitor played a -14 LUFS limiter the instant it
        # was added and then dropped the makeup on the next chain push — the Loud knob fell
        # -14 -> -24 (the dock's missing-param fallback is the range MINIMUM) with no user
        # action, while the limiter silently stopped targeting anything.
        "limiter":    lambda: LimiterAction(stem_id=MASTER, ceiling_dbtp=-1.0,
                                            target_lufs_i=-14.0),
        "dynamic_eq": lambda: DynamicEqAction(stem_id=MASTER, freq_hz=3000.0,
                          gain_db=-3.0, q=2.0, threshold_db=-24.0, ratio=2.0),
        # Phase 40: master-bus coloration + dynamics the user can add by hand.
        "saturation": lambda: SaturationAction(stem_id=MASTER, drive_db=3.0),
        "width":      lambda: WidthAction(stem_id=MASTER, width=1.0),
        # Device freedom: Ableton-style Utility (gain trim + width + mono + phase), freely placeable.
        "utility":    lambda: UtilityAction(stem_id=MASTER, gain_db=0.0, width=1.0),
        # Phase 41: pre-limiter clipper (the loudness chain) — shave peaks before limiting.
        "clip":       lambda: ClipAction(stem_id=MASTER, threshold_db=-3.0, hardness=0.5),
        # Phase 41: one-knob tilt EQ — broad tonal balance (master/stem/bus).
        "tilt":       lambda: TiltAction(stem_id=MASTER, tilt_db=0.0, pivot_hz=1000.0),
        "mb_comp":    lambda: MultibandCompAction(stem_id=MASTER,
                          low=MbBand(threshold_db=-24.0, ratio=2.0),
                          mid=MbBand(threshold_db=-24.0, ratio=2.0),
                          high=MbBand(threshold_db=-24.0, ratio=2.0)),
        # Phase 43 device freedom: the color/dynamics devices on the master too (real
        # mastering moves) — the native MonitorChain already processes them per-type; the
        # offline _apply_master_chain + _MASTER_KINDS render them in the chain's list order.
        "gate":       lambda: GateAction(stem_id=MASTER, threshold_db=-40.0, ratio=4.0,
                          attack_ms=1.0, release_ms=100.0),
        "deess":      lambda: DeessAction(stem_id=MASTER, threshold_db=-20.0, freq_hz=6000.0),
        "transient":  lambda: TransientAction(stem_id=MASTER, attack_db=2.0, sustain_db=0.0),
        "band_width": lambda: BandWidthAction(stem_id=MASTER, band="presence", width=1.0),
        # 2026-07-05: 3-band stereo imager (the band_width upgrade) — mastering imaging.
        "imager":     lambda: ImagerAction(stem_id=MASTER),
        # 2026-07-07: OTT (3-band up/down comp) + overdrive (drive-pedal distortion).
        # User-add on any target (master included — the user's hands); the AGENT
        # proposes them on stems/busses only (guardrails reject master OTT).
        "ott":        lambda: OttAction(stem_id=MASTER, depth=0.35),
        "overdrive":  lambda: OverdriveAction(stem_id=MASTER, drive_db=12.0),
        # 2026-09-06: chorus / ensemble. Seeded SUBTLE (mix 0.35) — a device that lands
        # fully wet on add is one the user immediately turns down.
        "chorus":     lambda: ChorusAction(stem_id=MASTER, mix=0.35),
        # 2026-09-06: flanger. Seeded with NO feedback and a half blend — the comb is
        # audible immediately, and the resonance is the knob the user reaches for first.
        "flanger":    lambda: FlangerAction(stem_id=MASTER, mix=0.5),
        # Phase 63: a low-pass at 1 kHz swept an octave either side at 0.5 Hz - movement you hear
        # the moment it lands; its envelope off and its trigger its own input.
        "auto_filter": lambda: AutoFilterAction(stem_id=MASTER),
        # 2026-07-06: Spectrogram analyzer — a pure no-op chain entry (the waterfall reads
        # LiveMeters in-process); persisted so it survives reload + agent chain writes.
        "spectrogram": lambda: SpectrogramAction(stem_id=MASTER),
        # match_eq is intentionally NOT user-addable: it's reference-driven (per-band
        # gains come from a mix-vs-reference comparison), a flat default renders no
        # monitor entry — no tile, no effect (a dead button) — and there's no per-band
        # hand-edit UI. The reference flow (set a reference -> the agent applies
        # match_eq) is its right path.
    }

    # Phase 40 (per-track co-edit): stem/bus-only COLOR the user hand-places on a
    # captured track — the processors the master can't host (effective_master_chain
    # excludes them). saturation/width live on the master rail instead.
    _ADD_STEM_DEVICE = {
        "transient":  lambda: TransientAction(stem_id="", attack_db=2.0,
                          sustain_db=0.0),
        "deess":      lambda: DeessAction(stem_id="", threshold_db=-20.0,
                          freq_hz=6000.0),
        "band_width": lambda: BandWidthAction(stem_id="", band="presence",
                          width=1.0),
        # 2026-07-05: 3-band stereo imager (the band_width upgrade).
        "imager":     lambda: ImagerAction(stem_id=""),
        # Phase 41: downward gate/expander — drum gating, breath/noise control.
        "gate":       lambda: GateAction(stem_id="", threshold_db=-40.0, ratio=4.0,
                          attack_ms=1.0, release_ms=100.0),
        # Phase 41: one-knob tilt EQ — broad tonal balance on a stem/bus too.
        "tilt":       lambda: TiltAction(stem_id="", tilt_db=0.0, pivot_hz=1000.0),
        # Device freedom: Ableton-style Utility (gain trim + width + mono + phase), freely placeable.
        "utility":    lambda: UtilityAction(stem_id="", gain_db=0.0, width=1.0),
        # 2026-07-06: Spectrogram analyzer (render no-op) — freely placeable on any track.
        "spectrogram": lambda: SpectrogramAction(stem_id=""),
        # 2026-07-11: user-addable sidechain Duck (the browser Devices tab). Lands UNKEYED
        # (key_stem_id "") — a render no-op until the user picks a key in the dock combo
        # (set_duck) or the box's SC AUTO button asks the agent to key + tune it.
        "duck":       lambda: SidechainDuckAction(stem_id="", key_stem_id="",
                          threshold_db=-30.0, depth_db=6.0),
    }

    # SPACE devices on an FX RETURN: wet-only (mix 1.0) — the return's dry lives on the source
    # tracks. eq/comp/sat/etc. on a return reuse the stem/master constructors above.
    # (`return_id="_"` only satisfies the action's exactly-one-target validator; the id is
    # excluded from the stored entry — the chain it lands in IS the target.)
    _ADD_RETURN_DEVICE = {
        "return_reverb":      lambda: ReturnReverbAction(return_id="_"),
        "return_delay":       lambda: ReturnDelayAction(return_id="_"),
        "return_convolution": lambda: ReturnConvolutionAction(return_id="_"),
    }

    # The SAME devices as ordinary INSERTS on a track / bus / master (2026-07-20 — Live's model:
    # a Reverb is a device, not a routing destination). Only the dry/wet differs: an insert lands
    # at INSERT_WET_MIX so it is immediately musical, where a return stays 100 % wet.
    INSERT_WET_MIX = 0.3
    _ADD_WET_INSERT = {
        "return_reverb":      lambda: ReturnReverbAction(stem_id="_", mix=Session.INSERT_WET_MIX),
        "return_delay":       lambda: ReturnDelayAction(stem_id="_", mix=Session.INSERT_WET_MIX),
        "return_convolution": lambda: ReturnConvolutionAction(stem_id="_", mix=Session.INSERT_WET_MIX),
    }

    def _kinds_for(self, target: str) -> tuple:
        """The canonical device-type set for a co-edit target (drives reorder canonicalisation)."""
        if target in ("", "master"):     return _MASTER_KINDS
        if target.startswith("return:"): return _RETURN_KINDS
        if target in self._state.get("buses", {}): return _BUS_KINDS
        return _STEM_KINDS

    def _device_ctor(self, target: str, device_type: str):
        """The default-action constructor for adding `device_type` to a stem / bus / FX-return
        target (master uses add_master_device — limiter-uniqueness). None if unsupported there.
        The SPACE devices (reverb/delay/IR reverb) are accepted on EVERY target since
        2026-07-20 — wet-only on a return, dry/wet blended as an insert anywhere else."""
        if device_type in ("match_eq", "width"):
            return None                                  # reference-driven / channel-strip scalar
        if device_type in _WET_KINDS:
            # Every placement consumes an in-chain native wet slot, including a real
            # return track. Apply one honest ceiling everywhere rather than allowing a
            # fourth return device that MonitorChain would silently omit.
            if sum(1 for e in self._coedit_chain(target)
                   if e["type"] in _WET_KINDS) >= _WET_INSERT_MAX:
                raise ValueError(
                    f"{target!r} already has {_WET_INSERT_MAX} space devices (reverb/delay/IR "
                    "reverb) — the engine hosts that many per chain; remove one first")
            if target.startswith("return:"):
                return self._ADD_RETURN_DEVICE[device_type]
            return self._ADD_WET_INSERT[device_type]
        # A device the native chain holds a FIXED NUMBER of slots for (P2 batch T, reader
        # E2·R5): the ninth band_width was named in the comment above _NATIVE_SINGLETON_KINDS
        # and enforced nowhere - the brain stored it, the node played eight, and the monitor
        # and the bounce parted from the render. The same honest ceiling as a fourth space
        # device; the dock asks the same numbers before it takes a device off its source
        # (DeviceChainView::capacityRefusal).
        if (slots := _NATIVE_SLOT_KINDS.get(device_type)) is not None:
            if sum(1 for e in self._coedit_chain(target) if e.get("type") == device_type) >= slots:
                raise ValueError(
                    f"{target!r} already has {slots} {device_type} devices - the engine "
                    f"hosts that many per chain; remove one first")
        if device_type in _NATIVE_SINGLETON_KINDS and any(
                e.get("type") == device_type for e in self._coedit_chain(target)):
            raise ValueError(
                f"{target!r} already has a {device_type} — the engine hosts one per chain "
                f"(a second would render offline but not in the live monitor); edit or "
                f"remove the existing one first")
        if device_type == "duck":
            # Stem-only (the native engine keys ONE sidechain per stem; _BUS_KINDS /
            # _RETURN_KINDS would silently drop the entry) — re-key the existing duck via
            # set_duck instead of stacking a second (the singleton rule above covers that).
            if target.startswith("return:") or target in self._state.get("buses", {}):
                return None
        if target.startswith("return:"):
            return (self._ADD_RETURN_DEVICE.get(device_type)
                    or self._ADD_STEM_DEVICE.get(device_type)
                    or self._ADD_MASTER_DEVICE.get(device_type))
        return self._ADD_STEM_DEVICE.get(device_type) or self._ADD_MASTER_DEVICE.get(device_type)

    def add_stem_device(self, target: str, device_type: str) -> dict:
        """USER co-edit channel (Phase 40 / native DAW shared-control): append a device
        to a stem's chain with sane defaults — immediate, no proposal, no guardrails (the
        user's hands). The native DAW (docs/10) is Ableton-style: the user can hand-place
        the FULL device set on ANY track — the stem-only COLOR devices (transient/de-ess/
        band_width) AND the rich processors (eq/comp/saturation/etc.), reusing the master
        constructors, the COLOR devices (transient/de-ess/band_width), AND (Phase 43 device
        freedom) a per-stem `clip` + `limiter` (a peak ceiling via _brickwall — no loudness
        makeup). Render order is the chain list (effective_chain). `match_eq`/`width` stay off a
        stem: match_eq is reference-driven (a flat default renders nothing), and stem width is the
        channel-strip scalar (`stem_state["width"]`), not a chain entry. The planning agent never
        gets this path. `target` may be a stem, a group bus, or an FX return (`return:<id>`) — a
        blank return is a free device chain (full palette incl. reverb/delay/convolution)."""
        if target in ("", "master"):
            return self.add_master_device(device_type)   # master has its own (limiter-unique) path
        chain = self._coedit_chain(target)               # stem / bus / return:<id>; raises if unknown
        ctor = self._device_ctor(target, device_type)
        if ctor is None:
            raise ValueError(f"add_stem_device: unsupported device {device_type!r} on {target!r}")
        if device_type in ("eq", "tilt"):   # the native bank is finite — refuse, never drop (2026-09-01)
            from .guardrails import LIMITS, eq_bands_in
            have = sum(eq_bands_in(e) for e in chain)
            if have + (2 if device_type == "tilt" else 1) > LIMITS["eq_bands_max"]:
                raise ValueError(f"{target!r} already carries {have} EQ bands; the native bank holds "
                                 f"{LIMITS['eq_bands_max']} — remove a band before adding one")
        # Neither target field is ever stored: the CHAIN the entry lands in is the target
        # (a space device carries return_id OR stem_id purely to satisfy its validator).
        entry = ctor().model_dump(exclude={"stem_id", "return_id"},
                                  exclude_none=True)   # no None placeholders
        entry["owner"] = "user"   # provenance: _replace_in_chain reports overwriting it
        chain.append(entry)
        self._persist()
        occ = sum(1 for e in chain if e["type"] == device_type) - 1
        return {"type": device_type, "occurrence": occ, "target": target}

    def remove_stem_device(self, target: str, device_type: str,
                           occurrence: int) -> dict:
        """USER co-edit channel: remove the Nth `device_type` from a stem / bus / FX-return chain."""
        chain = self._coedit_chain(target)               # stem / bus / return:<id>; raises if unknown
        matches = [e for e in chain if e["type"] == device_type]
        if not 0 <= occurrence < len(matches):
            raise ValueError(f"no {device_type} #{occurrence} on {target!r}")
        chain.remove(matches[occurrence])
        self._persist()
        return {"type": device_type, "occurrence": occurrence, "target": target}

    def add_master_device(self, device_type: str, *, safety: bool = False) -> dict:
        """USER co-edit channel (Phase 40, native VST): append a device to the
        master chain with sensible defaults — immediate, no proposal, no guardrails
        (the user's hands). Render order is canonical (effective_master_chain), so
        the entry lands in the right slot regardless of append position. One limiter
        max. The planning agent never gets this path.

        ``safety=True`` (2026-09-04, limiter only): the DAW's default master SAFETY
        limiter — a −1 dBTP ceiling and NOTHING else (no ``target_lufs_i``, so no
        loudness makeup). The palette limiter targets −14 LUFS, which is a loudness
        tool; the safety one is transparent until a peak actually reaches the
        ceiling, which is what "a limiter on the master by default" has to mean."""
        if device_type == "limiter" and safety:
            ctor = lambda: LimiterAction(stem_id=MASTER, ceiling_dbtp=-1.0)   # noqa: E731
        elif device_type in _WET_KINDS:   # 2026-07-20: space devices are master inserts too
            if sum(1 for e in self._state["master_chain"]
                   if e["type"] in _WET_KINDS) >= _WET_INSERT_MAX:
                raise ValueError(
                    f"the master already has {_WET_INSERT_MAX} space devices (reverb/delay/IR "
                    "reverb) — the engine hosts that many per chain; remove one first")
            ctor = self._ADD_WET_INSERT[device_type]
        elif device_type not in self._ADD_MASTER_DEVICE:
            raise ValueError(f"add_master_device: unsupported device {device_type!r}")
        else:
            ctor = self._ADD_MASTER_DEVICE[device_type]
        # ...and the fixed slot counts (_NATIVE_SLOT_KINDS), as on a track: this door had none,
        # and the master is where a Dynamic EQ lives (pre-merge review 2026-10-06).
        if (slots := _NATIVE_SLOT_KINDS.get(device_type)) is not None:
            if sum(1 for e in self._state["master_chain"] if e["type"] == device_type) >= slots:
                raise ValueError(
                    f"the master already has {slots} {device_type} devices - the engine hosts "
                    f"that many per chain; remove one first")
        # One parameter set per type in the native chain — see _NATIVE_SINGLETON_KINDS.
        if device_type in _NATIVE_SINGLETON_KINDS and device_type != "limiter" and any(
                e["type"] == device_type for e in self._state["master_chain"]):
            raise ValueError(
                f"the master already has a {device_type} — the engine hosts one per chain "
                f"(a second would render offline but not in the live monitor); edit or "
                f"remove the existing one first")
        if device_type == "limiter" and any(
                e["type"] == "limiter" for e in self._state["master_chain"]):
            raise ValueError("the master already has a limiter")
        # The EQ-bank budget, which `add_stem_device` has always enforced and this path never
        # did — so the MASTER, the one chain most likely to carry a match_eq plus a stack of
        # eq bands, was the unguarded one (2026-09-08 device audit). The native bank silently
        # DROPS bands past kMaxBands, which is the failure eq_bands_max exists to prevent:
        # they render offline and are absent from the live monitor and the DAW bounce.
        if device_type in ("eq", "tilt", "match_eq"):
            from .guardrails import LIMITS, eq_bands_in
            have = sum(eq_bands_in(e) for e in self._state["master_chain"])
            adding = eq_bands_in({"type": device_type,
                                  **({"band_gains_db": {b: 0.0 for b in BAND_LABELS}}
                                     if device_type == "match_eq" else {})})
            if have + max(1, adding) > LIMITS["eq_bands_max"]:
                raise ValueError(f"the master already carries {have} EQ bands; the native bank "
                                 f"holds {LIMITS['eq_bands_max']} — remove a band before adding one")
        entry = ctor().model_dump(
            exclude={"stem_id", "return_id"}, exclude_none=True)   # don't store None placeholders
        entry["owner"] = "user"   # provenance: _replace_in_chain reports overwriting it
        self._state["master_chain"].append(entry)
        self._persist()
        occ = sum(1 for e in self._state["master_chain"]
                  if e["type"] == device_type) - 1
        return {"type": device_type, "occurrence": occ}

    def remove_master_device(self, device_type: str, occurrence: int) -> dict:
        """USER co-edit channel: remove the Nth `device_type` from the master chain
        (the × on the native device tile). Immediate, no proposal."""
        matches = [e for e in self._state["master_chain"]
                   if e["type"] == device_type]
        if not 0 <= occurrence < len(matches):
            raise ValueError(f"no {device_type} #{occurrence} on the master chain")
        target = matches[occurrence]
        self._state["master_chain"] = [
            e for e in self._state["master_chain"] if e is not target]
        self._persist()
        return {"type": device_type, "occurrence": occurrence, "removed": True}

    def set_master_device_order(self, order: list[str]) -> dict:
        """USER co-edit channel (drag-to-reorder): reorder the master chain LIST to the
        requested device-type sequence (Phase 43 device freedom — the list IS the render
        order). Unlisted types keep their relative slot; the limiter is freely placeable
        (Stage 3 — a posted position after it is honored, so a newly-added device appends
        at the END / rightmost in signal flow). Immediate, no proposal — the user's hands;
        the planning agent never gets this path."""
        self._state["master_chain"] = _dock_order(
            self._state["master_chain"], _MASTER_KINDS, order)
        self._persist()
        return {"master_device_order": [e["type"] for e in self._state["master_chain"]]}

    def set_stem_device_order(self, target: str, order: list[str]) -> dict:
        """USER co-edit channel (drag-to-reorder): reorder a stem's chain LIST to the
        requested device-type sequence so added/reordered devices HOLD their position
        (Phase 43 device freedom — the list IS the render order). Unlisted types keep their
        relative slot. Immediate, no proposal — the user's hands; the agent reaches order
        via its own chain edits. `target` may be a stem, a group bus, or an FX return."""
        chain = self._coedit_chain(target)               # stem / bus / return:<id>; raises if unknown
        chain[:] = _dock_order(chain, self._kinds_for(target), order)   # in-place (chain is a live ref)
        self._persist()
        return {"target": target,
                "stem_device_order": [e["type"] for e in chain]}

    def user_edit(self, stem_id: str, changes: dict[str, Any]) -> dict:
        """The user's direct channel (Phase 9): absolute setters, applied
        immediately, no proposal record, no guardrails — the agent never gets
        this path (CLAUDE.md invariant). Returns the stem's new state.
        `stem_id` may also name the master (gain_db + the Phase 27 global
        automation switch) or a group bus (Phase 23; gain_db / pan / width /
        muted since 2026-09-05 — the group strip's own channel-strip scalars)."""
        if stem_id == MASTER:
            if set(changes) - {"gain_db", "automation_enabled"}:
                raise ValueError(
                    "master accepts gain_db and automation_enabled only")
            if "gain_db" in changes:
                lo, hi = self.USER_EDIT_FIELDS["gain_db"]
                self._state["master_gain_db"] = float(
                    min(hi, max(lo, changes["gain_db"])))
            if "automation_enabled" in changes:
                self._state["automation_enabled"] = bool(
                    changes["automation_enabled"])
            self._persist()
            return {"stem_id": MASTER, "gain_db": round(self.master_gain_db, 1),
                    "automation_enabled": self.automation_enabled}
        if stem_id in self.buses:
            # The bus strip's own channel-strip move (2026-09-05): gain / pan / width / mute,
            # the same four the DAW's ChannelStrip offers on a group and the same clamps a stem
            # takes. Before this a bus took gain_db only, so a pan or [M] on a group was refused
            # and lost. `excluded`/`automation_enabled` stay stem-and-master concepts (a bus is
            # removed via remove_bus; automation is per-stem).
            allowed = {"gain_db", "pan", "width", "width_on", "muted"}
            if ("width" in changes and "width_on" not in changes
                    and not self.buses[stem_id].get("width_on", True)):   # a hand on the width switches it on
                changes = {**changes, "width_on": True}
            if set(changes) - allowed:
                raise ValueError(f"a bus accepts {', '.join(sorted(allowed))} only")
            bus = self.buses[stem_id]
            for field, value in changes.items():
                bounds = self.USER_EDIT_FIELDS[field]
                if bounds is None:
                    bus[field] = bool(value)
                else:
                    lo, hi = bounds
                    bus[field] = float(min(hi, max(lo, value)))
            self._persist()
            return {"stem_id": stem_id, "gain_db": round(bus["gain_db"], 1),
                    "pan": round(bus.get("pan", 0.0), 3),
                    "width": round(bus.get("width", 1.0), 3),
                    "muted": bool(bus.get("muted", False))}
        if stem_id.startswith("return:") and stem_id[len("return:"):] in self._state["returns"]:
            # A RETURN's own strip (full audit 2026-10-02, reader S4·R7). Its head card posts
            # `set_stem?stem=return:<id>&gain_db=…` / `&pan=…`, and both surfaces post `&muted=…`;
            # there was no return branch here, so every one was refused ("no such stem:
            # return:rv") — the level snapped back at the next chain push, and a muted return
            # went on being rendered, measured and mastered by the brain. `pan` and `muted` are
            # new on a return (absent = centred, un-muted: every stored session renders as it
            # did); `gain_db` is set_return_level's field, with its clamp.
            allowed = {"gain_db", "pan", "muted", "delay_ms"}
            if set(changes) - allowed:
                raise ValueError(f"a return accepts {', '.join(sorted(allowed))} only")
            ret = self._state["returns"][stem_id[len("return:"):]]
            if "delay_ms" in changes:   # the delay its node runs (an engine fact, D2·R4)
                ret["delay_ms"] = round(float(min(REALIZED_DELAY_MS_RANGE[1], max(
                    REALIZED_DELAY_MS_RANGE[0], changes["delay_ms"]))), 2)
            if "gain_db" in changes:
                ret["gain_db"] = round(float(min(RETURN_DB_RANGE[1], max(
                    RETURN_DB_RANGE[0], changes["gain_db"]))), 1)
            if "pan" in changes:
                lo, hi = self.USER_EDIT_FIELDS["pan"]
                ret["pan"] = float(min(hi, max(lo, changes["pan"])))
            if "muted" in changes:
                ret["muted"] = bool(changes["muted"])
            self._persist()
            return {"stem_id": stem_id, "gain_db": round(ret["gain_db"], 1),
                    "pan": round(ret.get("pan", 0.0), 3),
                    "muted": bool(ret.get("muted", False))}
        if stem_id not in self._state["stems"] and changes.get("excluded"):
            # Deleting an EMPTY track (no audio landed, so no brain stem): its only brain-side
            # footprint is a possible PENDING group membership (grouped while empty) — clear it,
            # so the group doesn't hold a ghost member (and an emptied bus can drop).
            self._unpend(stem_id)
            self._drop_empty_buses()
            self._persist()
            return {"stem_id": stem_id, "excluded": True, "pending_cleared": True}
        stem = self._require_stem(stem_id)
        # The role tag(s) drive the agent's family targets — a list, not a clamped scalar, so it's
        # handled here rather than via USER_EDIT_FIELDS (the user sets it from the track header menu).
        if "tags" in changes:
            raw = changes.pop("tags")
            stem["tags"] = [str(t) for t in raw] if isinstance(raw, list) else ([str(raw)] if raw else [])
        # The engine's facts (D2·R3 / D2·R4) — see ENGINE_FACT_FIELDS. Not strip moves: they
        # are reported, clamped to the DAW's own bounds, and kept only when not the default.
        changes = dict(changes)
        if "vca_db" in changes:
            db = float(min(VCA_DB_RANGE[1], max(VCA_DB_RANGE[0], changes.pop("vca_db"))))
            stem.pop("vca_db", None) if abs(db) < 0.005 else stem.__setitem__("vca_db", round(db, 2))
        if "delay_ms" in changes:
            ms = float(min(REALIZED_DELAY_MS_RANGE[1], max(REALIZED_DELAY_MS_RANGE[0],
                                                             changes.pop("delay_ms"))))
            stem.pop("delay_ms", None) if ms < 0.005 else stem.__setitem__("delay_ms", round(ms, 2))
        if "frozen" in changes:
            by = str(changes.pop("frozen") or "").strip()[:256]
            # ":live" is the DAW's word for "not frozen": a query value cannot be empty, and no
            # track can be called that (a ':' in a name is reserved - reserved_stem_name).
            stem.pop("frozen", None) if by in ("", FROZEN_LIVE) else stem.__setitem__("frozen", by)
        if any(key in changes for key in ("clip_at_s", "clip_from_s", "clip_for_s")):
            window = normalize_clip_window(changes.pop("clip_at_s", 0.0), changes.pop("clip_from_s", 0.0),
                                           changes.pop("clip_for_s", 0.0))
            stem.pop("clip", None) if window is None else stem.__setitem__("clip", window)
        if "clips" in changes:                          # a track of several clips (see CLIP_WINDOWS_MAX)
            windows = normalize_clip_windows(changes.pop("clips"))
            stem.pop("clips", None) if windows is None else stem.__setitem__("clips", windows)
        if stem.get("clips"):
            stem.pop("clip", None)                      # one fact or the other, never both
        if "output" in changes:
            target = normalize_output_target(changes.pop("output"))
            stem.pop("output_cleared", None)        # the DAW has spoken since the brain's route
            if target:
                # ONE owner of a track's output (D2·R3): a target that is not a bus or the
                # master takes the track out of its group here, as it does in the engine —
                # the Audio To menu's Sends Only / track / hardware picks never told the
                # brain, which went on rendering the track through its old bus.
                stem["output"] = target
                if stem.get("bus") is not None:
                    stem["bus"] = None
                    self._drop_empty_buses()
            else:
                stem.pop("output", None)
        if "width" in changes and "width_on" not in changes and not stem.get("width_on", True):
            changes = {**changes, "width_on": True}   # a hand on a bypassed width switches it on (SW1)
        for field, value in changes.items():
            if field not in self.USER_EDIT_FIELDS:
                raise ValueError(f"not a user-editable field: {field!r} "
                                 f"(editable: {sorted(self.USER_EDIT_FIELDS)})")
            bounds = self.USER_EDIT_FIELDS[field]
            if bounds is None:
                stem[field] = bool(value)
            else:
                lo, hi = bounds
                stem[field] = float(min(hi, max(lo, value)))
        if changes.get("excluded"):
            # This edit deleted the track — the live track of this name is gone, so any
            # pending membership of the name (a track grouped while empty) is stale with it.
            self._unpend(stem_id)
            # ...and it is a DELETE now, whatever took the stem out before (reader D8·R9):
            # `missing` is the mark discovery reads to put a stem back when its file returns,
            # and it shares `excluded` with the user's delete - a track deleted while its file
            # was away rejoined the mix the moment the drive was plugged back in.
            stem.pop("missing", None)
        self._persist()
        return {"stem_id": stem_id,
                **{f: stem.get(f, True if f == "width_on" else None) for f in self.USER_EDIT_FIELDS}}

    # ---------- group busses (Phase 23; routing is the agent's, Phase 28) ----------
    # Busses are created, routed and glued only through proposals now (the
    # group/route actions in _mutate). There is no user routing channel — the
    # user's manual moves are volume/pan/mute/solo. _require_bus backs
    # analyze_bus and the bus-target snapshot in apply/undo.

    def _require_bus(self, bus_id: str) -> dict:
        if bus_id not in self.buses:
            raise KeyError(f"no such bus: {bus_id!r}; buses: {sorted(self.buses)}")
        return self.buses[bus_id]

    def advance_pass(self) -> int:
        """Move to the next pass (orchestrator calls this between passes, Phase 2)."""
        self._state["pass_number"] += 1
        self._persist()
        return self.pass_number

    def _fader_levels(self) -> dict[str, float]:
        """Where every fader stands - stems, buses, the master - for the guardrail that keeps a
        run's gain move inside the fader's range (`guardrails.FADER_DB_RANGE`)."""
        levels = {sid: float(stem.get("gain_db", 0.0)) for sid, stem in self.stems.items()}
        levels.update({bid: float(bus.get("gain_db", 0.0)) for bid, bus in self.buses.items()})
        levels[MASTER] = float(self.master_gain_db)
        return levels

    def _hold_faders_to_range(self) -> None:
        """A level stored past the fader's top opens AT the top (2026-10-05): sessions written
        while the range ran to +12, or by a run's gain move before it had a ceiling. Only the
        top - a level under the floor was never reachable."""
        top = FADER_DB_RANGE[1]
        for holder in list(self._state.get("stems", {}).values()) + list(self._state.get("buses", {}).values()):
            if isinstance(holder.get("gain_db"), (int, float)) and holder["gain_db"] > top:
                holder["gain_db"] = top
        if isinstance(self._state.get("master_gain_db"), (int, float)) and self._state["master_gain_db"] > top:
            self._state["master_gain_db"] = top

    # ---------- proposal lifecycle ----------

    def propose(self, req: ProposalRequest) -> ProposalResult:
        violations = self.validate_action_targets(req.actions)
        violations += validate_proposal(
            req,
            self._cumulative_gain_this_pass(req.actions),
            accepted_proposals_this_pass=self._accepted_proposals_this_pass(),
            stem_tags=self._stem_tags(),
            cumulative_gain_session_db=self._session_gain_baseline(req.actions),
            # ...and what STANDS, the entries this proposal would replace included: what a
            # path is judged against (guardrails._path_gain_violations)
            standing_gain_this_pass_db=self._cumulative_gain(self.pass_number),
            standing_gain_session_db=self._session_gain_baseline([]),
            ranged_gain_this_pass_db=self._ranged_gain(self.pass_number),
            ranged_gain_session_db=self._ranged_gain(None),
            stem_chains=self._chains_for_guardrails(),
            eq_ledger=self._eq_ledger(),
            fader_db=self._fader_levels(),
            signal_parents=self._signal_parents(),
        )
        # Monotonic id (scale/soak audit 2026-08-09): ids used to derive from the list LENGTH,
        # which was fine while the list never shrank — the retention trim below breaks that, so
        # a persisted counter owns uniqueness now. Sessions predating the counter seed it from
        # the length (identical numbering — no trim has happened yet on such a session).
        counter = int(self._state.get("proposal_counter", len(self._state["proposals"]))) + 1
        self._state["proposal_counter"] = counter
        record = {
            "proposal_id": f"p{counter:04d}",
            "status": "rejected" if violations else "proposed",
            "rationale": req.rationale,
            "predicted_impact": [p.model_dump() for p in req.predicted_impact],
            # exclude_none keeps optional fields (gain time_range_s) out of
            # session.json and the UI feed unless actually set
            "actions": [a.model_dump(exclude_none=True) for a in req.actions],
            "measured_impact": None,
            "pass_number": self.pass_number,
            "applied_seq": None,
            "inverse": None,
            "created_buses": [],  # Phase 28: busses this proposal creates (set at apply)
            "created_returns": [],  # Phase 38: returns this proposal creates
        }
        self._state["proposals"].append(record)
        # Retention trim (scale/soak audit 2026-08-09): this list was append-only FOREVER — each
        # record carries `inverse` (a deep copy of every touched stem/bus dict), session.json is
        # rewritten whole on every mutation, and Session() re-parses it several times per chat
        # turn, so a months-old project paid ever-growing cost on every action. Keep the last 400
        # (the agent-side undo horizon is 200); fold the trimmed records' APPLIED gain into a
        # persisted carry so the session gain budget (_cumulative_gain(None), a guardrail) still
        # counts spend the trim can no longer enumerate. Trimmed proposals can't be undone —
        # consistent: their gain is permanently spent, so it stays permanently counted.
        records = self._state["proposals"]
        if len(records) > _MAX_PROPOSAL_RECORDS + 50:
            cut = records[: len(records) - _MAX_PROPOSAL_RECORDS]
            carry = self._state.setdefault("trimmed_gain_totals", {})
            from .guardrails import eq_family_curves_db, level_charge_db
            for old in cut:
                if old.get("status") == "applied":
                    for index, action in enumerate(old.get("actions", [])):
                        # The carries are keyed by id, and an id is reused (D8·R6): a record
                        # that acted on the entity that held this id THEN is not a charge on
                        # the one that holds it now - the live ledger asks the same question
                        # (pre-merge review 2026-10-04: a new `bus1` inherited the dissolved
                        # one's spend the day its records passed the horizon).
                        sid = str(action.get("stem_id") or "")
                        if not self._same_entity(old, sid):
                            continue
                        # EVERY level charge, by the ledger's own rules (reader D8·R4): the
                        # fold used to know `gain` alone, so a trimmed Utility at +6 dB - still
                        # in the chain - read as 0 spent, and +12 dB of fader went on top.
                        charge = self._ledger_charge(old, index, level_charge_db(action))
                        ranged = action.get("time_range_s") is not None
                        if not charge:
                            pass
                        elif ranged:
                            # one-sided for good: never credit (D9·R4). A ranged comp's
                            # make-up is read off the live chain at session scope instead.
                            if action.get("type") != "comp":
                                up, down = self._state.setdefault(
                                    "trimmed_ranged_gain", {}).get(sid, (0.0, 0.0))
                                self._state["trimmed_ranged_gain"][sid] = [
                                    up + max(charge, 0.0), down + min(charge, 0.0)]
                        elif not self._is_singleton_record(action):
                            carry[sid] = carry.get(sid, 0.0) + charge          # append types sum
                        if not ranged and self._is_singleton_record(action) and charge is not None:
                            # A singleton REPLACES: the carry holds the LAST value per (target,
                            # type), and a live record of the same pair replaces it in turn
                            # (_cumulative_gain) - so re-tuning a trimmed Utility costs its delta.
                            self._state.setdefault("trimmed_singleton_charges", {}).setdefault(
                                sid, {})[action["type"]] = charge
                        # …and its static EQ, folded into per-target curves on the composed
                        # rail's grid, so a trim never refills the session EQ budget either. A
                        # singleton's curve is not summed: the carry keeps its LAST entry, and a
                        # live record of the same pair replaces it in turn (_eq_ledger).
                        counted = self._ledger_eq_action(old, index, action)
                        if self._is_eq_singleton(action):
                            self._state.setdefault("trimmed_eq_singletons", {}).setdefault(
                                sid, {})[action["type"]] = counted
                            continue
                        curves = eq_family_curves_db(counted)
                        if curves is not None:
                            eq_carry = self._state.setdefault("trimmed_eq_curves_db", {})
                            prev = eq_carry.get(sid) or {"boost": [0.0] * len(curves[0]),
                                                         "cut": [0.0] * len(curves[1])}
                            eq_carry[sid] = {
                                "boost": [round(x + float(y), 3) for x, y in zip(prev["boost"], curves[0])],
                                "cut": [round(x + float(y), 3) for x, y in zip(prev["cut"], curves[1])]}
            del records[: len(records) - _MAX_PROPOSAL_RECORDS]
        self._persist()
        if violations:
            return ProposalResult(accepted=False, violations=violations)
        return ProposalResult(accepted=True, proposal_id=record["proposal_id"])

    def apply(self, proposal_id: str) -> ApplyResult:
        record = self._require_proposal(proposal_id)
        if record["status"] != "proposed":
            return ApplyResult(ok=False, proposal_id=proposal_id,
                               message=f"proposal is '{record['status']}', not 'proposed'")

        # Re-validate the gain clamp against what is *applied* now: other proposals
        # may have been applied since this one was proposed.
        # `stored`: a record read back is not a request - one written by another build may
        # carry a field this one does not know, and is not refused for it (reader D9·R11).
        req = ProposalRequest.model_validate({
            "rationale": record["rationale"],
            "predicted_impact": record["predicted_impact"],
            "actions": record["actions"],
        }, context={"stored": True})
        violations = validate_proposal(
            req, self._cumulative_gain_this_pass(req.actions),
            # Re-checked at APPLY too (2026-09-01 sweep): apply re-stamps the record to the
            # current pass, so cards proposed in an earlier pass and approved later (Ask
            # mode) used to land six proposals / 48 actions in one pass against the 3 / 24 cap.
            accepted_proposals_this_pass=self._accepted_proposals_this_pass(exclude=proposal_id),
            stem_tags=self._stem_tags(),
            cumulative_gain_session_db=self._session_gain_baseline(req.actions),
            # ...and what STANDS, the entries this proposal would replace included: what a
            # path is judged against (guardrails._path_gain_violations)
            standing_gain_this_pass_db=self._cumulative_gain(self.pass_number),
            standing_gain_session_db=self._session_gain_baseline([]),
            ranged_gain_this_pass_db=self._ranged_gain(self.pass_number),
            ranged_gain_session_db=self._ranged_gain(None),
            stem_chains=self._chains_for_guardrails(),
            eq_ledger=self._eq_ledger(),
            fader_db=self._fader_levels(),
            signal_parents=self._signal_parents())
        if violations:
            return ApplyResult(ok=False, proposal_id=proposal_id,
                               message="; ".join(v.message for v in violations))

        # Pre-flight (atomicity): a stem this proposal targets may have been removed
        # between propose and apply. The snapshot loop below indexes self.stems[target]
        # directly, so a missing stem would raise KeyError AFTER earlier actions already
        # mutated state — a partial apply with no undo record. Validate up front and fail
        # cleanly before touching anything. master / return: targets never index self.stems
        # (guarded branches below), and a bus this proposal *itself creates* (a group action)
        # is a valid target even though it doesn't exist yet — so allow those too.
        will_exist_buses = set(self.buses)
        for action in req.actions:
            if action.type == "group":
                will_exist_buses.add(action.bus_id)
        for action in req.actions:
            for target in self._action_snapshot_targets(action):
                if (target == MASTER or target in will_exist_buses
                        or target.startswith("return:") or target in self.stems):
                    continue
                return ApplyResult(
                    ok=False, proposal_id=proposal_id,
                    message=f"cannot apply: target '{target}' no longer exists")
        # The snapshot-target loop above only covers each action's SOURCE. A
        # route's destination bus or a send's destination return may also have
        # been removed between propose and apply (shared control: remove_bus /
        # remove_return are direct user edits) — _mutate would write the
        # dangling reference and the render silently drops it. Re-run the
        # structural target validation against CURRENT state so a stale
        # destination fails cleanly instead.
        stale = self.validate_action_targets(req.actions)
        if stale:
            return ApplyResult(
                ok=False, proposal_id=proposal_id,
                message=f"cannot apply: {stale[0].message}"
                        + (f" (+{len(stale) - 1} more)" if len(stale) > 1 else ""))

        # Snapshot the whole prior state of every touched target, then mutate.
        # Undo restores the snapshot (never re-subtracts/removes), so a render
        # after undo is bit-identical.
        inverse: dict[str, Any] = {}
        # Phase 28: busses this proposal *creates* (vs. adds members to). Undo
        # deletes them — there is no prior snapshot to restore them from.
        created_buses: list[str] = []
        created_returns: list[str] = []
        replaced_user: list[str] = []   # user-placed master/return devices this overwrote
        entry_uids: list[Optional[int]] = []   # the chain entry each action made (D9·R5)
        for idx, action in enumerate(req.actions):
            if (action.type == "group"
                    and action.bus_id not in self._state["buses"]
                    and action.bus_id not in created_buses):
                created_buses.append(action.bus_id)
            if (action.type in ("return_reverb", "return_convolution", "return_delay")
                    and action.return_id
                    and action.return_id not in self._state["returns"]
                    and action.return_id not in created_returns):
                created_returns.append(action.return_id)
            for target in self._action_snapshot_targets(action):
                if target in inverse:
                    continue
                if target == MASTER:
                    inverse[target] = {
                        "master_gain_db": self._state["master_gain_db"],
                        "master_chain": json.loads(
                            json.dumps(self._state["master_chain"]))}
                elif target in self.buses:  # Phase 23
                    inverse[target] = json.loads(
                        json.dumps(self.buses[target]))
                elif target.startswith("return:"):
                    return_id = target.split(":", 1)[1]
                    if return_id in self.returns:
                        inverse[target] = json.loads(
                            json.dumps(self.returns[return_id]))
                else:
                    inverse[target] = json.loads(json.dumps(self.stems[target]))
            try:
                self._mutate(action)
                replaced_user += self._replaced_user_devices
                entry_uids.append(self._made_uid)
            except Exception as exc:
                # Atomicity: earlier actions in this proposal already mutated
                # in-memory state. Every touched target was snapshotted above
                # (first touch wins), so restore the lot — otherwise the session
                # silently diverges from the persisted/undo record until the
                # next _persist() bakes the partial apply in.
                self._restore_targets(inverse, created_buses, created_returns)
                return ApplyResult(
                    ok=False, proposal_id=proposal_id,
                    message=(f"apply failed on action {idx + 1}/"
                             f"{len(req.actions)} ({action.type}): {exc}; "
                             "all changes rolled back"))

        # Routing actions may have emptied a bus (route away / regroup). The
        # direct user paths (route_stem/group_stems) tidy via _drop_empty_buses
        # so realizeBuses never renders an orphan bus track — mirror that here.
        # A dropped bus must join the inverse first (undo restores member stems'
        # `bus` fields, which would otherwise point at a deleted bus); one this
        # proposal itself created and then emptied never existed outside it, so
        # it just leaves created_buses.
        if any(a.type in ("group", "route") for a in req.actions):
            for bus_id in self._unused_buses():
                if bus_id in created_buses:
                    created_buses.remove(bus_id)
                else:
                    inverse.setdefault(bus_id, json.loads(
                        json.dumps(self.buses[bus_id])))
                self._state["buses"].pop(bus_id, None)

        self._state["applied_seq"] += 1
        record["applied_seq"] = self._state["applied_seq"]
        # Re-stamp the pass to the one the spend actually LANDED in. It was set
        # once at propose time and never moved, but the only two readers of this
        # field are the per-pass wallets (_cumulative_gain, and
        # _accepted_proposals_this_pass), so a proposal held across an
        # advance_pass — ordinary in Ask mode, where a card can sit pending —
        # was billed to a pass that is over. Its gain then counted against
        # nobody: propose +5 dB in pass 1, apply it in pass 2, and pass 2's
        # wallet read empty, so a second +5 dB on the same stem sailed past the
        # 6 dB/pass cap (2026-08-16 review). Stamped AFTER the re-validation
        # above, so a proposal is never charged against its own budget check.
        record["pass_number"] = self.pass_number
        record["inverse"] = inverse
        record["created_buses"] = created_buses
        record["created_returns"] = created_returns
        # WHICH entities this acted on (D8·R6): an id is reused, an identity is not.
        named = (set(inverse) | set(created_buses) | {_return_target(r) for r in created_returns}
                 | {a["stem_id"] for a in record["actions"] if a.get("stem_id")})
        uids: dict[str, str] = {}
        for target in named:
            prior = inverse.get(target)
            uid = self._entity_uid(target) or (prior.get("uid") if isinstance(prior, dict) else None)
            if uid:
                uids[target] = uid
        record["target_uids"] = uids
        record["entry_uids"] = entry_uids
        record["status"] = "applied"
        # Don't let overwriting the user's own device be silent (see
        # _note_replaced_user_devices). Undo restores it — the inverse snapshot
        # above carries the whole chain — but the user should be told it went.
        if replaced_user:
            kinds = sorted(set(replaced_user))
            record["replaced_user_devices"] = kinds
            self._persist()
            return ApplyResult(
                ok=True, proposal_id=proposal_id,
                message=("replaced the "
                         + ", ".join(kinds)
                         + " you placed (undo restores it)"))
        self._persist()
        return ApplyResult(ok=True, proposal_id=proposal_id)

    def _action_snapshot_targets(self, action) -> list[str]:
        """The targets whose whole prior state apply() must snapshot so undo
        is bit-identical. A group touches each member stem (its `bus` field
        flips); everything else touches its single stem/master/bus target.
        Busses a group *creates* are handled separately (created_buses)."""
        if action.type == "group":
            return list(action.members)
        if action.type in _WET_KINDS and action.return_id:   # the RETURN form owns a return
            return [_return_target(action.return_id)]        # (the INSERT form falls through)
        if action.type == "send":
            return [action.stem_id]
        return [action.stem_id]

    def undo(self, proposal_id: str) -> ApplyResult:
        record = self._require_proposal(proposal_id)
        if record["status"] != "applied":
            return ApplyResult(ok=False, proposal_id=proposal_id,
                               message=f"proposal is '{record['status']}', not 'applied'")

        # v1 restriction (docs/03-mcp-tools.md): out-of-order undo only when no
        # later-applied proposal touches any of the same targets. Phase 28
        # folds in busses created/referenced by routing actions, so undoing a
        # group can't orphan a later proposal's stems on a deleted bus.
        mine = self._proposal_targets(record)
        for other in self._state["proposals"]:
            if (other["status"] == "applied"
                    and other["applied_seq"] > record["applied_seq"]
                    and mine & self._proposal_targets(other)):
                return ApplyResult(
                    ok=False, proposal_id=proposal_id,
                    message=(f"cannot undo: proposal '{other['proposal_id']}' was applied "
                             f"later and touches the same target(s); undo it first"))

        # Only onto the entities it acted on (D8·R6). A target whose id names ANOTHER track,
        # group or return now is left alone - and said, by `skipped_targets`.
        skipped = sorted(t for t in record["inverse"] if not self._same_entity(record, t))
        self.last_undo_skipped = skipped
        groups_before = set(self._state["buses"])
        returns_before = set(self._state["returns"])
        self._restore_targets(
            {t: prior for t, prior in record["inverse"].items() if t not in skipped},
            [b for b in record.get("created_buses") or [] if self._same_entity(record, b)],
            [r for r in record.get("created_returns") or []
             if self._same_entity(record, _return_target(r))])
        self._rebury_dissolved_groups(groups_before)
        self._rebury_removed_returns(returns_before)
        # A snapshot taken before entries had an identity restores entries with none: the
        # journal's replay falls back to the occurrence on those targets (daw_bridge).
        legacy = getattr(self, "unstamped_restores", set())
        for target in record["inverse"]:
            if target in skipped:
                continue
            try:
                chain = self._coedit_chain(target)
            except ValueError:
                continue
            if any(not isinstance(e.get("uid"), int) for e in chain):
                legacy.add(target)
        self.unstamped_restores = legacy
        record["status"] = "undone"
        self._persist()
        if skipped:
            return ApplyResult(
                ok=True, proposal_id=proposal_id,
                message=("left " + ", ".join(repr(t) for t in skipped) + " alone: "
                         + ("it is" if len(skipped) == 1 else "they are")
                         + " not what this move was made on (the name was taken by something new)"))
        return ApplyResult(ok=True, proposal_id=proposal_id)

    def _restore_targets(self, inverse: dict[str, Any],
                         created_buses: list[str],
                         created_returns: list[str]) -> None:
        """Restore every snapshotted target to its prior state — the shared
        machinery of undo() and apply()'s mid-proposal rollback."""
        for target, prior in inverse.items():
            if target == MASTER:
                if isinstance(prior, dict):
                    self._state["master_gain_db"] = prior["master_gain_db"]
                    self._state["master_chain"] = prior["master_chain"]
                else:  # inverse snapshot from before Phase 6: gain only
                    self._state["master_gain_db"] = prior
            elif target in self.stems:
                self.stems[target] = prior
            elif target.startswith("return:"):
                self._restore_record("return", target.split(":", 1)[1], prior)
            else:
                # Phase 23: a bus snapshot. One the user has since dissolved is
                # re-created here and, with no member left to hold it, taken out
                # again by undo() (`_rebury_dissolved_groups`, reader D8·R7).
                self._restore_record("bus", target, prior)
        # Phase 28: busses this proposal *created* have no prior state to
        # restore — delete them (after the snapshot restore, which may have
        # re-added an empty shell for a same-proposal bus comp). Members were
        # un-routed by their own stem snapshots above.
        for bus_id in created_buses:
            self._state["buses"].pop(bus_id, None)
        for return_id in created_returns:
            self._state["returns"].pop(return_id, None)
        self._reconcile_routing_refs()   # …and nothing is left pointing at what is not there (D8·R1)

    def _rebury_dissolved_groups(self, groups_before: set[str]) -> None:
        """A group the user DISSOLVED after the move is not brought back by the move's undo
        (reader D8·R7). Its snapshot re-created it - with no members: the dissolve re-homed
        them, and they are in nobody's snapshot - so the mixer drew an empty strip and the
        agent was shown a group that holds nothing. The undo is applied to the buried record
        instead: if the dissolve itself is undone (`restore_routing`), the group comes back
        as it was before the move. A group that was there before the undo is left alone."""
        for bus_id in [b for b in self._unused_buses() if b not in groups_before]:
            entry = self._state["buses"].pop(bus_id)
            uid = str(entry.get("uid") or "")
            for grave in reversed(self._state.get("routing_tombstones") or []):
                if (grave.get("kind") == "bus" and grave.get("id") == bus_id
                        and (not uid or not grave.get("uid") or grave.get("uid") == uid)):
                    grave["entry"] = entry
                    break

    def _rebury_removed_returns(self, returns_before: set[str]) -> None:
        """A return the user REMOVED after the move is not brought back by the move's undo
        (the return half of D8·R7; P3 batch Py8). Its snapshot re-created it - with nobody
        sending to it: the removal dropped every send, and they are in its tombstone, not in
        the snapshot - so the DAW drew a return track the user had deleted. As for a dissolved
        group, the undo is applied to the BURIED record: if the removal itself is undone
        (`restore_routing`), the return comes back with its sends and as it was before the
        move. A return that was there before the undo is left alone."""
        brought_back = [r for r in self._state["returns"] if r not in returns_before]
        for return_id in brought_back:
            entry = self._state["returns"].pop(return_id)
            uid = str(entry.get("uid") or "")
            for grave in reversed(self._state.get("routing_tombstones") or []):
                if (grave.get("kind") == "return" and grave.get("id") == return_id
                        and (not uid or not grave.get("uid") or grave.get("uid") == uid)):
                    grave["entry"] = entry
                    break
        if brought_back:
            self._reconcile_routing_refs()   # a stem's snapshot may hold a send into it

    def _restore_record(self, kind: str, record_id: str, prior: dict) -> None:
        """Put a bus / return snapshot back. One that is still there is replaced where it is;
        one the user has since REMOVED comes back at the place it left (reader D8·R5), which
        `_bury` recorded. A plain dict insert put it LAST, and the order of the returns is not
        cosmetic: "a return may only send to a LATER return" is an index comparison, so the
        restored return's cascade pointed backward and the reverse send was then accepted -
        session.json held two returns feeding each other. With no record of the place (the
        tombstones are bounded) it does come back last, and `_reconcile_routing_refs` drops a
        cascade that no longer runs forward."""
        key = "buses" if kind == "bus" else "returns"
        table = self._state[key]
        if record_id in table:
            table[record_id] = prior
            return
        graves = self._state.get("routing_tombstones") or []
        uid = str(prior.get("uid") or "") if isinstance(prior, dict) else ""
        tomb = next((g for g in reversed(graves)
                     if g.get("kind") == kind and g.get("id") == record_id
                     and (not uid or not g.get("uid") or g.get("uid") == uid)), None)
        items = list(table.items())
        at = len(items) if tomb is None else max(0, min(int(tomb.get("index", len(items))),
                                                        len(items)))
        items.insert(at, (record_id, prior))
        self._state[key] = dict(items)

    def _proposal_targets(self, record: dict) -> set[str]:
        """Every stem/bus/master id a proposal reads or writes — snapshot
        targets, busses it created, and busses its group/route actions point
        at — for the out-of-order-undo dependency check (Phase 28)."""
        targets = set(record.get("inverse") or {})
        targets |= set(record.get("created_buses") or [])
        targets |= {_return_target(rid) for rid in record.get("created_returns") or []}
        for action in record.get("actions") or []:
            if action.get("type") == "group":
                targets.add(action["bus_id"])
            elif action.get("type") == "route" and action.get("bus_id"):
                targets.add(action["bus_id"])
            elif action.get("type") == "send":
                targets.add(_return_target(action["return_id"]))
            elif (action.get("type") in _WET_KINDS
                  and action.get("return_id")):
                targets.add(_return_target(action["return_id"]))
        return targets

    def _replace_in_chain(self, chain: list[dict], drop) -> list[dict]:
        """Drop the entries `drop(e)` selects, RECORDING any the user placed.

        The Phase 22 "user-owned devices are not the agent's to replace"
        carve-out used to live in these filters, and it could not be turned on
        as written. Every type it guarded is in `_NATIVE_SINGLETON_KINDS` — the
        steps the native monitor hosts exactly ONCE — so sparing the user's
        entry leaves two of a kind in the chain: monitored last-wins, exported
        both, which is the 3.98 dB monitor/export divergence the 2026-08-06
        singleton audit was created to remove. Nothing in production ever wrote
        `owner` (only a test did), so the carve-out never fired and that
        divergence never happened; stamping the tag for real would have traded
        a silent data loss for silently wrong audio, which is the worse of the
        two (2026-08-16 review).

        So the agent still replaces, uniformly — and the loss stops being
        silent: what the user placed is reported back through ApplyResult, and
        undo restores it from the proposal's inverse snapshot. Real protection
        (as opposed to duplication) needs a different design — the agent
        editing the user's entry in place, or the action being refused with a
        reason — and that is a product decision, not a code fix.
        """
        self._replaced_user_devices += [e["type"] for e in chain
                                        if drop(e) and e.get("owner") == "user"]
        dropped = [e for e in chain if drop(e)]
        if len(dropped) == 1 and isinstance(dropped[0].get("uid"), int):
            self._carry_uid = (dropped[0]["type"], dropped[0]["uid"])   # see _stamp_entries
        return [e for e in chain if not drop(e)]

    def _mutate(self, action) -> None:
        """One action's state mutation, then its new entry's identity (`_stamp_entries`): the
        entry it replaced hands its own on, anything else is minted here - per ACTION, so two
        new entries of one type in one proposal are never confused. `_made_uid` is that
        entry's identity afterwards (None for an action that makes no chain entry): `apply`
        keeps it on the record, which is how the ledger later knows whether the entry an
        action made is still there (reader D9·R5)."""
        self._carry_uid: Optional[tuple] = None
        before = {e.get("uid") for chain in self._all_chains() for e in chain
                  if isinstance(e, dict) and isinstance(e.get("uid"), int)}
        self._mutate_one(action)
        carried = self._carry_uid
        self._stamp_entries(carried)
        self._carry_uid = None
        # WHICH entry this action made, by IDENTITY (pre-merge review 2026-10-06). It was found
        # by `id()` of the chain's dicts - and the dict a REPLACE drops is freed before the new
        # one is built, which then takes its address: "nothing new", no uid on the record, and
        # the ledger could never see that a replaced Utility's cut had been removed. The entry
        # that took over a replaced one carries that one's uid; anything else is the one uid
        # the chains did not hold before.
        after = {e.get("uid") for chain in self._all_chains() for e in chain
                 if isinstance(e, dict) and isinstance(e.get("uid"), int)}
        fresh = after - before
        if carried is not None and carried[1] in after:
            self._made_uid: Optional[int] = carried[1]
        else:
            self._made_uid = next(iter(fresh)) if len(fresh) == 1 else None

    def _mutate_one(self, action) -> None:
        self._replaced_user_devices: list[str] = []
        """One action's state mutation — no validation, no inverse snapshot,
        no persistence. Shared by apply() (which wraps it with all three)
        and probe scratch copies (Phase 12, which want none of them).
        Keeping a single mutation path is what makes probe → apply parity
        exact."""
        if action.type == "group":  # Phase 28: create bus (if absent) + route
            if action.bus_id not in self._state["buses"]:
                self._state["buses"][action.bus_id] = self._new_routing_record("bus", action.bus_id)
            for member in action.members:
                self.stems[member]["bus"] = action.bus_id
                self._clear_output_target(member)    # the agent's route is the output now (D2·R3)
            return
        if action.type == "route":  # Phase 28: re-home one stem (None = master)
            self.stems[action.stem_id]["bus"] = action.bus_id
            self._clear_output_target(action.stem_id)
            return
        if action.type in _WET_KINDS and action.return_id:   # the RETURN form (the INSERT form,
            if action.return_id not in self._state["returns"]:    # stem_id, falls through to the
                self._state["returns"][action.return_id] = self._new_routing_record(   # normal
                    "return", action.return_id)                                         # device paths)
            ret = self._state["returns"][action.return_id]
            ret["chain"] = self._replace_in_chain(
                ret["chain"], lambda e: e["type"] == action.type)
            ret["chain"].append(action.model_dump(exclude={"return_id", "stem_id"},
                                                  exclude_none=True))
            return
        if action.type == "send":
            holder = (self.buses[action.stem_id]
                      if action.stem_id in self.buses
                      else self.stems[action.stem_id])
            holder.setdefault("sends", {})[action.return_id] = action.gain_db
            return
        if action.stem_id == MASTER:
            if action.type == "gain":
                self._state["master_gain_db"] += action.delta_db
            elif action.type in ("limiter", "comp", "mb_comp", "ott", "overdrive",
                                 "transpose", "utility", "imager", "saturation",
                                 "clip", "deess", "gate", "transient",
                                 "width") + _WET_KINDS:
                # one per bus (limiter since Phase 6; comp/mb_comp since the
                # 2026-06 review — stacked dynamics were never intended;
                # transpose since Phase 32 — one key change; utility since device
                # freedom — the native chain hosts a single Utility step; imager
                # since 2026-07-05 — one 3-band imager; the space devices since
                # 2026-07-20 — one reverb + one delay + one IR reverb per chain is
                # exactly the native slot budget, and it keeps "make the reverb
                # longer" expressible as a re-propose; sat/clip/deess/gate/
                # transient/width since the 2026-08-06 singleton audit — every
                # _NATIVE_SINGLETON_KINDS step the monitor hosts once but the
                # offline export would render twice): a new one replaces the old
                chain = self._state["master_chain"]
                chain[:] = self._replace_in_chain(
                    chain, lambda e: e["type"] == action.type)
                _insert_canonical(chain, action.model_dump(
                    exclude={"stem_id", "return_id"}, exclude_none=True), _MASTER_KINDS)
            else:  # eq / match-eq / dynamic-eq — slot in at the canonical position
                # NB eq deliberately APPENDS, even for an identical band: the
                # deterministic agent iterates by re-proposing the same cut each
                # measured pass (e.g. the harshness composite), so same-band
                # entries are CUMULATIVE. The unmeasured repeat hazard is fenced
                # upstream (guardrails eq_gain_nonzero + the chat turn's
                # same-turn dedup, 2026-08-09).
                _insert_canonical(self._state["master_chain"], action.model_dump(
                    exclude={"stem_id", "return_id"}, exclude_none=True), _MASTER_KINDS)
            return
        if action.stem_id in self._state["buses"]:  # Phase 23: group bus
            bus = self._state["buses"][action.stem_id]
            if action.type == "gain":
                bus["gain_db"] += action.delta_db
            elif action.type == "comp":
                # one whole-file agent comp per bus (ranged bus actions are
                # rejected at validate time); user-racked comps survive
                bus["chain"] = self._replace_in_chain(
                    bus["chain"], lambda e: e["type"] == "comp")
                _insert_canonical(bus["chain"], action.model_dump(
                    exclude={"stem_id", "return_id"}, exclude_none=True), _BUS_KINDS)
            elif action.type == "mb_comp":
                bus["chain"] = self._replace_in_chain(
                    bus["chain"], lambda e: e["type"] == "mb_comp")
                _insert_canonical(bus["chain"], action.model_dump(
                    exclude={"stem_id", "return_id"}, exclude_none=True), _BUS_KINDS)
            elif action.type in ("utility", "imager", "ott", "overdrive", "saturation",
                                 "transient", "gate", "deess", "clip",
                                 "chorus", "flanger", "auto_filter") + _WET_KINDS:
                # one per bus — every native-singleton step (2026-08-06 singleton
                # audit: the monitor hosts one of each, last-wins, while the
                # offline export renders all of them); a new one replaces the
                # old. Ranged bus actions are rejected at validate time, so a
                # plain type match is the whole-file match here.
                bus["chain"] = self._replace_in_chain(
                    bus["chain"], lambda e: e["type"] == action.type)
                _insert_canonical(bus["chain"], action.model_dump(
                    exclude={"stem_id", "return_id"}, exclude_none=True), _BUS_KINDS)
            else:  # eq / match_eq / dynamic_eq / band_width — eq appends
                # (cumulative same-band iteration; see the master-chain note)
                _insert_canonical(bus["chain"], action.model_dump(
                    exclude={"stem_id", "return_id"}, exclude_none=True), _BUS_KINDS)
            return
        stem = self.stems[action.stem_id]
        if action.type == "gain" and action.time_range_s is not None:
            start_s, end_s = action.time_range_s
            stem["rides"].append({"start_s": start_s, "end_s": end_s,
                                  "delta_db": action.delta_db})
        elif action.type == "gain":
            stem["gain_db"] += action.delta_db
        elif action.type == "pan":
            stem["pan"] = action.pan
        elif action.type == "width":
            stem["width"] = action.width
        elif action.type == "mute":
            stem["muted"] = action.muted
        elif action.type == "exclude":
            stem["excluded"] = action.excluded
            if action.excluded:
                stem.pop("missing", None)   # a deliberate exclude: the file's return leaves it (D8·R9)
        elif action.type == "comp" and action.time_range_s is None:
            # one whole-file comp per stem (2026-06 review): a new one
            # replaces the old, so "adjust the compressor" is expressible.
            # Time-ranged comps coexist; overlap is rejected in guardrails.
            # User-racked comps (Phase 22) are not the agent's to replace.
            stem["chain"] = self._replace_in_chain(
                stem["chain"], lambda e: (e["type"] == "comp"
                                          and e.get("time_range_s") is None))
            _insert_canonical(stem["chain"], action.model_dump(
                exclude={"stem_id", "return_id"}, exclude_none=True), _STEM_KINDS)
        elif action.type == "mb_comp":
            # one multiband comp per stem, same replace semantics (and the
            # same Phase 22 carve-out for the user's own)
            stem["chain"] = self._replace_in_chain(
                stem["chain"], lambda e: e["type"] == "mb_comp")
            _insert_canonical(stem["chain"], action.model_dump(
                exclude={"stem_id", "return_id"}, exclude_none=True), _STEM_KINDS)
        elif action.type == "spectral_denoise":
            # one de-noise per stem; a new one replaces the old (Phase 29 repair)
            stem["chain"] = self._replace_in_chain(
                stem["chain"], lambda e: e["type"] == "spectral_denoise")   # reports a user-placed one (2026-09-01)
            _insert_canonical(stem["chain"], action.model_dump(
                exclude={"stem_id", "return_id"}, exclude_none=True), _STEM_KINDS)
        elif action.type == "tune":
            # one tuning correction per stem; a new one replaces the old
            # (Phase 31) — re-proposing is how the agent adjusts the cents
            stem["chain"] = self._replace_in_chain(
                stem["chain"], lambda e: e["type"] == "tune")   # reports a user-placed one (2026-09-01)
            _insert_canonical(stem["chain"], action.model_dump(
                exclude={"stem_id", "return_id"}, exclude_none=True), _STEM_KINDS)
        elif action.type == "deess" and action.time_range_s is None:
            # The agent's whole-file de-esser ADJUSTS the last whole-file one (2026-08-06
            # singleton audit: a new one replaces the old). Since 2026-10-17 a chain may hold
            # up to four (James: two-band de-essing, 4 kHz harshness + 8 kHz esses), so it
            # replaces that ONE and leaves any other where it is. Time-ranged de-essers coexist
            # (Phase 15/35).
            whole = [e for e in stem["chain"]
                     if e["type"] == "deess" and e.get("time_range_s") is None]
            last = whole[-1] if whole else None
            stem["chain"] = self._replace_in_chain(stem["chain"], lambda e: e is last)
            _insert_canonical(stem["chain"], action.model_dump(
                exclude={"stem_id", "return_id"}, exclude_none=True), _STEM_KINDS)
        elif action.type in ("utility", "imager", "ott", "overdrive", "saturation",
                             "transient", "gate", "duck", "clip",
                             "chorus", "flanger", "auto_filter") + _WET_KINDS:
            # one per stem — every native-singleton step (the chain hosts a single
            # Utility/imager/OTT/overdrive/sat/transient/gate/duck/clip stage, and
            # one reverb + delay + IR reverb is exactly its space-device slot
            # budget); a new one replaces the old, so re-proposing adjusts the
            # params (the 2026-08-06 singleton audit closed sat/transient/gate/
            # duck/clip, which used to append and diverge monitor from export).
            # chorus + flanger joined on 2026-10-05 (reader D9·R8): both are in
            # _NATIVE_SINGLETON_KINDS and both appended here, so "a bit more chorus"
            # rendered two in series offline while the DAW played one - and the 0.5
            # wet cap, judged per action, composed past itself.
            # test_every_native_singleton_the_agent_may_propose_replaces pins the two lists.
            # User-placed ones (Phase 22 carve-out) are not the agent's to replace.
            stem["chain"] = self._replace_in_chain(
                stem["chain"], lambda e: e["type"] == action.type)
            _insert_canonical(stem["chain"], action.model_dump(
                exclude={"stem_id", "return_id"}, exclude_none=True), _STEM_KINDS)
        else:  # eq / ranged comp+deess / match_eq / dynamic_eq / band_width append
            # (eq same-band entries are CUMULATIVE by design — see the
            # master-chain note above)
            _insert_canonical(stem["chain"], action.model_dump(
                exclude={"stem_id", "return_id"}, exclude_none=True), _STEM_KINDS)

    def probe_copy(self) -> "Session":
        """In-memory scratch copy for probe renders (Phase 12): identical
        state, read-only (never writes session.json), safe to mutate via
        _mutate and discard."""
        copy = Session.__new__(Session)
        copy.project_dir = self.project_dir
        copy._read_only = True
        copy._detached = True    # scratch: sync_from_disk must never reload it
        copy._stamp = None
        copy._session_path = self._session_path
        copy._state = json.loads(json.dumps(self._state))
        return copy

    def get_proposals(self, status: Optional[str] = None) -> list[ProposalSummary]:
        records = self._state["proposals"]
        if status is not None:
            records = [r for r in records if r["status"] == status]
        return [ProposalSummary.model_validate({
            k: r.get(k) for k in ("proposal_id", "status", "rationale",
                                  "actions", "predicted_impact",
                                  "measured_impact", "pass_number",
                                  "applied_seq")
        }) for r in records]

    # ---------- internals ----------

    def _discover_stems(self) -> None:
        wavs = project_wavs(self.project_dir)                 # any case of the extension (S1·R17)
        # A file named master.wav would mint the stem id "master", which collides with the
        # reserved master-bus id: actions, rendering and the guardrails all special-case
        # stem_id == "master", so the stem would silently drive the master chain. It is LEFT
        # OUT, and said (P2 batch T, reader S1·R18; James, 2026-10-05: "open the folder with
        # that one file left out and said; the same rule for master.wav and Master.wav"). It
        # used to raise - which refused the whole folder: the DAW opened it with no tracks, its
        # first save made that permanent, and every later Session() of the project raised too.
        # A stems folder holds a master.wav for an ordinary reason: it is the reference mix.
        # A file some record already claims (a session from before the name was reserved in
        # every case) stays that record's.
        claimed_files = {str(rec.get("path") or "") for rec in self.stems.values()
                         if isinstance(rec, dict)}
        self.left_out_wavs = sorted(name for name in wavs
                                    if Path(name).stem.casefold() == MASTER
                                    and name not in claimed_files)
        if self.left_out_wavs:
            wavs = [name for name in wavs if name not in self.left_out_wavs]
        on_disk = {Path(name).stem: name for name in wavs}
        # A stem's file is its record's ``path``, not its id: a RENAMED stem (rename_stem) keeps
        # its original wav under the new id, so claims are matched by filename. Matching by id
        # here used to be the only rule — it would have minted a duplicate stem from the old file
        # and marked the renamed one missing on the very next Session (2026-09-02).
        files_on_disk = set(wavs)
        # A record names its file as it was called when it was claimed. A file whose CASE has
        # changed since (re-cased in the Finder, or a session written before the listing matched
        # `.WAV`) is still that file wherever the volume says the two names are one: the record
        # follows the name on disk, so the stem is neither "missing" nor minted a second time.
        by_fold = {name.casefold(): name for name in wavs}
        for stem in self.stems.values():
            path = str(stem.get("path", ""))
            twin = by_fold.get(path.casefold()) if path and path not in files_on_disk else None
            if twin and _one_file(self.project_dir / path, self.project_dir / twin):
                stem["path"] = twin
        claimed: dict[str, str] = {}          # filename -> the stem id that owns it
        for sid, stem in self.stems.items():
            claimed.setdefault(str(stem.get("path", "")), sid)
        missing = {sid for sid, stem in self.stems.items()
                   if str(stem.get("path", "")) not in files_on_disk}
        for stem_id in missing:
            stem = self.stems[stem_id]
            if not stem.get("excluded"):
                # A hand-deleted (or unmounted) stem wav used to RAISE here — on every
                # Session construction, so one missing file silently bricked the whole
                # project brain-side: every run/analyze/readout failed the same way until
                # session.json was hand-edited (2026-08-09 §2 audit P0). Degrade instead:
                # the stem sits out (excluded) and is flagged `missing`; the rest of the
                # project keeps working. The DAW's own missing-media prompt is the user-
                # visible surface, and a relink re-registers the stem (which clears this).
                stem["excluded"] = True
                stem["missing"] = True
        retired = set(self._state.get("retired_wavs", []))
        registered = _BY_REGISTRATION.get(str(self.project_dir))   # None = read the folder
        for basename, filename in on_disk.items():
            owner = claimed.get(filename)
            if owner is None and registered is not None and filename not in registered:
                # A project the DAW holds (reader S1·R19): a file no record claims is a stem
                # only when a registration names it - and only then may it take over the
                # record of the stem it is named after (the re-drop rule below). Not retired:
                # an import of it later mints it. Not an error: it is simply not a track.
                continue
            if owner is None and filename in retired:
                # An orphan a rename + re-drop left behind (below), or a file claim_stem_file moved
                # a stem off. Checked BEFORE the re-drop rule (full audit 2026-10-02, S4·R8): a
                # retired file named after a live stem would pull its record back onto old audio.
                continue
            if owner is None and basename in self.stems:
                # A file named after an existing stem whose record points at ANOTHER file: audio
                # re-dropped onto a renamed track (import_stem writes <id>.wav). The record follows
                # the new file, and the old one is RETIRED — never minted as a ghost stem that the
                # offline render would quietly mix in.
                stem = self.stems[basename]
                old_file = str(stem.get("path", ""))
                if old_file and old_file != filename and old_file in files_on_disk:
                    self._state.setdefault("retired_wavs", []).append(old_file)
                    retired.add(old_file)
                stem["path"] = filename
                owner = basename
            if owner is None and (basename in self.buses or basename in self.returns):
                # One namespace (reader D8·R3): a file named after a GROUP or a RETURN is not
                # minted as a second holder of the name - every lookup that asks the groups
                # first would write the group for the track. It stays a file; an import of it
                # under a free name makes it a stem. (`master.wav` raises above: that name can
                # never be freed. This one is freed by renaming the group.)
                continue
            minted = (owner or basename) not in self.stems
            stem = self.stems.setdefault(owner or basename, {"path": filename})
            if stem.pop("missing", None):
                stem["excluded"] = False   # the file came back (remount/re-copy/relink) — rejoin
            for key, default in STEM_DEFAULTS.items():
                stem.setdefault(key, json.loads(json.dumps(default)))
            if minted:
                stem["uid"] = self._mint_uid(owner or basename)
        for stem_id, stem in self.stems.items():
            # A stem's identity (D8·R6): a record from before they were kept gets one derived
            # from its id, like a bus's below; `reset_stem` gives the record a NEW one when a
            # new track takes a deleted track's name.
            stem.setdefault("uid", f"{stem_id}@0")
        for bus_id, bus in self.buses.items():
            for key, default in BUS_DEFAULTS.items():
                bus.setdefault(key, json.loads(json.dumps(default)))
            bus.setdefault("uid", f"{bus_id}@0")   # see the returns loop below
            # Adopt pending members whose audio has since landed: a track grouped while
            # empty registers as a stem the moment audio drops on it (any import path) —
            # it becomes a REAL member of the bus it was grouped into, and leaves pending.
            # An EXCLUDED stem of the same name is a soft-deleted track whose name the live
            # empty track reuses — adoption waits for the import boundary's reset_stem to
            # revive it (adopting now would hand the bus to the dead record, and reset_stem
            # would wipe it straight after).
            for name in [n for n in bus["pending"]
                         if n in self.stems and not self.stems[n].get("excluded")]:
                bus["pending"].remove(name)
                self.stems[name]["bus"] = bus_id
        for return_id, ret in self.returns.items():
            for key, default in RETURN_DEFAULTS.items():
                ret.setdefault(key, json.loads(json.dumps(default)))
            # A record from before identities were kept: derived from its id, so a read-only
            # Session and a writable one agree, and persisted by the first writable one - a
            # rename after that carries it (the record moves; the uid is in it).
            ret.setdefault("uid", f"{return_id}@0")

    def frozen_by(self, target: str) -> str:
        """The frozen track that holds this stem or group ("" = live): the stem's own `frozen`
        fact, or - for a group - the fact any of its members reports for it."""
        target = str(target or "")
        stem = self.stems.get(target)
        if isinstance(stem, dict) and stem.get("frozen"):
            return str(stem["frozen"])
        if target in self._state.get("buses", {}):
            holder = f"bus:{target}"
            if any(isinstance(rec, dict) and rec.get("frozen") == holder
                   for rec in self.stems.values()):
                return holder
        return ""

    def _frozen_refusal(self, action) -> str:
        """Why this action is refused because its track is FROZEN in the DAW, or "" (S2·R12).
        A device, a level, a width, a ride on a frozen track - or on the group a frozen group
        freeze holds - would change what is exported and nothing that is heard. Routing and
        post-fader sends are not behind the freeze and stay open; the master and a return are
        never frozen."""
        if action.type in ("group", "route", "send"):
            return ""
        # Pan, mute and exclude stay LIVE on a frozen track in the DAW (M16: the strip locks
        # its fader and its width only) - they act after the freeze file. Refused here, the
        # agent was told "nothing that is heard" about a move that is (pre-merge review
        # 2026-10-06).
        if action.type in ("pan", "mute", "exclude"):
            return ""
        if action.type in _WET_KINDS and getattr(action, "return_id", None):
            return ""
        target = str(getattr(action, "stem_id", "") or "")
        if not target or target == MASTER:
            return ""
        by = self.frozen_by(target)
        if not by:
            return ""
        who = (f"'{target}' is frozen" if by == target
               else f"'{target}' is held by {by}, which is frozen")
        return (f"{who} in the DAW: it plays its freeze file, so this would change what is "
                f"exported and nothing that is heard. Unfreeze it first with the command "
                f"track.freeze {{track: \"{by}\", on: false}}, then propose this again")

    def validate_action_targets(self, actions: list) -> list[Violation]:
        """Structural target checks (unknown stems, master-only + bus rules,
        Phase 28 routing) — shared by the proposal path and probes (Phase 12),
        which skip guardrails but still need errors for nonsense targets."""
        violations = []
        # Phase 28: a group earlier in THIS proposal creates a bus that later
        # actions may target (e.g. group drums, then comp the drum bus).
        pending_buses: set[str] = set()
        pending_returns: set[str] = set()
        proposed_triggers: dict[str, str] = {}   # stem -> the trigger a keyed action in THIS proposal names
        for i, action in enumerate(actions):
            if action.type not in SUPPORTED_ACTION_TYPES:
                violations.append(Violation(
                    action_index=i, rule="unsupported_action_type",
                    message=f"action type '{action.type}' is not implemented yet",
                    limit=0, value=0))
                continue
            if (frozen := self._frozen_refusal(action)):
                violations.append(Violation(action_index=i, rule="frozen_track",
                                            message=frozen, limit=0, value=0))
                continue
            if action.type == "group":
                group_violations = self._validate_group(i, action, pending_returns)
                violations += group_violations
                if not group_violations:
                    pending_buses.add(action.bus_id)
                continue
            if action.type == "route":
                violations += self._validate_route(i, action, pending_buses)
                continue
            # A space device in its RETURN form owns/creates a return; in its INSERT form
            # (stem_id, 2026-07-20) it is an ordinary device and falls through to the normal
            # stem / bus / master target checks below.
            if action.type in _WET_KINDS and action.return_id:
                return_violations = self._validate_return_effect(
                    i, action, pending_buses, pending_returns)
                violations += return_violations
                if not return_violations:
                    pending_returns.add(action.return_id)
                continue
            if action.type == "send":
                violations += self._validate_send(
                    i, action, pending_buses, pending_returns)
                continue
            if action.stem_id == MASTER:
                if action.type not in MASTER_ACTION_TYPES:
                    violations.append(Violation(
                        action_index=i, rule="master_action_type",
                        message=f"'{action.type}' may not target 'master' "
                                f"(allowed: {sorted(MASTER_ACTION_TYPES)})",
                        limit=0, value=0))
                elif getattr(action, "time_range_s", None) is not None:
                    # Phases 5 + 15: time-confined moves are stems-only —
                    # the master is always whole-file.
                    violations.append(Violation(
                        action_index=i, rule="master_ride_unsupported",
                        message=f"time_range_s on '{action.type}' is stems-"
                                "only; master processing must be whole-file",
                        limit=0, value=0))
            elif action.stem_id in self.buses or action.stem_id in pending_buses:
                # Phase 23: group busses take the stem palette minus the
                # stem-bound moves, always whole-file. (pending_buses, Phase 28:
                # a bus a group created earlier in this same proposal.)
                if action.type not in BUS_ACTION_TYPES:
                    violations.append(Violation(
                        action_index=i, rule="bus_action_type",
                        message=f"'{action.type}' may not target bus "
                                f"'{action.stem_id}' "
                                f"(allowed: {sorted(BUS_ACTION_TYPES)})",
                        limit=0, value=0))
                elif action.type == "auto_filter" and action.key_stem_id:
                    # A trigger is stem machinery (the key renders as a stem), as the Duck's is;
                    # on a group the Auto Filter follows the group itself (Phase 63).
                    violations.append(Violation(
                        action_index=i, rule="auto_filter_key_on_group",
                        message=f"an Auto Filter on group '{action.stem_id}' follows the group's "
                                "own level - a trigger track is for one on a track; leave "
                                "key_stem_id empty",
                        limit=0, value=0))
                elif getattr(action, "time_range_s", None) is not None:
                    violations.append(Violation(
                        action_index=i, rule="bus_ride_unsupported",
                        message=f"time_range_s on '{action.type}' is stems-"
                                "only; bus processing must be whole-file",
                        limit=0, value=0))
            elif action.type == "limiter":
                violations.append(Violation(
                    action_index=i, rule="limiter_master_only",
                    message="the limiter lives on the master bus only "
                            f"(got stem '{action.stem_id}')",
                    limit=0, value=0))
            elif action.type == "transpose":
                violations.append(Violation(
                    action_index=i, rule="transpose_master_only",
                    message="transpose (whole-song key change) is master-only "
                            f"(got stem '{action.stem_id}')",
                    limit=0, value=0))
            elif action.type == "duck" and (
                    action.key_stem_id == action.stem_id
                    or action.key_stem_id not in self.stems
                    or self.stems[action.key_stem_id].get("excluded")
                    or self.duck_key_cycles(action.stem_id, action.key_stem_id)):
                violations.append(Violation(
                    action_index=i, rule="duck_key_stem",
                    message=(f"duck key must be a different existing stem that is in the "
                             f"mix and does not itself duck under the target (two tracks "
                             f"cannot trigger each other); got key '{action.key_stem_id}' for target "
                             f"'{action.stem_id}' (stems: {sorted(self._live_stem_ids())})"),
                    limit=0, value=0))
            elif action.type == "auto_filter" and action.key_stem_id and (
                    action.key_stem_id == action.stem_id
                    or action.key_stem_id not in self.stems
                    or self.stems[action.key_stem_id].get("excluded")
                    or self.key_cycles(action.stem_id, action.key_stem_id)):
                violations.append(Violation(
                    action_index=i, rule="auto_filter_key_stem",
                    message=(f"an Auto Filter's trigger must be a different existing stem in the "
                             f"mix that does not itself listen to the target (two tracks cannot "
                             f"trigger each other) - or empty, to follow its own input; got "
                             f"'{action.key_stem_id}' for '{action.stem_id}' "
                             f"(stems: {sorted(self._live_stem_ids())})"),
                    limit=0, value=0))
            elif action.type in KEYED_TYPES and action.key_stem_id and (
                    (proposed_triggers.get(action.stem_id) not in (None, action.key_stem_id))
                    or self.trigger_conflict(action.stem_id, action.key_stem_id,
                                             action.type) is not None):
                why = (self.trigger_conflict(action.stem_id, action.key_stem_id, action.type)
                       or f"this proposal already gives '{action.stem_id}' the trigger "
                          f"'{proposed_triggers.get(action.stem_id)}' - a track has one trigger")
                violations.append(Violation(action_index=i, rule="one_trigger_per_track",
                                            message=why, limit=0, value=0))
            elif action.stem_id not in self.stems:
                violations.append(Violation(
                    action_index=i, rule="unknown_stem",
                    message=f"unknown stem '{action.stem_id}'; stems: {sorted(self.stems)}",
                    limit=0, value=0))
            elif action.type != "exclude" and self.stems[action.stem_id].get("excluded"):
                # A stem that is OUT of the mix — soft-deleted by the DAW (the record stays for
                # the edit's clips + undo), sat out by an exclude action, or its file missing —
                # is a ghost: a move on it renders nowhere and is silently wiped the moment the
                # name is re-registered (2026-09-04: an agent added a loop whose deleted
                # namesake still held a stem, gated the ghost mid-turn, and the deferred
                # registration's reset took the gate with it). Only an exclude action (the way
                # back in) may target it; everything else needs the stem live first.
                violations.append(Violation(
                    action_index=i, rule="excluded_stem",
                    message=(f"stem '{action.stem_id}' is excluded from the mix (deleted, sat "
                             "out, or a re-added track the brain has not registered yet) — "
                             "only an exclude action may target it"),
                    limit=0, value=0))
            elif getattr(action, "time_range_s", None) is not None:
                # A range that starts where the track has already ended (reader D9·R9):
                # `[5000, 6000]` on an 8 s stem rendered as nothing and still took a comp or
                # de-ess slot and charged its make-up. One that starts inside the track and
                # runs past its end is fine - the render stops at the audio.
                end_s = self.stem_end_s(action.stem_id)
                start_s = float(action.time_range_s[0])
                if end_s is not None and start_s >= end_s:
                    violations.append(Violation(
                        action_index=i, rule="time_range_past_end",
                        message=(f"time_range_s starts at {start_s:.1f} s, and "
                                 f"'{action.stem_id}' ends at {end_s:.1f} s: the move would "
                                 "change nothing. Read the sections for where this track plays"),
                        limit=round(end_s, 1), value=round(start_s, 1)))
            if action.type in KEYED_TYPES and getattr(action, "key_stem_id", ""):
                proposed_triggers.setdefault(action.stem_id, action.key_stem_id)
        return violations

    def stem_end_s(self, stem_id: str, _seen: Optional[set] = None) -> Optional[float]:
        """Where a stem's audio ends on the timeline, in seconds - its file as its one clip
        plays it (the `clip` window, as `dsp._placed` reads it), or later if a track whose
        Audio To is this one runs longer. None when the file cannot be read (a missing file is
        somebody else's refusal)."""
        stem = self.stems.get(stem_id)
        if stem is None:
            return None
        seen = _seen if _seen is not None else set()
        if stem_id in seen:
            return None
        seen.add(stem_id)
        try:
            import soundfile as sf
            info = sf.info(str(self.project_dir / stem["path"]))
            file_s = info.frames / float(info.samplerate)
        except Exception:  # noqa: BLE001 - unreadable: no bound to give
            return None
        if stem.get("clips"):                           # several clips: where the last one ends
            end_s = max(float(w["at_s"]) + max(0.0, min(float(w["for_s"]), file_s - float(w["from_s"])))
                        for w in stem["clips"])
        else:
            clip = stem.get("clip") or {}
            first = min(file_s, float(clip.get("from_s", 0.0) or 0.0))
            length = float(clip.get("for_s", 0.0) or 0.0)
            played = file_s - first if length <= 0.0 else min(file_s - first, length)
            end_s = float(clip.get("at_s", 0.0) or 0.0) + max(0.0, played)
        for feeder in self.stem_feeders(stem_id):
            fed = self.stem_end_s(feeder, seen)
            if fed is not None:
                end_s = max(end_s, fed)
        return end_s

    def _live_stem_ids(self) -> list[str]:
        return [s for s, rec in self.stems.items() if not rec.get("excluded")]

    def _validate_group(self, i: int, action,
                        pending_returns: set[str] = frozenset()) -> list[Violation]:
        """Phase 28: a group action may name a new or existing bus, but not a
        stem, a return, or 'master' (one namespace), and every member must be
        a stem."""
        violations = []
        if not action.bus_id.strip():
            violations.append(Violation(
                action_index=i, rule="bus_name_empty",
                message="group bus_id must not be blank", limit=0, value=0))
        elif (self.name_taken(action.bus_id, "bus")
              or action.bus_id in pending_returns):
            violations.append(Violation(
                action_index=i, rule="bus_name_taken",
                message=f"group bus_id '{action.bus_id}' collides with a stem, "
                        "a return, or 'master', or has ':' in it (stems, busses, "
                        "returns and 'master' share one namespace)", limit=0, value=0))
        for member in action.members:
            if member not in self.stems:
                violations.append(Violation(
                    action_index=i, rule="unknown_stem",
                    message=f"group member '{member}' is not a stem; stems: "
                            f"{sorted(self.stems)}", limit=0, value=0))
        return violations

    def _validate_route(self, i: int, action,
                        pending_buses: set[str]) -> list[Violation]:
        """Phase 28: a route moves a real stem to an existing bus, a bus a
        group created earlier this proposal, or None (back to the master)."""
        violations = []
        if action.stem_id not in self.stems:
            violations.append(Violation(
                action_index=i, rule="unknown_stem",
                message=f"route target '{action.stem_id}' is not a stem; "
                        f"stems: {sorted(self.stems)}", limit=0, value=0))
        if (action.bus_id is not None and action.bus_id not in self.buses
                and action.bus_id not in pending_buses):
            violations.append(Violation(
                action_index=i, rule="unknown_bus",
                message=f"route destination '{action.bus_id}' is not a bus; "
                        f"busses: {sorted(self.buses)}", limit=0, value=0))
        return violations

    def _validate_return_effect(self, i: int, action, pending_buses: set[str],
                                pending_returns: set[str]) -> list[Violation]:
        """Phase 38: return ids are separate state, but may not collide with
        stems/busses/master because the UI and proposal grammar share names.
        Creating a NEW return is capped at MAX_RETURNS (Ableton parity), same
        as the direct add_return path."""
        violations = []
        if not action.return_id.strip():
            violations.append(Violation(
                action_index=i, rule="return_name_empty",
                message="return_id must not be blank", limit=0, value=0))
        elif (self.name_taken(action.return_id, "return")
              or action.return_id in pending_buses):
            violations.append(Violation(
                action_index=i, rule="return_name_taken",
                message=f"return_id '{action.return_id}' collides with a stem, "
                        "bus, or 'master', or has ':' in it", limit=0, value=0))
        elif (action.return_id not in self.returns
              and action.return_id not in pending_returns
              and len(self.returns) + len(pending_returns) >= MAX_RETURNS):
            violations.append(Violation(
                action_index=i, rule="return_limit",
                message=f"return limit reached — {MAX_RETURNS} max "
                        "(Ableton-style)", limit=MAX_RETURNS,
                value=len(self.returns) + len(pending_returns)))
        return violations

    def _validate_send(self, i: int, action, pending_buses: set[str],
                       pending_returns: set[str]) -> list[Violation]:
        """Phase 38: sends are agent-owned source -> return links. Sources may
        be stems or group busses; destinations must be existing or created
        earlier in the same proposal."""
        violations = []
        if action.stem_id == MASTER:
            violations.append(Violation(
                action_index=i, rule="send_source_master",
                message="master cannot send to a return; sends start from stems "
                        "or group busses", limit=0, value=0))
        elif (action.stem_id not in self.stems and action.stem_id not in self.buses
              and action.stem_id not in pending_buses):
            violations.append(Violation(
                action_index=i, rule="unknown_send_source",
                message=f"send source '{action.stem_id}' is not a stem or bus; "
                        f"stems: {sorted(self.stems)}, busses: {sorted(self.buses)}",
                limit=0, value=0))
        if (action.return_id not in self.returns
                and action.return_id not in pending_returns):
            violations.append(Violation(
                action_index=i, rule="unknown_return",
                message=f"send destination '{action.return_id}' is not a return; "
                        f"returns: {sorted(self.returns)}", limit=0, value=0))
        return violations

    # The device types whose level trim guardrails._charge_gain charges, and how a
    # record's charge is read. ONE table for both wallets (2026-09-01 sweep, P1):
    # `_cumulative_gain` used to count only `gain` actions, so every eq / tilt /
    # saturation / clip / transient / overdrive / ott / utility trim was charged
    # inside its own proposal and then forgotten — three proposals of eight EQ
    # bands at +0.75 dB each landed +18 dB of applied trim in ONE pass against a
    # 6 dB cap (the 2026-07-17 memo's "per-PASS cap is intact" was wrong). Positive
    # only where guardrails is positive-only (a cut is free); signed where it is.
    # SINGLETON types replace the stem's previous entry, so the LAST applied record
    # per (target, type) is what the chain carries; APPEND types (gain, eq, tilt)
    # sum. comp / mb_comp makeup are charged here for the per-pass wallet and read
    # off the live chains for the session wallet (see _session_gain_baseline).
    _TRIM_SINGLETONS = frozenset({"saturation", "overdrive", "clip", "transient",
                                  "ott", "utility", "comp", "mb_comp"})

    @staticmethod
    def _record_trim_charge(action: dict) -> Optional[float]:
        """What one applied record charged — guardrails' own function, so the ledger can never
        again forget what the check charged (full audit 2026-10-02, D9·R3)."""
        from .guardrails import level_charge_db
        return level_charge_db(action)

    #== A cut is credit only while it is still there (reader D9·R5) ==
    # The ledgers read the applied records, never the live chain - right for the user's own
    # devices, which are not the agent's budget. But a cut the AGENT made and the user then
    # deleted, dragged back to 0 or switched off stayed in the ledger as credit: four +6 dB
    # bells landed on a band whose -12 dB cut was gone, +24 dB against a +12 rail. Each applied
    # action keeps the identity of the entry it made (`entry_uids`); the ledger counts the side
    # of it that is CREDIT only while that entry is on the chain as the action wrote it. What it
    # spent is always counted. A record from before identities reads as it always did.

    def _entry_made_by(self, record: dict, index: int, action: dict) -> tuple[bool, Optional[dict]]:
        """(known, entry): whether the record says which entry this action made, and that entry
        if it is still on the target's chain."""
        uids = record.get("entry_uids")
        uid = uids[index] if isinstance(uids, list) and index < len(uids) else None
        if not isinstance(uid, int):
            return False, None
        try:
            chain = self._coedit_chain(str(action.get("stem_id") or ""))
        except ValueError:
            return True, None
        return True, next((e for e in chain if e.get("uid") == uid), None)

    @staticmethod
    def _same_numbers(a, b) -> bool:
        if isinstance(a, dict) and isinstance(b, dict):
            return all(Session._same_numbers(a.get(k), b.get(k)) for k in set(a) | set(b))
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) \
                and not isinstance(a, bool) and not isinstance(b, bool):
            return abs(float(a) - float(b)) < 1e-6
        return a == b

    def _still_as_written(self, record: dict, index: int, action: dict,
                          fields: tuple[str, ...]) -> bool:
        """False only when the record KNOWS the entry it made and that entry is gone, bypassed,
        or no longer holds the values the action wrote in `fields`."""
        known, entry = self._entry_made_by(record, index, action)
        if not known:
            return True
        return (entry is not None and entry.get("enabled", True) is not False
                and all(self._same_numbers(entry.get(f), action.get(f)) for f in fields))

    _EQ_CURVE_FIELDS = {"eq": ("band",), "tilt": ("tilt_db", "pivot_hz"),
                        "match_eq": ("band_gains_db",)}

    def _ledger_eq_action(self, record: dict, index: int, action: dict) -> dict:
        """The action as the composed-EQ rail should count it: one-sided (`spent_only` - its
        lift on the boost side, its dip on the cut side, neither as credit for the other) once
        the entry it made is no longer there as written."""
        fields = self._EQ_CURVE_FIELDS.get(action.get("type"))
        if fields is None or self._still_as_written(record, index, action, fields):
            return action
        return {**action, "spent_only": True}

    def _ledger_charge(self, record: dict, index: int, charge: Optional[float]) -> Optional[float]:
        """A record's level charge as the gain wallets should count it. A NEGATIVE charge is
        credit, and only a Utility's gain can be one that has an entry to ask (a fader cut is one
        number both the user and the agent write: there is nothing to tell whose it is now)."""
        if charge is None or charge >= 0.0:
            return charge
        action = record["actions"][index]
        if action.get("type") == "utility" \
                and not self._still_as_written(record, index, action, ("gain_db",)):
            return 0.0
        return charge

    @staticmethod
    def _is_singleton_record(action: dict) -> bool:
        """A whole-file singleton REPLACES the target's previous entry of its type; a ranged
        entry (several non-overlapping ranged comps are legal) is appended beside it."""
        return (action["type"] in Session._TRIM_SINGLETONS
                and action.get("time_range_s") is None)

    @staticmethod
    def _replaced_by(actions: list) -> set[tuple[str, str]]:
        """(target, type) pairs this proposal's whole-file singleton actions REPLACE —
        their previously applied trim leaves the wallet, or adjusting makeup 4 -> 6
        would cost 10 dB of budget instead of 2."""
        out: set[tuple[str, str]] = set()
        for a in actions or ():
            t = getattr(a, "type", None) if not isinstance(a, dict) else a.get("type")
            sid = getattr(a, "stem_id", None) if not isinstance(a, dict) else a.get("stem_id")
            rng = (getattr(a, "time_range_s", None) if not isinstance(a, dict)
                   else a.get("time_range_s"))
            if t in Session._TRIM_SINGLETONS and sid and rng is None:
                out.add((sid, t))
        return out

    def _cumulative_gain_this_pass(self, actions: Optional[list] = None) -> dict[str, float]:
        """Level already *applied* (and not undone) per target in the current pass —
        gain moves AND every device trim guardrails charges, minus the singleton
        entries this proposal would replace."""
        return self._cumulative_gain(self.pass_number, replaces=self._replaced_by(actions))

    def _ranged_gain(self, pass_number: Optional[int]) -> dict[str, tuple[float, float]]:
        """(up, down) per target: the TIME-RANGED level applied (and not undone) in one pass or
        the session, one-sided — `up` the sum of ranged lifts, `down` the sum of ranged cuts.
        Never netted against each other or against a whole-file move (guardrails.is_ranged).
        The session scope reads ranged comp makeup off the live chains, like the whole-file
        makeup in `_session_gain_baseline`, and adds what trimmed records spent."""
        from .guardrails import is_ranged, level_charge_db
        out: dict[str, list[float]] = {}

        def add(sid: str, charge: Optional[float]) -> None:
            if charge:
                slot = out.setdefault(sid, [0.0, 0.0])
                slot[0 if charge > 0 else 1] += charge

        for record in self._state["proposals"]:
            if record["status"] != "applied":
                continue
            if pass_number is not None and record["pass_number"] != pass_number:
                continue
            for action in record["actions"]:
                if not is_ranged(action):
                    continue
                if pass_number is None and action["type"] == "comp":
                    continue                      # session scope: read off the chain, below
                if not self._same_entity(record, action["stem_id"]):
                    continue                      # spent on what held this id before (D8·R6)
                add(action["stem_id"], level_charge_db(action))
        if pass_number is None:
            for sid, (up, down) in self._state.get("trimmed_ranged_gain", {}).items():
                add(sid, float(up))
                add(sid, float(down))
            holders = {sid: stem["chain"] for sid, stem in self.stems.items()}
            holders.update({bid: bus["chain"] for bid, bus in self.buses.items()})
            for sid, chain in holders.items():
                for entry in chain:
                    if entry["type"] == "comp" and is_ranged(entry):
                        add(sid, level_charge_db(entry))
        return {sid: (up, down) for sid, (up, down) in out.items()}

    def _session_gain_baseline(self, actions: list) -> dict[str, float]:
        """The agent's session gain budget already spent per target (Phase
        10): gain applied (and not undone) across the session — undo refunds
        it — plus, 2026-06 review, the level the chains' current comp/mb
        makeup adds (makeup draws from the same budget in guardrails).
        Makeup of entries this proposal would *replace* (one whole-file comp
        / mb_comp per stem) is excluded: adjusting makeup 4 -> 6 costs 2 dB
        of budget, not 10."""
        # comp / mb_comp makeup is read off the live chains below (with the
        # replace exclusion), so the record walk skips those two here.
        totals = self._cumulative_gain(None, replaces=self._replaced_by(actions),
                                       skip_types=frozenset({"comp", "mb_comp"}))
        replaces_comp = {a.stem_id for a in actions
                         if a.type == "comp"
                         and getattr(a, "time_range_s", None) is None}
        replaces_mb = {a.stem_id for a in actions if a.type == "mb_comp"}
        chain_holders = {stem_id: stem["chain"]
                         for stem_id, stem in self.stems.items()}
        chain_holders.update({bus_id: bus["chain"]  # Phase 23: same budget
                              for bus_id, bus in self.buses.items()})
        from .guardrails import level_charge_db
        for stem_id, chain in chain_holders.items():
            makeup = 0.0
            for entry in chain:
                if entry["type"] == "comp":
                    if entry.get("time_range_s") is not None:
                        continue             # one-sided: counted by _ranged_gain
                    if stem_id in replaces_comp:
                        continue
                    makeup += level_charge_db(entry)     # incl. a true M/S comp's makeup
                elif entry["type"] == "mb_comp" and stem_id not in replaces_mb:
                    makeup += level_charge_db(entry)     # incl. an upward band's boost
            if makeup:
                totals[stem_id] = totals.get(stem_id, 0.0) + makeup
        return totals

    @staticmethod
    def _is_eq_singleton(action: dict) -> bool:
        """A curve the composed-EQ rail charges whose entry REPLACES the last one of its type."""
        from .guardrails import EQ_CURVE_SINGLETONS
        return action.get("type") in EQ_CURVE_SINGLETONS and action.get("time_range_s") is None

    def _eq_ledger(self) -> dict:
        """The composed-EQ rail's input (design call 2026-09-11): the agent's APPLIED (not undone)
        eq / tilt / match_eq moves per target, this pass and this session, plus the trimmed
        records' folded curves. From the LEDGER, never the live chain: the chain carries the
        user's co-edits, and the bound is on the agent. Undo refunds, like the gain wallet.

        A singleton curve (the overdrive's tone) replaces itself, so it is not appended: per
        target and type, `singletons` holds the entry that stands (the last applied, else the
        trimmed carry's) and the one that stood when this pass began (`prior`)."""
        from .guardrails import EQ_CURVE_TYPES
        this_pass: dict[str, list] = {}
        session: dict[str, list] = {}
        singletons: dict[str, dict[str, dict]] = {}
        for sid, by_type in (self._state.get("trimmed_eq_singletons") or {}).items():
            for kind, action in by_type.items():
                singletons.setdefault(sid, {})[kind] = {"standing": action, "prior": action}
        for record in self._state["proposals"]:
            if record["status"] != "applied":
                continue
            for index, action in enumerate(record["actions"]):
                if action.get("type") in EQ_CURVE_TYPES and self._same_entity(record, action["stem_id"]):
                    counted = self._ledger_eq_action(record, index, action)   # D9·R5
                    if self._is_eq_singleton(action):
                        slot = singletons.setdefault(action["stem_id"], {}).setdefault(
                            action["type"], {"standing": None, "prior": None})
                        slot["standing"] = counted
                        if record["pass_number"] < self.pass_number:
                            slot["prior"] = counted
                        continue
                    session.setdefault(action["stem_id"], []).append(counted)
                    if record["pass_number"] == self.pass_number:
                        this_pass.setdefault(action["stem_id"], []).append(counted)
        return {"pass": this_pass, "session": session, "singletons": singletons,
                "trimmed": self._state.get("trimmed_eq_curves_db", {})}

    def _signal_parents(self) -> dict[str, Optional[str]]:
        """Every stem and bus -> the next target its signal goes through on the way to the
        master (None = the master): the group a stem is summed into, the track its Audio To
        feeds, the group a group plays into. The guardrails hold the gain and composed-EQ
        budgets along these paths as well as per target (reader D9·R7, guardrails._signal_path)."""
        parents: dict[str, Optional[str]] = {}
        for sid in self.stems:
            kind, dest = self.stem_output(sid)
            parents[sid] = dest if kind == "track" else None if kind else self.stem_bus(sid)
        for bid, bus in self.buses.items():
            out = bus.get("output")
            parents[bid] = out if out in self.buses and out != bid else None
        return parents

    def _chains_for_guardrails(self) -> dict[str, list[dict]]:
        """Raw chain entries per target (incl. master + group busses) — the
        comp-stacking rule validates against what is already applied."""
        chains = {stem_id: stem["chain"]
                  for stem_id, stem in self.stems.items()}
        chains.update({bus_id: bus["chain"]
                       for bus_id, bus in self.buses.items()})
        chains[MASTER] = self._state["master_chain"]
        return chains

    def _cumulative_gain(self, pass_number: Optional[int], *,
                         replaces: frozenset | set = frozenset(),
                         skip_types: frozenset = frozenset()) -> dict[str, float]:
        """Level applied (and not undone) per target — gain moves plus every device
        trim `_record_trim_charge` knows, in the scope of one pass or the session. WHOLE-FILE
        moves only: a time-ranged one is one-sided and lives in `_ranged_gain`.
        Singleton types count the LAST applied record per (target, type); `replaces`
        drops the pairs this proposal is about to overwrite.

        In the scope of a PASS a singleton is charged its DELTA against the value it replaced
        from an earlier pass (D9·R4): a utility at −6 replaced by one at +6 moved the level
        +12 in that pass and used to be charged +6. So a replaced pair contributes −prior
        here, and the proposal's own charge of the new value completes new − prior."""
        totals: dict[str, float] = {}
        if pass_number is None:   # session scope: include gain spent by trimmed records
            totals.update({k: float(v)
                           for k, v in self._state.get("trimmed_gain_totals", {}).items()})
        singleton_last: dict[tuple[str, str], float] = {}
        singleton_prior: dict[tuple[str, str], float] = {}   # its value when this pass began
        # What the trimmed records left as each singleton's LAST value (reader D8·R4): the
        # session's standing charge until a live record of the same pair replaces it, and - in
        # the scope of a pass - the value a replacement is a delta against.
        for sid, by_type in (self._state.get("trimmed_singleton_charges") or {}).items():
            for t, carried in by_type.items():
                if pass_number is not None:
                    singleton_prior[(sid, t)] = float(carried)
                elif t not in skip_types and (sid, t) not in replaces:
                    singleton_last[(sid, t)] = float(carried)
        for record in self._state["proposals"]:
            if record["status"] != "applied":
                continue
            earlier = pass_number is not None and record["pass_number"] < pass_number
            if pass_number is not None and record["pass_number"] != pass_number and not earlier:
                continue
            for index, action in enumerate(record["actions"]):
                t = action["type"]
                if t in skip_types or action.get("time_range_s") is not None:
                    continue
                charge = self._ledger_charge(record, index, self._record_trim_charge(action))
                if charge is None:
                    continue
                sid = action["stem_id"]
                if not self._same_entity(record, sid):
                    continue                              # spent on what held this id before (D8·R6)
                if earlier:
                    if self._is_singleton_record(action):
                        singleton_prior[(sid, t)] = charge
                elif self._is_singleton_record(action):
                    if (sid, t) in replaces:
                        continue
                    singleton_last[(sid, t)] = charge     # replace semantics
                elif charge:
                    totals[sid] = totals.get(sid, 0.0) + charge
        for (sid, t), charge in singleton_last.items():
            charge -= singleton_prior.get((sid, t), 0.0)
            if charge:
                totals[sid] = totals.get(sid, 0.0) + charge
        for (sid, t) in replaces:
            if (prior := singleton_prior.get((sid, t), 0.0)) and t not in skip_types:
                totals[sid] = totals.get(sid, 0.0) - prior
        return totals

    def set_baseline(self, proposal_id: str, baseline: dict[str, Optional[float]]) -> None:
        """Pre-apply values of the metrics this proposal predicted (impact.py)."""
        self._require_proposal(proposal_id)["baseline"] = baseline
        self._persist()

    def record_measured(self, proposal_id: str,
                        measurements: list[MetricMeasurement]) -> None:
        record = self._require_proposal(proposal_id)
        record["measured_impact"] = [m.model_dump() for m in measurements]
        self._persist()

    def applied_awaiting_measurement(self) -> list[dict]:
        """Applied proposals with a baseline but no measured impact yet."""
        return [r for r in self._state["proposals"]
                if r["status"] == "applied" and r.get("baseline")
                and r["measured_impact"] is None]

    def _stem_tags(self) -> dict[str, list[str]]:
        return {stem_id: stem["tags"] for stem_id, stem in self.stems.items()}

    def _accepted_proposals_this_pass(self, exclude: Optional[str] = None) -> int:
        return sum(1 for r in self._state["proposals"]
                   if r["status"] != "rejected" and r["pass_number"] == self.pass_number
                   and r["proposal_id"] != exclude)

    def _require_proposal(self, proposal_id: str) -> dict:
        for record in self._state["proposals"]:
            if record["proposal_id"] == proposal_id:
                return record
        raise KeyError(f"no such proposal: {proposal_id}")

    def _require_stem(self, stem_id: str) -> dict:
        if stem_id not in self.stems:
            raise KeyError(f"no such stem: {stem_id}; stems: {sorted(self.stems)}")
        return self.stems[stem_id]

    def _all_chains(self):
        """Every device chain in the session: the master's, each stem's, bus's and return's."""
        yield self._state.get("master_chain") or []
        for table in ("stems", "buses", "returns"):
            for record in (self._state.get(table) or {}).values():
                if isinstance(record, dict) and isinstance(record.get("chain"), list):
                    yield record["chain"]

    def _stamp_entries(self, carry: Optional[tuple] = None) -> None:
        """Give every chain entry that has none an identity (`uid`, an integer never reissued).

        An entry is addressed everywhere by (type, occurrence), and an occurrence is a position:
        it names another device as soon as one of that type is inserted ahead of it or leaves.
        The co-edit journal replays the user's dock edits after an undo has put a chain back
        WITHOUT the agent's devices, and replayed by occurrence it wrote them onto whichever
        device then held the number (full audit 2026-10-02, reader E2·R16). The journal names the
        entry by this instead (daw_bridge._coedit_who). An inverse snapshot is a deep copy, so a
        restored entry is still the one the journal named.

        ``carry`` = (type, uid) of the one entry an action just REPLACED (`_replace_in_chain`):
        the new entry of that type takes it over. A re-proposed compressor is the same device
        in the dock, and an edit the user made on it must find the one the undo puts back.
        Unique within a chain; a copied chain (`copy_stem_state`) repeats them across targets,
        which nothing reads."""
        serial = int(self._state.get("entry_serial") or 0)
        for chain in self._all_chains():
            seen = {e["uid"] for e in chain if isinstance(e, dict) and isinstance(e.get("uid"), int)}
            taken: set = set()
            for entry in chain:
                if not isinstance(entry, dict):
                    continue
                uid = entry.get("uid")
                if isinstance(uid, int) and not isinstance(uid, bool) and uid not in taken:
                    taken.add(uid)
                    continue
                if (carry is not None and entry.get("type") == carry[0]
                        and carry[1] not in seen and carry[1] not in taken):
                    uid, carry = carry[1], None
                else:
                    serial += 1
                    uid = serial
                entry["uid"] = uid
                taken.add(uid)
        if serial != int(self._state.get("entry_serial") or 0):
            self._state["entry_serial"] = serial

    def _persist(self) -> None:
        self._stamp_entries()
        if self._read_only:
            return
        tmp = self._session_path.with_suffix(".json.tmp")
        # Synced before the rename and the folder after (reader D8·R10): a rename of a file
        # nothing has flushed can survive a power cut as a NAME with no contents, and the
        # session has no fallback for a file that does not parse. A plain fsync (not
        # F_FULLFSYNC): 0.17 ms a save, both syncs included, on the build Mac (2026-10-07).
        # A volume that cannot sync (some network shares answer ENOTSUP) still gets the save,
        # as it did before the sync was there: the FILE's sync is forgiven like the folder's
        # (pre-merge review, 2026-10-07 - it raised, and every save on such a volume failed).
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(self._state, indent=2))
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        st = tmp.stat()   # a rename keeps inode + mtime: OUR write's stamp, with no race window
        os.replace(tmp, self._session_path)
        try:
            folder = os.open(self._session_path.parent, os.O_RDONLY)
            try:
                os.fsync(folder)
            finally:
                os.close(folder)
        except OSError:
            pass                                          # a volume that cannot sync a folder
        self._stamp = (st.st_ino, st.st_mtime_ns, st.st_size)   # not "another writer" (sync_from_disk)
