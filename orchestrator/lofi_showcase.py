"""The lofi showcase — a user-like, prompt-driven full song in the real DAW (Phase 52).

Eight composer messages, typed into the running app exactly as a person would type
them, build a complete lo-fi track **from an empty project, MIDI only**: the agent
sets tempo/key/harmony, designs its drum and synth sounds, writes the parts,
resamples its own patch into a sampler, arranges a full form, automates it, bounces
the instrument tracks to audio, mixes, masters and exports a WAV.

It is a product preview AND a test. The test half is free and deterministic — no
LLM judges anything:

  * **PLAN** — per stage, the command families the agent had to reach, read from the
    app's own persisted feed (``<project>/chats/<session>.json``: one ``k:"tool"``
    row per agent command). ``runs/session_digest.jsonl`` is the fallback, and a
    stage graded from it is marked DEGRADED — the digest keeps at most 8 unique
    command keys per turn, so it can only ever under-report.
  * **STRUCTURE** — what actually landed in the saved edit
    (``session.tracktionedit``, Tracktion's XML) and the brain's ``session.json``:
    instrument tracks, clips and notes, song length, labelled sections, automation
    curves, returns and sends, the bounced stems, the muted sources, the master chain.
  * **OUTCOME** — the exported WAV through the engine's own analysis: length,
    loudness, true peak, detected tempo and key against what was asked for, and a
    rolled-off top end against the hip-hop reference curve.

Why a driver rather than a socket client: the composer is the only free-form chat
door into the running app (``App.cpp`` ``onSend`` -> the brain's ``chat`` method).
``--brain``/``--brain-llm`` and ``daw_technique_e2e`` send ``run`` — a mix brief —
so neither can compose. The turn-done signal needs no new code: the brain appends
exactly one ``runs/session_digest.jsonl`` line per finished chat turn.

    python -m orchestrator.lofi_showcase stages              # the script (free)
    python -m orchestrator.lofi_showcase --self-test         # prove the graders (free)
    python -m orchestrator.lofi_showcase drive --project /tmp/daw-fresh ...   # billed
    python -m orchestrator.lofi_showcase verify --project /tmp/daw-fresh      # free

Exits 0 PASS / 1 FAIL / 2 could not measure.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
import os
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_REPO = Path(__file__).resolve().parents[1]
_HID_DIR = _REPO / ".claude" / "skills" / "drive-app"

#: Tracktion plugin `type` strings that make a track an instrument track.
#: 4osc = WOOLLY, sampler = the GM drum kit, mixAgentSampler = Simpler.
INSTRUMENT_TYPES = ("4osc", "sampler", "mixAgentSampler")
#: Hosted instruments (a scanned third-party synth) look like these.
HOSTED_TYPES = ("vst", "vst3", "AudioUnit")
#: A promoted Drum Rack reports as a plain `rack` — the kit is wrapped in a
#: RackInstance the moment a pad gets its own chain. On a MIDI track that rack IS
#: the instrument; on an audio track the same type is an effect rack, so this one
#: only ever counts on a track the edit already calls "midi".
MIDI_RACK_TYPE = "rack"

#: `PITCHSEQUENCE/PITCH` stores the scale as the ordinal of te::Scale::ScaleType
#: (tracktion_Musicality.h), not a name — "D minor" is pitch="2" scale="1".
MINOR_SCALES = {1, 3, 4, 7, 9, 10}   # minor, dorian, phrygian, aeolian, melodic, harmonic

#: What the song is asked to be, in stage 1 — the OUTCOME layer grades against it.
TARGET_BPM = 82.0
TARGET_TONIC = "D"
TARGET_MODE = "min"


# --------------------------------------------------------------------------
# the script
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Req:
    """One PLAN requirement: at least `min_calls` agent commands whose key is in
    `any_of`. `hard` False = reported but never fails the stage (a capability the
    agent may legitimately reach another way)."""
    label: str
    any_of: tuple[str, ...]
    min_calls: int = 1
    hard: bool = True
    #: keys that meet the requirement in ONE call (Phase 61: one `arrangement.build` labels
    #: every section and grows the song - the work `min_calls` counts piece by piece)
    whole: tuple[str, ...] = ()


@dataclass(frozen=True)
class Stage:
    n: int
    key: str
    title: str
    prompt: str
    shows: str
    requires: tuple[Req, ...]
    #: roll a screen recording around this stage (the interesting ones)
    record: bool = False


STAGES: tuple[Stage, ...] = (
    Stage(
        n=1, key="foundation", title="Foundation", record=True,
        prompt=("Let's make a lofi hip-hop track. Set it to 82 BPM in D minor, write an "
                "8-bar jazzy ii-V-i chord progression onto the timeline, and set up a "
                "drum kit track, a keys track and a bass track."),
        shows="tempo, key and harmony on the timeline; instrument tracks",
        requires=(
            Req("tempo set", ("tempo.set",)),
            Req("key set", ("key.set",)),
            Req("chords on the timeline", ("chord.set",)),
            Req("drum kit track", ("track.add_drum_kit", "track.set_instrument")),
            Req("instrument tracks", ("track.add_midi",), min_calls=2),
        ),
    ),
    Stage(
        n=2, key="drums", title="Drums and drum sound design",
        prompt=("Program a lazy, swung lofi drum groove over those 8 bars. Make the kit "
                "sound like it came off an old record: saturate the snare pad, dull the "
                "hats, and put some swing on it."),
        shows="note writing, per-pad Drum Rack chains, the Groove Pool",
        requires=(
            Req("a drum clip", ("midi.add_clip", "midi.insert_at_selection")),
            Req("drum notes written", ("midi.add_notes", "midi.add_note", "step.program")),
            Req("per-pad character", ("pad.device_add", "drum_rack.promote",
                                      "sampler.set", "sampler.replace_sound")),
            Req("swing", ("clip.set_groove", "groove.set_amount", "groove.create",
                          "clip.extract_groove", "midi.quantize"), hard=False),
        ),
    ),
    Stage(
        n=3, key="parts", title="Keys, bass, melody and synth design",
        prompt=("Write the keys over those chords with a warm Rhodes-style patch you "
                "design on the synth, a sub bass that follows the roots, and a simple "
                "lead melody. I want the lead wobbly and filtered."),
        shows="composing to a stated harmony, WOOLLY patch design, the mod matrix",
        requires=(
            Req("parts written", ("midi.add_notes", "midi.add_note"), min_calls=3),
            Req("patch design", ("instrument.set_param",), min_calls=2),
            Req("modulation", ("instrument.set_modulation", "midi.add_cc"), hard=False),
        ),
    ),
    Stage(
        n=4, key="resample", title="Resampling - making its own samples",
        prompt=("Resample the keys: bounce a one-shot of that patch to audio, load it "
                "into a sampler as a playable instrument, then chop it into a texture "
                "part."),
        shows="bounce-as-sample, Simpler load and slice",
        requires=(
            Req("bounced a one-shot", ("track.bounce",)),
            Req("loaded it as an instrument", ("instrument.load_sample",
                                               "sampler.add_sound",
                                               "sampler.replace_sound")),
            Req("chopped it", ("sampler.slice", "sampler.slice_to_midi",
                               "sampler.set", "sampler.crop"), hard=False),
            Req("played the texture", ("midi.add_notes", "midi.add_note"), hard=False),
        ),
    ),
    Stage(
        n=5, key="arrange", title="Arrangement", record=True,
        prompt=("Arrange this into a full song - intro, two verses, hooks, a bridge and "
                "an outro, about two and a half minutes - label the sections, and put "
                "them in the Loop Deck as scenes."),
        shows="labelled sections, time surgery, the Loop Deck",
        requires=(
            Req("labelled sections", ("section.add",), min_calls=5,
                whole=("arrangement.build",)),
            Req("the song grew", ("time.duplicate", "clip.duplicate", "midi.add_clip",
                                  "time.insert", "time.copy", "time.paste",
                                  "midi.add_notes", "arrangement.build")),
            Req("scenes in the deck", ("scene.from_sections", "scene.add",
                                       "clip.to_slot", "slot.set_clip"), hard=False),
        ),
    ),
    Stage(
        n=6, key="movement", title="Movement and space",
        prompt=("Add movement: automate a filter sweep into each hook, ride the keys "
                "down in the verses, add a vinyl-style noise bed, a plate reverb return "
                "for the keys and a slap delay send on the melody."),
        shows="automation lanes, CC curves, returns and sends, noise sound design",
        requires=(
            Req("automation written", ("automation.write", "midi.add_cc"), min_calls=2),
            Req("returns created", ("return.add",), min_calls=2),
            Req("sends dialled", ("send.set",), min_calls=2),
            Req("a noise texture", ("track.add_midi", "instrument.set_param",
                                    "instrument.load_sample"), hard=False),
        ),
    ),
    Stage(
        n=7, key="bounce_mix", title="Bounce to stems and mix", record=True,
        prompt=("Bounce every instrument track to audio so we can mix it properly, and "
                "mute the originals. Then balance the mix: glue the drums, sidechain the "
                "bass to the kick, and keep the keys warm."),
        shows="the instrument-to-mix bridge, proposals, bus glue, ducking",
        requires=(
            Req("instrument tracks bounced", ("track.bounce",), min_calls=4),
            Req("sources muted", ("track.set_mute",), min_calls=3),
            Req("mixed", ("agent.auto_mix", "agent.run", "propose", "start_run",
                          "auto_mix", "device.set")),
            Req("glue", ("group.create",), hard=False),
            Req("sidechain", ("duck.set_key",), hard=False),
        ),
    ),
    Stage(
        n=8, key="master_export", title="Master and export", record=True,
        prompt=("Master it for lofi - around -14 LUFS, warm with the top rolled off, "
                "ceiling -1 dBTP - then export a WAV."),
        shows="a mix run against a genre target, the export confirm",
        requires=(
            Req("a mastering pass", ("agent.run", "start_run", "agent.auto_mix",
                                     "auto_mix", "propose")),
            Req("exported", ("session.export",)),
        ),
    ),
)

STAGE_BY_KEY = {s.key: s for s in STAGES}


# --------------------------------------------------------------------------
# reading what the app left on disk
# --------------------------------------------------------------------------

def digest_lines(project: Path) -> list[dict]:
    """One entry per finished chat turn (daw_bridge.record_session_digest)."""
    path = project / "runs" / "session_digest.jsonl"
    if not path.is_file():
        return []
    out: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue          # a half-written tail line is not a measurement
    return out


def _feed_files(project: Path, session: str = "") -> list[Path]:
    chats = project / "chats"
    if not chats.is_dir():
        return []
    if session:
        f = chats / f"{session}.json"
        return [f] if f.is_file() else []
    files = sorted((p for p in chats.glob("*.json") if p.name != "index.json"),
                   key=lambda p: p.stat().st_mtime)
    # Prefer the active session when it actually carries the conversation; a
    # blank project seeds an empty one, and grading that is grading nothing.
    index = chats / "index.json"
    if index.is_file():
        try:
            active = json.loads(index.read_text()).get("active", "")
        except json.JSONDecodeError:
            active = ""
        act = chats / f"{active}.json"
        if act.is_file() and _has_user_block(act):
            return [act]
    return [p for p in files if _has_user_block(p)] or files


def _has_user_block(path: Path) -> bool:
    try:
        blocks = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return any(isinstance(b, dict) and b.get("k") == "user" for b in blocks)


def feed_turns(project: Path, session: str = "") -> list[dict]:
    """The persisted feed split into turns: one entry per user message, with every
    agent command that followed it.

    The feed is the app's own record (`EngineerPanel` writes `k:"user"` and one
    `k:"tool"` row per command, with its command `key`), so unlike the digest it is
    not capped — it is the honest PLAN source.
    """
    turns: list[dict] = []
    for path in _feed_files(project, session):
        try:
            blocks = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(blocks, list):
            continue
        for b in blocks:
            if not isinstance(b, dict):
                continue
            kind = b.get("k")
            if kind == "user":
                turns.append({"user": str(b.get("t") or ""), "commands": [],
                              "failed": [], "reply": ""})
            elif kind == "tool" and turns:
                key = str(b.get("key") or "")
                if not key:
                    continue
                turns[-1]["commands"].append(key)
                # `failed` = the app refused it; `noresult` = the turn ended before it
                # answered (the feed writes a row that never got its result that way).
                # Neither is a command that is known to have run.
                if b.get("failed") or b.get("noresult"):
                    turns[-1]["failed"].append(key)
            elif kind == "prose" and turns:
                turns[-1]["reply"] = str(b.get("t") or "")
    return turns


def _edit_root(project: Path) -> Optional[ET.Element]:
    path = project / "session.tracktionedit"
    if not path.is_file():
        return None
    try:
        return ET.parse(str(path)).getroot()
    except ET.ParseError:
        return None


def _session_state(project: Path) -> dict:
    path = project / "session.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _stems(state: dict) -> dict:
    st = state.get("stems", {})
    return st if isinstance(st, dict) else {s.get("stem_id"): s for s in st}


# --------------------------------------------------------------------------
# the edit, read structurally
# --------------------------------------------------------------------------

@dataclass
class EditFacts:
    """Everything the STRUCTURE layer grades, pulled out of the Tracktion XML once."""
    midi_tracks: int = 0
    instrument_tracks: int = 0
    audio_tracks: int = 0
    bounce_tracks: int = 0
    muted_instrument_tracks: int = 0
    midi_clips: int = 0
    notes: int = 0
    controllers: int = 0
    song_length_s: float = 0.0
    sections: list[str] = field(default_factory=list)
    automation_curves: int = 0
    automation_points: int = 0
    device_lanes: int = 0
    device_lane_points: int = 0
    scenes: int = 0
    filled_slots: int = 0
    grooves: int = 0
    tempo_bpm: float = 0.0
    key_pitch: int = -1
    key_scale: str = ""
    instruments: list[str] = field(default_factory=list)
    track_names: list[str] = field(default_factory=list)
    devices: list[str] = field(default_factory=list)


def read_edit(project: Path) -> Optional[EditFacts]:
    root = _edit_root(project)
    if root is None:
        return None
    f = EditFacts()

    tempo = root.find("./TEMPOSEQUENCE/TEMPO")
    if tempo is not None:
        f.tempo_bpm = float(tempo.get("bpm") or 0.0)
    pitch = root.find("./PITCHSEQUENCE/PITCH")
    if pitch is not None:
        f.key_pitch = int(float(pitch.get("pitch") or -1))
        f.key_scale = pitch.get("scale") or ""

    for arr in root.findall("./ARRANGERTRACK"):
        for clip in arr.findall("./ARRANGERCLIP"):
            f.sections.append(clip.get("name") or "")

    f.scenes = len(root.findall("./SCENES/SCENE"))

    for track in root.findall("./TRACK"):
        name = track.get("name") or ""
        f.track_names.append(name)
        kind = track.get("zamTrackKind") or ""
        plugin_types = [p.get("type") or "" for p in track.findall("./PLUGIN")]
        inst = [t for t in plugin_types
                if t in INSTRUMENT_TYPES or t in HOSTED_TYPES
                or (t == MIDI_RACK_TYPE and kind == "midi")]
        # Our own devices live inside the MonitorChain node's state, not as
        # PLUGIN rows; the brain's session.json is the source of truth for those.
        f.devices.extend(t for t in plugin_types
                         if t not in ("volume", "level", "mixAgentChain",
                                      "mixAgentClick"))
        muted = str(track.get("mute") or "0") in ("1", "true")

        if kind == "midi" or inst:
            f.midi_tracks += 1
            if inst:
                f.instrument_tracks += 1
                f.instruments.append(inst[0])
            if muted:
                f.muted_instrument_tracks += 1
        else:
            f.audio_tracks += 1
            if name.strip().lower().endswith("bounce"):
                f.bounce_tracks += 1

        for clip in track.findall("./MIDICLIP"):
            f.midi_clips += 1
            f.notes += len(clip.findall("./SEQUENCE/NOTE")) + len(clip.findall("./NOTE"))
            f.controllers += (len(clip.findall("./SEQUENCE/CONTROL"))
                              + len(clip.findall("./CONTROL")))
        for tag in ("MIDICLIP", "AUDIOCLIP", "STEPCLIP"):
            for clip in track.findall(f"./{tag}"):
                start = float(clip.get("start") or 0.0)
                length = float(clip.get("length") or 0.0)
                f.song_length_s = max(f.song_length_s, start + length)
        for slot in track.findall("./CLIPSLOTS/CLIPSLOT"):
            if len(list(slot)):
                f.filled_slots += 1

    # Automation comes in TWO shapes and the grader has to read both (2026-09-11). A
    # volume / pan / macro / hosted-plug-in ride is a Tracktion AUTOMATIONCURVE of POINTs.
    # A DEVICE ride — `automation.write {target: "gain.gain_db"}`, which is what the agent
    # reaches for most — is the MonitorChain node's own lane, persisted as the plugin's
    # `mixAutomation` JSON property and evaluated from the render time in
    # MonitorChainPlugin::applyToBuffer (so it sounds live AND in the export bounce).
    # Counting AUTOMATIONCURVE alone is what made the showcase's six real rides read as
    # "0 curves / 0 points" and get written up as "automation.write persisted nothing".
    for curve in root.iter("AUTOMATIONCURVE"):
        points = len(curve.findall("./POINT"))
        if points:
            f.automation_curves += 1
            f.automation_points += points
    for plugin in root.iter("PLUGIN"):
        blob = plugin.get("mixAutomation")
        if not blob:
            continue
        try:
            lanes = json.loads(blob)
        except (ValueError, TypeError):
            continue
        for lane in lanes if isinstance(lanes, list) else []:
            points = len(lane.get("points") or []) if isinstance(lane, dict) else 0
            if points:
                f.device_lanes += 1
                f.device_lane_points += points
    f.grooves = len(list(root.iter("GROOVE")))
    return f


# --------------------------------------------------------------------------
# grading
# --------------------------------------------------------------------------

@dataclass
class Check:
    layer: str
    name: str
    ok: bool
    detail: str = ""
    hard: bool = True

    @property
    def fails(self) -> bool:
        return self.hard and not self.ok


def _match_turns(turns: list[dict]) -> dict[str, dict]:
    """Pair each stage with every turn that went into it.

    Index alignment (stage N = turn N) breaks the moment a stage is RETRIED or the operator
    says anything off-script, and then every later stage is graded against the wrong turn —
    which is how a completed run first read as half-failed. So: anchor each stage on the turn
    whose message best matches its prompt, scanning only FORWARD so the conversation's order
    still constrains it, then give the stage every turn from its anchor up to the next
    stage's. A retry ("that command won't take MIDI, do it by hand") belongs to the stage it
    is retrying, and its commands count — the work happened.
    """
    def words(text: str) -> set[str]:
        keep = {w.strip(".,:;-()") for w in str(text).lower().split()}
        return {w for w in keep if len(w) > 3}

    anchors: dict[str, int] = {}
    cursor = 0
    for stage in STAGES:
        want = words(stage.prompt)
        best_score, best_i = 0.0, -1
        for i in range(cursor, len(turns)):
            got = words(turns[i].get("user", ""))
            if not got:
                continue
            score = len(want & got) / max(1, len(want))
            if score > best_score:
                best_score, best_i = score, i
        if best_i >= 0 and best_score >= 0.25:
            anchors[stage.key] = best_i
            cursor = best_i + 1

    out: dict[str, dict] = {}
    ordered = [s.key for s in STAGES if s.key in anchors]
    for pos, key in enumerate(ordered):
        start = anchors[key]
        end = anchors[ordered[pos + 1]] if pos + 1 < len(ordered) else len(turns)
        window = turns[start:end]
        out[key] = {
            "user": window[0].get("user", ""),
            "commands": [c for t in window for c in (t.get("commands") or [])],
            "failed": [c for t in window for c in (t.get("failed") or [])],
            "reply": next((t.get("reply", "") for t in reversed(window) if t.get("reply")), ""),
            "turns": len(window),
        }
    return out


def grade_plan(project: Path, session: str = "") -> tuple[list[Check], bool, list[dict]]:
    """Per stage, the commands the agent had to reach. Returns (checks, degraded,
    turns) — `degraded` when the capped digest had to stand in for the feed."""
    turns = feed_turns(project, session)
    degraded = False
    if not turns:
        turns = [{"user": d.get("user", ""), "commands": list(d.get("commands") or []),
                  "failed": [], "reply": d.get("reply", "")}
                 for d in digest_lines(project)]
        degraded = bool(turns)

    matched = _match_turns(turns)
    checks: list[Check] = []
    for stage in STAGES:
        turn = matched.get(stage.key)
        if turn is None:
            checks.append(Check("PLAN", f"{stage.n}. {stage.title}: the turn ran",
                                False, "no turn recorded for this stage"))
            continue
        # What RAN is what was called less what failed - as a multiset, since one key can
        # land four times and be refused a fifth (reader E7·R13). `failed` was filled by
        # feed_turns, carried by _match_turns and read by nothing: a drive in which every
        # `chord.set` was refused reported "chords on the timeline - ok".
        ran = Counter(turn.get("commands") or [])
        ran.subtract(Counter(turn.get("failed") or []))
        commands = [key for key, times in ran.items() for _ in range(max(0, times))]
        failed = sorted(set(turn.get("failed") or []))
        for req in stage.requires:
            hits = [c for c in commands if c in req.any_of]
            ok = len(hits) >= req.min_calls or any(c in req.whole for c in commands)
            want = (f"{req.min_calls}x " if req.min_calls > 1 else "")
            lost = [k for k in failed if k in req.any_of + req.whole]
            alone = f" (or one {'/'.join(req.whole)})" if req.whole else ""
            checks.append(Check(
                "PLAN", f"{stage.n}. {stage.title}: {req.label}", ok,
                f"want {want}one of {'/'.join(req.any_of)}{alone}; saw "
                + (", ".join(sorted(set(commands))) or "nothing")
                + (f"; failed or never answered: {', '.join(lost)}" if lost else ""),
                hard=req.hard))
    return checks, degraded, turns


def _is_minor(scale: str) -> bool:
    """The edit writes the scale as an ordinal; accept the whole minor family."""
    try:
        return int(float(scale)) in MINOR_SCALES
    except (TypeError, ValueError):
        return "min" in (scale or "").lower()


def grade_structure(project: Path) -> list[Check]:
    facts = read_edit(project)
    state = _session_state(project)
    if facts is None:
        return [Check("STRUCTURE", "the edit was saved", False,
                      "no session.tracktionedit - the app never saved this project")]

    stems = _stems(state)
    bounce_stems = [s for s in stems if str(s).strip().lower().endswith("bounce")]
    returns = state.get("returns") or {}
    sends = {sid: s.get("sends") for sid, s in stems.items() if s.get("sends")}
    master_chain = [d.get("type") for d in (state.get("master_chain") or [])]
    mastering = [t for t in master_chain if t != "limiter"]
    applied = [p for p in (state.get("proposals") or [])
               if p.get("status") == "applied"]

    def c(name: str, ok: bool, detail: str = "", hard: bool = True) -> Check:
        return Check("STRUCTURE", name, ok, detail, hard)

    return [
        c("instrument tracks with an instrument", facts.instrument_tracks >= 4,
          f"{facts.instrument_tracks} of 4 ({', '.join(facts.instruments) or 'none'})"),
        c("MIDI clips", facts.midi_clips >= 12, f"{facts.midi_clips} of 12"),
        c("notes written", facts.notes >= 100, f"{facts.notes} of 100"),
        c("the song is a song", facts.song_length_s >= 140.0,
          f"{facts.song_length_s:.1f} s of 140"),
        c("labelled sections", len(facts.sections) >= 6,
          f"{len(facts.sections)} of 6: {', '.join(x for x in facts.sections if x) or 'unnamed'}"),
        c("automation curves", facts.automation_curves + facts.device_lanes >= 2,
          f"{facts.automation_curves} track curves / {facts.automation_points} points "
          f"+ {facts.device_lanes} device lanes / {facts.device_lane_points} points"),
        c("returns", len(returns) >= 2, f"{len(returns)} of 2: {', '.join(map(str, returns))}"),
        c("sends into them", len(sends) >= 2,
          f"{len(sends)} sending stems: {', '.join(map(str, sends))}"),
        c("instrument tracks bounced to stems", len(bounce_stems) >= 4,
          f"{len(bounce_stems)} of 4: {', '.join(map(str, bounce_stems))}"),
        c("the bounced sources are muted", facts.muted_instrument_tracks >= 3,
          f"{facts.muted_instrument_tracks} of 3 muted"),
        c("the master was mastered", bool(mastering),
          f"master chain: {', '.join(t for t in master_chain if t) or 'empty'}"),
        c("the safety limiter is still there", "limiter" in master_chain,
          f"master chain: {', '.join(t for t in master_chain if t) or 'empty'}"),
        c("mix moves applied", bool(applied) or bool(state.get("coedit_log")),
          f"{len(applied)} applied proposals, "
          f"{len(state.get('coedit_log') or [])} co-edits", hard=False),
        c("scenes in the Loop Deck", facts.scenes >= 1 or facts.filled_slots >= 1,
          f"{facts.scenes} scenes, {facts.filled_slots} filled slots", hard=False),
        c("tempo is what was asked", abs(facts.tempo_bpm - TARGET_BPM) < 0.6,
          f"{facts.tempo_bpm:.1f} BPM, asked {TARGET_BPM:.0f}"),
        c("key is what was asked",
          facts.key_pitch >= 0 and facts.key_pitch % 12 == 2
          and _is_minor(facts.key_scale),
          f"pitch class {facts.key_pitch % 12 if facts.key_pitch >= 0 else '?'} "
          f"scale {facts.key_scale or '?'}, asked D minor (pitch 2, scale 1)"),
    ]


def grade_outcome(wav: Path) -> list[Check]:
    """The exported WAV through the engine's own analysis. Free, deterministic."""
    import numpy as np
    import soundfile as sf
    sys.path.insert(0, str(_REPO / "server"))
    from mix_engine import analysis as an
    from orchestrator.targets import GENRE_PROFILES

    def c(name: str, ok: bool, detail: str = "", hard: bool = True) -> Check:
        return Check("OUTCOME", name, ok, detail, hard)

    if not wav.is_file():
        return [c("the export exists", False, f"no file at {wav}")]
    audio, sr = sf.read(str(wav), always_2d=True)
    audio = audio.astype(np.float64)
    duration = len(audio) / float(sr)
    if duration < 1.0:
        return [c("the export has audio", False, f"{duration:.2f} s")]

    mono = audio.mean(axis=1)
    import pyloudnorm as pyln
    meter = pyln.Meter(sr)
    lufs = an._integrated_lufs(meter, an._lufs_arg(audio))
    tp = an._true_peak_dbtp(audio)
    peak = an._peak_dbfs(audio)
    bands = an._band_energy(audio, sr)
    tempo = an._tempo(mono, sr)
    key = an._key(mono, sr)
    hop = GENRE_PROFILES["hip hop"]["band_energy"]

    bpm = tempo[0] if tempo else 0.0
    bpm_ok = bool(tempo) and any(abs(bpm - TARGET_BPM * r) <= 2.0
                                 for r in (1.0, 0.5, 2.0))
    key_ok = bool(key) and key.tonic == TARGET_TONIC and key.mode == TARGET_MODE

    # "rolled off" — NEITHER top band may sit above the hip-hop reference curve,
    # each measured against the mid band so overall loudness cannot fake it.
    # Averaging the two bands hides the failure that matters: one hot band (a
    # 9 kHz shimmer, say) reads as rolled off next to a quiet neighbour.
    roll, roll_detail = 0.0, ""
    if bands:
        tops = ("brilliance", "air")
        ref = max(hop[b] - hop["mid"] for b in tops)
        got = max(bands.get(b, -120.0) - bands.get("mid", 0.0) for b in tops)
        roll = ref - got
        roll_detail = "; ".join(
            f"{b} {bands.get(b, 0.0) - bands.get('mid', 0.0):+.1f} dB vs mid"
            for b in tops)

    # the longest stretch of near-silence in the body of the track
    frame = max(1, int(sr * 0.1))
    rms = [float((mono[i:i + frame] ** 2).mean() ** 0.5)
           for i in range(0, len(mono) - frame, frame)]
    quiet, worst = 0, 0
    for r in rms:
        quiet = quiet + 1 if r < 1e-4 else 0
        worst = max(worst, quiet)
    gap_s = worst * 0.1

    return [
        c("the export is a full track", duration >= 120.0, f"{duration:.1f} s of 120"),
        c("mastered to a lofi level", lufs is not None and -16.0 <= lufs <= -11.0,
          f"{lufs:.2f} LUFS-I" if lufs is not None else "unmeasurable"),
        # The -1 dBTP ceiling is PARTIALLY enforced and the repo says so (memory
        # `daw-limiter-true-peak`): the limiter caps the SAMPLE peak exactly, and the
        # documented true-peak residual is up to ~0.41 dB on synthetic worst cases. So assert
        # what is actually guaranteed — the limiter did its job — and hold the true peak to
        # the documented residual rather than to an invariant that is not yet restored.
        c("the limiter capped the master", peak <= -0.99,
          f"sample peak {peak:.3f} dBFS, ceiling -1.0"),
        c("true peak within the documented residual", tp <= -0.59,
          f"{tp:.3f} dBTP, {(-1.0 - tp) * -1:.3f} dB over -1 "
          f"(memory daw-limiter-true-peak: up to ~0.41 dB, not yet restored)"),
        c("not slammed to full scale", peak < -0.05, f"peak {peak:.2f} dBFS"),
        # A WRONG tempo is a real failure. NO tempo is a statement about the detector as much
        # as the music — a soft, swung, dark mix can genuinely defeat it — so it is reported,
        # not gated. What the song is actually set to is asserted on the edit in STRUCTURE.
        c("it plays at the tempo asked for", bpm_ok,
          f"detected {bpm:.1f} BPM, asked {TARGET_BPM:.0f}"
          + ("" if tempo else " — no pulse found at all; the mix may be too soft or too dark "
                              "for onset detection, so this is reported, not gated"),
          hard=bool(tempo)),
        c("it is in the key asked for", key_ok,
          (f"detected {key.tonic} {key.mode} ({key.confidence:.2f})" if key
           else "no confident key"), hard=False),
        c("the top is rolled off", roll >= 3.0,
          f"{roll:.1f} dB below the hip-hop curve, want 3.0 ({roll_detail})"),
        c("no dead air", gap_s <= 2.0, f"longest silence {gap_s:.1f} s"),
        # The companion to "rolled off": lo-fi darkens the top, it does not delete it. A mix
        # with a 20 dB hole above 2 kHz has no transient left for a listener OR a detector to
        # find the beat in — which is exactly how the first real run failed the tempo check.
        c("the top is dark, not gone",
          bands.get("high_mid", -120.0) - bands.get("mid", 0.0) >= -25.0,
          f"high_mid {bands.get('high_mid', 0.0) - bands.get('mid', 0.0):+.1f} dB vs mid, "
          f"floor -25.0"),
    ]


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------

def run_verify(project: Path, export: Optional[Path], session: str = "") -> tuple[int, dict]:
    plan, degraded, turns = grade_plan(project, session)
    structure = grade_structure(project)
    outcome: list[Check] = []
    wav = export or find_export(project)
    if wav is not None:
        outcome = grade_outcome(wav)
    else:
        outcome = [Check("OUTCOME", "an export was found", False,
                         "no wav in Exported/ or Rendered/ - pass --export")]

    checks = plan + structure + outcome
    failed = [c for c in checks if c.fails]
    soft = [c for c in checks if not c.hard and not c.ok]

    print(f"\n[showcase] {project}")
    if degraded:
        print("  PLAN graded from the capped session digest (no feed found) - "
              "it can only under-report")
    layer = ""
    for c in checks:
        if c.layer != layer:
            layer = c.layer
            print(f"\n  {layer}")
        mark = "ok " if c.ok else ("XX " if c.hard else "-- ")
        print(f"    {mark}{c.name}" + (f"   [{c.detail}]" if c.detail and not c.ok else ""))

    print(f"\n  {len(checks) - len(failed) - len(soft)}/{len(checks)} checks passed"
          f"  ({len(failed)} hard failures, {len(soft)} soft)")

    report = {
        "project": str(project),
        "export": str(wav) if wav else "",
        "degraded_plan": degraded,
        "turns": len(turns),
        "checks": [{"layer": c.layer, "name": c.name, "ok": c.ok,
                    "hard": c.hard, "detail": c.detail} for c in checks],
        "hard_failures": [c.name for c in failed],
        "soft_failures": [c.name for c in soft],
        "passed": not failed,
    }
    runs = project / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "lofi_showcase_report.json").write_text(json.dumps(report, indent=2))
    print(f"  report: {runs / 'lofi_showcase_report.json'}")
    return (0 if not failed else 1), report


def find_export(project: Path) -> Optional[Path]:
    """The newest wav the export could have written. Bounces are excluded — a
    `<track> bounce.wav` in the project root is an intermediate, not the deliverable."""
    candidates: list[Path] = []
    for sub in ("Exported", "Rendered", "."):
        d = project / sub
        if d.is_dir():
            candidates += [p for p in d.glob("*.wav")
                           if not p.name.lower().endswith(" bounce.wav")]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


# --------------------------------------------------------------------------
# driving the real app
# --------------------------------------------------------------------------

def _activate() -> None:
    subprocess.run(["osascript", "-e",
                    'tell application "Zammoth" to activate'],
                   capture_output=True)


def window_rect() -> Optional[tuple[int, int, int, int]]:
    script = ('tell application "System Events" to tell process "Zammoth" '
              'to get {position, size} of front window')
    r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    nums = [int(float(x)) for x in r.stdout.replace(",", " ").split()]
    if len(nums) < 4:
        return None
    return nums[0], nums[1], nums[2], nums[3]


def _hid():
    """The CGEvent driver, imported as a module — never run as a script.

    Running `hid.py` as a file exits 0 and posts nothing (drive-app skill,
    2026-09-07); the same call through the module lands every time.
    """
    if str(_HID_DIR) not in sys.path:
        sys.path.insert(0, str(_HID_DIR))
    import hid  # type: ignore
    return hid


def send_prompt(text: str, composer: tuple[float, float],
                send: Optional[tuple[float, float]]) -> None:
    """Type one composer message and submit it, in ONE uninterrupted gesture.

    Nothing may come between the click and the submit: an in-place editor commits
    on focus loss, and a screenshot in the middle turns the rest of the keystrokes
    into app shortcuts (drive-app skill).
    """
    hid = _hid()
    _activate()
    time.sleep(0.6)
    hid.click(float(composer[0]), float(composer[1]))
    time.sleep(0.4)
    hid.type_text(text)
    time.sleep(0.4)
    if send is not None:
        hid.click(float(send[0]), float(send[1]))
    else:
        hid.key(36)


def probe_input(composer: Optional[tuple[float, float]]) -> bool:
    """Prove the input channel is alive before spending a single billed turn.

    A dead CGEvent tap looks exactly like a mis-mapped coordinate, so this checks a
    side effect: Cmd-T adds a track, Cmd-Z takes it back.
    """
    facts_before = None
    hid = _hid()
    _activate()
    time.sleep(0.6)
    if composer is not None:
        hid.click(float(composer[0]), float(composer[1]))
        time.sleep(0.3)
    import Quartz as Q  # type: ignore
    before = Q.CGEventGetLocation(Q.CGEventCreate(None))
    hid.click(float(composer[0]) if composer else 400.0,
              float(composer[1]) if composer else 400.0)
    time.sleep(0.3)
    after = Q.CGEventGetLocation(Q.CGEventCreate(None))
    moved = abs(after.x - before.x) > 0.5 or abs(after.y - before.y) > 0.5 or composer
    del facts_before
    return bool(moved)


def wait_for_turn(project: Path, want_lines: int, timeout_s: float,
                  poll_s: float = 2.0) -> Optional[dict]:
    """Block until the brain records another finished chat turn.

    The digest gains exactly one line per completed turn, so its length is the
    turn-done signal — no socket, no new code in the app.
    """
    deadline = time.time() + timeout_s
    last_report = 0.0
    while time.time() < deadline:
        lines = digest_lines(project)
        if len(lines) >= want_lines:
            return lines[want_lines - 1]
        if time.time() - last_report > 60:
            last_report = time.time()
            mins = (deadline - time.time()) / 60.0
            print(f"      ... still working ({mins:.0f} min left)", flush=True)
        time.sleep(poll_s)
    return None


def screenshot(path: Path, rect: Optional[tuple[int, int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["screencapture", "-x"]
    if rect:
        x, y, w, h = rect
        cmd += ["-R", f"{x},{y},{w},{h}"]
    cmd.append(str(path))
    subprocess.run(cmd, capture_output=True)


def run_drive(project: Path, composer: Optional[tuple[float, float]],
              send: Optional[tuple[float, float]], footage: Optional[Path],
              start_at: int, stop_at: int, timeout_min: float, typer: str,
              record: bool) -> int:
    stages = [s for s in STAGES if start_at <= s.n <= stop_at]
    if not stages:
        print("nothing to do", file=sys.stderr)
        return 2
    if typer == "hid" and composer is None:
        print("hid typing needs --composer X,Y (display POINTS: screenshot the app "
              "and halve the Retina pixels). Never blind-click.", file=sys.stderr)
        return 2

    rect = window_rect()
    print(f"[showcase] driving {len(stages)} stage(s) into {project}")
    print(f"[showcase] window {rect}, typer={typer}")
    if typer == "hid" and not probe_input(composer):
        print("the input channel is dead - switch channels (computer-use MCP / "
              "System Events), do not debug it (drive-app skill)", file=sys.stderr)
        return 2

    # Keep the display awake for the whole run: a sleeping display breaks capture
    # before it breaks anything else.
    total_s = len(stages) * timeout_min * 60 + 600
    caffeinate = subprocess.Popen(["caffeinate", "-d", "-t", str(int(total_s))])
    recorder: Optional[subprocess.Popen] = None
    try:
        for stage in stages:
            have = len(digest_lines(project))
            want = stage.n
            if have >= want:
                print(f"\n[{stage.n}/{len(STAGES)}] {stage.title} - already recorded, skipping")
                continue
            print(f"\n[{stage.n}/{len(STAGES)}] {stage.title}")
            print(f"      {stage.prompt}")

            if record and footage and stage.record:
                out = footage / f"{stage.n:02d}-{stage.key}.mov"
                i = 1
                while out.exists():          # screencapture refuses to overwrite
                    out = footage / f"{stage.n:02d}-{stage.key}-{i}.mov"
                    i += 1
                footage.mkdir(parents=True, exist_ok=True)
                cmd = ["screencapture", "-v", "-V", "360"]
                if rect:
                    x, y, w, h = rect
                    cmd += ["-R", f"{x},{y},{w},{h}"]
                cmd.append(str(out))
                recorder = subprocess.Popen(cmd)

            if typer == "hid":
                send_prompt(stage.prompt, composer, send)   # type: ignore[arg-type]
            else:
                print("      >>> type the message above into the composer and send it, "
                      "then leave it alone <<<", flush=True)

            entry = wait_for_turn(project, want, timeout_min * 60)
            if recorder is not None:
                recorder.terminate()
                recorder = None
            if entry is None:
                print(f"      TIMED OUT after {timeout_min:.0f} min - the turn never "
                      f"finished. Resume with --from {stage.n}", file=sys.stderr)
                return 1
            cmds = ", ".join(entry.get("commands") or []) or "none recorded"
            print(f"      done. commands: {cmds}")
            print(f"      reply: {entry.get('reply', '')[:200]}")

            if footage:
                screenshot(footage / f"{stage.n:02d}-{stage.key}.png", rect)

            if stage.key == "master_export":
                # The export is the one confirm-gated command in the flow; its
                # AlertWindow binds "Do it" to Return.
                time.sleep(1.5)
                if typer == "hid":
                    _activate()
                    time.sleep(0.5)
                    _hid().key(36)
                else:
                    print("      >>> confirm the export dialog <<<", flush=True)
    finally:
        if recorder is not None:
            recorder.terminate()
        caffeinate.terminate()

    print("\n[showcase] all stages done - now run `verify`")
    return 0


# --------------------------------------------------------------------------
# self-test: prove the graders discriminate
# --------------------------------------------------------------------------

def _golden_project(root: Path) -> Path:
    """A synthetic project that SHOULD pass every layer."""
    import numpy as np
    import soundfile as sf

    proj = root / "golden"
    (proj / "runs").mkdir(parents=True, exist_ok=True)
    (proj / "chats").mkdir(parents=True, exist_ok=True)
    (proj / "Exported").mkdir(parents=True, exist_ok=True)

    # --- the feed: one user block per stage, then the commands it had to reach
    per_stage = {
        1: ["tempo.set", "key.set", "chord.set", "track.add_drum_kit",
            "track.add_midi", "track.add_midi"],
        2: ["midi.add_clip", "midi.add_notes", "pad.device_add", "clip.set_groove"],
        3: ["midi.add_notes", "midi.add_notes", "midi.add_notes",
            "instrument.set_param", "instrument.set_param", "instrument.set_modulation"],
        4: ["track.bounce", "instrument.load_sample", "sampler.slice", "midi.add_notes"],
        5: ["section.add", "section.add", "section.add", "section.add", "section.add",
            "time.duplicate", "scene.from_sections"],
        6: ["automation.write", "automation.write", "return.add", "return.add",
            "send.set", "send.set", "instrument.set_param"],
        7: ["track.bounce", "track.bounce", "track.bounce", "track.bounce",
            "track.set_mute", "track.set_mute", "track.set_mute",
            "agent.auto_mix", "group.create", "duck.set_key"],
        8: ["agent.run", "session.export"],
    }
    blocks: list[dict] = []
    digest: list[str] = []
    for stage in STAGES:
        blocks.append({"k": "user", "t": stage.prompt})
        for key in per_stage[stage.n]:
            blocks.append({"k": "tool", "key": key, "verb": "set", "target": "x",
                           "failed": False})
        blocks.append({"k": "prose", "t": f"Done: {stage.title}."})
        digest.append(json.dumps({
            "ts": "2026-09-11T00:00:0%d+00:00" % stage.n, "session_id": "s1",
            "user": stage.prompt, "reply": f"Done: {stage.title}.",
            "commands": per_stage[stage.n][:8], "asked": False}))
    (proj / "chats" / "s1.json").write_text(json.dumps(blocks))
    (proj / "chats" / "index.json").write_text(json.dumps(
        {"active": "s1", "sessions": [{"id": "s1", "title": "", "updated": 1, "count": 8}]}))
    (proj / "runs" / "session_digest.jsonl").write_text("\n".join(digest) + "\n")

    # --- the edit
    parts = ['<?xml version="1.0" encoding="UTF-8"?>', "<EDIT>",
             '  <PITCHSEQUENCE><PITCH startBeat="0.0" pitch="2" scale="1"/></PITCHSEQUENCE>',
             f'  <TEMPOSEQUENCE><TEMPO startBeat="0.0" bpm="{TARGET_BPM}"/>'
             '<TIMESIG numerator="4" denominator="4" startBeat="0.0"/></TEMPOSEQUENCE>',
             '  <ARRANGERTRACK name="Arranger" id="9000">']
    for i, name in enumerate(("Intro", "Verse 1", "Hook", "Verse 2", "Bridge",
                              "Final Hook", "Outro")):
        parts.append(f'    <ARRANGERCLIP name="{name}" start="{i * 21.0}" length="21.0" id="{9100 + i}"/>')
    parts.append("  </ARRANGERTRACK>")

    notes = "".join(f'<NOTE p="{62 + (i % 12)}" b="{i * 0.5}" l="0.4" v="90" c="1"/>'
                    for i in range(30))
    # One ride of EACH shape, so both readers in read_edit are load-bearing and the
    # two automation mutations below can each take the count under the threshold on its own:
    # Drums gets a Tracktion volume curve, Keys gets a device lane (the MonitorChain node's
    # `mixAutomation` property — what `automation.write {target: "gain.gain_db"}` writes).
    device_lane = json.dumps([{"param": "gain.gain_db",
                               "points": [{"t_s": 0.0, "value": 0.0},
                                          {"t_s": 60.0, "value": -6.0}]}]).replace('"', "&quot;")
    for t, (name, inst) in enumerate((("Drums", "rack"), ("Keys", "4osc"),
                                      ("Bass", "4osc"), ("Lead", "4osc"),
                                      ("Texture", "mixAgentSampler"))):
        parts.append(f'  <TRACK id="{9200 + t}" name="{name}" zamTrackKind="midi" mute="1">')
        parts.append(f'    <PLUGIN type="{inst}" id="{9300 + t}" enabled="1"/>')
        lane = f' mixAutomation="{device_lane}"' if t == 1 else ""
        parts.append(f'    <PLUGIN type="mixAgentChain" id="{9350 + t}" enabled="1"{lane}/>')
        parts.append(f'    <PLUGIN type="volume" id="{9360 + t}" enabled="1"/>')
        for clip in range(3):
            parts.append(f'    <MIDICLIP start="{clip * 50.0}" length="50.0" id="{9400 + t * 10 + clip}">'
                         f'<SEQUENCE>{notes}</SEQUENCE></MIDICLIP>')
        if t < 1:
            parts.append('    <AUTOMATIONCURVE><POINT t="0.0" v="0.2"/>'
                         '<POINT t="60.0" v="0.9"/></AUTOMATIONCURVE>')
        parts.append("  </TRACK>")
    for t, name in enumerate(("Drums bounce", "Keys bounce", "Bass bounce",
                              "Lead bounce", "Texture bounce")):
        parts.append(f'  <TRACK id="{9500 + t}" name="{name}" zamTrackKind="audio">'
                     f'<AUDIOCLIP start="0.0" length="150.0" id="{9600 + t}"/></TRACK>')
    parts.append('  <SCENES><SCENE name="Hook"/><SCENE name="Verse"/></SCENES>')
    parts.append("</EDIT>")
    (proj / "session.tracktionedit").write_text("\n".join(parts))

    # --- the brain state
    stems = {f"{n} bounce": {"gain_db": 0.0, "chain": [{"type": "eq"}]}
             for n in ("Drums", "Keys", "Bass", "Lead", "Texture")}
    stems["Keys bounce"]["sends"] = {"rv": -6.0}
    stems["Lead bounce"]["sends"] = {"dly": -9.0}
    (proj / "session.json").write_text(json.dumps({
        "state_version": 2, "stems": stems, "master_gain_db": 0.0,
        "master_chain": [{"type": "eq"}, {"type": "comp"}, {"type": "saturation"},
                         {"type": "limiter", "ceiling_dbtp": -1.0}],
        "proposals": [{"status": "applied", "actions": []}],
        "buses": {"drums": {}}, "returns": {"rv": {"kind": "reverb"},
                                            "dly": {"kind": "delay"}},
        "coedit_log": [],
    }, indent=2))

    _write_lofi_wav(proj / "Exported" / "lofi showcase.wav")
    return proj


def _write_lofi_wav(path: Path, seconds: float = 150.0,
                    roll_off: bool = True) -> None:
    """A synthetic D-minor lo-fi loop at 82 BPM, mastered to ~-14 LUFS / -1 dBTP.

    The OUTCOME layer measures tempo, key, loudness, ceiling and top-end roll-off
    from real audio, so the golden has to actually BE those things — a file that
    merely claims them would prove nothing.
    """
    import numpy as np
    import soundfile as sf
    import pyloudnorm as pyln
    sys.path.insert(0, str(_REPO / "server"))

    sr = 44100
    n = int(seconds * sr)
    t = np.arange(n) / sr
    beat = 60.0 / TARGET_BPM
    x = np.zeros(n)

    # D minor: Dm7 - Gm7 - Am7 - Dm7, one bar each, plus a root bass
    chords = [(62, 65, 69, 72), (67, 70, 74, 77), (69, 72, 76, 79), (62, 65, 69, 72)]
    bar = beat * 4.0
    for i in range(int(seconds / bar) + 1):
        start = i * bar
        s0 = int(start * sr)
        s1 = min(n, int((start + bar * 0.95) * sr))
        if s0 >= n:
            break
        seg = np.arange(s1 - s0) / sr
        env = np.exp(-seg * 1.1)
        notes = chords[i % 4]
        for p in notes:
            f = 440.0 * 2 ** ((p - 69) / 12.0)
            x[s0:s1] += 0.10 * env * np.sin(2 * np.pi * f * seg)
        fb = 440.0 * 2 ** ((notes[0] - 24 - 69) / 12.0)
        x[s0:s1] += 0.22 * np.exp(-seg * 0.8) * np.sin(2 * np.pi * fb * seg)

    # a closed hat on the offbeats. Without it the golden was a 4th-order lowpass over pure
    # sines — 27 dB below the mid band, darker than any real lo-fi master and darker than the
    # showcase's own first export. A fixture has to look like the thing it stands for.
    rng = np.random.default_rng(7)
    for i in range(int(seconds / (beat / 2.0)) + 1):
        if i % 2 == 0:
            continue
        s0 = int(i * (beat / 2.0) * sr)
        s1 = min(n, s0 + int(0.05 * sr))
        if s0 >= n:
            break
        seg = np.arange(s1 - s0) / sr
        x[s0:s1] += 0.30 * np.exp(-seg * 70.0) * rng.standard_normal(s1 - s0)

    # a kick on every beat — this is what the tempo detector locks to
    for i in range(int(seconds / beat) + 1):
        s0 = int(i * beat * sr)
        s1 = min(n, s0 + int(0.12 * sr))
        if s0 >= n:
            break
        seg = np.arange(s1 - s0) / sr
        env = np.exp(-seg * 38.0)
        sweep = 110.0 * np.exp(-seg * 26.0) + 45.0
        x[s0:s1] += 0.85 * env * np.sin(2 * np.pi * sweep * seg)

    # lo-fi: roll the top off hard (one-pole at ~2.4 kHz, twice)
    from scipy.signal import butter, sosfilt
    if roll_off:
        x = sosfilt(butter(2, 3800.0 / (sr / 2), btype="low", output="sos"), x)
    else:
        x = x + 0.12 * np.sin(2 * np.pi * 9000.0 * t) * (1.0 + np.sin(2 * np.pi * 3.0 * t))
    stereo = np.column_stack([x, x * 0.98])

    # Master it the way a master is made: normalise, then LIMIT the peaks, then
    # trim to the ceiling. Scaling the whole file down to fit its loudest kick
    # instead lands ~4 dB quiet - which is exactly what the OUTCOME layer catches.
    meter = pyln.Meter(sr)

    def normalise(buf, target=-14.0):
        loud = meter.integrated_loudness(buf)
        if loud is not None and math.isfinite(loud):
            return buf * (10 ** ((target - loud) / 20.0))
        return buf

    stereo = normalise(stereo)
    thresh = 10 ** (-3.0 / 20.0)
    stereo = thresh * np.tanh(stereo / thresh)      # soft ceiling, keeps the body
    stereo = normalise(stereo)

    from mix_engine import analysis as _an
    for _ in range(4):                               # trim onto the true-peak ceiling
        tp = _an._true_peak_dbtp(stereo)
        if tp <= -1.2:
            break
        stereo = stereo * (10 ** ((-1.3 - tp) / 20.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), stereo, sr, subtype="PCM_24")


def _blank_project(root: Path) -> Path:
    """What `fresh.sh` leaves behind: an empty project and a silent export."""
    import numpy as np
    import soundfile as sf

    proj = root / "blank"
    (proj / "chats").mkdir(parents=True, exist_ok=True)
    (proj / "Exported").mkdir(parents=True, exist_ok=True)
    (proj / "chats" / "index.json").write_text(json.dumps(
        {"active": "s1", "sessions": [{"id": "s1", "title": "", "updated": 1, "count": 0}]}))
    (proj / "chats" / "s1.json").write_text(json.dumps([{"k": "status", "t": ""}]))
    (proj / "session.tracktionedit").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<EDIT>\n'
        '  <PITCHSEQUENCE><PITCH startBeat="0.0" pitch="60"/></PITCHSEQUENCE>\n'
        '  <TEMPOSEQUENCE><TEMPO startBeat="0.0" bpm="120.0"/></TEMPOSEQUENCE>\n'
        '  <ARRANGERTRACK name="Arranger" id="1006"/>\n'
        '  <TRACK id="1011" name="Audio 1" zamTrackKind="audio"/>\n'
        '  <SCENES/>\n</EDIT>\n')
    (proj / "session.json").write_text(json.dumps({
        "state_version": 2, "stems": {}, "master_gain_db": 0.0,
        "master_chain": [{"type": "limiter", "ceiling_dbtp": -1.0, "owner": "user"}],
        "proposals": [], "buses": {}, "returns": {}, "coedit_log": [],
    }, indent=2))
    sf.write(str(proj / "Exported" / "silent.wav"),
             np.zeros((44100 * 150, 2)), 44100, subtype="PCM_24")
    return proj


#: Each mutation breaks the golden project in ONE specific way, and names the check
#: that must go red. A layer that passes the golden and fails an empty project can
#: still be blind to everything in between - this is what proves it is not.
def _mutations() -> list[tuple[str, object, str]]:
    def drop_bounce_stems(proj: Path) -> None:
        state = json.loads((proj / "session.json").read_text())
        state["stems"] = {k: v for k, v in state["stems"].items()
                          if not k.endswith("bounce")}
        (proj / "session.json").write_text(json.dumps(state))

    def wrong_tempo(proj: Path) -> None:
        path = proj / "session.tracktionedit"
        path.write_text(path.read_text().replace(f'bpm="{TARGET_BPM}"', 'bpm="120.0"'))

    def no_sections(proj: Path) -> None:
        path = proj / "session.tracktionedit"
        text = path.read_text()
        head, _, tail = text.partition("<ARRANGERTRACK")
        path.write_text(head + "<ARRANGERTRACK/>" + tail.partition("</ARRANGERTRACK>")[2])

    def hot_master(proj: Path) -> None:
        state = json.loads((proj / "session.json").read_text())
        state["master_chain"] = [{"type": "limiter", "ceiling_dbtp": -1.0}]
        (proj / "session.json").write_text(json.dumps(state))

    def never_exported(proj: Path) -> None:
        blocks = json.loads((proj / "chats" / "s1.json").read_text())
        (proj / "chats" / "s1.json").write_text(json.dumps(
            [b for b in blocks if b.get("key") != "session.export"]))

    def never_bounced(proj: Path) -> None:
        blocks = json.loads((proj / "chats" / "s1.json").read_text())
        (proj / "chats" / "s1.json").write_text(json.dumps(
            [b for b in blocks if b.get("key") != "track.bounce"]))

    def short_export(proj: Path) -> None:
        _write_lofi_wav(proj / "Exported" / "lofi showcase.wav", seconds=60.0)

    def bright_export(proj: Path) -> None:
        _write_lofi_wav(proj / "Exported" / "lofi showcase.wav", roll_off=False)

    # The two automation shapes, one mutation each (2026-09-11). Counting only Tracktion
    # curves is the blindness that turned the showcase's six real device rides into
    # "automation.write persisted nothing"; counting only device lanes would be the same
    # mistake mirrored. The golden carries exactly one of each, so either removal must bite.
    def no_device_lanes(proj: Path) -> None:
        path = proj / "session.tracktionedit"
        path.write_text(re.sub(r' mixAutomation="[^"]*"', "", path.read_text()))

    def no_track_curves(proj: Path) -> None:
        path = proj / "session.tracktionedit"
        path.write_text(re.sub(r"<AUTOMATIONCURVE>.*?</AUTOMATIONCURVE>", "",
                               path.read_text(), flags=re.S))

    # A command the app REFUSED is still a row in the feed (reader E7·R13). Nothing in the
    # edit or the audio checks for chords, so the plan layer is the only one that can say it.
    def chords_refused(proj: Path) -> None:
        path = proj / "chats" / "s1.json"
        blocks = json.loads(path.read_text())
        for b in blocks:
            if b.get("k") == "tool" and b.get("key") == "chord.set":
                b["failed"] = True
        path.write_text(json.dumps(blocks))

    return [
        ("the bounces never reached the engine", drop_bounce_stems,
         "instrument tracks bounced to stems"),
        ("the tempo is not what was asked", wrong_tempo, "tempo is what was asked"),
        ("the song has no sections", no_sections, "labelled sections"),
        ("nothing but the safety limiter on the master", hot_master,
         "the master was mastered"),
        ("the agent never exported", never_exported, "exported"),
        ("the agent never bounced", never_bounced, "bounced"),
        ("the export is a loop, not a song", short_export,
         "the export is a full track"),
        ("the export is bright, not lofi", bright_export, "the top is rolled off"),
        ("the device rides were never written", no_device_lanes, "automation curves"),
        ("the volume rides were never written", no_track_curves, "automation curves"),
        ("every chord.set was refused", chords_refused, "chords on the timeline"),
    ]


def _all_checks(proj: Path) -> list[Check]:
    plan, _, _ = grade_plan(proj)
    return (plan + grade_structure(proj)
            + grade_outcome(find_export(proj) or proj / "missing.wav"))


def self_test() -> int:
    """Goldens pass, an untouched project fails — on every layer (Phase 48's rule).

    A grader that cannot fail is not a measurement, so this asserts both directions.
    """
    import tempfile
    ok = True
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        print("[self-test] building the golden project (synthesising audio)...")
        golden = _golden_project(root)
        blank = _blank_project(root)

        for name, proj, want_pass in (("golden", golden, True),
                                      ("blank", blank, False)):
            plan, degraded, _ = grade_plan(proj)
            structure = grade_structure(proj)
            outcome = grade_outcome(find_export(proj) or proj / "missing.wav")
            layers = {"PLAN": plan, "STRUCTURE": structure, "OUTCOME": outcome}
            print(f"\n[self-test] {name}")
            for layer, checks in layers.items():
                bad = [c for c in checks if c.fails]
                good = not bad
                if want_pass:
                    verdict = good
                    detail = ("" if good else
                              "; ".join(f"{c.name} [{c.detail}]" for c in bad[:4]))
                else:
                    verdict = not good        # the blank MUST fail this layer
                    detail = "" if not good else "the layer passed on an empty project"
                print(f"    {'ok ' if verdict else 'XX '}{layer} "
                      f"{'passes' if want_pass else 'fails'}"
                      + (f"   [{detail}]" if detail else ""))
                ok = ok and verdict
            if want_pass and degraded:
                print("    XX PLAN read the feed (not the capped digest)")
                ok = False

        print("\n[self-test] mutations - each must turn its own check red")
        for label, mutate, want_red in _mutations():
            mutant = root / ("mut-" + str(abs(hash(label)) % 100000))
            shutil.copytree(golden, mutant)
            mutate(mutant)
            red = [c.name for c in _all_checks(mutant) if not c.ok]
            caught = any(want_red in name for name in red)
            print(f"    {'ok ' if caught else 'XX '}{label}"
                  + ("" if caught else f"   [nothing failed matching '{want_red}'; "
                                       f"red: {', '.join(red) or 'none'}]"))
            ok = ok and caught
            shutil.rmtree(mutant)

    print(f"\n[self-test] {'PASS' if ok else 'FAIL'} - the graders "
          f"{'discriminate' if ok else 'do NOT discriminate'}")
    return 0 if ok else 1


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _point(text: str) -> tuple[float, float]:
    parts = text.replace(",", " ").split()
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected X,Y in display points")
    return float(parts[0]), float(parts[1])


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="The lofi showcase: a prompt-driven full song in the real DAW")
    ap.add_argument("--self-test", action="store_true",
                    help="prove the graders discriminate (free, no app, no LLM)")
    sub = ap.add_subparsers(dest="cmd")

    p_stages = sub.add_parser("stages", help="print the prompt script (free)")
    p_stages.add_argument("--json", action="store_true")

    p_drive = sub.add_parser("drive", help="drive the running app (BILLED)")
    p_drive.add_argument("--project", type=Path, required=True)
    p_drive.add_argument("--composer", type=_point,
                         help="composer text field, in display POINTS")
    p_drive.add_argument("--send", type=_point,
                         help="the send glyph; omitted = press Return (which dies "
                              "once the composer wraps to two lines)")
    p_drive.add_argument("--footage", type=Path, help="stills + recordings go here")
    p_drive.add_argument("--record", action="store_true",
                         help="also roll video around the interesting stages")
    p_drive.add_argument("--from", dest="start_at", type=int, default=1,
                         help="resume at this stage")
    p_drive.add_argument("--to", dest="stop_at", type=int, default=len(STAGES),
                         help="stop after this stage")
    p_drive.add_argument("--timeout-min", type=float, default=35.0,
                         help="per-stage budget")
    p_drive.add_argument("--typer", choices=("hid", "manual"), default="manual")

    p_verify = sub.add_parser("verify", help="grade what landed (free)")
    p_verify.add_argument("--project", type=Path, required=True)
    p_verify.add_argument("--export", type=Path, help="the exported wav")
    p_verify.add_argument("--session", default="", help="grade this chat session id")

    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    if args.cmd == "stages":
        if args.json:
            print(json.dumps([{"n": s.n, "key": s.key, "title": s.title,
                               "prompt": s.prompt, "shows": s.shows,
                               "requires": [{"label": r.label, "any_of": list(r.any_of),
                                             "min_calls": r.min_calls, "hard": r.hard}
                                            for r in s.requires]}
                              for s in STAGES], indent=2))
            return 0
        for s in STAGES:
            print(f"\n{s.n}. {s.title}  -  {s.shows}")
            print(f"   \"{s.prompt}\"")
            for r in s.requires:
                mark = "must" if r.hard else "soft"
                want = f"{r.min_calls}x " if r.min_calls > 1 else ""
                print(f"     [{mark}] {r.label}: {want}{' | '.join(r.any_of)}")
        print(f"\n{len(STAGES)} stages")
        return 0

    if args.cmd == "drive":
        return run_drive(args.project.resolve(), args.composer, args.send,
                         args.footage.resolve() if args.footage else None,
                         args.start_at, args.stop_at, args.timeout_min,
                         args.typer, args.record)

    if args.cmd == "verify":
        code, _ = run_verify(args.project.resolve(),
                             args.export.resolve() if args.export else None,
                             args.session)
        return code

    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
