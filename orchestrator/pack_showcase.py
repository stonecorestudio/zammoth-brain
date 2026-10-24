"""The sample-pack showcase: seven typed messages take an empty project to a pack (Phase 60).

    python -m orchestrator.pack_showcase stages                  # the prompts, and what each is graded on
    python -m orchestrator.pack_showcase verify --project <dir>  # grade a finished run - free
    python -m orchestrator.pack_showcase --self-test             # the graders, proven - free (needs the console)

A person (or `drive-app`'s `prompt_send.py`) types the seven prompts into the REAL app, one a
turn; the engineer designs the drums, the low end, three melodic one-shots, two loops and a
playable instrument out of the synth, keeps them in a pack, puts the drums on a kit with a
beat, and finishes the pack. `verify` then grades what is on the disk, with no model and no
judge, on four layers:

  COMMANDS  what the feed says ran in each turn (and did not fail)
  FILES     the pack: its manifest, the kinds and counts asked for, every file there with the
            patch that made it, the loops' tempo and key, the instrument's zones, "finished"
  AUDIO     every file read by `sample_profile` HERE against its kind (`sound_classes`), every
            loop exact to the sample with a clean seam
  PROJECT   the saved edit: a kit whose pads play the pack's files, and a beat on it

Running the prompts on a model is BILLED and is James's to start. `--self-test` builds the
golden by playing the scripted right answer through the chat rig (the real app, the real
brain, the real commands), requires `verify` to pass it and to fail an untouched project, and
then breaks the golden one way at a time - a file gone, a kick replaced by a plain saw, a loop
a hundred samples short, the finish undone, a kind missing, the adds marked failed, the beat
erased - each of which must turn its own named check red.

It says where every sound sits against its kind. Whether the pack sounds GOOD is an ear's.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from orchestrator import chat_rig as R
from orchestrator import lofi_showcase, sound_eval

REPO = Path(__file__).resolve().parents[1]
PACK = "Night Shift"
BPM = 120
KEY = "A minor"
EXIT_NOT_BUILT = 2
REPORT = "pack_showcase_report.json"


@dataclass(frozen=True)
class Req:
    label: str
    any_of: tuple[str, ...]
    min_calls: int = 1


@dataclass(frozen=True)
class Stage:
    n: int
    key: str
    title: str
    prompt: str
    requires: tuple[Req, ...]
    shows: str


STAGES: tuple[Stage, ...] = (
    Stage(1, "drums", "The drums",
          f"Let's build a sample pack from scratch with the synth - call it {PACK}. Start with the "
          "drums: a deep kick, a punchy kick, a snare, a clap, a closed hat and an open hat.",
          (Req("the pack is made", ("pack.create",)),
           Req("the sounds are heard before they are kept", ("sound.render",)),
           Req("the sounds are kept", ("pack.add",))),
          "six drum one-shots designed by measuring, each read against its kind"),
    Stage(2, "low_end", "The low end",
          "Now the low end: an 808 that rings and a clean sub, both on C.",
          (Req("heard", ("sound.render",)), Req("kept", ("pack.add",))),
          "two tuned bass one-shots, named with the note they were played on"),
    Stage(3, "melodic", "Melodic one-shots",
          "Add three melodic one-shots on C: a pluck, a slow pad and a bell-like key sound.",
          (Req("heard", ("sound.render",)), Req("kept", ("pack.add",))),
          "a pad's reverb tail and a pluck's echoes ring out in the kept files"),
    Stage(4, "loops", "Loops",
          f"Give me two loops at {BPM} BPM in {KEY}: a four-beat bass line and an eight-beat pluck arpeggio.",
          (Req("kept", ("pack.add",)),),
          "loops exact to the sample, their key and tempo in the file and its name"),
    Stage(5, "instrument", "An instrument",
          "Turn the pad into a playable instrument across three octaves.",
          (Req("the multisample", ("pack.multisample",)),),
          "one patch rendered across the keyboard with one gain"),
    Stage(6, "kit", "A kit and a beat",
          "Put the pack's drums on a kit in this project and write a two-bar beat with them.",
          (Req("the kit", ("pack.build_kit",)),
           Req("a clip for the beat", ("midi.add_clip", "midi.insert_at_selection", "step.program", "step.set_cell")),
           Req("the notes", ("midi.add_notes", "step.program", "step.set_cell"))),
          "the pack plays in the project at once, balanced against its kick"),
    Stage(7, "finish", "Finish",
          "Finish the pack, save the project and tell me what is in it.",
          (Req("the pack is checked and finished", ("pack.finish",)),
           Req("the project is saved", ("session.save",))),
          "the pack verified against its own manifest"),
)

#: What the pack must hold at the end: kind -> how many one-shots of it, at least.
ONE_SHOTS = {"kick": 2, "snare": 1, "clap": 1, "hat": 1, "open_hat": 1, "boom": 1, "sub": 1,
             "pluck": 1, "pad": 1, "keys": 1}
LOOPS = {"bass": 4.0, "pluck": 8.0}          # kind -> beats
ON_C = ("boom", "sub", "pluck", "pad", "keys")
DRUM_KINDS = ("kick", "snare", "clap", "hat", "open_hat")
MIN_ZONES, MIN_SPAN = 6, 24
MIN_BEAT_NOTES, MIN_PADS_ON_KIT = 8, 5


@dataclass
class Check:
    layer: str
    label: str
    ok: bool
    detail: str = ""

    def line(self) -> str:
        return f"  {'ok ' if self.ok else 'XX '} [{self.layer}] {self.label}" + (f" - {self.detail}" if self.detail else "")


# ---------------------------------------------------------------- COMMANDS

def match_turns(turns: list[dict]) -> dict[int, dict]:
    """Stage number -> the turn whose user message is that stage's prompt (scanning forward:
    a prompt typed twice is graded on its last turn)."""
    out: dict[int, dict] = {}
    for turn in turns:
        said = " ".join(str(turn.get("user") or "").split())
        best, score = None, 0.0
        for stage in STAGES:
            ratio = difflib.SequenceMatcher(None, said.lower(), stage.prompt.lower()).ratio()
            if ratio > score:
                best, score = stage, ratio
        if best is not None and score >= 0.6:
            out[best.n] = turn
    return out


def grade_commands(turns: list[dict]) -> list[Check]:
    checks: list[Check] = []
    by_stage = match_turns(turns)
    for stage in STAGES:
        turn = by_stage.get(stage.n)
        if turn is None:
            checks.append(Check("commands", f"{stage.n} {stage.title}: the prompt was sent", False,
                                "no turn in the feed carries it"))
            continue
        ran = list(turn.get("commands") or [])
        for failed in turn.get("failed") or []:           # a multiset: one failure cancels one call
            if failed in ran:
                ran.remove(failed)
        for req in stage.requires:
            got = sum(1 for k in ran if k in req.any_of)
            checks.append(Check("commands", f"{stage.n} {stage.title}: {req.label}", got >= req.min_calls,
                                "" if got >= req.min_calls else
                                f"none of {', '.join(req.any_of)} ran (ran: {', '.join(ran) or 'nothing'})"))
    return checks


# ---------------------------------------------------------------- FILES

def find_pack(root: Path, name: str = PACK) -> Optional[dict]:
    return sound_eval.read_manifest(root, name)


def _is_c(note) -> bool:
    return isinstance(note, int) and note % 12 == 0


def grade_files(manifest: Optional[dict]) -> list[Check]:
    if manifest is None:
        return [Check("files", f"the pack {PACK} exists", False, "no folder of that name with a manifest")]
    folder = Path(manifest["_folder"])
    rows = [r for r in manifest.get("sounds") or [] if isinstance(r, dict)]
    shots = [r for r in rows if not r.get("loop")]
    loops = [r for r in rows if r.get("loop")]
    checks = [Check("files", f"the pack {PACK} exists", True, f"{len(shots)} one-shots, {len(loops)} loops")]
    for kind, want in ONE_SHOTS.items():
        have = sum(1 for r in shots if r.get("kind") == kind)
        checks.append(Check("files", f"one-shots: {want} x {kind}", have >= want, "" if have >= want else f"it holds {have}"))
    missing = [str(r.get("file")) for r in rows if not (folder / str(r.get("file") or "")).is_file()]
    checks.append(Check("files", "every sound's file is in the pack", not missing,
                        "" if not missing else f"not there: {', '.join(missing[:4])}"))
    no_patch = [str(r.get("name")) for r in rows
                if not r.get("preset") or not (folder / str(r["preset"])).is_file()]
    checks.append(Check("files", "every sound has the patch that made it beside it", not no_patch,
                        "" if not no_patch else f"none for: {', '.join(no_patch[:4])}"))
    off_c = [f"{r.get('name')} ({r.get('root')})" for r in shots if r.get("kind") in ON_C and not _is_c(r.get("root_note"))]
    checks.append(Check("files", "the tuned one-shots are on C", not off_c, "" if not off_c else ", ".join(off_c)))
    for kind, beats in LOOPS.items():
        mine = [r for r in loops if r.get("kind") == kind]
        fine = [r for r in mine if abs(float(r.get("beats") or 0) - beats) < 1e-6
                and abs(float(r.get("bpm") or 0) - BPM) < 1e-6
                and re.sub(r"[^a-z#]", "", str(r.get("key") or "").lower()) in ("am", "amin", "aminor")]
        checks.append(Check("files", f"a {beats:g}-beat {kind} loop at {BPM} BPM in {KEY}", bool(fine),
                            "" if fine else (f"the {kind} loops are: " + "; ".join(
                                f"{r.get('beats')} beats at {r.get('bpm')} in {r.get('key')}" for r in mine) if mine
                                             else f"no {kind} loop")))
    instruments = [i for i in manifest.get("instruments") or [] if isinstance(i, dict)]
    spans = []
    for inst in instruments:
        roots = [z.get("root_note") for z in inst.get("zones") or [] if isinstance(z.get("root_note"), int)]
        if roots:
            spans.append((len(roots), max(roots) - min(roots), inst))
    good = [s for s in spans if s[0] >= MIN_ZONES and s[1] >= MIN_SPAN]
    checks.append(Check("files", f"an instrument of {MIN_ZONES}+ zones across {MIN_SPAN}+ semitones", bool(good),
                        "" if good else (f"instruments: {[(n, span) for n, span, _ in spans]}" if spans else "the pack holds no instrument")))
    gone = [str(z.get("file")) for _, _, inst in spans for z in inst.get("zones") or []
            if not (folder / str(z.get("file") or "")).is_file()]
    checks.append(Check("files", "every zone's file is in the pack", not gone and bool(spans),
                        "" if (not gone and spans) else (f"not there: {', '.join(gone[:3])}" if gone else "no zones")))
    checks.append(Check("files", "the pack is finished", bool(manifest.get("finished")),
                        "" if manifest.get("finished") else "its manifest carries no finished stamp"))
    return checks


# ---------------------------------------------------------------- AUDIO

def grade_audio(manifest: Optional[dict]) -> list[Check]:
    if manifest is None:
        return [Check("audio", "the pack's sounds can be measured", False, "there is no pack")]
    _, _, sound_classes = sound_eval._engine()
    folder = Path(manifest["_folder"])
    checks: list[Check] = []
    measured = 0
    for row in [r for r in manifest.get("sounds") or [] if isinstance(r, dict)]:
        path = folder / str(row.get("file") or "")
        label = f"{row.get('kind')} '{row.get('name')}'"
        if not path.is_file():
            continue                                        # the FILES layer names it
        try:
            profile = sound_eval.profile_of(path, row)
        except Exception as exc:  # noqa: BLE001 - a file that cannot be read is a failure of its own
            checks.append(Check("audio", f"{label} can be read", False, f"{type(exc).__name__}: {exc}"))
            continue
        measured += 1
        kind = str(row.get("kind") or "")
        if kind not in sound_classes.CLASSES:
            checks.append(Check("audio", f"{label} is a known kind", False, f"kind {kind!r}"))
            continue
        played = row.get("root_note") if isinstance(row.get("root_note"), int) else None
        said = [i["says"] for i in sound_classes.issues(profile, kind, played)]
        what = "loops exactly, with a clean seam" if row.get("loop") else "sits where its kind does"
        checks.append(Check("audio", f"{label} {what}", not said, "; ".join(said[:2])))
        peak = profile.get("true_peak_dbtp")
        if isinstance(peak, (int, float)) and not row.get("loop"):
            checks.append(Check("audio", f"{label} is levelled", -2.0 <= peak <= -0.5, "" if -2.0 <= peak <= -0.5 else f"its peak is {peak} dBTP"))
    for inst in [i for i in manifest.get("instruments") or [] if isinstance(i, dict)]:
        pitches = []
        for zone in inst.get("zones") or []:
            path = folder / str(zone.get("file") or "")
            if not path.is_file() or not isinstance(zone.get("root_note"), int):
                continue
            try:
                profile = sound_eval.profile_of(path, {})
            except Exception:  # noqa: BLE001
                continue
            measured += 1
            midi = profile.get("pitch_midi")
            if midi is None:
                pitches.append(f"{zone.get('root')}: no pitch")
            else:
                off = (float(midi) + float(profile.get("pitch_cents") or 0.0) / 100.0) - zone["root_note"]
                if abs(off - 12.0 * round(off / 12.0)) > 0.7:
                    pitches.append(f"{zone.get('root')}: sounds {off:+.1f} st from its root")
        checks.append(Check("audio", f"instrument '{inst.get('name')}': every zone sounds its root", not pitches,
                            "; ".join(pitches[:3])))
    checks.insert(0, Check("audio", "the pack's sounds can be measured", measured > 0, f"{measured} files read"))
    return checks


# ---------------------------------------------------------------- PROJECT

def grade_project(project: Path, manifest: Optional[dict]) -> list[Check]:
    edit = project / "session.tracktionedit"
    if not edit.is_file():
        return [Check("project", "the project was saved", False, f"no {edit.name} in {project}")]
    text = edit.read_text(errors="replace")
    checks = [Check("project", "the project was saved", True)]
    pads = set()
    if manifest is not None:
        for row in manifest.get("sounds") or []:
            name = Path(str(row.get("file") or "")).name
            if name and row.get("kind") in DRUM_KINDS and name in text:
                pads.add(name)
    checks.append(Check("project", f"a kit in the project plays {MIN_PADS_ON_KIT}+ of the pack's drums", len(pads) >= MIN_PADS_ON_KIT,
                        f"{len(pads)} of the pack's drum files are named in the edit"))
    facts = lofi_showcase.read_edit(project)
    notes = facts.notes if facts is not None else 0
    checks.append(Check("project", f"a beat of {MIN_BEAT_NOTES}+ notes", notes >= MIN_BEAT_NOTES, f"{notes} notes in the edit"))
    return checks


# ---------------------------------------------------------------- verify

def verify(project: Path, packs_root: Path, turns: Optional[list[dict]] = None, quiet: bool = False) -> tuple[bool, list[Check]]:
    manifest = find_pack(packs_root)
    feed = turns if turns is not None else lofi_showcase.feed_turns(project)
    checks = grade_commands(feed) + grade_files(manifest) + grade_audio(manifest) + grade_project(project, manifest)
    ok = all(c.ok for c in checks)
    if not quiet:
        for c in checks:
            print(c.line())
        failed = sum(1 for c in checks if not c.ok)
        print(f"pack_showcase verify: {len(checks) - failed} of {len(checks)} checks hold - " + ("PASS" if ok else "FAIL"))
    return ok, checks


def default_packs_root() -> Path:
    return Path.home().joinpath(*sound_eval.PACKS_UNDER_HOME)


# ---------------------------------------------------------------- the scripted right answer

def _n(pitch: int, start: float, length: float, velocity: int = 100) -> dict:
    return {"pitch": pitch, "start_beat": start, "length_beats": length, "velocity": velocity}


def _shot(name: str, kind: str, preset: str, pitch: int, beats: float) -> dict:
    return {"name": name, "kind": kind, "preset": preset, "bpm": BPM, "notes": [_n(pitch, 0, beats)]}


_DRUMS = [_shot("Deep", "kick", "KICK DEEP", 60, 0.25), _shot("Punch", "kick", "KICK PUNCH", 60, 0.25),
          _shot("Crack", "snare", "SNARE TIGHT", 60, 0.25), _shot("Flutter", "clap", "CLAP", 60, 0.25),
          _shot("Tick", "hat", "HAT CLOSED", 60, 0.25), _shot("Ring", "open_hat", "HAT OPEN", 60, 0.25)]
_LOW = [_shot("Long", "boom", "BOOM BASS", 36, 2), _shot("Pure", "sub", "DEEP SINE", 36, 2)]
_MELODIC = [_shot("Glint", "pluck", "ARP PLUCK", 72, 0.25), _shot("Slow", "pad", "WARM PAD", 60, 8),
            _shot("Chime", "keys", "GLASS BELLS", 72, 2)]
_BASS_LINE = [_n(33, 0, 0.75), _n(33, 1, 0.5), _n(36, 1.5, 0.5), _n(40, 2, 0.75), _n(38, 3, 0.5), _n(36, 3.5, 0.5)]
_ARP = [_n(p, i * 0.5, 0.25) for i, p in enumerate((57, 60, 64, 69, 72, 69, 64, 60, 57, 60, 64, 69, 76, 72, 69, 64))]
_LOOPS = [{"name": "Walk", "kind": "bass", "preset": "SQUARE BASS", "bpm": BPM, "loop_beats": 4, "key": KEY, "notes": _BASS_LINE},
          {"name": "Climb", "kind": "pluck", "preset": "ARP PLUCK", "bpm": BPM, "loop_beats": 8, "key": KEY, "notes": _ARP}]
_BEAT = ([_n(36, b, 0.25) for b in (0, 2.5, 4, 6.5)] + [_n(38, b, 0.25) for b in (1, 3, 5, 7)]
         + [_n(39, 7.5, 0.25)] + [_n(42, b * 0.5, 0.2, 80) for b in range(16) if b != 7] + [_n(46, 3.5, 0.4, 90)])


def _hear(sounds: list[dict]) -> list[dict]:
    return [{k: v for k, v in s.items() if k not in ("key",)} for s in sounds]


def golden_turns() -> list[R.Turn]:
    tapes = [
        [R.ask(R.call("s1a", "pack.create", name=PACK, description="drums, bass and keys from the synth")),
         R.ask(R.call("s1b", "sound.render", sounds=_hear(_DRUMS))),
         R.ask(R.call("s1c", "pack.add", pack=PACK, sounds=_DRUMS)),
         R.say("Six drums are in the pack.")],
        [R.ask(R.call("s2a", "sound.render", sounds=_hear(_LOW))),
         R.ask(R.call("s2b", "pack.add", pack=PACK, sounds=_LOW)),
         R.say("An 808 and a sub, both on C.")],
        [R.ask(R.call("s3a", "sound.render", sounds=_hear(_MELODIC))),
         R.ask(R.call("s3b", "pack.add", pack=PACK, sounds=_MELODIC)),
         R.say("A pluck, a pad and a chime.")],
        [R.ask(R.call("s4a", "pack.add", pack=PACK, sounds=_LOOPS)),
         R.say("Two loops in A minor at 120.")],
        [R.ask(R.call("s5a", "pack.multisample", pack=PACK, name="Slow Pad", preset="WARM PAD",
                      lo_note=48, hi_note=84, step_semitones=4, length_beats=4, bpm=BPM)),
         R.say("The pad plays from C2 to C5.")],
        [R.ask(R.call("s6a", "pack.build_kit", pack=PACK, name="Night Kit")),
         R.ask(R.call("s6b", "midi.add_clip", track="Night Kit", start_s=0, length_s=4)),
         R.ask(R.call("s6c", "midi.add_notes", track="Night Kit", clip=0, notes=_BEAT)),
         R.say("The kit is on Night Kit with a two-bar beat.")],
        [R.ask(R.call("s7a", "pack.finish", pack=PACK)),
         R.ask(R.call("s7b", "session.save")),
         R.say("Night Shift holds eleven one-shots, two loops and an instrument.")],
    ]
    return [R.Turn(stage.prompt, tape) for stage, tape in zip(STAGES, tapes)]


def null_turns() -> list[R.Turn]:
    return [R.Turn(stage.prompt, [R.say("Done.")]) for stage in STAGES]


def rig_feed(o: R.Outcome) -> list[dict]:
    """The rig's own record of each turn, in the shape `lofi_showcase.feed_turns` gives."""
    out = []
    for i, stage in enumerate(STAGES):
        if i >= len(o.turns):
            break
        rows = o.rows(i)
        out.append({"user": stage.prompt, "commands": [str(r.get("key") or "") for r in rows],
                    "failed": [str(r.get("key") or "") for r in rows if r.get("failed") or r.get("noresult")]})
    return out


# ---------------------------------------------------------------- mutations

def _mutations() -> list[tuple[str, Callable[[Path, Path, list[dict]], None], str]]:
    """(what was broken, how, a word of the check that must turn red). Each gets a COPY of the
    golden: the packs root, the project folder, and the feed's turns."""
    def manifest_of(root: Path) -> tuple[Path, dict]:
        m = find_pack(root)
        assert m is not None
        return Path(m["_folder"]) / "pack.json", m

    def a_file_gone(root: Path, project: Path, feed: list[dict]) -> None:
        path, m = manifest_of(root)
        row = next(r for r in m["sounds"] if r["kind"] == "snare")
        (path.parent / row["file"]).unlink()

    def kick_is_a_saw(root: Path, project: Path, feed: list[dict]) -> None:
        import numpy as np
        import soundfile as sf
        path, m = manifest_of(root)
        row = next(r for r in m["sounds"] if r["kind"] == "kick")
        t = np.arange(int(0.4 * 44100)) / 44100.0
        saw = 0.6 * (2.0 * ((t * 261.6) % 1.0) - 1.0) * np.minimum(1.0, (0.4 - t) / 0.05)
        sf.write(str(path.parent / row["file"]), saw.astype("float32"), 44100, subtype="PCM_24")

    def loop_short(root: Path, project: Path, feed: list[dict]) -> None:
        import soundfile as sf
        path, m = manifest_of(root)
        row = next(r for r in m["sounds"] if r.get("loop"))
        audio, rate = sf.read(str(path.parent / row["file"]), always_2d=True)
        sf.write(str(path.parent / row["file"]), audio[:-100], rate, subtype="PCM_24")

    def unfinished(root: Path, project: Path, feed: list[dict]) -> None:
        path, m = manifest_of(root)
        doc = json.loads(path.read_text())
        doc.pop("finished", None)
        path.write_text(json.dumps(doc))

    def no_clap(root: Path, project: Path, feed: list[dict]) -> None:
        path, m = manifest_of(root)
        doc = json.loads(path.read_text())
        doc["sounds"] = [r for r in doc["sounds"] if r.get("kind") != "clap"]
        path.write_text(json.dumps(doc))

    def adds_failed(root: Path, project: Path, feed: list[dict]) -> None:
        for turn in feed:
            turn["failed"] = list(turn.get("failed") or []) + [k for k in turn.get("commands") or [] if k == "pack.add"]

    def beat_erased(root: Path, project: Path, feed: list[dict]) -> None:
        edit = project / "session.tracktionedit"
        edit.write_text(re.sub(r"<NOTE\b[^>]*/>", "", edit.read_text(errors="replace")))

    def off_c(root: Path, project: Path, feed: list[dict]) -> None:
        path, m = manifest_of(root)
        doc = json.loads(path.read_text())
        for r in doc["sounds"]:
            if r.get("kind") == "sub":
                r["root_note"], r["root"] = 41, "F1"
        path.write_text(json.dumps(doc))

    return [("a snare's file deleted", a_file_gone, "every sound's file is in the pack"),
            ("a kick replaced by a plain saw", kick_is_a_saw, "sits where its kind does"),
            ("a loop cut a hundred samples short", loop_short, "loops exactly"),
            ("the finished stamp removed", unfinished, "the pack is finished"),
            ("the clap taken out of the manifest", no_clap, "1 x clap"),
            ("every pack.add marked failed", adds_failed, "kept"),
            ("the beat's notes erased from the edit", beat_erased, "a beat of"),
            ("the sub filed on F", off_c, "on C")]


# ---------------------------------------------------------------- the self-test

def self_test(keep: Optional[Path] = None) -> int:
    if not R.BINARY.exists():
        print(f"pack_showcase: the console is not built ({R.BINARY})")
        return EXIT_NOT_BUILT
    base = keep or Path(tempfile.mkdtemp(prefix="pack-showcase-"))
    wrong: list[str] = []

    def play(name: str, turns: list[R.Turn]) -> tuple[R.Outcome, Path, Path]:
        work = base / name
        bad, o = R.run_scenario(R.Scenario(f"pack_showcase_{name}", name, turns, lambda _o: [], ceiling_s=180.0), work)
        if name == "golden":
            wrong.extend(f"[golden] the rig: {b}" for b in bad)
        return o, sound_eval.packs_root(work / R.HOME_MARKER), o.project

    o, root, project = play("golden", golden_turns())
    feed = rig_feed(o)
    saved_feed = lofi_showcase.feed_turns(project)
    print(f"golden: the pack is in {root}")
    ok, checks = verify(project, root, feed)
    if not ok:
        wrong.append("the GOLDEN answer does not pass: " + "; ".join(c.label + (f" ({c.detail})" if c.detail else "")
                                                                     for c in checks if not c.ok)[:600])
    if saved_feed:
        ok_saved, saved_checks = verify(project, root, None, quiet=True)
        if not ok_saved:
            wrong.append("graded from the project's own saved feed the golden does not pass: "
                         + "; ".join(c.label for c in saved_checks if not c.ok)[:300])
    else:
        print("  (the rig's project kept no chats/ transcript: the commands were graded from the rig's rows)")

    o0, root0, project0 = play("null", null_turns())
    _, blank = verify(project0, root0, rig_feed(o0), quiet=True)
    for layer in ("commands", "files", "audio", "project"):
        if all(c.ok for c in blank if c.layer == layer):
            wrong.append(f"an agent that did nothing passes the {layer} layer")

    if not wrong:
        for label, mutate, red in _mutations():
            box = base / "mutants" / re.sub(r"\W+", "_", label)
            shutil.rmtree(box, ignore_errors=True)
            shutil.copytree(root, box / "packs")
            shutil.copytree(project, box / "project")
            turns = json.loads(json.dumps(feed))
            mutate(box / "packs", box / "project", turns)
            _, got = verify(box / "project", box / "packs", turns, quiet=True)
            hit = [c for c in got if not c.ok and red.lower() in c.label.lower()]
            print(f"  {'ok ' if hit else 'XX '} mutation: {label} -> " + (hit[0].label if hit else "NOT CAUGHT"))
            if not hit:
                wrong.append(f"mutation not caught - {label}: no failed check names '{red}' "
                             f"(failed: {[c.label for c in got if not c.ok][:4]})")
    for line in wrong:
        print("  XX " + line)
    print("pack_showcase --self-test: " + ("the graders hold" if not wrong else f"{len(wrong)} problems"))
    if keep is None:
        shutil.rmtree(base, ignore_errors=True)
    return 1 if wrong else 0


def main(argv: Optional[list[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if "--self-test" in argv:
        keep = argv[argv.index("--keep") + 1] if "--keep" in argv else None
        return self_test(Path(keep).resolve() if keep else None)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stages", help="the seven prompts and what each is graded on")
    s.add_argument("--json", action="store_true")
    s.add_argument("--prompts", action="store_true", help="only the prompts, one a line")
    v = sub.add_parser("verify", help="grade a finished run")
    v.add_argument("--project", required=True)
    v.add_argument("--packs-root", help="where the packs are (default: ~/Music/Zammoth/Sample Packs)")
    args = parser.parse_args(argv)
    if args.cmd == "stages":
        if args.json:
            print(json.dumps([{"n": st.n, "key": st.key, "title": st.title, "prompt": st.prompt,
                               "requires": [{"label": r.label, "any_of": r.any_of} for r in st.requires],
                               "shows": st.shows} for st in STAGES], indent=1))
        elif args.prompts:
            for st in STAGES:
                print(st.prompt)
        else:
            for st in STAGES:
                print(f"{st.n}. {st.title}\n   > {st.prompt}\n   shows: {st.shows}")
                for r in st.requires:
                    print(f"   - {r.label}: {' | '.join(r.any_of)}")
            print(f"\nAt the end the pack {PACK} holds: " + ", ".join(f"{n} x {k}" for k, n in ONE_SHOTS.items())
                  + f"; loops: {', '.join(f'{b:g}-beat {k}' for k, b in LOOPS.items())} at {BPM} BPM in {KEY}; "
                  f"an instrument of {MIN_ZONES}+ zones; a kit and a beat in the saved project.")
        return 0
    project = Path(args.project).expanduser().resolve()
    root = Path(args.packs_root).expanduser().resolve() if args.packs_root else default_packs_root()
    ok, checks = verify(project, root)
    try:
        (project / "runs").mkdir(exist_ok=True)
        (project / "runs" / REPORT).write_text(json.dumps(
            {"ok": ok, "checks": [c.__dict__ for c in checks]}, indent=1) + "\n")
    except OSError:
        pass
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
