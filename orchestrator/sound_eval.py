"""The sound-design eval: does the engineer design a sound WELL (Phase 60).

Fourteen briefs in a user's words - "a deep, long kick for a slow house track", "a tight closed
hi-hat", "a slow, warm pad" - each ending in a sound kept in a sample pack. Every case is
graded, with no judge model, on two layers:

  PLAN     what the feed says ran: the sound was HEARD (`sound.render`) before it was kept
           (`pack.add`), nothing was removed, and the project was left alone - the request
           was for a sample.
  OUTCOME  the END STATE: the finished file in the pack, read by `sample_profile` here (not
           the figures the model was shown) against its kind (`sound_classes`) and against
           what the brief itself asked for (deep, short, wide, on E ...).

It runs through the chat rig: the real `DawComponent`, the real source brain, the real
commands, a private engine's renders, the pack on a sandboxed disk.

    python -m orchestrator.sound_eval --list
    python -m orchestrator.sound_eval --self-test [--record]        # free: needs the console built
    python -m orchestrator.sound_eval --live api --billed --repeat 3 --check    # BILLED

`--self-test` proves the graders on scripted answers, with no model: each case's GOLDEN answer
passes both layers; the same answer with the INIT preset in place of the designed sound fails
the outcome; an agent that says "Done." and does nothing fails both; and each case's MUTANT -
the golden with one thing wrong - fails the outcome on the figure the case names. `--record`
writes what each measured as to `sound_eval_goldens.json`, which a pytest re-grades without
the app.

What it does not say: whether a sound is GOOD. A sound that passes sits where its kind usually
sits and where the brief pointed; the rest is an ear's.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from orchestrator import chat_rig as R

REPO = Path(__file__).resolve().parents[1]
GOLDENS = Path(__file__).with_name("sound_eval_goldens.json")
PACKS_UNDER_HOME = ("Music", "Zammoth", "Sample Packs")
BPM = 120
MAX_SWEEP_ERRORS = 2
EXIT_NOT_BUILT = 2

#: Commands that take something away. None is asked for by any brief here.
REMOVALS = ("pack.remove", "track.delete", "clip.delete", "bus.remove", "return.remove",
            "instrument.preset_delete", "scene.delete", "marker.delete")
#: Commands that only look, or that work outside the project: they may sit beside the reach.
HARMLESS = ("sound.render", "pack.create", "pack.add", "pack.list", "pack.read", "pack.finish",
            "instrument.preset_list", "project.peek")


def _engine():
    server = str(REPO / "server")
    if server not in sys.path:
        sys.path.insert(0, server)
    import soundfile as sf
    from mix_engine import sound_classes
    from mix_engine.sample_profile import sample_profile
    return sf, sample_profile, sound_classes


# ---------------------------------------------------------------- what a brief asks for

@dataclass(frozen=True)
class Ask:
    """One thing the BRIEF asked of the sound, beyond its kind: a figure of the kept file's
    profile inside `lo`..`hi` (either open), or - `pitch_class` - the note its body sits on."""
    field: str
    lo: Optional[float] = None
    hi: Optional[float] = None
    says: str = ""
    pitch_class: Optional[int] = None

    def failure(self, profile: dict, sound_classes) -> Optional[str]:
        if self.pitch_class is not None:
            midi, cents = profile.get("pitch_midi"), profile.get("pitch_cents") or 0.0
            if midi is None:
                return f"{self.says}: no clear pitch could be read"
            exact = float(midi) + float(cents) / 100.0
            off = abs((exact - self.pitch_class + 6.0) % 12.0 - 6.0)
            return None if off <= 0.5 else f"{self.says}: its body reads as note {exact:.2f}"
        value = sound_classes.figure(profile, self.field)
        if value is None:
            # a figure that was never reached is over any ceiling, and under no floor
            if self.field == "decay_ms" and self.hi is not None:
                return f"{self.says}: it never fell 20 dB in the file"
            return None if self.field == "decay_ms" else f"{self.says}: {self.field} could not be read"
        if (self.lo is not None and value < self.lo) or (self.hi is not None and value > self.hi):
            bounds = (f"{self.lo:g}..{self.hi:g}" if self.lo is not None and self.hi is not None
                      else f"at least {self.lo:g}" if self.lo is not None else f"at most {self.hi:g}")
            return f"{self.says}: {self.field} is {value:g} (asked: {bounds})"
        return None


def note(pitch: int, beats: float, velocity: int = 100) -> list[dict]:
    return [{"pitch": pitch, "start_beat": 0, "length_beats": beats, "velocity": velocity}]


@dataclass(frozen=True)
class SoundCase:
    key: str
    kind: str                         # the pack kind the sound is to be kept as
    brief: str                        # the user's words: what is wanted, and the pack's name
    pack: str
    asks: tuple[Ask, ...]
    golden: dict                      # the scripted right answer: a sound in sound.render's grammar
    mutant: dict                      # ...with one thing wrong
    mutant_turns_red: str             # a word of the failure the mutant must draw
    name: str = "Take"                # what the scripted answers call the sound

    def prompt(self) -> str:
        return f"{self.brief} Keep it in a sample pack called {self.pack}."


def _over(base: dict, **patch) -> dict:
    """`base` with a patch laid over it (a preset keeps its name; the patch is merged)."""
    out = dict(base)
    merged = {k: dict(v) for k, v in (base.get("patch") or {}).items()}
    for part, values in patch.items():
        merged.setdefault(part, {}).update(values)
    out["patch"] = merged
    return out


def _g(preset: str, pitch: int, beats: float, **patch) -> dict:
    sound: dict = {"preset": preset, "notes": note(pitch, beats), "bpm": BPM}
    return _over(sound, **patch) if patch else sound


_KICK_DEEP = _g("KICK DEEP", 60, 0.25)
_KICK_PUNCH = _g("KICK PUNCH", 60, 0.25)
_SNARE = _g("SNARE TIGHT", 60, 0.25)
_HAT = _g("HAT CLOSED", 60, 0.25)
_OPEN = _g("HAT OPEN", 60, 0.25)
_CLAP = _g("CLAP", 60, 0.25)
_BOOM = _g("BOOM BASS", 36, 2)
_SUB = _g("DEEP SINE", 36, 2)
_PLUCK = _g("ARP PLUCK", 72, 0.25)
_PAD = _g("WARM PAD", 60, 8)
_LEAD = _g("SUPERSAW", 72, 2)
_BELL = _g("GLASS BELLS", 72, 2)
_RISER = _g("RISER", 60, 16, knobs={"ampAttack": 7.4, "filterAttack": 7.8})

CASES: tuple[SoundCase, ...] = (
    SoundCase("kick_deep", "kick",
              "I need a deep, long kick for a slow house track.", "Eval Kick Deep",
              (Ask("pitch_hz", None, 62.0, "a DEEP kick's body is low"),
               Ask("decay_ms", 120.0, None, "a LONG kick rings")),
              _KICK_DEEP, dict(_KICK_DEEP, notes=note(72, 0.25)), "DEEP"),
    SoundCase("kick_tight", "kick",
              "Give me a short, punchy kick - it has to get out of the way fast.", "Eval Kick Tight",
              (Ask("decay_ms", None, 110.0, "a SHORT kick is gone quickly"),),
              _KICK_PUNCH, _over(_KICK_PUNCH, knobs={"ampDecay": 0.7, "ampRelease": 0.7}), "SHORT"),
    SoundCase("kick_on_e", "kick",
              "Make a kick whose body sits on the note E - the track is in E minor.", "Eval Kick E",
              (Ask("pitch_midi", says="the kick's body sits on E", pitch_class=4),),
              _over(_KICK_DEEP, knobs={"tune1": -32}), _KICK_DEEP, "sits on E"),
    SoundCase("snare_bright", "snare",
              "A bright, snappy snare please.", "Eval Snare",
              (Ask("centroid_hz", 2500.0, None, "a BRIGHT snare has its weight high"),
               Ask("decay_ms", None, 150.0, "a SNAPPY snare is short")),
              _SNARE, _over(_SNARE, set={"filter.type": "lp", "filter.slope": 24}, knobs={"filterFreq": 84}), "BRIGHT"),
    SoundCase("hat_closed", "hat",
              "I want a tight closed hi-hat.", "Eval Hat",
              (Ask("decay_ms", None, 70.0, "a TIGHT hat is very short"),),
              _HAT, _over(_HAT, knobs={"ampDecay": 0.5, "ampRelease": 0.5}), "TIGHT"),
    SoundCase("hat_open", "open_hat",
              "An open hi-hat that rings for about a quarter of a second.", "Eval Open Hat",
              (Ask("decay_ms", 160.0, 450.0, "it rings for about a quarter of a second"),),
              _OPEN, _over(_OPEN, knobs={"ampDecay": 0.05, "ampRelease": 0.05}), "quarter of a second"),
    SoundCase("clap", "clap",
              "Make me a clap.", "Eval Clap",
              (),
              _CLAP, _over(_CLAP, set={"osc1.wave": "sin", "osc2.wave": "sin", "osc3.wave": "sin", "osc4.wave": "sin"}),
              "a clap is noise"),
    # (the brief is in a user's word for it; the kind it must be kept as is `boom`)
    SoundCase("bass_boom", "boom",
              "I need an 808 bass note that rings for a good second.", "Eval Low Boom",
              (Ask("sounding_s", 1.0, None, "it rings for a second"),),
              _BOOM, _over(_BOOM, knobs={"tune1": 7}), "out of tune"),
    SoundCase("sub_on_c", "sub",
              "A clean sub bass, on a low C.", "Eval Sub",
              (Ask("pitch_midi", says="it sounds a C", pitch_class=0),
               Ask("noisiness", None, 0.1, "a CLEAN sub is a pure tone")),
              _SUB, dict(_SUB, notes=note(41, 2)), "sounds a C"),
    SoundCase("pluck_bright", "pluck",
              "A short bright pluck with a bit of echo on it.", "Eval Pluck",
              (Ask("centroid_hz", 600.0, None, "a BRIGHT pluck"),),
              _PLUCK, dict(_over(_PLUCK, knobs={"ampSustain": 100, "filterSustain": 100}, set={"fx.delay": False}),
                           notes=note(72, 2)), "held note"),
    SoundCase("pad_warm", "pad",
              "A slow, warm, wide pad.", "Eval Pad",
              (Ask("attack_ms", 150.0, None, "a SLOW pad fades in"),
               Ask("centroid_hz", None, 1200.0, "a WARM pad is not bright"),
               Ask("width_db", -20.0, None, "a WIDE pad has a side")),
              # (the amp's attack alone is not enough: this pad's FILTER also opens over 0.6 s, and its
              #  free-phase unison voices beat - one run in four still read a 150 ms rise)
              _PAD, _over(_PAD, knobs={"ampAttack": 0.002, "filterAttack": 0.002},
                          set={"osc1.unison": 1, "osc2.unison": 1}), "SLOW"),
    SoundCase("lead_wide", "lead",
              "A big, wide, detuned lead.", "Eval Lead",
              (Ask("width_db", -12.0, None, "a WIDE lead has a side"),),
              _LEAD, _over(_LEAD, set={"osc1.unison": 1, "osc2.unison": 1, "fx.reverb": False},
                           knobs={"spread1": 0, "spread2": 0}), "WIDE"),
    SoundCase("bell_keys", "keys",
              "A bell-like key sound that rings out for a second or more.", "Eval Bell",
              (Ask("crest_db", 12.0, None, "a bell is struck and dies away"),
               Ask("sounding_s", 1.0, None, "it rings for a second")),
              _BELL, _over(_BELL, knobs={"ampSustain": 100, "ampDecay": 0.05}, set={"fx.delay": False, "fx.reverb": False}),
              "struck"),
    SoundCase("riser", "riser",
              "A four-bar noise riser for the build before a drop, at 120 BPM.", "Eval Riser",
              (Ask("sounding_s", 7.0, None, "four bars at 120 BPM is eight seconds"),
               Ask("noisiness", 0.7, None, "a NOISE riser")),
              _RISER, _over(_RISER, knobs={"ampAttack": 0.01, "filterAttack": 0.01}), "a riser builds"),
)
CASE_BY_KEY = {c.key: c for c in CASES}


# ---------------------------------------------------------------- the two layers

def packs_root(home: Path) -> Path:
    return home.joinpath(*PACKS_UNDER_HOME)


def read_manifest(root: Path, pack: str) -> Optional[dict]:
    """A pack's manifest by the pack's name, in any case; None when there is no such pack."""
    if not root.is_dir():
        return None
    for folder in root.iterdir():
        if folder.is_dir() and folder.name.lower() == pack.lower():
            try:
                doc = json.loads((folder / "pack.json").read_text())
            except (OSError, ValueError):
                return None
            doc["_folder"] = str(folder)
            return doc
    return None


def profile_of(path: Path, row: dict) -> dict:
    """What a kept file measures as, read HERE - the grader's own reading of the end state."""
    sf, sample_profile, _ = _engine()
    audio, rate = sf.read(str(path), always_2d=True, dtype="float32")
    loop = bool(row.get("loop"))
    return sample_profile(audio, rate, kind="loop" if loop else "one_shot",
                          bpm=row.get("bpm") if loop else None,
                          beats=row.get("beats") if loop else None).model_dump(exclude_none=True)


def kept(case: SoundCase, root: Path) -> tuple[Optional[dict], Optional[dict], list[str]]:
    """(the manifest row, the file's profile, why not) for the sound a case asked to be kept."""
    manifest = read_manifest(root, case.pack)
    if manifest is None:
        return None, None, [f"there is no pack called {case.pack}"]
    rows = [r for r in manifest.get("sounds") or [] if isinstance(r, dict)]
    if not rows:
        return None, None, [f"the pack {case.pack} holds no sound"]
    mine = [r for r in rows if r.get("kind") == case.kind]
    if not mine:
        return None, None, [f"nothing in {case.pack} is kept as a {case.kind} "
                            f"(it holds: {', '.join(str(r.get('kind')) for r in rows)})"]
    row = mine[-1]                                  # the last one kept is the answer
    path = Path(manifest["_folder"]) / str(row.get("file") or "")
    if not path.is_file():
        return row, None, [f"the manifest names {row.get('file')} and the file is not there"]
    return row, profile_of(path, row), []


def outcome_of(case: SoundCase, row: Optional[dict], profile: Optional[dict]) -> list[str]:
    """The OUTCOME layer on a kept sound's own figures (pure: the pytest re-grades with it)."""
    if profile is None:
        return ["no kept sound to measure"]
    _, _, sound_classes = _engine()
    played = (row or {}).get("root_note")
    failures = [f"against its kind - {i['says']}"
                for i in sound_classes.issues(profile, case.kind, played if isinstance(played, int) else None)]
    for ask in case.asks:
        said = ask.failure(profile, sound_classes)
        if said:
            failures.append(f"against the brief - {said}")
    return failures


def outcome_failures(case: SoundCase, root: Path) -> tuple[list[str], Optional[dict], Optional[dict]]:
    row, profile, why = kept(case, root)
    return (why or outcome_of(case, row, profile)), row, profile


def plan_failures(rows: list[dict], tree_writes: list[str]) -> list[str]:
    """The PLAN layer, from the turn's feed rows (`key`, `failed`, `noresult`) and what the
    turn wrote to the edit."""
    ran = [str(r.get("key") or "") for r in rows if not (r.get("failed") or r.get("noresult"))]
    failures = []
    if "pack.add" not in ran:
        failures.append("nothing was kept: no pack.add ran")
    elif "sound.render" not in ran[:ran.index("pack.add")]:
        failures.append("the sound was kept without being heard first: no sound.render before pack.add")
    took = [k for k in ran if k in REMOVALS]
    if took:
        failures.append(f"something was removed that nobody asked to remove: {', '.join(took)}")
    wrote = [w for w in tree_writes if "INPUTDEVICE" not in w]
    if wrote:
        failures.append(f"the project was changed for a request that asked for a sample: {wrote[0][:80]}")
    return failures


# ---------------------------------------------------------------- scripted answers

def _kept_sound(case: SoundCase, sound: dict) -> dict:
    return {"name": case.name, "kind": case.kind, **sound}


def answer(case: SoundCase, sound: dict, n: int) -> list[dict]:
    """A scripted turn that hears `sound` and keeps it: create the pack, render, add."""
    return [R.ask(R.call(f"toolu_{n}a", "pack.create", name=case.pack)),
            R.ask(R.call(f"toolu_{n}b", "sound.render", sounds=[{**sound, "name": case.name, "kind": case.kind}])),
            R.ask(R.call(f"toolu_{n}c", "pack.add", pack=case.pack, sounds=[_kept_sound(case, sound)])),
            R.say(f"{case.name} is in {case.pack}.")]


def _init_of(case: SoundCase) -> dict:
    return {"preset": "INIT", "notes": case.golden["notes"], "bpm": BPM}


VARIANTS: dict[str, Callable[[SoundCase, int], list[dict]]] = {
    "golden": lambda c, n: answer(c, c.golden, n),
    "init": lambda c, n: answer(c, _init_of(c), n),
    "mutant": lambda c, n: answer(c, c.mutant, n),
    "null": lambda c, n: [R.say("Done.")],
}


def session(variant: str, cases: tuple[SoundCase, ...] = CASES) -> R.Scenario:
    """Every case as one turn of ONE session (a scripted model does not mind the history, and
    one app start serves them all)."""
    turns = [R.Turn(c.prompt(), VARIANTS[variant](c, i)) for i, c in enumerate(cases)]
    return R.Scenario(f"sound_eval_{variant}", f"the sound-design eval's {variant} answers",
                      turns, lambda o: [], ceiling_s=120.0)


def grade(o: R.Outcome, cases: tuple[SoundCase, ...], home: Path) -> list[dict]:
    """Each case of a session, on both layers. A turn that left no result is an ERROR, in
    neither class."""
    out = []
    root = packs_root(home)
    for i, case in enumerate(cases):
        if i >= len(o.turns):
            out.append({"key": case.key, "error": "the turn left no result"})
            continue
        outcome, row, profile = outcome_failures(case, root)
        out.append({"key": case.key, "kind": case.kind,
                    "plan_failures": plan_failures(o.rows(i), o.turns[i].get("tree_writes") or []),
                    "outcome_failures": outcome,
                    "commands": [r.get("key") for r in o.rows(i)],
                    "row": {k: v for k, v in (row or {}).items() if k in ("name", "kind", "file", "root_note", "length_s", "channels")},
                    "profile": profile})
    return out


def run_session(variant: str, work: Path, cases: tuple[SoundCase, ...] = CASES) -> tuple[list[str], list[dict]]:
    bad, o = R.run_scenario(session(variant, cases), work)
    return bad, grade(o, cases, work / R.HOME_MARKER)


# ---------------------------------------------------------------- the self-test

def self_test(record: bool = False, keep: Optional[Path] = None) -> int:
    """Goldens pass; INIT fails each kind's outcome; a do-nothing agent fails both layers;
    each mutant fails the outcome on the figure its case names. 0 = the graders hold."""
    if not R.BINARY.exists():
        print(f"sound_eval: the console is not built ({R.BINARY})")
        return EXIT_NOT_BUILT
    base = keep or Path(tempfile.mkdtemp(prefix="sound-eval-"))
    began = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        runs = dict(zip(VARIANTS, pool.map(lambda v: run_session(v, base / v), VARIANTS)))
    wrong: list[str] = []
    for variant, (bad, _) in runs.items():
        for line in bad:
            if variant == "null" and "console leg" not in line:
                continue
            wrong.append(f"[{variant}] the rig: {line}")
    by = {v: {g["key"]: g for g in graded} for v, (_, graded) in runs.items()}
    for case in CASES:
        g, i, m, n = (by[v].get(case.key, {"error": "no result"}) for v in ("golden", "init", "mutant", "null"))
        for variant, got in (("golden", g), ("init", i), ("mutant", m), ("null", n)):
            if got.get("error"):
                wrong.append(f"{case.key} [{variant}]: {got['error']}")
        if any(x.get("error") for x in (g, i, m, n)):
            continue
        if g["plan_failures"] or g["outcome_failures"]:
            wrong.append(f"{case.key}: the GOLDEN answer does not pass - "
                         + "; ".join(g["plan_failures"] + g["outcome_failures"]))
        if not i["outcome_failures"]:
            wrong.append(f"{case.key}: the INIT preset kept as a {case.kind} passes the outcome layer")
        if i["plan_failures"]:
            wrong.append(f"{case.key}: INIT heard and kept is a fine PLAN, and the plan layer fails it - "
                         + "; ".join(i["plan_failures"]))
        if not any(case.mutant_turns_red.lower() in f.lower() for f in m["outcome_failures"]):
            wrong.append(f"{case.key}: the mutant was not caught on '{case.mutant_turns_red}' - it drew: "
                         + ("; ".join(m["outcome_failures"]) or "nothing"))
        if not n["plan_failures"] or not n["outcome_failures"]:
            wrong.append(f"{case.key}: an agent that did nothing passes a layer")
    took = time.monotonic() - began
    for case in CASES:
        g = by["golden"].get(case.key, {})
        p = g.get("profile") or {}
        print(f"  {case.key:<13} {case.kind:<9} golden {'ok ' if not (g.get('plan_failures') or g.get('outcome_failures')) else 'XX '}"
              f" init-fails {len(by['init'].get(case.key, {}).get('outcome_failures') or [])}"
              f" mutant-fails {len(by['mutant'].get(case.key, {}).get('outcome_failures') or [])}"
              f"   dec {p.get('decay_ms')} cen {p.get('centroid_hz')} pitch {p.get('pitch_midi')}")
    for line in wrong:
        print("  XX " + line)
    print(f"sound_eval --self-test: {len(CASES)} cases x {len(VARIANTS)} answers in {took:.0f} s - "
          + ("the graders hold" if not wrong else f"{len(wrong)} problems"))
    if record and not wrong:
        doc = {"note": "what each case's scripted answers measured as in the real app; "
                       "written by `sound_eval --self-test --record`, re-graded by a pytest",
               "cases": {c.key: {v: {"row": by[v][c.key]["row"], "profile": by[v][c.key]["profile"]}
                                 for v in ("golden", "init", "mutant")} for c in CASES}}
        GOLDENS.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        print(f"wrote {GOLDENS.name}")
    if keep is None:
        shutil.rmtree(base, ignore_errors=True)
    return 1 if wrong else 0


# ---------------------------------------------------------------- a live run (BILLED)

def run_case_live(case: SoundCase, work: Path, transport: str) -> dict:
    scenario = R.Scenario(f"sound_eval_{case.key}", case.brief, [R.Turn(case.prompt(), [])],
                          lambda o: [], ceiling_s=600.0, unsettled_ok=True)
    try:
        bad, o = R.run_scenario(scenario, work, live=transport)
    except Exception as exc:  # noqa: BLE001 - a case that never reached the model is not a verdict
        return {"key": case.key, "error": f"{type(exc).__name__}: {exc}"}
    if not o.turns or not (o.rows(0) or o.rows(0, "prose")):
        return {"key": case.key, "error": "the turn left nothing in the feed: " + "; ".join(bad)[:300]}
    got = grade(o, (case,), work / R.HOME_MARKER)[0]
    got["rig"] = bad
    got["reply"] = " ".join(str(b.get("t") or "") for b in o.rows(0, "prose"))[-600:]
    return got


def run_live(transport: str, repeat: int, only: Optional[list[str]], out: Optional[Path], jobs: int = 2) -> dict:
    cases = [c for c in CASES if not only or c.key in only]
    base = Path(tempfile.mkdtemp(prefix="sound-eval-live-"))
    report: dict[str, Any] = {"transport": transport, "repeat": repeat, "cases": []}
    for case in cases:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            runs = list(pool.map(lambda r: run_case_live(case, base / f"{case.key}-{r}", transport), range(repeat)))
        measured = [r for r in runs if not r.get("error")]
        passed = [r for r in measured if not r["plan_failures"] and not r["outcome_failures"]]
        entry = {"key": case.key, "kind": case.kind, "runs": runs, "passed": len(passed),
                 "measured": len(measured), "unmeasured": not measured,
                 "stable_failure": bool(measured) and not passed}
        report["cases"].append(entry)
        print(f"  {case.key:<13} {len(passed)}/{len(measured)} passed"
              + (f"  ({repeat - len(measured)} errors)" if len(measured) < repeat else "")
              + ("" if passed or not measured else "   " + "; ".join((measured[0]["plan_failures"] + measured[0]["outcome_failures"])[:2])))
    errors = sum(1 for e in report["cases"] for r in e["runs"] if r.get("error"))
    report["summary"] = {"cases": len(cases), "errors": errors,
                         "passed_every_run": sum(1 for e in report["cases"] if e["measured"] and e["passed"] == e["measured"]),
                         "stable_failures": [e["key"] for e in report["cases"] if e["stable_failure"]],
                         "unmeasured": [e["key"] for e in report["cases"] if e["unmeasured"]]}
    if out:
        out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    shutil.rmtree(base, ignore_errors=True)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--list", action="store_true", help="the briefs and what each is graded on (free)")
    parser.add_argument("--self-test", action="store_true", help="prove the graders with scripted answers (free)")
    parser.add_argument("--record", action="store_true", help="with --self-test: write sound_eval_goldens.json")
    parser.add_argument("--keep", help="with --self-test: keep the sessions' folders here")
    parser.add_argument("--live", choices=sorted(R.LIVE_CREDENTIAL), help="run the briefs on a real model: BILLED")
    parser.add_argument("--billed", action="store_true", help="say that a live run is meant")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--check", action="store_true", help="exit 1 on a stable failure (needs --repeat 2 or more)")
    parser.add_argument("--case", action="append", help="only this case (repeatable)")
    parser.add_argument("--out", help="write the report here")
    args = parser.parse_args(argv)
    if args.list:
        for c in CASES:
            print(f"{c.key:<13} {c.kind:<9} {c.prompt()}")
            for a in c.asks:
                print(f"{'':<24} - {a.says}")
        return 0
    if args.self_test:
        return self_test(args.record, Path(args.keep).resolve() if args.keep else None)
    if args.live:
        if not args.billed:
            print("a live run asks a real model and is BILLED: say --billed as well")
            return 1
        if args.check and args.repeat < 2:
            parser.error("--check needs --repeat 2 or more: one run cannot tell a stable failure from a bad day")
        if not R.BINARY.exists():
            print(f"sound_eval: the console is not built ({R.BINARY})")
            return EXIT_NOT_BUILT
        report = run_live(args.live, args.repeat, args.case, Path(args.out).resolve() if args.out else None)
        s = report["summary"]
        print(f"sound_eval [{args.live}]: {s['passed_every_run']}/{s['cases']} cases passed every run; "
              f"stable failures: {s['stable_failures'] or 'none'}; errors: {s['errors']}")
        if s["errors"] > MAX_SWEEP_ERRORS or len(s["unmeasured"]) == s["cases"]:
            print("too many runs never reached the model: this is not a measurement")
            return 2
        return 1 if (args.check and s["stable_failures"]) else 0
    parser.error("give --list, --self-test or --live")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
