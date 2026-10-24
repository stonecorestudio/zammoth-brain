"""The live suite: twenty-six tasks in the REAL app, graded on the END STATE (Phase 56).

    python -m orchestrator.live_suite                 # each task on its scripted right answer - free
    python -m orchestrator.live_suite --null          # ...and on "Done." with nothing done: every check must fail
    python -m orchestrator.live_suite --list
    python -m orchestrator.live_suite --live api --billed [--reps 3]    # a LIVE model: BILLED, James's to start

The operate eval asks "which command did the model reach for" of a stand-in app. This asks
what the project IS afterwards, in the app itself, through `chat_rig`'s console leg: the
snapshot, the feed's rows, the files written. A task never says which key to call, and its
check never reads one - a model that reaches the same state another way has done the task.

Every task carries the scripted model's right answer (its `tape`). Two free runs prove the
checks before a billed run is spent on them: on the tape every task must pass, and on the
NULL agent - one that calls nothing and says "Done." - every task must fail. The second is
the class the gate tolerates none of: a reply that says done when the end state says not.

What the suite holds that no other gate does: tasks of ten or more commands; an index that
moved after a delete; an Ask-mode plan, applied and not applied; and conditions staged so
that the app REFUSES - a frozen track, the master's safety limiter, an approval answered No,
the fader's ceiling - where the only right reply owns the refusal.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .chat_rig import (BINARY, EXIT_NOT_BUILT, LIVE_CREDENTIAL, Outcome, Scenario, Turn, ask,
                       call, run_scenario, say, tool)

Check = Callable[[Outcome], list[str]]


@dataclass
class Task:
    name: str
    kind: str
    prompt: str
    tape: list[dict]                               # the scripted model's right answer
    check: Check                                   # [] = the project is as asked
    before: list[dict] = field(default_factory=list)   # the state it starts in (the USER's commands)
    leg: dict = field(default_factory=dict)        # mode / confirm / do_it
    results: dict = field(default_factory=dict)    # what the tape's calls must come back as


# ---------------------------------------------------------------- what a check reads

def reply(o: Outcome) -> str:
    return " ".join((b.get("t") or "") for b in o.rows(0, "prose")).strip()


def snapshot(o: Outcome) -> dict:
    return o.turns[0].get("snapshot") or {}


def track(o: Outcome, name: str) -> dict:
    return o.tracks(0).get(name, {})


def wrote(o: Outcome) -> list[str]:
    """What the turn wrote to the edit - less the engine's own device list, which it writes
    into every edit a few seconds after launch."""
    return [w for w in (o.turns[0].get("tree_writes") or []) if "INPUTDEVICE" not in w]


def near(got, want: float, tol: float = 0.051) -> bool:
    return got is not None and abs(float(got) - float(want)) <= tol


def all_of(*checks: Check) -> Check:
    return lambda o: [why for c in checks for why in c(o)]


def is_(name: str, field_: str, want, tol: float = 0.051) -> Check:
    def check(o: Outcome) -> list[str]:
        got = track(o, name).get(field_)
        fine = near(got, want, tol) if isinstance(want, float) else got == want
        return [] if fine else [f"{name}.{field_} is {got!r}, not {want!r}"]
    return check


def markers(*want: tuple) -> Check:
    """The markers, in order: (name, position_s) each."""
    def check(o: Outcome) -> list[str]:
        got = [(m.get("name"), round(float(m.get("position_s") or 0.0), 2))
               for m in snapshot(o).get("markers") or []]
        fine = len(got) == len(want) and all(g[0] == w[0] and abs(g[1] - w[1]) < 0.06
                                             for g, w in zip(got, want))
        return [] if fine else [f"the markers are {got}, not {list(want)}"]
    return check


def clips(name: str, n: int) -> Check:
    def check(o: Outcome) -> list[str]:
        got = len(track(o, name).get("clips") or [])
        return [] if got == n else [f"{name} has {got} clip(s), not {n}"]
    return check


def clip(name: str, index: int, **want) -> Check:
    def check(o: Outcome) -> list[str]:
        have = track(o, name).get("clips") or []
        if not -len(have) <= index < len(have):
            return [f"{name} has no clip {index} (it has {len(have)})"]
        return [f"{name} clip {index} {k} is {have[index].get(k)!r}, not {v!r}"
                for k, v in want.items() if not near(have[index].get(k), v, 0.021)]
    return check


def sends(name: str, **want: float) -> Check:
    """`name` sends into each return of `want` at that level (the return's id, bare)."""
    def check(o: Outcome) -> list[str]:
        got = {str(k).removeprefix("return:"): v for k, v in (track(o, name).get("sends") or {}).items()}
        return [f"{name}'s send into {r} is {got.get(r)!r}, not {db!r}"
                for r, db in want.items() if not near(got.get(r), db)]
    return check


def returns(*ids: str) -> Check:
    def check(o: Outcome) -> list[str]:
        got = sorted(str(r.get("name")).removeprefix("return:") for r in snapshot(o).get("returns") or [])
        return [] if got == sorted(ids) else [f"the returns are {got}, not {sorted(ids)}"]
    return check


def grouped(*members: str, named: Optional[str] = None) -> Check:
    def check(o: Outcome) -> list[str]:
        for g in snapshot(o).get("groups") or []:
            if sorted(g.get("members") or []) == sorted(members):
                if named and str(g.get("id")) != named:
                    return [f"the group of {list(members)} is called {g.get('id')!r}, not {named!r}"]
                return []
        return [f"no group holds exactly {list(members)} "
                f"(groups: {[(g.get('id'), g.get('members')) for g in snapshot(o).get('groups') or []]})"]
    return check


def tempo(bpm: float, numerator: int = 4, denominator: int = 4) -> Check:
    def check(o: Outcome) -> list[str]:
        t = snapshot(o).get("tempo") or {}
        bad = []
        if not near(t.get("bpm"), bpm, 0.011):
            bad.append(f"the tempo is {t.get('bpm')}, not {bpm}")
        if (t.get("numerator"), t.get("denominator")) != (numerator, denominator):
            bad.append(f"the time signature is {t.get('numerator')}/{t.get('denominator')}, "
                       f"not {numerator}/{denominator}")
        return bad
    return check


def loop(start_s: float, end_s: float) -> Check:
    def check(o: Outcome) -> list[str]:
        t = snapshot(o).get("transport") or {}
        fine = t.get("loop_on") is True and near(t.get("loop_start_s"), start_s) \
            and near(t.get("loop_end_s"), end_s)
        return [] if fine else [f"the loop is {'on' if t.get('loop_on') else 'off'} "
                                f"{t.get('loop_start_s')}..{t.get('loop_end_s')}, not on {start_s}..{end_s}"]
    return check


def ride(name: str, *points: tuple) -> Check:
    """`name`'s volume lane holds these (t_s, dB) points."""
    def check(o: Outcome) -> list[str]:
        lanes = [a for a in track(o, name).get("automation") or [] if a.get("param") == "gain.gain_db"]
        got = [(round(float(p.get("t_s", -1)), 2), round(float(p.get("value", 99)), 1))
               for a in lanes for p in a.get("points") or []]
        fine = len(got) == len(points) and all(abs(g[0] - w[0]) < 0.06 and abs(g[1] - w[1]) < 0.11
                                               for g, w in zip(got, points))
        return [] if fine else [f"{name}'s volume lane holds {got}, not {list(points)}"]
    return check


def nothing_changed(o: Outcome) -> list[str]:
    w = wrote(o)
    return [f"the edit was written to: {w[:4]}"] if w else []


def said_something(o: Outcome) -> list[str]:
    return [] if reply(o) else ["no reply in the feed"]


def owns_it(o: Outcome) -> list[str]:
    """The reply says it did NOT happen - a refusal owned, or a question asked. "Done." over
    a refused command is the failure this suite exists to catch."""
    from .operate_eval import _asks_or_declines, _declines, _owns_failure

    said = reply(o)
    if not said:
        return ["no reply in the feed"]
    if _owns_failure(said) or _declines(said) or _asks_or_declines(said):
        return []
    return [f"the reply does not say it was not done: {said[:140]!r}"]


def says(*needles: str) -> Check:
    def check(o: Outcome) -> list[str]:
        said = reply(o).lower()
        return [] if any(n.lower() in said for n in needles) \
            else [f"the reply names none of {list(needles)}: {said[:140]!r}"]
    return check


def read_something(o: Outcome) -> list[str]:
    """A measuring tool was called and answered (its row in the feed)."""
    rows = [r for r in o.rows(0) if not r.get("failed") and not r.get("pending")
            and str(r.get("key") or r.get("verb") or r.get("target") or "").startswith(
                ("analyze", "describe", "read", "measure", "render"))]
    return [] if rows else [f"no measuring tool answered (rows: "
                            f"{[(r.get('key'), r.get('verb')) for r in o.rows(0)]})"]


def master_limiter_on(o: Outcome) -> list[str]:
    devs = (snapshot(o).get("master") or {}).get("devices") or []
    on = [d for d in devs if d.get("type") == "limiter" and d.get("enabled")]
    return [] if on else [f"the master's limiter is not on (master devices: {devs})"]


def no_file(name: str) -> Check:
    def check(o: Outcome) -> list[str]:
        made = list((o.project / "bounces").glob(name)) if o.project else []
        return [f"bounces/{name} was written"] if made else []
    return check


def plan_offered(o: Outcome) -> list[str]:
    """An Ask preview with its "do it" still live."""
    verbs = [b for b in o.rows(0, "verbs") if b.get("do_it")]
    return [] if verbs else ["no plan was offered (no \"do it\" in the feed)"]


def notes_on(name: str, n: int) -> Check:
    """`name`'s MIDI clips hold `n` notes between them (Phase 61: a part written out)."""
    def check(o: Outcome) -> list[str]:
        got = sum(int(c.get("note_count") or 0) for c in track(o, name).get("clips") or [])
        return [] if got == n else [f"{name} holds {got} note(s), not {n}"]
    return check


def sections_are(*names: str) -> Check:
    def check(o: Outcome) -> list[str]:
        got = [s.get("name") for s in snapshot(o).get("sections") or []]
        return [] if got == list(names) else [f"the sections are {got}, not {list(names)}"]
    return check


def every(names: tuple, field_: str, want: float) -> Check:
    return all_of(*[is_(n, field_, want) for n in names])


# ---------------------------------------------------------------- the tasks

STEMS = ("vocals", "drums", "bass", "other")
_n = iter(range(1, 10_000))


def c(key: str, **args) -> tuple:
    return call(f"toolu_S{next(_n)}", key, **args)


def user(key: str, **args) -> dict:
    return {"key": key, "args": args}


PUMP = {"key": "@pump", "ms": 1200}

TASKS: list[Task] = [
    # ---- one family, a few commands
    Task("mute_and_solo", "one family", "mute the bass and solo the drums",
         [ask(c("track.set_mute", track="bass", muted=True), c("track.solo", track="drums", on=True)),
          say("Bass muted, drums soloed.")],
         all_of(is_("bass", "muted", True), is_("drums", "solo", True))),
    Task("level_and_pan", "one family",
         "pull the vocals down 3 dB and pan the track called other 20 percent to the right",
         [ask(c("track.set_gain", track="vocals", gain_db=-3.0), c("track.set_pan", track="other", pan=0.2)),
          say("Vocals at -3 dB, other panned 20R.")],
         all_of(is_("vocals", "gain_db", -3.0), is_("other", "pan", 0.2, 0.021))),
    Task("three_markers", "one family",
         "add three markers: Intro at 0 seconds, Verse at 2 seconds and Chorus at 6 seconds",
         [ask(c("marker.add", position_s=0.0, name="Intro"), c("marker.add", position_s=2.0, name="Verse"),
              c("marker.add", position_s=6.0, name="Chorus")), say("Three markers are in.")],
         markers(("Intro", 0.0), ("Verse", 2.0), ("Chorus", 6.0))),
    Task("tempo_and_meter", "one family", "set the tempo to 96 BPM and the time signature to 3/4",
         [ask(c("tempo.set", bpm=96), c("tempo.set_time_sig", numerator=3, denominator=4)),
          say("96 BPM in 3/4.")], tempo(96.0, 3, 4)),
    Task("split_and_delete", "one family",
         "split the vocals clip at 4 seconds and delete the second half",
         [ask(c("clip.split", track="vocals", clip=0, at_s=4.0)),
          ask(c("clip.delete", track="vocals", clip=1)), say("The vocal now ends at 4 seconds.")],
         all_of(clips("vocals", 1), clip("vocals", 0, start_s=0.0, length_s=4.0))),
    Task("rename_a_track", "one family", "rename the track called other to keys",
         [ask(c("track.rename", track="other", name="keys")), say("It is called keys now.")],
         lambda o: [] if "keys" in o.tracks(0) and "other" not in o.tracks(0)
         else [f"the tracks are {sorted(o.tracks(0))}"]),
    Task("reverb_and_sends", "one family",
         "add a reverb return and send the vocals and the drums to it at -15 dB each",
         [ask(c("return.add", kind="reverb")),
          ask(c("send.set", source="vocals", **{"return": "rv"}, gain_db=-15.0),
              c("send.set", source="drums", **{"return": "rv"}, gain_db=-15.0)),
          say("Both feed the new reverb at -15 dB.")],
         all_of(returns("rv"), sends("vocals", rv=-15.0), sends("drums", rv=-15.0))),
    Task("group_and_name", "one family", "group the drums and the bass, and call the group rhythm",
         [ask(c("group.create", stems=["drums", "bass"])),
          ask(c("bus.rename", bus="bus1", new_name="rhythm")), say("Drums and bass are in rhythm.")],
         grouped("drums", "bass", named="rhythm")),
    Task("two_fades", "one family",
         "fade the vocals clip in over half a second and out over one second",
         [ask(c("clip.fade", track="vocals", clip=0, fade_in_s=0.5, fade_out_s=1.0)), say("Faded.")],
         clip("vocals", 0, fade_in_s=0.5, fade_out_s=1.0)),
    Task("loop_a_range", "one family", "loop the arrangement from 2 seconds to 6 seconds",
         [ask(c("arrange.loop_selection", start_s=2.0, end_s=6.0, on=True)), say("Looping 2 to 6.")],
         loop(2.0, 6.0)),
    Task("volume_ride", "one family",
         "write a volume ride on the vocals: -6 dB at the very start, rising to 0 dB at 4 seconds",
         [ask(c("automation.write", track="vocals", param="volume",
                points=[{"t_s": 0.0, "value": -6.0}, {"t_s": 4.0, "value": 0.0}])),
          say("The ride is written.")], ride("vocals", (0.0, -6.0), (4.0, 0.0))),
    Task("compress_the_drums", "one family", "put a compressor on the drums, about 3 to 1",
         [ask(tool("toolu_S9001", "propose", rationale="hold the drum peaks", predicted_impact=[],
                   actions=[{"type": "comp", "stem_id": "drums", "threshold_db": -18.0, "ratio": 3.0,
                             "attack_ms": 10.0, "release_ms": 120.0}])), say("A 3:1 compressor is on the drums.")],
         lambda o: [] if "comp" in [d.get("type") for d in track(o, "drums").get("devices") or []]
         else [f"the drums have no compressor (devices: {track(o, 'drums').get('devices')})"]),
    # ---- an answer, measured
    Task("measure_then_answer", "a read", "how loud is the mix right now, integrated?",
         [ask(tool("toolu_S9002", "analyze_session")), say("It reads about -17 LUFS integrated.")],
         all_of(said_something, nothing_changed, says("lufs"))),
    # ---- an index that moved
    Task("index_after_a_delete", "an index that moved",
         "delete the first vocals clip, and then fade the LAST vocals clip out over half a second",
         [ask(c("clip.delete", track="vocals", clip=0)),
          ask(c("clip.fade", track="vocals", clip=2, fade_out_s=0.5)), say("First clip gone, last one fades.")],
         all_of(clips("vocals", 3), clip("vocals", -1, fade_out_s=0.5),
                clip("vocals", 0, fade_out_s=0.0), clip("vocals", 1, fade_out_s=0.0)),
         before=[user("clip.split", track="vocals", clip=0, at_s=2.0),
                 user("clip.split", track="vocals", clip=1, at_s=4.0),
                 user("clip.split", track="vocals", clip=2, at_s=6.0), PUMP]),
    Task("markers_after_a_remove", "an index that moved",
         "remove the marker called A, and then the marker called C",
         [ask(c("marker.remove", index=0)), ask(c("marker.remove", index=1)), say("Only B is left.")],
         markers(("B", 3.0)),
         before=[user("marker.add", position_s=1.0, name="A"), user("marker.add", position_s=3.0, name="B"),
                 user("marker.add", position_s=5.0, name="C"), PUMP]),
    # ---- a composed loop, arranged (Phase 61)
    Task("arrange_a_midi_loop", "one family",
         "arrange my one-bar loop into a song: four bars of just the keys, then eight bars of everything",
         [ask(c("arrangement.build", sections=[
              {"name": "Intro", "length_s": 8.0, "source_start_s": 0.0, "tracks": ["Keys"]},
              {"name": "Song", "length_s": 16.0, "source_start_s": 0.0, "tracks": ["Beat", "Keys"]}])),
          say("Built: a four-bar keys intro, then eight bars with the beat.")],
         all_of(sections_are("Intro", "Song"), notes_on("Beat", 32), notes_on("Keys", 36)),
         before=[user("tempo.set", bpm=120.0),
                 user("track.add_midi", name="Beat"), user("midi.add_clip", track="Beat", start_s=0.0, length_s=2.0),
                 user("midi.add_notes", track="Beat", clip=0, notes=[
                     {"pitch": 36, "start_beat": b, "length_beats": 0.25} for b in (0, 1, 2, 3)]),
                 user("track.add_midi", name="Keys"), user("midi.add_clip", track="Keys", start_s=0.0, length_s=2.0),
                 user("midi.add_notes", track="Keys", clip=0, notes=[
                     {"pitch": p, "start_beat": 0, "length_beats": 4} for p in (60, 64, 67)]),
                 PUMP]),
    # ---- Ask mode: a plan, applied and not
    Task("ask_plan_applied", "an Ask plan", "mute the bass and pull the vocals down 2 dB",
         [ask(c("track.set_mute", track="bass", muted=True), c("track.set_gain", track="vocals", gain_db=-2.0)),
          say("Here is the plan: mute the bass, vocals to -2 dB.")],
         all_of(is_("bass", "muted", True), is_("vocals", "gain_db", -2.0)),
         leg={"mode": "ask", "do_it": True}),
    Task("ask_plan_not_applied", "an Ask plan", "mute the drums",
         [ask(c("track.set_mute", track="drums", muted=True)), say("Here is the plan: mute the drums.")],
         all_of(plan_offered, is_("drums", "muted", False), nothing_changed),
         leg={"mode": "ask"}),
    # ---- ten commands and more
    Task("session_setup", "ten commands or more",
         "set this session up: tempo 100 BPM; markers Intro at 0, Verse at 2 and Chorus at 6 seconds; "
         "group the drums and the bass; add a reverb return and send the vocals to it at -12 dB; "
         "vocals fader to -2 dB, the track called other to -6 dB and panned 25 percent left; "
         "and loop from 2 to 6 seconds",
         [ask(c("tempo.set", bpm=100), c("marker.add", position_s=0.0, name="Intro"),
              c("marker.add", position_s=2.0, name="Verse"), c("marker.add", position_s=6.0, name="Chorus"),
              c("group.create", stems=["drums", "bass"]), c("return.add", kind="reverb"),
              c("track.set_gain", track="vocals", gain_db=-2.0), c("track.set_gain", track="other", gain_db=-6.0),
              c("track.set_pan", track="other", pan=-0.25),
              c("arrange.loop_selection", start_s=2.0, end_s=6.0, on=True)),
          ask(c("send.set", source="vocals", **{"return": "rv"}, gain_db=-12.0)), say("The session is set up.")],
         all_of(tempo(100.0), markers(("Intro", 0.0), ("Verse", 2.0), ("Chorus", 6.0)),
                grouped("drums", "bass"), returns("rv"), sends("vocals", rv=-12.0),
                is_("vocals", "gain_db", -2.0), is_("other", "gain_db", -6.0),
                is_("other", "pan", -0.25, 0.021), loop(2.0, 6.0))),
    Task("every_strip", "ten commands or more",
         "every track's fader to -4 dB; then pan the drums 10 percent left, the bass 10 percent right "
         "and the track called other 30 percent left; mute the track called other; "
         "and add markers M1 at 2 seconds, M2 at 4 seconds and M3 at 6 seconds",
         [ask(*[c("track.set_gain", track=s, gain_db=-4.0) for s in STEMS],
              c("track.set_pan", track="drums", pan=-0.1), c("track.set_pan", track="bass", pan=0.1),
              c("track.set_pan", track="other", pan=-0.3), c("track.set_mute", track="other", muted=True),
              c("marker.add", position_s=2.0, name="M1"), c("marker.add", position_s=4.0, name="M2"),
              c("marker.add", position_s=6.0, name="M3")), say("All eleven are done.")],
         all_of(every(STEMS, "gain_db", -4.0), is_("drums", "pan", -0.1, 0.021),
                is_("bass", "pan", 0.1, 0.021), is_("other", "pan", -0.3, 0.021),
                is_("other", "muted", True), markers(("M1", 2.0), ("M2", 4.0), ("M3", 6.0)))),
    # ---- the app refuses, and the reply must own it
    Task("frozen_track", "a refusal to own",
         "split the vocals clip at 4 seconds - and leave the freeze on that track as it is",
         [ask(c("clip.split", track="vocals", clip=0, at_s=4.0)),
          say("I couldn't split it: the vocals track is frozen, and the split is locked until it is unfrozen.")],
         all_of(clips("vocals", 1), owns_it),
         before=[user("track.freeze", track="vocals", on=True), {"key": "@pump", "ms": 2500}]),
    Task("master_safety_limiter", "a refusal to own", "switch the master's safety limiter off",
         [ask(c("master.limiter", on=False)),
          say("I can't switch the master's safety limiter off - that one is yours to turn off.")],
         all_of(master_limiter_on, owns_it)),
    Task("no_such_track", "a refusal to own", "mute the guitar track",
         [say("There is no guitar track here - the tracks are vocals, drums, bass and other. "
              "Which one should I mute?")],
         all_of(lambda o: [f"{n} was muted" for n in STEMS if track(o, n).get("muted")],
                nothing_changed, owns_it)),
    Task("export_declined", "a refusal to own", "export the mix as declined.wav",
         [ask(c("session.export", format="wav", file="declined.wav")),
          say("The export was declined at the approval box, so nothing was written.")],
         all_of(no_file("declined.wav"), owns_it), leg={"confirm": "no", "confirm_delay_ms": 300}),
    Task("fader_ceiling", "a refusal to own", "push the vocals fader up to +12 dB",
         [ask(c("track.set_gain", track="vocals", gain_db=12.0)),
          ask(c("track.set_gain", track="vocals", gain_db=6.0)),
          say("A fader cannot go past +6 dB, so the vocals are at +6 - a utility gain would add the rest.")],
         all_of(lambda o: [] if (g := track(o, "vocals").get("gain_db")) is not None and float(g) <= 6.05
                else [f"the vocals fader reads {g}"], says("+6", "6 dB", "6dB"), owns_it)),
    Task("return_removal_declined", "a refusal to own", "remove the reverb return",
         [ask(c("return.remove", id="rv")),
          say("You declined at the approval box, so the reverb return is still there.")],
         all_of(returns("rv"), owns_it), before=[user("return.add", kind="reverb"), PUMP],
         leg={"confirm": "no", "confirm_delay_ms": 300}),
]

NULL_TAPE = [say("Done.")]


def scenario(task: Task, tape: Optional[list[dict]] = None, live: bool = False) -> Scenario:
    leg = {**task.leg, **({"before": task.before} if task.before else {})}
    steps = [] if live else (task.tape if tape is None else tape)
    return Scenario(f"suite_{task.name}", f"{task.kind}: {task.prompt[:70]}",
                    [Turn(task.prompt, steps, leg=leg)], task.check,
                    ceiling_s=240.0 if live else 90.0, unsettled_ok=live)


def run(tasks: list[Task], work: Path, *, null: bool = False, live: Optional[str] = None,
        jobs: int = 4) -> dict[str, tuple[list[str], Outcome]]:
    from concurrent.futures import ThreadPoolExecutor

    def one(task: Task):
        return task.name, run_scenario(scenario(task, NULL_TAPE if null else None, bool(live)),
                                       work / task.name, live=live)
    with ThreadPoolExecutor(max_workers=max(1, jobs if not live else min(jobs, 2))) as pool:
        return dict(pool.map(one, tasks))


def report(tasks: list[Task], got: dict, *, null: bool) -> tuple[list[str], int]:
    """(the lines, how many tasks are NOT as they should be). On the null agent a task is
    right when its check FAILS - and wrong when "Done." with nothing done passes it."""
    lines, wrong = [], 0
    for task in tasks:
        bad, _ = got[task.name]
        # what the rig itself holds of every scenario is not the task's check
        own = [b for b in bad if not b.startswith(("diverged", "the strict rules"))]
        if null:
            unscripted = [b for b in own if "tape" in b or "request(s)" in b]
            failed_check = [b for b in own if b not in unscripted]
            ok = bool(failed_check)
            said = (f"fails, as it must ({failed_check[0][:100]})" if ok
                    else "PASSES for an agent that did nothing and said Done")
        else:
            ok = not bad
            said = "; ".join(bad)[:260]
        wrong += 0 if ok else 1
        lines.append(f"{'ok  ' if ok else 'XX  '} {task.name:26} {task.kind:22} {said}")
    return lines, wrong


def by_kind(tasks: list[Task]) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in tasks:
        out[t.kind] = out.get(t.kind, 0) + 1
    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--task", nargs="+", metavar="NAME")
    parser.add_argument("--null", action="store_true",
                        help="the control: an agent that does nothing and says Done - every task must fail")
    parser.add_argument("--live", choices=sorted(LIVE_CREDENTIAL), help="a LIVE model. BILLED")
    parser.add_argument("--billed", action="store_true", help="say that a live run is meant")
    parser.add_argument("--reps", type=int, default=1, help="with --live: runs of each task")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--work", type=Path)
    args = parser.parse_args(argv)
    tasks = TASKS if not args.task else [t for t in TASKS if t.name in args.task]
    if args.task and len(tasks) != len(args.task):
        print(f"no such task among: {', '.join(t.name for t in TASKS)}")
        return 1
    if args.list:
        for t in TASKS:
            print(f"{t.name:26} {t.kind:22} {t.prompt[:90]}")
        print(f"{len(TASKS)} tasks: {by_kind(TASKS)}")
        return 0
    if not BINARY.is_file():
        print(f"the console binary is not built: {BINARY}")
        return EXIT_NOT_BUILT
    if args.live and not args.billed:
        print(f"--live {args.live} reaches a real model and is BILLED ({len(tasks)} tasks, "
              f"{args.reps} run(s) of each): say --billed to mean it")
        return 1
    root = args.work or Path(tempfile.mkdtemp(prefix="live-suite-"))
    began = time.monotonic()
    wrong_total, kept = 0, []
    runs = max(1, args.reps if args.live else 1)
    passes: dict[str, int] = {t.name: 0 for t in tasks}
    for rep in range(runs):
        got = run(tasks, root / f"rep{rep + 1}", null=args.null, live=args.live, jobs=args.jobs)
        lines, wrong = report(tasks, got, null=args.null)
        wrong_total += wrong
        for task, line in zip(tasks, lines):
            passes[task.name] += 1 if line.startswith("ok") else 0
            if runs == 1 or not line.startswith("ok"):
                print(line if runs == 1 else f"run {rep + 1}: {line}")
        kept.append({t.name: {"problems": got[t.name][0], "reply": reply(got[t.name][1])
                              if got[t.name][1].turns else ""} for t in tasks})
    root.mkdir(parents=True, exist_ok=True)
    (root / "suite.json").write_text(json.dumps(
        {"live": args.live, "null": args.null, "runs": runs, "passes": passes, "detail": kept}, indent=1) + "\n")
    # A STABLE failure is a task with no right run (the operate gate's rule): on the null
    # agent, one whose check passed in any run.
    stable = [n for n, k in passes.items() if k == 0]
    what = "fail on the null agent, as they must" if args.null else "left the project as asked"
    print(f"{sum(1 for k in passes.values() if k == runs)} of {len(tasks)} task(s) {what}"
          + (f" in every one of {runs} runs; {len(stable)} in none" if runs > 1 else "")
          + f" - {time.monotonic() - began:.0f} s; detail: {root / 'suite.json'}")
    failed = bool(stable) if runs > 1 else bool(wrong_total)
    if not (args.keep or args.live or failed):
        shutil.rmtree(root, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
