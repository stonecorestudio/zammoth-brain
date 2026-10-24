"""The chat rig: chat turns through the REAL app and the REAL brain, scripted, with no mouse.

Phase 56. Until 2026-10-12 a chat turn ran in the real app only through the screen (the
drive-app HID scripts): the GUI app, a live model on James's plan, no way to press Stop or
answer an approval box, and a failed turn that never ended for the watcher. So what the
engineer does when a turn goes WRONG - a rate limit, a dropped connection, a refusal, a Stop,
a dead brain - was proven on the chat loop in isolation and never on the app around it.

This drives `MixAgentDAW --chat-rig` (daw/Source/ChatRigLeg.h): a real `DawComponent` over a
scratch project, the source brain started as the app starts it, each prompt sent by the
composer's own `send()`. The model behind the brain is `server/tests/fake_anthropic.py`,
standalone, strict, with its script required - a SCENARIO is a list of turns, each a prompt
and the model's scripted replies ("a tape").

    python -m orchestrator.chat_rig --list
    python -m orchestrator.chat_rig --run mute_and_undo
    python -m orchestrator.chat_rig --all [--keep]         # every scenario; --keep = leave the sandbox
    python -m orchestrator.chat_rig --sweep [--jobs 4]     # every command through the agent's door
    python -m orchestrator.chat_rig --sweep --update-census
    python -m orchestrator.chat_rig --ladder                       # the 14 rungs on the scripted model (free)
    python -m orchestrator.chat_rig --ladder --live api --reps 5 --billed   # BILLED: a live model
    python -m orchestrator.chat_rig --ladder --check-budget        # ...held to orchestrator/ladder_budget.json
    python -m orchestrator.chat_rig --replay <project>             # a recorded session's tapes, model taken out

Free: no model is reached. Exit 0 = every scenario passed, 1 = one did not, 2 = the console
binary is not built.

What a scenario's checks read: the leg's per-turn result (did the turn END, the working
line, every feed block, the edit snapshot, the undo depth, the approval boxes, the longest
message-thread stall), the brain's own timing record, and the fake (what was requested, what
the strict rules refused, what no step was left for). A tool result that differs from the
tape (`results`) is `diverged`: the rig then says so instead of passing on a refusal.

WHERE THE RIG IS NOT THE PRODUCT - it never stands in for a run of the packaged app: no
window (an approval box is answered by a seam, never drawn), the SOURCE brain (no pinned API
host, no licence door, no keychain: `MIX_AGENT_KEYCHAIN=none`), the console build's own
differences, no menus / quit / recovery prompt. And the model's words are the tape's.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

REPO = Path(__file__).resolve().parents[1]
BINARY = REPO / "daw" / "build.noindex" / "MixAgentDAW_artefacts" / "Release" / "MixAgentDAW"
STEMS = REPO / "test_projects" / "suno_demo"
HOME_MARKER = "chat-rig-home"
EXIT_NOT_BUILT = 2


def _fake():
    sys.path.insert(0, str(REPO / "server"))
    from tests import fake_anthropic      # the fake lives with the tests that also use it
    return fake_anthropic


# ---------------------------------------------------------------- a scenario

def call(call_id: str, command: str, **args: Any) -> tuple:
    """A `daw_command` tool call as a fake block."""
    return ("tool_use", call_id, "daw_command", {"command": command, "args": args})


def tool(call_id: str, name: str, **args: Any) -> tuple:
    return ("tool_use", call_id, name, args)


def say(text: str) -> dict:
    return {"blocks": [("thinking", ""), ("text", text)]}


def ask(*blocks: tuple, **step: Any) -> dict:
    """A round in which the model calls tools."""
    return {"blocks": [("thinking", ""), *blocks], **step}


@dataclass
class Turn:
    prompt: str
    tape: list[dict]                       # the fake's steps for this turn, in order
    #: what each tool call of the tape must come back as: {"toolu_1": "ok" | "error"}
    results: dict[str, str] = field(default_factory=dict)
    leg: dict = field(default_factory=dict)        # stop / kill_brain / confirm / after / mode ...


@dataclass
class Scenario:
    name: str
    about: str
    turns: list[Turn]
    check: Callable[["Outcome"], list[str]]        # the problems found; [] = pass
    ceiling_s: float = 90.0
    env: dict = field(default_factory=dict)        # extra environment for the console and its brain
    light: bool = False                            # a long script: per-turn rows + edit version only
    #: turns that are MEANT to leave tool rows without a result (a Stop, a dead brain): the
    #: standing check "no row still waits" is then the scenario's own to make
    unsettled_ok: bool = False


@dataclass
class Outcome:
    """Everything a scenario's checks may read."""
    turns: list[dict]                      # the leg's result per turn
    rig: dict
    timing: list[dict]                     # the brain's timing record, a line a turn
    requests: list[dict]                   # every body the fake received
    violations: list[str]
    unscripted: int
    left: int                              # steps of the tape nobody asked for
    fake_s: list[float]                    # how long the fake itself took over each request
    log: str
    project: Path

    def tracks(self, turn: int = -1, key: str = "snapshot") -> dict[str, dict]:
        return {t.get("name"): t for t in (self.turns[turn].get(key) or {}).get("tracks") or []}

    def rows(self, turn: int = -1, kind: str = "tool") -> list[dict]:
        t = self.turns[turn]
        return [b for b in (t.get("feed") or [])[int(t.get("first_block") or 0):]
                if b.get("k") == kind]


def tool_results(requests: list[dict]) -> dict[str, dict]:
    """Every tool result the brain sent back to the model, by call id: {ok, text}."""
    out: dict[str, dict] = {}
    for body in requests:
        for msg in body.get("messages") or []:
            content = msg.get("content")
            if msg.get("role") != "user" or not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    text = block.get("content")
                    if isinstance(text, list):
                        text = " ".join(str(b.get("text", "")) for b in text if isinstance(b, dict))
                    out[str(block.get("tool_use_id"))] = {"ok": not block.get("is_error"),
                                                          "text": str(text or "")}
    return out


def standing(o: Outcome, scenario: Scenario) -> list[str]:
    """What must hold of EVERY scenario, whatever it is about (Phase 56's six, less the two a
    scenario states for itself - the sentence and the undo)."""
    bad: list[str] = []
    for i, t in enumerate(o.turns, 1):
        if not t.get("ended"):
            bad.append(f"turn {i} did not end (the run thread never came free)")
        if t.get("working_line"):
            bad.append(f"turn {i}: the working line still runs after the turn")
        pending = [r.get("key") or r.get("verb") for r in o.rows(i - 1) if r.get("pending")]
        if pending and not scenario.unsettled_ok:
            bad.append(f"turn {i}: tool rows still waiting for a result: {pending}")
    if o.violations:
        bad.append(f"the strict rules refused a request: {o.violations[:2]}")
    if o.unscripted:
        bad.append(f"{o.unscripted} request(s) the tape had no step for")
    if o.left:
        bad.append(f"{o.left} step(s) of the tape were never asked for")
    got = tool_results(o.requests)
    for turn in scenario.turns:
        for call_id, want in turn.results.items():
            if call_id not in got:
                bad.append(f"diverged: the tape's call {call_id} never came back")
            elif got[call_id]["ok"] != (want == "ok"):
                bad.append(f"diverged: call {call_id} came back "
                           f"{'ok' if got[call_id]['ok'] else 'an error'} where the tape says "
                           f"{want} ({got[call_id]['text'][:120]})")
    return bad


# ---------------------------------------------------------------- the scenarios

def _mute_and_undo(o: Outcome) -> list[str]:
    bad = []
    if not o.tracks(0).get("vocals", {}).get("muted"):
        bad.append("after the turn `vocals` is not muted in the snapshot")
    rows = o.rows(0)
    if len(rows) != 1 or rows[0].get("key") != "track.set_mute" or rows[0].get("failed"):
        bad.append(f"the feed should hold one track.set_mute row that did not fail: {rows}")
    if not any("muted" in (b.get("t") or "").lower() for b in o.rows(0, "prose")):
        bad.append("the model's reply is not in the feed")
    if o.tracks(0, "after_snapshot").get("vocals", {}).get("muted"):
        bad.append("Undo Agent Move did not unmute `vocals`")
    if len(o.timing) != 1 or o.timing[0].get("failed"):
        bad.append(f"the brain should have recorded one finished turn: {o.timing}")
    else:
        t = o.timing[0]
        if t.get("v") != 3 or "first_activity_s" not in t:
            bad.append("the timing record is not v3 or has no first activity")
        if "track.set_mute" not in (t.get("commands") or {}):
            bad.append("the timing record has no wire row for track.set_mute")
        elif "app_s" not in t["commands"]["track.set_mute"]:
            bad.append("the app did not stamp its own time on the reply")
    return bad


MUTE = Turn("mute the vocals",
            [ask(call("toolu_1", "track.set_mute", track="vocals", muted=True)),
             say("Vocals muted.")],
            results={"toolu_1": "ok"},
            leg={"after": [{"key": "agent.undo", "args": {}}, {"key": "@pump", "ms": 1500}]})

def errors(o: Outcome, turn: int) -> list[str]:
    """The failure lines the turn left in the feed."""
    return [b.get("t") or "" for b in o.rows(turn, "status") if b.get("error")]


def retry_offered(o: Outcome, turn: int) -> bool:
    return any(b.get("retry") for b in o.rows(turn, "verbs"))


def said(o: Outcome, turn: int, needle: str, kinds: tuple = ("status", "prose", "meta")) -> bool:
    return any(needle.lower() in (b.get("t") or "").lower()
               for kind in kinds for b in o.rows(turn, kind))


def muted(o: Outcome, turn: int, track: str = "vocals", key: str = "snapshot") -> bool:
    return bool(o.tracks(turn, key).get(track, {}).get("muted"))


def failed_with(needle: str, retry: Optional[bool], turn: int = 0,
                then_works: bool = True) -> Callable[[Outcome], list[str]]:
    """The turn ENDED IN THE FEED with a sentence holding `needle`; a retry verb is offered
    exactly when `retry` (None = not checked); and (the fifth of the six) the next turn of
    the scenario works."""
    def check(o: Outcome) -> list[str]:
        bad = []
        if not any(needle.lower() in e.lower() for e in errors(o, turn)):
            bad.append(f"no failure line holding {needle!r}: {errors(o, turn) or o.rows(turn, 'status')}")
        if retry is not None and retry_offered(o, turn) != retry:
            bad.append(f"a retry verb is {'not ' if retry else ''}offered, and should "
                       f"{'' if retry else 'not '}be")
        if then_works and len(o.turns) > turn + 1:
            if not muted(o, turn + 1):
                bad.append("the NEXT turn did not work: `vocals` is not muted after it")
            if errors(o, turn + 1):
                bad.append(f"the next turn left a failure line: {errors(o, turn + 1)}")
        return bad
    return check


def then_mute(n: int = 2) -> Turn:
    """A plain turn after the one under test: 'the next turn works'."""
    return Turn("mute the vocals",
                [ask(call(f"toolu_{n}1", "track.set_mute", track="vocals", muted=True)),
                 say("Vocals muted.")], results={f"toolu_{n}1": "ok"})


#: Anthropic's own bodies (the rate-limits page), as the fake's steps.
RATE_LIMITED = {"error": [429, "This request would exceed your organization's rate limit of "
                               "400,000 output tokens per minute.", "rate_limit_error"],
                "headers": {"retry-after": "1"}}
SPEND_CAP = {"error": [429, "You have reached your API usage limits: your organization has crossed "
                            "its monthly API usage threshold, set based on your organization's API "
                            "tier. You will regain access on 2026-11-01 at 00:00 UTC.",
                       "rate_limit_error"],
             "details": {"error_code": "enforced_spend_limit_reached"}}
OWN_LIMIT = {"error": [400, "You have reached your specified API usage limits. You will regain "
                            "access on 2026-11-01 at 00:00 UTC.", "invalid_request_error"]}
BAD_KEY = {"error": [401, "invalid x-api-key", "authentication_error"]}
OVERLOADED = {"error": [529, "Overloaded", "overloaded_error"]}
HELD = {"blocks": [("text", "Pulling the vocal"), ("text", " down now")],
        "hold": "mid", "hold_after": 3, "hold_max_s": 40}


def _rate_limited(o: Outcome) -> list[str]:
    bad = [] if muted(o, 0) else ["the turn did not carry on after the wait: `vocals` is not muted"]
    if errors(o, 0):
        bad.append(f"a rate limit the SDK waited out left a failure line: {errors(o, 0)}")
    if len(o.requests) != 3:
        bad.append(f"{len(o.requests)} requests where 3 were expected (the 429, the call, the reply)")
    return bad


def _dropped_before(o: Outcome) -> list[str]:
    bad = [] if muted(o, 0) else ["the round asked again did not land: `vocals` is not muted"]
    if errors(o, 0):
        bad.append(f"a round that was asked again left a failure line: {errors(o, 0)}")
    if len(o.rows(0)) != 1:
        bad.append(f"the dead round's call must not draw a row of its own: {o.rows(0)}")
    return bad


def _dropped_after_a_command(o: Outcome) -> list[str]:
    bad = failed_with("connection to Anthropic dropped", retry=False, then_works=False)(o)
    if not muted(o, 0):
        bad.append("the command that ran before the drop was lost: `vocals` is not muted")
    if not said(o, 0, "already applied"):
        bad.append("the failure does not say the turn's earlier steps were applied")
    if muted(o, 0, key="after_snapshot"):
        bad.append("Undo Agent Move did not take the cut-off turn's mute back")
    return bad


def _stopped(o: Outcome) -> list[str]:
    bad = []
    if not o.turns[0].get("stopped"):
        bad.append("the composer's button was not Stop while the turn ran")
    if not said(o, 0, "Stopped"):
        bad.append(f"the feed does not say the turn was stopped: {o.turns[0].get('kinds')}")
    if o.turns[0].get("seconds", 99) > 15:
        bad.append(f"Stop took {o.turns[0]['seconds']:.1f} s to end the turn")
    if not muted(o, 1):
        bad.append("the turn after a Stop did not work")
    return bad


def _stopped_after_a_command(o: Outcome) -> list[str]:
    bad = []
    if not o.turns[0].get("stopped") or not said(o, 0, "Stopped"):
        bad.append(f"the turn was not stopped and said so: {o.turns[0].get('kinds')}")
    if not muted(o, 0):
        bad.append("what ran before Stop was lost: `vocals` is not muted")
    if muted(o, 0, key="after_snapshot"):
        bad.append("Undo Agent Move did not take the stopped turn's mute back")
    return bad


def _brain_killed(o: Outcome) -> list[str]:
    bad = []
    if not o.turns[0].get("killed"):
        bad.append("the brain was not killed (no turn was running at the cue)")
    if not errors(o, 0):
        bad.append(f"a turn whose brain died left no failure line: {o.rows(0, 'status')}")
    if not o.turns[1].get("brain_up") or not muted(o, 1):
        bad.append("the next turn did not bring a new brain up and work")
    return bad


def _brain_killed_after_a_command(o: Outcome) -> list[str]:
    bad = _brain_killed(o)[:2]
    if not muted(o, 0):
        bad.append("the command that ran before the brain died was lost")
    # the new brain is told what landed, and must not do it again
    again = [r for r in o.rows(1) if r.get("key") == "track.set_mute"]
    if again:
        bad.append(f"the next turn repeated the dead turn's command: {again}")
    return bad


def _refused_by_the_model(o: Outcome) -> list[str]:
    bad = [] if said(o, 0, "declined") else [f"the feed does not say the model declined: {o.rows(0, 'prose')}"]
    if muted(o, 0) or o.rows(0):
        bad.append("a declined round ran something")
    return bad + ([] if muted(o, 1) else ["the turn after a decline did not work"])


def _cut_off_call(o: Outcome) -> list[str]:
    bad = [] if said(o, 0, "nothing was applied") else ["the feed does not say nothing was applied"]
    if muted(o, 0):
        bad.append("half a call ran: `vocals` is muted")
    return bad


def _invented(o: Outcome) -> list[str]:
    rows = o.rows(0)
    bad = []
    if len(rows) != 1 or not rows[0].get("failed"):
        bad.append(f"the invented call should be one FAILED row: {rows}")
    if not said(o, 0, "can't do that", ("prose",)):
        bad.append("the model's reply after the refusal is not in the feed")
    return bad


def _refused_at_the_door(needle: str) -> Callable[[Outcome], list[str]]:
    """A command turned away before its handler - by the app (unknown, a confirm answered
    No, the master's safety) or by the brain (a routing command it refused itself) - is
    still something the engineer TRIED: one row, failed, with the reason under it. Until
    2026-10-12 it left no trace but whatever the model chose to say."""
    def check(o: Outcome) -> list[str]:
        rows, bad = o.rows(0), []
        if len(rows) != 1 or not rows[0].get("failed") or rows[0].get("pending"):
            bad.append(f"the refused command should be ONE settled, failed row: {rows}")
        elif needle.lower() not in (rows[0].get("detail") or "").lower():
            bad.append(f"the row does not carry the reason ({needle!r}): {rows[0].get('detail')!r}")
        if not o.rows(0, "prose"):
            bad.append("the model's reply after the refusal is not in the feed")
        return bad
    return check


def _confirm(answer: str) -> Callable[[Outcome], list[str]]:
    def check(o: Outcome) -> list[str]:
        t, rows, bad = o.turns[0], o.rows(0), []
        if len(t.get("confirms") or []) != 1:
            bad.append(f"one approval box was expected, {len(t.get('confirms') or [])} were raised")
        elif "undo" not in t["confirms"][0].lower():
            bad.append(f"the box does not say it is not covered by undo: {t['confirms'][0][:120]}")
        if answer == "no":
            bad += _refused_at_the_door("the user declined")(o)
        elif len(rows) != 1 or rows[0].get("failed"):
            bad.append(f"after 'yes' the command's row should be there and not failed: {rows}")
        timing = (o.timing[0] if o.timing else {})
        if not 0.2 <= float(timing.get("confirm_wait_s") or 0.0) <= 5.0:
            bad.append(f"the wait at the box is not in the timing record: {timing.get('confirm_wait_s')}")
        return bad
    return check


def _spend_pause(o: Outcome) -> list[str]:
    bad = []
    if not o.rows(0, "question"):
        bad.append(f"no question card when the turn passed its spend pause: {o.turns[0].get('kinds')}")
    if not o.turns[0].get("answered"):
        bad.append("the card's 'Stop here' could not be clicked")
    if muted(o, 0):
        bad.append("the turn went on spending after 'Stop here'")
    return bad


def _refused_after_new_return(o: Outcome) -> list[str]:
    """Found by the command sweep: the command after `return.add` - any command, a refused one
    too - left an undo step of its own. Before a reply the app sorts every lane, the sort
    also puts a track's OTHER children in the engine's order, and a track nothing had sorted
    yet (a new return has no clips) was reordered through the undo manager."""
    t = o.turns[1]
    wrote = [w for w in (t.get("tree_writes") or []) if "INPUTDEVICE" not in w]
    bad = []
    if wrote:
        bad.append(f"a turn whose only command was refused wrote to the edit: {wrote[:3]}")
    if t.get("undo_after") != t.get("undo_before"):
        bad.append(f"...and left an undo step ({t.get('undo_before')} -> {t.get('undo_after')})")
    return bad


def _render_and_read(o: Outcome) -> list[str]:
    """A sound rendered and READ in one round: the tool result the model is sent holds the
    file and what it measures as, and the project did not move."""
    bad: list[str] = []
    got = tool_results(o.requests).get("toolu_1") or {}
    try:
        res = json.loads(got.get("text") or "")
    except ValueError:
        return [f"the render's tool result is not JSON: {str(got.get('text'))[:160]}"]
    sounds = res.get("sounds") or []
    if len(sounds) != 1 or not sounds[0].get("file"):
        return [f"one rendered sound with its file was expected: {str(res)[:200]}"]
    profile = sounds[0].get("profile") or {}
    if profile.get("pitch_midi") != 45:
        bad.append(f"the result does not carry the file's measured pitch (A2 = 45): {profile.get('pitch_midi')!r} "
                   f"{sounds[0].get('profile_error') or ''}")
    for field_name in ("attack_ms", "loudness_lufs", "band_share_pct", "centroid_hz"):
        if field_name not in profile:
            bad.append(f"the profile has no {field_name}")
    rows = o.rows(0)
    if len(rows) != 1 or rows[0].get("key") != "sound.render" or rows[0].get("failed"):
        bad.append(f"the feed should hold one sound.render row that did not fail: {rows}")
    elif len(rows[0].get("sounds") or []) != 1:
        bad.append(f"the row lists no sound to play: {rows[0]}")
    t = o.turns[0]
    wrote = [w for w in (t.get("tree_writes") or []) if "INPUTDEVICE" not in w]
    if wrote or t.get("undo_after") != t.get("undo_before"):
        bad.append(f"a render wrote to the project: {wrote[:3]} undo {t.get('undo_before')} -> {t.get('undo_after')}")
    return bad


def _pack_and_read(o: Outcome) -> list[str]:
    """A sound added to a sample pack: the tool result names the FINISHED file in the pack
    and carries what that file measures as, the file and the manifest are on the disk, and
    the project did not move."""
    bad: list[str] = []
    got = tool_results(o.requests).get("toolu_2") or {}
    try:
        res = json.loads(got.get("text") or "")
    except ValueError:
        return [f"the add's tool result is not JSON: {str(got.get('text'))[:160]}"]
    sounds = res.get("sounds") or []
    if len(sounds) != 1 or not sounds[0].get("file"):
        return [f"one added sound with its file was expected: {str(res)[:240]}"]
    path = Path(str(sounds[0]["file"])).expanduser()
    if path.name != "Bass_Tone_A1.wav" or path.parent.name != "Bass" or path.parent.parent.name != "One Shots":
        bad.append(f"the file is not named and filed for what it is: {path}")
    if not path.is_file() or not (path.parents[2] / "pack.json").is_file():
        bad.append(f"the file or its pack's manifest is not on the disk: {path}")
    else:
        rows = json.loads((path.parents[2] / "pack.json").read_text()).get("sounds") or []
        if [r.get("name") for r in rows] != ["Tone"]:
            bad.append(f"the manifest's rows: {rows}")
    profile = sounds[0].get("profile") or {}
    if profile.get("pitch_midi") != 45:
        bad.append(f"the result does not carry the FINISHED file's measured pitch (45): {profile.get('pitch_midi')!r} "
                   f"{sounds[0].get('profile_error') or ''}")
    if not isinstance(sounds[0].get("finish"), dict) or sounds[0].get("peak_dbtp") is None:
        bad.append(f"the result does not say how the sound was finished: {str(sounds[0])[:200]}")
    elif abs(float(sounds[0]["peak_dbtp"]) + 1.0) > 0.15:
        bad.append(f"the finished peak is not at the ceiling: {sounds[0]['peak_dbtp']}")
    rows = o.rows(0)
    keys = [r.get("key") for r in rows]
    if keys != ["pack.create", "pack.add"] or any(r.get("failed") for r in rows):
        bad.append(f"the feed should hold a pack.create and a pack.add row that did not fail: {rows}")
    elif len(rows[1].get("sounds") or []) != 1:
        bad.append(f"the pack.add row lists no sound to play: {rows[1]}")
    t = o.turns[0]
    wrote = [w for w in (t.get("tree_writes") or []) if "INPUTDEVICE" not in w]
    if wrote or t.get("undo_after") != t.get("undo_before"):
        bad.append(f"a pack command wrote to the project: {wrote[:3]} undo {t.get('undo_before')} -> {t.get('undo_after')}")
    return bad


def _soak(o: Outcome) -> list[str]:
    bad = []
    if len(o.timing) != len(o.turns):
        bad.append(f"{len(o.timing)} timing lines for {len(o.turns)} turns")
        return bad
    prep = [float((t.get("marks_s") or {}).get("prep") or 0.0) for t in o.timing]
    # The brain's and the app's OWN seconds: the wall less what the scripted API itself took
    # to answer (the fake re-reads the whole conversation for its checks, and that grows too).
    per_turn = len(o.fake_s) // max(1, len(o.timing))
    own = [max(0.0, float(t.get("wall_s") or 0.0) - sum(o.fake_s[i * per_turn:(i + 1) * per_turn]))
           for i, t in enumerate(o.timing)]
    head, tail = sorted(own[1:6])[2], sorted(own[-5:])[2]
    print(f"     soak: {len(own)} turns; the brain's and the app's own seconds, median of the first "
          f"five {head:.2f}, of the last five {tail:.2f}; prep first {prep[1]:.2f} s, last "
          f"{prep[-1]:.2f} s; longest UI stall "
          f"{max(float(t.get('ui_stall_ms') or 0.0) for t in o.turns):.0f} ms")
    if tail > max(3.0 * head, head + 1.0):
        bad.append(f"a turn got slower as the session grew: {head:.2f} s -> {tail:.2f} s")
    if muted(o, -1) != (len(o.turns) % 2 == 1):
        bad.append("the last turn's mute state is not what the session's turns add up to")
    return bad


def _soak_turns(n: int) -> list[Turn]:
    return [Turn(f"turn {k}: {'mute' if k % 2 else 'unmute'} the vocals",
                 [ask(call(f"toolu_s{k}", "track.set_mute", track="vocals", muted=bool(k % 2))),
                  say(f"Done ({k}).")], results={f"toolu_s{k}": "ok"})
            for k in range(1, n + 1)]


UNDO = {"after": [{"key": "agent.undo", "args": {}}, {"key": "@pump", "ms": 1500}]}

SCENARIOS: dict[str, Scenario] = {s.name: s for s in [
    Scenario("mute_and_undo",
             "the rig proves itself: one command through the real handler, its row, its "
             "reply, the wire's timing, and Undo Agent Move taking it back",
             [MUTE], _mute_and_undo),
    # The same tape, told the call fails. It must NOT pass: a rig that cannot say "diverged"
    # passes while the app refuses everything it is asked.
    Scenario("mute_expected_to_fail",
             "negative control: the tape claims the mute is refused - the rig reports `diverged`",
             [Turn(MUTE.prompt, MUTE.tape, results={"toolu_1": "error"})], lambda o: []),

    # ---- the API says no -----------------------------------------------------------------------
    Scenario("rate_limited", "a 429 with retry-after is waited out and the turn carries on",
             [Turn("mute the vocals", [RATE_LIMITED, *MUTE.tape], results={"toolu_1": "ok"})],
             _rate_limited),
    Scenario("spend_cap", "the monthly spend cap: its sentence, no retry verb, the next turn works",
             [Turn("mute the vocals", [SPEND_CAP] * 5), then_mute()],
             failed_with("monthly spend cap", retry=False)),
    Scenario("own_spend_limit", "a limit the user set: its sentence at once, no retry verb",
             [Turn("mute the vocals", [OWN_LIMIT]), then_mute()],
             failed_with("the spend limit you set", retry=False)),
    Scenario("rejected_key", "a 401: the key sentence, no retry verb",
             [Turn("mute the vocals", [BAD_KEY]), then_mute()],
             failed_with("API key was rejected", retry=False)),
    Scenario("overloaded", "529 on every attempt: a sentence that can be retried",
             [Turn("mute the vocals", [OVERLOADED] * 5), then_mute()],
             failed_with("overloaded", retry=True)),

    # ---- the connection dies -------------------------------------------------------------------
    Scenario("dropped_before_anything", "the socket dies before anything was shown: asked again, once run",
             [Turn("mute the vocals", [{**MUTE.tape[0], "cut_after": 1}, *MUTE.tape],
                   results={"toolu_1": "ok"})], _dropped_before),
    Scenario("dropped_after_words", "the socket dies after words were shown: the turn ends, retry offered",
             [Turn("mute the vocals", [{"blocks": [("text", "Working on it"), ("text", " now")],
                                        "cut_after": 4, "pace_s": 0.05}]), then_mute()],
             failed_with("connection to Anthropic dropped", retry=True)),
    Scenario("dropped_after_a_command",
             "the socket dies after a command ran: no retry (it would repeat), and undo takes it back",
             [Turn("mute the vocals",
                   [MUTE.tape[0], {"blocks": [("text", "Muted"), ("text", " it")], "cut_after": 4,
                                   "pace_s": 0.05}], results={"toolu_1": "ok"}, leg=UNDO)],
             _dropped_after_a_command),

    # ---- the user stops it, the brain dies ------------------------------------------------------
    Scenario("stop_mid_sentence", "Stop while the model is mid-sentence, then a turn that works",
             [Turn("pull the vocal down", [HELD], leg={"stop": {"after_text": True}}), then_mute()],
             _stopped),
    Scenario("stop_after_a_command", "Stop after one command ran: it stays, and undo takes it back",
             [Turn("mute the vocals and then tidy up", [MUTE.tape[0], HELD], results={"toolu_1": "ok"},
                   leg={"stop": {"after_text": True}, **UNDO})],
             _stopped_after_a_command),
    Scenario("brain_killed_mid_sentence", "the brain dies mid-sentence: the turn ends, the next starts a new one",
             [Turn("pull the vocal down", [HELD], leg={"kill_brain": {"after_text": True}}), then_mute()],
             _brain_killed, unsettled_ok=True),
    Scenario("brain_killed_after_a_command",
             "the brain dies after a command ran: the next brain is told, and does not repeat it",
             [Turn("mute the vocals and then tidy up", [MUTE.tape[0], HELD], results={"toolu_1": "ok"},
                   leg={"kill_brain": {"after_text": True}}),
              Turn("carry on", [say("The mute from before is in place; nothing else to do.")])],
             _brain_killed_after_a_command, unsettled_ok=True),

    # ---- the model gets it wrong ----------------------------------------------------------------
    Scenario("declined", "a `refusal` stop: said in the feed, nothing ran, the next turn works",
             [Turn("mute the vocals", [{"blocks": [], "stop": "refusal",
                                        "stop_details": {"type": "refusal", "category": "cyber"}}]),
              then_mute()], _refused_by_the_model),
    Scenario("call_cut_off", "a tool call cut off by max_tokens runs nothing",
             [Turn("mute the vocals", [{"blocks": [("tool_use_raw", "toolu_1", "daw_command",
                                                    '{"command": "track.set_mute", "args": {"track": "vo')],
                                        "stop": "max_tokens"}])], _cut_off_call),
    Scenario("invented_tool", "a tool that does not exist is one failed row, and the turn goes on",
             [Turn("delete everything", [ask(tool("toolu_1", "delete_everything", sure=True)),
                                         say("I can't do that: there is no such tool.")],
                   results={"toolu_1": "error"})], _invented),
    Scenario("invented_command", "a command the app does not have is refused by the app",
             [Turn("explode the vocals", [ask(call("toolu_1", "track.explode", track="vocals")),
                                          say("I can't do that: the app has no such command.")],
                   results={"toolu_1": "error"})], _refused_at_the_door("unknown command")),
    Scenario("master_safety_refused", "the agent may not switch the master's limiter off: refused, and shown",
             [Turn("turn the master limiter off",
                   [ask(call("toolu_1", "master.limiter", on=False)),
                    say("I can't do that: the master's safety limiter stays on.")],
                   results={"toolu_1": "error"})], _refused_at_the_door("limiter")),
    Scenario("refused_by_the_brain", "a routing command the brain refuses itself is shown too",
             [Turn("route the vocalz into the Band bus",
                   [ask(call("toolu_1", "route.stem", stem="vocalz", bus="Band")),
                    say("There is no track called vocalz.")],
                   results={"toolu_1": "error"})], _refused_at_the_door("no track named")),

    # ---- a person decides -----------------------------------------------------------------------
    Scenario("confirm_declined", "an approval box answered No: the command does not run, the wait is timed",
             [Turn("switch OSC off", [ask(call("toolu_1", "osc.set", enabled=False)),
                                      say("Left as it was.")], results={"toolu_1": "error"},
                   leg={"confirm": "no", "confirm_delay_ms": 600})], _confirm("no")),
    Scenario("confirm_accepted", "an approval box answered Yes: the command runs, the wait is timed",
             [Turn("switch OSC off", [ask(call("toolu_1", "osc.set", enabled=False)),
                                      say("OSC is off.")],
                   leg={"confirm": "yes", "confirm_delay_ms": 600})], _confirm("yes")),
    Scenario("spend_pause", "a turn that passes its spend pause asks, and 'Stop here' stops it",
             [Turn("mute the vocals",
                   [{**MUTE.tape[0], "blocks": [("thinking", ""), tool("toolu_0", "analyze_session")],
                     "usage": {"input_tokens": 900_000, "cache_read_input_tokens": 0,
                               "cache_creation_input_tokens": 0}, "out": 50_000},
                    say("Stopped where you asked.")],
                   leg={"answer": 1})],
             _spend_pause, env={"MIX_AGENT_TURN_PAUSE_USD": "0.5"}),

    Scenario("refused_after_new_return",
             "a refused command in the turn after return.add writes nothing and leaves no undo step",
             [Turn("add a return", [ask(call("toolu_1", "return.add")), say("Added.")],
                   results={"toolu_1": "ok"}),
              Turn("set a level", [ask(call("toolu_2", "track.set_gain")), say("That needs a track.")],
                   results={"toolu_2": "error"})],
             _refused_after_new_return),

    # ---- a sound, rendered and read (Phase 58) ---------------------------------------------------
    Scenario("render_and_read",
             "sound.render: the file's measured profile is in the SAME tool result, the row lists "
             "the sound, and the project did not move",
             [Turn("render INIT on a low A and tell me what note it reads as",
                   [ask(call("toolu_1", "sound.render",
                             sounds=[{"preset": "INIT",
                                      "notes": [{"pitch": 45, "start_beat": 0, "length_beats": 1}]}])),
                    say("It reads as A2 (note 45).")],
                   results={"toolu_1": "ok"})],
             _render_and_read),

    # ---- a sound, finished and kept in a pack (Phase 59) -------------------------------------------
    Scenario("pack_and_read",
             "pack.add: the FINISHED file's profile is in the same tool result, the file and the "
             "manifest are on the disk, the row lists the sound, and the project did not move",
             [Turn("start a pack called Rig Pack and keep a plain low A in it as Tone",
                   [ask(call("toolu_1", "pack.create", name="Rig Pack")),
                    ask(call("toolu_2", "pack.add", pack="Rig Pack",
                             sounds=[{"name": "Tone", "kind": "bass",
                                      "patch": {"set": {"osc1.wave": "sin", "voice.phase": "reset"}},
                                      "notes": [{"pitch": 45, "start_beat": 0, "length_beats": 1}]}])),
                    say("Tone is in Rig Pack.")],
                   results={"toolu_1": "ok", "toolu_2": "ok"})],
             _pack_and_read),

    # ---- a long session ---------------------------------------------------------------------------
    Scenario("soak", "30 turns in one session: no turn slower for the session's length",
             _soak_turns(30), _soak, ceiling_s=60.0),
]}
#: Scenarios that pass only by FAILING (a control proves the rig can).
CONTROLS = {"mute_expected_to_fail": "diverged"}


# ---------------------------------------------------------------- running one

def sandbox_env(work: Path, port: int) -> dict[str, str]:
    """The console's environment: the fake as the model, a home of its own, no keychain, no
    control plane, the source brain."""
    fake = _fake()
    home = work / HOME_MARKER
    for sub in ("Library/Application Support", "Library/Caches", "Library/Logs", "Music", "tmp"):
        (home / sub).mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("ANTHROPIC_", "CLAUDE_CODE_", "MIX_AGENT_", "MIXAGENT_", "ZAMMOTH_"))}
    env.update(fake.env(port))
    env.update({
        "CFFIXED_USER_HOME": str(home),
        "TMPDIR": str(home / "tmp") + "/",
        "MIXAGENT_REPO": str(REPO),
        "ZAMMOTH_LOG_DIR": str(work / "logs"),
        "ZAMMOTH_PREFS_FILE": str(work / "settings" / "prefs.json"),
        "ZAMMOTH_ENGINE_SETTINGS_DIR": str(work / "settings" / "engine"),
        "ZAMMOTH_KEYMAP_FILE": str(work / "settings" / "key-mappings.xml"),
        "MIX_AGENT_DATA_DIR": str(work / "data"),
        "MIX_AGENT_KEYCHAIN": "none",                 # the brain never touches the login keychain
        "MIX_AGENT_CLOUD_URL": "http://127.0.0.1:1",   # nothing reaches the control plane
        "MIX_AGENT_LICENSE": "licensed",
        "MIX_AGENT_SEPARATOR": "fake",
    })
    if os.environ.get("MIX_AGENT_TAPE"):          # the caller wants this session kept as tapes
        env["MIX_AGENT_TAPE"] = os.environ["MIX_AGENT_TAPE"]
    for sub in ("logs", "settings/engine", "data"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    return env


#: What a LIVE run needs from the caller's environment, by transport: the one credential.
LIVE_CREDENTIAL = {"api": "ANTHROPIC_API_KEY", "subscription": "CLAUDE_CODE_OAUTH_TOKEN"}


def live_env(work: Path, transport: str) -> dict[str, str]:
    """The sandbox with a LIVE model behind the brain instead of the fake: BILLED. The one
    credential of that transport is carried over, the leg is told a live model is meant, and
    everything else of the sandbox stands (its own home, no keychain, no control plane - the
    knowledge book is the source checkout's own)."""
    name = LIVE_CREDENTIAL[transport]
    if not os.environ.get(name, "").strip():
        raise SystemExit(f"a live run on `{transport}` needs {name} in the environment")
    env = sandbox_env(work, 0)
    for gone in ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY"):
        env.pop(gone, None)
    env.update({name: os.environ[name], "MIX_AGENT_LLM_BACKEND": transport, "MIX_AGENT_RIG_LIVE": "1"})
    for passed in ("MIX_AGENT_PROMPT", "MIX_AGENT_EFFORT"):       # the prompt comparison, an effort
        if os.environ.get(passed):
            env[passed] = os.environ[passed]
    return env


def run_scenario(scenario: Scenario, work: Path, binary: Path = BINARY,
                 live: Optional[str] = None) -> tuple[list[str], Outcome]:
    """Run one scenario in `work` (made empty first). Returns the problems and the outcome.
    `live` names a transport: the model is then real and BILLED, and the tapes are not used."""
    F = _fake()
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    fake = F.FakeApi()
    fake.strict = fake.require_script = True
    fake.limits = {"anthropic-ratelimit-requests-limit": "1000",
                   "anthropic-ratelimit-input-tokens-limit": "2000000",
                   "anthropic-ratelimit-output-tokens-limit": "400000"}
    fake.script = [] if live else [dict(step) for turn in scenario.turns for step in turn.tape]
    _, port, shutdown = F.start(fake)
    out = work / "out"
    scratch = work / HOME_MARKER / "project"
    script = {"out": str(out), "scratch": str(scratch), "ceiling_s": scenario.ceiling_s,
              "light": scenario.light,
              "turns": [{"prompt": t.prompt, **t.leg} for t in scenario.turns]}
    (work / "script.json").write_text(json.dumps(script, indent=1))
    log = work / "console.log"
    began = time.monotonic()
    try:
        with log.open("w") as sink:
            code = subprocess.run(
                [str(binary), "--chat-rig", str(STEMS), str(work / "script.json")],
                cwd=REPO, stdout=sink, stderr=subprocess.STDOUT,
                env={**(live_env(work, live) if live else sandbox_env(work, port)), **scenario.env},
                timeout=scenario.ceiling_s * len(scenario.turns) + 240).returncode
    except subprocess.TimeoutExpired:
        code = -1
    finally:
        shutdown()
    turns = [json.loads(p.read_text()) for p in sorted(out.glob("turn_*.json"),
                                                       key=lambda p: int(p.stem.split("_")[1]))]
    rig = json.loads((out / "rig.json").read_text()) if (out / "rig.json").is_file() else {}
    timing_file = scratch / "runs" / "turn_timing.jsonl"
    timing = [json.loads(x) for x in timing_file.read_text().splitlines()] if timing_file.is_file() else []
    outcome = Outcome(turns=turns, rig=rig, timing=timing, requests=list(fake.requests),
                      violations=list(fake.violations), unscripted=len(fake.unscripted),
                      left=len(fake.script), fake_s=list(fake.spent_s), log=log.read_text(errors="replace"), project=scratch)
    bad: list[str] = []
    if code not in (0, 868 & 0xFF):       # 868 = a turn did not end: `standing` says which
        bad.append(f"the console leg exited {code} after {time.monotonic() - began:.0f} s: "
                   + next((x for x in reversed(outcome.log.splitlines()) if "FAIL" in x or "rig]" in x),
                          "no line"))
    if len(turns) != len(scenario.turns):
        bad.append(f"{len(turns)} of {len(scenario.turns)} turns left a result")
    else:
        bad += standing(outcome, scenario)
        try:
            bad += scenario.check(outcome)
        except Exception as exc:  # noqa: BLE001 - a check that cannot read its outcome has failed
            bad.append(f"the scenario's check raised {type(exc).__name__}: {exc}")
    return bad, outcome


# ---------------------------------------------------------------- the ladder

@dataclass
class Rung:
    """One short request of a known KIND, with the END STATE that says it was done. The tape
    is the scripted model's right answer: `--ladder` on the fake proves the checks against the
    real handlers before a billed run is spent on them."""
    kind: str
    prompt: str
    tape: list[dict]
    check: Callable[[Outcome, int], str]           # "" = done
    leg: dict = field(default_factory=dict)


def _t(o: Outcome, i: int, name: str) -> dict:
    return o.tracks(i).get(name, {})


def _is(name: str, field_: str, want, tol: float = 0.0) -> Callable[[Outcome, int], str]:
    def check(o: Outcome, i: int) -> str:
        got = _t(o, i, name).get(field_)
        fine = (abs(float(got) - float(want)) <= tol) if isinstance(want, float) and got is not None \
            else got == want
        return "" if fine else f"{name}.{field_} is {got!r}, not {want!r}"
    return check


def _answered(o: Outcome, i: int) -> str:
    """A question: words in the feed, and nothing in the project moved."""
    if not any((b.get("t") or "").strip() for b in o.rows(i, "prose")):
        return "no reply in the feed"
    wrote = [w for w in (o.turns[i].get("tree_writes") or []) if "INPUTDEVICE" not in w]
    return f"a question wrote to the edit: {wrote[:3]}" if wrote else ""


def _grouped_with_a_return(o: Outcome, i: int) -> str:
    drums, bass = _t(o, i, "drums").get("output"), _t(o, i, "bass").get("output")
    if not drums or drums == "master" or drums != bass:
        return f"drums and bass are not in one group (they play into {drums!r} and {bass!r})"
    if not (o.turns[i].get("snapshot") or {}).get("returns"):
        return "there is no return"
    return ""


def _every_fader(want: float) -> Callable[[Outcome, int], str]:
    def check(o: Outcome, i: int) -> str:
        off = {n: _t(o, i, n).get("gain_db") for n in ("vocals", "drums", "bass", "other")
               if abs(float(_t(o, i, n).get("gain_db", 99.0)) - want) > 0.05}
        return f"not at {want} dB: {off}" if off else ""
    return check


def _has_device(name: str, kind: str) -> Callable[[Outcome, int], str]:
    def check(o: Outcome, i: int) -> str:
        kinds = [d.get("type") for d in _t(o, i, name).get("devices") or []]
        return "" if kind in kinds else f"{name} has no {kind} device (it has {kinds})"
    return check


def _marker(name: str) -> Callable[[Outcome, int], str]:
    def check(o: Outcome, i: int) -> str:
        names = [m.get("name") for m in (o.turns[i].get("snapshot") or {}).get("markers") or []]
        return "" if name in names else f"no marker named {name!r} (markers: {names})"
    return check


def _tempo(bpm: float) -> Callable[[Outcome, int], str]:
    def check(o: Outcome, i: int) -> str:
        got = ((o.turns[i].get("snapshot") or {}).get("tempo") or {}).get("bpm")
        return "" if got is not None and abs(float(got) - bpm) < 0.01 else f"the tempo is {got}, not {bpm}"
    return check


def _one_more_track(o: Outcome, i: int) -> str:
    before = len(o.tracks(i - 1)) if i else 0
    return "" if len(o.tracks(i)) > before else f"no new track ({before} before, {len(o.tracks(i))} after)"


def _exported(name: str) -> Callable[[Outcome, int], str]:
    def check(o: Outcome, i: int) -> str:
        made = list((o.project / "bounces").glob(name)) if o.project else []
        return "" if made and made[0].stat().st_size > 1000 else f"bounces/{name} was not written"
    return check


def _set(n: int, track: str, **args) -> tuple:
    return call(f"toolu_L{n}", "track.set_gain" if "gain_db" in args else
                "track.set_pan" if "pan" in args else "track.set_mute", track=track, **args)


LADDER: list[Rung] = [
    Rung("one command", "mute the bass",
         [ask(_set(1, "bass", muted=True)), say("Bass muted.")], _is("bass", "muted", True)),
    Rung("one command", "add a marker called Drop at 8 seconds",
         [ask(call("toolu_L2", "marker.add", position_s=8.0, name="Drop")), say("Marker added.")],
         _marker("Drop")),
    Rung("one command", "set the tempo to 100 BPM",
         [ask(call("toolu_L3", "tempo.set", bpm=100)), say("Tempo is 100.")], _tempo(100.0)),
    Rung("one command", "unmute the bass",
         [ask(_set(4, "bass", muted=False)), say("Bass is back.")], _is("bass", "muted", False)),
    Rung("one level move", "bring the vocals up 2 dB",
         [ask(_set(5, "vocals", gain_db=2.0)), say("Vocals up 2 dB.")], _is("vocals", "gain_db", 2.0, 0.05)),
    Rung("one level move", "pan the track called other 30 percent to the left",
         [ask(_set(6, "other", pan=-0.3)), say("Panned 30L.")], _is("other", "pan", -0.3, 0.02)),
    Rung("answer from the snapshot", "which tracks are muted right now?",
         [say("None of them are muted.")], _answered),
    Rung("answer needing one read", "how loud is the mix, integrated?",
         [ask(tool("toolu_L8", "analyze_session")), say("It reads about -17 LUFS integrated.")],
         _answered),
    Rung("one device move", "put a high-pass at 100 Hz on the vocals",
         [ask(tool("toolu_L9", "propose", rationale="clear the rumble under the vocal",
                   predicted_impact=[],
                   actions=[{"type": "eq", "stem_id": "vocals",
                             "band": {"freq_hz": 100.0, "gain_db": 0.0, "q": 0.7,
                                      "shape": "hpf"}}])),
          say("Low end of the vocal is cleared.")], _has_device("vocals", "eq")),
    Rung("a task of several commands",
         "group the drums and bass into a bus, add a reverb return, and send the vocals to it at -12 dB",
         [ask(call("toolu_L10a", "group.create", stems=["drums", "bass"]),
              call("toolu_L10b", "return.add", kind="reverb")),
          ask(call("toolu_L10c", "send.set", source="vocals", **{"return": "rv"}, gain_db=-12.0)),
          say("Grouped, and the vocal has a reverb send.")], _grouped_with_a_return),
    Rung("a batch", "set every track's fader to -6 dB",
         [ask(*[_set(110 + n, name, gain_db=-6.0) for n, name in
                enumerate(("vocals", "drums", "bass", "other"))]), say("All four at -6.")],
         _every_fader(-6.0)),
    Rung("answer from the snapshot", "what is the tempo now, and is anything soloed?",
         [say("100 BPM, and nothing is soloed.")], _answered),
    Rung("render-bound", "bounce the vocals to a new track",
         [ask(call("toolu_L13", "track.bounce", track="vocals")), say("Printed.")], _one_more_track),
    Rung("render-bound", "export the mix as ladder.wav",
         [ask(call("toolu_L14", "session.export", format="wav", file="ladder.wav")), say("Exported.")],
         _exported("ladder.wav"), leg={"confirm": "yes", "confirm_delay_ms": 300}),
]


def ladder_scenario(live: bool = False) -> Scenario:
    def check(o: Outcome) -> list[str]:
        return [f"rung {i + 1} ({rung.kind}: {rung.prompt[:44]!r}): {why}"
                for i, rung in enumerate(LADDER) if i < len(o.turns)
                for why in [rung.check(o, i)] if why]
    return Scenario("ladder", "the latency ladder's 14 rungs, each held to the end state it asks for",
                    [Turn(r.prompt, [] if live else r.tape, leg=r.leg) for r in LADDER], check,
                    ceiling_s=240.0 if live else 120.0, unsettled_ok=live)


def ladder_report(runs: list[tuple[list[str], Outcome]]) -> tuple[str, dict]:
    """What the ladder is for: per KIND of request, the seconds a user waits and what they
    are made of - across every repetition, with the spread. (p95 by nearest rank: with five
    readings it is the slowest of the five, which is the honest answer for five.)"""
    from .turn_timing import percentile

    by_kind: dict[str, list[dict]] = {}
    done: dict[str, list[bool]] = {}
    for problems, o in runs:
        failed = {int(p.split(" ")[1]) - 1 for p in problems if p.startswith("rung ")}
        for i, rung in enumerate(LADDER):
            if i >= len(o.timing) or i >= len(o.turns):
                continue
            by_kind.setdefault(rung.kind, []).append({**o.timing[i], "_stall": o.turns[i].get("ui_stall_ms")})
            done.setdefault(rung.kind, []).append(i not in failed)
    table: dict[str, dict] = {}
    lines = [f"{'kind':28} {'n':>3} {'done':>5} {'wall p50':>9} {'p95':>7} {'first':>6} {'rounds':>6} "
             f"{'calls':>5} {'model':>6} {'tools':>6} {'brain':>6} {'$ p50':>7}", "-" * 104]
    for kind, rows in by_kind.items():
        def col(name, q=0.5):
            return percentile([float(r.get(name) or 0.0) for r in rows], q)
        brain = percentile([sum(float(v or 0.0) for v in (r.get("marks_s") or {}).values()) for r in rows], 0.5)
        row = {"n": len(rows), "done": sum(done[kind]), "wall_p50_s": col("wall_s"),
               "wall_p95_s": col("wall_s", 0.95), "first_activity_p50_s": col("first_activity_s"),
               "rounds_p50": col("rounds"), "tool_calls_p50": col("tool_calls"),
               "model_p50_s": col("model_s"), "tools_p50_s": col("tools_s"), "brain_p50_s": brain,
               "cost_p50_usd": col("cost_usd")}
        table[kind] = row
        lines.append(f"{kind:28} {row['n']:3d} {row['done']:3d}/{row['n']:<2d}{row['wall_p50_s']:9.1f} "
                     f"{row['wall_p95_s']:7.1f} {row['first_activity_p50_s']:6.1f} {row['rounds_p50']:6.0f} "
                     f"{row['tool_calls_p50']:5.0f} {row['model_p50_s']:6.1f} {row['tools_p50_s']:6.1f} "
                     f"{brain:6.1f} {row['cost_p50_usd']:7.3f}")
    return "\n".join(lines), table


SCENARIOS["ladder"] = ladder_scenario()      # the scripted ladder is a scenario of `--all` too


# ---- Auto Mix over a project of MIDI parts (2026-10-13) ---------------------------------------
# The staged stems are deleted and one MIDI part written, as a user would; the button's Auto Mix
# must decline WITH the offer, and the offer clicked must print the part through the app's own
# bounce - inside the Auto Mix errand, the first errand of that kind to send the app commands -
# register the print, mute the MIDI track and mix the print.
_MIDI_ONLY = [*({"key": "track.delete", "args": {"track": t}}
                for t in ("vocals", "drums", "bass", "other")),
              {"key": "track.add_midi", "args": {"name": "Keys"}},
              {"key": "midi.add_clip", "args": {"track": "Keys", "start_s": 0, "length_s": 4}},
              {"key": "midi.add_notes", "args": {"track": "Keys", "clip": 0, "notes": [
                  {"pitch": p, "start_beat": b, "length_beats": 1.5, "velocity": 100}
                  for b, p in ((0, 60), (2, 64), (4, 67), (6, 72))]}},
              {"key": "@pump", "ms": 1500},                    # the edit reaches the brain
              {"key": "agent.auto_mix", "args": {"brief": "master for streaming"}},
              {"key": "@pump", "ms": 800}]


def _automix_midi_print(o: Outcome) -> list[str]:
    bad: list[str] = []
    if len(o.turns) < 2:
        return ["the scenario did not reach its second turn"]
    feed = o.turns[0].get("feed") or []
    if not any("MIDI" in str(b.get("t", "")) and "Keys" in str(b.get("t", "")) for b in feed):
        bad.append("Auto Mix's decline did not say the part is MIDI")
    if not any(b.get("k") == "verbs" and b.get("print_midi") for b in feed):
        bad.append("the decline offered no print verb")
    pressed = (o.turns[1].get("before_replies") or [{}])[0]
    if not pressed.get("ok"):
        bad.append(f"the print verb could not be pressed: {pressed}")
    tracks = o.tracks(1)
    printed = tracks.get("Keys bounce")
    if printed is None:
        bad.append(f"no print track: {sorted(tracks)}")
    elif printed.get("track_kind") != "audio":
        bad.append(f"the print is not an audio track: {printed.get('track_kind')}")
    if not (tracks.get("Keys") or {}).get("muted"):
        bad.append("the MIDI track was not muted - it plays twice")
    try:
        stems = json.loads((o.project / "session.json").read_text()).get("stems") or {}
    except Exception:  # noqa: BLE001
        stems = {}
    if stems.get("Keys bounce", {}).get("excluded", True):
        bad.append(f"the print is not a stem the brain mixes: {sorted(stems)}")
    # what the feed says: the print and the mute, as rows of their own; and no bundle refused
    # over the tracks the user had deleted
    rows = {r.get("key"): r for r in (o.turns[1].get("feed") or []) if r.get("k") == "tool"}
    for key in ("track.bounce", "track.set_mute"):
        if key not in rows or rows[key].get("failed") or rows[key].get("pending"):
            bad.append(f"the feed has no settled {key} row: {rows.get(key)}")
    refusals = [str(b.get("t", ""))[:120] for b in (o.turns[1].get("feed") or [])
                if "refused" in str(b.get("t", ""))]
    if refusals:
        bad.append(f"Auto Mix proposed on deleted tracks: {refusals[:1]}")
    return bad


SCENARIOS["automix_midi_print"] = Scenario(
    "automix_midi_print",
    "Auto Mix over MIDI parts: declined with the offer; the offer clicked prints, mutes and mixes",
    [Turn("is it mixed yet?", [say("Not yet.")], leg={"before": _MIDI_ONLY}),
     Turn("thanks", [say("You're welcome.")],
          leg={"before": [{"key": "@press_print_offer"}, {"key": "@pump", "ms": 1500}]})],
    _automix_midi_print, ceiling_s=120.0)

def _rename_then_send(o: Outcome) -> list[str]:
    """Real use, 2026-10-22: the engineer renamed a split's stems and then sent the new
    "Vocals" to two returns in the same turn. The app renamed the track at once, the brain
    heard the rename only after the turn, so every move named "Vocals" was refused by the
    brain - and the next routing push took the sends out of the app as well. The reply said
    done twice."""
    bad: list[str] = []
    lead = o.tracks(0).get("Lead")
    if lead is None:
        return [f"no track named Lead: {sorted(o.tracks(0))}"]
    if "rv" not in (lead.get("sends") or {}):
        bad.append(f"the app's Lead sends nowhere after the turn: {lead.get('sends')}")
    try:
        stems = json.loads((o.project / "session.json").read_text()).get("stems") or {}
    except Exception:  # noqa: BLE001
        stems = {}
    held = stems.get("Lead")
    if held is None:
        bad.append(f"the brain holds no stem named Lead: {sorted(stems)}")
    else:
        if "rv" not in (held.get("sends") or {}):
            bad.append(f"the brain's Lead sends nowhere: {held.get('sends')}")
        if abs(float(held.get("pan") or 0.0) + 0.3) > 0.01:
            bad.append(f"the brain's Lead pan is {held.get('pan')}, not -0.3")
    brain_log = o.project.parent.parent / "logs" / "brain.log"
    said = brain_log.read_text(errors="replace") if brain_log.is_file() else ""
    skipped = [x for x in (o.log + "\n" + said).splitlines() if "coedit skip" in x]
    if skipped:
        bad.append(f"the brain refused a move of the turn: {skipped[0][:160]}")
    return bad


SCENARIOS["rename_then_send"] = Scenario(
    "rename_then_send",
    "a track renamed and then sent, panned and routed in the same turn: the brain follows the "
    "new name before the moves that use it, and the sends survive the next routing push",
    [Turn("call the vocals Lead, send it to a reverb, pan it a little left, reverb pre-fader",
          [ask(call("toolu_1", "return.add", kind="reverb")),
           ask(call("toolu_2", "track.rename", track="vocals", name="Lead")),
           ask(call("toolu_3", "send.set", source="Lead", **{"return": "rv"}, gain_db=-12),
               call("toolu_4", "track.set_pan", track="Lead", pan=-0.3)),
           ask(call("toolu_5", "return.set_prefader", id="rv", pre_fader=True)),
           say("Done.")],
          results={"toolu_1": "ok", "toolu_2": "ok", "toolu_3": "ok", "toolu_4": "ok",
                   "toolu_5": "ok"})],
    _rename_then_send, ceiling_s=120.0)

def _rename_and_undo(o: Outcome) -> list[str]:
    """The pre-merge review of 2026-10-23 (finding 1): Undo Agent Move after the agent renamed a
    track. The brain asks for `edit.undo` holding its mutate_lock; the app takes the rename back,
    and its reply carries the rename back (`renames`). The brain followed it by taking that same
    lock - from the thread that held it - and the undo never answered."""
    bad: list[str] = []
    if "vocals" not in o.tracks(0, "after_snapshot"):
        bad.append(f"after the undo the app has no track named vocals: {sorted(o.tracks(0, 'after_snapshot'))}")
    try:
        stems = json.loads((o.project / "session.json").read_text()).get("stems") or {}
    except Exception:  # noqa: BLE001
        stems = {}
    if "vocals" not in stems or "Lead" in stems:
        bad.append(f"the brain did not follow the undone rename: it holds {sorted(stems)}")
    return bad


SCENARIOS["rename_and_undo"] = Scenario(
    "rename_and_undo",
    "Undo Agent Move after a rename: the brain follows the rename back and the undo answers",
    [Turn("call the vocals Lead",
          [ask(call("toolu_1", "track.rename", track="vocals", name="Lead")), say("Renamed.")],
          results={"toolu_1": "ok"}, leg=UNDO)],
    _rename_and_undo, ceiling_s=120.0)

#: What the scripted ladder may cost: the brain's and the app's own work, the model taken out.
LADDER_BUDGET = REPO / "orchestrator" / "ladder_budget.json"
#: The numbers of `turn_timing.metrics` that COUNT work (held exactly: a count does not vary
#: from run to run) and the ones that are seconds (held loosely: three times and a second).
_COUNTED = ("rounds", "tool_calls", "renders_master", "renders_stem")
_TIMED = ("agent_time_s", "brain_share_s", "prep_max_s", "measure_s", "first_activity_max_s")


def ladder_budget(measured: dict) -> dict:
    """A budget from one scripted ladder session, in `turn_timing.check`'s format. Counts are
    the session's own: a turn that starts measuring twice, or rendering where it did not, is
    over at once. Seconds get room - they are tenths of a second on a quiet machine."""
    limits = {k: measured[k] for k in _COUNTED if k in measured}
    limits.update({k: round(3.0 * float(measured[k]) + 1.0, 2) for k in _TIMED if k in measured})
    return {"turns": measured.get("turns", 0), "slack": "counts exact; seconds x3 + 1", "max": limits}


# ---------------------------------------------------------------- a recorded session, replayed

def tape_turns(project: Path) -> list[Turn]:
    """A project's recorded turns (`MIX_AGENT_TAPE=1`, orchestrator/turn_tape.py) as rig
    turns: the same prompts, the model's own rounds as the tape, and every call held to how
    it came back when it was recorded - so a replay that meets a refusal the recording did
    not is `diverged`, never a pass."""
    from . import turn_tape

    from .operate_eval import CONFIRM_KEYS

    def blocks(step: dict) -> dict:
        return {**step, "blocks": [tuple(b) for b in step.get("blocks") or []]}

    def leg(t: dict) -> dict:
        """An approval box is answered as it was: a confirm-gated command that came back ok
        was allowed, one that came back an error was declined (the tape keeps no click)."""
        results = t.get("results") or {}
        gated = [results.get(b[1]) for s in t.get("steps") or [] for b in s.get("blocks") or []
                 if b[0] == "tool_use" and b[2] == "daw_command"
                 and isinstance(b[3], dict) and b[3].get("command") in CONFIRM_KEYS]
        return {"confirm": "no" if "error" in gated else "yes", "confirm_delay_ms": 50} if gated else {}
    return [Turn(t.get("prompt") or "", [blocks(s) for s in t.get("steps") or []],
                 results=dict(t.get("results") or {}), leg=leg(t)) for t in turn_tape.read(project)]


# ---------------------------------------------------------------- the command sweep


# ---------------------------------------------------------------- the command sweep

CENSUS = REPO / "server" / "tests" / "data" / "command_sweep_census.json"
#: Never sent, each with its reason. Everything else in the catalog is.
SWEEP_DENY = {
    "agent.run": "starts a mix run, which asks the model",
    "agent.auto_mix": "starts Auto Mix",
    "agent.advise": "starts an engineer errand",
    "agent.reference": "starts an engineer errand",
    "agent.undo": "the rig's own scenarios drive it; in a sweep it would undo the sweep",
    "agent.cancel": "stops the turn it is called in, so the tape's next step answers the next turn",
    "transport.record": "opens the audio inputs",
    "record.arm": "opens the audio inputs",
    "record.retrospective": "the audio inputs",
    "slot.record": "opens the audio inputs",
    "launcher.record": "opens the audio inputs",
    "track.set_input": "the hardware inputs",
    "track.set_output": "the hardware outputs",
    "audio_input.configure": "the hardware inputs",
}
#: Argument names that say WHICH thing: a call naming one that does not exist must be refused.
#: (Not `name`: it is nearly always the NEW name of something a command makes, and a sweep
#: that made tracks called "no_such_thing_zz" then found them by that name.)
_TARGET_ARGS = ("track", "stem", "source", "from", "to", "return", "bus", "rack", "id")
_INDEX_ARGS = ("clip", "scene", "slot", "marker", "index", "take")


def _arg_names(schema: str) -> list[str]:
    """Argument names out of a command's grammar text - the audit fuzz's parser, which the
    full audit already ran over every command."""
    import importlib.util
    path = REPO / "docs" / "audits" / "full-2026-10-02" / "tools" / "command_fuzz.py"
    spec = importlib.util.spec_from_file_location("command_fuzz", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.arg_names(schema)


def sweep_plan() -> dict[str, list[tuple[str, dict]]]:
    """Every agent-invocable command the sweep sends, by key: its variants in order.
    `bare` = no arguments; `wrong_type` = every named argument a list; `no_target` = every
    argument that says WHICH thing naming one that is not there; `word_index` = a word where
    an index belongs."""
    from .operate_eval import _arg_schemas, _registry_commands

    schemas = _arg_schemas()
    plan: dict[str, list[tuple[str, dict]]] = {}
    for key, user_only, _ in _registry_commands():
        if user_only or key in SWEEP_DENY:
            continue
        names = _arg_names(schemas.get(key, ""))
        variants: list[tuple[str, dict]] = [("bare", {})]
        if names:
            variants.append(("wrong_type", {n: [] for n in names}))
        targets = {n: "no_such_thing_zz" for n in names if n in _TARGET_ARGS}
        targets.update({n: 9999 for n in names if n in _INDEX_ARGS})
        if targets:
            variants.append(("no_target", targets))
        # A WORD where an index belongs ("the first clip"): a model does write this, and a
        # handler that reads it with (int) reads zero.
        words = {n: "zzz" for n in names if n in _INDEX_ARGS}
        if words:
            variants.append(("word_index", {**{n: "no_such_thing_zz" for n in names if n in _TARGET_ARGS},
                                            **words}))
        plan[key] = variants
    return plan


def _sweep_scenario(family: str, keys: list[str], plan: dict) -> Scenario:
    turns = []
    for key in keys:
        calls = [call(f"{key}/{variant}", key, **args) for variant, args in plan[key]]
        turns.append(Turn(f"sweep {key}", [ask(*calls), say("next")]))
    return Scenario(f"sweep_{family}", f"every {family}.* command, each variant, through the agent's door",
                    turns, lambda o: [], ceiling_s=60.0, light=True, unsettled_ok=True)


def _sweep_family(family: str, keys: list[str], plan: dict, work: Path) -> tuple[dict, list[str], dict]:
    """One family in one session: (census rows, problems, the app's own seconds per key)."""
    scenario = _sweep_scenario(family, keys, plan)
    _, o = run_scenario(scenario, work)
    rows: dict[str, dict] = {}
    problems: list[str] = []
    got = tool_results(o.requests)
    if len(o.turns) != len(keys):
        last = keys[len(o.turns)] if len(o.turns) < len(keys) else "?"
        problems.append(f"{family}: the session ended at {last} ({len(o.turns)} of {len(keys)} "
                        f"commands) - " + next((x for x in reversed(o.log.splitlines()) if x.strip()), "")[:160])
    for i, key in enumerate(keys[:len(o.turns)]):
        t = o.turns[i]
        # What the TURN wrote to the edit's tree. Not the engine's own device list, which it
        # writes into every edit a few seconds after launch, whoever is doing what.
        wrote = [w for w in (t.get("tree_writes") or []) if "INPUTDEVICE" not in w]
        moved = bool(wrote)
        outcomes = []
        for variant, _args in plan[key]:
            r = got.get(f"{key}/{variant}")
            if r is None:
                rows[f"{key} {variant}"] = {"outcome": "no result"}
                problems.append(f"{key} ({variant}): the call never came back")
                continue
            try:
                body = json.loads(r["text"])
            except (TypeError, ValueError):
                body = None
            if r["ok"]:
                outcome = "queued" if isinstance(body, dict) and body.get("queued") else "ok"
            else:
                outcome = "refused"
                sentence = (body.get("error") if isinstance(body, dict) else r["text"]) or ""
                if len(str(sentence).strip()) < 8:
                    problems.append(f"{key} ({variant}): refused with no sentence ({r['text'][:80]!r})")
            outcomes.append(outcome)
            rows[f"{key} {variant}"] = {"outcome": outcome}
        if outcomes and all(x == "refused" for x in outcomes) and moved:
            problems.append(f"{key}: every call was refused and the edit changed all the same "
                            f"({', '.join(wrote[:4])})")
        if outcomes and outcomes[0] == "ok":
            rows[f"{key} bare"]["moved"] = moved
        if not t.get("ended"):
            problems.append(f"{key}: the turn did not end")
    app_s = {}
    for line in o.timing:
        for key, row in (line.get("commands") or {}).items():
            if "app_s" in row and row.get("n"):
                app_s[key] = max(app_s.get(key, 0.0), float(row["app_s"]) - float(row.get("confirm_s") or 0.0))
    return rows, problems, app_s


def run_sweep(work: Path, jobs: int = 4, only: Optional[str] = None) -> tuple[dict, list[str], dict]:
    """Every command of `sweep_plan`, a family a session, `jobs` sessions at a time."""
    from concurrent.futures import ThreadPoolExecutor

    plan = sweep_plan()
    families: dict[str, list[str]] = {}
    for key in plan:
        families.setdefault(key.split(".")[0], []).append(key)
    if only:
        families = {f: k for f, k in families.items() if f.startswith(only)}
    census: dict[str, dict] = {}
    problems: list[str] = []
    app_s: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        # the valid examples are one more session, beside the families
        examples = None if only else pool.submit(run_examples, work / "examples")
        for rows, bad, seconds in pool.map(
                lambda item: _sweep_family(item[0], item[1], plan, work / f"sweep_{item[0]}"),
                sorted(families.items())):
            census.update(rows)
            problems += bad
            app_s.update(seconds)
        if examples is not None:
            rows, bad = examples.result()
            census.update(rows)
            problems += bad
    return dict(sorted(census.items())), problems, app_s


def census_changes(now: dict, pinned: dict) -> list[str]:
    """What differs between a sweep and the pinned census: a command whose answer changed
    class, one the census does not know, one the sweep no longer sends."""
    out = []
    for name in sorted(set(now) | set(pinned)):
        a, b = now.get(name), pinned.get(name)
        if a is None:
            out.append(f"{name}: in the census, no longer sent")
        elif b is None:
            out.append(f"{name}: new - {a}")
        elif a != b:
            out.append(f"{name}: was {b}, now {a}")
    return out


# ---------------------------------------------------------------- a valid call of each

@dataclass
class Example:
    """ONE VALID CALL of a command, in a session that makes it valid: the calls before it
    are its setup. The sweep's other variants show what a handler does with a call that is
    wrong; this is the call that is right, and it must answer ok and - where `check` says
    what to look for - leave the project as asked."""
    key: str
    args: dict = field(default_factory=dict)
    check: Optional[Callable[[Outcome, int], str]] = None
    leg: dict = field(default_factory=dict)
    tool: str = "daw_command"              # "propose" for the setup a device command needs


def _clips(name: str, n: int) -> Callable[[Outcome, int], str]:
    def check(o: Outcome, i: int) -> str:
        got = len(_t(o, i, name).get("clips") or [])
        return "" if got == n else f"{name} has {got} clip(s), not {n}"
    return check


def _clip(name: str, index: int, field_: str, want, tol: float = 0.011) -> Callable[[Outcome, int], str]:
    def check(o: Outcome, i: int) -> str:
        clips = _t(o, i, name).get("clips") or []
        if index >= len(clips):
            return f"{name} has no clip {index}"
        got = clips[index].get(field_)
        fine = (got is not None and abs(float(got) - float(want)) <= tol) \
            if isinstance(want, float) else got == want
        return "" if fine else f"{name} clip {index} {field_} is {got!r}, not {want!r}"
    return check


def _snap(path: str, want, tol: float = 0.011) -> Callable[[Outcome, int], str]:
    """A field of the snapshot by dotted path ("arrange_view.snap", "transport.loop_on")."""
    def check(o: Outcome, i: int) -> str:
        got: Any = o.turns[i].get("snapshot") or {}
        for part in path.split("."):
            got = got.get(part) if isinstance(got, dict) else None
        fine = (got is not None and abs(float(got) - float(want)) <= tol) \
            if isinstance(want, float) else got == want
        return "" if fine else f"{path} is {got!r}, not {want!r}"
    return check


def _markers(*names: str) -> Callable[[Outcome, int], str]:
    def check(o: Outcome, i: int) -> str:
        got = [m.get("name") for m in (o.turns[i].get("snapshot") or {}).get("markers") or []]
        return "" if got == list(names) else f"the markers are {got}, not {list(names)}"
    return check


def _tracks(*names: str, absent: tuple = ()) -> Callable[[Outcome, int], str]:
    def check(o: Outcome, i: int) -> str:
        have = set(o.tracks(i))
        missing = [n for n in names if n not in have]
        extra = [n for n in absent if n in have]
        return (f"no track {missing}" if missing else f"still a track: {extra}" if extra else "")
    return check


def _returns(*ids: str) -> Callable[[Outcome, int], str]:
    def check(o: Outcome, i: int) -> str:
        got = sorted(str(r.get("id") or r.get("name")).removeprefix("return:") for r in
                     (o.turns[i].get("snapshot") or {}).get("returns") or [])
        return "" if got == sorted(ids) else f"the returns are {got}, not {sorted(ids)}"
    return check


def _replied(field_: str, want, tol: float = 0.011) -> Callable[[Outcome, int], str]:
    """A field of the command's own REPLY. The playhead is read here, never in the snapshot:
    the rig closes the audio device, and with no audio callback the engine copies the
    playhead's stale position back over the transport's 200 ms after any move."""
    def check(o: Outcome, i: int) -> str:
        r = tool_results(o.requests).get(f"ex{i}") or {}
        try:
            got = json.loads(r.get("text") or "{}").get(field_)
        except (TypeError, ValueError):
            got = None
        fine = (got is not None and abs(float(got) - float(want)) <= tol) \
            if isinstance(want, float) else got == want
        return "" if fine else f"the reply's {field_} is {got!r}, not {want!r}"
    return check


def _folded(group: str) -> Callable[[Outcome, int], str]:
    def check(o: Outcome, i: int) -> str:
        got = (o.turns[i].get("snapshot") or {}).get("arrange_view", {}).get("folded_groups") \
            or (o.turns[i].get("snapshot") or {}).get("folded_groups") or []
        return "" if group in got or f"bus:{group}" in got else f"{group} is not folded (folded: {got})"
    return check


def _plays_into(name: str, where: str) -> Callable[[Outcome, int], str]:
    def check(o: Outcome, i: int) -> str:
        got = _t(o, i, name).get("output") or "master"
        return "" if got == where else f"{name} plays into {got!r}, not {where!r}"
    return check


_COMP = {"rationale": "hold the drums", "predicted_impact": [],
         "actions": [{"type": "comp", "stem_id": "drums", "threshold_db": -18.0, "ratio": 3.0,
                      "attack_ms": 10.0, "release_ms": 120.0}]}
_YES = {"confirm": "yes", "confirm_delay_ms": 200}

#: The session. test_projects/suno_demo: vocals, drums, bass, other - one 8 s clip each.
#: ORDER MATTERS: each call is valid because of the ones before it.
EXAMPLES: list[Example] = [
    # ---- the transport and the view
    Example("transport.seek", {"position_s": 4.0}),
    Example("transport.return_to_start"),
    Example("arrange.playhead_bar_right"),
    Example("arrange.playhead_right"),
    Example("arrange.playhead_left"),
    Example("arrange.playhead_bar_left"),
    Example("arrange.playhead_end"),
    Example("arrange.playhead_home", {}, _replied("position_s", 0.0)),
    Example("arrange.zoom_in"),
    Example("arrange.zoom_out"),
    Example("arrange.zoom_back"),
    Example("arrange.fit_width"),
    Example("arrange.fit_height"),
    Example("arrange.loop_selection", {"start_s": 2.0, "end_s": 6.0, "on": True}),
    Example("arrange.select_loop"),
    Example("arrange.zoom_selection"),
    Example("arrange.loop_selection", {"on": False}),
    Example("arrange.toggle_follow", {"on": False}, _snap("arrange_view.follow", False)),
    Example("grid.toggle_snap", {"on": False}, _snap("arrange_view.snap", False)),
    Example("grid.toggle_mode", {"mode": "fixed"}, _snap("arrange_view.grid_mode", "fixed")),
    Example("grid.narrower"),
    Example("grid.wider"),
    Example("grid.toggle_triplet", {"on": True}, _snap("arrange_view.triplet", True)),
    Example("automation.edit_toggle", {"on": True}, _snap("arrange_view.automation_edit", True)),
    Example("arrange.draw_mode", {"on": True}),
    # ---- markers, tempo, key
    Example("marker.add", {"position_s": 2.0, "name": "Verse"}, _markers("Verse")),
    Example("marker.add", {"position_s": 6.0, "name": "Drop"}, _markers("Verse", "Drop")),
    Example("marker.set", {"index": 1, "name": "Hook"}, _markers("Verse", "Hook")),
    Example("transport.go_to_marker", {"index": 1}),
    Example("marker.remove", {"index": 0}, _markers("Hook")),
    Example("tempo.set", {"bpm": 100}, _tempo(100.0)),
    Example("tempo.add", {"at_s": 4.0, "bpm": 110}),
    Example("tempo.move", {"index": 1, "bpm": 105}),
    Example("tempo.remove", {"index": 1}),
    Example("tempo.set_time_sig", {"numerator": 3, "denominator": 4}),
    Example("tempo.set_time_sig", {"numerator": 4, "denominator": 4}),
    Example("key.set", {"tonic": "A", "scale": "minor"}),
    # ---- a channel's strip
    Example("track.set_gain", {"track": "vocals", "gain_db": -3.0}, _is("vocals", "gain_db", -3.0, 0.05)),
    Example("track.set_pan", {"track": "other", "pan": -0.3}, _is("other", "pan", -0.3, 0.02)),
    Example("track.set_width", {"track": "other", "width": 1.2}),
    Example("track.set_mute", {"track": "bass", "muted": True}, _is("bass", "muted", True)),
    Example("track.mute_toggle", {"track": "bass", "on": False}, _is("bass", "muted", False)),
    Example("track.solo", {"track": "drums", "on": True}, _is("drums", "solo", True)),
    Example("track.solo", {"track": "drums", "on": False}, _is("drums", "solo", False)),
    Example("track.set_colour", {"track": "bass", "colour": "#3366cc"}),
    Example("track.select", {"track": "vocals"}),
    Example("track.select_all"),
    Example("track.set_delay", {"track": "drums", "delay_ms": 5.0}),
    Example("track.keep_latency", {"track": "drums", "on": False}),
    Example("track.fold_height", {"track": "vocals"}),
    # ---- tracks come and go
    Example("track.add_blank", {"name": "scratch"}, _tracks("scratch")),
    Example("track.rename", {"track": "scratch", "name": "sketch"}, _tracks("sketch", absent=("scratch",))),
    Example("track.add_midi", {"name": "pad"}, _tracks("pad")),
    Example("track.delete", {"track": "sketch"}, _tracks("pad", absent=("sketch",))),
    Example("track.delete", {"track": "pad"}, _tracks("vocals", absent=("pad",))),
    Example("track.duplicate", {"track": "bass"}, _tracks("bass", "bass 2")),
    Example("track.delete", {"track": "bass 2"}, _tracks("bass", absent=("bass 2",))),
    # ---- clips
    Example("clip.split", {"track": "vocals", "clip": 0, "at_s": 4.0}, _clips("vocals", 2)),
    Example("clip.move", {"track": "vocals", "clip": 1, "start_s": 4.5}, _clip("vocals", 1, "start_s", 4.5)),
    Example("clip.trim", {"track": "vocals", "clip": 1, "end_s": 7.5}),
    Example("clip.fade", {"track": "vocals", "clip": 0, "fade_in_s": 0.05, "fade_out_s": 0.1}),
    Example("clip.gain", {"track": "vocals", "clip": 0, "gain_db": -2.0}),
    Example("clip.duplicate", {"track": "vocals", "clip": 1}, _clips("vocals", 3)),
    Example("clip.delete", {"track": "vocals", "clip": 2}, _clips("vocals", 2)),
    Example("clip.copy", {"track": "other", "clip": 0}),
    Example("clip.paste", {"track": "other", "at_s": 10.0}, _clips("other", 2)),
    Example("clip.cut", {"track": "other", "clip": 1}, _clips("other", 1)),
    Example("clip.loop", {"track": "other", "clip": 0, "on": True}, _replied("looping", True)),
    Example("clip.normalize", {"track": "drums", "clip": 0}),
    Example("clip.pitch", {"track": "other", "clip": 0, "semitones": 2}),
    Example("clip.reverse", {"track": "other", "clip": 0, "reversed": True}),
    Example("clip.add_effect", {"track": "bass", "clip": 0, "type": "filter", "freq_hz": 200.0,
                                "highpass": True}),
    Example("clip.remove_effect", {"track": "bass", "clip": 0, "index": 0}),
    Example("clip.warp", {"track": "drums", "clip": 0, "on": True}),
    Example("clip.warp_marker", {"track": "drums", "clip": 0, "source_s": 2.0, "warp_s": 2.1}),
    Example("clip.warp_remove", {"track": "drums", "clip": 0}),
    Example("clip.warp_all", {"on": False}),
    Example("clip.crop", {"track": "bass", "start_s": 1.0, "end_s": 7.0}),
    # ---- the whole timeline
    Example("time.insert", {"start_s": 0.0, "length_s": 2.0}),
    Example("time.delete", {"start_s": 0.0, "end_s": 2.0}),
    Example("time.copy", {"start_s": 0.0, "end_s": 2.0}),
    Example("time.paste", {"at_s": 12.0}),
    Example("time.duplicate", {"start_s": 0.0, "end_s": 2.0}),
    Example("time.cut", {"start_s": 0.0, "end_s": 2.0}),
    # ---- automation
    Example("automation.write", {"track": "vocals", "param": "volume",
                                 "points": [{"t_s": 0.0, "value": -6.0}, {"t_s": 4.0, "value": 0.0}]}),
    Example("automation.set_curve", {"track": "vocals", "index": 0, "curve": 0.5, "param": "volume"}),
    Example("automation.reenable", {"track": "vocals"}),
    Example("automation.clear", {"track": "vocals", "param": "volume"}),
    # ---- routing: groups, returns, sends, a duck
    Example("group.create", {"stems": ["drums", "bass"]}, _plays_into("drums", "bus:bus1")),
    Example("bus.rename", {"bus": "bus1", "new_name": "rhythm"}),
    Example("bus.route", {"bus": "rhythm", "target": "master"}),
    Example("route.stem", {"stem": "other", "bus": "new"}),
    Example("route.stem", {"stem": "other", "bus": "master"}, _plays_into("other", "master")),
    Example("return.add", {"kind": "reverb"}, _returns("rv")),
    Example("return.add", {"kind": "delay"}),
    Example("send.set", {"source": "vocals", "return": "rv", "gain_db": -12.0}),
    Example("return.set_level", {"id": "rv", "gain_db": -3.0}),
    Example("return.set_prefader", {"id": "rv", "pre_fader": True}),
    Example("return.rename", {"id": "rv", "new_name": "plate"}),
    Example("duck.set_key", {"stem": "bass", "key": "drums", "threshold_db": -30.0, "depth_db": 6.0}),
    Example("track.fold_toggle", {"track": "bus:rhythm", "folded": True}, _folded("rhythm")),
    Example("track.unfold_all"),
    Example("return.remove", {"id": "plate"}, leg=_YES),
    Example("bus.remove", {"id": "rhythm"}, _plays_into("drums", "master"), leg=_YES),
    # ---- devices
    Example("propose", _COMP, _has_device("drums", "comp"), tool="propose"),
    Example("device.set", {"track": "drums", "target": "comp.threshold_db", "value": -20.0}),
    Example("device.enable", {"track": "drums", "device": "comp", "on": False}),
    Example("device.move_to_track", {"type": "comp", "from": "drums", "to": "vocals"},
            _has_device("vocals", "comp")),
    Example("device.add", {"track": "vocals", "type": "utility", "params": {"invert": True}},
            _has_device("vocals", "utility")),
    Example("device.remove", {"track": "vocals", "device": "utility"}),
    Example("master.limiter", {"on": True}),
]


#: The mix-session families: a command of one of these has an example above, or a line
#: below saying why not - a new one fails `test_phase56_chat_rig` until it has either.
EXAMPLE_FAMILIES = ("track.", "clip.", "device.", "send.", "return.", "transport.", "marker.",
                    "tempo.", "automation.", "bus.", "group.", "route.", "master.", "duck.",
                    "key.", "time.", "grid.", "arrange.")
_PLAYS = "plays audio, and the rig's audio device is closed"
_RENDERS = "a render: the ladder holds a bounce and an export, the live suite a freeze"
_ERRAND = "needs the brain's own analysis of the clip, an errand queued behind the turn"
_NOT_MIX = "MIDI, instruments and the Loop Deck are not a mix session's commands"
NO_EXAMPLE_YET: dict[str, str] = {
    "transport.play_pause": _PLAYS, "transport.play_continue": _PLAYS, "transport.stop": _PLAYS,
    "transport.play_clips": _PLAYS, "transport.stop_clips": _PLAYS, "track.stop_clips": _PLAYS,
    "transport.back_to_arrangement": _PLAYS, "automation.record": _PLAYS,
    "track.spectrogram": "reads the live meters, which need playback",
    "track.set_midi_input": "needs a MIDI input device", "track.set_monitor": "needs an audio input",
    "track.freeze": _RENDERS, "track.bounce": _RENDERS, "clip.consolidate": _RENDERS,
    "track.separate": _RENDERS,
    "track.add": "needs an audio file outside the project",
    "clip.match_key": _ERRAND, "clip.pitch_correct": _ERRAND, "clip.extract_groove": _ERRAND,
    "clip.set_groove": "needs a groove in the pool", "clip.auto_pitch": _ERRAND,
    "clip.warp_move": "needs the index of a warp marker the session made",
    "clip.stretch_mode": "its values are the engine's own names, read from the snapshot",
    "track.set_instrument": _NOT_MIX, "track.add_drum_kit": _NOT_MIX, "track.set_arp": _NOT_MIX,
    "clip.to_slot": _NOT_MIX,
}


def example_keys() -> list[str]:
    """Every command that has a valid example, once, in the session's order."""
    return list(dict.fromkeys(e.key for e in EXAMPLES if e.tool == "daw_command"))


def examples_scenario() -> Scenario:
    turns = []
    for n, e in enumerate(EXAMPLES):
        made = tool(f"ex{n}", e.tool, **e.args) if e.tool != "daw_command" \
            else call(f"ex{n}", e.key, **e.args)
        turns.append(Turn(f"example {n}: {e.key}", [ask(made), say("next")], leg=e.leg))
    return Scenario("examples", "one valid call of each mix-session command, in a session that "
                    "makes it valid", turns, lambda o: [], ceiling_s=420.0, unsettled_ok=True)


def run_examples(work: Path) -> tuple[dict, list[str]]:
    """The examples session: (census rows `<key> example`, problems). A row is the WORST any
    example of that key did; a problem is a call that was not ok, or one whose check found
    the project not as asked."""
    _, o = run_scenario(examples_scenario(), work)
    got = tool_results(o.requests)
    rows: dict[str, dict] = {}
    problems: list[str] = []
    if len(o.turns) != len(EXAMPLES):
        problems.append(f"examples: the session ended at example {len(o.turns)} of {len(EXAMPLES)} - "
                        + next((x for x in reversed(o.log.splitlines()) if x.strip()), "")[:160])
    for n, e in enumerate(EXAMPLES[:len(o.turns)]):
        r = got.get(f"ex{n}")
        said = ""
        if r is None:
            outcome, said = "no result", "the call never came back"
        elif not r["ok"]:
            outcome, said = "refused", str(r["text"])[:200]
        else:
            outcome = "ok"
            try:
                body = json.loads(r["text"])
            except (TypeError, ValueError):
                body = None
            if isinstance(body, dict) and body.get("queued"):
                outcome = "queued"
            if e.check is not None and (why := e.check(o, n)):
                outcome, said = "not as asked", why
        if said:
            problems.append(f"example {n} {e.key} {json.dumps(e.args)[:90]}: {outcome} - {said}")
        if e.tool == "daw_command":
            row = rows.setdefault(f"{e.key} example", {"outcome": outcome})
            if outcome != "ok" and row["outcome"] == "ok":
                row["outcome"] = outcome
    return rows, problems


def verdict(name: str, bad: list[str]) -> tuple[bool, str]:
    """A scenario passes with no problems; a CONTROL passes only with the problem it stages."""
    if name in CONTROLS:
        wanted = [b for b in bad if CONTROLS[name] in b]
        other = [b for b in bad if CONTROLS[name] not in b]
        ok = bool(wanted) and not other
        return ok, (f"control held ({wanted[0][:110]})" if ok
                    else "control BROKEN: " + ("; ".join(other) if other else "it passed"))
    return not bad, "; ".join(bad)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--run", nargs="+", metavar="NAME")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--keep", action="store_true", help="leave the sandbox on disk")
    parser.add_argument("--ladder", action="store_true",
                        help="the 14 rungs: on the scripted model (free), or with --live")
    parser.add_argument("--live", choices=sorted(LIVE_CREDENTIAL),
                        help="with --ladder: a LIVE model on this transport. BILLED")
    parser.add_argument("--reps", type=int, default=1, help="with --ladder --live: sessions to run")
    parser.add_argument("--billed", action="store_true",
                        help="say that a live run is meant: without it --live is refused")
    parser.add_argument("--check-budget", action="store_true",
                        help="with --ladder (scripted): fail over orchestrator/ladder_budget.json")
    parser.add_argument("--write-budget", action="store_true",
                        help="with --ladder (scripted): write this session as the budget")
    parser.add_argument("--replay", type=Path, metavar="PROJECT",
                        help="play a project's recorded tapes (MIX_AGENT_TAPE=1) through the rig")
    parser.add_argument("--sweep", action="store_true",
                        help="every command through the agent's door, held to the pinned census")
    parser.add_argument("--family", help="with --sweep: only families starting with this")
    parser.add_argument("--examples", action="store_true",
                        help="only the sweep's valid-example session (one right call of each "
                             "mix-session command)")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--update-census", action="store_true",
                        help="with --sweep: write what this sweep found as the census")
    parser.add_argument("--work", type=Path, help="where to run (default: a temp folder)")
    args = parser.parse_args(argv)
    if args.ladder:
        if not BINARY.is_file():
            print(f"the console binary is not built: {BINARY}")
            return EXIT_NOT_BUILT
        if args.live and not args.billed:
            print(f"--live {args.live} reaches a real model and is BILLED ({len(LADDER)} turns a "
                  f"session, {args.reps} session(s)): say --billed to mean it")
            return 1
        root = args.work or Path(tempfile.mkdtemp(prefix="chat-rig-ladder-"))
        runs = []
        for rep_no in range(max(1, args.reps if args.live else 1)):
            bad, outcome = run_scenario(ladder_scenario(bool(args.live)), root / f"rep{rep_no + 1}",
                                        live=args.live)
            runs.append((bad, outcome))
            for line in bad:
                print(f"XX  rep {rep_no + 1}: {line}")
        said, table = ladder_report(runs)
        print(said)
        over = False
        if not args.live and (args.check_budget or args.write_budget):
            from . import turn_timing
            project = runs[0][1].project
            if args.write_budget and not runs[0][0]:
                budget = ladder_budget(turn_timing.metrics(turn_timing.selected(project)))
                LADDER_BUDGET.write_text(json.dumps(budget, indent=1, sort_keys=True) + "\n")
                print(f"budget written: {LADDER_BUDGET.relative_to(REPO)}")
            elif args.check_budget:
                ok, lines = turn_timing.check(project, LADDER_BUDGET)
                print(lines)
                over = not ok
        root.mkdir(parents=True, exist_ok=True)
        (root / "ladder.json").write_text(json.dumps(
            {"live": args.live, "reps": len(runs), "kinds": table,
             "problems": [b for bad, _ in runs for b in bad]}, indent=1) + "\n")
        failed = sum(1 for bad, _ in runs if bad)
        print(f"{len(runs) - failed} of {len(runs)} session(s) with every rung done; "
              f"the table and the problems: {root / 'ladder.json'}")
        if not (args.keep or args.live or failed or over):
            shutil.rmtree(root, ignore_errors=True)
        return 1 if failed or over else 0
    if args.replay:
        if not BINARY.is_file():
            print(f"the console binary is not built: {BINARY}")
            return EXIT_NOT_BUILT
        turns = tape_turns(args.replay.expanduser())
        if not turns:
            print(f"no tapes under {args.replay}/runs/tape (record with MIX_AGENT_TAPE=1)")
            return 1
        from . import turn_timing
        root = args.work or Path(tempfile.mkdtemp(prefix="chat-rig-replay-"))
        scenario = Scenario("replay", "a recorded session with the model taken out", turns,
                            lambda o: [], ceiling_s=600.0)
        bad, outcome = run_scenario(scenario, root / "replay")
        for line in bad:
            print("XX  " + line)
        print(turn_timing.table(outcome.project))
        print(f"{len(turns)} recorded turn(s) replayed; sandbox: {root}")
        return 1 if bad else 0
    if args.examples:
        if not BINARY.is_file():
            print(f"the console binary is not built: {BINARY}")
            return EXIT_NOT_BUILT
        root = args.work or Path(tempfile.mkdtemp(prefix="chat-rig-examples-"))
        rows, problems = run_examples(root / "examples")
        for line in problems:
            print("XX  " + line)
        fine = sum(1 for r in rows.values() if r["outcome"] in ("ok", "queued"))
        print(f"{len(EXAMPLES)} calls, {len(rows)} commands with a valid example: {fine} answered ok "
              f"(or queued behind the turn) and left the project as asked")
        if args.keep or problems:
            print(f"sandbox: {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)
        return 1 if problems else 0
    if args.sweep:
        if not BINARY.is_file():
            print(f"the console binary is not built: {BINARY}")
            return EXIT_NOT_BUILT
        root = args.work or Path(tempfile.mkdtemp(prefix="chat-rig-sweep-"))
        began = time.monotonic()
        census, problems, app_s = run_sweep(root, args.jobs, args.family)
        for line in problems:
            print("XX  " + line)
        slow = sorted(app_s.items(), key=lambda kv: -kv[1])[:8]
        print("slowest in the app: " + ", ".join(f"{k} {v * 1000:.0f} ms" for k, v in slow))
        root.mkdir(parents=True, exist_ok=True)
        (root / "census.json").write_text(json.dumps(census, indent=1, sort_keys=True) + "\n")
        tally: dict[str, int] = {}
        for row in census.values():
            tally[row["outcome"]] = tally.get(row["outcome"], 0) + 1
        still = sorted(k for k, r in census.items() if r.get("moved") is False)
        print(f"{len(census)} calls of {len({k.split(' ')[0] for k in census})} commands in "
              f"{time.monotonic() - began:.0f} s: {tally}; {len(still)} answered ok and moved nothing")
        changes: list[str] = []
        if args.update_census and not args.family:
            if problems:
                print("the census is NOT written: the sweep has problems")
            else:
                CENSUS.parent.mkdir(parents=True, exist_ok=True)
                CENSUS.write_text(json.dumps(census, indent=1, sort_keys=True) + "\n")
                print(f"census written: {CENSUS.relative_to(REPO)}")
        elif CENSUS.is_file():
            pinned = json.loads(CENSUS.read_text())
            if args.family:
                pinned = {k: v for k, v in pinned.items() if k.startswith(args.family)}
            changes = census_changes(census, pinned)
            for line in changes[:40]:
                print("CHANGED  " + line)
        else:
            print("no census is pinned yet (--update-census writes one)")
        if args.keep or problems or changes:
            print(f"sandbox: {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)
        return 1 if problems or changes else 0
    if args.list or not (args.run or args.all):
        for s in SCENARIOS.values():
            print(f"{s.name:28} {'[control] ' if s.name in CONTROLS else ''}{s.about}")
        return 0
    if not BINARY.is_file():
        print(f"the console binary is not built: {BINARY}")
        return EXIT_NOT_BUILT
    names = list(SCENARIOS) if args.all else args.run
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        print(f"no such scenario: {', '.join(unknown)}")
        return 1
    root = args.work or Path(tempfile.mkdtemp(prefix="chat-rig-"))
    failed = 0
    for name in names:
        began = time.monotonic()
        bad, _ = run_scenario(SCENARIOS[name], root / name)
        ok, said = verdict(name, bad)
        failed += 0 if ok else 1
        print(f"{'ok  ' if ok else 'XX  '} {name:28} {time.monotonic() - began:5.1f} s  {said}")
    if args.keep or failed:
        print(f"sandbox: {root}")
    else:
        shutil.rmtree(root, ignore_errors=True)
    print(f"{len(names) - failed} of {len(names)} scenario(s) passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
