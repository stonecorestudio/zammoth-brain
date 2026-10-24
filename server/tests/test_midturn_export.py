"""A render the agent asks for mid-gesture carries that gesture's own mix moves.

FIXED 2026-09-22 (docs/10-daw.md, "A mid-turn export bounced the pre-turn mix"). The Phase 52 lofi
showcase's last turn proposed a master shelf and a -14 LUFS limiter target, measured -14.1 LUFS on
the brain's render, exported in the same turn — and shipped the pre-mastering mix (-13.14 LUFS-I,
the brain's BEFORE reading). A gesture's proposals reach the native graph only when its reply
carries `chains` (Brain.chat attaches them after chat_turn returns; App.cpp applies them at turn
end), and App.cpp's exportMaster renders the live graph the moment the command arrives.

Now Brain.counting_invoke hands the DAW the gesture's mix before any RENDER_COMMANDS key
(Brain._realize_mix_before): `coedits.take` drains the user's queued moves and mirrors them, one
render resolves the chains, and a `routing.apply` flagged `sync` realizes them before it replies.

These drive the REAL Brain doors (chat, apply_plan) with a scripted engineer and a DAW end that
records, in order, what it received and which master chain it held when each command reached it.
"""
from __future__ import annotations

import json

import pytest

from mix_engine.session import Session
from orchestrator import chat as chatmod
from orchestrator import daw_bridge as B
from orchestrator import daw_server as S
from orchestrator import operate_eval as O


def _master(chains) -> list[tuple]:
    """The master's entries as (type, the value that identifies this move)."""
    chains = chains or {}
    chains = chains.get("chains", chains)
    master = chains.get("master") or {}
    return [(e.get("type"), e.get("freq_hz") if e.get("type") == "eq" else e.get("target_lufs_i"))
            for e in master.get("entries", []) if isinstance(e, dict)]


def _stem_gain(chains, stem: str):
    chains = chains or {}
    chains = chains.get("chains", chains)
    for e in (chains.get(stem) or {}).get("entries", []):
        if isinstance(e, dict) and e.get("type") == "gain":
            return e.get("gain_db")
    return None


class _Scripted:
    """Stands in for ChatSession: fixed tool calls through the real execute_tool. `outputs`
    keeps each tool's result text, so a test can read what the engineer was told."""

    def __init__(self, script):
        self.script, self.history, self.usage, self._model = list(script), [], {}, ""
        self.outputs: list[tuple[str, str]] = []

    def set_settings(self, model, reasoning):
        pass

    async def turn(self, text, *, tools, context, emit, should_cancel=None,
                   live_context=None, during_run=False):
        self.history.append({"role": "user", "content": text})
        for name, args in self.script:
            out = await chatmod.execute_tool(tools, name, args)
            self.outputs.append((name, out.text or ""))
        await emit({"type": "chat_message", "text": "done"})
        self.history.append({"role": "assistant", "content": "done"})


class _DawEnd:
    """App.cpp's choke point, reduced to what matters here: routing.apply realizes chains (the
    only mid-turn door that does), coedits.take hands back whatever `takes` scripts for that
    call, and every other command runs on the live graph as it stands — recorded with the
    master (and one stem's gain) it held at that moment."""

    def __init__(self, brain, *, takes=(), refuse=(), watch_stem="stem_a"):
        self.live = brain.get_chains()
        self.takes = list(takes)
        self.refuse = dict(refuse)
        self.watch = watch_stem
        self.log: list[tuple] = []

    def __call__(self, command, args=None, *_a, **_k):
        if command in self.refuse:
            self.log.append((command, None))
            return {"ok": False, "error": self.refuse[command]}
        if command == "coedits.take":
            urls = self.takes.pop(0) if self.takes else []
            self.log.append((command, list(urls)))
            return {"ok": True, **({"coedits": list(urls)} if urls else {})}
        if command == "routing.apply" and isinstance(args, dict) and "chains" in args:
            self.live = args["chains"]
            self.log.append((command, {"sync": args.get("sync"), "master": _master(args["chains"]),
                                       "gain": _stem_gain(args["chains"], self.watch)}))
            return {"ok": True}
        self.log.append((command, {"master": _master(self.live),
                                   "gain": _stem_gain(self.live, self.watch)}))
        return {"ok": True, "deferred": True} if command == "session.export" else {"ok": True}

    def seen(self, command) -> list:
        return [held for c, held in self.log if c == command]


SHELF = ("eq", 8000.0)
TARGET = ("limiter", -14.0)
MASTER_IT = ("propose", {
    "rationale": "master it: roll the top off, land near -14 LUFS",
    "predicted_impact": [{"metric": "lufs_i", "predicted_delta": -1.0, "unit": "LU"}],
    "actions": [
        {"type": "eq", "stem_id": "master",
         "band": {"freq_hz": 8000, "gain_db": -2.5, "q": 0.7, "shape": "high_shelf"}},
        {"type": "limiter", "stem_id": "master", "ceiling_dbtp": -1.0, "target_lufs_i": -14.0}]})


def _export(tmp_path, name="master.wav"):
    return ("daw_command", {"command": "session.export",
                            "args": {"format": "wav", "file": str(tmp_path / name)}})


def _brain(project, monkeypatch, script, **daw):
    Session(project)                       # opened, as the brain's first request would
    brain = S.Brain(project)
    end = _DawEnd(brain, **daw)
    brain.daw_invoke = end
    session = _Scripted(script)
    monkeypatch.setitem(B._CHAT_SESSIONS, B._session_key(project, ""), session)
    return brain, end, session


def test_a_mid_turn_export_carries_the_turns_own_master_moves(demo_project, tmp_path,
                                                             monkeypatch):
    """The showcase's own shape: propose -> render -> session.export, in one turn."""
    brain, end, _s = _brain(demo_project, monkeypatch, [MASTER_IT, ("render", {}), _export(tmp_path)])
    assert SHELF not in _master(end.live) and TARGET not in _master(end.live)   # a clean start
    out = brain.chat("master it for streaming, then export a wav", mode="agent", use_llm=True)

    # Not vacuous: the moves DID land (they ride chat_done), and the export WAS invoked.
    assert SHELF in _master(out.get("chains")) and TARGET in _master(out.get("chains")), out
    exported = end.seen("session.export")
    assert exported, end.log
    assert SHELF in exported[0]["master"] and TARGET in exported[0]["master"], (
        f"session.export ran on master {exported[0]['master']}: the turn's own shelf + -14 LUFS "
        f"limiter were still waiting for chat_done")
    # …and it got there through ONE flagged realize, ahead of the export.
    order = [c for c, _h in end.log]
    assert order.index("routing.apply") < order.index("session.export"), order
    assert [h["sync"] for h in end.seen("routing.apply")] == ["session.export"], end.log


def test_a_turn_that_moved_nothing_does_not_push(demo_project, tmp_path, monkeypatch):
    """The fingerprint's job: an export with no mix change this gesture renders what the DAW
    already holds — no render, no push, the same traffic as before the fix."""
    brain, end, _s = _brain(demo_project, monkeypatch, [_export(tmp_path)])
    brain.chat("export a wav", mode="agent", use_llm=True)
    assert end.seen("session.export"), end.log
    assert not end.seen("routing.apply"), end.log


def test_the_users_queued_move_rides_the_push_instead_of_being_reset(demo_project, tmp_path,
                                                                     monkeypatch):
    """A fader the user moved during the turn sits in the DAW's queue. It is taken and mirrored
    BEFORE the chains are resolved, so the push carries it — resolved without it, the push would
    have reset that fader in the live graph and the export would have missed it (05·F15)."""
    move = "/api/set_stem?stem=stem_a&gain_db=-4.00"
    brain, end, _s = _brain(demo_project, monkeypatch, [MASTER_IT, _export(tmp_path)],
                            takes=[[move]])
    brain.chat("master it, then export", mode="agent", use_llm=True)
    pushes = end.seen("routing.apply")
    assert len(pushes) == 1 and pushes[0]["gain"] == pytest.approx(-4.0), end.log
    assert end.seen("session.export")[0]["gain"] == pytest.approx(-4.0), end.log
    # the mirror really happened: the brain's session holds the user's move
    assert json.loads((demo_project / "session.json").read_text())["stems"]["stem_a"]["gain_db"] \
        == pytest.approx(-4.0)


def test_a_move_made_during_the_render_gets_one_more_push(demo_project, tmp_path, monkeypatch):
    """The first push was resolved before this move reached the brain, so it overwrote it: the
    take after the push finds it, mirrors it, and a second push restores it before the export."""
    late = "/api/set_stem?stem=stem_a&gain_db=-2.50"
    # Three takes since 2026-10-03 (reader S4·R3): the queue is also taken before the PROPOSAL
    # (so it lands on top of the user's moves), then before the first push, then after it — and
    # that third take is the one that finds a move made while the render ran.
    brain, end, _s = _brain(demo_project, monkeypatch, [MASTER_IT, _export(tmp_path)],
                            takes=[[], [], [late]])
    brain.chat("master it, then export", mode="agent", use_llm=True)
    pushes = end.seen("routing.apply")
    assert len(pushes) == 2, end.log
    assert pushes[0]["gain"] != pytest.approx(-2.5) and pushes[1]["gain"] == pytest.approx(-2.5), pushes
    assert end.seen("session.export")[0]["gain"] == pytest.approx(-2.5), end.log


@pytest.mark.parametrize("door", ["chat", "apply_plan"])
def test_a_routing_push_mid_turn_is_not_mistaken_for_the_start_mix(demo_project, tmp_path,
                                                                   monkeypatch, door):
    """The fingerprint taken at the gesture's start says what the DAW holds only until something
    else pushes chains. A routing op does (perform_routing -> routing.apply), carrying the turn's
    proposals with it; an undo_proposal then walks the SESSION back to the start mix without a
    push. The start fingerprint matches again, but the DAW still holds the undone master.
    Both doors: a chat turn's routing push passes through counting_invoke as `routing.apply`, an
    approved plan's (mirror=True) through its `routed` branch."""
    brain, end, session = _brain(demo_project, monkeypatch, [
        MASTER_IT,
        ("daw_command", {"command": "return.add", "args": {"kind": "blank"}}),
        ("daw_command", {"command": "return.remove", "args": {"id": "fx"}}),   # the id add mints
        ("undo_proposal", {"proposal_id": ""}),
        _export(tmp_path)])
    start = B.mix_fingerprint(demo_project)
    ask = "master it, try a return, put it all back, export"
    if door == "chat":
        brain.chat(ask, mode="agent", use_llm=True)
    else:
        out = brain.chat(ask, mode="ask", use_llm=True)
        assert out.get("plan") and not end.seen("routing.apply"), out   # Ask touched nothing
        result = brain.apply_plan(out["plan"])
        assert result.get("ok"), result
    # Not vacuous, twice: the session really is back at the start mix (a return left behind would
    # move the fingerprint and force the push anyway)…
    assert B.mix_fingerprint(demo_project) == start, session.outputs
    # …and the routing pushes carried the proposal into the live graph before the undo.
    pre = [h for h in end.seen("routing.apply") if not h["sync"]]
    assert pre and SHELF in pre[-1]["master"], end.log
    held = end.seen("session.export")
    assert held and SHELF not in held[0]["master"] and TARGET not in held[0]["master"], (
        f"session.export ran on master {held[0]['master'] if held else None}: the undone shelf "
        f"was still in the live graph")


@pytest.mark.parametrize("command,args", [
    ("track.bounce", {"track": "stem_a"}),
    ("track.freeze", {"track": "stem_a", "on": True}),
])
def test_bounce_and_freeze_render_the_turns_mix_too(demo_project, monkeypatch, command, args):
    brain, end, _s = _brain(demo_project, monkeypatch,
                            [MASTER_IT, ("daw_command", {"command": command, "args": args})])
    brain.chat(f"master it, then {command}", mode="agent", use_llm=True)
    held = end.seen(command)
    assert held and SHELF in held[0]["master"] and TARGET in held[0]["master"], end.log


def test_an_approved_plan_realizes_before_its_export(demo_project, tmp_path, monkeypatch):
    """The other door: an Ask plan records propose + export, and "do it" replays them through
    counting_invoke(mirror=True) — the same wrapper, so the same realize."""
    brain, end, _s = _brain(demo_project, monkeypatch, [MASTER_IT, _export(tmp_path)])
    out = brain.chat("master it, then export", mode="ask", use_llm=True)
    assert out.get("plan") and not end.seen("session.export"), out   # Ask touched nothing
    result = brain.apply_plan(out["plan"])
    assert result.get("ok"), result
    held = end.seen("session.export")
    assert held and SHELF in held[0]["master"] and TARGET in held[0]["master"], end.log


def test_an_older_daw_without_coedits_take_still_gets_the_push(demo_project, tmp_path,
                                                               monkeypatch):
    """A DAW older than this brain answers coedits.take as an unknown command. Nothing to take;
    the realize goes ahead."""
    brain, end, _s = _brain(demo_project, monkeypatch, [MASTER_IT, _export(tmp_path)],
                            refuse={"coedits.take": "unknown command: coedits.take"})
    brain.chat("master it, then export", mode="agent", use_llm=True)
    held = end.seen("session.export")
    assert held and SHELF in held[0]["master"], end.log


def test_a_realize_the_daw_refused_is_told_to_the_engineer(demo_project, tmp_path, monkeypatch):
    """If the push fails the render still runs — on the old mix — so the command's reply says so
    and the engineer can tell the user, instead of reporting a master it did not print."""
    brain, end, session = _brain(demo_project, monkeypatch, [MASTER_IT, _export(tmp_path)],
                                 refuse={"routing.apply": "DAW disconnected"})
    brain.chat("master it, then export", mode="agent", use_llm=True)
    assert end.seen("session.export"), end.log
    told = dict(session.outputs).get("daw_command", "")
    assert "may be missing" in told and "session.export" in told, session.outputs


def test_the_render_commands_are_real_registry_keys():
    """A renamed key would silently fall out of the realize — pin the set to the registry."""
    keys = {k for k, _user, _confirm in O._registry_commands()}
    assert B.RENDER_COMMANDS <= keys, B.RENDER_COMMANDS - keys
