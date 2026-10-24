"""One turn of the DAW's resident engineer: what each of its tools DOES (Phase 55).

`daw_bridge.chat_turn` was one 600-line function - fifteen closures over fifteen shared
mutable locals - and `apply_plan`, which runs what an Ask turn recorded, carried a second copy
of most of what those closures did (propose and apply, the three commands that make a stem,
the guards a command passes), with the order of two guards reversed between the copies.

- `LiveTurn` is one turn: its state as fields, each tool as a method. `daw_bridge.chat_turn`
  builds one, runs it and returns its result; its signature and what it returns are as they
  were.
- The functions above the class are what a tool call does, whoever asks: the live turn and
  an approved plan's step both call them (`checked_command`, `registering_command`,
  `take_queued_moves`, `propose_and_apply`).

A name of `daw_bridge` is reached through the module (`B.session_readout`), never bound here
at import: tests and the evals patch that module, and a copy taken at import would not see it.
"""
from __future__ import annotations

import contextlib
import functools
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import anyio

from . import daw_bridge as B
from . import turn_tape
from .engine_client import in_memory_engine
from .chat import ChatTools


# ---------------------------------------------------------------- what a tool call does

def checked_command(project_dir: Path, command: str, args: Optional[dict],
                    edit: Optional[dict]) -> tuple[Optional[dict], dict]:
    """(a refusal, or None) and the arguments as the app is handed them.

    The questions every command the AGENT names is asked before anything else, in this
    order, at both doors a model's command enters the bridge (the live tool; a plan's step):
    is the key one only the brain may use (`brain_only_refusal`), what are its path
    arguments read against the project folder (`command_door.resolve_path_args`), and which
    scene does a scene NAME mean (`command_door.resolve_scene_names`, from the edit's
    `scenes[]`). The plan's copy resolved first and refused second."""
    refused = B.brain_only_refusal(command)
    if refused is not None:
        return refused, dict(args or {})
    sent = B.resolve_path_args(project_dir, command, args, (edit or {}).get("projects"))
    sent, why = B.resolve_scene_names(sent, (edit or {}).get("scenes"))   # a scene by its name (Q9)
    if why is not None:
        return {"ok": False, "error": why}, dict(sent or {})
    return None, sent


async def registering_command(project_dir: Path, command: str, args: Optional[dict], *,
                              daw_invoke, register_stem, mirror_coedits) -> Optional[dict]:
    """The reply of a command that makes a NEW STEM, registered by the brain as it lands -
    or None when `command` is not one, and the caller sends it on its own way.

    `track.add {path}`, `library.import` as a new track (no pad named), `track.duplicate` and
    `track.bounce`: the app's own background registration cannot run while a turn holds its
    thread, and the agent's next call wants to mix - or measure - what it just made."""
    args = args or {}
    if command == "track.add" and str(args.get("path") or "").strip():
        res = await B._import_sample_step(project_dir, str(args["path"]), daw_invoke,
                                          register_stem=register_stem,
                                          mirror_coedits=mirror_coedits)
    elif command == "library.import" and "track" not in args:
        res = await B._import_sample_step(project_dir, "", daw_invoke,
                                          register_stem=register_stem,
                                          mirror_coedits=mirror_coedits,
                                          library_args=dict(args))
    elif command == "track.bounce":
        # the app names the file it printed (`file`), as it does for a library item
        res = await B._import_sample_step(project_dir, "", daw_invoke,
                                          register_stem=register_stem,
                                          mirror_coedits=mirror_coedits,
                                          library_args=dict(args), daw_command="track.bounce")
    elif command == "track.duplicate":
        res = await B._duplicate_track_step(project_dir, args, daw_invoke,
                                            register_stem=register_stem,
                                            mirror_coedits=mirror_coedits)
    else:
        return None
    if isinstance(res, dict):
        res.pop("_coedits_mirrored", None)      # the step's own bookkeeping, nobody's to read
    return res


async def measured_command(command: str, res: Any, project_dir: Optional[Path] = None) -> Any:
    """The reply of a command whose result the brain MEASURES before the model reads it -
    `sound.render`: each file it made, profiled, in the same result (Phase 58) - and the two
    pack commands that render: what they answer with is the FINISHED file, so that is what is
    read (Phase 59). One implementation for the live tool and an approved plan's step.

    And a command whose answer is a PICTURE the model is shown (`view.snapshot`, Phase 64):
    read through the one door, beside the text of the same result. Only where a model reads
    the result - the live tool passes its project; an approved plan's step reports to the app,
    and a picture would mean nothing there."""
    if command in B.MEASURED_COMMANDS:
        return await anyio.to_thread.run_sync(B.attach_sound_profiles, res)
    if command in B.PICTURED_COMMANDS and project_dir is not None:
        return await anyio.to_thread.run_sync(B.attach_snapshot, res, project_dir)
    return res


async def take_queued_moves(take_user_moves) -> None:
    """Have the brain take the moves the user made since the agent's last command reply,
    before something is computed against the session (full audit 2026-10-02, reader S4·R3).
    A failure is logged: the turn goes on, and the idle flush still has them."""
    if take_user_moves is None:
        return
    try:
        await anyio.to_thread.run_sync(take_user_moves)
    except Exception as exc:  # noqa: BLE001
        print(f"[daw_bridge] could not take the DAW's queued moves "
              f"({type(exc).__name__}: {exc})", file=sys.stderr)


async def propose_and_apply(engine_call, take_user_moves, req: dict,
                            ) -> tuple[Optional[str], Optional[dict], Optional[dict]]:
    """(proposal id, the proposal's own answer, what applying it returned) - the id is None
    when the guardrails refused it, and then nothing was applied.

    The user's queued moves are taken FIRST: a proposal is applied brain-side with no DAW
    round trip, so it was computed against a session.json that had not heard the fader the
    user pulled while the model thought - and that move, an absolute value, then overwrote
    it. And a full pass budget (stale run leftovers, a long turn, a plan with more proposal
    steps than one pass allows) is not the agent's fault: a fresh pass is opened and the same
    proposal asked once more, instead of bouncing it back for ever."""
    await take_queued_moves(take_user_moves)
    result = await engine_call("propose_actions", {"req": req})
    if (result and not result.get("accepted") and result.get("violations")
            and all(v.get("rule") == "proposals_per_pass" for v in result["violations"])):
        await engine_call("advance_pass", {})
        result = await engine_call("propose_actions", {"req": req})
    pid = (result or {}).get("proposal_id")
    if not pid:
        return None, result, None
    applied = await engine_call("apply_proposal", {"proposal_id": pid})
    return pid, result, await with_phase_check(engine_call, req, applied)


def with_stemless_hint(result: dict, req: dict, edit) -> dict:
    """A proposal refused because it named a track the engine holds no stem for, with why and
    the door that does reach it (2026-10-13): "unknown stem 'Phase B'; stems: []" was all the
    engineer read when it tried to flip an instrument track's polarity, and it concluded the
    fix could not be made. `daw_bridge.stemless_track_hint` names `device.add`."""
    actions = req.get("actions") or []
    for violation in result.get("violations") or []:
        if violation.get("rule") != "unknown_stem":
            continue
        index = violation.get("action_index")
        if not isinstance(index, int) or not 0 <= index < len(actions):
            continue
        hint = B.stemless_track_hint(edit, {"stem_id": actions[index].get("stem_id")})
        if hint:
            return {**result, "hint": hint}
    return result


#: The action types that move a channel's stereo picture. A move of one of these comes back
#: with the phase check of what it touched (2026-10-13): the agent widened a pad to 1.40 to
#: clear masking, took its mids from +0.42 to +0.09 - below 0 a third of the time - and nothing
#: it read said so, while the master's imager flashed PHASE at the user.
STEREO_ACTION_TYPES = frozenset({"width", "imager", "band_width", "utility", "chorus", "flanger",
                                 "auto_filter"})   # Phase 63: its stereo offset


def stereo_targets(req: dict) -> list[str]:
    """The channels a proposal's stereo moves touch, in order, once each."""
    out: list[str] = []
    for action in req.get("actions") or []:
        kind = action.get("type")
        if kind in STEREO_ACTION_TYPES or (
                kind == "eq" and action.get("channel") in ("side", "left", "right")):
            target = B._move_target(action)
            if target not in out:
                out.append(target)
    return out


def phase_check(reading: dict, targets: list[str]) -> dict:
    """What a stereo move did, in the lines the agent needs: each touched channel's correlation
    per band, the issues on it and on the master, the conflicts it is in. `clean` is about
    those, not the whole mix (`analyze_phase` reads the whole mix)."""
    touched = set(targets)
    keep = ("target", "band", "kind", "cause", "corr", "below_zero_pct", "lag_ms",
            "from_stems", "fix")
    issues = [{k: v for k, v in issue.items() if k in keep and v not in (None, [])}
              for issue in reading.get("issues") or []
              if issue.get("target") in touched or issue.get("target") == "master"]
    conflicts = [c for c in reading.get("conflicts") or [] if touched & set(c.get("stems") or [])]
    rows = [row for row in reading.get("targets") or [] if row.get("target") in touched]
    if "master" in touched and reading.get("master"):
        rows.append(reading["master"])
    out = {"clean": not issues and not conflicts,
           "corr_by_band": {row["target"]: {band: b.get("corr") for band, b
                                            in (row.get("bands") or {}).items()}
                            for row in rows}}
    if issues:
        out["issues"] = issues
    if conflicts:
        out["conflicts"] = conflicts
    if reading.get("not_heard"):
        out["not_heard"] = reading["not_heard"]
    return out


async def with_phase_check(engine_call, req: dict, applied: Optional[dict]) -> Optional[dict]:
    """`applied`, and when the proposal moved a channel's stereo picture, what that did to its
    phase (`phase_check`) - measured before the model sees the result, as the repo's rule asks of
    a result the model needs measured. A failed check never fails the move: it says why."""
    targets = stereo_targets(req)
    if not targets or not isinstance(applied, dict) or not applied.get("ok", True):
        return applied
    channels = [name for name in targets if name != "master"]
    try:
        reading = await engine_call("analyze_phase",
                                    {"stem_ids": None if "master" in targets else channels})
    except Exception as exc:  # noqa: BLE001 - the move landed; only the reading of it did not
        return {**applied, "phase": {"error": f"{type(exc).__name__}: {exc}"}}
    if not isinstance(reading, dict):
        return applied
    return {**applied, "phase": phase_check(reading, targets)}


# ---------------------------------------------------------------- one turn

class LiveTurn:
    """One turn of the engineer in the DAW: its state, and each tool it may call.

    In **agent** mode a tool executes as it is called (shared control: guardrailed and
    undoable, no approval gate). In **ask** mode every tool that changes something records a
    plan step instead (`recorder`), which the DAW shows and can hand back to `apply_plan`."""

    def __init__(self, project_dir: Path, message: str, *, mode: str = "agent",
                 routing_ops=None, mirror_coedits=None, register_stem=None,
                 take_user_moves=None, use_llm: bool = True, edit_state=None,
                 daw_invoke=None, cache: Optional[dict] = None, on_event=None,
                 should_cancel=None, session_id: str = "", llm: Optional[dict] = None,
                 images: Optional[list] = None) -> None:
        self.project_dir, self.message, self.mode = project_dir, message, mode
        self.pictures = list(images or [])   # the user's, read by `chat_images` (Phase 64)
        self.routing_ops, self.mirror_coedits = routing_ops, mirror_coedits
        self.register_stem, self.take_user_moves = register_stem, take_user_moves
        self.use_llm, self.edit_state, self.daw_invoke = use_llm, edit_state, daw_invoke
        self.cache, self.on_event, self.should_cancel = cache, on_event, should_cancel
        self.session_id, self.llm = session_id, llm
        self.engine, self.looks, self.native = None, 0, None  # the engine for `run`; snapshots taken; the app's mix (`_native_mix`)
        # Each chat TURN is its own engine pass, opened at its first proposal (`do_propose`).
        self.turn_pass = {"opened": False}

        self.session = B._chat_session(self.project_dir, self.session_id)
        if self.llm:
            B._apply_turn_llm(self.session, self.llm)
        # Where this turn's seconds go (Phase 54, `turn_timing`): the clock starts here, before the
        # context is built, and the renders the turn makes are counted from here.
        from mix_engine import dsp as _dsp
        self.turn_began = time.monotonic()
        self.renders_before = dict(_dsp.RENDER_COUNTS)
        self.prep: dict = {"s": 0.0}
        # What this ONE turn costs. `session.usage` is cumulative for the life of the
        # conversation (it is the same ChatSession across turns), so the turn's own cost is
        # the delta across it — the run loop already reports `cost_usd` this way, and the
        # DAW's session rail sums both to say what a conversation has cost.
        # getattr, not session.usage: ScriptedChat (the no-LLM --rule-agent path) and the test
        # doubles have no usage counter, and a PRICE READOUT must never be able to kill the turn
        # it is pricing (_turn_cost_usd says exactly that, but this read sat outside its guard —
        # every scripted turn raised AttributeError before the agent ran, 2026-09-05).
        self.usage_before = B._usage_snapshot(self.session)
        # Replay anything the mid-run side channel answered while the previous turn was
        # still executing. Without this, a message typed during a run was answered once by
        # the stateless status path and then forgotten — the next real turn had never seen
        # it, so follow-ups looked ignored ("separated threads", 2026-08-09).
        B._replay_midrun(self.session, self.project_dir, self.session_id)
        # The recorder needs the edit AS THE USER SEES IT NOW, to pin what each
        # clip step points at (see _clip_fingerprint).
        self.recorder = B._PlanRecorder(B._live_edit(self.edit_state)) if self.mode == "ask" else None
        self.session.records_only = self.recorder is not None
        self.sandbox: dict = {}   # Ask: the in-memory session the recorded proposals are checked on
        self.reply_parts: list[str] = []
        self.moves: list[dict] = []
        # A run the agent starts from this turn (2026-09-10 audit P1-22): its spend, passes and
        # stop reason ride chat_done like a direct run's, instead of reaching only the model.
        self.run_totals: dict = {"cost_usd": 0.0, "passes": 0, "stop_reason": ""}
        # Action sets already applied THIS TURN (canonical JSON). Within one turn
        # nothing is re-measured, so an identical re-propose is the model looping —
        # tonight's failure (2026-08-09) applied the same duck/tune/eq trio three
        # times off the back of proposal-cap rejections. Across turns/passes the
        # same set stays legal (the deterministic agent iterates cumulative
        # same-band eq cuts between measurements).
        self.applied_action_keys: set[str] = set()
        self.before_seq = B.max_applied_seq(self.project_dir)
        self.invoked: list[str] = []          # daw_command keys this turn invoked (agent mode)
        # ...and every MOVE of the turn in the order it landed - a proposal by its id, a command
        # by its key - so an A/B can be about the last one (reader E6·R6).
        self.turn_moves: list[dict] = []

        self.stale_picture: dict = {}         # {"seq": n}: a reply named a snapshot newer than the one held
        self.measured_midturn = False         # a measured line was shown this turn
        self.readout_before: dict = {}        # the master as measured going INTO the turn
        # The measure a response's level commands OWE (Phase 55). A fader, a mute, a send or a knob
        # used to re-render and re-measure the whole mix when its own reply came back - ten mutes
        # in one response were ten measures, 63 s - though the model reads a response's results
        # together and cannot act between two of them. A command now only marks the measure owed;
        # `do_after_calls` takes it once, when the response's calls are all in.
        #   `before`  what the delta is against: the levels as last measured, or None when the mix
        #             has moved since without a measure (then the plain figures are said).
        self.owed: dict = {"commands": [], "before": None}
        # The levels as last measured this turn, and whether nothing has moved the mix since. NOT
        # `cache["readout"]`: the brain empties that dict at every mirror (`_mirror_coedits`), so
        # once a clip edit's report had been mirrored on its own, every level command after it in
        # the turn found no baseline and was never measured at all.
        self.levels: dict = {"last": {}, "exact": False}

        # Co-edits the turn could not mirror (2026-09-10 audit P2, 02·F13). The DAW has already
        # swapped them out of its queue into a command reply, so a refused one used to vanish —
        # the model saw {"ok": true}, chat_done said nothing, and the knob diverged from
        # session.json. A mirror that THREW lost the whole batch. Refusals now reach the tool
        # result and chat_done; a failed batch goes back to the DAW to re-queue.
        self.coedit_refusals: list[str] = []
        self.coedits_unmirrored: list[str] = []
        if self.mirror_coedits is not None:
            raw_mirror = self.mirror_coedits

            def mirror(urls: list[str]):
                try:
                    res = raw_mirror(urls)
                except Exception:
                    self.coedits_unmirrored.extend(urls)
                    raise
                if isinstance(res, dict) and res.get("failed"):
                    self.coedit_refusals.extend(str(e) for e in (res.get("errors") or []))
                return res
            self.mirror_coedits = mirror
        # The user's click on an A/B the agent offered last turn (ask_preference): record
        # it as taste and tell the model what was chosen — the text alone is ambiguous.
        try:
            self.answered = B.record_preference_answer(self.project_dir, self.message)
        except Exception as exc:  # noqa: BLE001 — taste is a bonus, never a blocker
            print(f"[daw_bridge] preference record skipped ({type(exc).__name__}: {exc})",
                  file=sys.stderr)
            self.answered = None
        if self.answered is not None:
            self.message = self.message + B.preference_note(self.answered)


    async def emit(self, event: dict) -> None:
        """Every event of the turn on its way to the app; the reply's words are kept."""
        t = event.get("type")
        if t == "chat_message" and event.get("text"):
            self.reply_parts.append(event["text"])
        if t in ("chat_delta", "chat_tool", "chat_message"):
            # the first thing of this turn the user can see (record v3)
            if (timer := getattr(self.session, "timing", None)) is not None:
                with contextlib.suppress(Exception):
                    timer.activity()
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:  # noqa: BLE001 — progress never kills a turn
                pass

    async def engine_call(self, name: str, args: dict):
        # A reading of the MIX on a project whose render cannot hear it is made on the app's
        # own render of the mix (Q13); every other call is the engine's alone.
        native = await self._native_mix() if B.reads_the_mix(name, args) else None
        if native is not None:
            from mix_engine import dsp
            on_app = dsp.native_mix(self.project_dir, native["audio"], native["rate"], native["fp"])
        else:
            on_app = contextlib.nullcontext()
        with on_app:
            result = await self.engine.call_tool(name, args or {})
        # Same unwrapping EngineClient.call does (loop.py): surface tool
        # errors, and peel the {"result": …} envelope the MCP layer adds.
        if result.isError:
            # A name the DAW knows and the engine does not (an instrument track) gets
            # the reason and the way round it, not the engine's bare `no such stem`.
            if "no such stem" in str(result.content):
                hint = B.stemless_track_hint(B._live_edit(self.edit_state), args)
                if hint:
                    raise RuntimeError(hint)
            # Worded as the run loop words it: the MCP content list printed as a repr put
            # pydantic's whole dump in the feed ("134 validation errors" for one Q, 2026-10-16).
            from .loop import tool_error_text
            raise RuntimeError(tool_error_text(name, result.content))
        payload = result.structuredContent
        if isinstance(payload, dict) and set(payload) == {"result"}:
            payload = payload["result"]
        return B.with_blind_spots(name, args, payload, B._live_edit(self.edit_state), self.project_dir,
                                  native=native is not None)

    async def _native_mix(self) -> Optional[dict]:
        """The app's render of the whole mix, cut at the master's last limiter (`mix.measure`),
        when the brain's render leaves out something it plays - or None, when the brain's render
        is the whole mix, the turn only records a plan, or the app could not render it. Kept
        until the session's render fingerprint moves or the turn moves the mix
        (`_mix_moved_unmeasured`). Never raises: a reading falls back to the brain's render,
        which then says what it leaves out."""
        if self.recorder is not None or self.daw_invoke is None:
            return None
        try:
            blind = B.render_blind_spots(B._live_edit(self.edit_state), B.arranged_held(self.project_dir))
            if not B.native_mix_owed(blind):
                return None
            fp = await anyio.to_thread.run_sync(B.render_fingerprint_of, self.project_dir)
            if self.native is not None and self.native["fp"] == fp:
                return self.native
            began = time.monotonic()
            res = await anyio.to_thread.run_sync(self.daw_invoke, "mix.measure", {})
            if not isinstance(res, dict) or not res.get("ok") or not res.get("file"):
                print(f"[daw_bridge] the app could not render the mix: "
                      f"{(res or {}).get('error') if isinstance(res, dict) else res}", file=sys.stderr)
                return None
            urls = res.get("coedits") or []
            if urls and self.mirror_coedits is not None:   # the limiter's make-up, as a fact
                await anyio.to_thread.run_sync(self.mirror_coedits, list(urls))
            audio, rate = await anyio.to_thread.run_sync(B.read_native_mix, res["file"])
            if (timer := getattr(self.session, "timing", None)) is not None:
                timer.mark("measure", time.monotonic() - began)
            fp = await anyio.to_thread.run_sync(B.render_fingerprint_of, self.project_dir)
            self.native = {"fp": fp, "audio": audio, "rate": rate}
            return self.native
        except Exception as exc:  # noqa: BLE001 — a reading is never the reason a turn fails
            print(f"[daw_bridge] the app's mix was not read ({type(exc).__name__}: {exc})",
                  file=sys.stderr)
            return None

    async def _take_user_moves(self) -> None:
        if self.take_user_moves is None:
            return
        try:
            await anyio.to_thread.run_sync(self.take_user_moves)
        except Exception as exc:  # noqa: BLE001 — the turn goes on; the idle flush still has them
            print(f"[daw_bridge] could not take the DAW's queued moves "
                  f"({type(exc).__name__}: {exc})", file=sys.stderr)

    def _note_command(self, command: str) -> None:
        self.invoked.append(command)
        self.turn_moves.append({"command": command})
        if not B.mix_neutral(command):
            self.native = None   # what the app plays may have moved; the session's fingerprint cannot say

    def _note_proposals_since(self, floor: int) -> None:
        for p in B.applied_proposals_since(self.project_dir, floor):
            if p.get("proposal_id"):
                self.turn_moves.append({"proposal_id": p["proposal_id"]})

    def _mix_moved_unmeasured(self) -> None:
        """Something changed what the mix sounds like and nothing measured it: the next
        measure has no exact figure to be a delta of - and the app's render of the mix, if one
        was taken, is of the mix before it."""
        self.levels["exact"] = False
        self.native = None

    async def do_propose(self, req: dict) -> dict:
        if self.recorder is not None:
            refused = await anyio.to_thread.run_sync(
                B._sandbox_check, self.project_dir, self.sandbox, req,
                any(st.get("kind") != "proposal" for st in self.recorder.steps))
            if refused is not None:
                return refused
            return {"accepted": True, "status": self.recorder.record(
                "proposal", req,
                "recorded — checked against the guardrails; the user will "
                "approve it before it runs")}
        action_key = json.dumps(req.get("actions", []), sort_keys=True)
        if action_key in self.applied_action_keys:
            return {"accepted": False, "violations": [],
                    "status": "skipped — this exact action set already "
                              "applied this turn; do not resubmit it, "
                              "adjust the values or move on"}
        if not self.turn_pass["opened"]:
            # Each chat TURN is its own pass: pass_number otherwise persisted across turns,
            # so the composed EQ rail's +6 dB PER-PASS budget spanned the whole chat and
            # refused "a bit more" the +12 session budget allowed (2026-09-11 pre-merge review).
            self.turn_pass["opened"] = True
            await self.engine_call("advance_pass", {})
        # The user's queued moves first, then propose, a fresh pass when only the pass's
        # own budget refused it, then apply: `propose_and_apply`, shared with a plan's step.
        pid, result, applied = await propose_and_apply(self.engine_call, self.take_user_moves, req)
        if not pid:
            return with_stemless_hint(result or {"accepted": False}, req,
                                      B._live_edit(self.edit_state))
        self._mix_moved_unmeasured()
        self.applied_action_keys.add(action_key)
        self.turn_moves.append({"proposal_id": pid})
        for action in req.get("actions", []):
            self.moves.append({"stem": B._move_target(action),
                          "type": action.get("type", ""),
                          "params": {k: v for k, v in action.items()
                                     if k not in ("type", "stem_id")
                                     and v is not None},
                          "rationale": (req.get("rationale") or "").strip()})
        return {"accepted": True, "proposal_id": pid,
                "status": "applied — undoable", "result": applied}

    async def do_look(self, command: str, args: dict):
        """A picture of the window (`view.snapshot`, Phase 64): a LOOK, so it runs in Ask mode as in
        Agent - recorded as a plan step it would show the model nothing - and is neither a move nor
        a gesture. A turn takes `daw_bridge.SNAPSHOTS_PER_TURN`: each one is a picture re-sent with
        every later request of the conversation."""
        if self.looks >= B.SNAPSHOTS_PER_TURN:
            return {"ok": False, "error": (
                f"{command}: this turn has already looked at the window {self.looks} times - say "
                "what you saw, or ask the user")}
        if self.daw_invoke is None:
            return {"ok": False, "error": "no DAW attached"}
        res = await anyio.to_thread.run_sync(self.daw_invoke, command, args)
        res = await measured_command(command, res, self.project_dir)
        if not (isinstance(res, dict) and res.get("ok") is False):
            self.looks += 1
        return res

    async def do_daw_command(self, command: str, args: dict):
        # The two questions every agent command is asked first, by one function for the
        # live tool and for an approved plan's step (`checked_command`): is its key the
        # brain's alone (S9·R6), and what are its paths read against the project folder
        # (reader E7·R2) - above the Ask recorder, so a plan runs the path the user was shown.
        asked_args = args
        refused, args = checked_command(self.project_dir, command, args,
                                        B._live_edit(self.edit_state))
        if refused is not None:
            return refused
        if command in B.PICTURED_COMMANDS:
            return await self.do_look(command, args)
        if self.recorder is not None:
            recorded = {"ok": True, "status": self.recorder.record(
                "command", {"command": command, "args": args},
                "recorded — the user will approve it before it runs")}
            if (field := B._RECORDED_ID_FIELDS.get(command)) is not None:
                # A later step needs the id this one will mint (2026-09-10 audit
                # P2, 02·F7): it used to come back with none, so "send the vocal to
                # the new return" had to guess. apply_plan fills it in.
                recorded[field] = f"$step{len(self.recorder.steps)}.{field}"
                recorded["status"] += (f"; later steps name what it creates as "
                                       f"{recorded[field]} (filled in when it runs)")
            return recorded
        if self.daw_invoke is None:
            return {"ok": False, "error": "no DAW attached"}
        # A command that makes a NEW STEM (a track added from a file, a library item as a
        # track, a duplicate) is registered by the brain inside the turn - the agent's next
        # call wants to mix it, and the app's own background registration cannot run while
        # the turn holds its thread (2026-09-04; reader C1·R3). One implementation, shared
        # with an approved plan (`registering_command`).
        res = await registering_command(self.project_dir, command, args,
                                        daw_invoke=self.daw_invoke,
                                        register_stem=self.register_stem,
                                        mirror_coedits=self.mirror_coedits)
        if res is not None:
            if res.get("ok") is not False:
                self._note_command(command)
            if res.get("ok"):
                self._mix_moved_unmeasured()
                if self.cache is not None:
                    self.cache.clear()
            B._note_stale_picture(res, B._live_edit(self.edit_state), self.stale_picture)
            return res
        # A look inside ANOTHER project is the brain's own read of that folder. Sent to
        # the DAW it could only queue behind this turn, and its listing lands in the
        # browser, where you cannot read it: "what's in older_song?" had no route at all
        # (full audit 2026-10-02, readers C4·R6 = C2·R2).
        if command == "project.peek":
            peeked = await anyio.to_thread.run_sync(
                B.peek_project_for_agent, str((args or {}).get("dir") or ""))
            if peeked.get("ok") is not False:
                self._note_command(command)
            return peeked
        # routing the brain performs itself mid-turn (see perform_routing)
        routed = await anyio.to_thread.run_sync(
            B.perform_routing, command, args, self.routing_ops, self.daw_invoke, self.mirror_coedits)
        if routed is not None:
            if routed.get("ok") is not False:
                self._note_command(command)
                self._mix_moved_unmeasured()
            return routed
        res = await anyio.to_thread.run_sync(self.daw_invoke, command, args)
        res = await measured_command(command, res, self.project_dir)
        if isinstance(res, dict) and (moved := B.export_redirect_note(command, asked_args, args)):
            res.setdefault("note", moved)   # an export named at the project's top went to bounces/ (D6)
        if not (isinstance(res, dict) and res.get("ok") is False):
            self._note_command(command)
        B._note_stale_picture(res, B._live_edit(self.edit_state), self.stale_picture)
        # The DAW's reply carries the co-edits this command queued (a fader, a
        # device knob): mirror them NOW, so this round's picture is current, and
        # measure the master — the agent sees the effect on THIS tool result.
        urls = res.pop("coedits", None) if isinstance(res, dict) else None
        if urls and self.mirror_coedits is not None:
            try:
                m = await anyio.to_thread.run_sync(self.mirror_coedits, list(urls))
            except Exception as exc:  # noqa: BLE001 — handed back to the DAW at chat_done
                print(f"[daw_bridge] mid-turn co-edit mirror failed "
                      f"({type(exc).__name__}: {exc})", file=sys.stderr)
                res["coedits_unmirrored"] = len(urls)
            else:
                if isinstance(m, dict) and m.get("failed"):
                    # the model must see it: the knob is live, session.json refused it
                    res["coedit_refusals"] = list(m.get("errors") or [])
                # A move is measured; a report of the engine's facts is not (Phase 54) -
                # and a move is measured once per RESPONSE, not here (Phase 55).
                if all(B.coedit_is_fact_report(u) for u in urls):
                    if not self.owed["commands"]:
                        self._mix_moved_unmeasured()   # where a clip plays is what a render reads
                else:
                    if not self.owed["commands"]:
                        self.owed["before"] = dict(self.levels["last"]) if self.levels["exact"] else None
                    self.owed["commands"].append(command)
        return res

    async def do_after_calls(self) -> str:
        """What the brain has to say once a response's tool calls are all in: the
        master as the response's level commands left it, measured ONCE. A delta
        against the levels as last measured; the plain figures when something moved
        the mix unmeasured since (a clip edit, a mix move, a new track); nothing when
        no measure is owed or nothing moved. Never raises: a measurement is a
        readout, and the turn goes on without it."""
        if self.recorder is not None or not self.owed["commands"]:
            return ""
        before = self.owed["before"]
        self.owed["commands"].clear()
        self.owed["before"] = None
        try:
            measure_began = time.monotonic()
            self.native = None        # the level commands have moved what the app plays
            native = await self._native_mix()
            if native is None:
                after = await anyio.to_thread.run_sync(B.level_readout, self.project_dir)
            else:
                from mix_engine import dsp
                with dsp.native_mix(self.project_dir, native["audio"], native["rate"], native["fp"]):
                    after = await anyio.to_thread.run_sync(B.level_readout, self.project_dir)
            if (timer := getattr(self.session, "timing", None)) is not None:
                timer.mark("measure", time.monotonic() - measure_began)
        except Exception as exc:  # noqa: BLE001
            print(f"[daw_bridge] mid-turn measure skipped "
                  f"({type(exc).__name__}: {exc})", file=sys.stderr)
            self._mix_moved_unmeasured()
            return ""
        self.levels["last"], self.levels["exact"] = dict(after), True
        if before:
            d = B.turn_delta(before, after)
            if not d.get("changed"):
                return ""
            self.measured_midturn = True
            await self.emit({"type": "chat_measured", "turn_id": "",
                        "was": d["was"], "now": d["now"]})
            return d["text"].replace("measured after your last turn",
                                     "measured after these commands")
        said = B.levels_text(after)
        if not said:
            return ""
        self.measured_midturn = True
        return ("measured after these commands (the master as it stands; the mix "
                "moved earlier in this turn without a measurement, so this is not a "
                "before-and-after): " + said)

    async def do_ask_preference(self, about: str, move: str = "") -> None:
        # raises ValueError (the tool's refusal) when there is no such move - and it
        # runs BEFORE the feed's card is shown (chat._bind_turn_tools)
        B.offer_preference(self.project_dir, about=about,
                         move=B.preference_move(self.project_dir, move, self.turn_moves))

    async def do_list_files(self, path: str, query: str = ""):
        # On a worker thread (the pre-merge review of 2026-10-23): a listing asks the disk, and a
        # sleeping share or a drive going away held the turn's event loop for as long as it took.
        return await anyio.to_thread.run_sync(B.folder_listing, path, query)

    async def do_read_track(self, name: str, clip=None, from_note=None):
        return B.read_track_entry(B._live_edit(self.edit_state), name, clip, from_note)

    async def do_start_run(self, brief: str, gating: str, max_passes: int) -> str:
        if self.recorder is not None:
            # The pass count is part of what the user approves (reader E6·R5): it
            # was dropped HERE, so "one quick pass" in Ask mode ran the full five on
            # the user's key once they clicked "do it". run_agent_pass clamps it too.
            from .loop import MAX_PASSES
            passes = (MAX_PASSES if not max_passes
                      else max(1, min(int(max_passes), MAX_PASSES)))
            return self.recorder.record(
                "run", {"brief": brief, "max_passes": passes},
                f"recorded — will run the mix loop on {brief!r}, "
                f"{passes} pass{'' if passes == 1 else 'es'} at most")
        # should_cancel MUST reach the loop. A run the agent starts from a
        # chat turn is the longest, most expensive thing the DAW does, and
        # without this the user's Cancel returned `cancelled: 0` while the
        # run billed to completion behind a spinner.
        # …and its live feed (P1-22): the same run_progress a direct run sends — pass
        # N/M, the ticking $, each move as it lands. These run on the pass's worker
        # thread; on_event is the brain's thread-safe forwarder.
        def _progress(event: dict) -> None:
            if self.on_event is not None:
                try:
                    self.on_event({"type": "run_progress", **event})
                except Exception:  # noqa: BLE001 — progress never kills a run
                    pass

        run_floor = B.max_applied_seq(self.project_dir)
        info = await anyio.to_thread.run_sync(
            functools.partial(B.run_agent_pass, self.project_dir, brief,
                              use_llm=self.use_llm, edit_state=self.edit_state,
                              daw_invoke=self.daw_invoke,
                              should_cancel=self.should_cancel,
                              max_passes=max_passes,   # F7: was accepted and dropped
                              on_move=lambda mv: _progress({"moves": [mv]}),
                              on_phase=lambda ph: _progress(dict(ph)),
                              # the conversation's own model + effort: the run the
                              # engineer starts runs on what the user picked (E·F1)
                              llm=dict(getattr(self.session, "settings", None) or {})))
        self.moves.extend(B.moves_since(self.project_dir, self.before_seq))
        self._mix_moved_unmeasured()
        self._note_proposals_since(run_floor)
        self.run_totals["cost_usd"] += float(info.get("cost_usd") or 0.0)
        self.run_totals["passes"] += int(info.get("passes") or 0)
        self.run_totals["stop_reason"] = str(info.get("stop_reason") or "")
        return json.dumps(info)

    async def do_auto_mix(self, target: str) -> str:
        # A project whose only parts are MIDI has nothing Auto Mix can read: the agent is told
        # what to say and what to offer (print them first), and nothing runs or is recorded.
        if not B.mixable_stems(self.project_dir):
            parts = B.midi_parts(self.edit_state)
            if parts:
                return B.midi_print_offer_for_agent(parts)
        if self.recorder is not None:
            return self.recorder.record("auto_mix", {"target": target},
                                   "recorded — will run Auto Mix")
        if self.should_cancel is not None and self.should_cancel():
            return "cancelled before Auto Mix started"
        # should_cancel MUST reach the bundle loop, not just guard the start
        # (2026-08-05 review). Auto Mix is minutes of profiling and rendering;
        # a pre-check only made Cancel work for the instant before it began.
        # What it ACTUALLY did (2026-09-11 pre-merge review, C3). This said
        # "Auto Mix applied (undoable)" whatever happened: every refused bundle was
        # dropped on the floor (`refused=` was never passed), a pass that applied
        # NOTHING still read as applied, and a Stop part-way reported success. The
        # apply_plan call site has reported all three since 2026-09-10; this one is
        # the same call, so it says the same things.
        refused: list = []
        notes: list = []      # why it did nothing at all (nothing here was split)
        left_out: list = []   # stems the cleanup's de-noise was not applied to
        mix_floor = B.max_applied_seq(self.project_dir)
        applied = await anyio.to_thread.run_sync(
            functools.partial(B.auto_mix_pass, self.project_dir, target,
                              should_cancel=self.should_cancel, refused=refused,
                              notes=notes, left_out=left_out))
        self.moves.extend(B.moves_since(self.project_dir, self.before_seq))
        self._mix_moved_unmeasured()
        self._note_proposals_since(mix_floor)
        return B.auto_mix_said(int(applied or 0), refused=refused, notes=notes,
                             left_out=left_out,
                             stopped=bool(self.should_cancel is not None and self.should_cancel()))

    async def do_save_note(self, note: str) -> str:
        """Per-project memory. Must be `runs/notes.jsonl`, one
        {"ts", "note"} object per line — the shape loop._load_notes and
        gateway._load_notes read. This wrote a `notes.json` ARRAY to a
        filename nothing reads, so "remember I always want the vocal
        bright" answered "noted" and was never seen again."""
        runs_dir = self.project_dir / "runs"
        runs_dir.mkdir(exist_ok=True)
        with (runs_dir / "notes.jsonl").open("a") as handle:
            handle.write(json.dumps({"ts": round(time.time(), 3),
                                     "note": note}) + "\n")
        return f"noted: {note}"

    async def do_undo_proposal(self, proposal_id: str) -> str:
        """Take back one applied move. Without this the agent could not
        answer "no, put that back" by ANY path mid-turn: the tool was
        wired to `unsupported`, and daw_command("agent.undo") cannot run
        inside the turn - a pass is active for the whole of it, so it queues
        behind the turn (App.cpp queueAfterTurn)."""
        if self.recorder is not None:
            return self.recorder.record("undo", {"proposal_id": proposal_id},
                                   "recorded — will undo that move")
        await self._take_user_moves()     # an undo replays the user's co-edits: have them all (S4·R3)
        record = B._proposal_record(self.project_dir, proposal_id)
        undone, reason = await anyio.to_thread.run_sync(
            functools.partial(B._undo_one, self.project_dir, proposal_id))
        if undone is None:
            return (f"nothing undone — {reason}. Check get_proposals for "
                    "what is applied and in what order.")
        self._mix_moved_unmeasured()
        # "No, put that back" is the user's verdict on the move (2026-09-10 audit
        # P2, 05·F11): it used to record nothing, and the move was banked as KEPT.
        # An A/B answer recorded its own verdict already — don't count it twice.
        if self.answered is None and record is not None and record.get("proposal_id") == undone:
            try:
                B.record_taste(self.project_dir, "undone", [record])
            except Exception as exc:  # noqa: BLE001 — taste never blocks an undo
                print(f"[daw_bridge] taste (undone) skipped: {exc}", file=sys.stderr)
        self.moves.extend(B.moves_since(self.project_dir, self.before_seq))
        return f"undone: {undone}"

    async def do_import_sample(self, path: str) -> str:
        """Import a local audio file as a new stem.

        Was wired to `unsupported` (2026-08-05 audit) — advertised in the chat
        prompt and always refused. It routes through the DAW's own `track.add`,
        which already does BOTH halves: places the clip on a new track AND
        registers the file with the brain (`dropAudioClip` -> "register", which
        copies/decodes it and re-tags). Calling `import_stem` here as well would
        register the same file twice. Off the DAW (no invoke channel) the brain
        side is all there is, so do it directly.
        """
        # Ask mode is an APPROVAL gate: record, don't run. Every other
        # mutating tool here opens with this (do_propose, do_daw_command,
        # do_start_run, do_auto_mix, do_undo_proposal) — this one was
        # rewired from `unsupported` to functional in the 2026-08-05
        # audit and the gate was missed, so an approval-gated turn really
        # imported the file while narrating "will apply once you approve"
        # (2026-08-16 review).
        if self.recorder is not None:
            return json.dumps({"ok": True, "status": self.recorder.record(
                "import", {"path": path},
                "recorded — the user will approve it before it runs")})
        result = await B._import_sample_step(
            self.project_dir, path, self.daw_invoke,
            register_stem=self.register_stem, mirror_coedits=self.mirror_coedits)
        if result.get("ok"):
            self._mix_moved_unmeasured()
            self.cache.clear() if self.cache is not None else None
        return json.dumps(result)

    async def do_render(self) -> str:
        """Re-bounce and re-measure. Read-only — it renders what is
        already applied. Without it the agent narrated results it had
        never measured, because its context numbers are frozen from
        before its own changes."""
        self.cache.clear() if self.cache is not None else None
        # On a project the brain's render cannot hear whole, the numbers are of the app's own
        # render of the mix (Q13) - and then nothing is left out of them.
        native = await self._native_mix()
        if native is None:
            readout = await anyio.to_thread.run_sync(
                functools.partial(B.session_readout, self.project_dir, self.cache))
        else:
            from mix_engine import dsp
            with dsp.native_mix(self.project_dir, native["audio"], native["rate"], native["fp"]):
                readout = await anyio.to_thread.run_sync(
                    functools.partial(B.session_readout, self.project_dir, self.cache))
        # …and what those numbers leave out (render_blind_spots): absent when the
        # project is only stems and their chains, which is when they are the whole mix.
        if isinstance(readout, dict):
            if not self.owed["commands"]:   # a measurement the agent asked for is one
                self.levels["last"], self.levels["exact"] = dict(readout), True
            if native is not None:
                readout = {**readout, "measured_on": B.MEASURED_ON_APP}
            else:
                blind = B.render_blind_spots(B._live_edit(self.edit_state), B.arranged_held(self.project_dir))
                if blind:
                    readout = {**readout, "not_heard": blind}
        return json.dumps(readout)

    async def unsupported(self, *_args, **_kwargs) -> str:
        return ("not available in the DAW — use daw_command "
                "(see daw_commands in your context)")

    def tools(self) -> ChatTools:
        return ChatTools(
            engine_call=self.engine_call,
            propose=self.do_propose,
            start_run=self.do_start_run,
            stop_run=self.unsupported,
            auto_mix=self.do_auto_mix,
            file_instruction=self.unsupported,
            save_note=self.do_save_note,
            workspace_command=self.unsupported,
            list_files=self.do_list_files,
            import_sample=self.do_import_sample,
            undo_proposal=self.do_undo_proposal,
            render=self.do_render,
            daw_command=self.do_daw_command,
                          ask_preference=self.do_ask_preference,
                          read_track=self.do_read_track,
                          after_calls=self.do_after_calls
        )

    async def run(self) -> None:
        """The turn itself: the engine for its length, the context, the model's loop."""
        async with in_memory_engine(self.project_dir) as engine:
            self.engine = engine
            # Two halves with different lifetimes (2026-08-05 audit). The stable
            # half is built once and cached; the live edit is re-read per round,
            # because the agent's own daw_commands change it — the DAW pushes a
            # fresh snapshot before replying to each one, so `edit_state()` is
            # current by the time the next round asks for it.
            stable = B._chat_context_stable(self.project_dir, self.mode, self.cache,
                                          B._live_edit(self.edit_state))
            # the readout the stable context just measured (cache-served)
            self.readout_before.update(dict((self.cache or {}).get("readout") or {}))
            self.levels["last"], self.levels["exact"] = dict(self.readout_before), bool(self.readout_before)
            self.prep["s"] = time.monotonic() - self.turn_began
            # passed only when there are some: a test's scripted session need not know the word
            pictured = {"images": self.pictures} if self.pictures else {}
            await self.session.turn(
                self.message, tools=self.tools(),
                context=stable,
                live_context=lambda: B._chat_context_live(B._live_edit(self.edit_state),
                                                        self.stale_picture.get("seq", 0)),
                emit=self.emit, should_cancel=self.should_cancel, **pictured)

    def note_failure(self, exc: BaseException) -> None:
        """Hang on a turn's failure what it billed and where its seconds went.

        A turn that fails after billed rounds says what it billed (2026-09-22 audit A·F4 /
        E·F5): the next turn's delta starts after these tokens, so they were never priced
        anywhere — the session rail read $0.00 for a 10-round build that died on round 11.
        And a turn that died recorded no timing (Phase 55), so the slowest and most
        expensive turns were the ones the timing file never held."""
        try:
            exc.cost_usd = round(B._turn_cost_usd(self.session, self.usage_before)   # type: ignore[attr-defined]
                                 + float(self.run_totals.get("cost_usd") or 0.0), 4)
        except Exception:  # noqa: BLE001 — a price is a readout, never a blocker
            pass
        turn_tape.write(self.project_dir, self.session)      # a failed turn's tape too
        try:
            timing = B._turn_timing(self.session, self.session_id, self.message,
                                    self.reply_parts, self.turn_began, self.prep["s"],
                                    self.renders_before, pictures=len(self.pictures))
            timing["failed"] = type(exc).__name__
            timing["cost_usd"] = getattr(exc, "cost_usd", None)
            exc.timing = timing   # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 — a readout, never a blocker
            pass

    def result(self) -> dict:
        """What the turn hands back: the reply, the moves, the price, its timing record, what
        it could not mirror, the plan an Ask turn recorded - and the delta the brain settles
        when the turn ends."""
        turn_tape.write(self.project_dir, self.session)
        reply = "\n\n".join(p for p in self.reply_parts if p.strip())
        out: dict = {"reply": reply, "moves": self.moves,
                     "cost_usd": B._turn_cost_usd(self.session, self.usage_before)}
        try:
            out["timing"] = B._turn_timing(self.session, self.session_id, self.message, self.reply_parts, self.turn_began,
                                         self.prep["s"], self.renders_before,
                                         pictures=len(self.pictures))
        except Exception as exc:  # noqa: BLE001 — a readout, never a blocker
            print(f"[daw_bridge] turn timing skipped ({type(exc).__name__}: {exc})", file=sys.stderr)
        if self.coedit_refusals:
            out["coedit_refused"] = len(self.coedit_refusals)
            out["coedit_refusals"] = self.coedit_refusals[:3]
        if self.coedits_unmirrored:
            out["coedits_unmirrored"] = list(dict.fromkeys(self.coedits_unmirrored))
        if self.run_totals["passes"] or self.run_totals["stop_reason"]:
            out["cost_usd"] = round(out["cost_usd"] + self.run_totals["cost_usd"], 4)
            out["passes"] = self.run_totals["passes"]
            out["stop_reason"] = self.run_totals["stop_reason"]
        try:
            B.record_session_digest(self.project_dir, session_id=self.session_id, user=self.message,
                                  reply=reply, commands=self.invoked, reply_parts=self.reply_parts)
        except Exception as exc:  # noqa: BLE001 — memory is a bonus, never a blocker
            print(f"[daw_bridge] session digest skipped ({type(exc).__name__}: {exc})",
                  file=sys.stderr)
        if (self.invoked or self.moves) and self.readout_before:
            # Settled by Brain.chat at chat_done, ALWAYS (2026-09-10 audit P2, 02·F12). Every
            # command reply carries the co-edits it queued, and they are mirrored mid-turn, so
            # the brain is current at chat_done. A command with no co-edits (a clip edit) is
            # one the brain's stem render cannot hear — waiting for "the next co-edit batch"
            # meant a user's fader move minutes later was reported as the agent's effect.
            # `measured_midturn` suppresses the turn-level FEED row (the per-command rows
            # already said it); the context sentence stays.
            out["turn_delta_pending"] = {"before": self.readout_before, "commands": self.invoked,
                                         "settle": "chat_done",
                                         # a measure still owed (Stop, a host with no hook) was
                                         # never shown: the turn's own row is then the only one
                                         "measured_midturn": self.measured_midturn and not self.owed["commands"],
                                         "ts": B._now_iso()}
        if isinstance(out.get("timing"), dict):
            out["timing"]["cost_usd"] = out["cost_usd"]   # with a chat-started run's, as the feed shows it
        if self.recorder is not None:
            # advised_seq: the state this plan was PROPOSED against. Most step kinds replay
            # verbatim with no per-target staleness check (only clip-shaped steps carry
            # fingerprints), so apply_plan compares this stamp and says so honestly when the
            # project moved between preview and approval (2026-08-09 §4 audit).
            out["plan"] = {"narration": reply, "steps": self.recorder.steps,
                           "advised_seq": self.before_seq}
        return out
