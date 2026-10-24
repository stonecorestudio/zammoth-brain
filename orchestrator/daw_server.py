"""Mix Agent DAW — live brain↔DAW bridge (docs/10-daw.md, Milestone B).

A small **NDJSON socket server**: the DAW's chosen transport to the brain (option C).
The param push to the graph is in-process (C++ ``MonitorChain.setChain``); this channel
carries only event-driven agent *actions* + streamed results — NOT the gateway's
``/api/audition`` poll (we do not port the gateway plumbing). One persistent localhost
connection per DAW; the brain's real agent logic is reused via ``daw_bridge``.

Protocol — newline-delimited JSON, one object per line:
    auth      {"id": N, "method": "auth", "params": {"token": "<from the handshake>"}}
              MUST be the first line of every connection (2026-09-10 agent audit P0-2).
              Anything else first (an HTTP request line, any other method, a wrong
              token) closes the connection with nothing executed.
    request   {"id": N, "method": "run"|"get_chains"|"hello", "params": {...}}
    response  {"id": N, "result": {...}}   or   {"id": N, "error": "..."}
    event     {"event": "run_started"|"run_done", ...}     (unsolicited, server→DAW)

Resolved chains use the existing ``monitor_chain_params`` contract the C++ node applies.

    python -m orchestrator.daw_server <project_dir> [--port N]
    python <repo>/orchestrator/daw_server.py <project_dir>     # direct (no -m)

Prints ``MIXAGENT_LISTENING port=<port> token=<token>`` on stdout once bound — the
handshake the DAW reads to learn the ephemeral port and this process's per-spawn
secret. The DAW redirects that stdout into ``~/Library/Logs/Zammoth/brain.log`` (a 0600
file inside the user's 0700 ``~/Library``): a web page cannot read it and another macOS
user cannot open it, the two callers the old open port let in.
"""
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import hmac
import itertools
import json
import math
import re
import os
import secrets
import sys
import threading
import time
from pathlib import Path

# Runnable both as a package module (-m) and as a direct script (the DAW spawns the
# file by absolute path so juce::ChildProcess captures stdout cleanly — no shell).
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from orchestrator.arrangement import (ArrangementError, apply_arrangement,
                                          arrangement_summary,
                                          build_arrangement_plan,
                                          is_arrangement_brief)
    from orchestrator.daw_bridge import (LATEST_EDIT, apply_owed_coedits, coedit_batches_done, resolve_sent, ensure_saved_checkpoint, honour_discard, set_aside_unreadable_session, level_inverses, level_snapshot, level_values,
                                         _master_makeup, _master_resolve_inputs, advise, apply_coedits,
                                         auto_mix_pass, midi_parts, midi_print_offer_said, mixable_stems,
                                         print_midi_parts, brief_not_understood,
                                         chains_with_readout, chat_fallback_cost, chat_reply, chat_status_reply, strip_state, applied_seqs,
                                         chat_turn, apply_plan, record_midrun_exchange,
                                         record_taste, PENDING_PREFERENCE_FILE, pending_preference_proposal,
                                         applied_proposals_since,
                                         import_chain_from, import_stem, import_track_from, keep_strip_over_copy, register_stem_copy,
                                         extract_clip_groove,
                                         match_clip_to_key, tune_clip_notes, moves_since, peek_project,
                                         reference_match, separate_track, session_observations, warm_observations,
                                         max_applied_seq, resolve_chains, run_agent_pass,
                                         session_readout, transpose_to_target_key,
                                         perform_routing, NotBrainRouting, undo_last, undo_run,
        turn_delta, write_turn_delta,
        settle_turn_delta, write_atomic, RENDER_COMMANDS, mix_fingerprint)
    from orchestrator import chat_images
    from orchestrator.chat import PREFERENCE_OPTIONS
    from orchestrator.llm_backend import (NoRetry, active_backend, drop_rerouting_env,
                                          retry_helps)
else:
    from .arrangement import (ArrangementError, apply_arrangement,
                              arrangement_summary, build_arrangement_plan,
                              is_arrangement_brief)
    from .daw_bridge import (LATEST_EDIT, apply_owed_coedits, coedit_batches_done, resolve_sent, ensure_saved_checkpoint, honour_discard, set_aside_unreadable_session, level_inverses, level_snapshot, level_values,
                             _master_makeup, _master_resolve_inputs, advise, apply_coedits, auto_mix_pass,
                             midi_parts, midi_print_offer_said, mixable_stems, print_midi_parts,
                             brief_not_understood, chains_with_readout, chat_fallback_cost, chat_reply, chat_status_reply, strip_state, applied_seqs,
                             chat_turn, apply_plan, record_midrun_exchange,
                             record_taste, PENDING_PREFERENCE_FILE, pending_preference_proposal,
                             applied_proposals_since,
                             import_chain_from, import_stem, import_track_from, keep_strip_over_copy, register_stem_copy,
                             extract_clip_groove,
                             match_clip_to_key, tune_clip_notes,
                             max_applied_seq, moves_since, peek_project,
                             reference_match, session_observations, warm_observations,
                             resolve_chains, run_agent_pass, separate_track, session_readout,
                             transpose_to_target_key, perform_routing, NotBrainRouting, undo_last, undo_run,
    turn_delta, write_turn_delta,
        settle_turn_delta, write_atomic, RENDER_COMMANDS, mix_fingerprint)
    from . import chat_images
    from .chat import PREFERENCE_OPTIONS
    from .llm_backend import NoRetry, active_backend, drop_rerouting_env, retry_helps

HOST = "127.0.0.1"
HANDSHAKE_PREFIX = "MIXAGENT_LISTENING port="


# ---- the wire is STRICT json (full audit 2026-10-02, F5) ----------------------------------
# juce::JSON::parse refuses a whole document that contains `NaN`, `Infinity`, `-Infinity`, a
# string holding U+0000, or a lone surrogate - and the DAW's read loop treated a refused line as
# "not for me" and dropped it, so the request it answered sat until its timeout and reported
# "brain disconnected or timed out". json.dumps emits all three tokens by default. So nothing
# non-finite, no NUL and no lone surrogate leaves this process: a non-finite number goes out as
# null (the DAW reads a missing metric, not a missing message).
_UNSAFE_WIRE = re.compile(r"\\u0000|\\ud[89ab][0-9a-f]{2}(?!\\ud[c-f][0-9a-f]{2})"
                          r"|(?<!\\ud[89ab][0-9a-f]{2})\\ud[c-f][0-9a-f]{2}", re.I)


def _wire_clean(v):
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, str):
        if "\x00" in v or any(0xD800 <= ord(c) <= 0xDFFF for c in v):
            v = v.replace("\x00", "").encode("utf-16", "surrogatepass").decode("utf-16", "replace")
        return v
    if isinstance(v, dict):
        return {(_wire_clean(k) if isinstance(k, str) else k): _wire_clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_wire_clean(x) for x in v]
    return v


# The DAW reads every index and count as a 32-bit int, and its JSON parser WRAPS an integer written
# past int64 instead of refusing it: 2**64 + 1 arrives as 1, which no check on the other side can
# tell from a real 1 (2026-10-02 audit, F18). Python integers are unbounded, so the only place that
# can still see what the model actually wrote is here, before it is serialised.
_DAW_INT_MAX = 2 ** 31 - 1


def implausible_integer(args, where: str = "") -> str | None:
    """The reason a DAW command's arguments carry an integer the DAW cannot hold, else None."""
    if isinstance(args, bool):
        return None
    if isinstance(args, int):
        if abs(args) > _DAW_INT_MAX:
            return f"{where or 'a number'} {args} is not a plausible value"
        return None
    if isinstance(args, dict):
        for k, v in args.items():
            if (why := implausible_integer(v, str(k))) is not None:
                return why
    elif isinstance(args, (list, tuple)):
        for v in args:
            if (why := implausible_integer(v, where or "a number in a list")) is not None:
                return why
    return None


def wire_line(obj) -> bytes:
    """One NDJSON line the DAW's parser is guaranteed to accept."""
    try:
        text = json.dumps(obj, allow_nan=False)
    except ValueError:              # a non-finite float somewhere in the payload
        text = None
    if text is None or _UNSAFE_WIRE.search(text):
        text = json.dumps(_wire_clean(obj), allow_nan=False)
    return (text + "\n").encode()


class Hub:
    """Multi-project workspace (docs/10-daw.md Inc 2): the resident-project registry —
    ONE brain process, one ``Brain`` per open project root, created on demand. Any
    request may carry ``params.project`` (a project root dir) to route to that
    project's ``Brain``; without it, the connection's default brain (the boot
    project, or the one the connection named in ``hello``)."""

    def __init__(self, boot_dir: Path, token: str | None = None):
        self.brains: dict[str, Brain] = {}
        # The per-spawn secret every connection must present first (_serve_client). Minted
        # here and never optional: a Hub without one would be the open port again.
        self.token = token or secrets.token_urlsafe(32)
        self.boot = self.get(boot_dir)

    @staticmethod
    def _key(project_dir) -> str:
        return str(Path(project_dir).expanduser().resolve())

    def get(self, project_dir) -> "Brain":
        key = self._key(project_dir)
        brain = self.brains.get(key)
        if brain is None:
            brain = self.brains[key] = Brain(Path(key))
        return brain

    def any_run_active(self) -> bool:
        # The pass gate is app-global, not per-project: billing is shared and the
        # DAW drives one project at a time, so two projects never run passes at once.
        return any(b.run_active for b in self.brains.values())


# Agent-turn ids, monotonic across every Brain in the process (see begin_agent_turn).
# Starts at 1 — 0 means "unstamped" on the wire, which the DAW treats as a legacy
# caller and does not group.
_AGENT_TURNS = itertools.count(1)

# How many agent gestures stay undoable. Undo is LIFO and the DAW's own undo
# history is bounded too, so the oldest marks are unreachable — this only stops
# the two stacks growing one entry per gesture forever.
_MAX_UNDO_DEPTH = 200

# Run ids and daw_request ids are unique per PROCESS, not per connection (2026-09-10 audit
# P2, 02·F9 / 03·F11). Both restarted at 1 on every connection, so after a reconnect a late
# reply from the old connection's confirm box resolved the new turn's request #N with the
# wrong command's result, and a run id could name two different runs.
_RUN_IDS = itertools.count(1)
_DAW_REQ_IDS = itertools.count(1)

#: How long a confirm-gated daw_command waits for the user's decision. The 30 s reply bound
#: is for the DAW, not for a person reading "This is not covered by undo" — the agent was
#: told "did not respond" at 30 s, and the command then ran when the user clicked Do it.
CONFIRM_WAIT_S = 600.0
#: How long any other daw_command waits for the DAW's reply.
DAW_REPLY_TIMEOUT_S = 30.0
# A command that RENDERS in the app answers when the render is done, and a render of a long
# track through its plug-ins is not done in thirty seconds. At 30 s the brain gave up, told the
# app to drop a request that was half-way through, and told the agent it "may already have run"
# (it had: the bounce landed a moment later, under a reply that said it failed).
RENDER_REPLY_TIMEOUT_S = 600.0
# (`sound.render` renders on an engine of its own and needs nothing realized first - an
# instrument track's devices are the app's own, never a proposal's - but it too answers when
# its files are written.)
RENDERS_IN_THE_APP = frozenset({*RENDER_COMMANDS, "clip.consolidate", "sound.render",
                                "pack.add", "pack.multisample"})
#: Render commands that print ONE track and never pass the master (its devices and fader are
#: applied when the print plays): what they need from the brain first is the chains, not the
#: master limiter's makeup.
ONE_TRACK_RENDERS = frozenset({"track.bounce", "track.freeze"})


def _bounded(args: dict, key: str, lo: float, hi: float, default: float) -> float:
    """A numeric command arg, refused — never clamped — when it is not a finite number in
    [lo, hi], so the agent sees its own mistake (DawCommands.h's rule, on the brain side)."""
    raw = args.get(key, default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number, got {raw!r}") from None
    if not math.isfinite(value) or not lo <= value <= hi:
        raise ValueError(f"{key} {raw!r} is outside {lo:g}..{hi:g} dB")
    return value


def reply_wait_s(command: str, needs_confirm: bool, timeout: float | None = None) -> float:
    """How long the brain waits for the app's answer to one command: a person deciding
    (`CONFIRM_WAIT_S`), the caller's own bound, a render (`RENDER_REPLY_TIMEOUT_S`), or the
    ordinary thirty seconds."""
    if needs_confirm:
        return CONFIRM_WAIT_S
    if timeout is not None:
        return timeout
    return RENDER_REPLY_TIMEOUT_S if command in RENDERS_IN_THE_APP else DAW_REPLY_TIMEOUT_S


def _needs_confirm(edit: dict | None, command: str) -> bool:
    """Whether the live catalog marks `command` confirm-gated (DawCommands.h Spec::confirm)."""
    for entry in (edit or {}).get("commands") or (edit or {}).get("daw_commands") or []:
        if isinstance(entry, dict) and entry.get("key") == command:
            return bool(entry.get("needs_confirm") or entry.get("confirm"))
    return False


class Brain:
    """The session the DAW is attached to. State lives in the project's session.json
    (the same file the agent pass mutates), so get_chains always reflects the latest."""

    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        # Awareness backbone (docs/10-daw.md): the latest live Tracktion edit snapshot the DAW
        # pushes (fire-and-forget `edit_state` notification) — tracks/clips/tempo/transport +
        # the command catalog. Folded into every agent pass's context so the brain isn't blind
        # to the arrangement. None until the DAW pushes the first snapshot.
        self.latest_edit: dict | None = None
        # Operate path: the invoke hook of the DAW connection attached to THIS project.
        # None until a client connects + provides it. Ownership (Inc 2 hijack fix):
        # daw_invoke used to be installed by EVERY new connection (last-connection-wins),
        # so any second socket client — a smoke, a stray tool — silently stole the
        # agent's command channel from the real DAW. Now the connection that declared
        # itself the DAW (hello {role:"daw"}) owns it; an undeclared client may only
        # seed an unclaimed slot (legacy console harnesses) and never displaces an owner.
        self.daw_invoke = None
        self._pending_turn_delta: dict | None = None   # set by chat(), settled by coedit()
        self._operate_owner: object | None = None
        self._operate_declared = False
        # Every live claim, oldest first: (token, invoke, declared). When the owner leaves,
        # the newest still-connected claimant takes the path back (2026-09-10 audit P3,
        # 02·F14) — a second declared client that came and went used to leave the real DAW
        # with "no DAW attached" for the rest of the session.
        self._operate_claims: list[tuple[object, object, bool]] = []
        # Pass mutual-exclusion is SHARED, not per-connection: the session/billing it drives is
        # shared, so a reconnect (a second connection arriving mid-run) must not be able to start
        # a concurrent pass that double-mutates session.json and double-bills. Claimed by run,
        # auto_mix, AND advise (every agent pass). Checked+set on the single event-loop thread,
        # so a plain flag is race-free (no await between check and set).
        self.run_active = False
        # run_id -> cancel flag, on the BRAIN (2026-09-10 audit P2, 02·F8). It lived on the
        # connection, so a turn orphaned by a dropped socket could be stopped by nobody:
        # the reconnected DAW's Cancel answered `cancelled: 0` while the turn kept applying.
        self.cancels: dict[int, threading.Event] = {}
        # What the in-flight pass is doing, in memory, for the mid-run status turn
        # (2026-08-05). That turn must not render or read session.json, so the run
        # publishes here as it goes. Worker-thread dict/list updates are GIL-safe.
        self.live_run: dict | None = None
        # One-shot session mutations (set_send, routing, undo, …) each do their own
        # session.json load→mutate→persist on a worker thread; serialize them so two
        # connections can't interleave read-modify-writes (last-writer-wins clobber).
        # RE-ENTRANT (the pre-merge review of 2026-10-23): a thread that holds it can ask the
        # DAW for something (`Brain.undo` asks for `edit.undo`), and the DAW's reply is followed
        # on that same thread (`follow_renames` - the renames an undo of a rename hands back).
        # With a plain Lock that thread waited for itself, and the undo never answered.
        self.mutate_lock = threading.RLock()
        # The co-edit batches this process has taken (pre-merge review 2026-10-06). The DAW
        # sends a batch it got no answer for again, under the same id - in its next flush, or
        # inside a command's reply - and whoever takes the id first applies it. A lock of its
        # own: a reply is read on a thread that may already hold `mutate_lock`.
        self._batches_taken: set[str] = set()
        self._batches_lock = threading.Lock()
        # Undo boundaries: the applied_seq high-water-mark captured BEFORE each run/auto-mix that
        # applied something — so the DAW's Undo reverts a whole RUN (many proposals), not one. A
        # stack, so repeated Undo walks back run-by-run; empty → undo everything applied.
        self._run_marks: list[int] = []
        # Per-edit render cache (limiter makeup / pre-LUFS / readout from the last full master
        # render) — lets audio-neutral routing ops (group_stems) answer instantly via
        # chains_with_readout's light path instead of re-rendering the whole song.
        self.render_cache: dict = {}
        # Agent-turn id (2026-07-30). Every daw_request of ONE agent gesture carries the same
        # id, so the DAW opens a single Tracktion undo transaction per gesture instead of one
        # per command — a turn that moves three clips costs the user one ⌘Z, not three.
        # Bumped at the start of each run / advise / auto_mix.
        self.agent_turn = 0
        # What this turn asked of the app, one row a command (timing record v3): how long the
        # brain waited, how long the app says it took, and how much of the wait was a person
        # at an approval box. `chat` empties it at a turn's start and `_record_timing` files it.
        self.command_log: list[dict] = []
        # Taste learning under shared control (2026-07-30): the proposals the LAST
        # gesture applied, awaiting a verdict. An undo banks them as `undone`; the
        # next gesture starting with them still in place banks them as `kept`.
        self._pending_taste: list[dict] = []
        # The previous gesture's moves while the NEXT gesture runs (2026-09-10 audit P2,
        # 05·F11). They used to be banked `kept` the moment the next turn began — before
        # the user's "no, put that back" could take them out — so a rejected move was
        # learned as a weak yes. Now they are banked when that gesture ENDS, and only
        # the ones still applied; an A/B answer drops them (its own verdict is recorded).
        self._taste_carry: list[dict] = []
        # Co-edit URLs a gesture's replies carried but the brain failed to mirror. The DAW
        # swapped them out of its queue into the reply, so they go BACK to it at the end of
        # the gesture (`coedits_unmirrored`) to re-queue, instead of vanishing (02·F13).
        self._unmirrored: list[str] = []
        # Two undo authorities (2026-07-30). A chat turn can move the mix (proposals,
        # reverted by undo_run) AND edit the arrangement (daw_commands, reverted by the
        # Tracktion UndoManager). Undoing only the first leaves the clip edits standing —
        # a half-revert. This counts the DAW commands each gesture issued, pushed onto a
        # stack parallel to _run_marks, so Undo can revert BOTH. One `edit.undo` suffices
        # per gesture because the turn transaction already groups them (DawCommands.h).
        self._turn_daw_ops = 0
        self._run_daw_ops: list[int] = []
        # Each mark's identity (2026-09-10 audit P1-4), parallel to _run_marks: the feed's
        # "undo run" names the gesture it sits under, and Undo refuses one that is no longer
        # on top rather than popping whatever is newest. Random, so an id from before a brain
        # restart can never match a new gesture. `last_gesture` = what the latest
        # _close_gesture pushed (None if it pushed nothing), for the result the DAW gets.
        self._run_gestures: list[str] = []
        self.last_gesture: str | None = None
        # WHICH TURN each mark is (P2 batch T, reader S4·R14), parallel to _run_marks: the
        # agent-turn id its commands were stamped with. ⌥⌘Z sends it with `edit.undo`, so the
        # DAW takes back THAT turn's timeline sets or says the turn left none - it used to undo
        # whichever "Agent:" set was on top, which was an older turn's whenever the newest one
        # had only set a level, pressed play or been refused.
        self._run_turns: list[int] = []
        # ...and the LEVELS each turn's own commands set (James, 2026-10-05: "⌥⌘Z takes back
        # the agent's turn, levels included"): the co-edit that puts each control back to what
        # the session held before the turn first touched it (daw_bridge.level_inverses).
        self._turn_levels: dict = {}
        self._run_levels: list[list[str]] = []

    def begin_agent_turn(self) -> int:
        """Start a new agent gesture. The DAW closes the previous turn's undo transaction
        and opens a fresh one the next time it sees a request with this id.

        The counter is process-global, not per-Brain: in the multi-project workspace each
        Brain would otherwise start at 0, so switching projects could re-issue an id the
        DAW had already seen — and an unchanged id means "same gesture", which would
        silently fold two turns into one undo transaction.

        This is also where the weak-positive taste signal is taken up: the user is asking
        for the next thing, and the previous gesture is still standing (an undo would have
        cleared `_pending_taste`). It is BANKED when this gesture closes, for the moves
        still applied then — this turn may be the one that takes them back.
        """
        self._taste_carry.extend(self._pending_taste)
        self._pending_taste = []
        self._turn_daw_ops = 0
        self._turn_levels = {}
        self._ensure_session()
        # What was applied when this gesture began (S4·R22): a gesture that takes a proposal
        # BACK is a mix change the highest sequence number cannot show.
        try:
            self._applied_at_start = applied_seqs(self.project_dir)
        except Exception:  # noqa: BLE001 — no readable session: nothing to compare with
            self._applied_at_start = None
        # ...and every level as the turn finds it (pre-merge review 2026-10-06): a fader this
        # turn's own PROPOSAL moves before its command does goes back to this, not to the
        # proposal's value (see `replied` below).
        try:
            self._levels_at_start = level_snapshot(self.project_dir)
            self._seq_at_start = max_applied_seq(self.project_dir)
        except Exception:  # noqa: BLE001
            self._levels_at_start, self._seq_at_start = {}, None
        self.agent_turn = next(_AGENT_TURNS)
        return self.agent_turn

    def _ensure_session(self) -> None:
        """A gesture works on session.json. A folder of stems that has none yet (a brain
        driven directly on a fresh folder: the evals, a console leg) gets it here, ONCE.

        This used to happen by accident, at every turn: `level_snapshot`, a read, opened the
        session with the constructor that writes, so session.json was rewritten and synced
        as each gesture began - and was first made there for a folder that had none. The
        read is read-only now (Phase 55); the first save is this line. A folder whose writes
        are held (an open that is not a project yet) stays unwritten, as before."""
        from mix_engine.session import Session
        try:
            if not (self.project_dir / "session.json").is_file():
                Session(self.project_dir)
        except Exception:  # noqa: BLE001 - an unreadable folder is the gesture's own refusal
            pass

    def _channels_this_turn_proposed_on(self) -> set:
        """The targets of the proposals this turn has applied so far (a stem id, a bare bus id,
        `master`): a level on one of them was the turn's own before any command touched it."""
        floor = getattr(self, "_seq_at_start", None)
        if floor is None:
            return set()
        try:
            return {str(a.get("stem_id")) for record in applied_proposals_since(self.project_dir, floor)
                    for a in (record.get("actions") or []) if a.get("stem_id")}
        except Exception:  # noqa: BLE001
            return set()

    def counting_invoke(self, *, mirror: bool = False):
        """`daw_invoke` wrapped to count this gesture's DAW commands, so Undo knows
        whether the Tracktion side also needs reverting. None when no DAW is attached.

        ``mirror=True`` is for every consumer that does NOT mirror replies itself — the
        run planner, an approved Ask plan, arrangement.build (2026-09-10 audit P1-5). The
        DAW SWAPS its whole co-edit queue into every command reply (App.cpp), the user's
        queued knob and fader moves included, so a caller that ignores `coedits` loses
        them for good: the fader moves live while session.json never hears of it. The
        same wrapper performs routing brain-side (perform_routing), because a return.add
        the DAW receives mid-gesture only queues behind it. chat_turn keeps the plain
        wrapper: it mirrors and measures per command itself, and a second mirror would
        replay an add_device twice."""
        inner = self.daw_invoke
        if inner is None:
            return None

        def replied(command: str, args: dict | None = None, *a, **kw):
            res = inner(command, args, *a, **kw)
            # The levels this command set, before anything mirrors them (S4·R14): what the
            # session holds NOW is what ⌥⌘Z puts back. The first touch of a control in the
            # turn is the one that counts - a second set would read the turn's own value.
            if isinstance(res, dict) and res.get("coedits_own"):
                try:
                    for control, back in level_inverses(
                            self.project_dir, list(res.pop("coedits_own") or []),
                            at_start=getattr(self, "_levels_at_start", None),
                            moved_by_turn=self._channels_this_turn_proposed_on()):
                        self._turn_levels.setdefault(control, back)
                except Exception as exc:  # noqa: BLE001 - never let the bookkeeping fail a command
                    print(f"[daw_server] level undo not recorded after {command} "
                          f"({type(exc).__name__}: {exc})", file=sys.stderr)
            if mirror and isinstance(res, dict) and res.get("coedits"):
                urls = list(res.pop("coedits") or [])
                try:
                    m = self._mirror_coedits(urls)
                except Exception as exc:  # noqa: BLE001 — handed back to the DAW (02·F13)
                    print(f"[daw_server] co-edit mirror failed after {command} "
                          f"({type(exc).__name__}: {exc})", file=sys.stderr)
                    self._unmirrored.extend(urls)
                else:
                    if isinstance(m, dict) and m.get("failed"):
                        res["coedit_refusals"] = list(m.get("errors") or [])
            return res

        routing = self.routing_ops() if mirror else None
        # The mix the DAW holds as this gesture starts: the chains the last reply carried. A
        # render the agent asks for later in the gesture must not run on it once the gesture has
        # moved the mix on (2026-09-22, _realize_mix_before). None = unknown (a routing push).
        held = {"fp": self._mix_fp()}
        self._held = held      # `_register_stem` reads it: a track the app made is not news to it

        def invoke(command: str, args: dict | None = None, *a, **kw):
            self._turn_daw_ops += 1
            note = (self._realize_mix_before(command, inner, held)
                    if command in RENDER_COMMANDS else None)
            if routing is not None:
                routed = perform_routing(command, dict(args or {}), routing, replied)
                if routed is not None:
                    held["fp"] = None   # a routing push: see below
                    held.pop("tracks", None)
                    return routed
            res = replied(command, args, *a, **kw)
            if (isinstance(res, dict) and res.get("queued") is True
                    and command == "agent.transpose"):
                # ...and when it arrives (Brain.transpose) it belongs to THIS turn's gesture.
                self._transpose_queued = True
            if isinstance(res, dict) and res.get("queued") is True:
                # The DAW queued it behind this turn (queueRouting / queueAfterTurn): it edits
                # nothing INSIDE the turn's undo step. Counted, a turn that only asked for
                # agent.undo closed a gesture of its own - and the queued undo then found "the
                # newest gesture" was the request for it (readers C4·R6 = C2·R2); and a turn that
                # only queued a transpose left a mark whose ⌥⌘Z asked the DAW to take back an
                # "Agent:" step that was some older turn's.
                self._turn_daw_ops -= 1
            elif (isinstance(res, dict) and res.get("ok") is False
                    and not res.get("may_have_run")):
                # A REFUSED command changed nothing (S4·R14): counted, a turn whose only
                # command was refused left a mark, and its ⌥⌘Z was a press that undid nothing -
                # or, before the turn was named, an older turn's timeline edit.
                # Not daw_invoke's own TIMEOUT (pre-merge review 2026-10-06): "it may already
                # have run" - a long bounce or freeze did - and a turn that left no mark had
                # the next ⌥⌘Z pop the turn BEFORE it.
                self._turn_daw_ops -= 1
            if command == "routing.apply":
                # A routing op pushed chains (chat_turn's perform_routing sends through here),
                # and they carried the gesture's mix as it was THEN. What the DAW holds is no
                # longer the start fingerprint, so the next render pushes whatever it finds: an
                # undo_proposal can walk the session back to the start mix without a push.
                held["fp"] = None
            if note and isinstance(res, dict):
                res["note"] = note   # the render ran without the gesture's moves: the agent says so
            return res
        return invoke

    def _mix_fp(self) -> str:
        """mix_fingerprint, or "" when the session cannot be read — never an exception on the
        command path (a gesture with no readable session has no mix to realize)."""
        try:
            return mix_fingerprint(self.project_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[daw_server] mix fingerprint unavailable ({type(exc).__name__}: {exc})",
                  file=sys.stderr)
            return ""

    def _take_coedits(self, raw) -> bool:
        """Drain the DAW's co-edit queue mid-gesture (`coedits.take`) and mirror it. True when it
        held any. The queue holds the user's own knob and fader moves since the agent's last
        command reply — every reply swaps it out, and this is the same swap without a command. A
        DAW older than this brain refuses the key as unknown: there is nothing to take then, and
        the push goes ahead as it would have."""
        res = raw("coedits.take", {})
        urls = list(res.get("coedits") or []) if isinstance(res, dict) else []
        if not urls:
            return False
        try:
            m = self._mirror_coedits(urls)
        except Exception as exc:  # noqa: BLE001 — handed back to the DAW (02·F13)
            print(f"[daw_server] co-edit mirror failed before a render "
                  f"({type(exc).__name__}: {exc})", file=sys.stderr)
            self._unmirrored.extend(urls)
        else:
            if isinstance(m, dict) and m.get("failed"):
                print(f"[daw_server] {m['failed']} co-edit(s) refused before a render: "
                      f"{m.get('errors')}", file=sys.stderr)
        return True

    def take_user_moves(self) -> None:
        """Take the DAW's queued co-edits NOW and mirror them: the user's own fader, pan and
        knob moves since the agent's last command reply (full audit 2026-10-02, reader S4·R3).

        Called before anything that reads session.json FOR the DAW or computes against it: the
        chains a chat turn or an approved plan ends with, a mid-turn routing push, a proposal,
        an undo. Without it the brain's chains were laid over every node with those moves still
        in the queue — the fader the user pulled while the model thought jumped back, and
        session.json took the new value 1.5 s later — and a proposal was computed against a
        session that had not heard them. The run path and the user's own routing actions flush
        DAW-side (they apply on the RunThread); these arrive on the message thread and cannot.
        Never raises, never counts as a DAW command of the gesture."""
        raw = self.daw_invoke
        if raw is None:
            return
        try:
            self._take_coedits(raw)
        except Exception as exc:  # noqa: BLE001 — the idle flush still has them
            print(f"[daw_server] could not take the DAW's queued moves "
                  f"({type(exc).__name__}: {exc})", file=sys.stderr)

    def _realize_mix_before(self, command: str, raw, held: dict) -> str | None:
        """Hand the DAW this gesture's mix before a command that renders it (RENDER_COMMANDS,
        2026-09-22). None when done or not needed, else a note for the command's reply.

        A gesture's proposals reach the native graph only when its reply carries `chains`, and a
        render reads that graph the moment it starts — so a propose then an export in one turn
        bounced the pre-turn mix (the lofi showcase's master). Order matters three times here:
        the user's queued moves are taken and mirrored FIRST, so the chain pushed below carries
        them instead of resetting those controls in the live graph (05·F15's fader jumping back,
        mid-turn); then, if the mix differs from what the DAW holds, ONE render resolves the
        chains, because the limiter's makeup is its target minus the pre-limiter loudness of THIS
        mix; and `routing.apply`, flagged `sync` (no tool row, the turn's live line not sealed),
        realizes them before it replies. A move made while that render ran has just been
        overwritten, so it is taken, mirrored and pushed once more."""
        try:
            self._take_coedits(raw)
            for _ in range(2):
                fp = self._mix_fp()
                if fp == held["fp"]:
                    return None
                if command in ONE_TRACK_RENDERS and fp == held.get("tracks"):
                    # All that moved since the app held the mix is a track the app made itself
                    # (a print registered in this turn): it holds every chain there is, and a
                    # one-track render never passes the master, whose makeup is the one thing
                    # behind. Eight prints were eight renders of the whole mix (Phase 55).
                    return None
                payload = chains_with_readout(self.project_dir, cache=self.render_cache)
                realized = raw("routing.apply", {**payload, "command": "", "status": "",
                                                 "sync": command})
                if isinstance(realized, dict) and realized.get("ok") is False:
                    raise RuntimeError(str(realized.get("error") or "refused"))
                held["fp"] = fp
                if not self._take_coedits(raw):
                    return None
            return None
        except Exception as exc:  # noqa: BLE001 — the render still runs; the agent must say so
            print(f"[daw_server] could not hand the DAW this turn's mix before {command} "
                  f"({type(exc).__name__}: {exc})", file=sys.stderr)
            return (f"this turn's mix moves may be missing from the {command}: the DAW could not "
                    f"take them first ({exc})")

    def _settle_kept_taste(self) -> None:
        """Bank the previous gesture's moves as `kept` — the ones this gesture left
        applied. A move this gesture took back (undo_proposal, "no, put that back") is not
        a yes; the undo recorded its own verdict."""
        carry, self._taste_carry = self._taste_carry, []
        if not carry:
            return
        try:
            still = {p.get("proposal_id")
                     for p in applied_proposals_since(self.project_dir, -1)}
            kept = [p for p in carry if p.get("proposal_id") in still]
            if kept:
                record_taste(self.project_dir, "kept", kept)
        except Exception as exc:  # noqa: BLE001 — learning must never break a turn
            print(f"[daw_server] taste (kept) skipped: {exc}", file=sys.stderr)

    def _arm_kept_taste(self, floor_seq: int) -> None:
        """Remember what this gesture applied, so the NEXT one can bank it as kept."""
        try:
            self._pending_taste = applied_proposals_since(self.project_dir, floor_seq)
        except Exception as exc:  # noqa: BLE001
            print(f"[daw_server] taste arm skipped: {exc}", file=sys.stderr)
            self._pending_taste = []

    def _close_gesture(self, before: int) -> bool:
        """Record this gesture's undo boundary if it CHANGED anything — whether by
        applying a proposal or by running daw_commands. Returns whether the MIX moved
        (a proposal landed), which is what the callers re-resolve chains on.

        The boundary used to be recorded only when `max_applied_seq` moved, i.e. only
        for proposal-bearing gestures. But the agent's ordinary path in the DAW is
        `daw_command` (delete these bars, split that clip), which applies no proposal —
        so those gestures pushed nothing, and the NEXT Undo popped an empty stack and
        reverted the user's entire mix history to flat while the clip edits stayed.
        Counting DAW ops too keeps the two stacks aligned and makes one Undo mean one
        gesture.

        A turn that changed NOTHING records nothing (2026-08-05 review). Chat is the
        DAW's front door, so most turns are questions — "what's the LUFS?" is a
        gesture in no sense the user would recognise, and a no-op mark makes the next
        Undo a no-op too: they press it, a full render happens, and nothing moves.
        Ask three questions and Undo appears broken three times.
        """
        self._settle_kept_taste()   # the PREVIOUS gesture's verdict, now that this one is done
        moved = max_applied_seq(self.project_dir) > before
        # ...or a proposal that was applied when the gesture began no longer is (full audit
        # 2026-10-02, reader S4·R22). "Moved" meant "the highest applied sequence number went
        # UP", and an undo makes it go down: the agent's `undo_proposal` ("no, put that back",
        # the A answer of an A/B, an approved plan whose step is an undo) reverted session.json
        # and the reply carried no chains, so the live mix kept the move the agent had just said
        # was undone. Such a turn has nothing new for Undo to take back, so it records no mark —
        # but the mix did move, and the callers re-resolve the chains on that.
        took_back = False
        start, self._applied_at_start = getattr(self, "_applied_at_start", None), None
        if start is not None:
            try:
                took_back = bool(start - applied_seqs(self.project_dir))
            except Exception:  # noqa: BLE001
                took_back = False
        self.last_gesture = None
        # A transpose this turn queued behind itself joins whatever mark the turn leaves (or
        # none: it is then a gesture of its own). Cleared by the next turn - see transpose().
        queued, self._transpose_queued = getattr(self, "_transpose_queued", False), False
        self._transpose_joins = None
        if not moved and self._turn_daw_ops == 0:
            return took_back
        if len(self._run_gestures) != len(self._run_marks):
            self._run_gestures = [""] * len(self._run_marks)   # unnamed marks: never matchable
        self.last_gesture = secrets.token_hex(6)
        if len(self._run_turns) != len(self._run_marks):
            self._run_turns = [0] * len(self._run_marks)       # marks from before turns were kept
        if len(self._run_levels) != len(self._run_marks):
            self._run_levels = [[] for _ in self._run_marks]
        self._run_marks.append(before)
        self._run_gestures.append(self.last_gesture)
        self._run_turns.append(int(self.agent_turn or 0))
        # ...each with the value its control holds as the turn ENDS: at the undo, one that no
        # longer holds it was moved since - by the user - and is left alone (pre-merge review
        # 2026-10-06; the proposal half replays the user's later moves the same way).
        try:
            ends = level_values(self.project_dir, list(self._turn_levels))
        except Exception:  # noqa: BLE001
            ends = {}
        self._run_levels.append([(control, back, ends.get(control))
                                 for control, back in self._turn_levels.items()])
        self._turn_levels = {}
        if queued:
            self._transpose_joins = self.last_gesture
        self._arm_kept_taste(before)
        self._run_daw_ops.append(self._turn_daw_ops)
        # Bounded: undo is LIFO, so the oldest marks are the ones nobody can
        # still reach. Without this both stacks grow one entry per gesture for
        # the life of the process.
        if len(self._run_marks) > _MAX_UNDO_DEPTH:
            del self._run_marks[:-_MAX_UNDO_DEPTH]
            del self._run_daw_ops[:-_MAX_UNDO_DEPTH]
            del self._run_gestures[:-_MAX_UNDO_DEPTH]
            del self._run_turns[:-_MAX_UNDO_DEPTH]
            del self._run_levels[:-_MAX_UNDO_DEPTH]
        return moved or took_back

    # A turn in flight, on disk (2026-09-10 audit P2, 03·F18). If the brain dies mid-turn, the
    # moves that landed are in session.json but were never reported — the feed said only "the
    # engineer process died", and a retry re-sent the brief over them. The next process hands
    # them to the DAW (recover_inflight) so they show before anything is asked again.
    INFLIGHT_FILE = "inflight_turn.json"

    def _mark_inflight(self, kind: str, before: int) -> None:
        try:
            runs = self.project_dir / "runs"
            runs.mkdir(parents=True, exist_ok=True)
            write_atomic(runs / self.INFLIGHT_FILE, json.dumps(   # tmp + replace (02·F15)
                {"kind": kind, "before_seq": int(before), "ts": time.time()}))
        except OSError:
            pass   # a bonus: a turn never fails for want of this note

    def _clear_inflight(self) -> None:
        try:
            (self.project_dir / "runs" / self.INFLIGHT_FILE).unlink(missing_ok=True)
        except OSError:
            pass

    def recover_inflight(self) -> dict:
        """The moves a turn landed before its brain died, once; {ok, moves: [] } when none.

        The recovered turn is ONE undo (reader S3·R7). The feed shows these moves under "⌥⌘Z
        takes them back", and this process holds no mark for a turn another process ran: with
        an empty stack `undo` falls back to "the newest proposal", so one press took back one
        move of the several and the next walked on into earlier work. The note's `before_seq`
        IS that turn's boundary - it is pushed as a mark, exactly as the turn's own close would
        have (all five stacks: no DAW ops, because the app's turn gate forgot that turn with
        the process; no levels, for the same reason), and its gesture id rides the reply."""
        # A reconnect INSIDE this process, while the turn the note belongs to still runs
        # (part b): that turn is not dead, and reading its note here cleared it - a real death
        # a moment later then had nothing to recover.
        if self.run_active:
            return {"ok": True, "moves": []}
        path = self.project_dir / "runs" / self.INFLIGHT_FILE
        try:
            note = json.loads(path.read_text())
            before = int(note.get("before_seq", 0))
        except (OSError, ValueError, TypeError, AttributeError):
            return {"ok": True, "moves": []}
        self._clear_inflight()
        try:
            moves = moves_since(self.project_dir, before)
        except Exception:  # noqa: BLE001 — an unreadable session has nothing to show
            moves = []
        out: dict = {"ok": True, "moves": moves, "kind": str(note.get("kind", ""))}
        if moves:
            if len(self._run_gestures) != len(self._run_marks):
                self._run_gestures = [""] * len(self._run_marks)
            if len(self._run_turns) != len(self._run_marks):
                self._run_turns = [0] * len(self._run_marks)
            if len(self._run_levels) != len(self._run_marks):
                self._run_levels = [[] for _ in self._run_marks]
            if len(self._run_daw_ops) != len(self._run_marks):
                self._run_daw_ops = [0] * len(self._run_marks)
            gesture = secrets.token_hex(6)
            self._run_marks.append(before)
            self._run_gestures.append(gesture)
            self._run_turns.append(0)
            self._run_levels.append([])
            self._run_daw_ops.append(0)
            out["gesture"] = gesture
        return out

    def _gesture_out(self) -> dict:
        """The id of the gesture this call just closed, for the result the DAW gets: its
        feed's "undo run" sends it back so Undo takes back THAT gesture (P1-4) — plus any
        co-edits its replies carried that the brain could not mirror, for the DAW to
        re-queue (02·F13)."""
        out: dict = {"gesture": self.last_gesture} if self.last_gesture else {}
        if self._unmirrored:
            out["coedits_unmirrored"] = list(dict.fromkeys(self._unmirrored))
            self._unmirrored = []
        return out

    def _raise_if_landed(self, exc: BaseException, before: int) -> None:
        """A gesture that fails after moves landed must SAY so. Auto Mix got this rule on
        2026-08-09 (§4 audit); a run, a chat turn and an approved plan did not, so the feed
        read "run failed" over a mix that had already changed (2026-09-10 audit P1-2)."""
        if not isinstance(exc, Exception) or "already applied" in _error_text(exc):
            return   # the chat layer already said so (chat.TurnCutOff)
        try:
            n = len(moves_since(self.project_dir, before))
        except Exception:  # noqa: BLE001 — the original error is the one to surface
            return
        if n:
            raise NoRetry(     # a retry would send the brief over the moves that landed
                f"{_error_text(exc)} — {n} move(s) had already applied before the "
                f"failure; one ⌥⌘Z (undo agent move) reverts them") from exc

    def claim_operate(self, token: object, invoke, *, declared: bool) -> bool:
        """Attach a connection's invoke hook as this project's operate path. A declared
        DAW (hello {role:"daw"}) always takes over — a reconnect must replace its dead
        predecessor; an undeclared client only seeds an unclaimed slot."""
        if not declared and self._operate_owner is not None \
                and self._operate_owner is not token:
            return False   # an undeclared client never displaces an existing owner
        if declared:
            self._operate_declared = True
        elif self._operate_owner is not token:
            self._operate_declared = False   # fresh undeclared seed of the empty slot
        self._operate_claims = [c for c in self._operate_claims if c[0] is not token]
        self._operate_claims.append((token, invoke, bool(declared) or self._operate_declared))
        self._operate_owner = token
        self.daw_invoke = invoke
        return True

    def release_operate(self, token: object) -> None:
        self._operate_claims = [c for c in self._operate_claims if c[0] is not token]
        if self._operate_owner is not token:
            return
        # Hand the path back to the newest claimant still connected — a declared DAW first.
        back = (next((c for c in reversed(self._operate_claims) if c[2]), None)
                or (self._operate_claims[-1] if self._operate_claims else None))
        if back is None:
            self._operate_owner, self._operate_declared, self.daw_invoke = None, False, None
        else:
            self._operate_owner, self.daw_invoke, self._operate_declared = back

    def session_problem(self) -> str:
        """Why session.json can't be read, or "" when it can or doesn't exist yet (2026-09-10
        audit P3, 02·F16). stems() reads an unreadable file as NO stems, so Run and Auto Mix
        answered "there's no audio for me to mix yet" over a project full of audio."""
        try:
            json.loads((self.project_dir / "session.json").read_text())
        except FileNotFoundError:
            return ""
        except Exception as exc:  # noqa: BLE001 — whatever it is, it is the answer
            return f"{type(exc).__name__}: {exc}"
        return ""

    def stems(self) -> list[str]:
        try:
            state = json.loads((self.project_dir / "session.json").read_text())
            return list(state.get("stems", {}).keys())
        except FileNotFoundError:
            if self.unwritten:      # not a project yet: the names are the folder's (S1·R25 d)
                try:
                    from mix_engine.session import Session
                    return list(Session(self.project_dir, read_only=True)._state["stems"])
                except Exception:  # noqa: BLE001 - a folder discovery refuses: no names
                    return []
            return []
        except Exception as exc:  # noqa: BLE001 — said, not swallowed (02·F16)
            print(f"[daw_server] session.json unreadable ({type(exc).__name__}: {exc})",
                  file=sys.stderr, flush=True)
            return []

    def open_edit(self, target_dir: str) -> dict:
        """Multi-edit (docs/10-daw.md Inc 2): re-point the brain at the switched-to edit's session
        dir, so the agent mixes THAT edit's tracks. Each edit has its own session.json (+ stems);
        edit 0 = the project root, new edits = <project>/.edits/<id>/. The DAW sends this on
        edit.open. A fresh edit dir starts empty (no stems) and grows as tracks are imported."""
        p = Path(target_dir)
        p.mkdir(parents=True, exist_ok=True)
        if self.unwritten:        # an edit switch is the DAW's, made after it became a project
            from mix_engine.session import hold_writes
            hold_writes(self.project_dir, False)
            self.unwritten = False
        self.project_dir = p
        self.latest_edit = None   # the switched-to edit pushes its own snapshot
        LATEST_EDIT.pop(str(Path(self.project_dir).resolve()), None)
        self.render_cache.clear()   # the cache is per-edit — never serve another edit's readout
        from mix_engine import dsp
        dsp.RENDER_CACHE.clear()    # ...and the edit left behind does not hold memory here (Phase 54)
        # The undo boundaries and the pending taste signal are `applied_seq` values in
        # the OUTGOING edit's session.json — meaningless here. Left in place, the first
        # Undo after a switch reverted this edit's proposals against the other edit's
        # floor, and the outgoing edit's moves were later banked as "kept" into this
        # edit's calibration store.
        self._run_marks.clear()
        self._run_daw_ops.clear()
        self._run_turns.clear()
        self._run_levels.clear()
        self._pending_taste = []
        self._taste_carry = []
        # A pending turn delta measures the OUTGOING edit; settled against this one it
        # would report the other edit's loudness as the agent's effect (02·F12).
        self._pending_turn_delta = None
        # a "Discard" answered for THIS edit while another was open (S1·R3); an edit's folder is
        # the DAW's from its first moment (an edit switch only ever comes from the DAW)
        self.open_session(daw=True)
        return {"ok": True, "project_dir": str(p), "stems": self.stems()}

    #: The folder is open in the DAW and is not a project yet: nothing is written into it
    #: (reader S1·R25 d). See open_session / become_project.
    unwritten = False

    def take_open_notices(self) -> list:
        """What the last open has to tell the user, once (the hello carries it)."""
        notes, self._open_notices = list(getattr(self, "_open_notices", [])), []
        return notes

    def become_project(self) -> None:
        """The folder becomes a project: the hold on its writes ends and the open it was owed
        runs - the session minted from the folder's stems, the first checkpoint. Called for the
        first request that would write (the dispatcher), and by a hello that no longer says
        `unwritten`."""
        from mix_engine.session import hold_writes
        if not self.unwritten:
            return
        self.unwritten = False
        hold_writes(self.project_dir, False)
        self.render_cache.clear()
        self.open_session(daw=True, stems_from_folder=True)

    def open_session(self, daw: bool = False, stems_from_folder: bool = False,
                     unwritten: bool = False) -> None:
        """What an OPEN owes the session before anything reads it (the DAW's hello, an edit
        switch): honour a declined save, then make the first checkpoint if there is none.

        ``unwritten`` (reader S1·R25 d; James, 2026-10-06): the DAW has opened a folder that
        is not a project and has written nothing into it - and neither does the brain. Every
        Session of the folder is read-only from here (mix_engine.session.hold_writes): the
        folder's wavs are still read, so the boot's topology and readout are what they would
        be, and nothing lands on disk until `become_project`. A folder that already holds a
        session is a project whatever the hello says.

        ``daw`` (reader S1·R19): the DAW holds this project, so from here on a stem is a track
        the DAW registered - a wav that merely turns up at the top of the folder is not one
        (session.stems_by_registration). The open itself still reads the folder in the two
        cases where the folder IS the track list: a project with no session yet, and one the
        DAW has just built its lanes from (``stems_from_folder``: it found no edit file)."""
        from mix_engine.session import (SESSION_FILE, Session, hold_writes,
                                        stems_by_registration)

        if unwritten and not (self.project_dir / SESSION_FILE).exists():
            self.unwritten = True
            hold_writes(self.project_dir, True)
            stems_by_registration(self.project_dir, False)   # the folder IS the track list
            return
        if self.unwritten:                       # a later hello: the DAW has made it a project
            self.unwritten = False
            hold_writes(self.project_dir, False)
            self.render_cache.clear()
        # An unreadable session.json is set aside before anything reads it (reader S1·R25 b).
        if (said := set_aside_unreadable_session(self.project_dir)):
            self._open_notices = [*getattr(self, "_open_notices", []), said]
            self.render_cache.clear()
        self.honour_discard()
        first = not (self.project_dir / SESSION_FILE).exists()
        stems_by_registration(self.project_dir, daw and not (first or stems_from_folder))
        ensure_saved_checkpoint(self.project_dir, mint=True)
        if daw:
            if stems_from_folder and not first:
                try:
                    Session(self.project_dir)        # the lanes the DAW made have their stems
                except Exception as exc:  # noqa: BLE001 — a folder discovery refuses (master.wav)
                    print(f"[daw_server] the folder's stems were not read "
                          f"({type(exc).__name__}: {exc})", file=sys.stderr, flush=True)
            stems_by_registration(self.project_dir, True)
        self.apply_owed()

    def apply_owed(self):
        """The moves the DAW could not hand over before it went (daw_bridge.apply_owed_coedits):
        applied once, after a declined save was honoured and the first checkpoint was made.

        Not while a run of this project is still winding up (pre-merge review 2026-10-06): the
        turn a quit cancelled is the very reason the moves were parked, and its tool - still
        holding its own Session - persisted over them a moment later, with the file already
        gone. It stays for the next quiet open; `coedit` is refused in that window for the
        same reason."""
        if getattr(self, "run_active", False):
            return None
        try:
            done = apply_owed_coedits(self.project_dir)
        except Exception as exc:  # noqa: BLE001 - an open must not fail on a queue from last time
            print(f"[daw_server] owed co-edits not applied ({type(exc).__name__}: {exc})",
                  file=sys.stderr, flush=True)
            return None
        if done is not None and done.get("applied"):
            self.render_cache.clear()
        return done

    def honour_discard(self):
        """The DAW's "Don't Save" / "Discard", if it left one (daw_bridge.honour_discard). The
        ledger under every undo mark and taste signal has just been replaced, so they go: a mark
        is an `applied_seq` of the session that was declined."""
        done = honour_discard(self.project_dir)
        if done is None:
            return None
        self.render_cache.clear()
        self._run_marks.clear()
        self._run_daw_ops.clear()
        self._run_gestures.clear()
        self._run_turns.clear()
        self._run_levels.clear()
        self._pending_taste = []
        self._taste_carry = []
        self._pending_turn_delta = None
        print(f"[daw_server] declined session put back to its last save: {done}",
              file=sys.stderr, flush=True)
        return done

    def _state_now(self) -> dict:
        """session.json as it stands - or, in a folder that is not a project yet (S1·R25 d),
        what a read-only Session finds there. `get_chains` and `get_strip` are requests the
        app may make of an unwritten folder (`_LEAVES_FOLDER_ALONE`), and the file they read
        does not exist until the folder is written: they answered FileNotFoundError to the
        re-read the app makes after its queued moves (pre-merge review, 2026-10-07)."""
        path = self.project_dir / "session.json"
        if self.unwritten and not path.exists():
            from mix_engine.session import Session
            return Session(self.project_dir, read_only=True)._state
        return json.loads(path.read_text())

    def get_chains(self) -> dict:
        state = self._state_now()
        makeup_db, pre_lufs_i, rate = _master_resolve_inputs(self.project_dir)
        return resolve_chains(state, makeup_db, pre_lufs_i, native_returns=True,  # DAW: real return tracks
                              sample_rate=rate)   # ...at the project's rate (reader E4·R6)

    def get_strip(self) -> dict:
        """The strip map alone (daw_bridge.strip_state) — the DAW re-reads it, with get_chains,
        after flushing the user's queued moves ahead of a reply, so a pan the user set while a
        run was working is not put back by the reply that was computed before it."""
        return strip_state(self._state_now())

    def run(self, brief: str, use_llm: bool = False, should_cancel=None,
            on_move=None, on_phase=None, llm: dict | None = None) -> dict:
        # use_llm=False: deterministic RuleAgent (reproducible, no key). use_llm=True: the
        # real ClaudeAgent, honouring MIX_AGENT_LLM_BACKEND (subscription/api) — the brain
        # inherits the DAW's env. Mutates session.json with the agent's mix. The run info
        # ({stop_reason, cost_usd, budget_usd}) rides back with the chains so the DAW shows
        # spend + why it stopped (incl. "cancelled" / "spend limit reached").
        # Arrangement is an EDIT operation, not a mix target. Route it before the ordinary
        # brief compiler so "create a lo-fi song arrangement" works with the deterministic
        # agent too. The native arrangement.build command applies every clip + label in one
        # Tracktion undo transaction and replies with a conversational summary.
        self.begin_agent_turn()   # this gesture's commands = one DAW undo transaction
        if is_arrangement_brief(brief):
            arrangement_floor = max_applied_seq(self.project_dir)
            try:
                try:
                    session_state = json.loads((self.project_dir / "session.json").read_text())
                except Exception:
                    session_state = {}
                # counting_invoke, not the raw hook: building a song is a gesture like
                # any other, and its commands have to be counted or Undo cannot revert
                # the clips it laid down. mirror: the reply carries the co-edit queue.
                try:
                    arranged = apply_arrangement(brief, self.latest_edit,
                                                 self.counting_invoke(mirror=True),
                                                 session_state=session_state)
                finally:
                    self._close_gesture(arrangement_floor)
                return {"stop_reason": "arranged", "passes": 0, "cost_usd": 0.0,
                        "budget_usd": None, "chat": arranged["message"],
                        "arrangement": arranged["result"], **self._gesture_out()}
            except ArrangementError as exc:
                return {"stop_reason": "arrangement failed", "passes": 0,
                        "cost_usd": 0.0, "budget_usd": None,
                        "chat": f"I couldn't build that arrangement: {exc}."}

        # Nothing to mix → say so for free. Without this, Run on a fresh blank project
        # entered the full pass pipeline — a billed LLM pass proposing nothing actionable
        # (2026-08-09 §1 audit). Arrangement briefs were routed above; this guards only
        # the ordinary mix path.
        if not self.stems():
            if (why := self.session_problem()):   # unreadable is not empty (02·F16)
                return {"stop_reason": "session unreadable", "passes": 0, "cost_usd": 0.0,
                        "budget_usd": None, "chat": _unreadable_session_chat(why)}
            return {"stop_reason": "nothing to mix", "passes": 0, "cost_usd": 0.0,
                    "budget_usd": None,
                    "chat": ("there's no audio for me to mix yet — drop a stem or a "
                             "sample onto the arrangement first, then run me again")}

        before = max_applied_seq(self.project_dir)   # run boundary for a whole-run undo
        self._mark_inflight("run", before)
        try:
            # mirror: the planner's daw_command replies carry the DAW's co-edit queue (P1-5)
            info = run_agent_pass(self.project_dir, brief, use_llm=use_llm,
                                  should_cancel=should_cancel,
                                  edit_state=self.latest_edit,
                                  daw_invoke=self.counting_invoke(mirror=True),
                                  on_move=on_move, on_phase=on_phase, llm=llm)
        except BaseException as exc:   # noqa: BLE001 — a non-mix message chats, never crashes
            if brief_not_understood(exc):
                # not a mixing brief (a greeting/question) — reply conversationally, no mix. The
                # DAW shows `chat` in the Engineer panel; no chains changed so none are re-applied.
                spent: dict = {}
                chat = chat_reply(self.project_dir, brief, use_llm=use_llm,
                                  edit_state=self.latest_edit, cache=self.render_cache,
                                  llm=llm, spent=spent)
                return {"stop_reason": "chat", "passes": 0,
                        "cost_usd": chat_fallback_cost(exc, spent), "budget_usd": None,
                        "chat": chat}
            self._raise_if_landed(exc, before)
            raise
        finally:
            # try/FINALLY, like Auto Mix (2026-09-10 audit P1-2): a pass that raised after
            # moves landed pushed no mark, so the next ⌥⌘Z also took back the gesture BEFORE
            # this one — a whole Auto Mix under "undo run", with no redo.
            self._close_gesture(before)
            self._clear_inflight()
        # The Engineer panel's flat ledger (round 5): the run's applied actions with per-move
        # rationales, + the pre-run loudness (last render's readout) for the LUFS delta line.
        lufs_before = (self.render_cache or {}).get("readout", {}).get("lufs_i")
        return {**chains_with_readout(self.project_dir, cache=self.render_cache), **info,
                "moves": moves_since(self.project_dir, before), "lufs_before": lufs_before,
                **self._gesture_out()}

    def chat(self, message: str, mode: str = "agent", use_llm: bool = False,
             on_event=None, should_cancel=None, session_id: str = "",
             llm: dict | None = None, images: list | None = None) -> dict:
        """The DAW's single front door for anything the user types (2026-07-30).

        With an LLM available this is the resident engineer: routing happens by TOOL
        CHOICE inside one persistent, context-carrying conversation, so "the snare has
        no crack", "a bit less", "why did you do that" and "delete the last 8 bars" all
        work — and a request about one track stays about that track.

        Without an LLM there is no conversational agent to run, so it falls back to the
        deterministic cascade this method replaced (arrangement detector -> metric
        compiler -> canned reply). That path is byte-identical to what shipped before,
        which is what keeps the rule-agent console tests meaningful.

        `images` (Phase 64) are the paths of the pictures the user attached. They are read
        and checked here, before the turn opens anything, through the one door
        (`chat_images`): a refusal is a sentence and nothing of the turn has happened."""
        pictures = chat_images.read_all(images or [], self.project_dir) if images else []
        if pictures and not use_llm:
            raise chat_images.PictureRefused(
                "Pictures need the engineer on your Anthropic key; without one the built-in "
                "engineer reads words only. Send the message without them.")
        self.begin_agent_turn()   # this whole turn = one DAW undo transaction
        self.command_log = []
        if (message.strip() in PREFERENCE_OPTIONS
                and (self.project_dir / "runs" / PENDING_PREFERENCE_FILE).is_file()):
            # An A/B click: its own verdict (approved / rejected) is recorded by the turn,
            # so the same move is not ALSO banked as passively kept (05·F11 / 02·F11). The
            # ONE move the A/B is about - the turn's other moves were left standing like any
            # others, and are kept (reader E6·R6: the whole carry used to be dropped).
            asked = pending_preference_proposal(self.project_dir)
            self._taste_carry = [p for p in self._taste_carry
                                 if p.get("proposal_id") != asked]
        if not use_llm:
            info = self.run(message, use_llm=False, should_cancel=should_cancel)
            # Spread, don't nest: run() already returns chains/readout at the top
            # level, which is the shape applyChainsAndReadout reads on the DAW side.
            return {**info, "reply": info.get("chat") or "", "fallback": "rule"}
        before = max_applied_seq(self.project_dir)
        # A GETTER, not the dict: the DAW pushes a fresh snapshot before replying to
        # each agent command, so later rounds of this turn must read `latest_edit`
        # as it is THEN — not as it was when the turn started. Passing the value
        # meant round 2 addressed clips by an index into the pre-command list.
        moved = False
        self._mark_inflight("chat", before)
        try:
            # The plain counting wrapper: chat_turn mirrors each reply's co-edits itself.
            result = chat_turn(self.project_dir, message, mode=mode, use_llm=True,
                               edit_state=lambda: self.latest_edit,
                               daw_invoke=self.counting_invoke(),
                               cache=self.render_cache, on_event=on_event,
                               should_cancel=should_cancel, session_id=session_id,
                               llm=llm,   # the composer's picker, per request (P1-21)
                               routing_ops=self.routing_ops(),
                               mirror_coedits=self._mirror_coedits,
                               take_user_moves=self.take_user_moves,
                               register_stem=self._register_stem,
                               images=pictures)
        except Exception as exc:
            self._record_timing(getattr(exc, "timing", None), 0.0)   # a failed turn too
            self._raise_if_landed(exc, before)
            raise
        finally:
            # A boundary is recorded for any turn that CHANGED something (see
            # _close_gesture): the agent's ordinary move in the DAW is a daw_command, which
            # applies no proposal, and those turns used to record nothing at all. FINALLY
            # (2026-09-10 audit P1-2): a turn that raised after landing moves — the idle
            # guard, a usage limit — recorded nothing either.
            moved = self._close_gesture(before)
            self._clear_inflight()
        tail_began = time.monotonic()
        own_unmirrored = result.pop("coedits_unmirrored", None) or []
        result.update(self._gesture_out())
        if own_unmirrored:   # the turn's own mirror failures join the brain's
            result["coedits_unmirrored"] = list(dict.fromkeys(
                own_unmirrored + list(result.get("coedits_unmirrored") or [])))
        pending = result.pop("turn_delta_pending", None)
        if moved:
            self.take_user_moves()      # so the chains below carry them, not reset them (S4·R3)
            self.render_cache.clear()   # the mix changed — don't serve stale numbers
            result.update(chains_with_readout(self.project_dir, cache=self.render_cache))
        # The measured delta settles HERE when the brain's mirror is current — a turn of
        # only proposals (just re-rendered above), or one whose daw_command co-edits were
        # mirrored mid-turn (the readout is then already in the cache, no re-render).
        # Otherwise the DAW's co-edit flush settles it (coedit()).
        if pending and (not pending.get("commands") or pending.get("settle") == "chat_done"):
            try:
                readout = result.get("readout")
                if not isinstance(readout, dict):
                    readout = session_readout(self.project_dir, self.render_cache)
                if (delta := settle_turn_delta(self.project_dir, pending, readout)):
                    result["turn_delta"] = delta
            except Exception as exc:  # noqa: BLE001 — a delta is a bonus
                print(f"[daw_server] turn delta skipped ({type(exc).__name__}: {exc})",
                      file=sys.stderr)
            pending = None
        self._pending_turn_delta = pending
        self._record_timing(result.get("timing"), time.monotonic() - tail_began)
        warm_observations(self.project_dir)   # what this turn made stale, ready for the next
        return result

    def _record_timing(self, timing, tail_s: float) -> None:
        """One line in brain.log and one in `runs/turn_timing.jsonl` for a finished chat
        turn (Phase 54). The tail is what this method's caller did after the reply."""
        if not isinstance(timing, dict):
            return
        try:
            from orchestrator import turn_timing
            timing.setdefault("marks_s", {})["tail"] = round(tail_s, 2)
            timing.setdefault("mark_counts", {})["tail"] = 1
            asked, self.command_log = self.command_log, []
            turn_timing.merge_commands(timing, asked)
            timing["wall_s"] = round(float(timing.get("wall_s") or 0.0) + tail_s, 1)
            print(turn_timing.line(str(timing.get("turn") or ""), timing), file=sys.stderr)
            turn_timing.record(self.project_dir, timing)
        except Exception as exc:  # noqa: BLE001 — a readout, never a blocker
            print(f"[daw_server] turn timing skipped ({type(exc).__name__}: {exc})",
                  file=sys.stderr)

    def apply_plan(self, plan: dict, should_cancel=None, llm: dict | None = None) -> dict:
        """Execute an ask-mode plan verbatim — the steps the user actually approved,
        not a fresh derivation of the same request."""
        self.begin_agent_turn()
        before = max_applied_seq(self.project_dir)
        # A CALLABLE, like chat's (2026-08-09 §4 audit): a plain value froze the edit at
        # do-it-click time, so a clip the user moved while an earlier plan step (a run,
        # an automix) executed was invisible to a later step's fingerprint re-check —
        # defeating exactly the staleness protection the fingerprints exist for.
        moved = False
        self._mark_inflight("apply_plan", before)
        try:
            # mirror (2026-09-10 audit P1-5): a "do it" step's reply carries the DAW's co-edit
            # queue, and a routing step has to be performed brain-side — the plan used to
            # report "ok" over a fader session.json never saw and a return it never created.
            result = apply_plan(self.project_dir, plan, use_llm=True,
                                edit_state=lambda: self.latest_edit,
                                daw_invoke=self.counting_invoke(mirror=True),
                                should_cancel=should_cancel,
                                register_stem=self._register_stem,
                                mirror_coedits=self._mirror_coedits,
                                take_user_moves=self.take_user_moves,
                                llm=llm)
        except Exception as exc:
            self._raise_if_landed(exc, before)
            raise
        finally:
            moved = self._close_gesture(before)   # FINALLY, like every gesture (P1-2)
            self._clear_inflight()
        result.update(self._gesture_out())
        if moved:
            self.take_user_moves()      # as at the end of a chat turn (S4·R3)
            self.render_cache.clear()
            result.update(chains_with_readout(self.project_dir, cache=self.render_cache))
        return result

    def advise(self, brief: str, use_llm: bool = False, llm: dict | None = None,
               should_cancel=None) -> dict:
        # Ask mode (advice-only): run the agent on a throwaway COPY + report what it WOULD do,
        # without touching the real mix. Returns {ok, advice, count}; the DAW shows it in chat.
        self.begin_agent_turn()
        if is_arrangement_brief(brief):
            try:
                try:
                    session_state = json.loads((self.project_dir / "session.json").read_text())
                except Exception:
                    session_state = {}
                plan = build_arrangement_plan(brief, self.latest_edit, session_state)
                return {"ok": True, "advice": arrangement_summary(plan, preview=True),
                        "count": len(plan["sections"])}
            except ArrangementError as exc:
                return {"ok": True,
                        "advice": f"I can't preview that arrangement yet: {exc}.",
                        "count": 0}
        return advise(self.project_dir, brief, use_llm=use_llm, edit_state=self.latest_edit,
                      llm=llm, should_cancel=should_cancel)

    def undo(self, gesture: str = "") -> dict:
        # Revert the whole most-recent GESTURE (every proposal it applied), so one Undo undoes
        # the agent's mix — not a single proposal of many. The mark stack gives gesture-granular,
        # repeatable undo. Returns the reverted chains for the graph.
        #
        # With no recorded boundary this used to pass floor = -1, which undo_run reads as
        # "revert ALL applied proposals" — a full revert to flat. That is never a safe
        # answer to a button press: the boundary stack lives only in memory, so a brain
        # restart (or, until 2026-08-05, any daw_command-only gesture) silently turned Undo
        # into "throw away every mix decision in this project". Nothing else in the product
        # can destroy that much work in one click, and it is not recoverable.
        #
        # Fall back to the ordinary meaning of undo instead: take back the single most
        # recent applied proposal. It is bounded, repeatable (press again for the one
        # before), and it can never revert more than the user can see.
        # WHICH gesture (2026-09-10 audit P1-4). The feed's "undo run" names the gesture it
        # sits under; one that is no longer on top is refused. Popping the newest regardless
        # meant a click under an older run — or after a daw_command turn, which lands no
        # ledger to retire the old verb — took back the wrong gesture. ⌥⌘Z names none.
        aligned = len(self._run_gestures) == len(self._run_marks)
        top = self._run_gestures[-1] if (aligned and self._run_gestures) else ""
        if gesture and gesture != top:
            return {"ok": False, "gesture": gesture,
                    "refused": ("that exchange is no longer the newest agent gesture, so undo "
                                "run left it alone — ⌥⌘Z steps back one gesture at a time")}
        had_mark = bool(self._run_marks)
        turn, levels = 0, []
        if self._run_marks:
            if len(self._run_turns) == len(self._run_marks):
                turn = self._run_turns.pop()
            if len(self._run_levels) == len(self._run_marks):
                levels = self._run_levels.pop()
            floor = self._run_marks.pop()
            if aligned and self._run_gestures:
                self._run_gestures.pop()
        else:
            latest = max_applied_seq(self.project_dir)
            floor = max(latest - 1, 0)
        # Taste (2026-07-30): capture what is about to be reverted BEFORE reverting it,
        # and record it as `undone` — the strongest signal shared control still gives us.
        # The user heard the move and took it back; that outranks any passive "kept".
        # It also clears the pending kept-signal, so a reverted gesture is never later
        # banked as a positive.
        try:
            reverted = applied_proposals_since(self.project_dir, floor)
        except Exception:  # noqa: BLE001 — never let learning block an undo
            reverted = []
        self._pending_taste = []
        self._taste_carry = []
        undo_run(self.project_dir, floor)
        if reverted:
            try:
                record_taste(self.project_dir, "undone", reverted)
            except Exception as exc:  # noqa: BLE001
                print(f"[daw_server] taste (undone) skipped: {exc}", file=sys.stderr)
        # The OTHER undo authority (2026-07-30). A gesture can move the mix (proposals,
        # reverted above) AND edit the arrangement (daw_commands, owned by the Tracktion
        # UndoManager). Reverting only the mix left the clip edits standing — a
        # half-revert the user reads as "undo is broken". One edit.undo is enough because
        # the turn transaction groups the whole gesture (docs/10-daw.md).
        # The turn's own LEVELS (S4·R14): a fader, a pan, a width, a mute, a send - back to what
        # the session held before the turn first touched each. After undo_run, so a proposal's
        # whole-target restore does not write over them; the chains and the strip rows of this
        # reply then move the DAW's controls back.
        levels_back, levels_left = 0, 0
        if levels:
            try:
                now = level_values(self.project_dir, [control for control, _b, _e in levels])
                back = []
                for control, url, at_end in levels:
                    held = now.get(control)
                    moved = (at_end is not None and held is not None
                             and (held != at_end if isinstance(at_end, bool) or isinstance(held, bool)
                                  else abs(float(held) - float(at_end)) > 0.05))
                    if moved:
                        levels_left += 1       # the user's since: theirs
                    else:
                        back.append(url)
                if back:
                    levels_back = int(apply_coedits(self.project_dir, back).get("applied", 0))
            except Exception as exc:  # noqa: BLE001 - the rest of the undo still stands
                print(f"[daw_server] the turn's levels were not put back "
                      f"({type(exc).__name__}: {exc})", file=sys.stderr)
        daw_ops = self._run_daw_ops.pop() if self._run_daw_ops else 0
        daw_undone = 0
        timeline_note = ""
        if daw_ops and self.daw_invoke is not None:
            # expect_agent: revert ONLY the turn's own "Agent: …" transaction. If the user already
            # took it back with a plain ⌘Z (or edited since), the DAW refuses instead of undoing
            # whatever sits on top of its stack (2026-09-02 audit). `turn` names WHICH turn
            # (S4·R14): the DAW undoes that turn's sets, answers `timeline: "none"` when the
            # turn left nothing there (it set a level, pressed play) - never another turn's.
            undo_args: dict = {"expect_agent": True}
            if turn:
                undo_args["turn"] = turn
            result = self.daw_invoke("edit.undo", undo_args)
            if isinstance(result, dict) and result.get("ok") is False:
                print(f"[daw_server] DAW undo refused: {result.get('error')}",
                      file=sys.stderr)
                timeline_note = str(result.get("error") or "")
            elif isinstance(result, dict) and result.get("timeline") == "none":
                daw_ops = 0          # nothing of this turn was on the timeline: nothing is owed
            else:
                daw_undone = daw_ops
                # The reply carries the co-edits the undo ITSELF queued (full audit 2026-10-02,
                # reader S4·R2). `edit.undo` reverts the native chain, the node's undo mirror
                # queues that revert as a co-edit inside the command, and every reply swaps the
                # queue out — but this read only `ok`. session.json kept the agent's value, and
                # the chains resolved below put it straight back over the node the undo had just
                # reverted: the comp snapped to −18 and back to −30 inside one reply, "undid the
                # run (0 moves)", and the Tracktion transaction was spent. Applied directly, not
                # through _mirror_coedits: this thread already holds mutate_lock.
                urls = list(result.get("coedits") or []) if isinstance(result, dict) else []
                if urls:
                    self.render_cache.clear()
                    try:
                        mirrored = apply_coedits(self.project_dir, urls)
                    except Exception as exc:  # noqa: BLE001 — handed back to the DAW (02·F13)
                        print(f"[daw_server] co-edit mirror failed after edit.undo "
                              f"({type(exc).__name__}: {exc})", file=sys.stderr)
                        self._unmirrored.extend(urls)
                    else:
                        if isinstance(mirrored, dict) and mirrored.get("failed"):
                            print(f"[daw_server] {mirrored['failed']} co-edit(s) refused after "
                                  f"edit.undo: {mirrored.get('errors')}", file=sys.stderr)
        elif daw_ops:
            # Be honest rather than silently half-reverting.
            print("[daw_server] gesture had DAW edits but no DAW is attached to undo "
                  "them", file=sys.stderr)
        out = chains_with_readout(self.project_dir, cache=self.render_cache)   # {chains, readout} from one render
        out["daw_ops_undone"] = daw_undone
        out["daw_ops_pending"] = daw_ops - daw_undone
        # Scope honesty (2026-08-09 §4 audit): with the mark stack exhausted (brain restart,
        # depth cap) this quietly degrades to single-proposal undo — the DAW's status must
        # say WHICH scope a press had, or the user can't tell one move from a whole run.
        out["undone"] = {"scope": "run" if had_mark else "single",
                         "proposals": len(reverted),
                         "levels": levels_back,
                         "gesture": top if had_mark else ""}   # the DAW retires that verb
        if levels_left:
            out["undone"]["levels_left"] = levels_left        # moved by the user since: kept
        if timeline_note:
            out["undone"]["timeline_left"] = timeline_note    # why the timeline half stayed
        return out

    def take_batch(self, batch: str) -> bool:
        """True ONCE per co-edit batch id: to the caller that is about to apply it. A batch a
        brain before this one applied is in session.json (`Session.note_coedit_batches`)."""
        batch = str(batch or "")
        if not batch:
            return True
        with self._batches_lock:
            if batch in self._batches_taken:
                return False
            self._batches_taken.add(batch)
            return batch not in coedit_batches_done(self.project_dir)

    def resolve_sent(self, entries) -> list[str]:
        """A reply's `coedits` as plain URLs: the groups the DAW had already sent are taken
        here or dropped (daw_bridge.resolve_sent), so every consumer of a reply mirrors
        strings, as before."""
        urls, _ = resolve_sent(entries, self.take_batch)
        return urls

    def coedit(self, urls: list, batch: str | None = None) -> dict:
        # Persist the DAW's queued device-rack edits to session.json (shared control), so the
        # agent's next pass starts from the user's manual tweaks. The DAW flushes the queue
        # right before a run/undo, on the same socket thread (no concurrent socket access).
        #
        # `batch` names this request. The DAW gives up on a flush after its wait and sends
        # the moves again - while this brain, busy behind a long render, applies the first
        # send when it comes free. The same id comes back with them: taken once.
        if batch and not self.take_batch(batch):
            return {"ok": True, "applied": 0, "of": 0, "failed": 0, "errors": [],
                    "already": True}
        urls, taken = resolve_sent(urls, self.take_batch)
        if urls:
            # device edits change the audio: drop the per-edit render cache so a later
            # light readout (⌘G grouping) re-renders instead of serving the pre-edit numbers
            self.render_cache.clear()
        result = apply_coedits(self.project_dir, urls,
                               batches=[*taken, *([str(batch)] if batch else [])])
        # No turn delta settles here any more (2026-09-10 audit P2, 02·F12): the next
        # co-edit batch is the USER's, whatever it holds, and a fader they pulled minutes
        # later was reported as "the measured effect of your last turn". Every turn now
        # settles at chat_done, where the brain is current (see Brain.chat).
        return dict(result) if isinstance(result, dict) else {"result": result}

    def auto_mix(self, brief: str = "", should_cancel=None, on_stage=None,
                 print_midi: bool = False) -> dict:
        # Auto Mix (Phase 16): the deterministic one-button bootstrap chain, applied +
        # persisted to session.json; returns the resolved chains for the DAW to apply to
        # the graph. Like run() but the automix policy (no LLM). Multi-render → async.
        # `should_cancel` reaches the bundle loop, so Stop lands on a bundle boundary
        # with everything applied so far still inside this gesture's undo.
        #
        # `print_midi` (the feed's "print to audio + auto mix", 2026-10-13): every sounding MIDI
        # part is printed to a new audio track first - by the app's own bounce, registered as a
        # stem in THIS gesture, its MIDI track muted - so one undo takes the prints back with
        # the mix. Without it, a project whose only parts are MIDI is declined with the offer.
        self.begin_agent_turn()
        parts = midi_parts(self.latest_edit)
        if parts and not print_midi and not mixable_stems(self.project_dir):
            if (why := self.session_problem()):
                return {"ok": False, "moves": [], "chat": _unreadable_session_chat(why)}
            return {"ok": False, "moves": [], "chat": midi_print_offer_said(parts),
                    "offer_print": [p["track"] for p in parts]}
        printing = bool(print_midi and parts and self.daw_invoke is not None)
        if not printing and not self.stems():   # the §1 run guard: nothing to mix = free decline
            if (why := self.session_problem()):   # unreadable is not empty (02·F16)
                return {"ok": False, "moves": [], "chat": _unreadable_session_chat(why)}
            return {"ok": False, "moves": [],
                    "chat": ("there's no audio to auto-mix yet — drop a stem or a sample "
                             "onto the arrangement first")}
        before = max_applied_seq(self.project_dir)   # run boundary for a whole-run undo
        # Crash recovery covers Auto Mix too (2026-09-11 pre-merge review, C8): run/chat/apply_plan
        # left a note, this gesture did not, so a brain death mid-Auto-Mix left the landed bundles
        # audible with no ledger rows and nothing for recover_inflight to report.
        self._mark_inflight("auto_mix", before)
        refused: list[str] = []   # bundles the guardrails refused (2026-09-10 audit P1-23)
        left_out: list[str] = []  # stems the cleanup's de-noise was NOT applied to (E4·R3)
        notes: list[str] = []     # why the pass did nothing at all (E5·R7: nothing was split)
        printed: list[str] = []   # the MIDI parts printed to audio for this mix
        print_problems: list[str] = []
        try:
            if printing:
                def on_part(name: str) -> None:
                    if on_stage is not None:
                        on_stage({"type": "automix_stage", "stage": "printing to audio",
                                  "detail": name})
                printed, print_problems = print_midi_parts(
                    self.project_dir, parts, self.counting_invoke(mirror=True),
                    register_stem=self._register_stem, mirror_coedits=self._mirror_coedits,
                    should_cancel=should_cancel, on_part=on_part)
                self.render_cache.clear()
                if not mixable_stems(self.project_dir):
                    return {"ok": False, "moves": [],
                            "chat": ("none of the MIDI parts could be printed, so there is "
                                     "still no audio to mix"
                                     + (f" — {print_problems[0]}" if print_problems else ""))}
            auto_mix_pass(self.project_dir, brief, should_cancel=should_cancel,
                          on_stage=on_stage, refused=refused, left_out=left_out, notes=notes)
        except Exception as exc:
            # A partial Auto Mix must SAY moves landed (2026-08-09 §4 audit): the bare
            # "auto mix failed" hid that the mix had already changed.
            self._raise_if_landed(exc, before)
            raise
        finally:
            # try/FINALLY (2026-08-09 §4 audit P0): a failure mid-Auto-Mix used to skip this,
            # so bundles already applied stayed OUTSIDE any run mark — the one-click Undo
            # promise silently degraded to single-proposal undo on exactly the partial-failure
            # case that needs it most. Like every other gesture (see _close_gesture): whatever
            # applied before the failure/cancel must still be one Undo away.
            self._close_gesture(before)
            self._clear_inflight()   # …and the crash note goes with it (C8, same as run/chat)
        if notes and not moves_since(self.project_dir, before) and not printed:
            # The pass declined in words: the DAW shows `chat` as the reason nothing ran.
            return {"ok": False, "moves": [], "chat": notes[0]}
        lufs_before = (self.render_cache or {}).get("readout", {}).get("lufs_i")
        return {**chains_with_readout(self.project_dir, cache=self.render_cache),   # one render
                "moves": moves_since(self.project_dir, before), "lufs_before": lufs_before,
                **({"printed": printed} if printed else {}),
                **({"print_problems": print_problems} if print_problems else {}),
                **({"refused": refused} if refused else {}),
                **({"notes": notes} if notes else {}),   # e.g. the muted tracks it left alone
                **({"left_out": ("the spectral de-noise was left out — it cannot run in the live mix "
                                 "or the export yet (" + ", ".join(left_out) + ")")}
                   if left_out else {}),
                **self._gesture_out()}

    def separate(self, path: str, stems: int = 4, should_cancel=None,
                 on_stage=None) -> dict:
        # Stem split (the Phase 29 boundary, DAW vertical 2026-08-13): htdemucs writes
        # the stem wavs into the project dir and returns their names; the DAW then
        # creates the tracks through its normal add-track import path (which owns
        # brain registration). Nothing touches session.json here, so no gesture/undo
        # bookkeeping — removing the created tracks is ordinary track deletion.
        return separate_track(self.project_dir, path, stems,
                              should_cancel=should_cancel, on_stage=on_stage)

    def transpose(self, tonic: str) -> dict:
        # Phase 32 key picker: a deterministic whole-song transpose proposal onto `tonic`,
        # applied (master-only, undoable). Returns {ok, message, chains} — chains are the
        # post-transpose graph (unchanged if it was a no-op / no confident key).
        #
        # A gesture of its own (full audit 2026-10-02, reader S4·R30). Every other door that
        # lands a proposal records a mark first; this one did not, so the transpose sat above
        # the PREVIOUS gesture's mark and one ⌥⌘Z took back the key pick and the agent's
        # move before it, with no redo. Bracketed like auto_mix; a pick that changes nothing
        # ("already in A minor") leaves no mark, because _close_gesture records none.
        #
        # The one exception is the agent's own `agent.transpose`: from inside a turn it is
        # queued behind it and arrives HERE, after that turn's mark. It belongs to the turn
        # that asked for it (one turn, one undo), so it lands above that mark and opens none.
        joins, self._transpose_joins = getattr(self, "_transpose_joins", None), None
        top = (self._run_gestures[-1]
               if self._run_gestures and len(self._run_gestures) == len(self._run_marks) else "")
        if joins and joins == top:
            result = transpose_to_target_key(self.project_dir, tonic, cache=self.render_cache)
            return {**result, **chains_with_readout(self.project_dir, cache=self.render_cache)}
        self.begin_agent_turn()
        before = max_applied_seq(self.project_dir)
        try:
            result = transpose_to_target_key(self.project_dir, tonic, cache=self.render_cache)
        finally:
            self._close_gesture(before)
        return {**result,
                **chains_with_readout(self.project_dir, cache=self.render_cache),   # + chains + readout, one render
                **self._gesture_out()}

    def match_clip_key(self, path: str, target_tonic=None, target_mode=None) -> dict:
        # Audio Clip View Match Key (Stage 2): detect ONE clip's key + compute the nearest signed
        # interval onto the target (the project key if the DAW passed one, else the detected song
        # key). No render / no chains of its own — and the song-key fallback reuses the per-edit
        # render cache (never a fresh master render just to look up the key).
        return match_clip_to_key(self.project_dir, path, target_tonic, target_mode,
                                 cache=self.render_cache)

    def tune_clip_notes(self, path: str, source_offset_s=0.0, length_s=None,
                        min_cents=15.0) -> dict:
        # clip.pitch_correct's ears (2026-08-08): measure every off note on the
        # clip's window; the DAW composes the fix from split/pitch/fade itself.
        # Read-only — no render, no chains, deliberately not a session mutator.
        return tune_clip_notes(self.project_dir, path,
                               source_offset_s=float(source_offset_s or 0.0),
                               length_s=float(length_s) if length_s else None,
                               min_cents=float(min_cents or 15.0))

    def extract_clip_groove(self, path: str, bpm=None, source_offset_s=0.0,
                            length_s=None, beat_offset_s=0.0) -> dict:
        # Extract Groove (2026-08-08): measure ONE audio clip's per-cell timing shape so the DAW
        # can turn it into a groove template. Read-only, no render and no chains of its own — and
        # the tempo fallback reuses the per-edit render cache rather than detecting afresh. A MIDI
        # clip never reaches here: its note beats ARE the groove, folded in C++ with no brain.
        # source_offset_s/length_s scope the read to the CLIP's window; beat_offset_s
        # is where the first downbeat falls inside it (both from the DAW's edit).
        return extract_clip_groove(self.project_dir, path,
                                   float(bpm) if bpm else None, cache=self.render_cache,
                                   source_offset_s=float(source_offset_s or 0.0),
                                   length_s=float(length_s) if length_s else None,
                                   beat_offset_s=float(beat_offset_s or 0.0))

    def _drop_stale_render_cache(self) -> None:
        """Called by an AUDIO-NEUTRAL change before it changes anything (a rename, an empty
        group, a blank return): such a change answers from the last render's numbers
        (`chains_with_readout (…, light=True)`) instead of rendering the song. That is right
        only while those numbers are of the mix as it stands. The cache used to be trusted
        because of call order - and an in-turn mix move, an undo or an import that did not
        clear it left the old limiter make-up, which the next ⌘G then pushed to the live
        graph (Phase 55 audit). It carries the fingerprint it was filled at now; another one
        empties it, and the light path falls back to a render."""
        from mix_engine.session import Session
        if not self.render_cache:
            return
        try:
            now = Session(self.project_dir, read_only=True).render_fingerprint()
        except Exception:  # noqa: BLE001 - no readable session: nothing to answer from
            now = None
        if now is None or self.render_cache.get("fp") != now:
            self.render_cache.clear()

    def follow_renames(self, renames) -> list[str]:
        """The track renames a DAW reply carries (`renames`, real use 2026-10-22): inside an
        agent turn the app cannot send its rename request - the turn holds the thread that
        sends it - so it hands the renames over with the reply, ahead of the moves that may use
        the new names. Followed here, before anything reads those moves. A rename of a track
        the brain holds no stem for (a MIDI track) changes nothing.

        Answers a sentence for each rename the brain REFUSED (the pre-merge review of
        2026-10-23): the app has handed the rename over and will not send it again, so the
        reply says it (`renames_not_followed`, which the model reads) - the drain this replaced
        said a refusal in the status line. Called on a thread that may hold `mutate_lock`
        (`Brain.undo`); the lock is re-entrant."""
        from orchestrator.daw_bridge import rename_stem
        refused: list[str] = []
        for entry in renames or []:
            if not isinstance(entry, dict):
                continue
            old, new = str(entry.get("old") or ""), str(entry.get("new") or "")
            if not old or not new or old == new:
                continue
            with self.mutate_lock:
                self._drop_stale_render_cache()
                try:
                    rename_stem(self.project_dir, old, new)
                except Exception as exc:  # noqa: BLE001 - said, and the reply goes on
                    print(f"[daw_server] the rename {old!r} -> {new!r} was not followed "
                          f"({type(exc).__name__}: {exc})", file=sys.stderr)
                    refused.append(f"the mix engine did not follow the rename of {old!r} to "
                                   f"{new!r} ({exc}): it still knows the track as {old!r}, so "
                                   f"moves that name {new!r} are refused there")
        return refused

    def _mirror_coedits(self, urls: list[str]) -> dict:
        """Mid-turn co-edit mirror (2026-09-03): the DAW's reply to an agent command
        carries the co-edits it queued; apply them under the mutate lock exactly as the
        idle flush would (coedit()), so the brain's render is current for the next round."""
        with self.mutate_lock:
            self.render_cache.clear()
            return apply_coedits(self.project_dir, list(urls or []))

    def _register_stem(self, path: str, stem_id: str | None, bus: str | None, *,
                       copy_from: str | None = None, keep: dict | None = None) -> dict:
        """In-turn registration of a track the agent just added (2026-09-04): the same
        `import_stem` the DAW's background `import` request runs, under the same mutate
        lock, but called from the chat turn itself — the DAW's own register cannot run
        while the turn holds it, and the agent's next tool call needs the stem to exist."""
        with self.mutate_lock:
            held = getattr(self, "_held", None)
            before = self._mix_fp() if held is not None else ""
            self.render_cache.clear()
            out = register_stem_copy(self.project_dir, path, stem_id, bus,
                                     copy_from=copy_from, keep=keep)
            if (held is not None and before and isinstance(out, dict) and out.get("ok")
                    and before in (held.get("fp"), held.get("tracks"))):
                # The app made this track and plays it already: its chains are still the
                # app's. What the new track changes for the app is the master's makeup, so
                # this is said apart from `fp` - an export still gets its push
                # (`_realize_mix_before`).
                held["tracks"] = self._mix_fp()
            return out

    def routing_ops(self) -> dict:
        """The routing keys the agent may invoke mid-turn, performed by the brain
        (daw_bridge.perform_routing). Each takes the daw_command args and returns
        the same {chains, readout, send_returns, …} payload the DAW's own routing
        requests get — realized through `routing.apply`. Under mutate_lock, like
        the socket handlers (the chat thread never holds it)."""
        def guard(fn):
            def run(args: dict) -> dict:
                # The payload this returns is pushed to the DAW (routing.apply) and laid over
                # every node: take the user's queued moves first, so it carries them (S4·R3).
                # Before the lock — the mirror takes it.
                self.take_user_moves()
                with self.mutate_lock:
                    return fn(args)
            return run

        def bare(v) -> str:
            v = str(v or "")
            return v[4:] if v.startswith("bus:") else v

        def stems_of(a: dict) -> list[str]:
            raw = a.get("stems") or []
            return [str(s) for s in (raw if isinstance(raw, list) else [])
                    if str(s) != "master" and not str(s).startswith(("bus:", "return:"))]

        def send_set(a: dict) -> dict:
            # A source the session holds nothing for - an instrument track, a lane whose audio
            # has not registered - keeps its sends in the DAW's edit: the DAW dials it (S1·R5 =
            # S4·R21). The level is bounded first, so the refusal is the same at either door.
            gain_db = _bounded(a, "gain_db", -60.0, 0.0, -60.0)
            source = str(a.get("source", ""))
            from mix_engine.session import Session
            if source and not Session(self.project_dir, read_only=True).holds_send_source(source):
                raise NotBrainRouting()
            return self.set_send(source, str(a.get("return", "")), gain_db)

        def duck_set_key(a: dict) -> dict:
            # A track the session holds no stem for - an instrument (MIDI) track - keeps its
            # devices in the DAW's edit, so the DAW adds and keys its Duck itself, like a
            # stemless send (the showcase, 2026-10-03: "duck the pad from the kick" came back
            # `unknown stem 'Pad'` and the agent faked the duck with a tempo-synced LFO).
            # `key_stem_id` is the propose action's spelling of the same argument: read here
            # too, because a key that is not read is "none", and "none" clears the trigger.
            stem = str(a.get("stem", ""))
            key = str(a.get("key") or a.get("key_stem_id") or "none")
            from mix_engine.session import Session
            held = Session(self.project_dir, read_only=True).stems
            if stem and stem not in held:
                raise NotBrainRouting()
            # ...and a TRIGGER the session holds no stem for (a MIDI kick): the DAW keys it from
            # its track and checks the track is there (a typo is refused there by name); the
            # brain is told by the DAW's co-edit (2026-10-17, the duck audit).
            if key.lower() not in ("", "none", "off", "master") and key not in held:
                raise NotBrainRouting()
            return self.set_duck(stem, key, a.get("threshold_db"), a.get("depth_db"))

        def a_track(name: str) -> str:
            """`name`, if it is a track of this project - else a refusal. The session routes
            an UNREGISTERED name as a pending member (an empty DAW track has no stem yet), so
            it cannot tell a new track from a typo; the agent's door holds the live edit and
            can (Phase 56, the command sweep: `route.stem {stem: "no_such_thing_zz"}` answered
            ok and left an empty group here and a group track in the app). With no picture of
            the edit - a stand-in app, a test - nothing is refused: there is nothing to ask."""
            from mix_engine.session import Session

            from orchestrator.edit_state import live_edit   # absolute: the DAW starts this file by path
            tracks = (live_edit(self.latest_edit) or {}).get("tracks")
            if not isinstance(tracks, list):
                return name
            named = {str(t.get("name") or "") for t in tracks if isinstance(t, dict)}
            stems = Session(self.project_dir, read_only=True).stems
            if name in named or (name in stems and not stems[name].get("excluded")):
                return name
            raise ValueError(f"there is no track named {name!r} - the tracks are: "
                             + ", ".join(sorted(n for n in named if n)[:24]))

        return {
            "return.add": guard(lambda a: self.add_return(str(a.get("kind") or "blank"))),
            "return.remove": guard(lambda a: self.remove_return(
                str(a.get("id") or a.get("return") or ""))),
            "return.set_prefader": guard(lambda a: self.set_return_prefader(
                str(a.get("id") or a.get("return") or ""), bool(a.get("pre_fader", False)))),
            # REFUSED, not clamped (2026-09-10 audit P2, 04·F6): Session clamps a send to
            # -60..0, so "+3 dB into the plate" answered ok over a stored 0.0, and a "NaN"
            # became 0 dB — the loudest send there is.
            "send.set": guard(send_set),
            "route.stem": guard(lambda a: self.route_stem(a_track(str(a.get("stem", ""))),
                                                          bare(a.get("bus", "")))),
            "group.create": guard(lambda a: self.group_stems([a_track(n) for n in stems_of(a)])),
            "bus.remove": guard(lambda a: self.remove_bus(bare(a.get("id") or a.get("bus")))),
            "bus.route": guard(lambda a: self.route_bus(bare(a.get("bus", "")),
                                                        bare(a.get("target") or "master"))),
            "duck.set_key": guard(duck_set_key),
        }

    def set_send(self, source: str, return_id: str, gain_db: float) -> dict:
        # Sends/returns shared control: the user dials a send level live. Apply it (Session.set_send,
        # the single source of truth) + re-resolve the graph so the wet is immediately AUDIBLE — the
        # send is baked into each track's chain (the send_return entry), so a re-resolve + re-apply is
        # required, unlike a fader which moves the node directly. Returns {chains, readout, send_returns}.
        from mix_engine.session import Session
        Session(self.project_dir).set_send(source, return_id, float(gain_db))
        return chains_with_readout(self.project_dir, cache=self.render_cache)

    def add_return(self, kind: str = "blank") -> dict:
        self._drop_stale_render_cache()   # before the change: see the method
        # Sends/returns shared control: the USER creates a BLANK return (Ableton-style — add devices
        # to it after). Session.add_return mutates session.json; re-resolve so the DAW realizes the new
        # return track + its strip. Returns {return_id, kind, chains, readout, send_returns}.
        from mix_engine.session import Session
        res = Session(self.project_dir).add_return(str(kind))
        # light=True, like group_stems (P2 batch T, reader S3·R6): a blank return with nothing
        # sent into it is audio-neutral, so the chains are re-resolved over the cached readout
        # instead of rendering the whole song for a return that adds no sound yet.
        return {**res, **chains_with_readout(self.project_dir, cache=self.render_cache,
                                             light=True)}

    def remove_return(self, return_id: str) -> dict:
        # Sends/returns shared control: the USER deletes a return (inverse of add_return). Session.remove_return
        # drops it + its sends; re-resolve so realizeReturns drops the orphaned return:<id> track + its strip.
        from mix_engine.session import Session
        res = Session(self.project_dir).remove_return(str(return_id))
        return {**res, **chains_with_readout(self.project_dir, cache=self.render_cache)}

    def set_return_prefader(self, return_id: str, pre_fader: bool) -> dict:
        # Sends/returns shared control: the per-return Pre/Post tap toggle. A routing change (it
        # moves the AuxSend insert point relative to the source's chain), so re-resolve + re-realize
        # so the new tap is audible. Returns {return, pre_fader, chains, readout, send_returns}.
        from mix_engine.session import Session
        res = Session(self.project_dir).set_return_prefader(str(return_id), bool(pre_fader))
        return {**res, **chains_with_readout(self.project_dir, cache=self.render_cache)}

    def rename_return(self, return_id: str, new_name: str) -> dict:
        self._drop_stale_render_cache()   # before the change: see the method
        # Sends/returns shared control (2026-09-07): the USER (or the agent) renames a return from
        # its strip / pinned head card. A return's id IS its displayed name, so this MOVES the id —
        # Session.rename_return carries every reference with it (each sender's send, the ledger,
        # journaled co-edits). light=True like rename_bus: a rename changes no audio, so the chains
        # re-resolve under their new keys over the cached readout instead of re-rendering the song.
        from mix_engine.session import Session
        res = Session(self.project_dir).rename_return(str(return_id), str(new_name))
        if not res.get("renamed"):
            return res
        return {**res, **chains_with_readout(self.project_dir, cache=self.render_cache, light=True)}

    def rename_stem(self, old: str, new: str) -> dict:
        self._drop_stale_render_cache()   # before the change: see the method
        # A track rename (⌘R / track.rename) follows into the stem store — the id IS the track
        # name everywhere the brain speaks about it (2026-09-02 agent-workspace audit: without
        # this a renamed audio track kept its old id, every later co-edit on it was refused, and
        # the agent's chains never landed on its node again). No render: a rename changes no
        # audio, so the light path re-resolves the chains (new keys) over the cached readout.
        from orchestrator.daw_bridge import rename_stem
        res = rename_stem(self.project_dir, old, new)
        if not res.get("renamed"):
            return res
        return {**res, **chains_with_readout(self.project_dir, cache=self.render_cache, light=True)}

    def route_stem(self, stem: str, bus: str) -> dict:
        # Routing shared control: the USER routes a stem to a group bus (created if absent) or the
        # master. Session.route_stem mutates session.json; re-resolve so the DAW realizes the bus track
        # + its strip. Returns {stem, bus, chains, readout, buses, send_returns}.
        from mix_engine.session import Session
        res = Session(self.project_dir).route_stem(stem, bus)
        return {**res, **chains_with_readout(self.project_dir, cache=self.render_cache)}

    def group_stems(self, stems: list[str]) -> dict:
        self._drop_stale_render_cache()   # before the change: see the method
        # Routing shared control (Ableton ⌘G): group several stems into ONE new bus in a single
        # action — the one-shot form of route_stem("new"). Session.group_stems mutates session.json;
        # re-resolve so the DAW realizes the bus track + its strip. Returns {bus_id, stems, chains,
        # readout, buses, send_returns} — the DAW selects bus:<bus_id> so the device dock targets it.
        from mix_engine.session import Session
        res = Session(self.project_dir).group_stems(list(stems))
        # light=True: grouping mints an EMPTY unity-gain bus — audio-neutral, so the chains are
        # re-resolved from session state with the cached render numbers instead of re-rendering
        # the whole song (that render was the multi-second ⌘G hang).
        return {**res, **chains_with_readout(self.project_dir, cache=self.render_cache,
                                             light=True)}

    def remove_bus(self, bus_id: str) -> dict:
        # Routing shared control: the USER dissolves a group bus (inverse of routing stems into it).
        # Session.remove_bus re-homes members to the master + drops the bus; re-resolve so realizeBuses
        # drops the orphaned bus:<id> track + its strip. Returns {bus_id, chains, readout, buses, ...}.
        from mix_engine.session import Session
        res = Session(self.project_dir).remove_bus(str(bus_id))
        return {**res, **chains_with_readout(self.project_dir, cache=self.render_cache)}

    def rename_bus(self, bus: str, new_name: str) -> dict:
        self._drop_stale_render_cache()   # before the change: see the method
        # Routing shared control (2026-09-06): the USER renames a group bus from its strip / head
        # card. The bus id IS its displayed name, so this MOVES the id — Session.rename_bus carries
        # every reference with it (members, a nested bus's output, the ledger, journaled co-edits).
        # light=True like group_stems: a rename changes no audio, so the chains re-resolve under
        # their new keys over the cached readout instead of re-rendering the song.
        from mix_engine.session import Session
        res = Session(self.project_dir).rename_bus(str(bus), str(new_name))
        if not res.get("renamed"):
            return res
        return {**res, **chains_with_readout(self.project_dir, cache=self.render_cache, light=True)}

    def route_bus(self, bus: str, target: str) -> dict:
        # Routing shared control (2026-07-03): a group's OWN output — route bus_id into another bus
        # (nested) or the master. Session.route_bus mutates session.json (cycle-checked there);
        # re-resolve so realizeBuses re-wires the bus track's own output. Returns
        # {bus, output, chains, readout, buses, bus_outputs, send_returns}.
        from mix_engine.session import Session
        res = Session(self.project_dir).route_bus(bus, target)
        return {**res, **chains_with_readout(self.project_dir, cache=self.render_cache)}

    def set_duck(self, stem: str, key: str, threshold_db=None, depth_db=None) -> dict:
        # Sidechain shared control (Tier-1): the USER keys a stem's duck off another stem (or clears
        # it), like the agent's duck action. Session.set_duck mutates session.json (the duck chain
        # entry carries key_stem_id); re-resolve so the DAW re-applies the chain → wireDuckSidechains
        # routes the key track → the duck is audible. Returns {stem, key_stem_id, chains, readout, ...}.
        from mix_engine.session import Session
        res = Session(self.project_dir).set_duck(
            stem, key,
            None if threshold_db is None else float(threshold_db),
            None if depth_db is None else float(depth_db))
        return {**res, **chains_with_readout(self.project_dir, cache=self.render_cache)}

    def copy_stem_state(self, src: str, dest: str, optional: bool = False,
                        keep: dict | None = None) -> dict:
        self._drop_stale_render_cache()   # before the change: see the method
        # Track duplicate / paste (⌘D, ⌘C→⌘V): the native side clones the track's CLIPS +
        # fader; this brings the mix state the brain owns — devices, gain/pan/width, tags,
        # rides, sends and group membership — onto the freshly registered copy. Without it a
        # duplicate came back flat (registration mints a fresh stem). light=True: the copy is
        # a new audible track, but re-rendering the whole song here is the multi-second hang
        # ⌘D must not have; the readout refreshes on the next real render.
        # optional=True: the source may legitimately have no brain stem — a drop filling a
        # blank never-registered lane (the seeded "Audio 1"), or a cross-project paste whose
        # source id belongs to another project's session. Nothing to copy is then a clean
        # no-op, not the "no track … in this project" warning. Strict callers (⌘D of a live
        # registered track) keep the loud error, so a rename-keying bug still speaks.
        from mix_engine.session import Session
        session = Session(self.project_dir)
        if optional and str(src) not in session.stems:
            return {"ok": True, "skipped": True}
        res = session.copy_stem_state(str(src), str(dest))
        if not res.get("ok"):
            return res
        # `keep`: the copy's LIVE strip as the DAW read it when it asked (reader C1·R3). The
        # copy starts as its source and may have been moved since - a fader pulled down in the
        # second the registration takes - and the source's level laid over it here came back
        # in the chains below and put the fader back.
        keep_strip_over_copy(session, str(dest), keep)
        return {**res, **chains_with_readout(self.project_dir, cache=self.render_cache,
                                             light=True)}

    def restore_stem(self, stem: str) -> dict:
        self._drop_stale_render_cache()   # before the change: see the method
        # Track cut → paste (⌘X→⌘V): deleting a DAW track only SOFT-excludes its stem (the wav
        # and the whole mix state stay — the edit's clips and undo still need them), so pasting
        # it back is an un-exclude, NOT a re-registration (which would reset the chain). The
        # devices, fader and sends come back exactly as they were.
        from mix_engine.session import Session
        s = Session(self.project_dir)
        if stem not in s.stems:
            return {"ok": False, "error": f"restore: no track {stem!r} in this project"}
        s.user_edit(str(stem), {"excluded": False})
        return {"ok": True, "stem_id": stem,
                **chains_with_readout(self.project_dir, cache=self.render_cache, light=True)}

    def set_llm(self, model: str, reasoning: str) -> dict:
        # Composer model + effort picker. Validated against the model table and kept as the
        # process env — now only a FALLBACK: every DAW request that can reach the LLM (chat,
        # run, advise, apply_plan, chat_async) carries its own {model, effort} since
        # 2026-09-22 (audit E·F1), because this notify is dropped whenever the brain is not
        # up yet (every launch, every respawn). It no longer re-points live conversations
        # either (E·F4): one brain serves every open project, and a pick in project B used to
        # switch project A's turn mid-way — a model switch inside another conversation.
        # Absolute, not relative: the DAW spawns this file BY PATH, so there is
        # no parent package and `from .chat` raises. The top-of-file bootstrap
        # already put the repo root on sys.path for exactly this reason.
        from orchestrator.chat import validate_chat_settings
        validate_chat_settings(model, reasoning)
        os.environ["MIX_AGENT_MODEL"] = model
        os.environ["MIX_AGENT_EFFORT"] = reasoning
        return {"ok": True, "model": model, "reasoning": reasoning}

    def auth_start(self) -> dict:
        # The DAW shows what this returns and then polls auth_status. The brain
        # runs the protocol because the brain is what needs the token, and it
        # has to keep refreshing one long after this screen is gone.
        from orchestrator.device_auth import DeviceAuthError, shared_login
        try:
            pending = shared_login().begin()
        except DeviceAuthError as exc:
            return {"state": "error", "message": str(exc)}
        return {
            "state": "pending",
            "user_code": pending.user_code,
            "verification_uri": pending.verification_uri,
            "expires_in": pending.expires_in,
            "interval": pending.interval,
        }

    def auth_status(self) -> dict:
        from orchestrator.device_auth import DeviceAuthError, shared_login
        try:
            login = shared_login()
            state = login.poll_once()
        except DeviceAuthError as exc:
            return {"state": "error", "message": str(exc)}
        if state == "linked":
            # Phase 53: linking no longer flips the transport to `cloud` — the agent runs
            # on the user's own key, and the sign-in is the LICENCE (trial or purchase).
            # The book is reachable now, and so is the licence state: re-ask both.
            from orchestrator import knowledge, license_state
            knowledge.reset_backoff()   # don't wait out the negative cache
            license_state.reset()       # the trial starts at first link
            return {"state": "linked"}
        return {"state": state}

    def auth_state(self) -> dict:
        from orchestrator.device_auth import DeviceAuthError, shared_login
        try:
            return {"signed_in": shared_login().signed_in()}
        except DeviceAuthError as exc:
            return {"signed_in": False, "message": str(exc)}

    def auth_sign_out(self) -> dict:
        from orchestrator.device_auth import DeviceAuthError, shared_login
        from orchestrator import license_state
        try:
            shared_login().sign_out()
        except DeviceAuthError as exc:
            return {"ok": False, "message": str(exc)}
        # ...and the licence this machine remembered goes with the session (reader S8·R8): a
        # signed-out machine otherwise believes its cached `licensed` for thirty days. The
        # reply carries what the gate reads now, so the app need not ask again.
        license_state.forget()
        return {"ok": True, "signed_in": False, "license": license_state.state()}

    def analyze(self) -> dict:
        # Descriptive readout for the DAW's top-bar BPM·key chip + loudness target —
        # detected key/tempo/LUFS off a master render (NOT the edit's default tempo).
        # Seeds the render cache, so the first ⌘G after load takes the light path.
        return session_readout(self.project_dir, cache=self.render_cache)

    def boot_sync(self) -> dict:
        # The DAW's load-time convergence (2026-08-06). session.json owns bus/return
        # EXISTENCE; the `bus:`/`return:` tracks in the edit are a projection of it that
        # realizeBuses/realizeReturns reconcile. Nothing ran that projection at load —
        # boot asked only for `analyze` (the readout), which carries no topology — so the
        # two could disagree indefinitely, and the disagreement was then resolved by
        # whichever unrelated gesture next happened to carry a topology, silently deleting
        # the user's return tracks mid-action. Worse than cosmetic: an edit holding return
        # tracks the session does not know sums wet into the bounce that the Python engine
        # cannot see, so the export oracle is comparing two different mixes.
        #
        # Same payload every routing/send/run/undo response already carries, and the same
        # single master render `analyze` does (shared render_cache), so boot costs no more
        # than before and the first ⌘G still takes the light path. READ-ONLY by
        # construction — chains_with_readout opens the Session read-only — which is what
        # makes running it on every load idempotent.
        if not self.unwritten:
            self.honour_discard()   # a marker left after this connection's hello (S1·R3)
            self.apply_owed()       # ...and the moves a reloaded project's teardown left since (S3·R2)
        return chains_with_readout(self.project_dir, cache=self.render_cache)

    def reference(self, path: str) -> dict:
        # Profile a reference track + compare to the current master (the reference-match
        # card). The reference audio never enters the mix path.
        return reference_match(self.project_dir, path, cache=self.render_cache)

    def import_stem(self, path: str, stem_id: str | None = None,
                    bus: str | None = None, replace: bool = False, start_s: float = 0.0) -> dict:
        # Register a browser-picked / dropped audio file as a project stem (copy/decode +
        # refresh_stems + auto-tag), so the agent's next pass sees + mixes the new track. The DAW
        # drag-drop places the clip instantly + calls this in the background with stem_id = the live
        # track name, so the brain stem matches the track (the agent's chains map to the live node).
        # `bus` routes the new stem into an existing group (the track-duplicate path: the copy
        # inherits its source's bus membership). `replace`: the Loop Deck's follow (D4·R1).
        # `start_s`: where the file's first sample plays - a take's punch-in (D1·R2).
        out = import_stem(self.project_dir, path, stem_id, bus, replace=bool(replace),
                          start_s=start_s)
        # New audio landed outside a turn (a drop, a split's stems): render the master and
        # compute the engine's findings NOW, on a worker, so the next message does not wait
        # for them (Phase 55 - they sat in front of the model request, 21-26 s).
        warm_observations(self.project_dir, readout=True)
        return out

    def peek_project(self, src_dir: str) -> dict:
        # Multi-project Inc 1: list another project's tracks read-only (the browser's
        # PROJECTS section + the agent's project.peek). Never touches this session.
        return peek_project(src_dir)

    def import_chain_from(self, src_dir: str, track: str, onto: str) -> dict:
        # Multi-project Inc 3: adopt another project's track's DEVICE CHAIN onto an
        # existing stem here (no audio copies; scalars/sends/routing stay). Returns the
        # fresh chains payload so the DAW realizes the adopted devices immediately.
        res = import_chain_from(self.project_dir, src_dir, track, onto)
        if not res.get("ok"):
            return res
        self.render_cache.clear()                     # the stem's processing changed
        return {**res, **chains_with_readout(self.project_dir, cache=self.render_cache)}

    def import_track_from(self, src_dir: str, track: str) -> dict:
        # Multi-project Inc 1: materialize one track from another project into this one
        # (copy audio + register + adopt chain/scalars/sends under the dependency policy).
        # Returns the fresh chains payload so the DAW realizes the new track's devices
        # in the same round-trip (like set_send/add_return).
        res = import_track_from(self.project_dir, src_dir, track)
        if not res.get("ok"):
            return res
        self.render_cache.clear()                     # a new audible track invalidates the readout
        return {**res, **chains_with_readout(self.project_dir, cache=self.render_cache)}


# NDJSON methods that read-modify-write session.json as one-shots. Refused while
# an agent pass is active (the pass's working copy and the one-shot's write would
# clobber each other — the DAW already queues device edits and flushes them before
# a run; this enforces the same rule server-side), and serialized against each
# other via Brain.mutate_lock. `import`/`open_edit` carry their own guards;
# `match_clip_key`/`extract_clip_groove`/`tune_clip_notes`/`analyze`/`boot_sync`/`get_chains` are read-only;
# `cancel` must stay free.
_SESSION_MUTATORS = frozenset({
    "undo", "coedit", "reference", "transpose", "set_send", "add_return",
    "remove_return", "set_return_prefader", "rename_return", "route_stem", "rename_stem",
    "remove_bus", "rename_bus", "route_bus", "group_stems", "set_duck", "copy_stem_state",
    "restore_stem"})


#: What a folder that is not a project yet may be asked without becoming one (reader S1·R25 d):
#: the open's own reads, the account and the key (kept outside the folder), and the traffic
#: that carries no change. ANY other request makes the folder a project first
#: (`Brain.become_project`) - by default, so a new method writes to a session that exists.
_LEAVES_FOLDER_ALONE = frozenset({
    "hello", "edit_state", "get_chains", "get_strip", "recover_inflight", "analyze",
    "boot_sync", "observations", "auth_start", "auth_status", "auth_state", "auth_sign_out",
    "set_api_key", "clear_api_key", "api_key_state", "license_state", "set_llm", "cancel",
    "redeem_product_key",
    "project_peek"})


def _resolve_use_llm(agent: str) -> bool:
    """Map the DAW's `agent` param to use_llm. "auto" defers to the brain — the
    sign-in state + backend selection live in this process, not in the shell's
    env (a Finder-launched DAW has neither MIX_AGENT_LLM_BACKEND nor a key)."""
    agent = agent.strip().lower()
    if agent in ("claude", "llm"):
        return True
    if agent == "auto":
        # Absolute import: this file is spawned by path, so a deferred relative
        # import would raise ImportError (the set_llm bug).
        from orchestrator.llm_backend import llm_available

        return llm_available()
    return False


async def _license_refusal(use_llm: bool) -> "str | None":
    """Phase 53: the sentence to refuse an LLM turn with, or None. Off the loop — the
    first ask of the day is one HTTP call. The rule agent is never gated: it costs
    nothing and the console self-tests run on it.

    Awaited INSIDE each gesture's task, never before the request's ack (2026-09-22 audit
    C·F4/F7): in front of the ack it could outlast the DAW's 15 s wait (a hung control plane
    costs a 20 s connect + an 8 s fetch) — the DAW said "rejected" while the brain went on to
    run and bill the turn — and the await sat between the run-slot check and the claim, so two
    connections could start concurrent turns on one project."""
    if not use_llm:
        return None
    from orchestrator.license_state import refusal
    return await asyncio.to_thread(refusal)


async def _auto_mix_refusal() -> "str | None":
    """Auto Mix's door: the licence, whatever agent would run (see _do_auto_mix)."""
    from orchestrator.license_state import refusal
    return await asyncio.to_thread(refusal)


async def _use_llm(agent: str) -> bool:
    """`_resolve_use_llm` off the event loop: it reads the keychain (a `security` subprocess,
    which can sit on macOS's unlock prompt), and every project's traffic waited on it
    (2026-09-22 audit C·F9)."""
    return await asyncio.to_thread(_resolve_use_llm, agent)


def spend_is_estimate() -> bool:
    """Whether this brain's `cost_usd` figures are a PRICE, not a charge (2026-09-10 audit P3,
    03·F21). On the subscription nothing is billed per token; the brain prices the SDK's
    reported tokens at API list rates, and the DAW showed that as "$0.41" spent. Told to the
    DAW once, in the hello — the backend is fixed for the life of the process. Never raises:
    a label must not be able to fail the hello, and with it the DAW's whole connection."""
    try:
        return active_backend() == "subscription"
    except ValueError:
        return False


#: The file daw/sidecar/build.sh bundles at the frozen brain's archive root (its spec's datas).
BRAIN_STAMP_FILE = "brain_stamp.txt"


def brain_stamp() -> str:
    """Which build of the brain this process is, for the hello (2026-09-22 launch review, R1).

    The frozen sidecar carries the source stamp it was built from (daw/sidecar/stamp.sh);
    package.sh launches the binary inside the finished app and refuses to ship unless this
    matches the current sources — builds 1397 and 1402 had shipped a brain frozen five weeks
    earlier. "dev" = running from source (nothing to compare), "unstamped" = a freeze made
    without build.sh. Never raises: a label must not be able to fail the hello."""
    base = getattr(sys, "_MEIPASS", None)
    if not base:
        return "dev"
    try:
        return (Path(base) / BRAIN_STAMP_FILE).read_text().strip() or "unstamped"
    except OSError:
        return "unstamped"


def _unreadable_session_chat(why: str) -> str:
    return (f"I couldn't read this project's session file (session.json — {why}), so I "
            "haven't changed anything. Your audio is untouched; the file needs repairing "
            "before I can mix.")


def _error_text(exc: BaseException) -> str:
    """Human-readable text for a pass failure. An ExceptionGroup's own message
    is just "unhandled errors in a TaskGroup" — the cause is in its leaves, so
    unwrap (recursively) and report those instead."""
    if isinstance(exc, BaseExceptionGroup):
        leaves: list[str] = []
        for sub in exc.exceptions:
            leaves.append(_error_text(sub))
        return "; ".join(dict.fromkeys(leaves))
    if isinstance(exc, RuntimeError):
        # The brain's own friendly messages (raise_cloud_friendly, guardrails)
        # travel as RuntimeError — the class name is noise in the status bar.
        return str(exc)
    return f"{type(exc).__name__}: {exc}"


def coedit_unless_gone(brain, urls: list, gone, batch: str | None = None) -> dict:
    """`Brain.coedit`, unless the DAW that asked has hung up meanwhile (pre-merge review
    2026-10-06). This runs when the brain's mutate lock comes free - behind a render that can
    be half a minute and more. A DAW that was quitting gave the batch 1.5 s, parked it for the
    next open (`session.owed.coedits.json`) and went; applied here as well, an `add_device` in
    it landed twice. A caller that is gone has kept the moves: they are not applied."""
    if gone():
        print("[daw_server] a co-edit batch was not applied: the DAW that sent it had gone "
              "(it keeps what it could not hand over)", file=sys.stderr, flush=True)
        return {"ok": False, "applied": 0, "failed": 0, "errors": [],
                "skipped": "the DAW had gone before this could be applied"}
    return brain.coedit(list(urls or []), batch)


def _failure(exc: BaseException) -> dict:
    """What a failed request or turn tells the DAW: the sentence, and - when asking again would
    only repeat it - `retryable: false` (llm_backend.NoRetry). The DAW's "retry" verb goes by
    this, not by the words in the sentence."""
    out: dict = {"error": _error_text(exc)}
    if not retry_helps(exc):
        out["retryable"] = False
    return out


def _request_llm(params: dict) -> dict:
    """The composer's pick as a request carries it: `model` + `effort` (the brain's name for
    the level is `reasoning`). Every DAW request that can reach the LLM sends both."""
    return {"model": str(params.get("model") or ""),
            "reasoning": str(params.get("effort") or ""),
            "fast": params.get("fast") is True}


async def _do_run(send, brain: Brain, run_id: int, brief: str, use_llm: bool,
                  should_cancel=None, llm: dict | None = None) -> None:
    """A run is asynchronous: the request was ack'd immediately; the (possibly multi-minute)
    agent pass runs here off the event loop and the result is streamed back as a `run_done`
    event carrying the resolved chains — so the DAW never blocks a synchronous request for
    the whole pass. `should_cancel` lets the DAW stop the run at the next pass boundary."""
    if (why := await _license_refusal(use_llm)):   # Phase 53 (C·F4 — in the task)
        await send({"event": "run_failed", "run_id": run_id, "error": why, "licence": True, "retryable": False})
        return
    await send({"event": "run_started", "run_id": run_id, "brief": brief,
                "agent": "claude" if use_llm else "rule"})
    try:
        # Live run progress (Engineer 5.2/5.3): the pass runs in a worker thread; each
        # applied action + phase/spend update is marshalled back onto THIS loop as an
        # unsolicited `run_progress` event, so the DAW's feed ticks while the agent works.
        loop = asyncio.get_running_loop()

        def on_move(move: dict) -> None:   # worker thread
            # Also publish to live_run so a mid-run status question can be answered
            # from memory — that turn must not render or read session.json.
            if (live := brain.live_run) is not None:
                live.setdefault("moves", []).append(move)
                del live["moves"][:-24]     # the answer only needs the recent ones
            asyncio.run_coroutine_threadsafe(
                send({"event": "run_progress", "run_id": run_id, "moves": [move]}), loop)

        def on_phase(info: dict) -> None:  # worker thread — {pass_number,max_passes}|{cost_usd}
            if (live := brain.live_run) is not None:
                live.update(info)
            asyncio.run_coroutine_threadsafe(
                send({"event": "run_progress", "run_id": run_id, **info}), loop)

        result = await asyncio.to_thread(brain.run, brief, use_llm, should_cancel,
                                         on_move, on_phase, llm)
        await send({"event": "run_done", "run_id": run_id, "ok": True, **result})
    except Exception as exc:  # noqa: BLE001 — report any pass failure to the DAW
        await send({"event": "run_failed", "run_id": run_id,
                    **_failure(exc)})


async def _do_chat_status(send, brain: Brain, chat_id: int, message: str,
                          session_id: str = "", llm: dict | None = None) -> None:
    """Answer a mid-run question and push it back as `chat_async_done`.

    The event name is UNIQUE and that is load-bearing: `BrainClient::waitForEvent`
    matches on event NAME ONLY (no run_id), so a reply named `chat_done` or
    `run_failed` would satisfy the RunThread's parked wait and end the real run's
    wait early. Failures therefore report through this same name too — never
    `run_failed`, which is what `_do_chat` uses.
    """
    if (why := await _license_refusal(await _use_llm("auto"))):   # Phase 53 (C·F4)
        await send({"event": "chat_async_done", "chat_id": chat_id, "ok": False, "error": why})
        return
    try:
        spent: dict = {}
        reply = await asyncio.to_thread(
            chat_status_reply, message, brain.live_run,
            dict(brain.render_cache).get("readout"), brain.latest_edit, spent, llm)
        # The side channel is stateless; hand the exchange to the real session so
        # the next turn remembers what was said mid-run instead of dropping it.
        record_midrun_exchange(brain.project_dir, message, reply, session_id)
        # A real model call: its spend reaches the session total (2026-09-10 audit P3, 03·F21).
        await send({"event": "chat_async_done", "chat_id": chat_id,
                    "ok": True, "reply": reply,
                    **({"cost_usd": spent["cost_usd"]} if spent.get("cost_usd") else {})})
    except Exception as exc:  # noqa: BLE001 — a failed question must never touch the run
        record_midrun_exchange(brain.project_dir, message,
                               "(the mid-run side channel failed to answer this)",
                               session_id)
        await send({"event": "chat_async_done", "chat_id": chat_id,
                    "ok": False, **_failure(exc)})


async def _do_auto_mix(send, brain: Brain, run_id: int, brief: str,
                       should_cancel=None, print_midi: bool = False) -> None:
    """Auto Mix mirrors a run: ack'd immediately, the (multi-render) bootstrap runs off
    the event loop, and the resolved chains stream back as a `run_done` event."""
    # The licence door (P2 batch T, reader S8·R6). Ruling 5 of Phase 53: the licence gates
    # "Chat, runs, Auto Mix and /v1/knowledge". Auto Mix is deterministic, so the door the LLM
    # turns share - keyed on `use_llm` - never met it, and an unlicensed install had the
    # product's one-button mix for good. Keyed on the LICENCE here: `refusal` answers None on
    # every developer transport, where the console legs and the suite run it.
    if (why := await _auto_mix_refusal()):
        await send({"event": "run_failed", "run_id": run_id, "error": why,
                    "licence": True, "retryable": False})
        return
    await send({"event": "run_started", "run_id": run_id, "brief": brief,
                "agent": "automix"})
    loop = asyncio.get_running_loop()

    def on_stage(event: dict) -> None:   # bootstrap thread → the DAW's status label
        asyncio.run_coroutine_threadsafe(
            send({"event": "automix_stage", "run_id": run_id,
                  "stage": str(event.get("stage", "")),
                  "detail": str(event.get("detail", ""))}), loop)

    try:
        result = await asyncio.to_thread(brain.auto_mix, brief, should_cancel, on_stage,
                                         print_midi)
        # A Stop lands on a bundle boundary with everything so far applied (one Undo away) —
        # name it, or the DAW reads a partial Auto Mix as "applied" (2026-09-02 audit).
        if should_cancel is not None and should_cancel():
            result.setdefault("stop_reason", "cancelled")
        await send({"event": "run_done", "run_id": run_id, "ok": True, **result})
    except Exception as exc:  # noqa: BLE001 — report any pass failure to the DAW
        await send({"event": "run_failed", "run_id": run_id,
                    **_failure(exc)})


async def _do_separate(send, brain: Brain, run_id: int, path: str, stems: int,
                       should_cancel=None) -> None:
    """Stem split mirrors auto-mix's shape: ack'd immediately, the model runs off
    the event loop, stages stream as `separation_stage`, and completion is a single
    `separation_done` event (ok + the stem wav names, or ok:False + error — one
    event NAME either way, because the DAW's waitForEvent matches names only).
    Nothing applies to chains; the DAW creates tracks from the returned wavs."""
    loop = asyncio.get_running_loop()

    def on_stage(event: dict) -> None:   # model thread → the DAW's status label
        asyncio.run_coroutine_threadsafe(
            send({"event": "separation_stage", "run_id": run_id,
                  "stage": str(event.get("stage", "")),
                  "detail": str(event.get("detail", ""))}), loop)

    from mix_engine.separation import ModelNotAvailable
    try:
        result = await asyncio.to_thread(brain.separate, path, stems,
                                         should_cancel, on_stage)
        await send({"event": "separation_done", "run_id": run_id,
                    "ok": bool(result.get("ok")), **{k: v for k, v in result.items()
                                                     if k != "ok"}})
    except ModelNotAvailable as exc:
        # Missing package or a failed first-use weight download (~166 MB): a clear
        # message, never a crash — the DAW stays usable.
        await send({"event": "separation_done", "run_id": run_id, "ok": False,
                    "error": f"stem separation unavailable: {exc}"})
    except Exception as exc:  # noqa: BLE001 — report any failure to the DAW
        await send({"event": "separation_done", "run_id": run_id, "ok": False,
                    **_failure(exc)})


async def _do_advise(send, brain: Brain, run_id: int, brief: str, use_llm: bool,
                     llm: dict | None = None, should_cancel=None) -> None:
    """Ask mode: like a run but advice-only — the agent runs on a throwaway copy off the event
    loop and the advice text streams back as a run_done event (no chains; nothing is applied)."""
    if (why := await _license_refusal(use_llm)):   # Phase 53 (C·F4 — in the task)
        await send({"event": "run_failed", "run_id": run_id, "error": why, "licence": True, "retryable": False})
        return
    await send({"event": "run_started", "run_id": run_id, "brief": brief, "agent": "advise"})
    try:
        result = await asyncio.to_thread(brain.advise, brief, use_llm, llm,
                                         should_cancel)   # {ok, advice, count, cost_usd}
        await send({"event": "run_done", "run_id": run_id, "ok": True, **result})
    except Exception as exc:  # noqa: BLE001 — report any pass failure to the DAW
        await send({"event": "run_failed", "run_id": run_id,
                    **_failure(exc)})


async def _do_chat(send, brain: Brain, run_id: int, message: str, mode: str,
                   use_llm: bool, should_cancel=None,
                   signed_out: bool = False, session_id: str = "",
                   llm: dict | None = None, images: list | None = None) -> None:
    """One resident-engineer turn, off the event loop. Tool calls and streamed text
    ride back as events so the Engineer panel shows work happening rather than a
    silent spinner; the final reply (+ any moves, or the ask-mode plan) lands in
    chat_done."""
    if (why := await _license_refusal(use_llm)):   # Phase 53: the door (C·F4 — in the task)
        await send({"event": "run_failed", "run_id": run_id, "error": why, "licence": True, "retryable": False})
        return
    await send({"event": "run_started", "run_id": run_id, "brief": message,
                "agent": "chat", "mode": mode})
    loop = asyncio.get_running_loop()

    def on_event(event: dict) -> None:
        kind = event.get("type")
        # `chat_message` is deliberately NOT forwarded (2026-08-21). Its texts are
        # already folded into the turn's `reply` by chat_turn's own emit hook
        # (daw_bridge.reply_parts), so chat_done delivers every one of them — the
        # streamed ones the panel has already drawn, and the ones that never
        # tokenise (the cancel notice, the max_tokens and round-cap messages)
        # through EngineerPanel::unstreamedTail. Forwarding it as well put each
        # assistant message on the wire a second time for a consumer that dropped
        # it unread on arrival. The gateway's /ws is a different path and still
        # sends chat_message to the web/VST transcript.
        if kind == "run_progress":
            # A run the agent started from this turn (2026-09-10 audit P1-22): the same live
            # feed a direct run gets — pass N/M, the ticking $, each move — and the same
            # live_run a mid-run "how's it going?" reads. Called from the pass's worker thread.
            fields = {k: v for k, v in event.items() if k != "type"}
            if (live := brain.live_run) is not None:
                if fields.get("moves"):
                    live.setdefault("moves", []).extend(fields["moves"])
                    del live["moves"][:-24]
                live.update({k: v for k, v in fields.items()
                             if k in ("pass_number", "max_passes", "cost_usd")})
            asyncio.run_coroutine_threadsafe(
                send({"event": "run_progress", "run_id": run_id, **fields}), loop)
            return
        if kind not in ("chat_delta", "chat_tool", "chat_tool_done", "chat_question",
                        "chat_plan", "chat_measured", "chat_notice"):
            return
        asyncio.run_coroutine_threadsafe(
            send({"event": "chat_progress", "run_id": run_id, **event}), loop)

    try:
        result = await asyncio.to_thread(brain.chat, message, mode, use_llm,
                                         on_event, should_cancel, session_id, llm, images)
        # Signed-out is otherwise nearly invisible: "auto" silently resolves to the
        # deterministic engineer and its replies read like AI output (2026-08-09 §1
        # audit). Say so ONCE per brain session, in the reply itself.
        if signed_out and not getattr(brain, "signed_out_noticed", False):
            brain.signed_out_noticed = True
            result = dict(result)
            # Phase 53: what makes the LLM unavailable on a shipped install is a missing API
            # key, not a sign-in — the old sentence sent people to the wrong place
            result["reply"] = ("(heads up: there's no Anthropic API key, so this is the "
                               "built-in engineer, not Claude — add your key in Preferences "
                               "▸ Agent)\n\n" + str(result.get("reply", "")))
        await send({"event": "chat_done", "run_id": run_id, "ok": True, **result})
    except Exception as exc:  # noqa: BLE001 — report any failure to the DAW
        # ...with what the failed turn billed, which the DAW adds to the session's spend
        # (2026-09-22 audit A·F4 / E·F5 — it used to read $0.00)
        spent = (getattr(exc, "cost_usd", None)   # _raise_if_landed re-raises FROM it
                 or getattr(exc.__cause__, "cost_usd", None))
        await send({"event": "run_failed", "run_id": run_id,
                    **_failure(exc),
                    **({"cost_usd": spent} if spent else {})})


async def _do_apply_plan(send, brain: Brain, run_id: int, plan: dict,
                         should_cancel=None, llm: dict | None = None) -> None:
    # A plan whose steps include a mix RUN reaches the LLM: the same licence door as the other
    # LLM entry points (2026-09-22 audit E·F13 / C·F6).
    if any(str((st or {}).get("kind", "")) == "run"
           for st in (plan.get("steps") or []) if isinstance(st, dict)):
        if (why := await _license_refusal(await _use_llm("auto"))):
            await send({"event": "run_failed", "run_id": run_id, "error": why, "licence": True, "retryable": False})
            return
    await send({"event": "run_started", "run_id": run_id, "brief": "(approved plan)",
                "agent": "chat"})
    try:
        result = await asyncio.to_thread(brain.apply_plan, plan, should_cancel, llm)
        await send({"event": "chat_done", "run_id": run_id, **result})
    except Exception as exc:  # noqa: BLE001
        await send({"event": "run_failed", "run_id": run_id,
                    **_failure(exc)})


async def _serve_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
                        hub: Hub) -> None:
    loop = asyncio.get_running_loop()
    # Connection identity (Inc 2): each connection binds to a default Brain (the boot
    # project until hello names another) and owns an opaque token for operate-path
    # claims. Any request may still route elsewhere via params.project.
    token = object()
    conn_brain: Brain = hub.boot

    def brain_for(params: dict) -> Brain:
        proj = (params or {}).get("project")
        return hub.get(str(proj)) if proj else conn_brain
    send_lock = asyncio.Lock()          # serialize writes (the read loop + background runs)
    tasks: set[asyncio.Task] = set()
    # The cancel flags of the turns THIS connection started: a connection that goes away
    # stops them (02·F8) — they would otherwise keep applying moves nobody sees, each
    # daw_command waiting out its timeout against a dead socket.
    owned: dict[int, threading.Event] = {}
    conn = {"closed": False}
    daw_acks: set[int] = set()          # confirm-gated requests the DAW says a person is reading
    daw_ack_at: dict[int, float] = {}   # ...and when it said so: the person's time, not the tool's
    daw_meta: dict[int, dict] = {}      # what the app stamped BESIDE a reply (never inside it)
    # Mid-run status questions live OUTSIDE `tasks` on purpose (2026-08-05): the
    # run guards read `any(not t.done() for t in tasks)`, so a status turn parked
    # in there would refuse the user's next run, and `cancels` iteration would
    # sweep it up and inflate Cancel's reported count. They mutate nothing, so
    # they need neither guard.
    status_tasks: set[asyncio.Task] = set()
    MAX_STATUS_TASKS = 2
    # Operate path (awareness backbone, docs/10-daw.md): the agent's daw_command tool sends a
    # daw_request to the DAW + blocks for the daw_reply. Replies resolve these futures.
    daw_replies: dict[int, concurrent.futures.Future] = {}

    async def send(obj: dict) -> None:
        async with send_lock:
            writer.write(wire_line(obj))
            await writer.drain()

    async def locked_thread(target: Brain, fn, *args):
        # One-shot session.json mutations run off-loop like everything else, but
        # under the TARGET brain's mutate_lock so two connections can't interleave
        # their load→mutate→persist cycles (last-writer-wins would drop one edit).
        def _call():
            with target.mutate_lock:
                return fn(*args)
        return await asyncio.to_thread(_call)

    # A MUTATION THAT ENDS IN A RENDER ANSWERS FROM A TASK (P2 batch T, reader S3·R6), as `undo`
    # and `invoke` already do. The user's routing gestures (remove a return, flip its tap,
    # dissolve or re-route a group, route a track) and the boot convergence each end in a full
    # offline render of the song. Awaited inline, this loop read nothing for the length of
    # that render: the DAW gave up at its timeout and called it a refusal while the brain
    # finished the change anyway (session.json had the route, the live graph did not; a second
    # click made a second one), and every later request sat behind it. Two of them still run one
    # after the other: `locked_thread` holds the brain's mutate_lock for the whole of each.
    def reply_from_task(rid, target: Brain, fn, *args) -> None:
        async def _do():
            try:
                await send({"id": rid, "result": await locked_thread(target, fn, *args)})
            except Exception as exc:  # noqa: BLE001 - report, never kill the loop
                await send({"id": rid, **_failure(exc)})
        task = asyncio.create_task(_do())
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    def daw_invoke(command: str, args: dict | None = None,
                   timeout: float | None = None) -> dict:
        """Called from the agent worker thread (the daw_command tool runs in asyncio.to_thread,
        so the event loop stays free to read the reply). Sends a daw_request to the DAW and
        blocks for its daw_reply; returns the DAW's {ok, ...} result.

        A confirm-gated command waits for the PERSON (CONFIRM_WAIT_S), and so does any request
        the DAW acknowledges with `daw_ack` (its confirm box is up). On giving up, the brain
        tells the DAW `daw_abandon` so the box goes away rather than running the command later
        under a reply that said it failed (2026-09-10 audit P2, 01·F8 / 02·F9 / 03·F11)."""
        if conn["closed"]:
            return {"ok": False, "error": "DAW disconnected"}
        if (why := implausible_integer(args)) is not None:   # F18: it would wrap in the DAW's parser
            return {"ok": False, "error": why}
        req_id = next(_DAW_REQ_IDS)
        fut: concurrent.futures.Future = concurrent.futures.Future()
        daw_replies[req_id] = fut
        asyncio.run_coroutine_threadsafe(
            send({"daw_request": req_id, "command": command, "args": args or {},
                  # groups this gesture's commands into one DAW undo transaction
                  "turn": conn_brain.agent_turn,
                  # the run it belongs to: a DAW that abandoned that run refuses it (03·F10)
                  "run_id": (conn_brain.live_run or {}).get("run_id")}), loop)
        started = time.monotonic()
        limit = reply_wait_s(command, _needs_confirm(conn_brain.latest_edit, command), timeout)
        try:
            while True:
                left = started + limit - time.monotonic()
                if left <= 0:
                    break
                try:
                    res = fut.result(timeout=min(left, 1.0))
                    _log_command(command, started, req_id)
                    # Renames first: the moves below may use the new names (follow_renames).
                    if isinstance(res, dict) and res.get("renames"):
                        if refused := conn_brain.follow_renames(res.pop("renames")):
                            res["renames_not_followed"] = refused
                    # The queue the DAW swapped into this reply can hold moves it had already
                    # SENT and got no answer for: taken here, once, or dropped - before any
                    # of the reply's consumers mirrors them.
                    if isinstance(res, dict) and isinstance(res.get("coedits"), list):
                        res["coedits"] = conn_brain.resolve_sent(res["coedits"])
                    return res
                except concurrent.futures.TimeoutError:
                    if req_id in daw_acks:
                        limit = max(limit, CONFIRM_WAIT_S)   # a person is deciding
            asyncio.run_coroutine_threadsafe(send({"daw_abandon": req_id}), loop)
            return {"ok": False, "may_have_run": True, "error": (
                f"the DAW gave no answer in {limit:.0f} s — it was told to drop the request, "
                "but it may already have run: check the edit state before trying again")}
        finally:
            daw_replies.pop(req_id, None)
            daw_acks.discard(req_id)
            daw_ack_at.pop(req_id, None)
            daw_meta.pop(req_id, None)

    def _log_command(command: str, started: float, req_id: int) -> None:
        """One row of the turn's wire log (timing record v3). The app's own milliseconds ride
        beside the reply, where the model never reads them; a confirm box's wait is dated
        from the app's `daw_ack`. A readout: it cannot fail the command."""
        try:
            now = time.monotonic()
            row: dict = {"key": command, "wait_s": round(now - started, 3)}
            app_ms = (daw_meta.get(req_id) or {}).get("app_ms")
            if isinstance(app_ms, (int, float)) and not isinstance(app_ms, bool):
                row["app_s"] = round(float(app_ms) / 1000.0, 3)
            if req_id in daw_ack_at:
                row["confirm_s"] = round(max(0.0, now - daw_ack_at[req_id]), 3)
            conn_brain.command_log.append(row)
            del conn_brain.command_log[:-2000]      # a turn of thousands keeps its newest
        except Exception:  # noqa: BLE001
            pass

    # AUTH FIRST, or nothing at all (2026-09-10 agent audit P0-2). The port used to take
    # anything that could reach 127.0.0.1: a malformed line was skipped and the loop read
    # on, so a web page's fetch() POST (its HTTP header lines skipped, its body lines
    # executed) could make the attached DAW delete a track, start a billed run, re-point
    # the brain or rewrite session.json. A browser can only ever open with an HTTP request
    # line, so requiring the FIRST line to be a JSON `auth` carrying this process's
    # per-spawn token (printed only into the DAW's owner-only log) closes that door, and
    # keeps other macOS users out too. Anything else first: close, having run nothing.
    # The operate-path seed waits for it as well, so an unauthenticated connection cannot
    # even seat itself on an empty operate slot.
    authed = False

    try:
        while True:
            line = await reader.readline()
            if not line:
                break
            if not authed:
                try:
                    first = json.loads(line)
                except ValueError:   # JSONDecodeError and UnicodeDecodeError are ValueErrors
                    first = None
                is_auth = isinstance(first, dict) and first.get("method") == "auth"
                params = first.get("params") if is_auth else None
                given = params.get("token") if isinstance(params, dict) else None
                if not (isinstance(given, str)
                        and hmac.compare_digest(given.encode(), hub.token.encode())):
                    print(f"[daw_server] refused a connection that did not authenticate "
                          f"first ({line[:16]!r})", file=sys.stderr, flush=True)
                    if is_auth and first.get("id") is not None:
                        await send({"id": first["id"], "error": "auth failed"})
                    break
                authed = True
                # Legacy seed only (undeclared): the console harnesses connect without a
                # hello role. The real claim happens in hello {role:"daw"} — and a second
                # undeclared client can no longer hijack an owned operate path.
                conn_brain.claim_operate(token, daw_invoke, declared=False)
                if first.get("id") is not None:
                    await send({"id": first["id"], "result": {"ok": True}})
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError as exc:
                # Dropping the line silently stranded the DAW caller for its full timeout
                # (120 s import, 30 min waitForEvent) with no diagnosis (2026-09-01 sweep).
                # Say so, and answer the request if its id can still be read off the line.
                # Name the METHOD that was lost. A dropped `edit_state` is not a nuisance —
                # it is the agent going blind, and the next turn then reasons about a project
                # that has moved on. The showcase hit this at ~147 KB of snapshot, where the
                # DAW's socket write returned short and this line read as "malformed": the
                # log said nothing about which method died, so it looked like noise.
                method = ""
                if (mm := re.search(rb'"method"\s*:\s*"([a-z_]+)"', line)) is not None:
                    method = mm.group(1).decode()
                blind = " — THE AGENT'S VIEW OF THE PROJECT IS NOW STALE" if method == "edit_state" else ""
                # Never the bytes of a line that carries a secret (2026-09-22 audit C·F11): a
                # truncated `set_api_key` / `auth` line put ~70 characters of the user's key
                # into brain.log — the file support asks for.
                secret = method in ("set_api_key", "auth", "redeem_product_key") or b"sk-ant-" in line
                shown = "<withheld: it carries a secret>" if secret else repr(line[:120])
                print(f"[daw_server] malformed {method or 'request'} line ignored "
                      f"({len(line)} bytes; {exc}){blind}: {shown}",
                      file=sys.stderr, flush=True)
                if (m := re.search(rb'"id"\s*:\s*(\d+)', line)) is not None:
                    await send({"id": int(m.group(1)),
                                "error": f"malformed request line: {exc}"})
                continue

            if "daw_reply" in req:   # the DAW's response to an agent daw_command
                fut = daw_replies.get(req.get("daw_reply"))
                if fut is not None and not fut.done():
                    daw_meta[req.get("daw_reply")] = {"app_ms": req.get("app_ms")}
                    fut.set_result(req.get("result"))
                continue
            if "daw_ack" in req:     # a confirm box is up for that request: wait for the person
                if req.get("daw_ack") in daw_replies:
                    daw_acks.add(req.get("daw_ack"))
                    daw_ack_at.setdefault(req.get("daw_ack"), time.monotonic())
                continue

            rid = req.get("id")
            method = req.get("method")
            params = req.get("params") or {}
            brain = brain_for(params)   # project-keyed routing (Inc 2)
            # Per-BRAIN, not hub-wide (2026-08-09 §2 audit): a run on project A used to
            # refuse co-edits/undo/imports — and new runs — on an unrelated resident
            # project B ("a run is already in progress" with no run of its own). Each
            # connection is bound to its project's Brain (hello{project}), and run_active
            # is shared by every connection on the SAME project, which is the real hazard.
            if method in _SESSION_MUTATORS and brain.run_active:
                if rid is not None:
                    await send({"id": rid, "error": "a run is already in progress"})
                continue
            try:
                if brain.unwritten and method not in _LEAVES_FOLDER_ALONE:
                    # The first request that would write makes the folder a project (S1·R25 d):
                    # held, a Session is read-only and the change would be answered and lost.
                    await locked_thread(brain, brain.become_project)
                if method == "edit_state":
                    # Awareness backbone: a fire-and-forget push of the live edit snapshot
                    # (no id, no response). Stored on the brain → folded into the next pass.
                    brain.latest_edit = params or {}
                    LATEST_EDIT[str(Path(brain.project_dir).resolve())] = brain.latest_edit   # measured_pre_limiter (Q14)
                elif method == "hello":
                    # Inc 2 handshake: `project` rebinds this connection's default brain
                    # (the resident-project model — one connection per open project);
                    # `role: "daw"` declares this connection THE DAW for that project,
                    # taking ownership of its operate path (the daw_invoke hijack fix).
                    if params.get("project"):
                        if conn_brain is not brain:
                            conn_brain.release_operate(token)
                        conn_brain = brain
                    if str(params.get("role", "")).lower() == "daw":
                        conn_brain.claim_operate(token, daw_invoke, declared=True)
                    else:
                        conn_brain.claim_operate(token, daw_invoke, declared=False)
                    # "Don't Save" / "Discard" (reader S1·R3): the DAW left its answer as a file,
                    # because it came while the link was going down or before it was up. Before
                    # the stems below are read, and before any op of this connection.
                    # A DAW's open also settles which wavs are stems (S1·R19): see open_session.
                    await locked_thread(conn_brain, conn_brain.open_session,
                                        str(params.get("role", "")).lower() == "daw",
                                        bool(params.get("stems_from_folder")),
                                        bool(params.get("unwritten")))
                    # Phase 53: the key + licence state ride the hello, so the DAW can draw
                    # its door before the first turn. Off the loop: the licence read may be
                    # one HTTP call, and the keychain is a subprocess.
                    from orchestrator import license_state
                    from orchestrator.api_key import api_key_state
                    lic, keyst = await asyncio.gather(
                        asyncio.to_thread(license_state.state),
                        asyncio.to_thread(api_key_state))
                    await send({"id": rid, "result": {"ok": True,
                                "project": str(conn_brain.project_dir),
                                "stems": conn_brain.stems(),
                                # what the open has to tell the user (an unreadable session set
                                # aside, S1·R25 b), and whether the folder is still unwritten
                                "notices": conn_brain.take_open_notices(),
                                "unwritten": bool(conn_brain.unwritten),
                                "spend_is_estimate": spend_is_estimate(),
                                "brain_stamp": brain_stamp(),
                                "license": lic, "api_key": keyst}})
                elif method == "get_chains":
                    await send({"id": rid, "result": await asyncio.to_thread(brain.get_chains)})
                elif method == "get_strip":
                    await send({"id": rid, "result": await asyncio.to_thread(brain.get_strip)})
                elif method == "recover_inflight":   # a reconnect after the brain died (03·F18)
                    await send({"id": rid, "result": await locked_thread(
                        brain, brain.recover_inflight)})
                elif method == "undo":
                    # AS A TASK, like `invoke` below (full audit 2026-10-02, reader S3·R1).
                    # Brain.undo asks the DAW for `edit.undo` through daw_invoke and waits for
                    # the daw_reply — which only THIS loop can read. Awaited inline, the loop sat
                    # in the await for the whole DAW_REPLY_TIMEOUT_S (30 s) with the DAW's answer
                    # unread in the buffer, then told the user the timeline edit "was left alone"
                    # when it had been undone half a minute earlier.
                    async def _do_undo(rid=rid, gesture=str(params.get("gesture") or "")):
                        try:
                            await send({"id": rid, "result": await locked_thread(
                                brain, brain.undo, gesture)})
                        except Exception as exc:  # noqa: BLE001 — report, never kill the loop
                            await send({"id": rid, **_failure(exc)})
                    undo_task = asyncio.create_task(_do_undo())
                    tasks.add(undo_task)
                    undo_task.add_done_callback(tasks.discard)
                elif method == "coedit":
                    urls = params.get("urls") or []
                    res = await locked_thread(brain, coedit_unless_gone, brain, urls,
                                              reader.at_eof, params.get("batch"))
                    if isinstance(res, dict) and res.get("turn_delta"):
                        # the feed's delta row for the agent's last turn, now measured
                        await send({"event": "turn_delta", **res["turn_delta"]})
                    await send({"id": rid, "result": res})
                elif method == "analyze":
                    await send({"id": rid, "result": await asyncio.to_thread(brain.analyze)})
                elif method == "boot_sync":
                    reply_from_task(rid, brain, brain.boot_sync)
                elif method == "observations":
                    # Engineer console (5.4): engine-grade findings on the raw stems —
                    # masking pair / detuned stem / sibilance / resonance. Read-only.
                    # Answered from a TASK (Phase 55): awaited here, the read loop took no
                    # other line - a daw_reply, a cancel - for as long as the analysis ran,
                    # and the app asks at every project open.
                    async def _observe(rid=rid, where=brain.project_dir) -> None:
                        try:
                            await send({"id": rid, "result": await asyncio.to_thread(
                                lambda: {"ok": True,
                                         "observations": session_observations(where)})})
                        except Exception as exc:  # noqa: BLE001 - report, never kill the loop
                            await send({"id": rid, **_failure(exc)})
                    # ...and outside `tasks`: those are what "a run is active" is read from,
                    # and a read of the stems must not make the next message answer "busy".
                    observing = asyncio.create_task(_observe())
                    status_tasks.add(observing)
                    observing.add_done_callback(status_tasks.discard)
                elif method in ("auth_start", "auth_status", "auth_state",
                                "auth_sign_out"):
                    # Each of these talks to the control plane over HTTP, so it
                    # runs off the loop like every other blocking handler.
                    handler = getattr(brain, method)
                    await send({"id": rid,
                                "result": await asyncio.to_thread(handler)})
                elif method in ("set_api_key", "clear_api_key", "api_key_state",
                                "license_state", "redeem_product_key"):
                    # Phase 53: the user's own Anthropic key + the licence. The key is
                    # validated with one free call and stored in the keychain by the brain;
                    # it is never echoed, never logged, never written into os.environ. Off
                    # the loop (a subprocess + one HTTP call each).
                    from orchestrator import api_key as _api_key, license_state
                    if method == "set_api_key":
                        res = await asyncio.to_thread(_api_key.set_api_key,
                                                      str(params.get("key", "")))
                        if res.get("ok"):
                            license_state.reset()   # the gate may just have become live
                    elif method == "clear_api_key":
                        res = await asyncio.to_thread(_api_key.clear_api_key)
                        license_state.reset()
                    elif method == "api_key_state":
                        res = await asyncio.to_thread(_api_key.api_key_state)
                    elif method == "redeem_product_key":
                        # Phase 62: a product key becomes a licence on the signed-in
                        # account; never echoed, never logged (the secret line rule above)
                        res = await asyncio.to_thread(license_state.redeem,
                                                      str(params.get("key", "")))
                    else:
                        res = await asyncio.to_thread(license_state.state,
                                                      bool(params.get("force", False)))
                    if rid is not None:
                        await send({"id": rid, "result": res})
                elif method == "set_llm":
                    # composer model/effort picker — notify (no id) or request; applies next run
                    try:
                        res = brain.set_llm(str(params.get("model", "")), str(params.get("reasoning", "")))
                        if rid is not None:
                            await send({"id": rid, "result": res})
                    except ValueError as exc:
                        if rid is not None:
                            await send({"id": rid, "error": str(exc)})
                elif method == "open_edit":
                    # Multi-edit (Inc 2): switch the brain to the edit's per-edit session dir. May
                    # arrive as a notify (no id → no reply) or a request. Refuse mid-run —
                    # including a pass started on ANOTHER connection (brain.run_active).
                    if brain.run_active or any(not t.done() for t in tasks):
                        if rid is not None:
                            await send({"id": rid, "error": "a run is already in progress"})
                        else:
                            # A NOTIFY refused in silence left the brain on the old edit and
                            # the DAW believing it had switched — every later co-edit landed
                            # in the wrong edit's session.json (2026-09-10 audit P2, 02·F8).
                            await send({"event": "open_edit_refused",
                                        "dir": str(params.get("dir", "")),
                                        "error": "a run is still in progress"})
                        continue
                    res = await locked_thread(brain, brain.open_edit, str(params.get("dir", "")))
                    if rid is not None:
                        await send({"id": rid, "result": res})
                elif method == "invoke":
                    # Invoke a DAW command via the operate path (the same channel the agent's
                    # daw_command tool uses). Run it as a TASK — not inline — so the read loop
                    # keeps reading the daw_reply (awaiting daw_invoke inline would block the loop
                    # from reading its own reply → deadlock; the agent run path is a task for the
                    # same reason). A test hook + a programmatic command trigger.
                    inv_cmd = str(params.get("command", ""))
                    inv_args = params.get("args") or {}

                    # Route through the target project's OWNED operate path (the real
                    # DAW) when one is attached; fall back to this connection's own
                    # hook (the console harness invokes against itself).
                    inv_target = brain.daw_invoke or daw_invoke

                    async def _do_invoke(_rid=rid, _cmd=inv_cmd, _args=inv_args,
                                         _target=inv_target) -> None:
                        result = await asyncio.to_thread(_target, _cmd, _args)
                        await send({"id": _rid, "result": result})

                    inv_task = asyncio.create_task(_do_invoke())
                    tasks.add(inv_task)
                    inv_task.add_done_callback(tasks.discard)
                elif method == "reference":
                    path = str(params.get("path", ""))
                    await send({"id": rid, "result": await locked_thread(brain, brain.reference, path)})
                elif method == "import":
                    # don't import into a session mid-run — on ANY connection
                    if brain.run_active or any(not t.done() for t in tasks):
                        await send({"id": rid, "error": "a run is already in progress"})
                        continue
                    path = str(params.get("path", ""))
                    sid = params.get("stem_id") or None
                    bus = params.get("bus") or None   # a duplicate inherits its source's group
                    replace = params.get("replace") is True   # the Loop Deck's follow (D4·R1)
                    start_s = params.get("start_s")           # a take's punch-in (D1·R2)
                    if (isinstance(start_s, bool) or not isinstance(start_s, (int, float))
                            or not math.isfinite(start_s) or start_s < 0.0):
                        start_s = 0.0

                    # AS A TASK (the `run` rule at _do_run): awaited inline, one big decode
                    # stalled this connection's whole read loop — every coedit/undo/chat from
                    # the tester queued silently behind the import, looking like a hang
                    # (2026-08-09 §1 audit). The reply carries the same id either way; only
                    # the loop is freed. The task also owns its errors: after the DAW's 120 s
                    # client timeout gives up, an unhandled raise here would go nowhere.
                    async def _do_import(rid=rid, path=path, sid=sid, bus=bus, replace=replace,
                                         start_s=start_s):
                        try:
                            await send({"id": rid, "result": await locked_thread(
                                brain, brain.import_stem, path, sid, bus, replace, start_s)})
                        except Exception as exc:  # noqa: BLE001 — report, never kill the loop
                            await send({"id": rid, **_failure(exc)})
                    import_task = asyncio.create_task(_do_import())
                    tasks.add(import_task)
                    import_task.add_done_callback(tasks.discard)
                elif method == "project_peek":
                    # Multi-project Inc 1: read-only listing of ANOTHER project's tracks —
                    # no session mutation, so no run guard / mutate lock needed.
                    await send({"id": rid, "result": await asyncio.to_thread(
                        brain.peek_project, str(params.get("dir", "")))})
                elif method == "project_import":
                    # Multi-project Inc 1: materialize a track from another project. Same
                    # mid-run guard as "import" (it writes session.json + renders).
                    if brain.run_active or any(not t.done() for t in tasks):
                        await send({"id": rid, "error": "a run is already in progress"})
                        continue
                    await send({"id": rid, "result": await locked_thread(brain,
                        brain.import_track_from, str(params.get("dir", "")),
                        str(params.get("track", "")))})
                elif method == "project_import_chain":
                    # Multi-project Inc 3: chain-only adopt — same mid-run guard as
                    # "project_import" (it writes session.json + renders).
                    if brain.run_active or any(not t.done() for t in tasks):
                        await send({"id": rid, "error": "a run is already in progress"})
                        continue
                    await send({"id": rid, "result": await locked_thread(brain,
                        brain.import_chain_from, str(params.get("dir", "")),
                        str(params.get("track", "")), str(params.get("onto", "")))})
                elif method == "transpose":
                    tonic = str(params.get("tonic", ""))
                    await send({"id": rid, "result": await locked_thread(brain, brain.transpose, tonic)})
                elif method == "extract_clip_groove":
                    # The window + grid-phase params were being DROPPED here after the
                    # 2026-08-08 clip-scoping fix — the DAW sent them, the bridge accepted
                    # them, and this seam silently forwarded only path+bpm, so extraction
                    # was still whole-file-from-sample-0. The invoke-surface drift class.
                    await send({"id": rid, "result": await asyncio.to_thread(
                        brain.extract_clip_groove, str(params.get("path", "")),
                        params.get("bpm"),
                        params.get("source_offset_s", 0.0),
                        params.get("length_s"),
                        params.get("beat_offset_s", 0.0))})
                elif method == "tune_clip_notes":
                    await send({"id": rid, "result": await asyncio.to_thread(
                        brain.tune_clip_notes, str(params.get("path", "")),
                        params.get("source_offset_s", 0.0),
                        params.get("length_s"),
                        params.get("min_cents", 15.0))})
                elif method == "match_clip_key":
                    await send({"id": rid, "result": await asyncio.to_thread(
                        brain.match_clip_key, str(params.get("path", "")),
                        params.get("target_tonic"), params.get("target_mode"))})
                elif method == "set_send":
                    await send({"id": rid, "result": await locked_thread(brain,
                        brain.set_send, str(params.get("source", "")),
                        str(params.get("return", "")), float(params.get("gain_db", -60.0)))})
                elif method == "add_return":
                    await send({"id": rid, "result": await locked_thread(brain,
                        brain.add_return, str(params.get("kind", "reverb")))})
                elif method == "remove_return":
                    reply_from_task(rid, brain, brain.remove_return,
                                    str(params.get("id", params.get("return", ""))))
                elif method == "set_return_prefader":
                    reply_from_task(rid, brain, brain.set_return_prefader,
                                    str(params.get("return", params.get("id", ""))),
                                    bool(params.get("pre_fader", False)))
                elif method == "rename_return":
                    await send({"id": rid, "result": await locked_thread(brain,
                        brain.rename_return, str(params.get("return", params.get("id", ""))),
                        str(params.get("new_name", params.get("new", ""))))})
                elif method == "rename_stem":
                    await send({"id": rid, "result": await locked_thread(brain,
                        brain.rename_stem, str(params.get("old", "")), str(params.get("new", "")))})
                elif method == "route_stem":
                    reply_from_task(rid, brain, brain.route_stem,
                                    str(params.get("stem", "")), str(params.get("bus", "")))
                elif method == "remove_bus":
                    reply_from_task(rid, brain, brain.remove_bus,
                                    str(params.get("id", params.get("bus", ""))))
                elif method == "rename_bus":
                    await send({"id": rid, "result": await locked_thread(brain,
                        brain.rename_bus, str(params.get("bus", params.get("old", ""))),
                        str(params.get("new_name", params.get("new", ""))))})
                elif method == "route_bus":
                    reply_from_task(rid, brain, brain.route_bus,
                                    str(params.get("bus", "")), str(params.get("target", "")))
                elif method == "group_stems":
                    raw = params.get("stems", [])
                    stems = [str(s) for s in raw] if isinstance(raw, list) else []
                    await send({"id": rid, "result": await locked_thread(brain,
                        brain.group_stems, stems)})
                elif method == "copy_stem_state":
                    await send({"id": rid, "result": await locked_thread(brain,
                        brain.copy_stem_state, str(params.get("src", "")),
                        str(params.get("dest", "")), bool(params.get("optional", False)),
                        params.get("keep") if isinstance(params.get("keep"), dict) else None)})
                elif method == "restore_stem":
                    await send({"id": rid, "result": await locked_thread(brain,
                        brain.restore_stem, str(params.get("stem", "")))})
                elif method == "set_duck":
                    await send({"id": rid, "result": await locked_thread(brain,
                        brain.set_duck, str(params.get("stem", "")), str(params.get("key", "")),
                        params.get("threshold_db"), params.get("depth_db"))})
                elif method == "auto_mix":
                    # one agent pass at a time (they mutate session.json) — shared
                    # across connections via brain.run_active, like "run" below
                    if brain.run_active or any(not t.done() for t in tasks):
                        await send({"id": rid, "error": "a run is already in progress"})
                        continue
                    brief = str(params.get("brief", ""))
                    run_id = next(_RUN_IDS)
                    cancel_ev = threading.Event()
                    brain.cancels[run_id] = owned[run_id] = cancel_ev
                    brain.run_active = True   # claim before any await (see "run")
                    brain.live_run = {"kind": method, "run_id": run_id,
                                      "brief": str(params.get("brief", "")),
                                      "started_at": time.time(), "moves": []}
                    task = asyncio.create_task(_do_auto_mix(
                        send, brain, run_id, brief, cancel_ev.is_set,
                        params.get("print_midi") is True))   # the feed's print offer, taken
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)
                    task.add_done_callback(lambda _t, k=run_id, b=brain: (
                        b.cancels.pop(k, None), owned.pop(k, None)))
                    task.add_done_callback(lambda _t, b=brain: setattr(b, "run_active", False))
                    task.add_done_callback(lambda _t, b=brain: setattr(b, "live_run", None))
                    await send({"id": rid, "result": {"run_id": run_id, "status": "started"}})
                elif method == "separate_track":
                    # stem split shares the pass slot: it's minutes of CPU and its
                    # result lands as new tracks, so one at a time, cancellable
                    # like a run (the cancel token lands on a stage boundary)
                    if brain.run_active or any(not t.done() for t in tasks):
                        await send({"id": rid, "error": "a run is already in progress"})
                        continue
                    path = str(params.get("path", ""))
                    stems = int(params.get("stems", 4) or 4)
                    run_id = next(_RUN_IDS)
                    cancel_ev = threading.Event()
                    brain.cancels[run_id] = owned[run_id] = cancel_ev
                    brain.run_active = True   # claim before any await (see "run")
                    brain.live_run = {"kind": method, "run_id": run_id,
                                      "brief": f"splitting {Path(path).name} into stems",
                                      "started_at": time.time(), "moves": []}
                    task = asyncio.create_task(_do_separate(send, brain, run_id, path,
                                                            stems, cancel_ev.is_set))
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)
                    task.add_done_callback(lambda _t, k=run_id, b=brain: (
                        b.cancels.pop(k, None), owned.pop(k, None)))
                    task.add_done_callback(lambda _t, b=brain: setattr(b, "run_active", False))
                    task.add_done_callback(lambda _t, b=brain: setattr(b, "live_run", None))
                    await send({"id": rid, "result": {"run_id": run_id, "status": "started"}})
                elif method == "advise":
                    # one agent pass at a time — advise applies nothing but bills
                    # and reads the session, so it shares the pass slot too
                    if brain.run_active or any(not t.done() for t in tasks):
                        await send({"id": rid, "error": "a run is already in progress"})
                        continue
                    brief = str(params.get("brief", ""))
                    brain.run_active = True   # claim BEFORE any await (see "run", C·F7)
                    try:
                        use_llm = await _use_llm(str(params.get("agent", "rule")))
                    except BaseException:
                        brain.run_active = False
                        raise
                    run_id = next(_RUN_IDS)
                    # Stop reaches a preview too (2026-09-22 audit B·F9): it is a whole billed
                    # run on a throwaway copy, and it answered "can't be stopped part-way".
                    cancel_ev = threading.Event()
                    brain.cancels[run_id] = owned[run_id] = cancel_ev
                    brain.live_run = {"kind": method, "run_id": run_id,
                                      "brief": str(params.get("brief", "")),
                                      "started_at": time.time(), "moves": []}
                    task = asyncio.create_task(_do_advise(send, brain, run_id, brief, use_llm,
                                                          _request_llm(params),
                                                          cancel_ev.is_set))
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)
                    task.add_done_callback(lambda _t, k=run_id, b=brain: (
                        b.cancels.pop(k, None), owned.pop(k, None)))
                    task.add_done_callback(lambda _t, b=brain: setattr(b, "run_active", False))
                    task.add_done_callback(lambda _t, b=brain: setattr(b, "live_run", None))
                    # Ask mode: ack now; the advice (on a copy, nothing applied) streams back.
                    # The ack carries the resolved agent so an "auto" caller can label its UI.
                    await send({"id": rid, "result": {"run_id": run_id, "status": "started",
                                                      "agent": "claude" if use_llm else "rule"}})
                elif method == "chat":
                    # The DAW's single front door (2026-07-30). Shares the pass slot:
                    # the resident engineer can propose, run and operate, so it mutates
                    # the same session/billing a run does.
                    if brain.run_active or any(not t.done() for t in tasks):
                        await send({"id": rid, "error": "a run is already in progress"})
                        continue
                    text = str(params.get("message", ""))
                    # Phase 64: the pictures the user attached, by path - read and checked
                    # in Brain.chat (`chat_images`), never here
                    images = params.get("images") or []
                    if not isinstance(images, list) or not all(isinstance(x, str) for x in images):
                        await send({"id": rid, "error": "images must be a list of file paths"})
                        continue
                    mode = "ask" if str(params.get("mode", "agent")) == "ask" else "agent"
                    agent_param = str(params.get("agent", "rule")).strip().lower()
                    brain.run_active = True   # claim BEFORE any await (see "run", C·F7)
                    try:
                        use_llm = await _use_llm(agent_param)
                    except BaseException:
                        brain.run_active = False
                        raise
                    # "auto" resolving to the rule agent = no key (llm unavailable) —
                    # the one-time notice rides the first reply (_do_chat).
                    chat_signed_out = agent_param == "auto" and not use_llm
                    run_id = next(_RUN_IDS)
                    # Chat is the DAW's front door and a turn can start a full mix
                    # run inside itself, so it needs a cancel event exactly like
                    # "run" does. Without one, Cancel answered `cancelled: 0` and
                    # the DAW waited out its 30-minute timeout while the run billed.
                    cancel_ev = threading.Event()
                    brain.cancels[run_id] = owned[run_id] = cancel_ev
                    # `text`, not params["brief"] — a chat request carries `message`, so
                    # reading "brief" left the mid-run status turn describing the run it
                    # was asked about as having no brief at all (2026-08-05 review).
                    brain.live_run = {"kind": method, "run_id": run_id, "brief": text,
                                      "started_at": time.time(), "moves": []}
                    task = asyncio.create_task(
                        _do_chat(send, brain, run_id, text, mode, use_llm,
                                 cancel_ev.is_set, signed_out=chat_signed_out,
                                 session_id=str(params.get("session_id", "")),
                                 llm=_request_llm(params), images=images))
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)
                    task.add_done_callback(lambda _t, k=run_id, b=brain: (
                        b.cancels.pop(k, None), owned.pop(k, None)))
                    task.add_done_callback(lambda _t, b=brain: setattr(b, "run_active", False))
                    task.add_done_callback(lambda _t, b=brain: setattr(b, "live_run", None))
                    await send({"id": rid, "result": {
                        "run_id": run_id, "status": "started", "mode": mode,
                        "agent": "claude" if use_llm else "rule"}})
                elif method == "apply_plan":
                    if brain.run_active or any(not t.done() for t in tasks):
                        await send({"id": rid, "error": "a run is already in progress"})
                        continue
                    plan = params.get("plan") or {}
                    run_id = next(_RUN_IDS)
                    cancel_ev = threading.Event()
                    brain.cancels[run_id] = owned[run_id] = cancel_ev
                    brain.run_active = True
                    # An approved plan has no brief of its own — name it the way the run
                    # feed does rather than leaving the status turn with an empty string.
                    brain.live_run = {"kind": method, "run_id": run_id,
                                      "brief": "(approved plan)",
                                      "started_at": time.time(), "moves": []}
                    task = asyncio.create_task(_do_apply_plan(send, brain, run_id, plan,
                                                              cancel_ev.is_set,
                                                              _request_llm(params)))
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)
                    task.add_done_callback(lambda _t, k=run_id, b=brain: (
                        b.cancels.pop(k, None), owned.pop(k, None)))
                    task.add_done_callback(lambda _t, b=brain: setattr(b, "run_active", False))
                    task.add_done_callback(lambda _t, b=brain: setattr(b, "live_run", None))
                    await send({"id": rid, "result": {"run_id": run_id, "status": "started"}})
                elif method == "cancel":
                    # The DAW's Cancel: flag the in-flight run(s). The agent loop polls this
                    # at each pass boundary and stops cleanly (the current pass finishes).
                    target = params.get("run_id")
                    n = 0
                    # The brain's flags, not this connection's: a reconnected DAW must be
                    # able to stop the turn its dropped predecessor started (02·F8).
                    for run_id, ev in list(brain.cancels.items()):
                        if target in (None, run_id):
                            ev.set()
                            n += 1
                    # Every long operation now carries a cancel token (run, chat,
                    # auto_mix, apply_plan), so `cancelled: 0` genuinely means there
                    # was nothing running rather than "we can't stop this" — but say
                    # which it is, because the DAW paints "cancelling…" on any ok:true.
                    busy = brain.run_active or any(not t.done() for t in tasks)
                    await send({"id": rid, "result": {
                        "ok": True, "cancelled": n,
                        "stoppable": n > 0,
                        "message": "" if n else (
                            "that operation is finishing and can't be stopped part-way "
                            "— Undo Agent Move (⌥⌘Z) takes it back" if busy else "nothing is running")}})
                elif method == "run":
                    # Shared guard (brain.run_active), not just this connection's tasks — a second
                    # connection must not start a concurrent run against the shared session/billing.
                    if brain.run_active or any(not t.done() for t in tasks):
                        await send({"id": rid, "error": "a run is already in progress"})
                        continue
                    brief = str(params.get("brief", ""))
                    # Claim the shared slot synchronously (no await before this) so a run request
                    # racing in on another connection sees it taken. Released when the task ends
                    # — or right here if resolving the agent fails. The licence door is the
                    # task's first step, never an await in front of this claim (C·F4/F7).
                    brain.run_active = True
                    try:
                        use_llm = await _use_llm(str(params.get("agent", "rule")))
                    except BaseException:
                        brain.run_active = False
                        raise
                    run_id = next(_RUN_IDS)
                    cancel_ev = threading.Event()
                    brain.cancels[run_id] = owned[run_id] = cancel_ev
                    brain.live_run = {"kind": method, "run_id": run_id,
                                      "brief": str(params.get("brief", "")),
                                      "started_at": time.time(), "moves": []}
                    task = asyncio.create_task(
                        _do_run(send, brain, run_id, brief, use_llm, cancel_ev.is_set,
                                _request_llm(params)))
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)
                    task.add_done_callback(lambda _t, k=run_id, b=brain: (
                        b.cancels.pop(k, None), owned.pop(k, None)))
                    task.add_done_callback(lambda _t, b=brain: setattr(b, "run_active", False))
                    task.add_done_callback(lambda _t, b=brain: setattr(b, "live_run", None))
                    # Ack after the slot is claimed; stream the result later as run_done/run_failed.
                    # The ack carries the resolved agent so an "auto" caller can label its UI.
                    await send({"id": rid, "result": {"run_id": run_id, "status": "started",
                                                      "agent": "claude" if use_llm else "rule"}})
                elif method == "chat_async":
                    # A question asked WHILE a run is in flight. A notify (no id): the
                    # DAW's message thread only ever WRITES to the socket — the reply
                    # comes back as a uniquely-named event that whichever read loop is
                    # currently draining picks up. Deliberately NOT gated on
                    # any_run_active: an answer that arrives after the run ends is
                    # still a correct answer.
                    chat_id = int(params.get("chat_id", 0))
                    text = str(params.get("message", ""))
                    # the licence door is the task's first step (C·F4): awaited HERE it blocked
                    # this connection's read loop — daw_reply, daw_ack and cancel for the run in
                    # flight waited behind a licence fetch
                    if len(status_tasks) >= MAX_STATUS_TASKS:
                        await send({"event": "chat_async_done", "chat_id": chat_id,
                                    "ok": False,
                                    "error": "one question at a time while a run is going"})
                        continue
                    st = asyncio.create_task(
                        _do_chat_status(send, brain, chat_id, text,
                                        str(params.get("session_id", "")),
                                        _request_llm(params)))
                    status_tasks.add(st)
                    st.add_done_callback(status_tasks.discard)
                else:
                    await send({"id": rid, "error": f"unknown method: {method}"})
            except Exception as exc:  # noqa: BLE001 — report any handler failure to the DAW
                await send({"id": rid, "error": f"{type(exc).__name__}: {exc}",
                            **({} if retry_helps(exc) else {"retryable": False})})
    finally:
        # Cancel the STATUS coroutines only. The run / chat / auto-mix / separate tasks are
        # blocked in asyncio.to_thread: cancelling the awaiting coroutine does not stop the
        # worker thread, but its done-callbacks fired at once and cleared brain.run_active
        # / live_run while the orphaned thread was still mutating session.json — so a
        # crash-and-reconnect could claim the pass slot and start a second run on the same
        # session (2026-09-01 sweep). Left alone, the task ends when its thread does (the
        # send on the dead connection raises and is caught), and the callbacks release the
        # claim at the right moment; a reconnect meanwhile sees "a run is already in progress".
        for task in status_tasks:
            task.cancel()
        conn["closed"] = True                         # later daw_commands fail at once
        for ev in list(owned.values()):               # stop what this connection started
            ev.set()
        conn_brain.release_operate(token)             # only OUR claim — a reconnect may have
                                                      # already installed a newer connection's
        for fut in daw_replies.values():              # unblock any worker awaiting a reply
            if not fut.done():
                fut.set_result({"ok": False, "error": "DAW disconnected"})
        writer.close()


SUPERVISED_ENV = "MIXAGENT_DAW_SUPERVISED"
HEARTBEAT_ENV = "MIXAGENT_DAW_HEARTBEAT"


#: The DAW touches its heartbeat file once a second; this long with no NEW touch and it is gone.
HEARTBEAT_STALE_S = 5.0
HEARTBEAT_POLL_S = 1.0


class _HeartbeatWatch:
    """Has the DAW stopped touching its heartbeat file?

    Asked by what CHANGED, on a clock that stands still while the machine sleeps - never by
    the file's age on the wall clock (full audit 2026-10-02, reader S3·R4). The watcher read
    `time.time() - mtime` once a second and left on the first reading over five seconds. A
    sleeping laptop runs neither side's one-second timer, but its wall clock runs: on wake,
    whenever the brain's check came due before the DAW's next touch - about one sleep in two -
    the age it read was the length of the sleep, and the brain was gone with exit code 0 and
    nothing in the log, taking every open project's gesture marks, render cache and
    conversation with it. So did any forward step of the date.

    Here a reading counts as life when the file's stamp DIFFERS from the last one seen
    (whatever the two stamps are - the DAW's clock may step either way), and the silence is
    ADDED UP reading by reading, each one worth at most two polls however far the clock went
    between them. So no single reading is the DAW's absence on any clock: `time.monotonic()`
    stands still over a sleep here (mach_absolute_time), and where it does not the cap does
    the same job. Five seconds of silence is three readings at the least. A file that is not
    there reads as one last change and then as silence: the same few seconds, then the DAW
    is gone."""

    def __init__(self, now: float, poll_s: float = HEARTBEAT_POLL_S) -> None:
        self._seen: object = self      # no stamp yet: the first one read is a touch
        self._last = now               # when the previous reading was taken
        self._silent = 0.0
        self._most_per_reading = 2.0 * poll_s

    def silent_for(self, stamp, now: float) -> float:
        """Seconds of readings since the file last changed, this one included (`stamp` None =
        no file). Call it once per reading, on a clock that does not run backwards."""
        step = min(now - self._last, self._most_per_reading)
        self._last = now
        if stamp != self._seen:
            self._seen, self._silent = stamp, 0.0
        else:
            self._silent += step
        return self._silent


def _exit_on_stale_heartbeat() -> None:
    """Die with the DAW even on a crash/kill: the DAW touches the heartbeat file every
    second; if it stops (or the file vanishes), the DAW is gone, so don't orphan the brain.
    More robust than the stdin guard, which the DAW's child stdin (closed) trips instantly.
    No-op unless MIXAGENT_DAW_HEARTBEAT points at the file. brain.log is told why."""
    path = os.environ.get(HEARTBEAT_ENV)
    if not path:
        return
    hb = Path(path)

    def watch() -> None:
        state = _HeartbeatWatch(time.monotonic())
        while True:
            try:
                stamp = hb.stat().st_mtime_ns
            except OSError:
                stamp = None   # file gone -> no touch
            silent = state.silent_for(stamp, time.monotonic())
            if silent > HEARTBEAT_STALE_S:
                print(f"[daw_server] the DAW's heartbeat ({hb.name}) has not moved for "
                      f"{silent:.0f} s"
                      f"{' - the file is gone' if stamp is None else ''}: the DAW is gone, "
                      "ending the brain with it", file=sys.stderr, flush=True)
                os._exit(0)
            time.sleep(HEARTBEAT_POLL_S)

    threading.Thread(target=watch, daemon=True).start()


def _exit_on_stdin_eof() -> None:
    """Die with the DAW: when it holds our stdin pipe, EOF (incl. a crash) means it's
    gone, so we don't orphan a brain process. Mirrors orchestrator/sidecar.py — but ONLY
    when the parent opted in via MIXAGENT_DAW_SUPERVISED (and thus keeps stdin open).
    Without it, a child whose stdin is /dev/null would see instant EOF and exit before
    serving; the DAW kills the child explicitly instead."""
    if not os.environ.get(SUPERVISED_ENV):
        return

    def watch() -> None:
        try:
            while sys.stdin.buffer.read(4096):
                pass
        except Exception:
            pass
        os._exit(0)

    threading.Thread(target=watch, daemon=True).start()


# The live edit snapshot (method="edit_state") is one NDJSON line and grows with the
# project — many tracks × clips × automation points easily exceed asyncio's default
# 64 KiB StreamReader limit, and readline() would then raise and tear down the whole
# connection (killing an in-flight run). Give the reader generous headroom.
NDJSON_LINE_LIMIT = 64 * 1024 * 1024   # 64 MiB per line


async def _main(project_dir: Path, port: int) -> None:
    hub = Hub(project_dir)
    server = await asyncio.start_server(
        lambda r, w: _serve_client(r, w, hub), HOST, port,
        limit=NDJSON_LINE_LIMIT)
    actual_port = server.sockets[0].getsockname()[1]
    # The per-spawn secret rides the handshake (2026-09-10 agent audit P0-2): the DAW reads
    # it from its owner-only log and presents it as every connection's first line.
    print(f"{HANDSHAKE_PREFIX}{actual_port} token={hub.token}", flush=True)
    async with server:
        await server.serve_forever()


def render_cache_mb() -> int:
    """The DAW brain's budget for kept stem renders, in MB: `MIX_AGENT_RENDER_CACHE_MB` when
    set (0 switches the caches off), else a quarter of this machine's memory, 4 GB at most.
    A rendered stem is float64 - 230 MB for five stereo minutes at 48 kHz - and the cache is
    a bounded LRU, so a budget too small for a session costs re-renders, never wrong audio."""
    raw = os.environ.get("MIX_AGENT_RENDER_CACHE_MB", "").strip()
    if raw:
        try:
            return max(0, int(float(raw)))
        except ValueError:
            pass
    try:
        ram_mb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // (1024 * 1024)
    except (ValueError, OSError, AttributeError):
        ram_mb = 8192
    return int(min(4096, ram_mb // 4))


def _keep_renders() -> None:
    """Switch on the engine's stem render cache and the master memo beside it (Phase 54).

    Both existed for the old gateway and were never switched on here, so every analysis tool,
    every proposal's baseline and every readout rendered the whole session from its files.
    In `main()`, never at import: the tests and the benchmark count renders with them off."""
    from mix_engine import dsp
    dsp.configure_render_cache(render_cache_mb())


def main() -> None:
    ap = argparse.ArgumentParser(description="Mix Agent DAW brain socket server")
    ap.add_argument("project_dir", type=Path)
    ap.add_argument("--port", type=int, default=0, help="0 = ephemeral (default)")
    args = ap.parse_args()

    # A shipped brain talks to Anthropic on the user's key and nowhere else: what the SDK
    # would read from the app's environment to go elsewhere is dropped before any client
    # exists (readers S8·R5 = S9·R3). A source checkout keeps its environment.
    # (Imported at the top with the rest: the DAW starts this file BY PATH, where a relative
    # import inside a function raises. One stood here from Py5 until Sh1's first run of daw.yml,
    # and every source brain died on it before its handshake - test_brain_starts_as_script.py.)
    drop_rerouting_env()
    _keep_renders()
    _exit_on_stale_heartbeat()   # die with the DAW (crash-safe orphan guard)
    _exit_on_stdin_eof()         # extra guard when the parent opts in (supervised)
    try:
        asyncio.run(_main(args.project_dir, args.port))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
