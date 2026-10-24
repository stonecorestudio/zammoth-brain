"""Localhost gateway — the VST sidecar's brain service (Phase 3 origin; Phase 49
scope: the web/desktop shells are archived — tag `archive/web-desktop`).

FastAPI app bound strictly to 127.0.0.1 (CLAUDE.md: localhost-only; since
Phase 49 there is NO CORS at all — no browser client remains, so no origin may
script this service). Its consumers today: the native VST plugin (HTTP polling
— /api/audition, /api/chat_log, /api/readout, devices, imports) and the pytest
contract suite, which also drives the shared session logic over the /ws
websocket (the tests' harness; no shipping UI speaks it any more). The UI
never mutates audio state directly — every inbound command goes through the
engine's proposal lifecycle tools.

The engine runs in-process over the in-memory MCP transport (the same single
process the Phase 4 sidecar will be); all engine calls share one lock so a
browser undo never interleaves with the pass loop.

Run: `python -m orchestrator.gateway [project_dir] [--port 8765] [--rule-agent]`
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import quote

import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse, Response

from . import automix
from .agents import Agent, RuleAgent
from .calibration import load_lessons, record_decision, taste_summary
from .chat import (CHAT_MODELS, DEFAULT_REASONING, MODEL, REASONING_LEVELS,
                   ChatSession, ChatTools, ScriptedChat)
from .engine_models import LIMITS  # also bootstraps server/ onto sys.path
from .loop import (GATING_POLICIES, EngineClient,
                   _is_confident_and_conservative, _is_conservative,
                   run_passes)
from .targets import reference_path_from_brief

from mcp.shared.memory import create_connected_server_and_client_session
from mix_engine.mcp_server import create_server
from mix_engine.analysis import PITCH_CLASSES
from mix_engine.session import (effective_bus_chain, effective_chain,
                                effective_master_chain, effective_return_chain,
                                send_return_entries as _send_return_entries)
from mix_engine.monitor_chain import monitor_chain_params

HOST = "127.0.0.1"  # never configurable: nothing here may be network-exposed
#: The names a request to this machine is addressed to (its `Host` header, port aside).
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]"})


def addressed_to_this_machine(headers) -> bool:
    """True when the request's `Host` names this machine (reader S9·R16).

    The Origin refusal below rests on "a browser always sends Origin", and that is false
    for a same-origin GET - which is what a DNS-REBINDING page makes: its own hostname,
    re-pointed at 127.0.0.1, so the browser treats this server as the page's own origin and
    the read routes (`/api/fs`, `/api/snapshot`, the stem audio) answered it. Such a page
    reaches the socket but still says ITS name in `Host`. A request with no Host, or any
    name but these, is not ours."""
    host = (headers.get("host") or "").strip().lower()
    if host.startswith("["):                      # [::1]:8765
        name = host.split("]", 1)[0] + "]"
    else:
        name = host.rsplit(":", 1)[0] if ":" in host else host
    return name in LOCAL_HOSTS
DEFAULT_PORT = 8765

# Phase 40 (native VST): Auto Mix progress, in plain words, for the plugin's live
# status line — so a one-press run reads "building the chain… 12s", never a blank
# "thinking". Keyed by the automix_stage events automix.bootstrap emits.
_AUTOMIX_STAGE_TEXT = {
    "tagging": "tagging the stems",
    "profiling": "profiling the mix",
    "building": "building the chain",
    "applying": "applying the moves",
    "rendering": "rendering",
}


def _monitor_full_chain(state: dict) -> list[dict]:
    """Phase 40 (native VST): the chain the live MonitorChain must mirror.

    The agent has its FULL mixing reach in the plugin — single stems, group
    busses, and the master. But the live monitor can only replay the captured
    signal through ONE serial path, so it mirrors the render path for that
    signal: ``(the stem's chain) -> (its bus chain, if routed) -> (the master
    chain)``. The plugin captures the bus as one stem (``capture-<ts>``) and the
    agent shapes THAT stem (eq/gain/comp on ``state["stems"][...]["chain"]``),
    routes it to glue busses, and tops it with the master chain. The monitor used
    to read the master chain alone, so every stem-/bus-targeted move was applied
    to the engine state and the offline export but was SILENT in the live A/B —
    "it says it applied it but I don't hear the plugins". Fold the captured
    stem's (and its bus's) effective chain ahead of the master chain so the live
    monitor matches the export.

    Only when exactly one stem is active: with several (the separation-rebalance
    path) the rebalanced multi-stem sum can't be reconstructed from a single
    captured input, so fall back to master-only — that mix is heard in the
    printed master, not the live monitor. ``effective_chain`` /
    ``effective_bus_chain`` / ``effective_master_chain`` each lead with their
    gain entry; the gains multiply in ``monitor_chain_params`` (stem fader * bus
    fader * master gain)."""
    active = [s for s in state.get("stems", {}).values() if not s.get("excluded")]
    chain: list[dict] = []
    if len(active) == 1:
        stem = active[0]
        chain += effective_chain(stem)
        bus = state.get("buses", {}).get(stem.get("bus"))
        if bus is not None:                      # the stem is glued through a bus
            chain += effective_bus_chain(bus)
        # Drop no-op faders (0 dB) so the normal case shows one master Gain tile;
        # real level moves still fold in and are heard.
        chain = [e for e in chain
                 if not (e["type"] == "gain" and e["gain_db"] == 0.0)]
        # Wet returns sum in parallel before the master chain (Phase 38 topology).
        chain += _send_return_entries(state, stem)
        if bus is not None:
            chain += _send_return_entries(state, bus)
    return chain + effective_master_chain(state)


def _audition_chain(state: dict, stem: str) -> list[dict]:
    """Phase 40 (per-track instances M3/M5): the chain ONE plugin insert plays live.

    A per-track instance passes ``stem`` = its own stem/bus id (or "master") and
    gets just THAT chain — each insert processes only its own track, and the DAW
    sums the processed tracks into the live multi-track mix.

    M5: when the requested stem is the ONLY active track, the insert is really a
    single master-bus assistant, so it gets the FULL fold (its chain + bus +
    master) — otherwise it would miss the agent's master moves (a regression M3
    introduced). With several active stems it's one of many tracks → just its own."""
    if stem == "master":
        return effective_master_chain(state)
    stems = state.get("stems", {})
    if stem and stem in stems:
        active = [s for s in stems.values() if not s.get("excluded")]
        if len(active) <= 1:
            return _monitor_full_chain(state)        # the sole track = a master-bus insert
        # one of several tracks: just this track + its own wet sends (the DAW sums)
        return effective_chain(stems[stem]) + _send_return_entries(state, stems[stem])
    if stem and stem in state.get("buses", {}):
        bus = state["buses"][stem]
        return effective_bus_chain(bus) + _send_return_entries(state, bus)
    return _monitor_full_chain(state)


def _is_per_stem(state: dict, stem: str) -> bool:
    """M4/M5: is this insert showing a read-only per-track/bus chain (vs the
    editable master / single-insert fold)? A bus, or one stem among several."""
    if not stem or stem == "master":
        return False
    if stem in state.get("buses", {}):
        return True
    stems = state.get("stems", {})
    if stem not in stems:
        return False
    active = [s for s in stems.values() if not s.get("excluded")]
    return len(active) > 1
# Phase 49: NO CORS middleware at all. The webview origins allow-list left
# with the Tauri shell (tag archive/web-desktop) — the remaining clients are
# native code (the VST) and tests, which browsers' origin rules don't govern.
# The invariant hardened from "only the webview's origins" to "no origin":
# no website in the user's browser can script this local engine, ever.
FEED_LIMIT = 500  # feed events kept for replay to (re)connecting clients
# Where File > New Project creates folders (the retired desktop shell used
# the same location, so existing users' projects are found).
DEFAULT_PROJECTS_DIR = Path.home() / "Music" / "Mix Agent"

# Proactive observations (Phase 17): hard rate limit between comments, and
# the same trigger never fires twice in a row. The chat layer decides what
# to say; these thresholds decide when it may speak at all.
OBSERVE_COOLDOWN_S = 120.0
OBSERVE_LUFS_SWING_LU = 3.0
RECENT_EDITS_KEPT = 5
# Direct-channel edits (faders, device racks) confirm instantly via a
# snapshot; the expensive bounce+analysis re-render runs in the background,
# coalesced — a burst of knob commits settles for this long, then renders
# exactly once at the latest state.
RENDER_DEBOUNCE_S = 0.4
NOTES_FILE = "notes.jsonl"
NOTES_IN_CONTEXT = 10

# Phase 26: what the import boundary accepts. Non-wav audio decodes to a wav
# copy exactly once on entry — the engine itself stays wav-only, so render
# determinism never depends on a lossy decoder's version.
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".aiff", ".aif", ".ogg")
AUDIO_EXTS_LABEL = "wav/mp3/flac/aiff/ogg"
# First bytes per extension — drag-drop uploads are sniffed before decode
# (mp3 is an ID3 tag or a bare 0xFF MPEG frame sync).
AUDIO_MAGIC = {".wav": (b"RIFF",), ".flac": (b"fLaC",),
               ".aiff": (b"FORM",), ".aif": (b"FORM",),
               ".ogg": (b"OggS",), ".mp3": (b"ID3", b"\xff")}


def transcode_to_wav(source: Path, target: Path) -> None:
    """Decode a supported audio file to a float32 wav (Phase 26) — the
    one-time conversion at the import boundary. Atomic rename so stem
    discovery (which globs *.wav) never sees a half-written file."""
    data, sample_rate = sf.read(str(source), dtype="float32", always_2d=True)
    tmp = target.with_name(target.name + ".tmp")
    sf.write(str(tmp), data, sample_rate, format="WAV", subtype="FLOAT")
    tmp.replace(target)


def adopt_foreign_audio(folder: Path) -> list[str]:
    """Phase 26: a picked project folder may hold mp3/flac/aiff/ogg stems.
    Decode each to `<name>.wav` (skipped when that wav already exists) so
    the wav-only engine can discover them. Idempotent; sources stay
    untouched. Returns the wav names created."""
    converted: list[str] = []
    for entry in sorted(folder.iterdir()):
        if (not entry.is_file() or entry.name.startswith(".")
                or entry.suffix.lower() not in AUDIO_EXTS
                or entry.suffix.lower() == ".wav"):
            continue
        target = entry.with_suffix(".wav")
        if target.exists():
            continue
        transcode_to_wav(entry, target)
        converted.append(target.name)
    return converted


def _copyfile_atomic(src: Path, target: Path) -> None:
    """Copy via tmp + rename — never truncate a project wav in place. A shell's
    engine may hold the target memory-mapped while it plays (the 2026-08-13
    slot.launch SIGBUS in the DAW — see daw_bridge._copyfile_atomic); a rename
    swaps to a new inode and any mapped reader keeps the old bytes alive."""
    tmp = target.with_name(target.name + ".tmp")
    shutil.copyfile(src, tmp)
    tmp.replace(target)


def _write_wav_atomic(target: Path, audio, sample_rate: int) -> None:
    """Write an in-memory stem to a float32 wav atomically (tmp-rename) so stem
    discovery never globs a half-written file — the Phase 29 separation analogue
    of transcode_to_wav (which writes from a source file)."""
    tmp = target.with_name(target.name + ".tmp")
    sf.write(str(tmp), audio, sample_rate, format="WAV", subtype="FLOAT")
    tmp.replace(target)


def transpose_interval(detected_tonic: str, target_tonic: str) -> int:
    """Nearest signed semitone interval from `detected_tonic` to `target_tonic`
    (Phase 32 key-picker; both pitch-class names, C..B). Range [-5, 6]: prefer
    the smaller move, so G->A is +2 (up), not +10 — and B->C is +1, not -11."""
    interval = ((PITCH_CLASSES.index(target_tonic)
                 - PITCH_CLASSES.index(detected_tonic)) % 12)
    return interval - 12 if interval > 6 else interval


class Gateway:
    """One open project, one optional active run, N websocket clients."""

    def __init__(self, agent_factory: Optional[Callable[[], Agent]] = None,
                 chat_factory: Optional[Callable[[], Any]] = None):
        self.agent_factory = agent_factory or _default_agent_factory
        self.chat_factory = chat_factory or ChatSession
        self.project_dir: Optional[Path] = None
        self.engine: Optional[EngineClient] = None
        self.clients: set[asyncio.Queue] = set()
        self.feed: list[dict] = []
        self.ui_state: dict = {}  # ui.json: per-stem display colors etc.
        self.pending: dict[str, asyncio.Future] = {}  # proposal_id -> approve?
        # Phase 20: the proposal dicts behind pending approvals, so a human
        # click can be recorded as a taste decision with its action kinds.
        self.pending_proposals: dict[str, dict] = {}
        self.chat_proposals: dict[str, dict] = {}
        self.chat: Optional[Any] = None  # conversational agent (chat.py)
        self.chat_task: Optional[asyncio.Task] = None
        # Server-authoritative "the engineer is thinking" flag (the working
        # indicator + composer Stop read it). Tied to the chat turn's actual
        # lifecycle, not the feed: a turn can end without a final text message
        # (the model only made a tool call) or be buried under the run/auto-mix
        # events it kicked off — the old client-side feed heuristic then left
        # "thinking…" stuck forever with no way to clear it.
        self.chat_busy = False
        self.chat_stream = ""  # Phase 40: in-progress assistant text for the native UI poll
        self.chat_awaiting: set[str] = set()  # chat proposals pending approval
        self.instructions: list[str] = []
        self.run_task: Optional[asyncio.Task] = None
        self.run = {"running": False, "brief": None, "gating": None,
                    "max_passes": None, "pass_number": 0, "stop_reason": None,
                    "working_stem": None}
        # Phase 40: a plain-words progress line for the native plugin UI's thinking
        # strip while a run / Auto Mix works (it emits feed events, not chat text).
        self.run_status = ""
        # Phase 16: proposal ids the last Auto Mix applied (bootstrap +
        # refinement) — the one-click-undo set.
        self.automix_undoable: list[str] = []
        # Phase 40 (native VST): the Ozone-style one-press AI button. Pressed
        # before anything is captured, it ARMS — the button reads "listening" and
        # Auto Mix fires automatically on the next capture ("press, then play").
        # Holds the brief while armed; None when idle or already running.
        self.auto_mix_armed: Optional[str] = None
        # Phase 17: the resident engineer's situational awareness. The
        # engineer is always listening — observations have no off switch,
        # only the rate limits above.
        self.workspace: dict = {}            # playhead/loop/solo from the UI
        # Phase 39: which shell the UI is — "plugin" (VST/AU; audio arrives by
        # capturing the bus on play) vs the desktop/web app (stems loaded from a
        # folder). Set by the UI's client_mode message; shapes the chat agent's
        # guidance so an empty project reads as "play to capture", not "load files".
        self.client_mode: str | None = None
        self.recent_user_edits: list[dict] = []
        # Composer settings (model + reasoning effort): survive project
        # switches — each fresh chat session is re-armed with them.
        self.chat_settings = {"model": MODEL, "reasoning": DEFAULT_REASONING}
        # Phase 19: how the engineer's moves land. "ask" = every chat
        # proposal awaits approval (manual run gating); "agent" = chat
        # proposals auto-apply and runs default to auto gating. Either way
        # moves stay guardrailed proposals — atomic, undoable. A workspace
        # preference, so it survives project switches like chat_settings.
        self.agent_mode = "ask"
        self._last_observation_key: Optional[str] = None
        self._last_observation_ts = 0.0
        self._last_observed_analysis: Optional[dict] = None
        self.last_analysis: Optional[dict] = None
        self.last_flags: list[str] = []
        self.last_master_report: Optional[dict] = None
        self.bounce_rev = 0
        # Phase 39 (VST): which signal the plugin host hears — "dry" (the live
        # bus, pass-through) or "processed" (the rendered master bounce). The
        # plugin polls /api/audition; the webview's Dry/Processed toggle sets it.
        self.audition_mode = "dry"
        # Phase 40 (native co-edit): the master device (type, occurrence) most
        # recently changed by the user — the native UI tints that knob.
        self.recent_device: Optional[tuple[str, int]] = None
        # Phase 39: seconds of bus audio captured in the current play, streamed
        # from the plugin to drive the assistant's "how much to play" ring.
        self.capture_seconds = 0.0
        # Phase 40 (per-track instances polish): live meters PER stem, so each
        # insert's analyzer shows its OWN track (not a last-writer race across
        # instances). Keyed by stem id; "" is the master / single insert.
        self.last_meters: dict = {}
        self.live_meters_by_stem: dict[str, dict] = {}
        # Phase 39 (VST): host-timeline offset (seconds) of the current capture's
        # first frame. The plugin sends it on import; /api/audition hands it back so
        # the plugin aligns the rendered bounce to where playback began (mid-song,
        # not 0). Stays put until the next capture replaces it.
        self.capture_start_s = 0.0
        # Cooperative stop for the run / auto-mix loops (2026-09-01 sweep, P1): a
        # bare task.cancel() never reached a planner blocked in a worker thread
        # (anyio's to_thread ignores cancellation until the thread returns), so
        # Stop did nothing for up to INVESTIGATE_DEADLINE_S while billing went on.
        # The DAW path has passed should_cancel since 2026-08-09; this is its twin.
        self._cancel_requested = False
        self.reference: Optional[dict] = None       # analyze_reference result
        # Phase 25: what reference runs compare against — "mix" (pre-master),
        # "master" (default, the old behavior), or "both" (mix stage, then
        # master stage). Gateway state like the reference itself.
        self.reference_scope = "master"
        # Snapshot caches (presentation-layer only — the engine stays pure):
        # per-stem analysis keyed on render-relevant state, sections keyed on
        # the mix's *structure* (boundaries don't move on gain/EQ changes).
        self._stem_cache: dict[str, tuple[str, dict]] = {}
        self._peaks_cache: dict[str, tuple[tuple, dict]] = {}
        # Phase 21: sample-preview peaks for files outside the project.
        self._preview_peaks_cache: dict[tuple, dict] = {}
        # Phase 19 perf: audio headers (sf.info) keyed on mtime — snapshots
        # broadcast on every metrics event, and re-reading every stem's
        # header each time is needless sync I/O on the event loop.
        self._info_cache: dict[str, tuple[int, int, int]] = {}
        self._sections: Optional[dict] = None
        self._sections_key: Optional[str] = None
        # Coalesced background renders for the user's direct channel: state
        # changes apply + confirm immediately; one worker task re-renders
        # after a debounce, folding op bursts into a single bounce.
        self._render_task: Optional[asyncio.Task] = None
        self._render_dirty = False
        self._render_pending = False  # snapshot bounce older than user state
        self._render_wanted_at = 0.0
        self._observe_task: Optional[asyncio.Task] = None
        self._owner_task: Optional[asyncio.Task] = None
        self._close_engine: Optional[asyncio.Event] = None

    # ---------- project / engine lifecycle ----------

    async def open_project(self, path: str | Path) -> None:
        await self.close_project()
        # Persistent stem-render cache (engine-side, off for standalone/
        # benchmark runs): a knob commit's catch-up render re-renders the
        # touched stem, not the whole session. Cleared here so a project
        # switch frees the old project's audio.
        from mix_engine import dsp
        dsp.RENDER_CACHE.clear()
        dsp.configure_render_cache(
            int(os.environ.get("MIX_AGENT_RENDER_CACHE_MB", "1024")))
        project_path = Path(path)
        if project_path.is_dir():
            # Phase 26: a picked folder may hold mp3/flac/aiff/ogg stems —
            # adopt them as wavs before the engine's wav-only discovery runs.
            await asyncio.to_thread(adopt_foreign_audio, project_path)
        server = create_server(project_path)  # raises if the dir is no project
        ready: asyncio.Event = asyncio.Event()
        self._close_engine = asyncio.Event()

        async def owner() -> None:
            # The in-memory transport's task group must be entered and exited
            # in the same task, so a dedicated task owns the connection.
            async with create_connected_server_and_client_session(server) as session:
                # Two lanes: renders run on their own lock (engine-side they
                # work on frozen state), so user mutations on the main lock
                # land in milliseconds even while a bounce is rendering.
                self.engine = EngineClient(session, lock=asyncio.Lock(),
                                           render_lock=asyncio.Lock())
                ready.set()
                await self._close_engine.wait()

        self._owner_task = asyncio.create_task(owner())
        ready_wait = asyncio.create_task(ready.wait())
        done, _ = await asyncio.wait({ready_wait, self._owner_task},
                                     return_when=asyncio.FIRST_COMPLETED)
        if self._owner_task in done:  # crashed before ready
            ready_wait.cancel()
            exc = self._owner_task.exception()
            self._owner_task = None
            raise exc or RuntimeError("engine connection closed unexpectedly")
        self.project_dir = Path(path).resolve()
        # Phase 14: classify untagged stems once at open (user tags win) —
        # tags drive guardrails and the agent's reasoning from pass one.
        await self.engine.call("auto_tag_stems")
        ui_path = self.project_dir / "ui.json"
        try:
            self.ui_state = json.loads(ui_path.read_text()) if ui_path.exists() else {}
        except json.JSONDecodeError:
            self.ui_state = {}
        self.feed = []
        self.chat = self.chat_factory()  # fresh conversation per project
        self.chat.set_settings(**self.chat_settings)
        self.chat_awaiting = set()
        self.chat_busy = False  # close_project cancelled any prior turn
        self.last_analysis = None
        self.last_flags = []
        self.last_master_report = None
        self.bounce_rev = 0
        self.reference = None
        self.reference_scope = "master"
        self._stem_cache = {}
        self._peaks_cache = {}
        self._info_cache = {}
        self._sections = None
        self._sections_key = None
        self.workspace = {}
        self.recent_user_edits = []
        self._last_observation_key = None
        self._last_observation_ts = 0.0
        self._last_observed_analysis = None

    async def new_project(self, base_dir: str | Path | None = None) -> Path:
        """Create and open a fresh empty project folder ("Untitled Project",
        then "Untitled Project 2", …) under base_dir — default the same
        ~/Music/Mix Agent the desktop shell uses for its first-run project.
        Projects save continuously (session.json on every mutation), so
        'new' is just a new folder; stems arrive via import."""
        base = Path(base_dir).expanduser() if base_dir else DEFAULT_PROJECTS_DIR
        base.mkdir(parents=True, exist_ok=True)
        target = base / "Untitled Project"
        number = 2
        while target.exists():
            target = base / f"Untitled Project {number}"
            number += 1
        target.mkdir()
        await self.open_project(target)
        return target

    async def save_project_as(self, path: str | Path) -> Path:
        """Copy the whole project folder (stems, session.json, bounces, runs)
        to `path` and switch to the copy — DAW Save As semantics. The source
        folder stays behind untouched."""
        if self.project_dir is None:
            raise RuntimeError("no project open")
        if self.run_task and not self.run_task.done():
            raise RuntimeError("cannot save while a run is active")
        target = Path(path).expanduser().resolve()
        if target.exists():
            raise RuntimeError(f"'{target}' already exists — pick a new name")
        shutil.copytree(self.project_dir, target)
        await self.open_project(target)
        return target

    async def close_project(self) -> None:
        if self.chat_task and not self.chat_task.done():
            self.chat_task.cancel()
            try:
                await self.chat_task
            except (asyncio.CancelledError, Exception):
                pass
        self.chat = None
        self.chat_awaiting.clear()
        if self.run_task and not self.run_task.done():
            self.run_task.cancel()
            try:
                await self.run_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._render_task and not self._render_task.done():
            self._render_task.cancel()
            try:
                await self._render_task
            except (asyncio.CancelledError, Exception):
                pass
        self._render_task = None
        self._render_dirty = False
        self._render_pending = False
        if self._owner_task is not None:
            self._close_engine.set()
            await self._owner_task
            self._owner_task = None
        self.engine = None
        self.project_dir = None
        self.run = dict(self.run, running=False, working_stem=None)
        self.pending.clear()
        self.instructions.clear()
        # Proposal ids are PER-PROJECT sequential (p0001…), so an Auto Mix undo set
        # carried across a project switch pointed the "Undo Auto Mix" button at the
        # NEXT project's own applied proposals (2026-09-01 sweep, P1).
        self.automix_undoable = []

    # ---------- state snapshot (engine state is ground truth) ----------

    async def snapshot(self, stale_ok: bool = False) -> dict:
        """`stale_ok=True` is the direct channel's instant confirmation: a
        user edit has changed engine state but the background re-render has
        not landed yet — serve cached per-stem analyses (and the current
        sections) instead of re-measuring synchronously. Structure (chains,
        gains, routing) always reflects live state; only the measured
        numbers may lag one render, and the post-render snapshot that
        follows refreshes them."""
        state = json.loads((self.project_dir / "session.json").read_text())
        sample_rate = None
        stems = []
        for stem_id in sorted(state["stems"]):
            stem = state["stems"][stem_id]
            path = self.project_dir / stem["path"]
            frames, file_rate = self._audio_info(path)
            if sample_rate is None:
                sample_rate = file_rate
            stem_analysis = await self._stem_analysis(stem_id, state, path,
                                                      stale_ok=stale_ok)
            stems.append({
                "stem_id": stem_id,
                "file": stem["path"],
                "duration_s": round(frames / file_rate, 3),
                "gain_db": round(stem["gain_db"], 1),
                "pan": stem["pan"],
                "width": stem["width"],
                "muted": stem["muted"],
                "excluded": stem["excluded"],
                "rides": stem["rides"],
                "lufs_s": stem_analysis["lufs_s"],
                "peak_dbfs": stem_analysis["peak_dbfs"],
                # Phase 31: this stem's tuning readout (null on unpitched stems)
                "tuning_offset_cents": stem_analysis.get("tuning_offset_cents"),
                "tuning_coherence": stem_analysis.get("tuning_coherence"),
                # composed from live state, not the (possibly stale)
                # analysis blob: a racked device must appear immediately
                "applied_chain": effective_chain(stem),
                # quote the id into the path: a key-named stem ("..._F#")
                # otherwise truncates at '#' when the webview fetches it.
                "audio_url": f"/api/audio/stem/{quote(stem_id, safe='')}",
                "color": self.ui_state.get("stem_colors", {}).get(stem_id),
                "bus": stem.get("bus"),  # Phase 23 routing
                "sends": {rid: round(gain, 1)
                          for rid, gain in sorted(stem.get("sends", {}).items())},
                # Phase 27: this stem's automation switch (section moves)
                "automation_enabled": stem.get("automation_enabled", True),
            })
        if stems:
            analysis = self.last_analysis
            if analysis is None:
                # pre-first-render only; keep it so reconnects don't re-render
                analysis = self.last_analysis = await self.engine.call("analyze_session")
            structure_key = self._structure_key(state)
            if structure_key != self._sections_key and not (
                    stale_ok and self._sections is not None):
                # stale_ok keeps serving the old boundaries (analyze_sections
                # renders every stem — never on the confirmation path); the
                # key stays unset so the post-render snapshot recomputes.
                self._sections = await self.engine.call("analyze_sections")
                self._sections_key = structure_key
        else:
            # Zero-stem project: the desktop shell opens straight into an
            # empty workspace on first run (no folder picker), and the user
            # imports stems from the sample browser. Nothing to render or
            # analyze yet — null metrics, no sections. Not cached in
            # last_analysis: the first import's render() must own that.
            analysis = {
                "lufs_i": None, "lufs_s_max": None, "true_peak_dbtp": None,
                "stereo_correlation": None, "spectral_tilt_db_per_oct": None,
                "crest_factor_db": None, "band_energy": None,
                "tempo_bpm": None, "tempo_confidence": None, "key": None,
                "stems": [], "pass_number": state["pass_number"],
                "silent": True,
            }
            self._sections = None
            self._sections_key = None
        proposals = await self.engine.call("get_proposals")
        bounce_exists = (self.project_dir / "bounces" / "master.wav").exists()
        prev_exists = (self.project_dir / "bounces" / "master.prev.wav").exists()

        def return_sources(return_id: str) -> dict[str, float]:
            sources = {}
            for stem_id, stem in sorted(state["stems"].items()):
                if return_id in stem.get("sends", {}):
                    sources[stem_id] = round(stem["sends"][return_id], 1)
            for bus_id, bus in sorted(state.get("buses", {}).items()):
                if return_id in bus.get("sends", {}):
                    sources[bus_id] = round(bus["sends"][return_id], 1)
            return sources

        return {
            "type": "snapshot",
            "project": {"name": self.project_dir.name,
                        "sample_rate": sample_rate,
                        "stem_count": len(stems),
                        "master_gain_db": round(state["master_gain_db"], 1),
                        "master_chain": effective_master_chain(state)},
            "master_report": self.last_master_report,
            "sections": self._sections,
            "reference": self.reference,
            "reference_scope": self.reference_scope,
            "stems": stems,
            # Phase 23: group busses — gain + chain + member routing
            "buses": [{
                "bus_id": bus_id,
                "gain_db": round(bus["gain_db"], 1),
                "members": sorted(
                    s for s, st in state["stems"].items()
                    if st.get("bus") == bus_id),
                "sends": {rid: round(gain, 1)
                          for rid, gain in sorted(bus.get("sends", {}).items())},
                "applied_chain": effective_bus_chain(bus),
            } for bus_id, bus in sorted(state.get("buses", {}).items())],
            # Phase 38: agent-owned wet return busses fed by stem/bus sends
            "returns": [{
                "return_id": return_id,
                "gain_db": round(ret.get("gain_db", 0.0), 1),
                "sources": return_sources(return_id),
                "applied_chain": effective_return_chain(ret),
            } for return_id, ret in sorted(state.get("returns", {}).items())],
            "metrics": analysis,
            "flags": self.last_flags,
            "proposals": proposals,
            # Phase 27: the session-global automation switch (per-stem
            # switches travel on each stem row)
            "automation_enabled": state.get("automation_enabled", True),
            "run": dict(self.run),
            # proposal ids awaiting approval right now (run gates + chat
            # proposals), so a reconnecting client regains its buttons
            "awaiting": [pid for pid, fut in self.pending.items()
                         if not fut.done()] + sorted(self.chat_awaiting),
            "limits": dict(LIMITS),
            # True while the bounce/metrics lag the user's latest direct
            # edits (a coalesced re-render is queued or running) — playback
            # holds the preview path until a fresh bounce lands.
            "render_pending": self._render_pending,
            "bounce_url": (f"/api/audio/bounce?rev={self.bounce_rev}"
                           if bounce_exists else None),
            "prev_bounce_url": (
                f"/api/audio/bounce?prev=1&rev={self.bounce_rev}"
                if prev_exists else None),
            "automix": {"undoable": list(self.automix_undoable)},
            "agent_mode": self.agent_mode,
            # Server truth for the engineer's "thinking…" indicator + its Stop.
            "chat_busy": self.chat_busy,
            # Composer settings + the menus the UI renders them from, so the
            # backend stays the single source of truth for what's selectable.
            "chat_settings": {**self.chat_settings,
                              "models": CHAT_MODELS,
                              "reasoning_levels": REASONING_LEVELS},
            "feed": self.feed[-FEED_LIMIT:],
        }

    # ---------- snapshot caches ----------

    def _audio_info(self, path: Path) -> tuple[int, int]:
        """(frames, samplerate) with an mtime-keyed cache — headers don't
        change unless the file does, and snapshots broadcast often."""
        mtime_ns = path.stat().st_mtime_ns
        cached = self._info_cache.get(str(path))
        if cached is not None and cached[0] == mtime_ns:
            return cached[1], cached[2]
        info = sf.info(str(path))
        self._info_cache[str(path)] = (mtime_ns, info.frames, info.samplerate)
        return info.frames, info.samplerate

    @staticmethod
    def _render_state(state: dict, stem_id: str) -> dict:
        """The fields that determine a stem's rendered audio. muted/excluded
        stay out: render_stem ignores them (they only gate the master sum).
        Phase 27: the automation switches gate rides/ranged entries at
        render time, so both belong in the key."""
        stem = state["stems"][stem_id]
        return {"automation": [state.get("automation_enabled", True),
                               stem.get("automation_enabled", True)],
                **{field: stem[field]
                   for field in ("path", "gain_db", "pan", "width", "chain",
                                 "rides")}}

    def _stem_analysis_key(self, stem_id: str, state: dict, mtime_ns: int) -> str:
        """Cache key covering everything that can change analyze_stem's
        output: the stem's own render state, its file, and — for sidechain
        ducks — each key stem's render state (keys render duck-free, so one
        level suffices; cycles are impossible by construction)."""
        duck_keys = sorted({entry["key_stem_id"]
                            for entry in state["stems"][stem_id]["chain"]
                            if entry["type"] == "duck"
                            and entry.get("key_stem_id") in state["stems"]})
        return json.dumps({
            "stem": self._render_state(state, stem_id),
            "mtime_ns": mtime_ns,
            "duck_keys": {key: self._render_state(state, key)
                          for key in duck_keys},
        }, sort_keys=True)

    async def _stem_analysis(self, stem_id: str, state: dict, path: Path,
                             stale_ok: bool = False) -> dict:
        """analyze_stem with a render-state cache: a fader commit re-analyzes
        only the moved stem instead of re-rendering every stem per snapshot.
        `stale_ok=True` (instant edit confirmation) serves a mismatched cache
        entry rather than re-rendering; the key stays unchanged, so the next
        fresh snapshot re-measures."""
        key = self._stem_analysis_key(stem_id, state, path.stat().st_mtime_ns)
        cached = self._stem_cache.get(stem_id)
        if cached is not None and (stale_ok or cached[0] == key):
            return cached[1]
        analysis = await self.engine.call("analyze_stem", {"stem_id": stem_id})
        self._stem_cache[stem_id] = (key, analysis)
        return analysis

    @staticmethod
    def _structure_key(state: dict) -> str:
        """What section *boundaries* depend on: which stems are audible,
        their audio files, and their time structure (rides). Gain/pan/EQ
        moves shift levels, not boundaries — per-section lufs_i in the UI
        may lag a little until the structure next changes; the agent's own
        analyze_sections tool calls always measure fresh."""
        return json.dumps([
            [state.get("automation_enabled", True)],  # Phase 27: gates rides
        ] + [
            [stem_id, stem["path"], stem["muted"], stem["excluded"],
             stem["rides"], stem.get("automation_enabled", True)]
            for stem_id, stem in sorted(state["stems"].items())
        ], sort_keys=True)

    # ---------- waveform peaks (clip canvases + sample preview) ----------

    def stem_peaks(self, stem_id: str, path: Path, buckets: int) -> dict:
        """wav_peaks with the per-stem cache (cleared on project switch)."""
        key = (path.stat().st_mtime_ns, buckets)
        cached = self._peaks_cache.get(stem_id)
        if cached and cached[0] == key:
            return cached[1]
        result = wav_peaks(path, buckets)
        self._peaks_cache[stem_id] = (key, result)
        return result

    PREVIEW_PEAKS_CACHED = 32  # browsing a big sample folder stays bounded

    def preview_peaks(self, path: Path, buckets: int) -> dict:
        """wav_peaks for sample-browser previews (Phase 21): keyed on
        absolute path + mtime, small FIFO eviction."""
        key = (str(path), path.stat().st_mtime_ns, buckets)
        cached = self._preview_peaks_cache.get(key)
        if cached is not None:
            return cached
        result = wav_peaks(path, buckets)
        self._preview_peaks_cache[key] = result
        while len(self._preview_peaks_cache) > self.PREVIEW_PEAKS_CACHED:
            self._preview_peaks_cache.pop(next(iter(self._preview_peaks_cache)))
        return result

    # ---------- broadcast ----------

    def broadcast(self, event: dict) -> None:
        for queue in self.clients:
            queue.put_nowait(event)

    async def broadcast_snapshot(self, stale_ok: bool = False) -> None:
        if self.engine is not None:
            self.broadcast(await self.snapshot(stale_ok=stale_ok))

    # ---------- coalesced background renders (user direct channel) ----------

    def request_render(self) -> None:
        """Queue a re-render of ground truth without blocking the caller.
        Bursts coalesce: ops landing while a render is queued or running
        mark it dirty, and exactly one render of the *latest* state follows
        — never a backlog of stale bounces."""
        self._render_dirty = True
        self._render_pending = True
        self._render_wanted_at = asyncio.get_running_loop().time()
        if self._render_task is None or self._render_task.done():
            self._render_task = asyncio.create_task(self._render_worker())

    async def _render_worker(self) -> None:
        loop = asyncio.get_running_loop()
        while self._render_dirty:
            # sliding debounce: let a burst of commits settle into one render
            wait = self._render_wanted_at + RENDER_DEBOUNCE_S - loop.time()
            if wait > 0:
                await asyncio.sleep(wait)
                continue
            self._render_dirty = False
            if self.engine is None:  # project closed under us
                self._render_pending = False
                return
            try:
                await self.render()  # _emit settles _render_pending
            except Exception as exc:
                self._render_pending = self._render_dirty
                self.broadcast({"type": "error", "message": str(exc)})

    # ---------- run lifecycle ----------

    def start_run(self, brief: str, gating: str = "manual",
                  max_passes: int = 5) -> None:
        if self.engine is None:
            raise RuntimeError("no project open")
        if self.run_task and not self.run_task.done():
            raise RuntimeError("a run is already active")
        # "mode" (2026-06-12 bug fix) follows the Ask/Agent switch live —
        # the gateway-only policy chat runs get when no explicit gating is
        # asked for. The loop's own policies stay frozen per run.
        if gating != "mode" and gating not in GATING_POLICIES:
            raise RuntimeError(f"unknown gating policy: {gating}")
        self.run.update(running=True, brief=brief, gating=gating,
                        max_passes=max_passes, pass_number=0, stop_reason=None,
                        working_stem=None)
        self.run_task = asyncio.create_task(self._run(brief, gating, max_passes))

    def stop_run(self) -> None:
        """User-initiated abort (UI Stop button)."""
        if self.run_task is None or self.run_task.done():
            raise RuntimeError("no active run")
        self._cancel_requested = True     # the planner checks this between rounds
        self.run_task.cancel()

    # ---------- Auto Mix (Phase 16) ----------

    def start_auto_mix(self, brief: str, gating: str = "auto") -> None:
        """The one-button flow: deterministic bootstrap chain, then refinement
        passes. `gating` defaults to "auto" (the one-click button / Agent mode);
        the chat path passes "mode" so it follows the Ask/Agent switch — Ask turns
        every move (bootstrap + refine) into an approve/reject card, Agent runs
        hands-off. Same task slot as a run. The plugin runs ONE refinement pass
        (the deterministic bootstrap already lays a full chain) so the one-press
        button finishes fast; the desktop keeps two."""
        if self.engine is None:
            raise RuntimeError("no project open")
        if self.run_task and not self.run_task.done():
            raise RuntimeError("a run is already active")
        passes = 1 if self.client_mode == "plugin" else 2
        self.run.update(running=True, brief=f"auto mix — {brief}",
                        gating=gating, max_passes=passes, pass_number=0,
                        stop_reason=None, working_stem=None)
        self.run_status = "listening"        # first thing the plugin status shows
        self.run_task = asyncio.create_task(self._auto_mix(brief, gating, passes))

    async def _auto_mix(self, brief: str, gating: str = "auto",
                        max_passes: int = 2) -> None:
        try:
            # bootstrap appends to this list as each bundle lands, rather than only
            # returning it at the end. A cancel raises CancelledError mid-loop, so the
            # return value never arrived — and every proposal already applied was left
            # OUT of the one-click-undo set, unreachable by the button whose whole job
            # is to take Auto Mix back. Now the ids are here either way.
            self.automix_undoable = []
            bootstrap_ids = self.automix_undoable
            self._cancel_requested = False
            await automix.bootstrap(self.engine, brief,
                                    emit=self._emit,
                                    gate=self._gate(gating),
                                    applied=bootstrap_ids,
                                    should_cancel=lambda: self._cancel_requested)
            # Phase 25: a "both"-scope reference target refines in stages —
            # mix first, then master — on one agent (one LLM budget).
            agent = self.agent_factory()
            refined: list[str] = []
            for stage, scope in self._reference_stages(brief):
                report = await run_passes(
                    self.engine, self.project_dir, brief, agent, max_passes=max_passes,
                    gate=self._gate(gating), emit=self._emit,
                    drain_instructions=self._drain_instructions,
                    reference_scope=scope, stage=stage,
                    should_cancel=lambda: self._cancel_requested)
                refined += report.applied_proposal_ids
            self.automix_undoable = bootstrap_ids + refined
            await self._emit({"type": "automix_finished",
                              "stop_reason": report.stop_reason,
                              "applied": len(self.automix_undoable)})
        except asyncio.CancelledError:
            # Report what actually landed, not 0. Anything applied before the stop is
            # still in the session and still in automix_undoable, so Undo reaches it.
            self.run.update(stop_reason="stopped")
            self.broadcast({"type": "automix_finished",
                            "stop_reason": "stopped",
                            "applied": len(self.automix_undoable)})
        except Exception as exc:  # surfaced, never silent (docs/04)
            self.run.update(stop_reason="error")
            self.broadcast({"type": "error", "message": f"auto mix failed: {exc}"})
            # Also land it in the transcript so the native plugin UI shows the
            # failure (it doesn't render bare `error` events) instead of just
            # dropping back to idle with no explanation.
            await self._emit({"type": "chat_message", "role": "assistant",
                              "text": f"Auto Mix hit an error and stopped: {exc}"})
            self.broadcast({"type": "automix_finished", "stop_reason": "error",
                            "applied": 0})
        finally:
            self.run.update(running=False, working_stem=None)
            self.run_status = ""             # clear the native status line
            for future in self.pending.values():
                if not future.done():
                    future.set_result(False)
            await self.broadcast_snapshot()

    async def undo_auto_mix(self) -> None:
        """One click undoes the entire Auto Mix (bootstrap + refinement),
        newest first — restoring the pre-automix render bit-identically."""
        if self.engine is None:
            raise RuntimeError("no project open")
        if not self.automix_undoable:
            raise RuntimeError("no auto mix to undo")
        # Mirrors loop._revert_to (2026-09-01 sweep, P1): the engine REFUSES an
        # out-of-order undo by returning ok=False when a later applied proposal
        # holds the same target. Raising on the first refusal without trimming
        # the list left an already-undone id at its head, so every retry failed
        # on "'p0005' is 'undone', not 'applied'" — the button was dead and the
        # mix half-reverted. Now: an id that is already gone is dropped, an id
        # that undoes is dropped, and a genuine dependency refusal stops the walk
        # with the ids that DID undo removed, so the next click resumes.
        remaining = list(self.automix_undoable)
        stopped: Optional[str] = None
        for proposal_id in reversed(self.automix_undoable):
            result = await self.engine.call("undo_proposal",
                                            {"proposal_id": proposal_id})
            if not result["ok"]:
                message = result.get("message", "")
                if "not 'applied'" in message:
                    remaining.remove(proposal_id)   # a user undo got there first — fine
                    continue
                stopped = f"undo of '{proposal_id}' stopped: {message}"
                break
            remaining.remove(proposal_id)
            self.broadcast({"type": "proposal_status",
                            "proposal_id": proposal_id,
                            "status": "undone", "message": ""})
        self.automix_undoable = remaining
        await self.render()  # metrics event + snapshot follow
        if stopped:
            raise RuntimeError(stopped + "; the rest of the auto mix is still "
                               "undoable once that later proposal is undone")

    # ---------- Stem separation (Phase 29) ----------

    def start_separation(self, path: str, stems: int = 4,
                         separator: Optional[str] = None) -> None:
        """Split a mixed track into stems — a USER import action, never the
        agent's and never an MCP tool (hard invariant). Runs the model off the
        event loop with staged progress; shares the run task slot, so it is
        mutually exclusive with runs / auto mix and cancellable via stop_run."""
        if self.engine is None:
            raise RuntimeError("no project open")
        if self.run_task and not self.run_task.done():
            raise RuntimeError("a run is already active")
        # No mix-run state: separation isn't a mixing pass — the UI's progress
        # comes from separation_stage events; the task slot gives mutual
        # exclusion + cancellation (stop_run).
        self.run_task = asyncio.create_task(self._separate(path, stems, separator))

    async def _separate(self, path: str, stems: int,
                        separator: Optional[str]) -> None:
        from mix_engine.separation import (ModelNotAvailable, get_separator,
                                           stem_names)
        try:
            source = Path(path).expanduser().resolve()
            if not source.is_file() or source.suffix.lower() not in AUDIO_EXTS:
                raise ValueError(
                    f"not a supported audio file ({AUDIO_EXTS_LABEL}): {source}")
            await self._emit({"type": "separation_stage", "stage": "loading",
                              "detail": source.name})
            audio, sample_rate = await asyncio.to_thread(
                sf.read, str(source), dtype="float32", always_2d=True)
            sep = get_separator(stems, separator)
            # First use downloads the weights — its own stage with the size, or
            # the UI reads as a hang while 166 MB quietly pulls.
            if hasattr(sep, "models_cached") and not sep.models_cached():
                await self._emit({"type": "separation_stage",
                                  "stage": "downloading",
                                  "detail": sep.download_detail()})
                await asyncio.to_thread(sep.fetch_models)
            await self._emit({"type": "separation_stage", "stage": "separating",
                              "detail": f"{stems}-stem"})
            result = await asyncio.to_thread(sep.separate, audio, sample_rate)
            written: list[str] = []
            for name in stem_names(stems):
                await self._emit({"type": "separation_stage", "stage": "writing",
                                  "detail": name})
                target = self.project_dir / f"{source.stem}.{name}.wav"
                if target.exists():
                    continue  # idempotent — re-splitting skips existing stems
                await asyncio.to_thread(_write_wav_atomic, target,
                                        result[name], sample_rate)
                written.append(target.stem)
            # ...and say which files a split made (E5·R7): the separation check reads only those
            await self.engine.call("refresh_stems", {"separated": [
                f"{source.stem}.{name}.wav" for name in stem_names(stems)]})
            if written:
                await self.engine.call("auto_tag_stems", {"stem_ids": written})
            await self.render()  # bounce + metrics include the new stems
            await self._emit({"type": "separation_finished", "stop_reason": "done",
                              "stems": written})
        except asyncio.CancelledError:
            await self._emit({"type": "separation_finished",
                              "stop_reason": "stopped", "stems": []})
        except ModelNotAvailable as exc:
            # Missing package/model or a failed first-use weight download
            # (demucs resolves inside separate(), off-loop): a clear message,
            # not a crash — the engine stays usable. No LLM tokens here, so
            # MIX_AGENT_RUN_BUDGET_USD is never touched. Must precede the
            # generic branch (ModelNotAvailable subclasses RuntimeError).
            self.broadcast({"type": "error",
                            "message": f"stem separation unavailable: {exc}"})
            await self._emit({"type": "separation_finished",
                              "stop_reason": "no_model", "stems": []})
        except (ValueError, RuntimeError, KeyError) as exc:  # surfaced, never silent
            self.broadcast({"type": "error",
                            "message": f"stem separation failed: {exc}"})
            await self._emit({"type": "separation_finished",
                              "stop_reason": "error", "stems": []})
        finally:
            await self.broadcast_snapshot()

    def _reference_stages(self, brief: str) -> list[tuple[Optional[str], str]]:
        """Phase 25: (stage_label, reference_scope) sequence for this brief.
        Non-reference briefs and master scope keep the single old-behavior
        run; "both" = mix stage then master stage."""
        if reference_path_from_brief(brief) is None:
            return [(None, "master")]
        if self.reference_scope == "both":
            return [("mix", "mix"), ("master", "master")]
        return [(None, self.reference_scope)]

    async def _run(self, brief: str, gating: str, max_passes: int) -> None:
        try:
            # One agent across stages: MIX_AGENT_RUN_BUDGET_USD bounds the
            # whole run, not each stage (2026-06-12 cost incident).
            agent = self.agent_factory()
            self._cancel_requested = False
            for stage, scope in self._reference_stages(brief):
                await run_passes(self.engine, self.project_dir, brief,
                                 agent, max_passes,
                                 gate=self._gate(gating), emit=self._emit,
                                 drain_instructions=self._drain_instructions,
                                 reference_scope=scope, stage=stage,
                                 should_cancel=lambda: self._cancel_requested)
        except asyncio.CancelledError:  # stop_run / close_project
            self.run.update(stop_reason="stopped")
            self.broadcast({"type": "run_finished", "stop_reason": "stopped",
                            "passes_run": self.run["pass_number"],
                            "applied_proposal_ids": []})
        except Exception as exc:  # surfaced, never silent (docs/04)
            self.run.update(stop_reason="error")
            self.broadcast({"type": "error", "message": f"run failed: {exc}"})
            self.broadcast({"type": "run_finished", "stop_reason": "error",
                            "passes_run": self.run["pass_number"],
                            "applied_proposal_ids": []})
        finally:
            self.run.update(running=False, working_stem=None)
            for future in self.pending.values():
                if not future.done():
                    future.set_result(False)
            await self.broadcast_snapshot()

    def _gate(self, gating: str):
        async def gate(proposal_id: str, proposal: dict) -> bool:
            # "mode" resolves against the Ask/Agent switch at *gate time*,
            # not run start (2026-06-12 bug fix: flipping the switch
            # mid-run must take effect on the very next proposal — the old
            # closure froze the policy for the whole run). Explicit
            # policies stay frozen: Auto Mix's "auto" and a requested
            # "auto_conservative" keep meaning what they said.
            effective = gating
            if gating == "mode":
                effective = "auto" if self.agent_mode == "agent" else "manual"
            if effective == "auto":
                return True
            if (effective == "auto_conservative"
                    and all(_is_conservative(a) for a in proposal["actions"])):
                return True
            # Phase 13 earned autonomy: confident + conservative moves
            # auto-apply; everything else queues for browser approval.
            if (effective == "auto_confident"
                    and _is_confident_and_conservative(proposal)):
                return True
            future: asyncio.Future = asyncio.get_running_loop().create_future()
            self.pending[proposal_id] = future
            # Phase 20: kept so decide() can record the human's click as a
            # taste decision; run teardown resolves futures False without a
            # human, which is why recording lives in decide(), not here.
            self.pending_proposals[proposal_id] = proposal
            self.broadcast({"type": "awaiting_approval", "proposal_id": proposal_id})
            try:
                return await future
            finally:
                self.pending.pop(proposal_id, None)
                self.pending_proposals.pop(proposal_id, None)
        return gate

    async def _emit(self, event: dict) -> None:
        if event["type"] == "pass_started":
            self.run["pass_number"] = event["pass_number"]
            # " · pass N/M", not an em dash: the native run line SPLITS on that middle-dot
            # separator to fold the pass over beside the clock ("REFINING    pass 2/3 · 0:41").
            # It searched for a separator nothing ever wrote, so the fold never happened and the
            # whole string sat glued in the phase slot.
            total = self.run.get("max_passes") or 0
            counted = (f"{event['pass_number']}/{total}" if total
                       else str(event['pass_number']))
            self.run_status = f"refining · pass {counted}"
        elif event["type"] == "automix_stage":   # Phase 40: native status line
            self.run_status = _AUTOMIX_STAGE_TEXT.get(event.get("stage", ""),
                                                      "working")
        elif event["type"] == "proposal":
            stems = [a["stem_id"] for a in event["actions"]
                     if a.get("stem_id") and a["stem_id"] != "master"]
            self.run["working_stem"] = stems[0] if stems else None
        elif event["type"] == "metrics":
            self.last_analysis = event["analysis"]
            self.last_flags = event["flags"]
            self.last_master_report = event.get("master")
            self.bounce_rev += 1
            # any full render reflects all state applied before it started;
            # pending stays up only if edits arrived since (dirty again)
            self._render_pending = self._render_dirty
        elif event["type"] in ("run_finished", "automix_finished"):
            self.run["stop_reason"] = event.get("stop_reason")
            self.run_status = ""
        self.feed.append(event)
        del self.feed[:-FEED_LIMIT]
        self.broadcast(event)
        if event["type"] == "metrics":  # chains/gains may have changed
            await self.broadcast_snapshot()

    def _drain_instructions(self) -> list[str]:
        drained, self.instructions = self.instructions, []
        return drained

    # ---------- conversational agent (chat.py) ----------

    def chat_message(self, text: str) -> None:
        """One user chat message. The resident engineer stays available
        during runs (Phase 17): it answers with the read tools and files
        directive asks as instructions for the loop — it never goes mute."""
        if self.engine is None:
            raise RuntimeError("no project open")
        if self.chat_task and not self.chat_task.done():
            raise RuntimeError("the agent is still responding — give it a moment")
        during_run = bool(self.run_task and not self.run_task.done())
        self.chat_stream = ""            # Phase 40: reset the native UI's stream buffer
        event = {"type": "chat_message", "role": "user", "text": text}
        self.feed.append(event)
        self.broadcast(event)
        self._chat_log(event)
        # Mark busy synchronously, before the task is even scheduled, so a
        # snapshot in the gap can't read "idle" while a turn is queued.
        self._set_chat_busy(True)
        self.chat_task = asyncio.create_task(self._chat_turn(text, during_run))

    def _set_chat_busy(self, busy: bool) -> None:
        """Flip the engineer's working flag and tell the clients instantly
        (like agent_mode — the confirming snapshot can sit behind the engine
        lock for seconds during a run). The snapshot echoes the same value, so
        a reconnecting client regains the right indicator."""
        self.chat_busy = busy
        self.broadcast({"type": "chat_busy", "busy": busy})

    def stop_chat(self) -> None:
        """User-initiated abort of the engineer's current work (the composer Stop /
        the plugin's Stop) — the chat turn AND any run / Auto Mix it kicked off, so
        one Stop halts everything the agent is doing. Tolerant — a no-op if nothing
        is running, since the UI may fire it racing the work's own completion."""
        if self.chat_task and not self.chat_task.done():
            self.chat_task.cancel()
        if self.run_task and not self.run_task.done():
            self._cancel_requested = True
            self.run_task.cancel()     # the agent's Auto Mix / run is "the agent thinking" too
        self.auto_mix_armed = None      # disarm a pending "listening"

    async def _chat_turn(self, text: str, during_run: bool = False) -> None:
        try:
            await self.chat.turn(text, tools=self._chat_tools(),
                                 context=self._chat_context(),
                                 emit=self._emit_chat,
                                 during_run=during_run)
        except asyncio.CancelledError:
            pass  # the user stopped the engineer (stop_chat / project close)
        except Exception as exc:  # surfaced, never silent
            self.broadcast({"type": "error", "message": f"chat failed: {exc}"})
        finally:
            # Always clears the working indicator, however the turn ended —
            # a no-text turn, an exception, or a cancellation included.
            self._set_chat_busy(False)

    async def _emit_chat(self, event: dict) -> None:
        if event["type"] == "chat_delta":  # deltas are transient, not replayed
            self.chat_stream += event.get("text", "")   # Phase 40: native UI polls this
        else:
            if event.get("role") == "assistant":
                self.chat_stream = ""      # the final reply is in the feed now
            self.feed.append(event)
            del self.feed[:-FEED_LIMIT]
            self._chat_log(event)
        self.broadcast(event)

    def _chat_log(self, event: dict) -> None:
        """Persist the conversation next to the run logs (runs/chat.jsonl)."""
        if self.project_dir is None:
            return
        runs_dir = self.project_dir / "runs"
        runs_dir.mkdir(exist_ok=True)
        with (runs_dir / "chat.jsonl").open("a") as f:
            f.write(json.dumps({**event, "ts": round(time.time(), 3)}) + "\n")

    def _chat_tools(self) -> ChatTools:
        return ChatTools(engine_call=self._chat_engine_call,
                         propose=self._chat_propose,
                         start_run=self._chat_start_run,
                         stop_run=self._chat_stop_run,
                         auto_mix=self._chat_auto_mix,
                         file_instruction=self._chat_file_instruction,
                         save_note=self._chat_save_note,
                         workspace_command=self._chat_workspace_command,
                         list_files=self._chat_list_files,
                         import_sample=self._chat_import_sample,
                         undo_proposal=self._chat_undo_proposal,
                         render=self._chat_render)

    # ---------- Phase 19: the agent's workspace controls ----------

    # Looping the workspace is the user's own control (the loop strip / the
    # transport loop button), never the agent's — so the agent's transport
    # tool can't set or clear a loop. Reject any loop action server-side.
    _TRANSPORT_ACTIONS = {"play", "pause", "stop", "seek"}

    async def _chat_workspace_command(self, command: str, args: dict) -> str:
        """UI commands (transport / focus / solo): broadcast to the clients,
        which drive the shared playback engine. No audio state is touched —
        these are the agent's hands on the same controls the user has."""
        if self.engine is None:
            raise RuntimeError("no project open")
        state = json.loads((self.project_dir / "session.json").read_text())
        if command == "transport":
            action = args.get("action")
            if action not in self._TRANSPORT_ACTIONS:
                raise RuntimeError(f"unknown transport action: {action!r}")
            event = {"type": "workspace_command", "command": "transport",
                     "action": action,
                     "position_s": args.get("position_s")}
            detail = action
        else:  # select_stem / solo_stem
            stem_id = args.get("stem_id")
            if stem_id is not None and stem_id not in state["stems"]:
                raise RuntimeError(f"unknown stem: {stem_id}")
            if command == "select_stem" and stem_id is None:
                raise RuntimeError("select_stem needs a stem_id")
            event = {"type": "workspace_command", "command": command,
                     "stem_id": stem_id}
            detail = stem_id if stem_id is not None else "cleared"
        self.broadcast(event)  # transient UI command — not replayed in feed
        return f"{command}: {detail} — sent to the workspace"

    async def _chat_list_files(self, path: str, query: str = "") -> dict:
        # The DAW's words rule, the one function (`daw_bridge.holds_every_word`): folders and
        # audio alike - this narrowed the audio alone (the pre-merge review of 2026-10-23). Off
        # the event loop, as the DAW's: a listing asks the disk.
        from .daw_bridge import holds_every_word
        listed = await asyncio.to_thread(folder_listing, path)
        if query:
            for key in ("folders", "audio"):
                listed[key] = [n for n in listed[key] if holds_every_word(n, query)]
            listed["audio_count"] = len(listed["audio"])
        return listed

    async def _chat_import_sample(self, path: str) -> dict:
        return await self.import_sample(path)

    async def _chat_undo_proposal(self, proposal_id: str) -> str:
        await self.undo(proposal_id)
        return f"proposal '{proposal_id}' undone; the mix re-rendered"

    async def _chat_render(self) -> str:
        await self.render()
        analysis = self.last_analysis or {}
        return (f"rendered: {analysis.get('lufs_i')} LUFS-I, true peak "
                f"{analysis.get('true_peak_dbtp')} dBTP")

    async def _chat_auto_mix(self, target: str) -> str:
        # The full Auto Mix (bootstrap chain + refinement), gated by the Ask/Agent
        # switch: Ask turns every move into an approve/reject card AND still carries
        # the whole task to completion; Agent runs it hands-off. (Earlier this routed
        # Ask mode to a shallow target-driven run that stopped as soon as the loose
        # targets were met — that's the "doesn't finish the task" bug, now fixed.)
        self.start_auto_mix(target, gating="mode")
        if self.agent_mode == "agent":
            return (f"auto mix started toward {target!r}; staged progress "
                    "streams into the feed, one click undoes all of it")
        return (f"auto mix started toward {target!r} — each move lands as an "
                "approve/reject card (Ask mode); approve them in turn and it runs "
                "the whole chain. Flip to Agent to let it apply hands-off.")

    async def _chat_file_instruction(self, text: str) -> str:
        self.instruct(text)
        return "instruction filed — the loop reads it between passes"

    async def _chat_save_note(self, note: str) -> str:
        """Per-project memory (Phase 17): notes survive history trims and
        gateway restarts — they live next to the run logs."""
        runs_dir = self.project_dir / "runs"
        runs_dir.mkdir(exist_ok=True)
        with (runs_dir / NOTES_FILE).open("a") as f:
            f.write(json.dumps({"ts": round(time.time(), 3),
                                "note": note}) + "\n")
        return f"noted: {note}"

    def _load_notes(self) -> list[str]:
        path = self.project_dir / "runs" / NOTES_FILE
        if not path.is_file():
            return []
        notes = []
        for line in path.read_text().splitlines():
            try:
                notes.append(json.loads(line)["note"])
            except (json.JSONDecodeError, KeyError):
                continue
        return notes[-NOTES_IN_CONTEXT:]

    async def _chat_engine_call(self, name: str, args: dict):
        result = await self.engine.call(name, args)
        if name in {"analyze_reference", "analyze_reference_set"}:
            # The chat agent profiling a reference is the user setting one:
            # keep the workspace reference curve in sync (Phase 10).
            self.reference = result
            await self.broadcast_snapshot()
        return result

    async def _chat_propose(self, req: dict) -> dict:
        """Chat's mutation path: the same engine proposal lifecycle as the
        loop. Returns immediately; the user approves/rejects in the UI."""
        result = await self.engine.call("propose_actions", {"req": req})
        if (not result["accepted"] and result["violations"]
                and all(v["rule"] == "proposals_per_pass"
                        for v in result["violations"])):
            # A stale pass (aborted run, old session) exhausted the per-pass
            # proposal budget; conversational moves start a fresh pass
            # instead of failing on leftovers.
            await self.engine.call("advance_pass")
            result = await self.engine.call("propose_actions", {"req": req})
        if not result["accepted"]:
            return result  # violations go back to the model to revise
        proposal_id = result["proposal_id"]
        event = {"type": "proposal", "pass_number": 0,
                 "proposal_id": proposal_id,
                 "rationale": req.get("rationale", ""),
                 "actions": req.get("actions", []),
                 "predicted_impact": req.get("predicted_impact", [])}
        self.feed.append(event)
        del self.feed[:-FEED_LIMIT]
        self.broadcast(event)
        if self.agent_mode == "agent":
            # Agent mode (Phase 19): the move lands now — still a guardrailed,
            # atomic, undoable proposal; only the approval click is skipped.
            applied = await self._apply_and_render(proposal_id)
            if not applied["ok"]:
                return {"accepted": True, "proposal_id": proposal_id,
                        "status": f"apply failed: {applied.get('message', '')}"}
            analysis = self.last_analysis or {}
            return {"accepted": True, "proposal_id": proposal_id,
                    "status": "applied (agent mode) — undoable",
                    "measured": {
                        "lufs_i": analysis.get("lufs_i"),
                        "true_peak_dbtp": analysis.get("true_peak_dbtp")}}
        self.chat_awaiting.add(proposal_id)
        self.chat_proposals[proposal_id] = req  # Phase 20: for taste recording
        self.broadcast({"type": "awaiting_approval", "proposal_id": proposal_id})
        return {"accepted": True, "proposal_id": proposal_id,
                "status": "submitted — awaiting user approval in the workspace"}

    async def transpose_to_key(self, tonic: str) -> None:
        """Phase 32: the user picks a target tonic in the transport; build a
        deterministic transpose proposal (no LLM, like Auto Mix) that flows
        through the normal approve/undo lifecycle via _chat_propose (Ask gates
        it, Agent auto-applies). Transposition is a mutation, so it is always a
        proposal — never a user-direct edit."""
        if tonic not in PITCH_CLASSES:
            self.broadcast({"type": "error", "message": f"unknown key '{tonic}'"})
            return
        key = (self.last_analysis or {}).get("key")
        if not key:
            self.broadcast({"type": "error", "message": "No confident key "
                            "detected — can't transpose to a key yet."})
            return
        interval = transpose_interval(key["tonic"], tonic)
        if interval == 0:
            self.broadcast({"type": "error",
                            "message": f"Already in {tonic} {key['mode']}."})
            return
        direction = "up" if interval > 0 else "down"
        await self._chat_propose({
            "rationale": (f"Transpose {key['tonic']} {key['mode']} → {tonic} "
                          f"{key['mode']} ({interval:+d} semitones, {direction}) "
                          "— keeps the mode; layer a tune for concert pitch"),
            "predicted_impact": [],
            "actions": [{"type": "transpose", "stem_id": "master",
                         "semitones": interval}],
        })

    async def _chat_start_run(self, brief: str, gating: str,
                              max_passes: int) -> str:
        if self.client_mode == "plugin":
            # The plugin must FINISH the task with the devices in place, not leave
            # an open-ended, target-driven loop streaming (the "it said it'd do it
            # but nothing showed / it never finishes" report). The prompt steers the
            # agent to apply moves directly; if it still reaches for start_run,
            # redirect to the bounded Auto Mix, which builds + applies a full chain
            # and completes (one-click undoable) instead of a run that can stop
            # early at "targets met".
            return await self._chat_auto_mix(brief)
        # "" = follow the Ask/Agent switch (Phase 19; live since the
        # 2026-06-12 fix): hands-off while the switch says Agent, gated
        # while it says Ask — re-checked per proposal, so flipping the
        # switch mid-run changes behavior from the next move on.
        gating = gating or "mode"
        self.start_run(brief, gating, max_passes)
        return (f"run started (brief={brief!r}, gating={gating}, "
                f"max_passes={max_passes}); progress streams into the feed")

    async def _chat_stop_run(self) -> str:
        self.stop_run()
        return "run stop requested"

    def _chat_context(self) -> str:
        """Compact current-state block for the chat agent's system prompt —
        cheap reads only, no renders. Chains carry full parameters (2026-06
        review): the agent can only adjust an existing processor instead of
        stacking a new one if it can see what's already on the stem."""
        state = json.loads((self.project_dir / "session.json").read_text())
        stems = {stem_id: {"gain_db": round(stem["gain_db"], 1),
                           "pan": stem["pan"], "muted": stem["muted"],
                           "excluded": stem["excluded"],
                           # What each stem IS (kick/bass/vocal/…), auto-tagged
                           # at import. Without it the agent had to infer the
                           # source from the filename alone.
                           "tags": stem.get("tags", []),
                           "chain": stem["chain"],
                           "sends": stem.get("sends", {}),
                           "rides": len(stem["rides"])}
                 for stem_id, stem in sorted(state["stems"].items())}
        # Phase 39: in the plugin there are no stem files — audio is captured from
        # the DAW bus when the user plays. An empty project means "not captured
        # yet", not "wrong project / load a folder".
        host_note = None
        if self.client_mode == "plugin":
            cap = round(self.capture_seconds, 1)
            if cap > 0:   # audio is flowing into the plugin right now
                capture_line = (
                    f"Audio is playing through me RIGHT NOW ({cap}s heard so far), "
                    "but a capture only FINALIZES when the transport STOPS. So if "
                    "'stems' is still empty, the user is mid-playback — tell them to "
                    "STOP playback (a few seconds of a loud section is plenty) and "
                    "you'll instantly grab and analyze it. Don't say you heard "
                    "nothing — you can hear it, it just isn't captured until they "
                    "stop. ")
            else:
                capture_line = (
                    "Audio is captured from the bus when the user PLAYS, and the "
                    "capture finalizes when they STOP. If 'stems' is empty, tell "
                    "them to play a few seconds of their song through the plugin and "
                    "then STOP, and you'll capture and analyze it. ")
            host_note = (
                "You are running as a VST/AU plugin on a DAW bus. There are NO "
                "stem files to load and no folder to point at. "
                + capture_line +
                "This project is the correct, intended one — never tell the "
                "user they have the wrong project open or ask them for a stems "
                "folder. "
                "You are a MIXING AND MASTERING engineer with your FULL reach "
                "here — single stems, group busses, and the master, exactly as in "
                "the desktop app; never restrict yourself to master-only moves. "
                "This is a MULTI-INSTANCE plugin: the user can drop an instance on "
                "each DAW track (drums, bass, vocals…) plus one on the master, and "
                "they all share THIS one session and you. Each track instance "
                "captures ITS track as a stem, so the stems you see ARE the user's "
                "tracks — mix them together (balance, carve clashes, duck, glue "
                "busses) and master the sum. Each insert applies its own track's "
                "moves live and the DAW sums them, so your per-track moves ARE heard "
                "live; the master insert masters the blend. If you only see ONE "
                "stem you're on a single track/bus/master — mix or master that one "
                "signal (you can't pull apart elements inside one summed signal; "
                "for that the user adds an instance per track). Read each stem's "
                "analysis + tags and what the user asks, and act like the engineer "
                "for the whole session. "
                "MODE — there is NO Ask/Agent toggle in the plugin: you ALWAYS run "
                "hands-off. Ignore any talk of 'ask' vs 'agent' modes, approval, or "
                "an Ask/Agent switch from your general instructions — none of that "
                "exists here. Never mention modes, switches, approve/reject cards, "
                "or 'waiting for approval'; just make the moves and report them. "
                "FINISH THE JOB — when the user asks you to do something ('compress "
                "this', 'fix the harshness', 'make it loud and punchy'), DO IT and "
                "complete it in this reply: apply the actual moves with `propose` "
                "(they apply immediately here — no approval step), so the devices "
                "land in the chain and the audio changes NOW. Apply the whole chain "
                "the job needs — e.g. dynamic_eq cuts on the resonances, a comp for "
                "glue, a limiter for loudness — then report what you applied and how "
                "the numbers moved. Do NOT just describe a plan, and do NOT kick off "
                "a background multi-pass run/loop and leave it going (`start_run` is "
                "the wrong tool here — it streams and doesn't visibly finish): the "
                "user wants the work done and the devices in place when you reply. "
                "For a whole 'just mix/master it' brief, `auto_mix` is fine (it "
                "builds and applies a full chain); otherwise apply the moves "
                "directly. The live monitor now reflects eq / dynamic_eq / comp / "
                "mb_comp / saturation / de-ess / transient / width / band_width / "
                "limiter — the user hears those moves live; only pitch (tune / "
                "transpose) prints "
                "into the exported render without a live preview. So reach for any "
                "of them freely. "
                "DUCK just works — `duck` keys off the kick (or whatever key track "
                "you name) automatically: I pull that track's level from our shared "
                "session, so the user does NOT route anything in their DAW. Duck "
                "freely (e.g. duck the bass under the kick); the live preview keys "
                "off the captured key and the exported render is exact. "
                "WHOLE-SONG COVERAGE — the duck's key is the CAPTURED kick, so it "
                "only covers the part of the song that was played/captured (this "
                "applies to BOTH the live preview AND a DAW bounce of it — coverage "
                "always follows the capture). If the user wants the duck over the "
                "WHOLE song, tell them to either (a) play the track start-to-finish "
                "once through the plugins so the capture spans the whole song, or "
                "(b) route the kick into this track's plugin sidechain input in their "
                "DAW, which keys the duck live at any position for the entire song "
                "with no capture needed. "
                "You appear in a NARROW plugin panel, so keep replies short and "
                "conversational — a few sentences, not an essay. Write plain "
                "prose: no bold headers, no markdown tables, and no long numbered "
                "step-by-step plans; at most a short dash list when you genuinely "
                "need one. Lead with the verdict and the move, not formatting.")
        return json.dumps({
            "project": self.project_dir.name,
            "host": self.client_mode or "app",
            "host_note": host_note,
            "stems": stems,
            # Phase 23: the routing graph — who sums where, what's racked
            "buses": {bus_id: {"gain_db": round(bus["gain_db"], 1),
                               "chain": bus["chain"],
                               "sends": bus.get("sends", {}),
                               "members": sorted(
                                   s for s, st in state["stems"].items()
                                   if st.get("bus") == bus_id)}
                      for bus_id, bus in sorted(state.get("buses", {}).items())},
            # Phase 38: agent-owned return busses and their wet chains.
            "returns": {return_id: {"gain_db": round(ret.get("gain_db", 0.0), 1),
                                    "chain": ret.get("chain", [])}
                        for return_id, ret in sorted(state.get("returns", {}).items())},
            "master_gain_db": round(state["master_gain_db"], 1),
            "master_chain": state["master_chain"],
            "latest_metrics": self.last_analysis,
            "reference_set": bool(self.reference),
            "run": dict(self.run),
            "agent_mode": self.agent_mode,
            "guardrail_limits": dict(LIMITS),
            # Phase 17: what the user is doing right now — "this part"
            # resolves against the playhead/loop/solo, not a guess.
            "workspace": dict(self.workspace),
            "recent_user_edits": list(self.recent_user_edits),
            "project_notes": self._load_notes(),
            # Phase 20: how this user has received past proposal kinds.
            "user_taste_decisions": taste_summary(self.project_dir / "runs"),
            # Phase 13: lessons past runs concluded for this project (personalization).
            "learned_lessons": load_lessons(self.project_dir / "runs"),
        }, indent=2)

    # ---------- proactive observations (Phase 17) ----------

    def _observation_triggers(self, analysis: dict,
                              flags: list[str]) -> list[dict]:
        """Deterministic thresholds — what is worth speaking up about."""
        triggers = []
        if "master_true_peak_exceeded" in flags:
            triggers.append({"key": "true_peak",
                             "detail": f"the master true peak is "
                                       f"{analysis.get('true_peak_dbtp')} dBTP"
                                       f" — over the -1.0 ceiling"})
        if "mono_compatibility_risk" in flags:
            triggers.append({"key": "mono",
                             "detail": "a band of the master cancels on mono "
                                       "fold-down — analyze_phase says where"})
        if "over_compression" in flags:
            triggers.append({"key": "over_compression",
                             "detail": "a dynamics processor is squashing "
                                       "its source (sustained gain "
                                       "reduction) — check analyze_stem's "
                                       "dynamics readout"})
        if "over_limiting" in flags:
            triggers.append({"key": "over_limiting",
                             "detail": "the limiter is working hard and the "
                                       "master has low PSR/PLR — back off "
                                       "loudness or revise the master chain"})
        if "out_of_tune" in flags:
            triggers.append({"key": "out_of_tune",
                             "detail": f"the song is off concert pitch "
                                       f"({analysis.get('tuning_offset_cents')} "
                                       "cents) or its intonation is smeared — "
                                       "check analyze_stem's tuning readout to "
                                       "find the culprit, then a tune action"})
        last = self._last_observed_analysis or {}
        lufs, last_lufs = analysis.get("lufs_i"), last.get("lufs_i")
        if (lufs is not None and last_lufs is not None
                and abs(lufs - last_lufs) >= OBSERVE_LUFS_SWING_LU):
            triggers.append({"key": "lufs_swing",
                             "detail": f"loudness moved {lufs - last_lufs:+.1f}"
                                       f" LU (now {lufs} LUFS-I)"})
        corr, last_corr = (analysis.get("stereo_correlation"),
                           last.get("stereo_correlation"))
        if (corr is not None and last_corr is not None
                and last_corr >= 0.0 > corr):
            triggers.append({"key": "correlation",
                             "detail": f"stereo correlation flipped negative "
                                       f"({corr})"})
        return triggers

    async def _maybe_observe(self, analysis: dict, flags: list[str]) -> None:
        """At most one short comment, rate-limited, never the same trigger
        twice in a row, never during a run. Always armed — the engineer is
        permanently listening; the thresholds keep it from chattering."""
        triggers = self._observation_triggers(analysis, flags)
        self._last_observed_analysis = analysis
        if self.chat is None or self.run["running"] or not triggers:
            return
        await self._speak_observation(triggers[0])

    async def _speak_observation(self, trigger: dict) -> None:
        """Shared gate + emit for a proactive observation — analysis-driven OR a
        co-edit reaction to a manual move. Never the same trigger twice running,
        rate-limited by OBSERVE_COOLDOWN_S. The cooldown is the hard cost guard:
        at most one observation LLM call per window, so a flurry of edits or
        renders can't run up spend (the runaway-run incident must not recur)."""
        if self.chat is None:
            return
        if trigger["key"] == self._last_observation_key:
            return  # it never repeats itself
        if time.monotonic() - self._last_observation_ts < OBSERVE_COOLDOWN_S:
            return
        # Stamp the ATTEMPT, not the success: observe() answers None when the model
        # says SKIP — the majority path by design — and until 2026-09-01 that path
        # left both stamps untouched, so every fader drag and every render made a
        # full billed turn and the "hard cost guard" never engaged (sweep P1).
        self._last_observation_key = trigger["key"]
        self._last_observation_ts = time.monotonic()
        text = await self.chat.observe(trigger, self._chat_context())
        if not text:
            return
        await self._emit_chat({"type": "chat_message", "role": "assistant",
                               "observation": True, "text": text})

    def _observation_ready(self) -> bool:
        """Cheap gate (no LLM, no state change): chat present, not mid-run, past
        the cooldown, and no observation already in flight. Lets a caller skip
        spawning a task that _speak_observation would just no-op — keeps manual
        fader drags from churning doomed coroutines during the cooldown window."""
        return (self.chat is not None and not self.run["running"]
                and time.monotonic() - self._last_observation_ts >= OBSERVE_COOLDOWN_S
                and (self._observe_task is None or self._observe_task.done()))

    def _user_edit_trigger(self, stem_id: str, changes: dict) -> Optional[dict]:
        """A manual move worth reacting to — a tonal-balance change (fader / pan /
        width). Mute / solo / delete / colour return None so the engineer stays
        quiet about non-tonal toggles. The move is already in _chat_context
        (recent_user_edits); this decides whether it's worth speaking up at all."""
        if stem_id == "master":
            return None  # master loudness/ceiling is the analysis triggers' job
            # (true_peak/over_limiting); the co-edit reaction is about stem balance.
        fields = [f for f in ("gain_db", "pan", "width") if f in changes]
        if not fields:
            return None
        parts = []
        if "gain_db" in changes:
            parts.append(f"set '{stem_id}' to {float(changes['gain_db']):+.1f} dB")
        if "pan" in changes:
            parts.append(f"panned '{stem_id}' to {float(changes['pan']):+.2f}")
        if "width" in changes:
            parts.append(f"set '{stem_id}' width to {float(changes['width']):.2f}")
        return {"key": "user_edit:" + stem_id + ":" + "/".join(fields),
                "detail": "the user just " + ", ".join(parts) + " by hand — "
                          "consider whether the balance now needs a compensating "
                          "move elsewhere (a stem to nudge back, a bus to glue)."}

    def _user_param_trigger(self, device_type: str, value: float,
                            target: str) -> Optional[dict]:
        """A manual device-knob co-edit worth reacting to — the shared-control case:
        the user grabbed a control the agent set. The move is persisted + re-rendered;
        this decides whether to offer a complementary thought."""
        where = "the master" if target == "master" else f"'{target}'"
        return {"key": "user_param:" + target + ":" + device_type,
                "detail": f"the user just adjusted the {device_type} on {where} to "
                          f"{value:g} by hand — consider whether the chain now wants "
                          "a complementary move (you set it up; they're shaping it)."}

    async def _maybe_observe_coedit(self, trigger: dict) -> None:
        """Phase 40 co-edit reaction: speak up (once, gated) after a manual move —
        the shared-control loop where the agent reacts to what the user just did, by
        fader (user_edit) or by a device knob (set_master_param). Shares the
        analysis-observation cooldown via _speak_observation, so a burst of moves
        yields at most one comment per window, never a per-move LLM call. The caller
        builds the trigger (None = nothing worth reacting to) and gates on
        _observation_ready; this re-guards in case a run started since scheduling."""
        if self.chat is None or self.run["running"]:
            return
        await self._speak_observation(trigger)

    # ---------- inbound commands ----------

    async def decide(self, proposal_id: str, approve: bool) -> None:
        """Approve/reject. Run-gated proposals resolve their gate future;
        chat-submitted proposals (no future) apply/skip directly. Either
        way this is a human click — record it for taste (Phase 20)."""
        future = self.pending.get(proposal_id)
        if future is not None and not future.done():
            self._record_taste(approve, self.pending_proposals.get(proposal_id),
                               source="run")
            future.set_result(approve)
            return
        if proposal_id not in self.chat_awaiting:
            # Already resolved — a double-click or a stale UI button (the engine
            # apply/render round-trip takes a beat). Treat a redundant decision as
            # a harmless no-op instead of raising: surfacing it as an error chip
            # for an ordinary double-tap is worse than silently ignoring it. The
            # UI's busy-latch is the first line of defence; this is belt-and-braces.
            return
        self.chat_awaiting.discard(proposal_id)
        self._record_taste(approve, self.chat_proposals.pop(proposal_id, None),
                           source="chat")
        if not approve:
            self.broadcast({"type": "proposal_status", "proposal_id": proposal_id,
                            "status": "skipped", "message": ""})
            return
        await self._apply_and_render(proposal_id)

    def _record_taste(self, approve: bool, proposal: Optional[dict],
                      source: str) -> None:
        """One human gate decision into runs/decisions.jsonl (Phase 20)."""
        if proposal is None or self.project_dir is None:
            return
        record_decision(self.project_dir / "runs",
                        "approved" if approve else "rejected", proposal,
                        source=source)

    async def _apply_and_render(self, proposal_id: str) -> dict:
        """Apply one chat proposal and re-render ground truth (shared by the
        approve path and agent-mode auto-apply). Advances the pass so per-pass
        guardrail budgets don't accumulate across separate conversational
        asks."""
        applied = await self.engine.call("apply_proposal",
                                         {"proposal_id": proposal_id})
        self.broadcast({"type": "proposal_status", "proposal_id": proposal_id,
                        "status": "applied" if applied["ok"] else "skipped",
                        "message": applied.get("message", "")})
        render = await self.engine.render_and_analyze()
        await self._emit({"type": "metrics", "analysis": render["analysis"],
                          "flags": render.get("flags", []),
                          "master": render.get("master"),
                          "bounce_path": render.get("bounce_path")})
        if applied["ok"]:
            await self.engine.call("advance_pass")
        return applied

    async def undo(self, proposal_id: str) -> None:
        if self.engine is None:
            raise RuntimeError("no project open")
        result = await self.engine.call("undo_proposal", {"proposal_id": proposal_id})
        if not result["ok"]:
            raise RuntimeError(result["message"])
        # Phase 20: an undo is the strongest taste signal — the user let the
        # move through, heard it, and took it back.
        for summary in await self.engine.call("get_proposals",
                                              {"status": "undone"}):
            if summary["proposal_id"] == proposal_id:
                record_decision(self.project_dir / "runs", "undone", summary,
                                source="workspace")
                break
        # Confirmation first, then the re-rendered ground truth (metrics +
        # snapshot follow from _emit).
        self.broadcast({"type": "proposal_status", "proposal_id": proposal_id,
                        "status": "undone", "message": ""})
        render = await self.engine.render_and_analyze()
        await self._emit({"type": "metrics", "analysis": render["analysis"],
                          "flags": render.get("flags", []),
                          "master": render.get("master"),
                          "bounce_path": render.get("bounce_path")})

    async def render(self) -> None:
        """User-initiated render (desktop Render menu item — docs/07). Read-only
        with respect to mix state: just re-bounces and re-measures."""
        if self.engine is None:
            raise RuntimeError("no project open")
        result = await self.engine.render_and_analyze()
        await self._emit({"type": "metrics", "analysis": result["analysis"],
                          "flags": result.get("flags", []),
                          "master": result.get("master"),
                          "bounce_path": result.get("bounce_path")})
        # Phase 17: the resident engineer may speak up about what the
        # meters just did (thresholded, rate-limited, off during runs).
        # Fired in the background: an observation is an LLM round-trip, and
        # it must never hold up the render pipeline or the command loop. One
        # at a time — if the last one is still thinking, this moment passes.
        if self._observe_task is None or self._observe_task.done():
            self._observe_task = asyncio.create_task(
                self._observe_quietly(result["analysis"],
                                      result.get("flags", [])))

    async def _observe_quietly(self, analysis: dict, flags: list[str]) -> None:
        try:
            await self._maybe_observe(analysis, flags)
        except Exception:
            pass  # advisory only — never surfaces as a workspace error

    async def user_edit(self, stem_id: str, changes: dict) -> None:
        """The user's direct DAW channel (Phase 9): fader/pan/mute/delete —
        immediate, no proposals, no guardrails (CLAUDE.md). Re-renders so the
        bounce and metrics always reflect what the user hears. `color` is a
        UI-only field handled here (ui.json), never sent to the engine."""
        if self.engine is None:
            raise RuntimeError("no project open")
        changes = dict(changes)
        has_color = "color" in changes
        if has_color:
            self._set_stem_color(stem_id, changes.pop("color"))
        if changes:
            await self.engine.call("set_stem", {"stem_id": stem_id, **changes})
            self.recent_user_edits.append(
                {"stem_id": stem_id, "changes": dict(changes),
                 "ts": round(time.time(), 1)})
            del self.recent_user_edits[:-RECENT_EDITS_KEPT]
            # instant confirmation (chains/gains from live state), then the
            # coalesced background render refreshes bounce + metrics; the
            # render is requested first so the snapshot says render_pending
            self.request_render()
            await self.broadcast_snapshot(stale_ok=True)
            # Phase 40 co-edit reaction: the engineer may comment on the manual
            # move (gated by the observation cooldown — at most one per window).
            # Fire-and-forget on the shared observe-task slot so the edit stays
            # instant and no second observation runs concurrently.
            trigger = self._user_edit_trigger(stem_id, dict(changes))
            if trigger is not None and self._observation_ready():
                self._observe_task = asyncio.create_task(
                    self._maybe_observe_coedit(trigger))
        elif has_color:
            await self.broadcast_snapshot()  # cosmetic only — no re-render


    async def set_master_param(self, device_type: str, occurrence: int,
                               value: float, target: str = "master") -> dict:
        """Phase 40 (native VST co-edit): the user's direct edit of a chain device
        control — the shared-control channel. `target` = master (default) or a
        stem/bus id (a per-track insert edits its own stem). Immediate, no proposal;
        bumps the chain rev so the plugin's monitor re-reads the persisted value, and
        coalesce-renders so the export + the agent's next analysis reflect it."""
        if self.engine is None:
            raise RuntimeError("no project open")
        result = await self.engine.call(
            "set_master_param",
            {"device_type": device_type, "occurrence": occurrence,
             "value": value, "target": target})
        self.recent_device = (device_type, occurrence)   # tint this knob (recency)
        self.bounce_rev += 1            # chain_rev: prompt the monitor's re-read
        self.request_render()           # export + analysis (coalesced)
        await self.broadcast_snapshot(stale_ok=True)
        # Phase 40 co-edit reaction: react to a manual device-knob move (shared
        # control), gated by the same observation cooldown as fader moves.
        trigger = self._user_param_trigger(device_type, value, target)
        if trigger is not None and self._observation_ready():
            self._observe_task = asyncio.create_task(
                self._maybe_observe_coedit(trigger))
        return result

    async def add_master_device(self, device_type: str) -> dict:
        """Phase 40 (native VST co-edit): the user adds a device to the master
        chain from the native palette — immediate, no proposal (the user's hands).
        Bumps the chain rev so the monitor re-reads, and renders so the export +
        the agent's next analysis reflect it."""
        if self.engine is None:
            raise RuntimeError("no project open")
        result = await self.engine.call("add_master_device",
                                        {"device_type": device_type})
        self.recent_device = (device_type, result.get("occurrence", 0))
        self.bounce_rev += 1
        self.request_render()
        await self.broadcast_snapshot(stale_ok=True)
        return result

    async def remove_master_device(self, device_type: str, occurrence: int) -> dict:
        """Phase 40 (native VST co-edit): the user removes a master device (the ×
        on a native device tile)."""
        if self.engine is None:
            raise RuntimeError("no project open")
        result = await self.engine.call(
            "remove_master_device",
            {"device_type": device_type, "occurrence": occurrence})
        self.bounce_rev += 1
        self.request_render()
        await self.broadcast_snapshot(stale_ok=True)
        return result

    async def set_master_device_order(self, order: list[str]) -> dict:
        """Phase 42 (native VST co-edit): the user drag-reorders the master device
        chain. `order` is the device-type modules left-to-right; the limiter always
        renders last (ceiling). User channel — no proposal."""
        if self.engine is None:
            raise RuntimeError("no project open")
        result = await self.engine.call("set_master_device_order", {"order": order})
        self.bounce_rev += 1
        self.request_render()
        await self.broadcast_snapshot(stale_ok=True)
        return result

    async def add_stem_device(self, target: str, device_type: str) -> dict:
        """Phase 40 per-track co-edit: add a stem-only COLOR device (transient /
        de-ess / band_width) to a captured track. User channel — no proposal."""
        if self.engine is None:
            raise RuntimeError("no project open")
        result = await self.engine.call(
            "add_stem_device", {"target": target, "device_type": device_type})
        self.recent_device = (device_type, result.get("occurrence", 0))
        self.bounce_rev += 1
        self.request_render()
        await self.broadcast_snapshot(stale_ok=True)
        return result

    async def remove_stem_device(self, target: str, device_type: str,
                                 occurrence: int) -> dict:
        """Phase 40 per-track co-edit: remove a stem color device (the × on a tile)."""
        if self.engine is None:
            raise RuntimeError("no project open")
        result = await self.engine.call(
            "remove_stem_device",
            {"target": target, "device_type": device_type, "occurrence": occurrence})
        self.bounce_rev += 1
        self.request_render()
        await self.broadcast_snapshot(stale_ok=True)
        return result

    async def set_master_device_param(self, device_type: str, occurrence: int,
                                      param: str, value: float,
                                      target: str = "master") -> dict:
        """Phase 40 (native VST co-edit): the device detail popup sets any param on
        a device — immediate, no proposal. `target` = master or a stem/bus id."""
        if self.engine is None:
            raise RuntimeError("no project open")
        result = await self.engine.call(
            "set_master_device_param",
            {"device_type": device_type, "occurrence": occurrence,
             "param": param, "value": value, "target": target})
        self.recent_device = (device_type, occurrence)
        self.bounce_rev += 1
        self.request_render()
        await self.broadcast_snapshot(stale_ok=True)
        return result

    async def set_eq_shape(self, occurrence: int, shape: str,
                           target: str = "master") -> dict:
        """Phase 40 (native VST co-edit): the detail popup changes an EQ band's
        filter shape — immediate, no proposal. `target` = master or a stem/bus id."""
        if self.engine is None:
            raise RuntimeError("no project open")
        await self.engine.call("set_eq_shape",
                               {"occurrence": occurrence, "shape": shape,
                                "target": target})
        self.recent_device = ("eq", occurrence)
        self.bounce_rev += 1
        self.request_render()
        await self.broadcast_snapshot(stale_ok=True)
        return {"ok": True, "shape": shape}

    async def set_device_band(self, occurrence: int, band: str,
                              target: str = "master",
                              lo_hz: float | None = None,
                              hi_hz: float | None = None) -> dict:
        """Band-width band selector (2026-07-05): the user picks the device's band
        LABEL — immediate, no proposal. `target` = master or a stem/bus id.
        2026-07-19: explicit lo_hz/hi_hz place the band freely (override the label)."""
        if self.engine is None:
            raise RuntimeError("no project open")
        await self.engine.call("set_device_band",
                               {"occurrence": occurrence, "band": band,
                                "lo_hz": lo_hz, "hi_hz": hi_hz,
                                "target": target})
        self.recent_device = ("band_width", occurrence)
        self.bounce_rev += 1
        self.request_render()
        await self.broadcast_snapshot(stale_ok=True)
        return {"ok": True, "band": band}

    async def set_device_channel(self, device_type: str, occurrence: int,
                                 channel: str, target: str = "master") -> dict:
        """Phase 41 (native VST co-edit): the eq/comp M/S selector (stereo/mid/side)
        — immediate, no proposal. `target` = master or a stem/bus id."""
        if self.engine is None:
            raise RuntimeError("no project open")
        await self.engine.call("set_device_channel",
                               {"device_type": device_type, "occurrence": occurrence,
                                "channel": channel, "target": target})
        self.recent_device = (device_type, occurrence)
        self.bounce_rev += 1
        self.request_render()
        await self.broadcast_snapshot(stale_ok=True)
        return {"ok": True, "channel": channel}

    async def set_sat_multiband(self, occurrence: int, on: bool,
                                target: str = "master") -> dict:
        """Phase 45 (native VST co-edit): the saturation single ↔ 3-band MB toggle —
        seeds/clears low/mid/high. `target` = master or a stem/bus id."""
        if self.engine is None:
            raise RuntimeError("no project open")
        await self.engine.call("set_sat_multiband",
                               {"occurrence": occurrence, "on": on, "target": target})
        self.recent_device = ("saturation", occurrence)
        self.bounce_rev += 1
        self.request_render()
        await self.broadcast_snapshot(stale_ok=True)
        return {"ok": True, "multiband": on}

    async def set_master_device_enabled(self, device_type: str, occurrence: int,
                                        enabled: bool, target: str = "master") -> dict:
        """Phase 40 (native VST co-edit): the tile on/off bypasses a device in both
        the live monitor and the export — immediate, no proposal. `target` = master
        or a stem/bus id."""
        if self.engine is None:
            raise RuntimeError("no project open")
        await self.engine.call(
            "set_master_device_enabled",
            {"device_type": device_type, "occurrence": occurrence,
             "enabled": enabled, "target": target})
        self.recent_device = (device_type, occurrence)
        self.bounce_rev += 1
        self.request_render()
        await self.broadcast_snapshot(stale_ok=True)
        return {"ok": True, "enabled": enabled}

    def _set_stem_color(self, stem_id: str, color: str | None) -> None:
        """Per-stem display color, chosen by the user (Ableton-style). Lives
        in ui.json next to session.json — project-portable (Save As copies
        it) but never part of engine state: the agent reads numbers, and
        colors are not numbers it should ever see. None resets to automatic."""
        state = json.loads((self.project_dir / "session.json").read_text())
        if stem_id not in state["stems"]:
            raise RuntimeError(f"unknown stem: {stem_id}")
        if color is not None and not re.fullmatch(r"#[0-9a-fA-F]{6}", str(color)):
            raise RuntimeError("color must be '#rrggbb' or null")
        colors = self.ui_state.setdefault("stem_colors", {})
        if color is None:
            colors.pop(stem_id, None)
        else:
            colors[stem_id] = color
        (self.project_dir / "ui.json").write_text(
            json.dumps(self.ui_state, indent=2))

    async def set_reference(self, path: str) -> None:
        if self.engine is None:
            raise RuntimeError("no project open")
        self.reference = await self.engine.call("analyze_reference", {"path": path})
        await self.broadcast_snapshot()

    def set_reference_scope(self, scope: str) -> None:
        """Phase 25: what reference runs compare against — the reference
        card's Mix / Master / Both control."""
        if scope not in ("mix", "master", "both"):
            raise ValueError(f"unknown reference scope: {scope!r} "
                             "(mix, master, both)")
        self.reference_scope = scope

    async def _import_capture(self, source: Path, target: Path,
                              stem_id: str) -> dict:
        """Land one plugin instance's per-track capture (Phase 40, per-track
        instances). Overwrites this track's wav and replaces ONLY its own stem;
        other instances' stems are untouched, so N inserts = N coexisting stems."""
        sess = self.project_dir / "session.json"
        existed = (json.loads(sess.read_text())["stems"].get(stem_id)
                   if sess.exists() else None)
        _copyfile_atomic(source, target)             # (over)write this track's wav — rename, never truncate
        await self.engine.call("refresh_stems")      # register a new track / re-read content
        if existed is not None and existed.get("excluded"):
            await self.engine.call("set_stem",
                                   {"stem_id": stem_id, "excluded": False})
        if existed is None:                          # a freshly seen track: tag it
            await self.engine.call("auto_tag_stems", {"stem_ids": [stem_id]})
        await self.render()                          # re-analyze with the new audio
        # the Ozone AI button armed before a capture existed → fire it now
        if self.auto_mix_armed is not None and not (
                self.run_task and not self.run_task.done()):
            brief, self.auto_mix_armed = self.auto_mix_armed, None
            self.start_auto_mix(brief)
        return {"ok": True, "stem_id": stem_id, "recaptured": existed is not None}

    async def import_sample(self, path: str, capture: bool = False) -> dict:
        """Copy a local audio file into the project and pick it up in place —
        shared by the sample browser's POST /api/import and the chat agent's
        import_sample tool (Phase 19). Wav copies byte-identical; mp3/flac/
        aiff/ogg decode to a wav copy at this boundary (Phase 26).
        ValueError = bad input (HTTP 400), RuntimeError = conflict (409).

        `capture=True` is the plugin's per-track path (Phase 40, per-track
        instances): the wav is named by the instance's track, lands as a
        stably-named stem, and a re-capture REPLACES that one stem in place —
        other instances' stems are left untouched, so N inserts coexist as N
        stems in the one shared session (the single master-bus insert is N=1)."""
        if self.project_dir is None:
            raise RuntimeError("no project open")
        if self.run_task and not self.run_task.done():
            raise RuntimeError("cannot import stems while a run is active")
        source = Path(path).expanduser().resolve()
        if not source.is_file() or source.suffix.lower() not in AUDIO_EXTS:
            raise ValueError(
                f"not a supported audio file ({AUDIO_EXTS_LABEL}): {source}")
        is_wav = source.suffix.lower() == ".wav"
        if source.parent == self.project_dir and is_wav:
            raise ValueError("that file is already a project stem")
        target = self.project_dir / f"{source.stem}.wav"
        stem_id = source.stem
        if capture:
            return await self._import_capture(source, target, stem_id)
        # Phase 39: a plugin capture IS the current bus — re-capturing replaces it
        # rather than stacking, so the audition is always the latest play and not a
        # pile of old ones summed together. Capture stems are named capture-<ts>;
        # exclude the prior ones (file stays, undoable) before the new one lands.
        if re.fullmatch(r"capture-\d+", stem_id):
            prior = json.loads((self.project_dir / "session.json").read_text())
            for sid, st in list(prior["stems"].items()):
                if (sid != stem_id and re.fullmatch(r"capture-\d+", sid)
                        and not st["excluded"]):
                    await self.engine.call("set_stem",
                                           {"stem_id": sid, "excluded": True})
        if target.exists():
            state = json.loads((self.project_dir / "session.json").read_text())
            stem = state["stems"].get(stem_id)
            if stem is not None and not stem["excluded"]:
                raise RuntimeError(
                    f"a stem named '{target.name}' is already in the mix")
            if stem is not None:
                # The stem was removed with ✕ (excluded, hidden everywhere)
                # but its file stays in the project — re-importing it is the
                # user asking for it back, not a name collision.
                await self.engine.call("set_stem",
                                       {"stem_id": stem_id, "excluded": False})
                await self.render()
                return {"ok": True, "stem_id": stem_id, "restored": True}
            # orphan wav (an earlier import failed between copy and
            # registration): fall through and register it in place
        elif is_wav:
            _copyfile_atomic(source, target)
        else:
            await asyncio.to_thread(transcode_to_wav, source, target)
        await self.engine.call("refresh_stems")
        # Tag only the new arrival: unclassifiable older stems would
        # otherwise re-render + re-analyze on every single import.
        await self.engine.call("auto_tag_stems", {"stem_ids": [stem_id]})
        await self.render()  # bounce + metrics include the new stem
        # Phase 40 (native VST): the Ozone AI button was pressed before a capture
        # existed ("press, then play"). The capture just landed → fire Auto Mix now.
        if self.auto_mix_armed is not None and not (
                self.run_task and not self.run_task.done()):
            brief, self.auto_mix_armed = self.auto_mix_armed, None
            self.start_auto_mix(brief)
        return {"ok": True, "stem_id": stem_id}

    def instruct(self, text: str) -> None:
        self.instructions.append(text)
        event = {"type": "user_instruction", "text": text}
        self.feed.append(event)
        self.broadcast(event)

    def set_agent_mode(self, mode: str) -> None:
        """Ask/Agent switch (Phase 19). Applies to the next chat move and —
        2026-06-12 fix — to the next proposal of a mode-following run; an
        already-awaiting proposal keeps waiting for its click."""
        if mode not in ("ask", "agent"):
            raise RuntimeError(f"unknown agent mode: {mode!r}")
        self.agent_mode = mode
        # Instant confirmation (2026-06-12 fix): the full snapshot that
        # follows can sit behind the engine lock for seconds during a run —
        # the switch must flip in the UI the moment it is clicked.
        self.broadcast({"type": "agent_mode", "mode": mode})

    def set_chat_settings(self, model: str, reasoning: str) -> None:
        """The composer's model + reasoning pickers. Validates against the
        registry in chat.py; applies mid-conversation (next turn)."""
        if self.chat is None:
            raise RuntimeError("no project open")
        self.chat.set_settings(model, reasoning)  # raises on unknown values
        self.chat_settings = {"model": model, "reasoning": reasoning}


def _default_agent_factory() -> Agent:
    from .agents import ClaudeAgent
    return ClaudeAgent()


def wav_peaks(path: Path, buckets: int) -> dict:
    """Per-bucket min/max of a wav, normalized to its own absolute peak so
    quiet files read as well as loud ones. The UI draws these on a canvas
    and never decodes audio itself — WebKit (the desktop webview) can't
    decode every wav (float32/24-bit PCM); peaks render identically
    everywhere. Shared by the clip canvases and the sample preview."""
    import numpy as np
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    edges = np.linspace(0, len(mono), buckets + 1, dtype=int)
    peak = float(np.max(np.abs(mono))) or 1.0
    top, bottom = [], []
    for i in range(buckets):
        chunk = mono[edges[i]:edges[i + 1]]
        if chunk.size == 0:
            top.append(0.0)
            bottom.append(0.0)
            continue
        top.append(round(max(float(chunk.max()), 0.0) / peak, 3))
        bottom.append(round(min(float(chunk.min()), 0.0) / peak, 3))
    return {"duration_s": round(len(mono) / sample_rate, 3),
            "top": top, "bottom": bottom}


def folder_listing(path: str) -> dict:
    """Folders + audio files (AUDIO_EXTS) under `path` — the sample
    browser's view. Shared by GET /api/fs and the chat agent's list_files
    tool (Phase 19). Raises NotADirectoryError for bad paths."""
    target = Path(path).expanduser().resolve()
    if not target.is_dir():
        raise NotADirectoryError(f"not a directory: {target}")
    folders = sorted(d.name for d in target.iterdir()
                     if d.is_dir() and not d.name.startswith("."))
    audio = sorted(f.name for f in target.iterdir()
                   if f.is_file() and not f.name.startswith(".")
                   and f.suffix.lower() in AUDIO_EXTS)
    return {"path": str(target),
            "parent": str(target.parent),
            "folders": folders,
            "audio": audio,
            "audio_count": len(audio)}


# ---------- app ----------

def create_app(project_dir: Optional[str | Path] = None,
               agent_factory: Optional[Callable[[], Agent]] = None,
               chat_factory: Optional[Callable[[], Any]] = None) -> FastAPI:
    from contextlib import asynccontextmanager

    gateway = Gateway(agent_factory, chat_factory)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if project_dir is not None:
            try:
                await gateway.open_project(project_dir)
            except Exception as exc:
                # A bad startup folder (deleted, stems missing) must not
                # crash-loop the sidecar — start with no project and let the
                # UI fall back to the picker.
                print(f"could not open startup project {project_dir}: {exc}",
                      file=sys.stderr)
        yield
        await gateway.close_project()

    app = FastAPI(title="mix-agent gateway", lifespan=lifespan)
    app.state.gateway = gateway

    @app.middleware("http")
    async def refuse_browser_origins(request, call_next):
        # Phase 49 removed CORS — but CORS only stops a page READING a
        # cross-origin response. The browser still SENDS the request, and
        # every mutating route here takes its arguments from the query string,
        # so a `fetch(..., {mode: "no-cors"})` from any web page executed
        # /api/chat, /api/export and /api/import with no preflight (2026-09-01
        # sweep, P0). A browser attaches Origin to every cross-origin request and
        # every POST; the native clients (the VST, the tests) never do — so the
        # header is refused on every method. It is NOT the whole test: a
        # same-origin GET carries none, and a DNS-rebinding page is same-origin
        # in the browser's eyes — `addressed_to_this_machine` is the other half.
        # The WebSocket route applies both rules below.
        if "origin" in request.headers:
            return JSONResponse({"ok": False, "error": "browser origins may not "
                                 "script the local engine"}, status_code=403)
        if not addressed_to_this_machine(request.headers):
            return JSONResponse({"ok": False, "error": "this engine answers only requests "
                                 "addressed to this machine"}, status_code=403)
        return await call_next(request)

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True,
                "project": str(gateway.project_dir) if gateway.project_dir else None}

    @app.get("/api/fs")
    def list_folders(path: str = "~") -> dict:
        """Server-side folder browser for the project picker and the sample
        browser (no upload flow: projects and samples are local files —
        CLAUDE.md). `wavs` lists the audio files in the folder (Phase 18)."""
        try:
            return folder_listing(path)
        except NotADirectoryError as exc:
            raise HTTPException(404, str(exc))

    @app.get("/api/audio/stem/{stem_id}")
    def stem_audio(stem_id: str, processed: bool = False, rev: str = "0"):
        """Raw stem file, or (processed=1) the stem rendered through its
        applied chain — what Solo plays. Sync endpoint: FastAPI runs it in a
        worker thread, and the read-only Session never writes state."""
        if gateway.project_dir is None:
            raise HTTPException(409, "no project open")
        state = json.loads((gateway.project_dir / "session.json").read_text())
        if stem_id not in state["stems"]:
            raise HTTPException(404, f"unknown stem: {stem_id}")
        if not processed:
            return FileResponse(gateway.project_dir / state["stems"][stem_id]["path"],
                                media_type="audio/wav")
        import io

        import numpy as np
        from scipy.io import wavfile

        from mix_engine import dsp
        from mix_engine.session import Session as EngineSession

        session = EngineSession(gateway.project_dir, read_only=True)
        audio, sample_rate = dsp.render_stem(session, stem_id)
        buffer = io.BytesIO()
        wavfile.write(buffer, sample_rate, audio.astype(np.float32))
        return Response(content=buffer.getvalue(), media_type="audio/wav",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/duck_key")
    def duck_key_envelope(stem: str) -> dict:
        """Phase 41 (agent-owned duck key): the block-level envelope of a key stem,
        so a ducked plugin instance keys off it WITHOUT a host sidechain route. The
        key is rendered DUCK-FREE — the exact signal dsp._duck uses for the export —
        and returned as its 5 ms-block RMS in dB, aligned to the host timeline via
        capture_start_s. A direct localhost poll; the plugin fetches it only when the
        key/rev changes, then indexes it at the playhead. Live preview keys off the
        captured key (a snapshot); the exported render stays the exact live duck."""
        if gateway.project_dir is None:
            raise HTTPException(409, "no project open")
        state = json.loads((gateway.project_dir / "session.json").read_text())
        if stem not in state.get("stems", {}):
            raise HTTPException(404, f"unknown stem: {stem}")
        import numpy as np

        from mix_engine import dsp
        from mix_engine.session import Session as EngineSession

        session = EngineSession(gateway.project_dir, read_only=True)
        audio, sample_rate = dsp.render_stem(session, stem, exclude_ducks=True)
        # The block the EXPORT measures this key at is _env_block_s(attack_ms) of the duck
        # keyed off it (below 5 ms attack the block halves down to 0.5 ms); a fixed 5 ms
        # envelope here gave the plugin a coarser, differently-timed signal than the bounce
        # (2026-09-01 sweep). Use the finest block any duck keyed off this stem needs.
        block_s = dsp.ENV_BLOCK_S
        for holder in list(state.get("stems", {}).values()) + list(state.get("buses", {}).values()):
            for entry in holder.get("chain", []):
                if entry.get("type") == "duck" and entry.get("key_stem_id") == stem:
                    block_s = min(block_s, dsp._env_block_s(float(entry.get("attack_ms", 10.0))))
        env_db = dsp._block_env_db(np.asarray(audio), sample_rate, block_s)
        # Phase 41: align to the KEY stem's OWN capture offset (set at import),
        # not the single last-capture value — so separately-captured tracks line up.
        # Fall back to the shared offset for stems imported before per-stem offsets.
        key_offset = state["stems"][stem].get("capture_start_s")
        if key_offset is None:
            key_offset = gateway.capture_start_s
        return {"block_s": block_s,
                "sample_rate": sample_rate,
                "capture_start_s": float(key_offset),
                "key_rev": gateway.bounce_rev,
                "env_db": [round(float(v), 1) for v in env_db]}

    @app.get("/api/audio/peaks/{stem_id}")
    def stem_peaks(stem_id: str, buckets: int = 1200) -> dict:
        """Waveform peaks for the clip canvases (see Gateway.stem_peaks)."""
        if gateway.project_dir is None:
            raise HTTPException(409, "no project open")
        state = json.loads((gateway.project_dir / "session.json").read_text())
        if stem_id not in state["stems"]:
            raise HTTPException(404, f"unknown stem: {stem_id}")
        path = gateway.project_dir / state["stems"][stem_id]["path"]
        return gateway.stem_peaks(stem_id, path, max(100, min(8000, buckets)))

    def _preview_target(path: str) -> Path:
        """Validate a sample-preview path (Phase 21): local audio files only —
        the same trust model as /api/fs and /api/import (localhost gateway,
        webview-restricted CORS, the user's own file browser)."""
        target = Path(path).expanduser()
        if target.suffix.lower() not in AUDIO_EXTS:
            raise HTTPException(400,
                                f"previews are audio files ({AUDIO_EXTS_LABEL})")
        if not target.is_file():
            raise HTTPException(404, f"no such file: {target}")
        return target

    @app.get("/api/preview/audio")
    def preview_audio(path: str) -> Response:
        """Sample-browser preview audio (Phase 21): the file re-encoded as
        float32 PCM wav so the desktop webview can decode it regardless of
        the source bit depth (24-bit PCM doesn't decode in WebKit) — which
        since Phase 26 doubles as the decode step for mp3/flac/aiff/ogg."""
        import io

        import numpy as np
        from scipy.io import wavfile

        target = _preview_target(path)
        data, sample_rate = sf.read(str(target), dtype="float32",
                                    always_2d=True)
        buffer = io.BytesIO()
        wavfile.write(buffer, sample_rate, data.astype(np.float32))
        return Response(content=buffer.getvalue(), media_type="audio/wav",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/preview/peaks")
    def preview_peaks(path: str, buckets: int = 600) -> dict:
        """Waveform peaks for the sample-preview dock (Phase 21)."""
        return gateway.preview_peaks(_preview_target(path),
                                     max(100, min(8000, buckets)))

    @app.get("/api/audio/bounce")
    def bounce_audio(rev: int = 0, prev: bool = False) -> FileResponse:
        if gateway.project_dir is None:
            raise HTTPException(409, "no project open")
        name = "master.prev.wav" if prev else "master.wav"
        bounce = gateway.project_dir / "bounces" / name
        if not bounce.exists():
            raise HTTPException(404, "no bounce rendered yet")
        return FileResponse(bounce, media_type="audio/wav",
                            headers={"Cache-Control": "no-store"})

    @app.post("/api/import")
    async def import_stem_path(path: str, capture_start_s: float = -1.0,
                               capture: bool = False) -> dict:
        """Sample-browser import (Phase 18): copy a local wav into the
        project and pick it up in place — the server-side twin of the
        drag-and-drop PUT below, for files the browser panel lists. Shares
        Gateway.import_sample with the chat agent's tool (Phase 19).

        Phase 39 (VST): the plugin also posts here to land a bus capture, passing
        capture_start_s = the host-timeline offset of the capture's first frame so
        the audition can align the rendered bounce. The desktop omits it (-1).
        Phase 40 (per-track instances): `capture=True` lands the wav as the
        instance's own stably-named track stem (re-capture replaces it in place,
        other instances untouched)."""
        if capture_start_s >= 0:
            gateway.capture_start_s = capture_start_s
        try:
            result = await gateway.import_sample(path, capture=capture)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        # Phase 41: stamp THIS capture's timeline offset on its own stem, so a
        # duck keyed by it aligns even if other tracks were captured separately.
        if capture_start_s >= 0 and result.get("stem_id"):
            await gateway.engine.call("set_stem_capture_start",
                                      {"stem_id": result["stem_id"],
                                       "capture_start_s": capture_start_s})
        return result

    @app.post("/api/export")
    async def export_mix(dir: str, format: str = "wav24",
                         include_stems: bool = False) -> dict:
        """Phase 24: print deliverables — the mixed/mastered wav (and
        optionally the processed stems) into a local folder the user picked
        in the transport's Export dialog. Same trust model as /api/fs and
        /api/import: localhost gateway, webview-restricted CORS, the user's
        own disk. Renders serialize on the engine lock, so exporting during
        a run just waits its turn."""
        if gateway.project_dir is None:
            raise HTTPException(409, "no project open")
        try:
            return await gateway.engine.call("export_mix", {"req": {
                "out_dir": dir, "format": format,
                "include_stems": include_stems}})
        except RuntimeError as exc:
            # engine-side rejections (folder inside the project, bad format)
            raise HTTPException(400, str(exc))

    @app.put("/api/stems/{filename}")
    async def add_stem(filename: str, request: Request) -> dict:
        """Drag-and-drop import (Phase 9; in-place since Phase 10): write the
        dropped audio into the project folder (non-wav decodes to a wav copy,
        Phase 26) and re-run stem discovery via `refresh_stems` — no project
        reopen, so the chat conversation and feed survive. Local files only —
        this is the desktop drop affordance, not a hosted upload."""
        if gateway.project_dir is None:
            raise HTTPException(409, "no project open")
        if gateway.run_task and not gateway.run_task.done():
            # mid-run stem changes would shift the loop's ground truth
            raise HTTPException(409, "cannot import stems while a run is active")
        safe = Path(filename).name
        ext = Path(safe).suffix.lower()
        if ext not in AUDIO_EXTS:
            raise HTTPException(400,
                                f"stems are audio files ({AUDIO_EXTS_LABEL})")
        data = await request.body()
        if not data.startswith(AUDIO_MAGIC[ext]):
            raise HTTPException(400, f"not a {ext.lstrip('.')} file")
        stem_id = Path(safe).stem
        if ext == ".wav":
            (gateway.project_dir / safe).write_bytes(data)
        else:
            # the bytes land as a dotfile (invisible to *.wav discovery and
            # /api/fs) so libsndfile can sniff the container from disk
            tmp = gateway.project_dir / f".import-{safe}"
            tmp.write_bytes(data)
            try:
                await asyncio.to_thread(transcode_to_wav, tmp,
                                        gateway.project_dir / f"{stem_id}.wav")
            except (RuntimeError, sf.LibsndfileError) as exc:
                raise HTTPException(400, f"could not decode {safe}: {exc}")
            finally:
                tmp.unlink(missing_ok=True)
        await gateway.engine.call("refresh_stems")
        state = json.loads((gateway.project_dir / "session.json").read_text())
        if state["stems"].get(stem_id, {}).get("excluded"):
            # dropping a wav means "into the mix" — clear a leftover ✕
            # removal instead of importing into a hidden stem
            await gateway.engine.call("set_stem",
                                      {"stem_id": stem_id, "excluded": False})
        # new stem gets its class; only the new one — older unclassifiable
        # stems must not re-analyze on every import
        await gateway.engine.call("auto_tag_stems", {"stem_ids": [stem_id]})
        await gateway.render()  # bounce + metrics include the new stem; snapshot follows
        return {"ok": True, "stem_id": stem_id}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        # CORS never governed WebSockets at all: any page could open
        # ws://127.0.0.1:<port>/ws, read every frame and send open_project /
        # start_run / chat (2026-09-01 sweep, P0). Same rule as the HTTP
        # middleware — an Origin header means a browser, and a Host that is
        # not this machine's means a rebinding page (S9·R16). Closing before
        # accept() answers the upgrade with 403.
        if "origin" in websocket.headers or not addressed_to_this_machine(websocket.headers):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        queue: asyncio.Queue = asyncio.Queue()
        gateway.clients.add(queue)
        if gateway.engine is not None:
            queue.put_nowait(await gateway.snapshot())
        else:
            queue.put_nowait({"type": "no_project"})

        async def sender() -> None:
            while True:
                await websocket.send_json(await queue.get())

        send_task = asyncio.create_task(sender())
        try:
            while True:
                try:
                    message = await websocket.receive_json()
                except ValueError:
                    # A malformed (non-JSON) frame must not tear the client down —
                    # report it and keep serving. (JSONDecodeError is a ValueError.)
                    queue.put_nowait({"type": "error",
                                      "message": "malformed message (expected JSON)"})
                    continue
                try:
                    await _handle(message)
                except Exception as exc:
                    queue.put_nowait({"type": "error", "message": str(exc)})
        except WebSocketDisconnect:
            pass
        finally:
            send_task.cancel()
            gateway.clients.discard(queue)

    async def _handle(message: dict) -> None:
        kind = message.get("type")
        if kind == "open_project":
            await gateway.open_project(message["path"])
            # absolute path so the desktop shell can track recent projects
            gateway.broadcast({"type": "project_opened",
                               "path": str(gateway.project_dir)})
            await gateway.broadcast_snapshot()
        elif kind == "new_project":
            await gateway.new_project(message.get("base_dir"))
            gateway.broadcast({"type": "project_opened",
                               "path": str(gateway.project_dir)})
            await gateway.broadcast_snapshot()
        elif kind == "save_project_as":
            await gateway.save_project_as(message["path"])
            gateway.broadcast({"type": "project_opened",
                               "path": str(gateway.project_dir)})
            await gateway.broadcast_snapshot()
        elif kind == "start_run":
            gateway.start_run(message["brief"],
                              message.get("gating", "manual"),
                              int(message.get("max_passes", 5)))
        elif kind == "stop_run":
            gateway.stop_run()
        elif kind == "stop_chat":
            gateway.stop_chat()
        elif kind == "auto_mix":
            gateway.start_auto_mix(message["target"])
        elif kind == "transpose_to_key":  # Phase 32: pick a key -> transpose
            await gateway.transpose_to_key(message["tonic"])
        elif kind == "undo_automix":
            await gateway.undo_auto_mix()
        elif kind == "separate_track":
            # Phase 29: user-only stem splitting (not an MCP tool / agent path).
            gateway.start_separation(message["path"],
                                     int(message.get("stems", 4)),
                                     message.get("separator"))
        elif kind == "stop_separation":
            gateway.stop_run()  # separation shares the run task slot
        elif kind == "approve":
            await gateway.decide(message["proposal_id"], True)
        elif kind == "reject":
            await gateway.decide(message["proposal_id"], False)
        elif kind == "undo":
            await gateway.undo(message["proposal_id"])
        elif kind == "instruction":
            gateway.instruct(message["text"])
        elif kind == "chat":
            gateway.chat_message(message["text"])
        elif kind == "render":
            await gateway.render()
        elif kind == "user_edit":
            await gateway.user_edit(message["stem_id"], message.get("changes", {}))
        elif kind == "set_reference":
            await gateway.set_reference(message["path"])
        elif kind == "set_reference_scope":
            gateway.set_reference_scope(message["scope"])
            await gateway.broadcast_snapshot()
        elif kind == "workspace_state":
            # Phase 17: the UI's situational report (playhead, loop, solo,
            # selection) — context for the chat, never broadcast back.
            gateway.workspace = {k: v for k, v in message.items()
                                 if k != "type"}
        elif kind == "set_chat_settings":
            gateway.set_chat_settings(message["model"], message["reasoning"])
            await gateway.broadcast_snapshot()
        elif kind == "set_agent_mode":
            gateway.set_agent_mode(message["mode"])
            await gateway.broadcast_snapshot()
        elif kind == "set_audition":
            # Phase 39 (VST): the plugin's Dry/Processed audition toggle. The
            # plugin polls /api/audition for this; nothing to broadcast.
            gateway.audition_mode = (
                "processed" if message.get("mode") == "processed" else "dry")
        elif kind == "client_mode":
            # Phase 39 (VST): the UI declares its shell (e.g. "plugin") so the
            # chat agent knows audio is captured from the bus, not loaded.
            gateway.client_mode = message.get("mode") or None
            if gateway.client_mode == "plugin":
                gateway.agent_mode = "agent"   # the plugin is hands-off (no Ask/Agent toggle)
        else:
            raise RuntimeError(f"unknown message type: {kind!r}")

    @app.get("/api/audition")
    def audition_state(stem: str = "") -> dict:
        """Phase 39/40 (VST): the plugin polls this for (a) the Dry/Processed
        mode; (b) the resolved chain params the native MonitorChain runs live
        (gain/EQ/comp/limiter), with `chain_rev` so it only rebuilds coefficients
        when the chain actually changes; and (c) the legacy bounce fields for the
        retired substitution path. The limiter makeup is the value the offline
        render computed (last_master_report). A direct localhost poll — no CORS.

        Phase 40 (per-track instances M3): `stem` = the insert's own stem/bus id
        (or "master") returns ITS chain, so each insert plays only its own track."""
        monitor = {"entries": [], "print_only": []}
        per_stem = False
        stem_exists = False
        try:
            state = json.loads((gateway.project_dir / "session.json").read_text())
            report = gateway.last_master_report or {}
            makeup = report.get("limiter_makeup_db") or 0.0
            # The pre-limiter integrated LUFS is fixed regardless of the loudness
            # target, so pass it through: monitor_chain_params resolves the limiter
            # makeup as (target - pre_lufs) live, and the Loud knob no longer lags
            # the monitor until the next render. Derive it from limiter_input_lufs_i
            # (post-makeup) when the cached pre value is absent (target was unset).
            pre_lufs = report.get("pre_limiter_lufs_i")
            if pre_lufs is None and report.get("limiter_input_lufs_i") is not None:
                pre_lufs = report["limiter_input_lufs_i"] - makeup
            monitor = monitor_chain_params(_audition_chain(state, stem), makeup, pre_lufs)
            # M4/M5: a real track/bus insert (one of several) shows the agent's
            # per-stem devices READ-ONLY (the co-edit endpoints target the master
            # chain, so editing there mis-writes). master / single-insert fold stay
            # editable.
            per_stem = _is_per_stem(state, stem)
            # Phase 41: does THIS insert's stem actually exist yet? The stem-only
            # COLOR add path (add_stem_device) needs a captured stem to target; the
            # host reports a track name before any capture, so gate the UI on this.
            stem_exists = stem in state.get("stems", {})
        except Exception:
            pass  # no project / unreadable state -> empty chain (plugin passes through)
        return {"mode": gateway.audition_mode,
                "chain_rev": gateway.bounce_rev,
                "monitor_chain": monitor,
                "per_stem": per_stem,
                "stem_exists": stem_exists,
                "bounce_rev": gateway.bounce_rev,
                "capture_start_s": gateway.capture_start_s}

    @app.post("/api/audition_mode")
    def post_audition_mode(mode: str = "dry") -> dict:
        """Phase 40 (native VST): the native Dry/Processed A/B toggle (no ws
        client). Sets the mode the MonitorChain polls from /api/audition; the
        toggle's current state is mirrored back in /api/readout."""
        gateway.audition_mode = "processed" if mode == "processed" else "dry"
        return {"ok": True, "mode": gateway.audition_mode}

    @app.get("/api/snapshot")
    async def snapshot_json() -> dict:
        if gateway.project_dir is None:   # the picker state — every neighbour 409s here too (2026-09-01)
            raise HTTPException(409, "no project open")
        """Phase 40: full UI state for the NATIVE plugin editor. JUCE ships no
        websocket client, so the native UI polls this instead of the websocket the
        webview uses. stale_ok=True — serve current state, never trigger a
        re-analysis per poll."""
        return await gateway.snapshot(stale_ok=True)

    @app.post("/api/chain_param")
    async def chain_param(device_type: str, occurrence: int = 0,
                          value: float = 0.0, stem: str = "master") -> dict:
        """Phase 40 (native VST co-edit): the native editor POSTs a user knob edit
        here (it has no websocket client). Persists to engine state (no proposal,
        no guardrails) + re-renders, so the agent re-reads it and the export
        reflects it. `stem` = master (default) or a track/bus id (a per-track insert
        edits its own stem)."""
        try:
            return await gateway.set_master_param(device_type, occurrence, value, stem)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/add_device")
    async def add_device(device_type: str, stem: str = "master") -> dict:
        """Phase 40 (native VST co-edit): the native rail adds a device. `stem` =
        master (default → eq/comp/limiter/dynamic_eq/saturation/width/mb_comp) or a
        track id (the COLOR sub-rail → transient/de-ess/band_width on that stem).
        User channel — no proposal."""
        try:
            if stem and stem != "master":
                return await gateway.add_stem_device(stem, device_type)
            return await gateway.add_master_device(device_type)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/remove_device")
    async def remove_device(device_type: str, occurrence: int = 0,
                            stem: str = "master") -> dict:
        """Phase 40 (native VST co-edit): remove a device (the × on a tile).
        `stem` = master (default) or a track id (a stem color device)."""
        try:
            if stem and stem != "master":
                return await gateway.remove_stem_device(stem, device_type, occurrence)
            return await gateway.remove_master_device(device_type, occurrence)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/device_param")
    async def device_param(device_type: str, occurrence: int = 0,
                           param: str = "", value: float = 0.0,
                           stem: str = "master") -> dict:
        """Phase 40 (native VST co-edit): the device detail popup sets one param
        (freq/q/threshold/etc.). User channel — no proposal. `stem` = master or a
        track/bus id."""
        try:
            return await gateway.set_master_device_param(
                device_type, occurrence, param, value, stem)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/eq_shape")
    async def eq_shape(occurrence: int = 0, shape: str = "peak",
                       stem: str = "master") -> dict:
        """Phase 40 (native VST co-edit): the detail popup sets an EQ band's shape.
        `stem` = master or a track/bus id."""
        try:
            return await gateway.set_eq_shape(occurrence, shape, stem)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/device_channel")
    async def device_channel(device_type: str = "eq", occurrence: int = 0,
                             channel: str = "stereo", stem: str = "master") -> dict:
        """Phase 41 (native VST co-edit): the eq/comp M/S selector (stereo/mid/side).
        `stem` = master or a track/bus id."""
        try:
            return await gateway.set_device_channel(device_type, occurrence, channel, stem)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/device_band")
    async def device_band(occurrence: int = 0, band: str = "presence",
                          stem: str = "master",
                          lo_hz: float | None = None,
                          hi_hz: float | None = None) -> dict:
        """Band-width band selector (2026-07-05): the user picks the device's band
        LABEL — or, 2026-07-19, explicit lo_hz/hi_hz to place the band freely.
        `stem` = master or a track/bus id."""
        try:
            return await gateway.set_device_band(occurrence, band, stem,
                                                 lo_hz=lo_hz, hi_hz=hi_hz)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/sat_multiband")
    async def sat_multiband(occurrence: int = 0, on: bool = True,
                            stem: str = "master") -> dict:
        """Phase 45 (native VST co-edit): the saturation single ↔ 3-band MB toggle."""
        try:
            return await gateway.set_sat_multiband(occurrence, on, stem)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/device_enabled")
    async def device_enabled(device_type: str, occurrence: int = 0,
                             enabled: bool = True, stem: str = "master") -> dict:
        """Phase 40 (native VST co-edit): the tile on/off bypasses a device.
        `stem` = master or a track/bus id."""
        try:
            return await gateway.set_master_device_enabled(
                device_type, occurrence, enabled, stem)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/device_order")
    async def device_order(order: str = "") -> dict:
        """Phase 42 (native VST co-edit): drag-to-reorder the master device chain.
        `order` is a comma-separated list of device-type modules left-to-right
        (e.g. "eq,comp,saturation,limiter"). Master only; the limiter is pinned
        last server-side. User channel — no proposal."""
        try:
            kinds = [k for k in order.split(",") if k]
            return await gateway.set_master_device_order(kinds)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/chat")
    async def post_chat(text: str) -> dict:
        """Phase 40 (native VST): the native editor sends a user chat message here
        (no ws). Schedules the resident-engineer turn; the reply streams into the
        chat_stream buffer the native UI polls via /api/chat_log."""
        try:
            gateway.chat_message(text)
            return {"ok": True}
        except RuntimeError as exc:
            return {"ok": False, "error": str(exc)}

    @app.post("/api/chat_stop")
    async def post_chat_stop() -> dict:
        """Phase 40: the native composer's Stop — abort the current chat turn."""
        gateway.stop_chat()
        return {"ok": True}

    @app.post("/api/auto_mix")
    async def post_auto_mix(target: str = "") -> dict:
        """Phase 40 (native VST): the Ozone-style one-press AI button. It TOGGLES:
        - while a run / Auto Mix is active (or armed-listening), pressing again
          stops it (cost safety — see run-cost-controls);
        - with a capture present, it starts the full Auto Mix now (deterministic
          bootstrap chain + bounded refinement, applied hands-off, one-click
          undoable) — finishes with the devices in place;
        - with nothing captured yet, it ARMS: the button reads "listening" and
          Auto Mix fires on the next capture (press, then play your track)."""
        running = bool(gateway.run_task and not gateway.run_task.done())
        if running or gateway.auto_mix_armed is not None:
            gateway.auto_mix_armed = None      # disarm a pending "listening"
            if running:
                gateway.stop_run()             # only cancels when a run is live
            return {"ok": True, "status": "stopped"}
        brief = target.strip() or "mix and master this — balanced, clean and loud"
        try:
            state = json.loads((gateway.project_dir / "session.json").read_text())
            has_capture = any(not s.get("excluded")
                              for s in state.get("stems", {}).values())
        except Exception:
            has_capture = False
        if not has_capture:
            gateway.auto_mix_armed = brief            # fire on the next capture
            return {"ok": True, "status": "listening"}
        try:
            gateway.start_auto_mix(brief)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return {"ok": True, "status": "started"}

    @app.post("/api/proposal_decide")
    async def proposal_decide(proposal_id: str, accept: bool = True) -> dict:
        """Phase 40 (native VST): the native UI's proposal Approve/Reject (no ws).
        A decision on an already-resolved id is a harmless no-op (see decide())."""
        await gateway.decide(proposal_id, accept)
        return {"ok": True}

    @app.post("/api/agent_mode")
    def post_agent_mode(mode: str = "ask") -> dict:
        """Phase 40 (native VST): the native UI's Ask/Agent toggle (no ws). Ask
        gates every proposal for approval; Agent auto-applies."""
        gateway.set_agent_mode("agent" if mode == "agent" else "ask")
        return {"ok": True}

    @app.post("/api/chat_settings")
    def post_chat_settings(model: str, reasoning: str) -> dict:
        """Phase 40 (native VST): the native model + thinking-level pickers (no ws).
        Mirrors the webview's `set_chat_settings` ws message; takes effect on the
        next turn. Unknown values -> 400."""
        try:
            gateway.set_chat_settings(model, reasoning)
        except ValueError as exc:        # unknown model / reasoning level
            raise HTTPException(400, str(exc))
        except RuntimeError as exc:      # no project / chat not ready
            raise HTTPException(409, str(exc))
        return {"ok": True, "model": model, "reasoning": reasoning}

    @app.post("/api/client_mode")
    def post_client_mode(mode: str = "") -> dict:
        """Phase 40 (native VST): the native editor announces its host (no ws), so the
        chat agent uses the plugin playbook — captures are the user's tracks (one
        plugin insert per track + the master), never hunt for stem files or a folder.
        Mirrors the webview's client_mode ws msg."""
        gateway.client_mode = mode or None
        if gateway.client_mode == "plugin":
            gateway.agent_mode = "agent"   # the plugin is hands-off — no Ask/Agent toggle
        return {"ok": True}

    # `async def` (2026-09-01 sweep): as plain `def` these ran in the threadpool and iterated
    # `pending_proposals` / `feed` / `clients` while the event loop mutated them ("dictionary
    # changed size during iteration" 500s on the native UI's poll), and `broadcast()` called
    # asyncio.Queue.put_nowait off the loop thread, which never woke it. None of them block.
    @app.get("/api/chat_log")
    async def chat_log() -> dict:
        """Phase 40: the native UI's chat transcript poll (no ws). Completed
        messages from the feed + the in-progress streamed reply + the busy flag."""
        msgs = [{"role": e.get("role", ""), "text": e.get("text", "")}
                for e in gateway.feed if e.get("type") == "chat_message"]
        proposals = []
        for pid in sorted(gateway.chat_awaiting):          # chat-initiated proposals
            req = gateway.chat_proposals.get(pid)
            if req:
                proposals.append({"id": pid, "rationale": req.get("rationale", ""),
                                  "actions": [a.get("type", "") for a in req.get("actions", [])]})
        for pid, prop in gateway.pending_proposals.items():  # run / Auto Mix gated proposals
            fut = gateway.pending.get(pid)
            if fut is not None and not fut.done():
                proposals.append({"id": pid, "rationale": prop.get("rationale", ""),
                                  "actions": [a.get("type", "") for a in prop.get("actions", [])]})
        return {"messages": msgs, "streaming": gateway.chat_stream,
                "busy": gateway.chat_busy, "agent_mode": gateway.agent_mode,
                # Phase 40: drives the native AI button ("listening") + the
                # thinking strip during a hands-off Auto Mix (no chat turn busy).
                "run_active": bool(gateway.run.get("running")),
                "listening": gateway.auto_mix_armed is not None,
                "run_status": gateway.run_status,   # plain-words progress line
                # Phase 40: the native model + thinking-level pickers — current
                # selection plus the registry so the UI builds the menus without drift.
                "model": gateway.chat_settings["model"],
                "reasoning": gateway.chat_settings["reasoning"],
                "models": CHAT_MODELS,
                "reasoning_levels": REASONING_LEVELS,
                "proposals": proposals}

    @app.post("/api/capture_progress")
    async def capture_progress(seconds: float = 0.0) -> dict:
        """Phase 39 (VST): the plugin reports how many seconds of bus audio it has
        captured in the current play (0 when it stops); broadcast to the assistant
        so its ring shows how much more to play for a confident read."""
        gateway.capture_seconds = max(0.0, seconds)
        gateway.broadcast({"type": "capture_progress",
                           "seconds": round(gateway.capture_seconds, 2)})
        return {"ok": True}

    @app.post("/api/meters")
    async def meters(lufs: float = -100.0, peak: float = -100.0,
               corr: float = 0.0, bands: str = "", spectrum: str = "",
               stem: str = "") -> dict:
        """Phase 39 (VST): the plugin streams live meters (~10 Hz while the host
        plays) — momentary loudness, peak, correlation, 8-band spectrum. A feel
        display: the agent still reasons on the engine's offline analysis.

        Phase 40 (per-track instances): `stem` keys the meters by track, so each
        insert's analyzer shows its OWN signal instead of whichever instance posted
        last. The master / single insert ("") drives the legacy whole-mix feed."""
        try:
            band_vals = [float(x) for x in bands.split(",") if x != ""]
        except ValueError:
            band_vals = []
        try:
            spec_vals = [float(x) for x in spectrum.split(",") if x != ""]
        except ValueError:
            spec_vals = []
        snapshot = {"lufs": lufs, "peak": peak, "corr": corr, "bands": band_vals,
                    "spectrum": spec_vals,
                    "_ts": time.monotonic()}   # stamp so a reader can tell live from stale
        gateway.live_meters_by_stem[stem] = snapshot
        if not stem:                       # whole-mix / master drives the webview feed
            gateway.last_meters = snapshot
            gateway.broadcast({"type": "meters", "lufs": lufs, "peak": peak,
                               "corr": corr, "bands": band_vals})
        return {"ok": True}

    @app.get("/api/readout")
    def readout(stem: str = "") -> dict:
        """Phase 40: the native editor's analyzer feed (no ws client). Live meters
        (while the host plays) + the engine's analyzed metrics. Phase 40 (per-track
        instances): `stem` makes a track insert show ITS OWN live spectrum + band
        energy; the whole-mix LUFS/peak/key chips stay as shared context."""
        a = gateway.last_analysis or {}
        m = gateway.live_meters_by_stem.get(stem) or gateway.last_meters or {}
        # Live meters stop posting when playback stops, so the last value lingers.
        # Treat anything older than ~1 s as NOT live (silent) — otherwise a reopened
        # editor reads stale meters and thinks audio is still flowing.
        if m.get("_ts") and time.monotonic() - m["_ts"] > 1.0:
            m = {}
        if stem and stem in gateway._stem_cache:   # per-track paused-spectrum fallback
            sa = gateway._stem_cache[stem][1] or {}
            if sa.get("band_energy"):
                a = {**a, "band_energy": sa["band_energy"]}
        # Phase 40: the reference's tonal balance, in the native 8-band order, as
        # a target curve the analyzer overlays (only when a reference is loaded).
        _bands = ["sub", "bass", "low_mid", "mid", "high_mid",
                  "presence", "brilliance", "air"]
        ref_be = (gateway.reference or {}).get("band_energy") or {}
        target_bands = ([ref_be[b] for b in _bands]
                        if all(b in ref_be for b in _bands) else None)
        rd = gateway.recent_device
        return {"bands_live": m.get("bands"),
                "spectrum_live": m.get("spectrum"),
                "live_lufs": m.get("lufs"), "live_peak": m.get("peak"),
                "live_corr": m.get("corr"),
                "lufs_i": a.get("lufs_i"),
                "true_peak_dbtp": a.get("true_peak_dbtp"),
                "stereo_correlation": a.get("stereo_correlation"),
                "key": a.get("key"),
                "band_energy": a.get("band_energy"),
                "target_bands": target_bands,
                "audition_mode": gateway.audition_mode,
                "recent_device": ([rd[0], rd[1]] if rd else None)}

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Mix agent websocket gateway")
    parser.add_argument("project_dir", nargs="?", default=None,
                        help="project folder to open at startup (optional)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--rule-agent", action="store_true",
                        help="use the deterministic no-LLM agent (smoke runs)")
    args = parser.parse_args()
    factory = RuleAgent if args.rule_agent else None
    chat_factory = ScriptedChat if args.rule_agent else None
    if args.rule_agent:
        print("LLM backend: none (--rule-agent: deterministic, no LLM)")
    else:
        from .llm_backend import active_backend
        if active_backend() == "subscription":
            print("LLM backend: subscription — Claude Agent SDK on your Claude "
                  "plan (dev path; production/benchmark stay on api)")
            if os.environ.get("ANTHROPIC_API_KEY"):
                print("  warning: ANTHROPIC_API_KEY is set and overrides the "
                      "subscription — unset it.")
            if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
                print("  warning: CLAUDE_CODE_OAUTH_TOKEN not set — run "
                      "`claude setup-token` and export it.")
        elif active_backend() == "cloud":
            print("LLM backend: cloud — metered control plane (user JWT; "
                  "company Anthropic key stays server-side)")
            if not os.environ.get("MIX_AGENT_CLOUD_URL"):
                print("  warning: MIX_AGENT_CLOUD_URL not set")
            if not os.environ.get("MIX_AGENT_ACCESS_TOKEN"):
                print("  warning: MIX_AGENT_ACCESS_TOKEN not set")
        else:
            print("LLM backend: api — direct Anthropic Messages API "
                  "(evaluation path; ANTHROPIC_API_KEY billed per token)")
    app = create_app(args.project_dir, agent_factory=factory,
                     chat_factory=chat_factory)
    uvicorn.run(app, host=HOST, port=args.port)


if __name__ == "__main__":
    main()
