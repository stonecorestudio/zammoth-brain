"""Pass-control loop (docs/04-agent-loop.md — the pseudocode there is the spec).

Holds: pass budget, gating policies (manual / auto_conservative / auto),
stopping criteria, prediction accountability to runs/*.jsonl. Talks to the
engine only over MCP; tool semantics come from the engine.

The core is `run_passes`, which is transport-agnostic: gating decisions and
progress events are injected callables, so the same loop serves the CLI
(policy gate, prints) and the websocket gateway (browser approvals, event
stream — see gateway.py).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import sys
import time
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import asyncio

import anyio
import anyio.from_thread
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .agents import Agent, PassContext, ReviewContext, RuleAgent, TargetState, _match_eq_band
from .calibration import (append_lesson, calibrate_proposal, calibration_table,
                          confidence_for, exemplars, load_global_rows,
                          load_lessons, load_records, load_records_file,
                          record_decision, taste_summary, update_global_store)
from .edit_state import live_edit
from .engine_models import LIMITS, MASKING_METRIC_RE, STEM_METRIC_RE
from .targets import (brief_notes, hold_loudness_to_limiter,
                      interpret_brief, resolve_stem, Target, compiled_targets,
                      reference_mix_targets, reference_set_mix_targets,
                      reference_set_targets, reference_specs_from_brief,
                      reference_targets, SECTION_BAND_METRIC_RE,
                      targets_from_brief)

def _plain_validation(text: str) -> Optional[str]:
    """A request model's refusal as sentences, or None when `text` is not one.

    What pydantic prints is for a developer: a count, each problem's path through the model
    (`actions.0.function-after[_check_range(), EqAction].band`), the message, then
    `[type=..., input_value=..., input_type=...]` and a link. An action is one of a plain
    union, so ONE wrong value is reported once for every action type there is - 67 problems
    for a gain whose `delta_db` was a word - and all but the action's own are "this is not an
    eq". Kept: the problems of the member whose `type` matched, as `action 0, delta_db:
    Input should be a valid number`; a model's own sentence (the ones `mix_engine.models`
    raises) as it was written. The showcase rehearsal put the dump in the feed (2026-10-11)."""
    lines = text.splitlines()
    if not lines or not re.match(r"^\d+ validation errors? for \S+", lines[0].strip()):
        return None
    entries: list[tuple[str, str, str]] = []            # (where, what, pydantic's kind)
    where = ""
    for line in lines[1:]:
        if not line.strip() or line.lstrip().startswith("For further information"):
            continue
        if not line.startswith(" "):
            # the tool's own argument is called `req`: not a name the agent ever wrote
            where = re.sub(r"^req(\.|$)", "", line.strip())
            continue
        cut = line.find(" [type=")
        what = (line if cut < 0 else line[:cut]).strip()
        kind = re.search(r"\[type=(\w+)", line[cut:] if cut >= 0 else "")
        entries.append((where, re.sub(r"^Value error,\s*", "", what), kind.group(1) if kind else ""))
        where = ""

    def slot(path: str) -> tuple[str, list[str]]:
        """(the path up to and including a union member, the fields under it)."""
        parts = path.split(".")
        for i, part in enumerate(parts):
            if "Action" in part:
                return ".".join(parts[: i + 1]), parts[i + 1:]
        return "", parts

    not_this_type = {slot(path)[0] for path, _what, kind in entries
                     if kind == "literal_error" and slot(path)[1] == ["type"] and slot(path)[0]}
    said: list[str] = []
    for path, what, _kind in entries:
        member, fields = slot(path)
        if member in not_this_type:
            continue
        if member:
            index = re.search(r"(?:^|\.)actions\.(\d+)\.", path)
            lead = (f"action {index.group(1)}" if index else "an action") + (
                f", {'.'.join(fields)}" if fields else "")
            line = f"{lead}: {what}"
        else:
            line = f"{path}: {what}" if path else what
        if line not in said:
            said.append(line)
    if not said:                                        # every member refused the type
        said = [what for _path, what, _kind in entries[:1]]
    more = len(said) - 3
    return "; ".join(said[:3]) + (f" (and {more} more)" if more > 0 else "")


def tool_error_text(name: str, content: Any) -> str:
    """What a failed engine tool SAYS (2026-09-10 audit P3, 05·F17). The MCP content list went
    into the message as a Python repr, so a stem file deleted outside the app put
    `[TextContent(type='text', text="Error executing tool auto_tag_stems: Error opening
    '/…/drums.wav': System error.", …)]` in the status bar. The text itself, the SDK's
    prefix off, and a file that cannot be opened named plainly."""
    texts = [getattr(c, "text", None) or str(c) for c in (content or [])]
    text = " ".join(t.strip() for t in texts if t and t.strip()) or "no detail"
    text = re.sub(r"^Error executing tool \S+:\s*", "", text)
    text = _plain_validation(text) or text
    if (m := re.search(r"Error opening '([^']+)'", text)):
        missing = Path(m.group(1))
        # The same anchored redaction (2026-09-11 pre-merge review, D5): a bare replace turned
        # /Users/parrot/Music into "~ot/Music" in the run's own failure text.
        from .edit_state import redact_home_text
        where = redact_home_text(str(missing.parent))
        return (f"{missing.name} is missing or unreadable (in {where}) — was it moved or "
                "deleted outside the app?")
    return f"engine tool '{name}' failed: {text}"


#: How far off the limiter's own goal still counts as ITS residual rather than a move the run
#: could still make. The makeup re-resolves at every render, so a few tenths is the limiter
#: settling; EPSILON (0.3) is the smallest improvement worth a pass, and this is its loudness
#: sibling with room for one render's wobble (2026-09-11 pre-merge review, C5).
LIMITER_RESIDUAL_LU = 1.0


def rail_spent(states: list[TargetState], eq_headroom: dict,
               master_chain: list[dict]) -> Optional[list[str]]:
    """The open targets only more EQ could move, once the composed EQ rail has no room left
    for them (audit follow-ups 2026-09-11, item 1). Returns one line per saturated band, or
    None while any open target can still move.

    Measured on the rule benchmark's pop master: the rail was spent after pass 4, and passes
    5 and 6 re-applied the SAME -10 LUFS limiter for a 0.3 LU residual — passes that cost
    nothing on the rule agent and a full planner round each on the LLM one. So a loudness
    target the master limiter already aims at counts as settled here: the limiter's makeup
    re-resolves at every render, and what remains is its residual, not a move."""
    unmet = [s for s in states if s.distance > 0]
    if not unmet:
        return None
    limiter_goal = next((a.get("target_lufs_i") for a in master_chain
                         if a.get("type") == "limiter"), None)
    saturated: list[str] = []
    for s in unmet:
        if s.metric == "lufs_i":
            # …and only while what remains IS the residual. The limiter aiming at the goal
            # said nothing about the DISTANCE to it, so a run 3-6 LU short stopped claiming
            # "EQ rail reached" while loudness could still plainly move (2026-09-11 pre-merge
            # review, C5). Past the residual the limiter is not achieving the goal, and a
            # gain/comp/limiter move still can.
            if (limiter_goal is not None and s.goal is not None
                    and abs(float(limiter_goal) - s.goal) <= 0.1
                    and s.distance <= LIMITER_RESIDUAL_LU):
                continue
            return None
        target, metric = "master", s.metric
        if (m := STEM_METRIC_RE.match(s.metric)):
            target, metric = m.group(1).strip(), m.group(2).strip()
        band = _match_eq_band(metric)
        room = (eq_headroom.get(target) or {}).get(band) if band else None
        if room is None or s.goal is None or s.current is None:
            return None
        need = s.goal - s.current
        if need > 0 and room.get("boost", 1.0) <= 0.0:
            saturated.append(f"{band} {need:.1f} dB under"
                             + ("" if target == "master" else f" on {target}"))
        elif need < 0 and room.get("cut", -1.0) >= 0.0:
            saturated.append(f"{band} {-need:.1f} dB over"
                             + ("" if target == "master" else f" on {target}"))
        else:
            return None
    return saturated or None


def _only_eq_or_loudness_open(states: list[TargetState]) -> bool:
    """Cheap pre-check, so the extra get_state runs only when rail_spent could say yes."""
    for s in states:
        if s.distance <= 0 or s.metric == "lufs_i":
            continue
        m = STEM_METRIC_RE.match(s.metric)
        if _match_eq_band(m.group(2).strip() if m else s.metric) is None:
            return False
    return True


MAX_PASSES = 5
EPSILON = 0.3  # min meaningful improvement, in target-metric units
STALL_PASSES = 2  # consecutive sub-EPSILON passes before "diminishing returns"
#: An empty pass the model did not choose (agent.last_empty_reason → the run's stop
#: reason). Anything else empty is the model's own "nothing to propose".
EMPTY_PASS_REASONS = {
    "out of turns": "the planner ran out of turns before proposing",
    "deadline": "the planner ran out of time before proposing",
    "invalid proposals": "the planner's proposals never validated",
    "spend limit": "spend limit reached",
    # 2026-09-22 audit B·F4: both used to read "nothing to propose" — the words for a fine mix
    "declined": "the model declined this pass (Anthropic's safety check)",
    "cut off": "the planner ran out of room before proposing — lower the effort level or ask again",
}

# Phase 12: what the planner's investigation bridge will serve — reads and
# the probe sandbox only. Mutations stay behind the proposal lifecycle
# (CLAUDE.md invariant); this whitelist is enforced here, not in prompts.
PLANNER_ENGINE_TOOLS = frozenset({
    "analyze_session", "analyze_stem", "analyze_bus", "analyze_return",
    "analyze_masking", "analyze_sections", "analyze_phrases", "analyze_reference",
    "analyze_reference_set", "get_state", "get_proposals", "probe_actions",
    # The 2026-08-08 ears. They were chat-only at first, so a RUN brief like
    # "tighten the drums" had the planner guessing where chat could measure —
    # and INVESTIGATE_INSTRUCTIONS told it to size spectral_denoise "from
    # analyze_separation", a tool this very allowlist refused (unfollowable
    # advice since ba1ea7e).
    "analyze_timing", "analyze_chords", "analyze_notes", "analyze_separation",
    # 2026-10-11 (the showcase rehearsal): one part's own tempo, and where each part plays
    # to the bar
    "analyze_tempo", "analyze_activity",
    # one sound - a one-shot or a loop - at any length (Phase 58)
    "analyze_sample",
    # what cancels when the song is folded to mono, and why (2026-10-13)
    "analyze_phase",
})

ATTRIBUTION_MAX_ACTIONS = 4  # probe per-action deltas for bundles up to this

# 2026-06 review: a comp threshold deeper than this compresses essentially
# everything — never "conservative", whatever the ratio says.
CONSERVATIVE_COMP_THRESHOLD_DB = -35.0

# Phase 14: scalar fields worth a pass-to-pass delta in the agent context.
_DELTA_SCALARS = ("lufs_i", "lufs_s_max", "lufs_range", "true_peak_dbtp",
                  "plr_db", "psr_db", "crest_factor_db", "stereo_correlation",
                  "harshness_db", "mud_db", "clarity_index")


def _metrics_delta(prev: Optional[dict], curr: dict) -> Optional[dict]:
    """What changed since the previous pass's context: scalar deltas + per-
    band energy deltas. None on the first pass or when nothing is comparable."""
    if prev is None:
        return None
    delta: dict = {}
    for key in _DELTA_SCALARS:
        before, after = prev.get(key), curr.get(key)
        if before is not None and after is not None:
            delta[key] = round(after - before, 2)
    prev_bands = prev.get("band_energy") or {}
    curr_bands = curr.get("band_energy") or {}
    bands = {band: round(curr_bands[band] - prev_bands[band], 2)
             for band in curr_bands if band in prev_bands}
    if bands:
        delta["band_energy"] = bands
    return delta or None

# Gate: (proposal_id, proposal dict) -> approve? Called after the engine
# accepts a proposal, before apply. Emit: progress events for a UI/log.
GateFn = Callable[[str, dict], Awaitable[bool]]
EmitFn = Callable[[dict], Awaitable[None]]


@dataclass
class SessionReport:
    stop_reason: str
    passes_run: int
    applied_proposal_ids: list[str] = field(default_factory=list)
    # Rolled back because the run ended measurably worse than an earlier pass
    # (2026-08-05). Kept separate so `applied_proposal_ids` keeps meaning
    # "still standing" — the DAW's ledger and whole-run Undo both rely on that.
    reverted_proposal_ids: list[str] = field(default_factory=list)
    runs_path: Optional[str] = None
    final_metrics: Optional[dict] = None
    warnings: list[str] = field(default_factory=list)


class EngineClient:
    """Thin convenience wrapper over an MCP ClientSession.

    An optional async lock serializes tool calls so a concurrent caller
    (e.g. the gateway servicing an undo) never interleaves with the loop.

    Two lanes (gateway responsiveness): renders take `render_lock` and the
    frozen read-only analyses take no lock at all, so quick user mutations
    on the main lock never queue behind a multi-second render. Safe because
    those engine tools work on a state copy frozen before the thread starts
    (see mcp_server.create_server's concurrency contract); every tool that
    touches live state stays on the main lock. With `render_lock` omitted,
    renders share the main lock — the old single-lane behavior.
    """

    # Long renders: exclusive among themselves (bounce files), concurrent
    # with everything on the main lock.
    RENDER_LANE = frozenset({"render_and_analyze"})
    # Frozen-copy, read-only, no files written: free-running.
    LOCK_FREE = frozenset({"analyze_stem", "analyze_sections",
                           "analyze_phrases"})

    def __init__(self, session: ClientSession, lock: Optional[anyio.Lock] = None,
                 render_lock: Optional[anyio.Lock] = None):
        self._session = session
        self._lock = lock if lock is not None else contextlib.nullcontext()
        self._render_lock = render_lock if render_lock is not None else self._lock

    def _lane(self, name: str):
        if name in self.RENDER_LANE:
            return self._render_lock
        if name in self.LOCK_FREE:
            return contextlib.nullcontext()
        return self._lock

    # Wall-clock ceiling per engine tool call (2026-08-09). Without one, a stuck
    # render/lock held the whole pass — and the user's Stop — indefinitely; with
    # it, the stall surfaces as a tool error the pass handler already reports.
    # Generous on purpose: a legitimate full-session render on a big project is
    # minutes, and tripping this must mean "wedged", never "slow but working".
    CALL_TIMEOUT_S = float(os.environ.get("MIX_AGENT_ENGINE_TIMEOUT_S", "600"))

    async def call(self, name: str, args: Optional[dict] = None):
        try:
            with anyio.fail_after(self.CALL_TIMEOUT_S):
                async with self._lane(name):
                    result = await self._session.call_tool(name, args or {})
        except TimeoutError:
            raise RuntimeError(
                f"engine tool '{name}' timed out after {self.CALL_TIMEOUT_S:.0f}s "
                "(wedged render or lock)") from None
        if result.isError:
            raise RuntimeError(tool_error_text(name, result.content))
        payload = result.structuredContent
        if isinstance(payload, dict) and set(payload) == {"result"}:
            payload = payload["result"]
        if payload is None:
            # A tool annotated with a bare `dict` (reset_stem, …) gets no output schema, so
            # FastMCP ships its result as TEXT only — the JSON in content[0] — and this
            # returned None for it (2026-09-04: import_stem read reset_stem's `reset` flag
            # as always-false). Parse the text when it is JSON; leave a real None alone.
            texts = [getattr(c, "text", None) for c in (result.content or [])]
            if len(texts) == 1 and isinstance(texts[0], str):
                try:
                    payload = json.loads(texts[0])
                except ValueError:
                    payload = None
        return payload

    async def render_and_analyze(self) -> dict:
        # Render failure -> retry once, then surface (docs/04 failure handling).
        try:
            return await self.call("render_and_analyze")
        except RuntimeError:
            return await self.call("render_and_analyze")

    async def measure_all(self, metrics: list[str],
                          pre_master: bool = False) -> dict[str, Optional[float]]:
        """Resolve target metrics to current values with minimal renders
        (Phase 10): one `analyze_session` serves every non-masking metric;
        one `analyze_masking` per unique stem pair serves the rest. (The old
        per-metric path re-rendered the master once per target — ~10 renders
        per measurement round on reference briefs.) `pre_master` (Phase 25)
        measures session metrics on the pre-master render — mix-scope
        reference matching; masking pairs are per-stem and unaffected."""
        values: dict[str, Optional[float]] = {}
        maskings: dict[tuple[str, str], dict] = {}
        stem_analyses: dict[str, dict] = {}   # one analyze_stem per stem, reused
        analysis: Optional[dict] = None
        sections: Optional[dict] = None
        for metric in metrics:
            if match := MASKING_METRIC_RE.match(metric):
                stem_a, stem_b, fname = match.groups()
                pair = (stem_a.strip(), stem_b.strip())
                if pair not in maskings:
                    maskings[pair] = await self.call(
                        "analyze_masking", {"stem_a": pair[0], "stem_b": pair[1]})
                masking = maskings[pair]
                if fname == "worst_band_excess_db":
                    values[metric] = masking["worst_band_excess_db"]
                elif fname == "spatial_release_db":  # Phase 20
                    values[metric] = masking.get("spatial_release_db")
                else:
                    values[metric] = masking["per_band_excess_db"].get(fname)
                continue
            if match := SECTION_BAND_METRIC_RE.match(metric):
                section_rank = int(match.group(1))
                band = match.group(2)
                if sections is None:
                    sections = await self.call(
                        "analyze_sections",
                        {"pre_master": True} if pre_master else {})
                section_list = sections.get("sections") or []
                # The metric's index is a loudness rank (0 = loudest), so the
                # target stays pinned to the same section even as boundaries /
                # ordering drift between passes (targets.py / analysis.py).
                ranked = sorted(
                    section_list,
                    key=lambda s: (s.get("lufs_i")
                                   if s.get("lufs_i") is not None else -120.0),
                    reverse=True)
                if section_rank >= len(ranked):
                    values[metric] = None
                else:
                    values[metric] = (
                        ranked[section_rank].get("band_energy") or {}
                    ).get(band)
                continue
            if match := STEM_METRIC_RE.match(metric):
                # Per-stem targets (2026-07-30). Without these the loop could only
                # ever aim at SESSION-level numbers, so every per-instrument ask
                # ("the snare has no crack", "the bass is inconsistent") had to be
                # flattened into a master-wide goal — which is the wrong move even
                # when the words were understood perfectly.
                stem_id, field = match.group(1).strip(), match.group(2).strip()
                if stem_id not in stem_analyses:
                    try:
                        stem_analyses[stem_id] = await self.call(
                            "analyze_stem", {"stem_id": stem_id})
                    except RuntimeError:
                        stem_analyses[stem_id] = {}
                stem = stem_analyses[stem_id] or {}
                if field.startswith("band_energy."):
                    values[metric] = (stem.get("band_energy") or {}).get(
                        field.removeprefix("band_energy."))
                else:
                    values[metric] = stem.get(field)
                continue
            if analysis is None:
                analysis = await self.call(
                    "analyze_session",
                    {"pre_master": True} if pre_master else {})
            if metric.startswith("band_energy."):  # Phase 7 reference deltas
                values[metric] = (analysis.get("band_energy") or {}).get(
                    metric.removeprefix("band_energy."))
            else:
                values[metric] = analysis.get(metric)
        return values

    async def measure(self, metric: str) -> Optional[float]:
        """Resolve one target metric name to its current value."""
        return (await self.measure_all([metric]))[metric]


@asynccontextmanager
async def _stdio_engine(project_dir: Path):
    """Spawn the engine MCP server as a subprocess over stdio (dev default)."""
    server_dir = Path(__file__).resolve().parent.parent / "server"
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mix_engine.mcp_server", str(project_dir)],
        cwd=str(server_dir))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


GATING_POLICIES = ("manual", "auto_conservative", "auto_confident", "auto")


def policy_gate(gating: str, runs_dir: Optional[Path] = None) -> GateFn:
    """The built-in gating policies as a GateFn. `manual` prompts on stdin
    (CLI use); the gateway supplies its own GateFn for browser approvals.
    `auto_confident` (Phase 13) is earned autonomy: auto-apply only when
    the move is conservative AND the agent's own record says predictions
    for this action/metric mix land (the loop attaches that verdict to the
    proposal as `confidence`); everything else queues like `manual`.
    With `runs_dir`, decisions made at the stdin prompt — a human, never a
    policy — are recorded for the taste summary (Phase 20)."""
    if gating not in GATING_POLICIES:
        raise ValueError(f"unknown gating policy: {gating}")

    async def gate(proposal_id: str, proposal: dict) -> bool:
        if gating == "auto":
            return True
        if gating == "auto_conservative":
            return all(_is_conservative(a) for a in proposal["actions"])
        if gating == "auto_confident" and _is_confident_and_conservative(proposal):
            return True
        print(f"\nProposal: {proposal['rationale']}")
        print(json.dumps(proposal["actions"], indent=2))
        approve = input("Apply? [y/N] ").strip().lower() == "y"
        if runs_dir is not None:
            record_decision(runs_dir, "approved" if approve else "rejected",
                            proposal, source="cli")
        return approve

    return gate


def _is_confident_and_conservative(proposal: dict) -> bool:
    return (bool(proposal.get("confidence", {}).get("confident"))
            and all(_is_conservative(a) for a in proposal["actions"]))


def _is_conservative(action: dict) -> bool:
    """Auto-approvable = every parameter <= half the guardrail limit (docs/04)."""
    if action["type"] == "gain":
        return abs(action["delta_db"]) <= LIMITS["gain_per_stem_per_pass_db"] / 2
    if action["type"] == "eq":
        band = action["band"]
        half_ok = LIMITS["eq_cut_min_db"] / 2 <= band["gain_db"] <= LIMITS["eq_boost_max_db"] / 2
        hpf_ok = band["shape"] != "hpf" or band["freq_hz"] <= LIMITS["hpf_max_hz_default"] / 2
        return half_ok and hpf_ok
    if action["type"] == "comp":
        return (action["ratio"] <= LIMITS["comp_ratio_max"] / 2
                and action["threshold_db"] >= CONSERVATIVE_COMP_THRESHOLD_DB)
    if action["type"] == "pan":
        return abs(action["pan"]) <= 0.5
    if action["type"] == "width":
        return 0.5 <= action["width"] <= 1.5
    return False  # de-ess and anything new: never auto-approve


async def _no_emit(event: dict) -> None:
    pass


# Phase 39: durable per-project notes (saved by the chat agent) feed free-form
# brief compilation — they live under runs/ next to the run logs, the same file
# the gateway reads for chat context (gateway.NOTES_FILE).
NOTES_FILE = "notes.jsonl"
NOTES_FOR_BRIEF = 20


def _load_notes(runs_dir: Path) -> list[str]:
    """The last NOTES_FOR_BRIEF saved notes' text. Mirrors the gateway's
    reader; tolerant of malformed lines.

    The notes reach the brief compiler as `project_notes` - "hard constraints" - and the
    file travels with the project (reader S9·R9). So: only a line that is an object with a
    TEXT `note` (a JSON line that is a list raised TypeError here, past the two exceptions
    this caught, and that failed the run), and never one shaped like an order to the reader
    (`looks_like_instruction`, the filter the chat's copy of these notes already passes)."""
    from .calibration import looks_like_instruction

    path = runs_dir / NOTES_FILE
    if not path.is_file():
        return []
    notes: list[str] = []
    for line in path.read_text().splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        note = entry.get("note") if isinstance(entry, dict) else None
        if isinstance(note, str) and note.strip() and not looks_like_instruction(note):
            notes.append(note)
    return notes[-NOTES_FOR_BRIEF:]


async def run_passes(engine: EngineClient, project_dir: Path, brief: str,
                     agent: Agent, max_passes: int = MAX_PASSES,
                     gate: Optional[GateFn] = None,
                     emit: Optional[EmitFn] = None,
                     drain_instructions: Optional[Callable[[], list[str]]] = None,
                     reference_scope: str = "master",
                     stage: Optional[str] = None,
                     should_cancel: Optional[Callable[[], bool]] = None,
                     edit_state: Optional[dict] = None,
                     daw_invoke: Optional[Callable[[str, dict], Any]] = None,
                     ) -> SessionReport:
    """Run the gated pass loop against an already-connected engine.

    Always terminates: targets met, diminishing returns, or pass budget.

    `reference_scope` (Phase 25) shapes reference briefs only: "master"
    (default) is the full target set on the master render; "mix" is band
    goals measured pre-master, with master-targeted actions stripped from
    agent bundles — a mix stage must not be "fixed" by mastering moves.
    ("both" never reaches here: the gateway sequences a mix run then a
    master run on one agent.) `stage` labels this run's events in the feed.
    """
    gate = gate or policy_gate("manual")
    emit = emit or _no_emit

    report = SessionReport(stop_reason="pass budget exhausted", passes_run=0)
    runs_dir = project_dir / "runs"
    runs_dir.mkdir(exist_ok=True)
    runs_path = runs_dir / f"{time.strftime('%Y%m%dT%H%M%S')}.jsonl"
    report.runs_path = str(runs_path)

    metrics = await engine.render_and_analyze()
    # Phase 39 (intent understanding): durable notes + taste shape free-form
    # brief compilation, and stem tags let the brief name instruments by role
    # ("the vocal") not just by id. All cheap reads, resolved before targets.
    notes = _load_notes(runs_dir)
    taste = taste_summary(runs_dir)
    try:
        live_state = await engine.call("get_state")
        stem_tags = {s["stem_id"]: s.get("tags", [])
                     for s in live_state.get("stems", [])}
    except RuntimeError:
        stem_tags = {}
    mix_scope = False  # Phase 25: measure pre-master + strip master moves
    if ref_specs := reference_specs_from_brief(brief):
        # "sit like this record": goals come from the reference's profile
        mix_scope = reference_scope == "mix"
        if len(ref_specs) == 1:
            reference = await engine.call(
                "analyze_reference", {"path": ref_specs[0]["path"]})
            targets = (reference_mix_targets(reference) if mix_scope
                       else reference_targets(reference))
        else:
            mix_sections = await engine.call(
                "analyze_sections",
                {"pre_master": True} if mix_scope else {})
            reference = await engine.call(
                "analyze_reference_set",
                {"references": ref_specs, "mix_sections": mix_sections})
            targets = (reference_set_mix_targets(reference) if mix_scope
                       else reference_set_targets(reference))
    else:
        # A brief that names an instrument compiles to targets on THAT stem, which
        # need the stem's own current band energy as their anchor. SessionAnalysis
        # doesn't carry per-stem bands (adding them would render every stem on every
        # analyze_session call), so resolve first and fetch just the one.
        stem_analysis = None
        named = resolve_stem(brief, metrics["analysis"]["stems"], stem_tags)
        if named is not None:
            try:
                stem_analysis = await engine.call("analyze_stem",
                                                  {"stem_id": named})
            except RuntimeError:
                pass    # unmeasurable stem — no anchor, so it falls to the LLM
        try:
            targets = targets_from_brief(brief, metrics["analysis"]["stems"],
                                         metrics["analysis"], stem_tags,
                                         stem_analysis)
        except ValueError:
            # Phase 11: free-form briefs compile through the agent (one
            # structured turn), validated against the metric grammar —
            # the deterministic regex table stays the fast path, and the
            # no-LLM path is unchanged (RuleAgent has no compiler).
            compile_brief = getattr(agent, "compile_brief", None)
            if compile_brief is None:
                raise
            raw = await anyio.to_thread.run_sync(
                compile_brief, brief, metrics["analysis"], notes, taste)
            if not raw:
                raise
            drop_warnings: list[str] = []
            targets = compiled_targets(raw, metrics["analysis"]["stems"],
                                       warnings=drop_warnings)
            report.warnings.extend(drop_warnings)
        else:
            report.warnings.extend(brief_notes(brief))   # e.g. "-30 LUFS ... aimed at -24"
    # A loudness goal nothing can reach is held to the limiter's range and SAID, whatever
    # compiled it (reader E6·R10) - a reference's own loudness is a goal too. The two brief
    # compilers above have already held theirs, so this adds nothing to what they said.
    targets, held = hold_loudness_to_limiter(targets)
    report.warnings.extend(held)
    # Phase 39 (D): plain-English read-back so the feed shows how the brief was
    # understood, not just metric names. Falls back to the metric list when the
    # (LLM-compiled) brief matches nothing in the deterministic interpreter.
    interpretation = (interpret_brief(brief, metrics["analysis"]["stems"],
                                      stem_tags)
                      or [t.metric for t in targets])
    await emit({"type": "run_started", "brief": brief, "max_passes": max_passes,
                "targets": [t.metric for t in targets],
                "interpretation": interpretation, "stage": stage})
    await emit({"type": "metrics", "analysis": metrics["analysis"],
                "flags": metrics.get("flags", []),
                "master": metrics.get("master"),
                "bounce_path": metrics.get("bounce_path")})

    # Phase 8/13: prior runs teach the agent its own prediction error —
    # project-local records plus the cross-project store; predictions below
    # are shifted by the learned (shrunk) bias and the table rides along in
    # the agent's context. Lessons and worked examples (Phase 13) too.
    records = load_records(runs_dir)
    calibration = calibration_table(records, load_global_rows())
    lessons = load_lessons(runs_dir)
    run_exemplars = exemplars(records, [t.metric for t in targets])
    # Phase 20/39: how the human has received past proposal kinds — loaded
    # above now (taste also feeds free-form brief compilation).

    prev_distance: Optional[float] = None
    stall = 0
    history: list[dict] = []
    instructions: list[str] = []
    plan: list[dict] = []
    plan_path = runs_path.with_suffix(".plan.json")
    last_ctx_analysis: Optional[dict] = None  # Phase 14: pass-to-pass deltas
    proposed_last_pass = 0  # 2026-06 cost-control: empty pass => stop the run

    def llm_budget_exceeded() -> bool:
        """The agent's own spend verdict (LLM agents only — RuleAgent has no
        meter and never trips this)."""
        check = getattr(agent, "budget_exceeded", None)
        return bool(check()) if callable(check) else False

    async def emit_usage() -> None:
        usage = getattr(agent, "usage", None)
        if usage:
            await emit({"type": "llm_usage", **usage,
                        "budget_usd": getattr(agent, "budget_usd", None)})

    async def measure_targets() -> list[TargetState]:
        values = await engine.measure_all([t.metric for t in targets],
                                          pre_master=mix_scope)
        return [TargetState(t.metric, t.unit, values[t.metric],
                            t.distance(values[t.metric]), goal=t.goal,
                            max_value=t.max_value, min_value=t.min_value)
                for t in targets]

    def update_plan(states: list[TargetState]) -> None:
        """Check stages off as their targets land; persist next to the run
        log (Phase 12)."""
        if not plan:
            return
        by_metric = {s.metric: s for s in states}
        for stage in plan:
            stage["done"] = all(by_metric[m].distance <= 0
                                for m in stage["metrics"] if m in by_metric)
        plan_path.write_text(json.dumps(plan, indent=2))

    main_loop = asyncio.get_running_loop()   # the gateway loop the engine lives on

    def engine_call_sync(name: str, args: Optional[dict] = None):
        """The agent's investigation bridge (Phase 12): sync because agent
        turns run in a worker thread; whitelisted because mutations belong
        to the proposal lifecycle, enforced here rather than in prompts.

        Bridges to the engine via run_coroutine_threadsafe on the captured main
        loop — NOT anyio.from_thread.run. The subscription backend nests the
        planner under asyncio.run + asyncio.to_thread, so the engine tool runs on
        a plain thread with no anyio token, where from_thread.run dies ("Not
        running inside an AnyIO worker thread") and every metric read fails — the
        agent then can't investigate and submits nothing (empty pass, "no
        results"). run_coroutine_threadsafe works from ANY thread; the API path
        (called straight from the anyio worker thread) is unaffected."""
        if name not in PLANNER_ENGINE_TOOLS:
            raise ValueError(f"'{name}' is not a planner tool "
                             f"(allowed: {sorted(PLANNER_ENGINE_TOOLS)})")
        return asyncio.run_coroutine_threadsafe(
            engine.call(name, args or {}), main_loop).result()

    async def run_one_pass(pass_n: int, states: list[TargetState],
                           applied_this_pass: Optional[list[str]] = None) -> list[str]:
        """One full pass: propose -> gate -> apply -> render -> record.
        Shared by the main loop and the critic's corrective pass.

        `applied_this_pass` is the CALLER'S list when given. Nothing in this
        function was wrapped in a try/except, and every step can raise —
        EngineClient.call raises on any tool error, and the agent turns are LLM
        calls. A failure after proposal 1 applied and before proposal 2 did used
        to escape with the ids only in a local, so the caller could neither undo
        what landed nor record it: the session was left half-mutated and
        calibration lost the pass entirely."""
        nonlocal metrics, instructions, last_ctx_analysis, proposed_last_pass
        await emit({"type": "pass_started", "pass_number": pass_n,
                    "max_passes": max_passes,
                    "targets": [vars(s) for s in states],
                    "plan": [dict(stage) for stage in plan]})
        if drain_instructions is not None:
            instructions += drain_instructions()

        # get_state is a cheap read (no rendering) — the old path was a full
        # analyze_stem render per stem just to fetch applied_chain (Phase 10).
        state = await engine.call("get_state")
        chains = {s["stem_id"]: s["applied_chain"] for s in state["stems"]}
        stem_tags = {s["stem_id"]: s["tags"] for s in state["stems"]}
        # The agent sees the WHOLE path: the ordered master chain + each bus's chain.
        master_chain = state.get("master_chain", [])
        bus_chains = {b["bus_id"]: {"chain": b["applied_chain"],
                                    "members": b["members"], "sends": b.get("sends", {}),
                                    "gain_db": b["gain_db"]}
                      for b in state.get("buses", [])}

        def progress_sync(info: dict) -> None:
            """Investigation progress from the agent's worker thread onto
            the event stream — the UI must never show a frozen 'pass N'
            label while paid LLM rounds run (2026-06 cost-control)."""
            anyio.from_thread.run(
                emit, {"type": "agent_working", "pass_number": pass_n, **info})

        ctx = PassContext(pass_number=pass_n, metrics=metrics["analysis"],
                          brief=brief,
                          targets=states, stem_chains=chains,
                          master_chain=master_chain, bus_chains=bus_chains,
                          eq_headroom=state.get("eq_headroom_db") or {},
                          history=history[-2:], instructions=list(instructions),
                          calibration=calibration,
                          engine_call=engine_call_sync,
                          plan=[dict(stage) for stage in plan],
                          lessons=lessons, exemplars=run_exemplars,
                          taste=taste,
                          metrics_delta=_metrics_delta(last_ctx_analysis,
                                                       metrics["analysis"]),
                          progress=progress_sync,
                          # asked HERE, once a pass (QA 2026-10-08, D21): a chat-started run
                          # is handed the turn's getter, and each pass should read the edit
                          # as the DAW holds it then - after the last pass's moves landed
                          edit_state=live_edit(edit_state), daw_invoke=daw_invoke,
                          should_cancel=should_cancel)
        last_ctx_analysis = metrics["analysis"]

        if applied_this_pass is None:
            applied_this_pass = []
        attribution: dict[str, list] = {}  # proposal_id -> per-action deltas
        raw_predictions: dict[str, list] = {}  # proposal_id -> raw impact
        wants_attribution = bool(getattr(agent, "attribution", False))
        # Agent turns are sync (LLM SDK); run them off the event loop so the
        # gateway keeps servicing approvals/edits while the agent thinks.
        agent_t0 = time.monotonic()
        proposals = await anyio.to_thread.run_sync(agent.propose, ctx)
        timings_s = {"agent": round(time.monotonic() - agent_t0, 1)}
        if mix_scope:
            # Phase 25: a mix-scope stage never touches the master — enforced
            # here in code (busses are part of the mix and stay allowed).
            # Post-filter counts feed the empty-pass stop, so an agent that
            # only wants mastering moves ends the stage instead of looping.
            for proposal in proposals:
                kept = [a for a in proposal["actions"]
                        if a.get("stem_id") != "master"]
                if len(kept) != len(proposal["actions"]):
                    await emit({"type": "actions_filtered",
                                "pass_number": pass_n,
                                "dropped": len(proposal["actions"]) - len(kept),
                                "reason": "mix scope: master moves wait for "
                                          "the mastering stage"})
                proposal["actions"] = kept
            proposals = [p for p in proposals if p["actions"]]
        proposed_last_pass = len(proposals)

        def prepare(p: dict) -> tuple[dict, list]:
            """Calibrate and score a proposal, snapshotting the agent's OWN
            prediction first. Phase 13: bias must be learned from raw values, or
            the table feeds back on its own corrections."""
            raw = json.loads(json.dumps(p.get("predicted_impact", [])))
            p = calibrate_proposal(p, calibration, stem_tags)
            p["confidence"] = confidence_for(p, calibration, stem_tags)
            return p, raw

        apply_t0 = time.monotonic()
        for proposal in proposals[:3]:
            # Stop pressed mid-pass: don't start another propose/probe/apply.
            # Anything already applied falls through to the render below, so
            # the session's analysis stays consistent with its chains.
            if should_cancel is not None and should_cancel():
                break
            proposal, raw_impact = prepare(proposal)
            result = await engine.call("propose_actions", {"req": proposal})
            if not result["accepted"]:
                await emit({"type": "proposal_rejected", "pass_number": pass_n,
                            "proposal": proposal, "violations": result["violations"]})
                revised = await anyio.to_thread.run_sync(
                    agent.revise, proposal, result["violations"], ctx)
                if revised is None:
                    continue
                # A revision is a NEW proposal and must go through `prepare`
                # again, for two independent reasons:
                #   - it carries no `confidence` (ClaudeAgent.revise returns a
                #     bare ProposalRequest dump, whose fields are rationale /
                #     predicted_impact / actions), so reading it below raised
                #     KeyError and killed the run mid-pass, with this pass's
                #     earlier proposals already applied;
                #   - its predictions are the ones calibration must learn from.
                #     Keeping the pre-revision snapshot taught the table a bias
                #     measured against a move that never happened.
                proposal, raw_impact = prepare(revised)
                result = await engine.call("propose_actions", {"req": proposal})
                if not result["accepted"]:
                    await emit({"type": "proposal_rejected", "pass_number": pass_n,
                                "proposal": proposal,
                                "violations": result["violations"]})
                    continue  # one revision attempt, then drop (docs/04)
            proposal_id = result["proposal_id"]
            raw_predictions[proposal_id] = raw_impact
            await emit({"type": "proposal", "pass_number": pass_n,
                        "proposal_id": proposal_id,
                        "rationale": proposal["rationale"],
                        "actions": proposal["actions"],
                        "predicted_impact": proposal["predicted_impact"],
                        "confidence": proposal["confidence"]})
            if not await gate(proposal_id, proposal):
                await emit({"type": "proposal_status", "proposal_id": proposal_id,
                            "status": "skipped"})
                continue
            # Per-action attribution (Phase 12): probe each bundled action
            # alone against the pre-apply state, so calibration learns which
            # action moved which metric. Capability-gated — only
            # investigating agents opt in (probe renders aren't free).
            if (wants_attribution and proposal["predicted_impact"]
                    and 1 < len(proposal["actions"]) <= ATTRIBUTION_MAX_ACTIONS):
                metric_names = [p["metric"]
                                for p in proposal["predicted_impact"]][:8]
                try:
                    attribution[proposal_id] = [
                        {"action_index": i,
                         "deltas": {m["metric"]: m["delta"]
                                    for m in (await engine.call(
                                        "probe_actions", {"req": {
                                            "actions": [action],
                                            "metrics": metric_names,
                                        }}))["measurements"]}}
                        for i, action in enumerate(proposal["actions"])]
                except RuntimeError:
                    attribution.pop(proposal_id, None)  # never blocks an apply
            applied = await engine.call("apply_proposal", {"proposal_id": proposal_id})
            await emit({"type": "proposal_status", "proposal_id": proposal_id,
                        "status": "applied" if applied["ok"] else "skipped",
                        "message": applied.get("message", "")})
            if applied["ok"]:
                applied_this_pass.append(proposal_id)

        timings_s["apply"] = round(time.monotonic() - apply_t0, 1)

        def write_timings() -> None:
            """One runs/ line per pass saying where the time went (2026-08-09) —
            tonight's 230 s pass was undiagnosable because nothing recorded
            whether the minutes were LLM, probes, or renders. Heterogeneous
            lines are already the file's format (critic verdicts, Phase 8)."""
            with runs_path.open("a") as f:
                f.write(json.dumps({"pass": pass_n, "timings_s": timings_s})
                        + "\n")

        if not applied_this_pass:
            # Nothing changed: a render would bill time for a bit-identical
            # bounce, and there is no measured impact to record.
            history.append({"pass": pass_n, "proposals": []})
            write_timings()
            await emit({"type": "pass_timing", "pass_number": pass_n,
                        **timings_s})
            await engine.call("advance_pass")
            await emit_usage()
            return []

        render_t0 = time.monotonic()
        metrics = await engine.render_and_analyze()  # engine fills measured impact
        timings_s["render"] = round(time.monotonic() - render_t0, 1)
        write_timings()
        await emit({"type": "pass_timing", "pass_number": pass_n, **timings_s})
        await emit({"type": "metrics", "analysis": metrics["analysis"],
                    "flags": metrics.get("flags", []),
                    "master": metrics.get("master"),
                    "bounce_path": metrics.get("bounce_path")})

        pass_record = {"pass": pass_n, "proposals": []}
        for summary in await engine.call("get_proposals", {"status": "applied"}):
            if summary["proposal_id"] not in applied_this_pass:
                continue
            entry = {"pass": pass_n,
                     "proposal_id": summary["proposal_id"],
                     "rationale": summary["rationale"],
                     "targets": [vars(s) for s in states],
                     "actions": summary["actions"],  # calibration key (Phase 8)
                     "predicted_impact": summary["predicted_impact"],
                     # Phase 13: pre-calibration predictions — what bias is
                     # learned from — plus the stem tags that key the
                     # cross-project store.
                     "raw_predicted_impact": raw_predictions.get(
                         summary["proposal_id"], summary["predicted_impact"]),
                     # .get: group/route actions target busses and carry no
                     # stem_id — bare indexing crashed the pass AFTER the
                     # proposal applied (glue_rhythm_section, 2026-08-18).
                     "stem_tags": {a["stem_id"]: stem_tags.get(a["stem_id"], [])
                                   for a in summary["actions"]
                                   if a.get("stem_id") in stem_tags},
                     "measured_impact": summary["measured_impact"]}
            if per_action := attribution.get(summary["proposal_id"]):
                entry["attribution"] = per_action  # Phase 12, feeds Phase 13
            with runs_path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
            pass_record["proposals"].append(entry)
        history.append(pass_record)
        await engine.call("advance_pass")
        await emit_usage()
        return applied_this_pass

    def distance_key(states) -> tuple[int, float]:
        """How bad the mix is, as a comparable key. Lower is better.

        `sum(distance)` alone can't rank passes: an unmeasurable target contributes
        `inf`, so any pass with one is inf-vs-inf against any other. Counting the
        unmeasurable ones FIRST and summing only the finite rest makes the order
        total, and makes losing a measurement strictly worse — otherwise a pass
        could look best merely because a target stopped being measurable in it
        (the finite sum drops when a term leaves, and without the count that reads
        as an improvement). NaN counts as non-finite.

        Known edge, accepted: two states with equal inf-counts over DIFFERENT
        unmeasurable targets compare on non-identical finite sets. Rare, and the
        tie-break is still the least-bad ranking available."""
        return (sum(1 for s in states if not math.isfinite(s.distance)),
                sum(s.distance for s in states if math.isfinite(s.distance)))

    # One entry per MEASURED boundary, so a regressing run can be put back to the
    # best state it actually reached. `n_applied` indexes report.applied_proposal_ids,
    # which is append-only, so the tail past it is exactly what to undo.
    boundaries: list[dict] = []

    def record_boundary(states, pass_done: int) -> None:
        boundaries.append({"n_applied": len(report.applied_proposal_ids),
                           "pass_done": pass_done, "states": states,
                           "key": distance_key(states)})

    for pass_n in range(1, max_passes + 1):
        if should_cancel is not None and should_cancel():
            report.stop_reason = "cancelled"
            break
        report.passes_run = pass_n
        states = await measure_targets()
        record_boundary(states, pass_n - 1)   # pass 0 = the mix as the user left it
        if pass_n == 1:  # Phase 12: the planner writes the mix plan once
            plan_fn = getattr(agent, "plan", None)
            if plan_fn is not None:
                plan[:] = plan_fn(states) or []
        update_plan(states)
        total_distance = sum(s.distance for s in states)

        # Phase 48 iteration 3: pass 1 always reaches the planner. Compiled
        # targets are a lossy reading of the brief — "de-ess it" compiled to
        # targets that already measured met, and the run used to exit here
        # with ZERO actions: an explicit instruction silently dropped. The
        # planner now gets one look (its sufficiency rule makes "propose
        # nothing" the correct, cheap reply when the mix truly needs
        # nothing); from pass 2 on, met targets still stop the run.
        if all(s.distance <= 0 for s in states) and pass_n > 1:
            report.stop_reason = "targets met"
            break
        # The EQ rail is spent for everything still open (audit follow-ups 2026-09-11, item
        # 1): another pass can only re-propose what the rail refuses, or repeat a no-op.
        if pass_n > 1 and _only_eq_or_loudness_open(states):
            snap = await engine.call("get_state")
            if (spent := rail_spent(states, snap.get("eq_headroom_db") or {},
                                    snap.get("master_chain") or [])):
                report.stop_reason = "EQ rail reached"
                report.warnings.append(
                    "the EQ limit is spent where the brief still asks for more ("
                    + ", ".join(spent[:4]) + ("…" if len(spent) > 4 else "")
                    + "): +12 dB of boost and -24 dB of cut per session, per target. "
                      "Closing the rest needs a different source or arrangement, not more EQ.")
                break
        # inf distances (unmeasurable targets) make inf - inf = nan, which
        # would never trip the stall counter; treat "no finite progress
        # signal" as a stalled pass instead of looping to the budget.
        if prev_distance is not None:
            improvement = prev_distance - total_distance
            stall = 0 if math.isfinite(improvement) and improvement >= EPSILON \
                else stall + 1
            if stall >= STALL_PASSES:
                report.stop_reason = "diminishing returns"
                break
        prev_distance = total_distance

        # The pass's applied ids accumulate into the report AS THEY LAND, so a
        # failure part-way through still leaves the caller able to see (and undo)
        # what actually happened. Every step inside can raise: EngineClient.call
        # raises on any engine tool error, and the propose/revise turns are LLM
        # calls that can time out or 429 past their retry. Unwrapped, that escaped
        # run_passes entirely — the caller got an exception instead of a report,
        # advance_pass never ran (so the NEXT run started with this pass's gain
        # budget already partly spent), and the records for everything applied
        # were never written, because they are written after the render.
        landed: list[str] = []
        try:
            await run_one_pass(pass_n, states, landed)
        except Exception as exc:  # noqa: BLE001 — surfaced, never silent (docs/04)
            # WHY it failed is written where it can be read afterwards (QA 2026-10-08, D21).
            # The report kept one sentence, `str(exc)` - for a KeyError that is the key and
            # nothing else - and the brain log (stderr) got `stop=error` with no traceback:
            # a run that died on its first pass in the real app left no trace of its cause.
            # A RuntimeError is one of the loop's own sentences ("engine tool 'x' failed:
            # ..."); anything else is named by its type, which is half of what it says.
            reason = str(exc) if type(exc) is RuntimeError else f"{type(exc).__name__}: {exc}"
            print(f"[loop] pass {pass_n} failed after {len(landed)} applied move(s) - "
                  f"{reason}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            report.applied_proposal_ids += landed
            report.warnings.append(f"pass {pass_n} failed after {len(landed)} "
                                   f"applied move(s): {reason}")
            await emit({"type": "error", "pass_number": pass_n,
                        "message": f"pass {pass_n} stopped: {reason}"})
            with contextlib.suppress(Exception):
                await engine.call("advance_pass")   # don't leak the budget forward
            report.stop_reason = "error"
            break
        except BaseException:
            # A cancel (CancelledError is a BaseException) skipped the handler above
            # and its advance_pass, so a stopped run left this pass's gain budget
            # partly spent for the NEXT run's first pass — the exact leak the comment
            # above says it fixed (2026-09-01 sweep, P1). Book what landed and close
            # the pass before letting the cancellation through.
            report.applied_proposal_ids += landed
            with contextlib.suppress(Exception):
                await engine.call("advance_pass")
            raise
        report.applied_proposal_ids += landed

        # A cancel that landed mid-pass (planner round / proposal loop) must
        # read as "cancelled", not fall through to the empty-pass "nothing to
        # propose" stop below (2026-08-09).
        if should_cancel is not None and should_cancel():
            report.stop_reason = "cancelled"
            break
        # 2026-06 cost-control: an LLM pass costs real money — never spend
        # another one when the agent is out of budget or out of ideas. An
        # empty pass means "targets met or nothing sensible remains"
        # (docs/04); re-running the same investigation on identical state
        # was pure credit drain.
        if llm_budget_exceeded():
            report.stop_reason = "spend limit reached"
            break
        if proposed_last_pass == 0:
            # An empty pass is not always "the mix is fine" (2026-09-10 audit P2, 05·F9):
            # a planner that ran out of turns or time, or whose proposals never validated,
            # said nothing — and "nothing to propose" read as a clean bill of health.
            why = getattr(agent, "last_empty_reason", None)
            report.stop_reason = EMPTY_PASS_REASONS.get(why, "nothing to propose")
            if why == "declined" and (detail := getattr(agent, "last_empty_detail", "")):
                report.stop_reason += f" [{detail}]"
            break

    async def _revert_to(best: dict, why: str) -> bool:
        """Undo everything applied after `best`, newest-first. Returns True if the
        session moved. Shared by the end-of-run guard and the critic's own output."""
        nonlocal metrics
        doomed = report.applied_proposal_ids[best["n_applied"]:]
        if not doomed:
            return False
        reverted: list[str] = []
        for proposal_id in reversed(doomed):
            # Session.undo REFUSES by returning ok=False — it does not raise, and
            # EngineClient.call only raises on isError. Checking the exception alone
            # would silently "revert" nothing.
            result = await engine.call("undo_proposal", {"proposal_id": proposal_id})
            if not result.get("ok"):
                message = result.get("message", "")
                if "not 'applied'" in message:
                    continue    # already gone (a user undo landed mid-run) — fine
                # A dependency refusal means a later proposal still holds the same
                # target. Stop here and say so rather than half-reverting silently.
                report.warnings.append(
                    f"rollback stopped at {proposal_id}: {message}")
                break
            reverted.append(proposal_id)
            await emit({"type": "proposal_status", "proposal_id": proposal_id,
                        "status": "undone", "message": why})
        if not reverted:
            return False
        report.reverted_proposal_ids += reverted
        report.applied_proposal_ids = [p for p in report.applied_proposal_ids
                                       if p not in set(reverted)]
        # One render puts the bounce, the analysis and final_metrics back in
        # agreement. The web app PLAYS bounces/master.wav, so without this the
        # browser would keep playing the mix we just reverted away from. (The DAW
        # is safe either way — it only applies chains resolved after this function
        # returns — but the bounce is the deliverable on both paths.)
        metrics = await engine.render_and_analyze()
        await emit({"type": "metrics", "analysis": metrics["analysis"],
                    "flags": metrics.get("flags", []),
                    "master": metrics.get("master"),
                    "bounce_path": metrics.get("bounce_path")})
        update_plan(best["states"])
        with runs_path.open("a") as f:
            f.write(json.dumps({"rollback": {
                "reverted_proposal_ids": reverted,
                "kept_through_pass": best["pass_done"], "reason": why}}) + "\n")
        return True

    async def _rollback_if_regressed() -> None:
        """A run that ends worse than a pass it already reached puts itself back.

        The stall counter treats "got worse" exactly like "did nothing", so two
        worsening passes used to exit as "diminishing returns" — a success-shaped
        ending — with the worse mix left applied.

        Deliberately NOT run on `cancelled` (the UI promises the moves that landed
        are kept) or `error` (the state is part-applied and unmeasured; a failing
        path should not also start rendering).
        """
        if report.stop_reason in ("cancelled", "error"):
            return
        if not report.applied_proposal_ids or not boundaries:
            return
        # `spend limit reached` / `pass budget exhausted` break AFTER an applying
        # pass, so the final state was never measured — measure it now or the run
        # is judged on stale numbers. But ONLY when some earlier pass already gave
        # us somewhere to roll back to: with no measured post-pass boundary the new
        # measurement would be the sole candidate, hence also the best, so it could
        # never change the outcome. That measurement is 1-2 master renders and this
        # is the ordinary ending of a short run — Phase 10's render budget
        # (test_phase10_efficiency) is a deliberate gate, not an accident.
        if (boundaries[-1]["n_applied"] != len(report.applied_proposal_ids)
                and any(b["n_applied"] > 0 for b in boundaries)):
            record_boundary(await measure_targets(), report.passes_run)
        final = boundaries[-1]
        # Only ever roll back to a state the RUN REACHED — never to the mix as the
        # user left it. The consult's plan allowed the pre-run baseline to win, which
        # means a single pass that doesn't move the compiled metrics erases itself:
        # measured against the suite, that reverted whole 1-pass reference-match runs
        # (test_phase10_efficiency, test_phase25_reference_scope) and left `applied`
        # empty. Undoing an entire run because it didn't help is the USER's call —
        # that is what the Undo button is — not something to do silently.
        candidates = [b for b in boundaries if b["n_applied"] > 0]
        if not candidates:
            return
        # Latest of equals: undo as little as possible.
        best = min(reversed(candidates), key=lambda b: b["key"])
        if best is final:
            return
        worse = (final["key"][0] > best["key"][0]
                 or final["key"][1] - best["key"][1] >= EPSILON)
        if not worse:
            return    # within noise — keep today's benign "diminishing returns"
        kept = ("the starting mix" if best["pass_done"] == 0
                else f"pass {best['pass_done']}")
        why = (f"pass {final['pass_done']} left the mix measurably worse; "
               f"reverted to {kept}")
        if await _revert_to(best, why):
            report.stop_reason = f"regressed — {why}"
            report.warnings.append(why)

    await _rollback_if_regressed()

    async def _critic_and_reflect(review, cancelled) -> None:
        """The post-run critic + reflection, lifted into a function purely so the
        whole thing can be guarded in one place — see the call site below."""
        # `instructions` is augmented-assigned below, which would otherwise make it
        # a local of THIS function and shadow the run's list.
        nonlocal instructions
        # A run that applied nothing has nothing for the critic to review (2026-09-10 audit
        # P2, 05·F9): it billed a verdict and then a whole corrective planner pass after an
        # empty — often out-of-turns — pass, past the run's budget. Reflection already
        # skipped this case (Phase 51).
        # ...nor one that ERRORED (2026-09-22 audit B·F7): after a failed pass the critic and a
        # corrective planner pass still ran on the failing connection (a rate limit, an outage,
        # a bad key), each call up to 5 attempts, and the reflection recorded lessons from it.
        if (review is None or llm_budget_exceeded() or cancelled
                or not report.applied_proposal_ids or report.stop_reason == "error"):
            return
        states = await measure_targets()
        applied = [s for s in await engine.call("get_proposals", {"status": "applied"})
                   if s["proposal_id"] in report.applied_proposal_ids]
        verdict = await anyio.to_thread.run_sync(
            review, ReviewContext(brief=brief, metrics=metrics["analysis"],
                                  targets=states, applied=applied))
        if verdict is not None:
            # No corrective pass once the EQ rail is spent (follow-ups 2026-09-11, item 1): every
            # target still open is one only more EQ could move, so the pass could only propose
            # what the rail refuses — a paid planner pass on the LLM agent, for nothing.
            corrective = bool(not verdict.get("targets_met")
                              and report.stop_reason not in ("targets met", "EQ rail reached")
                              and verdict.get("suggested_followup"))
            if corrective and llm_budget_exceeded():
                # The verdict itself can cross the budget; a corrective pass is a whole
                # planner pass more (05·F9) — the budget was checked only before the critic.
                corrective = False
                report.warnings.append("the critic's corrective pass was skipped: "
                                       "spend limit reached")
            await emit({"type": "critic", "verdict": verdict,
                        "corrective_pass": corrective})
            with runs_path.open("a") as f:
                f.write(json.dumps({"critic": verdict,
                                    "stop_reason": report.stop_reason,
                                    "corrective_pass": corrective}) + "\n")
            if corrective:
                report.passes_run += 1
                instructions += [f"critic: {c}" for c in verdict.get("concerns", [])]
                if report.reverted_proposal_ids:
                    # `history` still holds the reverted moves, so without this the
                    # corrective pass reasons from a mix that no longer exists.
                    instructions.append(
                        "note: a regressing pass was reverted — the session is back "
                        "at its best measured state; do not re-apply those moves")
                before_corrective = {"n_applied": len(report.applied_proposal_ids),
                                     "pass_done": report.passes_run - 1,
                                     "states": states, "key": distance_key(states)}
                # Caller-owned list, like the main loop's `landed`: run_one_pass can
                # raise after applying proposal 1 (an engine error, a 429 past its
                # retry), and the guard around the whole critic swallows it — so the
                # applied move was audible but outside the run's undo set and every
                # rollback account (2026-09-01 sweep, P1).
                corrected: list[str] = []
                try:
                    await run_one_pass(report.passes_run, states, corrected)
                finally:
                    report.applied_proposal_ids += corrected
                states = await measure_targets()
                # The corrective pass is just another pass, and can make things
                # worse like any other. It was the one pass nothing checked.
                after = distance_key(states)
                if (after[0] > before_corrective["key"][0]
                        or after[1] - before_corrective["key"][1] >= EPSILON):
                    why = "the critic's corrective pass made it worse; reverted"
                    if await _revert_to(before_corrective, why):
                        report.stop_reason = f"regressed — {why}"
                        report.warnings.append(why)
                        states = before_corrective["states"]
                update_plan(states)
                if all(s.distance <= 0 for s in states):
                    report.stop_reason = "targets met (critic pass)"
                with runs_path.open("a") as f:
                    f.write(json.dumps({"critic_followup": {
                        "applied_proposal_ids": corrected,
                        "stop_reason": report.stop_reason}}) + "\n")
            # Phase 13: reflection — one fresh turn distills the run into
            # lessons future runs on this project load at start. Phase 51:
            # skipped when the run applied nothing — there is no outcome to
            # distill, and the turn was a full billed spawn on every empty run.
            reflect = getattr(agent, "reflect", None)
            if reflect is not None and not report.applied_proposal_ids:
                reflect = None
            if reflect is not None:
                run_lessons = await anyio.to_thread.run_sync(
                    reflect, brief, verdict, report.passes_run,
                    report.stop_reason)
                append_lesson(runs_dir, brief, run_lessons or [])

    # ---- critic pass (Phase 8): fresh-eyes verdict + at most one corrective
    # pass, gated like any other (docs/04, docs/05). Skipped over budget —
    # the verdict and its corrective pass are paid turns too — and on cancel
    # (the user asked to stop; the critic is another agent turn).
    # Guarded as a whole: everything below runs AFTER the mix is applied and
    # rendered, so an exception here used to throw away a completed run — the
    # caller got the exception instead of a SessionReport, the cross-project
    # calibration store was never updated, and the UI never saw run_finished. A
    # 429 on the critic turn past its single retry was enough to do it.
    review = getattr(agent, "review", None)
    cancelled = should_cancel is not None and should_cancel()
    try:
        await _critic_and_reflect(review, cancelled)
    except Exception as exc:  # noqa: BLE001 — never lose a finished run to the critic
        report.warnings.append(f"critic/reflection skipped: {exc}")
        await emit({"type": "error",
                    "message": f"the critic pass failed ({exc}) — the mix itself "
                               "is applied and unaffected"})

    # Phase 13: fold this run's record into the cross-project store so the
    # learned prediction error follows the agent to the next project.
    update_global_store(load_records_file(runs_path))

    report.final_metrics = metrics["analysis"]
    await emit_usage()  # the critic/reflection turns spend after the last pass
    await emit({"type": "run_finished", "stop_reason": report.stop_reason,
                "passes_run": report.passes_run,
                "applied_proposal_ids": report.applied_proposal_ids,
                "usage": getattr(agent, "usage", None)})
    return report


async def _run(project_dir: Path, brief: str, gating: str, agent: Agent,
               max_passes: int, connect, should_cancel=None, edit_state=None,
               daw_invoke=None, emit=None) -> SessionReport:
    async with connect(project_dir) as session:
        return await run_passes(EngineClient(session), project_dir, brief, agent,
                                max_passes,
                                gate=policy_gate(gating,
                                                 runs_dir=project_dir / "runs"),
                                emit=emit,
                                should_cancel=should_cancel,
                                edit_state=edit_state, daw_invoke=daw_invoke)


def run_session(project_dir: str | Path, brief: str, gating: str = "manual",
                agent: Optional[Agent] = None, max_passes: int = MAX_PASSES,
                connect=None, should_cancel=None, edit_state=None,
                daw_invoke=None, emit=None) -> SessionReport:
    """Run a full gated mix session against a project dir. Always terminates.

    `should_cancel` (DAW) is polled at each pass boundary; when it returns True the
    run stops cleanly (the in-flight pass finishes, no further passes) — a cooperative
    cancel for the DAW's multi-minute LLM runs.

    `edit_state` (DAW awareness backbone, docs/10-daw.md) is the live Tracktion edit snapshot
    (tracks/clips/tempo/transport + the command catalog) the agent reads each pass, so it isn't
    blind to the arrangement. `daw_invoke(command, args)` lets the agent operate the DAW live
    (the daw_command tool). Both are None on the offline/benchmark path (unchanged)."""
    if agent is None:
        from .agents import ClaudeAgent
        agent = ClaudeAgent()
    return anyio.run(_run, Path(project_dir), brief, gating, agent,
                     max_passes, connect or _stdio_engine, should_cancel,
                     edit_state, daw_invoke, emit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mix agent pass-control loop")
    parser.add_argument("project_dir")
    parser.add_argument("brief", help='e.g. "stem A clarity" or "master for streaming"')
    parser.add_argument("--gating", default="manual",
                        choices=list(GATING_POLICIES))
    parser.add_argument("--max-passes", type=int, default=MAX_PASSES)
    parser.add_argument("--rule-agent", action="store_true",
                        help="use the deterministic no-LLM agent (smoke runs)")
    args = parser.parse_args()
    agent = RuleAgent() if args.rule_agent else None
    report = run_session(args.project_dir, args.brief, gating=args.gating,
                         agent=agent, max_passes=args.max_passes)
    print(f"\nstopped: {report.stop_reason} after {report.passes_run} pass(es)")
    print(f"applied: {report.applied_proposal_ids or 'nothing'}")
    print(f"runs log: {report.runs_path}")


if __name__ == "__main__":
    main()
