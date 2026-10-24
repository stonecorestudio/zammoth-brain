"""Agent turns for the pass-control loop (docs/04-agent-loop.md).

An Agent looks at compact metrics + targets and returns <= 3 proposal requests
(dicts shaped like the engine's ProposalRequest — the engine validates them).
ClaudeAgent is the production implementation; RuleAgent is the deterministic
no-LLM agent used by tests and smoke runs.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

from .engine_models import (BAND_EDGES_HZ, BAND_LABELS, LIMITS, STEM_METRIC_RE,
                            ProposalRequest)
from . import llm_models
from .knowledge import BOOK_MISSING_NOTICE, book_loaded as _book_loaded
from .knowledge import get_block as _knowledge_block
from .targets import clamp_loudness_goal
from .llm_backend import (active_backend, client_kwargs, cloud_token,
                          max_retries_for, metering_headers, raise_friendly)

# Per-run LLM spend ceiling (2026-06 cost-control work): the loop stops the
# run cleanly at this estimate instead of draining credits on a brief that
# can't converge (reference matching is the worst case — some band targets
# are simply unreachable). Estimated client-side from the price table below.
RUN_BUDGET_USD_DEFAULT = 5.0
RUN_BUDGET_ENV = "MIX_AGENT_RUN_BUDGET_USD"

# Prices live in ONE table now (`llm_models`, 2026-09-22). The old substring table here priced
# every "opus" at $5/$25 (Opus 5.5 is $4/$20), every "sonnet" at $3/$15 (Sonnet 5 is $2/$10),
# and had no "fable" row at all — so a Fable run fell back to the Opus rate and its budget
# counted HALF of what the user's key was actually billed.

# A single degraded API call must never stall a pass for the SDK default of
# 10 minutes x 3 attempts — the run looks hung while billing continues.
LLM_TIMEOUT_S = 180.0   # per HTTP READ; 300 let one stalled call hold a pass
                        # (and the user's Stop) for 5-10 min (2026-08-09)
LLM_MAX_RETRIES = 1
# A connect that has not happened in 10 s is not going to; the read timeout above is between
# stream events now, so a healthy long generation never trips it (2026-09-22 audit B·F1).
LLM_CONNECT_TIMEOUT_S = 10.0
# How often an in-flight call checks the user's Stop (the chat and the subscription path agree).
STOP_POLL_S = 0.2


class LLMTurnEnded(RuntimeError):
    """A model call that ended without a usable answer (2026-09-22 audit B·F4/F5): declined by
    Anthropic's safety classifier (`stop_reason: "refusal"` — HTTP 200, content empty or a
    partial the API says to discard), cut off (`max_tokens` / `model_context_window_exceeded`),
    or a structured reply that does not parse. Its words are what the user reads."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(reason + (f" [{detail}]" if detail else ""))
        self.reason = reason
        self.detail = detail


def ended_reason(response: Any) -> Optional[LLMTurnEnded]:
    """Why `response` is not a usable answer, or None when it is."""
    stop = getattr(response, "stop_reason", None)
    if stop == "refusal":
        details = getattr(response, "stop_details", None)
        category = (details.get("category") if isinstance(details, dict)
                    else getattr(details, "category", None))
        ended = LLMTurnEnded("the model declined this (Anthropic's safety check)",
                             str(category or ""))
        ended.no_retry = True   # the same brief is declined again (llm_backend.retry_helps)
        return ended
    if stop in ("max_tokens", "model_context_window_exceeded"):
        return LLMTurnEnded("the model ran out of room before it finished — lower the effort "
                            "level, or ask again", stop)
    return None
# Wall-clock budget for one pass's whole investigation (all planner rounds +
# their probes). Round COUNTS alone never bounded time: 6 rounds x thinking x
# unbounded probes was tonight's 230 s pass. Checked between rounds — a round
# in flight finishes; this stops the NEXT one from opening.
INVESTIGATE_DEADLINE_S = float(
    os.environ.get("MIX_AGENT_INVESTIGATE_DEADLINE_S", "300"))


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int,
                      cache_read_tokens: int = 0,
                      cache_write_tokens: int = 0,
                      cache_write_1h_tokens: int = 0, fast: bool = False) -> float:
    """Estimated spend for one response, in USD (`llm_models.cost_usd`). An estimate is
    enough: it gates the run budget, it is not a billing record. `cache_write_1h_tokens` is
    the part of `cache_write_tokens` written at the 1-hour TTL, as the API reports it. An
    unknown model id is priced at the most expensive rate on the table, so the run-budget
    gate can only stop EARLY on it."""
    return llm_models.cost_usd(model, input_tokens, output_tokens, cache_read_tokens,
                               cache_write_tokens, cache_write_1h_tokens, fast=fast)


@dataclass
class TargetState:
    metric: str
    unit: str
    current: Optional[float]
    distance: float  # 0 = met
    goal: Optional[float] = None       # the brief's goal value, when it has one
    max_value: Optional[float] = None  # the brief's ceiling, when it has one
    min_value: Optional[float] = None  # the brief's floor (2026-06: PLR guard)


@dataclass
class PassContext:
    """What the agent sees each turn — keep it small (docs/04)."""
    pass_number: int
    metrics: dict                      # latest SessionAnalysis dump
    targets: list[TargetState]
    stem_chains: dict[str, list]       # stem_id -> applied chain
    # Phase 48 (first technique-eval finding, 2026-08-17): the planner used to
    # see ONLY the compiled targets — never the user's sentence — so a brief
    # that named a technique ("glue the rhythm bus", "parallel compression")
    # was chased as bare numbers with generic tools. The brief rides along
    # verbatim now; targets remain the measurable derivation of it.
    brief: str = ""
    # The agent must see the WHOLE signal path, not just stems: the ordered master
    # chain (its devices + values) and each group bus's chain/members/sends.
    master_chain: list = field(default_factory=list)         # ordered master devices
    bus_chains: dict[str, dict] = field(default_factory=dict)  # bus_id -> {chain, members, sends, gain_db}
    history: list[dict] = field(default_factory=list)  # last 2 passes of proposals
    instructions: list[str] = field(default_factory=list)  # mid-run user notes (UI)
    limits: dict = field(default_factory=lambda: dict(LIMITS))
    calibration: list[dict] = field(default_factory=list)  # Phase 8: own error stats
    # Phase 12: sync bridge into the engine's read/probe tools — the loop
    # whitelists what it serves; mutations stay behind the proposal
    # lifecycle. None in legacy/single-shot contexts.
    engine_call: Optional[Callable[[str, dict], Any]] = None
    plan: list[dict] = field(default_factory=list)  # mix plan, done flags live
    # Phase 13: memory — recent reflection lessons from this project's past
    # runs, and the historically best proposals matched on target kinds.
    lessons: list[dict] = field(default_factory=list)
    exemplars: list[dict] = field(default_factory=list)
    # Phase 20: taste — human approve/reject/undo counts per action kind
    # (calibration.taste_summary). Empty until the user has decided enough.
    taste: list[dict] = field(default_factory=list)
    # Phase 14: what changed since the previous pass's context (scalars +
    # per-band energy) — regression is visible at a glance, not re-derived.
    metrics_delta: Optional[dict] = None
    # 2026-06 cost-control: optional sync progress sink. The loop bridges it
    # onto the websocket so investigation rounds are visible (and worth
    # cancelling) instead of minutes of silence. Advisory — never required.
    progress: Optional[Callable[[dict], None]] = None
    # DAW awareness backbone (docs/10-daw.md): the live Tracktion edit snapshot the agent
    # reads each pass — tracks/clips/tempo/transport + the command catalog ("what can I do
    # here"). None on the offline/headless path (the agent then only sees the mix state).
    edit_state: Optional[dict] = None
    # DAW operate path: daw_invoke(command, args) -> result drives a registered DAW command
    # live (the daw_command tool). None off the DAW (the agent can't operate an edit it has none of).
    daw_invoke: Optional[Callable[[str, dict], Any]] = None
    # The composed-EQ rail's remaining room per target/band (get_state.eq_headroom_db, 2026-09-11).
    eq_headroom: dict = field(default_factory=dict)
    # 2026-08-09: the user's Stop must land mid-pass, not at the next pass
    # boundary — a pass is up to 6 LLM rounds plus probe renders, minutes of
    # ignoring the button. Polled between planner rounds and before each tool.
    should_cancel: Optional[Callable[[], bool]] = None

    def cancelled(self) -> bool:
        return self.should_cancel is not None and self.should_cancel()


# Phase 12: the mix plan's stage families, in working order — carve space
# first, then tone, then loudness (compression of the master comes last for
# the same reason it does on a console).
_PLAN_STAGES = (
    ("carve space", lambda m: m.startswith("masking(")),
    ("match tonal balance",
     lambda m: m.startswith("band_energy.")
     or m.startswith("section_band_energy.")
     or m in {"harshness_db", "mud_db", "clarity_index"}),
    ("hit loudness and ceiling",
     lambda m: m in {"lufs_i", "lufs_s_max", "lufs_range", "true_peak_dbtp",
                     "plr_db", "psr_db"}),
)


def derive_plan(targets: list[TargetState]) -> list[dict]:
    """A deterministic mix plan from the brief's targets (Phase 12): ordered
    stages, each owning its metrics; the loop checks stages off as their
    target distances reach 0 and persists the plan next to the run log."""
    plan, claimed = [], set()
    for stage_name, matches in _PLAN_STAGES:
        metrics = [t.metric for t in targets if matches(t.metric)]
        if metrics:
            plan.append({"stage": stage_name, "metrics": metrics, "done": False})
            claimed.update(metrics)
    leftovers = [t.metric for t in targets if t.metric not in claimed]
    if leftovers:
        plan.append({"stage": "remaining targets", "metrics": leftovers,
                     "done": False})
    return plan


def _match_eq_band(metric: str) -> Optional[str]:
    if metric.startswith("band_energy."):
        band = metric.removeprefix("band_energy.")
    elif metric.startswith("section_band_energy."):
        parts = metric.split(".", 2)
        band = parts[2] if len(parts) == 3 else ""
    else:
        return None
    return band if band in BAND_LABELS else None


def _has_resonance_in_band(resonances: list[dict], band: str) -> bool:
    if band not in BAND_LABELS:
        return False
    idx = BAND_LABELS.index(band)
    lo_hz, hi_hz = BAND_EDGES_HZ[idx], BAND_EDGES_HZ[idx + 1]
    return any(lo_hz <= resonance.get("freq_hz", 0) < hi_hz
               for resonance in resonances)


@dataclass
class ReviewContext:
    """What the critic sees (Phase 8): final state + what was done — NOT the
    mixing conversation. Fresh eyes by construction."""
    brief: str
    metrics: dict                 # final SessionAnalysis dump
    targets: list[TargetState]    # freshly measured
    applied: list[dict]           # applied ProposalSummary dumps from this run


class Agent(Protocol):
    def propose(self, ctx: PassContext) -> list[dict]: ...
    def revise(self, proposal: dict, violations: list[dict],
               ctx: PassContext) -> Optional[dict]: ...
    def review(self, ctx: ReviewContext) -> Optional[dict]: ...


class RuleAgent:
    """Deterministic, masking/loudness-only agent. No LLM, no surprises."""

    GAIN_STEP_DB = -2.0

    # Per-pass fraction of a band gap to take with the master match-EQ
    # (Phase 15). band_energy is relative, so the FIR's uniform level shift
    # cancels in the gaps — full-gap corrections converge in one move; what
    # remains is band-edge approximation error. (Pre-15, with ≤4 peak
    # filters, this was 0.8 to avoid share-interaction overshoot.)
    BAND_STEP = 1.0
    RESONANT_MATCH_EQ_BOOST_CAP_DB = 2.0
    HARSH_MATCH_EQ_BOOST_CAP_DB = 3.0
    HARSHNESS_BOOST_GUARD_DB = -12.0
    HARSH_MATCH_EQ_BANDS = {"presence", "brilliance", "air"}

    def propose(self, ctx: PassContext) -> list[dict]:
        proposals: list[dict] = []
        if band_proposal := self._band_match_proposal(ctx):
            proposals.append(band_proposal)
        for target in ctx.targets:
            if target.distance <= 0 or len(proposals) >= 3:
                continue
            if target.metric.startswith("masking("):
                proposals.append(self._masking_proposal(ctx, target))
            elif target.metric in ("harshness_db", "mud_db"):
                proposals.append(self._composite_proposal(ctx, target))
            elif (target.metric == "lufs_i" and target.current is not None
                    and target.goal is not None):
                if not self._limiter_already_aims(ctx, target):
                    proposals.append(self._loudness_proposal(ctx, target))
            elif (target.current is not None
                    and (m := STEM_METRIC_RE.match(target.metric))
                    and m.group(2).strip().startswith("band_energy.")):
                if (p := self._stem_band_proposal(ctx, target, m)) is not None:
                    proposals.append(p)
        return proposals

    def plan(self, targets: list[TargetState]) -> list[dict]:
        """Mix plan for the run (Phase 12) — deterministic stage derivation."""
        return derive_plan(targets)

    def reflect(self, brief: str, verdict: dict, passes_run: int,
                stop_reason: str) -> list[str]:
        """Deterministic post-run lessons (Phase 13): what future runs on
        this project should know going in."""
        if verdict.get("targets_met"):
            return [f"brief {brief!r}: targets met in {passes_run} pass(es) "
                    f"({stop_reason})"]
        return [f"brief {brief!r}: stopped '{stop_reason}' with unmet "
                f"targets — {concern}" for concern in verdict.get("concerns", [])[:3]]

    def _masking_proposal(self, ctx: PassContext, target: TargetState) -> dict:
        masker = self._masker_of(target)
        delta = max(self.GAIN_STEP_DB, -target.distance)
        return {
            "rationale": f"{target.metric} is {target.distance:+.1f} dB over target; "
                         f"pull '{masker}' down",
            "predicted_impact": [{"metric": target.metric,
                                  "predicted_delta": round(delta, 1),
                                  "unit": "dB"}],
            "actions": [{"type": "gain", "stem_id": masker,
                         "delta_db": round(delta, 1)}],
        }

    # Phase 14 composites: where the offending region lives and where to cut.
    COMPOSITE_CUTS = {
        "harshness_db": {"freq_hz": 2800, "q": 0.8},  # 2–5 kHz aggression
        "mud_db": {"freq_hz": 250, "q": 1.0},         # low-mid buildup
    }
    COMPOSITE_CUT_DB = -2.0

    def _composite_proposal(self, ctx: PassContext, target: TargetState) -> dict:
        """One master-EQ cut per pass at the composite's region center; the
        composite is energy-share minus crest, so a region cut walks it down."""
        cut = self.COMPOSITE_CUTS[target.metric]
        word = "harshness" if target.metric == "harshness_db" else "mud"
        return {
            "rationale": f"{word} is {target.distance:+.1f} dB over the brief's "
                         f"ceiling; cut {abs(self.COMPOSITE_CUT_DB):.0f} dB at "
                         f"{cut['freq_hz']} Hz on the master",
            "predicted_impact": [{"metric": target.metric,
                                  "predicted_delta": -1.0, "unit": "dB"}],
            "actions": [{"type": "eq", "stem_id": "master",
                         "band": {"freq_hz": cut["freq_hz"],
                                  "gain_db": self.COMPOSITE_CUT_DB,
                                  "q": cut["q"], "shape": "peak"}}],
        }

    # Per-stem band targets (2026-08-05). Without this the compiler could name the
    # instrument but the deterministic agent had nothing to close it with, which turns
    # a wrong-but-reachable target into an unreachable one — worse than the bug it
    # replaced. One EQ move on THAT stem, at the band's geometric centre.
    # A flat Q for every band, NOT f0/bandwidth — which would run ~0.6 in the bass to
    # ~1.4 in the low-mids. One value is deliberate: these bands are roughly octave-wide,
    # a peak at any Q in this region covers most of one, and the thing that actually
    # matters — how much band POWER a given knob setting removes — is absorbed by the
    # taper below, which was fitted with this Q in place. Change one and the other is
    # wrong.
    STEM_BAND_Q = 1.0
    # A peak TAPERS to the edges, so it attenuates less power than its knob says.
    # Measured against the benchmark fixture at STEM_BAND_Q, a band-wide peak delivers
    # about 70% of its gain in band power; over-drive by the inverse so one pass lands.
    # An empirical fit, not a derivation — test_stem_targets pins it to 2 dp.
    STEM_BAND_TAPER = 0.7

    def _stem_band_proposal(self, ctx: PassContext, target: TargetState,
                            match) -> Optional[dict]:
        stem_id = match.group(1).strip()
        band = match.group(2).strip().removeprefix("band_energy.")
        i = BAND_LABELS.index(band)
        lo, hi = BAND_EDGES_HZ[i], BAND_EDGES_HZ[i + 1]
        freq = int(round((lo * hi) ** 0.5))   # geometric centre reads as the band's middle
        # `distance` is UNSIGNED (targets.Target.distance), so it says how far, never
        # which way. A ceiling always means DOWN and a floor always means UP, but a
        # goal has to be signed against the current value — using the distance raw
        # made "the piano is darker" LIFT the very bands it was asked to lower.
        if target.max_value is not None:
            drop = -target.distance
        elif target.min_value is not None:
            drop = target.distance
        else:
            drop = target.goal - target.current
        gain = self._share_gain_db(target.current, drop)
        floor = ctx.limits.get("eq_cut_min_db", -12.0)
        ceiling = ctx.limits.get("eq_boost_max_db", 6.0)
        clamped = max(floor, min(ceiling, round(gain, 1)))
        if clamped == 0.0:
            # A residual small enough to round to 0.0 dB is a no-op EQ that can
            # never close its target — proposing it re-queued the same dead move
            # every pass (2026-08-09), and guardrails now reject it anyway.
            return None
        # If a guardrail bit, we only get the fraction of the move we were allowed.
        share = clamped / gain if gain else 1.0
        return {
            "rationale": f"{stem_id}'s {band} share is {target.distance:+.1f} dB from "
                         f"the brief's goal; {'cut' if clamped < 0 else 'lift'} "
                         f"{abs(clamped):.1f} dB at {freq} Hz on {stem_id}",
            "predicted_impact": [{"metric": target.metric,
                                  "predicted_delta": round(drop * share, 1),
                                  "unit": "dB"}],
            "actions": [{"type": "eq", "stem_id": stem_id,
                         "band": {"freq_hz": freq, "gain_db": clamped,
                                  "q": self.STEM_BAND_Q, "shape": "peak"}}],
        }

    def _share_gain_db(self, current_db: float, drop_db: float) -> float:
        """EQ gain that moves a band_energy SHARE by `drop_db`.

        band_energy is 10log10(band power / total power), so attenuating the band
        by g dB does NOT move the metric by g: the rest of the spectrum keeps its
        energy and the denominator shrinks too. With p the band's current linear
        share and s the wanted one, p' = ap/(ap+1-p) inverts to a = s(1-p)/(p(1-s)) —
        which also shows why a stem that is ALL one band (p->1) can't move its own
        share at all, and why the benchmark fixture had to be broadband."""
        p = 10.0 ** (current_db / 10.0)
        s = 10.0 ** ((current_db + drop_db) / 10.0)
        if not 0.0 < p < 1.0 or not 0.0 < s < 1.0:
            # A share at or past the whole signal has no solution — ask for the move
            # itself and let the guardrail clamp; the caller only reaches this with a
            # measured `current`, so it is a degenerate signal, not a missing one.
            return drop_db * 2.0
        return 10.0 * math.log10(s * (1.0 - p) / (p * (1.0 - s))) / self.STEM_BAND_TAPER

    @staticmethod
    def _limiter_already_aims(ctx: PassContext, target: TargetState) -> bool:
        """The master limiter already targets this goal at this ceiling: re-proposing it is a
        no-op (its makeup re-resolves at every render). The pop-master benchmark re-applied an
        identical -10 LUFS limiter three passes running for a 0.3 LU residual (follow-ups
        2026-09-11, item 1)."""
        ceiling = ctx.limits.get("master_true_peak_max_dbtp", -1.0)
        goal = clamp_loudness_goal(target.goal)[0]
        return any(a.get("type") == "limiter"
                   and abs(float(a.get("target_lufs_i", 99.0)) - goal) <= 0.1
                   and abs(float(a.get("ceiling_dbtp", 99.0)) - ceiling) <= 0.1
                   for a in (ctx.master_chain or []))

    def _loudness_proposal(self, ctx: PassContext, target: TargetState) -> dict:
        # Phase 6: loudness goals go through the master limiter — the
        # only processor allowed to push loudness (docs/03).
        ceiling = ctx.limits.get("master_true_peak_max_dbtp", -1.0)
        goal = clamp_loudness_goal(target.goal)[0]  # limiter schema range (targets.py)
        return {
            "rationale": f"master is {target.distance:.1f} LU from the "
                         f"loudness goal; limit to {goal:.0f} LUFS "
                         f"at {ceiling:.1f} dBTP",
            "predicted_impact": [{"metric": "lufs_i",
                                  "predicted_delta": round(
                                      goal - target.current, 1),
                                  "unit": "LUFS"}],
            "actions": [{"type": "limiter", "stem_id": "master",
                         "ceiling_dbtp": ceiling,
                         "target_lufs_i": goal}],
        }

    @staticmethod
    def _masker_of(target: TargetState) -> str:
        return target.metric.split(",")[1].split(")")[0].strip()

    def _band_match_proposal(self, ctx: PassContext,
                             step: Optional[float] = None) -> Optional[dict]:
        """One master match-EQ toward the reference tonal balance (Phase 15
        — replaces the Phase 7 ≤4-peak-filter approximation): every gapped
        band's correction in a single FIR action. `step` overrides the
        per-pass gap fraction (Phase 12 probing)."""
        step = self.BAND_STEP if step is None else step
        gaps = [(t, t.goal - t.current) for t in ctx.targets
                if _match_eq_band(t.metric) is not None and t.distance > 0
                and t.current is not None and t.goal is not None]
        if not gaps:
            return None
        gaps.sort(key=lambda pair: abs(pair[1]), reverse=True)
        band_gains, predicted, noted = {}, [], []
        for target, gap in gaps[:8]:  # ≤ 8 bands; ≤ 8 predicted metrics
            band = _match_eq_band(target.metric)
            if band is None:
                continue
            gain_db = max(ctx.limits["eq_cut_min_db"],
                          min(ctx.limits["eq_boost_max_db"], step * gap))
            gain_db = self._safe_match_eq_gain(ctx, band, gain_db)
            # The composed rail (2026-09-11): size to the room left on the master, per band. One
            # band past it used to refuse the WHOLE match-EQ, so bands with room stopped closing.
            room = (ctx.eq_headroom.get("master") or {}).get(band)
            if room is not None:
                gain_db = (min(gain_db, room["boost"]) if gain_db > 0
                           else max(gain_db, room["cut"]))
            if abs(gain_db) < 0.1:
                continue
            band_gains[band] = round(gain_db, 1)
            # relative band energies: the FIR's uniform shift cancels in the
            # gaps, so the expected delta is the correction itself
            predicted.append({"metric": target.metric,
                              "predicted_delta": round(gain_db, 1),
                              "unit": "dB"})
            # the gain that APPLIES, not the gap it chases — the why-line quoted +16.6 dB
            # over a +6 dB move
            noted.append(f"{band} {gain_db:+.1f} dB (gap {gap:+.1f})")
        if not band_gains:
            return None
        return {
            "rationale": "match the target tonal balance (match-EQ): "
                         + ", ".join(noted),
            "predicted_impact": predicted,
            "actions": [{"type": "match_eq", "stem_id": "master",
                         "band_gains_db": band_gains}],
        }

    def _safe_match_eq_gain(self, ctx: PassContext, band: str,
                            gain_db: float) -> float:
        """Phase 36 safety: reference matching can ask for broad boosts. When
        current ears already report harshness or a narrow resonance in that band,
        cap the boost and let later passes cut/refine instead of lighting it up."""
        if gain_db <= 0.0:
            return gain_db
        cap = ctx.limits["eq_boost_max_db"]
        harshness = ctx.metrics.get("harshness_db")
        if (band in self.HARSH_MATCH_EQ_BANDS
                and harshness is not None
                and harshness >= self.HARSHNESS_BOOST_GUARD_DB):
            cap = min(cap, self.HARSH_MATCH_EQ_BOOST_CAP_DB)
        if _has_resonance_in_band(ctx.metrics.get("resonances") or [], band):
            cap = min(cap, self.RESONANT_MATCH_EQ_BOOST_CAP_DB)
        return min(gain_db, cap)

    def revise(self, proposal: dict, violations: list[dict],
               ctx: PassContext) -> Optional[dict]:
        # One structured revision attempt (docs/04): halve every gain move.
        revised = json.loads(json.dumps(proposal))
        for action in revised["actions"]:
            if action["type"] == "gain":
                action["delta_db"] = round(action["delta_db"] / 2.0, 1)
        return revised

    def review(self, ctx: ReviewContext) -> Optional[dict]:
        """Deterministic critic verdict (Phase 8): structured, no taste."""
        unmet = [t for t in ctx.targets if t.distance > 0]
        concerns = [f"{t.metric} is {t.distance:.1f} {t.unit} from target "
                    f"(currently {t.current})" for t in unmet]
        if not ctx.applied:
            concerns.append("no proposals were applied this run")
        return {
            "targets_met": not unmet,
            "concerns": concerns,
            "suggested_followup": ("one corrective pass on: "
                                   + ", ".join(t.metric for t in unmet)
                                   if unmet else None),
        }


class ProbingRuleAgent(RuleAgent):
    """Deterministic investigating planner (Phase 12): probes candidate
    moves through the engine bridge and submits the measured winner, with
    the probe results in the rationale and probe deltas as predictions.
    The no-LLM twin of ClaudeAgent's investigation loop, and the
    benchmark's "investigating" configuration. Falls back to the parent's
    single-shot heuristics when no bridge is available."""

    attribution = True  # the loop records per-action probe deltas to runs/
    GAIN_CANDIDATES = (-2.0, -4.0, -6.0)   # masking: candidate masker cuts
    BAND_SCALES = (0.6, 0.8, 1.0)          # tonal: candidate gap fractions

    def _probe(self, ctx: PassContext, actions: list[dict],
               metrics: list[str]) -> dict[str, Optional[float]]:
        """Probe one candidate; metric -> after value."""
        result = ctx.engine_call("probe_actions", {"req": {
            "actions": actions, "metrics": metrics[:8]}})
        return {m["metric"]: m["after"] for m in result["measurements"]}

    def _masking_proposal(self, ctx: PassContext, target: TargetState) -> dict:
        if ctx.engine_call is None or target.max_value is None:
            return super()._masking_proposal(ctx, target)
        masker = self._masker_of(target)
        outcomes = []  # (step, probed excess)
        for step in self.GAIN_CANDIDATES:
            after = self._probe(ctx, [{"type": "gain", "stem_id": masker,
                                       "delta_db": step}],
                                [target.metric])[target.metric]
            if after is not None:
                outcomes.append((step, after))
        if not outcomes:
            return super()._masking_proposal(ctx, target)
        meeting = [o for o in outcomes if o[1] <= target.max_value]
        # smallest move that meets the target; otherwise the biggest cut
        step, after = (max(meeting, key=lambda o: o[0]) if meeting
                       else min(outcomes, key=lambda o: o[0]))
        probed = ", ".join(f"{s:+.0f}→{v:.1f}" for s, v in outcomes)
        return {
            "rationale": f"probed '{masker}' cuts ({probed} dB excess): "
                         f"{step:+.0f} dB lands {target.metric} at {after:.1f} "
                         f"(target ≤ {target.max_value:.1f})",
            "predicted_impact": [{"metric": target.metric,
                                  "predicted_delta": round(
                                      after - target.current, 1),
                                  "unit": "dB"}],
            "actions": [{"type": "gain", "stem_id": masker,
                         "delta_db": round(step, 1)}],
        }

    def _band_match_proposal(self, ctx: PassContext,
                             step: Optional[float] = None) -> Optional[dict]:
        if ctx.engine_call is None or step is not None:
            return super()._band_match_proposal(ctx, step)
        candidates = [(scale, super(ProbingRuleAgent, self)
                       ._band_match_proposal(ctx, step=scale))
                      for scale in self.BAND_SCALES]
        candidates = [(scale, p) for scale, p in candidates if p is not None]
        if not candidates:
            return None
        goals = {t.metric: t.goal for t in ctx.targets
                 if t.metric.startswith("band_energy.") and t.goal is not None}
        metrics = [p["metric"] for p in candidates[0][1]["predicted_impact"]]
        best = None  # (residual gap, scale, proposal, probed afters)
        for scale, proposal in candidates:
            afters = self._probe(ctx, proposal["actions"], metrics)
            if any(afters.get(m) is None for m in metrics):
                continue
            residual = sum(abs(afters[m] - goals[m]) for m in metrics)
            if best is None or residual < best[0]:
                best = (residual, scale, proposal, afters)
        if best is None:
            return candidates[0][1]  # probes failed: fall back to a candidate
        residual, scale, proposal, afters = best
        proposal["rationale"] += (
            f" — probed gap fractions {self.BAND_SCALES}: {scale} leaves the "
            f"smallest residual ({residual:.1f} dB summed)")
        # predictions become the probe-measured deltas
        currents = {t.metric: t.current for t in ctx.targets}
        proposal["predicted_impact"] = [
            {"metric": m, "predicted_delta": round(afters[m] - currents[m], 1),
             "unit": "dB"} for m in metrics]
        return proposal


SYSTEM_PROMPT = """You are an audio mix engineer operating a headless mix engine.
Each pass you see the user's brief verbatim, compact metrics, targets with
distances, the per-stem applied chains, and recent proposal history with
prediction errors. The brief is the instruction; the targets are a derived,
lossy approximation of it. When the brief names a technique or tool —
parallel compression (comp with mix < 1), a duck, glue on a group bus,
saturation for warmth/tape, a de-esser, widening (imager/band_width),
a reverb/space — EXECUTE that technique with the matching action: closing
the numeric targets some other way does not satisfy the brief.
Sufficiency: before proposing, read the applied chains — when the technique
the brief asked for is ALREADY applied and this pass's metrics confirm it is
doing its job, submit NO proposals: a pass that changes nothing is a correct
answer, and a second, deeper helping of the same technique each pass is how
mixes get worse. Adjust an existing device's settings only when a measurement
names the problem; never stack a new copy of the technique on top of a
working one, and never pile side-moves (extra EQs, comps, fader nudges onto
other stems) around a technique that already landed.
Level discipline: a technique changes character, not level. Keep each
touched stem's overall level within about 1.5 dB of where it started unless
the brief itself asks for loudness — compensate on the device's own output
trim, and after probing check the level as well as the target metric.
Measure before you aim: two techniques have a measurement that hands you the
parameter — never guess it. (1) De-essing: esses live at 5-10 kHz. Before any
deess, call analyze_phrases(stem_id) and copy freq_hz and threshold_db from
its sibilance_bursts (measured on the 5-10 kHz band — the split point and
a conservative threshold, ready to use). If it returns no bursts the esses
are quiet, not absent: still de-ess at freq_hz 5000-6500 with a
threshold_db near the stem's own lufs_s minus ~10, and NEVER below 4000 —
a 2 kHz de-esser is a melody compressor (it compresses the tune and leaves
the esses alone). A dynamic_eq cut is the alternative only
when its freq_hz is >= 5000. (2) Space via send/return: after adding a return + send, call
analyze_return(return_id) and compare its lufs to the source's — a return
sitting 12+ dB under its source is inaudible decoration, so raise the send
or the return level until it reads within ~6-10 dB of the source (a
tasteful tail), then stop. A space the brief asked for must be AUDIBLE
first and subtle second: sends cap at -3 dB (guardrail), so start the send
AT -3 dB and pull it back only if analyze_return proves it is too loud —
a send at -5 or -8 with a room IR is decoration you will not hear. If it
still isn't audible at -3, the fix is the RETURN: raise its level, or
lengthen the decay — "short room" means ~1.0-1.5 s decay (0.5-0.8 s dies
before the ear registers it as space); ir "room" is the darkest/shortest
of the three IRs and needs the most send, "plate" or "hall" at the same
decay reads as space at lower send. And a space is a SEND/RETURN, not a
0.2-mix insert on the stem — a quarter-wet insert cannot lift a tail.
(3) A low-end problem that comes and goes — boom on the swells, mud on
the loud notes — is BAND-SELECTIVE dynamics, never a full-band comp and
never a static EQ: a full-band comp pumps the constant top of the sound
down with the boom (the note's harmonics duck every swell), and a static
cut thins the steady foundation that was fine. Use mb_comp with only the
low band active (crossover_lo_hz ~100-150, low ratio 4-6, mid/high ratio
1) or a dynamic_eq cut at the boom frequency. The threshold is the whole
technique: read the swell level first (analyze_phrases level_moves with
reason too_loud, or the stem's loud-phrase lufs_s) and set threshold_db
AT that level, ~2-4 dB under the swell peak — set it above the swells and
the device never engages (a threshold 6 dB too high does exactly nothing);
set it 6+ dB below and it eats the steady low end you were told to keep.
And NO makeup on the band you are taming: the level-discipline rule
(compensate on the device's trim) does not apply to a taming move — makeup
on the low band lifts the steady lows straight back up (+1.5-2 dB) and
cancels the reduction, so the boom barely moves and the foundation you were
told to keep shifts anyway. Taming means the band comes down; leave
makeup_db 0 and let the level fall where the technique puts it.
Respond with at most
3 proposals. Each proposal bundles related actions with one rationale and a
predicted measurable impact per metric you expect to move. Prefer small moves;
guardrails are enforced server-side and listed in the context. Metric names use
the engine grammar, e.g. "lufs_i", "stem(vocals).lufs_s",
"stem(vocals).dynamic_spread_db",
"masking(vocals,synths).worst_band_excess_db".
Loudness goals go through the master limiter (type "limiter", stem_id
"master", ceiling_dbtp, target_lufs_i) — the only processor allowed to push
loudness; bus-comp makeup must stay 0 on the master. If flags include
over_limiting, do not push the target louder again: back off the limiter target,
reduce master/bus drive, or rebalance the mix before limiting. Time-confined
moves use gain with time_range_s ([start_s, end_s], stems only) after checking
analyze_sections. For phrase-level vocal rides or de-essing, call
analyze_phrases(stem_id) and use its level_moves / sibilance_bursts as range
candidates instead of whole-stem compression or de-ess.
Device freedom: gate / de-ess / transient / band_width may target the master
(stem_id "master"), not just stems/busses — e.g. master de-ess to tame 2-bus sibilance.
Dynamics hygiene: one whole-file comp (and one mb_comp) per stem — a new
one REPLACES the existing entry, which is how you adjust settings; check
the applied chain before any dynamics move and never try to stack. Stem
comp makeup draws from the same gain budget as gain moves. Choose comp
settings by intent: vocal leveling (ratio ~2.5, attack ~8 ms, release
~120 ms), transient/peak control (ratio ~3, attack ~2 ms, release ~80 ms),
bass density (ratio ~3, attack ~20 ms, release ~180 ms), or gentle density
(ratio ~2, attack ~12 ms, release ~180 ms, optional mix ~0.8). Predict
dynamic-spread changes with stem(<id>).dynamic_spread_db when that is the
intended movement.
Group busses: you own routing. Group related stems with a
group action — {type:"group", bus_id, members} creates the bus and routes
the members — then glue it: prefer ONE bus move (comp threshold_db >= -30,
ratio 2-4, attack 1-3 ms, release 100-200 ms; mix 0.3-0.5 for parallel —
heavy settings blended thicken a bus without flattening it) over identical
per-stem moves across its members. Glue wants the fast attack: at 10 ms
and slower the hits pass before the comp closes and the bus's crest RISES.
Re-home a single stem with route ({type:"route", stem_id, bus_id|null};
null = straight to the master). The bus_id NAMES the family it holds and
every member must belong to that family — a bass or 808 does not go in a
bus called "drums", however low it sits; give it its own bus or leave it on
the master. Busses take the stem palette minus duck,
whole-file only; check analyze_bus before and after. A group is ON its
members' signal path, so the budgets count along it: gain on a track and gain
on its group add up against the same 6 dB per pass / 12 dB per session, and so
do their EQ moves against the composed EQ rail. Lift the group OR the track;
a group you have raised 6 dB leaves its members no gain this pass. The user's manual
channel is only volume/pan/mute/solo — every move you make is a proposal
that shows its work.
Advanced tools: use dynamic_eq for resonances or harsh/muddy
bursts that appear only when a band crosses threshold; it is a downward-only
cut on stems, busses, or master. Use returns for space + parallel processing: a
return is a BLANK track (a bus fed by sends) with a FREE device chain — add any
device to it (eq, comp, saturation, … AND return_delay / return_reverb /
return_convolution, the IR/real-space reverb), in any order, like a stem/bus
chain. A return with just a compressor is parallel/NY compression. Add devices
with return_delay/return_reverb/return_convolution (or any effect), then add
send actions from stems, busses, or even another return (cascading — a return
may only feed a LATER return, forward-only, so it can't feed back). Keep sends
conservative. Each return has a Pre/Post tap (pre_fader: Post, the
default, scales the send with the source's fader; Pre taps the source after
its devices but before its fader, so the send holds its level when the fader
moves) and a level (gain_db, the return's output
fader). Check analyze_return to hear a return + balance its wet against the dry
mix. The send level, the Pre/Post tap, and the return level are ALL shared
control — the user dials them live (set_send / return.set_prefader /
return.set_level), so treat the dry/wet balance as co-owned, not yours alone.
A return is the usual home for a space, but it is not the only one: reverb,
delay, and IR reverb are also ordinary INSERTS — pass stem_id (a stem, a group
bus, or "master") instead of return_id and set mix (dry/wet). Reach for the
insert when the effect belongs to that ONE source (a slap delay that IS the
vocal sound); reach for a return when several sources should share one space.
user_taste_decisions counts how the human received each proposal kind
(approved / rejected / undone / kept). Respect it: prefer kinds they accept, and
avoid kinds they consistently reject unless nothing else can reach the
target — then say so in the rationale. `undone` is the strongest negative — they
heard it and took it back. `kept` is a WEAK positive (the move was still standing
when they asked for the next thing), so it supports a kind but never outweighs a
rejection or an undo.
Your EQ moves on one target ADD UP: every eq band, tilt and match_eq you have
applied there composes into one response, bounded at +6 / -12 dB per pass and
+12 / -24 dB per session. eq_headroom_db says, per target and band, how much
boost and cut is still left (a target missing from it has the full rail). Size
moves inside it — a move past it is refused whole — and when a band has no room,
adjust the band already there or leave it: a gap bigger than the rail is not a
mix move (a wrong reference, the wrong stem, or material that isn't there).
When you run inside the native DAW the context also carries daw_edit_state —
the live session the user sees: tracks (name, kind, colour, mute/solo, gain,
output routing, device_count, per-device LIVE levels (devices[]: in_db /
out_db / delta_db / out_peak_db + gr_db per chain device, from the dock's
boundary meters — read them to gain-stage per device: a stage hit too hot, a
device adding uncompensated level, or the one device whose OUTPUT clips;
meaningful only while audio is playing, floors at -120 when idle), automation
lanes + automatable_targets, hosted
3rd-party plugins + their params), each
track's clips (start/length/fades/source, and any recorded takes), the tempo +
time-signature, and the
transport (position, playing, loop, punch). Read it so your mix fits the real
arrangement (clip positions, sections, routing), not only the summed metrics.
A track carrying frozen (or frozen_by: its group) is FROZEN in the DAW: it plays its
freeze file, and a proposal that targets it is rejected (frozen_track). Leave it out of
this pass and say that it is frozen - the user unfreezes it, a run cannot.
A track carrying brain_audio_stale is a Loop Deck channel that was launched onto
a different loop mid-jam: your MEASUREMENTS are of brain_audio_measured while the
user hears brain_audio_playing (the copy is remade when the transport stops).
Don't mix that track this pass on numbers you know are about the wrong audio —
say which loop you measured instead.
A track carrying brain_take_gone is a recorded track whose measured take (the
file it names) is no longer on it: with brain_take_playing it plays that file
instead (the copy is remade when the transport stops), without it the track
plays nothing at all while your render still mixes the old take in. Leave that
track alone this pass and say so.
A track carrying audio_lost has clips the user cannot hear right now: the files
it names can no longer be read (a drive unplugged, or moved, while the project
is open), while your render still mixes your own copy in. Leave that track alone
this pass and say so; plugin_files_lost is the same for a hosted plug-in's own
files, and the plug-in it happened to carries files_lost on its own row.
A MIX RUN does not operate the DAW directly: it has no daw_command tool. Every
move you make here is a PROPOSAL, so the pass guardrails, the ledger, the
per-action measurement and the regression rollback all see it (routing too: the
group and route actions land in the DAW for you). Arranging, importing, MIDI,
transport and the rest of the command surface belong to the engineer's chat,
before or after a run. If the brief needs one of those, say so in the rationale
rather than working around it."""


MAX_PLANNER_ROUNDS = 6  # investigation rounds per pass, then drop (docs/04 spirit)

# The investigating planner's tool surface (Phase 12): the engine's read
# tools + the probe sandbox + the terminal submit. The loop's bridge
# whitelists the engine names server-side; this list is presentation.
PLANNER_TOOL_DEFS = [
    # (2026-08-08) the timing/harmony/intonation ears + the split-quality read —
    # same contracts as the chat surface; kept in lockstep with
    # loop.PLANNER_ENGINE_TOOLS, which is the enforced allowlist.
    {"name": "analyze_timing",
     "description": "Onset timing against the song's grid for one stem: "
                    "bias_ms (rush/drag on structural beats), spread_ms "
                    "(jitter), swing_percent, onsets on/off grid. Read before "
                    "any timing judgement — a deliberate lean is feel, "
                    "scatter is a fault.",
     "input_schema": {"type": "object",
                      "properties": {"stem_id": {"type": "string"}},
                      "required": ["stem_id"]}},
    {"name": "analyze_tempo",
     "description": "ONE stem's own tempo and where its beat falls "
                    "(tempo_bpm, tempo_confidence, beat_offset_s). Every "
                    "other grid is the master's; nulls with a reason mean "
                    "the part has no steady pulse.",
     "input_schema": {"type": "object",
                      "properties": {"stem_id": {"type": "string"}},
                      "required": ["stem_id"]}},
    {"name": "analyze_activity",
     "description": "Where each stem comes in and drops out, to the bar: "
                    "runs of whole bars that are in / low / out, on the grid "
                    "you give (bpm, bar1_s). Read it before a ranged move "
                    "that depends on a section's edges - a section starts "
                    "where a part enters, not on a multiple of eight bars. A "
                    "run a part enters with carries enters_s and enters_beat "
                    "(1.0 = on the bar line): an entry on beat 2, 3 or 4 means "
                    "bar1_s is that many beats early.",
     "input_schema": {"type": "object",
                      "properties": {
                          "stem_ids": {"type": "array", "items": {"type": "string"}},
                          "bpm": {"type": "number"},
                          "bar1_s": {"type": "number"},
                          "beats_per_bar": {"type": "integer"}}}},
    {"name": "analyze_phase",
     "description": "What cancels when the song is folded to mono: per stem, "
                    "group and the master, each band's L/R correlation and "
                    "below_zero_pct (the share of its ~0.3 s windows below 0); "
                    "issues (a master band below 0 over the song or 20 % of "
                    "the time, one side upside down, a stem band anti-phase, a "
                    "Haas delay) with cause and fix; conflicts (two stems that "
                    "cancel each other in the lows). Read it before you widen "
                    "anything and before the mastering pass; a width move comes "
                    "back with its own phase check.",
     "input_schema": {"type": "object",
                      "properties": {
                          "stem_ids": {"type": "array", "items": {"type": "string"}}}}},
    {"name": "analyze_chords",
     "description": "Per-beat chord spans (maj/min + sevenths) for a harmonic "
                    "stem, scored against the detected key. Use to localise "
                    "harmonic mud (low voicings) or name the progression "
                    "before tonal moves on keys/guitar stems.",
     "input_schema": {"type": "object",
                      "properties": {"stem_id": {"type": "string"}},
                      "required": ["stem_id"]}},
    {"name": "analyze_notes",
     "description": "Per-note intonation for a MONOPHONIC stem (lead vocal, "
                    "bass): each held note with cents_off, worst first. "
                    "`played` lists the pitch classes the part holds, longest "
                    "first - which notes it plays, where `notes` is which are "
                    "out of tune. "
                    "Declines polyphony with a reason. Read before pitch "
                    "judgements or a tune move.",
     "input_schema": {"type": "object",
                      "properties": {"stem_id": {"type": "string"}},
                      "required": ["stem_id"]}},
    {"name": "analyze_separation",
     "description": "Artifact metrics for AI-split stems (shimmer, transient "
                    "smear, spectral holes, leakage). Size spectral_denoise "
                    "from these instead of guessing. Measures only stems the "
                    "app's own split made; the rest are listed as not_separated.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "analyze_session",
     "description": "Master-bus metrics for the current mix (lufs_i, "
                    "true_peak_dbtp, band_energy, ...).",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "analyze_stem",
     "description": "Per-stem metrics + the stem's applied chain.",
     "input_schema": {"type": "object",
                      "properties": {"stem_id": {"type": "string"}},
                      "required": ["stem_id"]}},
    {"name": "analyze_phrases",
     "description": "Phrase readout for one stem: local level "
                    "outliers for ranged gain rides and 5-10 kHz sibilant "
                    "bursts for ranged de-ess. Read this before proposing "
                    "phrase-level vocal control.",
     "input_schema": {"type": "object",
                      "properties": {"stem_id": {"type": "string"}},
                      "required": ["stem_id"]}},
    {"name": "analyze_bus",
     "description": "A group bus's metrics + chain (the member sum through "
                    "the bus chain, with gain-reduction telemetry) — check "
                    "before and after gluing a group.",
     "input_schema": {"type": "object",
                      "properties": {"bus_id": {"type": "string"}},
                      "required": ["bus_id"]}},
    {"name": "analyze_return",
     "description": "A send/return bus's metrics + chain (every active source's "
                    "send summed through the return's wet reverb/delay chain) — "
                    "your ears on a return, so you can judge + balance the wet "
                    "level against the dry mix after sending stems to it.",
     "input_schema": {"type": "object",
                      "properties": {"return_id": {"type": "string"}},
                      "required": ["return_id"]}},
    {"name": "analyze_masking",
     "description": "Per-band excess of stem_b over stem_a where stem_a is "
                    "active (positive = stem_a masked), with worst_windows "
                    "timestamps. per_channel_worst_excess_db + "
                    "spatial_release_db show the L/R picture: when release "
                    "is low and the mono excess high, probe a pan move — "
                    "separation counts here even though the mono sum can't "
                    "see it.",
     "input_schema": {"type": "object",
                      "properties": {"stem_a": {"type": "string"},
                                     "stem_b": {"type": "string"}},
                      "required": ["stem_a", "stem_b"]}},
    {"name": "analyze_sections",
     "description": "Labeled sections with per-section loudness/band energy "
                    "and stem activity, plus tempo_bpm + bar annotations "
                    "when the grid is confident. Check before time-ranged "
                    "moves; prefer range edges on bar lines.",
     "input_schema": {"type": "object",
                      "properties": {"pre_master": {"type": "boolean"}}}},
    {"name": "analyze_reference",
     "description": "Profile one reference track: loudness, dynamics, tonal "
                    "balance, per-band width, sample peak and true peak "
                    "(start_s + end_s: one stretch of the file). Read-only; "
                    "path may be wav, mp3, flac, aiff or ogg.",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "start_s": {"type": "number"},
                                     "end_s": {"type": "number"}},
                      "required": ["path"]}},
    {"name": "analyze_sample",
     "description": "ONE sound (a one-shot or a loop, any length): attack, "
                    "decay, peaks, a loudness that answers under 0.4 s, "
                    "centroid, band shares, pitch and a pitch drop, "
                    "noisiness, width; kind 'loop' with bpm and beats reads "
                    "its length and seam; sound_kind (kick, hat, boom, pad...) "
                    "adds issues against that kind. Read-only.",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "kind": {"type": "string", "enum": ["one_shot", "loop"]},
                                     "bpm": {"type": "number"},
                                     "beats": {"type": "number"},
                                     "sound_kind": {"type": "string"},
                                     "played_note": {"type": "integer"}},
                      "required": ["path"]}},
    {"name": "analyze_reference_set",
     "description": "Profile a weighted multi-reference set. Each reference "
                    "has {path, weight, roles}; roles are tonality, width, "
                    "loudness. Pass mix_sections from analyze_sections for "
                    "section-aware targets.",
     "input_schema": {"type": "object",
                      "properties": {
                          "references": {"type": "array",
                                         "items": {"type": "object"}},
                          "mix_sections": {"type": "object"}},
                      "required": ["references"]}},
    {"name": "get_state",
     "description": "Stems' chains/gains/rides/tags/sends + group busses "
                    "(members + chains + sends), wet returns, and master "
                    "chain. Cheap read, no rendering.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "probe_actions",
     "description": ("What-if sandbox: apply actions to a scratch copy, "
                     "measure metrics before/after over an optional "
                     "[start_s, end_s] range, discard. Nothing persists and "
                     "guardrails don't apply. Probe 2-3 candidate moves on "
                     "the worst window or loudest section, then submit the "
                     "measured winner — probe deltas ARE your predictions."),
     "input_schema": {"type": "object",
                      "properties": {
                          "actions": {"type": "array",
                                      "items": {"type": "object"},
                                      "maxItems": 8},
                          "metrics": {"type": "array",
                                      "items": {"type": "string"},
                                      "maxItems": 8},
                          "range_s": {"type": "array",
                                      "items": {"type": "number"},
                                      "minItems": 2, "maxItems": 2}},
                      "required": ["actions", "metrics"]}},
    {"name": "submit_proposals",
     "description": ("Terminal: submit your final proposals for this pass "
                     "(at most 3; an empty list means targets are met or "
                     "nothing sensible remains). Call exactly once, after "
                     "investigating. Each proposal: {rationale, "
                     "predicted_impact: [{metric, predicted_delta, unit}], "
                     "actions: [...]} per the action grammar in your "
                     "instructions."),
     "input_schema": {"type": "object",
                      "properties": {"proposals": {"type": "array",
                                                   "items": {"type": "object"},
                                                   "maxItems": 3}},
                      "required": ["proposals"]}},
]

INVESTIGATE_INSTRUCTIONS = """\
You investigate before you act. Each pass: read what you need (masking on
suspect pairs, sections for timing, get_state for what's already applied),
probe 2-3 candidate moves with probe_actions on the worst window or loudest
section, then call submit_proposals exactly once with the measured winner(s).
Probe deltas are your predicted_impact — commit to what you measured.

Timing and harmony are measurable, not vibes (2026-08-08): before judging a
part rushed, stiff, or muddy-because-of-voicings, read analyze_timing (bias vs
spread vs swing — quantising a good pocket is the classic mistake) or
analyze_chords (low close voicings explain mud EQ cannot fix; say so in the
rationale instead of carving low-mids forever). analyze_notes names the out-of-
tune notes on a monophonic stem — evidence for a `tune` move or for saying the
fix is a retake, never a guess.

When several stems play together, cross-track masking is usually the biggest
win: run analyze_masking on every pair that shares a range (vocal vs synth,
kick vs bass) and carve the MASKER, don't just push the maskee up. Match the
tool to the clash — a static eq cut for a constant overlap, `duck` for a
rhythmic/pulsing one (kick→bass), `dynamic_eq` when it only blooms on loud
phrases.

Action grammar for probes and proposals:
- gain: {type:"gain", stem_id, delta_db, time_range_s?: [start_s, end_s]}
  (ranged gain = a ride; stems only, never master)
- eq: {type:"eq", stem_id, band: {freq_hz, gain_db, q: 0.3..8, shape:
  peak|low_shelf|high_shelf|hpf|lpf|notch|bandpass|tilt_shelf,
  slope_db_oct?: 6|12|24|48|96 (hpf/lpf only; >6 = steep Butterworth cut)},
  channel?: stereo|mid|side|left|right, output_db?: -12..12} — stem_id "master"
  allowed. output_db is the EQ DEVICE's output trim (a full-signal post gain,
  summed across the node's eq entries): after a boost, compensate the level
  HERE — e.g. band +4 with output_db -2 — so the move changes tone, not
  loudness (a positive trim charges the gain budget). channel
  "mid"/"side" (M/S) applies the band to only the centre or the stereo sides:
  e.g. side-only high-shelf to add air without hyping the mono vocal, or a mid
  low-cut to clear the centre bass. "notch" is a full cut at freq_hz (gain_db
  ignored; q sets the width) — use it to kill a resonance, hum, or narrow ring,
  not for tone shaping. "bandpass" passes only a narrow band around freq_hz
  (gain_db ignored; q sets width) — isolating a element inside a busy stem, or
  telephone/lo-fi effects. "tilt_shelf" tips the whole spectrum low↔high in one
  move around freq_hz (gain_db = total tilt) — a broad brighten/darken instead
  of stacking shelves. Steep hpf/lpf (slope_db_oct 24-48) clears rumble or DC
  without the gentle 6 dB/oct default's slow skirt dulling what sits just above
  the cutoff (a steeper cut shifts phase more near the cutoff, not less); q
  shapes only a 12 dB/oct cut. channel left/right
  targets one stereo side directly (no M/S encode) for asymmetric stereo
  problems — rare; prefer mid/side for width-related work.
- comp: {type:"comp", stem_id, threshold_db, ratio, attack_ms, release_ms,
  makeup_db, mix?, channel?: stereo|mid|side} (makeup 0 on master, charged to the gain budget on stems;
  one whole-file comp per stem — a new one replaces it; ranged comps must not
  overlap; and the chain holds at most 4 comps in all — entries already on it
  count, so ranged ones accumulate across passes and the 5th is refused (rule
  comps_per_stem_max): widen a range or undo one). Pick settings by intent: vocal leveling ~= 2.5:1 / 8 ms / 120 ms;
  peak control ~= 3:1 / 2 ms / 80 ms; bass density ~= 3:1 / 20 ms / 180 ms;
  gentle density ~= 2:1 / 3 ms / 150 ms with mix around 0.8. The ATTACK
  decides which way the dynamics go: up to ~3 ms the comp catches the hit
  and the crest falls (denser); from ~10 ms up the hit passes first and the
  crest rises (punchier) while the level between hits comes down.
  PARALLEL (NY) compression — mix is a wet/dry CROSSFADE, not an added
  channel: at mix 0.5 half the signal is the compressed (quieter) copy, so
  with no makeup the result is simply a level DROP and nothing thickens.
  attack_ms and release_ms are literal time constants, and the render you
  measure runs the same compressor the user hears. The recipe that works:
  the FASTEST attack (<= 1 ms — at 2 ms most of each hit is through before
  the comp closes, and from 3 ms up the crest RISES: the blend gets
  punchier, not thicker) AND a FAST release (20-80 ms — a slow release
  holds the gain reduction between hits, so the blended copy stays quiet
  and the level sags; at 250 ms the same comp fails), ratio 4-6, threshold
  -20 to -24 on a dense drum stem, mix 0.4-0.7, and makeup_db 6 — the
  makeup is the technique, not an afterthought; it is what raises the body
  up toward the peaks (crest falls, level holds). Makeup is capped at 6 dB,
  so the threshold must not dig deeper than that can restore: at -26 the
  blend starts to lose level and at -30 it loses it no matter what, and
  the only "fix" the loop can find is a second comp, which loses more.
  ONE parallel comp, settled in one move: if the level reads slightly low
  afterwards the cause is release or threshold, not mix (mix 0.4-0.7 all
  hold level at these settings — raising it does not restore level); fix
  release/threshold on THAT comp or leave it, never add another.
  channel "mid"/"side" compresses only that part. For TRUE M/S (different
  compression on the mid AND the side at once) give both mid:{threshold_db,ratio,
  attack_ms,release_ms,makeup_db,mix} and side:{...} blocks (the flat params are
  then ignored) — e.g. glue the center but leave the sides open: mid 2.5:1, side 1:1.
- deess: {type:"deess", stem_id, threshold_db: -60..0, freq_hz: 2000..12000,
  ratio?: 1..10 (your guardrail — the same dynamics ceiling as a comp; the user
  knob goes to 20), attack_ms?: 0.1..20, release_ms?: 10..500, range_db?: -80..0,
  wide?: bool} — split-band de-esser: compress the band above freq_hz when it
  pokes over threshold. range_db caps the reduction depth — set -6..-12 so esses
  soften instead of lisping (the -80 default is uncapped); wide=true dips the
  WHOLE signal on ess hits (detector stays the high band) — often more
  transparent on dense vocals than carving the band. Defaults reproduce the old
  fixed 4:1 / 1 ms / 50 ms. Use analyze_phrases first for phrase de-ess; prefer
  a ranged deess over whole-stem de-essing when sibilance_bursts names a local
  problem. Same slot bound as comp: at most 4 de-essers per chain, entries
  already on it counted, so ranged ones accumulate across passes and the 5th is
  refused (rule deessers_per_stem_max) — widen a range or undo one.
- tune: {type:"tune", stem_id, cents: -100..100} — uniform pitch correction
  onto concert pitch; stems only, one per stem; size from
  stem(<id>).tuning_offset_cents (a flat/sharp source, not per-note pitchiness)
- transpose: {type:"transpose", stem_id:"master", semitones: -12..12} —
  whole-song key change; master only, one per master; nearest interval from the
  detected key.tonic to the target tonic (keeps the mode)
- pan/width: {type:"pan", stem_id, pan} / {type:"width", stem_id, width: 0..2}
  On a stem or bus, width 0 (full mono) is yours. On the MASTER your floor is
  0.5: narrowing the finished mix past half the side — or monoing it — is the
  user's call, not an automatic move (rule master_width_min). Folding only the
  LOWS stays free at any depth: a band whose TOP edge is at or below 300 Hz is
  exempt, which is imager width_low at its default crossover, utility bass_mono,
  and band_width on sub/bass. The same floor binds utility.width/mono, imager
  and band_width on the master — no device is the way round it, and they are in
  SERIES: what the master already carries and what you propose multiply (width
  0.7 with a utility at 0.7 is 0.49, refused). Narrow with one device.
- utility: {type:"utility", stem_id, gain_db?: -24..24, width?: 0..2, mono?, invert?,
  invert_left?, invert_right?, bass_mono?, bass_mono_hz?: 50..500, dc_filter?,
  balance?: -0.5..0.5 (your guardrail — and exactly 0 on the MASTER: an L/R skew
  of the finished mix is never an automatic move; the user's knob goes to ±1)}
  — Ableton-style Utility insert: a freely-placeable output/trim
  stage (DC filter → gain trim → per-channel polarity → width/mono → bass mono →
  balance). Stem/bus/master; one per chain (a new one replaces it). bass_mono folds
  the band below bass_mono_hz (~100-150) to mono — keep the lows centered/club-safe
  while the top stays wide; invert_left/right fix a single out-of-phase channel
  (unset = follow invert) — a STEM repair: on the MASTER a one-channel flip guts
  the mono sum and is refused (master_polarity_split), so flip both or neither
  there. width/mono on the master obey the 0.5 floor above — mono IS width 0
  downstream, so it is refused the same way. dc_filter cleans DC offset;
  balance is an output balance law. DISTINCT from the channel-strip fader — use it
  when the PLACEMENT in the chain matters (e.g. drive into a saturator). gain_db
  counts against the per-pass gain budget.
- limiter: {type:"limiter", stem_id:"master", ceiling_dbtp ≤ -1.0,
  target_lufs_i?, release_ms?, lookahead_ms?, channel_link?: 0..1,
  style?: modern|transparent|punchy} — the only loudness pusher; if over_limiting
  is flagged, back off/rebalance rather than pushing the target louder. `style` is a
  PRESET for release_ms+lookahead_ms (transparent = slow release/clean/less pumping;
  punchy = fast/tight/aggressive; modern = neutral) — use style OR set those two knobs
  directly, not both. channel_link 1 = stereo-linked, lower = independent L/R (wider,
  can pull the image). The ceiling is always held. Only set these when intent calls for it.
- match_eq: {type:"match_eq", stem_id, band_gains_db: {band: dB}} — one FIR
  from per-band corrections; the right tool for tonal-balance gaps. Before
  broad boosts, check resonances/harshness; cap or cut trouble bands instead of
  boosting a whistle.
- tilt: {type:"tilt", stem_id, tilt_db: -6..6, pivot_hz?, output_db?: -12..12} —
  one-knob spectral tilt around `pivot_hz` (default 1 kHz): +tilt brightens
  (lows down, highs up), -tilt warms; tilt_db moves EACH end, so the
  low-to-high span is twice it (an eq tilt_shelf's gain_db is the whole span);
  output_db is the post-tilt level trim
  (compensate a tilt that lands louder/quieter; positive charges the gain
  budget). The cleanest fix for a broadly dull or harsh balance — reach for
  it instead of stacking a low-shelf cut + high-shelf boost. Master/stem/bus;
  capped at ±6 dB. Use eq/match_eq for narrow/surgical tone.
- dynamic_eq: {type:"dynamic_eq", stem_id, freq_hz, gain_db:-12..0, q: 0.3..12,
  threshold_db, ratio, attack_ms?, release_ms?, time_range_s?} — downward-only
  surgical control for resonances/harshness/mud that blooms only on loud
  phrases; stems, busses and master allowed (ranges are stems only).
- mb_comp: {type:"mb_comp", stem_id, low/mid/high: {threshold_db, ratio,
  makeup_db, up?}, attack_ms, release_ms, crossover_lo_hz?: 30..4000,
  crossover_hi_hz?: 200..16000} — 3-band comp; crossovers default 150 Hz / 2 kHz,
  move them to fit the source. per-band makeup 0 on master; up=true is UPWARD
  comp (lift quiet detail below threshold) — not on master (pushes loudness). One
  per stem, a new one replaces it. A band with ratio 1 passes untouched, so a
  single-band move is {low:{threshold_db,ratio 4}, mid:{threshold_db 0, ratio
  1}, high:{threshold_db 0, ratio 1}} — that is how you compress ONLY the
  lows (boom control) and leave the body and top alone.
- ott: {type:"ott", stem_id, depth: 0..0.25, time?: 0.1..4, in_gain_db?: -12..12,
  out_gain_db?: -12..12, low/mid/high?: {up: 0..1, down: 0..1, gain_db?: -12..12},
  crossover_lo_hz?: 30..4000, crossover_hi_hz?: 200..16000} — OTT-style 3-band
  UPWARD+DOWNWARD compressor: each band is levelled toward an internal target,
  so quiet detail is lifted AND peaks are tamed at once. THE aggressive-density
  tool for modern pop/EDM vocals, synths and drum busses; depth is the one knob
  and you must SET it (your cap is 0.25: its upward side lifts quiet material by
  up to 24 dB x depth, so 0.25 is +6 dB, a whole pass of gain; the device's own
  default, 0.5, is refused; deeper is the user's knob). time <1 = snappier, >1 = smoother;
  per-band up/down rebalance the two sides (down-only on a band ≈ mb_comp);
  in_gain_db drives more material into the squash. Stems + busses ONLY (its
  upward side pushes loudness — never the master); one per chain, a new one
  replaces it. Prefer mb_comp for transparent/surgical band dynamics.
- imager: {type:"imager", stem_id, width_low?: 0..2, width_mid?: 0..2,
  width_high?: 0..2, crossover_lo_hz?: 30..4000, crossover_hi_hz?: 200..16000} —
  3-band stereo imager (stems/busses/master; one per chain, a new one replaces
  it). THE multiband width tool: mono the lows (width_low 0 = tight, club-safe
  bottom: a 48 dB/oct split, the low side about 40 dB down below 100 Hz at the
  default crossover, 6 dB at the crossover itself), widen the highs (width_high 1.2-1.4 = air) without touching the mids;
  crossovers default 150 Hz / 2 kHz, move them to fit the source. Unset widths
  stay 1 (untouched). Prefer this over several band_width entries. On the MASTER
  each band is judged by its OWN top edge against the 0.5 floor: width_low 0 at
  the default 150 Hz crossover is free (that is monoing the lows), width_mid and
  width_high are floored at 0.5 — push crossover_lo_hz up past 300 Hz and the low
  band loses the exemption too.
- band_width: {type:"band_width", stem_id, band, width: 0..2, lo_hz?, hi_hz?} —
  a labeled band, or pass lo_hz+hi_hz TOGETHER (20..20000, lo < hi) to scrub a
  custom one — that is how you narrow "just the 700 Hz-1.2 kHz smear" without
  rounding to a named band; prefer imager for multiband width work. The band is a
  48 dB/oct split: width 0 takes a one-octave band's side about 11-17 dB down (deeper
  at its centre, half at its edges), not to silence; a wider band goes deeper. On the MASTER
  the 0.5 floor applies unless the band's TOP edge is at or below 300 Hz (sub and
  bass are free; presence/air are not).
- spectrogram: {type:"spectrogram", stem_id} — the Spectrogram ANALYZER, and the
  only entry here that PROCESSES NOTHING: both engines treat it as a render
  no-op (the audio passes through untouched, bit-identical) and the waterfall is
  drawn from the track's live meters. It is a chain entry so it persists,
  reorders and undoes like a device and so you can see it in a chain. Stem, bus
  or master. Add it only when the user asks for a spectrum analyzer / spectrogram
  on something — never as a fix for tone or dynamics: nothing you measure moves.
- spectral_denoise: {type:"spectral_denoise", stem_id, reduction_db: 1..24,
  sensitivity_db?: 0..24} — per-frequency soft gate that repairs separation
  artifacts (musical noise, inter-stem leakage), NOT a dynamics/tone tool. Stems
  only; one per stem (a new one replaces it). Size from analyze_separation.
- eq/comp/deess take time_range_s for range-confined moves (stems only); a range
  must start before that track's audio ends (rule time_range_past_end)
- saturation: {type:"saturation", stem_id, drive_db: 1..12, output_db?: -12..12,
  curve?: tanh|warm|tube|hard, mix?: 0..1, oversample?: 1|2|4|8} — waveshaper; adds harmonic density/warmth
  and softens peaks. curve picks the character (warm=gentle, tube=asymmetric/even
  harmonics - its negative peaks come out louder and it leaves a small DC offset,
  hard=punchy clip). Driving it lowers level, so output_db is a
  post-shaper output trim (a small + restores level; charged to the gain budget).
  mix is dry/wet (parallel saturation: keep transients clean while adding grit).
  MULTIBAND (Saturn-style): give low/mid/high:{drive_db:0..12 (your guardrail
  caps the LOUDEST band at 12; the user knob goes to 24), output_db?, mix?,
  curve?} + crossover_lo_hz?:30..4000 + crossover_hi_hz?:200..16000 + mix?
  (global dry/wet) — saturate each band independently (e.g. drive the low for
  bass warmth, leave the highs clean - a band's mix 0 is clean; drive 0 still
  shapes a loud band a little). flat drive_db is ignored when bands are
  given. oversample runs the curve at N x the sample rate: a hard curve on a
  BRIGHT source (cymbals, hats, air-heavy vocals) folds its harmonics back as
  inharmonic fizz, and 4 removes it — reach for it there or at high drive_db;
  it costs a little latency (declared, so the DAW compensates) and 1 (default)
  is the untouched path. SINGLE-BAND only. Thicken a thin/sterile source or
  glue a bus; not a loudness/tone tool.
- chorus: {type:"chorus", stem_id, rate_hz?: 0.01..10, depth?: 0..1,
  delay_ms?: 1..50, mix?: 0..0.5 (your cap; the user knob goes to 1),
  voices?: 1..3, stereo_deg?: 0..180} — MODULATION: voices read the signal
  through a delay swept by an LFO. stereo_deg offsets the right channel's
  sweep and is what makes it WIDE rather than just detuned; voices 2-3 =
  ensemble (thicker, more diffuse). The fix for a THIN MONO synth, a lifeless
  doubled vocal, or a guitar that must sit wider without EQ — reach for it when
  a source is narrow or static, not when it is dull (that is EQ) or uneven
  (that is a compressor). mix 0.2-0.4 thickens; higher is an effect and smears
  transients, so keep it off drums and off a lead you want to stay a point.
  Faster rate + more depth = seasick; 0.5-1.5 Hz at depth 0.2-0.3 is the safe
  ground. No feedback here — that is the flanger, below.
- flanger: {type:"flanger", stem_id, rate_hz?: 0.01..10, depth?: 0..1,
  delay_ms?: 0.5..20, feedback?: -0.7..0.7 (your cap; the user knob goes to
  +-0.95), mix?: 0..0.5 (your cap; user knob to 1), stereo_deg?: 0..180} —
  the chorus's swept delay WITH A RETURN PATH, and that return is the whole
  device: it turns the comb's gentle dips into resonant teeth, which is the
  jet-plane sweep. The SIGN of feedback picks the sound — POSITIVE resonates
  peaks at multiples of 1/delay_ms (first notch between them, at 1/(2*delay_ms));
  NEGATIVE puts the notches ON the multiples, so the first lands an octave
  HIGHER - but only with mix above about 0.5; below it, your whole range, they
  sit between the multiples as for positive. TRAP, and it is on the defaults:
  dry+wet with NEGATIVE feedback near mix 0.5 is close to an allpass, whose
  magnitude is FLAT — mix 0.5 at feedback -0.62 is a comb 0.1 dB deep, i.e.
  INAUDIBLE. Within your caps negative is deepest at mix 0.2-0.45 (the stronger
  the feedback, the lower: -0.62 at mix 0.3 is a 20 dB comb) and flattens toward
  0.5; positive is deepest at mix 0.5; feedback 0 with mix 0.5 is total
  cancellation (the textbook comb). delay_ms is much SHORTER than a chorus: 0.5-5 ms is the range, and
  above ~10 ms the teeth crowd into the low end and it stops sounding like a
  flanger. Reach for it when a flat DI guitar/synth needs MOVEMENT rather than
  width, or a loop needs a hook on its last bar — not to widen (chorus,
  band_width) and never as tone: a comb REMOVES energy at every notch, so on a
  full mix or a bus it reads as phasiness. Stems, and keep mix at 0.3-0.5.
- auto_filter: {type:"auto_filter", stem_id, filter_type?:
  "lp"|"hp"|"bp"|"notch", slope_db_oct?: 12|24, cutoff_hz?: 20..20000, resonance?:
  0..0.3 (your cap; user knob to 1), lfo_amount_oct?: 0..3 (your cap; user knob to
  6), lfo_shape?: "sine"|"triangle"|"saw_up"|"saw_down"|"square"|"random",
  lfo_rate_hz?: 0.01..20, lfo_sync?: true|false, lfo_rate_note?: "1/32" to "8/1"
  (e.g. "1/4", "1/8T", "1/16D", "1/1" = a bar of 4/4), lfo_phase_deg?: 0..360,
  lfo_stereo_deg?: 0..180, env_amount_oct?: -3..3 (your cap; user knob to +-6),
  env_attack_ms?: 0.1..500, env_release_ms?: 10..3000, key_stem_id?: <another
  stem>, mix?: 0..1} — MODULATION: a resonant filter whose cutoff MOVES (12 or 24
  dB/oct; "bp" peaks at 0 dB). Three things push the cutoff, in octaves around
  cutoff_hz, and they add: the LFO (lfo_amount_oct either side, at lfo_rate_hz or,
  with lfo_sync, once per lfo_rate_note on the song's tempo map), and the envelope
  (env_amount_oct; negative CLOSES the filter on a hit) following the stem's own
  level - or, with key_stem_id, ANOTHER stem's (the hats opening a pad's filter).
  A track has ONE trigger: a duck on it and its auto_filter listen to the same
  one. lfo_stereo_deg offsets the right channel's sweep. resonance is the peak at
  the cutoff (0 = none; your 0.3 is about a 5 dB peak). For movement on a static
  part (a pad that breathes with the beat, a synth that opens over a bar), a
  filtered intro or breakdown, a wah, a band-pass "telephone" vocal. A moving CUT,
  not a tone fix (that is eq) - a deep sweep on a full mix reads as a mistake.
  Stems and groups (on a group, no key_stem_id); not the master.
- overdrive: {type:"overdrive", stem_id, drive_db: 0..24, filter_on?: bool,
  freq_hz?: 40..12000, q?: 0.3..8, tone?: 0..1, output_db?: -24..12, mix?: 0..1,
  oversample?: 1|2|4|8} — drive-pedal DISTORTION, harder and dirtier than saturation: an exponential
  soft-clip at drive_db, with an optional resonant band-pass pre-filter
  (filter_on + freq_hz/q — focus the drive on a band, e.g. 1-3 kHz for vocal
  grit) and a tone high shelf on the shaped signal, corner near 3 kHz (0 dark /
  0.5 neutral / 1 bright: up to +6 dB at the top, charged to the EQ boost budget). mix < 1 = parallel grit (keep the dry body, blend
  the dirt). Aggressive color for drums, bass, vocals, synths on stems/busses;
  reach for saturation instead for polite warmth/glue. Driving lowers level
  (unity small-signal shaper); output_db restores it (positive draws from the
  gain budget).
- clip: {type:"clip", stem_id, threshold_db: -6..0, hardness: 0..1,
  output_db?: -12..12, oversample?: 1|2|4|8} — pre-limiter clipper that shaves
  transient peaks (soft tanh knee at hardness 0 → hard brickwall at 1) so the
  limiter does less work and the mix gets louder + cleaner. Master, a bus, or a stem (a stem clipper is the right move on a snare/drum
  stem whose peaks are stealing the headroom).
  On the master it runs just before the limiter; on a stem or bus nothing after
  it holds a ceiling. oversample 4 runs the shaper at 4× — far less inter-sample
  overshoot and aliasing, though not a ceiling of its own (its filter rings past
  the threshold); prefer it whenever you clip hard (hardness > 0.5 or threshold
  below -3); 1 (default) is the plain memoryless path. output_db trims the
  clipped signal post-stage (− backs off the limiter, + drives it; the master's
  limiter holds the ceiling). Reach for it on a loud/club master ahead of the limiter;
  use sparingly — deeper than ~-6 dB is rejected and crushes transients. Not a
  coloration or tone tool (that's saturation/eq).
- gate: {type:"gate", stem_id, threshold_db: -80..-20, ratio: 1..20, attack_ms?,
  release_ms?, hold_ms?: 0..500, range_db?: -80..0, hysteresis_db?: 0..24,
  sc_hpf_hz?: 0..2000, sc_lpf_hz?: 200..20000} — stem/bus downward gate/expander:
  attenuates BELOW threshold to silence bleed/noise between hits, tighten loose
  drums, or cut breaths/room tails. Higher ratio = harder gate; set threshold just
  under the source's hits (the agent limit is -20: gating at program level eats
  wanted signal). hold_ms keeps it open after a hit decays (50-250 on drums stops
  chatter); range_db is the closed floor — prefer -10..-20 to DUCK bleed naturally
  instead of the -80 full-mute default; hysteresis_db (3-6) stops flutter when the
  source hovers at threshold. The SC filters shape only the DETECTOR: sc_hpf_hz
  ~150-300 keys a snare/vocal gate off kick rumble; sc_lpf_hz ~5000 ignores cymbal
  spill. NOT for the master, and not artifact repair (that's spectral_denoise).
- transient: {type:"transient", stem_id, attack_db: -6..6, sustain_db: -6..6,
  speed?: 0.25..4, output_db?: -12..12} (≥1 of attack/sustain non-zero) —
  attack/sustain shaper: boost attack for drum/percussion punch, cut sustain to
  tame boomy ring and room tails; size off stem(<id>).transient_energy_ratio /
  transient_density. speed scales the detector — >1 = a tighter transient window
  (fast percussion, hi-hats), <1 = longer (plucked bass, piano); output_db is a
  post-shaper level trim. A LIFT is level: the larger of attack_db / sustain_db,
  when positive, draws from the stem's gain budget, and so does a positive
  output_db (attack +3 then gain +4 in one pass is +7: refused). Cuts are free.
- duck: {type:"duck", stem_id, key_stem_id, threshold_db, depth_db: 1..12,
  attack_ms?, release_ms?} — sidechain stem_id under key_stem_id. The clean fix
  for RHYTHMIC masking: when analyze_masking concentrates in worst_windows that
  pulse with a transient source (kick→bass, kick→sub, lead→pad), duck the maskee
  to the masker rather than a static EQ cut (~3-6 dB depth, ~5 ms attack, release
  to the groove). Two stems cannot trigger each other (a key that already ducks
  under the target, directly or along a chain, is refused).
- mute/exclude: {type:"mute", stem_id, muted} / {type:"exclude", stem_id,
  excluded} — pull a stem from the mix, or remove it from the project entirely.
- group: {type:"group", bus_id, members:[stem_id,...]} creates the bus (if
  new) and routes the members; route: {type:"route", stem_id, bus_id|null}
  re-homes one stem. You own routing — target a bus (its id, or one a group
  made earlier in the same proposal) with the stem palette minus duck/pan/
  width, whole-file; a group + its glue comp is one atomic proposal. The
  bus_id names the family it holds, and a stem only joins a family it is a
  member of: a bass belongs in a "drums" bus no more than a vocal does, so
  give it its own bus (or none) rather than widening what "drums" means. The user
  controls only volume/pan/mute/solo.
- return_delay: {type:"return_delay", return_id, delay_ms, feedback, tone},
  return_reverb: {type:"return_reverb", return_id, decay_s, pre_delay_ms, tone}
  (algorithmic), and return_convolution: {type:"return_convolution", return_id,
  ir:"room"|"hall"|"plate", decay_s:0.2..8, pre_delay_ms, tone} (a CONVOLUTION
  reverb on a made impulse, denser + smoother than return_reverb; prefer it for
  natural ambience. Its true RT60 is decay_s x 0.55 room / 1.0 hall / 0.75 plate;
  a room has three audible early reflections, a hall a soft 20 ms onset, a plate
  is dense at once; tone is a shelf on the whole tail - the lows never move)
  create/update wet-only agent-owned returns. Then feed
  them with send: {type:"send", stem_id, return_id, gain_db:-60..-3 (your
  guardrail keeps ambience sends below unity; the schema goes to 0)}; stem_id
  may be a stem or group bus, never master. The return action must appear
  before a send to a new return in the same proposal. Use returns for
  ambience/depth, not loudness.
  All three also work as ORDINARY INSERTS: pass stem_id (a stem, a group bus,
  or "master") INSTEAD of return_id, plus mix:0..0.5 (dry/wet; that 0.5 is your
  cap — 100% wet belongs on a return). Exactly one of return_id / stem_id.
  Choose the placement the way an engineer would: a RETURN when several sources
  should share one space (the usual choice — one room, cheaper, glues the mix),
  an INSERT when the effect belongs to that one source (a slap delay that is
  part of the vocal, a reverb printed into a single pad). One reverb + one delay
  + one IR reverb per chain; re-propose the same type to adjust it."""


def extract_json(text: str) -> dict:
    """The JSON object out of a model reply: tolerate code fences and any
    prose around the outermost braces."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in reply: {text[:200]!r}")
    return json.loads(text[start:end + 1])


def compile_brief_prompt(brief: str, analysis: dict,
                         notes: Optional[list[str]] = None,
                         taste: Optional[list[dict]] = None) -> str:
    """The user prompt for ClaudeAgent.compile_brief (Phase 11; Phase 39 folds
    in durable notes + taste). Pure and module-level so the contract is unit-
    testable without the SDK."""
    from .targets import SESSION_TARGET_METRICS, STEM_TARGET_FIELDS

    payload: dict = {"brief": brief, "current_metrics": analysis}
    if notes:
        payload["project_notes"] = notes
    if taste:
        payload["user_taste_decisions"] = taste
    prompt = json.dumps(payload, indent=2) + (
        "\n\nTranslate the brief into 1-6 measurable targets for the mix "
        "loop. Allowed metric names: "
        + ", ".join(sorted(SESSION_TARGET_METRICS))
        + "; band_energy.<band> with band one of "
        + ", ".join(BAND_LABELS)
        + " (relative dB — set goals relative to current_metrics values); "
        "masking(<stem_a>,<stem_b>).worst_band_excess_db for clarity asks "
        "(stems must come from current_metrics.stems); and stem(<id>).<field> "
        "for a goal about ONE instrument, with field one of "
        + ", ".join(sorted(STEM_TARGET_FIELDS))
        + " or band_energy.<band>. Prefer a stem(...) target whenever the "
        "brief names an instrument \u2014 a per-instrument problem solved with a "
        "session-wide target changes the whole mix to fix one track. Each target needs "
        "exactly one of goal (with tolerance) or max_value. Keep moves "
        "modest: band goals within ±3 dB of current. If the brief changes "
        "loudness, include true_peak_dbtp with max_value -1.0.")
    if notes or taste:
        prompt += (" Honor project_notes as hard constraints (durable user "
                   "preferences), and lean toward target kinds the user has "
                   "accepted in user_taste_decisions over kinds they reject.")
    return prompt


class ClaudeAgent:
    """Production agent: one Claude turn per pass, validated against the
    engine's own ProposalRequest schema (not hardcoded here).

    Plan turns ask for plain JSON and validate locally rather than using
    strict structured outputs: the action union grew past the API's compiled-
    grammar size limit in Phase 7 ("The compiled grammar is too large").
    Local pydantic validation + the engine's server-side guardrails enforce
    exactly the same contract. The critic verdict is three fields, so it
    keeps strict parsing."""

    def __init__(self, model: str | None = None, effort: str | None = None,
                 fast: bool = False):
        from pydantic import BaseModel

        # Fast output (Phase 54): the user's switch for this run. `fast_notice` is what the
        # run has to say when a request asked for it and did not get it; a refusal turns it
        # off for the rest of the run (a change of speed restarts the prompt cache).
        self._fast = bool(fast)
        self.fast_notice = ""

        # Model + reasoning effort: the composer's pick, passed EXPLICITLY by every DAW path
        # that starts a run (2026-09-22 audit E·F1 — a run built from the process env ran on
        # Opus 5 at the API default whatever the composer said, after every launch and every
        # brain respawn, since the only writer of that env was a droppable notify). The env
        # (MIX_AGENT_MODEL / MIX_AGENT_EFFORT) remains the fallback for the evals and the
        # offline gateway; explicit arguments win (the benchmark + tests pin their own).
        model = (model or os.environ.get("MIX_AGENT_MODEL", "").strip()
                 or llm_models.DEFAULT_MODEL)
        self._effort = (effort or os.environ.get("MIX_AGENT_EFFORT", "")).strip()

        class AgentPlan(BaseModel):
            proposals: list[ProposalRequest]

        class CriticVerdict(BaseModel):
            targets_met: bool
            concerns: list[str]
            suggested_followup: Optional[str] = None

        class CompiledTarget(BaseModel):
            metric: str
            unit: str = "dB"
            goal: Optional[float] = None
            tolerance: float = 0.5
            max_value: Optional[float] = None

        class CompiledBrief(BaseModel):
            targets: list[CompiledTarget]

        self._plan_model = AgentPlan
        self._verdict_model = CriticVerdict
        self._brief_model = CompiledBrief
        self._model = model
        # `cloud` is production (metered control plane); `api` bills a direct
        # ANTHROPIC_API_KEY for evaluation/benchmark; `subscription` is local dev.
        self._backend = active_backend()
        self._client = None
        self._sub = None
        if self._backend == "subscription":
            from .llm_subscription import SubscriptionLLM
            self._sub = SubscriptionLLM(model)
        else:
            # `api`: the key comes from the env or the keychain (Phase 53), handed to each
            # call's client here and nowhere else; it is never written into os.environ. Read
            # once per run: a keychain read is a subprocess, and a run is ~19 calls. `cloud`
            # re-mints its short-lived token per call instead (`_client_args`).
            self._client_kwargs = client_kwargs(self._backend) if self._backend == "api" else {}
        # Cumulative spend for the run this agent instance serves (the
        # gateway builds a fresh agent per run). The loop reads `usage` for
        # the UI and stops on `budget_exceeded()`.
        self.usage = {"turns": 0, "input_tokens": 0, "output_tokens": 0,
                      "cache_read_tokens": 0, "cache_write_tokens": 0,
                      "estimated_cost_usd": 0.0}
        self.budget_usd = float(os.environ.get(RUN_BUDGET_ENV,
                                               RUN_BUDGET_USD_DEFAULT))
        # Knowledge + persona + schema are stable across every turn of every
        # pass: one system block with a cache breakpoint (Phase 10) instead of
        # re-sent schema text in each user prompt. The knowledge block goes
        # FIRST — it is the most stable text we send, so it makes the best
        # cache prefix, and the craft it carries should colour how the persona
        # below is read (see orchestrator/knowledge.py).
        self._plan_system = [
            {"type": "text", "text": _knowledge_block()},
            {"type": "text", "text": SYSTEM_PROMPT},
            {"type": "text",
             "text": ("Respond to every planning request with ONLY a JSON "
                      'object of the form {"proposals": [...]} conforming to '
                      "this JSON schema — no prose, no code fences:\n"
                      + json.dumps(AgentPlan.model_json_schema())),
             "cache_control": {"type": "ephemeral"}},
        ]
        # Phase 12: the investigating planner's system — knowledge + persona +
        # grammar + investigation discipline, all stable, one cache breakpoint.
        self._investigate_system = [
            {"type": "text", "text": _knowledge_block()},
            {"type": "text", "text": SYSTEM_PROMPT},
            {"type": "text", "text": INVESTIGATE_INSTRUCTIONS,
             "cache_control": {"type": "ephemeral"}},
        ]
        # A run plans on the system prompts built here, so whether it has the knowledge book
        # is decided here too; without it the run says so in its first note (C·F8).
        self.book_notice = "" if _book_loaded() else BOOK_MISSING_NOTICE

    @property
    def pick(self) -> tuple[str, Optional[str]]:
        """(model, the effort level it is actually sent — None where the model takes none):
        what the brain log names, so "did my pick reach the API?" has an answer."""
        return self._model, llm_models.effective_effort(
            self._model, self._effort or llm_models.DEFAULT_EFFORT)

    def budget_exceeded(self) -> bool:
        return self.usage["estimated_cost_usd"] >= self.budget_usd

    #: The run's Stop, as the loop last handed it over (each pass's ctx carries it). Revise and
    #: the structured turns have no ctx of their own.
    _should_cancel: Optional[Callable[[], bool]] = None

    def _cancelled(self) -> bool:
        return self._should_cancel is not None and bool(self._should_cancel())

    def _client_args(self) -> dict:
        return client_kwargs(self._backend) if self._backend == "cloud" else self._client_kwargs

    def _stream_call(self, kwargs: dict, should_cancel=None) -> Any:
        """One API call, streamed and raced against the user's Stop — None when Stop landed.

        It was one blocking `messages.create` with a 180 s WHOLE-RESPONSE timeout and four
        retries on the api transport: a slow but healthy generation (xhigh/max, Fable) was cut
        off and regenerated, a dead connection held a pass ~15 minutes, and nothing checked
        Stop until the call returned — so the DAW decided the brain was stuck after 45 s and
        killed it, ending every open project's turn (2026-09-22 audit B·F1). Streamed, the read
        timeout is per event; the task is polled every STOP_POLL_S and cancelled on Stop, which
        cancels the HTTP request. Its own event loop (the planner runs in a worker thread) and
        its own client, closed with the call. Usage is recorded here, complete or partial."""
        import asyncio

        import anthropic   # deferred: tests never need the SDK
        import httpx

        holder: dict = {}
        # `speed: "fast"` + its beta, when this run may ask for it; such a request goes out
        # through the SDK's beta door (the only one that takes `speed`), as the chat's does.
        speed = llm_models.fast_kwargs(self._model, self._fast, self._backend)

        async def call() -> Any:
            client = anthropic.AsyncAnthropic(
                timeout=httpx.Timeout(LLM_TIMEOUT_S, connect=LLM_CONNECT_TIMEOUT_S),
                max_retries=(llm_models.FAST_MAX_RETRIES if speed
                             else max_retries_for(self._backend, LLM_MAX_RETRIES)),
                **self._client_args())
            try:
                async def consume() -> Any:
                    door = client.beta.messages if speed else client.messages
                    async with door.stream(**kwargs, **speed) as stream:
                        holder["stream"] = stream
                        return await stream.get_final_message()

                task = asyncio.ensure_future(consume())
                while True:
                    done, _ = await asyncio.wait({task}, timeout=STOP_POLL_S)
                    if done:
                        return task.result()
                    if should_cancel is not None and should_cancel():
                        task.cancel()
                        try:
                            await task
                        except BaseException:   # noqa: BLE001 — the cancel itself
                            pass
                        return None
            finally:
                await client.close()

        try:
            response = asyncio.run(call())
        except Exception as exc:
            self._record_partial(holder.get("stream"))
            how = llm_models.fast_refusal(exc) if speed else ""
            if not how:
                raise
            # its own rate limit, or an account without the preview: the run carries on at
            # standard speed, and says so once (never the reason a pass fails)
            self._fast = False
            if how == "unavailable":
                llm_models.note_fast_unavailable(str(exc)[:200])
            self.fast_notice = ("Fast output is not available on this Anthropic account; the "
                                "run carried on at standard speed." if how == "unavailable"
                                else "Fast output hit its own rate limit; the run carried on "
                                     "at standard speed.")
            print(f"[agent] fast output {how} - continuing at standard speed: {str(exc)[:200]}",
                  file=sys.stderr)
            return self._stream_call(kwargs, should_cancel)
        if response is None:
            self._record_partial(holder.get("stream"))
            return None
        self._record_usage(response.usage)
        return response

    def _record_partial(self, stream: Any) -> None:
        """What a call that died or was stopped mid-stream billed (its message_start usage)."""
        if stream is None:
            return
        try:
            snapshot = stream.current_message_snapshot
        except Exception:   # noqa: BLE001 — no message_start yet: nothing was billed
            return
        if getattr(snapshot, "usage", None) is not None:
            self._record_usage(snapshot.usage)

    def _record_usage(self, usage: Any) -> None:
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.usage["turns"] += 1
        self.usage["input_tokens"] += usage.input_tokens
        self.usage["output_tokens"] += usage.output_tokens
        self.usage["cache_read_tokens"] += cache_read
        self.usage["cache_write_tokens"] += cache_write
        self.usage["estimated_cost_usd"] = round(
            self.usage["estimated_cost_usd"]
            + estimate_cost_usd(self._model, usage.input_tokens,
                                usage.output_tokens, cache_read, cache_write,
                                fast=getattr(usage, "speed", None) == "fast"),
            4)

    def _create(self, *, request_type: str = "agent_plan", should_cancel=None, **kwargs):
        """One model call with the run's usage meter — every turn of every kind goes through
        here (or _parse) so the budget can't be bypassed. None when the user pressed Stop in
        the middle of it. On the subscription backend the SubscriptionLLM shim stands in
        (no-tool turns only; the investigate tool loop has its own branch)."""
        if self._backend == "cloud":
            kwargs["extra_headers"] = metering_headers(request_type)
        try:
            if self._sub is not None:
                response = self._sub.create(**kwargs)
                self._record_usage(response.usage)
                return response
            return self._stream_call(kwargs, should_cancel or self._should_cancel)
        except Exception as exc:
            raise_friendly(self._backend, exc)   # 402/429/403/401 → actionable text
            raise

    def _parse(self, *, request_type: str, output_format, effort_low: bool = False, **kwargs):
        """A structured turn: the reply validated against `output_format` (a pydantic model),
        which is returned. Raises LLMTurnEnded — never an AttributeError on a missing
        `parsed_output` or a raw pydantic dump — when the model declined, ran out of room or
        wrote something that does not parse (2026-09-22 audit B·F5). The schema rides
        `output_config.format` (structured outputs) and the text is validated HERE, after the
        usage is metered — `messages.parse` raised on a truncated reply before metering it."""
        # Phase 51: small structured turns (brief compile, reflection) run at
        # LOW effort on every transport — they are JSON extraction, not mixing
        # judgment, and at the adaptive default they billed planner-grade
        # thinking on each (the baseline's 24% post-pass share). Planner +
        # critic turns never set this. A model with no effort ladder (none on
        # the picker today) accepts no effort config at all.
        if effort_low and llm_models.effective_effort(self._model, "low"):
            if self._sub is not None:
                kwargs["effort"] = "low"
            else:
                kwargs["output_config"] = {**(kwargs.get("output_config") or {}), "effort": "low"}
        if self._sub is not None:
            try:
                response = self._sub.parse(output_format=output_format, **kwargs)
            except Exception as exc:
                raise_friendly(self._backend, exc)
                raise
            self._record_usage(response.usage)
            if response.parsed_output is None:
                raise LLMTurnEnded("the model's reply could not be read")
            return response.parsed_output
        from pydantic import TypeAdapter, ValidationError

        from anthropic import transform_schema   # the PUBLIC name: a private module path moves between 0.x releases
        kwargs["output_config"] = {
            **(kwargs.get("output_config") or {}),
            "format": {"type": "json_schema",
                       "schema": transform_schema(TypeAdapter(output_format).json_schema())}}
        response = self._create(request_type=request_type, **kwargs)
        if response is None:
            raise LLMTurnEnded("stopped")
        if (ended := ended_reason(response)) is not None:
            raise ended
        text = "".join(getattr(b, "text", "") for b in response.content
                       if getattr(b, "type", None) == "text")
        try:
            return output_format.model_validate_json(text)
        except ValidationError as exc:
            raise LLMTurnEnded("the model's reply could not be read",
                               str(exc).splitlines()[0][:160]) from exc

    @staticmethod
    def _progress(ctx: PassContext, info: dict) -> None:
        if ctx.progress is None:
            return
        try:
            ctx.progress(info)
        except Exception:
            pass  # progress is advisory — it never fails a pass

    def propose(self, ctx: PassContext) -> list[dict]:
        # Why the LAST pass came back empty, when it wasn't the model's choice: "out of
        # turns", "deadline", "invalid proposals", "spend limit". The loop names it in the
        # stop reason instead of "nothing to propose" (2026-09-10 audit P2, 05·F9).
        self.last_empty_reason = None
        self.last_empty_detail = ""   # e.g. the refusal's category (2026-09-22 audit B·F4)
        self._should_cancel = ctx.should_cancel
        if ctx.engine_call is not None:
            return self._investigate(ctx)
        return self._turn(self._context_block(ctx)
                          + "\n\nPropose your next moves (or none if targets are met).")

    def plan(self, targets: list[TargetState]) -> list[dict]:
        """Mix plan for the run (Phase 12) — deterministic stage derivation."""
        return derive_plan(targets)

    @staticmethod
    def _route_tool(ctx: PassContext, name: str, args: dict) -> Any:
        """Route a planner tool call: every tool reads/probes the engine; a daw_command is
        refused (a run has no command surface — design call 2026-09-11). One
        router for both the API + subscription paths. probe_actions arg-wrapping stays caller-side
        (it differs per path), so here `args` is already in engine-call shape."""
        if name == "daw_command":
            # A mix run has no command surface (design call 2026-09-11, docs/10-daw.md): every
            # run move is a proposal, where the pass guardrails, the ledger, the measurement and
            # the regression rollback see it. The tool is not offered; this refuses it anyway, so
            # a model that invents the call still reaches nothing. It used to be a denylist
            # (RUN_REFUSED_COMMANDS), which left every new mutating key unguarded by default.
            return {"ok": False, "error": (
                "a mix run has no daw_command: every run move is a proposal here, so the "
                "guardrails and the ledger see it. Propose the move instead, or leave "
                "arranging and transport to the engineer's chat.")}
        return ctx.engine_call(name, args)

    def _investigate(self, ctx: PassContext) -> list[dict]:
        """The investigating planner (Phase 12): a bounded tool-use loop —
        read tools + probe sandbox, terminated by submit_proposals. Submitted
        proposals validate against the engine's own schema; a validation
        failure goes back as a tool error so the model can repair within the
        round budget."""
        if self._sub is not None:
            return self._investigate_subscription(ctx)
        # Moving cache breakpoints (2026-06 cost-control): the transcript
        # grows every round, and without markers each round re-reads the
        # whole thing at full input price. Marking the newest user block
        # makes rounds bill incrementally; keeping <=3 message markers plus
        # the system one stays within the API's 4-breakpoint cap.
        marked: list[dict] = []

        def mark_for_cache(block: dict) -> None:
            block["cache_control"] = {"type": "ephemeral"}
            marked.append(block)
            if len(marked) > 3:
                marked.pop(0).pop("cache_control", None)

        opening = {"type": "text",
                   "text": self._context_block(ctx)
                   + "\n\nInvestigate as needed, probe your candidates, then "
                     "call submit_proposals once."}
        mark_for_cache(opening)
        messages: list[dict] = [{"role": "user", "content": [opening]}]
        started_at = time.monotonic()
        for round_n in range(1, MAX_PLANNER_ROUNDS + 1):
            if ctx.cancelled():
                return []  # Stop pressed — the run loop reports "cancelled"
            if self.budget_exceeded():
                return []  # the loop sees the spend and stops the run
            if time.monotonic() - started_at > INVESTIGATE_DEADLINE_S:
                self._progress(ctx, {"status": "deadline",
                                     "round": round_n,
                                     "rounds_max": MAX_PLANNER_ROUNDS})
                self.last_empty_reason = "deadline"
                return []  # out of wall-clock: skip the pass rather than grind
            self._progress(ctx, {"status": "thinking", "round": round_n,
                                 "rounds_max": MAX_PLANNER_ROUNDS})
            self._should_cancel = ctx.should_cancel
            response = self._create(
                model=self._model,
                max_tokens=llm_models.max_tokens_for(
                    self._model, self._effort or llm_models.DEFAULT_EFFORT),
                **self._think_kwargs(),
                system=self._investigate_system,
                tools=PLANNER_TOOL_DEFS,
                messages=messages,
                should_cancel=ctx.should_cancel,
            )
            if response is None:
                return []  # Stop landed mid-call — the run loop reports "cancelled"
            # Branch on stop_reason BEFORE trusting the content (2026-09-22 audit B·F4): a
            # refusal's partial or a max_tokens cut can carry a tool call with truncated input
            # — a proposal cut after its first action was applied as the whole move — and a
            # thinking-only cut read as "nothing to propose", the same words as a fine mix.
            if (ended := ended_reason(response)) is not None:
                self.last_empty_reason = ("declined" if response.stop_reason == "refusal"
                                          else "cut off")
                self.last_empty_detail = ended.detail
                return []  # the reply is discarded, never stored or run
            messages.append({"role": "assistant", "content": response.content})
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                return []  # declined to act (e.g. targets already met)
            engine_tools = [b.name for b in tool_uses
                            if b.name != "submit_proposals"]
            if engine_tools:
                self._progress(ctx, {"status": "measuring", "round": round_n,
                                     "rounds_max": MAX_PLANNER_ROUNDS,
                                     "tools": engine_tools})
            results, submitted = [], None
            submits = [b for b in tool_uses if b.name == "submit_proposals"]
            for block in submits[:-1]:
                # Every tool call gets a result: two submits in one reply left the first
                # unanswered, and the repair request 400'd (2026-09-22 audit B·F11).
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": "ignored — only the last submit_proposals in a "
                                           "reply counts", "is_error": True})
            for block in tool_uses:
                if ctx.cancelled():
                    return []  # don't start another probe/render after Stop
                args = dict(block.input or {})
                if block.name == "submit_proposals":
                    if block is submits[-1]:
                        submitted = (block.id, list(args.get("proposals", []))[:3])
                    continue
                if block.name == "probe_actions":
                    args = {"req": args}  # the engine tool takes one req param
                try:
                    payload = self._route_tool(ctx, block.name, args)
                    results.append({"type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": json.dumps(payload)})
                except Exception as exc:
                    results.append({"type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": f"error: {exc}",
                                    "is_error": True})
            if submitted is not None:
                tool_use_id, proposals = submitted
                try:
                    plan = self._plan_model.model_validate(
                        {"proposals": proposals})
                    return [p.model_dump(exclude_none=True)
                            for p in plan.proposals[:3]]
                except Exception as exc:  # repair within the round budget
                    results.append({"type": "tool_result",
                                    "tool_use_id": tool_use_id,
                                    "content": f"invalid proposals: {exc}",
                                    "is_error": True})
            mark_for_cache(results[-1])
            messages.append({"role": "user", "content": results})
        # round budget exhausted: skip the pass rather than guess — and say so
        self.last_empty_reason = ("invalid proposals" if submitted is not None
                                  else "out of turns")
        return []

    def _investigate_subscription(self, ctx: PassContext) -> list[dict]:
        """The investigating planner on the subscription dev backend: the
        Claude Agent SDK drives the tool loop natively (the engine reads/probe
        + the terminal submit are in-process tools). Behaviour diverges from
        the API path's hand-rolled loop — dev only; the benchmark gate stays on
        the api backend."""
        if ctx.cancelled() or self.budget_exceeded():
            return []
        opening = (self._context_block(ctx)
                   + "\n\nInvestigate as needed, probe your candidates, then "
                     "call submit_proposals once.")
        system = "\n\n".join(block["text"] for block in self._investigate_system)

        def validate(proposals: list) -> list[dict]:
            plan = self._plan_model.model_validate(
                {"proposals": list(proposals)[:3]})
            return [p.model_dump(exclude_none=True) for p in plan.proposals[:3]]

        started_at = time.monotonic()

        def should_stop(spent) -> str:
            """Polled WHILE the SDK request runs (2026-09-10 audit P2, 05·F8) — the API
            loop checks these per round; the SDK loop used to be checked only before it."""
            if ctx.cancelled():
                return "cancelled"
            in_flight = estimate_cost_usd(
                self._model, spent.input_tokens, spent.output_tokens,
                spent.cache_read_input_tokens, spent.cache_creation_input_tokens)
            if self.usage["estimated_cost_usd"] + in_flight >= self.budget_usd:
                return "spend limit"
            if time.monotonic() - started_at > INVESTIGATE_DEADLINE_S:
                return "deadline"
            return ""

        outcome: dict = {}
        proposals, usage = self._sub.investigate(
            # the level the API path sends (`_think_kwargs`): the picker's, else the product
            # default, clamped per model — never the Claude Code CLI's own default, which is
            # not what ships (2026-09-22 parity)
            effort=llm_models.effective_effort(self._model,
                                               self._effort or llm_models.DEFAULT_EFFORT),
            system=system, opening_prompt=opening,
            planner_tool_defs=PLANNER_TOOL_DEFS,
            engine_call=lambda n, a: self._route_tool(ctx, n, a),  # a daw_command is refused
            validate=validate, max_rounds=MAX_PLANNER_ROUNDS,
            should_stop=should_stop, outcome=outcome,
            max_tokens=llm_models.max_tokens_for(
                self._model, self._effort or llm_models.DEFAULT_EFFORT))
        self._record_usage(usage)
        if not proposals:
            self.last_empty_reason = outcome.get("empty_reason")
        return proposals

    def revise(self, proposal: dict, violations: list[dict],
               ctx: PassContext) -> Optional[dict]:
        # The run's spend limit and Stop cover revisions too (2026-09-22 audit B·F8): a pass
        # could cross the budget and still make six more calls (revise + repair per refused
        # proposal), each carrying the whole system prompt.
        self._should_cancel = ctx.should_cancel
        if self.budget_exceeded() or ctx.cancelled():
            return None
        prompt = (self._context_block(ctx)
                  + "\n\nThis proposal was rejected by the guardrails:\n"
                  + json.dumps(proposal, indent=2)
                  + "\n\nViolations:\n" + json.dumps(violations, indent=2)
                  + "\n\nReturn one revised proposal that satisfies the limits, "
                    "or no proposals to drop the move.")
        revised = self._turn(prompt)
        return revised[0] if revised else None

    def compile_brief(self, brief: str, analysis: dict,
                      notes: Optional[list[str]] = None,
                      taste: Optional[list[dict]] = None) -> Optional[list[dict]]:
        """Compile a free-form brief into measurable targets (Phase 11) — the
        fallback when targets.targets_from_brief's regex table can't parse it.
        One structured-output turn; the result is validated server-side against
        the metric grammar (targets.compiled_targets), so this only has to be
        sensible, not trusted. Phase 39: durable notes + taste ride along so the
        compiled goals respect known preferences, not just later proposals."""
        response = self._parse(
            request_type="brief_compile",
            effort_low=True,
            model=self._model,
            max_tokens=4000,
            **self._think_kwargs(),
            # This is the single hardest translation in the product — free-form
            # producer language into metrics — and it used to run on one
            # sentence of guidance. The knowledge block is what makes "boxy" or
            # "no crack" resolvable instead of a guess.
            system=[{"type": "text", "text": _knowledge_block(),
                     "cache_control": {"type": "ephemeral"}},
                    {"type": "text",
                     "text": "You translate mixing briefs into measurable "
                             "engineering targets. Be conservative and "
                             "concrete. Use the reference above to map the "
                             "user's words onto the right band and metric; "
                             "when the language is genuinely ambiguous, "
                             "prefer the more conservative reading."}],
            messages=[{"role": "user",
                       "content": compile_brief_prompt(brief, analysis,
                                                       notes, taste)}],
            output_format=self._brief_model,
        )
        return [t.model_dump(exclude_none=True) for t in response.targets]

    def review(self, ctx: ReviewContext) -> Optional[dict]:
        """One fresh-context critic turn (Phase 8): final state + the applied
        record, never the mixing conversation."""
        prompt = json.dumps({
            "brief": ctx.brief,
            "final_metrics": ctx.metrics,
            "targets": [vars(t) for t in ctx.targets],
            "applied_proposals": ctx.applied,
        }, indent=2) + ("\n\nYou are reviewing a finished mix with fresh ears. "
                        "Return your verdict: are the brief's targets met, what "
                        "concerns remain, and (only if something is genuinely "
                        "unmet) a one-line suggested followup.")
        try:
            response = self._parse(
                request_type="critic",
                model=self._model,
                # Room for thinking + the verdict at any level: 4000 could all go on thinking
                # at xhigh/max, and the verdict (and the reflection) was lost (B·F5).
                max_tokens=llm_models.max_tokens_for(
                    self._model, self._effort or llm_models.DEFAULT_EFFORT, base=8000),
                **self._think_kwargs(),
                system="You are a second mix engineer reviewing a colleague's work.",
                messages=[{"role": "user", "content": prompt}],
                output_format=self._verdict_model,
            )
        except LLMTurnEnded as exc:
            print(f"[agents] critic verdict unavailable: {exc}", file=sys.stderr)
            return None
        return response.model_dump()

    def _turn(self, prompt: str) -> list[dict]:
        try:
            reply = self._text_turn(prompt)
        except LLMTurnEnded as exc:   # declined / cut off / stopped: nothing to repair
            print(f"[agents] plan turn ended: {exc}", file=sys.stderr)
            return []
        try:
            plan = self._plan_model.model_validate(extract_json(reply))
        except Exception as error:  # one structured repair attempt, then drop
            if self.budget_exceeded() or self._cancelled():
                return []
            try:
                repair = self._text_turn(
                "Your previous reply was not valid against the schema in "
                f"your instructions ({error}).\n\nPrevious reply:\n{reply}"
                "\n\nReply again with ONLY the corrected JSON object.")
            except LLMTurnEnded:
                return []
            try:
                plan = self._plan_model.model_validate(extract_json(repair))
            except Exception:
                return []
        return [p.model_dump(exclude_none=True) for p in plan.proposals[:3]]

    def _text_turn(self, prompt: str) -> str:
        response = self._create(
            request_type="agent_plan",
            model=self._model,
            max_tokens=llm_models.max_tokens_for(
                self._model, self._effort or llm_models.DEFAULT_EFFORT),
            **self._think_kwargs(),
            system=self._plan_system,
            messages=[{"role": "user", "content": prompt}],
        )
        if response is None:
            raise LLMTurnEnded("stopped")
        if (ended := ended_reason(response)) is not None:
            raise ended
        return "".join(block.text for block in response.content
                       if getattr(block, "type", None) == "text")

    # A short conversational reply (no tools, no mix) — used when a message isn't a mixing
    # brief, so the DAW chats instead of crashing (docs/10-daw.md / daw_bridge.chat_reply).
    # Goes through _create, so it's budget-metered + subscription-aware like every other turn.
    CHAT_SYSTEM = (
        "You are the resident mix engineer inside the user's DAW, chatting with the producer. "
        "Reply in 1-3 short, friendly, concrete sentences. You are NOT taking a mix action from "
        "this message — if they want one, nudge them to phrase it as a request, e.g. \"make the "
        "vocal sit\", \"less harsh\", \"warmer\", or \"master for streaming\". Answer mixing "
        "questions briefly when asked. When the message is preceded by session context (the "
        "master's real measured analysis — LUFS, true peak, key, tempo, band energy — and/or a "
        "live DAW edit snapshot), answer questions from those numbers as facts about THIS "
        "session: e.g. read the key from key.tonic/mode, call clipping from true_peak_dbtp "
        "vs -1 dBTP, name tracks from the snapshot. Never claim you can't see the session "
        "when that context is present.")

    def _think_kwargs(self) -> dict:
        """Adaptive-thinking + output-effort request kwargs, per model capability, from the
        ONE table (`llm_models.thinking_kwargs`). The level comes from the composer's picker
        (MIX_AGENT_EFFORT via Brain.set_llm); unset = the product default (`llm_models.DEFAULT_EFFORT`), sent
        explicitly — on Opus 5.5 and Haiku 5.5 the API's own default is `medium`."""
        return llm_models.thinking_kwargs(self._model, self._effort or llm_models.DEFAULT_EFFORT)

    def chat(self, message: str, context: Optional[str] = None) -> str:
        prompt = message if not context else f"{context}\n\nUser: {message}"
        response = self._create(
            request_type="chat",
            model=self._model,
            # 2026-07-30: NOT 400 any more. On Opus 5 (now the default) omitting
            # `thinking` runs ADAPTIVE thinking — on Opus 4.8 it meant no thinking —
            # and max_tokens caps thinking + reply TOGETHER. At 400 the reasoning
            # could eat the whole budget and the user would get an empty or
            # truncated reply. The prompt still asks for 1-3 sentences; this is
            # headroom, not a longer answer.
            max_tokens=4000,
            # Same knowledge every other path gets: a question about what "boxy"
            # means, or what key a song is in, deserves the same craft here as
            # inside a run. Cached, so it costs nothing after the first turn.
            system=[{"type": "text", "text": _knowledge_block(),
                     "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": self.CHAT_SYSTEM}],
            messages=[{"role": "user", "content": prompt}],
            # The composer's effort, like every other turn (reader E6·R13). This was the one
            # request that left it out, so a mid-run answer ran at the API's own default -
            # medium on Opus 5.5 under a composer showing Low, no thinking at all on Opus
            # 4.8 / 4.7. 4000 tokens stays the ceiling: it is headroom for thinking + a
            # three-sentence reply, and the ceiling is not billed.
            **self._think_kwargs(),
        )
        if response is None or ended_reason(response) is not None:
            return ""   # declined / cut / stopped: the caller says so in its own words
        return "".join(block.text for block in response.content
                       if getattr(block, "type", None) == "text").strip()

    def reflect(self, brief: str, verdict: dict, passes_run: int,
                stop_reason: str) -> list[str]:
        """One fresh structured turn (Phase 13): lessons future runs on this
        project should know going in. Fresh context like the critic — the
        verdict and outcome, never the mixing conversation."""
        from pydantic import BaseModel

        class Lessons(BaseModel):
            lessons: list[str]

        prompt = json.dumps({
            "brief": brief, "verdict": verdict, "passes_run": passes_run,
            "stop_reason": stop_reason,
        }, indent=2) + (
            "\n\nWrite 1-3 one-line lessons a future mixing run on this "
            "project should know going in (what worked, what to avoid, "
            "which moves landed). Concrete and short; no praise.")
        response = self._parse(
            request_type="reflection",
            effort_low=True,
            model=self._model,
            max_tokens=2000,
            **self._think_kwargs(),
            system="You distill finished mixing runs into terse lessons.",
            messages=[{"role": "user", "content": prompt}],
            output_format=Lessons,
        ) if not self.budget_exceeded() else None
        return response.lessons[:3] if response is not None else []

    @staticmethod
    def _context_block(ctx: PassContext) -> str:
        block = {
            # The brief leads: it is the instruction; everything below is
            # measurement and derivation (Phase 48 finding).
            "brief": ctx.brief or None,
            "pass_number": ctx.pass_number,
            "metrics": ctx.metrics,
            "targets": [vars(t) for t in ctx.targets],
            "applied_chains": ctx.stem_chains,
            "master_chain": ctx.master_chain,   # the master devices + order + values
            "bus_chains": ctx.bus_chains,        # group busses: chain/members/sends
            "mix_plan": ctx.plan,
            "metrics_delta_since_last_pass": ctx.metrics_delta,
            "recent_history": ctx.history,
            "user_instructions": ctx.instructions,
            "guardrail_limits": ctx.limits,
            "eq_headroom_db": ctx.eq_headroom,   # composed-EQ room left, target -> band
            # Phase 8: the agent's own historical prediction error, e.g.
            # "eq band_energy.air predictions run +1.6 dB cold (bias)".
            "prediction_calibration": ctx.calibration,
            # Phase 13: memory — lessons from past runs on this project and
            # the historically best proposals for similar targets.
            "lessons_from_past_runs": ctx.lessons,
            "worked_examples": ctx.exemplars,
            # Phase 20: how the human has received each proposal kind.
            "user_taste_decisions": ctx.taste,
        }
        # DAW awareness backbone (docs/10-daw.md): when running inside the native DAW, the live
        # Tracktion edit (tracks/clips/tempo/transport) and the operate surface (the command
        # catalog, CHAT only) — so the agent reasons over the REAL session, not just the mix
        # state. A run gets the snapshot but not the catalog: it has no daw_command (design
        # call 2026-09-11). Absent on the offline/headless path.
        if ctx.edit_state:
            # Field-wise, like the chat (full audit 2026-10-02, reader C4·R8). An oversized
            # session used to arrive as the first 24 000 CHARACTERS of its JSON - a string,
            # the head of `tracks` (MIDI notes first) and none of the session-global tail the
            # prompt above tells the planner to read: on an 8-audio + 2-MIDI song it saw no
            # tempo, transport, sections or selection, so a section-aware or ranged move was
            # planned blind. The compactor drops the least load-bearing fields first, names
            # them in `omitted`, leaves the catalog out and writes home as `~` (06·F8); the
            # block stays a dict, and a session that fits is untouched.
            from .edit_state import compact_edit_state
            block["daw_edit_state"] = json.loads(compact_edit_state(ctx.edit_state, 24000))
        return json.dumps(block, indent=2)
