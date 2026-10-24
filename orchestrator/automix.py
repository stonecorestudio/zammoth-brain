"""Auto Mix bootstrap (Phase 16 — docs/05-roadmap.md).

The Ozone-style one-button flow's engine side: profile the project, classify
the stems, and build a *starting chain* — balance moves, an HPF policy per
stem class, corrective EQ where masking measurably happens, gentle
compression on the dynamic sources, and the master limiter to the chosen
loudness target. Everything lands through the ordinary proposal lifecycle
under auto gating: every move carries a rationale and predicted impact, the
feed shows the cards, and the whole thing is undoable.

`build_bundles` is pure (profiles in, proposal bundles out) so the policy is
unit-testable; `bootstrap` drives the engine and emits staged progress
("tagging… profiling… building… applying… rendering") for the UI. The
refinement that follows is the normal pass loop — the gateway runs
`run_passes(max_passes=2)` after this returns.
"""
from __future__ import annotations

from typing import Awaitable, Callable, Optional

import sys

from .targets import _loudness_goal, _mentions

EmitFn = Callable[[dict], Awaitable[None]]

# Policy constants — product opinion, tuned like targets.py.
#
# Balance: CLASS-RELATIVE ACCEPTANCE WINDOWS, not a pull toward one median
# (rewritten 2026-08-05). The old rule moved every stem to within ±6 dB of the median
# `lufs_s`, which is a false model of a finished mix: a correct balance has a huge
# natural spread in solo stem loudness. Measured on a real 4-stem project, vocals sat
# 21.1 dB above bass AT THE CORRECT BALANCE, and the rule "fixed" that by pushing bass
# +6 and the lead vocal -6, producing +12 dB of bass band energy in the render.
#
# Note the spread is NOT a K-weighting artifact — that was the first theory and it is
# wrong. Measured through pyloudnorm's own filters (analysis._k_weighted), K-weighting
# costs this bass stem only 2.2 dB (the curve is -3.6 dB at 60 Hz; the oft-quoted
# -10 dB is down at ~28 Hz), and the same 18.8 dB gap survives on unweighted RMS. So
# no metric swap fixes this — `lufs_s` stays exactly as it is, and the POLICY changes.
#
# `d = pack_median - stem lufs_s`, so d > 0 means "quieter than the pack". A stem is
# left alone while lo <= d <= hi, and only the excess PAST the window edge is
# corrected — one-sided, because a point-offset with a symmetric tolerance just breaks
# in the other direction (it would trim a bass sitting 2 dB under its expected place).
# The numbers come from the measured correct-mix deviations (every window clears them
# by 3-6 dB), the measured K-curve (a low-end stem reading >6 dB OVER the pack really
# is taking over), and do-no-harm bootstrap practice: only trim genuinely hot low-end
# and pads, only lift clearly buried leads.
BALANCE_WINDOWS_DB = {
    "vocal":  (-15.0, 9.0),    # leads live above the pack; lift only if truly buried
    "kick":   (-6.0, 18.0),    # low end reads K-quiet; only trim a real takeover
    "bass":   (-6.0, 18.0),
    "cymbal": (-6.0, 21.0),
    "pad":    (-6.0, 21.0),
}
BALANCE_DEFAULT_WINDOW_DB = (-9.0, 18.0)   # untagged: extreme outliers only
BALANCE_WINDOW_ORDER = ("vocal", "kick", "bass", "cymbal", "pad")
MIN_BALANCE_MOVE_DB = 1.0    # don't propose sub-dB nudges
HPF_POLICY_HZ = {"vocal": 80, "pad": 90, "cymbal": 120, "full_mix": 40}
HPF_DEFAULT_HZ = 40          # untagged stems get a conservative rumble cut
SUB_RUMBLE_FLOOR_DB = -25.0  # only HPF stems that actually hold sub energy
MASKING_EXCESS_DB = 3.0      # corrective EQ kicks in above this excess
MASKING_CUT_MAX_DB = 4.0
COMP_SPREAD_DB = 12.0        # generic fallback; per-intent floors live in
                             # STEM_DYNAMICS_PROFILES["spread_at"] (see below)
MAX_MASKING_PAIRS = 4        # bounded profiling, deterministic order
ACTIONS_PER_BUNDLE = 8       # the engine's actions-per-proposal guardrail
BUNDLES_PER_PASS = 3         # the engine's proposals-per-pass guardrail

# Phase 37 — Auto Mix intent presets. Product names compile to ordinary
# targets/actions; they are not a new engine mode.
INTENT_ALIASES = {
    "vocal-forward": ("vocal-forward", "vocal forward", "vocal upfront",
                      "vocal up front"),
    "punchy-drums": ("punchy-drums", "punchy drums", "drum punch"),
    # The morphological variants are spelled out because matching is whole-word now:
    # a bare "warm" used to catch "warmer"/"warmth" as a substring, but it also caught
    # "s(warm)" and "luke(warm)". Listing the real words keeps what was meant and drops
    # what wasn't. (targets.TONE_INTENT_KEYWORDS spells the same three out.)
    "warm-master": ("warm-master", "warm master", "warm", "warmer", "warmth"),
    "wide-pop": ("wide-pop", "wide pop", "wider pop"),
    "club-loud": ("club-loud", "club loud"),
    "podcast-clean": ("podcast-clean", "podcast clean", "spoken clean"),
}
INTENT_GOAL_LUFS = {
    "club-loud": -8.0,
    "podcast-clean": -16.0,
    "wide-pop": -12.0,
    "warm-master": -14.0,
}
VOCAL_FORWARD_GAIN_DB = 1.5
DRUM_PUNCH_ATTACK_DB = 2.0
DRUM_PUNCH_SUSTAIN_DB = -1.0
WIDE_POP_WIDTH = 1.25
PODCAST_DEESS_AT = 0.15
PODCAST_DEESS_FREQ_HZ = 5500
PODCAST_DEESS_THRESHOLD_DB = -30.0
PODCAST_HPF_HZ = 85

# Phase 28 — Auto Mix owns routing + automation now (shown through proposals).
# Group stems by tag family onto a bus, glue each with one parallel comp, then
# ride the lead up in the one section where the arrangement buries it.
TAG_BUS_FAMILY = {"kick": "drums", "cymbal": "drums", "vocal": "vox"}
MIN_BUS_MEMBERS = 2          # a bus of one is pointless
MAX_BUSSES = 3               # bounded, deterministic order
GLUE_THRESHOLD_DB = -24.0
GLUE_RATIO = 2.5
GLUE_MIX = 0.5               # parallel (NY) compression — thicken, don't flatten
# 15 ms until 2026-10-03. That thickened only in the offline render, whose compressor read
# "15 ms" as a 2.4 ms time constant (reader E4·R1); on the law the DAW runs, 15 ms lets the
# hits through and the bus's crest RISES 0.3 dB. Measured on the technique fixture's rhythm
# bus at these settings: 2 ms -> crest down 0.7 dB, master -0.8 LU.
GLUE_ATTACK_MS = 2.0
#: What each dynamics profile is FOR, in the words the user reads under the move. The reason line
#: used to show the profile's own name ("FullDrums: peak_control") - user QA 2026-10-08, D22.
_DYNAMICS_WORDS = {
    "vocal_level": "evens out the vocal",
    "peak_control": "catches the loudest hits",
    "bass_density": "firms up the bass",
    "density": "adds gentle body",
}


def dynamics_reason(intent: str) -> str:
    """A dynamics profile's purpose in plain words; an unknown one reads as its name, spaced."""
    return _DYNAMICS_WORDS.get(intent, str(intent).replace("_", " "))


STEM_DYNAMICS_PROFILES = {
    # Phase 34: same comp tool, better intent-specific defaults.
    #
    # `below_db` — where the threshold sits BELOW the stem's own measured lufs_s, and
    # `makeup` is derived from it, not typed in (both 2026-08-05). These used to be
    # fixed ABSOLUTE thresholds with makeup_db 0.0, which only makes sense if every
    # stem happens to sit near -24 dBFS. Measured on a real lead vocal (lufs_s -12.8,
    # peaks -5.6 dBFS), the -24 threshold sat ~18 dB under the signal, so the
    # compressor attenuated the whole stem uniformly instead of leveling it: it cost
    # 9.5 dB of level to buy 1.0 dB of dynamic-range reduction, and with no makeup
    # nothing gave the level back. Uniform gain reduction is not compression, it's a
    # fader move with extra steps. A threshold placed relative to the source sits at
    # the top of its range where the peaks actually are, and the makeup returns what
    # the ratio takes (see _stem_dynamics_action).
    #
    # `spread_at` — the dynamic_spread_db floor that earns this profile a compressor.
    # Deliberately still the conservative COMP_SPREAD_DB for every profile, and that
    # is a MEASURED decision, not an untouched default (2026-08-05). Lowering it so a
    # 9 dB-spread lead vocal would earn a comp looked obviously right and was wrong:
    # dynamic_spread_db is p95 - p10 of 400 ms RMS frames, and on that real vocal the
    # spread lives entirely at the QUIET end (p10 -19.0, p50 -12.0 dBFS) while the loud
    # end is already flat (p75 -10.5 through p99 -9.9, a 0.6 dB span). A downward
    # compressor only acts above its threshold, so it cannot raise quiet phrases — it
    # measured 2.9 dB of level surrendered for 0.6 dB of spread closed. The right tool
    # for that shape is already in the bootstrap and already fires: the Phase 35
    # phrase-level ranged gain moves. Keep the bar high; a comp here is a mismatched
    # tool, not a missing one.
    "vocal_level": {"below_db": 6.0, "ratio": 2.5,
                    "attack_ms": 8.0, "release_ms": 120.0, "mix": 1.0,
                    "predicted_delta": -2.0, "spread_at": COMP_SPREAD_DB},
    "peak_control": {"below_db": 4.0, "ratio": 3.0,
                     "attack_ms": 2.0, "release_ms": 80.0, "mix": 1.0,
                     "predicted_delta": -1.5, "spread_at": COMP_SPREAD_DB},
    "bass_density": {"below_db": 6.0, "ratio": 3.0,
                     "attack_ms": 20.0, "release_ms": 180.0, "mix": 1.0,
                     "predicted_delta": -2.0, "spread_at": COMP_SPREAD_DB},
    "density": {"below_db": 5.0, "ratio": 2.0,
                "attack_ms": 12.0, "release_ms": 180.0, "mix": 0.8,
                "predicted_delta": -1.5, "spread_at": COMP_SPREAD_DB},
}
COMP_THRESHOLD_FLOOR_DB = -60.0   # CompAction's own bound
COMP_MAKEUP_MAX_DB = 12.0         # CompAction's own bound
RIDE_LEAD_TAGS = {"vocal"}
RIDE_ACTIVE_FLOOR = 0.5      # a stem counts as present in a section above this
RIDE_MIN_COMPETERS = 3       # the lead is buried when this many others crowd it
RIDE_LIFT_DB = 1.5           # one conservative corrective lift
RIDE_MIN_SPAN_S = 1.0
RIDE_MIN_SECTIONS = 2        # a ride picks ONE section — it needs a choice to make
RIDE_MAX_SPAN_FRACTION = 0.9  # a section covering the whole song isn't a ride

# Phase 35 — phrase-aware vocal control through existing ranged tools.
PHRASE_GAIN_MAX_DB = 2.0
PHRASE_LEVEL_ACTIONS_MAX = 3
PHRASE_DEESS_ACTIONS_MAX = 2
PHRASE_DEESS_DENSITY_AT = 0.45

# Phase 29 — AI cleanup preset: repair separation artifacts read from
# analyze_separation (the "double check"). Thresholds are product opinion over
# the 0..1 artifact metrics; the de-noise/attack amounts stay well under the
# guardrail caps (gentle by default — the loop can refine).
SHIMMER_DENOISE_AT = 0.30      # hf_shimmer ≥ this → spectral de-noise
LEAKAGE_DENOISE_AT = 0.40      # leakage ≥ this → spectral de-noise
DENOISE_REDUCTION_DB = 9.0     # guardrail cap is 24
DENOISE_SENSITIVITY_DB = 8.0
SMEAR_RESTORE_AT = 0.30        # transient_smear ≥ this → attack restore (non-pads)
TRANSIENT_ATTACK_DB = 3.0      # schema cap is 6
# (no LUFS_SPREAD_MATCH_AT — inter-stem loudness matching was removed 2026-08-05;
#  separated stems sum to the source, so their spread is the source's own balance)
SPECTRAL_HOLE_REPAIR_AT = 0.35
CLEANUP_WATCH_MARGIN = 0.08
CLEANUP_GLUE_REPAIRED_AT = 2
CLEANUP_GLUE_THRESHOLD_DB = -26.0
CLEANUP_GLUE_RATIO = 2.0
CLEANUP_GLUE_MIX = 0.35


def intent_from_brief(text: str) -> Optional[str]:
    """Which preset the brief asks for, if any.

    Whole-word matching (`_mentions`), for the same reason as targets._loudness_goal:
    as a bare substring 'warm' fires inside "s(warm) of synths" and "luke(warm)", each
    of which then laid a whole warm-master chain — including a master limiter — over a
    mix nobody asked to master that way.

    A brief can name more than one ("warmer and a bit wider") and the presets are
    single-valued, so the first in table order wins and the other is dropped. That is
    the existing design, but it used to be silent; say so, because "I asked for two
    things and got one" is otherwise indistinguishable from the preset misfiring."""
    lowered = text.lower()
    matched = [intent for intent, aliases in INTENT_ALIASES.items()
               if _mentions(lowered, aliases)]
    if not matched:
        return None
    if len(matched) > 1:
        print(f"[automix] brief names {len(matched)} intents {matched} — applying "
              f"{matched[0]!r}; the others are not layered", file=sys.stderr)
    return matched[0]


def _intent_goal_lufs(intent: Optional[str]) -> Optional[float]:
    return INTENT_GOAL_LUFS.get(intent or "")


def refinement_brief(brief: str) -> str:
    """What Auto Mix's refinement loop may aim at (2026-09-10 agent audit P1-24): the brief's
    LOUDNESS part only. A preset name is not a tonal brief — refining toward "wide pop"
    compiled pop tonal balance + a wider image, stacked two master match-EQs and moved the
    target to -10 LUFS. No preset: the brief as it was. A preset: its words removed, then the
    brief's own loudness words if any are left, else the preset's goal LUFS, else "" (the
    bootstrap stands alone)."""
    import re

    intent = intent_from_brief(brief)
    if intent is None:
        return brief
    rest = brief.lower()
    for alias in sorted(INTENT_ALIASES[intent], key=len, reverse=True):
        rest = re.sub(rf"(?<![\w-]){re.escape(alias)}(?![\w-])", " ", rest)
    rest = re.sub(r"\s+", " ", rest).strip(" ,;.-")
    if rest and _loudness_goal(rest) is not None:
        return rest
    goal = _intent_goal_lufs(intent)
    return f"master to {goal:g} LUFS" if goal is not None else ""


def suspect_pairs(stems: dict[str, dict]) -> list[tuple[str, str]]:
    """(maskee, masker) pairs worth measuring: stems whose two dominant bands
    overlap, RANKED so the caller's cap keeps the worst ones.

    Ranking added 2026-08-05. The list used to be built in alphabetical id order and
    sliced to MAX_MASKING_PAIRS, so on any session with more than two overlapping
    stems the alphabet decided which measurements were spent — `analyze_masking` is
    the expensive part of profiling, and spending it on ('a_gtr', 'b_gtr') while a
    vocal and a synth fight over the mids is the wrong trade. Rank by how much energy
    the two stems actually pile into their shared bands (loudest overlap first), keep
    both directions adjacent (masking is directional), and break ties on the ids so
    the order stays deterministic.

    Returns the FULL ranked list — `bootstrap` applies the cap, so it can say how many
    candidates it left unmeasured instead of truncating in silence."""
    dominant: dict[str, set] = {}
    energy: dict[str, dict] = {}
    for stem_id, analysis in sorted(stems.items()):
        bands = analysis.get("band_energy") or {}
        energy[stem_id] = bands
        top = sorted(bands, key=bands.get, reverse=True)[:2]
        dominant[stem_id] = set(top)

    ranked: list[tuple[float, str, str]] = []
    ids = sorted(dominant)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            shared = dominant[a] & dominant[b]
            if not shared:
                continue
            # dB, so the max over the shared bands of the quieter stem is the honest
            # "how loud is the collision" proxy — two stems both loud in one band.
            severity = max(min(energy[a].get(band, -80.0),
                               energy[b].get(band, -80.0)) for band in shared)
            ranked.append((severity, a, b))

    pairs: list[tuple[str, str]] = []
    for _, a, b in sorted(ranked, key=lambda r: (-r[0], r[1], r[2])):
        pairs += [(a, b), (b, a)]  # masking is directional
    return pairs


def _chunk(actions_with_predictions: list[tuple[dict, Optional[dict]]],
           rationale: str) -> list[dict]:
    """Bundle (action, prediction) pairs into ≤ ACTIONS_PER_BUNDLE proposals."""
    bundles = []
    for start in range(0, len(actions_with_predictions), ACTIONS_PER_BUNDLE):
        chunk = actions_with_predictions[start:start + ACTIONS_PER_BUNDLE]
        bundles.append({
            "rationale": rationale,
            "predicted_impact": [p for _, p in chunk if p is not None][:8],
            "actions": [a for a, _ in chunk],
        })
    return bundles


def _chain_of(analysis: dict) -> list[dict]:
    return analysis.get("applied_chain") or []


def _has_hpf(analysis: dict) -> bool:
    return any(e["type"] == "eq" and e["band"]["shape"] == "hpf"
               for e in _chain_of(analysis))


def _has_dynamics(analysis: dict) -> bool:
    return any(e["type"] in ("comp", "mb_comp") for e in _chain_of(analysis))


def _has_deess(analysis: dict) -> bool:
    return any(e["type"] == "deess" for e in _chain_of(analysis))


def _has_cut_in(analysis: dict, lo_hz: float, hi_hz: float) -> bool:
    """A static EQ cut already sits in [lo_hz, hi_hz) on this stem — a previous Auto Mix's
    masking carve, or the user's own. A re-run must not stack a second one (2026-09-10
    audit P2, 05·F7: two runs left −8 dB at 12.6 kHz on the drums)."""
    for e in _chain_of(analysis):
        band = e.get("band") if e.get("type") == "eq" else None
        if (isinstance(band, dict) and band.get("shape") not in ("hpf", "lpf")
                and not e.get("time_range_s")
                and (band.get("gain_db") or 0.0) < 0.0
                and lo_hz <= float(band.get("freq_hz") or 0.0) < hi_hz):
            return True
    return False


def _bus_groupings(tags: dict[str, list[str]],
                   stems: dict[str, dict],
                   grouped: Optional[set[str]] = None) -> list[tuple[dict, Optional[dict]]]:
    """Group stems by tag family (drums = kick/cymbal, vox = vocals) and glue
    each resulting bus with one parallel comp. Deterministic and bounded — the
    agent owns routing now, so the bootstrap lays the obvious busses instead of
    waiting on the user. A `group` action creates the bus + routes its members;
    the comp that follows targets that same bus in the same proposal.

    `grouped` names the tracks that already play through a group (real use, 2026-10-22: the
    user had made a Rhythm group of drums, bass and music, and Auto Mix moved the drums out of
    it into a family group of its own). Those stay where they are: a family is made of the
    loose tracks only."""
    families: dict[str, list[str]] = {}
    for stem_id in sorted(stems):
        if grouped and stem_id in grouped:
            continue
        family = next((TAG_BUS_FAMILY[t] for t in sorted(tags.get(stem_id, []))
                       if t in TAG_BUS_FAMILY), None)
        if family is not None:
            families.setdefault(family, []).append(stem_id)

    actions: list[tuple[dict, Optional[dict]]] = []
    busses = 0
    for family in sorted(families):
        members = families[family]
        if len(members) < MIN_BUS_MEMBERS or busses >= MAX_BUSSES:
            continue
        # The group's id is the family's name only where no STEM has it (reader E6·R4): stems,
        # groups and returns share one namespace, and a project with a `vox.wav` - a common
        # export - had its vocal group refused for colliding with its own lead ("vox bus",
        # then "vox bus 2"…). A group already called that is the group to use, as before.
        bus_id, n = family, 1
        while bus_id in stems:
            n += 1
            bus_id = f"{family} bus" if n == 2 else f"{family} bus {n - 1}"
        actions.append(({"type": "group", "bus_id": bus_id,
                         "members": members}, None))
        actions.append(({"type": "comp", "stem_id": bus_id,
                         "threshold_db": GLUE_THRESHOLD_DB, "ratio": GLUE_RATIO,
                         "attack_ms": GLUE_ATTACK_MS, "release_ms": 150.0,
                         "makeup_db": 0.0, "mix": GLUE_MIX}, None))
        busses += 1
    return actions


def _bus_grouped_members(actions: list[tuple[dict, Optional[dict]]]) -> set[str]:
    """Members covered by a just-created Phase 28 bus. Auto Mix should prefer
    the bus glue comp over extra member comps inside the same bootstrap."""
    grouped: set[str] = set()
    for action, _ in actions:
        if action["type"] == "group":
            grouped.update(action["members"])
    return grouped


def _dynamics_profile(stem_tags: set[str], analysis: dict) -> tuple[str, dict]:
    """Pick a compressor intent from existing metrics. No new DSP: this only
    chooses parameters for the existing `CompAction`."""
    transient_ratio = analysis.get("transient_energy_ratio") or 0.0
    transient_density = analysis.get("transient_density") or 0.0
    if "vocal" in stem_tags:
        return "vocal_level", STEM_DYNAMICS_PROFILES["vocal_level"]
    if "bass" in stem_tags:
        return "bass_density", STEM_DYNAMICS_PROFILES["bass_density"]
    if (stem_tags & {"kick", "cymbal"}
            or (transient_ratio >= 0.25 and transient_density >= 0.5)):
        return "peak_control", STEM_DYNAMICS_PROFILES["peak_control"]
    return "density", STEM_DYNAMICS_PROFILES["density"]


def _stem_dynamics_action(stem_id: str, analysis: dict,
                          stem_tags: set[str]) -> Optional[tuple[dict, dict, str]]:
    """One chain-aware stem compressor when the source's level really moves.
    Returns (action, prediction, intent) or None.

    The profile is chosen BEFORE the spread test, because the floor is part of the
    profile now — what counts as "really moves" depends on the source (see
    STEM_DYNAMICS_PROFILES)."""
    spread = analysis.get("dynamic_spread_db")
    if spread is None or "pad" in stem_tags:
        return None
    intent, profile = _dynamics_profile(stem_tags, analysis)
    if spread <= profile["spread_at"]:
        return None
    level = analysis.get("lufs_s")
    if level is None:
        return None                     # can't place a threshold without a level
    # Threshold just under the source's own loud parts, so the ratio acts on the peaks
    # rather than on the whole signal; makeup returns what the ratio takes there, so
    # the stem keeps its place in the mix and the QUIET parts come up relative to the
    # loud ones — which is what "leveling" means.
    threshold_db = max(COMP_THRESHOLD_FLOOR_DB,
                       min(0.0, level - profile["below_db"]))
    makeup_db = min(COMP_MAKEUP_MAX_DB,
                    round(profile["below_db"] * (1.0 - 1.0 / profile["ratio"]), 1))
    action = {"type": "comp", "stem_id": stem_id,
              "threshold_db": round(threshold_db, 1),
              "ratio": profile["ratio"],
              "attack_ms": profile["attack_ms"],
              "release_ms": profile["release_ms"],
              "makeup_db": max(0.0, makeup_db),
              "mix": profile["mix"]}
    prediction = {"metric": f"stem({stem_id}).dynamic_spread_db",
                  "predicted_delta": profile["predicted_delta"],
                  "unit": "dB"}
    return action, prediction, intent


def _phrase_actions(tags: dict[str, list[str]], stems: dict[str, dict],
                    phrases: Optional[dict[str, dict]],
                    existing_rides: Optional[dict[str, list]] = None,
                    ) -> tuple[list[tuple[dict, Optional[dict]]], list[str]]:
    """Phase 35: phrase readout -> existing ranged gain/de-ess actions.
    Conservative by design; the loop can refine after the first render.

    Returns (actions, notes). The caps below are real product opinion, but they used
    to truncate in silence — 4 detected level moves became 3 applied ones with nothing
    said, which reads as the detector missing a phrase rather than the policy bounding
    itself. Say what was left on the table (no silent caps)."""
    if not phrases:
        return [], []
    actions: list[tuple[dict, Optional[dict]]] = []
    notes: list[str] = []
    for stem_id in sorted(phrases):
        if "vocal" not in set(tags.get(stem_id, [])):
            continue
        analysis = stems.get(stem_id, {})
        all_moves = phrases[stem_id].get("level_moves") or []
        if all_moves and (existing_rides or {}).get(stem_id):
            # Rides MULTIPLY (dsp.py): a second Auto Mix leveled the same phrases again on
            # top of the first run's rides (2026-09-10 audit P2, 05·F7). The lead ride has
            # had this guard since 2026-08-09; the phrase rides never did.
            notes.append(f"{stem_id}: already rides its phrases — left as they are")
            all_moves = []
        if len(all_moves) > PHRASE_LEVEL_ACTIONS_MAX:
            notes.append(f"{stem_id}: {len(all_moves)} level moves detected, "
                         f"applying the first {PHRASE_LEVEL_ACTIONS_MAX}")
        for move in all_moves[:PHRASE_LEVEL_ACTIONS_MAX]:
            delta = max(-PHRASE_GAIN_MAX_DB,
                        min(PHRASE_GAIN_MAX_DB, move["delta_db"]))
            actions.append((
                {"type": "gain", "stem_id": stem_id,
                 "delta_db": round(delta, 1),
                 "time_range_s": [move["start_s"], move["end_s"]]},
                {"metric": f"stem({stem_id}).dynamic_spread_db",
                 "predicted_delta": -abs(round(delta, 1)), "unit": "dB"}))
        if _has_deess(analysis):
            continue
        bursts = [b for b in (phrases[stem_id].get("sibilance_bursts") or [])
                  if b.get("density", 0.0) >= PHRASE_DEESS_DENSITY_AT]
        if len(bursts) > PHRASE_DEESS_ACTIONS_MAX:
            notes.append(f"{stem_id}: {len(bursts)} sibilant bursts over "
                         f"threshold, de-essing the first "
                         f"{PHRASE_DEESS_ACTIONS_MAX}")
        for burst in bursts[:PHRASE_DEESS_ACTIONS_MAX]:
            actions.append((
                {"type": "deess", "stem_id": stem_id,
                 "threshold_db": burst["threshold_db"],
                 "freq_hz": burst["freq_hz"],
                 "time_range_s": [burst["start_s"], burst["end_s"]]},
                {"metric": f"stem({stem_id}).sibilance_density",
                 "predicted_delta": -0.1, "unit": "ratio"}))
    return actions, notes


def _buried_lead_ride(tags: dict[str, list[str]], sections: list[dict]
                      ) -> Optional[tuple[dict, Optional[dict]]]:
    """One conservative automation move: lift a lead vocal in the single
    section where the arrangement most buries it (lead present, the most other
    stems crowding it). Returns one ranged gain ride, or None when nothing
    clearly qualifies — Auto Mix should *touch* automation, not over-ride."""
    # A ride is a corrective move in ONE section, so it needs sections to choose
    # between (fixed 2026-08-05). `analyze_sections` returns a single whole-file
    # section for short or unstructured audio, and every guard below then passed on
    # it: on a real 8 s project the bootstrap emitted a [0.0, 8.0] "ride" — a static
    # gain wearing an automation costume, stacked on top of the balance move that had
    # just cut the same stem. No structure, no ride; the balance step owns that.
    if len(sections or []) < RIDE_MIN_SECTIONS:
        return None
    total_s = (max(s["end_s"] for s in sections)
               - min(s["start_s"] for s in sections))

    leads = [stem_id for stem_id in sorted(tags)
             if set(tags.get(stem_id, [])) & RIDE_LEAD_TAGS]
    best = None  # ((competers, -start_s), lead, section) — max competers wins
    for lead in leads:
        for section in sections:
            activity = section.get("stem_activity") or {}
            span_s = section["end_s"] - section["start_s"]
            if (span_s < RIDE_MIN_SPAN_S
                    or (total_s > 0
                        and span_s / total_s > RIDE_MAX_SPAN_FRACTION)
                    or activity.get(lead, 0.0) < RIDE_ACTIVE_FLOOR):
                continue
            competers = sum(1 for sid, frac in activity.items()
                            if sid != lead and frac >= RIDE_ACTIVE_FLOOR)
            if competers < RIDE_MIN_COMPETERS:
                continue
            key = (competers, -section["start_s"])
            if best is None or key > best[0]:
                best = (key, lead, section)
    if best is None:
        return None
    _, lead, section = best
    return ({"type": "gain", "stem_id": lead, "delta_db": RIDE_LIFT_DB,
             "time_range_s": [round(section["start_s"], 2),
                               round(section["end_s"], 2)]}, None)


def _intent_actions(intent: Optional[str], tags: dict[str, list[str]],
                    stems: dict[str, dict]) -> tuple[
                        list[tuple[dict, Optional[dict]]], list[str]]:
    """Phase 37 preset overlays. They are deliberately small nudges on top of
    the normal bootstrap, all through existing actions and measured tags."""
    actions: list[tuple[dict, Optional[dict]]] = []
    notes: list[str] = []
    if intent is None:
        return actions, notes

    def tagged(*wanted: str) -> list[str]:
        wanted_set = set(wanted)
        return [sid for sid in sorted(stems)
                if wanted_set & set(tags.get(sid, []))]

    if intent == "vocal-forward":
        vocals = tagged("vocal")
        if not vocals:
            notes.append("skipped vocal lift: no confident vocal tag")
        for sid in vocals[:2]:
            lufs = stems[sid].get("lufs_s")
            actions.append((
                {"type": "gain", "stem_id": sid,
                 "delta_db": VOCAL_FORWARD_GAIN_DB},
                {"metric": f"stem({sid}).lufs_s",
                 "predicted_delta": VOCAL_FORWARD_GAIN_DB, "unit": "dB"}))
            notes.append(f"{sid} vocal tag, lufs_s={lufs}")

    elif intent == "punchy-drums":
        drums = tagged("kick", "cymbal")
        if not drums:
            drums = [sid for sid, stem in sorted(stems.items())
                     if (stem.get("transient_energy_ratio") or 0.0) >= 0.25]
        if not drums:
            notes.append("skipped drum punch: no transient drum-like stem")
        for sid in drums[:3]:
            ratio = stems[sid].get("transient_energy_ratio")
            actions.append((
                {"type": "transient", "stem_id": sid,
                 "attack_db": DRUM_PUNCH_ATTACK_DB,
                 "sustain_db": DRUM_PUNCH_SUSTAIN_DB},
                {"metric": f"stem({sid}).transient_energy_ratio",
                 "predicted_delta": 0.1, "unit": "ratio"}))
            notes.append(f"{sid} transient_energy_ratio={ratio}")

    elif intent == "warm-master":
        actions += [
            ({"type": "eq", "stem_id": "master",
              "band": {"freq_hz": 180, "gain_db": 1.5, "q": 0.7,
                       "shape": "low_shelf"}},
             {"metric": "band_energy.low_mid",
              "predicted_delta": 0.8, "unit": "dB"}),
            ({"type": "eq", "stem_id": "master",
              "band": {"freq_hz": 6500, "gain_db": -1.0, "q": 0.7,
                       "shape": "high_shelf"}},
             {"metric": "band_energy.brilliance",
              "predicted_delta": -0.5, "unit": "dB"}),
        ]
        notes.append("warm tilt: low shelf +1.5 dB, high shelf -1 dB")

    elif intent == "wide-pop":
        candidates = [sid for sid in sorted(stems)
                      if set(tags.get(sid, [])) & {"vocal", "pad", "cymbal"}]
        candidates = [sid for sid in candidates
                      if not (set(tags.get(sid, [])) & {"kick", "bass"})]
        if not candidates:
            notes.append("skipped widening: no safe non-bass stereo candidate")
        for sid in candidates[:3]:
            actions.append((
                {"type": "width", "stem_id": sid, "width": WIDE_POP_WIDTH},
                {"metric": "stereo_correlation",
                 "predicted_delta": -0.05, "unit": "corr"}))
            notes.append(f"{sid} tag={','.join(tags.get(sid, []))}")

    elif intent == "podcast-clean":
        voices = tagged("vocal", "full_mix") or sorted(stems)
        for sid in voices[:3]:
            stem = stems[sid]
            if not _has_hpf(stem):
                actions.append((
                    {"type": "eq", "stem_id": sid,
                     "band": {"freq_hz": PODCAST_HPF_HZ, "gain_db": 0.0,
                              "q": 0.7, "shape": "hpf"}},
                    {"metric": f"stem({sid}).band_energy.sub",
                     "predicted_delta": -6.0, "unit": "dB"}))
            if (not _has_deess(stem)
                    and (stem.get("sibilance_density") or 0.0) >= PODCAST_DEESS_AT):
                actions.append((
                    {"type": "deess", "stem_id": sid,
                     "threshold_db": PODCAST_DEESS_THRESHOLD_DB,
                     "freq_hz": PODCAST_DEESS_FREQ_HZ},
                    {"metric": f"stem({sid}).sibilance_density",
                     "predicted_delta": -0.1, "unit": "ratio"}))
            notes.append(f"{sid} podcast cleanup, "
                         f"sibilance={stem.get('sibilance_density')}")

    return actions, notes


def build_bundles(tags: dict[str, list[str]], stems: dict[str, dict],
                  maskings: list[dict], session_analysis: dict,
                  goal_lufs: Optional[float],
                  sections: Optional[list[dict]] = None,
                  phrases: Optional[dict[str, dict]] = None,
                  intent: Optional[str] = None,
                  existing_rides: Optional[dict[str, list]] = None,
                  grouped: Optional[set[str]] = None) -> list[dict]:
    """The starting chain as proposal bundles (pure — unit-testable policy).
    Themes: balance + low-end policy, group busses + glue (Phase 28), clarity +
    dynamics, one corrective lead ride (Phase 28), master loudness.
    Chain-aware (2026-06 review): stems that already carry an HPF or a
    compressor keep them — a re-run refines balance and masking, it never
    stacks a second rumble cut or comp on top."""
    balance: list[tuple[dict, Optional[dict]]] = []
    # A `full_mix` stem IS the mix — it belongs to no "pack", so it neither skews the
    # median nor gets balanced against the parts it already contains.
    packed = {sid: a["lufs_s"] for sid, a in stems.items()
              if a.get("lufs_s") is not None
              and "full_mix" not in set(tags.get(sid, []))}
    levels = sorted(packed.values())
    median = (levels[len(levels) // 2] if len(levels) % 2
              else (levels[len(levels) // 2 - 1]
                    + levels[len(levels) // 2]) / 2.0) if levels else None
    for stem_id in sorted(packed):
        deviation = median - packed[stem_id]     # >0 = quieter than the pack
        stem_tags = set(tags.get(stem_id, []))
        low, high = next((BALANCE_WINDOWS_DB[t] for t in BALANCE_WINDOW_ORDER
                          if t in stem_tags), BALANCE_DEFAULT_WINDOW_DB)
        if deviation < low:
            excess = deviation - low             # too loud: trim by the overshoot only
        elif deviation > high:
            excess = deviation - high            # too quiet: lift to the window edge
        else:
            continue                             # inside its class window — leave it
        delta = round(max(-6.0, min(6.0, excess)), 1)
        if abs(delta) < MIN_BALANCE_MOVE_DB:
            continue
        balance.append((
            {"type": "gain", "stem_id": stem_id, "delta_db": delta},
            {"metric": f"stem({stem_id}).lufs_s",
             "predicted_delta": delta, "unit": "dB"}))

    for stem_id, analysis in sorted(stems.items()):
        stem_tags = set(tags.get(stem_id, []))
        if stem_tags & {"kick", "bass"}:
            continue  # never thin out the low-end sources
        if _has_hpf(analysis):
            continue  # already cleaned — don't stack a second rumble cut
        bands = analysis.get("band_energy") or {}
        if bands.get("sub", -80.0) <= SUB_RUMBLE_FLOOR_DB:
            continue  # nothing down there to clean
        freq_hz = next((HPF_POLICY_HZ[t] for t in sorted(stem_tags)
                        if t in HPF_POLICY_HZ), HPF_DEFAULT_HZ)
        balance.append((
            {"type": "eq", "stem_id": stem_id,
             "band": {"freq_hz": freq_hz, "gain_db": 0.0, "q": 0.7,
                      "shape": "hpf"}},
            {"metric": f"stem({stem_id}).band_energy.sub",
             "predicted_delta": -6.0, "unit": "dB"}))

    bus_actions = _bus_groupings(tags, stems, grouped)
    grouped_members = _bus_grouped_members(bus_actions)
    phrase_actions, phrase_notes = _phrase_actions(tags, stems, phrases, existing_rides)
    intent_actions, intent_notes = _intent_actions(intent, tags, stems)

    clarity: list[tuple[dict, Optional[dict]]] = []
    member_comps: dict[str, tuple[dict, Optional[dict]]] = {}   # a grouped member's own comp
    dynamics_intents: list[str] = []
    # ONE cut per (stem, band), sized by the worst masker (fixed 2026-08-05).
    # `suspect_pairs` is directional and several maskers can crowd the same maskee in
    # the same band: on a real 4-stem project drums->vocals (28.7 dB excess) and
    # other->vocals (15.7 dB) both resolved to -4.0 dB at 632 Hz, so the bootstrap
    # notched the vocal -8 dB inside one bundle and showed the user the identical row
    # twice. Insertion-ordered, so the emitted order stays deterministic.
    worst_per_band: dict[tuple[str, str], dict] = {}
    for masking in maskings:
        if masking["worst_band_excess_db"] <= MASKING_EXCESS_DB:
            continue
        key = (masking["stem_b"], masking["worst_band"])
        previous = worst_per_band.get(key)
        if (previous is None
                or masking["worst_band_excess_db"]
                > previous["worst_band_excess_db"]):
            worst_per_band[key] = masking
    for masking in worst_per_band.values():
        excess = masking["worst_band_excess_db"]
        cut_db = round(min(MASKING_CUT_MAX_DB, excess - 2.0), 1)
        from .engine_models import BAND_EDGES_HZ, BAND_LABELS
        index = BAND_LABELS.index(masking["worst_band"])
        if _has_cut_in(stems.get(masking["stem_b"], {}),
                       BAND_EDGES_HZ[index], BAND_EDGES_HZ[index + 1]):
            continue   # carved already: a re-run refines, it never deepens the notch
        center_hz = int((BAND_EDGES_HZ[index] * BAND_EDGES_HZ[index + 1]) ** 0.5)
        clarity.append((
            {"type": "eq", "stem_id": masking["stem_b"],
             "band": {"freq_hz": center_hz, "gain_db": -cut_db, "q": 1.2,
                      "shape": "peak"}},
            {"metric": f"masking({masking['stem_a']},{masking['stem_b']})"
                       ".worst_band_excess_db",
             "predicted_delta": -cut_db, "unit": "dB"}))
    for stem_id, analysis in sorted(stems.items()):
        stem_tags = set(tags.get(stem_id, []))
        if _has_dynamics(analysis):
            continue  # already compressed: a re-run adjusts via the loop,
            # never stacks a second comp (the 2026-06 double-comp bug)
        dynamic = _stem_dynamics_action(stem_id, analysis, stem_tags)
        if stem_id in grouped_members:
            # Group glue is the cleaner move for related families - IF the group lands. The
            # member's own comp is kept as the group bundle's fallback (reader E6·R4): a
            # refused group left its members with neither glue nor a compressor.
            if dynamic:
                member_comps.setdefault(stem_id, dynamic[:2])
            continue
        if dynamic:
            # NB: not `intent` — that's the Auto Mix preset param, still read
            # below for the preset/loudness rationales.
            action, prediction, stem_intent = dynamic
            clarity.append((action, prediction))
            dynamics_intents.append(f"{stem_id}: {dynamics_reason(stem_intent)}")
    ride = _buried_lead_ride(tags, sections or [])
    if ride is not None and (existing_rides or {}).get(ride[0]["stem_id"]):
        # Idempotence (2026-08-09 §4 audit): overlapping rides MULTIPLY (dsp.py), and
        # unlike the HPF/comp/bus moves this one had no re-run guard — Auto Mix twice
        # doubled the lift on the same lead in the same section (a prior ride's boost
        # doesn't change the stem_activity fractions that select it). One corrective
        # ride per stem from this policy, ever; the refinement loop owns adjustments.
        ride = None

    bundles = []
    if balance:
        bundles += _chunk(balance, "auto mix: balance the stems and clear "
                                   "low-end rumble per stem class")
    # A family a bundle (reader E6·R4): a proposal is atomic, so with every family in one a
    # refusal of the vocal group took the drum group and both glues down with it.
    for at in range(0, len(bus_actions), 2):
        group, glue = bus_actions[at], bus_actions[at + 1]
        for bundle in _chunk([group, glue], "auto mix: group related stems onto "
                                            "a bus and glue it with parallel comp"):
            fallback = [member_comps[m] for m in group[0]["members"] if m in member_comps]
            if fallback:
                bundle["_if_refused"] = _chunk(
                    fallback, "auto mix: tame the dynamic sources (their group was not made)")
            bundles.append(bundle)
    if phrase_actions:
        note = f" ({'; '.join(phrase_notes[:3])})" if phrase_notes else ""
        bundles += _chunk(phrase_actions, "auto mix: level vocal phrases and "
                                          "de-ess only the detected bursts"
                                          + note)
    if intent_actions:
        note = f" ({'; '.join(intent_notes[:4])})" if intent_notes else ""
        # Its own pass (2026-09-10 audit P1-23 — see _apply_bundles).
        for b in _chunk(intent_actions, f"auto mix preset {intent}: "
                                        "apply intent-specific existing "
                                        f"tools{note}"):
            b["_own_pass"] = True
            bundles.append(b)
    if clarity:
        suffix = (f" ({'; '.join(dynamics_intents)})"
                  if dynamics_intents else "")
        bundles += _chunk(clarity, "auto mix: carve space where masking "
                                   "measures and tame the dynamic sources"
                                   + suffix)
    if ride is not None:
        bundles.append({
            "rationale": "auto mix: ride the lead up where the arrangement "
                         "buries it",
            "predicted_impact": [],
            "actions": [ride[0]]})
    # The master limiter lands on EVERY bootstrap, not only when the brief named a
    # loudness (fixed 2026-08-05). The DAW's one-button Auto Mix sends an empty brief
    # (App.cpp, `startAction("automix")`), so `goal_lufs` was None, this bundle was
    # skipped entirely, and the one move that guarantees the deliverable never ran:
    # measured on a real 4-stem project the bootstrap finished at +2.0 -> -0.9 dBTP
    # with `master_true_peak_exceeded` still raised. The ceiling is an invariant
    # (CLAUDE.md), not an opt-in. Without a named goal it is a CEILING-ONLY limiter —
    # `target_lufs_i` stays None, because guaranteeing -1.0 dBTP is our promise while
    # picking a loudness the user never asked for is not.
    goal = max(-24.0, min(-6.0, goal_lufs)) if goal_lufs is not None else None
    current = session_analysis.get("lufs_i")
    if goal is not None:
        rationale = (f"auto mix: master to {goal:.0f} LUFS at -1.0 dBTP"
                     + (f" for {intent}" if intent else ""))
        impact = [{"metric": "lufs_i",
                   "predicted_delta": (round(goal - current, 1)
                                       if current is not None else 0.0),
                   "unit": "LUFS"}]
    else:
        rationale = ("auto mix: hold the master at -1.0 dBTP (no loudness target "
                     "in the brief, so the ceiling is guaranteed and the level "
                     "is left alone)")
        impact = [{"metric": "true_peak_dbtp",
                   "predicted_delta": (round(min(0.0, -1.0 - tp), 1)
                                       if (tp := session_analysis.get(
                                           "true_peak_dbtp")) is not None
                                       else 0.0),
                   "unit": "dBTP"}]
    bundles.append({
        "rationale": rationale,
        "predicted_impact": impact,
        "actions": [{"type": "limiter", "stem_id": "master",
                     "ceiling_dbtp": -1.0, "target_lufs_i": goal}],
    })
    return bundles


async def _apply_bundles(engine, emit: EmitFn, bundles: list[dict],
                         gate=None, *, should_cancel=None,
                         applied: Optional[list[str]] = None,
                         rejected: Optional[list[str]] = None) -> list[str]:
    """Land a list of proposal bundles through the ordinary lifecycle: emit the
    proposal/status feed and advance a pass at the proposals-per-pass guardrail.
    Shared by bootstrap() and cleanup(); returns the applied proposal ids (the
    one-click-undo set). `gate(proposal_id, bundle) -> bool` (Phase 40): when given,
    each bundle waits for it before applying — Ask mode turns every Auto Mix move
    into an approve/reject card; None (the default) applies straight through.

    `applied` is the CALLER'S list, appended to as each bundle lands. That matters
    more than it looks: the only way to stop Auto Mix used to be `task.cancel()`,
    which raises CancelledError mid-loop, so the return value never reached the
    caller — and the ids of everything already applied were lost. The gateway then
    reported "applied: 0" while N bundles sat in the session OUTSIDE the one-click
    undo set, unreachable by the button that exists to take them back. Passing the
    list in means a cancelled Auto Mix is still fully undoable.

    `should_cancel` is the cooperative path: checked between bundles, so a stop
    lands on a bundle boundary with the session consistent, instead of tearing a
    propose/apply pair in half.
    """
    if applied is None:
        applied = []
    proposals_this_pass = 0
    queue = list(bundles)
    while queue:
        bundle = queue.pop(0)
        # What to propose INSTEAD if this bundle is refused (build_bundles: a group's members
        # get their own compressors). Popped either way - the engine is never sent the key.
        if_refused = bundle.pop("_if_refused", None) or []
        if should_cancel is not None and should_cancel():
            await emit({"type": "automix_stage", "stage": "cancelled",
                        "detail": f"stopped after {len(applied)} move(s)"})
            break
        # A bundle marked `_own_pass` (the intent preset) starts a fresh pass (2026-09-10 audit
        # P1-23): the per-pass gain wallet is shared, and "vocal-forward"'s +1.5 dB was refused
        # because the phrase rides earlier in the same pass had already spent it.
        own_pass = bool(bundle.pop("_own_pass", False))
        if proposals_this_pass >= BUNDLES_PER_PASS or (own_pass and proposals_this_pass):
            await engine.call("advance_pass")
            proposals_this_pass = 0
        result = await engine.call("propose_actions", {"req": bundle})
        proposals_this_pass += 1
        if not result["accepted"]:
            await emit({"type": "proposal_rejected", "pass_number": 0,
                        "proposal": bundle,
                        "violations": result["violations"]})
            # …and a refusal SAYS so. It reached only the event above, which nothing in the
            # DAW reads, so the feed said "Auto Mix applied" over a preset that never landed.
            why = "; ".join(str(v.get("message", "")) for v in result["violations"])
            label = bundle["rationale"].split(":")[0]
            if rejected is not None:
                rejected.append(f"{label} ({why})")
            await emit({"type": "automix_stage", "stage": "refused",
                        "detail": f"{label} \u2014 {why}"})
            queue[0:0] = if_refused
            continue
        proposal_id = result["proposal_id"]
        await emit({"type": "proposal", "pass_number": 0,
                    "proposal_id": proposal_id,
                    "rationale": bundle["rationale"],
                    "actions": bundle["actions"],
                    "predicted_impact": bundle["predicted_impact"]})
        if gate is not None and not await gate(proposal_id, bundle):
            await emit({"type": "proposal_status", "proposal_id": proposal_id,
                        "status": "skipped", "message": "rejected"})
            continue
        outcome = await engine.call("apply_proposal",
                                    {"proposal_id": proposal_id})
        await emit({"type": "proposal_status", "proposal_id": proposal_id,
                    "status": "applied" if outcome["ok"] else "skipped",
                    "message": outcome.get("message", "")})
        if outcome["ok"]:
            applied.append(proposal_id)
    await engine.call("advance_pass")  # the next pass / loop gets fresh budgets
    return applied


def is_cleanup_brief(brief: str) -> bool:
    """The AI-cleanup preset's route (Phase 29): separation-artifact repair, not mixing.
    Keyed on 'artifact' so an ordinary 'clean up the mix' brief still routes to the
    bootstrap — the menu item passes 'clean up separation artifacts'."""
    return "artifact" in (brief or "").lower()


async def bootstrap(engine, brief: str, emit: EmitFn, gate=None, *,
                    should_cancel=None,
                    applied: Optional[list[str]] = None,
                    existing_rides: Optional[dict[str, list]] = None,
                    rejected: Optional[list[str]] = None,
                    leave_out: Optional[set[str]] = None,
                    grouped: Optional[set[str]] = None) -> list[str]:
    """Profile → classify → starting chain, applied through the proposal
    lifecycle. Returns the applied proposal ids (the one-click-undo set —
    the gateway appends the refinement loop's ids to it). `gate` (Phase 40) makes
    each starting-chain move an approve/reject card in Ask mode (see _apply_bundles).

    `should_cancel` is polled at every stage boundary AND inside the profiling
    loops, because those are the slow part: one analyze_stem per stem plus one
    analyze_masking per suspect pair is where a big session spends its minutes, and
    a stop that only took effect between stages would sit there ignoring the user.
    `applied` is the caller's undo set — see _apply_bundles for why it is passed in.
    `leave_out` names stems that are not in the mix (a deleted track's stem stays in the
    session, excluded): never profiled, never proposed on - a bundle on one is refused by the
    guardrails, and Auto Mix then reported "refused" over tracks the user had deleted. A MUTED
    track is left out the same way (real use, 2026-10-22): it does not play, so it is not
    balanced against, grouped or processed. `grouped` names the tracks already in a group,
    which no family group takes (_bus_groupings).
    """
    def cancelled() -> bool:
        return should_cancel is not None and should_cancel()

    if applied is None:
        applied = []
    intent = intent_from_brief(brief)
    goal_lufs = _loudness_goal(brief.lower())
    if goal_lufs is None:
        goal_lufs = _intent_goal_lufs(intent)

    await emit({"type": "automix_stage", "stage": "tagging"})
    tags = await engine.call("auto_tag_stems")

    if cancelled():
        return applied

    await emit({"type": "automix_stage", "stage": "profiling"})
    session_analysis = await engine.call("analyze_session")
    stems = {}
    phrases = {}
    for stem_id in session_analysis["stems"]:
        if leave_out and stem_id in leave_out:
            continue
        if cancelled():
            return applied
        stems[stem_id] = await engine.call("analyze_stem", {"stem_id": stem_id})
        if "vocal" in set(tags.get(stem_id, [])):
            phrases[stem_id] = await engine.call("analyze_phrases",
                                                 {"stem_id": stem_id})
    candidates = suspect_pairs(stems)          # ranked worst-collision first
    maskings = []
    for stem_a, stem_b in candidates[:MAX_MASKING_PAIRS]:
        if cancelled():
            return applied
        try:
            maskings.append(await engine.call(
                "analyze_masking", {"stem_a": stem_a, "stem_b": stem_b}))
        except RuntimeError:
            continue  # e.g. the maskee is never active — nothing to protect
    try:
        sections = (await engine.call("analyze_sections")).get("sections", [])
    except (RuntimeError, KeyError):
        sections = []  # short/odd audio — skip the ride, never crash bootstrap

    skipped = max(0, len(candidates) - MAX_MASKING_PAIRS)
    await emit({"type": "automix_stage", "stage": "building",
                "detail": f"{len(stems)} stems, "
                          f"{len(maskings)} masking pairs measured"
                          + (f" ({skipped} lower-ranked pair(s) not measured)"
                             if skipped else "")})
    bundles = build_bundles(tags, stems, maskings, session_analysis, goal_lufs,
                            sections=sections, phrases=phrases, intent=intent,
                            existing_rides=existing_rides, grouped=grouped)

    await emit({"type": "automix_stage", "stage": "applying",
                "detail": f"{len(bundles)} bundle(s)"})
    await _apply_bundles(engine, emit, bundles, gate,
                         should_cancel=should_cancel, applied=applied, rejected=rejected)

    # Skip the render on a cancel — it is the most expensive step here and the user
    # just asked us to stop. What landed is still applied and still undoable; the
    # caller re-renders when it next needs numbers.
    if cancelled():
        return applied

    await emit({"type": "automix_stage", "stage": "rendering"})
    render = await engine.render_and_analyze()
    await emit({"type": "metrics", "analysis": render["analysis"],
                "flags": render.get("flags", []),
                "master": render.get("master"),
                "bounce_path": render.get("bounce_path")})
    return applied


def build_cleanup_bundles(report: dict, tags: Optional[dict] = None, *,
                          offline_only: bool = True,
                          left_out: Optional[list] = None) -> list[dict]:
    """AI-cleanup preset (Phase 29, pure policy): repair separation artifacts
    from an analyze_separation report. De-noise stems with musical-noise shimmer
    or inter-stem leakage, restore attacks on transient-smeared non-pad stems, and
    saturate over spectral holes. Bounded + deterministic; lands through the ordinary
    proposal lifecycle (cleanup()). It does NOT touch relative loudness — see the note
    where the gain-matching used to be.

    ``offline_only=False`` is the DAW (full audit 2026-10-02, reader E4·R3): the spectral
    de-noise exists only in the offline render - the native chain has no such device, so in
    the DAW it was in neither the monitor nor the export, while the proposals predicted its
    improvement, the closing metrics came from a render that included it and the feed said
    "cleaned". There it is not proposed at all, and the stems it would have been applied to
    are appended to ``left_out`` so the caller can say so."""
    per_stem = report.get("per_stem") or []
    tags = tags or {}
    denoise: list[tuple[dict, Optional[dict]]] = []
    transients: list[tuple[dict, Optional[dict]]] = []
    saturation: list[tuple[dict, Optional[dict]]] = []
    repaired: set[str] = set()
    warnings: list[str] = []
    for art in sorted(per_stem, key=lambda a: a["stem_id"]):
        if art.get("silent"):
            continue
        sid = art["stem_id"]
        shimmer = art.get("hf_shimmer") or 0.0
        leakage = art.get("leakage") or 0.0
        if (shimmer >= SHIMMER_DENOISE_AT or leakage >= LEAKAGE_DENOISE_AT) and not offline_only:
            if left_out is not None:
                left_out.append(sid)
        elif shimmer >= SHIMMER_DENOISE_AT or leakage >= LEAKAGE_DENOISE_AT:
            reason = "bleed" if leakage >= LEAKAGE_DENOISE_AT else "shimmer"
            denoise.append((
                {"type": "spectral_denoise", "stem_id": sid,
                 "reduction_db": DENOISE_REDUCTION_DB,
                 "sensitivity_db": DENOISE_SENSITIVITY_DB},
                {"metric": (f"separation({sid}).leakage" if reason == "bleed"
                            else f"separation({sid}).hf_shimmer"),
                 "predicted_delta": -round(0.5 * max(shimmer, leakage), 2),
                 "unit": "ratio"}))
            repaired.add(sid)
        elif (max(shimmer, leakage)
              >= min(SHIMMER_DENOISE_AT, LEAKAGE_DENOISE_AT) - CLEANUP_WATCH_MARGIN):
            warnings.append(f"{sid} shimmer/leakage near threshold "
                            f"({shimmer:.2f}/{leakage:.2f})")
        smear = art.get("transient_smear") or 0.0
        if smear >= SMEAR_RESTORE_AT and "pad" not in set(tags.get(sid, [])):
            transients.append((
                {"type": "transient", "stem_id": sid,
                 "attack_db": TRANSIENT_ATTACK_DB, "sustain_db": 0.0},
                {"metric": f"separation({sid}).transient_smear",
                 "predicted_delta": -round(0.3 * smear, 2), "unit": "ratio"}))
            repaired.add(sid)
        elif smear >= SMEAR_RESTORE_AT - CLEANUP_WATCH_MARGIN:
            warnings.append(f"{sid} transient smear near threshold ({smear:.2f})")
        holes = art.get("spectral_hole_ratio") or 0.0
        if holes >= SPECTRAL_HOLE_REPAIR_AT:
            saturation.append((
                {"type": "saturation", "stem_id": sid, "drive_db": 2.0},
                {"metric": f"separation({sid}).spectral_hole_ratio",
                 "predicted_delta": -round(0.2 * holes, 2), "unit": "ratio"}))
            repaired.add(sid)
        elif holes >= SPECTRAL_HOLE_REPAIR_AT - CLEANUP_WATCH_MARGIN:
            warnings.append(f"{sid} spectral holes near threshold ({holes:.2f})")

    # NO loudness matching here (removed 2026-08-05). This was the same median-pull as
    # the old balance step, and on separated stems it is even more clearly wrong: the
    # stems sum back to the source track, so their loudness spread IS the source's
    # balance. Measured on a real separation, `inter_stem_lufs_spread_db` was 21.1 —
    # tripping LUFS_SPREAD_MATCH_AT and "correcting" a mix that was already correct.
    # `inter_stem_lufs_spread_db` stays a descriptive readout; separation pathology is
    # owned by the artifact metrics above (shimmer / smear / holes / leakage), not by
    # how loud the parts happen to be relative to each other.

    bundles: list[dict] = []
    warning_note = (f" Watched but left alone: {'; '.join(warnings[:3])}."
                    if warnings else "")
    if denoise:
        bundles += _chunk(denoise, "ai cleanup: spectral de-noise/de-bleed the "
                                   "stems whose separation metrics crossed the "
                                   f"repair threshold.{warning_note}")
    if transients:
        bundles += _chunk(transients, "ai cleanup: restore attacks on "
                                      f"transient-smeared stems.{warning_note}")
    if saturation:
        bundles += _chunk(saturation, "ai cleanup: add gentle saturation to "
                                      "mask spectral holes in separated stems."
                                      f"{warning_note}")
    glue_members = sorted(repaired)
    if len(glue_members) >= CLEANUP_GLUE_REPAIRED_AT:
        bundles += _chunk([
            ({"type": "group", "bus_id": "cleanup_glue",
              "members": glue_members}, None),
            ({"type": "comp", "stem_id": "cleanup_glue",
              "threshold_db": CLEANUP_GLUE_THRESHOLD_DB,
              "ratio": CLEANUP_GLUE_RATIO,
              "attack_ms": GLUE_ATTACK_MS, "release_ms": 200.0,   # was 20 ms: see GLUE_ATTACK_MS
              "makeup_db": 0.0, "mix": CLEANUP_GLUE_MIX}, None),
        ], "ai cleanup: glue repaired separated stems on a light bus so "
           f"the cleanup feels coherent.{warning_note}")
    return bundles


#: what the preset says when no stem here came out of the app's own split (reader E5·R7)
NOTHING_SEPARATED = ("nothing in this project was split into stems here, so there are no "
                     "separation artifacts to repair - the check only reads stems that "
                     "\"Split into stems\" made (a stem split elsewhere and imported is "
                     "not one it can vouch for)")


async def cleanup(engine, emit: EmitFn, *, should_cancel=None,
                  applied: Optional[list[str]] = None,
                  offline_only: bool = True,
                  left_out: Optional[list] = None,
                  notes: Optional[list] = None) -> list[str]:
    """The AI-cleanup preset's engine side (Phase 29): read the separation
    'double check', propose repair, apply through the proposal lifecycle. Same
    one-click-undo set shape as bootstrap(); a single corrective pass (no
    loudness target / refinement loop — repair, not mastering).

    Cancellation works the same way as bootstrap's — see _apply_bundles."""
    def cancelled() -> bool:
        return should_cancel is not None and should_cancel()

    if applied is None:
        applied = []
    await emit({"type": "automix_stage", "stage": "tagging"})
    tags = await engine.call("auto_tag_stems")

    await emit({"type": "automix_stage", "stage": "profiling"})
    report = await engine.call("analyze_separation")
    if not report.get("per_stem"):
        # Nothing here was separated: there is nothing these metrics can be trusted on, and
        # nothing is proposed (E5·R7 - the preset used to "repair" a clean vocal's esses, a
        # guitar's sustain and three parts that merely play the same song form).
        if notes is not None:
            notes.append(NOTHING_SEPARATED)
        await emit({"type": "automix_stage", "stage": "building",
                    "detail": "no separated stems to check"})
        return applied

    await emit({"type": "automix_stage", "stage": "building",
                "detail": f"{len(report.get('per_stem', []))} stems checked"})
    bundles = build_cleanup_bundles(report, tags, offline_only=offline_only, left_out=left_out)

    await emit({"type": "automix_stage", "stage": "applying",
                "detail": f"{len(bundles)} repair bundle(s)"})
    await _apply_bundles(engine, emit, bundles,
                         should_cancel=should_cancel, applied=applied)

    if cancelled():
        return applied

    await emit({"type": "automix_stage", "stage": "rendering"})
    render = await engine.render_and_analyze()
    await emit({"type": "metrics", "analysis": render["analysis"],
                "flags": render.get("flags", []), "master": render.get("master"),
                "bounce_path": render.get("bounce_path")})
    return applied
