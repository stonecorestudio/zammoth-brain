"""Server-side guardrails. The agent cannot talk its way past this module.

All limits live in LIMITS so the UI can display them and tests can assert them.
See docs/03-mcp-tools.md for the rule table.
"""
from __future__ import annotations

import contextlib
import math
from typing import Optional

from .models import (BAND_EDGES_HZ, BAND_LABELS,
                     ClipAction, CompAction, DeessAction, DynamicEqAction, EqAction,
                     GainAction,
                     GateAction, LimiterAction, MatchEqAction,
                     MultibandCompAction,
                     ProposalRequest, ReturnConvolutionAction, ReturnDelayAction,
                     ReturnReverbAction,
                     OttAction, OverdriveAction, ChorusAction, FlangerAction, AutoFilterAction,
                     SaturationAction, SendAction, SidechainDuckAction,
                     SpectralDenoiseAction, TiltAction, TransientAction,
                     TransposeAction, TuneAction,
                     UtilityAction, Violation, WidthAction)

# Mirrors dsp.MB_MAX_BOOST_DB (importing dsp here would drag pedalboard/scipy into every
# guardrail call); test_guardrail_mb_up_charge asserts they agree.
MB_MAX_BOOST_DB = 12.0



def level_charge_db(action) -> Optional[float]:
    """The level ONE action (or one chain entry) adds to its target, as the gain wallets count
    it — or None for a type no wallet charges. ONE function for the guardrail check
    (`validate_proposal`), the ledger that seeds the next check (`Session._record_trim_charge`)
    and the session baseline read off the live chains (full audit 2026-10-02, D9·R1–R3): the
    three used to carry their own formulas, and what the check charged the ledger then forgot
    (M/S comp makeup, a multiband upward boost, a ranged comp's makeup) — +42 dB on one stem in
    one pass with the wallet reading +6.

    Signed for `gain` and `utility` (a real cut is a real cut); >= 0 for every device trim (a
    trim cut is free, never credit). Stages in SERIES add: OTT's input, band and output gains
    multiply in `dsp._ott`, and an upward multiband band is boosted and THEN made up. Bands in
    PARALLEL take their max. Upward multiband compression is charged its cap, MB_MAX_BOOST_DB,
    at any ratio > 1: `min(under * (1 - 1/ratio), cap)` reaches the cap as the band falls away
    from its threshold, whatever the ratio (the old `cap * (1 - 1/ratio)` under-read by
    cap/ratio — ratio 2 was charged 6 and lifted 12)."""
    d = action if isinstance(action, dict) else action.model_dump()
    t = d.get("type")

    def pos(v) -> float:
        return max(0.0, float(v or 0.0))

    def blocks(*names):
        return [d.get(n) for n in names if isinstance(d.get(n), dict)]

    if t == "gain":
        return float(d.get("delta_db") or 0.0)
    if t == "utility":
        return float(d.get("gain_db") or 0.0)
    if t == "overdrive":
        # An amp character keeps its level as it is driven (od_characters.level_db, the pre-merge
        # review of 2026-10-26): its small-signal lift is +3.3..+6.5 dB at drive 0 and rises past its
        # ceiling - a lift no trim shows, charged like the saturation's. The pedal's is 0.
        from . import od_characters as odc
        ch = odc.character_index(d.get("character", odc.DEFAULT))
        drive = float(d.get("drive_db") if d.get("drive_db") is not None else 12.0)
        return pos(odc.level_db(drive, ch) + float(d.get("output_db") or 0.0))
    if t == "clip":
        # The clipper's GAIN is before the clip, and nothing takes it back out after (clip_law,
        # 2026-10-28): what sits under the knee comes out that much louder.
        from .clip_law import lift_db
        return pos(lift_db(d))
    if t in ("eq", "tilt"):
        return pos(d.get("output_db"))
    if t == "chorus":
        # The chorus's law (chorus_law, 2026-10-28): its MIX curve keeps the dry full and lays the
        # wet on top (+4.5 dB on music at 0.5, +6.6 for I+II) - a lift no knob shows - plus VOLUME.
        from .chorus_law import lift_db
        return pos(lift_db(d))
    if t == "transient":
        # The shaper applies `attack_db * t + sustain_db * (1 - t)` and THEN the output trim
        # (dsp._transient): attack and sustain both at +6 is a flat +6 dB no wallet saw, and a
        # gain +6 in the same proposal put the stem +12 up in one pass (reader D9·R6). The
        # larger of the two is the most it lifts anything; a cut is free, as everywhere.
        return (pos(max(float(d.get("attack_db") or 0.0), float(d.get("sustain_db") or 0.0)))
                + pos(d.get("output_db")))
    if t == "saturation":
        # The character's level law lifts with drive (sat_characters.LEVEL, 2026-10-26: +0.8..+4 dB
        # at drive 0, up to ~+8 on a band at 24 dB, TUBE +13) - a lift no trim shows. entry_lift_db reads the
        # drives as the render does (a band's own output trim included); the entry's own trim adds.
        from .sat_characters import entry_lift_db
        if blocks("low", "mid", "high"):
            # The multiband form plays each band's own trim and NOT the flat one (dsp's and the
            # native's multiband paths ignore it): a flat -12 must not cancel three bands at +12
            # it never plays against (pre-merge review 2026-10-26).
            return pos(entry_lift_db(d))
        return pos(entry_lift_db(d) + float(d.get("output_db") or 0.0))
    if t == "ott":
        # OTT's law (ott_law, 2026-10-28) lifts every band by a FIXED makeup x depth (+14 / +9 / +14 dB
        # at full depth) before its downward side cuts loud material back: that lift is charged with
        # the band's own gain, the largest band, in series with the input and output gains.
        from .ott_law import makeup_lift_db
        return pos(d.get("in_gain_db")) + pos(makeup_lift_db(d)) + pos(d.get("out_gain_db"))
    if t == "comp":
        return max([pos(d.get("makeup_db"))]
                   + [pos(b.get("makeup_db")) for b in blocks("mid", "side")])
    if t == "mb_comp":
        return max([pos(b.get("makeup_db"))
                    + (MB_MAX_BOOST_DB if b.get("up") and float(b.get("ratio") or 1.0) > 1.0
                       else 0.0)
                    for b in blocks("low", "mid", "high")] or [0.0])
    return None


def is_ranged(action) -> bool:
    """A move confined to `time_range_s`. The wallets count it ONE-SIDED: a ranged boost adds to
    what the target may be lifted somewhere, a ranged cut to what it may be lowered somewhere,
    and neither refunds the other (D9·R4 — counted as signed whole-file moves, a −12 dB cut on
    the chorus was credit a +12 dB boost on the verse then spent, every pass, without limit)."""
    rng = action.get("time_range_s") if isinstance(action, dict) \
        else getattr(action, "time_range_s", None)
    return rng is not None


def ranged_peak(spans) -> tuple[float, float]:
    """(up, down): the most ranged lift, and the most ranged cut, that play AT ONE MOMENT among
    `spans` = (start_s, end_s, charge_db). Still one-sided - lifts and cuts are never netted
    (D9·R4) - but two moves on ranges that do not overlap are not one move: "down 4 dB
    everywhere except the drop" is two ranges and -4 at any moment, and summed it read -8
    and was refused (the mixing audit of 2026-11-10, M4)."""
    def most(items) -> float:
        best = 0.0
        for start, _, _ in items:
            at = sum(c for s, e, c in items if s <= start < e)
            best = max(best, abs(at))
        return best
    ups = [(float(s), float(e), float(c)) for s, e, c in spans if c > 0]
    downs = [(float(s), float(e), float(c)) for s, e, c in spans if c < 0]
    return most(ups), -most(downs)


def _span_of(action) -> Optional[tuple[float, float]]:
    rng = action.get("time_range_s") if isinstance(action, dict) \
        else getattr(action, "time_range_s", None)
    try:
        start, end = float(rng[0]), float(rng[1])
    except (TypeError, ValueError, IndexError):
        return None
    return (start, end) if end > start else None


LIMITS = {
    # MonitorChain::kMaxBands — the native EQ bank holds 32 bands across eq / tilt (2 each) /
    # match_eq entries and silently DROPPED the rest while dsp.py applied them (2026-09-01
    # sweep). Refused here for the agent (and in add_stem_device for the user) instead.
    "eq_bands_max": 32,
    # MonitorChain::kMaxComps / kMaxDeess — the native chain holds this many COMP and DE-ESS
    # slots. Guardrails deliberately allow several non-overlapping time-ranged ones per stem
    # (Phase 35 phrase compression is exactly that), and until 2026-09-08 the native folded them
    # all onto ONE slot: the last entry's params ran under the first entry's range, an 18.86 dB
    # divergence between the monitor/bounce and the offline render. The native now gives each
    # entry its own slot; these refuse the N+1th here, the way eq_bands_max does, so the overflow
    # is a violation the agent can read rather than an entry that silently vanishes.
    # Since Phase 65 the native chain runs a comp per entry (kMaxComps = 128): this is now the
    # AGENT's own bound on how many compressors its moves leave on a chain, entries already there
    # counted - a policy, held below the engine's count (test_guardrail_holes).
    "comps_per_stem_max": 4,
    "deessers_per_stem_max": 4,
    "gain_per_stem_per_pass_db": 6.0,
    # Phase 10: cumulative agent-channel gain across the whole session. The
    # per-pass clamp resets on advance_pass (every chat move is its own
    # pass), so without this an agent could walk a stem ±6 dB per message
    # indefinitely. Undoing a gain proposal refunds its budget.
    "gain_per_stem_per_session_db": 12.0,
    "eq_boost_max_db": 6.0,
    "eq_cut_min_db": -12.0,
    # The COMPOSED static-EQ response per target (design call 2026-09-11, docs/10-daw.md): the
    # agent's applied-and-not-undone eq / tilt / match_eq moves multiply into one response, and
    # that response is bounded in the gain wallet's shape — per pass and per session. The per-band
    # rails above only ever saw ONE band: eight legal +6 dB bells at 2.5 kHz measured +21 dB
    # (2026-09-10 audit 05·F13). Capping the total at 6 was tried and dropped the pop-master
    # closure 0.95 -> 0.42: a band honestly 8 dB under a reference needs two passes of reach.
    "eq_boost_pass_db": 6.0,
    "eq_boost_session_db": 12.0,
    "eq_cut_pass_db": -12.0,
    "eq_cut_session_db": -24.0,
    # The SAME +6 dB ceiling, applied to the resonant peak a filter shape produces.
    # hpf/lpf/notch/bandpass ignore `band.gain_db` entirely, so `eq_boost_max_db`
    # never saw them: a legal `lpf q=8 slope_db_oct=96` measures +162 dB at its
    # corner (each cascade section's Q scales by q/BUTTERWORTH_Q) and validated
    # clean, while a plain +12 dB bell was rejected. Checked against the band's
    # ACTUAL cascade response, not a proxy Q bound, so it cannot drift from the DSP.
    "eq_resonance_max_db": 6.0,
    "utility_balance_max": 0.5,        # agent L/R skew on a stem/bus; 0 on the master
    # 2026-09-08 device audit (07·F3): the master's stereo image had NO floor, only the
    # schema's 0..2 range — and the range includes total destruction. Measured on a
    # synthetic master, validate_proposal accepted `utility mono`, `width 0.0` and
    # a 3-band imager at 0/0/0 with ZERO violations while refusing a 1.92 dB `balance` nudge; the
    # 12.39 dB mono-sum collapse was free and the milder move was banned. Half the side
    # is a real "tighten the image" mastering move; below it the agent is monoing the
    # deliverable, which is never an automatic decision. Binds the AGENT only — the
    # user's own width knob still goes to 0 (shared control), and folding the LOWS to
    # mono stays free at any depth (see _MASTER_MONO_LOWS_MAX_HZ).
    "master_width_min": 0.5,
    "dynamic_eq_cut_min_db": -12.0,
    "hpf_max_hz_default": 120.0,
    "hpf_max_hz_bass_tagged": 40.0,
    # where engineers really high-pass (2026-11-10): hats and cymbals, a voice
    "hpf_max_hz_cymbal_tagged": 500.0,
    "hpf_max_hz_vocal_tagged": 200.0,
    "comp_ratio_max": 10.0,
    "master_true_peak_max_dbtp": -1.0,
    "master_comp_makeup_max_db": 0.0,  # only the limiter pushes loudness (Phase 6)
    "saturation_drive_max_db": 12.0,   # Phase 7
    # OTT's UPWARD side lifts quiet material, and no wallet charges it: at the old cap of 0.8 that
    # was 19 dB (a -45 dB stem measured +8.4 with every gain at 0). Charging the worst case would
    # take OTT on a stem away from the agent, as it did for upward multiband; the cap is set instead
    # so the uncharged lift stays near the per-pass rail (James, 2026-10-05, reader D9·R6). Under
    # ott_law (2026-10-28) the FIXED makeup (+14 dB x depth at most, 3.5 at 0.25) is charged as gain
    # (level_charge_db); the upward side beyond it at 0.25 is 3-4 dB for a band 15 dB below its
    # threshold and 6-7 for one 30 dB below. The RUN's cap only: the user's knob goes to 1.
    "ott_depth_max": 0.25,
    "overdrive_drive_max_db": 24.0,    # 2026-07-07: agent overdrive (user knob to 36)
    # 2026-09-06 modulation: the agent may THICKEN, not swirl. Chorus is the one device
    # whose damage is not level but smearing — past a half-wet blend a mono source stops
    # being a point and a doubled vocal starts to phase. The user knob goes to fully wet.
    "chorus_mix_max": 0.5,
    # The flanger's blend is capped the same way, and its RETURN PATH is capped separately:
    # a comb whose feedback approaches 1 rings on the edge of self-oscillation, which is a
    # sound the user can choose and the agent must not stumble into while "adding movement".
    # Past ~0.7 the resonance also adds real level at the peaks, which no gain budget sees.
    "flanger_mix_max": 0.5,
    "flanger_feedback_max": 0.7,       # |feedback|; the user knob still goes to 0.95
    # Phase 63: the Auto Filter. Its RESONANCE is a static lift at the cutoff that no wallet
    # charges (a swept peak is not an EQ boost the composed rail can hold), so the cap is sized
    # to the pass rail instead (CLAUDE.md batch Py1 rule 2): at 0.64 the peak is 5.95 dB at 12
    # dB/oct and 5.6 at 24 - inside eq_resonance_max_db's 6 (auto_filter_law since 2026-10-27; it was
    # 0.3 on the old exponential Q). Its sweep is capped to keep
    # "movement" movement: three octaves either way from the LFO, three from the envelope.
    "auto_filter_resonance_max": 0.64,  # the user knob still goes to 1 (Q 27)
    "auto_filter_lfo_amount_max_oct": 3.0,
    "auto_filter_env_amount_max_oct": 3.0,   # |env_amount_oct|
    "clip_threshold_min_db": -6.0,     # Phase 41: agent can't clip deeper than -6 dBFS
    "gate_threshold_max_db": -20.0,    # gate v2: an agent gate above -20 dBFS eats program
                                       # material, not bleed (the user knob still goes to 0)
    "tilt_max_db": 6.0,                # Phase 41: agent tilt capped at ±6 (like eq boosts)
    "duck_depth_max_db": 12.0,         # Phase 7: the most a duck may cut, at its curve's deepest
                                       # (duck_law: 1 - mix at a curve's 0 - so a mix of at most 0.75)
    "spectral_denoise_reduction_max_db": 24.0,  # Phase 29 repair de-noise gate depth
    "tune_cents_max": 100.0,  # Phase 31: max |tuning correction|, a semitone
    "transpose_semitones_max": 12,  # Phase 32: max |whole-song transpose|, an octave
    "return_send_max_db": -3.0,  # Phase 38: keep ambience sends below unity
    "return_reverb_decay_max_s": 12.0,
    "return_convolution_decay_max_s": 8.0,   # Tier-1 IR reverb (model also caps at 8 s)
    "return_delay_feedback_max": 0.85,
    # Space devices as INSERTS (2026-07-20): the agent may blend a track's own reverb/delay
    # in, but not drown it — past half wet the dry source stops being the mix. Binds the
    # AGENT only; the user's dry/wet knob goes to 100 % (shared control).
    "wet_insert_mix_max": 0.5,
    # 2026-06 review: stacked compressors were never intended. A new
    # whole-file comp *replaces* the stem's existing one (session._mutate);
    # ranged comps coexist only on non-overlapping ranges (validated here).
    "whole_file_comps_per_stem": 1,
    "actions_per_proposal": 8,
    "proposals_per_pass": 3,
}


def pass_cap_among(violations) -> bool:
    """Whether the per-pass proposal cap is AMONG `violations` (Violation models or their dicts).

    Every door that heals a full pass (the live engineer, Ask's chat, a plan's preview, the
    transpose helper) opens a fresh pass and asks again when this is true, and the answer it
    returns is the fresh pass's. Until 2026-11-02 they healed only when the cap was the ONLY
    violation: a move that also broke a per-pass budget came back "too many proposals this
    pass" first, and the engineer told the user it had hit a cap and made the same move with
    a direct fader write, outside every guardrail."""
    for v in violations or ():
        rule = v.get("rule") if isinstance(v, dict) else getattr(v, "rule", None)
        if rule == "proposals_per_pass":
            return True
    return False


#: The device types the user may stack on one chain (Phase 65) that a mix proposal REPLACES rather
#: than adds - a proposal of one is refused on a channel that holds two (rule instance_which).
INSTANCE_KINDS = ("comp", "gate", "transient", "duck", "limiter", "mb_comp", "ott", "clip",
                  "saturation", "overdrive", "chorus", "flanger", "auto_filter", "utility")


def _field(action, name: str):
    return action.get(name) if isinstance(action, dict) else getattr(action, name, None)


def _after_last_limiter(chain, entry) -> bool:
    """Whether `entry` stands after the last limiter of `chain` (the master's user-only tail)."""
    chain = list(chain or ())
    lims = [i for i, e in enumerate(chain) if e.get("type") == "limiter"]
    at = next((i for i, e in enumerate(chain) if e is entry), None)
    return bool(lims) and at is not None and at > lims[-1]


def replaced_entries(chain, action) -> list[dict]:
    """The entries of `chain` a whole-file proposal of an INSTANCE_KINDS type replaces
    (session._mutate_one): no `instance` = every whole-file one of its type (the guardrails
    refuse that where there are two or more), `instance: N` = the type's N-th entry (counted
    over every entry, as `comp#N` is) when it is whole-file, `"new"` = none."""
    kind = _field(action, "type")
    if kind not in INSTANCE_KINDS or _field(action, "time_range_s") is not None:
        return []
    inst = _field(action, "instance")
    of_type = [e for e in (chain or ()) if e.get("type") == kind]
    if inst == "new":
        return []
    if inst is None:
        return [e for e in of_type if not e.get("time_range_s")]
    if 0 <= inst < len(of_type) and not of_type[inst].get("time_range_s"):
        return [of_type[inst]]
    return []


def instance_slot(kind: str, uid) -> str:
    """The ledger's key for one instance of a replacing type: `type@uid` once the entry has an
    identity, the bare type for a record from before identities (it stands for any of them)."""
    return f"{kind}@{uid}" if isinstance(uid, int) else kind


def slot_replaced(slot: str, replaced) -> bool:
    """Whether the ledger slot `slot` is one the proposal replaces: `replaced` holds slots
    (`type@uid`) and bare types (every instance of the type)."""
    kind, _, uid = slot.partition("@")
    if slot in replaced or kind in replaced:
        return True
    return not uid and any(r.partition("@")[0] == kind for r in replaced)


def replaced_slots(chain, action) -> set[str]:
    """The ledger slots one action replaces on its target: the replaced entries' identities,
    or the bare type for a replace whose entry has none (or nothing yet stands)."""
    kind = _field(action, "type")
    if _field(action, "time_range_s") is not None:
        return set()
    if kind not in INSTANCE_KINDS:
        return {kind}
    if _field(action, "instance") == "new":
        return set()
    gone = replaced_entries(chain, action)
    if _field(action, "instance") is None and not gone:
        return {kind}
    return {instance_slot(kind, e.get("uid")) for e in gone}


def _comp_coverage(stem_chains: Optional[dict[str, list[dict]]],
                   device_type: str = "comp",
                   ) -> dict[str, list[Optional[tuple[float, float]]]]:
    """Per stem, the time ranges existing entries of `device_type` cover (None =
    whole-file). Seeds the comp-stacking check below, and the de-ess slot count."""
    coverage: dict[str, list[Optional[tuple[float, float]]]] = {}
    for stem_id, chain in (stem_chains or {}).items():
        for entry in chain:
            if entry.get("type") != device_type:
                continue
            range_s = entry.get("time_range_s")
            coverage.setdefault(stem_id, []).append(
                tuple(range_s) if range_s else None)
    return coverage


#: The devices whose gain reduction adds up in series (analysis.STACKED_LEVELLING): a track may
#: carry one of them whole-file from the engineer (2026-11-02).
LEVELLING_KINDS = ("comp", "mb_comp", "ott")


def _stacked_levelling(req: ProposalRequest,
                       stem_chains: Optional[dict[str, list[dict]]]) -> list[Violation]:
    """Refuse a proposal that would leave a track or the master with two whole-file levelling
    devices (2026-11-02). Told that compressors in series add up, the engineer still kept two on a
    rap taking 12.4 dB together: a note can be weighed away, a refusal cannot. A move that
    replaces the one there is taken; what stands is counted whoever placed it (a device the
    engineer adds by command is filed as the user's)."""
    if stem_chains is None:
        return []
    out: list[Violation] = []
    targets: dict[str, list[tuple[int, object]]] = {}
    for i, action in enumerate(req.actions):
        if (getattr(action, "type", None) in LEVELLING_KINDS
                and getattr(action, "time_range_s", None) is None):
            targets.setdefault(getattr(action, "stem_id", None) or "master", []).append((i, action))
    for target, moves in targets.items():
        chain = stem_chains.get(target) or []
        standing = [e for e in chain
                    if e.get("type") in LEVELLING_KINDS and not e.get("time_range_s")]
        gone = {id(e) for _, a in moves for e in replaced_entries(chain, a)}
        kept = [e for e in standing if id(e) not in gone]
        adding = [i for i, a in moves if not replaced_entries(chain, a)]
        if adding and len(kept) + len(adding) > 1:
            have = ", ".join(e["type"] for e in kept) or "this proposal's own"
            out.append(Violation(
                action_index=adding[-1], rule="levelling_stacked",
                message=f"'{target}' would carry {len(kept) + len(adding)} whole-file levelling "
                        f"devices ({have} already) and their gain reduction adds up in series; "
                        "adjust the one there by proposing it again, or take it off with "
                        "device.remove first",
                limit=1, value=len(kept) + len(adding)))
    return out


def _wet_insert_mix(i: int, action) -> list[Violation]:
    """Policy bound on a space device the agent places as an INSERT (2026-07-20). The
    RETURN form is exempt: a return is *meant* to be 100 % wet (its dry lives on the
    source tracks), so only an action carrying `stem_id` is checked. The user's own
    dry/wet knob is untouched — guardrails bind the agent, never the hands."""
    if not getattr(action, "stem_id", "") or action.mix <= LIMITS["wet_insert_mix_max"]:
        return []
    return [Violation(
        action_index=i, rule="wet_insert_mix_max",
        message=f"{action.type} on '{action.stem_id}' at {action.mix:.0%} wet drowns the "
                f"dry source (limit {LIMITS['wet_insert_mix_max']:.0%} for an insert; a "
                "100 % wet ambience belongs on a return)",
        limit=LIMITS["wet_insert_mix_max"], value=action.mix)]


# The one narrowing move that is NOT destruction: folding the bottom to mono. Our own
# prompt and knowledge file recommend it by name ("utility bass_mono"), and it IMPROVES mono
# compatibility rather than costing it. So the master width floor exempts a band whose TOP EDGE
# sits at or below this — and only that band: a bass mono pushed to 4 kHz would be the same
# destruction wearing the exemption's clothes (the Utility holds it to 500 Hz). 300 Hz is
# generous against the 120–200 Hz mono-maker defaults; above it lives the stereo image.
_MASTER_MONO_LOWS_MAX_HZ = 300.0


def _master_narrowing(i: int, what: str, width: float,
                      up_to_hz: float) -> list[Violation]:
    """Floor under the MASTER's stereo width (2026-09-08 audit 07·F3).

    `width` is the M/S side scale the DSP will ACTUALLY apply — a Utility's `mono`
    resolves to 0.0 here exactly as `dsp._utility` resolves it, so the rule cannot be
    walked past by spelling the same move differently. `up_to_hz` is the top edge of
    the band the scale applies to; a band that ends in the lows is exempt (above).
    Callers pass this only for `stem_id == "master"`: a stem or bus may be monoed
    freely, and so may the user's own hands on any target.
    """
    if width >= LIMITS["master_width_min"] or up_to_hz <= _MASTER_MONO_LOWS_MAX_HZ:
        return []
    return [Violation(
        action_index=i, rule="master_width_min",
        message=(f"{what} {width:.2f} narrows the finished mix below the "
                 f"{LIMITS['master_width_min']:.2f} floor — monoing the deliverable is "
                 f"the user's call, not an automatic move. Folding the LOWS is still "
                 f"free (a band ending at or below {_MASTER_MONO_LOWS_MAX_HZ:.0f} Hz: "
                 "utility bass_mono)."),
        limit=LIMITS["master_width_min"], value=width)]


def _action_eq_bands(action) -> int:
    """Bands this ACTION will occupy once applied — the request-side mirror of eq_bands_in.

    match_eq used to be charged NOTHING here while eq_bands_in counts an applied one as the full
    bank, so the tally and the thing it guards disagreed by 8. Since the contract began FITTING
    the bell bank to the offline FIR (2026-09-08), a two-key match_eq reaches the native as up to
    eight bands: a 25-band master + a match_eq validated clean and the contract then carried 33,
    with the 33rd silently dropped from the monitor and the DAW bounce — the exact failure
    eq_bands_max exists to prevent, newly reachable (2026-09-08 pre-merge review)."""
    if action.type == "eq":
        return 1
    if action.type == "tilt":
        return 2
    if action.type == "match_eq":
        # …but a correction of nothing costs nothing, exactly as eq_bands_in reads it.
        gains = getattr(action, "band_gains_db", None) or {}
        return len(BAND_LABELS) if any(abs(float(g)) > 0.01 for g in gains.values()) else 0
    return 0


#: The devices that scale the master's SIDE - the Imager (`width`) and the Utility - each one a
#: proposal replaces (session._mutate). The 3-band imager and band width were retired 2026-10-28.
_MASTER_WIDTH_KINDS = ("width", "utility")
_MASTER_WIDTH_SINGLETONS = ("width", "utility")
_WIDTH_TOP_HZ = 20000.0


def _side_pieces(d: dict) -> list[tuple[float, float, float]]:
    """(from Hz, to Hz, side scale) for one width device, exactly as the DSP applies it - a
    chain entry or an action's dump. A device that is switched off scales nothing."""
    if d.get("enabled", True) is False:
        return []
    t = d.get("type")
    top = float("inf")
    if t == "width":
        return [(0.0, top, float(d.get("width", 1.0)))]
    if t == "utility":                      # `mono` IS width 0 downstream (dsp._utility)
        return [(0.0, top, 0.0 if d.get("mono") else float(d.get("width", 1.0)))]
    return []


def _master_composed_width(req: ProposalRequest,
                           stem_chains: Optional[dict[str, list[dict]]]) -> list[Violation]:
    """The master's stereo floor, on the PRODUCT (reader D9·R10).

    Each device was held at `master_width_min` alone, and they are in series: width 0.5 x
    utility 0.5 x a 3-band imager 0.5 x band width 0.5 was 0.0625 of the side in the mids - the
    deliverable mono-folded (07·F3) from four legal moves, in one accepted proposal. The side
    scale the master would carry is composed here per frequency: what its chain holds, with a
    proposed width / utility REPLACING the one there (as `session._mutate` does). Judged above `_MASTER_MONO_LOWS_MAX_HZ` only, and only where
    THIS proposal narrows - the user's own narrowing is never the agent's refusal, so an EQ
    or a widening move on a master the user has narrowed still lands."""
    proposed = [(i, a.model_dump()) for i, a in enumerate(req.actions)
                if getattr(a, "stem_id", None) == "master" and a.type in _MASTER_WIDTH_KINDS]
    if not proposed:
        return []
    master_chain = (stem_chains or {}).get("master", [])

    def gone_with(d: dict) -> list[dict]:
        if d["type"] == "width":      # the master's width device stays one per chain
            return [e for e in master_chain if e.get("type") == "width"]
        return replaced_entries(master_chain, d)

    replaced = {id(e) for _, d in proposed if d["type"] in _MASTER_WIDTH_SINGLETONS
                for e in gone_with(d)}
    standing = [e for e in master_chain
                if e.get("type") in _MASTER_WIDTH_KINDS and id(e) not in replaced]
    # Two moves on the same device: the later one is what lands. "new" moves each add one.
    def slot(i: int, d: dict):
        return (d["type"], d.get("instance")) if d.get("instance") != "new" else (d["type"], "new", i)

    last = {slot(i, d): i for i, d in proposed if d["type"] in _MASTER_WIDTH_SINGLETONS}
    moves = [(i, d) for i, d in proposed
             if d["type"] not in _MASTER_WIDTH_SINGLETONS or last[slot(i, d)] == i]
    chain_pieces = [p for e in standing for p in _side_pieces(e)]
    move_pieces = [(i, p) for i, d in moves for p in _side_pieces(d)]
    # ...and what the master carries NOW, the device a proposal replaces included: a width
    # opened from 0.5 to 0.55 beside the user's own narrowing is not this proposal narrowing
    # (pre-merge review 2026-10-06; it was refused for any replacement under 1.0).
    now_pieces = [p for e in (stem_chains or {}).get("master", [])
                  if e.get("type") in _MASTER_WIDTH_KINDS for p in _side_pieces(e)]
    edges = sorted({_MASTER_MONO_LOWS_MAX_HZ, _WIDTH_TOP_HZ}
                   | {x for lo, hi, _ in chain_pieces + now_pieces + [p for _, p in move_pieces]
                      for x in (lo, hi) if _MASTER_MONO_LOWS_MAX_HZ < x < _WIDTH_TOP_HZ})
    floor = LIMITS["master_width_min"]
    worst: Optional[tuple[float, int, float, float]] = None      # (scale, index, from, to)
    for lo, hi in zip(edges[:-1], edges[1:]):
        at = (lo * hi) ** 0.5
        scale, narrowed_by = 1.0, None
        for a, b, w in chain_pieces:
            if a <= at < b:
                scale *= w
        for i, (a, b, w) in move_pieces:
            if a <= at < b:
                scale *= w
                if w < 1.0:
                    narrowed_by = i
        was = 1.0
        for a, b, w in now_pieces:
            if a <= at < b:
                was *= w
        if narrowed_by is not None and scale < floor - 1e-9 and scale < was - 1e-9 \
                and (worst is None or scale < worst[0]):
            worst = (scale, narrowed_by, lo, hi)
    if worst is None:
        return []
    scale, index, lo, hi = worst
    span = (f"{lo:.0f} Hz - {hi / 1000:.0f} kHz" if hi >= 1000 else f"{lo:.0f} - {hi:.0f} Hz")
    return [Violation(
        action_index=index, rule="master_width_min",
        message=(f"the master's width devices are in series and would compose to {scale:.2f} "
                 f"of its side at {span}, below the {floor:.2f} floor - the Imager's width and the "
                 "Utility's multiply. Narrow with ONE of them, or less with each; "
                 "monoing the deliverable is the user's call"),
        limit=floor, value=round(scale, 3))]


_RESONANT_SHAPES = ("hpf", "lpf", "notch", "bandpass")
# The shelf family resonates too: RBJ shelves use q as a genuine resonance term,
# so past ~0.707 the response overshoots the declared plateau — a +6 dB low
# shelf at q=8 measures +18.3 dB, and a −6 dB CUT at q=8 measures +12.3 dB of
# boost (sign-inverted). The nominal-gain rules can't see any of that.
_SHELF_SHAPES = ("low_shelf", "high_shelf", "tilt_shelf")


def _eq_band_sections(shape: str, freq_hz, q, gain_db, slope_db_oct,
                      sample_rate: float = 48000.0) -> list:
    """One EQ band's biquad sections, built exactly as `dsp._eq_band` renders it (`eq_law`, the
    EQ's law since 2026-10-27). Shared by the single-band resonance rail and the composed rail."""
    from .eq_law import band_sections
    return band_sections(shape, sample_rate, float(freq_hz), float(q), float(gain_db or 0.0),
                         int(slope_db_oct or 6))


def _eq_band_peak_db(band, sample_rate: float = 48000.0) -> float:
    """Peak magnitude (dB) of ONE EQ band's real filter cascade.

    Built from the same coefficient makers `dsp._eq_band` renders with
    (`reference_monitor.biquad_coeffs` / `slope_cascade_coeffs`), so this can
    never drift from the audible response the way a hand-picked Q bound would.
    48 kHz is a stand-in for the project rate — a resonant peak well below
    Nyquist is rate-invariant to well under 0.1 dB, and the guardrail layer
    has no session to ask.
    """
    import numpy as np
    from scipy.signal import freqz

    sections = _eq_band_sections(band.shape, band.freq_hz, band.q, band.gain_db,
                                 getattr(band, "slope_db_oct", 6), sample_rate)

    w = np.logspace(np.log10(2.0 * np.pi * 10.0 / sample_rate),
                    np.log10(np.pi * 0.999), 2048)
    h = np.ones_like(w, dtype=complex)
    for b, a in sections:
        h = h * freqz(b, a, worN=w)[1]
    return float(20.0 * np.log10(float(np.max(np.abs(h))) + 1e-12))


#: The static-EQ family whose moves COMPOSE on one target (all three append in session._mutate).
#: dynamic_eq is not in it: its gain moves with the signal and has its own bound.
EQ_FAMILY = ("eq", "tilt", "match_eq")
# What the COMPOSED EQ rail charges a curve for: the EQ family, and the overdrive's TONE - a
# first-order high shelf around 3 kHz that lifts the top by up to +6 dB (2026-10-17, the overdrive
# audit: it was charged nowhere, so drive 0 / tone 1 was a +6 dB treble boost outside the rail).
EQ_CURVE_TYPES = EQ_FAMILY + ("overdrive", "saturation")
#: The curve types that REPLACE the target's entry of their type (a whole-file singleton, as the
#: gain wallet's session._TRIM_SINGLETONS): the ledger holds the ONE that stands - and the one
#: that stood when the pass began - never every one applied (the pre-merge review of 2026-10-20:
#: the agent re-proposing its own tone-1 overdrive was refused at +12 dB, the first one counted
#: as if it were still there beside the second).
EQ_CURVE_SINGLETONS = ("overdrive", "saturation")
_EQ_GRID_N = 4096         # 20 Hz..20 kHz, ~411 points/octave: a q=40 bell's peak reads within 0.02 dB
                          # (512 read a +6 dB q=40 bell as +5.21 - the agent's q reaches 40 since the EQ law)
_EQ_GRID_RATE = 48000.0   # the rate with no project to ask (a caller with no session)
#: The rate the rails measure at: the project's, when validate_proposal is told it (the EQ law's
#: response near Nyquist moves with the rate - an 18 kHz shelf at q 8 read 5.69 dB at 48 kHz and
#: played 6.58 at 44.1, the pre-merge review of 2026-10-28).
import contextvars as _contextvars
_RAIL_RATE: "_contextvars.ContextVar[float]" = _contextvars.ContextVar("rail_rate", default=_EQ_GRID_RATE)


def _rail_rate() -> float:
    return float(_RAIL_RATE.get())
_EQ_TOL_DB = 0.1          # dB are 0.1-quantized in this engine, and the EQ law's matched sections sit within
                          # ~0.05 dB of their knob (eq_law, 2026-10-27): admit the limit itself
#: Where a proposal pushes less than this, the point is a bell's SKIRT, not the move (audit
#: follow-ups 2026-09-11, item 2): a q=0.707 +6 dB bell at 8 kHz puts 0.16 dB on 1 kHz, and a
#: 1 kHz band already at the +12 session rail refused the whole move for a tail nobody hears.
#: A skirt point is held to the rail + _EQ_SKIRT_SLACK_DB instead, so stacked small pushes still
#: stop — one decibel past the rail at most, finer than any move the rail exists to stop.
_EQ_PUSH_MATERIAL_DB = 0.5
_EQ_SKIRT_SLACK_DB = 1.0


def _eq_grid():
    import numpy as np
    f = np.logspace(np.log10(20.0), np.log10(20000.0), _EQ_GRID_N)
    return f, 2.0 * np.pi * f / _rail_rate()


def on_eq_grid(curve):
    """A stored curve on the rail's grid: one kept on a grid of another size (session.json's
    `trimmed_eq_curves_db`, written before the grid became 4096 points) is read across in log
    frequency, the grid's own spacing - never added point for point to a grid it does not match."""
    import numpy as np
    c = np.asarray(curve, dtype=np.float64)
    if c.shape == (_EQ_GRID_N,):
        return c
    if c.ndim != 1 or c.size < 2:
        return np.zeros(_EQ_GRID_N)
    old = np.linspace(0.0, 1.0, c.size)
    new = np.linspace(0.0, 1.0, _EQ_GRID_N)
    return np.interp(new, old, c)


def eq_family_curves_db(action):
    """(boost, cut) contribution of ONE static-EQ move on the composed grid, or None for any
    other action. Signed curves add in dB because magnitudes multiply. The cut shapes (hpf / lpf
    / notch / bandpass) feed only their resonant OVERSHOOT to the boost curve: their
    attenuation is their purpose (the HPF ceiling bounds where it lands), and a stopband would
    otherwise read as -100 dB of "cut". Channel scope (mid/side/left/right) is counted as
    full-signal. A TIME-RANGED move is counted whole-file but ONE-SIDED — its lift feeds only
    the boost curve and its dip only the cut curve — so it is conservative and never credit:
    as a signed whole-file curve, a ranged cut in one section paid for a boost in another
    (full audit 2026-10-02, D9·R4: +119.9 dB at 440 Hz in ten accepted passes)."""
    import json

    import numpy as np
    d = action if isinstance(action, dict) else action.model_dump()
    if d.get("type") not in EQ_CURVE_TYPES:
        return None
    key = json.dumps({k: d.get(k) for k in ("type", "band", "tilt_db", "pivot_hz",
                                            "band_gains_db", "tone", "character", "curve")}
                     | {k: {"curve": d[k].get("curve")} for k in ("low", "mid", "high")
                        if isinstance(d.get(k), dict)}, sort_keys=True, default=str)
    # (`low` / `mid` / `high` above are the bands' CURVES only: the key is what the curve reads)
    boost, cut = _curves_cached(f"{_rail_rate():.1f}|{key}")
    # `spent_only` (reader D9·R5): the session ledger's mark on a move whose chain entry the
    # user has since deleted, zeroed or switched off - what it lifted or cut is still spent,
    # and nothing of it is credit, exactly as for a ranged move.
    if d.get("time_range_s") is not None or d.get("spent_only"):
        return np.maximum(boost, 0.0), np.minimum(cut, 0.0)
    return boost, cut


def _curves_cached(key: str):
    hit = _CURVE_CACHE.get(key)
    if hit is None:
        hit = _compute_curves(key)
        if len(_CURVE_CACHE) > 4096:
            _CURVE_CACHE.clear()
        _CURVE_CACHE[key] = hit
    return hit


_CURVE_CACHE: dict = {}


def _compute_curves(key: str):
    import json

    import numpy as np
    from scipy.signal import freqz

    d = json.loads(key.partition("|")[2])
    _, w = _eq_grid()

    def db(sections):
        h = np.ones_like(w, dtype=complex)
        for b, a in sections:
            h = h * freqz(b, a, worN=w)[1]
        return 20.0 * np.log10(np.abs(h) + 1e-12)

    if d["type"] == "eq":
        band = d["band"]
        resp = db(_eq_band_sections(band["shape"], band["freq_hz"], band["q"],
                                    band.get("gain_db", 0.0), band.get("slope_db_oct", 6),
                                    _rail_rate()))
        if band["shape"] in _RESONANT_SHAPES:
            return np.maximum(resp, 0.0), np.zeros_like(resp)
        return resp, resp
    if d["type"] == "tilt":
        from .dsp import TILT_Q   # exactly dsp._tilt: low shelf −tilt + high shelf +tilt
        from .reference_monitor import biquad_coeffs
        tilt, pivot = float(d.get("tilt_db") or 0.0), float(d.get("pivot_hz") or 1000.0)
        resp = db([biquad_coeffs("low_shelf", _rail_rate(), pivot, TILT_Q, -tilt),
                   biquad_coeffs("high_shelf", _rail_rate(), pivot, TILT_Q, tilt)])
        return resp, resp
    if d["type"] == "overdrive":
        # the tone tilt, exactly dsp._overdrive: lp + 2 tone (x - lp) around a 1st-order low-pass at
        # OD_TONE_HZ. Its LIFT only (tone > 0.5) is charged; a darker tone is never credit.
        from .dsp import OD_TONE_HZ
        from .reference_monitor import biquad_coeffs
        tone = min(max(float(d.get("tone", 0.5) if d.get("tone") is not None else 0.5), 0.0), 1.0)
        b, a = biquad_coeffs("lpf", _rail_rate(), OD_TONE_HZ, 0.707, 0.0)
        lp = freqz(b, a, worN=w)[1]
        resp = 20.0 * np.log10(np.abs(lp + 2.0 * tone * (1.0 - lp)) + 1e-12)
        # ...and an amp character's own voicing (b5549bad): its small-signal response with its
        # trim is up to +12.5 dB at 6 kHz - a treble boost no rail saw (pre-merge review 2026-10-26).
        from . import od_characters as odc
        ch = odc.character_index(d.get("character", 0))
        if ch > 0:
            f, _ = _eq_grid()
            secs = odc.pre_sections(ch, _rail_rate()) + odc.post_sections(ch, _rail_rate())
            amp = odc._cascade_response(secs, _rail_rate(), f) * odc._taps_response(ch, f)
            resp = resp + 20.0 * np.log10(np.abs(amp) + 1e-12) + odc.TRIMS_DB[ch]
        return np.maximum(resp, 0.0), np.zeros_like(resp)
    if d["type"] == "saturation":
        # a TAPE / TUBE character's fixed tone (sat_characters.tone_sections) - +1.7..+2.3 dB near
        # 50 Hz that the level law's charge leaves out. A multiband form takes the most any of its
        # bands' characters lifts. Its LIFT only, as the overdrive's tone.
        from . import od_characters as odc
        from . import sat_characters as sc
        f, _ = _eq_grid()
        bands = [d[k] for k in ("low", "mid", "high") if isinstance(d.get(k), dict)]
        flat = d.get("curve", 0)
        curves = [flat if b.get("curve") is None else b["curve"] for b in bands] or [flat]
        resp = np.zeros(len(f))
        for c in curves:
            secs = sc.tone_sections(c, _rail_rate())
            if secs:
                resp = np.maximum(resp, 20.0 * np.log10(np.abs(odc._cascade_response(
                    secs, _rail_rate(), f)) + 1e-12))
        return np.maximum(resp, 0.0), np.zeros_like(resp)
    from .dsp import match_eq_fir   # the rendered FIR itself, so this cannot drift from it
    fir = match_eq_fir(d.get("band_gains_db") or {}, int(_rail_rate()))
    resp = 20.0 * np.log10(np.abs(freqz(fir, [1.0], worN=w)[1]) + 1e-12)
    return resp, resp


def _ledger_curves(ledger: dict, target: str, replaced=()):
    """((pass boost, pass cut), (session boost, session cut)) the ledger already holds on one
    target — the session scope includes the trimmed records' folded carry.

    A singleton (`EQ_CURVE_SINGLETONS`) is the entry that STANDS in the session, and in the pass
    what it moved since the pass began (`standing` - `prior`, the gain wallet's delta). `replaced`
    names the singleton slots (`instance_slot`) the caller's proposal brings its own entry of: what stands leaves
    the session, and the pass is charged against what stood when it began."""
    import numpy as np
    zero = np.zeros(_EQ_GRID_N)

    def curves_of(d):
        got = eq_family_curves_db(d) if d else None
        return got if got is not None else (zero, zero)

    def summed(scope: str):
        b, c = np.zeros(_EQ_GRID_N), np.zeros(_EQ_GRID_N)
        for d in (ledger.get(scope) or {}).get(target, []):
            curves = eq_family_curves_db(d)
            if curves is not None:
                b, c = b + curves[0], c + curves[1]
        return b, c

    pb, pc = summed("pass")
    sb, sc = summed("session")
    carry = (ledger.get("trimmed") or {}).get(target)
    if carry:
        sb, sc = sb + on_eq_grid(carry["boost"]), sc + on_eq_grid(carry["cut"])
    for kind, slot in ((ledger.get("singletons") or {}).get(target) or {}).items():
        (s_b, s_c), (p_b, p_c) = curves_of(slot.get("standing")), curves_of(slot.get("prior"))
        if slot_replaced(kind, replaced):   # `kind` is the instance's slot (Phase 65)
            pb, pc = pb - p_b, pc - p_c
        else:
            sb, sc = sb + s_b, sc + s_c
            pb, pc = pb + s_b - p_b, pc + s_c - p_c
    return (pb, pc), (sb, sc)


def eq_headroom_db(ledger: dict, target: str,
                   paths: Optional[list[list[str]]] = None) -> dict[str, dict[str, float]]:
    """What the composed rail still allows on one target, per analysis band: the most a NEW
    move may lift (`boost`, >= 0) or cut (`cut`, <= 0) anywhere in that band, this pass and this
    session together. The worst point in the band binds, rounded TOWARD zero, so a move sized
    to it clears the rail. The agent reads it in its context; the rule agent sizes by it.
    `paths` = the signal paths `target` is on (D9·R7): a move on it is also held to what each
    whole path has left, so the room is the least of them."""
    import math

    import numpy as np
    f, _ = _eq_grid()
    room_boost = np.full(_EQ_GRID_N, np.inf)
    room_cut = np.full(_EQ_GRID_N, -np.inf)
    for scope in [[target]] + [p for p in (paths or []) if len(p) > 1]:
        pb = sb = pc = sc = np.zeros(_EQ_GRID_N)
        for node in scope:
            (b1, c1), (b2, c2) = _ledger_curves(ledger, node)
            pb, pc, sb, sc = pb + b1, pc + c1, sb + b2, sc + c2
        room_boost = np.minimum(room_boost, np.minimum(LIMITS["eq_boost_pass_db"] - pb,
                                                       LIMITS["eq_boost_session_db"] - sb))
        room_cut = np.maximum(room_cut, np.maximum(LIMITS["eq_cut_pass_db"] - pc,
                                                   LIMITS["eq_cut_session_db"] - sc))
    out: dict[str, dict[str, float]] = {}
    for label, lo, hi in zip(BAND_LABELS, BAND_EDGES_HZ[:-1], BAND_EDGES_HZ[1:]):
        m = (f >= lo) & (f <= hi)
        out[label] = {"boost": max(0.0, math.floor(float(room_boost[m].min()) * 10) / 10),
                      "cut": min(0.0, math.ceil(float(room_cut[m].max()) * 10) / 10)}
    return out


#== A group is on its members' signal path (reader D9·R7) ==
# Every level and EQ budget is keyed by target, and a track's audio goes through its group: a
# +6 dB move on the group lands on every member, on top of the member's own +6. One proposal -
# `group solo = [vocal]`, gain and EQ on both - put the vocal +24 dB up at 440 Hz with every
# wallet inside its limit, and a two-member group is the same alias at half strength. So the
# budgets are also held along the PATH: a track, the group it is summed into, a group that
# group plays into (and a track another track's Audio To feeds). `signal_parents` maps each
# stem and bus to the next target on its way to the master (None = the master, which is not
# on the path: it feeds the limiter, whose ceiling is the bound there).
#
# A path is judged only where the proposal PUSHES it - what it would be against what it is -
# so a corrective move, or any move on a path the user's own routing put past a rail, is
# never refused for an excess that was already there.
#
# "What it is" COUNTS THE ENTRY A PROPOSAL REPLACES (pre-merge review 2026-10-06): the wallets
# `validate_proposal` walks have that singleton taken out (it is about to be overwritten), and
# judged against those any positive replacement read as a push - a Utility turned from +6 to
# +3 on a path past the rail was refused as "+9 with the group". The session hands the
# standing totals for this comparison (`standing_gain_*`); the wallets themselves are unchanged.

def _signal_path(parents: dict, node: str) -> list[str]:
    out: list[str] = []
    seen: set = set()
    at: Optional[str] = node
    while at is not None and at not in seen:
        out.append(at)
        seen.add(at)
        at = parents.get(at)
    return out


def _parents_after(req: ProposalRequest, parents: dict) -> dict:
    """The paths as this proposal's own group / route actions would leave them."""
    after = dict(parents)
    for action in req.actions:
        if action.type == "group":
            after.setdefault(action.bus_id, None)
            for member in action.members:
                after[member] = action.bus_id
        elif action.type == "route":
            after[action.stem_id] = action.bus_id
    return after


def _path_names(path: list[str]) -> str:
    return " -> ".join(f"'{t}'" for t in path[1:])


def _path_gain_violations(req: ProposalRequest, parents: dict, wallets) -> list[Violation]:
    """`wallets`: (rule, what, tail, net before, ranged before, net after, ranged after) for
    the pass and the session - the gain wallets as `validate_proposal` walked them."""
    after_parents = _parents_after(req, parents)
    out: list[Violation] = []
    for rule, what, tail, net0, ranged0, net1, ranged1 in wallets:
        limit = LIMITS[rule]
        worst: Optional[tuple[float, float, list[str]]] = None       # (excess, value, path)

        def level(path, net, ranged, side):
            return sum(net.get(t, 0.0) + ranged.get(t, (0.0, 0.0))[side] for t in path)

        for node in after_parents:
            path = _signal_path(after_parents, node)
            if len(path) < 2:
                continue
            was = _signal_path(parents, node)
            for side, sign in ((0, 1.0), (1, -1.0)):
                now = sign * level(path, net1, ranged1, side)
                before = sign * level(was, net0, ranged0, side)
                if now > limit + 1e-9 and now > before + 1e-9 \
                        and (worst is None or now - limit > worst[0]):
                    worst = (now - limit, sign * now, path)
        if worst is None:
            continue
        _, value, path = worst
        index = -1
        for i, action in enumerate(req.actions):
            if (action.type == "group" and path[0] in action.members) \
                    or (action.type == "route" and action.stem_id == path[0]) \
                    or (getattr(action, "stem_id", None) in path and level_charge_db(action)):
                index = i
        out.append(Violation(
            action_index=index, rule=rule,
            message=(f"cumulative gain on '{path[0]}' {what} would be {value:+.1f} dB counting "
                     f"the group it plays through ({_path_names(path)}): a group's level moves "
                     f"every track in it, so what is spent on the track and on the group add "
                     f"up{tail}"),
            limit=limit, value=round(value, 1)))
    return out


def _composed_eq_violations(req: ProposalRequest, ledger: dict,
                            signal_parents: Optional[dict] = None) -> list[Violation]:
    """The composed static-EQ rail (design call 2026-09-11). Per target this proposal touches:
    this pass's applied moves + the proposal may not compose past +6 / -12 dB anywhere, and the
    session's applied (not undone) moves + trimmed carry + the proposal may not pass +12 / -24.
    Checked only where the PROPOSAL pushes that way, so a corrective move on a target already
    past a rail (a user edit, a changed limit) is never refused for someone else's excess."""
    import numpy as np
    f, _ = _eq_grid()
    new: dict[str, list] = {}
    replaces: dict[str, set] = {}     # the singleton types each target's proposal brings anew
    for i, action in enumerate(req.actions):
        curves = eq_family_curves_db(action)
        if curves is None:
            continue
        slot = new.setdefault(action.stem_id, [np.zeros(_EQ_GRID_N), np.zeros(_EQ_GRID_N), i])
        slot[0] = slot[0] + curves[0]
        slot[1] = slot[1] + curves[1]
        if action.type in EQ_CURVE_SINGLETONS and getattr(action, "time_range_s", None) is None:
            replaces.setdefault(action.stem_id, set()).update(replaced_slots(
                (ledger.get("chains") or {}).get(action.stem_id, ()), action))

    zero = np.zeros(_EQ_GRID_N)
    held: dict[tuple, tuple] = {}

    def ledger_of(target: str, after: bool = False):
        """What the ledger holds on `target` - before the proposal, or (`after`) without the
        singletons the proposal replaces, ready for the proposal's own curves to be added."""
        key = (target, after)
        if key not in held:
            gone = replaces.get(target, ()) if after else ()
            (pb, pc), (sb, sc) = _ledger_curves(ledger, target, gone)
            held[key] = (pb, sb, pc, sc)
        return held[key]

    def added(target: str):
        slot = new.get(target)
        return (slot[0], slot[0], slot[1], slot[1]) if slot else (zero, zero, zero, zero)

    def summed(parts):
        parts = list(parts)
        return tuple(sum((p[k] for p in parts), zero) for k in range(4))

    rails = (("eq_boost_pass_db", "this pass", True), ("eq_boost_session_db", "this session", True),
             ("eq_cut_pass_db", "this pass", False), ("eq_cut_session_db", "this session", False))

    def judge(before, after, k: int):
        """(excess, worst total, grid index) on rail k, where the proposal pushes; None = in budget."""
        rule, _, is_boost = rails[k]
        limit = LIMITS[rule]
        sign = 1.0 if is_boost else -1.0   # fold cuts onto the boost arithmetic
        total = after[k]
        push = sign * (after[k] - before[k])   # >= 0 where the proposal pushes this way
        mask = push > _EQ_TOL_DB
        if not mask.any():
            return None
        # The exact rail where the proposal pushes materially; rail + slack on its skirt.
        allowed = sign * limit + np.where(push >= _EQ_PUSH_MATERIAL_DB,
                                          _EQ_TOL_DB, _EQ_SKIRT_SLACK_DB)
        excess = np.where(mask, sign * total - allowed, -np.inf)
        j = int(np.argmax(excess))
        return (float(excess[j]), float(total[j]), j) if excess[j] > 0.0 else None

    def where(j: int) -> str:
        hz = f[j]
        return f"{hz:.0f} Hz" if hz < 1000 else f"{hz / 1000:.1f} kHz"

    out: list[Violation] = []
    refused: set[tuple[str, str]] = set()          # (rule, target) already said
    for target, (_, _, idx) in new.items():
        before = ledger_of(target)
        after = summed([ledger_of(target, True), added(target)])
        for k, (rule, scope, _) in enumerate(rails):
            hit = judge(before, after, k)
            if hit is not None:
                refused.add((rule, target))
                out.append(Violation(
                    action_index=idx, rule=rule,
                    message=(f"EQ on '{target}' would compose to {hit[1]:+.1f} dB at {where(hit[2])} "
                             f"{scope} (limit {LIMITS[rule]:+.0f} dB) — stacked EQ moves add up; "
                             "adjust the band already there instead of adding another"),
                    limit=LIMITS[rule], value=round(hit[1], 1)))

    # ...and along the signal path (D9·R7): a track's EQ and its group's compose on the track.
    if signal_parents is not None:
        after_parents = _parents_after(req, signal_parents)
        moved = {a.stem_id for a in req.actions if a.type == "route"} \
            | {m for a in req.actions if a.type == "group" for m in a.members}
        worst: dict[int, tuple] = {}
        for node in after_parents:
            path = _signal_path(after_parents, node)
            if len(path) < 2 or not (node in moved or any(t in new for t in path)):
                continue
            before = summed(ledger_of(t) for t in _signal_path(signal_parents, node))
            after = summed([summed(ledger_of(t, True) for t in path), summed(added(t) for t in path)])
            for k, (rule, _, _) in enumerate(rails):
                if any((rule, t) in refused for t in path):
                    continue
                hit = judge(before, after, k)
                if hit is not None and (k not in worst or hit[0] > worst[k][0][0]):
                    worst[k] = (hit, path)
        for k, (hit, path) in sorted(worst.items()):
            rule, scope, _ = rails[k]
            index = min((new[t][2] for t in path if t in new), default=-1)
            if index < 0:
                index = next((i for i, a in enumerate(req.actions)
                              if (a.type == "route" and a.stem_id == path[0])
                              or (a.type == "group" and path[0] in a.members)), -1)
            out.append(Violation(
                action_index=index, rule=rule,
                message=(f"EQ on '{path[0]}' would compose to {hit[1]:+.1f} dB at {where(hit[2])} "
                         f"{scope} (limit {LIMITS[rule]:+.0f} dB) counting the group it plays "
                         f"through ({_path_names(path)}): a group's EQ shapes every track in it, "
                         "so the track's moves and the group's add up"),
                limit=LIMITS[rule], value=round(hit[1], 1)))
    return out


# A channel's fader - a stem's, a bus's, the master's: the range of the control in the DAW, and
# so the range of every door that sets that level (James, 2026-10-05, reader E1.R4: "Ableton only
# goes up to 6"). The command, the co-edit table's `gain` row and the brain's own setters took
# +12 against a fader that stops at +6, and a run's gain move added to the stored level with no
# top at all: a channel above +6 showed 6.0 and lost the excess at the first touch. Level past
# the fader's top is a Utility's gain. `Session.USER_EDIT_FIELDS["gain_db"]` IS this tuple.
FADER_DB_RANGE = (-60.0, 6.0)


@contextlib.contextmanager
def rail_rate(sample_rate):
    """Measure the EQ rails at `sample_rate` (the project's) inside the block; None = the stand-in."""
    if sample_rate is None:
        yield
        return
    tok = _RAIL_RATE.set(float(sample_rate))
    try:
        yield
    finally:
        _RAIL_RATE.reset(tok)


def validate_proposal(*args, sample_rate: Optional[float] = None, **kwargs) -> list[Violation]:
    """`_validate_proposal` with its EQ rails measured at `sample_rate` (the project's rate)."""
    with rail_rate(sample_rate):
        return _validate_proposal(*args, **kwargs)


def _validate_proposal(
    req: ProposalRequest,
    cumulative_gain_this_pass_db: dict[str, float],
    accepted_proposals_this_pass: int = 0,
    stem_tags: Optional[dict[str, list[str]]] = None,
    cumulative_gain_session_db: Optional[dict[str, float]] = None,
    stem_chains: Optional[dict[str, list[dict]]] = None,
    eq_ledger: Optional[dict] = None,
    ranged_gain_this_pass_db: Optional[dict[str, tuple[float, float]]] = None,
    ranged_gain_session_db: Optional[dict[str, tuple[float, float]]] = None,
    ranged_spans_this_pass: Optional[dict[str, list]] = None,
    ranged_spans_session: Optional[dict[str, list]] = None,
    fader_db: Optional[dict[str, float]] = None,
    signal_parents: Optional[dict[str, Optional[str]]] = None,
    standing_gain_this_pass_db: Optional[dict[str, float]] = None,
    standing_gain_session_db: Optional[dict[str, float]] = None,
) -> list[Violation]:
    """Return violations; empty list means the proposal is acceptable.

    `cumulative_gain_this_pass_db` maps stem_id -> gain already applied this pass,
    supplied by the session layer, so repeated small moves can't bypass the clamp.
    `accepted_proposals_this_pass` counts previously accepted (non-rejected)
    proposals in the current pass. `stem_tags` maps stem_id -> tags (the HPF
    frequency limit is stricter on bass-tagged stems).
    `cumulative_gain_session_db` maps stem_id -> gain applied (and not undone)
    across the whole session — the Phase 10 budget that pass resets can't refill.
    The session layer folds the chains' current comp/mb makeup into it (2026-06
    review): makeup is level the agent already added, drawn from the same budget.
    `stem_chains` maps stem_id (and "master") -> the raw chain entries — the
    comp-stacking rule needs to see what is already applied.
    `eq_ledger` ({"pass"|"session": {target: [applied eq-family action dicts]},
    "trimmed": {target: {"boost", "cut"}}}) arms the composed-EQ rail; None skips it.
    `ranged_gain_this_pass_db` / `ranged_gain_session_db` map stem_id -> (up, down): the
    TIME-RANGED level already applied, one-sided (up >= 0 the sum of ranged lifts, down <= 0
    the sum of ranged cuts — see `is_ranged`). The two `cumulative_*` maps hold the whole-file
    moves only. A target is in budget while net + up and net + down both are.
    `fader_db` maps a target (stem, bus, "master") -> where its fader stands. A whole-file gain
    move lands ON the fader, so it may not take one past `FADER_DB_RANGE`'s top; None skips the
    rule (a caller with no session).
    `signal_parents` maps every stem and bus to the next target its signal goes through (a
    group, the track its Audio To feeds; None = the master). It arms the PATH budgets - gain
    and composed EQ along a track and the groups it plays through (see `_signal_path`); None
    skips them.
    """
    stem_tags = stem_tags or {}
    violations: list[Violation] = []

    if accepted_proposals_this_pass >= LIMITS["proposals_per_pass"]:
        violations.append(Violation(
            action_index=-1, rule="proposals_per_pass",
            message="too many proposals this pass",
            limit=LIMITS["proposals_per_pass"], value=accepted_proposals_this_pass + 1,
        ))

    if len(req.actions) > LIMITS["actions_per_proposal"]:
        violations.append(Violation(
            action_index=-1, rule="actions_per_proposal",
            message="too many actions in one proposal",
            limit=LIMITS["actions_per_proposal"], value=len(req.actions),
        ))

    running = dict(cumulative_gain_this_pass_db)
    running_session = dict(cumulative_gain_session_db or {})
    # (not `ranged`: the comp-stacking rule below binds that name, and _charge_gain reads these
    # through its closure — the clash made the wallet a list for any later action)
    # A ranged wallet is a BASE (what has no range any more: the retention carry) plus the
    # peak of the SPANS that still have one (`ranged_peak`). A caller that hands no spans
    # hands sums, and is charged as before.
    ranged_pass_base = {k: tuple(v) for k, v in (ranged_gain_this_pass_db or {}).items()}
    ranged_session_base = {k: tuple(v) for k, v in (ranged_gain_session_db or {}).items()}
    spans_pass_w = {k: list(v) for k, v in (ranged_spans_this_pass or {}).items()}
    spans_session_w = {k: list(v) for k, v in (ranged_spans_session or {}).items()}

    def _ranged_total(base: dict, spans: dict) -> dict:
        out = {}
        for sid in set(base) | set(spans):
            b_up, b_down = base.get(sid, (0.0, 0.0))
            p_up, p_down = ranged_peak(spans.get(sid, ()))
            out[sid] = (b_up + p_up, b_down + p_down)
        return out

    ranged_pass_w = _ranged_total(ranged_pass_base, spans_pass_w)
    ranged_session_w = _ranged_total(ranged_session_base, spans_session_w)
    ranged_pass_before, ranged_session_before = dict(ranged_pass_w), dict(ranged_session_w)
    coverage = _comp_coverage(stem_chains)
    deess_coverage = _comp_coverage(stem_chains, "deess")

    def _charge_gain(i: int, stem_id: str, delta_db: float,
                     note: str = "", in_range: bool = False,
                     span: Optional[tuple[float, float]] = None) -> None:
        """Walk both gain budgets — gain moves and (2026-06 review) comp
        makeup draw from the same wallet. A whole-file move nets (a −3 after a +3 is 0, and it
        is); a time-ranged one (`in_range`) is one-sided, so a cut in one region never pays
        for a lift in another (D9·R4)."""
        for net_w, ranged_w, base_w, spans_w, rule, what, tail in (
                (running, ranged_pass_w, ranged_pass_base, spans_pass_w,
                 "gain_per_stem_per_pass_db", "this pass", ""),
                (running_session, ranged_session_w, ranged_session_base, spans_session_w,
                 "gain_per_stem_per_session_db",
                 "across the session", "; undo earlier gain moves to free budget")):
            net = net_w.get(stem_id, 0.0)
            up, down = ranged_w.get(stem_id, (0.0, 0.0))
            trial = None
            if in_range and span is not None:
                # by what plays at once: this move beside the ranged ones already there
                trial = list(spans_w.get(stem_id, ())) + [(span[0], span[1], delta_db)]
                b_up, b_down = base_w.get(stem_id, (0.0, 0.0))
                p_up, p_down = ranged_peak(trial)
                up, down = b_up + p_up, b_down + p_down
            elif in_range:
                up, down = up + max(delta_db, 0.0), down + min(delta_db, 0.0)
                b_up, b_down = base_w.get(stem_id, (0.0, 0.0))
                trial_base = (b_up + max(delta_db, 0.0), b_down + min(delta_db, 0.0))
            else:
                net += delta_db
            hi, lo = net + up, net + down
            limit = LIMITS[rule]
            if hi > limit or lo < -limit:
                worst = hi if hi > limit else lo
                where = " in a time-ranged region" if (up or down) else ""
                violations.append(Violation(
                    action_index=i, rule=rule,
                    message=(f"cumulative gain on '{stem_id}' {what} would be "
                             f"{worst:+.1f} dB{where}{note}{tail}"),
                    limit=limit, value=worst,
                ))
            elif in_range:
                ranged_w[stem_id] = (up, down)
                if trial is not None:
                    spans_w[stem_id] = trial
                else:
                    base_w[stem_id] = trial_base
            else:
                net_w[stem_id] = net

    violations += _stacked_levelling(req, stem_chains)
    faders = dict(fader_db or {})
    added_new: dict[str, int] = {}
    for i, action in enumerate(req.actions):
        # A whole-file proposal of a device the user may stack (Phase 65: INSTANCE_KINDS) REPLACES
        # the target's one (session._mutate_one). On a channel that already carries two or more
        # it says which (`instance: N`, the type's N-th entry as `comp#N` counts them) or adds one
        # (`instance: "new"`); without either it cannot say which: refused, with the way to say it.
        kind = getattr(action, "type", None)
        inst = getattr(action, "instance", None)
        if kind in INSTANCE_KINDS:
            chain_now = (stem_chains or {}).get(getattr(action, "stem_id", ""), ())
            of_type = [e for e in chain_now if e.get("type") == kind]
            whole_now = sum(1 for e in of_type if not e.get("time_range_s"))
            if inst is not None and getattr(action, "time_range_s", None) is not None:
                violations.append(Violation(
                    action_index=i, rule="instance_ranged",
                    message=f"a time-ranged {kind} is placed beside the others by its range; "
                            "leave `instance` off a time-ranged move",
                    limit=0, value=1))
            elif isinstance(inst, int) and not 0 <= inst < len(of_type):
                violations.append(Violation(
                    action_index=i, rule="instance_missing",
                    message=f"'{action.stem_id}' has no {kind}#{inst}: it carries {len(of_type)} "
                            f"{kind} device(s), {kind}#0 the first - or give instance \"new\" "
                            "to add one",
                    limit=max(len(of_type) - 1, 0), value=inst))
            elif isinstance(inst, int) and of_type[inst].get("time_range_s"):
                violations.append(Violation(
                    action_index=i, rule="instance_ranged",
                    message=f"{kind}#{inst} on '{action.stem_id}' is time-ranged; a whole-file "
                            f"proposal does not replace it - name a whole-file {kind}, or add "
                            "one with instance \"new\"",
                    limit=0, value=inst))
            elif (isinstance(inst, int) and getattr(action, "stem_id", None) == "master"
                  and kind != "limiter" and _after_last_limiter(chain_now, of_type[inst])):
                # After the master's limiter is the USER'S (James, 2026-10-05). A replace with
                # `instance` keeps the entry where it stands, so naming one past the limiter would
                # put the agent's move behind the ceiling (pre-merge review 2026-10-26).
                violations.append(Violation(
                    action_index=i, rule="post_limiter",
                    message=f"{kind}#{inst} on the master plays after its limiter, which is the "
                            "user's; change a device in front of the limiter, or add one with "
                            "instance \"new\"",
                    limit=0, value=inst))
            elif inst == "new":
                from .session import CHAIN_MAX_ENTRIES
                added_new[action.stem_id] = added_new.get(action.stem_id, 0) + 1
                if len(chain_now) + added_new[action.stem_id] > CHAIN_MAX_ENTRIES:
                    violations.append(Violation(
                        action_index=i, rule="chain_full",
                        message=f"'{action.stem_id}' already holds {len(chain_now)} devices; a "
                                f"chain holds {CHAIN_MAX_ENTRIES} - change one that is there",
                        limit=CHAIN_MAX_ENTRIES, value=len(chain_now) + added_new[action.stem_id]))
            elif inst is None and getattr(action, "time_range_s", None) is None and whole_now >= 2:
                violations.append(Violation(
                    action_index=i, rule="instance_which",
                    message=f"'{action.stem_id}' carries {whole_now} {kind} devices; say which one "
                            f"this proposal changes with \"instance\": N ({kind}#0 is the first, "
                            f"{kind}#1 the next), or add one with \"instance\": \"new\" - or change "
                            f"one by hand with device.set {kind}#N",
                    limit=1, value=whole_now))
        if isinstance(action, GainAction):
            _charge_gain(i, action.stem_id, action.delta_db, in_range=is_ranged(action), span=_span_of(action))
            # The fader's own top. A time-ranged move is a ride over the fader, not its level;
            # a cut is always allowed (a level already past the top may come down).
            if fader_db is not None and not is_ranged(action) and action.stem_id in faders:
                was = faders[action.stem_id]
                faders[action.stem_id] = now = was + action.delta_db
                top = FADER_DB_RANGE[1]
                if action.delta_db > 0 and now > top + 1e-9:
                    violations.append(Violation(
                        action_index=i, rule="fader_max_db",
                        message=(f"the fader on '{action.stem_id}' stops at {top:+.0f} dB: it is at "
                                 f"{was:+.1f} and this move would put it at {now:+.1f}. For more "
                                 "level than the fader has, add gain with a utility, or bring "
                                 "the other tracks down instead"),
                        limit=top, value=round(now, 1)))

        elif isinstance(action, UtilityAction):
            # Device freedom: the Utility's gain trim is a level change — charge it to the
            # per-stem gain budget (±6 dB/pass) just like comp makeup. The user's
            # hand-placed Utility isn't guardrailed.
            if action.gain_db:
                _charge_gain(i, action.stem_id, action.gain_db,
                             note=" (incl. utility gain)")
            if action.stem_id == "master":
                # 2026-09-08 (07·F3). This branch used to end at `balance`, on the reading
                # that "width/mono/phase are shaping (bounded by the model)". Two of those
                # three destroy the deliverable inside their legal range.
                #
                # POLARITY: resolved exactly as dsp._utility resolves it (an unset per-side
                # flag inherits the legacy whole-device `invert`). Both sides flipped is a
                # whole-mix polarity inversion — mono-sum preserving, harmless, legal. ONE
                # side flipped takes 12.39 dB out of the mono sum of a finished mix, and the
                # agent's own prompt steers it here ("invert_left/right fix a single
                # out-of-phase channel") — true of a stem, never of a master, where there is
                # no single bad channel to repair.
                flip_l = action.invert if action.invert_left is None else action.invert_left
                flip_r = action.invert if action.invert_right is None else action.invert_right
                if flip_l != flip_r:
                    violations.append(Violation(
                        action_index=i, rule="master_polarity_split",
                        message=("flipping ONE channel's polarity on the master guts its "
                                 "mono sum (measured −12.4 dB) — a single out-of-phase "
                                 "channel is a STEM repair; on the master, invert both "
                                 "(`invert`) or neither"),
                        limit=0.0, value=1.0))
                # WIDTH/MONO: `mono` IS width 0 downstream (dsp._utility), one check for both.
                violations += _master_narrowing(
                    i, "utility mono (width 0.00)" if action.mono else "utility width",
                    0.0 if action.mono else action.width, up_to_hz=float("inf"))
                # BASS MONO folds the lows only up to the exemption's edge (the Utility audit,
                # 2026-10-21): since it is an LR8 split (James's pick) the side below the knob is
                # gone - at 500 Hz the side at 300 Hz is 0.02 - so on the master a knob above
                # _MASTER_MONO_LOWS_MAX_HZ is the image, not the lows. (The old one-pole left 0.51
                # there, and the 0.5 floor held only by that.)
                if action.bass_mono:
                    violations += _master_narrowing(
                        i, f"utility bass_mono at {action.bass_mono_hz:.0f} Hz (width 0.00 below it)",
                        0.0, up_to_hz=action.bass_mono_hz)
            # `balance` was unbounded: balance=+1.0 on the master is a legal action that
            # collapses the whole mix into one channel. An L/R skew of the finished mix is
            # never an automatic move, so the agent gets none of it on the master and only
            # a nudge elsewhere (the user's hand-placed Utility stays unguardrailed).
            if action.balance:
                bal_limit = (0.0 if action.stem_id == "master"
                             else LIMITS["utility_balance_max"])
                if abs(action.balance) > bal_limit:
                    violations.append(Violation(
                        action_index=i, rule="utility_balance_max",
                        message=(f"utility balance {action.balance:+.2f} exceeds the limit "
                                 f"on {action.stem_id!r}"),
                        limit=bal_limit, value=action.balance,
                    ))

        # The other three ways to narrow the master (2026-09-08, 07·F3). Same floor, same
        # lows exemption, applied to whatever band each device's scale actually covers —
        # so no device is the soft way round the rule the Utility now enforces.
        elif isinstance(action, WidthAction):
            if action.stem_id == "master":
                violations += _master_narrowing(i, "master width", action.width,
                                                up_to_hz=float("inf"))

        # Pydantic field constraints already reject out-of-range raw values;
        # the EQ/comp checks below are defense-in-depth (the doc'd rule table
        # must hold even if a schema bound drifts), and the HPF rule is
        # context-dependent policy that only this layer can enforce.
        elif isinstance(action, EqAction):
            if (action.band.gain_db == 0.0
                    and action.band.shape not in _RESONANT_SHAPES
                    and action.output_db == 0.0):
                # A 0.0 dB peak/shelf is unity gain — a no-op that can never
                # move its target metric, so the planner re-proposed it every
                # pass forever (2026-08-09). Only the gain-driven shapes: the
                # cut shapes (hpf/lpf/notch/bandpass) ignore gain_db, and a
                # band carrying the device's output trim still does something.
                violations.append(Violation(
                    action_index=i, rule="eq_gain_nonzero",
                    message=(f"EQ at {action.band.freq_hz:.0f} Hz has 0.0 dB "
                             "gain — a no-op; propose a real move or none"),
                    limit=0.0, value=action.band.gain_db,
                ))
            if action.band.gain_db > LIMITS["eq_boost_max_db"]:
                violations.append(Violation(
                    action_index=i, rule="eq_boost_max_db",
                    message=f"EQ boost {action.band.gain_db:+.1f} dB exceeds limit",
                    limit=LIMITS["eq_boost_max_db"], value=action.band.gain_db,
                ))
            if action.band.gain_db < LIMITS["eq_cut_min_db"]:
                violations.append(Violation(
                    action_index=i, rule="eq_cut_min_db",
                    message=f"EQ cut {action.band.gain_db:+.1f} dB exceeds limit",
                    limit=LIMITS["eq_cut_min_db"], value=action.band.gain_db,
                ))
            if action.band.shape in _RESONANT_SHAPES + _SHELF_SHAPES:
                # gain_db is IGNORED for the cut shapes, and for shelves q is a
                # real resonance term the nominal gain doesn't describe; bound
                # what the band actually does to the signal instead.
                peak_db = _eq_band_peak_db(action.band, _rail_rate())
                # +0.05: a max-legal shelf's PLATEAU sits exactly on the limit
                # (eq_boost_max_db == eq_resonance_max_db == 6.0), so admit the
                # plateau itself and reject only a resonant lip above it — dB
                # are 0.1-quantized everywhere in this engine's outputs.
                if peak_db > LIMITS["eq_resonance_max_db"] + 0.05:
                    violations.append(Violation(
                        action_index=i, rule="eq_resonance_max_db",
                        message=(f"{action.band.shape} at {action.band.freq_hz:.0f} Hz "
                                 f"q={action.band.q:g} resonates {peak_db:+.1f} dB, "
                                 f"over the limit"),
                        limit=LIMITS["eq_resonance_max_db"], value=round(peak_db, 1),
                    ))
            if action.band.shape == "hpf":
                bass_tagged = "bass" in stem_tags.get(action.stem_id, [])
                own_tags = stem_tags.get(action.stem_id, [])
                hpf_limit = (LIMITS["hpf_max_hz_bass_tagged"] if bass_tagged
                             else LIMITS["hpf_max_hz_cymbal_tagged"] if "cymbal" in own_tags
                             else LIMITS["hpf_max_hz_vocal_tagged"] if "vocal" in own_tags
                             else LIMITS["hpf_max_hz_default"])
                if action.band.freq_hz > hpf_limit:
                    violations.append(Violation(
                        action_index=i, rule="hpf_max_hz",
                        message=(f"HPF at {action.band.freq_hz:.0f} Hz exceeds "
                                 f"{hpf_limit:.0f} Hz limit on "
                                 f"'{action.stem_id}'"
                                 + (" (bass-tagged)" if bass_tagged else "")),
                        limit=hpf_limit, value=action.band.freq_hz,
                    ))
            # 2026-07-12 gain staging: a positive EQ output trim adds level — charge
            # it to the same gain wallet as gain moves / comp makeup (a cut is free).
            if action.output_db > 0:
                _charge_gain(i, action.stem_id, action.output_db,
                             note=" (incl. eq output)", in_range=is_ranged(action), span=_span_of(action))

        elif isinstance(action, CompAction):
            if action.ratio > LIMITS["comp_ratio_max"]:
                violations.append(Violation(
                    action_index=i, rule="comp_ratio_max",
                    message=f"compressor ratio {action.ratio:.1f}:1 exceeds limit",
                    limit=LIMITS["comp_ratio_max"], value=action.ratio,
                ))
            # Only the limiter pushes loudness on the master bus (Phase 6):
            # bus-comp makeup would dodge the limiter's ceiling accounting. For a true
            # M/S comp the mid/side blocks carry their own makeup — count the largest.
            eff_makeup = level_charge_db(action)
            if (action.stem_id == "master"
                    and eff_makeup > LIMITS["master_comp_makeup_max_db"]):
                violations.append(Violation(
                    action_index=i, rule="master_comp_makeup_max_db",
                    message="bus compressor makeup must be 0 on the master; "
                            "use the limiter to push loudness",
                    limit=LIMITS["master_comp_makeup_max_db"], value=eff_makeup,
                ))
            elif eff_makeup:
                _charge_gain(i, action.stem_id, eff_makeup,
                             note=" (incl. comp makeup)", in_range=is_ranged(action), span=_span_of(action))
            # One whole-file comp per stem (2026-06 review): a new one
            # *replaces* the old (session._mutate), so re-proposing is how
            # the agent adjusts settings. What gets rejected is stacking: a
            # whole-file comp over existing ranged ones, or range overlap.
            covered = coverage.setdefault(action.stem_id, [])
            new_range = (tuple(action.time_range_s)
                         if action.time_range_s else None)
            if new_range is None:
                ranged = [r for r in covered if r is not None]
                if ranged:
                    violations.append(Violation(
                        action_index=i, rule="comp_stacking",
                        message=f"'{action.stem_id}' already has "
                                f"{len(ranged)} time-ranged compressor(s); "
                                "a whole-file comp would stack on top — "
                                "undo them first or stay time-ranged",
                        limit=LIMITS["whole_file_comps_per_stem"],
                        value=len(ranged) + 1))
                elif getattr(action, "instance", None) == "new":
                    covered.append(None)  # one more whole-file comp beside the others (Phase 65)
                elif getattr(action, "instance", None) is None:
                    covered[:] = [c for c in covered if c is not None] + [None]   # replaces THE one
                # an `instance: N` replaces that one: the count does not move
            else:
                clashes = [r for r in covered
                           if r is None or (r[0] < new_range[1]
                                            and new_range[0] < r[1])]
                if clashes:
                    violations.append(Violation(
                        action_index=i, rule="comp_stacking",
                        message=f"time-ranged comp on '{action.stem_id}' "
                                "overlaps a compressor already on the "
                                "chain; ranges must not overlap (a "
                                "whole-file comp is adjusted by proposing "
                                "its replacement)",
                        limit=LIMITS["whole_file_comps_per_stem"],
                        value=len(clashes) + 1))
                else:
                    covered.append(new_range)
            # …and however they are arranged, the chain can only hold so many. The native engine
            # gives each entry its own slot up to kMaxComps; beyond that it would drop one.
            if len(covered) > LIMITS["comps_per_stem_max"]:
                violations.append(Violation(
                    action_index=i, rule="comps_per_stem_max",
                    message=f"'{action.stem_id}' would carry {len(covered)} compressors; "
                            f"a mix run adds at most {LIMITS['comps_per_stem_max']} per chain — "
                            "widen the ranges or undo one",
                    limit=LIMITS["comps_per_stem_max"], value=len(covered)))

        # Schema bounds are the first line for the Phase 7 processors too;
        # these keep the doc'd rule table true even if a bound drifts.
        elif isinstance(action, SaturationAction):
            # Multiband saturation carries per-band drive/output — cap the deepest drive
            # and charge the largest positive output, same as the single-band path.
            bands = [b for b in (action.low, action.mid, action.high) if b is not None]
            eff_drive = max([action.drive_db] + [b.drive_db for b in bands])
            if eff_drive > LIMITS["saturation_drive_max_db"]:
                violations.append(Violation(
                    action_index=i, rule="saturation_drive_max_db",
                    message=f"saturation drive {eff_drive:+.1f} dB exceeds limit",
                    limit=LIMITS["saturation_drive_max_db"], value=eff_drive,
                ))
            # Positive output trim adds level — charge it to the same gain wallet
            # as gain moves / comp makeup so the agent can't dodge the cap with it.
            eff_out = level_charge_db(action)
            if eff_out > 0:
                _charge_gain(i, action.stem_id, eff_out,
                             note=" (incl. saturation output)")

        # OTT (2026-07-07): full depth is the user's call; and like the mb_comp `up`
        # mode, its upward side pushes loudness — not the agent's to do on the master
        # (only the limiter may). Positive out/band gains are charged to the gain
        # wallet like comp makeup.
        elif isinstance(action, OttAction):
            if action.depth > LIMITS["ott_depth_max"]:
                violations.append(Violation(
                    action_index=i, rule="ott_depth_max",
                    message=f"OTT depth {action.depth:.2f} exceeds the agent limit "
                            f"({LIMITS['ott_depth_max']:.2f}); deeper is the user's knob",
                    limit=LIMITS["ott_depth_max"], value=action.depth))
            if action.stem_id == "master":
                violations.append(Violation(
                    action_index=i, rule="master_comp_makeup_max_db",
                    message="OTT's upward compression pushes loudness; not allowed "
                            "on the master (use the limiter — the user may still "
                            "hand-add one)",
                    limit=0.0, value=1.0))
            else:
                # in_gain_db is applied at the input and never taken back out
                # (dsp._ott), so at depth 0 it is exactly a +12 dB level move that
                # nothing charged — and the OTT telemetry measures post-in-gain, so
                # over_compressed could not see it either (2026-09-01 sweep, P1).
                # …and the three stages are in SERIES (input, the loudest band, output), so
                # the level move is their SUM, not their largest: in +6, bands +6, out +6 was
                # charged 6 and played +18 (full audit 2026-10-02, D9·R1).
                makeup = level_charge_db(action)
                if makeup:
                    _charge_gain(i, action.stem_id, makeup,
                                 note=" (incl. OTT input/output gain)")

        # Overdrive (2026-07-07): cap the agent's drive (the user knob goes to 36)
        # and charge a positive output trim like saturation output.
        elif isinstance(action, OverdriveAction):
            if action.drive_db > LIMITS["overdrive_drive_max_db"]:
                violations.append(Violation(
                    action_index=i, rule="overdrive_drive_max_db",
                    message=f"overdrive drive {action.drive_db:+.1f} dB exceeds limit",
                    limit=LIMITS["overdrive_drive_max_db"], value=action.drive_db))
            if action.output_db > 0:
                _charge_gain(i, action.stem_id, action.output_db,
                             note=" (incl. overdrive output)")

        # Chorus (chorus_law, 2026-10-28): bound the MIX, the knob that decides whether this thickens
        # a source or replaces it, and charge what it adds in level: the reference's curve keeps the
        # dry full and lays the wet on top, so it LIFTS (level_charge_db: + VOLUME).
        elif isinstance(action, ChorusAction):
            if action.mix > LIMITS["chorus_mix_max"]:
                violations.append(Violation(
                    action_index=i, rule="chorus_mix_max",
                    message=f"chorus mix {action.mix:.2f} exceeds limit "
                            f"(that is an effect, not a thickener)",
                    limit=LIMITS["chorus_mix_max"], value=action.mix))
            lift = level_charge_db(action)
            if lift:
                _charge_gain(i, action.stem_id, lift, note=" (the chorus's dry + wet, and its volume)")

        # Flanger (2026-09-06): the same wet-blend bound as the chorus, plus the one the
        # chorus does not need — the return path. Both are the agent's ceiling only.
        elif isinstance(action, FlangerAction):
            if action.mix > LIMITS["flanger_mix_max"]:
                violations.append(Violation(
                    action_index=i, rule="flanger_mix_max",
                    message=f"flanger mix {action.mix:.2f} exceeds limit "
                            f"(that is an effect, not a movement)",
                    limit=LIMITS["flanger_mix_max"], value=action.mix))
            if abs(action.feedback) > LIMITS["flanger_feedback_max"]:
                violations.append(Violation(
                    action_index=i, rule="flanger_feedback_max",
                    message=f"flanger feedback {action.feedback:+.2f} exceeds limit "
                            f"(a comb that near self-oscillation is the user's call)",
                    limit=LIMITS["flanger_feedback_max"], value=abs(action.feedback)))

        # Auto Filter (Phase 63): the resonance's peak and the sweep's reach - the agent's ceiling
        # only; the user's knobs go all the way.
        elif isinstance(action, AutoFilterAction):
            if action.resonance > LIMITS["auto_filter_resonance_max"]:
                violations.append(Violation(
                    action_index=i, rule="auto_filter_resonance_max",
                    message=f"auto filter resonance {action.resonance:.2f} exceeds limit "
                            f"(its peak would pass the 6 dB a pass may lift)",
                    limit=LIMITS["auto_filter_resonance_max"], value=action.resonance))
            if action.lfo_amount_oct > LIMITS["auto_filter_lfo_amount_max_oct"]:
                violations.append(Violation(
                    action_index=i, rule="auto_filter_lfo_amount_max_oct",
                    message=f"auto filter LFO amount {action.lfo_amount_oct:.1f} oct exceeds "
                            f"limit (a sweep that wide is an effect, the user's call)",
                    limit=LIMITS["auto_filter_lfo_amount_max_oct"], value=action.lfo_amount_oct))
            if abs(action.env_amount_oct) > LIMITS["auto_filter_env_amount_max_oct"]:
                violations.append(Violation(
                    action_index=i, rule="auto_filter_env_amount_max_oct",
                    message=f"auto filter envelope amount {action.env_amount_oct:+.1f} oct "
                            f"exceeds limit",
                    limit=LIMITS["auto_filter_env_amount_max_oct"],
                    value=abs(action.env_amount_oct)))

        elif isinstance(action, TiltAction):
            if abs(action.tilt_db) > LIMITS["tilt_max_db"]:
                violations.append(Violation(
                    action_index=i, rule="tilt_max_db",
                    message=f"tilt {action.tilt_db:+.1f} dB exceeds ±{LIMITS['tilt_max_db']:.0f} dB",
                    limit=LIMITS["tilt_max_db"], value=action.tilt_db,
                ))
            # 2026-07-12 gain staging: a positive tilt output trim adds level —
            # same gain wallet as gain moves / comp makeup (a cut is free).
            if action.output_db > 0:
                _charge_gain(i, action.stem_id, action.output_db,
                             note=" (incl. tilt output)")

        elif isinstance(action, ClipAction):
            # Clipping deeper than the floor crushes the master — bind the agent (not the user, who
            # can dial the knobs further; guardrails bind the agent only). The depth is where the
            # ceiling sits under the input's full scale: CLIP in dB minus GAIN (clip_law.crush_db).
            from .clip_law import crush_db
            crush = crush_db(action.model_dump())
            if crush < LIMITS["clip_threshold_min_db"]:
                violations.append(Violation(
                    action_index=i, rule="clip_threshold_min_db",
                    message=(f"a clip of {action.clip_pct:.0f} % after {action.gain_db:+.1f} dB of gain puts "
                             f"the ceiling {crush:+.1f} dB under the input's full scale: too aggressive"),
                    limit=LIMITS["clip_threshold_min_db"], value=crush,
                ))
            # GAIN (up to +12) adds level and slams what follows harder — charge it to the same
            # gain wallet as tilt / transient / saturation output (a cut is free). Without this an
            # agent gets +12 dB a pass off-budget.
            if action.gain_db > 0:
                _charge_gain(i, action.stem_id, action.gain_db,
                             note=" (incl. the clipper's gain)")

        # Transient v2: a positive output trim adds level — charge it to the same
        # gain wallet as gain moves / comp makeup / saturation output (a cut is free).
        # Its attack / sustain lift is level too (2026-10-05, reader D9·R6).
        elif isinstance(action, TransientAction):
            lift = level_charge_db(action)      # its larger lift + a positive output trim
            if lift > 0:
                _charge_gain(i, action.stem_id, lift,
                             note=" (incl. the transient shaper's lift and output)")

        # The de-esser (2026-10-27, deess_law): no ratio to bound - its depth is range_db, at most 24 dB
        # in the schema; what is bounded here is how many a chain holds.
        elif isinstance(action, DeessAction):
            # Same slot bound as the comp above: Phase 35 emits up to two ranged de-essers per
            # pass on sibilant bursts, and they accumulate across passes.
            de_covered = deess_coverage.setdefault(action.stem_id, [])
            if action.time_range_s:
                de_covered.append(tuple(action.time_range_s))
            else:
                # A WHOLE-FILE de-esser REPLACES the agent's existing one rather than appending
                # (session._apply_action: "a new one replaces the old"), so re-tuning it costs no
                # new slot. Counting it as one made a chain already holding kMaxDeess entries
                # refuse the very re-propose that IS how a whole-file de-esser is adjusted — the
                # comp twin 200 lines up models the same replacement (2026-09-08 pre-merge review).
                de_covered[:] = [r for r in de_covered if r is not None] + [None]
            if len(de_covered) > LIMITS["deessers_per_stem_max"]:
                violations.append(Violation(
                    action_index=i, rule="deessers_per_stem_max",
                    message=f"'{action.stem_id}' would carry {len(de_covered)} de-essers; "
                            f"a mix run adds at most {LIMITS['deessers_per_stem_max']} per chain — "
                            "widen the ranges or undo one",
                    limit=LIMITS["deessers_per_stem_max"], value=len(de_covered)))

        # Gate v2: a threshold up in program-level territory mutes wanted signal,
        # not bleed — bind the agent (the user's hand knob still spans the model
        # range). Hold/range/hysteresis/SC bounds live in the schema.
        elif isinstance(action, GateAction):
            if action.threshold_db > LIMITS["gate_threshold_max_db"]:
                violations.append(Violation(
                    action_index=i, rule="gate_threshold_max_db",
                    message=f"gate threshold {action.threshold_db:+.1f} dB is above "
                            f"the {LIMITS['gate_threshold_max_db']:+.0f} dB agent limit "
                            "(set it just under the source's hits, not up at "
                            "program level)",
                    limit=LIMITS["gate_threshold_max_db"], value=action.threshold_db,
                ))

        elif isinstance(action, SidechainDuckAction):
            # duck_law: gain = 1 - mix x (1 - curve), and every curve reaches (about) 0 - so the
            # deepest cut is 1 - mix, held to the agent's 12 dB (2026-10-28: the depth knob went)
            mix_max = 1.0 - 10.0 ** (-LIMITS["duck_depth_max_db"] / 20.0)
            if action.mix > mix_max + 1e-9:
                violations.append(Violation(
                    action_index=i, rule="duck_depth_max_db",
                    message=(f"duck mix {action.mix:.2f} cuts "
                             f"{-20.0 * math.log10(max(1e-6, 1.0 - action.mix)):.1f} dB at the curve's "
                             f"deepest - over the agent's {LIMITS['duck_depth_max_db']:.0f} dB "
                             f"(mix {mix_max:.2f} at most)"),
                    limit=round(mix_max, 3), value=action.mix,
                ))
            if action.trigger == 1 and not action.key_stem_id:
                violations.append(Violation(
                    action_index=i, rule="duck_track_needs_key",
                    message="a duck on trigger 'track' names its trigger track (key_stem_id)",
                    limit=0.0, value=0.0,
                ))
            # A replace that names neither a key nor a trigger, over a duck keyed by a track: every
            # field a proposal leaves out takes its default, and the default is SYNC - the kick's
            # duck became a grid pump (the pre-merge review of 2026-10-28). Said, not guessed.
            if (not action.key_stem_id and "trigger" not in action.model_fields_set
                    and stem_chains is not None):
                from . import duck_law
                keyed = [e for e in replaced_entries(stem_chains.get(action.stem_id, []), action)
                         if duck_law.trigger_index(e.get("trigger", 0)) == 1 and e.get("key_stem_id")]
                if keyed:
                    violations.append(Violation(
                        action_index=i, rule="duck_keeps_trigger",
                        message=(f"this replaces the duck on {action.stem_id} that listens to "
                                 f"{keyed[0]['key_stem_id']!r}: name key_stem_id to keep that trigger, "
                                 "or trigger 'sync' to make it a grid pump"),
                        limit=0.0, value=0.0,
                    ))

        # Phase 29: the repair de-noise gate depth is capped (defense-in-depth;
        # the schema bound le=24 is the first line). master/bus targets are
        # refused structurally in session.validate_action_targets (stems only).
        elif isinstance(action, SpectralDenoiseAction):
            if action.reduction_db > LIMITS["spectral_denoise_reduction_max_db"]:
                violations.append(Violation(
                    action_index=i, rule="spectral_denoise_reduction_max_db",
                    message=f"spectral de-noise reduction {action.reduction_db:.1f} dB "
                            "exceeds limit",
                    limit=LIMITS["spectral_denoise_reduction_max_db"],
                    value=action.reduction_db,
                ))

        # Phase 31: tuning correction is bounded to ±a semitone (defense-in-depth
        # over the schema bound le=100; master/bus targets are refused
        # structurally in session.validate_action_targets — stems only).
        elif isinstance(action, TuneAction):
            if abs(action.cents) > LIMITS["tune_cents_max"]:
                violations.append(Violation(
                    action_index=i, rule="tune_cents_max",
                    message=f"tuning correction {action.cents:+.1f} cents exceeds "
                            f"the ±{LIMITS['tune_cents_max']:.0f} cent limit",
                    limit=LIMITS["tune_cents_max"], value=action.cents,
                ))

        # Phase 32: whole-song transpose bounded to ±an octave (defense-in-depth
        # over the schema bound le=12; master-only is enforced structurally in
        # session.validate_action_targets).
        elif isinstance(action, TransposeAction):
            if abs(action.semitones) > LIMITS["transpose_semitones_max"]:
                violations.append(Violation(
                    action_index=i, rule="transpose_semitones_max",
                    message=f"transpose {action.semitones:+d} semitones exceeds "
                            f"the ±{LIMITS['transpose_semitones_max']} semitone limit",
                    limit=LIMITS["transpose_semitones_max"], value=action.semitones,
                ))

        # Phase 15: match-EQ band corrections obey the same boost/cut rule
        # table as ordinary EQ bands (schema bounds are the first line).
        elif isinstance(action, MatchEqAction):
            for band, gain_db in action.band_gains_db.items():
                if gain_db > LIMITS["eq_boost_max_db"]:
                    violations.append(Violation(
                        action_index=i, rule="eq_boost_max_db",
                        message=f"match-EQ boost on '{band}' "
                                f"{gain_db:+.1f} dB exceeds limit",
                        limit=LIMITS["eq_boost_max_db"], value=gain_db))
                if gain_db < LIMITS["eq_cut_min_db"]:
                    violations.append(Violation(
                        action_index=i, rule="eq_cut_min_db",
                        message=f"match-EQ cut on '{band}' "
                                f"{gain_db:+.1f} dB exceeds limit",
                        limit=LIMITS["eq_cut_min_db"], value=gain_db))

        # Phase 38: the agent's dynamic EQ is downward-only and bounded in depth (it has no ratio
        # since 2026-10-27: dyneq_law). Schema bounds are still the first line; these are the
        # executable rule table.
        elif isinstance(action, DynamicEqAction):
            if action.gain_db < LIMITS["dynamic_eq_cut_min_db"]:
                violations.append(Violation(
                    action_index=i, rule="dynamic_eq_cut_min_db",
                    message=f"dynamic EQ cut {action.gain_db:+.1f} dB exceeds limit",
                    limit=LIMITS["dynamic_eq_cut_min_db"], value=action.gain_db,
                ))
            if action.gain_db > 0.0:
                violations.append(Violation(
                    action_index=i, rule="dynamic_eq_downward_only",
                    message="dynamic EQ is downward-only; use ordinary EQ for boosts",
                    limit=0.0, value=action.gain_db,
                ))

        # Phase 15: each multiband-comp band obeys the comp rule table; on
        # the master, per-band makeup must be 0 (only the limiter pushes
        # loudness — Phase 6 rule, applied per band).
        elif isinstance(action, MultibandCompAction):
            for band_name in ("low", "mid", "high"):
                band = getattr(action, band_name)
                if band.ratio > LIMITS["comp_ratio_max"]:
                    violations.append(Violation(
                        action_index=i, rule="comp_ratio_max",
                        message=f"multiband {band_name} ratio "
                                f"{band.ratio:.1f}:1 exceeds limit",
                        limit=LIMITS["comp_ratio_max"], value=band.ratio))
                if (action.stem_id == "master"
                        and band.makeup_db > LIMITS["master_comp_makeup_max_db"]):
                    violations.append(Violation(
                        action_index=i, rule="master_comp_makeup_max_db",
                        message=f"multiband {band_name} makeup must be 0 on "
                                "the master; use the limiter to push loudness",
                        limit=LIMITS["master_comp_makeup_max_db"],
                        value=band.makeup_db))
                if action.stem_id == "master" and band.up:
                    violations.append(Violation(
                        action_index=i, rule="master_comp_makeup_max_db",
                        message=f"multiband {band_name} upward mode pushes loudness; "
                                "not allowed on the master (use the limiter)",
                        limit=0.0, value=1.0))
            if action.stem_id != "master":
                # The loudest band's makeup is level added to the stem
                # (2026-06 review) — charge it like comp makeup above.
                # UPWARD mode is level the agent added too. Its worst case is the CAP,
                # MB_MAX_BOOST_DB, at any ratio > 1 (level_charge_db has the derivation), and
                # the band's makeup then rides on top of it — so on a stem an upward band is
                # past the per-pass budget by itself: it is the user's knob (D9·R2).
                charge = level_charge_db(action)
                lifts = any(band.up and band.ratio > 1.0
                            for band in (action.low, action.mid, action.high))
                if charge:
                    _charge_gain(i, action.stem_id, charge,
                                 note=(f" (incl. multiband upward boost, up to "
                                       f"+{MB_MAX_BOOST_DB:.0f} dB on a quiet band — upward "
                                       "mode is the user's knob)" if lifts
                                       else " (incl. multiband makeup)"))

        # The ceiling rule is enforced pre-apply, not just flagged post-render
        # (docs/05 Phase 6). The schema bound (le=-1.0) is the first line;
        # this keeps the doc'd rule table true even if the bound drifts.
        elif isinstance(action, LimiterAction):
            if action.ceiling_dbtp > LIMITS["master_true_peak_max_dbtp"]:
                violations.append(Violation(
                    action_index=i, rule="master_true_peak_max_dbtp",
                    message=f"limiter ceiling {action.ceiling_dbtp:+.1f} dBTP is "
                            "above the master true-peak limit",
                    limit=LIMITS["master_true_peak_max_dbtp"],
                    value=action.ceiling_dbtp,
                ))

        # Phase 38: agent-owned ambience returns stay conservative. The
        # schema owns absolute raw bounds; these own policy intent.
        elif isinstance(action, SendAction):
            if action.gain_db > LIMITS["return_send_max_db"]:
                violations.append(Violation(
                    action_index=i, rule="return_send_max_db",
                    message=f"return send {action.gain_db:+.1f} dB is hotter "
                            f"than the {LIMITS['return_send_max_db']:+.1f} dB limit",
                    limit=LIMITS["return_send_max_db"], value=action.gain_db,
                ))

        elif isinstance(action, ReturnReverbAction):
            violations += _wet_insert_mix(i, action)
            if action.decay_s > LIMITS["return_reverb_decay_max_s"]:
                violations.append(Violation(
                    action_index=i, rule="return_reverb_decay_max_s",
                    message=f"reverb decay {action.decay_s:.1f} s exceeds limit",
                    limit=LIMITS["return_reverb_decay_max_s"],
                    value=action.decay_s,
                ))

        elif isinstance(action, ReturnConvolutionAction):
            violations += _wet_insert_mix(i, action)
            if action.decay_s > LIMITS["return_convolution_decay_max_s"]:
                violations.append(Violation(
                    action_index=i, rule="return_convolution_decay_max_s",
                    message=f"convolution reverb decay {action.decay_s:.1f} s exceeds limit",
                    limit=LIMITS["return_convolution_decay_max_s"],
                    value=action.decay_s,
                ))

        elif isinstance(action, ReturnDelayAction):
            violations += _wet_insert_mix(i, action)
            if action.feedback > LIMITS["return_delay_feedback_max"]:
                violations.append(Violation(
                    action_index=i, rule="return_delay_feedback_max",
                    message=f"delay feedback {action.feedback:.2f} exceeds limit",
                    limit=LIMITS["return_delay_feedback_max"],
                    value=action.feedback,
                ))

    # The native EQ bank is finite (kMaxBands = 32 across eq / tilt / match_eq on one chain);
    # a 33rd band renders offline and NOT in the monitor or the DAW export. Count what the
    # chain already carries plus what this proposal adds, per target.
    added: dict[str, int] = {}
    for i, action in enumerate(req.actions):
        n = _action_eq_bands(action)
        if n:
            added[action.stem_id] = added.get(action.stem_id, 0) + n
    for target, n_new in added.items():
        existing = sum(eq_bands_in(e) for e in (stem_chains or {}).get(target, []))
        if existing + n_new > LIMITS["eq_bands_max"]:
            idx = next(i for i, a in enumerate(req.actions)
                       if _action_eq_bands(a) and a.stem_id == target)
            violations.append(Violation(
                action_index=idx, rule="eq_bands_max",
                message=(f"'{target}' would carry {existing + n_new} EQ bands; the native bank "
                         f"holds {LIMITS['eq_bands_max']} (eq = 1, tilt = 2, match_eq = its bands) "
                         f"— remove a band before adding one"),
                limit=LIMITS["eq_bands_max"], value=existing + n_new))

    if signal_parents is not None:
        violations += [v for v in _path_gain_violations(req, signal_parents, (
            ("gain_per_stem_per_pass_db", "this pass", "",
             dict(cumulative_gain_this_pass_db if standing_gain_this_pass_db is None
                  else standing_gain_this_pass_db), ranged_pass_before,
             running, ranged_pass_w),
            ("gain_per_stem_per_session_db", "across the session",
             "; undo earlier gain moves to free budget",
             dict((cumulative_gain_session_db if standing_gain_session_db is None
                   else standing_gain_session_db) or {}), ranged_session_before,
             running_session, ranged_session_w)))
            # a wallet already refused on its own says so once
            if not any(w.rule == v.rule and w.action_index == v.action_index for w in violations)]

    if eq_ledger is not None:
        violations += _composed_eq_violations(req, eq_ledger, signal_parents)

    # The master's width floor on the product of its width devices (D9·R10) - said once: a
    # device already refused alone needs no second line.
    if not any(v.rule == "master_width_min" for v in violations):
        violations += _master_composed_width(req, stem_chains)

    return violations


def eq_bands_in(entry: dict) -> int:
    """How many native EQ-bank bands one chain entry occupies (0 for non-EQ types)."""
    t = entry.get("type")
    if t == "eq":
        return 1
    if t == "tilt":
        return 2
    if t == "match_eq":
        # SESSION STATE spells a match_eq's corrections `band_gains_db` (MatchEqAction;
        # dsp._match_eq reads that key) — `bands` is the MONITOR CONTRACT's key
        # (monitor_chain.py), and session._chains_for_guardrails never passes that shape.
        # Reading only `bands` counted every match_eq as ZERO (2026-09-08 audit 04·F5):
        # 31 eq + an 8-band match_eq measured 31, so a 32nd eq band validated clean and
        # the contract then carried 40 — 8 bands rendering offline and silently dropped
        # from the live monitor and the DAW bounce, the exact failure eq_bands_max was
        # written to prevent. Both spellings count.
        #
        # A state-shaped match_eq counts the FULL band set, not the keys the request happens
        # to name: since 2026-09-08 the contract FITS the bell bank to the offline FIR, and a
        # band the request omits still gets a bell to cancel its neighbours' bleed (audit
        # 04-F4). A two-key request therefore reaches the native bank as up to eight bands,
        # and counting two would under-reserve the bank in exactly the way this rule exists to
        # prevent. Conservative in the safe direction: a near-zero correction reserves more of
        # the bank than the native ends up using, and a refusal is cheaper than a silent drop.
        gains = entry.get("band_gains_db")
        if gains is not None:
            # …but a correction of nothing costs nothing: the fit of all-zeros is all-zeros and
            # the contract emits no bells at all, so an empty or flat match_eq reserves none of
            # the bank (the negative control this rule already carried).
            if not any(abs(float(g)) > 0.01 for g in gains.values()):
                return 0
            return len(BAND_LABELS)
        return len(entry.get("bands") or ())
    return 0
