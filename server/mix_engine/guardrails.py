"""Server-side guardrails. The agent cannot talk its way past this module.

All limits live in LIMITS so the UI can display them and tests can assert them.
See docs/03-mcp-tools.md for the rule table.
"""
from __future__ import annotations

from typing import Optional

from .models import (BAND_EDGES_HZ, BAND_LABELS, imager_crossovers,
                     BandWidthAction,
                     ClipAction, CompAction, DeessAction, DynamicEqAction, EqAction,
                     GainAction,
                     GateAction, ImagerAction, LimiterAction, MatchEqAction,
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
    if t in ("eq", "tilt", "overdrive", "clip"):
        return pos(d.get("output_db"))
    if t == "transient":
        # The shaper applies `attack_db * t + sustain_db * (1 - t)` and THEN the output trim
        # (dsp._transient): attack and sustain both at +6 is a flat +6 dB no wallet saw, and a
        # gain +6 in the same proposal put the stem +12 up in one pass (reader D9·R6). The
        # larger of the two is the most it lifts anything; a cut is free, as everywhere.
        return (pos(max(float(d.get("attack_db") or 0.0), float(d.get("sustain_db") or 0.0)))
                + pos(d.get("output_db")))
    if t == "saturation":
        return pos(max([d.get("output_db") or 0.0]
                       + [b.get("output_db") or 0.0 for b in blocks("low", "mid", "high")]))
    if t == "ott":
        band = max([b.get("gain_db") or 0.0 for b in blocks("low", "mid", "high")] or [0.0])
        return pos(d.get("in_gain_db")) + pos(band) + pos(d.get("out_gain_db"))
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
    # `imager 0/0/0` with ZERO violations while refusing a 1.92 dB `balance` nudge; the
    # 12.39 dB mono-sum collapse was free and the milder move was banned. Half the side
    # is a real "tighten the image" mastering move; below it the agent is monoing the
    # deliverable, which is never an automatic decision. Binds the AGENT only — the
    # user's own width knob still goes to 0 (shared control), and folding the LOWS to
    # mono stays free at any depth (see _MASTER_MONO_LOWS_MAX_HZ).
    "master_width_min": 0.5,
    "dynamic_eq_cut_min_db": -12.0,
    "hpf_max_hz_default": 120.0,
    "hpf_max_hz_bass_tagged": 40.0,
    "comp_ratio_max": 10.0,
    "master_true_peak_max_dbtp": -1.0,
    "master_comp_makeup_max_db": 0.0,  # only the limiter pushes loudness (Phase 6)
    "saturation_drive_max_db": 12.0,   # Phase 7
    # OTT's UPWARD side lifts quiet material by up to dsp.OTT_MAX_BOOST_DB (24) x depth, and no
    # wallet charges it: at the old cap of 0.8 that was 19 dB (a -45 dB stem measured +8.4 with
    # every gain at 0). Charging the worst case would take OTT on a stem away from the agent, as
    # it did for upward multiband; the cap is set instead so that worst case IS the per-pass
    # rail - 24 x 0.25 = 6 dB (James, 2026-10-05, reader D9·R6). The RUN's cap only: the
    # device's own range, and the user's knob, still go to 1.
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
    # to the pass rail instead (CLAUDE.md batch Py1 rule 2): at 0.3 the peak is 5.4 dB at 12
    # dB/oct and 4.0 at 24 - inside eq_resonance_max_db's 6. Its sweep is capped to keep
    # "movement" movement: three octaves either way from the LFO, three from the envelope.
    "auto_filter_resonance_max": 0.3,  # the user knob still goes to 1 (Q 16)
    "auto_filter_lfo_amount_max_oct": 3.0,
    "auto_filter_env_amount_max_oct": 3.0,   # |env_amount_oct|
    "clip_threshold_min_db": -6.0,     # Phase 41: agent can't clip deeper than -6 dBFS
    "gate_threshold_max_db": -20.0,    # gate v2: an agent gate above -20 dBFS eats program
                                       # material, not bleed (the user knob still goes to 0)
    "tilt_max_db": 6.0,                # Phase 41: agent tilt capped at ±6 (like eq boosts)
    "duck_depth_max_db": 12.0,         # Phase 7
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
# prompt and knowledge file recommend it by name ("imager width_low 0 = tight, club-safe
# bottom"; "utility bass_mono"), and it IMPROVES mono compatibility rather than costing
# it. So the master width floor exempts a band whose TOP EDGE sits at or below this — and
# only that band: `imager crossover_lo_hz 4000 / width_low 0`, or a `band_width` scrubbed
# to 20 Hz–20 kHz, is the same destruction wearing the exemption's clothes. 300 Hz is
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
                 "imager width_low at its default crossover, utility bass_mono, "
                 "band_width on sub/bass)."),
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


def _band_width_key(band, lo_hz, hi_hz):
    """Which band a band_width entry narrows. A free lo/hi scrub is its own band; otherwise the
    fixed label. Entries sharing a key COMPOSE (see _master_band_width_scales)."""
    return (float(lo_hz), float(hi_hz)) if (lo_hz is not None and hi_hz is not None) else band


def _master_band_width_scales(
        stem_chains: Optional[dict[str, list[dict]]]) -> dict:
    """The band_width scale the MASTER already carries, per band, as a PRODUCT.

    Unlike utility / width / imager — native singletons session._mutate REPLACES — a band_width
    entry APPENDS (session.py: "eq / ... / band_width append") and dsp._band_width_scale
    re-isolates the same mask each time, so N entries multiply. A per-action floor therefore
    bound nothing: eight legal 0.5s inside ONE proposal took the mid band to 0.0039 (-48 dB of
    side) with zero violations, while a milder single imager move was refused (2026-09-08
    pre-merge review)."""
    acc: dict = {}
    for entry in (stem_chains or {}).get("master", []):
        if entry.get("type") != "band_width":
            continue
        key = _band_width_key(entry.get("band"), entry.get("lo_hz"), entry.get("hi_hz"))
        acc[key] = acc.get(key, 1.0) * float(entry.get("width", 1.0))
    return acc


#: The devices that scale the master's SIDE, and the three of them a chain holds ONE of (a new
#: one replaces the old - session._mutate); band_width appends.
_MASTER_WIDTH_KINDS = ("width", "utility", "imager", "band_width")
_MASTER_WIDTH_SINGLETONS = ("width", "utility", "imager")
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
    if t == "imager":
        lo, hi = imager_crossovers(d.get("crossover_lo_hz", 150.0), d.get("crossover_hi_hz", 2000.0))
        return [(0.0, lo, float(d.get("width_low", 1.0))),
                (lo, hi, float(d.get("width_mid", 1.0))),
                (hi, top, float(d.get("width_high", 1.0)))]
    if t == "band_width":
        if d.get("lo_hz") is not None and d.get("hi_hz") is not None:
            lo, hi = float(d["lo_hz"]), float(d["hi_hz"])
        elif d.get("band") in BAND_LABELS:
            k = BAND_LABELS.index(d["band"])
            lo, hi = float(BAND_EDGES_HZ[k]), float(BAND_EDGES_HZ[k + 1])
        else:
            return []
        return [(lo, hi, float(d.get("width", 1.0)))]
    return []


def _master_composed_width(req: ProposalRequest,
                           stem_chains: Optional[dict[str, list[dict]]]) -> list[Violation]:
    """The master's stereo floor, on the PRODUCT (reader D9·R10).

    Each device was held at `master_width_min` alone, and they are in series: width 0.5 x
    utility 0.5 x imager 0.5 x band_width 0.5 is 0.0625 of the side in the mids - the
    deliverable mono-folded (07·F3) from four legal moves, in one accepted proposal. The side
    scale the master would carry is composed here per frequency: what its chain holds, with a
    proposed width / utility / imager REPLACING the one there (as `session._mutate` does) and
    a band_width multiplying in. Judged above `_MASTER_MONO_LOWS_MAX_HZ` only, and only where
    THIS proposal narrows - the user's own narrowing is never the agent's refusal, so an EQ
    or a widening move on a master the user has narrowed still lands."""
    proposed = [(i, a.model_dump()) for i, a in enumerate(req.actions)
                if getattr(a, "stem_id", None) == "master" and a.type in _MASTER_WIDTH_KINDS]
    if not proposed:
        return []
    replaced = {d["type"] for _, d in proposed if d["type"] in _MASTER_WIDTH_SINGLETONS}
    standing = [e for e in (stem_chains or {}).get("master", [])
                if e.get("type") in _MASTER_WIDTH_KINDS and e.get("type") not in replaced]
    last = {d["type"]: i for i, d in proposed if d["type"] in _MASTER_WIDTH_SINGLETONS}
    moves = [(i, d) for i, d in proposed
             if d["type"] not in _MASTER_WIDTH_SINGLETONS or last[d["type"]] == i]
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
                 f"of its side at {span}, below the {floor:.2f} floor - width, utility, imager "
                 "and band_width multiply. Narrow with ONE of them, or less with each; "
                 "monoing the deliverable is the user's call"),
        limit=floor, value=round(scale, 3))]


def _band_top_hz(action: BandWidthAction) -> float:
    """Upper edge of a band_width's band — the free lo_hz/hi_hz scrub if it carries
    one (2026-07-19), otherwise the fixed label's edge. Mirrors dsp._band_width_scale."""
    if action.lo_hz is not None and action.hi_hz is not None:
        return float(action.hi_hz)
    return float(BAND_EDGES_HZ[BAND_LABELS.index(action.band) + 1])


_RESONANT_SHAPES = ("hpf", "lpf", "notch", "bandpass")
# The shelf family resonates too: RBJ shelves use q as a genuine resonance term,
# so past ~0.707 the response overshoots the declared plateau — a +6 dB low
# shelf at q=8 measures +18.3 dB, and a −6 dB CUT at q=8 measures +12.3 dB of
# boost (sign-inverted). The nominal-gain rules can't see any of that.
_SHELF_SHAPES = ("low_shelf", "high_shelf", "tilt_shelf")


def _eq_band_sections(shape: str, freq_hz, q, gain_db, slope_db_oct,
                      sample_rate: float = 48000.0) -> list:
    """One EQ band's biquad sections, built exactly as `dsp._eq_band` renders it (the same
    coefficient makers). Shared by the single-band resonance rail and the composed rail."""
    from .reference_monitor import biquad_coeffs, slope_cascade_coeffs

    freq_hz, q = float(freq_hz), float(q)
    gain_db = float(gain_db or 0.0)   # ignored by the cut shapes, real for shelves
    slope = int(slope_db_oct or 6)
    if shape in ("hpf", "lpf") and slope > 6:
        return slope_cascade_coeffs(shape, sample_rate, freq_hz, q, slope)
    if shape == "tilt_shelf":
        # exactly apply_eq_band's construction: a matched low-shelf (−g/2) +
        # high-shelf (+g/2) pair at the pivot, sharing the band's q
        return [biquad_coeffs("low_shelf", sample_rate, freq_hz, q, -gain_db / 2.0),
                biquad_coeffs("high_shelf", sample_rate, freq_hz, q, gain_db / 2.0)]
    return [biquad_coeffs(shape, sample_rate, freq_hz, q, gain_db)]


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
EQ_CURVE_TYPES = EQ_FAMILY + ("overdrive",)
#: The curve types that REPLACE the target's entry of their type (a whole-file singleton, as the
#: gain wallet's session._TRIM_SINGLETONS): the ledger holds the ONE that stands - and the one
#: that stood when the pass began - never every one applied (the pre-merge review of 2026-10-20:
#: the agent re-proposing its own tone-1 overdrive was refused at +12 dB, the first one counted
#: as if it were still there beside the second).
EQ_CURVE_SINGLETONS = ("overdrive",)
_EQ_GRID_N = 512          # 20 Hz..20 kHz, ~51 points/octave: a q=8 bell's peak reads within 0.05 dB
_EQ_GRID_RATE = 48000.0   # the stand-in rate _eq_band_peak_db uses, for the same reason
_EQ_TOL_DB = 0.05         # dB are 0.1-quantized in this engine; admit the limit itself
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
    return f, 2.0 * np.pi * f / _EQ_GRID_RATE


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
                                            "band_gains_db", "tone")}, sort_keys=True, default=str)
    boost, cut = _curves_cached(key)
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

    d = json.loads(key)
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
                                    _EQ_GRID_RATE))
        if band["shape"] in _RESONANT_SHAPES:
            return np.maximum(resp, 0.0), np.zeros_like(resp)
        return resp, resp
    if d["type"] == "tilt":
        from .dsp import TILT_Q   # exactly dsp._tilt: low shelf −tilt + high shelf +tilt
        from .reference_monitor import biquad_coeffs
        tilt, pivot = float(d.get("tilt_db") or 0.0), float(d.get("pivot_hz") or 1000.0)
        resp = db([biquad_coeffs("low_shelf", _EQ_GRID_RATE, pivot, TILT_Q, -tilt),
                   biquad_coeffs("high_shelf", _EQ_GRID_RATE, pivot, TILT_Q, tilt)])
        return resp, resp
    if d["type"] == "overdrive":
        # the tone tilt, exactly dsp._overdrive: lp + 2 tone (x - lp) around a 1st-order low-pass at
        # OD_TONE_HZ. Its LIFT only (tone > 0.5) is charged; a darker tone is never credit.
        from .dsp import OD_TONE_HZ
        from .reference_monitor import biquad_coeffs
        tone = min(max(float(d.get("tone", 0.5) if d.get("tone") is not None else 0.5), 0.0), 1.0)
        b, a = biquad_coeffs("lpf", _EQ_GRID_RATE, OD_TONE_HZ, 0.707, 0.0)
        lp = freqz(b, a, worN=w)[1]
        resp = 20.0 * np.log10(np.abs(lp + 2.0 * tone * (1.0 - lp)) + 1e-12)
        return np.maximum(resp, 0.0), np.zeros_like(resp)
    from .dsp import match_eq_fir   # the rendered FIR itself, so this cannot drift from it
    fir = match_eq_fir(d.get("band_gains_db") or {}, int(_EQ_GRID_RATE))
    resp = 20.0 * np.log10(np.abs(freqz(fir, [1.0], worN=w)[1]) + 1e-12)
    return resp, resp


def _ledger_curves(ledger: dict, target: str, replaced=()):
    """((pass boost, pass cut), (session boost, session cut)) the ledger already holds on one
    target — the session scope includes the trimmed records' folded carry.

    A singleton (`EQ_CURVE_SINGLETONS`) is the entry that STANDS in the session, and in the pass
    what it moved since the pass began (`standing` - `prior`, the gain wallet's delta). `replaced`
    names the singleton types the caller's proposal brings its own entry of: what stands leaves
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
        sb, sc = sb + np.asarray(carry["boost"]), sc + np.asarray(carry["cut"])
    for kind, slot in ((ledger.get("singletons") or {}).get(target) or {}).items():
        (s_b, s_c), (p_b, p_c) = curves_of(slot.get("standing")), curves_of(slot.get("prior"))
        if kind in replaced:
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
            replaces.setdefault(action.stem_id, set()).add(action.type)

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


def validate_proposal(
    req: ProposalRequest,
    cumulative_gain_this_pass_db: dict[str, float],
    accepted_proposals_this_pass: int = 0,
    stem_tags: Optional[dict[str, list[str]]] = None,
    cumulative_gain_session_db: Optional[dict[str, float]] = None,
    stem_chains: Optional[dict[str, list[dict]]] = None,
    eq_ledger: Optional[dict] = None,
    ranged_gain_this_pass_db: Optional[dict[str, tuple[float, float]]] = None,
    ranged_gain_session_db: Optional[dict[str, tuple[float, float]]] = None,
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
    ranged_pass_w = {k: tuple(v) for k, v in (ranged_gain_this_pass_db or {}).items()}
    ranged_session_w = {k: tuple(v) for k, v in (ranged_gain_session_db or {}).items()}
    coverage = _comp_coverage(stem_chains)
    deess_coverage = _comp_coverage(stem_chains, "deess")
    master_bw_scales = _master_band_width_scales(stem_chains)

    def _charge_gain(i: int, stem_id: str, delta_db: float,
                     note: str = "", in_range: bool = False) -> None:
        """Walk both gain budgets — gain moves and (2026-06 review) comp
        makeup draw from the same wallet. A whole-file move nets (a −3 after a +3 is 0, and it
        is); a time-ranged one (`in_range`) is one-sided, so a cut in one region never pays
        for a lift in another (D9·R4)."""
        for net_w, ranged_w, rule, what, tail in (
                (running, ranged_pass_w, "gain_per_stem_per_pass_db", "this pass", ""),
                (running_session, ranged_session_w, "gain_per_stem_per_session_db",
                 "across the session", "; undo earlier gain moves to free budget")):
            net = net_w.get(stem_id, 0.0)
            up, down = ranged_w.get(stem_id, (0.0, 0.0))
            if in_range:
                up, down = up + max(delta_db, 0.0), down + min(delta_db, 0.0)
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
            else:
                net_w[stem_id] = net

    faders = dict(fader_db or {})
    for i, action in enumerate(req.actions):
        if isinstance(action, GainAction):
            _charge_gain(i, action.stem_id, action.delta_db, in_range=is_ranged(action))
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

        elif isinstance(action, ImagerAction):
            if action.stem_id == "master":
                # Each band judged by its OWN top edge: the low band ends at
                # crossover_lo_hz (150 Hz by default — "mono the lows", free), the mid at
                # crossover_hi_hz, the high at Nyquist. A low band pushed up to 4 kHz is
                # not the lows and gets no exemption; a mid band sitting under 200 Hz is.
                xlo, xhi = imager_crossovers(action.crossover_lo_hz, action.crossover_hi_hz)
                for name, width, top in (
                        ("width_low", action.width_low, xlo),
                        ("width_mid", action.width_mid, xhi),
                        ("width_high", action.width_high, float("inf"))):
                    violations += _master_narrowing(i, f"imager {name}", width,
                                                    up_to_hz=top)

        elif isinstance(action, BandWidthAction):
            if action.stem_id == "master":
                # CUMULATIVE: band_width appends and composes multiplicatively, so the floor is
                # judged on the product -- across passes (seeded from the chain) and within this
                # proposal. Judging each action alone bound nothing.
                key = _band_width_key(action.band, action.lo_hz, action.hi_hz)
                scale = master_bw_scales.get(key, 1.0) * action.width
                master_bw_scales[key] = scale
                what = (f"band_width {action.band!r} width" if scale == action.width
                        else f"band_width {action.band!r} width, {scale:.3f} once it composes "
                             f"with the band_width already on the master,")
                violations += _master_narrowing(i, what, scale,
                                                up_to_hz=_band_top_hz(action))

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
                peak_db = _eq_band_peak_db(action.band)
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
                hpf_limit = (LIMITS["hpf_max_hz_bass_tagged"] if bass_tagged
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
                             note=" (incl. eq output)", in_range=is_ranged(action))

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
                             note=" (incl. comp makeup)", in_range=is_ranged(action))
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
                else:
                    covered[:] = [None]  # replaces any existing whole-file comp
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
                            f"the engine holds {LIMITS['comps_per_stem_max']} per chain — "
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

        # Chorus (2026-09-06): bound the WET BLEND, the one knob that decides whether this
        # thickens a source or replaces it. Nothing to charge to the gain budget — chorus has
        # no gain control (though dry+wet can sum a few dB at some frequencies, which is what
        # the mix cap also keeps in hand).
        elif isinstance(action, ChorusAction):
            if action.mix > LIMITS["chorus_mix_max"]:
                violations.append(Violation(
                    action_index=i, rule="chorus_mix_max",
                    message=f"chorus mix {action.mix:.2f} exceeds limit "
                            f"(that is an effect, not a thickener)",
                    limit=LIMITS["chorus_mix_max"], value=action.mix))

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
            # Clipping deeper than the floor crushes the master — bind the agent (not
            # the user, who can dial the knob further; guardrails bind the agent only).
            if action.threshold_db < LIMITS["clip_threshold_min_db"]:
                violations.append(Violation(
                    action_index=i, rule="clip_threshold_min_db",
                    message=f"clip threshold {action.threshold_db:+.1f} dB is too aggressive",
                    limit=LIMITS["clip_threshold_min_db"], value=action.threshold_db,
                ))
            # A positive output trim (level makeup, up to +12) adds level and slams the
            # limiter harder — charge it to the same gain wallet as tilt/transient/sat
            # output (a cut is free). Without this an agent gets +12 dB/pass off-budget.
            if action.output_db > 0:
                _charge_gain(i, action.stem_id, action.output_db,
                             note=" (incl. clip output)")

        # Transient v2: a positive output trim adds level — charge it to the same
        # gain wallet as gain moves / comp makeup / saturation output (a cut is free).
        # Its attack / sustain lift is level too (2026-10-05, reader D9·R6).
        elif isinstance(action, TransientAction):
            lift = level_charge_db(action)      # its larger lift + a positive output trim
            if lift > 0:
                _charge_gain(i, action.stem_id, lift,
                             note=" (incl. the transient shaper's lift and output)")

        # De-ess v2: the freed ratio obeys the same dynamics ceiling as compressors
        # and dynamic EQ (the schema allows 20 for the user's hand knob).
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
                            f"the engine holds {LIMITS['deessers_per_stem_max']} per chain — "
                            "widen the ranges or undo one",
                    limit=LIMITS["deessers_per_stem_max"], value=len(de_covered)))
            if action.ratio > LIMITS["comp_ratio_max"]:
                violations.append(Violation(
                    action_index=i, rule="comp_ratio_max",
                    message=f"de-ess ratio {action.ratio:.1f}:1 exceeds limit",
                    limit=LIMITS["comp_ratio_max"], value=action.ratio,
                ))

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
            if action.depth_db > LIMITS["duck_depth_max_db"]:
                violations.append(Violation(
                    action_index=i, rule="duck_depth_max_db",
                    message=f"duck depth {action.depth_db:+.1f} dB exceeds limit",
                    limit=LIMITS["duck_depth_max_db"], value=action.depth_db,
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

        # Phase 38: dynamic EQ is downward-only and follows the same dynamics
        # ratio ceiling as compressors. Schema bounds are still the first line;
        # these are the executable rule table.
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
            if action.ratio > LIMITS["comp_ratio_max"]:
                violations.append(Violation(
                    action_index=i, rule="comp_ratio_max",
                    message=f"dynamic EQ ratio {action.ratio:.1f}:1 exceeds limit",
                    limit=LIMITS["comp_ratio_max"], value=action.ratio,
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
                  else standing_gain_this_pass_db), dict(ranged_gain_this_pass_db or {}),
             running, ranged_pass_w),
            ("gain_per_stem_per_session_db", "across the session",
             "; undo earlier gain moves to free budget",
             dict((cumulative_gain_session_db if standing_gain_session_db is None
                   else standing_gain_session_db) or {}), dict(ranged_gain_session_db or {}),
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
