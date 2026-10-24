"""Guardrail holes found in the 2026-08-06 and 2026-09-08 device audits.

Each of these was a documented invariant the agent could walk straight past,
because the rule inspected a parameter the DSP ignores — or, in the 2026-09-08
pair, a parameter that does not exist in the shape the rule is handed.
"""
from __future__ import annotations

import pytest

from mix_engine import guardrails
from mix_engine.dsp import MB_MAX_BOOST_DB
from mix_engine.guardrails import LIMITS, _eq_band_peak_db, validate_proposal
from mix_engine.models import (BAND_LABELS, BandWidthAction, EqAction, EqBand,
                               ImagerAction, MatchEqAction, MbBand,
                               MultibandCompAction, ProposalRequest,
                               UtilityAction, WidthAction)


def _rules(action, chains=None):
    req = ProposalRequest(rationale="x", predicted_impact=[], actions=[action])
    return [v.rule for v in validate_proposal(req, {}, stem_chains=chains)]


def _eq(**band):
    band.setdefault("gain_db", 0.0)
    return EqAction(stem_id="stem_a", band=EqBand(**band))


# ---------- resonant filters bypassed eq_boost_max_db entirely ----------

def test_a_resonant_cut_is_rejected():
    """`gain_db` is ignored for cut shapes, so nothing bounded `q`, and the guardrail
    measures the band's ACTUAL cascade response rather than a proxy.

    This used to be asserted on `slope_db_oct=96`, which measured +162 dB. That hole
    is gone: since 2026-09-08 the ENGINE clamps a real cascade to Butterworth, because
    the +127 dB it rendered was reachable from the user's own knobs and from
    `device.set`, neither of which ever meets a guardrail (device audit 04-eq-tone
    F10). So a 96 dB/oct band is flat now and correctly passes — the guardrail's
    subject moved to the single-section resonance the engine deliberately still
    allows, which is where a musical resonant filter lives and where the agent's
    +6 dB policy is the only thing binding it.
    """
    action = _eq(shape="lpf", freq_hz=3000.0, q=8.0, slope_db_oct=12)
    assert _eq_band_peak_db(action.band) > LIMITS["eq_resonance_max_db"]   # +18.1 dB
    assert "eq_resonance_max_db" in _rules(action)
    # …and a milder one is caught too, so the rule is not a single-value tripwire.
    assert "eq_resonance_max_db" in _rules(_eq(shape="lpf", freq_hz=3000.0, q=4.0, slope_db_oct=12))


def test_the_engine_bound_removed_the_high_slope_hole():
    """The other half: the band that used to measure +162 dB is now FLAT, so the
    guardrail passing it is correct rather than a regression. If a future change
    re-opened the cascade, this fails and `test_a_resonant_cut_is_rejected` would
    still pass — which is why both exist."""
    band = _eq(shape="lpf", freq_hz=3000.0, q=8.0, slope_db_oct=96).band
    assert abs(_eq_band_peak_db(band)) < 1.0, "a high-slope cascade must run Butterworth"


@pytest.mark.parametrize("band", [
    dict(shape="hpf", freq_hz=80.0, q=0.707, slope_db_oct=12),
    dict(shape="hpf", freq_hz=100.0, q=0.707, slope_db_oct=24),
    dict(shape="lpf", freq_hz=12000.0, q=0.707, slope_db_oct=48),
    dict(shape="notch", freq_hz=1000.0, q=4.0),
    dict(shape="bandpass", freq_hz=1000.0, q=2.0),
    dict(shape="hpf", freq_hz=120.0, q=1.5, slope_db_oct=12),   # mildly resonant, +4 dB
])
def test_ordinary_filters_still_pass(band):
    """The rule bounds the EFFECT, so musical filters — including a mildly
    resonant one — are untouched."""
    assert _rules(_eq(**band)) == []


def test_resonance_limit_matches_the_boost_limit():
    assert LIMITS["eq_resonance_max_db"] == LIMITS["eq_boost_max_db"]


# ---------- utility balance could hard-pan the master ----------

def test_master_balance_is_refused():
    """balance=+1.0 on the master collapses the mix into one channel."""
    assert "utility_balance_max" in _rules(
        UtilityAction(stem_id="master", balance=1.0, mono=True, width=0.0))
    assert "utility_balance_max" in _rules(
        UtilityAction(stem_id="master", balance=0.2))


def test_stem_balance_nudge_is_allowed():
    assert _rules(UtilityAction(stem_id="stem_a", balance=0.3)) == []
    assert "utility_balance_max" in _rules(UtilityAction(stem_id="stem_a", balance=0.9))


def test_master_width_and_mono_are_no_longer_free():
    """SUPERSEDED 2026-09-08 (audit 07·F3). This asserted `_rules(...) == []` for
    `mono=True, width=0.0` on the master, on the reading that "width/mono are
    shaping (bounded by the model)". The model bounds the RANGE, and the range
    includes monoing the deliverable — see the master-image section below."""
    assert "master_width_min" in _rules(
        UtilityAction(stem_id="master", mono=True, width=0.0))


# ---------- upward multiband compression dodged the gain wallet ----------

def _mb(**band_kw):
    def band(**kw):
        return MbBand(threshold_db=-24.0, ratio=4.0, **kw)
    return MultibandCompAction(stem_id="stem_a", low=band(**band_kw),
                               mid=band(), high=band())


def test_upward_mode_is_charged_to_the_gain_budget():
    """Up mode added up to +12 LU in one action against a 6 dB/pass wallet."""
    assert "gain_per_stem_per_pass_db" in _rules(_mb(up=True))


def test_downward_multiband_is_unaffected():
    assert _rules(_mb()) == []


def test_mb_boost_constant_matches_the_engine():
    """guardrails mirrors the constant rather than importing dsp (pedalboard)."""
    assert guardrails.MB_MAX_BOOST_DB == MB_MAX_BOOST_DB


# ---------- shelves resonate too (2026-08-06 review follow-up) ----------
# RBJ shelves use q as a genuine resonance term: past ~0.707 the response
# overshoots the declared plateau, and the nominal-gain rules can't see it.

def test_resonant_shelf_is_rejected():
    """A legal-looking +6 dB low shelf at q=8 actually peaks at +18.3 dB."""
    action = _eq(shape="low_shelf", freq_hz=200.0, gain_db=6.0, q=8.0)
    assert _eq_band_peak_db(action.band) > 12.0          # the hole was real
    assert "eq_resonance_max_db" in _rules(action)


def test_shelf_cut_that_boosts_is_rejected():
    """Sign inversion: a −6 dB shelf CUT at q=8 measures +12.3 dB of BOOST —
    the agent asks for attenuation and the filter amplifies."""
    action = _eq(shape="low_shelf", freq_hz=200.0, gain_db=-6.0, q=8.0)
    assert _eq_band_peak_db(action.band) > 10.0
    assert "eq_resonance_max_db" in _rules(action)


def test_resonant_tilt_is_rejected():
    """tilt_shelf is two shelf halves sharing the band's q — same failure."""
    assert "eq_resonance_max_db" in _rules(
        _eq(shape="tilt_shelf", freq_hz=700.0, gain_db=6.0, q=8.0))


@pytest.mark.parametrize("band", [
    dict(shape="low_shelf", freq_hz=200.0, gain_db=6.0, q=0.5),
    dict(shape="low_shelf", freq_hz=200.0, gain_db=6.0, q=0.71),
    dict(shape="high_shelf", freq_hz=8000.0, gain_db=6.0, q=0.71),
    dict(shape="low_shelf", freq_hz=120.0, gain_db=-6.0, q=0.71),
    dict(shape="tilt_shelf", freq_hz=700.0, gain_db=6.0, q=0.71),
])
def test_max_legal_shelf_still_passes(band):
    """The +6 dB plateau IS the legal maximum: the rule rejects a resonant lip
    above the plateau, never the shelf itself (the +0.05 boundary guard)."""
    assert _rules(_eq(**band)) == []


# ---------- a 0.0 dB peak/shelf is a no-op the planner re-queued forever ----------

def test_zero_gain_peak_is_rejected():
    """2026-08-09: a residual that rounded to 0.0 dB produced a unity-gain EQ
    that could never close its target, so it was re-proposed every pass."""
    assert "eq_gain_nonzero" in _rules(
        _eq(shape="peak", freq_hz=4000.0, gain_db=0.0, q=1.0))


def test_zero_gain_cut_shapes_still_pass():
    """gain_db is IGNORED for the cut shapes — 0.0 is their normal value."""
    assert "eq_gain_nonzero" not in _rules(
        _eq(shape="hpf", freq_hz=80.0, q=0.707, slope_db_oct=12))


def test_zero_gain_band_with_output_trim_still_passes():
    """A band carrying the device's output trim still does something."""
    action = EqAction(stem_id="stem_a", output_db=-2.0,
                      band=EqBand(shape="peak", freq_hz=4000.0,
                                  gain_db=0.0, q=1.0))
    assert "eq_gain_nonzero" not in _rules(action)


# ---------- 2026-09-08 audit 07·F3: the master's stereo image had NO floor ----------
# `validate_proposal` refused a 1.92 dB `balance` nudge on the master and accepted a
# 12.39 dB mono-sum collapse (`invert_left`), a whole-mix `mono`, `width 0.0` and
# `imager 0/0/0` — all with ZERO violations.


def test_single_channel_polarity_flip_on_the_master_is_refused():
    """The measured worst case: one flipped channel takes 12.39 dB out of the mono
    sum of the finished mix. The prompt actively steers here ("invert_left/right fix
    a single out-of-phase channel") — true of a stem, never of a master."""
    for kw in ({"invert_left": True},
               {"invert_right": True},
               {"invert": True, "invert_left": False},    # legacy both, one opted out
               {"invert": True, "invert_right": False}):
        assert "master_polarity_split" in _rules(UtilityAction(stem_id="master", **kw)), kw


def test_both_channel_invert_on_the_master_stays_legal():
    """A whole-mix polarity flip is mono-sum preserving — harmless, and the only
    polarity move a master ever wants. Both spellings."""
    assert _rules(UtilityAction(stem_id="master", invert=True)) == []
    assert _rules(UtilityAction(stem_id="master", invert_left=True,
                                invert_right=True)) == []


def test_single_channel_flip_on_a_stem_stays_legal():
    """The repair the prompt actually describes: one out-of-phase mic or DI channel.
    The new rule is master-only, so this is untouched."""
    assert _rules(UtilityAction(stem_id="stem_a", invert_left=True)) == []
    assert _rules(UtilityAction(stem_id="drum_bus", invert_right=True)) == []


@pytest.mark.parametrize("action", [
    UtilityAction(stem_id="master", mono=True),                 # mono resolves to width 0
    UtilityAction(stem_id="master", width=0.0),
    UtilityAction(stem_id="master", width=0.25),
    WidthAction(stem_id="master", width=0.0),
    ImagerAction(stem_id="master", width_low=0.0, width_mid=0.0, width_high=0.0),
    ImagerAction(stem_id="master", width_high=0.0),
    ImagerAction(stem_id="master", width_mid=0.2),
    # The lows exemption is not a keyhole: a "low" band reaching 4 kHz is not the lows.
    ImagerAction(stem_id="master", width_low=0.0,
                 crossover_lo_hz=4000.0, crossover_hi_hz=8000.0),
    # ...nor is a band_width scrubbed across the whole spectrum.
    BandWidthAction(stem_id="master", band="air", width=0.0,
                    lo_hz=20.0, hi_hz=20000.0),
    BandWidthAction(stem_id="master", band="presence", width=0.0),
    # ...nor a Bass Mono above the lows: an LR8 split since the Utility audit (2026-10-21), and at
    # 500 Hz it takes the side at 300 Hz to 0.02.
    UtilityAction(stem_id="master", bass_mono=True, bass_mono_hz=500.0),
])
def test_narrowing_the_masters_image_below_the_floor_is_refused(action):
    assert "master_width_min" in _rules(action)


@pytest.mark.parametrize("action", [
    # The documented club-safe move at the imager's own 150 Hz default crossover —
    # our prompts and knowledge file both recommend it, and it IMPROVES mono compat.
    ImagerAction(stem_id="master", width_low=0.0),
    ImagerAction(stem_id="master", width_low=0.0, width_high=1.4),
    # A mid band that lives entirely in the lows is the same move under another name.
    ImagerAction(stem_id="master", width_mid=0.0,
                 crossover_lo_hz=30.0, crossover_hi_hz=200.0),
    # The Utility's own bass fold, up to the lows' edge (300 Hz).
    UtilityAction(stem_id="master", bass_mono=True, bass_mono_hz=120.0),
    UtilityAction(stem_id="master", bass_mono=True, bass_mono_hz=300.0),
    # Tightening the image is not monoing it.
    UtilityAction(stem_id="master", width=0.5),
    WidthAction(stem_id="master", width=0.8),
    ImagerAction(stem_id="master", width_high=1.3),
    # Per-band mono of the bottom.
    BandWidthAction(stem_id="master", band="sub", width=0.0),
    BandWidthAction(stem_id="master", band="bass", width=0.0),
    BandWidthAction(stem_id="master", band="air", width=0.0,
                    lo_hz=25.0, hi_hz=90.0),
    # ...and every one of these moves on a NON-master target: the floor binds the
    # deliverable, not the palette.
    UtilityAction(stem_id="stem_a", mono=True, width=0.0),
    WidthAction(stem_id="stem_a", width=0.0),
    ImagerAction(stem_id="stem_a", width_low=0.0, width_mid=0.0, width_high=0.0),
    BandWidthAction(stem_id="stem_a", band="air", width=0.0),
    BandWidthAction(stem_id="drum_bus", band="presence", width=0.0),
])
def test_the_floor_binds_only_the_master_and_only_its_image(action):
    assert _rules(action) == []


def test_the_master_floor_is_a_declared_limit():
    assert LIMITS["master_width_min"] == 0.5
    assert 0.0 < LIMITS["master_width_min"] < 1.0        # a floor, not a ban
    assert guardrails._MASTER_MONO_LOWS_MAX_HZ == 300.0


# ---------- 2026-09-08 audit 04·F5: a match_eq counted as ZERO bands ----------
# `eq_bands_in` read `entry["bands"]` — the MONITOR CONTRACT's key. Session state
# (what `_chains_for_guardrails` actually passes) spells it `band_gains_db`.


def _match_eq_entry(n_bands: int) -> dict:
    """A real MatchEqAction, dumped the way session._mutate stores it."""
    gains = {b: -1.5 for b in BAND_LABELS[:n_bands]}
    return MatchEqAction(stem_id="stem_a",
                         band_gains_db=gains).model_dump(exclude={"stem_id"})


def test_eq_bands_in_counts_a_match_eqs_bands():
    entry = _match_eq_entry(8)
    assert "band_gains_db" in entry and "bands" not in entry   # the shape the bug missed
    assert guardrails.eq_bands_in(entry) == 8


def test_eq_bands_in_still_reads_the_monitor_contract_spelling():
    """monitor_chain.py builds `bands`; both spellings have to count."""
    assert guardrails.eq_bands_in(
        {"type": "match_eq", "bands": [{"freq_hz": 100.0}] * 3}) == 3


def test_eq_bands_in_negative_control():
    """The counts that were already right stay right."""
    assert guardrails.eq_bands_in({"type": "eq"}) == 1
    assert guardrails.eq_bands_in({"type": "tilt"}) == 2
    assert guardrails.eq_bands_in({"type": "comp"}) == 0
    assert guardrails.eq_bands_in({"type": "match_eq", "band_gains_db": {}}) == 0


def test_the_32nd_band_over_a_match_eq_is_refused():
    """Measured: 31 eq + one 8-band match_eq counted 31, so a 32nd eq band validated
    clean and the monitor contract then carried 40 — 8 bands silently dropped from
    the live monitor and the DAW bounce while they rendered offline."""
    chain = [{"type": "eq"} for _ in range(31)] + [_match_eq_entry(8)]
    assert sum(guardrails.eq_bands_in(e) for e in chain) == 39
    assert "eq_bands_max" in _rules(
        _eq(shape="peak", freq_hz=3000.0, gain_db=-2.0, q=1.0),
        chains={"stem_a": chain})


def test_a_band_that_still_fits_over_a_match_eq_is_accepted():
    """Negative control: the rule counts, it does not just refuse near a match_eq."""
    chain = [{"type": "eq"} for _ in range(20)] + [_match_eq_entry(8)]   # 28 + 1 = 29
    assert _rules(_eq(shape="peak", freq_hz=3000.0, gain_db=-2.0, q=1.0),
                  chains={"stem_a": chain}) == []


def test_a_tilt_over_a_match_eq_charges_two_bands():
    """31 = 23 eq + 8 match_eq bands; a tilt costs 2, so it must not fit."""
    from mix_engine.models import TiltAction
    chain = [{"type": "eq"} for _ in range(23)] + [_match_eq_entry(8)]
    assert sum(guardrails.eq_bands_in(e) for e in chain) == 31
    assert "eq_bands_max" in _rules(
        TiltAction(stem_id="stem_a", tilt_db=1.5, pivot_hz=1000.0),
        chains={"stem_a": chain})


# ---------------------------------------------------------- the USER's own doors --
# The agent's proposal path is bounded by validate_proposal; the user's co-edit channel is
# bounded by session.py's own checks. `add_stem_device` has always carried the EQ-bank budget
# and `add_master_device` never did — so the MASTER, the chain most likely to carry a match_eq
# on top of a stack of eq bands, was the unguarded one (2026-09-08 device audit).

def test_add_master_device_enforces_the_eq_bank_budget(tmp_path):
    import shutil
    from pathlib import Path

    from mix_engine.guardrails import LIMITS
    from mix_engine.session import Session

    proj = tmp_path / "proj"
    shutil.copytree(Path(__file__).resolve().parents[2] / "test_projects" / "suno_demo", proj)
    session = Session(proj)
    session._state["master_chain"] = []

    # Fill the bank to one band under the limit, then the next one must be refused.
    for _ in range(LIMITS["eq_bands_max"] - 1):
        session.add_master_device("eq")
    session.add_master_device("eq")          # exactly at the limit — still fine
    with pytest.raises(ValueError, match="EQ bands"):
        session.add_master_device("eq")

    # NEGATIVE CONTROL: a non-EQ device is unaffected by the bank budget, so the guard is a
    # band budget and not a blanket "the master chain is full".
    session.add_master_device("comp")

    # …and an existing match_eq reserves the FULL band set, because the resolver fits a bell
    # per band. It is not user-addable (it comes from a reference match), so place it the way
    # the agent does — which is exactly the case the count was getting wrong.
    session._state["master_chain"] = [
        {"type": "match_eq", "stem_id": "master",
         "band_gains_db": {b: 1.0 for b in BAND_LABELS}}]
    for _ in range(LIMITS["eq_bands_max"] - len(BAND_LABELS)):
        session.add_master_device("eq")
    with pytest.raises(ValueError, match="EQ bands"):
        session.add_master_device("eq")


# ================= 2026-09-08 PRE-MERGE REVIEW ==================================================
# Three holes the audit's own fixes opened or left, each measured before it was closed.

def _rules_multi(actions, chains=None):
    req = ProposalRequest(rationale="x", predicted_impact=[], actions=list(actions))
    return [v.rule for v in validate_proposal(req, {}, stem_chains=chains)]


def _bw(width, band="mid", stem_id="master"):
    return BandWidthAction(stem_id=stem_id, band=band, width=width)


# ---------- band_width COMPOSES, so the master floor must judge the product ----------
# utility / width / imager are native singletons session._mutate REPLACES, so a per-action floor
# binds them. band_width APPENDS, and dsp._band_width_scale re-isolates the same mask each time,
# so N entries multiply. Measured before the fix: eight legal 0.5s in ONE proposal (inside the
# 8-action cap) took the mid band to a 0.0039 scale — -48 dB of side on a pure-side 600 Hz
# signal — with ZERO violations, while a far milder single imager move was refused.

def test_stacked_band_widths_cannot_mono_the_master():
    assert "master_width_min" in _rules_multi([_bw(0.5), _bw(0.5)])          # 0.25, under the floor


def test_eight_stacked_band_widths_are_refused():
    assert "master_width_min" in _rules_multi([_bw(0.5) for _ in range(8)])


def test_one_band_width_on_the_floor_stays_legal():
    """The control that matters: the floor is 0.5 and a single 0.5 sits exactly ON it, so a
    rule that simply refused band_width on the master would pass the test above and fail here."""
    assert _rules_multi([_bw(0.5)]) == []


def test_band_width_composes_with_what_the_chain_already_carries():
    """…and across passes, not only within one proposal."""
    chain = {"master": [{"type": "band_width", "band": "mid", "width": 0.5}]}
    assert "master_width_min" in _rules_multi([_bw(0.5)], chain)


def test_band_widths_on_different_bands_do_not_compose():
    assert _rules_multi([_bw(0.5, "mid"), _bw(0.5, "air")]) == []


def test_stacked_band_width_on_the_lows_stays_free():
    """Folding the bottom to mono is the one narrowing move the floor exempts, at any depth."""
    assert _rules_multi([_bw(0.0, "sub") for _ in range(4)]) == []


def test_stacked_band_width_on_a_stem_is_not_bounded():
    """The floor binds the DELIVERABLE. A stem may be monoed freely."""
    assert _rules_multi([_bw(0.0, "mid", "vox") for _ in range(4)]) == []


# ---------- a match_eq ACTION reserves the bank it will occupy ----------
# eq_bands_in counts an APPLIED match_eq as the full fitted bank, but the propose-time tally
# charged only eq/tilt — so the rule and the thing it guards disagreed by 8. Measured: a 25-band
# master + a two-key match_eq validated clean and the contract then carried 33, with the 33rd
# dropped from the monitor and the DAW bounce while it rendered offline.

def _eq_entries(n):
    return [{"type": "eq"} for _ in range(n)]


def _match_eq(**gains):
    return MatchEqAction(stem_id="master", band_gains_db=dict(gains))


def test_a_match_eq_action_charges_the_bank_it_will_occupy():
    chain = {"master": _eq_entries(25)}
    assert "eq_bands_max" in _rules_multi([_match_eq(sub=-3.0, air=2.0)], chain)


def test_a_match_eq_action_that_still_fits_is_accepted():
    """24 + 8 = 32, exactly the bank — the control that stops this becoming 'refuse match_eq'."""
    chain = {"master": _eq_entries(24)}
    assert _rules_multi([_match_eq(sub=-3.0, air=2.0)], chain) == []


def test_a_flat_match_eq_action_reserves_nothing():
    """A correction of nothing emits no bells, exactly as eq_bands_in reads it."""
    chain = {"master": _eq_entries(25)}
    assert _rules_multi([_match_eq(sub=0.0, air=0.0)], chain) == []


def test_the_action_and_entry_band_counters_agree():
    """The two halves of the rule must not drift apart again."""
    action = _match_eq(sub=-3.0, air=2.0)
    entry = action.model_dump(exclude={"stem_id"})
    assert guardrails._action_eq_bands(action) == guardrails.eq_bands_in(entry) == len(BAND_LABELS)


# ---------- the comp / de-ess slot limits must equal the native chain's ----------
# LIMITS says these mirror MonitorChain::kMaxComps / kMaxDeess, and nothing held them there —
# the only two rules in the file with neither a test nor a parity pin. A native bump would make
# guardrails refuse work the engine can do; a native CUT would re-open the silent-drop the slot
# work exists to close.

def test_the_slot_limits_match_the_native_chain():
    import pathlib
    import re
    src = (pathlib.Path(__file__).resolve().parents[2]
           / "native" / "Source" / "MonitorChain.h").read_text()
    for cname, limit in (("kMaxComps", "comps_per_stem_max"),
                         ("kMaxDeess", "deessers_per_stem_max")):
        m = re.search(rf"static constexpr int {cname}\s*=\s*(\d+)\s*;", src)
        assert m, f"could not find {cname} in MonitorChain.h — has the declaration moved?"
        assert int(m.group(1)) == LIMITS[limit], (
            f"{cname} is {m.group(1)} but LIMITS['{limit}'] is {LIMITS[limit]}")


# ---------- the de-ess slot count must model the whole-file REPLACE ----------
# session._apply_action replaces the agent's whole-file de-esser rather than appending, so
# re-tuning it costs no new slot — but the new slot rule counted it as one, and a chain already
# holding kMaxDeess entries refused the very re-propose that IS how a whole-file de-esser is
# adjusted. The comp twin has modelled its replacement since 2026-06 (2026-09-08 pre-merge review).

def _deess(**kw):
    from mix_engine.models import DeessAction
    kw.setdefault("freq_hz", 7000.0)
    kw.setdefault("threshold_db", -24.0)
    kw.setdefault("ratio", 4.0)
    return DeessAction(stem_id="vox", **kw)


def _full_deess_chain():
    """3 ranged + 1 whole-file = exactly LIMITS['deessers_per_stem_max']."""
    return {"vox": [{"type": "deess", "stem_id": "vox", "time_range_s": [i * 2.0, i * 2.0 + 1.0]}
                    for i in range(3)] + [{"type": "deess", "stem_id": "vox"}]}


def test_retuning_the_whole_file_deesser_at_the_slot_cap_is_allowed():
    assert _rules_multi([_deess()], _full_deess_chain()) == []


def test_a_fifth_RANGED_deesser_is_still_refused():
    """The control: the cap itself must still bind, or the fix above just deleted the rule."""
    assert "deessers_per_stem_max" in _rules_multi(
        [_deess(time_range_s=[9.0, 10.0])], _full_deess_chain())


def test_two_whole_file_deessers_in_one_proposal_still_cost_one_slot():
    """They replace each other too, so the pair is one slot — not two."""
    assert _rules_multi([_deess(), _deess(threshold_db=-20.0)], _full_deess_chain()) == []
