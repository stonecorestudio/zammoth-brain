"""The pre-merge review of 2026-10-26 (betaV18 -> main), the brain's half: each test is one finding,
written to fail on the code before its fix. The native half is `MixAgentDAW --premerge-1026`."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from mix_engine import dsp, guardrails, monitor_chain, od_characters as odc
from mix_engine.models import ProposalRequest
from mix_engine.session import Session

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "test_projects" / "two_stem_demo"
SR = 48000


@pytest.fixture
def project(tmp_path):
    dst = tmp_path / "p"
    shutil.copytree(DEMO, dst, ignore=shutil.ignore_patterns("runs"))
    return dst


def _req(*actions) -> ProposalRequest:
    return ProposalRequest.model_validate(
        {"pass_id": 1, "rationale": "t", "predicted_impact": [], "actions": list(actions)})


def _master(project, chain) -> Session:
    s = Session(project)
    s._state["master_chain"] = chain
    s._persist()
    return Session(project)


# ---- blocker 1: an `instance` replace past the user's master limiter ---------------------------

def test_an_instance_replace_past_the_master_limiter_is_refused(project):
    s = _master(project, [{"type": "limiter", "ceiling_dbtp": -1.0, "style": 0, "owner": "user"},
                          {"type": "utility", "gain_db": 0.0, "width": 1.0, "owner": "user"}])
    r = s.propose(_req({"type": "utility", "stem_id": "master", "instance": 0, "gain_db": 5.0}))
    assert not r.accepted and "post_limiter" in [v.rule for v in r.violations]


def test_without_instance_the_move_still_lands_in_front_of_the_limiter(project):
    s = _master(project, [{"type": "limiter", "ceiling_dbtp": -1.0, "style": 0, "owner": "user"},
                          {"type": "utility", "gain_db": 0.0, "width": 1.0, "owner": "user"}])
    r = s.propose(_req({"type": "utility", "stem_id": "master", "gain_db": 5.0}))
    assert r.accepted
    s.apply(r.proposal_id)
    assert [e["type"] for e in Session(project)._state["master_chain"]] == ["utility", "limiter"]


# ---- blocker 2: a multiband saturation's flat trim cancelled its bands' charge -----------------

def test_a_flat_trim_no_engine_plays_does_not_cancel_the_bands_charge():
    bands = {k: {"drive_db": 6.0, "curve": 0, "output_db": 12.0, "mix": 1.0} for k in ("low", "mid", "high")}
    e = {"type": "saturation", "drive_db": 3.0, "output_db": -12.0, "oversample": 4,
         "crossover_lo_hz": 200.0, "crossover_hi_hz": 2500.0, **bands}
    t = np.arange(SR) / SR
    x = np.stack([0.05 * np.sin(2 * np.pi * 440 * t)] * 2, axis=1)
    lift = 20 * np.log10(np.sqrt(np.mean(dsp._saturate(x.copy(), e, SR) ** 2)) / np.sqrt(np.mean(x ** 2)))
    assert lift > 11.0                                   # what plays: the bands' +12
    assert guardrails.level_charge_db(e) >= lift - 1.0   # what is charged


# ---- `instance` numbers are the chain as the proposal found it ---------------------------------

def test_instance_numbers_are_pinned_before_a_new_one_lands(project):
    s = Session(project)
    sid = next(iter(s.stems))
    s.stems[sid]["chain"] = [{"type": "comp", "threshold_db": -20, "ratio": 2, "attack_ms": 10, "release_ms": 100, "makeup_db": 0},
                             {"type": "saturation", "drive_db": 3.0, "curve": 0},
                             {"type": "comp", "threshold_db": -18, "ratio": 3, "attack_ms": 10, "release_ms": 100, "makeup_db": 0}]
    s._persist()
    s = Session(project)
    uid_second = [e for e in s.stems[sid]["chain"] if e["type"] == "comp"][1]["uid"]
    r = s.propose(_req(
        {"type": "comp", "stem_id": sid, "instance": "new", "threshold_db": -30, "ratio": 2, "attack_ms": 10, "release_ms": 100, "makeup_db": 0},
        {"type": "comp", "stem_id": sid, "instance": 1, "threshold_db": -9, "ratio": 2, "attack_ms": 10, "release_ms": 100, "makeup_db": 0}))
    assert r.accepted, r.violations
    s.apply(r.proposal_id)
    comps = [e for e in Session(project).stems[sid]["chain"] if e["type"] == "comp"]
    by_uid = {e["uid"]: e for e in comps}
    assert by_uid[uid_second]["threshold_db"] == -9          # the comp the proposal named
    assert any(e["threshold_db"] == -30 for e in comps)     # the new one is there too
    assert len(comps) == 3


# ---- the delay's Ping-Pong is a switch at every door ------------------------------------------

def test_ping_pong_is_snapped_and_pushed_as_the_renders_read_it(project):
    s = Session(project)
    sid = next(iter(s.stems))
    s.stems[sid]["chain"] = [{"type": "return_delay", "delay_ms": 300.0, "feedback": 0.3, "tone": 0.5, "mix": 0.3}]
    s._persist()
    s = Session(project)
    s.set_master_device_param("return_delay", 0, "ping_pong", 0.4, target=sid)
    assert Session(project).stems[sid]["chain"][0]["ping_pong"] == 0.0
    pushed = monitor_chain.monitor_chain_params([{"type": "return_delay", "delay_ms": 300.0, "feedback": 0.3, "tone": 0.5, "ping_pong": 0.4, "mix": 0.3}])
    entry = next(e for e in pushed["entries"] if e.get("type") == "return_delay")
    assert entry["ping_pong"] == 0.0                     # >= 0.5 is on, in every engine


# ---- what an amp character and a tape / tube tone lift is charged ------------------------------

@pytest.mark.parametrize("c", [1, 2, 3])
def test_an_amps_voicing_is_on_the_eq_rail(c):
    boost, _ = guardrails.eq_family_curves_db({"type": "overdrive", "character": c})
    assert float(np.max(boost)) > 5.0


@pytest.mark.parametrize("curve", ["warm", "tube"])
def test_a_tape_or_tube_tone_is_on_the_eq_rail(curve):
    boost, _ = guardrails.eq_family_curves_db({"type": "saturation", "curve": curve, "drive_db": 3.0})
    assert float(np.max(boost)) > 1.0


def test_an_amps_level_law_is_charged_to_the_gain_wallet():
    assert guardrails.level_charge_db({"type": "overdrive", "character": "crunchy", "drive_db": 12.0}) > 6.0
    assert guardrails.level_charge_db({"type": "overdrive", "character": "pedal", "drive_db": 12.0}) == 0.0


# ---- the amps keep their level, and their voicing, at every rate -------------------------------

def _noise(sr, secs=2.0, db=-12.0):
    rng = np.random.default_rng(3)
    x = rng.standard_normal(int(sr * secs)) * 10 ** (db / 20)
    return np.stack([x, x], axis=1)


@pytest.mark.parametrize("c,floor_db", [(1, -4.0), (2, -16.0), (3, -7.5)])
def test_an_amp_keeps_its_level_as_it_is_driven(c, floor_db):
    """From drive 0 to 24, noise at -12 dBFS: the reference's amps fall about 1 / 10 / 6 dB (Smooth /
    Crunchy / Power, on a full mix); ours fell 5.2 / 21.2 / 9.1 without the law and 1.8 / 13 / 5 with it."""
    x = _noise(SR)
    lev = lambda d: 20 * np.log10(np.sqrt(np.mean(dsp._overdrive(x.copy(), SR, {"drive_db": d, "character": c, "oversample": 1}) ** 2)))
    assert lev(24.0) - lev(0.0) > floor_db


@pytest.mark.parametrize("c", [1, 2, 3])
@pytest.mark.parametrize("sr", [44100, 96000, 192000])
def test_an_amps_filters_answer_alike_at_every_rate(c, sr):
    f = np.geomspace(100.0, 16000.0, 120)
    def db(rate, which):
        secs = (odc.pre_sections if which == "pre" else odc.post_sections)(c, rate)
        return 20 * np.log10(np.abs(odc._cascade_response(secs, rate, f)) + 1e-12)
    for which in ("pre", "post"):
        ref = db(48000, which)
        keep = ref > ref.max() - 45.0
        assert np.max(np.abs(db(sr, which) - ref)[keep]) < 1.5, which


def test_the_curve_inputs_are_held_where_float_cannot_overflow():
    from mix_engine import sat_characters as sc
    assert sc.U_CLAMP <= 1.0e6 and odc.U_CLAMP <= 1.0e6
    hdr_sat = (ROOT / "native/Source/SatCharacters.h").read_text()
    hdr_od = (ROOT / "native/Source/OdCharacters.h").read_text()
    assert "kUClamp" in hdr_sat and "std::fmin (kUClamp, u)" in hdr_sat
    assert "kUClamp" in hdr_od and "std::fmin (kUClamp, u)" in hdr_od


# ---- a duplicate's devices after a rack name the copy's rack -----------------------------------

def test_a_copy_names_the_copys_plugins_in_after(project):
    s = Session(project)
    src, dest = list(s.stems)[:2]
    s.stems[src]["chain"] = [{"type": "limiter", "ceiling_dbtp": -1.0, "style": 0, "after": "1001"}]
    s._persist()
    Session(project).copy_stem_state(src, dest, {"1001": "2002"})
    assert Session(project).stems[dest]["chain"][0]["after"] == "2002"


def test_the_multiband_compressors_fallbacks_are_its_defaults():
    pushed = monitor_chain.monitor_chain_params([{"type": "mb_comp",
        "low": {"threshold_db": -24, "ratio": 2}, "mid": {"threshold_db": -24, "ratio": 2},
        "high": {"threshold_db": -24, "ratio": 2}}])
    e = next(x for x in pushed["entries"] if x.get("type") == "mb_comp")
    assert (e["attack_ms"], e["release_ms"]) == (2.0, 85.0)
