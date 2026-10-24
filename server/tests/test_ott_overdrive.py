"""OTT + Overdrive (2026-07-07): the 3-band upward+downward compressor and the
drive-pedal distortion — model contracts, DSP behavior, session registration,
guardrails, the monitor contract, and the reference-monitor tier.

Reference-tier note (the imager gotcha): the native/oracle LR4 split keeps a
low-frequency phase residual vs the offline exactly-complementary FFT split, so
OTT reference tests assert DIRECTION only, never a null. Overdrive shares the
offline biquad family coefficient-for-coefficient, so it nulls tightly.
"""
import numpy as np
import pytest
from pydantic import ValidationError

from mix_engine import dsp
from mix_engine import reference_monitor as rm
from mix_engine.models import OttAction, OverdriveAction, ProposalRequest
from mix_engine.monitor_chain import MONITOR_TYPES, monitor_chain_params
from mix_engine.session import (BUS_ACTION_TYPES, MASTER_ACTION_TYPES,
                                SUPPORTED_ACTION_TYPES, Session)

SR = 48000


def _apply(session: Session, actions: list[dict]):
    result = session.propose(ProposalRequest(
        rationale="ott/overdrive", predicted_impact=[], actions=actions))
    assert result.accepted, result.violations
    assert session.apply(result.proposal_id).ok
    return result


def _quiet_loud(seconds=2.0, quiet=0.01, loud=0.5, seed=0):
    """(channels, frames) float32 noise: a quiet first half, a loud second half
    — well under / over dsp.OTT_TARGET_DB so both sides of the law engage."""
    rng = np.random.default_rng(seed)
    n = int(SR * seconds)
    x = rng.standard_normal((2, n))
    x[:, : n // 2] *= quiet
    x[:, n // 2:] *= loud
    return np.ascontiguousarray(x.astype(np.float32))


def _rms(a):
    return float(np.sqrt(np.mean(np.asarray(a, dtype=np.float64) ** 2)))


def _ott_entry(**over) -> dict:
    e = {"type": "ott", "depth": 1.0, "time": 1.0,
         "low": {"up": 1.0, "down": 1.0, "gain_db": 0.0},
         "mid": {"up": 1.0, "down": 1.0, "gain_db": 0.0},
         "high": {"up": 1.0, "down": 1.0, "gain_db": 0.0}}
    e.update(over)
    return e


# ---------------------------------------------------------------- OTT: DSP ----

def test_ott_lifts_quiet_and_tames_loud():
    x = _quiet_loud()
    out = dsp._ott(x, SR, _ott_entry())
    n = x.shape[1]
    settle = int(0.4 * SR)                      # skip the ballistics settling
    q = slice(settle, n // 2 - settle)
    l = slice(n // 2 + settle, n - settle)
    assert _rms(out[:, q]) > _rms(x[:, q]) * 1.5, "quiet half should be lifted"
    assert _rms(out[:, l]) < _rms(x[:, l]) * 0.7, "loud half should be tamed"


def _split_alone(x):
    """What the real 3-way LR4 split (2026-10-17, the DAW's) gives back with every band at unity:
    an ALLPASS of the input - flat in magnitude, not the same samples."""
    return sum(dsp._lr4_three_way(x.astype(np.float64), SR, 150.0, 2000.0))


def test_ott_depth_zero_is_transparent():
    """depth 0 = unity band gains: the split alone, magnitude-flat."""
    x = _quiet_loud(seed=1)
    out = dsp._ott(x, SR, _ott_entry(depth=0.0))
    assert np.allclose(out, _split_alone(x), atol=1e-5)
    assert abs(_rms(out) / _rms(x) - 1.0) < 1e-3


def test_ott_band_sides_disable():
    """up/down 0 on every band = no levelling at all → the split alone."""
    x = _quiet_loud(seed=2)
    off = {"up": 0.0, "down": 0.0, "gain_db": 0.0}
    out = dsp._ott(x, SR, _ott_entry(low=dict(off), mid=dict(off), high=dict(off)))
    assert np.allclose(out, _split_alone(x), atol=1e-5)


def test_ott_gains_bracket():
    """in/out gains bracket the device; band gain_db is a post gain."""
    x = _quiet_loud(seed=3)
    base = dsp._ott(x, SR, _ott_entry())
    up6 = dsp._ott(x, SR, _ott_entry(out_gain_db=6.0))
    assert np.allclose(up6, base * 10.0 ** (6.0 / 20.0), atol=1e-6)


# ------------------------------------------------------- OTT: model + session ----

def test_ott_model_validation():
    with pytest.raises(ValidationError):
        OttAction(stem_id="stem_a", crossover_lo_hz=3000.0, crossover_hi_hz=2000.0)
    with pytest.raises(ValidationError):
        OttAction(stem_id="stem_a", depth=1.5)
    a = OttAction(stem_id="stem_a")             # defaults are a valid device
    assert a.low.up == 1.0 and a.mid.down == 1.0 and a.depth == 0.5


def test_ott_registration():
    assert "ott" in SUPPORTED_ACTION_TYPES
    assert "ott" in BUS_ACTION_TYPES
    assert "ott" not in MASTER_ACTION_TYPES     # master OTT is user-add only
    assert "ott" in MONITOR_TYPES
    assert "depth" in Session.DEVICE_PARAMS["ott"]
    assert Session.MASTER_PARAM["ott"][0] == "depth"


def test_ott_applies_renders_and_replaces(demo_project):
    session = Session(demo_project)
    before, _ = dsp.render_stem(session, "stem_a")
    _apply(session, [{"type": "ott", "stem_id": "stem_a", "depth": 0.25}])   # the run's cap (2026-10-05)
    otts = [e for e in session.applied_chain("stem_a") if e["type"] == "ott"]
    assert len(otts) == 1 and otts[0]["depth"] == 0.25
    dsp.RENDER_CACHE.clear()
    after, _ = dsp.render_stem(session, "stem_a")
    assert np.isfinite(after).all()
    assert not np.allclose(before, after)
    # one per chain: a new OTT replaces the old (re-proposing adjusts it)
    _apply(session, [{"type": "ott", "stem_id": "stem_a", "depth": 0.15}])
    otts = [e for e in session.applied_chain("stem_a") if e["type"] == "ott"]
    assert len(otts) == 1 and otts[0]["depth"] == 0.15


def test_ott_guardrails(demo_project):
    session = Session(demo_project)
    deep = session.propose(ProposalRequest(
        rationale="too deep", predicted_impact=[],
        actions=[{"type": "ott", "stem_id": "stem_a", "depth": 0.95}]))
    assert not deep.accepted
    assert any(v.rule == "ott_depth_max" for v in deep.violations)
    on_master = session.propose(ProposalRequest(
        rationale="master ott", predicted_impact=[],
        actions=[{"type": "ott", "stem_id": "master", "depth": 0.2}]))   # inside the depth cap: the MASTER rule refuses it
    assert not on_master.accepted
    assert "ott_depth_max" not in {v.rule for v in on_master.violations}


def test_ott_user_add_on_master(demo_project):
    """The user's hands can place an OTT on the master (guardrails bind the
    agent only — the shared-control rule)."""
    session = Session(demo_project)
    res = session.add_master_device("ott")
    assert res["type"] == "ott"
    assert any(e["type"] == "ott" for e in session.master_chain)


def test_ott_monitor_contract():
    entry = OttAction(stem_id="stem_a", depth=0.7, time=0.5, in_gain_db=2.0,
                      out_gain_db=-1.0).model_dump(exclude={"stem_id"})
    params = monitor_chain_params([entry])
    assert params["entries"] == [{
        "type": "ott", "depth": 0.7, "time": 0.5,
        "in_gain_db": 2.0, "out_gain_db": -1.0,
        "low": {"up": 1.0, "down": 1.0, "gain_db": 0.0},
        "mid": {"up": 1.0, "down": 1.0, "gain_db": 0.0},
        "high": {"up": 1.0, "down": 1.0, "gain_db": 0.0},
        "crossover_lo_hz": 150.0, "crossover_hi_hz": 2000.0,
        "enabled": True}]


def test_ott_reference_direction():
    """Reference tier (LR4 split + per-sample envelope): direction only, never
    a null against the offline FFT/block path (the imager gotcha)."""
    x = _quiet_loud(seed=4).T.astype(np.float64)    # (frames, channels)
    band = {"up": 1.0, "down": 1.0, "gain_db": 0.0}
    out = rm._ott(x, SR, 1.0, 1.0, 0.0, 0.0, dict(band), dict(band), dict(band))
    n = x.shape[0]
    settle = int(0.4 * SR)
    q = slice(settle, n // 2 - settle)
    l = slice(n // 2 + settle, n - settle)
    assert _rms(out[q]) > _rms(x[q])
    assert _rms(out[l]) < _rms(x[l])


# ------------------------------------------------------------- Overdrive ----

def test_overdrive_shapes_peaks_nonlinearly():
    t = np.arange(SR) / SR
    sine = (0.9 * np.sin(2 * np.pi * 220.0 * t))[:, np.newaxis]
    out = dsp._overdrive(sine, SR, {"drive_db": 24.0})
    assert float(np.max(np.abs(out))) < float(np.max(np.abs(sine)))
    # nonlinear: odd harmonics appear (energy at 660 Hz where the input has none)
    spec_in = np.abs(np.fft.rfft(sine[:, 0]))
    spec_out = np.abs(np.fft.rfft(out[:, 0]))
    h3 = 660 * SR // SR  # bin at 660 Hz for a 1 s window
    assert spec_out[h3] > spec_in[h3] * 10.0


def test_overdrive_mix_and_tone():
    rng = np.random.default_rng(5)
    x = (rng.standard_normal((SR, 2)) * 0.3)
    dry = dsp._overdrive(x, SR, {"drive_db": 18.0, "mix": 0.0})
    assert np.allclose(dry, x)                        # mix 0 = bypass
    half = dsp._overdrive(x, SR, {"drive_db": 18.0, "mix": 0.5})
    wet = dsp._overdrive(x, SR, {"drive_db": 18.0})
    assert np.allclose(half, 0.5 * wet + 0.5 * x)
    # tone: 1 keeps more HF energy than 0 (tilt around OD_TONE_HZ)
    bright = dsp._overdrive(x, SR, {"drive_db": 18.0, "tone": 1.0})
    dark = dsp._overdrive(x, SR, {"drive_db": 18.0, "tone": 0.0})
    hf = lambda a: _rms(np.diff(np.asarray(a), axis=0))   # differencing ~ HF emphasis
    assert hf(bright) > hf(dark)


def test_overdrive_filter_focuses_the_drive():
    rng = np.random.default_rng(6)
    x = (rng.standard_normal((SR, 2)) * 0.3)
    focused = dsp._overdrive(x, SR, {"drive_db": 12.0, "filter_on": True,
                                     "freq_hz": 1000.0, "q": 2.0})
    open_ = dsp._overdrive(x, SR, {"drive_db": 12.0})
    # the band-pass wet path carries far less out-of-band energy
    assert _rms(focused) < _rms(open_) * 0.8


def test_overdrive_dsp_matches_reference():
    """Same biquad family + memoryless shaper on both sides — a tight null
    (the de-ess-split tier), unlike the OTT's LR4-vs-FFT split."""
    rng = np.random.default_rng(7)
    x = (rng.standard_normal((SR, 2)) * 0.3)
    entry = {"drive_db": 18.0, "filter_on": True, "freq_hz": 900.0, "q": 1.5,
             "tone": 0.8, "output_db": -2.0, "mix": 0.7}
    a = dsp._overdrive(x, SR, entry)
    b = rm._overdrive(x, SR, entry["drive_db"], entry["filter_on"],
                      entry["freq_hz"], entry["q"], entry["tone"],
                      entry["output_db"], entry["mix"])
    peak = float(np.max(np.abs(a))) or 1.0
    err_db = 20.0 * np.log10(max(float(np.max(np.abs(a - b))), 1e-20) / peak)
    assert err_db < -80.0, f"overdrive oracle null too loose: {err_db:.1f} dB"


def test_overdrive_model_and_registration():
    with pytest.raises(ValidationError):
        OverdriveAction(stem_id="stem_a", drive_db=50.0)
    assert "overdrive" in SUPPORTED_ACTION_TYPES
    assert "overdrive" in BUS_ACTION_TYPES
    assert "overdrive" not in MASTER_ACTION_TYPES   # master overdrive = user-add
    assert "overdrive" in MONITOR_TYPES
    assert "drive_db" in Session.DEVICE_PARAMS["overdrive"]
    assert Session.MASTER_PARAM["overdrive"][0] == "drive_db"


def test_overdrive_applies_and_renders(demo_project):
    session = Session(demo_project)
    before, _ = dsp.render_stem(session, "stem_a")
    _apply(session, [{"type": "overdrive", "stem_id": "stem_a",
                      "drive_db": 18.0, "tone": 0.7}])
    ods = [e for e in session.applied_chain("stem_a") if e["type"] == "overdrive"]
    assert len(ods) == 1 and ods[0]["drive_db"] == 18.0
    dsp.RENDER_CACHE.clear()
    after, _ = dsp.render_stem(session, "stem_a")
    assert np.isfinite(after).all()
    assert not np.allclose(before, after)
    # one per chain (the native ChainState hosts a single overdrive step, like
    # OTT): a new overdrive replaces the old — re-proposing adjusts it
    _apply(session, [{"type": "overdrive", "stem_id": "stem_a",
                      "drive_db": 9.0}])
    ods = [e for e in session.applied_chain("stem_a") if e["type"] == "overdrive"]
    assert len(ods) == 1 and ods[0]["drive_db"] == 9.0


def test_overdrive_guardrail(demo_project):
    session = Session(demo_project)
    hot = session.propose(ProposalRequest(
        rationale="too hot", predicted_impact=[],
        actions=[{"type": "overdrive", "stem_id": "stem_a", "drive_db": 30.0}]))
    assert not hot.accepted
    assert any(v.rule == "overdrive_drive_max_db" for v in hot.violations)


def test_overdrive_monitor_contract():
    entry = OverdriveAction(stem_id="stem_a", drive_db=15.0, filter_on=True,
                            freq_hz=1200.0, q=2.0, tone=0.6,
                            output_db=3.0, mix=0.9).model_dump(exclude={"stem_id"})
    params = monitor_chain_params([entry])
    assert params["entries"] == [{
        "type": "overdrive", "drive_db": 15.0, "filter_on": True,
        "freq_hz": 1200.0, "q": 2.0, "tone": 0.6, "output_db": 3.0,
        # oversample 1 = off, the exact elementwise path (2026-09-06)
        "mix": 0.9, "oversample": 1, "enabled": True}]
