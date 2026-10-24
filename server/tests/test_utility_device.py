"""Ableton-style Utility device (device freedom) — a freely-placeable output/trim insert
(gain trim + stereo width + mono fold-down + polarity invert), distinct from the
channel-strip fader (the track's separate internal gain). Covers the user co-edit add/
edit/remove, the agent proposal path + guardrail, canonical ordering + the live-monitor
bridge, and the offline DSP math (which the native MonitorChain::applyUtility mirrors)."""
import math

import numpy as np
import pytest

from mix_engine import dsp
from mix_engine import guardrails
from mix_engine.models import ProposalRequest, UtilityAction
from mix_engine.session import (Session, _BUS_KINDS, _MASTER_KINDS, _STEM_KINDS,
                                 effective_chain)
from mix_engine.monitor_chain import monitor_chain_params


def _rms_db(x: np.ndarray) -> float:
    return 20.0 * math.log10(math.sqrt(float(np.mean(x ** 2))) + 1e-12)


def test_utility_user_add_edit_remove(demo_project):
    """The user hand-places a Utility on a stem AND the master, dials every param via the
    co-edit path (clamped), and removes it. mono/invert are 0/1 toggles."""
    s = Session(demo_project)
    stem = next(iter(s._state["stems"]))

    s.add_stem_device(stem, "utility")
    s.add_master_device("utility")
    assert any(e["type"] == "utility" for e in s._state["stems"][stem]["chain"])
    assert any(e["type"] == "utility" for e in s._state["master_chain"])

    # every editable param via the per-param co-edit endpoint (clamped to its range)
    s.set_master_device_param("utility", 0, "gain_db", -3.0, target=stem)
    s.set_master_device_param("utility", 0, "width", 1.6, target=stem)
    s.set_master_device_param("utility", 0, "mono", 1.0, target=stem)
    s.set_master_device_param("utility", 0, "invert", 1.0, target=stem)
    u = [e for e in s._state["stems"][stem]["chain"] if e["type"] == "utility"][0]
    assert u["gain_db"] == -3.0 and u["width"] == 1.6 and u["mono"] == 1.0 and u["invert"] == 1.0
    s.set_master_device_param("utility", 0, "gain_db", 999.0, target=stem)  # clamp to +24
    assert u["gain_db"] == 24.0

    s.remove_stem_device(stem, "utility", 0)
    assert not [e for e in s._state["stems"][stem]["chain"] if e["type"] == "utility"]


def test_utility_canonical_order_and_live_monitor(demo_project):
    """The Utility sits BEFORE band_width/width in every canonical chain (matching the native
    trailing block) and resolves to a live monitor entry — so what's arranged is what's heard."""
    s = Session(demo_project)
    stem = next(iter(s._state["stems"]))

    for kinds in (_STEM_KINDS, _MASTER_KINDS, _BUS_KINDS):
        assert "utility" in kinds
    for kinds in (_MASTER_KINDS, _BUS_KINDS):
        assert kinds.index("utility") < kinds.index("band_width") < kinds.index("width" if "width" in kinds else "clip")

    s.add_stem_device(stem, "utility")
    mc = monitor_chain_params(effective_chain(s._state["stems"][stem]))
    types = [e["type"] for e in mc["entries"]]
    assert "utility" in types          # heard live (not print-only)
    assert mc["print_only"] == []


def test_utility_agent_guardrail_and_replace_one(demo_project):
    """The agent may propose a Utility (total-agent-control); its gain trim charges the per-pass
    gain budget (a moved output stage is still a level change); one Utility per chain (replace)."""
    s = Session(demo_project)
    stem = next(iter(s._state["stems"]))

    def _check(actions):
        req = ProposalRequest(actions=actions, rationale="x", predicted_impact=[])
        return guardrails.validate_proposal(req, cumulative_gain_this_pass_db={},
                                            accepted_proposals_this_pass=0, stem_tags={})

    assert len(_check([UtilityAction(stem_id=stem, gain_db=-2.0, width=1.2)])) == 0
    big = _check([UtilityAction(stem_id=stem, gain_db=20.0)])
    assert any(v.rule.startswith("gain_per_stem") for v in big), "utility gain must be guardrailed"

    s._mutate(UtilityAction(stem_id=stem, gain_db=-2.0, width=1.2))
    s._mutate(UtilityAction(stem_id=stem, gain_db=-4.0, width=1.1))   # replaces, not stacks
    n = sum(1 for e in s._state["stems"][stem]["chain"] if e["type"] == "utility")
    assert n == 1


def test_utility_dsp_math():
    """The offline _utility (the export oracle + the native applyUtility mirror): gain trim,
    mono fold-down, polarity invert, M/S width — all on (frames, channels) float64."""
    sr = 48000
    tone = (0.5 * np.sin(2 * np.pi * 1000 * np.arange(sr) / sr)).astype(np.float64)
    stereo = np.stack([tone, tone], axis=1)

    # gain: a -6 dB trim drops RMS by exactly 6 dB
    flat = dsp._utility(stereo, 44100, {"gain_db": 0.0, "width": 1.0})
    cut = dsp._utility(stereo, 44100, {"gain_db": -6.0, "width": 1.0})
    assert abs((_rms_db(cut) - _rms_db(flat)) + 6.0) < 0.01

    # mono fold-down collapses a pure-side signal (L = -R) to silence
    side = np.stack([tone, -tone], axis=1)
    assert _rms_db(dsp._utility(side, 44100, {"mono": True})) < -120.0

    # polarity invert: x + invert(x) == 0
    inv = dsp._utility(stereo, 44100, {"invert": True})
    assert float(np.max(np.abs(stereo + inv))) < 1e-9

    # all-default is a pass-through (oracle-neutral)
    nop = dsp._utility(stereo, 44100, {"gain_db": 0.0, "width": 1.0, "mono": False, "invert": False})
    assert float(np.max(np.abs(nop - stereo))) < 1e-12


def test_utility_v2_live_parity_features():
    """Utility v2 (2026-07-05): bass mono folds only the lows, per-channel
    polarity (with the legacy-invert fallback), DC filter, balance — and the
    resolver pre-resolves the per-channel flips for the native monitor."""
    from mix_engine.monitor_chain import monitor_chain_params
    sr = 44100
    rng = np.random.default_rng(5)
    sig = rng.standard_normal((sr, 2)) * 0.2

    def side_db(a, lo, hi):
        s = np.asarray(a[:, 0] - a[:, 1], np.float64)
        spec = np.fft.rfft(s)
        f = np.fft.rfftfreq(s.size, 1.0 / sr)
        m = (f >= lo) & (f < hi)
        return 20.0 * np.log10(float(np.sqrt(np.mean(np.abs(spec[m]) ** 2))) + 1e-12)

    # bass mono: low side collapses, high side untouched
    bm = dsp._utility(sig, sr, {"bass_mono": True, "bass_mono_hz": 120.0})
    assert side_db(bm, 20, 60) < side_db(sig, 20, 60) - 6.0
    assert abs(side_db(bm, 4000, 12000) - side_db(sig, 4000, 12000)) < 0.2

    # per-channel polarity + the legacy fallback (invert=True, one side overridden)
    il = dsp._utility(sig, sr, {"invert_left": True})
    assert np.array_equal(il[:, 0], -sig[:, 0]) and np.array_equal(il[:, 1], sig[:, 1])
    lr = dsp._utility(sig, sr, {"invert": True, "invert_left": False})
    assert np.array_equal(lr[:, 0], sig[:, 0]) and np.array_equal(lr[:, 1], -sig[:, 1])

    # DC filter strips a constant offset
    dc = dsp._utility(sig + 0.1, sr, {"dc_filter": True})
    assert abs(float(np.mean(dc))) < 0.01

    # balance: linear far-channel law, near channel at unity
    b = dsp._utility(sig, sr, {"balance": 0.5})
    assert np.allclose(b[:, 0], sig[:, 0] * 0.5) and np.array_equal(b[:, 1], sig[:, 1])

    # the resolver pre-resolves the flips from the legacy flag
    out = monitor_chain_params([{"type": "utility", "stem_id": "m", "invert": True}])
    u = next(e for e in out["entries"] if e["type"] == "utility")
    assert u["invert_left"] is True and u["invert_right"] is True
    out2 = monitor_chain_params([{"type": "utility", "stem_id": "m",
                                  "invert": True, "invert_right": False}])
    u2 = next(e for e in out2["entries"] if e["type"] == "utility")
    assert u2["invert_left"] is True and u2["invert_right"] is False
    assert u2["bass_mono"] is False and u2["balance"] == 0.0   # back-compat defaults


# ---- native parity: the numpy spec MonitorChain::applyUtility transcribes -------------
# Added 2026-08-06. `utility` had no branch in reference_monitor at all, so a six-stage
# processor shipped in both shells with its native implementation never null-tested — an
# entry was silently SKIPPED, meaning any parity assertion written against it would have
# passed vacuously. These pin the spec to the engine stage by stage.

@pytest.mark.parametrize("params", [
    {},                                                   # all-default must be a pass-through
    {"gain_db": -6.0},
    {"mono": True},
    {"width": 1.6},
    {"invert_left": True},
    {"invert": True},                                     # legacy both-channel flip
    {"bass_mono": True, "bass_mono_hz": 120.0},
    {"dc_filter": True},
    {"balance": 0.5},
    {"dc_filter": True, "gain_db": -3.0, "invert_right": True, "width": 1.4,
     "bass_mono": True, "bass_mono_hz": 150.0, "balance": -0.3},   # every stage at once
])
def test_utility_reference_monitor_matches_engine(params):
    from mix_engine import reference_monitor as rm

    sr = 48000
    rng = np.random.default_rng(5)
    sig = rng.standard_normal((sr // 4, 2)) * 0.1
    engine = dsp._utility(sig.copy(), sr, params)
    spec = rm.render_monitor(sig.copy(), sr,
                             {"entries": [{"type": "utility", "enabled": True, **params}]})
    assert np.allclose(engine, spec, atol=1e-12), "utility spec diverged from the engine"


def test_utility_entry_is_not_skipped_by_the_monitor():
    """The regression that hid the gap: an unknown type falls through untouched, so a
    -12 dB Utility that changes nothing means the branch is missing again."""
    from mix_engine import reference_monitor as rm

    sr = 48000
    sig = np.ones((256, 2)) * 0.5
    out = rm.render_monitor(sig.copy(), sr,
                            {"entries": [{"type": "utility", "enabled": True,
                                          "gain_db": -12.0}]})
    assert not np.allclose(out, sig), "utility was skipped by render_monitor"
    assert np.allclose(out, sig * 10.0 ** (-12.0 / 20.0))
