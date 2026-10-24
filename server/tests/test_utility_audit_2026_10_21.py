"""The Utility's audit, 2026-10-21 (the device-by-device audit).

James's two calls: Bass Mono is a Linkwitz-Riley 8th-order split (the band width's and the
imager's law; it was a 6 dB/oct one-pole under texts that said it folded everything below the
knob), and the Utility's switches crossfade over 30 ms. The native half is `--utility-audit`.
Here:
  * the offline Utility IS the samples the native leg pins (kUtilPins) - one law, three engines;
  * every value is held to the table and every switch is ON only above 0.5, at every reader
    (`reference_monitor.utility_values`), and the brain snaps a switch to 0 / 1 when it is set;
  * on the master the engineer's Bass Mono stops at the guardrail's 300 Hz (an LR8 at 500 Hz
    takes the side at 300 Hz to 0.02 - the image, not the lows);
  * both prompts and the model say what the split does.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

from mix_engine import dsp, guardrails
from mix_engine import reference_monitor as rm
from mix_engine.models import ProposalRequest, UtilityAction
from mix_engine.session import Session

ROOT = Path(__file__).resolve().parents[2]
LEG = ROOT / "daw" / "Source" / "UtilityAuditLegs.h"

SETS = {
    0: dict(dc_filter=True, gain_db=-3.0, invert_left=True, width=1.4, bass_mono=True,
            bass_mono_hz=150.0, balance=0.3),
    1: dict(gain_db=4.0, invert=True, width=0.5, bass_mono=True, bass_mono_hz=60.0, balance=-0.6),
}


def _pins():
    text = LEG.read_text(encoding="utf-8")
    block = re.search(r"kUtilPins\[\]\s*=\s*\{(.*?)\n\s*\};", text, re.S).group(1)
    rows = re.findall(r"\{\s*(\d+),\s*(\d+),\s*(\d+),\s*([-\d.e+]+)f,\s*([-\d.e+]+)f\s*\}", block)
    assert len(rows) == 10
    return [(int(a), int(b), int(c), float(d), float(e)) for a, b, c, d, e in rows]


def _input(sr: int) -> np.ndarray:
    i = np.arange(4096)
    left = (0.4 * np.sin(2 * np.pi * 55 * i / sr) + 0.2 * np.sin(2 * np.pi * 1000 * i / sr) + 0.05)
    right = (0.3 * np.sin(2 * np.pi * 55 * i / sr + 1.1) - 0.15 * np.sin(2 * np.pi * 1700 * i / sr) - 0.03)
    return np.stack([left.astype(np.float32), right.astype(np.float32)], 1).astype(np.float64)


@pytest.mark.parametrize("engine", [rm._utility, dsp._utility])
def test_the_offline_utility_is_the_samples_the_native_leg_pins(engine):
    for s, sr, k, want_l, want_r in _pins():
        y = engine(_input(sr), sr, SETS[s])
        assert y[k, 0] == pytest.approx(want_l, abs=2e-8) and y[k, 1] == pytest.approx(want_r, abs=2e-8)


def test_bass_mono_is_an_lr8_split():
    sr, t = 48000, np.arange(96000) / 48000

    def side_left(f):
        s = 0.5 * np.sin(2 * np.pi * f * t)
        y = dsp._utility(np.stack([s, -s], 1), sr, {"bass_mono": True, "bass_mono_hz": 120.0})
        side = 0.5 * (y[48000:, 0] - y[48000:, 1])
        return 20 * math.log10(np.sqrt(np.mean(side ** 2)) / np.sqrt(np.mean(s[48000:] ** 2)))
    assert side_left(30.0) < -90.0 and side_left(60.0) < -45.0
    assert side_left(120.0) == pytest.approx(-6.0, abs=0.2) and side_left(1000.0) > -0.05


def test_every_value_is_held_and_every_switch_is_on_only_above_a_half():
    u = rm.utility_values({"gain_db": float("nan"), "width": 5, "balance": 50, "bass_mono_hz": 9,
                           "mono": 0.3, "invert_left": 0.7, "dc_filter": "x"})
    assert (u["gain_db"], u["width"], u["balance"], u["bass_mono_hz"]) == (0.0, 2.0, 1.0, 50.0)
    assert u["mono"] is False and u["flip_l"] is True and u["flip_r"] is False and u["dc_filter"] is False
    assert rm.utility_values({"invert": True})["flip_r"] is True   # unset inherits `invert`
    native = (ROOT / "native" / "Source" / "MonitorChain.cpp").read_text(encoding="utf-8")
    parse = native[native.index("auto parseUtility = [this]"):]
    parse = parse[:parse.index("\n    };\n")]
    for row in ('"gain_db", 0.0, -24.0, 24.0', '"width", 1.0, 0.0, 2.0', '"balance", 0.0, -1.0, 1.0',
                '"bass_mono_hz", 120.0, 50.0, 500.0', "d > 0.5"):
        assert row in parse, row


def test_the_brain_snaps_a_utility_switch(demo_project):
    s = Session(demo_project)
    stem = next(iter(s._state["stems"]))
    s.add_stem_device(stem, "utility")
    s.set_master_device_param("utility", 0, "mono", 0.3, target=stem)
    s.set_master_device_param("utility", 0, "bass_mono", 0.7, target=stem)
    u = [e for e in s._state["stems"][stem]["chain"] if e["type"] == "utility"][0]
    assert u["mono"] == 0.0 and u["bass_mono"] == 1.0


def _violations(action):
    req = ProposalRequest(actions=[action], rationale="x", predicted_impact=[])
    return guardrails.validate_proposal(req, cumulative_gain_this_pass_db={},
                                        accepted_proposals_this_pass=0, stem_tags={})


def test_master_bass_mono_stops_at_the_lows():
    assert not _violations(UtilityAction(stem_id="master", bass_mono=True, bass_mono_hz=300.0))
    high = _violations(UtilityAction(stem_id="master", bass_mono=True, bass_mono_hz=400.0))
    assert any(v.rule == "master_width_min" for v in high)
    assert not _violations(UtilityAction(stem_id="drums", bass_mono=True, bass_mono_hz=500.0))


def test_the_texts_say_what_bass_mono_does():
    for p in (ROOT / "orchestrator" / "chat.py", ROOT / "orchestrator" / "prompts.py"):
        text = p.read_text(encoding="utf-8").replace("\\\n", "")
        assert "48 dB/oct split" in text and "bass_mono_hz at or below 300" in text, p.name
    assert "48 dB/oct" in UtilityAction.__doc__
    manual = (ROOT / "docs" / "17-user-manual.md").read_text(encoding="utf-8")
    assert "48 dB per octave" in manual and "fade over 30 ms" in manual


def test_the_export_oracle_compares_every_utility_stage():
    """07-F5 / U6: the stereo profile compared the master Utility's gain and width only. It now
    carries a one-channel flip on a stem and, on the master, the DC block, Bass Mono at 400 Hz
    (at 150 Hz the fixture's side barely reaches it: the one-pole law failed by 0.03 dB of true
    peak; at 400 Hz by 3.7 dB of side loudness) and a balance - and the negative control has
    them all at unity."""
    from orchestrator.daw_export_oracle import _stereo_chain

    class _S:
        def __init__(self):
            self._state = {"stems": {"centre_lead": {"chain": []}, "wide_pad": {"chain": []}}}

    on, off = _S(), _S()
    _stereo_chain(on)
    _stereo_chain(off, widths=False)
    mu = next(e for e in on._state["master_chain"] if e["type"] == "utility")
    assert mu["dc_filter"] and mu["bass_mono"] and mu["bass_mono_hz"] == 400.0 and mu["balance"] == 0.25
    su = next(e for e in on._state["stems"]["centre_lead"]["chain"] if e["type"] == "utility")
    assert su["invert_left"] is True
    mo = next(e for e in off._state["master_chain"] if e["type"] == "utility")
    so = next(e for e in off._state["stems"]["centre_lead"]["chain"] if e["type"] == "utility")
    assert not mo["bass_mono"] and mo["balance"] == 0.0 and so["invert_left"] is False
