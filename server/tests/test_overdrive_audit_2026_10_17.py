"""The overdrive's audit, 2026-10-17 (the device-by-device audit) - the offline engine and the rail.

The native half is `--od-audit`. Here:
  * the tone is a high shelf that lifts the top by up to +6 dB, and that lift is charged to the
    composed EQ rail - it was charged nowhere, so drive 0 / tone 1 beside a +6 dB high shelf was
    +12 dB of treble in one pass;
  * every value is held to the table offline as the native's parse holds it;
  * the reference monitor's overdrive is the same function.
"""
from __future__ import annotations

import numpy as np

from mix_engine import dsp, reference_monitor
from mix_engine.models import ProposalRequest
from mix_engine.session import Session

SR = 48000


def test_the_tones_lift_is_charged_to_the_eq_rail(demo_project):
    session = Session(demo_project)
    shelf = {"type": "eq", "stem_id": "stem_a",
             "band": {"shape": "high_shelf", "freq_hz": 6000.0, "gain_db": 6.0, "q": 0.7}}
    bright = {"type": "overdrive", "stem_id": "stem_a", "drive_db": 0.0, "tone": 1.0}
    alone = session.propose(ProposalRequest(rationale="od", predicted_impact=[], actions=[shelf]))
    assert alone.accepted, alone.violations
    both = session.propose(ProposalRequest(rationale="od", predicted_impact=[], actions=[shelf, bright]))
    assert not both.accepted
    assert any("eq" in v.rule for v in both.violations), both.violations


def test_every_value_is_held_to_the_table():
    t = np.arange(4800) / SR
    x = np.stack([0.3 * np.sin(2 * np.pi * 220 * t)] * 2, axis=1)
    at_end = dsp._overdrive(x, SR, {"drive_db": 36.0, "filter_on": True, "freq_hz": 12000.0, "q": 8.0,
                                    "output_db": 12.0, "mix": 1.0})
    past = dsp._overdrive(x, SR, {"drive_db": 60.0, "filter_on": True, "freq_hz": 30000.0, "q": 20.0,
                                  "output_db": 30.0, "mix": 1.5})
    assert np.max(np.abs(np.asarray(past) - np.asarray(at_end))) < 1e-9


def test_the_reference_monitors_overdrive_is_the_same_function():
    t = np.arange(4800) / SR
    x = np.stack([0.3 * np.sin(2 * np.pi * 220 * t)] * 2, axis=1)
    e = {"drive_db": 18.0, "filter_on": True, "freq_hz": 1500.0, "q": 1.0, "tone": 0.7,
         "output_db": 3.0, "mix": 0.5, "oversample": 1}
    a = dsp._overdrive(x, SR, e)
    b = reference_monitor._overdrive(x, SR, e["drive_db"], True, 1500.0, 1.0, 0.7, 3.0, 0.5, 1)
    assert np.max(np.abs(np.asarray(a) - np.asarray(b))) < 1e-9
