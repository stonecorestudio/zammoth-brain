"""The saturation characters' LEVEL LAW (2026-10-26, measured on a reference saturator's multiband form,
James's pick C): the reference takes the drive back out only up to a ceiling and adds a fixed gain of its
own, so a driven band keeps its level against the others. Without it, a tube / tube / heavy split read
4 dB darker above 4 kHz than the reference. Also: the multiband form has no DC blocker after its sum
(the reference has none; the tube takes its own DC out in its tone).

  * level_db holds the figures it was fitted to (the reference's level over the bare curve, by drive)
  * a band at a quiet level is lifted by its law, in the single and the multiband form
  * the guardrail charges the lift: a saturation is no longer free level
"""
from __future__ import annotations

import numpy as np
import pytest

from mix_engine import dsp, guardrails, sat_characters as sc

SR = 48000

# (curve, our drive dB, the reference's level over ours, dB) - read off the reference on a dry vocal
MEASURED = [("tanh", 8.0, 0.94), ("tanh", 14.1, 1.64), ("tanh", 20.5, 4.57), ("tanh", 26.7, 10.66),
            ("warm", 14.1, 1.56), ("warm", 25.9, 9.89),
            ("tube", 5.2, 0.97), ("tube", 14.8, 4.41), ("tube", 19.5, 8.49),
            ("hard", 14.1, 5.06), ("hard", 24.7, 7.58), ("hard", 39.2, 15.91)]


@pytest.mark.parametrize("curve,drive,want", MEASURED)
def test_the_level_law_is_the_measured_one(curve, drive, want):
    assert abs(sc.level_db(drive, curve) - want) < 0.35, (curve, drive, sc.level_db(drive, curve), want)


def _gain_db(entry, hz, level_db):
    t = np.arange(2 * SR) / SR
    x = (10 ** (level_db / 20) * np.sin(2 * np.pi * hz * t))[:, None].repeat(2, 1)
    y = dsp._saturate(x, entry, SR)[SR:, 0]
    tt = t[SR:]
    h1 = 2 * abs(np.sum(y * np.exp(-2j * np.pi * hz * tt))) / SR
    return 20 * np.log10(h1 / 10 ** (level_db / 20))


def _mb(low, mid, high):
    return {"type": "saturation", "oversample": 4, "crossover_lo_hz": 250.0, "crossover_hi_hz": 2500.0,
            "low": low, "mid": mid, "high": high}


def test_a_driven_band_keeps_its_level_against_the_others():
    # a heavy high band at 14 dB, quiet: the law lifts it ~5 dB, where the bare curve read -0.5
    e = _mb({"curve": "tanh", "drive_db": 0.0}, {"curve": "tanh", "drive_db": 0.0}, {"curve": "hard", "drive_db": 14.0})
    assert abs(_gain_db(e, 6000.0, -40.0) - 4.67) < 0.1
    # the mid band, undriven, sits at SOFT's fixed gain (600 Hz: at 1 kHz the high band's skirt, -25 dB,
    # still carries the heavy band's lift)
    assert abs(_gain_db(e, 600.0, -40.0) - 0.97) < 0.1


def test_the_multiband_form_has_no_dc_blocker():
    # a 10 Hz high-pass after the sum read -0.8 dB at 15 Hz; the reference has none
    e = _mb(*({"curve": "tanh", "drive_db": 0.0},) * 3)
    assert abs(_gain_db(e, 15.0, -40.0) - 0.97) < 0.1


def test_the_single_band_form_carries_the_law_too():
    e = {"type": "saturation", "drive_db": 6.0, "curve": "hard", "oversample": 1}
    assert abs(_gain_db(e, 1000.0, -24.0) - 3.451) < 0.05


def test_the_guardrail_charges_the_law():
    # the single form: HEAVY at its 12 dB top lifts ~4.9 dB with no trim at all
    one = {"type": "saturation", "drive_db": 12.0, "curve": "hard"}
    assert abs(guardrails.level_charge_db(one) - sc.level_db(12.0, "hard")) < 1e-9
    assert guardrails.level_charge_db(one) > 4.5
    # the multiband form: its bands in parallel take their max, a band's own trim added
    mb = _mb({"curve": "tanh", "drive_db": 0.0}, {"curve": "tanh", "drive_db": 24.0, "output_db": -2.0},
             {"curve": "tube", "drive_db": 6.0})
    assert abs(guardrails.level_charge_db(mb) - (sc.level_db(24.0, "tanh") - 2.0)) < 1e-9
    # read as the render reads it: a band drive past 24 is held there, a missing band takes the flat drive
    assert abs(sc.entry_lift_db(_mb({"drive_db": 40.0}, {}, {})) - sc.level_db(24.0, 0)) < 1e-9
    assert abs(sc.entry_lift_db({"low": {}, "drive_db": 9.0, "curve": "tube"}) - sc.level_db(9.0, "tube")) < 1e-9
