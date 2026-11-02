"""A legal EQ must not build a resonator.

2026-09-08 device audit, 04-eq-tone F10. `DEVICE_PARAMS["eq"]["q"]` allows 0.3..8.0
and `EQ_SLOPES` allows up to 96 dB/oct. They are independent clamps, and the cascade
scales EVERY section's Q by `q / BUTTERWORTH_Q` — so the resonant peak compounds with
the order. Two knob moves, both inside their published ranges, both reachable from
the device dock and from `device.set`, measured on broadband noise:

    lpf 1 kHz q=8 slope=24 : +12.6 dB
    lpf 1 kHz q=8 slope=48 : +49.4 dB
    lpf 1 kHz q=8 slope=96 : +127.3 dB   (peak 5.7e+05 from a 0.246 input)

`guardrails.LIMITS["eq_resonance_max_db"]` catches this inside `validate_proposal`
ONLY — the offline proposal path. The DAW's agent channel is `device.set`, which
consults no guardrail, and the user's own knobs consult only `DEVICE_PARAMS`. Only
the master safety limiter stood between that and the monitor; a soloed track or a
bus tap had nothing at all, and the EQ face drew it as a flat line because the curve
is clamped to +/-12 dB for display.

So the PAIR is bound where the coefficients are built - the one place every write path
reaches. Since 2026-10-27 that is the EQ's law (`eq_law`, `EqLaw.h`), measured on a
reference EQ: q shapes ONLY the cascade's most resonant section (q 1 = Butterworth), so a
steep cut can resonate as a 12 dB/oct one can, and the resonance is one section's - it
never compounds with the order.
"""
from __future__ import annotations

import numpy as np
import pytest

from mix_engine.dsp import _eq_band
from mix_engine.eq_law import band_sections
from scipy.signal import freqz

BUTTERWORTH_Q = 1.0   # the EQ law's flat cut

SR = 48000


def _noise() -> np.ndarray:
    rng = np.random.default_rng(20260908)
    return (rng.standard_normal((2, SR)) * 0.08).astype(np.float32)


def _peak_db(shape: str, q: float, slope: int) -> float:
    x = _noise()
    y = _eq_band(x, SR, {"shape": shape, "freq_hz": 1000.0, "gain_db": 0.0,
                         "q": q, "slope_db_oct": slope})
    return 20.0 * np.log10(float(np.abs(y).max()) / float(np.abs(x).max()))


@pytest.mark.parametrize("shape", ["lpf", "hpf"])
@pytest.mark.parametrize("slope", [24, 48, 96])
def test_a_legal_q_cannot_detonate_a_cascade(shape, slope):
    got = _peak_db(shape, 8.0, slope)
    assert got < 6.0, (
        f"{shape} q=8 slope={slope} peaks {got:+.1f} dB above the input "
        "(it measured +127.3 dB at slope 96 before the bound)"
    )


@pytest.mark.parametrize("slope", [24, 48, 96])
def test_above_12_db_per_oct_q_shapes_one_section_only(slope):
    """q acts on a steep cut (the reference EQ's law), and the cascade's peak is never more
    than its most resonant section's alone: the resonance does not compound."""
    hot = _peak_db("lpf", 8.0, slope)
    flat = _peak_db("lpf", BUTTERWORTH_Q, slope)
    assert hot > flat + 1.0, f"q changes nothing at slope {slope} ({hot} vs {flat})"
    f = np.geomspace(200, 4000, 4000)
    H = np.ones(len(f), complex)
    for b, a in band_sections("lpf", SR, 1000.0, 8.0, 0.0, slope):
        H *= freqz(b, a, f, fs=SR)[1]
    order = slope // 6
    top_q = min(8.0 / (2 * np.cos((order // 2 - 1 + 0.5) * np.pi / order)), 60.0)
    one_section = 20 * np.log10(top_q / np.sqrt(1 - 1 / (4 * top_q * top_q)))
    assert 20 * np.log10(np.abs(H).max()) < one_section + 1.0


def test_a_single_section_still_resonates():
    """NEGATIVE CONTROL, and the reason the bound is on the PAIR rather than on q.

    A bound that simply disabled q everywhere would satisfy every assertion above and
    would make a resonant 12 dB/oct filter — a musical sound, and what the knob is for
    — inert. One section cannot compound, so q keeps shaping it.
    """
    res = _peak_db("lpf", 8.0, 12)
    flat = _peak_db("lpf", BUTTERWORTH_Q, 12)
    assert res > flat + 6.0, (
        f"q no longer shapes a 12 dB/oct filter ({res:+.1f} vs {flat:+.1f} dB) — "
        "the bound was applied too widely"
    )
    # …and the 6 dB/oct legacy path never had a cascade to compound in the first place.
    assert len(band_sections("lpf", SR, 1000.0, 8.0, 0.0, 6)) == 1


def test_the_measurement_can_see_a_resonance_at_all():
    """Non-vacuity: if `_eq_band` were returning silence or a pass-through, every
    assertion here would pass. A plain resonant bell must register."""
    got = _peak_db("peak", 8.0, 6)
    assert got > -3.0, "a Q=8 bell at unity gain should not attenuate the peak much"
    quiet = _peak_db("lpf", BUTTERWORTH_Q, 96)
    assert quiet < -6.0, "a 96 dB/oct lowpass at 1 kHz must clearly cut broadband noise"
