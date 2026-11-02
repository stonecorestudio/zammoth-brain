"""The saturation characters (2026-10-26, sat_characters.py): four curves fitted to a reference saturator.

The native half is `--sat-characters` (the same four characters out of a real MonitorChain against
this module) and the export oracle's saturation profiles. Here:
  * the header the native reads is this module's table, written by `--write-header`;
  * each character's curve keeps the reference's measured shape - its gain against level at its
    lowest drive, the small-signal gain divided out (the points below were read off the reference,
    sample by sample on a 20 Hz sine) - to 0.5 dB (the level law is test_sat_level_law.py's);
  * what tells each one apart: the tube's second harmonic grows with drive, the heavy curve is
    nonlinear at levels where the soft one is straight, the tape's tone lifts the lows near 45 Hz;
  * every character sits at its level law at low level, bounded, odd or (the tube) DC-free once its tone ran.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mix_engine import dsp, reference_monitor, sat_characters as sc

REPO = Path(__file__).resolve().parents[2]
SR = 48000


def test_the_native_header_is_this_table():
    assert (REPO / "native/Source/SatCharacters.h").read_text() == sc.header_text()
    assert sc.NAMES == dsp.SAT_TYPES
    from mix_engine import models
    assert tuple(models.SAT_CURVES) == sc.NAMES


# (x, y) read off the reference at its lowest drive: its gentle and heavy styles, and the small-signal
# gain the fit divides out (+0.96 / +4.35 dB: the heavy curve already bends at -40 dBFS, so it is the
# fitted gain, not the first point). Ours carries its level law (+0.97 / +4.20 at drive 0, fitted on
# music across the whole range), divided out here the same way: this is a test of the SHAPE.
GAIN_DB = {0: 0.96, 3: 4.35}
# each curve's fit error at this drive (the heavy one is fitted across its whole range: 0.16 dB RMS)
TOL_DB = {0: 0.5, 3: 0.75}
REFERENCE = {
    0: [(0.01, 0.0112), (0.1, 0.1121), (0.3, 0.3302), (1.0, 0.8679), (3.0, 1.0)],
    3: [(0.01, 0.0160), (0.1, 0.1494), (0.3, 0.3839), (1.0, 0.7929), (3.0, 0.9972)],
}


@pytest.mark.parametrize("curve", sorted(REFERENCE))
def test_a_character_keeps_the_references_shape(curve):
    g = 10 ** ((GAIN_DB[curve] - sc.level_db(0.0, curve)) / 20)
    for x, y in REFERENCE[curve]:
        ours = float(sc.shape(np.array([x]), 0.0, curve)[0])
        assert abs(20 * np.log10(ours * g / y)) < TOL_DB[curve], (curve, x, ours * g, y)


def _harmonics(y: np.ndarray, n: int) -> np.ndarray:
    c = np.fft.rfft(y) / (len(y) / 2)
    return np.abs(c[[1, 2, 3]])


def _tone_db(curve, drive_db: float, level_db: float) -> tuple[float, float]:
    """H2 and H3 against the fundamental, of the curve alone, for a sine at `level_db`."""
    th = 2 * np.pi * np.arange(4096) / 4096
    y = sc.shape(10 ** (level_db / 20) * np.sin(th), drive_db, curve)
    h1, h2, h3 = _harmonics(y, 4096)
    return 20 * np.log10(h2 / h1 + 1e-15), 20 * np.log10(h3 / h1 + 1e-15)


def test_the_tube_makes_a_second_harmonic_that_grows_with_drive():
    # the reference's warm tube at -18 dBFS: H2 -33 dB at 25 %, -26 dB at 50 % (our 3.4 and 9.0 dB)
    h2_light, _ = _tone_db("tube", 3.4, -18.0)
    h2_pushed, _ = _tone_db("tube", 9.0, -18.0)
    assert abs(h2_light - (-33.0)) < 2.5 and abs(h2_pushed - (-26.0)) < 2.5, (h2_light, h2_pushed)
    for c in ("tanh", "warm", "hard"):                 # the others are odd: no even harmonic at all
        assert _tone_db(c, 9.0, -6.0)[0] < -120.0


def test_heavy_is_nonlinear_where_smooth_is_straight():
    assert _tone_db("hard", 3.0, -24.0)[1] > -40.0     # dense odd harmonics at a quiet level
    assert _tone_db("tanh", 3.0, -24.0)[1] < -70.0


@pytest.mark.parametrize("curve", sc.NAMES)
def test_a_character_sits_at_its_level_law_at_low_level_and_is_bounded(curve):
    x = np.array([1e-4])
    for d in (1.0, 6.0, 12.0, 24.0):
        assert abs(20 * np.log10(float(sc.shape(x, d, curve)[0]) / 1e-4) - sc.level_db(d, curve)) < 0.03
        big = sc.shape(np.array([-1e6, -8.0, 8.0, 1e6]), d, curve)
        assert np.all(np.isfinite(big)) and np.max(np.abs(big)) <= 1.34 * 10 ** (sc.level_db(d, curve) / 20) + 1e-9


def test_the_tape_tone_lifts_the_lows_and_cuts_the_rumble():
    def gain_db(hz):
        t = np.arange(SR * 2) / SR
        x = 1e-3 * np.sin(2 * np.pi * hz * t)[:, None]
        y = dsp._saturate(x.copy(), {"drive_db": 1.0, "curve": "warm", "oversample": 1}, SR)
        h = slice(SR, 2 * SR)
        return 10 * np.log10(np.sum(y[h, 0] ** 2) / np.sum(x[h, 0] ** 2))
    # measured on the reference: +2.1 at 45 Hz, +0.3 at 400 Hz, -3.6 at 17 Hz (relative to its gentle
    # style - so the level law, which the gentle style has too, is taken back out)
    law = sc.level_db(1.0, "warm")
    assert abs(gain_db(45.0) - law - 2.1) < 0.4
    assert abs(gain_db(400.0) - law - 0.3) < 0.3
    assert gain_db(17.0) - law < -2.5


def test_the_tube_leaves_no_dc_once_its_tone_ran():
    t = np.arange(SR * 2) / SR
    x = 0.5 * np.sin(2 * np.pi * 220 * t)[:, None]
    y = dsp._saturate(x.copy(), {"drive_db": 12.0, "curve": "tube"}, SR)
    assert abs(float(np.mean(y[SR:, 0]))) < 1e-4


@pytest.mark.parametrize("curve", sc.NAMES)
def test_the_oracle_renders_the_character_the_export_does(curve):
    x = np.random.default_rng(7).standard_normal((SR, 2)) * 0.3
    e = {"type": "saturation", "drive_db": 7.0, "curve": curve, "oversample": 4, "output_db": 0.0, "mix": 1.0}
    want = dsp._saturate(x.copy(), e, SR)
    got = reference_monitor.render_monitor(x, SR, {"entries": [e]})
    assert np.max(np.abs(np.asarray(got) - want)) < 1e-9
