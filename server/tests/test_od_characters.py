"""The overdrive's characters (2026-10-26, od_characters.py): the pedal it was, and three fitted amps.

The native half is `--od-characters` (each character out of a real MonitorChain against this module's
figures) and the export oracle. Here:
  * the header the native reads is this module's table, written by `--write-header`; the names are the
    model's; the trims are what `trim_db` computes;
  * the PEDAL is the overdrive as it was - an entry with no character renders what character 0 renders;
  * each amp keeps the reference's TONE (its small-signal response by octave, read off the reference's
    impulse response, relative to its 250 Hz - 4 kHz mean) and its GRIT (its compression, H2 and H3 at
    two drives, two frequencies, two levels, read off the reference with sines) to the fit's own error;
  * an amp is finite and bounded on any input, leaves no DC, and the oracle renders what the export does.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mix_engine import dsp, models, od_characters as oc, reference_monitor

REPO = Path(__file__).resolve().parents[2]
SR = 48000


def test_the_native_header_is_this_table():
    assert (REPO / "native/Source/OdCharacters.h").read_text() == oc.header_text()
    assert models.OD_CHARACTERS == oc.NAMES


@pytest.mark.parametrize("c", (1, 2, 3))
def test_the_trims_are_what_trim_db_computes(c):
    assert oc.TRIMS_DB[c] == pytest.approx(oc.trim_db(c), abs=0.002)


def _sig():
    return np.random.default_rng(3).standard_normal((SR, 2)) * 0.2


def test_the_pedal_is_the_overdrive_as_it_was():
    e = {"type": "overdrive", "drive_db": 18.0, "tone": 0.7, "oversample": 4}
    assert np.array_equal(dsp._overdrive(_sig(), SR, e), dsp._overdrive(_sig(), SR, {**e, "character": 0}))
    assert np.array_equal(dsp._overdrive(_sig(), SR, e), dsp._overdrive(_sig(), SR, {**e, "character": "pedal"}))
    assert models.OverdriveAction(stem_id="a").character == 0
    assert models.OverdriveAction(stem_id="a", character="power").character == 3


# The reference, read off it: its small-signal tone by octave (dB, relative to its 250 Hz - 4 kHz mean)
TONE = {
    1: {63: -11.1, 125: -0.2, 250: -6.2, 500: -9.1, 1000: -6.1, 2000: 0.6, 4000: 5.1, 8000: -7.1},
    2: {63: -23.1, 125: -11.3, 250: -9.5, 500: -6.0, 1000: -8.0, 2000: -2.1, 4000: 5.9, 8000: -0.6},
    3: {63: -3.1, 125: 3.3, 250: 0.4, 500: 3.7, 1000: 1.5, 2000: -8.8, 4000: -15.0, 8000: -53.4},
}
TOP_UNDER_DB = 40.0   # the reference's 16 kHz octave sits 78-110 dB under its mids: an amp's top is gone
TONE_TOL_DB = 3.0


def _octave_db(y, fc):
    f = np.fft.rfftfreq(len(y), 1 / SR)
    m = (f > fc / 2 ** (1 / 6)) & (f < fc * 2 ** (1 / 6))
    return 10 * np.log10(np.mean(np.abs(np.fft.rfft(y)[m]) ** 2))


@pytest.mark.parametrize("c", (1, 2, 3))
def test_an_amp_keeps_the_references_tone(c):
    x = np.zeros((SR, 1)); x[100, 0] = 1e-4            # an impulse far under the curve: the linear path
    y = dsp._overdrive(x, SR, {"drive_db": 0.0, "character": c, "oversample": 1})[:, 0]
    ours = {fc: _octave_db(y, fc) for fc in TONE[c]}
    mid = 10 * np.log10(np.mean([10 ** (ours[k] / 10) for k in (250, 500, 1000, 2000, 4000)]))
    for fc, want in TONE[c].items():
        assert abs((ours[fc] - mid) - want) < TONE_TOL_DB, (c, fc, round(ours[fc] - mid, 1), want)
    assert _octave_db(y, 16000) - mid < -TOP_UNDER_DB, (c, "16 kHz", round(_octave_db(y, 16000) - mid, 1))


# (our drive dB, Hz, input dBFS, compression dB, H2 dB, H3 dB): the reference at 25 / 50 % drive
GRIT = {
    1: [(8.04, 300, -12, -0.1, -44.7, -54.6), (8.04, 1000, -12, -0.1, -31.7, -34.4),
        (16.5, 300, -12, -0.6, -28.5, -39.4), (16.5, 1000, -12, -0.9, -16.5, -19.7)],
    2: [(8.99, 300, -24, -2.7, -28.8, -28.6), (8.99, 1000, -12, -12.0, -22.5, -1.5),
        (17.94, 300, -12, -16.8, -28.8, -17.8), (17.94, 1000, -24, -9.6, -21.9, -2.5)],
    3: [(22.0, 300, -24, -2.0, -22.3, -25.9), (22.0, 300, -12, -8.5, -18.9, -16.1),
        (22.0, 1000, -12, -8.5, -30.7, -20.0)],
}
COMP_TOL_DB, HARM_TOL_DB = 2.0, 6.0


def _fund_h(y, hz):
    t = np.arange(len(y)) / SR
    m = slice(len(y) // 2, len(y))
    return [abs(np.sum(y[m] * np.exp(-2j * np.pi * hz * k * t[m]))) for k in (1, 2, 3)]


@pytest.mark.parametrize("c", (1, 2, 3))
def test_an_amp_keeps_the_references_grit(c):
    for drive, hz, lv, comp, h2, h3 in GRIT[c]:
        t = np.arange(SR) / SR
        def run(level):
            x = (10 ** (level / 20) * np.sin(2 * np.pi * hz * t))[:, None]
            return _fund_h(dsp._overdrive(x, SR, {"drive_db": drive, "character": c, "oversample": 8})[:, 0], hz)
        small = run(-60.0)[0] / 10 ** (-60 / 20)
        a1, a2, a3 = run(lv)
        got = (20 * np.log10(a1 / 10 ** (lv / 20) / small), 20 * np.log10(a2 / a1), 20 * np.log10(a3 / a1))
        assert abs(got[0] - comp) < COMP_TOL_DB, (c, drive, hz, lv, "compression", got, (comp, h2, h3))
        assert abs(got[1] - h2) < HARM_TOL_DB, (c, drive, hz, lv, "H2", got, (comp, h2, h3))
        assert abs(got[2] - h3) < HARM_TOL_DB, (c, drive, hz, lv, "H3", got, (comp, h2, h3))


@pytest.mark.parametrize("c", (1, 2, 3))
def test_an_amp_is_bounded_and_leaves_no_dc(c):
    t = np.arange(SR * 2) / SR
    x = (0.9 * np.sin(2 * np.pi * 110 * t))[:, None]
    y = dsp._overdrive(x, SR, {"drive_db": 36.0, "character": c, "oversample": 1})
    assert np.all(np.isfinite(y)) and abs(float(np.mean(y[SR:, 0]))) < 1e-3
    big = dsp._overdrive(np.full((4800, 1), 1e6), SR, {"drive_db": 36.0, "character": c, "oversample": 1})
    assert np.all(np.isfinite(big))


@pytest.mark.parametrize("c", oc.NAMES)
def test_the_oracle_renders_the_character_the_export_does(c):
    e = {"type": "overdrive", "drive_db": 15.0, "character": c, "oversample": 4, "tone": 0.6, "mix": 0.8}
    want = dsp._overdrive(_sig(), SR, e)
    got = reference_monitor.render_monitor(_sig(), SR, {"entries": [e]})
    assert np.max(np.abs(np.asarray(got) - want)) < 1e-9
