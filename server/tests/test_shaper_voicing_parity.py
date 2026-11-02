"""The overdrive's voicing and the shapers' oversampling (2026-10-24, James: "they sound awful").

Two things are held here. (1) A hard drive no longer aliases: at 1x a 24 dB overdrive or a
24 dB multiband band folded its harmonics back within 15-20 dB of the note; at the new defaults
(overdrive 8x, saturation 4x - the multiband form's whole split) that is held under -40 dB.
(2) The voicing constants and the defaults are ONE set across the engines: dsp.py,
reference_monitor.py, monitor_chain.py, models.py, MonitorChain.h and DeviceSpecs.h (the dock's
seed for a new device).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from mix_engine import dsp, models, reference_monitor

SR = 48000
ROOT = Path(__file__).resolve().parents[2]
HDR = (ROOT / "native/Source/MonitorChain.h").read_text()
SPECS = (ROOT / "native/Source/DeviceSpecs.h").read_text()


def _inharmonic_db(y: np.ndarray, f0: float) -> float:
    """Energy that is not a harmonic of f0 (aliasing), relative to the whole, in dB."""
    w = np.blackman(len(y))
    spec = np.abs(np.fft.rfft(y[:, 0] * w)) ** 2
    fr = np.fft.rfftfreq(len(y), 1.0 / SR)
    harm = fr < 15.0
    h = 1
    while h * f0 <= SR / 2:
        harm |= np.abs(fr - h * f0) < 15.0
        h += 1
    return float(10.0 * np.log10(spec[~harm].sum() / spec.sum()))


def _tone(f0: float = 7013.9, amp: float = 0.5) -> np.ndarray:
    t = np.arange(2 * SR) / SR
    return np.stack([amp * np.sin(2 * np.pi * f0 * t)] * 2, axis=1)


def test_a_hard_overdrive_at_its_default_does_not_alias():
    assert _inharmonic_db(dsp._overdrive(_tone(), SR, {"drive_db": 24.0}), 7013.9) < -40.0


def test_a_hard_multiband_saturation_at_its_default_does_not_alias():
    entry = {"drive_db": 3.0, "low": {"drive_db": 24.0}, "mid": {"drive_db": 24.0},
             "high": {"drive_db": 24.0}}
    assert _inharmonic_db(dsp._saturate(_tone(), entry, SR), 7013.9) < -40.0


def test_the_overdrive_clipper_is_zero_at_zero_unity_small_signal_and_asymmetric():
    k = 10.0 ** (18.0 / 20.0)
    assert abs(float(dsp.od_shape(0.0, k))) < 1e-12
    eps = 1e-6
    assert abs(float(dsp.od_shape(eps, k)) / eps - 1.0) < 1e-3
    top, bottom = float(dsp.od_shape(10.0, k)), float(dsp.od_shape(-10.0, k))
    assert top > 0 > bottom and abs(abs(top) - abs(bottom)) > 0.05 * abs(top)


def test_the_tube_curve_leaves_no_dc_in_the_multiband_form():
    t = np.arange(2 * SR) / SR
    x = np.stack([0.5 * np.sin(2 * np.pi * 110.0 * t)] * 2, axis=1)
    entry = {"drive_db": 3.0, "curve": 2, "low": {"drive_db": 18.0}, "mid": {"drive_db": 18.0},
             "high": {"drive_db": 18.0}}
    y = dsp._saturate(x, entry, SR)
    assert abs(float(y[SR:].mean())) < 1e-4


def _cpp_number(name: str) -> float:
    m = re.search(r"\b" + name + r"\s*=\s*([0-9.]+)f?\b", HDR)
    assert m, name
    return float(m.group(1))


def test_the_voicing_is_one_set_in_both_engines():
    assert _cpp_number("kOdEmphasisHz") == dsp.OD_EMPHASIS_HZ
    assert _cpp_number("kOdEmphasisDb") == dsp.OD_EMPHASIS_DB
    assert _cpp_number("kOdFizzHz") == dsp.OD_FIZZ_HZ
    assert _cpp_number("kShaperDcHz") == dsp.SHAPER_DC_HZ
    assert _cpp_number("kOdBias") == dsp.OD_BIAS


def test_the_oversampling_defaults_are_one_set_everywhere():
    od, sat = dsp.OD_OVERSAMPLE_DEFAULT, dsp.SAT_OVERSAMPLE_DEFAULT
    assert (od, sat) == (8, 4)
    assert _cpp_number("kOdOversampleDefault") == od
    assert _cpp_number("kSatOversampleDefault") == sat
    assert (reference_monitor._OD_OS_DEFAULT, reference_monitor._SAT_OS_DEFAULT) == (od, sat)
    assert models.OverdriveAction(stem_id="a").oversample == od
    assert models.SaturationAction(stem_id="a").oversample == sat
    seed = re.search(r'type == "overdrive"\)\s*\{.*?p \("oversample", (\d+)\)', SPECS, re.S)
    assert seed and int(seed.group(1)) == od
    seed = re.search(r'type == "saturation"\)\s*\{.*?p \("oversample", (\d+)\)', SPECS, re.S)
    assert seed and int(seed.group(1)) == sat


def test_the_overdrive_shapes_bias_constants_are_computed_as_a_sample_is():
    """odShape's 1 - e^-bias and e^bias are computed at load by the same float expression a sample uses
    (2026-10-26: decimal literals were a unit in the last place off, silence in gave 1e-8 out and no chain
    with an overdrive slept), and are no function-local static on the audio thread."""
    cpp = (ROOT / "native/Source/MonitorChain.cpp").read_text()
    assert "const float MonitorChain::kOdF0 = 1.0f - std::exp (-MonitorChain::kOdBias);" in cpp
    assert "const float MonitorChain::kOdEb = std::exp (MonitorChain::kOdBias);" in cpp
    assert "const float f0 = kOdF0, eb = kOdEb;" in HDR
    assert "static const float f0" not in HDR
    assert _cpp_number("kOdBias") == dsp.OD_BIAS
