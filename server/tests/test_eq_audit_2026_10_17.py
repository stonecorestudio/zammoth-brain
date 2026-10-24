"""The EQ's audit, 2026-10-17 (the device-by-device audit) - the offline engine.

The native half is `--eq-audit`. Here:
  * a band plays its own design: peak, shelves and the 1st-order cuts run the native's
    coefficients (`biquad_coeffs`) in float64 (`dsp._rbj_band`) - through Pedalboard (JUCE's float
    maths) a 30 Hz Q 8 tilt shelf at 96 kHz was 6 dB off its design, the DAW's 2 dB the other way;
  * the transcription clamps q as `makeBiquad` does: q 0 is the crossover legs' sentinel ONLY, and
    a peak at q 0 was a division by zero.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import freqz

from mix_engine import dsp
from mix_engine.reference_monitor import biquad_coeffs


def _response_db(sr: int, band: dict, freqs: np.ndarray) -> np.ndarray:
    n = sr * 8
    imp = np.zeros((1, n), dtype=np.float32)
    imp[0, 0] = 1.0
    y = np.asarray(dsp._eq_band(imp, sr, band), dtype=np.float64)[0]
    spec = np.abs(np.fft.rfft(y))
    return 20 * np.log10(np.interp(freqs, np.fft.rfftfreq(n, 1 / sr), spec))


def test_a_low_shelf_at_a_high_rate_plays_its_design():
    sr = 96000
    freqs = np.geomspace(10, 2000, 200)
    band = {"shape": "tilt_shelf", "freq_hz": 30.0, "q": 8.0, "gain_db": 12.0}
    design = np.ones(len(freqs), complex)
    for shape, g in (("low_shelf", -6.0), ("high_shelf", 6.0)):
        b, a = biquad_coeffs(shape, sr, 30.0, 8.0, g)
        design *= freqz(b, a, worN=freqs, fs=sr)[1]
    err = np.max(np.abs(_response_db(sr, band, freqs) - 20 * np.log10(np.abs(design))))
    assert err < 0.05, err


def test_a_peak_at_q_zero_is_the_natives_q_tenth():
    """`makeBiquad` keeps q 0 for lpf2 / hpf2 (the crossover sentinel) and clamps it to 0.1 for
    every other shape; the transcription kept it for every shape (and divided by it)."""
    for shape in ("peak", "low_shelf", "high_shelf", "notch", "bandpass"):
        b0, a0 = biquad_coeffs(shape, 44100, 2000.0, 0.0, 6.0)
        b1, a1 = biquad_coeffs(shape, 44100, 2000.0, 0.1, 6.0)
        assert np.isfinite(b0).all() and np.allclose(b0, b1) and np.allclose(a0, a1), shape
    lr_b, _ = biquad_coeffs("lpf2", 44100, 2000.0, 0.0, 0.0)          # the sentinel still means
    bw_b, _ = biquad_coeffs("lpf2", 44100, 2000.0, 0.70710678, 0.0)   # Q = 1/sqrt 2
    assert np.allclose(lr_b, bw_b)


def test_the_eq_face_draws_at_the_nodes_rate():
    """M22 rule (3): a face that draws a filter evaluates the chain's coefficients at the node's own
    rate. EqGraph drew every band at 48 kHz (a 19 kHz cut at 96 kHz read 4.5 dB off what played).
    A source pin: the curve cache reads `sampleRate`, the dock hands it the monitor's rate, and the
    face's coefficient maker clamps frequency as `makeBiquad` does."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    graphs = (root / "native" / "Source" / "DeviceGraphs.h").read_text()
    eq = graphs[graphs.index("class EqGraph : public Component"):graphs.index("class DynEqGraph")]
    assert "const double sr = sampleRate;" in eq and "6, sampleRate, f)" in eq
    assert "f = jlimit (2.0, sr * 0.49, f);" in graphs[graphs.index("inline void eqCoeffs"):][:900]
    dock = (root / "daw" / "Source" / "DeviceChainView.h").read_text()
    assert "graph.setSampleRate (owner.host.getMonitor().getSampleRate());" in dock


def test_a_frequency_is_swept_log_in_the_lane_and_by_a_controller():
    """A linear 20 Hz..20 kHz lane gave 20-200 Hz 0.9 % of its height, and a CC or a macro the same
    share of its travel. A source pin: the lane maps a device `_hz` target log both ways, and the
    parameter a controller drives carries a log range."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    arrange = (root / "daw" / "Source" / "ArrangeView.h").read_text()
    assert 'l.target.endsWith ("_hz")' in arrange
    assert "laneValToY (i, ref.range," not in arrange and "laneYToVal (li, addRef.range," not in arrange
    plugin = (root / "daw" / "Source" / "MonitorChainPlugin.h").read_text()
    body = plugin[plugin.index("static juce::NormalisableRange<float> rangeFor"):][:1200]
    assert 't.endsWith ("_hz")' in body and "std::log" in body
