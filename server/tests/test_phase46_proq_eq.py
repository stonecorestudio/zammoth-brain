"""Phase 46 Tranche 1 (Pro-Q-class EQ, offline half): variable-slope hpf/lpf
cuts, bandpass, tilt_shelf, and per-band left/right — the dsp.py render path
plus the EqBand/EqAction model contract. reference_monitor / native parity is
covered in test_phase40_monitor_fidelity.py (the oracle harness)."""
import numpy as np
import pytest
from pedalboard import HighpassFilter, LowpassFilter, Pedalboard
from pydantic import ValidationError

from mix_engine import dsp
from mix_engine.models import EqAction, EqBand

from test_phase2_engine import band_power

SR = 48000


def _tone(freq_hz: float, seconds: float = 1.0, amp: float = 0.3,
         sr: int = SR) -> np.ndarray:
    n = int(sr * seconds)
    t = np.arange(n) / sr
    return (amp * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


def _settled_amp_db(sig: np.ndarray, ref_amp: float) -> float:
    """Peak amplitude (dB rel. `ref_amp`) over the settled second half of a
    steady tone — skips the filter's transient ramp-up."""
    tail = sig[-sig.shape[-1] // 2:].astype(np.float64)
    return 20.0 * np.log10(max(float(np.max(np.abs(tail))), 1e-12) / ref_amp)


# ---------- (a) a steeper slope attenuates a below-cutoff tone more ----------

def test_steeper_slope_attenuates_more_below_cutoff():
    cutoff = 1000.0
    tone = 500.0   # one octave below cutoff -> ~slope_db_oct of attenuation
    sig = _tone(tone, seconds=2.0)
    audio = np.stack([sig, sig])
    amp_db = {}
    for slope in (12, 24, 48, 96):
        band = {"shape": "hpf", "freq_hz": cutoff, "gain_db": 0.0, "q": 0.70711,
                "slope_db_oct": slope}
        wet = dsp._eq_band(audio, SR, band)
        amp_db[slope] = _settled_amp_db(wet[0], 0.3)
    assert amp_db[48] < amp_db[12] - 20.0   # 48 dB/oct clearly steeper than 12
    assert amp_db[24] < amp_db[12]
    assert amp_db[48] < amp_db[24]
    assert amp_db[96] < amp_db[48]


# ---------- (b) slope 6 (and unset) stays bit-identical to the legacy path ----------

@pytest.mark.parametrize("shape,cls,freq", [
    ("hpf", HighpassFilter, 250.0),
    ("lpf", LowpassFilter, 4000.0),
])
def test_slope6_bit_identical_to_pedalboard(shape, cls, freq):
    """Since 2026-10-17 (the EQ audit) the legacy 1st-order cut is the native's own coefficients in
    float64 (`dsp._rbj_band`), not Pedalboard's: bit-identical to that law, and within a hair of
    Pedalboard's filter, which it replaced."""
    from scipy.signal import lfilter

    from mix_engine.reference_monitor import biquad_coeffs
    sig = _tone(300.0, seconds=0.5)
    audio = np.stack([sig, sig])
    b, a = biquad_coeffs(shape, SR, freq, 0.707, 0.0)
    ref = lfilter(b, a, audio.astype(np.float64), axis=-1).astype(audio.dtype)
    pedal = Pedalboard([cls(cutoff_frequency_hz=freq)])(audio, SR)
    assert np.max(np.abs(ref - pedal)) < 1e-4

    band_explicit = {"shape": shape, "freq_hz": freq, "gain_db": 0.0, "q": 0.707,
                     "slope_db_oct": 6}
    band_unset = {"shape": shape, "freq_hz": freq, "gain_db": 0.0, "q": 0.707}
    assert np.array_equal(dsp._eq_band(audio, SR, band_explicit), ref)
    assert np.array_equal(dsp._eq_band(audio, SR, band_unset), ref)


# ---------- (c) a left-channel band is silent on the right channel ----------

def test_left_channel_band_leaves_right_bit_identical():
    rng = np.random.default_rng(46)
    n = SR // 2
    left = (rng.standard_normal(n) * 0.2).astype(np.float32)
    right = (rng.standard_normal(n) * 0.2).astype(np.float32)
    audio = np.stack([left, right])
    band = {"shape": "peak", "freq_hz": 1000.0, "gain_db": 6.0, "q": 1.0,
            "slope_db_oct": 6}
    wet = dsp._channel_apply(audio, "left", lambda a: dsp._eq_band(a, SR, band))
    assert np.array_equal(wet[1], audio[1])       # right untouched, bit-identical
    assert not np.array_equal(wet[0], audio[0])   # left actually changed


# ---------- (d) tilt_shelf raises the highs and lowers the lows ----------

def test_tilt_shelf_raises_highs_lowers_lows():
    n = SR
    rng = np.random.default_rng(46)
    noise = (rng.standard_normal(n) * 0.1).astype(np.float32)
    audio = np.stack([noise, noise])
    band = {"shape": "tilt_shelf", "freq_hz": 1000.0, "gain_db": 12.0, "q": 0.5,
            "slope_db_oct": 6}
    wet = dsp._eq_band(audio, SR, band)
    dry_frames = audio.T.astype(np.float64)
    wet_frames = wet.T.astype(np.float64)
    assert band_power(wet_frames, SR, 60, 200) < band_power(dry_frames, SR, 60, 200)
    assert band_power(wet_frames, SR, 5000, 12000) > band_power(dry_frames, SR, 5000, 12000)


# ---------- (e) bandpass passes the center, attenuates distant tones ----------

def test_bandpass_passes_center_attenuates_distant():
    center = 2000.0
    at_center = _tone(center, seconds=1.0)
    far = _tone(center * 8.0, seconds=1.0)   # 3 octaves up
    band = {"shape": "bandpass", "freq_hz": center, "gain_db": 0.0, "q": 2.0,
            "slope_db_oct": 6}
    wet_center = dsp._eq_band(np.stack([at_center, at_center]), SR, band)
    wet_far = dsp._eq_band(np.stack([far, far]), SR, band)
    center_db = _settled_amp_db(wet_center[0], 0.3)
    far_db = _settled_amp_db(wet_far[0], 0.3)
    assert center_db > -3.0             # near-unity at the center
    assert far_db < center_db - 20.0    # clearly attenuated far away


# ---------- (f) model round-trip ----------

def test_model_accepts_new_shapes_slopes_channels():
    action = EqAction(stem_id="stem_a",
                      band=EqBand(freq_hz=200.0, gain_db=0.0, q=0.9,
                                  shape="hpf", slope_db_oct=48),
                      channel="left")
    assert action.band.slope_db_oct == 48
    assert action.channel == "left"

    for shape in ("bandpass", "tilt_shelf"):
        EqBand(freq_hz=1000.0, gain_db=3.0, q=1.0, shape=shape)

    for channel in ("left", "right"):
        EqAction(stem_id="stem_a",
                 band=EqBand(freq_hz=1000.0, gain_db=0.0, q=1.0, shape="peak"),
                 channel=channel)

    # default slope is 6, unaffected by the new shapes
    assert EqBand(freq_hz=1000.0, gain_db=0.0, q=1.0, shape="peak").slope_db_oct == 6


def test_model_rejects_invalid_slope():
    with pytest.raises(ValidationError):
        EqBand(freq_hz=200.0, gain_db=0.0, q=0.9, shape="hpf", slope_db_oct=18)
