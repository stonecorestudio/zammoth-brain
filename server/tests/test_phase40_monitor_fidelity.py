"""Phase 40 oracle harness: the native MonitorChain reference must match the
offline engine (dsp.py / Pedalboard). EQ + gain + brickwall null tightly (the
near-bit-exact claim); the compressor is checked loosely (documented
approximation). This is the gate every native module passes before its export
path flips to native.
"""
import numpy as np
import pytest
from pedalboard import Pedalboard

from mix_engine import dsp
from mix_engine import reference_monitor as rm
from mix_engine.monitor_chain import monitor_chain_params

SR = 48000


def _sig(seconds=1.0, ch=2, seed=0, amp=0.25):
    rng = np.random.default_rng(seed)
    n = int(SR * seconds)
    return (rng.standard_normal((n, ch)) * amp).astype(np.float64)


def _err_db(ref, test):
    """Peak error relative to the reference's peak, in dB. -inf means identical."""
    ref = np.asarray(ref, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    peak = float(np.max(np.abs(ref))) or 1.0
    return 20.0 * np.log10(max(float(np.max(np.abs(ref - test))), 1e-20) / peak)


def _pedalboard_band(sig, band):
    """Run one EQ band exactly as dsp._apply_master_chain does: (channels,
    frames) float32 through Pedalboard, back to (frames, channels)."""
    dry = np.ascontiguousarray(sig.T.astype(np.float32))
    wet = Pedalboard([dsp._eq_plugin(band, SR)])(dry, SR)
    return wet.T.astype(np.float64)


# ---------- EQ: null the numpy reference against Pedalboard's JUCE filters ----

@pytest.mark.parametrize("shape,freq,gain,q", [
    ("peak", 1000.0, 4.0, 1.0),
    ("peak", 3000.0, -6.0, 2.5),
    ("low_shelf", 120.0, 3.0, 0.7),
    ("low_shelf", 200.0, -5.0, 0.7),
    ("high_shelf", 8000.0, -4.0, 0.7),
    ("high_shelf", 6000.0, 2.0, 0.7),
    ("lpf", 9000.0, 0.0, 0.707),
    ("hpf", 80.0, 0.0, 0.707),
])
def test_eq_band_matches_pedalboard(shape, freq, gain, q):
    sig = _sig()
    band = {"shape": shape, "freq_hz": freq, "gain_db": gain, "q": q}
    wet = _pedalboard_band(sig, band)
    ref = rm.apply_eq_band(sig, SR, shape, freq, q, gain)
    err = _err_db(wet, ref)
    assert err < -55.0, f"{shape} @ {freq}Hz: {err:.1f} dB (expected float32 floor)"


# ---------- Phase 46 (Pro-Q-class EQ): dsp render vs reference_monitor parity ----

@pytest.mark.parametrize("shape,freq,gain,q,slope", [
    ("bandpass", 2000.0, 0.0, 1.5, 6),
    ("bandpass", 300.0, 0.0, 4.0, 6),
    ("tilt_shelf", 1000.0, 6.0, 0.7, 6),
    ("tilt_shelf", 500.0, -8.0, 1.2, 6),
    ("hpf", 300.0, 0.0, 0.9, 12),
    ("hpf", 300.0, 0.0, 0.9, 24),
    ("hpf", 300.0, 0.0, 0.9, 48),
    ("hpf", 300.0, 0.0, 0.9, 96),
    ("lpf", 5000.0, 0.0, 0.9, 12),
    ("lpf", 5000.0, 0.0, 0.9, 24),
    ("lpf", 5000.0, 0.0, 0.9, 48),
    ("lpf", 5000.0, 0.0, 0.9, 96),
])
def test_phase46_eq_band_dsp_matches_reference(shape, freq, gain, q, slope):
    """dsp._eq_band ((channels, frames)) vs reference_monitor.apply_eq_band
    ((frames, channels)) — same coefficients (dsp imports them from
    reference_monitor), so this should null far tighter than the float32
    Pedalboard comparison above."""
    sig = _sig()
    band = {"shape": shape, "freq_hz": freq, "gain_db": gain, "q": q,
            "slope_db_oct": slope}
    dry32 = np.ascontiguousarray(sig.T.astype(np.float32))
    wet = dsp._eq_band(dry32, SR, band).T.astype(np.float64)
    ref = rm.apply_eq_band(sig, SR, shape, freq, q, gain, slope)
    err = _err_db(ref, wet)
    assert err < -50.0, f"{shape} @ {freq}Hz slope={slope}: {err:.1f} dB"


def test_gain_then_eq_chain_matches_sequential_pedalboard():
    """A realistic master EQ chain: float64 gain (like render_master) then each
    band via Pedalboard float32 (like _apply_master_chain), vs render_monitor."""
    sig = _sig()
    gain_db = -1.5
    bands = [
        {"shape": "low_shelf", "freq_hz": 120.0, "gain_db": 2.0, "q": 0.7},
        {"shape": "peak", "freq_hz": 2500.0, "gain_db": -3.0, "q": 1.5},
        {"shape": "high_shelf", "freq_hz": 9000.0, "gain_db": 1.5, "q": 0.7},
    ]
    oracle = sig * 10.0 ** (gain_db / 20.0)
    for b in bands:
        oracle = _pedalboard_band(oracle, b)

    entries = [{"type": "gain", "gain_db": gain_db}]
    entries += [{"type": "eq", **b} for b in bands]
    ref = rm.render_monitor(sig, SR, {"entries": entries, "print_only": []})
    err = _err_db(oracle, ref)
    assert err < -50.0, f"chain null: {err:.1f} dB"


# ---------- notch: a hand-written biquad (Pedalboard has no NotchFilter) -------

def test_notch_offline_matches_oracle_and_cuts_centre():
    """The notch lives outside Pedalboard, so dsp._notch and the numpy oracle are
    the same biquad — they must agree to ~bit-exact, deeply cut a tone at f0, and
    leave a distant tone alone."""
    f0, q = 1000.0, 4.0
    sig = _sig()
    # dsp render path is (channels, frames); the oracle is (frames, channels)
    off = dsp._notch(np.ascontiguousarray(sig.T), SR, f0, q).T
    ref = rm.apply_eq_band(sig, SR, "notch", f0, q, 0.0)
    assert _err_db(off, ref) < -100.0, "dsp notch must match the oracle notch"

    t = np.arange(int(SR * 0.5)) / SR

    def cut_db(freq):
        x = (np.sin(2 * np.pi * freq * t)[:, None] * np.ones((1, 2)))
        y = rm.apply_eq_band(x, SR, "notch", f0, q, 0.0)
        rms = lambda a: np.sqrt(np.mean(a[SR // 10:] ** 2))   # drop the settling tail
        return 20.0 * np.log10(rms(y) / rms(x) + 1e-12)

    assert cut_db(f0) < -20.0, f"notch should cut hard at centre: {cut_db(f0):.1f} dB"
    assert cut_db(f0 / 8) > -1.0, f"notch should pass a distant tone: {cut_db(f0 / 8):.1f} dB"


def test_eq_band_dispatch_routes_notch():
    """dsp._eq_band sends notch to the numpy biquad and everything else to
    Pedalboard — the path the render loop actually calls."""
    sig = np.ascontiguousarray(_sig().T)   # (channels, frames)
    band = {"shape": "notch", "freq_hz": 2000.0, "gain_db": 0.0, "q": 3.0}
    out = dsp._eq_band(sig, SR, band)
    assert out.shape == sig.shape
    assert _err_db(out, dsp._notch(sig, SR, 2000.0, 3.0)) < -120.0


def test_eqband_model_accepts_notch():
    """The proposal/co-edit contract accepts notch (the Literal was widened)."""
    from mix_engine.models import EqAction, EqBand
    a = EqAction(stem_id="master",
                 band=EqBand(freq_hz=1000.0, gain_db=0.0, q=4.0, shape="notch"))
    assert a.band.shape == "notch"


def test_notch_master_render_attenuates_centre(demo_project):
    """End-to-end: a user-set notch on the master renders through the offline
    dispatch and clearly drops the energy around its centre on the real demo mix."""
    from numpy.fft import rfft, rfftfreq
    from mix_engine.session import Session
    s = Session(demo_project)
    base, sr = dsp.render_master(s)
    s._state["master_chain"].append(
        {"type": "eq", "stem_id": "master",
         "band": {"shape": "notch", "freq_hz": 3000.0, "gain_db": 0.0, "q": 1.0}})
    notched, _ = dsp.render_master(s)

    def band_energy(x):
        m = x.mean(axis=1) if x.ndim == 2 else x
        mag = np.abs(rfft(m))
        f = rfftfreq(m.shape[0], 1.0 / sr)
        sel = (f > 3000.0 * 0.88) & (f < 3000.0 * 1.12)
        return float(np.sum(mag[sel] ** 2))

    assert band_energy(notched) < 0.6 * band_energy(base), "notch should gut the 3 kHz band"


# ---------- gain + limiter -------------------------------------------------

def test_gain_entry_is_exact():
    sig = _sig()
    out = rm.render_monitor(sig, SR, {"entries": [{"type": "gain", "gain_db": -6.0}],
                                      "print_only": []})
    assert _err_db(sig * 10.0 ** (-6.0 / 20.0), out) < -120.0


def test_brickwall_matches_dsp():
    """The offline limiter brickwall reference must equal dsp._brickwall."""
    hot = _sig(seed=1, amp=0.9) * 3.0
    ref = rm.brickwall(hot.copy(), SR, {"ceiling_dbtp": -1.0})
    oracle = dsp._brickwall(hot.copy(), SR, {"ceiling_dbtp": -1.0})
    assert _err_db(oracle, ref) < -100.0


def test_brickwall_bounds_true_peak():
    """ceiling_dbtp means dBTP (2026-08-13): a signal whose SAMPLE peak sits under
    the ceiling but whose inter-sample (reconstruction) peak is ~0 dBTP must still
    be limited. The old sample-peak detector returned it untouched."""
    from mix_engine.analysis import _true_peak_dbtp
    t = np.arange(SR // 2)
    # 8 kHz @ 48k, phase 0: 6 samples/cycle at 0/60/120/... degrees, so the waveform
    # peak falls BETWEEN two 0.866*A samples — sample peak -1.25 dBFS (under the -1.0
    # ceiling) while the reconstructed waveform peaks at A (~0 dBTP).
    x = 0.999 * np.sin(2.0 * np.pi * 8000.0 * t / SR)
    audio = np.stack([x, x], axis=1)
    out = dsp._brickwall(audio, SR, {"ceiling_dbtp": -1.0})
    # The old sample-peak detector returned this signal bit-untouched.
    assert not np.array_equal(out, audio)
    # The meters' estimate of what is left (the 4x Catmull-Rom LiveMeters and the CLIP lamp
    # read) is under the ceiling. The limiter's own detector is the native's 33-tap polyphase
    # table since batch T (mix_engine/tp_detector.py, reader E4·R7) - it reads higher than
    # the cubic, so this still holds; test_reader_p2_t.py holds the contractual measurement.
    pad = np.pad(out, ((1, 2), (0, 0)), mode="edge")
    p0, p1, p2, p3 = pad[:-3], pad[1:-2], pad[2:-1], pad[3:]
    est = np.abs(out)
    for tt in (0.25, 0.5, 0.75):
        v = 0.5 * ((2.0 * p1) + (-p0 + p2) * tt
                   + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * tt * tt
                   + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * tt ** 3)
        np.maximum(est, np.abs(v), out=est)
    ceiling = 10.0 ** (-1.0 / 20.0)
    assert float(est.max()) <= ceiling * (1.0 + 1e-3)
    # And the sinc-measured true peak improves materially vs the input's (which the
    # sample-peak detector left at ~+0.4 dBTP — actual inter-sample clipping).
    assert _true_peak_dbtp(out) <= _true_peak_dbtp(audio) - 0.6
    assert _true_peak_dbtp(out) <= 0.0


def test_limiter_respects_ceiling():
    hot = _sig(seed=2, amp=0.9) * 2.0
    out = rm.render_monitor(hot, SR, {
        "entries": [{"type": "limiter", "ceiling_dbtp": -1.0, "makeup_db": 0.0}],
        "print_only": []})
    ceiling = 10.0 ** (-1.0 / 20.0)
    assert float(np.max(np.abs(out))) <= ceiling + 1e-6


# ---------- compressor (loose; documented approximation) -------------------

def test_compressor_reduces_dynamics_and_is_bounded():
    sig = _sig(seed=3, amp=0.5)
    out = rm._compress(sig, SR, threshold_db=-24.0, ratio=4.0,
                       attack_ms=5.0, release_ms=120.0)
    assert float(np.max(np.abs(out))) <= float(np.max(np.abs(sig))) + 1e-9
    assert np.all(np.isfinite(out))


def test_dynamic_eq_cuts_a_loud_band_only_downward():
    """Real-time dynamic EQ (perceptual, not oracle-tight): a loud tone in the
    band is attenuated, output is never louder than dry (downward only), finite."""
    t = np.arange(int(SR * 0.5)) / SR
    tone = (0.5 * np.sin(2 * np.pi * 3000.0 * t)).astype(np.float64)
    sig = np.stack([tone, tone], axis=1)
    out = rm._dynamic_eq(sig, SR, freq_hz=3000.0, q=4.0, threshold_db=-30.0,
                         ratio=4.0, attack_ms=5.0, release_ms=80.0, gain_db=-12.0)
    assert np.all(np.isfinite(out))
    assert float(np.max(np.abs(out))) <= float(np.max(np.abs(sig))) + 1e-6
    rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
    assert rms(out) < 0.9 * rms(sig)            # the dominant 3 kHz tone is cut


def test_multiband_comp_unity_reconstruction_and_compresses():
    """The real 3-way LR4 split (2026-10-17, the multiband audit; it was a complementary mid,
    which made a compressed low band LOUDER): at ratio 1 the three bands sum to an ALLPASS of
    the input - its magnitude, not its samples; with compression, peaks are reduced + bounded."""
    sig = _sig(seed=5, amp=0.3)
    flat = {"threshold_db": 0.0, "ratio": 1.0, "makeup_db": 0.0}
    recon = rm._multiband_comp(sig, SR, flat, flat, flat, 10.0, 150.0)
    mag = lambda a: np.abs(np.fft.rfft(a, axis=0))
    rel_db = 20 * np.log10((mag(recon)[50:-50] + 1e-9) / (mag(sig)[50:-50] + 1e-9))
    assert float(np.median(np.abs(rel_db))) < 0.05   # unity: magnitude-flat (an allpass)
    band = {"threshold_db": -40.0, "ratio": 6.0, "makeup_db": 0.0}
    comp = rm._multiband_comp(sig * 2.0, SR, band, band, band, 5.0, 120.0)
    assert float(np.max(np.abs(comp))) <= float(np.max(np.abs(sig * 2.0))) + 1e-6
    assert np.all(np.isfinite(comp))


def test_match_eq_lifts_its_band():
    """Min-phase match-EQ approximation: a +6 dB presence correction raises energy
    around its band center (finite, real-time, no FIR latency)."""
    from numpy.fft import rfft, rfftfreq
    from mix_engine.monitor_chain import _BAND_CENTER_HZ, MATCH_EQ_Q
    sig = _sig(seed=6, amp=0.3)
    bands = [{"freq_hz": _BAND_CENTER_HZ["presence"], "gain_db": 6.0, "q": MATCH_EQ_Q}]
    out = rm._match_eq(sig, SR, bands)
    assert np.all(np.isfinite(out))
    freqs = rfftfreq(sig.shape[0], 1.0 / SR)
    m = (freqs > 2000.0) & (freqs < 4000.0)
    energy = lambda a: float(np.mean(np.abs(rfft(a[:, 0]))[m]))
    assert energy(out) > energy(sig) * 1.2


# ---------- saturation (bit-exact: memoryless tanh waveshaper) -------------

@pytest.mark.parametrize("drive_db", [3.0, 6.0, 12.0])
def test_saturation_matches_dsp_bit_exact(drive_db):
    """The monitor's saturation reference must equal dsp._saturate — same
    memoryless formula (tanh(k·x)/k), so it nulls to the float noise floor,
    a tight tier like gain/EQ (not a loose approximation)."""
    sig = _sig(seed=7, amp=0.6)
    oracle = dsp._saturate(sig.copy(), {"drive_db": drive_db})
    ref = rm.render_monitor(sig, SR, {
        "entries": [{"type": "saturation", "drive_db": drive_db}],
        "print_only": []})
    assert _err_db(oracle, ref) < -100.0, f"drive {drive_db}: {_err_db(oracle, ref):.1f} dB"


@pytest.mark.parametrize("curve", [0, 1, 2, 3])
@pytest.mark.parametrize("mix", [1.0, 0.4])
def test_saturation_curves_and_mix_match_dsp_bit_exact(curve, mix):
    """warm / tube / hard and a parallel mix: the reference monitor knew only tanh
    at full wet, so the null test for the other three curves and for any dry/wet
    blend was run by nothing (2026-09-01 sweep)."""
    sig = _sig(seed=11, amp=0.6)
    entry = {"drive_db": 9.0, "curve": curve, "mix": mix}
    oracle = dsp._saturate(sig.copy(), entry)
    ref = rm.render_monitor(sig, SR, {"entries": [{"type": "saturation", **entry}],
                                      "print_only": []})
    assert _err_db(oracle, ref) < -100.0, f"curve {curve} mix {mix}: {_err_db(oracle, ref):.1f} dB"


def test_saturation_in_monitor_contract():
    """Saturation resolves into the live entries (heard live), not print_only —
    it appears on stem/bus auditions (never the master chain)."""
    out = monitor_chain_params([{"type": "saturation", "stem_id": "vox",
                                 "drive_db": 6.0}])
    assert out["entries"] == [{"type": "saturation", "drive_db": 6.0,
                               "output_db": 0.0, "curve": 0, "mix": 1.0,
                               # oversample 1 = off, the exact elementwise path (2026-09-06)
                               "oversample": 1, "enabled": True}]
    assert out["print_only"] == []


@pytest.mark.parametrize("output_db", [-6.0, 3.0, 9.0])
def test_saturation_output_trim_matches_dsp_bit_exact(output_db):
    """The post-shaper output trim is a plain gain applied last, so the monitor
    reference still nulls to dsp._saturate when output_db is set (tight tier)."""
    sig = _sig(seed=8, amp=0.5)
    oracle = dsp._saturate(sig.copy(), {"drive_db": 6.0, "output_db": output_db})
    ref = rm.render_monitor(sig, SR, {
        "entries": [{"type": "saturation", "drive_db": 6.0, "output_db": output_db}],
        "print_only": []})
    assert _err_db(oracle, ref) < -100.0, f"output {output_db}: {_err_db(oracle, ref):.1f} dB"


def test_mb_comp_crossovers_round_trip_in_monitor():
    """User-movable crossovers must serialize into the monitor chain, else a drag in the
    native editor snaps back on the next poll (the value never returns from the engine)."""
    band = {"threshold_db": -24.0, "ratio": 2.0}
    base = {"type": "mb_comp", "stem_id": "vox", "low": band, "mid": band,
            "high": band, "attack_ms": 10.0, "release_ms": 150.0}
    e = monitor_chain_params([dict(base, crossover_lo_hz=80.0,
                                   crossover_hi_hz=3000.0)])["entries"][0]
    assert e["crossover_lo_hz"] == 80.0 and e["crossover_hi_hz"] == 3000.0
    d = monitor_chain_params([base])["entries"][0]          # absent -> the default split
    assert d["crossover_lo_hz"] == 150.0 and d["crossover_hi_hz"] == 2000.0
    # per-band upward-compression mode must round-trip too (same snap-back risk)
    up = monitor_chain_params([{**base, "low": {**band, "up": True}}])["entries"][0]
    assert up["low"]["up"] is True and up["mid"]["up"] is False


# ---------- width (bit-exact: memoryless M/S) ------------------------------

@pytest.mark.parametrize("width", [0.0, 0.5, 1.5, 2.0])
def test_width_matches_dsp_bit_exact(width):
    """M/S width is memoryless — the monitor reference must equal dsp._width
    (a tight null, the bit-matched tier). This is the user's own channel-strip
    width control, now heard live instead of only in the printed render."""
    sig = _sig(seed=8, amp=0.4)
    oracle = dsp._width(sig.copy(), width)
    ref = rm.render_monitor(sig, SR, {
        "entries": [{"type": "width", "width": width}], "print_only": []})
    assert _err_db(oracle, ref) < -100.0, f"width {width}: {_err_db(oracle, ref):.1f} dB"


def test_width_mono_is_noop():
    """No side signal on mono — width passes through (matches dsp._width)."""
    sig = _sig(seed=9, amp=0.4, ch=1)
    out = rm.render_monitor(sig, SR, {
        "entries": [{"type": "width", "width": 1.8}], "print_only": []})
    assert _err_db(sig, out) < -120.0


# ---------- de-ess + transient (perceptual; documented approximations) -----

def test_deess_reduces_sibilance_and_is_bounded():
    """Split-band de-ess: a bright/sibilant signal loses high-band energy; output
    is bounded and finite (perceptual approximation of dsp._deess, not a tight null)."""
    from numpy.fft import rfft, rfftfreq
    sig = _sig(seed=10, amp=0.3)
    t = np.arange(sig.shape[0]) / SR
    sib = (0.4 * np.sin(2 * np.pi * 7000.0 * t)).astype(np.float64)
    sig = sig + np.stack([sib, sib], axis=1)
    out = rm.render_monitor(sig, SR, {
        "entries": [{"type": "deess", "threshold_db": -30.0, "freq_hz": 5000.0}],
        "print_only": []})
    assert np.all(np.isfinite(out))
    assert float(np.max(np.abs(out))) <= float(np.max(np.abs(sig))) + 1e-6
    freqs = rfftfreq(sig.shape[0], 1.0 / SR)
    hi = freqs > 5000.0
    energy = lambda a: float(np.mean(np.abs(rfft(a[:, 0]))[hi]))
    assert energy(out) < energy(sig)             # sibilance reduced


def test_transient_v2_speed_output_and_guardrail():
    """Transient v2: speed=1 is bit-identical to the legacy detector; a different
    speed changes the shaping; output_db is a plain trim; the contract carries the
    new fields with back-compat defaults; positive output charges the gain budget."""
    from mix_engine import dsp
    from mix_engine.guardrails import validate_proposal
    from mix_engine.models import ProposalRequest, TransientAction
    t = np.arange(SR) / SR
    env = np.exp(-((t % 0.25) * 30.0))
    sig = np.repeat((0.5 * env * np.sin(2 * np.pi * 220.0 * t))[:, None], 2, axis=1)
    base = {"type": "transient", "attack_db": 6.0, "sustain_db": 0.0}
    legacy = dsp._transient(sig, SR, base)
    assert np.array_equal(legacy, dsp._transient(sig, SR, {**base, "speed": 1.0,
                                                           "output_db": 0.0}))
    assert not np.array_equal(legacy, dsp._transient(sig, SR, {**base, "speed": 4.0}))
    trimmed = dsp._transient(sig, SR, {**base, "output_db": -6.0})
    rms = lambda a: float(np.sqrt(np.mean(np.asarray(a) ** 2)))
    assert 20.0 * np.log10(rms(trimmed) / rms(legacy)) == pytest.approx(-6.0, abs=0.1)

    out = monitor_chain_params(
        [{"type": "transient", "stem_id": "d", "attack_db": 3.0, "sustain_db": -2.0,
          "speed": 2.0, "output_db": -1.5}])
    tr = next(e for e in out["entries"] if e["type"] == "transient")
    assert tr["speed"] == 2.0 and tr["output_db"] == -1.5
    legacy_c = monitor_chain_params(
        [{"type": "transient", "stem_id": "d", "attack_db": 3.0, "sustain_db": 0.0}])
    lt = next(e for e in legacy_c["entries"] if e["type"] == "transient")
    assert lt["speed"] == 1.0 and lt["output_db"] == 0.0
    ref = rm.render_monitor(sig, SR, out)
    assert np.all(np.isfinite(ref))

    mk = lambda o: ProposalRequest(
        rationale="x", predicted_impact=[],
        actions=[TransientAction(stem_id="drums", attack_db=3.0, output_db=o)])
    # 4 dB already spent this pass + a 4 dB output trim = 8 dB > the 6 dB/pass cap.
    deep = validate_proposal(mk(4.0), {"drums": 4.0})
    assert any(v.rule == "gain_per_stem_per_pass_db" for v in deep)
    # an output CUT never charges; the +3 dB attack lift does (2026-10-05, reader D9·R6)
    assert validate_proposal(mk(-6.0), {"drums": 3.0}) == []
    assert any(v.rule == "gain_per_stem_per_pass_db" for v in validate_proposal(mk(-6.0), {"drums": 4.0}))


def _deess_test_sig():
    """A 200 Hz bed + pulsed 8 kHz esses, stereo (channels, frames) float32 —
    the de-ess v2 test fixture."""
    t = np.arange(SR) / SR
    bed = 0.2 * np.sin(2 * np.pi * 200.0 * t)
    ess = np.where((t % 0.25) < 0.05, 0.4, 0.0) * np.sin(2 * np.pi * 8000.0 * t)
    sig = (bed + ess).astype(np.float32)
    return np.stack([sig, sig])


def _band_rms_db(a, lo, hi):
    x = np.asarray(a, np.float64).mean(axis=0)
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(x.size, 1.0 / SR)
    m = (f >= lo) & (f < hi)
    return 20.0 * np.log10(float(np.sqrt(np.mean(np.abs(spec[m]) ** 2))) + 1e-12)


def test_deess_v2_range_caps_and_wide_dips_the_whole_signal():
    """De-ess v2 offline: split mode cuts only the sibilant band (the low bed is
    untouched); wide mode dips the bed too; range_db caps the reduction depth;
    the render is deterministic."""
    from mix_engine import dsp
    audio = _deess_test_sig()
    base = {"type": "deess", "threshold_db": -30.0, "freq_hz": 5000.0}
    split = dsp._deess(audio, SR, base)
    assert _band_rms_db(split, 6e3, 10e3) - _band_rms_db(audio, 6e3, 10e3) < -3.0
    assert abs(_band_rms_db(split, 150, 300) - _band_rms_db(audio, 150, 300)) < 0.1
    wide = dsp._deess(audio, SR, {**base, "wide": True})
    assert _band_rms_db(wide, 150, 300) - _band_rms_db(audio, 150, 300) < -1.5
    ranged = dsp._deess(audio, SR, {**base, "range_db": -3.0})
    assert (_band_rms_db(ranged, 6e3, 10e3) - _band_rms_db(audio, 6e3, 10e3)
            > -3.5)                                        # depth capped near -3
    assert np.array_equal(split, dsp._deess(audio, SR, base))   # deterministic


def test_deess_v2_contract_carries_new_fields_and_reference_renders():
    """monitor_chain_params passes the freed dynamics + range/wide through (with
    back-compat defaults for pre-v2 entries); reference_monitor renders them."""
    out = monitor_chain_params(
        [{"type": "deess", "stem_id": "v", "threshold_db": -24.0, "freq_hz": 6000.0,
          "ratio": 8.0, "attack_ms": 2.0, "release_ms": 80.0,
          "range_db": -9.0, "wide": True}])
    d = next(e for e in out["entries"] if e["type"] == "deess")
    assert d["ratio"] == 8.0 and d["range_db"] == -9.0 and d["wide"] is True
    legacy = monitor_chain_params(
        [{"type": "deess", "stem_id": "v", "threshold_db": -24.0,
          "freq_hz": 6000.0}])                             # a pre-v2 session entry
    ld = next(e for e in legacy["entries"] if e["type"] == "deess")
    assert ld["ratio"] == 4.0 and ld["range_db"] == -80.0 and ld["wide"] is False
    ref = rm.render_monitor(_deess_test_sig().T.astype(np.float64), SR, out)
    assert np.all(np.isfinite(ref))


def test_deess_ratio_guardrailed_for_agent():
    """De-ess v2: the agent's ratio obeys the comp dynamics ceiling (10:1); the
    schema still allows up to 20 for the user's hand knob."""
    from mix_engine.guardrails import validate_proposal
    from mix_engine.models import DeessAction, ProposalRequest
    mk = lambda r: ProposalRequest(
        rationale="x", predicted_impact=[],
        actions=[DeessAction(stem_id="vox", threshold_db=-24.0, freq_hz=6000.0,
                             ratio=r)])
    assert validate_proposal(mk(8.0), {}) == []
    hot = validate_proposal(mk(16.0), {})
    assert any(v.rule == "comp_ratio_max" for v in hot)


def test_transient_shapes_attack_and_is_bounded():
    """Attack/sustain shaper: boosting attack raises a percussive onset's peak,
    cutting it lowers the peak; gain stays within the ±range, output is finite."""
    n = int(SR * 0.4)
    t = np.arange(n) / SR
    click = (np.exp(-t * 25.0) * np.sin(2 * np.pi * 200.0 * t)).astype(np.float64)
    sig = np.stack([click, click], axis=1)
    boosted = rm.render_monitor(sig, SR, {
        "entries": [{"type": "transient", "attack_db": 6.0, "sustain_db": 0.0}],
        "print_only": []})
    cut = rm.render_monitor(sig, SR, {
        "entries": [{"type": "transient", "attack_db": -6.0, "sustain_db": 0.0}],
        "print_only": []})
    assert np.all(np.isfinite(boosted)) and np.all(np.isfinite(cut))
    onset = slice(0, int(SR * 0.02))
    sig_pk = float(np.max(np.abs(sig[onset])))
    assert float(np.max(np.abs(boosted[onset]))) > sig_pk      # onset emphasised
    assert float(np.max(np.abs(cut[onset]))) < sig_pk          # onset softened
    assert float(np.max(np.abs(boosted))) <= float(np.max(np.abs(sig))) * 2.05  # ~+6 dB cap


# ---------- band_width (perceptual; min-phase bandpass approximation) ------

def test_band_width_widens_and_narrows_its_band():
    """Per-band width: widening a band raises its side-channel energy in that band,
    narrowing toward mono lowers it; output finite (perceptual approximation of
    dsp._band_width_scale)."""
    from numpy.fft import rfft, rfftfreq
    sig = _sig(seed=11, amp=0.3)
    common = {"freq_hz": 2828.0, "q": 1.41}   # presence band (2–4 kHz)
    wide = rm.render_monitor(sig, SR, {
        "entries": [{"type": "band_width", "width": 2.0, **common}], "print_only": []})
    narrow = rm.render_monitor(sig, SR, {
        "entries": [{"type": "band_width", "width": 0.0, **common}], "print_only": []})
    assert np.all(np.isfinite(wide)) and np.all(np.isfinite(narrow))
    freqs = rfftfreq(sig.shape[0], 1.0 / SR)
    band = (freqs > 2000.0) & (freqs < 4000.0)
    side = lambda a: float(np.mean(np.abs(rfft(a[:, 0] - a[:, 1])[band])))
    assert side(wide) > side(sig)        # widened band -> more side energy
    assert side(narrow) < side(sig)      # mono'd band -> less


def test_imager_isolates_bands_and_reference_mirrors():
    """Imager: ONE LR8 law in both engines since 2026-10-21 (the imager audit; it was an offline
    FFT split against a native LR4) - mono'd lows lose their side below the low crossover (about
    40 dB, not all of it: a causal split), widened highs gain it above the high one, the mids keep
    theirs; all-unity widths keep every level (the split's allpass, not a bit-identical pass). The
    reference monitor runs the same numbers."""
    from mix_engine import dsp
    rng = np.random.default_rng(21)
    sig = rng.standard_normal((SR, 2)) * 0.2
    e = {"type": "imager", "width_low": 0.0, "width_mid": 1.0, "width_high": 1.5,
         "crossover_lo_hz": 150.0, "crossover_hi_hz": 2000.0}

    def side_energy(a, lo, hi):
        s = np.asarray(a[:, 0] - a[:, 1], np.float64)
        spec = np.fft.rfft(s)
        f = np.fft.rfftfreq(s.size, 1.0 / SR)
        m = (f >= lo) & (f < hi)
        return float(np.sqrt(np.mean(np.abs(spec[m]) ** 2)))

    out = dsp._imager(sig, SR, e)
    assert side_energy(out, 30, 100) < side_energy(sig, 30, 100) * 10 ** (-30 / 20)   # lows mono'd
    assert side_energy(out, 400, 1000) == pytest.approx(side_energy(sig, 400, 1000), rel=0.03)
    assert side_energy(out, 4000, 10000) == pytest.approx(side_energy(sig, 4000, 10000) * 1.5, rel=0.03)
    unity = dsp._imager(sig, SR, {"type": "imager"})
    for lo, hi in ((30, 150), (150, 2000), (2000, 20000)):
        assert side_energy(unity, lo, hi) == pytest.approx(side_energy(sig, lo, hi), rel=0.01)

    resolved = monitor_chain_params([{**e, "stem_id": "m"}])
    ie = next(x for x in resolved["entries"] if x["type"] == "imager")
    assert ie["width_low"] == 0.0 and ie["crossover_hi_hz"] == 2000.0
    ref = rm.render_monitor(sig, SR, resolved)
    assert np.all(np.isfinite(ref))
    np.testing.assert_allclose(ref, dsp._imager(sig, SR, e), rtol=0, atol=1e-12)   # one law


def test_imager_crossed_crossovers_stay_complementary():
    """Crossed crossovers (lo >= hi — reachable via the per-param co-edit clamps,
    which don't cross-check the pair) must NOT double-count the overlap into two
    bands: the guard re-orders hi to lo*1.5 (the _multiband_comp convention), so
    total energy stays bounded and the reference LR4 mirror agrees."""
    from mix_engine import dsp
    rng = np.random.default_rng(22)
    sig = rng.standard_normal((SR, 2)) * 0.2
    e = {"type": "imager", "width_low": 1.0, "width_mid": 1.0, "width_high": 1.2,
         "crossover_lo_hz": 3000.0, "crossover_hi_hz": 250.0}   # crossed on purpose
    out = dsp._imager(sig, SR, e)
    assert np.all(np.isfinite(out))
    # Without the guard the 250-3000 Hz bins land in BOTH the low and high band
    # (~2x amplitude: 1.36x total RMS on this signal). With it, ~1.08x (the 1.2
    # width's share) — assert well under the unguarded figure.
    assert float(np.sqrt(np.mean(out**2))) < float(np.sqrt(np.mean(sig**2))) * 1.2
    ref = rm._imager(sig, SR, e)
    assert np.all(np.isfinite(ref))
    assert float(np.sqrt(np.mean(ref**2))) < float(np.sqrt(np.mean(sig**2))) * 1.2


def test_imager_session_roundtrip_and_replace(demo_project):
    """The imager is user-addable on stems + master, DEVICE_PARAMS edits its flat
    width/crossover keys, and an agent re-proposal REPLACES the existing one."""
    from mix_engine.session import Session
    s = Session(demo_project)
    stem = next(iter(s._state["stems"]))
    s.add_stem_device(stem, "imager")
    assert any(e["type"] == "imager" for e in s._state["stems"][stem]["chain"])
    s.add_master_device("imager")
    assert any(e["type"] == "imager" for e in s._state["master_chain"])
    out = s.set_master_device_param("imager", 0, "width_low", 0.2, target=stem)
    assert out["value"] == 0.2
    entry = next(e for e in s._state["stems"][stem]["chain"] if e["type"] == "imager")
    assert entry["width_low"] == 0.2 and "low" not in entry   # FLAT key, no nesting
    from mix_engine.models import ImagerAction
    s._mutate(ImagerAction(stem_id=stem, width_high=1.4))
    imagers = [e for e in s._state["stems"][stem]["chain"] if e["type"] == "imager"]
    assert len(imagers) == 1 and imagers[0]["width_high"] == 1.4   # replaced, not stacked


def test_band_width_mono_is_noop():
    sig = _sig(seed=12, amp=0.3, ch=1)
    out = rm.render_monitor(sig, SR, {
        "entries": [{"type": "band_width", "freq_hz": 2828.0, "q": 1.41, "width": 1.8}],
        "print_only": []})
    assert _err_db(sig, out) < -120.0


def test_band_width_contract_resolves_label_to_bandpass():
    """Contract: the band LABEL resolves to a bandpass center+Q the native engine
    consumes (it has no band names), heard live (not print_only)."""
    out = monitor_chain_params([{"type": "band_width", "stem_id": "vox",
                                 "band": "presence", "width": 1.5}])
    (e,) = out["entries"]
    assert e["type"] == "band_width" and e["width"] == 1.5 and e["enabled"] is True
    assert 2000 < e["freq_hz"] < 4000 and e["q"] > 0    # presence geometric center
    assert out["print_only"] == []


# ---------- master-bus saturation + width (Piece A) -----------------------

def test_master_saturation_width_render_matches_monitor():
    """Saturation + width on the MASTER null between the offline master render
    (dsp._apply_master_chain) and the live-monitor reference — both memoryless,
    bit-exact. This is what makes the master-bus add tiles 'like Ozone'."""
    from mix_engine.monitor_chain import monitor_chain_params
    sig = _sig(seed=20, amp=0.3)
    chain = [{"type": "saturation", "drive_db": 6.0},
             {"type": "width", "width": 1.6}]
    oracle = dsp._apply_master_chain(sig.copy(), SR, chain, report=None)
    ref = rm.render_monitor(sig, SR, monitor_chain_params(chain))
    assert _err_db(oracle, ref) < -100.0


def test_master_color_devices_add_edit_and_order(demo_project):
    """Piece A/B: the user adds master-bus saturation / width / mb_comp / match_eq
    by hand; primary knobs edit + clamp; render order follows the user's add/arrange
    order (device freedom, Phase 43), with the limiter pinned last (the ceiling)."""
    from mix_engine.session import Session, effective_master_chain
    from mix_engine.monitor_chain import monitor_chain_params
    s = Session(demo_project)

    for dev in ("saturation", "width", "mb_comp"):
        s.add_master_device(dev)
        assert any(e["type"] == dev for e in s._state["master_chain"])
    # match_eq is intentionally rejected as a user add (reference-driven, dead tile)
    with pytest.raises(ValueError):
        s.add_master_device("match_eq")

    pick = lambda t: [e for e in s._state["master_chain"] if e["type"] == t][0]
    s.set_master_param("saturation", 0, 8.0)
    assert pick("saturation")["drive_db"] == 8.0
    s.set_master_param("saturation", 0, 99.0)          # clamp to the 12 dB ceiling
    assert pick("saturation")["drive_db"] == 12.0
    s.set_master_param("width", 0, 1.7)
    assert pick("width")["width"] == 1.7
    s.set_master_param("mb_comp", 0, 4.0)              # primary knob = mid-band ratio
    assert pick("mb_comp")["mid"]["ratio"] == 4.0

    s.add_master_device("limiter")
    order = [e["type"] for e in effective_master_chain(s._state)]
    # device freedom (Phase 43): the chain renders in the order the user ADDED
    # (saturation, width, mb_comp), the limiter pinned last (the ceiling). The user
    # re-sorts via set_master_device_order — see test_master_device_order_user_reorder.
    assert order == ["gain", "saturation", "width", "mb_comp", "limiter"]
    # saturation + width are heard live; a flat (0 dB) match_eq is a no-op, so the
    # monitor correctly emits no entry until its per-band gains are set.
    mc = monitor_chain_params(effective_master_chain(s._state))
    types = [e["type"] for e in mc["entries"]]
    assert "saturation" in types and "width" in types
    assert mc["print_only"] == []


def test_add_stem_color_device(demo_project):
    """Piece C: the user hand-places color (transient/de-ess/band_width) on a captured
    track via add_stem_device; it lands in the chain, resolves to a live monitor entry,
    edits/clamps via the co-edit param path, and removes. Phase 43 device freedom: these
    now also render on the master (add_master_device), so the master rail accepts them too."""
    from mix_engine.session import Session, effective_chain
    from mix_engine.monitor_chain import monitor_chain_params
    s = Session(demo_project)
    stem = next(iter(s._state["stems"]))

    for dev in ("transient", "deess", "band_width"):
        s.add_stem_device(stem, dev)
        assert any(e["type"] == dev for e in s._state["stems"][stem]["chain"])
    s.add_master_device("transient")              # device freedom: color on the master too
    assert any(e["type"] == "transient" for e in s._state["master_chain"])
    with pytest.raises(ValueError):
        s.add_stem_device("nope", "transient")    # unknown stem still rejected

    s.set_master_param("transient", 0, 4.0, target=stem)   # edit on the stem
    tr = [e for e in s._state["stems"][stem]["chain"] if e["type"] == "transient"][0]
    assert tr["attack_db"] == 4.0
    s.set_master_param("transient", 0, 99.0, target=stem)  # clamp to +6
    assert tr["attack_db"] == 6.0

    mc = monitor_chain_params(effective_chain(s._state["stems"][stem]))
    types = [e["type"] for e in mc["entries"]]
    assert {"transient", "deess", "band_width"} <= set(types)   # all heard live
    assert mc["print_only"] == []

    s.remove_stem_device(stem, "deess", 0)
    assert not [e for e in s._state["stems"][stem]["chain"] if e["type"] == "deess"]


# ---------- the param contract --------------------------------------------

def test_set_master_param_user_coedit(demo_project):
    """Phase 40 co-edit: the user channel edits master-chain device params directly
    (no proposal), clamped to each control's range; a bad address raises."""
    from mix_engine.session import Session
    s = Session(demo_project)

    r = s.set_master_param("gain", 0, -3.0)          # master gain (special-cased)
    assert r["value"] == -3.0 and s._state["master_gain_db"] == -3.0

    s._state["master_chain"].append(                 # a chain device to edit
        {"type": "eq", "stem_id": "master",
         "band": {"shape": "peak", "freq_hz": 1000.0, "gain_db": 0.0, "q": 1.0}})
    s.set_master_param("eq", 0, 4.0)
    assert s._state["master_chain"][-1]["band"]["gain_db"] == 4.0
    s.set_master_param("eq", 0, 99.0)                # clamp to the shared ±12 knob range
    # ±12 like the stem eq + DeviceSpecs.h (2026-09-01 sweep): the master clamped at ±6
    # while the native knob reached ±12, so a master-EQ knob past 6 dB snapped back. The
    # +6 boost bound is the AGENT's guardrail, not the user's.
    assert s._state["master_chain"][-1]["band"]["gain_db"] == 12.0

    with pytest.raises(ValueError):
        s.set_master_param("eq", 5, 0.0)             # no 6th eq on the chain


def test_add_remove_master_device_user_coedit(demo_project):
    """Phase 40 co-edit: the user adds/removes master devices directly (no proposal).
    Defaults match the render shape, the freshly added device is immediately editable,
    render order stays canonical, and there is at most one limiter."""
    from mix_engine.session import Session, effective_master_chain
    s = Session(demo_project)

    r = s.add_master_device("eq")
    assert r == {"type": "eq", "occurrence": 0}
    s.set_master_param("eq", 0, 3.0)                 # editable immediately
    assert s._state["master_chain"][-1]["band"]["gain_db"] == 3.0

    s.add_master_device("comp")
    s.add_master_device("limiter")
    with pytest.raises(ValueError):
        s.add_master_device("limiter")               # one limiter max
    with pytest.raises(ValueError):
        s.add_master_device("reverb")                # unknown device

    order = [e["type"] for e in effective_master_chain(s._state)]
    assert order == ["gain", "eq", "comp", "limiter"]   # canonical, not insertion order

    s.add_master_device("dynamic_eq")                # default is a valid cut (gain != 0)
    assert any(e["type"] == "dynamic_eq" for e in s._state["master_chain"])

    s.remove_master_device("comp", 0)
    assert not [e for e in s._state["master_chain"] if e["type"] == "comp"]
    with pytest.raises(ValueError):
        s.remove_master_device("comp", 0)            # already gone


def test_set_master_device_param_full_config(demo_project):
    """Phase 40 device detail popup: the user sets ANY editable param (not just the
    primary knob), clamped; eq params resolve under band.*; bad params raise. This
    is what lets each dynamic EQ target a different frequency."""
    from mix_engine.session import Session
    s = Session(demo_project)

    s.add_master_device("dynamic_eq")
    s.set_master_device_param("dynamic_eq", 0, "freq_hz", 800.0)
    de = [e for e in s._state["master_chain"] if e["type"] == "dynamic_eq"][0]
    assert de["freq_hz"] == 800.0                    # off the 3 kHz default
    s.set_master_device_param("dynamic_eq", 0, "freq_hz", 99999.0)
    assert de["freq_hz"] == 20000.0                  # clamped to the band ceiling (user co-edit: full 20 Hz–20 kHz)

    s.add_master_device("eq")                        # eq params live under band.*
    s.set_master_device_param("eq", 0, "freq_hz", 5000.0)
    assert [e for e in s._state["master_chain"]
            if e["type"] == "eq"][0]["band"]["freq_hz"] == 5000.0

    with pytest.raises(ValueError):
        s.set_master_device_param("comp", 0, "freq_hz", 100.0)   # comp has no freq


def test_output_and_makeup_gains_user_coedit(demo_project):
    """Phase 40 co-edit: the newly-surfaced output/makeup gains are editable and
    clamped — mb_comp per-band makeup (nested under low/mid/high), saturation +
    clip output trim, and the limiter's loudness target (its output gain)."""
    from mix_engine.session import Session
    s = Session(demo_project)

    s.add_master_device("mb_comp")
    s.set_master_device_param("mb_comp", 0, "low_makeup_db", 4.0)
    s.set_master_device_param("mb_comp", 0, "high_makeup_db", 99.0)   # clamp to +12
    mb = [e for e in s._state["master_chain"] if e["type"] == "mb_comp"][0]
    assert mb["low"]["makeup_db"] == 4.0 and mb["high"]["makeup_db"] == 12.0

    s.add_master_device("saturation")
    s.set_master_device_param("saturation", 0, "output_db", -6.0)
    assert [e for e in s._state["master_chain"]
            if e["type"] == "saturation"][0]["output_db"] == -6.0

    s.add_master_device("clip")
    s.set_master_device_param("clip", 0, "output_db", 99.0)          # clamp to +12
    assert [e for e in s._state["master_chain"]
            if e["type"] == "clip"][0]["output_db"] == 12.0

    s.add_master_device("limiter")
    s.set_master_device_param("limiter", 0, "target_lufs_i", -14.0)
    lim = [e for e in s._state["master_chain"] if e["type"] == "limiter"][0]
    assert lim["target_lufs_i"] == -14.0
    s.set_master_device_param("limiter", 0, "target_lufs_i", -99.0)  # clamp to -24
    assert lim["target_lufs_i"] == -24.0


def test_limiter_loudness_target_round_trips_to_editor():
    """The loudness target must serialize into the limiter's monitor entry (display
    only — the native DSP drives off makeup_db) so the 'Loud' knob doesn't snap back
    on the next poll; it's omitted when unset."""
    with_target = monitor_chain_params(
        [{"type": "limiter", "stem_id": "master", "ceiling_dbtp": -1.0,
          "target_lufs_i": -12.0}], limiter_makeup_db=2.0)
    lim = next(e for e in with_target["entries"] if e["type"] == "limiter")
    assert lim["target_lufs_i"] == -12.0 and lim["makeup_db"] == 2.0
    unset = monitor_chain_params(
        [{"type": "limiter", "stem_id": "master", "ceiling_dbtp": -1.0}])
    assert "target_lufs_i" not in next(
        e for e in unset["entries"] if e["type"] == "limiter")


def test_limiter_makeup_resolves_from_pre_lufs_live():
    """The Loud knob must move the monitor without a re-render: the makeup is
    (target - pre_lufs) because the pre-limiter loudness is target-independent. The
    cached render makeup is ignored once we have the pre-LUFS; pre_limiter_lufs_i
    is forwarded so the native editor resolves the knob locally too."""
    out = monitor_chain_params(
        [{"type": "limiter", "stem_id": "master", "ceiling_dbtp": -1.0,
          "target_lufs_i": -10.0}], limiter_makeup_db=99.0, limiter_pre_lufs_i=-16.0)
    lim = next(e for e in out["entries"] if e["type"] == "limiter")
    assert lim["makeup_db"] == 6.0              # -10 - (-16); ignores the stale 99
    assert lim["pre_limiter_lufs_i"] == -16.0
    # No target -> use the passed makeup as-is, but still carry pre_limiter_lufs_i
    # so the editor resolves the FIRST Loud-knob turn instantly (bootstrap).
    none = monitor_chain_params(
        [{"type": "limiter", "stem_id": "master", "ceiling_dbtp": -1.0}],
        limiter_makeup_db=1.5, limiter_pre_lufs_i=-16.0)
    nlim = next(e for e in none["entries"] if e["type"] == "limiter")
    assert nlim["makeup_db"] == 1.5 and nlim["pre_limiter_lufs_i"] == -16.0
    assert "target_lufs_i" not in nlim


def test_eq_shape_and_device_bypass(demo_project):
    """Phase 40 'finish the devices': the user changes an EQ filter shape, and
    bypasses a device so BOTH the render and the live-monitor contract skip it
    while the tile still shows (the flag rides on the entry)."""
    from mix_engine.session import Session, effective_master_chain
    from mix_engine.monitor_chain import monitor_chain_params
    s = Session(demo_project)

    s.add_master_device("eq")
    s.set_eq_shape(0, "high_shelf")
    assert [e for e in s._state["master_chain"]
            if e["type"] == "eq"][0]["band"]["shape"] == "high_shelf"
    s.set_eq_shape(0, "notch")                        # notch is a valid shape
    assert [e for e in s._state["master_chain"]
            if e["type"] == "eq"][0]["band"]["shape"] == "notch"
    with pytest.raises(ValueError):
        s.set_eq_shape(0, "bogus")                    # unknown shape

    s.add_master_device("comp")
    s.set_master_device_enabled("comp", 0, False)
    assert [e for e in s._state["master_chain"]
            if e["type"] == "comp"][0]["enabled"] is False
    # the monitor contract carries the flag (entry still present, so the tile shows)
    mc = monitor_chain_params(effective_master_chain(s._state))
    comp_entry = [e for e in mc["entries"] if e["type"] == "comp"][0]
    assert comp_entry["enabled"] is False
    s.set_master_device_enabled("comp", 0, True)
    assert [e for e in s._state["master_chain"]
            if e["type"] == "comp"][0]["enabled"] is True
    with pytest.raises(ValueError):
        s.set_master_device_enabled("comp", 9, False)   # no such occurrence


def test_master_device_order_user_reorder(demo_project):
    """Phase 42 drag-to-reorder + Phase 43 device freedom: the user re-sorts the master
    chain. Order is honored; an UNPLACED limiter sits canonical-last, but the user can now
    place it ANYWHERE (Stage 3 — no longer pinned last). Unknown types are dropped."""
    from mix_engine.session import Session, effective_master_chain
    s = Session(demo_project)
    s._state["master_chain"] = []
    for dev in ("eq", "comp", "saturation", "limiter"):
        s.add_master_device(dev)

    # default = canonical: gain, eq, comp, saturation, limiter
    assert [e["type"] for e in effective_master_chain(s._state)] == \
        ["gain", "eq", "comp", "saturation", "limiter"]

    # user drag: saturation -> comp -> eq (limiter stays pinned last)
    s.set_master_device_order(["saturation", "comp", "eq"])
    assert [e["type"] for e in effective_master_chain(s._state)] == \
        ["gain", "saturation", "comp", "eq", "limiter"]

    # device freedom (Stage 3): the user CAN place the limiter first now — it renders at its
    # chain position, no longer pinned last (the ceiling comes from the limiter wherever it sits
    # + the export hard-clip fallback). gain still stays first (synthesized).
    s.set_master_device_order(["limiter", "eq", "comp", "saturation"])
    assert [e["type"] for e in effective_master_chain(s._state)] == \
        ["gain", "limiter", "eq", "comp", "saturation"]

    # unknown types are ignored (never placed in the chain order)
    result = s.set_master_device_order(["bogus", "comp"])
    assert "bogus" not in result["master_device_order"]
    assert "bogus" not in [e["type"] for e in s._state["master_chain"]]


def test_master_device_order_dedups_repeated_type(demo_project):
    """A repeated type in the order must not double-render its entries."""
    from mix_engine.session import Session, effective_master_chain
    s = Session(demo_project)
    s._state["master_chain"] = []
    # Built directly, not via add_master_device: since 2026-08-06 the co-edit path refuses a
    # second comp (the native hosts one CompParams per chain, so it would render offline and
    # not in the monitor — see _NATIVE_SINGLETON_KINDS). The ordering logic under test here
    # still has to cope with a repeated type, e.g. from a legacy session.
    s._state["master_chain"] = [
        {"type": "comp", "threshold_db": -18.0, "ratio": 2.0},
        {"type": "comp", "threshold_db": -12.0, "ratio": 3.0},
    ]
    s.set_master_device_order(["comp", "comp"])   # duplicated type token
    types = [e["type"] for e in effective_master_chain(s._state)]
    assert types.count("comp") == 2      # both comps once each, not four


def test_device_order_preserves_an_interleaved_type(demo_project):
    """`eq, clip, eq` must survive the order channel.

    It did not. Both ends grouped a chain by TYPE and the wire between them carried a
    de-duplicated type list, so this chain came back as `eq, eq, clip` — the clipper moved in
    front of the second EQ and the render shifted 3.95 dB at peak. Three ordinary clicks in the
    dock reached it, because the dock re-posts the order on every device ADD and REMOVE, not
    only on a drag (dock audit 2026-09-10, P0).

    The test above this one — count("comp") == 2 on two ADJACENT comps — was green throughout:
    a chain that is already grouped cannot detect regrouping. Interleaving is the whole subject.
    """
    from mix_engine.session import Session, effective_master_chain
    s = Session(demo_project)
    s._state["master_chain"] = [
        {"type": "eq", "freq_hz": 100.0, "gain_db": 3.0},
        {"type": "clip", "ceiling_db": -6.0},
        {"type": "eq", "freq_hz": 8000.0, "gain_db": -2.0},
    ]
    s.set_master_device_order(["eq#0", "clip#0", "eq#1"])
    assert [e["type"] for e in effective_master_chain(s._state)] == ["gain", "eq", "clip", "eq"]
    # and the entries are the RIGHT way round, not merely the right shape
    assert [e.get("freq_hz") for e in s._state["master_chain"] if e["type"] == "eq"] == [100.0, 8000.0]

    # Negative control: the bare-type spelling still means what it always did — every entry of
    # that type as one block. Old sessions and callers that only know types keep working, and
    # this is the behaviour the occurrence tokens exist to escape.
    s._state["master_chain"] = [
        {"type": "eq", "freq_hz": 100.0},
        {"type": "clip", "ceiling_db": -6.0},
        {"type": "eq", "freq_hz": 8000.0},
    ]
    s.set_master_device_order(["eq", "clip"])
    assert [e["type"] for e in s._state["master_chain"]] == ["eq", "eq", "clip"]


def test_device_order_names_every_entry_including_box_less_ones(demo_project):
    """A device with no dock box — a master `transpose` — keeps its place through a dock order.

    The 09-10 dock audit read this as the dock's fault ("it posts an order that omits a device
    it does not draw") and this test hand-wrote the order `transpose#0, eq#0, …` as the fix. No
    producer emits that token: the dock builds its order from its NODE's chain, and the brain
    pushes `transpose` as `print_only`, never as an entry (full audit 2026-10-02, reader E2·R4 —
    the old "control" below was the order the app really posts, and it swept the transpose past
    the limiter: -0.366 dBTP against a stated -1.000). The consumer keeps it now
    (`session._dock_order`); `test_reader_p2_py2` drives the order `resolve_chains` lets the
    dock build. Here: both spellings end with the transpose ahead of the limiter.
    """
    from mix_engine.session import Session
    s = Session(demo_project)
    chain = [
        {"type": "transpose", "semitones": 2},
        {"type": "eq", "freq_hz": 100.0},
        {"type": "comp", "threshold_db": -18.0},
        {"type": "limiter", "ceiling_dbtp": -1.0},
    ]
    s._state["master_chain"] = list(chain)
    s.set_master_device_order(["transpose#0", "eq#0", "comp#0", "limiter#0"])
    assert [e["type"] for e in s._state["master_chain"]] == [
        "transpose", "eq", "comp", "limiter"], "a complete order keeps transpose ahead of the limiter"

    # The order the dock posts: it cannot name the transpose, and the transpose stays put.
    s._state["master_chain"] = list(chain)
    s.set_master_device_order(["comp#0", "eq#0", "limiter#0"])
    assert [e["type"] for e in s._state["master_chain"]] == ["transpose", "comp", "eq", "limiter"]

    # The control that shows the test can tell the cases apart: a NAMEABLE device a partial
    # order leaves out still falls to its canonical slot, after everything named.
    s._state["master_chain"] = [{"type": "limiter", "ceiling_dbtp": -1.0},
                                {"type": "eq", "freq_hz": 100.0}]
    s.set_master_device_order(["limiter#0"])
    assert [e["type"] for e in s._state["master_chain"]] == ["limiter", "eq"]


def test_second_singleton_device_is_refused(demo_project):
    """The native holds ONE param set for these types and its step walk de-dupes, so a
    second entry renders offline but not in the monitor (measured 3.98 dB apart on two
    comps). Refuse it honestly rather than accept a silently divergent edit."""
    import pytest
    from mix_engine.session import Session

    s = Session(demo_project)
    s._state["master_chain"] = []
    s.add_master_device("comp")
    with pytest.raises(ValueError, match="one per chain"):
        s.add_master_device("comp")
    # multi-entry types are unaffected — an EQ bank is many bands by design
    s.add_master_device("eq")
    s.add_master_device("eq")
    assert sum(1 for e in s._state["master_chain"] if e["type"] == "eq") == 2


def test_monitor_chain_params_splits_live_and_print_only():
    chain = [
        {"type": "gain", "gain_db": -1.5},
        {"type": "eq", "band": {"shape": "peak", "freq_hz": 1000.0,
                                "gain_db": 2.0, "q": 1.0}},
        {"type": "match_eq", "stem_id": "master",
         "band_gains_db": {"presence": 3.0, "air": -4.0}},
        {"type": "transpose", "semitones": 2},
        {"type": "dynamic_eq", "stem_id": "master", "freq_hz": 3000.0, "q": 4.0,
         "threshold_db": -28.0, "ratio": 3.0, "attack_ms": 8.0,
         "release_ms": 90.0, "gain_db": -6.0},
        {"type": "comp", "threshold_db": -18.0, "ratio": 2.0, "attack_ms": 10.0,
         "release_ms": 120.0, "makeup_db": 0.0, "mix": 1.0},
        {"type": "mb_comp", "stem_id": "master",
         "low": {"threshold_db": -30.0, "ratio": 3.0, "makeup_db": 0.0},
         "mid": {"threshold_db": -28.0, "ratio": 2.0, "makeup_db": 0.0},
         "high": {"threshold_db": -26.0, "ratio": 4.0, "makeup_db": 0.0},
         "attack_ms": 12.0, "release_ms": 180.0},
        {"type": "limiter", "ceiling_dbtp": -1.0, "target_lufs_i": -14.0},
    ]
    out = monitor_chain_params(chain, limiter_makeup_db=3.2)
    assert [e["type"] for e in out["entries"]] == [
        "gain", "eq", "match_eq", "dynamic_eq", "comp", "mb_comp", "limiter"]
    assert set(out["print_only"]) == {"transpose"}
    me = next(e for e in out["entries"] if e["type"] == "match_eq")
    # Since 2026-09-08 the bank is FITTED to the offline FIR rather than fed the raw band
    # gains, so a request naming two bands emits bells on more than two: a band the request
    # omits still needs one to cancel its neighbours' bleed (device audit 04-F4). Assert what
    # the bank MEANS — its summed response at the requested centre — instead of how many
    # filters it happens to take to say it.
    import numpy as np

    from mix_engine.monitor_chain import MATCH_EQ_Q, _BAND_CENTER_HZ
    from mix_engine.reference_monitor import biquad_coeffs

    def _bank_db_at(bands, freq_hz, sr=48000):
        z = np.exp(-1j * 2.0 * np.pi * freq_hz / sr)
        total = 0.0
        for bd in bands:
            b, a = biquad_coeffs("peak", float(sr), bd["freq_hz"], bd["q"], bd["gain_db"])
            h = (b[0] + b[1] * z + b[2] * z ** 2) / (a[0] + a[1] * z + a[2] * z ** 2)
            total += 20.0 * np.log10(abs(h))
        return total

    assert 1 <= len(me["bands"]) <= 8
    assert all(abs(b["q"] - MATCH_EQ_Q) < 1e-9 for b in me["bands"])
    # the +3 dB the chain asked for, delivered by the bank as a whole
    assert abs(_bank_db_at(me["bands"], _BAND_CENTER_HZ["presence"]) - 3.0) < 0.6

    eq = out["entries"][1]
    assert eq["shape"] == "peak" and eq["freq_hz"] == 1000.0 and eq["q"] == 1.0
    deq = next(e for e in out["entries"] if e["type"] == "dynamic_eq")
    assert deq["freq_hz"] == 3000.0 and deq["gain_db"] == -6.0 and deq["ratio"] == 3.0
    mb = next(e for e in out["entries"] if e["type"] == "mb_comp")
    assert mb["low"]["ratio"] == 3.0 and mb["high"]["threshold_db"] == -26.0
    assert mb["attack_ms"] == 12.0
    lim = out["entries"][-1]
    assert lim["ceiling_dbtp"] == -1.0
    assert lim["makeup_db"] == pytest.approx(3.2)


# ---------- send/return wet effects (Phase 40 Step 2; bit-exact tier) -------
# The agent's reverb/delay returns (Phase 38). Unlike comp/de-ess these null
# *tightly*: the return DSP is a causal IIR recurrence (no lookahead) and the
# whole return chain is linear, so the reference transcribes dsp.py verbatim and
# a per-instance local wet send sums to the offline shared-return topology.

_VERB = {"type": "return_reverb", "decay_s": 2.0, "pre_delay_ms": 20.0, "tone": 0.5}
_DLY = {"type": "return_delay", "delay_ms": 220.0, "feedback": 0.45, "tone": 0.6}


@pytest.mark.parametrize("entry,fn", [
    (_DLY, "_return_delay"),
    (_VERB, "_return_reverb"),
])
def test_return_effect_reference_matches_dsp(entry, fn):
    """The numpy return reference is a verbatim port of dsp.py — bit-identical."""
    sig = _sig(seed=5, amp=0.3)
    ref = getattr(dsp, fn)(sig.copy(), SR, entry)
    got = getattr(rm, fn)(sig.copy(), SR, entry)
    assert _err_db(ref, got) < -120.0


def test_send_return_parallel_superposition_is_exact():
    """The per-track model is faithful: because the return chain is linear,
    summing each instance's own wet (DAW sum) equals the offline shared return
    fed by both sends. This is what makes per-instance returns bit-exact, not an
    approximation."""
    a, b = _sig(seed=1, amp=0.3), _sig(seed=2, amp=0.3)
    sa, sb = 10.0 ** (-6.0 / 20.0), 10.0 ** (-9.0 / 20.0)
    chain = [{"type": "gain", "gain_db": 0.0}, _DLY, _VERB]
    shared = dsp._apply_return_chain(a * sa + b * sb, SR, chain)
    per_instance = (rm._apply_return_chain(a * sa, SR, chain)
                    + rm._apply_return_chain(b * sb, SR, chain))
    assert _err_db(shared, per_instance) < -120.0


def test_send_return_render_matches_offline_apply():
    """End-to-end contract -> render: a send_return entry through
    monitor_chain_params + render_monitor equals the offline dry + parallel wet
    return (dsp._apply_return_chain). The send taps the post-chain signal."""
    sig = _sig(seed=7, amp=0.3)
    return_chain = [{"type": "gain", "gain_db": -3.0}, _DLY, _VERB]
    params = monitor_chain_params(
        [{"type": "send_return", "send_db": -6.0, "return": return_chain}])
    got = rm.render_monitor(sig, SR, params)
    send_lin = 10.0 ** (-6.0 / 20.0)
    ref = sig + dsp._apply_return_chain(sig * send_lin, SR, return_chain)
    assert _err_db(ref, got) < -120.0


def test_send_return_contract_resolves_and_skips_dry_only_return():
    """A wet return resolves to one send_return entry (with its delay/reverb
    params); a return with no wet effect emits nothing — and never print_only
    (it's synthetic, the agent's sends are always 'heard live')."""
    dry_only = monitor_chain_params(
        [{"type": "send_return", "send_db": -6.0,
          "return": [{"type": "gain", "gain_db": 0.0}]}])
    assert dry_only["entries"] == [] and dry_only["print_only"] == []

    wet = monitor_chain_params(
        [{"type": "send_return", "send_db": -6.0,
          "return": [{"type": "gain", "gain_db": -3.0}, _DLY]}])["entries"]
    assert len(wet) == 1 and wet[0]["type"] == "send_return"
    assert wet[0]["send_db"] == -6.0
    rd = next(r for r in wet[0]["return_entries"] if r["type"] == "return_delay")
    assert rd["delay_ms"] == 220.0 and rd["feedback"] == 0.45


def test_gateway_injects_send_return_for_a_sending_track():
    """The gateway folds a sending stem's wet return into the monitor chain. With
    several active tracks the insert plays just its own track + its sends (the DAW
    sums); a non-sending track gets none."""
    from orchestrator.gateway import _audition_chain
    state = {
        "master_gain_db": 0.0, "master_chain": [],
        "returns": {"verb": {"gain_db": 0.0, "chain": [
            {"type": "return_reverb", "return_id": "verb", "decay_s": 2.0,
             "pre_delay_ms": 20.0, "tone": 0.5}]}},
        "stems": {
            "vox": {"gain_db": 0.0, "pan": 0.0, "width": 1.0, "muted": False,
                    "excluded": False, "rides": [], "chain": [],
                    "sends": {"verb": -6.0}},
            "gtr": {"gain_db": 0.0, "pan": 0.0, "width": 1.0, "muted": False,
                    "excluded": False, "rides": [], "chain": [], "sends": {}},
        },
    }
    vox = [e for e in _audition_chain(state, "vox") if e["type"] == "send_return"]
    assert len(vox) == 1 and vox[0]["send_db"] == -6.0
    assert vox[0]["return"][-1]["type"] == "return_reverb"
    gtr = [e for e in _audition_chain(state, "gtr") if e["type"] == "send_return"]
    assert gtr == []


# ---------- pre-limiter clipper (Phase 41; memoryless, bit-exact tier) ------

@pytest.mark.parametrize("hardness", [0.0, 0.5, 1.0])
def test_master_clip_render_matches_monitor(hardness):
    """The master clipper nulls between the offline render (dsp._apply_master_chain ->
    _clip) and the live-monitor reference — memoryless, so bit-exact (like saturation)."""
    sig = _sig(seed=21, amp=0.6)
    chain = [{"type": "clip", "threshold_db": -3.0, "hardness": hardness}]
    oracle = dsp._apply_master_chain(sig.copy(), SR, chain, report=None)
    ref = rm.render_monitor(sig, SR, monitor_chain_params(chain))
    assert _err_db(oracle, ref) < -100.0


def test_clip_shaves_peaks():
    """Hard clip caps at the threshold; any clip shaves the peak below the input's."""
    sig = _sig(seed=22, amp=0.8)
    t = 10.0 ** (-6.0 / 20.0)
    hard = rm.render_monitor(sig, SR, monitor_chain_params(
        [{"type": "clip", "threshold_db": -6.0, "hardness": 1.0}]))
    assert float(np.max(np.abs(hard))) <= t + 1e-6                  # capped at threshold
    assert float(np.max(np.abs(hard))) < float(np.max(np.abs(sig)))  # peaks shaved


def test_clip_contract_resolves():
    out = monitor_chain_params([{"type": "clip", "stem_id": "master",
                                 "threshold_db": -4.0, "hardness": 0.7}])
    assert out["print_only"] == []
    c = next(e for e in out["entries"] if e["type"] == "clip")
    assert c["threshold_db"] == -4.0 and c["hardness"] == 0.7
    assert c["output_db"] == 0.0   # the output trim serializes (default neutral)


@pytest.mark.parametrize("output_db", [-4.0, 4.0])
def test_clip_output_trim_matches_dsp_bit_exact(output_db):
    """The clipper's output trim is a plain gain applied last — the offline render
    and the live-monitor reference still null (memoryless, like the rest of clip)."""
    sig = _sig(seed=23, amp=0.6)
    chain = [{"type": "clip", "threshold_db": -3.0, "hardness": 0.5,
              "output_db": output_db}]
    oracle = dsp._apply_master_chain(sig.copy(), SR, chain, report=None)
    ref = rm.render_monitor(sig, SR, monitor_chain_params(chain))
    assert _err_db(oracle, ref) < -100.0


def test_clip_v2_oversampling_reduces_true_peak_overshoot():
    """Clip v2: on a tone with inter-sample peaks, the naive hard clip overshoots
    the threshold true-peak-wise; the 4x oversampled path measurably reduces that
    overshoot. Length-preserving, deterministic, and oversample=1 stays
    bit-identical to the legacy memoryless path."""
    from scipy.signal import resample_poly
    from mix_engine import dsp
    t = np.arange(SR) / SR
    sig = np.repeat((0.9 * np.sin(2 * np.pi * 10937.0 * t + 0.3))[:, None], 2, axis=1)
    base = {"type": "clip", "threshold_db": -6.0, "hardness": 1.0, "output_db": 0.0}
    naive = dsp._clip(sig, base)
    os4 = dsp._clip(sig, {**base, "oversample": 4})
    tp = lambda a: 20.0 * np.log10(float(np.max(np.abs(resample_poly(a, 8, 1, axis=0)))))
    assert os4.shape == sig.shape
    assert tp(os4) < tp(naive) - 0.5                       # measurably less overshoot
    assert np.array_equal(naive, dsp._clip(sig, {**base, "oversample": 1}))  # 1 == legacy
    assert np.array_equal(os4, dsp._clip(sig, {**base, "oversample": 4}))    # deterministic


def test_clip_v2_contract_and_model_bounds():
    """The chain contract carries oversample (default 1 for pre-v2 entries), the
    reference monitor renders the oversampled entry, and the model rejects
    non-power-of-2 factors."""
    from mix_engine.models import ClipAction
    out = monitor_chain_params(
        [{"type": "clip", "stem_id": "master", "threshold_db": -4.0,
          "hardness": 0.7, "oversample": 4}])
    c = next(e for e in out["entries"] if e["type"] == "clip")
    assert c["oversample"] == 4
    legacy = monitor_chain_params(
        [{"type": "clip", "stem_id": "master", "threshold_db": -4.0}])
    lc = next(e for e in legacy["entries"] if e["type"] == "clip")
    assert lc["oversample"] == 1
    sig = _sig(seed=24, amp=0.6)
    ref = rm.render_monitor(sig, SR, out)
    assert np.all(np.isfinite(ref))
    assert ClipAction(stem_id="master", oversample=8).oversample == 8
    with pytest.raises(ValueError):
        ClipAction(stem_id="master", oversample=3)


def test_clip_guardrailed_for_agent_but_user_addable_on_stem(demo_project):
    """The agent can't clip deeper than the -6 dB floor (guardrails bind the agent, not the
    user); the USER can hand-place clip on a stem too (Phase 43 device freedom)."""
    from mix_engine.guardrails import validate_proposal
    from mix_engine.models import ClipAction, ProposalRequest
    from mix_engine.session import Session
    s = Session(demo_project)
    stem = next(iter(s._state["stems"]))
    mk = lambda thr: ProposalRequest(
        rationale="x", predicted_impact=[],
        actions=[ClipAction(stem_id="master", threshold_db=thr, hardness=0.5)])
    assert validate_proposal(mk(-3.0), {}) == []                 # tasteful master clip
    deep = validate_proposal(mk(-9.0), {})                       # too aggressive
    assert any(v.rule == "clip_threshold_min_db" for v in deep)
    # device freedom: the user CAN hand-add clip on a stem (it renders as a peak clipper)
    s.add_stem_device(stem, "clip")
    assert any(e["type"] == "clip" for e in s._state["stems"][stem]["chain"])


def test_saturation_output_charges_gain_budget(demo_project):
    """A positive saturation output trim draws from the same per-stem gain wallet as
    gain moves, so the agent can't use it to dodge the ±6 dB/pass cap; a negative
    trim is a reduction and is free."""
    from mix_engine.guardrails import validate_proposal
    from mix_engine.models import ProposalRequest, SaturationAction
    from mix_engine.session import Session
    s = Session(demo_project)
    stem = next(iter(s._state["stems"]))
    mk = lambda out: ProposalRequest(
        rationale="x", predicted_impact=[],
        actions=[SaturationAction(stem_id=stem, drive_db=3.0, output_db=out)])
    # 4 dB already spent this pass + a 4 dB output trim = 8 dB > the 6 dB/pass cap.
    deep = validate_proposal(mk(4.0), {stem: 4.0})
    assert any(v.rule == "gain_per_stem_per_pass_db" for v in deep)
    assert validate_proposal(mk(-6.0), {stem: 4.0}) == []        # a cut never charges


# ---------- M/S mode for eq + comp (Phase 41) -------------------------------

@pytest.mark.parametrize("channel", ["mid", "side"])
def test_ms_eq_render_matches_monitor(channel):
    """An M/S eq band nulls between the offline render and the monitor — the M/S
    encode/decode is linear+exact and the shelf/peak biquad is the same JUCE maker."""
    sig = _sig(seed=50, amp=0.3)
    chain = [{"type": "eq", "channel": channel,
              "band": {"shape": "peak", "freq_hz": 3000.0, "gain_db": 5.0, "q": 1.0}}]
    oracle = dsp._apply_master_chain(sig.copy(), SR, chain, report=None)
    ref = rm.render_monitor(sig, SR, monitor_chain_params(chain))
    assert _err_db(oracle, ref) < -50.0


def test_ms_eq_targets_the_right_channel():
    """On a mono-in-stereo signal (pure mid, zero side): a SIDE eq is a no-op; a MID
    eq changes it. Proves the band hits only the chosen M/S component."""
    n = SR
    mono = np.random.default_rng(9).standard_normal((n, 1)) * 0.3
    stereo = np.repeat(mono, 2, axis=1)
    eq = lambda ch: rm.render_monitor(stereo, SR, monitor_chain_params(
        [{"type": "eq", "channel": ch,
          "band": {"shape": "peak", "freq_hz": 3000.0, "gain_db": 6.0, "q": 1.0}}]))
    assert _err_db(stereo, eq("side")) < -60.0     # zero side -> no change
    assert _err_db(stereo, eq("mid")) > -30.0      # mid eq does change it


def test_ms_comp_side_noop_on_mono():
    """A side comp on a zero-side (mono) signal does nothing — the M/S path is wired."""
    n = SR
    mono = np.random.default_rng(10).standard_normal((n, 1)) * 0.5
    stereo = np.repeat(mono, 2, axis=1)
    out = rm.render_monitor(stereo, SR, monitor_chain_params(
        [{"type": "comp", "channel": "side", "threshold_db": -30.0, "ratio": 4.0,
          "attack_ms": 5.0, "release_ms": 100.0}]))
    assert _err_db(stereo, out) < -60.0


def test_set_device_channel_roundtrips(demo_project):
    """Phase 41 native co-edit: the M/S selector sets an eq/comp device's channel on a
    stem; bad channel or a non-eq/comp device is rejected."""
    from mix_engine.session import Session
    s = Session(demo_project)
    stem = next(iter(s._state["stems"]))
    s._state["stems"][stem]["chain"].append(
        {"type": "eq", "stem_id": stem,
         "band": {"shape": "high_shelf", "freq_hz": 8000.0, "gain_db": 3.0, "q": 0.7}})
    assert s.set_device_channel("eq", 0, "side", target=stem)["channel"] == "side"
    eq = next(e for e in s._state["stems"][stem]["chain"] if e["type"] == "eq")
    assert eq["channel"] == "side"
    with pytest.raises(ValueError):
        s.set_device_channel("eq", 0, "nope", target=stem)
    with pytest.raises(ValueError):
        s.set_device_channel("saturation", 0, "mid", target=stem)


def test_set_device_band_roundtrips(demo_project):
    """Band-width band selector (2026-07-05): the user picks the device's band
    LABEL on a stem; the contract re-resolves it to the new bandpass center; a
    bogus label or missing device is rejected."""
    from mix_engine.session import Session
    s = Session(demo_project)
    stem = next(iter(s._state["stems"]))
    s._state["stems"][stem]["chain"].append(
        {"type": "band_width", "stem_id": stem, "band": "presence", "width": 1.4})
    assert s.set_device_band(0, "bass", target=stem)["band"] == "bass"
    bw = next(e for e in s._state["stems"][stem]["chain"] if e["type"] == "band_width")
    assert bw["band"] == "bass"
    resolved = monitor_chain_params(s._state["stems"][stem]["chain"])
    rbw = next(e for e in resolved["entries"] if e["type"] == "band_width")
    # bass = 60..150 Hz -> fc = round(sqrt(60*150), 1) = 94.9, q = round(fc/90, 4)
    assert rbw["freq_hz"] == pytest.approx(94.9, abs=0.05)
    assert rbw["band"] == "bass"       # the label rides along for the UI selector
    with pytest.raises(ValueError):
        s.set_device_band(0, "bogus", target=stem)
    with pytest.raises(ValueError):
        s.set_device_band(3, "bass", target=stem)   # no such occurrence


def test_set_device_band_custom_edges(demo_project):
    """Free band scrub (2026-07-19): explicit lo/hi place the band anywhere —
    stored on the entry, resolved to fc = sqrt(lo*hi) with the edges riding
    along; the offline render honours them; a preset pick clears them."""
    import numpy as np

    from mix_engine import dsp
    from mix_engine.session import Session
    s = Session(demo_project)
    stem = next(iter(s._state["stems"]))
    s._state["stems"][stem]["chain"].append(
        {"type": "band_width", "stem_id": stem, "band": "presence", "width": 1.6})
    out = s.set_device_band(0, lo_hz=300.0, hi_hz=1200.0, target=stem)
    assert out["lo_hz"] == 300.0 and out["hi_hz"] == 1200.0
    resolved = monitor_chain_params(s._state["stems"][stem]["chain"])
    rbw = next(e for e in resolved["entries"] if e["type"] == "band_width")
    assert rbw["freq_hz"] == pytest.approx((300.0 * 1200.0) ** 0.5, abs=0.1)
    assert rbw["band"] == "custom"
    assert rbw["lo_hz"] == 300.0 and rbw["hi_hz"] == 1200.0
    # offline render: the custom band moves, the presence label band does NOT
    sr = 44100
    t = np.arange(sr) / sr
    sine = np.stack([np.sin(2 * np.pi * 600.0 * t),
                     -np.sin(2 * np.pi * 600.0 * t)], axis=1)  # pure side @600 Hz
    entry = {"type": "band_width", "band": "presence", "width": 0.0,
             "lo_hz": 300.0, "hi_hz": 1200.0}
    narrowed = dsp._band_width_scale(sine, sr, entry)
    # 600 Hz side content at the centre of a two-octave band: -42 dB once the split has settled
    # (a 48 dB/oct split since 2026-10-21, the band width audit; the FFT mask erased it to 1e-6)
    assert np.abs(narrowed[sr // 2:]).max() < 10 ** (-36 / 20)
    entry_label_only = {"type": "band_width", "band": "presence", "width": 0.0}
    untouched = dsp._band_width_scale(sine, sr, entry_label_only)
    assert np.abs(untouched).max() > 0.9          # presence band (2-4k) misses 600 Hz
    # bad edges rejected; a preset pick clears the custom edges
    with pytest.raises(ValueError):
        s.set_device_band(0, lo_hz=1200.0, hi_hz=300.0, target=stem)
    s.set_device_band(0, "bass", target=stem)
    bw = next(e for e in s._state["stems"][stem]["chain"] if e["type"] == "band_width")
    assert "lo_hz" not in bw and "hi_hz" not in bw and bw["band"] == "bass"


def test_ms_contract_carries_channel():
    band = {"shape": "peak", "freq_hz": 1000.0, "gain_db": 3.0, "q": 1.0}
    e = monitor_chain_params([{"type": "eq", "stem_id": "x", "channel": "side",
                               "band": band}])["entries"][0]
    assert e["channel"] == "side"
    e2 = monitor_chain_params([{"type": "eq", "stem_id": "x", "band": band}])["entries"][0]
    assert e2["channel"] == "stereo"   # default


# ---------- tilt EQ (Phase 41; two shelves, bit-exact tier) -----------------

def test_master_tilt_render_matches_monitor():
    """The tilt nulls between the offline render (dsp._tilt = two shelves) and the
    live-monitor reference — the native builds the same two shelf biquads (bit-exact
    like the eq bands)."""
    sig = _sig(seed=40, amp=0.3)
    chain = [{"type": "tilt", "tilt_db": 4.0, "pivot_hz": 1000.0}]
    oracle = dsp._apply_master_chain(sig.copy(), SR, chain, report=None)
    ref = rm.render_monitor(sig, SR, monitor_chain_params(chain))
    assert _err_db(oracle, ref) < -80.0


def test_tilt_brightens_high_over_low():
    """A positive tilt lifts highs and drops lows (vs a negative tilt)."""
    sig = _sig(seed=41, amp=0.3)
    p = lambda t: rm.render_monitor(sig, SR, monitor_chain_params(
        [{"type": "tilt", "tilt_db": t, "pivot_hz": 1000.0}]))
    freqs = np.fft.rfftfreq(sig.shape[0], 1.0 / SR)
    def hi_lo(a):
        spec = np.abs(np.fft.rfft(a[:, 0]))
        hi = spec[freqs > 4000.0].sum()
        lo = spec[(freqs > 20.0) & (freqs < 250.0)].sum()
        return hi / (lo + 1e-12)
    assert hi_lo(p(6.0)) > hi_lo(p(-6.0))


def test_tilt_contract_resolves():
    out = monitor_chain_params([{"type": "tilt", "stem_id": "master",
                                 "tilt_db": 3.0, "pivot_hz": 800.0}])
    assert out["print_only"] == []
    t = next(e for e in out["entries"] if e["type"] == "tilt")
    assert t["tilt_db"] == 3.0 and t["pivot_hz"] == 800.0 and t["q"] > 0.0


# ---------- output trims (2026-07-12 gain staging: eq/tilt output_db) --------

def test_tilt_output_trim_renders_and_nulls():
    """A tilt carrying output_db lands ~output_db quieter AND still nulls between
    the offline render and the live-monitor reference (the parity triangle holds)."""
    sig = _sig(seed=42, amp=0.3)
    chain = [{"type": "tilt", "tilt_db": 4.0, "pivot_hz": 1000.0, "output_db": -6.0}]
    oracle = dsp._apply_master_chain(sig.copy(), SR, chain, report=None)
    ref = rm.render_monitor(sig, SR, monitor_chain_params(chain))
    assert _err_db(oracle, ref) < -80.0
    flat = dsp._apply_master_chain(
        sig.copy(), SR,
        [{"type": "tilt", "tilt_db": 4.0, "pivot_hz": 1000.0}], report=None)
    rms = lambda a: float(np.sqrt((a ** 2).mean()) + 1e-12)
    drop_db = 20.0 * np.log10(rms(oracle) / rms(flat))
    assert abs(drop_db + 6.0) < 0.5


def test_eq_output_trim_sums_across_entries_and_nulls():
    """Two eq bands each carrying an output_db: the DEVICE's trim is their sum
    (the bank is linear), exact between the offline render and the reference."""
    sig = _sig(seed=43, amp=0.3)
    band = lambda f, g: {"shape": "peak", "freq_hz": f, "gain_db": g, "q": 1.0}
    chain = [{"type": "eq", "band": band(300.0, 3.0), "output_db": -2.0},
             {"type": "eq", "band": band(3000.0, -2.0), "output_db": -1.0}]
    oracle = dsp._apply_master_chain(sig.copy(), SR, chain, report=None)
    ref = rm.render_monitor(sig, SR, monitor_chain_params(chain))
    assert _err_db(oracle, ref) < -80.0
    flat = dsp._apply_master_chain(
        sig.copy(), SR,
        [{"type": "eq", "band": band(300.0, 3.0)},
         {"type": "eq", "band": band(3000.0, -2.0)}], report=None)
    rms = lambda a: float(np.sqrt((a ** 2).mean()) + 1e-12)
    drop_db = 20.0 * np.log10(rms(oracle) / rms(flat))
    assert abs(drop_db + 3.0) < 0.5


def test_tilt_guardrail_and_registration():
    from mix_engine.guardrails import validate_proposal
    from mix_engine.models import ProposalRequest, TiltAction
    from mix_engine.session import (BUS_ACTION_TYPES, MASTER_ACTION_TYPES,
                                    SUPPORTED_ACTION_TYPES)
    assert {"tilt"} <= MASTER_ACTION_TYPES & BUS_ACTION_TYPES & SUPPORTED_ACTION_TYPES
    mk = lambda t: ProposalRequest(rationale="x", predicted_impact=[],
                                   actions=[TiltAction(stem_id="master", tilt_db=t)])
    assert validate_proposal(mk(4.0), {}) == []
    assert any(v.rule == "tilt_max_db" for v in validate_proposal(mk(10.0), {}))


# ---------- gate / downward expander (Phase 41; perceptual tier) ------------

def test_gate_attenuates_below_threshold():
    """The gate passes signal above threshold and attenuates below it (downward
    expander) — per-sample approximation of the offline dsp._gate block law."""
    sig = _sig(seed=30, amp=0.5)
    n = sig.shape[0]
    sig[n // 2:] *= 0.02                       # second half ~ -34 dB, below -20
    out = rm.render_monitor(sig, SR, monitor_chain_params(
        [{"type": "gate", "stem_id": "x", "threshold_db": -20.0, "ratio": 8.0,
          "attack_ms": 1.0, "release_ms": 50.0}]))
    rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
    half = n // 2
    assert rms(out[:half]) == pytest.approx(rms(sig[:half]), rel=0.1)   # loud passes
    tail = slice(half + SR // 4, n)                                     # past the release
    assert rms(out[tail]) < 0.5 * rms(sig[tail])                        # quiet attenuated
    assert np.all(np.isfinite(out))


def test_gate_renders_on_a_stem(demo_project):
    """Offline gate (dsp._gate, 5 ms block-envelope) renders in a stem chain and
    never adds energy (it only attenuates below threshold)."""
    from mix_engine import dsp
    from mix_engine.session import Session
    s = Session(demo_project)
    stem = next(iter(s._state["stems"]))
    base, _ = dsp.render_stem(s, stem)
    s._state["stems"][stem]["chain"].append(
        {"type": "gate", "stem_id": stem, "threshold_db": -18.0, "ratio": 6.0,
         "attack_ms": 1.0, "release_ms": 100.0})
    gated, _ = dsp.render_stem(s, stem)
    rms = lambda a: float(np.sqrt(np.mean(np.asarray(a) ** 2)))
    assert np.all(np.isfinite(gated))
    assert rms(gated) <= rms(base) + 1e-6


def test_gate_contract_resolves():
    out = monitor_chain_params([{"type": "gate", "stem_id": "d", "threshold_db": -30.0,
                                 "ratio": 6.0, "attack_ms": 2.0, "release_ms": 80.0}])
    assert out["print_only"] == []
    g = next(e for e in out["entries"] if e["type"] == "gate")
    assert g["threshold_db"] == -30.0 and g["ratio"] == 6.0 and g["release_ms"] == 80.0


def test_gate_available_on_stem_bus_and_master(demo_project):
    """Phase 43 device freedom: gate is first-class everywhere — the AGENT proposes it on
    stems/busses AND the master (action-type sets), and the USER hand-adds it on any target."""
    from mix_engine.session import (BUS_ACTION_TYPES, MASTER_ACTION_TYPES,
                                    SUPPORTED_ACTION_TYPES, Session)
    assert "gate" in SUPPORTED_ACTION_TYPES
    assert "gate" in BUS_ACTION_TYPES
    assert "gate" in MASTER_ACTION_TYPES   # device freedom: the agent can now propose master gate
    s = Session(demo_project)
    stem = next(iter(s._state["stems"]))
    s.add_stem_device(stem, "gate")
    assert any(e["type"] == "gate" for e in s._state["stems"][stem]["chain"])
    s.add_master_device("gate")            # the user hand-adds it on the master too
    assert any(e["type"] == "gate" for e in s._state["master_chain"])


def _gate_burst_sig():
    """0.5 s loud 220 Hz tone then 0.5 s of the same tone ~40 dB down, stereo
    (channels, frames) float32 — the gate v2 test fixture."""
    t = np.arange(SR) / SR
    sig = (np.where(t < 0.5, 0.5, 0.005) * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    return np.stack([sig, sig])


def _rms_db(a, s=slice(None)):
    x = np.asarray(a, dtype=np.float64)[:, s]
    return 20.0 * np.log10(float(np.sqrt(np.mean(x ** 2))) + 1e-12)


def test_gate_v2_range_floor_and_hold():
    """Gate v2: range_db floors the closed-gate attenuation (duck, don't mute);
    hold_ms keeps the gate open past the hit; the render is deterministic."""
    from mix_engine import dsp
    audio = _gate_burst_sig()
    base = {"type": "gate", "threshold_db": -30.0, "ratio": 8.0,
            "attack_ms": 1.0, "release_ms": 50.0}
    quiet = slice(int(0.6 * SR), SR)
    full = dsp._gate(audio, SR, base)
    assert _rms_db(full, quiet) - _rms_db(audio, quiet) < -40.0      # default ~ full mute
    ranged = dsp._gate(audio, SR, {**base, "range_db": -12.0})
    assert (_rms_db(ranged, quiet) - _rms_db(audio, quiet)
            == pytest.approx(-12.0, abs=1.5))                        # floored at the range
    held = dsp._gate(audio, SR, {**base, "hold_ms": 200.0})
    early = slice(int(0.52 * SR), int(0.65 * SR))
    assert _rms_db(held, early) - _rms_db(audio, early) > -1.0       # hold keeps it open
    assert _rms_db(full, early) - _rms_db(audio, early) < -6.0       # no hold: closing
    assert np.array_equal(full, dsp._gate(audio, SR, base))          # deterministic


def test_gate_v2_sc_filter_keys_detector_only():
    """A detector HPF well above the source's band keeps the gate CLOSED on a loud
    low tone (the detector no longer sees it), while without the SC filter the same
    tone opens the gate — and the audio path itself is never filtered."""
    from mix_engine import dsp
    t = np.arange(SR) / SR
    low = (0.5 * np.sin(2 * np.pi * 80.0 * t)).astype(np.float32)
    audio = np.stack([low, low])
    base = {"type": "gate", "threshold_db": -30.0, "ratio": 8.0,
            "attack_ms": 1.0, "release_ms": 50.0}
    opened = dsp._gate(audio, SR, base)                 # -6 dB tone: open, passthrough
    assert _rms_db(opened) == pytest.approx(_rms_db(audio), abs=0.5)
    keyed = dsp._gate(audio, SR, {**base, "sc_hpf_hz": 1000.0})
    assert _rms_db(keyed) < _rms_db(audio) - 20.0       # detector blind -> gate closed


def test_gate_v2_contract_carries_new_fields_and_reference_renders():
    """monitor_chain_params passes hold/range/hysteresis/SC through to the native
    contract (with back-compat defaults), and reference_monitor renders them with
    the range floor binding."""
    out = monitor_chain_params(
        [{"type": "gate", "stem_id": "d", "threshold_db": -30.0, "ratio": 6.0,
          "attack_ms": 2.0, "release_ms": 80.0, "hold_ms": 120.0,
          "range_db": -15.0, "hysteresis_db": 4.0, "sc_hpf_hz": 200.0}])
    g = next(e for e in out["entries"] if e["type"] == "gate")
    assert g["hold_ms"] == 120.0 and g["range_db"] == -15.0
    assert g["hysteresis_db"] == 4.0 and g["sc_hpf_hz"] == 200.0
    assert g["sc_lpf_hz"] == 20000.0                     # omitted -> the off sentinel
    legacy = monitor_chain_params(
        [{"type": "gate", "stem_id": "d", "threshold_db": -30.0, "ratio": 6.0,
          "attack_ms": 2.0, "release_ms": 80.0}])        # a pre-v2 session entry
    lg = next(e for e in legacy["entries"] if e["type"] == "gate")
    assert lg["hold_ms"] == 0.0 and lg["range_db"] == -80.0

    sig = _sig(seed=31, amp=0.5)
    n = sig.shape[0]
    sig[n // 2:] *= 0.02
    ref = rm.render_monitor(sig, SR, out)
    assert np.all(np.isfinite(ref))
    rmsf = lambda a: float(np.sqrt(np.mean(a ** 2)))
    tail = slice(n // 2 + SR // 4, n)
    # the -15 dB range floor binds: the tail never falls much deeper than 15 dB
    assert rmsf(ref[tail]) > rmsf(sig[tail]) * 10 ** (-16.5 / 20.0)


def test_gate_threshold_guardrailed_for_agent():
    """Gate v2: the agent can't set a gate threshold above -20 dBFS (that eats
    program material); a sensible drum-gate threshold passes."""
    from mix_engine.guardrails import validate_proposal
    from mix_engine.models import GateAction, ProposalRequest
    mk = lambda thr: ProposalRequest(
        rationale="x", predicted_impact=[],
        actions=[GateAction(stem_id="drums", threshold_db=thr, ratio=8.0)])
    assert validate_proposal(mk(-45.0), {}) == []
    hot = validate_proposal(mk(-10.0), {})
    assert any(v.rule == "gate_threshold_max_db" for v in hot)


def test_user_can_edit_a_stem_duck_param(demo_project):
    """Phase 41: the user dials the agent's duck directly (native VST co-edit) — on a
    per-track insert the edit targets THAT stem's duck, clamped to the model range."""
    from mix_engine.session import Session
    s = Session(demo_project)
    stem = next(iter(s._state["stems"]))
    s._state["stems"][stem]["chain"].append(
        {"type": "duck", "stem_id": stem, "key_stem_id": "kick",
         "threshold_db": -40.0, "depth_db": 5.0, "attack_ms": 5.0, "release_ms": 120.0})
    out = s.set_master_device_param("duck", 0, "depth_db", 8.0, target=stem)
    assert out["value"] == 8.0
    duck = next(e for e in s._state["stems"][stem]["chain"] if e["type"] == "duck")
    assert duck["depth_db"] == 8.0
    s.set_master_device_param("duck", 0, "depth_db", 99.0, target=stem)   # clamps to 12
    assert duck["depth_db"] == 12.0


# ---------- sidechain duck (Phase 41; perceptual tier, host sidechain key) --

def _duck_entry(**kw):
    base = {"type": "duck", "stem_id": "bass", "key_stem_id": "kick",
            "threshold_db": -40.0, "depth_db": 8.0, "attack_ms": 5.0,
            "release_ms": 150.0}
    base.update(kw)
    return base


def test_duck_reduces_under_a_loud_key_and_releases():
    """The native duck reads the key off the host sidechain (Phase 41 Option A):
    it ducks the main where the key is loud and releases where the key is silent,
    bounded by depth_db. Per-sample causal approximation of dsp._duck (perceptual,
    not a tight null) — the export keeps the offline duck-free key render."""
    sig = _sig(seconds=2.0, seed=11, amp=0.3)
    n = sig.shape[0]
    key = np.zeros_like(sig)
    key[: n // 2] = _sig(seconds=2.0, seed=12, amp=0.5)[: n // 2]   # key loud in the first half only
    params = monitor_chain_params([_duck_entry()])
    out = rm.render_monitor(sig, SR, params, sidechain=key)
    assert np.all(np.isfinite(out))

    rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
    half = n // 2
    assert rms(out[:half]) < 0.85 * rms(sig[:half])           # ducked under the key
    tail = slice(half + int(0.35 * SR), n)                    # well past the release
    assert rms(out[tail]) == pytest.approx(rms(sig[tail]), rel=0.02)
    assert rms(out[:half]) >= rms(sig[:half]) * 10.0 ** (-8.0 / 20.0) * 0.9   # <= depth


def test_duck_no_sidechain_is_noop():
    """No key connected (sidechain=None) -> the duck passes through untouched, so
    inserting the plugin without routing a sidechain never gates the audio."""
    sig = _sig(seed=13, amp=0.3)
    params = monitor_chain_params([_duck_entry()])
    out = rm.render_monitor(sig, SR, params, sidechain=None)
    assert _err_db(sig, out) < -200.0


def test_duck_contract_resolves_amount_and_timing():
    """The contract carries only amount/timing (the native engine reads the key
    live); key_stem_id rides along as advisory intent. It is heard live, never
    print_only."""
    out = monitor_chain_params([_duck_entry(threshold_db=-38.0, depth_db=6.0,
                                            attack_ms=4.0, release_ms=120.0)])
    assert out["print_only"] == []
    d = next(e for e in out["entries"] if e["type"] == "duck")
    assert d["depth_db"] == 6.0 and d["threshold_db"] == -38.0
    assert d["attack_ms"] == 4.0 and d["key_stem_id"] == "kick"


def test_tune_and_transpose_carry_out_of_band_pitch_fields():
    """tune/transpose stay print_only (no native chain node), but carry summed
    out-of-band fields the DAW realizes as reconciled clip pitch (2026-08-09 —
    before this they were inaudible in the DAW's monitor AND its bounce). A
    bypassed entry must NOT count: its zero is what un-tunes the clips."""
    out = monitor_chain_params([
        {"type": "tune", "cents": -15.0},
        {"type": "tune", "cents": -5.0, "enabled": False},
        {"type": "transpose", "semitones": 2},
    ])
    assert out["entries"] == []
    assert out["print_only"] == ["tune", "tune", "transpose"]
    assert out["tune_cents"] == -15.0
    assert out["transpose_semitones"] == 2.0


def test_pitch_fields_default_to_zero():
    """Always present (the C++ side reconciles against the value, so an absent
    field would leave stale clip pitch behind when the action is removed)."""
    out = monitor_chain_params([{"type": "gain", "gain_db": -3.0}])
    assert out["tune_cents"] == 0.0
    assert out["transpose_semitones"] == 0.0
