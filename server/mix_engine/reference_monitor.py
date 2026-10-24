"""Phase 40 (native VST) — numpy reference of the native real-time MonitorChain.

This is the executable spec the C++ ``MonitorChain`` transcribes. Its only job is
to be *provably equal to the offline engine* (``dsp.py`` / Pedalboard) so the
native monitor and the deliverable render agree — that equality is asserted in
``tests/test_phase40_monitor_fidelity.py`` (the oracle harness).

Fidelity notes:
- **EQ biquads** are transcribed verbatim from JUCE ``dsp::IIR`` coefficient
  makers (``juce_IIRFilter.cpp``: array layout ``{b0,b1,b2,a0,a1,a2}``, applied
  normalised by ``a0``). Pedalboard wraps those same JUCE makers, so the null vs
  Pedalboard is float32 noise floor, and the C++ — calling the makers directly —
  matches by construction. Pedalboard's HPF/LPF are first-order (one-pole,
  6 dB/oct -> JUCE makeFirstOrder*), not biquads; the test confirms.
- **Brickwall limiter** mirrors ``dsp._brickwall`` (default 3 ms lookahead max-filter
  + 50 ms moving-average smoothing, matching the live native limiter; release_ms /
  lookahead_ms / style / channel_link override). That is the *offline* (non-causal)
  form used by the export. The real-time monitor's limiter is a causal approximation
  with a reported lookahead latency — validated perceptually in Step 1, not here.
- **Compressor** is a feed-forward peak compressor, linked across the channels,
  with one-pole ballistics on the gain reduction in dB and literal time
  constants. It is the law ``MonitorChain::processOneComp`` runs, and - since
  2026-10-03 (reader E4.R1) - the law the OFFLINE engine runs too
  (``dsp._native_comp``, pinned to ``_compress`` by test_comp_native_law.py). It
  was written as "an approximation of Pedalboard's ... until the C++ comp is
  characterised", and never was: Pedalboard's attack is a time constant of
  ms/2pi on the level, per channel, so the two engines disagreed by up to
  4.7 dB on drums.

Audio convention: ``(frames, channels)`` float64, the engine's internal layout.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import lfilter

from .tp_detector import intersample_peak   # the limiter's detector: the native's table (reader E4·R7)

# Mirror dsp.py's limiter windows (audio-dsp skill: 5 ms / 5 ms).
LIMITER_LOOKAHEAD_S = 0.005
LIMITER_SMOOTH_S = 0.005
MB_CROSSOVERS_HZ = (150.0, 2000.0)   # multiband split (mirrors dsp.MB_CROSSOVERS_HZ)

# OTT internal character (mirrors dsp.OTT_* — the fixed target/slopes/clamps/ballistics).
OTT_TARGET_DB = -30.0
OTT_UP_SLOPE = 0.6
OTT_DOWN_SLOPE = 0.85
OTT_MAX_BOOST_DB = 24.0
OTT_MAX_CUT_DB = 36.0
OTT_BASE_ATTACK_MS = 12.0
OTT_BASE_RELEASE_MS = 120.0

OD_TONE_HZ = 3000.0   # overdrive tone-tilt pivot (mirrors dsp.OD_TONE_HZ)
# De-ess + transient constants — mirror dsp.py so the reference tracks the engine.
DEESS_RATIO = 4.0
DEESS_ATTACK_MS = 1.0
DEESS_RELEASE_MS = 50.0
TRANSIENT_FAST_ATTACK_MS = 1.0
TRANSIENT_FAST_RELEASE_MS = 20.0
TRANSIENT_SLOW_ATTACK_MS = 30.0
TRANSIENT_SLOW_RELEASE_MS = 300.0
TRANSIENT_FULL_DB = 6.0   # fast env this far above slow = fully transient


def biquad_coeffs(shape: str, sample_rate: float, freq_hz: float,
                  q: float, gain_db: float) -> tuple[np.ndarray, np.ndarray]:
    """Return normalised ``(b, a)`` (length-3 each, ``a[0] == 1``) for one band,
    transcribed from JUCE ``ArrayCoefficients`` makers. ``b/a`` feed
    ``scipy.signal.lfilter`` — the direct-form difference equation JUCE applies
    after dividing by ``a0``.

    ``freq_hz``/``q`` are clamped to the SAME range the native monitor uses
    (``MonitorChain.cpp`` ``makeBiquad``: ``jlimit (2.0, sr * 0.49, f)`` and
    ``jmax (0.1, q)``). Without the high clamp a legal band (``EqBand.freq_hz``
    allows 20 kHz) at or above Nyquist — reachable whenever the project's stems
    are below ~40 kHz, since ``Session.project_sample_rate`` is the max stem
    rate — produced a divergent filter that rendered all-NaN offline while the
    native monitor, which already clamped, sounded fine."""
    sr = float(sample_rate)
    f = min(max(float(freq_hz), 2.0), sr * 0.49)
    # q == 0 is RESERVED: the lpf2/hpf2 crossover call uses it to mean "the fixed
    # Linkwitz-Riley Q" (see below), so it must survive the clamp untouched.
    # ... for those two shapes ONLY: `makeBiquad` captures q before its clamp for lpf2/hpf2 alone and
    # clamps it for every other shape, so a peak at q 0 is a q 0.1 peak there and was a division by
    # zero here (2026-10-17, the EQ audit: the band path runs on this now).
    q = float(q) if (float(q) == 0.0 and shape in ("lpf2", "hpf2")) else max(0.1, float(q))
    pi = np.pi

    if shape in ("lpf", "hpf"):
        # Pedalboard's HPF/LPF are first-order (one-pole, 6 dB/oct), not biquads.
        # JUCE makeFirstOrder{Low,High}Pass: array {b0, b1, a0, a1}, n = tan(pi f/sr).
        # `q` is unused here (a one-pole has no resonance); kept for a uniform API.
        n = np.tan(pi * f / sr)
        if shape == "lpf":
            b = np.array([n, n])
        else:                            # hpf
            b = np.array([1.0, -1.0])
        a = np.array([n + 1.0, n - 1.0])
        return b / a[0], a / a[0]

    if shape in ("lpf2", "hpf2"):
        # 2nd-order Butterworth-Q, JUCE makeLowPass/makeHighPass. `q` is the pole Q;
        # 0 (the multiband crossover's call) keeps the fixed Q=1/sqrt2 (Linkwitz-Riley
        # 4th-order crossover: cascade two of these) — Phase 46's variable-slope
        # cascade (`slope_cascade_coeffs`) passes an explicit per-section Q instead.
        inv_q = np.sqrt(2.0) if not q else 1.0 / q
        if shape == "lpf2":
            n = 1.0 / np.tan(pi * f / sr)
            n2 = n * n
            c1 = 1.0 / (1.0 + inv_q * n + n2)
            b = np.array([c1, c1 * 2.0, c1])
            a = np.array([1.0, c1 * 2.0 * (1.0 - n2), c1 * (1.0 - inv_q * n + n2)])
        else:                            # hpf2
            n = np.tan(pi * f / sr)
            n2 = n * n
            c1 = 1.0 / (1.0 + inv_q * n + n2)
            b = np.array([c1, c1 * -2.0, c1])
            a = np.array([1.0, c1 * 2.0 * (n2 - 1.0), c1 * (1.0 - inv_q * n + n2)])
        return b / a[0], a / a[0]

    if shape == "bandpass":
        # JUCE makeBandPass — isolates a band for the dynamic-EQ detector +
        # delta recombine. `q` is the bandpass Q (no gain term). a0 == 1.
        n = 1.0 / np.tan(pi * f / sr)
        n2 = n * n
        inv_q = 1.0 / q
        c1 = 1.0 / (1.0 + inv_q * n + n2)
        b = np.array([c1 * n * inv_q, 0.0, -c1 * n * inv_q])
        a = np.array([1.0, c1 * 2.0 * (1.0 - n2), c1 * (1.0 - inv_q * n + n2)])
        return b / a[0], a / a[0]

    # peak / shelves: A = sqrt(linear gain) = 10^(dB/40); omega floored at 2 Hz.
    A = 10.0 ** (gain_db / 40.0)
    omega = 2.0 * pi * max(f, 2.0) / sr
    coso = np.cos(omega)
    sino = np.sin(omega)

    if shape == "peak":
        alpha = sino / (2.0 * q)
        c2 = -2.0 * coso
        ata = alpha * A
        aoa = alpha / A
        b = np.array([1.0 + ata, c2, 1.0 - ata])
        a = np.array([1.0 + aoa, c2, 1.0 - aoa])
    elif shape == "notch":               # RBJ notch — full cut at f0, width set by q
        alpha = sino / (2.0 * q)
        c2 = -2.0 * coso
        b = np.array([1.0, c2, 1.0])
        a = np.array([1.0 + alpha, c2, 1.0 - alpha])
    elif shape == "low_shelf":
        am1, ap1 = A - 1.0, A + 1.0
        beta = sino * np.sqrt(A) / q
        am1c = am1 * coso
        b = np.array([A * (ap1 - am1c + beta),
                      A * 2.0 * (am1 - ap1 * coso),
                      A * (ap1 - am1c - beta)])
        a = np.array([ap1 + am1c + beta,
                      -2.0 * (am1 + ap1 * coso),
                      ap1 + am1c - beta])
    elif shape == "high_shelf":
        am1, ap1 = A - 1.0, A + 1.0
        beta = sino * np.sqrt(A) / q
        am1c = am1 * coso
        b = np.array([A * (ap1 + am1c + beta),
                      A * -2.0 * (am1 + ap1 * coso),
                      A * (ap1 + am1c - beta)])
        a = np.array([ap1 - am1c + beta,
                      2.0 * (am1 - ap1 * coso),
                      ap1 - am1c - beta])
    else:
        raise ValueError(f"unknown EQ shape: {shape!r}")
    return b / a[0], a / a[0]


# Phase 46 (Pro-Q-class EQ, Tranche 1): variable-slope hpf/lpf cuts are a cascade
# of Butterworth-Q 2nd-order sections. BUTTERWORTH_Q is the reference Q (a flat,
# maximally-flat cascade); the band's own `q` scales every section relative to it,
# so `q == BUTTERWORTH_Q` is pure Butterworth.
BUTTERWORTH_Q = 1.0 / np.sqrt(2.0)


def butterworth_section_qs(order: int) -> list[float]:
    """Per-section pole Q for an `order`-th order (even) Butterworth lpf/hpf —
    `order // 2` cascaded 2nd-order sections, ``Q_k = 1/(2*cos((2k+1)*pi/(2*order)))``
    for ``k = 0 .. order//2 - 1``. ``order=2`` -> one section at ``BUTTERWORTH_Q``
    (12 dB/oct); ``order=4`` -> two sections (24 dB/oct); etc."""
    return [1.0 / (2.0 * np.cos((2 * k + 1) * np.pi / (2 * order)))
            for k in range(order // 2)]


def slope_cascade_coeffs(shape: str, sample_rate: float, freq_hz: float,
                         q: float, slope_db_oct: int,
                         ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Coefficients for a Phase-46 variable-slope hpf/lpf: a list of ``(b, a)``
    biquad sections to cascade. ``slope_db_oct <= 6`` returns the legacy
    single first-order section (unused by ``apply_eq_band``, which keeps the
    exact 1st-order path for that case, but kept here for callers that want the
    cascade form uniformly). ``slope_db_oct // 12`` sections otherwise, each a
    ``lpf2``/``hpf2`` biquad at a Butterworth-cascade Q, scaled by
    ``q / BUTTERWORTH_Q`` (so ``q == BUTTERWORTH_Q`` is a flat Butterworth
    response), clamped to the biquad Q range ``[0.1, 40]``."""
    if slope_db_oct <= 6:
        return [biquad_coeffs(shape, sample_rate, freq_hz, q, 0.0)]
    order = int(slope_db_oct) // 6
    section_shape = "lpf2" if shape == "lpf" else "hpf2"
    # Above 12 dB/oct the cascade has two or more sections and EVERY one of them is scaled by the
    # same q, so the resonant peak compounds: measured on broadband noise at -20 dBFS peak, a
    # perfectly legal `lpf 1 kHz q=8` (DEVICE_PARAMS allows 0.3..8, EQ_SLOPES allows 96) rendered
    # +12.6 dB at slope 24, +49.4 dB at 48 and **+127.3 dB** at 96 — a two-knob-move detonation,
    # with only the master safety limiter between it and the monitor, and nothing at all in front
    # of a soloed track or a bus tap. `eq_resonance_max_db` catches it inside validate_proposal
    # only, so neither the DAW's `device.set` nor the user's own knob ever met it.
    #
    # So the pair is bound where the COEFFICIENTS are built, which is the one place every write
    # path reaches: q shapes a single section (a resonant 12 dB/oct filter is a musical sound and
    # stays), and a real cascade runs Butterworth — the same thing Pro-Q does by not offering Q
    # above 12 dB/oct (2026-09-08 device audit).
    scale = 1.0 if slope_db_oct > 12 else q / BUTTERWORTH_Q
    return [biquad_coeffs(section_shape, sample_rate, freq_hz,
                          min(40.0, max(0.1, qk * scale)), 0.0)
            for qk in butterworth_section_qs(order)]


def apply_eq_band(audio: np.ndarray, sample_rate: float, shape: str,
                  freq_hz: float, q: float, gain_db: float,
                  slope_db_oct: int = 6) -> np.ndarray:
    """Filter ``(frames, channels)`` audio through one EQ band. Phase 46:
    ``tilt_shelf`` cascades a low-shelf (-gain_db/2) then a high-shelf
    (+gain_db/2), both at ``freq_hz``/``q``; ``hpf``/``lpf`` with
    ``slope_db_oct > 6`` cascade the Butterworth-Q sections from
    ``slope_cascade_coeffs`` instead of the legacy 1st-order filter."""
    if shape == "tilt_shelf":
        out = apply_eq_band(audio, sample_rate, "low_shelf", freq_hz, q, -gain_db / 2.0)
        return apply_eq_band(out, sample_rate, "high_shelf", freq_hz, q, gain_db / 2.0)
    if shape in ("hpf", "lpf") and slope_db_oct > 6:
        out = np.asarray(audio, dtype=np.float64)
        for b, a in slope_cascade_coeffs(shape, sample_rate, freq_hz, q, slope_db_oct):
            out = lfilter(b, a, out, axis=0)
        return out
    b, a = biquad_coeffs(shape, sample_rate, freq_hz, q, gain_db)
    return lfilter(b, a, np.asarray(audio, dtype=np.float64), axis=0)


def _compress(audio: np.ndarray, sample_rate: float, threshold_db: float,
              ratio: float, attack_ms: float, release_ms: float,
              makeup_db: float = 0.0, mix: float = 1.0) -> np.ndarray:
    """Feed-forward peak compressor (approximate; see module docstring). Linked
    across channels (one gain envelope), hard knee, one-pole ballistics on the
    gain-reduction in dB."""
    x = np.asarray(audio, dtype=np.float64)
    detect = np.max(np.abs(x), axis=1)
    level_db = 20.0 * np.log10(np.maximum(detect, 1e-12))
    over = level_db - threshold_db
    target = np.where(over > 0.0, -over * (1.0 - 1.0 / ratio), 0.0)

    a_att = np.exp(-1.0 / (sample_rate * attack_ms / 1000.0))
    a_rel = np.exp(-1.0 / (sample_rate * release_ms / 1000.0))
    g = np.empty_like(target)
    prev = 0.0
    for i in range(target.size):
        t = target[i]
        coef = a_att if t < prev else a_rel   # faster move toward more reduction
        prev = coef * prev + (1.0 - coef) * t
        g[i] = prev
    gain_lin = 10.0 ** ((g + makeup_db) / 20.0)
    wet = x * gain_lin[:, np.newaxis]
    return mix * wet + (1.0 - mix) * x


def _gate(audio: np.ndarray, sample_rate: float, threshold_db: float, ratio: float,
          attack_ms: float, release_ms: float, hold_ms: float = 0.0,
          range_db: float = -80.0, hysteresis_db: float = 0.0,
          sc_hpf_hz: float = 0.0, sc_lpf_hz: float = 20000.0) -> np.ndarray:
    """Real-time downward gate/expander - `MonitorChain::applyGate` (`gate_gain_db`, the one
    law, which `dsp._gate` runs too); the sidechain HP/LP biquads (q=0 Butterworth sentinel)
    filter the DETECTOR only. (frames, channels)."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if ratio <= 1.0:
        return x
    det = x
    if sc_hpf_hz >= 20.0:
        b, a = biquad_coeffs("hpf2", sample_rate,
                             min(sc_hpf_hz, 0.45 * sample_rate), 0.0, 0.0)
        det = lfilter(b, a, det, axis=0)
    if sc_lpf_hz < 20000.0:
        b, a = biquad_coeffs("lpf2", sample_rate,
                             min(sc_lpf_hz, 0.45 * sample_rate), 0.0, 0.0)
        det = lfilter(b, a, det, axis=0)
    from .dsp import gate_gain_db   # the one law (dsp imports this module: a late import)
    gain_db = gate_gain_db(np.max(np.abs(det), axis=1), sample_rate, threshold_db, ratio,
                           attack_ms, release_ms, hold_ms, range_db, hysteresis_db)
    return x * (10.0 ** (gain_db / 20.0))[:, np.newaxis]


def _dynamic_eq(audio: np.ndarray, sample_rate: float, freq_hz: float, q: float,
                threshold_db: float, ratio: float, attack_ms: float,
                release_ms: float, gain_db: float) -> np.ndarray:
    """The dynamic EQ - `dsp._dynamic_eq`, the DAW's law since 2026-10-17 (this was a copy of the
    native's old dB-domain detector). (frames, channels) in and out."""
    from .dsp import _dynamic_eq as one_law   # dsp imports this module: a late import
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    entry = {"freq_hz": freq_hz, "q": q, "threshold_db": threshold_db, "ratio": ratio,
             "attack_ms": attack_ms, "release_ms": release_ms, "gain_db": gain_db}
    return one_law(np.ascontiguousarray(x.T.astype(np.float32)), int(sample_rate), entry).T.astype(np.float64)


def _lr4(audio: np.ndarray, sample_rate: float, shape: str, fc: float) -> np.ndarray:
    """Linkwitz-Riley 4th-order = two cascaded 2nd-order Butterworth sections."""
    b, a = biquad_coeffs(shape, sample_rate, fc, 0.0, 0.0)
    return lfilter(b, a, lfilter(b, a, audio, axis=0), axis=0)


_LR8_QS = (0.54119610014619701, 1.3065629648763766)   # a 4th-order Butterworth's sections


def _lr8_split(t: np.ndarray, sample_rate: float, lo: float, hi: float) -> tuple:
    """The 3-way LR8 split on (channels, frames) - the band width's and the imager's
    (MonitorChain::applyBandWidth / applyImager): each crossover the Butterworth's two sections
    run twice; low = AP@hi(LP@lo x), mid = LP@hi(HP@lo x), high = HP@hi(HP@lo x)."""
    def lr8(v, shape, f):
        for qq in _LR8_QS + _LR8_QS:
            b, a = biquad_coeffs(shape, sample_rate, f, qq, 0.0)
            v = lfilter(b, a, v, axis=-1)
        return v
    low_raw = lr8(t, "lpf2", lo)
    rest = lr8(t, "hpf2", lo)
    low = lr8(low_raw, "lpf2", hi) + lr8(low_raw, "hpf2", hi)
    return low, lr8(rest, "lpf2", hi), lr8(rest, "hpf2", hi)


def _imager(audio: np.ndarray, sample_rate: float, entry: dict) -> np.ndarray:
    """3-band stereo imager - ``MonitorChain::applyImager``'s law (the imager audit, 2026-10-21):
    the LR8 split at the crossovers, each band's side scaled by its width (held to 0..2), summed;
    always split (all widths 1 is the split's allpass). No-op on mono."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if x.shape[1] != 2:
        return x
    lo = min(4000.0, max(30.0, float(entry.get("crossover_lo_hz", 150.0))))
    hi = min(16000.0, max(200.0, float(entry.get("crossover_hi_hz", 2000.0))))
    if hi <= lo:
        hi = lo * 1.5
    widths = [min(2.0, max(0.0, float(entry.get(k, 1.0)))) for k in ("width_low", "width_mid", "width_high")]
    bands = _lr8_split(x.T, sample_rate, lo, hi)
    out_l = np.zeros(x.shape[0])
    out_r = np.zeros(x.shape[0])
    for band, w in zip(bands, widths):
        m = 0.5 * (band[0] + band[1])
        sd = 0.5 * (band[0] - band[1]) * w
        out_l = out_l + (m + sd)
        out_r = out_r + (m - sd)
    return np.stack([out_l, out_r], axis=1)

MB_MAX_BOOST_DB = 12.0   # mirrors dsp.MB_MAX_BOOST_DB (upward-mode cap)


def _one_pole_lp(audio: np.ndarray, sample_rate: float, freq_hz: float) -> np.ndarray:
    """First-order LP on (frames, channels) — the same JUCE one-pole coefficients
    `MonitorChain::applyUtility` builds via makeBiquad("lpf", ...)."""
    b, a = biquad_coeffs("lpf", sample_rate, freq_hz, 0.707, 0.0)
    return lfilter(b, a, np.asarray(audio, dtype=np.float64), axis=0)


UTILITY_TOGGLES = ("mono", "invert", "invert_left", "invert_right", "bass_mono", "dc_filter")


def utility_values(e: dict) -> dict:
    """A Utility entry's values as every engine plays them (the Utility audit, 2026-10-21, U4/U5):
    each number held to its row of `Session.DEVICE_PARAMS` (`kDeviceParamClamps`), a value that is
    not finite read as its default, and a switch ON only above 0.5 - the native parse, the live
    write and both offline engines read one rule (a stored 0.3 played mono in one and not in the
    other). `invert_left` / `invert_right` unset inherit `invert`."""
    def num(key, default, lo, hi):
        try:
            v = float(e.get(key, default))
        except (TypeError, ValueError):
            v = default
        return min(hi, max(lo, v)) if math.isfinite(v) else default

    def on(key, default=False):
        v = e.get(key, default)
        if v is None:
            return None
        try:
            v = float(v)
        except (TypeError, ValueError):
            return default
        return math.isfinite(v) and v > 0.5

    legacy = bool(on("invert"))
    flip_l, flip_r = on("invert_left", None), on("invert_right", None)
    return {"gain_db": num("gain_db", 0.0, -24.0, 24.0),
            "width": num("width", 1.0, 0.0, 2.0),
            "mono": bool(on("mono")),
            "flip_l": legacy if flip_l is None else bool(flip_l),
            "flip_r": legacy if flip_r is None else bool(flip_r),
            "bass_mono": bool(on("bass_mono")),
            "bass_mono_hz": num("bass_mono_hz", 120.0, 50.0, 500.0),
            "dc_filter": bool(on("dc_filter")),
            "balance": num("balance", 0.0, -1.0, 1.0)}


def lr8(v: np.ndarray, sample_rate: float, shape: str, fc: float) -> np.ndarray:
    """One Linkwitz-Riley 8th-order filter along the last axis: a 4th-order Butterworth's two
    sections, run twice (`lpf2` / `hpf2`) - MonitorChain's BiquadD law, in double."""
    for qq in _LR8_QS + _LR8_QS:
        b, a = biquad_coeffs(shape, sample_rate, fc, qq, 0.0)
        v = lfilter(b, a, v, axis=-1)
    return v


def _utility(audio: np.ndarray, sample_rate: float, e: dict) -> np.ndarray:
    """Ableton-style Utility — the numpy spec for `MonitorChain::applyUtility`: DC filter ->
    gain -> per-channel polarity -> M/S width (mono = width 0) -> bass mono -> balance, the
    values held by `utility_values`. BASS MONO is a Linkwitz-Riley 8th-order split at
    bass_mono_hz (James, 2026-10-21: the band width's and the imager's law): the side below the
    knob is gone (-48 dB an octave down, -6 at the knob), above it untouched, and while it is on
    the whole signal passes the split's allpass. Until the Utility audit it was a one-pole
    (6 dB/oct: -3 dB at the knob, -7 an octave below) under texts that said it folded
    everything below the knob. Order mirrors dsp._utility exactly."""
    u = utility_values(e)
    out = np.asarray(audio, dtype=np.float64)
    if out.ndim == 1:
        out = out[:, np.newaxis]
    if u["dc_filter"]:
        out = out - _one_pole_lp(out, sample_rate, 5.0)
    if u["gain_db"] != 0.0:
        out = out * (10.0 ** (u["gain_db"] / 20.0))
    if u["flip_l"] and u["flip_r"]:
        out = -out
    elif (u["flip_l"] or u["flip_r"]) and out.shape[1] == 2:
        out = out * np.array([[-1.0 if u["flip_l"] else 1.0, -1.0 if u["flip_r"] else 1.0]])
    width = 0.0 if u["mono"] else u["width"]
    if out.shape[1] == 2 and width != 1.0:
        left, right = out[:, 0], out[:, 1]
        mid = 0.5 * (left + right)
        side = 0.5 * (left - right) * width
        out = np.stack([mid + side, mid - side], axis=1)
    if u["bass_mono"] and out.shape[1] == 2:
        t = out.T
        low = lr8(t, sample_rate, "lpf2", u["bass_mono_hz"])
        high = lr8(t, sample_rate, "hpf2", u["bass_mono_hz"])
        out = (high + low.mean(axis=0, keepdims=True)).T
    if u["balance"] != 0.0 and out.shape[1] == 2:
        out = out * np.array([[min(1.0, 1.0 - u["balance"]), min(1.0, 1.0 + u["balance"])]])
    return out


def _multiband_comp(audio: np.ndarray, sample_rate: float, low: dict, mid: dict,
                    high: dict, attack_ms: float, release_ms: float,
                    crossover_lo_hz: float = MB_CROSSOVERS_HZ[0],
                    crossover_hi_hz: float = MB_CROSSOVERS_HZ[1]) -> np.ndarray:
    """The 3-band compressor - `dsp._multiband_comp`, the DAW's law since 2026-10-17 (the real
    3-way LR4 split; it specified the complement split, and the upward band without its makeup).
    (frames, channels) in and out."""
    from .dsp import _multiband_comp as one_law   # dsp imports this module: a late import
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    entry = {"low": low, "mid": mid, "high": high, "attack_ms": attack_ms,
             "release_ms": release_ms, "crossover_lo_hz": crossover_lo_hz,
             "crossover_hi_hz": crossover_hi_hz}
    return one_law(np.ascontiguousarray(x.T.astype(np.float32)), int(sample_rate), entry).T.astype(np.float64)


def _ott(audio: np.ndarray, sample_rate: float, depth: float, time_scale: float,
         in_gain_db: float, out_gain_db: float, low: dict, mid: dict, high: dict,
         crossover_lo_hz: float = 150.0,
         crossover_hi_hz: float = 2000.0) -> np.ndarray:
    """The OTT - `dsp._ott`, the DAW's law since 2026-10-17 (it specified the complement split
    and its own follower). (frames, channels) in and out."""
    from .dsp import _ott as one_law   # dsp imports this module: a late import
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    entry = {"depth": depth, "time": time_scale, "in_gain_db": in_gain_db,
             "out_gain_db": out_gain_db, "low": low, "mid": mid, "high": high,
             "crossover_lo_hz": crossover_lo_hz, "crossover_hi_hz": crossover_hi_hz}
    return one_law(np.ascontiguousarray(x.T.astype(np.float32)), int(sample_rate), entry).T.astype(np.float64)


def _chorus(audio: np.ndarray, sample_rate: float, rate_hz: float = 0.8,
            depth: float = 0.25, delay_ms: float = 20.0, mix: float = 0.5,
            voices: int = 1, stereo_deg: float = 90.0) -> np.ndarray:
    """Chorus / ensemble — the oracle's own copy of ``dsp._chorus`` (this module
    re-implements the engine on purpose and must not import it).

    Voices read the dry input through a delay swept around ``delay_ms``; the LFO phase comes
    from the HOST TIMELINE (sample index / rate), which is what lets the native monitor land
    on the same sweep. Reads use JUCE's Lagrange3rd stencil so the C++
    ``juce::dsp::DelayLine<float, Lagrange3rd>`` and this are one filter, not two."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    n, nch = x.shape
    if n == 0:
        return x
    voices = int(min(3, max(1, int(voices or 1))))
    depth = float(min(1.0, max(0.0, depth)))
    mix = float(min(1.0, max(0.0, mix)))
    idx = np.arange(n, dtype=np.int64)
    t = idx.astype(np.float64) / float(sample_rate)
    max_d = (delay_ms * (1.0 + depth) / 1000.0) * sample_rate + 4.0

    def read(sig, d):
        di = np.floor(d).astype(np.int64)
        f = d - di
        # JUCE's Lagrange3rd `updateInternalVariables()` shifts the stencil back one sample
        # (delayFrac++/delayInt-- for every delay >= 1); mirror it or this is a DIFFERENT
        # filter at the same nominal delay. See the note above `_lagrange3_taps` in dsp.py.
        sh = di >= 1
        di = di - sh
        f = f + sh
        g = [np.where(idx - di - k >= 0, sig[np.clip(idx - di - k, 0, n - 1)], 0.0)
             for k in range(4)]
        d1, d2, d3 = f - 1.0, f - 2.0, f - 3.0
        return (g[0] * (-d1 * d2 * d3 / 6.0)
                + f * (g[1] * (d2 * d3 * 0.5) + g[2] * (-d1 * d3 * 0.5) + g[3] * (d1 * d2 / 6.0)))

    wet = np.zeros_like(x)
    for ch in range(nch):
        # per VOICE SPACING, not per cycle (dsp._chorus carries the reasoning)
        off = (stereo_deg / 360.0 / voices) if ch == 1 else 0.0
        acc = np.zeros(n, dtype=np.float64)
        for v in range(voices):
            ph = 2.0 * np.pi * (rate_hz * t + off + v / voices)
            d = np.clip(delay_ms * (1.0 + depth * np.sin(ph)) / 1000.0 * sample_rate,
                        4.0, max_d)
            acc += read(x[:, ch], d)
        wet[:, ch] = acc / np.sqrt(voices)   # decorrelated voices sum in POWER
    return wet * mix + x * (1.0 - mix) if mix < 1.0 else wet


def _flanger(audio: np.ndarray, sample_rate: float, rate_hz: float = 0.5,
             depth: float = 0.6, delay_ms: float = 3.0, feedback: float = 0.0,
             mix: float = 0.5, stereo_deg: float = 0.0) -> np.ndarray:
    """Flanger — the oracle's own copy of ``dsp._flanger`` (this module re-implements the
    engine on purpose and must not import it).

    Deliberately a DIFFERENT derivation from the engine's. `dsp._flanger` walks the
    recursion in blocks whose length tracks the sweep, recomputed as it moves, with the
    tap indices and coefficients precomputed and the range mask folded into them; this walks
    it in fixed blocks of the sweep's GLOBAL minimum delay, the plainest statement of the
    same recurrence. If the two null, the engine's blocking is what is proven — which is the
    only reason to have an oracle at all."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    n, nch = x.shape
    if n == 0:
        return x
    depth = float(min(1.0, max(0.0, depth)))
    mix = float(min(1.0, max(0.0, mix)))
    fb = float(min(0.95, max(-0.95, feedback)))
    idx = np.arange(n, dtype=np.int64)
    t = idx.astype(np.float64) / float(sample_rate)
    max_d = min((delay_ms * (1.0 + depth) / 1000.0) * sample_rate + 4.0,
                float(int(0.105 * sample_rate) + 8 - 1))

    # per-channel swept delay + the JUCE Lagrange3rd stencil (see `_chorus` above)
    dly, dint, frac = [], [], []
    for ch in range(nch):
        off = (stereo_deg / 360.0) if ch == 1 else 0.0     # one voice: no lattice to dodge
        ph = 2.0 * np.pi * (rate_hz * t + off)
        d = np.clip(delay_ms * (1.0 + depth * np.sin(ph)) / 1000.0 * sample_rate, 4.0, max_d)
        di = np.floor(d).astype(np.int64)
        f = d - di
        sh = di >= 1
        di = di - sh
        f = f + sh
        dly.append(d); dint.append(di); frac.append(f)

    buf = x.copy()
    wet = np.zeros_like(x)
    step = max(1, int(min(int(di.min()) for di in dint)))   # the plainest safe block
    for s in range(0, n, step):
        e = min(s + step, n)
        for ch in range(nch):
            di, f = dint[ch][s:e], frac[ch][s:e]
            b = idx[s:e] - di
            g = [np.where(b - k >= 0, buf[np.clip(b - k, 0, n - 1), ch], 0.0) for k in range(4)]
            d1, d2, d3 = f - 1.0, f - 2.0, f - 3.0
            w = (g[0] * (-d1 * d2 * d3 / 6.0)
                 + f * (g[1] * (d2 * d3 * 0.5) + g[2] * (-d1 * d3 * 0.5) + g[3] * (d1 * d2 / 6.0)))
            wet[s:e, ch] = w
            buf[s:e, ch] = x[s:e, ch] + fb * w
    return wet * mix + x * (1.0 - mix) if mix < 1.0 else wet


# ----------------------------------------------------------------------------- Auto Filter
# Phase 63. ONE law in three engines: the pieces below are the law's closed-form parts, and
# `dsp._auto_filter` and `MonitorChain::applyAutoFilter` compute exactly these (dsp imports them).
# What differs between the engines is only HOW the filter's recursion is walked: here sample by
# sample, the plainest statement; in dsp block-parallel (it must render whole songs); natively
# sample by sample in double. If this and dsp null, dsp's blocking is what is proven.
#
# The filter is a TPT state-variable filter (Zavalishin, "The Art of VA Filter Design" ch. 4):
# g = tan(pi fc / fs) is the bilinear transform with the cutoff prewarped, so its response is
# the analog SVF's at Omega = tan(pi f / fs) / g, and it stays well behaved when g moves every
# sample - which is the whole point of the device. 24 dB/oct is a Butterworth section and then
# the resonant one (never two resonant sections: the EQ's equal-Q cascade measured +127 dB).

AF_Q_BUTTERWORTH_12 = 0.70710678118654752   # the 2nd-order Butterworth Q
AF_Q_BUTTERWORTH_24 = (0.54119610014619698, 1.30656296487637653)   # the 4th order's pair
AF_Q_MAX = 16.0                 # resonance 1 = this Q (the whistle; nothing self-oscillates)
AF_MIN_HZ = 20.0
AF_MAX_FRACTION = 0.49          # of the sample rate: tan() stays finite, as biquad_coeffs holds it
AF_ENV_FLOOR_DB = -48.0         # the envelope reads 0 at -48 dBFS and 1 at full scale
AF_EDGE = 0.02                  # a stepped LFO's edges are raised cosines over 2 % of the cycle
AF_RANDOM_SEED = 0x41465251     # the random shape's hash seed ("AFRQ")


def auto_filter_values(e: dict) -> dict:
    """An Auto Filter entry's parameters, held to the ranges every engine parses (the native
    `parseChain`, `kDeviceParamClamps`), discrete ones snapped. The word values arrive as
    indices (session stores them so); a name is read too, for an entry written by hand."""
    from .models import (AUTO_FILTER_NOTES, AUTO_FILTER_NOTE_NAMES, AUTO_FILTER_SHAPES,
                         AUTO_FILTER_TYPES)

    def num_of(v, default, lo, hi):
        try:
            f = float(v)
        except (TypeError, ValueError):
            f = float(default)
        return min(hi, max(lo, f if math.isfinite(f) else float(default)))

    def num(key, default, lo, hi):
        return num_of(e.get(key, default), default, lo, hi)

    def index(key, default, names):
        v = e.get(key, default)
        if isinstance(v, str):
            low = [n.lower() for n in names]
            v = low.index(v.strip().lower()) if v.strip().lower() in low else default
        return int(round(num_of(v, default, 0, len(names) - 1)))

    slope = num("slope_db_oct", 12, 12, 24)
    note = index("lfo_rate_note", 8, AUTO_FILTER_NOTE_NAMES)
    return {
        "filter_type": index("filter_type", 0, AUTO_FILTER_TYPES),
        "slope_db_oct": 24 if slope >= 18.0 else 12,
        "cutoff_hz": num("cutoff_hz", 1000.0, 20.0, 20000.0),
        "resonance": num("resonance", 0.3, 0.0, 1.0),
        "lfo_amount_oct": num("lfo_amount_oct", 1.0, 0.0, 6.0),
        "lfo_shape": index("lfo_shape", 0, AUTO_FILTER_SHAPES),
        "lfo_sync": 1 if num("lfo_sync", 0, 0, 1) > 0.5 else 0,
        "lfo_rate_hz": num("lfo_rate_hz", 0.5, 0.01, 20.0),
        "lfo_rate_note": note,
        "lfo_quarters": AUTO_FILTER_NOTES[note][1],
        "lfo_phase_deg": num("lfo_phase_deg", 0.0, 0.0, 360.0),
        "lfo_stereo_deg": num("lfo_stereo_deg", 0.0, 0.0, 180.0),
        "env_amount_oct": num("env_amount_oct", 0.0, -6.0, 6.0),
        "env_attack_ms": num("env_attack_ms", 10.0, 0.1, 500.0),
        "env_release_ms": num("env_release_ms", 200.0, 10.0, 3000.0),
        "mix": num("mix", 1.0, 0.0, 1.0),
    }


def auto_filter_qs(resonance: float, slope_db_oct: int) -> tuple:
    """The section Qs: 12 dB = one section, Butterworth at resonance 0 rising to AF_Q_MAX;
    24 dB = the Butterworth pair's low-Q section fixed, then the resonant one rising from the
    pair's high Q. Resonance is exponential in Q, so equal turns are equal steps in peak dB."""
    if slope_db_oct == 24:
        q_fixed, q0 = AF_Q_BUTTERWORTH_24
        return (q_fixed, q0 * (AF_Q_MAX / q0) ** resonance)
    q0 = AF_Q_BUTTERWORTH_12
    return (q0 * (AF_Q_MAX / q0) ** resonance,)


def auto_filter_cycles(n: int, sample_rate: float, v: dict, channel: int,
                       offset: int = 0, tempo=None) -> np.ndarray:
    """The LFO's UNWRAPPED phase in cycles at each sample, on the host timeline (the modulation
    family's rule: never a free accumulator). Hz: rate x seconds. Synced: quarter notes from the
    tempo map (`tempo_map.TempoMap`) over the note's length in quarters - the native reads the
    same quarters from Tracktion's sequence. A project with no map is 4/4 at 120."""
    t = np.arange(offset, offset + n, dtype=np.float64) / float(sample_rate)
    if v["lfo_sync"]:
        if tempo is None:
            from .tempo_map import TempoMap
            tempo = TempoMap.from_fact(None)
        base = tempo.ppq_at(t) / v["lfo_quarters"]
    else:
        base = v["lfo_rate_hz"] * t
    shift = v["lfo_phase_deg"] / 360.0 + (v["lfo_stereo_deg"] / 360.0 if channel == 1 else 0.0)
    return base + shift


def _af_hash01(k: np.ndarray) -> np.ndarray:
    """lowbias32 of the cycle index, as -1..1 (`MonitorChain::afHash`)."""
    i = (np.asarray(k, dtype=np.int64) & 0xFFFFFFFF).astype(np.uint32)
    with np.errstate(over="ignore"):
        h = i * np.uint32(0x9E3779B1) + np.uint32(AF_RANDOM_SEED)
        h ^= h >> np.uint32(16)
        h *= np.uint32(0x7FEB352D)
        h ^= h >> np.uint32(15)
        h *= np.uint32(0x846CA68B)
        h ^= h >> np.uint32(16)
    return (h >> np.uint32(8)).astype(np.float64) * (2.0 / 16777216.0) - 1.0


def auto_filter_lfo(cycles: np.ndarray, shape: int) -> np.ndarray:
    """The LFO's value (-1..1) at each unwrapped phase - a closed form, no state, so the native
    computes the same number from the same phase. Every shape starts at 0 rising, like the sine;
    a stepped one (saw, square, random) crosses its step on a raised cosine over AF_EDGE of the
    cycle, so the cutoff never jumps."""
    c = np.asarray(cycles, dtype=np.float64)
    p = c - np.floor(c)
    E = AF_EDGE
    if shape == 0:                                   # sine
        return np.sin(2.0 * np.pi * p)
    if shape == 1:                                   # triangle
        return 4.0 * np.abs(np.mod(p + 0.75, 1.0) - 0.5) - 1.0
    if shape in (2, 3):                              # saw up / down: the step at p = 0.5
        q = np.mod(p + 0.5, 1.0)
        out = 2.0 * q - 1.0
        hi, lo = 1.0 - E, -1.0 + E
        u = np.where(q < E / 2, (q + E / 2) / E, (q - (1.0 - E / 2)) / E)
        edge = (q < E / 2) | (q > 1.0 - E / 2)
        out = np.where(edge, hi + (lo - hi) * (1.0 - np.cos(np.pi * u)) / 2.0, out)
        return out if shape == 2 else -out
    if shape == 4:                                   # square: up at 0, down at 0.5
        out = np.where(p < 0.5, 1.0, -1.0)
        near0 = (p < E / 2) | (p > 1.0 - E / 2)
        u0 = np.mod(p + E / 2, 1.0) / E
        out = np.where(near0, -np.cos(np.pi * u0), out)
        near5 = np.abs(p - 0.5) < E / 2
        u5 = (p - 0.5 + E / 2) / E
        return np.where(near5, np.cos(np.pi * u5), out)
    # random: a new level each cycle, eased in over the cycle's first AF_EDGE
    k = np.floor(c)
    cur, prev = _af_hash01(k), _af_hash01(k - 1)
    w = np.where(p < E, (1.0 - np.cos(np.pi * p / E)) / 2.0, 1.0)
    return prev + (cur - prev) * w


def auto_filter_env01(level: np.ndarray) -> np.ndarray:
    """The envelope follower's level read as 0..1: 0 at AF_ENV_FLOOR_DB, 1 at full scale."""
    db = 20.0 * np.log10(np.maximum(np.asarray(level, dtype=np.float64), 1.0e-12))
    return np.clip((db - AF_ENV_FLOOR_DB) / -AF_ENV_FLOOR_DB, 0.0, 1.0)


def auto_filter_cutoff(v: dict, sample_rate: float, lfo: np.ndarray,
                       env01) -> np.ndarray:
    """The cutoff each sample: `cutoff * 2^(lfo_amount*lfo + env_amount*env)`, held to
    [AF_MIN_HZ, AF_MAX_FRACTION * fs]."""
    octaves = v["lfo_amount_oct"] * lfo + v["env_amount_oct"] * env01
    fc = v["cutoff_hz"] * np.exp2(octaves)
    return np.clip(fc, AF_MIN_HZ, AF_MAX_FRACTION * float(sample_rate))


def auto_filter_outputs(filter_type: int, k: float) -> tuple:
    """(c1, c2, d): y = c1*v1 + c2*v2 + d*x - LP v2, BP k*v1 (0 dB peak), HP x - k*v1 - v2,
    notch x - k*v1."""
    return ((0.0, 1.0, 0.0), (-k, -1.0, 1.0), (k, 0.0, 0.0), (-k, 0.0, 1.0))[filter_type]


def auto_filter_modulation(x: np.ndarray, sample_rate: float, v: dict, key=None,
                           offset: int = 0, tempo=None, follower=None) -> np.ndarray:
    """The cutoff for every (frame, channel). The envelope follows the louder channel of the
    device's INPUT, or of `key` (the trigger's audio, already aligned) when the entry is keyed;
    `follower(level, sr, attack_ms, release_ms)` is the linear attack/release follower, from
    rest (dsp passes its exact vectorised one; the oracle its own loop)."""
    n, nch = x.shape
    env01 = 0.0
    if v["env_amount_oct"] != 0.0:
        src = x if key is None else key
        level = np.max(np.abs(src), axis=1) if src.ndim == 2 else np.abs(src)
        if level.shape[0] < n:
            level = np.concatenate([level, np.zeros(n - level.shape[0])])
        env = (follower or _af_follower)(level[:n], sample_rate,
                                         v["env_attack_ms"], v["env_release_ms"])
        env01 = auto_filter_env01(env)
    fc = np.empty((n, nch))
    for ch in range(nch):
        cyc = auto_filter_cycles(n, sample_rate, v, ch, offset, tempo)
        fc[:, ch] = auto_filter_cutoff(v, sample_rate, auto_filter_lfo(cyc, v["lfo_shape"]), env01)
    return fc


def _af_follower(level: np.ndarray, sample_rate: float, attack_ms: float,
                 release_ms: float) -> np.ndarray:
    """env = c*env + (1-c)*x from 0, c = attack where x is above the envelope - per sample."""
    a_att = math.exp(-1000.0 / (sample_rate * attack_ms))
    a_rel = math.exp(-1000.0 / (sample_rate * release_ms))
    out = np.empty(level.shape[0])
    env = 0.0
    for i, x in enumerate(level.tolist()):
        c = a_att if x > env else a_rel
        env = c * env + (1.0 - c) * x
        out[i] = env
    return out


def _auto_filter(audio: np.ndarray, sample_rate: float, e: dict, key=None,
                 tempo=None, offset: int = 0) -> np.ndarray:
    """The Auto Filter, sample by sample - the plainest statement of the recursion (only ever
    run on short test signals; dsp renders whole songs block-parallel and must null with it)."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    n, nch = x.shape
    if n == 0:
        return x
    v = auto_filter_values(e)
    fc = auto_filter_modulation(x, sample_rate, v, key, offset, tempo)
    g = np.tan(np.pi * fc / float(sample_rate))
    y = x.copy()
    for q in auto_filter_qs(v["resonance"], v["slope_db_oct"]):
        k = 1.0 / q
        c1, c2, d = auto_filter_outputs(v["filter_type"], k)
        out = np.empty_like(y)
        for ch in range(nch):
            ic1 = ic2 = 0.0
            gs, xs = g[:, ch].tolist(), y[:, ch].tolist()
            col = out[:, ch]
            for i in range(n):
                gi = gs[i]
                a1 = 1.0 / (1.0 + gi * (gi + k))
                a2 = gi * a1
                a3 = gi * a2
                xi = xs[i]
                v3 = xi - ic2
                v1 = a1 * ic1 + a2 * v3
                v2 = ic2 + a2 * ic1 + a3 * v3
                ic1 = 2.0 * v1 - ic1
                ic2 = 2.0 * v2 - ic2
                col[i] = c1 * v1 + c2 * v2 + d * xi
        y = out
    mix = v["mix"]
    return y * mix + x * (1.0 - mix) if mix < 1.0 else y


def _oversampled(x: np.ndarray, os_n: int, shape) -> np.ndarray:
    """Run a memoryless shaper at ``os_n``x (2/4/8) and come back down — the oracle's
    own copy of ``dsp._oversampled`` (this module re-implements the engine on purpose,
    so it must not import it). ``os_n`` 1 is the plain elementwise call."""
    if os_n <= 1:
        return shape(x)
    from scipy.signal import resample_poly
    a = np.asarray(x, dtype=np.float64)
    up = resample_poly(a, os_n, 1, axis=0)
    return resample_poly(shape(up), 1, os_n, axis=0)[: a.shape[0]]


def _overdrive(audio: np.ndarray, sample_rate: float, drive_db: float,
               filter_on: bool = False, freq_hz: float = 800.0, q: float = 0.7,
               tone: float = 0.5, output_db: float = 0.0,
               mix: float = 1.0, oversample: int = 1) -> np.ndarray:
    """The overdrive - `dsp._overdrive`, one function since 2026-10-17 (this was a copy that clamped
    less). (frames, channels) in and out."""
    from .dsp import _overdrive as one_law   # dsp imports this module: a late import
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    return one_law(x, int(sample_rate), {"drive_db": drive_db, "filter_on": filter_on, "freq_hz": freq_hz,
                                         "q": q, "tone": tone, "output_db": output_db, "mix": mix,
                                         "oversample": oversample})


def _match_eq(audio: np.ndarray, sample_rate: float, bands: list) -> np.ndarray:
    """Real-time match-EQ — approximation of dsp._match_eq. The engine renders the
    per-band corrections as one linear-phase FIR (4097 taps, frequency-sampled at
    the band centers); the live monitor applies one broad peaking biquad per band
    instead — zero latency, magnitude-matched, min-phase (the offline export keeps
    the linear-phase FIR)."""
    out = np.asarray(audio, dtype=np.float64)
    if out.ndim == 1:
        out = out[:, np.newaxis]
    for bd in bands:
        out = apply_eq_band(out, sample_rate, "peak", bd["freq_hz"], bd["q"], bd["gain_db"])
    return out


def _tilt(audio: np.ndarray, sample_rate: float, tilt_db: float, pivot_hz: float,
          q: float, output_db: float = 0.0) -> np.ndarray:
    """One-knob tilt EQ — bit-identical to ``dsp._tilt``: a low shelf at −tilt then a
    high shelf at +tilt, both at the pivot. Two standard shelf biquads (the same JUCE
    makers Pedalboard wraps), so it nulls to the float noise floor like the eq bands.
    2026-07-12: plus the device's ``output_db`` post trim (gain staging)."""
    out = apply_eq_band(audio, sample_rate, "low_shelf", pivot_hz, q, -tilt_db)
    out = apply_eq_band(out, sample_rate, "high_shelf", pivot_hz, q, tilt_db)
    if output_db:
        out = out * 10.0 ** (output_db / 20.0)
    return out


_SAT_CURVES = {"tanh": 0, "warm": 1, "tube": 2, "hard": 3}


def _saturate(audio: np.ndarray, drive_db: float, output_db: float = 0.0,
              curve=0, mix: float = 1.0, oversample: int = 1) -> np.ndarray:
    """Memoryless waveshaper — bit-identical to ``dsp._waveshape`` + ``dsp._saturate``
    (the export) and to ``MonitorChain::applySaturation``: ``k = 10^(drive_db/20)``,
    ``u = k·x``, one of four curves divided back by ``k`` (unity at 0), a plain
    ``output_db`` trim, then a dry/wet ``mix``. Until 2026-09-01 this oracle knew only
    the tanh curve at full wet, so the null test for warm / tube / hard and for any
    parallel saturation was run by nothing (sweep P2). Stateless + elementwise."""
    x = np.asarray(audio, dtype=np.float64)
    c = _SAT_CURVES.get(curve, curve) if isinstance(curve, str) else int(curve or 0)
    k = 10.0 ** (drive_db / 20.0)

    def _shape(v):
        if k <= 0.0:
            return v
        u = k * v
        if c == 1:                                            # warm — arctan
            return np.arctan(u) / k
        if c == 2:   # tube — asymmetric (+ tanh, − atan: the NEGATIVE half is softer and louder; a DC offset)
            return np.where(u >= 0.0, np.tanh(u), np.arctan(u)) / k
        if c == 3:                                            # hard — cubic soft-clip, ±1 beyond
            cc = np.clip(u, -1.0, 1.0)
            return (1.5 * cc - 0.5 * cc ** 3) / (1.5 * k)
        return np.tanh(u) / k                                 # tanh (default)

    out = _oversampled(x, int(oversample or 1), _shape)
    if output_db:
        out = out * 10.0 ** (output_db / 20.0)
    mix = float(mix)
    if mix < 1.0:
        out = out * mix + x * (1.0 - mix)
    return out


def _clip(audio: np.ndarray, threshold_db: float, hardness: float,
          output_db: float = 0.0, oversample: int = 1) -> np.ndarray:
    """Pre-limiter clipper — at ``oversample`` 1, bit-identical to ``dsp._clip``:
    blend a soft tanh knee (hardness 0) and a hard brickwall (hardness 1) at the
    linear threshold, then a plain ``output_db`` trim (stateless + elementwise, so
    it nulls to the float noise floor — the C++ applies the same formula per
    sample). v2's oversampled path (2/4/8) shapes at N× via ``resample_poly`` —
    the native monitor uses juce::dsp::Oversampling instead, so oversampled clip
    is the perceptual tier, not a tight null."""
    t = 10.0 ** (threshold_db / 20.0)
    h = min(max(hardness, 0.0), 1.0)
    x = np.asarray(audio, dtype=np.float64)
    n_frames = x.shape[0]
    os_n = int(oversample or 1)
    if os_n > 1:
        from scipy.signal import resample_poly
        x = resample_poly(x, os_n, 1, axis=0)
    soft = t * np.tanh(x / t)
    hard = np.clip(x, -t, t)
    out = (1.0 - h) * soft + h * hard
    if os_n > 1:
        from scipy.signal import resample_poly
        out = resample_poly(out, 1, os_n, axis=0)[:n_frames]
    if output_db:
        out = out * 10.0 ** (output_db / 20.0)
    return out


def _ms(audio: np.ndarray, channel: str, fn) -> np.ndarray:
    """Apply ``fn`` ((frames,channels)->(frames,channels)) to only the mid or side of a
    stereo signal, then recombine (Phase 41 M/S) — mirrors ``dsp._ms_apply``. Stereo or
    mono runs ``fn`` directly; the target channel is processed as a 2-channel mono-dup.
    Phase 46: ``left``/``right`` process ONLY that stereo channel (index 0/1) directly —
    no M/S encode/decode, so the other channel is bit-identical; mono (or any
    non-stereo channel count) degrades left/right to running ``fn`` directly, mirroring
    ``dsp._channel_apply``."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if channel in ("left", "right"):
        if x.shape[1] != 2:
            return fn(x)
        idx = 0 if channel == "left" else 1
        out = x.copy()
        out[:, idx] = fn(x[:, idx:idx + 1])[:, 0]
        return out
    if channel == "stereo" or x.shape[1] != 2:
        return fn(x)
    mid = (x[:, 0] + x[:, 1]) * 0.5
    side = (x[:, 0] - x[:, 1]) * 0.5
    target = mid if channel == "mid" else side
    proc = fn(np.stack([target, target], axis=1))[:, 0]
    if channel == "mid":
        mid = proc
    else:
        side = proc
    return np.stack([mid + side, mid - side], axis=1)


def _width(audio: np.ndarray, width: float) -> np.ndarray:
    """Mid/side width — bit-identical to ``dsp._width`` (memoryless): no-op on mono
    or width == 1, else L/R = mid ± side·width, mid = (L+R)/2, side = (L-R)/2. The
    user's channel-strip width control, now heard live (the bit-matched tier)."""
    x = np.asarray(audio, dtype=np.float64)
    width = min(2.0, max(0.0, float(width)))   # held to the table, as the native (dsp._width)
    if width == 1.0 or x.ndim < 2 or x.shape[1] != 2:
        return x
    mid = (x[:, 0] + x[:, 1]) / 2.0
    side = (x[:, 0] - x[:, 1]) / 2.0 * width
    return np.stack([mid + side, mid - side], axis=1)


def _deess(audio: np.ndarray, sample_rate: float, threshold_db: float,
           freq_hz: float, ratio: float = DEESS_RATIO,
           attack_ms: float = DEESS_ATTACK_MS, release_ms: float = DEESS_RELEASE_MS,
           range_db: float = -80.0, wide: bool = False) -> np.ndarray:
    """Real-time de-esser — per-sample mirror of the offline ``dsp._deess`` v2 law
    (which runs it on 5 ms block envelopes) and of ``MonitorChain::applyDeess``:
    a Linkwitz-Riley 4th-order LP/HP split (two cascaded Butterworth sections per leg —
    the 1st-order pair leaked so much of the ess band back through the low leg that the
    device saturated at ~5 dB of attenuation whatever `ratio`/`range_db` asked for), the
    high band's linked peak drives
    a compressor law over ``threshold_db`` at ``ratio``, the reduction clamped at
    ``range_db``; the gain applies to the high band (split) or the whole signal
    (``wide``). Perceptual tier, not a tight null (block vs per-sample detector)."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if ratio <= 1.0:
        return x                            # not reducing: exactly transparent
    low = _lr4(x, sample_rate, "lpf2", freq_hz)
    high = _lr4(x, sample_rate, "hpf2", freq_hz)
    detect = np.max(np.abs(high), axis=1)              # linked across channels
    level_db = 20.0 * np.log10(np.maximum(detect, 1e-12))
    over = level_db - threshold_db
    target = np.maximum(np.where(over > 0.0, -over * (1.0 - 1.0 / ratio), 0.0),
                        range_db)                       # gain dB <= 0, floored
    a_att = np.exp(-1.0 / (sample_rate * attack_ms / 1000.0))
    a_rel = np.exp(-1.0 / (sample_rate * release_ms / 1000.0))
    prev = 0.0
    gain = np.empty(target.shape[0])
    for i, t in enumerate(target):
        coef = a_att if t < prev else a_rel            # faster toward more reduction
        prev = coef * prev + (1.0 - coef) * t
        gain[i] = 10.0 ** (prev / 20.0)
    if wide:
        return x * gain[:, np.newaxis]                  # wide-band: dip the whole signal
    return low + high * gain[:, np.newaxis]


def _transient(audio: np.ndarray, sample_rate: float, attack_db: float,
               sustain_db: float, speed: float = 1.0,
               output_db: float = 0.0) -> np.ndarray:
    """The attack/sustain shaper - `dsp._transient`, the DAW's law since 2026-10-17 (this was a
    copy of the native's old dB-domain followers). (frames, channels) in and out."""
    from .dsp import _transient as one_law   # dsp imports this module: a late import
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    return one_law(x, int(sample_rate), {"attack_db": attack_db, "sustain_db": sustain_db,
                                         "speed": speed, "output_db": output_db})


def _duck(audio: np.ndarray, sample_rate: float, key: np.ndarray,
          threshold_db: float, depth_db: float, attack_ms: float,
          release_ms: float) -> np.ndarray:
    """Real-time sidechain duck — approximation of ``dsp._duck``. The KEY arrives
    live on the host sidechain input (Phase 41 Option A), not a rendered stem: its
    per-sample level (the LOUDER channel's |x|, in dB - since 2026-10-17, as the native) engages ``depth_db`` of cut
    while above ``threshold_db``, and a one-pole on the reduction amount gives
    attack (engage fast) / release (let go slow). Per-sample causal envelope vs the
    engine's offline 5 ms-block RMS — perceptual, not bit-exact (the export keeps the
    offline duck-free key render). No key (no sidechain) -> released to unity."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    n = x.shape[0]
    if key is None:
        return x
    k = np.asarray(key, dtype=np.float64)
    if k.ndim == 1:
        k = k[:, np.newaxis]
    klevel = np.max(np.abs(k), axis=1)
    m = min(n, klevel.shape[0])
    target = np.zeros(n)
    klevel_db = 20.0 * np.log10(np.maximum(klevel[:m], 1e-12))
    target[:m] = np.where(klevel_db > threshold_db, float(depth_db), 0.0)
    a_att = np.exp(-1.0 / (sample_rate * attack_ms / 1000.0))
    a_rel = np.exp(-1.0 / (sample_rate * release_ms / 1000.0))
    red = 0.0
    gain_lin = np.empty(n)
    for i in range(n):
        t = target[i]
        coef = a_att if t > red else a_rel        # engage fast, release slow
        red = coef * red + (1.0 - coef) * t
        gain_lin[i] = 10.0 ** (-red / 20.0)
    return x * gain_lin[:, np.newaxis]


def _band_width(audio: np.ndarray, sample_rate: float, entry: dict) -> np.ndarray:
    """Per-band stereo width - the DAW's law (MonitorChain::applyBandWidth, the band width audit
    2026-10-21): a real 3-way LR8 split at the band's edges, the band's side scaled, summed back.
    Its own transcription of ``dsp._band_width_scale``: the edges, the 48 dB/oct crossovers (a
    4th-order Butterworth's two sections, run twice), the allpass on the low. Stereo-only."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim < 2 or x.shape[1] != 2:
        return x
    import math as _m
    e_lo, e_hi = float(entry.get("lo_hz") or 0.0), float(entry.get("hi_hz") or 0.0)
    fc, q = float(entry.get("freq_hz") or 0.0), float(entry.get("q") or 0.0)
    labels = ["sub", "bass", "low_mid", "mid", "high_mid", "presence", "brilliance", "air"]
    edges = [20, 60, 150, 400, 1000, 2000, 4000, 8000, 20000]
    if e_lo > 0.0 and e_hi > e_lo:
        lo, hi = e_lo, e_hi
    elif entry.get("band") in labels:
        i = labels.index(entry["band"])
        lo, hi = float(edges[i]), float(edges[i + 1])
    elif fc > 0.0 and q > 0.0:
        lo = fc * (_m.sqrt(1.0 + 1.0 / (4.0 * q * q)) - 1.0 / (2.0 * q))
        hi = lo + fc / q
    else:
        lo, hi = 2000.0, 4000.0
    top = min(20000.0, 0.45 * sample_rate)
    lo = max(20.0, min(lo, top / 1.05))
    hi = max(lo * 1.05, min(min(hi, lo * 100.0), top))
    w = min(2.0, max(0.0, float(entry.get("width", 1.0))))
    qs = (0.54119610014619701, 1.3065629648763766)

    def lr8(v, shape, f):
        for qq in qs + qs:
            b, a = biquad_coeffs(shape, sample_rate, f, qq, 0.0)
            v = lfilter(b, a, v, axis=-1)
        return v
    t = x.T
    low_raw = lr8(t, "lpf2", lo)
    rest = lr8(t, "hpf2", lo)
    low = lr8(low_raw, "lpf2", hi) + lr8(low_raw, "hpf2", hi)
    mid, high = lr8(rest, "lpf2", hi), lr8(rest, "hpf2", hi)
    m = 0.5 * (mid[0] + mid[1])
    sd = 0.5 * (mid[0] - mid[1]) * w
    return np.stack([(low[0] + high[0]) + (m + sd), (low[1] + high[1]) + (m - sd)], axis=1)


# ---------- send/return wet effects (Phase 40 Step 2) ----------------------
# The agent's reverb/delay returns (Phase 38). Unlike comp/de-ess these are
# *bit-exact*, not approximations: the return DSP is built from causal IIR
# recurrences with no lookahead (a feedback comb + a one-pole tone filter), so
# the C++ transcribes them sample-for-sample. The whole return chain is also
# linear time-invariant, so a per-instance local wet send sums (in the DAW) to
# the same result as the offline shared-return topology:
# ``Σ return(sourceᵢ·sendᵢ) == return(Σ sourceᵢ·sendᵢ)`` — see the fidelity test.
REVERB_COMB_DELAYS_MS = (29.7, 37.1, 41.1, 43.7)   # mirror dsp._return_reverb
REVERB_WET_SCALE = 0.35 / len(REVERB_COMB_DELAYS_MS)
_IR_SEEDS = {"room": 0x12001, "hall": 0x12002, "plate": 0x12003}
_IR_DECAY_SCALE = {"room": 0.55, "hall": 1.0, "plate": 0.75}
_IR_WET_SCALE = 0.34   # unit-L2 (energy) normalized before the tone shelf — see dsp._IR_WET_SCALE
_IR_EARLY = ((7.0, 0.6), (11.0, 0.45), (19.0, 0.35))   # a room's reflections (ms, gain) - dsp._IR_EARLY
_IR_EARLY_LIFT = 10.0 ** (8.0 / 20.0)


def _feedback_comb(x: np.ndarray, delay: int, feedback: float) -> np.ndarray:
    """``y[n] = x[n-delay] + feedback·y[n-delay]`` — mirrors ``dsp._feedback_comb``
    (the block-vectorised form of the per-sample recurrence the C++ runs)."""
    n = x.shape[0]
    out = np.zeros_like(x, dtype=np.float64)
    if delay >= n:
        return out
    for start in range(delay, n, delay):
        prev = start - delay
        width = min(delay, n - start)
        out[start:start + width] = (x[prev:prev + width]
                                    + feedback * out[prev:prev + width])
    return out


def _tone_filter(audio: np.ndarray, sample_rate: float, tone: float) -> np.ndarray:
    """One-pole wet-return tone control — mirrors ``dsp._tone_filter``. ``tone=0``
    is dark, ``tone=1`` mostly open. Seeded so ``y[0] == x[0]`` (no click)."""
    if audio.shape[0] == 0:
        return audio
    cutoff_hz = 1200.0 + np.clip(tone, 0.0, 1.0) * 16800.0
    alpha = float(np.exp(-2.0 * np.pi * cutoff_hz / sample_rate))
    feed = 1.0 - alpha
    x = np.asarray(audio, dtype=np.float64)
    out, _ = lfilter([feed], [1.0, -alpha], x, axis=0, zi=alpha * x[:1])
    return out


def _return_delay(audio: np.ndarray, sample_rate: float, entry: dict) -> np.ndarray:
    """Wet-only feedback delay — mirrors ``dsp._return_delay``."""
    n_frames = audio.shape[0]
    # Held to the table (20..2000 ms) as every door holds it - the DAW's ring is two seconds
    # (2026-10-18, the delay audit: a stored 5 s rendered a 5 s echo here).
    delay_ms = float(np.clip(float(entry["delay_ms"]), 20.0, 2000.0))
    delay = max(1, int(delay_ms * sample_rate / 1000.0))
    if n_frames == 0 or delay >= n_frames:
        return np.zeros_like(audio, dtype=np.float64)
    feedback = float(np.clip(entry["feedback"], 0.0, 0.85))
    out = _feedback_comb(np.asarray(audio, dtype=np.float64), delay, feedback)
    return _tone_filter(out, sample_rate, float(entry["tone"]))


def _return_reverb(audio: np.ndarray, sample_rate: float, entry: dict) -> np.ndarray:
    """Wet-only Schroeder-style reverb — mirrors ``dsp._return_reverb``."""
    n_frames = audio.shape[0]
    if n_frames == 0:
        return np.zeros_like(audio, dtype=np.float64)
    pre_delay = int(min(250.0, max(0.0, float(entry.get("pre_delay_ms", 20.0)))) * sample_rate / 1000.0)
    fed = np.zeros_like(audio, dtype=np.float64)
    if pre_delay < n_frames:
        fed[pre_delay:] = audio[: n_frames - pre_delay]
    decay_s = min(12.0, max(float(entry.get("decay_s", 1.8)), 0.2))
    wet = np.zeros_like(audio, dtype=np.float64)
    for delay_ms in REVERB_COMB_DELAYS_MS:
        delay = max(1, int(delay_ms * sample_rate / 1000.0))
        if delay >= n_frames:
            continue
        feedback = min(0.985, 10.0 ** (-3.0 * (delay / sample_rate) / decay_s))
        wet += _feedback_comb(fed, delay, feedback)
    wet *= REVERB_WET_SCALE
    return _tone_filter(wet, sample_rate, float(entry["tone"]))


def _ir_noise(seed: int, n: int) -> np.ndarray:
    """Unit-variance uniform noise from a counter hash - mirrors ``dsp._ir_noise`` and
    ``MonitorChain::irNoise`` (lowbias32 of index + seed, the top 24 bits)."""
    i = np.arange(n, dtype=np.uint32)
    with np.errstate(over="ignore"):
        h = i * np.uint32(0x9E3779B1) + np.uint32(seed)
        h ^= h >> np.uint32(16)
        h *= np.uint32(0x7FEB352D)
        h ^= h >> np.uint32(15)
        h *= np.uint32(0x846CA68B)
        h ^= h >> np.uint32(16)
    u = (h >> np.uint32(8)).astype(np.float64) * (1.0 / 16777216.0)
    return (2.0 * u - 1.0) * np.sqrt(3.0)


def _ir_held(entry: dict) -> tuple[str, float, float, float]:
    mode = str(entry.get("ir", "hall"))
    mode = mode if mode in _IR_SEEDS else "hall"
    return (mode,
            min(8.0, max(0.2, float(entry.get("decay_s", 2.0)))),
            min(1.0, max(0.0, float(entry.get("tone", 0.5)))),
            min(250.0, max(0.0, float(entry.get("pre_delay_ms", 0.0)))))


def _procedural_ir(mode: str, decay_s: float, tone: float,
                   sample_rate: float) -> np.ndarray:
    """Deterministic mono IR - mirrors ``dsp._procedural_ir`` exactly: the noise L2-normalised
    BEFORE the one-pole tone shelf, a room's reflections lifted over the tail."""
    sr = int(sample_rate)
    mode, decay_s, tone, _ = _ir_held({"ir": mode, "decay_s": decay_s, "tone": tone})
    rt60 = max(0.2, decay_s * _IR_DECAY_SCALE[mode])
    n = max(8, int(np.floor(min(rt60, 8.0) * sr + 0.5)))
    t = np.arange(n) / sr
    ir = _ir_noise(_IR_SEEDS[mode], n) * (10.0 ** (-3.0 * t / rt60))
    if mode == "room":
        lift = _IR_EARLY_LIFT * np.sqrt(sr / 1000.0)
        for delay_ms, gain in _IR_EARLY:
            k = int(np.floor(delay_ms * sr / 1000.0 + 0.5))
            if k < n:
                ir[k] += gain * lift
    elif mode == "hall":
        ramp = int(np.floor(0.02 * sr + 0.5))
        ir[:ramp] *= np.arange(ramp) / ramp
    l2 = float(np.sqrt(np.sum(ir ** 2)))   # NOT rms
    if l2 > 1e-12:
        ir = ir / l2 * _IR_WET_SCALE
    return _tone_filter(ir[:, np.newaxis], sr, tone)[:, 0]


def _return_convolution(audio: np.ndarray, sample_rate: float,
                        entry: dict) -> np.ndarray:
    """Wet-only deterministic convolution — mirrors the offline export oracle."""
    from scipy.signal import fftconvolve

    n_frames = audio.shape[0]
    if n_frames == 0:
        return np.zeros_like(audio, dtype=np.float64)
    mode, decay_s, tone, pre_ms = _ir_held(entry)
    pre_delay = int(np.floor(pre_ms * sample_rate / 1000.0 + 0.5))   # the native's llround
    fed = np.zeros_like(audio, dtype=np.float64)
    if pre_delay < n_frames:
        fed[pre_delay:] = audio[: n_frames - pre_delay]
    ir = _procedural_ir(mode, decay_s, tone, sample_rate)
    out = np.zeros_like(fed, dtype=np.float64)
    for channel in range(fed.shape[1]):
        out[:, channel] = fftconvolve(fed[:, channel], ir)[:n_frames]
    return out


def _apply_return_chain(audio: np.ndarray, sample_rate: float,
                        return_entries: list) -> np.ndarray:
    """Render one wet return bus from its summed post-source sends — mirrors
    ``dsp._apply_return_chain``: input gain, then delay, then reverb (wet-only);
    a return with no wet effect contributes silence."""
    wet = np.asarray(audio, dtype=np.float64)
    has_wet_effect = False
    for entry in return_entries:
        kind = entry["type"]
        if kind == "gain":
            if entry["gain_db"]:
                wet = wet * 10.0 ** (entry["gain_db"] / 20.0)
        elif kind == "return_delay":
            has_wet_effect = True
            wet = _return_delay(wet, sample_rate, entry)
        elif kind == "return_reverb":
            has_wet_effect = True
            wet = _return_reverb(wet, sample_rate, entry)
        elif kind == "return_convolution":
            has_wet_effect = True
            wet = _return_convolution(wet, sample_rate, entry)
    if not has_wet_effect:
        return np.zeros_like(audio, dtype=np.float64)
    return wet


def _wet_device(audio: np.ndarray, sample_rate: float, entry: dict) -> np.ndarray:
    """One SPACE device applied IN a chain — mirrors ``dsp._wet_device`` and the
    non-parallelAdd branch of ``MonitorChain::applySendReturns``: wet, then crossfaded
    against the dry by ``mix`` (1.0 = wet only, which is what an FX return runs at and
    what every entry without the key means).

    Convolution uses the same seeded procedural IR as the offline export, so the
    reference remains an audible oracle rather than silently passing the dry input."""
    kind = entry["type"]
    if kind == "return_delay":
        wet = _return_delay(audio, sample_rate, entry)
    elif kind == "return_reverb":
        wet = _return_reverb(audio, sample_rate, entry)
    elif kind == "return_convolution":
        wet = _return_convolution(audio, sample_rate, entry)
    else:
        return audio
    mix = float(entry.get("mix", 1.0))
    if mix >= 1.0:
        return wet
    return wet * mix + np.asarray(audio, dtype=np.float64) * (1.0 - mix)


def _send_return(audio: np.ndarray, sample_rate: float, send_db: float,
                 return_entries: list) -> np.ndarray:
    """Add the parallel wet return for one send: ``dry + return(dry·send)``. The
    send taps the post-chain signal (``dsp.render`` taps the rendered source);
    the dry path is untouched, the wet is summed in (Phase 38 parallel return)."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    send_lin = 10.0 ** (send_db / 20.0)
    wet = _apply_return_chain(x * send_lin, sample_rate, return_entries)
    return x + wet


def brickwall(audio: np.ndarray, sample_rate: float,
              entry: dict) -> np.ndarray:
    """The limiter - ``dsp._brickwall``, the DAW's law (``MonitorChain::applyLookaheadLimiter``),
    which this module used to copy as the old windowed law (2026-10-17, the limiter audit)."""
    from .dsp import _brickwall   # dsp imports this module: a late import
    return _brickwall(np.asarray(audio, dtype=np.float64), int(sample_rate), entry)


def render_monitor(audio: np.ndarray, sample_rate: float,
                   params: dict, sidechain: np.ndarray | None = None,
                   tempo=None) -> np.ndarray:
    """Apply the monitor contract (``monitor_chain.monitor_chain_params``) to
    ``(frames, channels)`` audio, in signal order. Print-only entries are absent
    from ``params['entries']`` by construction, so this is the live subset only.

    ``sidechain`` is the host sidechain-input key buffer (Phase 41): ``duck``
    entries read it; ``None`` means no key is connected, so a duck is a no-op. A keyed
    ``auto_filter`` reads it too (no key: its own input). ``tempo`` (a ``tempo_map.TempoMap``)
    is the song's map for a synced Auto Filter; None = 4/4 at 120."""
    out = np.array(audio, dtype=np.float64, copy=True)
    if out.ndim == 1:
        out = out[:, np.newaxis]
    for e in params["entries"]:
        if not e.get("enabled", True):   # Phase 40 per-device bypass: the native monitor
            continue                     # skips a disabled device's DSP (monitor_chain emits
            # the entry with enabled=false so the tile still shows).
        t = e["type"]
        if t == "gain":
            out = out * 10.0 ** (e["gain_db"] / 20.0)
        elif t == "eq":
            out = _ms(out, e.get("channel", "stereo"),
                      lambda a: apply_eq_band(a, sample_rate, e["shape"], e["freq_hz"],
                                              e["q"], e["gain_db"],
                                              e.get("slope_db_oct", 6)))
            if e.get("output_db"):   # 2026-07-12: the entry's FULL-signal output trim
                out = out * 10.0 ** (e["output_db"] / 20.0)
        elif t == "comp":
            out = _ms(out, e.get("channel", "stereo"),
                      lambda a: _compress(a, sample_rate, e["threshold_db"], e["ratio"],
                                          e["attack_ms"], e["release_ms"],
                                          e.get("makeup_db", 0.0), e.get("mix", 1.0)))
        elif t == "dynamic_eq":
            out = _dynamic_eq(out, sample_rate, e["freq_hz"], e["q"],
                              e["threshold_db"], e["ratio"],
                              e.get("attack_ms", 10.0), e.get("release_ms", 120.0),
                              e["gain_db"])
        elif t == "mb_comp":
            out = _multiband_comp(out, sample_rate, e["low"], e["mid"], e["high"],
                                  e.get("attack_ms", 10.0), e.get("release_ms", 150.0),
                                  e.get("crossover_lo_hz", MB_CROSSOVERS_HZ[0]),
                                  e.get("crossover_hi_hz", MB_CROSSOVERS_HZ[1]))
        elif t == "ott":
            out = _ott(out, sample_rate, e.get("depth", 0.5), e.get("time", 1.0),
                       e.get("in_gain_db", 0.0), e.get("out_gain_db", 0.0),
                       e.get("low", {}), e.get("mid", {}), e.get("high", {}),
                       e.get("crossover_lo_hz", 150.0),
                       e.get("crossover_hi_hz", 2000.0))
        elif t == "overdrive":
            out = _overdrive(out, sample_rate, e.get("drive_db", 12.0),
                             bool(e.get("filter_on", False)),
                             e.get("freq_hz", 800.0), e.get("q", 0.7),
                             e.get("tone", 0.5), e.get("output_db", 0.0),
                             e.get("mix", 1.0), int(e.get("oversample", 1) or 1))
        elif t == "chorus":
            out = _chorus(out, sample_rate, e.get("rate_hz", 0.8), e.get("depth", 0.25),
                          e.get("delay_ms", 20.0), e.get("mix", 0.5),
                          int(e.get("voices", 1) or 1), e.get("stereo_deg", 90.0))
        elif t == "flanger":
            out = _flanger(out, sample_rate, e.get("rate_hz", 0.5), e.get("depth", 0.6),
                           e.get("delay_ms", 3.0), e.get("feedback", 0.0),
                           e.get("mix", 0.5), e.get("stereo_deg", 0.0))
        elif t == "auto_filter":
            keyed = bool(e.get("key_stem_id")) and sidechain is not None
            out = _auto_filter(out, sample_rate, e, key=sidechain if keyed else None,
                               tempo=tempo)
        elif t == "match_eq":
            out = _match_eq(out, sample_rate, e["bands"])
        elif t == "tilt":
            out = _tilt(out, sample_rate, e["tilt_db"], e["pivot_hz"], e["q"],
                        e.get("output_db", 0.0))
        elif t == "saturation" and isinstance(e.get("low"), dict):
            # the 3-band form: the one law, `dsp._multiband_saturate` (2026-10-17, the multiband
            # saturation audit - this rendered a FLAT saturator for it, audit 05 F9)
            from .dsp import _multiband_saturate
            out = _multiband_saturate(np.asarray(out, dtype=np.float64), e, int(sample_rate))
        elif t == "saturation":
            # curve + mix are part of the contract (monitor_chain_params emits them, the
            # native implements them)
            out = _saturate(out, e["drive_db"], e.get("output_db", 0.0),
                            curve=e.get("curve", 0), mix=float(e.get("mix", 1.0)),
                            oversample=int(e.get("oversample", 1) or 1))
        elif t == "clip":
            out = _clip(out, e["threshold_db"], e["hardness"], e.get("output_db", 0.0),
                        int(e.get("oversample", 1) or 1))
        elif t == "width":
            out = _width(out, e["width"])
        elif t == "utility":
            out = _utility(out, sample_rate, e)
        elif t == "deess":
            out = _deess(out, sample_rate, e["threshold_db"], e["freq_hz"],
                         e.get("ratio", DEESS_RATIO),
                         e.get("attack_ms", DEESS_ATTACK_MS),
                         e.get("release_ms", DEESS_RELEASE_MS),
                         e.get("range_db", -80.0), bool(e.get("wide", False)))
        elif t == "gate":
            out = _gate(out, sample_rate, e["threshold_db"], e["ratio"],
                        e["attack_ms"], e["release_ms"],
                        e.get("hold_ms", 0.0), e.get("range_db", -80.0),
                        e.get("hysteresis_db", 0.0),
                        e.get("sc_hpf_hz", 0.0), e.get("sc_lpf_hz", 20000.0))
        elif t == "transient":
            out = _transient(out, sample_rate, e["attack_db"], e["sustain_db"],
                             e.get("speed", 1.0), e.get("output_db", 0.0))
        elif t == "band_width":
            out = _band_width(out, sample_rate, e)
        elif t == "imager":
            out = _imager(out, sample_rate, e)
        elif t == "duck":
            out = _duck(out, sample_rate, sidechain, e["threshold_db"],
                        e["depth_db"], e["attack_ms"], e["release_ms"])
        elif t == "send_return":
            out = _send_return(out, sample_rate, e["send_db"], e["return_entries"])
        elif t in ("return_delay", "return_reverb", "return_convolution"):
            # A SPACE device sitting IN this chain (2026-07-20): an FX return's wet engine
            # (mix 1.0 → wet replaces) or an ordinary dry/wet insert on a track/bus/master.
            # Mirrors dsp._wet_device and MonitorChain's non-parallelAdd branch.
            out = _wet_device(out, sample_rate, e)
        elif t == "limiter":
            if e.get("makeup_db"):
                out = out * 10.0 ** (e["makeup_db"] / 20.0)
            out = brickwall(out, sample_rate, e)
    return out
