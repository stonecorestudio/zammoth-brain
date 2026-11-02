"""The chorus's law (2026-10-28): a reference two-line chorus, measured as a black box.

James's pick after hearing A (ours) / REF / B (this law) on a released pop master in each of the
reference's three modes: B, with the reference's knob set. The chorus has no rate, depth, delay or
voices any more: MODE (I, II or I+II), STEREO (0..1), MIX (the reference's dry / wet curve) and
VOLUME (`output_db`, a trim after it).

Measured with sines (demodulated for the delay), clicks, white noise, a level sweep and the song:

    lines     each mode is one or two delay lines swept by a pure TRIANGLE (its 3rd / 5th / 7th
              harmonics 1/9, 1/25, 1/49 of the fundamental to four places), the right channel's
              sweep the left's half a cycle on, each line reading ITS OWN channel:
                  I    0.39986 Hz, +-1.9502 ms around 3.2524 ms, the left at a quarter cycle
                  II   0.82982 Hz, +-2.0964 ms around 3.5021 ms, the left at three quarters
                  I+II both lines, summed
              The phase is the host timeline's (sample index / rate), as every modulation device here.
    tone      the wet only: a 1st-order high-pass at 38 Hz (60 Hz when both lines play - measured,
              the one thing I+II changes), then a 1st-order shelf with a DC gain of 1.3455 (+2.6 dB),
              its pole at 3456 Hz and its zero at 9973 Hz (-3 dB near 6 kHz, -9 dB at the top). The
              reference applies its filter per SAMPLE, so it is darker at 44.1 kHz than at 96 kHz;
              this law is the reference at 48 kHz, designed from the analog shape at every rate.
    read      the reference reads its line with an allpass interpolator; this law reads it with the
              Lagrange stencil every modulation device here shares (dsp._lagrange3_taps). On the song
              the difference is in the top octave: 1 dB of null.
    stereo    L = the left's wet; R = (1 - stereo) x the left's wet + stereo x the right's.
    mix       up to 0.5 the dry stays near full and the wet comes in: dry 1 + w - 2w^2, wet w + 2w^2
              - a plain linear blend ((1 - w) dry + w wet) times (1 + 2w); 0.5 = both at 1, +4.5 dB
              over the dry on the song; above 0.5 the mirror image.
    linear    no saturation (H3 -92 dB at any level), no noise, no dependence on block size.

On the song (3-24 s, wet 0.5) this law nulls 34.6 / 32.2 dB under what the reference changed in I /
II; the chorus it replaces did not null at all (+1..+2 dB).

A chorus saved before (`rate_hz`, `depth`, `delay_ms`, `voices`, `stereo_deg`) is read once by
`from_legacy`: two or three voices = I+II, else a rate up to 0.6 Hz = I and a faster one II; STEREO =
stereo_deg / 180; MIX unchanged (the reference curve IS the old linear blend, louder by 1 + 2w, so
the same knob keeps the same wet-to-dry balance), and VOLUME takes back what that adds in level (a
power estimate), so a project plays about as loud as it did.

`native/Source/ChorusLaw.h` is written from this file (`python -m mix_engine.chorus_law
--write-header`); test_chorus_law.py holds the two equal.
"""
from __future__ import annotations

import math
import sys
from typing import Any

import numpy as np

MODES = ("I", "II", "I+II")
#: per line: rate (Hz), swing either side (ms), centre (ms), the left channel's phase (turns)
LINES = ((0.39986, 1.9502, 3.2524, 0.25),
         (0.82982, 2.0964, 3.5021, 0.75))
HP_ONE_HZ = 38.0
HP_BOTH_HZ = 60.0
TONE_GAIN = 1.3455
TONE_POLE_HZ = 3456.23
TONE_ZERO_HZ = 9973.2
OUTPUT_MIN_DB, OUTPUT_MAX_DB = -24.0, 6.0
MIN_DELAY_SAMPLES = 4.0          # the Lagrange stencil's floor (dsp.CHORUS_MIN_DELAY_SAMPLES)
DEFAULTS = {"mode": 0, "stereo": 1.0, "mix": 0.5, "output_db": 0.0}
LEGACY_FIELDS = ("rate_hz", "depth", "delay_ms", "voices", "stereo_deg")
NEW_FIELDS = ("mode", "stereo", "output_db")


def mode_index(v: Any) -> int:
    """A mode as stored (0..2) or named ("I", "II", "I+II", any case)."""
    if isinstance(v, str):
        s = v.strip().upper().replace(" ", "")
        if s in MODES:
            return MODES.index(s)
        try:
            v = float(s)
        except ValueError:
            return DEFAULTS["mode"]
    try:
        f = float(v)
    except (TypeError, ValueError):
        return DEFAULTS["mode"]
    if not math.isfinite(f):
        return DEFAULTS["mode"]
    return int(min(max(round(f), 0), 2))


def lines_of(mode: int) -> tuple[int, ...]:
    return ((0,), (1,), (0, 1))[mode]


def dry_wet(mix: float) -> tuple[float, float]:
    """The reference's dry / wet gains at a MIX of 0..1."""
    w = min(max(float(mix), 0.0), 1.0)
    if w <= 0.5:
        return 1.0 + w - 2.0 * w * w, w + 2.0 * w * w
    d, wt = dry_wet(1.0 - w)
    return wt, d


def is_legacy(entry: dict) -> bool:
    """A chorus saved before this law: it names an old knob and none of the new ones."""
    return (any(k in entry for k in LEGACY_FIELDS)
            and not any(k in entry for k in NEW_FIELDS))


def level_db(mix: float, mode: int) -> float:
    """What a chorus adds in level, as a power sum of the dry and each line's toned wet (the lines
    decorrelate from the dry and from each other as they sweep). On the song: I / II +4.5 dB at MIX
    0.5 (measured +4.6 / +4.5), I+II +6.6 (measured +7.4) - an estimate, a little under on I+II."""
    dry, wet = dry_wet(mix)
    return 10.0 * math.log10(dry * dry + len(lines_of(mode)) * (wet * TONE_GAIN) ** 2)


def _num(entry: dict, key: str, default: float) -> float:
    try:
        v = float(entry.get(key, default))
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def legacy_mix(old_mix: float) -> float:
    """An old chorus's MIX: the same number - the reference curve is the old linear blend times
    (1 + 2w) (and its mirror), so the wet-to-dry balance is the same at the same knob."""
    return round(min(max(float(old_mix), 0.0), 1.0), 4)


def from_legacy(entry: dict) -> dict:
    """An old chorus's knobs as this law's (see the module text)."""
    rate = min(max(_num(entry, "rate_hz", 0.8), 0.01), 10.0)
    voices = int(min(max(round(_num(entry, "voices", 1.0)), 1), 3))
    mode = 2 if voices >= 2 else (0 if rate <= 0.6 else 1)
    stereo = min(max(_num(entry, "stereo_deg", 90.0), 0.0), 180.0) / 180.0
    m_old = min(max(_num(entry, "mix", 0.5), 0.0), 1.0)
    mix = legacy_mix(m_old)
    old_db = 10.0 * math.log10(max((1.0 - m_old) ** 2 + m_old ** 2, 1e-12))
    out = round(min(max(old_db - level_db(mix, mode), OUTPUT_MIN_DB), OUTPUT_MAX_DB), 1)
    return {"mode": mode, "stereo": round(stereo, 4), "mix": mix, "output_db": out}


def values(entry: dict) -> dict:
    """The four knobs of a stored entry, held to their ranges - an old one read through
    `from_legacy`. Every reader of a chorus entry (the render, the monitor resolver, the reference
    monitor, the guardrails) goes through here."""
    v = from_legacy(entry) if is_legacy(entry) else entry
    return {"mode": mode_index(v.get("mode", DEFAULTS["mode"])),
            "stereo": min(max(_num(v, "stereo", 1.0), 0.0), 1.0),
            "mix": min(max(_num(v, "mix", 0.5), 0.0), 1.0),
            "output_db": min(max(_num(v, "output_db", 0.0), OUTPUT_MIN_DB), OUTPUT_MAX_DB)}


def lift_db(entry: dict) -> float:
    """What the chorus adds in level (the gain wallet's charge): the dry + toned wet (`level_db`)
    plus VOLUME. Signed: a chorus turned down by its VOLUME takes away."""
    v = values(entry)
    return level_db(v["mix"], v["mode"]) + v["output_db"]


def tri(phase_rad: np.ndarray) -> np.ndarray:
    """The triangle the lines sweep by: +1 at a quarter cycle, -1 at three quarters (asin of sin)."""
    return (2.0 / np.pi) * np.arcsin(np.sin(phase_rad))


def _first_order_highpass(fc: float, sr: float) -> tuple[np.ndarray, np.ndarray]:
    """The bilinear 1st-order Butterworth high-pass, prewarped (scipy's butter(1, fc, 'high'))."""
    k = math.tan(math.pi * fc / sr)
    return np.array([1.0, -1.0]) / (1.0 + k), np.array([1.0, (k - 1.0) / (1.0 + k)])


def _tone(sr: float) -> tuple[np.ndarray, np.ndarray]:
    """The shelf g (1 + s/wz) / (1 + s/wp), bilinear (unwarped) at `sr`."""
    big = 2.0 * sr
    wz, wp = 2.0 * math.pi * TONE_ZERO_HZ, 2.0 * math.pi * TONE_POLE_HZ
    b0, b1 = TONE_GAIN * (1.0 + big / wz), TONE_GAIN * (1.0 - big / wz)
    a0, a1 = 1.0 + big / wp, 1.0 - big / wp
    return np.array([b0 / a0, b1 / a0]), np.array([1.0, a1 / a0])


def wet_coeffs(mode: int, sr: float) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """The wet's two first-order filters, (b, a) for the high-pass then the shelf."""
    return _first_order_highpass(HP_BOTH_HZ if mode == 2 else HP_ONE_HZ, sr), _tone(sr)


def delay_samples(line: int, right: bool, n: int, sr: float, offset: int = 0) -> np.ndarray:
    """One line's delay, in samples, for samples offset..offset+n of the host timeline."""
    rate, swing, centre, ph = LINES[line]
    t = np.arange(offset, offset + n, dtype=np.float64) / float(sr)
    phase = 2.0 * np.pi * (rate * t + ph + (0.5 if right else 0.0))
    return (centre + swing * tri(phase)) / 1000.0 * sr


def apply(audio: np.ndarray, sample_rate: float, entry: dict, timeline_offset: int = 0) -> np.ndarray:
    """The chorus on (frames, channels) float64. `timeline_offset` is the host sample the audio
    starts on (the sweep's phase); a render starts on 0."""
    from scipy.signal import lfilter
    from .dsp import _lagrange3_taps
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    n, nch = x.shape
    if n == 0:
        return x
    v = values(entry)
    sr = float(sample_rate)
    (hb, ha), (tb, ta) = wet_coeffs(v["mode"], sr)
    wet = np.zeros((n, min(nch, 2)))
    chunk = 1 << 19      # bounded temporaries (the gather reads the whole channel at absolute indices)
    for ch in range(wet.shape[1]):
        col = x[:, ch]
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            for line in lines_of(v["mode"]):
                d = np.maximum(delay_samples(line, ch == 1, stop - start, sr, offset=start + timeline_offset),
                               MIN_DELAY_SAMPLES)
                wet[start:stop, ch] += _lagrange3_taps(col, d, offset=start)
        wet[:, ch] = lfilter(tb, ta, lfilter(hb, ha, wet[:, ch]))
    gd, gw = dry_wet(v["mix"])
    out = x.copy() * gd
    out[:, 0] += gw * wet[:, 0]
    if nch >= 2:
        s = v["stereo"]
        out[:, 1] += gw * ((1.0 - s) * wet[:, 0] + s * wet[:, 1])
    if v["output_db"] != 0.0:
        out *= 10.0 ** (v["output_db"] / 20.0)
    return out


#: what an old chorus's ride / co-edit of a field means now: None = the control is gone
def legacy_field(field: str, value: float) -> tuple[str, float] | None:
    if field == "stereo_deg":
        return "stereo", min(max(value, 0.0), 180.0) / 180.0
    if field == "mix":
        return "mix", legacy_mix(value)
    if field in ("rate_hz", "depth", "delay_ms", "voices"):
        return None
    return field, value


def convert_legacy(obj) -> int:
    """Walk anything read from an old file and convert every chorus in it in place (a dict of type
    "chorus" holding the old knobs): the session's chains, its undo snapshots, its recorded actions.
    Returns how many it converted (session STATE_VERSION 9)."""
    n = 0
    if isinstance(obj, dict):
        if obj.get("type") == "chorus" and is_legacy(obj):
            conv = from_legacy(obj)
            for k in LEGACY_FIELDS:
                obj.pop(k, None)
            obj.update(conv)
            n += 1
        for v in obj.values():
            if isinstance(v, (dict, list)):
                n += convert_legacy(v)
    elif isinstance(obj, list):
        for v in obj:
            n += convert_legacy(v)
    return n


def header_text() -> str:
    def lit(x: float) -> str:
        return repr(float(x))
    lines = ", ".join("{ " + ", ".join(lit(c) for c in row) + " }" for row in LINES)
    L = [
        "#pragma once",
        "// GENERATED by `python -m mix_engine.chorus_law --write-header` from",
        "// server/mix_engine/chorus_law.py - the chorus's law (that file says how it was measured).",
        "// Do not edit: change the Python and write this again; test_chorus_law.py holds the two equal.",
        "",
        "#include <cmath>",
        "",
        "namespace choruslaw",
        "{",
        "    // per line: rate (Hz), swing either side (ms), centre (ms), the left channel's phase (turns)",
        f"    static constexpr double kLines[2][4] = {{ {lines} }};",
        f"    static constexpr double kHpOneHz = {lit(HP_ONE_HZ)}, kHpBothHz = {lit(HP_BOTH_HZ)};",
        f"    static constexpr double kToneGain = {lit(TONE_GAIN)}, kTonePoleHz = {lit(TONE_POLE_HZ)}, "
        f"kToneZeroHz = {lit(TONE_ZERO_HZ)};",
        f"    static constexpr double kOutputMinDb = {lit(OUTPUT_MIN_DB)}, kOutputMaxDb = {lit(OUTPUT_MAX_DB)};",
        f"    static constexpr double kMinDelaySamples = {lit(MIN_DELAY_SAMPLES)};",
        f"    static constexpr int kDefaultMode = {DEFAULTS['mode']};",
        f"    static constexpr double kDefaultStereo = {lit(DEFAULTS['stereo'])}, kDefaultMix = "
        f"{lit(DEFAULTS['mix'])};",
        "",
        "    inline bool lineOn (int mode, int line) noexcept { return mode == 2 || mode == line; }",
        "",
        "    // the reference's dry / wet gains at a MIX of 0..1 (chorus_law.dry_wet)",
        "    inline void dryWet (double mix, double& dry, double& wet) noexcept",
        "    {",
        "        double w = mix < 0.0 ? 0.0 : (mix > 1.0 ? 1.0 : mix);",
        "        const bool hi = w > 0.5;",
        "        if (hi) w = 1.0 - w;",
        "        const double d = 1.0 + w - 2.0 * w * w, wt = w + 2.0 * w * w;",
        "        dry = hi ? wt : d; wet = hi ? d : wt;",
        "    }",
        "",
        "    // what a chorus adds in level, a power sum of the dry and each line's toned wet (chorus_law.level_db)",
        "    inline double levelDb (double mix, int mode) noexcept",
        "    {",
        "        double d, w; dryWet (mix, d, w);",
        "        return 10.0 * std::log10 (d * d + (mode == 2 ? 2.0 : 1.0) * (w * kToneGain) * (w * kToneGain));",
        "    }",
        "",
        "    // one line's delay in samples at host sample `clock` (chorus_law.delay_samples). The triangle is",
        "    // `tri`'s asin (sin (2 pi u)) in its exact piecewise form: no trig on the audio thread (two",
        "    // calls a line, a channel and a sample were a tenth of a busy session's CPU - 2026-10-29)",
        "    inline double delaySamples (int line, bool right, double clock, double sr) noexcept",
        "    {",
        "        const double* L = kLines[line];",
        "        const double t = clock / sr;",
        "        double u = L[0] * t + L[3] + (right ? 0.5 : 0.0);",
        "        u -= std::floor (u);",
        "        const double tri = u < 0.25 ? 4.0 * u : (u < 0.75 ? 2.0 - 4.0 * u : 4.0 * u - 4.0);",
        "        return (L[2] + L[1] * tri) / 1000.0 * sr;",
        "    }",
        "",
        "    // the wet's two first-order filters as { b0, b1, a1 } (chorus_law.wet_coeffs): the high-pass",
        "    // (prewarped bilinear Butterworth), then the shelf (unwarped bilinear)",
        "    inline void wetCoeffs (int mode, double sr, double hp[3], double tone[3]) noexcept",
        "    {",
        "        const double k = std::tan (3.14159265358979323846 * (mode == 2 ? kHpBothHz : kHpOneHz) / sr);",
        "        hp[0] = 1.0 / (1.0 + k); hp[1] = -1.0 / (1.0 + k); hp[2] = (k - 1.0) / (1.0 + k);",
        "        const double big = 2.0 * sr;",
        "        const double wz = 2.0 * 3.14159265358979323846 * kToneZeroHz, wp = 2.0 * 3.14159265358979323846 * kTonePoleHz;",
        "        const double a0 = 1.0 + big / wp;",
        "        tone[0] = kToneGain * (1.0 + big / wz) / a0;",
        "        tone[1] = kToneGain * (1.0 - big / wz) / a0;",
        "        tone[2] = (1.0 - big / wp) / a0;",
        "    }",
        "",
        "    // an old chorus's MIX: the same number (the curve is the old linear blend times 1 + 2w)",
        "    inline double legacyMix (double m) noexcept",
        "    {",
        "        m = m < 0.0 ? 0.0 : (m > 1.0 ? 1.0 : m);",
        "        return std::round (m * 1.0e4) / 1.0e4;",
        "    }",
        "",
        "    struct Knobs { int mode; double stereo, mix, output_db; };",
        "    // an old chorus's knobs (chorus_law.from_legacy)",
        "    inline Knobs fromLegacy (double rateHz, double voices, double stereoDeg, double oldMix) noexcept",
        "    {",
        "        rateHz = rateHz < 0.01 ? 0.01 : (rateHz > 10.0 ? 10.0 : rateHz);",
        "        const int v = (int) std::nearbyint (voices < 1.0 ? 1.0 : (voices > 3.0 ? 3.0 : voices));   // Python's round",
        "        Knobs k;",
        "        k.mode = v >= 2 ? 2 : (rateHz <= 0.6 ? 0 : 1);",
        "        stereoDeg = stereoDeg < 0.0 ? 0.0 : (stereoDeg > 180.0 ? 180.0 : stereoDeg);",
        "        k.stereo = std::round (stereoDeg / 180.0 * 1.0e4) / 1.0e4;",
        "        const double m = oldMix < 0.0 ? 0.0 : (oldMix > 1.0 ? 1.0 : oldMix);",
        "        k.mix = legacyMix (m);",
        "        const double oldDb = 10.0 * std::log10 (std::fmax ((1.0 - m) * (1.0 - m) + m * m, 1.0e-12));",
        "        double out = oldDb - levelDb (k.mix, k.mode);",
        "        out = out < kOutputMinDb ? kOutputMinDb : (out > kOutputMaxDb ? kOutputMaxDb : out);",
        "        k.output_db = std::round (out * 10.0) / 10.0;",
        "        return k;",
        "    }",
        "}",
        "",
    ]
    return "\n".join(L)


if __name__ == "__main__":
    if "--write-header" in sys.argv:
        from pathlib import Path
        out = Path(__file__).resolve().parents[2] / "native" / "Source" / "ChorusLaw.h"
        out.write_text(header_text())
        print(f"wrote {out}")
