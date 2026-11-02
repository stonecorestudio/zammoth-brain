"""Oversampled waveshapers (2026-09-06).

Saturation and overdrive are hard nonlinearities. Run at the base rate they generate
harmonics above Nyquist, and those FOLD BACK into the audible band as inharmonic
aliasing — a defect no downstream EQ can undo. `oversample` (1/2/4/8) runs the shaper
at N x so the fold point moves N x higher and the decimation filter discards the rest.

These tests MEASURE the fold-back rather than merely exercising the code path: a 9 kHz
tone driven hard puts its 5th harmonic at 45 kHz, which at 48 kHz reflects to 3 kHz —
a frequency the input does not contain and no harmonic of it lands on. That 3 kHz bin
is the aliasing, and it is what must drop when oversampling is on.
"""
import numpy as np
import pytest

from mix_engine import dsp, reference_monitor

SR = 48_000
TONE_HZ = 9_000.0        # 5th harmonic = 45 kHz -> folds to |48000 - 45000| = 3 kHz
ALIAS_HZ = 3_000.0       # inharmonic: not the tone, not any harmonic of it
N = SR                   # 1 s


def _tone(amp: float = 0.9) -> np.ndarray:
    t = np.arange(N, dtype=np.float64) / SR
    x = amp * np.sin(2.0 * np.pi * TONE_HZ * t)
    return np.repeat(x[:, np.newaxis], 2, axis=1)


def _band_db(audio: np.ndarray, centre_hz: float, half_width_hz: float = 60.0) -> float:
    """Energy in a narrow band, in dB — averaged over channels."""
    mono = np.asarray(audio, dtype=np.float64).mean(axis=1)
    spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
    freqs = np.fft.rfftfreq(len(mono), 1.0 / SR)
    sel = (freqs >= centre_hz - half_width_hz) & (freqs <= centre_hz + half_width_hz)
    return 20.0 * np.log10(max(float(np.sqrt(np.sum(spec[sel] ** 2))), 1e-12))


# ---------------------------------------------------------------- saturation

def test_saturation_oversampling_removes_the_fold_back():
    """The 3 kHz alias is real at 1x and must fall away at 8x."""
    base = {"type": "saturation", "stem_id": "a", "drive_db": 12.0, "curve": 3}
    dirty = dsp._saturate(_tone(), dict(base, oversample=1), SR)
    clean = dsp._saturate(_tone(), dict(base, oversample=8), SR)

    alias_1x = _band_db(dirty, ALIAS_HZ)
    alias_8x = _band_db(clean, ALIAS_HZ)

    # the defect exists at all (guards against a vacuous test if the curve ever softens)
    assert alias_1x > _band_db(dirty, ALIAS_HZ + 900.0) + 12.0, (
        f"no measurable alias to remove at 1x ({alias_1x:.1f} dB) - test proves nothing")
    assert alias_8x < alias_1x - 20.0, (
        f"8x did not suppress the fold-back: 1x {alias_1x:.1f} dB -> 8x {alias_8x:.1f} dB")


def test_saturation_keeps_the_wanted_harmonics():
    """Oversampling must remove the ALIAS, not the saturation itself. Proved on a 1 kHz
    tone, whose odd harmonics (3/5/7 kHz) all sit well inside the band and so must
    survive untouched - if they moved, we would have built a low-pass, not an
    anti-aliasing stage. (The 9 kHz probe tone cannot show this: a cubic hard-clip is
    odd-symmetric, so its 3rd harmonic is already at 27 kHz, above Nyquist.)"""
    t = np.arange(N, dtype=np.float64) / SR
    low = np.repeat((0.9 * np.sin(2.0 * np.pi * 1_000.0 * t))[:, np.newaxis], 2, axis=1)
    base = {"type": "saturation", "stem_id": "a", "drive_db": 12.0, "curve": 3}
    dirty = dsp._saturate(low, dict(base, oversample=1), SR)
    clean = dsp._saturate(low, dict(base, oversample=8), SR)
    for h in (3_000.0, 5_000.0, 7_000.0):
        a, b = _band_db(dirty, h), _band_db(clean, h)
        assert b > _band_db(clean, 1_000.0) - 60.0, f"{h:.0f} Hz harmonic missing at 8x"
        assert abs(a - b) < 1.0, f"{h:.0f} Hz harmonic moved {a:.1f} -> {b:.1f} dB at 8x"

    # and the 9 kHz probe: at 8x the fundamental towers over the fold-back bin
    clean9 = dsp._saturate(_tone(), dict(base, oversample=8), SR)
    assert _band_db(clean9, TONE_HZ) > _band_db(clean9, ALIAS_HZ) + 30.0


def test_saturation_oversample_1_is_bit_identical_to_the_legacy_path():
    """1 (the default) must not change a single sample of any older render."""
    # an entry with no key runs at the default (dsp.SAT_OVERSAMPLE_DEFAULT, 2026-10-24); 1 is the
    # elementwise legacy path, bit for bit
    entry = {"type": "saturation", "stem_id": "a", "drive_db": 6.0, "curve": 0}
    explicit = dsp._saturate(_tone(), dict(entry, oversample=1), SR)
    assert np.array_equal(explicit, dsp._waveshape(_tone(), 6.0, 0))
    unkeyed = dsp._saturate(_tone(), dict(entry), SR)
    assert np.array_equal(unkeyed, dsp._saturate(_tone(), dict(entry, oversample=dsp.SAT_OVERSAMPLE_DEFAULT), SR))


def test_saturation_monotonic_in_the_factor():
    """More oversampling must not make aliasing worse — and must actually CHANGE it.

    `got == sorted(got, reverse=True)` is true for a CONSTANT list, so this passed on a dead
    feature: neutering `dsp._oversampled` to `shape(x)` left it green while three sibling tests
    failed correctly (2026-09-06 review). A strict decrease is what the claim actually is.
    """
    base = {"type": "saturation", "stem_id": "a", "drive_db": 12.0, "curve": 3}
    got = [_band_db(dsp._saturate(_tone(), dict(base, oversample=n), SR), ALIAS_HZ)
           for n in (1, 2, 4, 8)]
    assert all(b < a for a, b in zip(got, got[1:])), (
        f"alias did not fall STRICTLY with the factor: {got}"
    )
    assert got[0] - got[-1] > 20.0, (
        f"8x removed only {got[0] - got[-1]:.1f} dB of fold-back — the factor is barely doing "
        f"anything: {got}"
    )


# ----------------------------------------------------------------- overdrive

def test_overdrive_oversampling_removes_the_fold_back():
    base = {"type": "overdrive", "stem_id": "a", "drive_db": 24.0, "tone": 0.5}
    dirty = dsp._overdrive(_tone(), SR, dict(base, oversample=1))
    clean = dsp._overdrive(_tone(), SR, dict(base, oversample=8))

    alias_1x = _band_db(dirty, ALIAS_HZ)
    alias_8x = _band_db(clean, ALIAS_HZ)
    assert alias_1x > _band_db(dirty, ALIAS_HZ + 900.0) + 12.0, (
        f"no measurable alias to remove at 1x ({alias_1x:.1f} dB) - test proves nothing")
    assert alias_8x < alias_1x - 20.0, (
        f"8x did not suppress the fold-back: 1x {alias_1x:.1f} dB -> 8x {alias_8x:.1f} dB")


def test_an_unkeyed_overdrive_runs_at_the_default_factor():
    # dsp.OD_OVERSAMPLE_DEFAULT since 2026-10-24 (the voicing changed then too, so there is no
    # "legacy path" left to be bit-identical to)
    entry = {"type": "overdrive", "stem_id": "a", "drive_db": 12.0, "tone": 0.5}
    unkeyed = dsp._overdrive(_tone(), SR, dict(entry))
    explicit = dsp._overdrive(_tone(), SR, dict(entry, oversample=dsp.OD_OVERSAMPLE_DEFAULT))
    assert np.array_equal(unkeyed, explicit)


def test_overdrive_tone_filter_still_runs_when_oversampled():
    """Only the shaper is oversampled; the tone tilt must still act on the result."""
    base = {"type": "overdrive", "stem_id": "a", "drive_db": 18.0, "oversample": 4}
    dark = dsp._overdrive(_tone(), SR, dict(base, tone=0.0))
    bright = dsp._overdrive(_tone(), SR, dict(base, tone=1.0))
    assert _band_db(bright, 18_000.0) > _band_db(dark, 18_000.0) + 3.0


# -------------------------------------------------------- the oracle agrees

@pytest.mark.parametrize("os_n", [1, 2, 4, 8])
def test_reference_monitor_saturation_matches_the_engine(os_n):
    """reference_monitor is the export oracle; it re-implements dsp on purpose, so it
    must track this change. Bit-exact at 1x, and to float tolerance above it (both
    sides use resample_poly - it is the NATIVE monitor that filters differently)."""
    entry = {"drive_db": 9.0, "curve": 2, "mix": 1.0}
    a = dsp._saturate(_tone(), dict(entry, type="saturation", stem_id="a",
                                    oversample=os_n), SR)
    b = reference_monitor._saturate(_tone(), entry["drive_db"], 0.0,
                                    curve=entry["curve"], mix=1.0, oversample=os_n)
    assert np.allclose(a, b, atol=1e-9), f"oracle diverged at {os_n}x"


@pytest.mark.parametrize("os_n", [1, 2, 4, 8])
def test_reference_monitor_overdrive_matches_the_engine(os_n):
    entry = {"drive_db": 15.0, "tone": 0.35, "filter_on": True,
             "freq_hz": 900.0, "q": 0.8, "oversample": os_n}
    a = dsp._overdrive(_tone(), SR, dict(entry, type="overdrive", stem_id="a"))
    b = reference_monitor._overdrive(_tone(), SR, entry["drive_db"], True,
                                     entry["freq_hz"], entry["q"], entry["tone"],
                                     0.0, 1.0, os_n)
    assert np.allclose(a, b, atol=1e-9), f"oracle diverged at {os_n}x"


# --------------------------------------------------------------- the models

def test_models_reject_a_non_power_of_two_factor():
    from mix_engine.models import OverdriveAction, SaturationAction
    for cls in (SaturationAction, OverdriveAction):
        with pytest.raises(ValueError):
            cls(stem_id="a", oversample=3)
        assert cls(stem_id="a", oversample=4).oversample == 4
    assert OverdriveAction(stem_id="a").oversample == dsp.OD_OVERSAMPLE_DEFAULT
    assert SaturationAction(stem_id="a").oversample == dsp.SAT_OVERSAMPLE_DEFAULT


def test_multiband_saturation_oversampling_removes_the_fold_back():
    """Since 2026-10-24 the multiband form runs its whole split and its shapers at the factor
    (it ignored it, and this test asserted so until the day it gained one)."""
    band = {"drive_db": 18.0, "output_db": 0.0, "mix": 1.0}
    entry = {"type": "saturation", "stem_id": "a", "drive_db": 9.0,
             "low": dict(band), "mid": dict(band), "high": dict(band)}
    # a stored 1 (every multiband made before 2026-10-24; the form has no OS control) runs at the floor
    stored1 = dsp._saturate(_tone(), dict(entry, oversample=1), SR)
    assert np.array_equal(stored1, dsp._saturate(_tone(), dict(entry, oversample=dsp.SAT_OVERSAMPLE_DEFAULT), SR))
    # ...and against the old 1x path, built here (the split and the curve at the base rate):
    x = np.ascontiguousarray(_tone().T)
    old = sum(dsp._waveshape(b, 18.0, 0) for b in dsp._lr4_three_way(x, SR, 150.0, 2000.0)).T
    a, b = _band_db(old, ALIAS_HZ), _band_db(stored1, ALIAS_HZ)
    assert b < a - 20.0, f"4x did not suppress the fold-back: 1x {a:.1f} dB -> 4x {b:.1f} dB"
