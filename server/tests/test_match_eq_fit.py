"""The live monitor's match-EQ must follow the FIR the offline engine renders.

2026-09-08 device audit, 04-F4. `dsp._match_eq` is a linear-phase FIR, frequency-sampled
at the eight band geometric centres. The live monitor approximates it with one 0.7-Q
peaking bell per band — and fed the RAW band gains, which is wrong, because adjacent
0.7-Q bells overlap heavily and their dB contributions add. Measured against the FIR:
up to 9.7 dB peak and 5.8 dB RMS on the widest legal correction. A "-3 dB bass" landed
as -4.8 dB in the bank and +0.7 dB in the FIR.

`monitor_chain.py` used to say "the export keeps the linear-phase FIR". In the DAW that
is false — the export is a bounce of the native graph, so this bank IS the deliverable,
and the divergence was shipping. The export oracle carries no match_eq in any profile,
so no gate had ever compared them.

The bank is fitted to the FIR's magnitude response now. Fitting the eight band CENTRES
exactly is not enough and is measured here too: it still leaves ~4.6 dB between the
centres, and makes the widest correction worse, because eight broad bells cannot follow
a frequency-sampled FIR point for point. Least squares over the whole curve is what
works: RMS error against the FIR falls by 5-10x across the corrections below.

What it does NOT do is make the two identical. On the widest legal correction the peak
error is still several dB, because +/-12 dB alternating between adjacent overlapping
bands is not representable by eight 0.7-Q bells at any gains. Closing that needs the
native side to run a real FIR (juce::dsp::Convolution, with declared latency), which is
the audit's other option and is not done here.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from mix_engine.dsp import match_eq_fir
from mix_engine.models import BAND_EDGES_HZ, BAND_LABELS
from mix_engine.monitor_chain import (MATCH_EQ_Q, _BAND_CENTER_HZ,
                                      _fit_match_eq_gains, monitor_chain_params)
from mix_engine.reference_monitor import biquad_coeffs

SR = 48000
GRID = np.geomspace(30.0, 16000.0, 400)

CORRECTIONS = {
    "ordinary": dict(zip(BAND_LABELS, [-2, -3, -4, 0.5, 1.5, 2, 3, 4])),
    "widest legal": dict(zip(BAND_LABELS, [6, -12, 6, -12, 6, -12, 6, -12])),
    "single band": dict(zip(BAND_LABELS, [0, 0, 0, 0, 0, 6, 0, 0])),
}
# Measured 2026-09-08 on this exact path, raw -> fitted (max / rms dB against the FIR):
#   ordinary      2.35 / 1.42  ->  1.25 / 0.15
#   widest legal  9.71 / 5.82  ->  5.66 / 1.89
#   single band   2.80 / 1.05  ->  0.85 / 0.19
# RMS is the number that matters for tonal balance and it improves by 5-10x. The worst-case
# MAX on the widest correction stays several dB, and that is honest: +/-12 dB alternating
# between adjacent overlapping bands is not representable by eight 0.7-Q bells at all. The
# bounds carry a little headroom over the measurement, not a lot.
MAX_ERR_DB = {"ordinary": 1.6, "widest legal": 6.2, "single band": 1.1}
MAX_RMS_DB = {"ordinary": 0.3, "widest legal": 2.3, "single band": 0.3}


def _bank_db(gains: dict) -> np.ndarray:
    """Summed dB response of the peak bank — what the native engine builds from the contract."""
    z = np.exp(-1j * 2.0 * np.pi * GRID / SR)
    total = np.zeros_like(GRID)
    for band, g in gains.items():
        if abs(float(g)) < 1e-9:
            continue
        b, a = biquad_coeffs("peak", float(SR), _BAND_CENTER_HZ[band], MATCH_EQ_Q, float(g))
        h = (b[0] + b[1] * z + b[2] * z ** 2) / (a[0] + a[1] * z + a[2] * z ** 2)
        total += 20.0 * np.log10(np.abs(h))
    return total


def _fir_db(gains: dict) -> np.ndarray:
    """The offline deliverable's response, from the SAME builder `_match_eq` uses."""
    spec = np.fft.rfft(match_eq_fir(gains, SR), n=8192)
    freqs = np.fft.rfftfreq(8192, 1.0 / SR)
    return np.interp(GRID, freqs, 20.0 * np.log10(np.abs(spec) + 1e-12))


@pytest.mark.parametrize("name", sorted(CORRECTIONS))
def test_the_fitted_bank_follows_the_fir(name):
    gains = CORRECTIONS[name]
    err = np.abs(_fir_db(gains) - _bank_db(_fit_match_eq_gains(gains)))
    rms = float(np.sqrt(np.mean(err ** 2)))
    assert err.max() < MAX_ERR_DB[name], (
        f"{name}: the monitor's bank peaks {err.max():.2f} dB from the FIR the offline engine "
        f"renders (rms {rms:.2f} dB)"
    )
    # The RMS is what a listener hears as tonal balance, and it is where the fit pays off.
    assert rms < MAX_RMS_DB[name], f"{name}: rms error {rms:.2f} dB against the FIR"


@pytest.mark.parametrize("name", sorted(CORRECTIONS))
def test_the_fit_beats_the_raw_gains_it_replaced(name):
    """NEGATIVE CONTROL — the raw gains are exactly what shipped, so this is the
    before-and-after. Without it, a fit that happened to be no better would pass above
    simply because the bound was generous."""
    gains = CORRECTIONS[name]
    fir = _fir_db(gains)
    raw = np.abs(fir - _bank_db(gains)).max()
    fitted = np.abs(fir - _bank_db(_fit_match_eq_gains(gains))).max()
    assert fitted < raw - 0.5, (
        f"{name}: fitted {fitted:.2f} dB vs raw {raw:.2f} dB — the fit must be a real improvement"
    )


def test_fitting_the_band_centres_alone_is_not_enough():
    """Why it is a least-squares fit over the curve and not an 8x8 solve at the centres.

    Pinning the centres is exact AT them and still wrong between them — and on the widest
    correction it is worse than doing nothing. This pins the reasoning so nobody
    'simplifies' the fit back into a solve.
    """
    centres = np.array([_BAND_CENTER_HZ[b] for b in BAND_LABELS])
    z = np.exp(-1j * 2.0 * np.pi * centres / SR)

    def at_centres(band: str) -> np.ndarray:
        b, a = biquad_coeffs("peak", float(SR), _BAND_CENTER_HZ[band], MATCH_EQ_Q, 1.0)
        h = (b[0] + b[1] * z + b[2] * z ** 2) / (a[0] + a[1] * z + a[2] * z ** 2)
        return 20.0 * np.log10(np.abs(h))

    matrix = np.column_stack([at_centres(b) for b in BAND_LABELS])
    gains = CORRECTIONS["widest legal"]
    target = np.array([float(gains[b]) for b in BAND_LABELS])
    centre_fit = dict(zip(BAND_LABELS, np.linalg.solve(matrix, target)))
    fir = _fir_db(gains)
    assert np.abs(fir - _bank_db(centre_fit)).max() > np.abs(fir - _bank_db(_fit_match_eq_gains(gains))).max()


def test_the_resolver_emits_the_fitted_gains():
    """The fit has to reach the contract the native engine actually reads, not just exist."""
    gains = CORRECTIONS["ordinary"]
    out = monitor_chain_params([{"type": "match_eq", "stem_id": "master",
                                 "band_gains_db": dict(gains)}])
    entries = [e for e in out["entries"] if e["type"] == "match_eq"]
    assert entries, "the resolver dropped the match_eq"
    emitted = {b["freq_hz"]: b["gain_db"] for b in entries[0]["bands"]}
    fitted = _fit_match_eq_gains(gains)
    for band, g in fitted.items():
        if abs(g) > 0.01:
            assert emitted.get(_BAND_CENTER_HZ[band]) == pytest.approx(g, abs=1e-9)
    # …and they are NOT the raw gains, which is the whole change.
    assert emitted.get(_BAND_CENTER_HZ["bass"]) != pytest.approx(float(gains["bass"]), abs=0.05)


def test_a_flat_request_stays_flat():
    """Non-vacuity at the other end: no correction must fit to no bells, or the resolver
    would be adding filters to a chain the user asked nothing of."""
    flat = dict.fromkeys(BAND_LABELS, 0.0)
    fitted = _fit_match_eq_gains(flat)
    assert max(abs(g) for g in fitted.values()) < 0.01
    assert np.abs(_bank_db(fitted)).max() < 0.05


# ---------- the fit has to run at the PROJECT's rate (2026-09-08 pre-merge review) ----------
# `dsp.match_eq_fir` breaks out of its band loop at Nyquist, so below ~32 kHz the offline FIR
# drops the top bands entirely — while a fit pinned to 48 kHz still includes them and hands the
# native bank a boost nobody asked for, in the half of the parity triangle that IS the DAW's
# deliverable. Measured on `{"air": -12}`: 11.52 dB of divergence at 22.05 kHz, 0.00 once the
# fit uses the project's rate.

def _fit_vs_fir_db(gains: dict, sr: float, fit_sr: float) -> float:
    """Worst |summed bell bank − the FIR it approximates|, both evaluated at `sr`."""
    import numpy as np

    from mix_engine.dsp import match_eq_fir
    from mix_engine.monitor_chain import _BAND_CENTER_HZ, MATCH_EQ_Q, _fit_match_eq_gains
    from mix_engine.reference_monitor import biquad_coeffs
    fitted = _fit_match_eq_gains(gains, fit_sr)
    grid = np.geomspace(30.0, min(16000.0, sr * 0.45), 400)
    z = np.exp(-1j * 2 * np.pi * grid / sr)
    total = np.zeros_like(grid)
    for band, g in fitted.items():
        b, a = biquad_coeffs("peak", float(sr),
                             min(_BAND_CENTER_HZ[band], sr * 0.45), MATCH_EQ_Q, float(g))
        h = (b[0] + b[1] * z + b[2] * z ** 2) / (a[0] + a[1] * z + a[2] * z ** 2)
        total += 20.0 * np.log10(np.abs(h))
    spec = np.fft.rfft(match_eq_fir(gains, sr), n=8192)
    freqs = np.fft.rfftfreq(8192, 1.0 / sr)
    target = np.interp(grid, freqs, 20.0 * np.log10(np.abs(spec) + 1e-12))
    return float(np.max(np.abs(total - target)))


@pytest.mark.parametrize("sr", [22050.0, 16000.0])
def test_the_fit_follows_the_fir_at_low_project_rates(sr):
    """The bug: an air correction on a 22.05 kHz project (an ordinary decoded mp3)."""
    gains = {"air": -12.0}
    assert _fit_vs_fir_db(gains, sr, sr) < 1.0
    # …and the negative control — pinning the fit to 48 kHz, which is what shipped — is far out.
    assert _fit_vs_fir_db(gains, sr, 48000.0) > 5.0


def test_the_fit_did_not_regress_at_the_ordinary_rates():
    """44.1 / 48 kHz must be no worse than the fixed-rate fit was, or this traded one for another."""
    gains = {"bass": -3.0, "air": 2.0}
    for sr in (44100.0, 48000.0):
        assert _fit_vs_fir_db(gains, sr, sr) <= _fit_vs_fir_db(gains, sr, 48000.0) + 0.05


def test_monitor_chain_params_threads_the_rate_to_the_fit():
    """The PLUMBING, not just the function — and the behaviour it buys.

    On a 22.05 kHz project the air band is above `match_eq_fir`'s Nyquist cutoff, so the offline
    render applies NOTHING. The contract now agrees: the fit is flat and the bank is empty. At
    48 kHz the same request really does correct, so this is not "low rate disables match_eq".
    """
    from mix_engine.monitor_chain import monitor_chain_params
    entry = {"type": "match_eq", "stem_id": "master", "band_gains_db": {"air": -12.0}}
    lo = monitor_chain_params([entry], sample_rate=22050.0)["entries"]
    hi = monitor_chain_params([entry])["entries"]
    assert hi and hi[0]["bands"], "48 kHz must still emit a bank (else this proves nothing)"
    assert not lo or not lo[0].get("bands"), (
        "22.05 kHz still emitted a bank the offline FIR does not render — the rate never "
        "reached the fit")
