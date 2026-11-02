"""The EQ's law (2026-10-27): fitted to a reference EQ - `eq_law`.

Held here:
  * the reference's MEASURED shapes, which the cookbook EQ before this did not have (the red half):
    a bell's width at the same Q, a band near the top of the spectrum, the tilt shelf's swing, a
    steep cut's resonance;
  * a band stored under the old law is converted once, at load, and plays as it did;
  * every engine that renders an EQ band runs this law (dsp, the oracle's reference, the rail);
  * the native fixture `data/eq_law_parity.json` is what Python computes (the console leg
    `--eq-law` asserts MonitorChain's coefficients are it). A law change on purpose:
    `python server/tests/test_eq_law.py --write`, then the leg.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.signal import freqz, lfilter

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))

from mix_engine import dsp, eq_law, guardrails, reference_monitor  # noqa: E402
from mix_engine.reference_monitor import biquad_coeffs  # noqa: E402

SR = 48000
FIXTURE = HERE / "data" / "eq_law_parity.json"


def _band(shape, f, g, q, slope=12):
    return {"shape": shape, "freq_hz": float(f), "gain_db": float(g), "q": float(q), "slope_db_oct": slope}


def _render_db(band, freqs, sr=SR):
    """What dsp renders: an impulse through `dsp._eq_band`, read at `freqs`."""
    n = 1 << 15
    x = np.zeros((1, n), np.float32)
    x[0, 64] = 1.0
    h = dsp._eq_band(x, sr, band)[0, 64:].astype(np.float64)
    k = np.arange(len(h))
    return np.array([20 * math.log10(abs(np.sum(h * np.exp(-2j * math.pi * f * k / sr)))) for f in freqs])


def _law_db(shape, f, g, q, slope, freqs, sr=SR):
    H = np.ones(len(freqs), complex)
    for b, a in eq_law.band_sections(shape, sr, f, q, g, slope):
        H *= freqz(b, a, freqs, fs=sr)[1]
    return 20 * np.log10(np.abs(H))


# --------------------------------------------------- the reference's shapes (the red half)
# A +6 dB bell, Q 1, at 1 kHz, read off the reference at 250 / 500 / 707 / 1000 / 1414 / 2000 / 4000 Hz.
MEASURED_BELL = [0.78, 2.82, 4.75, 5.99, 4.75, 2.82, 0.77]


def test_a_bell_is_as_wide_as_the_references():
    got = _render_db(_band("peak", 1000, 6, 1.0), [250, 500, 707, 1000, 1414, 2000, 4000])
    assert np.max(np.abs(got - MEASURED_BELL)) < 0.1, got


def test_a_band_near_the_top_keeps_its_shape():
    """A +12 dB bell at 15 kHz, Q 1: the reference reads 10.0 dB at 20 kHz (the cookbook 2.6)."""
    got = _render_db(_band("peak", 15000, 12, 1.0), [12000, 15000, 18000, 20000])
    want = [10.56, 11.99, 10.98, 10.0]
    assert np.max(np.abs(got - want)) < 1.0, got   # within 1 dB at the top of the band (the cookbook: 7)


def test_the_tilt_shelf_swings_its_gain_each_way():
    got = _render_db(_band("tilt_shelf", 1000, 6, 1.0), [30, 1000, 15000])
    assert abs(got[0] + 6.0) < 0.1 and abs(got[1]) < 0.1 and abs(got[2] - 6.0) < 0.1, got


def test_a_steep_cut_takes_its_q_on_its_last_section():
    """A 24 dB/oct high-pass at 1 kHz, Q 2: the reference reads +3 dB at its corner (Butterworth -3)."""
    got = _render_db(_band("hpf", 1000, 0, 2.0, 24), [1000])
    assert abs(got[0] - 3.0) < 0.15, got
    flat = _render_db(_band("hpf", 1000, 0, 1.0, 24), [1000])
    assert abs(flat[0] + 3.01) < 0.05, flat


# --------------------------------------------------- old projects play as they did
def _cookbook_db(shape, f, g, q, slope, freqs):
    if shape == "tilt_shelf":
        secs = [biquad_coeffs("low_shelf", SR, f, q, -g / 2), biquad_coeffs("high_shelf", SR, f, q, g / 2)]
    elif shape in ("hpf", "lpf") and slope > 6:
        secs = reference_monitor.slope_cascade_coeffs(shape, SR, f, q, slope)
    else:
        secs = [biquad_coeffs(shape, SR, f, q, g)]
    H = np.ones(len(freqs), complex)
    for b, a in secs:
        H *= freqz(b, a, freqs, fs=SR)[1]
    return 20 * np.log10(np.abs(H) + 1e-12)


def test_a_converted_band_sounds_as_it_did():
    freqs = np.geomspace(20, 3000, 120)       # below where the cookbook cramped
    for shape, f, g, q, slope, tol in [("peak", 300, -6, 2.0, 12, 0.1), ("peak", 1000, 9, 0.5, 12, 0.1),
                                       ("notch", 500, 0, 3.0, 12, 0.1), ("bandpass", 200, 0, 1.0, 12, 0.15),
                                       ("low_shelf", 120, 6, 0.707, 12, 0.15), ("high_shelf", 2000, -5, 1.2, 12, 0.15),
                                       ("tilt_shelf", 900, 6, 0.707, 12, 0.15), ("hpf", 100, 0, 1.4, 12, 0.1),
                                       ("lpf", 3000, 0, 0.5, 24, 0.1), ("hpf", 60, 0, 4.0, 48, 0.1),
                                       ("hpf", 80, 0, 0.707, 6, 0.05)]:
        nq, ng = eq_law.legacy_band(shape, q, g, slope)
        old = _cookbook_db(shape, f, g, q, slope, freqs)
        new = _law_db(shape, f, ng, nq, slope, freqs)
        m = old > -40
        assert np.max(np.abs(old - new)[m]) < tol, (shape, q, nq, np.max(np.abs(old - new)[m]))


def test_an_old_session_is_converted_once(tmp_path):
    from mix_engine.session import Session, STATE_VERSION
    import shutil
    src = REPO / "test_projects" / "demo"
    if not src.is_dir():
        src = next(p for p in (REPO / "test_projects").iterdir() if p.is_dir() and any(p.glob("*.wav")))
    proj = tmp_path / "p"
    proj.mkdir()
    for w in list(src.glob("*.wav"))[:1]:
        shutil.copy(w, proj / w.name)
    stem = next(proj.glob("*.wav")).stem
    eq = {"type": "eq", "band": _band("peak", 1000, 6, 2.0), "uid": "u1"}
    state = {"state_version": 2, "stems": {stem: {"path": f"{stem}.wav", "gain_db": 0.0, "chain": [eq]}},
             "master_gain_db": 0.0, "master_chain": [{"type": "eq", "band": _band("tilt_shelf", 1000, 4, 1.0)}],
             "proposals": [{"inverse": {stem: {"path": f"{stem}.wav", "chain": [dict(eq, band=_band("peak", 500, -3, 1.0))]}},
                            "actions": [{"type": "eq", "stem_id": stem, "band": _band("peak", 1000, 6, 2.0)}]}],
             "pass_number": 1, "applied_seq": 0}
    (proj / "session.json").write_text(json.dumps(state))
    assert STATE_VERSION >= 3
    s = Session(proj)
    st = json.loads((proj / "session.json").read_text())
    assert st["state_version"] == STATE_VERSION
    assert abs(st["stems"][stem]["chain"][0]["band"]["q"] - 2.0 * math.sqrt(2)) < 1e-9
    assert abs(st["master_chain"][0]["band"]["gain_db"] - 2.0) < 1e-9
    assert abs(st["proposals"][0]["inverse"][stem]["chain"][0]["band"]["q"] - math.sqrt(2)) < 1e-9
    assert abs(st["proposals"][0]["actions"][0]["band"]["q"] - 2.0 * math.sqrt(2)) < 1e-9
    Session(proj)                                       # a second open converts nothing again
    st2 = json.loads((proj / "session.json").read_text())
    assert st2["stems"][stem]["chain"][0]["band"]["q"] == st["stems"][stem]["chain"][0]["band"]["q"]
    del s


# --------------------------------------------------- every engine runs the law
def test_every_engine_runs_the_law():
    freqs = np.array([100.0, 1000.0, 9000.0, 17000.0])
    for shape, f, g, q, sl in [("peak", 12000, 9, 2.0, 12), ("tilt_shelf", 500, 4, 1.0, 12), ("lpf", 9000, 0, 3.0, 24)]:
        want = _law_db(shape, f, g, q, sl, freqs)
        secs = guardrails._eq_band_sections(shape, f, q, g, sl, SR)
        H = np.ones(len(freqs), complex)
        for b, a in secs:
            H *= freqz(b, a, freqs, fs=SR)[1]
        assert np.max(np.abs(20 * np.log10(np.abs(H)) - want)) < 1e-6, ("rail", shape)
        x = np.zeros((1 << 14, 1))
        x[0, 0] = 1.0
        h = reference_monitor.apply_eq_entry_band(x, SR, _band(shape, f, g, q, sl))[:, 0]
        k = np.arange(len(h))
        got = [20 * math.log10(abs(np.sum(h * np.exp(-2j * math.pi * fr * k / SR)))) for fr in freqs]
        assert np.max(np.abs(np.array(got) - want)) < 1e-3, ("reference", shape, got, want)


def test_the_q_range_is_the_references():
    from mix_engine import models
    from mix_engine.session import Session
    assert Session.DEVICE_PARAMS["eq"]["q"] == ("band.q", eq_law.Q_MIN, eq_law.Q_MAX)
    f = models.EqBand.model_fields["q"]
    assert [m.ge for m in f.metadata if hasattr(m, "ge")] == [eq_law.Q_MIN]
    assert [m.le for m in f.metadata if hasattr(m, "le")] == [eq_law.Q_MAX]
    specs = (REPO / "native/Source/DeviceSpecs.h").read_text()
    assert '{ "eq",                 "q",                 0.1f,     40.0f,' in specs.replace("  ", "  ")


# --------------------------------------------------- the native fixture
RATES = (44100, 48000, 96000)
CASES = [(sh, f, g, q, sl)
         for sh, fs_, gs, qs, sls in [("peak", (40, 1000, 15000, 19500), (-12, 0, 7.5), (0.1, 1.0, 12.0), (12,)),
                                      ("low_shelf", (60, 3000, 18000), (-9, 6), (0.3, 1.0, 8.0), (12,)),
                                      ("high_shelf", (200, 9000, 20000), (-6, 12), (0.5, 2.0), (12,)),
                                      ("tilt_shelf", (800, 16000), (-4, 6), (0.7, 3.0), (12,)),
                                      ("notch", (100, 5000, 18000), (0,), (0.4, 6.0), (12,)),
                                      ("bandpass", (150, 12000), (0,), (0.5, 9.0), (12,)),
                                      ("hpf", (20, 900, 17000), (0,), (0.5, 1.0, 6.0), (6, 12, 24, 48, 96)),
                                      ("lpf", (50, 4000, 21000), (0,), (0.7, 3.0, 40.0), (6, 12, 24, 96))]
         for f in fs_ for g in gs for q in qs for sl in sls]


def answers() -> dict:
    out = []
    for sr in RATES:
        for sh, f, g, q, sl in CASES:
            secs = eq_law.band_sections(sh, sr, f, q, g, sl)
            out.append({"sr": sr, "shape": sh, "freq_hz": f, "gain_db": g, "q": q, "slope_db_oct": sl,
                        "sections": [[float(b[0]), float(b[1]), float(b[2]), float(a[1]), float(a[2])] for b, a in secs]})
    legacy = []
    for sh, sl in [("peak", 12), ("notch", 12), ("bandpass", 12), ("low_shelf", 12), ("high_shelf", 12),
                   ("tilt_shelf", 12), ("hpf", 6), ("hpf", 12), ("lpf", 24), ("lpf", 96)]:
        for q in (0.3, 0.707, 2.0, 8.0):
            nq, ng = eq_law.legacy_band(sh, q, 6.0, sl)
            legacy.append({"shape": sh, "slope_db_oct": sl, "q": q, "gain_db": 6.0, "new_q": nq, "new_gain_db": ng})
    return {"cases": out, "legacy": legacy}


def test_the_fixture_is_what_python_computes():
    have, want = json.loads(FIXTURE.read_text()), answers()
    assert len(have["cases"]) == len(want["cases"])
    for h, w in zip(have["cases"], want["cases"]):
        assert {k: h[k] for k in ("sr", "shape", "freq_hz", "gain_db", "q", "slope_db_oct")} == \
               {k: w[k] for k in ("sr", "shape", "freq_hz", "gain_db", "q", "slope_db_oct")}
        assert np.max(np.abs(np.array(h["sections"]) - np.array(w["sections"]))) < 1e-12, h
    assert have["legacy"] == want["legacy"]


if __name__ == "__main__":
    if "--write" in sys.argv:
        FIXTURE.write_text(json.dumps(answers()))
        print(f"wrote {FIXTURE} ({len(answers()['cases'])} bands)")
