"""The chorus's law (2026-10-28): a reference two-line chorus, measured as a black box (chorus_law.py
says how), with its knob set - MODE I / II / I+II, STEREO, MIX, VOLUME.

ONE fixture, `server/tests/data/chorus_law_parity.json`, carries Python's answers on a signal both
engines can make; `MixAgentDAW --chorus` renders it through a bare MonitorChain and must land on the
same samples. Rewrite it only after a change on purpose: `python server/tests/test_chorus_law.py
--write`, then the leg.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mix_engine import chorus_law as cl  # noqa: E402
from mix_engine import dsp  # noqa: E402

REPO = HERE.parents[1]
FIXTURE = HERE / "data" / "chorus_law_parity.json"
SR = 48000
N = 3000
OFFSET = 37 * SR + 1234          # the fixture starts mid-song: the sweep is somewhere, not at its start


def fixture_signal(n: int = N) -> np.ndarray:
    i = np.arange(n, dtype=np.float64)
    left = 0.6 * np.sin(2 * np.pi * i * 0.0137) * (0.3 + 0.7 * i / n)
    right = 0.4 * np.sin(2 * np.pi * i * 0.0311 + 0.4)
    return np.stack([left, right], axis=1).astype(np.float32).astype(np.float64)


CASES = [
    {"type": "chorus", "mode": 0, "stereo": 1.0, "mix": 0.5, "output_db": 0.0},
    {"type": "chorus", "mode": 1, "stereo": 0.5, "mix": 1.0, "output_db": -3.0},
    {"type": "chorus", "mode": 2, "stereo": 1.0, "mix": 0.3, "output_db": 0.0},
    {"type": "chorus", "mode": 0, "stereo": 0.0, "mix": 0.8, "output_db": 2.0},
    # saved before the law: read through from_legacy at both ends
    {"type": "chorus", "rate_hz": 0.8, "depth": 0.25, "delay_ms": 20.0, "mix": 0.35, "voices": 1,
     "stereo_deg": 90.0},
]


def fixture() -> dict:
    x = fixture_signal()
    cases = []
    for e in CASES:
        y = cl.apply(x, SR, e, timeline_offset=OFFSET)
        cases.append({"entry": e, "out": [[round(float(v), 7) for v in y[:, ch]] for ch in (0, 1)]})
    return {"n": N, "offset": OFFSET,
            "signal": "left = 0.6 sin(2 pi i 0.0137)(0.3 + 0.7 i / n); right = 0.4 sin(2 pi i 0.0311 + 0.4)",
            "cases": cases}


def test_the_fixture_is_this_law():
    want = json.loads(FIXTURE.read_text())
    got = fixture()
    assert want["n"] == got["n"] and want["offset"] == got["offset"]
    for w, g in zip(want["cases"], got["cases"]):
        assert w["entry"] == g["entry"]
        assert np.max(np.abs(np.array(w["out"]) - np.array(g["out"]))) < 2e-7


def test_the_header_is_written_from_this_file():
    assert (REPO / "native" / "Source" / "ChorusLaw.h").read_text() == cl.header_text()


def test_the_render_plays_the_law():
    x = fixture_signal()
    for e in CASES:
        assert np.array_equal(dsp._chorus(x, SR, e), cl.apply(x, SR, e))


def test_the_mix_curve_is_the_old_blend_made_louder():
    for w in np.linspace(0.0, 0.5, 11):
        d, wt = cl.dry_wet(w)
        assert math.isclose(d, (1 - w) * (1 + 2 * w)) and math.isclose(wt, w * (1 + 2 * w))
        d2, w2 = cl.dry_wet(1 - w)
        assert math.isclose(d2, wt) and math.isclose(w2, d)
    assert cl.dry_wet(0.5) == (1.0, 1.0) and cl.dry_wet(0.0) == (1.0, 0.0) and cl.dry_wet(1.0) == (0.0, 1.0)


def test_each_line_is_a_triangle_between_its_measured_ends():
    for line, (rate, swing, centre, _ph) in enumerate(cl.LINES):
        n = int(SR / rate) + 10
        d = cl.delay_samples(line, False, n, SR) / SR * 1000.0
        assert math.isclose(d.max(), centre + swing, abs_tol=1e-3)
        assert math.isclose(d.min(), centre - swing, abs_tol=1e-3)
        steps = np.abs(np.diff(d))              # a triangle: one slope up, one down (its corners aside)
        assert np.mean(np.isclose(steps, 4 * swing * rate / SR, rtol=1e-3)) > 0.999
        # the right's sweep is the left's half a cycle on: mirrored about the centre
        left, right = cl.delay_samples(line, False, n, SR), cl.delay_samples(line, True, n, SR)
        assert np.allclose(left + right, 2 * centre / 1000 * SR, atol=1e-6)


def test_the_wet_is_the_measured_one():
    # the reference's averaged wet response at 48 kHz (Chorus I, wet only, 20 s of white noise, Welch
    # 4096 - the harness's measure); the whole path here (the reader's own top-end loss included)
    # holds it within 1 dB from 100 Hz to 20 kHz
    from scipy.signal import welch
    ref = {100: 2.05, 500: 2.59, 1000: 2.36, 2000: 1.60, 4000: -0.23, 6000: -3.03, 8000: -4.39,
           12000: -6.49, 16000: -8.22, 20000: -8.85}
    x = np.random.default_rng(1).standard_normal((SR * 20, 1)) * 0.1
    y = cl.apply(np.repeat(x, 2, axis=1), SR, {"mode": 0, "mix": 1.0, "stereo": 1.0})[:, 0]
    f, px = welch(x[:, 0], SR, nperseg=4096)
    _, py = welch(y, SR, nperseg=4096)
    g = 10 * np.log10(py / px)
    got = {fr: float(g[np.argmin(np.abs(f - fr))]) for fr in ref}
    assert all(abs(got[fr] - ref[fr]) < 1.0 for fr in ref), got


def test_stereo_zero_is_one_wet_and_one_is_two():
    t = np.arange(SR // 2) / SR
    mono = np.stack([0.3 * np.sin(2 * np.pi * 440 * t)] * 2, axis=1)
    narrow = cl.apply(mono, SR, {"mode": 0, "stereo": 0.0, "mix": 1.0})
    wide = cl.apply(mono, SR, {"mode": 0, "stereo": 1.0, "mix": 1.0})
    assert np.array_equal(narrow[:, 0], narrow[:, 1])
    assert np.max(np.abs(wide[:, 0] - wide[:, 1])) > 0.05


def test_mix_zero_is_the_dry_and_volume_scales_it():
    x = fixture_signal()
    assert np.array_equal(cl.apply(x, SR, {"mode": 2, "mix": 0.0}), x)
    a = cl.apply(x, SR, {"mode": 1, "mix": 0.4, "output_db": -6.0})
    b = cl.apply(x, SR, {"mode": 1, "mix": 0.4})
    assert np.allclose(a, b * 10 ** (-6 / 20))


def test_i_plus_ii_is_both_lines_with_more_low_cut():
    x = fixture_signal()
    both = cl.apply(x, SR, {"mode": 2, "mix": 1.0, "stereo": 1.0})
    assert cl.wet_coeffs(2, SR)[0][1][1] != cl.wet_coeffs(0, SR)[0][1][1]   # 60 Hz, not 38
    one = cl.apply(x, SR, {"mode": 0, "mix": 1.0, "stereo": 1.0})
    assert np.max(np.abs(both - one)) > 0.01


def test_a_mode_is_taken_by_name_and_rounds():
    assert [cl.mode_index(v) for v in ("I", "ii", "I+II", "i + ii", 1.6, -3, "x", float("nan"))] == [0, 1, 2, 2, 2, 0, 0, 0]


def test_an_old_chorus_is_read_once_as_the_laws_knobs():
    old = {"type": "chorus", "rate_hz": 0.4, "depth": 0.5, "delay_ms": 22.0, "mix": 0.5, "voices": 1,
           "stereo_deg": 90.0}
    k = cl.from_legacy(old)
    assert k["mode"] == 0 and k["stereo"] == 0.5 and k["mix"] == 0.5
    # VOLUME takes back what the curve adds over the old blend: old 0.5/0.5 = -3 dB, new +4.5
    assert math.isclose(k["output_db"], round(-3.0103 - cl.level_db(0.5, 0), 1))
    assert cl.from_legacy({"rate_hz": 2.0, "voices": 1})["mode"] == 1
    assert cl.from_legacy({"rate_hz": 0.3, "voices": 3})["mode"] == 2
    state = {"stems": {"a": {"chain": [dict(old)]}}, "proposals": [{"actions": [dict(old)]}]}
    assert cl.convert_legacy(state) == 2
    for e in (state["stems"]["a"]["chain"][0], state["proposals"][0]["actions"][0]):
        assert set(e) == {"type", "mode", "stereo", "mix", "output_db"}
    assert cl.convert_legacy(state) == 0


def test_the_session_converts_old_choruses_and_says_so(tmp_path):
    from mix_engine.session import Session, take_migration_notices
    import soundfile as sf
    sf.write(tmp_path / "vox.wav", np.zeros((4800, 2), np.float32), SR)
    old = {"type": "chorus", "rate_hz": 2.0, "depth": 0.3, "delay_ms": 12.0, "mix": 0.4, "voices": 1,
           "stereo_deg": 0.0, "uid": "c1"}
    state = {"state_version": 8, "stems": {"vox": {"path": "vox.wav", "gain_db": 0.0, "pan": 0.0,
                                                   "chain": [old]}},
             "master_chain": [], "buses": {}, "returns": {}, "proposals": [], "applied_seq": 0,
             "coedit_log": [{"url": "/api/device_param?stem=vox&device_type=chorus&occurrence=0&param=stereo_deg&value=90"},
                            {"url": "/api/device_param?stem=vox&device_type=chorus&occurrence=0&param=rate_hz&value=1.0"}]}
    (tmp_path / "session.json").write_text(json.dumps(state))
    s = Session(tmp_path)
    e = s._state["stems"]["vox"]["chain"][0]
    assert e["mode"] == 1 and e["stereo"] == 0.0 and e["mix"] == 0.4 and "rate_hz" not in e
    urls = [c["url"] for c in s._state["coedit_log"]]
    assert len(urls) == 1 and "param=stereo&" in urls[0] and "value=0.5" in urls[0]
    assert any("chorus" in n for n in take_migration_notices(tmp_path))


def test_a_request_with_the_old_knobs_is_refused_by_name():
    from mix_engine.models import ProposalRequest
    with pytest.raises(Exception) as err:
        ProposalRequest.model_validate({"rationale": "x", "actions": [
            {"type": "chorus", "stem_id": "vox", "rate_hz": 1.0, "mix": 0.3}]})
    assert "rate_hz" in str(err.value)


def test_the_agent_pays_for_the_level_it_adds():
    from mix_engine.guardrails import level_charge_db
    from mix_engine.models import ChorusAction
    assert 2.0 < level_charge_db(ChorusAction(stem_id="v", mix=0.3)) < 2.5
    assert level_charge_db(ChorusAction(stem_id="v", mix=0.3, output_db=-3.0)) == 0.0
    assert level_charge_db(ChorusAction(stem_id="v", mode="I+II", mix=0.5)) > \
        level_charge_db(ChorusAction(stem_id="v", mode="I", mix=0.5))


def test_the_native_parse_and_the_names_are_the_laws():
    src = (REPO / "native" / "Source" / "DeviceSpecs.h").read_text()
    assert 'static const juce::StringArray chorusModes  { "i", "ii", "i+ii" };' in src
    assert [m.lower() for m in cl.MODES] == ["i", "ii", "i+ii"]
    mc = (REPO / "native" / "Source" / "MonitorChain.cpp").read_text()
    assert "choruslaw::fromLegacy" in mc and '"rate_hz", 0.8' not in mc.split("void MonitorChain::applyChorus")[0].split('t == "chorus"')[-1][:400]


if __name__ == "__main__" and "--write" in sys.argv:
    FIXTURE.write_text(json.dumps(fixture()))
    print(f"wrote {FIXTURE}")
