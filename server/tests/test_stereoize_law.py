"""The Imager - the one stereo device (2026-10-28, James: "keep 1 stereo device instead of our 3 ...
just blackbox imager 2").

Width is a pure mid/side scale (it nulled against the reference imager to 0.000 dB, nothing here moves
it); STEREOIZE is new: above unity width, side made from the mid through the velvet taps
`stereoize_law` builds. The 3-band imager and band width are gone: every one stored before is a
Utility now (`convert_retired_stereo`, session STATE_VERSION 8), and a proposal of one is refused.
The native half is `--stereoize-law` (daw/Source/StereoizeLawLegs.h).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mix_engine import dsp, reference_monitor as rm, stereoize_law as L
from mix_engine.models import ProposalRequest
from mix_engine.monitor_chain import monitor_chain_params
from mix_engine.session import STATE_VERSION, Session, convert_retired_stereo, effective_chain

SR = 48000


def _stereo(n=SR, seed=3):
    rng = np.random.default_rng(seed)
    mid = rng.standard_normal(n) * 0.1
    side = rng.standard_normal(n) * 0.03
    return np.stack([mid + side, mid - side], axis=1)


def test_the_header_is_the_law():
    assert L.HEADER.read_text() == L.header_text()


def test_the_taps_follow_the_reference_counts_and_never_collide():
    for ms in range(1, 21):
        for sr in (44100, 48000, 96000):
            taps = L.taps(ms, sr)
            assert len(taps) == sum(L.counts(ms)), (ms, sr)          # no two on one sample
            assert taps[-1][0] < ms * 1e-3 * sr + 1                   # inside the time
        energy = sum(g * g for _, g in L.taps(ms, 48000))
        want = sum(c * (L.UNIT * a) ** 2 for c, a in zip(L.counts(ms), L.AMPS))
        assert energy == pytest.approx(want, rel=1e-12)


def test_stereoize_never_touches_the_mid_and_acts_only_above_unity():
    x = _stereo()
    mid = 0.5 * (x[:, 0] + x[:, 1])
    for w in (0.0, 0.5, 1.0, 1.4, 2.0):
        y = L.apply(x, SR, w, True, 12)
        np.testing.assert_allclose(0.5 * (y[:, 0] + y[:, 1]), mid, atol=1e-15)
    np.testing.assert_array_equal(L.apply(x, SR, 1.0, True, 12), L.apply(x, SR, 1.0, False, 12))
    np.testing.assert_array_equal(L.apply(x, SR, 0.6, True, 12), L.apply(x, SR, 0.6, False, 12))
    more = L.apply(x, SR, 1.5, True, 12)
    plain = L.apply(x, SR, 1.5, False, 12)
    assert np.std(more[:, 0] - more[:, 1]) > np.std(plain[:, 0] - plain[:, 1])


def test_a_mono_part_is_widened_by_stereoize_alone():
    mono = np.repeat(_stereo()[:, :1], 2, axis=1)
    assert np.max(np.abs(L.apply(mono, SR, 2.0, False)[:, 0] - L.apply(mono, SR, 2.0, False)[:, 1])) == 0
    y = L.apply(mono, SR, 2.0, True, 20)
    side = 0.5 * (y[:, 0] - y[:, 1])
    assert 20 * np.log10(side.std() / mono[:, 0].std()) > -6.0


def test_the_three_engines_play_one_law():
    x = _stereo()
    for w, on, ms in ((1.6, True, 6.0), (2.0, True, 20.0), (1.3, False, 6.0), (0.4, True, 9.0)):
        a = dsp._width(x, w, on, ms, SR)
        b = rm._width(x, w, on, ms, SR)
        np.testing.assert_allclose(a, b, atol=1e-12)
        np.testing.assert_allclose(a, L.apply(x, SR, w, on, ms) if w != 1.0 else x, atol=1e-15)


def test_a_strip_carries_stereoize_to_the_render_and_the_native(demo_project):
    s = Session(demo_project)
    stem = next(iter(s._state["stems"]))
    s.user_edit(stem, {"width": 1.5, "stereoize": True, "stereoize_ms": 11.6})
    st = s._state["stems"][stem]
    assert st["stereoize"] is True and st["stereoize_ms"] == 12.0      # whole ms
    width = [e for e in effective_chain(st) if e["type"] == "width"]
    assert width == [{"type": "width", "width": 1.5, "stereoize": True, "stereoize_ms": 12.0}]
    native = [e for e in monitor_chain_params(effective_chain(st))["entries"] if e["type"] == "width"]
    assert native[0]["stereoize"] is True and native[0]["stereoize_ms"] == 12.0


@pytest.mark.parametrize("kind", ["imager", "band_width"])
def test_a_retired_device_cannot_be_proposed_or_added(kind, demo_project):
    with pytest.raises(Exception):
        ProposalRequest(rationale="x", predicted_impact=[],
                        actions=[{"type": kind, "stem_id": "master", "width": 1.2}])
    s = Session(demo_project)
    with pytest.raises(Exception):
        s.add_master_device(kind)


def test_old_imagers_and_band_widths_become_utilities():
    state = {"master_chain": [
                {"type": "imager", "uid": "a1", "width_low": 0.0, "width_mid": 0.8, "width_high": 1.6,
                 "crossover_lo_hz": 180.0, "crossover_hi_hz": 2500.0, "enabled": False}],
             "stems": {"pad": {"chain": [
                 {"type": "eq", "uid": "e1"},
                 {"type": "band_width", "uid": "b1", "band": "bass", "width": 0.0},
                 {"type": "band_width", "uid": "b2", "band": "presence", "width": 1.6},
                 {"type": "imager", "uid": "a2", "width_low": 0.9}]}},
             "proposals": [{"actions": [{"type": "imager", "stem_id": "pad", "width_mid": 1.2}]}]}
    assert convert_retired_stereo(state) == (3, 1, 1)
    assert state["master_chain"] == [{"type": "utility", "uid": "a1", "enabled": False, "width": 0.8,
                                      "bass_mono": True, "bass_mono_hz": 180.0}]
    assert state["stems"]["pad"]["chain"] == [
        {"type": "eq", "uid": "e1"},
        {"type": "utility", "uid": "b1", "bass_mono": True, "bass_mono_hz": 150.0},
        {"type": "utility", "uid": "a2", "width": 1.0}]
    assert state["proposals"][0]["actions"] == [{"type": "utility", "stem_id": "pad", "width": 1.2}]


def test_a_saved_project_converts_once_and_says_so(demo_project):
    s = Session(demo_project)
    stem = next(iter(s._state["stems"]))
    s._state["stems"][stem]["chain"].append(
        {"type": "imager", "stem_id": stem, "width_mid": 1.4, "width_low": 0.2})
    s._state["coedit_log"] = [{"url": "/api/device_param?device_type=imager&occurrence=0&param=width_mid&value=1.2"},
                              {"url": "/api/device_band?occurrence=0&band=bass"},
                              {"url": "/api/device_param?device_type=comp&occurrence=0&param=ratio&value=4"}]
    s._state["state_version"] = 7
    s._persist()
    again = Session(demo_project)
    assert again._state["state_version"] == STATE_VERSION >= 8
    chain = again._state["stems"][stem]["chain"]
    assert [e["type"] for e in chain][-1] == "utility" and chain[-1]["width"] == 1.4
    assert [e["url"] for e in again._state["coedit_log"]] == [
        "/api/device_param?device_type=comp&occurrence=0&param=ratio&value=4"]
    saved = json.loads((Path(demo_project) / "session.json").read_text())
    assert "imager" not in json.dumps(saved)


# ---------------------------------------------------------------- the native half's fixture
# `--stereoize-law` (daw/Source/StereoizeLawLegs.h) reads this file: the law's taps at three rates
# and every time, and the samples one chain renders (every 4th held), so the native chain is held to
# the Python law. Regenerate with `python tests/test_stereoize_law.py --write`.
FIXTURE = Path(__file__).resolve().parent / "data" / "stereoize_law_parity.json"
FIX_N, FIX_SR, FIX_WIDTH, FIX_MS = 9600, 48000, 1.6, 10


def fixture_signal(n: int = FIX_N) -> np.ndarray:
    """A stereo test signal both languages build the same way, rounded to float32 as the chain reads it."""
    i = np.arange(n, dtype=np.float64)
    tau = 2.0 * np.pi
    left = 0.5 * np.sin(tau * i * 0.0031) + 0.2 * np.sin(tau * i * 0.0473)
    right = 0.45 * np.sin(tau * i * 0.0031 + 0.3) + 0.1 * np.sin(tau * i * 0.0119)
    return np.stack([left, right], axis=1).astype(np.float32).astype(np.float64)


def fixture() -> dict:
    y = L.apply(fixture_signal(), FIX_SR, FIX_WIDTH, True, FIX_MS)
    taps = {f"{sr}:{ms}": [[p, g] for p, g in L.taps(ms, sr)]
            for sr in (44100, 48000, 96000) for ms in range(1, 21)}
    return {"sr": FIX_SR, "n": FIX_N, "width": FIX_WIDTH, "ms": FIX_MS, "taps": taps,
            "left": [float(v) for v in y[::4, 0]], "right": [float(v) for v in y[::4, 1]]}


def test_the_native_fixture_is_the_law():
    assert json.loads(FIXTURE.read_text()) == json.loads(json.dumps(fixture()))


if __name__ == "__main__":
    import sys
    if "--write" in sys.argv:
        FIXTURE.write_text(json.dumps(fixture()))
        print("wrote", FIXTURE)
