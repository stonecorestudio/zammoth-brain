"""The clipper's law (2026-10-28): a reference clipper, measured as a black box - `clip_law`.

What is held here:
  * the reference's MEASURED knee (points read off the reference on a slow sine; the red half - the
    tanh blend before it missed them by up to 13 %), its pre-clip gain, its unlinked channels, and
    with oversampling on the 0 dBFS clamp after the resampling;
  * the knob set is the reference's - GAIN, CLIP, SOFTNESS, OVERSAMPLE (off / on) - in the model, the
    session's table and the native clamp table; a request that names the old knobs is refused by
    name; a clipper SAVED with them is read once through `from_legacy`;
  * the agent: its GAIN is charged to the gain wallet, and its ceiling may not sit more than 6 dB
    under its input's full scale;
  * the two offline engines and the monitor resolver read the same knobs (an old entry included).

ONE fixture, `server/tests/data/clip_law_parity.json`, carries Python's answers on a signal both
engines make from one formula; the console leg `--clip-law` asserts MonitorChain renders them. A law
change on purpose: `python server/tests/test_clip_law.py --write`, then the leg.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))

from mix_engine import clip_law as cl, dsp, models, reference_monitor  # noqa: E402
from mix_engine.monitor_chain import monitor_chain_params  # noqa: E402

SR = 48000
FIXTURE = HERE / "data" / "clip_law_parity.json"

# (clip, softness, input, the reference's output) - read off the reference with a 10 Hz sine to +6 dBFS
MEASURED = [
    (1.0, 0.5, 0.7, 0.68), (1.0, 0.5, 0.9, 0.82), (1.0, 0.5, 1.0, 0.875), (1.0, 0.5, 1.2, 0.955),
    (1.0, 1.0, 0.25, 0.2344), (1.0, 1.0, 0.5, 0.4375), (1.0, 1.0, 1.0, 0.75), (1.0, 1.0, 1.5, 0.9375),
    (1.0, 1.0, 2.0, 1.0), (0.5, 0.5, 0.5, 0.4375), (0.5, 1.0, 0.25, 0.2187), (0.5, 1.0, 0.7, 0.455),
    (0.25, 1.0, 0.1, 0.09), (0.25, 1.0, 0.25, 0.1875), (0.1, 0.5, 0.1, 0.0875), (1.0, 0.0, 0.9, 0.9),
    (1.0, 0.0, 1.5, 1.0), (0.5, 0.0, 0.7, 0.5),
]


def _entry(**kw) -> dict:
    e = {"type": "clip", "gain_db": 0.0, "clip_pct": 100.0, "softness": 0.0, "oversample": 1}
    e.update(kw)
    return e


def _ramp(peak: float, n: int = 4801) -> np.ndarray:
    x = np.linspace(-peak, peak, n)
    return np.stack([x, x], axis=1)


# ------------------------------------------------------------------------- the reference's knee
@pytest.mark.parametrize("c,s,x,ref", MEASURED)
def test_the_knee_is_the_references(c, s, x, ref):
    y = dsp._clip(np.array([[x, -x]]), _entry(clip_pct=100.0 * c, softness=s))
    assert y[0, 0] == pytest.approx(ref, abs=6e-4)
    assert y[0, 1] == pytest.approx(-ref, abs=6e-4)


def test_gain_comes_before_the_clip_and_nothing_makes_up():
    x = _ramp(0.1)
    y = dsp._clip(x, _entry(gain_db=6.0, clip_pct=50.0, softness=0.5))
    assert np.allclose(y, x * 10 ** (6 / 20), atol=1e-12)     # all under the knee: just the gain
    y = dsp._clip(_ramp(1.0), _entry(gain_db=12.0, clip_pct=50.0))
    assert np.max(np.abs(y)) == pytest.approx(0.5, abs=1e-12)


def test_the_channels_are_clipped_on_their_own():
    x = np.array([[2.0, 0.3]])
    y = dsp._clip(x, _entry(clip_pct=50.0, softness=0.5))
    assert y[0, 1] == pytest.approx(0.3 - 0.05 ** 2 / 1.0)   # 0.2975, as the reference


def test_clip_zero_is_silence_and_softness_zero_is_a_brickwall():
    x = _ramp(2.0)
    assert np.max(np.abs(dsp._clip(x, _entry(clip_pct=0.0)))) == 0.0
    assert np.allclose(dsp._clip(x, _entry(clip_pct=70.0)), np.clip(x, -0.7, 0.7))


def test_oversampled_never_passes_full_scale_and_aliases_less():
    t = np.arange(SR) / SR
    x = 4.0 * np.sin(2 * np.pi * 7100 * t)
    x = np.stack([x, x], axis=1)
    assert np.max(np.abs(dsp._clip(x, _entry(oversample=2)))) <= 1.0   # the clamp after the resampling
    # aliasing is read under a 50 % ceiling, as on the reference: at 100 % the clamp clips the
    # resampler's ringing again (the reference does the same)
    y1 = dsp._clip(x, _entry(clip_pct=50.0))
    y2 = dsp._clip(x, _entry(clip_pct=50.0, oversample=2))
    def alias_db(y):   # the 5th harmonic folded back to 12.5 kHz, against the fundamental
        Y = np.abs(np.fft.rfft(y[SR // 4:, 0] * np.hanning(len(y) - SR // 4)))
        f = np.fft.rfftfreq(len(y) - SR // 4, 1 / SR)
        return 20 * math.log10(Y[np.argmin(np.abs(f - 12500))] / Y[np.argmin(np.abs(f - 7100))])
    assert alias_db(y2) < alias_db(y1) - 30


# ------------------------------------------------------------------------------ the knob set
def test_the_knob_set_is_the_references():
    fields = set(models.ClipAction.model_fields)
    assert {"gain_db", "clip_pct", "softness", "oversample"} <= fields
    assert not ({"threshold_db", "hardness", "output_db"} & fields)
    from mix_engine.session import Session
    assert set(Session.DEVICE_PARAMS["clip"]) == {"gain_db", "clip_pct", "softness", "oversample"}
    specs = (REPO / "native/Source/DeviceSpecs.h").read_text()
    for row in ('{ "clip",               "clip_pct",          0.0f,     100.0f,   Snap::none }',
                '{ "clip",               "gain_db",           0.0f,     12.0f,    Snap::none }',
                '{ "clip",               "oversample",        1.0f,     2.0f,     Snap::oversample }',
                '{ "clip",               "softness",          0.0f,     1.0f,     Snap::none }'):
        assert row in specs
    for gone in ("threshold_db", "hardness", "output_db"):
        assert f'{{ "clip",               "{gone}"' not in specs


def test_a_request_with_the_old_knobs_is_refused_by_name():
    a = models.ClipAction(stem_id="master")
    assert (a.gain_db, a.clip_pct, a.softness, a.oversample) == (0.0, 100.0, 0.0, 1)
    with pytest.raises(Exception, match="threshold_db"):
        models.ProposalRequest.model_validate(
            {"rationale": "x", "actions": [{"type": "clip", "stem_id": "master", "threshold_db": -3.0}]})
    with pytest.raises(Exception):
        models.ClipAction(stem_id="master", oversample=4)


def test_a_stored_old_clipper_is_read_as_the_law():
    old = {"type": "clip", "stem_id": "master", "threshold_db": -6.0, "hardness": 0.75, "output_db": 3.0,
           "oversample": 4}
    a = models.ClipAction.model_validate(old, context={"stored": True})
    assert a.gain_db == pytest.approx(3.0) and a.softness == pytest.approx(0.25) and a.oversample == 2
    assert a.clip_pct == pytest.approx(100 * 10 ** (-3 / 20), abs=1e-3)
    # a hard old clipper with a positive trim plays the same after the conversion
    hard = {"threshold_db": -6.0, "hardness": 1.0, "output_db": 4.0}
    x = _ramp(1.5)
    old_out = np.clip(x, -10 ** (-6 / 20), 10 ** (-6 / 20)) * 10 ** (4 / 20)
    assert np.allclose(dsp._clip(x, hard), old_out, atol=1e-9)
    # a negative trim has nowhere to go: dropped, said in clip_law's text
    assert cl.from_legacy({"threshold_db": -3.0, "output_db": -2.0})["gain_db"] == 0.0


def test_every_reader_reads_the_same_knobs_for_an_old_entry():
    old = {"type": "clip", "stem_id": "master", "threshold_db": -4.0, "hardness": 0.3}
    [m] = monitor_chain_params([old])["entries"] if isinstance(monitor_chain_params([old]), dict) \
        else monitor_chain_params([old])
    assert {k: m[k] for k in ("gain_db", "clip_pct", "softness", "oversample")} == cl.values(old)
    x = _ramp(1.2)
    assert np.array_equal(dsp._clip(x, old), reference_monitor._clip(x, cl.values(old)))


def test_a_session_saved_before_is_converted_once(demo_project):
    """STATE_VERSION 4: the chains, the undo snapshots and the recorded actions."""
    from mix_engine.session import STATE_VERSION, Session
    path = Path(demo_project) / "session.json"
    Session(demo_project)._persist()
    state = json.loads(path.read_text())
    old = {"type": "clip", "stem_id": "master", "threshold_db": -6.0, "hardness": 0.25, "oversample": 8}
    state["master_chain"] = [dict(old)]
    state["proposals"] = [{"id": "p1", "actions": [dict(old)], "inverse": {"master": {"master_chain": [dict(old)]}}}]
    state["state_version"] = 3
    path.write_text(json.dumps(state))
    s = Session(demo_project)
    assert s._state["state_version"] == STATE_VERSION and STATE_VERSION >= 4   # 5 since the Duck's law
    want = cl.from_legacy(old)
    for e in (s._state["master_chain"][0], s._state["proposals"][0]["actions"][0],
              s._state["proposals"][0]["inverse"]["master"]["master_chain"][0]):
        assert not (set(cl.LEGACY_FIELDS) & set(e))
        assert {k: e[k] for k in want} == want
    assert want == {"gain_db": 0.0, "clip_pct": pytest.approx(50.12, abs=0.01), "softness": 0.75, "oversample": 2}


# ------------------------------------------------------------------------------ the agent
def test_the_agents_gain_is_charged_and_its_crush_is_bounded():
    from mix_engine import guardrails
    assert guardrails.level_charge_db({"type": "clip", "gain_db": 5.0, "clip_pct": 100.0}) == pytest.approx(5.0)
    assert cl.crush_db(_entry(gain_db=6.0, clip_pct=100.0)) == pytest.approx(-6.0)
    assert cl.crush_db(_entry(gain_db=3.0, clip_pct=50.0)) < guardrails.LIMITS["clip_threshold_min_db"]


# ------------------------------------------------------------------------------ the fixture
def fixture_cases() -> list[dict]:
    return [_entry(), _entry(gain_db=6.0, clip_pct=70.0, softness=0.5),
            _entry(gain_db=9.0, clip_pct=100.0, softness=1.0), _entry(clip_pct=30.0, softness=0.2),
            {"type": "clip", "threshold_db": -6.0, "hardness": 0.5, "output_db": 2.0}]


def fixture_signal(n: int = 2048) -> np.ndarray:
    i = np.arange(n)
    left = 1.6 * np.sin(2 * np.pi * i * 0.0137) * (0.3 + 0.7 * (i / n))
    right = 0.9 * np.sin(2 * np.pi * i * 0.0311 + 0.4)
    return np.stack([left, right], axis=1)


def fixture() -> dict:
    x = fixture_signal()
    return {"n": int(x.shape[0]), "signal": "left = 1.6 sin(2 pi i 0.0137)(0.3 + 0.7 i / n); "
                                            "right = 0.9 sin(2 pi i 0.0311 + 0.4)",
            "cases": [{"entry": e, "out": [[round(float(v), 7) for v in row]
                                           for row in dsp._clip(x.astype(np.float32).astype(np.float64), e).T]}
                      for e in fixture_cases()]}


def test_the_parity_fixture_is_this_law():
    assert json.loads(FIXTURE.read_text()) == fixture()


if __name__ == "__main__" and "--write" in sys.argv:
    FIXTURE.write_text(json.dumps(fixture()))
    print("wrote", FIXTURE)
