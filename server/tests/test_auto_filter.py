"""Auto Filter — the modulation family's third device (Phase 63).

A resonant state-variable filter whose cutoff an LFO, an envelope and a trigger track move. The
claims worth testing:

  * ONE law: the offline engine (`dsp._auto_filter`, block-parallel) and the oracle
    (`reference_monitor._auto_filter`, sample by sample) are the same filter - the blocking
    changes no sample, for every type, slope, LFO shape, sync mode and envelope source;
  * the filter is the DIGITAL response it claims (the TPT SVF is the bilinear transform with the
    cutoff prewarped), at 12 and 24 dB/oct;
  * the LFO rides the HOST TIMELINE, and a synced one rides the song's QUARTER NOTES read the way
    Tracktion's tempo sequence reads them;
  * the envelope opens the filter with the level - of the input, or of a trigger track; a track
    has ONE trigger, shared with its Duck;
  * the agent's caps hold (the resonance's static peak inside the 6 dB pass rail), the user's
    knobs go further, and the device lands, renders and is emitted to the native monitor.
"""
import time

import numpy as np
import pytest

from mix_engine import dsp, reference_monitor as rm
from mix_engine.guardrails import LIMITS
from mix_engine.models import AUTO_FILTER_NOTES, AutoFilterAction, ProposalRequest
from mix_engine.monitor_chain import monitor_chain_params
from mix_engine.session import Session
from mix_engine.tempo_map import TempoMap

SR = 48_000


def _noise(seconds: float = 0.4, seed: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((int(SR * seconds), 2)) * 0.2


def _entry(**kw) -> dict:
    base = {"type": "auto_filter", "filter_type": 0, "slope_db_oct": 12, "cutoff_hz": 1000.0,
            "resonance": 0.3, "lfo_amount_oct": 1.0, "lfo_shape": 0, "lfo_sync": 0,
            "lfo_rate_hz": 2.0, "lfo_rate_note": 8, "lfo_phase_deg": 0.0,
            "lfo_stereo_deg": 0.0, "env_amount_oct": 0.0, "env_attack_ms": 10.0,
            "env_release_ms": 200.0, "mix": 1.0}
    base.update(kw)
    return base


# ------------------------------------------------------------------ one law

@pytest.mark.parametrize("kw", [
    dict(filter_type=0, slope_db_oct=12, resonance=0.6, lfo_amount_oct=2.0, lfo_rate_hz=3.0),
    dict(filter_type=1, slope_db_oct=24, resonance=0.9, lfo_shape=4, lfo_stereo_deg=90.0,
         env_amount_oct=2.0),
    dict(filter_type=2, slope_db_oct=24, cutoff_hz=300.0, resonance=0.2, lfo_shape=5,
         lfo_sync=1, lfo_rate_note=2, mix=0.6),
    dict(filter_type=3, cutoff_hz=5000.0, resonance=1.0, lfo_shape=2, env_amount_oct=-3.0,
         env_attack_ms=0.1),
    dict(filter_type=0, slope_db_oct=24, lfo_shape=1, lfo_phase_deg=270.0, lfo_rate_hz=20.0),
    dict(filter_type=1, lfo_shape=3, lfo_amount_oct=6.0, cutoff_hz=20000.0),
])
def test_the_engine_and_the_oracle_are_one_filter(kw):
    x = _noise()
    e = _entry(**kw)
    a = dsp._auto_filter(x, SR, e)
    b = rm._auto_filter(x, SR, e)
    assert np.max(np.abs(b)) > 0.05, "the oracle produced (near) silence - the null proves nothing"
    assert np.max(np.abs(a - b)) < 1e-10 * np.max(np.abs(b))


def test_a_keyed_envelope_is_one_law_too():
    x = _noise(seed=4)
    key = np.zeros(x.shape[0])
    key[SR // 10: SR // 5] = 0.9                     # a hit on the trigger, nothing on the input
    e = _entry(env_amount_oct=3.0, lfo_amount_oct=0.0, resonance=0.5)
    a = dsp._auto_filter(x, SR, e, key_level=key)
    b = rm._auto_filter(x, SR, e, key=key)
    assert np.max(np.abs(a - b)) < 1e-10 * np.max(np.abs(b))
    assert np.max(np.abs(a - dsp._auto_filter(x, SR, e))) > 1e-3, "the key changed nothing"


@pytest.mark.parametrize("block", [1, 7, 64, 513, 4096])
def test_the_blocks_change_no_sample(block):
    """The block-parallel walk is exact whatever the block: a block of 1 is the plain recursion."""
    x = _noise(0.2)
    g = np.tan(np.pi * np.geomspace(200.0, 8000.0, x.shape[0])[:, None].repeat(2, 1) / SR)
    k = 1.0 / 2.5
    ref = dsp._svf_sweep(x, g, k, rm.auto_filter_outputs(0, k), block=1)
    got = dsp._svf_sweep(x, g, k, rm.auto_filter_outputs(0, k), block=block)
    assert np.max(np.abs(got - ref)) < 1e-11


# ------------------------------------------------------------------ the response

def _digital_db(f: float, fc: float, kind: int, slope: int, res: float) -> float:
    """The analog SVF at Omega = tan(pi f/fs)/tan(pi fc/fs) - what the bilinear transform with
    the cutoff prewarped IS. Written from the transfer functions, not from the engine."""
    w = np.tan(np.pi * f / SR) / np.tan(np.pi * fc / SR)
    s = 1j * w
    h = 1.0
    for q in rm.auto_filter_qs(res, slope):
        den = s * s + s / q + 1.0
        num = (1.0, s * s, s / q, s * s + 1.0)[kind]
        h *= num / den
    return 20.0 * np.log10(max(abs(h), 1e-12))


@pytest.mark.parametrize("kind", [0, 1, 2, 3])
@pytest.mark.parametrize("slope", [12, 24])
def test_a_still_filter_is_its_digital_response(kind, slope):
    t = np.arange(SR) / SR
    e = _entry(filter_type=kind, slope_db_oct=slope, cutoff_hz=1000.0, resonance=0.4,
               lfo_amount_oct=0.0)
    for f in (250.0, 700.0, 1000.0, 1500.0, 4000.0):
        x = np.stack([np.sin(2 * np.pi * f * t)] * 2, 1)
        y = dsp._auto_filter(x, SR, e)[SR // 2:]
        got = 20 * np.log10(np.sqrt(np.mean(y[:, 0] ** 2)) / np.sqrt(0.5))
        want = _digital_db(f, 1000.0, kind, slope, 0.4)
        if want > -60.0:
            assert abs(got - want) < 0.05, (f, got, want)


def test_twelve_and_twentyfour_cut_as_their_names_say():
    t = np.arange(SR) / SR
    x = np.stack([np.sin(2 * np.pi * 4000.0 * t)] * 2, 1)
    for slope, want in ((12, -24.1), (24, -48.2)):
        y = dsp._auto_filter(x, SR, _entry(slope_db_oct=slope, resonance=0.0,
                                           lfo_amount_oct=0.0))[SR // 2:]
        got = 20 * np.log10(np.sqrt(np.mean(y[:, 0] ** 2)) / np.sqrt(0.5))
        # the analog figure (6 dB per order per octave); the warped digital response sits a
        # few tenths steeper this near Nyquist's quarter - its exact value is the test above
        assert abs(got - want) < 1.0, (slope, got)


@pytest.mark.parametrize("slope", [12, 24])
def test_the_agents_resonance_cap_keeps_the_peak_inside_the_pass_rail(slope):
    fs = np.geomspace(200.0, 5000.0, 4000)
    for kind in (0, 1):
        peak = max(_digital_db(f, 1000.0, kind, slope, LIMITS["auto_filter_resonance_max"])
                   for f in fs)
        assert peak <= LIMITS["eq_resonance_max_db"], (kind, slope, peak)
    # ...and the cap is where it has to be: a little more resonance passes the rail at 12 dB/oct
    assert max(_digital_db(f, 1000.0, 0, 12, LIMITS["auto_filter_resonance_max"] + 0.06)
               for f in fs) > LIMITS["eq_resonance_max_db"]


# ------------------------------------------------------------------ the clock

def test_the_lfo_rides_the_host_timeline():
    v = rm.auto_filter_values(_entry(lfo_rate_hz=1.7, lfo_phase_deg=40.0))
    whole = rm.auto_filter_cycles(SR, SR, v, 0)
    tail = rm.auto_filter_cycles(SR // 2, SR, v, 0, offset=SR // 2)
    assert np.array_equal(whole[SR // 2:], tail)
    assert abs(whole[0] - 40.0 / 360.0) < 1e-12


def test_the_right_channel_is_offset_by_the_stereo_angle():
    v = rm.auto_filter_values(_entry(lfo_stereo_deg=90.0))
    left = rm.auto_filter_cycles(100, SR, v, 0)
    right = rm.auto_filter_cycles(100, SR, v, 1)
    assert np.allclose(right - left, 0.25)


@pytest.mark.parametrize("shape", range(6))
def test_no_shape_steps_the_cutoff(shape):
    """A square, a saw or a random level crosses its step on a raised cosine: at the fastest
    rate the LFO moves by less than a tenth of its range between two samples."""
    v = rm.auto_filter_values(_entry(lfo_rate_hz=20.0, lfo_shape=shape))
    lfo = rm.auto_filter_lfo(rm.auto_filter_cycles(SR, SR, v, 0), shape)
    assert np.max(np.abs(np.diff(lfo))) < 0.1
    assert np.min(lfo) >= -1.0 - 1e-12 and np.max(lfo) <= 1.0 + 1e-12


def test_every_shape_starts_at_zero_rising_like_the_sine():
    for shape in (0, 1, 2):
        lfo = rm.auto_filter_lfo(np.array([0.0, 0.1]), shape)
        assert abs(lfo[0]) < 1e-9 and lfo[1] > 0.15, shape
    # the square sits ON its rising step at 0 (the middle of its edge) and is up after it
    sq = rm.auto_filter_lfo(np.array([0.0, 0.1, 0.6]), 4)
    assert abs(sq[0]) < 1e-9 and sq[1] == 1.0 and sq[2] == -1.0
    assert rm.auto_filter_lfo(np.array([0.1]), 3)[0] < 0.0     # the saw down falls


def test_a_synced_lfo_counts_quarter_notes_on_the_tempo_map():
    tempo = TempoMap([[0.0, 120.0, 0.0]], [[0.0, 4, 4]])
    v = rm.auto_filter_values(_entry(lfo_sync=1, lfo_rate_note="1/4"))
    assert v["lfo_quarters"] == 1.0
    cyc = rm.auto_filter_cycles(SR * 2, SR, v, 0, tempo=tempo)
    # 120 bpm: a quarter every half second - the cycle count is the quarter count
    for q in range(4):
        assert abs(cyc[q * SR // 2] - q) < 1e-9


def test_the_tempo_map_reads_as_the_engine_does():
    assert abs(TempoMap([[0, 120, 0]], [[0, 4, 4]]).ppq_at(1.0) - 2.0) < 1e-12
    # 6/8: the beat is an eighth (`dependsOnTimeSignature`, the DAW's setting) and lasts
    # 240 / (bpm x 8) s, so the BEATS run twice as fast while the QUARTERS do not move - a
    # synced "1/4" is the same speed in any time signature
    six8 = TempoMap([[0, 120, 0]], [[0, 6, 8]])
    assert abs(six8.ppq_at(1.0) - 2.0) < 1e-12 and abs(six8.beats_at(1.0) - 4.0) < 1e-12
    # a step from 120 to 60 at beat 4 (two seconds in): a quarter a second after it
    step = TempoMap([[0, 120, 1.0], [4, 60, 1.0]], [[0, 4, 4]])
    assert abs(step.ppq_at(2.0) - 4.0) < 1e-12 and abs(step.ppq_at(3.0) - 5.0) < 1e-12
    # a linear ramp 120 -> 60 over 8 beats is cut into 32 constant sections, each at the curve's
    # tempo where it starts: the first quarter takes 0.5 s, the ramp ends later than at 120
    ramp = TempoMap([[0, 120, 0.0], [8, 60, 0.0]], [[0, 4, 4]])
    assert len(ramp.sections) == 33
    assert abs(ramp.ppq_at(0.125) - 0.25) < 1e-12
    assert ramp.ppq_at(4.0) < 8.0


def test_the_tempo_fact_is_checked_and_kept(demo_project):
    session = Session(demo_project)
    assert session.tempo_map is None
    session.set_tempo_map({"tempos": [[0, 96, 0]], "time_sigs": [[0, 4, 4]]})
    assert session.tempo_map["tempos"] == [[0.0, 96.0, 0.0]]
    for bad in ({"tempos": [[1, 96, 0]], "time_sigs": [[0, 4, 4]]},
                {"tempos": [[0, 0, 0]], "time_sigs": [[0, 4, 4]]},
                {"tempos": [[0, 96, 0]], "time_sigs": []},
                {"tempos": [[0, float("nan"), 0]], "time_sigs": [[0, 4, 4]]}):
        with pytest.raises(ValueError):
            session.set_tempo_map(bad)
    before = session.render_fingerprint()
    session.set_tempo_map({"tempos": [[0, 140, 0]], "time_sigs": [[0, 4, 4]]})
    assert session.render_fingerprint() != before


# ------------------------------------------------------------------ the envelope

def test_the_envelope_opens_the_filter_with_the_level():
    x = np.zeros((SR // 2, 2))
    x[SR // 10: SR // 5] = 0.8
    v = rm.auto_filter_values(_entry(env_amount_oct=3.0, lfo_amount_oct=0.0,
                                     env_attack_ms=1.0, env_release_ms=50.0))
    fc = rm.auto_filter_modulation(x, SR, v, follower=dsp._rising_follower)[:, 0]
    assert abs(fc[SR // 20] - 1000.0) < 1e-6                         # silence: at rest
    # the hit (0.8 = -1.9 dBFS, 0.96 of the envelope's range): near three octaves up
    assert fc[SR // 5 - 1] > 1000.0 * 2 ** 2.85
    assert fc[-1] < fc[SR // 5 - 1] / 4                              # and back down after it


def test_a_negative_envelope_closes_it():
    x = np.full((SR // 4, 2), 0.8)
    v = rm.auto_filter_values(_entry(env_amount_oct=-2.0, lfo_amount_oct=0.0))
    fc = rm.auto_filter_modulation(x, SR, v, follower=dsp._rising_follower)[:, 0]
    assert fc[-1] < 1000.0 / 3.5


# ------------------------------------------------------------------ the trigger

def _with_filter(session: Session, stem: str, **kw) -> None:
    res = session.propose(ProposalRequest(rationale="move it", predicted_impact=[],
                                          actions=[{"type": "auto_filter", "stem_id": stem,
                                                    **kw}]))
    assert res.accepted, res.violations
    session.apply(res.proposal_id)


def test_a_keyed_filter_follows_its_trigger_in_the_render(demo_project):
    session = Session(demo_project)
    _with_filter(session, "stem_a", env_amount_oct=2.0, lfo_amount_oct=0.0, cutoff_hz=400.0)
    own, _ = dsp.render_stem(session, "stem_a")
    session.set_key("stem_a", "auto_filter", "stem_b")
    keyed, _ = dsp.render_stem(session, "stem_a")
    n = min(len(own), len(keyed))
    assert not np.allclose(own[:n], keyed[:n]), "the trigger changed nothing"
    session.set_key("stem_a", "auto_filter", "")       # back to its own input
    again, _ = dsp.render_stem(session, "stem_a")
    assert np.allclose(own[:n], again[:n])


def test_the_trigger_doors_refuse_what_cannot_be(demo_project):
    session = Session(demo_project)
    _with_filter(session, "stem_a")
    with pytest.raises(ValueError, match="own trigger"):
        session.set_key("stem_a", "auto_filter", "stem_a")
    with pytest.raises(ValueError, match="unknown trigger"):
        session.set_key("stem_a", "auto_filter", "nope")
    with pytest.raises(ValueError, match="no Auto Filter"):
        session.set_key("stem_b", "auto_filter", "stem_a")
    # a cycle across the two keyed types: B ducks under A, so A cannot listen to B
    session.set_duck("stem_b", "stem_a")
    with pytest.raises(ValueError, match="trigger each other"):
        session.set_key("stem_a", "auto_filter", "stem_b")


def test_a_track_has_one_trigger(demo_project, tmp_path):
    import shutil
    third = demo_project / "stem_c.wav"
    shutil.copy(demo_project / "stem_b.wav", third)
    session = Session(demo_project)
    _with_filter(session, "stem_a")
    session.set_duck("stem_a", "stem_b")
    with pytest.raises(ValueError, match="one trigger"):
        session.set_key("stem_a", "auto_filter", "stem_c")
    session.set_key("stem_a", "auto_filter", "stem_b")              # the same one is fine
    with pytest.raises(ValueError, match="one trigger"):
        session.set_duck("stem_a", "stem_c")                        # ...and it holds both ways
    res = session.propose(ProposalRequest(
        rationale="re-key", predicted_impact=[],
        actions=[{"type": "auto_filter", "stem_id": "stem_a", "key_stem_id": "stem_c"}]))
    assert not res.accepted and any(v.rule == "one_trigger_per_track" for v in res.violations)


def test_a_group_filter_takes_no_trigger(demo_project):
    session = Session(demo_project)
    res = session.propose(ProposalRequest(
        rationale="group them", predicted_impact=[],
        actions=[{"type": "group", "bus_id": "pair", "members": ["stem_a", "stem_b"]},
                 {"type": "auto_filter", "stem_id": "pair", "key_stem_id": "stem_a"}]))
    assert not res.accepted and any(v.rule == "auto_filter_key_on_group" for v in res.violations)


def test_a_pasted_filter_whose_trigger_is_missing_keeps_the_device(demo_project):
    session = Session(demo_project)
    chain, dropped = session._validated_chain(
        [{"type": "auto_filter", **{k: v for k, v in _entry().items() if k != "type"},
          "key_stem_id": "not_here"}], "stem_a")
    assert len(chain) == 1 and chain[0]["key_stem_id"] == ""
    assert any("own input" in d for d in dropped)


# ------------------------------------------------------------------ bounds + registration

def test_guardrails_cap_the_agent(demo_project):
    session = Session(demo_project)
    ok = session.propose(ProposalRequest(
        rationale="breathe", predicted_impact=[],
        actions=[{"type": "auto_filter", "stem_id": "stem_a",
                  "resonance": LIMITS["auto_filter_resonance_max"],
                  "lfo_amount_oct": LIMITS["auto_filter_lfo_amount_max_oct"],
                  "env_amount_oct": -LIMITS["auto_filter_env_amount_max_oct"]}]))
    assert ok.accepted, ok.violations
    for field, value, rule in (
            ("resonance", LIMITS["auto_filter_resonance_max"] + 0.2, "auto_filter_resonance_max"),
            ("lfo_amount_oct", 5.0, "auto_filter_lfo_amount_max_oct"),
            ("env_amount_oct", -4.0, "auto_filter_env_amount_max_oct")):
        res = session.propose(ProposalRequest(
            rationale="too much", predicted_impact=[],
            actions=[{"type": "auto_filter", "stem_id": "stem_a", field: value}]))
        assert not res.accepted and any(v.rule == rule for v in res.violations), field


def test_the_master_is_the_users():
    from mix_engine.session import BUS_ACTION_TYPES, MASTER_ACTION_TYPES, SUPPORTED_ACTION_TYPES
    assert "auto_filter" in SUPPORTED_ACTION_TYPES and "auto_filter" in BUS_ACTION_TYPES
    assert "auto_filter" not in MASTER_ACTION_TYPES


def test_the_user_knob_goes_further_than_the_agent_may():
    a = AutoFilterAction(stem_id="a", resonance=1.0, lfo_amount_oct=6.0, env_amount_oct=-6.0)
    assert a.resonance == 1.0 and a.lfo_amount_oct == 6.0


def test_the_word_values_are_read():
    a = AutoFilterAction(stem_id="a", filter_type="HP", lfo_shape="saw_up",
                         lfo_rate_note="1/8t", lfo_sync="on")
    assert (a.filter_type, a.lfo_shape, a.lfo_sync) == (1, 2, 1)
    assert AUTO_FILTER_NOTES[a.lfo_rate_note][0] == "1/8T"
    for bad in ({"filter_type": "lowpass"}, {"lfo_shape": "wobble"}, {"lfo_rate_note": "3/7"},
                {"slope_db_oct": 18}):
        with pytest.raises(ValueError):
            AutoFilterAction(stem_id="a", **bad)


def test_the_co_edit_snaps_the_discrete_fields(demo_project):
    session = Session(demo_project)
    session.add_stem_device("stem_a", "auto_filter")
    for param, value, want in (("slope_db_oct", 20.0, 24.0), ("slope_db_oct", 15.0, 12.0),
                               ("filter_type", 2.6, 3.0), ("lfo_sync", 0.7, 1.0),
                               ("lfo_rate_note", 99.0, 16.0), ("cutoff_hz", 1e6, 20000.0)):
        session.set_master_device_param("auto_filter", 0, param, value, target="stem_a")
        entry = next(e for e in session.stems["stem_a"]["chain"] if e["type"] == "auto_filter")
        assert entry[param] == want, (param, entry[param])


def test_monitor_contract_emits_every_param():
    out = monitor_chain_params([_entry(stem_id="a", filter_type=1, lfo_sync=1,
                                       lfo_rate_note=5, key_stem_id="kick")])
    (e,) = out["entries"]
    assert e["type"] == "auto_filter" and e["filter_type"] == 1 and e["lfo_sync"] == 1
    assert e["lfo_quarters"] == 0.5 and e["key_stem_id"] == "kick" and e["enabled"] is True
    for field in ("slope_db_oct", "cutoff_hz", "resonance", "lfo_amount_oct", "lfo_shape",
                  "lfo_rate_hz", "lfo_rate_note", "lfo_phase_deg", "lfo_stereo_deg",
                  "env_amount_oct", "env_attack_ms", "env_release_ms", "mix"):
        assert field in e, field


def test_it_lands_in_the_chain_and_renders(demo_project):
    session = Session(demo_project)
    before, _ = dsp.render_stem(session, "stem_a")
    _with_filter(session, "stem_a", filter_type="lp", cutoff_hz=500.0, lfo_amount_oct=1.5)
    entries = [e for e in session.applied_chain("stem_a") if e["type"] == "auto_filter"]
    assert len(entries) == 1 and entries[0]["cutoff_hz"] == 500.0
    after, _ = dsp.render_stem(session, "stem_a")
    n = min(len(before), len(after))
    assert not np.allclose(before[:n], after[:n]), "the filter made no difference to the render"
    # one per chain: a second proposal adjusts the first
    _with_filter(session, "stem_a", cutoff_hz=900.0)
    entries = [e for e in session.applied_chain("stem_a") if e["type"] == "auto_filter"]
    assert len(entries) == 1 and entries[0]["cutoff_hz"] == 900.0


def test_a_song_renders_in_seconds():
    x = np.random.default_rng(0).standard_normal((SR * 180, 2)) * 0.1
    t0 = time.perf_counter()
    dsp._auto_filter(x, SR, _entry(slope_db_oct=24, resonance=0.5, lfo_amount_oct=2.0,
                                   env_amount_oct=1.0))
    assert time.perf_counter() - t0 < 8.0


# ------------------------------------------------------------------ the two co-edits

def test_the_tempo_fact_reaches_the_session_through_its_coedit(demo_project):
    """The app derives the map on its 2 Hz watch (dawsession::tempoMapFact) and sends it as a
    co-edit the brain applies - a FACT report, which starts no measurement."""
    from orchestrator import daw_bridge as B
    session = Session(demo_project)
    url = ("/api/set_tempo_map?tempos=0.000000000000%3A96.000000000000%3A0.000000000000%3B"
           "8.000000000000%3A120.000000000000%3A0.300000011921&time_sigs=0.000000000000%3A4%3A4"
           "&key_beats=-")
    B._apply_one_coedit(session, url)
    held = Session(demo_project).tempo_map
    assert held["tempos"] == [[0.0, 96.0, 0.0], [8.0, 120.0, 0.300000011921]]
    assert held["time_sigs"] == [[0.0, 4, 4]] and held["key_beats"] == []
    assert B.coedit_is_fact_report(url)
    # ...and the curve the app printed to 12 places is the engine's float again
    assert TempoMap.from_fact(held).sections[-1].bpm == 120.0


def test_the_app_calls_the_tempo_fact_a_fact_too():
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    app = (repo / "daw" / "Source" / "App.cpp").read_text()
    body = app[app.index("static bool coeditIsFactReport"):][:400]
    assert 'url.startsWith ("/api/set_tempo_map?")' in body   # (and, since Q14, the app's measure of the mix)
    assert "dawsession::tempoMapFact (*edit)" in app
    assert "tempoFactSent.clear()" in app      # another edit is another map to tell


def test_a_trigger_coedit_keys_the_filter(demo_project):
    from orchestrator import daw_bridge as B
    session = Session(demo_project)
    session.add_stem_device("stem_a", "auto_filter")
    B._apply_one_coedit(session, "/api/set_key?type=auto_filter&stem=stem_a&key=stem_b")
    entry = next(e for e in Session(demo_project).stems["stem_a"]["chain"] if e["type"] == "auto_filter")
    assert entry["key_stem_id"] == "stem_b"
    B._apply_one_coedit(Session(demo_project), "/api/set_key?type=auto_filter&stem=stem_a&key=none")
    entry = next(e for e in Session(demo_project).stems["stem_a"]["chain"] if e["type"] == "auto_filter")
    assert entry["key_stem_id"] == ""
