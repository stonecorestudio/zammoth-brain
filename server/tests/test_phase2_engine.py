"""Phase 2 engine tests: full action surface, masking, new guardrails."""
import numpy as np
import pytest
import soundfile as sf
from mcp.shared.memory import create_connected_server_and_client_session

from mix_engine import dsp
from mix_engine.mcp_server import create_server
from mix_engine.models import ProposalRequest
from mix_engine.session import Session

from test_phase1_acceptance import call
from test_session import gain_request


def request_with(actions: list[dict], rationale: str = "test") -> ProposalRequest:
    return ProposalRequest(rationale=rationale, predicted_impact=[], actions=actions)


def apply_actions(session: Session, actions: list[dict]) -> None:
    result = session.propose(request_with(actions))
    assert result.accepted, result.violations
    applied = session.apply(result.proposal_id)
    assert applied.ok, applied.message


def band_power(audio: np.ndarray, sample_rate: int, lo_hz: float, hi_hz: float) -> float:
    spectrum = np.abs(np.fft.rfft(audio.mean(axis=1))) ** 2
    freqs = np.fft.rfftfreq(audio.shape[0], 1.0 / sample_rate)
    return float(spectrum[(freqs >= lo_hz) & (freqs < hi_hz)].sum())


def test_eq_peak_boost_raises_band(demo_project):
    session = Session(demo_project)
    before, sr = dsp.render_stem(session, "stem_b")
    apply_actions(session, [{"type": "eq", "stem_id": "stem_b",
                             "band": {"freq_hz": 3000, "gain_db": 6.0, "q": 1.0,
                                      "shape": "peak"}}])
    after, _ = dsp.render_stem(session, "stem_b")
    assert band_power(after, sr, 2000, 4000) > band_power(before, sr, 2000, 4000) * 1.5


def test_hpf_attenuates_lows(demo_project):
    session = Session(demo_project)
    before, sr = dsp.render_stem(session, "stem_b")
    apply_actions(session, [{"type": "eq", "stem_id": "stem_b",
                             "band": {"freq_hz": 120, "gain_db": 0.0, "q": 0.7,
                                      "shape": "hpf"}}])
    after, _ = dsp.render_stem(session, "stem_b")
    assert band_power(after, sr, 20, 60) < band_power(before, sr, 20, 60) * 0.5


def test_compressor_reduces_dynamic_spread(demo_project):
    from mix_engine import analysis
    session = Session(demo_project)
    before, sr = dsp.render_stem(session, "stem_a")
    spread_before = analysis._dynamic_spread_db(before, sr)
    apply_actions(session, [{"type": "comp", "stem_id": "stem_a",
                             "threshold_db": -30.0, "ratio": 4.0,
                             "attack_ms": 5.0, "release_ms": 100.0, "makeup_db": 0.0}])
    after, _ = dsp.render_stem(session, "stem_a")
    assert analysis._dynamic_spread_db(after, sr) < spread_before


def test_deess_attenuates_highs(demo_project):
    session = Session(demo_project)
    before, sr = dsp.render_stem(session, "stem_b")
    apply_actions(session, [{"type": "deess", "stem_id": "stem_b",
                             "threshold_db": -45.0, "freq_hz": 4000.0}])
    after, _ = dsp.render_stem(session, "stem_b")
    assert band_power(after, sr, 6000, 12000) < band_power(before, sr, 6000, 12000)
    # the band below the de-esser split stays roughly untouched
    assert band_power(after, sr, 100, 1000) == pytest.approx(
        band_power(before, sr, 100, 1000), rel=0.5)


def test_pan_and_width():
    mono = np.linspace(-0.5, 0.5, 1000).reshape(-1, 1)
    panned = dsp._pan(mono, 0.8)
    assert panned.shape[1] == 2
    assert np.abs(panned[:, 1]).sum() > np.abs(panned[:, 0]).sum()
    assert np.allclose(dsp._pan(mono, 0.0), mono)  # unity at center

    stereo = np.hstack([mono, -mono])  # pure side signal
    collapsed = dsp._width(stereo, 0.0)
    assert np.allclose(collapsed[:, 0], collapsed[:, 1])
    wider = dsp._width(stereo, 2.0)
    assert np.abs(wider[:, 0] - wider[:, 1]).sum() > np.abs(stereo[:, 0] - stereo[:, 1]).sum()


def test_pan_action_changes_master_channels(demo_project):
    session = Session(demo_project)
    apply_actions(session, [{"type": "pan", "stem_id": "stem_a", "pan": 1.0}])
    master, _ = dsp.render_master(session)
    assert master.shape[1] == 2
    assert np.abs(master[:, 1]).sum() > np.abs(master[:, 0]).sum()


def test_full_chain_undo_is_bit_identical(demo_project):
    """The determinism invariant must survive the Phase 2 processors."""
    session = Session(demo_project)
    baseline, sr = dsp.render_master(session)
    result = session.propose(request_with([
        {"type": "eq", "stem_id": "stem_b",
         "band": {"freq_hz": 2500, "gain_db": 3.0, "q": 1.2, "shape": "peak"}},
        {"type": "comp", "stem_id": "stem_a", "threshold_db": -24.0, "ratio": 3.0,
         "attack_ms": 10.0, "release_ms": 80.0, "makeup_db": 2.0},
        {"type": "pan", "stem_id": "stem_a", "pan": -0.3},
        {"type": "gain", "stem_id": "stem_b", "delta_db": -1.5},
    ]))
    assert result.accepted
    assert session.apply(result.proposal_id).ok
    changed, _ = dsp.render_master(session)
    assert changed.shape != baseline.shape or not np.array_equal(changed, baseline)

    assert session.undo(result.proposal_id).ok
    restored, _ = dsp.render_master(session)
    assert restored.shape == baseline.shape
    assert np.array_equal(restored, baseline)


def test_hpf_guardrail_by_stem_tag(demo_project):
    session = Session(demo_project)
    too_high = session.propose(request_with([
        {"type": "eq", "stem_id": "stem_b",
         "band": {"freq_hz": 200, "gain_db": 0.0, "q": 0.7, "shape": "hpf"}}]))
    assert not too_high.accepted
    assert too_high.violations[0].rule == "hpf_max_hz"
    assert too_high.violations[0].limit == 120.0

    session.set_stem_tags("stem_b", ["bass"])
    bass_limited = session.propose(request_with([
        {"type": "eq", "stem_id": "stem_b",
         "band": {"freq_hz": 60, "gain_db": 0.0, "q": 0.7, "shape": "hpf"}}]))
    assert not bass_limited.accepted
    assert bass_limited.violations[0].limit == 40.0

    fine = session.propose(request_with([
        {"type": "eq", "stem_id": "stem_b",
         "band": {"freq_hz": 35, "gain_db": 0.0, "q": 0.7, "shape": "hpf"}}]))
    assert fine.accepted


@pytest.mark.anyio
async def test_analyze_masking_roundtrip(demo_project):
    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        masking = await call(client, "analyze_masking",
                             {"stem_a": "stem_a", "stem_b": "stem_b"})
        assert masking["worst_band"] == "mid"  # the sine's band is all that counts
        baseline_excess = masking["worst_band_excess_db"]

        proposal = await call(client, "propose_actions", {"req": {
            "rationale": "duck the noise bed",
            "predicted_impact": [{
                "metric": "masking(stem_a,stem_b).worst_band_excess_db",
                "predicted_delta": -6.0, "unit": "dB"}],
            "actions": [{"type": "gain", "stem_id": "stem_b", "delta_db": -6.0}],
        }})
        assert (await call(client, "apply_proposal",
                           {"proposal_id": proposal["proposal_id"]}))["ok"]
        await call(client, "render_and_analyze")

        after = await call(client, "analyze_masking",
                           {"stem_a": "stem_a", "stem_b": "stem_b"})
        assert after["worst_band_excess_db"] == pytest.approx(baseline_excess - 6.0, abs=0.3)

        # the engine measured the predicted metric on its own
        (applied,) = await call(client, "get_proposals", {"status": "applied"})
        (measurement,) = applied["measured_impact"]
        assert measurement["metric"] == "masking(stem_a,stem_b).worst_band_excess_db"
        assert measurement["measured_delta"] == pytest.approx(-6.0, abs=0.3)


def test_phase2_master_metrics(demo_project):
    from mix_engine import analysis
    session = Session(demo_project)
    audio, sr = dsp.render_master(session)
    result = analysis.analyze_master(audio, sr, stems=["stem_a", "stem_b"], pass_number=1)
    assert result.crest_factor_db is not None and result.crest_factor_db > 0
    assert result.spectral_tilt_db_per_oct is not None
    assert result.stereo_correlation == 1.0  # two mono stems play from both speakers (E4·R4)

    stem = analysis.analyze_stem(*dsp.render_stem(session, "stem_a"),
                                 stem_id="stem_a", applied_chain=[])
    assert stem.band_energy is not None
    assert stem.dynamic_spread_db is not None and stem.dynamic_spread_db > 0
    assert stem.transient_density is not None
