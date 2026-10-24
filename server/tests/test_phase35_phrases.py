"""Phase 35 — phrase-aware readout for existing ranged tools."""
import numpy as np
import pytest
import soundfile as sf
from mcp.shared.memory import create_connected_server_and_client_session

from mix_engine import analysis, dsp
from mix_engine.mcp_server import create_server
from mix_engine.session import Session
from mix_engine.models import ANALYSIS_VERSION

from test_phase1_acceptance import call

SR = 48_000


def _tone(duration_s: float = 4.0, amp: float = 0.04) -> np.ndarray:
    t = np.arange(int(SR * duration_s)) / SR
    return (amp * np.sin(2 * np.pi * 220.0 * t)
            + amp * 0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


def _sibilant_stem() -> np.ndarray:
    t = np.arange(int(8.0 * SR)) / SR
    period = 1.0 / 1.5
    envelope = np.exp(-(t % period) / 0.08)
    return (0.4 * np.sin(2 * np.pi * 800.0 * t)
            + 0.6 * envelope * np.sin(2 * np.pi * 7000.0 * t)).astype(np.float32)


def _band_rms(audio: np.ndarray, sample_rate: int,
              lo_hz: float, hi_hz: float) -> float:
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    spectrum = np.fft.rfft(mono)
    freqs = np.fft.rfftfreq(mono.size, d=1.0 / sample_rate)
    mask = (freqs >= lo_hz) & (freqs < hi_hz)
    filtered = np.fft.irfft(spectrum * mask, n=mono.size)
    return float(np.sqrt(np.mean(filtered ** 2)))


def test_phrase_level_moves_find_local_outliers():
    audio = _tone()
    audio[int(1.0 * SR):int(1.8 * SR)] *= 0.2   # buried phrase
    audio[int(2.4 * SR):int(3.1 * SR)] *= 4.0   # jumped-out phrase

    result = analysis.analyze_phrases(audio, SR, stem_id="vox")
    quiet = [m for m in result.level_moves if m.reason == "too_quiet"]
    loud = [m for m in result.level_moves if m.reason == "too_loud"]

    assert quiet and quiet[0].delta_db > 0
    assert 0.8 <= quiet[0].start_s <= 1.2
    assert loud and loud[0].delta_db < 0
    assert 2.0 <= loud[0].start_s <= 2.6
    assert result.silent is False


def test_phrase_analysis_finds_sibilance_bursts():
    audio = _tone(amp=0.02)
    t = np.arange(audio.shape[0]) / SR
    burst = (t >= 1.2) & (t < 1.45)
    audio[burst] += 0.18 * np.sin(2 * np.pi * 7200.0 * t[burst])

    result = analysis.analyze_phrases(audio, SR, stem_id="vox")

    assert result.sibilance_bursts
    top = result.sibilance_bursts[0]
    assert 1.0 <= top.start_s <= 1.3
    assert 5000 <= top.freq_hz <= 6500
    assert top.threshold_db < -12.0


@pytest.mark.anyio
async def test_analyze_phrases_round_trips_over_mcp(tmp_path):
    project = tmp_path / "phrases"
    project.mkdir()
    audio = _tone()
    audio[int(1.0 * SR):int(1.8 * SR)] *= 0.2
    sf.write(project / "vox.wav", audio, SR)

    async with create_connected_server_and_client_session(
            create_server(project)) as client:
        result = await call(client, "analyze_phrases", {"stem_id": "vox"})

    assert result["stem_id"] == "vox"
    assert result["analysis_version"] == ANALYSIS_VERSION
    assert result["level_moves"]
    assert result["level_moves"][0]["end_s"] > result["level_moves"][0]["start_s"]


@pytest.mark.anyio
async def test_phrase_level_move_applies_locally_and_undoes(tmp_path):
    project = tmp_path / "phrase_gain"
    project.mkdir()
    audio = _tone()
    audio[int(1.0 * SR):int(1.8 * SR)] *= 0.2
    sf.write(project / "vox.wav", audio, SR)
    dry, _ = dsp.render_stem(Session(project), "vox")

    async with create_connected_server_and_client_session(
            create_server(project)) as client:
        phrases = await call(client, "analyze_phrases", {"stem_id": "vox"})
        move = next(m for m in phrases["level_moves"]
                    if m["reason"] == "too_quiet")
        proposal = await call(client, "propose_actions", {"req": {
            "rationale": "level the quiet phrase",
            "predicted_impact": [{
                "metric": "stem(vox).dynamic_spread_db",
                "predicted_delta": -abs(move["delta_db"]), "unit": "dB"}],
            "actions": [{"type": "gain", "stem_id": "vox",
                         "delta_db": min(move["delta_db"], 2.0),
                         "time_range_s": [move["start_s"], move["end_s"]]}],
        }})
        assert proposal["accepted"], proposal["violations"]
        assert (await call(client, "apply_proposal",
                           {"proposal_id": proposal["proposal_id"]}))["ok"]

        wet, sample_rate = dsp.render_stem(Session(project), "vox")
        start = int((move["start_s"] + dsp.RIDE_FADE_S) * sample_rate)
        end = int((move["end_s"] - dsp.RIDE_FADE_S) * sample_rate)
        assert np.sqrt(np.mean(wet[start:end] ** 2)) > np.sqrt(
            np.mean(dry[start:end] ** 2))
        assert (await call(client, "undo_proposal",
                           {"proposal_id": proposal["proposal_id"]}))["ok"]

    undone, _ = dsp.render_stem(Session(project), "vox")
    assert np.array_equal(undone, dry)


@pytest.mark.anyio
async def test_phrase_deess_applies_locally_and_undoes(tmp_path):
    project = tmp_path / "phrase_deess"
    project.mkdir()
    audio = _sibilant_stem()
    sf.write(project / "vox.wav", audio, SR)
    dry, _ = dsp.render_stem(Session(project), "vox")

    async with create_connected_server_and_client_session(
            create_server(project)) as client:
        phrases = await call(client, "analyze_phrases", {"stem_id": "vox"})
        burst = phrases["sibilance_bursts"][0]
        proposal = await call(client, "propose_actions", {"req": {
            "rationale": "de-ess the burst only",
            "predicted_impact": [{
                "metric": "stem(vox).sibilance_density",
                "predicted_delta": -0.1, "unit": "ratio"}],
            "actions": [{"type": "deess", "stem_id": "vox",
                         "threshold_db": -40.0,
                         "freq_hz": burst["freq_hz"],
                         "time_range_s": [burst["start_s"], burst["end_s"]]}],
        }})
        assert proposal["accepted"], proposal["violations"]
        assert (await call(client, "apply_proposal",
                           {"proposal_id": proposal["proposal_id"]}))["ok"]

        wet, sample_rate = dsp.render_stem(Session(project), "vox")
        start = int((burst["start_s"] + dsp.RIDE_FADE_S) * sample_rate)
        end = int((burst["end_s"] - dsp.RIDE_FADE_S) * sample_rate)
        dry_sibilance = _band_rms(
            dry[start:end], sample_rate, burst["freq_hz"],
            analysis.SIBILANCE_HI_HZ)
        wet_sibilance = _band_rms(
            wet[start:end], sample_rate, burst["freq_hz"],
            analysis.SIBILANCE_HI_HZ)
        assert wet_sibilance < dry_sibilance * 0.8
        outside = int((burst["end_s"] + dsp.RIDE_FADE_S) * sample_rate)
        assert np.array_equal(wet[outside:], dry[outside:])
        assert (await call(client, "undo_proposal",
                           {"proposal_id": proposal["proposal_id"]}))["ok"]

    undone, _ = dsp.render_stem(Session(project), "vox")
    assert np.array_equal(undone, dry)
