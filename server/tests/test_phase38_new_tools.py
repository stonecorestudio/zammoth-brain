"""Phase 38 (docs/05-roadmap.md): highest-value new tools.

Covers the new proposal-owned palette:
- dynamic_eq: a deterministic downward spectral cut that only bites when its
  band crosses threshold.
- return_delay / return_reverb / send: agent-owned wet return busses fed by
  stem or group-bus sends, deterministic and undoable.
"""
import numpy as np
import pytest
import soundfile as sf
from mcp.shared.memory import create_connected_server_and_client_session

from mix_engine import dsp
from mix_engine.mcp_server import create_server
from mix_engine.session import Session

from test_phase1_acceptance import call

SR = 48000


def engine(project):
    return create_connected_server_and_client_session(create_server(project))


async def submit(client, actions: list[dict], rationale: str = "phase38") -> str:
    proposal = await call(client, "propose_actions", {"req": {
        "rationale": rationale,
        "predicted_impact": [],
        "actions": actions,
    }})
    assert proposal["accepted"], proposal["violations"]
    pid = proposal["proposal_id"]
    assert (await call(client, "apply_proposal", {"proposal_id": pid}))["ok"]
    return pid


def _write(project, name: str, audio: np.ndarray, sample_rate: int = SR) -> None:
    sf.write(project / f"{name}.wav", audio.astype(np.float32), sample_rate)


def _band_rms(audio: np.ndarray, sample_rate: int, center_hz: float,
              start_s: float, end_s: float, q: float = 10.0) -> float:
    segment = audio[int(start_s * sample_rate): int(end_s * sample_rate)]
    mono = segment.mean(axis=1) if segment.ndim == 2 else segment
    spectrum = np.fft.rfft(mono)
    freqs = np.fft.rfftfreq(mono.shape[0], d=1.0 / sample_rate)
    width_hz = center_hz / q
    mask = np.abs(freqs - center_hz) <= width_hz / 2.0
    band = np.fft.irfft(spectrum * mask, n=mono.shape[0])
    return float(np.sqrt(np.mean(band ** 2)))


@pytest.mark.anyio
async def test_dynamic_eq_reduces_triggered_band_more_than_quiet_region(tmp_path):
    project = tmp_path / "dynamic-eq"
    project.mkdir()
    t = np.arange(int(4.0 * SR)) / SR
    tone = np.sin(2 * np.pi * 4000.0 * t)
    envelope = np.where(t < 2.0, 0.04, 0.5)
    anchor = 0.12 * np.sin(2 * np.pi * 300.0 * t)
    _write(project, "lead", anchor + envelope * tone)

    dry, sample_rate = dsp.render_stem(Session(project), "lead")
    async with engine(project) as client:
        pid = await submit(client, [{
            "type": "dynamic_eq",
            "stem_id": "lead",
            "freq_hz": 4000.0,
            "gain_db": -9.0,
            "q": 10.0,
            "threshold_db": -20.0,
            "ratio": 6.0,
            "attack_ms": 2.0,
            "release_ms": 80.0,
        }], "clamp the 4 kHz flare")

    wet, _ = dsp.render_stem(Session(project), "lead")
    quiet_before = _band_rms(dry, sample_rate, 4000.0, 0.4, 1.6)
    quiet_after = _band_rms(wet, sample_rate, 4000.0, 0.4, 1.6)
    loud_before = _band_rms(dry, sample_rate, 4000.0, 2.4, 3.6)
    loud_after = _band_rms(wet, sample_rate, 4000.0, 2.4, 3.6)
    quiet_delta_db = 20.0 * np.log10(quiet_after / quiet_before)
    loud_delta_db = 20.0 * np.log10(loud_after / loud_before)

    assert loud_delta_db < -3.0
    assert quiet_delta_db > -1.0
    assert loud_delta_db < quiet_delta_db - 3.0

    async with engine(project) as client:
        assert (await call(client, "undo_proposal", {"proposal_id": pid}))["ok"]
    undone, _ = dsp.render_stem(Session(project), "lead")
    assert np.array_equal(undone, dry)


@pytest.mark.anyio
async def test_return_reverb_delay_audible_deterministic_and_undoable(tmp_path):
    project = tmp_path / "returns"
    project.mkdir()
    impulse = np.zeros((int(2.0 * SR), 1), dtype=np.float32)
    impulse[0, 0] = 0.8
    _write(project, "snare", impulse)

    dry, _ = dsp.render_master(Session(project))
    async with engine(project) as client:
        pid = await submit(client, [
            {"type": "return_delay", "return_id": "slap",
             "delay_ms": 45.0, "feedback": 0.3, "tone": 0.7},
            {"type": "return_reverb", "return_id": "room",
             "decay_s": 0.7, "pre_delay_ms": 8.0, "tone": 0.55},
            {"type": "send", "stem_id": "snare", "return_id": "slap",
             "gain_db": -9.0},
            {"type": "send", "stem_id": "snare", "return_id": "room",
             "gain_db": -12.0},
        ], "add a little room and slap")

    wet, sample_rate = dsp.render_master(Session(project))
    again, _ = dsp.render_master(Session(project))
    assert np.array_equal(wet, again)
    assert not np.array_equal(wet, dry)
    tail_start = int(0.02 * sample_rate)
    assert np.sum(wet[tail_start:] ** 2) > 1e-8
    assert np.sum(dry[tail_start:] ** 2) == pytest.approx(0.0)

    async with engine(project) as client:
        assert (await call(client, "undo_proposal", {"proposal_id": pid}))["ok"]
    undone, _ = dsp.render_master(Session(project))
    assert np.array_equal(undone, dry)


@pytest.mark.anyio
async def test_return_routing_rejects_unknown_return_and_hot_send(tmp_path):
    project = tmp_path / "return-rules"
    project.mkdir()
    t = np.arange(int(1.0 * SR)) / SR
    _write(project, "vox", 0.2 * np.sin(2 * np.pi * 440.0 * t))

    async with engine(project) as client:
        missing = await call(client, "propose_actions", {"req": {
            "rationale": "send to nowhere",
            "predicted_impact": [],
            "actions": [{"type": "send", "stem_id": "vox",
                         "return_id": "ghost", "gain_db": -12.0}],
        }})
        assert not missing["accepted"]
        assert "unknown_return" in [v["rule"] for v in missing["violations"]]

        hot = await call(client, "propose_actions", {"req": {
            "rationale": "too much wet level",
            "predicted_impact": [],
            "actions": [
                {"type": "return_reverb", "return_id": "room",
                 "decay_s": 1.2, "pre_delay_ms": 10.0, "tone": 0.5},
                {"type": "send", "stem_id": "vox", "return_id": "room",
                 "gain_db": 0.0},
            ],
        }})
        assert not hot["accepted"]
        assert "return_send_max_db" in [v["rule"] for v in hot["violations"]]


@pytest.mark.anyio
async def test_get_state_exposes_returns_and_bus_sends(tmp_path):
    project = tmp_path / "return-state"
    project.mkdir()
    t = np.arange(int(1.0 * SR)) / SR
    _write(project, "vox", 0.2 * np.sin(2 * np.pi * 440.0 * t))

    async with engine(project) as client:
        await submit(client, [
            {"type": "group", "bus_id": "vocals", "members": ["vox"]},
            {"type": "return_delay", "return_id": "slap",
             "delay_ms": 90.0, "feedback": 0.2, "tone": 0.6},
            {"type": "send", "stem_id": "vocals", "return_id": "slap",
             "gain_db": -10.0},
        ], "route vocal bus into slap")

        state = await call(client, "get_state")
        (bus,) = state["buses"]
        assert bus["bus_id"] == "vocals"
        assert bus["sends"] == {"slap": -10.0}
        (ret,) = state["returns"]
        assert ret["return_id"] == "slap"
        assert ret["sources"] == {"vocals": -10.0}
        assert [entry["type"] for entry in ret["applied_chain"]] == [
            "gain", "return_delay"]
