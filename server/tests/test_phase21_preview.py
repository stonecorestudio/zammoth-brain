"""Phase 21 — sample preview endpoints (docs/05-roadmap.md): preview audio
re-encodes any wav as decodable float32, peaks share the clip-canvas bucket
math, and bad paths fail honestly."""
import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from starlette.testclient import TestClient

from orchestrator.agents import RuleAgent
from orchestrator.chat import ScriptedChat
from orchestrator.gateway import create_app

SR = 44100


@pytest.fixture()
def client(demo_project: Path):
    app = create_app(demo_project, agent_factory=RuleAgent,
                     chat_factory=ScriptedChat)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def sample_24bit(tmp_path: Path) -> Path:
    """A 24-bit PCM one-shot outside any project — the case WebKit cannot
    decode natively, which is why the endpoint re-encodes."""
    t = np.arange(int(SR * 0.5)) / SR
    tone = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    path = tmp_path / "oneshot.wav"
    sf.write(path, tone, SR, subtype="PCM_24")
    return path


def test_preview_audio_reencodes_to_float32(client, sample_24bit: Path):
    res = client.get(f"/api/preview/audio?path={sample_24bit}")
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav"
    data, sample_rate = sf.read(io.BytesIO(res.content), always_2d=True)
    assert sample_rate == SR
    assert len(data) == int(SR * 0.5)  # same frames, new encoding
    info = sf.info(io.BytesIO(res.content))
    assert info.subtype == "FLOAT"


def test_preview_peaks_match_request(client, sample_24bit: Path):
    res = client.get(f"/api/preview/peaks?path={sample_24bit}&buckets=300")
    assert res.status_code == 200
    body = res.json()
    assert body["duration_s"] == 0.5
    assert len(body["top"]) == len(body["bottom"]) == 300
    assert all(0.0 <= v <= 1.0 for v in body["top"])
    assert all(-1.0 <= v <= 0.0 for v in body["bottom"])
    # a full-scale-normalized sine fills the canvas
    assert max(body["top"]) == 1.0


def test_preview_rejects_non_wav(client, tmp_path: Path):
    other = tmp_path / "notes.txt"
    other.write_text("not audio")
    assert client.get(f"/api/preview/audio?path={other}").status_code == 400
    assert client.get(f"/api/preview/peaks?path={other}").status_code == 400


def test_preview_404_on_missing_file(client, tmp_path: Path):
    missing = tmp_path / "ghost.wav"
    assert client.get(f"/api/preview/audio?path={missing}").status_code == 404
    assert client.get(f"/api/preview/peaks?path={missing}").status_code == 404
