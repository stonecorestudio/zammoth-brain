"""Phase 18 (docs/05-roadmap.md): sample browser endpoints.

The browser panel is only as real as its backend: /api/fs lists the wav
files next to the folders, and /api/import copies a local wav into the
project in place (refresh + auto-tag + render) with honest failure modes.
"""
import shutil

import numpy as np
import pytest
import soundfile as sf
from starlette.testclient import TestClient

from orchestrator.agents import RuleAgent
from orchestrator.chat import ScriptedChat
from orchestrator.gateway import create_app

from test_phase16_automix import recv_until


@pytest.fixture()
def client(demo_project):
    app = create_app(demo_project, agent_factory=RuleAgent,
                     chat_factory=ScriptedChat)
    with TestClient(app) as test_client:
        yield test_client


def write_sample(folder, name="snare_tail.wav") -> str:
    folder.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(2.0 * 48000)) / 48000
    tone = 0.3 * np.sin(2 * np.pi * 220.0 * t) * np.exp(-t)
    sf.write(folder / name, tone.astype(np.float32), 48000)
    return str(folder / name)


def test_fs_lists_wavs_alongside_folders(client, tmp_path):
    write_sample(tmp_path / "samples")
    (tmp_path / "samples" / "subfolder").mkdir()
    (tmp_path / "samples" / "notes.txt").write_text("not audio")
    listing = client.get("/api/fs", params={"path": str(tmp_path / "samples")})
    assert listing.status_code == 200
    body = listing.json()
    assert body["folders"] == ["subfolder"]
    assert body["audio"] == ["snare_tail.wav"]


def test_import_by_path_adds_a_tagged_stem(client, tmp_path):
    sample = write_sample(tmp_path / "samples", "pad_swell.wav")
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        response = client.post("/api/import", params={"path": sample})
        assert response.status_code == 200, response.text
        assert response.json()["stem_id"] == "pad_swell"
        for _ in range(20):
            snapshot = recv_until(ws, "snapshot")
            if len(snapshot["stems"]) == 3:
                break
        ids = {s["stem_id"] for s in snapshot["stems"]}
        assert "pad_swell" in ids

    # honest failure modes
    duplicate = client.post("/api/import", params={"path": sample})
    assert duplicate.status_code == 409
    not_wav = tmp_path / "samples" / "notes.txt"
    not_wav.write_text("x")
    assert client.post("/api/import",
                       params={"path": str(not_wav)}).status_code == 400


def test_import_rejects_existing_project_stems(client, demo_project):
    inside = demo_project / "stem_a.wav"
    response = client.post("/api/import", params={"path": str(inside)})
    assert response.status_code == 400
    assert "already a project stem" in response.text


def test_reimport_restores_a_removed_stem(client, tmp_path):
    """✕ removes a stem (excluded) but its file stays in the project, so a
    second import of the same wav must restore it — not 409 into a dead end
    where the stem is hidden everywhere and the name is taken forever."""
    sample = write_sample(tmp_path / "samples", "pad_swell.wav")
    assert client.post("/api/import",
                       params={"path": sample}).status_code == 200
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "user_edit", "stem_id": "pad_swell",
                      "changes": {"excluded": True}})
        recv_until(ws, "snapshot")

        response = client.post("/api/import", params={"path": sample})
        assert response.status_code == 200, response.text
        assert response.json() == {"ok": True, "stem_id": "pad_swell",
                                   "restored": True}
        for _ in range(20):
            snapshot = recv_until(ws, "snapshot")
            stem = next(s for s in snapshot["stems"]
                        if s["stem_id"] == "pad_swell")
            if not stem["excluded"]:
                break
        assert not stem["excluded"]

    # a stem actually in the mix still collides honestly
    duplicate = client.post("/api/import", params={"path": sample})
    assert duplicate.status_code == 409
    assert "already in the mix" in duplicate.text
