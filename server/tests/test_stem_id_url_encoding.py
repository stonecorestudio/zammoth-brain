"""Stem ids carry musical-key suffixes now (Phase 30/31), so a perfectly
ordinary stem can be named like ``Under_The_Blanket_..._F#``. The ``#`` is a
URL fragment delimiter: interpolated raw into a path it truncates the request,
so the clip waveform (``/api/audio/peaks/{stem_id}``) and Solo source
(``/api/audio/stem/{stem_id}``) silently 404 — a blank clip and a Solo that
keeps playing the full mix. The fix is to percent-encode the id everywhere it
enters a URL path; the snapshot's ``audio_url`` must arrive already encoded.
"""
from urllib.parse import quote

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


def _import_sharp_stem(client, tmp_path) -> str:
    """Import a stem whose name ends in a sharp key (the '#' is the bug
    trigger) and return its stem id once it shows up in a snapshot."""
    folder = tmp_path / "samples"
    folder.mkdir(parents=True, exist_ok=True)
    name = "Under_The_Blanket_90BPM_F#.wav"
    t = np.arange(int(1.0 * 48000)) / 48000
    sf.write(folder / name,
             (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32), 48000)
    stem_id = "Under_The_Blanket_90BPM_F#"

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        assert client.post("/api/import",
                           params={"path": str(folder / name)}).status_code == 200
        for _ in range(20):
            snapshot = recv_until(ws, "snapshot")
            stem = next((s for s in snapshot["stems"]
                         if s["stem_id"] == stem_id), None)
            if stem is not None:
                # the URL the webview fetches must already be encoded ('#' →
                # '%23'), or it truncates the moment the browser parses it
                assert stem["audio_url"] == f"/api/audio/stem/{quote(stem_id, safe='')}"
                return stem_id
    raise AssertionError("imported sharp-named stem never appeared in a snapshot")


def test_peaks_and_solo_serve_a_sharp_named_stem(client, tmp_path):
    stem_id = _import_sharp_stem(client, tmp_path)
    encoded = quote(stem_id, safe="")  # what encodeURIComponent produces too
    assert "%23" in encoded  # the '#' really is being escaped

    # clip waveform: encoded path serves real peaks (a drawable waveform)
    peaks = client.get(f"/api/audio/peaks/{encoded}")
    assert peaks.status_code == 200, peaks.text
    body = peaks.json()
    assert len(body["top"]) == len(body["bottom"]) > 0
    assert any(v != 0 for v in body["top"])  # not a flat (blank) clip

    # Solo source: encoded path renders the isolated stem's processed chain
    solo = client.get(f"/api/audio/stem/{encoded}", params={"processed": 1})
    assert solo.status_code == 200, solo.text
    assert solo.headers["content-type"] == "audio/wav"

    # the raw id is exactly the old, broken request: a client parses '#' as a
    # fragment, so the server only ever sees ".../Under_The_Blanket_90BPM_F" —
    # an unknown stem. This is why the waveform and Solo failed.
    truncated = client.get(f"/api/audio/peaks/{stem_id}")
    assert truncated.status_code == 404
