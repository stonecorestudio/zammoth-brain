"""Phase 3 acceptance (docs/05-roadmap.md): a user can run a full mix pass on
the demo project entirely from the browser — approving/undoing individual
proposals — through the orchestrator's websocket gateway."""
import json
from pathlib import Path

import pytest
import soundfile as sf
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from orchestrator.agents import RuleAgent
from orchestrator.gateway import HOST, create_app


@pytest.fixture()
def masked_project(demo_project: Path) -> Path:
    """Demo project with the 'vocal' pulled down 30 dB so the noise bed
    genuinely masks it (initial worst-band excess ~ +6 dB, target <= 3 dB)."""
    data, sample_rate = sf.read(demo_project / "stem_a.wav", dtype="float32")
    sf.write(demo_project / "stem_a.wav", data * 10 ** (-30 / 20), sample_rate)
    return demo_project


@pytest.fixture()
def client(masked_project: Path):
    app = create_app(masked_project, agent_factory=RuleAgent)
    with TestClient(app) as test_client:
        yield test_client


def recv_until(ws, event_type: str, limit: int = 300) -> dict:
    """Read events until one of `event_type` arrives (snapshots etc. interleave)."""
    for _ in range(limit):
        event = ws.receive_json()
        if event["type"] == event_type:
            return event
    pytest.fail(f"no '{event_type}' event within {limit} messages")


def drive_run(ws, decision: str) -> tuple[dict, list[str]]:
    """Approve or reject every gated proposal until the run finishes."""
    applied: list[str] = []
    while True:
        event = ws.receive_json()
        if event["type"] == "awaiting_approval":
            ws.send_json({"type": decision, "proposal_id": event["proposal_id"]})
        elif event["type"] == "proposal_status" and event["status"] == "applied":
            applied.append(event["proposal_id"])
        elif event["type"] == "run_finished":
            return event, applied
        elif event["type"] == "error":
            pytest.fail(f"gateway error: {event['message']}")


def test_full_pass_from_browser_with_approvals_and_undo(client):
    with client.websocket_connect("/ws") as ws:
        snapshot = ws.receive_json()
        assert snapshot["type"] == "snapshot"
        assert snapshot["project"]["stem_count"] == 2
        assert snapshot["project"]["sample_rate"] > 0
        assert {s["stem_id"] for s in snapshot["stems"]} == {"stem_a", "stem_b"}
        assert snapshot["limits"]["gain_per_stem_per_pass_db"] == 6.0

        ws.send_json({"type": "start_run", "brief": "stem A clarity",
                      "gating": "manual"})
        finished, applied = drive_run(ws, "approve")

        assert finished["stop_reason"] == "targets met"
        assert finished["passes_run"] <= 5
        assert applied, "manual approvals should have applied proposals"

        # final snapshot: engine ground truth with measured impact recorded
        snapshot = recv_until(ws, "snapshot")
        assert snapshot["run"]["running"] is False
        by_id = {p["proposal_id"]: p for p in snapshot["proposals"]}
        for proposal_id in applied:
            assert by_id[proposal_id]["status"] == "applied"
            assert by_id[proposal_id]["measured_impact"], proposal_id
        stem_b = next(s for s in snapshot["stems"] if s["stem_id"] == "stem_b")
        gain_after_run = stem_b["gain_db"]
        assert gain_after_run < 0  # the masker was pulled down

        # out-of-order undo of a non-commuting proposal is refused (docs/03):
        # an older proposal can't be undone while a newer one touches the
        # same stem
        if len(applied) > 1:
            ws.send_json({"type": "undo", "proposal_id": applied[0]})
            error = recv_until(ws, "error")
            assert "cannot undo" in error["message"]

        # undo the most recent applied proposal from the browser
        ws.send_json({"type": "undo", "proposal_id": applied[-1]})
        status = recv_until(ws, "proposal_status")
        assert status == {"type": "proposal_status", "proposal_id": applied[-1],
                          "status": "undone", "message": ""}
        snapshot = recv_until(ws, "snapshot")
        by_id = {p["proposal_id"]: p for p in snapshot["proposals"]}
        assert by_id[applied[-1]]["status"] == "undone"
        stem_b = next(s for s in snapshot["stems"] if s["stem_id"] == "stem_b")
        assert stem_b["gain_db"] > gain_after_run  # the gain move was reverted


def test_rejecting_every_proposal_applies_nothing(client):
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "snapshot"
        ws.send_json({"type": "start_run", "brief": "stem A clarity",
                      "gating": "manual"})
        finished, applied = drive_run(ws, "reject")

        assert applied == []
        assert finished["stop_reason"] in ("diminishing returns",
                                           "pass budget exhausted")
        snapshot = recv_until(ws, "snapshot")
        assert all(p["status"] != "applied" for p in snapshot["proposals"])
        assert all(s["gain_db"] == 0.0 for s in snapshot["stems"])


def test_instruction_reaches_feed_and_run_context(client):
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "snapshot"
        ws.send_json({"type": "instruction", "text": "keep the noise bed dark"})
        echoed = recv_until(ws, "user_instruction")
        assert echoed["text"] == "keep the noise bed dark"


def test_audio_and_fs_endpoints(client, masked_project: Path):
    response = client.get("/api/audio/stem/stem_a")
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert client.get("/api/audio/stem/nope").status_code == 404

    listing = client.get("/api/fs", params={"path": str(masked_project.parent)})
    assert listing.status_code == 200
    assert "two_stem_demo" in listing.json()["folders"]
    assert client.get("/api/fs", params={"path": "/nope/nothing"}).status_code == 404


def test_gateway_is_localhost_only():
    # CLAUDE.md: the gateway binds 127.0.0.1, no auth layer. Since Phase 49
    # (the webview shells archived) the invariant HARDENED: no browser origin
    # may script the local engine. Until 2026-09-01 this test only asserted
    # that no CORS header came back — which governs whether a page can READ a
    # response, not whether the request RUNS: WebSockets ignore CORS entirely
    # and a no-preflight POST executes before any CORS check, so any web page
    # could drive /ws, /api/chat, /api/export. The door is now held shut by
    # refusing every request that carries an Origin header (a browser sends one
    # on every cross-origin request and every POST; the native VST never does),
    # and this test drives that door. A SAME-origin GET carries none - which is
    # what a DNS-rebinding page makes - so the gateway also refuses a request
    # whose Host is not this machine's (test_reader_p3_py8.py, reader S9·R16).
    assert HOST == "127.0.0.1"
    app = create_app()
    with TestClient(app) as test_client:
        for origin in ("tauri://localhost", "http://localhost:5173",
                       "https://evil.example"):
            assert test_client.get("/api/health",
                                   headers={"Origin": origin}).status_code == 403, origin
            assert test_client.post("/api/agent_mode", params={"mode": "agent"},
                                    headers={"Origin": origin}).status_code == 403, origin
            with pytest.raises(WebSocketDisconnect):
                with test_client.websocket_connect("/ws", headers={"Origin": origin}):
                    pass
        # the native (origin-less) path is untouched
        assert test_client.get("/api/health").status_code == 200


def test_empty_project_opens_with_null_metrics(tmp_path: Path):
    """Startup without a folder pick: the desktop shell may hand the sidecar a
    brand-new empty folder, and the workspace must come up (zero stems, null
    metrics) instead of the picker."""
    app = create_app(tmp_path / "untitled")
    (tmp_path / "untitled").mkdir()
    with TestClient(app) as test_client:
        with test_client.websocket_connect("/ws") as ws:
            snapshot = ws.receive_json()
            assert snapshot["type"] == "snapshot"
            assert snapshot["project"]["stem_count"] == 0
            assert snapshot["project"]["sample_rate"] is None
            assert snapshot["stems"] == []
            assert snapshot["metrics"]["silent"] is True
            assert snapshot["metrics"]["lufs_i"] is None
            assert snapshot["sections"] is None


def test_bad_startup_project_falls_back_to_no_project(tmp_path: Path):
    # A deleted/invalid startup folder must not crash the app (the sidecar
    # would otherwise crash-loop); the UI gets no_project and shows the picker.
    app = create_app(tmp_path / "gone")
    with TestClient(app) as test_client:
        with test_client.websocket_connect("/ws") as ws:
            assert ws.receive_json()["type"] == "no_project"


def test_new_project_creates_and_opens_untitled_folder(client, tmp_path: Path):
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "snapshot"
        ws.send_json({"type": "new_project", "base_dir": str(tmp_path)})
        opened = recv_until(ws, "project_opened")
        assert opened["path"] == str(tmp_path / "Untitled Project")
        snapshot = recv_until(ws, "snapshot")
        assert snapshot["project"]["name"] == "Untitled Project"
        assert snapshot["project"]["stem_count"] == 0
        # a second New picks the next free name instead of colliding
        ws.send_json({"type": "new_project", "base_dir": str(tmp_path)})
        assert recv_until(ws, "project_opened")["path"] == \
            str(tmp_path / "Untitled Project 2")


def test_save_project_as_copies_folder_and_switches(client, tmp_path: Path,
                                                    masked_project: Path):
    target = tmp_path / "saved_copy"
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "snapshot"
        ws.send_json({"type": "save_project_as", "path": str(target)})
        assert recv_until(ws, "project_opened")["path"] == str(target)
        snapshot = recv_until(ws, "snapshot")
        assert snapshot["project"]["name"] == "saved_copy"
        assert snapshot["project"]["stem_count"] == 2
    # full copy: stems and session state, source left in place
    assert (target / "stem_a.wav").exists()
    assert (target / "session.json").exists()
    assert (masked_project / "stem_a.wav").exists()
    # saving onto an existing folder is refused
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "snapshot"
        ws.send_json({"type": "save_project_as", "path": str(target)})
        error = recv_until(ws, "error")
        assert "already exists" in error["message"]


def test_stem_color_is_ui_state_not_engine_state(client, masked_project: Path):
    """Per-stem display colors (Ableton-style picker) live in ui.json and ride
    the snapshot; they never enter session.json or the engine's state."""
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "snapshot"
        ws.send_json({"type": "user_edit", "stem_id": "stem_a",
                      "changes": {"color": "#ff9f4f"}})
        snapshot = recv_until(ws, "snapshot")
        by_id = {s["stem_id"]: s for s in snapshot["stems"]}
        assert by_id["stem_a"]["color"] == "#ff9f4f"
        assert by_id["stem_b"]["color"] is None

        # invalid colors are rejected; bad stem ids too
        ws.send_json({"type": "user_edit", "stem_id": "stem_a",
                      "changes": {"color": "tomato"}})
        assert "rrggbb" in recv_until(ws, "error")["message"]
        ws.send_json({"type": "user_edit", "stem_id": "nope",
                      "changes": {"color": "#ff9f4f"}})
        assert "unknown stem" in recv_until(ws, "error")["message"]

        # null resets to automatic
        ws.send_json({"type": "user_edit", "stem_id": "stem_a",
                      "changes": {"color": None}})
        snapshot = recv_until(ws, "snapshot")
        assert {s["stem_id"]: s for s in snapshot["stems"]}["stem_a"]["color"] is None

    ui = json.loads((masked_project / "ui.json").read_text())
    assert ui["stem_colors"] == {}
    session = json.loads((masked_project / "session.json").read_text())
    assert "color" not in session["stems"]["stem_a"]


def test_peaks_endpoint_serves_normalized_waveform(client):
    """Clip waveforms come from server-computed peaks (the desktop webview
    cannot decode every wav format) — normalized so the loudest bucket hits
    full scale regardless of the stem's level."""
    response = client.get("/api/audio/peaks/stem_a", params={"buckets": 500})
    assert response.status_code == 200
    body = response.json()
    assert len(body["top"]) == 500 and len(body["bottom"]) == 500
    assert body["duration_s"] > 0
    assert max(body["top"]) == pytest.approx(1.0, abs=0.01)  # per-stem normalize
    assert all(v >= 0 for v in body["top"]) and all(v <= 0 for v in body["bottom"])
    assert client.get("/api/audio/peaks/nope").status_code == 404
