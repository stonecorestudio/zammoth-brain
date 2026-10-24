"""Phase 9 (DAW-style user controls): the user's direct channel.

set_stem mutates immediately with no proposals and no guardrails; mute/exclude
take stems out of the mix; the gateway exposes user_edit / set_reference over
the websocket and post-chain solo audio over HTTP. Agent mutations keep the
proposal lifecycle (mute/exclude also exist as gated agent actions).
"""
import json

import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from starlette.testclient import TestClient

from mix_engine.mcp_server import create_server
from orchestrator.agents import RuleAgent
from orchestrator.gateway import create_app

from test_phase1_acceptance import call


@pytest.mark.anyio
async def test_set_stem_direct_and_unbounded_by_guardrails(demo_project):
    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        # -20 dB in one move: way past the agent's ±6 dB/pass clamp — fine
        # for the user (guardrails bind the agent, not the human).
        state = await call(client, "set_stem",
                           {"stem_id": "stem_b", "gain_db": -20.0, "pan": 0.5})
        assert state["gain_db"] == -20.0 and state["pan"] == 0.5
        assert (await call(client, "get_proposals")) == []  # no proposal record

        stem = await call(client, "analyze_stem", {"stem_id": "stem_b"})
        assert stem["applied_chain"][0] == {"type": "gain", "gain_db": -20.0}

        master = await call(client, "set_stem",
                            {"stem_id": "master", "gain_db": -3.0})
        assert master["gain_db"] == -3.0


@pytest.mark.anyio
async def test_mute_and_exclude_take_stems_out_of_the_mix(demo_project):
    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        full = await call(client, "render_and_analyze")
        full_bytes = (demo_project / full["bounce_path"]).read_bytes()
        await call(client, "set_stem", {"stem_id": "stem_b", "muted": True})
        without_b = await call(client, "render_and_analyze")
        # the bed carries the high end: with it muted, the air band collapses
        assert (demo_project / without_b["bounce_path"]).read_bytes() != full_bytes
        assert (without_b["analysis"]["band_energy"]["air"]
                < full["analysis"]["band_energy"]["air"] - 10)

        # mute the rest -> the master is silence, not a crash
        await call(client, "set_stem", {"stem_id": "stem_a", "excluded": True})
        silent = await call(client, "render_and_analyze")
        assert silent["analysis"]["silent"] is True

        # bring both back -> bit-identical to the original mix
        await call(client, "set_stem", {"stem_id": "stem_b", "muted": False})
        await call(client, "set_stem", {"stem_id": "stem_a", "excluded": False})
        restored = await call(client, "render_and_analyze")
        assert ((demo_project / restored["bounce_path"]).read_bytes()
                == (demo_project / full["bounce_path"]).read_bytes()
                or restored["analysis"] == full["analysis"])


@pytest.mark.anyio
async def test_agent_mute_still_goes_through_proposals(demo_project):
    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        proposal = await call(client, "propose_actions", {"req": {
            "rationale": "drop the bed for the intro edit",
            "predicted_impact": [],
            "actions": [{"type": "mute", "stem_id": "stem_b", "muted": True}]}})
        assert proposal["accepted"], proposal["violations"]
        assert (await call(client, "apply_proposal",
                           {"proposal_id": proposal["proposal_id"]}))["ok"]
        render = await call(client, "render_and_analyze")
        assert (await call(client, "undo_proposal",
                           {"proposal_id": proposal["proposal_id"]}))["ok"]


# ---------- gateway surface ----------

@pytest.fixture()
def ui(demo_project):
    app = create_app(demo_project, agent_factory=RuleAgent)
    with TestClient(app) as client:
        yield client, demo_project


def recv_until(ws, event_type: str, limit: int = 100) -> dict:
    for _ in range(limit):
        event = ws.receive_json()
        if event["type"] == event_type:
            return event
    raise AssertionError(f"never saw {event_type}")


def test_gateway_user_edit_roundtrip(ui):
    client, project = ui
    with client.websocket_connect("/ws") as ws:
        snapshot = recv_until(ws, "snapshot")
        assert snapshot["stems"][0]["muted"] is False
        assert snapshot["sections"] is not None  # sections ride the snapshot

        ws.send_json({"type": "user_edit", "stem_id": "stem_a",
                      "changes": {"muted": True, "gain_db": -4.0}})
        recv_until(ws, "metrics")  # the edit re-rendered
        snapshot = recv_until(ws, "snapshot")
        (stem_a,) = [s for s in snapshot["stems"] if s["stem_id"] == "stem_a"]
        assert stem_a["muted"] is True and stem_a["gain_db"] == -4.0


def test_gateway_drop_import_adds_stem(ui):
    client, project = ui
    wav_bytes = (project / "stem_a.wav").read_bytes()
    response = client.put("/api/stems/dropped_loop.wav", content=wav_bytes)
    assert response.status_code == 200
    assert response.json()["stem_id"] == "dropped_loop"
    with client.websocket_connect("/ws") as ws:
        snapshot = recv_until(ws, "snapshot")
        ids = [s["stem_id"] for s in snapshot["stems"]]
        assert "dropped_loop" in ids
        assert all(s["duration_s"] > 0 for s in snapshot["stems"])
    # junk is refused
    assert client.put("/api/stems/notes.txt", content=b"hi").status_code == 400
    assert client.put("/api/stems/fake.wav", content=b"not riff").status_code == 400


def test_gateway_set_reference_and_solo_audio(ui):
    client, project = ui
    # a reference: any wav outside the stem glob
    (project / "refs").mkdir()
    import shutil
    shutil.copy(project / "stem_a.wav", project / "refs" / "reference.wav")

    with client.websocket_connect("/ws") as ws:
        recv_until(ws, "snapshot")
        ws.send_json({"type": "set_reference", "path": "refs/reference.wav"})
        snapshot = recv_until(ws, "snapshot")
        assert snapshot["reference"]["band_energy"] is not None

    raw = client.get("/api/audio/stem/stem_a")
    solo = client.get("/api/audio/stem/stem_a?processed=1")
    assert raw.status_code == solo.status_code == 200
    assert solo.headers["content-type"] == "audio/wav"
    assert len(solo.content) > 1000
