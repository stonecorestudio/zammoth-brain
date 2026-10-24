"""Phase 19 (docs/05-roadmap.md): Ask/Agent modes + the agent's workspace
control surface.

Agent mode skips only the approval click: chat moves still land as
guardrailed, atomic, undoable proposals. The workspace tools (transport,
selection, browser/import) broadcast UI commands or reuse the Phase 18
import path — never `set_stem`, which stays the user's own channel.
Driven with the deterministic ScriptedChat, like test_chat_gateway.py.
"""
import asyncio
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from orchestrator.agents import RuleAgent
from orchestrator.chat import ScriptedChat
from orchestrator.gateway import Gateway, create_app

from test_chat_gateway import recv_until
from test_phase18_browser import write_sample


@pytest.fixture()
def client(demo_project: Path):
    app = create_app(demo_project, agent_factory=RuleAgent,
                     chat_factory=ScriptedChat)
    with TestClient(app) as test_client:
        yield test_client


def test_default_mode_is_ask_and_gates_proposals(client):
    with client.websocket_connect("/ws") as ws:
        snapshot = ws.receive_json()
        assert snapshot["agent_mode"] == "ask"
        ws.send_json({"type": "chat", "text": "make stem_a 2 dB quieter"})
        proposal = recv_until(ws, "proposal")
        gate = recv_until(ws, "awaiting_approval")
        assert gate["proposal_id"] == proposal["proposal_id"]


def test_agent_mode_auto_applies_and_stays_undoable(client):
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "set_agent_mode", "mode": "agent"})
        snapshot = recv_until(ws, "snapshot")
        assert snapshot["agent_mode"] == "agent"

        ws.send_json({"type": "chat", "text": "make stem_a 2 dB quieter"})
        proposal = recv_until(ws, "proposal")
        # applied without any approve message from the client
        status = recv_until(ws, "proposal_status")
        assert status["proposal_id"] == proposal["proposal_id"]
        assert status["status"] == "applied"
        recv_until(ws, "metrics")  # auto-apply re-renders ground truth
        snapshot = recv_until(ws, "snapshot")
        stem_a = next(s for s in snapshot["stems"] if s["stem_id"] == "stem_a")
        assert stem_a["gain_db"] == -2.0
        assert not snapshot["awaiting"]

        # still a first-class proposal: one message undoes it exactly
        ws.send_json({"type": "undo", "proposal_id": proposal["proposal_id"]})
        undone = recv_until(ws, "proposal_status")
        assert undone["status"] == "undone"
        recv_until(ws, "metrics")
        snapshot = recv_until(ws, "snapshot")
        stem_a = next(s for s in snapshot["stems"] if s["stem_id"] == "stem_a")
        assert stem_a["gain_db"] == 0.0


def test_agent_mode_cannot_pass_guardrails(client):
    """Auto-apply skips the approval click, never the validator."""
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "set_agent_mode", "mode": "agent"})
        recv_until(ws, "snapshot")
        ws.send_json({"type": "chat", "text": "make stem_a 20 dB quieter"})
        reply = recv_until(ws, "chat_message")  # user echo
        assert reply["role"] == "user"
        reply = recv_until(ws, "chat_message")
        assert reply["role"] == "assistant"
        assert "rejected" in reply["text"]
    # nothing applied, nothing rendered: a fresh client sees gain untouched
    with client.websocket_connect("/ws") as ws2:
        snapshot = ws2.receive_json()
        stem_a = next(s for s in snapshot["stems"] if s["stem_id"] == "stem_a")
        assert stem_a["gain_db"] == 0.0


def test_set_agent_mode_confirms_instantly(client):
    """2026-06-12 fix: the switch must flip the moment it is clicked — a
    lightweight agent_mode event lands before the full snapshot, which can
    sit behind the engine lock for seconds during a run."""
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "set_agent_mode", "mode": "agent"})
        event = ws.receive_json()  # the very next event, snapshot later
        assert event == {"type": "agent_mode", "mode": "agent"}
        snapshot = recv_until(ws, "snapshot")
        assert snapshot["agent_mode"] == "agent"


@pytest.mark.anyio
async def test_mode_gating_follows_switch_live(anyio_backend):
    """2026-06-12 fix: a run without explicit gating gates per-proposal
    against the *current* Ask/Agent switch — flipping it mid-run takes
    effect on the next proposal instead of being frozen at run start."""
    gateway = Gateway(agent_factory=RuleAgent, chat_factory=ScriptedChat)
    gate = gateway._gate("mode")
    proposal = {"proposal_id": "p1", "rationale": "r", "actions": []}

    gateway.set_agent_mode("agent")
    assert await gate("p1", proposal) is True  # hands-off: no approval wait

    gateway.set_agent_mode("ask")  # mid-"run" flip: next proposal gates
    waiting = asyncio.create_task(gate("p2", proposal))
    await asyncio.sleep(0)  # let the gate park its approval future
    assert "p2" in gateway.pending
    gateway.pending["p2"].set_result(True)  # the user's approve click
    assert await waiting is True


def test_mode_survives_reconnect(client):
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "set_agent_mode", "mode": "agent"})
        recv_until(ws, "snapshot")
    with client.websocket_connect("/ws") as ws2:
        snapshot = ws2.receive_json()
        assert snapshot["agent_mode"] == "agent"


def test_transport_tool_broadcasts_workspace_command(client):
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "chat", "text": "play"})
        command = recv_until(ws, "workspace_command")
        assert command["command"] == "transport"
        assert command["action"] == "play"


def test_select_stem_tool_validates_and_broadcasts(client):
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "chat", "text": "select stem_b"})
        command = recv_until(ws, "workspace_command")
        assert command == {"type": "workspace_command",
                           "command": "select_stem", "stem_id": "stem_b"}
        reply = recv_until(ws, "chat_message")  # assistant confirmation
        assert reply["role"] == "assistant"
        # unknown stems answer with an error in chat, no broadcast crash
        ws.send_json({"type": "chat", "text": "select no_such_stem"})
        echo = recv_until(ws, "chat_message")
        assert echo["role"] == "user"
        reply = recv_until(ws, "chat_message")
        assert reply["role"] == "assistant"
        assert "unknown stem" in reply["text"]


def test_import_sample_tool_lands_a_stem(client, tmp_path):
    sample = write_sample(tmp_path / "samples", "pad_swell.wav")
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "chat", "text": f"import {sample}"})
        for _ in range(30):
            snapshot = recv_until(ws, "snapshot")
            if len(snapshot["stems"]) == 3:
                break
        assert "pad_swell" in {s["stem_id"] for s in snapshot["stems"]}
