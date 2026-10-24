"""Phase 20 — taste: human gate decisions land in runs/decisions.jsonl and
their summary reaches the agent context (docs/05-roadmap.md Phase 20)."""
import json
from pathlib import Path

import anyio
import pytest
from starlette.testclient import TestClient

from orchestrator.agents import RuleAgent
from orchestrator.calibration import (load_decisions, record_decision,
                                      taste_summary)
from orchestrator.chat import ScriptedChat
from orchestrator.gateway import create_app
from orchestrator.loop import policy_gate, run_session

from mcp.shared.memory import create_connected_server_and_client_session
from mix_engine.mcp_server import create_server


def in_memory_engine(project_dir):
    return create_connected_server_and_client_session(create_server(project_dir))


PROPOSAL = {"rationale": "test move",
            "predicted_impact": [],
            "actions": [{"type": "eq", "stem_id": "master",
                         "band": {"freq_hz": 250, "gain_db": -2.0, "q": 1.0,
                                  "shape": "peak"}}]}


def test_record_and_summary(tmp_path: Path):
    runs = tmp_path / "runs"
    for _ in range(3):
        record_decision(runs, "rejected", PROPOSAL, source="run")
    record_decision(runs, "approved",
                    {"rationale": "", "actions": [{"type": "gain",
                                                   "stem_id": "stem_a",
                                                   "delta_db": -2.0}]},
                    source="chat")
    assert len(load_decisions(runs)) == 4
    summary = taste_summary(runs)
    # only the eq kind clears TASTE_MIN_DECISIONS; one gain click does not
    assert summary == [{"action_key": "eq", "approved": 0, "rejected": 3,
                        "undone": 0, "kept": 0}]


def test_cli_manual_gate_records_decision(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *args: "n")
    runs = tmp_path / "runs"
    gate = policy_gate("manual", runs_dir=runs)
    assert anyio.run(gate, "p0001", PROPOSAL) is False
    (decision,) = load_decisions(runs)
    assert decision["decision"] == "rejected"
    assert decision["source"] == "cli"
    assert decision["action_key"] == "eq"


def test_taste_rides_in_pass_context(demo_project: Path):
    runs = demo_project / "runs"
    for _ in range(3):
        record_decision(runs, "rejected", PROPOSAL, source="run")

    seen: list[list] = []

    class CapturingAgent:
        def propose(self, ctx):
            seen.append(ctx.taste)
            return []

        def revise(self, proposal, violations, ctx):
            return None

    run_session(demo_project, "master for streaming", gating="auto",
                agent=CapturingAgent(), connect=in_memory_engine)
    assert seen, "the agent never got a turn"
    assert seen[0] == [{"action_key": "eq", "approved": 0, "rejected": 3,
                        "undone": 0, "kept": 0}]


@pytest.fixture()
def client(demo_project: Path):
    app = create_app(demo_project, agent_factory=RuleAgent,
                     chat_factory=ScriptedChat)
    with TestClient(app) as test_client:
        yield test_client


def recv_until(ws, event_type: str, limit: int = 300) -> dict:
    for _ in range(limit):
        event = ws.receive_json()
        if event["type"] == "error":
            pytest.fail(f"gateway error: {event['message']}")
        if event["type"] == event_type:
            return event
    pytest.fail(f"no '{event_type}' event within {limit} messages")


def test_browser_reject_and_undo_are_recorded(client, demo_project: Path):
    decisions_path = demo_project / "runs" / "decisions.jsonl"
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # snapshot

        # reject one chat proposal -> a "rejected" chat decision
        ws.send_json({"type": "chat", "text": "make stem_b 3 dB louder"})
        proposal = recv_until(ws, "proposal")
        recv_until(ws, "awaiting_approval")
        ws.send_json({"type": "reject", "proposal_id": proposal["proposal_id"]})
        recv_until(ws, "proposal_status")

        # approve another, then undo it -> "approved" + "undone"
        ws.send_json({"type": "chat", "text": "make stem_a 2 dB quieter"})
        proposal = recv_until(ws, "proposal")
        recv_until(ws, "awaiting_approval")
        ws.send_json({"type": "approve", "proposal_id": proposal["proposal_id"]})
        status = recv_until(ws, "proposal_status")
        assert status["status"] == "applied"
        recv_until(ws, "metrics")
        ws.send_json({"type": "undo", "proposal_id": proposal["proposal_id"]})
        status = recv_until(ws, "proposal_status")
        assert status["status"] == "undone"

    decisions = [json.loads(line)
                 for line in decisions_path.read_text().splitlines()]
    assert [(d["decision"], d["source"], d["action_key"]) for d in decisions] \
        == [("rejected", "chat", "gain"), ("approved", "chat", "gain"),
            ("undone", "workspace", "gain")]
