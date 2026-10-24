"""Conversational agent layer over the gateway (orchestrator/chat.py):
a chat message produces a streamed reply in the feed, chat-requested moves
flow through the proposal lifecycle (guardrails + user approval), and the
chat can start the pass loop. Driven with the deterministic ScriptedChat —
the LLM ChatSession shares the same ChatTools surface."""
import asyncio
import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from orchestrator.agents import RuleAgent
from orchestrator.chat import ScriptedChat
from orchestrator.gateway import create_app


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


def test_chat_answers_questions_with_metrics(client):
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # snapshot
        ws.send_json({"type": "chat", "text": "how does the mix look?"})
        echo = recv_until(ws, "chat_message")
        assert echo["role"] == "user"
        reply = recv_until(ws, "chat_message")
        assert reply["role"] == "assistant"
        assert "LUFS" in reply["text"]  # grounded in analyze_session numbers


def test_chat_proposal_is_gated_then_applies(client, demo_project: Path):
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "chat", "text": "make stem_a 2 dB quieter"})

        tool = recv_until(ws, "chat_tool")
        assert tool["name"] == "propose"
        proposal = recv_until(ws, "proposal")
        assert proposal["actions"] == [
            {"type": "gain", "stem_id": "stem_a", "delta_db": -2.0}]
        gate = recv_until(ws, "awaiting_approval")
        assert gate["proposal_id"] == proposal["proposal_id"]

        # a reconnecting client sees the pending chat proposal in the snapshot
        with client.websocket_connect("/ws") as ws2:
            snapshot = ws2.receive_json()
            assert proposal["proposal_id"] in snapshot["awaiting"]

        ws.send_json({"type": "approve", "proposal_id": proposal["proposal_id"]})
        status = recv_until(ws, "proposal_status")
        assert status["status"] == "applied"
        recv_until(ws, "metrics")  # approval re-renders ground truth

        snapshot = recv_until(ws, "snapshot")
        stem_a = next(s for s in snapshot["stems"] if s["stem_id"] == "stem_a")
        assert stem_a["gain_db"] == -2.0
        assert not snapshot["awaiting"]

        # the engine recorded it like any other proposal — and the chat log
        # persisted the conversation next to the run logs
        by_id = {p["proposal_id"]: p for p in snapshot["proposals"]}
        assert by_id[proposal["proposal_id"]]["status"] == "applied"
        chat_log = demo_project / "runs" / "chat.jsonl"
        events = [json.loads(line) for line in chat_log.read_text().splitlines()]
        assert any(e["type"] == "chat_message" and e["role"] == "user"
                   for e in events)


def test_chat_rejection_skips_the_move(client):
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "chat", "text": "make stem_b 3 dB louder"})
        proposal = recv_until(ws, "proposal")
        recv_until(ws, "awaiting_approval")
        ws.send_json({"type": "reject", "proposal_id": proposal["proposal_id"]})
        status = recv_until(ws, "proposal_status")
        assert status["status"] == "skipped"


def test_chat_busy_flag_brackets_the_turn(client):
    """The 'thinking…' indicator is server-authoritative: a chat_busy=true the
    instant a turn starts, a chat_busy=false the instant it ends, and the
    snapshot echoes the resting value for reconnects."""
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # snapshot — chat_busy false at rest
        ws.send_json({"type": "chat", "text": "how does the mix look?"})
        assert recv_until(ws, "chat_busy")["busy"] is True
        assert recv_until(ws, "chat_busy")["busy"] is False  # turn ended
    with client.websocket_connect("/ws") as ws2:
        assert ws2.receive_json()["chat_busy"] is False


def test_chat_busy_clears_when_the_turn_kicks_off_auto_mix(client):
    """The exact stuck-'thinking…' bug: a turn that starts Auto Mix finishes
    long before the run does, and the run then floods the feed. The flag is
    tied to the turn (not 'last feed entry is an assistant message'), so it
    clears even though automix events keep streaming after it."""
    client.app.state.gateway.agent_mode = "agent"   # Agent mode = the auto path
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "chat", "text": "auto mix for streaming"})
        assert recv_until(ws, "chat_busy")["busy"] is True
        assert recv_until(ws, "chat_busy")["busy"] is False  # turn done early
        recv_until(ws, "automix_finished")  # the run lands afterwards
    with client.websocket_connect("/ws") as ws2:
        assert ws2.receive_json()["chat_busy"] is False  # never stuck on


class _BlockingChat(ScriptedChat):
    """A turn that never completes on its own — stands in for an LLM stream
    that hangs (or a turn that left 'thinking…' with nothing to clear it)."""

    async def turn(self, text, *, tools, context, emit, during_run=False,
                   should_cancel=None, live_context=None):
        await asyncio.Event().wait()  # only cancellation ends this


def test_stop_chat_aborts_a_stuck_turn(demo_project: Path):
    """The escape hatch: a hung engineer turn is cancellable from the UI, and
    cancelling clears the working indicator — no app restart needed."""
    app = create_app(demo_project, agent_factory=RuleAgent,
                     chat_factory=_BlockingChat)
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "chat", "text": "think about this forever"})
            assert recv_until(ws, "chat_busy")["busy"] is True
            ws.send_json({"type": "stop_chat"})
            assert recv_until(ws, "chat_busy")["busy"] is False


def test_chat_starts_and_stops_the_run(client):
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "chat", "text": "master to -30 LUFS please"})
        recv_until(ws, "run_started")
        recv_until(ws, "awaiting_approval")  # gated limiter proposal
        # Phase 17: the chat stays available mid-run — a directive ask files
        # as an instruction for the loop (questions get answered instead;
        # covered in test_phase17_chat.py)
        ws.send_json({"type": "chat", "text": "cut it gently, 2 dB at most"})
        note = recv_until(ws, "user_instruction")
        assert note["text"] == "cut it gently, 2 dB at most"
        ws.send_json({"type": "stop_run"})
        finished = recv_until(ws, "run_finished")
        assert finished["stop_reason"] == "stopped"
