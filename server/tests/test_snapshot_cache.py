"""Snapshot performance: per-stem analysis is cached on render-relevant
state and section detection re-runs only when the mix's *structure* changes.
A fader commit re-analyzes exactly one stem and leaves sections alone, so
snapshot cost scales with what changed — not with stem count."""
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


def recv_until(ws, event_type: str, limit: int = 100) -> dict:
    for _ in range(limit):
        event = ws.receive_json()
        if event["type"] == "error":
            pytest.fail(f"gateway error: {event['message']}")
        if event["type"] == event_type:
            return event
    pytest.fail(f"no '{event_type}' event within {limit} messages")


def test_snapshot_reuses_cached_analysis(client):
    gateway = client.app.state.gateway
    calls: list[tuple[str, dict]] = []
    inner = gateway.engine.call

    async def counting(name, args=None):
        calls.append((name, args or {}))
        return await inner(name, args)

    gateway.engine.call = counting

    def args_for(name: str) -> list[dict]:
        return [args for called, args in calls if called == name]

    # cold caches: every stem analyzed once, sections computed once
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "snapshot"
    assert len(args_for("analyze_stem")) == 2
    assert len(args_for("analyze_sections")) == 1

    # reconnect: warm caches — the snapshot triggers no renders at all
    calls.clear()
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "snapshot"
    assert args_for("analyze_stem") == []
    assert args_for("analyze_sections") == []
    assert args_for("analyze_session") == []

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # connect snapshot (cached)

        # fader commit: an instant confirmation snapshot first — state has
        # moved but nothing re-measures; the coalesced background render
        # then re-analyzes exactly the moved stem. Gain moves don't shift
        # section boundaries, so sections are never recomputed.
        calls.clear()
        ws.send_json({"type": "user_edit", "stem_id": "stem_a",
                      "changes": {"gain_db": -3.0}})
        confirm = recv_until(ws, "snapshot")
        stem_a = next(s for s in confirm["stems"] if s["stem_id"] == "stem_a")
        assert stem_a["gain_db"] == -3.0  # confirmed before any render
        assert confirm["render_pending"] is True
        assert args_for("analyze_stem") == []
        recv_until(ws, "metrics")  # the background render landing
        fresh = recv_until(ws, "snapshot")
        assert fresh["render_pending"] is False
        assert args_for("analyze_stem") == [{"stem_id": "stem_a"}]
        assert args_for("analyze_sections") == []

        # mute changes the audible structure: sections recompute (on the
        # post-render snapshot — the confirmation serves the old
        # boundaries), but no per-stem render (render_stem ignores mute)
        calls.clear()
        ws.send_json({"type": "user_edit", "stem_id": "stem_b",
                      "changes": {"muted": True}})
        recv_until(ws, "metrics")
        snapshot = recv_until(ws, "snapshot")
        assert args_for("analyze_stem") == []
        assert len(args_for("analyze_sections")) == 1

        # cached values still reflect engine ground truth
        stem_a = next(s for s in snapshot["stems"] if s["stem_id"] == "stem_a")
        assert stem_a["gain_db"] == -3.0
