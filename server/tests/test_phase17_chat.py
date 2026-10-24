"""Phase 17 (docs/05-roadmap.md): the resident engineer chat.

Covers: the chat answering a status question MID-RUN (no more going mute)
while directive asks file as instructions; a seeded true-peak trigger
producing exactly one proactive observation (dedupe + cooldown); per-project
notes surviving a gateway restart; the chat triggering the Auto Mix flow;
and workspace state + recent user edits reaching the chat context.
"""
import json

import pytest
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


# ---------- available during runs ----------

def test_chat_answers_status_mid_run_and_files_directives(client):
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # snapshot
        # manual gating blocks the run on its first approval: a live run
        ws.send_json({"type": "start_run", "brief": "master for streaming",
                      "gating": "manual", "max_passes": 2})
        gate = recv_until(ws, "awaiting_approval")

        # a question gets an ANSWER mid-run, not a silent instruction
        ws.send_json({"type": "chat", "text": "how is it going?"})
        echo = recv_until(ws, "chat_message")
        assert echo["role"] == "user"
        reply = recv_until(ws, "chat_message")
        assert reply["role"] == "assistant"
        assert "master for streaming" in reply["text"]
        assert "pass" in reply["text"]

        # a directive ask files as an instruction for the loop
        ws.send_json({"type": "chat", "text": "make stem_a 2 db quieter"})
        recv_until(ws, "chat_message")  # user echo
        instruction = recv_until(ws, "user_instruction")
        assert "stem_a" in instruction["text"]
        ack = recv_until(ws, "chat_message")
        assert "filed" in ack["text"].lower()

        # release the gate so the run finishes cleanly
        ws.send_json({"type": "approve", "proposal_id": gate["proposal_id"]})
        recv_until(ws, "run_finished")


# ---------- proactive observations ----------

@pytest.fixture()
def hot_client(demo_project):
    """The demo with both stems already at the fader's top (+6): the master's own +6 then puts
    the mix 12 dB up. A fader stops at +6 since 2026-10-05 (reader E1.R4), so the master alone
    can no longer be pushed to +12 - and this test waited for ever on a level it never reached."""
    from mix_engine.session import Session
    session = Session(demo_project)
    for stem in list(session.stems):
        session.user_edit(stem, {"gain_db": 6.0})
    app = create_app(demo_project, agent_factory=RuleAgent,
                     chat_factory=ScriptedChat)
    with TestClient(app) as test_client:
        yield test_client


def test_true_peak_trigger_observes_exactly_once(hot_client):
    client = hot_client
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        # push the master hot enough to blow the -1.0 dBTP ceiling
        ws.send_json({"type": "user_edit", "stem_id": "master",
                      "changes": {"gain_db": 6.0}})
        observation = recv_until(ws, "chat_message")
        assert observation.get("observation") is True
        assert "true peak" in observation["text"].lower()

        # same trigger again: it never repeats itself
        ws.send_json({"type": "user_edit", "stem_id": "master",
                      "changes": {"gain_db": 5.0}})
        recv_until(ws, "snapshot")  # the edit's re-render completes
        ws.send_json({"type": "user_edit", "stem_id": "master",
                      "changes": {"gain_db": 6.0}})
        snapshot = recv_until(ws, "snapshot")
        observations = [e for e in snapshot["feed"]
                        if e.get("observation")]
        assert len(observations) == 1


def test_user_balance_move_draws_one_coedit_reaction(client):
    """Phase 40 co-edit: a manual stem fader move draws ONE proactive reaction —
    the agent reacting to what the user did by hand — and the shared cooldown
    holds a burst of moves to a single comment (never a per-move LLM call)."""
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # snapshot
        ws.send_json({"type": "user_edit", "stem_id": "stem_a",
                      "changes": {"gain_db": -4.0}})
        observation = recv_until(ws, "chat_message")
        assert observation.get("observation") is True
        assert "stem_a" in observation["text"]

        # a second hand move immediately after: the cooldown keeps it to one
        ws.send_json({"type": "user_edit", "stem_id": "stem_b",
                      "changes": {"gain_db": -3.0}})
        snapshot = recv_until(ws, "snapshot")
        observations = [e for e in snapshot["feed"] if e.get("observation")]
        assert len(observations) == 1


def test_master_device_coedit_draws_reaction(client):
    """Phase 40 co-edit: a manual device-knob move (the native editor's
    /api/chain_param -> set_master_param) draws one gated reaction, same shared
    path as a fader move."""
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # snapshot
        resp = client.post("/api/chain_param",
                           params={"device_type": "gain", "occurrence": 0,
                                   "value": -3.0, "stem": "master"})
        assert resp.status_code == 200
        observation = recv_until(ws, "chat_message")
        assert observation.get("observation") is True
        assert "gain" in observation["text"].lower()


# ---------- composer settings (model + reasoning) ----------

def test_chat_settings_roundtrip_and_validation(client):
    with client.websocket_connect("/ws") as ws:
        snapshot = ws.receive_json()
        settings = snapshot["chat_settings"]
        assert settings["model"] == "claude-opus-5-5"
        assert settings["reasoning"] == "medium"   # the default since Phase 54 (llm_models)
        # The 5.5 family and nothing else since 2026-10-08 (James: "remove all models that
        # are not in the 5.5 family"), each under its Anthropic name, from llm_models.MODELS.
        ids = [m["id"] for m in settings["models"]]
        assert ids == ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
        labels = {m["id"]: m["label"] for m in settings["models"]}
        assert labels["claude-opus-5-5"] == "Opus 5.5"
        assert settings["reasoning_levels"] == ["low", "medium", "high", "xhigh", "max"]

        ws.send_json({"type": "set_chat_settings",
                      "model": "claude-sonnet-5-5", "reasoning": "medium"})
        snapshot = recv_until(ws, "snapshot")
        assert snapshot["chat_settings"]["model"] == "claude-sonnet-5-5"
        assert snapshot["chat_settings"]["reasoning"] == "medium"
        chat = client.app.state.gateway.chat
        assert chat.settings == {"model": "claude-sonnet-5-5",
                                 "reasoning": "medium"}

        # unknown values are rejected server-side, settings unchanged
        ws.send_json({"type": "set_chat_settings",
                      "model": "gpt-9000", "reasoning": "medium"})
        error = ws.receive_json()  # recv_until treats errors as failures
        assert error["type"] == "error"
        assert "unknown chat model" in error["message"]
        assert chat.settings["model"] == "claude-sonnet-5-5"
        # ...and so is a model the picker held until 2026-10-08
        ws.send_json({"type": "set_chat_settings",
                      "model": "claude-opus-5", "reasoning": "medium"})
        assert "unknown chat model" in ws.receive_json()["message"]
        assert chat.settings["model"] == "claude-sonnet-5-5"


# ---------- per-project notes memory ----------

def test_notes_persist_across_gateway_restarts(demo_project):
    app = create_app(demo_project, agent_factory=RuleAgent,
                     chat_factory=ScriptedChat)
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "chat",
                          "text": "remember: keep the vocal bright"})
            recv_until(ws, "chat_message")  # echo
            reply = recv_until(ws, "chat_message")
            assert "keep the vocal bright" in reply["text"]
    notes_file = demo_project / "runs" / "notes.jsonl"
    assert notes_file.exists()

    # a fresh gateway on the same project loads the note into context
    app2 = create_app(demo_project, agent_factory=RuleAgent,
                      chat_factory=ScriptedChat)
    with TestClient(app2) as client2:
        context = json.loads(client2.app.state.gateway._chat_context())
        assert "keep the vocal bright" in context["project_notes"]


# ---------- chat triggers Auto Mix ----------

def test_chat_triggers_auto_mix(client, demo_project):
    # Auto Mix auto-applies only in Agent mode; in Ask mode it gates (see the
    # test below). This exercises the hands-off Agent-mode path.
    client.app.state.gateway.agent_mode = "agent"
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "chat",
                      "text": "auto mix this for streaming please"})
        recv_until(ws, "chat_message")  # echo
        reply = recv_until(ws, "chat_message")
        assert "auto mix started" in reply["text"].lower()
        finished = recv_until(ws, "automix_finished")
        assert finished["applied"] >= 1
        snapshot = recv_until(ws, "snapshot")
        assert snapshot["metrics"]["lufs_i"] == pytest.approx(-14.0, abs=1.0)


def test_chat_auto_mix_in_ask_mode_gates_instead_of_auto_applying(client, demo_project):
    """Ask mode: a chat 'mix this' must NOT auto-apply. It runs the gated pass
    loop so moves surface as approve/reject cards (the native plugin reads these
    from /api/chat_log) — the user's complaint that no cards appeared."""
    gateway = client.app.state.gateway
    gateway.agent_mode = "ask"
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "chat",
                      "text": "auto mix this for streaming please"})
        recv_until(ws, "chat_message")  # echo
        reply = recv_until(ws, "chat_message")
        assert "ask mode" in reply["text"].lower()
    # it started the full Auto Mix under mode gating (follows the Ask/Agent switch)
    assert gateway.run["brief"].startswith("auto mix —")
    assert gateway.run["gating"] == "mode"


def test_auto_mix_in_ask_mode_completes_via_approvals(client, demo_project):
    """The Ask-mode fix (the user's 'it stops, doesn't finish the task' report):
    Auto Mix now gates every move as an approve/reject card AND, when each is
    approved, carries the FULL chain (bootstrap + refinement) to completion — not
    the shallow early stop of a target-driven run."""
    client.app.state.gateway.agent_mode = "ask"
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # snapshot
        ws.send_json({"type": "chat", "text": "auto mix this for streaming"})
        applied = 0
        for _ in range(600):
            ev = ws.receive_json()
            if ev["type"] == "error":
                pytest.fail(ev["message"])
            if ev["type"] == "awaiting_approval":          # gated — approve each move
                ws.send_json({"type": "approve",
                              "proposal_id": ev["proposal_id"]})
            elif ev["type"] == "proposal_status" and ev["status"] == "applied":
                applied += 1
            elif ev["type"] == "automix_finished":
                break
        else:
            pytest.fail("ask-mode auto mix never finished")
    assert applied >= 2   # the whole starting chain went through, one card at a time


# ---------- workspace awareness ----------

def test_workspace_state_and_edits_reach_chat_context(client):
    gateway = client.app.state.gateway
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "workspace_state", "playhead_s": 12.3,
                      "loop_region": [8.0, 16.0], "solo_stem": "stem_a",
                      "selected_stem": None})
        ws.send_json({"type": "user_edit", "stem_id": "stem_b",
                      "changes": {"gain_db": -3.0}})
        recv_until(ws, "snapshot")

    context = json.loads(gateway._chat_context())
    assert context["workspace"]["playhead_s"] == 12.3
    assert context["workspace"]["loop_region"] == [8.0, 16.0]
    assert context["workspace"]["solo_stem"] == "stem_a"
    (edit,) = context["recent_user_edits"]
    assert edit["stem_id"] == "stem_b" and edit["changes"]["gain_db"] == -3.0


def test_region_selection_reaches_chat_context(client):
    """Phase 33: an Alt-drag region highlight rides the workspace_state
    passthrough into the chat context, so "adjust this part" resolves to a
    ranged edit on that stem."""
    gateway = client.app.state.gateway
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "workspace_state", "selected_stem": "stem_a",
                      "selection": {"stem_id": "stem_a",
                                    "start_s": 4.0, "end_s": 9.5}})
        # a user_edit forces a snapshot to wait on — by the time it arrives the
        # gateway has processed the workspace_state sent just before it.
        ws.send_json({"type": "user_edit", "stem_id": "stem_b",
                      "changes": {"gain_db": -1.0}})
        recv_until(ws, "snapshot")
    selection = json.loads(gateway._chat_context())["workspace"]["selection"]
    assert selection == {"stem_id": "stem_a", "start_s": 4.0, "end_s": 9.5}
