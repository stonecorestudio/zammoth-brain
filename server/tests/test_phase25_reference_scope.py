"""Phase 25 (docs/05-roadmap.md): reference scope — mix / master / both.

Pre-master analysis, the mix-only target set, the mix-scope master-action
filter, the ws scope control, and the staged "both" run.
"""
import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from starlette.testclient import TestClient

from mix_engine.mcp_server import create_server
from orchestrator.agents import RuleAgent
from orchestrator.gateway import create_app
from orchestrator.loop import EngineClient, run_passes
from orchestrator.targets import reference_mix_targets, reference_targets

from test_phase1_acceptance import call
from test_phase3_gateway import recv_until
from test_phase7_reference import synthesize_reference

LIMITER = {"req": {
    "rationale": "master to -14",
    "predicted_impact": [{"metric": "lufs_i", "predicted_delta": 2.0,
                          "unit": "LUFS"}],
    "actions": [{"type": "limiter", "stem_id": "master",
                 "ceiling_dbtp": -1.0, "target_lufs_i": -14.0}],
}}


# ---------- engine: analyze_session(pre_master) ----------

@pytest.mark.anyio
async def test_pre_master_equals_master_with_empty_chain(demo_project):
    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        master = await call(client, "analyze_session")
        mix = await call(client, "analyze_session", {"pre_master": True})
    assert master["pre_master"] is False
    assert mix["pre_master"] is True
    # nothing on the master bus: the two renders are the same audio
    assert mix["lufs_i"] == master["lufs_i"]
    assert mix["band_energy"] == master["band_energy"]


@pytest.mark.anyio
async def test_pre_master_bypasses_master_chain(demo_project):
    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        baseline = await call(client, "analyze_session", {"pre_master": True})
        accepted = await call(client, "propose_actions", LIMITER)
        await call(client, "apply_proposal",
                   {"proposal_id": accepted["proposal_id"]})
        master = await call(client, "analyze_session")
        mix = await call(client, "analyze_session", {"pre_master": True})
        again = await call(client, "analyze_session", {"pre_master": True})

    # the limiter pushed the master toward -14; the pre-master mix is the
    # untouched sum — exactly the baseline, and deterministic call to call
    assert master["lufs_i"] == pytest.approx(-14.0, abs=1.0)
    assert mix["lufs_i"] == baseline["lufs_i"]
    assert mix["lufs_i"] != master["lufs_i"]
    assert again == mix


# ---------- targets: the mix-scope set ----------

def test_reference_mix_targets_are_tonal_only():
    reference = {"lufs_i": -9.0,
                 "band_energy": {"bass": -8.0, "mid": -12.0, "air": -40.0}}
    metrics = {t.metric for t in reference_mix_targets(reference)}
    # band goals above the floor only — no loudness/peak/PLR, no air (-40
    # sits under REFERENCE_BAND_FLOOR_DB)
    assert metrics == {"band_energy.bass", "band_energy.mid"}
    full = {t.metric for t in reference_targets(reference)}
    assert {"lufs_i", "true_peak_dbtp", "plr_db"} <= full


def test_reference_mix_targets_reject_silence():
    with pytest.raises(ValueError):
        reference_mix_targets({"lufs_i": None, "band_energy": None})


# ---------- loop: mix scope strips master moves ----------

class MasterHappyAgent:
    """Always proposes one bundle mixing a stem move and a master move."""

    def propose(self, ctx):
        return [{
            "rationale": "tame stem_b and tilt the master",
            "predicted_impact": [],
            "actions": [
                {"type": "gain", "stem_id": "stem_b", "delta_db": -1.0},
                {"type": "limiter", "stem_id": "master",
                 "ceiling_dbtp": -1.0, "target_lufs_i": -14.0},
            ],
        }]

    def revise(self, proposal, violations, ctx):
        return None


@pytest.mark.anyio
async def test_mix_scope_filters_master_actions(demo_project):
    synthesize_reference(demo_project)
    events = []

    async def emit(event):
        events.append(event)

    async def approve_all(proposal_id, proposal):
        return True

    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        await run_passes(EngineClient(client), demo_project,
                         "match refs/reference.wav", MasterHappyAgent(),
                         max_passes=1, gate=approve_all, emit=emit,
                         reference_scope="mix", stage="mix")
        applied = await call(client, "get_proposals", {"status": "applied"})
        state = await call(client, "get_state")

    (proposal,) = applied
    assert [a["type"] for a in proposal["actions"]] == ["gain"]
    # the limiter never reached the bus (the chain's gain entry is the fader)
    assert [e for e in state["master_chain"] if e["type"] != "gain"] == []
    filtered = [e for e in events if e["type"] == "actions_filtered"]
    assert filtered and filtered[0]["dropped"] == 1
    (started,) = [e for e in events if e["type"] == "run_started"]
    assert started["stage"] == "mix"


@pytest.mark.anyio
async def test_master_scope_keeps_master_actions(demo_project):
    synthesize_reference(demo_project)

    async def approve_all(proposal_id, proposal):
        return True

    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        await run_passes(EngineClient(client), demo_project,
                         "match refs/reference.wav", MasterHappyAgent(),
                         max_passes=1, gate=approve_all,
                         reference_scope="master")
        state = await call(client, "get_state")
    assert [e["type"] for e in state["master_chain"]
            if e["type"] != "gain"] == ["limiter"]


# ---------- gateway: scope control + the staged "both" run ----------

@pytest.fixture()
def referenced_client(demo_project):
    synthesize_reference(demo_project)
    app = create_app(demo_project, agent_factory=RuleAgent)
    with TestClient(app) as test_client:
        yield test_client


def test_scope_roundtrip_and_validation(referenced_client):
    with referenced_client.websocket_connect("/ws") as ws:
        snapshot = ws.receive_json()
        assert snapshot["reference_scope"] == "master"  # default = old behavior
        ws.send_json({"type": "set_reference_scope", "scope": "both"})
        assert recv_until(ws, "snapshot")["reference_scope"] == "both"
        ws.send_json({"type": "set_reference_scope", "scope": "loud"})
        error = recv_until(ws, "error")
        assert "reference scope" in error["message"]


def test_both_scope_runs_mix_stage_then_master_stage(referenced_client):
    with referenced_client.websocket_connect("/ws") as ws:
        ws.receive_json()  # snapshot
        ws.send_json({"type": "set_reference", "path": "refs/reference.wav"})
        ws.send_json({"type": "set_reference_scope", "scope": "both"})
        ws.send_json({"type": "start_run", "brief": "match refs/reference.wav",
                      "gating": "auto", "max_passes": 1})
        stages = []
        finishes = 0
        while finishes < 2:
            event = ws.receive_json()
            if event["type"] == "run_started":
                stages.append(event["stage"])
            elif event["type"] == "run_finished":
                finishes += 1
            elif event["type"] == "error":
                pytest.fail(f"gateway error: {event['message']}")
        assert stages == ["mix", "master"]
