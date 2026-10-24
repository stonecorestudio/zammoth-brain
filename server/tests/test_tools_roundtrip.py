"""Round-trip every Phase 1 MCP tool against the two-stem demo (CLAUDE.md rule).

The apply/undo/render happy path is covered end-to-end by the acceptance test;
this file covers the remaining tools and edge behavior at the MCP boundary.
"""
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from mix_engine.mcp_server import create_server
from mix_engine.models import BAND_LABELS

from test_phase1_acceptance import call

GAIN_PROPOSAL = {"req": {
    "rationale": "nudge stem_b down",
    "predicted_impact": [{"metric": "stem(stem_b).lufs_s",
                          "predicted_delta": -1.0, "unit": "LUFS"}],
    "actions": [{"type": "gain", "stem_id": "stem_b", "delta_db": -1.0}],
}}


@pytest.mark.anyio
async def test_analyze_session_fields(demo_project):
    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        result = await call(client, "analyze_session")
        assert result["stems"] == ["stem_a", "stem_b"]
        assert result["pass_number"] == 1
        assert not result["silent"]
        assert result["lufs_i"] is not None
        assert result["lufs_s_max"] >= result["lufs_i"]
        assert result["true_peak_dbtp"] < 0.0
        assert set(result["band_energy"]) == set(BAND_LABELS)
        # stem_a is a 440 Hz sine; its band must carry real energy in the mix
        assert result["band_energy"]["mid"] > -20.0


@pytest.mark.anyio
async def test_get_proposals_status_filter(demo_project):
    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        accepted = await call(client, "propose_actions", GAIN_PROPOSAL)
        rejected = await call(client, "propose_actions", {"req": {
            "rationale": "way too hot",
            "predicted_impact": [],
            "actions": [{"type": "gain", "stem_id": "stem_a", "delta_db": 9.0}],
        }})
        assert accepted["accepted"] and not rejected["accepted"]

        everything = await call(client, "get_proposals")
        assert [p["status"] for p in everything] == ["proposed", "rejected"]

        proposed = await call(client, "get_proposals", {"status": "proposed"})
        assert [p["proposal_id"] for p in proposed] == [accepted["proposal_id"]]


@pytest.mark.anyio
async def test_guardrail_violations_are_machine_readable(demo_project):
    from mix_engine.session import Session
    Session(demo_project).user_edit("stem_a", {"gain_db": -12.0})  # the fader starts low: these moves test the gain WALLET, and a fader stops at +6 (test_fader_ceiling)
    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        result = await call(client, "propose_actions", {"req": {
            "rationale": "slam it",
            "predicted_impact": [],
            "actions": [{"type": "gain", "stem_id": "stem_a", "delta_db": 9.0}],
        }})
        assert result["accepted"] is False and result["proposal_id"] is None
        (violation,) = result["violations"]
        assert violation["rule"] == "gain_per_stem_per_pass_db"
        assert violation["limit"] == 6.0
        assert violation["value"] == 9.0
        assert violation["action_index"] == 0


@pytest.mark.anyio
async def test_render_is_deterministic(demo_project):
    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        first = await call(client, "render_and_analyze")
        first_bytes = (demo_project / first["bounce_path"]).read_bytes()
        second = await call(client, "render_and_analyze")
        assert (demo_project / second["bounce_path"]).read_bytes() == first_bytes
        assert first["analysis"] == second["analysis"]


@pytest.mark.anyio
async def test_render_range(demo_project):
    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        result = await call(client, "render_and_analyze", {"range_s": [0.0, 2.0]})
        assert result["analysis"]["lufs_i"] is not None
        import soundfile as sf
        info = sf.info(demo_project / result["bounce_path"])
        assert info.frames == 2 * info.samplerate


@pytest.mark.anyio
async def test_apply_and_undo_roundtrip_state(demo_project):
    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        proposal = await call(client, "propose_actions", GAIN_PROPOSAL)
        assert (await call(client, "apply_proposal",
                           {"proposal_id": proposal["proposal_id"]}))["ok"]
        (applied,) = await call(client, "get_proposals", {"status": "applied"})
        # ignore optional fields the schema defaults to None (e.g. Phase 5's
        # gain time_range_s) — the round-trip contract is about set values
        returned = [{k: v for k, v in a.items() if v is not None}
                    for a in applied["actions"]]
        assert returned == GAIN_PROPOSAL["req"]["actions"]

        assert (await call(client, "undo_proposal",
                           {"proposal_id": proposal["proposal_id"]}))["ok"]
        assert await call(client, "get_proposals", {"status": "applied"}) == []
