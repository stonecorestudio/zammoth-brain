"""Phase 6 (docs/05-roadmap.md): master chain + loudness targets.

Round-trips the limiter and master-chain rules through the MCP boundary, the
brief -> loudness-goal table, and the acceptance criterion: a gated loop with
brief "master for streaming" lands lufs_i within ±0.5 LU of -14 with true peak
<= -1.0 dBTP, terminating via targets-met.
"""
import numpy as np
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from mix_engine import dsp
from mix_engine.mcp_server import create_server
from mix_engine.session import Session
from orchestrator.agents import RuleAgent
from orchestrator.loop import run_session
from orchestrator.targets import targets_from_brief

from test_phase1_acceptance import call


def in_memory_engine(project_dir):
    return create_connected_server_and_client_session(create_server(project_dir))


def limiter_proposal(target_lufs_i: float = -14.0, ceiling_dbtp: float = -1.0,
                     stem_id: str = "master") -> dict:
    return {"req": {
        "rationale": "master for streaming",
        "predicted_impact": [{"metric": "lufs_i", "predicted_delta": 4.8,
                              "unit": "LUFS"}],
        "actions": [{"type": "limiter", "stem_id": stem_id,
                     "ceiling_dbtp": ceiling_dbtp,
                     "target_lufs_i": target_lufs_i}],
    }}


@pytest.mark.anyio
async def test_limiter_apply_render_undo(demo_project):
    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        baseline = await call(client, "render_and_analyze")
        assert baseline["master"] is None  # no master chain yet
        baseline_bytes = (demo_project / baseline["bounce_path"]).read_bytes()

        proposal = await call(client, "propose_actions", limiter_proposal())
        assert proposal["accepted"], proposal["violations"]
        assert (await call(client, "apply_proposal",
                           {"proposal_id": proposal["proposal_id"]}))["ok"]

        result = await call(client, "render_and_analyze")
        assert result["analysis"]["lufs_i"] == pytest.approx(-14.0, abs=0.5)
        assert result["analysis"]["true_peak_dbtp"] <= -1.0
        assert "master_true_peak_exceeded" not in result["flags"]
        report = result["master"]
        # 1.8, not 4.8 (full audit 2026-10-02, E5·R1): the all-mono demo master used to be
        # measured as ONE channel, 3.01 dB under what a mono file plays at, so the limiter made
        # up 3 dB too much and the "−14" it reported exported at −11.
        assert report["limiter_makeup_db"] == pytest.approx(1.8, abs=0.3)
        assert report["pre_limiter_lufs_i"] is not None
        assert report["limiter_input_lufs_i"] == pytest.approx(-14.0, abs=0.3)
        assert report["post_limiter_lufs_i"] == pytest.approx(-14.0, abs=0.5)
        assert report["limiter_gr_mean_db"] >= 0.0
        assert report["limiter_gr_max_db"] >= report["limiter_gr_mean_db"]
        assert report["limiter_gr_active_pct"] >= 0.0
        assert report["true_peak_overshoot_db"] >= 0.0
        assert report["true_peak_margin_db"] >= -0.1

        # one limiter per bus: a second limiter replaces, not stacks
        again = await call(client, "propose_actions", limiter_proposal(-12.0))
        assert (await call(client, "apply_proposal",
                           {"proposal_id": again["proposal_id"]}))["ok"]
        result = await call(client, "render_and_analyze")
        assert result["analysis"]["lufs_i"] == pytest.approx(-12.0, abs=0.5)

        # undo both -> bit-identical baseline
        for pid in (again["proposal_id"], proposal["proposal_id"]):
            assert (await call(client, "undo_proposal", {"proposal_id": pid}))["ok"]
        after_undo = await call(client, "render_and_analyze")
        assert (demo_project / after_undo["bounce_path"]).read_bytes() == baseline_bytes


@pytest.mark.anyio
async def test_master_chain_rules(demo_project):
    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        # limiter is master-only
        on_stem = await call(client, "propose_actions", limiter_proposal(stem_id="stem_a"))
        assert not on_stem["accepted"]
        assert on_stem["violations"][0]["rule"] == "limiter_master_only"

        # only the limiter pushes loudness: no makeup on the bus comp
        comp = await call(client, "propose_actions", {"req": {
            "rationale": "glue with makeup (should be rejected)",
            "predicted_impact": [],
            "actions": [{"type": "comp", "stem_id": "master", "threshold_db": -18.0,
                         "ratio": 2.0, "attack_ms": 30.0, "release_ms": 200.0,
                         "makeup_db": 3.0}],
        }})
        assert not comp["accepted"]
        assert comp["violations"][0]["rule"] == "master_comp_makeup_max_db"

        # makeup-free bus comp and master eq are welcome
        ok = await call(client, "propose_actions", {"req": {
            "rationale": "bus glue + gentle top shelf",
            "predicted_impact": [],
            "actions": [
                {"type": "comp", "stem_id": "master", "threshold_db": -18.0,
                 "ratio": 2.0, "attack_ms": 30.0, "release_ms": 200.0},
                {"type": "eq", "stem_id": "master",
                 "band": {"freq_hz": 12000, "gain_db": 1.5, "q": 0.7,
                          "shape": "high_shelf"}},
            ],
        }})
        assert ok["accepted"], ok["violations"]
        assert (await call(client, "apply_proposal",
                           {"proposal_id": ok["proposal_id"]}))["ok"]
        rendered = await call(client, "render_and_analyze")
        assert rendered["analysis"]["lufs_i"] is not None  # chain renders


def test_limiter_report_is_pure_observation(demo_project):
    session = Session(demo_project)
    session._state["master_chain"].append(   # −5: see test_limiter_apply_render_undo's note
        {"type": "limiter", "ceiling_dbtp": -1.0, "target_lufs_i": -5.0})

    plain, sample_rate = dsp.render_master(session)
    report: dict = {}
    telemetry: list = []
    observed, observed_rate = dsp.render_master(session, report=report,
                                                telemetry=telemetry)

    assert observed_rate == sample_rate
    assert np.array_equal(observed, plain)
    assert report["post_limiter_lufs_i"] is not None
    assert report["limiter_gr_max_db"] > 0.0
    assert telemetry and telemetry[0]["processor"] == "limiter"


@pytest.mark.anyio
async def test_over_limited_master_flags_render(demo_project):
    # A limiter driven 3 dB past the loudest target the agent may ask for (−6), written straight
    # into the chain as a user's hand-set device would be. It used to be `limiter_proposal(-6.0)`:
    # with the all-mono demo master measured as ONE channel (3 dB low, E5·R1) that target made
    # up 12.8 dB and over-limited; measured as it plays, −6 is a 9.8 dB push the demo takes
    # cleanly (GR max 0.8 dB), which is the honest answer.
    seeded = Session(demo_project)
    seeded._state["master_chain"].append(
        {"type": "limiter", "ceiling_dbtp": -1.0, "target_lufs_i": -3.0})
    seeded._persist()
    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        result = await call(client, "render_and_analyze")
        assert "over_limiting" in result["flags"]
        assert "over_compression" not in result["flags"]
        assert result["master"]["limiter_gr_max_db"] > 0.0


def test_loudness_goal_briefs():
    def lufs_goal(brief: str) -> float:
        by_metric = {t.metric: t for t in targets_from_brief(brief, ["stem_a"])}
        # every loudness goal carries the ceiling and (2026-06) the PLR floor
        assert set(by_metric) == {"lufs_i", "true_peak_dbtp", "plr_db"}
        assert by_metric["true_peak_dbtp"].max_value == -1.0
        assert by_metric["plr_db"].min_value == 4.0
        return by_metric["lufs_i"].goal

    assert lufs_goal("master for streaming") == -14.0
    assert lufs_goal("master for Spotify please") == -14.0
    assert lufs_goal("club master") == -8.0
    assert lufs_goal("master to -11 LUFS") == -11.0  # explicit number wins
    assert lufs_goal("master for spotify at -10.5 lufs") == -10.5


def test_phase6_acceptance_streaming_master(demo_project):
    """The roadmap criterion, end to end through the gated loop."""
    report = run_session(demo_project, "master for streaming",
                         gating="auto", agent=RuleAgent(),
                         connect=in_memory_engine)
    assert report.stop_reason == "targets met"
    assert report.passes_run <= 5
    assert report.applied_proposal_ids
    assert report.final_metrics["lufs_i"] == pytest.approx(-14.0, abs=0.5)
    assert report.final_metrics["true_peak_dbtp"] <= -1.0
