"""Phase 5 round-trips (docs/05-roadmap.md): analyze_sections through the MCP
boundary, the extended hearing metrics, and masking worst_windows.

Metric definitions under test are contractual: .claude/skills/mix-analysis/SKILL.md.
"""
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from mix_engine.mcp_server import create_server
from mix_engine.models import ANALYSIS_VERSION, BAND_LABELS

from test_phase1_acceptance import call


@pytest.mark.anyio
async def test_analyze_sections_uniform_track_is_one_section(demo_project):
    """The 8 s two-stem demo has no structure: one section spanning the file."""
    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        result = await call(client, "analyze_sections")
        assert result["analysis_version"] == ANALYSIS_VERSION
        assert result["duration_s"] == pytest.approx(8.0, abs=0.01)
        (section,) = result["sections"]
        assert section["label"] == "A"
        assert section["start_s"] == 0.0
        assert section["end_s"] == pytest.approx(8.0, abs=0.01)
        assert section["lufs_i"] is not None
        assert set(section["stem_activity"]) == {"stem_a", "stem_b"}


@pytest.mark.anyio
async def test_analyze_sections_finds_three_sections(three_section_project):
    """Verse/chorus/outro demo: three sections, boundaries at ~8 s and ~16 s,
    distinct labels, the chorus is the loudest, the outro bed is inactive."""
    async with create_connected_server_and_client_session(
            create_server(three_section_project)) as client:
        result = await call(client, "analyze_sections")
        sections = result["sections"]
        assert len(sections) == 3
        assert [s["start_s"] for s in sections] == pytest.approx([0.0, 8.0, 16.0], abs=1.0)
        assert sections[-1]["end_s"] == pytest.approx(24.0, abs=0.01)
        assert len({s["label"] for s in sections}) == 3  # all sound different
        loudest = max(sections, key=lambda s: s["lufs_i"])
        assert loudest is sections[1]  # the chorus
        assert sections[2]["stem_activity"]["stem_b"] < 0.1  # outro bed silent


@pytest.mark.anyio
async def test_extended_hearing_metrics(demo_project):
    """lufs_range / band_crest_db on the master; transient_energy_ratio /
    sibilance_density on stems — present, in range, version-bumped."""
    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        master = await call(client, "analyze_session")
        assert master["analysis_version"] == ANALYSIS_VERSION
        assert master["lufs_range"] is not None and master["lufs_range"] >= 0.0
        assert set(master["band_crest_db"]) == set(BAND_LABELS)
        assert all(v >= 0.0 for v in master["band_crest_db"].values())

        stem = await call(client, "analyze_stem", {"stem_id": "stem_b"})
        assert stem["transient_energy_ratio"] is not None
        assert 0.0 <= stem["transient_energy_ratio"] <= 1.0
        assert stem["sibilance_density"] is not None
        assert 0.0 <= stem["sibilance_density"] <= 1.0


@pytest.mark.anyio
async def test_masking_worst_windows(demo_project):
    """Masking now has a *when*: up to 3 timestamped worst runs in worst_band,
    sorted worst-first, none exceeding the whole-file cap."""
    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        result = await call(client, "analyze_masking",
                            {"stem_a": "stem_a", "stem_b": "stem_b"})
        windows = result["worst_windows"]
        assert 1 <= len(windows) <= 3
        for window in windows:
            assert window["end_s"] > window["start_s"] >= 0.0
            assert window["end_s"] <= 8.0
        excesses = [w["excess_db"] for w in windows]
        assert excesses == sorted(excesses, reverse=True)
        # the worst moment is at least as bad as the whole-file average
        assert excesses[0] >= result["worst_band_excess_db"] - 0.1


@pytest.mark.anyio
async def test_master_ride_is_rejected(demo_project):
    """Gain rides are stems-only (docs/03): master + time_range_s violates."""
    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        result = await call(client, "propose_actions", {"req": {
            "rationale": "ride the master (should be rejected)",
            "predicted_impact": [],
            "actions": [{"type": "gain", "stem_id": "master", "delta_db": -2.0,
                         "time_range_s": [0.0, 4.0]}],
        }})
        assert not result["accepted"]
        assert result["violations"][0]["rule"] == "master_ride_unsupported"
