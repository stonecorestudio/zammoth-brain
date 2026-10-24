"""Phase 7 (docs/05-roadmap.md): reference matching + expanded palette.

Round-trips analyze_reference and the three new processors through the MCP
boundary, the reference-target translation, and the acceptance criterion:
with a synthesized reference (the demo mix rendered with a known tilt +
loudness offset), the loop closes >= 70 % of the initial per-band gap on
every considered band within 5 passes. Bands whose initial gap is already
inside the matching tolerance (1 dB) are met by definition and only checked
for staying small.
"""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from mcp.shared.memory import create_connected_server_and_client_session
from scipy.signal import butter, sosfilt

from mix_engine.mcp_server import create_server
from orchestrator.agents import RuleAgent
from orchestrator.loop import run_session
from orchestrator.targets import (REFERENCE_BAND_FLOOR_DB,
                                  REFERENCE_BAND_TOLERANCE_DB,
                                  reference_path_from_brief, reference_targets)

from test_phase1_acceptance import call


def in_memory_engine(project_dir):
    return create_connected_server_and_client_session(create_server(project_dir))


def synthesize_reference(project: Path) -> Path:
    """The demo master with a known tilt (+~6 dB above 2 kHz) and +3 dB level,
    written to refs/reference.wav — NOT the project root, where it would be
    discovered as a stem and join the mix (docs/03)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mix_engine import dsp
    from mix_engine.session import Session

    mix, sample_rate = dsp.render_master(Session(project))
    mono = mix.mean(axis=1)
    sos = butter(2, 2000.0, btype="highpass", fs=sample_rate, output="sos")
    tilted = (mono + 1.0 * sosfilt(sos, mono)) * 10 ** (3.0 / 20.0)
    tilted /= max(1.0, np.abs(tilted).max() / 0.9)
    (project / "refs").mkdir(exist_ok=True)
    sf.write(project / "refs" / "reference.wav", tilted.astype(np.float32),
             sample_rate)
    return project / "refs" / "reference.wav"


# ---------- analyze_reference ----------

@pytest.mark.anyio
async def test_analyze_reference_roundtrip_and_cache(demo_project):
    synthesize_reference(demo_project)
    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        result = await call(client, "analyze_reference",
                            {"path": "refs/reference.wav"})
        assert result["lufs_i"] is not None
        assert result["crest_factor_db"] is not None
        assert len(result["band_energy"]) == 8
        assert result["band_width_db"] is None  # mono reference
        assert (demo_project / ".reference_cache.json").exists()
        # second call comes from the cache and matches exactly
        assert await call(client, "analyze_reference",
                          {"path": "refs/reference.wav"}) == result


# ---------- new processors ----------

@pytest.mark.anyio
@pytest.mark.parametrize("actions", [
    [{"type": "saturation", "stem_id": "stem_a", "drive_db": 9.0}],
    [{"type": "transient", "stem_id": "stem_b", "attack_db": 4.0,
      "sustain_db": -2.0}],
    [{"type": "duck", "stem_id": "stem_b", "key_stem_id": "stem_a",
      "threshold_db": -30.0, "depth_db": 8.0}],
])
async def test_processor_apply_render_undo(demo_project, actions):
    """Each Phase 7 processor: changes the bounce, renders deterministically,
    undoes bit-identically, and shows in the applied chain."""
    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        baseline = await call(client, "render_and_analyze")
        baseline_bytes = (demo_project / baseline["bounce_path"]).read_bytes()

        proposal = await call(client, "propose_actions", {"req": {
            "rationale": "phase 7 processor", "predicted_impact": [],
            "actions": actions}})
        assert proposal["accepted"], proposal["violations"]
        assert (await call(client, "apply_proposal",
                           {"proposal_id": proposal["proposal_id"]}))["ok"]

        first = await call(client, "render_and_analyze")
        bounce = demo_project / first["bounce_path"]
        changed = bounce.read_bytes()
        assert changed != baseline_bytes
        await call(client, "render_and_analyze")
        assert bounce.read_bytes() == changed  # deterministic

        stem = await call(client, "analyze_stem", {"stem_id": actions[0]["stem_id"]})
        assert any(e["type"] == actions[0]["type"] for e in stem["applied_chain"])

        assert (await call(client, "undo_proposal",
                           {"proposal_id": proposal["proposal_id"]}))["ok"]
        await call(client, "render_and_analyze")
        assert bounce.read_bytes() == baseline_bytes


@pytest.mark.anyio
async def test_duck_reduces_target_under_key(demo_project):
    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        before = await call(client, "analyze_stem", {"stem_id": "stem_b"})
        proposal = await call(client, "propose_actions", {"req": {
            "rationale": "kick->bass classic", "predicted_impact": [],
            "actions": [{"type": "duck", "stem_id": "stem_b",
                         "key_stem_id": "stem_a", "threshold_db": -30.0,
                         "depth_db": 8.0}]}})
        assert (await call(client, "apply_proposal",
                           {"proposal_id": proposal["proposal_id"]}))["ok"]
        after = await call(client, "analyze_stem", {"stem_id": "stem_b"})
        assert after["lufs_s"] < before["lufs_s"] - 1.0  # audibly ducked


@pytest.mark.anyio
async def test_duck_key_rules(demo_project):
    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        for key in ("stem_b", "nope"):  # self-key and unknown key
            result = await call(client, "propose_actions", {"req": {
                "rationale": "bad duck", "predicted_impact": [],
                "actions": [{"type": "duck", "stem_id": "stem_b",
                             "key_stem_id": key, "threshold_db": -30.0,
                             "depth_db": 6.0}]}})
            assert not result["accepted"]
            assert result["violations"][0]["rule"] == "duck_key_stem"


# ---------- reference targets ----------

def test_reference_brief_parsing():
    assert reference_path_from_brief("match reference.wav") == "reference.wav"
    assert reference_path_from_brief("sit like refs/mix.wav") == "refs/mix.wav"
    assert reference_path_from_brief("stem A clarity") is None


def test_reference_targets_floor_and_goals():
    targets = reference_targets({
        "lufs_i": -12.0,
        "band_energy": {"mid": -6.0, "air": -45.0},  # air below the -30 floor
    })
    metrics = {t.metric: t for t in targets}
    assert set(metrics) == {"lufs_i", "true_peak_dbtp", "plr_db",
                            "band_energy.mid"}
    assert metrics["band_energy.mid"].goal == -6.0
    assert metrics["true_peak_dbtp"].max_value == -1.0
    assert metrics["plr_db"].min_value == 4.0  # the 2026-06 punch guard


# ---------- acceptance ----------

def test_phase7_acceptance_reference_matching(demo_project):
    """The roadmap criterion: >= 70 % closure of every meaningful initial
    band gap within 5 passes (sub-tolerance gaps just have to stay small)."""
    synthesize_reference(demo_project)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mix_engine import analysis, dsp
    from mix_engine.session import Session

    mix, sample_rate = dsp.render_master(Session(demo_project))
    ref_audio, _ = sf.read(demo_project / "refs" / "reference.wav", always_2d=True)
    reference = analysis.analyze_reference(ref_audio, sample_rate, "reference.wav")
    baseline = analysis.analyze_master(mix, sample_rate, stems=[], pass_number=1)
    considered = {band: value for band, value in reference.band_energy.items()
                  if value > REFERENCE_BAND_FLOOR_DB}
    initial = {band: abs(baseline.band_energy[band] - value)
               for band, value in considered.items()}

    report = run_session(demo_project, "match refs/reference.wav", gating="auto",
                         agent=RuleAgent(), connect=in_memory_engine)
    assert report.stop_reason == "targets met"
    assert report.passes_run <= 5

    final_bands = report.final_metrics["band_energy"]
    for band, goal in considered.items():
        final_gap = abs(final_bands[band] - goal)
        if initial[band] >= 1.0:
            closure = 1.0 - final_gap / initial[band]
            assert closure >= 0.7, (band, initial[band], final_gap)
        else:
            assert final_gap <= REFERENCE_BAND_TOLERANCE_DB + 0.2, (band, final_gap)
    assert report.final_metrics["lufs_i"] == pytest.approx(reference.lufs_i, abs=1.0)
    assert report.final_metrics["true_peak_dbtp"] <= -1.0
