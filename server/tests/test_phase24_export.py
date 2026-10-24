"""Phase 24: export_mix — deliverable wavs round-tripped on the two-stem demo."""
import numpy as np
import pytest
import soundfile as sf
from mcp.shared.memory import create_connected_server_and_client_session

from mix_engine.mcp_server import create_server

from test_phase1_acceptance import call


@pytest.mark.anyio
@pytest.mark.parametrize("fmt,subtype", [("wav24", "PCM_24"),
                                         ("wav16", "PCM_16")])
async def test_master_export_roundtrip(demo_project, tmp_path, fmt, subtype):
    out_dir = tmp_path / "deliverables"
    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        result = await call(client, "export_mix", {"req": {
            "out_dir": str(out_dir), "format": fmt}})

    assert result["format"] == fmt
    assert result["out_dir"] == str(out_dir)
    (master,) = result["files"]
    assert master["kind"] == "master"
    assert master["path"] == str(out_dir / "two_stem_demo - Master.wav")
    assert master["clipped"] is False
    assert master["peak_dbfs"] is not None and master["peak_dbfs"] < 0.0

    info = sf.info(master["path"])
    assert info.format == "WAV" and info.subtype == subtype
    assert info.samplerate == result["sample_rate_hz"]
    assert info.duration == pytest.approx(master["duration_s"], abs=0.01)


@pytest.mark.anyio
async def test_wav32f_export_matches_bounce(demo_project, tmp_path):
    out_dir = tmp_path / "deliverables"
    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        await call(client, "render_and_analyze")
        result = await call(client, "export_mix", {"req": {
            "out_dir": str(out_dir), "format": "wav32f"}})

    exported = (out_dir / "two_stem_demo - Master.wav").read_bytes()
    bounce = (demo_project / "bounces" / "master.wav").read_bytes()
    assert exported == bounce


@pytest.mark.anyio
async def test_stems_export_skips_inactive(demo_project, tmp_path):
    out_dir = tmp_path / "deliverables"
    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        await call(client, "set_stem", {"stem_id": "stem_b", "muted": True})
        result = await call(client, "export_mix", {"req": {
            "out_dir": str(out_dir), "include_stems": True}})

    kinds = [(f["kind"], f["stem_id"]) for f in result["files"]]
    assert kinds == [("master", None), ("stem", "stem_a")]
    stems_dir = out_dir / "two_stem_demo - Stems"
    assert [p.name for p in sorted(stems_dir.iterdir())] == ["stem_a.wav"]


@pytest.mark.anyio
async def test_export_into_project_rejected(demo_project, tmp_path):
    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        for target in (demo_project, demo_project / "exports"):
            result = await client.call_tool(
                "export_mix", {"req": {"out_dir": str(target)}})
            assert result.isError
            assert "outside the project" in result.content[0].text
        # nothing was written into the project
        assert not (demo_project / "exports").exists()


@pytest.mark.anyio
async def test_pcm_export_is_deterministic(demo_project, tmp_path):
    out_dir = tmp_path / "deliverables"
    master = out_dir / "two_stem_demo - Master.wav"
    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        await call(client, "export_mix", {"req": {"out_dir": str(out_dir)}})
        first = master.read_bytes()
        await call(client, "export_mix", {"req": {"out_dir": str(out_dir)}})
        assert master.read_bytes() == first


@pytest.mark.anyio
@pytest.mark.parametrize("fmt,ext,sf_format,subtype",
                         [("flac", ".flac", "FLAC", "PCM_24"),
                          ("mp3", ".mp3", "MP3", "MPEG_LAYER_III")])
async def test_compressed_export_roundtrip(demo_project, tmp_path, fmt, ext,
                                           sf_format, subtype):
    """Phase 26 format expansion: flac (24-bit lossless) and mp3 (320 CBR)
    deliverables, including the swapped file extension on stems."""
    out_dir = tmp_path / "deliverables"
    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        result = await call(client, "export_mix", {"req": {
            "out_dir": str(out_dir), "format": fmt, "include_stems": True}})

    assert result["format"] == fmt
    master = result["files"][0]
    assert master["path"] == str(out_dir / f"two_stem_demo - Master{ext}")
    info = sf.info(master["path"])
    assert info.format == sf_format and info.subtype == subtype
    assert info.samplerate == result["sample_rate_hz"]
    # mp3 grows by encoder padding; never shorter than the render
    assert info.duration >= master["duration_s"] - 0.01
    stems_dir = out_dir / "two_stem_demo - Stems"
    assert sorted(p.name for p in stems_dir.iterdir()) == [
        f"stem_a{ext}", f"stem_b{ext}"]


@pytest.mark.anyio
async def test_mp3_export_rejected_above_48k(demo_project, tmp_path):
    """No resampling on export: a 96 kHz session (rates resample *up* at
    read time, Phase 15) must refuse mp3 with a clear pointer to wav/flac,
    and write nothing."""
    t = np.arange(2 * 96000) / 96000
    sf.write(demo_project / "air.wav",
             (0.1 * np.sin(2 * np.pi * 12000 * t)).astype(np.float32), 96000)
    out_dir = tmp_path / "deliverables"
    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        result = await client.call_tool("export_mix", {"req": {
            "out_dir": str(out_dir), "format": "mp3"}})
        assert result.isError
        assert "48 kHz" in result.content[0].text
        assert "flac" in result.content[0].text
    assert list(out_dir.glob("*.mp3")) == []


@pytest.mark.anyio
async def test_hot_master_clips_instead_of_wrapping(demo_project, tmp_path):
    out_dir = tmp_path / "deliverables"
    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        # user channel: push the mix well past full scale (no guardrails)
        await call(client, "set_stem", {"stem_id": "stem_a", "gain_db": 12.0})
        await call(client, "set_stem", {"stem_id": "master", "gain_db": 12.0})
        result = await call(client, "export_mix", {"req": {
            "out_dir": str(out_dir), "format": "wav24"}})

    (master,) = result["files"]
    assert master["clipped"] is True
    assert master["peak_dbfs"] > 0.0
    # the written file is hard-clipped at full scale, never wrapped
    audio, _ = sf.read(master["path"], dtype="float64", always_2d=True)
    assert np.max(np.abs(audio)) <= 1.0
    assert np.max(audio) == pytest.approx(8388607 / 8388608, abs=1e-6)


# ---------- gateway endpoint (Phase 24) ----------

def test_gateway_export_endpoint(demo_project, tmp_path):
    from starlette.testclient import TestClient

    from orchestrator.agents import RuleAgent
    from orchestrator.gateway import create_app

    out_dir = tmp_path / "deliverables"
    app = create_app(demo_project, agent_factory=RuleAgent)
    with TestClient(app) as client:
        ok = client.post("/api/export", params={
            "dir": str(out_dir), "format": "wav16", "include_stems": True})
        assert ok.status_code == 200
        result = ok.json()
        assert result["format"] == "wav16"
        assert [f["kind"] for f in result["files"]] == ["master", "stem", "stem"]
        assert (out_dir / "two_stem_demo - Master.wav").exists()

        bad = client.post("/api/export", params={"dir": str(demo_project)})
        assert bad.status_code == 400
        assert "outside the project" in bad.json()["detail"]
