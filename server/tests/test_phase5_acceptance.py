"""Phase 5 acceptance (docs/05-roadmap.md): on the three-section demo, a
scripted (no-LLM) loop applies a +3 dB vocal ride only in the chorus;
per-section analysis shows the delta in the chorus and < 0.2 dB drift
elsewhere; renders are bit-identical across repeats and undo restores the
previous bounce exactly.
"""
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from mix_engine.mcp_server import create_server

from test_phase1_acceptance import call


def section_lufs(sections: list[dict]) -> list[float]:
    return [s["lufs_i"] for s in sections]


@pytest.mark.anyio
async def test_chorus_only_vocal_ride(three_section_project):
    server = create_server(three_section_project)
    async with create_connected_server_and_client_session(server) as client:
        baseline_render = await call(client, "render_and_analyze")
        bounce = three_section_project / baseline_render["bounce_path"]
        baseline_bytes = bounce.read_bytes()

        # The "agent" reads sections and targets the loudest one (the chorus).
        before = await call(client, "analyze_sections")
        assert len(before["sections"]) == 3
        chorus = max(before["sections"], key=lambda s: s["lufs_i"])

        proposal = await call(client, "propose_actions", {"req": {
            "rationale": "ride the vocal up in the chorus only",
            "predicted_impact": [
                {"metric": "section(B).lufs_i", "predicted_delta": 1.0, "unit": "LUFS"},
            ],
            "actions": [{"type": "gain", "stem_id": "stem_a", "delta_db": 3.0,
                         "time_range_s": [chorus["start_s"], chorus["end_s"]]}],
        }})
        assert proposal["accepted"], proposal["violations"]
        applied = await call(client, "apply_proposal",
                             {"proposal_id": proposal["proposal_id"]})
        assert applied["ok"], applied["message"]

        # Determinism: the same state renders bit-identically twice.
        first = await call(client, "render_and_analyze")
        ridden_bytes = bounce.read_bytes()
        await call(client, "render_and_analyze")
        assert bounce.read_bytes() == ridden_bytes
        assert ridden_bytes != baseline_bytes

        # The ride lands in the chorus and nowhere else.
        after = await call(client, "analyze_sections")
        deltas = [a - b for a, b in zip(section_lufs(after["sections"]),
                                        section_lufs(before["sections"]))]
        chorus_index = before["sections"].index(chorus)
        assert deltas[chorus_index] > 0.5, f"chorus did not move: {deltas}"
        for i, delta in enumerate(deltas):
            if i != chorus_index:
                assert abs(delta) < 0.2, f"section {i} drifted: {deltas}"

        # The ride shows up in the stem's applied chain (the agent must see it).
        stem = await call(client, "analyze_stem", {"stem_id": "stem_a"})
        rides = [e for e in stem["applied_chain"] if e["type"] == "gain_ride"]
        assert rides == [{"type": "gain_ride", "start_s": chorus["start_s"],
                          "end_s": chorus["end_s"], "delta_db": 3.0}]

        # Undo restores the previous bounce bit-identically.
        undone = await call(client, "undo_proposal",
                            {"proposal_id": proposal["proposal_id"]})
        assert undone["ok"], undone["message"]
        await call(client, "render_and_analyze")
        assert bounce.read_bytes() == baseline_bytes
