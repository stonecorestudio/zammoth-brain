"""Phase 1 acceptance test (docs/05-roadmap.md).

The scripted (no-LLM) loop, exercised through the real MCP tool surface:
propose +2 dB on stem A of the two-stem demo, apply, render; assert stem A LUFS
rises ~2 dB; undo; assert the re-render is bit-identical to the original bounce.
"""
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from mix_engine.mcp_server import create_server


async def call(client, name: str, args: dict | None = None) -> dict | list:
    result = await client.call_tool(name, args or {})
    assert not result.isError, result.content
    payload = result.structuredContent
    if isinstance(payload, dict) and set(payload) == {"result"}:
        payload = payload["result"]  # FastMCP wraps non-object returns
    return payload


@pytest.mark.anyio
async def test_gain_apply_undo_roundtrip(demo_project):
    server = create_server(demo_project)
    async with create_connected_server_and_client_session(server) as client:
        baseline = await call(client, "render_and_analyze")
        bounce_path = demo_project / baseline["bounce_path"]
        baseline_bytes = bounce_path.read_bytes()

        stem_before = await call(client, "analyze_stem", {"stem_id": "stem_a"})
        assert stem_before["lufs_s"] is not None

        proposal = await call(client, "propose_actions", {"req": {
            "rationale": "bring the vocal stand-in forward",
            "predicted_impact": [
                {"metric": "stem(stem_a).lufs_s", "predicted_delta": 2.0, "unit": "LUFS"},
            ],
            "actions": [{"type": "gain", "stem_id": "stem_a", "delta_db": 2.0}],
        }})
        assert proposal["accepted"], proposal["violations"]

        applied = await call(client, "apply_proposal",
                             {"proposal_id": proposal["proposal_id"]})
        assert applied["ok"], applied["message"]

        after_apply = await call(client, "render_and_analyze")
        assert (demo_project / after_apply["bounce_path"]).read_bytes() != baseline_bytes

        stem_after = await call(client, "analyze_stem", {"stem_id": "stem_a"})
        delta_lufs = stem_after["lufs_s"] - stem_before["lufs_s"]
        assert delta_lufs == pytest.approx(2.0, abs=0.2)

        undone = await call(client, "undo_proposal",
                            {"proposal_id": proposal["proposal_id"]})
        assert undone["ok"], undone["message"]

        after_undo = await call(client, "render_and_analyze")
        assert (demo_project / after_undo["bounce_path"]).read_bytes() == baseline_bytes
