"""Phase 28 (agent-owned routing).

Routing is the agent's now — the user's manual channel is only
volume/pan/mute/solo. The agent creates busses and routes stems through the
*proposal* lifecycle (atomic, guardrailed, bit-identical undo):

- `group` {bus_id, members[]}: create the bus if absent and route the members.
- `route` {stem_id, bus_id|None}: re-home one stem (None = back to master).

A `group` early in a proposal makes its bus a legal target for later actions in
the same proposal, so "group the drums and glue them" is one atomic move.
"""
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from mix_engine.mcp_server import create_server

from test_phase1_acceptance import call


def engine(project):
    return create_connected_server_and_client_session(create_server(project))


async def bounce_bytes(client, project) -> bytes:
    result = await call(client, "render_and_analyze")
    return (project / result["bounce_path"]).read_bytes()


async def submit(client, actions: list[dict], rationale: str = "phase28") -> str:
    """Propose + apply in one shot; return the proposal id."""
    proposal = await call(client, "propose_actions", {"req": {
        "rationale": rationale, "predicted_impact": [], "actions": actions}})
    assert proposal["accepted"], proposal["violations"]
    pid = proposal["proposal_id"]
    assert (await call(client, "apply_proposal", {"proposal_id": pid}))["ok"]
    return pid


@pytest.mark.anyio
async def test_group_creates_bus_and_routes(demo_project):
    async with engine(demo_project) as client:
        await submit(client, [{"type": "group", "bus_id": "drums",
                               "members": ["stem_a", "stem_b"]}],
                     "group the kit")

        state = await call(client, "get_state")
        (drums,) = state["buses"]
        assert drums["bus_id"] == "drums"
        assert drums["members"] == ["stem_a", "stem_b"]
        assert {s["stem_id"]: s["bus"] for s in state["stems"]} \
            == {"stem_a": "drums", "stem_b": "drums"}

        # analyze_bus hears the member sum through the (gain-only) bus chain
        bus = await call(client, "analyze_bus", {"bus_id": "drums"})
        assert bus["applied_chain"] == [{"type": "gain", "gain_db": 0.0}]
        assert bus["lufs_s"] is not None


@pytest.mark.anyio
async def test_group_and_glue_one_proposal_bitwise(demo_project):
    """The headline gesture: group + glue the bus in a single proposal, then
    undo restores the prior render bit-for-bit AND removes the created bus."""
    async with engine(demo_project) as client:
        baseline = await bounce_bytes(client, demo_project)

        pid = await submit(client, [
            {"type": "group", "bus_id": "drums", "members": ["stem_a"]},
            {"type": "comp", "stem_id": "drums", "threshold_db": -30.0,
             "ratio": 4.0, "attack_ms": 10.0, "release_ms": 120.0, "mix": 0.5},
        ], "group + glue the drums")

        glued = await bounce_bytes(client, demo_project)
        assert glued != baseline                                   # bus is in the graph
        assert glued == await bounce_bytes(client, demo_project)   # deterministic

        # the comp lives on the bus, never copied onto the member
        stem = await call(client, "analyze_stem", {"stem_id": "stem_a"})
        assert [e for e in stem["applied_chain"] if e["type"] == "comp"] == []
        bus = await call(client, "analyze_bus", {"bus_id": "drums"})
        (comp,) = [e for e in bus["applied_chain"] if e["type"] == "comp"]
        assert comp["mix"] == 0.5

        assert (await call(client, "undo_proposal", {"proposal_id": pid}))["ok"]
        assert await bounce_bytes(client, demo_project) == baseline
        # the created bus is gone and the member is back on the master
        state = await call(client, "get_state")
        assert state["buses"] == []
        assert {s["stem_id"]: s["bus"] for s in state["stems"]}["stem_a"] is None


@pytest.mark.anyio
async def test_route_rehomes_and_unroutes(demo_project):
    async with engine(demo_project) as client:
        # two busses, each from one stem
        await submit(client, [
            {"type": "group", "bus_id": "drums", "members": ["stem_a"]},
            {"type": "group", "bus_id": "synths", "members": ["stem_b"]},
        ], "two busses")

        # re-home stem_b onto drums, then send it straight to the master
        await submit(client, [{"type": "route", "stem_id": "stem_b",
                               "bus_id": "drums"}], "move b to drums")
        routing = {s["stem_id"]: s["bus"]
                   for s in (await call(client, "get_state"))["stems"]}
        assert routing == {"stem_a": "drums", "stem_b": "drums"}

        await submit(client, [{"type": "route", "stem_id": "stem_b",
                               "bus_id": None}], "b back to master")
        routing = {s["stem_id"]: s["bus"]
                   for s in (await call(client, "get_state"))["stems"]}
        assert routing == {"stem_a": "drums", "stem_b": None}


@pytest.mark.anyio
async def test_route_into_same_proposal_bus(demo_project):
    """`route` may target a bus a `group` created earlier in the proposal."""
    async with engine(demo_project) as client:
        await submit(client, [
            {"type": "group", "bus_id": "drums", "members": ["stem_a"]},
            {"type": "route", "stem_id": "stem_b", "bus_id": "drums"},
        ], "group a, route b into it")
        (drums,) = (await call(client, "get_state"))["buses"]
        assert drums["members"] == ["stem_a", "stem_b"]


@pytest.mark.anyio
async def test_group_route_validation(demo_project):
    async with engine(demo_project) as client:
        async def rules(action: dict) -> list[str]:
            result = await call(client, "propose_actions", {"req": {
                "rationale": "rule probe", "predicted_impact": [],
                "actions": [action]}})
            assert not result["accepted"]
            return [v["rule"] for v in result["violations"]]

        # a bus may not be named after a stem or 'master' (shared namespace)
        assert "bus_name_taken" in await rules(
            {"type": "group", "bus_id": "stem_a", "members": ["stem_b"]})
        assert "bus_name_taken" in await rules(
            {"type": "group", "bus_id": "master", "members": ["stem_a"]})
        # members must be real stems
        assert "unknown_stem" in await rules(
            {"type": "group", "bus_id": "drums", "members": ["nope"]})
        # route needs a real stem and a real destination bus
        assert "unknown_stem" in await rules(
            {"type": "route", "stem_id": "nope", "bus_id": None})
        assert "unknown_bus" in await rules(
            {"type": "route", "stem_id": "stem_a", "bus_id": "ghost"})

        # group carries no gain — it must not trip the gain budget, so a group
        # bundled with a legal full-budget move on a member still applies
        ok = await call(client, "propose_actions", {"req": {
            "rationale": "group + a 6 dB lift", "predicted_impact": [],
            "actions": [
                {"type": "group", "bus_id": "drums", "members": ["stem_a"]},
                {"type": "gain", "stem_id": "stem_a", "delta_db": 6.0}]}})
        assert ok["accepted"], ok["violations"]
