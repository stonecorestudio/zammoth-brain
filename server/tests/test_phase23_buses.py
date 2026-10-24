"""Phase 23 (group busses + parallel compression) under Phase 28 routing.

Stems route into named busses (member sum -> bus gain -> bus chain ->
master). Routing is the agent's now: busses are created and stems routed
through group/route proposals (the mechanics live in test_phase28_grouping).
This file covers what rides on top of an existing bus — the render path, the
bus target rules, parallel (NY) compression via CompAction.mix, analyze_bus —
plus the one surviving *user* capability on a bus: its fader (volume).
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


async def apply_actions(client, actions: list[dict],
                        rationale: str = "bus setup") -> str:
    """Propose + apply one bundle (the agent's only way to touch routing)."""
    proposal = await call(client, "propose_actions", {"req": {
        "rationale": rationale, "predicted_impact": [], "actions": actions}})
    assert proposal["accepted"], proposal["violations"]
    pid = proposal["proposal_id"]
    assert (await call(client, "apply_proposal", {"proposal_id": pid}))["ok"]
    return pid


@pytest.mark.anyio
async def test_bus_state_analyze_and_fader(demo_project):
    async with engine(demo_project) as client:
        await apply_actions(client, [{"type": "group", "bus_id": "drums",
                                      "members": ["stem_a"]}])

        # the bus + its routing show in state
        state = await call(client, "get_state")
        (drums,) = state["buses"]
        assert drums["bus_id"] == "drums" and drums["members"] == ["stem_a"]
        assert {s["stem_id"]: s["bus"]
                for s in state["stems"]}["stem_a"] == "drums"

        # analyze_bus hears the routed member through the gain-only bus chain
        bus = await call(client, "analyze_bus", {"bus_id": "drums"})
        assert bus["applied_chain"] == [{"type": "gain", "gain_db": 0.0}]
        assert bus["lufs_s"] is not None

        # the bus FADER is still a user move — volume stays manual (Phase 28)
        edited = await call(client, "set_stem",
                            {"stem_id": "drums", "gain_db": -3.0})
        assert edited["gain_db"] == -3.0


@pytest.mark.anyio
async def test_bus_target_rules(demo_project):
    async with engine(demo_project) as client:
        await apply_actions(client, [{"type": "group", "bus_id": "drums",
                                      "members": ["stem_a"]}])

        async def violations(action: dict) -> list[str]:
            result = await call(client, "propose_actions", {"req": {
                "rationale": "rule probe", "predicted_impact": [],
                "actions": [action]}})
            assert not result["accepted"]
            return [v["rule"] for v in result["violations"]]

        # the bus palette is the stem palette minus duck, whole-file only,
        # under the same guardrails as a stem
        assert "bus_action_type" in await violations(
            {"type": "duck", "stem_id": "drums", "key_stem_id": "stem_b",
             "threshold_db": -30.0, "depth_db": 4.0})
        assert "bus_action_type" in await violations(
            {"type": "limiter", "stem_id": "drums"})
        # …but a CLIPPER is allowed on a bus since 2026-09-08 (device audit 05-F3): it was
        # already in _BUS_KINDS and the bus renderer, and addable from the device dock, so a
        # bus clipper rendered perfectly well and only this path refused it. Asserted as an
        # ACCEPTANCE, which `violations` cannot express — it requires a rejection.
        clip_result = await call(client, "propose_actions", {"req": {
            "rationale": "bus clipper", "predicted_impact": [],
            "actions": [{"type": "clip", "stem_id": "drums",
                         "threshold_db": -3.0, "hardness": 0.5}]}})
        assert clip_result["accepted"], (
            "a bus clipper must be accepted: "
            + str([v["rule"] for v in clip_result["violations"]]))
        assert "bus_ride_unsupported" in await violations(
            {"type": "comp", "stem_id": "drums", "threshold_db": -30.0,
             "ratio": 2.0, "attack_ms": 10.0, "release_ms": 100.0,
             "time_range_s": [0.0, 2.0]})
        assert "gain_per_stem_per_pass_db" in await violations(
            {"type": "gain", "stem_id": "drums", "delta_db": 9.0})


@pytest.mark.anyio
async def test_parallel_comp_mix_blend(demo_project):
    async with engine(demo_project) as client:
        baseline = await bounce_bytes(client, demo_project)
        comp = {"type": "comp", "stem_id": "stem_a", "threshold_db": -35.0,
                "ratio": 6.0, "attack_ms": 1.0, "release_ms": 60.0}

        async def render_with(action: dict) -> bytes:
            proposal = await call(client, "propose_actions", {"req": {
                "rationale": "mix-law probe", "predicted_impact": [],
                "actions": [action]}})
            assert proposal["accepted"], proposal["violations"]
            pid = proposal["proposal_id"]
            assert (await call(client, "apply_proposal",
                               {"proposal_id": pid}))["ok"]
            rendered = await bounce_bytes(client, demo_project)
            assert (await call(client, "undo_proposal",
                               {"proposal_id": pid}))["ok"]
            return rendered

        fully_wet = await render_with({**comp, "mix": 1.0})
        dry = await render_with({**comp, "mix": 0.0})
        half = await render_with({**comp, "mix": 0.5})

        assert dry == baseline  # mix: 0 renders bit-exactly dry
        assert fully_wet != baseline
        assert half not in (dry, fully_wet)  # a real in-between blend


@pytest.mark.anyio
async def test_analyze_bus_errors(demo_project):
    async with engine(demo_project) as client:
        with pytest.raises(Exception):
            await call(client, "analyze_bus", {"bus_id": "nope"})

        # route the only member away (route to None = back to master) and the
        # bus has nothing to analyze
        await apply_actions(client, [{"type": "group", "bus_id": "empty",
                                      "members": ["stem_b"]}])
        await apply_actions(client, [{"type": "route", "stem_id": "stem_b",
                                      "bus_id": None}])
        with pytest.raises(Exception):  # no members left
            await call(client, "analyze_bus", {"bus_id": "empty"})


# ---------- the bus strip's own scalars: pan / width / mute (2026-09-05) ----------
# The DAW's group strip always offered the pan knob, the W field and [M]; they moved the live
# native graph while the brain had nowhere to keep them, so every move was refused ("a bus
# accepts gain_db only") and the next chains push snapped the control home. They are the bus's
# channel-strip scalars now, exactly as on a stem — trailing stages in effective_bus_chain, so
# the offline render, the native node and the export oracle read one truth.

@pytest.mark.anyio
async def test_bus_strip_scalars_are_user_edits(demo_project):
    async with engine(demo_project) as client:
        await apply_actions(client, [{"type": "group", "bus_id": "drums",
                                      "members": ["stem_a"]}])

        edited = await call(client, "set_stem", {"stem_id": "drums", "pan": -0.5})
        assert edited["pan"] == -0.5
        edited = await call(client, "set_stem", {"stem_id": "drums", "width": 1.4})
        assert edited["width"] == 1.4
        edited = await call(client, "set_stem", {"stem_id": "drums", "muted": True})
        assert edited["muted"] is True

        # clamped like a stem's, not silently accepted out of range
        assert (await call(client, "set_stem",
                           {"stem_id": "drums", "pan": 4.0}))["pan"] == 1.0

        # …and the fields a bus genuinely doesn't have still refuse
        with pytest.raises(Exception):
            await call(client, "set_stem", {"stem_id": "drums", "excluded": True})


def test_bus_pan_and_width_are_trailing_stages():
    from mix_engine.session import effective_bus_chain

    assert effective_bus_chain({"gain_db": 0.0, "chain": []}) == [
        {"type": "gain", "gain_db": 0.0}]           # absent at default — legacy buses unchanged
    chain = effective_bus_chain({"gain_db": 0.0, "pan": -0.5, "width": 1.5,
                                 "chain": [{"type": "eq", "band": {}}]})
    # WIDTH then PAN — the order the deliverable runs, and not cosmetic: natively the
    # MonitorChain (which carries width) is the FIRST plugin on the group track and Tracktion's
    # VolumeAndPan is the LAST, and an M/S width does not commute with a per-channel pan. The
    # other half of the match is the pan LAW (daw/patches/0007's PanLawBalance). 2026-09-05.
    assert [e["type"] for e in chain] == ["gain", "eq", "width", "pan"]


def test_a_muted_bus_is_silent_in_the_mix(demo_project):
    """Muting a group silences it in the bounce — nothing into the master, nothing out of
    its sends — the mixdown-level filter a muted stem already gets."""
    import numpy as np

    from mix_engine.dsp import render_master
    from mix_engine.session import Session

    s = Session(demo_project)
    s.refresh_stems()
    bus_id = s.group_stems(sorted(s.stems))["bus_id"]   # every stem into one group

    loud, _ = render_master(s)
    assert float(np.abs(loud).max()) > 1e-4

    s.user_edit(bus_id, {"muted": True})
    silent, _ = render_master(Session(demo_project))
    assert float(np.abs(silent).max()) < 1e-9


def test_a_panned_bus_moves_the_mix(demo_project):
    """The bus pan is audible in the offline render (it was UI-only before 2026-09-05)."""
    import numpy as np

    from mix_engine.dsp import render_master
    from mix_engine.session import Session

    s = Session(demo_project)
    s.refresh_stems()
    bus_id = s.group_stems(sorted(s.stems))["bus_id"]

    centred, _ = render_master(s)
    assert float(np.abs(centred).max()) > 1e-4

    s.user_edit(bus_id, {"pan": -1.0})
    hard_left, _ = render_master(Session(demo_project))

    # a mono group upmixes when panned, exactly as a panned stem does
    assert hard_left.shape[1] == 2
    assert float(np.abs(hard_left[:, 1]).max()) < 1e-12          # right is empty
    assert float(np.abs(hard_left[:, 0]).max()) > 1e-4           # left still carries the group

    # …and the LAW, at a mid pan. Hard left satisfies BOTH the balance law and Tracktion's
    # linear law, so the assertions above could not see the divergence the DAW was bouncing
    # (PanLawLinear boosts the near channel to 2.0 = +6 dB; _pan clamps it to unity). Pin the
    # near channel at unity and the far one at 1-|p|, which is what daw/patches/0007 makes the
    # native side compute too — the two engines now agree on a panned channel. 2026-09-05.
    s2 = Session(demo_project)
    s2.refresh_stems()
    b2 = s2.group_stems(sorted(s2.stems))["bus_id"]
    ref, _ = render_master(s2)
    ref_peak = float(np.abs(ref).max())
    s2.user_edit(b2, {"pan": 0.5})
    half_right, _ = render_master(Session(demo_project))
    near = float(np.abs(half_right[:, 1]).max())
    far = float(np.abs(half_right[:, 0]).max())
    assert near == pytest.approx(ref_peak, rel=1e-3), "the NEAR channel must not be boosted"
    assert far == pytest.approx(ref_peak * 0.5, rel=1e-3), "the far channel attenuates by 1-|pan|"


def test_a_muted_bus_is_silent_out_of_its_sends(demo_project):
    """The mute is a mixdown filter BEFORE add_sends, so a muted group feeds no return either.

    The existing mute case has no returns in its fixture, so the `add_sends` half of the claim
    was never exercised (2026-09-05 review).
    """
    import numpy as np

    from mix_engine.dsp import render_master
    from mix_engine.session import Session

    s = Session(demo_project)
    s.refresh_stems()
    bus_id = s.group_stems(sorted(s.stems))["bus_id"]
    ret = s.add_return("reverb")["return_id"]
    s.set_send(bus_id, ret, 0.0)                     # unity send off the group

    wet, _ = render_master(s)
    assert float(np.abs(wet).max()) > 1e-4

    s2 = Session(demo_project)
    s2.user_edit(bus_id, {"muted": True})
    muted, _ = render_master(s2)
    assert float(np.abs(muted).max()) < 1e-9, "a muted group still fed its return"


def test_muting_a_parent_silences_the_subtree(demo_project):
    """Muting a PARENT bus silences the children summed into it — the claim docs/10-daw.md
    makes and no case covered (2026-09-05 review)."""
    import numpy as np

    from mix_engine.dsp import render_master
    from mix_engine.session import Session

    s = Session(demo_project)
    s.refresh_stems()
    stems = sorted(s.stems)
    child = s.group_stems(stems[:1])["bus_id"]      # bus1
    parent = s.group_stems(stems[1:])["bus_id"]     # bus2
    s.route_bus(child, parent)                      # bus1's processed sum feeds bus2

    loud, _ = render_master(s)
    assert float(np.abs(loud).max()) > 1e-4

    # …mute the PARENT only. Its own members go with it, and so must the child summed into it.
    s.user_edit(parent, {"muted": True})
    s2 = Session(demo_project)
    assert not s2.buses[child].get("muted"), "the child must stay unmuted — the parent carries it"
    silent, _ = render_master(s2)
    assert float(np.abs(silent).max()) < 1e-9, "muting the parent did not silence its subtree"



def test_rename_bus_carries_every_reference_with_it(demo_project):
    """A bus id IS its displayed name — the DAW paints it on the strip, the head card, the rail
    and both routing menus — so renaming one has to MOVE the id, and the id is a reference.
    Members point at it, a nested bus's output points at it, the proposal ledger names it, its
    inverse snapshot is keyed by it, and a journaled co-edit carries it as `bus:<id>`. A
    reference left behind is not a stale label: it is a member that has silently stopped
    belonging to the group you just named."""
    from mix_engine.session import Session
    from mix_engine.models import ProposalRequest

    s = Session(demo_project)
    s.group_stems(["stem_a", "stem_b"])
    s = Session(demo_project)
    bus = next(iter(s.buses))
    child = s.group_stems(["stem_a"])["bus_id"]        # a second bus…
    s = Session(demo_project)
    s.route_bus(child, bus)                            # …nested into the first
    s.add_return("reverb")
    s = Session(demo_project)
    ret = next(iter(s.returns))
    s.set_send(bus, ret, -9.0)
    prop = s.propose(ProposalRequest(
        rationale="glue",
        predicted_impact=[{"metric": "master.lufs_i", "predicted_delta": 0.5, "unit": "LUFS"}],
        actions=[{"type": "comp", "stem_id": bus, "threshold_db": -18.0, "ratio": 2.0,
                  "attack_ms": 20.0, "release_ms": 200.0},
                 {"type": "gain", "stem_id": bus, "delta_db": 4.0}]))
    assert prop.accepted and s.apply(prop.proposal_id).ok
    s = Session(demo_project)
    s._state.setdefault("coedit_log", []).append(
        {"url": f"http://127.0.0.1:1/api/set_send?source=bus:{bus}&return={ret}&gain_db=-9"})
    s._persist()

    order_before = list(Session(demo_project).buses)
    res = Session(demo_project).rename_bus(bus, "Drum Glue")
    assert res["renamed"] and res["bus_id"] == "Drum Glue" and res["was"] == bus

    s2 = Session(demo_project)
    assert "Drum Glue" in s2.buses and bus not in s2.buses
    assert s2.buses["Drum Glue"]["sends"][ret] == -9.0, "the bus kept its own state"
    assert [e["type"] for e in s2.buses["Drum Glue"]["chain"]] == ["comp"]
    assert s2.stems["stem_b"]["bus"] == "Drum Glue", "a member still points at its group"
    assert s2.buses[child]["output"] == "Drum Glue", "a nested bus still points at its parent"
    # …and the group keeps its PLACE: buses render and list in insertion order, so a rename that
    # popped and re-added would send the renamed group to the end of the mixer.
    assert [("Drum Glue" if b == bus else b) for b in order_before] == list(s2.buses)
    record = s2._state["proposals"][0]
    assert "Drum Glue" in record["inverse"] and bus not in record["inverse"]
    # …and the ACTIONS the ledger recorded, which name a bus through `stem_id` (only group /
    # route actions carry `bus_id`). This is the reference whose consequence is not cosmetic:
    # `_cumulative_gain` keys the agent's gain budget by stem_id, so a rename that left it
    # behind filed the 4 dB already spent under a bus that no longer exists AND handed the
    # agent a fresh wallet on the renamed one.
    assert {a["stem_id"] for a in record["actions"]} == {"Drum Glue"}
    assert s2._cumulative_gain(None).get("Drum Glue") == pytest.approx(4.0)
    assert bus not in s2._cumulative_gain(None)
    assert all(f"bus:{bus}" not in e["url"] for e in s2._state["coedit_log"])
    assert any("bus%3ADrum+Glue" in e["url"] or "bus:Drum Glue" in e["url"]
               for e in s2._state["coedit_log"])

    # the namespace is shared with stems/returns/the master, and ':' is the prefix separator
    for taken in ("master", "stem_a", child, "half:baked"):
        with pytest.raises(ValueError):
            Session(demo_project).rename_bus("Drum Glue", taken)
    with pytest.raises(ValueError):
        Session(demo_project).rename_bus("no such bus", "Whatever")
    # a no-op rename is not an error, and the DAW may hand either spelling of the id
    assert Session(demo_project).rename_bus("bus:Drum Glue", "Drum Glue")["renamed"] is False
