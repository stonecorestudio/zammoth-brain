"""A channel's device rows predate the turn's own mix moves until the turn ends (2026-10-29).

A proposal lands in the session at once, but reaches the DAW's live chain only when the reply
that ends the turn carries `chains`. Mid-turn, `read_track` handed the engineer the app's row
for a track it had just EQ'd and de-essed - `devices: []`, `device_count: 0`, the shape
`EditSnapshot.h` writes for an empty chain - and it told the user "the device dock shows no
processing on hats, snare, perc and keys ... the DAW may not be playing those chains yet",
which was false. The row now says why, and what will play; the live block says it too.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import anyio
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator import daw_bridge as B  # noqa: E402
from orchestrator.engine_client import in_memory_engine  # noqa: E402
from orchestrator.turn_host import LiveTurn  # noqa: E402


def _row(name: str, devices: list | None = None) -> dict:
    """A track row as `EditSnapshot.h` writes one: `devices` always published, empty included."""
    return {"name": name, "kind": "audio", "gain_db": 0.0, "width": 1.0,
            "device_count": len(devices or []), "devices": list(devices or [])}


_EQ = {"rationale": "lift presence", "predicted_impact": [],
       "actions": [{"type": "eq", "stem_id": "stem_a",
                    "band": {"freq_hz": 3000, "gain_db": 2.0, "q": 1.0, "shape": "peak"}}]}


def _turn(project: Path, edit: dict) -> LiveTurn:
    return LiveTurn(project, "mix it", use_llm=False, edit_state=lambda: edit)


def test_read_track_mid_turn_names_the_chain_the_dock_does_not_show_yet(demo_project):
    edit = {"tracks": [_row("stem_a"), _row("stem_b")]}
    turn = _turn(demo_project, edit)

    async def go():
        async with in_memory_engine(demo_project) as engine:
            turn.engine = engine
            assert (await turn.do_propose(_EQ))["accepted"] is True
            return (await turn.do_read_track("stem_a"), await turn.do_read_track("stem_b"))

    moved, untouched = anyio.run(go)
    track = moved["track"]
    assert track["devices"] == []                       # the app's row, as it stands
    assert track["chain_after_this_turn"] == ["eq"]     # ...and what will play
    assert "when this turn ends" in track["chain_note"]
    assert "chain_after_this_turn" not in untouched["track"]   # nothing moved there


def test_the_marker_goes_once_the_app_shows_the_chain(demo_project):
    edit = {"tracks": [_row("stem_a"), _row("stem_b")]}
    turn = _turn(demo_project, edit)

    async def go():
        async with in_memory_engine(demo_project) as engine:
            turn.engine = engine
            await turn.do_propose(_EQ)
            edit["tracks"][0] = _row("stem_a", [{"type": "eq", "occ": 0, "enabled": True}])
            return await turn.do_read_track("stem_a")

    assert "chain_after_this_turn" not in anyio.run(go)["track"]


def test_a_thinned_row_is_compared_by_its_device_types(demo_project):
    edit = {"tracks": [{"name": "stem_a", "device_types": ["eq"], "detail_omitted": True}]}
    turn = _turn(demo_project, edit)

    async def go():
        async with in_memory_engine(demo_project) as engine:
            turn.engine = engine
            await turn.do_propose(_EQ)
            return turn.chains_owed()

    assert anyio.run(go) == {}


def test_the_live_block_says_which_rows_predate_the_turn():
    live = B._chat_context_live({"seq": 3, "tracks": [_row("hats")]}, 0, {"hats": ["eq", "deess"]})
    assert live.startswith("NOTE: these channels' devices[] are the DAW's chain from before")
    assert '{"hats":["eq","deess"]}' in live
    assert "Live DAW edit (daw_edit_state):" in live
    assert not B._chat_context_live({"seq": 3, "tracks": [_row("hats")]}, 0, {}).startswith("NOTE")


@pytest.mark.parametrize("name,state,want", [
    ("hats", {"stems": {"hats": {"chain": [{"type": "gain"}, {"type": "eq"}]}}}, ("hats", ["eq"])),
    ("return:vox_verb", {"returns": {"vox_verb": {"chain": [{"type": "return_delay"}]}}},
     ("vox_verb", ["return_delay"])),
    ("master", {"master_chain": [{"type": "limiter"}]}, ("master", ["limiter"])),
    ("nope", {"stems": {}}, None),
])
def test_a_channel_is_found_in_the_session_as_the_snapshot_names_it(name, state, want):
    assert B._session_chain_types(state, name) == want
    json.dumps(want)
