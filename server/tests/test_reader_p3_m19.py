"""Reader P3s, batch M19 - what the agent is shown, and the amber glow (2026-10-07).

The behaviour is driven by the console legs in `daw/Source/AuditReaderP3LegsM19.h`. This file
holds what a leg cannot: rules about the SOURCE that a later edit could quietly undo.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = (ROOT / "daw" / "Source" / "App.cpp").read_text(encoding="utf-8")
GLOW = (ROOT / "daw" / "Source" / "AgentGlow.h").read_text(encoding="utf-8")
ARRANGE = (ROOT / "daw" / "Source" / "ArrangeView.h").read_text(encoding="utf-8")
SESSION = (ROOT / "daw" / "Source" / "DawSession.h").read_text(encoding="utf-8")
SNAPSHOT = (ROOT / "daw" / "Source" / "EditSnapshot.h").read_text(encoding="utf-8")
COMMANDS_H = (ROOT / "daw" / "Source" / "DawCommands.h").read_text(encoding="utf-8")


def test_the_cached_snapshot_push_has_one_caller_besides_the_turn_start():
    """C4.R17. `pushEditSnapshot()` sends the CACHE: called after a change it sends the picture
    from before it. Two dozen sites did, under comments saying the agent was told at once. A
    site that changed something sets `editDirty`; the cached push is for a push that is OWED."""
    calls = [m.start() for m in re.finditer(r"(?<![\w>])pushEditSnapshot \(\);|->pushEditSnapshot \(\);|pushEditSnapshot\(\);", APP)]
    assert len(calls) == 1, f"{len(calls)} bare pushEditSnapshot() calls - a change sets editDirty instead"
    before = APP[calls[0] - 200: calls[0]]
    assert "pumpPendingPush" in before, "the one cached push left is the owed push the tick re-offers"


def test_no_comment_says_a_cached_push_shows_the_agent_the_change():
    for claim in ("make the agent aware of the new (empty) track at once", "the agent sees the new clip",
                  "the agent learns about the new track(s)", "the agent sees the master chain it will be reasoning about"):
        assert claim not in APP, claim


def test_the_halo_is_lit_after_the_command_answers_and_never_before():
    """E8.R9 b. The choke point touched the glow keys BEFORE invokeCommand: a refused command
    haloed its target. The one touch for a command is in glowForAgentCommand, called from the
    reply's own path."""
    perform = APP[APP.index("auto perform = [safe, id, command, args, reply, reqId]"):]
    perform = perform[: perform.index("c2->pushFreshEditSnapshot();")]
    invoke = perform.index("cc->invokeCommand (id, args)")
    assert "agentglow::touch" not in perform[:invoke], "the halo is lit before the command has run"
    assert "glowForAgentCommand (command, args, res, glowBefore)" in perform[invoke:]
    body = APP[APP.index("void glowForAgentCommand ("):]
    body = body[: body.index("\n    }\n")]
    assert body.index("if (! okay") < body.index("agentglow::touch"), "a refusal returns before anything is lit"


def test_a_clip_halo_is_keyed_by_the_clip_and_not_by_an_index():
    """C8.R12. `clip:<track>:<index>` named another clip after a delete, a move or a duplicate."""
    assert "getClips().indexOf (clip)" not in ARRANGE[ARRANGE.index("float clipGlowLevel ("): ARRANGE.index("float clipGlowLevel (") + 700]
    keys = GLOW[GLOW.index("inline juce::StringArray keysForCommand"):]
    keys = keys[: keys.index("\n}\n")]
    assert '"clip:"' not in keys, "keysForCommand builds no clip key: the host resolves the clip against the edit"


def test_every_track_delete_door_takes_the_tracks_racks():
    """D7.R9. A rack's TYPE lives in the edit's rack list; a deleted track left it there."""
    delete = APP[APP.index("void deleteTrack (juce::String id, bool freshTransaction = true)"):]
    delete = delete[: delete.index("void carryRenameOutsideEdit")]
    assert delete.index("dawsession::racksOnTrack (*track)") < delete.index("edit->deleteTrack (track)") < delete.index("dawsession::dropUnplacedRacks (*edit, racksHeld)")
    # the two realize doors that delete a group's or a return's track
    for m in re.finditer(r"for \(auto\* t : orphans\)\s*\{(.*?)\n\s*\}", SESSION, re.S):
        body = m.group(1)
        if "edit.deleteTrack (t)" in body:
            assert "racksOnTrack (*t)" in body and "dropUnplacedRacks (edit, held)" in body, body


def test_the_agents_rack_list_is_the_placed_racks():
    racks = SNAPSHOT[SNAPSHOT.index("// --- racks (Tier-2"):]
    racks = racks[: racks.index('root->setProperty ("racks"')]
    assert "rackIsPlaced (edit, *rt)" in racks


def test_a_deck_snapshot_entry_is_resolved_by_its_channel():
    """D3.R11. By name alone, a deleted channel's entry re-bound to the next track of that name."""
    recall = SESSION[SESSION.index("inline SnapshotRecall recallLauncherSnapshot"):]
    recall = recall[: recall.index("\n}\n")]
    assert recall.count("launcherSnapshotEntryTrack (edit, e)") == 2
    assert "getName() == name" not in recall and '.toString() == t->getName()' not in recall
    save = SESSION[SESSION.index("inline juce::String saveLauncherSnapshot"):]
    save = save[: save.index("\n}\n")]
    assert '"track_id", t->itemID.toString()' in save
    assert "launcherSnapshotsForAgent (edit)" in SNAPSHOT


def test_the_switches_that_read_back_say_so_in_the_catalog():
    """C4.R16. A switch the agent can press names the value it takes and where it reads back."""
    for key, words in {
        "grid.toggle_snap": ("on?", "arrange_view.snap"),
        "grid.toggle_triplet": ("on?", "arrange_view.triplet"),
        "grid.toggle_mode": ("mode?", "arrange_view.grid_mode"),
        "grid.narrower": ("arrange_view.grid",),
        "grid.wider": ("arrange_view.grid",),
        "automation.edit_toggle": ("on?", "arrange_view.automation_edit"),
        "view.track_delay": ("track_delay_shown",),
        "view.tooltips": ("Read tooltips",),
        "view.meter_history": ("meter_history",),
        "midi_learn.listen": ("midi_learn_listening",),
        "rack.preset_load": ("rack_presets",),
    }.items():
        row = re.search(r'\{ "' + re.escape(key) + r'", "((?:[^"\\]|\\.)*)"', COMMANDS_H)
        assert row, f"{key} has no grammar line"
        for w in words:
            assert w in row.group(1), f"{key}: the grammar line does not say {w!r}"
    view = APP[APP.index("void appendViewState (juce::var& snap)"):]
    view = view[: view.index("juce::var deckSelectionVar() const")]
    for field in ("gridStateInto (*av)", '"automation_edit"', '"track_delay_shown"', '"tooltips"', '"meter_history"',
                  '"midi_learn_listening"', '"rack_presets"', '"auditioning"'):
        assert field in view, f"appendViewState does not publish {field}"
