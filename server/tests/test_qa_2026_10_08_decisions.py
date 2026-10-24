"""User QA 2026-10-08, the decisions James handed over ("take care of these using your best
judgment"): what was changed, pinned where a leg cannot reach (the application object's menu),
and the manual's sentences beside the code they describe.

The behaviour itself is driven by console legs: `--qa-arrange-smalls` (i2) (j) (k) (l),
`--drop-hint-returns` (c) (c1), `--qa-lanes` (b3).

Record: docs/audits/user-qa-2026-10-08/QA-log.md, *The decisions*.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANUAL = (ROOT / "docs" / "17-user-manual.md").read_text(encoding="utf-8")
APP = (ROOT / "daw" / "Source" / "App.cpp").read_text(encoding="utf-8")
COMMANDS_H = (ROOT / "daw" / "Source" / "DawCommands.h").read_text(encoding="utf-8")
SESSION_H = (ROOT / "daw" / "Source" / "DawSession.h").read_text(encoding="utf-8")
ARRANGE_H = (ROOT / "daw" / "Source" / "ArrangeView.h").read_text(encoding="utf-8")
SNAPSHOT_H = (ROOT / "daw" / "Source" / "EditSnapshot.h").read_text(encoding="utf-8")
GLOW_H = (ROOT / "daw" / "Source" / "AgentGlow.h").read_text(encoding="utf-8")
TRANSPORT_H = (ROOT / "daw" / "Source" / "TransportBar.h").read_text(encoding="utf-8")

pytestmark = pytest.mark.reads_cpp

FLAT = re.sub(r"\s+", " ", MANUAL)


def _body(source: str, start: str, end: str = "\n    }\n") -> str:
    i = source.index(start)
    return source[i: source.index(end, i)]


def test_the_menu_bar_has_an_edit_menu_between_file_and_view():
    names = re.search(r"getMenuBarNames\(\) override \{ return \{ ([^}]+) \}; \}", APP).group(1)
    assert [n.strip().strip('"') for n in names.split(",")] == ["File", "Edit", "View", "Help"]
    # the index each branch answers moved with it: a Help that still asks for 2 would be the View menu
    assert "topLevelMenuIndex == 1)   // Edit" in APP
    assert "topLevelMenuIndex == 2)   // View" in APP
    assert "topLevelMenuIndex == 3)   // Help" in APP


def test_every_edit_menu_row_is_a_command_the_project_has():
    rows = _body(APP, "static const std::vector<EditMenuRow>& editMenuRows()")
    pairs = re.findall(r'\{ "([^"]+)", dawcmd::(\w+) \}', rows)
    assert [w for w, _ in pairs] == ["Undo", "Redo", "Cut", "Copy", "Paste", "Duplicate", "Delete", "Select All"]
    for _, command in pairs:
        assert re.search(r"\{ %s,\s+\"" % command, COMMANDS_H), f"dawcmd::{command} is not in the command table"
    for words, _ in pairs:
        assert words in _menus_section(), f"the manual's Edit menu does not list {words}"


def _menus_section() -> str:
    i = MANUAL.index("## The menus")
    return re.sub(r"\s+", " ", MANUAL[i: MANUAL.index("\n## ", i + 5)])


def test_an_edit_menu_item_presses_its_key():
    """Undo, Copy, Delete mean different things to a text field, the piano roll and the
    arrangement, and each already answers the KEY. An item that invoked the command directly
    would copy a CLIP while the composer has the cursor."""
    press = _body(APP, "void pressChordOf (int cmdId)")
    assert "getKeyPressesAssignedToCommand" in press and "peer->handleKeyPress (keys.getLast())" in press
    edit = _body(APP, "topLevelMenuIndex == 1)   // Edit", "\n        }\n")
    assert "pressChordOf (command)" in edit and "invokeChecked" not in edit
    # ...and its key is the menu's own column, not text in the title (seen in the window,
    # 2026-10-09: "Undo<tab>Cmd-Z" drew the chord inside the name). JUCE's native menu takes
    # the key from the item's command manager, and runs `action` in preference to the command.
    assert "item.commandManager = &d->commandsForMenu();" in edit and "item.action = " in edit
    assert "menuTextWithChord" not in edit
    assert "does exactly what its key does" in _menus_section()


def test_the_app_menu_has_an_about_item():
    assert 'appleExtras.addItem ("About Zammoth"' in APP
    assert "juce::MenuBarModel::setMacMainMenu (this, &appleExtras);" in APP
    about = _body(APP, "static juce::AlertWindow* showAboutAlert (", "\n}\n")
    assert 'aw->setTitle ("About Zammoth");' in about and "aw->setDescription (words);" in about
    assert "About Zammoth" in _menus_section()


def test_the_discontinued_plug_in_is_filtered_at_every_door_that_hosts_by_name():
    assert "inline bool isDiscontinuedOwnPlugin (const juce::PluginDescription& d)" in SNAPSHOT_H
    assert "const auto types = hostablePluginTypes (allTypes);" in SNAPSHOT_H
    for label in ("std::vector<FileBrowserPanel::PluginInfo> availablePluginInfos() const",
                  "case dawcmd::pluginAdd:", "case dawcmd::rackCreate:",
                  "case dawcmd::padDeviceAdd:", "case dawcmd::rackPluginAdd:"):
        i = APP.index(label)
        nxt = APP.index("knownPluginList.getTypes()", i)
        line = APP[APP.rindex("\n", 0, nxt): APP.index("\n", nxt)]
        assert "editsnapshot::hostablePluginTypes (" in line, f"{label} reads the whole plug-in list"


def test_the_painted_keys_have_names():
    assert "inline juce::String spokenKeyName (const juce::String& glyph, bool ring = false)" in GLOW_H
    assert "setTitle (spokenKeyName (glyph));" in GLOW_H
    assert "setTitle (spokenKeyName (label, mark == Mark::ring));" in GLOW_H
    for key in ("rtzButton", "playButton", "stopButton", "recButton", "loopButton", "punchButton", "metroButton"):
        assert re.search(r"%s\.setTitle \(\"[A-Z]" % key, TRANSPORT_H), f"{key} has no spoken name"


def test_a_vca_is_made_where_its_first_member_is():
    body = _body(SESSION_H, "inline te::FolderTrack* createVca (", "\n}\n")
    assert "top.indexOf (members.getFirst())" in body
    assert "edit.insertNewFolderTrack (at, nullptr, false)" in body
    assert "they are drawn together, where the first of them was" in FLAT


def test_the_pointer_on_a_return_lane_takes_no_clip():
    """The row is asked of the middle of the clip's body, half a lane below a hand on the title
    band; with the pointer on the last return's lane that is past the band's foot, which reads
    as "below everything". Below everything still makes a track (`--drop-below`)."""
    body = _body(ARRANGE_H, "    int rowForMove (int pointerY, int bodyY) const")
    assert "if (layout.laneAt (pointerY) >= stemLaneCount()" in body and "return rowAt (bodyY);" in body
    assert "moveTargetLane = rowForMove (e.y, e.y + moveLaneGrabOffsetY);" in ARRANGE_H
    assert "noteMoveTargetLane = rowForMove (e.y, e.y + noteMoveGrabOffsetY);" in ARRANGE_H
    assert "with the pointer, or the middle of the clip, over one" in FLAT


def test_a_drawn_step_ends_where_it_ends():
    body = _body(ARRANGE_H, "    void drawAddPoint (const juce::MouseEvent& e)")
    assert "insertPt (strokeHi, drawOrig.empty() ? drawOrigRest : interpPts (drawOrig, strokeHi));" in body
    start = _body(ARRANGE_H, "    bool beginAutoDraw (const juce::MouseEvent& e, int li)")
    assert "drawHiT = -1.0;" in start
    assert "**Nor does what plays after it**" in MANUAL


def test_the_report_dialog_does_not_promise_a_listener():
    """There is no support address in the product. Until there is one, the dialog says what a
    useful report holds and does not say "Tell us"."""
    body = _body(APP, "    void showReportProblem()")
    assert "Tell us" not in body and "Copy Diagnostics puts them on" in body


def test_the_manual_says_a_key_can_read_as_its_relative():
    # (2026-10-09: no longer a bare limit - a near-tie is settled by the bass, analysis v10 -
    # and the manual says both what decides and where it can still read the relative)
    assert "the **bass line decides**" in FLAT
    assert "a tune in A minor that leans on C chords can still read C major" in FLAT
