"""What the user QA of 2026-10-08 still had open on 2026-10-09 - the DAW's half, pinned where a
leg cannot reach, or where the ORDER of two calls is the fix.

The behaviour is driven by console legs (daw/Source/QaLeftoverLegs.h): `--qa-time-leftovers`,
`--qa-arrange-leftovers`, `--qa-observations`, `--qa-spoken-painted`, `--qa-refreeze-group`. The
brain's half is test_qa_leftovers_2026_10_09.py.

Record: docs/audits/user-qa-2026-10-08/QA-log.md, *The leftovers*.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "daw" / "Source"
NATIVE = ROOT / "native" / "Source"
APP = (SRC / "App.cpp").read_text(encoding="utf-8")
MAIN = (SRC / "Main.cpp").read_text(encoding="utf-8")
SESSION_H = (SRC / "DawSession.h").read_text(encoding="utf-8")
ARRANGE_H = (SRC / "ArrangeView.h").read_text(encoding="utf-8")
NODE_H = (SRC / "MonitorChainPlugin.h").read_text(encoding="utf-8")
SNAPSHOT_H = (SRC / "EditSnapshot.h").read_text(encoding="utf-8")
STRIP_H = (SRC / "ChannelStrip.h").read_text(encoding="utf-8")
COMMANDS_H = (SRC / "DawCommands.h").read_text(encoding="utf-8")
DOCK_H = (SRC / "DeviceChainView.h").read_text(encoding="utf-8")
SPOKEN_H = (SRC / "SpokenZones.h").read_text(encoding="utf-8")
GRAPHS_H = (NATIVE / "DeviceGraphs.h").read_text(encoding="utf-8")

pytestmark = pytest.mark.reads_cpp


def _body(source: str, start: str, end: str = "\n    }\n") -> str:
    i = source.index(start)
    return source[i: source.index(end, i)]


# ---- the switch a red run was made with is gone -------------------------------------------------

def test_the_temporary_old_path_switch_is_not_in_the_tree():
    assert not (SRC / "LeftOld.h").exists()
    for folder in (SRC, NATIVE):
        for path in list(folder.glob("*.h")) + list(folder.glob("*.cpp")) + list(folder.glob("*.mm")):
            text = path.read_text(encoding="utf-8", errors="replace")
            assert "leftOld" not in text and "LeftOld.h" not in text, path.name
            assert "MIXAGENT_LEFT_OLD" not in text, path.name


def test_the_legs_are_wired():
    for flag, fn in (("--qa-time-leftovers", "runQaTimeLeftovers"), ("--qa-arrange-leftovers", "runQaArrangeLeftovers"),
                     ("--qa-observations", "runQaObservations"), ("--qa-spoken-painted", "runQaSpokenPainted"),
                     ("--qa-refreeze-group", "runQaRefreezeGroup"), ("--qa-midi-ports", "runQaMidiPorts"),
                     ("--qa-clip-live-pitch", "runQaClipLivePitch")):
        assert f'flag == "{flag}"' in MAIN and f"dawapp::{fn} (project)" in MAIN, flag


# ---- the time clipboard --------------------------------------------------------------------------

TIME_EDITS = ("void insertTimeRange (double a, double b)", "void deleteTimeRange (double a, double b)",
              "bool pasteTimeAt (double at)", "void duplicateTimeRange (double a, double b)")


def test_every_whole_timeline_edit_makes_the_markers_points_before_the_engine_moves_anything():
    """The engine makes a marker a clip one bar long and SPLITS it when an insert lands inside
    that bar: one marker became two. Inside the transaction, before the engine's call."""
    for start in TIME_EDITS:
        body = _body(ARRANGE_H, start)
        made = body.index("dawsession::makeMarkersPoints (edit)")
        assert body.index("dawsession::beginTransaction") < made, start
        engine = min(i for i in (body.find("te::insertSpaceIntoEdit"), body.find("te::deleteRegionOfTracks")) if i >= 0)
        assert made < engine, start
    assert "kMarkerLengthSeconds" in _body(APP, "case dawcmd::markerAdd:", "case dawcmd::markerRemove:")


def test_a_bend_is_straightened_before_anything_reads_or_moves_it():
    """A paste reads the host curve's value at the paste point, a duplicate reads what the curve
    did over the range: both AFTER the bend they cut has been written out, and all of it inside
    the transaction (the straightening writes points)."""
    dup = _body(ARRANGE_H, "void duplicateTimeRange (double a, double b)")
    assert dup.index("dawsession::beginTransaction") < dup.index("straightenEngineCurvesAt (edit, { a, b })") \
        < dup.index("collectCurveRepeats") < dup.index("te::insertSpaceIntoEdit")
    paste = _body(ARRANGE_H, "bool pasteTimeAt (double at)")
    assert paste.index("dawsession::beginTransaction") < paste.index("straightenEngineCurvesAt (edit, { at })") \
        < paste.index("prepareCurvePaste") < paste.index("te::insertSpaceIntoEdit")
    delete = _body(ARRANGE_H, "void deleteTimeRange (double a, double b)")
    assert delete.index("straightenEngineCurvesAt (edit, { a, b })") < delete.index("te::deleteRegionOfTracks")


def test_a_copy_straightens_a_list_of_its_own():
    """Copy Time is no edit: it may not write a point into a lane or a curve."""
    copy_lanes = _body(NODE_H, "std::vector<LaneSlice> copyRegionOfAutoLanes (double aS, double bS) const")
    assert "for (auto l : lanesCopy())" in copy_lanes and "swapLanesIn" not in copy_lanes
    copy_curves = _body(SESSION_H, "inline std::vector<CurveClip> copyRegionOfEveryCurve", "\n}\n")
    assert "curvedSegmentAt (all, cut)" in copy_curves
    assert "addPoint" not in copy_curves and "setCurveValue" not in copy_curves


def test_the_lane_edits_straighten_on_their_working_copy():
    for start in ("bool removeRegionFromAutoLanes (double aS, double bS)", "bool duplicateRegionInAutoLanes (double aS, double bS)"):
        body = _body(NODE_H, start)
        assert body.count("straightenAt (l.points, aS)") == 1 and body.count("straightenAt (l.points, bS)") == 1, start
    assert "straightenAt (l.points, atS)" in _body(NODE_H, "bool openAndPasteAutoLanes (")


def test_a_marquee_duplicate_repeats_the_lanes_automation():
    body = _body(ARRANGE_H, "bool duplicateSelectionSpan()")
    assert "dawsession::duplicateRegionInLanesOf (edit, *lanes[li], a, b)" in body
    assert "insertSpaceIntoAutoLanes" not in body           # the bare shift it replaced
    per_lane = _body(SESSION_H, "inline void duplicateRegionInLanesOf", "\n}\n")
    assert "node->duplicateRegionInAutoLanes (aS, bS)" in per_lane and "applyCurveRepeats (edit, repeats)" in per_lane
    assert "racksOnTrack (track)" in per_lane                # what a rack on the lane holds goes with it


def test_a_delete_does_not_close_a_racks_inner_curves_twice():
    """te::deleteRegionOfTracks closes up the plug-ins inside every rack that sits on a track; its
    insert reaches none of them. What the app closes by hand leaves those out."""
    body = _body(SESSION_H, "inline void removeRegionFromCurvesTheEngineLeaves", "\n}\n")
    assert "pluginsTheEnginesDeleteReaches (edit)" in body and "engineDid.contains (owner.param->getPlugin())" in body
    shift = _body(SESSION_H, "inline void shiftCurvesTheEngineLeaves", "\n}\n")
    assert "pluginsTheEnginesDeleteReaches" not in shift     # (an insert is not symmetrical: every rack curve is shifted by hand)


def test_markers_ride_the_time_clipboard_under_a_key_no_track_can_have():
    assert 'kMarkerTrackKey   = "::markers"' in ARRANGE_H
    assert "takeGlobal (edit.getMarkerTrack(), kMarkerTrackKey)" in _body(ARRANGE_H, "std::vector<ClipboardEntry> collectTimeEntries (double a, double b) const")
    place = _body(ARRANGE_H, "void placeTimeEntries (")
    assert "checkForDuplicates (*marker, false)" in place    # a copy gets a number of its own


# ---- the observations ----------------------------------------------------------------------------

def test_one_pan_readout_for_the_strip_the_snapshot_and_the_reply():
    assert "inline juce::String panText (double v) { return dawcmd::panReadout (v); }" in STRIP_H
    assert SNAPSHOT_H.count('"pan_shown", dawcmd::panReadout') == 2          # a track's row and a group's / return's
    handler = _body(APP, "case dawcmd::setStemPan:", "case dawcmd::setStemWidth:")
    assert 'o->setProperty ("shown", dawcmd::panReadout' in handler


def test_the_untouched_default_key_is_not_published_as_a_key():
    assert "if (! dawsession::keyIsUntouchedDefault (edit))" in SNAPSHOT_H
    view = _body(APP, "void appendViewState (juce::var& snap)")
    assert 'v->setProperty ("key", juce::var (ko))' in view
    for source in ('"detected"', '"set"', '"none"'):
        assert source in view, source
    # the same test the KEY cell's own text makes
    cell = _body(APP, "juce::String projectKeyText() const")
    assert "getPitch() == 60" in cell and "getNumPitches() <= 1" in cell
    rule = _body(SESSION_H, "inline bool keyIsUntouchedDefault", "\n}\n")
    assert "getPitch() == 60" in rule and "getNumPitches() <= 1" in rule


def test_a_routing_row_describes_what_was_asked():
    block = _body(APP, 'if (command == "routing.apply")', "c->applyChainsAndReadout")
    assert 'args.getProperty ("asked", juce::var())' in block
    assert "c->describeCommand (key, asked, verb, target, detail)" in block
    assert "toolFoldFor (key, asked, target)" in block and "keysForCommand (key, asked)" in block
    assert "(key, args" not in block                         # the payload: as DETAILS, as the fold's key, as what glows


# ---- spoken names --------------------------------------------------------------------------------

def test_every_text_a_device_face_draws_can_be_recorded():
    """The dock names a face's readouts from what the face PRINTS: a text drawn past faceText()
    is a readout no screen reader hears."""
    assert not re.search(r"\bg\.drawText \(", GRAPHS_H.split("inline void faceText", 1)[1].split("}", 1)[1])
    assert GRAPHS_H.count("faceText (g, ") > 350


def test_every_face_with_a_zone_map_says_what_is_at_a_point():
    classes = re.findall(r"^class (\w+)\s*: public Component[^\n]*public SettableTooltipClient([^\n]*)\n", GRAPHS_H, flags=re.M)
    assert len(classes) >= 19
    for name, rest in classes:
        assert "public FaceZones" in rest, name
    assert GRAPHS_H.count("String zoneTipAt (Point<int> at) const override { return tooltipAt (at); }") == len(classes)


def test_spoken_children_exist_only_while_an_assistive_client_runs():
    for source, start in ((ARRANGE_H, "void syncSpokenZones()"), ((SRC / "PianoRollView.h").read_text(encoding="utf-8"), "void syncSpokenZones()")):
        body = _body(source, start)
        assert "if (! dawui::spokenZonesWanted())" in body and "spoken.clear()" in body
        assert "if (gestureInFlight()) return;" in body      # a child that moves repaints the view under the hand
    faces = _body(DOCK_H, "void syncSpokenFaces()")
    assert "dawui::spokenZonesWanted()" in faces and "spokenFace.clear()" in faces
    # the extra paint a face's readouts cost is behind the same question
    assert "if ((tick % 30) == 0) syncSpokenFaces();" in DOCK_H


def test_a_zone_follows_an_owner_that_was_rebuilt_and_a_readout_is_not_a_button():
    assert "if (kid->getParentComponent() != &owner) owner.addAndMakeVisible (kid);" in SPOKEN_H
    assert "juce::AccessibilityRole::staticText" in SPOKEN_H


# ---- found by driving the real app with CoreMIDI ports -------------------------------------------

def test_a_change_of_clock_or_timecode_rebuilds_the_playback_graph():
    """The port that sends clock is a node of the graph, put there when the graph is built: the
    engine's setters rebuild nothing. Not under a take (a rebuild ends one)."""
    body = _body(SESSION_H, "inline int setMidiSync (te::Engine& engine", "\n}\n")
    assert "graphOwed = true" in body
    assert "tc->edit.restartPlayback()" in body and "! tc->isRecording()" in body


def test_a_surface_bound_while_a_project_is_open_is_told_the_edit():
    handler = _body(APP, "case dawcmd::controlSurfaceSet:", "case dawcmd::oscSet:")
    bound = handler.index("dawsession::setControlSurface (engine, name, enabled, midiIn, midiOut)")
    assert bound < handler.index("dawsession::detachControlSurfaces (engine, edit.get())") \
        < handler.index("dawsession::attachControlSurfaces (engine, *edit, &selectionManager)")


def test_the_watch_tells_a_surface_that_became_active_by_itself():
    """At a launch the MIDI ports are listed seconds after the project opens: a surface bound in
    the last session is not active when the edit is set, and is never told. The project on screen
    only."""
    assert "if (! inBackground && dawsession::surfacesMissingTheEdit (engine, *edit))" in APP
    rule = _body(SESSION_H, "inline bool surfacesMissingTheEdit", "\n}\n")
    assert "c->isEnabled()" in rule and "c->getEdit() != &edit" in rule
