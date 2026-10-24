"""The limits the user QA of 2026-10-08 left, taken care of on 2026-10-09 - the DAW's half, pinned
where a leg cannot reach or where the ORDER of two calls is the fix.

The behaviour is driven by console legs (daw/Source/QaLimitsLegs.h): `--qa-paste-curves`,
`--qa-refreeze`, `--qa-spoken`, `--qa-routing-guard`, and the exact-meter cases of
`--qa-limiter-dense`. The brain's half is test_qa_limits_2026_10_09.py.

Record: docs/audits/user-qa-2026-10-08/QA-log.md, *The limits, taken care of*.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "daw" / "Source"
APP = (SRC / "App.cpp").read_text(encoding="utf-8")
MAIN = (SRC / "Main.cpp").read_text(encoding="utf-8")
SESSION_H = (SRC / "DawSession.h").read_text(encoding="utf-8")
ARRANGE_H = (SRC / "ArrangeView.h").read_text(encoding="utf-8")
STRIP_H = (SRC / "ChannelStrip.h").read_text(encoding="utf-8")
HEADS_H = (SRC / "TrackHeadColumn.h").read_text(encoding="utf-8")
BROWSER_H = (SRC / "FileBrowserPanel.h").read_text(encoding="utf-8")
DECK_H = (SRC / "LauncherPanel.h").read_text(encoding="utf-8")
SPOKEN_H = (SRC / "SpokenZones.h").read_text(encoding="utf-8")
LEGS_H = (SRC / "QaLimitsLegs.h").read_text(encoding="utf-8")
BRIDGE = (ROOT / "orchestrator" / "daw_bridge.py").read_text(encoding="utf-8")

pytestmark = pytest.mark.reads_cpp


def _body(source: str, start: str, end: str = "\n    }\n") -> str:
    i = source.index(start)
    return source[i: source.index(end, i)]


# ---- the switch a red run was made with is gone -------------------------------------------------

def test_the_temporary_old_path_switch_is_not_in_the_tree():
    assert not (SRC / "QaLimitsOldSwitch.h").exists()
    for path in list(SRC.glob("*.h")) + list(SRC.glob("*.cpp")) + list(SRC.glob("*.mm")):
        text = path.read_text(encoding="utf-8", errors="replace")
        assert "qaLimitsOld" not in text and "QaLimitsOldSwitch" not in text, path.name


# ---- Paste Time and the engine's own curves ------------------------------------------------------

def test_copy_time_takes_the_engines_curves_and_the_clipboard_survives_a_view_rebuild():
    body = _body(ARRANGE_H, "void copyTimeRange (double a, double b)")
    assert "dawsession::copyRegionOfEveryCurve (edit, a, b)" in body
    carried = _body(ARRANGE_H, "struct CarriedViewState", "void restoreCarriedViewState")
    assert "timeClipCurves" in carried
    assert "timeClipCurves  = s.timeClipCurves;" in ARRANGE_H


def test_a_paste_reads_where_each_curve_stands_before_the_engine_opens_the_gap():
    """The island's edges are the destination curve's own value at the paste point. Read after
    te::insertSpaceIntoEdit, that value is the one the engine has already moved."""
    body = _body(ARRANGE_H, "bool pasteTimeAt (double at)")
    prepare = body.index("dawsession::prepareCurvePaste (edit, timeClipCurves, at)")
    opened = body.index("te::insertSpaceIntoEdit")
    lanes = body.index("dawsession::pasteRegionIntoEveryLane")
    applied = body.index("dawsession::applyCurvePaste (edit, curvePaste, at, timeClipLen)")
    assert prepare < opened < lanes < applied


def test_every_whole_timeline_edit_moves_the_curves_the_engine_leaves():
    """The master's and a rack's (its macros, the plug-ins inside it): one helper, at every door."""
    for start in ("inline void insertSpaceIntoEveryLane", "inline void pasteRegionIntoEveryLane",
                  "inline void duplicateRegionInEveryLane"):
        assert "shiftCurvesTheEngineLeaves (edit," in _body(SESSION_H, start, "\n}\n"), start
    assert "removeRegionFromCurvesTheEngineLeaves (edit, aS, bS)" in _body(
        SESSION_H, "inline void removeRegionFromEveryLane", "\n}\n")
    owners = _body(SESSION_H, "inline std::vector<CurveOwner> everyEngineCurveParam", "\n}\n")
    assert "edit.getRackList().getTypes()" in owners and "getMacroParameters()" in owners
    assert "edit.getMasterPluginList().getPlugins()" in owners


def test_a_rack_chains_lanes_ride_the_time_clipboard_under_a_name_no_track_can_have():
    assert 'kLaneClipNodePrefix = "::node:"' in SESSION_H        # a track's name holds no ':'
    assert "take (kLaneClipNodePrefix + node->itemID.toString())" in SESSION_H


# ---- a freeze the load took away ----------------------------------------------------------------

def test_both_doors_that_load_an_edit_ask_which_freezes_the_load_took_away():
    assert "dawsession::loadEditForProject (engine, sessionFile, project, &freezesOwed)" in APP
    assert "dawsession::loadEditForProject (engine, f, project, &lostFreezes)" in APP
    assert len(re.findall(r"dawsession::loadEditForProject \(", APP)) == 2
    assert APP.count("scheduleRefreezeLostAtLoad();") >= 2      # the constructor's settle, the edit switch


def test_the_refreeze_waits_for_the_brains_first_reply_and_is_posted_never_called_from_a_timer():
    body = _body(APP, "void watchFreezesOwed()")
    assert "brainComing && ! brainStemsKnown && waited < 20000" in body
    assert "runThread.isThreadRunning()" in body
    assert "juce::MessageManager::callAsync" in body and "refreezeLostAtLoad();" in body
    # the only caller of the work itself is that posted message
    assert APP.count("c->refreezeLostAtLoad();") == 1
    assert "callAfterDelay" not in _body(APP, "void scheduleRefreezeLostAtLoad()")


def test_a_track_is_frozen_again_through_the_one_door_every_freeze_goes_through():
    body = _body(APP, "void refreezeLostAtLoad()")
    assert "invokeCommand (dawcmd::trackFreeze" in body
    assert "setFrozen (" not in body                             # never the engine's call directly
    # nobody edited anything: no undo step and no unsaved mark - but only when it was clean before
    assert "wasClean && ! hadHistory" in body and "clearUndoHistory()" in body and "resetChangedStatus()" in body


def test_what_the_file_says_is_read_before_the_load_and_compared_after_it():
    body = _body(SESSION_H, "inline std::unique_ptr<te::Edit> loadEditForProject", "\n}\n")
    assert body.index("freezesTheFileNames (asSaved, said)") < body.index("te::loadEditFromFile")
    assert "freezesLostAtLoad (*edit, said)" in body


# ---- the undo's report of routes ----------------------------------------------------------------

def test_a_track_playing_where_nothing_routed_it_is_not_reported_as_a_route():
    body = _body(APP, "void reconcileRoutingAfterUndo()")
    guard = body.index("impossible.add (")
    route = body.index('routes.add ("/api/route_stem?stem="')
    assert guard < route
    assert "continue;" in body[guard: route]                     # it does not fall through to the report
    assert "chainsResyncOwed = true;" in body                    # the topology is realized again
    # ...and it is SAID once that re-read has put it back, not before (the apply writes the status itself)
    assert "routingRepairNote = " in body[body.index("if (! impossible.isEmpty())"):]
    assert "if (resync && c->routingRepairNote.isNotEmpty())" in APP


def test_the_undos_report_of_routes_keeps_an_emptied_group():
    assert 'session.route_stem(stem_raw, bus or "master", keep_emptied=True)' in BRIDGE


# ---- what is said to a screen reader ------------------------------------------------------------

def test_strips_and_headers_name_their_controls_after_their_channel():
    assert STRIP_H.count("nameForSpeech();") >= 2                # the constructor, a send rebuild
    assert HEADS_H.count("nameForSpeech();") >= 2
    strip = _body(STRIP_H, "void nameForSpeech()")
    for control, said in (("fader", "Volume"), ("pan", "Pan"), ("mute", "Mute"), ("solo", "Solo"),
                          ("activator", "Power"), ("freeze", "Freeze"), ("armBtn", "Record arm")):
        assert f'dawui::speakAs ({control}, "{said}", who);' in strip, control


def test_a_dock_knob_is_named_for_its_parameter_and_its_device():
    dock = (SRC / "DeviceChainView.h").read_text(encoding="utf-8")
    assert "dawui::speakAs (*k, dawui::spokenParamName (sp.param), deviceLabel (type));" in dock
    assert 'dawui::speakAs (*k, dawui::spokenParamName (sp.param), deviceLabel ("dynamic_eq"));' in dock


def test_the_app_names_every_unnamed_control_in_every_window_it_has():
    assert "dawui::nameUnnamedControls (*top);" in APP
    sweep = _body(SPOKEN_H, "inline int nameUnnamedControls (juce::Component& root)", "\n}\n")
    assert "spokenFromTooltip (tip->getTooltip())" in sweep
    assert "hasThreeLetters (spokenNameOf (*c))" in sweep        # a control that says something is left alone


def test_the_decks_spoken_children_take_no_mouse_and_exist_only_for_a_client():
    assert "setInterceptsMouseClicks (false, false);" in _body(SPOKEN_H, "struct Zone : juce::Component", "\n    };\n")
    sync = _body(DECK_H, "void syncSpokenZones()")
    assert "if (! dawui::spokenZonesWanted()) { if (spoken.size() > 0) spoken.clear(); return; }" in sync
    assert "voiceOverOn()" in _body(SPOKEN_H, "inline bool spokenZonesWanted()", "\n}\n")


def test_the_browsers_list_answers_the_list_boxes_own_naming_hook():
    assert "juce::String getNameForRow (int row) override" in BROWSER_H


# ---- the legs are wired ---------------------------------------------------------------------------

@pytest.mark.parametrize("flag, runner", [
    ("--qa-paste-curves", "runQaPasteCurves"), ("--qa-refreeze", "runQaRefreeze"),
    ("--qa-spoken", "runQaSpoken"), ("--qa-routing-guard", "runQaRoutingGuard")])
def test_each_leg_is_reachable_from_the_console(flag: str, runner: str):
    assert f'if (flag == "{flag}")' in MAIN and f"dawapp::{runner} (project)" in MAIN
    assert f"int {runner} (const juce::File& projectDir)" in LEGS_H
    assert '#include "QaLimitsLegs.h"' in APP
