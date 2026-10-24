"""What driving the real window by hand found on 2026-10-09 (James, of the list "not checked by
hand in the window": "we need to take care of these") - pinned where a leg cannot reach, or where
the ORDER of two things is the fix.

Every gesture on that list was driven in the app with synthetic mouse and key events; nine
defects came out of it. Their behaviour is driven by two console legs
(daw/Source/QaLimitsLegs.h): ``--qa-hand-fixes`` and ``--qa-groove-grid``, both seen red with
the old path of every part switched in (QA-log.md has the fourteen red lines).

Record: docs/audits/user-qa-2026-10-08/QA-log.md, *Driven by hand, 2026-10-09*.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from mix_engine.session import Session
from orchestrator import daw_bridge

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "daw" / "Source"
APP = (SRC / "App.cpp").read_text(encoding="utf-8")
MAIN = (SRC / "Main.cpp").read_text(encoding="utf-8")
SESSION_H = (SRC / "DawSession.h").read_text(encoding="utf-8")
ARRANGE_H = (SRC / "ArrangeView.h").read_text(encoding="utf-8")
NODE_H = (SRC / "MonitorChainPlugin.h").read_text(encoding="utf-8")
LEGS_H = (SRC / "QaLimitsLegs.h").read_text(encoding="utf-8")
MANUAL = re.sub(r"\s+", " ", (ROOT / "docs" / "17-user-manual.md").read_text(encoding="utf-8"))

reads_cpp = pytest.mark.reads_cpp


def _body(source: str, start: str, end: str = "\n    }\n") -> str:
    i = source.index(start)
    return source[i: source.index(end, i)]


# ---- the switch a red run was made with is gone -------------------------------------------------

@reads_cpp
def test_the_temporary_old_path_switch_is_not_in_the_tree():
    assert not (SRC / "HandOldSwitch.h").exists()
    for path in list(SRC.glob("*.h")) + list(SRC.glob("*.cpp")) + list(SRC.glob("*.mm")):
        text = path.read_text(encoding="utf-8", errors="replace")
        assert "handOld" not in text and "HandOldSwitch" not in text, path.name


# ---- a Touch ride returns ------------------------------------------------------------------------

@reads_cpp
def test_a_touch_ride_returns_where_nothing_was_written_after_it():
    """The return was written only when the old lane had a point AFTER the punch-out: on a lane
    that held nothing, or one written only before the ride, Touch never came back."""
    body = _body(NODE_H, "    void endRide (const juce::String& target, double t, double glideS, bool returnsToOld = false)")
    assert "const bool touchBack = returnsToOld && ! origAfter && (! r.orig.empty() || r.before.has_value());" in body
    assert "if (origAfter || touchBack)" in body
    assert "r.orig.empty() ? *r.before : evalPoints (r.orig, tg)" in body
    assert "r.before = before;" in _body(NODE_H, "    bool beginRide (")


@reads_cpp
def test_every_way_the_app_ends_a_ride_says_whether_it_was_touch():
    ends = re.findall(r"endRide \(kRideTarget,[^;]+;", APP)
    assert len(ends) == 2, ends
    for call in ends:
        assert "te::AutomationMode::touch" in call, call
    assert "the lane is back where it was a quarter of a second later" in MANUAL


# ---- Shift-Cmd-T ---------------------------------------------------------------------------------

@reads_cpp
def test_the_instrument_track_key_lands_on_its_new_track():
    body = _body(APP, "            case dawcmd::trackAddMidi:", "            case dawcmd::midiAddClip:")
    reveal = body.index("revealTrackLane (track->getName());")
    assert body.index("buildTrackUI();") < reveal
    user = body[body.index("if (! forAgent)"):]
    assert "selectNode (track->getName());" in user and "sayStatus (" in user
    # the agent's call leaves the user's selection alone
    assert "selectNode" not in body[: body.index("if (! forAgent)")]


# ---- Automation Mode across a rebuild ------------------------------------------------------------

@reads_cpp
def test_automation_mode_rides_across_the_view_a_track_add_rebuilds():
    carried = _body(ARRANGE_H, "struct CarriedViewState", "CarriedViewState carriedViewState() const")
    assert "bool autoEdit = false, draw = false;" in carried
    # ...and whether the PENCIL is what switched Automation Mode on (2026-10-13): Draw Mode takes
    # the mode back out when it goes, and a track added while it is out must not lose that.
    assert "bool autoByDraw = false;" in carried
    assert "autoEditMode, drawMode, autoByDraw }; }" in ARRANGE_H
    restore = _body(ARRANGE_H, "    void restoreCarriedViewState (const CarriedViewState& s)")
    assert "autoEditMode = s.autoEdit;" in restore and "drawMode     = s.draw && s.autoEdit;" in restore
    assert "autoByDraw   = s.autoByDraw && drawMode;" in restore


# ---- the computer MIDI keyboard and an instrument's own window -----------------------------------

@reads_cpp
def test_a_key_from_an_editor_window_is_a_note_before_it_is_a_shortcut():
    body = _body(APP, "    bool keyFromEditorWindow (const juce::KeyPress& k)")
    assert body.index("if (computerKeyboard.keyPressed (k, this)) return true;") < body.index("getKeyMappings()->keyPressed")
    # the key-UP: JUCE reports it only to the window that has the keys
    window = _body(APP, "struct FxWindow  : public juce::DocumentWindow", "\n};\n")
    assert "bool keyStateChanged (bool isKeyDown) override" in window and "if (onKeyState != nullptr) onKeyState();" in window
    assert "ew->window->onKeyState = [safe]" in APP and "c->computerKeyboard.keyStateChanged (false, c);" in APP
    # ...and "does this app have the keys" counts that window, or the tick ends the note at once
    active = _body(APP, "        computerKeyboard.windowActive = [this]", "\n        };\n")
    assert "for (auto& w : editorWindows)" in active and "wp->isFocused()" in active


# ---- Set role ------------------------------------------------------------------------------------

def test_the_strip_row_says_the_role_a_stem_is_mixed_as(demo_project):
    s = Session(demo_project)
    s.set_stem_tags("stem_a", ["keys"])
    s.set_stem_tags("stem_b", [])
    rows = daw_bridge.strip_state(Session(demo_project)._state)
    assert rows["stem_a"]["role"] == "keys"
    assert rows["stem_b"]["role"] == ""                         # no tag: nothing ticked


@reads_cpp
def test_the_set_role_menu_ticks_the_role_and_says_what_it_set():
    assert "roleTags[ri] == roleNow" in APP
    assert "brainRoles[kv.name.toString()] = kv.value.getProperty (\"role\", \"\").toString();" in APP
    assert "(what the engineer mixes it as)" in APP
    assert "brainRoles.clear();" in APP                         # per edit, with the engine's facts


# ---- what the top bar says is readable -----------------------------------------------------------

@reads_cpp
def test_a_status_is_drawn_on_two_lines_before_it_is_cut():
    assert "if (busy || recent) g.drawFittedText (eng, inner, juce::Justification::centredRight, 2, 0.8f);" in APP


@reads_cpp
def test_the_refreeze_is_said_in_the_feed_as_well():
    body = _body(APP, "    void refreezeLostAtLoad()")
    assert "if (engineer != nullptr) engineer->statusLine (said);" in body


# ---- Extract Groove ------------------------------------------------------------------------------

@reads_cpp
def test_extract_groove_is_measured_on_the_files_own_grid():
    """The brain measures the FILE. A warped clip's file is not at the song's tempo."""
    window = _body(SESSION_H, "inline GrooveMeasureWindow grooveMeasureWindow (te::Edit& edit, te::AudioClipBase& acb)", "\n}\n")
    assert "if (acb.getAutoTempo())" in window
    assert "clipSourceStartSeconds" in window and "clipSourceWindowSeconds" in window
    start = APP.index("            case dawcmd::clipExtractGroove:", APP.index("juce::var invokeCommand ("))
    handler = APP[start: APP.index("            case dawcmd::", start + 40)]
    assert "dawsession::grooveMeasureWindow (*edit, *acb)" in handler
    assert "getBpmAt (te::TimePosition())" not in handler      # never the song's first tempo


# ---- the Edit menu's keys (seen in the window) ---------------------------------------------------

@reads_cpp
def test_the_legs_are_reachable_from_the_console():
    for flag, runner in (("--qa-hand-fixes", "runQaHandFixes"), ("--qa-groove-grid", "runQaGrooveGrid")):
        assert f'if (flag == "{flag}")' in MAIN and f"dawapp::{runner} (project)" in MAIN
        assert f"int {runner} (const juce::File& projectDir)" in LEGS_H
