"""User-style QA of 2026-10-08, the evening's findings (D37-D57) - what the C++ must go on saying.

The behaviours are driven by console legs (--qa-undo-group, --qa-lanes, --qa-tempo-loop,
--qa-arrange-smalls, --qa-dock-smalls, --qa-capture-arm). These pins hold the other half: that
the app still CALLS what those legs drive, and that a class of defect stays closed.

Record: docs/audits/user-qa-2026-10-08/QA-log.md.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
APP = (REPO / "daw" / "Source" / "App.cpp").read_text(encoding="utf-8")
ARRANGE = (REPO / "daw" / "Source" / "ArrangeView.h").read_text(encoding="utf-8")
SESSION = (REPO / "daw" / "Source" / "DawSession.h").read_text(encoding="utf-8")
DOCK = (REPO / "daw" / "Source" / "DeviceChainView.h").read_text(encoding="utf-8")
HEADS = (REPO / "daw" / "Source" / "TrackHeadColumn.h").read_text(encoding="utf-8")
NODE = (REPO / "daw" / "Source" / "MonitorChainPlugin.h").read_text(encoding="utf-8")
PATCHES = REPO / "daw" / "patches"


def _body(text: str, opening: str, span: int = 12000) -> str:
    start = text.index(opening)
    brace = text.index("{", start)
    depth = 0
    for i in range(brace, min(len(text), brace + span)):
        depth += text[i] == "{"
        depth -= text[i] == "}"
        if depth == 0:
            return text[start:i + 1]
    raise AssertionError(f"no end found for {opening!r}")


# ---- D37: a track output's stored position is not an undo step ------------------------------------

def test_the_engine_patch_that_keeps_a_stored_output_out_of_the_undo_history_is_recorded():
    patch = [p for p in PATCHES.glob("0040-*.patch")]
    assert len(patch) == 1, "patch 0040 (a track output's stored position) is not in daw/patches"
    text = patch[0].read_text(encoding="utf-8")
    assert "tracktion_TrackOutput.cpp" in text
    # the flush writes with no undo manager, and each setter flushes first
    assert "setProperty (outputDevice.getPropertyID(), fresh, nullptr)" in text
    assert text.count("+    flushStateToValueTree();") == 3, "the three setters flush before they write"


# ---- D50: a status said on the message thread is stamped ------------------------------------------

def test_no_status_is_written_past_the_stamp_the_top_bar_asks_for():
    """About forty sites wrote `runStatus` directly; the top bar shows an idle status only when
    `runStatusAt` is fresh, so none of them was ever shown. Every write on the message thread
    goes through sayStatus, every other through setStatus (a posted lambda writes `c->runStatus`
    and stamps beside it, or is a turn's own status)."""
    allowed = _body(APP, "void sayStatus (const juce::String& s)")
    bare = []
    for m in re.finditer(r"(?<![>\w.])runStatus\s*(?:=(?!=)|\+=)", APP):
        if allowed and APP.find(allowed) <= m.start() < APP.find(allowed) + len(allowed):
            continue
        line = APP[APP.rfind("\n", 0, m.start()) + 1:APP.find("\n", m.start())]
        if "setRunStatusForConsole" in line or "runStatus +=" in line:
            continue          # the console's seam; an addition to a status just said
        bare.append(line.strip()[:110])
    assert not bare, "a status written with no stamp (use sayStatus):\n  " + "\n  ".join(bare)


def test_the_arrangement_and_the_take_hold_say_their_status_through_the_stamp():
    assert re.search(r"arrange->onStatus = \[this\] \(const juce::String& s\) \{ sayStatus \(s\);", APP)
    hold = _body(APP, "dawsession::TakeHold takeHoldForViews()")
    assert "sayStatus (s)" in hold


# ---- D38 / D55 / D56: lanes -----------------------------------------------------------------------

def test_an_empty_device_lane_asks_its_node_for_the_controls_value():
    body = _body(ARRANGE, "float currentParamValue (const AutoLaneRef& l) const")
    assert "readEntryField (l.target)" in body


def test_a_ride_is_begun_with_the_level_that_played_before_it():
    assert "node->beginRide (kRideTarget, t, db, node->getStemGainBeforeLastSet())" in APP
    start = NODE.index("bool beginRide (const juce::String& target, double t, float value, std::optional<float> before = {})")
    begin = NODE[start:start + 1800]
    assert "evalPoints (r.orig, ta)" in begin and "*before" in begin


def test_the_headers_db_cell_is_a_hand_on_the_fader():
    assert "if (onRide) onRide (h.id, db); else h.node->faderMovedByHand();" in HEADS
    assert "trackHead->onRide = [this] (const juce::String& id, float db) { rideGain (id, db); };" in APP
    assert "trackHead->onGainHold" in APP


# ---- D40: the loop brace across an undo of a tempo change -----------------------------------------

def test_undo_and_redo_reseat_the_loop_brace_when_the_tempo_map_changed():
    for case in ("case dawcmd::editUndo:", "case dawcmd::editRedo:"):
        body = APP[APP.index(case):APP.index(case) + 9000]
        assert "dawsession::LoopBarsAcrossUndo" in body and "loopBars->reseat()" in body, case


# ---- D42 / D45 / D46: the arrangement --------------------------------------------------------------

def test_every_door_that_makes_a_track_brings_it_on_screen():
    assert APP.count("revealTrackLane (") >= 6      # the definition, and blank / MIDI / duplicate / bounce / drop


def test_a_tracks_colour_ordinal_is_its_own_at_all_three_places():
    assert "dawsession::trackColourOrdinal (*edit, *t)" in _body(APP, "juce::Colour effectiveTrackColour (const juce::String& id)")
    assert "stemColourIdx[t->getName()] = dawsession::trackColourOrdinal (*edit, *t)" in APP
    assert "dawsession::trackColourOrdinal (edit, *lanes[i])" in _body(ARRANGE, "juce::Colour laneColourRaw (int i) const")
    stamp = _body(SESSION, "inline void stampColourSlots (te::Edit& edit)")
    assert "nullptr" in stamp, "the stamp is nobody's edit: no undo manager"


def test_a_header_click_and_a_strip_click_drop_the_time_highlight():
    assert APP.count("arrange->clearTimeSelection()") == 2


# ---- D48 / D49 / D53 / D57: the dock ---------------------------------------------------------------

def test_a_pasted_stereo_width_on_a_track_is_the_strips_width():
    paste = _body(DOCK, "int pasteDeviceEntries (const juce::Array<juce::var>& entries)")
    assert 't == "width" && stripWidthNode()' in paste and "setStemWidthUrl" in paste


def test_the_undo_mirror_sends_only_parameters_the_brain_has():
    start = APP.index("node->onChainReverted = [this, stemNow]")
    body = APP[start:start + 2600]
    assert "if (deviceParamClamp (type, field) == nullptr) return;" in body


def test_the_modulator_lookup_holds_the_list_it_reads():
    body = _body(DOCK, "juce::var modifierOn (const juce::String& target)")
    assert "modifiersProvider().getArray()" not in body
    assert "const auto all = modifiersProvider();" in body


def test_cmd_d_on_a_device_says_why_nothing_was_duplicated():
    start = APP.index("const int n = chainDock->pasteDeviceEntries (copied);")
    assert "chainDock->lastPasteRefusal" in APP[start:start + 900]


# ---- D54: the capture's floor ----------------------------------------------------------------------

def test_a_launch_from_a_stop_moves_an_empty_captures_floor_to_the_park():
    body = _body(APP, "void startDeckTransport()")
    park, arm, play = body.index("parkOnLaunchGrid"), body.index("captureArmBeat ="), body.index("tr.play (false)")
    assert park < arm < play
