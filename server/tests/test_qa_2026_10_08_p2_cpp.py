"""User-style QA of 2026-10-08, the P2 batch - what the C++ must go on saying.

Each defect of the batch is held by a console leg that DRIVES it (see the catalogue in
.claude/skills/daw/SKILL.md: --qa-limiter-dense, --qa-stem-clipped, --qa-time-lanes,
--qa-undo-selection, --qa-window-size, and cases added to --browser, --dropped-source,
--bus-rename, --prefader-key and --engineer-feed). These pins are the other half: a leg that
drives a function proves the function, and says nothing about whether the app still calls it.

Record: docs/audits/user-qa-2026-10-08/QA-log.md.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
APP = REPO / "daw" / "Source" / "App.cpp"
ARRANGE = REPO / "daw" / "Source" / "ArrangeView.h"
SESSION = REPO / "daw" / "Source" / "DawSession.h"
CHAIN = REPO / "native" / "Source" / "MonitorChain.cpp"
PANEL = REPO / "native" / "Source" / "EngineerPanel.h"
COMMANDS = REPO / "daw" / "Source" / "DawCommands.h"
DETECTOR = REPO / "server" / "mix_engine" / "tp_detector.py"


def _body(text: str, opening: str, span: int = 6000) -> str:
    """The text of a function from its signature to the brace that closes it."""
    start = text.index(opening)
    brace = text.index("{", start)
    depth = 0
    for i in range(brace, min(len(text), brace + span)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise AssertionError(f"no end found for {opening!r}")


# ---- D28: one detector, one step, in both engines --------------------------------------------

def test_the_limiter_and_the_meter_read_a_peak_through_the_same_step():
    text = CHAIN.read_text(encoding="utf-8")
    assert "static inline float tpStep (const float* h, float& prevLast) noexcept" in text
    limiter = _body(text, "void MonitorChain::applyLookaheadLimiter (AudioBuffer<float>& buffer, bool doLimit, const LimParams& lp)", 12000)
    assert "tpStep (h, limTpPrev[ch])" in limiter, "the limiter's loop no longer reads its peak through tpStep"
    meter = _body(text, "static float truePeakOfBlock (float* hist, float& prevLast, const float* d, int n) noexcept\n{")
    assert "tpStep (hist, prevLast)" in meter, "the meter no longer reads its peak through tpStep"
    # no second walk over the table: a peak estimate made anywhere else is a second detector
    assert len(re.findall(r"kTpPhase\[p\]\[k\]", text)) == 1


def test_the_step_takes_the_crest_between_its_points_in_both_engines():
    step = _body(CHAIN.read_text(encoding="utf-8"), "static inline float tpStep (const float* h, float& prevLast) noexcept")
    assert "(b * b) / (4.0f * a)" in step, "the native step lost its parabola"
    assert "h[kTpTapsN - kTpCentre]" in step, "the native step no longer reads the next sample"
    py = DETECTOR.read_text(encoding="utf-8")
    assert "(b[bends] ** 2) / (4.0 * a[bends])" in py, "the Python detector lost its parabola"


def test_the_gain_floor_is_read_where_the_exiting_sample_was_judged():
    limiter = _body(CHAIN.read_text(encoding="utf-8"),
                    "void MonitorChain::applyLookaheadLimiter (AudioBuffer<float>& buffer, bool doLimit, const LimParams& lp)", 12000)
    assert "targets[(delayPos + k) % ringLen]" in limiter
    assert "for (int k = kTpCentre - kLimFloorHalf; k <= kTpCentre + kLimFloorHalf; ++k)" in limiter
    assert "out = jlimit (-ceiling, ceiling, out);" in limiter, "the backstop stays as the last resort"


def test_the_limiter_memory_has_a_line_in_prepare_and_in_the_reset():
    text = CHAIN.read_text(encoding="utf-8")
    assert text.count("std::memset (limTpPrev, 0, sizeof (limTpPrev));") == 2
    assert text.count("std::memset (clipTpPrev, 0, sizeof (clipTpPrev));") == 2
    assert text.count("limTargetBuf") >= 4


# ---- D31: a whole-timeline edit carries the automation ----------------------------------------

def test_every_time_edit_calls_its_lane_helper():
    text = ARRANGE.read_text(encoding="utf-8")
    assert "dawsession::insertSpaceIntoEveryLane (edit, a, b - a);" in _body(text, "void insertTimeRange (double a, double b)")
    assert "dawsession::removeRegionFromEveryLane (edit, a, b);" in _body(text, "void deleteTimeRange (double a, double b)")
    paste = _body(text, "bool pasteTimeAt (double at)")
    assert "dawsession::pasteRegionIntoEveryLane (edit, timeClipLanes, at, timeClipLen);" in paste
    copy = _body(text, "void copyTimeRange (double a, double b)")
    assert "timeClipLanes   = dawsession::copyRegionOfEveryLane (edit, a, b);" in copy, "the rides are not on the time clipboard"
    # ...and the clipboard outlives the view: the host builds a new one at every track add
    # (+ the engine's curves, 2026-10-09; and the two modes the view is in, the same day)
    assert "timeClipEntries, timeClipStart, timeClipLen, timeClipLanes, timeClipCurves," in text
    assert "timeClipLanes   = s.timeClipLanes;" in text
    duplicate = _body(text, "void duplicateTimeRange (double a, double b)")
    assert "dawsession::collectCurveRepeats (edit, a, b)" in duplicate
    assert duplicate.index("collectCurveRepeats") < duplicate.index("te::insertSpaceIntoEdit"), \
        "the engine's curves are read before the engine moves them"
    assert "dawsession::duplicateRegionInEveryLane (edit, a, b, engineCurves);" in duplicate


def test_no_time_edit_opens_the_timeline_without_the_lanes():
    """`te::insertSpaceIntoEdit` moves clips, the tempo map and the engine's own curves on tracks -
    never a device lane, never the master. Every call of it is followed by a lane helper."""
    text = ARRANGE.read_text(encoding="utf-8")
    calls = [m.start() for m in re.finditer(r"te::insertSpaceIntoEdit \(", text)]
    assert calls, "the arrange view no longer inserts time?"
    for at in calls:
        after = text[at:at + 700]
        assert "EveryLane (edit" in after, text[at:at + 160]


def test_the_lane_helpers_reach_every_node_and_the_master():
    text = SESSION.read_text(encoding="utf-8")
    nodes = _body(text, "inline juce::Array<te::MonitorChainPlugin*> everyChainNode (te::Edit& edit)")
    assert "te::getAllPlugins (edit, false)" in nodes and "edit.getMasterPluginList()" in nodes


def test_the_time_clipboard_carries_the_chords_and_the_sections():
    text = ARRANGE.read_text(encoding="utf-8")
    collect = _body(text, "std::vector<ClipboardEntry> collectTimeEntries (double a, double b) const")
    assert "takeGlobal (edit.getChordTrack(), kChordTrackKey);" in collect
    assert "takeGlobal (edit.getArrangerTrack(), kArrangerTrackKey);" in collect
    place = _body(text, "void placeTimeEntries (const std::vector<ClipboardEntry>& entries, double rangeStart, double at, double len)")
    assert "en.trackId == kChordTrackKey" in place and "en.trackId == kArrangerTrackKey" in place
    # a key no track can have: a track's name never holds a colon
    assert 'kChordTrackKey    = "::chords"' in text and 'kArrangerTrackKey = "::sections"' in text


# ---- D1: the engine's selection lets go of a track that has left ------------------------------

def test_the_structural_heal_tells_the_selection_manager():
    heal = _body(APP.read_text(encoding="utf-8"), "void healAfterStructuralUndo()")
    assert "dawsession::selectTrackOnSurfaces (selectionManager, trackById (selectedStem));" in heal
    assert heal.index("selectNode (selectedStem);") < heal.index("dawsession::selectTrackOnSurfaces")


def test_the_surface_callback_never_focuses_a_track_that_left_the_edit():
    callback = _body(APP.read_text(encoding="utf-8"),
                     "inline void DawComponent::SurfaceSelection::changeListenerCallback (juce::ChangeBroadcaster*)")
    assert "te::getAllTracks (*owner->edit).contains (t)" in callback
    assert callback.index("contains (t)") < callback.index("owner->focusNode (id);")


# ---- D9: the window keeps its size --------------------------------------------------------------

def test_a_project_switch_shows_the_project_through_the_one_function():
    text = APP.read_text(encoding="utf-8")
    assert "dawui::showInWindowKeepingItsSize (*mainWindow, *dc);" in text
    assert "mainWindow->setContentNonOwned (dc" not in text, "the plain call gives the window the component's size"


# ---- D17 / D18: a rename in place owes a rebuild -------------------------------------------------

def test_a_rename_in_place_rebuilds_the_strips_and_headers_at_the_next_realize():
    text = APP.read_text(encoding="utf-8")
    rename = _body(text, "void renameRoutingTrackInPlace (const juce::String& oldNode, const juce::String& newNode)")
    assert "routingNamesMoved = true;" in rename
    gate = text[text.index("if (! dropped.isEmpty()\n"):]
    gate = gate[:gate.index("c->routingNamesMoved = false;") + 40]
    assert "|| c->routingNamesMoved)" in gate and "c->buildTrackUI();" in gate


# ---- D19: the composer's button -------------------------------------------------------------------

def test_the_composers_button_asks_what_is_typed():
    text = PANEL.read_text(encoding="utf-8")
    assert "sendBtn.onClick = [this] { if (sendButtonStops()) { if (onStop) onStop(); } else send(); };" in text
    assert "bool sendButtonStops() const { return busy && input.getText().trim().isEmpty(); }" in text
    # every place that changes the composer's text without a change message refreshes the button
    for needle in ("input.clear();\n        refreshSendButton();",
                   "input.setText (lastSentRaw, false);\n        refreshSendButton();",
                   "input.setText (text, false); scanMentions(); refreshSendButton();"):
        assert needle in text, needle
    assert "busy ? MixPalette::warm : MixPalette::accent" not in text, "a colour still follows `busy` alone"


# ---- D15: the engine's own files -------------------------------------------------------------------

def test_the_lost_audio_watch_leaves_out_an_engine_file_nothing_plays():
    watch = _body(APP.read_text(encoding="utf-8"), "void watchLostAudio()", 9000)
    assert "f.isAChildOf (engineTemp)" in watch
    assert "! edit->areAnyClipsUsingFile (te::AudioFile (engine, f))" in watch


# ---- D29: the reply and the catalog say the same -----------------------------------------------------

def test_the_export_names_a_clipped_stem_and_the_catalog_says_so():
    app = APP.read_text(encoding="utf-8")
    assert 'oo->setProperty ("clipped_stems", juce::var (clipped));' in app
    assert "dawexport::fileLevelsClipped (levels)" in app
    assert "`clipped_stems`" in COMMANDS.read_text(encoding="utf-8")
    panel = (REPO / "daw" / "Source" / "ExportPanel.h").read_text(encoding="utf-8")
    assert "return lv.read && ! lv.isFloat && lv.peak >= 0.999f;" in panel, "a float file is not called clipped"


# ---- D14: the key a pack writes into its file names ------------------------------------------------

def test_the_filename_key_parser_takes_key_glued_to_the_note():
    parser = _body(SESSION.read_text(encoding="utf-8"), "inline MusicalKey keyFromFilename (const juce::String& name)")
    assert 'lower.startsWith ("key") && tok.length() > 3' in parser
    # a bare note after a glued "key" needs a capital, and never the `s` read elsewhere as a sharp
    assert "juce::CharacterFunctions::isUpperCase (tail[0])" in parser and 'tail.endsWithIgnoreCase ("s")' in parser
