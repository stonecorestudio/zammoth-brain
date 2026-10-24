"""User-style QA of 2026-10-08, D36 and the P3 batch - what the C++ must go on saying.

The behaviours are driven by console legs (--qa-export-unsaved, --qa-user-words,
--qa-short-window, --qa-key-align, --qa-small-ui, and cases in --engineer-feed, --menus and
--edit-dup-live). These pins hold the other half: that the app still CALLS what those legs drive.

Record: docs/audits/user-qa-2026-10-08/QA-log.md.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
APP = (REPO / "daw" / "Source" / "App.cpp").read_text(encoding="utf-8")
SESSION = (REPO / "daw" / "Source" / "DawSession.h").read_text(encoding="utf-8")
COMMANDS = (REPO / "daw" / "Source" / "DawCommands.h").read_text(encoding="utf-8")
PALETTE = (REPO / "daw" / "Source" / "CommandPalette.h").read_text(encoding="utf-8")
PREFS = (REPO / "daw" / "Source" / "PrefsPanel.h").read_text(encoding="utf-8")
BROWSER = (REPO / "daw" / "Source" / "FileBrowserPanel.h").read_text(encoding="utf-8")
EXPORT = (REPO / "daw" / "Source" / "ExportPanel.h").read_text(encoding="utf-8")
SNAPSHOT = (REPO / "daw" / "Source" / "EditSnapshot.h").read_text(encoding="utf-8")
LNF = (REPO / "native" / "Source" / "MixAgentLookAndFeel.h").read_text(encoding="utf-8")
PANEL = (REPO / "native" / "Source" / "EngineerPanel.h").read_text(encoding="utf-8")
MANUAL = (REPO / "docs" / "17-user-manual.md").read_text(encoding="utf-8")


def _body(text: str, opening: str, span: int = 9000) -> str:
    start = text.index(opening)
    brace = text.index("{", start)
    depth = 0
    for i in range(brace, min(len(text), brace + span)):
        depth += text[i] == "{"
        depth -= text[i] == "}"
        if depth == 0:
            return text[start:i + 1]
    raise AssertionError(f"no end found for {opening!r}")


# ---- D36: a frozen track across a relaunch, and an export that leaves a saved project saved -----

def test_every_edit_of_a_project_is_loaded_under_its_own_id():
    """The engine's loader throws the file's id away and mints one, so the temp folder a frozen
    track's render is looked for in had a new name at every launch."""
    assert APP.count("dawsession::loadEditForProject (engine, ") == 2
    assert not re.search(r"=\s*[^;\n]*\bte::loadEditFromFile \(engine, (sessionFile|f)\)", APP)
    loader = _body(SESSION, "inline std::unique_ptr<te::Edit> loadEditForProject (")
    assert loader.index("restoreFreezeFiles (engine, id, projectDir);") < loader.index("te::Edit::createEdit (")


def test_the_project_keeps_what_is_frozen_at_every_door_that_changes_it():
    freeze = APP[APP.index("case dawcmd::trackFreeze:"):]
    freeze = freeze[:freeze.index("case dawcmd::", 30)]
    assert "dawsession::keepFreezeFiles (*edit, project);" in freeze
    assert "dawsession::keepFreezeFiles (*edit, project);" in _body(APP, "    bool saveSession()")
    assert "dawsession::copyKeptFreezeFiles (project, item->getID(), dup->getID());" in APP


def test_an_export_that_found_the_project_saved_leaves_it_saved():
    assert "exportFoundSaved = ! edit->hasChangedSinceSaved();" in APP
    assert "settleAfterExport();" in APP[APP.index("watchLostAudio();           //"):][:400]
    settle = _body(APP, "    void settleAfterExport()")
    assert "dawsession::userEditCount().load() != exportEditMark" in settle and "resetChangedStatus()" in settle
    # ...and somebody's edit is counted where edits begin
    begin = SESSION[SESSION.index("inline void beginTransaction (te::Edit& edit, const juce::String& name = {})"):][:220]
    assert "++userEditCount();" in begin
    assert "if (marksEdit) ++dawsession::userEditCount();" in APP


# ---- D3: the agent's grammar is for the agent -------------------------------------------------

def test_a_key_the_user_presses_is_refused_in_plain_words():
    for agent_text in ("or give start_s and end_s", "slot.select {track, scene} selects one",
                       "time.insert needs a time selection", "time edits need a time selection",
                       "time.copy or time.cut a range first", 'or name one: track \\"bus:<id>\\"'):
        at = APP.index(agent_text)
        assert "forAgent ?" in APP[at - 160:at], agent_text
    assert "track.freeze {track" not in APP, "an unfreeze hint is said through dawsession::unfreezeHint"
    hint = _body(SESSION, "inline juce::String unfreezeHint (const juce::String& track)")
    assert "agentTurnActive() ?" in hint and "the freeze key on its header" in hint


# ---- D7 / D10 ----------------------------------------------------------------------------------

def test_the_palette_matches_every_word_and_the_loop_deck_has_its_name():
    rebuild = _body(PALETTE, "    void rebuild()")
    assert 'words.addTokens (box.getText().trim().toLowerCase(), " ", "");' in rebuild
    assert "for (const auto& word : words) all = all && hay.contains (word);" in rebuild
    assert '"Scenes"' not in COMMANDS
    assert COMMANDS.count('"Loop Deck",') >= 37
    assert '"Loop Deck Panel"' in COMMANDS and '"Clip Launcher Panel"' not in COMMANDS


def test_a_combo_popup_opens_under_its_box():
    options = _body(LNF, "juce::PopupMenu::Options getOptionsForComboBoxPopupMenu (juce::ComboBox& box, juce::Label& label) override")
    assert "withItemThatMustBeVisible" not in options
    assert "withInitiallySelectedItem (box.getSelectedId())" in options


# ---- D5 / D22 / D25 ----------------------------------------------------------------------------

def test_a_command_only_turn_is_offered_its_undo():
    assert 'c->engineer->offerUndoRun (done.getProperty ("gesture", {}).toString());' in APP
    assert "void offerUndoRun (const juce::String& gesture)" in PANEL


def test_a_planned_fader_step_reads_as_a_fader():
    assert 'if (command == "track.set_gain" && args.hasProperty ("gain_db"))' in APP
    assert 'm.devLabel  = "FADER";' in APP


def test_every_errand_of_the_run_thread_has_words():
    """`startAction ("x")`: the Export window names what is busy, and a new errand needs a row."""
    words = _body(APP, "    static juce::String errandWords (const juce::String& a)")
    mix_run = _body(APP, "    static bool isMixRunAction (const juce::String& a)")
    named = set(re.findall(r'"([a-z_]+)"', words)) | set(re.findall(r'"([a-z_]+)"', mix_run))
    started = set(re.findall(r'startAction \("([a-z_]+)"\)', APP)) - {"export", "console_park", "console_park_deaf"}
    missing = sorted(a for a in started if a not in named and not a.startswith("project_import"))
    assert not missing, f"errands with no words: {missing}"
    assert 'fromUTF8 ("the engine is busy' not in APP, "the nameless message is back"
    assert "panel->exportFinished (false, busyWords(), {});" in APP


# ---- D13 / D16 / D23 / D26 / D27 ---------------------------------------------------------------

def test_a_duplicate_edit_is_listed_at_its_length():
    assert "dup->setLength (ofOpenEdit ? edit->getLength().inSeconds() : item->getLength());" in APP


def test_the_mixer_leaves_the_arrangement_its_lanes():
    assert "const int roomWithLanes = centre.getHeight() - kSplit - 6 - kArrangeLanesMinH;" in APP
    assert re.search(r"kArrangeLanesMinH = \d+;", APP)


def test_the_key_align_line_is_for_a_grid_that_moved():
    at = APP.index("grid aligned to the detected key")
    block = APP[at - 1600:at]
    assert "((wantPitch % 12) + 12) % 12 == base->getPitch() % 12 && wantScale == base->getScale())" in block
    assert block.index("appliedDetectedKey = true;") < block.index("openAlign();")


def test_a_component_that_holds_a_stale_save_does_not_seed_the_brain():
    assert '! id.startsWith ("bus:") && ! recoverableTemp)' in APP
    assert 'if (startAction ("readout")) readoutKicked = true;' in APP


def test_quit_asks_about_one_project_at_a_time():
    confirm = _body(APP, "    void confirmSaveThen (", 6000)
    assert "askAboutNextProject (review);" in confirm
    assert confirm.index("dd->saveSession()") < confirm.index("dd->declineUnsavedChanges()"), \
        "every save is tried before any decline is applied: a decline sticks on its project"
    assert "if (cancelled) return;" in confirm
    ask = _body(APP, "    void askAboutNextProject (std::shared_ptr<UnsavedReview> review)")
    assert "showUnsavedChangesAlert (" in ask and "review->done (true)" in ask


# ---- D2 / D8 / D11 / D30 / D32 -----------------------------------------------------------------

def test_small_things_go_through_their_one_function():
    assert "const int labelRight = audioRowLabelRight (w, it.badge.isNotEmpty(), subW, plus);" in BROWSER
    assert "overwriteTagRect (getWidth(), filenameEd->getY())" in EXPORT
    assert "applyZoom (dawui::nextZoomStep (uiZoom, +1))" in APP and "applyZoom (dawui::nextZoomStep (uiZoom, -1))" in APP
    assert "info.setInfo (dawui::actualSizeLabel (uiZoom)," in APP
    assert "d->refreshPrefsDialog();" in _body(APP, "    void applyZoom (float z, bool persist = true)")
    assert "static const std::vector<int>& zoomSteps() { return dawui::zoomSteps(); }" in PREFS
    assert "appboundary::osNameForReport (juce::SystemStats::getOperatingSystemName())" in APP
    assert "appboundary::capitaliseButtons (*editor);" in APP
    assert "d.meterStyleLabels.add (meterlook::label ((meterlook::Style) i));" in APP


def test_no_page_narrates_its_own_history_to_the_user():
    for stale in ("they used to reset every time", "before 2026-08-06", "the File and View menu keys keep their standard chords"):
        assert stale not in PREFS, stale
    assert 'info.setInfo ("New Project", ' in APP, "a menu item that opens no dialog carries no ellipsis"


def test_a_loaded_plugin_always_says_its_latency():
    at = SNAPSHOT.index('po->setProperty ("latency_samples", inst->getLatencySamples());')
    assert "inst != nullptr)" in SNAPSHOT[at - 220:at] and "getLatencySamples() > 0" not in SNAPSHOT[at - 220:at]
    assert 'po->setProperty ("params_shown", cap);' in SNAPSHOT


# ---- D12: the manual says where things are -----------------------------------------------------

def test_the_manual_lists_the_view_menu_and_puts_osc_where_it_is():
    view = MANUAL[MANUAL.index("- **View** — Search Commands…"):][:700]
    for item in ("Computer MIDI Keyboard", "Keyboard\n  Shortcuts…", "In/Out", "Track Delay", "Time Ruler", "Skin ▸", "Meters ▸"):
        assert item in view, item
    midi = MANUAL[MANUAL.index("\n## MIDI\n"):MANUAL.index("\n## Control Surfaces\n")]
    assert "OSC remote control" not in midi
    surfaces = MANUAL[MANUAL.index("\n## Control Surfaces\n"):][:600]
    assert "OSC remote control" in surfaces
    assert "buildSurfaces()" in PREFS and PREFS.index("OSC remote control") > PREFS.index("void buildSurfaces()")
