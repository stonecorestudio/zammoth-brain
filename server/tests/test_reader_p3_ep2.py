"""Reader P3s, batch EP2 - the engine patches 0034-0038 (2026-10-07).

The behaviour is driven by the console legs in `daw/Source/AuditReaderP3LegsEP2.h`
(`--dither-once`, `--temp-sweep`, `--absent-plugin-record`, `--save-keeps-undo`). This file
holds what a leg cannot: that each patch is in the tracked series with its row, that the docs
say how many patches the engine is, and the two app-side parts' shape in the source.

The engine clone itself is git-ignored; that it IS `ENGINE_BASE` + these files is
`test_reader_p2_sh1.py`'s (`fetch_engine.sh --check`).
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATCHES = ROOT / "daw" / "patches"
APP = (ROOT / "daw" / "Source" / "App.cpp").read_text(encoding="utf-8")
SESSION = (ROOT / "daw" / "Source" / "DawSession.h").read_text(encoding="utf-8")


def _patch(number: int) -> str:
    found = sorted(PATCHES.glob(f"{number:04d}-*.patch"))
    assert len(found) == 1, f"patch {number:04d}: {[p.name for p in found]}"
    return found[0].read_text(encoding="utf-8")


def _added(patch: str) -> str:
    """The lines a patch adds, without their leading '+'."""
    return "\n".join(line[1:] for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++"))


def test_every_patch_has_its_row_in_the_readme():
    """The README's table is where a patch says why it exists and which leg holds it. A patch
    with no row is a change to the engine nobody can weigh at a re-vendor."""
    readme = (PATCHES / "README.md").read_text(encoding="utf-8")
    missing = [p.name for p in sorted(PATCHES.glob("[0-9][0-9][0-9][0-9]-*.patch")) if f"`{p.name}`" not in readme]
    assert not missing, f"no README row for: {missing}"


def test_the_docs_say_how_many_patches_the_engine_is():
    """`the engine is ENGINE_BASE + N`: in CLAUDE.md that sentence is a rule other sessions
    build from (a worktree is brought up to date by applying the patches past its own)."""
    count = len(list(PATCHES.glob("[0-9][0-9][0-9][0-9]-*.patch")))
    claude = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    said = {int(n) for n in re.findall(r"the engine is `ENGINE_BASE` \+ (\d+)", claude)}
    assert said == {count}, f"CLAUDE.md says ENGINE_BASE + {sorted(said)}; there are {count} patches"
    daw = (ROOT / "docs" / "10-daw.md").read_text(encoding="utf-8")
    assert f"`ENGINE_BASE` + {count}" in daw, "docs/10-daw.md does not name the current patch count"


def test_0034_the_intermediate_is_not_dithered():
    patch = _patch(34)
    assert "tracktion_NodeRenderContext.cpp" in patch
    assert "r.ditheringEnabled = false;" in _added(patch)


def test_0035_only_the_last_engine_sweeps():
    patch = _patch(35)
    assert "tracktion_Engine.cpp" in patch
    added = _added(patch)
    assert "if (engines.size() <= 1)" in added and "temporaryFileManager->cleanUp();" in added
    # ...and the bare call it replaced is gone
    assert "-    temporaryFileManager->cleanUp();" in patch


def test_0036_a_flush_with_no_instance_leaves_the_record():
    patch = _patch(36)
    assert "tracktion_ExternalPlugin.cpp" in patch
    added = _added(patch)
    assert "if (getAudioPluginInstance() != nullptr)" in added
    assert "Plugin::flushPluginStateToValueTree();" in added


def test_0037_unchanged_mapping_rows_are_not_rewritten():
    patch = _patch(37)
    assert "tracktion_ParameterControlMappings.cpp" in patch
    added = _added(patch)
    assert "isEquivalentTo" in added and "if (same)" in added
    # a table that differs is still written through the undo manager
    assert "state.addChild (row, -1, um);" in added


def test_0038_a_macro_counts_as_something_holding_the_knob():
    patch = _patch(38)
    assert "tracktion_ExternalPlugin.cpp" in patch
    assert "-            if (! getModifiersOfType<Modifier> (*param).isEmpty())" in patch
    assert "if (! param->getModifiers().isEmpty())" in _added(patch)


def test_the_consoles_engine_temp_folder_is_its_own():
    """A console leg's engine must not work in the app's ~/Library/Zammoth/Temporary: it left a
    folder there for every scratch edit and swept a running app's in-flight renders."""
    at = SESSION.index("struct EngineSettings  : te::PropertyStorage")
    body = SESSION[at: SESSION.index("\n};\n", at)]
    over = body.index("juce::File getAppCacheFolder() override")
    guard = body.rfind("#ifdef MIXAGENT_DAW_CONSOLE", 0, over)
    assert guard != -1 and "#endif" not in body[guard:over], "the override is not inside the console build's #ifdef"
    assert '"MixAgentDAW-console"' in body[over: body.index("#endif", over)]
    # the APP keeps the engine's own folder: there is exactly one override, and it is that one
    assert SESSION.count("getAppCacheFolder() override") == 1


def test_a_command_inside_the_engines_reset_window_is_marked_again():
    """S1.R23 c. The watch is in invokeCommand (every caller's door), it leaves a command that
    only looks alone, the edit switch arms it, the tick pays it, and a save clears it."""
    at = APP.index("    juce::var invokeCommand (int id, const juce::var& args)")
    head = APP[at: APP.index("dawcmd::badPathArg (args, why)", at)]
    assert "struct ResetWindowWatch" in head and "resetWindowWatch (*this, commandOnlyLooks);" in head
    assert "commandOnlyLooks = dawcmd::looksOnly (s.key);" in head
    watch = head[head.index("struct ResetWindowWatch"):]
    assert "d.editVersion() != before" in watch and "d.edit.get() != was" in watch

    switch = APP[APP.index("    juce::var openEditItem (const te::ProjectItem::Ptr& item)"):]
    switch = switch[: switch.index("\n    }\n")]
    assert "editOpenedAtMs = juce::Time::getMillisecondCounter();" in switch

    tick = APP[APP.index("if (changedInResetWindow && edit != nullptr)"):][:400]
    assert "edit->markAsChanged();" in tick and "++resetWindowTicksPast >= 2" in tick

    save = APP[APP.index("    bool saveSession()\n"):][:2500]
    assert "if (ok) changedInResetWindow = false;" in save


def test_the_engines_explicit_reset_is_not_patched_to_cancel_its_timer():
    """The fix the triage sketched (cancel the resetter from resetChangedStatus) would leave
    every project reading "unsaved" a moment after it opened: the constructor calls that before
    the engine's deferred updates have run. No patch may touch the resetter."""
    for path in sorted(PATCHES.glob("[0-9][0-9][0-9][0-9]-*.patch")):
        assert "EditChangeResetterTimer" not in _added(path.read_text(encoding="utf-8")), path.name
