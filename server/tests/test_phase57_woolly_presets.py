"""Phase 57 - WOOLLY tells the truth and keeps your sounds (2026-10-12).

What these read is source: the rules a console leg cannot hold because they are about what the
NEXT edit must not do. The behaviour itself is `MixAgentDAW --woolly-knob-truth`,
`--woolly-knob-undo`, `--woolly-patch-doc` and `--woolly-presets`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "daw" / "Source"
pytestmark = pytest.mark.reads_cpp


def _read(name: str) -> str:
    return (SRC / name).read_text(encoding="utf-8")


def _product_sources() -> dict[str, str]:
    """Every source the app is built from, the console's legs left out."""
    return {p.name: p.read_text(encoding="utf-8", errors="replace")
            for p in sorted(SRC.glob("*")) if p.suffix in (".h", ".cpp", ".mm")
            and "Leg" not in p.name and p.name != "Main.cpp"}


def _function(text: str, signature: str) -> str:
    start = text.index(signature)
    brace = text.index("{", start)
    depth, i = 0, brace
    while True:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return text[start:i + 1]
        i += 1


def test_a_state_built_by_hand_carries_the_stamp():
    """A 4OSC state with no stamp is one written under the old additive law, and the plug-in
    folds it: a new state that forgot the stamp would have its creation routes folded into
    knobs that are already right - every level and the cutoff pinned at the top again."""
    session = _read("DawSession.h")
    made = _function(session, "inline juce::ValueTree defaultSynthState()")
    assert "parkedControllerStamp" in made
    builders = [name for name, text in _product_sources().items()
                if re.search(r"setProperty \(te::IDs::type, te::FourOscPlugin::xmlTypeName", text)]
    assert builders == ["DawSession.h"], (
        f"a 4OSC state is built by hand in {builders}: build it in defaultSynthState, "
        "which carries the stamp")


def test_every_door_that_makes_a_synth_makes_init():
    """A new synth IS the INIT preset (James, 2026-10-12): both doors go through newSynth."""
    for name, text in _product_sources().items():
        for hit in re.finditer(r"createNewPlugin \(defaultSynthState\(\)\)", text):
            where = text[:hit.start()].rsplit("\ninline ", 1)[-1][:60]
            assert name == "DawSession.h" and where.startswith("te::Plugin::Ptr newSynth"), (
                f"{name} makes a synth outside dawsession::newSynth")
    made = _function(_read("DawSession.h"), "inline te::Plugin::Ptr newSynth (te::Edit& edit)")
    assert "synthResetToInit" in made and '"factory:INIT"' in made


def test_an_undo_and_a_redo_both_hand_the_knobs_back():
    app = _read("App.cpp")
    undo = app[app.index("case dawcmd::editUndo:"):app.index("case dawcmd::editRedo:")]
    redo = app[app.index("case dawcmd::editRedo:"):app.index("case dawcmd::automationEdit:")]
    assert "resyncInstrumentKnobsAfterUndo" in undo
    assert "resyncInstrumentKnobsAfterUndo" in redo


def test_the_engine_patch_is_the_scale_law_and_the_fold():
    patch = next((REPO / "daw" / "patches").glob("0045-*.patch")).read_text(encoding="utf-8")
    assert "isParkedController" in patch and "foldParkedControllerState" in patch
    # the fold runs before the matrix is read from the state, in the constructor
    ctor = patch[patch.index("if (foldParkedControllerState (state))"):]
    assert ctor.index("updateFromAttachedValue") < ctor.index("loadModMatrix();")


def _switch_fields() -> set[str]:
    body = _function(_read("DawSession.h"), "inline juce::StringArray synthSwitchFields()")
    return set(re.findall(r'"([a-z0-9]+\.[a-z]+)"', body))


def _knob_ids() -> set[str]:
    ids = {f"{stem}{n}" for n in range(1, 5)
           for stem in ("tune", "fineTune", "level", "pulseWidth", "detune", "spread", "pan")}
    ids |= {f"{stem}{n}" for n in (1, 2)
            for stem in ("lfoRate", "lfoDepth", "modAttack", "modDecay", "modSustain", "modRelease")}
    ids |= {"ampAttack", "ampDecay", "ampSustain", "ampRelease", "ampVelocity",
            "filterAttack", "filterDecay", "filterSustain", "filterRelease", "filterFreq",
            "filterResonance", "filterAmount", "filterKey", "filterVelocity",
            "distortion", "reverbSize", "reverbDamping", "reverbWidth", "reverbMix",
            "delayFeedback", "delayCrossfeed", "delayMix",
            "chorusSpeed", "chorusDepth", "chorusWidth", "chorusMix", "legato", "masterLevel"}
    assert len(ids) == 68
    return ids


def test_the_factory_bank_is_documents_in_the_synths_own_names():
    """Seven when this was written; fifty-eight since Phase 60 redesigned the bank (its census
    and its measured checks are test_phase60_sound_design.py's and `--factory-bank`'s)."""
    presets = _read("WoollyPresets.h")
    table = presets[presets.index("inline const std::vector<Factory>& factory()"):presets.index("inline const char* initRoutes()")]
    rows = re.findall(r'\{ "([A-Z0-9 ]+)", "([A-Z]*)", "([^"]+)", R"\((.*?)\)"', table, flags=re.S)
    names = [r[0] for r in rows]
    assert names[:2] == ["INIT", "SOFT KEYS"] and len(names) == len(set(names)) >= 48
    assert {"WARM PAD", "BIG SAW", "PLUCK", "SUB BASS", "GLASS BELLS"} <= set(names)     # the first seven are still there
    fields, knobs = _switch_fields(), _knob_ids()
    assert len(fields) == 26     # 24 + pitch.mod and voice.phase (Phase 58)
    for name, _category, description, text in rows:
        doc = json.loads(text)
        assert set(doc) <= {"set", "knobs", "routes"}, name
        assert set(doc.get("set", {})) <= fields, f"{name}: {set(doc.get('set', {})) - fields}"
        assert set(doc.get("knobs", {})) <= knobs, f"{name}: {set(doc.get('knobs', {})) - knobs}"
        assert description.strip() and len(description) <= 200
        # a level and the cutoff are on the KNOB: no preset reaches for the parked controllers
        assert not [r for r in doc.get("routes", []) if r.get("source") in ("cc11", "cc74")], name


def test_the_grammar_says_where_the_factory_bank_is_listed():
    """It named all seven in order until the bank grew past fifty; now it names a few and
    the command that lists them, with the kind and the key each is written for."""
    commands = _read("DawCommands.h")
    line = commands[commands.index('{ "instrument.set_patch",'):commands.index('{ "instrument.preset_save",')]
    assert "a factory preset's name (INIT, WARM PAD, KICK DEEP, BOOM BASS" in line
    assert 'instrument.preset_list {category: \\"factory\\"} lists the bank' in line
    listed = commands[commands.index('{ "instrument.preset_list",'):commands.index('{ "sound.render",')]
    assert "kind: the kind of sound it is" in listed and "note: the key it is" in listed


def test_the_new_arguments_are_single_values_at_both_doors():
    from orchestrator import command_door
    cpp = _function(_read("DawCommands.h"), "inline bool isScalarArgName (const juce::String& k)")
    for name in ("preset", "category", "description", "overwrite", "clear_routes"):
        assert f'"{name}"' in cpp and name in command_door.SCALAR_ARG_NAMES, name
    # `notes` is midi.add_notes' LIST: a preset's line of text must not be called that
    assert '"notes"' not in cpp and "notes" not in command_door.SCALAR_ARG_NAMES
    for part in ("set", "knobs", "routes", "tags"):
        assert f'"{part}"' not in cpp, f"{part} is an object or a list"


def test_a_delete_and_a_save_over_ask_the_user():
    commands = _read("DawCommands.h")
    row = next(line for line in commands.splitlines() if '"instrument.preset_delete"' in line
               and "instrumentPresetDelete," in line)
    assert re.search(r"true,\s*false,\s*true\s*\}", row), "instrument.preset_delete is confirm-gated"
    asks = _function(_read("App.cpp"), "static bool needsUserYes (")
    assert '"instrument.preset_save"' in asks and '"overwrite"' in asks


def test_a_preset_is_written_whole_and_removed_to_the_trash():
    presets = _read("WoollyPresets.h")
    assert "wholefile::write (file," in presets
    assert ".replaceWithText (" not in presets and ".replaceWithData (" not in presets
    trash = _function(presets, "inline juce::String trash (const Ref& ref)")
    assert "moveToTrash()" in trash


def test_both_prompts_say_one_call_and_name_the_preset_commands():
    from orchestrator import chat, prompts
    for text in (chat.daw_prompt(), prompts.DAW_PROMPT_WHOLE):
        assert "ONE `instrument.set_patch`" in text
        assert "A new synth is ONE sine" not in text
        for key in ("instrument.preset_save", "instrument.preset_list",
                    "instrument.preset_rename", "instrument.preset_delete"):
            assert f"`{key}`" in text
