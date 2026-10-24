"""A control surface's other keys (2026-10-09) - what a leg cannot hold, pinned.

The behaviour is driven by the console leg `--surface-keys` (daw/Source/ControlSurfaceLegs.h),
over real CoreMIDI virtual ports: Undo tells the brain, F2 switches nothing on the master off,
the automation keys never reach the engine's own recorder, Arm All goes through the arm door, a
marker key is held under a take, and with two projects open a key acts on the one in front.
Here: every key the engine patch offers is answered by the app - by one of its commands, or held
and said - and none falls through to the engine's own handler without that being written down.

Record: docs/audits/user-qa-2026-10-08/QA-log.md, *The control surface's other keys*.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "daw" / "Source"
PATCHES = ROOT / "daw" / "patches"
APP = (SRC / "App.cpp").read_text(encoding="utf-8")
MAIN = (SRC / "Main.cpp").read_text(encoding="utf-8")
LEG = (SRC / "ControlSurfaceLegs.h").read_text(encoding="utf-8")

pytestmark = pytest.mark.reads_cpp

# The three keys the app hands back to the engine, and why that is safe: the engine moves the
# playhead of the surface's OWN edit for them (never te::AppFunctions, which acts on the first
# engine made), and holds them under a rolling take itself (patch 0021).
ENGINE_KEEPS = {"rewind", "fastForward", "jog"}

# Keys that would switch something off underneath the app, or start something the app has no
# door for: these must be HELD, whatever else changes.
MUST_BE_HELD = {"masterPluginsPower", "trackPluginsPower", "pluginPower", "pluginPreset", "saveAs",
                "abort", "abortRestart", "timecodeChase", "playInStop", "setTempo",
                "insertTempoChange", "insertPitchChange", "insertTimeSigChange"}


def _patch() -> str:
    found = sorted(PATCHES.glob("0043-*.patch"))
    assert len(found) == 1, f"engine patch 0043: {[p.name for p in found]}"
    return found[0].read_text(encoding="utf-8")


def _added(patch: str) -> str:
    return "\n".join(line[1:] for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++"))


def _keys_the_engine_offers() -> list[str]:
    """The enumerators of `enum class SurfaceKey`, as patch 0043 adds it."""
    added = _added(_patch())
    body = added[added.index("enum class SurfaceKey"):]
    body = body[body.index("{") + 1: body.index("};")]
    body = re.sub(r"//[^\n]*", "", body)
    names = [n.strip() for n in body.split(",") if n.strip()]
    assert len(names) == len(set(names)) and len(names) > 60, names
    return names


def _handler() -> str:
    start = APP.index("    bool surfacePressedKey (te::SurfaceKey key, float arg)")
    return APP[start: APP.index("    // A KNOB ON A PARAMETER", start)]


def _case(handler: str, key: str) -> str:
    """The code a key's `case` label runs: from the label to the next `return` statement's end."""
    i = handler.index(f"case K::{key}:")
    rest = handler[i:]
    # the body ends where the next group of case labels begins after at least one return
    m = re.search(r"return [^;]*;(?:\s*\}\s*)?\n\s*(?=(?://[^\n]*\n\s*)*case K::|\})", rest, flags=re.S)
    assert m, key
    return rest[: m.end()]


def test_the_engine_offers_every_key_somewhere():
    """An enumerator nothing in the engine passes to the host is a key that still goes past it."""
    added = _added(_patch())
    for key in _keys_the_engine_offers():
        offered = (f"RETURN_IF_HOST_TOOK ({key})" in added or f"RETURN_IF_HOST_TOOK_WITH ({key}," in added
                   or re.search(rf"hostTookKey \([^;]*SurfaceKey::{key}\b", added))
        assert offered, f"{key} is in the enum and no engine handler offers it to the host"


def test_the_app_answers_every_key_the_engine_offers():
    handler = _handler()
    for key in _keys_the_engine_offers():
        assert f"case K::{key}:" in handler, f"{key} has no case in surfacePressedKey"
    # ...and the switch names no key the engine does not have
    for key in re.findall(r"case K::(\w+):", handler):
        assert key in _keys_the_engine_offers(), key
    assert "default:" not in handler, "a default would hide a key the engine gains"


def test_no_key_falls_through_to_the_engines_own_handler_unwritten():
    """`return false` is the engine doing what it always did. Three keys may: see ENGINE_KEEPS."""
    handler = _handler()
    assert "AppFunctions" not in handler.replace("te::AppFunctions, which acts", "").replace("falls through to te::AppFunctions", "")
    falls = [m.start() for m in re.finditer(r"return false;", handler)]
    assert len(falls) == 2, "only the rewind / fast-forward / jog case hands a key back"
    block = handler[handler.index("case K::rewind:"): handler.index("case K::markIn:")]
    assert block.count("return false;") == 2
    for key in ENGINE_KEEPS:
        assert f"case K::{key}:" in block, key
    # held under a take and under a render before the engine is let at the playhead
    assert "recordingNow()" in block and "exportActive.load()" in block and "freezeRendering" in block


def test_the_keys_that_switch_something_off_underneath_are_held_and_said():
    handler = _handler()
    for key in MUST_BE_HELD:
        body = _case(handler, key)
        assert "surfaceHeld (" in body, f"{key}: {body[:200]}"
        assert "invokeCommand" not in body and "run (" not in body, key


def test_undo_save_record_and_the_rest_run_the_apps_own_commands():
    handler = _handler()
    routed = {"undo": "dawcmd::editUndo", "redo": "dawcmd::editRedo", "save": "dawcmd::saveSession",
              "play": "dawcmd::transportPlayPause", "stop": "dawcmd::transportStop", "record": "dawcmd::recordToggle",
              "home": "dawcmd::transportReturnToStart", "createMarker": "dawcmd::markerAdd",
              "automationWrite": "dawcmd::automationRecord", "automationRead": "dawcmd::automationReenable",
              "armAll": "dawcmd::recordArm", "deleteSelected": "dawcmd::clipDelete", "loop": "dawcmd::loopFromSelection",
              "click": "dawcmd::metronomeSet", "snap": "dawcmd::gridToggleSnap", "freeze": "dawcmd::trackFreeze",
              "nextMarker": "dawcmd::goToMarker", "goToMarker": "dawcmd::goToMarker"}
    for key, command in routed.items():
        assert command in _case(handler, key), f"{key} does not run {command}"
    # nothing in the handler reaches the engine's undo, save or automation recorder directly
    for direct in ("edit->undo()", "edit->redo()", "getUndoManager().undo", "EditFileOperations", "toggleWriteAutomationMode",
                   "setReadingAutomation", "setRecordingEnabled", "flipAllPluginsEnablement", "->setEnabled ("):
        assert direct not in handler, direct


def test_a_freeze_is_posted_and_never_run_inside_the_surfaces_own_dispatch():
    """A freeze pumps the message loop: run from inside the controller's AsyncUpdater it would
    re-enter the surface's own message handling (the callAfterDelay rule)."""
    body = _case(_handler(), "freeze")
    assert "juce::MessageManager::callAsync" in body and "SafePointer<DawComponent>" in body
    assert body.index("callAsync") < body.index("dawcmd::trackFreeze")


def test_only_the_project_on_screen_takes_a_key():
    handler = _handler()
    assert "if (edit == nullptr || inBackground) return true;" in handler.split("switch (key)")[0]
    param = APP[APP.index("    bool surfaceMovedParameter (te::AutomatableParameter& p, float v, bool pressed)"):
                APP.index("    // The lights of three keys")]
    assert "if (edit == nullptr || inBackground) return true;" in param


def test_a_knob_on_the_channels_own_stage_is_the_mixers_command():
    param = APP[APP.index("    bool surfaceMovedParameter (te::AutomatableParameter& p, float v, bool pressed)"):
                APP.index("    // The lights of three keys")]
    assert "surfaceMovedPan (*t, v * 2.0f - 1.0f, false)" in param and "surfaceMovedSend (*t, *send, v)" in param
    for held in ("te::VCAPlugin", "te::RackInstance"):
        assert held in param
    # the engine's drivers move a parameter through the manager, where the host is asked first
    added = _added(_patch())
    assert added.count("externalControllerManager.userMovedParameter (*param, pos);") == 2      # the AlphaTrack's encoders
    assert added.count("externalControllerManager.userPressedParameter (*p);") == 3
    assert "getExternalControllerManager().userMovedParameter (*p," in added                    # a surface's plug-in mode
    assert "if (hostMovedParameter && hostMovedParameter (p, newNormalisedValue))" in added


def test_the_new_hooks_are_taken_and_given_back():
    wire = APP[APP.index("    void wireSurfaceStrips()"): APP.index("    void unwireSurfaceStrips()")]
    unwire_start = APP.index("    void unwireSurfaceStrips()")
    unwire = APP[unwire_start: APP.index("\n    }\n", unwire_start)]
    added = _added(_patch())
    hooks = [m.group(1) for m in re.finditer(r"std::function<[^;]*>\s+(host\w+);", added)]
    assert set(hooks) == {"hostPressedKey", "hostMovedParameter", "hostPressedParameter", "hostKeyLight"}
    for name in hooks:
        assert re.search(rf"ecm\.{name} = \[this\]", wire) or f"ecm.{name} =" in APP[APP.index("    void wireSurfaceKeys()"): APP.index("    void wireSurfaceStrips()")], name
        assert f"ecm.{name} = nullptr;" in unwire, f"{name} is not given back"


def test_the_marker_keys_are_held_under_a_take_in_the_engine_too():
    """Patch 0021 held Home, End, the jumps and the loop keys; the two marker keys and a marker
    picked on a pot were missed - each moves the playhead, and a position change ends a take."""
    added = _added(_patch())
    for name in ("userPressedNextMarker", "userPressedPreviousMarker"):
        line = next(l for l in added.splitlines() if f"void ControlSurface::{name}()" in l)
        assert "RETURN_IF_TAKE_ROLLING" in line and "RETURN_IF_HOST_TOOK" in line
        assert line.index("RETURN_IF_HOST_TOOK") < line.index("RETURN_IF_TAKE_ROLLING")   # the host says why first
    assert "if (ed->getTransport().isRecording())" in added


def test_three_lights_are_the_apps_state():
    added = _added(_patch())
    for key in ("automationWrite", "snap", "scroll"):
        assert f"getKeyLightToShow (SurfaceKey::{key}," in added, key
    light = APP[APP.index("    std::optional<bool> surfaceKeyLight (te::SurfaceKey key)"): APP.index("    void syncSurfaceLights()")]
    assert "automationArmed()" in light and "captureGridPrefs" in light
    assert "syncSurfaceLights();" in APP[APP.index("        syncSurfaceFaders();   // a bound control surface's faders"):][:400]


def test_the_leg_is_wired_and_drives_real_ports():
    assert 'flag == "--surface-keys"' in MAIN and "dawapp::runSurfaceKeys (project)" in MAIN
    leg = LEG[LEG.index("    int runSurfaceKeys (const juce::File& projectDir)"):]
    assert "juce::MidiOutput::createNewDevice" in leg and "sendMessageNow" in leg and "m14::Sandbox" in leg
    assert "m3::Told told;" in leg                         # "the brain is told" is what a stand-in brain hears
    assert not re.search(r"getControlSurface\(\)\.user\w+", leg)
    # the cases the batch was asked for
    for case in ('"(f2) ', '"(u1) ', '"(w) ', '"(a) ', '"(b) ', '"(m2) ', '"(r) '):
        assert case in leg, case
