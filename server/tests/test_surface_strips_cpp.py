"""A control surface's channel strips (2026-10-09) - what a leg cannot hold, pinned.

The behaviour is driven by the console leg `--surface-strips` (daw/Source/ControlSurfaceLegs.h),
over real CoreMIDI virtual ports: a fader, a pot, MUTE / SOLO / SELECT / REC and a send move the
app's own controls and tell the brain; a frozen track refuses; the app's faders reach the surface.
Here: every hook the engine patch offers is taken by the app and given back at teardown, each
write goes through one of the app's commands, and no control writes the engine's own plug-in.

Record: docs/audits/user-qa-2026-10-08/QA-log.md, *The control surface's strips*.
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
SESSION_H = (SRC / "DawSession.h").read_text(encoding="utf-8")
LEG = (SRC / "ControlSurfaceLegs.h").read_text(encoding="utf-8")

pytestmark = pytest.mark.reads_cpp


def _patch(number: str) -> str:
    found = sorted(PATCHES.glob(f"{number}-*.patch"))
    assert len(found) == 1, f"engine patch {number}: {[p.name for p in found]}"
    return found[0].read_text(encoding="utf-8")


def _body(source: str, start: str, end: str = "\n    }\n") -> str:
    i = source.index(start)
    return source[i: source.index(end, i)]


def _hooks_the_engine_offers() -> list[str]:
    """The std::function members patch 0041 adds to ExternalControllerManager."""
    added = [line[1:] for line in _patch("0041").splitlines() if line.startswith("+") and not line.startswith("+++")]
    names = [m.group(1) for line in added for m in [re.search(r"std::function<.*>\s+(host\w+);", line)] if m]
    assert len(names) == len(set(names))
    return names


def test_the_patch_offers_the_hooks_this_test_expects():
    assert set(_hooks_the_engine_offers()) == {
        "hostMovedFader", "hostMovedMasterFader", "hostMovedPanPot", "hostPressedMute", "hostPressedSolo",
        "hostPressedRecEnable", "hostMovedAux", "hostPressedAux", "hostFaderPosition", "hostMasterFaderPosition"}


def test_the_app_takes_every_hook_and_gives_every_one_back():
    """A hook left unset is a control that still writes the engine's hidden plug-in; one left set
    at teardown is a call into a component whose members are gone (the engine outlives them)."""
    wire = _body(APP, "    void wireSurfaceStrips()")
    unwire = _body(APP, "    void unwireSurfaceStrips()")
    for name in _hooks_the_engine_offers() + ["isVisibleOnControlSurface"]:
        assert re.search(rf"ecm\.{name} = \[", wire), f"{name} is not wired"
        assert re.search(rf"ecm\.{name} = nullptr;", unwire), f"{name} is not given back"


def test_the_hooks_are_set_before_the_edit_is_and_cleared_first_in_the_destructor():
    ctor_attach = APP.index("dawsession::attachControlSurfaces (engine, *edit, &selectionManager);")
    assert APP.rindex("wireSurfaceStrips();", 0, ctor_attach) > ctor_attach - 400
    dtor = _body(APP, "    ~DawComponent() override")
    assert dtor.index("stopTimer();") < dtor.index("unwireSurfaceStrips();") < dtor.index("flushCoeditsNow")


def test_every_write_hook_answers_handled_and_runs_a_command_or_says_why_not():
    """`return false` anywhere in the wiring is the engine writing the track itself."""
    wire = _body(APP, "    void wireSurfaceStrips()")
    assert "return false" not in wire
    block = APP[APP.index("    // A CONTROL SURFACE'S CHANNEL STRIPS (2026-10-09; engine patch 0041"): APP.index("    void unwireSurfaceStrips()")]
    for command in ("dawcmd::setStemGain", "dawcmd::setStemPan", "dawcmd::trackMuteToggle", "dawcmd::trackSolo",
                    "dawcmd::recordArm", "dawcmd::sendSet"):
        assert f"surfaceInvoke ({command}," in block, command
    # ...and nothing in it writes the engine's level, pan, mute or solo directly
    for direct in ("setSliderPos", "setVolumeDb", "->setPan (", "setMute (", "setSolo (", "setGainDb (", "setSoloIsolate"):
        assert direct not in block, direct


def test_a_refusal_is_said_in_the_top_bar_and_never_through_the_feeds_door():
    """A fader dragged on a frozen track sends a hundred messages a second: setStatus would put
    each refusal in the engineer's feed."""
    block = APP[APP.index("    void surfaceSay (const juce::String& text)"): APP.index("    void wireSurfaceStrips()")]
    assert "sayStatus (text);" in block
    assert "setStatus (" not in block and "invokeChecked (" not in block


def test_only_the_project_on_screen_takes_a_surfaces_moves():
    wire = _body(APP, "    void wireSurfaceStrips()")
    writes = re.findall(r"ecm\.(host(?:Moved|Pressed)\w+) = \[this\][^{]*\{(.*?)\n        \};", wire, flags=re.S)
    assert len(writes) == 8
    for name, body in writes:
        assert "surfaceIsMine (" in body, name
    assert "! inBackground" in _body(APP, "    bool surfaceIsMine (te::Edit* e) const", "\n")
    sync = _body(APP, "    void syncSurfaceFaders()")
    assert "inBackground" in sync.split("\n")[2]


def test_the_strips_are_tracks_and_the_fader_law_is_the_faders_range():
    is_channel = _body(SESSION_H, "inline bool isSurfaceChannel (const te::Track& t)", "\n}\n")
    assert "isAgentTrack (&t)" in is_channel and "isSinkTrack (&t)" in is_channel and "te::AudioTrack" in is_channel
    to_db = _body(SESSION_H, "inline double surfacePosToFaderDb (double pos)", "\n}\n")
    assert "dawcmd::kMinLevelDb" in to_db and "dawcmd::kMaxLevelDb" in to_db and "volumeFaderPositionToDB" in to_db
    send_db = _body(SESSION_H, "inline double surfacePosToSendDb (double pos)", "\n}\n")
    assert "dawcmd::kMaxSendDb" in send_db


def test_a_fader_moved_on_a_surface_is_a_hand_on_the_fader():
    """Armed and playing it records a ride, as the strip's and the header's do (D56)."""
    moved = _body(APP, "    bool surfaceMovedFader (const juce::String& id")
    assert moved.index("surfaceInvoke (dawcmd::setStemGain") < moved.index("rideGain (id, (float) db);")
    assert "faderHeld.addIfNotAlreadyThere (id);" in moved and "pushSurfaceFader (id);" in moved
    assert "rideRelease (id)" in _body(APP, "    void syncSurfaceFaders()")


def test_closing_the_devices_tells_the_surfaces():
    """Patch 0042: a surface held pointers to MIDI devices the engine had just freed, and its
    blink timer sent through one 750 ms later - any project with a surface bound to an output
    port crashed the app when it left the foreground."""
    patch = _patch("0042")
    assert "tracktion_DeviceManager.cpp" in patch
    assert "+    engine.getExternalControllerManager().midiInOutDevicesChanged();" in patch


def test_the_leg_is_wired_and_drives_real_ports():
    assert 'flag == "--surface-strips"' in MAIN and "dawapp::runSurfaceStrips (project)" in MAIN
    assert '#include "ControlSurfaceLegs.h"' in APP
    assert "juce::MidiOutput::createNewDevice" in LEG and "juce::MidiInput::createNewDevice" in LEG
    assert "sendMessageNow" in LEG and "m14::Sandbox" in LEG
    # nothing is simulated inside the engine: no surface handler is called by the leg
    assert not re.search(r"getControlSurface\(\)\.user\w+", LEG)


def test_no_temporary_switch_is_left_in_the_tree():
    # Every red-first control switch, whatever its name: a `*_OLD` environment read or a
    # TEMPORARY- marker, in the app's source AND native/ (2026-10-17: the compressor audit's
    # COMP_OLD switch was committed - its removal script stopped at its first step, and this
    # test knew only two names).
    import re
    native = SRC.parent.parent / "native" / "Source"
    paths = [p for d in (SRC, native) for p in list(d.glob("*.h")) + list(d.glob("*.cpp"))]
    assert len(paths) > 20
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        assert "ZAM_TMP_SURFACE" not in text and "TEMP-MUTATION" not in text, path.name
        assert "TEMPORARY-" not in text, path.name
        assert not re.search(r'getenv\s*\(\s*"[A-Z0-9_]*_OLD"', text), path.name
        assert not re.search(r'getEnvironmentVariable\s*\(\s*"[A-Z0-9_]*_OLD"', text), path.name
