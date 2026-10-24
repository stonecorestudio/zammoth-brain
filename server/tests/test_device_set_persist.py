"""Reader C3.R3 (full audit 2026-10-02, batch M9): where ONE parameter write is told to the brain.

`device.set` posted `/api/device_param` for every write. The brain has no `gain` device and no
`width` device on a stem, so the agent's fader or strip-width move was refused there and the next
chain push took it back. The DAW now picks the door per kind of control (`paramPersistDoor`,
native/Source/DeviceSpecs.h), shared by the dock's own knob and the command. This file holds the
brain's half: each URL that function builds is one the brain takes - and the old one is refused,
which is the defect.

The C++ half (which URL is queued) is `MixAgentDAW --device-set-persist`.
"""
from __future__ import annotations

import re
from pathlib import Path

from mix_engine.session import Session
from orchestrator.daw_bridge import apply_coedits

REPO = Path(__file__).resolve().parents[2]


def _apply(project, url: str) -> dict:
    return apply_coedits(project, [url])


def test_the_old_door_is_refused_for_the_fader_and_the_strip_width(demo_project):
    """The control: what device.set used to queue."""
    gain = _apply(demo_project, "/api/device_param?stem=stem_a&device_type=gain&occurrence=0&param=gain_db&value=-6.0000")
    width = _apply(demo_project, "/api/device_param?stem=stem_a&device_type=width&occurrence=0&param=width&value=0.5000")
    assert gain["failed"] == 1 and width["failed"] == 1
    s = Session(demo_project)
    assert s.stems["stem_a"]["gain_db"] == 0.0 and s.stems["stem_a"].get("width", 1.0) == 1.0


def test_a_tracks_fader_and_width_go_through_set_stem(demo_project):
    assert _apply(demo_project, "/api/set_stem?stem=stem_a&gain_db=-6.00")["failed"] == 0
    assert _apply(demo_project, "/api/set_stem?stem=stem_a&width=0.5000")["failed"] == 0
    s = Session(demo_project)
    assert s.stems["stem_a"]["gain_db"] == -6.0
    assert s.stems["stem_a"]["width"] == 0.5


def test_a_groups_fader_goes_through_set_stem_under_the_dock_s_name_for_it(demo_project):
    bus = Session(demo_project).group_stems(["stem_a"])["bus_id"]
    assert _apply(demo_project, f"/api/set_stem?stem=bus%3A{bus}&gain_db=-3.00")["failed"] == 0
    assert Session(demo_project).buses[bus]["gain_db"] == -3.0


def test_the_masters_gain_goes_through_chain_param(demo_project):
    assert _apply(demo_project, "/api/chain_param?stem=master&device_type=gain&occurrence=0&value=-3.0000")["failed"] == 0
    assert Session(demo_project).master_gain_db == -3.0


def test_a_returns_level_goes_through_set_return_level(demo_project):
    rid = Session(demo_project).add_return("blank")["return_id"]
    # What the dock's old Gain door sent for a return: chain_param. It indexed the groups (a
    # KeyError, so the move was dropped); since batch Py2 the brain sets the return's level for
    # it too - a journal written before the dock changed doors still replays.
    old = _apply(demo_project, f"/api/chain_param?device_type=gain&occurrence=0&value=-7.0000&stem=return%3A{rid}")
    assert old["failed"] == 0 and Session(demo_project).returns[rid]["gain_db"] == -7.0
    assert _apply(demo_project, f"/api/set_return_level?return={rid}&gain_db=-4.00")["failed"] == 0
    # the dock's host appends the stem it shows to every URL: the endpoint names its own target
    assert _apply(demo_project, f"/api/set_return_level?return={rid}&gain_db=-5.00&stem=return%3A{rid}")["failed"] == 0
    assert Session(demo_project).returns[rid]["gain_db"] == -5.0


def test_device_set_and_the_dock_share_the_door():
    """Neither site builds its own URL for a parameter write any more."""
    app = (REPO / "daw/Source/App.cpp").read_text(encoding="utf-8")
    dock = (REPO / "daw/Source/DeviceChainView.h").read_text(encoding="utf-8")
    specs = (REPO / "native/Source/DeviceSpecs.h").read_text(encoding="utf-8")
    assert "inline ParamPersistDoor paramPersistDoor" in specs
    handler = app[app.index("case dawcmd::deviceSet:"):]
    handler = handler[:handler.index("case dawcmd::pluginRemove:")]
    assert "paramPersistDoor (trackId" in handler
    assert not re.search(r'queueCoedit \("/api/device_param', handler), "device.set builds its own device_param URL again"
    edit_param = dock[dock.index("    void editParam (const juce::String& type, int occ, const juce::String& param, double value, bool persist)"):]
    edit_param = edit_param[:edit_param.index("    void setEnabled (")]
    assert "paramPersistDoor (selectedName" in edit_param
    assert "chain_param?device_type=gain" not in edit_param, "the dock's knob builds the fader's URL by hand again"
