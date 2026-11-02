"""Phase 65 step 1 (2026-10-24): a track holds as many plug-ins as the app's one setting says.

The brain names the cap when a send cannot be placed; the number it says is the DAW's, read here
from DawSession.h, and no sentence the app or the brain prints says "sixteen" of a track's plug-ins.
"""
from __future__ import annotations

import re
from pathlib import Path

from orchestrator import daw_bridge

ROOT = Path(__file__).resolve().parents[2]
SESSION_H = ROOT / "daw" / "Source" / "DawSession.h"


def _cpp_cap(name: str) -> int:
    m = re.search(rf"inline constexpr int {name}\s*=\s*(\d+);", SESSION_H.read_text())
    assert m, f"{name} not found in DawSession.h"
    return int(m.group(1))


def test_the_brain_names_the_daws_track_cap():
    assert daw_bridge.TRACK_PLUGIN_CAP == _cpp_cap("kTrackPluginCap")


def test_the_engine_reads_the_caps_from_the_one_setting():
    text = SESSION_H.read_text()
    body = text[text.index("te::EditLimits getEditLimits() override"):][:400]
    for field, name in (("maxPluginsOnTrack", "kTrackPluginCap"),
                        ("maxNumMasterPlugins", "kMasterPluginCap"),
                        ("maxPluginsOnClip", "kClipPluginCap")):
        assert f"{field}" in body and name in body, field


def test_no_sentence_says_a_track_holds_sixteen_plugins():
    said = re.compile(r'"[^"\n]*\bsixteen (?:plug-ins|devices)\b[^"\n]*"')
    for path in (ROOT / "daw" / "Source" / "App.cpp", ROOT / "orchestrator" / "daw_bridge.py"):
        assert not said.findall(path.read_text()), path.name
