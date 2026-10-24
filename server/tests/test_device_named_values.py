"""A device field's NAMES are one list at both ends (2026-10-20, the agent's failed device calls).

The brain teaches the engineer a saturation curve, a limiter style and an IR as words, and an EQ
band's shape, a channel and a band label are words on the chain - and `device.add` /
`device.set` took numbers only: `{"curve": "warm"}`, `{"saturation.curve": "tube"}` and
`{"eq.shape": "hpf"}` were each refused, in real sessions, and retried. The DAW's tables
(`deviceValueNames`, `deviceWordValues` in native/Source/DeviceSpecs.h) are pinned here against
the brain's own lists, and the command door's transcription is held to not judging a device's
fields by the command-argument name rule. The live behaviour is `--device-add` (i)-(m).
"""
from __future__ import annotations

import re
from pathlib import Path

from mix_engine import models
from mix_engine.session import IR_MODES, Session
from orchestrator import command_door

REPO = Path(__file__).resolve().parents[2]
SPECS = (REPO / "native" / "Source" / "DeviceSpecs.h").read_text()
DOOR = (REPO / "daw" / "Source" / "DawCommands.h").read_text()


def _list(name: str) -> list[str]:
    m = re.search(r"static const juce::StringArray " + name + r"\s*\{([^}]*)\}", SPECS)
    assert m, f"DeviceSpecs.h has no list {name}"
    return re.findall(r'"([^"]+)"', m.group(1))


def test_the_named_values_are_the_brains_own_lists():
    assert _list("satCurves") == list(models.SAT_CURVES)
    assert _list("limStyles") == list(models.LIMITER_STYLES)
    assert _list("irModes") == list(IR_MODES)


def test_the_auto_filters_names_are_the_brains_own_lists():
    # Phase 63: lower case at the native end, as the door reads a word before it looks it up
    assert _list("afTypes") == list(models.AUTO_FILTER_TYPES)
    assert _list("afShapes") == list(models.AUTO_FILTER_SHAPES)
    assert _list("afNotes") == [n.lower() for n in models.AUTO_FILTER_NOTE_NAMES]


def test_the_word_fields_are_the_brains_own_lists():
    assert _list("eqShapes") == list(Session.EQ_SHAPES)
    assert _list("bands") == list(models.BAND_LABELS)
    assert _list("eqChannels") == ["stereo", "mid", "side", "left", "right"]
    assert _list("msChannels") == ["stereo", "mid", "side"]   # a comp has no left / right (set_device_channel)


def test_the_door_does_not_judge_a_devices_fields_by_argument_names():
    # device.add's params: saturation's curve is a NAME there, not automation's curve number
    assert command_door.bad_numeric_arg(
        {"track": "Sub Bass", "type": "saturation",
         "params": {"drive_db": 7, "curve": "warm", "mix": 0.6, "oversample": 4}}) is None
    # ...while a command's own `curve` still is a number
    assert command_door.bad_numeric_arg({"track": "a", "points": [{"t_s": 0, "value": 1, "curve": "warm"}]})
    assert command_door.bad_numeric_arg({"curve": "nan"})
    # ...and a number in params is still checked for plausibility
    assert command_door.bad_numeric_arg({"params": {"drive_db": 1e300}})
    # the C++ door carries the same rule
    assert 'badNumericArg (x, why, deviceFields || key == "params")' in DOOR
    assert "if (deviceFields || ! isNumericArgName (key)" in DOOR
