"""No dashed or dotted lines on a device (2026-10-21).

James: "remove any dotted lines in the devices and replace with straight lines". Every line a
device face, a dock box or a device editor draws is solid; where a dash told two things apart,
weight and brightness do it now (docs/10-daw.md, *Devices are smoked glass*). This reads the
device sources for the two JUCE calls that draw a dash.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# The files that draw a device: the faces, the dock and its boxes, the instrument editors.
DEVICE_SOURCES = [
    "native/Source/DeviceGraphs.h",
    "native/Source/ChainMeters.h",
    "daw/Source/DeviceChainView.h",
    "daw/Source/WoollyEditor.h",
    "daw/Source/SimplerEditor.h",
    "daw/Source/ArpEditor.h",
    "daw/Source/ArpLane.h",
    "daw/Source/DrumRackView.h",
]

DASH_CALL = re.compile(r"\b(drawDashedLine|createDashedStroke)\b")


def _code (text: str) -> str:
    """The text with // comments taken out (a comment may say what a line used to be)."""
    return "\n".join (line.split ("//", 1)[0] for line in text.splitlines())


def dash_calls (text: str) -> list[int]:
    """1-based line numbers of every dash-drawing call in the code."""
    return [i + 1 for i, line in enumerate (_code (text).splitlines()) if DASH_CALL.search (line)]


def test_no_device_draws_a_dashed_line():
    found = {}
    for rel in DEVICE_SOURCES:
        lines = dash_calls ((REPO / rel).read_text (encoding="utf-8"))
        if lines:
            found[rel] = lines
    assert not found, ("a device draws a dashed line - draw it solid and tell two lines apart by "
                       "weight and brightness instead: " + repr (found))


def test_the_check_sees_a_dash():
    # the control: the check must fire on the two calls it is looking for
    assert dash_calls ("g.drawDashedLine (line, dashes, 2, 1.0f);") == [1]
    assert dash_calls ("PathStrokeType (1.6f).createDashedStroke (d, p, dash, 2);") == [1]
    assert dash_calls ("// it was drawDashedLine until 2026-10-21") == []
