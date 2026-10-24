"""A component's visibility is set AFTER it is added, never before `addAndMakeVisible`.

`addAndMakeVisible (x)` is `addChildComponent (x)` plus `x.setVisible (true)`, so

    x.setVisible (! drumMode());
    addAndMakeVisible (x);

shows `x` whatever the first line decided. Found on screen 2026-10-03, three times
in the piano roll's constructor: a drum roll carried a FOLD chip it cannot use, a
melodic roll showed SCALE with no project key, and the "Aa" note-name button sat
under the headphone button in a drum roll's corner (they share that slot). The
SCALE one also left the header's instrument tag drawn twice ("DRUM KIT  DRUM
KIT"), because the timer hid the chip a moment later and hiding repaints only the
chip's own rectangle.

What is correct: `addChildComponent (x);` and then `x.setVisible (cond);`.

This scans the DAW's and the shared native sources for a `name.setVisible (…)`
followed within three lines by `addAndMakeVisible (name)`.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SOURCES = [*sorted((REPO / "daw" / "Source").glob("*.h")),
           *sorted((REPO / "daw" / "Source").glob("*.cpp")),
           *sorted((REPO / "native" / "Source").glob("*.h")),
           *sorted((REPO / "native" / "Source").glob("*.cpp"))]

_SET_VISIBLE = re.compile(r"(\w+)\s*\.\s*setVisible\s*\(")
LOOKAHEAD_LINES = 3


def undone_visibility(text: str) -> list[tuple[int, str]]:
    """(line number, line) of every setVisible that the next lines' addAndMakeVisible undoes."""
    lines = text.split("\n")
    found = []
    for i, line in enumerate(lines):
        m = _SET_VISIBLE.search(line)
        if m is None:
            continue
        again = re.compile(r"addAndMakeVisible\s*\(\s*&?\s*" + re.escape(m.group(1)) + r"\s*[,)]")
        if any(again.search(later) for later in lines[i + 1:i + 1 + LOOKAHEAD_LINES]):
            found.append((i + 1, line.strip()))
    return found


def test_the_scanner_sees_the_fault_it_is_for():
    bad = "foldBtn.setVisible (! drumMode());\naddAndMakeVisible (foldBtn);\n"
    good = "addChildComponent (foldBtn);\nfoldBtn.setVisible (! drumMode());\n"
    other = "foldBtn.setVisible (false);\naddAndMakeVisible (scaleBtn);\n"
    assert [n for n, _ in undone_visibility(bad)] == [1]
    assert undone_visibility(good) == []
    assert undone_visibility(other) == []


def test_no_visibility_is_set_before_add_and_make_visible():
    assert SOURCES, "no C++ sources found - the paths moved"
    offenders = [f"{path.relative_to(REPO)}:{n}: {line}"
                 for path in SOURCES
                 for n, line in undone_visibility(path.read_text(encoding="utf-8", errors="replace"))]
    assert not offenders, (
        "setVisible is undone by the addAndMakeVisible after it "
        "(use addChildComponent, then setVisible):\n" + "\n".join(offenders))
