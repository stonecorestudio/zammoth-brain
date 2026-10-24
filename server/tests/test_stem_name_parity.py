"""One rule for what a track may be called, at both ends (full audit 2026-10-02, reader S4·R10).

The DAW refused three prefixes and the brain every ``:``: "Vox: lead" was renamed in the app,
refused by the brain, and the track then sat with no stem — every later move on it refused, the
agent's chains for it finding no node. Both ends now apply ``reserved_stem_name`` /
``dawsession::isReservedTrackName``, pinned against ONE table, ``data/stem_names.json``: this file
drives the brain's two doors (``Session.rename_stem``, ``import_stem``) over it, and the console
leg ``MixAgentDAW --stem-names`` drives the DAW's predicate and its real ``track.rename`` over the
same file.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from mix_engine.session import Session, reserved_stem_name

TABLE = json.loads((Path(__file__).parent / "data" / "stem_names.json").read_text())["names"]
REPO = Path(__file__).resolve().parents[2]


def test_the_table_holds_both_kinds():
    kinds = {row["reserved"] for row in TABLE}
    assert kinds == {True, False}          # a table of refusals alone could not fail an over-strict rule


@pytest.mark.parametrize("row", TABLE, ids=lambda r: repr(r["name"]))
def test_rename_stem_refuses_exactly_the_reserved_names(demo_project, row):
    session = Session(demo_project)
    name, reserved = row["name"], row["reserved"]
    assert reserved_stem_name(name) is reserved
    if reserved:
        with pytest.raises(ValueError, match="taken or reserved"):
            session.rename_stem("stem_a", name)
        assert "stem_a" in session.stems and name not in session.stems
    else:
        assert session.rename_stem("stem_a", name)["renamed"] is True
        assert name in session.stems and "stem_a" not in session.stems


@pytest.mark.parametrize("row", TABLE, ids=lambda r: repr(r["name"]))
def test_import_stem_refuses_exactly_the_reserved_names(demo_project, tmp_path, row):
    from orchestrator.daw_bridge import import_stem
    import shutil
    outside = tmp_path / "outside"
    outside.mkdir()
    src = outside / "take.wav"
    shutil.copy(demo_project / "stem_a.wav", src)
    before = sorted(p.name for p in demo_project.glob("*.wav"))
    out = import_stem(demo_project, str(src), row["name"])
    if row["reserved"]:
        assert out["ok"] is False and "reserved name" in out["error"]
        assert sorted(p.name for p in demo_project.glob("*.wav")) == before     # nothing written
    else:
        assert out["ok"] is True and out["stem_id"] == row["name"]


def test_the_daw_predicate_is_the_same_two_clauses():
    """The C++ twin cannot be run from here (the console leg does that); this pins that it is still
    the two clauses the table was written against, so a third clause added on one side shows up."""
    text = (REPO / "daw/Source/DawSession.h").read_text(encoding="utf-8")
    body = re.search(r"inline bool isReservedTrackName \(const juce::String& name\)\s*\{(.*?)\n\}", text, re.S)
    assert body, "isReservedTrackName not found"
    returns = [l.strip() for l in body.group(1).splitlines() if l.strip().startswith("return")]
    assert returns == ['return name.containsChar (\':\') || name.equalsIgnoreCase ("master");']
