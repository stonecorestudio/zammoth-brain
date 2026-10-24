"""The eval fixtures must be shaped like the DAW's real EditSnapshot (2026-09-02).

The eval audit found the three LLM evals feeding a hand-written session under
web-era names the DAW never publishes — `tempo_bpm`, `time_signature`,
`playhead_s`, a bare `key`, the analysis and the command catalog INSIDE the
edit — so a reach the eval trained on those could not find them in production,
where the live edit says `tempo.bpm` / `transport.position_s` and the analysis
arrives as its own labelled part. Nothing held the fixture to the product.

These tests do, from two directions:

* every key in every fixture (the base edit, the operate edit, each case's
  private patch) must be a property name `EditSnapshot.h` actually sets — the
  SOURCE of the shape, so a rename there fails here the same day — or sit on a
  short allowlist with a reason;
* the context part LABELS the evals use must be the literal strings
  `daw_bridge.py` builds the production context from;
* a real dump captured from the console DAW (`orchestrator/fixtures/`) must
  agree with the fixture on the TYPE of every field both carry.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchestrator import daw_bridge, operate_eval as O, qa_eval as Q, understanding as U

REPO = Path(__file__).resolve().parents[2]
SNAPSHOT_H = REPO / "daw" / "Source" / "EditSnapshot.h"
#: Everything the DAW and the shared native code set on a var: EditSnapshot.h
#: is the trunk, but device meters, modifiers, the sampler's layers and the
#: open-projects list are published by App.cpp / native/ helpers it calls.
SOURCE_DIRS = (REPO / "daw" / "Source", REPO / "native" / "Source")
REAL_DUMP = REPO / "orchestrator" / "fixtures" / "edit_snapshot_suno_demo.json"

#: Fixture keys the DAW never publishes — each with the reason it is still here.
#: Keep this SHORT; an entry without a reason rots into a hole.
EVAL_ONLY: dict[str, str] = {}
#: Fields the fixture carries that the DAW's snapshot does NOT publish but the
#: product does hand the agent by another route — each with that route.
APP_ADDED: dict[str, str] = {}
#: Subtrees that are not snapshot properties at all and are skipped: the catalog
#: is catalogToVar's shape (key / args / user_only / needs_confirm).
SKIP_SUBTREES = ("commands", "daw_commands")
#: Maps whose KEYS are ids from the session (a send map is track -> return ->
#: dB), so the keys are data, not property names.
ID_KEYED = ("sends", "synth")
#: `synth` (reader P1 D6.R9): a WOOLLY track's switches, keyed by FIELD name
#: ("osc2.wave", "fx.chorus"). The names are a table in DawSession.h, not
#: setProperty literals, so they are held to it by their own test below.


_PROPERTY = re.compile(r'setProperty \((?:juce::Identifier \()?"([A-Za-z_0-9]+)"')
_IDENTIFIER = re.compile(r'Identifier \("([A-Za-z_0-9]+)"')


def _universe() -> set[str]:
    trunk = set(_PROPERTY.findall(SNAPSHOT_H.read_text()))
    assert len(trunk) > 150, "EditSnapshot.h parse looks broken"
    names = set(trunk)
    for d in SOURCE_DIRS:
        for f in list(d.glob("*.h")) + list(d.glob("*.cpp")):
            text = f.read_text(errors="ignore")
            names |= set(_PROPERTY.findall(text)) | set(_IDENTIFIER.findall(text))
    return names


def _keys(obj, out: set[str], skip: tuple[str, ...] = SKIP_SUBTREES,
          under_ids: bool = False) -> set[str]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in skip:
                continue
            if not under_ids:
                out.add(k)
            # one level of id-keyed map, then its values' maps are ids too
            _keys(v, out, skip, under_ids=(k in ID_KEYED) or under_ids and isinstance(v, dict))
    elif isinstance(obj, list):
        for v in obj:
            _keys(v, out, skip, under_ids)
    return out


def _fixtures() -> list[tuple[str, dict]]:
    fixtures = [("understanding._LLM_EDIT", U._LLM_EDIT),
                ("operate_eval.OPERATE_EDIT", O.OPERATE_EDIT)]
    for case in O.CORPUS:
        if case.context:
            fixtures.append((f"operate case {case.text!r} context", case.context()))
        if case.live:
            fixtures.append((f"operate case {case.text!r} live", case.live()))
    for case in Q.CORPUS:
        if case.context:
            fixtures.append((f"qa case {case.text!r} context", case.context()))
    return fixtures


@pytest.mark.parametrize("name,edit", _fixtures(), ids=lambda x: x if isinstance(x, str) else "")
def test_every_fixture_key_is_a_real_snapshot_property(name: str, edit: dict):
    unknown = _keys(edit, set()) - _universe() - set(EVAL_ONLY) - set(APP_ADDED)
    assert not unknown, (f"{name} carries keys the DAW never publishes: "
                         f"{sorted(unknown)} — use EditSnapshot.h's name, or "
                         f"allowlist it WITH a reason")


def test_the_allowlist_has_no_dead_entries():
    """An allowlisted key nothing uses is a hole waiting for a new fixture."""
    used: set[str] = set()
    for _name, edit in _fixtures():
        _keys(edit, used)
    dead = (set(EVAL_ONLY) | set(APP_ADDED)) - used
    assert not dead, f"allowlisted but unused: {sorted(dead)}"


def test_no_web_era_key_survives():
    """The exact names the audit found — pinned so they cannot creep back. `tags`
    and `send_returns` joined the list the same day production started publishing
    both as their own parts: they are never fields of the edit."""
    for _name, edit in _fixtures():
        keys = _keys(edit, set())
        for old in ("tempo_bpm", "time_signature", "playhead_s", "master_analysis",
                    "daw_commands", "tags", "send_returns"):
            assert old not in keys, f"{_name} still carries web-era key {old!r}"


def test_context_part_labels_are_productions():
    """The evals assemble the context from the same labelled parts daw_bridge
    does — held to the source, so a reworded label in production fails here."""
    # adjacent string literals are one label in Python; join them before matching
    # the catalog label moved into chat.render_catalog (2026-09-03), which both
    # production and the fixtures call — so the shared module counts as source too
    from orchestrator import chat as _chat
    text = re.sub(r'"\s*\n\s*"', "", Path(daw_bridge.__file__).read_text()
                  + Path(_chat.__file__).read_text())
    text = re.sub(r'"\s+"', "", text)
    labels = [U.taste_part([]).split(": [")[0],
              U.lessons_part([]).split(": [")[0],
              U.observations_part([]).split(": [")[0],
              U.stem_tags_part({}).split(": {")[0],
              U.send_returns_part({}).split(": {")[0],
              "Live DAW edit (daw_edit_state): ",
              "Master analysis (measured): ",
              "Workspace mode: "]
    for label in labels:
        assert label in text, f"eval context label not found in daw_bridge.py: {label!r}"
    # the catalog label is built from chat.CATALOG_LABEL (2026-09-22, split_context cuts on
    # it), so hold the rendered part — what production and the fixtures both send — instead
    assert _chat.render_catalog([]).startswith(
        "Commands available now (daw_commands): ")
    stable = U.fixture_context()
    live = U.fixture_live()()
    assert "Workspace mode: agent" in stable
    assert "Master analysis (measured): " in stable
    assert "Commands available now (daw_commands): " in stable
    assert "Stem tags (auto_tag_stems" in stable, "the tagged stems ride their own part"
    assert "daw_edit_state" not in stable, "the edit is the LIVE half, as in production"
    with_returns = U.fixture_context(send_returns=O.OPERATE_SEND_RETURNS)
    assert "Send/return topology (send_returns): " in with_returns and "vox_plate" in with_returns
    assert "send_returns" not in U.fixture_context(), "no returns, no part — like production"
    assert live.startswith("Live DAW edit (daw_edit_state): ")
    assert '"commands"' not in live, "compact_edit_state strips the catalog from the live half"


def test_readout_part_has_productions_shape():
    """`_READOUT` must carry exactly the keys `_readout_from_analysis` returns."""
    stub = SimpleNamespace(key=SimpleNamespace(tonic="A", mode="min", confidence=0.8),
                           lufs_i=-16.0, true_peak_dbtp=-2.0, tempo_bpm=120.0,
                           tempo_confidence=0.9, band_energy={},
                           spectral_tilt_db_per_oct=-4.5, silent=False)
    assert set(U._READOUT) == set(daw_bridge._readout_from_analysis(stub))


def test_the_real_dump_agrees_on_shared_field_types():
    """A snapshot the console DAW actually wrote (export-oracle leg,
    MIXAGENT_ORACLE_SNAPSHOT) vs the operate fixture: every top-level field
    both carry must have the same JSON type, and the per-track / per-clip keys
    the real one publishes must all be names the universe knows."""
    real = json.loads(REAL_DUMP.read_text())
    fixture = O.OPERATE_EDIT
    kind = lambda v: "list" if isinstance(v, list) else "dict" if isinstance(v, dict) else "scalar"  # noqa: E731
    for key in set(real) & set(fixture):
        if key in SKIP_SUBTREES:
            continue
        assert kind(real[key]) == kind(fixture[key]), \
            f"{key}: real {kind(real[key])}, fixture {kind(fixture[key])}"
    # the real dump's own keys are in the universe too (the parse is honest)
    unknown = _keys(real, set()) - _universe() - set(APP_ADDED)
    assert not unknown, f"the real dump carries names the parse missed: {sorted(unknown)}"
    # and the fields production always sends are in the fixture
    for key in ("tempo", "transport", "tracks", "markers"):
        assert key in fixture and key in real
    assert "position_s" in fixture["transport"] and "position_s" in real["transport"]
    assert "bpm" in fixture["tempo"] and "bpm" in real["tempo"]


def test_a_patched_fixture_never_leaks_into_the_base():
    """`_patched` is one case's private copy — the neighbours' verdicts must
    not move because one case renamed a clip or removed the selection."""
    before = json.dumps(O.OPERATE_EDIT, sort_keys=True)
    for case in O.CORPUS:
        if case.context:
            case.context()
        if case.live:
            case.live()
    assert json.dumps(O.OPERATE_EDIT, sort_keys=True) == before
    assert "selection" in O.OPERATE_EDIT
    assert "selection" not in O._patched(selection=O.ABSENT)


def test_the_synth_block_uses_the_names_the_command_takes():
    """Reader P1 D6.R9: `synth{}` is published and written under ONE table,
    dawsession::synthSwitchFields. A fixture that spells a field its own way
    teaches the model a name instrument.set_patch refuses."""
    text = (SNAPSHOT_H.parent / "DawSession.h").read_text()
    body = text.split("inline juce::StringArray synthSwitchFields()", 1)[1]
    body = body.split("return {", 1)[1].split("};", 1)[0]
    table = set(re.findall(r'"([a-z0-9.]+)"', body))
    assert len(table) == 26, "synthSwitchFields parse looks broken"
    seen = 0
    for name, edit in _fixtures():
        for track in edit.get("tracks", []) if isinstance(edit, dict) else []:
            synth = track.get("synth") if isinstance(track, dict) else None
            if synth is None:
                continue
            seen += 1
            assert set(synth) - {"preset"} == table, (
                f"{name}: synth{{}} on {track.get('name')!r} differs from the "
                f"DAW's table: {sorted(set(synth) ^ table ^ {'preset'})}")
    assert seen, "no fixture carries a WOOLLY synth block any more"
