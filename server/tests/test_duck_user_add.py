"""User-addable sidechain Duck (2026-07-11): the browser Devices tab / dock add path.
A duck lands UNKEYED (key_stem_id "") and must be a render no-op until keyed; one duck
per stem (re-key via set_duck); stem-only (a bus/return add is refused). The keyed
render/duck math itself is covered by the Phase 7 tests + the DAW --sidechain smoke."""
import numpy as np
import pytest

from mix_engine.dsp import render_stem
from mix_engine.session import Session


def test_duck_user_add_unkeyed_is_render_noop(demo_project):
    s = Session(demo_project)
    stem = next(iter(s._state["stems"]))

    before, _ = render_stem(s, stem)
    res = s.add_stem_device(stem, "duck")
    assert res == {"type": "duck", "occurrence": 0, "target": stem}
    entry = [e for e in s._state["stems"][stem]["chain"] if e["type"] == "duck"][0]
    assert entry["key_stem_id"] == ""          # lands unkeyed
    assert entry["threshold_db"] == -30.0 and entry["depth_db"] == 6.0

    after, _ = render_stem(Session(demo_project), stem)   # fresh session: no cache aliasing
    assert np.array_equal(before, after)       # unkeyed duck = bit-identical no-op


def test_duck_one_per_stem_and_keying(demo_project):
    s = Session(demo_project)
    stems = list(s._state["stems"])
    stem, key = stems[0], stems[1]

    s.add_stem_device(stem, "duck")
    with pytest.raises(ValueError):            # one duck per stem — re-key, don't stack
        s.add_stem_device(stem, "duck")

    out = s.set_duck(stem, key)                # keying replaces the unkeyed entry
    assert out["key_stem_id"] == key
    ducks = [e for e in s._state["stems"][stem]["chain"] if e["type"] == "duck"]
    assert len(ducks) == 1 and ducks[0]["key_stem_id"] == key


def test_duck_refused_on_bus_and_return(demo_project):
    s = Session(demo_project)
    stems = list(s._state["stems"])
    bus = s.create_bus("busduck", stems[:1])["bus_id"] if hasattr(s, "create_bus") else None
    if bus is not None:
        with pytest.raises(ValueError):
            s.add_stem_device(bus, "duck")


def test_duck_rekey_keeps_tuning_and_clear_keeps_device(demo_project):
    """2026-09-04 duck audit: re-keying used to drop the entry and append a fresh −30/6/5/150
    one, so every trigger pick (combo, agent duck.set_key, a moved duck) reset the tuning."""
    s = Session(demo_project)
    stems = list(s._state["stems"])
    stem, key_a, key_b = stems[0], stems[1], stems[-1]
    s.add_stem_device(stem, "duck")
    s.set_duck(stem, key_a)
    # the user (or SC AUTO) tunes it…
    ducks = lambda: [e for e in s._state["stems"][stem]["chain"] if e["type"] == "duck"]
    d = ducks()[0]
    d["threshold_db"], d["depth_db"], d["attack_ms"], d["release_ms"], d["enabled"] = -42.0, 9.5, 12.0, 320.0, False
    s._persist()
    # …then re-keys: the tuning + bypass carry over, only the trigger changes
    s.set_duck(stem, key_b if key_b != stem else key_a)
    d = ducks()[0]
    assert d["threshold_db"] == -42.0 and d["depth_db"] == 9.5
    assert d["attack_ms"] == 12.0 and d["release_ms"] == 320.0 and d["enabled"] is False
    # explicit arguments still win (the agent's duck.set_key carries threshold/depth)
    s.set_duck(stem, key_a, threshold_db=-20.0)
    assert ducks()[0]["threshold_db"] == -20.0 and ducks()[0]["depth_db"] == 9.5
    # "none" disarms but keeps the device, its tuning intact
    s.set_duck(stem, "none")
    d = ducks()
    assert len(d) == 1 and d[0]["key_stem_id"] == "" and d[0]["depth_db"] == 9.5
    # a stem that never had a duck gets none from a clear
    other = key_a
    s.set_duck(other, "none")
    assert not [e for e in s._state["stems"][other]["chain"] if e["type"] == "duck"]


def test_duck_key_coedit_url(demo_project):
    """The DAW persists the user's trigger pick as a co-edit URL (native-first keying)."""
    from orchestrator.daw_bridge import apply_coedits
    s = Session(demo_project)
    stems = list(s._state["stems"])
    stem, key = stems[0], stems[1]
    s.add_stem_device(stem, "duck")
    res = apply_coedits(demo_project, [f"/api/set_duck?stem={stem}&key={key}"])
    assert res["failed"] == 0
    d = [e for e in Session(demo_project)._state["stems"][stem]["chain"] if e["type"] == "duck"]
    assert len(d) == 1 and d[0]["key_stem_id"] == key
    res = apply_coedits(demo_project, [f"/api/set_duck?stem={stem}&key=none"])
    assert res["failed"] == 0
    d = [e for e in Session(demo_project)._state["stems"][stem]["chain"] if e["type"] == "duck"]
    assert len(d) == 1 and d[0]["key_stem_id"] == ""
    # a self-key is refused, and reported (never a silent no-op)
    res = apply_coedits(demo_project, [f"/api/set_duck?stem={stem}&key={stem}"])
    assert res["failed"] == 1


def _third_stem(project) -> str:
    """The demo has TWO stems, so stems[1] and stems[-1] are the same track — a "re-key to a
    DIFFERENT trigger" written against them only ever re-keys to the SAME one. Add a third."""
    import shutil
    src = sorted(project.glob("*.wav"))[0]
    shutil.copy(src, project / "stem_c.wav")
    return "stem_c"


def test_rekey_to_a_genuinely_different_trigger_keeps_tuning(demo_project):
    """The three-stem version of test_duck_rekey_keeps_tuning_and_clear_keeps_device, whose
    key_b was the same track as key_a on the two-stem demo (2026-09-05 review)."""
    third = _third_stem(demo_project)
    s = Session(demo_project)
    stems = list(s._state["stems"])
    assert third in stems and len(stems) >= 3
    stem, key_a = stems[0], stems[1]
    assert key_a != third                                  # the point of the fixture
    s.add_stem_device(stem, "duck")
    s.set_duck(stem, key_a)
    ducks = lambda: [e for e in s._state["stems"][stem]["chain"] if e["type"] == "duck"]
    d = ducks()[0]
    d["threshold_db"], d["depth_db"], d["attack_ms"], d["release_ms"] = -42.0, 9.5, 12.0, 320.0
    s._persist()
    s.set_duck(stem, third)                                # a DIFFERENT trigger
    d = ducks()[0]
    assert d["key_stem_id"] == third
    assert (d["threshold_db"], d["depth_db"], d["attack_ms"], d["release_ms"]) == (-42.0, 9.5, 12.0, 320.0)


def test_set_duck_keeps_the_duck_where_it_is(demo_project):
    """2026-09-05: set_duck filtered the duck out and APPENDED it, so a single trigger pick
    moved it to the end of the BRAIN's chain — while MonitorChainPlugin::setDuckKeyStemId edits
    the entry in place and apply_coedits returns no chains, so the DAW never adopted the move.
    The two truths diverged: the Python render (and the export oracle) ducked after the EQ, and
    the next applyChainsAndReadout pushed the brain's order back, making the user's Duck jump to
    the last dock slot."""
    s = Session(demo_project)
    stems = list(s._state["stems"])
    stem, key = stems[0], stems[1]
    s.add_stem_device(stem, "duck")
    s.add_stem_device(stem, "eq")
    types = lambda st: [e["type"] for e in s._state["stems"][st]["chain"]]
    assert types(stem) == ["duck", "eq"]                   # the shape the bug needs to be visible
    s.set_duck(stem, key)
    assert types(stem) == ["duck", "eq"], "a trigger pick reordered the chain"
    s.set_duck(stem, "none")
    assert types(stem) == ["duck", "eq"], "disarming reordered the chain"
    s.set_duck(stem, key)                                  # and re-arming does not either
    assert types(stem) == ["duck", "eq"]
    # a stem with no duck yet still gets one appended at the end (the first stem's trigger
    # cleared: two stems may not trigger each other - 2026-10-17, the duck audit)
    s.set_duck(stem, "none")
    s.add_stem_device(key, "eq")
    s.set_duck(key, stem)
    assert types(key) == ["eq", "duck"]


def test_disarmed_duck_survives_a_copy(demo_project):
    """2026-09-04 made "Trigger: none" DISARM the duck while KEEPING the device (so the next
    pick doesn't start over) — but _validated_chain dropped any duck whose key_stem_id was not
    a stem, and "" never is. Duplicating the track therefore silently lost the device, and
    blamed a key track that was not the problem. An unkeyed duck renders as a no-op in BOTH
    engines (dsp.py's `key_stem_id in session.stems` guard, MonitorChain's `duckKey.isEmpty()`),
    so it is a legal state, not a broken one."""
    s = Session(demo_project)
    stems = list(s._state["stems"])
    stem, key = stems[0], stems[1]
    s.add_stem_device(stem, "duck")
    s.set_duck(stem, key, threshold_db=-24.0, depth_db=8.0)
    s.set_duck(stem, "none")                               # disarmed, tuning kept
    res = s.copy_stem_state(stem, key)
    assert res.get("devices_dropped") == [], res
    dest = [e for e in s._state["stems"][key]["chain"] if e["type"] == "duck"]
    assert len(dest) == 1 and dest[0]["key_stem_id"] == ""
    assert dest[0]["threshold_db"] == -24.0 and dest[0]["depth_db"] == 8.0

    # …and a duck whose key really IS gone is still dropped, with the reason it deserves
    s2 = Session(demo_project)
    s2._state["stems"][stem]["chain"] = [
        {"type": "duck", "key_stem_id": "no_such_stem", "threshold_db": -30.0,
         "depth_db": 6.0, "attack_ms": 5.0, "release_ms": 150.0}]
    chain, dropped = s2._validated_chain(s2._state["stems"][stem]["chain"], key)
    assert chain == [] and dropped == ["duck (its key track isn't in this project)"]
