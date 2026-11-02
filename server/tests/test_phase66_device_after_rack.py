"""Phase 66 (2026-10-24, James: "i cant put devices after the rack"): a device after a rack.

The DAW's chain is one list; an entry that plays after a rack or a hosted plug-in names it
(`after`: the plug-in's item id) and a tail plug-in right after that rack runs it. The brain keeps
the placement - it never renders it (a hosted plug-in is a named blind spot) - so it must survive
every door a chain entry passes: the order co-edit that sets it, the push back to the app, the
journal that replays it after an undo, a replace. The C++ half is `--device-after-rack`.
"""
from __future__ import annotations

from pathlib import Path

from mix_engine.monitor_chain import monitor_chain_params
from mix_engine.session import Session
from orchestrator import daw_bridge

ROOT = Path(__file__).resolve().parents[2]


def _stem(s: Session) -> str:
    return next(iter(s._state["stems"]))


def _chain(s: Session, stem: str) -> list[dict]:
    return s._state["stems"][stem]["chain"]


def test_the_push_carries_each_entrys_placement():
    chain = [{"type": "comp", "threshold_db": -20.0, "ratio": 3.0, "attack_ms": 10.0,
              "release_ms": 100.0, "after": "1045"},
             {"type": "utility", "gain_db": 0.0}]
    out = monitor_chain_params(chain)["entries"]
    by = {e["type"]: e for e in out}
    assert by["comp"]["after"] == "1045"
    assert "after" not in by["utility"]


def test_the_order_coedit_stamps_the_entries_it_names_before_they_move(demo_project):
    s = Session(demo_project)
    stem = _stem(s)
    s.add_stem_device(stem, "comp")
    s.add_stem_device(stem, "gate")
    first_comp = _chain(s, stem)[[e["type"] for e in _chain(s, stem)].index("comp")]
    # a drag: the gate goes in front, the comp after the rack - tokens in the brain's numbering
    daw_bridge._apply_one_coedit(s, f"/api/device_order?order=gate%230,comp%230&after=-,1045&stem={stem}")
    types = [e["type"] for e in _chain(s, stem) if e["type"] in ("comp", "gate")]
    assert types == ["gate", "comp"]
    assert first_comp["after"] == "1045"
    gate = next(e for e in _chain(s, stem) if e["type"] == "gate")
    assert "after" not in gate
    # ...and back in front: a dash clears it, and an order of ONE device still clears
    daw_bridge._apply_one_coedit(s, f"/api/device_order?order=gate%230,comp%230&after=-,-&stem={stem}")
    assert "after" not in first_comp


def test_an_order_without_a_placement_places_nothing(demo_project):
    s = Session(demo_project)
    stem = _stem(s)
    s.add_stem_device(stem, "comp")
    daw_bridge._apply_one_coedit(s, f"/api/device_order?order=comp%230&after=1045&stem={stem}")
    daw_bridge._apply_one_coedit(s, f"/api/device_order?order=comp%230&stem={stem}")   # an older order
    comp = next(e for e in _chain(s, stem) if e["type"] == "comp")
    assert comp["after"] == "1045"


def test_the_journal_replays_the_placement_with_the_order(demo_project):
    s = Session(demo_project)
    stem = _stem(s)
    s.add_stem_device(stem, "comp")
    url = f"/api/device_order?order=comp%230&after=1045&stem={stem}"
    who = daw_bridge._coedit_who(s, url)
    assert who is not None and who.get("afters") == ["1045"]
    comp = next(e for e in _chain(s, stem) if e["type"] == "comp")
    comp.pop("after", None)                        # what an undo's snapshot would put back
    daw_bridge._replay_one_coedit(s, {"url": url, "who": who}, stem)
    assert comp["after"] == "1045"


def test_a_replaced_device_keeps_where_it_played(demo_project):
    s = Session(demo_project)
    stem = _stem(s)
    s.add_stem_device(stem, "comp")
    comp = next(e for e in _chain(s, stem) if e["type"] == "comp")
    comp["after"] = "1045"
    s._replaced_user_devices = []   # what _mutate_one sets up before any replace
    kept = s._replace_in_chain(_chain(s, stem), lambda e: e is comp)
    assert s._carry_uid[2] == "1045"
    fresh = {"type": "comp", "threshold_db": -10.0, "ratio": 2.0, "attack_ms": 5.0, "release_ms": 80.0}
    s._state["stems"][stem]["chain"] = kept + [fresh]
    s._stamp_entries(s._carry_uid)
    assert fresh["after"] == "1045" and fresh["uid"] == comp["uid"]


def test_every_app_door_that_tells_the_brain_an_order_carries_the_placement():
    dock = (ROOT / "daw" / "Source" / "DeviceChainView.h").read_text()
    app = (ROOT / "daw" / "Source" / "App.cpp").read_text()
    specs = (ROOT / "native" / "Source" / "DeviceSpecs.h").read_text()
    # every device_order the dock posts carries `&after=` (or is built by orderQuery, which does)
    at = dock.find("/api/device_order?")
    while at >= 0:
        stmt = dock[at:dock.find(";", at)]
        assert "after=" in stmt or "orderQuery" in stmt, stmt[:160]
        at = dock.find("/api/device_order?", at + 1)
    assert "&after=" in app.split("d.order.joinIntoString")[1][:200]   # the undo mirror
    assert "d.after = places (aEs);" in specs
    # the dock's drop and adds make the tails agree in the same gesture
    assert dock.count("host.syncChainTails();") >= 3
