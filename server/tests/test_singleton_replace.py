"""2026-08-06 singleton-audit follow-up: the agent's PROPOSAL path must not
stack native-singleton devices.

`_NATIVE_SINGLETON_KINDS` (session.py) names the steps the native monitor
hosts exactly once — a second chain entry is monitored last-wins while the
offline export renders BOTH (the device audit measured 3.98 dB of divergence
for a doubled comp). The audit guarded the co-edit add path
(`add_stem_device` refuses a second); these tests pin `_mutate`, the path
every agent-approved proposal takes, which used to append deess / transient /
gate / saturation / clip / duck instead of replacing.
"""
from __future__ import annotations

import pytest

from mix_engine.models import (ClipAction, DeessAction, GateAction,
                               GroupAction, ProposalRequest, SaturationAction,
                               SidechainDuckAction, TransientAction)
from mix_engine.session import MASTER, Session


def _count(chain: list[dict], kind: str) -> int:
    return sum(1 for e in chain if e["type"] == kind)


def test_stem_singletons_replace_not_stack(demo_project):
    """Re-proposing a singleton adjusts it ("tighten the de-esser") — the
    second move's params win and the chain still holds exactly one entry."""
    s = Session(demo_project)
    stems = list(s._state["stems"])
    stem, key = stems[0], stems[-1]
    moves = [
        (DeessAction(stem_id=stem, threshold_db=-24.0, freq_hz=6000.0),
         DeessAction(stem_id=stem, threshold_db=-18.0, freq_hz=7000.0)),
        (TransientAction(stem_id=stem, attack_db=2.0),
         TransientAction(stem_id=stem, attack_db=3.0)),
        (GateAction(stem_id=stem, threshold_db=-40.0),
         GateAction(stem_id=stem, threshold_db=-35.0)),
        (SaturationAction(stem_id=stem, drive_db=3.0),
         SaturationAction(stem_id=stem, drive_db=4.0)),
        (SidechainDuckAction(stem_id=stem, key_stem_id=key,
                             threshold_db=-30.0, depth_db=6.0),
         SidechainDuckAction(stem_id=stem, key_stem_id=key,
                             threshold_db=-24.0, depth_db=4.0)),
    ]
    for first, second in moves:
        s._mutate(first)
        s._mutate(second)
        chain = s._state["stems"][stem]["chain"]
        assert _count(chain, first.type) == 1, f"{first.type} stacked"
        entry = next(e for e in chain if e["type"] == first.type)
        probe = {"deess": "threshold_db", "gate": "threshold_db",
                 "duck": "threshold_db", "transient": "attack_db",
                 "saturation": "drive_db"}[first.type]
        assert entry[probe] == getattr(second, probe)


def test_ranged_deess_still_coexists(demo_project):
    """Phase 15/35 ranged de-essers are NOT singletons: a whole-file replace
    must leave a time-ranged entry alone, and vice versa."""
    s = Session(demo_project)
    stem = next(iter(s._state["stems"]))
    s._mutate(DeessAction(stem_id=stem, threshold_db=-24.0, freq_hz=6000.0))
    s._mutate(DeessAction(stem_id=stem, threshold_db=-20.0, freq_hz=6000.0,
                          time_range_s=(10.0, 20.0)))
    assert _count(s._state["stems"][stem]["chain"], "deess") == 2
    s._mutate(DeessAction(stem_id=stem, threshold_db=-18.0, freq_hz=7000.0))
    chain = s._state["stems"][stem]["chain"]
    assert _count(chain, "deess") == 2   # replaced the whole-file one only
    whole = [e for e in chain
             if e["type"] == "deess" and e.get("time_range_s") is None]
    assert len(whole) == 1 and whole[0]["threshold_db"] == -18.0


def test_user_placed_singleton_is_replaced_but_recorded(demo_project):
    """A user-placed singleton is still REPLACED — and the loss is recorded.

    The Phase 22 "not the agent's to replace" carve-out lived in these filters
    and could never be switched on: every type it guarded is a
    `_NATIVE_SINGLETON_KINDS` step the monitor hosts once, so sparing the
    user's entry leaves two of a kind — monitored last-wins, exported both,
    the divergence this whole file exists to prevent. Nothing in production
    ever wrote `owner` (only a test did), so it never fired. Stamping the tag
    for real would have traded silent data loss for silently wrong audio
    (2026-08-16 review), so the tag is now provenance only: the agent replaces,
    and _replace_in_chain records whose device it was.
    """
    s = Session(demo_project)
    stem = next(iter(s._state["stems"]))
    s.add_stem_device(stem, "gate")          # the user's hands, via the co-edit channel
    chain = s._state["stems"][stem]["chain"]
    assert next(e for e in chain if e["type"] == "gate").get("owner") == "user", (
        "add_stem_device must tag what the user placed")
    s._mutate(GateAction(stem_id=stem, threshold_db=-35.0))   # the agent's
    assert _count(s._state["stems"][stem]["chain"], "gate") == 1, (
        "two gates in one chain is the monitor/export divergence, not a fix")
    assert s._replaced_user_devices == ["gate"], (
        "overwriting the user's device must be recorded, not silent")


def test_add_master_device_tags_what_the_user_placed(demo_project):
    """The master co-edit channel tags its entries like the stem one does.

    Whether _mutate should then SPARE a user-owned master device is an open
    design call (2026-08-16): the master singleton-replace has no carve-out,
    and adding one would let a user device and an agent device of the same
    type coexist on the master — which the native monitor hosts last-wins
    while the offline export renders both, the exact divergence the
    2026-08-06 singleton audit closed. Tagging is correct either way; the
    replace policy is James's call.
    """
    s = Session(demo_project)
    s.add_master_device("saturation")
    assert next(e for e in s._state["master_chain"]
                if e["type"] == "saturation").get("owner") == "user"


def test_replacing_a_user_master_device_is_reported(demo_project):
    """The agent still replaces on the master — but says so.

    Sparing it would put two saturations in a chain the native monitor hosts
    once (last-wins live, both in the export), and refusing the action would
    deadlock the mix loop the first time a user hand-places a master limiter.
    So the move stands and the message names what went; undo restores it.
    """
    s = Session(demo_project)
    s.add_master_device("transient")   # saturation is not a MASTER_ACTION_TYPE
    proposed = s.propose(ProposalRequest(
        rationale="snap the master", predicted_impact=[],
        actions=[{"type": "transient", "stem_id": MASTER, "attack_db": 2.0}]))
    assert proposed.accepted, proposed.violations
    result = s.apply(proposed.proposal_id)
    assert result.ok
    assert "transient" in result.message, (
        "overwriting a user-placed master device must not be silent")
    assert _count(s._state["master_chain"], "transient") == 1   # still one per chain


def test_master_singletons_replace_not_stack(demo_project):
    s = Session(demo_project)
    for first, second in [
        (ClipAction(stem_id=MASTER, threshold_db=-3.0),
         ClipAction(stem_id=MASTER, threshold_db=-2.0)),
        (SaturationAction(stem_id=MASTER, drive_db=3.0),
         SaturationAction(stem_id=MASTER, drive_db=5.0)),
        (DeessAction(stem_id=MASTER, threshold_db=-24.0, freq_hz=6000.0),
         DeessAction(stem_id=MASTER, threshold_db=-20.0, freq_hz=6500.0)),
        (TransientAction(stem_id=MASTER, attack_db=1.0),
         TransientAction(stem_id=MASTER, attack_db=2.0)),
    ]:
        s._mutate(first)
        s._mutate(second)
        assert _count(s._state["master_chain"], first.type) == 1, (
            f"{first.type} stacked on master")


def test_bus_singletons_replace_not_stack(demo_project):
    s = Session(demo_project)
    stems = list(s._state["stems"])
    s._mutate(GroupAction(bus_id="glue", members=stems))
    for first, second in [
        (GateAction(stem_id="glue", threshold_db=-40.0),
         GateAction(stem_id="glue", threshold_db=-35.0)),
        (SaturationAction(stem_id="glue", drive_db=3.0),
         SaturationAction(stem_id="glue", drive_db=4.0)),
        (TransientAction(stem_id="glue", attack_db=2.0),
         TransientAction(stem_id="glue", attack_db=3.0)),
        (DeessAction(stem_id="glue", threshold_db=-24.0, freq_hz=6000.0),
         DeessAction(stem_id="glue", threshold_db=-20.0, freq_hz=6500.0)),
    ]:
        s._mutate(first)
        s._mutate(second)
        chain = s._state["buses"]["glue"]["chain"]
        assert _count(chain, first.type) == 1, f"{first.type} stacked on bus"


def test_safety_limiter_is_ceiling_only(demo_project):
    """The DAW's default master limiter (2026-09-04) is a SAFETY device: a −1 dBTP
    ceiling and no loudness target — transparent until a peak reaches it. The
    palette limiter (safety=False) keeps its −14 LUFS target; the safety one must
    not inherit it, or "a limiter on the master by default" would push every new
    project's loudness. Still one limiter max, and it removes like any other."""
    s = Session(demo_project)
    res = s.add_master_device("limiter", safety=True)
    assert res == {"type": "limiter", "occurrence": 0}
    entry = next(e for e in s._state["master_chain"] if e["type"] == "limiter")
    assert entry["ceiling_dbtp"] == -1.0
    assert "target_lufs_i" not in entry
    assert entry.get("owner") == "user"
    with pytest.raises(ValueError, match="already has a limiter"):
        s.add_master_device("limiter")            # one limiter max, palette or safety
    s.remove_master_device("limiter", 0)
    assert not any(e["type"] == "limiter" for e in s._state["master_chain"])
    # the palette limiter is unchanged by the flag's existence
    s.add_master_device("limiter")
    assert next(e for e in s._state["master_chain"]
                if e["type"] == "limiter")["target_lufs_i"] == -14.0
