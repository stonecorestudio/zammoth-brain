"""The pre-merge review of 2026-10-29 (betaV18 -> main): the Python halves.

* A width move that does not name stereoize leaves the user's stereoize as it is - a stem's strip
  and the master's Imager alike. WidthAction defaulted stereoize to False and the move wrote it,
  so Auto Mix's wide-pop (and any "widen it") switched a user's Stereoize off.
* The "chain not yet in the DAW" marker compares only what both sides can list: a strip's width is
  a `width` row in the DAW and a field in the session, and a `tune` is never a native entry. It
  fired for ever on such a channel, and told the model the mismatch was not a fault.
* chains_owed is kept on what it reads: the live context asks for it every round, on the event loop.
* A move still owed to a chorus saved before its law, setting its MIX, carries the VOLUME that
  keeps the level at that mix.
The native halves are `--premerge-1029` (daw/Source/PremergeReviewLegs.h).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import anyio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mix_engine.models import ProposalRequest  # noqa: E402
from mix_engine.session import Session  # noqa: E402
from orchestrator import daw_bridge as B  # noqa: E402
from orchestrator.engine_client import in_memory_engine  # noqa: E402
from orchestrator.turn_host import LiveTurn  # noqa: E402


def _apply(s: Session, actions: list[dict]) -> None:
    res = s.propose(ProposalRequest(rationale="t", predicted_impact=[], actions=actions))
    assert res.accepted, res
    s.apply(res.proposal_id)


def test_a_width_move_that_does_not_name_stereoize_keeps_the_users(demo_project):
    s = Session(demo_project)
    stem = next(iter(s._state["stems"]))
    s.user_edit(stem, {"width": 1.2, "stereoize": True, "stereoize_ms": 12})
    _apply(s, [{"type": "width", "stem_id": stem, "width": 1.4}])
    st = s._state["stems"][stem]
    assert st["width"] == 1.4
    assert st["stereoize"] is True and st["stereoize_ms"] == 12.0
    _apply(s, [{"type": "width", "stem_id": stem, "width": 1.3, "stereoize": False}])
    assert s._state["stems"][stem]["stereoize"] is False          # named: it is the move's


def test_the_masters_imager_keeps_its_stereoize_through_a_width_move(demo_project):
    s = Session(demo_project)
    _apply(s, [{"type": "width", "stem_id": "master", "width": 1.2, "stereoize": True,
                "stereoize_ms": 9}])
    _apply(s, [{"type": "width", "stem_id": "master", "width": 1.1}])
    widths = [e for e in s._state["master_chain"] if e["type"] == "width"]
    assert len(widths) == 1 and widths[0]["width"] == 1.1
    assert widths[0]["stereoize"] is True and widths[0]["stereoize_ms"] == 9.0


def _turn(project: Path, edit: dict) -> LiveTurn:
    return LiveTurn(project, "mix it", use_llm=False, edit_state=lambda: edit)


_EQ = {"rationale": "lift presence", "predicted_impact": [],
       "actions": [{"type": "eq", "stem_id": "stem_a",
                    "band": {"freq_hz": 3000, "gain_db": 2.0, "q": 1.0, "shape": "peak"}}]}


def test_a_strip_width_row_is_not_read_as_a_chain_still_owed(demo_project):
    # the DAW lists the strip's width as a device row; the session holds it as a field
    edit = {"seq": 1, "tracks": [{"name": "stem_a", "devices": [{"type": "eq", "occ": 0},
                                                               {"type": "width", "occ": 0}]}]}
    turn = _turn(demo_project, edit)

    async def go():
        async with in_memory_engine(demo_project) as engine:
            turn.engine = engine
            assert (await turn.do_propose(_EQ))["accepted"] is True
            return await turn.do_read_track("stem_a")

    assert "chain_after_this_turn" not in anyio.run(go)["track"]


def test_a_type_the_native_chain_never_holds_is_not_owed(demo_project):
    state = {"stems": {"stem_a": {"chain": [{"type": "eq"}, {"type": "tune"}]}}}
    edit = {"tracks": [{"name": "stem_a", "devices": [{"type": "eq"}]}]}
    real = Session.__init__
    try:
        def fake(self, *a, **k):
            self._state = state
        Session.__init__ = fake
        assert B.chains_not_yet_in_daw(Path("."), edit, {"stem_a"}) == {}
    finally:
        Session.__init__ = real


def test_chains_owed_is_kept_on_what_it_reads(demo_project, monkeypatch):
    edit = {"seq": 4, "tracks": [{"name": "stem_a", "devices": []}]}
    turn = _turn(demo_project, edit)
    calls = []
    real = B.chains_not_yet_in_daw
    monkeypatch.setattr(B, "chains_not_yet_in_daw",
                        lambda *a: (calls.append(1), real(*a))[1])

    async def go():
        async with in_memory_engine(demo_project) as engine:
            turn.engine = engine
            await turn.do_propose(_EQ)          # warms it on a worker thread
            n = len(calls)
            for _ in range(5):                  # the live context, every round
                turn.chains_owed()
            return n

    warmed = anyio.run(go)
    assert warmed == 1 and len(calls) == 1      # no read on the event loop after the warm


def test_an_owed_chorus_mix_move_carries_the_level_it_kept(demo_project):
    s = Session(demo_project)
    stem = next(iter(s._state["stems"]))
    s._state["stems"][stem]["chain"] = [{"type": "chorus", "uid": 7, "mode": 0, "stereo": 0.5,
                                         "mix": 0.2, "output_db": -1.0}]
    url = (f"/api/device_param?stem={stem}&device_type=chorus&occurrence=0&param=mix&value=0.6")
    extra = s._chorus_level_kept(url, was=8)
    assert len(extra) == 1
    q = dict(parse_qsl(urlsplit(extra[0]).query))
    assert q["param"] == "output_db"
    from mix_engine.chorus_law import level_db
    import math
    want = round(10 * math.log10(0.4 ** 2 + 0.6 ** 2) - level_db(0.6, 0), 1)
    assert float(q["value"]) == want
    assert s._chorus_level_kept(url, was=9) == []                 # a move made under the law
    assert s._chorus_level_kept(url.replace("param=mix", "param=stereo"), was=8) == []
    json.dumps(extra)
