"""The real-use QA of 2026-10-30 (the before-and-after mixes, round 4).

A send the agent set rendered the whole song. `send.set` is performed by the brain
(`routing_ops`), and `Brain.set_send` answered with `chains_with_readout`: a full master render
and its readout, about 6.6 s on a two-minute ballad - six sends were 40 s of a 5.4-minute turn.
The turn already treats the brain's reading as stale after a routing change
(`LiveTurn._mix_moved_unmeasured`) and pushes measured chains at its end, so every one of those
renders was thrown away.

Now a routing op inside an agent turn resolves its chains over the last measurement
(`daw_bridge.IN_AGENT_TURN`) and does NOT stamp that measurement as this state's, so the next
reader measures afresh. Outside a turn (the user's door) a send still renders.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from mix_engine import dsp
from mix_engine.session import Session
from orchestrator import daw_bridge as B
from orchestrator import daw_server as S

from test_phase54_render_once import _turn, kept  # noqa: F401 - the fixture is used by name


def _project(tmp_path: Path) -> Path:
    proj = tmp_path / "proj"
    proj.mkdir()
    sr = 44100
    t = np.arange(2 * sr) / sr
    for name, hz, amp in (("vocals", 220, 0.2), ("bass", 55, 0.3), ("keys", 440, 0.1)):
        sf.write(proj / f"{name}.wav", (amp * np.sin(2 * np.pi * hz * t)).astype("float32"), sr)
    Session(proj)
    return proj


def _masters() -> int:
    return dsp.RENDER_COUNTS["master"]


def test_six_sends_in_a_turn_render_the_song_at_most_once(tmp_path, monkeypatch, kept):
    proj = _project(tmp_path)
    brain = S.Brain(proj)
    rid = brain.add_return("reverb")["return_id"]
    B.session_readout(proj, brain.render_cache)          # the session as the brain holds it
    script = [("daw_command", {"command": "send.set",
                               "args": {"source": src, "return": rid, "gain_db": g}})
              for src in ("vocals", "bass", "keys") for g in (-18.0, -20.0)]
    session, out, masters, _ = _turn(proj, script, {}, monkeypatch, brain=brain)
    assert all(not error for _, _, error in session.log), session.log
    sends = Session(proj, read_only=True)._state["stems"]["keys"].get("sends") or {}
    assert sends, "the sends WERE set"
    assert masters <= 1, f"{masters} master renders for six sends"


def test_a_send_in_a_turn_does_not_claim_the_old_reading_is_current(tmp_path, kept):
    proj = _project(tmp_path)
    brain = S.Brain(proj)
    rid = brain.add_return("reverb")["return_id"]
    B.session_readout(proj, brain.render_cache)
    token = B.IN_AGENT_TURN.set(True)
    try:
        brain.set_send("vocals", rid, -12.0)
    finally:
        B.IN_AGENT_TURN.reset(token)
    assert brain.render_cache.get("fp") != Session(proj, read_only=True).render_fingerprint(), \
        "the reading was made before the send: it is not this state's"


def test_a_send_outside_a_turn_still_renders(tmp_path, kept):
    proj = _project(tmp_path)
    brain = S.Brain(proj)
    rid = brain.add_return("reverb")["return_id"]
    B.session_readout(proj, brain.render_cache)
    before = _masters()
    brain.set_send("vocals", rid, -12.0)
    assert _masters() - before == 1
