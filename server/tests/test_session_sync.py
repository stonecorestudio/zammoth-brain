"""A long-lived engine follows session.json (2026-09-10 agent audit P0-1).

A chat turn opens ONE engine (one Session) for the whole turn, while brain-performed
routing, the mid-turn co-edit mirror, in-turn stem registration, undo and nested runs
each write session.json through a fresh Session. The engine never re-read, so its next
persist — any propose, even a rejected one — wrote its stale copy over all of them:
"add a return, then brighten the vocal" lost the return. The end-to-end legs drive the
REAL daw_bridge.chat_turn with the real Brain hooks and a scripted stand-in for the
model; against the pre-fix engine they fail on the lost write."""
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from mix_engine.session import Session
from orchestrator import daw_bridge
from orchestrator.daw_server import Brain

SET_BASS = "http://127.0.0.1:8765/api/set_stem?stem=bass&gain_db=-7.0"


def _project(tmp_path: Path) -> Path:
    proj = tmp_path / "proj"
    proj.mkdir()
    sr = 44100
    t = np.arange(2 * sr) / sr
    for name, hz, amp in (("vocals", 220, 0.2), ("bass", 55, 0.3)):
        sf.write(proj / f"{name}.wav", (amp * np.sin(2 * np.pi * hz * t)).astype("float32"), sr)
    Session(proj)
    return proj


def _state(proj: Path) -> dict:
    return json.loads((proj / "session.json").read_text())


def _gain_req(stem: str, db: float) -> dict:
    return {"rationale": f"{stem} {db:+} dB",
            "predicted_impact": [{"metric": "stem.lufs_s", "predicted_delta": db,
                                  "unit": "LUFS"}],
            "actions": [{"type": "gain", "stem_id": stem, "delta_db": db}]}


class _Scripted:
    """Stands in for ChatSession: runs fixed tool calls through the real execute_tool."""

    def __init__(self, script):
        self.script, self.history, self.usage, self._model, self.log = list(script), [], {}, "", []

    def set_settings(self, model, reasoning):
        pass

    async def turn(self, text, *, tools, context, emit, should_cancel=None,
                   live_context=None, during_run=False):
        from orchestrator.chat import execute_tool
        self.history.append({"role": "user", "content": text})
        for name, args in self.script:
            out = await execute_tool(tools, name, args)
            self.log.append((name, out.text, out.error))
        await emit({"type": "chat_message", "text": "ok"})
        self.history.append({"role": "assistant", "content": "ok"})


def _turn(proj: Path, script, monkeypatch, coedits=None) -> _Scripted:
    monkeypatch.setenv("MIX_AGENT_DATA_DIR", str(proj.parent / "data"))
    brain = Brain(proj)

    def invoke(command, args=None, *a, **kw):   # the DAW end of daw_command
        if command == "routing.apply":
            return {"ok": True}
        out = {"ok": True}
        if coedits and command in coedits:
            out["coedits"] = list(coedits[command])
        return out
    brain.daw_invoke = invoke
    session = _Scripted(script)
    monkeypatch.setitem(daw_bridge._CHAT_SESSIONS, daw_bridge._session_key(proj, ""), session)
    daw_bridge.chat_turn(proj, "scripted", mode="agent", use_llm=False,
                         edit_state=lambda: brain.latest_edit,
                         daw_invoke=brain.counting_invoke(), cache=brain.render_cache,
                         routing_ops=brain.routing_ops(),
                         mirror_coedits=brain._mirror_coedits,
                         register_stem=brain._register_stem)
    return session


def test_a_return_added_in_the_turn_survives_a_later_propose(tmp_path, monkeypatch):
    proj = _project(tmp_path)
    session = _turn(proj, [("daw_command", {"command": "return.add", "args": {"kind": "blank"}}),
                           ("propose", _gain_req("vocals", -1.0))], monkeypatch)
    state = _state(proj)
    assert state["returns"], f"the propose erased the return the turn added: {session.log}"
    assert [p for p in state["proposals"] if p["status"] == "applied"], session.log


def test_a_fader_move_mirrored_in_the_turn_survives_a_later_propose(tmp_path, monkeypatch):
    proj = _project(tmp_path)
    session = _turn(proj, [("daw_command", {"command": "track.set_gain",
                                            "args": {"track": "bass", "gain_db": -7}}),
                           ("propose", _gain_req("vocals", -1.0))],
                    monkeypatch, coedits={"track.set_gain": [SET_BASS]})
    assert _state(proj)["stems"]["bass"]["gain_db"] == -7.0, \
        f"the propose erased the mirrored fader move: {session.log}"


def test_sync_reloads_only_after_another_writer(tmp_path):
    proj = _project(tmp_path)
    engine = Session(proj)
    assert engine.sync_from_disk() is False          # nothing changed since it loaded
    daw_bridge.apply_coedits(proj, [SET_BASS])       # another Session writes
    assert engine.sync_from_disk() is True
    assert engine.stems["bass"]["gain_db"] == -7.0
    engine._persist()                                # its own write is not a change
    assert engine.sync_from_disk() is False


def test_a_probe_copy_never_reloads(tmp_path):
    proj = _project(tmp_path)
    probe = Session(proj).probe_copy()
    daw_bridge.apply_coedits(proj, [SET_BASS])
    assert probe.sync_from_disk() is False
    assert probe.stems["bass"].get("gain_db", 0.0) == 0.0
