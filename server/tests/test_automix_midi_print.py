"""Auto Mix over a project of MIDI parts (2026-10-13).

Auto Mix reads AUDIO: an instrument track is played by the DAW's graph alone. A project whose
parts were all MIDI got "there's no audio to auto-mix yet - drop a stem or a sample onto the
arrangement first": true, and no way forward. Now the button's decline names the parts and
offers to print them (`offer_print`, a verb in the feed), the offer taken prints each part by
the app's own bounce - registered as a stem in the same gesture, its MIDI track muted - and
mixes the prints, and the agent's own auto_mix tool is told what to say and offer."""
from __future__ import annotations

import shutil
from pathlib import Path

import orchestrator.daw_bridge as daw_bridge
from orchestrator import daw_server
from orchestrator.daw_server import Brain


def _edit(*tracks: dict) -> dict:
    return {"tracks": list(tracks)}


def _synth(name: str, *, clips=((0.0, 8.0),), **extra) -> dict:
    row = {"name": name, "kind": "stem", "track_kind": "midi", "instrument": "4osc",
           "muted": False, "clips": [{"index": i, "kind": "midi", "start_s": a, "length_s": n}
                                     for i, (a, n) in enumerate(clips)]}
    row.update(extra)
    return row


def test_the_parts_are_the_instrument_tracks_that_sound():
    edit = _edit(
        _synth("Bass", clips=((2.0, 4.0), (8.0, 8.0))),
        _synth("Muted keys", muted=True),
        _synth("Empty lead", clips=()),
        {"name": "Arp lane", "kind": "stem", "track_kind": "midi", "muted": False,
         "clips": [{"start_s": 0.0, "length_s": 4.0}]},             # no instrument: silent
        {"name": "drums", "kind": "bus", "instrument": "4osc"},      # a group is never a part
        {"name": "Vox", "kind": "stem", "track_kind": "audio", "muted": False,
         "clips": [{"start_s": 0.0, "length_s": 4.0}]},
        {"name": "Pad", "kind": "stem", "track_kind": "midi", "instrument": "plugin",
         "muted": False},                                            # no clip rows shown
    )
    parts = daw_bridge.midi_parts(edit)
    assert [p["track"] for p in parts] == ["Bass", "Pad"], parts
    assert parts[0] == {"track": "Bass", "start_s": 2.0,
                        "end_s": 16.0 + daw_bridge.MIDI_PRINT_TAIL_S}
    assert "start_s" not in parts[1]    # the app's bounce takes the track's own clips
    assert daw_bridge.midi_parts(lambda: edit) == parts   # a getter reads the same


def test_the_button_declines_a_midi_project_with_the_offer(tmp_path):
    project = tmp_path / "midi_only"
    project.mkdir()
    brain = Brain(project)
    brain.latest_edit = _edit(_synth("Bass"), _synth("Keys"))
    out = brain.auto_mix("master for streaming")
    assert out["ok"] is False and out["moves"] == []
    assert out["offer_print"] == ["Bass", "Keys"]
    assert "MIDI" in out["chat"] and "Bass, Keys" in out["chat"] and "print" in out["chat"]
    # the control: nothing on the lanes is the old sentence, with no offer
    brain.latest_edit = _edit()
    plain = brain.auto_mix("master for streaming")
    assert "offer_print" not in plain and "no audio to auto-mix" in plain["chat"]


def test_the_offer_taken_prints_mutes_registers_and_mixes(tmp_path, demo_project):
    project = tmp_path / "midi_song"
    project.mkdir()
    source = demo_project / "stem_a.wav"
    calls: list[tuple[str, dict]] = []

    def app(command: str, args: dict | None = None, *a, **kw) -> dict:
        args = dict(args or {})
        if command != "coedits.take":     # the brain taking the user's queued moves first
            calls.append((command, args))
        if command == "track.bounce":
            name = f"{args['track']} bounce"
            printed = project / f"{name}.wav"
            shutil.copy(source, printed)
            return {"ok": True, "track": name, "registered": False, "file": str(printed)}
        if command == "track.set_mute":
            return {"ok": True}
        return {"ok": True}

    brain = Brain(project)
    brain.daw_invoke = app
    brain.latest_edit = _edit(_synth("Bass", clips=((1.0, 6.0),)))
    stages: list[dict] = []
    out = brain.auto_mix("master for streaming", on_stage=stages.append, print_midi=True)

    assert [c for c, _ in calls] == ["track.bounce", "track.set_mute"], calls
    bounce = calls[0][1]
    assert bounce["track"] == "Bass" and bounce["register"] is False
    assert (bounce["start_s"], bounce["end_s"]) == (1.0, 7.0 + daw_bridge.MIDI_PRINT_TAIL_S)
    assert calls[1][1] == {"track": "Bass", "muted": True}
    assert out.get("printed") == ["Bass bounce"], out
    assert "Bass bounce" in daw_bridge.mixable_stems(project)      # registered in the gesture
    assert out.get("chains", {}).get("Bass bounce") is not None    # ...and mixed with the rest
    assert out.get("gesture")                                       # one undo takes it all back
    assert any(s.get("stage") == "printing to audio" and s.get("detail") == "Bass"
               for s in stages)


def test_a_print_that_fails_says_so_and_mixes_nothing(tmp_path):
    project = tmp_path / "midi_song"
    project.mkdir()
    brain = Brain(project)
    brain.daw_invoke = lambda command, args=None, *a, **kw: (
        {"ok": False, "error": "bounce render failed"} if command == "track.bounce" else {"ok": True})
    brain.latest_edit = _edit(_synth("Bass"))
    out = brain.auto_mix("", print_midi=True)
    assert out["ok"] is False and "could be printed" in out["chat"]
    assert "bounce render failed" in out["chat"]
    assert not out.get("gesture")       # nothing changed: no mark for Undo to pop


def test_the_request_carries_the_taken_offer_to_the_brain(monkeypatch):
    """`print_midi` reaches Brain.auto_mix from the socket's request, and only as true."""
    import asyncio

    seen: list = []

    class FakeBrain:
        def auto_mix(self, brief, should_cancel=None, on_stage=None, print_midi=False):
            seen.append(print_midi)
            return {"ok": True, "moves": []}

    async def send(_msg):
        return None

    async def no_door():
        return None

    monkeypatch.setattr(daw_server, "_auto_mix_refusal", no_door)
    asyncio.run(daw_server._do_auto_mix(send, FakeBrain(), 1, "", lambda: False, True))
    asyncio.run(daw_server._do_auto_mix(send, FakeBrain(), 2, "", lambda: False))
    assert seen == [True, False]
    src = Path(daw_server.__file__).read_text()
    assert 'params.get("print_midi") is True' in src


def test_the_agents_auto_mix_is_told_to_offer_the_print(tmp_path, monkeypatch):
    """The chat tool: nothing runs over MIDI parts, and the model is handed what to say."""
    from orchestrator.chat import execute_tool

    project = tmp_path / "midi_only"
    project.mkdir()
    ran: list = []
    monkeypatch.setattr(daw_bridge, "auto_mix_pass", lambda *a, **kw: ran.append(1) or 0)

    class Turn:
        results: list = []

        async def turn(self, message, *, tools, emit, **kw):
            self.results.append(await execute_tool(tools, "auto_mix",
                                                   {"target": "master for streaming"}))
            await emit({"type": "chat_message", "text": "done"})

    turn = Turn()
    daw_bridge._CHAT_SESSIONS[str(project)] = turn
    try:
        daw_bridge.chat_turn(project, "auto mix it", mode="agent", use_llm=False,
                             edit_state=lambda: _edit(_synth("Bass")))
    finally:
        daw_bridge._CHAT_SESSIONS.clear()
    said = turn.results[-1].text
    assert ran == [], "Auto Mix ran over a project with no audio"
    assert "MIDI" in said and "Bass" in said and "track.bounce" in said and "offer" in said


def test_auto_mix_leaves_a_deleted_track_alone(demo_project):
    """A deleted track's stem stays in the session, excluded. Auto Mix profiled it and proposed
    on it, the guardrails refused every such bundle, and the feed said "Auto Mix: refused -
    stem 'vocals' is excluded" over tracks the user had deleted (seen in the chat rig, printing
    a MIDI part in a project whose audio tracks had gone)."""
    from mix_engine.session import Session

    Session(demo_project).user_edit("stem_b", {"excluded": True})
    refused: list = []
    daw_bridge.auto_mix_pass(demo_project, "master for streaming", refused=refused)
    assert refused == [], refused
    applied = daw_bridge.applied_proposals_since(demo_project, 0)
    targets = {a.get("stem_id") for r in applied for a in (r.get("actions") or [])}
    assert "stem_b" not in targets, targets
    assert "stem_a" in targets or applied, "nothing was mixed at all"
