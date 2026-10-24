"""The pre-merge review of 2026-10-23: the renames a DAW reply carries (findings 1 and 2).

Since the real-use QA of 2026-10-22 the app hands the brain its waiting renames on every reply
that hands it the co-edit queue, and the brain follows them (`Brain.follow_renames`) before it
reads the reply's moves.

1. It took the brain's `mutate_lock` to do so - and `Brain.undo` holds that lock while it asks
   the app for `edit.undo`. An undo that takes an agent's rename back queues the rename back,
   its reply carries it, and the brain waited for a lock its own thread held: for ever.
2. A rename it could not follow was a line on stderr. The drain it replaced said a refusal in
   the status line; on a reply the one reading it is the model, so the reply says it.
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading

from orchestrator import daw_bridge as B
from orchestrator import daw_server as S
from tests import test_agent_audit_p2 as P


def test_a_rename_on_a_reply_is_followed_by_a_thread_that_holds_the_lock(tmp_path):
    """Finding 1: `Brain.undo` runs under `mutate_lock` (`locked_thread`) and reads the reply
    of `edit.undo` on that same thread."""
    proj = P._project(tmp_path)
    brain = S.Brain(proj)
    done = threading.Event()

    def an_undo_reading_its_reply():
        with brain.mutate_lock:
            brain.follow_renames([{"old": "bass", "new": "Low"}])
        done.set()

    threading.Thread(target=an_undo_reading_its_reply, daemon=True).start()
    assert done.wait(10.0), "follow_renames waited for a lock its own thread holds"
    assert "Low" in P._state(proj)["stems"] and "bass" not in P._state(proj)["stems"]


def test_a_rename_the_brain_cannot_follow_is_said_in_the_reply(tmp_path, monkeypatch):
    """Finding 2: a chat turn's command reply carries a rename the brain refuses (`master` is
    reserved); the result the model reads says so, by name."""
    proj = P._project(tmp_path)
    P._sub_env(monkeypatch)
    monkeypatch.setenv("MIX_AGENT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(B, "_CHAT_SESSIONS", {})
    seen: list = []

    async def script(tools, m):
        seen.append(await tools["daw_command"].handler(
            {"command": "track.rename", "args": {"track": "bass", "name": "master"}}))
        yield m.AssistantMessage([])
        yield m.AssistantMessage([m.TextBlock("Renamed.")])
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", P._fake_sdk(script))

    async def scenario(hub, port):
        reader, writer = await P._authed(port, hub.token, proj, "daw")
        try:
            writer.write(P._line({"id": 5, "method": "chat", "params": {
                "message": "call the bass master", "mode": "agent", "agent": "claude"}}))
            await writer.drain()
            while True:
                msg = json.loads(await asyncio.wait_for(reader.readline(), 60))
                if "daw_request" in msg:
                    writer.write(P._line({"daw_reply": msg["daw_request"], "result": {
                        "ok": True, "renames": [{"old": "bass", "new": "master"}]}}))
                    await writer.drain()
                ev = str(msg.get("event", ""))
                if ev == "chat_done" or "fail" in ev:
                    return msg
        finally:
            writer.close()

    done = P._serve(proj, scenario)
    assert done.get("event") == "chat_done", done
    text = json.dumps(seen)
    assert "renames_not_followed" in text and "bass" in text, text
    assert "bass" in P._state(proj)["stems"]          # nothing half-done in the session


# ------------------------------------------------- findings 6, 7, 9, 10 (the second half)

import time
from pathlib import Path

import anyio

from orchestrator import arrangement, gateway, turn_host
from tests import test_user_qa_2026_10_22_open as Q


def test_the_planner_leaves_out_a_track_whose_audio_clips_are_all_muted():
    """Finding 6: a split mutes its source (`landSplitStems`), and the build reads only the clips
    that play (Q10) - the app refused the WHOLE plan over a track it had nothing to arrange from,
    and the planner named that track in every "arrange the song" it made."""
    edit = {"tracks": [
        {"name": "song", "clips": [{"source": "song.wav", "length_s": 120.0, "muted": True}]},
        {"name": "drums", "clips": [{"source": "drums.wav", "length_s": 120.0}]},
        {"name": "keys", "clips": [{"kind": "midi", "length_s": 8.0, "muted": True}]}]}
    assert [t["name"] for t in arrangement._arrangeable_tracks(edit)] == ["drums"]


def test_a_command_that_only_looks_keeps_the_apps_render_of_the_mix(demo_project, tmp_path):
    """Finding 7: every command the agent ran dropped the app's render of the mix, reads
    included, so the next reading of the master rendered the whole song in the app again."""
    calls: list = []
    turn = Q._turn(demo_project, Q.MIDI_EDIT, calls, tmp_path / "mix.wav")
    anyio.run(turn.do_render)
    for key in ("track.spectrogram", "view.snapshot", "transport.play_pause", "project.peek"):
        turn._note_command(key)
    anyio.run(turn.do_render)
    assert calls == ["mix.measure"], calls
    turn._note_command("track.set_gain")             # control: a move still drops it
    anyio.run(turn.do_render)
    assert calls == ["mix.measure"] * 2, calls


def test_the_mix_neutral_keys_are_commands_the_app_has():
    """...and the list of commands that cannot change what plays names real catalog keys."""
    catalog = (Path(__file__).resolve().parents[2] / "daw" / "Source" / "DawCommands.h").read_text()
    for key in B.MIX_NEUTRAL_COMMANDS:
        assert f'"{key}"' in catalog, key


def test_the_apps_mix_stands_only_for_the_state_it_was_measured_at(demo_project):
    """Finding 9: `dsp.native_mix` is the process's: a co-edit applied on another thread while a
    turn reads the master rendered the app's mix of the state BEFORE that co-edit, and its
    make-up and readout were kept for the state after it."""
    from mix_engine import dsp
    from mix_engine.session import Session

    stem = sorted(Session(demo_project).stems)[0]
    native = Q._tone(db=-12.0)
    with dsp.native_mix(demo_project, native, Q.RATE):
        report: dict = {}
        dsp.render_master(Session(demo_project, read_only=True), report=report)
        assert report.get("native_mix"), "control: at the state it was measured at, it stands in"
        B.apply_coedits(demo_project, [f"http://127.0.0.1:8765/api/set_stem?stem={stem}&gain_db=-9.0"])
        report = {}
        dsp.render_master(Session(demo_project, read_only=True), report=report)
        assert not report.get("native_mix"), "a state the app's mix was not measured at"


def test_listing_a_folder_leaves_the_turns_event_loop_free(tmp_path, monkeypatch):
    """Finding 10: `list_files` read the disk on the turn's event loop - a sleeping share held
    every other task of the turn (the stop poll, the stream) for as long as it took."""
    def slow(path, query=""):
        time.sleep(0.6)
        return {"ok": True, "path": path, "folders": [], "audio": [], "audio_count": 0}
    monkeypatch.setattr(B, "folder_listing", slow)
    turn = turn_host.LiveTurn.__new__(turn_host.LiveTurn)
    gaps: list = []

    async def main():
        done = False

        async def ticker():
            last = time.monotonic()
            while not done:
                await anyio.sleep(0.01)
                now = time.monotonic()
                gaps.append(now - last)
                last = now
        async with anyio.create_task_group() as tg:
            tg.start_soon(ticker)
            await anyio.sleep(0.05)
            await turn.do_list_files(str(tmp_path), "")
            done = True
    anyio.run(main)
    assert max(gaps) < 0.3, f"the loop stood still for {max(gaps):.2f} s"


def test_the_offline_engineers_listing_keeps_what_the_daws_keeps(tmp_path):
    """...and the offline gateway's `list_files` answered the same `query` its own way (audio
    narrowed, folders not)."""
    for d in ("kick loops", "snare kit"):
        (tmp_path / d).mkdir()
    for f in ("kick 1.wav", "snare.wav"):
        (tmp_path / f).write_bytes(b"")
    g = gateway.Gateway(None, None)
    got = anyio.run(g._chat_list_files, str(tmp_path), "kick")
    want = B.folder_listing(str(tmp_path), "kick")
    assert (got["folders"], got["audio"]) == (want["folders"], want["audio"]) == (["kick loops"], ["kick 1.wav"])


def test_the_engines_own_reading_still_takes_the_apps_mix(demo_project):
    """...and the state the engine reads at IS the state measured: through the real in-memory
    engine, `analyze_session` inside the block reads the app's mix (Q13 stands)."""
    from mix_engine import dsp
    from mix_engine.session import Session
    from orchestrator.engine_client import in_memory_engine

    Session(demo_project)
    native = Q._tone(db=-12.0)

    async def main():
        async with in_memory_engine(demo_project) as engine:
            with dsp.native_mix(demo_project, native, Q.RATE):
                inside = await engine.call_tool("analyze_session", {})
            outside = await engine.call_tool("analyze_session", {})
        return inside.structuredContent, outside.structuredContent
    inside, outside = anyio.run(main)
    unwrap = lambda p: p.get("result", p)
    assert abs(unwrap(inside)["lufs_i"] - Q._lufs(native)) < 0.5, unwrap(inside).get("lufs_i")
    assert abs(unwrap(outside)["lufs_i"] - Q._lufs(native)) > 1.0, "control: outside, the brain's own"
