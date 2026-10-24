"""Gates for batch Py4 of the reader P2s (full audit 2026-10-02): what the brain tells the agent.

Nine findings - C4·R8, C4·R9, C4·R11, C4·R12, E6·R5, E6·R6, E6·R7, E6·R8, S4·R30. Each test
drives the door the product uses and carries its control. Every one was run on the unfixed tree
first. The record is docs/audits/full-2026-10-02/readers/P2-fixes.md (Py4).
"""
from __future__ import annotations

import json
from pathlib import Path

import anyio
import pytest
import soundfile as sf
from mcp.shared.memory import create_connected_server_and_client_session

import test_agent_audit_p2 as P2
import test_agent_senses as SENSES
from mix_engine import analysis
from mix_engine.mcp_server import create_server
from orchestrator import chat as chatmod
from orchestrator import daw_bridge as B
from orchestrator import daw_server as S
from orchestrator import edit_state as E
from orchestrator.agents import ClaudeAgent, PassContext
from orchestrator.calibration import load_decisions
from test_phase1_acceptance import call

TAIL = ("tempo", "transport", "sections", "markers", "selection", "undo")
METERS = ("in_db", "out_db", "delta_db", "out_peak_db", "gr_db")


# ------------------------------------------------------------------ snapshots in the app's shape

def _device_row(kind: str, occ: int = 0, level: float = -18.0) -> dict:
    """One row of `tracks[].devices` as MonitorChainPlugin::getDeviceLevelsVar writes it."""
    return {"type": kind, "occ": occ, "enabled": True, "in_db": level, "out_db": level - 1.5,
            "delta_db": -1.5, "out_peak_db": level + 6.0, "gr_db": -2.0}


def _track(i: int, *, midi: bool = False, notes: int = 64) -> dict:
    row = {"index": i, "name": f"track_{i:02d}", "kind": "midi" if midi else "audio",
           "mute": False, "solo": False, "gain_db": -3.0, "pan": 0.0, "width": 1.0,
           "output": "master", "device_count": 3,
           "devices": [_device_row("eq"), _device_row("comp"), _device_row("saturation")],
           "automatable_targets": [f"eq#0.band{b}.gain_db" for b in range(8)],
           "clips": [{"index": c, "name": f"clip {c}", "start_s": 4.0 * c, "len_s": 4.0,
                      "source": f"/Users/someone/Music/proj/track_{i:02d}.wav"}
                     for c in range(4)]}
    if midi:
        row["clips"] = [{"index": 0, "name": "part", "start_s": 0.0, "len_s": 16.0,
                         "notes": [{"n": 48 + (k % 24), "b": k * 0.25, "l": 0.25, "v": 96}
                                   for k in range(notes)]} for _ in range(2)]
    return row


def _snapshot(audio: int = 8, midi: int = 2, *, catalog: bool = False, notes: int = 64) -> dict:
    """tracks first, the session-global fields after it - the order EditSnapshot.h writes."""
    snap = {"seq": 41,
            "tracks": [_track(i) for i in range(audio)]
                      + [_track(audio + i, midi=True, notes=notes) for i in range(midi)],
            "tempo": {"bpm": 96.0, "time_signature": "4/4"},
            "transport": {"position_s": 12.5, "playing": False, "loop": False},
            "undo": {"can_undo": True, "next": "Agent: brighten the vocal"},
            "markers": [{"name": "drop", "at_s": 32.0}],
            "sections": [{"name": "verse", "start_s": 0.0, "end_s": 16.0},
                         {"name": "chorus", "start_s": 16.0, "end_s": 32.0}],
            "selection": {"tracks": ["track_03"], "focused_track": "track_03"}}
    if catalog:
        snap["commands"] = [{"key": f"family.command_{k}", "args": "{ track, clip?, value_db }",
                             "help": "x" * 180} for k in range(300)]
    return snap


def _rack(i: int, chains: int = 3, devices: int = 4) -> dict:
    """One `racks[]` entry as EditSnapshot.h `rackVar` writes it: the parameters sit at
    racks[].chains[].devices[].params - one level below where the tier used to look."""
    def chain(c: int) -> dict:
        return {"index": c, "name": f"chain {c}", "mute": False, "solo": False,
                "volume_db": 0.0, "pan": 0.0,
                "devices": [{"slot": 0, "kind": "device", "name": "Compressor", "device": "comp",
                             "device_index": d, "occ": d, "node": f"rack:{i}:{c}",
                             "params": {"type": "comp", "threshold_db": -18.0, "ratio": 4.0,
                                        "attack_ms": 10.0, "release_ms": 120.0, "knee_db": 6.0,
                                        "makeup_db": 0.0, "mix": 1.0, "lookahead_ms": 0.0,
                                        "sidechain_hpf_hz": 80, "detector": "peak",
                                        "channel": "stereo", "enabled": True}}
                            for d in range(devices)]
                           + [{"slot": 1, "name": "Hosted Synth", "format": "VST3",
                               "param_count": 300,
                               "params": [{"index": p, "name": f"Parameter {p}", "value": 0.5}
                                          for p in range(12)]}]}
    return {"name": f"Rack {i}", "plugins": ["Zammoth", "Hosted Synth"],
            "connections": 9, "chains": [chain(c) for c in range(chains)],
            "macros": [{"index": m, "name": f"Macro {m + 1}", "value": 0.5,
                        "targets": [{"chain": 0, "slot": 0, "param": "threshold_db"}]}
                       for m in range(8)]}


# ====================================================================================
# C4·R8 - the planner and the mid-run answer cut the snapshot by characters
# ====================================================================================

def _planner_block(snap: dict) -> dict:
    ctx = PassContext(pass_number=1, metrics={}, targets=[], stem_chains={}, edit_state=snap)
    return json.loads(ClaudeAgent._context_block(ctx))


def test_the_planner_reads_a_whole_snapshot_however_big_the_song():
    snap = _snapshot(audio=8, midi=2, catalog=True, notes=256)
    assert len(json.dumps(snap["tracks"], separators=(",", ":"))) > 24000, "vacuity: must overflow"
    state = _planner_block(snap)["daw_edit_state"]
    assert isinstance(state, dict), "the planner was handed the head of a string, not a snapshot"
    for key in TAIL:
        assert key in state, f"{key} was cut off the end"
    assert state["selection"]["focused_track"] == "track_03"
    assert state["sections"][1]["name"] == "chorus"
    assert "commands" not in state
    assert state["omitted"], "what went must be named"
    assert [t["name"] for t in state["tracks"]] == [t["name"] for t in snap["tracks"]]


def test_a_small_song_reaches_the_planner_whole():
    """Control: nothing is dropped from a session that fits."""
    snap = _snapshot(audio=1, midi=0)
    state = _planner_block(snap)["daw_edit_state"]
    assert "omitted" not in state
    assert state["tracks"][0]["devices"] == snap["tracks"][0]["devices"]
    assert state["tracks"][0]["clips"][0]["source"] == "/Users/someone/Music/proj/track_00.wav"


def _midrun_context(monkeypatch, snap: dict, message: str = "what's selected?") -> str:
    seen = {}

    class FakeAgent:
        usage = {"estimated_cost_usd": 0.0}

        def chat(self, message, context=None):
            seen["context"] = context
            return "answered"

    monkeypatch.setattr("orchestrator.llm_backend.llm_available", lambda: True)
    monkeypatch.setattr(B, "_llm_agent", lambda llm, what: FakeAgent())
    live = {"kind": "run", "brief": "warm master", "pass_number": 2, "max_passes": 5,
            "started_at": 0.0, "moves": []}
    assert B.chat_status_reply(message, live, readout={"lufs_i": -14.0}, edit_state=snap) == "answered"
    return seen["context"]


def _arrangement_of(context: str) -> dict:
    part = next(p for p in context.split("\n\n") if p.startswith("The arrangement: "))
    return json.loads(part[len("The arrangement: "):])


def test_the_mid_run_answer_reads_a_whole_snapshot_without_the_catalog(monkeypatch):
    snap = _snapshot(audio=4, midi=0, catalog=True)
    context = _midrun_context(monkeypatch, snap)
    state = _arrangement_of(context)          # valid JSON: no character cut
    for key in TAIL:
        assert key in state, f"{key} was cut off the end"
    assert "commands" not in state and "family.command_7" not in context, \
        "the command catalog rode the arrangement and pushed the session out"
    assert [t["name"] for t in state["tracks"]] == [f"track_{i:02d}" for i in range(4)]


def test_the_mid_run_answer_keeps_a_small_song_whole(monkeypatch):
    """Control."""
    snap = _snapshot(audio=1, midi=0)
    state = _arrangement_of(_midrun_context(monkeypatch, snap))
    assert "omitted" not in state and state["tracks"][0]["devices"] == snap["tracks"][0]["devices"]


# ====================================================================================
# C4·R9 - the engine's reference tools and the `~/...` path the model is shown
# ====================================================================================

@pytest.mark.anyio
async def test_a_reference_named_by_its_home_path_is_read_from_home(demo_project, tmp_path,
                                                                    monkeypatch):
    home = tmp_path / "home"
    (home / "Music" / "refs").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    audio, sr = sf.read(str(demo_project / "stem_a.wav"), dtype="float32")
    absolute = home / "Music" / "refs" / "song.wav"
    sf.write(str(absolute), audio, sr)
    shown = E.redact_home_text(str(absolute))
    assert shown == "~/Music/refs/song.wav", "vacuity: this is the form the model holds"

    async with create_connected_server_and_client_session(create_server(demo_project)) as client:
        by_absolute = await call(client, "analyze_reference", {"path": str(absolute)})   # control
        by_home = await call(client, "analyze_reference", {"path": shown})
        assert by_home["lufs_i"] == by_absolute["lufs_i"]
        as_set = await call(client, "analyze_reference_set",
                            {"references": [{"path": shown, "role": "tonality"}]})
        assert as_set["references"][0]["analysis"]["lufs_i"] == by_home["lufs_i"]
        # ...and a path inside the project is still read from the project (control)
        (demo_project / "refs").mkdir()
        sf.write(str(demo_project / "refs" / "local.wav"), audio, sr)
        local = await call(client, "analyze_reference", {"path": "refs/local.wav"})
        assert local["lufs_i"] == by_home["lufs_i"]
        missing = await client.call_tool("analyze_reference", {"path": "~/Music/refs/nope.wav"})
        assert missing.isError
        said = missing.content[0].text
        assert str(demo_project) not in said, f"a home path was looked for inside the project: {said}"


# ====================================================================================
# C4·R11 - racks could not be compacted; the tests passed on a shape the app never writes
# ====================================================================================

def _rack_snapshot(racks: int = 6, tracks: int = 8) -> dict:
    snap = _snapshot(audio=tracks, midi=0)
    snap["racks"] = [_rack(i) for i in range(racks)]
    snap["library"] = {"name": "My Song", "item_count": 137,
                       "items": [{"id": f"1/{i}", "name": f"sample {i}.wav", "type": "wave",
                                  "category": "imported", "length_s": 2.0,
                                  "path": f"/Users/someone/Music/proj/Imported/sample {i}.wav"}
                                 for i in range(137)]}
    return snap


def _rack_params_left(state: dict) -> int:
    return sum(1 for r in state.get("racks") or [] for c in r.get("chains") or []
               for d in c.get("devices") or [] if "params" in d)


def test_six_racks_fit_the_budget_and_every_track_keeps_its_name():
    """The reader's scenario: 6 racks + 8 tracks at the chat's 12 000 was 30 000 characters
    after every tier, with `tracks_omitted: 8`."""
    snap = _rack_snapshot()
    text = E.compact_edit_state(snap, 12000)
    state = json.loads(text)
    assert len(text) <= 12000, f"{len(text)} characters after every tier"
    assert "tracks_omitted" not in state, "the track list went before a single rack did"
    assert [t["name"] for t in state["tracks"]] == [t["name"] for t in snap["tracks"]]
    assert [r["name"] for r in state["racks"]] == [f"Rack {i}" for i in range(6)], \
        "a rack is still named when it is thinned"
    for key in TAIL:
        assert key in state


def test_rack_parameters_go_with_the_device_params_tier():
    snap = _rack_snapshot(racks=2, tracks=2)
    whole = len(E._dumps(snap))
    assert _rack_params_left(json.loads(E.compact_edit_state(snap, whole + 1000))) == 2 * 3 * 5, \
        "control: a snapshot that fits keeps every rack parameter"
    lean = json.loads(E._dumps(snap))
    E._drop(lean, "device_params")
    assert _rack_params_left(lean) == 0, "the tier is recorded as applied and removed nothing"
    rows = [d for r in lean["racks"] for c in r["chains"] for d in c["devices"]]
    assert all(d.get("params_omitted") is True for d in rows), "what went must be labelled"
    assert rows[0]["node"] == "rack:0:0" and rows[0]["device"] == "comp", \
        "the address the agent needs to act on the device stays"


def test_a_thinned_rack_keeps_what_names_it():
    snap = _rack_snapshot(racks=1, tracks=1)
    E._drop(snap, "racks")
    rack = snap["racks"][0]
    assert rack["detail_omitted"] is True
    assert rack["name"] == "Rack 0" and rack["plugins"] == ["Zammoth", "Hosted Synth"]
    assert rack["chain_names"] == ["chain 0", "chain 1", "chain 2"]
    assert rack["macro_names"] == [f"Macro {m + 1}" for m in range(8)]
    assert "chains" not in rack and "macros" not in rack
    assert E.DROP_ORDER.index("racks") < E.DROP_ORDER.index("track_detail"), \
        "a rack's detail goes before any track is thinned"


def test_a_thinned_rack_is_one_read_away():
    """`detail_omitted` is a promise that the detail can be had: read_track answers for a rack."""
    snap = _rack_snapshot(racks=2, tracks=2)
    out = B.read_track_entry(snap, "rack:Rack 1")
    assert out["ok"] is True and out["rack"]["name"] == "Rack 1"
    assert out["rack"]["chains"][0]["devices"][0]["params"]["threshold_db"] == -18.0
    assert B.read_track_entry(snap, "Rack 1")["rack"]["name"] == "Rack 1"
    snap["racks"][1]["name"] = "track_00"          # control: a track of that name still wins
    assert B.read_track_entry(snap, "track_00")["track"]["name"] == "track_00"
    assert B.read_track_entry(snap, "rack:track_00")["rack"]["name"] == "track_00"
    missing = B.read_track_entry(snap, "rack:nope")
    assert missing["ok"] is False and "Rack 0" in str(missing)


def test_the_library_is_counted_by_its_items():
    snap = _rack_snapshot(racks=0, tracks=1)
    E._drop(snap, "library")
    assert snap["library_omitted"] == 137, "it counted the library object's three keys"
    listed = {"library": [{"id": i} for i in range(9)]}      # control: the old list shape
    E._drop(listed, "library")
    assert listed["library_omitted"] == 9


# ====================================================================================
# C4·R12 - the delta said every track was removed, and re-sent every track while audio played
# ====================================================================================

def _compact(snap: dict) -> dict:
    return json.loads(E.compact_edit_state(snap, 10 ** 9))


def test_a_track_list_that_no_longer_fits_is_not_reported_as_removed():
    old = _compact(_snapshot(audio=6, midi=0))
    new = json.loads(json.dumps(old))
    E._drop(new, "tracks")
    new["omitted"] = ["tracks"]
    delta = E.edit_state_delta(old, new)
    assert "tracks_removed" not in delta, "29 tracks 'removed' by a budget, not by anyone"
    assert "track_order" not in delta, "an empty order reads as an emptied session"
    assert delta["tracks_omitted"] == 6
    assert "still" in str(delta["tracks_not_shown"]) and "read_track" in str(delta["tracks_not_shown"])


def test_a_track_that_was_really_removed_is_still_reported():
    """Control: the delta's own job is unchanged."""
    old = _compact(_snapshot(audio=6, midi=0))
    new = json.loads(json.dumps(old))
    del new["tracks"][2]
    delta = E.edit_state_delta(old, new)
    assert delta["tracks_removed"] == ["name:track_02"]
    assert delta["track_order"] == [f"name:track_{i:02d}" for i in (0, 1, 3, 4, 5)]
    assert "tracks_not_shown" not in delta


def test_the_note_the_model_stores_never_says_removed_at_a_tier_crossing():
    """Through `chat._live_refresh`, the text that goes into the stored history."""
    head = P2.LIVE_HEAD
    old = _compact(_snapshot(audio=6, midi=0))
    new = json.loads(json.dumps(old))
    E._drop(new, "tracks")
    new["omitted"] = ["tracks"]
    note = chatmod._live_refresh(head + E._dumps(old), head + E._dumps(new))
    assert "tracks_removed" not in note and '"track_order":[]' not in note


def _with_meters_moved(state: dict, by: float) -> dict:
    moved = json.loads(json.dumps(state))

    def walk(v):
        if isinstance(v, dict):
            for key in METERS:
                if key in v:
                    v[key] = round(v[key] + by, 1)
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
    walk(moved)
    return moved


def _playing_snapshot() -> dict:
    snap = _compact(_snapshot(audio=12, midi=0))
    row = {"name": "master", "kind": "master", "gain_db": 0.0, "device_count": 2,
           "devices": [_device_row("eq"), _device_row("limiter")]}
    snap["master"] = row
    snap["returns"] = [{**row, "name": "return:plate", "kind": "return",
                        "devices": [_device_row("return_reverb")]}]
    snap["groups"] = [{"name": "drums", "members": ["track_00", "track_01"],
                       "devices": [_device_row("comp")]}]
    return snap


def test_meters_alone_are_not_a_change():
    old = _playing_snapshot()
    new = _with_meters_moved(old, 2.3)
    assert new["tracks"][0]["devices"][0]["in_db"] != old["tracks"][0]["devices"][0]["in_db"], "vacuity"
    assert E.edit_state_delta(old, new) is None, \
        "every device-bearing track was re-sent because audio was playing"


def test_a_real_change_beside_moving_meters_is_reported_alone_and_whole():
    """Control: one fader moved while every meter did - that track goes, with its meters as
    they are NOW, and only that track."""
    old = _playing_snapshot()
    new = _with_meters_moved(old, 2.3)
    new["tracks"][4]["gain_db"] = -9.0
    new["master"]["devices"][1]["enabled"] = False
    delta = E.edit_state_delta(old, new)
    assert [t["name"] for t in delta["tracks_changed"]] == ["track_04"]
    assert delta["tracks_changed"][0] == new["tracks"][4], "the row sent is the current row"
    assert delta["master"] == new["master"]
    assert "returns" not in delta and "groups" not in delta


def test_ten_commands_under_a_rolling_transport_cost_what_they_cost_stopped():
    """The reader's measurement: 105 407 characters of notes against 9 893."""
    head = P2.LIVE_HEAD

    def notes(playing: bool) -> int:
        shown = _playing_snapshot()
        total = 0
        for step in range(10):
            fresh = _with_meters_moved(shown, 0.7 * (step + 1)) if playing else json.loads(json.dumps(shown))
            fresh["tracks"][step]["gain_db"] = -6.0 - step        # the command's own change
            total += len(chatmod._live_refresh(head + E._dumps(shown), head + E._dumps(fresh)))
            shown = fresh
        return total
    stopped, rolling = notes(False), notes(True)
    assert rolling <= stopped * 1.1, f"{rolling} characters rolling against {stopped} stopped"


def test_a_row_that_was_only_thinned_is_named_not_sent_again():
    old = _compact(_snapshot(audio=4, midi=0))
    new = json.loads(json.dumps(old))
    for t in new["tracks"][:3]:
        E._thin_track(t)
    new["tracks"][1]["mute"] = True                # ...and one of them really changed
    delta = E.edit_state_delta(old, new)
    assert [t["name"] for t in delta["tracks_changed"]] == ["track_01"], \
        "a thinned row whose mute changed is a change (control)"
    assert delta["tracks_detail_not_shown"] == ["track_00", "track_02"]
    assert "track_order" not in delta
    back = E.edit_state_delta(new, old)            # control: detail that came back is sent
    assert sorted(t["name"] for t in back["tracks_changed"]) == ["track_00", "track_01", "track_02"]


# ====================================================================================
# E6·R5 - an approved plan ran the full five passes
# ====================================================================================

def _capture_runs(monkeypatch) -> list[dict]:
    seen: list[dict] = []

    def fake_run(project_dir, brief, **kw):
        seen.append({"brief": brief, **kw})
        return {"passes": 1, "stop_reason": "pass budget exhausted", "cost_usd": 0.0}
    monkeypatch.setattr(B, "run_agent_pass", fake_run)
    return seen


def test_an_approved_plan_runs_the_passes_the_model_asked_for(tmp_path, monkeypatch):
    proj = P2._project(tmp_path)
    seen = _capture_runs(monkeypatch)
    session, out, _ = P2._scripted_turn(
        proj, [("start_run", {"brief": "master for streaming", "max_passes": 1})],
        monkeypatch, mode="ask")
    assert not seen, "vacuity: Ask mode records, it does not run"
    step = out["plan"]["steps"][0]
    assert step["kind"] == "run" and step["payload"]["max_passes"] == 1, \
        "the pass count the model asked for was dropped at the recorder"
    assert "1 pass" in session.log[0][1], "the model is told what it recorded"

    B.apply_plan(proj, out["plan"], use_llm=False)
    assert len(seen) == 1 and seen[0]["brief"] == "master for streaming"
    assert seen[0].get("max_passes") == 1, "the plan the user approved is not the run that ran"


def test_a_plan_with_no_pass_count_runs_the_default(tmp_path, monkeypatch):
    """Control: `start_run {brief}` alone is the ordinary run, in a plan and out of one."""
    from orchestrator import loop
    proj = P2._project(tmp_path)
    seen = _capture_runs(monkeypatch)
    _, out, _ = P2._scripted_turn(proj, [("start_run", {"brief": "balance it"})],
                                  monkeypatch, mode="ask")
    B.apply_plan(proj, out["plan"], use_llm=False)
    asked = seen[0].get("max_passes")
    assert not asked or asked >= loop.MAX_PASSES
    P2._scripted_turn(proj, [("start_run", {"brief": "one quick pass", "max_passes": 1})],
                      monkeypatch, mode="agent")
    assert seen[1]["max_passes"] == 1, "agent mode already passed it (F7)"


# ====================================================================================
# E6·R6 - an A/B about one move took back, and recorded, the whole turn
# ====================================================================================

SET_GAIN = ("daw_command", {"command": "track.set_gain", "args": {"track": "bass", "db": -4.0}})
SET_PAN = ("daw_command", {"command": "track.set_pan", "args": {"track": "bass", "pan": 0.2}})


def _pending(proj: Path) -> dict:
    return json.loads((proj / "runs" / B.PENDING_PREFERENCE_FILE).read_text())


def _applied_ids(proj: Path) -> list[str]:
    return [p["proposal_id"] for p in B.applied_proposals_since(proj, -1)]


def _decisions(proj: Path) -> list[tuple[str, str]]:
    return [(d["decision"], d.get("proposal_id") or d["action_key"])
            for d in load_decisions(proj / "runs")]


def _three_move_turn(proj: Path, monkeypatch, ask: dict):
    return P2._scripted_turn(proj, [("propose", P2._gain_req("vocals", -1.0)), SET_GAIN,
                                    ("propose", P2._gain_req("bass", -2.0)),
                                    ("ask_preference", ask)], monkeypatch)


def test_an_ab_is_about_the_last_move_of_the_turn(tmp_path, monkeypatch):
    proj = P2._project(tmp_path)
    session, _, _ = _three_move_turn(proj, monkeypatch, {"about": "the bass level"})
    assert not session.log[-1][2], session.log[-1]
    first, last = _applied_ids(proj)
    assert _pending(proj)["move"] == {"proposal_id": last}

    answer, _, _ = P2._scripted_turn(proj, [], monkeypatch, message=chatmod.PREFERENCE_OPTIONS[0])
    told = answer.history[0]["content"]
    assert f"undo_proposal {last}" in told
    assert first not in told, "A on the last move told the model to undo the move before it too"
    assert "track.set_gain" not in told and "edit.undo for" not in told, \
        "A on the last move told the model to take back a fader nobody asked about"
    assert _decisions(proj) == [("rejected", last)], \
        "taste learned a rejection for moves the user never objected to"


def test_an_ab_can_name_the_move_it_is_about(tmp_path, monkeypatch):
    proj = P2._project(tmp_path)
    _three_move_turn(proj, monkeypatch, {"about": "the vocal level", "move": "p0001"})
    first, last = _applied_ids(proj)
    assert first == "p0001", "vacuity: the id the script names is the first move's"
    assert _pending(proj)["move"] == {"proposal_id": first}
    out = B.record_preference_answer(proj, chatmod.PREFERENCE_OPTIONS[1])
    assert out["proposal_ids"] == [first] and out["commands"] == []
    assert _decisions(proj) == [("approved", first)]


def test_an_ab_about_a_command_names_that_command_alone(tmp_path, monkeypatch):
    proj = P2._project(tmp_path)
    P2._scripted_turn(proj, [SET_PAN, SET_GAIN,
                             ("ask_preference", {"about": "the bass level",
                                                 "move": "track.set_gain"})], monkeypatch)
    assert _pending(proj)["move"] == {"command": "track.set_gain", "alone": False}
    out = B.record_preference_answer(proj, chatmod.PREFERENCE_OPTIONS[0])
    assert out["commands"] == ["track.set_gain"] and out["proposal_ids"] == []
    rows = load_decisions(proj / "runs")
    assert [(r["decision"], r["action_key"]) for r in rows] == [("rejected", "track.set_gain")], \
        "the pan the turn also set was recorded with it"
    note = B.preference_note(out)
    assert "track.set_gain" in note and "track.set_pan" not in note
    assert "NOT edit.undo" in note, \
        "edit.undo takes back every command of that turn - the pan with the fader"


def test_a_turn_whose_one_command_is_the_move_may_undo_it(tmp_path, monkeypatch):
    """Control for the note: one command in the turn, so one Cmd-Z is exactly that move - when
    the command is a step of the timeline's undo. (This test used a fader until the pre-merge
    review of 2026-10-06: a level is written past that undo, so `edit.undo` took back whatever
    else was on top and left the fader. test_premerge_2026_10_06 pins the level case.)"""
    proj = P2._project(tmp_path)
    split = ("daw_command", {"command": "clip.split", "args": {"track": "bass", "clip": 0, "at_s": 1.0}})
    P2._scripted_turn(proj, [split, ("ask_preference", {"about": "the cut"})], monkeypatch)
    assert _pending(proj)["move"] == {"command": "clip.split", "alone": True}
    note = B.preference_note(B.record_preference_answer(proj, chatmod.PREFERENCE_OPTIONS[0]))
    assert "edit.undo" in note and "NOT edit.undo" not in note


def test_a_lone_fader_move_is_put_back_by_hand(tmp_path, monkeypatch):
    proj = P2._project(tmp_path)
    P2._scripted_turn(proj, [SET_GAIN, ("ask_preference", {"about": "the bass level"})],
                      monkeypatch)
    assert _pending(proj)["move"] == {"command": "track.set_gain", "alone": True}
    note = B.preference_note(B.record_preference_answer(proj, chatmod.PREFERENCE_OPTIONS[0]))
    assert "NOT edit.undo" in note and "same command" in note


def test_looking_or_playing_is_not_the_move_an_ab_is_about(tmp_path, monkeypatch):
    proj = P2._project(tmp_path)
    P2._scripted_turn(proj, [SET_GAIN, ("daw_command", {"command": "transport.play", "args": {}}),
                             ("daw_command", {"command": "view.toggle_mixer", "args": {}}),
                             ("ask_preference", {"about": "the bass level"})], monkeypatch)
    assert _pending(proj)["move"]["command"] == "track.set_gain"


def test_the_last_move_of_a_run_the_turn_started_is_the_move(tmp_path, monkeypatch):
    """A run or an Auto Mix the turn starts lands its own proposals: they are moves of the
    turn, in the order they landed, and come after a command made before them."""
    proj = P2._project(tmp_path)
    landed: list[str] = []

    def fake_run(project_dir, brief, **kw):
        landed.append(P2._apply(proj, P2._gain_req("vocals", -1.0)))
        landed.append(P2._apply(proj, P2._gain_req("bass", -1.0)))
        return {"passes": 1, "stop_reason": "targets met", "cost_usd": 0.0}
    monkeypatch.setattr(B, "run_agent_pass", fake_run)
    P2._scripted_turn(proj, [SET_GAIN, ("start_run", {"brief": "balance it"}),
                             ("ask_preference", {"about": "the balance"})], monkeypatch)
    assert _pending(proj)["move"] == {"proposal_id": landed[-1]}

    def fake_auto_mix(project_dir, target, **kw):
        from mix_engine.session import Session
        Session(proj).advance_pass()
        landed.append(P2._apply(proj, P2._gain_req("vocals", -0.5)))
        return 1
    monkeypatch.setattr(B, "auto_mix_pass", fake_auto_mix)
    P2._scripted_turn(proj, [SET_GAIN, ("auto_mix", {"target": "vocal-forward"}),
                             ("ask_preference", {"about": "the vocal"})], monkeypatch)
    assert _pending(proj)["move"] == {"proposal_id": landed[-1]}


def test_an_ab_with_no_move_behind_it_is_refused(tmp_path, monkeypatch):
    proj = P2._project(tmp_path)
    session, _, _ = P2._scripted_turn(
        proj, [("ask_preference", {"about": "the vocal"})], monkeypatch)
    assert session.log[0][2] is True, "an A/B about nothing was offered"
    assert "no move" in session.log[0][1]
    assert not (proj / "runs" / B.PENDING_PREFERENCE_FILE).exists()
    # ...and a name that is not a move of this turn says what the moves were
    session, _, _ = P2._scripted_turn(
        proj, [SET_GAIN, ("ask_preference", {"about": "x", "move": "track.set_pan"})], monkeypatch)
    assert session.log[-1][2] is True and "track.set_gain" in session.log[-1][1]
    assert not (proj / "runs" / B.PENDING_PREFERENCE_FILE).exists()


def test_an_offer_an_older_build_parked_is_one_move_too(tmp_path):
    """The file's old shape was the turn's floor and every command: a click across an update
    answers for its newest move, inside the ceiling the offer recorded."""
    proj = P2._project(tmp_path)
    floor = B.max_applied_seq(proj)
    first = P2._apply(proj, P2._gain_req("vocals", -1.0))
    asked = P2._apply(proj, P2._gain_req("bass", -1.0))
    ceil = B.max_applied_seq(proj)
    P2._apply(proj, P2._gain_req("vocals", -0.5))          # landed after the offer
    (proj / "runs").mkdir(exist_ok=True)
    (proj / "runs" / B.PENDING_PREFERENCE_FILE).write_text(json.dumps(
        {"ts": B._now_iso(), "about": "the bass", "floor_seq": floor, "ceil_seq": ceil,
         "commands": ["track.set_pan", "track.set_gain"]}))
    assert B.pending_preference_proposal(proj) == asked
    out = B.record_preference_answer(proj, chatmod.PREFERENCE_OPTIONS[0])
    assert out["proposal_ids"] == [asked] and out["commands"] == [] and first not in str(out)


def test_a_refused_ab_shows_no_card(monkeypatch):
    """The feed's two buttons must not appear for an offer the brain refused."""
    session = SENSES._session(monkeypatch)

    class _PrefSub:
        async def chat_turn(self, *, system, prompt, tool_defs, execute, on_delta, on_tool,
                            max_turns, effort=None, should_stop=None, max_tokens=None, timer=None):
            self.outcome = await execute("ask_preference", {"about": "the vocal reverb"})
            return "done", {}
    session._sub = _PrefSub()
    events = []

    async def emit(e):
        events.append(e)

    async def refuse(about, move):
        raise ValueError("no move this turn")

    anyio.run(lambda: session.turn("x", tools=SENSES._tools(ask_preference=refuse),
                                   context="CTX", emit=emit))
    assert not [e for e in events if e["type"] == "chat_question"], "the card was shown anyway"

    async def accept(about, move):                 # control: an accepted offer shows it
        return None
    session2 = SENSES._session(monkeypatch)
    session2._sub = _PrefSub()
    anyio.run(lambda: session2.turn("x", tools=SENSES._tools(ask_preference=accept),
                                    context="CTX", emit=emit))
    assert [e for e in events if e["type"] == "chat_question" and e.get("preference")]


def test_the_tool_says_one_move():
    props = chatmod._ASK_PREFERENCE_TOOL_DEF["input_schema"]["properties"]
    assert "move" in props and chatmod._ASK_PREFERENCE_TOOL_DEF["input_schema"]["required"] == ["about"]
    text = chatmod._ASK_PREFERENCE_TOOL_DEF["description"]
    assert "ONE move" in text and "last move" in text


def test_the_other_moves_of_an_ab_turn_are_still_kept(tmp_path, monkeypatch):
    """An A/B used to stand for the whole turn, so its click dropped every move of the turn
    from the passive `kept` signal. It is about one move now: that one has its own verdict,
    the others are left standing like any other."""
    proj = P2._project(tmp_path)
    monkeypatch.setenv("MIX_AGENT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(S.Brain, "run", lambda self, *a, **k: {"chat": "ok"})
    brain = S.Brain(proj)
    brain.begin_agent_turn()
    before = B.max_applied_seq(proj)
    kept_one = P2._apply(proj, P2._gain_req("vocals", -1.0))
    asked = P2._apply(proj, P2._gain_req("bass", -1.0))
    brain._close_gesture(before)
    B.offer_preference(proj, about="the bass level", move={"proposal_id": asked})
    brain.chat(chatmod.PREFERENCE_OPTIONS[1], use_llm=False)
    brain.begin_agent_turn()
    brain._close_gesture(B.max_applied_seq(proj))
    rows = [(d["decision"], d["proposal_id"]) for d in load_decisions(proj / "runs")]
    assert ("kept", asked) not in rows, "the A/B'd move was recorded twice (05·F11)"
    assert ("kept", kept_one) in rows, "a move left standing beside an A/B was never kept"


# ====================================================================================
# E6·R7 - every chat turn recomputed the observations from the raw stems
# ====================================================================================

def _count_masking(monkeypatch) -> list[int]:
    """How often a stem's masking table was built: once a stem per computation since Phase 55
    (it was `analyze_masking` per ordered pair, which rebuilt both stems' tables each time)."""
    calls = [0]
    real = analysis.masking_table

    def counting(*a, **k):
        calls[0] += 1
        return real(*a, **k)
    monkeypatch.setattr(analysis, "masking_table", counting)
    return calls


def test_observations_are_computed_once_for_a_set_of_stem_files(demo_project, monkeypatch):
    from mix_engine.session import Session
    Session(demo_project)
    calls = _count_masking(monkeypatch)
    first = B.session_observations(demo_project)
    cost = calls[0]
    assert cost >= 2, "vacuity: the first call must analyse"
    second = B.session_observations(demo_project)
    assert calls[0] == cost, "a second call re-read and re-analysed every stem"
    assert second == first
    second.append({"text": "poison"})
    if first:
        first[0]["text"] = "poison"
    assert all("poison" not in str(o) for o in B.session_observations(demo_project)), \
        "a caller's edit reached the kept list"
    assert B.session_observations(demo_project, cap=1) == B.session_observations(demo_project)[:1]


def test_a_changed_stem_file_or_fader_is_measured_again(demo_project, monkeypatch):
    """Controls for the key: the files, and the session gains the resonance rule sums with.

    Each half on its own since Phase 54: a moved fader re-measures the ONE rule that sums the
    stems at their faders, not the masking of every pair of files (which no fader changes) -
    that was up to 16 s in front of the next message after any level move."""
    from mix_engine import analysis
    from mix_engine.session import Session
    Session(demo_project)
    calls = _count_masking(monkeypatch)
    summed = [0]
    real = analysis._resonances

    def counting(*a, **k):
        summed[0] += 1
        return real(*a, **k)
    monkeypatch.setattr(analysis, "_resonances", counting)
    B.session_observations(demo_project)
    cost = calls[0]
    assert summed[0] == 1
    path = demo_project / "stem_a.wav"
    audio, sr = sf.read(str(path), dtype="float32")
    sf.write(str(path), audio[: len(audio) - 4800] * 0.5, sr)
    B.session_observations(demo_project)
    assert calls[0] == 2 * cost, "a rewritten stem was answered from the old measurement"
    assert summed[0] == 2
    state = json.loads((demo_project / "session.json").read_text())
    next(iter(state["stems"].values()))["gain_db"] = -9.0
    (demo_project / "session.json").write_text(json.dumps(state))
    B.session_observations(demo_project)
    assert summed[0] == 3, "a moved fader was answered from the old mix"
    assert calls[0] == 2 * cost, "a moved fader re-measured the masking of files it did not change"
    B.session_observations(demo_project)
    assert calls[0] == 2 * cost and summed[0] == 3


# ====================================================================================
# E6·R8 - the mid-run answer promised to act "once this pass finishes"; nothing acts
# ====================================================================================

def test_the_mid_run_answer_promises_nothing(monkeypatch):
    context = _midrun_context(monkeypatch, _snapshot(audio=1, midi=0),
                              message="stop - leave the drums alone")
    head = context.split("\n\n")[0]
    assert "once this pass finishes" not in head, "nothing acts when the pass finishes"
    assert "NOT queued" in head and "ask again" in head
    assert "Stop button" in head, "a typed 'stop' does not stop a run; the button does"
    assert "NO TOOLS" in head and "never claim to have made one" in head      # the rule before it
    assert "never promise" in head


def test_the_replayed_exchange_says_the_same(tmp_path):
    """Control: the next real turn is told nothing from that message was applied."""
    proj = tmp_path
    key = B._session_key(proj, "")
    B._MIDRUN_EXCHANGES[key] = [("make the vocal louder", "not yet - ask again when it ends")]

    class Session:
        history: list = []
    B._replay_midrun(Session, proj)
    assert "nothing from this message has been applied" in Session.history[0]["content"]


# ====================================================================================
# S4·R30 - the key picker's transpose was applied outside any gesture
# ====================================================================================

def _brain_with_a_key(proj: Path, monkeypatch) -> S.Brain:
    monkeypatch.setenv("MIX_AGENT_DATA_DIR", str(proj.parent / "data"))
    real = B.session_readout

    def readout(project_dir, cache=None):
        return {**real(project_dir, cache), "key": {"tonic": "A", "mode": "min",
                                                     "confidence": 0.9}}
    monkeypatch.setattr(B, "session_readout", readout)
    return S.Brain(proj)


def _agent_gesture(brain: S.Brain, proj: Path, stem: str, db: float, *, also=None) -> str:
    from mix_engine.session import Session
    Session(proj).advance_pass()          # a gesture is a pass: three proposals is its limit
    brain.begin_agent_turn()
    before = B.max_applied_seq(proj)
    pid = P2._apply(proj, P2._gain_req(stem, db))
    if also is not None:
        also()
    brain._close_gesture(before)
    return pid


def _mix(proj: Path) -> dict:
    state = P2._state(proj)
    return {"vocals": state["stems"]["vocals"].get("gain_db", 0.0),
            "master": [e["type"] for e in state.get("master_chain", [])]}


def test_a_key_pick_is_a_gesture_of_its_own(tmp_path, monkeypatch):
    proj = P2._project(tmp_path)
    brain = _brain_with_a_key(proj, monkeypatch)
    _agent_gesture(brain, proj, "vocals", -3.0)
    out = brain.transpose("C")
    assert out["ok"] is True and _mix(proj) == {"vocals": -3.0, "master": ["transpose"]}
    assert len(brain._run_marks) == 2, "the key pick landed above the agent's mark, with none of its own"
    assert out.get("gesture") == brain._run_gestures[-1], "the DAW's undo cannot name it"

    undone = brain.undo()
    assert _mix(proj) == {"vocals": -3.0, "master": []}, \
        f"one Undo Agent Move took the agent's move with the key pick: {undone}"
    brain.undo()
    assert _mix(proj) == {"vocals": 0.0, "master": []}, "control: the next undo is the agent's move"


def test_a_key_pick_that_changes_nothing_leaves_no_mark(tmp_path, monkeypatch):
    """Control: 'already in A minor' is not a gesture, and Undo must not find one."""
    proj = P2._project(tmp_path)
    brain = _brain_with_a_key(proj, monkeypatch)
    _agent_gesture(brain, proj, "vocals", -3.0)
    out = brain.transpose("A")
    assert out["ok"] is False and "gesture" not in out
    assert len(brain._run_marks) == 1
    brain.undo()
    assert _mix(proj)["vocals"] == 0.0


def test_a_transpose_the_agent_queued_stays_inside_its_turn(tmp_path, monkeypatch):
    """`agent.transpose` from inside a turn is queued behind it and arrives at this same door.
    It belongs to the turn that asked for it: one turn, one undo."""
    proj = P2._project(tmp_path)
    brain = _brain_with_a_key(proj, monkeypatch)
    brain.daw_invoke = lambda command, args=None, *a, **k: (
        {"ok": True, "queued": True, "note": "runs as the turn ends"}
        if command == "agent.transpose" else {"ok": True})
    invoke = brain.counting_invoke()
    _agent_gesture(brain, proj, "vocals", -3.0,
                   also=lambda: invoke("agent.transpose", {"tonic": "C"}))
    assert len(brain._run_marks) == 1
    out = brain.transpose("C")
    assert out["ok"] is True and len(brain._run_marks) == 1, \
        "the turn's own transpose became a second gesture"
    brain.undo()
    assert _mix(proj) == {"vocals": 0.0, "master": []}

    # a turn that ONLY queued the transpose leaves no mark; the transpose is then the gesture
    brain.begin_agent_turn()
    before = B.max_applied_seq(proj)
    invoke("agent.transpose", {"tonic": "C"})
    brain._close_gesture(before)
    assert len(brain._run_marks) == 0
    brain.transpose("C")
    assert len(brain._run_marks) == 1
    # ...and a key pick long after a queued transpose that never arrived is the user's own
    _agent_gesture(brain, proj, "vocals", -1.0,
                   also=lambda: invoke("agent.transpose", {"tonic": "D"}))
    brain.begin_agent_turn()
    brain._close_gesture(B.max_applied_seq(proj))      # another turn passed; the queue is stale
    marks = len(brain._run_marks)
    brain.transpose("D")
    assert len(brain._run_marks) == marks + 1
