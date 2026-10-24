"""Real-use QA, 2026-10-22 - the findings left open by the first batch, fixed (James: "fix all
open"). The record is docs/audits/user-qa-2026-10-22/QA-log.md.

Each test was seen red on the code that had the defect (the brain's halves had no door at all:
a scene name was refused, the app's measure had nowhere to go, every reading of a beat made in
the app was a reading of silence).
"""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import anyio
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from orchestrator import command_door, daw_bridge, turn_host

RATE = 48_000


# ---------------------------------------------------------------- Q9: a scene by its name

SCENES = [{"index": 0, "name": "Intro"}, {"index": 1, "name": "Drop  A"},
          {"index": 2, "name": ""}, {"index": 3, "name": "twice"}, {"index": 4, "name": "Twice"}]


def test_a_scene_named_by_its_name_is_sent_as_its_index():
    """Q9: `scene.launch {scene: "intro"}` was refused, and the engineer looked the number up
    in scenes[] itself. A name is matched case and spaces aside, and sent as the index."""
    args = {"scene": "intro"}
    sent, why = command_door.resolve_scene_names(args, SCENES)
    assert why is None and sent == {"scene": 0}
    assert args == {"scene": "intro"}, "the caller's dict is never written to"
    sent, why = command_door.resolve_scene_names({"scene": "drop a", "dest_scene": "INTRO"}, SCENES)
    assert why is None and sent == {"scene": 1, "dest_scene": 0}


def test_a_scene_number_is_the_apps_to_judge_and_a_wrong_name_is_refused_with_the_names():
    for value in (3, "3", "-1"):
        sent, why = command_door.resolve_scene_names({"scene": value}, SCENES)
        assert why is None and sent == {"scene": value}
    _, why = command_door.resolve_scene_names({"scene": "outro"}, SCENES)
    assert why and "no scene is called" in why and '0 "Intro"' in why
    _, why = command_door.resolve_scene_names({"scene": "twice"}, SCENES)
    assert why and "2 scenes" in why and "3, 4" in why


def test_both_doors_ask_the_scene_names_of_the_edit():
    """The live tool and an approved plan's step pass `checked_command`, and the eval's fake
    app asks the same function - a door that skipped it would refuse a name the other takes."""
    refused, sent = turn_host.checked_command(Path("/tmp"), "scene.launch", {"scene": "Intro"},
                                              {"scenes": SCENES})
    assert refused is None and sent == {"scene": 0}
    refused, _ = turn_host.checked_command(Path("/tmp"), "scene.launch", {"scene": "nope"},
                                           {"scenes": SCENES})
    assert refused and refused["ok"] is False
    src = Path(__file__).resolve().parents[2].joinpath("orchestrator", "operate_eval.py").read_text()
    assert "command_door.resolve_scene_names" in src


# ------------------------------------------- Q14: the limiter's target met on what the app plays

def _limited(project: Path, target: float = -14.0):
    from mix_engine.session import Session
    session = Session(project)
    session._state["master_chain"] = [{"type": "limiter", "stem_id": "master",
                                       "ceiling_dbtp": -1.0, "target_lufs_i": target}]
    session._persist()
    return session


def test_the_apps_measure_of_the_mix_is_a_fact_and_is_kept(demo_project: Path):
    """The app sends what it measured at the master limiter's input as
    `/api/set_master_loudness`: a fact report at both ends (it starts no measurement), stored
    in session.json, not part of the render fingerprint."""
    from mix_engine.session import Session

    url = "/api/set_master_loudness?pre_limiter_lufs_i=-21.5"
    assert daw_bridge.coedit_is_fact_report(url)
    app = Path(__file__).resolve().parents[2].joinpath("daw", "Source", "App.cpp").read_text()
    assert "set_master_loudness" in app[app.index("static bool coeditIsFactReport"):][:2500]
    session = Session(demo_project)
    before = session.render_fingerprint()
    daw_bridge._apply_one_coedit(session, url)
    assert session._state["native_loudness"] == {"pre_limiter_lufs_i": -21.5}
    assert session.render_fingerprint() == before
    again = Session(demo_project)
    assert again._state["native_loudness"]["pre_limiter_lufs_i"] == -21.5


def test_the_makeup_comes_from_the_apps_measure_when_the_brain_cannot_hear_the_mix(demo_project: Path):
    """Q14: on a beat made in the app the brain's render held nothing (or only the audio
    tracks), so the limiter's make-up was worked out on silence and a -14 target did nothing."""
    from mix_engine.session import Session

    session = _limited(demo_project)
    own = -30.0
    assert daw_bridge.measured_pre_limiter(session, own) == own, "no measure: the render's own"
    session.set_native_loudness(-21.0)
    key = str(demo_project.resolve())
    try:
        daw_bridge.LATEST_EDIT.pop(key, None)
        assert daw_bridge.measured_pre_limiter(session, own) == own, \
            "stems the render hears whole keep the render's own figure"
        daw_bridge.LATEST_EDIT[key] = {"tracks": [
            {"name": "keys", "track_kind": "midi", "clips": [{"start_s": 0.0, "length_s": 4.0}]}]}
        assert daw_bridge.measured_pre_limiter(session, own) == -21.0, \
            "an instrument track the render cannot hear: the app's figure"
    finally:
        daw_bridge.LATEST_EDIT.pop(key, None)
    empty = demo_project.parent / "beat_only"
    empty.mkdir()
    bare = Session(empty)
    bare.set_native_loudness(-19.0)
    assert daw_bridge.measured_pre_limiter(bare, None) == -19.0, "no stems at all: the app's figure"


# ------------------------------- Q13: a reading of the mix made on the app's own render of it

def _tone(seconds: float = 3.0, db: float = -20.0) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    x = (10 ** (db / 20.0)) * np.sin(2 * math.pi * 997.0 * t)
    return np.stack([x, x], axis=1)


def _lufs(audio: np.ndarray, rate: int = RATE) -> float:
    return float(pyln.Meter(rate).integrated_loudness(audio))


def test_a_render_takes_the_apps_mix_and_runs_the_limiter_on_it(demo_project: Path):
    """`dsp.native_mix`: inside the block the master IS the app's mix, put through the master's
    last limiter (its make-up reaching the target from the app's figure), and nothing of it is
    kept for a later render of the same state."""
    from mix_engine import dsp

    session = _limited(demo_project).probe_copy()
    own, rate = dsp.render_master(session)
    native = _tone()
    with dsp.native_mix(demo_project, native, RATE):
        assert dsp.has_native_mix(session)
        mix, got_rate = dsp.render_master(session)
        kept = dsp.kept_with_master(session, "probe", lambda: "of the app's mix")
    assert got_rate == RATE and mix.shape[0] == native.shape[0]
    assert abs(_lufs(mix) - -14.0) < 1.0, "the limiter reached its target on the app's mix"
    assert kept == "of the app's mix"
    assert not dsp.has_native_mix(session)
    after, _ = dsp.render_master(session)
    assert after.shape == own.shape and np.allclose(after, own), "outside the block: the brain's own"


def test_the_level_readout_of_a_beat_with_no_stems_reads_the_apps_mix(tmp_path: Path):
    from mix_engine import dsp

    project = tmp_path / "beat"
    project.mkdir()
    assert daw_bridge.level_readout(project) == {"silent": True}
    native = _tone(db=-12.0)
    with dsp.native_mix(project, native, RATE):
        got = daw_bridge.level_readout(project)
    assert got.get("silent") is not True and abs(got["lufs_i"] - _lufs(native)) < 0.5


def _turn(project: Path, edit: dict, calls: list, file: Path) -> turn_host.LiveTurn:
    def daw_invoke(command, args):
        calls.append(command)
        sf.write(str(file), _tone(db=-12.0), RATE, subtype="FLOAT")
        return {"ok": True, "file": str(file), "pre_limiter_lufs_i": -15.0}

    turn = turn_host.LiveTurn.__new__(turn_host.LiveTurn)
    turn.project_dir, turn.recorder, turn.daw_invoke = project, None, daw_invoke
    turn.edit_state, turn.mirror_coedits, turn.native = edit, None, None
    turn.session = SimpleNamespace(timing=None)
    turn.cache = {}
    turn.owed = {"commands": [], "before": None}
    turn.levels = {"last": {}, "exact": False}
    turn.invoked, turn.turn_moves = [], []
    return turn


MIDI_EDIT = {"tracks": [{"name": "keys", "track_kind": "midi",
                         "clips": [{"start_s": 0.0, "length_s": 3.0}]}]}


def test_the_render_tool_measures_the_apps_mix_and_says_so(demo_project: Path, tmp_path: Path):
    """Q13: on a project with an instrument track `render` said `not_heard` and gave the
    numbers of the stems alone; the engineer then set levels by guess. It now asks the app for
    its render of the mix once, measures that, and says what it was measured on."""
    calls: list = []
    turn = _turn(demo_project, MIDI_EDIT, calls, tmp_path / "mix.wav")
    got = json.loads(anyio.run(turn.do_render))
    assert calls == ["mix.measure"]
    assert got.get("measured_on") == daw_bridge.MEASURED_ON_APP and "not_heard" not in got
    assert abs(got["lufs_i"] - _lufs(_tone(db=-12.0))) < 0.5
    anyio.run(turn.do_render)
    assert calls == ["mix.measure"], "kept until something moves the mix"
    turn._note_command("track.set_gain")
    anyio.run(turn.do_render)
    assert calls == ["mix.measure"] * 2, "a command may have moved what the app plays"


def test_a_project_the_brain_hears_whole_never_asks_the_app(demo_project: Path, tmp_path: Path):
    calls: list = []
    turn = _turn(demo_project, {"tracks": []}, calls, tmp_path / "mix.wav")
    got = json.loads(anyio.run(turn.do_render))
    assert calls == [] and "measured_on" not in got


def test_a_reading_of_the_mix_through_the_engine_is_made_on_the_apps_mix(demo_project: Path,
                                                                          tmp_path: Path):
    """The read tools that measure the master (`daw_bridge.reads_the_mix`) take the same
    substitution; every other engine call is untouched."""
    from mix_engine import dsp

    seen: list = []

    class Engine:
        async def call_tool(self, name, args):
            # the engine's session (the app's mix stands in for the state it was measured at:
            # the pre-merge review of 2026-10-23 - a stand-in with no state has no fingerprint)
            from mix_engine.session import Session
            seen.append((name, dsp.has_native_mix(Session(demo_project, read_only=True))))
            return SimpleNamespace(isError=False, structuredContent={"result": {"lufs_i": -10.0}},
                                   content=[])

    calls: list = []
    turn = _turn(demo_project, MIDI_EDIT, calls, tmp_path / "mix.wav")
    turn.engine = Engine()
    got = anyio.run(turn.engine_call, "analyze_session", {})
    assert seen[-1] == ("analyze_session", True)
    assert got.get("measured_on") == daw_bridge.MEASURED_ON_APP and "not_heard" not in got
    anyio.run(turn.engine_call, "analyze_stem", {"stem_id": "stem_a"})
    assert seen[-1] == ("analyze_stem", False)
    got = anyio.run(turn.engine_call, "describe_region",
                    {"stem_id": "master", "start_s": 0.0, "end_s": 1.0, "pre_master": True})
    assert seen[-1][1] is True and "pre_master_is" in got
    assert calls == ["mix.measure"]


# ----------------------------------------- Q33: a looping clip is not its file to the brain

def test_a_looping_audio_clip_is_named_as_audio_the_render_holds_otherwise():
    """The brain holds one plain clip as a window of its file; a LOOPING clip repeats a stretch
    of it, which no window says. The snapshot's audio clip row carries `loop`, and the render's
    blind spots name the track (`arranged_audio`)."""
    plain = {"name": "drums", "track_kind": "audio",
             "clips": [{"start_s": 0.0, "offset_s": 0.0, "length_s": 16.0, "source": "drums.wav"}]}
    looped = {**plain, "clips": [{**plain["clips"][0], "loop": True}]}
    assert not daw_bridge.render_blind_spots({"tracks": [plain]})
    blind = daw_bridge.render_blind_spots({"tracks": [looped]})
    assert [b["what"] for b in blind] == ["arranged_audio"] and blind[0]["tracks"] == ["drums"]
    snap = Path(__file__).resolve().parents[2].joinpath("daw", "Source", "EditSnapshot.h").read_text()
    assert 'setProperty ("loop"' in snap


def test_the_temporary_old_path_switch_is_gone():
    """Every leg of this batch was run red on the old code through one temporary switch
    (`qaOld`, read from an environment variable), so a single build showed both. It is deleted
    before the commit: a product build must not carry a way back to a defect."""
    src = Path(__file__).resolve().parents[2] / "daw" / "Source"
    assert not (src / "QaOldPaths.h").exists()
    hits = [p.name for p in src.rglob("*") if p.suffix in (".h", ".cpp", ".mm")
            and ("qaOld (" in p.read_text(errors="ignore") or "QaOldPaths" in p.read_text(errors="ignore"))]
    assert hits == []


# ------------------------------------- the perception note: a whole song is not a kick

def _synth(kind: str, seconds: float = 12.0) -> np.ndarray:
    rng = np.random.default_rng(7)
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    beat = 0.5
    phase = (t % beat) / beat
    out = np.zeros(n)
    if kind in ("song", "kick_loop", "drum_loop"):
        tl = phase * beat                                        # a kick on every beat: a falling
        sweep = 50.0 * tl + 60.0 * 0.02 * (1.0 - np.exp(-tl / 0.02))   # sine, its phase from 0 each hit
        out += 0.9 * np.exp(-tl / 0.08) * np.sin(2 * math.pi * sweep)
    if kind == "song":
        out += 0.5 * np.sin(2 * math.pi * 55.0 * t)              # a held bass under it
        for hz in (220.0, 277.2, 329.6, 440.0, 1320.0, 2640.0):  # a chord, and a voice's partials
            out += 0.1 * np.sin(2 * math.pi * hz * t + hz)
        out += 0.06 * rng.standard_normal(n)                     # the air of a finished master
    if kind == "drum_loop":
        snare = ((t + beat) % (2 * beat)) / (2 * beat)
        out += 0.5 * np.exp(-snare * 2 * beat / 0.03) * rng.standard_normal(n)
        hats = (t % (beat / 2)) / (beat / 2)
        hp = np.diff(rng.standard_normal(n + 1))
        out += 0.15 * np.exp(-hats * beat / 2 / 0.01) * hp
    return np.stack([out, out], axis=1) * 0.5


def test_a_whole_song_is_tagged_a_full_mix_and_a_drum_part_is_still_a_kick():
    """A finished hip-hop song puts most of its power under 250 Hz, and the tagger read it as a
    `kick` - Auto Mix then put the whole song in its drums group. A song has a soft attack and
    mids and top a kick or a bass never has; a drum part keeps its sharp attack."""
    from mix_engine import analysis as A

    def tag(kind):
        return A.classify_stem(A.analyze_stem(_synth(kind), RATE, "track", []))

    assert tag("song") == "full_mix"
    assert tag("kick_loop") == "kick"
    assert tag("drum_loop") == "kick"
