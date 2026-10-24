"""Gates for batch T of the reader P2s (full audit 2026-10-02): both sides of the wire - the
Python halves. The C++ halves are console legs (the daw skill lists them); the table the two
limiters share is pinned in test_tp_detector_parity.py.

Each test drives the door the product uses and carries its control. Every one was run on the
unfixed tree first. The record is docs/audits/full-2026-10-02/readers/P2-fixes.md (T).
"""
from __future__ import annotations

import json
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from mix_engine import analysis, dsp, reference_monitor

REPO = Path(__file__).resolve().parents[2]
SR = 48000


# ====================================================================================
# E4·R7 - the offline limiter read peaks through the cubic the native replaced
# ====================================================================================

def _square(freq_hz: float, level: float = 0.9) -> np.ndarray:
    t = np.arange(SR) / SR
    x = np.sign(np.sin(2.0 * np.pi * freq_hz * t + 0.3)) * level
    return np.stack([x, x], axis=1)


def _drum_bursts(level: float = 0.8) -> np.ndarray:
    rng = np.random.default_rng(7)
    x = np.zeros(SR * 2)
    for k in range(0, len(x), SR // 4):
        m = min(2000, len(x) - k)
        x[k:k + m] += rng.standard_normal(m) * np.exp(-np.arange(m) / 300.0)
    return np.stack([x, x], axis=1) * level


_HOT = {"a 12 kHz square": lambda: _square(12000.0), "a 7 kHz square": lambda: _square(7000.0),
        "drum bursts": _drum_bursts,
        "noise": lambda: np.random.default_rng(3).standard_normal((SR, 2)) * 0.5}


@pytest.mark.parametrize("name", sorted(_HOT))
@pytest.mark.parametrize("brickwall", [dsp._brickwall, reference_monitor.brickwall],
                         ids=["dsp", "reference_monitor"])
def test_the_offline_brickwall_holds_its_ceiling_as_true_peak(name, brickwall):
    """The contractual measurement of what the brickwall leaves is the ceiling, to a tenth of a
    decibel. With the cubic it was 1.2 to 2.0 dB over on these four."""
    out = brickwall(_HOT[name](), SR, {"ceiling_dbtp": -1.0})
    assert analysis._true_peak_dbtp(out) <= -1.0 + 0.1


def test_the_master_limiter_does_not_turn_the_whole_file_down():
    """...so the trim that guarantees the ceiling has nothing left to take: it was 1.5 dB off the
    whole master, which the DAW's own export never loses."""
    report: dict = {}
    dsp._master_limiter(_drum_bursts(), SR, {"ceiling_dbtp": -1.0}, report)
    assert report["true_peak_overshoot_db"] <= 0.1
    assert report["limiter_trim_db"] >= -0.1
    assert report["true_peak_margin_db"] >= 0.0          # control: the ceiling still holds


def test_a_signal_under_the_ceiling_is_untouched():
    """Control: the detector over-reads by little - a tone 3 dB under the ceiling comes back as
    it went in."""
    t = np.arange(SR // 2) / SR
    x = 0.6 * np.sin(2.0 * np.pi * 1000.0 * t)
    audio = np.stack([x, x], axis=1)
    assert np.array_equal(dsp._brickwall(audio, SR, {"ceiling_dbtp": -1.0}), audio)


# ====================================================================================
# S1·R12 / S3·R2 - moves the brain never heard died with the app
# ====================================================================================
# The user's fader and knob moves are live at once and QUEUED for the brain. A quit with the
# brain down, or still winding a cancelled turn up, dropped the queue: the saved edit held the
# moves, session.json did not, and the next open laid the old chains back over them. The DAW
# now leaves what it could not hand over in `session.owed.coedits.json` (dawsession::
# parkOwedCoedits); these are the brain's half - what an open does with that file.

from mix_engine.session import (DISCARD_MARKER_FILE, SAVED_COEDITS_FILE, SAVED_SESSION_FILE,
                                SESSION_FILE, Session)
from orchestrator import daw_bridge as B
from orchestrator import daw_server as S

GAIN = "/api/set_stem?stem=stem_a&gain_db=-7.00"
ADD_DELAY = "/api/add_device?stem=stem_a&device_type=return_delay"


def _owed_file(project: Path) -> Path:
    from mix_engine.session import OWED_COEDITS_FILE
    return project / OWED_COEDITS_FILE


def _daw_parks(project: Path, urls) -> None:
    """What `dawsession::parkOwedCoedits` leaves at teardown."""
    _owed_file(project).write_text(json.dumps(urls))


def _state(project: Path) -> dict:
    return json.loads((project / SESSION_FILE).read_text())


def _devices(project: Path, stem: str, kind: str) -> int:
    return sum(1 for e in _state(project)["stems"][stem]["chain"] if e.get("type") == kind)


def test_an_open_applies_the_moves_the_last_session_could_not_hand_over(demo_project):
    Session(demo_project)
    _daw_parks(demo_project, [GAIN])
    brain = S.Brain(demo_project)
    brain.open_session(daw=True)
    assert _state(demo_project)["stems"]["stem_a"]["gain_db"] == -7.0
    assert not _owed_file(demo_project).exists()
    # ...and the chains a boot reads next carry it: the fader holds
    chains = brain.boot_sync()["chains"]
    gains = [e for e in chains["stem_a"]["entries"] if e.get("type") == "gain"]
    assert gains and gains[0]["gain_db"] == -7.0


def test_owed_moves_are_applied_once(demo_project):
    """`add_device` is not idempotent: a second open, a boot_sync and an edit switch after the
    first must find nothing left to apply."""
    Session(demo_project)
    _daw_parks(demo_project, [ADD_DELAY])
    brain = S.Brain(demo_project)
    brain.open_session(daw=True)
    assert _devices(demo_project, "stem_a", "return_delay") == 1
    brain.open_session(daw=True)
    brain.boot_sync()
    assert _devices(demo_project, "stem_a", "return_delay") == 1


def test_boot_sync_applies_what_a_teardown_left_after_the_hello(demo_project):
    """A project reloaded in a running app: the old component's teardown writes the file after
    the new one's hello may already have run."""
    Session(demo_project)
    brain = S.Brain(demo_project)
    brain.open_session(daw=True)
    _daw_parks(demo_project, [GAIN])
    brain.boot_sync()
    assert _state(demo_project)["stems"]["stem_a"]["gain_db"] == -7.0
    assert not _owed_file(demo_project).exists()


def test_the_first_checkpoint_does_not_take_the_owed_moves(demo_project):
    """The checkpoint is the session as the last SAVE left it. Owed moves of a session that was
    killed are not saved work: a later "Don't Save" must be able to decline them."""
    Session(demo_project)
    _daw_parks(demo_project, [GAIN])
    S.Brain(demo_project).open_session(daw=True)
    saved = json.loads((demo_project / SAVED_SESSION_FILE).read_text())
    assert saved["stems"]["stem_a"].get("gain_db", 0.0) == 0.0
    assert _state(demo_project)["stems"]["stem_a"]["gain_db"] == -7.0     # control: they were applied


def test_declined_work_takes_its_owed_moves_with_it(demo_project):
    """"Discard" at the recovery prompt after a kill: the checkpoint comes back with the moves
    queued at the SAVE, and the moves made after it - the owed file - are part of what was
    declined."""
    Session(demo_project)
    shutil.copyfile(demo_project / SESSION_FILE, demo_project / SAVED_SESSION_FILE)
    (demo_project / SAVED_COEDITS_FILE).write_text(json.dumps(["/api/set_stem?stem=stem_b&gain_db=-3.00"]))
    _daw_parks(demo_project, ["/api/set_stem?stem=stem_b&gain_db=-3.00", GAIN])
    (demo_project / DISCARD_MARKER_FILE).write_text("declined\n")
    S.Brain(demo_project).open_session(daw=True)
    state = _state(demo_project)
    assert state["stems"]["stem_b"]["gain_db"] == -3.0           # queued at the save: kept
    assert state["stems"]["stem_a"].get("gain_db", 0.0) == 0.0   # made after it: declined
    assert not _owed_file(demo_project).exists()


def test_an_unreadable_owed_file_is_dropped_and_said(demo_project, capsys):
    Session(demo_project)
    _owed_file(demo_project).write_text("{not json")
    S.Brain(demo_project).open_session(daw=True)
    assert not _owed_file(demo_project).exists()
    assert "owed" in capsys.readouterr().err


def test_a_refused_owed_move_does_not_stop_the_rest(demo_project, capsys):
    Session(demo_project)
    _daw_parks(demo_project, ["/api/set_stem?stem=nobody&gain_db=-2.00", GAIN])
    S.Brain(demo_project).open_session(daw=True)
    assert _state(demo_project)["stems"]["stem_a"]["gain_db"] == -7.0
    assert "nobody" in capsys.readouterr().err


def test_with_no_owed_file_an_open_changes_nothing(demo_project):
    """Control."""
    Session(demo_project)
    before = (demo_project / SESSION_FILE).read_text()
    S.Brain(demo_project).open_session(daw=True)
    assert (demo_project / SESSION_FILE).read_text() == before


# ====================================================================================
# S4·R14 / C7·R6 - ⌥⌘Z took back another turn's timeline edit, and left a turn's levels
# ====================================================================================

def _turn(brain, replies: dict, floor: int | None = None):
    """One agent turn: each (command, args) is invoked through the counting wrapper the chat
    uses, answered by `replies[command]`, and the reply's co-edits are mirrored as chat does."""
    floor = B.max_applied_seq(brain.project_dir) if floor is None else floor
    seen = []

    def daw(command, args=None, *a, **kw):
        seen.append((command, dict(args or {})))
        return dict(replies.get(command, {"ok": True}))

    brain.daw_invoke = daw
    turn = brain.begin_agent_turn()
    invoke = brain.counting_invoke()
    for command in [c for c in replies if c != "edit.undo"]:
        res = invoke(command, {})
        if isinstance(res, dict) and res.get("coedits"):
            brain._mirror_coedits(list(res["coedits"]))
    brain._close_gesture(floor)
    return turn, seen


def test_alt_cmd_z_names_the_turn_it_takes_back(demo_project):
    brain = S.Brain(demo_project)
    first, _ = _turn(brain, {"clip.split": {"ok": True}})
    second, seen = _turn(brain, {"clip.move": {"ok": True}})
    assert first != second
    brain.undo()
    undo = [args for cmd, args in seen if cmd == "edit.undo"]
    assert undo == [{"expect_agent": True, "turn": second}]
    brain.undo()
    assert [args for cmd, args in seen if cmd == "edit.undo"][-1]["turn"] == first


def test_a_turn_that_left_nothing_on_the_timeline_owes_the_timeline_nothing(demo_project):
    """The turn pressed play. The DAW says the turn left no set; nothing is pending, and the
    undo is not reported as half done."""
    brain = S.Brain(demo_project)
    _turn(brain, {"transport.play_pause": {"ok": True},
                  "edit.undo": {"ok": True, "timeline": "none"}})
    out = brain.undo()
    assert out["daw_ops_undone"] == 0 and out["daw_ops_pending"] == 0
    assert "timeline_left" not in out["undone"]


def test_a_timeline_refusal_is_said_with_its_reason(demo_project):
    """Control for the one above: a real refusal (the user edited since) IS pending, and the
    reply carries why."""
    brain = S.Brain(demo_project)
    _turn(brain, {"clip.split": {"ok": True},
                  "edit.undo": {"ok": False, "error": "edited since"}})
    out = brain.undo()
    assert out["daw_ops_pending"] == 1
    assert out["undone"]["timeline_left"] == "edited since"


def test_a_refused_command_is_not_a_gesture(demo_project):
    brain = S.Brain(demo_project)
    before = len(brain._run_marks)
    _turn(brain, {"clip.delete": {"ok": False, "error": "no such clip"}})
    assert len(brain._run_marks) == before
    _turn(brain, {"clip.delete": {"ok": True}})                    # control: a real one is
    assert len(brain._run_marks) == before + 1


def test_alt_cmd_z_takes_back_the_levels_the_turn_set(demo_project):
    """`track.set_gain` is a co-edit, not a proposal and not a timeline step: nothing took it
    back. The user's own move riding the same reply is not the turn's and stays."""
    Session(demo_project)
    B.apply_coedits(demo_project, ["/api/set_stem?stem=stem_a&gain_db=-2.00"])
    brain = S.Brain(demo_project)
    users = "/api/set_stem?stem=stem_b&gain_db=-4.00"
    agents = ["/api/set_stem?stem=stem_a&gain_db=-9.00", "/api/set_stem?stem=stem_a&pan=0.500",
              "/api/set_stem?stem=stem_a&muted=true"]
    _turn(brain, {"track.set_gain": {"ok": True, "coedits": [users, agents[0]],
                                     "coedits_own": [agents[0]]},
                  "track.set_pan": {"ok": True, "coedits": [agents[1]], "coedits_own": [agents[1]]},
                  "track.set_mute": {"ok": True, "coedits": [agents[2]], "coedits_own": [agents[2]]},
                  "edit.undo": {"ok": True, "timeline": "none"}})
    held = _state(demo_project)["stems"]
    assert (held["stem_a"]["gain_db"], held["stem_a"]["pan"], held["stem_a"]["muted"]) == (-9.0, 0.5, True)
    out = brain.undo()
    held = _state(demo_project)["stems"]
    assert held["stem_a"]["gain_db"] == -2.0          # what it was before the TURN, not the default
    assert held["stem_a"]["pan"] == 0.0 and held["stem_a"]["muted"] is False
    assert held["stem_b"]["gain_db"] == -4.0          # the user's: not the turn's to take back
    assert out["undone"]["levels"] == 3
    # ...and the reply that moves the DAW's controls carries them
    gains = [e for e in out["chains"]["stem_a"]["entries"] if e.get("type") == "gain"]
    assert gains and gains[0]["gain_db"] == -2.0


def test_the_first_touch_in_a_turn_is_what_comes_back(demo_project):
    """Two sets of one fader in a turn: the undo goes to before the first, not to the first."""
    Session(demo_project)
    brain = S.Brain(demo_project)
    a, b = "/api/set_stem?stem=stem_a&gain_db=-3.00", "/api/set_stem?stem=stem_a&gain_db=-6.00"
    brain.daw_invoke = lambda cmd, args=None, *x, **k: (
        {"ok": True, "timeline": "none"} if cmd == "edit.undo"
        else {"ok": True, "coedits": [args["u"]], "coedits_own": [args["u"]]})
    brain.begin_agent_turn()
    invoke = brain.counting_invoke()
    for url in (a, b):
        brain._mirror_coedits(list(invoke("track.set_gain", {"u": url})["coedits"]))
    brain._close_gesture(B.max_applied_seq(demo_project))
    assert _state(demo_project)["stems"]["stem_a"]["gain_db"] == -6.0
    brain.undo()
    assert _state(demo_project)["stems"]["stem_a"].get("gain_db", 0.0) == 0.0


def test_a_send_and_a_return_level_come_back_too(demo_project):
    session = Session(demo_project)
    rid = session.add_return("blank")["return_id"]
    session.set_send("stem_a", rid, -12.0)
    brain = S.Brain(demo_project)
    send = f"/api/set_send?source=stem_a&return={rid}&gain_db=-3.00"
    level = f"/api/set_return_level?return={rid}&gain_db=-5.00"
    _turn(brain, {"send.set": {"ok": True, "coedits": [send], "coedits_own": [send]},
                  "return.set_level": {"ok": True, "coedits": [level], "coedits_own": [level]},
                  "edit.undo": {"ok": True, "timeline": "none"}})
    state = _state(demo_project)
    assert state["stems"]["stem_a"]["sends"][rid] == -3.0 and state["returns"][rid]["gain_db"] == -5.0
    brain.undo()
    state = _state(demo_project)
    assert state["stems"]["stem_a"]["sends"][rid] == -12.0
    assert state["returns"][rid].get("gain_db", 0.0) == 0.0


def test_a_device_knob_and_an_engine_fact_are_not_levels(demo_project):
    """Controls: the timeline's own undo takes a device knob back (and says so in its reply); a
    report of the engine's facts is not a move at all."""
    Session(demo_project)
    urls = ["/api/device_param?stem=stem_a&device_type=comp&occurrence=0&param=threshold_db&value=-20",
            "/api/set_stem?stem=stem_a&delay_ms=12.0", "/api/set_stem?stem=stem_a&tag=vocals",
            "/api/set_stem?stem=nobody&gain_db=-3.0"]
    assert B.level_inverses(demo_project, urls) == []
    assert len(B.level_inverses(demo_project, ["/api/set_stem?stem=stem_a&gain_db=-3.0"])) == 1


# ====================================================================================
# S8·R3 / S8·R6 - the licence: a sentence nobody drew, a door Auto Mix walked round
# ====================================================================================

import test_api_key_license as LIC
from orchestrator import api_key, license_state

_isolated = LIC._isolated          # no network, no keychain, a scratch data dir (autouse there, named here)


@pytest.fixture()
def shipped(monkeypatch, tmp_path):
    """A shipped-shaped install: a stored key (so the gate applies), signed out."""
    monkeypatch.setattr(api_key, "validate_api_key", lambda key, client_factory=None: None)
    api_key.set_api_key(LIC.GOOD)
    assert license_state.gate_applies() is True


def _licence_cache(status: str, age_days: float) -> None:
    import time as _time
    path = Path(license_state._cache_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    import licence_signing   # the service signs its answer (2026-10-20): signed that long ago
    path.write_text(json.dumps(licence_signing.cache(
        {"status": status, "checked_at": _time.time() - age_days * 86400.0})))
    license_state.reset()


def test_a_licence_past_its_offline_grace_is_asked_to_sign_in_again_not_to_start_a_trial(shipped):
    """Signed out with a cache that says LICENSED, 31 days old: the person paid. They were told
    to "start your 14-day trial"."""
    _licence_cache("licensed", 31.0)
    state = license_state.state(force=True)
    assert state["gated"] is True and state["status"] != "licensed"
    assert state["message"] == license_state.MSG_RECONFIRM
    assert "trial" not in state["message"]


def test_someone_who_never_had_a_licence_is_still_offered_the_trial(shipped):
    """Controls: no cache at all, and a lapsed TRIAL cache."""
    assert license_state.state(force=True)["message"] == license_state.MSG_SIGN_IN
    _licence_cache("trial", 31.0)
    assert license_state.state(force=True)["message"] == license_state.MSG_SIGN_IN


def test_inside_the_grace_the_lapsed_sign_in_is_said_and_the_agent_still_runs(shipped):
    """Control: the sentence S8·R3 is about - `licensed`, with a message. The DAW draws it
    (console leg --licence-door)."""
    _licence_cache("licensed", 3.0)
    state = license_state.state(force=True)
    assert state["status"] == "licensed" and state["message"] == license_state.MSG_RESIGN_IN
    assert license_state.refusal() is None


def test_the_manual_quotes_the_new_sentence():
    text = " ".join((REPO / "docs" / "17-user-manual.md").read_text(encoding="utf-8").split())
    assert license_state.MSG_RECONFIRM in text
    assert license_state.MSG_RESIGN_IN.split(" to keep")[0] in text     # control: the reading works


def _auto_mix(proj: Path) -> list:
    return LIC._talk(proj, lambda token: (
        LIC._line({"id": 1, "method": "auth", "params": {"token": token}})
        + LIC._line({"id": 2, "method": "auto_mix", "params": {"brief": ""}})),
        until=LIC._said("run_done", "run_failed"))


def test_auto_mix_is_behind_the_licence(tmp_path, shipped):
    """Ruling 5 of Phase 53: the licence gates "Chat, runs, Auto Mix and /v1/knowledge". Auto
    Mix is deterministic, so the door keyed on `use_llm` never met it."""
    proj = LIC._project(tmp_path)
    before = (proj / "session.json").read_text() if (proj / "session.json").exists() else None
    out = _auto_mix(proj)
    ack = next(o for o in out if o.get("id") == 2)
    assert ack["result"]["status"] == "started", ack
    failed = [o for o in out if o.get("event") == "run_failed"]
    assert failed and failed[0]["error"] == license_state.MSG_SIGN_IN, out
    assert failed[0].get("licence") is True and failed[0].get("retryable") is False
    assert not [o for o in out if o.get("event") in ("run_started", "run_done", "automix_stage")]
    after = (proj / "session.json").read_text() if (proj / "session.json").exists() else None
    assert after == before                      # nothing was mixed


def test_auto_mix_runs_on_a_developer_transport(tmp_path, monkeypatch):
    """Control: no stored key, no cloud - a source checkout. The console legs and the suite
    run Auto Mix here, and must go on doing so."""
    assert license_state.gate_applies() is False
    out = _auto_mix(LIC._project(tmp_path))
    assert [o for o in out if o.get("event") == "run_started"], out
    assert not [o for o in out if o.get("licence")]


# ====================================================================================
# S1·R18 - a stems folder holding master.wav could not be opened at all
# ====================================================================================

def _folder_with(tmp_path: Path, *names: str) -> Path:
    import soundfile as sf
    proj = tmp_path / "stems"
    proj.mkdir()
    t = np.arange(4410) / 44100.0
    for name in names:
        sf.write(proj / name, (0.2 * np.sin(2 * np.pi * 220 * t)).astype("float32"), 44100)
    return proj


@pytest.mark.parametrize("trap", ["master.wav", "Master.wav", "MASTER.WAV"])
def test_a_folder_with_a_master_file_opens_with_that_file_left_out(tmp_path, trap):
    """It raised - so every Session of the project raised, the DAW opened the folder with no
    tracks, and its first save made that permanent. The reference mix is what such a file
    usually is."""
    proj = _folder_with(tmp_path, "vox.wav", "beat.wav", trap)
    session = Session(proj)
    assert sorted(session.stems) == ["beat", "vox"]
    assert session.left_out_wavs == [trap]
    assert (proj / trap).exists()                                   # left alone on disk
    again = Session(proj)                                           # ...and at every later open
    assert sorted(again.stems) == ["beat", "vox"] and again.left_out_wavs == [trap]


def test_a_folder_without_one_leaves_nothing_out(tmp_path):
    """Control."""
    session = Session(_folder_with(tmp_path, "vox.wav", "beat.wav"))
    assert sorted(session.stems) == ["beat", "vox"] and session.left_out_wavs == []


def test_a_master_file_an_old_session_already_holds_stays_its_stem(tmp_path):
    """A session written before the name was reserved in every case: its record keeps its file
    (the oracle's demo project is one - it pops the record itself)."""
    proj = _folder_with(tmp_path, "vox.wav", "Master.wav")
    Session(proj)
    state = json.loads((proj / "session.json").read_text())
    state["stems"]["Master"] = {"path": "Master.wav"}
    (proj / "session.json").write_text(json.dumps(state))
    session = Session(proj)
    assert "Master" in session.stems and session.left_out_wavs == []


def test_the_wild_corpus_expects_the_file_left_out_not_a_refusal():
    from orchestrator import wild_fixture
    spec = next(s for s in wild_fixture.PROJECTS if s.name == "master_trap")
    assert spec.expect_error is None and spec.left_out == ("master.wav",)


# ====================================================================================
# C4·R13 - MIDI controller data was write-only: a count, and no values
# ====================================================================================
# The snapshot now carries each controller's points (EditSnapshot.h; console leg --cc-points).
# The brain's half: a dense lane must not be what pushes a track out of the agent's picture,
# and what the budget takes away must be readable back.

from orchestrator import edit_state


def _cc_edit(points: int, tracks: int = 1) -> dict:
    def track(i: int) -> dict:
        return {"name": f"keys {i}", "clips": [{
            "index": 0, "kind": "midi", "start_s": 0.0, "length_s": 8.0,
            "notes": [{"pitch": 60, "start_beat": 0.0, "length_beats": 1.0, "velocity": 100}],
            "note_count": 1,
            "controllers": [{"type": 1, "name": "Mod Wheel", "count": points,
                             "points": [[round(k * 0.03125, 5), k % 128] for k in range(points)]}]}]}
    return {"tracks": [track(i) for i in range(tracks)]}


def test_controller_points_reach_the_agent_when_they_fit():
    out = json.loads(edit_state.compact_edit_state(_cc_edit(8), 4000))
    cc = out["tracks"][0]["clips"][0]["controllers"][0]
    assert cc["points"][3] == [0.09375, 3] and cc["count"] == 8
    assert "points_omitted" not in cc


def test_a_dense_controller_lane_is_thinned_before_any_track_goes():
    out = json.loads(edit_state.compact_edit_state(_cc_edit(256, tracks=6), 4000))
    assert len(out["tracks"]) == 6                                   # nothing was pushed out
    cc = out["tracks"][0]["clips"][0]["controllers"][0]
    assert "points" not in cc and cc["points_omitted"] == 256 and cc["count"] == 256
    assert "clip_notes" in out["omitted"]


def test_read_track_brings_the_points_back():
    """An omission marker needs a reader: the clip read returns the lane whole."""
    edit = _cc_edit(256, tracks=6)
    got = B.read_track_entry(edit, "keys 2", clip=0)
    assert got["ok"] and len(got["clip_entry"]["controllers"][0]["points"]) == 256
    whole = B.read_track_entry(_cc_edit(4000), "keys 0")
    assert whole["ok"] and "points_omitted" in json.dumps(whole["track"]) and "hint" in whole
    assert "points_omitted" in whole["hint"]


# ====================================================================================
# S2·R12 - a frozen track took edits that were exported and never heard
# ====================================================================================
# The DAW locks its own doors (console leg --frozen-lock) and reports WHICH frozen track holds
# each stem as an engine fact; one rule in the session refuses a proposal on it, whoever
# made the proposal.

from mix_engine.models import ProposalRequest

FROZEN = "/api/set_stem?stem=stem_a&frozen=stem_a"
THAWED = "/api/set_stem?stem=stem_a&frozen=%3Alive"


def _propose(project: Path, actions: list):
    return Session(project).propose(ProposalRequest(
        rationale="t", actions=actions,
        predicted_impact=[{"metric": "stem.lufs_s", "predicted_delta": -1.0, "unit": "LUFS"}]))


def test_a_proposal_on_a_frozen_track_is_refused_with_the_way_out(demo_project):
    Session(demo_project)
    assert B.apply_coedits(demo_project, [FROZEN])["failed"] == 0
    assert Session(demo_project).frozen_by("stem_a") == "stem_a"
    for action in ({"type": "gain", "stem_id": "stem_a", "delta_db": -1.0},
                   {"type": "comp", "stem_id": "stem_a", "threshold_db": -20.0, "ratio": 2.0,
                    "attack_ms": 10.0, "release_ms": 120.0},
                   {"type": "eq", "stem_id": "stem_a",
                    "band": {"shape": "peak", "freq_hz": 3000.0, "gain_db": -2.0, "q": 1.0}}):
        result = _propose(demo_project, [action])
        assert not result.accepted
        rules = [v.rule for v in result.violations]
        assert rules == ["frozen_track"], rules
        assert 'track.freeze {track: "stem_a", on: false}' in result.violations[0].message
    # controls: the live track beside it, and the master, are untouched by the rule
    assert _propose(demo_project, [{"type": "gain", "stem_id": "stem_b", "delta_db": -1.0}]).accepted


def test_unfreezing_opens_it_again(demo_project):
    Session(demo_project)
    B.apply_coedits(demo_project, [FROZEN])
    B.apply_coedits(demo_project, [THAWED])
    assert "frozen" not in _state(demo_project)["stems"]["stem_a"]
    assert _propose(demo_project, [{"type": "gain", "stem_id": "stem_a", "delta_db": -1.0}]).accepted


def test_a_frozen_group_holds_its_group_and_its_members(demo_project):
    session = Session(demo_project)
    session.route_stem("stem_a", "band")
    B.apply_coedits(demo_project, ["/api/set_stem?stem=stem_a&frozen=bus%3Aband"])
    session = Session(demo_project)
    assert session.frozen_by("band") == "bus:band" and session.frozen_by("stem_a") == "bus:band"
    for target in ("band", "stem_a"):
        result = _propose(demo_project, [{"type": "gain", "stem_id": target, "delta_db": -1.0}])
        assert [v.rule for v in result.violations] == ["frozen_track"], target
        assert 'track: "bus:band"' in result.violations[0].message


def test_a_send_from_a_frozen_track_is_still_the_agents(demo_project):
    """Control: a post-fader send is not behind the freeze (the manual says so)."""
    session = Session(demo_project)
    rid = session.add_return("blank")["return_id"]
    B.apply_coedits(demo_project, [FROZEN])
    result = _propose(demo_project, [{"type": "send", "stem_id": "stem_a",
                                      "return_id": rid, "gain_db": -12.0}])
    assert "frozen_track" not in [v.rule for v in result.violations]


def test_the_fact_is_one_the_app_can_compare(demo_project):
    """`noteEngineFacts` compares what the edit says with the reply's strip row."""
    Session(demo_project)
    B.apply_coedits(demo_project, [FROZEN])
    rows = B.strip_state(_state(demo_project))
    assert rows["stem_a"]["frozen"] == "stem_a" and rows["stem_b"]["frozen"] == ""


def test_the_agent_is_told_what_frozen_means():
    spots = B.render_blind_spots({"tracks": [
        {"name": "drums", "frozen": True},
        {"name": "hat", "frozen_by": "bus:kit"},
        {"name": "bass"}]})
    frozen = [s for s in spots if s["what"] == "frozen_tracks"]
    assert frozen and frozen[0]["tracks"] == ["drums", "hat (its group bus:kit)"]
    assert "track.freeze" in frozen[0]["to_change"]
    assert not [s for s in B.render_blind_spots({"tracks": [{"name": "bass"}]})
                if s["what"] == "frozen_tracks"]                      # control
    import inspect

    from orchestrator import agents, chat
    assert "frozen_track" in inspect.getsource(chat) and "frozen_track" in inspect.getsource(agents)


# ====================================================================================
# The lead from M6 - `library.import` to a new track registered its stem after the turn
# ====================================================================================

def test_a_library_item_imported_as_a_track_is_a_stem_before_the_tool_returns(demo_project, tmp_path):
    import anyio
    import soundfile as sf
    Session(demo_project)
    item = tmp_path / "library" / "shaker.wav"
    item.parent.mkdir()
    sf.write(item, (0.2 * np.sin(np.arange(4410) / 7.0)).astype("float32"), 44100)
    calls = []

    def daw(command, args=None, *a, **kw):
        calls.append((command, dict(args or {})))
        return {"ok": True, "track": "shaker", "registered": False, "file": str(item),
                "imported": "Shaker loop"}

    out = anyio.run(lambda: B._import_sample_step(
        demo_project, "", daw, library_args={"id": "123/45"}))
    assert calls == [("library.import", {"id": "123/45", "register": False})]
    assert out["ok"] and out["registered"] is True and out["stem_id"] == "shaker"
    assert "shaker" in Session(demo_project).stems            # ...inside the step, not after the turn
    assert (demo_project / "shaker.wav").exists()


def test_a_library_item_the_daw_registered_itself_is_not_registered_twice(demo_project):
    """Control: a reply that does not say `registered: false` (a pad fill, an older app)."""
    import anyio
    Session(demo_project)
    before = sorted(Session(demo_project).stems)
    out = anyio.run(lambda: B._import_sample_step(
        demo_project, "", lambda c, a=None, *x, **k: {"ok": True, "track": "pad"},
        library_args={"id": "1/2"}))
    assert out["ok"] and sorted(Session(demo_project).stems) == before


def test_the_chat_door_and_the_plan_door_both_take_the_library_step():
    # ONE implementation since Phase 55 (`turn_host.registering_command`), called by both doors
    import inspect
    from orchestrator import turn_host
    # the library item, and (Phase 55) a bounce's print: the app names the file for both
    source = inspect.getsource(turn_host.registering_command)
    assert source.count('library_args=dict(') == 2 and source.count('daw_command="track.bounce"') == 1
    assert "registering_command(" in inspect.getsource(turn_host.LiveTurn.do_daw_command)
    assert "registering_command(" in inspect.getsource(B.apply_plan)


# ====================================================================================
# E2·R5 - the ninth band-width device was stored and never played
# ====================================================================================

def test_the_ninth_band_width_is_refused(demo_project):
    session = Session(demo_project)
    for _ in range(8):
        session.add_stem_device("stem_a", "band_width")
    with pytest.raises(ValueError, match="already has 8 band_width"):
        session.add_stem_device("stem_a", "band_width")
    assert _devices(demo_project, "stem_a", "band_width") == 8
    session.add_stem_device("stem_b", "band_width")                 # control: per chain


def test_the_co_edit_door_says_the_refusal(demo_project):
    Session(demo_project)
    add = "/api/add_device?stem=stem_a&device_type=band_width"
    res = B.apply_coedits(demo_project, [add] * 9)
    assert res["applied"] == 8 and res["failed"] == 1 and "8 band_width" in res["errors"][0]


# ====================================================================================
# S3·R6 - a routing gesture held the brain's one line for a whole render of the song
# ====================================================================================

def _routing_project(tmp_path: Path) -> tuple[Path, str]:
    proj = LIC._project(tmp_path)
    return proj, Session(proj).add_return("blank")["return_id"]


def test_the_line_answers_other_requests_while_a_routing_render_runs(tmp_path, monkeypatch):
    """`remove_return` ends in a full render. Awaited inline, nothing else was read until it
    had finished - so the DAW timed out and called a change that HAPPENED a refusal."""
    import time as _time
    proj, rid = _routing_project(tmp_path)
    real = S.Brain.remove_return

    def slow(self, return_id):
        _time.sleep(0.8)                       # the render
        return real(self, return_id)

    monkeypatch.setattr(S.Brain, "remove_return", slow)
    out = LIC._talk(proj, lambda token: (
        LIC._line({"id": 1, "method": "auth", "params": {"token": token}})
        + LIC._line({"id": 2, "method": "remove_return", "params": {"id": rid}})
        + LIC._line({"id": 3, "method": "get_strip"})), until=LIC._replied(2, 3))
    order = [o["id"] for o in out if o.get("id") in (2, 3)]
    assert order == [3, 2], order              # the read came back first
    done = next(o for o in out if o.get("id") == 2)
    assert "result" in done and rid not in json.loads((proj / "session.json").read_text())["returns"]


def test_two_routing_changes_still_land_in_the_order_they_were_asked(tmp_path, monkeypatch):
    """Control for the tasks: two changes still run one after the other, in the order asked -
    the brain's own mutate lock is held for the whole of each (the second is the slow one's
    inverse)."""
    import time as _time
    proj, rid = _routing_project(tmp_path)
    real = S.Brain.set_return_prefader

    def slow_first(self, return_id, pre):
        if pre:
            _time.sleep(0.5)
        return real(self, return_id, pre)

    monkeypatch.setattr(S.Brain, "set_return_prefader", slow_first)
    out = LIC._talk(proj, lambda token: (
        LIC._line({"id": 1, "method": "auth", "params": {"token": token}})
        + LIC._line({"id": 2, "method": "set_return_prefader", "params": {"return": rid, "pre_fader": True}})
        + LIC._line({"id": 3, "method": "set_return_prefader", "params": {"return": rid, "pre_fader": False}})),
        until=LIC._replied(2, 3))
    assert [o["id"] for o in out if o.get("id") in (2, 3)] == [2, 3]
    assert json.loads((proj / "session.json").read_text())["returns"][rid].get("pre_fader", False) is False


def test_a_failing_routing_change_still_answers(tmp_path):
    proj = LIC._project(tmp_path)
    out = LIC._talk(proj, lambda token: (
        LIC._line({"id": 1, "method": "auth", "params": {"token": token}})
        + LIC._line({"id": 2, "method": "remove_return", "params": {"id": "nobody"}})),
        until=LIC._replied(2))
    reply = next(o for o in out if o.get("id") == 2)
    assert "error" in reply and "nobody" in reply["error"]


def test_adding_a_blank_return_does_not_render_the_song(demo_project, monkeypatch):
    """An empty return with nothing sent into it adds no sound: the cached readout stands."""
    brain = S.Brain(demo_project)
    brain.boot_sync()                                                 # the cache is warm
    renders = []
    real = dsp.render_master
    monkeypatch.setattr(dsp, "render_master", lambda *a, **k: (renders.append(1), real(*a, **k))[1])
    out = brain.add_return("blank")
    assert renders == [] and out["return_id"] and "chains" in out and "readout" in out
    brain.remove_return(out["return_id"])                             # control: this one does
    assert renders


# ====================================================================================
# E4·R5 - the export oracle's "approximation tier" never exercised its stem dynamics,
#         and claimed a peak comparison it did not make
# ====================================================================================
# The profiles themselves are measured on the real binary (`daw_export_oracle --profile all`);
# these are the parts that are pure Python.

from orchestrator import daw_export_oracle as ORACLE


def _oracle_session(tmp_path: Path):
    """Three stems: a quiet low one FIRST (the demo's bass, -28 dBFS), a hot burst, a mid one."""
    import soundfile as sf
    proj = tmp_path / "oracle"
    proj.mkdir()
    t = np.arange(SR * 2) / SR
    sf.write(proj / "bass.wav", (0.04 * np.sin(2 * np.pi * 60 * t)).astype("float32"), SR)
    sf.write(proj / "drums.wav", _drum_bursts(0.3)[:, 0].astype("float32"), SR)
    sf.write(proj / "keys.wav", (0.2 * np.sin(2 * np.pi * 440 * t)).astype("float32"), SR)
    return Session(proj)


def test_the_stem_dynamics_go_on_the_hottest_stem(tmp_path):
    session = _oracle_session(tmp_path)
    assert list(session._state["stems"])[0] == "bass"               # control: not simply the first
    hot = ORACLE._nonlinear_chain(session)
    assert hot == "drums"
    kinds = [e["type"] for e in session._state["stems"]["drums"]["chain"]]
    assert "comp" in kinds and "saturation" in kinds
    assert "comp" not in [e["type"] for e in session._state["stems"]["bass"]["chain"]]


def test_the_engagement_guard_sees_a_stem_comp_that_does_nothing(tmp_path):
    session = _oracle_session(tmp_path)
    hot = ORACLE._nonlinear_chain(session)
    gr_db, sat_dbfs, deess_dbfs, gate_dbfs = ORACLE._stem_dynamics_engage(session, hot)
    assert gate_dbfs >= ORACLE.STEM_GATE_CHANGE_MIN_DBFS      # the gate engages too (2026-10-17)
    assert gr_db >= ORACLE.STEM_COMP_GR_MIN_DB and sat_dbfs >= ORACLE.STEM_SAT_CHANGE_MIN_DBFS
    assert deess_dbfs >= ORACLE.STEM_DEESS_CHANGE_MIN_DBFS     # the de-esser engages too (2026-10-17)
    # ...and on the stem the profile USED to pick, both read as inert: the guard would have failed
    chain = session._state["stems"]["bass"]["chain"]
    chain.extend(e for e in session._state["stems"]["drums"]["chain"]
                 if e["type"] in ("comp", "saturation", "deess", "gate"))
    gr_db, sat_dbfs, _, _ = ORACLE._stem_dynamics_engage(session, "bass")
    assert gr_db < ORACLE.STEM_COMP_GR_MIN_DB


def test_the_oracle_compares_the_peak_it_says_it_compares():
    rng = np.random.default_rng(7)
    mono = rng.standard_normal(SR * 6) * 0.1
    ok, lines = ORACLE.compare(mono, SR, mono.copy(), SR, ref_tp_dbtp=-1.0, daw_tp_dbtp=-0.9)
    assert ok and any("true peak" in line for line in lines), lines
    ok, lines = ORACLE.compare(mono, SR, mono.copy(), SR, ref_tp_dbtp=-1.0,
                               daw_tp_dbtp=-1.0 + ORACLE.TP_TOL_DB + 0.1)
    assert not ok, lines                       # everything else agrees: the peak alone fails it
    ok, lines = ORACLE.compare(mono, SR, mono.copy(), SR)
    assert ok and not any("true peak" in line for line in lines)     # control: not given, not judged


def test_a_one_decibel_level_fault_fails_every_profile():
    """LUFS_TOL_DB was 1.5 over a measured 0.18: a 1 dB fault in the export path passed."""
    rng = np.random.default_rng(7)
    mono = rng.standard_normal(SR * 6) * 0.1
    for profile in ORACLE.PROFILES:
        ok, _ = ORACLE.compare(mono, SR, mono * 10 ** (-1.0 / 20.0), SR, profile)
        assert not ok, profile
        ok, _ = ORACLE.compare(mono, SR, mono * 10 ** (-0.2 / 20.0), SR, profile)
        assert ok, profile                     # control: what the engines really differ by passes
