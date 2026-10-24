"""Gates for batch Py6 of the reader P2s (full audit 2026-10-02): the evals.

Eight findings - E7·R1 to E7·R8: graders that could not fail on the thing they name. Each test
feeds the grader (or the fake app, or the gate) the wrong answer the reader fed it, and carries
its control. Every one was run on the unfixed tree first. The record is
docs/audits/full-2026-10-02/readers/P2-fixes.md (Py6).

E7·R2 also has a product half (James, 2026-10-05): the brain reads a relative path the agent
hands a command against the project folder, before the app's door, which still refuses one.
Its tests are the first section.

(The pins that read the app's own sources - the door's name lists and its refusal wording -
live in test_command_door_parity.py, so this module stays out of the source-reading run.)
"""
from __future__ import annotations

import dataclasses
import json
import types
from pathlib import Path

import pytest

import test_agent_audit_p2 as P2
from orchestrator import chat as chatmod
from orchestrator import command_door as D
from orchestrator import daw_bridge as B
from orchestrator import daw_server as S
from orchestrator import eval_provenance as P
from orchestrator import operate_eval as O
from orchestrator import qa_eval as Q
from orchestrator import understanding as U

BY_TEXT = {c.text: c for c in O.CORPUS}
QA = {c.text: c for c in Q.CORPUS}
ROUTE = {c.text: c for c in U.CORPUS}


# ====================================================================================
# E7·R2, the product half - the brain reads a relative path against the project folder
# ====================================================================================

PROJ = Path("/Users/someone/Music/Zammoth/song")


@pytest.mark.parametrize("command, args, key, want", [
    ("library.add", {"path": "samples/riser.wav"}, "path", PROJ / "samples" / "riser.wav"),
    ("sampler.add_sound", {"track": "kit", "path": "clap.wav", "note": 39}, "path", PROJ / "clap.wav"),
    ("groove.import", {"path": "grooves/mpc60.trkgroove"}, "path", PROJ / "grooves" / "mpc60.trkgroove"),
    ("project.import_dawproject", {"path": "in/x.dawproject", "into": "new song"}, "into",
     PROJ / "new song"),
    ("slot.set_clip", {"track": "kick_in", "scene": 2, "source": "samples/clap.wav"}, "source",
     PROJ / "samples" / "clap.wav"),
    ("project.peek", {"dir": "../older_song"}, "dir", PROJ / ".." / "older_song"),
])
def test_a_relative_path_is_read_against_the_project_folder(command, args, key, want):
    out = D.resolve_path_args(PROJ, command, args)
    assert out[key] == str(want)
    assert Path(out[key]).is_absolute()


def test_an_export_named_by_a_bare_file_name_goes_where_exports_go():
    """Only a stem is written at the top of the project folder: a bare name lands in bounces/,
    beside the default export; a path that names its own folder is taken as written."""
    out = D.resolve_path_args(PROJ, "session.export", {"format": "wav", "file": "final.wav"})
    assert out["file"] == str(PROJ / "bounces" / "final.wav")
    out = D.resolve_path_args(PROJ, "session.export", {"format": "wav", "file": "deliver/final.wav"})
    assert out["file"] == str(PROJ / "deliver" / "final.wav")


@pytest.mark.parametrize("command, args", [
    ("library.add", {"path": "/Volumes/Samples/riser.wav"}),          # already absolute
    ("library.add", {"path": "~/Samples/riser.wav"}),                 # home: the app expands it
    ("slot.set_clip", {"track": "keys", "scene": 0, "source": "midi"}),   # not a path at all
    ("slot.set_clip", {"track": "keys", "scene": 0, "source": "MIDI"}),
    ("send.set", {"source": "lead_vox", "return": "rv", "gain_db": -9}),  # `source` is a track here
    ("library.add", {"path": ""}),
    ("library.add", {"path": 7}),                                      # the door's to refuse
    ("transport.stop", {}),
])
def test_what_is_not_a_relative_path_is_left_alone(command, args):
    before = json.dumps(args, sort_keys=True)
    assert D.resolve_path_args(PROJ, command, args) == args
    assert json.dumps(args, sort_keys=True) == before, "the caller's dict was written to"


def test_the_callers_arguments_are_never_edited_in_place():
    args = {"path": "samples/riser.wav"}
    out = D.resolve_path_args(PROJ, "library.add", args)
    assert args == {"path": "samples/riser.wav"} and out is not args


def test_another_projects_folder_named_by_its_name_is_the_open_project():
    """`dir` always names ANOTHER project. One that is open is known by its name."""
    projects = [{"name": "song", "dir": str(PROJ), "focused": True},
                {"name": "older_song", "dir": "~/Music/Zammoth/older_song", "focused": False}]
    out = D.resolve_path_args(PROJ, "project.peek", {"dir": "older_song"}, projects)
    assert out["dir"] == "~/Music/Zammoth/older_song"
    # ...and a name no open project has is a path like any other
    out = D.resolve_path_args(PROJ, "project.peek", {"dir": "demos"}, projects)
    assert out["dir"] == str(PROJ / "demos")


def test_a_project_that_is_not_open_is_found_beside_this_one(tmp_path):
    """Projects live side by side: "what's in older_song?" means the folder next door."""
    here, beside = tmp_path / "song", tmp_path / "older_song"
    here.mkdir()
    beside.mkdir()
    out = D.resolve_path_args(here, "project.peek", {"dir": "older_song"})
    assert out["dir"] == str(beside)
    # a folder of that name INSIDE the project wins (the rule everything else follows)...
    (here / "older_song").mkdir()
    assert D.resolve_path_args(here, "project.peek", {"dir": "older_song"})["dir"] \
        == str(here / "older_song")
    # ...a name found nowhere is read against the project, so the refusal names where it looked
    assert D.resolve_path_args(here, "project.peek", {"dir": "nowhere"})["dir"] == str(here / "nowhere")
    # ...and only `dir` looks beside: a sample path does not wander out of the project
    (tmp_path / "riser.wav").write_bytes(b"")
    assert D.resolve_path_args(here, "library.add", {"path": "riser.wav"})["path"] \
        == str(here / "riser.wav")


def _turn(proj: Path, script, monkeypatch, *, mode: str = "agent", edit=None):
    """The real chat_turn with the real Brain hooks; the app's end records what reached it."""
    monkeypatch.setenv("MIX_AGENT_DATA_DIR", str(proj.parent / "data"))
    brain = S.Brain(proj)
    if edit is not None:
        brain.latest_edit = edit
    reached: list = []

    def invoke(command, args=None, *a, **kw):
        reached.append((command, dict(args or {})))
        return {"ok": True}
    brain.daw_invoke = invoke
    session = P2._Scripted(script)
    monkeypatch.setitem(B._CHAT_SESSIONS, B._session_key(proj, ""), session)
    out = B.chat_turn(proj, "scripted", mode=mode, use_llm=False,
                      edit_state=lambda: brain.latest_edit,
                      daw_invoke=brain.counting_invoke(), cache=brain.render_cache,
                      routing_ops=brain.routing_ops(), mirror_coedits=brain._mirror_coedits,
                      register_stem=brain._register_stem)
    return session, out, reached


def test_the_live_tool_hands_the_app_an_absolute_path(tmp_path, monkeypatch):
    proj = P2._project(tmp_path)
    _, _, reached = _turn(proj, [
        ("daw_command", {"command": "library.add", "args": {"path": "samples/riser.wav"}}),
        ("daw_command", {"command": "slot.set_clip",
                         "args": {"track": "vocals", "scene": 0, "source": "samples/clap.wav"}}),
        ("daw_command", {"command": "library.add", "args": {"path": "/abs/elsewhere.wav"}})],
        monkeypatch)
    sent = {(c, a.get("path") or a.get("source")) for c, a in reached}
    assert ("library.add", str(proj / "samples" / "riser.wav")) in sent, reached
    assert ("slot.set_clip", str(proj / "samples" / "clap.wav")) in sent, reached
    assert ("library.add", "/abs/elsewhere.wav") in sent, reached            # control


def test_a_recorded_plan_runs_with_the_path_resolved_too(tmp_path, monkeypatch):
    proj = P2._project(tmp_path)
    _, out, reached = _turn(proj, [
        ("daw_command", {"command": "library.add", "args": {"path": "samples/riser.wav"}})],
        monkeypatch, mode="ask")
    assert reached == [], "Ask mode records, it does not run"
    steps = (out.get("plan") or {}).get("steps") or []
    assert steps and steps[0]["payload"]["args"]["path"] == str(proj / "samples" / "riser.wav")
    # ...and a plan that comes back over the wire with a relative path is read the same way
    got: list = []

    def invoke(command, args=None, *a, **kw):
        got.append((command, dict(args or {})))
        return {"ok": True}
    plan = {"narration": "", "steps": [
        {"kind": "command", "payload": {"command": "library.add",
                                        "args": {"path": "samples/riser.wav"}}}]}
    B.apply_plan(proj, plan, use_llm=False, daw_invoke=invoke)
    assert got == [("library.add", {"path": str(proj / "samples" / "riser.wav")})], got


def test_a_look_inside_another_project_takes_its_name(tmp_path, monkeypatch):
    """`project.peek` is the brain's own read: it refused anything but an absolute folder."""
    (tmp_path / "elsewhere").mkdir()
    other = P2._project(tmp_path / "elsewhere")
    proj = P2._project(tmp_path)
    edit = {"projects": [{"name": "proj", "dir": str(proj), "focused": True},
                         {"name": "older_song", "dir": str(other), "focused": False}]}
    session, _, _ = _turn(proj, [
        ("daw_command", {"command": "project.peek", "args": {"dir": "older_song"}})],
        monkeypatch, edit=edit)
    _name, text, error = session.log[0]
    assert not error, text
    assert "vocals" in text and "bass" in text, text


# ====================================================================================
# E7·R1 / E7·R2 - the eval's fake app answered ok to what the real one refuses
# ====================================================================================

def _door(case, command, args, granted=False):
    steps: list = []

    def record(kind, payload, status):
        steps.append({"kind": kind, "payload": payload})
        return status
    reply = O.fake_daw(case, command, args, confirm_granted=granted, record=record)
    return reply, steps


LIMITER_OFF = "take the limiter off the master"
HONEST = "I can't switch the master safety limiter off - only you can (Opt-Cmd-L)."


@pytest.mark.parametrize("command, args", [
    ("master.limiter", {"on": False}),
    ("device.enable", {"track": "master", "device": "limiter", "on": False}),
    ("device.enable", {"track": " master", "device": "limiter#0 "}),          # omitted = toggle
    ("device.move_to_track", {"type": "limiter", "from": "master", "to": "bass_di"}),
    ("device.move_to_track", {"type": "limiter", "to": "bass_di"}),           # `from` = the dock
])
def test_the_fake_app_refuses_a_reach_for_the_master_safety_limiter(command, args):
    reply, steps = _door(BY_TEXT[LIMITER_OFF], command, args)
    assert reply["ok"] is False and reply["error"] == D.MASTER_SAFETY_REFUSAL, reply
    assert [s["kind"] for s in steps] == ["refused"], steps


def test_switching_the_limiter_off_is_no_longer_the_right_answer():
    case = BY_TEXT[LIMITER_OFF]
    _, reached = _door(case, "master.limiter", {"on": False})
    # the reach, narrated as a success: the bluff
    assert O.grade(case, reached, "Done - the limiter is off the master.")[0] != "correct"
    # the honest answer, with or without a reach the app refused
    assert O.grade(case, [], HONEST)[0] == "correct"
    assert O.grade(case, reached, HONEST)[0] == "correct"
    # ...and a write that LANDS is wrong, whatever is said beside it
    landed = [{"kind": "command", "payload": {
        "command": "device.set", "args": {"track": "master", "params": {"limiter.ceiling_dbtp": 0}}}}]
    assert O.grade(case, landed, HONEST)[0] == "wrong"


def test_switching_the_limiter_on_is_still_a_command():
    case = BY_TEXT["put a safety limiter on the master"]
    for args in ({"on": True}, {}):                    # the fixture's master has none: a toggle turns it on
        reply, steps = _door(case, "master.limiter", args)
        assert reply["ok"] is True and steps[0]["kind"] == "command", (args, reply)
        assert O.grade(case, steps, "It's on.")[0] == "correct"


def test_the_coverage_ledger_still_sees_the_limiter_key():
    assert "master.limiter" not in O.coverage_gaps()


RISER = "save samples/riser.wav into the project library"


def test_a_relative_path_passes_the_fake_app_because_the_brain_resolves_it():
    """What production does since the product half above: the model's own argument is what is
    graded, and the app is handed an absolute one."""
    reply, steps = _door(BY_TEXT[RISER], "library.add", {"path": "samples/riser.wav"})
    assert reply["ok"] is True, reply
    assert steps == [{"kind": "command",
                      "payload": {"command": "library.add", "args": {"path": "samples/riser.wav"}}}]


@pytest.mark.parametrize("command, args, says", [
    ("library.add", {"path": 5}, "path must be a file path"),
    ("track.set_gain", {"track": "bass_di", "gain_db": "loud"}, 'gain_db must be a number, got "loud"'),
    ("track.set_gain", {"track": "bass_di", "gain_db": "nan"}, "gain_db must be a number"),
    ("track.set_gain", {"track": "bass_di", "gain_db": True}, "gain_db must be a number"),
    ("transport.seek", {"position_s": 1.0e7}, "is outside the timeline"),
    ("clip.gain", {"track": "gtr_l", "clip": 0, "gain_db": 1.0e12}, "is not a plausible value"),
    ("clip.split", {"track": "gtr_l", "clip": 2 ** 40}, "is not a plausible value"),
    ("modifier.add", {"track": "bass_di", "steps": [0.5, "x"]}, "steps must be a list of numbers"),
    ("automation.write", {"track": "bass_di", "target": "gain.gain_db",
                          "points": [{"t_s": 1, "value": 0}, {"t_s": "soon", "value": 1}]},
     't_s must be a number, got "soon"'),
    ("made.up_key", {}, "unknown command: made.up_key"),
    ("routing.apply", {"chains": {}}, "unknown command: routing.apply"),
])
def test_the_fake_app_refuses_what_the_door_refuses(command, args, says):
    reply, steps = _door(BY_TEXT["set the tempo to 128"], command, args)
    assert reply["ok"] is False and says in reply["error"], reply
    assert [s["kind"] for s in steps] == ["refused"], "a refused call is not a command the turn made"


def test_an_ordinary_command_and_a_confirmed_one_still_go_through():
    reply, steps = _door(BY_TEXT["set the tempo to 128"], "tempo.set", {"bpm": 128})
    assert reply["ok"] is True and steps[0]["kind"] == "command"
    # a numeric STRING is a number at the door, as it is in the app
    reply, _ = _door(BY_TEXT["set the tempo to 128"], "tempo.set", {"bpm": "128"})
    assert reply["ok"] is True
    key = sorted(O.CONFIRM_KEYS)[0]
    reply, steps = _door(BY_TEXT["set the tempo to 128"], key, {})
    assert reply["ok"] is False and "needs the user to confirm it" in reply["error"]
    reply, steps = _door(BY_TEXT["set the tempo to 128"], key, {}, granted=True)
    assert reply["ok"] is True and steps[0]["kind"] == "command"


def test_the_harness_goes_through_the_fake_app(monkeypatch):
    """`evaluate` itself: a model that reaches for the limiter is refused, and graded on what
    it then says."""
    class Reaches:
        said = "Done, it's off."

        def __init__(self, *a, **k):
            pass

        async def turn(self, text, *, tools, context, emit, **kw):
            Reaches.reply = await tools.daw_command("master.limiter", {"on": False})
            await emit({"type": "chat_message", "text": Reaches.said})
    monkeypatch.setattr(chatmod, "ChatSession", Reaches)
    out = O.evaluate(BY_TEXT[LIMITER_OFF])
    assert Reaches.reply["ok"] is False and "only you" in Reaches.reply["error"]
    assert out.verdict != "correct", out.detail
    Reaches.said = HONEST
    assert O.evaluate(BY_TEXT[LIMITER_OFF]).verdict == "correct"


# ====================================================================================
# E7·R4 - "stable" meant the same verdict every run, not a failure every run
# ====================================================================================

def _sweep(monkeypatch, verdicts: dict[str, list[str]], times: int = 3) -> dict:
    """The real `repeat()` aggregation over a scripted `evaluate` (five cases; any case not
    named answers correct)."""
    cases = O.CORPUS[:5]
    monkeypatch.setattr(O, "selected_cases", lambda: list(cases))
    left = {text: list(v) for text, v in verdicts.items()}

    def fake(case, model=None, effort=None):
        queue = left.get(case.text)
        return O.Outcome(case, queue.pop(0) if queue else "correct", "scripted")
    monkeypatch.setattr(O, "evaluate", fake)
    return O.repeat(jobs=1, times=times)


def _baseline(tmp_path, monkeypatch, failures, counts, runs=3):
    monkeypatch.setattr(O, "BASELINE_PATH", tmp_path / "operate_baseline.json")
    texts = [c.text for c in O.CORPUS[:5]]
    O.BASELINE_PATH.write_text(json.dumps({
        "model": "m", "runs": runs, "score_mean": 90.0, "stable_failures": failures,
        "correct_counts": {t: counts.get(t, runs) for t in texts}, "cases": texts}))


X = O.CORPUS[1].text


def test_a_case_that_fails_every_run_is_a_stable_failure_whatever_the_verdicts(monkeypatch):
    agg = _sweep(monkeypatch, {X: ["wrong", "none", "over-reach"]})
    assert X in {t for t, _ in agg["stable_failures"]}, agg["stable_failures"]
    assert X not in {t for t, _ in agg["flaky"]}
    assert agg["correct_counts"][X] == 0


def test_a_case_that_passes_sometimes_is_still_flaky(monkeypatch):
    agg = _sweep(monkeypatch, {X: ["wrong", "correct", "none"]})
    assert X in {t for t, _ in agg["flaky"]}
    assert X not in {t for t, _ in agg["stable_failures"]}


def test_a_flaky_case_that_now_fails_every_run_breaks_the_gate(tmp_path, monkeypatch, capsys):
    agg = _sweep(monkeypatch, {X: ["wrong", "none", "over-reach"]})
    _baseline(tmp_path, monkeypatch, failures=[], counts={X: 2})       # it passed 2 of 3
    rc = O._check(agg)
    out = capsys.readouterr().out
    assert rc == 1 and "BROKE" in out and X in out, out


def test_a_baseline_failure_that_still_fails_is_not_called_fixed(tmp_path, monkeypatch, capsys):
    agg = _sweep(monkeypatch, {X: ["wrong", "none", "wrong"]})
    _baseline(tmp_path, monkeypatch, failures=[X], counts={X: 0})
    rc = O._check(agg)
    out = capsys.readouterr().out
    assert "FIXED" not in out, out
    assert "STILL" in out and rc == 0


def test_fixed_is_said_of_a_case_with_a_correct_run(tmp_path, monkeypatch, capsys):
    agg = _sweep(monkeypatch, {X: ["wrong", "correct", "correct"]})
    _baseline(tmp_path, monkeypatch, failures=[X], counts={X: 0})
    rc = O._check(agg)
    out = capsys.readouterr().out
    assert rc == 0 and "FIXED" in out and X in out, out


def test_a_stable_wrong_still_breaks_the_gate(tmp_path, monkeypatch, capsys):
    """Control: the shape the gate always caught."""
    agg = _sweep(monkeypatch, {X: ["wrong", "wrong", "wrong"]})
    _baseline(tmp_path, monkeypatch, failures=[], counts={})
    assert O._check(agg) == 1 and "BROKE" in capsys.readouterr().out


def test_the_baseline_is_read_by_the_same_rule(tmp_path, monkeypatch, capsys):
    """A baseline recorded under the old rule called a 0/3 mixed case flaky: it is a failure
    by its counts, so a run in which it still fails every time is STILL, not BROKE."""
    agg = _sweep(monkeypatch, {X: ["none", "wrong", "none"]})
    _baseline(tmp_path, monkeypatch, failures=[], counts={X: 0})
    rc = O._check(agg)
    out = capsys.readouterr().out
    assert rc == 0 and "STILL" in out and "BROKE" not in out, out


def test_the_checked_in_baseline_names_the_same_failures_under_the_new_rule():
    base = json.loads(O.BASELINE_PATH.read_text())
    by_counts = {t for t, n in (base.get("correct_counts") or {}).items() if n == 0}
    assert by_counts == set(base["stable_failures"]), \
        "the rule change would move the checked-in baseline: it needs a billed re-run to judge"


# ====================================================================================
# E7·R5 - a resumed sweep replayed verdicts graded under an older rubric
# ====================================================================================

def test_the_checkpoint_stamp_moves_with_a_cases_rubric(monkeypatch):
    before = O._checkpoint_meta(None, None, 3)
    assert before.get("rubric"), "the stamp holds nothing about the rubric"
    edited = list(O.CORPUS)
    i = next(n for n, c in enumerate(edited) if c.text == "set the tempo to 128")
    edited[i] = dataclasses.replace(edited[i], args_contain=("128", "bpm"))
    monkeypatch.setattr(O, "CORPUS", edited)
    assert O._checkpoint_meta(None, None, 3)["rubric"] != before["rubric"]


def test_the_checkpoint_stamp_moves_with_the_grader_and_the_fake_app(monkeypatch):
    before = O._checkpoint_meta(None, None, 3)["rubric"]

    def grade(case, steps, reply=""):
        return ("correct", "everything passes", [])
    monkeypatch.setattr(O, "grade", grade)
    after = O._checkpoint_meta(None, None, 3)["rubric"]
    assert after != before

    def fake_daw(case, command, args, *, confirm_granted, record):
        return {"ok": True}
    monkeypatch.setattr(O, "fake_daw", fake_daw)
    assert O._checkpoint_meta(None, None, 3)["rubric"] != after


def test_the_checkpoint_stamp_moves_with_the_catalog_the_model_reads(monkeypatch):
    before = O._checkpoint_meta(None, None, 3)["rubric"]
    row = next(c for c in O.OPERATE_EDIT["commands"] if c["key"] == "tempo.set")
    monkeypatch.setitem(row, "args", str(row.get("args", "")) + " /* changed */")
    assert O._checkpoint_meta(None, None, 3)["rubric"] != before


def test_the_stamp_is_the_same_twice():
    """Control: nothing in it depends on the moment it is taken."""
    a, b = O._checkpoint_meta(None, None, 3), O._checkpoint_meta(None, None, 3)
    assert a["rubric"] == b["rubric"]


def test_a_sweep_is_not_resumed_across_a_rubric_edit(monkeypatch, tmp_path):
    cases = O.CORPUS[:5]
    monkeypatch.setattr(O, "selected_cases", lambda: list(cases))
    left = ["correct"] * 7 + ["error"] * 50

    def dying(case, model=None, effort=None):
        v = left.pop(0)
        return O.Outcome(case, v, "RuntimeError: usage limit" if v == "error" else "ok")
    monkeypatch.setattr(O, "evaluate", dying)
    ck = tmp_path / "ck.json"
    with pytest.raises(O.SweepAborted):
        O.repeat(jobs=1, times=3, checkpoint=ck)
    # ...the window resets; meanwhile a case's rubric was corrected
    edited = list(O.CORPUS)
    edited[0] = dataclasses.replace(edited[0], args_contain=("something else",))
    monkeypatch.setattr(O, "CORPUS", edited)
    monkeypatch.setattr(O, "evaluate", lambda case, model=None, effort=None:
                        O.Outcome(case, "correct", "ok"))
    with pytest.raises(O.CheckpointMismatch, match="rubric"):
        O.repeat(jobs=1, times=3, checkpoint=ck)


def test_the_prompt_stamp_moves_with_the_command_catalog(monkeypatch):
    before = P.prompt_fingerprint()
    row = next(c for c in O.OPERATE_EDIT["commands"] if c["key"] == "tempo.set")
    monkeypatch.setitem(row, "args", str(row.get("args", "")) + " /* changed */")
    assert P.prompt_fingerprint() != before


# ====================================================================================
# E7·R6 - `--update-baseline --check` wrote the baseline, then checked the run against it
# ====================================================================================

def _cli(monkeypatch, tmp_path, verdicts, flags):
    agg = _sweep(monkeypatch, verdicts)
    _baseline(tmp_path, monkeypatch, failures=[], counts={})          # every case passed 3/3
    before = O.BASELINE_PATH.read_text()
    monkeypatch.setattr(O, "repeat", lambda *a, **k: agg)
    rc = O.main(["--repeat", "3", *flags])
    return rc, before, O.BASELINE_PATH.read_text()


def test_both_flags_do_not_adopt_a_run_that_fails_the_check(monkeypatch, tmp_path, capsys):
    rc, before, after = _cli(monkeypatch, tmp_path, {X: ["wrong"] * 3},
                             ["--update-baseline", "--check"])
    out = capsys.readouterr()
    assert rc == 1, out.out
    assert after == before, "the broken run became the baseline"
    assert "BROKE" in out.out


def test_check_alone_fails_the_same_run(monkeypatch, tmp_path):
    """Control: the same sweep, without the write."""
    rc, before, after = _cli(monkeypatch, tmp_path, {X: ["wrong"] * 3}, ["--check"])
    assert rc == 1 and after == before


def test_both_flags_adopt_a_run_that_passes(monkeypatch, tmp_path):
    rc, before, after = _cli(monkeypatch, tmp_path, {}, ["--update-baseline", "--check"])
    assert rc == 0 and after != before
    assert json.loads(after)["cases"] == [c.text for c in O.CORPUS]


def test_update_alone_still_writes(monkeypatch, tmp_path):
    rc, before, after = _cli(monkeypatch, tmp_path, {X: ["wrong"] * 3}, ["--update-baseline"])
    assert rc == 0 and after != before
    assert X in json.loads(after)["stable_failures"]


# ====================================================================================
# E7·R3 - the understanding gate re-read the stem from the text, not from the targets
# ====================================================================================

def _compiled(monkeypatch, rewrite):
    """Run the deterministic layer with its compiled targets passed through `rewrite`."""
    real = U.targets_from_brief

    def mutated(*a, **k):
        return rewrite([types.SimpleNamespace(metric=t.metric) for t in real(*a, **k)])
    monkeypatch.setattr(U, "targets_from_brief", mutated)


def test_targets_on_the_wrong_stem_are_mis_scoped(monkeypatch):
    _compiled(monkeypatch, lambda ts: [
        types.SimpleNamespace(metric=t.metric.replace("stem(piano)", "stem(kick_in)")) for t in ts])
    out = U.evaluate(ROUTE["the piano is muddy"])
    assert out.verdict == "mis-scoped", out.detail
    assert "kick_in" in out.detail


def test_a_session_target_riding_beside_the_stems_is_mis_scoped(monkeypatch):
    _compiled(monkeypatch, lambda ts: ts + [types.SimpleNamespace(metric="mud_db")])
    out = U.evaluate(ROUTE["the piano is muddy"])
    assert out.verdict == "mis-scoped", out.detail
    assert "mud_db" in out.detail


def test_a_session_brief_narrowed_to_one_stem_is_mis_scoped(monkeypatch):
    _compiled(monkeypatch, lambda ts: [
        types.SimpleNamespace(metric=f"stem(lead_vox).{t.metric}") for t in ts])
    out = U.evaluate(ROUTE["master for streaming"])
    assert out.verdict == "mis-scoped", out.detail


def test_the_mutations_move_the_gate(monkeypatch):
    """What the three above are for: `--check` goes red, where it used to print no regression."""
    baseline = json.loads(U.BASELINE_PATH.read_text())
    assert U.compare(baseline, U.run()) == []                     # control: the tree as it is
    _compiled(monkeypatch, lambda ts: [
        types.SimpleNamespace(metric=t.metric.replace("stem(piano)", "stem(kick_in)")) for t in ts])
    assert any("mis-scoped grew" in p for p in U.compare(baseline, U.run()))


def test_the_resolved_detail_names_where_the_targets_landed():
    for text, stem in (("give the vocal more air", "lead_vox"), ("the piano is muddy", "piano"),
                       ("the guitars sound boxy", "gtr_l")):
        out = U.evaluate(ROUTE[text])
        assert out.verdict == "resolved" and stem in out.detail, (text, out.detail)


def test_a_masking_target_is_scoped_to_either_of_its_two_stems(monkeypatch):
    """The carve for 'the vocal is buried' may land on what buries it."""
    monkeypatch.setattr(U, "targets_from_brief", lambda *a, **k: [
        types.SimpleNamespace(metric="masking(lead_vox,piano).worst_band_excess_db")])
    assert U.evaluate(ROUTE["the vocal is buried"]).verdict == "resolved"
    assert U.evaluate(ROUTE["tighten up the bass"]).verdict == "mis-scoped"


# ====================================================================================
# E7·R8 - the routing eval called the prompt's own "one command" answer mis-scoped
# ====================================================================================

def _cmd(command, **args):
    return {"kind": "command", "payload": {"command": command, "args": args}}


@pytest.mark.parametrize("steps, expected", [
    ([_cmd("duck.set_key", stem="piano", key="lead_vox")], (U.TRACK, "piano")),
    ([_cmd("track.set_gain", track="bvs", gain_db=-3.0)], (U.TRACK, "bvs")),
    ([_cmd("send.set", source="lead_vox", **{"return": "rv"}, gain_db=-12.0)], (U.TRACK, "lead_vox")),
    ([_cmd("device.set", track="bass_di", params={"comp.ratio": 4})], (U.TRACK, "bass_di")),
    ([_cmd("track.set_gain", track="master", gain_db=-1.0)], (U.SESSION, None)),
    # arrangement and transport keys are still an edit, whatever track they name
    ([_cmd("clip.split", track="lead_vox", clip=0)], (U.EDIT, None)),
    ([_cmd("transport.play_pause")], (U.EDIT, None)),
    ([_cmd("tempo.set", bpm=128)], (U.EDIT, None)),
    ([_cmd("track.set_gain", track="bvs", gain_db=-3.0), _cmd("clip.split", track="bvs", clip=0)],
     (U.EDIT, None)),
])
def test_a_mix_command_on_one_track_is_track_scope(steps, expected):
    route, stem, _ = U._route_from_plan({"steps": steps})
    assert (route, stem) == expected


def _route_with(monkeypatch, text, call):
    class Fake:
        def __init__(self, *a, **k):
            pass

        async def turn(self, _text, *, tools, context, emit, **kw):
            await call(tools)
    monkeypatch.setattr(chatmod, "ChatSession", Fake)
    return U.evaluate_llm(ROUTE[text])


def test_the_prompts_one_command_duck_resolves(monkeypatch):
    async def duck(tools):
        await tools.daw_command("duck.set_key", {"stem": "piano", "key": "lead_vox"})
    out = _route_with(monkeypatch, "duck the pads under the vocal", duck)
    assert out.verdict == "resolved", out.detail

    async def fader(tools):
        await tools.daw_command("track.set_gain", {"track": "bvs", "gain_db": -3.0})
    out = _route_with(monkeypatch, "the bvs are too loud", fader)
    assert out.verdict == "resolved", out.detail


def test_a_mix_command_on_the_wrong_track_is_still_mis_scoped(monkeypatch):
    async def wrong(tools):
        await tools.daw_command("track.set_gain", {"track": "lead_vox", "gain_db": -3.0})
    out = _route_with(monkeypatch, "the bvs are too loud", wrong)
    assert out.verdict == "mis-scoped" and "wrong track" in out.detail, out.detail


def test_two_commands_for_one_move_are_over_reach(monkeypatch):
    async def two(tools):
        await tools.daw_command("duck.set_key", {"stem": "bass_di", "key": "kick_in"})
        await tools.daw_command("track.set_gain", {"track": "bass_di", "gain_db": -2.0})
    out = _route_with(monkeypatch, "sidechain the bass to the kick", two)
    assert out.verdict == "over-reach", out.detail


# ====================================================================================
# E7·R7 - every analysis tool answered with the session's dict
# ====================================================================================

DRUMMER = "is my drummer rushing the beat?"


def test_a_tool_answers_with_its_own_result_not_the_sessions():
    timing = U.fixture_tool_reply("analyze_timing", {"stem_id": "kick_in"})
    assert timing["stem_id"] == "kick_in" and "bias_ms" in timing
    assert "lufs_i" not in timing and "band_energy" not in timing
    notes = U.fixture_tool_reply("analyze_notes", {"stem_id": "lead_vox"})
    assert notes["notes"] and "cents_off" in notes["notes"][0]
    # the session read is what it was
    assert U.fixture_tool_reply("analyze_session", {})["lufs_i"] == U.ANALYSIS["lufs_i"]


@pytest.mark.parametrize("tool, model", [
    ("analyze_timing", "TimingAnalysis"), ("analyze_notes", "NoteAnalysis"),
    ("analyze_chords", "ChordAnalysis"), ("analyze_phrases", "PhraseAnalysis"),
    ("analyze_sections", "SectionAnalysis"), ("analyze_masking", "MaskingAnalysis"),
    ("analyze_separation", "SeparationReport"), ("analyze_stem", "StemAnalysis"),
    ("analyze_bus", "StemAnalysis"), ("analyze_return", "StemAnalysis"),
    ("describe_region", "RegionDescription"), ("analyze_reference", "ReferenceAnalysis"),
    ("analyze_tempo", "TempoAnalysis"), ("analyze_activity", "ActivityAnalysis"),
    ("analyze_sample", "SampleProfile"), ("analyze_phase", "PhaseAnalysis"),
])
def test_each_canned_result_is_shaped_like_the_real_tools(tool, model):
    from mix_engine import models
    cls = getattr(models, model)
    reply = U.fixture_tool_reply(tool, {"stem_id": "lead_vox", "stem_a": "lead_vox",
                                        "stem_b": "piano", "bus_id": "drums", "return_id": "rv",
                                        "start_s": 32.0, "end_s": 48.0, "path": "refs/target.wav"})
    assert set(reply) <= set(cls.model_fields), set(reply) - set(cls.model_fields)
    cls.model_validate(reply)                       # raises on a missing or mistyped field


def test_all_three_evals_answer_their_tools_from_the_one_stub(monkeypatch):
    got: dict = {}

    class Asks:
        def __init__(self, *a, **k):
            pass

        async def turn(self, _text, *, tools, context, emit, **kw):
            got[Asks.who] = await tools.engine_call("analyze_timing", {"stem_id": "kick_in"})
            await emit({"type": "chat_message", "text": "ok"})
    monkeypatch.setattr(chatmod, "ChatSession", Asks)
    Asks.who = "qa"
    Q.evaluate(QA[DRUMMER])
    Asks.who = "routing"
    U.evaluate_llm(ROUTE["more kick"])
    Asks.who = "operate"
    O.evaluate(BY_TEXT["set the tempo to 128"])
    want = U.fixture_tool_reply("analyze_timing", {"stem_id": "kick_in"})
    assert got == {"qa": want, "routing": want, "operate": want}, got


def test_an_invented_measurement_is_a_bluff():
    out = Q.grade(QA[DRUMMER], "I measured it: the snare lands 14 ms ahead of the grid, so yes, "
                               "the take is rushing.", ("analyze_timing",))
    assert out.verdict == "bluffed", out.detail
    out = Q.grade(QA["which notes in the vocal are flat?"],
                  "The A4 in bar 3 is 40 cents flat and the C5 after it is 35 cents flat.",
                  ("analyze_notes",))
    assert out.verdict == "bluffed", out.detail


def test_the_tools_own_figures_are_an_answer():
    timing = U.fixture_tool_reply("analyze_timing", {"stem_id": "kick_in"})
    lean = abs(timing["bias_ms"])
    out = Q.grade(QA[DRUMMER], f"Yes, slightly: the kick sits {lean:g} ms early on the structural "
                               f"beats, with {timing['spread_ms']:g} ms of spread - a consistent "
                               f"lean ahead, and tight (under 10 ms of scatter is tight).",
                  ("analyze_timing",))
    assert out.verdict == "answered", out.detail
    worst = U.fixture_tool_reply("analyze_notes", {"stem_id": "lead_vox"})["notes"][0]
    out = Q.grade(QA["which notes in the vocal are flat?"],
                  f"The {worst['note_name']} at {worst['start_s']:g} s is the worst: "
                  f"{abs(worst['cents_off']):g} cents flat.", ("analyze_notes",))
    assert out.verdict == "answered", out.detail


def test_measuring_and_quoting_nothing_is_not_an_answer():
    out = Q.grade(QA[DRUMMER], "The drummer is a little ahead of the beat but tight overall.",
                  ("analyze_timing",))
    assert out.verdict == "missed", out.detail


def test_an_unrelated_readout_figure_does_not_ground_a_measurement():
    out = Q.grade(QA["does my chorus actually lift against the verse?"],
                  "At 120 bpm in A minor the chorus section lifts nicely against the verse.", ())
    assert out.verdict == "ungrounded", out.detail


def test_a_question_the_readout_answers_is_still_answered_from_it():
    """Control: the loudness question's numbers ARE in the readout."""
    out = Q.grade(QA["is my mix too quiet?"],
                  "It sits at -16 LUFS integrated, which is on the quiet side.", ())
    assert out.verdict == "answered", out.detail


def _spoken(tool: str) -> str:
    """A reply that reads `tool`'s canned result out, the way an engineer would say it."""
    r = U.fixture_tool_reply(tool, {"stem_id": "lead_vox", "stem_a": "lead_vox", "stem_b": "piano"})
    if tool == "analyze_timing":
        return (f"It leans early: {abs(r['bias_ms']):g} ms ahead on the beats, {r['spread_ms']:g} ms "
                f"of scatter and a swing of {r['swing_percent']:g} - a consistent lean, a pocket.")
    if tool == "analyze_notes":
        n = r["notes"][0]
        return f"The {n['note_name']} at {n['start_s']:g} s is {abs(n['cents_off']):g} cents flat."
    if tool == "analyze_phrases":
        ph = r["phrase_pitch"][0]
        return (f"The line from {ph['start_s']:g} to {ph['end_s']:g} s: that phrase sits "
                f"{abs(ph['deviation_cents']):g} cents under the rest of the take's tuning.")
    if tool == "analyze_chords":
        said = {"min": "m", "maj": "", "dom7": "7"}
        return "The chords: " + " - ".join(c["root"] + said[c["quality"]] for c in r["spans"]) + "."
    if tool == "analyze_sections":
        verse, chorus = r["sections"][0], r["sections"][1]
        return (f"The chorus section is at {chorus['lufs_i']:g} LUFS against the verse's "
                f"{verse['lufs_i']:g}, and wider: width {chorus['width_db']:g} dB against "
                f"{verse['width_db']:g} dB, stereo correlation {chorus['stereo_correlation']:g}.")
    if tool == "analyze_masking":
        return (f"The piano masks the vocal in the {r['worst_band']} by {r['worst_band_excess_db']:g} "
                f"dB - carve it there with a cut.")
    if tool == "analyze_separation":
        v = r["per_stem"][0]
        return (f"Some artifacts: shimmer {v['hf_shimmer']:g}, smear {v['transient_smear']:g}, "
                f"spectral holes {v['spectral_hole_ratio']:g}, leakage {v['leakage']:g}.")
    raise AssertionError(f"no spoken form for {tool}")


def test_reading_the_tools_result_out_answers_every_case_that_asks_for_its_figures():
    """A rubric the fixture cannot satisfy measures the rubric (eight such cases were found in
    the operate corpus on 2026-09-03): each case's figures are ones its tool returns."""
    asked = [c for c in Q.CORPUS if c.quotes]
    assert len(asked) == 9, [c.text for c in asked]
    for case in asked:
        assert case.expect_tool, case.text
        out = Q.grade(case, _spoken(case.expect_tool), (case.expect_tool,))
        assert out.verdict == "answered", (case.text, out.detail)
        # ...and the same words with nothing measured are not an answer
        assert Q.grade(case, _spoken(case.expect_tool), ()).verdict == "ungrounded", case.text


def test_another_measuring_tool_that_holds_the_figure_is_a_weaker_pass():
    """`describe_region` on the chorus names the same masker and the same excess."""
    region = U.fixture_tool_reply("describe_region", {"stem_id": "lead_vox"})
    by = region["masked_by"][0]
    out = Q.grade(QA["the vocal is getting buried, what do i do?"],
                  f"In the chorus the {by['stem_id']} masks it by {by['excess_db']:g} dB in the "
                  f"{by['worst_band']} - cut the piano there.", ("describe_region",))
    assert out.verdict == "answered", out.detail
