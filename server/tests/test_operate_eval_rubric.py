"""The operate eval's rubric is deterministic — so it is testable for free.

`operate_eval.grade(case, steps)` turns the recorded tool steps of one chat
turn into a verdict with no LLM in the loop (Phase 48 Track 2 factored it out
of `evaluate` for exactly this). These tests feed synthetic steps in the real
recorder shape (`{"kind", "payload"}`, from `daw_bridge._PlanRecorder`) and
assert every verdict path — the operate-eval equivalent of technique_eval's
`--self-test`: a red billed case indicts the model, not the rubric.

The two Track 2 additions under test:

  * `sequence` — a technique VERTICAL: every key, in order (subsequence). A
    send into a return that was never created is not "a send".
  * `wants_propose` / `propose_stem` — a DEVICE technique (de-esser, mb_comp,
    parallel comp on a stem) has no command key; the right route is ONE
    `propose` whose actions carry one of the named engine types, on the named
    stem. A daw_command is "wrong" there, a run is "escalated", as always.

Plus the `--check` gate scoped to baseline-known cases: a case the baseline
never judged cannot have "broken" — it never passed.
"""
import json

import pytest

from orchestrator import operate_eval
from orchestrator.operate_eval import CORPUS, OpCase, grade

BY_TEXT = {c.text: c for c in CORPUS}


def cmd(command, **args):
    return {"kind": "command", "payload": {"command": command, "args": args}}


def prop(*actions):
    return {"kind": "proposal", "payload": {"actions": list(actions)}}


def run():
    return {"kind": "run", "payload": {"brief": "x"}}


def verdict(text, steps):
    return grade(BY_TEXT[text], steps)[0]


# --------------------------------------------------------------------------
# verticals: every key, in order
# --------------------------------------------------------------------------

ROOM = "give lead_vox a short room reverb on a return, not too wet"


@pytest.mark.parametrize("steps, want", [
    ([cmd("return.add", kind="reverb"),
      cmd("send.set", source="lead_vox", gain_db=-3)], "correct"),
    # an extra command BETWEEN the sequence keys is fine within max_commands
    ([cmd("return.add"), cmd("return.set_level", id="r1", gain_db=-2),
      cmd("send.set", source="lead_vox")], "correct"),
    # sent into the EXISTING plate — the vertical was never built
    ([cmd("send.set", source="lead_vox", gain_db=-3)], "wrong"),
    # right keys, wrong order
    ([cmd("send.set", source="lead_vox"), cmd("return.add")], "wrong"),
    # create + SHAPE (device.set x3) + send is the honest cost, not over-reach
    ([cmd("return.add"), cmd("device.set"), cmd("device.set"),
      cmd("device.set"), cmd("send.set")], "correct"),
    # ...but a second send plus starting the transport "so they can hear it"
    # is the habit the prompt names: more than the request needed
    ([cmd("return.add"), cmd("device.set"), cmd("device.set"),
      cmd("device.set"), cmd("send.set"), cmd("send.set"),
      cmd("transport.play_pause")], "over-reach"),
    ([run()], "escalated"),
    # dropped to a proposal: the stall failure the prompt warns about
    ([prop({"type": "return_reverb", "return_id": "r"},
           {"type": "send", "stem_id": "lead_vox"})], "none"),
])
def test_sequence_vertical(steps, want):
    assert verdict(ROOM, steps) == want


def test_sequence_is_subsequence_not_exact_match():
    """Other commands may sit between the sequence keys - the ones the case allows beside
    its reach (`also`, reader E7·R9); any other is a command the request did not ask for."""
    case = OpCase("x", ("a.b", "c.d"), max_commands=5, sequence=("a.b", "c.d"),
                  also=("z.z",))
    ok, _, _ = grade(case, [cmd("a.b"), cmd("z.z"), cmd("z.z"), cmd("c.d")])
    assert ok == "correct"
    stranger, detail, _ = grade(case, [cmd("a.b"), cmd("q.q"), cmd("c.d")])
    assert stranger == "over-reach" and "q.q" in detail
    bad, detail, _ = grade(case, [cmd("c.d"), cmd("a.b")])
    assert bad == "wrong" and "in order" in detail


# --------------------------------------------------------------------------
# a co-edit that IS a command in the DAW: duck, return level
# --------------------------------------------------------------------------

DUCK = "duck the piano under lead_vox so the vocal stays on top"


@pytest.mark.parametrize("steps, want", [
    ([cmd("duck.set_key", stem="piano", key="lead_vox")], "correct"),
    ([prop({"type": "duck", "stem_id": "piano", "key_stem_id": "lead_vox"})],
     "none"),
    ([run()], "escalated"),
])
def test_duck_is_a_command_here(steps, want):
    assert verdict(DUCK, steps) == want


def test_return_level_has_a_referent_in_the_fixture():
    """'That reverb return' must exist or the honest answer is a question."""
    # the topology is its own context part since 2026-09-02 (production publishes
    # it that way — EditSnapshot.h skips return tracks), not a field of the edit
    assert "send_returns" not in json.loads(operate_eval.OPERATE_CONTEXT)
    returns = operate_eval.OPERATE_SEND_RETURNS["returns"]
    assert any(r["kind"] == "return_reverb" for r in returns)
    assert returns[0]["id"] in operate_eval.fixture_context(
        operate_eval.OPERATE_EDIT, send_returns=operate_eval.OPERATE_SEND_RETURNS)
    lvl = "that reverb return is too loud, bring it down 4 dB"
    assert verdict(lvl, [cmd("return.set_level", id=returns[0]["id"],
                             gain_db=-4)]) == "correct"


# --------------------------------------------------------------------------
# device techniques: ONE proposal, right type, named stem
# --------------------------------------------------------------------------

DEESS = "de-ess lead_vox, the esses are piercing"


@pytest.mark.parametrize("steps, want, why", [
    ([prop({"type": "deess", "stem_id": "lead_vox"})], "correct", "the tool"),
    ([prop({"type": "dynamic_eq", "stem_id": "lead_vox", "freq_hz": 6500,
            "gain_db": -6})], "correct", "the alternative form"),
    ([prop({"type": "eq", "stem_id": "lead_vox", "gain_db": -4})], "wrong",
     "a static EQ is not a de-esser"),
    ([prop({"type": "deess", "stem_id": "bvs"})], "wrong",
     "right tool, wrong stem — scope"),
    ([prop({"type": "deess", "stem_id": "lead_vox"}),
      prop({"type": "eq", "stem_id": "lead_vox"})], "over-reach",
     "two proposals for one technique"),
    ([cmd("device.set", track="lead_vox", target="deess.threshold_db",
          value=-30)], "wrong", "no device exists to set; a command is wrong"),
    ([run()], "escalated", "the wallet failure"),
    ([], "none", "talked, proposed nothing"),
])
def test_device_technique_is_one_proposal(steps, want, why):
    assert verdict(DEESS, steps) == want, why


@pytest.mark.parametrize("text, good, bad", [
    ("add parallel compression to the kick_in to thicken it",
     {"type": "comp", "stem_id": "kick_in", "mix": 0.5, "makeup_db": 6},
     {"type": "eq", "stem_id": "kick_in"}),
    ("the bass_di gets boomy on the swells, tame just the lows",
     {"type": "mb_comp", "stem_id": "bass_di"},
     {"type": "comp", "stem_id": "bass_di"}),        # full-band = wrong tool
    ("warm up the bass_di with some tape saturation",
     {"type": "saturation", "stem_id": "bass_di"},
     {"type": "overdrive", "stem_id": "bass_di"}),
    ("widen the piano but keep its low end mono",
     {"type": "imager", "stem_id": "piano"},
     {"type": "width", "stem_id": "bass_di"}),        # wrong stem
    ("tame the harsh 3k on lead_vox only when it gets loud",
     {"type": "dynamic_eq", "stem_id": "lead_vox", "freq_hz": 3200},
     {"type": "eq", "stem_id": "lead_vox", "freq_hz": 3200}),
])
def test_each_device_case_accepts_its_tool_and_rejects_the_cheat(text, good, bad):
    assert verdict(text, [prop(good)]) == "correct"
    assert verdict(text, [prop(bad)]) == "wrong"


def test_every_wants_propose_type_is_a_real_engine_action():
    """A corpus case naming a type the engine has no Action for measures
    nothing — the model could never satisfy it."""
    import re
    from pathlib import Path
    models = Path(__file__).resolve().parents[1] / "mix_engine" / "models.py"
    real = set(re.findall(r'type: Literal\["([a-z_]+)"\]', models.read_text()))
    for case in CORPUS:
        for t in case.wants_propose:
            assert t in real, f"{case.text!r} names unknown action type {t!r}"


# --------------------------------------------------------------------------
# run guard + regression on the old-style cases
# --------------------------------------------------------------------------

def test_master_for_streaming_is_the_one_technique_that_is_a_run():
    text = "master this for streaming, around -14 LUFS"
    assert verdict(text, [run()]) == "correct"
    assert verdict(text, [cmd("track.set_gain")]) == "wrong"


@pytest.mark.parametrize("text, steps, want", [
    ("mute gtr_r", [cmd("track.set_mute", track="gtr_r", mute=True)], "correct"),
    ("mute gtr_r", [cmd("track.set_mute"), cmd("transport.play_pause")],
     "over-reach"),
    # (an answer has words since Phase 56: with none, the turn answered nothing)
    ("what key is this in?", [], "none"),
    ("what key is this in?", [run()], "escalated"),
])
def test_refactor_left_old_cases_unchanged(text, steps, want):
    assert verdict(text, steps) == want


def test_a_question_answered_in_words_is_correct():
    case = next(c for c in CORPUS if c.text == "what key is this in?")
    assert grade(case, [], "It reads as A minor.")[0] == "correct"


# --------------------------------------------------------------------------
# --check gates only baseline-known cases
# --------------------------------------------------------------------------

def test_check_reports_new_cases_instead_of_breaking(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(operate_eval, "BASELINE_PATH",
                        tmp_path / "operate_baseline.json")
    # A baseline stable failure must be a REAL corpus case — one whose text
    # left the corpus is ORPHANED and fails the gate (next test).
    real = operate_eval.CORPUS[0].text
    operate_eval.BASELINE_PATH.write_text(json.dumps({
        "model": "m", "runs": 3, "score_mean": 90.0,
        "stable_failures": [real],
        "cases": [real, "old passing case"],
    }))
    agg = {"score_mean": 80.0, "runs": 3, "stable_failures": [
        (real, "wrong"),                        # unchanged
        ("brand new technique case", "wrong"),  # never in the baseline
    ]}
    rc = operate_eval._check(agg)
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "NEW" in out and "brand new technique case" in out
    assert "BROKE" not in out


def test_check_still_fails_on_a_real_regression(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(operate_eval, "BASELINE_PATH",
                        tmp_path / "operate_baseline.json")
    operate_eval.BASELINE_PATH.write_text(json.dumps({
        "model": "m", "runs": 3, "score_mean": 90.0,
        "stable_failures": [], "cases": ["used to pass"],
    }))
    rc = operate_eval._check({"score_mean": 70.0, "runs": 3,
                              "stable_failures": [("used to pass", "wrong")]})
    assert rc == 1
    assert "BROKE" in capsys.readouterr().out


# ---- the confirm second turn (2026-08-26) -------------------------------
# A `needs_confirm` key's contract is ask, get the yes, invoke. `evaluate`
# gives a case expecting one of those keys a single "Yes — go ahead." turn
# when the first turn invoked nothing — these tests prove the turn fires
# exactly there, with a fake ChatSession in place of the model.

MACKIE = ("hook up the Mackie control surface over the IAC bus, both "
          "directions")


def _fake_session(act_on_turn=None, command="control_surface.set",
                  args=None):
    """A ChatSession stand-in: records its turns; on turn `act_on_turn`
    invokes `command` through the harness's own daw_command tool."""
    class Fake:
        instances = []

        def __init__(self, *a, **k):
            self.turns = []
            Fake.instances.append(self)

        async def turn(self, text, *, tools, context, emit, **kw):
            self.turns.append(text)
            if act_on_turn is not None and len(self.turns) == act_on_turn:
                await tools.daw_command(command, args or {})
    return Fake


def _evaluate_with(monkeypatch, fake, text):
    from orchestrator import chat
    monkeypatch.setattr(chat, "ChatSession", fake)
    return operate_eval.evaluate(BY_TEXT[text])


def test_confirm_case_gets_a_yes_turn_and_the_act_counts(monkeypatch):
    fake = _fake_session(act_on_turn=2, args={
        "name": "Mackie Control Universal", "enabled": True,
        "midi_in": "IAC Driver Bus 1", "midi_out": "IAC Driver Bus 1"})
    out = _evaluate_with(monkeypatch, fake, MACKIE)
    assert fake.instances[-1].turns == [MACKIE, "Yes — go ahead."]
    assert out.verdict == "correct", out.detail


def test_confirm_case_that_still_does_nothing_earned_its_none(monkeypatch):
    fake = _fake_session(act_on_turn=None)
    out = _evaluate_with(monkeypatch, fake, MACKIE)
    assert fake.instances[-1].turns == [MACKIE, "Yes — go ahead."]
    assert out.verdict == "none"


def test_confirm_case_acting_immediately_is_refused_like_production(monkeypatch):
    """The DAW's choke point refuses a confirm-gated key until the user says yes, so
    the fixture does too (2026-09-01 sweep, P2 — it used to answer ok, and a model
    that fired the command without asking graded exactly like one that asked). A
    model that acts on turn one and never comes back has NOT done the job."""
    fake = _fake_session(act_on_turn=1, args={
        "name": "Mackie Control Universal", "enabled": True,
        "midi_in": "IAC Driver Bus 1", "midi_out": "IAC Driver Bus 1"})
    out = _evaluate_with(monkeypatch, fake, MACKIE)
    assert fake.instances[-1].turns == [MACKIE, "Yes — go ahead."]
    assert out.verdict == "none", out.detail


def test_confirm_case_refused_then_accepted_after_the_yes(monkeypatch):
    """...and one that retries after the refusal, once the user has said yes, is correct."""
    class Fake:
        instances = []

        def __init__(self, *a, **k):
            self.turns = []
            self.replies = []
            Fake.instances.append(self)

        async def turn(self, text, *, tools, context, emit, **kw):
            self.turns.append(text)
            self.replies.append(await tools.daw_command("control_surface.set", {
                "name": "Mackie Control Universal", "enabled": True,
                "midi_in": "IAC Driver Bus 1", "midi_out": "IAC Driver Bus 1"}))
    out = _evaluate_with(monkeypatch, Fake, MACKIE)
    fake = Fake.instances[-1]
    assert fake.turns == [MACKIE, "Yes — go ahead."]
    assert "confirm" in str(fake.replies[0]).lower(), fake.replies[0]
    assert out.verdict == "correct", out.detail


def test_ordinary_case_never_gets_the_yes_turn(monkeypatch):
    fake = _fake_session(act_on_turn=None)
    out = _evaluate_with(monkeypatch, fake,
                         "split the kick_in clip at the playhead")
    assert fake.instances[-1].turns == ["split the kick_in clip at the "
                                        "playhead"]
    assert out.verdict == "none"


# ---- a renamed baseline failure is not a fixed one (2026-09-01 sweep, P0) ----
# Three baseline stable failures were reworded in the corpus after the baseline
# was written. `was - now` put their old texts in FIXED and the gate printed
# "7 stable failures -> 4", PASS, for a run that fixed nothing (the failures came
# back under the new texts as ungated NEW). A baseline failure whose text left
# the corpus is ORPHANED: it cannot be judged, so it is a hard FAIL until the
# baseline is regenerated.

def _baseline(tmp_path, monkeypatch, failures, cases):
    monkeypatch.setattr(operate_eval, "BASELINE_PATH",
                        tmp_path / "operate_baseline.json")
    operate_eval.BASELINE_PATH.write_text(json.dumps({
        "model": "m", "runs": 3, "score_mean": 90.0,
        "stable_failures": failures, "cases": cases}))


def test_check_does_not_report_a_renamed_baseline_failure_as_fixed(
        tmp_path, monkeypatch, capsys):
    _baseline(tmp_path, monkeypatch, ["old wording of a case"],
              ["old wording of a case"])
    # the same case still fails every run, under its new text
    rc = operate_eval._check({"score_mean": 90.0, "runs": 3, "stable_failures": [
        ("new wording of the same case", "wrong")]})
    out = capsys.readouterr().out
    assert rc == 1, out
    assert "FIXED" not in out
    assert "ORPHANED" in out and "old wording of a case" in out
    assert "regenerate" in out.lower()


def test_check_says_fixed_only_for_a_real_case_that_stopped_failing(
        tmp_path, monkeypatch, capsys):
    real = operate_eval.CORPUS[0].text
    _baseline(tmp_path, monkeypatch, [real], [real])
    rc = operate_eval._check({"score_mean": 95.0, "runs": 3,
                              "stable_failures": [], "correct_counts": {real: 3}})
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "FIXED" in out and real in out
    # ...and only then (reader E7·R4): this test used to pass a run with NO per-case counts
    # and expect FIXED - "left the failing set" is not "has a correct run now". A case the
    # sweep holds no graded run for is said to be unseen, not fixed.
    rc = operate_eval._check({"score_mean": 95.0, "runs": 3, "stable_failures": []})
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "FIXED" not in out and "UNSEEN" in out and real in out


def test_check_lists_a_baseline_failure_that_still_fails(
        tmp_path, monkeypatch, capsys):
    """The D2 gap from the same sweep: the gate cannot fail on a baseline
    failure that persists, so it must at least SAY so — the only signal used
    to be the absence of a FIXED line, which nothing reads."""
    real = operate_eval.CORPUS[0].text
    _baseline(tmp_path, monkeypatch, [real], [real])
    rc = operate_eval._check({"score_mean": 90.0, "runs": 3,
                              "stable_failures": [(real, "wrong")]})
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "STILL" in out and real in out


# ---- a value the KEY carries (2026-10-04: the keyboard's own keys became the agent's) ----

def test_a_value_named_by_the_key_itself_counts_as_named():
    """`launcher.loop_4_bars` says "4 bars" as surely as `set_loop_length {bars: 4}`.

    The Loop Deck's loop-length keys were user-only until 2026-10-04; once the
    agent may press them, "set the launcher loop to 4 bars" has two right
    answers, and one of them carries its value in the key and no args at all.
    """
    case = BY_TEXT["set the launcher loop to 4 bars"]
    assert grade(case, [cmd("launcher.loop_4_bars")])[0] == "correct"
    assert grade(case, [cmd("launcher.set_loop_length", bars=4)])[0] == "correct"
    # ...and it can still fail, both ways
    assert grade(case, [cmd("launcher.set_loop_length", bars=8)])[0] == "wrong"
    assert grade(case, [cmd("launcher.loop_8_bars")])[0] == "wrong"


def test_no_other_assertion_is_met_by_its_own_key():
    """The key's words join the graded text, so an assertion that merely repeats
    a word of its own expected key could never fail. Exactly one case is meant
    to be answerable that way; a second one is a rubric that grades nothing."""
    import re

    met = set()
    for case in CORPUS:
        for want in case.args_contain:
            for alt in want.split("|"):
                for key in case.expect:
                    if re.search(r"(?<![A-Za-z0-9_])" + re.escape(alt) + r"(?![A-Za-z0-9_])",
                                 operate_eval._key_words(key), re.IGNORECASE):
                        met.add((case.text, alt, key))
    assert met == {("set the launcher loop to 4 bars", "4", "launcher.loop_4_bars")}, met

