"""Phase 56, steps 3-4: the chat rig's driver - what it calls a pass.

The rig itself (`python -m orchestrator.chat_rig --all`) needs the console binary and takes
minutes, so it is a harness, not a pytest. This file holds what can be held without it: the
driver's own judgement. A rig that cannot say "diverged" passes while the app refuses
everything it is asked; one that does not notice a turn that never ended, a working line left
running or a request the tape had no step for is green on exactly the failures it exists for.

And two things the rig's first run made necessary: a brain started by a harness never
touches the developer's login keychain, and the scenario table is well formed.
"""
from __future__ import annotations

import json
import json
import subprocess

import pytest

from orchestrator import chat_rig as R
from orchestrator import device_auth


def _turn(**over) -> dict:
    return {"ended": True, "working_line": False, "first_block": 0, "kinds": "UTP",
            "feed": [{"k": "user", "t": "mute the vocals"},
                     {"k": "tool", "key": "track.set_mute", "failed": False, "pending": False},
                     {"k": "prose", "t": "Vocals muted."}],
            "snapshot": {"tracks": [{"name": "vocals", "muted": True}]},
            "after_snapshot": {"tracks": [{"name": "vocals", "muted": False}]}, **over}


def _answered(call_id: str = "toolu_1", ok: bool = True) -> dict:
    block = {"type": "tool_result", "tool_use_id": call_id, "content": '{"ok": true}'}
    if not ok:
        block["is_error"] = True
    return {"messages": [{"role": "user", "content": [{"type": "text", "text": "mute"}]},
                         {"role": "assistant", "content": [{"type": "tool_use", "id": call_id}]},
                         {"role": "user", "content": [block]}]}


def _outcome(turns=None, requests=None, **over) -> R.Outcome:
    base = dict(turns=turns if turns is not None else [_turn()], rig={},
                timing=[{"v": 3, "first_activity_s": 0.2,
                         "commands": {"track.set_mute": {"n": 1, "wait_s": 0.01, "app_s": 0.003}}}],
                requests=requests if requests is not None else [_answered()],
                violations=[], unscripted=0, left=0, fake_s=[], log="", project=None)
    return R.Outcome(**{**base, **over})


SCENARIO = R.SCENARIOS["mute_and_undo"]


# ---------------------------------------------------------------- what always has to hold

def test_a_clean_outcome_passes():
    o = _outcome()
    assert R.standing(o, SCENARIO) == [] and SCENARIO.check(o) == []


@pytest.mark.parametrize("change, says", [
    (dict(turns=[_turn(ended=False)]), "did not end"),
    (dict(turns=[_turn(working_line=True)]), "the working line still runs"),
    (dict(turns=[_turn(feed=[{"k": "user"}, {"k": "tool", "key": "track.set_mute", "pending": True}])]),
     "still waiting for a result"),
    (dict(violations=["tool_choice: nope"]), "the strict rules refused a request"),
    (dict(unscripted=1), "the tape had no step for"),
    (dict(left=2), "never asked for"),
    (dict(requests=[]), "diverged: the tape's call toolu_1 never came back"),
    (dict(requests=[_answered(ok=False)]), "diverged: call toolu_1 came back an error where the tape says ok"),
])
def test_each_thing_that_always_has_to_hold_is_noticed(change, says):
    problems = R.standing(_outcome(**change), SCENARIO)
    assert any(says in p for p in problems), problems


def test_a_scenario_that_expects_unsettled_rows_says_so_itself():
    killed = R.SCENARIOS["brain_killed_mid_sentence"]
    assert killed.unsettled_ok
    pending = _outcome(turns=[_turn(feed=[{"k": "user"}, {"k": "tool", "pending": True}])] * 2,
                       requests=[])
    assert not any("still waiting" in p for p in R.standing(pending, killed))


def test_the_first_scenarios_own_checks_each_bite():
    """The six things, as far as one command can show them."""
    assert any("not muted" in p for p in SCENARIO.check(_outcome(
        turns=[_turn(snapshot={"tracks": [{"name": "vocals", "muted": False}]})])))
    assert any("did not unmute" in p for p in SCENARIO.check(_outcome(
        turns=[_turn(after_snapshot={"tracks": [{"name": "vocals", "muted": True}]})])))
    assert any("one finished turn" in p for p in SCENARIO.check(_outcome(timing=[])))
    assert any("did not stamp" in p for p in SCENARIO.check(_outcome(
        timing=[{"v": 3, "first_activity_s": 0.2,
                 "commands": {"track.set_mute": {"n": 1, "wait_s": 0.01}}}])))


# ---------------------------------------------------------------- a control passes by failing

def test_a_control_passes_only_with_the_problem_it_stages():
    name = "mute_expected_to_fail"
    ok, said = R.verdict(name, ["diverged: call toolu_1 came back ok where the tape says error"])
    assert ok and "control held" in said
    ok, said = R.verdict(name, [])
    assert not ok and "it passed" in said, "a control that passes is a rig that cannot fail"
    ok, _ = R.verdict(name, ["diverged: ...", "turn 1 did not end"])
    assert not ok, "another problem beside the staged one is still a problem"


def test_an_ordinary_scenario_passes_with_no_problems_and_fails_with_one():
    assert R.verdict("mute_and_undo", []) == (True, "")
    assert R.verdict("mute_and_undo", ["turn 1 did not end"])[0] is False


# ---------------------------------------------------------------- the table

def test_every_scenario_is_well_formed():
    assert set(R.CONTROLS) <= set(R.SCENARIOS)
    for name, scenario in R.SCENARIOS.items():
        ids = [block[1] for turn in scenario.turns for step in turn.tape
               for block in step.get("blocks", []) if block[0].startswith("tool_use")]
        for turn in scenario.turns:
            assert turn.prompt.strip(), name
            assert set(turn.results) <= set(ids), f"{name}: a result names a call the tape lacks"
            assert set(turn.results.values()) <= {"ok", "error"}, name
            assert not (set(turn.leg) - {"stop", "kill_brain", "confirm", "confirm_delay_ms",
                                         "answer", "before", "after", "mode", "ceiling_s"}), \
                f"{name}: a leg field the console leg does not read"
        # a call id is the address of a result: one scenario never uses one twice for two calls
        distinct = {(i, str(b)) for turn in scenario.turns for step in turn.tape
                    for b in step.get("blocks", []) if b[0].startswith("tool_use") for i in [b[1]]}
        assert len({i for i, _ in distinct}) == len(distinct), f"{name}: one id for two calls"


def test_the_sandbox_reaches_nothing_of_the_developers(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-a-real-looking-key")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "a-plan-token")
    monkeypatch.setenv("MIX_AGENT_LLM_BACKEND", "subscription")
    env = R.sandbox_env(tmp_path, 4321)
    assert env["ANTHROPIC_API_KEY"] != "sk-ant-a-real-looking-key", "only the placeholder is sent"
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in env, "no plan token: nothing here can be billed"
    assert env["MIX_AGENT_LLM_BACKEND"] == "api" and env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4321"
    assert R.HOME_MARKER in env["CFFIXED_USER_HOME"] and env["MIX_AGENT_KEYCHAIN"] == "none"
    assert env["MIX_AGENT_CLOUD_URL"].startswith("http://127.0.0.1:"), "nothing reaches the control plane"
    assert "MIX_AGENT_RIG_LIVE" not in env, "a live model is asked for by name, never by default"


# ---------------------------------------------------------------- no keychain

def test_a_harness_started_brain_never_runs_the_keychain_tool(monkeypatch):
    """`MIX_AGENT_KEYCHAIN=none`: every read finds nothing and /usr/bin/security is not run.
    (A fake `subprocess.run` lets the suite's own guard through to the real door.)"""
    ran = []
    monkeypatch.setattr(subprocess, "run",
                        lambda argv, **k: ran.append(argv) or subprocess.CompletedProcess(argv, 0, "a-secret", ""))
    argv = ["/usr/bin/security", "find-generic-password", "-s", "x", "-a", "y", "-w"]
    monkeypatch.setenv(device_auth.NO_KEYCHAIN_ENV, "none")
    found = device_auth._security(argv)
    assert found.returncode == device_auth.NOT_FOUND and found.stdout == "" and ran == []
    assert device_auth.Keychain().get() is None
    monkeypatch.delenv(device_auth.NO_KEYCHAIN_ENV)          # control: without it the tool is run
    assert device_auth._security(argv).stdout == "a-secret" and len(ran) == 1


def test_a_shipped_brain_ignores_the_switch(monkeypatch):
    from orchestrator import llm_backend
    monkeypatch.setenv(device_auth.NO_KEYCHAIN_ENV, "none")
    monkeypatch.setattr(llm_backend, "is_frozen", lambda: True)
    assert device_auth.no_keychain() is False, "a development switch never reaches a customer"


def test_the_switch_is_off_unless_it_says_none(monkeypatch):
    monkeypatch.delenv(device_auth.NO_KEYCHAIN_ENV, raising=False)
    assert device_auth.no_keychain() is False
    monkeypatch.setenv(device_auth.NO_KEYCHAIN_ENV, "memory")
    assert device_auth.no_keychain() is False


# ---------------------------------------------------------------- the command sweep

def test_the_sweep_sends_every_agent_command_or_says_why_not():
    """Phase 56, layer 1: every command through the agent's door. A key is swept, or it is
    in `SWEEP_DENY` with its reason - never neither (a new command is then swept by itself),
    and never both."""
    from orchestrator.operate_eval import _registry_commands

    keys = {k for k, user_only, _ in _registry_commands() if not user_only}
    plan = R.sweep_plan()
    assert set(plan) | set(R.SWEEP_DENY) == keys
    assert not set(plan) & set(R.SWEEP_DENY)
    assert all(len(reason) > 8 for reason in R.SWEEP_DENY.values())
    for key, variants in plan.items():
        assert variants[0] == ("bare", {}), key
        assert [v for v, _ in variants] == sorted({v for v, _ in variants},
                                                  key=["bare", "wrong_type", "no_target", "word_index"].index), key


def test_a_command_that_names_which_thing_is_also_asked_for_one_that_is_not_there():
    plan = R.sweep_plan()
    assert ("no_target", {"track": "no_such_thing_zz"}) in plan["track.set_gain"]
    assert dict(plan["clip.move"])["no_target"] == {"track": "no_such_thing_zz", "clip": 9999}
    assert dict(plan["track.set_gain"])["wrong_type"] == {"track": [], "gain_db": []}
    assert "no_target" not in dict(plan["track.add_blank"]), "a NEW name is not a target"
    assert [v for v, _ in plan["transport.play_pause"]] == ["bare"], "nothing to get wrong"


def test_the_census_holds_exactly_what_the_sweep_sends():
    """The pinned census is what `--sweep` compares with. A command the catalog gains has no
    row until a sweep is run and looked at (`--sweep --update-census`)."""
    census = json.loads(R.CENSUS.read_text())
    sent = {f"{key} {variant}" for key, variants in R.sweep_plan().items() for variant, _ in variants}
    sent |= {f"{key} example" for key in R.example_keys()}
    assert set(census) == sent, (sorted(sent - set(census))[:5], sorted(set(census) - sent)[:5])
    assert {row["outcome"] for row in census.values()} <= {"ok", "refused", "queued"}


def test_no_call_that_is_wrong_is_answered_ok():
    """What the sweep is for, pinned: a LIST where one value belongs, a WORD where an index
    belongs, a target that is not there - each is refused. Before the door's scalar rule 60
    list-for-a-value calls and 3 word-for-an-index calls answered ok (`scene.delete {scene:
    "zzz"}` deleted scene 0; `return.add {kind: []}` made a return)."""
    census = json.loads(R.CENSUS.read_text())
    for variant in ("wrong_type", "no_target", "word_index"):
        accepted = sorted(k for k, row in census.items()
                          if k.endswith(" " + variant) and row["outcome"] == "ok")
        assert accepted == [], (variant, accepted[:8])


def test_every_valid_example_answers_ok():
    census = json.loads(R.CENSUS.read_text())
    off = {k: row["outcome"] for k, row in census.items()
           if k.endswith(" example") and row["outcome"] not in ("ok", "queued")}
    assert off == {}


def test_every_mix_session_command_has_an_example_or_a_reason():
    """A command the mix-session families gain is given one valid call in the examples
    session, or a line saying why it has none - never left out unnoticed."""
    plan = R.sweep_plan()
    have = set(R.example_keys())
    family = {k for k in plan if k.startswith(R.EXAMPLE_FAMILIES)}
    assert have <= set(plan), sorted(have - set(plan))             # each is a command the agent may call
    assert family - have == set(R.NO_EXAMPLE_YET), (sorted(family - have - set(R.NO_EXAMPLE_YET)),
                                                     sorted(set(R.NO_EXAMPLE_YET) - (family - have)))
    assert len(have) >= 80
    assert all(len(why) > 12 for why in R.NO_EXAMPLE_YET.values())


def test_the_examples_are_one_session_of_one_call_a_turn():
    scenario = R.examples_scenario()
    assert len(scenario.turns) == len(R.EXAMPLES)
    for turn, example in zip(scenario.turns, R.EXAMPLES):
        calls = [b for step in turn.tape for b in step.get("blocks") or [] if b[0] == "tool_use"]
        assert len(calls) == 1 and calls[0][2] == ("daw_command" if example.tool == "daw_command" else example.tool)
    # the two removals no undo puts back are answered Yes, and nothing else is
    confirmed = {e.key for e in R.EXAMPLES if e.leg.get("confirm") == "yes"}
    assert confirmed == {"return.remove", "bus.remove"}


def test_a_change_of_answer_is_named():
    pinned = {"a.b bare": {"outcome": "refused"}, "a.c bare": {"outcome": "ok", "moved": True},
              "a.gone bare": {"outcome": "ok"}}
    now = {"a.b bare": {"outcome": "ok", "moved": True}, "a.c bare": {"outcome": "ok", "moved": True},
           "a.new bare": {"outcome": "refused"}}
    changes = R.census_changes(now, pinned)
    assert len(changes) == 3
    assert any(c.startswith("a.b bare: was {'outcome': 'refused'}") for c in changes)
    assert any("a.gone bare: in the census, no longer sent" in c for c in changes)
    assert any(c.startswith("a.new bare: new") for c in changes)
    assert R.census_changes(pinned, pinned) == []


# ---------------------------------------------------------------- found by the sweep

def _brain_with_edit(project, tracks):
    from orchestrator import daw_server as S
    brain = S.Brain(project)
    brain.latest_edit = {"tracks": [{"name": t} for t in tracks]}
    return brain


def test_the_agent_cannot_route_a_track_that_is_not_there(demo_project):
    """The command sweep: `route.stem {stem: "no_such_thing_zz", bus: "x"}` answered ok and
    left an empty group in the session and a group track in the app. The brain routes an
    UNREGISTERED name as a pending member (an empty DAW track has no stem yet), so it cannot
    tell a new track from a typo - but the agent's door can: it holds the live edit."""
    from mix_engine.session import Session

    brain = _brain_with_edit(demo_project, list(Session(demo_project).stems) + ["Synth"])
    ops = brain.routing_ops()
    with pytest.raises(ValueError, match="no track named 'vocalz'"):
        ops["route.stem"]({"stem": "vocalz", "bus": "Band"})
    with pytest.raises(ValueError, match="no track named 'vocalz'"):
        ops["group.create"]({"stems": ["vocalz"]})
    assert Session(demo_project)._state["buses"] == {}, "a refusal leaves no group behind"
    # controls: a stem routes, and so does a track the session has no stem for yet
    stem = next(iter(Session(demo_project).stems))
    assert ops["route.stem"]({"stem": stem, "bus": "Band"}) is not None
    assert Session(demo_project).stems[stem]["bus"] == "Band"
    assert ops["route.stem"]({"stem": "Synth", "bus": "Band"}) is not None


def test_with_no_picture_of_the_edit_the_door_does_not_guess(demo_project):
    """A brain that has been pushed no snapshot (a stand-in app, a test) cannot say a name
    is not a track: the old behaviour stands."""
    from orchestrator import daw_server as S
    brain = S.Brain(demo_project)
    assert brain.routing_ops()["route.stem"]({"stem": "not_yet_a_stem", "bus": "Band"}) is not None


# ---------------------------------------------------------------- the ladder

def _untouched(n: int) -> R.Outcome:
    """A session in which every turn ended and NOTHING was done or said."""
    stems = [{"name": name, "kind": "stem", "muted": name == "bass", "gain_db": 0.0, "pan": 0.0,
              "output": "master", "devices": []} for name in ("vocals", "drums", "bass", "other")]
    turn = {"ended": True, "working_line": False, "first_block": 0, "feed": [], "tree_writes": [],
            "snapshot": {"tracks": stems, "markers": [], "returns": [], "tempo": {"bpm": 120.0}}}
    return _outcome(turns=[dict(turn) for _ in range(n)], requests=[], timing=[])


def test_no_rung_is_done_by_an_agent_that_did_nothing():
    """A rung's check is its end state. One an untouched project passes is no check - the
    rule the technique eval's self-test holds its graders to. (Rung 4 asks for the bass
    UNMUTED: the untouched session here has it muted, so that one bites too.)"""
    blank = _untouched(len(R.LADDER))
    passed = [rung.prompt for i, rung in enumerate(R.LADDER) if rung.check(blank, i) == ""]
    assert passed == ["mute the bass"], passed      # the one this fixture already satisfies
    blank.turns[0]["snapshot"]["tracks"][2]["muted"] = False
    assert R.LADDER[0].check(blank, 0) != ""


def test_the_ladder_is_fourteen_rungs_of_eight_kinds_with_a_tape_each():
    assert len(R.LADDER) == 14 and len({r.kind for r in R.LADDER}) == 8
    assert all(r.tape and r.prompt.strip() for r in R.LADDER)
    scripted, live = R.ladder_scenario(), R.ladder_scenario(live=True)
    assert all(t.tape for t in scripted.turns) and not any(t.tape for t in live.turns), \
        "a live run plays no tape: the model answers"
    assert R.SCENARIOS["ladder"].name == "ladder", "the scripted ladder is part of --all"


def test_a_live_run_is_refused_unless_it_is_called_billed(capsys, monkeypatch):
    monkeypatch.setattr(R, "run_scenario", lambda *a, **k: pytest.fail("a live run was started"))
    assert R.main(["--ladder", "--live", "api", "--reps", "5"]) == 1
    assert "BILLED" in capsys.readouterr().out


def test_a_live_runs_environment_carries_one_credential_and_says_live(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-the-callers-own")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "a-plan-token")
    api = R.live_env(tmp_path, "api")
    assert api["ANTHROPIC_API_KEY"] == "sk-ant-the-callers-own" and "CLAUDE_CODE_OAUTH_TOKEN" not in api
    assert api["MIX_AGENT_LLM_BACKEND"] == "api" and api["MIX_AGENT_RIG_LIVE"] == "1"
    assert "ANTHROPIC_BASE_URL" not in api, "no fake behind a live run"
    plan = R.live_env(tmp_path, "subscription")
    assert plan["CLAUDE_CODE_OAUTH_TOKEN"] == "a-plan-token" and "ANTHROPIC_API_KEY" not in plan, \
        "a key in a plan run would bill the API account"
    assert plan["MIX_AGENT_KEYCHAIN"] == "none" and R.HOME_MARKER in plan["CFFIXED_USER_HOME"]
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(SystemExit, match="needs ANTHROPIC_API_KEY"):
        R.live_env(tmp_path, "api")


def test_the_report_is_by_kind_with_the_spread_and_what_was_done():
    o = _untouched(len(R.LADDER))
    o.timing = [{"wall_s": 1.0 + i, "first_activity_s": 0.5, "rounds": 2, "tool_calls": 1,
                 "model_s": 0.6, "tools_s": 0.2, "marks_s": {"prep": 0.1}, "cost_usd": 0.01}
                for i in range(len(R.LADDER))]
    problems = ["rung 3 (one command: 'set the tempo'): the tempo is 120.0, not 100.0"]
    said, table = R.ladder_report([(problems, o), ([], o)])
    one = table["one command"]                      # rungs 1-4, two sessions
    assert one["n"] == 8 and one["done"] == 7, "the rung a session failed is not counted done"
    assert one["wall_p50_s"] == 2.0 and one["wall_p95_s"] == 4.0
    assert "one command" in said and "render-bound" in said
