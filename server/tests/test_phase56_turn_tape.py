"""Phase 56: a chat turn can be kept as a tape, and a tape is a rig turn.

Nothing kept a tool call's arguments: the transcript holds display rows, the digest command
keys, the timing record keys and seconds. So a real session could not be played again with
the model taken out. `MIX_AGENT_TAPE=1` (a source checkout only) keeps each turn's rounds
as the fake API's own steps.
"""
from __future__ import annotations

import json

import anyio
import pytest

import orchestrator.chat as chatmod
from orchestrator import chat_rig, llm_backend, turn_tape
from orchestrator.chat import CATALOG_LABEL, ChatSession, ChatTools
from orchestrator.turn_timing import TurnTimer
from tests.fake_anthropic import serve

TOOLS = [{"name": "daw_command", "description": "operate the DAW",
          "input_schema": {"type": "object", "properties": {
              "command": {"type": "string"}, "args": {"type": "object"}}}}]
CONTEXT = ("Workspace mode: agent (your moves apply immediately).\n\n"
           + CATALOG_LABEL + "): [{\"key\": \"track.set_gain\"}]")
ARGS = {"command": "track.set_gain", "args": {"track": "vox", "gain_db": -2.5, "note": "x" * 300}}


class _Tools(ChatTools):
    def __init__(self, refuse=()):
        self.refuse = refuse

    async def daw_command(self, command, args):
        return {"ok": command not in self.refuse}


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(chatmod, "_DISABLED_BETAS", set())
    fake, shutdown = serve(monkeypatch)
    try:
        yield fake
    finally:
        shutdown()


async def _discard(_event):
    pass


def _turn(api, tools=None):
    api.script = [{"blocks": [("thinking", ""), ("tool_use", "toolu_1", "daw_command", ARGS),
                              ("tool_use", "toolu_2", "daw_command", {"command": "track.delete", "args": {}})],
                   "usage": {"input_tokens": 40, "cache_read_input_tokens": 9000,
                             "cache_creation_input_tokens": 100}, "out": 77},
                  {"blocks": [("thinking", ""), ("text", "Vocal down 2.5 dB.")]}]
    session = ChatSession(reasoning="medium", tool_defs_for=lambda during_run: TOOLS)
    anyio.run(lambda: session.turn("pull the vocal down", tools=tools or _Tools(),
                                   context=CONTEXT, emit=_discard))
    return session


def test_a_turn_is_not_taped_unless_a_developer_asked(api, monkeypatch):
    monkeypatch.delenv(turn_tape.TAPE_ENV, raising=False)
    assert _turn(api).tape is None
    monkeypatch.setenv(turn_tape.TAPE_ENV, "1")
    assert turn_tape.wanted() is True
    monkeypatch.setattr(llm_backend, "is_frozen", lambda: True)
    assert turn_tape.wanted() is False, "a shipped brain keeps no tape, whatever the environment says"


def test_a_taped_turn_holds_each_round_whole_and_how_each_call_came_back(api, monkeypatch):
    monkeypatch.setenv(turn_tape.TAPE_ENV, "1")
    kept = _turn(api, _Tools(refuse=("track.delete",))).tape.as_dict()
    assert kept["prompt"] == "pull the vocal down"
    first, second = kept["steps"]
    assert first["blocks"][1] == ["tool_use", "toolu_1", "daw_command", ARGS], \
        "the whole input, not a display row cut at 120 characters"
    assert first["stop"] == "tool_use" and first["out"] == 77
    assert first["usage"] == {"input_tokens": 40, "cache_read_input_tokens": 9000,
                              "cache_creation_input_tokens": 100}
    assert second["blocks"][-1] == ["text", "Vocal down 2.5 dB."]
    assert ["thinking", ""] in first["blocks"], "a thinking block is kept as a place, never its words"
    assert kept["results"] == {"toolu_1": "ok", "toolu_2": "error"}


def test_the_tape_is_written_in_the_projects_own_folder_and_read_back(api, monkeypatch, tmp_path):
    monkeypatch.setenv(turn_tape.TAPE_ENV, "1")
    session = _turn(api)
    path = turn_tape.write(tmp_path, session)
    assert path.parent == tmp_path / "runs" / "tape" and json.loads(path.read_text())["prompt"]
    assert [t["prompt"] for t in turn_tape.read(tmp_path)] == ["pull the vocal down"]
    session.tape = None
    assert turn_tape.write(tmp_path, session) is None and len(turn_tape.read(tmp_path)) == 1


def test_a_transport_that_keeps_no_rounds_gets_them_back_from_its_calls():
    """The subscription path: the SDK owns the request, so only the calls are seen. How many
    each response made is what decides how often the brain measures, so that is kept."""
    tape = turn_tape.Tape("mix it")
    for n in range(5):
        tape.call(f"c{n}", "daw_command", {"command": "track.set_gain", "args": {"n": n}}, n != 3)
    steps = tape.steps(calls_per_round=[2, 0, 3, 0])
    assert [len([b for b in s["blocks"] if b[0] == "tool_use"]) for s in steps] == [2, 3, 0]
    assert steps[1]["blocks"][1][1] == "c2" and steps[-1]["blocks"][-1][0] == "text"
    assert tape.as_dict([2, 0, 3, 0])["results"]["c3"] == "error"


def test_a_tape_is_a_rig_turn_held_to_how_each_call_came_back(api, monkeypatch, tmp_path):
    monkeypatch.setenv(turn_tape.TAPE_ENV, "1")
    turn_tape.write(tmp_path, _turn(api, _Tools(refuse=("track.delete",))))
    (turn,) = chat_rig.tape_turns(tmp_path)
    assert turn.prompt == "pull the vocal down" and len(turn.tape) == 2
    assert turn.tape[0]["blocks"][1] == ("tool_use", "toolu_1", "daw_command", ARGS)
    assert turn.results == {"toolu_1": "ok", "toolu_2": "error"}, \
        "a replay that meets another answer than the recording is `diverged`"


def test_the_ladders_budget_holds_counts_exactly_and_seconds_loosely():
    measured = {"turns": 14, "rounds": 29, "tool_calls": 17, "renders_master": 6, "renders_stem": 0,
                "agent_time_s": 5.0, "brain_share_s": 3.1, "prep_max_s": 0.4, "measure_s": 1.2,
                "first_activity_max_s": 0.3, "out_tokens": 400, "model_s": 0.9}
    budget = chat_rig.ladder_budget(measured)
    assert budget["turns"] == 14
    assert budget["max"]["rounds"] == 29 and budget["max"]["renders_master"] == 6
    assert budget["max"]["brain_share_s"] == 10.3 and budget["max"]["prep_max_s"] == 2.2
    assert "model_s" not in budget["max"] and "out_tokens" not in budget["max"], \
        "the scripted model's own figures are no budget"


def test_the_committed_budget_is_for_the_ladder_as_it_stands():
    budget = json.loads(chat_rig.LADDER_BUDGET.read_text())
    assert budget["turns"] == len(chat_rig.LADDER)
    assert set(chat_rig._COUNTED) <= set(budget["max"])


def test_a_replay_answers_an_approval_box_as_the_recording_did(tmp_path):
    """The tape keeps no click. A confirm-gated command that came back ok was allowed; one
    that came back an error was declined."""
    folder = tmp_path / "runs" / "tape"
    folder.mkdir(parents=True)
    export = ["tool_use", "c1", "daw_command", {"command": "session.export", "args": {"format": "wav"}}]
    mute = ["tool_use", "c2", "daw_command", {"command": "track.set_mute", "args": {"track": "bass"}}]
    for n, (blocks, results) in enumerate([([export], {"c1": "ok"}), ([export], {"c1": "error"}),
                                           ([mute], {"c2": "error"})]):
        (folder / f"{n}-t.json").write_text(json.dumps(
            {"prompt": f"turn {n}", "steps": [{"blocks": blocks}], "results": results}))
    allowed, declined, plain = chat_rig.tape_turns(tmp_path)
    assert allowed.leg["confirm"] == "yes" and declined.leg["confirm"] == "no"
    assert plain.leg == {}, "a turn with no approval box arms nothing"
