"""Safety and bounds — the brain half of the 2026-09-10 agent audit's P1-7 to P1-10.

- P1-7: a mix run's planner could reach the live DAW through `daw_command`, which skipped
  the pass guardrails. In a faked run a refused +12 dB proposal came back as
  `track.set_gain +24` with `master.limiter {on:false}`, and nothing reached the ledger.
- P1-8: the command path accepted −96..+24 dB while the brain persisted −60..+12, so the
  live graph (and the export, a bounce of it) played one level and session.json held
  another. The two bounds must be one.
- P1-10: a session id from a project's `chats/index.json` was used as a file path, so an
  id of "../../x" or an absolute path had that file read in as conversation history.
"""
from __future__ import annotations

import json
import re
import types
from pathlib import Path

import pytest

from mix_engine.session import Session
from orchestrator import agents
from orchestrator.daw_bridge import (BRAIN_ONLY_COMMANDS, RESTORED_USER_PREFIX,
                                     rehydrate_history)

REPO = Path(__file__).resolve().parents[2]


def _ctx(calls: list):
    return types.SimpleNamespace(
        daw_invoke=lambda command, args: calls.append((command, args)) or {"ok": True},
        engine_call=lambda name, args: {"engine": name})


# ------------------------------------------------------------------ P1-7

def test_the_run_planner_offers_no_daw_command_tool():
    # Design call 2026-09-11: a mix run has no command surface at all (not a denylist).
    assert "daw_command" not in {d["name"] for d in agents.PLANNER_TOOL_DEFS}


@pytest.mark.parametrize("command, args", [
    ("track.set_gain", {"track": "vocals", "gain_db": 24}),
    ("master.limiter", {"on": False}),
    ("device.enable", {"track": "master", "device": "limiter", "on": False}),
    ("device.set", {"track": "master", "target": "utility.gain_db", "value": 12}),
    ("transport.seek", {"position_s": 12}),
    ("track.delete", {"track": "vocals"}),
    ("some.key_added_next_year", {}),
])
def test_a_mix_run_refuses_every_daw_command_it_invents(command, args):
    calls: list = []
    out = agents.ClaudeAgent._route_tool(_ctx(calls), "daw_command",
                                         {"command": command, "args": args})
    assert out["ok"] is False and "Propose the move instead" in out["error"]
    assert calls == [], "the DAW never received it"


def test_the_run_context_carries_the_snapshot_but_not_the_catalog():
    src = (REPO / "orchestrator/agents.py").read_text()
    assert 'block["daw_commands"]' not in src
    assert 'block["daw_edit_state"]' in src
    assert "no daw_command tool" in agents.SYSTEM_PROMPT


# ------------------------------------------------------------------ P1-8

def test_the_command_level_bound_is_the_bound_the_brain_persists():
    src = (REPO / "daw/Source/DawCommands.h").read_text()
    lo = float(re.search(r"kMinLevelDb\s*=\s*(-?[\d.]+)", src).group(1))
    hi = float(re.search(r"kMaxLevelDb\s*=\s*(-?[\d.]+)", src).group(1))
    assert (lo, hi) == Session.USER_EDIT_FIELDS["gain_db"], \
        "a level the command path accepts must be a level session.json can hold"


def test_send_and_return_command_bounds_are_what_the_brain_persists():
    from mix_engine import session as sessmod
    src = (REPO / "daw/Source/DawCommands.h").read_text()
    ret = float(re.search(r"kMaxReturnDb\s*=\s*(-?[\d.]+)", src).group(1))
    snd = float(re.search(r"kMaxSendDb\s*=\s*(-?[\d.]+)", src).group(1))
    assert ret == sessmod.RETURN_DB_RANGE[1] and snd == sessmod.SEND_DB_RANGE[1], \
        "return.set_level / send.set must refuse what set_return_level / set_send would clamp"


# ------------------------------------------------------------------ P1-10

@pytest.mark.parametrize("hostile", ["../../outside/victim", "/ABSOLUTE", "s1/../../x", "t1"])
def test_a_session_id_is_never_a_path(tmp_path, hostile):
    project = tmp_path / "proj"
    (project / "chats").mkdir(parents=True)
    victim = tmp_path / "outside" / "victim.json"
    victim.parent.mkdir()
    victim.write_text(json.dumps([{"k": "user", "t": "hi"},
                                  {"k": "prose", "t": "OUTSIDE-THE-PROJECT"}]))
    sid = str(victim.with_suffix("")) if hostile == "/ABSOLUTE" else hostile
    assert rehydrate_history(project, sid) == []


def test_a_real_session_id_still_rehydrates(tmp_path):
    project = tmp_path / "proj"
    (project / "chats").mkdir(parents=True)
    (project / "chats" / "s1789000000000.json").write_text(json.dumps(
        [{"k": "user", "t": "hi"}, {"k": "prose", "t": "hello"}]))
    assert rehydrate_history(project, "s1789000000000") == [
        {"role": "user", "content": RESTORED_USER_PREFIX + "hi"},
        {"role": "assistant", "content": "hello"}]


# ------------------------------------------------------------------ reader S9·R6

def test_the_keys_the_daw_answers_above_its_choke_point_are_the_brains_alone():
    """The DAW's request handler answers a few keys BEFORE the catalog lookup and the checks
    every command passes (user-only, the master safety limiter, the take hold, confirm): they
    are the brain's own messages. Each of them must be a key the bridge refuses when the MODEL
    names it (`daw_bridge.BRAIN_ONLY_COMMANDS`) - a new raw branch in the handler with no row
    there is a door around the choke point."""
    app = (REPO / "daw/Source/App.cpp").read_text(encoding="utf-8")
    start = app.index("brain.onDawRequest = [this]")
    choke = app.index("dawcmd::idForKey (command)", start)
    raw = set(re.findall(r'command\s*==\s*"([^"]+)"', app[start:choke]))
    assert raw, "the handler's raw branches were not found - has it moved?"
    assert raw == set(BRAIN_ONLY_COMMANDS), (
        f"answered above the choke point: {sorted(raw)}; refused when a model names them: "
        f"{sorted(BRAIN_ONLY_COMMANDS)}")
