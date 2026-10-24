"""The 2026-09-10 agent-workspace audit's P3s — the Python half (docs/audits/agent-2026-09-10).

Each leg drives the shipping code (a scripted stand-in for the SDK or the model where one
would be, a real socket where the DAW would be) and fails at an ASSERTION against the pre-fix
code — except the side channel's spend, whose fix IS a new parameter (`spent`), said at the leg.
The C++ half (the project-scoped glow, the "≈ API $" label) is in `--glow` / `--engineer-feed`.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import types
from pathlib import Path

import anyio
import pytest

from orchestrator import chat as chatmod
from orchestrator import daw_bridge as B
from orchestrator import daw_server as S
from orchestrator import eval_provenance as P
from orchestrator import llm_subscription as subm
from orchestrator.calibration import exemplars, load_records_file
from orchestrator.loop import EngineClient

from test_agent_audit_p2 import (_Client, _Final, _fake_sdk, _line, _project, _read_until,
                                 _serve, _sub_env, _tools)

HOME = os.path.expanduser("~")


def _capturing_sdk(script) -> tuple[types.ModuleType, list[dict]]:
    """_fake_sdk, plus every ClaudeAgentOptions the transport built."""
    sdk = _fake_sdk(script)
    seen: list[dict] = []
    build = sdk.ClaudeAgentOptions

    def options(**kw):
        seen.append(kw)
        return build(**kw)
    sdk.ClaudeAgentOptions = options
    return sdk, seen


async def _quiet(_event) -> None:
    pass


def _sub_session(**kw) -> chatmod.ChatSession:
    return chatmod.ChatSession(client=object(), tool_defs_for=chatmod.daw_tool_defs, **kw)


# ------------------------------------------------------------------ the subscription transport

def test_a_model_switch_mid_conversation_reaches_the_subscription(monkeypatch):
    """01·F3: the SDK transport was built once with the model of that moment, so every later
    turn ran the OLD model — and Haiku → Opus sent Opus's effort to Haiku, which 400s."""
    _sub_env(monkeypatch)

    async def script(tools, m):
        yield m.AssistantMessage([m.TextBlock("ok")])
    sdk, seen = _capturing_sdk(script)
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", sdk)
    models = [m["id"] for m in chatmod.CHAT_MODELS]
    first, second = models[0], next(m for m in models if m != models[0])
    session = _sub_session(model=first)
    asyncio.run(session.turn("one", tools=_tools(), context="ctx", emit=_quiet))
    session.set_settings(second, chatmod.DEFAULT_REASONING)
    asyncio.run(session.turn("two", tools=_tools(), context="ctx", emit=_quiet))
    assert [o["model"] for o in seen] == [first, second], seen


def test_a_failed_subscription_turn_still_counts_what_it_burned(monkeypatch):
    """01·F10: the CLI reports a failed call's usage in its ResultMessage before the error, and
    the chat turn dropped it — eval token lines read low on exactly the failing turns."""
    _sub_env(monkeypatch)

    async def script(tools, m):
        yield m.ResultMessage(usage={"input_tokens": 1200, "output_tokens": 300,
                                     "cache_read_input_tokens": 40000},
                              is_error=True, subtype="success", api_error_status=429,
                              result="You've hit your limit")
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", _fake_sdk(script))
    session = _sub_session()
    with pytest.raises(subm.SubscriptionError):
        asyncio.run(session.turn("hi", tools=_tools(), context="ctx", emit=_quiet))
    assert session.usage["input_tokens"] == 1200, session.usage
    assert session.usage["cache_read_input_tokens"] == 40000, session.usage


def test_reply_paragraphs_from_separate_responses_are_not_glued(monkeypatch):
    """01·F11: "…first.It sits at…" in history, the session digest and plan narration."""
    _sub_env(monkeypatch)

    async def script(tools, m):
        yield m.AssistantMessage([m.TextBlock("Let me measure the master first.")], message_id="r1")
        yield m.AssistantMessage([m.TextBlock("It sits at -9 LUFS.")], message_id="r2")
        # one response streamed as two messages sharing an id stays ONE paragraph
        yield m.AssistantMessage([m.TextBlock("Plenty of ")], message_id="r3")
        yield m.AssistantMessage([m.TextBlock("headroom.")], message_id="r3")
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", _fake_sdk(script))
    session = _sub_session()
    asyncio.run(session.turn("how loud?", tools=_tools(), context="ctx", emit=_quiet))
    assert session.history[-1]["content"] == (
        "Let me measure the master first.\n\nIt sits at -9 LUFS.\n\nPlenty of headroom.")


_REROUTERS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
              "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY")


@pytest.mark.parametrize("var,value", [
    ("ANTHROPIC_AUTH_TOKEN", "sk-another-account"),
    ("CLAUDE_CODE_USE_BEDROCK", "1"),
    ("CLAUDE_CODE_USE_VERTEX", "true"),
    ("ANTHROPIC_BASE_URL", "https://proxy.example"),
])
def test_the_subscription_refuses_what_would_reroute_it(monkeypatch, var, value):
    """01·F12: the guard checked only ANTHROPIC_API_KEY, and the CLI inherits the brain's whole
    environment — a bearer token, a proxy or a provider switch sent turns elsewhere, silently."""
    _sub_env(monkeypatch)
    for key in _REROUTERS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(var, value)
    with pytest.raises(RuntimeError, match=var):
        subm.SubscriptionLLM("claude-opus-5-5")


def test_the_official_endpoint_and_off_switches_are_fine(monkeypatch):
    _sub_env(monkeypatch)
    for key in _REROUTERS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/")
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "0")
    assert subm.SubscriptionLLM("claude-opus-5-5")._model == "claude-opus-5-5"


def test_no_request_leaves_a_transcript_behind(monkeypatch):
    """01·F13: every SDK call persisted a full-prompt JSONL under ~/.claude/projects for 30 days."""
    _sub_env(monkeypatch)

    async def script(tools, m):
        yield m.AssistantMessage([m.TextBlock("ok")])
    sdk, seen = _capturing_sdk(script)
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", sdk)
    asyncio.run(_sub_session().turn("hi", tools=_tools(), context="ctx", emit=_quiet))
    assert seen and all(o.get("extra_args") == {"no-session-persistence": None} for o in seen), seen


def test_the_cli_never_runs_where_claude_code_memory_lives(monkeypatch):
    """Follow-ups 2026-09-11, item 5, found in the section-F smoke: the CLI keys its auto-memory
    on the working directory, so run from the repo it put the developer's own Claude Code project
    memory — and the username — into every brain request (measured against a local capture
    server). The transport must run it from a neutral directory, never under the home folder."""
    _sub_env(monkeypatch)

    async def script(tools, m):
        yield m.AssistantMessage([m.TextBlock("ok")])
    sdk, seen = _capturing_sdk(script)
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", sdk)
    asyncio.run(_sub_session().turn("hi", tools=_tools(), context="ctx", emit=_quiet))
    cwd = seen[0].get("cwd") if seen else None
    assert cwd, "no cwd: the CLI inherits the brain's, the repo"
    assert not str(Path(cwd).resolve()).startswith(str(Path(HOME).resolve())), cwd
    assert Path(cwd).is_dir(), cwd


def test_eval_provenance_names_the_cli_that_ran(monkeypatch):
    """01·F14: the SDK spawns its BUNDLED CLI, and no baseline said which one it measured."""
    monkeypatch.setenv("MIX_AGENT_LLM_BACKEND", "subscription")
    monkeypatch.setattr(subm, "cli_version", lambda: "2.1.177", raising=False)
    stamp = P.provenance()
    assert stamp.get("cli") == "2.1.177", stamp
    notes = P.provenance_notes({"meta": {**stamp, "cli": "2.1.100"}})
    assert any("CLI" in n and "2.1.100" in n for n in notes), notes


def test_a_huge_tool_result_is_cut_the_same_on_both_paths(monkeypatch):
    """01·F15: the CLI cut results over 25k tokens on the subscription only, pointing at
    pagination tools the agent does not have; on cloud the same call came back whole."""
    monkeypatch.setattr(chatmod.knowledge_deep, "lookup", lambda topic: "x" * 200_000)
    out = asyncio.run(chatmod.execute_tool(_tools(), "lookup_knowledge", {"topic": "eq"}))
    assert len(out.text) < 100_000, len(out.text)
    assert "cut at" in out.text and "200,000" in out.text
    # …and leaves an ordinary result alone
    monkeypatch.setattr(chatmod.knowledge_deep, "lookup", lambda topic: "short")
    assert asyncio.run(chatmod.execute_tool(_tools(), "lookup_knowledge", {"topic": "eq"})).text == "short"


def test_an_empty_api_reply_is_never_stored_empty(monkeypatch):
    """01·F16: an empty end_turn stored as `content: []` 400s every later turn — the API
    refuses an empty non-final assistant message."""
    monkeypatch.setenv("MIX_AGENT_LLM_BACKEND", "api")
    session = chatmod.ChatSession(client=_Client([_Final([], "end_turn")]),
                                  tool_defs_for=chatmod.daw_tool_defs)
    asyncio.run(session.turn("hm", tools=_tools(), context="ctx", emit=_quiet))
    last = session.history[-1]
    assert last["role"] == "assistant" and last["content"], last


# ------------------------------------------------------------------ privacy + cost

def _edit_with_home_paths() -> dict:
    return {"tracks": [{"name": "Vox",
                        "clips": [{"index": 0, "io": {"path": f"{HOME}/Music/proj/vox.wav"}}],
                        "sample_path": f"{HOME}/Samples/kick.wav"}],
            "projects": [{"dir": HOME}]}


def test_home_paths_never_reach_the_model(monkeypatch):
    """06·F8: absolute clip and sample paths (/Users/<name>/…) rode every turn to the model."""
    edit = _edit_with_home_paths()
    live = B._chat_context_live(edit)
    assert HOME not in live, live
    assert "~/Music/proj/vox.wav" in live and "~/Samples/kick.wav" in live
    assert HOME not in json.dumps(B.read_track_entry(edit, "Vox"))
    assert edit["tracks"][0]["sample_path"].startswith(HOME), "the DAW's own copy is untouched"


def test_the_side_channel_counts_its_spend_and_hides_home(tmp_path, monkeypatch):
    """03·F21 + 06·F8: a mid-run question is a real model call whose cost never reached the
    session total, and its prompt carried the raw edit. The fix is the `spent` parameter — so
    on the pre-fix code this leg fails on the call, not an assertion."""
    import orchestrator.agents as agents
    import orchestrator.llm_backend as llm_backend

    class _Agent:
        prompt = ""

        def __init__(self):
            self.usage = {"estimated_cost_usd": 0.0123}

        def chat(self, message, context=None):
            _Agent.prompt = context or ""
            return "two passes in"
    monkeypatch.setattr(agents, "ClaudeAgent", _Agent)
    monkeypatch.setattr(llm_backend, "llm_available", lambda: True)
    brain = S.Brain(_project(tmp_path))
    brain.live_run = {"kind": "run", "brief": "master it", "pass_number": 2}
    brain.latest_edit = _edit_with_home_paths()
    sent: list[dict] = []

    async def send(msg):
        sent.append(msg)
    asyncio.run(S._do_chat_status(send, brain, 7, "how's it going?"))
    assert sent and sent[-1].get("ok") and sent[-1].get("cost_usd") == 0.0123, sent
    assert HOME not in _Agent.prompt and "~/Samples/kick.wav" in _Agent.prompt


@pytest.mark.parametrize("backend,estimate", [("subscription", True), ("api", False)])
def test_the_hello_says_whether_spend_is_a_price(tmp_path, monkeypatch, backend, estimate):
    """03·F21: on the subscription the brain prices tokens at API rates, and the DAW showed that
    as money spent. It now says so once, in the hello, and the panel labels it "≈ API $"."""
    proj = _project(tmp_path)
    monkeypatch.setenv("MIX_AGENT_LLM_BACKEND", backend)

    async def scenario(hub, port):
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            writer.write(_line({"id": 0, "method": "auth", "params": {"token": hub.token}}))
            writer.write(_line({"id": 1, "method": "hello",
                                "params": {"project": str(proj), "role": "daw"}}))
            await writer.drain()
            return (await _read_until(reader, lambda m: m.get("id") == 1))["result"]
        finally:
            writer.close()
    result = _serve(proj, scenario)
    assert result.get("spend_is_estimate") is estimate, result


# ------------------------------------------------------------------ the brain process

def test_a_runs_file_survives_a_write_that_dies_midway(tmp_path, monkeypatch):
    """02·F15: runs/ files were rewritten in place; a brain killed mid-write (the 45 s cancel
    grace, the heartbeat watcher's os._exit) left "where we left off" truncated."""
    proj = tmp_path / "p"
    proj.mkdir()
    B.record_session_digest(proj, session_id="s1", user="first", reply="ok", commands=[])
    path = proj / "runs" / B.SESSION_DIGEST_FILE
    before = path.read_text()
    real = Path.write_text

    def dying(self, text, *a, **kw):
        # A fixed short cut, not "half": half of two similar lines IS the first line, and a
        # control that cut there passed on the pre-fix code too.
        real(self, text[:5], *a, **kw)
        raise OSError("killed mid-write")
    monkeypatch.setattr(Path, "write_text", dying)
    with pytest.raises(OSError):
        B.record_session_digest(proj, session_id="s1", user="second", reply="ok", commands=[])
    monkeypatch.setattr(Path, "write_text", real)
    assert path.read_text() == before


def test_reading_the_gesture_boundary_does_not_rewrite_session_json(tmp_path):
    """02·F15: every plain Session() persists in its constructor, and max_applied_seq — read at
    every gesture's edges — rewrote session.json each time."""
    proj = _project(tmp_path)
    before = (proj / "session.json").stat()
    B.max_applied_seq(proj)
    after = (proj / "session.json").stat()
    assert (before.st_ino, before.st_mtime_ns) == (after.st_ino, after.st_mtime_ns)


def test_an_unreadable_session_is_not_an_empty_project(tmp_path, monkeypatch, capsys):
    """02·F16: an unreadable session.json read as no stems, so Run and Auto Mix answered
    "there's no audio for me to mix yet" — and the context dropped the applied chains silently."""
    monkeypatch.setenv("MIX_AGENT_DATA_DIR", str(tmp_path / "data"))
    proj = _project(tmp_path)
    (proj / "session.json").write_text("{ not json")
    brain = S.Brain(proj)
    ran = brain.run("mix this")
    assert ran.get("stop_reason") == "session unreadable", ran
    assert "couldn't read" in ran["chat"] and "no audio" not in ran["chat"]
    mixed = brain.auto_mix("")
    assert "couldn't read" in mixed["chat"], mixed
    B._chat_context_stable(proj, "agent")
    assert "applied chains skipped" in capsys.readouterr().err


def test_a_second_daw_that_comes_and_goes_hands_the_path_back(tmp_path):
    """02·F14: a second declared client (a smoke, a tool) that connected and left cleared the
    operate slot, so the real DAW's next command got "no DAW attached"."""
    brain = S.Brain(_project(tmp_path))
    real, stray = object(), object()

    def real_invoke(*_a, **_k):
        return {"ok": True}

    def stray_invoke(*_a, **_k):
        return {"ok": True}
    brain.claim_operate(real, real_invoke, declared=False)   # the auth seed…
    brain.claim_operate(real, real_invoke, declared=True)    # …then hello {role: daw}
    brain.claim_operate(stray, stray_invoke, declared=True)
    assert brain.daw_invoke is stray_invoke
    brain.release_operate(stray)
    assert brain.daw_invoke is real_invoke, "the real DAW lost its operate path"
    brain.release_operate(real)
    assert brain.daw_invoke is None


# ------------------------------------------------------------------ runs + learning

def test_a_vanished_stem_reads_as_a_sentence(tmp_path):
    """05·F17: a stem deleted outside the app put a raw `[TextContent(…)]` repr in the status bar."""
    class _Failed:
        isError = True
        structuredContent = None

        def __init__(self, text):
            self.content = [types.SimpleNamespace(type="text", text=text)]

    class _Session:
        def __init__(self, text):
            self.text = text

        async def call_tool(self, name, args):
            return _Failed(self.text)

    missing = str(tmp_path / "drums.wav")
    with pytest.raises(RuntimeError) as err:
        anyio.run(EngineClient(_Session(
            f"Error executing tool auto_tag_stems: Error opening '{missing}': System error.")).call,
            "auto_tag_stems", {})
    text = str(err.value)
    assert "drums.wav is missing or unreadable" in text, text
    assert "namespace(" not in text and "TextContent" not in text and "[" not in text, text
    with pytest.raises(RuntimeError) as err:
        anyio.run(EngineClient(_Session("Error executing tool render: disk full")).call, "render", {})
    assert str(err.value) == "engine tool 'render' failed: disk full"


def test_a_move_rolled_back_as_a_regression_is_never_a_worked_example(tmp_path):
    """05·F16: the loop reverted a proposal for making the mix worse, then offered it to the
    next run as a model to imitate — exemplars rank on prediction error, not outcome."""
    record = {"proposal_id": "p0004", "rationale": "master is 0.2 LU from the goal",
              "actions": [{"type": "gain", "stem_id": "master", "delta_db": 0.2}],
              "targets": [{"metric": "master.lufs_i"}],
              "predicted_impact": [{"metric": "master.lufs_i", "predicted_delta": 0.2}],
              "measured_impact": [{"metric": "master.lufs_i", "measured_delta": 0.2}]}
    rollback = {"rollback": {"reverted_proposal_ids": ["p0004"], "kept_through_pass": 1,
                             "reason": "the last pass made it worse"}}
    log = tmp_path / "run.jsonl"
    log.write_text(json.dumps(record) + "\n")
    assert len(exemplars(load_records_file(log), ["master.lufs_i"])) == 1, "control: served"
    log.write_text(json.dumps(record) + "\n" + json.dumps(rollback) + "\n")
    records = load_records_file(log)
    assert len(records) == 1, "still calibration data — its prediction error is real"
    assert exemplars(records, ["master.lufs_i"]) == []


def test_a_move_the_user_took_back_is_never_a_worked_example(tmp_path):
    """Follow-ups 2026-09-11, item 3: the loop's own rollback was excluded (05·F16), but a
    user's ⌥⌘Z or "no, put that back" went to decisions.jsonl with no proposal id, so the move
    kept being served to the next run as a model to imitate."""
    from orchestrator.calibration import load_records, record_decision
    record = {"proposal_id": "p0004", "rationale": "brighten the vocal",
              "actions": [{"type": "gain", "stem_id": "vocals", "delta_db": 1.0}],
              "targets": [{"metric": "master.lufs_i"}],
              "predicted_impact": [{"metric": "master.lufs_i", "predicted_delta": 1.0}],
              "measured_impact": [{"metric": "master.lufs_i", "measured_delta": 1.0}]}

    def runs_with(*decisions) -> Path:
        runs = tmp_path / f"runs{len(list(tmp_path.iterdir()))}"
        runs.mkdir()
        (runs / "run1.jsonl").write_text(json.dumps(record) + "\n")
        for decision, proposal in decisions:
            record_decision(runs, decision, proposal, source="daw")
        return runs

    assert len(exemplars(load_records(runs_with()), ["master.lufs_i"])) == 1, "control: served"
    undone = runs_with(("undone", record))
    assert exemplars(load_records(undone), ["master.lufs_i"]) == []
    assert len(load_records(undone)) == 1, "still calibration data"
    assert exemplars(load_records(runs_with(("rejected", record))), ["master.lufs_i"]) == [], \
        "an A/B answer of A turns the move down too"
    other = dict(record, proposal_id="p0009", rationale="something else")
    assert len(exemplars(load_records(runs_with(("undone", other))), ["master.lufs_i"])) == 1, \
        "an undo of a DIFFERENT move leaves this one served"
    # a decision line written before ids were recorded still matches by action + rationale
    legacy = runs_with()
    (legacy / "decisions.jsonl").write_text(json.dumps(
        {"decision": "undone", "action_key": __import__("orchestrator.calibration", fromlist=["x"])
         .action_key(record["actions"]), "rationale": record["rationale"]}) + "\n")
    assert exemplars(load_records(legacy), ["master.lufs_i"]) == []


def test_no_home_path_in_any_tool_result():
    """Follow-ups 2026-09-11, item 4: P3-12 redacted the edit snapshot, but a tool RESULT — a DAW
    reply, an import, an error — still carried /Users/<name>/… to the model."""
    async def peek(command, args):
        return {"ok": True, "dir": f"{HOME}/Music/proj", "root": HOME}
    out = asyncio.run(chatmod.execute_tool(_tools(daw_command=peek), "daw_command",
                                           {"command": "project.peek", "args": {"dir": "~/Music/proj"}}))
    assert HOME not in out.text, out.text
    assert "~/Music/proj" in out.text and '"~"' in out.text and not out.error

    async def refused(command, args):
        return {"ok": False, "error": f"no such folder {HOME}/nowhere"}
    bad = asyncio.run(chatmod.execute_tool(_tools(daw_command=refused), "daw_command",
                                           {"command": "project.peek", "args": {}}))
    assert bad.error, "the error FLAG survives"
    assert HOME not in bad.text and "~/nowhere" in bad.text, bad.text
