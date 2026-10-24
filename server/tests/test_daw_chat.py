"""The DAW's resident engineer — routing by tool choice (2026-07-30).

The DAW used to route a user's message through a keyword table into the metric
compiler, with a stateless, tool-less 400-token reply as the failure fallback. It
now goes to the same persistent chat agent the web workspace has had since Phase
17, carrying the DAW's tool surface. These tests pin the WIRING — that each tool
the agent can choose actually does the right thing to the project — using a
scripted stand-in for the model, so they run with no LLM and no spend.

The ask-mode tests are the important ones: they pin the contract that Ask never
touches the project, and that "do it" applies the very steps the user saw rather
than re-deriving them (which is what the old advise-on-a-copy path did).
"""
import json
from pathlib import Path

import pytest

from orchestrator import daw_bridge


class ScriptedTurn:
    """Stands in for the model: calls the tools it was told to, in order, and
    "says" a line. Mirrors chat.ScriptedChat's philosophy (a deterministic twin)."""

    def __init__(self, calls, reply="done"):
        self._calls = calls          # [(tool_name, args), ...]
        self._reply = reply
        self.results = []

    async def turn(self, message, *, tools, context, emit, during_run=False,
                   should_cancel=None, live_context=None):
        from orchestrator.chat import execute_tool

        self.context = context
        self.should_cancel = should_cancel
        # the live half is re-read per round in production; capture it once here
        self.live_context = live_context
        if live_context is not None:
            self.context = context + "\n\n" + live_context()
        for name, args in self._calls:
            self.results.append(await execute_tool(tools, name, args))
        await emit({"type": "chat_message", "text": self._reply})


@pytest.fixture()
def scripted(monkeypatch):
    """Install a scripted turn as the project's chat session."""
    def install(project: Path, calls, reply="done"):
        turn = ScriptedTurn(calls, reply)
        daw_bridge._CHAT_SESSIONS[str(project)] = turn
        return turn
    yield install
    daw_bridge._CHAT_SESSIONS.clear()


def _session_state(project: Path) -> dict:
    path = project / "session.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _register(project: Path):
    """Load the project so its stems are discovered + persisted."""
    from mix_engine.session import Session
    Session(project)   # _discover_stems writes session.json
    return sorted(_session_state(project).get("stems", {}))


# ---------- agent mode: the tools actually act ----------

def test_track_scoped_move_applies_to_that_stem(demo_project, scripted):
    stems = _register(demo_project)
    assert stems, "demo project should register its stems"
    target = stems[0]
    scripted(demo_project, [("propose", {
        "rationale": "more air on the named track",
        "predicted_impact": [{"metric": f"stem({target}).lufs_s",
                              "predicted_delta": 0.0, "unit": "dB"}],
        "actions": [{"type": "eq", "stem_id": target,
                     "band": {"freq_hz": 10000, "gain_db": 2.0, "q": 0.7,
                              "shape": "high_shelf"}}]})])

    out = daw_bridge.chat_turn(demo_project, "give it more air", mode="agent")

    assert out["reply"] == "done"
    assert out["moves"], "an applied move should be reported back"
    assert out["moves"][0]["stem"] == target, "the move must stay on the named stem"
    chain = _session_state(demo_project)["stems"][target].get("chain", [])
    assert any(e.get("type") == "eq" for e in chain), "the eq should be applied"


def test_a_full_pass_budget_self_heals_instead_of_bouncing(demo_project, scripted):
    """Tonight's 'proposal cap stuck' failure (2026-08-09): once 3 proposals had
    landed in a pass, every further chat proposal bounced with proposals_per_pass
    and the agent resubmitted the identical rejected set forever. The DAW path
    now heals like gateway._chat_propose: advance the pass and retry once."""
    from mix_engine.models import ProposalRequest
    from mix_engine.session import Session

    stems = _register(demo_project)
    target = stems[0]
    session = Session(demo_project)
    for i in range(3):
        result = session.propose(ProposalRequest.model_validate({
            "rationale": f"filler {i}", "predicted_impact": [],
            "actions": [{"type": "gain", "stem_id": target, "delta_db": 0.1}]}))
        assert result.accepted, "budget-filling proposals must land"
    pass_before = session.pass_number

    turn = scripted(demo_project, [("propose", {
        "rationale": "one more move after the budget is gone",
        "predicted_impact": [],
        "actions": [{"type": "eq", "stem_id": target,
                     "band": {"freq_hz": 3000, "gain_db": -1.5, "q": 1.0,
                              "shape": "peak"}}]})])
    out = daw_bridge.chat_turn(demo_project, "cut the harshness", mode="agent")

    assert not turn.results[-1].error, turn.results[-1].text
    assert "applied" in turn.results[-1].text, turn.results[-1].text
    assert out["moves"], "the move must land, not bounce on a stale pass budget"
    assert _session_state(demo_project)["pass_number"] > pass_before


def test_an_identical_move_is_not_applied_twice_in_one_turn(demo_project, scripted):
    """2026-08-09: within one turn nothing is re-measured, so an identical
    re-propose is the model looping — tonight it applied the same duck/tune/eq
    trio three times over. The repeat is refused with a pointed message; across
    turns the same set stays legal (measured iteration is how the deterministic
    agent converges)."""
    stems = _register(demo_project)
    target = stems[0]
    move = {"rationale": "cut the mud", "predicted_impact": [],
            "actions": [{"type": "eq", "stem_id": target,
                         "band": {"freq_hz": 250, "gain_db": -2.0, "q": 1.0,
                                  "shape": "peak"}}]}
    turn = scripted(demo_project, [("propose", dict(move)),
                                   ("propose", dict(move))])
    out = daw_bridge.chat_turn(demo_project, "cut the mud", mode="agent")

    assert "skipped" in turn.results[-1].text, turn.results[-1].text
    assert len(out["moves"]) == 1, "the identical repeat must not land"
    chain = _session_state(demo_project)["stems"][target].get("chain", [])
    assert len([e for e in chain if e.get("type") == "eq"]) == 1


def test_daw_command_reaches_the_daw(demo_project, scripted):
    _register(demo_project)
    seen = []

    def fake_invoke(command, args):
        seen.append((command, args))
        return {"ok": True}

    scripted(demo_project, [("daw_command",
                             {"command": "transport.seek", "args": {"position_s": 12}})])
    out = daw_bridge.chat_turn(demo_project, "jump to 12 seconds", mode="agent",
                               daw_invoke=fake_invoke)

    assert seen == [("transport.seek", {"position_s": 12})]
    assert out["reply"] == "done"


def test_daw_command_track_add_registers_in_turn(demo_project, scripted, tmp_path):
    """The agent's own `daw_command track.add {path}` (advertised in its prompt) is
    the second way it adds a track; it must take the import_sample step — the DAW
    asked for `register: false`, the stem registered before the tool returns —
    not the bare invoke that left registration to a RunThread the turn was holding
    (2026-09-04)."""
    _register(demo_project)
    wav = tmp_path / "loop.wav"
    wav.write_bytes(b"RIFF....WAVEfmt ")
    log = []

    def fake_invoke(command, args):
        log.append(("invoke", command, dict(args)))
        return {"ok": True, "track": "loop", "registered": False}

    def register_stem(path, stem_id, bus):
        log.append(("register", stem_id))
        return {"ok": True, "stem_id": stem_id, "wav": "loop.wav"}

    turn = scripted(demo_project, [("daw_command",
                                    {"command": "track.add", "args": {"path": str(wav)}})])
    daw_bridge.chat_turn(demo_project, "add the loop", mode="agent",
                         daw_invoke=fake_invoke, register_stem=register_stem)
    assert log == [("invoke", "track.add", {"path": str(wav), "register": False}),
                   ("register", "loop")]
    assert turn.results[0].error is not True, turn.results[0].text
    assert "loop" in turn.results[0].text


def test_a_refused_command_is_reported_as_an_error(demo_project, scripted):
    """A confirm-gated or unknown command comes back {ok: false}. The agent must
    see that as a tool ERROR, not narrate a success it never got."""
    _register(demo_project)
    turn = scripted(demo_project, [("daw_command", {"command": "edit.delete", "args": {}})])
    daw_bridge.chat_turn(demo_project, "delete this edit", mode="agent",
                         daw_invoke=lambda c, a: {"ok": False,
                                                  "error": "needs the user to confirm it"})
    assert turn.results[0].error is True
    assert "confirm" in turn.results[0].text


def test_no_daw_attached_is_an_honest_error(demo_project, scripted):
    _register(demo_project)
    turn = scripted(demo_project, [("daw_command", {"command": "transport.play_pause"})])
    daw_bridge.chat_turn(demo_project, "play", mode="agent", daw_invoke=None)
    assert turn.results[0].error is True


# ---------- ask mode: plan, never touch ----------

def test_ask_mode_records_a_plan_and_changes_nothing(demo_project, scripted):
    stems = _register(demo_project)
    target = stems[0]
    before = json.dumps(_session_state(demo_project), sort_keys=True)
    invoked = []

    scripted(demo_project, [
        ("propose", {"rationale": "r", "predicted_impact": [],
                     "actions": [{"type": "gain", "stem_id": target, "delta_db": -1.0}]}),
        ("daw_command", {"command": "transport.seek", "args": {"position_s": 4}}),
    ])
    out = daw_bridge.chat_turn(demo_project, "turn it down and jump to the drop",
                               mode="ask",
                               daw_invoke=lambda c, a: invoked.append(c))

    plan = out["plan"]
    assert [s["kind"] for s in plan["steps"]] == ["proposal", "command"]
    assert not invoked, "ask mode must not touch the DAW"
    assert json.dumps(_session_state(demo_project), sort_keys=True) == before, \
        "ask mode must not touch the project"
    assert not out["moves"]


def test_apply_plan_runs_the_steps_the_user_approved(demo_project, scripted):
    """The old Ask re-ran the brief from scratch on 'do it', so the preview and the
    execution were two independent derivations. apply_plan executes the recorded
    steps verbatim."""
    stems = _register(demo_project)
    target = stems[0]
    invoked = []
    scripted(demo_project, [
        ("propose", {"rationale": "r", "predicted_impact": [],
                     "actions": [{"type": "gain", "stem_id": target, "delta_db": -1.0}]}),
        ("daw_command", {"command": "transport.seek", "args": {"position_s": 4}}),
    ])
    plan = daw_bridge.chat_turn(demo_project, "do stuff", mode="ask",
                                daw_invoke=lambda c, a: {"ok": True})["plan"]

    result = daw_bridge.apply_plan(demo_project, plan,
                                   daw_invoke=lambda c, a: (invoked.append(c) or {"ok": True}))

    assert result["ok"], result
    assert invoked == ["transport.seek"], "the recorded command must run verbatim"
    # a gain action moves the stem's fader scalar, not its device chain
    assert _session_state(demo_project)["stems"][target]["gain_db"] == pytest.approx(-1.0)


def test_apply_plan_reports_a_failed_step_instead_of_swallowing_it(demo_project):
    _register(demo_project)
    plan = {"narration": "", "steps": [
        {"kind": "command", "payload": {"command": "does.not.exist", "args": {}}},
        {"kind": "nonsense", "payload": {}},
    ]}
    result = daw_bridge.apply_plan(
        demo_project, plan,
        daw_invoke=lambda c, a: {"ok": False, "error": "unknown command"})
    assert result["ok"] is False
    assert result["failed"] == 2
    assert [r["kind"] for r in result["results"]] == ["command", "nonsense"]


# ---------- context ----------

def test_the_engineer_sees_the_session_and_the_command_catalog(demo_project, scripted):
    _register(demo_project)
    turn = scripted(demo_project, [])
    daw_bridge.chat_turn(
        demo_project, "what is going on", mode="agent",
        edit_state={"tracks": [{"name": "Snare Top"}],
                    "daw_commands": [{"key": "clip.split", "enabled": True}]})

    assert "Snare Top" in turn.context, "the live edit must reach the agent"
    assert "clip.split" in turn.context, "the command catalog must reach the agent"
    assert "agent" in turn.context.lower()


def test_history_persists_across_turns(demo_project):
    """'a bit less' only works if the conversation is the same object next turn."""
    from orchestrator.chat import ChatSession

    first = daw_bridge._chat_session(demo_project)
    second = daw_bridge._chat_session(demo_project)
    assert first is second
    assert isinstance(first, ChatSession)
    daw_bridge.reset_chat_session(demo_project)
    assert daw_bridge._chat_session(demo_project) is not first


# ---------- proactive observations + teaching ----------

def test_engine_observations_reach_the_agent(demo_project, scripted, monkeypatch):
    """The engine computes deterministic findings for the DAW's console; until
    2026-07-30 the agent never saw them, so it could only answer what it was
    asked. An engineer that volunteers "the kick and bass are fighting" is a
    different product from one that waits."""
    _register(demo_project)
    monkeypatch.setattr(daw_bridge, "session_observations", lambda _p, **k: [
        {"text": "bass is crowding vocals between 200 and 400 Hz",
         "verb": "untangle it", "brief": "resolve the masking"}])
    turn = scripted(demo_project, [])
    daw_bridge.chat_turn(demo_project, "how's it sounding", mode="agent")

    assert "crowding" in turn.context, "the engine's findings must reach the agent"
    assert "did NOT have to be asked" in turn.context, \
        "the agent must be told these are unprompted observations, not answers"


def test_stem_tags_and_the_return_topology_reach_the_agent(demo_project, scripted, monkeypatch):
    """The brain tags stems at import and owns the returns, and until 2026-09-02 told
    the chat neither: "the pad" had to be inferred from a name, and a return was
    visible only through its chain. Both ride the stable context now, as their own
    labelled parts — and an untagged project with no returns sends neither."""
    import json as _json
    _register(demo_project)
    turn = scripted(demo_project, [])
    daw_bridge.chat_turn(demo_project, "hello", mode="agent")
    assert "Stem tags (auto_tag_stems" not in turn.context, "no tags: no part"
    assert "Send/return topology" not in turn.context, "no returns: no part"

    state_path = demo_project / "session.json"
    state = _json.loads(state_path.read_text())
    stems = state["stems"]
    first = sorted(stems)[0]
    stems[first]["tags"] = ["vocal"]
    state_path.write_text(_json.dumps(state))
    monkeypatch.setattr(daw_bridge, "returns_topology", lambda _s: {
        "returns": [{"id": "plate", "kind": "return_reverb"}],
        "sends": {first: {"plate": -12.0}}})
    turn = scripted(demo_project, [])
    daw_bridge.chat_turn(demo_project, "hello again", mode="agent")
    assert f'Stem tags (auto_tag_stems, stem -> tags): {{"{first}": ["vocal"]}}' in turn.context
    assert 'Send/return topology (send_returns): {"returns": [{"id": "plate"' in turn.context


def test_a_failing_observation_never_blocks_the_turn(demo_project, scripted, monkeypatch):
    _register(demo_project)

    def boom(*_a, **_k):
        raise RuntimeError("analysis exploded")

    monkeypatch.setattr(daw_bridge, "session_observations", boom)
    scripted(demo_project, [])
    out = daw_bridge.chat_turn(demo_project, "hello", mode="agent")
    assert out["reply"] == "done", "observations are a bonus, not a blocker"


def test_the_daw_prompt_asks_for_explanation_and_noticing():
    from orchestrator.chat import DAW_SYSTEM_PROMPT

    assert "Say what you notice" in DAW_SYSTEM_PROMPT
    assert "Explain the why" in DAW_SYSTEM_PROMPT
    # the knowledge pack is what makes "why" answerable with real craft
    from orchestrator.knowledge import KNOWLEDGE_BLOCK
    assert "producers say" in KNOWLEDGE_BLOCK


# ---------- the command catalog reaches the agent (2026-08-05 audit) ----------

def test_the_command_catalog_survives_snapshot_truncation(tmp_path):
    """EditSnapshot.h sets the catalog as `commands`; the bridge used to pop
    `daw_commands`, so the catalog stayed in the snapshot dict, was serialized
    last, and got eaten by the 24k truncation. The prompt meanwhile calls it the
    definitive "what can I do here" — so the agent invented keys instead."""
    from orchestrator.daw_bridge import _chat_context

    project = tmp_path / "proj"
    project.mkdir()
    # A snapshot big enough that truncation definitely bites.
    edit_state = {
        "tracks": [{"name": f"track_{i}", "clips": ["x" * 200]} for i in range(200)],
        "commands": [{"key": "vca.create", "name": "Create VCA", "wants_args": True},
                     {"key": "clip.split", "name": "Split", "wants_args": True}],
    }

    context = _chat_context(project, edit_state, mode="agent")

    # 2026-09-01: the snapshot is compacted FIELD-WISE (orchestrator/edit_state.py) — a dropped
    # field is named in `omitted`, never a character cut with a marker.
    assert '"omitted"' in context, "this fixture must exercise the compaction"
    assert "vca.create" in context, "the catalog must reach the agent verbatim"
    assert "clip.split" in context


def test_the_planner_spelling_of_the_catalog_also_works(tmp_path):
    """agents.py renames it to `daw_commands`; accept either producer."""
    from orchestrator.daw_bridge import _chat_context

    project = tmp_path / "proj"
    project.mkdir()
    context = _chat_context(
        project,
        {"tracks": [], "daw_commands": [{"key": "transport.play"}]},
        mode="agent")
    assert "transport.play" in context


# ---------- cancellation reaches a chat-started run (2026-08-05 audit) ----------

def test_a_run_the_agent_starts_from_chat_is_cancellable(demo_project, monkeypatch):
    """The DAW's Cancel used to be registered only for `method == "run"`, while
    chat is the front door and a chat turn can start a full mix run inside
    itself. The user pressed Stop, got `cancelled: 0`, and watched the run bill
    to completion behind a spinner."""
    import orchestrator.daw_bridge as bridge

    seen = {}

    def fake_run_agent_pass(project_dir, brief, **kw):
        seen["should_cancel"] = kw.get("should_cancel")
        return {"stop_reason": "cancelled", "passes": 0, "cost_usd": 0.0,
                "budget_usd": None}

    monkeypatch.setattr(bridge, "run_agent_pass", fake_run_agent_pass)
    monkeypatch.setattr(bridge, "_chat_session",
                        lambda _p, _sid="": ScriptedTurn([("start_run",
                                                           {"brief": "master it"})]))

    cancelled = lambda: True
    bridge.chat_turn(demo_project, "master it", use_llm=True,
                     should_cancel=cancelled)

    assert seen["should_cancel"] is cancelled, (
        "the loop must receive the cancel token, or Stop does nothing")


def test_the_turn_stops_between_tool_rounds_when_cancelled():
    """`should_cancel` is polled per round, so a cancelled turn stops before
    paying for the next one."""
    from orchestrator.chat import ChatSession

    session = ChatSession(client=object(), tool_defs_for=lambda _d: [])
    events = []

    async def emit(event):
        events.append(event)

    import anyio
    anyio.run(lambda: session.turn("hello", tools=None, context="", emit=emit,
                                   should_cancel=lambda: True))

    said = " ".join(e.get("text", "") for e in events
                    if e["type"] == "chat_message")
    assert "Stopped" in said


# ---------- undo / render / notes are real in the DAW (2026-08-05 audit) ----------

def test_the_agent_can_take_back_its_own_move(demo_project, scripted):
    """"No, put that back" was unreachable by ANY path mid-turn: undo_proposal was
    wired to `unsupported`, and daw_command("agent.undo") is gated busy because a
    pass is active for the whole turn."""
    stems = _register(demo_project)
    target = stems[0]
    turn = scripted(demo_project, [
        ("propose", {"rationale": "pull it down", "predicted_impact": [],
                     "actions": [{"type": "gain", "stem_id": target, "delta_db": -3.0}]}),
    ])
    daw_bridge.chat_turn(demo_project, "pull the vocal down", mode="agent")
    assert _session_state(demo_project)["stems"][target]["gain_db"] == pytest.approx(-3.0)

    # now the user says "no, put that back"
    turn2 = scripted(demo_project, [("undo_proposal", {"proposal_id": ""})])
    out = daw_bridge.chat_turn(demo_project, "no, put that back", mode="agent")

    assert turn2.results[0].error is not True, turn2.results[0].text
    assert "undone" in turn2.results[0].text
    assert _session_state(demo_project)["stems"][target]["gain_db"] == pytest.approx(0.0)


def test_a_refused_undo_says_why(demo_project, scripted):
    """The engine refuses two different ways — unknown id, and out-of-order (it only
    undoes newest-first). Those need different responses from the agent, so the
    reason is carried back rather than flattened into a bare failure."""
    stems = _register(demo_project)
    target = stems[0]

    turn = scripted(demo_project, [("undo_proposal", {"proposal_id": "p-does-not-exist"})])
    daw_bridge.chat_turn(demo_project, "undo that", mode="agent")
    assert "no such proposal" in turn.results[0].text

    # apply two, then try to undo the OLDER one — the out-of-order refusal
    scripted(demo_project, [
        ("propose", {"rationale": "a", "predicted_impact": [],
                     "actions": [{"type": "gain", "stem_id": target, "delta_db": -1.0}]}),
    ])
    daw_bridge.chat_turn(demo_project, "one", mode="agent")
    first = _session_state(demo_project)["proposals"][0]["proposal_id"]
    scripted(demo_project, [
        ("propose", {"rationale": "b", "predicted_impact": [],
                     "actions": [{"type": "gain", "stem_id": target, "delta_db": -1.0}]}),
    ])
    daw_bridge.chat_turn(demo_project, "two", mode="agent")

    turn3 = scripted(demo_project, [("undo_proposal", {"proposal_id": first})])
    daw_bridge.chat_turn(demo_project, "undo the first one", mode="agent")
    assert "nothing undone" in turn3.results[0].text


def test_render_gives_the_agent_fresh_numbers(demo_project, scripted):
    """Its context numbers are frozen from before its own changes, so without this it
    narrated results it never measured."""
    _register(demo_project)
    turn = scripted(demo_project, [("render", {})])
    daw_bridge.chat_turn(demo_project, "how does it measure now", mode="agent")
    assert turn.results[0].error is not True, turn.results[0].text
    assert "lufs_i" in turn.results[0].text


def test_a_saved_note_lands_where_the_readers_look(demo_project, scripted):
    """It wrote a notes.json ARRAY; every reader uses notes.jsonl with {"note": ...}
    lines — so "remember I always want the vocal bright" was never seen again."""
    from orchestrator.loop import _load_notes

    _register(demo_project)
    scripted(demo_project, [("save_note", {"note": "always keep the vocal bright"})])
    daw_bridge.chat_turn(demo_project, "remember: keep the vocal bright", mode="agent")

    assert _load_notes(demo_project / "runs") == ["always keep the vocal bright"]


def test_the_daw_does_not_advertise_a_tool_it_refuses():
    """stop_run was offered and wired to a refusal — the model burned a call on it."""
    from orchestrator.chat import daw_tool_defs

    names = {d["name"] for d in daw_tool_defs(False)}
    assert "stop_run" not in names
    assert {"undo_proposal", "render"} <= names


# ---------- the model picker actually picks (2026-08-05 audit) ----------

def test_the_picker_reaches_the_session_the_user_talks_to(monkeypatch, tmp_path):
    """set_llm validated the choice and stored it in env — which only ClaudeAgent
    read. So picking a model changed the RUN loop and left the chat agent (the
    thing the user actually types at) pinned to its default."""
    from orchestrator import daw_bridge as bridge
    from orchestrator.chat import MODEL

    bridge._CHAT_SESSIONS.clear()
    # A model that is NOT the hard default, or this proves nothing.
    monkeypatch.setenv("MIX_AGENT_MODEL", "claude-sonnet-5-5")
    monkeypatch.setenv("MIX_AGENT_EFFORT", "medium")

    session = bridge._chat_session(tmp_path)
    assert session.settings == {"model": "claude-sonnet-5-5", "reasoning": "medium"}
    assert session.settings["model"] != MODEL, "should not be the hard default"
    bridge._CHAT_SESSIONS.clear()


def test_a_model_pick_reaches_a_live_conversation_through_its_own_requests(monkeypatch, tmp_path):
    """Switching model mid-conversation takes effect on the next turn — through the pick that
    rides EVERY chat request (`_apply_turn_llm`), not through set_llm.

    set_llm used to re-point every live session of every project in the brain (the old
    `apply_chat_settings`). One brain serves every open project, so a pick in project B
    switched project A's conversation mid-turn — with the model picker open, a switch between
    models inside another conversation (2026-09-22 audit E·F4). set_llm now only sets the
    fallback env; this pins both halves."""
    import os

    from orchestrator import daw_bridge as bridge
    from orchestrator.daw_server import Brain

    bridge._CHAT_SESSIONS.clear()
    monkeypatch.delenv("MIX_AGENT_MODEL", raising=False)
    monkeypatch.delenv("MIX_AGENT_EFFORT", raising=False)
    a = bridge._chat_session(tmp_path / "a")
    b = bridge._chat_session(tmp_path / "b")
    before = dict(a.settings)

    brain = Brain.__new__(Brain)          # set_llm touches no project state
    res = brain.set_llm("claude-sonnet-5-5", "high")
    assert res["ok"] and os.environ["MIX_AGENT_MODEL"] == "claude-sonnet-5-5"
    assert a.settings == before and b.settings == before, \
        "set_llm must not re-point live conversations (any project's)"

    bridge._apply_turn_llm(b, {"model": "claude-opus-5-5", "reasoning": "xhigh"})
    assert b.settings == {"model": "claude-opus-5-5", "reasoning": "xhigh"}
    assert a.settings == before, "one project's pick never touches another's conversation"
    bridge._CHAT_SESSIONS.clear()


def test_a_bad_stored_model_does_not_brick_chat(monkeypatch, tmp_path):
    """A malformed env value must degrade to the default, not raise on every turn."""
    from orchestrator import daw_bridge as bridge
    from orchestrator.chat import MODEL

    bridge._CHAT_SESSIONS.clear()
    monkeypatch.setenv("MIX_AGENT_MODEL", "claude-not-a-model")
    monkeypatch.setenv("MIX_AGENT_EFFORT", "high")

    session = bridge._chat_session(tmp_path)
    assert session.settings["model"] == MODEL
    bridge._CHAT_SESSIONS.clear()


# ---------- the snapshot is fresh per round, not per turn (2026-08-05 audit) ----------

class RoundCountingTurn:
    """Re-reads the live half on each simulated round, like the real turn loop."""

    def __init__(self, rounds):
        self._rounds = rounds
        self.seen = []

    async def turn(self, message, *, tools, context, emit, during_run=False,
                   should_cancel=None, live_context=None):
        for _ in range(self._rounds):
            self.seen.append(live_context() if live_context else "")
        await emit({"type": "chat_message", "text": "done"})


def test_later_rounds_see_the_edit_as_the_agents_own_commands_left_it(demo_project,
                                                                     monkeypatch):
    """After clip.split the agent addressed clips by an index into the PRE-split
    list and got ok:true for editing the wrong one. The edit_state is now a getter,
    re-read per round, and the DAW pushes a fresh snapshot before replying."""
    _register(demo_project)
    turn = RoundCountingTurn(rounds=3)
    monkeypatch.setitem(daw_bridge._CHAT_SESSIONS, str(demo_project), turn)

    # a mutable "live" edit — stands in for what the DAW keeps pushing
    live = {"tracks": [{"name": "drums", "clips": [{"index": 0}]}]}

    def bump(_c=[0]):
        _c[0] += 1
        live["tracks"][0]["clips"] = [{"index": i} for i in range(_c[0])]
        return live

    daw_bridge.chat_turn(demo_project, "split it", mode="agent",
                         edit_state=bump, use_llm=True)

    daw_bridge._CHAT_SESSIONS.clear()
    assert len(turn.seen) == 3
    assert turn.seen[0] != turn.seen[1] != turn.seen[2], \
        "each round must see the edit as it is NOW, not as it was at turn start"


def test_the_stable_half_never_carries_the_live_edit(demo_project):
    """The catalog (~28k chars) sits behind a cache breakpoint, so it must not move
    when the edit does — otherwise refreshing per round re-sends it uncached."""
    _register(demo_project)
    # The engine's findings are computed on a worker since Phase 55 and join the context when
    # they are ready: the first build starts it, and the block is stable from then on.
    daw_bridge._chat_context_stable(demo_project, "agent", None, {"commands": []})
    assert daw_bridge.wait_for_observations(demo_project)
    a = daw_bridge._chat_context_stable(demo_project, "agent", None,
                                        {"commands": [{"key": "clip.split"}]})
    b = daw_bridge._chat_context_stable(demo_project, "agent", None,
                                        {"commands": [{"key": "clip.split"}]})
    assert a == b, "the cached half must be byte-stable across rounds"
    assert "clip.split" in a, "the catalog belongs in the cached half"
    assert "daw_edit_state" not in a, "the live edit must not be in the cached half"


def test_the_live_half_is_only_the_edit(demo_project):
    live = daw_bridge._chat_context_live({"tracks": [], "commands": [{"key": "x"}]})
    assert "daw_edit_state" in live
    assert "\"key\"" not in live, "the catalog must not ride the uncached half"


# ---------- the dev backend gets the live edit too (2026-08-12) ----------

class _StubSub:
    """Stands in for SubscriptionLLM: records the system prompt it was handed,
    runs one tool call, and replies. The SDK owns the loop in production, so the
    system prompt is fixed for the turn — which is exactly what this pins."""

    def __init__(self):
        self.system = ""
        self.results = []

    async def chat_turn(self, *, system, prompt, tool_defs, execute,
                        on_delta, on_tool, max_turns, effort=None, should_stop=None,
                        max_tokens=None, timer=None):
        self.system = system
        self.prompt = prompt
        self.effort = effort
        outcome = await execute("daw_command", {"command": "clip.split", "args": {}})
        self.results.append(outcome.text)
        return "done", {}


def _sub_session(monkeypatch):
    from orchestrator import chat as chatmod

    monkeypatch.setenv("MIX_AGENT_LLM_BACKEND", "subscription")
    session = chatmod.ChatSession(client=object())
    session._sub = _StubSub()
    return session, chatmod


def _noop_tools(chatmod, daw_command=None):
    async def unsupported(*_a, **_k):
        return "unsupported"

    return chatmod.ChatTools(
        engine_call=unsupported, propose=unsupported, start_run=unsupported,
        stop_run=unsupported, auto_mix=unsupported, file_instruction=unsupported,
        save_note=unsupported, workspace_command=unsupported, list_files=unsupported,
        import_sample=unsupported, undo_proposal=unsupported, render=unsupported,
        daw_command=daw_command or unsupported)


def test_the_dev_backend_sees_the_live_edit(monkeypatch):
    """The context split by lifetime (2026-08-05) handed this backend the STABLE
    half only, so on `dawdev` the agent lost the whole live edit and answered
    "the project has no tracks" with four of them on screen."""
    import anyio

    session, chatmod = _sub_session(monkeypatch)
    live = daw_bridge._chat_context_live({"tracks": [{"name": "Audio 1"}]})

    async def emit(_e):
        return None

    anyio.run(lambda: session.turn(
        "mix my song", tools=_noop_tools(chatmod), context="STABLE HALF",
        emit=emit, live_context=lambda: live))

    # the api path's layout since 2026-09-22 (audit D·F4): the turn's context and the live
    # edit ride AFTER the conversation, labelled as the app's — never the system prompt
    assert "STABLE HALF" in session._sub.prompt
    assert "Audio 1" in session._sub.prompt, "the live edit must reach the dev backend"
    assert "Audio 1" not in session._sub.system and "STABLE HALF" not in session._sub.system
    assert session._sub.prompt.index("mix my song") < session._sub.prompt.index("STABLE HALF")
    assert chatmod.LIVE_NOTE.strip() in session._sub.prompt


def test_the_dev_backends_tool_result_carries_the_moved_edit(monkeypatch):
    """The SDK's system prompt is fixed for the turn, so the refresh the API loop
    does per round has to ride the tool RESULT — otherwise the agent addresses
    clips by an index into the list from before its own clip.split."""
    import anyio

    session, chatmod = _sub_session(monkeypatch)
    states = [daw_bridge._chat_context_live({"tracks": [{"clips": [{"index": 0}]}]}),
              daw_bridge._chat_context_live({"tracks": [{"clips": [{"index": 0},
                                                                   {"index": 1}]}]})]

    async def emit(_e):
        return None

    anyio.run(lambda: session.turn(
        "split it", tools=_noop_tools(chatmod), context="STABLE HALF",
        emit=emit, live_context=lambda: states.pop(0) if states else ""))

    assert "\"index\":1" in session._sub.results[0].replace(" ", ""), \
        "a command that moved the edit must hand the new snapshot back"


# ---------- a run started from chat shows its progress and counts its cost (2026-09-10) ----------

def test_a_chat_started_run_feeds_the_panel_and_counts_its_cost(demo_project, scripted,
                                                                monkeypatch):
    """Agent audit P1-22: `start_run` from chat — the only documented run door — ran with no
    on_move/on_phase, so the panel showed no pass N/M, no moves and no $, and chat_done
    reported $0.00 for a $2.50 run. The run's result reached only the model."""
    _register(demo_project)

    def fake_pass(project_dir, brief, *, on_move=None, on_phase=None, **_kw):
        on_phase({"pass_number": 1, "max_passes": 2})
        on_move({"stem": "master", "type": "limiter", "params": {}, "rationale": "ceiling"})
        on_phase({"cost_usd": 2.5})
        return {"stop_reason": "targets met", "passes": 2, "cost_usd": 2.5,
                "budget_usd": None}
    monkeypatch.setattr(daw_bridge, "run_agent_pass", fake_pass)
    scripted(demo_project, [("start_run", {"brief": "master this for streaming",
                                           "gating": "", "max_passes": 2})])
    events: list = []
    out = daw_bridge.chat_turn(demo_project, "master this for streaming", mode="agent",
                               on_event=events.append)
    progress = [e for e in events if e.get("type") == "run_progress"]
    assert any(e.get("pass_number") == 1 for e in progress), "pass N/M reaches the panel"
    assert any(e.get("moves") for e in progress), "each move ticks in as it lands"
    assert out["cost_usd"] >= 2.5, "the run's spend is the turn's spend"
    assert out["passes"] == 2 and out["stop_reason"] == "targets met"


# ---------- C3: Auto Mix says what it actually did (2026-09-11 pre-merge review) ----------

def _automix_reply(project, scripted, monkeypatch, *, applied, refusals, cancelled=False):
    """Drive the chat tool's auto_mix and return the text the MODEL is handed."""
    def fake_pass(proj, target="", should_cancel=None, on_stage=None, refused=None,
                  notes=None, left_out=None):
        if refused is not None:
            refused.extend(refusals)
        return applied

    monkeypatch.setattr(daw_bridge, "auto_mix_pass", fake_pass)
    turn = scripted(project, [("auto_mix", {"target": "master for streaming"})])
    # A Stop MID-PASS, not before it: cancelled-at-the-start is a different (already honest)
    # answer, so the stand-in only goes true once Auto Mix is under way.
    started = {"n": 0}

    def stop_part_way() -> bool:
        started["n"] += 1
        return started["n"] > 1

    daw_bridge.chat_turn(project, "auto mix it", mode="agent", use_llm=False,
                         should_cancel=stop_part_way if cancelled else None)
    return turn.results[-1].text


def test_auto_mix_reports_what_it_applied_and_what_was_refused(demo_project, scripted,
                                                               monkeypatch):
    """It returned the constant "Auto Mix applied (undoable)" whatever happened: refused
    bundles were never asked for, a pass that applied NOTHING read as applied, and a Stop
    part-way reported success — the agent telling the user something untrue."""
    clean = _automix_reply(demo_project, scripted, monkeypatch, applied=3, refusals=[])
    assert "3 bundles" in clean and "refused" not in clean, clean

    refused = _automix_reply(demo_project, scripted, monkeypatch, applied=1,
                             refusals=["vocal lift (+7.5 dB over the pass rail)"])
    assert "1 bundle (undoable)" in refused and "+7.5 dB" in refused, refused

    nothing = _automix_reply(demo_project, scripted, monkeypatch, applied=0,
                             refusals=["vocal lift (+7.5 dB over the pass rail)"])
    assert "applied nothing" in nothing and "+7.5 dB" in nothing, nothing


def test_an_auto_mix_stopped_part_way_does_not_read_as_success(demo_project, scripted,
                                                               monkeypatch):
    stopped = _automix_reply(demo_project, scripted, monkeypatch, applied=2, refusals=[],
                             cancelled=True)
    assert "stopped part-way" in stopped and "still applied" in stopped, stopped
