"""Two open findings of the 2026-09-22 agent audit, closed while the Developer ID is reviewed.

- C·F8: when the knowledge book could not be fetched, the engineer ran without it in
  silence. A chat turn now says so once per conversation (a `chat_notice`, the app's words),
  through the real SDK against the local fake Messages API, and a mix run says so in its
  first note.
- B·F9: a message that was not a mix brief was answered in chat and reported $0, although
  reading the brief (the LLM brief compiler) and the reply are both billed calls. Both
  callers - the run (`Brain.run`) and the preview (`daw_bridge.advise`) - now report both.
"""
from __future__ import annotations

import anyio
import pytest

import orchestrator.chat as chatmod
from orchestrator import daw_bridge, daw_server, knowledge
from orchestrator.chat import CATALOG_LABEL, ChatSession, ChatTools
from orchestrator.targets import BriefNotUnderstood
from tests.fake_anthropic import serve

OPUS = "claude-opus-5-5"
TOOLS = [{"name": "daw_command", "description": "operate the DAW",
          "input_schema": {"type": "object", "properties": {
              "command": {"type": "string"}, "args": {"type": "object"}}}}]
CONTEXT = ("Workspace mode: agent (your moves apply immediately).\n\n"
           + CATALOG_LABEL + "): [{\"key\": \"track.set_gain\"}]")
TEXT_ROUND = {"blocks": [("thinking", ""), ("text", "done")]}


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(chatmod, "_DISABLED_BETAS", set())
    fake, shutdown = serve(monkeypatch)
    try:
        yield fake
    finally:
        shutdown()


class _Tools(ChatTools):
    def __init__(self):
        pass


def _turn(session: ChatSession, text: str = "how does the vocal sit?") -> list[dict]:
    events: list[dict] = []

    async def emit(event):
        events.append(event)
    anyio.run(lambda: session.turn(text, tools=_Tools(), context=CONTEXT, emit=emit))
    return events


def _without_the_book(monkeypatch) -> None:
    """The shipped brain with the control plane unreachable: no private module, no fetch."""
    monkeypatch.setattr(knowledge, "_cached", None)
    monkeypatch.setattr(knowledge, "_retry_after", 0.0)
    for source in ("_from_env_file", "_from_private_module", "_from_cloud"):
        monkeypatch.setattr(knowledge, source, lambda: None)


def _notices(events: list[dict]) -> list[str]:
    return [e["text"] for e in events if e["type"] == "chat_notice"]


# ---------------------------------------------------------------- C·F8

def test_a_turn_without_the_book_says_so_once_per_conversation(api, monkeypatch):
    _without_the_book(monkeypatch)
    session = ChatSession(model=OPUS, reasoning="medium", tool_defs_for=lambda during_run: TOOLS)
    api.script = [dict(TEXT_ROUND), dict(TEXT_ROUND)]
    first = _turn(session)
    second = _turn(session, "and the drums?")

    assert _notices(first) == [knowledge.BOOK_MISSING_NOTICE]
    assert _notices(second) == [], "said once a conversation, not at every turn"
    assert api.requests[0]["system"][0]["text"] == knowledge.fallback_block()

    # a new conversation is told again; and with the book there, nothing is said
    fresh = ChatSession(model=OPUS, reasoning="medium", tool_defs_for=lambda during_run: TOOLS)
    api.script = [dict(TEXT_ROUND)]
    assert _notices(_turn(fresh)) == [knowledge.BOOK_MISSING_NOTICE]


def test_a_turn_with_the_book_says_nothing_about_it(api, monkeypatch):
    monkeypatch.setattr(knowledge, "_cached", {"block": "# the book", "topics": {}})
    session = ChatSession(model=OPUS, reasoning="medium", tool_defs_for=lambda during_run: TOOLS)
    api.script = [dict(TEXT_ROUND)]
    assert _notices(_turn(session)) == []


def test_a_run_without_the_book_says_so_in_its_first_note(api, monkeypatch):
    from orchestrator.agents import ClaudeAgent
    _without_the_book(monkeypatch)
    assert ClaudeAgent(model=OPUS).book_notice == knowledge.BOOK_MISSING_NOTICE
    monkeypatch.setattr(knowledge, "_cached", {"block": "# the book", "topics": {}})
    assert ClaudeAgent(model=OPUS).book_notice == ""


# ---------------------------------------------------------------- B·F9

class _BilledAgent:
    """Stands in for ClaudeAgent: reading the brief cost 3 cents, the chat reply 2 more."""

    def __init__(self):
        self.usage = {"estimated_cost_usd": 0.0}

    def chat(self, message, context=None):
        self.usage["estimated_cost_usd"] += 0.02
        return "Hi - ask me for a change and I'll make it."


def _brief_not_understood_after_a_billed_read(agents: list):
    def run_session(project_dir, brief, *, agent, **kwargs):
        agent.usage["estimated_cost_usd"] += 0.03     # the LLM brief compiler, then it gave up
        raise BriefNotUnderstood("not a mixing brief")
    return run_session


def _stage(monkeypatch) -> list:
    agents: list = []

    def llm_agent(llm, what):
        agents.append(_BilledAgent())
        return agents[-1]
    monkeypatch.setattr(daw_bridge, "_llm_agent", llm_agent)
    import orchestrator.loop as loop
    monkeypatch.setattr(loop, "run_session", _brief_not_understood_after_a_billed_read(agents))
    monkeypatch.setattr(daw_bridge, "session_readout", lambda *a, **k: {})
    return agents


def test_a_message_that_is_not_a_brief_reports_what_it_cost_in_a_run(demo_project, monkeypatch):
    _stage(monkeypatch)
    out = daw_server.Brain(demo_project).run("hello there", use_llm=True)
    assert out["stop_reason"] == "chat" and out["chat"].startswith("Hi")
    assert out["cost_usd"] == pytest.approx(0.05), "the brief's reading and the reply, not $0"


def test_a_message_that_is_not_a_brief_reports_what_it_cost_in_a_preview(demo_project,
                                                                          monkeypatch):
    _stage(monkeypatch)
    out = daw_bridge.advise(demo_project, "hello there", use_llm=True)
    assert out["advice"].startswith("Hi") and out["count"] == 0
    assert out["cost_usd"] == pytest.approx(0.05)


def test_the_rule_agent_still_answers_for_nothing(demo_project):
    out = daw_server.Brain(demo_project).run("hello there", use_llm=False)
    assert out["stop_reason"] == "chat" and out["cost_usd"] == 0.0
