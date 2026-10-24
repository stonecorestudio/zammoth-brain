"""Phase 54, step 2: the engineer writes short replies, and says what is true now.

Measured 2026-10-12 on the two showcase rehearsals: all 16 final replies were 110-606 words
(median about 360) - 111 words for "Split both songs into stems". The prompt's own rule ("Two
or three sentences is the target") was 794 of its 64,200 characters, mid-prompt, about single
moves only, among about 45 lines telling the model to say, report, name or tell.

James chose the shape (an outcome line plus at most three bullets). It is the LAST section of
the DAW prompt, it gives its reason (the feed already shows the work), and its one line rides
the end of every turn's request - where Anthropic's guidance says a style rule holds best.
What a model then writes can only be measured with a model: `python -m
orchestrator.turn_timing <project>` prints each turn's reply length.
"""
from __future__ import annotations

import json

import anyio
import pytest

import orchestrator.chat as chatmod
from orchestrator import llm_models
from orchestrator.chat import (DAW_SYSTEM_PROMPT, LIVE_NOTE, REPLY_NOTE, SYSTEM_PROMPT,
                               ChatSession, ChatTools, daw_tool_defs)
from tests.fake_anthropic import _strip_cc, serve


# ---------------------------------------------------------------- the contract's text

def test_the_reply_contract_is_the_last_section_of_the_daw_prompt():
    at = DAW_SYSTEM_PROMPT.rindex("How you write to the user:")
    tail = DAW_SYSTEM_PROMPT[at:]
    assert at > len(DAW_SYSTEM_PROMPT) * 0.9 and "Asking and planning:" not in tail
    flat = " ".join(tail.split())
    for piece in ("one sentence", "at most three short bullets", "Then stop.",
                  "already in the feed", "Asked for the detail", "<examples>"):
        assert piece in flat, piece
    assert tail.count("\nAsked ") == 4, "four varied examples, labelled as shape"
    assert "These show the shape, not the words." in tail
    assert "How you write to the user" not in SYSTEM_PROMPT, "the offline shell keeps its own"


def test_the_rules_that_asked_for_more_words_were_reworded_not_left_to_compete():
    for gone in ("report what you did and how the", "say so beside the number",
                 "tell the user what you changed", "Two or three sentences is the target",
                 "Write for a narrow panel"):
        assert gone not in DAW_SYSTEM_PROMPT.replace("\n", " "), gone
        assert all(gone not in json.dumps(d).replace("\\n", " ") for d in daw_tool_defs(False)), gone
    # ...and what they were FOR is still said, once and briefly
    flat = " ".join(DAW_SYSTEM_PROMPT.split())
    assert "names what it left out in a clause" in flat          # not_heard
    assert "say which in a clause" in flat                       # what could not be verified
    assert "`not_heard`" in DAW_SYSTEM_PROMPT


def test_the_daw_prompt_does_not_promise_approve_and_reject_cards():
    assert "approve/reject cards" in SYSTEM_PROMPT               # the offline workspace has them
    assert "approve/reject cards" not in DAW_SYSTEM_PROMPT
    assert "a row for every command you ran" in " ".join(DAW_SYSTEM_PROMPT.split())


def test_the_prompt_asks_for_independent_calls_in_one_response():
    flat = " ".join(DAW_SYSTEM_PROMPT.split())
    assert "Send together what does not depend on each other." in flat
    assert "wait for a reply only when the next call needs something that reply says" in flat
    command = next(d for d in daw_tool_defs(False) if d["name"] == "daw_command")
    assert "go out together, in one response" in command["description"]
    assert "address your next call by it" not in command["description"]


def test_the_default_effort_is_medium():
    assert llm_models.DEFAULT_EFFORT == "medium" == chatmod.DEFAULT_REASONING
    assert llm_models.thinking_kwargs(llm_models.DEFAULT_MODEL, llm_models.DEFAULT_EFFORT) == {
        "thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}}


# ---------------------------------------------------------------- the note on every request

@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(chatmod, "_DISABLED_BETAS", set())
    fake, shutdown = serve(monkeypatch)
    try:
        yield fake
    finally:
        shutdown()


TOOLS = [{"name": "daw_command", "description": "operate the DAW",
          "input_schema": {"type": "object", "properties": {
              "command": {"type": "string"}, "args": {"type": "object"}}}}]
CONTEXT = ("Workspace mode: agent (your moves apply immediately).\n\n"
           + chatmod.CATALOG_LABEL + "): [{\"key\": \"track.set_gain\"}]")
LIVE = "Live DAW edit (daw_edit_state): {\"tempo_bpm\": 90}"


class _Tools(ChatTools):
    def __init__(self):
        pass

    async def daw_command(self, command, args):
        return {"ok": True}


def _turn(session: ChatSession, text: str) -> None:
    async def emit(_event):
        pass
    anyio.run(lambda: session.turn(text, tools=_Tools(), context=CONTEXT, emit=emit,
                                   live_context=lambda: LIVE))


def _tool_round(n: int) -> dict:
    return {"blocks": [("thinking", ""), ("tool_use", f"toolu_{n}", "daw_command",
                                          {"command": "track.set_gain", "args": {}})]}


FINAL = {"blocks": [("thinking", ""), ("text", "done")]}


def _last_text(message: dict) -> str:
    return message["content"][-1]["text"]


def test_the_daw_session_carries_the_note_and_the_offline_one_does_not():
    assert ChatSession(system=DAW_SYSTEM_PROMPT)._reply_note == REPLY_NOTE
    assert ChatSession()._reply_note == ""
    assert ChatSession(system=DAW_SYSTEM_PROMPT, reply_note="")._reply_note == ""
    assert REPLY_NOTE.startswith("[How to reply") and "not the user's words" in REPLY_NOTE


def test_the_note_is_the_last_thing_in_a_turns_message_and_the_history_stays_append_only(api):
    session = ChatSession(system=DAW_SYSTEM_PROMPT, tool_defs_for=lambda during_run: TOOLS)
    api.script = [_tool_round(1), FINAL]
    _turn(session, "pull the vocal down")
    api.script = [FINAL]
    _turn(session, "and again")
    bodies = api.requests
    assert len(bodies) == 3 and api.rejected == []
    first = bodies[0]["messages"][-1]
    assert first["role"] == "user" and _last_text(first) == REPLY_NOTE
    assert first["content"][-2]["text"].startswith(LIVE_NOTE), "after the live edit"
    assert first["content"][0]["text"] == "pull the vocal down", "the user's words stay first"
    assert _last_text(bodies[2]["messages"][-1]) == REPLY_NOTE, "every turn carries its own"
    for earlier, later in zip(bodies, bodies[1:]):
        before, after = _strip_cc(earlier["messages"]), _strip_cc(later["messages"])
        assert after[:len(before)] == before, "an earlier message was edited between requests"


def test_an_id_off_the_table_gets_the_note_the_same_way(api):
    """One layout since Phase 55: a model that does not bind its thinking had the note
    re-attached per request and never stored. The stored layout is valid on every model."""
    session = ChatSession(model="claude-opus-5", system=DAW_SYSTEM_PROMPT,
                          tool_defs_for=lambda during_run: TOOLS)
    api.script = [_tool_round(1), FINAL]
    _turn(session, "pull the vocal down")
    first = api.requests[0]["messages"][-1]
    assert _last_text(first) == REPLY_NOTE
    assert session.history[0]["content"][-1]["text"] == REPLY_NOTE, "stored with the turn"
    before, after = (_strip_cc(b["messages"]) for b in api.requests)
    assert after[:len(before)] == before


def test_the_subscription_prompt_ends_with_the_note(monkeypatch):
    monkeypatch.setenv("MIX_AGENT_LLM_BACKEND", "subscription")
    seen: dict = {}

    class _Sub:
        async def chat_turn(self, *, system, prompt, tool_defs, execute, on_delta, on_tool,
                            max_turns, effort=None, should_stop=None, max_tokens=None, timer=None):
            seen["prompt"], seen["system"], seen["effort"] = prompt, system, effort
            return "done", {}
    session = ChatSession(system=DAW_SYSTEM_PROMPT, tool_defs_for=lambda during_run: TOOLS)
    monkeypatch.setattr(session, "_subscription", lambda: _Sub())
    _turn(session, "pull the vocal down")
    assert seen["prompt"].endswith(REPLY_NOTE)
    assert seen["prompt"].index(LIVE_NOTE) < seen["prompt"].index(REPLY_NOTE)
    assert REPLY_NOTE not in seen["system"]
    assert seen["effort"] == "medium", "the default effort reaches the CLI too"


def test_a_bullet_is_one_line():
    """Phase 55: the run-through's replies had the shape - an outcome line and three bullets -
    and the bullets ran two or three sentences each (median 166 words against a target of
    about 80). The contract and its per-turn line both say a bullet is one line."""
    contract = DAW_SYSTEM_PROMPT.split("How you write to the user:")[1]
    assert "one line each" in contract and "A bullet is one fact or one choice" in contract
    assert "one line each" in REPLY_NOTE
    examples = contract.split("<examples>")[1]
    bullets = [line for line in examples.splitlines() if line.startswith("- ")]
    assert bullets and all(len(line.split()) <= 22 for line in bullets), "the examples show it"

