"""Every model on the picker, on the production chat transport (2026-09-22).

James, 2026-09-22: "add all models and effort levels associated with the anthropic api". Three
things decide whether a model the picker offers actually works, and each is driven here through
the REAL `ChatSession` and a REAL `anthropic.AsyncAnthropic` against a local fake of the
Messages API (no key, no network, nothing billed):

1. **Preserved thinking.** Opus 5.5 and Fable 5.1 bind every thinking block to the conversation
   that produced it — the system prompt, the tool set and every earlier message. For Anthropic
   accounts created on or after 2026-08-31 (every new user on their own key) a request that
   replays a block after any of those changed is a 400, and so is every later request of that
   conversation. The fake API below ENFORCES that check the way Anthropic documents it (strict:
   byte-equal after removing `cache_control`, which may move freely), so these tests fail on the
   pre-2026-09-22 loop — it re-attached the live edit and the turn's context block to the newest
   user message on every request and dropped them on the next. The negative control proves the
   simulator can see that.
2. **Per-model request shapes.** Haiku 4.5 takes neither adaptive thinking nor `effort`; the 4.6
   family has no `xhigh`; Opus 5.5 / Fable write their between-tool notes as thinking blocks.
3. **Refusals.** A `stop_reason: "refusal"` is an HTTP 200 whose content is empty or a partial the
   API says to discard: nothing in it may run, and the next turn must still work.
"""
from __future__ import annotations

import json

import anyio
import pytest

import orchestrator.chat as chatmod
from orchestrator import llm_models
from orchestrator.chat import CATALOG_LABEL, LIVE_NOTE, TURN_NOTE, ChatSession, ChatTools
from tests.fake_anthropic import PRESERVED, _canon, _strip_cc, serve


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(chatmod, "_DISABLED_BETAS", set())
    fake, shutdown = serve(monkeypatch)
    try:
        yield fake
    finally:
        shutdown()


# ---------------------------------------------------------------- a small DAW to drive

TOOLS_IDLE = [{"name": "daw_command", "description": "operate the DAW",
               "input_schema": {"type": "object", "properties": {
                   "command": {"type": "string"}, "args": {"type": "object"}}}},
              {"name": "propose", "description": "one guarded move",
               "input_schema": {"type": "object", "properties": {}}}]
TOOLS_RUN = [TOOLS_IDLE[0], {"name": "file_instruction", "description": "note for the run",
                             "input_schema": {"type": "object", "properties": {}}}]


class Daw:
    """A live edit the agent's own commands change, like the real one."""

    def __init__(self) -> None:
        self.state = {"tempo_bpm": 90, "tracks": [{"name": "Vox", "gain_db": 0.0}]}
        self.ran: list[str] = []

    def live(self) -> str:
        return "Live DAW edit (daw_edit_state): " + json.dumps(self.state)

    def tools(self) -> ChatTools:
        daw = self

        class T(ChatTools):
            def __init__(self):
                pass

            async def daw_command(self, command, args):
                daw.ran.append(command)
                daw.state["tracks"][0]["gain_db"] -= 1.0
                return {"ok": True}
        return T()


CONTEXT = ("Workspace mode: agent (your moves apply immediately).\n\n"
           "Master analysis (measured): {\"lufs_i\": -14.0}\n\n"
           "Applied chains: {\"Vox\": []}\n\n"
           + CATALOG_LABEL + "): [{\"key\": \"track.set_gain\"}]")


def _tool_round(n: int) -> dict:
    return {"blocks": [("thinking", ""), ("tool_use", f"toolu_{n}", "daw_command",
                                          {"command": "track.set_gain", "args": {}})]}


def _final(text: str = "done") -> dict:
    return {"blocks": [("thinking", ""), ("text", text)]}


def _run_turn(session: ChatSession, daw: Daw, text: str, *, context: str = CONTEXT,
              during_run: bool = False) -> list[dict]:
    events: list[dict] = []

    async def emit(e):
        events.append(e)
    # a NEW event loop per turn, like daw_bridge.chat_turn's anyio.run
    anyio.run(lambda: session.turn(text, tools=daw.tools(), context=context, emit=emit,
                                   during_run=during_run, live_context=daw.live))
    return events


def _session(model: str) -> ChatSession:
    return ChatSession(model=model, reasoning="high",
                       tool_defs_for=lambda during_run: TOOLS_RUN if during_run else TOOLS_IDLE)


def _append_only(requests: list[dict]) -> None:
    """Anthropic's three-step check, step 1: consecutive requests share system, tools and the
    whole previous message list byte for byte (cache_control aside)."""
    for a, b in zip(requests, requests[1:]):
        assert _strip_cc(a["system"]) == _strip_cc(b["system"]), "system changed"
        assert _canon(None, a["tools"], []) == _canon(None, b["tools"], []), "tools changed"
        prev, cur = _strip_cc(a["messages"]), _strip_cc(b["messages"])
        assert cur[:len(prev)] == prev, "an earlier message was edited between requests"


# ---------------------------------------------------------------- 1. preserved thinking

@pytest.mark.parametrize("model", sorted(PRESERVED))
def test_a_preserved_thinking_conversation_is_append_only(api, model):
    daw, session = Daw(), _session(model)
    api.script = [_tool_round(1), _tool_round(2), _final()]
    events = _run_turn(session, daw, "pull the vocal down, twice")
    api.script = [_tool_round(3), _final("done again")]
    events += _run_turn(session, daw, "and once more")

    assert daw.ran == ["track.set_gain"] * 3
    # never rejected — the strip-and-retry recovery would otherwise hide an edit
    assert api.rejected == []
    assert len(api.requests) == 5
    _append_only(api.requests)
    # the notes are IN the stored history: the turn's context + the live edit, then per
    # round only what moved (a delta), never a second full copy of an unchanged snapshot
    first_user = session.history[0]["content"]
    texts = [b["text"] for b in first_user if b.get("type") == "text"]
    assert any(t.startswith(TURN_NOTE) for t in texts)
    assert any(t.startswith(LIVE_NOTE + "Live DAW edit") for t in texts)
    refreshes = [b["text"] for m in session.history if m["role"] == "user"
                 and isinstance(m["content"], list) for b in m["content"]
                 if b.get("type") == "text" and b["text"].startswith(LIVE_NOTE)
                 and "CHANGED since the last snapshot" in b["text"]]
    assert len(refreshes) == 3, "each command's effect rides the next results message as a delta"
    # the second turn's context block names what it already holds instead of repeating it
    second_turn = next(m for m in session.history[1:] if chatmod._is_turn_start(m))
    ctx = next(b["text"] for b in second_turn["content"] if b["text"].startswith(TURN_NOTE))
    assert "unchanged since your previous context block" in ctx
    assert "Applied chains: {" not in ctx


def test_the_simulator_rejects_the_old_edit_pattern(api):
    """Negative control: the request pattern the loop had before 2026-09-22 - a note attached
    to the newest user message of one request and gone from the next - is refused by the
    enforcing fake, so the test above measures something. (Built by hand: the loop itself
    can no longer make it. Its second layout, which did, went in Phase 55.)"""
    model = "claude-opus-5-5"
    system = [{"type": "text", "text": "you mix"}]
    note = {"type": "text", "text": LIVE_NOTE + "Live DAW edit (daw_edit_state): {}"}
    first = {"model": model, "system": system, "tools": [], "messages": [
        {"role": "user", "content": [{"type": "text", "text": "pull the vocal down"}, note]}]}
    api.script = [_tool_round(1)]
    status, *_ = api.reply(first, {})
    assert status == 200
    signature = next(iter(api.bound))
    replay = [{"role": "assistant", "content": [
        {"type": "thinking", "thinking": "", "signature": signature},
        {"type": "tool_use", "id": "toolu_1", "name": "daw_command", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1",
                                      "content": "ok"}]}]
    kept = {**first, "messages": first["messages"] + replay}
    assert api.reply(kept, {})[0] == 200, "append-only: accepted"
    edited = {**first, "messages": [
        {"role": "user", "content": [{"type": "text", "text": "pull the vocal down"}]}] + replay}
    status, payload, *_ = api.reply(edited, {})
    assert status == 400 and b"bound to a different conversation" in payload


def test_an_id_off_the_table_stores_its_notes_too(api):
    """One layout (Phase 55). Opus 5 does not bind thinking to the conversation, and its notes
    used to ride each request unstored; the stored, append-only layout is valid everywhere."""
    daw, session = Daw(), _session("claude-opus-5")
    api.script = [_tool_round(1), _final()]
    _run_turn(session, daw, "pull the vocal down")
    opening = session.history[0]["content"]
    assert opening[0]["text"] == "pull the vocal down"
    assert any(b.get("text", "").startswith(LIVE_NOTE) for b in opening)
    first, second = (api.requests[k]["messages"] for k in (0, 1))
    assert [m["role"] for m in second[:len(first)]] == [m["role"] for m in first]


def test_a_new_tool_set_drops_the_bound_thinking_instead_of_failing(api):
    """A run starting between two turns swaps the tool set — the thinking bound to the old set
    is dropped (the API's documented recovery), never replayed into a 400."""
    daw, session = Daw(), _session("claude-opus-5-5")
    api.script = [_tool_round(1), _final()]
    _run_turn(session, daw, "pull the vocal down")
    assert any(chatmod._block_type(b) == "thinking" for m in session.history
               if m["role"] == "assistant" for b in m["content"])
    api.script = [_final("noted")]
    events = _run_turn(session, daw, "how is the run going?", during_run=True)
    assert api.rejected == [], "the swap was handled before the request, not by a 400"
    assert any(e.get("text") == "noted" for e in events)
    replayed = [b for m in api.requests[-1]["messages"] if m["role"] == "assistant"
                for b in m["content"] if b.get("type") == "thinking"]
    assert replayed == []


def test_a_trim_drops_thinking_and_restarts_the_notes_whole(api, monkeypatch):
    # by size since Phase 55 (it was a cap of 60 messages): any prompt is over this budget
    monkeypatch.setattr(llm_models, "history_budget", lambda model: (1, 0))
    daw, session = Daw(), _session("claude-sonnet-5-5")
    for i in range(4):
        api.script = [_tool_round(10 + i), _final(f"done {i}")]
        _run_turn(session, daw, f"move {i}")
    assert api.rejected == []
    first = session.history[0]
    assert chatmod._is_turn_start(first)
    # the turn opened at the last trim carries its notes WHOLE — whatever an earlier delta
    # was relative to may have been cut away with the old turns
    live = [b["text"] for b in first["content"] if b["text"].startswith(LIVE_NOTE)]
    assert live and live[0].startswith(LIVE_NOTE + "Live DAW edit (daw_edit_state):")
    ctx = [b["text"] for b in first["content"] if b["text"].startswith(TURN_NOTE)]
    assert ctx and "Applied chains: {" in ctx[0]


def test_a_signature_400_is_recovered_once(api):
    """Something outside this loop edited the conversation (a host appending to history, an
    upgrade mid-session): the API's recovery — drop the thinking, send once more — keeps the
    conversation alive instead of failing every request from then on."""
    daw, session = Daw(), _session("claude-opus-5-5")
    api.script = [_tool_round(1), _final()]
    _run_turn(session, daw, "pull the vocal down")
    session.history[0]["content"][0]["text"] += " (edited)"   # an out-of-band edit
    api.script = [_final("still here")]
    events = _run_turn(session, daw, "again")
    assert any(e.get("text") == "still here" for e in events)
    assert len(api.requests) == 4, "one rejected request, one re-send"


# ---------------------------------------------------------------- 2. per-model request shapes

@pytest.mark.parametrize("model", [m.id for m in llm_models.MODELS])
@pytest.mark.parametrize("effort", list(llm_models.EFFORT_LEVELS))
def test_every_model_and_effort_sends_a_request_that_model_accepts(api, model, effort):
    session = ChatSession(model=model, reasoning=effort, tool_defs_for=lambda _d: TOOLS_IDLE)
    _run_turn(session, Daw(), "hello")
    body, headers = api.requests[-1], api.headers[-1]
    spec = llm_models.spec(model)
    if not spec.efforts:   # Haiku 4.5: adaptive thinking and effort both 400
        assert "thinking" not in body and "output_config" not in body
    else:
        assert body["thinking"]["type"] == "adaptive"
        assert body["output_config"]["effort"] in spec.efforts
        if effort in spec.efforts:
            assert body["output_config"]["effort"] == effort
    assert body["max_tokens"] <= spec.max_output_tokens
    for forbidden in ("temperature", "top_p", "top_k", "tool_choice"):
        assert forbidden not in body
    assert body["messages"][-1]["role"] == "user", "no assistant prefill"
    if spec.progress_thinking:
        assert body["thinking"].get("display") == "updates"
        assert chatmod.BETA_DISPLAY_UPDATES in headers.get("anthropic-beta", "")
    else:
        assert "display" not in (body.get("thinking") or {})


def test_a_level_a_model_lacks_is_clamped_to_the_one_below(odd_ladders):
    no_xhigh, no_effort = odd_ladders
    assert llm_models.effective_effort(no_xhigh, "xhigh") == "high"
    assert llm_models.effective_effort(no_xhigh, "max") == "max"
    assert llm_models.effective_effort(no_effort, "high") is None
    assert llm_models.thinking_kwargs(no_effort, "high") == {}
    assert chatmod.chat_max_tokens(no_effort, "max") == 16_000   # no effort there
    # the three on the picker take every level as asked
    for m in ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"):
        assert [llm_models.effective_effort(m, e) for e in llm_models.EFFORT_LEVELS] \
            == list(llm_models.EFFORT_LEVELS)
        # 64,000 at every level since Phase 55 (it was 16k up to high, 32k at xhigh)
        assert [chatmod.chat_max_tokens(m, e) for e in llm_models.EFFORT_LEVELS] == [64_000] * 5
    # an id off the picker is sent no thinking config at all: a plain request is valid anywhere
    assert llm_models.thinking_kwargs("claude-opus-5", "high") == {}


def test_progress_notes_reach_the_feed(api):
    """Opus 5.5 writes its between-tool notes as thinking blocks — with display "updates"
    they stream into the feed the way Opus 5's text notes always did."""
    daw, session = Daw(), _session("claude-opus-5-5")
    api.script = [{"blocks": [("thinking", "Found the vocal; pulling it down."),
                              ("tool_use", "toolu_1", "daw_command",
                               {"command": "track.set_gain", "args": {}})]}, _final()]
    events = _run_turn(session, daw, "pull the vocal down")
    streamed = "".join(e["text"] for e in events if e.get("type") == "chat_delta")
    assert "Found the vocal; pulling it down." in streamed


def test_a_key_without_the_progress_beta_keeps_working(api):
    daw, session = Daw(), _session("claude-opus-5-5")
    api.script = [{"error": (400, "thinking.display: 'updates' requires the "
                                  "thinking-display-updates-2026-08-18 beta")}, _final("fine")]
    events = _run_turn(session, daw, "hello")
    assert any(e.get("text") == "fine" for e in events)
    assert chatmod.BETA_DISPLAY_UPDATES in chatmod._DISABLED_BETAS
    assert "display" not in api.requests[-1]["thinking"]


# ---------------------------------------------------------------- 3. refusals

def test_a_refusal_runs_nothing_and_the_next_turn_still_works(api):
    daw, session = Daw(), _session("claude-opus-5-5")
    api.script = [{"blocks": [("text", "Sure, I'll"), ("tool_use", "toolu_9", "daw_command",
                                                       {"command": "track.set_gain"})],
                   "stop": "refusal", "stop_details": {"type": "refusal", "category": "cyber"}}]
    events = _run_turn(session, daw, "do the thing")
    assert daw.ran == [], "a declined reply's tool call must never run"
    said = " ".join(e.get("text", "") for e in events if e.get("type") == "chat_message")
    assert "declined" in said and "[cyber]" in said
    assert session.history[-1] == {"role": "assistant", "content": [
        {"type": "text", "text": chatmod._DECLINED_HISTORY}]}
    api.script = [_final("ok")]
    events = _run_turn(session, daw, "fine, just tell me the tempo")
    assert any(e.get("text") == "ok" for e in events)


def test_a_cut_off_reply_says_so(api):
    daw, session = Daw(), _session("claude-opus-5-5")
    api.script = [{"blocks": [("text", "The vocal sits at")], "stop": "max_tokens"}]
    events = _run_turn(session, daw, "where does the vocal sit?")
    said = " ".join(e.get("text", "") for e in events if e.get("type") == "chat_message")
    assert "cut off" in said


def test_the_brain_log_names_each_turns_model_and_the_effort_it_gets(api, capsys, odd_ladders):
    """BETA_TEST_CHECKLIST 10.9 asks a tester to confirm the pick reached the API: after the
    status bar's "from the next message" nothing on screen names the model, so the brain log
    does — per turn, with the effort CLAMPED to what the model is actually sent."""
    daw = Daw()
    no_xhigh, _no_effort = odd_ladders
    session = ChatSession(model=no_xhigh, reasoning="xhigh",
                          tool_defs_for=lambda during_run: TOOLS_IDLE)
    api.script = [_final()]
    _run_turn(session, daw, "how loud is the master?")
    assert f"[chat] t0001 on {no_xhigh} · effort high (api)" in capsys.readouterr().err
    assert api.requests[-1]["output_config"]["effort"] == "high"   # the log tells the truth
