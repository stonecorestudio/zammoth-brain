"""What the production chat transport SENDS (2026-09-22 launch review, A1 + A2).

The shipped DAW runs the resident engineer on the `api` transport, on the USER's own
Anthropic key, so every request the loop makes is the user's money. Two defects made it send
more than it had to, and neither could be seen on the subscription dev backend (the SDK owns
that loop) or by a test that injects a stub client:

- A1: ChatSession kept one AsyncAnthropic for the conversation while the DAW runs each turn in
  a FRESH event loop (daw_bridge.chat_turn -> anyio.run). The kept connection belonged to the
  previous, closed loop: every later turn's first request was delivered in full, failed on the
  read, and the SDK retried it — sent, and billed, twice. Driven here with a REAL SDK client
  against a local HTTP server, one anyio.run per turn, exactly as production does.
- A2: the per-round live edit was a SYSTEM block after the last cache breakpoint, and it
  changes every round, so no part of the growing conversation could ever be read from cache.
  Driven here through a real multi-round turn, priced by a small model of the API's prefix
  cache (exact-match on breakpoint prefixes, cache_control markers excluded from the bytes).
- Cross-turn: the turn's context block (mode, master, chains, memory, digest) shared one
  system block with the ~60k-character command catalog. It changes every turn, so every
  turn's first round re-wrote the catalog AND the whole conversation after it. The catalog
  now has a system block of its own (1-hour TTL, like the persona) and the context block rides
  the last user message with the live edit, so the next turn reads this one from cache.
"""
from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import anyio
import pytest

import orchestrator.chat as chatmod
from orchestrator import llm_models
from orchestrator.chat import (CATALOG_LABEL, LIVE_NOTE, STATIC_CACHE, TURN_NOTE,
                               ChatSession, ChatTools, render_catalog, split_context)
from tests.fake_anthropic import _strip_cc

#: ONE layout since Phase 55: the turn's context block and the live edit are STORED in the
#: history, append-only (test_preserved_thinking.py), on every model. A second layout
#: re-attached them to the newest message on each request, for a model that does not bind its
#: thinking; no picker model was one, and the stored layout is valid everywhere. What a
#: conversation costs is still tested on a picker model and on an id off the table.
STORED, OFF_TABLE = llm_models.DEFAULT_MODEL, "claude-opus-5"
LAYOUTS = pytest.mark.parametrize("model", [STORED, OFF_TABLE], ids=["picker", "off_table"])


def test_one_layout_on_every_model():
    assert llm_models.spec(STORED).preserved_thinking
    assert llm_models.spec(OFF_TABLE) is None
    import inspect
    source = inspect.getsource(ChatSession._turn_api_rounds)
    assert "_request_messages(self.history)" in source and "_preserved" not in source


# ------------------------------------------------------------------ A1: one request per turn

def _sse(text: str) -> bytes:
    events = [
        {"type": "message_start", "message": {
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5",
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
         "usage": {"output_tokens": 2}},
        {"type": "message_stop"},
    ]
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


class _Api(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"      # keep-alive: the pooled connection is the whole bug
    hits: list[str] = []

    def log_message(self, *_a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length", 0)))
        type(self).hits.append(self.path)
        body = _sse("ok")
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _Tools(ChatTools):
    def __init__(self):
        pass


@pytest.fixture
def local_api(monkeypatch):
    _Api.hits = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Api)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01},
                     daemon=True).start()      # shutdown() waits out one poll
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{server.server_address[1]}")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-placeholder-not-a-key")   # only ever sent here
    monkeypatch.setenv("MIX_AGENT_LLM_BACKEND", "api")
    try:
        yield _Api
    finally:
        server.shutdown()
        server.server_close()


def test_every_turn_sends_its_request_once(local_api):
    session = ChatSession(tool_defs_for=lambda _d: [])

    async def emit(_e):
        pass

    per_turn = []
    for i in range(3):
        before = len(local_api.hits)
        # a NEW event loop per turn, exactly like daw_bridge.chat_turn's anyio.run
        anyio.run(lambda: session.turn(f"turn {i}", tools=_Tools(), context="ctx", emit=emit))
        per_turn.append(len(local_api.hits) - before)
    assert per_turn == [1, 1, 1], f"requests the API received per turn: {per_turn}"
    assert session._client is None, "the turn's own client must be closed with the turn"


def test_an_injected_client_is_never_replaced_or_closed():
    """Tests (and any host) that inject a client keep it across turns."""
    class Client:
        closed = False

        async def close(self):
            Client.closed = True
    client = Client()
    session = ChatSession(client=client, tool_defs_for=lambda _d: [])
    assert session._client is client
    anyio.run(session._close_owned, False)
    assert session._client is client and not Client.closed


# ------------------------------------------------------------------ A2: the growing history caches

class _Block:
    def __init__(self, type, **kw):
        self.type = type
        self.__dict__.update(kw)


class _Final:
    def __init__(self, content, stop_reason):
        self.content, self.stop_reason, self.usage = content, stop_reason, None


class _Stream:
    def __init__(self, final):
        self.final = final

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_a):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def get_final_message(self):
        return self.final


class _RecordingClient:
    def __init__(self, finals):
        self.finals, self.requests, self.api_key, self.messages = list(finals), [], None, self

    def stream(self, **kw):
        self.requests.append(json.loads(json.dumps(kw, default=lambda o: dict(vars(o)))))
        return _Stream(self.finals.pop(0))


def _flatten(request: dict) -> list[tuple[str, bool]]:
    """Every cacheable unit in render order (tools -> system -> messages' blocks), as
    (canonical bytes without cache_control, carries a breakpoint)."""
    units = [(json.dumps(t, sort_keys=True), False) for t in request.get("tools") or []]
    for block in request.get("system") or []:
        b = {k: v for k, v in block.items() if k != "cache_control"}
        units.append((json.dumps(b, sort_keys=True), "cache_control" in block))
    for m in request["messages"]:
        content = m["content"]
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
        for block in blocks:
            b = {k: v for k, v in block.items() if k != "cache_control"}
            units.append((m["role"] + json.dumps(b, sort_keys=True), "cache_control" in block))
    return units


class _PrefixCache:
    """The API's prompt cache, reduced to what matters here: a request reads the longest
    prefix ending at one of ITS breakpoints that an earlier request wrote at one of theirs,
    and writes the prefix at every one of its breakpoints. Sizes in characters."""

    def __init__(self):
        self.written: set[str] = set()

    def price(self, request: dict) -> tuple[int, int]:
        units = _flatten(request)
        total = sum(len(u) for u, _ in units)
        digest, size, read, marks = hashlib.sha256(), 0, 0, []
        for text, bp in units:
            digest.update(text.encode())
            size += len(text)
            if bp:
                key = digest.hexdigest()
                marks.append(key)
                if key in self.written:
                    read = size
        self.written.update(marks)
        return total, total - read


#: A catalog of the real one's order of size (the shipped one is ~60k characters).
_CATALOG = [{"key": f"fam{i // 12}.cmd{i}", "name": f"Command {i}", "category": "edit",
             "wants_args": True, "args": "{track, clip, at_s} " * 6} for i in range(200)]


def _context(turn: int) -> str:
    """What daw_bridge._chat_context_stable builds, in its order: the parts that change every
    turn, then the catalog (render_catalog) LAST."""
    return "\n\n".join([
        "Workspace mode: agent (your moves apply immediately).",
        f'Master analysis (measured): {{"lufs_i": {-14.0 - turn / 10:.1f}}}',
        (f"Where we left off: turn {turn - 1} brightened the vocal" if turn > 1
         else "Where we left off: nothing yet"),
        render_catalog(_CATALOG)])


def _stored(session) -> str:
    """The session's history as text, unescaped — json.dumps' default escapes the em dash and
    the newline in LIVE_NOTE / TURN_NOTE, and `NOTE not in dumps(...)` then passes whatever
    the history holds."""
    return json.dumps(session.history, default=str, ensure_ascii=False).replace("\\n", "\n")


def _build_turn(rounds: int, result_chars: int, model: str = STORED):
    finals = [_Final([_Block("tool_use", id=f"tu{i}", name="daw_command",
                             input={"command": "track.set_gain",
                                    "args": {"track": "vocals", "gain_db": -1}})], "tool_use")
              for i in range(rounds)]
    finals.append(_Final([_Block("text", text="done")], "end_turn"))
    client = _RecordingClient(finals)
    session = ChatSession(client=client, model=model, tool_defs_for=chatmod.daw_tool_defs)
    lives = iter(range(1000))

    async def daw_command(_command, _args):
        return {"ok": True, "detail": "x" * result_chars}   # a fat result: a snapshot, a read

    tools = ChatTools(engine_call=None, propose=None, start_run=None, stop_run=None,
                      auto_mix=None, file_instruction=None, save_note=None,
                      workspace_command=None, list_files=None, import_sample=None,
                      undo_proposal=None, render=None, daw_command=daw_command)

    async def emit(_e):
        pass
    return session, client, tools, emit, lives


@LAYOUTS
def test_the_growing_conversation_is_read_from_cache_every_round(monkeypatch, model):
    monkeypatch.setenv("MIX_AGENT_LLM_BACKEND", "api")
    rounds, result_chars = 8, 6000
    session, client, tools, emit, lives = _build_turn(rounds, result_chars, model)
    live_texts = []

    def live():
        text = f"Live DAW edit (daw_edit_state): {{\"round\": {next(lives)}}}"
        live_texts.append(text)
        return text

    anyio.run(lambda: session.turn("brighten the vocal, step by step", tools=tools,
                                   context=_context(1), emit=emit, live_context=live))
    reqs = client.requests
    assert len(reqs) == rounds + 1

    # The system prompt is FIXED within a turn — the live edit is not in it.
    assert all(r["system"] == reqs[0]["system"] for r in reqs)
    assert not any("daw_edit_state" in b["text"] for b in reqs[0]["system"])

    turn_part, _catalog = split_context(_context(1))
    for r in reqs:
        n_marks = json.dumps(r).count('"cache_control"')
        assert n_marks <= 4, f"the API allows 4 breakpoints, this request carries {n_marks}"
        assert json.dumps(r).count("Where we left off") == 1
    assert "cache_control" not in _stored(session)
    # Stored: the block and the first live edit are in the history, and every request is
    # the one before it with more on the end — nothing earlier is touched.
    stored = _stored(session)
    assert TURN_NOTE in stored and "Where we left off: nothing yet" in stored
    assert LIVE_NOTE in stored
    for a, b in zip(reqs, reqs[1:]):
        prev, cur = _strip_cc(a["messages"]), _strip_cc(b["messages"])
        assert cur[:len(prev)] == prev, "an earlier message was edited between requests"

    # Priced by the prefix-cache model: past round 1, each round pays fresh for what the
    # previous round added, not for the whole conversation again.
    cache = _PrefixCache()
    priced = [cache.price(r) for r in reqs]
    last_total, last_fresh = priced[-1]
    assert last_total > rounds * result_chars, "the fixture must build a large history"
    assert last_fresh < 0.2 * last_total, (
        f"the last round paid {last_fresh} of {last_total} chars uncached — the conversation "
        "is not being read from cache")
    fresh = [f for _t, f in priced[1:]]
    assert max(fresh) < 2 * min(fresh), f"uncached size grows with the history: {fresh}"


@LAYOUTS
def test_a_later_turn_caches_across_the_turn_boundary_too(monkeypatch, model):
    """The second turn's first round has TWO user messages to mark: the new text and the
    previous turn's last tool results. Its breakpoints must stay within the API's four."""
    monkeypatch.setenv("MIX_AGENT_LLM_BACKEND", "api")
    session, client, tools, emit, lives = _build_turn(2, 1000, model)
    session._client.finals.extend([_Final([_Block("text", text="and again")], "end_turn")])
    for n, text in enumerate(("first", "second"), start=1):
        anyio.run(lambda: session.turn(text, tools=tools, context=_context(n), emit=emit,
                                       live_context=lambda: "Live DAW edit (daw_edit_state): {}"))
    last = client.requests[-1]
    assert json.dumps(last).count('"cache_control"') == 4
    newest = last["messages"][-1]["content"]
    assert newest[0]["text"] == "second"
    # the notes are part of the message, so the mark sits after them
    assert "cache_control" in newest[-1] and "cache_control" not in newest[0]


# ------------------------------------------------------------------ cross-turn

def _run_turns(briefs: list[tuple[str, int]], result_chars: int = 3000, model: str = STORED):
    """Several turns on one session, each (text, tool rounds), each with ITS OWN context (the
    digest and the master move between turns, the catalog does not) — the way the DAW runs
    them: one anyio.run per turn. Returns the recorded requests, grouped by turn."""
    rounds_total = sum(r for _t, r in briefs)
    session, client, tools, emit, lives = _build_turn(0, result_chars, model)
    client.finals = []
    for _text, rounds in briefs:
        client.finals += [_Final([_Block("tool_use", id=f"tu{len(client.finals)}_{i}",
                                         name="daw_command",
                                         input={"command": "track.set_gain",
                                                "args": {"track": "vocals", "gain_db": -1}})],
                                 "tool_use") for i in range(rounds)]
        client.finals.append(_Final([_Block("text", text="done — the vocal sits forward now")],
                                    "end_turn"))
    by_turn = []
    for n, (text, _rounds) in enumerate(briefs, start=1):
        before = len(client.requests)
        anyio.run(lambda: session.turn(
            text, tools=tools, context=_context(n), emit=emit,
            live_context=lambda: f'Live DAW edit (daw_edit_state): {{"round": {next(lives)}}}'))
        by_turn.append(client.requests[before:])
    assert sum(len(t) for t in by_turn) == rounds_total + len(briefs)
    return session, by_turn


@LAYOUTS
def test_the_next_turn_reads_the_previous_turn_from_cache(monkeypatch, model):
    monkeypatch.setenv("MIX_AGENT_LLM_BACKEND", "api")
    session, turns = _run_turns([("brighten the vocal", 4), ("now the drums", 3),
                                 ("and the bass", 2)], result_chars=3000, model=model)
    reqs = [r for t in turns for r in t]

    # The system prompt is the SAME in every turn (the context block is not in it) ...
    assert all(r["system"] == reqs[0]["system"] for r in reqs)
    system_text = json.dumps(reqs[0]["system"])
    assert "Where we left off" not in system_text and "Master analysis" not in system_text
    # ... and it ends in the catalog, on a breakpoint of its own.
    assert reqs[0]["system"][-1]["text"].startswith(CATALOG_LABEL)

    # Each turn's first round: everything up to the previous turn's last tool results comes
    # from cache; what is paid fresh is only what that turn added after them (its closing
    # reply) plus this turn's own message, context block and live edit.
    cache = _PrefixCache()
    priced = [[cache.price(r) for r in t] for t in turns]
    for n in (1, 2):
        total, fresh = priced[n][0]
        new_tail = sum(len(json.dumps(b)) for b in turns[n][0]["messages"][-1]["content"])
        assert fresh < 3 * new_tail, (
            f"turn {n + 1}'s first round paid {fresh} of {total} chars uncached; the new tail is "
            f"{new_tail} — the previous turn was not read from cache")
        assert fresh < 0.1 * total

    # The turn's context block reached the model, current: this turn's digest, not the last.
    first_of_turn_3 = json.dumps(turns[2][0]["messages"][-1]["content"])
    assert "turn 2 brightened" in first_of_turn_3 and "nothing yet" not in first_of_turn_3
    assert TURN_NOTE in _stored(session)


@LAYOUTS
def test_the_static_prefix_and_the_conversation_both_live_an_hour(monkeypatch, model):
    """All four breakpoints are one lifetime since Phase 55 (so the longer-before-shorter rule
    holds by itself): the persona and the catalog, and the two moving breakpoints on the
    conversation - which were five minutes, so a pause of six rewrote the whole conversation
    (`chat.CONVERSATION_CACHE` has the arithmetic)."""
    monkeypatch.setenv("MIX_AGENT_LLM_BACKEND", "api")
    _session, turns = _run_turns([("first", 2), ("second", 1)], model=model)
    assert STATIC_CACHE == {"type": "ephemeral", "ttl": "1h"}
    for r in (x for t in turns for x in t):
        marks = [b.get("cache_control") for b in r["system"] if "cache_control" in b]
        assert marks == [STATIC_CACHE, STATIC_CACHE]
        for m in r["messages"]:
            content = m["content"] if isinstance(m["content"], list) else []
            for b in content:
                if isinstance(b, dict) and "cache_control" in b:
                    assert b["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_split_context_takes_the_catalog_from_its_label_on():
    full = render_catalog([{"key": "clip.split"}])
    assert full.startswith(CATALOG_LABEL)
    assert split_context("mode\n\nchains\n\n" + full) == ("mode\n\nchains", full)
    assert split_context(full) == ("", full)
    # no catalog (the offline gateway's context): it is all this turn's
    assert split_context('{"stems": []}') == ('{"stems": []}', "")
    # a part QUOTING the label (a digest of the user's words) does not move the cut
    quoted = "Where we left off: the user wrote\n\n" + CATALOG_LABEL + " — is that all?"
    assert split_context(quoted + "\n\n" + full) == (quoted, full)


def test_the_production_context_ends_in_the_catalog(tmp_path):
    """daw_bridge builds the context the split reads: the per-turn parts first, the catalog
    LAST — if a part were ever added after the catalog it would ride the static prefix and
    re-write it every turn, so this pins the order on the real builder."""
    from orchestrator import daw_bridge as B
    B.record_session_digest(tmp_path, session_id="", user="pull the vocal down", reply="Done.",
                            commands=["track.set_gain"])
    ctx = B._chat_context_stable(tmp_path, "agent", {}, {"commands": _CATALOG})
    turn_part, catalog = split_context(ctx)
    assert catalog == render_catalog(_CATALOG)
    assert turn_part.startswith("Workspace mode: agent")
    assert CATALOG_LABEL not in turn_part


def test_one_hour_cache_writes_are_counted_and_priced_at_twice_input():
    from orchestrator.agents import estimate_cost_usd

    session = ChatSession(client=object(), tool_defs_for=lambda _d: [])

    class _Creation:
        ephemeral_1h_input_tokens = 30_000
        ephemeral_5m_input_tokens = 10_000

    class _Usage:
        input_tokens, output_tokens = 1_000, 500
        cache_read_input_tokens, cache_creation_input_tokens = 50_000, 40_000
        cache_creation = _Creation()
    session._add_usage(_Usage())
    session._add_usage({"cache_creation_input_tokens": 2_000,
                        "cache_creation": {"ephemeral_1h_input_tokens": 2_000}})
    assert session.usage["cache_creation_input_tokens"] == 42_000
    assert session.usage["cache_creation_1h_input_tokens"] == 32_000
    # 10k at 1.25x + 30k at 2x on Opus 5's $5 input
    assert estimate_cost_usd("claude-opus-5", 0, 0, 0, 40_000, 30_000) == pytest.approx(
        (10_000 * 5 * 1.25 + 30_000 * 5 * 2.0) / 1e6)
    assert estimate_cost_usd("claude-opus-5", 0, 0, 0, 40_000) == pytest.approx(
        40_000 * 5 * 1.25 / 1e6)


def test_a_conversation_is_priced_one_request_at_a_time():
    """Haiku 5.5 costs five times as much for a prompt over 100,000 tokens, so a turn cannot
    be priced from its tokens added together: twelve 30k-token rounds are twelve short
    prompts. The session prices each request as it lands, and the turn's cost is what was
    added to that."""
    from orchestrator.daw_bridge import _turn_cost_usd, _usage_snapshot

    haiku = "claude-haiku-5-5"
    one = llm_models.cost_usd(haiku, 2_000, 400, cache_read_tokens=28_000)
    session = ChatSession(client=object(), model=haiku, tool_defs_for=lambda _d: [])
    session._add_usage({"input_tokens": 2_000, "output_tokens": 400,
                        "cache_read_input_tokens": 28_000})          # an earlier turn
    before = _usage_snapshot(session)
    for _ in range(12):
        session._add_usage({"input_tokens": 2_000, "output_tokens": 400,
                            "cache_read_input_tokens": 28_000})
    assert _turn_cost_usd(session, before) == pytest.approx(12 * one, abs=1e-4)
    # the tokens summed read as ONE 360k-token prompt: five times the truth
    summed = llm_models.cost_usd(haiku, 24_000, 4_800, cache_read_tokens=336_000)
    assert summed == pytest.approx(5 * 12 * one)
    # a request that IS long pays the long price, and only that request
    before = _usage_snapshot(session)
    session._add_usage({"input_tokens": 5_000, "output_tokens": 400,
                        "cache_read_input_tokens": 120_000})
    assert _turn_cost_usd(session, before) == pytest.approx(
        llm_models.cost_usd(haiku, 5_000, 400, cache_read_tokens=120_000), abs=1e-4)
    # the meter's token counts are still the plain sums, and hold no price
    assert session.usage["input_tokens"] == 13 * 2_000 + 5_000
    assert set(session.usage) == set(chatmod.USAGE_KEYS)
    # a session with no running price (a test double, the scripted chat) is priced from its
    # tokens, as before
    class _Plain:
        _model = "claude-opus-5-5"
        usage = {"input_tokens": 1_000, "output_tokens": 100}
    assert _turn_cost_usd(_Plain(), {}) == pytest.approx(
        llm_models.cost_usd("claude-opus-5-5", 1_000, 100), abs=1e-4)
