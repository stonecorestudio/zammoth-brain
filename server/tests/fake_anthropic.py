"""A local fake of Anthropic's Messages API, for driving the REAL SDK clients (2026-09-22).

No key, no network, nothing billed: a ThreadingHTTPServer on 127.0.0.1 that speaks the SSE the
SDK streams (`messages.stream`) and records every request body + headers. Replies are scripted
per request (`script`: blocks, stop reason, stop details, usage, an error - an HTTP status or
one inside the stream - or a delay). It also
ENFORCES Anthropic's preserved-thinking check for the models that bind thinking to the
conversation (Opus 5.5, Fable 5.1): a replayed thinking block whose system / tools / earlier
messages changed since it was produced is a 400, strictly (byte-equal after `cache_control`,
which may move freely) — see test_preserved_thinking.py for why.

Used by test_preserved_thinking.py (the chat) and test_run_agent_transport.py (mix runs).

Since Phase 56 (2026-10-12) it also runs OUTSIDE pytest (`start`, `env`), for the chat rig
(`orchestrator/chat_rig.py`: the real app and the real brain against this fake), and a step
can do what only a real network does:

- `pace_s`: the response is written one SSE event at a time, that many seconds apart (a step
  with none of the keys below is still one write, as it always was);
- `cut_after: N`: the connection is dropped after N events - a socket that dies, not an SSE
  `error` event;
- `stall_after: N` + `stall_s`: N events, then nothing at all for that long;
- `hold` + `hold_after: N`: N events, then keep-alive pings until `release(name)`, the
  client going away (a Stop) or `hold_max_s` - a moment a test can meet without a clock;
- `headers`: response headers for that step (`retry-after`); `details`: an error's
  `error.details` (the spend cap's `error_code`);
- a block `("tool_use_raw", id, name, "<text>")` streams that text as the call's input,
  valid JSON or not.

`require_script = True` makes an unscripted request an error (the default reply, "ok", hides
a turn that asked more often than its script expected).

`strict = True` refuses what the real API refuses, as far as this file knows: `violation`
lists the rules. THEY ARE A TRANSCRIPTION of Anthropic's documents, not the API - each is a
rule the first session on a real key did not contradict, and a 400 the real API gives that
this file did not becomes a rule here. A request `strict` accepts is not thereby valid.
"""
from __future__ import annotations

import base64
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

from orchestrator import llm_models

PRESERVED = {m.id for m in llm_models.MODELS if m.preserved_thinking}


def _strip_cc(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _strip_cc(v) for k, v in x.items() if k != "cache_control"}
    if isinstance(x, list):
        return [_strip_cc(v) for v in x]
    return x


def _canon(system: Any, tools: Any, messages: list) -> str:
    tools = sorted((_strip_cc(t) for t in (tools or [])), key=lambda t: t.get("name", ""))
    return json.dumps([_strip_cc(system), tools, _strip_cc(messages)], sort_keys=True)


def _event_bytes(e: dict) -> bytes:
    return f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode()


def _sse(blocks: list[tuple], stop: str, model: str,
         stop_details: Optional[dict] = None, speed: Optional[str] = None,
         usage_in: Optional[dict] = None, out_tokens: int = 5,
         stream_error: Optional[tuple] = None) -> list[bytes]:
    """`usage_in` replaces the input side a response reports at its start (a step's `usage`),
    `out_tokens` is the final count its `message_delta` carries (a step's `out`), and
    `stream_error` = (type, message) ends the stream with an SSE `error` event after
    `message_start` - a failure INSIDE an HTTP 200, which the SDK does not retry."""
    usage: dict = {"input_tokens": 10, "output_tokens": 1, **(usage_in or {})}
    if speed:
        usage["speed"] = speed   # what the response says it ran at (fast mode)
    events: list[dict] = [{"type": "message_start", "message": {
        "id": "msg_x", "type": "message", "role": "assistant", "model": model, "content": [],
        "stop_reason": None, "stop_sequence": None,
        "usage": usage}}]
    for i, b in enumerate(blocks):
        def start(block):
            return {"type": "content_block_start", "index": i, "content_block": block}

        def delta(d):
            return {"type": "content_block_delta", "index": i, "delta": d}
        if b[0] == "thinking":
            events.append(start({"type": "thinking", "thinking": "", "signature": ""}))
            if b[1]:
                events.append(delta({"type": "thinking_delta", "thinking": b[1]}))
            events.append(delta({"type": "signature_delta", "signature": b[2]}))
        elif b[0] == "text":
            events.append(start({"type": "text", "text": ""}))
            events.append(delta({"type": "text_delta", "text": b[1]}))
        elif b[0] == "tool_use":
            events.append(start({"type": "tool_use", "id": b[1], "name": b[2], "input": {}}))
            events.append(delta({"type": "input_json_delta", "partial_json": json.dumps(b[3])}))
        elif b[0] == "tool_use_raw":      # the input exactly as given: cut short, or not JSON
            events.append(start({"type": "tool_use", "id": b[1], "name": b[2], "input": {}}))
            events.append(delta({"type": "input_json_delta", "partial_json": str(b[3])}))
        events.append({"type": "content_block_stop", "index": i})
    if stream_error is not None:      # after whatever blocks the step streamed first
        events.append({"type": "error", "error": {"type": stream_error[0],
                                                  "message": stream_error[1]}})
        return [_event_bytes(e) for e in events]
    md: dict = {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                "usage": {"output_tokens": int(out_tokens)}}
    if stop_details:
        md["delta"]["stop_details"] = stop_details
    events += [md, {"type": "message_stop"}]
    return [_event_bytes(e) for e in events]


#: The step keys that make a response arrive event by event (see the header).
_NETWORK_KEYS = ("pace_s", "cut_after", "stall_after", "hold")
_EFFORTS = ("low", "medium", "high", "xhigh", "max")
_BETA_FAST = "fast-mode-2026-02-01"
_BETA_UPDATES = "thinking-display-updates-2026-08-18"
MAX_OUTPUT_TOKENS = 128_000
MAX_CACHE_BREAKPOINTS = 4


def _breakpoints(body: dict) -> list[str]:
    """Every `cache_control` of a request in the order the API renders it (tools, system,
    messages), as its TTL ("5m" when none is named)."""
    out: list[str] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            cc = x.get("cache_control")
            if isinstance(cc, dict):
                out.append(str(cc.get("ttl") or "5m"))
            for k, v in x.items():
                if k != "cache_control":
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    for part in ("tools", "system", "messages"):
        walk(body.get(part))
    return out


_IMAGE_TYPES = ("image/jpeg", "image/png", "image/gif", "image/webp")
_IMAGE_B64_MAX = 5 * 1024 * 1024


def _image_problem(block: dict) -> Optional[str]:
    """Why the API would refuse this image block's source, or None (rule 13)."""
    source = block.get("source")
    if not isinstance(source, dict) or source.get("type") != "base64":
        return "only a base64 source is used here"
    if source.get("media_type") not in _IMAGE_TYPES:
        return f"media_type: Input should be one of {', '.join(_IMAGE_TYPES)}"
    data = source.get("data")
    if not isinstance(data, str) or not data:
        return "data: Field required"
    if len(data) > _IMAGE_B64_MAX:
        return f"image exceeds 5 MB maximum: {len(data)} bytes > {_IMAGE_B64_MAX} bytes"
    try:
        base64.b64decode(data, validate=True)
    except (ValueError, TypeError):
        return "data: the image data is not valid base64"
    return None


def violation(body: dict, headers: dict, path: str = "") -> Optional[str]:
    """Why the real API would refuse this request, or None. A transcription (the header says
    of what); each rule is one sentence of Anthropic's documents:

    1 `max_tokens` is required and at most the model's output limit; 2 the model is one the
    picker holds; 3 the conversation starts with a user message and never ends on an assistant
    one (no prefill on these models); 4 every `tool_use` is answered by a `tool_result` in the
    NEXT message, a `tool_result` answers a call of the message before it, and results come
    first in their message; 5 forced `tool_choice` (`any` / `tool`) is refused on the 5.5
    family; 6 `budget_tokens` / `type: enabled` thinking is refused, and `disabled` on Opus
    5.5 and Sonnet 5.5; 7 sampling parameters are refused; 8 at most four cache breakpoints,
    and never an hour-long one after a five-minute one; 9 a beta field needs its beta header
    (`speed`, `thinking.display: updates`); 10 `output_config.effort` is one of five words;
    11 a text block is not empty; 12 tool names are unique; 13 an image block (a user
    message's, or inside a tool result) is base64 of one of four media types, at most 5 MB of
    it, and no assistant message holds one (Phase 64)."""
    model = body.get("model")
    known = {m.id for m in llm_models.MODELS}
    if model not in known:
        return f"model: {model}"
    mt = body.get("max_tokens")
    if not isinstance(mt, int) or isinstance(mt, bool) or mt < 1:
        return "max_tokens: Field required"
    if mt > MAX_OUTPUT_TOKENS:
        return (f"max_tokens: {mt} > {MAX_OUTPUT_TOKENS}, which is the maximum allowed number "
                f"of output tokens for {model}")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return "messages: at least one message is required"
    if messages[0].get("role") != "user":
        return 'messages: first message must use the "user" role'
    if messages[-1].get("role") == "assistant":
        return ("This model does not support assistant message prefill. The conversation "
                "must end with a user message.")

    def blocks(msg: dict) -> list[dict]:
        c = msg.get("content")
        return [b for b in c if isinstance(b, dict)] if isinstance(c, list) else []
    for k, msg in enumerate(messages):
        content = msg.get("content")
        if content == "" or content == []:
            return f"messages.{k}: all messages must have non-empty content"
        for j, b in enumerate(blocks(msg)):
            if b.get("type") == "text" and not b.get("text"):
                return f"messages.{k}.content.{j}.text: text content blocks must be non-empty"
        if msg.get("role") == "assistant":
            asked = [b.get("id") for b in blocks(msg) if b.get("type") == "tool_use"]
            nxt = messages[k + 1] if k + 1 < len(messages) else {}
            answered = [b.get("tool_use_id") for b in blocks(nxt)
                        if nxt.get("role") == "user" and b.get("type") == "tool_result"]
            missing = [i for i in asked if i not in answered]
            if missing:
                return (f"messages.{k}: `tool_use` ids were found without `tool_result` blocks "
                        f"immediately after: {', '.join(map(str, missing))}. Each `tool_use` "
                        "block must have a corresponding `tool_result` block in the next message.")
        if msg.get("role") == "user":
            prev = messages[k - 1] if k else {}
            asked = [b.get("id") for b in blocks(prev)
                     if prev.get("role") == "assistant" and b.get("type") == "tool_use"]
            seen_other = False
            for j, b in enumerate(blocks(msg)):
                if b.get("type") != "tool_result":
                    seen_other = True
                    continue
                if b.get("tool_use_id") not in asked:
                    return (f"messages.{k}.content.{j}: unexpected `tool_use_id` found in "
                            f"`tool_result` blocks: {b.get('tool_use_id')}. Each `tool_result` "
                            "block must have a corresponding `tool_use` block in the previous "
                            "message.")
                if seen_other:
                    return (f"messages.{k}: `tool_result` blocks must come first in the "
                            "content of the message that answers a tool call")
    for k, msg in enumerate(messages):
        for j, b in enumerate(blocks(msg)):
            inner = b.get("content") if b.get("type") == "tool_result" else None
            for where, img in ([(f"messages.{k}.content.{j}", b)]
                               + [(f"messages.{k}.content.{j}.content.{n}", x)
                                  for n, x in enumerate(inner if isinstance(inner, list) else [])
                                  if isinstance(x, dict)]):
                if img.get("type") != "image":
                    continue
                if msg.get("role") != "user":
                    return f"{where}: image content is only allowed in user messages"
                if (why := _image_problem(img)) is not None:
                    return f"{where}.source: {why}"
    choice = body.get("tool_choice")
    if isinstance(choice, dict) and choice.get("type") in ("any", "tool"):
        return 'tool_choice: type "tool" and "any" are not supported for this model.'
    thinking = body.get("thinking")
    if isinstance(thinking, dict):
        if "budget_tokens" in thinking or thinking.get("type") == "enabled":
            return "thinking.budget_tokens: Extra inputs are not permitted"
        if thinking.get("type") == "disabled" and model in ("claude-opus-5-5", "claude-sonnet-5-5"):
            return "thinking.type: `disabled` is not supported for this model"
        if thinking.get("display") == "updates" and _BETA_UPDATES not in headers.get("anthropic-beta", ""):
            return "thinking.display: Extra inputs are not permitted"
    for knob in ("temperature", "top_p", "top_k"):
        if knob in body:
            return f"{knob}: this parameter is not supported for this model"
    marks = _breakpoints(body)
    if len(marks) > MAX_CACHE_BREAKPOINTS:
        return (f"A maximum of {MAX_CACHE_BREAKPOINTS} blocks with cache_control may be "
                f"provided. Found {len(marks)}.")
    if "1h" in marks and "5m" in marks[:len(marks) - 1 - marks[::-1].index("1h")]:
        return ("cache_control: a `ttl` of 1h must not come after a block with a shorter "
                "`ttl` (tools, then system, then messages)")
    if "speed" in body and (_BETA_FAST not in headers.get("anthropic-beta", "")
                            or "beta=true" not in path):
        return "speed: Extra inputs are not permitted"
    effort = (body.get("output_config") or {}).get("effort")
    if effort is not None and effort not in _EFFORTS:
        return f"output_config.effort: Input should be one of {', '.join(_EFFORTS)}"
    names = [t.get("name") for t in body.get("tools") or [] if isinstance(t, dict)]
    if len(names) != len(set(names)):
        return "tools: Tool names must be unique."
    return None


class FakeApi:
    """Scripted replies + Anthropic's preserved-thinking check, enforced."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.headers: list[dict] = []
        self.paths: list[str] = []      # each request's path + query (the beta door adds one)
        self.script: list[dict] = []
        self.bound: dict[str, tuple[str, str]] = {}   # signature -> (model, canonical prefix)
        self.enforce = True
        self.rejected: list[str] = []   # every binding 400 this fake returned
        self._sig = 0
        # Phase 56 (the header says what each is for)
        self.strict = False
        self.violations: list[str] = []      # every 400 `strict` returned
        self.require_script = False
        self.unscripted: list[dict] = []     # requests no step was left for
        self.limits: dict[str, str] = {}     # headers every reply carries (the account's limits)
        self.dropped = 0                     # replies the client left before they ended
        self.waiting: list[str] = []         # holds a response has reached, in order
        self.spent_s: list[float] = []       # how long THIS FILE took to decide each reply
        self._holds: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def hold_event(self, name: str) -> threading.Event:
        with self._lock:
            return self._holds.setdefault(name, threading.Event())

    def release(self, name: str) -> None:
        """Let the response held at `name` go on (now, or when it gets there)."""
        self.hold_event(name).set()

    def wait_for_hold(self, name: str, timeout: float = 10.0) -> bool:
        """True once a response has reached the hold `name`."""
        until = time.monotonic() + timeout
        while time.monotonic() < until:
            if name in self.waiting:
                return True
            time.sleep(0.01)
        return False

    def binding_error(self, body: dict) -> Optional[str]:
        if not self.enforce or body.get("model") not in PRESERVED:
            return None
        messages = body.get("messages") or []
        for k, msg in enumerate(messages):
            if msg.get("role") != "assistant" or not isinstance(msg.get("content"), list):
                continue
            for j, block in enumerate(msg["content"]):
                if block.get("type") != "thinking" or block.get("signature") not in self.bound:
                    continue
                model, prefix = self.bound[block["signature"]]
                if model != body["model"]:
                    continue   # another model's block: dropped, never an error
                if _canon(body.get("system"), body.get("tools"), messages[:k]) != prefix:
                    return (f"messages.{k}.content.{j}: Invalid `signature` in `thinking` "
                            "block. The block is bound to a different conversation.")
        return None

    def reply(self, body: dict, headers: dict, path: str = "") -> tuple[int, Any, str, dict]:
        """(status, payload, content type, the step). The payload is bytes for an error and a
        LIST of SSE events for a response, which the handler writes whole or one at a time."""
        self.requests.append(body)
        self.headers.append(headers)

        def refused(status: int, kind: str, message: str, details: Optional[dict] = None,
                    step: Optional[dict] = None) -> tuple[int, Any, str, dict]:
            error: dict = {"type": kind, "message": message}
            if details:
                error["details"] = details
            return (status, json.dumps({"type": "error", "error": error,
                                        "request_id": f"req_fake{len(self.requests)}"}).encode(),
                    "application/json", step or {})
        if self.strict and (why := violation(body, headers, path)) is not None:
            self.violations.append(why)
            unknown = why.startswith("model: ")
            return refused(404 if unknown else 400,
                           "not_found_error" if unknown else "invalid_request_error", why)
        err = self.binding_error(body)
        if err:
            self.rejected.append(err)
            return refused(400, "invalid_request_error", err)
        if not self.script and self.require_script:
            self.unscripted.append(body)
            # a 400: the SDK retries a 5xx, and one unscripted request would be recorded five times
            return refused(400, "invalid_request_error", "fake_anthropic: a request the script "
                                                         "has no step for (require_script)")
        step = self.script.pop(0) if self.script else {"blocks": [("text", "ok")]}
        if step.get("delay_s"):
            time.sleep(float(step["delay_s"]))
        if "error" in step:
            status, message, *kind = step["error"]
            return refused(status, kind[0] if kind else "invalid_request_error", message,
                           step.get("details"), step)
        blocks = []
        for b in step.get("blocks", []):
            if b[0] == "thinking":
                self._sig += 1
                sig = f"sig{self._sig}"
                self.bound[sig] = (body["model"], _canon(body.get("system"), body.get("tools"),
                                                         body.get("messages") or []))
                blocks.append(("thinking", b[1] if len(b) > 1 else "", sig))
            else:
                blocks.append(b)
        stop = step.get("stop") or ("tool_use" if any(b[0] in ("tool_use", "tool_use_raw")
                                                      for b in blocks) else "end_turn")
        # a step that names no speed answers at the speed it was asked for, as the API does
        speed = step.get("speed", body.get("speed"))
        return (200, _sse(blocks, stop, body["model"], step.get("stop_details"), speed,
                          usage_in=step.get("usage"), out_tokens=step.get("out", 5),
                          stream_error=(tuple(step["stream_error"]) if "stream_error" in step
                                        else None)),
                "text/event-stream", step)


PLACEHOLDER_KEY = "test-placeholder-not-a-key"     # only ever sent to this fake


def env(port: int) -> dict[str, str]:
    """What points every SDK client of a process at the fake on `port`: the base URL, a
    placeholder key that only ever reaches this fake, the api transport. For a child process
    (a source brain: a frozen one drops the base URL and pins Anthropic's)."""
    return {"ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
            "ANTHROPIC_API_KEY": PLACEHOLDER_KEY,
            "MIX_AGENT_LLM_BACKEND": "api"}


def serve(monkeypatch) -> tuple[FakeApi, Any]:
    """Start a fake on an ephemeral port and point every SDK client at it (ANTHROPIC_BASE_URL,
    a placeholder key that only ever reaches this fake, the api transport). Returns the fake and
    a shutdown callable."""
    fake, port, shutdown = start()
    for name, value in env(port).items():
        monkeypatch.setenv(name, value)
    return fake, shutdown


def start(fake: Optional[FakeApi] = None, port: int = 0) -> tuple[FakeApi, int, Any]:
    """The fake on 127.0.0.1 (`port` 0 = any free one), with no test framework: (the fake,
    its port, a shutdown callable). Nothing in the environment is touched - `env(port)` says
    what a client needs."""
    fake = fake or FakeApi()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_a):
            pass

        def _chunk(self, data: bytes) -> None:
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        def _drop(self) -> None:
            """The socket dies: no closing chunk, nothing more. Not a clean end of stream."""
            self.close_connection = True
            try:
                self.wfile.flush()
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        def _stream(self, events: list, step: dict) -> None:
            """One SSE event at a time, with whatever the step puts in the way."""
            pace = float(step.get("pace_s") or 0.0)
            for i, event in enumerate(events):
                if step.get("cut_after") is not None and i == int(step["cut_after"]):
                    return self._drop()
                if step.get("stall_after") is not None and i == int(step["stall_after"]):
                    time.sleep(float(step.get("stall_s") or 0.0))      # not one byte
                if step.get("hold") and i == int(step.get("hold_after") or 0):
                    name = str(step["hold"])
                    released = fake.hold_event(name)
                    fake.waiting.append(name)
                    until = time.monotonic() + float(step.get("hold_max_s") or 120.0)
                    while not released.wait(0.2) and time.monotonic() < until:
                        self._chunk(b"event: ping\ndata: {\"type\": \"ping\"}\n\n")
                self._chunk(event)
                if pace:
                    time.sleep(pace)
            if step.get("cut_after") is not None and int(step["cut_after"]) >= len(events):
                return self._drop()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

        def do_POST(self):
            began = time.monotonic()
            body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
            fake.paths.append(self.path)
            status, payload, ctype, step = fake.reply(
                body, {k.lower(): v for k, v in self.headers.items()}, self.path)
            # less a step's own `delay_s`: that is the script's wait, not this file's work
            fake.spent_s.append(max(0.0, time.monotonic() - began - float(step.get("delay_s") or 0.0)))
            try:
                self.send_response(status)
                self.send_header("content-type", ctype)
                for name, value in {**fake.limits, **(step.get("headers") or {})}.items():
                    self.send_header(name, str(value))
                if isinstance(payload, list) and any(k in step for k in _NETWORK_KEYS):
                    self.send_header("transfer-encoding", "chunked")
                    self.end_headers()
                    return self._stream(payload, step)
                if isinstance(payload, list):
                    payload = b"".join(payload)
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                fake.dropped += 1   # the client went away (a Stop) — nothing to answer

    class Server(ThreadingHTTPServer):
        daemon_threads = True

        def handle_error(self, request, client_address):
            """A client that hangs up (a Stop, a killed brain) is the test, not an error to
            print a traceback for; anything else is printed as ever."""
            if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError)):
                return      # (a reply cut short is counted where it was being written)
            super().handle_error(request, client_address)

    server = Server(("127.0.0.1", port), Handler)
    # `shutdown()` returns only when the serving loop next looks up, and that loop looks up
    # once per poll interval: at the default half second, every test that used a fake spent
    # half a second closing it (about a hundred tests - fifty seconds of the suite).
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01},
                     daemon=True).start()

    def shutdown():
        for held in list(fake._holds.values()):      # nothing waits on a fake that is going
            held.set()
        server.shutdown()
        server.server_close()
    return fake, server.server_address[1], shutdown
