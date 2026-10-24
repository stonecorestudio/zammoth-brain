"""Subscription-backed LLM transport — a development convenience, not the
SaaS path.

Production talks through the metered cloud control plane; the direct Anthropic
API remains the evaluation path. For local iteration that bill adds up, so this module
routes the same turns through the Claude Code engine via the **Claude Agent
SDK**, which authenticates with your Claude subscription. Activated with
``MIX_AGENT_LLM_BACKEND=subscription``; this module is imported lazily on that
path only, so the shipped build does not depend on ``claude-agent-sdk``.

Setup (one time):
  1. install the Claude Code CLI (the SDK drives it) + ``pip install
     claude-agent-sdk`` (see ``requirements-dev.txt``);
  2. ``claude setup-token`` → ``export CLAUDE_CODE_OAUTH_TOKEN=…``;
  3. ensure ``ANTHROPIC_API_KEY`` is **unset** (the engine prefers it and would
     bill your API account — enforced in ``SubscriptionLLM.__init__``), and so are
     ``ANTHROPIC_AUTH_TOKEN``, a non-Anthropic ``ANTHROPIC_BASE_URL`` and the
     Bedrock / Vertex / Foundry switches (``rerouting_env``, 2026-09-10 audit P3).

Fidelity caveat: the Agent SDK runs its OWN agent loop and executes tools
itself, so this is not a byte-faithful twin of the API path. Keep
``python -m orchestrator.benchmark --check`` on the ``api`` backend as the
quality gate. The SDK calls below follow the documented Agent SDK surface and
should be validated live with ``python -m orchestrator.smoke_subscription``
before you lean on them.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional


# --- response shims: the slice of the anthropic SDK that the run loop reads ---

@dataclass
class _Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    # the part of the writes made at the 1-hour TTL (billed 2x, not 1.25x) — the CLI caches
    # every SDK request at 1 h on a subscription, and this was never read (2026-09-22 audit D·F6)
    cache_creation_1h_input_tokens: int = 0


@dataclass
class _TextBlock:
    text: str
    type: str = "text"


@dataclass
class TextResponse:
    """Shaped like the part of ``anthropic`` ``Message`` ``_create`` reads."""
    content: list
    usage: _Usage


@dataclass
class ParsedResponse:
    """Shaped like the part of a ``messages.parse`` result ``_parse`` reads."""
    parsed_output: Any
    usage: _Usage


def _usage_from(raw: Optional[dict]) -> _Usage:
    raw = raw or {}
    breakdown = raw.get("cache_creation") if isinstance(raw.get("cache_creation"), dict) else {}
    return _Usage(
        input_tokens=int(raw.get("input_tokens", 0) or 0),
        output_tokens=int(raw.get("output_tokens", 0) or 0),
        cache_read_input_tokens=int(raw.get("cache_read_input_tokens", 0) or 0),
        cache_creation_input_tokens=int(
            raw.get("cache_creation_input_tokens", 0) or 0),
        cache_creation_1h_input_tokens=int(breakdown.get("ephemeral_1h_input_tokens", 0) or 0),
    )


def _extract_json(text: str) -> dict:
    """The JSON object out of a reply, tolerating fences/prose (mirrors
    agents.extract_json; duplicated to keep this module import-cycle free)."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in reply: {text[:200]!r}")
    return json.loads(text[start:end + 1])


def render_history(history: list[dict]) -> str:
    """Render the chat transcript into a single prompt for the SDK. The
    subscription chat path keeps history as plain user/assistant text, so any
    structured tool blocks are skipped."""
    lines = []
    for message in history:
        content = message.get("content")
        if isinstance(content, list):
            # A message stored as blocks keeps its WORDS (Phase 64): skipped whole, a turn whose
            # content was a list lost what the user said along with whatever else it held.
            # Tool calls and results stay out, as before.
            if any(isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result")
                   for b in content):
                continue
            content = "\n\n".join(str(b.get("text") or "") for b in content
                                   if isinstance(b, dict) and b.get("type") == "text"
                                   and str(b.get("text") or "").strip())
        if not isinstance(content, str) or not content:
            continue
        who = "User" if message.get("role") == "user" else "Assistant"
        lines.append(f"{who}: {content}")
    return "\n\n".join(lines)


# IDLE ceiling on ONE SDK request: the longest silence (no message from the
# SDK) before the request is cut off. The SDK loop has no timeout of its own;
# a single wedged turn used to hang the pass forever (2026-08-09). It was a
# TOTAL ceiling until 2026-09-07 — which became the new turn cap the moment
# chat lost max_turns (a 40-call build takes longer than 600 s of healthy
# work), so it now re-arms on every message and only a genuine hang trips it.
SDK_REQUEST_TIMEOUT_S = float(
    os.environ.get("MIX_AGENT_SDK_TIMEOUT_S", "600"))

# How often a stoppable request (chat) checks its stop condition.
STOP_POLL_S = 0.2


class _IdleGuard:
    """Suspends the idle ceiling while one of OUR tool handlers runs (2026-09-10 audit
    P1-12). The SDK sends no message while a tool executes, so a chat-started `start_run`
    longer than the ceiling read as a hang: the turn was cut off while the run kept
    applying in its worker thread, orphaned from its cancel handle and the pass slot. The
    ceiling now times the MODEL's silence only; tools carry their own bounds (the run
    loop's cancel checks, the DAW's command timeouts)."""

    def __init__(self) -> None:
        self._timeout: Optional[asyncio.Timeout] = None
        self._limit = SDK_REQUEST_TIMEOUT_S
        self.depth = 0

    def bind(self, timeout: "asyncio.Timeout", limit: float) -> None:
        self._timeout, self._limit = timeout, limit

    def unbind(self) -> None:
        self._timeout = None

    def _set(self, when: Optional[float]) -> None:
        if self._timeout is None:
            return
        try:
            self._timeout.reschedule(when)
        except RuntimeError:   # the request already ended: nothing left to move
            pass

    def activity(self) -> None:
        """An SDK message arrived: re-arm — unless a tool is still mid-flight."""
        if self.depth == 0:
            self._set(asyncio.get_running_loop().time() + self._limit)

    def enter(self) -> None:
        self.depth += 1
        self._set(None)

    def exit(self) -> None:
        self.depth = max(0, self.depth - 1)
        self.activity()


class TurnStopped(Exception):
    """The caller's stop condition flipped (the DAW's Stop) and the SDK request was
    taken down between tool calls. Carries what the model had said and spent so far."""

    def __init__(self, text: str, usage: "_Usage"):
        super().__init__("stopped by the user")
        self.text = text
        self.usage = usage


def _with_usage(exc: BaseException, usage: "_Usage") -> BaseException:
    """Hang what a request burned on the exception that ended it, as TurnStopped carries it
    (2026-09-10 audit P3, 01·F10). The ResultMessage of a failed call reports its usage before
    the error is raised, and a failed chat turn recorded zero — eval token lines and the
    cache-share check read low on exactly the failing turns."""
    if getattr(exc, "usage", None) is None:
        try:
            exc.usage = usage
        except (AttributeError, TypeError):
            pass
    return exc


def _run_sync(coro):
    """Run an async SDK call from synchronous code (the pass loop). Works
    whether or not the calling thread already has a running event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class SubscriptionError(RuntimeError):
    """An API failure the CLI reported in its result (2026-09-10 audit P2, 01·F6). The SDK
    turned every one into "Claude Code returned an error result: success" — the subtype of
    an API error IS "success" — and threw away the CLI's own words ("You've hit your limit ·
    resets 3pm"). The text here keeps them and names the remedy; `status` is the HTTP code."""

    def __init__(self, text: str, status: Optional[int] = None):
        super().__init__(text)
        self.status = status


def _error_result_text(status: Optional[int], detail: str) -> str:
    """What a failed subscription request says. The DAW withholds its "retry" verb on
    "usage limit" and "subscription" (App.cpp setStatus), which is right for both: the
    limit and a dead token fail identically on a retry. An overload is worth a retry."""
    detail = (detail or "").strip()
    tail = f" ({detail})" if detail else ""
    if status == 429:
        return f"Claude usage limit reached{tail}"
    if status in (401, 403):
        return ("the Claude subscription token was rejected — run `claude setup-token` and "
                f"export CLAUDE_CODE_OAUTH_TOKEN again{tail}")
    if status in (500, 502, 503, 529):
        return f"Claude is overloaded right now (HTTP {status}) — retry in a moment{tail}"
    if status:
        return f"Claude returned HTTP {status}{tail}"
    return f"Claude Code failed{tail}" if detail else "Claude Code failed with no detail"


#: Environment that re-routes the Claude Code CLI away from the subscription (2026-09-10 audit
#: P3, 01·F12). The SDK hands the CLI the brain's WHOLE environment, and the CLI prefers an API
#: key or a bearer token over the OAuth token and a provider switch over Anthropic, so a dev
#: shell exporting any of these silently billed another account or provider.
#: Key/transport sources that would bill or route somewhere other than the subscription. The SDK
#: hands the CLI the brain's WHOLE environment, so this list is the door. ANTHROPIC_UNIX_SOCKET and
#: CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR were read straight out of the bundled CLI (2026-09-11
#: pre-merge review, C10): the first re-routes API transport, the second is an API-key source.
_REROUTING_KEYS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_UNIX_SOCKET",
                   "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR")
#: …and its provider switch reads two more than we checked (same review): the CLI's own
#: `USE_BEDROCK / USE_ANTHROPIC_AWS / USE_MANTLE / …` chain.
_PROVIDER_SWITCHES = ("CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX",
                      "CLAUDE_CODE_USE_FOUNDRY", "CLAUDE_CODE_USE_ANTHROPIC_AWS",
                      "CLAUDE_CODE_USE_MANTLE")
_OFFICIAL_BASE_URL = "https://api.anthropic.com"


def rerouting_env(environ=None) -> list[str]:
    """The variables in `environ` that would send a subscription turn somewhere else."""
    env = os.environ if environ is None else environ
    bad = [k for k in _REROUTING_KEYS if env.get(k)]
    bad += [k for k in _PROVIDER_SWITCHES
            if str(env.get(k, "")).strip().lower() in ("1", "true", "yes", "on")]
    base = str(env.get("ANTHROPIC_BASE_URL", "")).strip().rstrip("/")
    if base and base != _OFFICIAL_BASE_URL:
        bad.append("ANTHROPIC_BASE_URL")
    return bad


def neutral_cwd() -> str:
    """Where the Claude Code CLI runs (audit follow-ups 2026-09-11, item 5). The CLI keys its
    auto-memory on the working directory, and `setting_sources=[]` does not switch it off: run
    from the repo (the brain's cwd), it loaded the developer's own Claude Code project memory —
    "Contents of ~/.claude/projects/<repo>/memory/MEMORY.md" — into EVERY brain request, and with
    it the username. Measured with the CLI pointed at a local capture server: from the repo the
    home path was in the request, from this directory it was not. A neutral directory under the
    system temp carries no memory, no instructions and no username."""
    path = Path(tempfile.gettempdir()) / "zammoth-sdk"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def cli_version() -> str:
    """The Claude Code CLI the SDK actually spawns: its BUNDLED binary, not the system `claude`
    (2026-09-10 audit P3, 01·F14), so upgrading the system CLI changes nothing. Recorded in
    eval provenance so a baseline says which one it measured."""
    try:
        from claude_agent_sdk._cli_version import __cli_version__
        return str(__cli_version__)
    except Exception:  # noqa: BLE001 — a dev-only dependency; absence is an answer
        return "unknown"


def _is_max_turns_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "claude code returned an error result" in text
        and "maximum number of turns" in text
    )


#: The CLI's retries per request, matched to the api transport's (llm_backend.API_MAX_RETRIES).
SDK_MAX_RETRIES = 4


class SubscriptionLLM:
    """Routes orchestrator turns through the Claude Agent SDK on the user's
    subscription. The run loop (``agents.py``) calls the sync ``create`` /
    ``parse`` / ``investigate``; the chat layer (``chat.py``) awaits
    ``chat_turn`` / ``observe``."""

    SERVER = "mixagent"

    def __init__(self, model: str):
        self._model = model
        # A KNOWN model with no effort ladder (none on the picker today) gets thinking switched off, as the api
        # path sends it (llm_models); an unknown id keeps the CLI's default, like a plain request.
        from .llm_models import spec
        known = spec(model)
        self._no_thinking = bool(known is not None and not known.efforts)
        # The Claude Code engine prefers ANTHROPIC_API_KEY over the
        # subscription when both are present, and would silently bill the API
        # account — make the dev requirement explicit instead of surprising.
        if os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "MIX_AGENT_LLM_BACKEND=subscription, but ANTHROPIC_API_KEY is "
                "set — the Claude Code engine would use the API key and bill "
                "your API account. Unset ANTHROPIC_API_KEY in the dev "
                "environment.")
        if (rerouted := rerouting_env()):
            raise RuntimeError(
                "MIX_AGENT_LLM_BACKEND=subscription, but " + ", ".join(rerouted)
                + (" is" if len(rerouted) == 1 else " are") + " set — the Claude Code "
                "engine would send these turns to another account or provider instead of "
                "your subscription. Unset them in the dev environment.")
        if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
            raise RuntimeError(
                "MIX_AGENT_LLM_BACKEND=subscription needs a subscription "
                "token. Run `claude setup-token` and export "
                "CLAUDE_CODE_OAUTH_TOKEN.")

    # -- core ----------------------------------------------------------------
    async def _run(self, *, system: str, prompt: str,
                   sdk_tools: Optional[list] = None,
                   allowed_tools: Optional[list] = None,
                   max_turns: int = 1,
                   on_text: Optional[Callable[[str], Awaitable[None]]] = None,
                   effort: Optional[str] = None,
                   usage: Optional[_Usage] = None,
                   guard: Optional[_IdleGuard] = None,
                   should_stop: Optional[Callable[[], bool]] = None,
                   max_tokens: Optional[int] = None,
                   timer: Any = None,
                   ) -> tuple[str, _Usage]:
        """One agentic request through the SDK. Tools (if any) are executed by
        the SDK via the in-process handlers; we collect assistant text and the
        final usage. `usage` may be a caller-owned accumulator: the tokens a
        request burned before an exception (a max-turns exhaustion) are then
        still visible to the caller — returning a fresh zero used to record a
        dozen internal turns as $0, so the run budget never tripped (2026-09-01)."""
        from claude_agent_sdk import (  # deferred: dev-only dependency
            ClaudeAgentOptions, create_sdk_mcp_server, query)

        # Token streaming (chat): ask the SDK for partial messages so the reply
        # flows out like a real LLM chat instead of landing as one static block.
        streaming = on_text is not None
        opts: dict[str, Any] = dict(
            system_prompt=system, model=self._model,
            permission_mode="bypassPermissions", max_turns=max_turns,
            include_partial_messages=streaming,
            setting_sources=[],  # ignore the repo's own CLAUDE.md / settings
            # The brain's only tools are the MCP ones registered below. Without
            # these two, Claude Code also shipped its BUILT-IN tools (Bash, Read,
            # Edit, Write, …) and every MCP server configured for the cwd
            # (~/.claude.json's project entry): ~90k tokens on every request —
            # an operate case cost 201k prompt tokens instead of 111k, measured
            # 2026-09-03 — and, under bypassPermissions, a live Bash/Edit on the
            # user's machine that `allowed_tools` never restricted (it governs
            # permission prompts, not the tool list).
            tools=[], strict_mcp_config=True,
            # Every request is a fresh CLI session, and each one was persisted as a JSONL
            # transcript under ~/.claude/projects/<brain cwd>/ for 30 days — the knowledge
            # block, the catalog, the project, the user's words — written and never read
            # (2026-09-10 audit P3, 01·F13). Accepted in the SDK's stream-json mode
            # (checked against the bundled CLI: an unknown flag exits 1, this one 0).
            extra_args={"no-session-persistence": None},
            # Never the repo: the CLI loads Claude Code memory keyed on its cwd (neutral_cwd).
            cwd=neutral_cwd())
        # Phase 51: small structured turns (brief compile, reflection) run at
        # LOW effort — they are JSON extraction, not mixing judgment, and at
        # the default adaptive effort they were billing planner-grade thinking
        # on every one (the baseline's 24% post-pass share). Planner + critic
        # turns pass no effort and keep the default.
        if effort:
            opts["thinking"] = {"type": "adaptive"}
            opts["effort"] = effort
        elif self._no_thinking:
            # the api path sends NO thinking to a model without an effort ladder;
            # the CLI's default gave it a 31,999-token budget (2026-09-22 audit D·F1)
            opts["thinking"] = {"type": "disabled"}
        # Make the CLI honour what the api path sends (2026-09-22 audit D·F1/F2). Claude Code
        # rebuilds the request from its own model tables: it pinned some models' effort unless
        # ~/.claude.json carried /effort's unpin flags (a fresh machine and CI do not), let an
        # ambient CLAUDE_CODE_EFFORT_LEVEL override the brain outright, dropped every max_tokens
        # the brain set (32k-64k instead of 16k), compacted long turns and retried 10x — so a
        # behaviour seen on the sub predicted nothing about the key. These outrank all of that.
        cli_env = {"DISABLE_COMPACT": "1", "CLAUDE_CODE_MAX_RETRIES": str(SDK_MAX_RETRIES)}
        if effort:
            cli_env["CLAUDE_CODE_EFFORT_LEVEL"] = effort
        if max_tokens:
            cli_env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(int(max_tokens))
        opts["env"] = cli_env
        if sdk_tools:
            server = create_sdk_mcp_server(
                name=self.SERVER, version="1.0.0", tools=sdk_tools)
            opts["mcp_servers"] = {self.SERVER: server}
            opts["allowed_tools"] = allowed_tools or []
        options = ClaudeAgentOptions(**opts)

        parts: list[str] = []
        usage = usage if usage is not None else _Usage()
        limit = SDK_REQUEST_TIMEOUT_S

        async def consume() -> None:
            async with asyncio.timeout(limit) as idle:
                if guard is not None:
                    guard.bind(idle, limit)
                try:
                    await self._consume(query(prompt=prompt, options=options),
                                        parts, usage, on_text, streaming, idle, guard, timer)
                finally:
                    if guard is not None:
                        guard.unbind()

        try:
            if should_stop is None:
                await consume()
            else:
                await self._consume_until_stopped(consume, should_stop, parts, usage)
        except TimeoutError:
            raise _with_usage(RuntimeError(
                f"subscription request went silent for {limit:.0f}s — "
                "the SDK turn was cut off"), usage) from None
        except BaseException as exc:
            _with_usage(exc, usage)
            raise
        return "".join(parts), usage

    @staticmethod
    async def _consume_until_stopped(consume, should_stop: Callable[[], bool],
                                     parts: list[str], usage: _Usage) -> None:
        """Run the request while polling `should_stop` (2026-09-10 audit P1-11: the DAW's
        Stop never reached the SDK loop, so a build kept applying command after command
        until the DAW killed the brain 45 s later). Cancelling the consuming task unwinds
        the SDK's own generators, whose `finally` terminates the CLI subprocess."""
        task = asyncio.ensure_future(consume())
        try:
            while not task.done():
                await asyncio.wait({task}, timeout=STOP_POLL_S)
                if not task.done() and should_stop():
                    task.cancel()
                    await asyncio.wait({task})
                    if task.cancelled():
                        raise TurnStopped("".join(parts), usage)
        finally:
            if not task.done():   # our own caller was cancelled: take the request down too
                task.cancel()
                await asyncio.wait({task})
        task.result()   # a finished request's own error (the idle TimeoutError included)

    @staticmethod
    async def _consume(stream, parts: list[str], usage: _Usage,
                       on_text, streaming: bool,
                       idle: Optional["asyncio.Timeout"] = None,
                       guard: Optional[_IdleGuard] = None,
                       timer: Any = None) -> None:
        """`timer` (turn_timing.TurnTimer, Phase 54) is told when each response starts and
        ends and what it generated - the round count and model time of a turn, which nothing
        recorded on this path."""
        from claude_agent_sdk import (  # deferred: dev-only dependency
            AssistantMessage, ResultMessage, StreamEvent, TextBlock)

        loop = asyncio.get_running_loop()
        # The CLI's own words for a failed API call: they arrive as a synthetic assistant
        # text block flagged `error`, and again in the ResultMessage's `result`.
        api_error = ""
        # Per-API-response usage as it streams, so a caller's budget can see a request's
        # spend BEFORE it ends (2026-09-10 audit P2, 05·F8: the planner overshot a $0.50
        # budget 5.3× because spend was only known at the ResultMessage). One response can
        # arrive as several AssistantMessages sharing an id; count each id once. The
        # ResultMessage's cumulative figure then replaces the running one.
        seen_ids: set[str] = set()
        # Which response the last text came from. Text from the NEXT response starts a new
        # paragraph (2026-09-10 audit P3, 01·F11): the blocks were glued ("…first.It sits
        # at…") into history, the session digest and Ask's plan narration, where the API
        # path joins its rounds with a blank line. One response can arrive as several
        # AssistantMessages sharing an id, and those stay one paragraph.
        last_text_from: object = None
        # The response now streaming, as its own events describe it: `message_start` carries
        # its id and what it read, `message_delta` what it generated in the end and how it
        # stopped. The AssistantMessage's usage is the count at the START of the response (a
        # seven-round turn recorded 65 output tokens), so the timer is told from the stream;
        # that message's figure is the fallback for a stream that carried none.
        current: dict = {"id": None, "usage": {}, "out": None, "stop": None, "fallback": None}
        counted: set[str] = set()      # ids whose output the timer already holds
        async for message in stream:
            if guard is not None:    # activity re-arms — unless one of our tools is mid-flight
                guard.activity()
            elif idle is not None:   # activity: the idle guard re-arms
                idle.reschedule(loop.time() + SDK_REQUEST_TIMEOUT_S)
            if isinstance(message, StreamEvent):
                # partial text deltas (the raw Anthropic content_block_delta) →
                # stream them out token-by-token; the AssistantMessage below carries
                # the final text, so don't double-emit there when streaming.
                ev = message.event or {}
                kind = ev.get("type")
                if kind == "message_start":
                    head = ev.get("message") if isinstance(ev.get("message"), dict) else {}
                    current = {"id": head.get("id"), "out": None, "stop": None, "fallback": None,
                               "usage": dict(head.get("usage") or {})}
                elif kind == "message_delta":
                    said = ev.get("usage") if isinstance(ev.get("usage"), dict) else {}
                    if isinstance(said.get("output_tokens"), (int, float)):
                        current["out"] = int(said["output_tokens"])
                    # the delta can restate the input side (cache reads settle late)
                    current["usage"].update({k: v for k, v in said.items() if k != "output_tokens"})
                    stop = (ev.get("delta") or {}).get("stop_reason")
                    current["stop"] = stop or current["stop"]
                if timer is not None:
                    if kind == "message_stop":
                        out = current["out"] if current["out"] is not None else current["fallback"]
                        if out is not None and current["id"]:
                            counted.add(current["id"])
                        timer.end_round(out or 0, usage=current["usage"] or None,
                                        stop_reason=current["stop"])
                        current = {"id": None, "usage": {}, "out": None, "stop": None,
                                   "fallback": None}
                    else:
                        timer.stream_started()
                        if (kind == "content_block_start"
                                and (ev.get("content_block") or {}).get("type") == "tool_use"):
                            timer.announce_call()
                if ev.get("type") == "content_block_delta":
                    delta = ev.get("delta") or {}
                    if delta.get("type") == "text_delta" and on_text is not None:
                        await on_text(delta.get("text", ""))
            elif isinstance(message, AssistantMessage):
                if getattr(message, "error", None):
                    # Not the model speaking: the CLI's notice of a failed call. Keep it
                    # out of the reply and use it as the error's detail below.
                    api_error = " ".join(b.text for b in message.content
                                         if isinstance(b, TextBlock)).strip() or api_error
                    continue
                mid = getattr(message, "message_id", None)
                raw = getattr(message, "usage", None)
                if raw and mid and mid not in seen_ids:
                    seen_ids.add(mid)
                    step = _usage_from(raw)
                    if timer is not None:
                        if not streaming:   # no stream events: the message is the round
                            timer.stream_started()
                            timer.end_round(step.output_tokens, usage=raw)
                        elif mid == current["id"]:
                            current["fallback"] = step.output_tokens   # its stream is still open
                        elif mid not in counted:
                            timer.add_output(step.output_tokens)
                    usage.input_tokens += step.input_tokens
                    usage.output_tokens += step.output_tokens
                    usage.cache_read_input_tokens += step.cache_read_input_tokens
                    usage.cache_creation_input_tokens += step.cache_creation_input_tokens
                    usage.cache_creation_1h_input_tokens += step.cache_creation_1h_input_tokens
                if any(isinstance(b, TextBlock) and b.text for b in message.content):
                    source = mid if mid else object()   # no id: every message is its own
                    if parts and source != last_text_from:
                        parts.append("\n\n")
                    last_text_from = source
                for block in message.content:
                    if isinstance(block, TextBlock):
                        parts.append(block.text)
                        if on_text is not None and not streaming:
                            await on_text(block.text)
            elif isinstance(message, ResultMessage):
                if getattr(message, "usage", None):
                    # Mutate, don't rebind — the caller holds this _Usage.
                    fresh = _usage_from(message.usage)
                    usage.input_tokens = fresh.input_tokens
                    usage.output_tokens = fresh.output_tokens
                    usage.cache_read_input_tokens = fresh.cache_read_input_tokens
                    usage.cache_creation_input_tokens = \
                        fresh.cache_creation_input_tokens
                    usage.cache_creation_1h_input_tokens = \
                        fresh.cache_creation_1h_input_tokens
                # An API error ends as subtype "success" with is_error set; the SDK then
                # raises "Claude Code returned an error result: success" and every limit,
                # dead token and overload read the same (01·F6). Raise the real words here.
                # Other error subtypes (error_max_turns, …) keep the SDK's own exception.
                if (getattr(message, "is_error", False)
                        and getattr(message, "subtype", "") == "success"):
                    status = getattr(message, "api_error_status", None)
                    detail = (getattr(message, "result", None) or api_error
                              or "; ".join(getattr(message, "errors", None) or []))
                    raise SubscriptionError(_error_result_text(status, detail), status)

    # -- run loop (sync) -----------------------------------------------------
    @staticmethod
    def _effort_of(effort: Optional[str], kwargs: dict) -> Optional[str]:
        # The API path spells effort as `output_config={"effort": ...}` (agents._think_kwargs);
        # the shim took only its own `effort=` and dropped the other spelling in **kwargs, so
        # MIX_AGENT_EFFORT=low was honoured on the planner and silently ignored on the plain
        # text turns, revise, chat and the critic (2026-09-01 sweep). One translation, here.
        return effort or ((kwargs.get("output_config") or {}).get("effort") if kwargs else None)

    def create(self, effort: Optional[str] = None, **kwargs) -> TextResponse:
        """Stand-in for ``client.messages.create`` for no-tool turns."""
        effort = self._effort_of(effort, kwargs)
        text, usage = _run_sync(self._run(
            system=self._system_text(kwargs.get("system")),
            prompt=self._prompt_text(kwargs.get("messages")), effort=effort,
            max_tokens=kwargs.get("max_tokens")))
        return TextResponse(content=[_TextBlock(text)], usage=usage)

    def parse(self, *, output_format, effort: Optional[str] = None,
              **kwargs) -> ParsedResponse:
        """Stand-in for ``client.messages.parse``: ask for JSON, validate it
        against the same pydantic model the API path uses."""
        effort = self._effort_of(effort, kwargs)
        system = self._system_text(kwargs.get("system")) + (
            "\n\nRespond with ONLY a JSON object conforming to this schema — "
            "no prose, no code fences:\n"
            + json.dumps(output_format.model_json_schema()))
        text, usage = _run_sync(self._run(
            system=system, prompt=self._prompt_text(kwargs.get("messages")),
            effort=effort, max_tokens=kwargs.get("max_tokens")))
        try:
            parsed = output_format.model_validate(_extract_json(text))
        except (ValueError, TypeError):   # pydantic's ValidationError is a ValueError
            # the same outcome the api path reports (agents.LLMTurnEnded), never a raise here
            parsed = None
        return ParsedResponse(parsed_output=parsed, usage=usage)

    def investigate(self, *, system: str, opening_prompt: str,
                    planner_tool_defs: list,
                    engine_call: Callable[[str, dict], Any],
                    validate: Callable[[list], list],
                    max_rounds: int,
                    effort: Optional[str] = None,
                    should_stop: Optional[Callable[[_Usage], str]] = None,
                    outcome: Optional[dict] = None,
                    max_tokens: Optional[int] = None) -> tuple[list, _Usage]:
        """The Phase 12 investigating planner, run as a native SDK tool loop:
        the engine reads/probe + the terminal ``submit_proposals`` are
        registered as in-process tools and the SDK drives the rounds.

        `should_stop(spent)` returns a reason ("cancelled", "spend limit", "deadline") or
        "" (2026-09-10 audit P2, 05·F8): Stop, the run budget and the pass deadline used to
        be checked once, BEFORE the request, so 11 model turns ran after Stop and a $0.50
        budget overshot 5.3×. It is now polled while the request runs — tools refuse once
        it flips, and the request is taken down as soon as no tool is in flight. `outcome`
        receives `empty_reason` when the pass comes back empty for a reason worth naming."""
        captured: dict[str, Any] = {"proposals": None, "invalid": None}
        note = outcome if outcome is not None else {}
        guard = _IdleGuard()   # a long probe render must not read as a hang (P1-12)
        spent = _Usage()       # accumulates even when the request dies mid-way

        def stop_reason() -> str:
            return (should_stop(spent) or "") if should_stop is not None else ""

        sdk_tools, allowed = self._planner_tools(
            planner_tool_defs, engine_call, captured, guard,
            validate=validate, stop=stop_reason)
        # The SAME round budget as the api planner (2026-09-22 audit D·F3). This gave the sub
        # max(rounds x 2, 12) = 12 "for extra internal turns around MCP tool calls" — but the
        # CLI counts exactly one turn per model round, so a pass that needed 7-12 rounds landed
        # on the sub (where the technique gate runs) and came back empty in the product.
        # Max-turn exhaustion still degrades to an empty pass instead of failing the run.
        try:
            _text, usage = _run_sync(self._run(
                system=system, prompt=opening_prompt, sdk_tools=sdk_tools,
                allowed_tools=allowed, max_turns=max_rounds,
                effort=effort, usage=spent, guard=guard, max_tokens=max_tokens,
                should_stop=((lambda: bool(stop_reason()) and guard.depth == 0)
                             if should_stop is not None else None)))
        except TurnStopped:
            why = stop_reason() or "stopped"
            note["empty_reason"] = why
            print(f"[planner] request taken down: {why}", file=sys.stderr, flush=True)
            # Proposals already submitted and validated still count — unless the user
            # pressed Stop, which means "no more moves".
            if why != "cancelled" and captured["proposals"]:
                note.pop("empty_reason", None)
                return captured["proposals"], spent
            return [], spent
        except Exception as exc:
            if _is_max_turns_error(exc):
                print(f"[planner] empty pass: MAX TURNS hit before submit_proposals "
                      f"({exc})", file=sys.stderr, flush=True)
                if captured["proposals"]:
                    return captured["proposals"], spent
                note["empty_reason"] = ("invalid proposals" if captured["invalid"]
                                        else "out of turns")
                return [], spent          # what it cost, not a zero
            print(f"[planner] SDK error: {exc!r}", file=sys.stderr, flush=True)
            raise
        proposals = captured["proposals"]
        if proposals:
            return proposals, usage       # validated in the submit handler
        if captured["invalid"]:
            note["empty_reason"] = "invalid proposals"
            print(f"[planner] empty pass: proposals never validated: {captured['invalid']}",
                  file=sys.stderr, flush=True)
        else:
            print(f"[planner] empty pass: submit_proposals was NOT called "
                  f"(model text {_text[:200]!r})", file=sys.stderr, flush=True)
        return [], usage

    # -- chat (async) --------------------------------------------------------
    async def chat_turn(self, *, system: str, prompt: str, tool_defs: list,
                        execute: Callable[[str, dict], Awaitable[Any]],
                        on_delta: Callable[[str], Awaitable[None]],
                        on_tool: Callable[[str, dict], Awaitable[None]],
                        max_turns: Optional[int],
                        effort: Optional[str] = None,
                        should_stop: Optional[Callable[[], bool]] = None,
                        max_tokens: Optional[int] = None,
                        timer: Any = None,
                        ) -> tuple[str, _Usage]:
        # `effort` was missing here until 2026-09-03: the API path sends the
        # session's reasoning level (`high` by default, the composer's picker
        # otherwise) as output_config.effort, while this path sent nothing and
        # the SDK ran every chat turn at Claude Code's own default — a higher
        # level than production, on every eval turn and every dawdev turn, and
        # the composer's picker was silently ignored on the dev backend.
        # `should_stop` (2026-09-10 audit P1-11): the DAW's Stop, which this path used to
        # ignore. The guard keeps the idle ceiling from firing while a tool runs (P1-12).
        guard = _IdleGuard()
        sdk_tools, allowed = self._chat_tools(tool_defs, execute, on_tool, guard)
        return await self._run(
            system=system, prompt=prompt, sdk_tools=sdk_tools,
            allowed_tools=allowed, max_turns=max_turns, on_text=on_delta,
            effort=effort, guard=guard, should_stop=should_stop, max_tokens=max_tokens,
            timer=timer)

    async def observe(self, *, system: str, prompt: str,
                      effort: Optional[str] = None) -> tuple[str, _Usage]:
        return await self._run(system=system, prompt=prompt, max_turns=1,
                               effort=effort, max_tokens=4000)

    # -- in-process tool builders -------------------------------------------
    def _planner_tools(self, defs, engine_call, captured,
                       guard: Optional[_IdleGuard] = None,
                       validate: Optional[Callable[[list], list]] = None,
                       stop: Optional[Callable[[], str]] = None):
        if not defs:
            # Keep the module's promise (and the shipped-cloud-client CI leg's
            # proof) that the dev-only SDK is imported at CALL time: a run
            # with no tools must not import it at all.
            return [], []
        from claude_agent_sdk import tool
        sdk_tools = []

        def refused() -> Optional[dict]:
            why = stop() if stop is not None else ""
            if not why:
                return None
            return {"content": [{"type": "text", "text": (
                f"not run — the pass is over ({why}). Call no more tools and end your "
                "turn now.")}], "is_error": True}

        for defn in defs:
            name = defn["name"]
            schema = defn.get("input_schema",
                              {"type": "object", "properties": {}})
            if name == "submit_proposals":
                async def fn(args, _cap=captured):
                    if (r := refused()) is not None:
                        return r
                    raw = list((args or {}).get("proposals", []))[:3]
                    # Validate HERE, while the model can still repair (2026-09-10 audit
                    # P2, 05·F10): it used to hear "proposals received" before anything
                    # was checked, so one schema slip dropped the whole pass as "nothing
                    # to propose" — the API path returns the error for a resubmit.
                    if validate is not None:
                        try:
                            _cap["proposals"] = validate(raw)
                        except Exception as exc:
                            _cap["invalid"] = str(exc)
                            return {"content": [{"type": "text", "text": (
                                f"invalid proposals: {exc} — fix them and call "
                                "submit_proposals again")}], "is_error": True}
                    else:
                        _cap["proposals"] = raw
                    return {"content": [{"type": "text",
                                         "text": "proposals received"}]}
            else:
                async def fn(args, _name=name):
                    if (r := refused()) is not None:
                        return r
                    payload_args = ({"req": dict(args or {})}
                                    if _name == "probe_actions"
                                    else dict(args or {}))
                    if guard is not None:
                        guard.enter()   # a probe render is not a silent model (P1-12)
                    try:
                        result = await asyncio.to_thread(
                            engine_call, _name, payload_args)
                        return {"content": [{"type": "text",
                                             "text": json.dumps(result)}]}
                    except Exception as exc:  # back to the model as a tool error
                        return {"content": [{"type": "text",
                                             "text": f"error: {exc}"}],
                                "is_error": True}
                    finally:
                        if guard is not None:
                            guard.exit()
            sdk_tools.append(tool(name, defn["description"], schema)(fn))
        allowed = [f"mcp__{self.SERVER}__{d['name']}" for d in defs]
        return sdk_tools, allowed

    def _chat_tools(self, defs, execute, on_tool, guard: Optional[_IdleGuard] = None):
        if not defs:
            return [], []   # same deferred-SDK promise as _planner_tools
        from claude_agent_sdk import tool
        sdk_tools = []
        for defn in defs:
            name = defn["name"]
            schema = defn.get("input_schema",
                              {"type": "object", "properties": {}})

            async def fn(args, _name=name):
                call_args = dict(args or {})
                if guard is not None:
                    guard.enter()   # a running tool is not a silent model
                try:
                    await on_tool(_name, call_args)
                    outcome = await execute(_name, call_args)
                finally:
                    if guard is not None:
                        guard.exit()
                result = {"content": [{"type": "text", "text": outcome.text}]}
                if getattr(outcome, "error", False):
                    result["is_error"] = True
                return result
            sdk_tools.append(tool(name, defn["description"], schema)(fn))
        allowed = [f"mcp__{self.SERVER}__{d['name']}" for d in defs]
        return sdk_tools, allowed

    # -- request adapters ----------------------------------------------------
    @staticmethod
    def _system_text(system) -> str:
        if system is None:
            return ""
        if isinstance(system, str):
            return system
        return "\n\n".join(block.get("text", "") for block in system
                           if isinstance(block, dict))

    @staticmethod
    def _prompt_text(messages) -> str:
        if not messages:
            return ""
        parts = [m["content"] for m in messages
                 if isinstance(m.get("content"), str)]
        return "\n\n".join(parts)
