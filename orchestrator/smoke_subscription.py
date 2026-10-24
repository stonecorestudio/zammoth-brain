"""Live smoke test for the subscription dev backend.

The subscription path (MIX_AGENT_LLM_BACKEND=subscription) calls Claude through
the Claude Agent SDK, which can't be exercised by the unit tests — they have no
subscription token and don't install the SDK. Run this on a machine logged into
your Claude subscription to confirm the wiring reaches Claude on your plan:

    pip install -r orchestrator/requirements-dev.txt   # claude-agent-sdk
    claude setup-token            # → export CLAUDE_CODE_OAUTH_TOKEN=<token>
    unset ANTHROPIC_API_KEY
    python -m orchestrator.smoke_subscription

It runs one no-tool completion (the run loop's plain turns / chat observe) and
one in-process tool round trip (the chat + investigate tool loops), printing
the replies and the usage the SDK reports. If both print, both subscription
paths can talk to Claude.
"""
from __future__ import annotations

import asyncio

from .llm_subscription import SubscriptionLLM


class _Outcome:
    def __init__(self, text: str) -> None:
        self.text = text
        self.error = False


async def main() -> None:
    # __init__ enforces: CLAUDE_CODE_OAUTH_TOKEN set, ANTHROPIC_API_KEY unset.
    llm = SubscriptionLLM("claude-opus-5-5")

    print("→ no-tool completion …")
    text, usage = await llm.observe(
        system="You are terse.",
        prompt="Reply with exactly: subscription backend OK")
    print("  reply:", text.strip())
    print("  usage:", usage)

    print("→ in-process tool round trip …")
    called: list[str] = []

    async def execute(name: str, args: dict) -> _Outcome:
        called.append(name)
        return _Outcome(f"{name} ran with {args}")

    async def on_delta(_chunk: str) -> None:
        pass

    async def on_tool(name: str, _args: dict) -> None:
        pass

    reply, usage2 = await llm.chat_turn(
        system="Call the `ping` tool exactly once, then reply 'done'.",
        prompt="User: ping the system\n",
        tool_defs=[{"name": "ping", "description": "Ping the system.",
                    "input_schema": {"type": "object", "properties": {}}}],
        execute=execute, on_delta=on_delta, on_tool=on_tool, max_turns=4)
    print("  reply:", reply.strip())
    print("  tools called:", called)
    print("  usage:", usage2)
    print("\nOK — subscription backend reached Claude on your plan.")


if __name__ == "__main__":
    asyncio.run(main())
