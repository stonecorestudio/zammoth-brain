"""The brain's LLM request surface, as the metered proxy needs it (2026-08-08).

The control plane validates every ``/v1/messages`` request's SHAPE — the system
block must start with a registered prompt prefix and every tool name must be on
the allowlist (``docs/11-cloud-handoff.md`` §11). This module is the single
source of that surface, DERIVED from the real prompts and tool definitions
rather than hand-copied — the same rule qa_eval's ``_MEASURED`` lives by,
because a hand-copied list goes stale the same day a tool is added.

    python -m orchestrator.llm_surface          # the registry, as JSON

Pipe it to the registration step (``python -m backend.app.manage
register-shapes --from -``) whenever a prompt constant or the tool surface
changes — publishing a build ADDS rows, it never replaces them, so old
installed apps keep working. A row is known by its hash; the label is that
family's name plus the first eight hex of the hash, so every build's prompt has
a row and a name of its own.

Why these prefixes cover everything:
- Every major path — chat in both shells, the plan/investigate loops, the
  reflection distiller's context turns — builds its system as
  ``[KNOWLEDGE_BLOCK, <persona>, …dynamic]``, so one non-exact row hashing
  KNOWLEDGE_BLOCK (thousands of bytes of entropy) anchors them all.
- The two one-line personas (critic, reflection) are EXACT rows: a prefix
  match on a 56-byte sentence would let anything ride after it, so those
  require the whole canonical text to match.
Anything that slips through this enumeration surfaces in the proxy's log-only
mode before enforcement is switched on — that rollout order is part of the
design, not an accident.
"""
from __future__ import annotations

import hashlib
import json


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prefix(label: str, text: str, *, exact: bool) -> dict:
    # The label carries the content's own hash (full audit 2026-10-02, reader S6·R6):
    # the control plane keeps labels unique, and a fixed one made a CHANGED prompt look
    # like the row it replaced - the publish reported OK and registered nothing.
    data = text.encode("utf-8")
    sha256 = hashlib.sha256(data).hexdigest()
    return {"label": f"{label}-{sha256[:8]}", "sha256": sha256,
            "prefix_len": len(data), "exact": exact}


def surface() -> dict:
    from .agents import PLANNER_TOOL_DEFS
    from .chat import daw_tool_defs, tool_defs
    from .knowledge import fallback_block, get_block

    prefixes = [
        # Chat (web + DAW), plan, investigate, reflection-context — all start here.
        _prefix("knowledge-block", get_block(), exact=False),
        # The degraded-mode block (knowledge fetch failed but the LLM proxy is
        # up): its turns must still pass shape validation.
        _prefix("knowledge-fallback", fallback_block(), exact=False),
        # The two fixed one-line personas (agents.py). Exact: a short prefix
        # would anchor nothing.
        _prefix("critic",
                "You are a second mix engineer reviewing a colleague's work.",
                exact=True),
        _prefix("reflection",
                "You distill finished mixing runs into terse lessons.",
                exact=True),
    ]

    names: set[str] = set()
    for defs in (tool_defs(False), tool_defs(True),
                 daw_tool_defs(False), daw_tool_defs(True),
                 PLANNER_TOOL_DEFS):
        for tool in defs:
            names.add(tool["name"])
    # The SDK's structured-output helper tool (output_format= paths).
    names.add("StructuredOutput")

    return {"prompt_prefixes": prefixes, "tool_names": sorted(names)}


def main() -> int:
    print(json.dumps(surface(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
