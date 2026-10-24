"""Deep reference the agent pulls ON DEMAND (2026-08-08).

`knowledge.py` is the always-on core: it rides in every prompt, so it holds only
what the agent needs on EVERY turn — how to read our own numbers, where each
band lives, what a producer's word probably means. This module is the other
half: the depth a master engineer has recall for but does not recite, loaded
one topic at a time through the `lookup_knowledge` tool.

The split is a token argument, not a knowledge one. Everything here would be
correct in the core block; none of it is needed on a turn about a kick drum.

House rules, same as the core block:
- Working opinion, not contract. What the engine MEASURES always wins.
- Tie knowledge to what this app can actually do. A fact the agent cannot act
  on, or that names a tool we do not have, is worse than no fact — it produces
  confident advice the user cannot follow.
- Say what is NOT fixable here. Naming the limit is part of the craft.
"""
from __future__ import annotations

# Each entry: (one-line index summary shown in the tool description, body).
# Keep summaries short — they are always-on cost. Keep bodies self-contained:
# the agent may pull exactly one and answer from it.

# The CONTENT lives in the private `knowledge_content` module — excluded from
# the public brain-source cut and from the PyInstaller sidecar bundle — or
# arrives from the control plane per session (James, 2026-08-08: "keep the
# knowledge book server-side"). This module is only the loader + the stable
# API the rest of the brain was already written against.

from . import knowledge as _knowledge


def topics_map() -> dict[str, tuple[str, str]]:
    """{name: (summary, body)} from whichever source is available (may be {})."""
    return _knowledge.get_topics()


# `knowledge_deep.TOPICS` — the topic names, for the tests and the evals. Asked for, not
# snapshotted: `TOPICS = tuple(sorted(topics_map()))` at module level ran the loader at IMPORT
# (reader E6·R9; see knowledge.py), which in the shipped brain is a keychain read and a request
# to the control plane before the handshake. Anything that must be fresh in production — the
# lookup tool's topic enum, the index line — is built PER CALL from topics_map() (see
# chat.tool_defs), and so, now, is this.
def __getattr__(name: str):
    if name == "TOPICS":
        return tuple(sorted(topics_map()))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def topic_index() -> str:
    """One line per topic for the tool description: 'name — summary'."""
    return "; ".join(f"{name}: {summary}"
                     for name, (summary, _body) in sorted(topics_map().items()))


def lookup(topic: str) -> str:
    """The body for one topic, or an honest miss naming the real ones."""
    topics = topics_map()
    entry = topics.get(topic)
    if entry is None:
        names = ", ".join(sorted(topics)) or "(none available this session)"
        return f"no such topic: {topic!r}. Available: {names}"
    return entry[1]
