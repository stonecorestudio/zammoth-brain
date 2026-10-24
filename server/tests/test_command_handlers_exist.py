"""Every agent-invocable command key has a handler in the DAW's dispatch.

`dawcmd::specs()` (daw/Source/DawCommands.h) is what the agent is told it can do;
`DawComponent::invokeCommand` (daw/Source/App.cpp) is what actually runs, and its
fallthrough answers "unknown command". A key present in the table with no `case`
label is the exact "ships agent-blind" failure the total-agent-control invariant
exists to prevent — and nothing checked it: `--commands` validates the table's
shape, `--agent-guard` its bits, the operate-eval coverage gate the CORPUS against
the table (2026-09-01 sweep, P2). This is a free text parse of both files.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
COMMANDS = REPO / "daw" / "Source" / "DawCommands.h"
APP = REPO / "daw" / "Source" / "App.cpp"

# ids deliberately dispatched somewhere other than a `case dawcmd::<id>:` label
EXEMPT: set[str] = set()


def _spec_ids() -> list[tuple[str, str, bool]]:
    text = COMMANDS.read_text(encoding="utf-8")
    table = text[text.index("static const std::vector<Spec> table"):]
    table = table[: table.index("};")]
    out = []
    for line in table.splitlines():
        m = re.match(r'\s*\{\s*(\w+),\s*"([a-z_0-9]+\.[a-z_0-9]+)"\s*,\s*"[^"]*",\s*"[^"]+",\s*(.*?)\s*\},?',
                     line)
        if m:
            flags = re.findall(r"\b(?:true|false)\b", m.group(3))
            out.append((m.group(1), m.group(2), len(flags) > 1 and flags[1] == "true"))
    assert len(out) >= 267, f"parsed only {len(out)} specs — the table moved?"
    return out


def test_every_agent_invocable_spec_has_a_case_in_invokeCommand():
    cases = set(re.findall(r"case dawcmd::(\w+)\s*:", APP.read_text(encoding="utf-8")))
    assert len(cases) >= 267, f"only {len(cases)} case labels found — the dispatch moved?"
    missing = [f"{ident} ({key})" for ident, key, user_only in _spec_ids()
               if not user_only and ident not in cases and ident not in EXEMPT]
    assert missing == [], (
        f"{len(missing)} agent-invocable command(s) advertised in DawCommands.h have no "
        f"`case dawcmd::<id>:` in App.cpp — the agent would get 'unknown command': {missing}")
