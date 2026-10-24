"""Every performer the agent can reach opens its undo transaction through the wrapper.

`dawsession::beginTransaction` is a no-op while an agent turn is open, so a whole turn
is ONE ⌘Z (`AgentTurnGate`, DawSession.h). A performer that calls
`getUndoManager().beginNewTransaction` directly splits the turn into N undos — and
twelve of ArrangeView's `agent*` clip performers did exactly that until 2026-09-01
(sweep T8). The user-gesture paths (mouse drags, key handlers) keep calling the
UndoManager directly by design; this check covers the paths a command can reach:
ArrangeView's `agent*` functions, its shared clip helpers, and App.cpp itself, where
the only transaction opener is the gate.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARRANGE = REPO / "daw" / "Source" / "ArrangeView.h"
APP = REPO / "daw" / "Source" / "App.cpp"
DIRECT = "getUndoManager().beginNewTransaction ("


def _functions(text: str):
    """(name, body) for every 4-space-indented member function, by brace matching."""
    for m in re.finditer(r"^    [A-Za-z_][\w:<>&* ]* ([A-Za-z_]\w*) \([^;{]*\)[^;{]*\{", text, re.M):
        depth, i = 0, m.end() - 1
        while i < len(text):
            depth += text[i] == "{"
            depth -= text[i] == "}"
            if depth == 0:
                break
            i += 1
        yield m.group(1), text[m.start(): i + 1]


def test_arrangeview_agent_performers_use_the_wrapper():
    text = ARRANGE.read_text(encoding="utf-8")
    offenders = [name for name, body in _functions(text)
                 if name.startswith("agent") and DIRECT in body]
    assert offenders == [], (
        f"agent-reachable performers open transactions directly (split an agent turn): {offenders}")


def test_the_shared_clip_helpers_use_the_wrapper():
    text = ARRANGE.read_text(encoding="utf-8")
    shared = {"removeClip", "moveClipToLane", "addWarpMarkerAt"}   # called from both gestures and agent paths
    offenders = [name for name, body in _functions(text) if name in shared and DIRECT in body]
    assert offenders == [], offenders


def test_no_product_view_opens_a_transaction_directly():
    """The transitive rule (full audit 2026-10-02, reader C1.R4 = C7.R2). The two tests above
    look inside `agent*` functions and three named helpers; the agent's clip.split,
    clip.duplicate, comp.set_segment and the time.* edits reached a direct opener ONE CALL
    DOWN (`splitClipAt`, `duplicateClip`, `compSetSegment`, `deleteTimeRange`...), which no
    scan of the callers can see. So the rule is on the files: nothing in the arrange view, the
    piano roll, the step grid or App.cpp calls the UndoManager's opener — a user's gesture runs
    with the turn flag off, so the wrapper is the same call for it."""
    opener = re.compile(r"\bbeginNewTransaction\s*\(")
    for rel in ("daw/Source/ArrangeView.h", "daw/Source/PianoRollView.h",
                "daw/Source/StepGridView.h", "daw/Source/App.cpp"):
        lines = (REPO / rel).read_text(encoding="utf-8").splitlines()
        direct = [f"{rel}:{n}: {ln.strip()[:90]}" for n, ln in enumerate(lines, 1)
                  if opener.search(ln.split("//")[0])]
        assert direct == [], "open it with dawsession::beginTransaction:\n" + "\n".join(direct)


def test_app_cpp_opens_transactions_only_through_the_gate():
    text = APP.read_text(encoding="utf-8")
    direct = [ln for ln in text.splitlines() if DIRECT in ln and "//" not in ln.split(DIRECT)[0]]
    assert direct == [], f"App.cpp must route every transaction through AgentTurnGate / the wrapper: {direct}"
    assert "agentTurnGate.begin (" in text and "agentTurnGate.seal (" in text or "sealAgentTurn()" in text
