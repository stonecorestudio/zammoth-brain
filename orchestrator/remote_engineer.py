"""The engineer's own tools for a chat app on the user's subscription (Phase 69, 2026-11-04).

James, 2026-11-04: "we need to let people use zammoth's full power through their subs". The
remote MCP door (Phase 68, `daw/Source/McpBridge.h`) first offered three generic tools: a read of
the project's settings, the command list and one command. A chat app's model then had no ears -
no loudness, no masking, no phase - no guardrailed mix moves, no Auto Mix, no measured sounds and
none of the engineer's know-how. None of that is the engineer's MODEL: it is the engine's work on
this Mac, and the model is the part the chat app brings.

So the door now runs the engineer's OWN tools: a tool call from the chat app is one tool call of
a `turn_host.LiveTurn` with no model in it (`run_one`). Every tool is the function the in-app
engineer calls - the same measurements, the same proposals under the same guardrails, the same
`daw_command` choke point (the app adds the door's own rules there), the level measure a move
owes, the sound profile a render brings back. One implementation: nothing here re-does a tool.

What is left out, and why (`LEFT_OUT`): `start_run` runs Zammoth's own model on the user's key -
the chat app is the model; `ask_user` / `set_plan` / `ask_preference` are cards in the app's feed
that the remote model would wait on for ever; `list_files` lists the user's folders. A path a
read tool takes (`analyze_sample`, `analyze_reference`) must be inside the project or Zammoth's
own folders (`path_refusal`) - the door never reads the rest of the disk.

The engineer's know-how is `engineer_guide`: the DAW prompt the in-app engineer works from, in
sections, plus `lookup_knowledge` (the knowledge book, licence-gated where it is loaded).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

#: The engineer's tools the door offers, by the names the in-app engineer knows them by.
REMOTE_TOOLS: tuple[str, ...] = (
    "analyze_session", "analyze_stem", "analyze_phrases", "analyze_bus", "analyze_return",
    "describe_region", "analyze_masking", "analyze_phase", "analyze_sections", "analyze_notes",
    "analyze_chords", "analyze_timing", "analyze_tempo", "analyze_activity",
    "analyze_separation", "analyze_reference", "analyze_sample", "analyze_reference_set",
    "get_proposals", "propose", "undo_proposal", "auto_mix", "render", "daw_command",
    "read_track", "save_note", "lookup_knowledge",
)
#: ...and the ones it does not, with the reason a test reads.
LEFT_OUT: dict[str, str] = {
    "start_run": "runs Zammoth's own model on the user's key; the chat app is the model",
    "ask_user": "a card in the app's feed the remote model would wait on",
    "set_plan": "a card in the app's feed",
    "ask_preference": "an A/B card in the app's feed",
    "list_files": "lists the user's folders; the door does not reach the disk",
}
#: The tools that carry a file path, and the argument it is in.
PATH_ARGS: dict[str, str] = {"analyze_sample": "path", "analyze_reference": "path"}

#: The longest part of the guide one call returns.
GUIDE_PART_CHARS = 12000

#: The session id the door's turns use: its own conversation record, never the user's.
REMOTE_SESSION_ID = "remote-mcp"


def remote_tool_defs() -> list[dict]:
    """The engineer's tool definitions the door offers, as MCP tools (name, description,
    inputSchema), with `engineer_guide` first. A description that sends the model to its
    context (the in-app engineer has the catalog there) points at list_commands instead."""
    from .chat import daw_tool_defs
    by_name = {d["name"]: d for d in daw_tool_defs(False)}
    out: list[dict] = [{
        "name": "engineer_guide",
        "title": "Read the engineer's guide",
        "description": ("How Zammoth's own audio engineer works: what to measure before and after a move, "
                        "the guardrails, how each device and command is meant to be used, how to talk "
                        "to the user. With no section: the list of sections. Read the opening and the "
                        "section on the task at hand before you start; it is what makes you work like "
                        "Zammoth's engineer."),
        "inputSchema": {"type": "object", "properties": {
            "section": {"type": "string", "description": "A section title from the list (or its number)."}},
            "additionalProperties": False},
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False},
    }]
    for name in REMOTE_TOOLS:
        d = by_name.get(name)
        if d is None:
            continue
        desc = str(d.get("description") or "")
        desc = desc.replace("daw_commands in your context", "list_commands").replace(
            "in your context", "in get_project")
        read_only = name.startswith("analyze_") or name in (
            "describe_region", "get_proposals", "read_track", "lookup_knowledge")
        out.append({
            "name": name,
            "description": desc,
            "inputSchema": d.get("input_schema") or {"type": "object", "properties": {}},
            "annotations": {"readOnlyHint": read_only, "destructiveHint": not read_only,
                            "openWorldHint": False},
        })
    return out


def guide_sections(prompt: Optional[str] = None) -> list[tuple[str, str]]:
    """The DAW prompt cut at its headings: [(title, text)], the opening first."""
    if prompt is None:
        from .chat import daw_prompt
        prompt = daw_prompt()
    parts: list[tuple[str, str]] = []
    title, buf = "Opening", []
    lines = prompt.splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        # a heading is a short line of its own that ends with a colon after a blank line
        # ("Ground rules:", "Craft:", "How you write to the user:") - or a markdown heading
        md = re.match(r"^#{1,3}\s+(.+?)\s*$", s)
        colon = (re.match(r"^[A-Z][A-Za-z0-9 ,'/&()-]{2,40}:$", s) is not None
                 and (i == 0 or not lines[i - 1].strip()))
        if md or colon:
            text = "\n".join(buf).strip()
            if text:
                parts.append((title, text))
            title = (md.group(1) if md else s).rstrip(":").strip()
            buf = []
            continue
        buf.append(line)
    text = "\n".join(buf).strip()
    if text:
        parts.append((title, text))
    # A section longer than a small chat window reads comfortably is cut at paragraph breaks
    # into numbered parts ("Craft (2 of 3)").
    out: list[tuple[str, str]] = []
    for title, text in parts:
        if len(text) <= GUIDE_PART_CHARS:
            out.append((title, text))
            continue
        # balanced parts, cut at a line that starts a new item (a "- " bullet or a blank line)
        n = -(-len(text) // GUIDE_PART_CHARS)
        target = len(text) / n
        chunks, cur = [], []
        size = 0
        lines = text.split("\n")
        for j, ln in enumerate(lines):
            starts_item = not ln.strip() or ln.lstrip().startswith("- ")
            if cur and len(chunks) < n - 1 and ((size >= target and starts_item) or size >= GUIDE_PART_CHARS):
                chunks.append("\n".join(cur).strip())
                cur, size = [], 0
            cur.append(ln)
            size += len(ln) + 1
        if cur:
            chunks.append("\n".join(cur).strip())
        for k, c in enumerate(chunks):
            out.append((f"{title} ({k + 1} of {len(chunks)})", c))
    return out


def engineer_guide(section: str = "") -> str:
    """The guide: the list of sections, or one section's text."""
    parts = guide_sections()
    if not section.strip():
        lines = [f"{i + 1}. {t} ({len(x):,} characters)" for i, (t, x) in enumerate(parts)]
        return ("The engineer's guide, in sections. Ask for one by title or number.\n"
                + "\n".join(lines))
    want = section.strip()
    if want.isdigit() and 1 <= int(want) <= len(parts):
        t, x = parts[int(want) - 1]
        return f"## {t}\n\n{x}"
    low = want.lower()
    for t, x in parts:
        if t.lower() == low:
            return f"## {t}\n\n{x}"
    for t, x in parts:
        if low in t.lower():
            return f"## {t}\n\n{x}"
    raise KeyError(f"no section called {section!r}: call engineer_guide with no section for the list")


def allowed_roots(project_dir: Path) -> list[Path]:
    """Where a read tool's path may point: the project, Zammoth's sounds and packs and presets,
    and the app's render cache."""
    home = Path.home()
    roots = [project_dir, home / "Music" / "Zammoth", home / "Library" / "Caches" / "Zammoth"]
    import os
    for var in ("ZAMMOTH_SOUNDS_DIR", "ZAMMOTH_PACKS_DIR", "ZAMMOTH_PRESETS_DIR"):
        if (v := os.environ.get(var)):
            roots.append(Path(v))
    return roots


def path_refusal(project_dir: Path, name: str, args: dict) -> Optional[str]:
    """Why a read tool's path is refused over the door, or None."""
    arg = PATH_ARGS.get(name)
    if arg is None or not isinstance(args, dict) or args.get(arg) in (None, ""):
        return None
    raw = Path(str(args[arg])).expanduser()
    p = raw if raw.is_absolute() else project_dir / raw
    try:
        real = p.resolve()
    except OSError:
        return f"{arg}: cannot read {args[arg]!r}"
    for root in allowed_roots(project_dir):
        try:
            real.relative_to(root.resolve())
            return None
        except (ValueError, OSError):
            continue
    return (f"{arg}: remote control reads files only inside the project and Zammoth's own folders "
            "(sample packs, presets, rendered sounds) - not elsewhere on the user's Mac")


def fill_export_file(project_dir: Path, name: str, args: Any) -> Any:
    """An export over the door names a NEW file in the project's bounces/ folder when the chat app
    gave none. With no file the app asks the user's yes in a box, and a chat app does not wait for
    a person: Claude.ai gave up on the box, retried, and the export was written twice (2026-11-04,
    the real-song run). A new file inside the project is the case the engineer's own rule writes
    without asking (`exportNeedsNoYes`), so this changes where nothing was chosen, never a choice."""
    if name != "daw_command" or not isinstance(args, dict):
        return args
    if str(args.get("command") or "") != "session.export":
        return args
    cargs = args.get("args") if isinstance(args.get("args"), dict) else {}
    if str(cargs.get("file") or "").strip() or cargs.get("stems"):
        return args
    fmt = str(cargs.get("format") or "wav").strip().lower() or "wav"
    import time
    stamp = time.strftime("%Y%m%d-%H%M%S")
    folder = project_dir / "bounces"
    target = folder / f"remote-{stamp}.{fmt}"
    n = 2
    while target.exists():
        target = folder / f"remote-{stamp}-{n}.{fmt}"
        n += 1
    return {**args, "args": {**cargs, "file": str(target)}}


def refusal(project_dir: Path, name: str, args: Any) -> Optional[str]:
    """Why the door turns this tool call away before it runs, or None."""
    if name in LEFT_OUT:
        return f"{name} is not available remotely: {LEFT_OUT[name]}"
    if name != "engineer_guide" and name not in REMOTE_TOOLS:
        return f"unknown tool: {name}"
    if not isinstance(args, dict):
        return "the arguments must be an object"
    return path_refusal(project_dir, name, args)
