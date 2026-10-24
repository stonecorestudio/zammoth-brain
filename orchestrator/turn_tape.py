"""A chat turn kept as a TAPE: what the model asked for, round by round (Phase 56).

`MIX_AGENT_TAPE=1` (a source checkout only - a shipped brain ignores it) makes every chat turn
leave `<project>/runs/tape/<when>-<turn>.json`: the prompt, each model round as the fake API's
own step (`server/tests/fake_anthropic.py`: text, thinking and tool_use blocks with their
whole input, how the round stopped, its usage), and how each tool call came back.

Nothing else keeps a call's ARGUMENTS: the saved transcript holds display rows (arguments
cut at 120 characters, runs folded), the session digest holds command keys, the timing
record keys and seconds. So a real session - a billed one above all - could not be played
again through the chat rig with the model taken out. With this on, it can
(`chat_rig.tape_turns`): the brain's and the app's own seconds, for nothing, as often as
wanted.

A tape holds what the model SAID, which on a real project is the user's own prompt and the
names in their session: it is written inside the project's folder, never anywhere else.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

TAPE_ENV = "MIX_AGENT_TAPE"
TAPE_DIR = "tape"


def wanted() -> bool:
    from .llm_backend import is_frozen
    return os.environ.get(TAPE_ENV, "").strip() not in ("", "0") and not is_frozen()


def _plain(value: Any) -> Any:
    """A block's input as JSON can hold it."""
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return str(value)


class Tape:
    """One turn. Every method is cheap, and none can raise into the turn."""

    def __init__(self, prompt: str, images: Any = None) -> None:
        self.prompt = prompt
        # the pictures the user attached, by PATH (Phase 64): a tape is a developer's record,
        # and the data is the file's
        self.images = [str(getattr(p, "path", p)) for p in images or []]
        self.rounds: list[dict] = []
        self.calls: list[dict] = []

    def round(self, final: Any) -> None:
        """A finished API round, from the SDK's final message."""
        try:
            blocks: list[list] = []
            for block in getattr(final, "content", None) or []:
                kind = getattr(block, "type", "")
                if kind == "text":
                    blocks.append(["text", getattr(block, "text", "")])
                elif kind == "thinking":
                    blocks.append(["thinking", ""])       # the words are not ours to keep or replay
                elif kind == "tool_use":
                    blocks.append(["tool_use", getattr(block, "id", ""), getattr(block, "name", ""),
                                   _plain(getattr(block, "input", {}))])
            usage = getattr(final, "usage", None)
            step: dict = {"blocks": blocks, "stop": getattr(final, "stop_reason", None)}
            if usage is not None:
                step["usage"] = {k: int(getattr(usage, k, 0) or 0) for k in
                                 ("input_tokens", "cache_read_input_tokens",
                                  "cache_creation_input_tokens")}
                step["out"] = int(getattr(usage, "output_tokens", 0) or 0)
            self.rounds.append(step)
        except Exception as exc:  # noqa: BLE001 - a recorder never fails a turn
            print(f"[turn_tape] round not kept ({type(exc).__name__}: {exc})", file=sys.stderr)

    def call(self, call_id: str, name: str, args: Any, ok: bool) -> None:
        self.calls.append({"id": str(call_id), "name": name, "args": _plain(args), "ok": bool(ok)})

    def steps(self, calls_per_round: Optional[list[int]] = None) -> list[dict]:
        """The turn as fake-API steps. The API path recorded them; the subscription path
        recorded only its calls, so its rounds are rebuilt from how many calls each response
        made (`TurnTimer.rounds[i]["calls"]`) - the grouping is what decides how often the
        brain measures."""
        if self.rounds:
            return self.rounds
        groups: list[list[dict]] = []
        rest = list(self.calls)
        for n in calls_per_round or []:
            if n > 0 and rest:
                groups.append(rest[:n])
                rest = rest[n:]
        if rest:
            groups.append(rest)
        steps = [{"blocks": [["thinking", ""]] + [["tool_use", c["id"], c["name"], c["args"]]
                                                  for c in group]} for group in groups]
        return steps + [{"blocks": [["thinking", ""], ["text", "(the recorded reply)"]]}]

    def as_dict(self, calls_per_round: Optional[list[int]] = None) -> dict:
        out = {"prompt": self.prompt, "steps": self.steps(calls_per_round),
               "results": {c["id"]: "ok" if c["ok"] else "error" for c in self.calls}}
        if self.images:
            out["images"] = list(self.images)
        return out


def write(project_dir: Path, session: Any) -> Optional[Path]:
    """Keep `session.tape` (this turn's) under the project's runs/tape/. Best effort."""
    tape = getattr(session, "tape", None)
    if not isinstance(tape, Tape):
        return None
    try:
        timing = getattr(session, "timing", None)
        per_round = [int(r.get("calls") or 0) for r in getattr(timing, "rounds", None) or []]
        folder = Path(project_dir) / "runs" / TAPE_DIR
        folder.mkdir(parents=True, exist_ok=True)
        # by the clock: a turn id restarts with every brain process
        path = folder / f"{int(time.time() * 1000)}-{getattr(session, 'turn_id', 't')}.json"
        path.write_text(json.dumps(tape.as_dict(per_round), indent=1))
        return path
    except Exception as exc:  # noqa: BLE001
        print(f"[turn_tape] not written ({type(exc).__name__}: {exc})", file=sys.stderr)
        return None


def read(project_dir: Path) -> list[dict]:
    """A project's tapes, oldest first."""
    folder = Path(project_dir) / "runs" / TAPE_DIR
    out = []
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        try:
            out.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            pass
    return out
