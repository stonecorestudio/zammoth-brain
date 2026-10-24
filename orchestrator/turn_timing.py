"""Where a chat turn's seconds went (Phase 54, 2026-10-12).

Until this module nothing logged a round's timing: the table that showed a 23-minute arrange
turn was 40 rounds of one clip command each had to be rebuilt by hand from the MCP library's
one-second timestamps in ``brain.log``. A turn is now recorded as it runs, on both transports:

- each model ROUND: how long the request waited for its first byte, how long it streamed,
  its tokens (fresh input, cache reads and writes, output, thinking where the API says) and
  how it stopped;
- each TOOL call: its name (a ``daw_command`` by its key) and its seconds;
- what the brain did around the model: the context it built before the first request
  (``prep``), the mix it measured once a response's calls were in (``measure``) and what it
  settled after the reply (``tail``);
- how many master and stem renders the turn made (``dsp.RENDER_COUNTS``).

Record v3 (Phase 56) keeps what v2 summed away, so one file answers "what did the user wait
for": each round as a row, every tool's count / total / longest (and each call's seconds),
when the first thing reached the feed (``first_activity_s``), how many seconds the model's
time and the tools' time cover TWICE (``overlap_s``: the subscription transport runs a call
while its response still streams), what the app itself spent on each command against what the
brain waited (``commands``), the wait for a person's click on an approval box
(``confirm_wait_s``), the model that served the turn and the account's rate limits.

``TurnTimer`` is the recorder (``ChatSession.timing``). ``record`` appends one JSON line a
turn to ``<project>/runs/turn_timing.jsonl`` and ``line`` is what ``brain.log`` gets.

    python -m orchestrator.turn_timing <project dir>      # the per-turn table, newest session
    python -m orchestrator.turn_timing <project dir> --all
    python -m orchestrator.turn_timing <project dir> --detail          # rounds, tools, commands
    python -m orchestrator.turn_timing <project dir> --compare <other project dir>
    python -m orchestrator.turn_timing <project dir> --write-budget b.json [--slack 1.25]
    python -m orchestrator.turn_timing <project dir> --check b.json    # exit 1 over a budget

prints the table, with each turn's reply length read from the saved feed (``chats/``).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Callable, Optional

TURN_TIMING_FILE = "turn_timing.jsonl"
TURN_TIMING_KEEP = 400


class TurnTimer:
    """One chat turn's timeline. Every method is cheap and none can raise into the turn."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self.t0 = clock()
        self._idle_since = self.t0        # when the model was last handed the turn
        self._round_start: Optional[float] = None
        self._first_byte: Optional[float] = None
        self._pending_out = 0
        self.rounds: list[dict] = []
        self.tools: list[dict] = []
        self.marks: dict[str, float] = {}
        self.mark_counts: dict[str, int] = {}
        # The response now streaming, or the last one that ended: how many tool calls it
        # announced and how many of them have come back. A caller reads `last_call_back` to
        # know a response's calls are all in (the measure a turn owes rides the last one).
        self._announced = 0
        self._handled = 0
        self._open_calls = 0              # calls that came back while their response streamed
        self.saw_stream = False
        # Record v3. Kept beside `rounds` / `tools`, never inside them: those two lists are
        # what the turn itself reads, and their entries are pinned shape for shape.
        self._round_spans: list[tuple[float, float]] = []
        self._tool_spans: list[tuple[float, float]] = []
        self.first_activity_at: Optional[float] = None
        self.served_model: Optional[str] = None
        self.rate_limits: dict[str, int] = {}

    # ---- model rounds ------------------------------------------------------------------
    def begin_round(self) -> None:
        """A request is about to go out (the API loop). The subscription path has no such
        moment of its own: there a round opens at its first stream event (`stream_started`)
        and is dated from when the model was last handed the turn."""
        self._round_start = self._clock()
        self._first_byte = None
        self._announced = self._handled = self._open_calls = 0

    def stream_started(self) -> None:
        """The first event of a response arrived."""
        self.saw_stream = True
        if self._round_start is None:
            self._round_start = self._idle_since
            self._announced = self._handled = self._open_calls = 0
        if self._first_byte is None:
            self._first_byte = self._clock()

    def announce_call(self) -> None:
        """The response now streaming holds one more tool call (its `tool_use` block began).
        The subscription path: there the SDK runs a call while the rest of the response is
        still arriving, so a call can come back before its round has an entry."""
        self._announced += 1

    @property
    def round_open(self) -> bool:
        return self._round_start is not None

    @property
    def calls_behind(self) -> bool:
        """The response announced a call that has not come back yet."""
        return self._handled < self._announced

    @property
    def last_call_back(self) -> bool:
        """The response has ended and every call it announced has come back."""
        return not self.round_open and self._handled >= self._announced

    def end_round(self, output_tokens: int = 0, usage: object = None,
                  stop_reason: Optional[str] = None, failed: str = "") -> None:
        """Close the round. `usage` is the response's own (an SDK `Usage` or the raw dict) and
        adds its token counts; `stop_reason` how it ended. Keys are added only when given, so
        a host that reports neither records what it always did."""
        now = self._clock()
        start = self._round_start if self._round_start is not None else self._idle_since
        first = self._first_byte if self._first_byte is not None else now
        entry: dict = {"wait_s": round(max(0.0, first - start), 2),
                       "stream_s": round(max(0.0, now - first), 2),
                       "out": int(output_tokens or 0) + self._pending_out,
                       "calls": self._open_calls}
        entry.update(usage_fields(usage))
        if stop_reason:
            entry["stop"] = str(stop_reason)
        if failed:
            entry["failed"] = failed
        self.rounds.append(entry)
        self._round_spans.append((start, now))
        self._round_start = self._first_byte = None
        self._pending_out = 0
        self._open_calls = 0
        self._idle_since = now

    def fail_round(self, why: str) -> None:
        """A request that died: its seconds are the turn's, and the next one starts fresh."""
        self.end_round(failed=(why or "failed")[:80])

    def add_output(self, output_tokens: int) -> None:
        """A response's tokens, whenever they become known: the subscription path reports
        them on a message that can arrive before or after the response's last stream event."""
        if not output_tokens:
            return
        if self.round_open or not self.rounds:
            self._pending_out += int(output_tokens)
        else:
            self.rounds[-1]["out"] = int(self.rounds[-1].get("out", 0)) + int(output_tokens)

    # ---- tools and the brain's own work ------------------------------------------------
    def tool(self, name: str, seconds: float, key: str = "") -> None:
        self.tools.append({"name": name, "key": key, "s": round(max(0.0, seconds), 2)})
        now = self._clock()
        self._tool_spans.append((now - max(0.0, seconds), now))
        self._handled += 1
        if self.round_open and self._announced:
            self._open_calls += 1     # its own response is still streaming: filed when it ends
        elif self.rounds:             # the call belongs to the response that asked for it
            self.rounds[-1]["calls"] += 1
        self._idle_since = self._clock()

    def handed_back(self) -> None:
        """The brain has finished what it does after a tool call (a measure): the model's
        wait for its next response starts now, not when the tool itself returned."""
        self._idle_since = self._clock()

    def mark(self, phase: str, seconds: float) -> None:
        self.marks[phase] = round(self.marks.get(phase, 0.0) + max(0.0, seconds), 2)
        self.mark_counts[phase] = self.mark_counts.get(phase, 0) + 1

    def activity(self) -> None:
        """Something of this turn reached the feed (a word, a tool row). The first is kept:
        it is when the user stops looking at a line that only says the engineer is working."""
        if self.first_activity_at is None:
            self.first_activity_at = self._clock()

    def note_served(self, model: object = None, headers: object = None) -> None:
        """What answered a request: the model the RESPONSE names (an eval's score is of that
        model, whatever was asked for) and the account's limits as the API's own headers give
        them. A new organisation can start under the published limits, and nothing else in
        the product says what a customer's account allows."""
        if isinstance(model, str) and model:
            self.served_model = model
        get = getattr(headers, "get", None)
        if get is None:
            return
        for name, key in RATE_LIMIT_HEADERS.items():
            try:
                value = int(get(name))
            except (TypeError, ValueError):
                continue
            # a limit is the account's; what is left is kept at its lowest
            self.rate_limits[key] = (min(value, self.rate_limits.get(key, value))
                                     if key.endswith("_left") else value)

    def overlap_s(self) -> float:
        """Seconds that are in BOTH the rounds' time and the tools' time."""
        tools = _merged(self._tool_spans)
        return round(sum(max(0.0, min(r1, t1) - max(r0, t0))
                         for r0, r1 in self._round_spans for t0, t1 in tools), 2)

    # ---- the record --------------------------------------------------------------------
    def summary(self) -> dict:
        if self.round_open:           # a response whose end was never signalled
            self.end_round()
        wall = self._clock() - self.t0
        by_tool: dict[str, list[float]] = {}
        for t in self.tools:
            label = t["key"] or t["name"]
            by_tool.setdefault(label, []).append(t["s"])
        slow = sorted(by_tool.items(), key=lambda kv: -sum(kv[1]))[:8]
        waits = sorted(r["wait_s"] for r in self.rounds)
        out = {
            "wall_s": round(wall, 1),
            "rounds": len(self.rounds),
            "model_s": round(sum(r["wait_s"] + r["stream_s"] for r in self.rounds), 1),
            "first_byte_median_s": waits[len(waits) // 2] if waits else 0.0,
            "out_tokens": sum(r["out"] for r in self.rounds),
            "tool_calls": len(self.tools),
            "tools_s": round(sum(t["s"] for t in self.tools), 1),
            "rounds_with_several_calls": sum(1 for r in self.rounds if r["calls"] > 1),
            "slowest_tools": [{"tool": k, "n": len(v), "s": round(sum(v), 1)} for k, v in slow],
            "marks_s": dict(self.marks),
            "mark_counts": dict(self.mark_counts),
            # v3: nothing above is changed by what follows
            "round_rows": [{"at_s": round(max(0.0, span[0] - self.t0), 2), **r}
                           for span, r in zip(self._round_spans, self.rounds)],
            "tools": {k: {"n": len(v), "total_s": round(sum(v), 2), "max_s": round(max(v), 2)}
                      for k, v in by_tool.items()},
            "tool_rows": [[t["key"] or t["name"], t["s"]] for t in self.tools],
            "overlap_s": self.overlap_s(),
        }
        if self.served_model:
            out["served_model"] = self.served_model
        if self.rate_limits:
            out["rate_limits"] = dict(self.rate_limits)
        out.update(self._tokens())
        return out

    def _tokens(self) -> dict:
        """What the rounds that reported usage read, wrote and generated - absent when none
        did, so a record never shows a zero nobody measured."""
        metered = [r for r in self.rounds if "in" in r]
        out: dict = {}
        if metered:
            fresh = sum(r["in"] for r in metered)
            read = sum(r.get("read", 0) for r in metered)
            write = sum(r.get("write", 0) for r in metered)
            prompt = fresh + read + write
            out.update(in_tokens=fresh, cache_read_tokens=read, cache_write_tokens=write,
                       cache_write_1h_tokens=sum(r.get("write_1h", 0) for r in metered),
                       cache_share=round(read / prompt, 3) if prompt else 0.0)
            # A round after the turn's first should read what the round before it wrote. One
            # that reads under half its prompt paid for the conversation again (a pause past
            # the cache's lifetime, a trim, or an edit of the prefix).
            cold = [i for i, r in enumerate(metered[1:], 1)
                    if (p := r["in"] + r.get("read", 0) + r.get("write", 0)) > 0
                    and r.get("read", 0) < 0.5 * p]
            if cold:
                out["cache_cold_rounds"] = len(cold)
        thinking = [r["think"] for r in self.rounds if "think" in r]
        if thinking:
            out["thinking_tokens"] = sum(thinking)
        stops: dict[str, int] = {}
        for r in self.rounds:
            if r.get("stop"):
                stops[r["stop"]] = stops.get(r["stop"], 0) + 1
        if stops:
            out["stops"] = stops
        failed = sum(1 for r in self.rounds if r.get("failed"))
        if failed:
            out["failed_rounds"] = failed
        return out


#: The API's response headers a record keeps, under the names the record uses.
RATE_LIMIT_HEADERS = {
    "anthropic-ratelimit-requests-limit": "rpm",
    "anthropic-ratelimit-input-tokens-limit": "itpm",
    "anthropic-ratelimit-output-tokens-limit": "otpm",
    "anthropic-ratelimit-input-tokens-remaining": "itpm_left",
    "anthropic-ratelimit-output-tokens-remaining": "otpm_left",
}


def _merged(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Spans with their overlaps joined, so a second counted twice is counted once."""
    out: list[tuple[float, float]] = []
    for start, end in sorted(spans):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def merge_commands(timing: dict, commands: list[dict]) -> None:
    """Add the WIRE's view of a turn to its record: what the brain asked of the app, how long
    it waited, how long the app says each command took, and how much of the wait was a person
    deciding at an approval box. `commands` is one dict a `daw_invoke`:
    {key, wait_s, app_s?, confirm_s?}. A turn that asked nothing adds nothing."""
    if not commands:
        return
    by_key: dict[str, dict] = {}
    for c in commands:
        row = by_key.setdefault(str(c.get("key") or ""), {"n": 0, "wait_s": 0.0})
        row["n"] += 1
        row["wait_s"] = round(row["wait_s"] + float(c.get("wait_s") or 0.0), 3)
        for field in ("app_s", "confirm_s"):
            if isinstance(c.get(field), (int, float)):
                row[field] = round(row.get(field, 0.0) + float(c[field]), 3)
    timing["commands"] = by_key
    confirm = sum(r.get("confirm_s", 0.0) for r in by_key.values())
    if confirm:
        timing["confirm_wait_s"] = round(confirm, 1)


def usage_fields(raw: object) -> dict:
    """One response's usage as a round keeps it: `in` (fresh input), `read` / `write` (the
    cache), `write_1h` (the part written for an hour), `think` (thinking tokens, where the API
    reports them). {} for a response that reported nothing. Accepts the SDK's `Usage` or the
    raw dict a stream event carries."""
    if raw is None:
        return {}

    def field(obj: object, key: str):
        return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)

    def number(value) -> Optional[int]:
        return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    fresh = number(field(raw, "input_tokens"))
    if fresh is None:
        return {}
    out = {"in": fresh, "read": number(field(raw, "cache_read_input_tokens")) or 0,
           "write": number(field(raw, "cache_creation_input_tokens")) or 0}
    hour = number(field(raw, "cache_creation_1h_input_tokens"))
    if hour is None and (breakdown := field(raw, "cache_creation")) is not None:
        hour = number(field(breakdown, "ephemeral_1h_input_tokens"))
    if hour:
        out["write_1h"] = hour
    details = field(raw, "output_tokens_details")
    think = number(field(details, "thinking_tokens")) if details is not None else None
    if think is not None:
        out["think"] = think
    return out


def line(turn_id: str, s: dict) -> str:
    """The one line `brain.log` gets for a finished turn."""
    marks = s.get("marks_s") or {}
    counts = s.get("mark_counts") or {}
    extra = "".join(f", {k} {v:.1f} s x{counts.get(k, 1)}" for k, v in sorted(marks.items()))
    renders = s.get("renders") or {}
    rendered = (f"; renders master {renders.get('master', 0)} stem {renders.get('stem', 0)}"
                if renders else "")
    words = f"; reply {s['reply_words']} words" if "reply_words" in s else ""
    tokens = f"; out {s.get('out_tokens', 0)} tok"
    if "in_tokens" in s:
        prompt = s["in_tokens"] + s.get("cache_read_tokens", 0) + s.get("cache_write_tokens", 0)
        tokens += (f", prompt {prompt} tok ({100 * s.get('cache_share', 0.0):.0f} % from cache"
                   + (f", {s['cache_cold_rounds']} COLD round(s)" if s.get("cache_cold_rounds") else "")
                   + ")")
    if "thinking_tokens" in s:
        tokens += f", thinking {s['thinking_tokens']} tok"
    price = f"; ${s['cost_usd']:.2f}" if isinstance(s.get("cost_usd"), (int, float)) else ""
    ended = " FAILED" if s.get("failed") else " done"
    return (f"[chat] {turn_id}{ended} {s.get('wall_s', 0):.1f} s: {s.get('rounds', 0)} rounds, "
            f"model {s.get('model_s', 0):.1f} s (first byte median "
            f"{s.get('first_byte_median_s', 0):.1f} s), {s.get('tool_calls', 0)} tool calls "
            f"{s.get('tools_s', 0):.1f} s{extra}{rendered}{tokens}{price}{words}")


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def record(project_dir: Path, entry: dict) -> None:
    """Append one turn to `runs/turn_timing.jsonl` (bounded). Best effort: timing is a
    readout and never the reason a turn fails."""
    from .daw_bridge import _now_iso, write_atomic

    try:
        runs = Path(project_dir) / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        path = runs / TURN_TIMING_FILE
        lines = path.read_text().splitlines() if path.is_file() else []
        lines.append(json.dumps({"ts": _now_iso(), **entry}))
        write_atomic(path, "\n".join(lines[-TURN_TIMING_KEEP:]) + "\n")
    except Exception as exc:  # noqa: BLE001
        print(f"[turn_timing] not recorded ({type(exc).__name__}: {exc})", file=sys.stderr)


# ---- the table ---------------------------------------------------------------------------

def _entries(project_dir: Path) -> list[dict]:
    path = Path(project_dir) / "runs" / TURN_TIMING_FILE
    out = []
    for raw in path.read_text().splitlines() if path.is_file() else []:
        try:
            out.append(json.loads(raw))
        except json.JSONDecodeError:
            pass                      # a half-written last line
    return out


def feed_replies(project_dir: Path, session_id: str = "") -> list[dict]:
    """Each turn of a saved conversation as the feed holds it: the user's words, the tool
    rows, and the words of the final prose block (what the user reads as the reply)."""
    chats = Path(project_dir) / "chats"
    try:
        sid = session_id or json.loads((chats / "index.json").read_text())["active"]
        rows = json.loads((chats / f"{sid}.json").read_text())
    except Exception:  # noqa: BLE001 - no saved conversation
        return []
    turns: list[dict] = []
    for row in rows:
        kind = row.get("k")
        if kind == "user":
            turns.append({"user": row.get("t") or "", "tools": 0, "prose": []})
        elif turns and kind == "tool":
            turns[-1]["tools"] += 1
        elif turns and kind == "prose":
            turns[-1]["prose"].append(row.get("t") or "")
    for t in turns:
        t["reply_words"] = word_count(t["prose"][-1]) if t["prose"] else 0
        t["note_words"] = sum(word_count(p) for p in t["prose"][:-1])
    return turns


def table(project_dir: Path, every_session: bool = False) -> str:
    entries = _entries(project_dir)
    if entries and not every_session:
        last = entries[-1].get("session_id", "")
        entries = [e for e in entries if e.get("session_id", "") == last]
    head = (f"{'turn':6} {'wall':>7} {'rounds':>6} {'model':>7} {'tools':>7} {'calls':>5} "
            f"{'prep':>5} {'measure':>8} {'tail':>5} {'renders':>8} {'out tok':>8} "
            f"{'prompt':>8} {'cache':>6} {'$':>6} {'words':>6}  request")
    out = [head, "-" * len(head)]
    total_wall = total_cost = brain_s = 0.0
    for e in entries:
        marks = e.get("marks_s") or {}
        counts = e.get("mark_counts") or {}
        renders = e.get("renders") or {}
        total_wall += float(e.get("wall_s") or 0.0)
        total_cost += float(e.get("cost_usd") or 0.0)
        # Since record v2 (Phase 55) a measure is taken after a response's calls, outside any
        # tool's seconds; before, it was inside the command's own.
        brain_s += brain_share_s(e)
        measure = (f"{marks.get('measure', 0.0):.0f}x{counts.get('measure', 0)}"
                   if marks.get("measure") else "-")
        if "in_tokens" in e:
            prompt = f"{e['in_tokens'] + e.get('cache_read_tokens', 0) + e.get('cache_write_tokens', 0):d}"
            cache = (f"{100 * e.get('cache_share', 0.0):.0f}%"
                     + ("!" if e.get("cache_cold_rounds") else ""))
        else:
            prompt = cache = "-"
        cost = f"{e['cost_usd']:.2f}" if isinstance(e.get("cost_usd"), (int, float)) else "-"
        out.append(f"{e.get('turn', ''):6} {e.get('wall_s', 0):7.1f} {e.get('rounds', 0):6d} "
                   f"{e.get('model_s', 0):7.1f} {e.get('tools_s', 0):7.1f} "
                   f"{e.get('tool_calls', 0):5d} {marks.get('prep', 0.0):5.1f} {measure:>8} "
                   f"{marks.get('tail', 0.0):5.1f} "
                   f"{str(renders.get('master', 0)) + '/' + str(renders.get('stem', 0)):>8} "
                   f"{e.get('out_tokens', 0):8d} {prompt:>8} {cache:>6} {cost:>6} "
                   f"{e.get('reply_words', 0):6d}  "
                   f"{'FAILED ' if e.get('failed') else ''}{(e.get('user') or '')[:48]}")
    if entries:
        out.append(f"{len(entries)} turn(s), {total_wall / 60.0:.1f} min of agent time; the "
                   f"brain's own share (tools, measuring, the wait before a first request) {brain_s:.0f} s"
                   + (f"; ${total_cost:.2f}" if total_cost else ""))
        out.append("cache = the share of the prompt read from cache (! = a round after the "
                   "turn's first read under half)")
    else:
        out.append("no recorded turns (runs/turn_timing.jsonl): this project's turns predate "
                   "the recorder")
        replies = feed_replies(project_dir)
        if replies:
            out.append("")
            out.append("reply lengths from the saved feed:")
            for i, t in enumerate(replies, 1):
                out.append(f"  turn {i}: {t['tools']:3d} tool rows, reply {t['reply_words']:4d} "
                           f"words  {t['user'][:60]}")
    return "\n".join(out)


def brain_share_s(e: dict) -> float:
    """One turn's seconds that are the brain's and the app's own: tool time, the wait before
    the first request, and (since record v2, where it is outside the tool seconds) the
    measuring. Never a person's wait at an approval box: that is nobody's work."""
    marks = e.get("marks_s") or {}
    return max(0.0, float(e.get("tools_s") or 0.0) + float(marks.get("prep") or 0.0)
               + (float(marks.get("measure") or 0.0) if int(e.get("v") or 1) >= 2 else 0.0)
               - float(e.get("confirm_wait_s") or 0.0))


def percentile(values: list[float], q: float) -> float:
    """The q-th percentile (0..1) by nearest rank: with five readings the 95th is the
    largest, which is the honest answer for five readings."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(-(-q * len(ordered) // 1)) - 1))]


def selected(project_dir: Path, every_session: bool = False) -> list[dict]:
    entries = _entries(project_dir)
    if entries and not every_session:
        last = entries[-1].get("session_id", "")
        entries = [e for e in entries if e.get("session_id", "") == last]
    return entries


def metrics(entries: list[dict]) -> dict:
    """A session as numbers a budget can hold. Every one is "lower is better", and each is
    absent when no record of the session carries what it is made of."""
    out: dict = {"turns": len(entries)}
    if not entries:
        return out
    out["agent_time_s"] = round(sum(float(e.get("wall_s") or 0.0) for e in entries), 1)
    out["brain_share_s"] = round(sum(brain_share_s(e) for e in entries), 1)
    out["model_s"] = round(sum(float(e.get("model_s") or 0.0) for e in entries), 1)
    out["prep_max_s"] = round(max(float((e.get("marks_s") or {}).get("prep") or 0.0)
                                  for e in entries), 2)
    out["measure_s"] = round(sum(float((e.get("marks_s") or {}).get("measure") or 0.0)
                                 for e in entries), 1)
    out["rounds"] = sum(int(e.get("rounds") or 0) for e in entries)
    out["tool_calls"] = sum(int(e.get("tool_calls") or 0) for e in entries)
    out["renders_master"] = sum(int((e.get("renders") or {}).get("master") or 0) for e in entries)
    out["renders_stem"] = sum(int((e.get("renders") or {}).get("stem") or 0) for e in entries)
    out["out_tokens"] = sum(int(e.get("out_tokens") or 0) for e in entries)
    firsts = [float(e["first_activity_s"]) for e in entries if "first_activity_s" in e]
    if firsts:
        out["first_activity_p50_s"] = percentile(firsts, 0.5)
        out["first_activity_max_s"] = max(firsts)
    costs = [float(e["cost_usd"]) for e in entries if isinstance(e.get("cost_usd"), (int, float))]
    if costs:
        out["cost_usd"] = round(sum(costs), 4)
    return out


def _calls(entries: list[dict]) -> dict[str, list[float]]:
    """Every tool call's seconds by tool, from the rows a v3 record keeps; a v2 record gives
    only its eight slowest tools' totals, shown as one reading each."""
    out: dict[str, list[float]] = {}
    for e in entries:
        rows = e.get("tool_rows")
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, list) and len(row) == 2:
                    out.setdefault(str(row[0]), []).append(float(row[1]))
        else:
            for t in e.get("slowest_tools") or []:
                out.setdefault(str(t.get("tool")) + " (v2: a turn's total)", []).append(
                    float(t.get("s") or 0.0))
    return out


def detail(project_dir: Path, every_session: bool = False) -> str:
    """Rounds, tools and commands of the selected turns: the tables the per-turn one sums."""
    entries = selected(project_dir, every_session)
    if not entries:
        return "no recorded turns (runs/turn_timing.jsonl)"
    out = [f"{'turn':6} {'wall':>7} {'first':>6} {'model':>7} {'tools':>7} {'both':>6} "
           f"{'click':>6} {'brain':>6} {'left':>6}  model / limits",
           "-" * 78]
    for e in entries:
        marks = e.get("marks_s") or {}
        wall, model, tools = (float(e.get(k) or 0.0) for k in ("wall_s", "model_s", "tools_s"))
        both = float(e.get("overlap_s") or 0.0)
        own = sum(float(v or 0.0) for v in marks.values())
        first = f"{e['first_activity_s']:.1f}" if "first_activity_s" in e else "-"
        limits = e.get("rate_limits") or {}
        said = " ".join(x for x in (
            str(e.get("served_model") or ""),
            f"itpm {limits['itpm']:,}" if "itpm" in limits else "",
            f"otpm {limits['otpm']:,}" if "otpm" in limits else "") if x)
        out.append(f"{e.get('turn', ''):6} {wall:7.1f} {first:>6} {model:7.1f} {tools:7.1f} "
                   f"{both:6.1f} {float(e.get('confirm_wait_s') or 0.0):6.1f} {own:6.1f} "
                   f"{wall - model - tools - own + both:6.1f}  {said}")
    out.append("first = seconds until anything reached the feed; both = seconds counted in "
               "model AND tools; click = a person at an approval box (inside tools); brain = "
               "prep + measure + tail; left = wall not accounted for")

    rounds = [r for e in entries for r in (e.get("round_rows") or [])]
    if rounds:
        waits = [float(r.get("wait_s") or 0.0) for r in rounds]
        streamed = sum(float(r.get("stream_s") or 0.0) for r in rounds)
        tokens = sum(int(r.get("out") or 0) for r in rounds)
        out += ["", f"{len(rounds)} round(s): first byte p50 {percentile(waits, 0.5):.2f} s, "
                    f"p95 {percentile(waits, 0.95):.2f} s, longest {max(waits):.2f} s; "
                    f"{tokens} output tokens in {streamed:.0f} s of streaming"
                    + (f" ({tokens / streamed:.0f} a second)" if streamed > 0 else "")]

    calls = _calls(entries)
    if calls:
        out += ["", f"{'tool':38} {'n':>4} {'total':>8} {'p50':>7} {'p95':>7} {'max':>7}",
                "-" * 76]
        for name, secs in sorted(calls.items(), key=lambda kv: -sum(kv[1])):
            out.append(f"{name[:38]:38} {len(secs):4d} {sum(secs):8.1f} "
                       f"{percentile(secs, 0.5):7.2f} {percentile(secs, 0.95):7.2f} {max(secs):7.2f}")

    asked: dict[str, dict] = {}
    for e in entries:
        for key, row in (e.get("commands") or {}).items():
            total = asked.setdefault(key, {"n": 0, "wait_s": 0.0, "app_s": 0.0, "confirm_s": 0.0,
                                           "timed": False})
            total["n"] += int(row.get("n") or 0)
            total["wait_s"] += float(row.get("wait_s") or 0.0)
            total["confirm_s"] += float(row.get("confirm_s") or 0.0)
            if "app_s" in row:
                total["app_s"] += float(row["app_s"])
                total["timed"] = True
    if asked:
        out += ["", f"{'command (the brain asking the app)':38} {'n':>4} {'waited':>8} "
                    f"{'in app':>8} {'click':>7} {'wire':>7}", "-" * 78]
        for key, row in sorted(asked.items(), key=lambda kv: -kv[1]["wait_s"]):
            app = f"{row['app_s']:8.2f}" if row["timed"] else f"{'-':>8}"
            # the app's own figure includes the click; the wire is what neither side worked
            wire = (f"{max(0.0, row['wait_s'] - row['app_s']):7.2f}" if row["timed"] else f"{'-':>7}")
            out.append(f"{key[:38]:38} {row['n']:4d} {row['wait_s']:8.2f} {app} "
                       f"{row['confirm_s']:7.1f} {wire}")
    return "\n".join(out)


def compare(project_dir: Path, other_dir: Path, every_session: bool = False) -> str:
    """Two sessions side by side, this project against another: the session's numbers, then
    each tool's total. A difference is a difference between two RUNS; one run of each says
    nothing about spread."""
    a, b = metrics(selected(project_dir, every_session)), metrics(selected(other_dir, every_session))
    out = [f"{'':24} {'this':>12} {'other':>12} {'change':>9}", "-" * 60]
    for key in sorted(set(a) | set(b)):
        x, y = a.get(key), b.get(key)
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            out.append(f"{key:24} {str(x if x is not None else '-'):>12} "
                       f"{str(y if y is not None else '-'):>12}")
            continue
        change = f"{100.0 * (x - y) / y:+8.0f}%" if y else f"{'':>9}"
        out.append(f"{key:24} {x:12.1f} {y:12.1f} {change}")
    ta = {k: sum(v) for k, v in _calls(selected(project_dir, every_session)).items()}
    tb = {k: sum(v) for k, v in _calls(selected(other_dir, every_session)).items()}
    rows = sorted(set(ta) | set(tb), key=lambda k: -abs(ta.get(k, 0.0) - tb.get(k, 0.0)))[:12]
    if rows:
        out += ["", f"{'tool (seconds; largest difference first)':38} {'this':>9} {'other':>9}",
                "-" * 60]
        out += [f"{k[:38]:38} {ta.get(k, 0.0):9.1f} {tb.get(k, 0.0):9.1f}" for k in rows]
    return "\n".join(out)


def write_budget(project_dir: Path, path: Path, slack: float, every_session: bool = False) -> dict:
    """A budget from a session as it measured: each number times `slack`. A count (rounds,
    calls, renders) is a count of WORK and gets no slack below one whole unit."""
    measured = metrics(selected(project_dir, every_session))
    budget = {}
    for key, value in measured.items():
        if key == "turns" or not isinstance(value, (int, float)):
            continue
        limit = value * slack
        budget[key] = int(-(-limit // 1)) if isinstance(value, int) else round(limit, 2)
    Path(path).write_text(json.dumps({"turns": measured.get("turns", 0), "slack": slack,
                                      "max": budget}, indent=2) + "\n")
    return budget


def check(project_dir: Path, path: Path, every_session: bool = False) -> tuple[bool, str]:
    """Hold a session to a budget file. Over any number is a failure, and so is a session of
    another length than the budget's (a shorter session is under every budget for nothing),
    and so is a number the budget names that this session could not produce."""
    budget = json.loads(Path(path).read_text())
    measured = metrics(selected(project_dir, every_session))
    out, ok = [], True
    if int(budget.get("turns") or 0) != int(measured.get("turns") or 0):
        ok = False
        out.append(f"OVER  turns: the budget is for {budget.get('turns')} turn(s), "
                   f"this session has {measured.get('turns')}")
    for key, limit in sorted((budget.get("max") or {}).items()):
        value = measured.get(key)
        if not isinstance(value, (int, float)):
            ok = False
            out.append(f"OVER  {key}: not measured in this session (budget {limit})")
            continue
        over = value > limit
        ok = ok and not over
        out.append(f"{'OVER ' if over else 'ok   '} {key}: {value} (budget {limit})")
    return ok, "\n".join(out)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("project", type=Path)
    parser.add_argument("--all", action="store_true", help="every conversation, not the newest")
    parser.add_argument("--detail", action="store_true",
                        help="rounds, every tool's spread, and the app's own time per command")
    parser.add_argument("--compare", type=Path, metavar="OTHER",
                        help="this project's session against another project's")
    parser.add_argument("--write-budget", type=Path, metavar="FILE",
                        help="write this session's numbers, times --slack, as a budget")
    parser.add_argument("--slack", type=float, default=1.25)
    parser.add_argument("--check", type=Path, metavar="FILE",
                        help="exit 1 when this session is over a budget file")
    args = parser.parse_args(argv)
    project = args.project.expanduser()
    if args.write_budget:
        budget = write_budget(project, args.write_budget, args.slack, args.all)
        print(f"budget written to {args.write_budget}: {json.dumps(budget)}")
        return 0
    if args.check:
        ok, said = check(project, args.check, args.all)
        print(said)
        return 0 if ok else 1
    if args.compare:
        print(compare(project, args.compare.expanduser(), args.all))
        return 0
    print(table(project, args.all))
    if args.detail:
        print()
        print(detail(project, args.all))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
