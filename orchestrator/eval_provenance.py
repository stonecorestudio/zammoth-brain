"""What a billed eval baseline was measured UNDER — and per-case memory (2026-09-02).

The eval audit found three ways a baseline said less than it appeared to:

* **No provenance.** `routing_baseline.json`, `qa_baseline.json` and
  `operate_baseline.json` recorded a score but not the model, backend, effort
  level or prompt it was scored under. The DAW's effort default is `high`, the
  evals pass none and inherit chat's default, so a default change silently
  changed what every number meant — and the 13 prompt commits after the last
  qa/routing run were invisible until someone diffed git by hand.
* **Counts only.** The routing and qa gates compare CLASS counts (mis-scoped,
  bluffed …), by design: one case flipping on a non-deterministic model is
  noise, a class growing is a regression. But counts hide a SWAP — one case
  fixed and another newly broken is the same count, green — and with no
  per-case record nobody could even name the 14 mis-scoped routing cases.
* **Silent staleness.** `--check` never said "this baseline predates the prompt
  you are testing", so a green check on a stale baseline read as proof.

This module is the shared fix: `provenance()` stamps a baseline with the
conditions it was recorded under, `provenance_notes()` turns the stamp into
warnings at `--check` time (informational — a changed prompt is the REASON to
run the gate, not a failure of it), and `case_flips()` names every case whose
verdict moved (informational too, for the same noise reason the count gate
exists; the operate gate has `--repeat` and gates per case, the single-run
gates report).
"""
from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Optional


def prompt_fingerprint() -> str:
    """A short hash of everything the resident engineer is prompted with: the
    shared knowledge block, the DAW system prompt, the tool definitions and the
    command catalog. Derived from the live objects, never hand-copied, so it
    moves whenever any of them does.

    The catalog (reader E7·R5, 2026-10-06): every command's key, its argument
    grammar and its user-only / needs-confirm bits - about 20k tokens the model
    reads, built from `DawCommands.h`. It rides the CONTEXT, not the system
    prompt, so a grammar edit used to leave this stamp unchanged and no gate
    said "the prompt changed since the baseline". It is the registry as the
    operate eval's fixture publishes it, whichever eval is being stamped: that
    is the surface the app ships."""
    from . import knowledge as _knowledge
    from .chat import daw_prompt, daw_tool_defs
    from .operate_eval import OPERATE_EDIT

    material = "\n".join([
        _knowledge.get_block(),
        daw_prompt(),
        json.dumps(daw_tool_defs(False), sort_keys=True),
        json.dumps(OPERATE_EDIT.get("commands") or [], sort_keys=True),
    ])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]


def provenance(model: Optional[str] = None,
               reasoning: Optional[str] = None) -> dict:
    """The conditions a baseline is recorded under. `reasoning` is the effort
    the eval ran at: its `--effort` when given (2026-09-03 — cheap iteration
    runs), else chat's default, which is what the session runs at when the eval
    passes nothing. Either way the stamp says which."""
    from .chat import DEFAULT_REASONING, MODEL
    from .llm_backend import active_backend

    from .llm_models import effective_effort
    stamp = {"model": model or MODEL, "backend": active_backend(),
             "reasoning": reasoning or DEFAULT_REASONING,
             "prompt": prompt_fingerprint(),
             "recorded": date.today().isoformat()}
    # the level the model was actually SENT (2026-09-22 audit D·F6): a level the model lacks
    # is clamped, and a model with no ladder takes none — the requested level alone misstates it
    stamp["effort_sent"] = effective_effort(stamp["model"], stamp["reasoning"])
    if stamp["backend"] == "subscription":
        # WHICH Claude Code ran it (2026-09-10 audit P3, 01·F14): the SDK spawns its bundled
        # CLI, not the system `claude`, and nothing recorded which one a baseline measured.
        from .llm_subscription import cli_version
        stamp["cli"] = cli_version()
    return stamp


def provenance_notes(baseline: dict, model: Optional[str] = None,
                     reasoning: Optional[str] = None) -> list[str]:
    """Human-readable drift between a baseline's recorded conditions and the
    conditions of the run about to be compared against it. Empty when the
    baseline carries no stamp (pre-2026-09-02) — that fact is itself reported."""
    meta = baseline.get("meta")
    if not isinstance(meta, dict):
        return ["baseline carries no provenance (recorded before 2026-09-02): "
                "model / backend / effort / prompt it measured are unknown — "
                "regenerate it to stamp them"]
    now = provenance(model, reasoning)
    notes = []
    for field, label in (("model", "model"), ("backend", "backend"),
                         ("reasoning", "effort")):
        if meta.get(field) != now[field]:
            notes.append(f"{label} differs: baseline {meta.get(field)!r}, "
                         f"this run {now[field]!r}")
    if meta.get("cli") and now.get("cli") and meta["cli"] != now["cli"]:
        notes.append(f"Claude Code CLI differs: baseline {meta['cli']!r}, "
                     f"this run {now['cli']!r}")
    if meta.get("prompt") != now["prompt"]:
        notes.append(f"the prompt changed since the baseline was recorded "
                     f"({meta.get('prompt')} -> {now['prompt']}, recorded "
                     f"{meta.get('recorded', '?')}): the baseline describes a "
                     f"prompt that no longer ships")
    return notes


#: Below this share of prompt tokens read from cache, a sweep is paying the
#: full prompt on every case. The evals' prompt is ~27k tokens of which ~25k
#: is identical across cases, so a healthy sweep reads most of it back.
CACHE_SHARE_WARN = 0.5

USAGE_FIELDS = ("input_tokens", "cache_read_tokens", "cache_write_tokens",
                "output_tokens")


def usage_totals(outcomes) -> dict:
    """Sum the four token counters over eval outcomes (any object with the
    `USAGE_FIELDS` attributes), plus `prompt_tokens` (everything sent) and
    `cache_share` (the fraction of it read from cache)."""
    t = {k: sum(int(getattr(o, k, 0) or 0) for o in outcomes) for k in USAGE_FIELDS}
    prompt = t["input_tokens"] + t["cache_read_tokens"] + t["cache_write_tokens"]
    t["prompt_tokens"] = prompt
    t["cache_share"] = round(t["cache_read_tokens"] / prompt, 3) if prompt else 0.0
    t["cases"] = len(outcomes)
    return t


def usage_line(t: dict) -> str:
    """One human line for a report: the totals, the per-case prompt, and a
    plain warning when the cache is not landing. Empty when nothing was billed
    (a no-LLM or all-error run).

    On the subscription the line measures Claude Code's OWN request caching (the CLI
    rebuilds every request), not the brain's cache layout — the one that bills users on
    the api transport never runs there (2026-09-22 audit D·F6), so it is labelled."""
    if not t.get("prompt_tokens") and not t.get("output_tokens"):
        return ""
    try:
        from .llm_backend import active_backend
        transport = active_backend()
    except ValueError:
        transport = "?"
    per_case = round(t["prompt_tokens"] / t["cases"]) if t.get("cases") else 0
    line = (f"tokens [{transport}]: prompt {t['prompt_tokens']:,} "
            f"(uncached {t['input_tokens']:,}, cache read {t['cache_read_tokens']:,}, "
            f"cache write {t['cache_write_tokens']:,}), output {t['output_tokens']:,}"
            f"; ~{per_case:,} prompt tokens per case; "
            f"cache share {t['cache_share']:.0%}")
    if t["prompt_tokens"] and t["cache_share"] < CACHE_SHARE_WARN:
        line += (f"\n  CACHE NOT LANDING: only {t['cache_share']:.0%} of prompt tokens "
                 f"were read from cache (expect well over {CACHE_SHARE_WARN:.0%}) — "
                 f"every case is paying the full prompt")
    if transport == "subscription":
        line += ("\n  (subscription: this is Claude Code's own caching — the api transport's "
                 "cache layout, what users are billed on, is only measured on `api`)")
    return line


def case_flips(baseline_cases: dict, current: dict) -> list[str]:
    """Every case whose verdict differs from the baseline's, as
    'verdict_then -> verdict_now  text'. Both maps are text -> verdict. A case
    missing on either side is reported as such rather than guessed."""
    lines = []
    for text, then in sorted(baseline_cases.items()):
        now = current.get(text)
        if now is None:
            lines.append(f"{then} -> (gone)      {text!r}")
        elif now != then:
            lines.append(f"{then} -> {now}  {text!r}")
    for text in sorted(set(current) - set(baseline_cases)):
        lines.append(f"(new) -> {current[text]}  {text!r}")
    return lines
