"""The Anthropic models the engineer can run on — ONE table for the whole brain (2026-09-22).

James, 2026-10-08: "remove all models that are not in the 5.5 family ... keep opus 5.5,
sonnet 5.5 and the new haiku 5.5 ... no fable yet". The picker is the 5.5 family and nothing
else: Opus 5.5 (the default), Sonnet 5.5 and Haiku 5.5 (released 2026-10-07). From 2026-09-22
it had offered every current-generation model, ten of them; before that the DAW was
Opus-5-only. Every model-dependent decision in the brain reads this table:

- which models the picker offers, and in what words (`label`);
- which effort levels each accepts (`efforts`) and how a request asks it to think
  (`thinking_kwargs`) — all three take the whole ladder, low to max;
- what its tokens cost (`cost_usd`) — the run budget and every $ the DAW shows. Haiku 5.5 is
  priced by the length of the prompt (`long_prompt_over`): one request, one price;
- whether its thinking is bound to the conversation (`preserved_thinking` — all three): a
  replayed thinking block whose earlier conversation was edited is a 400 for Anthropic
  accounts created on or after 2026-08-31, so the chat keeps those histories strictly
  append-only (chat.py);
- whether the notes it writes between tool calls arrive as `thinking` blocks
  (`progress_thinking`, Opus 5.5 and Sonnet 5.5) rather than text.

The DAW carries a copy of the picker half (native/Source/AgentModels.h), pinned to this
table by server/tests/test_agent_model_parity.py — the model list once lived in six places
and drifted (memory `model-beast-names`); there are two now, and a test between them.

Source: Anthropic's models overview, pricing page and the three migration guides, read
2026-10-08. Check a key against
the live Models API (free — `models.retrieve` bills nothing) with
``python -m orchestrator.llm_models --verify``.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Optional

#: The API's effort ladder, low to high. `xhigh` sits between high and max.
EFFORT_LEVELS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")
_ALL = EFFORT_LEVELS

#: What the DAW ships at. `medium` since 2026-10-12 (James, Phase 54): it is Anthropic's own
#: default for Opus 5.5, whose levels are not Opus 5's - the model thinks more per step at a
#: given level - and `high` was a setting carried over from Opus 5, where the gates' baselines
#: were measured. Lower effort is fewer tool calls, less preamble and terser replies, which is
#: what a 38-minute seven-turn session needed. No gate has been run on a 5.5 model at either
#: level; the picker still offers the whole ladder.
DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "medium"


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str                      # what the UI shows — the Anthropic name (James, 2026-09-22)
    family: str                     # Opus / Sonnet / Haiku — the picker's grouping
    input_usd: float                # $ per million input tokens
    output_usd: float               # $ per million output tokens
    cache_read_x: float = 0.1       # cache reads bill at this multiple of input
    context_tokens: int = 1_000_000
    max_output_tokens: int = 128_000
    efforts: tuple[str, ...] = _ALL  # () = no effort and no adaptive thinking (no such row today)
    preserved_thinking: bool = False
    progress_thinking: bool = False
    # A request whose PROMPT (fresh input + cache reads + cache writes) is over this many
    # tokens is billed at `long_prompt_x` times every rate — input, output and cache alike.
    # 0 = one price at every length (every model but Haiku 5.5).
    long_prompt_over: int = 0
    long_prompt_x: float = 1.0
    # Fast mode (Anthropic's research preview: `speed: "fast"`, up to 2.5x the output tokens a
    # second from the same model). Opus 5.5 only on the picker, the Claude API only, and only
    # for accounts Anthropic has approved; a fast response bills `fast_x` times every rate.
    fast: bool = False
    fast_x: float = 2.0


# Order = the picker's order: most capable first.
MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("claude-opus-5-5", "Opus 5.5", "Opus", 4.0, 20.0, cache_read_x=0.05,
              preserved_thinking=True, progress_thinking=True, fast=True),
    ModelSpec("claude-sonnet-5-5", "Sonnet 5.5", "Sonnet", 2.0, 10.0, cache_read_x=0.05,
              preserved_thinking=True, progress_thinking=True),
    # $0.10 / $0.50 for a prompt up to 100,000 tokens, $0.50 / $2.50 over it (cache reads
    # $0.01 / $0.05, 5-minute writes $0.125 / $0.625, 1-hour writes $0.20 / $1): five times
    # everything. Its thinking runs the same history-editing check as the other two; its notes
    # between tool calls are not on the `display: "updates"` list.
    ModelSpec("claude-haiku-5-5", "Haiku 5.5", "Haiku", 0.10, 0.50,
              preserved_thinking=True, long_prompt_over=100_000, long_prompt_x=5.0),
)
_BY_ID = {m.id: m for m in MODELS}

# Anything priced that is NOT on the picker: ids an eval, an old baseline or an environment
# override may still name — the seven the picker held until 2026-10-08 among them. Pricing
# only (id, input, output, cache-read multiple) — never offered, never validated as a chat model.
_PRICE_ONLY: tuple[tuple[str, float, float, float], ...] = (
    ("claude-fable-5-1", 10.0, 50.0, 0.025),
    ("claude-fable-5", 10.0, 50.0, 0.1),
    ("claude-mythos-5-1", 10.0, 50.0, 0.025),
    ("claude-mythos-5", 10.0, 50.0, 0.1),
    ("claude-opus-5", 5.0, 25.0, 0.1),
    ("claude-opus-4-8", 5.0, 25.0, 0.1),
    ("claude-opus-4-7", 5.0, 25.0, 0.1),
    ("claude-opus-4-6", 5.0, 25.0, 0.1),
    ("claude-opus-4-5", 5.0, 25.0, 0.1),
    ("claude-opus-4-1", 15.0, 75.0, 0.1),
    ("claude-opus-4-0", 15.0, 75.0, 0.1),
    ("claude-sonnet-5", 2.0, 10.0, 0.1),
    ("claude-sonnet-4-6", 3.0, 15.0, 0.1),
    ("claude-sonnet-4-5", 3.0, 15.0, 0.1),
    ("claude-sonnet-4-0", 3.0, 15.0, 0.1),
    ("claude-haiku-4-5", 1.0, 5.0, 0.1),
)
#: An id nobody priced is billed at the most expensive rate Anthropic currently serves
#: (Fable's — a retired id does not set it), so a run budget can only stop EARLY on it, never
#: late (the 2026-08 table priced Fable at Opus rates: half its real cost). It is no longer the
#: picker's own maximum: that is $4 / $20 now, under what an unknown id may well cost.
_UNKNOWN_RATES = (10.0, 50.0, 0.1)


def _is_snapshot_of(model: str, alias: str) -> bool:
    """`model` is `alias` followed by an 8-digit date and nothing else:
    `claude-opus-5-5` is a different model from `claude-opus-5`, not a snapshot of it."""
    suffix = model[len(alias) + 1:]
    return model.startswith(alias + "-") and len(suffix) == 8 and suffix.isdigit()


def spec(model: str) -> Optional[ModelSpec]:
    """The picker entry for `model`, or None. Exact id only: a dated snapshot of a known
    alias (`claude-opus-5-5-20260922`) resolves to its alias."""
    model = (model or "").strip()
    if model in _BY_ID:
        return _BY_ID[model]
    for m in MODELS:
        if _is_snapshot_of(model, m.id):
            return m
    return None


def is_known(model: str) -> bool:
    return spec(model) is not None


def label(model: str) -> str:
    s = spec(model)
    return s.label if s else model


def efforts_for(model: str) -> tuple[str, ...]:
    s = spec(model)
    return s.efforts if s else ()


def effective_effort(model: str, level: str) -> Optional[str]:
    """The effort `model` will actually be sent for a requested `level`: the level itself
    when the model has it, else the highest level it has BELOW it, else its lowest. None when
    the model takes no effort at all — or is unknown, where sending a field the model might
    reject is the wrong gamble. (Every row has the whole ladder since 2026-10-08, so the clamp
    is idle; it is kept for the next row that does not.)"""
    have = efforts_for(model)
    if not have:
        return None
    if level in have:
        return level
    rank = EFFORT_LEVELS.index(level) if level in EFFORT_LEVELS else EFFORT_LEVELS.index(DEFAULT_EFFORT)
    below = [e for e in have if EFFORT_LEVELS.index(e) <= rank]
    return below[-1] if below else have[0]


def at_most(level: str, cap: str) -> str:
    """The lower of two ladder levels — for a call that never needs more than `cap`."""
    if level not in EFFORT_LEVELS:
        return cap
    return level if EFFORT_LEVELS.index(level) <= EFFORT_LEVELS.index(cap) else cap


def thinking_kwargs(model: str, level: str) -> dict:
    """The request kwargs that make `model` think at `level`.

    Adaptive thinking + `output_config.effort` on every model that has an effort ladder. The
    effort is sent explicitly even where it is the default: on Opus 5.5 and Haiku 5.5 an
    omitted effort runs at `medium`, not the level the user picked. An unknown id gets nothing:
    a plain request is valid on every model."""
    effort = effective_effort(model, level)
    if effort is None:
        return {}
    return {"thinking": {"type": "adaptive"}, "output_config": {"effort": effort}}


#: The ceiling of one streamed agentic round on a model that thinks. Anthropic's own figure for
#: long agentic turns on Opus 5.5 (its migration guide and cost guide: "64K has worked well").
AGENTIC_MAX_TOKENS = 64_000


def max_tokens_for(model: str, level: str, base: int = 16_000) -> int:
    """`max_tokens` for one agentic round — which caps thinking AND the reply together.

    64,000 for a model with an effort ladder, at every level (Phase 55). On the 5.5 family
    thinking is always on and counts toward the ceiling, and a round cut off mid-command ends
    the turn "out of room" with its earlier moves applied. The ceiling is not billed; only
    what is generated is. It was held at `base` up to `high` because "a large max_tokens can
    count against a new account's per-minute output limit before a single token is produced":
    Anthropic's rate-limit page says the opposite (read 2026-10-12: "The max_tokens parameter
    does not factor into OTPM rate limit calculations, so there is no rate limit downside to
    setting a higher max_tokens value"). Every caller streams, which a ceiling this size needs.

    An id off the table (no effort, no thinking config sent) keeps `base`: it may be a model
    whose own maximum is lower."""
    effort = effective_effort(model, level)
    want = max(base, AGENTIC_MAX_TOKENS) if effort else base
    s = spec(model)
    return min(want, s.max_output_tokens) if s else want


#: When a conversation is trimmed, in tokens of the whole prompt (static prefix + history) as
#: the API counted it: past `trigger` at a turn's start, the oldest whole turns go until it is
#: under `floor`. Sized for cost, not for the window (1M): at 200,000 tokens a round reads
#: about $0.04 of cache on Opus 5.5, and a trim drops the earlier reasoning and rewrites the
#: cached conversation once - so it should be rare, and a 60-MESSAGE cap made it every few
#: turns (one eleven-round turn is 22 messages).
HISTORY_TRIGGER_TOKENS = 200_000
HISTORY_FLOOR_TOKENS = 120_000


def history_budget(model: str) -> tuple[int, int]:
    """(trigger, floor) for `model`'s conversation, in prompt tokens.

    A model priced by the length of the prompt (Haiku 5.5: five times every rate past
    100,000 tokens) is held well under that line - 0.7x to trim, 0.5x to trim to - because a
    turn cannot trim mid-way and its own tool results grow the prompt as it runs. Nothing
    kept it under before: a long turn crossed the line and every request after it paid
    five times."""
    s = spec(model)
    if s is not None and s.long_prompt_over:
        return int(0.7 * s.long_prompt_over), int(0.5 * s.long_prompt_over)
    if s is None:
        return 120_000, 80_000      # an id off the table: assume a 200k window
    return HISTORY_TRIGGER_TOKENS, HISTORY_FLOOR_TOKENS


#: The beta a fast request names. Requests go out as Anthropic documents them:
#: `client.beta.messages.stream(..., speed="fast", betas=[BETA_FAST_MODE])` (the SDK's
#: non-beta `messages.stream` has no `speed`).
BETA_FAST_MODE = "fast-mode-2026-02-01"
#: How often the SDK re-sends a fast request by itself (it waits out a short `retry-after`,
#: which keeps the prompt cache) before the caller falls back to standard speed. The api
#: transport's usual count is four: too long to sit on a preview's own rate limit.
FAST_MAX_RETRIES = 1
#: Why fast mode is off for the life of this process ("" = it is not): the account does not
#: have the research preview. Asked once; a refused request is not billed.
_FAST_UNAVAILABLE = {"why": ""}


def has_fast(model: str) -> bool:
    s = spec(model)
    return bool(s and s.fast)


def fast_unavailable() -> str:
    return _FAST_UNAVAILABLE["why"]


def note_fast_unavailable(why: str) -> None:
    _FAST_UNAVAILABLE["why"] = why or "this account does not have fast mode"


def fast_kwargs(model: str, want: bool, backend: str) -> dict:
    """What a request adds to run in fast mode, or {}: only when the user switched it on, the
    model has it, the transport is the direct API (not the subscription's CLI, not the
    dormant proxy), and this account has not already been told no."""
    if not want or backend != "api" or not has_fast(model) or fast_unavailable():
        return {}
    return {"speed": "fast", "betas": [BETA_FAST_MODE]}


def fast_refusal(exc: BaseException) -> str:
    """How a fast request was turned away: "limit" (its own rate limit - carry on at standard
    speed, the documented fallback), "unavailable" (a 4xx that names the feature: the account
    is not in the preview), or "" (some other error: not about fast mode)."""
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    text = ""
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        text = str(body["error"].get("message") or "")
    text = (text or str(exc)).lower()
    from .llm_backend import spend_limit   # deferred: llm_backend is the lighter import
    if spend_limit(exc):
        return ""        # the ACCOUNT's spend limit, not fast mode's: standard speed hits it too
    if status == 429:
        return "limit"
    if status in (400, 403, 404) and ("fast" in text or "speed" in text):
        return "unavailable"
    return ""


def validate(model: str, level: str) -> None:
    """The composer's pair: a picker model and a level on the API's ladder. A level the
    model lacks is NOT an error — it is clamped at request time (`effective_effort`)."""
    if not is_known(model):
        raise ValueError(f"unknown chat model: {model}")
    if level not in EFFORT_LEVELS:
        raise ValueError(f"unknown reasoning level: {level}")


def _rates(model: str) -> tuple[float, float, float]:
    s = spec(model)
    if s is not None:
        return s.input_usd, s.output_usd, s.cache_read_x
    model = (model or "").strip()
    for mid, i, o, read_x in _PRICE_ONLY:
        if model == mid or _is_snapshot_of(model, mid):
            return i, o, read_x
    return _UNKNOWN_RATES


def cost_usd(model: str, input_tokens: int, output_tokens: int,
             cache_read_tokens: int = 0, cache_write_tokens: int = 0,
             cache_write_1h_tokens: int = 0, fast: bool = False) -> float:
    """Estimated spend for ONE response at `model`'s list price. `cache_write_1h_tokens` is
    the part of `cache_write_tokens` written at the 1-hour TTL (2x input; 5-minute writes
    are 1.25x). An estimate — it gates the run budget and labels the DAW's $, it is not a
    billing record.

    One response, never a sum of several: a model priced by prompt length (Haiku 5.5) is
    judged on THIS request's prompt, and ten 30k-token rounds added together would read as
    one 300k-token prompt and be priced five times over. `fast` is what the RESPONSE said
    (`usage.speed == "fast"`), never what was asked for: a fast request that fell back to
    standard speed bills the standard price."""
    in_rate, out_rate, read_x = _rates(model)
    s = spec(model)
    prompt_tokens = input_tokens + cache_read_tokens + cache_write_tokens
    tier = s.long_prompt_x if s and s.long_prompt_over and prompt_tokens > s.long_prompt_over else 1.0
    if fast and s is not None and s.fast:
        tier *= s.fast_x
    return tier * (input_tokens * in_rate
                   + cache_read_tokens * in_rate * read_x
                   + max(0, cache_write_tokens - cache_write_1h_tokens) * in_rate * 1.25
                   + cache_write_1h_tokens * in_rate * 2.0
                   + output_tokens * out_rate) / 1_000_000


def picker_rows() -> list[dict]:
    """The picker as data: what a shell needs to draw it."""
    return [{"id": m.id, "label": m.label, "family": m.family, "efforts": list(m.efforts),
             "effort": bool(m.efforts), "fast": m.fast} for m in MODELS]


# ---- --verify: the table against the live Models API (free) --------------------------

def _as_dict(obj) -> dict:
    """The SDK hands `capabilities` back as a pydantic model, not a dict (the first version of
    `_verify` subscripted it, and would have died on its first model)."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    dump = getattr(obj, "model_dump", None)
    return dump() if callable(dump) else {}


def _verify(client) -> list[str]:
    """Differences between this table and what the API says about each model."""
    problems: list[str] = []
    for m in MODELS:
        try:
            live = client.models.retrieve(m.id)
        except Exception as exc:  # noqa: BLE001 — every failure is a finding here
            problems.append(f"{m.id}: not retrievable ({type(exc).__name__}: {exc})")
            continue
        caps = _as_dict(getattr(live, "capabilities", None))
        if getattr(live, "max_input_tokens", None) not in (None, m.context_tokens):
            problems.append(f"{m.id}: context {live.max_input_tokens} != {m.context_tokens}")
        if getattr(live, "max_tokens", None) not in (None, m.max_output_tokens):
            problems.append(f"{m.id}: max output {live.max_tokens} != {m.max_output_tokens}")
        try:
            effort = caps["effort"]
            live_levels = tuple(e for e in EFFORT_LEVELS
                                if effort.get("supported")
                                and (effort.get(e) or {}).get("supported"))   # xhigh may be null
            if live_levels != m.efforts:
                problems.append(f"{m.id}: effort levels {live_levels} != {m.efforts}")
            adaptive = bool(caps["thinking"]["types"]["adaptive"]["supported"])
            if adaptive != bool(m.efforts):
                problems.append(f"{m.id}: adaptive thinking {adaptive}, table says {bool(m.efforts)}")
            # The brief compile, the critic and the reflection ask for output_config.format
            # (agents._parse); a model without it would 400 all three (2026-09-22 audit B·F10).
            if not caps["structured_outputs"]["supported"]:
                problems.append(f"{m.id}: no structured outputs (the brief compile, critic and "
                                "reflection use them)")
        except (KeyError, TypeError, AttributeError):
            problems.append(f"{m.id}: capabilities unreadable: "
                            f"{json.dumps(caps, default=str)[:200]}")
    return problems


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--verify", action="store_true",
                        help="check the table against the live Models API with the key the brain "
                             "would use (env ANTHROPIC_API_KEY, else the keychain) — free")
    args = parser.parse_args(argv)
    if not args.verify:
        print(json.dumps(picker_rows(), indent=2))
        return 0
    import anthropic

    from .api_key import api_key
    key = api_key()
    if not key:
        print("no Anthropic API key (env ANTHROPIC_API_KEY or the keychain)", file=sys.stderr)
        return 2
    problems = _verify(anthropic.Anthropic(api_key=key, timeout=20.0, max_retries=1))
    for p in problems:
        print("MISMATCH", p)
    print("OK — the table matches the Models API" if not problems
          else f"{len(problems)} mismatch(es)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
