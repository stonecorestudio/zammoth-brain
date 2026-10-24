"""The knowledge book's LOADER — the book itself is not in this file (2026-08-08).

The always-on craft block and the deep topics are the product's distilled
know-how. The brain ships GPLv3 (it links Pedalboard), so its SOURCE is
published — but the knowledge text is not source code the binary needs to
build; it is content the running program is SERVED, like any SaaS response.
James's call: keep the book server-side. Concretely:

- `knowledge_content.py` (the whole book) is PRIVATE: excluded from the public
  brain-source cut and from the PyInstaller sidecar bundle
  (`daw/sidecar/mixagent_daw_brain.spec` excludes it).
- The dev repo, the tests, and the evals have the module present, so
  everything local works exactly as before, offline.
- The shipped sidecar fetches the book from the control plane
  (`GET /v1/knowledge`, with the signed-in device token; the endpoint gates on
  the licence or trial, Phase 53) on first use — lazily, because sign-in
  happens AFTER brain boot — on every transport, since the agent itself runs
  on the user's own key. A fetch
  failure degrades to the minimal block below rather than an error. Since Phase
  53 that premise changed: the LLM goes straight to Anthropic, so a control-plane
  outage no longer takes the agent down — it runs on the fallback until the book
  arrives. When it does, the system prompt changes under a live conversation:
  chat.py rebinds (a cache miss; on the preserved-thinking models - the whole
  picker - the thinking bound to the old prompt is dropped instead of 400ing).

Source order: MIX_AGENT_KNOWLEDGE_FILE (a JSON override for self-hosting and
tests) → the private module → the control plane → the fallback. Success is
cached for the process; failure is NOT, so the first post-sign-in turn heals.
"""
from __future__ import annotations

import json
import os
from typing import Optional

_FALLBACK_BLOCK = """\
# Audio engineering reference (fallback)

The house knowledge book could not be loaded this session. Work from general
mixing craft, and lean harder on the analyze tools: measure before every
claim, prefer carving maskers to boosting maskees, respect the true-peak
ceiling, and say plainly when a judgement would need the book. Do not invent
house numbers or genre targets — measure, or say you cannot.
"""

_cached: Optional[dict] = None   # {"block": str, "topics": {name: (summary, body)}}

# What the user is told, once per conversation and once per run, when the engineer works
# without the book (agent audit 2026-09-22, C·F8: the run without it was silent). The app's
# words, never the model's: a `chat_notice` in a chat turn, a run's first note in a run.
BOOK_MISSING_NOTICE = ("The engineer's knowledge book could not be fetched from Zammoth, so it "
                       "is working from general mixing craft and its own measurements for now. "
                       "It picks the book up by itself once the connection is back.")


def _from_env_file() -> Optional[dict]:
    path = os.environ.get("MIX_AGENT_KNOWLEDGE_FILE", "").strip()
    if not path or not os.path.exists(path):
        return None
    try:
        raw = json.loads(open(path, encoding="utf-8").read())
        return {"block": str(raw["block"]),
                "topics": {str(k): (str(v[0]), str(v[1]))
                           for k, v in dict(raw.get("topics", {})).items()}}
    except Exception:  # noqa: BLE001 — a broken override falls through, loudly enough
        return None


def _from_private_module() -> Optional[dict]:
    try:
        from . import knowledge_content
    except ImportError:
        return None
    return {"block": knowledge_content.KNOWLEDGE_BLOCK,
            "topics": dict(knowledge_content.DEEP_TOPICS)}


def _from_cloud() -> Optional[dict]:
    # Any signed-in brain, on ANY transport (Phase 53): the shipped DAW runs the agent on
    # the user's own key, and the book still lives on the control plane — behind the
    # LICENCE gate there, not the dormant subscription one. `cloud_token()` raises when
    # nobody is signed in (and when no cloud URL is configured), which is the "no book
    # from here" answer.
    try:
        import httpx

        from .llm_backend import CLOUD_URL_ENV, cloud_token
        base = os.environ.get(CLOUD_URL_ENV, "").strip().rstrip("/")
        if not base:
            return None
        response = httpx.get(f"{base}/v1/knowledge", timeout=6.0,
                             headers={"Authorization": f"Bearer {cloud_token()}"})
        if response.status_code != 200:
            return None
        raw = response.json()
        return {"block": str(raw["block"]),
                "topics": {str(k): (str(v[0]), str(v[1]))
                           for k, v in dict(raw.get("topics", {})).items()}}
    except Exception:  # noqa: BLE001 — no book is a degraded turn, never a crash
        return None


_retry_after = 0.0      # monotonic: no source is asked again before this
_backoff_s = 60.0       # doubles per failure, capped (below)


def _load() -> Optional[dict]:
    global _cached, _retry_after, _backoff_s
    if _cached is not None:
        return _cached
    # Negative cache (2026-09-01 sweep): get_block() runs inside EVERY chat round, and a
    # failure used to retry immediately — with the private module absent (the DAW brain
    # bundle) and the control plane unreachable, each round paid a fresh 6 s connect
    # timeout (+ the token fetch), so a six-round turn was 36+ s of dead wall-clock and,
    # in the gateway, a blocked event loop. Back off instead; a success still caches forever.
    import time
    if time.monotonic() < _retry_after:
        return None
    for source in (_from_env_file, _from_private_module, _from_cloud):
        loaded = source()
        if loaded is not None and loaded["block"].strip():
            _cached = loaded             # cache success only
            return _cached
    _retry_after = time.monotonic() + _backoff_s
    _backoff_s = min(_backoff_s * 2.0, 15 * 60.0)
    return None


def reset_backoff() -> None:
    """Forget a failed load NOW — called when a source just became available (the
    device sign-in linked), so the first signed-in turn gets the book instead of
    waiting out the negative cache."""
    global _retry_after, _backoff_s
    _retry_after = 0.0
    _backoff_s = 60.0


def get_block() -> str:
    loaded = _load()
    return loaded["block"] if loaded is not None else _FALLBACK_BLOCK


def get_topics() -> dict[str, tuple[str, str]]:
    loaded = _load()
    return dict(loaded["topics"]) if loaded is not None else {}


def book_loaded() -> bool:
    """True once the book has loaded. Ask it AFTER `get_block()`: that is the load, and
    whatever it returned is what this answer describes."""
    return _cached is not None


def fallback_block() -> str:
    """The degraded-mode block — llm_surface registers its prefix too, so a
    fallback turn still passes the proxy's shape validation."""
    return _FALLBACK_BLOCK


# Back-compat: the rest of the brain historically did
# `from .knowledge import KNOWLEDGE_BLOCK` (and the tests import the table
# helpers). Wherever the private module or an env override exists — every
# local context — these are the real thing at import time. Production code
# paths use get_block(), which can heal after sign-in; this constant cannot.
try:
    from .knowledge_content import *          # noqa: F401,F403 — deliberate re-export
    from .knowledge_content import KNOWLEDGE_BLOCK  # noqa: F401 — explicit for greppers
except ImportError:
    # The shipped brain: no private module. `KNOWLEDGE_BLOCK = get_block()` here ran the
    # LOADER at import (reader E6·R9) - daw_server imports the chat, the chat imports this -
    # so a signed-in user's brain read the keychain and asked the control plane for the book
    # (a six-second timeout) before its handshake, and a failed load armed the sixty-second
    # negative cache before the first turn. "On first use" is now true: nothing in the brain
    # reads this constant, and whoever does gets what get_block() answers THEN.
    def __getattr__(name: str):
        if name == "KNOWLEDGE_BLOCK":
            return get_block()
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
