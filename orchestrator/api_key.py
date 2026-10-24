"""The user's own Anthropic API key (Phase 53).

The DAW's agent runs on the USER's key. The brain owns it the way it owns the device
refresh token (`device_auth.py`): the DAW hands it over once, on the authenticated
NDJSON socket (`set_api_key`), the brain validates it with one free call and stores it
in the login keychain, and every later start reads it from there. The DAW never keeps
a copy, and the key never rides the brain's spawn command line — `BrainClient::launch`
builds a `/bin/sh -c` prefix, and anything in it is readable by every local user
through `ps` and lands in crash reports.

Resolution order, first hit wins:
  1. ``ANTHROPIC_API_KEY`` in the environment — dev shells, CI, the evals.
  2. the keychain entry this module writes — a shipped install.
The key goes into the SDK client as ``api_key=``; it is never written into
``os.environ``, so nothing that inherits this process's environment sees it.
"""
from __future__ import annotations

import os
import re
from typing import Optional

from .device_auth import DeviceAuthError, Keychain, MemoryKeychain
from .llm_models import DEFAULT_MODEL

ENV = "ANTHROPIC_API_KEY"
KEYCHAIN_ACCOUNT = "anthropic_api_key"
# What Anthropic issues (sk-ant-<tier>-<body>). The body is URL-safe; nothing else
# is allowed anywhere near the keychain command line.
KEY_RE = re.compile(r"^sk-ant-[A-Za-z0-9_-]{20,300}$")
#: What a new key is checked against: the default model. A key that cannot see it is refused
#: (MSG_NO_MODEL) even though the picker offers other models since 2026-09-22 — the default is
#: what a fresh install runs, and every quality gate is measured on it.
VALIDATION_MODEL = DEFAULT_MODEL

# The sentences the DAW shows. `docs/17-user-manual.md` quotes them; keep them in step.
MSG_MALFORMED = ("that doesn't look like an Anthropic API key — it starts with "
                 "\"sk-ant-\"; copy the whole key from console.anthropic.com")
MSG_REJECTED = ("Anthropic rejected that key — check it was copied whole and has not "
                "been revoked")
MSG_NO_MODEL = ("this key can't reach the model (a workspace-scoped or restricted key?) "
                "— create one with access to Claude Opus")
MSG_OFFLINE = "can't reach Anthropic to check the key — check your connection and try again"
MSG_NO_KEY = ("no Anthropic API key — add yours in Preferences ▸ Agent (about a dollar "
              "a run on the default model, billed by Anthropic, not by us)")


class ApiKeyError(RuntimeError):
    pass


_store = None   # swapped by tests (a MemoryKeychain) — see use_store()


def _keychain():
    global _store
    if _store is None:
        _store = Keychain(account=KEYCHAIN_ACCOUNT, pattern=KEY_RE)
    return _store


def use_store(store) -> None:
    """Tests: point this module at a MemoryKeychain instead of the login keychain."""
    global _store
    _store = store


def memory_store() -> MemoryKeychain:
    return MemoryKeychain(pattern=KEY_RE)


def api_key_source() -> str:
    """'env' | 'keychain' | 'none' — where the next client's key would come from."""
    if os.environ.get(ENV, "").strip():
        return "env"
    try:
        if _keychain().get() is not None:
            return "keychain"
    except Exception:  # noqa: BLE001 — no keychain (Linux CI) means no key, not a crash
        pass
    return "none"


def api_key() -> Optional[str]:
    """The key to construct a client with, or None. Env wins (dev, CI, evals)."""
    supplied = os.environ.get(ENV, "").strip()
    if supplied:
        return supplied
    try:
        return _keychain().get()
    except Exception:  # noqa: BLE001
        return None


def api_key_state() -> dict:
    """What the DAW's Preferences show. Never the key itself.

    `needed`: whether THIS brain's transport runs on an Anthropic key at all (2026-09-22 audits
    D·F7 / E·F7). The DAW's composer door refused every send with no key stored — also on the
    subscription dev backend, which must run WITHOUT one, so `dawdev` only worked when a real
    production key happened to sit in the keychain."""
    source = api_key_source()
    try:
        from .llm_backend import active_backend
        needed = active_backend() == "api"
    except ValueError:
        needed = True
    return {"set": source != "none", "source": source, "needed": needed}


def validate_api_key(key: str, *, client_factory=None) -> None:
    """One FREE call — `models.retrieve` bills no tokens — that proves the key is live
    AND that its workspace can see the model the DAW runs. Raises ApiKeyError with the
    sentence the user should read."""
    if client_factory is None:
        import anthropic  # deferred: tests inject a factory

        # One attempt, 15 s: the DAW waits 45 s for the answer, and the SDK honours a
        # `retry-after` of up to 60 s — a 529 made validation outlast the wait, the DAW said
        # "did not answer" and the brain stored the key anyway (2026-09-22 audit C·F11).
        # ...and to Anthropic: in a shipped brain the check is pinned there like every turn
        # (`shipped_api_pin`, readers S8·R5 = S9·R3). It followed an ambient ANTHROPIC_BASE_URL,
        # so the key in the field went to that host before it was ever stored.
        def client_factory(k: str):
            from .llm_backend import shipped_api_pin
            return anthropic.Anthropic(api_key=k, timeout=15.0, max_retries=0,
                                       **shipped_api_pin())
    try:
        client_factory(key).models.retrieve(VALIDATION_MODEL)
    except Exception as exc:  # noqa: BLE001 — every failure is one of four sentences
        status_code = getattr(exc, "status_code", None)
        if status_code == 401:
            raise ApiKeyError(MSG_REJECTED) from exc
        if status_code in (403, 404):
            raise ApiKeyError(MSG_NO_MODEL) from exc
        if status_code is None:
            raise ApiKeyError(MSG_OFFLINE) from exc
        raise ApiKeyError(f"Anthropic answered HTTP {status_code} while checking the key "
                          "— try again in a moment") from exc


def set_api_key(key: str, *, client_factory=None) -> dict:
    """Validate, then store. A key that fails validation is never kept: the DAW asks
    for a replacement, it does not run on a dead one. Returns the new `api_key_state()`
    plus `ok`/`error`; the key itself is never echoed."""
    key = (key or "").strip()
    if not KEY_RE.fullmatch(key):
        return {"ok": False, "error": MSG_MALFORMED, **api_key_state()}
    try:
        validate_api_key(key, client_factory=client_factory)
    except ApiKeyError as exc:
        return {"ok": False, "error": str(exc), **api_key_state()}
    try:
        _keychain().set(key)
    except DeviceAuthError as exc:
        return {"ok": False, "error": f"could not store the key: {exc}", **api_key_state()}
    return {"ok": True, **api_key_state()}


def clear_api_key() -> dict:
    """Remove the stored key — and say so only when it is gone (2026-09-22 audit C·F11: a
    failed delete reported "API key removed" while the key stayed in use)."""
    try:
        _keychain().delete()
    except Exception:  # noqa: BLE001
        pass
    try:
        still = _keychain().get() is not None
    except Exception:  # noqa: BLE001 — no keychain to read = nothing stored
        still = False
    if still:
        return {"ok": False, "error": "the key could not be removed from the keychain",
                **api_key_state()}
    return {"ok": True, **api_key_state()}
