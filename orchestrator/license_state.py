"""Whether this install may run the agent (Phase 53): the licence, or the trial.

The app is a one-time perpetual licence; the agent runs on the user's own key. The
control plane answers `GET /v1/license/me` for a signed-in device — `licensed`,
`trial` (with `trial_ends_at`), `expired`, or `none` — and this module holds the
brain's copy of that answer:

- fetched at start and again once a day while online;
- SIGNED by the service (Ed25519, `backend/app/license_signing.py`, 2026-10-20): an answer -
  live or cached - is believed only when a key in `license_keys.py` verifies it
  (`verify`), so a fake licence server or an edited cache file grants nothing;
- believed live only as the answer to THIS request (a fresh nonce the service signs in) for the
  account this machine is signed in as (the signed `sub`), so a signed answer captured earlier -
  before a refund, or from a friend's account - is nothing when a server of one's own plays it
  back (the pre-merge review of 2026-10-20);
- cached on disk (`data_dir()/license.json`, the signed token) and trusted for
  OFFLINE_GRACE_DAYS when the service cannot be reached, so a licensed machine keeps working
  on a plane - aged by the SERVICE's signed clock (`iat`), and only a check that lies in this
  machine's past (`_cache_age_s`);
- never consulted at all on a developer transport: the `subscription` backend, or the
  `api` backend on an ambient ``ANTHROPIC_API_KEY`` (a dev shell, CI, the evals). The
  gate is for a shipped install, whose key comes from the keychain.

Enforced at two doors — the DAW's composer and `daw_server`'s LLM turns — as defence in
depth. This brain is GPL and a determined user can patch it: the licence is a business
mechanism, not DRM. The signature and the nonce close the doors that needed no patch - a file
edit and a server of one's own. What stays a BOUND, not a lock: a cache is believed offline for
the grace from its signed `iat`, so another machine's `license.json` copied in is a licence
there for what is left of its thirty days while the copy cannot name its account (signed out,
or the service unreachable from launch) - a signed-in machine that knows its account refuses
one for another (`_for_this_account`).
"""
from __future__ import annotations

import base64
import json
import math
import os
import secrets
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ENV_OVERRIDE = "MIX_AGENT_LICENSE"       # "licensed": harnesses that drive a shipped-shaped brain
#: A source checkout's own trusted keys, `<kid>:<base64url>[,...]` - a dev control plane, the
#: tests. A shipped (frozen) brain never reads it: its keys are `license_keys.py`'s, or a
#: `launchctl setenv` plus a server of one's own would be a licence.
KEYS_ENV = "MIX_AGENT_LICENSE_KEYS"
#: The statuses a signed answer may carry.
_STATUSES = frozenset({"licensed", "trial", "expired", "none"})
#: The fields of a signed body the brain reads (backend `license_signing.SIGNED_FIELDS` + `iat`).
_BODY_FIELDS = ("status", "trial_ends_at", "trial_days", "purchased_at", "refunded", "iat", "sub")
OFFLINE_GRACE_DAYS = 30
REFRESH_S = 24 * 3600
# A NEGATIVE answer (expired / refunded / none) is re-asked within a minute, and a refusal
# re-asks before refusing: it was trusted for 24 h, so a user who had just paid kept reading
# "your trial has ended — buy Zammoth" until they relaunched (2026-09-22 audit C·F1).
NEGATIVE_REFRESH_S = 60.0
REFUSAL_RECHECK_S = 15.0
FETCH_TIMEOUT_S = 8.0
#: How far AHEAD of this machine's clock a cache's `checked_at` may lie and still be a cache: a
#: clock that was corrected backwards a little since the check. Past it the check "has not
#: happened yet" - an edited file, or the date set back to before it - and there is no cache.
CLOCK_SKEW_S = 300.0

# The sentences the DAW shows (docs/17-user-manual.md quotes them).
#: The trial's length before the service has said (Phase 62: three days, `TRIAL_DAYS` on the
#: control plane). Once it has, `state()["trial_days"]` is the service's own number.
TRIAL_DAYS_DEFAULT = 3
MSG_SIGN_IN = ("sign in to start your 3-day trial of the engineer — File ▸ Sign In "
               "(the trial runs on your own Anthropic key)")
MSG_EXPIRED = ("your trial has ended — buy Zammoth at zammoth.com/account, or enter a "
               "product key in Preferences ▸ Account, to keep the engineer; everything "
               "else in the app still works")
MSG_REFUNDED = ("this licence was refunded — buy Zammoth again at zammoth.com/account to "
                "use the engineer")
MSG_UNREACHABLE = ("can't reach the licence service to confirm your trial or licence — "
                   "check your connection and try again")
# The service answered, and the answer did not carry Zammoth's signature: something between this
# computer and zammoth.com answered in its place (a proxy, a filter, a redirect).
MSG_UNVERIFIED = ("can't confirm your trial or licence — the answer from the licence service "
                  "could not be verified; check that nothing on this computer redirects "
                  "zammoth.com, then try again")
# Signed out, but this machine's last confirmed answer was a licence (a sign-in session lapses
# after 90 days UNUSED or a year after it was made, or is revoked): not "start your trial" — the licence is theirs (2026-09-22 audit C·F5).
MSG_RESIGN_IN = ("your sign-in has lapsed — sign in again (File ▸ Sign In) to keep your "
                 "licence confirmed; it keeps working offline meanwhile")

# ...and the same person once that cached answer is older than the offline grace (P2 batch T,
# reader S8·R3): the agent stops until they sign in, and the sentence says THAT - they were told
# to "start your trial", thirty days after paying.
MSG_RECONFIRM = ("sign in again to confirm your licence — File ▸ Sign In (this computer last "
                 "confirmed it more than 30 days ago)")

_cached: Optional[dict] = None    # the last fetched payload + checked_at, in memory
_cached_offline = False           # _cached came from the disk cache, the service unreachable
_next_fetch_at = 0.0
_last_recheck = 0.0
#: The service did not answer and there was no usable cache: not asked again before this
#: (monotonic). `_next_fetch_at` could not do it (reader S8·R17) - it is consulted only when
#: `_cached` holds something, and in exactly this case nothing is cached - so the sixty
#: seconds were set and never read, and every hello and every send paid a connect timeout.
_unreachable_until = 0.0
#: The nonce of the request `_fetch` is making, while it waits for the answer (None between).
_awaiting_nonce: Optional[str] = None


class UnverifiedAnswer(RuntimeError):
    """The licence service's reply carried no signature a trusted key verifies."""


def _b64url_decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def trusted_keys() -> dict[str, bytes]:
    """kid -> 32-byte Ed25519 public key. `license_keys.py`, or on a source checkout only,
    `MIX_AGENT_LICENSE_KEYS`."""
    from .license_keys import TRUSTED_LICENSE_KEYS
    from .llm_backend import is_frozen
    table: dict[str, str] = dict(TRUSTED_LICENSE_KEYS)
    raw = os.environ.get(KEYS_ENV, "").strip()
    if raw and not is_frozen():
        table = {}
        for pair in raw.split(","):
            kid, _, text = pair.strip().partition(":")
            if kid and text:
                table[kid] = text
    keys: dict[str, bytes] = {}
    for kid, text in table.items():
        try:
            key = _b64url_decode(text)
        except (ValueError, TypeError):
            continue
        if len(key) == 32:
            keys[kid] = key
    return keys


def verify(token) -> Optional[dict]:
    """The body of a signed licence answer, or None when it does not verify: not a token, an
    unknown kid, a bad signature, or a body that is not an answer. Never raises."""
    if not isinstance(token, str) or token.count(".") != 1 or len(token) > 4096:
        return None
    body_text, sig_text = token.split(".")
    try:
        body = json.loads(_b64url_decode(body_text))
        signature = _b64url_decode(sig_text)
    except (ValueError, TypeError):
        return None
    if not isinstance(body, dict) or body.get("v") != 1:
        return None
    key = trusted_keys().get(str(body.get("kid")))
    if key is None:
        return None
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    try:
        Ed25519PublicKey.from_public_bytes(key).verify(signature, body_text.encode("ascii"))
    except (InvalidSignature, ValueError, UnicodeEncodeError):
        return None
    iat = body.get("iat")
    if (isinstance(iat, bool) or not isinstance(iat, (int, float)) or not math.isfinite(iat)
            or body.get("status") not in _STATUSES):
        return None
    return body


def _answer(token: str, body: dict, checked_at) -> dict:
    """A payload as the rest of this module reads it, from a verified token."""
    return {**{name: body.get(name) for name in _BODY_FIELDS},
            "signed": token, "checked_at": checked_at}


def _cache_path() -> Path:
    from .calibration import data_dir
    return data_dir() / "license.json"


def reset() -> None:
    """Forget the in-memory copy (after a sign-in or a purchase) — the next state()
    fetches again. The on-disk cache stays: it is the offline fallback."""
    global _cached, _cached_offline, _next_fetch_at, _last_recheck, _unreachable_until
    _cached = None
    _cached_offline = False
    _next_fetch_at = 0.0
    _last_recheck = 0.0
    _unreachable_until = 0.0     # a fresh link's first check is asked, never backed off


def forget() -> None:
    """A SIGN-OUT: the licence this machine remembers goes with the session (reader S8·R8;
    James, 2026-10-06). `reset()` alone left license.json, and a signed-out machine believes a
    `licensed` cache for the whole offline grace - right for a session that LAPSED, wrong for
    somebody who signed out of a shared computer."""
    try:
        _cache_path().unlink(missing_ok=True)
    except OSError as exc:
        print(f"[license_state] the licence cache could not be removed ({exc})", file=sys.stderr)
    reset()


def gate_applies() -> bool:
    """False on every developer transport; True for a shipped install.

    A shipped (frozen) brain is always gated (2026-09-22 audit C·F10): the harness override
    and the "ambient ANTHROPIC_API_KEY means a dev shell" rule are for source checkouts, and a
    `launchctl setenv` of either used to switch the licence off in the installed app. A
    frozen brain still USES an ambient key — it just doesn't read it as a developer's."""
    from .llm_backend import active_backend, is_frozen
    frozen = is_frozen()
    if not frozen and os.environ.get(ENV_OVERRIDE, "").strip().lower() == "licensed":
        return False
    try:
        backend = active_backend()
    except ValueError:
        return frozen
    if backend == "subscription":
        return False   # never reached frozen: active_backend refuses it there
    if backend == "api":
        if frozen:
            return True
        from .api_key import api_key_source
        return api_key_source() == "keychain"
    return True   # cloud: the metered proxy's user is a licence holder too


def _read_cache() -> Optional[dict]:
    """The cached answer, if its signature verifies. A cache from before 2026-10-20 (the plain
    fields, no `signed`) and an edited one are each no cache."""
    try:
        raw = json.loads(_cache_path().read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — no cache is just no cache
        return None
    if not isinstance(raw, dict):
        return None
    body = verify(raw.get("signed"))
    return None if body is None else _answer(raw["signed"], body, raw.get("checked_at"))


def _write_cache(payload: dict) -> None:
    """Keep the signed token: the only part of an answer worth keeping, since it is the only
    part a later read will believe."""
    token = payload.get("signed")
    if not token:
        return
    try:
        path = _cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"signed": token, "checked_at": payload.get("checked_at")}),
                        encoding="utf-8")
    except Exception:  # noqa: BLE001 — a cache write must never fail a turn
        pass


def _fetch() -> dict:
    """GET /v1/license/me as the signed-in device. Raises on anything but a 200, and
    `UnverifiedAnswer` on a 200 whose signature does not verify, that answers another request
    (its nonce), or that names another account than the token the request carried."""
    import httpx

    from .device_auth import shared_login, token_subject
    global _awaiting_nonce
    login = shared_login()
    access = login.access_token()
    if not access:
        raise RuntimeError("not signed in")
    nonce = secrets.token_urlsafe(16)
    _awaiting_nonce = nonce
    try:
        response = httpx.get(f"{login.cloud_url}/v1/license/me", params={"nonce": nonce},
                             timeout=FETCH_TIMEOUT_S,
                             headers={"Authorization": f"Bearer {access}"})
        if response.status_code != 200:
            raise RuntimeError(f"license service answered HTTP {response.status_code}")
        try:
            answer = response.json()
        except ValueError:
            raise UnverifiedAnswer("the licence service's reply is not JSON") from None
    finally:
        _awaiting_nonce = None
    signed = answer.get("signed") if isinstance(answer, dict) else None
    body = verify(signed)
    if body is None:
        raise UnverifiedAnswer("the licence service's reply carries no verifiable signature")
    if body.get("nonce") != nonce:
        raise UnverifiedAnswer("the licence service's reply answers another request "
                               "(a signed answer played back)")
    account = token_subject(access)
    if account is not None and body.get("sub") != account:
        raise UnverifiedAnswer("the licence service's reply is for another account")
    return _answer(signed, body, time.time())


def _for_this_account(payload: dict) -> bool:
    """False when this machine can name the account it is signed in as and `payload` was signed
    for another: a `license.json` copied from someone else's machine."""
    try:
        from .device_auth import shared_login
        account = shared_login().account_id()
    except Exception:  # noqa: BLE001 — no login = nobody to compare with
        return True
    return account is None or payload.get("sub") == account


def _parse_ts(value) -> Optional[datetime]:
    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _derive(payload: dict, *, offline: bool) -> dict:
    """The DAW-facing state from a service payload (fresh or cached)."""
    status = str(payload.get("status") or "none")
    ends = _parse_ts(payload.get("trial_ends_at"))
    days_left = None
    if status == "trial":
        # A cached trial keeps counting down offline: the clock, not the cache, ends it.
        if ends is not None:
            remaining = (ends - datetime.now(timezone.utc)).total_seconds()
            if remaining <= 0:
                status = "expired"
            else:
                days_left = max(1, int(remaining // 86400) + (1 if remaining % 86400 else 0))
        else:
            status = "expired"
    message = ""
    if status == "expired":
        message = MSG_REFUNDED if payload.get("refunded") else MSG_EXPIRED
    elif status == "none":
        message = MSG_SIGN_IN
    try:
        trial_days = int(payload.get("trial_days") or TRIAL_DAYS_DEFAULT)
    except (TypeError, ValueError):
        trial_days = TRIAL_DAYS_DEFAULT
    return {"status": status, "gated": True, "offline": offline,
            "days_left": days_left, "trial_days": trial_days,
            "trial_ends_at": payload.get("trial_ends_at"),
            "purchased_at": payload.get("purchased_at"), "message": message}


def _cache_age_s(cached: dict) -> float:
    """Seconds since the service signed the answer a cache holds (`iat`, the SERVICE's clock) -
    infinity (no cache) when it records no check that can have happened.

    Until 2026-10-20 the age came from `checked_at`, a number on the user's disk: the year 2100
    there was "under 30 days old" for ever (full audit 2026-10-02, reader S8·R4, bounded then
    from above), and any date inside the grace was a licence. Now the number is signed. A check
    dated ahead of the clock (beyond CLOCK_SKEW_S) is still no cache - the date set back to
    before the check - and a clock set back to INSIDE the grace still reads it: a bound, not a
    lock. The brain is GPL and the licence is a business mechanism, not DRM."""
    try:
        age = time.time() - float(cached.get("iat") or 0)
    except (TypeError, ValueError):   # a corrupted license.json is no cache (C·F12)
        return float("inf")
    if not math.isfinite(age) or age < -CLOCK_SKEW_S:
        return float("inf")
    return age


def state(force: bool = False) -> dict:
    """The current licence state. Blocking (may do one HTTP call) — call it off the loop.
    Never raises (2026-09-22 audit C·F12): the hello carries it, and a bad field in the disk
    cache failed the whole hello — routing, co-edits, export resync — not just the agent.

    Keys: status (licensed | trial | expired | none | unknown), gated (False on a
    developer transport, where status is 'licensed'), offline, days_left, message.
    """
    try:
        return _state(force)
    except Exception as exc:  # noqa: BLE001
        print(f"[license_state] state unavailable ({type(exc).__name__}: {exc})",
              file=sys.stderr)
        return {"status": "unknown", "gated": True, "offline": True, "days_left": None,
                "message": MSG_UNREACHABLE}


def _state(force: bool) -> dict:
    global _cached, _cached_offline, _next_fetch_at, _unreachable_until
    if not gate_applies():
        return {"status": "licensed", "gated": False, "offline": False,
                "days_left": None, "message": ""}
    try:
        from .device_auth import shared_login
        signed_in = shared_login().signed_in()
    except Exception:  # noqa: BLE001 — no cloud URL / no keychain = not signed in
        signed_in = False
    if not signed_in:
        cached = _read_cache()
        if (cached is not None and cached.get("status") == "licensed"
                and _cache_age_s(cached) <= OFFLINE_GRACE_DAYS * 86400):
            return {**_derive(cached, offline=True), "message": MSG_RESIGN_IN}
        # ...a licence this machine really confirmed, once: a check dated ahead of the clock, or
        # with no date at all, is no cache (_cache_age_s) and proves nothing about a purchase.
        lapsed_licence = (cached is not None and cached.get("status") == "licensed"
                          and math.isfinite(_cache_age_s(cached))
                          and bool(cached.get("iat")))
        return {"status": "none", "gated": True, "offline": False, "days_left": None,
                "message": MSG_RECONFIRM if lapsed_licence else MSG_SIGN_IN}
    now = time.monotonic()
    if _cached is not None and not force and now < _next_fetch_at:
        return _derive(_cached, offline=_cached_offline)
    if _cached is None and not force and now < _unreachable_until:
        return {"status": "unknown", "gated": True, "offline": True, "days_left": None,
                "message": MSG_UNREACHABLE}
    try:
        payload = _fetch()
    except Exception as exc:  # noqa: BLE001 — offline, the service down, or an unsigned reply
        unverified = isinstance(exc, UnverifiedAnswer)
        if unverified:
            print(f"[license_state] {exc}", file=sys.stderr)
        cached = _cached or _read_cache()
        if (cached is not None and _for_this_account(cached)
                and _cache_age_s(cached) <= OFFLINE_GRACE_DAYS * 86400):
            # held IN MEMORY too, so the next five minutes of turns don't each pay a
            # connect timeout — the disk cache alone never engaged this backoff (C·F4)
            _cached, _cached_offline = cached, True
            _next_fetch_at = now + 300.0     # ask again in five minutes, not per turn
            return _derive(cached, offline=True)
        _unreachable_until = now + 60.0
        return {"status": "unknown", "gated": True, "offline": True, "days_left": None,
                "message": MSG_UNVERIFIED if unverified else MSG_UNREACHABLE}
    _cached, _cached_offline = payload, False
    _unreachable_until = 0.0
    positive = _derive(payload, offline=False)["status"] in ("licensed", "trial")
    _next_fetch_at = now + (REFRESH_S if positive else NEGATIVE_REFRESH_S)
    _write_cache(payload)
    return _derive(payload, offline=False)


MSG_REDEEM_SIGN_IN = "sign in first (File ▸ Sign In) — a product key goes on your account"


def redeem(key: str) -> dict:
    """Phase 62: spend a product key on the signed-in account (`POST /v1/license/redeem`).

    Answers `{ok: True, license: <state>}`, or `{ok: False, error: <the service's sentence>}`.
    The key is sent once, as the request's body, and never logged or kept."""
    import httpx

    from .device_auth import DeviceAuthError, shared_login
    typed = str(key or "").strip()
    if not typed:
        return {"ok": False, "error": "type a product key first"}
    try:
        login = shared_login()
        token = login.access_token()
    except DeviceAuthError as exc:
        return {"ok": False, "error": str(exc)}
    if not token:
        return {"ok": False, "error": MSG_REDEEM_SIGN_IN}
    try:
        response = httpx.post(f"{login.cloud_url}/v1/license/redeem", json={"key": typed},
                              timeout=FETCH_TIMEOUT_S,
                              headers={"Authorization": f"Bearer {token}"})
    except httpx.HTTPError:
        return {"ok": False, "error": "can't reach the licence service — check your "
                                      "connection and try again"}
    if response.status_code != 200:
        detail = ""
        try:
            detail = str(response.json().get("detail") or "")
        except ValueError:
            pass
        if not detail or len(detail) > 300:
            detail = f"the licence service refused the key (HTTP {response.status_code})"
        return {"ok": False, "error": detail}
    reset()                      # the next read asks the service, and finds the licence
    return {"ok": True, "license": state(force=True)}


def refusal() -> Optional[str]:
    """The sentence to refuse an LLM turn with, or None when the agent may run.

    A negative answer is re-read from the service before it refuses (at most every
    REFUSAL_RECHECK_S): the moment that matters most is the one right after a purchase."""
    global _last_recheck
    current = state()
    if current["status"] in ("licensed", "trial"):
        return None
    now = time.monotonic()
    if current["status"] != "none" and now - _last_recheck >= REFUSAL_RECHECK_S:
        _last_recheck = now
        current = state(force=True)
        if current["status"] in ("licensed", "trial"):
            return None
    return current.get("message") or MSG_UNREACHABLE
