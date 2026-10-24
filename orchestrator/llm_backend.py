"""Which LLM transport the orchestrator's Claude paths use.

Three backends:

- ``api`` (default) — the Anthropic Messages API on an API key, billed per token
  to whoever owns the key. In the shipped DAW that is the USER's own key (Phase
  53, ``api_key.py``: the brain keeps it in the keychain; an ambient
  ``ANTHROPIC_API_KEY`` wins for dev shells, CI and the evals). This is also the
  direct evaluation path and what the ``benchmark --check`` gate runs on.
- ``subscription`` — a *development* path that routes the same turns through
  the Claude Code engine (Claude Agent SDK) on your Claude subscription, so
  iterating locally doesn't burn API credits. See ``llm_subscription.py``.
- ``cloud`` — the metered control plane: the local brain carries a user's
  short-lived access token; only the control plane owns a company key. Dormant
  at launch (Phase 53); it returns as the per-use hosted tier.

Selected with the ``MIX_AGENT_LLM_BACKEND`` env var. The subscription module is
imported lazily (only when this returns ``"subscription"``), so the shipped
build never depends on ``claude-agent-sdk``.
"""
from __future__ import annotations

import os
import sys
import uuid

BACKEND_ENV = "MIX_AGENT_LLM_BACKEND"
DEFAULT_BACKEND = "api"
_VALID = {"api", "subscription", "cloud"}
CLOUD_URL_ENV = "MIX_AGENT_CLOUD_URL"
ACCESS_TOKEN_ENV = "MIX_AGENT_ACCESS_TOKEN"


def is_frozen() -> bool:
    """True inside the shipped (PyInstaller-frozen) brain, False in a source checkout."""
    return bool(getattr(sys, "frozen", False) or hasattr(sys, "_MEIPASS"))


_warned_frozen_sub = False


def active_backend() -> str:
    """The configured backend: direct API, dev subscription, or cloud proxy.

    `subscription` is a DEVELOPMENT transport, and a shipped brain refuses it (2026-09-22
    audit D·F8): one `launchctl setenv MIX_AGENT_LLM_BACKEND subscription` reached the frozen
    brain, switched the licence gate off and labelled spend "≈ API $" — only the bundle's
    missing SDK kept it from running the agent licence-free on the user's Claude plan. A frozen
    brain answers `api` instead, and says so once on stderr."""
    global _warned_frozen_sub
    value = (os.environ.get(BACKEND_ENV) or DEFAULT_BACKEND).strip().lower()
    if value not in _VALID:
        raise ValueError(
            f"{BACKEND_ENV} must be one of {sorted(_VALID)}, got {value!r}")
    if value == "subscription" and is_frozen():
        if not _warned_frozen_sub:
            _warned_frozen_sub = True
            print(f"[llm_backend] {BACKEND_ENV}=subscription ignored: the subscription "
                  "transport is development-only and this is a shipped brain", file=sys.stderr)
        return DEFAULT_BACKEND
    return value


def llm_available() -> bool:
    """Whether the active backend can actually reach a model right now.

    The DAW asks the brain (``agent: "auto"``) instead of sniffing its own
    process env — the sign-in state and the backend selection live here, not
    in the shell that spawned us.
    """
    try:
        backend = active_backend()
    except ValueError:
        return False
    if backend == "api":
        from .api_key import api_key   # env, else the keychain — never os.environ alone

        return api_key() is not None
    if backend == "subscription":
        return True
    # cloud: a handed-in token (evaluation/CI) or a keychain-backed sign-in.
    if os.environ.get(ACCESS_TOKEN_ENV, "").strip():
        return True
    try:
        from .device_auth import shared_login

        return shared_login().signed_in()
    except Exception:  # noqa: BLE001 — no keychain/login state means no LLM
        return False


def cloud_token() -> str:
    """A token good for the next call.

    ``MIX_AGENT_ACCESS_TOKEN`` wins when set: that is the evaluation and CI path,
    where a token is handed in deliberately and nothing is signed in. Otherwise
    this is a signed-in DAW, and the keychain-backed login mints a fresh one —
    an access token lasts an hour and a mixing session does not, so the token
    cannot be read once and held.
    """
    supplied = os.environ.get(ACCESS_TOKEN_ENV, "").strip()
    if supplied:
        return supplied
    from .device_auth import shared_login  # deferred: only the cloud path needs it

    token = shared_login().access_token()
    if not token:
        raise NoRetry(
            "not signed in — link this machine from the DAW, or set "
            f"{ACCESS_TOKEN_ENV}")
    return token


def cloud_client_kwargs() -> dict:
    """Anthropic SDK constructor args for the compatible metered proxy."""
    raw_url = os.environ.get(CLOUD_URL_ENV, "").strip()
    if not raw_url:
        raise RuntimeError(f"MIX_AGENT_LLM_BACKEND=cloud needs {CLOUD_URL_ENV}")
    from .device_auth import normalize_cloud_url

    base_url = normalize_cloud_url(raw_url)
    return {"base_url": base_url, "api_key": cloud_token()}


#: Where the `api` transport of a SHIPPED brain sends, whatever the environment says.
OFFICIAL_API_URL = "https://api.anthropic.com"
#: What the Anthropic SDK reads from the environment that sends a client somewhere else, or as
#: someone else: the base URL (plain http included); extra headers, which it merges OVER the
#: auth header (`X-Api-Key: <another key>` there replaces the user's); a bearer token.
SDK_REROUTING_ENV = ("ANTHROPIC_BASE_URL", "ANTHROPIC_CUSTOM_HEADERS", "ANTHROPIC_AUTH_TOKEN")


def drop_rerouting_env() -> list[str]:
    """In a shipped brain, take SDK_REROUTING_ENV out of this process's environment and answer
    the names that were there (said once on stderr - names, never values). A no-op in a source
    checkout, whose fakes and dev shells steer every SDK client by exactly these.

    The brain inherits the app's environment, and no `api` client named a base URL (full audit
    2026-10-02, readers S8·R5 = S9·R3): with `launchctl setenv ANTHROPIC_BASE_URL …` on the
    machine - anyone who runs another tool through a gateway has one - the key check
    ("checking the key with Anthropic…"), the key itself, every prompt, the session and the
    knowledge book went to that host, under copy that says "billed by Anthropic", and its 401
    read "Anthropic rejected that key". The development transport already refused to start
    under the variable (`llm_subscription.rerouting_env`); the one that ships did not look."""
    if not is_frozen():
        return []
    dropped = [name for name in SDK_REROUTING_ENV if os.environ.pop(name, None) is not None]
    if dropped:
        print(f"[llm_backend] ignored in a shipped brain: {', '.join(dropped)} - the engineer "
              f"talks to {OFFICIAL_API_URL} on your own key and nowhere else", file=sys.stderr)
    return dropped


def shipped_api_pin() -> dict:
    """Constructor args that hold a direct-API client on Anthropic in a shipped brain; `{}` in
    a source checkout. Every client built on the user's key takes these - the turns
    (`api_client_kwargs`) and the key check (`api_key.validate_api_key`) - and the environment
    is cleaned on the way, for the header and the token no constructor argument can refuse."""
    if not is_frozen():
        return {}
    drop_rerouting_env()
    return {"base_url": OFFICIAL_API_URL}


def api_client_kwargs() -> dict:
    """Anthropic SDK constructor args for the direct API on the user's (or the dev
    shell's) key. Raises the sentence the DAW should show when there is none."""
    from .api_key import MSG_NO_KEY, api_key

    key = api_key()
    if not key:
        raise NoRetry(MSG_NO_KEY)
    return {"api_key": key, **shipped_api_pin()}


def client_kwargs(backend: str) -> dict:
    """Constructor kwargs for whichever transport `backend` names (not subscription)."""
    if backend == "cloud":
        return cloud_client_kwargs()
    if backend == "api":
        return api_client_kwargs()
    return {}


# A brand-new Anthropic account sits on a low usage tier, and a run is ~19 calls in a
# few minutes: a per-minute 429 mid-run is the normal case there, not a fault. The SDK
# waits out `retry-after` itself; on the user's own key it gets more attempts before
# the friendly sentence below gives up the pass. (The cloud proxy's 429 is a usage
# ALLOTMENT — retrying that only re-asks the same question.)
API_MAX_RETRIES = 4


def max_retries_for(backend: str, default: int) -> int:
    return API_MAX_RETRIES if backend == "api" else default


class NoRetry(RuntimeError):
    """A failure that asking again repeats: the sentence names the remedy (a rejected key, an
    ended trial, a model the key cannot use), or the moves have already landed and a retry
    would apply them twice.

    The DAW's "retry" verb re-sends the brief. Whether that makes sense used to be decided in
    the app, by searching the sentence for twenty words ("API key", "trial", "declined", …) -
    so a transient failure that happened to contain one lost its verb, and every new such
    failure needed another word there (pre-merge review 2026-10-04). Whoever builds the
    sentence knows; it says so with this type (or `no_retry = True` on any exception), and
    `daw_server` sends `retryable: false` beside the text."""

    no_retry = True


def retry_helps(exc: BaseException | None) -> bool:
    """False when `exc` - any leaf of a group, or anything it was raised from - is marked
    `no_retry`. Unmarked failures are worth another try: a dropped connection, an overloaded
    API, a bug."""
    seen = 0
    while exc is not None and seen < 8:
        if isinstance(exc, BaseExceptionGroup):
            if not all(retry_helps(sub) for sub in exc.exceptions):
                return False
        if getattr(exc, "no_retry", False):
            return False
        exc, seen = exc.__cause__, seen + 1
    return True


def spend_limit(exc: BaseException) -> str:
    """Which SPEND limit an Anthropic error is, or "": "cap" - the organization's monthly cap
    for its usage tier (a 429 `rate_limit_error` whose `error.details.error_code` is
    `enforced_spend_limit_reached`, with no `retry-after`: retrying cannot succeed until the
    month turns or the tier is raised) - or "own" - a limit the user set themselves in the
    Console (a 400 whose message begins "You have reached your specified API usage limits",
    or "...specified workspace API usage limits"). Neither is a rate limit to wait out."""
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    error = body.get("error") if isinstance(body, dict) and isinstance(body.get("error"), dict) else {}
    text = str(error.get("message") or exc).lower()
    details = error.get("details") if isinstance(error.get("details"), dict) else {}
    if status == 429 and (details.get("error_code") == "enforced_spend_limit_reached"
                          or "reached your api usage limits" in text):
        return "cap"
    if status == 400 and "reached your specified" in text and "usage limits" in text:
        return "own"
    return ""


#: The SDK's and httpx's class names for a request that got no answer in time, and for one
#: whose connection went away. Names, so this module never imports the SDK or httpx.
_TIMEOUT_ERRORS = ("APITimeoutError", "ReadTimeout", "WriteTimeout", "PoolTimeout")
_CONNECTION_ERRORS = ("APIConnectionError", "RemoteProtocolError", "ReadError", "WriteError",
                      "ConnectError", "ConnectTimeout", "NetworkError")


def transient_stream_failure(exc: BaseException) -> str:
    """Why a request that had ALREADY been answered (HTTP 200) then died, when asking again
    may well work: "overloaded" / "server" (an `error` event inside the stream), "timeout"
    (the stream went silent), "dropped" (the connection went away) - or "" for anything else.

    The SDK retries a request that fails before its response starts. Once the stream is open
    it does not, so one overloaded moment or one dropped connection ended the whole turn -
    after a ten-round build, with everything so far applied. The chat loop re-sends such a
    round itself (`ChatSession._turn_api_rounds`), when nothing of it was shown or run."""
    status = getattr(exc, "status_code", None)
    if status not in (None, 200):
        return ""                 # an HTTP status: the SDK has already retried it
    body = getattr(exc, "body", None)
    etype = ""
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        etype = str(body["error"].get("type") or "")
    if etype == "overloaded_error":
        return "overloaded"
    if etype == "api_error":
        return "server"
    if etype:
        return ""                 # a rate limit, a bad request: not a blip
    kind = type(exc).__name__
    if kind in _TIMEOUT_ERRORS:
        return "timeout"
    if kind in _CONNECTION_ERRORS:
        return "dropped"
    return ""


def raise_friendly(backend: str, exc: Exception) -> None:
    """The one call site pattern: translate `exc` for `backend`, or return."""
    if backend == "cloud":
        raise_cloud_friendly(exc)
    elif backend == "api":
        raise_api_friendly(exc)


def raise_api_friendly(exc: Exception) -> None:
    """Translate an Anthropic API error on the user's own key into what to DO.

    Mirror of `raise_cloud_friendly` for the `api` transport (Phase 53): before it, a
    bad key reached the DAW's status bar as `AuthenticationError: Error code: 401 -
    {...}`. The sentences below are quoted by `docs/17-user-manual.md`. Since 2026-09-22 the
    user picks the model, so "this key can't see that model" (404), a Fable model on a
    zero-data-retention organization (400), a conversation past a smaller model's window, a
    5xx and a dropped connection each get their own sentence instead of the SDK's dump.
    """
    status_code = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    detail = ""
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        detail = str(body["error"].get("message") or "")
    text = (detail or str(exc)).lower()
    etype = ""
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        etype = str(body["error"].get("type") or "")
    kind = type(exc).__name__
    if status_code is None or status_code == 200:
        # No HTTP answer at all, or an error that arrived INSIDE a stream that had already
        # answered 200 (2026-09-22 audit A·F3): the SDK's words are "Connection error." /
        # "Request timed out." / `{'type': 'error', 'error': {'type': 'overloaded_error'…}}`,
        # and a dropped stream is a bare httpx error — true, but nothing to act on. (Class
        # names, so this module never imports the SDK or httpx.)
        if etype == "overloaded_error":
            status_code = 529
        elif etype == "rate_limit_error":
            status_code = 429
        elif etype == "api_error":
            status_code = 500
        elif kind in _TIMEOUT_ERRORS:
            raise RuntimeError(
                "Anthropic took too long to answer and the request timed out — try again; "
                "a lower effort level answers sooner") from exc
        elif kind in _CONNECTION_ERRORS:
            raise RuntimeError(
                "the connection to Anthropic dropped — check your internet connection, then "
                "try again") from exc
        else:
            return
    limit = spend_limit(exc)
    if limit == "cap":
        # A 429 that is NOT a rate limit (Anthropic's rate-limit page, "Reaching your spend
        # cap"): it read as "your account's rate limit ... a new account's limits are low",
        # with a retry verb that could only fail until the month turned.
        raise NoRetry(
            "your Anthropic organization has reached its monthly spend cap — it resets on "
            "the 1st (UTC), or request a higher tier at console.anthropic.com ▸ Limits"
        ) from exc
    if limit == "own":
        raise NoRetry(
            "your Anthropic account has reached the spend limit you set for it — raise or "
            "remove it at console.anthropic.com ▸ Settings ▸ Billing") from exc
    if status_code == 400 and "retention" in text:
        # A "covered model" (the Fable line, off the picker since 2026-10-08, was one): an
        # organization on zero data retention gets a 400.
        raise NoRetry(
            "your Anthropic organization's data-retention setting doesn't allow this model "
            "\u2014 pick another model in Preferences \u25b8 Agent") from exc
    if status_code in (400, 413) and ("prompt is too long" in text or "too large" in text
                                      or "context window" in text):
        raise NoRetry(
            "this conversation no longer fits the model's context window \u2014 start a new "
            "session, or pick a model with a larger window") from exc
    if status_code == 404:
        raise NoRetry(
            "this API key can't use the model you picked \u2014 choose another in Preferences "
            "\u25b8 Agent, or check the key's workspace at console.anthropic.com") from exc
    if status_code == 401:
        raise NoRetry(
            "your Anthropic API key was rejected — replace it in Preferences \u25b8 Agent"
        ) from exc
    if status_code == 400 and "credit" in text:
        raise RuntimeError(
            "your Anthropic account is out of credit — top it up at console.anthropic.com, "
            "then try again") from exc
    if status_code == 429:
        raise RuntimeError(
            "your Anthropic account's rate limit — the run waited it out and it did not clear; "
            "a new account's limits are low, and they rise with use") from exc
    if status_code == 529 or (status_code == 503 and "overloaded" in text):
        raise RuntimeError(
            "Anthropic is overloaded right now — nothing is wrong with your key; try again "
            "in a minute") from exc
    if status_code == 403:
        raise NoRetry(
            "this API key isn't allowed to use the model — check the key's workspace "
            "permissions at console.anthropic.com") from exc
    if status_code >= 500:
        raise RuntimeError(
            "Anthropic had a server error — nothing is wrong with your key; try again in a "
            "moment") from exc


def raise_cloud_friendly(exc: Exception) -> None:
    """Translate a control-plane business error into an actionable message.

    The metered proxy answers with plain HTTP statuses — 402 no subscription,
    429 usage allotment, 403 account disabled, 401 auth — whose SDK rendering
    is a raw JSON dump the DAW would show verbatim in the status bar. Call
    sites on the *cloud* backend pass any caught exception through here: if it
    is one of those, a clean RuntimeError is raised (chained, so nothing is
    lost); otherwise this returns and the caller re-raises the original.
    """
    status_code = getattr(exc, "status_code", None)
    if status_code not in (401, 402, 403, 429, 503, 409, 413, 422):
        return
    body = getattr(exc, "body", None)
    detail = ""
    if isinstance(body, dict):
        detail = str(body.get("detail") or "")
        if not detail and isinstance(body.get("error"), dict):
            detail = str(body["error"].get("message") or "")
    suffix = f" ({detail})" if detail else ""
    if status_code == 402:
        raise NoRetry(
            "your ZAMMOTH subscription isn't active" + suffix
            + " — sign in on the ZAMMOTH site to start or renew a plan, "
              "then try again") from exc
    if status_code == 429:
        raise NoRetry(
            "usage limit reached" + suffix
            + " — your plan's allotment resets with the billing cycle; "
              "see your usage on the ZAMMOTH site") from exc
    if status_code == 403:
        raise NoRetry(
            "account access is disabled" + suffix
            + " — contact support via the ZAMMOTH site") from exc
    if status_code == 503:
        # The kill switch, the global spend cap, and a missing model rate all answer 503
        # (backend/app/metering.py) — the one business status this map missed, so a tester
        # saw the raw SDK dump with a retry button that could only fail again (2026-08-09
        # §1 audit). The backend's own detail line says which of the three it was.
        raise NoRetry(
            ("the AI service is paused on our side" if not detail else detail)
            + " — nothing is wrong with your account; try again later") from exc
    if status_code in (409, 413, 422):
        raise RuntimeError(
            "the AI service refused this request" + suffix
            + " — try again; if it repeats, tell us what you did") from exc
    raise NoRetry(
        "your sign-in has expired" + suffix
        + " — use File ▸ Sign In to link this machine again") from exc


def metering_headers(request_type: str) -> dict[str, str]:
    """Per-upstream-call identity: route selector plus retry idempotency."""
    return {
        "X-Metering-Request-Type": request_type,
        "Idempotency-Key": str(uuid.uuid4()),
    }
