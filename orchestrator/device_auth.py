"""Device login for the brain — the client half of RFC 8628.

The DAW never holds the company key and the user never types a password into it.
The app shows a short code, the user approves it on the website, and this module
trades its secret ``device_code`` for tokens the control plane mints.

What lives where matters:

- The **refresh token** is durable, so it goes in the login keychain.
- The **access token** lasts an hour and a mixing session does not, so it is kept
  in memory and renewed on demand rather than read once at startup.

The brain owns this rather than the DAW because the brain is what calls the
control plane, and it is the thing that needs a *fresh* token per request.
"""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import hashlib
import ipaddress
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

import httpx


KEYCHAIN_SERVICE = "com.zammoth.cloud"
KEYCHAIN_ACCOUNT = "device_refresh_token"
# Renew a little early: a token that expires mid-flight fails the call it was
# fetched for.
EXPIRY_SKEW_S = 60
# What the control plane mints (secrets.token_urlsafe). Anything else is not
# ours, and must never reach the keychain command line.
TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{16,256}$")

CLOUD_URL_ENV = "MIX_AGENT_CLOUD_URL"



# The control plane's word for "this code was already exchanged" (backend/app/devices.py
# CLAIMED_REASON), and what the DAW says when it hears it.
CLAIMED_REASON = "already_claimed"
MSG_CLAIM_LOST = ("the sign-in was approved, but its reply did not reach this computer "
                  "— start the sign-in again")

class DeviceAuthError(RuntimeError):
    pass


def normalize_cloud_url(value: str) -> str:
    """Return a safe origin for credential-bearing control-plane requests."""
    raw = value.strip()
    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except ValueError as exc:
        raise DeviceAuthError("the cloud URL is not valid") from exc
    hostname = parsed.hostname or ""
    try:
        loopback = hostname.lower() == "localhost" or ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        loopback = hostname.lower() == "localhost"
    if parsed.scheme not in {"http", "https"} or (
            parsed.scheme == "http" and not loopback):
        raise DeviceAuthError("the cloud URL must use HTTPS (HTTP is loopback-only)")
    if (not hostname or parsed.username or parsed.password
            or parsed.path not in {"", "/"} or parsed.query or parsed.fragment
            or (port is None and parsed.netloc.endswith(":"))):
        raise DeviceAuthError("the cloud URL must be an origin without credentials or a path")
    try:
        ascii_host = hostname.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise DeviceAuthError("the cloud URL hostname is not valid") from exc
    if ":" in ascii_host:
        ascii_host = f"[{ascii_host}]"
    default_port = 443 if parsed.scheme == "https" else 80
    authority = ascii_host if port in {None, default_port} else f"{ascii_host}:{port}"
    return f"{parsed.scheme.lower()}://{authority}"


def _keychain_account(cloud_url: str) -> str:
    origin_hash = hashlib.sha256(cloud_url.encode("utf-8")).hexdigest()[:20]
    return f"{KEYCHAIN_ACCOUNT}:{origin_hash}"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


#: What a `security` call that outlived its bound is turned into (never the raw
#: TimeoutExpired: its text carries the command line).
MSG_KEYCHAIN_SILENT = "the keychain did not answer - unlock it and try again"


def _security(argv: list[str], **kwargs) -> "subprocess.CompletedProcess[str]":
    """The ONE call to /usr/bin/security, always bounded (reader S8·R9: `set` had no timeout,
    so a locked login keychain held the brain's connection on the unlock prompt for as long
    as the prompt stood). One door, too, so the test suite can stand in front of it: no test
    reads or writes this machine's login keychain (S8·R15, server/tests/conftest.py)."""
    if no_keychain():
        return subprocess.CompletedProcess(argv, NOT_FOUND, "", "")
    return subprocess.run(argv, capture_output=True, text=True,
                          timeout=Keychain.TIMEOUT_S, **kwargs)


#: `security`'s exit status for "the specified item could not be found in the keychain".
NOT_FOUND = 44
#: A source checkout run with this set to "none" never calls /usr/bin/security: every read
#: finds nothing and every write is refused (Phase 56, the chat rig - a brain started by a test
#: harness must not read the developer's sign-in or leave an item in their login keychain).
#: A shipped brain ignores it, like every other development switch.
NO_KEYCHAIN_ENV = "MIX_AGENT_KEYCHAIN"


def no_keychain() -> bool:
    from .llm_backend import is_frozen
    return os.environ.get(NO_KEYCHAIN_ENV, "").strip().lower() == "none" and not is_frozen()


#: Where a brain signed with a Developer ID keeps its secrets: items it makes itself through
#: Security.framework, which trust it and not `/usr/bin/security` (orchestrator/
#: keychain_framework.py, 2026-10-23). `<service>.v2`, so an item from before is never read by the
#: brain directly - that would prompt, as the old item trusts only the security tool.
SIGNED_SERVICE_SUFFIX = ".v2"

_TEAM_RE = re.compile(r"^TeamIdentifier=([A-Z0-9]{10})$", re.M)
_signed_team: "str | None | bool" = False        # False = not asked yet


def signed_brain_team() -> str | None:
    """The Developer ID team this brain is signed by, or None: a source checkout (its program is
    Python), or an ad-hoc build, whose identity changes with every build - binding its items would
    ask for the login password at each new build. Asked of the brain's own signature once."""
    global _signed_team
    if _signed_team is False:
        from .llm_backend import is_frozen
        team = None
        if is_frozen() and sys.platform == "darwin":
            try:
                out = subprocess.run(["/usr/bin/codesign", "-dvv", sys.executable],
                                     capture_output=True, text=True, timeout=10)
                match = _TEAM_RE.search(out.stderr)
                team = match.group(1) if match else None
            except (OSError, subprocess.TimeoutExpired):
                team = None
        _signed_team = team
    return _signed_team


class Keychain:
    """The macOS login keychain.

    A brain signed with a Developer ID keeps the secret in an item of its own
    (`<service>.v2`, through Security.framework: the item trusts the signed brain and nothing
    else), and moves an item from before over once. Any other brain uses /usr/bin/security,
    as before.

    `pattern` is what a stored secret must look like — the refresh token's shape by
    default; the user's Anthropic API key (Phase 53, `orchestrator/api_key.py`) passes
    its own. Nothing outside the pattern ever reaches the keychain command line.

    `signed` and `keychain_path` are for tests: a FrameworkStore on a throwaway keychain, and
    that keychain's file, which every `security` command is then aimed at.
    """

    def __init__(self, service: str = KEYCHAIN_SERVICE,
                 account: str = KEYCHAIN_ACCOUNT,
                 pattern: "re.Pattern[str]" = TOKEN_RE, *,
                 signed=None, keychain_path: "str | None" = None) -> None:
        self.service = service
        self.account = account
        self.pattern = pattern
        self._signed = signed            # None = decide at first use (signed_brain_team)
        self._keychain_path = keychain_path

    #: A `security` call that has not answered in this long is sitting on macOS's keychain
    #: unlock prompt (2026-09-22 audit C·F9) — read as "no secret", never a hang.
    TIMEOUT_S = 20.0

    def _signed_store(self):
        if self._signed is None:
            if signed_brain_team() is None:
                self._signed = False
            else:
                from .keychain_framework import FrameworkStore
                self._signed = FrameworkStore(self.service + SIGNED_SERVICE_SUFFIX, self.account)
        return self._signed or None

    def _where(self) -> list[str]:
        return [self._keychain_path] if self._keychain_path else []

    def get(self) -> str | None:
        store = self._signed_store()
        if store is None:
            return self._tool_get()
        from .keychain_framework import FrameworkError
        try:
            raw = store.get()
        except FrameworkError:
            return None
        if raw is not None:
            secret = raw.decode("utf-8", errors="replace")
            return secret if self.pattern.fullmatch(secret) else None
        # Nothing bound yet: an item from before moves over. It is read through the tool, which
        # it trusts (no prompt), written as the brain's own, and only then removed.
        secret = self._tool_get()
        if secret is not None:
            try:
                store.set(secret.encode())
            except FrameworkError:
                return secret            # the old item stays; the move is tried at the next read
            self._tool_delete_quietly()
        return secret

    def set(self, secret: str) -> None:
        if not self.pattern.fullmatch(secret):
            raise DeviceAuthError("refusing to store a malformed token")
        store = self._signed_store()
        if store is None:
            return self._tool_set(secret)
        from .keychain_framework import FrameworkError
        try:
            store.set(secret.encode())
        except FrameworkError as exc:
            raise DeviceAuthError(MSG_KEYCHAIN_SILENT if exc.status is None
                                  else f"could not write to the keychain (status {exc.status})"
                                  ) from None
        self._tool_delete_quietly()      # an item from before must not outlive the new one

    def delete(self) -> None:
        store = self._signed_store()
        if store is not None:
            from .keychain_framework import FrameworkError
            try:
                store.delete()
            except FrameworkError as exc:
                raise DeviceAuthError(MSG_KEYCHAIN_SILENT if exc.status is None
                                      else f"could not remove the keychain item "
                                           f"(status {exc.status})") from None
        try:
            _security(["/usr/bin/security", "delete-generic-password",
                       "-s", self.service, "-a", self.account, *self._where()])
        except subprocess.TimeoutExpired:
            raise DeviceAuthError(MSG_KEYCHAIN_SILENT) from None

    # ---- /usr/bin/security: every brain that is not signed, and an item from before ----------

    def _tool_get(self) -> str | None:
        try:
            result = _security(["/usr/bin/security", "find-generic-password",
                                "-s", self.service, "-a", self.account, "-w", *self._where()])
        except subprocess.TimeoutExpired:
            return None
        if result.returncode != 0:
            return None
        secret = result.stdout.strip()
        return secret if self.pattern.fullmatch(secret) else None

    def _tool_set(self, secret: str) -> None:
        # `security -i` reads its command from stdin. Passing the secret as an
        # argument instead would publish it in `ps` for the life of the call.
        where = f" {self._keychain_path}" if self._keychain_path else ""
        try:
            result = _security(
                ["/usr/bin/security", "-i"],
                input=(f"add-generic-password -U -s {self.service} "
                       f"-a {self.account} -w {secret}{where}\n"))
        except subprocess.TimeoutExpired:
            # `subprocess.run` has killed the child, so there is no late write either.
            raise DeviceAuthError(MSG_KEYCHAIN_SILENT) from None
        if result.returncode != 0:
            # The keychain tool may echo the rejected input command in an
            # error. That command contains the refresh token, so never surface
            # its stderr in logs or the DAW UI.
            raise DeviceAuthError(
                f"could not write to the keychain (exit {result.returncode})")

    def _tool_delete_quietly(self) -> None:
        try:
            _security(["/usr/bin/security", "delete-generic-password",
                       "-s", self.service, "-a", self.account, *self._where()])
        except (OSError, subprocess.TimeoutExpired):
            pass


class MemoryKeychain:
    """For tests and for anywhere without a login keychain."""

    def __init__(self, pattern: "re.Pattern[str]" = TOKEN_RE) -> None:
        self._secret: str | None = None
        self.pattern = pattern

    def get(self) -> str | None:
        return self._secret

    def set(self, secret: str) -> None:
        if not self.pattern.fullmatch(secret):
            raise DeviceAuthError("refusing to store a malformed token")
        self._secret = secret

    def delete(self) -> None:
        self._secret = None


@dataclass(frozen=True)
class PendingLogin:
    user_code: str
    verification_uri: str
    expires_in: int
    interval: int


class DeviceLogin:
    """One brain's link to one account."""

    def __init__(self, cloud_url: str, *, keychain=None, http=None,
                 now=_utcnow) -> None:
        self.cloud_url = normalize_cloud_url(cloud_url)
        self._keychain = (keychain if keychain is not None
                          else Keychain(account=_keychain_account(self.cloud_url)))
        # Per-origin accounts landed 2026-07-21 with no fallback, which silently
        # signed out every machine linked BEFORE it: the refresh token sat in the
        # keychain under the bare legacy account while every lookup used the new
        # hashed one, and "not signed in" told the user to link again for no
        # reason (2026-08-08). Migrate the legacy entry forward once, on first
        # touch; the legacy copy is left in place (harmless, and deleting it from
        # a constructor would be a surprising side effect).
        if keychain is None and self._keychain.get() is None:
            legacy = Keychain(account=KEYCHAIN_ACCOUNT).get()
            if legacy is not None:
                self._keychain.set(legacy)
        self._http = http if http is not None else httpx.Client(timeout=20.0)
        self._now = now
        self._device_code: str | None = None
        self._poll_net_failures = 0   # consecutive network errors mid-poll (≤3 = "pending")
        self._access_token: str | None = None
        self._expires_at: datetime | None = None
        self._interval_s = 5
        self._next_poll_at: datetime | None = None
        self._code_expires_at: datetime | None = None

    # ---- linking -------------------------------------------------------

    def begin(self) -> PendingLogin:
        """Ask for a code. The DAW shows what this returns."""
        try:
            response = self._http.post(f"{self.cloud_url}/v1/auth/device/code")
        except httpx.HTTPError as exc:
            # A raw ConnectError fell through the generic dispatch catch and reached the
            # DAW as "The engineer did not answer." (2026-08-09 §1 audit) — name the cause.
            raise DeviceAuthError(
                "can't reach the sign-in service — check your connection and try again"
            ) from exc
        if response.status_code == 429:
            raise DeviceAuthError(
                "the sign-in service is busy — wait a minute and try again")
        if response.status_code != 200:
            raise DeviceAuthError(
                f"the service would not start a login (HTTP {response.status_code})")
        body = response.json()
        self._device_code = body["device_code"]
        self._interval_s = max(1, int(body.get("interval", 5)))
        self._next_poll_at = None
        # The code's own life, kept here: "the service is busy" is waiting (below), and
        # waiting has to end somewhere even if the service never says `expired_token`.
        self._code_expires_at = self._now() + timedelta(seconds=int(body["expires_in"]))
        return PendingLogin(
            user_code=body["user_code"],
            verification_uri=body["verification_uri"],
            expires_in=int(body["expires_in"]),
            interval=int(body["interval"]),
        )

    def poll_once(self) -> str:
        """Returns 'pending' or 'linked'. Safe to call as fast as the UI likes.

        The protocol's pacing lives here, not in the caller. A UI polling every
        frame would otherwise earn `slow_down` on every request and never link
        at all — so calls made before the interval elapses are answered from
        memory without touching the network.

        RFC 8628 errors are states, not failures: only a genuinely dead code
        (denied/expired) raises, because that is the one the user must act on.
        """
        if not self._device_code:
            raise DeviceAuthError("no login in progress")
        now = self._now()
        if self._code_expires_at is not None and now >= self._code_expires_at:
            self._device_code = None
            raise DeviceAuthError("that code expired before it was approved")
        if self._next_poll_at is not None and now < self._next_poll_at:
            return "pending"
        self._next_poll_at = now + timedelta(seconds=self._interval_s)
        try:
            response = self._http.post(
                f"{self.cloud_url}/v1/auth/device/token",
                json={"device_code": self._device_code})
        except httpx.HTTPError as exc:
            # One blip mid-poll is "pending" (wifi hiccups happen); a RUN of them means the
            # service is genuinely unreachable — before this, a network failure looked like
            # "pending" forever, silently, for the code's whole 10-minute life (2026-08-09
            # §1 audit).
            self._poll_net_failures += 1
            if self._poll_net_failures <= 3:
                return "pending"
            self._device_code = None
            raise DeviceAuthError(
                "lost the connection to the sign-in service — check your network and "
                "start the sign-in again") from exc
        self._poll_net_failures = 0
        if response.status_code == 429 or response.status_code >= 500:
            # The service is BUSY (its rate limit, a deploy rolling) - which says nothing
            # about the code. It used to end the sign-in with "unexpected reply while
            # linking: 429", and the DAW treats that as final: a user who had already
            # approved the code on the site was told to start again (full audit
            # 2026-10-02, reader S5·R4). Wait longer, like `slow_down`, until the code's
            # own expiry.
            self._interval_s = min(self._interval_s + 5, 60)
            self._next_poll_at = self._now() + timedelta(seconds=self._interval_s)
            return "pending"
        try:
            body = response.json() if response.content else {}
        except ValueError:
            body = {}
        if not isinstance(body, dict):
            body = {}
        if response.status_code == 200 and body.get("access_token"):
            # A RE-LINK replaces a session this machine already holds (reader S8·R8): the old
            # refresh token was written over and its server session lived on for ninety days,
            # with nothing here able to end it. Read before the new one is stored, revoked
            # after - best effort, as a sign-out's revoke is: linking must work in an outage.
            try:
                replaced = self._keychain.get()
            except Exception:  # noqa: BLE001 - a keychain that will not answer holds nothing to revoke
                replaced = None
            # THE CODE IS SPENT BY NOW, WHATEVER THE KEYCHAIN SAYS (pre-merge review,
            # 2026-10-07). The service claimed it to answer this poll, so it can never be
            # presented again: a second poll of it ends the session it made and reads "the
            # reply did not reach this computer". And a session whose refresh token is stored
            # nowhere is one nobody can use or end. So the token is stored first; if the
            # keychain will not take it (locked, its prompt unanswered for `TIMEOUT_S`), the
            # session is ended here, nothing of it is kept in memory, and the keychain's own
            # refusal is what the user is told - a link this machine already had is untouched.
            self._device_code = None
            try:
                self._keychain.set(body["refresh_token"])
            except DeviceAuthError:
                try:
                    self._http.post(f"{self.cloud_url}/v1/auth/device/revoke",
                                    json={"refresh_token": body["refresh_token"]})
                except Exception:  # noqa: BLE001 - best effort, as a sign-out's revoke is
                    pass
                raise
            self._remember(body)
            if replaced and replaced != body["refresh_token"]:
                try:
                    self._http.post(f"{self.cloud_url}/v1/auth/device/revoke",
                                    json={"refresh_token": replaced})
                except Exception:  # noqa: BLE001
                    pass
            return "linked"
        error = body.get("error", "")
        if error == "authorization_pending":
            return "pending"
        if error == "slow_down":
            # RFC 8628: back off by five seconds and keep waiting. To the caller
            # this is indistinguishable from pending, which is the truth of it.
            self._interval_s += 5
            self._next_poll_at = self._now() + timedelta(seconds=self._interval_s)
            return "pending"
        if error == "expired_token":
            self._device_code = None
            if body.get("reason") == CLAIMED_REASON:
                # The code WAS approved and exchanged - and the reply that carried the
                # tokens never arrived here (the poll before this one ended in a network
                # error, which reads as "pending"). The service has ended the session
                # that exchange made; saying the code "expired before it was approved"
                # to someone who had just approved it was false (full audit 2026-10-02,
                # reader S5·R9).
                raise DeviceAuthError(MSG_CLAIM_LOST)
            raise DeviceAuthError("that code expired before it was approved")
        if error == "access_denied":
            self._device_code = None
            raise DeviceAuthError("that login was refused")
        raise DeviceAuthError(f"unexpected reply while linking: {error or response.status_code}")

    # ---- using ---------------------------------------------------------

    def signed_in(self) -> bool:
        return self._keychain.get() is not None

    def account_id(self) -> str | None:
        """The account this link last held a token for (`token_subject` of the access token in
        hand, stale or not), or None. For telling one account's licence answer from another's -
        never for trusting anything: the brain cannot verify the token, only the service can."""
        return token_subject(self._access_token)

    def access_token(self) -> str | None:
        """A token good for the next call, refreshed if the old one is stale.

        Returns None when nobody has linked this machine — the caller decides
        whether that is fatal, because the env-var path is still valid for
        evaluation runs and CI.
        """
        if (self._access_token and self._expires_at
                and self._now() < self._expires_at):
            return self._access_token
        refresh_token = self._keychain.get()
        if not refresh_token:
            return None
        response = self._http.post(
            f"{self.cloud_url}/v1/auth/device/refresh",
            json={"refresh_token": refresh_token})
        if response.status_code in {401, 403}:
            # Revoked, disabled, or the account is gone. Drop it: holding a
            # dead token only produces the same failure on every later call.
            try:
                self._keychain.delete()
            except DeviceAuthError:
                pass    # a keychain that will not answer: the token is dead either way
            self._access_token = None
            self._expires_at = None
            return None
        if response.status_code != 200:
            raise DeviceAuthError(
                f"could not refresh the session (HTTP {response.status_code})")
        self._remember(response.json())
        return self._access_token

    def sign_out(self) -> None:
        refresh_token = self._keychain.get()
        if refresh_token:
            try:
                self._http.post(
                    f"{self.cloud_url}/v1/auth/device/revoke",
                    json={"refresh_token": refresh_token})
            except Exception:
                # Signing out locally must remain possible during an outage.
                pass
        try:
            # May raise DeviceAuthError (a keychain that will not answer): the caller says
            # so - the token is still stored, and "signed out" would not be true.
            self._keychain.delete()
        finally:
            self._access_token = None
            self._expires_at = None
            self._device_code = None

    def _remember(self, body: dict) -> None:
        self._access_token = body["access_token"]
        self._expires_at = self._now() + timedelta(
            seconds=max(1, int(body.get("expires_in", 3600)) - EXPIRY_SKEW_S))


def token_subject(token) -> str | None:
    """The `sub` claim of a JWT, read WITHOUT checking its signature - or None for anything that
    is not a JWT with a string `sub`."""
    if not isinstance(token, str) or token.count(".") != 2:
        return None
    try:
        part = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
    except (ValueError, TypeError):
        return None
    sub = claims.get("sub") if isinstance(claims, dict) else None
    return sub if isinstance(sub, str) and sub else None


_shared: DeviceLogin | None = None


def shared_login(cloud_url: str | None = None) -> DeviceLogin:
    """The brain's one login, so a refresh is shared rather than repeated."""
    global _shared
    raw_url = (cloud_url or os.environ.get(CLOUD_URL_ENV, "")).strip()
    if not raw_url:
        raise DeviceAuthError(f"{CLOUD_URL_ENV} is not set")
    url = normalize_cloud_url(raw_url)
    if _shared is None or _shared.cloud_url != url:
        _shared = DeviceLogin(url)
    return _shared


def reset_shared() -> None:
    global _shared
    _shared = None
