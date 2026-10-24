"""The brain's half of device login: link once, then stay signed in.

The interesting behaviour is not the happy path — it is what happens an hour in,
when the access token the DAW was handed at launch has expired and the user is
in the middle of a mix.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orchestrator.device_auth import (DeviceAuthError, DeviceLogin,
                                      Keychain, MemoryKeychain, _keychain_account,
                                      normalize_cloud_url)


NOW = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.content = b"{}" if payload is not None else b""

    def json(self):
        return self._payload


class FakeHttp:
    """Answers the three control-plane endpoints from a queued script."""

    def __init__(self, **by_path):
        self.by_path = {k: list(v) for k, v in by_path.items()}
        self.calls: list[tuple[str, dict]] = []

    def post(self, url, json=None, **kwargs):
        self.calls.append((url, json or {}))
        for fragment, queue in self.by_path.items():
            if fragment in url:
                return queue.pop(0) if len(queue) > 1 else queue[0]
        raise AssertionError(f"unexpected POST to {url}")


def _clock(start=NOW):
    holder = {"now": start}
    return holder, lambda: holder["now"]


def test_link_shows_a_code_then_stores_only_the_refresh_token():
    keychain = MemoryKeychain()
    holder, clock = _clock()
    http = FakeHttp(
        code=[FakeResponse(200, {
            "device_code": "dev-secret", "user_code": "T5G3-SS9Q",
            "verification_uri": "https://zammoth.test/link",
            "expires_in": 600, "interval": 5})],
        token=[FakeResponse(400, {"error": "authorization_pending"}),
               FakeResponse(200, {"access_token": "access-1", "expires_in": 3600,
                                  "refresh_token": "refresh-token-value-1"})],
    )
    login = DeviceLogin("https://api.test", keychain=keychain, http=http, now=clock)

    pending = login.begin()
    assert pending.user_code == "T5G3-SS9Q"
    assert pending.verification_uri == "https://zammoth.test/link"
    assert pending.interval == 5

    assert login.poll_once() == "pending"
    assert keychain.get() is None, "nothing stored until the user approves"

    holder["now"] = NOW + timedelta(seconds=5)  # the user approves in the meantime
    assert login.poll_once() == "linked"
    # Only the durable half is persisted; the access token stays in memory.
    assert keychain.get() == "refresh-token-value-1"
    assert login.access_token() == "access-1"
    assert login.signed_in()


def test_an_expired_access_token_is_refreshed_mid_session():
    keychain = MemoryKeychain()
    keychain.set("refresh-token-value-1")
    holder, clock = _clock()
    http = FakeHttp(refresh=[FakeResponse(200, {"access_token": "access-2",
                                                "expires_in": 3600})])
    login = DeviceLogin("https://api.test", keychain=keychain, http=http, now=clock)

    assert login.access_token() == "access-2"
    assert len(http.calls) == 1

    # Still inside the hour: served from memory, no second round trip.
    holder["now"] = NOW + timedelta(minutes=30)
    assert login.access_token() == "access-2"
    assert len(http.calls) == 1

    # Past it: this is the moment a naive read-once design starts 401ing.
    holder["now"] = NOW + timedelta(hours=2)
    http.by_path["refresh"] = [FakeResponse(200, {"access_token": "access-3",
                                                  "expires_in": 3600})]
    assert login.access_token() == "access-3"
    assert len(http.calls) == 2


def test_the_token_is_renewed_before_it_actually_expires():
    # Renewing exactly on expiry races the request it was fetched for.
    keychain = MemoryKeychain()
    keychain.set("refresh-token-value-1")
    holder, clock = _clock()
    http = FakeHttp(refresh=[FakeResponse(200, {"access_token": "access-2",
                                                "expires_in": 3600})])
    login = DeviceLogin("https://api.test", keychain=keychain, http=http, now=clock)
    assert login.access_token() == "access-2"

    holder["now"] = NOW + timedelta(seconds=3600 - 30)  # 30s of life left
    http.by_path["refresh"] = [FakeResponse(200, {"access_token": "access-3",
                                                  "expires_in": 3600})]
    assert login.access_token() == "access-3", "should renew inside the skew"


def test_a_revoked_session_signs_the_machine_out_rather_than_looping():
    keychain = MemoryKeychain()
    keychain.set("refresh-token-value-1")
    http = FakeHttp(refresh=[FakeResponse(401, {"detail": "invalid refresh token"})])
    login = DeviceLogin("https://api.test", keychain=keychain, http=http, now=lambda: NOW)

    assert login.access_token() is None
    # Keeping a dead token only produces a confusing 401 on every later call.
    assert keychain.get() is None
    assert not login.signed_in()


def test_a_disabled_session_is_also_removed_locally():
    keychain = MemoryKeychain()
    keychain.set("refresh-token-value-1")
    http = FakeHttp(refresh=[FakeResponse(403, {"detail": "account disabled"})])
    login = DeviceLogin("https://api.test", keychain=keychain, http=http,
                        now=lambda: NOW)

    assert login.access_token() is None
    assert keychain.get() is None


def test_a_machine_that_was_never_linked_reports_it_plainly():
    login = DeviceLogin("https://api.test", keychain=MemoryKeychain(),
                        http=FakeHttp(), now=lambda: NOW)
    assert login.access_token() is None
    assert not login.signed_in()


def test_a_ui_polling_every_frame_cannot_trip_the_rate_limit():
    """The bug a mocked backend hides.

    The service answers `slow_down` to anything faster than its interval. A UI
    timer polling on every repaint would earn that forever and never link, so
    the pacing has to live in the client: extra calls are answered from memory
    without touching the network.
    """
    holder, clock = _clock()
    http = FakeHttp(
        code=[FakeResponse(200, {"device_code": "d", "user_code": "A",
                                 "verification_uri": "u", "expires_in": 600,
                                 "interval": 5})],
        token=[FakeResponse(400, {"error": "authorization_pending"})],
    )
    login = DeviceLogin("https://api.test", keychain=MemoryKeychain(), http=http,
                        now=clock)
    login.begin()

    assert login.poll_once() == "pending"
    network_calls = len(http.calls)
    for _ in range(50):                       # a UI hammering it
        assert login.poll_once() == "pending"
    assert len(http.calls) == network_calls, "the interval was not honoured"

    holder["now"] = NOW + timedelta(seconds=5)
    assert login.poll_once() == "pending"
    assert len(http.calls) == network_calls + 1


def test_slow_down_backs_off_and_still_reads_as_waiting():
    holder, clock = _clock()
    http = FakeHttp(
        code=[FakeResponse(200, {"device_code": "d", "user_code": "A",
                                 "verification_uri": "u", "expires_in": 600,
                                 "interval": 5})],
        token=[FakeResponse(400, {"error": "slow_down"})],
    )
    login = DeviceLogin("https://api.test", keychain=MemoryKeychain(), http=http,
                        now=clock)
    login.begin()
    # Nothing for the DAW to render differently: it is still just waiting.
    assert login.poll_once() == "pending"

    # RFC 8628 says add five seconds, so the old interval must no longer be enough.
    holder["now"] = NOW + timedelta(seconds=5)
    calls = len(http.calls)
    assert login.poll_once() == "pending"
    assert len(http.calls) == calls, "backed-off interval was ignored"
    holder["now"] = NOW + timedelta(seconds=10)
    assert login.poll_once() == "pending"
    assert len(http.calls) == calls + 1


def test_a_dead_code_is_an_error_the_user_must_act_on():
    holder, clock = _clock()
    http = FakeHttp(
        code=[FakeResponse(200, {"device_code": "d", "user_code": "A",
                                 "verification_uri": "u", "expires_in": 600,
                                 "interval": 5})],
        token=[FakeResponse(400, {"error": "expired_token"})],
    )
    login = DeviceLogin("https://api.test", keychain=MemoryKeychain(), http=http,
                        now=clock)
    login.begin()
    with pytest.raises(DeviceAuthError, match="expired"):
        login.poll_once()
    # The dead code is dropped, so a stray later poll cannot resurrect it.
    with pytest.raises(DeviceAuthError, match="no login in progress"):
        login.poll_once()


def test_sign_out_forgets_everything():
    keychain = MemoryKeychain()
    keychain.set("refresh-token-value-1")
    http = FakeHttp(refresh=[FakeResponse(200, {"access_token": "a", "expires_in": 3600})])
    login = DeviceLogin("https://api.test", keychain=keychain, http=http, now=lambda: NOW)
    assert login.access_token() == "a"

    login.sign_out()
    assert keychain.get() is None
    assert not login.signed_in()
    assert login.access_token() is None


def test_sign_out_revokes_the_server_session_before_forgetting_it():
    keychain = MemoryKeychain()
    keychain.set("refresh-token-value-1")
    http = FakeHttp(revoke=[FakeResponse(200, {"revoked": True})])
    login = DeviceLogin("https://api.test", keychain=keychain, http=http,
                        now=lambda: NOW)
    login.sign_out()
    assert http.calls == [
        ("https://api.test/v1/auth/device/revoke",
         {"refresh_token": "refresh-token-value-1"})]
    assert keychain.get() is None


def test_cloud_urls_are_safe_origins_and_keychain_tokens_are_origin_bound():
    assert normalize_cloud_url("https://API.TEST:443/") == "https://api.test"
    assert _keychain_account("https://api.one") != _keychain_account("https://api.two")
    for unsafe in (
        "http://api.test",
        "https://trusted.test@evil.test",
        "https://api.test/proxy",
        "https://api.test/?next=https://evil.test",
    ):
        with pytest.raises(DeviceAuthError):
            normalize_cloud_url(unsafe)


def test_only_a_token_we_could_have_minted_reaches_the_keychain():
    # The keychain write builds a `security` command; a value with spaces in it
    # is not ours and must not be pasted into one.
    keychain = MemoryKeychain()
    with pytest.raises(DeviceAuthError, match="malformed"):
        keychain.set("not a token; rm -rf /")
    with pytest.raises(DeviceAuthError, match="malformed"):
        keychain.set("")
    keychain.set("Legitimate_urlsafe-token-0123456789")


def test_keychain_failures_never_echo_the_refresh_token(monkeypatch):
    token = "refresh-token-value-that-must-stay-secret"

    class Failed:
        returncode = 1
        stderr = f"rejected command containing {token}"

    monkeypatch.setattr("orchestrator.device_auth.subprocess.run",
                        lambda *args, **kwargs: Failed())
    with pytest.raises(DeviceAuthError) as failure:
        Keychain().set(token)
    assert token not in str(failure.value)


def test_network_failures_name_the_cause_instead_of_pending_forever():
    """begin() offline names the unreachable service; a mid-poll blip reads as
    'pending' (wifi hiccups happen) but a RUN of them raises — before this a
    network failure looked like 'pending', silently, for the code's whole
    10-minute life (2026-08-09 §1 audit)."""
    import httpx

    class DeadHttp:
        def post(self, url, json=None, **kwargs):
            raise httpx.ConnectError("no route to host")

    holder, clock = _clock()
    login = DeviceLogin("https://cloud.test", keychain=MemoryKeychain(),
                        http=DeadHttp(), now=clock)
    with pytest.raises(DeviceAuthError, match="can't reach the sign-in service"):
        login.begin()

    class FlakyHttp:
        """Answers the code request, then fails every token poll with a network error."""
        def post(self, url, json=None, **kwargs):
            if url.endswith("/device/code"):
                return FakeResponse(200, {
                    "device_code": "dev", "user_code": "AAAA-BBBB",
                    "verification_uri": "https://zammoth.test/link",
                    "expires_in": 600, "interval": 5})
            raise httpx.ConnectError("blip")

    holder, clock = _clock()
    login = DeviceLogin("https://cloud.test", keychain=MemoryKeychain(),
                        http=FlakyHttp(), now=clock)
    login.begin()
    for _ in range(3):                              # ≤3 consecutive failures = pending
        assert login.poll_once() == "pending"
        holder["now"] += timedelta(seconds=6)       # step past the poll interval
    with pytest.raises(DeviceAuthError, match="lost the connection"):
        login.poll_once()
