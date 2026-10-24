"""Reader P3s, batch B2 - the half that ships in the BRAIN (full audit 2026-10-02).

S5·R9: the poll that claims the tokens is not repeatable. When the reply carrying them
is lost, the brain polls again (a network error is "pending"), and the service - which
has already spent the code - used to answer a bare `expired_token`: a user who HAD
approved their code was told "that code expired before it was approved". The service
now says which it is (`reason: already_claimed`) and ends the session that claim made;
the brain says what happened and what to do.

No audio fixtures: like the B1 client file, this runs under `--noconftest` too.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pytest

from orchestrator.device_auth import DeviceAuthError, DeviceLogin, MemoryKeychain


NOW = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)

CODE = {"device_code": "dev-secret", "user_code": "T5G3-SS9Q",
        "verification_uri": "https://zammoth.test/link", "expires_in": 600,
        "interval": 5}


class Reply:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.content = b"{}" if payload is not None else b""

    def json(self):
        return self._payload


class Wire:
    """Answers the token endpoint from a queue; an exception in the queue is raised."""

    def __init__(self, *token_answers):
        self.token_answers = list(token_answers)

    def post(self, url, json=None, **_kwargs):
        if url.endswith("/v1/auth/device/code"):
            return Reply(200, CODE)
        answer = self.token_answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


def _poll_to_the_end(wire) -> tuple[list[str], str | None]:
    holder = {"now": NOW}
    keychain = MemoryKeychain()
    login = DeviceLogin("https://api.test", keychain=keychain, http=wire,
                        now=lambda: holder["now"])
    login.begin()
    seen: list[str] = []
    while wire.token_answers:
        try:
            seen.append(login.poll_once())
        except DeviceAuthError as exc:
            seen.append(f"ERROR: {exc}")
            break
        holder["now"] += timedelta(seconds=login._interval_s + 1)
    return seen, keychain.get()


def test_a_claim_whose_reply_was_lost_is_not_called_an_unapproved_code():
    seen, token = _poll_to_the_end(Wire(
        httpx.ReadError("the reply never arrived"),
        Reply(400, {"error": "expired_token", "reason": "already_claimed"})))
    assert seen[0] == "pending" and seen[1].startswith("ERROR: ")
    assert "before it was approved" not in seen[1]
    assert "start" in seen[1] and "again" in seen[1]
    assert token is None


def test_a_code_that_really_expired_still_says_so():
    seen, _token = _poll_to_the_end(Wire(
        Reply(400, {"error": "authorization_pending"}),
        Reply(400, {"error": "expired_token"})))
    assert seen == ["pending", "ERROR: that code expired before it was approved"]


@pytest.mark.parametrize("reason", [None, 7, "", "something_else"])
def test_an_unknown_reason_changes_nothing(reason):
    seen, _token = _poll_to_the_end(Wire(
        Reply(400, {"error": "expired_token", "reason": reason})))
    assert seen == ["ERROR: that code expired before it was approved"]
