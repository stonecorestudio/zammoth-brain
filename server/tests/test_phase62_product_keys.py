"""Phase 62: the brain's half of a product key, and the trial's length.

`license_state.redeem` sends a typed key to the control plane as the signed-in device and
answers the licence it now holds, or the service's own sentence; the brain's NDJSON door has
a `redeem_product_key` method that leaves an unwritten folder alone and never logs its line.
"""
from __future__ import annotations

import inspect

import httpx
import pytest

from orchestrator import api_key, daw_server, license_state

from test_api_key_license import GOOD, _Client, _Http, _isolated, _sign_in  # noqa: F401  (the fixture)


class _Plane(_Http):
    """A stand-in control plane: POST /v1/license/redeem, then GET /v1/license/me."""
    def __init__(self, *, redeem_status=200, detail="", fail=False):
        super().__init__(payload={"status": "trial", "trial_days": 3,
                                  "trial_ends_at": "2999-01-01T00:00:00+00:00"})
        self.redeem_status, self.detail, self.fail_post = redeem_status, detail, fail
        self.posted: list[dict] = []

    def post(self, url, **kw):
        if self.fail_post:
            raise httpx.ConnectError("offline")
        assert url.endswith("/v1/license/redeem")
        assert kw["headers"]["Authorization"] == "Bearer tok"
        self.posted.append(kw["json"])
        plane = self

        class _Reply:
            status_code = plane.redeem_status

            def json(self):
                return {"detail": plane.detail} if plane.detail else {"status": "licensed"}
        if self.redeem_status == 200:
            self.payload = {"status": "licensed", "purchased_at": "2026-10-16T00:00:00+00:00"}
        return _Reply()


@pytest.fixture()
def plane(monkeypatch):
    p = _Plane()
    monkeypatch.setattr(httpx, "get", p.get)
    monkeypatch.setattr(httpx, "post", p.post)
    # a key in the keychain: what makes this a shipped install the licence gates
    api_key.set_api_key(GOOD, client_factory=lambda k: _Client(None))
    return p


def test_the_trial_is_said_as_three_days_and_read_from_the_service(plane):
    assert "3-day trial" in license_state.MSG_SIGN_IN
    assert "product key" in license_state.MSG_EXPIRED
    _sign_in()
    st = license_state.state(force=True)
    assert st["status"] == "trial" and st["trial_days"] == 3
    plane.payload = {"status": "trial", "trial_days": 7,
                     "trial_ends_at": "2999-01-01T00:00:00+00:00"}
    assert license_state.state(force=True)["trial_days"] == 7


def test_redeeming_signed_out_asks_for_a_sign_in_and_sends_nothing(plane):
    res = license_state.redeem("ZAM-AAAAA-AAAAA-AAAAA-AAAAA")
    assert res == {"ok": False, "error": license_state.MSG_REDEEM_SIGN_IN}
    assert plane.posted == []
    assert license_state.redeem("   ")["ok"] is False


def test_a_redeemed_key_comes_back_as_a_licence(plane):
    _sign_in()
    assert license_state.state(force=True)["status"] == "trial"
    res = license_state.redeem("  zam-aaaaa-aaaaa-aaaaa-aaaaa ")
    assert res["ok"] is True and res["license"]["status"] == "licensed"
    assert plane.posted == [{"key": "zam-aaaaa-aaaaa-aaaaa-aaaaa"}]
    assert license_state.refusal() is None


def test_the_services_refusal_is_the_sentence_the_app_shows(monkeypatch, plane):
    _sign_in()
    plane.redeem_status, plane.detail = 409, "This key has already been used by another account."
    res = license_state.redeem("ZAM-AAAAA-AAAAA-AAAAA-AAAAA")
    assert res == {"ok": False, "error": plane.detail}
    plane.detail = ""
    plane.redeem_status = 500
    assert "HTTP 500" in license_state.redeem("ZAM-AAAAA-AAAAA-AAAAA-AAAAA")["error"]
    plane.fail_post = True
    assert "can't reach the licence service" in license_state.redeem("x" * 20)["error"]


def test_the_brain_door_leaves_the_folder_alone_and_never_logs_the_line():
    assert "redeem_product_key" in daw_server._LEAVES_FOLDER_ALONE
    source = inspect.getsource(daw_server)
    assert 'secret = method in ("set_api_key", "auth", "redeem_product_key")' in source
    assert 'elif method == "redeem_product_key":' in source
