"""The licence answer is signed, and the brain believes nothing else (2026-10-20).

James: "sign the licence cache (Ed25519 from the backend, public key in the app)". Before it,
two doors gave the engineer away with no patch to any code: a `license.json` written by hand
(`{"status": "licensed"}`, dated inside the grace), and a licence server of one's own
(`MIX_AGENT_CLOUD_URL` is honoured in a shipped build) answering `licensed`. Each case here
was RED on the brain before the signature.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

import licence_signing as SIG
import test_api_key_license as LIC
from orchestrator import api_key, license_state


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    # the licence tests' own isolation (a memory keychain, a cloud URL, a fresh login)
    from orchestrator import device_auth
    for name in ("ANTHROPIC_API_KEY", "MIX_AGENT_LLM_BACKEND", "MIX_AGENT_LICENSE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MIX_AGENT_DATA_DIR", str(tmp_path / "appdata"))
    monkeypatch.setenv("MIX_AGENT_CLOUD_URL", "https://meter.example.test")
    api_key.use_store(api_key.memory_store())
    license_state.reset()
    device_auth.reset_shared()
    device_auth._shared = device_auth.DeviceLogin(
        "https://meter.example.test", keychain=device_auth.MemoryKeychain())
    api_key.set_api_key(LIC.GOOD, client_factory=lambda k: LIC._Client(None))
    assert license_state.gate_applies() is True        # a shipped-shaped brain: the gate applies
    yield
    api_key.use_store(None)
    license_state.reset()
    device_auth.reset_shared()


def _cache_file(obj) -> Path:
    path = Path(license_state._cache_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))
    license_state.reset()
    return path


class _Server:
    """A licence server that answers whatever it is told - the user's own, or ours."""

    def __init__(self, answer):
        self.answer, self.calls = answer, 0

    def get(self, url, **kw):
        self.calls += 1
        return self

    status_code = 200

    def json(self):
        # an answer built at the request (a callable) can answer the request; one built before
        # it is a replay (the pre-merge review of 2026-10-20: the brain's nonce is signed in)
        return self.answer() if callable(self.answer) else self.answer


def test_a_hand_written_cache_grants_nothing(monkeypatch):
    """Offline, signed in: a `licensed` file dated an hour ago ran the engineer."""
    LIC._sign_in()
    import httpx

    def offline(*a, **k):
        raise ConnectionError("offline")
    monkeypatch.setattr(httpx, "get", offline)
    _cache_file({"status": "licensed", "checked_at": time.time() - 3600})
    st = license_state.state()
    assert st["status"] == "unknown", st
    assert license_state.refusal() == license_state.MSG_UNREACHABLE
    # signed out (a "lapsed" session): the same file is no licence either
    from orchestrator import device_auth
    device_auth._shared._keychain.delete()
    license_state.reset()
    assert license_state.state()["status"] == "none"
    # the control: the same answer, SIGNED, is the offline grace it always was
    _cache_file(SIG.cache({"status": "licensed", "checked_at": time.time() - 3600}))
    st = license_state.state()
    assert st["status"] == "licensed" and st["message"] == license_state.MSG_RESIGN_IN


def test_a_server_of_ones_own_grants_nothing(monkeypatch):
    """An unsigned `licensed` from whatever answers MIX_AGENT_CLOUD_URL is refused, said as
    such, and never written to the cache."""
    LIC._sign_in()
    import httpx
    fake = _Server({"status": "licensed", "purchased_at": "2026-10-20T00:00:00+00:00"})
    monkeypatch.setattr(httpx, "get", fake.get)
    st = license_state.state()
    assert fake.calls == 1
    assert st["status"] == "unknown" and st["message"] == license_state.MSG_UNVERIFIED, st
    assert license_state.refusal() == license_state.MSG_UNVERIFIED
    assert not Path(license_state._cache_path()).exists(), "an unsigned answer was cached"
    # a forged signature, and a real one over another answer, are the same nothing
    real = SIG.sign({"status": "expired"})
    body, sig = real.split(".")
    forged = SIG.b64url(json.dumps({"v": 1, "kid": SIG.KID, "sub": "u", "iat": time.time(),
                                    "status": "licensed"}).encode())
    flipped = sig[:20] + ("B" if sig[20] != "B" else "C") + sig[21:]   # one byte of the signature
    for token in (f"{forged}.{sig}", f"{body}.{flipped}", "x.y", None, 7):
        fake.answer = {"status": "licensed", "signed": token}
        assert license_state.state(force=True)["status"] == "unknown", token
    # a key the brain does not trust
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    fake.answer = {"signed": SIG.sign({"status": "licensed"}, key=Ed25519PrivateKey.generate())}
    assert license_state.state(force=True)["status"] == "unknown"
    # a real answer, signed before this request was made, answers another request
    fake.answer = SIG.reply({"status": "licensed"})
    assert license_state.state(force=True)["status"] == "unknown"
    # the control: the service's own signature over THIS request is believed, and cached
    fake.answer = lambda: SIG.reply({"status": "licensed"})
    assert license_state.state(force=True)["status"] == "licensed"
    assert json.loads(Path(license_state._cache_path()).read_text())["signed"]


def test_an_unsigned_reply_falls_back_to_the_signed_cache(monkeypatch):
    """A redirect while offline-capable: the last SIGNED answer stands for the grace."""
    LIC._sign_in()
    _cache_file(SIG.cache({"status": "licensed", "checked_at": time.time() - 86400}))
    import httpx
    monkeypatch.setattr(httpx, "get", _Server({"status": "expired"}).get)
    st = license_state.state()
    assert st["status"] == "licensed" and st["offline"] is True


def test_the_age_is_the_services_clock_not_the_files(monkeypatch):
    """`checked_at` in the file is the brain's own note: re-dating it buys nothing."""
    LIC._sign_in()
    import httpx

    def offline(*a, **k):
        raise ConnectionError("offline")
    monkeypatch.setattr(httpx, "get", offline)
    old = SIG.cache({"status": "licensed",
                     "checked_at": time.time() - (license_state.OFFLINE_GRACE_DAYS + 1) * 86400})
    old["checked_at"] = time.time()                       # the file says "just now"
    _cache_file(old)
    assert license_state.state()["status"] == "unknown"
    # a signature dated in the future is no cache (the clock set back past the check)
    _cache_file(SIG.cache({"status": "licensed", "checked_at": time.time() + 3 * 3600}))
    assert license_state.state()["status"] == "unknown"


def test_a_shipped_brain_trusts_only_its_own_keys(monkeypatch):
    """`MIX_AGENT_LICENSE_KEYS` is a source checkout's: frozen, a `launchctl setenv` of a key
    of one's own (plus a server of one's own) would be a licence."""
    from orchestrator.license_keys import TRUSTED_LICENSE_KEYS
    assert SIG.KID in license_state.trusted_keys()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    keys = license_state.trusted_keys()
    assert SIG.KID not in keys
    assert set(keys) == set(TRUSTED_LICENSE_KEYS) and all(len(k) == 32 for k in keys.values())
    assert license_state.verify(SIG.sign({"status": "licensed"})) is None


def test_the_control_planes_token_verifies_in_the_brain(monkeypatch):
    """One format at both ends: the backend's own signer, checked by the brain's verifier."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from datetime import datetime, timezone

    from backend.app import license_signing as BK
    key = Ed25519PrivateKey.generate()
    seed = key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                             serialization.NoEncryption())
    pub = key.public_key().public_bytes(serialization.Encoding.Raw,
                                        serialization.PublicFormat.Raw)
    monkeypatch.setenv(license_state.KEYS_ENV, f"k9:{BK.b64url(pub)}")
    ends = datetime(2999, 1, 1, tzinfo=timezone.utc)
    token = BK.sign_license({"status": "trial", "trial_ends_at": ends, "trial_days": 3,
                             "purchased_at": None, "refunded": False},
                            "user-7", f"k9:{BK.b64url(seed)}")
    body = license_state.verify(token)
    assert body is not None
    assert body["status"] == "trial" and body["sub"] == "user-7" and body["kid"] == "k9"
    assert license_state._derive(body, offline=False)["status"] == "trial"
    assert tuple(BK.SIGNED_FIELDS) == SIG.FIELDS            # the test signer is the backend's shape


def test_the_shipped_public_key_is_a_real_key():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from orchestrator.license_keys import TRUSTED_LICENSE_KEYS
    assert TRUSTED_LICENSE_KEYS, "no trusted key: every shipped brain would refuse the engineer"
    for text in TRUSTED_LICENSE_KEYS.values():
        Ed25519PublicKey.from_public_bytes(license_state._b64url_decode(text))
