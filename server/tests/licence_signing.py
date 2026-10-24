"""A test signing key for licence answers (2026-10-20).

The brain believes a licence answer only when it verifies (`license_state.verify`). conftest
trusts THIS key on every test (`MIX_AGENT_LICENSE_KEYS`), and a test that stages an answer - a
fake reply, a cache file - signs it here, the way the control plane does
(`backend/app/license_signing.py`; `test_licence_signature.py` holds the two to one format).
"""
from __future__ import annotations

import base64
import json
import time

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

KID = "test"
_KEY = Ed25519PrivateKey.generate()
FIELDS = ("status", "trial_ends_at", "trial_days", "purchased_at", "refunded")


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


PUBLIC = b64url(_KEY.public_key().public_bytes(serialization.Encoding.Raw,
                                               serialization.PublicFormat.Raw))
KEYS_ENV_VALUE = f"{KID}:{PUBLIC}"


def sign(fields: dict, *, iat=None, kid: str = KID, key: Ed25519PrivateKey | None = None,
         sub: str = "user-1", nonce: str | None = None) -> str:
    body = {"v": 1, "kid": kid, "sub": sub, "iat": time.time() if iat is None else iat}
    body.update({name: fields.get(name) for name in FIELDS})
    if nonce is not None:
        body["nonce"] = nonce
    body_text = b64url(json.dumps(body, sort_keys=True, separators=(",", ":")).encode())
    return f"{body_text}.{b64url((key or _KEY).sign(body_text.encode('ascii')))}"


def reply(payload: dict, nonce: str | None = None) -> dict:
    """What `/v1/license/me` answers: the plain fields, and the same answer signed - for the
    request the brain is making now (its nonce, read where the brain keeps it while it waits)
    unless the caller names another. Built BEFORE the brain asks, it answers no request."""
    if nonce is None:
        from orchestrator import license_state
        nonce = getattr(license_state, "_awaiting_nonce", None)
    return {**payload, "signed": sign(payload, nonce=nonce)}


def cache(payload: dict) -> dict:
    """A `license.json` holding `payload`, checked at its `checked_at` (now when absent): the
    answer signed at that moment, as the service would have signed it then."""
    checked = payload.get("checked_at", time.time())
    return {"signed": sign(payload, iat=checked), "checked_at": checked}
