"""The public keys the brain checks a licence answer against (2026-10-20).

`GET /v1/license/me` signs its answer with Ed25519 (`backend/app/license_signing.py`); the
brain believes an answer - live, or from its disk cache - only when one of these keys verifies
it (`license_state.verify`). Public keys: they are meant to be read, and they go into the
published brain source like everything else here.

`z1`'s private half is the control plane's `LICENSE_SIGNING_KEY` secret; the operator's copy is
in the login keychain of the machine it was made on (service `zammoth-license-signing-key`),
pushed to Fly by `backend/scripts/push_license_signing_key.sh`. To rotate: add the new key here
under a new kid, ship a build, then switch the secret - an old kid is removed only once no
answer signed with it can still be inside a cache's offline grace.
"""
from __future__ import annotations

#: kid -> base64url (no padding) of the 32-byte Ed25519 public key.
TRUSTED_LICENSE_KEYS: dict[str, str] = {
    "z1": "xzboamV8LImJHhNiJDeqHLDxMrrhihgH4Rd0QzydAjk",
}
