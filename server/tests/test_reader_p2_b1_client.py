"""Reader P2s, batch B1 - the two halves that ship in the BRAIN (full audit 2026-10-02).

S5·R4: a 429 from the sign-in service while a code is being linked ended the sign-in
("unexpected reply while linking: 429"), and the DAW treats that as final - a user who
had already approved their code on the site was told to start again.

S6·R6: the request-shape surface named its prompts by fixed labels, so a changed prompt
was registered under the label of the one it replaced.

No audio fixtures: this file runs under `--noconftest` in the backend's CI job.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orchestrator.device_auth import DeviceAuthError, DeviceLogin, MemoryKeychain
from orchestrator.llm_surface import _prefix


NOW = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)

CODE = {"device_code": "dev-secret", "user_code": "T5G3-SS9Q",
        "verification_uri": "https://zammoth.test/link", "expires_in": 600,
        "interval": 5}
TOKENS = {"access_token": "access-1", "expires_in": 3600,
          "refresh_token": "refresh-token-value-1"}


class Reply:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.content = b"{}" if payload is not None else b""

    def json(self):
        return self._payload


class Scripted:
    """Answers each endpoint from its own queue, in order."""

    def __init__(self, **by_path):
        self.by_path = {k: list(v) for k, v in by_path.items()}
        self.calls: list[str] = []

    def post(self, url, json=None, **kwargs):
        self.calls.append(url)
        for fragment, queue in self.by_path.items():
            if fragment in url:
                return queue.pop(0) if len(queue) > 1 else queue[0]
        raise AssertionError(f"unexpected POST to {url}")


def _login(http, start=NOW):
    holder = {"now": start}
    login = DeviceLogin("https://api.test", keychain=MemoryKeychain(), http=http,
                        now=lambda: holder["now"])
    return login, holder


def test_a_busy_service_while_linking_is_waiting_not_a_dead_sign_in():
    busy = Reply(429, {"detail": "too many device-token requests — try again shortly"})
    http = Scripted(code=[Reply(200, CODE)], token=[busy, busy, Reply(200, TOKENS)])
    login, holder = _login(http)
    login.begin()
    waits = []
    for _ in range(2):
        assert login.poll_once() == "pending"
        waits.append(login._interval_s)
        holder["now"] += timedelta(seconds=login._interval_s)
    # It backs off while it is being refused, as it does for `slow_down`.
    assert waits[0] > CODE["interval"] and waits[1] > waits[0]
    assert login.poll_once() == "linked"
    assert login.signed_in() and login.access_token() == "access-1"


def test_a_code_that_outlives_the_busy_spell_still_expires():
    busy = Reply(429, {"detail": "busy"})
    http = Scripted(code=[Reply(200, CODE)], token=[busy])
    login, holder = _login(http)
    login.begin()
    assert login.poll_once() == "pending"
    holder["now"] += timedelta(seconds=CODE["expires_in"] + 1)
    with pytest.raises(DeviceAuthError, match="expired"):
        login.poll_once()
    with pytest.raises(DeviceAuthError, match="no login in progress"):
        login.poll_once()


def test_a_busy_service_at_the_start_says_so():
    http = Scripted(code=[Reply(429, {"detail": "too many device login starts"})])
    login, _ = _login(http)
    with pytest.raises(DeviceAuthError, match="busy") as refused:
        login.begin()
    assert "429" not in str(refused.value)


def test_the_surface_names_each_prompt_by_its_content():
    book_1 = "The knowledge book, as build 1 shipped it. " * 20
    book_2 = "The knowledge book, rewritten for build 2. " * 20
    one = _prefix("knowledge-block", book_1, exact=False)
    two = _prefix("knowledge-block", book_2, exact=False)
    assert one["label"] != two["label"]
    assert one["label"] == f"knowledge-block-{one['sha256'][:8]}"
    assert _prefix("knowledge-block", book_1, exact=False) == one
    assert len(_prefix("knowledge-fallback", book_1, exact=False)["label"]) <= 64


def test_the_served_knowledge_book_is_the_brains_book():
    """Found on the way to B1's deploy: `backend/app/knowledge.json` - what
    GET /v1/knowledge serves every shipped brain - was last generated on 2026-08-08, one
    line behind the book the source brain reads. Nothing compared the two. Regenerate:

        python -m orchestrator.export_knowledge > backend/app/knowledge.json   (then deploy)
    """
    import json
    from pathlib import Path

    # The book's module imports the engine's models (numpy). The backend's CI job runs
    # this file without the engine installed; the engine job - which every change to the
    # book starts - has it, and runs this.
    pytest.importorskip("numpy", reason="the engine suite runs this one")
    from orchestrator.export_knowledge import payload

    served = json.loads((Path(__file__).resolve().parents[2]
                         / "backend" / "app" / "knowledge.json").read_text("utf-8"))
    assert served == payload(), "regenerate backend/app/knowledge.json and deploy it"
