"""Phase 53: the user's own Anthropic key, and the licence that gates the agent.

The brain owns the key the way it owns the refresh token: validated with one free call,
kept in the keychain, handed to the SDK client as `api_key=` and never written into the
environment or a log. The licence state comes from the control plane, is cached for an
offline grace, and is consulted only on a shipped-shaped transport — never on a dev
shell's env key. These run with no network and no keychain.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from mix_engine.session import Session
import licence_signing
from orchestrator import api_key, license_state
from orchestrator.llm_backend import (api_client_kwargs, client_kwargs, llm_available,
                                      max_retries_for, raise_api_friendly)

GOOD = "sk-ant-api03-" + "a" * 40


class _Err(Exception):
    def __init__(self, status_code, message="", body=None):
        super().__init__(f"Error code: {status_code} - {message}")
        self.status_code = status_code
        self.body = body


class _Client:
    """A stand-in for anthropic.Anthropic(...).models.retrieve."""
    def __init__(self, outcome):
        self.outcome = outcome
        self.models = self

    def retrieve(self, model):
        assert model == api_key.VALIDATION_MODEL
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return {"id": model}


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("MIX_AGENT_LLM_BACKEND", raising=False)
    monkeypatch.delenv("MIX_AGENT_LICENSE", raising=False)
    monkeypatch.setenv("MIX_AGENT_DATA_DIR", str(tmp_path / "appdata"))
    monkeypatch.setenv("MIX_AGENT_CLOUD_URL", "https://meter.example.test")
    api_key.use_store(api_key.memory_store())
    license_state.reset()
    from orchestrator import device_auth
    device_auth.reset_shared()
    device_auth._shared = device_auth.DeviceLogin(
        "https://meter.example.test", keychain=device_auth.MemoryKeychain())
    yield
    api_key.use_store(None)
    license_state.reset()
    device_auth.reset_shared()


# ------------------------------------------------------------------ the key

def test_a_malformed_key_is_refused_before_any_network():
    res = api_key.set_api_key("hunter2", client_factory=lambda k: pytest.fail("no call"))
    assert res["ok"] is False and "sk-ant-" in res["error"]
    assert api_key.api_key_state() == {"set": False, "source": "none", "needed": True}


def test_a_good_key_is_validated_then_stored_and_never_echoed():
    seen = []
    res = api_key.set_api_key(GOOD, client_factory=lambda k: seen.append(k) or _Client(None))
    assert res == {"ok": True, "set": True, "source": "keychain", "needed": True}
    assert seen == [GOOD]
    assert GOOD not in json.dumps(res)
    assert api_key.api_key() == GOOD
    assert client_kwargs("api") == {"api_key": GOOD}
    assert llm_available() is True
    assert api_key.clear_api_key() == {"ok": True, "set": False, "source": "none", "needed": True}
    assert llm_available() is False


@pytest.mark.parametrize("outcome, fragment", [
    (_Err(401, "invalid x-api-key"), "rejected"),
    (_Err(404, "model not found"), "can't reach the model"),
    (_Err(403, "forbidden"), "can't reach the model"),
    (ConnectionError("dns"), "connection"),
])
def test_a_key_that_fails_validation_is_not_kept(outcome, fragment):
    res = api_key.set_api_key(GOOD, client_factory=lambda k: _Client(outcome))
    assert res["ok"] is False and fragment in res["error"]
    assert api_key.api_key() is None
    with pytest.raises(RuntimeError, match="Preferences"):
        api_client_kwargs()


def test_the_key_door_knows_when_no_key_is_needed(monkeypatch):
    """The subscription dev backend runs WITHOUT an Anthropic key; the DAW's composer door
    refused every send there unless a real key sat in the keychain (2026-09-22 audits D·F7 /
    E·F7). The hello's api_key state now says whether this transport needs one."""
    assert api_key.api_key_state()["needed"] is True                  # api (the default)
    monkeypatch.setenv("MIX_AGENT_LLM_BACKEND", "subscription")
    assert api_key.api_key_state() == {"set": False, "source": "none", "needed": False}
    monkeypatch.setenv("MIX_AGENT_LLM_BACKEND", "cloud")
    assert api_key.api_key_state()["needed"] is False


def test_a_shipped_brain_refuses_the_dev_transport_and_the_dev_bypasses(monkeypatch):
    """One `launchctl setenv` used to reach the frozen brain and switch the licence gate off:
    MIX_AGENT_LLM_BACKEND=subscription, MIX_AGENT_LICENSE=licensed, or any ambient
    ANTHROPIC_API_KEY (read as "a dev shell") (2026-09-22 audits D·F8 / C·F10)."""
    import sys

    from orchestrator.llm_backend import active_backend
    monkeypatch.setenv("MIX_AGENT_LLM_BACKEND", "subscription")
    monkeypatch.setenv("MIX_AGENT_LICENSE", "licensed")
    monkeypatch.setenv("ANTHROPIC_API_KEY", GOOD)
    # the control: a source checkout honours all three, as the dev loop needs
    assert active_backend() == "subscription" and license_state.gate_applies() is False
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert active_backend() == "api"
    assert license_state.gate_applies() is True
    monkeypatch.delenv("MIX_AGENT_LLM_BACKEND")
    assert license_state.gate_applies() is True, "an ambient key is not a developer, frozen"


def test_the_environment_key_wins_over_the_keychain(monkeypatch):
    api_key.set_api_key(GOOD, client_factory=lambda k: _Client(None))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env-" + "b" * 30)
    assert api_key.api_key_source() == "env"
    assert client_kwargs("api")["api_key"].startswith("sk-ant-env-")


def test_the_api_transport_gets_more_retries_and_its_own_sentences():
    assert max_retries_for("api", 1) > 1 and max_retries_for("cloud", 1) == 1
    cases = [
        (_Err(401), "rejected"),
        (_Err(400, body={"error": {"message": "Your credit balance is too low"}}), "out of credit"),
        (_Err(429), "rate limit"),
        (_Err(529, "overloaded"), "overloaded"),
        (_Err(403), "workspace"),
        # 2026-09-22 (every model on the picker): the new ways a USER's pick can fail, and the
        # 5xx that used to reach the status bar as the SDK's JSON dump
        (_Err(404), "can't use the model you picked"),
        (_Err(400, body={"error": {"message": "In order to access this model, your organization "
                                              "or workspace must have data retention enabled."}}),
         "data-retention"),
        (_Err(400, body={"error": {"message": "prompt is too long: 250000 tokens > 200000"}}),
         "context window"),
        (_Err(500), "server error"),
    ]
    for exc, fragment in cases:
        with pytest.raises(RuntimeError, match=fragment):
            raise_api_friendly(exc)
    # not ours: returns, the caller re-raises (an ordinary 400, anything without a status)
    raise_api_friendly(_Err(400, body={"error": {"message": "messages.0: bad shape"}}))
    raise_api_friendly(ValueError("x"))

    class APITimeoutError(Exception):   # the SDK's class NAME is what the mapping reads
        pass
    with pytest.raises(RuntimeError, match="timed out"):
        raise_api_friendly(APITimeoutError("Request timed out."))


# -------------------------------------------------------------- the licence

def _sign_in():
    from orchestrator import device_auth
    login = device_auth._shared
    login._keychain.set("r" * 40)
    login._access_token = "tok"
    from datetime import datetime, timedelta, timezone
    login._expires_at = datetime.now(timezone.utc) + timedelta(hours=1)


class _Http:
    def __init__(self, payload=None, status=200, fail=False):
        self.payload, self.status, self.fail, self.calls = payload, status, fail, 0

    def get(self, url, **kw):
        self.calls += 1
        if self.fail:
            raise ConnectionError("offline")
        assert url.endswith("/v1/license/me")
        assert kw["headers"]["Authorization"] == "Bearer tok"
        return self

    @property
    def status_code(self):
        return self.status

    def json(self):
        import licence_signing   # the service signs its answer (2026-10-20)
        return licence_signing.reply(self.payload)


def test_the_gate_never_applies_on_a_developer_transport(monkeypatch):
    assert license_state.gate_applies() is False            # api, no key at all
    monkeypatch.setenv("ANTHROPIC_API_KEY", GOOD)
    assert license_state.gate_applies() is False            # api on an env key
    assert license_state.state()["gated"] is False
    assert license_state.refusal() is None
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.setenv("MIX_AGENT_LLM_BACKEND", "subscription")
    assert license_state.gate_applies() is False
    monkeypatch.setenv("MIX_AGENT_LLM_BACKEND", "cloud")
    assert license_state.gate_applies() is True
    monkeypatch.setenv("MIX_AGENT_LICENSE", "licensed")
    assert license_state.gate_applies() is False            # the harness override


def test_a_shipped_install_is_gated_from_sign_in_to_purchase(monkeypatch):
    api_key.set_api_key(GOOD, client_factory=lambda k: _Client(None))
    assert license_state.gate_applies() is True
    assert license_state.refusal() == license_state.MSG_SIGN_IN   # keychain key, no sign-in
    _sign_in()
    import httpx
    from datetime import datetime, timedelta, timezone
    ends = (datetime.now(timezone.utc) + timedelta(days=3, hours=1)).isoformat()
    http = _Http({"status": "trial", "trial_ends_at": ends})
    monkeypatch.setattr(httpx, "get", http.get)
    st = license_state.state()
    assert st["status"] == "trial" and st["days_left"] == 4 and st["gated"] is True
    assert license_state.refusal() is None
    assert http.calls == 1
    license_state.state()                                    # cached for the day
    assert http.calls == 1
    # the purchase lands: a forced re-read sees it
    http.payload = {"status": "licensed", "purchased_at": "2026-09-21T00:00:00+00:00"}
    assert license_state.state(force=True)["status"] == "licensed"
    # expired, and refunded, each word their own door
    http.payload = {"status": "expired"}
    assert license_state.state(force=True)["message"] == license_state.MSG_EXPIRED
    assert license_state.refusal() == license_state.MSG_EXPIRED
    http.payload = {"status": "expired", "refunded": True}
    assert license_state.state(force=True)["message"] == license_state.MSG_REFUNDED


def test_offline_trusts_the_cache_for_the_grace_period_and_not_beyond(monkeypatch):
    api_key.set_api_key(GOOD, client_factory=lambda k: _Client(None))
    _sign_in()
    import httpx
    http = _Http({"status": "licensed"})
    monkeypatch.setattr(httpx, "get", http.get)
    assert license_state.state()["status"] == "licensed"
    cache = Path(license_state._cache_path())
    assert cache.is_file()
    # the service goes away: a fresh process (no memory) reads the disk cache
    license_state.reset()
    http.fail = True
    st = license_state.state()
    assert st["status"] == "licensed" and st["offline"] is True
    # ...but not a cache older than the grace period
    import licence_signing
    stale = licence_signing.cache(       # signed that long ago: the age is the service's clock
        {"status": "licensed",
         "checked_at": time.time() - (license_state.OFFLINE_GRACE_DAYS + 1) * 86400})
    cache.write_text(json.dumps(stale))
    license_state.reset()
    st = license_state.state()
    assert st["status"] == "unknown" and st["message"] == license_state.MSG_UNREACHABLE
    assert license_state.refusal() == license_state.MSG_UNREACHABLE
    # a cached TRIAL keeps counting down offline
    cache.write_text(json.dumps(licence_signing.cache(
        {"status": "trial", "checked_at": time.time(),
         "trial_ends_at": "2020-01-01T00:00:00+00:00"})))
    license_state.reset()
    assert license_state.state()["status"] == "expired"


# ---------------------------------------------------------- the brain's door

def _project(tmp_path: Path) -> Path:
    proj = tmp_path / "proj"
    proj.mkdir()
    sr = 44100
    t = np.arange(sr) / sr
    sf.write(proj / "vox.wav", (0.2 * np.sin(2 * np.pi * 220 * t)).astype("float32"), sr)
    Session(proj)
    return proj


#: how long the line is still listened to once what the caller waits for has arrived - time for
#: a message that should NOT follow it to show
SETTLE_S = 0.25


def _replied(*ids):
    """A `_talk` predicate: every one of these request ids has its reply."""
    return lambda out: set(ids) <= {o.get("id") for o in out}


def _said(*events):
    """A `_talk` predicate: one of these events has been emitted."""
    return lambda out: any(o.get("event") in events for o in out)


def _talk(proj: Path, lines, until=None) -> list:
    """One authenticated connection; `lines` is a callable(token) -> str.

    Everything the brain sends is collected until the line has been silent for three seconds.
    `until(out)` names what the caller is waiting for: once it holds, the line is listened to
    for SETTLE_S more and no longer - every caller used to sit out the whole three seconds
    after its last message. Without `until`, or if it never holds, the wait is the old one."""
    from orchestrator.daw_server import Hub, _serve_client

    async def scenario():
        hub = Hub(proj)
        server = await asyncio.start_server(
            lambda r, w: _serve_client(r, w, hub), "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        out = []
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            try:
                writer.write(lines(hub.token).encode())
                await writer.drain()
                while True:
                    quiet = SETTLE_S if until is not None and until(out) else 3.0
                    try:
                        line = await asyncio.wait_for(reader.readline(), quiet)
                    except asyncio.TimeoutError:
                        break
                    if not line:
                        break
                    out.append(json.loads(line))
            finally:
                writer.close()
        finally:
            server.close()
            await server.wait_closed()
        return out
    return asyncio.run(scenario())


def _line(obj) -> str:
    return json.dumps(obj) + "\n"


def test_the_socket_stores_the_key_reports_it_and_never_prints_it(tmp_path, capsys, monkeypatch):
    proj = _project(tmp_path)
    monkeypatch.setattr(api_key, "validate_api_key", lambda key, client_factory=None: None)
    out = _talk(proj, lambda token: (
        _line({"id": 1, "method": "auth", "params": {"token": token}})
        + _line({"id": 2, "method": "set_api_key", "params": {"key": GOOD}})
        + _line({"id": 3, "method": "api_key_state"})
        + _line({"id": 4, "method": "hello", "params": {"role": "daw", "project": str(proj)}})
        + _line({"id": 5, "method": "clear_api_key"})), until=_replied(2, 3, 4, 5))
    by_id = {o.get("id"): o for o in out}
    assert by_id[2]["result"] == {"ok": True, "set": True, "source": "keychain", "needed": True}
    assert by_id[3]["result"]["set"] is True
    hello = by_id[4]["result"]
    assert hello["api_key"]["set"] is True
    assert hello["license"]["status"] == "none"       # keychain key + not signed in
    assert hello["license"]["message"] == license_state.MSG_SIGN_IN
    assert by_id[5]["result"]["set"] is False
    captured = capsys.readouterr()
    assert GOOD not in captured.out and GOOD not in captured.err
    assert GOOD not in json.dumps(out)


def test_the_key_before_auth_runs_nothing(tmp_path):
    proj = _project(tmp_path)
    out = _talk(proj, lambda token: _line({"id": 1, "method": "set_api_key", "params": {"key": GOOD}}))
    assert out == []
    assert api_key.api_key() is None


def test_an_llm_turn_is_refused_at_the_door_and_the_rule_agent_is_not(tmp_path, monkeypatch):
    proj = _project(tmp_path)
    monkeypatch.setattr(api_key, "validate_api_key", lambda key, client_factory=None: None)
    api_key.set_api_key(GOOD)                    # a shipped-shaped install, not signed in
    # The door is the gesture's FIRST STEP since 2026-09-22 (audit C·F4/F7): the request is
    # acked at once (the run slot claimed with no await in front of it), and the refusal
    # arrives as the gesture's `run_failed`, carrying the sentence, before anything ran.
    for rid, method, params in ((2, "chat", {"message": "hi", "agent": "auto"}),
                                (3, "run", {"brief": "louder", "agent": "auto"})):
        out = _talk(proj, lambda token, m=method, p=params, i=rid: (
            _line({"id": 1, "method": "auth", "params": {"token": token}})
            + _line({"id": i, "method": m, "params": p})), until=_said("run_failed"))
        ack = next(o for o in out if o.get("id") == rid)
        assert ack["result"]["status"] == "started", ack
        failed = [o for o in out if o.get("event") == "run_failed"]
        assert failed and failed[0]["error"] == license_state.MSG_SIGN_IN, out
        assert failed[0].get("licence") is True
        assert not [o for o in out if o.get("event") in ("run_started", "chat_done")], \
            "a refused turn must not start"
    out = _talk(proj, lambda token: (
        _line({"id": 1, "method": "auth", "params": {"token": token}})
        + _line({"id": 4, "method": "advise", "params": {"brief": "louder", "agent": "rule"}})),
        until=_replied(4))
    by_id = {o.get("id"): o for o in out if "id" in o}
    assert by_id[4]["result"]["agent"] == "rule"    # the free agent is never gated
    assert not [o for o in out if o.get("event") == "run_failed" and o.get("licence")]


# ------------------------------------------------ what the manual promises

MANUAL = Path(__file__).resolve().parents[2] / "docs" / "17-user-manual.md"


def test_the_manual_quotes_the_sentences_the_code_says():
    """The four Anthropic-error sentences and the door's sentences are quoted in
    docs/17-user-manual.md ▸ Your account. A reworded sentence must move the manual."""
    text = MANUAL.read_text(encoding="utf-8")
    for exc, fragment in [(_Err(401), "your Anthropic API key was rejected"),
                          (_Err(400, body={"error": {"message": "credit balance"}}),
                           "your Anthropic account is out of credit"),
                          (_Err(429), "your Anthropic account's rate limit"),
                          # Phase 55: the two limits that are not rate limits
                          (_Err(429, body={"error": {"message": "You have reached your API usage limits",
                                                     "details": {"error_code": "enforced_spend_limit_reached"}}}),
                           "your Anthropic organization has reached its monthly spend cap"),
                          (_Err(400, body={"error": {"message": "You have reached your specified API usage limits."}}),
                           "your Anthropic account has reached the spend limit you set for it"),
                          (_Err(529), "Anthropic is overloaded right now")]:
        with pytest.raises(RuntimeError) as info:
            raise_api_friendly(exc)
        assert fragment in str(info.value) and fragment in text, fragment
    for sentence in (license_state.MSG_SIGN_IN, license_state.MSG_EXPIRED,
                     license_state.MSG_UNREACHABLE, license_state.MSG_REFUNDED,
                     license_state.MSG_RESIGN_IN):
        head = sentence.split(" — ")[0]
        assert head in text, head
    assert "add your Anthropic API key in Preferences" in text


class APITimeoutError(Exception):       # the SDK's class NAMES are what the mapping reads
    status_code = None


class APIConnectionError(Exception):
    status_code = None


def test_the_manual_quotes_the_2026_09_22_sentences_too():
    """The sentences added when the model became the user's pick (404 / retention / window /
    5xx) and the ones for a stream that died (timeout / dropped connection) are quoted in the
    manual by their head — the part before the dash."""
    text = MANUAL.read_text(encoding="utf-8")
    for exc in (_Err(404), _Err(400, body={"error": {"message": "data retention"}}),
                _Err(400, body={"error": {"message": "prompt is too long: 1 > 0"}}),
                _Err(500), APITimeoutError("Request timed out."),
                APIConnectionError("Connection error.")):
        with pytest.raises(RuntimeError) as info:
            raise_api_friendly(exc)
        head = str(info.value).split(" \u2014 ")[0].split(" (")[0]
        assert head and head in text, head


def test_the_price_table_is_pinned_to_the_current_list_rates():
    """On the user's own key the DAW's $ figure IS their bill at list price (Phase 53,
    `spend_is_estimate` is false on `api`) — so the table must be the current rates."""
    from orchestrator.agents import estimate_cost_usd
    # every picker model at its own list price (Anthropic's pricing page, read 2026-10-08) —
    # $ in, $ out per MTok. Haiku 5.5's is its price for a prompt up to 100,000 tokens, so it
    # is asked a tenth of a million at a time (test_agent_model_parity.py holds the long price).
    list_prices = {"claude-opus-5-5": (4, 20), "claude-sonnet-5-5": (2, 10),
                   "claude-haiku-5-5": (0.10, 0.50)}
    from orchestrator import llm_models
    assert set(list_prices) == {m.id for m in llm_models.MODELS}
    for model, (i, o) in list_prices.items():
        assert 10 * estimate_cost_usd(model, 100_000, 0) == pytest.approx(i), model
        assert 10 * estimate_cost_usd(model, 0, 100_000) == pytest.approx(o), model
    # cache reads: a twentieth of input on Opus 5.5 and Sonnet 5.5, a tenth on Haiku 5.5
    assert 10 * estimate_cost_usd("claude-opus-5-5", 0, 0, 100_000) == pytest.approx(0.20)
    assert 10 * estimate_cost_usd("claude-sonnet-5-5", 0, 0, 100_000) == pytest.approx(0.10)
    assert 10 * estimate_cost_usd("claude-haiku-5-5", 0, 0, 100_000) == pytest.approx(0.01)
    # the models the picker held until 2026-10-08 are still priced (an old baseline, an env
    # override), at their own rates
    for model, (i, o) in {"claude-fable-5-1": (10, 50), "claude-fable-5": (10, 50),
                          "claude-opus-5": (5, 25), "claude-opus-4-8": (5, 25),
                          "claude-opus-4-7": (5, 25), "claude-opus-4-6": (5, 25),
                          "claude-sonnet-5": (2, 10), "claude-sonnet-4-6": (3, 15),
                          "claude-haiku-4-5": (1, 5)}.items():
        assert estimate_cost_usd(model, 1_000_000, 0) == pytest.approx(i), model
        assert estimate_cost_usd(model, 0, 1_000_000) == pytest.approx(o), model
    assert estimate_cost_usd("claude-fable-5-1", 0, 0, 1_000_000) == pytest.approx(0.25)
    assert estimate_cost_usd("claude-opus-5", 0, 0, 1_000_000) == pytest.approx(0.50)
    from orchestrator.daw_server import spend_is_estimate
    assert spend_is_estimate() is False       # the api transport: a bill, not a price


def test_a_purchase_unlocks_on_the_next_message_not_the_next_launch(monkeypatch):
    """A negative answer was trusted for 24 h: a user who had just paid kept reading "your
    trial has ended — buy Zammoth" until they relaunched (2026-09-22 audit C·F1). A refusal
    now re-reads a negative state before refusing."""
    api_key.set_api_key(GOOD, client_factory=lambda k: _Client(None))
    _sign_in()
    answers = [{"status": "expired"}]
    monkeypatch.setattr(license_state, "_fetch",
                        lambda: {**answers[-1], "checked_at": time.time()})
    assert license_state.refusal() == license_state.MSG_EXPIRED
    answers.append({"status": "licensed"})                # ...then they pay
    monkeypatch.setattr(license_state, "REFUSAL_RECHECK_S", 0.0)
    assert license_state.refusal() is None
    # the control: with the re-read disabled and the negative cached for a day, still refused
    license_state.reset()
    answers.append({"status": "expired"})
    assert license_state.refusal() == license_state.MSG_EXPIRED
    answers.append({"status": "licensed"})
    monkeypatch.setattr(license_state, "REFUSAL_RECHECK_S", 1e9)
    monkeypatch.setattr(license_state, "NEGATIVE_REFRESH_S", 1e9)
    assert license_state.refusal() == license_state.MSG_EXPIRED


def test_a_lapsed_sign_in_keeps_a_confirmed_licence_and_says_so(monkeypatch):
    """A sign-in session lapses after 90 days; the licensed user was told "sign in to start your
    14-day trial" and locked out, the 30-day offline grace notwithstanding (C·F5)."""
    api_key.set_api_key(GOOD, client_factory=lambda k: _Client(None))
    license_state._write_cache(licence_signing.cache({"status": "licensed", "checked_at": time.time() - 86400}))
    st = license_state.state()                            # not signed in (nothing stored)
    assert st["status"] == "licensed" and st["message"] == license_state.MSG_RESIGN_IN
    assert license_state.refusal() is None
    license_state._write_cache(licence_signing.cache({"status": "expired", "checked_at": time.time()}))
    assert license_state.state()["message"] == license_state.MSG_SIGN_IN   # the control


def test_a_corrupt_licence_cache_never_fails_the_hello(monkeypatch):
    """`float(checked_at)` on a bad field raised out of state(), and the hello carrying it
    failed — the whole brain connection, not just the engineer (C·F12)."""
    api_key.set_api_key(GOOD, client_factory=lambda k: _Client(None))
    _sign_in()
    license_state._write_cache({"status": "licensed", "checked_at": "not-a-number"})

    def offline():
        raise ConnectionError("offline")
    monkeypatch.setattr(license_state, "_fetch", offline)
    st = license_state.state()
    assert st["status"] == "unknown" and st["message"] == license_state.MSG_UNREACHABLE

