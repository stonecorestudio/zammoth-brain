"""Subscription dev backend seam (MIX_AGENT_LLM_BACKEND).

Pins backend selection and the SubscriptionLLM guards without needing the
Claude Agent SDK or any credentials: the SDK import is deferred to call time,
and the credential guards run in __init__ before it. The live SDK paths are
covered by orchestrator/smoke_subscription.py instead.
"""
from datetime import datetime, timedelta, timezone

import pytest

from orchestrator.llm_backend import (
    ACCESS_TOKEN_ENV,
    BACKEND_ENV,
    CLOUD_URL_ENV,
    active_backend,
    cloud_client_kwargs,
    metering_headers,
)
from orchestrator.llm_subscription import (
    SubscriptionLLM,
    _is_max_turns_error,
    render_history,
)


def test_active_backend_defaults_to_api(monkeypatch):
    monkeypatch.delenv(BACKEND_ENV, raising=False)
    assert active_backend() == "api"


def test_active_backend_reads_env(monkeypatch):
    monkeypatch.setenv(BACKEND_ENV, "subscription")
    assert active_backend() == "subscription"
    monkeypatch.setenv(BACKEND_ENV, "  API ")  # case / whitespace tolerant
    assert active_backend() == "api"
    monkeypatch.setenv(BACKEND_ENV, "cloud")
    assert active_backend() == "cloud"


def test_active_backend_rejects_unknown(monkeypatch):
    monkeypatch.setenv(BACKEND_ENV, "bedrock")
    with pytest.raises(ValueError):
        active_backend()


@pytest.fixture(autouse=True)
def _no_real_keychain():
    """Keep these tests off this machine's login keychain.

    cloud_token() falls back to the keychain-backed device login when
    MIX_AGENT_ACCESS_TOKEN is unset, so without this a developer who has signed
    a DAW in sees different results from one who has not — and the fallback
    would try to reach the fake URL below for real.
    """
    from orchestrator import device_auth

    device_auth.reset_shared()
    device_auth._shared = device_auth.DeviceLogin(
        "https://meter.example.test", keychain=device_auth.MemoryKeychain())
    yield
    device_auth.reset_shared()


def test_cloud_backend_requires_a_url_and_some_token(monkeypatch):
    monkeypatch.delenv(CLOUD_URL_ENV, raising=False)
    monkeypatch.delenv(ACCESS_TOKEN_ENV, raising=False)
    with pytest.raises(RuntimeError, match=CLOUD_URL_ENV):
        cloud_client_kwargs()

    # A URL but nothing signed in and no token handed in: say so plainly rather
    # than firing a request that will come back 401.
    monkeypatch.setenv(CLOUD_URL_ENV, "https://meter.example.test/")
    with pytest.raises(RuntimeError, match="not signed in"):
        cloud_client_kwargs()

    # The env var still wins — that is the evaluation and CI path.
    monkeypatch.setenv(ACCESS_TOKEN_ENV, "user-jwt")
    assert cloud_client_kwargs() == {
        "base_url": "https://meter.example.test", "api_key": "user-jwt"}

    monkeypatch.setenv(CLOUD_URL_ENV, "http://attacker.example.test")
    with pytest.raises(RuntimeError, match="HTTPS"):
        cloud_client_kwargs()


def test_a_linked_machine_needs_no_env_token(monkeypatch):
    """The DAW's path: nothing in the environment, a session in the keychain."""
    from orchestrator import device_auth

    monkeypatch.setenv(CLOUD_URL_ENV, "https://meter.example.test")
    monkeypatch.delenv(ACCESS_TOKEN_ENV, raising=False)
    device_auth._shared._access_token = "minted-for-this-device"
    device_auth._shared._expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

    assert cloud_client_kwargs()["api_key"] == "minted-for-this-device"


def test_llm_available_tracks_backend_and_sign_in(monkeypatch):
    """The DAW sends agent:"auto" and the brain answers from its own state —
    the fix for a Finder-launched app silently running the rule agent."""
    from orchestrator import device_auth
    from orchestrator.llm_backend import llm_available

    # api: only a key makes an LLM reachable.
    monkeypatch.delenv(BACKEND_ENV, raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert llm_available() is False
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert llm_available() is True

    # cloud: a handed-in token or a keychain sign-in; neither → rule agent.
    # (The URL must match the fixture's fake login or shared_login() would
    # rebuild it against the real keychain.)
    monkeypatch.setenv(BACKEND_ENV, "cloud")
    monkeypatch.setenv(CLOUD_URL_ENV, "https://meter.example.test")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv(ACCESS_TOKEN_ENV, raising=False)
    assert llm_available() is False
    monkeypatch.setenv(ACCESS_TOKEN_ENV, "user-jwt")
    assert llm_available() is True
    monkeypatch.delenv(ACCESS_TOKEN_ENV, raising=False)
    device_auth._shared._keychain.set("refresh-token-0123456789")
    assert llm_available() is True


def test_cloud_business_errors_become_actionable_messages():
    from orchestrator.llm_backend import raise_cloud_friendly

    class FakeStatusError(Exception):
        def __init__(self, status_code, body=None):
            self.status_code = status_code
            self.body = body

    with pytest.raises(RuntimeError, match="subscription isn't active"):
        raise_cloud_friendly(FakeStatusError(
            402, {"detail": "an active subscription is required"}))
    with pytest.raises(RuntimeError, match="usage limit reached"):
        raise_cloud_friendly(FakeStatusError(
            429, {"detail": "usage allotment reached"}))
    with pytest.raises(RuntimeError, match="sign-in has expired"):
        raise_cloud_friendly(FakeStatusError(401))
    # Anything else returns so the caller re-raises the original.
    assert raise_cloud_friendly(FakeStatusError(500)) is None
    assert raise_cloud_friendly(ValueError("no status_code at all")) is None


def test_cloud_headers_route_and_make_each_call_idempotent():
    one = metering_headers("chat")
    two = metering_headers("chat")
    assert one["X-Metering-Request-Type"] == "chat"
    assert one["Idempotency-Key"] != two["Idempotency-Key"]


def test_subscription_requires_token(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="CLAUDE_CODE_OAUTH_TOKEN"):
        SubscriptionLLM("claude-opus-5-5")


def test_subscription_rejects_api_key(monkeypatch):
    # A set ANTHROPIC_API_KEY would make the engine bill the API account.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oat-test")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        SubscriptionLLM("claude-opus-5-5")


def test_subscription_constructs_with_token(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oat-test")
    llm = SubscriptionLLM("claude-opus-5-5")  # no SDK import, no network
    assert llm._model == "claude-opus-5-5"


def test_subscription_recognizes_max_turns_error():
    assert _is_max_turns_error(
        RuntimeError("Claude Code returned an error result: "
                     "Reached maximum number of turns (6)")
    )
    assert not _is_max_turns_error(RuntimeError("network down"))


def test_subscription_investigate_max_turns_returns_empty(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oat-test")
    llm = SubscriptionLLM("claude-opus-5-5")

    def fail_run(**_kwargs):
        raise RuntimeError("Claude Code returned an error result: "
                           "Reached maximum number of turns (6)")

    monkeypatch.setattr(llm, "_run", fail_run)
    proposals, usage = llm.investigate(
        system="system",
        opening_prompt="prompt",
        planner_tool_defs=[],
        engine_call=lambda _name, _args: {},
        validate=lambda proposals: proposals,
        max_rounds=6,
    )

    assert proposals == []
    assert usage.input_tokens == 0


def test_render_history_is_text_only():
    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
        {"role": "user", "content": [{"type": "tool_result", "content": "x"}]},
        {"role": "user", "content": "still here"},
    ]
    rendered = render_history(history)
    assert "User: hello" in rendered
    assert "Assistant: hi there" in rendered
    assert "tool_result" not in rendered  # structured blocks are skipped
    assert rendered.endswith("User: still here")


def test_request_adapters():
    assert SubscriptionLLM._system_text("plain") == "plain"
    assert SubscriptionLLM._system_text(
        [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    ) == "a\n\nb"
    assert SubscriptionLLM._prompt_text(
        [{"role": "user", "content": "q"}]) == "q"


def test_chat_session_defaults_to_api_backend(monkeypatch):
    monkeypatch.delenv(BACKEND_ENV, raising=False)
    from orchestrator.chat import ChatSession

    session = ChatSession()
    assert session._backend == "api"
    assert session._sub is None


def test_kill_switch_and_spend_cap_503_become_actionable():
    """The backend answers the kill switch, the global spend cap, AND a missing model
    rate all with 503 — the one business status the map missed, so a tester saw the
    raw SDK dump with a retry that could only fail again (2026-08-09 §1 audit)."""
    from orchestrator.llm_backend import raise_cloud_friendly

    class FakeStatusError(Exception):
        def __init__(self, status_code, body=None):
            self.status_code = status_code
            self.body = body

    with pytest.raises(RuntimeError, match="temporarily disabled"):
        raise_cloud_friendly(FakeStatusError(
            503, {"detail": "AI service temporarily disabled"}))
    with pytest.raises(RuntimeError, match="paused on our side"):
        raise_cloud_friendly(FakeStatusError(503))
    with pytest.raises(RuntimeError, match="refused this request"):
        raise_cloud_friendly(FakeStatusError(413, {"detail": "context too large"}))
    # A true server error still returns, so the caller re-raises the original.
    assert raise_cloud_friendly(FakeStatusError(500)) is None
