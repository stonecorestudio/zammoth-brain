"""The pre-merge review of 2026-10-20 (audit-fixes -> main): the findings with a Python half.

Each case here was RED before its fix:
  1. a licence answer is believed for the request it answers and the account it names - a
     signed answer replayed by a server of one's own, or one for another account, was a
     licence (`license_state`);
  2. an overdrive REPLACES itself, so the composed EQ rail counts its tone once - re-proposing
     the agent's own overdrive was refused at +12 dB;
  3. the brain-source cut finds the knowledge book's text however a copy writes its strings -
     a verbatim copy of the book's own source (implicit concatenation) went out;
  10. the phase tool's description says the rule's own number;
  11. CLAUDE.md names the field `track.spectrogram` sends.
"""
from __future__ import annotations

import ast
import base64
import json
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import licence_signing as SIG
import test_api_key_license as LIC
from orchestrator import api_key, license_state

REPO = Path(__file__).resolve().parents[2]


# ====================================================================================
# 1 - a licence answer answers ONE request, for ONE account
# ====================================================================================

@pytest.fixture()
def gated(monkeypatch, tmp_path):
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
    assert license_state.gate_applies() is True
    yield device_auth._shared
    api_key.use_store(None)
    license_state.reset()
    device_auth.reset_shared()


def _jwt(sub: str) -> str:
    """An access token shaped like the control plane's (a JWT whose `sub` is the account)."""
    def part(obj):
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()
    return f"{part({'alg': 'HS256', 'typ': 'JWT'})}.{part({'sub': sub, 'sid': 's1'})}.c2ln"


def _signed_in_as(login, token: str) -> None:
    login._keychain.set("r" * 40)
    login._access_token = token
    login._expires_at = datetime.now(timezone.utc) + timedelta(hours=1)


class _Service:
    """A licence service that signs the answer it is given for the request it is asked -
    or, as a server of one's own does, hands back a token it captured earlier."""

    def __init__(self, payload, *, sub="user-1", captured=None):
        self.payload, self.sub, self.captured = payload, sub, captured
        self.nonces: list = []

    def get(self, url, **kw):
        assert url.endswith("/v1/license/me")
        nonce = (kw.get("params") or {}).get("nonce")
        self.nonces.append(nonce)
        service = self

        class _Reply:
            status_code = 200

            def json(self):
                if service.captured is not None:
                    return {**service.payload, "signed": service.captured}
                return {**service.payload,
                        "signed": SIG.sign(service.payload, sub=service.sub, nonce=nonce)}
        return _Reply()


def _cache_file(obj) -> None:
    path = Path(license_state._cache_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))
    license_state.reset()


def test_a_replayed_signed_answer_is_refused(gated, monkeypatch):
    """A licensed answer the service really signed - captured before a refund, or from a
    friend's account - played back by a server of one's own (MIX_AGENT_CLOUD_URL is honoured in
    a shipped build). It verified, so it was a licence for as long as the server kept answering."""
    import httpx
    _signed_in_as(gated, "tok")
    captured = SIG.sign({"status": "licensed"}, iat=time.time() - 40 * 86400)
    replay = _Service({"status": "licensed"}, captured=captured)
    monkeypatch.setattr(httpx, "get", replay.get)
    st = license_state.state()
    assert st["status"] == "unknown" and st["message"] == license_state.MSG_UNVERIFIED, st
    assert not Path(license_state._cache_path()).exists(), "a replayed answer was cached"
    # ...and one captured a moment ago is the same nothing: it answers another request
    replay.captured = SIG.sign({"status": "licensed"}, nonce="an-earlier-request")
    assert license_state.state(force=True)["status"] == "unknown"
    # every request carried a nonce of its own
    assert all(isinstance(n, str) and len(n) >= 16 for n in replay.nonces), replay.nonces
    assert len(set(replay.nonces)) == len(replay.nonces)
    # the control: the service's answer to THIS request is believed, and cached
    monkeypatch.setattr(httpx, "get", _Service({"status": "licensed"}).get)
    assert license_state.state(force=True)["status"] == "licensed"
    assert Path(license_state._cache_path()).exists()


def test_a_live_answer_for_another_account_is_refused(gated, monkeypatch):
    import httpx
    _signed_in_as(gated, _jwt("user-2"))
    monkeypatch.setattr(httpx, "get", _Service({"status": "licensed"}, sub="user-1").get)
    st = license_state.state()
    assert st["status"] == "unknown" and st["message"] == license_state.MSG_UNVERIFIED, st
    # the control: an answer for the account this machine is signed in as
    monkeypatch.setattr(httpx, "get", _Service({"status": "licensed"}, sub="user-2").get)
    assert license_state.state(force=True)["status"] == "licensed"


def test_a_cache_for_another_account_is_no_cache(gated, monkeypatch):
    """A friend's license.json copied in, the service blocked: it was the friend's licence
    for thirty days, on an account that never bought one."""
    import httpx

    def offline(*a, **k):
        raise ConnectionError("offline")
    monkeypatch.setattr(httpx, "get", offline)
    _signed_in_as(gated, _jwt("user-2"))
    _cache_file(SIG.cache({"status": "licensed", "checked_at": time.time() - 3600}))   # user-1's
    assert license_state.state()["status"] == "unknown"
    # the control: this account's own cache is the offline grace it always was
    _cache_file({"signed": SIG.sign({"status": "licensed"}, iat=time.time() - 3600, sub="user-2"),
                 "checked_at": time.time() - 3600})
    st = license_state.state()
    assert st["status"] == "licensed" and st["offline"] is True, st


def test_the_service_signs_the_nonce_it_was_asked_with():
    """The backend's half: `/v1/license/me?nonce=` is signed into the answer's body."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from backend.app import license_signing as BK
    key = Ed25519PrivateKey.generate()
    seed = key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                             serialization.NoEncryption())
    token = BK.sign_license({"status": "licensed"}, "user-7", f"k9:{BK.b64url(seed)}",
                            nonce="n" * 22)
    body = json.loads(BK.b64url_decode(token.split(".")[0]))
    assert body["nonce"] == "n" * 22 and body["sub"] == "user-7"
    plain = json.loads(BK.b64url_decode(
        BK.sign_license({"status": "licensed"}, "user-7", f"k9:{BK.b64url(seed)}").split(".")[0]))
    assert "nonce" not in plain


# ====================================================================================
# 2 - an overdrive replaces itself, so its tone is counted once
# ====================================================================================

def _od(tone: float) -> dict:
    return {"type": "overdrive", "stem_id": "stem_a", "drive_db": 0.0, "tone": tone}


def _shelf(gain_db: float) -> dict:
    return {"type": "eq", "stem_id": "stem_a",
            "band": {"shape": "high_shelf", "freq_hz": 6000.0, "gain_db": gain_db, "q": 0.7}}


def _land(session, *actions):
    from mix_engine.models import ProposalRequest
    res = session.propose(ProposalRequest(rationale="od", predicted_impact=[], actions=list(actions)))
    if res.accepted:
        assert session.apply(res.proposal_id).ok
    return res


def test_an_overdrive_that_replaces_itself_is_charged_once(demo_project):
    from mix_engine.session import Session
    session = Session(demo_project)
    assert _land(session, _od(1.0)).accepted
    again = _land(session, _od(1.0))                    # the same move, the same pass
    assert again.accepted, again.violations
    for _ in range(3):                                  # ...and pass after pass
        session.advance_pass()
        res = _land(session, _od(1.0))
        assert res.accepted, res.violations
    # the standing tone is +6 at the top: a +6 shelf fits the session's +12 beside it
    session.advance_pass()
    res = _land(session, _shelf(6.0))
    assert res.accepted, res.violations
    # the controls: the rail still holds what really stands
    session.advance_pass()
    over = _land(session, _shelf(6.0))
    assert not over.accepted and any("eq_boost_session" in v.rule for v in over.violations)
    same_pass = _land(session, _od(1.0), _shelf(1.0))   # tone 1 + the session's +12 + 1
    assert not same_pass.accepted


def test_a_replaced_overdrive_is_charged_its_delta_in_the_pass(demo_project):
    """Tone 1 stood since pass 1; pass 2 turns it to 0.5 and adds a +6 shelf: the PASS moved the
    top by 0 dB, not +6 + 6."""
    from mix_engine.session import Session
    session = Session(demo_project)
    assert _land(session, _od(1.0)).accepted
    session.advance_pass()
    res = _land(session, _od(0.5), _shelf(6.0))
    assert res.accepted, res.violations


def test_a_trimmed_overdrive_is_still_one_overdrive(demo_project, monkeypatch):
    """The retention trim folds a record's curves into a carry: an overdrive's must keep its LAST
    value, not add up, or the trim refills the rail it was meant to keep."""
    from mix_engine import session as session_mod
    from mix_engine.session import Session
    monkeypatch.setattr(session_mod, "_MAX_PROPOSAL_RECORDS", 1)   # trims past 51 records
    session = Session(demo_project)
    for _ in range(53):
        res = _land(session, _od(1.0))
        assert res.accepted, res.violations
        session.advance_pass()
    assert len(session._state["proposals"]) < 53, "nothing was trimmed: the case proves nothing"
    res = _land(session, _shelf(6.0))
    assert res.accepted, res.violations
    over = _land(session, _shelf(6.0))
    assert not over.accepted, "the trim forgot the standing overdrive"


# ====================================================================================
# 3 - the book's text, however a copy writes its strings
# ====================================================================================

sys.path.insert(0, str(REPO / "scripts"))
import brain_source_cut as CUT  # noqa: E402

BOOK = (REPO / "orchestrator" / "knowledge_content.py").read_text(encoding="utf-8")


def _written_in_pieces() -> list[str]:
    """The book's long strings whose SOURCE does not hold their text: implicit concatenations
    ("..." "...") and escaped quotes. A whole-book copy is caught by the other 93; these 29 were
    each a way out."""
    out = []
    for node in ast.walk(ast.parse(BOOK)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and len(CUT._squash(node.value)) >= CUT.FINGERPRINT_MIN:
            segment = ast.get_source_segment(BOOK, node)
            if CUT._squash(node.value) not in CUT._squash(segment):
                out.append(segment)
    return out


@pytest.mark.parametrize("path", ["server/tests/test_copied.py", "orchestrator/notes.txt"])
def test_a_copy_of_the_books_strings_as_they_are_written_stops_the_cut(tmp_path, monkeypatch, path):
    pieces = _written_in_pieces()
    assert len(pieces) >= 20, len(pieces)
    real = CUT.collect
    for i, segment in enumerate((pieces[0], pieces[-1], max(pieces, key=len))):
        body = f"TEXT = (\n    {segment}\n)\n"
        monkeypatch.setattr(CUT, "collect", lambda ref, body=body: {**real(ref), path: body.encode()})
        problems = CUT.cut(tmp_path / f"brain{i}", "HEAD")
        assert problems and path in problems[0], (segment[:80], problems)


def test_a_copy_with_its_escapes_stops_the_cut(tmp_path, monkeypatch):
    """`as \\"C1\\" is` in the source is `as "C1" is` in the book - and in prose that is not
    Python, so no parse can read the string for it."""
    segment = next(p for p in _written_in_pieces() if '\\"' in p)
    prose = f"Some notes from the book, {segment} and more."
    with pytest.raises(SyntaxError):
        ast.parse(prose)
    real = CUT.collect
    monkeypatch.setattr(CUT, "collect", lambda ref: {
        **real(ref), "orchestrator/notes.md": prose.encode()})
    problems = CUT.cut(tmp_path / "brain", "HEAD")
    assert problems and "orchestrator/notes.md" in problems[0], problems


# ====================================================================================
# 10 / 11 - words that name a number or a field say the code's
# ====================================================================================

def test_the_phase_tool_says_the_rules_number():
    from mix_engine import phase
    src = (REPO / "server" / "mix_engine" / "mcp_server.py").read_text(encoding="utf-8")
    doc = src[src.index("async def analyze_phase"):]
    doc = doc[:doc.index("return await")]
    said = set(re.findall(r"(\d+(?:\.\d+)?) %", doc))
    assert said == {f"{phase.MASTER_CANCEL_TIME_PCT:g}"}, \
        f"analyze_phase's description says {sorted(said)} %, the rule is {phase.MASTER_CANCEL_TIME_PCT:g} %"


def test_claude_md_names_the_band_tilt_field():
    text = (REPO / "CLAUDE.md").read_text(encoding="utf-8")
    assert "`track.spectrogram` `tilt_db_per_oct`" not in text
    assert "`track.spectrogram` `band_tilt_db_per_oct`" in text
    assert "The DAW's own `tilt_db_per_oct`" not in text
