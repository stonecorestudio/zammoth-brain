"""Pre-merge review of `full-audit` (2026-10-04), findings 5-13: the brain's three, and the
source-level pins for the app's six (their behaviour is driven by console legs - see
`docs/10-daw.md`, the last section).

Findings 1-4 are in `test_saved_checkpoint.py` and `test_audit_p1_fixes.py`.
"""
from __future__ import annotations

import asyncio
import re
import types
from pathlib import Path

import pytest

from mix_engine.models import ProposalRequest
from mix_engine.session import Session

ROOT = Path(__file__).resolve().parents[2]


def _apply(session: Session, actions: list[dict]) -> None:
    result = session.propose(ProposalRequest(rationale="review", predicted_impact=[], actions=actions))
    assert result.accepted, result.violations
    assert session.apply(result.proposal_id).ok


def _push_past_the_horizon(session: Session) -> None:
    for _ in range(60):                  # the trim runs at the horizon + 50
        session.propose(ProposalRequest(rationale="filler", predicted_impact=[], actions=[
            {"type": "gain", "stem_id": "stem_a", "delta_db": 12.0}]))   # refused: never applied


# ---- 8: a trimmed record charges the entity it acted on, not whoever holds the id now

def test_the_trim_does_not_charge_a_new_entity_under_an_old_id(demo_project, monkeypatch):
    import mix_engine.session as session_mod
    monkeypatch.setattr(session_mod, "_MAX_PROPOSAL_RECORDS", 2)
    session = Session(demo_project)
    _apply(session, [{"type": "gain", "stem_id": "stem_b", "delta_db": 5.0}])
    session.reset_stem("stem_b")                         # a NEW track under the old name
    assert session._cumulative_gain(None).get("stem_b", 0.0) == 0.0
    _push_past_the_horizon(session)
    assert len(session._state["proposals"]) <= 52        # the +5 record is past the horizon
    assert session._state.get("trimmed_gain_totals", {}).get("stem_b", 0.0) == 0.0
    assert session._cumulative_gain(None).get("stem_b", 0.0) == 0.0


def test_the_control_the_same_entitys_trimmed_gain_is_still_carried(demo_project, monkeypatch):
    import mix_engine.session as session_mod
    monkeypatch.setattr(session_mod, "_MAX_PROPOSAL_RECORDS", 2)
    session = Session(demo_project)
    _apply(session, [{"type": "gain", "stem_id": "stem_b", "delta_db": 5.0}])
    _push_past_the_horizon(session)
    assert session._state["trimmed_gain_totals"]["stem_b"] == 5.0
    assert session._cumulative_gain(None)["stem_b"] == 5.0


# ---- 7: nothing is imported from a private module of the SDK

def test_the_brain_imports_nothing_from_a_private_anthropic_module():
    import anthropic

    assert "transform_schema" in anthropic.__all__       # the name `_parse` relies on is public
    private = re.compile(r"^\s*(?:from|import)\s+anthropic(?:\.\w+)*\._\w+", re.M)
    for folder in ("orchestrator", "server/mix_engine", "backend/app"):
        for source in sorted((ROOT / folder).rglob("*.py")):
            found = private.search(source.read_text(encoding="utf-8"))
            assert found is None, f"{source.relative_to(ROOT)}: {found.group(0).strip()}"


# ---- 9: Stop inside a round does not swallow the turn's own cancellation

class _Stream:
    """A round that never finishes, and whose request takes a moment to close once cancelled."""
    def __init__(self, close_s: float):
        self.close_s = close_s

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await asyncio.sleep(self.close_s)
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.Event().wait()


def _round(monkeypatch, close_s: float):
    from orchestrator import chat

    monkeypatch.setattr(chat, "STOP_POLL_S", 0.01)
    me = types.SimpleNamespace(
        _client=types.SimpleNamespace(messages=types.SimpleNamespace(stream=lambda **_: _Stream(close_s))),
        _add_partial_usage=lambda stream: None)

    async def emit(_event):
        return None

    return chat.ChatSession._stream_round(me, {}, "t1", emit, False, lambda: True)


def test_stop_ends_the_round_and_nothing_else(monkeypatch):
    assert asyncio.run(_round(monkeypatch, 0.01)) is None


def test_a_turn_cancelled_while_its_stopped_round_closes_stays_cancelled(monkeypatch):
    async def scenario():
        turn = asyncio.ensure_future(_round(monkeypatch, 0.4))
        await asyncio.sleep(0.15)        # Stop has been seen; the round is still closing
        turn.cancel()                    # the project closes
        with pytest.raises(asyncio.CancelledError):
            await turn
        assert turn.cancelled()
    asyncio.run(scenario())


# ---- 10: whether a retry helps is said by whoever built the sentence

# The words the app used to search a failure for, to withhold its "retry" verb. Kept HERE, as
# the oracle: every sentence the brain's own builders produce must get the verdict the old
# search gave it, so moving the decision loses nothing.
_OLD_NO_RETRY_WORDS = (
    "Sign In", "subscription", "paused on our side", "access is disabled", "usage limit",
    "allotment", "try again later", "nothing is wrong with your account", "already applied",
    "process died", "timed out waiting", "API key", "Preferences", "trial", "licen",
    "zammoth.com/account", "sign in", "data-retention", "context window", "declined")


class _HttpError(Exception):
    def __init__(self, status_code, message="", *, detail=None, etype=""):
        super().__init__(message or f"Error code: {status_code}")
        self.status_code = status_code
        self.body = ({"detail": detail} if detail is not None
                     else {"error": {"message": message, "type": etype}})


class APITimeoutError(Exception):
    status_code = None
    body = None


class APIConnectionError(Exception):
    status_code = None
    body = None


def _built(builder, exc):
    try:
        builder(exc)
    except RuntimeError as friendly:
        return friendly
    return None


def _api_failures():
    return [_HttpError(400, "zero data retention is not supported for this model"),
            _HttpError(400, "prompt is too long: 250000 tokens"),
            _HttpError(413, "request too large"),
            _HttpError(404, "model not found"),
            _HttpError(401, "invalid x-api-key"),
            _HttpError(400, "your credit balance is too low"),
            _HttpError(429, "rate limited"),
            _HttpError(529, "overloaded"),
            _HttpError(503, "overloaded"),
            _HttpError(403, "forbidden"),
            _HttpError(500, "internal"),
            _HttpError(200, "stream", etype="overloaded_error"),
            _HttpError(200, "stream", etype="rate_limit_error"),
            _HttpError(200, "stream", etype="api_error"),
            APITimeoutError("Request timed out."),
            APIConnectionError("Connection error.")]


def _cloud_failures():
    return [_HttpError(code, detail=detail)
            for code in (401, 402, 403, 429, 503, 409, 413, 422)
            for detail in ("", "the monthly cap was reached")]


def test_every_sentence_the_brain_builds_keeps_the_verdict_the_word_search_gave_it():
    from orchestrator import llm_backend
    from orchestrator.api_key import MSG_NO_KEY

    checked = 0
    for builder, failures in ((llm_backend.raise_api_friendly, _api_failures()),
                              (llm_backend.raise_cloud_friendly, _cloud_failures())):
        for exc in failures:
            friendly = _built(builder, exc)
            assert friendly is not None, f"{builder.__name__} left {exc!r} untranslated"
            old = not any(word in str(friendly) for word in _OLD_NO_RETRY_WORDS)
            assert llm_backend.retry_helps(friendly) == old, str(friendly)
            checked += 1
    assert checked == 32
    assert not llm_backend.retry_helps(llm_backend.NoRetry(MSG_NO_KEY))
    assert any(word in MSG_NO_KEY for word in _OLD_NO_RETRY_WORDS)


def test_moves_that_landed_and_a_refusal_are_not_worth_a_retry():
    from orchestrator import chat, llm_backend
    from orchestrator.agents import ended_reason

    cut = chat.TurnCutOff(RuntimeError("the connection to Anthropic dropped"), 3)
    assert "already applied" in str(cut) and not llm_backend.retry_helps(cut)
    refused = ended_reason(types.SimpleNamespace(stop_reason="refusal", stop_details={"category": "bio"}))
    assert "declined" in str(refused) and not llm_backend.retry_helps(refused)
    cut_short = ended_reason(types.SimpleNamespace(stop_reason="max_tokens", stop_details=None))
    assert llm_backend.retry_helps(cut_short)                # ran out of room: ask again
    # a failure raised FROM one, and a group holding one
    try:
        try:
            raise llm_backend.NoRetry("your Anthropic API key was rejected")
        except llm_backend.NoRetry as inner:
            raise ValueError("planner failed") from inner
    except ValueError as wrapped:
        assert not llm_backend.retry_helps(wrapped)
    group = ExceptionGroup("tasks", [RuntimeError("dropped"), llm_backend.NoRetry("rejected")])
    assert not llm_backend.retry_helps(group)
    assert llm_backend.retry_helps(ExceptionGroup("tasks", [RuntimeError("dropped")]))
    assert llm_backend.retry_helps(RuntimeError("a file named trial take.wav could not be read"))


def test_a_failed_turn_tells_the_daw_whether_to_offer_a_retry(demo_project, monkeypatch):
    from orchestrator import daw_server, llm_backend

    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    class _Brain:
        def __init__(self, exc):
            self.exc = exc

        def advise(self, *args, **kwargs):
            raise self.exc

    async def scenario():
        for exc in (llm_backend.NoRetry("your Anthropic API key was rejected"),
                    RuntimeError("the connection to Anthropic dropped")):
            await daw_server._do_advise(send, _Brain(exc), 7, "brief", False)
    asyncio.run(scenario())
    failed = [m for m in sent if m.get("event") == "run_failed"]
    assert [m.get("retryable", True) for m in failed] == [False, True], sent
    # a failure after moves landed: the brain says so, and says a retry would repeat them
    brain = daw_server.Brain(demo_project)
    monkeypatch.setattr(daw_server, "moves_since", lambda project_dir, before: [1, 2])   # two landed
    with pytest.raises(llm_backend.NoRetry) as landed:
        brain._raise_if_landed(RuntimeError("dropped"), 0)
    assert "already applied" in str(landed.value)


def test_the_app_no_longer_reads_the_sentence_to_decide():
    app = (ROOT / "daw" / "Source" / "App.cpp").read_text(encoding="utf-8")
    body = app[app.index("void setStatus (juce::String s, bool retryable = false, bool joinFeed = false)"):]
    body = body[:body.index("void timerCallback() override")]
    assert "const bool canRetry = retryable && c->runBrief.isNotEmpty();" in body
    for word in _OLD_NO_RETRY_WORDS:
        assert f's.contains ("{word}")' not in body, word
    client = (ROOT / "daw" / "Source" / "BrainClient.h").read_text(encoding="utf-8")
    assert 'v.getProperty ("retryable", true)' in client
    assert app.count("brain.lastError, brain.lastErrorRetryable)") == 4
    assert app.count(": brain.lastError), retryHelps (done));") == 4


# ---- 11: one parse per line from the brain

def test_each_brain_line_is_parsed_once():
    client = (ROOT / "daw" / "Source" / "BrainClient.h").read_text(encoding="utf-8")
    assert client.count("if (! parseLine (resp, v))") == 2          # request() and waitForEvent()
    assert "lineReadable (resp)" not in client and "juce::JSON::parse (resp)" not in client


# ---- 5, 12, 13: pins on the app's side (the behaviour is in the console legs)

def _body(text: str, start: str, end: str) -> str:
    at = text.index(start)
    return text[at:text.index(end, at)]


def test_an_export_removes_only_the_stems_an_earlier_export_listed():
    app = (ROOT / "daw" / "Source" / "App.cpp").read_text(encoding="utf-8")
    export = _body(app, "void doExport()", "void startExportFromPanel")
    assert "dawexport::removeStemsOfEarlierExport (sdir);" in export
    assert "dawexport::noteStemsWritten (sdir, writtenStems);" in export
    assert "deleteFile" not in _body(export, "auto sdir = dawexport::stemsFolderFor (out);", "sdir.createDirectory();")
    panel = (ROOT / "daw" / "Source" / "ExportPanel.h").read_text(encoding="utf-8")
    remover = _body(panel, "inline int removeStemsOfEarlierExport", "inline bool noteStemsWritten")
    assert "stemsAnExportWrote (stemsDir)" in remover and "findChildFiles" not in remover


def test_a_saved_edit_is_indexed_once_per_version():
    session = (ROOT / "daw" / "Source" / "DawSession.h").read_text(encoding="utf-8")
    use = _body(session, "inline juce::String libraryItemUse", "inline juce::String libraryRemoveItem")
    assert "sourcesNamedInSavedEdit (f).count (id.toStdString())" in use
    index = _body(session, "sourcesNamedInSavedEdit (const juce::File& f)", "inline juce::String libraryItemUse")
    assert "entry.modifiedMs != modifiedMs || entry.bytes != bytes" in index


def test_a_save_writes_the_transcript_head_as_the_text_it_already_is():
    panel = (ROOT / "native" / "Source" / "EngineerPanel.h").read_text(encoding="utf-8")
    assert "juce::Array<juce::var> savedHead" not in panel
    writer = _body(panel, "void writeHistoryJson (juce::OutputStream& out) const", "void historyStats")
    assert "out.write (savedHeadText.getData(), savedHeadText.getDataSize());" in writer
    app = (ROOT / "daw" / "Source" / "App.cpp").read_text(encoding="utf-8")
    save = _body(app, "void saveChatHistory (bool force = false)", "juce::File chatLogFile() const")
    assert "engineer->writeHistoryJson (json);" in save and "historyToVar" not in save
