"""The two cost fixes from the 2026-09-03 eval-budget review.

* The subscription (SDK) chat path sent NO effort level, so every eval turn and
  every dawdev chat turn ran at Claude Code's own default — above the `high`
  the API path sends — and the composer's picker was ignored on that backend.
  The session's reasoning now rides `chat_turn(effort=...)`, the evals take
  `--effort`, and the provenance stamp records whichever ran.
* The operate `--repeat` sweep (~860 turns) died at the 5-hour usage limit with
  nothing kept. It now checkpoints every graded case, stops itself after a wave
  of consecutive transport errors, and resumes from the same command.
"""
from __future__ import annotations

import inspect
import json

import pytest

from orchestrator import chat as chatmod
from orchestrator import eval_provenance as P
from orchestrator import operate_eval as O


# ------------------------------------------------ effort on the dev backend ----

class _Sub:
    def __init__(self):
        self.effort = "unset"

    async def chat_turn(self, *, system, prompt, tool_defs, execute,
                        on_delta, on_tool, max_turns, effort=None, should_stop=None, max_tokens=None, timer=None):
        from orchestrator.llm_subscription import _Usage
        self.effort = effort
        return "done", _Usage(input_tokens=1000, output_tokens=50,
                              cache_read_input_tokens=26000,
                              cache_creation_input_tokens=0)


def _turn(monkeypatch, **session_kwargs):
    import anyio

    monkeypatch.setenv("MIX_AGENT_LLM_BACKEND", "subscription")
    session = chatmod.ChatSession(client=object(), **session_kwargs)
    session._sub = _Sub()

    async def unsupported(*_a, **_k):
        return "unsupported"

    tools = chatmod.ChatTools(
        engine_call=unsupported, propose=unsupported, start_run=unsupported,
        stop_run=unsupported, auto_mix=unsupported, file_instruction=unsupported,
        save_note=unsupported, workspace_command=unsupported, list_files=unsupported,
        import_sample=unsupported, undo_proposal=unsupported, render=unsupported,
        daw_command=unsupported)

    async def emit(_e):
        return None

    anyio.run(lambda: session.turn("hello", tools=tools, context="CTX", emit=emit))
    _turn.last_session = session
    return session._sub.effort


def test_the_subscription_chat_path_sends_the_sessions_reasoning(monkeypatch):
    assert _turn(monkeypatch) == chatmod.DEFAULT_REASONING, \
        "the default session must run the SDK at chat's default, not the SDK's"
    assert _turn(monkeypatch, reasoning="low") == "low"


def test_a_model_without_effort_config_sends_none(monkeypatch, odd_ladders):
    """No picker model is one since 2026-10-08 (every 5.5 model takes the whole ladder), so
    the rule is held on a stand-in row; the three on the picker each send what was asked."""
    _no_xhigh, no_effort = odd_ladders
    assert _turn(monkeypatch, model=no_effort, reasoning="low") is None
    assert all(m["effort"] for m in chatmod.CHAT_MODELS)
    for m in chatmod.CHAT_MODELS:
        assert _turn(monkeypatch, model=m["id"], reasoning="xhigh") == "xhigh", m["id"]


def test_the_sdk_shim_accepts_effort_on_both_chat_entry_points():
    from orchestrator.llm_subscription import SubscriptionLLM
    for name in ("chat_turn", "observe"):
        assert "effort" in inspect.signature(getattr(SubscriptionLLM, name)).parameters


def test_the_provenance_stamp_records_the_effort_that_ran():
    assert P.provenance()["reasoning"] == chatmod.DEFAULT_REASONING
    assert P.provenance(reasoning="low")["reasoning"] == "low"
    base = {"meta": P.provenance()}
    notes = P.provenance_notes(base, reasoning="low")
    assert any("effort differs" in n and "'low'" in n for n in notes)
    assert P.provenance_notes(base) == []


def test_both_evals_expose_effort_and_thread_it_to_the_session():
    from orchestrator import qa_eval as Q
    for mod in (O, Q):
        assert "effort" in inspect.signature(mod.evaluate).parameters
        assert "effort" in inspect.signature(mod.run).parameters
        src = inspect.getsource(mod.evaluate)
        assert '"reasoning": effort' in src, f"{mod.__name__} must hand --effort to ChatSession"


# ------------------------------------------------ checkpoint + resume ----

@pytest.fixture
def small_corpus(monkeypatch):
    cases = O.CORPUS[:5]
    monkeypatch.setattr(O, "selected_cases", lambda: list(cases))
    return cases


def _scripted(verdicts):
    """A fake evaluate: pops the next scripted verdict per call (counts calls)."""
    calls = []

    def fake(case, model=None, effort=None):
        calls.append(case.text)
        v = verdicts.pop(0) if verdicts else "correct"
        return O.Outcome(case, v, "RuntimeError: usage limit" if v == "error" else "ok")
    fake.calls = calls
    return fake


def test_an_error_wave_aborts_the_sweep_and_keeps_the_measured_cases(
        small_corpus, monkeypatch, tmp_path):
    ck = tmp_path / "ck.json"
    # run 0 completes (5), run 1 measures 2, then the backend dies
    fake = _scripted(["correct"] * 7 + ["error"] * 50)
    monkeypatch.setattr(O, "evaluate", fake)
    with pytest.raises(O.SweepAborted) as exc:
        O.repeat(jobs=1, times=3, checkpoint=ck)
    assert exc.value.consecutive == O._ABORT_AFTER_CONSECUTIVE_ERRORS
    assert len(fake.calls) == 7 + O._ABORT_AFTER_CONSECUTIVE_ERRORS, \
        "the sweep must stop at the threshold, not burn the remaining cases"
    data = json.loads(ck.read_text())
    assert len(data["runs"]["0"]) == 5 and len(data["runs"]["1"]) == 2
    assert "2" not in data["runs"]
    assert not any(r["verdict"] == "error" for run in data["runs"].values()
                   for r in run.values()), "an error is not a measurement; never keep one"
    assert data["meta"]["times"] == 3 and data["meta"]["reasoning"] == chatmod.DEFAULT_REASONING


def test_the_same_command_resumes_and_measures_only_what_is_missing(
        small_corpus, monkeypatch, tmp_path):
    ck = tmp_path / "ck.json"
    monkeypatch.setattr(O, "evaluate", _scripted(["correct"] * 7 + ["error"] * 50))
    with pytest.raises(O.SweepAborted):
        O.repeat(jobs=1, times=3, checkpoint=ck)
    fake = _scripted([])          # the backend is back
    monkeypatch.setattr(O, "evaluate", fake)
    agg = O.repeat(jobs=1, times=3, checkpoint=ck)
    assert len(fake.calls) == 15 - 7, "kept measurements must not be paid for twice"
    assert agg["runs"] == 3 and agg["scores"] == [100.0, 100.0, 100.0]
    assert agg["stable_correct"] == 5 and agg["errors"] == [0, 0, 0]
    assert not ck.exists(), "a completed sweep must not seed the next one"


def test_a_checkpoint_from_different_conditions_is_refused(
        small_corpus, monkeypatch, tmp_path):
    ck = tmp_path / "ck.json"
    monkeypatch.setattr(O, "evaluate", _scripted(["correct"] * 7 + ["error"] * 50))
    with pytest.raises(O.SweepAborted):
        O.repeat(jobs=1, times=3, checkpoint=ck)
    monkeypatch.setattr(O, "evaluate", _scripted([]))
    with pytest.raises(O.CheckpointMismatch, match="times"):
        O.repeat(jobs=1, times=2, checkpoint=ck)
    with pytest.raises(O.CheckpointMismatch, match="reasoning"):
        O.repeat(jobs=1, times=3, effort="low", checkpoint=ck)
    # --fresh discards it and the sweep starts over
    fake = _scripted([])
    monkeypatch.setattr(O, "evaluate", fake)
    O.repeat(jobs=1, times=2, checkpoint=ck, fresh=True)
    assert len(fake.calls) == 10


def test_the_pool_path_aborts_too(small_corpus, monkeypatch, tmp_path):
    ck = tmp_path / "ck.json"
    monkeypatch.setattr(O, "evaluate", _scripted(["error"] * 50))
    with pytest.raises(O.SweepAborted):
        O.repeat(jobs=3, times=3, checkpoint=ck)
    # nothing was measured, so there is nothing to keep — and nothing to resume
    assert not ck.exists() or json.loads(ck.read_text())["runs"] == {}


def test_the_cli_reports_an_abort_with_the_resume_recipe(
        small_corpus, monkeypatch, tmp_path, capsys):
    ck = tmp_path / "ck.json"
    monkeypatch.setattr(O, "evaluate", _scripted(["correct"] * 3 + ["error"] * 50))
    rc = O.main(["--repeat", "2", "--jobs", "1", "--checkpoint", str(ck)])
    assert rc == 3
    err = capsys.readouterr().err
    assert "ABORTED" in err and "3 of 10 measurements" in err and "--fresh" in err
    assert O.CHECKPOINT_PATH.name == "operate_checkpoint.json"


# ------------------------------------------------ cache reads are visible ----

def test_the_session_keeps_the_usage_the_dev_backend_returns(monkeypatch):
    """ChatSession dropped the SDK's usage on the floor, so nobody could see
    whether the ~27k-token eval prompt was read from cache or paid in full."""
    _turn(monkeypatch)
    u = _turn.last_session.usage
    assert u == {"input_tokens": 1000, "cache_read_input_tokens": 26000,
                 "cache_creation_input_tokens": 0, "output_tokens": 50,
                 "cache_creation_1h_input_tokens": 0}
    # a dict or a fake with none of the fields adds nothing and never raises
    _turn.last_session._add_usage({"input_tokens": 5})
    _turn.last_session._add_usage(object())
    _turn.last_session._add_usage(None)
    assert _turn.last_session.usage["input_tokens"] == 1005


def test_usage_totals_and_the_cache_warning():
    class O_:  # noqa: N801 — any object with the four fields
        def __init__(self, i, r, w, o):
            self.input_tokens, self.cache_read_tokens = i, r
            self.cache_write_tokens, self.output_tokens = w, o
    healthy = P.usage_totals([O_(1000, 26000, 0, 50), O_(1000, 26000, 0, 50)])
    assert healthy["prompt_tokens"] == 54000 and healthy["cache_share"] == 0.963
    assert "CACHE NOT LANDING" not in P.usage_line(healthy)
    assert "~27,000 prompt tokens per case" in P.usage_line(healthy)
    cold = P.usage_totals([O_(27000, 0, 0, 50)] * 3)
    assert cold["cache_share"] == 0.0
    assert "CACHE NOT LANDING" in P.usage_line(cold)
    assert P.usage_line(P.usage_totals([])) == ""


def test_the_checkpoint_keeps_what_a_case_billed(small_corpus, tmp_path):
    ck = O._Checkpoint(tmp_path / "ck.json", {"x": 1})
    case = small_corpus[0]
    ck.record(0, O.Outcome(case, "correct", "ok", input_tokens=7, output_tokens=8,
                           cache_read_tokens=9, cache_write_tokens=10))
    back = O._Checkpoint(tmp_path / "ck.json", {"x": 1})
    back.runs = json.loads((tmp_path / "ck.json").read_text())["runs"]
    o = back.measured(0)[case.text]
    assert (o.input_tokens, o.output_tokens, o.cache_read_tokens, o.cache_write_tokens) \
        == (7, 8, 9, 10)


def test_the_sweep_report_carries_the_token_line(small_corpus, monkeypatch, capsys):
    def fake(case, model=None, effort=None):
        return O.Outcome(case, "correct", "ok", input_tokens=27000, cache_read_tokens=0)
    monkeypatch.setattr(O, "evaluate", fake)
    agg = O.repeat(jobs=1, times=2)
    assert agg["tokens"]["cases"] == 10 and agg["tokens"]["cache_share"] == 0.0
    O._report_repeat(agg)
    out = capsys.readouterr().out
    assert "prompt 270,000" in out and "tokens [" in out and "CACHE NOT LANDING" in out


# ------------------------------------------------ the SDK request carries only OUR tools ----

def test_the_sdk_request_ships_no_builtin_tools_and_no_configured_mcp_servers():
    """Measured 2026-09-03: an operate case cost 201k prompt tokens with Claude
    Code's built-in tools + the cwd's configured MCP servers riding along, 111k
    without. And under bypassPermissions those built-ins (Bash, Edit, Write…)
    were LIVE on the user's machine — allowed_tools never restricted them."""
    from orchestrator.llm_subscription import SubscriptionLLM
    src = inspect.getsource(SubscriptionLLM._run)
    assert "tools=[]" in src
    assert "strict_mcp_config=True" in src


# ------------------------------------------------ the 2026-09-03 regen's grader findings ----

def test_args_contain_is_satisfied_by_any_expected_command_the_turn_called():
    """'no, I meant the piano, not the vocal' is honestly two track.solo calls; the
    grader read the FIRST one's args only and could never see the piano."""
    case = next(c for c in O.CORPUS if c.text == "no, I meant the piano, not the vocal")
    steps = [{"kind": "command", "payload": {"command": "track.solo",
                                             "args": {"track": "lead_vox", "on": False}}},
             {"kind": "command", "payload": {"command": "track.solo",
                                             "args": {"track": "piano", "on": True}}}]
    verdict, detail, _ = O.grade(case, steps, "Piano soloed, vocal un-soloed.")
    assert verdict == "correct", detail


def test_every_args_contain_names_a_token_the_fixture_or_request_carries():
    """Eight assertions were unpassable: word-bounded matching against a fixture
    value glued to more identifier characters ('Valhalla' vs 'ValhallaRoom')."""
    import json as _json
    import re as _re
    haystack = _json.dumps(O.OPERATE_EDIT) + " " + _json.dumps(O.OPERATE_SEND_RETURNS)
    bad = []
    for c in O.CORPUS:
        for want in c.args_contain:
            alts = want.split("|")
            # a value the MODEL produces (a gain, a position) is not in any fixture
            if not any(_re.search("[A-Za-z]", alt) for alt in alts):
                continue
            text = " ".join([haystack, c.text, " ".join(c.turns),
                             _json.dumps(c.context()) if c.context else "",
                             _json.dumps(c.live()) if c.live else "",
                             _json.dumps(c.send_returns()) if c.send_returns else ""])
            ok = any(_re.search(r"(?<![A-Za-z0-9_])" + _re.escape(alt) + r"(?![A-Za-z0-9_])",
                                text, _re.IGNORECASE) for alt in alts)
            if not ok:
                bad.append((c.text, want))
    assert not bad, f"assertions no fixture/request token can satisfy word-bounded: {bad}"
