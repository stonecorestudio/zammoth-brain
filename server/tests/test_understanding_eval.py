"""The understanding eval itself (orchestrator/understanding.py, 2026-07-30).

benchmark.py measures whether the loop CLOSES its targets; it cannot tell you
whether those were the right targets. This eval covers the step before, and
these tests cover the eval — mostly so its gate cannot be quietly defanged
(deleting an awkward case, or letting mis-scoping grow).
"""
import pytest

from orchestrator import understanding as U


def test_corpus_is_labelled_consistently():
    assert len(U.CORPUS) >= 50, "the corpus is the asset — keep it wide"
    seen = set()
    for case in U.CORPUS:
        assert case.scope in (U.SESSION, U.TRACK, U.EDIT, U.TALK), case
        assert case.text not in seen, f"duplicate case: {case.text!r}"
        seen.add(case.text)
        if case.scope == U.TRACK:
            assert case.stem, f"a track-scoped case needs its stem: {case.text!r}"
            assert case.stem in U.STEMS, f"{case.stem} is not in the fixture session"
        else:
            assert case.stem is None, f"only track cases carry a stem: {case.text!r}"


def test_every_case_scores_to_a_known_verdict():
    for case in U.CORPUS:
        assert U.evaluate(case).verdict in ("resolved", "mis-scoped", "unresolved")


def test_a_named_instrument_scopes_to_that_instrument():
    """These two were the mis-scoped class until 2026-08-05: understood, then
    applied to the WHOLE mix. Fixed by the track-scope change (targets.py compiles
    a named stem's ask onto that stem's own bands). Asserting the verdict alone
    is not enough — 'resolved' on the wrong stem is the failure this guards."""
    for text, stem in (("give the vocal more air", "lead_vox"),
                       ("the piano is muddy", "piano")):
        outcome = U.evaluate(next(c for c in U.CORPUS if c.text == text))
        assert outcome.verdict == "resolved", f"{text!r}: {outcome.detail}"
        assert stem in outcome.detail, f"{text!r} resolved elsewhere: {outcome.detail}"


def test_a_question_never_becomes_a_mix():
    """'what does boxy mean' used to match the composite table on 'boxy' and EQ the
    user's master. It is a question: the only right answers are an LLM reply or
    nothing — never a compiled target."""
    outcome = U.evaluate(next(c for c in U.CORPUS if c.text == "what does boxy mean"))
    assert outcome.verdict == "unresolved"
    assert "became a mix" not in outcome.detail


def test_session_wide_asks_resolve_cleanly():
    for text in ("master for streaming", "master to -9 LUFS", "the mix is muddy"):
        case = next(c for c in U.CORPUS if c.text == text)
        assert U.evaluate(case).verdict == "resolved"


# ---------- the gate ----------

def _counts(resolved, mis_scoped, unresolved, over_reach=0):
    """A gate snapshot in the shape `_tally` actually produces — every gated
    class present. Built through a helper so adding a class breaks one line
    here, not every fixture (which is how `over-reach` came to be missing)."""
    return {"resolved": resolved, "mis-scoped": mis_scoped,
            "over-reach": over_reach, "unresolved": unresolved}


def test_growing_mis_scoping_fails_the_gate():
    base = {"total": 60, "counts": _counts(17, 4, 39)}
    worse = {"total": 60, "counts": _counts(17, 5, 38)}
    problems = U.compare(base, worse)
    assert problems and "mis-scoped grew" in problems[0]


def test_growing_over_reach_fails_the_gate():
    """The quietest harmful class (2026-08-04): right scope, more than was asked."""
    base = {"total": 60, "counts": _counts(17, 4, 39, over_reach=0)}
    worse = {"total": 60, "counts": _counts(17, 4, 38, over_reach=1)}
    assert any("over-reach grew" in p for p in U.compare(base, worse))


def test_losing_resolution_fails_the_gate():
    base = {"total": 60, "counts": _counts(17, 4, 39)}
    worse = {"total": 60, "counts": _counts(16, 4, 40)}
    assert any("resolved shrank" in p for p in U.compare(base, worse))


def test_deleting_cases_cannot_buy_a_pass():
    base = {"total": 60, "counts": _counts(17, 4, 39)}
    shrunk = {"total": 40, "counts": _counts(17, 0, 23)}
    assert any("corpus shrank" in p for p in U.compare(base, shrunk))


def test_improvement_passes():
    base = {"total": 60, "counts": _counts(17, 4, 39)}
    better = {"total": 60, "counts": _counts(25, 2, 33)}
    assert U.compare(base, better) == []


def test_a_baseline_predating_a_class_still_compares():
    """An older baseline has no `over-reach` key; the gate must read it as 0
    rather than raising, or adding a class bricks every checked-in baseline."""
    old = {"total": 60, "counts": {"resolved": 17, "mis-scoped": 4, "unresolved": 39}}
    assert U.compare(old, {"total": 60, "counts": _counts(17, 4, 39)}) == []
    assert any("over-reach grew" in p for p in
               U.compare(old, {"total": 60, "counts": _counts(17, 4, 38, over_reach=1)}))


def test_the_checked_in_baseline_matches_reality():
    """--check must pass on a clean tree, or the merge gate is noise."""
    import json
    assert U.BASELINE_PATH.exists(), "the baseline must be checked in"
    baseline = json.loads(U.BASELINE_PATH.read_text())
    assert U.compare(baseline, U.run()) == []


# ---------- the LLM path (scaffolding only — no LLM calls here) ----------

@pytest.mark.parametrize("steps,expected", [
    ([], (U.TALK, None)),
    ([{"kind": "command", "payload": {"command": "clip.split"}}], (U.EDIT, None)),
    ([{"kind": "run", "payload": {"brief": "master for streaming"}}], (U.SESSION, None)),
    ([{"kind": "auto_mix", "payload": {"target": "x"}}], (U.SESSION, None)),
    ([{"kind": "proposal", "payload": {"actions": [{"stem_id": "lead_vox"}]}}],
     (U.TRACK, "lead_vox")),
    ([{"kind": "proposal", "payload": {"actions": [{"stem_id": "master"}]}}],
     (U.SESSION, None)),
])
def test_plan_maps_to_a_route(steps, expected):
    """`_route_from_plan` also returns the actions it scored (the over-reach
    input); the route + stem are what this case pins."""
    route, stem, _actions = U._route_from_plan({"steps": steps})
    assert (route, stem) == expected


def test_a_run_anywhere_in_the_plan_is_session_scope():
    """The 2026-08-04 shape: 'propose the duck, and also start a run' read as a
    clean single move when only steps[0] was scored."""
    steps = [{"kind": "proposal", "payload": {"actions": [{"stem_id": "lead_vox"}]}},
             {"kind": "run", "payload": {"brief": "master for streaming"}}]
    route, stem, actions = U._route_from_plan({"steps": steps})
    assert (route, stem) == (U.SESSION, None)
    assert actions, "the proposal's actions still ride along for over-reach scoring"


def test_the_free_suite_never_runs_the_billed_sweep(monkeypatch):
    """`--agent claude` scores the real router and is BILLED (66 LLM calls).
    pytest must never reach it: this suite runs on every change, and a test that
    quietly bills is how a cost incident starts. The rule is enforced here rather
    than trusted — an earlier version of this test called `main(--agent claude)`
    directly and did bill on any machine with a key set."""
    def _refuse(*a, **kw):
        raise AssertionError("run_llm reached from the free test suite")

    monkeypatch.setattr(U, "run_llm", _refuse)
    assert U.main(["--check"]) == 0          # the free, deterministic gate
    assert U.main([]) == 0


def test_the_billed_sweep_keeps_its_own_baseline():
    """It is gated (2026-08-04) but against `routing_baseline.json`, never the
    free one — so a billed, non-deterministic sweep can't move the free gate."""
    assert U.LLM_BASELINE_PATH != U.BASELINE_PATH
    assert U.LLM_BASELINE_PATH.name == "routing_baseline.json"


def test_the_llm_fixture_context_names_the_tracks_and_commands():
    """The agent must be able to resolve a track and see what it can operate."""
    assert "snare_top" in U._LLM_CONTEXT and "lead_vox" in U._LLM_CONTEXT
    assert "clip.split" in U._LLM_CONTEXT
