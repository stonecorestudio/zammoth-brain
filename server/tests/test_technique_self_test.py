"""The technique eval's graders, proven for free in every full pytest (reader E7·R11 / E7·R12).

`technique_eval --self-test` is the proof that the Phase 48 graders measure something: every
golden passes both layers, the untouched project fails, the plausible cheat is rejected. Until
2026-10-07 nothing ran it - no test, no workflow - while its thresholds are calibrated to the
engine's DSP: a change to the de-esser, the dynamic EQ or the saturation curve could let a cheat
through, and the first to find out would be a billed run reading green.

It also could not see COLLATERAL damage: nine of the ten cases passed their golden plus -6 dB on
a stem the brief never mentioned. The outcome layer now holds every stem the technique does not
move to within `COLLATERAL_TOLERANCE_DB`, and the self-test proves it with a mutation.

About half a minute (it renders audio): `slow`, so a proportionate run can leave it out with
`-m "not slow"`. It writes nothing into the repo - the fixture is made in the work folder.
"""
from pathlib import Path

import pytest

from orchestrator import technique_eval as T

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.slow


def _fixture_stamp() -> dict:
    demo = ROOT / "test_projects" / T.PROJECT
    return {p.name: (p.stat().st_ino, p.stat().st_mtime_ns) for p in sorted(demo.glob("*.wav"))}


def test_the_self_test_is_green_and_leaves_the_repo_fixture_alone(capsys):
    before = _fixture_stamp()
    rc = T.self_test()
    out = capsys.readouterr().out
    assert rc == 0, out
    assert out.count(" ok") >= len(T.CASES), out
    assert _fixture_stamp() == before, "the self-test rewrote test_projects/technique_demo"


def test_a_grader_that_stops_seeing_the_cheat_turns_it_red(monkeypatch, capsys):
    """Seen red: the de-esser's outcome layer with its 'air' bound taken out accepts the
    static shelf, and the self-test says so."""
    case = next(c for c in T.CASES if c.key == "deess_lead")
    blind = lambda b, a: [f for f in T._out_deess(b, a) if "air" not in f and "melody" not in f]  # noqa: E731
    monkeypatch.setattr(T, "CASES", tuple(
        T.TechCase(**{**c.__dict__, "outcome": blind}) if c is case else c for c in T.CASES))
    assert T.self_test({"deess_lead"}) == 1
    assert "ACCEPTS the cheat" in capsys.readouterr().out


def test_a_golden_with_a_bystander_pulled_down_fails_the_outcome_layer(monkeypatch, capsys):
    """Seen red: with the collateral bound opened wide, 'the technique plus -6 dB on a stem
    nobody mentioned' passes again - and the self-test's mutation catches exactly that."""
    monkeypatch.setattr(T, "COLLATERAL_TOLERANCE_DB", 99.0)
    assert T.self_test({"deess_lead", "master_streaming"}) == 1
    out = capsys.readouterr().out
    assert out.count("collateral") >= 2, out


def test_every_case_says_which_stems_its_technique_may_move():
    from orchestrator.technique_fixture import STEM_NAMES
    for case in T.CASES:
        assert set(case.moves) <= set(STEM_NAMES), case.key
        assert T.bystander(case) not in case.moves, case.key
    assert {c.key: c.moves for c in T.CASES}["master_streaming"] == ()


def _levels(**moved) -> dict:
    base = {"drums": -20.0, "bassline": -18.0, "lead": -22.0, "pads": -24.0}
    return {"stem_rms_db": {**base, **{k: base[k] + v for k, v in moved.items()}}}


def test_the_collateral_check_names_the_stem_and_spares_the_target():
    case = next(c for c in T.CASES if c.key == "deess_lead")
    assert T.collateral_failures(case, _levels(), _levels()) == []
    assert T.collateral_failures(case, _levels(), _levels(lead=-4.0)) == [], "its own stem"
    assert T.collateral_failures(case, _levels(), _levels(pads=1.0)) == [], "inside the bound"
    fails = T.collateral_failures(case, _levels(), _levels(bassline=-6.0))
    assert len(fails) == 1 and "bassline" in fails[0] and "-6.0" in fails[0]
    assert T.collateral_failures(case, _levels(), _levels(drums=+3.0))


def test_the_outcome_layer_is_the_cases_grader_and_the_bystanders():
    own = T.TechCase("k", "t", "b", lambda actions: [], lambda b, a: ["own grader"], (),
                     moves=("lead",))
    assert T.outcome_failures(own, _levels(), _levels()) == ["own grader"]
    both = T.outcome_failures(own, _levels(), _levels(drums=-6.0))
    assert both[0] == "own grader" and "collateral" in both[1] and "drums" in both[1]


def test_a_billed_case_is_graded_with_the_collateral_check_too():
    import inspect
    src = inspect.getsource(T.run_case_llm)
    assert "outcome_failures(case" in src and "case.outcome(" not in src
