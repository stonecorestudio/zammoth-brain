"""The operate eval's CLI gate must refuse to run rather than pass vacuously.

`--check` and `--update-baseline` both judge a change on `stable_failures` —
the set of cases that fail EVERY run — which only means something across
repeated runs. Two ways that was silently defeated (2026-08-16 review):

  * `--check` without `--repeat` never reached `_check()` at all. It fell
    through to the single-run path and returned 0 however many cases failed,
    which is indistinguishable from a passing gate.
  * `--repeat 1` gives every case a one-entry verdict histogram, so `repeat()`
    classes all of them "stable" and the gate reports full confidence off a
    single noisy sample.

Both must now be argparse errors. These assertions are free: the validation
runs before any billed work, so no LLM call is made.
"""
import pytest

from orchestrator import operate_eval


@pytest.fixture
def sandboxed_baseline(tmp_path, monkeypatch):
    """Point BASELINE_PATH at a throwaway file for the duration of the test.

    Not paranoia — earned. Running these cases against the UNFIXED code (to
    prove they catch the bug) let `--update-baseline --repeat 1` through to
    `BASELINE_PATH.write_text(...)`, which clobbered the checked-in
    operate_baseline.json with a single-run, every-case-failed result. The
    guard under test now prevents that, but a test that asks for a baseline
    write must never be able to reach the real one.
    """
    monkeypatch.setattr(operate_eval, "BASELINE_PATH",
                        tmp_path / "operate_baseline.json")


@pytest.mark.parametrize("argv", [
    ["--check"],                            # never reached _check(), exited 0
    ["--update-baseline"],                  # same fall-through, wrote nothing
    ["--check", "--repeat", "1"],           # one sample: everything "stable"
    ["--update-baseline", "--repeat", "1"],
])
def test_gate_refuses_without_enough_runs(argv, sandboxed_baseline):
    with pytest.raises(SystemExit) as exc:
        operate_eval.main(argv)
    assert exc.value.code == 2, f"{argv} should be an argparse error"


def test_list_stays_free_and_needs_no_repeat():
    """--list prints the corpus and exits clean; it makes no LLM call, so the
    new validation must not stand in its way."""
    assert operate_eval.main(["--list"]) == 0


def test_check_refuses_a_grep_subset(monkeypatch, sandboxed_baseline):
    """A partial run's stable_failures omit every case outside the grep, so
    each baseline failure outside it would print FIXED (2026-09-01 sweep) —
    the same vacuity as a renamed case. Refused before any billed work; the
    module-level GREP is restored by monkeypatch so later tests see the whole
    corpus."""
    monkeypatch.setattr(operate_eval, "GREP", operate_eval.GREP)
    with pytest.raises(SystemExit) as exc:
        operate_eval.main(["--check", "--repeat", "3", "--grep", "technique"])
    assert exc.value.code == 2
