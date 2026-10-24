"""Phase 50 wild-corpus property gate (docs/05-roadmap.md).

Runs the full wild corpus through `orchestrator.wild_eval` — the engine-side
legs (discover / gestures / baseline / run / audible / export / bit-identical
undo / bounded), deterministic and LLM-free, so it can gate in CI. The DAW
console leg needs the native binary and is covered by the documented
`--daw` sweep instead (quality-gates skill).

These are the slowest tests in the suite by design (~2-3 min together: twelve
projects, two of them heavy). They earn the time: the corpus exists precisely to
catch what the friendly fixtures cannot — real-world-shaped projects breaking
import, mixing, export, or the undo guarantee.

One test a project (it was one test for the whole corpus): a red run names the
project, a parallel run (`pytest -n auto`) spreads them over the workers instead
of holding one for the full 160 s, and `-k long_6min` runs one alone. Each is the
same `wild_eval` run of that project the whole-corpus sweep makes, and the list is
read from the corpus itself, so a new project is a new test.
"""
import pytest
from orchestrator import wild_eval


@pytest.mark.parametrize("project", [spec.name for spec in wild_eval.PROJECTS])
def test_wild_corpus_all_green(project, capsys):
    rc = wild_eval.main(["--project", project])
    out = capsys.readouterr().out
    assert rc == 0, f"wild project {project} not green:\n{out}"

