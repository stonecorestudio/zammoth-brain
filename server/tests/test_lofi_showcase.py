"""The lofi showcase's graders (Phase 52, orchestrator/lofi_showcase.py).

The showcase itself needs the real app and a billed agent; these are the free
half — proof that the three grading layers MEASURE something. A grader that
passes its own golden fixture and nothing else is not a gate (memory
`gates-that-measure-nothing`), so the mutation leg is the one that matters:
each mutation breaks the golden project in one specific way and the check that
names that thing must go red.
"""
from pathlib import Path

import pytest

from orchestrator import lofi_showcase as L


@pytest.fixture(scope="module")
def golden(tmp_path_factory) -> Path:
    """The synthetic project that should pass every layer — including a real
    150 s D-minor 82 BPM master, since OUTCOME measures the audio, not a claim
    about it."""
    return L._golden_project(tmp_path_factory.mktemp("showcase"))


@pytest.fixture(scope="module")
def blank(tmp_path_factory) -> Path:
    """What `fresh.sh` leaves: an empty edit and a silent export."""
    return L._blank_project(tmp_path_factory.mktemp("showcase-blank"))


def _hard_failures(project: Path) -> list[str]:
    return [c.name for c in L._all_checks(project) if c.fails]


def test_the_script_covers_every_stage_and_names_real_commands():
    assert len(L.STAGES) == 8
    assert [s.n for s in L.STAGES] == list(range(1, 9))
    # every command family the PLAN layer grades must be a key the agent can
    # actually invoke, or the stage measures nothing
    from orchestrator.operate_eval import _registry_commands
    catalog = {k for k, _u, _c in _registry_commands()}
    # the mix-run tools are chat tools, not daw_command keys
    chat_tools = {"propose", "start_run", "auto_mix"}
    for stage in L.STAGES:
        assert stage.requires, f"stage {stage.n} grades nothing"
        for req in stage.requires:
            for key in req.any_of:
                assert key in catalog or key in chat_tools, (
                    f"stage {stage.n} wants {key!r}, which is not an invocable command")


def test_the_golden_passes_every_layer(golden: Path):
    assert _hard_failures(golden) == []


def test_the_plan_layer_reads_the_feed_not_the_capped_digest(golden: Path):
    _checks, degraded, turns = L.grade_plan(golden)
    assert not degraded and len(turns) == 8
    # stage 7 needs four bounces; the digest caps at 8 unique keys and would lose them
    assert turns[6]["commands"].count("track.bounce") == 4


def test_an_untouched_project_fails_every_layer(blank: Path):
    plan, _degraded, _turns = L.grade_plan(blank)
    assert [c for c in plan if c.fails]
    assert [c for c in L.grade_structure(blank) if c.fails]
    silent = L.find_export(blank)
    assert silent is not None
    outcome = L.grade_outcome(silent)
    assert [c for c in outcome if c.fails]
    # and it fails on MUSIC, not merely on a missing file
    red = {c.name for c in outcome if not c.ok}
    assert "it plays at the tempo asked for" in red
    assert "no dead air" in red


@pytest.mark.parametrize("label,mutate,want_red",
                         L._mutations(), ids=[m[0] for m in L._mutations()])
def test_each_mutation_turns_its_own_check_red(golden: Path, tmp_path: Path,
                                               label, mutate, want_red):
    import shutil
    mutant = tmp_path / "mutant"
    shutil.copytree(golden, mutant)
    mutate(mutant)
    red = _hard_failures(mutant)
    assert any(want_red in name for name in red), (
        f"{label}: nothing failed matching {want_red!r}; red: {red}")


def test_the_export_finder_ignores_the_intermediate_bounces(tmp_path: Path):
    proj = tmp_path / "p"
    (proj / "Exported").mkdir(parents=True)
    (proj / "Keys bounce.wav").write_bytes(b"")
    real = proj / "Exported" / "song.wav"
    real.write_bytes(b"")
    assert L.find_export(proj) == real
