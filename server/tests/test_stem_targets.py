"""Per-stem metric targets (2026-07-30).

Until now the pass loop could only measure SESSION-level numbers, so
`SESSION_TARGET_METRICS` deliberately excluded `stem(...)` and the brief
compiler had to flatten every per-instrument request into a master-wide goal.
That is the failure where the agent understands the words perfectly and then
changes the whole mix to fix one track: "the snare has no crack" became a
session crest-factor floor rather than that snare's transient ratio.

These tests prove the loop can now measure a per-stem target end to end.
"""
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from mix_engine.mcp_server import create_server
from orchestrator.loop import EngineClient
from orchestrator.targets import STEM_TARGET_FIELDS, Target, compiled_targets


def in_memory_engine(project_dir):
    return create_connected_server_and_client_session(create_server(project_dir))


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_measure_all_resolves_a_stem_metric(demo_project):
    async with in_memory_engine(demo_project) as client:
        engine = EngineClient(client)
        values = await engine.measure_all([
            "stem(stem_a).lufs_s",
            "stem(stem_a).transient_energy_ratio",
            "stem(stem_a).band_energy.presence",
        ])
    assert values["stem(stem_a).lufs_s"] is not None
    assert values["stem(stem_a).transient_energy_ratio"] is not None
    assert values["stem(stem_a).band_energy.presence"] is not None


@pytest.mark.anyio
async def test_two_metrics_on_one_stem_cost_one_analysis(demo_project):
    """The stem analysis is cached per measurement round, like analyze_session."""
    from mix_engine import dsp

    async with in_memory_engine(demo_project) as client:
        engine = EngineClient(client)
        for key in dsp.RENDER_COUNTS:
            dsp.RENDER_COUNTS[key] = 0
        await engine.measure_all([
            "stem(stem_a).lufs_s",
            "stem(stem_a).dynamic_spread_db",
            "stem(stem_a).band_energy.mid",
        ])
        assert dsp.RENDER_COUNTS["stem"] == 1, dsp.RENDER_COUNTS


@pytest.mark.anyio
async def test_an_unknown_stem_measures_as_none_not_a_crash(demo_project):
    async with in_memory_engine(demo_project) as client:
        engine = EngineClient(client)
        values = await engine.measure_all(["stem(ghost).lufs_s"])
    assert values["stem(ghost).lufs_s"] is None


@pytest.mark.anyio
async def test_a_stem_target_reads_as_unmet_then_met(demo_project):
    """A Target over a stem metric behaves like any other: distance 0 when met."""
    async with in_memory_engine(demo_project) as client:
        engine = EngineClient(client)
        value = (await engine.measure_all(
            ["stem(stem_a).lufs_s"]))["stem(stem_a).lufs_s"]
    met = Target(metric="stem(stem_a).lufs_s", unit="LUFS", goal=value,
                 tolerance=0.5)
    unmet = Target(metric="stem(stem_a).lufs_s", unit="LUFS", goal=value + 12.0,
                   tolerance=0.5)
    assert met.distance(value) == 0.0
    assert unmet.distance(value) > 0.0


def test_every_advertised_stem_field_is_a_real_analysis_field():
    """STEM_TARGET_FIELDS is what the compiler is TOLD it may aim at, so a name
    that analyze_stem does not return would compile into a permanently
    unmeasurable (inf-distance) target."""
    from mix_engine.models import StemAnalysis

    for field in STEM_TARGET_FIELDS:
        assert field in StemAnalysis.model_fields, f"{field} is not on StemAnalysis"


def test_compiler_grammar_accepts_stem_targets():
    targets = compiled_targets(
        [{"metric": "stem(stem_a).sibilance_density", "max_value": 0.1}],
        ["stem_a", "stem_b"])
    assert targets[0].metric == "stem(stem_a).sibilance_density"


# ---------- the brief compiler scopes to the named stem (2026-08-05) ----------
#
# Measuring a per-stem target has worked since 2026-07-30; COMPILING one did not.
# "the piano is muddy" produced a session-wide mud ceiling and a master EQ cut,
# leaving the piano untouched — understood perfectly, applied to the wrong thing.

def test_a_named_stem_compiles_to_that_stems_bands():
    from orchestrator.targets import targets_from_brief
    targets = targets_from_brief(
        "the piano is muddy", ["piano", "drums"],
        {"mud_db": -3.0, "harshness_db": -20.0},
        stem_analysis={"band_energy": {"low_mid": -4.0}})
    assert [t.metric for t in targets] == ["stem(piano).band_energy.low_mid"]
    assert targets[0].max_value == -6.0     # current − COMPOSITE_DROP_DB


def test_an_unnamed_stem_still_compiles_session_wide():
    """The session composite must survive: "the mix is muddy" names nothing."""
    from orchestrator.targets import targets_from_brief
    targets = targets_from_brief("the mix is muddy", ["piano", "drums"],
                                 {"mud_db": -3.0, "harshness_db": -20.0})
    assert [t.metric for t in targets] == ["mud_db"]


def test_a_question_is_not_a_mix_instruction():
    from orchestrator.targets import BriefNotUnderstood, targets_from_brief
    with pytest.raises(BriefNotUnderstood):
        targets_from_brief("what does boxy mean", ["piano"], {"mud_db": -3.0})
    # ...but a polite REQUEST is not a question, even though it opens with a word
    # that reads like one. Only a LEADING wh-word is a question.
    from orchestrator.understanding import ANALYSIS
    assert targets_from_brief("can you make it warmer", ["piano"], ANALYSIS)


@pytest.mark.parametrize("brief", [
    "make it warmer?",              # an imperative with a question mark on the end
    "can you make it warmer?",      # a request frame, punctuation and all
    "could you make it warmer?",
    "how about a bit warmer?",
    "please make it warmer?",
])
def test_a_question_mark_does_not_by_itself_refuse_a_request(brief):
    """Punctuation must not be what decides (2026-08-05 review). The "?" rule fired on
    ANY trailing question mark, so a polite request was refused — and on the no-LLM
    path this compiler IS the router, so a refusal ships nothing at all."""
    from orchestrator.targets import targets_from_brief
    from orchestrator.understanding import ANALYSIS
    assert targets_from_brief(brief, ["piano"], ANALYSIS)


@pytest.mark.parametrize("brief", [
    "is the mix too muddy?",        # auxiliary-fronted — a real yes/no question
    "do you think this is too harsh?",
    "why is the vocal buried?",
    "should i make it warmer?",
])
def test_an_actual_question_still_refuses(brief):
    """The narrowing must not cost the original catch: a sentence SHAPED like a
    question (wh-word or auxiliary fronted) is still not a mix instruction."""
    from orchestrator.targets import BriefNotUnderstood, targets_from_brief
    from orchestrator.understanding import ANALYSIS
    with pytest.raises(BriefNotUnderstood):
        targets_from_brief(brief, ["piano"], ANALYSIS)


def test_a_stem_id_matches_as_a_whole_word_only():
    """A bare substring test let a stem called "a" match the "a" in "less harsh",
    which scoped every composite ask to a random stem."""
    from orchestrator.targets import resolve_stem
    assert resolve_stem("make it less harsh", ["a", "b"]) is None
    assert resolve_stem("make stem A louder", ["stem_a", "stem_b"]) == "stem_a"
    assert resolve_stem("the snare top is dull", ["snare_top"]) == "snare_top"


def test_the_share_solve_scales_with_band_dominance():
    """band_energy is a SHARE, so the cut needed to move it by a fixed amount grows
    without bound as the band approaches the whole signal. A fixed-size cut (the
    first attempt) took five passes to move 2 dB and only just fit the budget."""
    from orchestrator.agents import RuleAgent
    agent = RuleAgent()
    dominant = agent._share_gain_db(-1.0, -2.0)    # band is ~79% of the stem
    minor = agent._share_gain_db(-12.0, -2.0)      # band is ~6%
    assert dominant < minor < 0
    # Pin the exact values, not a band: a loose tolerance here would pass with
    # STEM_BAND_TAPER mis-set, which is the constant most likely to be fiddled with.
    assert dominant == pytest.approx(-8.35, abs=0.02)
    assert minor == pytest.approx(-3.01, abs=0.02)  # ~= the ideal -2/0.7 taper
    # An UNMEASURED band never reaches here — RuleAgent.propose requires
    # `target.current is not None` (a None current means distance == inf, which
    # produced a NaN prediction and a blind maximum cut).


def test_the_run_loop_scopes_a_named_stem_end_to_end(tmp_path):
    """The PRODUCTION path, not just the fixtures. `targets_from_brief` is pure, so
    loop.py has to resolve the stem and fetch its analysis itself before compiling —
    a step that lives only in the caller and would otherwise go untested."""
    import numpy as np
    import soundfile as sf
    from orchestrator.agents import RuleAgent
    from orchestrator.loop import run_session

    project = tmp_path / "named"
    project.mkdir()
    sr, t = 44100, np.arange(44100 * 2) / 44100.0
    # broadband, low-mids hot — a share can only move if there is other content
    hot = 0.20 * sum(np.sin(2 * np.pi * f * t) for f in (200.0, 300.0))
    rest = 0.05 * sum(np.sin(2 * np.pi * f * t) for f in (90.0, 700.0, 3000.0, 9000.0))
    sf.write(project / "piano.wav", (hot + rest).astype(np.float32), sr)
    sf.write(project / "bed.wav", (0.1 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32), sr)

    report = run_session(project, "the piano is muddy", gating="auto",
                         agent=RuleAgent(), connect=in_memory_engine)
    assert report.stop_reason.startswith("targets met"), report.stop_reason

    # The user-visible claim: the PIANO got the move. Before the fix this brief
    # compiled to a session mud_db ceiling and the RuleAgent cut the MASTER.
    from mix_engine.session import Session
    session = Session(project)
    piano = [e for e in session.applied_chain("piano") if e.get("type") == "eq"]
    assert piano, "the named stem was never touched"
    band = piano[0]["band"]
    assert 150 <= band["freq_hz"] <= 400, band["freq_hz"]
    assert band["gain_db"] < 0
    assert not [e for e in session.master_chain if e.get("type") == "eq"], \
        "a named-instrument brief must not EQ the master"


def test_a_named_stem_cut_intent_actually_cuts(tmp_path):
    """Direction, not just scope. `Target.distance` is UNSIGNED, so a per-stem GOAL
    target read straight off it always looked like a boost: "make the piano darker"
    lifted the exact bands it was asked to lower, for four passes.

    Neither standing gate could have caught this — the understanding eval scores
    scope only, and no benchmark case pairs a per-stem target with a cut direction.
    Hence this test.
    """
    import numpy as np
    import soundfile as sf
    from orchestrator.agents import RuleAgent
    from orchestrator.loop import run_session

    project = tmp_path / "darker"
    project.mkdir()
    sr, t = 44100, np.arange(44100 * 2) / 44100.0
    piano = 0.12 * sum(np.sin(2 * np.pi * f * t) for f in
                       (90., 200., 300., 700., 1400., 3000., 6000., 11000., 15000.))
    sf.write(project / "piano.wav", piano.astype(np.float32), sr)
    sf.write(project / "bed.wav", (0.1 * np.sin(2 * np.pi * 440. * t)).astype(np.float32), sr)

    report = run_session(project, "make the piano darker", gating="auto",
                         agent=RuleAgent(), connect=in_memory_engine)
    assert report.stop_reason.startswith("targets met"), report.stop_reason

    from mix_engine.session import Session
    eqs = [e for e in Session(project).applied_chain("piano") if e.get("type") == "eq"]
    assert eqs, "darker never touched the piano"
    for entry in eqs:
        band = entry["band"]
        assert band["gain_db"] < 0, f"darker LIFTED {band['freq_hz']} Hz: {band}"
        assert band["freq_hz"] > 1000, band["freq_hz"]
