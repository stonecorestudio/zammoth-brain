"""Phase 14 (docs/05-roadmap.md): perception — auto-tagging + composites.

Covers: archetype stems classify correctly (user tags never clobbered, and a
"bass" auto-tag tightens the HPF guardrail), the composite metrics move when
the offending region is boosted, mastering QC metrics round-trip (PLR/PSR,
per-band correlation, the mono-compatibility flag), "make it less harsh"
running end to end on the deterministic path, and pass-to-pass metric deltas
reaching the agent context.
"""
import numpy as np
import pytest
import soundfile as sf

from orchestrator.agents import RuleAgent
from orchestrator.loop import run_session
from orchestrator.targets import targets_from_brief

from test_phase1_acceptance import call
from test_phase10_efficiency import in_memory_engine

from mix_engine.models import ANALYSIS_VERSION

SAMPLE_RATE = 48000
DURATION_S = 8.0


def _t() -> np.ndarray:
    return np.arange(int(DURATION_S * SAMPLE_RATE)) / SAMPLE_RATE


def _bursts(freqs, rate_hz: float, decay_s: float, amp: float) -> np.ndarray:
    """Deterministic tone bursts: `rate_hz` onsets/s, exponential decay."""
    t = _t()
    period = 1.0 / rate_hz
    phase = t % period
    envelope = np.exp(-phase / decay_s)
    return amp * envelope * sum(np.sin(2 * np.pi * f * t) for f in freqs)


def write_archetypes(project) -> dict[str, str]:
    """Synthesized stems with known classes (the acceptance fixture)."""
    t = _t()
    stems = {
        # 55 Hz thumps, sharp decay: low-dominated AND punchy
        "kick": _bursts([55.0], rate_hz=2.0, decay_s=0.05, amp=0.8),
        # sustained 80 Hz: low-dominated, no onsets
        "bass": 0.5 * np.sin(2 * np.pi * 80.0 * t),
        # bright fast bursts: high-dominated and transient
        "cymbal": _bursts([6000.0, 9000.0, 12000.0], rate_hz=3.0,
                          decay_s=0.08, amp=0.4),
        # mid tone + periodic 7 kHz "ess" bursts: mids + sibilance
        "vocal": (0.4 * np.sin(2 * np.pi * 800.0 * t)
                  + _bursts([7000.0], rate_hz=1.5, decay_s=0.08, amp=0.6)),
        # sustained low-mid chord: nothing transient anywhere
        "pad": 0.3 * (np.sin(2 * np.pi * 220.0 * t)
                      + np.sin(2 * np.pi * 277.0 * t)
                      + np.sin(2 * np.pi * 330.0 * t)),
    }
    for name, audio in stems.items():
        sf.write(project / f"{name}.wav", audio.astype(np.float32), SAMPLE_RATE)
    return {name: name for name in stems}


# ---------- auto-tagging ----------

@pytest.mark.anyio
async def test_archetype_stems_classify(tmp_path):
    project = tmp_path / "archetypes"
    project.mkdir()
    write_archetypes(project)
    async with in_memory_engine(project) as client:
        tags = await call(client, "auto_tag_stems")
    assert tags["kick"] == ["kick"]
    assert tags["bass"] == ["bass"]
    assert tags["cymbal"] == ["cymbal"]
    assert tags["vocal"] == ["vocal"]
    assert tags["pad"] == ["pad"]


@pytest.mark.anyio
async def test_user_tags_win_and_bass_tag_tightens_hpf(tmp_path):
    project = tmp_path / "archetypes"
    project.mkdir()
    write_archetypes(project)
    # the user already called this stem something else (before open): the
    # classifier must never clobber it
    from mix_engine.session import Session
    Session(project).set_stem_tags("bass", ["lead"])
    async with in_memory_engine(project) as client:
        tags = await call(client, "auto_tag_stems")
        assert tags["bass"] == ["lead"]
        assert tags["kick"] == ["kick"]

    # and the auto-tag is a real guardrail input: a "bass" auto-tag
    # tightens the HPF limit to 40 Hz
    project_2 = tmp_path / "auto"
    project_2.mkdir()
    write_archetypes(project_2)
    async with in_memory_engine(project_2) as client:
        await call(client, "auto_tag_stems")
        result = await call(client, "propose_actions", {"req": {
            "rationale": "thin out the bass", "predicted_impact": [],
            "actions": [{"type": "eq", "stem_id": "bass",
                         "band": {"freq_hz": 80, "gain_db": 0.0, "q": 0.7,
                                  "shape": "hpf"}}]}})
        assert not result["accepted"]
        assert result["violations"][0]["rule"] == "hpf_max_hz"
        assert "bass-tagged" in result["violations"][0]["message"]


# ---------- composites ----------

@pytest.mark.anyio
async def test_composites_present_and_move_with_eq(demo_project):
    async with in_memory_engine(demo_project) as client:
        before = await call(client, "analyze_session")
        assert before["analysis_version"] == ANALYSIS_VERSION
        for metric in ("harshness_db", "mud_db", "clarity_index",
                       "plr_db", "psr_db"):
            assert before[metric] is not None, metric

        async def boost(freq_hz, gain_db=6.0):
            proposal = await call(client, "propose_actions", {"req": {
                "rationale": f"boost {freq_hz}", "predicted_impact": [],
                "actions": [{"type": "eq", "stem_id": "master",
                             "band": {"freq_hz": freq_hz, "gain_db": gain_db,
                                      "q": 0.8, "shape": "peak"}}]}})
            await call(client, "apply_proposal",
                       {"proposal_id": proposal["proposal_id"]})
            return await call(client, "analyze_session")

        harsher = await boost(2800)
        assert harsher["harshness_db"] > before["harshness_db"]
        # A new pass: two broad +6 bells in ONE pass compose past the +6/pass EQ rail (2026-09-11) —
        # the 250 Hz bell's skirt lifts 2.8 kHz ~0.1 dB above the first. This test is about the
        # perception metrics, not the rail, so each boost gets its own pass.
        await call(client, "advance_pass", {})
        muddier = await boost(250)
        assert muddier["mud_db"] > harsher["mud_db"]


# ---------- mastering QC ----------

@pytest.mark.anyio
async def test_qc_metrics_and_mono_flag(tmp_path):
    project = tmp_path / "stereo"
    project.mkdir()
    t = _t()
    # anti-phase 500 Hz: cancels exactly on mono fold-down
    wide = np.stack([0.4 * np.sin(2 * np.pi * 500.0 * t),
                     -0.4 * np.sin(2 * np.pi * 500.0 * t)], axis=1)
    sf.write(project / "wide.wav", wide.astype(np.float32), SAMPLE_RATE)
    anchor = 0.2 * np.sin(2 * np.pi * 220.0 * t)
    sf.write(project / "anchor.wav",
             np.stack([anchor, anchor], axis=1).astype(np.float32), SAMPLE_RATE)

    async with in_memory_engine(project) as client:
        result = await call(client, "render_and_analyze")
        analysis = result["analysis"]
        assert analysis["plr_db"] == pytest.approx(
            analysis["true_peak_dbtp"] - analysis["lufs_i"], abs=0.11)
        assert analysis["psr_db"] == pytest.approx(
            analysis["true_peak_dbtp"] - analysis["lufs_s_max"], abs=0.11)
        # the anti-phase band reads strongly negative; the mono anchor +1
        assert analysis["band_correlation"]["mid"] < -0.9
        assert analysis["band_correlation"]["low_mid"] > 0.9
        assert "mono_compatibility_risk" in result["flags"]


# ---------- "make it less harsh" end to end ----------

def write_harsh_project(project) -> None:
    t = _t()
    # constant (squashed) 3 / 6 kHz content over a 300 Hz anchor.
    # The stem is "fizz", not "harsh": a stem id is resolved out of the brief, so a
    # stem literally called "harsh" makes "make it less harsh" a SESSION-vs-STEM
    # ambiguity no matcher can settle, and this case is about the session composite.
    fizz = (0.30 * np.sin(2 * np.pi * 3000.0 * t)
            + 0.20 * np.sin(2 * np.pi * 6000.0 * t))
    sf.write(project / "fizz.wav", fizz.astype(np.float32), SAMPLE_RATE)
    anchor = 0.4 * np.sin(2 * np.pi * 300.0 * t)
    sf.write(project / "anchor.wav", anchor.astype(np.float32), SAMPLE_RATE)


def test_less_harsh_brief_runs_end_to_end(tmp_path):
    """Acceptance: the composite ask works on the deterministic no-LLM path
    (regex table + RuleAgent), not just through the brief compiler."""
    project = tmp_path / "harsh"
    project.mkdir()
    write_harsh_project(project)

    from mix_engine import analysis as engine_analysis
    from mix_engine import dsp
    from mix_engine.session import Session
    audio, sr = dsp.render_master(Session(project))
    initial = engine_analysis.analyze_master(audio, sr, stems=[], pass_number=1)

    report = run_session(project, "make it less harsh", gating="auto",
                         agent=RuleAgent(), connect=in_memory_engine)
    assert report.stop_reason.startswith("targets met"), report.stop_reason
    assert report.applied_proposal_ids
    assert report.final_metrics["harshness_db"] <= initial.harshness_db - 2.0


def test_composite_targets_need_current_metrics():
    # the regex path only forms composite targets when it can anchor them
    with pytest.raises(ValueError):
        targets_from_brief("make it less harsh", ["a", "b"])  # no analysis
    (target,) = targets_from_brief("make it less harsh", ["a"],
                                   {"harshness_db": -20.0})
    assert target.metric == "harshness_db"
    assert target.max_value == pytest.approx(-22.0)


# ---------- pass-to-pass deltas ----------

class DeltaRecorder(RuleAgent):
    def __init__(self):
        self.deltas = []

    def propose(self, ctx):
        self.deltas.append(ctx.metrics_delta)
        return super().propose(ctx)


def test_metrics_delta_reaches_the_context(tmp_path):
    project = tmp_path / "harsh"
    project.mkdir()
    write_harsh_project(project)
    agent = DeltaRecorder()
    run_session(project, "make it less harsh", gating="auto", agent=agent,
                connect=in_memory_engine)
    assert len(agent.deltas) >= 2
    assert agent.deltas[0] is None  # nothing to compare on pass 1
    second = agent.deltas[1]
    assert second is not None
    assert "harshness_db" in second  # the cut showed up as a delta
    assert second["harshness_db"] < 0
    assert "band_energy" in second
