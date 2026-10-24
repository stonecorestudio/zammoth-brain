"""Phase 31 — tuning detection + global tuning correction
(docs/05-roadmap.md Phase 31; metric definitions in the mix-analysis skill,
the tune processor in the audio-dsp skill)."""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from pydantic import ValidationError

from mix_engine import analysis, dsp
from mix_engine.guardrails import LIMITS, validate_proposal
from mix_engine.models import ProposalRequest, TuneAction
from mix_engine.session import SUPPORTED_ACTION_TYPES, Session

from test_phase1_acceptance import call
from test_phase10_efficiency import in_memory_engine

SR = 48000


def _note(freq: float, dur: float = 3.0, sr: int = SR) -> np.ndarray:
    """A tone with two overtones — a real-instrument-ish chroma (as in the
    Phase 30 tests), so the tuning histogram has stable peaks to lock onto."""
    t = np.arange(int(sr * dur)) / sr
    return (np.sin(2 * np.pi * freq * t)
            + 0.5 * np.sin(2 * np.pi * 2 * freq * t)
            + 0.3 * np.sin(2 * np.pi * 3 * freq * t))


def _detuned_chord(cents: float, dur: float = 3.0) -> np.ndarray:
    """C-major triad, every note shifted uniformly by `cents` — a coherently
    flat/sharp source, exactly what a global tuning correction is for."""
    f = 2 ** (cents / 1200.0)
    x = _note(261.63 * f, dur) + _note(329.63 * f, dur) + _note(392.00 * f, dur)
    return (0.2 * x / np.max(np.abs(x))).astype(np.float32)


def _detuned_project(tmp_path: Path, cents: float) -> Path:
    project = tmp_path / f"detuned_{int(round(cents))}"
    project.mkdir()
    sf.write(project / "stem.wav", _detuned_chord(cents), SR)
    return project


# ---------- 1. detection (analysis._tuning) ----------

def test_tuning_detects_flat():
    offset, coherence = analysis._tuning(_detuned_chord(-30.0), SR)
    assert abs(offset - (-30.0)) < 5.0, offset      # harmonic bias is a few cents
    assert coherence > 0.8                          # a clean source is coherent


def test_tuning_detects_sharp():
    offset, coherence = analysis._tuning(_detuned_chord(22.0), SR)
    assert abs(offset - 22.0) < 5.0, offset
    assert coherence > 0.8


def test_tuning_in_tune_reads_near_zero():
    offset, coherence = analysis._tuning(_detuned_chord(0.0), SR)
    assert abs(offset) < 5.0, offset
    assert coherence > 0.8


def test_tuning_none_on_noise():
    noise = 0.2 * np.random.default_rng(7).standard_normal(SR * 3)
    assert analysis._tuning(noise, SR) is None


def test_tuning_none_on_silence():
    assert analysis._tuning(np.zeros(SR * 2), SR) is None


def test_tuning_deterministic():
    sig = _detuned_chord(-18.0)
    assert analysis._tuning(sig, SR) == analysis._tuning(sig, SR)


def test_chroma_refactor_preserves_key():
    """The shared _pitch_stft front-end must not change Phase 30 key output —
    same doubled-root C-major chord the Phase 30 suite disambiguates with."""
    chord = _note(130.81) + _note(261.63) + _note(329.63) + _note(392.00)
    key = analysis._key(chord, SR)
    assert key is not None and key.tonic == "C" and key.mode == "maj"


# ---------- 2. analysis wiring (stem + session fields) ----------

def test_analyze_stem_carries_tuning(tmp_path):
    project = _detuned_project(tmp_path, -30.0)
    session = Session(project)
    audio, sr = dsp.render_stem(session, "stem")
    result = analysis.analyze_stem(audio, sr, stem_id="stem",
                                   applied_chain=session.applied_chain("stem"))
    assert result.tuning_offset_cents is not None
    assert abs(result.tuning_offset_cents - (-30.0)) < 5.0
    assert 0.0 <= result.tuning_coherence <= 1.0


def test_analyze_master_carries_tuning(demo_project: Path):
    session = Session(demo_project)
    audio, sr = dsp.render_master(session)
    result = analysis.analyze_master(audio, sr, list(session.stems), 0)
    assert (result.tuning_offset_cents is None
            or isinstance(result.tuning_offset_cents, float))
    again = analysis.analyze_master(audio, sr, list(session.stems), 0)
    assert again.tuning_offset_cents == result.tuning_offset_cents  # deterministic


def test_analyze_stem_silent_has_no_tuning():
    silent = np.zeros((SR * 2, 1))
    result = analysis.analyze_stem(silent, SR, stem_id="s", applied_chain=[])
    assert result.silent is True
    assert result.tuning_offset_cents is None
    assert result.tuning_coherence is None


# ---------- 3. the tune action: surface + state ----------

def test_tune_is_a_supported_stem_action():
    assert "tune" in SUPPORTED_ACTION_TYPES


def test_tune_schema_bounds():
    TuneAction(stem_id="s", cents=100.0)            # a semitone is allowed
    TuneAction(stem_id="s", cents=-100.0)
    with pytest.raises(ValidationError):
        TuneAction(stem_id="s", cents=150.0)


def test_tune_guardrail_is_defense_in_depth():
    """Schema blocks |cents|>100; the guardrail catches it too if a bound drifts."""
    bypass = ProposalRequest.model_construct(
        rationale="x", predicted_impact=[],
        actions=[TuneAction.model_construct(type="tune", stem_id="s", cents=150.0)])
    rules = [v.rule for v in validate_proposal(bypass, {}, stem_chains={})]
    assert "tune_cents_max" in rules
    assert LIMITS["tune_cents_max"] == 100.0


# ---------- 4. correction round-trip (real tool surface) ----------

@pytest.mark.anyio
async def test_tune_corrects_offset_and_clears_flag(tmp_path):
    project = _detuned_project(tmp_path, -35.0)
    async with in_memory_engine(project) as client:
        before = await call(client, "render_and_analyze")
        assert "out_of_tune" in before["flags"]
        stem_before = await call(client, "analyze_stem", {"stem_id": "stem"})
        assert stem_before["tuning_offset_cents"] < -25.0

        proposal = await call(client, "propose_actions", {"req": {
            "rationale": "tune the flat source onto concert pitch",
            "predicted_impact": [{"metric": "stem(stem).tuning_offset_cents",
                                  "predicted_delta": 35.0, "unit": "cents"}],
            "actions": [{"type": "tune", "stem_id": "stem", "cents": 35.0}]}})
        assert proposal["accepted"], proposal["violations"]
        assert (await call(client, "apply_proposal",
                           {"proposal_id": proposal["proposal_id"]}))["ok"]

        after = await call(client, "render_and_analyze")
        stem_after = await call(client, "analyze_stem", {"stem_id": "stem"})
        assert abs(stem_after["tuning_offset_cents"]) < 8.0, stem_after
        assert "out_of_tune" not in after["flags"]


@pytest.mark.anyio
async def test_tune_deterministic_and_undoable(demo_project):
    async with in_memory_engine(demo_project) as client:
        baseline = await call(client, "render_and_analyze")
        bounce = demo_project / baseline["bounce_path"]
        baseline_bytes = bounce.read_bytes()

        proposal = await call(client, "propose_actions", {"req": {
            "rationale": "tune", "predicted_impact": [],
            "actions": [{"type": "tune", "stem_id": "stem_a", "cents": 20.0}]}})
        assert proposal["accepted"], proposal["violations"]
        await call(client, "apply_proposal",
                   {"proposal_id": proposal["proposal_id"]})
        await call(client, "render_and_analyze")
        changed = bounce.read_bytes()
        assert changed != baseline_bytes
        await call(client, "render_and_analyze")
        assert bounce.read_bytes() == changed                  # deterministic

        await call(client, "undo_proposal",
                   {"proposal_id": proposal["proposal_id"]})
        await call(client, "render_and_analyze")
        assert bounce.read_bytes() == baseline_bytes           # bit-identical undo


@pytest.mark.anyio
async def test_tune_replaces_not_stacks(demo_project):
    async with in_memory_engine(demo_project) as client:
        for cents in (10.0, -15.0):
            proposal = await call(client, "propose_actions", {"req": {
                "rationale": "adjust tuning", "predicted_impact": [],
                "actions": [{"type": "tune", "stem_id": "stem_a", "cents": cents}]}})
            assert proposal["accepted"], proposal["violations"]
            await call(client, "apply_proposal",
                       {"proposal_id": proposal["proposal_id"]})
        state = await call(client, "get_state")
        stem = next(s for s in state["stems"] if s["stem_id"] == "stem_a")
        tunes = [e for e in stem["applied_chain"] if e["type"] == "tune"]
        assert len(tunes) == 1 and tunes[0]["cents"] == -15.0   # latest wins


@pytest.mark.anyio
async def test_tune_refused_on_master(demo_project):
    async with in_memory_engine(demo_project) as client:
        proposal = await call(client, "propose_actions", {"req": {
            "rationale": "retune the whole mix (should be refused)",
            "predicted_impact": [],
            "actions": [{"type": "tune", "stem_id": "master", "cents": 10.0}]}})
    assert not proposal["accepted"]
    assert any(v["rule"] == "master_action_type" for v in proposal["violations"])


# ---------- 5. UI data path (the gateway snapshot the React UI reads) ----------

def test_gateway_snapshot_carries_tuning(tmp_path):
    """The UI data path: the transport chip reads `metrics.tuning_offset_cents`
    and the per-stem out-of-tune badge reads the stem view — confirm the gateway
    snapshot ships both. (The `out_of_tune` flag is a post-render flag like every
    other flag; its plumbing is covered by test_tune_corrects_offset_and_clears_flag.)"""
    from starlette.testclient import TestClient

    from orchestrator.agents import RuleAgent
    from orchestrator.chat import ScriptedChat
    from orchestrator.gateway import create_app

    project = _detuned_project(tmp_path, -35.0)
    app = create_app(project, agent_factory=RuleAgent, chat_factory=ScriptedChat)
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            snapshot = ws.receive_json()
            for _ in range(50):
                if snapshot.get("type") == "snapshot":
                    break
                snapshot = ws.receive_json()
    assert snapshot["type"] == "snapshot"
    assert snapshot["metrics"]["tuning_offset_cents"] < -25.0        # song-level chip
    stem = next(s for s in snapshot["stems"] if s["stem_id"] == "stem")
    assert stem["tuning_offset_cents"] < -25.0                       # per-stem badge
    assert stem["tuning_coherence"] is not None


# ---------- 6. probe-ability (impact.py resolves the new metric) ----------

@pytest.mark.anyio
async def test_tune_is_probeable(demo_project):
    """The agent can audition a tune and predict the offset move, with no
    impact.py change — the resolver reads any numeric model attribute."""
    async with in_memory_engine(demo_project) as client:
        probe = await call(client, "probe_actions", {"req": {
            "actions": [{"type": "tune", "stem_id": "stem_a", "cents": 30.0}],
            "metrics": ["stem(stem_a).tuning_offset_cents"]}})
        measurement = probe["measurements"][0]
        assert measurement["before"] is not None
        assert abs(measurement["delta"] - 30.0) < 3.0, measurement   # tracks the shift
