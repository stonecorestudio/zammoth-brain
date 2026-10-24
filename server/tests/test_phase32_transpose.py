"""Phase 32 — key transposition ("tune the song to a key")
(docs/05-roadmap.md Phase 32; the master transpose processor in the audio-dsp
skill). The user picks a target tonic; the AI transposes the whole song there
via a deterministic, undoable proposal."""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from pydantic import ValidationError

from mix_engine import dsp
from mix_engine.guardrails import LIMITS, validate_proposal
from mix_engine.models import ProposalRequest, TransposeAction
from mix_engine.session import MASTER_ACTION_TYPES, SUPPORTED_ACTION_TYPES, Session

from orchestrator.gateway import create_app, transpose_interval

from test_phase1_acceptance import call
from test_phase10_efficiency import in_memory_engine

SR = 48000


def _note(freq: float, dur: float = 3.0) -> np.ndarray:
    t = np.arange(int(SR * dur)) / SR
    return (np.sin(2 * np.pi * freq * t)
            + 0.5 * np.sin(2 * np.pi * 2 * freq * t)
            + 0.3 * np.sin(2 * np.pi * 3 * freq * t))


def _c_major_project(tmp_path: Path) -> Path:
    """A doubled-root C-major chord (the Phase 30 detector reads C maj)."""
    project = tmp_path / "cmaj"
    project.mkdir()
    chord = _note(130.81) + _note(261.63) + _note(329.63) + _note(392.00)
    sf.write(project / "stem.wav", (0.2 * chord / np.max(np.abs(chord))).astype(np.float32), SR)
    return project


def _peak_hz(audio: np.ndarray) -> float:
    mono = audio.mean(axis=1) if audio.ndim > 1 else audio
    spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
    return float(np.fft.rfftfreq(len(mono), 1 / SR)[np.argmax(spec)])


# ---------- 1. the nearest-interval helper ----------

@pytest.mark.parametrize("detected,target,expect", [
    ("G", "A", 2), ("G", "F", -2), ("B", "C", 1), ("C", "B", -1),
    ("A", "A", 0), ("C", "F#", 6), ("C", "G", -5), ("C", "G#", -4),
])
def test_transpose_interval_is_nearest(detected, target, expect):
    assert transpose_interval(detected, target) == expect


# ---------- 2. the transpose action: surface + schema + guardrail ----------

def test_transpose_is_master_only_supported_action():
    assert "transpose" in SUPPORTED_ACTION_TYPES
    assert "transpose" in MASTER_ACTION_TYPES


def test_transpose_schema_bounds():
    TransposeAction(stem_id="master", semitones=12)
    TransposeAction(stem_id="master", semitones=-12)
    with pytest.raises(ValidationError):
        TransposeAction(stem_id="master", semitones=0)     # no-op
    with pytest.raises(ValidationError):
        TransposeAction(stem_id="master", semitones=15)    # out of range


def test_transpose_guardrail_is_defense_in_depth():
    bypass = ProposalRequest.model_construct(
        rationale="x", predicted_impact=[],
        actions=[TransposeAction.model_construct(type="transpose",
                                                 stem_id="master", semitones=15)])
    rules = [v.rule for v in validate_proposal(bypass, {}, stem_chains={})]
    assert "transpose_semitones_max" in rules
    assert LIMITS["transpose_semitones_max"] == 12


# ---------- 3. the transpose render (master pitch shift) ----------

def test_transpose_shifts_master_pitch(tmp_path):
    project = tmp_path / "tone"
    project.mkdir()
    t = np.arange(int(SR * 3)) / SR
    sf.write(project / "stem.wav", (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), SR)
    session = Session(project)
    base, _ = dsp.render_master(session)
    assert abs(_peak_hz(base) - 220.0) < 3.0

    session.apply(session.propose(ProposalRequest(
        rationale="up a tone", predicted_impact=[],
        actions=[TransposeAction(stem_id="master", semitones=2)])).proposal_id)
    up, _ = dsp.render_master(session)
    assert abs(_peak_hz(up) - 220.0 * 2 ** (2 / 12)) < 8.0   # ~246.9 Hz
    again, _ = dsp.render_master(session)
    assert np.array_equal(up, again)                          # deterministic
    assert up.shape[0] == base.shape[0]                       # length preserved


@pytest.mark.anyio
async def test_transpose_deterministic_and_undoable(demo_project):
    async with in_memory_engine(demo_project) as client:
        baseline = await call(client, "render_and_analyze")
        bounce = demo_project / baseline["bounce_path"]
        baseline_bytes = bounce.read_bytes()

        proposal = await call(client, "propose_actions", {"req": {
            "rationale": "transpose +3", "predicted_impact": [],
            "actions": [{"type": "transpose", "stem_id": "master", "semitones": 3}]}})
        assert proposal["accepted"], proposal["violations"]
        await call(client, "apply_proposal", {"proposal_id": proposal["proposal_id"]})
        await call(client, "render_and_analyze")
        changed = bounce.read_bytes()
        assert changed != baseline_bytes
        await call(client, "render_and_analyze")
        assert bounce.read_bytes() == changed                 # deterministic
        await call(client, "undo_proposal", {"proposal_id": proposal["proposal_id"]})
        await call(client, "render_and_analyze")
        assert bounce.read_bytes() == baseline_bytes          # bit-identical undo


@pytest.mark.anyio
async def test_transpose_replaces_not_stacks(demo_project):
    async with in_memory_engine(demo_project) as client:
        for semis in (2, -3):
            proposal = await call(client, "propose_actions", {"req": {
                "rationale": "transpose", "predicted_impact": [],
                "actions": [{"type": "transpose", "stem_id": "master", "semitones": semis}]}})
            assert proposal["accepted"], proposal["violations"]
            await call(client, "apply_proposal", {"proposal_id": proposal["proposal_id"]})
        state = await call(client, "get_state")
        tr = [e for e in state["master_chain"] if e["type"] == "transpose"]
        assert len(tr) == 1 and tr[0]["semitones"] == -3      # latest wins


@pytest.mark.anyio
async def test_transpose_refused_on_stem(demo_project):
    async with in_memory_engine(demo_project) as client:
        proposal = await call(client, "propose_actions", {"req": {
            "rationale": "transpose one stem (should be refused)",
            "predicted_impact": [],
            "actions": [{"type": "transpose", "stem_id": "stem_a", "semitones": 2}]}})
    assert not proposal["accepted"]
    assert any(v["rule"] == "transpose_master_only" for v in proposal["violations"])


# ---------- 4. the key-picker (gateway → deterministic proposal) ----------

def _snapshot_then(send: dict, project: Path):
    """Connect, drain to the first snapshot, send a command, return the events."""
    from starlette.testclient import TestClient

    from orchestrator.agents import RuleAgent
    from orchestrator.chat import ScriptedChat

    app = create_app(project, agent_factory=RuleAgent, chat_factory=ScriptedChat)
    events: list[dict] = []
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            snap = ws.receive_json()
            for _ in range(50):
                if snap.get("type") == "snapshot":
                    break
                snap = ws.receive_json()
            ws.send_json(send)
            for _ in range(80):
                events.append(ws.receive_json())
                if events[-1]["type"] in ("proposal", "error"):
                    break
    return snap, events


def test_key_picker_builds_a_transpose_proposal(tmp_path):
    snap, events = _snapshot_then(
        {"type": "transpose_to_key", "tonic": "D"}, _c_major_project(tmp_path))
    assert snap["metrics"]["key"]["tonic"] == "C"              # detected basis
    proposal = next(e for e in events if e["type"] == "proposal")
    action = proposal["actions"][0]
    assert action["type"] == "transpose" and action["stem_id"] == "master"
    assert action["semitones"] == 2                            # C -> D = +2


def test_key_picker_without_a_key_errors(tmp_path):
    project = tmp_path / "noise"
    project.mkdir()
    noise = 0.2 * np.random.default_rng(3).standard_normal(SR * 3)
    sf.write(project / "stem.wav", noise.astype(np.float32), SR)
    snap, events = _snapshot_then({"type": "transpose_to_key", "tonic": "D"}, project)
    assert snap["metrics"]["key"] is None
    assert any(e["type"] == "error" for e in events)
    assert not any(e["type"] == "proposal" for e in events)
