"""Phase 10 (docs/05-roadmap.md): loop efficiency + hygiene.

Covers: get_state / refresh_stems round-trips, batched target measurement,
the render-count acceptance criterion (3 fixed master renders + 1 per applied
proposal — the old path did one per target), duck-key memoization, the
session-cumulative agent gain cap, v4 loudness parity, and in-place stem
import preserving the chat conversation.
"""
import shutil
from pathlib import Path

import numpy as np
import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from starlette.testclient import TestClient

from mix_engine import analysis, dsp
from mix_engine.mcp_server import create_server
from mix_engine.session import Session
from orchestrator.agents import RuleAgent
from orchestrator.chat import ScriptedChat
from orchestrator.gateway import create_app
from orchestrator.loop import EngineClient, run_session

from test_phase1_acceptance import call


def in_memory_engine(project_dir):
    return create_connected_server_and_client_session(create_server(project_dir))


def reset_render_counts() -> None:
    # Keep this helper aligned with every render class. A previous test can legitimately leave a
    # return render count behind, and a partial reset made the full suite order-dependent.
    for key in dsp.RENDER_COUNTS:
        dsp.RENDER_COUNTS[key] = 0


# ---------- get_state / refresh_stems ----------

@pytest.mark.anyio
async def test_get_state_is_a_cheap_read(demo_project):
    async with in_memory_engine(demo_project) as client:
        proposal = await call(client, "propose_actions", {"req": {
            "rationale": "state probe", "predicted_impact": [],
            "actions": [{"type": "gain", "stem_id": "stem_a", "delta_db": 2.0}]}})
        await call(client, "apply_proposal",
                   {"proposal_id": proposal["proposal_id"]})

        reset_render_counts()
        state = await call(client, "get_state")
        assert all(v == 0 for v in dsp.RENDER_COUNTS.values())  # no rendering of any kind

        stems = {s["stem_id"]: s for s in state["stems"]}
        assert set(stems) == {"stem_a", "stem_b"}
        assert stems["stem_a"]["gain_db"] == 2.0
        # chains match what analyze_stem reports, without the render
        analyzed = await call(client, "analyze_stem", {"stem_id": "stem_a"})
        assert stems["stem_a"]["applied_chain"] == analyzed["applied_chain"]
        assert state["pass_number"] == 1
        assert state["master_chain"][0]["type"] == "gain"


@pytest.mark.anyio
async def test_refresh_stems_discovers_new_wavs(demo_project):
    async with in_memory_engine(demo_project) as client:
        before = await call(client, "refresh_stems")
        assert before == ["stem_a", "stem_b"]
        shutil.copy(demo_project / "stem_a.wav", demo_project / "stem_c.wav")
        after = await call(client, "refresh_stems")
        assert after == ["stem_a", "stem_b", "stem_c"]
        state = await call(client, "get_state")
        assert any(s["stem_id"] == "stem_c" for s in state["stems"])


# ---------- batched measurement ----------

@pytest.mark.anyio
async def test_measure_all_uses_one_session_analysis(demo_project):
    async with in_memory_engine(demo_project) as client:
        engine = EngineClient(client)
        reset_render_counts()
        values = await engine.measure_all([
            "lufs_i", "true_peak_dbtp", "band_energy.mid", "band_energy.air",
            "masking(stem_a,stem_b).worst_band_excess_db",
        ])
        assert all(v is not None for v in values.values()), values
        # one master render serves the four session metrics; masking renders
        # stems, not the master
        assert dsp.RENDER_COUNTS["master"] == 1


# ---------- render-count acceptance ----------

class NoCriticRuleAgent(RuleAgent):
    """RuleAgent without the critic pass, to count one pass's renders only."""
    review = None


def synthesize_reference(project: Path) -> Path:
    mix, sample_rate = dsp.render_master(Session(project))
    import soundfile as sf
    louder = mix.mean(axis=1) * 10 ** (3.0 / 20.0)
    louder /= max(1.0, np.abs(louder).max() / 0.9)
    (project / "refs").mkdir(exist_ok=True)
    sf.write(project / "refs" / "reference.wav",
             louder.astype(np.float32), sample_rate)
    return project / "refs" / "reference.wav"


def test_reference_pass_render_budget(demo_project):
    """Acceptance: 3 fixed master renders per pass (initial bounce, one
    batched measurement, one post-apply bounce) + 1 baseline render per
    applied proposal. The pre-batching path was one render per target —
    ~15 for this brief."""
    synthesize_reference(demo_project)
    reset_render_counts()
    report = run_session(demo_project, "match refs/reference.wav",
                         gating="auto", agent=NoCriticRuleAgent(),
                         max_passes=1, connect=in_memory_engine)
    applied = len(report.applied_proposal_ids)
    assert applied >= 1  # the run actually did something
    assert dsp.RENDER_COUNTS["master"] <= 3 + applied, dsp.RENDER_COUNTS


# ---------- duck-key memoization ----------

def test_duck_key_renders_once_per_master_render(demo_project):
    shutil.copy(demo_project / "stem_b.wav", demo_project / "stem_c.wav")
    session = Session(demo_project)
    for target in ("stem_b", "stem_c"):  # both duck off the same key
        session.stems[target]["chain"].append({
            "type": "duck", "key_stem_id": "stem_a", "threshold_db": -30.0,
            "depth_db": 6.0, "attack_ms": 5.0, "release_ms": 150.0})
    reset_render_counts()
    dsp.render_master(session)
    # 3 mix stems + ONE shared duck-free key render (was 2 pre-memo)
    assert dsp.RENDER_COUNTS["stem"] == 4, dsp.RENDER_COUNTS


# ---------- session-cumulative agent gain cap ----------

@pytest.mark.anyio
async def test_session_gain_budget_caps_chat_style_moves(demo_project):
    """Each chat move is its own pass (advance_pass), so the per-pass clamp
    alone never binds; the session budget does — and undo refunds it."""
    Session(demo_project).user_edit("stem_a", {"gain_db": -12.0})  # the fader starts low: these moves test the gain WALLET, and a fader stops at +6 (test_fader_ceiling)
    async with in_memory_engine(demo_project) as client:
        async def gain_move(delta):
            result = await call(client, "propose_actions", {"req": {
                "rationale": f"chat move {delta:+}", "predicted_impact": [],
                "actions": [{"type": "gain", "stem_id": "stem_a",
                             "delta_db": delta}]}})
            if result["accepted"]:
                applied = await call(client, "apply_proposal",
                                     {"proposal_id": result["proposal_id"]})
                assert applied["ok"]
                await call(client, "advance_pass")
            return result

        first = await gain_move(6.0)    # session total 6
        second = await gain_move(6.0)   # session total 12 = at the limit
        third = await gain_move(6.0)    # would be 18 — over
        assert first["accepted"] and second["accepted"]
        assert not third["accepted"]
        assert third["violations"][0]["rule"] == "gain_per_stem_per_session_db"

        # undo refunds the budget: the same move is accepted again
        await call(client, "undo_proposal",
                   {"proposal_id": second["proposal_id"]})
        fourth = await gain_move(6.0)
        assert fourth["accepted"]


# ---------- v4 loudness parity ----------

def _v3_short_term_max(meter, audio, sample_rate) -> float:
    """The v3 definition: gated integrated loudness per sliding window."""
    window = int(3.0 * sample_rate)
    hop = int(0.1 * sample_rate)
    # the layout the engine measures: a 1-channel signal as it is PLAYED, dual-mono (analysis v7)
    values = [meter.integrated_loudness(analysis._lufs_arg(audio[s:s + window]))
              for s in range(0, audio.shape[0] - window + 1, hop)]
    return max(v for v in values if np.isfinite(v))


def test_short_term_loudness_parity_continuous():
    """On continuous signals (nothing for v3's per-window gating to drop)
    the v4 ungated short-term matches the v3 value to ±0.1 LU — this pins
    the K-weighting + windowing math itself."""
    import pyloudnorm as pyln

    sample_rate = 48000
    t = np.linspace(0, 12, 12 * sample_rate, endpoint=False)
    env = np.where(t < 6, 0.15, 0.4)  # level step, never near-silent
    sig = env * (np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 2200 * t))
    audio = np.stack([sig, 0.9 * sig], axis=1)
    meter = pyln.Meter(sample_rate)

    v4_max = analysis._short_term_max_lufs(meter, audio, sample_rate)
    assert v4_max == pytest.approx(_v3_short_term_max(meter, audio, sample_rate),
                                   abs=0.1)


def test_short_term_loudness_sane_on_program_material(demo_project):
    """On gappy program material the definitions legitimately differ (v3's
    per-window gating dropped quiet blocks, inflating windows that contain
    them) — bound the drift at 1 LU and keep the max ≥ integrated contract."""
    import pyloudnorm as pyln

    audio, sample_rate = dsp.render_master(Session(demo_project))
    meter = pyln.Meter(sample_rate)
    v4_max = analysis._short_term_max_lufs(meter, audio, sample_rate)
    assert v4_max == pytest.approx(
        _v3_short_term_max(meter, audio, sample_rate), abs=1.0)
    integrated = meter.integrated_loudness(analysis._lufs_arg(audio))
    assert round(v4_max, 1) >= round(integrated, 1)


# ---------- in-place stem import ----------

def test_stem_import_preserves_chat_and_feed(demo_project):
    app = create_app(demo_project, agent_factory=RuleAgent,
                     chat_factory=ScriptedChat)
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # snapshot
            ws.send_json({"type": "chat", "text": "how does the mix look?"})
            for _ in range(20):
                event = ws.receive_json()
                if (event["type"] == "chat_message"
                        and event["role"] == "assistant"):
                    break
            else:
                pytest.fail("no assistant chat reply")

            wav_bytes = (demo_project / "stem_a.wav").read_bytes()
            response = client.put("/api/stems/stem_c.wav", content=wav_bytes)
            assert response.status_code == 200, response.text

            for _ in range(20):
                event = ws.receive_json()
                if event["type"] == "snapshot" and len(event["stems"]) == 3:
                    break
            else:
                pytest.fail("no 3-stem snapshot after import")

            # the conversation survived the import (no project reopen)
            chat_events = [e for e in event["feed"]
                           if e["type"] == "chat_message"]
            assert any(e["role"] == "assistant" for e in chat_events)
            assert {s["stem_id"] for s in event["stems"]} == {
                "stem_a", "stem_b", "stem_c"}
