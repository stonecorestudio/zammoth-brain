"""A run that ends worse than a pass it already reached puts itself back
(2026-08-05).

The stall counter treated "this pass made it worse" exactly like "this pass did
nothing": both increment `stall`, so two worsening passes exited with
`stop_reason = "diminishing returns"` — a success-shaped ending — and left the
worse mix applied. The user was told the agent was done and handed a mix worse
than the one it had already reached.

These tests pin the fix and, just as importantly, pin that an ordinary FLAT
stall still ends the old way with nothing undone.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from mcp.shared.memory import create_connected_server_and_client_session

from mix_engine.mcp_server import create_server
from orchestrator.agents import PassContext
from orchestrator.loop import run_session


def in_memory_engine(project_dir):
    return create_connected_server_and_client_session(create_server(project_dir))


@pytest.fixture()
def loud_project(tmp_path: Path) -> Path:
    project = tmp_path / "regress"
    project.mkdir()
    sr = 44100
    t = np.arange(sr * 2) / sr
    # The same mix as 0.20 / 0.10 at unity, written 6 dB hotter with both faders 6 dB down: the
    # scripted walk lifts the lead 11 dB, and a fader stops at +6 (2026-10-05, test_fader_ceiling).
    sf.write(project / "lead.wav", (0.40 * np.sin(2 * np.pi * 330 * t)).astype("float32"), sr)
    sf.write(project / "bed.wav", (0.20 * np.sin(2 * np.pi * 110 * t)).astype("float32"), sr)
    from mix_engine.session import Session
    session = Session(project)
    for stem in ("lead", "bed"):
        session.user_edit(stem, {"gain_db": -6.0})
    return project


class ScriptedAgent:
    """Applies a fixed gain move per pass. The script drives the mix toward the
    loudness goal and then deliberately past it, which is a real regression the
    engine measures rather than one the test asserts into existence."""

    def __init__(self, deltas: list[float]):
        self.deltas = deltas
        self.pass_index = 0

    def propose(self, ctx: PassContext) -> list[dict]:
        if self.pass_index >= len(self.deltas):
            return []
        delta = self.deltas[self.pass_index]
        self.pass_index += 1
        return [{
            "rationale": f"scripted {delta:+.1f} dB",
            "predicted_impact": [{"metric": "lufs_i", "predicted_delta": delta,
                                  "unit": "LUFS"}],
            "actions": [{"type": "gain", "stem_id": "lead", "delta_db": delta}],
        }]

    def revise(self, proposal, violations, ctx):
        return None


# The fixture renders at -17.0 LUFS, true-peak -12.0, PLR 5.0. Against a -12 LUFS
# goal (tolerance 0.5) the scripted +4/+4/+3 walk produces lufs_i distances
# 4.50 -> 0.90 -> 1.90 -> 4.80: pass 1 genuinely improves, passes 2 and 3 genuinely
# worsen. The engine measures that; the test does not assert it into being. True-peak
# stays under the -1 dBTP ceiling (-2.4 at the end) and PLR stays over its 4.0 floor
# throughout, so those two targets hold at distance 0 and the ranking is driven by
# loudness alone. Total gain 11 dB, inside the 12 dB per-stem session cap — go over
# and the third proposal is REJECTED rather than applied, which is a different test.


def test_a_regressing_run_reverts_to_its_best_pass(loud_project):
    """Up toward the goal, then two passes that blow past it."""
    agent = ScriptedAgent([+4.0, +4.0, +3.0])
    report = run_session(loud_project, "master to -12 LUFS", gating="auto",
                         agent=agent, connect=in_memory_engine)

    assert report.reverted_proposal_ids, (
        f"nothing reverted; stop_reason={report.stop_reason!r}")
    assert report.stop_reason.startswith("regressed"), report.stop_reason
    assert "reverted to" in report.stop_reason

    # The two lists must not overlap: applied means STILL STANDING (the DAW's
    # ledger and whole-run Undo both depend on that).
    assert not (set(report.applied_proposal_ids)
                & set(report.reverted_proposal_ids))

    # The engine agrees: every reverted proposal is 'undone' on disk.
    state = json.loads((loud_project / "session.json").read_text())
    status = {p["proposal_id"]: p["status"] for p in state["proposals"]}
    for proposal_id in report.reverted_proposal_ids:
        assert status[proposal_id] == "undone", proposal_id
    for proposal_id in report.applied_proposal_ids:
        assert status[proposal_id] == "applied", proposal_id


def test_the_reverted_state_is_actually_better(loud_project):
    """The point of the feature: what the user is left with beats what they
    would have been left with. Measured, not asserted."""
    agent = ScriptedAgent([+4.0, +4.0, +3.0])
    report = run_session(loud_project, "master to -12 LUFS", gating="auto",
                         agent=agent, connect=in_memory_engine)
    assert report.reverted_proposal_ids

    kept = abs(report.final_metrics["lufs_i"] - (-12.0))
    # Re-run without rollback reach: the same script, stopped at its worst.
    worst = abs(_lufs_after(loud_project.parent, [+4.0, +4.0, +3.0]) - (-12.0))
    assert kept < worst, f"kept {kept:.2f} LUFS off goal vs unreverted {worst:.2f}"


def _lufs_after(parent: Path, deltas: list[float]) -> float:
    """The loudness the run WOULD have left behind with no rollback — built by
    applying the same moves directly, so the comparison is against reality."""
    from mix_engine.models import GainAction, ProposalRequest
    from mix_engine.session import Session
    from mix_engine import analysis as engine_analysis, dsp

    project = parent / "unreverted"
    project.mkdir()
    sr = 44100
    t = np.arange(sr * 2) / sr
    sf.write(project / "lead.wav", (0.40 * np.sin(2 * np.pi * 330 * t)).astype("float32"), sr)
    sf.write(project / "bed.wav", (0.20 * np.sin(2 * np.pi * 110 * t)).astype("float32"), sr)

    session = Session(project)
    for stem in ("lead", "bed"):   # the loud_project fixture's own staging: 6 dB hotter, 6 dB down
        session.user_edit(stem, {"gain_db": -6.0})
    for delta in deltas:
        result = session.propose(ProposalRequest(
            rationale="x", predicted_impact=[],
            actions=[GainAction(stem_id="lead", delta_db=delta)]))
        session.apply(result.proposal_id)
        session.advance_pass()
    audio, rate = dsp.render_master(session)
    return engine_analysis.analyze_master(audio, rate, stems=[], pass_number=1).lufs_i


def test_a_flat_stall_still_ends_the_old_way(loud_project):
    """The guard must fire on WORSE, not on 'no progress'. A run that simply
    stops improving keeps its benign ending and touches nothing — this is what
    stops the feature from rewriting every ordinary run's outcome."""
    agent = ScriptedAgent([0.0, 0.0, 0.0])
    report = run_session(loud_project, "master to -12 LUFS", gating="auto",
                         agent=agent, connect=in_memory_engine)
    assert not report.reverted_proposal_ids, report.stop_reason
    assert not report.stop_reason.startswith("regressed"), report.stop_reason


def test_final_metrics_match_the_kept_mix(loud_project):
    """final_metrics must describe the mix the user is actually left with. The
    rollback re-renders precisely so the report, the bounce and the session agree
    — without it the report would lie in the opposite direction."""
    from mix_engine import analysis as engine_analysis, dsp
    from mix_engine.session import Session

    agent = ScriptedAgent([+4.0, +4.0, +3.0])
    report = run_session(loud_project, "master to -12 LUFS", gating="auto",
                         agent=agent, connect=in_memory_engine)
    assert report.reverted_proposal_ids

    audio, rate = dsp.render_master(Session(loud_project))
    fresh = engine_analysis.analyze_master(audio, rate, stems=[], pass_number=1)
    assert report.final_metrics["lufs_i"] == pytest.approx(fresh.lufs_i, abs=0.1)


def test_reverted_proposals_keep_their_prediction_records(loud_project):
    """Calibration learns "I predicted X and measured Y" — which stays TRUE for a
    move that was later reverted. Dropping those records would throw away honest
    observations and bias the cross-project store toward moves that happened to
    survive."""
    agent = ScriptedAgent([+4.0, +4.0, +3.0])
    report = run_session(loud_project, "master to -12 LUFS", gating="auto",
                         agent=agent, connect=in_memory_engine)
    assert report.reverted_proposal_ids

    lines = [json.loads(line) for line
             in Path(report.runs_path).read_text().splitlines()]
    recorded = {line["proposal_id"] for line in lines if "proposal_id" in line}
    for proposal_id in report.reverted_proposal_ids:
        assert proposal_id in recorded, f"{proposal_id} lost its record"
    assert any("rollback" in line for line in lines), "the rollback is not in the run log"
