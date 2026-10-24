"""Phase 27 — automation toggles + applied_seq.

Section moves (Phase 5 gain rides, Phase 15 time-ranged entries) bypass at
render time when the per-stem or session-global automation switch is off —
bit-identically to a state that never had them — and re-enabling restores
the automated render bit-identically. The switches are the user channel
(set_stem); whole-file processing is never affected. get_proposals carries
applied_seq, the UI's recency ranking for "what changed last".
"""
from pathlib import Path

import numpy as np
import pytest

from mix_engine import dsp
from mix_engine.models import ProposalRequest
from mix_engine.session import Session


def _apply(session: Session, actions: list[dict]) -> str:
    result = session.propose(ProposalRequest.model_validate(
        {"rationale": "test", "predicted_impact": [], "actions": actions}))
    assert result.accepted, result.violations
    applied = session.apply(result.proposal_id)
    assert applied.ok, applied.message
    return result.proposal_id


def _render(session: Session) -> np.ndarray:
    return dsp.render_master(session)[0]


RIDE = {"type": "gain", "stem_id": "stem_a", "delta_db": 4.0,
        "time_range_s": [0.5, 1.5]}
RANGED_EQ = {"type": "eq", "stem_id": "stem_a",
             "band": {"freq_hz": 1000.0, "gain_db": -4.0, "q": 1.0,
                      "shape": "peak"},
             "time_range_s": [2.0, 3.0]}
WHOLE_FILE_GAIN = {"type": "gain", "stem_id": "stem_a", "delta_db": 2.0}


def test_set_stem_round_trips_both_switches(demo_project: Path) -> None:
    session = Session(demo_project)
    state = session.get_state()
    assert state.automation_enabled is True
    assert all(s.automation_enabled for s in state.stems)

    echo = session.user_edit("stem_a", {"automation_enabled": False})
    assert echo["automation_enabled"] is False
    echo = session.user_edit("master", {"automation_enabled": False})
    assert echo["automation_enabled"] is False

    # persisted: a fresh Session sees both switches
    reloaded = Session(demo_project)
    state = reloaded.get_state()
    assert state.automation_enabled is False
    by_id = {s.stem_id: s for s in state.stems}
    assert by_id["stem_a"].automation_enabled is False
    assert by_id["stem_b"].automation_enabled is True


def test_master_still_rejects_other_fields(demo_project: Path) -> None:
    session = Session(demo_project)
    with pytest.raises(ValueError, match="automation_enabled"):
        session.user_edit("master", {"pan": 0.5})


def test_stem_switch_bypasses_rides_bit_identically(demo_project: Path) -> None:
    session = Session(demo_project)
    baseline = _render(session)
    _apply(session, [RIDE])
    automated = _render(session)
    assert not np.array_equal(automated, baseline)

    session.user_edit("stem_a", {"automation_enabled": False})
    assert np.array_equal(_render(session), baseline)

    session.user_edit("stem_a", {"automation_enabled": True})
    assert np.array_equal(_render(session), automated)


def test_ranged_entries_bypass_whole_file_stays(demo_project: Path) -> None:
    session = Session(demo_project)
    _apply(session, [WHOLE_FILE_GAIN])
    gained = _render(session)
    _apply(session, [RANGED_EQ])
    automated = _render(session)
    assert not np.array_equal(automated, gained)

    # automation off: the ranged EQ bypasses, the whole-file gain stays
    session.user_edit("stem_a", {"automation_enabled": False})
    assert np.array_equal(_render(session), gained)
    # the entry never left the chain — it is bypassed, not removed
    assert any(e.get("time_range_s") for e in session.applied_chain("stem_a"))

    session.user_edit("stem_a", {"automation_enabled": True})
    assert np.array_equal(_render(session), automated)


def test_global_switch_bypasses_every_stem(demo_project: Path) -> None:
    session = Session(demo_project)
    baseline = _render(session)
    _apply(session, [RIDE,
                     {"type": "gain", "stem_id": "stem_b", "delta_db": -3.0,
                      "time_range_s": [1.0, 2.0]}])
    automated = _render(session)
    assert not np.array_equal(automated, baseline)

    session.user_edit("master", {"automation_enabled": False})
    assert np.array_equal(_render(session), baseline)
    # per-stem switches were not touched by the global one
    assert all(s.automation_enabled for s in session.get_state().stems)

    session.user_edit("master", {"automation_enabled": True})
    assert np.array_equal(_render(session), automated)


def test_get_proposals_carries_applied_seq(demo_project: Path) -> None:
    session = Session(demo_project)
    first = _apply(session, [WHOLE_FILE_GAIN])
    second = _apply(session, [{"type": "gain", "stem_id": "stem_b",
                               "delta_db": 1.0}])
    by_id = {p.proposal_id: p for p in session.get_proposals()}
    assert by_id[first].applied_seq == 1
    assert by_id[second].applied_seq == 2

    # undo keeps the seq (the record stays rankable), status flips
    assert session.undo(second).ok
    by_id = {p.proposal_id: p for p in session.get_proposals()}
    assert by_id[second].status == "undone"
    assert by_id[second].applied_seq == 2
