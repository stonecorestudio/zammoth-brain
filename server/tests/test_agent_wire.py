"""ClaudeAgent wire format (regression).

The action union outgrew the API's strict-structured-output grammar limit in
Phase 7 (400: "The compiled grammar is too large"), so plan turns now return
plain JSON validated locally against the engine schema. These tests pin the
local half of that contract without needing the SDK or an API key.
"""
import json

import pytest

from orchestrator.agents import extract_json
from orchestrator.engine_models import ProposalRequest


def test_extract_json_tolerates_fences_and_prose():
    payload = {"proposals": []}
    for reply in (
        json.dumps(payload),
        f"```json\n{json.dumps(payload)}\n```",
        f"Here is my plan:\n{json.dumps(payload)}\nLet me know.",
    ):
        assert extract_json(reply) == payload
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_plan_json_validates_against_engine_schema():
    """A reply using the full Phase 5-7 action surface round-trips through the
    engine's own ProposalRequest model — the validation ClaudeAgent now runs
    locally instead of compiling a grammar."""
    proposal = {
        "rationale": "limit the master, ride the chorus, duck the bass",
        "predicted_impact": [
            {"metric": "lufs_i", "predicted_delta": 4.0, "unit": "LUFS"}],
        "actions": [
            {"type": "limiter", "stem_id": "master",
             "ceiling_dbtp": -1.0, "target_lufs_i": -14.0},
            {"type": "gain", "stem_id": "vox", "delta_db": 3.0,
             "time_range_s": [8.0, 16.0]},
            {"type": "duck", "stem_id": "bass", "key_stem_id": "kick",
             "threshold_db": -30.0, "depth_db": 6.0},
            {"type": "saturation", "stem_id": "drums", "drive_db": 6.0},
        ],
    }
    validated = ProposalRequest.model_validate(proposal)
    assert [a.type for a in validated.actions] == ["limiter", "gain", "duck",
                                                   "saturation"]