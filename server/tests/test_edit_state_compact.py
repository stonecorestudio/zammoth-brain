"""The edit snapshot must fit its prompt budget without going blind.

2026-09-01 sweep, P1: every consumer cut the DAW's live edit snapshot with
`text[:budget]`. The snapshot writes `tracks` first and tempo / transport /
groups / markers / selection last, so on a real session (one MIDI clip is up to
512 notes, ~46 kB) the agent lost exactly the fields the prompt tells it to use
and was handed unparseable JSON with a 22-character marker at the end.
"""
from __future__ import annotations

import json

from orchestrator.edit_state import compact_edit_state


def _big_snapshot(n_tracks: int = 6, notes_per_clip: int = 512) -> dict:
    tracks = []
    for t in range(n_tracks):
        notes = [{"n": 36 + (i % 24), "b": round(i * 0.25, 3), "l": 0.25, "v": 96}
                 for i in range(notes_per_clip)]
        tracks.append({
            "name": f"track_{t}",
            "clips": [{"name": f"clip_{t}", "start_s": 0.0, "len_s": 8.0, "notes": notes}],
            "devices": [{"type": "eq", "params": {f"band{i}": {"freq_hz": 100 * i, "gain_db": 1.0}
                                                  for i in range(24)}}],
            "instrument_params": {f"p{i}": i / 64 for i in range(64)},
        })
    return {
        "commands": ["should", "be", "excluded"],
        "tracks": tracks,
        "tempo": {"bpm": 124.0, "changes": [{"at_s": 12.0, "bpm": 90.0}]},
        "transport": {"position_s": 17.5, "playing": True},
        "groups": [{"name": "drums", "members": ["track_0", "track_1"]}],
        "markers": [{"name": "drop", "at_s": 32.0}],
        "library": [{"id": f"lib{i}", "name": f"sample_{i}.wav"} for i in range(300)],
        "available_plugins": [{"name": f"Plugin {i}"} for i in range(200)],
        "selection": {"track": "track_3", "start_s": 4.0, "end_s": 8.0},
        "master_delay_ms": 0.0,
    }


def test_small_snapshots_pass_through_unchanged_except_the_command_catalog():
    snap = {"tracks": [{"name": "a"}], "tempo": {"bpm": 120}, "commands": ["x"]}
    out = json.loads(compact_edit_state(snap, 24000))
    assert out == {"tracks": [{"name": "a"}], "tempo": {"bpm": 120}}


def test_the_session_global_tail_survives_a_budget_the_notes_alone_would_blow():
    snap = _big_snapshot()
    raw = json.dumps(snap, separators=(",", ":"))
    assert len(raw) > 100_000, "fixture must genuinely overflow"
    text = compact_edit_state(snap, 24000)
    assert len(text) <= 24000
    out = json.loads(text)                       # valid JSON, always
    for key in ("tempo", "transport", "groups", "markers", "selection", "master_delay_ms"):
        assert key in out, key
    assert out["selection"]["track"] == "track_3"
    assert out["tempo"]["changes"][0]["bpm"] == 90.0
    assert "commands" not in out
    # what went is named, and the cheapest information went first
    assert out["omitted"][0] == "clip_notes"
    assert out["tracks"][0]["clips"][0]["notes_omitted"] == 512
    assert "notes" not in out["tracks"][0]["clips"][0]


def test_the_agent_still_sees_the_track_and_clip_list_when_only_notes_had_to_go():
    snap = _big_snapshot(n_tracks=2)
    raw = len(json.dumps(snap, separators=(",", ":")))
    budget = raw // 2                      # the notes alone are well over half of it
    out = json.loads(compact_edit_state(snap, budget))
    assert out["omitted"] == ["clip_notes"]
    assert [t["name"] for t in out["tracks"]] == ["track_0", "track_1"]
    assert out["tracks"][0]["clips"][0]["name"] == "clip_0"
    assert "params" in out["tracks"][0]["devices"][0]      # untouched at this level


def test_the_old_character_cut_would_have_lost_the_selection():
    """Pins the failure mode: the field the prompt reads as 'this part' sat past
    24k chars in the raw text."""
    snap = _big_snapshot()
    raw = json.dumps(snap, separators=(",", ":"))
    assert '"selection"' not in raw[:24000]
