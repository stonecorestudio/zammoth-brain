"""An approved plan step must still point at what the user approved (2026-08-05).

32 `clip.*` / `take.*` / `comp.*` commands address a clip by **track + index**, and
an index is not an identity: split, delete or add a clip and index 2 is a different
clip. `apply_plan` replayed approved steps verbatim, so "delete that clip" approved
before an edit deleted a DIFFERENT clip afterwards — silently, and destructively.
"""
import json
from pathlib import Path

import pytest

from orchestrator import daw_bridge as B


def edit_with(clips):
    return {"tracks": [{"name": "gtr", "clips": clips},
                       {"name": "drums", "clips": []}]}


CLIP_A = {"index": 0, "name": "riff", "start_s": 0.0, "length_s": 4.0}
CLIP_B = {"index": 1, "name": "solo", "start_s": 4.0, "length_s": 4.0}


def test_a_fingerprint_pins_name_and_position():
    fp = B._clip_fingerprint(edit_with([CLIP_A, CLIP_B]), "gtr", 1)
    assert fp == {"track": "gtr", "index": 1, "name": "solo",
                  "start_s": 4.0, "length_s": 4.0}
    assert B._clip_fingerprint(edit_with([CLIP_A]), "gtr", 9) is None    # no such clip
    assert B._clip_fingerprint(edit_with([CLIP_A]), "nope", 0) is None   # no such track


def test_an_unmoved_clip_is_not_stale():
    before = B._clip_fingerprint(edit_with([CLIP_A, CLIP_B]), "gtr", 1)
    now = B._clip_fingerprint(edit_with([CLIP_A, CLIP_B]), "gtr", 1)
    assert B._clip_moved(before, now) is None


def test_the_index_shift_that_motivated_this():
    """Delete clip 0 and index 1 becomes a DIFFERENT clip — the destructive case."""
    before = B._clip_fingerprint(edit_with([CLIP_A, CLIP_B]), "gtr", 1)
    after_delete = edit_with([{**CLIP_B, "index": 0},
                              {"index": 1, "name": "outro", "start_s": 8.0,
                               "length_s": 2.0}])
    why = B._clip_moved(before, B._clip_fingerprint(after_delete, "gtr", 1))
    assert why and "outro" in why


@pytest.mark.parametrize("changed,expect", [
    ({"start_s": 6.0}, "moved"),
    ({"length_s": 2.0}, "resized"),
    ({"name": "other"}, "not"),
])
def test_each_kind_of_movement_is_caught(changed, expect):
    before = B._clip_fingerprint(edit_with([CLIP_A, CLIP_B]), "gtr", 1)
    now = B._clip_fingerprint(edit_with([CLIP_A, {**CLIP_B, **changed}]), "gtr", 1)
    why = B._clip_moved(before, now)
    assert why and expect in why


def test_a_vanished_clip_is_caught():
    before = B._clip_fingerprint(edit_with([CLIP_A, CLIP_B]), "gtr", 1)
    assert "no longer there" in B._clip_moved(before, None)


def test_the_recorder_pins_clip_steps_only():
    """Proposals are relative deltas and `undo` is by immutable id — neither can be
    mis-aimed, so fingerprinting them would only create false staleness."""
    rec = B._PlanRecorder(edit_with([CLIP_A, CLIP_B]))
    rec.record("command", {"command": "clip.delete",
                           "args": {"track": "gtr", "clip": 1}}, "n")
    rec.record("command", {"command": "transport.play", "args": {}}, "n")
    rec.record("proposal", {"actions": [{"stem_id": "gtr"}]}, "n")
    rec.record("undo", {"proposal_id": "p0001"}, "n")
    assert rec.steps[0]["target"]["name"] == "solo"
    assert "target" not in rec.steps[1]     # no clip in args
    assert "target" not in rec.steps[2]
    assert "target" not in rec.steps[3]


def test_apply_plan_skips_the_stale_step_and_runs_the_rest(tmp_path):
    """The user's call: one nudged clip must not throw away the whole plan."""
    project = tmp_path / "p"
    project.mkdir()
    (project / "session.json").write_text(json.dumps(
        {"stems": {}, "proposals": [], "pass_number": 1}))

    invoked = []

    def daw_invoke(command, args):
        invoked.append(command)
        return {"ok": True}

    plan = {"steps": [
        {"kind": "command", "payload": {"command": "clip.delete",
                                        "args": {"track": "gtr", "clip": 1}},
         "target": B._clip_fingerprint(edit_with([CLIP_A, CLIP_B]), "gtr", 1)},
        {"kind": "command", "payload": {"command": "transport.play", "args": {}}},
    ]}
    # the clip moved between approval and "do it"
    moved = edit_with([CLIP_A, {**CLIP_B, "start_s": 9.0}])

    out = B.apply_plan(project, plan, use_llm=False, edit_state=moved,
                       daw_invoke=daw_invoke)
    assert out["stale"] == 1
    assert invoked == ["transport.play"], "the stale step must not have been invoked"
    assert out["results"][0]["ok"] is False
    assert "skipped" in out["results"][0]["error"]
    assert out["results"][1]["ok"] is True
