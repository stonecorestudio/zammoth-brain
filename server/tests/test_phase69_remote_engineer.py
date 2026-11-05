"""Phase 69: the engineer's own tools for a chat app over the remote door.

The door used to offer a read of the settings, the command list and one command. It now runs
the engineer's tools themselves - `daw_bridge.remote_tool_call`, one `LiveTurn` tool call with no
model - so a chat app on the user's subscription measures, proposes under the guardrails, and
reads the engineer's guide. These drive the brain's half with no DAW attached.
"""

from __future__ import annotations

import json
from pathlib import Path

from orchestrator import remote_engineer as R
from orchestrator.daw_bridge import remote_tool_call


def test_the_door_offers_the_engineers_tools_and_says_what_it_leaves_out():
    names = [d["name"] for d in R.remote_tool_defs()]
    assert names[0] == "engineer_guide"
    for want in ("analyze_session", "analyze_masking", "analyze_phase", "propose", "auto_mix",
                 "daw_command", "lookup_knowledge", "render", "undo_proposal"):
        assert want in names, want
    for gone in R.LEFT_OUT:
        assert gone not in names, gone
    # every one carries an MCP input schema and a read-only hint that matches what it does
    by = {d["name"]: d for d in R.remote_tool_defs()}
    assert by["analyze_session"]["annotations"]["readOnlyHint"] is True
    assert by["propose"]["annotations"]["readOnlyHint"] is False
    assert all(d["inputSchema"].get("type") == "object" for d in by.values())


def test_the_guide_is_the_engineers_prompt_in_parts_a_small_window_can_read():
    from orchestrator.chat import daw_prompt
    parts = R.guide_sections()
    assert len(parts) >= 5
    assert all(len(text) <= R.GUIDE_PART_CHARS * 1.1 for _, text in parts)
    joined = "".join(text for _, text in parts)
    # nothing of the prompt is lost but the headings and the whitespace around cuts
    assert len(joined) >= 0.97 * len(daw_prompt().replace("\n", "")) - 2000
    listing = R.engineer_guide("")
    assert "Ground rules" in listing
    assert R.engineer_guide("ground rules").startswith("## Ground rules")
    assert R.engineer_guide("2").startswith("## ")


def test_a_left_out_tool_and_a_path_outside_the_project_are_refused(demo_project: Path):
    r = remote_tool_call(demo_project, "start_run", {"brief": "louder"})
    assert r["error"] and "chat app is the model" in r["text"]
    r = remote_tool_call(demo_project, "analyze_sample", {"path": "/etc/hosts"})
    assert r["error"] and "inside the project" in r["text"]
    r = remote_tool_call(demo_project, "no_such_tool", {})
    assert r["error"] and "unknown tool" in r["text"]


def test_a_measurement_reads_the_mix(demo_project: Path):
    r = remote_tool_call(demo_project, "analyze_session", {})
    assert not r["error"], r["text"][:300]
    data = json.loads(r["text"])
    assert "lufs_i" in json.dumps(data)


def test_a_proposal_lands_under_the_guardrails_and_an_over_reach_is_refused(demo_project: Path):
    stems = sorted(p.stem for p in demo_project.glob("*.wav") if p.stem != "master")
    stem = stems[0]
    ok = remote_tool_call(demo_project, "propose", {
        "rationale": "a touch down",
        "predicted_impact": [{"metric": f"stem({stem}).lufs_s", "predicted_delta": -1.5, "unit": "dB"}],
        "actions": [{"type": "gain", "stem_id": stem, "delta_db": -1.5}]})
    assert not ok["error"], ok["text"][:400]
    session = json.loads((demo_project / "session.json").read_text())
    assert session["stems"][stem]["gain_db"] < 0 or any(
        e.get("type") == "gain" for e in session["stems"][stem].get("chain", []))
    too_far = remote_tool_call(demo_project, "propose", {
        "rationale": "far too much",
        "predicted_impact": [{"metric": f"stem({stem}).lufs_s", "predicted_delta": -20, "unit": "dB"}],
        "actions": [{"type": "gain", "stem_id": stem, "delta_db": -20}]})
    text = too_far["text"].lower()
    assert too_far["error"] or "guardrail" in text or "violation" in text or "rejected" in text


def test_an_export_with_no_file_names_a_new_file_in_the_projects_bounces(demo_project: Path):
    """A chat app does not wait on a person's box: with no file named, the door names a new one in
    bounces/ (the engineer's own no-box case). A file or a stems export the app chose is left alone."""
    plain = {"command": "session.export", "args": {"format": "wav"}}
    filled = R.fill_export_file(demo_project, "daw_command", plain)
    f = Path(filled["args"]["file"])
    assert f.parent == demo_project / "bounces" and f.suffix == ".wav" and not f.exists()
    chosen = {"command": "session.export", "args": {"format": "wav", "file": "/x/mine.wav"}}
    assert R.fill_export_file(demo_project, "daw_command", chosen) == chosen
    stems = {"command": "session.export", "args": {"format": "wav", "stems": True}}
    assert R.fill_export_file(demo_project, "daw_command", stems) == stems
    other = {"command": "track.set_gain", "args": {"track": "a", "gain_db": -1}}
    assert R.fill_export_file(demo_project, "daw_command", other) == other
