"""What the mashup showcase's MASTERING turn found (2026-10-11), after the mix: the agent
mastered and exported the mashup in the real app, and three things stood between it and a
truthful report.

1. It could not measure a bar range of the MIX. `describe_region` refused `master` ("no such
   stem"), in two turns running, so the agent split the detected sections by adding up each
   stem's loudness - and its Chorus / Drop / Finale figures came out about 1 dB off, with the
   Drop and the Chorus in the wrong order (the file itself: -13.2 / -13.8 / -13.0).
2. It could not read the exported FILE's true peak: `analyze_reference` gave loudness and no
   peak, and its next try (`analyze_reference_set`) was a validation dump.
3. The export ran 4:12 for a 1:57 song: a muted full-length track set the end. That half is the
   app's; the pins on it read the C++.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from mcp.shared.memory import create_connected_server_and_client_session

from mix_engine import analysis
from mix_engine.mcp_server import create_server
from mix_engine.models import ReferenceAnalysis, RegionDescription
from test_phase1_acceptance import call

SR = 48000
REPO = Path(__file__).resolve().parents[2]


def _two_part_project(folder: Path) -> None:
    """A quiet first half and a loud second half, on two stems: a window of the MIX is not a
    window of either stem."""
    t = np.arange(8 * SR) / SR
    quiet_then_loud = np.where(t < 4.0, 0.05, 0.5)
    sf.write(folder / "keys.wav", quiet_then_loud * np.sin(2 * np.pi * 440.0 * t), SR, subtype="FLOAT")
    sf.write(folder / "bass.wav", quiet_then_loud * np.sin(2 * np.pi * 110.0 * t), SR, subtype="FLOAT")


# ---------------------------------------------------------------------------------------------
# 1 - a bar range of the mix (`describe_region` on `master`)
# ---------------------------------------------------------------------------------------------

@pytest.mark.anyio
async def test_a_window_of_the_mix_is_measured(tmp_path):
    _two_part_project(tmp_path)
    async with create_connected_server_and_client_session(create_server(tmp_path)) as client:
        quiet = await call(client, "describe_region", {"stem_id": "master", "start_s": 0.5, "end_s": 3.5})
        loud = await call(client, "describe_region", {"stem_id": "master", "start_s": 4.5, "end_s": 7.5})
        keys = await call(client, "describe_region", {"stem_id": "keys", "start_s": 4.5, "end_s": 7.5})
    for got in (quiet, loud):
        RegionDescription.model_validate(got)
        assert got["stem_id"] == "master" and not got["silent"]
        assert got["masked_by"] == [] and got["masks"] == []      # the mix masks nothing: it is everything
    # the window's own integrated loudness - the figure a section is reported by
    assert loud["lufs_i"] is not None and quiet["lufs_i"] is not None
    assert 19.0 <= loud["lufs_i"] - quiet["lufs_i"] <= 21.0, (quiet["lufs_i"], loud["lufs_i"])
    # the mix is both parts: louder than one of them over the same bars
    assert loud["lufs_i"] > keys["lufs_i"] + 1.0, (loud["lufs_i"], keys["lufs_i"])


@pytest.mark.anyio
async def test_the_mix_can_be_read_in_front_of_the_master_chain(tmp_path):
    _two_part_project(tmp_path)
    async with create_connected_server_and_client_session(create_server(tmp_path)) as client:
        proposal = await call(client, "propose_actions", {"req": {
            "rationale": "hold the master at -6",
            "predicted_impact": [{"metric": "true_peak_dbtp", "predicted_delta": -5.0, "unit": "dB"}],
            "actions": [{"type": "limiter", "stem_id": "master", "ceiling_dbtp": -6.0}]}})
        assert proposal["accepted"], proposal["violations"]
        applied = await call(client, "apply_proposal", {"proposal_id": proposal["proposal_id"]})
        assert applied["ok"], applied["message"]
        window = {"stem_id": "master", "start_s": 4.5, "end_s": 7.5}
        mastered = await call(client, "describe_region", window)
        mix = await call(client, "describe_region", {**window, "pre_master": True})
    # the master is what leaves the limiter; `pre_master` is the mix as it enters mastering
    assert mastered["true_peak_dbtp"] <= -5.9, mastered["true_peak_dbtp"]
    assert mix["true_peak_dbtp"] >= -2.0, mix["true_peak_dbtp"]


@pytest.mark.anyio
async def test_pre_master_is_the_mixs_alone(tmp_path):
    """On a stem there is no "in front of the master": asking for it is refused, not ignored."""
    _two_part_project(tmp_path)
    async with create_connected_server_and_client_session(create_server(tmp_path)) as client:
        got = await client.call_tool("describe_region", {"stem_id": "keys", "start_s": 0.5, "end_s": 3.5,
                                                         "pre_master": True})
    assert got.isError and "master" in str(got.content)


@pytest.mark.anyio
async def test_a_stems_window_carries_the_same_two_figures(tmp_path):
    _two_part_project(tmp_path)
    async with create_connected_server_and_client_session(create_server(tmp_path)) as client:
        got = await call(client, "describe_region", {"stem_id": "bass", "start_s": 4.5, "end_s": 7.5})
        missing = await client.call_tool("describe_region", {"stem_id": "nobody", "start_s": 0.5, "end_s": 3.5})
    assert got["lufs_i"] is not None and got["true_peak_dbtp"] is not None
    assert got["true_peak_dbtp"] >= got["peak_dbfs"] - 0.05
    assert missing.isError                                         # an unknown stem is still refused


def test_a_windows_true_peak_is_between_the_samples():
    """A tone at a quarter of the rate, 45 degrees off the samples: every sample is 3 dB under
    the crest. The window's peak says -3.0 and its true peak says 0."""
    n = np.arange(2 * SR)
    tone = np.sin(2 * np.pi * (SR / 4) * n / SR + np.pi / 4)
    got = analysis.describe_region(np.stack([tone, tone], axis=1), SR, "t", 0.0, 2.0)
    assert abs(got.peak_dbfs - (-3.0)) <= 0.1, got.peak_dbfs
    assert abs(got.true_peak_dbtp) <= 0.3, got.true_peak_dbtp


# ---------------------------------------------------------------------------------------------
# 2 - the peak of a file on disk (`analyze_reference`)
# ---------------------------------------------------------------------------------------------

def _between_samples(path: Path, seconds: float = 3.0) -> None:
    n = np.arange(int(seconds * SR))
    tone = 0.5 * np.sin(2 * np.pi * (SR / 4) * n / SR + np.pi / 4)
    sf.write(path, np.stack([tone, tone], axis=1), SR, subtype="FLOAT")


def test_a_files_profile_holds_its_peaks():
    n = np.arange(3 * SR)
    tone = 0.5 * np.sin(2 * np.pi * (SR / 4) * n / SR + np.pi / 4)
    got = analysis.analyze_reference(np.stack([tone, tone], axis=1), SR, path="x.wav")
    assert abs(got.sample_peak_dbfs - (-9.0)) <= 0.1, got.sample_peak_dbfs      # 0.5 = -6, less 3
    assert abs(got.true_peak_dbtp - (-6.0)) <= 0.3, got.true_peak_dbtp
    silent = analysis.analyze_reference(np.zeros((SR, 2)), SR, path="x.wav")
    assert silent.silent and silent.true_peak_dbtp is None and silent.sample_peak_dbfs is None


@pytest.mark.anyio
async def test_an_exported_file_is_measured_where_it_lies(tmp_path, demo_project):
    """`bounces/` inside the project is where the app writes an export: the tool reads it there."""
    out = demo_project / "bounces"
    out.mkdir(exist_ok=True)
    _between_samples(out / "master.wav")
    async with create_connected_server_and_client_session(create_server(demo_project)) as client:
        got = await call(client, "analyze_reference", {"path": str(out / "master.wav")})
    ReferenceAnalysis.model_validate(got)
    assert abs(got["true_peak_dbtp"] - (-6.0)) <= 0.3 and abs(got["sample_peak_dbfs"] - (-9.0)) <= 0.1, got
    assert got["lufs_i"] is not None and got["duration_s"] == 3.0


@pytest.mark.anyio
async def test_a_profile_cached_before_the_peaks_is_measured_again(tmp_path, demo_project):
    ref = demo_project / "bounces"
    ref.mkdir(exist_ok=True)
    _between_samples(ref / "old.wav")
    async with create_connected_server_and_client_session(create_server(demo_project)) as client:
        first = await call(client, "analyze_reference", {"path": str(ref / "old.wav")})
        cache = demo_project / ".reference_cache.json"
        held = json.loads(cache.read_text())
        for entry in held["entries"]:                       # as a build before the peaks wrote it
            entry["result"].pop("true_peak_dbtp", None)
            entry["result"].pop("sample_peak_dbfs", None)
        cache.write_text(json.dumps(held))
        again = await call(client, "analyze_reference", {"path": str(ref / "old.wav")})
    assert first["true_peak_dbtp"] is not None
    assert again["true_peak_dbtp"] == first["true_peak_dbtp"], again


# ---------------------------------------------------------------------------------------------
# 2b - one section of a FILE, and what a reading of the mix leaves out
#
# Re-exported on the fixed build, the agent measured every section on `master` and all but one
# matched the file: the Finale read -12.6 where the file is -13.0. The Finale is where it had
# written pan automation, the offline engine plays no automation, and nothing on the reading
# said so. The file holds what the app played; a reading of the mix now says what it left out.
# ---------------------------------------------------------------------------------------------

def _quiet_then_loud(path: Path) -> None:
    t = np.arange(8 * SR) / SR
    tone = np.where(t < 4.0, 0.05, 0.5) * np.sin(2 * np.pi * 440.0 * t)
    sf.write(path, np.stack([tone, tone], axis=1), SR, subtype="FLOAT")


@pytest.mark.anyio
async def test_one_section_of_a_file_is_measured(demo_project):
    out = demo_project / "bounces"
    out.mkdir(exist_ok=True)
    _quiet_then_loud(out / "song.wav")
    path = str(out / "song.wav")
    async with create_connected_server_and_client_session(create_server(demo_project)) as client:
        quiet = await call(client, "analyze_reference", {"path": path, "start_s": 0.5, "end_s": 3.5})
        loud = await call(client, "analyze_reference", {"path": path, "start_s": 4.5, "end_s": 7.5})
        whole = await call(client, "analyze_reference", {"path": path})
        again = await call(client, "analyze_reference", {"path": path, "start_s": 0.5, "end_s": 3.5})
        half = await client.call_tool("analyze_reference", {"path": path, "start_s": 1.0})
        past = await client.call_tool("analyze_reference", {"path": path, "start_s": 9.0, "end_s": 12.0})
    for got in (quiet, loud, whole):
        ReferenceAnalysis.model_validate(got)
    assert 19.0 <= loud["lufs_i"] - quiet["lufs_i"] <= 21.0, (quiet["lufs_i"], loud["lufs_i"])
    assert quiet["window_s"] == [0.5, 3.5] and quiet["duration_s"] == 3.0
    assert abs(loud["true_peak_dbtp"] - (-6.0)) <= 0.3 and abs(quiet["true_peak_dbtp"] - (-26.0)) <= 0.3
    # the whole file is still the whole file: a window is never served from, or as, its cache
    assert whole["window_s"] is None and whole["duration_s"] == 8.0
    assert again == quiet
    assert half.isError and "both" in str(half.content)          # half a range is refused
    assert past.isError and "8.0" in str(past.content)           # ...and one past the file's end, by its length


def _edit_with_a_pan_lane() -> dict:
    return {"tracks": [{"name": "vocals", "track_kind": "audio", "muted": False,
                        "clips": [{"start_s": 0.0, "length_s": 8.0}],
                        "automation": [{"param": "pan", "points": 3}]}],
            "master": {"name": "master"}, "returns": []}


def test_a_reading_of_the_mix_says_what_it_leaves_out(tmp_path):
    from orchestrator import daw_bridge
    edit = _edit_with_a_pan_lane()
    for name, args in (("describe_region", {"stem_id": "master", "start_s": 0, "end_s": 4}),
                       ("analyze_session", {}), ("analyze_sections", {"pre_master": True})):
        assert daw_bridge.reads_the_mix(name, args), name
        got = daw_bridge.with_blind_spots(name, args, {"lufs_i": -14.0}, edit, tmp_path)
        assert got["lufs_i"] == -14.0
        kinds = [b["what"] for b in got["not_heard"]]
        assert "automation" in kinds, got
    # a stem's own reading, a file's, and anything that is not a dict are handed back as they came
    for name, args in (("describe_region", {"stem_id": "vocals", "start_s": 0, "end_s": 4}),
                       ("analyze_reference", {"path": "x.wav"}), ("analyze_stem", {"stem_id": "vocals"})):
        assert not daw_bridge.reads_the_mix(name, args), name
        payload = {"lufs_i": -14.0}
        assert daw_bridge.with_blind_spots(name, args, payload, edit, tmp_path) is payload
    assert daw_bridge.with_blind_spots("analyze_session", {}, [1, 2], edit, tmp_path) == [1, 2]
    # nothing left out, nothing said: a project of stems and chains is the whole mix
    plain = {"tracks": [{"name": "vocals", "track_kind": "audio", "muted": False}], "returns": []}
    assert "not_heard" not in daw_bridge.with_blind_spots("analyze_session", {}, {"a": 1}, plain, tmp_path)


def test_the_chats_read_door_is_where_it_is_said():
    import inspect
    from orchestrator import turn_host       # the turn's tools live there since Phase 55
    chat_door = inspect.getsource(turn_host.LiveTurn.engine_call)
    assert "with_blind_spots(name, args, payload" in chat_door


# ---------------------------------------------------------------------------------------------
# what the agent is told
# ---------------------------------------------------------------------------------------------

def test_the_agent_is_told_both_doors():
    from orchestrator import chat
    defs = {d["name"]: d for d in chat._READ_TOOL_DEFS}
    region = defs["describe_region"]
    assert "master" in region["description"] and "pre_master" in region["input_schema"]["properties"]
    assert "lufs_i" in region["description"] and "true peak" in region["description"].lower()
    ref = defs["analyze_reference"]["description"].lower()
    assert "true peak" in ref and "export" in ref
    assert {"start_s", "end_s"} <= set(defs["analyze_reference"]["input_schema"]["properties"])
    assert "not_heard" in region["description"]


def test_the_fixture_replies_have_the_new_figures():
    from orchestrator import understanding as U
    region = U.fixture_tool_reply("describe_region", {"stem_id": "lead_vox"})
    ref = U.fixture_tool_reply("analyze_reference", {"path": "x.wav"})
    assert region["lufs_i"] is not None and region["true_peak_dbtp"] is not None
    assert ref["true_peak_dbtp"] is not None and ref["sample_peak_dbfs"] is not None


# ---------------------------------------------------------------------------------------------
# 3 - the export ends where the song does (the app's half: these read the C++)
# ---------------------------------------------------------------------------------------------

_DAWSESSION = (REPO / "daw/Source/DawSession.h").read_text()
_APP = (REPO / "daw/Source/App.cpp").read_text()


@pytest.mark.reads_cpp
def test_the_song_ends_at_the_last_clip_that_plays():
    """The export's quiet-tail trim keeps `songS`; that is the end of the last clip on a track
    that plays, never the edit's length when a muted track runs longer."""
    assert "inline double lastPlayingClipEndSeconds (te::Edit& edit)" in _DAWSESSION
    estimate = _DAWSESSION[_DAWSESSION.index("inline double lastPlayingClipEndSeconds"):]
    estimate = estimate[:estimate.index("\n}\n")]
    assert "shouldBePlayed()" in estimate and "isMuted()" in estimate
    body = _DAWSESSION[_DAWSESSION.index("const RenderOptions& opt)\n{"):]     # the definition, not its declaration
    body = body[:body.index("\ninline ", 10)]
    assert "opt.songEndS < songS" in body
    assert re.search(r"songEndsEarly\)\s*&&\s*! opt\.trimSilence\)\s*\(void\) trimQuietTail", body), \
        "the trim after the song must run when the song ends before the edit does"
    # the range rendered is still the edit's: whatever DOES sound later is kept by the trim
    assert "edit.getLength() + te::TimeDuration::fromSeconds (tailS)" in body


@pytest.mark.reads_cpp
def test_the_export_asks_once_before_it_borrows_the_mutes():
    """One answer for the master and every stem (a stem's render mutes the other tracks), and
    only the export asks: a bounce, a glue and a freeze place their file by its length."""
    assert "double songEndS = 0.0;" in _DAWSESSION
    assert _APP.count("opt.songEndS = dawsession::lastPlayingClipEndSeconds (*edit);") == 1
    assert _APP.count("lastPlayingClipEndSeconds") == 1
    asked = _APP.index("opt.songEndS = dawsession::lastPlayingClipEndSeconds")
    rendered = _APP.index("const bool ok = dawsession::renderEditToFile (engine, *edit, out, opt);")
    assert asked < rendered


@pytest.mark.reads_cpp
def test_the_export_says_how_long_the_file_is_and_its_peak():
    start = _APP.index("auto outcome = dawcmd::exportReply (succeeded")
    block = _APP[start:start + 4200]
    assert 'oo->setProperty ("length_s"' in block and 'oo->setProperty ("peak_dbfs"' in block


@pytest.mark.reads_cpp
def test_the_red_runs_switch_is_gone():
    for text in (_APP, _DAWSESSION):
        assert "masteringOld" not in text and "ZAMMOTH_MASTERING_OLD" not in text
