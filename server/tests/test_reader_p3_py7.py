"""Gates for batch Py7 of the reader P3s (full audit 2026-10-02): the session, the offline
render, analysis.

Eleven findings - D8·R7, D8·R8, D8·R9, D8·R10, D9·R11, E4·R6, E4·R8, E5·R9, E5·R10, E5·R11,
D4·R13 - and the two leads the triage filed with them (two more `resolve_chains` callers with
no project rate; a short MONO file breaking every readout). Each test drives the door the
product uses and carries its control. Every one was run on the unfixed tree first. The record
is docs/audits/full-2026-10-02/readers/P3-fixes.md (Py7).
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from pydantic import ValidationError
from scipy.signal import lfilter

from mix_engine import analysis, dsp
from mix_engine.models import (ANALYSIS_VERSION, LimiterAction, ProbeRequest, ProposalRequest,
                               SaturationAction)
from mix_engine.session import Session
from orchestrator import daw_bridge

SR = 48000
ROOT = Path(__file__).resolve().parents[2]


def _apply(session: Session, actions: list[dict]) -> str:
    result = session.propose(ProposalRequest(rationale="py7", predicted_impact=[], actions=actions))
    assert result.accepted, result.violations
    assert session.apply(result.proposal_id).ok
    return result.proposal_id


def _write(path, audio: np.ndarray, sr: int = SR, subtype: str = "FLOAT") -> None:
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sr, subtype=subtype)


def _tone(hz: float, seconds: float = 4.0, sr: int = SR, level: float = 0.2) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return level * np.sin(2.0 * np.pi * hz * t)


def _project(tmp_path, stems: dict[str, np.ndarray], sr: int = SR) -> Path:
    proj = tmp_path / "proj"
    proj.mkdir(parents=True)
    for name, audio in stems.items():
        _write(proj / f"{name}.wav", audio, sr)
    return proj


def _four(tmp_path) -> Path:
    """Four short stems - enough tracks to group, nest and leave one outside."""
    return _project(tmp_path, {name: _tone(hz, 2.0) for name, hz in
                               (("kick", 110.0), ("snare", 220.0), ("bass", 82.4), ("pad", 330.0))})


# ====================================================================================
# D8·R7 - no door leaves a group with nothing in it
# ====================================================================================

def test_undo_does_not_bring_a_dissolved_group_back_empty(tmp_path):
    """A move on a group, the user dissolves the group, the move is undone: the snapshot
    re-created the group - with no members (the dissolve re-homed them, and they are not in
    the snapshot). An empty strip in the mixer, and a group the agent is shown that holds
    nothing."""
    proj = _four(tmp_path)
    session = Session(proj)
    bus = session.group_stems(["kick", "snare"])["bus_id"]
    move = _apply(session, [{"type": "eq", "stem_id": bus,
                             "band": {"freq_hz": 3000, "gain_db": -2.0, "q": 1.0, "shape": "peak"}}])
    session.remove_bus(bus)
    assert bus not in session.buses                       # control: the dissolve itself
    assert session.undo(move).ok
    assert bus not in session.buses, "the undo re-created the dissolved group, empty"
    assert all(s.get("bus") is None for s in session.stems.values())
    # ...and the file says the same (the DAW draws from the next read of it)
    assert bus not in Session(proj, read_only=True).buses


def test_the_dissolved_group_comes_back_as_it_was_before_the_move(tmp_path):
    """...and when the DISSOLVE is undone (the DAW's Cmd-Z of it: `restore_routing`), the
    group returns with its members and WITHOUT the move the agent's undo took back - the undo
    was applied to the buried record."""
    proj = _four(tmp_path)
    session = Session(proj)
    bus = session.group_stems(["kick", "snare"])["bus_id"]
    uid = session.buses[bus]["uid"]
    move = _apply(session, [{"type": "eq", "stem_id": bus,
                             "band": {"freq_hz": 3000, "gain_db": -2.0, "q": 1.0, "shape": "peak"}}])
    session.remove_bus(bus)
    assert session.undo(move).ok
    assert session.restore_routing("bus", bus, uid)["restored"]
    assert session.stems["kick"]["bus"] == bus and session.stems["snare"]["bus"] == bus
    assert not any(e["type"] == "eq" for e in session.buses[bus]["chain"]), \
        "the group came back with the move its undo had taken back"


def test_undo_still_restores_a_group_that_has_its_members(tmp_path):
    """The control: a group that was NOT dissolved is restored by the undo as it always was."""
    proj = _four(tmp_path)
    session = Session(proj)
    bus = session.group_stems(["kick", "snare"])["bus_id"]
    move = _apply(session, [{"type": "eq", "stem_id": bus,
                             "band": {"freq_hz": 3000, "gain_db": -2.0, "q": 1.0, "shape": "peak"}}])
    assert any(e["type"] == "eq" for e in session.buses[bus]["chain"])
    assert session.undo(move).ok
    assert bus in session.buses
    assert not any(e["type"] == "eq" for e in session.buses[bus]["chain"])
    assert session.stems["kick"]["bus"] == bus


def test_resetting_a_groups_only_member_drops_the_group(tmp_path):
    proj = _four(tmp_path)
    session = Session(proj)
    bus = session.group_stems(["kick"])["bus_id"]
    assert session.reset_stem("kick")["reset"]
    assert bus not in session.buses, "reset_stem left a group with no member"


def test_copying_state_onto_a_groups_only_member_drops_the_group(tmp_path):
    proj = _four(tmp_path)
    session = Session(proj)
    bus = session.group_stems(["kick"])["bus_id"]
    assert session.copy_stem_state("snare", "kick")["ok"]      # snare plays to the master
    assert session.stems["kick"]["bus"] is None
    assert bus not in session.buses, "copy_stem_state left a group with no member"


def test_copying_state_from_a_grouped_track_keeps_the_group(tmp_path):
    """The control: a copy that brings a member IN leaves the group alone."""
    proj = _four(tmp_path)
    session = Session(proj)
    bus = session.group_stems(["kick"])["bus_id"]
    assert session.copy_stem_state("kick", "snare")["bus"] == bus
    assert bus in session.buses and session.stems["snare"]["bus"] == bus


def test_an_emptied_child_takes_its_empty_parent_with_it(tmp_path):
    """`_drop_empty_buses` was one pass: a parent whose only reason to exist was the child
    that pass dropped survived it."""
    proj = _four(tmp_path)
    session = Session(proj)
    child = session.group_stems(["kick"])["bus_id"]
    parent = session.group_stems(["snare"])["bus_id"]
    session.route_bus(child, parent)
    session.route_stem("snare", None)                    # the parent now holds only the child
    assert parent in session.buses and child in session.buses    # control: a collector lives
    session.route_stem("kick", None)                     # the child empties
    assert child not in session.buses
    assert parent not in session.buses, "the parent of a dropped group was left, empty"


# ====================================================================================
# D8·R8 - a copied chain keeps the chain rules every other door enforces
# ====================================================================================

def _duck(key: str) -> dict:
    return {"type": "duck", "key_stem_id": key, "depth_db": 6.0, "threshold_db": -30.0,
            "attack_ms": 10.0, "release_ms": 120.0}


def test_a_copied_duck_keyed_to_its_new_track_is_dropped(tmp_path):
    """bass ducks under kick. Duplicate bass's state onto kick: the copy would duck kick under
    kick - which `set_duck` and the proposal validators both refuse."""
    proj = _four(tmp_path)
    session = Session(proj)
    session.stems["bass"]["chain"] = [_duck("kick")]
    session._persist()
    out = session.copy_stem_state("bass", "kick")
    assert out["ok"]
    assert not any(e.get("type") == "duck" for e in session.stems["kick"]["chain"])
    assert any("duck" in d for d in out["devices_dropped"]), out
    # the control: onto any other track the duck carries
    out = session.copy_stem_state("bass", "snare")
    assert [e["key_stem_id"] for e in session.stems["snare"]["chain"] if e["type"] == "duck"] == ["kick"]
    assert out["devices_dropped"] == []


def _sat(drive: float) -> dict:
    return {"type": "saturation", "drive_db": drive}


def test_an_adopted_chain_holds_one_of_a_singleton(tmp_path):
    """Two whole-file saturations in a foreign chain: the offline render ran both, the DAW's
    node holds one parameter set per type and the LAST wins. The one the node plays is kept."""
    proj = _four(tmp_path)
    session = Session(proj)
    out = session.adopt_stem_state("pad", {"chain": [_sat(3.0), _sat(9.0)]}, chain_only=True)
    sats = [e for e in session.stems["pad"]["chain"] if e["type"] == "saturation"]
    assert [e["drive_db"] for e in sats] == [9.0], sats
    assert any("saturation" in d for d in out["devices_dropped"]), out


def test_ranged_entries_and_slot_devices_are_not_singletons(tmp_path):
    """The control: two EQs, and a whole-file comp beside a time-ranged one, all carry."""
    proj = _four(tmp_path)
    session = Session(proj)
    band = {"freq_hz": 3000, "gain_db": -2.0, "q": 1.0, "shape": "peak"}
    comp = {"type": "comp", "threshold_db": -20.0, "ratio": 2.0, "attack_ms": 10.0,
            "release_ms": 100.0, "makeup_db": 0.0}
    chain = [{"type": "eq", "band": band}, {"type": "eq", "band": band},
             comp, {**comp, "time_range_s": [0.5, 1.0]}]
    out = session.adopt_stem_state("pad", {"chain": chain}, chain_only=True)
    assert out["devices_dropped"] == []
    assert sorted(e["type"] for e in session.stems["pad"]["chain"]) == ["comp", "comp", "eq", "eq"]


def test_a_legacy_session_with_two_of_a_singleton_loads_with_one(tmp_path):
    proj = _four(tmp_path)
    Session(proj)._persist()
    path = proj / "session.json"
    state = json.loads(path.read_text())
    state["stems"]["pad"]["chain"] = [_sat(3.0), _sat(9.0)]
    path.write_text(json.dumps(state))
    session = Session(proj)
    sats = [e for e in session.stems["pad"]["chain"] if e["type"] == "saturation"]
    assert [e["drive_db"] for e in sats] == [9.0], sats
    audio, _ = dsp.render_stem(session, "pad")           # ...and it renders
    assert np.isfinite(audio).all()


# ====================================================================================
# D8·R9 - a track deleted while its file was away stays deleted when the file returns
# ====================================================================================

def _lose_and_return(proj: Path, stem: str, between) -> Session:
    wav, gone = proj / f"{stem}.wav", proj / f"{stem}.wav.gone"
    wav.rename(gone)
    session = Session(proj)
    assert session.stems[stem]["missing"] is True and session.stems[stem]["excluded"] is True
    between(session)
    gone.rename(wav)
    return Session(proj)


def test_a_track_deleted_while_its_file_was_missing_stays_deleted(tmp_path):
    proj = _four(tmp_path)
    Session(proj)._persist()
    back = _lose_and_return(proj, "pad", lambda s: s.user_edit("pad", {"excluded": True}))
    assert back.stems["pad"]["excluded"] is True, "the file's return un-deleted the track"


def test_a_track_the_agent_excluded_while_missing_stays_excluded(tmp_path):
    proj = _four(tmp_path)
    Session(proj)._persist()

    def exclude(session: Session) -> None:
        _apply(session, [{"type": "exclude", "stem_id": "pad", "excluded": True}])

    assert _lose_and_return(proj, "pad", exclude).stems["pad"]["excluded"] is True


def test_a_missing_file_that_returns_untouched_still_rejoins(tmp_path):
    """The control: nobody deleted it, so it comes back into the mix."""
    proj = _four(tmp_path)
    Session(proj)._persist()
    back = _lose_and_return(proj, "pad", lambda s: None)
    assert back.stems["pad"]["excluded"] is False and not back.stems["pad"].get("missing")


# ====================================================================================
# D8·R10 - migration reaches the undo snapshots; a save is on the disk; a newer file is said
# ====================================================================================

def _v1_session(proj: Path) -> None:
    """A state_version 1 file: chains in INSERTION order (comp before eq - v1 rendered by type,
    so eq ran first), and one applied gain move whose snapshot holds that same order."""
    band = {"freq_hz": 3000, "gain_db": -2.0, "q": 1.0, "shape": "peak"}
    comp = {"type": "comp", "threshold_db": -20.0, "ratio": 2.0, "attack_ms": 10.0,
            "release_ms": 100.0, "makeup_db": 0.0}
    chain = [comp, {"type": "eq", "band": band}]
    stem = {"path": "pad.wav", "gain_db": 1.0, "pan": 0.0, "width": 1.0, "chain": chain,
            "rides": [], "muted": False, "excluded": False, "tags": [], "sends": {}, "bus": None}
    prior = {**json.loads(json.dumps(stem)), "gain_db": 0.0}
    proposal = {"proposal_id": "p1", "status": "applied", "applied_seq": 1, "pass_number": 1,
                "rationale": "v1", "predicted_impact": [],
                "actions": [{"type": "gain", "stem_id": "pad", "delta_db": 1.0}],
                "inverse": {"pad": prior}}
    (proj / "session.json").write_text(json.dumps({
        "state_version": 1, "stems": {"pad": stem}, "master_gain_db": 0.0, "master_chain": [],
        "proposals": [proposal], "pass_number": 2, "applied_seq": 1}))


def test_undo_on_a_migrated_v1_project_keeps_the_migrated_order(tmp_path):
    proj = _project(tmp_path, {"pad": _tone(330.0, 2.0)})
    _v1_session(proj)
    session = Session(proj)
    loaded = [e["type"] for e in session.stems["pad"]["chain"]]
    assert loaded == ["eq", "comp"]                       # control: the live chain is migrated
    assert session.undo("p1").ok
    assert [e["type"] for e in session.stems["pad"]["chain"]] == loaded, \
        "the undo put the v1 insertion order back: the comp now runs before the eq"
    assert session.stems["pad"]["gain_db"] == 0.0         # ...and it did take the move back
    after, _ = dsp.render_stem(Session(proj, read_only=True), "pad")
    # what plays is the project as it was before the move: the same file, never moved
    state = json.loads((proj / "session.json").read_text())
    state["proposals"] = []
    (proj / "session.json").write_text(json.dumps(state))
    assert np.array_equal(after, dsp.render_stem(Session(proj, read_only=True), "pad")[0])


def test_a_save_is_synced_to_the_disk_before_it_replaces_the_session(tmp_path, monkeypatch):
    """`_persist` renamed a temp file nothing had flushed: after a power cut the name can point
    at a file with no contents. The file is synced before the rename and the folder after."""
    proj = _four(tmp_path)
    session = Session(proj)
    events: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd):
        events.append("fsync")
        return real_fsync(fd)

    def replace(a, b, *args, **kwargs):
        events.append("replace")
        return real_replace(a, b, *args, **kwargs)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    session.user_edit("pad", {"gain_db": -3.0})
    assert "replace" in events, "the save no longer goes through os.replace - re-read this test"
    at = events.index("replace")
    assert "fsync" in events[:at], "the session was renamed into place unsynced"
    assert "fsync" in events[at + 1:], "the folder entry was not synced after the rename"
    assert json.loads((proj / "session.json").read_text())["stems"]["pad"]["gain_db"] == -3.0
    assert not (proj / "session.json.tmp").exists()


def test_a_session_from_a_newer_build_is_said(tmp_path, capsys):
    proj = _four(tmp_path)
    Session(proj)._persist()
    path = proj / "session.json"
    state = json.loads(path.read_text())
    state["state_version"] = 99
    path.write_text(json.dumps(state))
    capsys.readouterr()
    Session(proj, read_only=True)
    err = capsys.readouterr().err
    assert "newer" in err and "99" in err, err
    Session(proj, read_only=True)                         # the control: this build's own file
    state["state_version"] = 2
    path.write_text(json.dumps(state))
    capsys.readouterr()
    Session(proj, read_only=True)
    assert "newer" not in capsys.readouterr().err


# ====================================================================================
# D9·R11 - an unknown field or name on an agent's action is refused, never dropped
# ====================================================================================

def _request(action: dict) -> dict:
    return {"rationale": "py7", "predicted_impact": [], "actions": [action]}


def test_a_misspelt_time_range_is_refused_by_name():
    """`time_range` for `time_range_s`: the key was dropped and the section move became a
    whole-song one - accepted, applied, and reported as what was asked."""
    with pytest.raises(ValidationError) as refused:
        ProposalRequest.model_validate(_request(
            {"type": "gain", "stem_id": "pad", "delta_db": 3.0, "time_range": [0.0, 2.0]}))
    text = str(refused.value)
    assert "time_range" in text and "time_range_s" in text, text
    # the control: spelt right, it is a ranged move
    ok = ProposalRequest.model_validate(_request(
        {"type": "gain", "stem_id": "pad", "delta_db": 3.0, "time_range_s": [0.0, 2.0]}))
    assert ok.actions[0].time_range_s == (0.0, 2.0)


def test_a_misspelt_parameter_is_refused_not_defaulted():
    with pytest.raises(ValidationError) as refused:
        ProposalRequest.model_validate(_request(
            {"type": "limiter", "stem_id": "master", "ceiling_db": -3.0}))
    assert "ceiling_db" in str(refused.value) and "ceiling_dbtp" in str(refused.value)


def test_an_unknown_field_inside_a_band_is_refused():
    band = {"drive_db": 6.0, "mix": 0.5}
    ok = ProposalRequest.model_validate(_request(          # the control: spelt right
        {"type": "saturation", "stem_id": "pad", "low": band, "mid": band, "high": band}))
    assert ok.actions[0].low.mix == 0.5
    with pytest.raises(ValidationError) as refused:
        ProposalRequest.model_validate(_request(
            {"type": "saturation", "stem_id": "pad", "low": {"drive_db": 6.0, "mixx": 0.5},
             "mid": band, "high": band}))
    assert "unknown field 'mixx'" in str(refused.value) and "low" in str(refused.value)


def test_an_action_with_no_type_is_refused_not_guessed():
    """`{stem_id}` alone validated as a Utility that does nothing: the union took the first
    model whose fields all default."""
    with pytest.raises(ValidationError) as refused:
        ProposalRequest.model_validate(_request({"stem_id": "pad", "delta_db": 3.0}))
    assert "type" in str(refused.value)


def test_a_probe_refuses_an_unknown_field_too():
    with pytest.raises(ValidationError):
        ProbeRequest.model_validate({"actions": [{"type": "gain", "stem_id": "pad",
                                                  "delta_db": 3.0, "time_range": [0.0, 2.0]}],
                                     "metrics": ["lufs_i"]})


def test_the_engines_propose_tool_refuses_the_misspelt_key_and_records_nothing(tmp_path):
    """The door the agent uses: the MCP tool. A refusal there is a tool error the loop hands
    back to the model, and no proposal is left behind."""
    import anyio

    from orchestrator.benchmark import in_memory_engine

    proj = _four(tmp_path)

    async def run() -> tuple[bool, str]:
        async with in_memory_engine(proj) as client:
            result = await client.call_tool("propose_actions", {"req": _request(
                {"type": "gain", "stem_id": "pad", "delta_db": 3.0, "time_range": [0.0, 1.0]})})
            return bool(result.isError), " ".join(getattr(c, "text", "") for c in result.content)

    is_error, text = anyio.run(run)
    assert is_error and "time_range" in text, text
    assert Session(proj, read_only=True)._state["proposals"] == []


def test_a_stored_proposal_with_a_field_this_build_does_not_know_still_applies(tmp_path):
    """The control for the stored path: a record written by another build is not the agent's
    input. `apply` re-validates the stored actions, and must not refuse them for a key."""
    proj = _four(tmp_path)
    session = Session(proj)
    result = session.propose(ProposalRequest.model_validate(_request(
        {"type": "gain", "stem_id": "pad", "delta_db": 2.0})))
    assert result.accepted
    path = proj / "session.json"
    state = json.loads(path.read_text())
    state["proposals"][-1]["actions"][0]["from_another_build"] = 1
    path.write_text(json.dumps(state))
    again = Session(proj)
    assert again.apply(result.proposal_id).ok
    assert again.stems["pad"]["gain_db"] == 2.0


def test_a_copied_chain_keeps_every_device_with_its_storage_keys(tmp_path):
    """The control the triage named: stored chain entries carry keys no action model has
    (`uid`, `enabled`, `applied_seq`). A strict door for the agent must not drop them."""
    proj = _four(tmp_path)
    session = Session(proj)
    _apply(session, [{"type": "saturation", "stem_id": "kick", "drive_db": 4.0},
                     {"type": "eq", "stem_id": "kick",
                      "band": {"freq_hz": 3000, "gain_db": -2.0, "q": 1.0, "shape": "peak"}}])
    for entry in session.stems["kick"]["chain"]:
        entry["enabled"] = True
    session._persist()
    assert all("uid" in e for e in session.stems["kick"]["chain"])
    out = session.copy_stem_state("kick", "snare")
    assert out["devices_dropped"] == []
    assert sorted(out["devices"]) == ["eq", "saturation"]


@pytest.mark.parametrize("style, index", [("modern", 0), ("Transparent", 1), (" punchy ", 2), (2, 2)])
def test_a_limiter_style_by_name_or_number(style, index):
    assert LimiterAction(stem_id="master", style=style).style == index


def test_an_unknown_limiter_style_is_refused():
    """`style: "loud"` was read as style 0 (Modern)."""
    with pytest.raises(ValidationError) as refused:
        LimiterAction(stem_id="master", style="loud")
    text = str(refused.value)
    assert "loud" in text and "transparent" in text and "punchy" in text, text


@pytest.mark.parametrize("curve, index", [("tanh", 0), ("Warm", 1), ("tube", 2), ("hard", 3), (3, 3)])
def test_a_saturation_curve_by_name_or_number(curve, index):
    assert SaturationAction(stem_id="pad", curve=curve).curve == index


def test_an_unknown_saturation_curve_is_refused():
    """`curve: "fuzz"` was read as curve 0 (tanh) - on the device and on a band of it."""
    with pytest.raises(ValidationError) as refused:
        SaturationAction(stem_id="pad", curve="fuzz")
    assert "fuzz" in str(refused.value) and "tube" in str(refused.value)
    with pytest.raises(ValidationError):
        SaturationAction(stem_id="pad", low={"drive_db": 3.0, "curve": "fuzz"})


# ====================================================================================
# E4·R6 (+ lead) - every door that resolves chains for the DAW fits a match-EQ at the
# project's own rate
# ====================================================================================

def _match_eq_project(tmp_path, sr: int = 22050) -> Path:
    """A 22.05 kHz project with a master match-EQ that asks for the AIR band: above this
    project's Nyquist, so the offline FIR applies nothing and the bank fitted at the project's
    rate is empty - while one fitted at the 48 kHz fallback is not. The widest disagreement
    the rate can make, which is what makes a missing rate visible."""
    rng = np.random.default_rng(7)
    proj = _project(tmp_path, {"pad": 0.1 * rng.standard_normal(2 * sr)}, sr=sr)
    session = Session(proj)
    _apply(session, [{"type": "match_eq", "stem_id": "master", "band_gains_db": {"air": -9.0}}])
    assert session.project_sample_rate() == sr
    return proj


def _master_match_bands(chains: dict) -> list:
    return [e.get("bands") for e in chains["master"]["entries"] if e.get("type") == "match_eq"]


def test_the_fit_depends_on_the_rate_at_all(tmp_path):
    """The control for the three below: at this project's rate the bank is empty, at the
    fallback it is not."""
    proj = _match_eq_project(tmp_path)
    state = json.loads((proj / "session.json").read_text())
    at_rate = daw_bridge.resolve_chains(state, sample_rate=22050)
    fallback = daw_bridge.resolve_chains(state)
    assert not any(_master_match_bands(at_rate))
    assert any(_master_match_bands(fallback))


def test_get_chains_fits_at_the_projects_rate(tmp_path, monkeypatch):
    """`get_chains` is what the DAW re-reads after a turn whose moves were queued: it shipped
    the bank fitted at the 48 kHz fallback until the next full push."""
    from orchestrator import daw_server

    monkeypatch.setenv("MIX_AGENT_DATA_DIR", str(tmp_path / "data"))
    proj = _match_eq_project(tmp_path)
    got = daw_server.Brain(proj).get_chains()
    pushed = daw_bridge.chains_with_readout(proj)["chains"]
    assert _master_match_bands(got) == _master_match_bands(pushed)
    assert not any(_master_match_bands(got)), "the bank was fitted at 48 kHz"


def test_the_chains_the_agent_is_shown_are_fitted_at_the_projects_rate(tmp_path):
    proj = _match_eq_project(tmp_path)
    text = daw_bridge._chat_context_stable(proj, "agent", {}, {})
    line = next(part for part in text.split("\n") if "Applied chains: " in part)
    shown = json.loads(line.split("Applied chains: ", 1)[1])
    assert "master" in shown
    assert not any(_master_match_bands(shown)), "the agent was shown the 48 kHz fit"


def test_the_cli_writes_chains_fitted_at_the_projects_rate(tmp_path, monkeypatch, capsys):
    proj = _match_eq_project(tmp_path)
    out = tmp_path / "chains.json"
    monkeypatch.setattr(sys, "argv", ["daw_bridge", str(proj), "--out", str(out)])
    daw_bridge.main()
    assert not any(_master_match_bands(json.loads(out.read_text())))


# ====================================================================================
# E4·R8 - the offline detector hears a channel, not the sum of two
# ====================================================================================

def _stereo(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.stack([left, right], axis=1)                # (frames, channels)


def _db(x: np.ndarray) -> float:
    return 20.0 * math.log10(max(float(np.sqrt(np.mean(np.square(x)))), 1e-12))


def test_the_block_envelope_of_anti_phase_audio_is_the_channels_level():
    """`_block_env_db` took the RMS of the channel MEAN: content that is out of phase between
    the channels cancelled before it was measured, so five offline devices never saw it - and
    the DAW's, which follow the louder channel, acted at full depth."""
    tone = _tone(3000.0, 1.0, level=0.25)
    in_phase = dsp._block_env_db(_stereo(tone, tone), SR)
    anti = dsp._block_env_db(_stereo(tone, -tone), SR)
    level = 20.0 * math.log10(0.25 / math.sqrt(2.0))
    assert np.median(in_phase) == pytest.approx(level, abs=0.1)        # the control
    assert np.median(anti) == pytest.approx(level, abs=0.1), "anti-phase content read as silence"
    # one channel alone reads its own level, not 6 dB under it
    assert np.median(dsp._block_env_db(_stereo(tone, 0.0 * tone), SR)) == pytest.approx(level, abs=0.1)


def test_dual_mono_reads_exactly_what_it_did():
    """The control that keeps every mono fixture's render where it was: on two equal channels
    the linked reading IS the old one, sample for sample."""
    rng = np.random.default_rng(11)
    x = 0.2 * rng.standard_normal(SR)
    mono = x.copy()
    block = max(1, int(dsp.ENV_BLOCK_S * SR))
    n = -(-mono.shape[0] // block)
    padded = np.zeros(n * block)
    padded[: mono.shape[0]] = mono
    old = 20.0 * np.log10(np.maximum(np.sqrt(np.mean(padded.reshape(n, block) ** 2, axis=1)),
                                     10.0 ** (dsp.ENV_FLOOR_DB / 20.0)))
    assert np.array_equal(dsp._block_env_db(_stereo(x, x), SR), old)


DYN_EQ = {"type": "dynamic_eq", "freq_hz": 3000.0, "q": 2.0, "threshold_db": -40.0, "ratio": 8.0,
          "gain_db": -6.0, "attack_ms": 5.0, "release_ms": 80.0}


def test_a_dynamic_eq_cuts_anti_phase_band_content():
    tone = _tone(3000.0, 2.0, level=0.25)
    for name, audio in (("in phase", _stereo(tone, tone)), ("anti-phase", _stereo(tone, -tone))):
        out = dsp._dynamic_eq(audio.T.astype(np.float32), SR, DYN_EQ)        # (channels, frames)
        cut = _db(out[0, SR // 2:]) - _db(tone[SR // 2:])
        assert cut == pytest.approx(-6.0, abs=0.3), f"{name}: the band moved {cut:+.2f} dB"


def test_a_duck_follows_an_anti_phase_key():
    bed = _tone(220.0, 2.0, level=0.2)
    key = _tone(90.0, 2.0, level=0.3)
    entry = {"depth_db": 9.0, "threshold_db": -30.0, "attack_ms": 5.0, "release_ms": 80.0}
    for name, k in (("in phase", _stereo(key, key)), ("anti-phase", _stereo(key, -key))):
        out = dsp._duck(_stereo(bed, bed), SR, k, entry)
        moved = _db(out[SR // 2:, 0]) - _db(bed[SR // 2:])
        assert moved == pytest.approx(-9.0, abs=0.5), f"{name}: the bed moved {moved:+.2f} dB"


def test_a_transient_shaper_sees_anti_phase_hits():
    n = 2 * SR
    hits = np.zeros(n)
    rng = np.random.default_rng(5)
    for start in range(SR // 4, n - SR // 4, SR // 4):
        burst = rng.standard_normal(SR // 50) * np.exp(-np.arange(SR // 50) / (SR * 0.004))
        hits[start:start + burst.size] += 0.5 * burst
    entry = {"attack_db": 6.0, "sustain_db": 0.0}
    peaks = {}
    for name, audio in (("in phase", _stereo(hits, hits)), ("anti-phase", _stereo(hits, -hits))):
        peaks[name] = float(np.max(np.abs(dsp._transient(audio, SR, entry)[:, 0])))
    lifted = 20.0 * math.log10(peaks["in phase"] / float(np.max(np.abs(hits))))
    assert lifted > 2.0                                                    # the control: it shapes
    assert peaks["anti-phase"] == pytest.approx(peaks["in phase"], rel=1e-6), peaks


def test_ott_and_the_multiband_compressor_level_anti_phase_content():
    tone = _tone(3000.0, 2.0, level=0.5)
    ott = {"type": "ott", "depth": 1.0}
    mb = {"type": "mb_comp", "attack_ms": 5.0, "release_ms": 80.0,
          "low": {"threshold_db": -30.0, "ratio": 4.0}, "mid": {"threshold_db": -30.0, "ratio": 4.0},
          "high": {"threshold_db": -30.0, "ratio": 4.0, "mode": "up"}}
    for label, fn, entry in (("ott", dsp._ott, ott), ("mb_comp", dsp._multiband_comp, mb)):
        got = {}
        for name, audio in (("in phase", _stereo(tone, tone)), ("anti-phase", _stereo(tone, -tone))):
            out = fn(audio.T.astype(np.float32), SR, entry)
            got[name] = _db(out[0, SR // 2:]) - _db(tone[SR // 2:])
        assert abs(got["in phase"]) > 1.0, f"{label}: the fixture does not engage it ({got})"
        assert got["anti-phase"] == pytest.approx(got["in phase"], abs=0.2), f"{label}: {got}"


# ====================================================================================
# E5·R9 - a bass's low notes: in tune reads in tune, and below the floor is said as that
# ====================================================================================

def _bass_line(midi_notes, cents: float = 0.0, sr: int = SR, note_s: float = 0.6) -> np.ndarray:
    """A monophonic bass: each note a harmonic tone (six partials at 1/k), faded at its ends."""
    n = int(note_s * sr)
    t = np.arange(n) / sr
    fade = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.02)
    out = []
    for note in midi_notes:
        f0 = 440.0 * 2.0 ** ((note - 69 + cents / 100.0) / 12.0)
        out.append(fade * sum(0.3 / k * np.sin(2.0 * np.pi * k * f0 * t) for k in range(1, 7)))
    return np.concatenate(out)[:, np.newaxis]


E2_TO_B2 = (40, 41, 43, 45, 47)          # 82.4 .. 123.5 Hz
E1_TO_B1 = (28, 29, 31, 33, 35)          # 41.2 .. 61.7 Hz - a four-string's open-E octave
A3_TO_E4 = (57, 59, 60, 62, 64)


@pytest.mark.parametrize("notes, within", [(E2_TO_B2, 2.0), (A3_TO_E4, 1.0)], ids=["E2-B2", "A3-E4"])
@pytest.mark.parametrize("cents", [0.0, -20.0])
def test_a_notes_cents_are_read_true_down_to_the_floor(notes, within, cents):
    """E2-B2 read ~4 cents sharp, and 20 cents flat read as 15-16: the autocorrelation was
    normalised by lag 0 alone, so it fell with the lag (fewer samples overlap) and the peak
    was pulled to a shorter one - by more the longer the period. (Surveyed after the fix,
    C#2..E5 at -35..+35 cents and both base rates: within 1 cent from C3 up, 2.3 in the
    octave below, where a frame holds three to five periods.)"""
    result = analysis.analyze_notes(_bass_line(notes, cents), SR, "bass", max_notes=16)
    assert result.reason is None and len(result.notes) == len(notes), result
    assert sorted(n.midi_note for n in result.notes) == sorted(notes)
    worst = max(abs(n.cents_off - cents) for n in result.notes)
    assert worst <= within, [(n.note_name, n.cents_off) for n in result.notes]


def test_a_bass_below_the_tracking_floor_is_told_the_floor():
    """E1-B1 was refused with "this needs a MONOPHONIC source": the part IS monophonic, it
    plays below the 65 Hz the tracker can follow. The reason sends the agent to the wrong
    conclusion about the source."""
    result = analysis.analyze_notes(_bass_line(E1_TO_B1), SR, "bass")
    assert result.notes == [] and result.reason
    assert "65 Hz" in result.reason, result.reason
    assert "MONOPHONIC" not in result.reason, result.reason


def test_noise_is_still_told_it_has_no_pitch():
    """The control: what has no pitch at any lag keeps the reason it had."""
    rng = np.random.default_rng(2)
    result = analysis.analyze_notes(0.2 * rng.standard_normal((2 * SR, 1)), SR, "hat")
    assert result.notes == [] and "MONOPHONIC" in (result.reason or "")
    assert "65 Hz" not in result.reason


# ====================================================================================
# E5·R10 - three inputs that took a whole tool down
# ====================================================================================

def _call_tool(proj: Path, name: str, args: dict):
    import anyio

    from orchestrator.benchmark import in_memory_engine

    async def run():
        async with in_memory_engine(proj) as client:
            result = await client.call_tool(name, args)
            text = " ".join(getattr(c, "text", "") for c in result.content)
            return bool(result.isError), text

    return anyio.run(run)


@pytest.mark.parametrize("torn", ['{"entries": [{"key": {"path"', "[1, 2, 3]", "", '{"entries": 7}',
                                  '{"entries": [{"key": null, "result": {"nope": 1}}]}'])
def test_a_torn_reference_cache_is_a_cache_miss(tmp_path, torn):
    """`.reference_cache.json` cut short by a crash made `analyze_reference` raise on every
    retry, until someone deleted a hidden file. A cache that cannot be read is no cache."""
    proj = _four(tmp_path)
    rng = np.random.default_rng(4)
    _write(tmp_path / "ref.wav", 0.1 * rng.standard_normal((2 * SR, 2)))
    (proj / ".reference_cache.json").write_text(torn)
    is_error, text = _call_tool(proj, "analyze_reference", {"path": str(tmp_path / "ref.wav")})
    assert not is_error, text
    assert json.loads(text)["lufs_i"] is not None
    # ...and the cache it leaves is whole, and is used (the control: a second call hits it)
    cached = json.loads((proj / ".reference_cache.json").read_text())
    assert any(isinstance(e.get("key"), dict) and e.get("result") for e in cached["entries"])
    assert not list(proj.glob(".reference_cache.json.*")), "a temp file was left behind"
    is_error, again = _call_tool(proj, "analyze_reference", {"path": str(tmp_path / "ref.wav")})
    assert not is_error and json.loads(again) == json.loads(text)


def test_a_cache_entry_with_no_result_in_it_is_a_miss_and_the_write_is_whole(tmp_path, monkeypatch):
    """The key matches and the entry is not a result (a cache from a build whose fields
    differ, or a half-written one that still parses): analysed afresh. And the cache is
    written to a temp file and renamed over - never in place, which is how it was cut short."""
    proj = _four(tmp_path)
    ref = tmp_path / "ref.wav"
    _write(ref, 0.1 * np.random.default_rng(4).standard_normal((2 * SR, 2)))
    key = {"path": str(ref.resolve()), "mtime_ns": ref.stat().st_mtime_ns,
           "analysis_version": ANALYSIS_VERSION}
    cache = proj / ".reference_cache.json"
    cache.write_text(json.dumps({"entries": [{"key": key, "result": {"nope": 1}}]}))
    renamed: list[str] = []
    real_replace = os.replace

    def replace(a, b, *args, **kwargs):
        renamed.append(Path(b).name)
        return real_replace(a, b, *args, **kwargs)

    monkeypatch.setattr(os, "replace", replace)
    is_error, text = _call_tool(proj, "analyze_reference", {"path": str(ref)})
    assert not is_error, text
    assert json.loads(text)["lufs_i"] is not None
    assert ".reference_cache.json" in renamed, "the cache was written in place"
    entries = json.loads(cache.read_text())["entries"]
    assert [e["result"].get("lufs_i") is not None for e in entries if e["key"] == key] == [True]


def test_a_six_channel_stem_does_not_take_the_readout_down(tmp_path):
    """A 5.1 wav registered as a stem: every readout of the project raised ("Audio must have
    five channels or less") until the track was deleted. A stereo track plays the file's
    first two channels, and so does the render."""
    rng = np.random.default_rng(6)
    front = 0.2 * rng.standard_normal((2 * SR, 2))
    rest = 0.2 * rng.standard_normal((2 * SR, 4))
    proj = _project(tmp_path, {"surround": np.hstack([front, rest]), "pad": _tone(330.0, 2.0)})
    assert sf.info(str(proj / "surround.wav")).channels == 6
    readout = daw_bridge.session_readout(proj)
    assert readout.get("lufs_i") is not None
    audio, _ = dsp.render_stem(Session(proj, read_only=True), "surround")
    assert audio.shape[1] == 2
    assert np.allclose(audio, front.astype(np.float32), atol=1e-6)        # the front pair, as it is
    # the same answer as that pair alone in a stereo file (the control)
    twin = _project(tmp_path / "twin", {"surround": front, "pad": _tone(330.0, 2.0)})
    assert daw_bridge.session_readout(twin)["lufs_i"] == readout["lufs_i"]


def test_a_dc_region_is_described(tmp_path):
    """A region that holds only DC has no band above the masking content floor; with one other
    stem playing, `describe_region` raised instead of saying nothing masks it."""
    music = np.stack([_tone(330.0, 2.0)] * 2, axis=1)
    dc = np.full((2 * SR, 2), 0.2)
    region = analysis.describe_region(dc, SR, "dc", 0.0, 2.0, others={"music": music})
    assert region.masked_by == [] and region.masks == []
    # the control: music against music still reports its masker
    rng = np.random.default_rng(8)
    loud = np.stack([_tone(330.0, 2.0, level=0.6)] * 2, axis=1) + 0.3 * rng.standard_normal((2 * SR, 2))
    assert analysis.describe_region(music, SR, "music", 0.0, 2.0, others={"loud": loud}).masked_by


# ====================================================================================
# D4·R13 (+ lead) - a file too short to be audio breaks nothing, and is not taken as a track
# ====================================================================================

@pytest.mark.parametrize("channels", [1, 2], ids=["mono", "stereo"])
@pytest.mark.parametrize("frames, sr", [(1, 44100), (2, SR), (1, SR), (3, SR)])
def test_a_file_of_a_few_samples_does_not_break_the_readout(tmp_path, channels, frames, sr):
    """One frame at 44.1 k in a 48 k project is two frames after the resample: a 2 x 2 array,
    which the float32 device stages cannot tell channels from frames in - and every readout
    of the project raised until the track was deleted. Mono as well as stereo since a mono
    file renders as two channels; a two-frame file at the project's own rate is the same
    array."""
    proj = _project(tmp_path, {"pad": _tone(330.0, 2.0)})
    _write(proj / "blip.wav", np.full((frames, channels), 0.25), sr)
    session = Session(proj)
    assert "blip" in session.stems                         # the folder door took it as a stem
    _apply(session, [{"type": "saturation", "stem_id": "blip", "drive_db": 3.0}])
    report: dict = {}
    dsp.render_master(Session(proj, read_only=True), report=report)
    assert report.get("short_stems") == ["blip"], report
    readout = daw_bridge.session_readout(proj)
    assert readout.get("lufs_i") is not None
    assert readout.get("short_stems") == ["blip"]


def test_a_project_of_only_a_short_file_still_answers(tmp_path):
    proj = _project(tmp_path, {"blip": np.full((1, 2), 0.25)})
    readout = daw_bridge.session_readout(proj)
    assert isinstance(readout, dict)


@pytest.mark.parametrize("channels", [1, 2], ids=["mono", "stereo"])
def test_importing_a_file_of_a_few_samples_is_refused_and_registers_nothing(tmp_path, channels):
    proj = _project(tmp_path, {"pad": _tone(330.0, 2.0)})
    Session(proj)._persist()
    _write(tmp_path / "blip.wav", np.full((1, channels), 0.25), 44100)
    out = daw_bridge.import_stem(proj, str(tmp_path / "blip.wav"))
    assert out["ok"] is False and "too short" in out["error"], out
    assert not (proj / "blip.wav").exists()
    assert "blip" not in Session(proj, read_only=True).stems
    assert daw_bridge.session_readout(proj).get("lufs_i") is not None


def test_importing_a_short_one_shot_still_registers(tmp_path):
    """The control: a 20 ms click is audio, and is a track."""
    proj = _project(tmp_path, {"pad": _tone(330.0, 2.0)})
    Session(proj)._persist()
    _write(tmp_path / "click.wav", _tone(1000.0, 0.02)[:, np.newaxis])
    out = daw_bridge.import_stem(proj, str(tmp_path / "click.wav"))
    assert out["ok"], out
    assert "click" in Session(proj, read_only=True).stems
    assert daw_bridge.session_readout(proj).get("lufs_i") is not None


# ====================================================================================
# E5·R11 - a note the part plays is not a resonance (analysis v9)
# ====================================================================================

def _held_notes(vibrato_cents: float = 0.0, seconds: float = 12.0, seed: int = 1,
                notes=(57, 60, 64, 62, 57, 65, 64, 60)) -> np.ndarray:
    """A sung / played line: eight held notes (A3 C4 E4 D4 A3 F4 E4 C4), eight partials at
    1/k each, a 5.5 Hz vibrato of the given depth, over a noise bed 45 dB down."""
    rng = np.random.default_rng(seed)
    n = int(SR * seconds)
    t = np.arange(n) / SR
    x = np.zeros(n)
    seg = n // len(notes)
    for i, note in enumerate(notes):
        f0 = 440.0 * 2.0 ** ((note - 69) / 12.0)
        tt = t[i * seg:(i + 1) * seg] - t[i * seg]
        swing = vibrato_cents / 1200.0 * math.log(2.0) * f0 / 5.5 * -np.cos(2.0 * np.pi * 5.5 * tt)
        env = np.minimum(1.0, tt / 0.02) * np.minimum(1.0, (tt[-1] - tt) / 0.05)
        for k in range(1, 9):
            x[i * seg:(i + 1) * seg] += env * (0.25 / k) * np.sin(2.0 * np.pi * k * f0 * tt + k * swing)
    return x + 10.0 ** (-45.0 / 20.0) * rng.standard_normal(n)


def _with_a_ringing_mode(x: np.ndarray, hz: float, hits: int = 8, amp: float = 0.08) -> np.ndarray:
    """A mode that every note's onset excites and that then decays: a body or a room ringing
    at ITS frequency, whatever is played."""
    y = x.copy()
    seg = x.size // hits
    tt = np.arange(int(0.8 * SR)) / SR
    ring = amp * np.sin(2.0 * np.pi * hz * tt) * np.exp(-tt / 0.15)
    for i in range(hits):
        end = min(x.size, i * seg + ring.size)
        y[i * seg:end] += ring[: end - i * seg]
    return y


def _found(x: np.ndarray) -> list[int]:
    out = analysis._resonances(x[:, np.newaxis], SR)
    return [r.freq_hz for r in out] if out else []


def test_the_resonance_definition_is_version_nine():
    assert ANALYSIS_VERSION >= 9


@pytest.mark.parametrize("vibrato_cents", [0.0, 20.0, 30.0])
def test_held_notes_are_not_resonances(vibrato_cents):
    """Through v8 the three slots of this line held its own partials (2090, 2640 and 990 Hz
    with no vibrato; 439 and 217 Hz - the A4 and the A3 - with it): a peak 8 dB over its
    neighbourhood is what a held note looks like."""
    assert _found(_held_notes(vibrato_cents)) == []


def test_a_ring_between_the_pitches_is_reported_alone():
    """The same line with a mode ringing at 190 Hz (F#3 + 46 cents: no pitch of this part,
    and no harmonic of one). Through v8 the line's own notes took all three slots and the
    ring was not in the list at all."""
    found = _found(_with_a_ringing_mode(_held_notes(), 190.0))
    assert len(found) == 1 and abs(found[0] - 190) <= 6, found


def test_a_ring_a_quarter_tone_from_a_played_note_is_not_weighed_as_the_note():
    """1281 Hz is half way between D#6 and E6 - and E6 is the fourth partial of this line's
    E4, a line 30 dB over the ring at the edge of the ring's +-50 cent span. Weighed whole,
    that span's centre IS the note and the ring was dropped as one."""
    found = _found(_with_a_ringing_mode(_held_notes(), 1281.0))
    assert len(found) == 1 and abs(found[0] - 1281) <= 10, found
    # ...and with the note BELOW it: 804 Hz, 44 cents over the G5 that is C4's third partial
    found = _found(_with_a_ringing_mode(_held_notes(), 804.0))
    assert len(found) == 1 and abs(found[0] - 804) <= 6, found
    # ...and 767 Hz, 38 cents under that same G5: the flank of a louder line on either side
    # (a little louder, so the first bin of the G5's flank is within the rise the stretch allows)
    found = _found(_with_a_ringing_mode(_held_notes(), 767.0, amp=0.1))
    assert len(found) == 1 and abs(found[0] - 767) <= 6, found


def test_the_partials_between_the_pitches_are_the_notes_too():
    """One held A2 with sixteen strong partials. Its 7th, 11th and 13th (770, 1210 and 1430
    Hz) sit 31, 49 and 41 cents off the semitones: without the harmonic half of the rule
    they are what the list would hold."""
    t = np.arange(8 * SR) / SR
    rng = np.random.default_rng(12)
    tone = sum(0.1 * np.sin(2.0 * np.pi * 110.0 * k * t) for k in range(1, 17))
    assert _found(tone + 10.0 ** (-50.0 / 20.0) * rng.standard_normal(t.size)) == []


def test_a_ring_on_a_played_pitch_is_the_note():
    """The limit, pinned: a mode ringing ON a pitch the part plays (220 Hz, its A3) cannot
    be told from the note, and is not reported."""
    assert _found(_with_a_ringing_mode(_held_notes(), 220.0)) == []


def test_unpitched_material_reads_what_it_read():
    """The control: noise has no tuning, so nothing is taken for a note - a ring in it is
    reported exactly as v8 reported it."""
    rng = np.random.default_rng(3)
    noise = 0.05 * rng.standard_normal(SR * 8)
    a = 10.0 ** (14.0 / 40.0)
    w0 = 2.0 * np.pi * 700.0 / SR
    alpha = np.sin(w0) / (2.0 * 16.0)
    b = np.array([1 + alpha * a, -2 * np.cos(w0), 1 - alpha * a])
    den = np.array([1 + alpha / a, -2 * np.cos(w0), 1 - alpha / a])
    ringy = lfilter(b / den[0], den / den[0], noise)[:, np.newaxis]
    assert analysis._tuning(ringy[:, 0], SR) is None
    out = analysis._resonances(ringy, SR)
    assert [(r.freq_hz, r.prominence_db, r.q) for r in out] == [(697, 12.5, 39.7)]   # v8's answer
    assert analysis._resonances(0.05 * rng.standard_normal((SR * 8, 1)), SR) is None


FIXTURE_NOTES = {            # what v8 reported for each: the part's own pitches
    "suno_demo/vocals.wav": [439],                    # A4
    "suno_demo/other.wav": [393, 586],                # G4, D5
    "technique_demo/bassline.wav": [217, 439],        # A3, A4
    "technique_demo/pads.wav": [334, 223, 111],       # E4, A3, A2
    "two_stem_demo/stem_a.wav": [439],                # A4
}


@pytest.mark.parametrize("stem", sorted(FIXTURE_NOTES))
def test_a_fixtures_own_notes_are_no_longer_its_resonances(stem):
    path = ROOT / "test_projects" / stem
    if not path.exists():
        pytest.skip(f"{stem} is generated on first use")
    audio, sr = sf.read(str(path), dtype="float64", always_2d=True)
    found = analysis._resonances(audio, sr) or []
    for note in FIXTURE_NOTES[stem]:
        assert not any(abs(r.freq_hz - note) <= 0.03 * note for r in found), (note, found)


def test_the_technique_fixtures_harsh_tone_is_what_its_lead_reports():
    """The fixture was BUILT with a harsh tone at 3.2 kHz in its loud phrases
    (technique_fixture.HARSH_HZ). v8 reported the lead's E5, E6 and B6 and never the defect."""
    from orchestrator import technique_fixture

    path = ROOT / "test_projects" / "technique_demo" / "lead.wav"
    if not path.exists():
        pytest.skip("technique_demo is generated on first use")
    audio, sr = sf.read(str(path), dtype="float64", always_2d=True)
    found = analysis._resonances(audio, sr)
    assert found and [r.freq_hz for r in found] == [int(technique_fixture.HARSH_HZ)], found


def test_the_console_does_not_offer_to_notch_a_played_note(tmp_path):
    """The user-facing end: "a resonance rings at 439 Hz - notch it" was offered for the
    demo's sine stem, whose 440 Hz IS the stem."""
    src = ROOT / "test_projects" / "two_stem_demo"
    if not (src / "stem_a.wav").exists():
        pytest.skip("two_stem_demo is generated on first use")
    proj = tmp_path / "demo"
    proj.mkdir()
    for wav in src.glob("*.wav"):
        shutil.copy(wav, proj / wav.name)
    said = " | ".join(o["text"] for o in daw_bridge.session_observations(proj, cap=8))
    assert "resonance" not in said, said
