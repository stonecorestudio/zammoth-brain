"""Gates for batch Py2 of the reader P2s (full audit 2026-10-02): names and order in the session.

Six findings - D2·R14, D8·R3, D8·R5, E6·R4, S1·R17, E2·R4 - and the two leads the triage filed
with them (a return's level through `set_master_param`; an all-upper-case stems folder through
the DAW's registration). Each test drives the door the product uses and carries its control.
Every one was run on the unfixed tree first. The record is
docs/audits/full-2026-10-02/readers/P2-fixes.md (Py2).
"""
from __future__ import annotations

import asyncio
import os
import shutil

import numpy as np
import pytest
import soundfile as sf

from mix_engine.models import ProposalRequest
from mix_engine.session import Session, minting, stems_by_registration
from orchestrator import automix
from orchestrator.daw_bridge import apply_coedits, import_stem, resolve_chains, returns_topology

_U = "http://127.0.0.1:1/api/"


def _propose(session: Session, actions: list[dict]):
    return session.propose(ProposalRequest(rationale="py2", predicted_impact=[], actions=actions))


def _apply(session: Session, actions: list[dict]) -> str:
    result = _propose(session, actions)
    assert result.accepted, result.violations
    assert session.apply(result.proposal_id).ok
    return result.proposal_id


def _tone(path, hz: float = 220.0, seconds: float = 2.0, amp: float = 0.2) -> None:
    t = np.arange(int(seconds * 44100)) / 44100
    sf.write(path, (amp * np.sin(2 * np.pi * hz * t)).astype("float32"), 44100)


def _project(tmp_path, *names: str):
    proj = tmp_path / "proj"
    proj.mkdir(parents=True)
    for i, name in enumerate(names):
        _tone(proj / name, 110.0 * (i + 1))
    return proj


def _types(chain) -> list[str]:
    return [e["type"] for e in chain]


# ---------- D2·R14 / D8·R3: one namespace, at the door that mints a bus ----------

def _with_a_return(project) -> Session:
    session = Session(project)
    assert session.add_return("blank")["return_id"] == "fx"
    return session


@pytest.mark.parametrize("target, what", [
    ("stem_b", "a track"), ("fx", "a return"), ("a:b", "reserved"), ("bus:x:y", "reserved")])
def test_route_stem_mints_no_bus_under_a_name_something_else_holds(demo_project, target, what):
    """`route.stem {stem: "stem_a", bus: "stem_b"}` - the model naming the other TRACK - minted a
    bus called `stem_b` beside the stem; every bus-first lookup then wrote the bus."""
    session = _with_a_return(demo_project)
    with pytest.raises(ValueError, match=what):
        session.route_stem("stem_a", target)
    assert target not in session.buses and session.stems["stem_a"].get("bus") is None
    assert Session(demo_project).buses == {}                  # nothing reached session.json


def test_the_fader_of_a_track_a_route_named_still_writes_the_track(demo_project):
    """The harm, end to end: after the route, the user's fader and send on the real `stem_b` were
    persisted onto the phantom bus and snapped back on the track."""
    session = _with_a_return(demo_project)
    with pytest.raises(ValueError):
        session.route_stem("stem_a", "stem_b")
    out = apply_coedits(demo_project, [_U + "set_stem?stem=stem_b&gain_db=-9",
                                       _U + "set_send?source=stem_b&return=fx&gain_db=-6"])
    assert out["applied"] == 2, out
    after = Session(demo_project)
    assert after.stems["stem_b"]["gain_db"] == -9.0 and after.stems["stem_b"]["sends"] == {"fx": -6.0}


def test_route_stem_still_makes_and_reuses_a_group_by_a_free_name(demo_project):
    """The control: a free explicit id is made, "new" mints, an existing group is joined, and the
    master is the master however it is spelt."""
    session = _with_a_return(demo_project)
    assert session.route_stem("stem_a", "Drum Bus") == {"stem": "stem_a", "bus": "Drum Bus"}
    assert session.route_stem("stem_b", "Drum Bus")["bus"] == "Drum Bus"
    assert session.route_stem("stem_b", "new")["bus"] == "bus1"
    for spelling in ("master", "Master", "MASTER", ""):
        assert session.route_stem("stem_b", spelling)["bus"] is None
        assert session.route_stem("stem_b", "Drum Bus")["bus"] == "Drum Bus"
    assert sorted(session.buses) == ["Drum Bus"]


def test_every_door_that_names_a_group_or_a_return_asks_the_one_rule(demo_project):
    """One predicate, so the doors cannot disagree again: what `route_stem` refuses, a rename, a
    proposal's group and a proposal's return refuse too - "Master" in any case included, which
    only the stem doors refused before."""
    session = _with_a_return(demo_project)
    session.route_stem("stem_a", "new")
    for name in ("stem_b", "fx", "Master", "a:b"):
        assert session.name_taken(name, "bus"), name
        with pytest.raises(ValueError, match="taken or reserved"):
            session.rename_bus("bus1", name)
        result = _propose(session, [{"type": "group", "bus_id": name, "members": ["stem_b"]}])
        assert not result.accepted and {v.rule for v in result.violations} == {"bus_name_taken"}
    for name in ("stem_b", "bus1", "Master", "a:b"):
        assert session.name_taken(name, "return"), name
        with pytest.raises(ValueError, match="taken or reserved"):
            session.rename_return("fx", name)
        result = _propose(session, [{"type": "return_reverb", "return_id": name}])
        assert not result.accepted and "return_name_taken" in {v.rule for v in result.violations}
    assert session.name_taken("Drum Bus", "bus") is None      # the control: a free name is free
    assert session.name_taken("bus1", "bus") is None          # its own table is the caller's to ask
    assert session.name_taken("fx", "return") is None


@pytest.mark.parametrize("taken, what", [("bus1", "group"), ("fx", "return")])
def test_import_refuses_a_stem_named_like_a_group_or_a_return(demo_project, tmp_path, taken, what):
    """The brain's own import door: a stem `bus1` beside the group `bus1` answered ok, and the
    track's fader then wrote the group."""
    session = _with_a_return(demo_project)
    session.route_stem("stem_a", "new")
    src = tmp_path / "take.wav"
    _tone(src, 330.0)
    before = sorted(p.name for p in demo_project.iterdir())
    out = import_stem(demo_project, str(src), taken)
    assert out["ok"] is False and what in out["error"], out
    assert sorted(p.name for p in demo_project.iterdir()) == before          # nothing written
    assert taken not in Session(demo_project).stems
    assert import_stem(demo_project, str(src), "newtrack")["ok"] is True     # the control
    assert import_stem(demo_project, str(demo_project / "stem_a.wav"), "stem_a")["ok"] is True


def test_a_wav_named_like_a_group_is_not_minted_as_a_second_holder_of_the_name(demo_project):
    """The folder door (the offline engine reads the folder): `bus1.wav` copied in beside the
    group `bus1` minted the stem `bus1`. It is left as a file; the session still opens."""
    session = Session(demo_project)
    session.route_stem("stem_a", "new")
    shutil.copy(demo_project / "stem_b.wav", demo_project / "bus1.wav")
    shutil.copy(demo_project / "stem_b.wav", demo_project / "gtr.wav")
    after = Session(demo_project)
    assert "bus1" in after.buses and "bus1" not in after.stems
    assert "gtr" in after.stems                                # the control: a free name is minted


# ---------- lead: a return's level through the device door ----------

def test_a_gain_write_aimed_at_a_return_sets_the_return(demo_project):
    session = _with_a_return(demo_project)
    assert session.set_master_param("gain", 0, -7.0, target="return:fx")["value"] == -7.0
    assert Session(demo_project).returns["fx"]["gain_db"] == -7.0
    assert session.set_master_param("gain", 0, 40.0, target="return:fx")["value"] == 6.0
    with pytest.raises(ValueError, match="unknown"):
        session.set_master_param("gain", 0, -3.0, target="return:nope")
    with pytest.raises(ValueError, match="unknown"):
        session.set_master_param("gain", 0, -3.0, target="nope")
    session.route_stem("stem_a", "new")                        # the controls: the other three kinds
    assert session.set_master_param("gain", 0, -4.0, target="bus1")["value"] == -4.0
    assert session.set_master_param("gain", 0, -5.0, target="stem_b")["value"] == -5.0
    assert session.set_master_param("gain", 0, -2.0)["value"] == -2.0
    again = Session(demo_project)
    assert (again.buses["bus1"]["gain_db"], again.stems["stem_b"]["gain_db"],
            again._state["master_gain_db"]) == (-4.0, -5.0, -2.0)


# ---------- D8·R5: an undo puts a removed return back where it was ----------

def _cascade(project, remove: bool, forget: bool = False) -> Session:
    session = Session(project)
    assert [session.add_return("blank")["return_id"] for _ in range(3)] == ["fx", "fx2", "fx3"]
    session.set_send("return:fx", "fx2", -10.0)
    session.set_send("return:fx2", "fx3", -12.0)
    pid = _apply(session, [{"type": "return_reverb", "return_id": "fx2", "decay_s": 3.0}])
    if remove:
        session.remove_return("fx2")
        assert list(session.returns) == ["fx", "fx3"]
    if forget:
        session._state["routing_tombstones"] = []              # evicted (the list is bounded)
    assert session.undo(pid).ok
    return session


@pytest.mark.parametrize("remove", [False, True])
def test_undo_restores_a_removed_return_at_its_place(demo_project, remove):
    """The agent lengthens the reverb on `fx2`; the user deletes `fx2`; the move is undone. The
    return came back LAST, so its restored send into `fx3` pointed backward and `fx3 -> fx2` was
    accepted: session.json held two returns feeding each other."""
    session = _cascade(demo_project, remove)
    if remove:
        # Since P3 batch Py8 the UNDO does not bring back a return the user removed (it is
        # applied to the buried record); the DAW's own undo of the removal does, and it is
        # that door which must put it back at its place.
        assert list(session.returns) == ["fx", "fx3"]
        assert session.restore_routing("return", "fx2")["restored"]
    assert list(session.returns) == ["fx", "fx2", "fx3"]
    assert session.returns["fx2"]["sends"] == {"fx3": -12.0}
    assert not any(e.get("decay_s") == 3.0 for e in session.returns["fx2"]["chain"]), \
        "the return came back with the move its undo had taken back"
    with pytest.raises(ValueError, match="later return"):
        session.set_send("return:fx3", "fx2", -10.0)
    assert list(Session(demo_project).returns) == ["fx", "fx2", "fx3"]


def test_a_restored_cascade_that_points_backward_is_not_kept(demo_project):
    """With no record of where it was, the return comes back last - and a send it then holds into
    an EARLIER return is dropped, as both engines drop it: the picture says what plays."""
    session = _cascade(demo_project, remove=True, forget=True)
    assert list(session.returns) == ["fx", "fx3"]                # Py8: the undo re-creates nothing
    assert session.restore_routing("return", "fx2")["restored"]  # no tombstone: back, last
    assert list(session.returns) == ["fx", "fx3", "fx2"]
    session._state["returns"]["fx2"]["sends"] = {"fx3": -12.0}   # the cascade it used to hold
    session._reconcile_routing_refs()
    session._persist()
    assert session.returns["fx2"]["sends"] == {}
    assert "return:fx2" not in returns_topology(Session(demo_project)._state)["sends"]
    # the control: a forward send set afterwards is kept by the same reconcile
    session.set_send("return:fx3", "fx2", -9.0)
    session._reconcile_routing_refs()
    assert session.returns["fx3"]["sends"] == {"fx2": -9.0}
    assert returns_topology(session._state)["sends"] == {"return:fx3": {"fx2": -9.0}}


def test_a_group_an_undo_brings_back_keeps_its_place_too(demo_project):
    """The move touched the group AND its member, so the undo puts the member back in it: the
    group returns with something in it, at the place it left. (A group whose undo would bring
    it back EMPTY is not brought back at all since P3 batch Py7, reader D8·R7 - this test
    used to be that case.)"""
    session = Session(demo_project)
    session.route_stem("stem_a", "A")
    session.route_stem("stem_b", "B")
    pid = _apply(session, [{"type": "comp", "stem_id": "A", "threshold_db": -18.0, "ratio": 2.0,
                            "attack_ms": 10.0, "release_ms": 100.0, "makeup_db": 0.0},
                           {"type": "gain", "stem_id": "stem_a", "delta_db": -1.0}])
    session.remove_bus("A")
    assert list(session.buses) == ["B"]
    assert session.undo(pid).ok
    assert list(session.buses) == ["A", "B"]
    assert session.stems["stem_a"]["bus"] == "A"


# ---------- E6·R4: Auto Mix on a project whose stem is called `vox` ----------

_VOX_TAGS = {"vox": ["vocal"], "bgv": ["vocal"], "kick": ["kick"], "hats": ["cymbal"]}


def _moving(tags: dict) -> dict:
    return {sid: {"lufs_s": -20.0, "band_energy": {"sub": -60.0}, "dynamic_spread_db": 18.0,
                  "transient_density": 1.0, "transient_energy_ratio": 0.4} for sid in tags}


def _vox_project(tmp_path):
    return _project(tmp_path, *(f"{sid}.wav" for sid in _VOX_TAGS))


def _bus_bundles(bundles: list[dict]) -> list[dict]:
    return [b for b in bundles if any(a["type"] == "group" for a in b["actions"])]


def test_auto_mix_groups_the_vocals_of_a_project_with_a_stem_called_vox(tmp_path):
    """`vox.wav`, `bgv.wav`, `kick.wav`, `hats.wav`: the vocal group was named `vox`, the name of
    a stem, and rode ONE bundle with the drum group - both refused, and no member compressed."""
    project = _vox_project(tmp_path)
    bundles = automix.build_bundles(_VOX_TAGS, _moving(_VOX_TAGS), [], {"lufs_i": -18.0},
                                    goal_lufs=None)
    buses = _bus_bundles(bundles)
    ids = [a["bus_id"] for b in buses for a in b["actions"] if a["type"] == "group"]
    assert len(buses) == 2 and len(ids) == 2                   # a family a bundle: a refusal stays local
    assert "drums" in ids and not set(ids) & set(_VOX_TAGS)
    session = Session(project)
    for bundle in buses:
        result = _propose(session, bundle["actions"])
        assert result.accepted, result.violations
        assert session.apply(result.proposal_id).ok
    vocal_bus = next(i for i in ids if i != "drums")
    assert session.stems["vox"]["bus"] == session.stems["bgv"]["bus"] == vocal_bus
    assert _types(session.buses[vocal_bus]["chain"]) == ["comp"]
    assert _types(session.buses["drums"]["chain"]) == ["comp"]


def test_auto_mix_keeps_the_family_names_where_they_are_free():
    tags = {"lead": ["vocal"], "bgv": ["vocal"], "kick": ["kick"], "hats": ["cymbal"]}
    bundles = automix.build_bundles(tags, _moving(tags), [], {"lufs_i": -18.0}, goal_lufs=None)
    ids = [a["bus_id"] for b in bundles for a in b["actions"] if a["type"] == "group"]
    assert ids == ["drums", "vox"]


def test_members_of_a_refused_group_get_their_own_compressors(tmp_path):
    """A RETURN called `drums` is a collision the policy cannot see (it is handed stems). The
    group is refused - and its members, who were left without a compressor on the promise of the
    group's glue, get their own. The vocal group still lands."""
    from test_phase10_efficiency import in_memory_engine
    from orchestrator.loop import EngineClient

    project = _vox_project(tmp_path)
    session = Session(project)
    session.add_return("blank")
    session.rename_return("fx", "drums")
    bundles = automix.build_bundles(_VOX_TAGS, _moving(_VOX_TAGS), [], {"lufs_i": -18.0},
                                    goal_lufs=None)
    rejected: list[str] = []

    async def run() -> None:
        async def emit(_event) -> None:
            return None
        async with in_memory_engine(project) as client:
            await automix._apply_bundles(EngineClient(client), emit, bundles, rejected=rejected)

    asyncio.run(run())
    after = Session(project)
    assert len(rejected) == 1 and "drums" in rejected[0]
    assert "drums" not in after.buses
    for member in ("kick", "hats"):
        assert "comp" in _types(after.stems[member]["chain"]), member
    vocal_bus = after.stems["vox"]["bus"]
    assert vocal_bus and after.stems["bgv"]["bus"] == vocal_bus
    assert _types(after.buses[vocal_bus]["chain"]) == ["comp"]
    for member in ("vox", "bgv"):                              # the control: glue landed, no member comp
        assert "comp" not in _types(after.stems[member]["chain"]), member


# ---------- S1·R17: KICK.WAV is a stem ----------

def test_an_upper_case_wav_is_a_stem(tmp_path):
    """A folder from a hardware recorder: the DAW lists `*.wav` without regard to case, the brain
    listed it with - the lanes played and the engineer had "no audio to mix"."""
    project = _project(tmp_path, "KICK.WAV", "snare.wav", "Hat.Wav")
    session = Session(project)
    assert sorted(session.stems) == ["Hat", "KICK", "snare"]
    assert session.stems["KICK"]["path"] == "KICK.WAV"         # the file's own name is kept
    assert not any(s.get("missing") or s.get("excluded") for s in session.stems.values())
    again = Session(project)                                   # ...and it is the same stem next time
    assert sorted(again.stems) == ["Hat", "KICK", "snare"]
    from mix_engine import dsp
    audio, rate = dsp.render_stem(again, "KICK")
    assert rate == 44100 and float(np.abs(audio).max()) > 0.1


def test_an_upper_case_folder_registers_through_the_daw_door(tmp_path):
    """The lead: in a project the DAW holds, an all-upper-case folder reached the registration
    and was refused - "TRACK01.wav is in the project, but no stem holds it"."""
    project = _project(tmp_path, "TRACK01.WAV", "TRACK02.WAV")
    stems_by_registration(project)
    try:
        assert Session(project).stems == {}                    # by registration: nothing is minted yet
        out = import_stem(project, str(project / "TRACK01.WAV"), "TRACK01")
        assert out["ok"] is True and out["wav"] == "TRACK01.WAV", out
        assert sorted(p.name for p in project.iterdir() if p.suffix.lower() == ".wav") == [
            "TRACK01.WAV", "TRACK02.WAV"]                      # registered in place: no second copy
        session = Session(project)
        assert sorted(session.stems) == ["TRACK01"]            # ...and only the one that was named
        assert session.stems["TRACK01"]["path"] == "TRACK01.WAV"
        assert import_stem(project, str(project / "TRACK01.WAV"), "TRACK01")["ok"] is True
        assert sorted(Session(project).stems) == ["TRACK01"]   # a retried registration is the same stem
    finally:
        stems_by_registration(project, on=False)


def test_a_file_whose_case_changed_is_still_its_stems_file(tmp_path):
    """Widening the listing must not make a twin: a record that says `kick.wav` over a file now
    called `KICK.WAV` (re-cased in the Finder) is that stem, not a missing one beside a new one.
    Only where the volume itself says the two names are one file."""
    project = _project(tmp_path, "kick.wav", "snare.wav")
    Session(project)
    os.rename(project / "kick.wav", project / "kick.tmp")
    os.rename(project / "kick.tmp", project / "KICK.WAV")
    one_file = (project / "kick.wav").exists()                 # a case-insensitive volume
    session = Session(project)
    if one_file:
        assert sorted(session.stems) == ["kick", "snare"]
        assert session.stems["kick"]["path"] == "KICK.WAV" and not session.stems["kick"].get("missing")
    else:
        assert session.stems["kick"].get("missing") and "KICK" in session.stems


def test_the_discard_and_found_listings_see_an_upper_case_wav(tmp_path):
    """The two other places the brain lists the folder (pre-merge review 2026-10-04): a file the
    user copied in between sessions is noted at the open, so one "Don't Save" does not retire it."""
    from mix_engine.session import SAVED_SESSION_FILE
    from orchestrator.daw_bridge import note_found_wavs

    project = _project(tmp_path, "kick.wav")
    session = Session(project)
    (project / SAVED_SESSION_FILE).write_text(__import__("json").dumps(session._state))
    _tone(project / "TAKE.WAV", 330.0)
    _tone(project / "take2.wav", 440.0)                        # the control: lower case was always seen
    assert note_found_wavs(project) == ["TAKE.WAV", "take2.wav"]


def test_the_brain_lists_the_folder_through_the_one_listing():
    """`Path.glob("*.wav")` is case-sensitive: a new reader of the folder that uses it is this
    finding again. The engine and the bridge go through `session.project_wavs`."""
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    files = [*(repo / "server/mix_engine").glob("*.py"),
             *(repo / "orchestrator" / n for n in ("daw_bridge.py", "daw_server.py", "gateway.py"))]
    assert len(files) > 10
    for path in files:
        code = [l for l in path.read_text(encoding="utf-8").splitlines()
                if "glob(" in l and ".wav" in l and not l.lstrip().startswith(("#", "`"))
                and "Path.glob" not in l]
        assert code == [], f"{path.name} lists wavs with a case-sensitive glob: {code}"


def test_a_declined_sessions_upper_case_wav_is_retired_with_it(tmp_path):
    """…and "Don't Save": a take the declined session brought in is retired, not left for the
    next open to mint as a stem the edit does not play. With the folder read without regard to
    case, a listing that still read it with case would have made exactly that ghost."""
    from mix_engine.session import DISCARD_MARKER_FILE, SAVED_COEDITS_FILE, SAVED_SESSION_FILE, SESSION_FILE
    from orchestrator.daw_bridge import honour_discard

    project = _project(tmp_path, "kick.wav")
    Session(project)._persist()
    shutil.copyfile(project / SESSION_FILE, project / SAVED_SESSION_FILE)     # the DAW saves
    (project / SAVED_COEDITS_FILE).write_text("[]")
    _tone(project / "TAKE.WAV", 330.0)                                         # the session's take
    assert import_stem(project, str(project / "TAKE.WAV"), "TAKE")["ok"] is True
    assert "TAKE" in Session(project).stems
    (project / DISCARD_MARKER_FILE).write_text("declined\n")                   # ...and is declined
    assert honour_discard(project)["retired"] == ["TAKE.WAV"]
    assert sorted(Session(project).stems) == ["kick"]
    assert (project / "TAKE.WAV").exists()                                     # the audio is kept


# ---------- E2·R4: the dock's order cannot name what its chain does not hold ----------

def _dock_order(session: Session, target: str = "master") -> list[str]:
    """The order exactly as the dock builds it: one `type#occurrence` token per entry of the
    NODE's chain - what `resolve_chains` pushed - never the brain's own list."""
    entries = resolve_chains(session._state)[target]["entries"]
    seen: dict[str, int] = {}
    tokens = []
    for entry in entries:
        kind = entry["type"]
        if kind in ("gain", "send_return"):                    # the fader stage and sends have no box
            continue
        tokens.append(f"{kind}#{seen.get(kind, 0)}")
        seen[kind] = seen.get(kind, 0) + 1
    return tokens


def test_a_dock_reorder_leaves_the_master_transpose_ahead_of_the_limiter(demo_project):
    """The user picks a key (transpose on the master), then adds a Saturation in the dock. The
    node never holds `transpose`, so the order the dock posts cannot name it - and every unnamed
    entry went LAST: the pitch shift ran after the ceiling in every offline render."""
    session = Session(demo_project)
    _apply(session, [{"type": "transpose", "stem_id": "master", "semitones": 2.0}])
    session.add_master_device("limiter")
    session.add_master_device("saturation")
    assert _types(session._state["master_chain"])[0] == "transpose"
    order = _dock_order(session)
    assert sorted(order) == ["limiter#0", "saturation#0"]      # the premise: the dock cannot name it
    session.set_master_device_order(["limiter#0", "saturation#0"])
    assert _types(session._state["master_chain"]) == ["transpose", "limiter", "saturation"]
    session.set_master_device_order(["saturation#0", "limiter#0"])
    assert _types(session._state["master_chain"]) == ["transpose", "saturation", "limiter"]
    assert _types(Session(demo_project)._state["master_chain"]) == ["transpose", "saturation", "limiter"]


def test_a_dock_reorder_leaves_a_stems_tune_and_denoise_where_they_were(demo_project):
    session = Session(demo_project)
    _apply(session, [{"type": "tune", "stem_id": "stem_a", "cents": 12.0},
                     {"type": "spectral_denoise", "stem_id": "stem_a", "reduction_db": 6.0}])
    session.add_stem_device("stem_a", "comp")
    session.add_stem_device("stem_a", "saturation")
    before = _types(session.stems["stem_a"]["chain"])
    assert before == ["tune", "spectral_denoise", "comp", "saturation"]
    order = _dock_order(session, "stem_a")
    assert sorted(order) == ["comp#0", "saturation#0"]
    session.set_stem_device_order("stem_a", ["saturation#0", "comp#0"])
    assert _types(session.stems["stem_a"]["chain"]) == [
        "tune", "spectral_denoise", "saturation", "comp"]


def test_what_the_dock_can_name_is_what_the_push_puts_in_a_nodes_chain(demo_project):
    """`dock_can_name` is the consumer's one question; `monitor_chain_params` is the producer.
    Every device the two add doors make is on the node's chain and nameable, and the three the
    agent proposes that are print-only are neither."""
    from mix_engine.monitor_chain import INERT_TYPES, MONITOR_TYPES, dock_can_name
    session = Session(demo_project)
    made = set()
    for kind in sorted((MONITOR_TYPES | INERT_TYPES) - {"gain"}):
        for add, target in ((session.add_stem_device, "stem_a"), (session.add_master_device, None)):
            try:
                add(target, kind) if target else add(kind)
            except (ValueError, KeyError):
                continue                                       # not a device of that chain
            made.add(kind)
    _apply(session, [{"type": "tune", "stem_id": "stem_a", "cents": 12.0},
                     {"type": "spectral_denoise", "stem_id": "stem_a", "reduction_db": 6.0},
                     {"type": "transpose", "stem_id": "master", "semitones": 2.0}])
    assert len(made) >= 20, sorted(made)
    chains = resolve_chains(session._state)
    pushed = {"stem_a": chains["stem_a"], "master": chains["master"]}
    stored = {"stem_a": session.stems["stem_a"]["chain"], "master": session._state["master_chain"]}
    for where in pushed:
        on_node = {e["type"] for e in pushed[where]["entries"]}
        for kind in {e["type"] for e in stored[where]}:
            assert dock_can_name(kind) is (kind in on_node), (where, kind)
    assert set(pushed["stem_a"]["print_only"]) == {"tune", "spectral_denoise"}
    assert pushed["master"]["print_only"] == ["transpose"]


def test_an_unnameable_device_in_the_middle_follows_the_device_before_it(demo_project):
    session = Session(demo_project)
    eq = {"type": "eq", "band": {"shape": "peak", "freq_hz": 100.0, "gain_db": 1.0, "q": 1.0}}
    session._state["master_chain"] = [eq, {"type": "transpose", "semitones": 1.0},
                                      {"type": "comp"}, {"type": "limiter"}]
    session.set_master_device_order(["comp#0", "eq#0", "limiter#0"])
    assert _types(session._state["master_chain"]) == ["comp", "eq", "transpose", "limiter"]


def test_an_order_that_names_or_omits_a_nameable_device_reads_as_before(demo_project):
    """The controls. A token that names a print-only device places it (an old journal can hold
    one); and a NAMEABLE device a partial order leaves out still falls to its canonical slot -
    an unplaced limiter last - which callers rely on."""
    session = Session(demo_project)
    chain = [{"type": "transpose", "semitones": 2.0}, {"type": "eq"}, {"type": "comp"},
             {"type": "limiter"}]
    session._state["master_chain"] = [dict(e) for e in chain]
    session.set_master_device_order(["eq#0", "transpose#0", "comp#0", "limiter#0"])
    assert _types(session._state["master_chain"]) == ["eq", "transpose", "comp", "limiter"]
    session._state["master_chain"] = [{"type": "limiter"}, {"type": "eq"}, {"type": "comp"}]
    session.set_master_device_order(["comp"])
    assert _types(session._state["master_chain"]) == ["comp", "eq", "limiter"]


def test_the_journal_replay_of_a_dock_order_keeps_the_transpose_too(demo_project):
    """The other caller: a journaled `device_order` replayed through `apply_coedits`."""
    session = Session(demo_project)
    _apply(session, [{"type": "transpose", "stem_id": "master", "semitones": 2.0}])
    session.add_master_device("limiter")
    session.add_master_device("saturation")
    out = apply_coedits(demo_project, [_U + "device_order?order=limiter%230,saturation%230"])
    assert out["applied"] == 1, out
    assert _types(Session(demo_project)._state["master_chain"]) == [
        "transpose", "limiter", "saturation"]
