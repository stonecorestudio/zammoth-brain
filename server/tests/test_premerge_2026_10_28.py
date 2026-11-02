"""The pre-merge review of 2026-10-28 (betaV18: nine devices re-fitted to references) - the
brain's half of the fixes, each seen red on the commit the review read.

The Duck: a TRACK duck's hits are a reading of its trigger, answered fresh when the trigger moves
(a co-edit, the tempo map), never for a trigger the DAW plays otherwise than the brain renders it;
a key alone is a TRACK duck; a replace that would drop a keyed duck's trigger is refused;
duck.set_key reads its tuning before it writes and holds the agent to 12 dB.
Old projects: a copied chain is read as a stored one (an old de-esser, a deep dynamic EQ); an old
dynamic EQ / de-esser at ratio 1 is bypassed (it cut nothing); an old clipper's trim is kept as a
Utility; the co-edit journal moves with the laws; the open says what it did.
The EQ rail reads a narrow bell whole and at the project's rate; Auto Mix speaks the EQ law's Q.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from mix_engine.session import Session

REPO = Path(__file__).resolve().parents[2]


def _wav(path: Path, x: np.ndarray, sr: int = 48000) -> None:
    sf.write(str(path), x, sr, subtype="FLOAT")


def _kick_bass_project(tmp_path: Path) -> Path:
    sr, n = 48000, 48000 * 4
    t = np.arange(n) / sr
    bass = np.stack([0.3 * np.sin(2 * np.pi * 55 * t)] * 2, axis=1)
    kick = np.zeros((n, 2))
    for at in range(0, n, sr // 2):
        m = min(n - at, int(0.08 * sr))
        u = np.arange(m) / sr
        kick[at:at + m] += (0.8 * np.exp(-u * 40.0) * np.sin(2 * np.pi * 60.0 * u))[:, None]
    _wav(tmp_path / "bass.wav", bass)
    _wav(tmp_path / "kick.wav", kick)
    # a quiet held pad: under the detector's -20 dBFS, it fires no hit (a sustained part)
    _wav(tmp_path / "pad.wav", np.stack([0.05 * np.sin(2 * np.pi * 220 * t)] * 2, axis=1))
    return tmp_path


# ------------------------------------------------------------------------------------ the Duck

def test_a_duck_that_names_a_key_and_no_trigger_listens_to_the_key():
    from mix_engine.models import SidechainDuckAction
    a = SidechainDuckAction(stem_id="bass", key_stem_id="kick", mix=0.5)
    assert a.trigger == 1                                     # it was 0: a grid pump that ignored its key
    assert SidechainDuckAction(stem_id="bass", mix=0.5).trigger == 0
    assert SidechainDuckAction(stem_id="bass", key_stem_id="kick", trigger="sync", mix=0.5).trigger == 0


def test_a_replace_that_would_drop_a_keyed_ducks_trigger_is_refused(tmp_path):
    from mix_engine.models import ProposalRequest
    project = _kick_bass_project(tmp_path)
    s = Session(project)
    s.set_duck("bass", "kick")
    req = ProposalRequest(rationale="softer", predicted_impact=[],
                          actions=[{"type": "duck", "stem_id": "bass", "mix": 0.4}])
    res = s.propose(req)
    assert not res.accepted
    assert any(v.rule == "duck_keeps_trigger" for v in res.violations), res.violations
    ok = s.propose(ProposalRequest(rationale="softer", predicted_impact=[],
                                   actions=[{"type": "duck", "stem_id": "bass", "key_stem_id": "kick", "mix": 0.4}]))
    assert ok.accepted, ok.violations


def test_duck_set_key_reads_its_tuning_before_it_writes(tmp_path):
    from orchestrator.daw_server import Brain
    project = _kick_bass_project(tmp_path)
    Session(project)._persist()
    brain = Brain(project)
    before = (project / "session.json").read_text()
    for bad in ({"mix": "abc"}, {"mix": [0.5]}, {"length": "1/3"}, {"curve": 99}):
        with pytest.raises(ValueError):
            brain.set_duck("bass", "kick", bad, by_agent=True)
        assert (project / "session.json").read_text() == before, bad   # nothing was keyed


def test_the_agents_duck_set_key_cuts_12_db_at_most(tmp_path):
    from mix_engine.guardrails import LIMITS
    from orchestrator.daw_server import Brain
    project = _kick_bass_project(tmp_path)
    Session(project)._persist()
    brain = Brain(project)
    cap = 1.0 - 10.0 ** (-LIMITS["duck_depth_max_db"] / 20.0)
    made = brain.set_duck("bass", "kick", {}, by_agent=True)        # a NEW duck: at the cap, not mix 1
    duck = next(e for e in Session(project).stems["bass"]["chain"] if e["type"] == "duck")
    assert abs(duck["mix"] - cap) < 1e-6 and abs(made["mix"] - cap) < 1e-6
    res = brain.set_duck("bass", "kick", {"mix": 1.0}, by_agent=True)
    duck = next(e for e in Session(project).stems["bass"]["chain"] if e["type"] == "duck")
    assert abs(duck["mix"] - cap) < 1e-6 and "applied" in res
    user = brain.set_duck("bass", "kick", {"mix": 1.0})              # the user's hand has no cap
    assert next(e for e in Session(project).stems["bass"]["chain"] if e["type"] == "duck")["mix"] == 1.0
    assert "applied" not in user


def test_the_c_plus_plus_cap_is_the_guardrails():
    from mix_engine.guardrails import LIMITS
    src = (REPO / "daw/Source/DawCommands.h").read_text()
    m = re.search(r"kAgentDuckDepthMaxDb = ([\d.]+);", src)
    assert m and float(m.group(1)) == LIMITS["duck_depth_max_db"]


def test_a_tempo_change_is_new_hits(tmp_path):
    """The hits' quarters are the tempo map's: a plain trigger's cache key left it out, and a
    tempo change served the old quarters (the curve landed at the wrong phase)."""
    from orchestrator import daw_bridge as B
    project = _kick_bass_project(tmp_path)
    s = Session(project)
    s.set_duck("bass", "kick")
    a = B.duck_hits_owed(Session(project), wanted=("bass",))["bass"]
    s = Session(project)
    s._state["tempo_map"] = {"tempos": [[0.0, 90.0, 0.0]], "time_sigs": [[0.0, 4, 4]]}
    s._persist()
    b = B.duck_hits_owed(Session(project), wanted=("bass",))["bass"]
    assert a["hits_s"] == b["hits_s"] and a["hits_q"] != b["hits_q"]


def test_a_coedit_that_moves_the_trigger_is_answered_with_new_hits(tmp_path):
    from orchestrator import daw_bridge as B
    project = _kick_bass_project(tmp_path)
    s = Session(project)
    s.set_duck("bass", "kick")
    first = B.duck_hits_owed(Session(project))
    assert "bass" in first and first["bass"]["hits_s"]            # nothing sent yet: owed
    assert B.duck_hits_owed(Session(project)) == {}              # sent, nothing moved: none owed
    B.apply_coedits(project, ["/api/set_stem?stem=kick&gain_db=-24"])   # the kick under the detector's floor
    again = B.duck_hits_owed(Session(project))
    assert "bass" in again and again["bass"]["hits_s"] != first["bass"]["hits_s"]
    assert B.duck_hits_owed(Session(project), wanted=("bass",))["bass"]   # asked for: answered


def test_a_trigger_the_daw_plays_otherwise_gets_no_hits(tmp_path):
    """A looped kick clip renders brain-side as the file from 0:00 - hits for its first pass only.
    Such a trigger is heard live (no hits_s), as an instrument track is."""
    from orchestrator import daw_bridge as B
    project = _kick_bass_project(tmp_path)
    s = Session(project)
    s.set_duck("bass", "kick")
    key = str(Path(project).resolve())
    B.LATEST_EDIT[key] = {"tracks": [{"name": "kick", "clips": [{"start_s": 0.0, "offset_s": 0.0, "loop": True}]},
                                     {"name": "bass"}]}
    try:
        owed = B.duck_hits_owed(Session(project), wanted=("bass",))
        assert owed["bass"].get("live") is True and "hits_s" not in owed["bass"]
        chains = B.resolve_chains(Session(project)._state, session=Session(project))
        duck = next(e for e in chains["bass"]["entries"] if e.get("type") == "duck")
        assert "hits_s" not in duck
    finally:
        B.LATEST_EDIT.pop(key, None)


def test_a_duck_keyed_from_a_part_with_no_hits_says_so(tmp_path):
    from orchestrator.turn_host import duck_trigger_note
    project = _kick_bass_project(tmp_path)
    Session(project).set_duck("bass", "pad")                         # a held tone: no onsets
    note = duck_trigger_note(project, {"actions": [{"type": "duck", "stem_id": "bass", "key_stem_id": "pad"}]})
    assert note and "no hits" in note
    Session(project).set_duck("bass", "kick")
    assert duck_trigger_note(project, {"actions": [{"type": "duck", "stem_id": "bass", "key_stem_id": "kick"}]}) is None


# ------------------------------------------------------------------------------- old projects

def test_a_copied_chain_keeps_an_old_deesser_and_a_deep_dynamic_eq(tmp_path):
    project = _kick_bass_project(tmp_path)
    s = Session(project)
    s.stems["bass"]["chain"] += [
        {"type": "deess", "threshold_db": -30.0, "freq_hz": 6000.0, "range_db": -80.0},
        {"type": "dynamic_eq", "freq_hz": 300.0, "q": 2.0, "gain_db": -20.0, "threshold_db": -30.0},
        {"type": "dynamic_eq", "freq_hz": 18000.0, "q": 2.0, "gain_db": 6.0, "threshold_db": -30.0},
    ]
    s._persist()
    res = Session(project).copy_stem_state("bass", "pad")
    assert res["ok"] and not res.get("devices_dropped"), res
    kinds = [e["type"] for e in Session(project).stems["pad"]["chain"]]
    assert kinds.count("dynamic_eq") == 2 and "deess" in kinds


def _old_project(tmp_path: Path, chain: list, log: list | None = None, version: int = 2) -> Path:
    project = _kick_bass_project(tmp_path)
    s = Session(project)
    s._persist()
    state = json.loads((project / "session.json").read_text())
    state["state_version"] = version
    state["stems"]["bass"]["chain"] = chain
    if log is not None:
        state["coedit_log"] = log
    (project / "session.json").write_text(json.dumps(state))
    return project


def test_an_old_dynamic_eq_or_deesser_that_cut_nothing_still_cuts_nothing(tmp_path):
    project = _old_project(tmp_path, [
        {"type": "dynamic_eq", "freq_hz": 3000.0, "q": 2.0, "gain_db": -6.0, "threshold_db": -30.0, "ratio": 1.0},
        {"type": "dynamic_eq", "freq_hz": 500.0, "q": 1.0, "gain_db": -6.0, "threshold_db": -30.0, "ratio": 3.0},
        {"type": "deess", "threshold_db": -30.0, "freq_hz": 6000.0, "ratio": 1.0, "attack_ms": 1.0,
         "release_ms": 50.0, "range_db": -80.0},
    ], version=5)
    chain = Session(project).stems["bass"]["chain"]
    dyn = [e for e in chain if e["type"] == "dynamic_eq"]
    de = next(e for e in chain if e["type"] == "deess")
    assert dyn[0]["enabled"] is False and dyn[1].get("enabled", True) is True
    assert abs(dyn[0]["q"] - 2.0 * math.sqrt(2.0)) < 1e-3 and "ratio" not in dyn[1]
    assert de["enabled"] is False and de["range_db"] == -24.0 and "ratio" not in de


def test_an_old_clippers_trim_is_kept_as_a_utility(tmp_path):
    project = _old_project(tmp_path, [
        {"type": "clip", "threshold_db": -3.0, "hardness": 0.5, "output_db": -3.0, "oversample": 1},
        {"type": "limiter", "ceiling_dbtp": -1.0},
    ])
    chain = Session(project).stems["bass"]["chain"]
    i = next(k for k, e in enumerate(chain) if e["type"] == "clip")
    assert chain[i + 1]["type"] == "utility" and abs(chain[i + 1]["gain_db"] - (-3.0)) < 1e-6


def test_the_coedit_journal_moves_with_the_laws(tmp_path):
    project = _old_project(tmp_path, [
        {"type": "eq", "band": {"freq_hz": 1000.0, "gain_db": 3.0, "q": 1.0, "shape": "peak"}},
        {"type": "clip", "threshold_db": -3.0, "hardness": 0.5, "output_db": 0.0},
    ], log=[
        {"url": "/api/device_param?stem=bass&device_type=eq&occurrence=0&param=q&value=2.0", "after_seq": 0},
        {"url": "/api/device_param?stem=bass&device_type=clip&occurrence=0&param=threshold_db&value=-6", "after_seq": 0},
        {"url": "/api/set_stem?stem=bass&gain_db=-3", "after_seq": 0},
    ])
    log = Session(project)._state["coedit_log"]
    urls = [e["url"] for e in log]
    assert len(urls) == 2                                            # the clipper's threshold is gone
    q = float(re.search(r"value=([\d.]+)", urls[0]).group(1))
    assert abs(q - 2.0 * math.sqrt(2.0)) < 1e-3                       # in the EQ law's meaning


def test_the_open_says_what_the_migration_did(tmp_path):
    from orchestrator.daw_server import Brain
    project = _old_project(tmp_path, [
        {"type": "dynamic_eq", "freq_hz": 3000.0, "q": 2.0, "gain_db": -6.0, "threshold_db": -30.0, "ratio": 1.0},
        {"type": "return_reverb", "decay_s": 2.0, "tone": 0.5, "pre_delay_ms": 20.0, "mix": 0.3},
    ], version=5)
    brain = Brain(project)
    brain.open_session(daw=True)
    said = " | ".join(brain.take_open_notices())
    assert "ratio of 1" in said and "reverb" in said, said


def test_the_master_dynamic_eq_range_is_the_device_table():
    assert Session.MASTER_PARAM["dynamic_eq"][1:] == Session.DEVICE_PARAMS["dynamic_eq"]["gain_db"][1:]


def test_a_deessers_mode_is_stored_as_a_switch(tmp_path):
    project = _kick_bass_project(tmp_path)
    s = Session(project)
    s.add_stem_device("bass", "deess")
    s.set_master_device_param("deess", 0, "mode", 0.6, target="bass")
    assert next(e for e in Session(project).stems["bass"]["chain"] if e["type"] == "deess")["mode"] == 1.0


# ------------------------------------------------------------------------------ the EQ rails

def test_the_eq_rail_reads_a_narrow_bell_whole():
    import numpy as np
    from mix_engine.guardrails import eq_family_curves_db
    worst = min(float(np.max(eq_family_curves_db({"type": "eq", "band": {"freq_hz": f, "gain_db": 6.0,
                                                                          "q": 40.0, "shape": "peak"}})[0]))
                for f in np.logspace(np.log10(40), np.log10(16000), 40))
    assert worst > 5.95, worst                                       # 512 points read 5.21


def test_the_eq_rail_measures_at_the_projects_rate():
    import numpy as np
    from mix_engine.guardrails import eq_family_curves_db, rail_rate
    band = {"type": "eq", "band": {"freq_hz": 18000.0, "gain_db": 6.0, "q": 8.0, "shape": "high_shelf"}}
    with rail_rate(44100):
        at441 = float(np.max(eq_family_curves_db(band)[0]))
    with rail_rate(48000):
        at48 = float(np.max(eq_family_curves_db(band)[0]))
    assert at441 > at48 + 0.5, (at441, at48)                         # what plays at 44.1 is more


def test_auto_mix_carves_in_the_eq_laws_meaning():
    from mix_engine.eq_law import legacy_band
    from orchestrator import automix
    assert abs(automix.MASKING_CARVE_Q - legacy_band("peak", 1.2, -3.0, 6)[0]) < 0.01
    assert abs(automix.WARM_SHELF_Q - legacy_band("low_shelf", 0.7, 1.5, 6)[0]) < 0.01


# --------------------------------------------------------------------------- the offline reverb

def test_the_offline_reverb_holds_no_whole_song_line_history():
    import tracemalloc
    from mix_engine import reverb_law
    x = np.random.default_rng(1).standard_normal((48000 * 20, 2)) * 0.1
    tracemalloc.start()
    reverb_law.apply(x, 48000, {"decay_s": 2.5, "tone": 0.5, "pre_delay_ms": 20.0})
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 160e6, peak / 1e6                                  # three (frames, 16) arrays were 450 MB here


# ------------------------------------------------------------------------------ the C++ doors

def test_the_dead_view_commands_say_what_they_moved():
    src = (REPO / "daw/Source/App.cpp").read_text()
    for flag in ("gateTapeView", "clipScopeView", "odScopeView", "satTapeView", "delayTapeView", "revTapeView"):
        body = src[src.index(f"case dawcmd::{flag}:"):]
        body = body[:body.index("return")]
        assert "faceViewFlagReply" in src[src.index(f"case dawcmd::{flag}:"):][:2000], flag
    assert "isFaceViewFlag (s.id)) continue" in (REPO / "daw/Source/CommandPalette.h").read_text()


def test_the_red_runs_switch_is_gone():
    """The old path of each native fix sat behind ZAM_OLD_1028 for one red run (--premerge-1028);
    it is never shipped."""
    assert not (REPO / "native/Source/Old1028.h").exists()
    for rel in ("native/Source/MonitorChain.cpp", "native/Source/MonitorChain.h",
                "daw/Source/MonitorChainPlugin.h", "daw/Source/App.cpp"):
        text = (REPO / rel).read_text()
        assert "old1028" not in text and "ZAM_OLD_1028" not in text, rel
