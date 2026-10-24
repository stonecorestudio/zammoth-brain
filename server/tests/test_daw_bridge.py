"""Mix Agent DAW brain-bridge readouts (docs/10-daw.md) — the descriptive
session readout (top-bar BPM·key chip + loudness target) and the reference-match
card the DAW pulls over the NDJSON socket. Neither runs the agent or mutates the
session; the reference audio never enters the mix path."""
from pathlib import Path

import json
import shutil

import pytest

import numpy as np
import soundfile as sf

from orchestrator.daw_bridge import (advise, apply_coedits, auto_mix_pass,
                                      chains_with_readout, detect_clip_key,
                                      extract_clip_groove,
                                      import_stem, match_clip_to_key,
                                      max_applied_seq, moves_since,
                                      reference_match, separate_track,
                                      session_readout,
                                      transpose_to_target_key)
from orchestrator.arrangement import (arrangement_summary,
                                      build_arrangement_plan,
                                      is_arrangement_brief)


ARRANGEMENT_EDIT = {
    "tempo": {"bpm": 80.0, "time_sigs": [{"numerator": 4, "denominator": 4}]},
    "tracks": [
        {"name": "vocals", "kind": "stem",
         "clips": [{"index": 0, "source": "vocals.wav", "start_s": 0.0,
                    "length_s": 8.0}]},
        {"name": "drums", "kind": "stem",
         "clips": [{"index": 0, "source": "drums.wav", "start_s": 0.0,
                    "length_s": 8.0}]},
        {"name": "bass", "kind": "stem",
         "clips": [{"index": 0, "source": "bass.wav", "start_s": 0.0,
                    "length_s": 8.0}]},
        {"name": "keys", "kind": "stem",
         "clips": [{"index": 0, "source": "keys.wav", "start_s": 0.0,
                    "length_s": 8.0}]},
    ],
}


def test_lofi_arrangement_brief_builds_a_labeled_52_bar_plan():
    brief = "create a song arrangement for a lofi song"
    assert is_arrangement_brief(brief)
    assert not is_arrangement_brief("what is song arrangement?")
    assert not is_arrangement_brief("I like the current arrangement")
    assert not is_arrangement_brief("can you explain song arrangement?")

    plan = build_arrangement_plan(brief, ARRANGEMENT_EDIT)
    assert plan["replace"] is True
    assert plan["style"] == "lo-fi"
    assert plan["bpm"] == 80.0 and plan["total_bars"] == 52
    assert [s["name"] for s in plan["sections"]] == [
        "Intro", "Verse 1", "Hook", "Verse 2", "Hook 2", "Bridge",
        "Final Hook", "Outro",
    ]
    assert plan["sections"][0]["tracks"] == ["keys"]       # sparse intro
    assert plan["sections"][2]["tracks"] == ["vocals", "drums", "bass", "keys"]
    assert plan["sections"][1]["source_start_s"] == plan["sections"][0]["length_s"]


def test_brain_routes_arrangement_before_the_mix_target_compiler(tmp_path: Path):
    from orchestrator.daw_server import Brain

    calls = []
    brain = Brain(tmp_path)
    brain.latest_edit = ARRANGEMENT_EDIT
    brain.daw_invoke = lambda command, args: (
        calls.append((command, args))
        or {"ok": True, "sections": len(args["sections"]), "clips": 29,
            "length_s": sum(s["length_s"] for s in args["sections"])}
    )

    result = brain.run("create a song arrangement for a lofi song")
    assert calls and calls[0][0] == "arrangement.build"
    assert result["stop_reason"] == "arranged" and result["passes"] == 0
    assert result["arrangement"]["sections"] == 8
    assert "One Undo reverts the whole arrangement" in result["chat"]


def test_arrangement_ask_mode_previews_without_invoking_the_daw(tmp_path: Path):
    from orchestrator.daw_server import Brain

    brain = Brain(tmp_path)
    brain.latest_edit = ARRANGEMENT_EDIT
    brain.daw_invoke = lambda *_: pytest.fail("Ask mode must not mutate the DAW")
    result = brain.advise("arrange these stems as a lo-fi song")

    assert result["ok"] is True and result["count"] == 8
    assert result["advice"] == arrangement_summary(
        build_arrangement_plan("arrange these stems as a lo-fi song", ARRANGEMENT_EDIT),
        preview=True)


def _write_c_major(path: Path, sr: int = 44100, secs: float = 4.0) -> None:
    """A sustained C-major triad (C4/E4/G4 + a C5 tonic reinforcement) — a clean,
    confidently-keyed sample for the Match Key tests."""
    t = np.linspace(0, secs, int(secs * sr), endpoint=False)
    sig = sum(np.sin(2 * np.pi * f * t) for f in (261.63, 329.63, 392.00)) + 0.5 * np.sin(2 * np.pi * 523.25 * t)
    sig = (sig / np.max(np.abs(sig)) * 0.9).astype("float32")
    sf.write(str(path), sig, sr)


def test_session_readout_reports_loudness_and_song_metrics(demo_project: Path):
    readout = session_readout(demo_project)

    assert readout.get("silent") is False
    assert isinstance(readout["lufs_i"], float)
    # The Phase 30 key/tempo fields are always present (values may be None on
    # short/atonal material — the chip just hides them then).
    for field in ("tempo_bpm", "tempo_confidence", "key", "band_energy"):
        assert field in readout


def test_session_readout_empty_project_is_silent(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    assert session_readout(tmp_path / "empty") == {"silent": True}


def test_reference_match_compares_to_master(demo_project: Path):
    # Use a project stem as a stand-in reference file (a real local wav).
    ref = next(demo_project.glob("*.wav"))
    match = reference_match(demo_project, str(ref))

    assert match["ok"] is True
    assert match["name"] == ref.name
    assert isinstance(match["lufs_i"], float)
    # Per-band deltas (reference minus mix) — one per analysis band present in both.
    assert match["bands"] and all(
        "band" in b and isinstance(b["delta_db"], float) for b in match["bands"])
    assert isinstance(match["advice"], str)


def test_reference_match_missing_file(demo_project: Path):
    match = reference_match(demo_project, str(demo_project / "nope.wav"))
    assert match["ok"] is False
    assert "not found" in match["error"]


def _write_a_major_song(path: Path, sr: int = 48000) -> None:
    """I - IV - V - I in A major, three partials a note: music that spells a key. The demo's own
    stem is ONE tone, which read "A maj" until batch Py3 (reader E5·R6) and has no key now."""
    def tone(hz: float, secs: float) -> np.ndarray:
        t = np.arange(int(secs * sr)) / sr
        return sum(a * np.sin(2 * np.pi * hz * k * t) for k, a in ((1, 1.0), (2, 0.5), (3, 0.3)))

    def chord(root_hz: float) -> np.ndarray:
        return tone(root_hz / 2, 2.0) + sum(tone(root_hz * 2 ** (i / 12), 2.0) for i in (0, 4, 7))

    a4 = 440.0
    song = np.concatenate([chord(a4), chord(a4 * 2 ** (5 / 12)), chord(a4 * 2 ** (7 / 12)), chord(a4)])
    sf.write(str(path), (0.2 * song / np.max(np.abs(song))).astype("float32"), sr)


def test_transpose_refuses_a_song_with_no_key(demo_project: Path):
    # One held tone and a noise bed: nothing to transpose FROM (it used to read "A maj").
    result = transpose_to_target_key(demo_project, "C")
    assert result["ok"] is False and "no confident key" in result["message"]


def test_transpose_applies_a_master_proposal(demo_project: Path):
    # A song in A major → "to C" is a deterministic +3 semitone move.
    _write_a_major_song(demo_project / "stem_a.wav")
    assert session_readout(demo_project)["key"]["tonic"] == "A"
    result = transpose_to_target_key(demo_project, "C")

    assert result["ok"] is True
    assert result["semitones"] == 3
    state = json.loads((demo_project / "session.json").read_text())
    applied = [p for p in state["proposals"] if p["status"] == "applied"]
    assert any(a["type"] == "transpose" and a["stem_id"] == "master"
               for p in applied for a in p["actions"])


def test_detect_clip_key_finds_a_confident_key(tmp_path: Path):
    # NEUTRAL filename (2026-08-10): "cmaj.wav" would now short-circuit through the
    # filename-tag ladder and this test would stop covering the audio detector entirely
    # (the fixture-carries-the-answer trap).
    p = tmp_path / "progression.wav"
    _write_c_major(p)
    det = detect_clip_key(str(p))
    assert det is not None
    assert det["tonic"] == "C" and det["mode"] == "maj"
    assert det["confidence"] > 0.5
    assert det["source"] == "audio"

    assert detect_clip_key(str(tmp_path / "nope.wav")) is None   # bad path → None, never a crash


def test_detect_clip_key_refuses_untonal_material(tmp_path: Path):
    # The TONALITY gate (2026-08-12, James: a kick+snare loop got a key while the tagged
    # melodic loops beside it were refused). _key's confidence is a CORRELATION, so it is
    # not a tonality test: this fixture is all twelve pitch classes at once, amplitudes
    # shaped like the Krumhansl major profile — a near-PERFECT correlation (measured 0.98
    # "C maj") on material whose energy is smeared across the whole chromatic scale. Real
    # drum loops fail the same way with real numbers: the pack in the screenshot scored
    # 0.60–0.79-confident keys at concentrations of 0.31–0.36.
    from mix_engine.analysis import KRUMHANSL_MAJOR, _key

    sr = 44100
    t = np.arange(sr * 2) / sr
    profile = np.asarray(KRUMHANSL_MAJOR, dtype=float)
    profile = profile / profile.max()
    sig = sum(profile[pc] * np.sin(2 * np.pi * (261.63 * 2 ** (pc / 12.0)) * t + pc)
              for pc in range(12))
    sig = (sig / np.max(np.abs(sig)) * 0.5).astype("float32")

    est = _key(sig, sr)                      # the ears alone are happy to name it...
    assert est is not None and est.confidence > 0.9

    p = tmp_path / "spread.wav"              # neutral name — no tag to short-circuit on
    sf.write(str(p), sig, sr)
    assert detect_clip_key(str(p)) is None   # ...the concentration gate refuses it

    # ...and the gate must not swallow genuinely tonal material.
    tonal = tmp_path / "progression.wav"
    _write_c_major(tonal)
    assert detect_clip_key(str(tonal))["tonic"] == "C"


def test_detect_clip_key_reads_the_filename_tag(tmp_path: Path):
    # The tag ladder (2026-08-10, James's G-min bass loop): a pack-labelled key is
    # AUTHORITATIVE — like dawsession::bpmFromFilename for tempo — because chroma genuinely
    # cannot call a solo bass loop (measured 0.35 vs the 0.6 gate on this very pack). White
    # noise proves the tag path: the audio has no key at all, the name decides.
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(44100 * 2) * 0.3).astype("float32")
    cases = {
        "OSNS_Bass_01_keyGmin_85bpm.wav": ("G", "min"),   # the pack in the screenshot
        "lead_F#m_120bpm.wav":            ("F#", "min"),
        "pad-Abmaj.wav":                  ("G#", "maj"),  # flats fold to sharps
        "melody_A_minor_90.wav":          ("A", "min"),   # bare tonic + mode word pair
        # Tonic-only tags (2026-08-12, James: a whole pack refused). Mode is None — the
        # name never said major or minor, and only the tonic feeds the interval.
        "chords keyDb 74bpm.wav":         ("C#", None),   # key-prefixed
        "A_AfrofunkStab01.wav":           ("A", None),    # LEADING token = the pack convention
        "F_TrumpetGlitchx.wav":           ("F", None),
        "C_Destiny_Atmos.wav":            ("C", None),
        "Db_DreamsStretch_Atmos.wav":     ("C#", None),   # leading + flat folds to sharp
        "BpmB7_F#_Velvet_Vibes.wav":      ("F#", None),   # mid-name, but an ACCIDENTAL
        "Loop_Bb.wav":                    ("A#", None),
    }
    for name, (tonic, mode) in cases.items():
        p = tmp_path / name
        sf.write(str(p), noise, 44100)
        det = detect_clip_key(str(p))
        assert det is not None, name
        assert (det["tonic"], det["mode"]) == (tonic, mode), name
        assert det["source"] == "filename" and det["confidence"] == 1.0, name

    # ...and a name with no key evidence must NOT invent one: an ordinary word never
    # parses, a bare letter mid-name is a take/version marker rather than a key, and a
    # LOWER-CASE leading token is a word ("a_capella", "ab_test") — case + position are
    # what let the leading-tonic rule exist at all.
    for name in ("bass_A_85bpm.wav", "ambient_pad.wav", "drums_B_room.wav",
                 "a_capella_take.wav", "ab_test_render.wav", "e_piano_soft.wav"):
        p = tmp_path / name
        sf.write(str(p), noise, 44100)
        assert detect_clip_key(str(p)) is None, name


def test_match_clip_to_key_computes_nearest_interval(tmp_path: Path):
    p = tmp_path / "progression.wav"   # neutral name — the audio path must decide (see above)
    _write_c_major(p)
    # C → G is +7 up but −5 down; the nearest signed move (Phase 32) is −5.
    res = match_clip_to_key(tmp_path, str(p), target_tonic="G", target_mode="maj")
    assert res["ok"] is True
    assert res["semitones"] == -5
    assert res["detected_tonic"] == "C" and res["target_tonic"] == "G"
    # an AUDIO answer on clip-length material is a measured coin flip (2026-08-10 eval:
    # ~50% tonic accuracy, confidence anti-calibrated) — the reason must say it's a guess.
    assert "(guessed)" in res["reason"]

    # matching to the clip's OWN key is a 0-semitone no-op (still ok, honest "already in" reason).
    same = match_clip_to_key(tmp_path, str(p), target_tonic="C", target_mode="maj")
    assert same["ok"] is True and same["semitones"] == 0


def test_match_clip_to_key_uses_the_filename_tag(tmp_path: Path):
    # End-to-end through the tag ladder: the G-min-labelled loop pitches G → C = +5 (nearest
    # signed move), the reason says it trusted the tag, and the audio never had to confess.
    rng = np.random.default_rng(0)
    p = tmp_path / "OSNS_Bass_01_keyGmin_85bpm.wav"
    sf.write(str(p), (rng.standard_normal(44100 * 2) * 0.3).astype("float32"), 44100)
    res = match_clip_to_key(tmp_path, str(p), target_tonic="C", target_mode="maj")
    assert res["ok"] is True
    assert res["semitones"] == 5
    assert res["detected_tonic"] == "G" and res["detected_mode"] == "min"
    assert "(tagged)" in res["reason"]


def test_match_clip_to_key_no_confident_key(tmp_path: Path):
    # untagged white noise has no key → honest ok:False naming BOTH tries (no tag, atonal
    # audio) — the status the Audio Clip View / Loop Deck KEY key shows.
    p = tmp_path / "noise.wav"
    rng = np.random.default_rng(0)
    sf.write(str(p), (rng.standard_normal(44100 * 2) * 0.3).astype("float32"), 44100)
    res = match_clip_to_key(tmp_path, str(p), target_tonic="C", target_mode="maj")
    assert res["ok"] is False
    assert "no confident key" in res["reason"]
    assert "tag" in res["reason"]


def test_transpose_unknown_key_is_rejected(demo_project: Path):
    result = transpose_to_target_key(demo_project, "H")   # not a pitch class
    assert result["ok"] is False
    assert "unknown key" in result["message"]


def test_auto_mix_applies_a_bootstrap_chain(demo_project: Path):
    applied = auto_mix_pass(demo_project, "")

    assert applied >= 1
    state = json.loads((demo_project / "session.json").read_text())
    assert any(p["status"] == "applied" for p in state["proposals"])


def test_session_observations_finds_real_issues(demo_project: Path):
    # Engineer 5.4: engine-grade console observations on the raw stems — deterministic,
    # never raising, each item a sentence + verb + a runnable brief.
    from orchestrator.daw_bridge import session_observations

    obs = session_observations(demo_project)
    assert isinstance(obs, list) and len(obs) <= 3
    for o in obs:
        assert o["text"] and o["verb"] and o["brief"]
    # deterministic: same project, same findings
    assert obs == session_observations(demo_project)
    # the demo's synthetic stems reliably trip at least the masking rule
    assert len(obs) >= 1
    # an empty/na project yields no observations, not an error
    assert session_observations(demo_project / "nope") == []


def test_advise_returns_preview_moves(demo_project: Path):
    # Engineer 5.3: Ask mode's reply carries the proposed actions as move rows (the DAW's
    # jade preview ledger) — same flattened shape, nothing applied to the real session.
    state_before = json.loads((demo_project / "session.json").read_text()) \
        if (demo_project / "session.json").is_file() else {}
    result = advise(demo_project, "master for streaming", use_llm=False)

    assert result["ok"]
    assert isinstance(result["moves"], list)
    assert len(result["moves"]) >= 1
    for m in result["moves"]:
        assert m["stem"] and m["type"] and isinstance(m["params"], dict)
    # advice text and preview rows describe the same proposals
    assert result["count"] >= 1
    # the REAL session was untouched (advise runs on a throwaway copy)
    state_after = json.loads((demo_project / "session.json").read_text()) \
        if (demo_project / "session.json").is_file() else {}
    assert state_before.get("proposals", []) == state_after.get("proposals", [])


def test_run_agent_pass_streams_phase_and_cost(demo_project: Path):
    # 5.3: on_phase fires with pass numbers while the run works (cost only on LLM runs).
    from orchestrator.daw_bridge import run_agent_pass

    phases: list = []
    run_agent_pass(demo_project, "master for streaming", use_llm=False,
                   on_phase=phases.append)
    assert any(p.get("pass_number") for p in phases)


def test_run_agent_pass_ticks_moves_live(demo_project: Path):
    # Engineer round 5.2: on_move fires per APPLIED action while the run works, with the
    # same {stem,type,params,rationale} shape as moves_since — the live feed's data source.
    from orchestrator.daw_bridge import max_applied_seq as seq, moves_since as since
    from orchestrator.daw_bridge import run_agent_pass

    before = seq(demo_project)
    live: list = []
    run_agent_pass(demo_project, "master for streaming", use_llm=False,
                   on_move=live.append)

    final = since(demo_project, before)
    assert len(live) >= 1
    for m in live:
        assert m["stem"] and m["type"]
        assert "stem_id" not in m["params"] and "type" not in m["params"]
    # every live-ticked move appears in the final ledger (same flattening, same order)
    assert [(m["stem"], m["type"]) for m in live] == [(m["stem"], m["type"]) for m in final]


def test_moves_since_flattens_a_runs_actions(demo_project: Path):
    # The Engineer panel's ledger: a run's applied actions in apply order, each carrying
    # its own params + the proposal's rationale (the streamed run summary).
    before = max_applied_seq(demo_project)
    auto_mix_pass(demo_project, "")

    moves = moves_since(demo_project, before)
    assert len(moves) >= 1
    for m in moves:
        assert m["stem"] and m["type"]
        assert isinstance(m["params"], dict)
        assert "stem_id" not in m["params"] and "type" not in m["params"]
    assert any(m["rationale"] for m in moves)
    # the floor is respected: nothing applied before the run leaks in
    assert moves_since(demo_project, max_applied_seq(demo_project)) == []


def test_chains_with_readout_is_one_render(demo_project: Path):
    # run/undo/transpose/auto-mix return {chains, readout} from a SINGLE master render —
    # the readout must match the standalone session_readout (deterministic, same render).
    r = chains_with_readout(demo_project)
    assert "master" in r["chains"]
    assert isinstance(r["readout"]["lufs_i"], float)
    assert r["readout"] == session_readout(demo_project)


def test_chains_with_readout_light_path_skips_the_render(demo_project: Path, monkeypatch):
    # Audio-neutral routing ops (⌘G group_stems) resolve chains from session state + the
    # brain's render cache — no whole-song render (that render was the multi-second ⌘G hang).
    from mix_engine.session import Session

    cache: dict = {}
    full = chains_with_readout(demo_project, cache=cache)   # a full render fills the cache
    assert "readout" in cache and cache["readout"] == full["readout"]

    Session(demo_project).group_stems(sorted(Session(demo_project).stems)[:2])
    import mix_engine.dsp as dsp

    def boom(*a, **k):   # the light path must never reach render_master
        raise AssertionError("light path rendered")
    monkeypatch.setattr(dsp, "render_master", boom)

    light = chains_with_readout(demo_project, cache=cache, light=True)
    assert light["readout"] == full["readout"]
    assert "bus1" in light["buses"]
    assert "master" in light["chains"]


def test_chains_with_readout_light_cold_cache_falls_back(demo_project: Path):
    # With no cached render (brain restart), light=True still answers correctly via a full render.
    r = chains_with_readout(demo_project, cache={}, light=True)
    assert "master" in r["chains"] and isinstance(r["readout"]["lufs_i"], float)


def test_import_stem_registers_a_new_track(demo_project: Path, tmp_path: Path):
    # the DAW file-browser add-track: an external wav lands in the project + session.json,
    # so the agent sees + mixes it (not a Tracktion-only clip the brain can't see).
    external = tmp_path / "extra_synth.wav"
    shutil.copyfile(next(demo_project.glob("*.wav")), external)

    res = import_stem(demo_project, str(external))
    assert res["ok"] is True and res["stem_id"] == "extra_synth"
    assert (demo_project / res["wav"]).is_file()
    stems = json.loads((demo_project / "session.json").read_text())["stems"]
    assert "extra_synth" in stems


def test_import_stem_honours_stem_id_override(demo_project: Path, tmp_path: Path):
    # The DAW drag-drop places the clip instantly + registers in the background with stem_id = the
    # live track name, so the brain stem matches the track (the agent's chains map to the live node).
    external = tmp_path / "guitar.wav"
    shutil.copyfile(next(demo_project.glob("*.wav")), external)

    res = import_stem(demo_project, str(external), stem_id="Audio 1")
    assert res["ok"] is True and res["stem_id"] == "Audio 1"
    assert (demo_project / "Audio 1.wav").is_file()   # written under the override name, not "guitar"
    stems = json.loads((demo_project / "session.json").read_text())["stems"]
    assert "Audio 1" in stems


def test_import_stem_reused_name_does_not_resurrect_a_deleted_stem(
        demo_project: Path, tmp_path: Path):
    # A deleted DAW track only soft-excludes its stem (the wav must stay on disk for the
    # edit's clips + undo). When a NEW track reuses the name (duplicate "Vocals" → delete
    # "Vocals copy" → duplicate again), the re-register must start the record fresh — not
    # hand the new audio the dead track's excluded flag + device chain.
    external = tmp_path / "lead.wav"
    shutil.copyfile(next(demo_project.glob("*.wav")), external)
    assert import_stem(demo_project, str(external), stem_id="Vocals copy")["ok"]

    state = json.loads((demo_project / "session.json").read_text())
    state["stems"]["Vocals copy"]["excluded"] = True                      # the soft delete
    state["stems"]["Vocals copy"]["chain"] = [{"type": "gain", "delta_db": -3.0}]
    (demo_project / "session.json").write_text(json.dumps(state))

    res = import_stem(demo_project, str(external), stem_id="Vocals copy")
    assert res["ok"] and res["reset"] is True   # the DAW re-sends the lane's devices on this
    stems = json.loads((demo_project / "session.json").read_text())["stems"]
    assert stems["Vocals copy"]["excluded"] is False
    assert stems["Vocals copy"]["chain"] == []
    # a live re-register (same logical track) is not a reset
    assert import_stem(demo_project, str(external), stem_id="Vocals copy")["reset"] is False


def test_import_stem_routes_into_an_existing_bus(demo_project: Path, tmp_path: Path):
    # The track-duplicate path: a copy of a bus member inherits its source's group, so it keeps
    # playing through the group's processing instead of silently dropping to the master sum.
    from mix_engine.session import Session

    Session(demo_project).route_stem("stem_a", "drums")   # the source's group

    external = tmp_path / "kick2.wav"
    shutil.copyfile(next(demo_project.glob("*.wav")), external)
    assert import_stem(demo_project, str(external), stem_id="stem_a copy", bus="drums")["ok"]
    stems = json.loads((demo_project / "session.json").read_text())["stems"]
    assert stems["stem_a copy"]["bus"] == "drums"

    # a bus that vanished between duplicate and register is skipped, never resurrected
    assert import_stem(demo_project, str(external), stem_id="stray", bus="ghost")["ok"]
    state = json.loads((demo_project / "session.json").read_text())
    assert state["stems"]["stray"]["bus"] is None
    assert "ghost" not in state["buses"]


def test_import_stem_rejects_non_audio(demo_project: Path, tmp_path: Path):
    bad = tmp_path / "notes.txt"
    bad.write_text("not audio")
    res = import_stem(demo_project, str(bad))
    assert res["ok"] is False and "unsupported" in res["error"]


def test_run_agent_pass_returns_cost_info(demo_project: Path):
    # The DAW shows spend + why a run stopped; run_agent_pass surfaces it (rule agent = $0).
    from orchestrator.daw_bridge import run_agent_pass
    info = run_agent_pass(demo_project, "master for streaming")
    assert set(info) >= {"stop_reason", "passes", "cost_usd"}
    assert info["cost_usd"] == 0.0
    assert isinstance(info["stop_reason"], str)


def test_run_agent_pass_cancel_stops_before_first_pass(demo_project: Path):
    # The DAW's Cancel: should_cancel polled at each pass boundary stops the run cleanly.
    from orchestrator.daw_bridge import run_agent_pass
    info = run_agent_pass(demo_project, "master for streaming", should_cancel=lambda: True)
    assert info["stop_reason"] == "cancelled"
    assert info["passes"] == 0


def test_bus_topology_groups_stems_by_bus():
    # The DAW realizes track->bus->master routing from each stem's `bus` field; the topology
    # the brain hands it maps bus id -> member stem ids (excluded stems dropped).
    from orchestrator.daw_bridge import bus_topology
    state = {
        "stems": {
            "vox": {"bus": "glue"}, "gtr": {"bus": "glue"},
            "drm": {"bus": None}, "old": {"bus": "glue", "excluded": True},
        },
        "buses": {"glue": {}},
    }
    assert bus_topology(state) == {"glue": ["vox", "gtr"]}   # drm unbussed; old excluded


def test_apply_coedits_set_stem_persists_user_move(demo_project: Path):
    # The DAW channel-strip fader/pan is the user's direct channel — a set_stem co-edit
    # mirrors the move into session.json so the agent's next pass starts from the user's mix
    # (gain staging is unified: the fader is the node's single live gain stage).
    from mix_engine.session import Session
    Session(demo_project).refresh_stems()
    stem = next(iter(json.loads((demo_project / "session.json").read_text())["stems"]))

    res = apply_coedits(demo_project, [f"/api/set_stem?stem={stem}&gain_db=-4.0&pan=0.5"])
    assert res["ok"] and res["applied"] == 1

    s = json.loads((demo_project / "session.json").read_text())["stems"][stem]
    assert abs(s["gain_db"] - (-4.0)) < 1e-6
    assert abs(s["pan"] - 0.5) < 1e-6


def test_apply_coedits_resolves_a_bus_by_its_native_track_name(demo_project: Path):
    """A group-bus lane reaches the brain as its NATIVE track name ("bus:<id>" — what the DAW
    calls the track and what the chains reply is keyed by), while the brain keys buses bare.
    Before 2026-09-05 nothing stripped the prefix, so every fader move on a bus strip came back
    "no such stem: bus:bus1" and every device edit on it "unknown co-edit target 'bus:bus1'" —
    the bus half of shared control was UI-only and session.json (the agent's + the export
    oracle's truth) never saw a bus move."""
    from mix_engine.session import Session
    s = Session(demo_project)
    s.refresh_stems()
    stems = sorted(s.stems)
    bus_id = s.group_stems(stems[:1])["bus_id"]

    res = apply_coedits(demo_project, [
        f"/api/set_stem?stem=bus:{bus_id}&gain_db=-4.0",            # the strip fader
        f"/api/add_device?stem=bus:{bus_id}&device_type=eq",        # the dock, targeting the bus
        f"/api/device_param?stem=bus:{bus_id}&device_type=eq&occurrence=0"
        "&param=freq_hz&value=250",
    ])
    assert res["ok"] and res["applied"] == 3, res["errors"]

    bus = json.loads((demo_project / "session.json").read_text())["buses"][bus_id]
    assert abs(bus["gain_db"] - (-4.0)) < 1e-6
    eq = next(e for e in bus["chain"] if e["type"] == "eq")
    assert abs(eq["band"]["freq_hz"] - 250.0) < 1e-6


def test_apply_coedits_persists_the_bus_strips_pan_width_and_mute(demo_project: Path):
    """The group strip's pan / W / [M] are the bus's own channel-strip scalars now — they land
    in session.json like a stem's, so the agent's next pass and the export oracle see the move
    (they were refused as "a bus accepts gain_db only" and lost until 2026-09-05)."""
    from mix_engine.session import Session
    s = Session(demo_project)
    s.refresh_stems()
    bus_id = s.group_stems(sorted(s.stems)[:1])["bus_id"]

    res = apply_coedits(demo_project, [
        f"/api/set_stem?stem=bus:{bus_id}&pan=-0.4",
        f"/api/set_stem?stem=bus:{bus_id}&width=1.5",
        f"/api/set_stem?stem=bus:{bus_id}&muted=true",
    ])
    assert res["ok"] and res["applied"] == 3, res["errors"]

    bus = json.loads((demo_project / "session.json").read_text())["buses"][bus_id]
    assert abs(bus["pan"] - (-0.4)) < 1e-6
    assert abs(bus["width"] - 1.5) < 1e-6
    assert bus["muted"] is True


def test_apply_coedits_still_refuses_an_unknown_bus(demo_project: Path):
    """Only a bus that EXISTS is resolved — a stale or misspelled lane still refuses out loud
    rather than silently writing onto some other holder."""
    from mix_engine.session import Session
    Session(demo_project).refresh_stems()

    res = apply_coedits(demo_project, ["/api/set_stem?stem=bus:nope&gain_db=-4.0"])
    assert res["ok"] and res["applied"] == 0 and res["failed"] == 1
    assert "nope" in res["errors"][0]


def test_advise_reports_without_mutating(demo_project: Path):
    # Ask mode: the agent runs on a THROWAWAY copy and reports what it would do; the real
    # session.json is byte-for-byte untouched (nothing is applied).
    from mix_engine.session import Session
    Session(demo_project).refresh_stems()           # materialize + persist session.json
    before = (demo_project / "session.json").read_text()

    res = advise(demo_project, "make the master louder and brighter")
    assert res["ok"] is True
    assert isinstance(res["advice"], str) and res["advice"]
    assert res["count"] >= 0
    assert (demo_project / "session.json").read_text() == before   # the real mix is untouched


def test_resolve_chains_injects_agent_send_returns():
    """B2 (sends/returns): the DAW resolver must inject the Phase 38 send_return wet so the
    agent's reverb/delay returns are AUDIBLE in the DAW graph (the gateway already did; the DAW
    path didn't). Oracle-faithful — the same synthetic entry the plugin/offline path uses."""
    from orchestrator.daw_bridge import _send_return_entries

    state = {"returns": {"rv": {"gain_db": 0.0,
                                "chain": [{"type": "return_reverb", "wet": 1.0,
                                           "room_size": 0.6, "damping": 0.4,
                                           "pre_delay_ms": 20.0}]}}}
    out = _send_return_entries(state, {"sends": {"rv": -6.0}})
    assert len(out) == 1 and out[0]["type"] == "send_return"
    assert out[0]["send_db"] == -6.0 and out[0]["return"]          # carries the resolved return chain
    assert _send_return_entries(state, {"sends": {}}) == []        # no sends -> nothing
    assert _send_return_entries({"returns": {}}, {"sends": {"x": 0.0}}) == []   # unknown return -> skipped


def test_brain_open_edit_repoints_to_per_edit_session_dir(demo_project: Path, tmp_path: Path):
    """Multi-edit (Inc 2): edit.open tells the brain to switch to the edit's per-edit session dir,
    so the agent reads/mixes THAT edit's tracks. A fresh edit dir starts empty; the brain reads the
    new dir's session.json after the switch."""
    from orchestrator.daw_server import Brain

    brain = Brain(demo_project)
    brain.latest_edit = {"tracks": []}        # a stale snapshot from the previous edit

    # switch to a fresh per-edit dir → re-pointed, created, snapshot reset, no stems yet
    fresh = tmp_path / ".edits" / "verse1"
    res = brain.open_edit(str(fresh))
    assert res["ok"] and brain.project_dir == fresh and fresh.is_dir()
    assert brain.latest_edit is None and res["stems"] == []

    # an edit dir whose session.json has stems → the brain reads them after switching to it
    populated = tmp_path / ".edits" / "chorus"
    populated.mkdir(parents=True)
    (populated / "session.json").write_text(json.dumps({"stems": {"drums": {}, "bass": {}}}))
    res2 = brain.open_edit(str(populated))
    assert brain.project_dir == populated and sorted(res2["stems"]) == ["bass", "drums"]


def test_resolve_chains_native_returns_gives_dry_stems_plus_return_tracks():
    """Real AuxSend/AuxReturn (the DAW path): native_returns=True returns DRY per-track chains
    (no per-track send_return wet) + a return:<id> MonitorChain (return_track_mode); the default
    keeps the per-track wet (the gateway / offline-monitor path). The return is now a FREE device
    chain (Ableton blank returns) — reverb/delay/conv are TOP-LEVEL wet devices, not a send_return
    wrapper — so the native MonitorChain renders them serially at their chain position."""
    from orchestrator.daw_bridge import resolve_chains

    state = {"stems": {"vox": {"chain": [], "gain_db": 0.0, "pan": 0.0, "width": 1.0,
                               "rides": [], "sends": {"rv": -6.0}}},
             "returns": {"rv": {"gain_db": 0.0, "pre_fader": False, "sends": {},
                                "chain": [{"type": "eq", "band": {"shape": "high_shelf",
                                           "freq_hz": 8000.0, "gain_db": -3.0, "q": 0.7}},
                                          {"type": "return_reverb", "decay_s": 1.5,
                                           "pre_delay_ms": 20.0, "tone": 0.5}]}},
             "buses": {}, "master_chain": [], "master_gain_db": 0.0}

    def has_send_return(chain):
        return any(e.get("type") == "send_return" for e in chain.get("entries", []))

    legacy = resolve_chains(state)                       # default: per-track wet
    assert has_send_return(legacy["vox"])                # the wet is baked into the stem
    assert "return:rv" not in legacy

    native = resolve_chains(state, native_returns=True)  # real return tracks
    assert not has_send_return(native["vox"])            # the stem is DRY (the AuxSend taps it)
    assert "return:rv" in native and native["return:rv"].get("return_track_mode") is True
    # the return is a free device chain: its EQ + reverb are top-level entries in list order
    types = [e["type"] for e in native["return:rv"]["entries"]]
    assert types == ["gain", "eq", "return_reverb"]      # serial, list-order (not a send_return wrapper)
    assert not has_send_return(native["return:rv"])


def test_add_return_lets_the_user_create_a_return(demo_project: Path):
    """Sends/returns shared control: the USER creates a reverb/delay return themselves (not only the
    agent) — a direct co-edit like set_send. Defaults come from the return action models."""
    from mix_engine.session import Session
    Session(demo_project).refresh_stems()

    res = Session(demo_project).add_return("reverb")
    assert res["ok"] and res["return_id"] == "rv" and res["kind"] == "return_reverb"
    returns = json.loads((demo_project / "session.json").read_text())["returns"]
    assert "rv" in returns and any(e["type"] == "return_reverb" for e in returns["rv"]["chain"])

    # a second add gets a unique id; delay returns are "dly"
    assert Session(demo_project).add_return("delay")["return_id"] == "dly"
    assert Session(demo_project).add_return("reverb")["return_id"] == "rv2"


def test_route_stem_lets_the_user_create_and_route_to_a_bus(demo_project: Path):
    """Routing shared control: the USER routes a stem to a group bus (created if absent) or the master
    — a direct co-edit, the offline 'agent owns routing' rule reversed in the DAW."""
    from mix_engine.session import Session
    Session(demo_project).refresh_stems()
    stems = list(json.loads((demo_project / "session.json").read_text())["stems"])
    a, b = stems[0], stems[1]

    res = Session(demo_project).route_stem(a, "new")        # mint a bus + route a to it
    bus = res["bus"]
    assert bus == "bus1"
    Session(demo_project).route_stem(b, bus)                # route b to the same bus
    state = json.loads((demo_project / "session.json").read_text())
    assert bus in state["buses"]
    assert state["stems"][a]["bus"] == bus and state["stems"][b]["bus"] == bus

    Session(demo_project).route_stem(a, "master")           # back to master
    Session(demo_project).route_stem(b, "master")           # now the bus is empty -> dropped
    state = json.loads((demo_project / "session.json").read_text())
    assert state["stems"][a]["bus"] is None and bus not in state["buses"]


def test_group_stems_creates_one_bus_and_routes_all(demo_project: Path):
    """Ableton ⌘G: grouping several stems mints ONE new bus and routes every given stem into it in a
    single action (not a separate bus per stem). The user then drops plugins on the bus to process
    the sum. group_stems is the one-shot form of route_stem(..., 'new')."""
    from mix_engine.session import Session
    Session(demo_project).refresh_stems()
    stems = list(json.loads((demo_project / "session.json").read_text())["stems"])
    a, b = stems[0], stems[1]

    res = Session(demo_project).group_stems([a, b])     # one gesture → one bus, both members
    bus = res["bus_id"]
    assert bus == "bus1"
    assert res["stems"] == [a, b]
    state = json.loads((demo_project / "session.json").read_text())
    assert list(state["buses"]) == [bus]                # exactly ONE bus was made (not one per stem)
    assert state["stems"][a]["bus"] == bus and state["stems"][b]["bus"] == bus

    # regrouping a member into a new group empties the old bus → it is dropped (no orphan bus track)
    res2 = Session(demo_project).group_stems([a])
    assert res2["bus_id"] == "bus2"
    state = json.loads((demo_project / "session.json").read_text())
    assert state["stems"][a]["bus"] == "bus2" and state["stems"][b]["bus"] == "bus1"

    with pytest.raises(ValueError):                     # nothing groupable → a clear error, not a no-op bus
        Session(demo_project).group_stems([])
    with pytest.raises(ValueError):                     # reserved names alone are not groupable
        Session(demo_project).group_stems(["master", "bus:bus1", "return:fx"])


def test_group_empty_tracks_holds_pending_members(demo_project: Path):
    """Ableton parity (2026-07-19 bug: 'agent says grouped but no visual group'): grouping EMPTY
    DAW tracks — tracks with no audio landed, so no brain stem — must WORK, not error while the
    UI claims 'grouped'. Unregistered names join the bus as PENDING members: they hold the bus
    alive, bus_topology lists them (so realizeBuses wires the live tracks + the group draws),
    and refresh/adoption turns each into a real member when its audio registers."""
    from mix_engine.session import Session
    from orchestrator.daw_bridge import bus_topology
    Session(demo_project).refresh_stems()
    stems = list(json.loads((demo_project / "session.json").read_text())["stems"])
    a = stems[0]

    # 1) group two EMPTY tracks (unknown to the brain): a bus is minted, both names pending
    res = Session(demo_project).group_stems(["Track 1", "Audio 1"])
    bus = res["bus_id"]
    assert bus == "bus1"
    assert res["stems"] == [] and res["pending"] == ["Track 1", "Audio 1"]
    state = json.loads((demo_project / "session.json").read_text())
    assert state["buses"][bus]["pending"] == ["Track 1", "Audio 1"]

    # 2) the pending members hold the bus alive through the empty-bus tidy,
    #    and bus_topology lists them so the DAW routes + draws the group
    s = Session(demo_project)
    s.route_stem(a, bus)          # a real stem joins…
    s.route_stem(a, "master")     # …and leaves — the bus must SURVIVE (pending members)
    state = json.loads((demo_project / "session.json").read_text())
    assert bus in state["buses"]
    assert bus_topology(state)[bus] == ["Track 1", "Audio 1"]

    # 3) audio lands on a grouped-while-empty track → its stem is ADOPTED as a real member
    shutil.copy(demo_project / f"{a}.wav", demo_project / "Audio 1.wav")
    Session(demo_project).refresh_stems()
    state = json.loads((demo_project / "session.json").read_text())
    assert state["stems"]["Audio 1"]["bus"] == bus
    assert state["buses"][bus]["pending"] == ["Track 1"]
    assert bus_topology(state)[bus] == ["Audio 1", "Track 1"]

    # 4) a MIXED group (one real stem + one empty track) takes both kinds in one gesture
    res2 = Session(demo_project).group_stems([a, "Track 1"])
    assert res2["bus_id"] == "bus2"
    assert res2["stems"] == [a] and res2["pending"] == ["Track 1"]
    state = json.loads((demo_project / "session.json").read_text())
    assert state["stems"][a]["bus"] == "bus2"
    assert state["buses"]["bus2"]["pending"] == ["Track 1"]
    # …and Track 1 left bus1's pending (regrouped); bus1 keeps Audio 1 (real member)
    assert state["buses"]["bus1"]["pending"] == []
    assert state["stems"]["Audio 1"]["bus"] == "bus1"

    # 5) route_stem moves an empty track by name too (the strip's Route-to menu)
    Session(demo_project).route_stem("Track 1", "master")   # pending cleared…
    state = json.loads((demo_project / "session.json").read_text())
    assert state["buses"]["bus2"]["pending"] == []
    assert state["stems"][a]["bus"] == "bus2"             # …bus survives via its real member
    with pytest.raises(ValueError):                         # reserved names still refuse
        Session(demo_project).route_stem("bus:bus1", "bus2")

    # 6) remove_bus dissolves a pending-only group cleanly
    Session(demo_project).route_stem("Track 1", "new")      # bus3, pending-only
    state = json.loads((demo_project / "session.json").read_text())
    assert state["buses"]["bus3"]["pending"] == ["Track 1"]
    Session(demo_project).remove_bus("bus3")
    assert "bus3" not in json.loads((demo_project / "session.json").read_text())["buses"]


def test_group_in_an_empty_project_still_surfaces_the_bus(tmp_path: Path):
    """The zero-stem early-out in chains_with_readout must still carry the routing topology:
    a brand-new project (no audio imported at all — James's repro) groups two empty tracks,
    and the DAW must receive {buses} listing the pending members or realizeBuses never runs
    and nothing draws (the second half of the 2026-07-19 invisible-group bug)."""
    from mix_engine.session import Session
    project = tmp_path / "empty_project"
    project.mkdir()

    res = Session(project).group_stems(["Track 1", "Audio 1"])
    assert res == {"bus_id": "bus1", "stems": [], "pending": ["Track 1", "Audio 1"]}

    out = chains_with_readout(project, cache={}, light=True)   # cold cache → the zero-stem path
    assert out["readout"] == {"silent": True}
    assert out["buses"] == {"bus1": ["Track 1", "Audio 1"]}
    assert "bus:bus1" in out["chains"] and "master" in out["chains"]
    assert out["bus_outputs"] == {} and "send_returns" in out


def test_remove_bus_dissolves_a_group_and_rehomes_members(demo_project: Path):
    """Routing shared control: the USER dissolves a group bus directly — its members re-home to the
    master + the bus is dropped (the DAW's realizeBuses then removes the orphaned bus track). The
    inverse of routing stems in; lets the user delete a group the same gesture deletes a track."""
    from mix_engine.session import Session
    Session(demo_project).refresh_stems()
    stems = list(json.loads((demo_project / "session.json").read_text())["stems"])
    a, b = stems[0], stems[1]

    Session(demo_project).route_stem(a, "new")              # bus1 with two members
    Session(demo_project).route_stem(b, "bus1")
    assert json.loads((demo_project / "session.json").read_text())["buses"].get("bus1") is not None

    res = Session(demo_project).remove_bus("bus1")
    assert res == {"ok": True, "bus_id": "bus1"}
    state = json.loads((demo_project / "session.json").read_text())
    assert "bus1" not in state["buses"]                   # the bus is gone
    assert state["stems"][a]["bus"] is None and state["stems"][b]["bus"] is None   # members re-homed

    try:                                                    # unknown bus -> error, not a silent no-op
        Session(demo_project).remove_bus("ghost")
        assert False, "remove_bus on an unknown bus should raise"
    except ValueError:
        pass


def test_route_bus_routes_a_groups_own_output(demo_project: Path):
    """Routing shared control (2026-07-03, nested bus routing): a group's OWN output — SEPARATE
    from route_stem (which routes a STEM into a bus). route_bus nests one bus's processed sum into
    another (or back to the master)."""
    from mix_engine.session import Session
    Session(demo_project).refresh_stems()
    stems = list(json.loads((demo_project / "session.json").read_text())["stems"])
    a, b = stems[0], stems[1]

    s = Session(demo_project)
    s.route_stem(a, "new")     # bus "bus1" <- a
    s.group_stems([b])         # bus "bus2" <- b (group_stems mints bus2, since bus1 exists)
    state = json.loads((demo_project / "session.json").read_text())
    assert set(state["buses"]) == {"bus1", "bus2"}

    res = Session(demo_project).route_bus("bus1", "bus2")
    assert res == {"bus": "bus1", "output": "bus2"}
    state = json.loads((demo_project / "session.json").read_text())
    assert state["buses"]["bus1"]["output"] == "bus2"

    # back to master
    res2 = Session(demo_project).route_bus("bus1", "master")
    assert res2 == {"bus": "bus1", "output": None}
    assert json.loads((demo_project / "session.json").read_text())["buses"]["bus1"]["output"] is None

    with pytest.raises(ValueError):          # unknown target bus
        Session(demo_project).route_bus("bus1", "does_not_exist")
    with pytest.raises(ValueError):          # a bus can't route to itself
        Session(demo_project).route_bus("bus1", "bus1")
    with pytest.raises(ValueError):          # unknown source bus
        Session(demo_project).route_bus("ghost", "bus2")


def test_route_bus_refuses_a_cycle(demo_project: Path):
    """A bus routed A -> B may not also be routed B -> A (or any longer loop) — route_bus walks
    the target's existing output chain and refuses if it would ever reach back to the source."""
    from mix_engine.session import Session
    Session(demo_project).refresh_stems()
    stems = list(json.loads((demo_project / "session.json").read_text())["stems"])
    a, b = stems[0], stems[1]

    s = Session(demo_project)
    s.route_stem(a, "new")     # bus1
    s.group_stems([b])         # bus2
    s.route_bus("bus1", "bus2")   # bus1 -> bus2

    with pytest.raises(ValueError):
        Session(demo_project).route_bus("bus2", "bus1")   # would loop bus1 <-> bus2


def test_remove_bus_rehomes_other_buses_routed_into_it(demo_project: Path):
    """remove_bus is the inverse of BOTH route_stem (member stems re-home to master) AND route_bus
    (any OTHER bus whose output pointed at the removed bus re-homes to master too) — otherwise a
    dangling output reference would survive the removed bus."""
    from mix_engine.session import Session
    Session(demo_project).refresh_stems()
    stems = list(json.loads((demo_project / "session.json").read_text())["stems"])
    a, b = stems[0], stems[1]

    s = Session(demo_project)
    s.route_stem(a, "new")     # bus1
    s.group_stems([b])         # bus2
    s.route_bus("bus1", "bus2")   # bus1 -> bus2 (bus2 has no direct stem members)

    s.remove_bus("bus2")
    state = json.loads((demo_project / "session.json").read_text())
    assert "bus2" not in state["buses"]
    assert state["buses"]["bus1"]["output"] is None   # re-homed to the master, not left dangling


def test_route_bus_nested_render_sums_through_the_parent(demo_project: Path):
    """dsp.render_master must actually route a child bus's WET sum into its parent bus's input
    BEFORE the parent applies its own chain — so a device on the PARENT audibly affects everything
    summed into it, including a child bus nested underneath. Compares the same topology with vs.
    without the route_bus nesting: nesting under a -6 dB parent bus must quiet the master."""
    from mix_engine.dsp import render_master
    from mix_engine.session import Session

    s = Session(demo_project)
    s.refresh_stems()
    stems = list(s.stems)
    a, b = stems[0], stems[1]
    s._state["stems"][a]["bus"] = "bus1"
    s._state["stems"][b]["muted"] = True   # isolate bus1's content — the only thing in the mix
    s._state["buses"]["bus1"] = {"gain_db": 0.0, "chain": [], "sends": {}}
    s._state["buses"]["bus2"] = {"gain_db": -6.0, "chain": [], "sends": {}}   # no direct members

    baseline, sr = render_master(s)   # bus1 -> master directly; bus2 has no members/children -> skipped

    s._state["buses"]["bus1"]["output"] = "bus2"   # nest bus1 UNDER bus2's -6 dB
    nested, _ = render_master(s)

    import math

    base_peak = float(abs(baseline).max())
    nested_peak = float(abs(nested).max())
    assert base_peak > 1e-6 and nested_peak > 1e-6
    delta_db = 20.0 * math.log10(nested_peak / base_peak)
    assert -7.0 < delta_db < -5.0   # ~ -6 dB, the parent bus's gain


def test_bus_outputs_surfaces_nested_routing_to_the_daw(demo_project: Path):
    """daw_bridge.bus_outputs is the DAW-facing readout of route_bus (like bus_topology is for
    route_stem) — only buses whose target still exists are included; the DAW's realizeBuses /
    the head 'Route to' menu read this to wire + show the current nesting."""
    from orchestrator.daw_bridge import bus_outputs
    state = {"buses": {"a": {"output": "b"}, "b": {"output": None}, "c": {"output": "ghost"}}}
    assert bus_outputs(state) == {"a": "b"}   # b -> None omitted (master); c -> ghost omitted (orphan)


def test_returns_topology_keys_a_bus_source_by_its_native_track_name(demo_project: Path):
    """A bus's native Tracktion track is named 'bus:<id>' (unlike a stem, whose track name IS its
    bare id) — the sends map must key a bus source the same way a return is keyed 'return:<id>',
    or realizeReturns' by-name track lookup (and the channel-strip send knob, whose node id is the
    full track name) silently finds no track and never creates the AuxSend."""
    from orchestrator.daw_bridge import returns_topology
    state = {"stems": {"vox": {"sends": {"rv": -6.0}}},
             "buses": {"bus1": {"sends": {"rv": -3.0}}},
             "returns": {"rv": {}}}
    topo = returns_topology(state)
    assert topo["sends"]["vox"] == {"rv": -6.0}          # a stem source keeps its bare id
    assert topo["sends"]["bus:bus1"] == {"rv": -3.0}   # a bus source is prefixed like a return


def test_set_send_accepts_a_bus_prefixed_source(demo_project: Path):
    """Session.set_send must accept the SAME 'bus:<id>' source id the DAW now sends (mirroring the
    'return:<id>' stripping it already did) — the channel-strip send knob passes its node id, the
    full native track name, not the bare bus id."""
    from mix_engine.session import Session
    s = Session(demo_project)
    s.refresh_stems()
    stems = list(s.stems)
    s.route_stem(stems[0], "new")   # bus1
    s.add_return("reverb")          # "rv"

    res = s.set_send("bus:bus1", "rv", -4.0)
    assert res["source"] == "bus1" and abs(res["gain_db"] - (-4.0)) < 1e-6
    assert s.buses["bus1"]["sends"]["rv"] == -4.0


def test_apply_coedits_sets_an_fx_return_param(demo_project: Path):
    """The user adjusts an FX return's reverb/delay in the device dock → a device_param co-edit
    targeting the return:<id> node updates that return's chain in session.json (shared control)."""
    from mix_engine.session import Session
    s = Session(demo_project)
    s.refresh_stems()
    rid = s.add_return("reverb")["return_id"]      # "rv"

    res = apply_coedits(demo_project, [
        f"/api/device_param?device_type=return_reverb&occurrence=0&param=decay_s&value=4.5&stem=return:{rid}"])
    assert res["ok"] and res["applied"] == 1

    chain = json.loads((demo_project / "session.json").read_text())["returns"][rid]["chain"]
    rv = next(e for e in chain if e["type"] == "return_reverb")
    assert abs(rv["decay_s"] - 4.5) < 1e-6


def test_brain_set_llm_sets_env_and_validates(demo_project: Path, monkeypatch):
    """The composer's model+effort picker: Brain.set_llm validates against the chat registry and
    stores the choice as env in the brain process — ClaudeAgent reads MIX_AGENT_MODEL /
    MIX_AGENT_EFFORT at construction, so the change applies from the next run."""
    import os

    import pytest

    from orchestrator.daw_server import Brain

    monkeypatch.delenv("MIX_AGENT_MODEL", raising=False)
    monkeypatch.delenv("MIX_AGENT_EFFORT", raising=False)
    brain = Brain(demo_project)

    res = brain.set_llm("claude-sonnet-5-5", "max")
    assert res["ok"]
    assert os.environ["MIX_AGENT_MODEL"] == "claude-sonnet-5-5"
    assert os.environ["MIX_AGENT_EFFORT"] == "max"

    with pytest.raises(ValueError):
        brain.set_llm("claude-nonsense", "high")
    with pytest.raises(ValueError):
        brain.set_llm("claude-opus-5-5", "ultra")
    with pytest.raises(ValueError):
        brain.set_llm("claude-opus-5", "high")      # on the picker until 2026-10-08

    # ClaudeAgent's env default + effort kwargs honour the choice (an id off the table: no
    # thinking config)
    from orchestrator.agents import ClaudeAgent

    monkeypatch.setenv("MIX_AGENT_LLM_BACKEND", "subscription")  # no anthropic SDK needed
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "test-token")
    monkeypatch.setenv("MIX_AGENT_MODEL", "claude-sonnet-5-5")
    monkeypatch.setenv("MIX_AGENT_EFFORT", "max")
    agent = ClaudeAgent()
    assert agent._model == "claude-sonnet-5-5"
    assert agent._think_kwargs() == {"thinking": {"type": "adaptive"},
                                     "output_config": {"effort": "max"}}
    monkeypatch.setenv("MIX_AGENT_MODEL", "claude-haiku-5-5")
    assert ClaudeAgent()._think_kwargs() == {"thinking": {"type": "adaptive"},
                                             "output_config": {"effort": "max"}}
    monkeypatch.setenv("MIX_AGENT_MODEL", "claude-haiku-4-5")
    assert ClaudeAgent()._think_kwargs() == {}


# ---- multi-project workspace Inc 1 (docs/10-daw.md): peek + cross-project import ----

def _other_project(demo_project: Path, tmp_path: Path) -> Path:
    """A second throwaway project holding the SAME wav names — the built-in id collision."""
    other = tmp_path / "other_song"
    other.mkdir()
    for wav in demo_project.glob("*.wav"):
        shutil.copy(wav, other / wav.name)
    return other


def test_import_track_from_materializes_audio_and_state(demo_project: Path, tmp_path: Path):
    # The materialization core: the wav copies byte-identical into the destination folder
    # (collision-safe id), the chain + channel scalars carry (chain re-canonicalized), bus
    # routing STRIPS to master, and a send with no same-kind destination return drops.
    from mix_engine.session import Session
    from orchestrator.daw_bridge import import_track_from

    src = Session(demo_project)
    src.route_stem("stem_a", "drums")                      # group membership → must strip
    rid = src.add_return("reverb")["return_id"]            # "rv"
    src.set_send("stem_a", rid, -12.0)
    state = json.loads((demo_project / "session.json").read_text())
    state["stems"]["stem_a"].update({
        "gain_db": -2.5, "pan": -0.3, "muted": True,
        # saturation listed BEFORE eq — adoption must re-canonicalize to render order.
        # The malformed eq (no band) + the duck keyed to a track the destination lacks
        # must DROP with a report — a foreign session.json is untrusted input, and one
        # bad entry would otherwise poison every later destination render.
        "chain": [{"type": "saturation", "drive_db": 6.0},
                  {"type": "eq",
                   "band": {"freq_hz": 2000, "gain_db": 3.0, "q": 1.0, "shape": "peak"}},
                  {"type": "eq", "freq_hz": 500, "gain_db": 2.0},   # malformed: no band
                  {"type": "duck", "key_stem_id": "stem_b", "threshold_db": -30.0,
                   "depth_db": 6.0}]})
    (demo_project / "session.json").write_text(json.dumps(state))

    dest = _other_project(demo_project, tmp_path)
    (dest / "stem_b.wav").unlink()                         # dest has NO stem_b → the duck's key is dangling
    res = import_track_from(dest, str(demo_project), "stem_a")
    assert res["ok"] is True
    assert res["stem_id"] == "stem_a 2"                    # dest's own stem_a keeps its name
    assert (dest / "stem_a 2.wav").read_bytes() == (demo_project / "stem_a.wav").read_bytes()

    st = json.loads((dest / "session.json").read_text())["stems"]["stem_a 2"]
    assert [e["type"] for e in st["chain"]] == ["eq", "saturation"]   # canonical order, bad entries gone
    assert len(res["devices_dropped"]) == 2                # the malformed eq + the dangling duck
    assert any("duck" in d for d in res["devices_dropped"])
    assert st["gain_db"] == -2.5 and st["pan"] == -0.3 and st["muted"] is True
    assert st["bus"] is None                               # policy: routing strips to master
    assert st["sends"] == {} and res["sends_dropped"] == [rid]


def test_import_track_from_same_kind_send_carries(demo_project: Path, tmp_path: Path):
    # A send carries only onto a SAME-KIND destination return: the source's second reverb
    # ("rv2") has no exact id in the destination, but the destination's "rv" is the same
    # kind (id prefix) — the send re-targets it instead of dropping.
    from mix_engine.session import Session
    from orchestrator.daw_bridge import import_track_from

    src = Session(demo_project)
    assert src.add_return("reverb")["return_id"] == "rv"
    rid2 = src.add_return("reverb")["return_id"]           # "rv2"
    src.set_send("stem_a", rid2, -9.0)

    dest = _other_project(demo_project, tmp_path)
    assert Session(dest).add_return("reverb")["return_id"] == "rv"

    res = import_track_from(dest, str(demo_project), "stem_a")
    assert res["ok"] is True and res["sends_kept"] == ["rv"] and res["sends_dropped"] == []
    st = json.loads((dest / "session.json").read_text())["stems"][res["stem_id"]]
    assert st["sends"] == {"rv": -9.0}


def test_import_track_from_honest_errors(demo_project: Path, tmp_path: Path):
    from orchestrator.daw_bridge import import_track_from

    dest = _other_project(demo_project, tmp_path)
    assert "no track" in import_track_from(dest, str(demo_project), "ghost")["error"]
    assert "already open" in import_track_from(
        demo_project, str(demo_project), "stem_a")["error"]
    assert "not a folder" in import_track_from(
        dest, str(tmp_path / "nope"), "stem_a")["error"]


def test_peek_project_lists_tracks_readonly(demo_project: Path, tmp_path: Path):
    # The browser's PROJECTS section: another project's tracks + device types, excluded
    # stems hidden — and a bare stems folder (never brain-opened) lists WITHOUT a
    # session.json ever being written (read-only means read-only).
    from mix_engine.session import Session
    from orchestrator.daw_bridge import peek_project

    src = Session(demo_project)
    src.route_stem("stem_a", "drums")
    state = json.loads((demo_project / "session.json").read_text())
    state["stems"]["stem_a"]["chain"] = [{"type": "eq", "freq_hz": 2000, "gain_db": 3.0}]
    state["stems"]["stem_b"]["excluded"] = True
    (demo_project / "session.json").write_text(json.dumps(state))

    res = peek_project(str(demo_project))
    assert res["ok"] is True and res["name"] == demo_project.name
    tracks = {t["stem_id"]: t for t in res["tracks"]}
    assert "stem_b" not in tracks                          # soft-deleted stays hidden
    assert tracks["stem_a"]["devices"] == ["eq"] and tracks["stem_a"]["bus"] == "drums"
    assert res["buses"] == ["drums"]

    bare = tmp_path / "bare_stems"
    bare.mkdir()
    shutil.copy(demo_project / "stem_a.wav", bare / "loop.wav")
    res = peek_project(str(bare))
    assert res["ok"] is True and [t["stem_id"] for t in res["tracks"]] == ["loop"]
    assert not (bare / "session.json").exists()            # peek never writes

    assert peek_project(str(tmp_path / "missing"))["ok"] is False


# ---- multi-project workspace Inc 2 (docs/10-daw.md): project-keyed hub + operate ownership ----

def test_hub_keys_one_brain_per_project_root(tmp_path: Path):
    # ONE brain process, one Brain per open project root — the same dir (however
    # spelled) maps to the same Brain; a different dir gets its own.
    from orchestrator.daw_server import Hub

    a = tmp_path / "projA"; a.mkdir()
    b = tmp_path / "projB"; b.mkdir()
    hub = Hub(a)
    assert hub.get(a) is hub.boot
    assert hub.get(str(a) + "/") is hub.boot               # spelling-insensitive key
    assert hub.get(a / ".." / "projA") is hub.boot
    other = hub.get(b)
    assert other is not hub.boot and other.project_dir == b.resolve()
    assert hub.get(b) is other                             # stable on re-open


def test_operate_path_ownership_rules(tmp_path: Path):
    # The daw_invoke hijack fix: a declared DAW owns the operate path; an undeclared
    # client may only seed an unclaimed slot and never displaces an owner; a declared
    # reconnect replaces its dead predecessor; release is owner-only.
    from orchestrator.daw_server import Brain

    brain = Brain(tmp_path)
    daw1, daw2, tool = object(), object(), object()
    f_daw1, f_daw2, f_tool = (lambda *_: 1), (lambda *_: 2), (lambda *_: 3)

    assert brain.claim_operate(tool, f_tool, declared=False)     # legacy seed of empty slot
    assert brain.daw_invoke is f_tool
    assert brain.claim_operate(daw1, f_daw1, declared=True)      # the real DAW takes over
    assert brain.daw_invoke is f_daw1
    assert not brain.claim_operate(tool, f_tool, declared=False)  # a second client can't hijack
    assert brain.daw_invoke is f_daw1
    brain.release_operate(tool)                                   # non-owner release: no-op
    assert brain.daw_invoke is f_daw1
    assert brain.claim_operate(daw2, f_daw2, declared=True)       # DAW reconnect replaces
    assert brain.daw_invoke is f_daw2
    brain.release_operate(daw1)                                   # the DEAD connection's token
    assert brain.daw_invoke is f_daw2
    brain.release_operate(daw2)
    assert brain.daw_invoke is None


def test_socket_invoke_routes_to_the_declared_daw_per_project(tmp_path: Path):
    # End-to-end over real sockets: two projects, each with its own declared DAW
    # connection; a third (undeclared) client's `invoke {project}` reaches the RIGHT
    # DAW — and its own connection never receives a daw_request (the Inc 2 protocol:
    # explicit ownership + project keying, closing the last-connection-wins hijack).
    import asyncio
    from orchestrator.daw_server import Hub, _serve_client

    proj_a = tmp_path / "projA"; proj_a.mkdir()
    proj_b = tmp_path / "projB"; proj_b.mkdir()

    async def req(w, r, obj):
        w.write((json.dumps(obj) + "\n").encode()); await w.drain()
        return json.loads(await asyncio.wait_for(r.readline(), 5))

    async def scenario():
        hub = Hub(proj_a)
        server = await asyncio.start_server(
            lambda r, w: _serve_client(r, w, hub), "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        conns = [await asyncio.open_connection("127.0.0.1", port) for _ in range(3)]
        (ra, wa), (rb, wb), (rt, wt) = conns
        try:
            # every connection authenticates first (2026-09-10 agent audit P0-2)
            for r, w in conns:
                ok = await req(w, r, {"id": 0, "method": "auth",
                                      "params": {"token": hub.token}})
                assert ok["result"]["ok"]
            res = await req(wa, ra, {"id": 1, "method": "hello",
                                     "params": {"role": "daw", "project": str(proj_a)}})
            assert res["result"]["ok"] and res["result"]["project"] == str(proj_a.resolve())
            res = await req(wb, rb, {"id": 1, "method": "hello",
                                     "params": {"role": "daw", "project": str(proj_b)}})
            assert res["result"]["ok"]

            for target_dir, (rd, wd), tag in ((proj_a, (ra, wa), "A"),
                                              (proj_b, (rb, wb), "B")):
                wt.write((json.dumps({"id": 7, "method": "invoke",
                                      "params": {"project": str(target_dir),
                                                 "command": "noop", "args": {}}})
                          + "\n").encode())
                await wt.drain()
                # the daw_request lands on the DECLARED owner for that project…
                line = json.loads(await asyncio.wait_for(rd.readline(), 5))
                assert line.get("command") == "noop" and "daw_request" in line
                wd.write((json.dumps({"daw_reply": line["daw_request"],
                                      "result": {"ok": True, "via": tag}})
                          + "\n").encode())
                await wd.drain()
                # …and the tool client gets that DAW's reply back
                resp = json.loads(await asyncio.wait_for(rt.readline(), 5))
                assert resp["id"] == 7 and resp["result"]["via"] == tag
        finally:
            for r, w in conns:
                w.close()
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


# ---- multi-project workspace Inc 3 (docs/10-daw.md): chain-only import ----

def test_import_chain_from_adopts_only_the_devices(demo_project: Path, tmp_path: Path):
    # "Give my vocal the chain from my last song's vocal": ONLY the validated device
    # chain lands on the existing destination stem — no audio copies, and the
    # destination's own scalars / sends / routing stay untouched.
    from mix_engine.session import Session
    from orchestrator.daw_bridge import import_chain_from

    Session(demo_project)                                  # materialize session.json
    state = json.loads((demo_project / "session.json").read_text())
    state["stems"]["stem_a"].update({
        "gain_db": -4.0, "muted": True,
        "chain": [{"type": "saturation", "drive_db": 6.0},
                  {"type": "eq",
                   "band": {"freq_hz": 2000, "gain_db": 3.0, "q": 1.0, "shape": "peak"}},
                  {"type": "eq", "freq_hz": 500, "gain_db": 2.0},   # malformed: no band
                  {"type": "duck", "key_stem_id": "nope", "threshold_db": -30.0,
                   "depth_db": 6.0}]})                              # dangling key → drops
    (demo_project / "session.json").write_text(json.dumps(state))

    dest = _other_project(demo_project, tmp_path)
    d = Session(dest)
    d.route_stem("stem_a", "drums")                        # destination's OWN routing
    rid = d.add_return("reverb")["return_id"]
    d.set_send("stem_a", rid, -9.0)
    dstate = json.loads((dest / "session.json").read_text())
    dstate["stems"]["stem_a"]["gain_db"] = 1.5             # destination's own fader
    (dest / "session.json").write_text(json.dumps(dstate))

    res = import_chain_from(dest, str(demo_project), "stem_a", "stem_a")
    assert res["ok"] is True and res["stem_id"] == "stem_a"
    assert res["source_project"] == demo_project.name
    assert res["devices"] == ["eq", "saturation"]          # canonical order, bad entries gone
    assert len(res["devices_dropped"]) == 2

    st = json.loads((dest / "session.json").read_text())["stems"]["stem_a"]
    assert [e["type"] for e in st["chain"]] == ["eq", "saturation"]
    assert st["gain_db"] == 1.5 and not st.get("muted")    # destination scalars kept
    assert st["bus"] == "drums"                            # destination routing kept
    assert st["sends"] == {rid: -9.0}                      # destination sends kept
    assert not (dest / "stem_a 2.wav").exists()            # no audio materialized


def test_import_chain_from_honest_errors(demo_project: Path, tmp_path: Path):
    from orchestrator.daw_bridge import import_chain_from

    dest = _other_project(demo_project, tmp_path)
    assert import_chain_from(dest, str(tmp_path / "missing"), "x", "stem_a")["ok"] is False
    res = import_chain_from(dest, str(demo_project), "nope", "stem_a")
    assert res["ok"] is False and "nope" in res["error"]
    res = import_chain_from(dest, str(demo_project), "stem_a", "ghost")
    assert res["ok"] is False and "ghost" in res["error"]


# ---- track duplicate / cut-paste state (2026-07-19): the DAW's ⌘D + ⌘C/⌘X/⌘V ----

def test_copy_stem_state_duplicates_the_whole_mix_state_verbatim(demo_project: Path):
    # ⌘D / track paste: the native side clones clips + fader; this brings what the brain owns.
    # SAME project, so unlike the cross-project adopt everything carries verbatim — including
    # group membership and sends (the very same buses + returns exist here).
    from mix_engine.session import Session

    s = Session(demo_project)
    s.route_stem("stem_a", "drums")
    rid = s.add_return("reverb")["return_id"]
    s.set_send("stem_a", rid, -8.0)
    state = json.loads((demo_project / "session.json").read_text())
    state["stems"]["stem_a"].update({
        "gain_db": -3.0, "pan": -0.4, "muted": True, "tags": ["vocal"],
        "chain": [{"type": "saturation", "drive_db": 4.0},
                  {"type": "eq",
                   "band": {"freq_hz": 900, "gain_db": 2.0, "q": 1.0, "shape": "peak"}},
                  {"type": "eq", "freq_hz": 500, "gain_db": 2.0}]})   # malformed: no band
    (demo_project / "session.json").write_text(json.dumps(state))

    s = Session(demo_project)
    res = s.copy_stem_state("stem_a", "stem_b")
    assert res["ok"] is True
    assert res["devices"] == ["eq", "saturation"]      # canonical order; the bad entry dropped
    assert res["devices_dropped"] == ["eq"] and res["bus"] == "drums"

    st = json.loads((demo_project / "session.json").read_text())["stems"]["stem_b"]
    assert st["gain_db"] == -3.0 and st["pan"] == -0.4 and st["muted"] is True
    assert st["tags"] == ["vocal"]
    assert st["bus"] == "drums"                        # group membership carries (same project)
    assert st["sends"] == {rid: -8.0}                  # sends carry verbatim (same returns)
    assert st["path"] == "stem_b.wav"                  # its OWN audio is never touched

    # the source is untouched, and honest errors on unknown / self-copy
    assert json.loads((demo_project / "session.json").read_text())["stems"]["stem_a"]["gain_db"] == -3.0
    assert Session(demo_project).copy_stem_state("stem_a", "ghost")["ok"] is False
    assert Session(demo_project).copy_stem_state("stem_a", "stem_a")["ok"] is False


def test_copy_stem_state_reads_an_excluded_source(demo_project: Path):
    # ⌘X cuts the track: the DAW soft-excludes the stem but its state survives, so a paste
    # can still copy from it.
    from mix_engine.session import Session

    s = Session(demo_project)
    state = json.loads((demo_project / "session.json").read_text())
    state["stems"]["stem_a"].update({"gain_db": -5.0, "excluded": True,
                                     "chain": [{"type": "saturation", "drive_db": 3.0}]})
    (demo_project / "session.json").write_text(json.dumps(state))

    res = Session(demo_project).copy_stem_state("stem_a", "stem_b")
    assert res["ok"] is True and res["devices"] == ["saturation"]
    st = json.loads((demo_project / "session.json").read_text())["stems"]["stem_b"]
    assert st["gain_db"] == -5.0 and not st.get("excluded")   # the copy is LIVE, not excluded


def test_copy_stem_state_optional_skips_a_never_registered_source(demo_project: Path):
    # A drop filling a blank never-registered lane (the seeded "Audio 1") replaces a track
    # the brain has no stem for — there is nothing to copy, and that must be a clean no-op,
    # not the "⚠ effects copy failed — no track 'Audio 1' in this project" warning the DAW
    # showed on the very first drop into a fresh project (2026-08-09). Strict callers (⌘D
    # of a live registered track) keep the loud error so a rename-keying bug still speaks.
    from mix_engine.session import Session
    from orchestrator.daw_server import Brain

    Session(demo_project)                              # materialize session.json
    before = json.loads((demo_project / "session.json").read_text())["stems"]["stem_b"]

    res = Brain(demo_project).copy_stem_state("Audio 1", "stem_b", optional=True)
    assert res["ok"] is True and res["skipped"] is True
    after = json.loads((demo_project / "session.json").read_text())["stems"]["stem_b"]
    assert after == before                             # nothing copied, nothing touched

    # optional with a REAL source still copies (the deleted-clip-then-refill lane)
    res = Brain(demo_project).copy_stem_state("stem_a", "stem_b", optional=True)
    assert res["ok"] is True and "skipped" not in res and "chains" in res

    # strict stays loud on an unknown source (⌘D must not silently lose state)
    res = Brain(demo_project).copy_stem_state("Audio 1", "stem_b")
    assert res["ok"] is False and "no track" in res["error"]


def test_brain_restore_stem_unexcludes_without_resetting(demo_project: Path):
    # ⌘X → ⌘V: pasting a cut track back is an un-exclude, never a re-registration — a
    # re-register would reset_stem the excluded record and the devices would be gone.
    from mix_engine.session import Session
    from orchestrator.daw_server import Brain

    Session(demo_project)                              # materialize session.json
    state = json.loads((demo_project / "session.json").read_text())
    state["stems"]["stem_a"].update({
        "gain_db": -4.0, "excluded": True,
        "chain": [{"type": "eq",
                   "band": {"freq_hz": 700, "gain_db": 3.0, "q": 1.0, "shape": "peak"}}]})
    (demo_project / "session.json").write_text(json.dumps(state))

    res = Brain(demo_project).restore_stem("stem_a")
    assert res["ok"] is True and "chains" in res
    st = json.loads((demo_project / "session.json").read_text())["stems"]["stem_a"]
    assert st["excluded"] is False
    assert [e["type"] for e in st["chain"]] == ["eq"]   # the chain survived the round trip
    assert st["gain_db"] == -4.0
    assert Brain(demo_project).restore_stem("ghost")["ok"] is False


def test_peek_project_reports_track_colours_from_the_edit(demo_project: Path, tmp_path: Path):
    # Track colours live in the Tracktion EDIT, never in session.json — the peek carries the
    # explicit colour where one is set, plus the ORDINAL that the app's palette fallback is
    # keyed to, so a browsed-in track keeps the colour it had in the source project.
    from mix_engine.session import Session
    from orchestrator.daw_bridge import peek_project

    Session(demo_project)
    (demo_project / "session.tracktionedit").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<EDIT>\n'
        '  <TRACK id="1" name="stem_a" colour="ff2fb37a"/>\n'      # explicit
        '  <TRACK id="2" name="bus:drums" colour="ffaa3366"/>\n'   # group tint, not a stem
        '  <TRACK id="3" name="stem_b"/>\n'                        # no colour -> ordinal fallback
        '</EDIT>\n')

    res = peek_project(str(demo_project))
    assert res["ok"] is True
    tracks = {t["stem_id"]: t for t in res["tracks"]}
    assert tracks["stem_a"]["colour"] == "ff2fb37a" and tracks["stem_a"]["ordinal"] == 0
    assert tracks["stem_b"]["colour"] == "" and tracks["stem_b"]["ordinal"] == 1   # bus doesn't consume one
    assert res["bus_colours"] == {"drums": "ffaa3366"}


def test_peek_project_survives_a_missing_or_corrupt_edit(demo_project: Path, tmp_path: Path):
    # A bare stems folder (never opened in the DAW) has no edit at all, and a corrupt one must
    # not take the peek down with it — colours are a nicety, listing the tracks is the job.
    from mix_engine.session import Session
    from orchestrator.daw_bridge import peek_project

    Session(demo_project)
    res = peek_project(str(demo_project))                    # no edit file written yet
    assert res["ok"] is True and res["bus_colours"] == {}
    assert all(t["colour"] == "" and t["ordinal"] == -1 for t in res["tracks"])

    (demo_project / "session.tracktionedit").write_text("<EDIT><TRACK name=oops")   # malformed
    res = peek_project(str(demo_project))
    assert res["ok"] is True and [t["stem_id"] for t in res["tracks"]] == ["stem_a", "stem_b"]


def test_boot_sync_carries_the_topology_and_never_writes(demo_project: Path):
    # Load-time convergence (2026-08-06): session.json owns bus/return EXISTENCE and the
    # bus:/return: tracks are a projection of it, but nothing ran that projection at load —
    # boot asked only for `analyze`, which carries no topology. So the edit and the session
    # could disagree indefinitely, and the disagreement was then resolved by whichever
    # unrelated gesture next happened to carry one, silently deleting the user's return
    # tracks mid-action (and, until then, bouncing wet the Python engine could not see).
    #
    # Two properties make running this on EVERY load safe: it carries the whole topology,
    # and it is strictly read-only — a repair that mutated session.json on open would make
    # the file a function of how often the project was opened.
    from mix_engine.session import Session
    from orchestrator.daw_server import Brain

    sess = Session(demo_project)
    sess.add_return("reverb")
    before = (demo_project / "session.json").read_bytes()

    res = Brain(str(demo_project)).boot_sync()

    for key in ("chains", "buses", "bus_outputs", "send_returns", "readout"):
        assert key in res, f"boot_sync must carry {key} — it is the DAW's whole reconcile payload"
    assert [r["id"] for r in res["send_returns"]["returns"]] == ["rv"]
    assert (demo_project / "session.json").read_bytes() == before, "boot_sync must not write"


def _write_swung_loop(path: Path, bpm: float = 120.0, bars: int = 4,
                      swing: float = 0.65) -> None:
    """A click loop whose off-eighths land late — a groove worth stealing."""
    import numpy as np
    import soundfile as sf
    sr, beat = 44100, 60.0 / bpm
    buf = np.zeros(int(sr * (bars * 4 * beat + 1.0)), dtype="float32")
    click = (np.exp(-np.arange(1500) / 220.0)
             * np.sin(2 * np.pi * 1400.0 * np.arange(1500) / sr)).astype("float32")
    for bar in range(bars):
        for b in range(4):
            base = (bar * 4 + b) * beat
            for t in (base, base + swing * beat):
                i = int(t * sr)
                buf[i:i + len(click)] += click
    sf.write(path, np.stack([buf, buf], axis=1), sr)


def test_extract_clip_groove_reads_a_loop_and_reports_the_feel(tmp_path: Path):
    """The DAW's audio path: a file path in, a pool template out. The DAW's own
    tempo is passed as authoritative — a loop's filename is not evidence."""
    p = tmp_path / "swung.wav"
    _write_swung_loop(p)
    res = extract_clip_groove(tmp_path, str(p), bpm=120.0)

    assert res["ok"] is True, res.get("reason")
    assert res["notes_per_beat"] == 2, "the swing belongs in the latenesses, not a triplet grid"
    lat = res["latenesses"]
    assert len(lat) == 8
    assert all(lat[i] > 0.5 for i in (1, 3, 5, 7)), lat     # the off-beats carry it
    assert all(abs(lat[i]) < 0.05 for i in (0, 2, 4, 6)), lat


def test_extract_clip_groove_is_honest_rather_than_flat(tmp_path: Path):
    """A quantized loop has no groove. Returning a near-zero template would
    report success and change nothing the user can hear."""
    p = tmp_path / "straight.wav"
    _write_swung_loop(p, swing=0.5)
    res = extract_clip_groove(tmp_path, str(p), bpm=120.0)
    assert res["ok"] is False
    assert "already dead on the grid" in res["reason"]

    # a bad path is a reason, never a crash (the detect_clip_key rule)
    bad = extract_clip_groove(tmp_path, str(tmp_path / "nope.wav"), bpm=120.0)
    assert bad["ok"] is False and "could not read" in bad["reason"]


# ---- clip.pitch_correct's ears (2026-08-08) --------------------------------

def _write_two_note_take(path: Path, cents_off_second: float = -40.0) -> None:
    """Two held monophonic notes; the second is detuned. A vocal in miniature."""
    import numpy as np
    import soundfile as sf
    sr = 44100

    def tone(midi: float, seconds: float) -> "np.ndarray":
        t = np.arange(int(sr * seconds)) / sr
        f = 440.0 * 2.0 ** ((midi - 69) / 12.0)
        sig = sum(np.sin(2 * np.pi * f * h * t) / (h ** 1.2) for h in (1, 2, 3, 4))
        return (sig * np.minimum(1.0, np.minimum(t * 25, (seconds - t) * 25)) * 0.2)

    take = np.concatenate([tone(57.0, 1.2), np.zeros(int(sr * 0.3)),
                           tone(60.0 + cents_off_second / 100.0, 1.2)])
    sf.write(path, np.stack([take, take], axis=1).astype("float32"), sr)


def test_tune_clip_notes_measures_the_off_note_clip_relative(tmp_path: Path):
    from orchestrator.daw_bridge import tune_clip_notes

    p = tmp_path / "take.wav"
    _write_two_note_take(p)
    res = tune_clip_notes(tmp_path, str(p), min_cents=15.0)
    assert res["ok"] is True, res.get("reason")
    assert res["notes_found"] == 2
    assert len(res["notes"]) == 1                      # only the off one crosses 15c
    note = res["notes"][0]
    assert note["cents_off"] == pytest.approx(-40.0, abs=6.0)
    assert 1.3 < note["start_s"] < 1.7                  # clip-relative, the SECOND note


def test_tune_clip_notes_declines_in_tune_and_polyphony(tmp_path: Path):
    from orchestrator.daw_bridge import tune_clip_notes

    p = tmp_path / "fine.wav"
    _write_two_note_take(p, cents_off_second=0.0)
    res = tune_clip_notes(tmp_path, str(p), min_cents=15.0)
    assert res["ok"] is False and "already within" in res["reason"]

    import numpy as np
    import soundfile as sf
    sr = 44100
    t = np.arange(int(sr * 1.5)) / sr
    chord = sum(sum(np.sin(2 * np.pi * 440.0 * 2 ** ((m - 69) / 12.0) * h * t) / h
                    for h in (1, 2, 3, 4)) for m in (60, 64, 67)) * 0.1
    poly = tmp_path / "chord.wav"
    sf.write(poly, np.stack([chord, chord], axis=1).astype("float32"), sr)
    res = tune_clip_notes(tmp_path, str(poly))
    assert res["ok"] is False and "POLYPHONIC" in res["reason"]

    bad = tune_clip_notes(tmp_path, str(tmp_path / "nope.wav"))
    assert bad["ok"] is False and "could not read" in bad["reason"]


def test_extract_groove_window_params_survive_the_dispatch_seam():
    """The 2026-08-08 clip-scoping fix was silently DROPPED at daw_server's
    dispatch (App sent source_offset_s/length_s/beat_offset_s; the seam
    forwarded path+bpm only). Pin every parameter name across all three layers
    by signature, so the next added param cannot go missing silently."""
    import inspect

    from orchestrator import daw_server
    from orchestrator.daw_bridge import extract_clip_groove, tune_clip_notes

    for hub_fn, bridge_fn, params in [
        (daw_server.Brain.extract_clip_groove, extract_clip_groove,
         ("source_offset_s", "length_s", "beat_offset_s")),
        (daw_server.Brain.tune_clip_notes, tune_clip_notes,
         ("source_offset_s", "length_s", "min_cents")),
    ]:
        hub_sig = set(inspect.signature(hub_fn).parameters)
        bridge_sig = set(inspect.signature(bridge_fn).parameters)
        for name in params:
            assert name in hub_sig, f"{hub_fn.__qualname__} lost {name}"
            assert name in bridge_sig, f"{bridge_fn.__qualname__} lost {name}"
    # …and the dispatch source itself forwards them (the layer that dropped them).
    src = inspect.getsource(daw_server)
    for name in ("source_offset_s", "beat_offset_s", "min_cents"):
        assert src.count(f'params.get("{name}"') >= 1, f"dispatch drops {name}"


def test_midrun_exchange_replays_into_the_next_turn(tmp_path: Path):
    """A message typed while a run is in flight is answered by the stateless
    side channel (chat_status_reply) — the real ChatSession is mid-turn(). The
    exchange must land in the session's history at the next real turn, marked
    as mid-run and unapplied, or follow-ups look ignored ("separated threads",
    2026-08-09). Replay is once: the queue drains."""
    from types import SimpleNamespace

    from orchestrator import daw_bridge

    session = SimpleNamespace(history=[])
    daw_bridge.record_midrun_exchange(tmp_path, "cap the Q at 8",
                                      "will do once this pass lands")
    daw_bridge._replay_midrun(session, tmp_path)
    assert [m["role"] for m in session.history] == ["user", "assistant"]
    assert "cap the Q at 8" in session.history[0]["content"]
    assert "nothing from this message has been applied" in session.history[0]["content"]
    assert session.history[1]["content"] == "will do once this pass lands"
    daw_bridge._replay_midrun(session, tmp_path)
    assert len(session.history) == 2


def _write_transcript(path: Path, *pairs: tuple[str, str]) -> None:
    """A DAW chat-log.json: rendered display blocks, user/prose interleaved."""
    blocks = []
    for user, reply in pairs:
        blocks.append({"k": "user", "t": user})
        blocks.append({"k": "move", "type": "gain"})   # UI chrome — never rehydrated
        blocks.append({"k": "prose", "t": reply})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blocks))


def test_a_session_rehydrates_from_its_own_transcript(tmp_path: Path):
    """Sessions (2026-08-15): several conversations live under one edit, so the
    agent's memory has to follow the one the DAW says is live. Without the id it
    reads the pre-sessions single log — that path is what the VST and gateway
    still use, and must not move."""
    from orchestrator import daw_bridge

    _write_transcript(tmp_path / "chats" / "s1.json", ("more space on the hats", "plate in"))
    _write_transcript(tmp_path / "chats" / "s2.json", ("de-ess the vocal", "6 dB at 7k"))
    _write_transcript(tmp_path / "chat-log.json", ("legacy talk", "legacy reply"))

    # A restored user line says it is one (reader S9·R5, test_reader_p2_py5.py); the reply is
    # as it was said.
    was = daw_bridge.RESTORED_USER_PREFIX
    assert [m["content"] for m in daw_bridge.rehydrate_history(tmp_path, "s1")] \
        == [was + "more space on the hats", "plate in"]
    assert [m["content"] for m in daw_bridge.rehydrate_history(tmp_path, "s2")] \
        == [was + "de-ess the vocal", "6 dB at 7k"]
    assert [m["content"] for m in daw_bridge.rehydrate_history(tmp_path)] \
        == [was + "legacy talk", "legacy reply"]
    # A session with no transcript yet is the whole point of "new session": empty.
    assert daw_bridge.rehydrate_history(tmp_path, "s-brand-new") == []


def test_chat_sessions_are_keyed_per_conversation(tmp_path: Path):
    """One ChatSession per (edit, session id). Switching must switch what the agent
    remembers; starting a new one must leave it with nothing — both fall out of the
    key, so this pins the key itself rather than the two behaviours separately."""
    from orchestrator import daw_bridge

    _write_transcript(tmp_path / "chats" / "s1.json", ("more space on the hats", "plate in"))
    daw_bridge.reset_chat_session(tmp_path)

    one = daw_bridge._chat_session(tmp_path, "s1")
    assert [m["content"] for m in one.history] == [
        daw_bridge.RESTORED_USER_PREFIX + "more space on the hats", "plate in"]
    two = daw_bridge._chat_session(tmp_path, "s2")
    assert two is not one and two.history == []          # a fresh session starts blank
    assert daw_bridge._chat_session(tmp_path, "s1") is one   # …and switching back returns

    # A mid-run exchange belongs to the conversation it was typed into.
    daw_bridge.record_midrun_exchange(tmp_path, "cap the Q", "will do", "s1")
    daw_bridge._replay_midrun(two, tmp_path, "s2")
    assert two.history == []
    daw_bridge._replay_midrun(one, tmp_path, "s1")
    assert "cap the Q" in one.history[-2]["content"]

    # An edit switch drops every conversation in that folder, not just the live one.
    daw_bridge.reset_chat_session(tmp_path)
    assert daw_bridge._chat_session(tmp_path, "s2") is not two


def test_run_on_an_empty_project_declines_for_free(tmp_path: Path):
    """Run on a blank project used to enter the full pass pipeline — a billed LLM
    pass proposing nothing actionable (2026-08-09 §1 audit). It must decline before
    any pass, for free, with a next step in the reply."""
    from orchestrator.daw_server import Brain

    result = Brain(tmp_path).run("make it loud", use_llm=False)
    assert result["stop_reason"] == "nothing to mix"
    assert result["passes"] == 0 and result["cost_usd"] == 0.0
    assert "no audio" in result["chat"]


def test_undo_run_replays_user_coedits_made_after_the_run(demo_project: Path):
    """A proposal's undo restores each touched stem's ENTIRE prior dict — which used
    to silently erase any co-edit the user made AFTER the run (2026-08-09 §4 audit
    P0: run → user rides the fader → 'undo run' ate the fader move). Co-edits are
    journaled at the apply_coedits choke point and REPLAYED after the restores."""
    from mix_engine.session import Session

    from orchestrator.daw_bridge import (apply_coedits, max_applied_seq, undo_run)

    # the agent's run: one applied proposal moving stem_a's gain
    session = Session(demo_project)
    floor = max_applied_seq(demo_project)
    from mix_engine.models import ProposalRequest
    prop = session.propose(ProposalRequest(
        rationale="run move",
        predicted_impact=[{"metric": "stem(stem_a).lufs_s",
                           "predicted_delta": 2.0, "unit": "LUFS"}],
        actions=[{"type": "gain", "stem_id": "stem_a", "delta_db": 2.0}]))
    assert prop.accepted and session.apply(prop.proposal_id).ok

    # the user's later co-edit on the SAME stem (fader ride via the coedit channel)
    assert apply_coedits(demo_project, [
        "http://127.0.0.1:1/api/set_stem?stem=stem_a&pan=-0.4"])["applied"] == 1

    assert undo_run(demo_project, floor) == 1
    after = Session(demo_project)
    assert after.stems["stem_a"]["gain_db"] == 0.0     # the run's move is gone…
    assert after.stems["stem_a"]["pan"] == -0.4        # …the user's later pan SURVIVES


# --------------------------------------------------------------------------- #
# separate_track — the DAW's AI stem-split boundary (2026-08-13)               #
# --------------------------------------------------------------------------- #

def test_separate_track_writes_stem_wavs_only(demo_project: Path, monkeypatch):
    """The bridge op writes <source>.<stem>.wav and returns their names — and
    deliberately does NOT register them (the DAW's add-track import path owns
    registration; the loop-deck lesson)."""
    monkeypatch.setenv("MIX_AGENT_SEPARATOR", "fake")
    src = next(demo_project.glob("*.wav"))
    stages: list[str] = []
    result = separate_track(demo_project, str(src), stems=4,
                            on_stage=lambda e: stages.append(e["stage"]))
    assert result["ok"] is True
    names = [s["name"] for s in result["stems"]]
    assert names == ["vocals", "drums", "bass", "other"]
    for s in result["stems"]:
        wav = demo_project / s["wav"]
        assert wav.exists() and wav.name.startswith(src.stem + ".")
        wav.unlink()   # keep the fixture pristine for later assertions
    assert {"loading", "separating", "writing"} <= set(stages)
    # No side registration: the op never touches the engine session — an absent
    # session.json is the strongest form of that claim.
    session_file = demo_project / "session.json"
    if session_file.exists():
        session = json.loads(session_file.read_text())
        assert not any(sid.startswith(src.stem + ".")
                       for sid in session.get("stems", {}))


def test_separate_track_rejects_a_bad_file(demo_project: Path):
    result = separate_track(demo_project, str(demo_project / "nope.wav"))
    assert result["ok"] is False and "unsupported" in result["error"]


def test_separate_track_cancel_lands_on_a_stage_boundary(demo_project: Path, monkeypatch):
    monkeypatch.setenv("MIX_AGENT_SEPARATOR", "fake")
    src = next(demo_project.glob("*.wav"))
    result = separate_track(demo_project, str(src), stems=4,
                            should_cancel=lambda: True)
    assert result["ok"] is False and result["error"] == "cancelled"


def test_separate_track_announces_first_use_download(demo_project: Path, monkeypatch):
    """A 166 MB first-use pull must be its OWN stage with the size in the detail
    — a bare 'separating…' while it downloads reads as a hang (James, 2026-08-13)."""
    import mix_engine.separation as sep_mod

    class StubSep:
        stems, names = 4, sep_mod.FOUR_STEM
        fetched = False
        def models_cached(self):  # noqa: D102
            return False
        def download_detail(self):
            return "AI model htdemucs — about 166 MB, first use only"
        def fetch_models(self):
            self.fetched = True
        def separate(self, audio, sample_rate):
            assert self.fetched, "download stage must run before inference"
            return sep_mod.FakeSeparator(4).separate(audio, sample_rate)

    stub = StubSep()
    monkeypatch.setattr(sep_mod, "get_separator", lambda stems, kind=None: stub)
    src = next(demo_project.glob("*.wav"))
    stages: list[tuple[str, str]] = []
    result = separate_track(demo_project, str(src), stems=4,
                            on_stage=lambda e: stages.append((e["stage"], e["detail"])))
    assert result["ok"] is True and stub.fetched
    names = [s for s, _ in stages]
    assert "downloading" in names and names.index("downloading") < names.index("separating")
    detail = dict(stages)["downloading"]
    assert "166 MB" in detail and "first use" in detail
    for s in result["stems"]:
        (demo_project / s["wav"]).unlink()   # fixture pristine


def test_import_stem_overwrite_never_truncates_the_old_inode(demo_project: Path, tmp_path: Path):
    """Regression (2026-08-13 slot.launch SIGBUS): the DAW's engine memory-maps
    playing project wavs, and import_stem re-registering a deck channel used to
    shutil.copyfile ONTO the mapped inode — truncate-in-place, SIGBUS on the
    audio thread's next page-in. The copy must land by rename: a reader holding
    the OLD file (stand-in for the engine's mapping) keeps its bytes; the path
    points at a NEW inode with the new bytes."""
    import numpy as np
    import os

    old = tmp_path / "old.wav"
    new = tmp_path / "new.wav"
    sf.write(str(old), np.full((1000, 1), 0.25, dtype="float32"), 44100, subtype="FLOAT")
    sf.write(str(new), np.full((2000, 1), -0.5, dtype="float32"), 44100, subtype="FLOAT")

    assert import_stem(demo_project, str(old), stem_id="deck_ch")["ok"]
    target = demo_project / "deck_ch.wav"
    inode_before = os.stat(target).st_ino
    with open(target, "rb") as mapped:          # the engine's mmap, in spirit
        head_before = mapped.read(64)
        # replace=True: the Loop Deck's follow, the one import that replaces a stem's own file
        # with other audio (any other import writes the next free name - reader D4·R1)
        assert import_stem(demo_project, str(new), stem_id="deck_ch", replace=True)["ok"]
        assert os.stat(target).st_ino != inode_before   # rename = new inode
        mapped.seek(0)
        assert mapped.read(64) == head_before   # old reader: bytes intact, no truncation
    assert sf.info(str(target)).frames == 2000  # the path serves the new audio


# ---------- Auto Mix over the socket: a cancel is named, a decline is not "applied" (2026-09-02) ----------

def test_auto_mix_event_names_a_cancel_and_keeps_a_decline_honest(demo_project: Path, monkeypatch):
    """`_do_auto_mix` answers run_done for a finished, a cancelled AND a declined Auto Mix.
    The DAW used to read all three as "Auto Mix applied": a Stop mid-bundle carried no
    stop_reason, and the no-audio decline ({ok: False, chat}) had nothing telling it apart
    from a success. Now a cancel rides `stop_reason: cancelled` and the decline keeps its
    own ok:false + chat (the literal ok:True must not win over the brain's verdict)."""
    import asyncio
    import orchestrator.daw_server as ds

    sent: list[dict] = []

    async def send(msg: dict) -> None:
        sent.append(msg)

    class FakeBrain:
        def __init__(self, result):
            self._result = result

        def auto_mix(self, brief, should_cancel=None, on_stage=None, print_midi=False):
            return dict(self._result)

    # cancelled mid-bundle: the cancel flag is set by the time the pass returns
    sent.clear()
    asyncio.run(ds._do_auto_mix(send, FakeBrain({"ok": True, "moves": [], "chains": {}}),
                                7, "loud", lambda: True))
    done = [m for m in sent if m.get("event") == "run_done"]
    assert len(done) == 1 and done[0]["stop_reason"] == "cancelled"

    # not cancelled: no stop_reason is invented
    sent.clear()
    asyncio.run(ds._do_auto_mix(send, FakeBrain({"ok": True, "moves": [], "chains": {}}),
                                8, "loud", lambda: False))
    done = [m for m in sent if m.get("event") == "run_done"]
    assert len(done) == 1 and "stop_reason" not in done[0]

    # the no-audio decline keeps ok:false + its sentence
    sent.clear()
    asyncio.run(ds._do_auto_mix(send, FakeBrain({"ok": False, "moves": [],
                                                 "chat": "there's no audio to auto-mix yet"}),
                                9, "loud", lambda: False))
    done = [m for m in sent if m.get("event") == "run_done"]
    assert len(done) == 1 and done[0]["ok"] is False and "no audio" in done[0]["chat"]


# ---------- rename_stem: the brain follows a track rename (2026-09-02 agent-workspace audit) ----------

def test_rename_stem_rekeys_every_reference_and_keeps_the_audio(demo_project: Path):
    """The stem id IS the track name everywhere the brain speaks about it. Before the endpoint
    a renamed audio track kept its old id: later co-edits refused, chains never landed. The
    record moves; its wav does not (the DAW's clips reference the file) — discovery claims the
    file by the record's path, so the next Session neither mints a duplicate nor marks it missing."""
    from mix_engine.session import Session
    from mix_engine.models import ProposalRequest

    from orchestrator.daw_bridge import (apply_coedits, rename_stem, resolve_chains,
                                         chains_with_readout, import_stem)

    # state that names stem_a: a device + a level (co-edits, journaled), a duck on stem_b keyed
    # to it, a group bus it is routed into, a send to a return, and an applied proposal
    assert apply_coedits(demo_project, [
        "http://127.0.0.1:1/api/add_device?device_type=eq&stem=stem_a",
        "http://127.0.0.1:1/api/set_stem?stem=stem_a&gain_db=-3",
    ])["failed"] == 0
    s = Session(demo_project)
    s.set_duck("stem_b", "stem_a")
    s.route_stem("stem_a", "drums")
    s.add_return("reverb")
    s = Session(demo_project)
    ret = next(iter(s.returns))
    s.set_send("stem_a", ret, -12.0)
    prop = s.propose(ProposalRequest(
        rationale="run move",
        predicted_impact=[{"metric": "stem(stem_a).lufs_s", "predicted_delta": 2.0, "unit": "LUFS"}],
        actions=[{"type": "gain", "stem_id": "stem_a", "delta_db": 2.0}]))
    assert prop.accepted and s.apply(prop.proposal_id).ok

    res = rename_stem(demo_project, "stem_a", "lead a")
    assert res["renamed"] and res["stem"] == "lead a"

    s2 = Session(demo_project)                    # a fresh Session = a fresh discovery
    assert "lead a" in s2.stems and "stem_a" not in s2.stems, "no duplicate, no ghost"
    rec = s2.stems["lead a"]
    assert rec["path"] == "stem_a.wav" and not rec.get("excluded") and not rec.get("missing")
    assert rec["gain_db"] == -1.0                 # -3 co-edit + the proposal's +2
    assert [e["type"] for e in rec["chain"]] == ["eq"]
    assert rec["bus"] == "drums" and rec["sends"][ret] == -12.0
    duck = next(e for e in s2.stems["stem_b"]["chain"] if e["type"] == "duck")
    assert duck["key_stem_id"] == "lead a"
    log = s2._state["coedit_log"]
    assert log and all("stem=stem_a" not in e["url"] for e in log)
    assert any("stem=lead+a" in e["url"] for e in log)
    record = s2._state["proposals"][0]
    assert record["actions"][0]["stem_id"] == "lead a" and "lead a" in record["inverse"]

    # the resolved chains and a co-edit both use the new id
    chains = resolve_chains(s2._state, 0.0, None, native_returns=True)
    assert "lead a" in chains and "stem_a" not in chains
    assert apply_coedits(demo_project, [
        "http://127.0.0.1:1/api/set_stem?stem=lead+a&pan=0.5"])["failed"] == 0
    assert Session(demo_project).stems["lead a"]["pan"] == 0.5

    # a render still finds the audio behind the new id
    out = chains_with_readout(demo_project)
    assert "lead a" in out["chains"] and out["readout"].get("lufs_i") is not None

    # the proposal's undo restores the RENAMED record — no resurrected 'stem_a'
    s3 = Session(demo_project)
    assert s3.undo(record["proposal_id"]).ok
    s3 = Session(demo_project)
    assert "stem_a" not in s3.stems and s3.stems["lead a"]["gain_db"] == -3.0

    # audio re-dropped onto the renamed track (import_stem writes <id>.wav): the record follows
    # the new file and the old one is retired, never minted as a ghost the render would mix in
    assert import_stem(demo_project, str(demo_project / "stem_b.wav"), stem_id="lead a")["ok"]
    s4 = Session(demo_project)
    assert s4.stems["lead a"]["path"] == "lead a.wav"
    assert "stem_a" not in s4.stems and "stem_a.wav" in s4._state["retired_wavs"]
    assert (demo_project / "stem_a.wav").exists()   # the user's file is never deleted

    # refusals + no-ops
    with pytest.raises(ValueError):
        rename_stem(demo_project, "lead a", "stem_b")   # taken
    with pytest.raises(ValueError):
        rename_stem(demo_project, "lead a", "master")   # reserved
    with pytest.raises(ValueError):
        rename_stem(demo_project, "lead a", "drums")    # a bus id
    assert rename_stem(demo_project, "lead a", "lead a")["renamed"] is False
    assert rename_stem(demo_project, "Synth 2", "Synth 3")["renamed"] is False   # no brain stem: not an error


def test_a_turn_reports_what_it_cost_not_what_the_conversation_cost():
    """A chat turn's price is the tokens THAT turn added, not the session's running total.

    `ChatSession.usage` counts the whole conversation (it is the same object across turns),
    so pricing it directly would bill every turn for everything said before it. And a cached
    read is a tenth of the input rate: on a long conversation almost every input token is a
    cache hit, so charging them as fresh input reports several times the real number.
    """
    from orchestrator.daw_bridge import _turn_cost_usd

    class FakeSession:
        _model = "claude-opus-5"
        def __init__(self, **usage):
            self.usage = {"input_tokens": 0, "output_tokens": 0,
                          "cache_read_input_tokens": 0,
                          "cache_creation_input_tokens": 0, **usage}

    zero = {"input_tokens": 0, "output_tokens": 0,
            "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    assert _turn_cost_usd(FakeSession(), zero) == 0.0     # nothing said, nothing owed

    # the conversation so far, then ONE turn on top of it
    before = {"input_tokens": 1_000, "output_tokens": 500,
              "cache_read_input_tokens": 40_000, "cache_creation_input_tokens": 0}
    after = {"input_tokens": 1_200, "output_tokens": 900,
             "cache_read_input_tokens": 80_000, "cache_creation_input_tokens": 0}
    turn = _turn_cost_usd(FakeSession(**after), before)
    prior = _turn_cost_usd(FakeSession(**before), zero)
    whole = _turn_cost_usd(FakeSession(**after), zero)
    assert turn > 0.0
    assert turn < whole, "a turn must cost less than the conversation that contains it"
    # …and exactly the difference: price is linear in tokens, so this is the claim itself,
    # not a fraction tuned until it passed
    assert abs(whole - (turn + prior)) < 2e-4

    # a cached read is not fresh input: the same token count costs several times more when
    # none of it is cached
    cached = _turn_cost_usd(FakeSession(input_tokens=200, output_tokens=900,
                                        cache_read_input_tokens=40_000), zero)
    fresh = _turn_cost_usd(FakeSession(input_tokens=40_200, output_tokens=900), zero)
    assert fresh > cached * 3

    # an unusable session is a readout failure, never a raised turn
    class Broken:
        usage = {"input_tokens": 1}
        @property
        def _model(self):
            raise RuntimeError("gone")
    assert _turn_cost_usd(Broken(), {}) == 0.0


def test_applied_chains_reports_the_limiter_makeup_it_is_actually_applying(tmp_path):
    """2026-09-08 device audit, 01-dynamics-core F12.

    The limiter's makeup is resolved at RENDER time (`target_lufs_i - pre_lufs`) and lives in
    the readout cache, not in session.json. `_chat_context_stable` called `resolve_chains` with
    neither argument, so the "Applied chains" block the chat agent reasons from reported
    `limiter.makeup_db: 0.0` unconditionally — whatever the limiter was doing to the master.
    """
    import json
    import shutil
    from pathlib import Path

    from mix_engine.session import Session, MASTER
    from orchestrator import daw_bridge

    proj = tmp_path / "proj"
    repo_root = Path(__file__).resolve().parents[2]
    shutil.copytree(repo_root / "test_projects" / "suno_demo", proj)
    session = Session(proj)
    session._state["stems"].pop(MASTER, None)
    session._state["master_chain"] = [{"type": "limiter", "stem_id": "master",
                                       "ceiling_dbtp": -1.0, "target_lufs_i": -14.0}]
    session._persist()

    def _makeup(blob: str) -> float:
        line = [ln for ln in blob.splitlines() if "Applied chains" in ln][0]
        chains = json.loads(line.split("Applied chains: ", 1)[1])
        lims = [e for e in chains["master"]["entries"] if e.get("type") == "limiter"]
        assert lims, "the master chain must carry the limiter this test is about"
        return float(lims[0]["makeup_db"])

    cache: dict = {}
    got = _makeup(daw_bridge._chat_context_stable(proj, "agent", cache=cache))
    # session_readout measures and writes the real pre-limiter loudness back into the cache,
    # so the makeup must be exactly what the limiter derives from it.
    assert cache.get("pre_lufs_i") is not None, "the readout must have measured the pre-limiter LUFS"
    expected = -14.0 - float(cache["pre_lufs_i"])
    assert abs(got - expected) < 0.05, (
        f"Applied chains reports makeup {got:+.2f} dB; the limiter targeting -14 LUFS from a "
        f"measured {cache['pre_lufs_i']:+.2f} applies {expected:+.2f} dB")
    # NEGATIVE CONTROL: the whole point is that this is not 0. If the demo ever happened to sit
    # exactly at its target the assertion above would pass on the bug, so require a real value.
    assert abs(got) > 0.2, (
        f"makeup {got:+.2f} dB is too close to zero for this test to distinguish the fix from "
        "the bug — pick a target further from the demo's loudness")
