"""Phase 16 (docs/05-roadmap.md): Auto Mix — the one-button flow.

Covers: the deterministic bundle policy (pure), the full websocket-driven
flow (staged progress, loudness within ±1 LU at TP ≤ −1 dBTP, the worst
masking pair strictly reduced, one click restoring the pre-automix render
bit-identically), the engine-level flow on the second demo project, and the
previous-bounce A/B file.
"""
import json

import numpy as np
import pytest
import soundfile as sf
from starlette.testclient import TestClient

from mix_engine import analysis, dsp
from mix_engine.session import Session
from orchestrator import automix
from orchestrator.agents import RuleAgent
from orchestrator.chat import ScriptedChat
from orchestrator.gateway import create_app
from orchestrator.loop import EngineClient, policy_gate, run_passes

from test_phase1_acceptance import call
from test_phase10_efficiency import in_memory_engine


def add_masker(project) -> None:
    """A deterministic mid-band masker stem sitting ~7 dB over stem_a in its
    worst band (same fixture as the benchmark's clarity case)."""
    info = sf.info(str(project / "stem_a.wav"))
    t = np.arange(info.frames) / info.samplerate
    masker = 0.2 * sum(np.sin(2 * np.pi * f * t) for f in (500.0, 650.0, 800.0))
    sf.write(project / "masker.wav", masker.astype(np.float32),
             info.samplerate)


def worst_masking_excess(project) -> float:
    """masking(stem_a, masker) on the stems' current (post-chain) renders."""
    session = Session(project)
    audio_a, sample_rate = dsp.render_stem(session, "stem_a")
    audio_m, _ = dsp.render_stem(session, "masker")
    return analysis.analyze_masking(audio_a, audio_m, sample_rate,
                                    stem_a="stem_a",
                                    stem_b="masker").worst_band_excess_db


# ---------- the bundle policy (pure) ----------

def test_build_bundles_policy():
    tags = {"vox": ["vocal"], "bass": ["bass"], "pads": ["pad"], "loud": []}
    stems = {
        "vox": {"lufs_s": -20.0, "band_energy": {"sub": -20.0},
                "dynamic_spread_db": 15.0},
        "bass": {"lufs_s": -22.0, "band_energy": {"sub": -3.0},
                 "dynamic_spread_db": 4.0},
        "pads": {"lufs_s": -24.0, "band_energy": {"sub": -22.0},
                 "dynamic_spread_db": 14.0},
        # -4.0 puts it 17 dB over the pack median: past even the untagged window's
        # -9 edge by 8 dB, so the ±6 clamp still binds and the assertion below still
        # exercises it. (It was -10.0 when balance pulled everything to one median.)
        "loud": {"lufs_s": -4.0, "band_energy": {"sub": -60.0},
                 "dynamic_spread_db": 6.0},
    }
    maskings = [{"stem_a": "vox", "stem_b": "pads", "worst_band": "mid",
                 "worst_band_excess_db": 7.0}]
    bundles = build = automix.build_bundles(
        tags, stems, maskings, {"lufs_i": -20.0}, goal_lufs=-14.0)
    actions = [a for b in bundles for a in b["actions"]]
    by_type = {}
    for action in actions:
        by_type.setdefault(action["type"], []).append(action)

    # the outlier stem is trimmed back toward its class window, clamped to the guardrail
    (gain,) = by_type["gain"]
    assert gain["stem_id"] == "loud" and gain["delta_db"] == -6.0
    # HPF policy: vocal at 80, pad at 90; never the bass; never the
    # rumble-free stem
    hpfs = {a["stem_id"]: a["band"]["freq_hz"] for a in by_type["eq"]
            if a["band"]["shape"] == "hpf"}
    assert hpfs == {"vox": 80, "pads": 90}
    # corrective cut lands on the masker in the measured band
    (cut,) = [a for a in by_type["eq"] if a["band"]["shape"] == "peak"]
    assert cut["stem_id"] == "pads" and cut["band"]["gain_db"] == -4.0
    # vocal leveling comp on the dynamic source, but pads are exempt
    (comp,) = by_type["comp"]
    assert comp["stem_id"] == "vox"
    # threshold sits below the stem's OWN level (vox lufs_s -20.0), not at a fixed dBFS
    assert comp["threshold_db"] == -26.0 and comp["ratio"] == 2.5
    assert comp["attack_ms"] == 8.0 and comp["release_ms"] == 120.0
    comp_prediction = next(p for b in bundles for p in b["predicted_impact"]
                           if p["metric"] == "stem(vox).dynamic_spread_db")
    assert comp_prediction["predicted_delta"] == -2.0
    # the master limiter to the goal
    (limiter,) = by_type["limiter"]
    assert limiter["target_lufs_i"] == -14.0
    # every bundle obeys the engine guardrail shape
    assert all(1 <= len(b["actions"]) <= 8 for b in build)


def test_balance_leaves_a_correct_mix_alone():
    """The regression that started the 2026-08-05 rewrite.

    These are the MEASURED profiles of test_projects/suno_demo at its correct balance
    (a real 4-stem separation). The old median-pull read the 21 dB vocal-to-bass gap as
    an imbalance and "fixed" it with bass +6.0 / vocals -6.0, which put +12 dB of bass
    band energy into the render. A correct mix must produce no balance move at all."""
    tags = {"bass": ["bass"], "drums": ["cymbal"],
            "other": ["pad"], "vocals": ["vocal"]}
    stems = {
        "bass":   {"lufs_s": -33.9, "band_energy": {"sub": -7.4}},
        "drums":  {"lufs_s": -21.8, "band_energy": {"sub": -26.8}},
        "other":  {"lufs_s": -26.9, "band_energy": {"sub": -80.0}},
        "vocals": {"lufs_s": -12.8, "band_energy": {"sub": -46.4}},
    }
    bundles = automix.build_bundles(tags, stems, [], {"lufs_i": -12.2},
                                    goal_lufs=None)
    assert [a for b in bundles for a in b["actions"]
            if a["type"] == "gain"] == []


def test_balance_windows_are_class_relative():
    """Each class gets its own one-sided window: a bass may sit far under the pack,
    but not over it; a lead may sit over, but not far under; and a `full_mix` stem is
    not part of the pack at all (it IS the mix — it neither skews the median nor gets
    balanced against its own parts)."""
    tags = {"bass": ["bass"], "vox": ["vocal"], "whole": ["full_mix"],
            "gtr": [], "keys": [], "perc": []}
    stems = {
        "bass":  {"lufs_s": -12.0, "band_energy": {"sub": -3.0}},   # 8 over the pack
        "vox":   {"lufs_s": -32.0, "band_energy": {"sub": -60.0}},  # 12 under
        "whole": {"lufs_s": -60.0, "band_energy": {"sub": -60.0}},  # not in the pack
        "gtr":   {"lufs_s": -20.0, "band_energy": {"sub": -60.0}},
        "keys":  {"lufs_s": -20.0, "band_energy": {"sub": -60.0}},
        "perc":  {"lufs_s": -20.0, "band_energy": {"sub": -60.0}},
    }
    bundles = automix.build_bundles(tags, stems, [], {"lufs_i": -14.0},
                                    goal_lufs=None)
    gains = {a["stem_id"]: a["delta_db"] for b in bundles for a in b["actions"]
             if a["type"] == "gain"}
    assert gains == {"bass": -2.0, "vox": 3.0}   # pack median -20.0
    assert "whole" not in gains                  # excluded, and it didn't skew the median


def test_build_bundles_is_chain_aware():
    """A second Auto Mix run never stacks: stems already carrying an HPF or
    a compressor are skipped (2026-06 review — the double-comp bug)."""
    comp_entry = {"type": "comp", "threshold_db": -20.0, "ratio": 2.0,
                  "attack_ms": 10.0, "release_ms": 150.0, "makeup_db": 0.0}
    hpf_entry = {"type": "eq", "band": {"freq_hz": 80, "gain_db": 0.0,
                                        "q": 0.7, "shape": "hpf"}}
    tags = {"vox": ["vocal"], "drums": []}
    stems = {
        # would earn an HPF and a comp, but already has both
        "vox": {"lufs_s": -20.0, "band_energy": {"sub": -20.0},
                "dynamic_spread_db": 15.0,
                "applied_chain": [{"type": "gain", "gain_db": 0.0},
                                  hpf_entry, comp_entry]},
        # multiband comp counts as dynamics too; no HPF yet
        "drums": {"lufs_s": -20.0, "band_energy": {"sub": -20.0},
                  "dynamic_spread_db": 15.0,
                  "applied_chain": [{"type": "gain", "gain_db": 0.0},
                                    {"type": "mb_comp"}]},
    }
    bundles = automix.build_bundles(tags, stems, [], {"lufs_i": -20.0},
                                    goal_lufs=None)
    actions = [a for b in bundles for a in b["actions"]]
    assert not [a for a in actions if a["type"] in ("comp", "mb_comp")]
    hpfs = [a for a in actions
            if a["type"] == "eq" and a["band"]["shape"] == "hpf"]
    assert [a["stem_id"] for a in hpfs] == ["drums"]


def test_build_bundles_chooses_dynamics_intents():
    """Phase 34: Auto Mix uses the existing compressor with different
    intent-specific settings instead of one generic comp shape."""
    tags = {"vox": ["vocal"], "kick": ["kick"], "bass": ["bass"], "gtr": []}
    stems = {
        sid: {"lufs_s": -20.0, "band_energy": {"sub": -60.0},
              "dynamic_spread_db": 16.0,
              "transient_density": 0.1,
              "transient_energy_ratio": 0.05}
        for sid in tags
    }
    stems["kick"]["transient_density"] = 1.0
    stems["kick"]["transient_energy_ratio"] = 0.4
    bundles = automix.build_bundles(tags, stems, [], {"lufs_i": -18.0},
                                    goal_lufs=None)
    comps = {a["stem_id"]: a for b in bundles for a in b["actions"]
             if a["type"] == "comp"}

    # Thresholds are RELATIVE to each stem's own lufs_s (-20.0 here), so the ratio
    # acts on the loud parts instead of attenuating the whole stem, and the makeup
    # returns what the ratio takes there: vocal_level is 6 dB down at 2.5:1, so
    # -26.0 with 6 * (1 - 1/2.5) = 3.6 dB back.
    assert comps["vox"]["threshold_db"] == -26.0      # vocal level: lufs_s - 6
    assert comps["vox"]["makeup_db"] == 3.6
    assert comps["kick"]["threshold_db"] == -24.0     # peak control: lufs_s - 4
    assert comps["kick"]["attack_ms"] == 2.0
    assert comps["bass"]["attack_ms"] == 20.0         # bass density
    assert comps["gtr"]["mix"] == 0.8                 # parallel-ish density
    rationale = " ".join(b["rationale"] for b in bundles)
    # in the words the user reads under the move, not the profile's own name (QA 2026-10-08, D22)
    assert "vox: evens out the vocal" in rationale
    assert "kick: catches the loudest hits" in rationale
    assert "bass: firms up the bass" in rationale
    assert "gtr: adds gentle body" in rationale
    for raw in ("vocal_level", "peak_control", "bass_density"):
        assert raw not in rationale


def test_build_bundles_prefers_bus_glue_over_member_comps():
    """When Auto Mix creates a family bus, the glue comp lives on the bus;
    members don't also get individual stem comps in the same bootstrap."""
    tags = {"kik": ["kick"], "hat": ["cymbal"]}
    stems = {
        sid: {"lufs_s": -20.0, "band_energy": {"sub": -60.0},
              "dynamic_spread_db": 18.0,
              "transient_density": 1.0,
              "transient_energy_ratio": 0.4}
        for sid in tags
    }
    bundles = automix.build_bundles(tags, stems, [], {"lufs_i": -18.0},
                                    goal_lufs=None)
    comps = [a for b in bundles for a in b["actions"] if a["type"] == "comp"]
    assert [c["stem_id"] for c in comps] == ["drums"]
    assert comps[0]["mix"] == automix.GLUE_MIX


def test_build_bundles_uses_phrase_readout_for_vocal_rides_and_deess():
    tags = {"vox": ["vocal"], "gtr": []}
    stems = {
        "vox": {"lufs_s": -20.0, "band_energy": {"sub": -60.0},
                "dynamic_spread_db": 8.0, "applied_chain": []},
        "gtr": {"lufs_s": -20.0, "band_energy": {"sub": -60.0},
                "dynamic_spread_db": 8.0, "applied_chain": []},
    }
    phrases = {
        "vox": {
            "level_moves": [
                {"start_s": 1.0, "end_s": 1.8, "delta_db": 3.0,
                 "level_dbfs": -32.0, "reason": "too_quiet"},
            ],
            "sibilance_bursts": [
                {"start_s": 2.0, "end_s": 2.6, "density": 0.6,
                 "freq_hz": 7200, "threshold_db": -28.0},
            ],
        },
        # non-vocal phrase reports are ignored by the vocal control preset
        "gtr": {"level_moves": [{"start_s": 0.0, "end_s": 1.0,
                                 "delta_db": -3.0, "level_dbfs": -12.0,
                                 "reason": "too_loud"}],
                "sibilance_bursts": []},
    }

    bundles = automix.build_bundles(tags, stems, [], {"lufs_i": -18.0},
                                    goal_lufs=None, phrases=phrases)
    actions = [a for b in bundles for a in b["actions"]]
    ride = next(a for a in actions if a["type"] == "gain")
    deess = next(a for a in actions if a["type"] == "deess")

    assert ride == {"type": "gain", "stem_id": "vox", "delta_db": 2.0,
                    "time_range_s": [1.0, 1.8]}
    assert deess["stem_id"] == "vox"
    assert deess["time_range_s"] == [2.0, 2.6]
    assert deess["freq_hz"] == 7200
    assert not [a for a in actions if a.get("stem_id") == "gtr"]


def test_intent_presets_compile_to_existing_actions():
    tags = {"vox": ["vocal"], "kick": ["kick"], "pad": ["pad"],
            "mix": ["full_mix"]}
    stems = {
        "vox": {"lufs_s": -22.0, "band_energy": {"sub": -45.0},
                "dynamic_spread_db": 8.0, "sibilance_density": 0.3,
                "applied_chain": []},
        "kick": {"lufs_s": -18.0, "band_energy": {"sub": -8.0},
                 "dynamic_spread_db": 8.0, "transient_energy_ratio": 0.5,
                 "applied_chain": []},
        "pad": {"lufs_s": -20.0, "band_energy": {"sub": -50.0},
                "dynamic_spread_db": 8.0, "applied_chain": []},
        "mix": {"lufs_s": -19.0, "band_energy": {"sub": -50.0},
                "dynamic_spread_db": 8.0, "applied_chain": []},
    }

    cases = {
        "vocal-forward": ("gain", "vox"),
        "punchy-drums": ("transient", "kick"),
        "warm-master": ("eq", "master"),
        "wide-pop": ("width", "pad"),
        "podcast-clean": ("deess", "vox"),
    }
    for intent, (kind, target) in cases.items():
        bundles = automix.build_bundles(tags, stems, [], {"lufs_i": -18.0},
                                        goal_lufs=None, intent=intent)
        actions = [a for b in bundles for a in b["actions"]]
        assert any(a["type"] == kind and a["stem_id"] == target
                   for a in actions), intent
        assert any(f"auto mix preset {intent}" in b["rationale"]
                   for b in bundles), intent

    club = automix.build_bundles(
        tags, stems, [], {"lufs_i": -18.0},
        goal_lufs=automix._intent_goal_lufs("club-loud"), intent="club-loud")
    limiter = next(a for b in club for a in b["actions"]
                   if a["type"] == "limiter")
    assert limiter["target_lufs_i"] == -8.0
    assert automix.intent_from_brief("make it vocal forward") == "vocal-forward"


def test_intent_preset_rationale_survives_dynamics_loop():
    """Regression: the per-stem dynamics intent must not overwrite the Auto Mix
    preset name used in the preset + loudness rationales. Only fires when a
    stem's dynamic_spread_db crosses COMP_SPREAD_DB so the clarity loop runs —
    which is exactly the case the other intent test avoids."""
    tags = {"vox": ["vocal"], "synth": []}
    stems = {
        sid: {"lufs_s": -21.0, "band_energy": {"sub": -50.0},
              "dynamic_spread_db": 16.0, "applied_chain": []}
        for sid in tags
    }
    # a loudness preset (no preset overlay actions): only the master bundle's
    # "… for {intent}" tail can reveal the clobber.
    club = automix.build_bundles(
        tags, stems, [], {"lufs_i": -18.0},
        goal_lufs=automix._intent_goal_lufs("club-loud"), intent="club-loud")
    club_rationales = [b["rationale"] for b in club]
    assert any("for club-loud" in r for r in club_rationales)
    assert not any("for density" in r or "for vocal_level" in r or "for adds gentle body" in r
                   or "for evens out the vocal" in r for r in club_rationales)

    # a preset with overlay actions: its bundle must keep the real preset name.
    forward = automix.build_bundles(tags, stems, [], {"lufs_i": -18.0},
                                    goal_lufs=None, intent="vocal-forward")
    assert any("auto mix preset vocal-forward" in b["rationale"]
               for b in forward)
    assert not any("auto mix preset vocal_level" in b["rationale"]
                   or "auto mix preset density" in b["rationale"]
                   or "auto mix preset evens out the vocal" in b["rationale"]
                   or "auto mix preset adds gentle body" in b["rationale"]
                   for b in forward)


def test_build_bundles_skips_phrase_deess_when_stem_already_has_deess():
    tags = {"vox": ["vocal"]}
    stems = {"vox": {"lufs_s": -20.0, "band_energy": {"sub": -60.0},
                     "dynamic_spread_db": 8.0,
                     "applied_chain": [{"type": "deess"}]}}
    phrases = {"vox": {"level_moves": [],
                       "sibilance_bursts": [
                           {"start_s": 2.0, "end_s": 2.6, "density": 0.8,
                            "freq_hz": 7200, "threshold_db": -28.0}]}}

    bundles = automix.build_bundles(tags, stems, [], {"lufs_i": -18.0},
                                    goal_lufs=None, phrases=phrases)
    actions = [a for b in bundles for a in b["actions"]]
    assert not [a for a in actions if a["type"] == "deess"]


# ---------- Phase 28: busses + a corrective ride (pure policy) ----------

def _flat(stems):
    return {sid: {"lufs_s": -20.0, "band_energy": {"sub": -30.0},
                  "dynamic_spread_db": 5.0} for sid in stems}


def test_build_bundles_groups_busses():
    """Tag families of ≥2 become a bus with one parallel glue comp; loners and
    the bass stay loose (Phase 28 — the agent lays the obvious busses)."""
    tags = {"kik": ["kick"], "hat": ["cymbal"], "lead": ["vocal"],
            "bv": ["vocal"], "bassline": ["bass"]}
    bundles = automix.build_bundles(tags, _flat(tags), [], {"lufs_i": -18.0},
                                    goal_lufs=None)
    actions = [a for b in bundles for a in b["actions"]]

    groups = {a["bus_id"]: a["members"] for a in actions if a["type"] == "group"}
    assert groups == {"drums": ["hat", "kik"], "vox": ["bv", "lead"]}
    glue = {a["stem_id"]: a for a in actions
            if a["type"] == "comp" and a["stem_id"] in ("drums", "vox")}
    assert set(glue) == {"drums", "vox"}
    assert all(g["mix"] == 0.5 and g["ratio"] == 2.5 for g in glue.values())
    # the group precedes its glue comp in the same bundle (so the bus exists)
    drum_bundle = next(b for b in bundles
                       if any(a["type"] == "group" and a["bus_id"] == "drums"
                              for a in b["actions"]))
    types = [a.get("bus_id") or a.get("stem_id") for a in drum_bundle["actions"]]
    assert types.index("drums") == 0  # group drums is first


def test_build_bundles_rides_buried_lead():
    """One ranged ride lifts the lead in the single densest section it's in."""
    tags = {"lead": ["vocal"], "gtr": [], "keys": [], "bass": ["bass"]}
    sections = [
        {"label": "A", "start_s": 0.0, "end_s": 8.0, "lufs_i": -18.0,
         "band_energy": {}, "stem_activity": {"lead": 1.0, "gtr": 0.1,
                                              "keys": 0.0, "bass": 0.2}},
        {"label": "B", "start_s": 8.0, "end_s": 20.0, "lufs_i": -12.0,
         "band_energy": {}, "stem_activity": {"lead": 0.9, "gtr": 0.9,
                                              "keys": 0.8, "bass": 0.9}},
    ]
    bundles = automix.build_bundles(tags, _flat(tags), [], {"lufs_i": -18.0},
                                    goal_lufs=None, sections=sections)
    rides = [a for b in bundles for a in b["actions"]
             if a["type"] == "gain" and a.get("time_range_s")]
    assert len(rides) == 1
    (ride,) = rides
    assert ride["stem_id"] == "lead" and ride["delta_db"] == 1.5
    assert ride["time_range_s"] == [8.0, 20.0]  # the crowded section, not the sparse one


def test_build_bundles_skips_ride_when_lead_is_clear():
    """No ride when the lead is never crowded — Auto Mix touches automation,
    it doesn't invent rides."""
    tags = {"lead": ["vocal"], "gtr": []}
    sections = [{"label": "A", "start_s": 0.0, "end_s": 10.0, "lufs_i": -16.0,
                 "band_energy": {}, "stem_activity": {"lead": 1.0, "gtr": 0.9}}]
    bundles = automix.build_bundles(tags, _flat(tags), [], {"lufs_i": -18.0},
                                    goal_lufs=None, sections=sections)
    assert not [a for b in bundles for a in b["actions"]
                if a["type"] == "gain" and a.get("time_range_s")]


# ---------- websocket end to end (acceptance) ----------

def recv_until(ws, event_type: str, limit: int = 400) -> dict:
    for _ in range(limit):
        event = ws.receive_json()
        if event["type"] == "error":
            pytest.fail(f"gateway error: {event['message']}")
        if event["type"] == event_type:
            return event
    pytest.fail(f"no '{event_type}' event within {limit} messages")


def test_automix_end_to_end_over_websocket(demo_project):
    add_masker(demo_project)
    initial_excess = worst_masking_excess(demo_project)
    assert initial_excess > 3.0  # the fixture gives the flow real work

    app = create_app(demo_project, agent_factory=RuleAgent,
                     chat_factory=ScriptedChat)
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # snapshot
            ws.send_json({"type": "render"})  # the pre-automix ground truth
            recv_until(ws, "metrics")
            original = (demo_project / "bounces" / "master.wav").read_bytes()

            ws.send_json({"type": "auto_mix", "target": "master for streaming"})
            stages, applied = set(), 0
            for _ in range(400):
                event = ws.receive_json()
                if event["type"] == "error":
                    pytest.fail(event["message"])
                if event["type"] == "automix_stage":
                    stages.add(event["stage"])
                if (event["type"] == "proposal_status"
                        and event["status"] == "applied"):
                    applied += 1
                if event["type"] == "automix_finished":
                    break
            else:
                pytest.fail("auto mix never finished")
            assert stages >= {"tagging", "profiling", "building",
                              "applying", "rendering"}
            assert applied >= 2  # the chain + the limiter at minimum

            snapshot = recv_until(ws, "snapshot")
            metrics = snapshot["metrics"]
            assert metrics["lufs_i"] == pytest.approx(-14.0, abs=1.0)
            assert metrics["true_peak_dbtp"] <= -1.0
            assert snapshot["automix"]["undoable"]
            assert snapshot["prev_bounce_url"]  # A/B is live

            # the worst masking pair strictly improved
            assert worst_masking_excess(demo_project) < initial_excess

            # one click restores the untouched project bit-identically
            ws.send_json({"type": "undo_automix"})
            recv_until(ws, "metrics")
            snapshot = recv_until(ws, "snapshot")
            assert snapshot["automix"]["undoable"] == []
            restored = (demo_project / "bounces" / "master.wav").read_bytes()
            assert restored == original


# ---------- engine-level flow on the second demo project ----------

async def _noop(event: dict) -> None:
    pass


@pytest.mark.anyio
async def test_automix_three_section_engine_level(three_section_project):
    brief = "master to -16 LUFS"
    async with in_memory_engine(three_section_project) as client:
        engine = EngineClient(client)
        applied = await automix.bootstrap(engine, brief, emit=_noop)
        assert applied  # at least the limiter bundle landed
        report = await run_passes(engine, three_section_project, brief,
                                  RuleAgent(), max_passes=2,
                                  gate=policy_gate("auto"), emit=_noop)
        assert report.passes_run <= 2
        final = await engine.call("analyze_session")
    assert final["lufs_i"] == pytest.approx(-16.0, abs=1.0)
    assert final["true_peak_dbtp"] <= -1.0


# ---------- previous-bounce A/B ----------

@pytest.mark.anyio
async def test_prev_bounce_kept_on_every_full_render(demo_project):
    async with in_memory_engine(demo_project) as client:
        await call(client, "render_and_analyze")
        first = (demo_project / "bounces" / "master.wav").read_bytes()
        assert not (demo_project / "bounces" / "master.prev.wav").exists()

        proposal = await call(client, "propose_actions", {"req": {
            "rationale": "change something", "predicted_impact": [],
            "actions": [{"type": "gain", "stem_id": "stem_a",
                         "delta_db": 2.0}]}})
        await call(client, "apply_proposal",
                   {"proposal_id": proposal["proposal_id"]})
        await call(client, "render_and_analyze")

        prev = demo_project / "bounces" / "master.prev.wav"
        assert prev.read_bytes() == first  # the previous bounce, exactly
        # range renders never clobber the comparison file
        before = prev.read_bytes()
        await call(client, "render_and_analyze", {"range_s": [0.0, 2.0]})
        assert prev.read_bytes() == before


# ---------- intent matching is whole-word (2026-08-05 audit) ----------

# Same defect as targets._loudness_goal, in a second file: as a bare substring
# "warm" fired inside "s(warm) of synths" and "luke(warm) mix", and each of those
# laid a whole warm-master starting chain — master limiter included — over a mix
# nobody asked to master that way.
@pytest.mark.parametrize("brief", [
    "swarm of synths",
    "a lukewarm mix",
    "warmongering",
])
def test_a_word_containing_an_alias_is_not_an_intent(brief):
    from orchestrator.automix import intent_from_brief
    assert intent_from_brief(brief) is None


@pytest.mark.parametrize("brief,intent", [
    ("warm master", "warm-master"),
    ("make it warmer", "warm-master"),      # the morphological variants are still meant
    ("more warmth", "warm-master"),
    ("punchy drums", "punchy-drums"),
    ("club loud", "club-loud"),
    ("wide pop", "wide-pop"),
    ("podcast clean", "podcast-clean"),
    ("vocal forward", "vocal-forward"),
])
def test_the_real_presets_still_resolve(brief, intent):
    from orchestrator.automix import intent_from_brief
    assert intent_from_brief(brief) == intent


def test_a_multi_intent_brief_says_it_dropped_one(capsys):
    """Presets are single-valued, so one wins — but silently dropping the other is
    indistinguishable from the preset misfiring."""
    from orchestrator.automix import intent_from_brief

    assert intent_from_brief("warm master and a wide pop sound") == "warm-master"
    assert "not layered" in capsys.readouterr().err
