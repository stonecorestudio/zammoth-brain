"""Phase 15 (docs/05-roadmap.md): expanded palette.

Covers: match-EQ closing ≥ 90 % of every considered band gap in ≤ 2 passes,
ranged de-ess confined to its range (out-of-range samples bit-identical to
the dry render), ranged actions rejected on the master, per-band width
narrowing only its band, the multiband compressor taming one band and
refusing master makeup, deterministic renders + clean undo for the new
processors, and a mixed 44.1/48 kHz project opening and rendering.
"""
import json

import numpy as np
import pytest
import soundfile as sf

from mix_engine import analysis, dsp
from mix_engine.session import Session
from orchestrator.agents import RuleAgent
from orchestrator.loop import run_session
from orchestrator.targets import (REFERENCE_BAND_FLOOR_DB,
                                  REFERENCE_BAND_TOLERANCE_DB)

from test_phase1_acceptance import call
from test_phase7_reference import synthesize_reference
from test_phase10_efficiency import in_memory_engine

SAMPLE_RATE = 48000


# ---------- match-EQ acceptance ----------

def test_match_eq_closes_band_gaps_fast(demo_project):
    """Acceptance: ≥ 90 % closure of every considered initial band gap
    within 2 passes on the Phase 7 benchmark (the old 4-peak path needed
    ≥ 70 % within 5)."""
    synthesize_reference(demo_project)
    mix, sample_rate = dsp.render_master(Session(demo_project))
    ref_audio, _ = sf.read(demo_project / "refs" / "reference.wav",
                           always_2d=True)
    reference = analysis.analyze_reference(ref_audio, sample_rate, "ref")
    baseline = analysis.analyze_master(mix, sample_rate, stems=[], pass_number=1)
    considered = {band: value for band, value in reference.band_energy.items()
                  if value > REFERENCE_BAND_FLOOR_DB}
    initial = {band: abs(baseline.band_energy[band] - value)
               for band, value in considered.items()}

    report = run_session(demo_project, "match refs/reference.wav",
                         gating="auto", agent=RuleAgent(),
                         connect=in_memory_engine)
    assert report.stop_reason.startswith("targets met")
    assert report.passes_run <= 2, report.passes_run

    final_bands = report.final_metrics["band_energy"]
    for band, goal in considered.items():
        final_gap = abs(final_bands[band] - goal)
        if initial[band] >= 1.0:
            closure = 1.0 - final_gap / initial[band]
            assert closure >= 0.9, (band, initial[band], final_gap)
        else:
            assert final_gap <= REFERENCE_BAND_TOLERANCE_DB + 0.2, (band, final_gap)


# ---------- ranged processing ----------

def _sibilant_stem() -> np.ndarray:
    t = np.arange(int(8.0 * SAMPLE_RATE)) / SAMPLE_RATE
    period = 1.0 / 1.5
    envelope = np.exp(-(t % period) / 0.08)
    return (0.4 * np.sin(2 * np.pi * 800.0 * t)
            + 0.6 * envelope * np.sin(2 * np.pi * 7000.0 * t))


@pytest.mark.anyio
async def test_ranged_deess_bites_only_inside_the_range(tmp_path):
    project = tmp_path / "vocal"
    project.mkdir()
    sf.write(project / "vox.wav", _sibilant_stem().astype(np.float32),
             SAMPLE_RATE)
    dry, _ = dsp.render_stem(Session(project), "vox")

    async with in_memory_engine(project) as client:
        proposal = await call(client, "propose_actions", {"req": {
            "rationale": "de-ess the first half only", "predicted_impact": [],
            "actions": [{"type": "deess", "stem_id": "vox",
                         "threshold_db": -40.0, "freq_hz": 5000,
                         "time_range_s": [0.0, 4.0]}]}})
        assert proposal["accepted"], proposal["violations"]
        assert (await call(client, "apply_proposal",
                           {"proposal_id": proposal["proposal_id"]}))["ok"]

    wet, sample_rate = dsp.render_stem(Session(project), "vox")
    half = 4 * sample_rate
    fade = int(dsp.RIDE_FADE_S * sample_rate)
    # outside the range (past the fade) the signal is the dry render, bit
    # for bit; inside, sibilance measurably drops
    assert np.array_equal(wet[half + fade:], dry[half + fade:])
    assert not np.array_equal(wet[:half], dry[:half])
    sib_dry = analysis._sibilance_density(dry[:half], sample_rate)
    sib_wet = analysis._sibilance_density(wet[:half], sample_rate)
    assert sib_wet < sib_dry


@pytest.mark.anyio
async def test_ranged_actions_rejected_on_master(demo_project):
    async with in_memory_engine(demo_project) as client:
        for action in (
            {"type": "eq", "stem_id": "master",
             "band": {"freq_hz": 250, "gain_db": -3.0, "q": 1.0,
                      "shape": "peak"},
             "time_range_s": [0.0, 4.0]},
            {"type": "comp", "stem_id": "master", "threshold_db": -20.0,
             "ratio": 2.0, "attack_ms": 10.0, "release_ms": 100.0,
             "makeup_db": 0.0, "time_range_s": [0.0, 4.0]},
        ):
            result = await call(client, "propose_actions", {"req": {
                "rationale": "ranged master move", "predicted_impact": [],
                "actions": [action]}})
            assert not result["accepted"]
            assert result["violations"][0]["rule"] == "master_ride_unsupported"


# ---------- per-band width ----------

@pytest.mark.anyio
async def test_band_width_narrows_only_its_band(tmp_path):
    project = tmp_path / "wide"
    project.mkdir()
    t = np.arange(int(8.0 * SAMPLE_RATE)) / SAMPLE_RATE
    mono_low = 0.4 * np.sin(2 * np.pi * 220.0 * t)
    wide_high = 0.3 * np.sin(2 * np.pi * 6000.0 * t)
    audio = np.stack([mono_low + wide_high, mono_low - wide_high], axis=1)
    sf.write(project / "synth.wav", audio.astype(np.float32), SAMPLE_RATE)

    before, sample_rate = dsp.render_stem(Session(project), "synth")
    width_before = analysis._band_width_db(before, sample_rate)

    async with in_memory_engine(project) as client:
        proposal = await call(client, "propose_actions", {"req": {
            "rationale": "mono the brilliance band", "predicted_impact": [],
            "actions": [{"type": "band_width", "stem_id": "synth",
                         "band": "brilliance", "width": 0.0}]}})
        assert proposal["accepted"], proposal["violations"]
        await call(client, "apply_proposal",
                   {"proposal_id": proposal["proposal_id"]})

    after, _ = dsp.render_stem(Session(project), "synth")
    width_after = analysis._band_width_db(after, sample_rate)
    # the band's SIDE, measured directly (2026-10-21, the band width audit): the band is a 48 dB/oct
    # split, so a 6 kHz side tone in brilliance goes ~20.6 dB down, not to zero - and with no mid
    # in the band the side/mid width metric reads its ceiling whatever is left of the side.
    side = lambda a: float(np.sqrt(np.mean((0.5 * (a[sample_rate:, 0] - a[sample_rate:, 1])) ** 2)))
    assert 20.0 * np.log10(side(after) / side(before)) < -18.0
    assert width_after["low_mid"] == pytest.approx(width_before["low_mid"],
                                                   abs=0.5)


# ---------- multiband compressor ----------

@pytest.mark.anyio
async def test_multiband_comp_tames_one_band(tmp_path):
    project = tmp_path / "mb"
    project.mkdir()
    t = np.arange(int(8.0 * SAMPLE_RATE)) / SAMPLE_RATE
    audio = (0.6 * np.sin(2 * np.pi * 80.0 * t)      # loud low
             + 0.1 * np.sin(2 * np.pi * 5000.0 * t))  # quiet high
    sf.write(project / "synth.wav", audio.astype(np.float32), SAMPLE_RATE)

    async with in_memory_engine(project) as client:
        before = await call(client, "analyze_stem", {"stem_id": "synth"})
        gentle = {"threshold_db": 0.0, "ratio": 1.0}
        proposal = await call(client, "propose_actions", {"req": {
            "rationale": "tame the lows only", "predicted_impact": [],
            "actions": [{"type": "mb_comp", "stem_id": "synth",
                         "low": {"threshold_db": -30.0, "ratio": 8.0},
                         "mid": gentle, "high": gentle,
                         "attack_ms": 5.0, "release_ms": 100.0}]}})
        assert proposal["accepted"], proposal["violations"]
        await call(client, "apply_proposal",
                   {"proposal_id": proposal["proposal_id"]})
        after = await call(client, "analyze_stem", {"stem_id": "synth"})
        # band_energy is relative: compressing the lows shifts share to the
        # highs
        assert after["band_energy"]["bass"] < before["band_energy"]["bass"] - 2
        assert after["band_energy"]["brilliance"] > before["band_energy"]["brilliance"]


def test_multiband_crossover_moves_band_boundary():
    """User-movable crossovers (native MB-comp editor drags the band edges): a tone
    at 300 Hz sits in MID with the default 150/2000 split, but moving the low/mid
    crossover up to 500 Hz drops it into LOW — so a hard low-band setting now
    compresses it. The deliverable render (dsp._multiband_comp) honors the per-entry
    crossover, matching the native monitor."""
    sr = 48000
    t = np.arange(int(2.0 * sr)) / sr
    audio = (0.6 * np.sin(2 * np.pi * 300.0 * t)).astype(np.float32)[np.newaxis, :]
    hard = {"threshold_db": -40.0, "ratio": 8.0}
    gentle = {"threshold_db": 0.0, "ratio": 1.0}
    base = {"low": hard, "mid": gentle, "high": gentle,
            "attack_ms": 5.0, "release_ms": 100.0}
    rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
    out_default = dsp._multiband_comp(audio, sr, dict(base))            # 300 Hz in MID (gentle)
    out_moved = dsp._multiband_comp(audio, sr, dict(base, crossover_lo_hz=500.0))  # now in LOW (hard)
    assert rms(out_moved) < rms(out_default) - 0.05
    # and the clamp/order guard keeps the mid band non-empty for a degenerate request
    safe = dsp._multiband_comp(audio, sr, dict(base, crossover_lo_hz=3000.0,
                                               crossover_hi_hz=1000.0))
    assert np.all(np.isfinite(safe))


def test_multiband_upward_boosts_quiet_band():
    """Upward-compression mode (user co-edit `up`): a band BELOW its threshold is
    boosted (lifts quiet passages); downward mode leaves a quiet band untouched. The
    export (dsp._multiband_comp) honors per-band `up`, matching the native monitor."""
    sr = 48000
    t = np.arange(int(2.0 * sr)) / sr
    audio = (0.05 * np.sin(2 * np.pi * 80.0 * t)).astype(np.float32)[np.newaxis, :]   # quiet low
    band = {"threshold_db": -20.0, "ratio": 4.0}
    gentle = {"threshold_db": 0.0, "ratio": 1.0}
    base = {"mid": gentle, "high": gentle, "attack_ms": 10.0, "release_ms": 120.0}
    rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
    down = dsp._multiband_comp(audio, sr, dict(base, low=dict(band)))
    up = dsp._multiband_comp(audio, sr, dict(base, low={**band, "up": True}))
    assert abs(rms(down) - rms(audio)) < 1e-3       # downward comp leaves the quiet band alone
    assert rms(up) > rms(audio) + 0.005             # upward boosts it


def test_dynamic_eq_gain_sign_sets_direction():
    """dyn-EQ gain_db sign = direction: negative cuts a loud band, positive boosts it
    (user co-edit allows +; the agent action model stays a cut)."""
    sr = 48000
    t = np.arange(int(2.0 * sr)) / sr
    audio = (0.5 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)[np.newaxis, :]
    base = {"freq_hz": 1000.0, "q": 2.0, "threshold_db": -40.0, "ratio": 4.0,
            "attack_ms": 10.0, "release_ms": 120.0}
    rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
    assert rms(dsp._dynamic_eq(audio, sr, dict(base, gain_db=-12.0))) < rms(audio) - 0.02
    assert rms(dsp._dynamic_eq(audio, sr, dict(base, gain_db=12.0))) > rms(audio) + 0.02


@pytest.mark.anyio
async def test_multiband_master_makeup_rejected(demo_project):
    async with in_memory_engine(demo_project) as client:
        result = await call(client, "propose_actions", {"req": {
            "rationale": "sneaky loudness", "predicted_impact": [],
            "actions": [{"type": "mb_comp", "stem_id": "master",
                         "low": {"threshold_db": -30.0, "ratio": 4.0,
                                 "makeup_db": 6.0},
                         "mid": {"threshold_db": 0.0, "ratio": 1.0},
                         "high": {"threshold_db": 0.0, "ratio": 1.0},
                         "attack_ms": 5.0, "release_ms": 100.0}]}})
        assert not result["accepted"]
        assert result["violations"][0]["rule"] == "master_comp_makeup_max_db"


# ---------- determinism + undo for the new processors ----------

@pytest.mark.anyio
@pytest.mark.parametrize("actions", [
    [{"type": "match_eq", "stem_id": "master",
      "band_gains_db": {"low_mid": -3.0, "presence": 2.0}}],
    [{"type": "mb_comp", "stem_id": "stem_a",
      "low": {"threshold_db": -30.0, "ratio": 4.0},
      "mid": {"threshold_db": -30.0, "ratio": 2.0},
      "high": {"threshold_db": 0.0, "ratio": 1.0},
      "attack_ms": 5.0, "release_ms": 100.0}],
    [{"type": "eq", "stem_id": "stem_a",
      "band": {"freq_hz": 1000, "gain_db": -4.0, "q": 1.0, "shape": "peak"},
      "time_range_s": [2.0, 6.0]}],
])
async def test_new_processors_deterministic_and_undoable(demo_project, actions):
    async with in_memory_engine(demo_project) as client:
        baseline = await call(client, "render_and_analyze")
        bounce = demo_project / baseline["bounce_path"]
        baseline_bytes = bounce.read_bytes()

        proposal = await call(client, "propose_actions", {"req": {
            "rationale": "phase 15 processor", "predicted_impact": [],
            "actions": actions}})
        assert proposal["accepted"], proposal["violations"]
        await call(client, "apply_proposal",
                   {"proposal_id": proposal["proposal_id"]})
        first = await call(client, "render_and_analyze")
        changed = bounce.read_bytes()
        assert changed != baseline_bytes
        await call(client, "render_and_analyze")
        assert bounce.read_bytes() == changed  # deterministic

        await call(client, "undo_proposal",
                   {"proposal_id": proposal["proposal_id"]})
        await call(client, "render_and_analyze")
        assert bounce.read_bytes() == baseline_bytes  # bit-identical undo


# ---------- mixed sample rates ----------

@pytest.mark.anyio
async def test_mixed_sample_rate_project_opens_and_renders(tmp_path):
    project = tmp_path / "mixed"
    project.mkdir()
    for name, rate in (("tone44", 44100), ("tone48", 48000)):
        t = np.arange(int(4.0 * rate)) / rate
        sf.write(project / f"{name}.wav",
                 (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32),
                 rate)

    session = Session(project)
    assert session.project_sample_rate() == 48000  # never downsample

    async with in_memory_engine(project) as client:
        result = await call(client, "render_and_analyze")
        assert result["analysis"]["lufs_i"] is not None
        bounce = project / result["bounce_path"]
        info = sf.info(str(bounce))
        assert info.samplerate == 48000
        assert info.frames == 4 * 48000
        first = bounce.read_bytes()
        await call(client, "render_and_analyze")
        assert bounce.read_bytes() == first  # resampling is deterministic
