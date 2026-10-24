"""Phase 29 — stem separation + artifact cleanup.

The product separator is htdemucs-via-demucs-onnx (Spleeter kept as the legacy
``kind="onnx"`` path), but the weights are a large first-use download, so the
automated suite drives the deterministic, dependency-free ``FakeSeparator``
(no onnxruntime, no model file). Opt-in tests exercise the real models when
present (Spleeter: bundled file; demucs: MIX_AGENT_TEST_DEMUCS=1).
"""

import os

import numpy as np
import pytest
import soundfile as sf
from pydantic import ValidationError

from mix_engine import analysis, dsp, separation
from mix_engine.models import SpectralDenoiseAction
from mix_engine.separation import (
    FakeSeparator,
    ModelNotAvailable,
    SpleeterOnnxSeparator,
    get_separator,
    stem_names,
)
from mix_engine.session import SUPPORTED_ACTION_TYPES, Session

from orchestrator import automix

from test_phase1_acceptance import call
from test_phase10_efficiency import in_memory_engine

SR = 44100


def _mix(seconds: float = 1.0, channels: int = 2, seed: int = 0) -> np.ndarray:
    """A deterministic broadband stereo 'mix': a few tones + shaped noise so
    every frequency band carries energy."""
    rng = np.random.default_rng(seed)
    n = int(SR * seconds)
    t = np.arange(n) / SR
    base = (0.3 * np.sin(2 * np.pi * 80 * t)      # bass
            + 0.2 * np.sin(2 * np.pi * 440 * t)   # mid
            + 0.1 * np.sin(2 * np.pi * 6000 * t)) # highs
    audio = np.stack([base, base * 0.9], axis=1).astype(np.float32)
    audio += (0.02 * rng.standard_normal((n, channels))).astype(np.float32)
    return audio[:, :channels]


# --------------------------------------------------------------------------- #
# FakeSeparator — the plumbing the import boundary depends on                  #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("stems,expected", [(4, separation.FOUR_STEM),
                                            (2, separation.TWO_STEM)])
def test_fake_returns_canonical_named_stems(stems, expected):
    sep = FakeSeparator(stems)
    out = sep.separate(_mix(), SR)
    assert tuple(out) == expected
    for name in expected:
        assert out[name].shape == (SR, 2)
        assert out[name].dtype == np.float32


@pytest.mark.parametrize("stems", [2, 4])
def test_fake_is_deterministic(stems):
    audio = _mix(seed=7)
    a = FakeSeparator(stems).separate(audio, SR)
    b = FakeSeparator(stems).separate(audio, SR)
    for name in a:
        # bit-identical: separation must be reproducible (determinism note in
        # separation.py + Phase 29 acceptance).
        assert np.array_equal(a[name], b[name])


@pytest.mark.parametrize("stems", [2, 4])
def test_fake_stems_sum_back_to_input(stems):
    audio = _mix(seed=3)
    out = FakeSeparator(stems).separate(audio, SR)
    summed = np.sum(list(out.values()), axis=0)
    assert np.allclose(summed, audio, atol=1e-4)


def test_fake_handles_mono():
    mono = _mix(channels=1)
    out = FakeSeparator(4).separate(mono, SR)
    for name in separation.FOUR_STEM:
        assert out[name].shape == (SR, 1)


def test_stem_names_validates():
    assert stem_names(4) == separation.FOUR_STEM
    assert stem_names(2) == separation.TWO_STEM
    with pytest.raises(ValueError):
        stem_names(3)


def test_get_separator_fake_kind():
    assert isinstance(get_separator(4, kind="fake"), FakeSeparator)
    with pytest.raises(ValueError):
        get_separator(4, kind="bogus")


def test_get_separator_env_override(monkeypatch):
    monkeypatch.setenv("MIX_AGENT_SEPARATOR", "fake")
    assert isinstance(get_separator(2), FakeSeparator)


# --------------------------------------------------------------------------- #
# ONNX path — resolution + graceful absence (no onnxruntime/model needed)      #
# --------------------------------------------------------------------------- #

def test_onnx_separator_missing_model_raises_cleanly():
    # isolated_data_dir (conftest) points MIX_AGENT_DATA_DIR at an empty tmp dir,
    # and no model is bundled in tests -> construction fails before any inference.
    with pytest.raises(ModelNotAvailable):
        SpleeterOnnxSeparator(4)


def test_get_separator_onnx_without_model_raises():
    with pytest.raises(ModelNotAvailable):
        get_separator(4, kind="onnx")


# --------------------------------------------------------------------------- #
# Demucs path — the product default (no weights needed for these)              #
# --------------------------------------------------------------------------- #

def test_get_separator_default_is_demucs():
    pytest.importorskip("demucs_onnx")
    sep = get_separator(4)
    assert isinstance(sep, separation.DemucsOnnxSeparator)
    assert sep.names == separation.FOUR_STEM
    assert get_separator(2).names == separation.TWO_STEM


def test_demucs_download_failure_raises_model_not_available(monkeypatch):
    pytest.importorskip("demucs_onnx")
    from demucs_onnx import inference as dx

    def boom(*args, **kwargs):
        raise OSError("no network in tests")

    monkeypatch.setattr(dx, "download_single_model", boom)
    monkeypatch.setattr(dx, "download_stem_model", boom)
    sep = separation.DemucsOnnxSeparator(4)
    # The weight resolve happens inside separate() (off-loop in the gateway);
    # a failed download must surface as ModelNotAvailable -> "no_model", never
    # a hard crash.
    with pytest.raises(ModelNotAvailable):
        sep.separate(_mix(seconds=0.25), SR)


def test_demucs_real_model_round_trips_if_enabled(monkeypatch):
    """Opt-in (downloads ~166 MB on first run): MIX_AGENT_TEST_DEMUCS=1."""
    if not os.environ.get("MIX_AGENT_TEST_DEMUCS"):
        pytest.skip("set MIX_AGENT_TEST_DEMUCS=1 to run the real htdemucs round-trip")
    pytest.importorskip("demucs_onnx")
    # Use the real app-support cache (not the isolated tmp data dir) so the
    # weights persist across test runs.
    monkeypatch.delenv("MIX_AGENT_DATA_DIR", raising=False)
    sep = separation.DemucsOnnxSeparator(2)
    audio = _mix(seconds=2.0)
    out = sep.separate(audio, SR)
    assert tuple(out) == separation.TWO_STEM
    for stem in out.values():
        assert stem.shape == audio.shape and stem.dtype == np.float32
    # Demucs is a waveform-domain model: vocals + accompaniment ≈ the mix.
    summed = np.sum(list(out.values()), axis=0)
    assert np.corrcoef(summed.flatten(), audio.flatten())[0, 1] > 0.95
    # determinism (CPU provider, fixed chunking): same input -> identical stems
    again = sep.separate(audio, SR)
    assert all(np.array_equal(out[k], again[k]) for k in out)


# --------------------------------------------------------------------------- #
# Opt-in: the real bundled Spleeter ONNX model (skipped when absent)           #
# --------------------------------------------------------------------------- #

def test_real_model_round_trips_if_present():
    # The 2-stem Spleeter model ships in-package (gitignored); skip where it's
    # not present (fresh checkout / CI without the bundle).
    try:
        sep = SpleeterOnnxSeparator(2)
    except ModelNotAvailable:
        pytest.skip("Spleeter ONNX model not bundled in this environment")
    audio = _mix(seconds=2.0)
    out = sep.separate(audio, SR)
    assert tuple(out) == separation.TWO_STEM
    for stem in out.values():
        assert stem.shape == audio.shape and stem.dtype == np.float32
    # Spleeter's ratio mask sums to 1 across sources (the < 11 kHz bins it keeps),
    # so the stems reconstruct the input closely.
    summed = np.sum(list(out.values()), axis=0)
    assert np.corrcoef(summed.flatten(), audio.flatten())[0, 1] > 0.95
    # determinism (single-thread CPU): same input -> identical stems
    again = sep.separate(audio, SR)
    assert all(np.array_equal(out[k], again[k]) for k in out)


def test_stft_uses_torch_not_scipy_magnitude_scale():
    """Regression (2026-06-13): the UNet needs raw torch-convention STFT
    magnitudes. scipy.signal.stft normalizes by win.sum() (~2048x smaller),
    which starved the model and produced a thin/high-frequency/bleedy vocal.
    A unit sine's bin magnitude must be torch-scale (~win.sum()/2 ≈ 1024), not
    scipy-scale (~0.5)."""
    sr = separation.MODEL_SAMPLE_RATE
    t = np.arange(separation._STFT_FRAME * 4) / sr
    sine = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    Z = separation._stft(np.stack([sine, sine], axis=1))   # (2, 2049, T)
    assert float(np.abs(Z).max()) > 100.0, "STFT magnitude looks scipy-normalized"


# --------------------------------------------------------------------------- #
# Repair DSP — SpectralDenoiseAction (the "cleanup after" half)                #
# --------------------------------------------------------------------------- #

def _noisy_bursts(seconds: float = 3.0, seed: int = 1):
    """A time-varying signal (1 kHz tone bursts, 4 Hz on/off) over a steady
    broadband noise floor — matches a spectral noise gate's assumptions and
    leaves noise-only gaps to measure. Returns (mono stem, gap mask)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    on = np.sin(2 * np.pi * 4 * t) > 0
    signal = 0.5 * np.sin(2 * np.pi * 1000 * t) * on
    noise = 0.02 * rng.standard_normal(n)
    return (signal + noise).astype(np.float32), ~on


def test_spectral_denoise_is_a_supported_stem_action():
    assert "spectral_denoise" in SUPPORTED_ACTION_TYPES


@pytest.mark.anyio
async def test_spectral_denoise_reduces_noise_floor(tmp_path):
    project = tmp_path / "noisy"
    project.mkdir()
    stem, gap = _noisy_bursts()
    sf.write(project / "stem.wav", stem, SR)
    dry, _ = dsp.render_stem(Session(project), "stem")

    async with in_memory_engine(project) as client:
        proposal = await call(client, "propose_actions", {"req": {
            "rationale": "clean the separation noise floor", "predicted_impact": [],
            "actions": [{"type": "spectral_denoise", "stem_id": "stem",
                         "reduction_db": 18.0, "sensitivity_db": 12.0}]}})
        assert proposal["accepted"], proposal["violations"]
        assert (await call(client, "apply_proposal",
                           {"proposal_id": proposal["proposal_id"]}))["ok"]

    wet, _ = dsp.render_stem(Session(project), "stem")
    dry_rms = np.sqrt(np.mean(dry[gap, 0] ** 2))
    wet_rms = np.sqrt(np.mean(wet[gap, 0] ** 2))
    assert wet_rms < 0.7 * dry_rms, (dry_rms, wet_rms)


@pytest.mark.anyio
async def test_spectral_denoise_deterministic_and_undoable(demo_project):
    async with in_memory_engine(demo_project) as client:
        baseline = await call(client, "render_and_analyze")
        bounce = demo_project / baseline["bounce_path"]
        baseline_bytes = bounce.read_bytes()

        proposal = await call(client, "propose_actions", {"req": {
            "rationale": "repair de-noise", "predicted_impact": [],
            "actions": [{"type": "spectral_denoise", "stem_id": "stem_a",
                         "reduction_db": 9.0}]}})
        assert proposal["accepted"], proposal["violations"]
        await call(client, "apply_proposal",
                   {"proposal_id": proposal["proposal_id"]})
        await call(client, "render_and_analyze")
        changed = bounce.read_bytes()
        assert changed != baseline_bytes
        await call(client, "render_and_analyze")
        assert bounce.read_bytes() == changed  # deterministic

        await call(client, "undo_proposal",
                   {"proposal_id": proposal["proposal_id"]})
        await call(client, "render_and_analyze")
        assert bounce.read_bytes() == baseline_bytes  # bit-identical undo


@pytest.mark.anyio
async def test_spectral_denoise_refused_on_master(demo_project):
    async with in_memory_engine(demo_project) as client:
        proposal = await call(client, "propose_actions", {"req": {
            "rationale": "de-noise the whole mix (should be refused)",
            "predicted_impact": [],
            "actions": [{"type": "spectral_denoise", "stem_id": "master",
                         "reduction_db": 6.0}]}})
    assert not proposal["accepted"]
    assert any(v["rule"] == "master_action_type" for v in proposal["violations"])


def test_spectral_denoise_reduction_over_cap_rejected_by_schema():
    with pytest.raises(ValidationError):
        SpectralDenoiseAction(stem_id="stem_a", reduction_db=30.0)


# --------------------------------------------------------------------------- #
# Artifact detection — analyze_separation (the "double check")                 #
# --------------------------------------------------------------------------- #

def _by_id(report):
    return {a.stem_id: a for a in report.per_stem}


def test_separation_report_flags_hf_shimmer():
    n = int(3.0 * SR)
    t = np.arange(n) / SR
    on = (np.sin(2 * np.pi * 4 * t) > 0).astype(np.float64)  # 1 kHz tone bursts
    base = 0.5 * np.sin(2 * np.pi * 1000 * t) * on
    hf = 0.3 * np.sin(2 * np.pi * 9000 * t)
    clean = (base + hf * on).astype(np.float32)         # HF only during bursts
    shimmer = (base + hf * (1 - on)).astype(np.float32)  # HF only in the gaps
    rep = analysis.analyze_separation(
        {"clean": clean[:, None], "shimmer": shimmer[:, None]}, SR)
    by = _by_id(rep)
    assert by["shimmer"].hf_shimmer > by["clean"].hf_shimmer + 0.2


def test_separation_report_flags_spectral_holes():
    rng = np.random.default_rng(0)
    n = int(4.0 * SR)
    t = np.arange(n) / SR
    base = 0.02 * rng.standard_normal(n)         # keeps every block active
    tone = 0.5 * np.sin(2 * np.pi * 1000 * t)    # the content (mid) band
    block = int(0.4 * SR)
    gate = np.ones(n)
    for i in range(0, n // block, 2):
        gate[i * block:(i + 1) * block] = 0.0    # notch the mid band, every other block
    steady = (base + tone).astype(np.float32)
    notched = (base + tone * gate).astype(np.float32)
    rep = analysis.analyze_separation(
        {"steady": steady[:, None], "notched": notched[:, None]}, SR)
    by = _by_id(rep)
    assert by["notched"].spectral_hole_ratio > by["steady"].spectral_hole_ratio + 0.1


def test_separation_report_flags_transient_smear():
    rng = np.random.default_rng(0)
    n = int(3.0 * SR)
    clean = np.zeros(n)
    smeared = np.zeros(n)
    klen = int(0.06 * SR)
    pre = int(0.025 * SR)
    env = np.exp(-np.arange(klen) / (0.01 * SR))        # sharp attack, decay
    ramp = np.linspace(0.0, 0.4, pre)                   # pre-echo ramp into the attack
    for start in range(int(0.3 * SR), n - klen, int(0.35 * SR)):
        burst = env * rng.standard_normal(klen)
        clean[start:start + klen] += burst
        smeared[start:start + klen] += burst
        smeared[start - pre:start] += ramp * rng.standard_normal(pre)  # energy BEFORE onset
    rep = analysis.analyze_separation(
        {"clean": clean[:, None].astype(np.float32),
         "smeared": smeared[:, None].astype(np.float32)}, SR)
    by = _by_id(rep)
    assert by["clean"].transient_smear is not None
    assert by["smeared"].transient_smear is not None
    assert by["smeared"].transient_smear > by["clean"].transient_smear


def test_separation_report_flags_leakage():
    rng = np.random.default_rng(0)
    n = int(5.0 * SR)
    t = np.arange(n) / SR
    first = (t < 2.5).astype(float)
    second = (t >= 2.5).astype(float)
    ymod = 0.5 + 0.5 * np.sin(2 * np.pi * 1.5 * t)       # Y's loud-half envelope
    Y = (0.5 * np.sin(2 * np.pi * 500 * t) * second * ymod).astype(np.float32)
    base_x = 0.5 * np.sin(2 * np.pi * 1000 * t) * first
    x_bleed = (base_x + 0.1 * np.sin(2 * np.pi * 500 * t) * second * ymod).astype(np.float32)
    indep = 0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t + 1.0)  # uncorrelated modulation
    x_clean = (base_x + 0.05 * rng.standard_normal(n) * second * indep).astype(np.float32)
    lk_bleed = _by_id(analysis.analyze_separation(
        {"x": x_bleed[:, None], "y": Y[:, None]}, SR))["x"].leakage
    lk_clean = _by_id(analysis.analyze_separation(
        {"x": x_clean[:, None], "y": Y[:, None]}, SR))["x"].leakage
    assert lk_bleed is not None and lk_clean is not None
    assert lk_bleed > lk_clean


def test_separation_report_lufs_spread():
    n = int(3.0 * SR)
    t = np.arange(n) / SR
    loud = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    quiet = (0.05 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    rep = analysis.analyze_separation(
        {"loud": loud[:, None], "quiet": quiet[:, None]}, SR)
    assert rep.inter_stem_lufs_spread_db is not None
    assert rep.inter_stem_lufs_spread_db > 15.0


@pytest.mark.anyio
async def test_analyze_separation_tool_smoke(demo_project):
    async with in_memory_engine(demo_project) as client:
        untouched = await call(client, "analyze_separation")
        # only stems a split made are measured (batch Py3, E5·R7): say these two were
        await call(client, "refresh_stems", {"separated": ["stem_a.wav", "stem_b.wav"]})
        rep = await call(client, "analyze_separation")
    assert untouched["per_stem"] == [] and untouched["not_separated"] == ["stem_a", "stem_b"]
    by = {s["stem_id"]: s for s in rep["per_stem"]}
    assert set(by) == {"stem_a", "stem_b"}
    for s in by.values():
        assert {"hf_shimmer", "transient_smear", "spectral_hole_ratio",
                "leakage", "lufs_s"} <= set(s)


# --------------------------------------------------------------------------- #
# AI cleanup preset — build_cleanup_bundles policy (detect → repair)           #
# --------------------------------------------------------------------------- #

def _report(per_stem, spread=None):
    return {"per_stem": per_stem, "inter_stem_lufs_spread_db": spread}


def _art(stem_id, shimmer=0.0, leakage=0.0, smear=0.0, holes=0.0,
         lufs_s=-14.0, silent=False):
    return {"stem_id": stem_id, "hf_shimmer": shimmer, "leakage": leakage,
            "transient_smear": smear, "spectral_hole_ratio": holes,
            "lufs_s": lufs_s, "silent": silent}


def _actions(bundles):
    return [a for b in bundles for a in b["actions"]]


def test_cleanup_preset_denoises_shimmer_and_leakage():
    rep = _report([_art("vox", shimmer=0.6), _art("drums", leakage=0.7),
                   _art("bass", shimmer=0.05, leakage=0.05)])
    denoised = {a["stem_id"] for a in _actions(automix.build_cleanup_bundles(rep))
                if a["type"] == "spectral_denoise"}
    assert denoised == {"vox", "drums"}


def test_cleanup_preset_restores_smeared_transients_but_not_pads():
    rep = _report([_art("drums", smear=0.5), _art("pad", smear=0.5),
                   _art("vox", smear=0.1)])
    actions = _actions(automix.build_cleanup_bundles(rep, {"pad": ["pad"]}))
    restored = {a["stem_id"] for a in actions if a["type"] == "transient"}
    assert restored == {"drums"}          # pad excluded by tag, vox below threshold
    assert all(a["attack_db"] > 0 for a in actions if a["type"] == "transient")


def test_cleanup_never_touches_relative_loudness():
    """A wide inter-stem loudness spread is NOT a defect (2026-08-05).

    The preset used to gain-match stems whose solo loudness spread crossed 6 dB. But
    separated stems sum back to the source track, so their spread IS the source's own
    balance — measured on the real suno separation it is 21.1 dB, on a mix that is
    correct. Cleanup repairs separation ARTIFACTS; it has no opinion on balance."""
    rep = _report([_art("loud", lufs_s=-8.0), _art("quiet", lufs_s=-20.0)],
                  spread=12.0)
    assert automix.build_cleanup_bundles(rep) == []


def test_cleanup_v2_repairs_holes_glues_and_explains_near_misses():
    rep = _report([
        _art("vox", shimmer=0.6),
        _art("drums", holes=0.5),
        _art("bass", leakage=0.34),
    ])
    bundles = automix.build_cleanup_bundles(rep)
    actions = _actions(bundles)

    assert any(a["type"] == "saturation" and a["stem_id"] == "drums"
               for a in actions)
    group = next(a for a in actions if a["type"] == "group")
    assert group["bus_id"] == "cleanup_glue"
    assert group["members"] == ["drums", "vox"]
    glue = next(a for a in actions if a["type"] == "comp"
                and a["stem_id"] == "cleanup_glue")
    assert glue["mix"] == automix.CLEANUP_GLUE_MIX
    assert any("Watched but left alone" in b["rationale"] for b in bundles)


def test_cleanup_preset_noop_when_clean():
    rep = _report([_art("a", shimmer=0.05, leakage=0.05, smear=0.05, lufs_s=-14.0),
                   _art("b", shimmer=0.05, leakage=0.05, smear=0.05, lufs_s=-14.3)],
                  spread=0.3)
    assert automix.build_cleanup_bundles(rep) == []


def test_cleanup_preset_skips_silent_stems():
    rep = _report([{"stem_id": "gone", "silent": True}, _art("vox", shimmer=0.6)])
    actions = _actions(automix.build_cleanup_bundles(rep))
    assert {a["stem_id"] for a in actions} == {"vox"}


@pytest.mark.anyio
async def test_cleanup_pipeline_detects_repairs_and_undoes(tmp_path):
    """End-to-end (engine level): analyze_separation measures HF shimmer, the preset
    proposes a spectral de-noise, applying it narrows the measured shimmer, the render
    stays under the true-peak ceiling, and undo restores it.

    Keyed off shimmer since 2026-08-05 — it used to key off a wide inter-stem loudness
    spread and assert the gain-match narrowed it, but that repair was removed: a
    separation's loudness spread is the source's balance, not an artifact."""
    project = tmp_path / "split"
    project.mkdir()
    n = int(3.0 * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(7)   # deterministic musical-noise shimmer
    # Levels kept low so the two stems SUM under the -1.0 dBTP ceiling: cleanup is
    # repair, it has no limiter, so the ceiling assertion below is "the repair didn't
    # push it over", not "something pulled it back".
    noisy = (0.12 * np.sin(2 * np.pi * 220 * t)
             + 0.12 * rng.standard_normal(n) * (rng.random(n) < 0.5))
    sf.write(project / "noisy.wav", noisy.astype(np.float32), SR)
    sf.write(project / "clean.wav", (0.15 * np.sin(2 * np.pi * 660 * t)).astype(np.float32), SR)

    async with in_memory_engine(project) as client:
        await call(client, "refresh_stems", {"separated": ["noisy.wav", "clean.wav"]})   # a split's
        before = await call(client, "analyze_separation")
        shimmer_before = max(a["hf_shimmer"] for a in before["per_stem"])
        assert shimmer_before >= automix.SHIMMER_DENOISE_AT        # flagged

        tags = await call(client, "auto_tag_stems")
        bundles = automix.build_cleanup_bundles(before, tags)
        assert bundles                                            # repair proposed

        applied = []
        for bundle in bundles:
            result = await call(client, "propose_actions", {"req": bundle})
            assert result["accepted"], result["violations"]
            assert (await call(client, "apply_proposal",
                               {"proposal_id": result["proposal_id"]}))["ok"]
            applied.append(result["proposal_id"])

        after = await call(client, "analyze_separation")
        assert max(a["hf_shimmer"] for a in after["per_stem"]) < shimmer_before
        render = await call(client, "render_and_analyze")
        assert render["analysis"]["true_peak_dbtp"] <= -1.0        # ceiling held

        for proposal_id in reversed(applied):
            assert (await call(client, "undo_proposal",
                               {"proposal_id": proposal_id}))["ok"]
        restored = await call(client, "analyze_separation")
        assert abs(max(a["hf_shimmer"] for a in restored["per_stem"])
                   - shimmer_before) < 0.05


# --------------------------------------------------------------------------- #
# Gateway separate_track — the user-only split endpoint (websocket)            #
# --------------------------------------------------------------------------- #

def _gateway_app(project):
    from orchestrator.agents import RuleAgent
    from orchestrator.chat import ScriptedChat
    from orchestrator.gateway import create_app
    return create_app(project, agent_factory=RuleAgent, chat_factory=ScriptedChat)


def test_separate_track_over_websocket_registers_stems(demo_project, tmp_path):
    from starlette.testclient import TestClient
    n = int(2.0 * SR)
    t = np.arange(n) / SR
    mix = (0.4 * np.sin(2 * np.pi * 220 * t)
           + 0.2 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)
    track = tmp_path / "track.wav"
    sf.write(track, mix, SR)

    with TestClient(_gateway_app(demo_project)) as client:
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # initial snapshot
            ws.send_json({"type": "separate_track", "path": str(track),
                          "stems": 4, "separator": "fake"})
            stages = set()
            for _ in range(400):
                ev = ws.receive_json()
                if ev["type"] == "error":
                    pytest.fail(ev["message"])
                if ev["type"] == "separation_stage":
                    stages.add(ev["stage"])
                if ev["type"] == "separation_finished":
                    assert ev["stop_reason"] == "done"
                    assert set(ev["stems"]) == {"track.vocals", "track.drums",
                                                "track.bass", "track.other"}
                    break
            else:
                pytest.fail("separation never finished")
            # Regression: the post-split snapshot must carry the separation
            # events in its feed — they're _emit'd (persisted), not just
            # broadcast; a bare broadcast gets wiped when the snapshot replaces
            # the client feed.
            snapshot = None
            for _ in range(50):
                ev = ws.receive_json()
                if ev["type"] == "snapshot":
                    snapshot = ev
                    break
            assert snapshot is not None, "no snapshot after split"
            assert any(e.get("type") == "separation_finished"
                       for e in snapshot["feed"]), "split events wiped from feed"
    assert {"loading", "separating", "writing"} <= stages
    for name in ("vocals", "drums", "bass", "other"):
        assert (demo_project / f"track.{name}.wav").exists()


def test_separate_track_without_model_reports_cleanly(demo_project, tmp_path):
    """With no bundled model the onnx path degrades to a clear message, not a
    crash — documents the current state until the Spleeter model is bundled."""
    from starlette.testclient import TestClient
    track = tmp_path / "track.wav"
    sf.write(track, (0.3 * np.sin(2 * np.pi * 220 * np.arange(SR) / SR)).astype(np.float32), SR)

    with TestClient(_gateway_app(demo_project)) as client:
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "separate_track", "path": str(track),
                          "stems": 4, "separator": "onnx"})
            saw_error = False
            for _ in range(400):
                ev = ws.receive_json()
                if ev["type"] == "error" and "unavailable" in ev["message"]:
                    saw_error = True
                if ev["type"] == "separation_finished":
                    assert ev["stop_reason"] == "no_model"
                    break
            else:
                pytest.fail("separation never finished")
    assert saw_error
    assert not (demo_project / "track.vocals.wav").exists()


@pytest.mark.anyio
async def test_suno_demo_flags_artifacts_and_proposes_cleanup(suno_project):
    """The realistic AI-stem demo: analyze_separation flags its wide loudness
    spread and the AI-cleanup preset has real repair to propose."""
    async with in_memory_engine(suno_project) as client:
        await call(client, "refresh_stems", {"separated": [
            "vocals.wav", "drums.wav", "bass.wav", "other.wav"]})     # the demo IS a split
        report = await call(client, "analyze_separation")
        tags = await call(client, "auto_tag_stems")
    assert {a["stem_id"] for a in report["per_stem"]} == {
        "vocals", "drums", "bass", "other"}
    # The spread is a descriptive readout, not a defect — the stems sum to the source,
    # so 21 dB between a lead vocal and a bass is what a correct balance looks like.
    assert report["inter_stem_lufs_spread_db"] is not None
    assert automix.build_cleanup_bundles(report, tags)  # real repair to propose
    assert not [a for b in automix.build_cleanup_bundles(report, tags)
                for a in b["actions"] if a["type"] == "gain"]  # ...but no balancing


@pytest.mark.anyio
async def test_separation_is_not_an_agent_tool(demo_project):
    """Hard invariant: separation is a user import action. The engine exposes
    the read-only analyze_separation 'double check', but NO separation mutation
    tool the agent could call."""
    async with in_memory_engine(demo_project) as client:
        listing = await client.list_tools()
    names = {tool.name for tool in listing.tools}
    assert "analyze_separation" in names                 # read-only is fine
    assert not any(n.startswith("separate") for n in names)
    assert "separate_track" not in names
