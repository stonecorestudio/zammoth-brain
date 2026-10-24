"""Phase 20 — deeper ears: resonances, GR telemetry, tempo, spatial masking
(docs/05-roadmap.md Phase 20; definitions in the mix-analysis skill, v6)."""
from pathlib import Path

import numpy as np
import pytest

from mix_engine import analysis, dsp
from mix_engine.session import Session

SR = 48000


def _mono(x: np.ndarray) -> np.ndarray:
    return x[:, np.newaxis]


# ---------- 1. resonance detection ----------

def test_resonance_detects_narrow_tone():
    t = np.arange(SR * 5) / SR
    noise = 0.05 * np.random.default_rng(3).standard_normal(t.shape)
    tone = 0.1 * np.sin(2 * np.pi * 3200.0 * t)
    found = analysis._resonances(_mono(noise + tone), SR)
    assert found is not None
    top = found[0]
    assert abs(top.freq_hz - 3200) <= 10  # one PSD bin of slack
    assert top.prominence_db >= analysis.RESONANCE_MIN_PROMINENCE_DB
    assert 0 < top.q <= analysis.RESONANCE_Q_CAP


def test_resonance_silent_on_flat_noise():
    noise = 0.05 * np.random.default_rng(4).standard_normal(SR * 5)
    assert analysis._resonances(_mono(noise), SR) is None


# ---------- 3. gain-reduction telemetry ----------

COMP = {"type": "comp", "threshold_db": -45.0, "ratio": 8.0,
        "attack_ms": 5.0, "release_ms": 100.0, "makeup_db": 0.0}


def test_comp_reports_gain_reduction(demo_project: Path):
    session = Session(demo_project)
    session.stems["stem_b"]["chain"].append(dict(COMP))
    telemetry: list = []
    dsp.render_stem(session, "stem_b", telemetry=telemetry)
    (stats,) = telemetry
    assert stats["processor"] == "comp"
    assert stats["gr_mean_db"] > 2.0           # noise bed well over threshold
    assert stats["gr_max_db"] >= stats["gr_mean_db"]
    assert 0.0 <= stats["gr_active_pct"] <= 100.0


def test_telemetry_excludes_makeup(demo_project: Path):
    session = Session(demo_project)
    session.stems["stem_b"]["chain"].append(dict(COMP))
    plain: list = []
    dsp.render_stem(session, "stem_b", telemetry=plain)

    session_made_up = Session(demo_project)
    session_made_up.stems["stem_b"]["chain"].append(
        dict(COMP, makeup_db=6.0))
    made_up: list = []
    dsp.render_stem(session_made_up, "stem_b", telemetry=made_up)
    # telemetry is reduction, not net level: makeup must not hide GR
    assert abs(made_up[0]["gr_mean_db"] - plain[0]["gr_mean_db"]) <= 0.2


def test_telemetry_is_pure_observation(demo_project: Path):
    session = Session(demo_project)
    session.stems["stem_b"]["chain"].append(dict(COMP, makeup_db=3.0))
    silent, _ = dsp.render_stem(session, "stem_b")
    observed, _ = dsp.render_stem(session, "stem_b", telemetry=[])
    assert np.array_equal(silent, observed)


def test_limiter_telemetry_reaches_session_analysis(demo_project: Path):
    session = Session(demo_project)
    session._state["master_chain"].append(
        # −5, not −8: the all-mono demo master is measured as it plays (dual-mono, analysis v7),
        # 3 dB louder than the one-channel reading these targets were written against
        {"type": "limiter", "ceiling_dbtp": -1.0, "target_lufs_i": -5.0})
    telemetry: list = []
    audio, sample_rate = dsp.render_master(session, telemetry=telemetry)
    result = analysis.analyze_master(audio, sample_rate,
                                     stems=sorted(session.stems),
                                     pass_number=1,
                                     master_dynamics=telemetry or None)
    assert result.master_dynamics is not None
    (limiter,) = [d for d in result.master_dynamics
                  if d.processor == "limiter"]
    assert limiter.gr_max_db > 0.0  # pushing to -5 LUFS engages the brickwall


def test_clean_chain_reports_no_telemetry(demo_project: Path):
    session = Session(demo_project)
    telemetry: list = []
    dsp.render_stem(session, "stem_a", telemetry=telemetry)
    assert telemetry == []


# ---------- 4. tempo + musical time ----------

def _click_track(bpm: float, duration_s: float, offset_s: float = 0.0,
                 sr: int = SR) -> np.ndarray:
    """Deterministic click track: 10 ms 1 kHz blips on every beat."""
    audio = np.zeros(int(sr * duration_s))
    blip_t = np.arange(int(0.010 * sr)) / sr
    blip = 0.5 * np.sin(2 * np.pi * 1000.0 * blip_t) * np.hanning(blip_t.size)
    step = 60.0 / bpm
    beat = offset_s
    while beat < duration_s:
        start = int(beat * sr)
        end = min(start + blip.size, audio.size)
        audio[start:end] += blip[: end - start]
        beat += step
    return audio


def test_tempo_measures_a_click_track():
    tempo = analysis._tempo(_click_track(120.0, 10.0, offset_s=0.25), SR)
    assert tempo is not None
    bpm, confidence, beat_offset_s = tempo
    assert abs(bpm - 120.0) <= 2.0
    assert confidence >= analysis.TEMPO_MIN_CONFIDENCE
    assert abs(beat_offset_s - 0.25) <= 0.05


def test_tempo_none_on_unrhythmic_audio():
    noise = 0.1 * np.random.default_rng(5).standard_normal(SR * 8)
    assert analysis._tempo(noise, SR) is None


def test_sections_carry_bars_when_grid_is_confident():
    clicks = _click_track(120.0, 10.0)
    result = analysis.analyze_sections(_mono(clicks), SR,
                                       {"clicks": _mono(clicks)})
    assert result.tempo_bpm is not None and abs(result.tempo_bpm - 120.0) <= 2
    first = result.sections[0]
    assert first.start_bar is not None and first.end_bar is not None
    # 10 s at 120 BPM = 5 bars; the annotation should land near that
    assert abs(first.end_bar - first.start_bar - 5.0) <= 0.5


def test_sections_carry_no_bars_without_a_grid(demo_project: Path):
    session = Session(demo_project)
    audio, sample_rate = dsp.render_master(session)
    stems = {sid: dsp.render_stem(session, sid)[0] for sid in session.stems}
    result = analysis.analyze_sections(audio, sample_rate, stems)
    assert result.tempo_bpm is None
    assert all(s.start_bar is None for s in result.sections)


# ---------- 5. spatial masking ----------

def _stereo(mono: np.ndarray, pan: float) -> np.ndarray:
    """Balance-law stereo placement, matching dsp._pan."""
    out = np.tile(mono[:, np.newaxis], (1, 2))
    out[:, 0] *= min(1.0, 1.0 - pan)
    out[:, 1] *= min(1.0, 1.0 + pan)
    return out


def test_panned_apart_stems_show_spatial_release():
    t = np.arange(SR * 4) / SR
    voice = 0.2 * np.sin(2 * np.pi * 500.0 * t)
    rng = np.random.default_rng(6)
    masker = 0.4 * np.convolve(rng.standard_normal(t.shape),
                               np.ones(8) / 8.0, mode="same")

    apart = analysis.analyze_masking(_stereo(voice, -0.9), _stereo(masker, 0.9),
                                     SR, stem_a="voice", stem_b="masker")
    together = analysis.analyze_masking(_stereo(voice, 0.0), _stereo(masker, 0.0),
                                        SR, stem_a="voice", stem_b="masker")
    assert apart.spatial_release_db is not None
    assert apart.spatial_release_db > 3.0       # the left ear is much clearer
    assert together.spatial_release_db is not None
    assert together.spatial_release_db <= 0.5   # co-panned: no spatial help
    assert set(apart.per_channel_worst_excess_db) <= {"left", "right"}


def test_mono_stems_have_no_spatial_fields():
    t = np.arange(SR * 4) / SR
    voice = 0.2 * np.sin(2 * np.pi * 500.0 * t)
    noise = 0.2 * np.random.default_rng(7).standard_normal(t.shape)
    result = analysis.analyze_masking(_mono(voice), _mono(noise), SR,
                                      stem_a="voice", stem_b="noise")
    assert result.per_channel_worst_excess_db is None
    assert result.spatial_release_db is None


def test_spatial_release_is_probe_measurable(demo_project: Path):
    """A pan move must register in masking(a,b).spatial_release_db — the
    whole point: separation the mono metric can't see (Phase 20)."""
    from mix_engine import impact
    from mix_engine.models import PanAction

    session = Session(demo_project)
    metric = "masking(stem_a,stem_b).spatial_release_db"
    measurements = impact.probe(
        session,
        [PanAction(stem_id="stem_a", pan=-0.9),
         PanAction(stem_id="stem_b", pan=0.9)],
        [metric])
    (m,) = measurements
    assert m.after is not None
    assert m.after > 1.0  # panning apart creates measurable release


def test_resonances_ride_in_stem_analysis(demo_project: Path):
    """v9 (reader E5·R11): this test used to require that the demo's 440 Hz sine stem "must
    report a resonance" - which was the defect: that tone is the stem's NOTE. A stem with a
    ring that is not a pitch it plays carries it; the sine stem carries none."""
    import soundfile as sf
    t = np.arange(SR * 5) / SR
    ring = 0.05 * np.random.default_rng(3).standard_normal(t.shape) + 0.1 * np.sin(2 * np.pi * 3200.0 * t)
    sf.write(str(demo_project / "ringy.wav"), ring.astype(np.float32), SR, subtype="FLOAT")
    session = Session(demo_project)
    audio, sample_rate = dsp.render_stem(session, "ringy")
    result = analysis.analyze_stem(audio, sample_rate, stem_id="ringy",
                                   applied_chain=session.applied_chain("ringy"))
    assert result.resonances, "the ring must ride in the stem's analysis"
    assert abs(result.resonances[0].freq_hz - 3200) <= 10
    audio, sample_rate = dsp.render_stem(session, "stem_a")
    held = analysis.analyze_stem(audio, sample_rate, stem_id="stem_a",
                                 applied_chain=session.applied_chain("stem_a"))
    assert held.resonances is None, "the 440 Hz sine is the stem's note, not a resonance"
    # contract: the version that defines these fields
    assert result.analysis_version >= 9
