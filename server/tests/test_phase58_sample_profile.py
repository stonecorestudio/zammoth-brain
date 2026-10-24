"""Phase 58: `sample_profile` - one sound, measured - and `analyze_sample`, and the brain
attaching each rendered file's profile to the result of the command that made it.

Every signal here is made in the test and every answer is known beforehand: the profile is a
meter of this repo's own, and a meter is run on a known control before anything rests on it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import anyio
import numpy as np
import pyloudnorm as pyln
import pytest
import soundfile as sf
from mcp.shared.memory import create_connected_server_and_client_session

from mix_engine.mcp_server import create_server
from mix_engine.models import BAND_LABELS, SampleProfile
from mix_engine.sample_profile import sample_profile
from test_phase1_acceptance import call

REPO = Path(__file__).resolve().parents[2]
SR = 44100


def _t(seconds: float, sr: int = SR) -> np.ndarray:
    return np.arange(int(seconds * sr)) / sr


def _kick(sr: int = SR) -> np.ndarray:
    """A sine that falls from 200 Hz to 50 Hz with a 40 ms time constant under a 150 ms decay."""
    t = _t(0.6, sr)
    phase = 2 * np.pi * np.cumsum(50 + 150 * np.exp(-t / 0.040)) / sr
    return 0.9 * np.sin(phase) * np.exp(-t / 0.15)


def _tone(hz: float, seconds: float = 1.0, amp: float = 0.5, sr: int = SR) -> np.ndarray:
    t = _t(seconds, sr)
    return amp * np.sin(2 * np.pi * hz * t) * np.minimum(1.0, t / 0.01)


def _cents(a: float, b: float) -> float:
    return abs(1200.0 * np.log2(a / b))


# ---------------------------------------------------------------- how it starts and ends

def test_a_kick_reads_as_a_kick():
    p = sample_profile(_kick(), SR)
    assert p.attack_ms < 3.0                          # instant
    assert 320.0 < p.decay_ms < 370.0                 # 20 dB down a 150 ms decay: 345 ms
    assert p.leading_silence_ms == 0.0
    assert 45.0 < p.pitch_hz < 58.0                   # where it settles
    assert 140.0 < p.pitch_start_hz < 210.0           # where it started (200, read a frame in)
    assert p.pitch_drop_semitones > 17.0
    assert 110.0 < p.pitch_settle_ms < 190.0          # within a semitone of 50 Hz: 156 ms
    assert p.noisiness < 0.2
    assert p.band_share_pct["sub"] + p.band_share_pct["bass"] > 90.0
    assert p.decay_ms is not None and p.end_level_dbfs < -30.0


def test_leading_silence_and_a_slow_attack_are_read_on_the_samples():
    """100 ms of digital silence, then a tone whose level rises over 50 ms: 10 % to 90 % of
    it is 40 ms. An envelope with any memory (a moving average, an analytic signal) puts
    energy in front of the sound and read 0 ms of silence and a 146 ms attack."""
    t = _t(0.5)
    ramp = np.minimum(1.0, t / 0.05) * 0.8 * np.sin(2 * np.pi * 300 * t)
    p = sample_profile(np.concatenate([np.zeros(int(0.1 * SR)), ramp]), SR)
    assert 99.0 < p.leading_silence_ms < 103.0
    assert 36.0 < p.attack_ms < 44.0


def test_a_sound_the_file_cuts_off_has_no_decay_and_ends_loud():
    cut = sample_profile(_tone(110.0), SR)
    assert cut.decay_ms is None                       # it did not decay: the file ended
    assert cut.end_level_dbfs > -20.0                 # ...loud: a click when it is played
    faded = sample_profile(_tone(110.0) * np.linspace(1.0, 0.0, SR) ** 4, SR)
    assert faded.decay_ms is not None and faded.end_level_dbfs < -60.0


def test_steady_noise_has_no_long_attack():
    """The attack is read against the sound's LEVEL, not its one largest sample: on noise that
    sample is a four-sigma accident, and waiting for 90 % of it read "480 ms"."""
    rng = np.random.default_rng(1)
    p = sample_profile(0.1 * rng.standard_normal(3 * SR), SR)
    assert p.attack_ms < 2.0


# ---------------------------------------------------------------- pitch

@pytest.mark.parametrize("hz, note", [(41.2, "E1"), (55.0, "A1"), (110.0, "A2"),
                                      (440.0, "A4"), (880.0, "A5")])
def test_a_tones_pitch_is_within_five_cents_from_forty_hertz_up(hz, note):
    p = sample_profile(_tone(hz), SR)
    assert _cents(p.pitch_hz, hz) < 5.0 and p.pitch_note == note and abs(p.pitch_cents) < 5.0
    assert p.pitch_midi == round(69 + 12 * np.log2(hz / 440.0))     # the number a note's `pitch` is
    assert p.pitch_confidence > 0.9 and p.noisiness < 0.05
    assert p.pitch_start_hz is None and p.pitch_drop_semitones is None     # it did not move


def test_a_harmonic_rich_bass_reads_its_fundamental_not_an_octave():
    t = _t(1.0)
    p = sample_profile(0.4 * (2 * ((55 * t) % 1) - 1), SR)
    assert _cents(p.pitch_hz, 55.0) < 5.0 and p.pitch_note == "A1"


def test_a_short_blip_still_has_a_pitch():
    blip = 0.5 * np.sin(2 * np.pi * 1000 * _t(0.02))
    p = sample_profile(blip, SR)
    assert _cents(p.pitch_hz, 1000.0) < 5.0


def test_noise_is_noise_and_has_no_pitch():
    rng = np.random.default_rng(2)
    n = int(0.2 * SR)
    hat = 0.3 * rng.standard_normal(n) * np.exp(-np.arange(n) / SR / 0.03)
    p = sample_profile(hat, SR)
    assert p.pitch_hz is None and p.pitch_note is None and p.noisiness > 0.8
    assert p.centroid_hz > 5000 and p.band_share_pct["air"] > 40.0


def test_pitch_is_the_same_at_ninety_six_kilohertz():
    p = sample_profile(_tone(110.0, sr=96000), 96000)
    assert _cents(p.pitch_hz, 110.0) < 5.0


# ---------------------------------------------------------------- level and spectrum

def test_loudness_is_the_meters_on_a_long_signal_and_answers_on_a_short_one():
    rng = np.random.default_rng(3)
    long = 0.1 * rng.standard_normal((3 * SR, 2))
    assert abs(sample_profile(long, SR).loudness_lufs - pyln.Meter(SR).integrated_loudness(long)) < 0.1
    for seconds in (0.02, 0.005):
        short = sample_profile(0.5 * np.sin(2 * np.pi * 1000 * _t(seconds)), SR)
        assert short.loudness_lufs is not None and -9.0 < short.loudness_lufs < -3.0


def test_peaks_rms_and_crest_of_a_sine():
    p = sample_profile(_tone(440.0) * np.linspace(1.0, 0.0, SR), SR)   # faded: no cut at the end
    assert abs(p.sample_peak_dbfs + 6.0) < 0.2 and abs(p.true_peak_dbtp + 6.0) < 0.2
    assert p.full_scale_samples == 0 and not p.over_full_scale


def test_centroid_and_band_shares():
    p = sample_profile(_tone(1500.0), SR)                 # mid-band: 1 kHz itself is a band's edge
    assert abs(p.centroid_hz - 1500.0) <= 10.0
    assert set(p.band_share_pct) == set(BAND_LABELS)
    assert abs(sum(p.band_share_pct.values()) - 100.0) < 0.5
    assert p.band_share_pct["high_mid"] > 95.0


def test_dc_clipping_and_a_float_file_over_full_scale():
    p = sample_profile(_tone(220.0, amp=0.3) + 0.1, SR)
    assert abs(p.dc_offset_pct - 10.0) < 0.6
    clipped = sample_profile(np.clip(_tone(220.0, amp=2.0), -1.0, 1.0), SR)
    assert clipped.full_scale_samples > 1000 and not clipped.over_full_scale
    hot = sample_profile(_tone(220.0, amp=1.4), SR)
    assert hot.over_full_scale and hot.sample_peak_dbfs > 2.5


def test_width_of_mono_centred_and_anti_phase_sounds():
    mono = _tone(220.0)
    assert sample_profile(mono, SR).width_db is None
    centred = sample_profile(np.stack([mono, mono], axis=1), SR)
    assert centred.width_db == -40.0 and centred.stereo_correlation == pytest.approx(1.0, abs=0.01)
    wide = sample_profile(np.stack([mono, -mono], axis=1), SR)
    assert wide.width_db == 6.0 and wide.stereo_correlation == pytest.approx(-1.0, abs=0.01)
    assert _cents(wide.pitch_hz, 220.0) < 5.0            # an anti-phase pair still has a pitch


# ---------------------------------------------------------------- a loop

def test_a_loop_is_held_to_its_length_and_read_at_its_seam():
    n = 4 * SR                                           # 8 beats at 120 BPM
    exact = sample_profile(0.5 * np.sin(2 * np.pi * 110 * np.arange(n) / SR), SR,
                           kind="loop", bpm=120.0, beats=8.0)
    assert exact.loop.length_error_samples == 0 and exact.loop.length_error_ms == 0.0
    assert exact.loop.seam_excess_db <= 1.0              # 110 Hz x 4 s: whole cycles, no step
    assert exact.pitch_hz is None                        # a loop's "pitch" is not asked
    long = sample_profile(0.5 * np.cos(2 * np.pi * 110.3 * np.arange(n + 441) / SR), SR,
                          kind="loop", bpm=120.0, beats=8.0)
    assert long.loop.length_error_samples == 441 and long.loop.length_error_ms == 10.0
    assert long.loop.seam_excess_db > 6.0                # it ends on a peak and starts on another: a click


def test_what_it_refuses_and_what_it_survives():
    with pytest.raises(ValueError):
        sample_profile(_tone(220.0), SR, kind="phrase")
    with pytest.raises(ValueError):
        sample_profile(_tone(220.0), SR, kind="loop", bpm=120.0)      # bpm without beats
    assert sample_profile(np.zeros(SR), SR).silent
    assert sample_profile(np.zeros(4), SR).silent
    broken = _tone(220.0).copy()
    broken[100] = np.nan
    broken[200] = np.inf
    assert _cents(sample_profile(broken, SR).pitch_hz, 220.0) < 5.0   # a non-finite sample is dropped
    tiny = sample_profile(0.5 * np.sin(2 * np.pi * 1000 * _t(0.005)), SR)
    assert not tiny.silent and tiny.sample_peak_dbfs is not None      # five milliseconds answers


# ---------------------------------------------------------------- the read tool

@pytest.mark.anyio
async def test_analyze_sample_round_trips_on_the_demo(demo_project, tmp_path):
    kick = tmp_path / "kick.wav"
    sf.write(kick, _kick(), SR, subtype="FLOAT")
    async with create_connected_server_and_client_session(create_server(demo_project)) as client:
        got = await call(client, "analyze_sample", {"path": str(kick)})
        stem = await call(client, "analyze_sample", {"path": "stem_a.wav"})   # a project file, by its name
        with pytest.raises(Exception):
            await call(client, "analyze_sample", {"path": str(kick), "kind": "loop", "bpm": 120.0})
    SampleProfile.model_validate(got)
    assert got["kind"] == "one_shot" and got["pitch_drop_semitones"] > 17.0
    assert stem["sample_rate_hz"] > 0 and not stem["silent"]


def test_analyze_sample_is_a_read_tool_in_all_six_places():
    def text(rel: str) -> str:
        return (REPO / rel).read_text()
    assert "async def analyze_sample(" in text("server/mix_engine/mcp_server.py")
    from orchestrator import agents, chat, loop, understanding
    assert "analyze_sample" in chat.READ_TOOLS
    assert any(t["name"] == "analyze_sample" for t in chat._READ_TOOL_DEFS)
    assert "analyze_sample" in loop.PLANNER_ENGINE_TOOLS
    assert any(t["name"] == "analyze_sample" for t in agents.PLANNER_TOOL_DEFS)
    SampleProfile.model_validate(understanding.fixture_tool_reply("analyze_sample", {}))
    assert re.search(r'\("analyze_sample", "SampleProfile"\)', text("server/tests/test_reader_p2_py6.py"))


# ---------------------------------------------------------------- the brain measures a render

def test_a_rendered_sounds_profile_is_attached_to_the_same_result(tmp_path):
    from orchestrator import daw_bridge as B

    kick, loop_file = tmp_path / "kick.wav", tmp_path / "loop.wav"
    sf.write(kick, _kick(), SR, subtype="FLOAT")
    sf.write(loop_file, 0.5 * np.sin(2 * np.pi * 110 * np.arange(4 * SR) / SR), SR, subtype="FLOAT")
    res = {"ok": True, "took_ms": 12, "sounds": [
        {"name": "kick", "file": str(kick), "length_s": 0.6, "bpm": 120.0},
        {"name": "bass loop", "file": str(loop_file), "length_s": 4.0, "bpm": 120.0, "loop_beats": 8.0},
        {"name": "gone", "file": str(tmp_path / "nothing.wav"), "length_s": 1.0},
        {"name": "failed", "error": "the render failed"}]}
    out = B.attach_sound_profiles(res)
    assert out is res and res["ok"] is True                     # the render's own answer stands
    kick_p, loop_p = res["sounds"][0]["profile"], res["sounds"][1]["profile"]
    assert kick_p["pitch_drop_semitones"] > 17.0 and "loop" not in kick_p
    assert loop_p["loop"]["length_error_samples"] == 0
    assert all(v is not None for v in kick_p.values())           # nothing "not measurable" is sent
    for dropped in ("profile_version", "path", "sample_rate_hz", "duration_s"):
        assert dropped not in kick_p
    assert "profile" not in res["sounds"][2] and "profile_error" in res["sounds"][2]
    assert "profile" not in res["sounds"][3] and "profile_error" not in res["sounds"][3]
    json.dumps(res)                                              # it is still a tool result
    # a silent file says only that
    sf.write(kick, np.zeros(SR), SR, subtype="FLOAT")
    assert B.attach_sound_profiles({"sounds": [{"file": str(kick)}]})["sounds"][0]["profile"] == {"silent": True}
    # ...and what is not a render's reply is handed back as it came
    assert B.attach_sound_profiles({"ok": False, "error": "x"}) == {"ok": False, "error": "x"}
    assert B.attach_sound_profiles(None) is None


def test_only_a_render_is_measured_and_at_both_doors(tmp_path):
    from orchestrator import turn_host

    kick = tmp_path / "kick.wav"
    sf.write(kick, _kick(), SR, subtype="FLOAT")

    async def go():
        made = await turn_host.measured_command("sound.render", {"ok": True, "sounds": [{"file": str(kick)}]})
        other = await turn_host.measured_command("track.set_gain", {"ok": True, "sounds": [{"file": str(kick)}]})
        return made, other
    made, other = anyio.run(go)
    assert "profile" in made["sounds"][0] and "profile" not in other["sounds"][0]
    # the live tool and an approved plan's step both go through it
    host = (REPO / "orchestrator/turn_host.py").read_text()
    bridge = (REPO / "orchestrator/daw_bridge.py").read_text()
    # the live door hands its project too, and so does its look (Phase 64: a snapshot's
    # picture is read there - `do_look`, which runs in Ask mode as in Agent)
    assert host.count("await measured_command(command, res, self.project_dir)") == 2
    assert "r = await measured_command(command, r)" in bridge
    # ...and the brain waits for a render as long as it waits for any other
    from orchestrator import daw_server
    assert daw_server.reply_wait_s("sound.render", False) == daw_server.RENDER_REPLY_TIMEOUT_S
