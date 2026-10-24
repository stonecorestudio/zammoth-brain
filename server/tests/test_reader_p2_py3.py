"""Gates for batch Py3 of the reader P2s (full audit 2026-10-02): the offline render and analysis.

Seven findings - D2·R6, E4·R4, E5·R4, E5·R5, E5·R6, E5·R7, E5·R8 - and the two leads the triage
filed with them (a stem whose Audio To is a muted track still sending; `render_master`'s NaN
guard letting an inf through). Each test drives the door the product uses and carries its
control. Every one was run on the unfixed tree first. The record is
docs/audits/full-2026-10-02/readers/P2-fixes.md (Py3).
"""
from __future__ import annotations

import asyncio
import json
import math

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt

import test_analysis_ground_truth as gt
from mix_engine import analysis, dsp, impact
from mix_engine.models import ANALYSIS_VERSION, ProposalRequest
from mix_engine.session import Session
from orchestrator import automix, calibration, daw_bridge

SR = 48000


def _apply(session: Session, actions: list[dict]) -> str:
    result = session.propose(ProposalRequest(rationale="py3", predicted_impact=[], actions=actions))
    assert result.accepted, result.violations
    assert session.apply(result.proposal_id).ok
    return result.proposal_id


def _rms_db(x: np.ndarray) -> float:
    return 20.0 * math.log10(max(float(np.sqrt(np.mean(np.square(x)))), 1e-12))


def _write(path, audio: np.ndarray, sr: int = SR, subtype: str = "FLOAT") -> None:
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sr, subtype=subtype)


def _project(tmp_path, stems: dict[str, np.ndarray], sr: int = SR):
    proj = tmp_path / "proj"
    proj.mkdir(parents=True)
    for name, audio in stems.items():
        _write(proj / f"{name}.wav", audio, sr)
    return proj


def _music(seconds: float = 4.0, sr: int = SR, hz: float = 220.0, level: float = 0.2) -> np.ndarray:
    return gt._tone(hz, seconds, sr, level)


# ====================================================================================
# D2·R6 - a muted destination silences what plays into it, sends included
# ====================================================================================

def _sending_member(demo_project):
    """stem_a in a group and sending to a reverb at unity; stem_b beside it, dry."""
    s = Session(demo_project)
    s.refresh_stems()
    bus = s.group_stems(["stem_a"])["bus_id"]
    ret = s.add_return("reverb")["return_id"]
    s.set_send("stem_a", ret, 0.0)
    return s, bus, ret


def _master_without_send(demo_project, source: str, ret: str) -> np.ndarray:
    twin = Session(demo_project)
    twin.set_send(source, ret, -60.0)          # the floor: no send at all
    return dsp.render_master(twin)[0]


def test_a_member_of_a_muted_group_sends_nothing(demo_project):
    """Natively a member of a muted bus is "muted by destination" and its AuxSend returns 0;
    offline its sends were summed before the bus's mute was consulted, so the reverb the user
    muted stayed in the master the agent measures (reader: rms(diff) 0.0555)."""
    s, bus, ret = _sending_member(demo_project)
    loud = dsp.render_master(s)[0]
    dry_twin = _master_without_send(demo_project, "stem_a", ret)
    assert _rms_db(loud - dry_twin[: loud.shape[0]]) > -60.0, "control: the send is audible"

    s.user_edit(bus, {"muted": True})
    muted = dsp.render_master(Session(demo_project))[0]
    twin = Session(demo_project)
    twin.set_send("stem_a", ret, -60.0)
    without = dsp.render_master(twin)[0]
    assert np.allclose(muted, without, atol=1e-9), (
        f"the muted group's member still feeds its return: {_rms_db(muted - without):.1f} dB")


def test_a_child_group_under_a_muted_parent_sends_nothing(demo_project):
    s = Session(demo_project)
    s.refresh_stems()
    child = s.group_stems(["stem_a"])["bus_id"]
    parent = s.group_stems(["stem_b"])["bus_id"]
    s.route_bus(child, parent)
    ret = s.add_return("reverb")["return_id"]
    s.set_send(child, ret, 0.0)                 # the CHILD group's own send
    assert float(np.abs(dsp.render_master(s)[0]).max()) > 1e-4

    s.user_edit(parent, {"muted": True})
    silent = dsp.render_master(Session(demo_project))[0]
    assert float(np.abs(silent).max()) < 1e-9, "the child's send played on under its muted parent"


def test_a_track_whose_audio_to_is_a_muted_track_sends_nothing(demo_project):
    """The lead filed with D2·R6: `stem_output` == ("track", dest) with dest muted - the feeder's
    dry signal is gone with its destination, its sends were not."""
    s = Session(demo_project)
    s.refresh_stems()
    ret = s.add_return("reverb")["return_id"]
    s.set_send("stem_a", ret, 0.0)
    s.user_edit("stem_a", {"output": "track:stem_b"})
    assert s.stem_output("stem_a") == ("track", "stem_b")
    assert float(np.abs(dsp.render_master(s)[0]).max()) > 1e-4

    s.user_edit("stem_b", {"muted": True})
    silent = dsp.render_master(Session(demo_project))[0]
    assert float(np.abs(silent).max()) < 1e-9, "a feeder of a muted track still fed its return"


def test_the_return_alone_follows_the_same_rule(demo_project):
    """`render_return` is what `analyze_return` hears, and it "mirrors render_master"."""
    s, bus, ret = _sending_member(demo_project)
    assert float(np.abs(dsp.render_return(s, ret)[0]).max()) > 1e-4
    s.user_edit(bus, {"muted": True})
    with pytest.raises(ValueError, match="no active sends"):
        dsp.render_return(Session(demo_project), ret)
    # ...and the muted group's OWN send, which the master has refused since 2026-09-05
    again = Session(demo_project)
    again.set_send(bus, ret, 0.0)
    with pytest.raises(ValueError, match="no active sends"):
        dsp.render_return(Session(demo_project), ret)
    again.user_edit(bus, {"muted": False})
    assert float(np.abs(dsp.render_return(Session(demo_project), ret)[0]).max()) > 1e-4


# ====================================================================================
# E4·R4 - a mono stem plays from both speakers, so it is rendered as two channels
# ====================================================================================

def _mono_project(tmp_path):
    rng = np.random.default_rng(7)
    voice = _music(4.0) + 0.02 * rng.standard_normal(int(4.0 * SR))
    return _project(tmp_path, {"vox": voice, "bed": _music(4.0, hz=110.0)})


def test_a_mono_stem_renders_as_two_identical_channels(tmp_path):
    proj = _mono_project(tmp_path)
    audio, _ = dsp.render_stem(Session(proj), "vox")
    assert audio.shape[1] == 2, "the DAW's graph runs a mono clip dual-mono; offline it was 1 channel"
    assert np.array_equal(audio[:, 0], audio[:, 1])
    master, _ = dsp.render_master(Session(proj))
    assert master.shape[1] == 2, "the DAW's export of an all-mono project is a stereo file"


def test_one_sided_polarity_on_a_mono_stem_cancels_in_mono_as_it_does_in_the_daw(tmp_path):
    """The agent's prompt teaches invert_left as a stem repair; on a mono stem the offline render
    (its measurement) was unchanged while the DAW plays the stem anti-phase."""
    proj = _mono_project(tmp_path)
    s = Session(proj)
    before = dsp.render_stem(s, "vox")[0]
    _apply(s, [{"type": "utility", "stem_id": "vox", "invert_left": True}])
    after = dsp.render_stem(Session(proj), "vox")[0]
    assert _rms_db(after.mean(axis=1)) < -100.0, "the mono sum of L = -R is silence"
    assert _rms_db(after[:, 1]) == pytest.approx(_rms_db(before.mean(axis=1)), abs=0.01)


@pytest.mark.parametrize("action", [
    {"type": "comp", "stem_id": "vox", "channel": "side", "threshold_db": -40.0, "ratio": 8.0,
     "attack_ms": 1.0, "release_ms": 50.0, "makeup_db": 4.0},
    {"type": "eq", "stem_id": "vox", "channel": "side",
     "band": {"shape": "high_shelf", "freq_hz": 3000, "gain_db": 6.0, "q": 0.7}},
], ids=["comp-side", "eq-side"])
def test_a_side_only_device_does_nothing_to_a_mono_stem(tmp_path, action):
    """Side = 0.5 (L - R) = 0 on a mono source: the DAW's side comp / side shelf processes
    nothing. Offline the helpers fell through on one channel and processed the WHOLE stem
    (reader: +8.79 dB for the comp's make-up, -2.12 dB for the shelf)."""
    proj = _mono_project(tmp_path)
    s = Session(proj)
    before = dsp.render_stem(s, "vox")[0].mean(axis=1)
    _apply(s, [action])
    after = dsp.render_stem(Session(proj), "vox")[0].mean(axis=1)
    assert _rms_db(after - before) < -90.0, f"moved the mono sum: {_rms_db(after) - _rms_db(before):+.2f} dB"


def test_a_left_only_band_on_a_mono_stem_leaves_the_right_channel_alone(tmp_path):
    proj = _mono_project(tmp_path)
    s = Session(proj)
    before = dsp.render_stem(s, "vox")[0]
    _apply(s, [{"type": "eq", "stem_id": "vox", "channel": "left",
                "band": {"shape": "peak", "freq_hz": 220, "gain_db": -9.0, "q": 1.0}}])
    after = dsp.render_stem(Session(proj), "vox")[0]
    assert after.shape[1] == 2
    assert _rms_db(after[:, 1] - before[:, -1]) < -90.0, "the right channel is untouched"
    assert _rms_db(after[:, 0]) < _rms_db(before[:, 0]) - 3.0, "the left channel took the cut"


def test_a_plain_chain_on_a_mono_stem_is_the_same_samples_in_both_channels(tmp_path):
    """Control: what does not look at channels is unchanged - each channel is what the one
    channel used to be, and the loudness (measured as played since analysis v7) does not move."""
    proj = _mono_project(tmp_path)
    s = Session(proj)
    _apply(s, [{"type": "eq", "stem_id": "vox",
                "band": {"shape": "high_shelf", "freq_hz": 3000, "gain_db": 4.0, "q": 0.7}},
               {"type": "comp", "stem_id": "vox", "threshold_db": -24.0, "ratio": 3.0,
                "attack_ms": 10.0, "release_ms": 120.0}])
    audio = dsp.render_stem(Session(proj), "vox")[0]
    assert np.allclose(audio[:, 0], audio[:, 1], atol=1e-7)
    one = analysis.analyze_stem(audio[:, :1], SR, "vox", [])
    two = analysis.analyze_stem(audio, SR, "vox", [])
    assert one.lufs_s == two.lufs_s and one.peak_dbfs == two.peak_dbfs


# ====================================================================================
# E5·R4 - the readout pays for the readout
# ====================================================================================

_READOUT_NEVER = ("_band_correlation", "_resonances", "_lufs_range", "_band_crest_db",
                  "_tuning", "_short_term_max_lufs", "_transient_energy_ratio")


def test_the_readout_computes_only_what_it_shows(demo_project, monkeypatch):
    """`session_readout` / `chains_with_readout` ran the whole `analyze_master` and kept eight
    fields (reader: 5.7 s of 13.6 on band correlation, resonances, LRA ... only to drop them)."""
    session = Session(demo_project)
    audio, sr = dsp.render_master(session)
    full = daw_bridge._readout_from_analysis(
        analysis.analyze_master(audio, sr, stems=sorted(session.stems), pass_number=0))

    def never(name):
        def boom(*_a, **_k):
            raise AssertionError(f"the readout computed analysis.{name} and threw it away")
        return boom
    for name in _READOUT_NEVER:
        monkeypatch.setattr(analysis, name, never(name))

    assert daw_bridge.session_readout(demo_project) == full
    assert daw_bridge.chains_with_readout(demo_project)["readout"] == full


def test_true_peak_is_measured_in_blocks_and_reads_the_same(monkeypatch):
    """The 4x copy of the whole float64 master was 2.2 GB at six minutes."""
    rng = np.random.default_rng(3)
    n = 90 * SR
    audio = 0.25 * rng.standard_normal((n, 2))
    # an inter-sample peak astride a likely block edge, and one at each end
    for at in (0, 1 << 18, (1 << 18) + 1, n - 2):
        audio[at:at + 2, 0] = (0.99, -0.99) if at % 2 == 0 else (-0.99, 0.99)
    whole = 20.0 * math.log10(float(np.max(np.abs(resample_poly(audio, 4, 1, axis=0)))))

    longest = {"frames": 0}
    real = analysis.resample_poly

    def watched(x, *a, **k):
        longest["frames"] = max(longest["frames"], int(np.shape(x)[0]))
        return real(x, *a, **k)
    monkeypatch.setattr(analysis, "resample_poly", watched)

    assert analysis._true_peak_dbtp(audio) == pytest.approx(whole, abs=0.01)
    assert longest["frames"] <= 1 << 20, f"true peak oversampled {longest['frames']} frames at once"

    # A block's edge is not an edge of the audio: a steady level cut off there rings (the
    # filter sees silence beyond it) and would read as an over that is not in the file.
    held = np.full((n, 1), 0.9)
    ramp = np.linspace(0.0, 1.0, SR)
    held[:SR, 0] *= ramp
    held[-SR:, 0] *= ramp[::-1]
    assert analysis._true_peak_dbtp(held) == pytest.approx(20.0 * math.log10(0.9), abs=0.01)


def _busy(seconds: float = 40.0) -> np.ndarray:
    rng = np.random.default_rng(21)
    song = np.tile(gt._progression("D", "maj"), int(seconds / 8.0) + 1)[: int(seconds * SR)]
    return song + 0.05 * rng.standard_normal(song.size) + gt._click_track(120.0, seconds)[: song.size]


def test_the_chunked_welch_is_scipys_welch(monkeypatch):
    """Every whole-song Welch and STFT is computed a run of segments at a time (0.7 GB a Welch
    and 1.4 GB for the timing STFT at six minutes, before) - and gives the same numbers."""
    from scipy.signal import welch
    x = _busy()
    monkeypatch.setattr(analysis, "WELCH_CHUNK_SEGMENTS", 3)
    for nperseg in (4096, 8192, 16384, x.size, x.size + 5):
        nperseg = min(nperseg, x.size)
        f_ref, p_ref = welch(x, fs=SR, nperseg=nperseg)
        f, p = analysis._welch_psd(x, SR, nperseg)
        assert np.array_equal(f, f_ref) and np.allclose(p, p_ref, rtol=1e-10, atol=0.0)
    from scipy.signal import csd
    other = np.roll(x, 37) + 0.5 * x
    _, ref = csd(x, other, fs=SR, nperseg=4096)
    _, got = analysis._welch_psd(x, SR, 4096, y=other)
    assert np.allclose(got, ref, rtol=1e-10, atol=1e-18)


def test_no_whole_song_transform_bypasses_the_chunked_helpers():
    """`welch`, `csd` and `stft` are called in ONE place each in analysis.py."""
    import ast
    from pathlib import Path
    tree = ast.parse(Path(analysis.__file__).read_text(encoding="utf-8"))
    allowed = {"_welch_psd", "_stft_magnitudes"}
    strays = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        if fn.name in allowed or any(isinstance(p, ast.FunctionDef) and p.name in allowed
                                     and fn in ast.walk(p) for p in ast.walk(tree)):
            continue
        for call in [n for n in ast.walk(fn) if isinstance(n, ast.Call)]:
            name = getattr(call.func, "id", getattr(call.func, "attr", ""))
            if name in ("welch", "csd", "stft", "spectrogram"):
                strays.append(f"{fn.name}: {name}() line {call.lineno}")
    assert strays == [], strays


@pytest.mark.parametrize("nperseg, hop", [(512, 128), (1024, 512), (8192, 4096)])
def test_the_chunked_stft_is_scipys_stft(monkeypatch, nperseg, hop):
    from scipy.signal import stft
    monkeypatch.setattr(analysis, "STFT_CHUNK_SAMPLES", 7 * 512)     # 7, 3 and 1 frames a block
    for x in (_busy(12.0), _busy(12.0)[:-333]):                 # a length that needs padding too
        _, _, spec = stft(x, fs=SR, nperseg=nperseg, noverlap=nperseg - hop)
        ref = np.abs(spec)
        got = np.concatenate(list(analysis._stft_magnitudes(x, nperseg, hop)), axis=0).T
        assert got.shape == ref.shape and np.allclose(got, ref, rtol=1e-9, atol=1e-15)
        flux_ref = np.maximum(np.diff(ref, axis=1), 0.0).sum(axis=0)
        assert np.allclose(analysis._stft_flux(x, nperseg, hop), flux_ref, rtol=1e-9, atol=1e-15)


def test_the_readout_holds_a_few_copies_of_the_song_not_a_dozen():
    """Peak allocation of the readout's analysis, in units of the audio it is given: 5.5 at six
    minutes before (8 with the whole-file true peak), and it is what swaps on a long project."""
    import tracemalloc
    rng = np.random.default_rng(2)
    mono = _busy(120.0)
    audio = np.stack([mono, mono + 0.01 * rng.standard_normal(mono.size)], axis=1)
    tracemalloc.start()
    analysis.readout_metrics(audio, SR)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak / audio.nbytes < 2.2, f"the readout held {peak / audio.nbytes:.1f} copies of the song"


# ====================================================================================
# E5·R5 - the tilt is the slope of the octaves, not of the top one
# ====================================================================================

def _coloured(slope_db_oct: float, seconds: float = 30.0, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / SR)
    f[0] = f[1]
    spec *= (f / 1000.0) ** (slope_db_oct / (20.0 * math.log10(2.0)))
    x = np.fft.irfft(spec, n)
    return (0.1 * x / np.sqrt(np.mean(x ** 2)))[:, np.newaxis]


def _tilt(x: np.ndarray) -> float:
    return analysis._spectral_tilt_db_per_oct(x, SR)


@pytest.mark.parametrize("slope", [0.0, -3.0103, -6.0206])
def test_tilt_still_reads_white_pink_and_brown(slope):
    assert _tilt(_coloured(slope)) == pytest.approx(slope, abs=0.15)


def test_a_bass_boost_moves_the_tilt_and_a_16k_low_pass_barely_does():
    """On linear Welch bins half the regression points sat above 10 kHz: +6 dB below 200 Hz
    moved the tilt 0.06 dB/oct, a 16 kHz low-pass moved it 3.3."""
    pink = _coloured(-3.0103)[:, 0]
    base = _tilt(pink[:, None])
    low = sosfilt(butter(4, 200.0, btype="low", fs=SR, output="sos"), pink)
    warmer = pink + (10 ** (6.0 / 20.0) - 1.0) * low
    lossy = sosfilt(butter(12, 16000.0, btype="low", fs=SR, output="sos"), pink)
    by_bass = _tilt(warmer[:, None]) - base
    by_lowpass = _tilt(lossy[:, None]) - base
    assert by_bass < -0.4, f"+6 dB of bass moved the tilt {by_bass:+.2f} dB/oct"
    assert abs(by_lowpass) < abs(by_bass), (
        f"a 16 kHz low-pass ({by_lowpass:+.2f}) outweighs +6 dB of bass ({by_bass:+.2f})")


def test_the_tilt_redefinition_is_a_new_analysis_version():
    assert ANALYSIS_VERSION >= 8   # v9 since Py7 (resonances); this pins that v8 was a bump


def test_the_daws_band_slope_is_not_called_the_tilt():
    """`tilt_db_per_oct` is the DAW's own least-squares slope of eight BAND POWERS (pink reads
    about 0 there and -3 here); both were shown to the agent as "tilt dB/oct"."""
    ours = daw_bridge._DELTA_LABELS["spectral_tilt_db_per_oct"]
    theirs = daw_bridge._DELTA_LABELS["tilt_db_per_oct"]
    assert ours != theirs and "band" in theirs[1]


# ====================================================================================
# E5·R6 - a key is several notes; a tuning is a steady partial
# ====================================================================================

def _held_note(seconds: float = 20.0) -> np.ndarray:
    """One sustained A3 with six partials (the reader's `harmonic_tone`): its own harmonic
    series spells A - E - C#, which is why it read "A maj"."""
    t = np.arange(int(seconds * SR)) / SR
    x = sum((0.6 ** h) * np.sin(2 * np.pi * 220.0 * (h + 1) * t) for h in range(6))
    return 0.2 * x / np.max(np.abs(x))


def _kick_and_toms(seconds: float = 16.0) -> np.ndarray:
    """A drum loop the way the fixtures build one: a decaying sine kick, band-noise toms."""
    rng = np.random.default_rng(5)
    n = int(seconds * SR)
    out = np.zeros(n)
    kick_t = np.arange(int(0.12 * SR)) / SR
    kick = np.sin(2 * np.pi * 70.0 * kick_t) * np.exp(-kick_t / 0.03)
    tom = sosfilt(butter(4, [150.0, 400.0], btype="band", fs=SR, output="sos"),
                  rng.standard_normal(int(0.15 * SR))) * np.exp(-np.arange(int(0.15 * SR)) / (0.04 * SR))
    for beat in range(int(seconds * 2)):
        at = int(beat * 0.5 * SR)
        hit = kick if beat % 2 == 0 else 0.6 * tom
        out[at:at + hit.size] += hit[: n - at]
    return 0.5 * out / np.max(np.abs(out))


def _flagged_out_of_tune(mono: np.ndarray) -> bool:
    tuning = analysis._tuning(mono, SR)
    return tuning is not None and (abs(tuning[0]) >= analysis.TUNING_FLAG_CENTS
                                   or tuning[1] < analysis.TUNING_COHERENCE_FLOOR)


def test_one_held_note_has_no_key():
    assert analysis._key(_held_note(), SR) is None
    assert analysis._tuning(_held_note(), SR) is not None, "control: a held note still has a tuning"


def test_two_notes_have_no_key():
    pair = np.concatenate([gt._tone(f, 2.0) for f in (220.0, 329.63)] * 5)
    assert analysis._key(pair, SR) is None


def test_a_drum_loop_has_no_key_and_is_not_out_of_tune():
    drums = _kick_and_toms()
    assert analysis._key(drums, SR) is None
    assert not _flagged_out_of_tune(drums), f"a kick is not the song's tuning: {analysis._tuning(drums, SR)}"


@pytest.mark.parametrize("fixture", ["technique_demo/drums.wav", "wild/lofi_16_441/drums.wav",
                                     "wild/master_trap/beat.wav", "wild/mono_only/kick.wav"])
def test_the_fixture_drum_stems_have_no_key_and_no_tuning(fixture):
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "test_projects" / fixture
    if not path.exists():
        pytest.skip(f"{fixture} is generated on demand")
    audio, sr = sf.read(str(path), dtype="float64", always_2d=True)
    mono = audio.mean(axis=1)
    assert analysis._key(mono, sr) is None
    assert analysis._tuning(mono, sr) is None


def test_a_click_track_has_no_tuning():
    assert analysis._tuning(gt._click_track(120.0, 20.0), SR) is None


def test_a_drums_only_project_shows_no_key_in_the_daws_readout(tmp_path):
    proj = _project(tmp_path, {"drums": _kick_and_toms()})
    readout = daw_bridge.session_readout(proj)
    assert readout["key"] is None, "the DAW aligns its key strip to this readout"


def test_real_keys_survive_drums_on_top():
    """Control: the gates refuse a guess, not music with a loud drum loop in it."""
    song = gt._progression("G", "maj")
    drums = _kick_and_toms(song.size / SR + 0.5)[: song.size]
    drums *= 2.0 * np.sqrt(np.mean(song ** 2)) / np.sqrt(np.mean(drums ** 2))   # +6 dB over the chords
    key = analysis._key(song + drums, SR)
    assert key is not None and (key.tonic, key.mode) == ("G", "maj")
    tuning = analysis._tuning(song + drums, SR)
    assert tuning is not None and abs(tuning[0]) < 5.0


def test_a_single_triad_still_has_a_key():
    key = analysis._key(gt._chord("G", "maj", 8.0), SR)
    assert key is not None and (key.tonic, key.mode) == ("G", "maj")


# ====================================================================================
# E5·R7 - "separation artifacts" are measured on what was separated
# ====================================================================================

def _clean_song(tmp_path):
    """Three clean, never-separated parts that share one song form (the reader's sep_fp.py):
    a vocal with esses, comped chords, a drum loop. Each read as an artifact."""
    rng = np.random.default_rng(11)
    seconds, n = 16.0, int(16.0 * SR)
    t = np.arange(n) / SR
    form = np.where((t % 8.0) < 4.0, 0.5, 1.0)                     # verse quiet, chorus loud
    vocal = np.zeros(n)
    ess = sosfilt(butter(4, [5000.0, 9000.0], btype="band", fs=SR, output="sos"),
                  rng.standard_normal(int(0.12 * SR)))
    for k in range(int(seconds / 0.8)):
        at = int(k * 0.8 * SR)
        note = gt._tone(220.0 * 2 ** ((k % 5) / 12.0), 0.6, SR, 0.25)
        vocal[at:at + note.size] += note[: n - at]
        s_at = at + note.size
        if s_at + ess.size < n:
            vocal[s_at:s_at + ess.size] += 0.04 * ess
    guitar = np.concatenate([gt._chord(r, q, 2.0, SR) for r, q in
                             (("A", "min"), ("F", "maj"), ("C", "maj"), ("G", "maj"))] * 2)[:n] * 0.4
    drums = _kick_and_toms(seconds)[:n]
    return _project(tmp_path, {"vocal": vocal * form, "guitar": guitar * form, "drums": drums * form})


def _run_cleanup(project, *, offline_only: bool):
    from orchestrator.benchmark import in_memory_engine
    from orchestrator.loop import EngineClient

    stages: list[dict] = []
    left_out: list = []

    async def run():
        async def emit(event):
            stages.append(event)
        async with in_memory_engine(project) as client:
            return await automix.cleanup(EngineClient(client), emit,
                                         offline_only=offline_only, left_out=left_out)
    return asyncio.run(run()), left_out, stages


def _device_types(project) -> set[str]:
    state = json.loads((project / "session.json").read_text())
    return {e["type"] for stem in state["stems"].values() for e in stem.get("chain", [])
            if e["type"] != "gain"}


@pytest.mark.parametrize("offline_only", [True, False], ids=["gateway", "daw"])
def test_cleanup_repairs_nothing_in_a_project_nothing_was_split_in(tmp_path, offline_only):
    """The preset measured EVERY stem: sibilance read as shimmer, sustain as smear, playing in
    time as leakage - and "repaired" a clean vocal, guitar and drums (the DAW: a transient
    shaper on the guitar, and a notice that three stems' de-noise "was left out")."""
    proj = _clean_song(tmp_path)
    applied, left_out, _ = _run_cleanup(proj, offline_only=offline_only)
    assert applied == [] and left_out == []
    assert _device_types(proj) == set() and Session(proj).buses == {}


def test_the_daws_cleanup_says_that_nothing_here_was_split(tmp_path):
    """The DAW's door (`auto_mix_pass` with the cleanup brief): nothing applied, and the reason
    in words - the reply's `chat`, which the app shows as why Auto Mix did not run."""
    proj = _clean_song(tmp_path)
    Session(proj).refresh_stems()
    notes: list = []
    left_out: list = []
    assert automix.is_cleanup_brief("clean up separation artifacts")
    applied = daw_bridge.auto_mix_pass(proj, "clean up separation artifacts",
                                       left_out=left_out, notes=notes)
    assert applied == 0 and left_out == []
    assert notes == [automix.NOTHING_SEPARATED] and "Split into stems" in notes[0]
    assert _device_types(proj) == set()


def test_the_gateways_split_marks_its_files_through_refresh_stems(tmp_path):
    """The offline gateway's door: `refresh_stems {separated: [...]}` after it wrote the wavs."""
    from orchestrator.benchmark import in_memory_engine
    from orchestrator.loop import EngineClient

    proj = _clean_song(tmp_path)

    async def run():
        async with in_memory_engine(proj) as client:
            engine = EngineClient(client)
            await engine.call("refresh_stems", {"separated": ["guitar.wav", "drums.wav"]})
            return await engine.call("analyze_separation")
    report = asyncio.run(run())
    assert [art["stem_id"] for art in report["per_stem"]] == ["drums", "guitar"]
    assert report["not_separated"] == ["vocal"]


def test_the_same_material_is_repaired_once_it_is_known_to_be_split(tmp_path):
    """Control: the gate is provenance, not a blanket off-switch."""
    proj = _clean_song(tmp_path)
    Session(proj).note_separated(["vocal.wav", "guitar.wav", "drums.wav"])
    assert Session(proj).separated_stems() == ["drums", "guitar", "vocal"]
    applied, _, _ = _run_cleanup(proj, offline_only=True)
    assert applied and _device_types(proj)


def test_split_into_stems_records_where_its_stems_came_from(tmp_path, monkeypatch):
    """The real door: the DAW's `separate_track` writes the wavs, the import registers them."""
    monkeypatch.setenv("MIX_AGENT_SEPARATOR", "fake")
    proj = _project(tmp_path, {"bed": _music(4.0, hz=110.0)})
    Session(proj).refresh_stems()
    source = tmp_path / "song.wav"
    _write(source, np.stack([_music(4.0), _music(4.0, hz=330.0)], axis=1))
    result = daw_bridge.separate_track(proj, str(source), stems=2)
    assert result["ok"], result
    for stem in result["stems"]:
        assert daw_bridge.import_stem(proj, str(proj / stem["wav"]))["ok"]
    session = Session(proj)
    split = sorted(stem["wav"][:-4] for stem in result["stems"])
    assert session.separated_stems() == split
    assert "bed" not in session.separated_stems()
    session.rename_stem(split[0], "lead")                       # the record follows the FILE
    assert "lead" in Session(proj).separated_stems()


def test_the_separation_report_names_what_it_left_out(tmp_path):
    from orchestrator.benchmark import in_memory_engine
    from orchestrator.loop import EngineClient

    proj = _clean_song(tmp_path)
    Session(proj).note_separated(["vocal.wav"])

    async def run():
        async with in_memory_engine(proj) as client:
            return await EngineClient(client).call("analyze_separation")
    report = asyncio.run(run())
    assert [art["stem_id"] for art in report["per_stem"]] == ["vocal"]
    assert report["not_separated"] == ["drums", "guitar"]


# ====================================================================================
# E5·R8 - one non-finite sample is one dropped sample, never a NaN in the store
# ====================================================================================

@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "-inf"])
def test_a_non_finite_sample_is_dropped_at_the_read_as_the_daw_drops_it(tmp_path, bad):
    """The DAW's node zeroes a non-finite sample coming in (batch M4). Offline a NaN made every
    render raise without naming the file, and an inf went through to the master's metrics."""
    good = _music(4.0)
    spoiled = good.copy()
    spoiled[SR] = bad
    clean_proj = _project(tmp_path / "clean", {"vox": good, "bed": _music(4.0, hz=110.0)})
    proj = _project(tmp_path / "bad", {"vox": spoiled, "bed": _music(4.0, hz=110.0)})

    report: dict = {}
    audio, sr = dsp.render_master(Session(proj), report=report)
    assert np.isfinite(audio).all()
    assert report["non_finite_stems"] == ["vox"], "the render says which file it mended"
    twin = dsp.render_master(Session(clean_proj))[0]
    assert _rms_db(audio - twin) < -60.0, "one dropped sample, otherwise the same master"

    result = analysis.analyze_master(audio, sr, stems=["bed", "vox"], pass_number=0)
    assert all(np.isfinite(v) for v in result.band_energy.values())
    assert np.isfinite(result.true_peak_dbtp)
    clean_report: dict = {}
    dsp.render_master(Session(clean_proj), report=clean_report)
    assert not clean_report.get("non_finite_stems")


def test_the_dawside_readout_names_the_mended_stem(tmp_path):
    spoiled = _music(4.0)
    spoiled[100] = np.inf
    proj = _project(tmp_path, {"vox": spoiled})
    readout = daw_bridge.session_readout(proj)
    assert readout["non_finite_stems"] == ["vox"]
    assert np.isfinite(readout["true_peak_dbtp"]) and np.isfinite(readout["lufs_i"])


def test_analysis_given_a_non_finite_sample_measures_the_rest():
    """The doors that read a file directly (a reference, a region) get audio no render mended."""
    audio = np.stack([_music(4.0), _music(4.0)], axis=1)
    audio[SR, 0] = np.inf
    audio[2 * SR, 1] = np.nan
    master = analysis.analyze_master(audio, SR, stems=["x"], pass_number=0)
    stem = analysis.analyze_stem(audio, SR, "x", [])
    assert np.isfinite(master.true_peak_dbtp) and np.isfinite(master.crest_factor_db)
    assert all(np.isfinite(v) for v in master.band_energy.values())
    assert np.isfinite(stem.peak_dbfs) and all(np.isfinite(v) for v in stem.band_energy.values())


def test_the_master_render_refuses_an_inf_a_device_made(demo_project, monkeypatch):
    """The lead filed with E5·R8: the guard was `np.isnan(mix).any()` - an inf passed it."""
    def blown(audio, *_a, **_k):
        out = np.array(audio, dtype=np.float64, copy=True)
        out[10] = np.inf
        return out
    monkeypatch.setattr(dsp, "_apply_master_chain", blown)
    s = Session(demo_project)
    _apply(s, [{"type": "limiter", "stem_id": "master", "ceiling_dbtp": -1.0}])
    with pytest.raises(FloatingPointError, match="non-finite"):
        dsp.render_master(Session(demo_project))


def test_a_non_finite_measurement_is_no_measurement():
    class Model:
        band_energy = {"presence": float("nan"), "bass": -6.0}
        true_peak_dbtp = float("inf")
        lufs_i = -14.0
    assert impact._field(Model, "band_energy.presence") is None
    assert impact._field(Model, "true_peak_dbtp") is None
    assert impact._field(Model, "band_energy.bass") == -6.0 and impact._field(Model, "lufs_i") == -14.0


def _record(measured: float) -> dict:
    return {"actions": [{"type": "eq", "stem_id": "vox"}], "stem_tags": {"vox": ["vocal"]},
            "predicted_impact": [{"metric": "master.band_energy.presence", "predicted_delta": 1.0,
                                  "unit": "dB"}],
            "measured_impact": [{"metric": "master.band_energy.presence",
                                 "measured_delta": measured, "unit": "dB"}]}


def test_a_nan_never_reaches_the_cross_project_store():
    calibration.update_global_store([_record(1.4)])
    good = calibration._store_path().read_text()
    calibration.update_global_store([_record(float("nan")), _record(float("inf"))])
    store = json.loads(calibration._store_path().read_text())
    assert "NaN" not in calibration._store_path().read_text()
    assert store == json.loads(good), "a non-finite error was folded into the running sums"
    assert all(math.isfinite(row["bias"]) for row in calibration.load_global_rows())
    assert calibration.calibration_table([_record(float("nan")), _record(1.4), _record(1.4),
                                          _record(1.4)]) == calibration.calibration_table(
        [_record(1.4), _record(1.4), _record(1.4)])


def test_a_store_already_poisoned_heals():
    """The store is cross-project and its sums only ever accumulate: one NaN was for good."""
    calibration.update_global_store([_record(1.4)])
    path = calibration._store_path()
    store = json.loads(path.read_text())
    key = next(iter(store))
    store[key] = {"count": 3, "error_sum": float("nan"), "abs_sum": float("nan")}
    store["comp|loudness|"] = {"count": 2, "error_sum": 0.6, "abs_sum": 0.6}
    path.write_text(json.dumps(store))

    rows = calibration.load_global_rows()
    assert [r["action_key"] for r in rows] == ["comp"], "the poisoned row is dropped, the good one kept"
    calibration.update_global_store([_record(1.4)])
    healed = json.loads(path.read_text())
    assert healed[key]["count"] == 1 and math.isfinite(healed[key]["error_sum"])
    proposal = {"actions": [{"type": "eq", "stem_id": "vox"}],
                "predicted_impact": [{"metric": "master.band_energy.presence",
                                      "predicted_delta": 1.0, "unit": "dB"}]}
    calibration.calibrate_proposal(proposal, calibration.load_global_rows(), {"vox": ["vocal"]})
    assert math.isfinite(proposal["predicted_impact"][0]["predicted_delta"])
