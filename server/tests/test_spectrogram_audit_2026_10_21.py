"""The Spectrogram's audit, 2026-10-21 (the device-by-device audit; James: "read at its place").

The native half is `--spectrogram-audit`. Here what a leg cannot see:
  * a chain gives at most FOUR Spectrograms a tap of their own (`kMaxSpectrograms`); the brain
    refuses a fifth at that number, and so does the dock (`capacityRefusal`);
  * `track.spectrogram` says which audio it read (`tap`), how much of the window held sound
    (`active_s`), and its tilt under the name of what it is (`band_tilt_db_per_oct` - a slope through
    band POWERS, not the brain's `spectral_tilt_db_per_oct`); the grammar line, both prompts and the
    operate eval's canned reply say the same (SG7);
  * the history's readers size their copy OUTSIDE the lock the meter thread takes for every column
    (SG8);
  * the face's three switches are kept with the window's layout (SG6 = 07-F7);
  * the face's ruler places each column from its own transport time - nothing extrapolates a bar
    from one tempo any more (SG4).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from mix_engine.session import _NATIVE_SLOT_KINDS, Session

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "daw" / "Source"
NATIVE = ROOT / "native" / "Source"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _handler() -> str:
    app = _read(SRC / "App.cpp")
    start = app.index("            case dawcmd::trackSpectrogram:")
    return app[start:app.index("\n            default: break;", start)]


def test_the_brain_and_the_dock_cap_spectrograms_at_the_natives_count():
    header = _read(NATIVE / "MonitorChain.h")
    k = int(re.search(r"static constexpr int kMaxSpectrograms\s*=\s*(\d+)", header).group(1))
    assert _NATIVE_SLOT_KINDS["spectrogram"] == k
    dock = _read(SRC / "DeviceChainView.h")
    assert 'type == "spectrogram" && have + arriving > MonitorChain::kMaxSpectrograms' in dock


def test_a_fifth_spectrogram_is_refused(demo_project):
    session = Session(demo_project)
    for _ in range(4):
        session.add_stem_device("stem_a", "spectrogram")
    with pytest.raises(ValueError, match="spectrogram"):
        session.add_stem_device("stem_a", "spectrogram")


def test_track_spectrogram_says_what_it_read():
    cmd = _handler()
    for field in ('"tap"', '"active_s"', '"band_tilt_db_per_oct"'):
        assert field in cmd, field
    assert 'setProperty ("tilt_db_per_oct"' not in cmd, "the old name for the band tilt"
    assert "node->readBandHistory (sgOcc," in cmd, "it reads the track's Spectrogram where it stands"
    assert "w >= 1.0 && w <= 60.0" in cmd, "a window outside the history is refused, not clamped"
    cmds = _read(SRC / "DawCommands.h")
    line = next(l for l in cmds.splitlines() if l.strip().startswith('{ "track.spectrogram",'))
    assert "1..60" in line and "band_tilt_db_per_oct" in line and "`tap`" in line
    canned = _read(ROOT / "orchestrator" / "operate_eval.py")
    block = canned[canned.index('    "track.spectrogram": {'):]
    block = block[:block.index("},\n    #") if "},\n    #" in block else 2000]
    assert '"band_tilt_db_per_oct"' in block and '"tap"' in block


def test_both_prompts_say_where_the_spectrogram_reads():
    for p in (ROOT / "orchestrator" / "chat.py", ROOT / "orchestrator" / "prompts.py"):
        text = _read(p).replace("\\\n", "")
        assert "drawn from the track's live meters" not in text, p.name
        assert "where it stands in the chain, before the fader and the pan" in text, p.name


def test_the_history_is_sized_outside_the_meter_threads_lock():
    cpp = _read(NATIVE / "LiveMeters.cpp")
    for fn in ("juce::int64 LiveMeters::readSpectrogram", "int LiveMeters::readBandHistory"):
        body = cpp[cpp.index(fn):]
        body = body[:body.index("\n}\n")]
        reserve = body.index(".reserve (")
        held = body.index("const SpinLock::ScopedLockType sg (sgLock);\n    const")
        assert reserve < held, f"{fn} reserves under the lock"


def test_the_faces_switches_are_kept_with_the_layout():
    ui = _read(SRC / "UiState.h")
    for name in ("sgPicture", "sgUnder", "sgFull"):
        assert f'o->setProperty ("{name}", s.{name});' in ui
        assert f'root.getProperty ("{name}", -1)' in ui
    app = _read(SRC / "App.cpp")
    assert "s.sgPicture = sgPicture; s.sgUnder = sgUnder; s.sgFull = sgFull ? 1 : 0;" in app
    assert "if (s.sgPicture >= 0) sgPicture" in app


def test_the_ruler_places_columns_by_their_own_time():
    graphs = _read(NATIVE / "DeviceGraphs.h")
    sg = graphs[graphs.index("class SpectrogramGraph"):]
    sg = sg[:sg.index("\n} // namespace devicegraphs")]
    assert "songBpm" not in sg and "setTimeline" not in sg and "colTime" not in sg
    assert "colBar" in sg and "colBeat" in sg
    dock = _read(SRC / "DeviceChainView.h")
    assert "owner.host.readSpectrogram (occ, sgSince, sgPull, g->enhanced(), &sgTimes)" in dock
    assert "owner.barBeatAt (sgTimes[(size_t) c])" in dock
    assert "dawsession::barsBeatsFromOrigin (*edit, t, (double) edit->state.getProperty (\"mixagentGridOriginS\", 0.0))" in _read(SRC / "App.cpp")


def test_the_face_reads_a_key_the_way_the_brain_does():
    graphs = _read(NATIVE / "DeviceGraphs.h")
    fn = graphs[graphs.index("    String keyEstimate() const"):]
    fn = fn[:fn.index("\n    }\n")]
    assert "pearson (s, prof, 12)" in fn
    # the brain's own thresholds, held to the face's
    from mix_engine import analysis as A
    pairs = {"kKeyMinConfidence": A.KEY_MIN_CONFIDENCE, "kKeyMinRest": A.KEY_MIN_REST_CORRELATION,
             "kKeyPcShare": A.KEY_PITCH_CLASS_SHARE, "kKeyRelativeMargin": A.KEY_RELATIVE_MARGIN,
             "kKeyBassShare": A.KEY_BASS_SHARE, "kKeyBassRatio": A.KEY_BASS_RATIO,
             "kKeyMinPitchClasses": A.KEY_MIN_PITCH_CLASSES}
    for name, want in pairs.items():
        m = re.search(name + r"\s*=\s*([0-9.]+)", graphs)
        assert m and float(m.group(1)) == pytest.approx(want), name
