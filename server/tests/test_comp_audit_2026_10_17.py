"""The compressor's audit, 2026-10-17 (the device-by-device audit) - the offline engine and the brain.

The native half is `--comp-audit`. Here:
  * a Mid or Side comp with only one half stored: the half the comp is set to takes the knobs as
    they stand, the other passes through (1:1, no makeup) - the rule the native parse now shares;
    a lone `side{}` (a dock knob's first turn on a Side comp) used to be ignored here;
  * render telemetry credits a comp's makeup THROUGH ITS MIX, so a parallel comp that never
    reaches its threshold reports no reduction (it read 4 dB, 100 % of the time);
  * a flip the DAW seeded (`seed=1`) re-seeds the brain's halves from the knobs as they stand,
    and the half not chosen is seeded with no makeup.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mix_engine import analysis, dsp
from mix_engine.session import Session

SR = 48000


def _stereo(seconds: float = 1.0, mid_amp: float = 0.3, side_amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    m = mid_amp * np.sin(2 * np.pi * 200.0 * t)
    s = side_amp * np.sin(2 * np.pi * 330.0 * t)
    return np.stack([m + s, m - s]).astype(np.float32)


def _halves(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (audio[0] + audio[1]) * 0.5, (audio[0] - audio[1]) * 0.5


def _rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x[len(x) // 2:]))) + 1e-12))


BASE = {"type": "comp", "threshold_db": -24.0, "ratio": 4.0, "attack_ms": 5.0,
        "release_ms": 80.0, "makeup_db": 6.0, "mix": 1.0}


def test_a_side_comp_with_only_its_side_half_compresses_the_side_with_it_and_leaves_the_mid():
    dry = _stereo()
    dry_mid, dry_side = _halves(dry)
    edited = dsp._apply_comp(dry, SR, {**BASE, "channel": "side", "side": {"threshold_db": -40.0}})
    flat = dsp._apply_comp(dry, SR, {**BASE, "channel": "side"})
    mid, side = _halves(edited)
    assert np.max(np.abs(mid - dry_mid)) < 1e-5            # the mid passes through, makeup and all
    # the edit is HEARD: the side is compressed with the side's own threshold, not the flat one
    assert abs(_rms_db(side) - _rms_db(_halves(flat)[1])) > 1.0


def test_a_mid_comp_with_only_its_mid_half_leaves_the_side_without_the_makeup():
    dry = _stereo()
    _, dry_side = _halves(dry)
    out = dsp._apply_comp(dry, SR, {**BASE, "channel": "mid", "mid": {"threshold_db": -30.0}})
    _, side = _halves(out)
    assert np.max(np.abs(side - dry_side)) < 1e-5


def test_a_parallel_comp_that_never_reaches_its_threshold_reports_no_reduction(demo_project: Path):
    session = Session(demo_project)
    session.stems["stem_b"]["chain"].append(
        {"type": "comp", "threshold_db": 0.0, "ratio": 4.0, "attack_ms": 5.0,
         "release_ms": 100.0, "makeup_db": 12.0, "mix": 0.5})
    telemetry: list = []
    dsp.render_stem(session, "stem_b", telemetry=telemetry)
    (stats,) = telemetry
    assert stats["gr_mean_db"] < 0.5, stats
    assert not analysis.over_compressed(None, {"stem_b": telemetry})


def test_a_flip_the_daw_seeded_reseeds_the_brains_halves(demo_project: Path):
    session = Session(demo_project)
    session.stems["stem_b"]["chain"].append(dict(BASE, threshold_db=-18.0, ratio=2.0))
    session.set_device_channel("comp", 0, "mid", target="stem_b")
    session.set_device_channel("comp", 0, "stereo", target="stem_b")   # what an undo leaves here
    entry = [e for e in session.stems["stem_b"]["chain"] if e["type"] == "comp"][0]
    entry.update(threshold_db=-30.0, ratio=4.0)                          # the knobs moved since
    session.set_device_channel("comp", 0, "mid", target="stem_b", seed=True)
    entry = [e for e in session.stems["stem_b"]["chain"] if e["type"] == "comp"][0]
    assert entry["mid"]["threshold_db"] == -30.0 and entry["mid"]["ratio"] == 4.0
    assert entry["side"]["ratio"] == 1.0 and entry["side"]["makeup_db"] == 0.0
