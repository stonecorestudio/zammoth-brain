"""The compressor's characters (2026-10-25; James's picks of the black-box pass).

The `comp` device was one one-pole on the gain reduction with literal time constants. It is now six
CHARACTERS of one law (`dsp.COMP_CHARACTERS`: clean, punch, smooth, glue, vocal, mastering) - a FAST
stage (two attack followers, the slower held to within a gap of the quicker) and a slow BODY, each
stage on the dB or on the linear gain, over a static law that is the textbook one or a measured
table (`comp_tables`). Three engines run it: `dsp._character_comp` (fast, through
`_gr_ballistics_db`), `reference_monitor._compress` (the float64 loop) and
`MonitorChain::processOneComp` (per sample; the fixture in test_comp_native_parity.py and the leg
`--comp-clean` hold it). These tests hold the tables at both ends, the fast path to the loop, and
the behaviour each character was chosen for - CLEAN's two are red on the one-pole law.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from mix_engine import comp_tables, dsp, models, reference_monitor

SR = 48000
REPO = Path(__file__).resolve().parents[2]


def test_the_characters_are_one_table_in_both_languages():
    header = (REPO / "native/Source/MonitorChain.h").read_text()
    body = re.search(r"kCompCharacters\[kNumCompStyles\]\[10\] = \{(.*?)\n    \};", header, re.S).group(1)
    rows = [[float(v) for v in re.findall(r"(-?[0-9.]+)f", line)] for line in body.splitlines() if "{" in line]
    assert len(rows) == len(models.COMP_STYLES) == len(dsp.COMP_CHARACTERS)
    tables = ("smooth", "glue", "vocal")
    static = [int(v) for v in re.search(r"kCompStatic\[kNumCompStyles\] = \{([^}]*)\}", header).group(1).split(",")]
    for i, row in enumerate(rows):
        py = dsp.COMP_CHARACTERS[i]
        assert row[:7] == pytest.approx(list(py[:7]), abs=0.0), models.COMP_STYLES[i]
        assert row[7:] == [float(py[7]), float(py[8]), pytest.approx(py[10], abs=0.0)], models.COMP_STYLES[i]
        assert static[i] == (-1 if py[9] is None else -2 if py[9] == "clean_knee" else tables.index(py[9])), \
            models.COMP_STYLES[i]
    assert comp_tables.TABLE_ORDER == tables
    face = (REPO / "native/Source/DeviceGraphs.h").read_text()   # face A's word row (2026-10-26)
    words = re.search(r"compCharacterWords\(\).*?static const StringArray n \{([^}]*)\}", face, re.S).group(1)
    assert [w.strip().strip('"').lower() for w in words.split(",")] == list(models.COMP_STYLES)


def test_the_cpp_static_tables_are_the_python_ones():
    assert (REPO / "native/Source/CompTables.h").read_text() == comp_tables.header_text()


def test_the_static_law_reads_the_table_on_its_grid_and_is_textbook_where_it_should_be():
    for name, table in comp_tables.TABLES.items():
        style = list(dsp.COMP_STYLES).index(name)
        for i, thr in enumerate(comp_tables.THRESHOLDS_DB):
            for j, ratio in enumerate(comp_tables.RATIOS):
                if ratio <= 1.0:
                    continue
                got = dsp.comp_static_db(np.array(comp_tables.OVERS_DB), thr, ratio, style)
                assert got == pytest.approx(np.minimum(table[i][j], 0.0), abs=1e-9)
    over = np.array([-10.0, 0.0, 6.0, 30.0])
    for style in (0, 1, 5):
        assert dsp.comp_static_db(over, -20.0, 4.0, style) == pytest.approx([0.0, 0.0, -4.5, -22.5])


def _material(sr: int) -> np.ndarray:
    rng = np.random.default_rng(3)
    t = np.arange(sr) / sr
    x = (np.sin(2 * np.pi * 110.0 * t) * (0.2 + 0.8 * (np.sin(2 * np.pi * 3.0 * t) > 0))
         + 0.1 * rng.standard_normal(sr))
    return np.ascontiguousarray(np.stack([x, 0.7 * x]).astype(np.float32))


@pytest.mark.parametrize("sr", [44100, 48000, 96000])
@pytest.mark.parametrize("attack_ms,release_ms", [(0.1, 10.0), (1.0, 50.0), (10.0, 100.0), (100.0, 1000.0)])
def test_the_fast_path_is_the_loop(sr, attack_ms, release_ms):
    audio = _material(sr)
    entry = {"type": "comp", "threshold_db": -24.0, "ratio": 4.0, "attack_ms": attack_ms,
             "release_ms": release_ms, "makeup_db": 2.0, "mix": 0.8}
    fast = dsp._apply_comp(audio, sr, entry).astype(np.float64)
    loop = reference_monitor._compress(audio.T.astype(np.float64), sr, -24.0, 4.0, attack_ms, release_ms,
                                       makeup_db=2.0, mix=0.8).T
    assert float(np.max(np.abs(fast - loop))) < 2e-4          # three float32 followers in series
    plain = reference_monitor._compress(audio.T.astype(np.float64), sr, -24.0, 4.0, attack_ms, release_ms)
    reduced = 20 * np.log10(np.sqrt(np.mean(plain ** 2)) / np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    assert reduced < -1.0                                       # the material does compress


def _step(hold_s: float, over_db: float = 12.0, thr: float = -20.0):
    """A DC level (the level IS the signal) from -40 dBFS to `over_db` above the threshold."""
    n0 = int(0.2 * SR); nh = int(hold_s * SR); n = n0 + nh + SR
    lv = np.full(n, 10 ** (-40 / 20)); lv[n0:n0 + nh] = 10 ** ((thr + over_db) / 20)
    return np.ascontiguousarray(np.stack([lv, lv]).astype(np.float32)), n0, n0 + nh


def _gr(audio: np.ndarray, **knobs) -> np.ndarray:
    entry = {"type": "comp", "threshold_db": -20.0, "ratio": 4.0, "attack_ms": 10.0, "release_ms": 100.0}
    entry.update(knobs)
    out = dsp._apply_comp(audio, SR, entry)
    return 20 * np.log10(out[0].astype(np.float64) / audio[0].astype(np.float64))


def test_a_short_hit_lets_go_before_a_long_squash():
    """The reason for the law: the same knobs let a 20 ms hit go far sooner than a held 2 s over.
    The one-pole law took the same time for both."""
    times = []
    for hold in (0.02, 2.0):
        audio, _, off = _step(hold)
        g = _gr(audio)
        times.append(int(np.argmax(g[off:] > -1.0)) / SR * 1000.0)
    assert times[0] < 100.0 and times[1] > 150.0, times


def test_a_jump_is_caught_within_its_attack_and_the_static_law_holds():
    """At a 10 ms attack the reduction is within 1.5 dB of a 9 dB target inside 10 ms (the one-pole
    took ~18), and the held reduction is the static law exactly: 12 dB over at 4:1 is 9 dB."""
    audio, on, off = _step(1.0)
    g = _gr(audio)
    assert int(np.argmax(g[on:] <= -7.5)) / SR * 1000.0 < 10.0
    assert g[off - 1] == pytest.approx(-9.0, abs=0.02)


@pytest.mark.parametrize("style", range(6))
@pytest.mark.parametrize("attack_ms,release_ms", [(1.0, 50.0), (30.0, 600.0)])
def test_every_character_s_fast_path_is_its_loop(style, attack_ms, release_ms):
    audio = _material(SR)
    entry = {"type": "comp", "threshold_db": -24.0, "ratio": 4.0, "attack_ms": attack_ms,
             "release_ms": release_ms, "style": style}
    fast = dsp._apply_comp(audio, SR, entry).astype(np.float64)
    loop = reference_monitor._compress(audio.T.astype(np.float64), SR, -24.0, 4.0, attack_ms, release_ms,
                                       style=style).T
    assert float(np.max(np.abs(fast - loop))) < 2e-4


@pytest.mark.parametrize("style", range(6))
def test_a_ratio_of_one_is_off_in_every_character(style):
    audio = _material(SR)
    entry = {"type": "comp", "threshold_db": -40.0, "ratio": 1.0, "attack_ms": 10.0, "release_ms": 100.0,
             "style": style}
    assert dsp._apply_comp(audio, SR, entry) is audio


def test_every_character_is_heard():
    """Each character moves the gain by more than half a dB RMS against CLEAN, at CLEAN's knobs."""
    audio = _material(SR)
    level = 20 * np.log10(np.maximum(np.max(np.abs(audio.astype(np.float64)), axis=0), 1e-12))
    gains = []
    for style in range(6):
        entry = {"type": "comp", "threshold_db": -24.0, "ratio": 4.0, "attack_ms": 10.0, "release_ms": 100.0,
                 "style": style}
        out = dsp._apply_comp(audio, SR, entry).astype(np.float64)
        gains.append(20 * np.log10(np.maximum(np.abs(out[0]), 1e-12) / np.maximum(np.abs(audio[0]), 1e-12))[level > -40])
    for style in range(1, 6):
        assert float(np.sqrt(np.mean((gains[style] - gains[0]) ** 2))) > 0.5, models.COMP_STYLES[style]


@pytest.mark.parametrize("style", range(6))
def test_a_knee_of_zero_is_the_hard_law_bit_for_bit(style):
    """Every comp saved before the knee existed plays as it did."""
    audio = _material(SR)
    entry = {"type": "comp", "threshold_db": -24.0, "ratio": 4.0, "attack_ms": 10.0, "release_ms": 100.0,
             "style": style}
    assert np.array_equal(dsp._apply_comp(audio, SR, entry), dsp._apply_comp(audio, SR, {**entry, "knee_db": 0.0}))


def test_the_knee_bends_every_character_but_vocal():
    over = np.arange(-30.0, 31.0, 1.0)
    for style in range(6):
        hard = dsp.comp_static_db(over, -20.0, 4.0, style, 0.0)
        soft = dsp.comp_static_db(over, -20.0, 4.0, style, 24.0)
        moved = float(np.max(np.abs(soft - hard)))
        if dsp.COMP_STYLES[style] == "vocal":
            assert moved == 0.0                                    # the reference's VOCAL has no knee
        else:
            assert moved > 0.5, dsp.COMP_STYLES[style]


def test_punch_and_mastering_bend_on_the_exact_quadratic_knee():
    over = np.array([-12.0, -8.0, 0.0, 8.0, 12.0, 20.0])
    want = np.array([0.0, (4.0 ** 2) / 48.0, 3.0, (20.0 ** 2) / 48.0, 12.0, 20.0]) * -0.75
    for style in (1, 5):
        assert dsp.comp_static_db(over, -20.0, 4.0, style, 24.0) == pytest.approx(want, abs=1e-12)


def test_clean_s_measured_knee_goes_to_the_hard_law_and_never_boosts():
    over = np.arange(-40.0, 41.0, 0.5)
    hard = np.maximum(over, 0.0)
    assert comp_tables.clean_knee(over, 0.0) == pytest.approx(hard, abs=0.02)
    for knee in (3.0, 10.0, 36.0, 72.0):
        k = comp_tables.clean_knee(over, knee)
        assert np.all(k >= 0.0) and np.all(np.diff(k) >= -1e-9)   # a level higher is never pulled less


@pytest.mark.parametrize("style,knee", [(0, 12.0), (1, 6.0), (2, 24.0), (3, 48.0), (5, 18.0)])
def test_a_kneed_character_s_fast_path_is_its_loop(style, knee):
    audio = _material(SR)
    entry = {"type": "comp", "threshold_db": -24.0, "ratio": 4.0, "attack_ms": 10.0, "release_ms": 100.0,
             "style": style, "knee_db": knee}
    fast = dsp._apply_comp(audio, SR, entry).astype(np.float64)
    loop = reference_monitor._compress(audio.T.astype(np.float64), SR, -24.0, 4.0, 10.0, 100.0,
                                       style=style, knee_db=knee).T
    assert float(np.max(np.abs(fast - loop))) < 2e-4


def test_a_character_is_a_name_or_an_index_and_nothing_else():
    base = dict(stem_id="drums", threshold_db=-20.0, ratio=4.0, attack_ms=10.0, release_ms=100.0)
    assert models.CompAction(**base, style="Glue").style == 3
    assert models.CompAction(**base, style=5).style == 5
    with pytest.raises(ValueError):
        models.CompAction(**base, style="fat")


def test_a_restart_and_silence_change_nothing_under_the_threshold():
    audio = _material(SR)
    quiet = (audio * np.float32(1e-4)).astype(np.float32)
    entry = {"type": "comp", "threshold_db": -24.0, "ratio": 4.0, "attack_ms": 10.0, "release_ms": 100.0}
    assert np.array_equal(dsp._apply_comp(quiet, SR, entry), quiet)
    assert dsp._apply_comp(audio, SR, {**entry, "ratio": 1.0}) is audio
