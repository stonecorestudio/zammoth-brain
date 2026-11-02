"""A compressor CHARACTER sets the knobs it is heard at (2026-10-26, James: "the smooth, punch, and
clean settings are the exact same values"). Picking one writes its ratio, attack, release and knee
- the limiter's rule (models.LIMITER_STYLE_PRESETS) - in the brain and in the native writer, from
ONE table (models.COMP_STYLE_PRESETS = DeviceSpecs.h kCompStylePresets). Threshold, makeup and mix
are the user's and stay. A Mid / Side half takes the timing; its ratio only where it compresses,
so a half at 1:1 (off) is not switched on by a character."""
from __future__ import annotations

import re
from pathlib import Path

from mix_engine.models import COMP_STYLE_PRESETS, COMP_STYLES
from mix_engine.session import Session

ROOT = Path(__file__).resolve().parents[2]
KNOBS = ("ratio", "attack_ms", "release_ms", "knee_db")


def _comp(s: Session, target: str) -> dict:
    return next(e for e in s._coedit_chain(target) if e["type"] == "comp")


def test_every_character_has_its_own_knobs():
    assert sorted(COMP_STYLE_PRESETS) == list(range(len(COMP_STYLES)))
    rows = [COMP_STYLE_PRESETS[i] for i in range(len(COMP_STYLES))]
    assert len(set(rows)) == len(rows), "two characters set the same knobs: the switch reads as broken"
    for ratio, atk, rel, knee in rows:
        assert ratio > 1.0, "a ratio of 1 is off in every character"
        assert 0.1 <= atk <= 100.0 and 10.0 <= rel <= 1000.0 and 0.0 <= knee <= 24.0


def test_a_style_write_sets_the_characters_knobs_and_leaves_the_users(demo_project):
    s = Session(demo_project)
    stem = next(iter(s._state["stems"]))
    s.add_stem_device(stem, "comp")
    s.set_master_device_param("comp", 0, "threshold_db", -31.0, target=stem)
    s.set_master_device_param("comp", 0, "makeup_db", 3.0, target=stem)
    for style in range(len(COMP_STYLES)):
        s.set_master_device_param("comp", 0, "style", float(style), target=stem)
        e = _comp(s, stem)
        assert tuple(e[k] for k in KNOBS) == COMP_STYLE_PRESETS[style], COMP_STYLES[style]
        assert (e["threshold_db"], e["makeup_db"]) == (-31.0, 3.0)


def test_a_half_at_one_to_one_stays_off(demo_project):
    s = Session(demo_project)
    stem = next(iter(s._state["stems"]))
    s.add_stem_device(stem, "comp")
    s.set_device_channel("comp", 0, "mid", target=stem, seed=True)
    s.set_master_device_param("comp", 0, "style", 2.0, target=stem)   # SMOOTH
    e = _comp(s, stem)
    ratio, atk, rel, _knee = COMP_STYLE_PRESETS[2]
    assert (e["mid"]["ratio"], e["mid"]["attack_ms"], e["mid"]["release_ms"]) == (ratio, atk, rel)
    assert e["side"]["ratio"] == 1.0, "the half the user left off was switched on"
    assert (e["side"]["attack_ms"], e["side"]["release_ms"]) == (atk, rel)


def test_the_native_table_is_the_models():
    src = (ROOT / "native/Source/DeviceSpecs.h").read_text()
    body = re.search(r"kCompStylePresets\[6\]\[4\] = \{(.*?)\};", src, re.S).group(1)
    rows = [tuple(float(v) for v in r.split(",")) for r in re.findall(r"\{\s*([^{}]+?)\s*\}", body)]
    assert rows == [COMP_STYLE_PRESETS[i] for i in range(len(COMP_STYLES))]
