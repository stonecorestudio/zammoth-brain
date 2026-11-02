"""The limiter's audit, 2026-10-17 (the device-by-device audit) - the offline engine and the brain.

`dsp._brickwall` was held here to a per-sample loop of the native law (v2); since the two-stage
limiter (2026-10-25) that loop is test_limiter_two_stage.py's. What stays here: a style write
applies its knobs, the native's table is the model's, the loudness target's two doors.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

from mix_engine import dsp
from mix_engine.models import LIMITER_STYLE_PRESETS
from mix_engine.session import Session
from mix_engine.tp_detector import TP_CENTRE, intersample_peak

SR = 48000
ROOT = Path(__file__).resolve().parents[2]


def test_a_style_write_applies_its_preset(demo_project):
    s = Session(demo_project)
    s.add_master_device("limiter")
    s.set_master_device_param("limiter", 0, "style", 1.4)
    lim = next(e for e in s._state["master_chain"] if e["type"] == "limiter")
    assert lim["style"] == 1.0
    assert (lim["release_ms"], lim["lookahead_ms"]) == LIMITER_STYLE_PRESETS[1]


def test_the_native_style_table_is_the_models():
    src = (ROOT / "native/Source/DeviceSpecs.h").read_text()
    row = re.search(r"kLimiterStylePresets\[3\]\[2\] = \{(.*?)\};", src).group(1)
    pairs = [tuple(float(v) for v in p.split(",")) for p in re.findall(r"\{\s*([^{}]+?)\s*\}", row)]
    assert pairs == [LIMITER_STYLE_PRESETS[i] for i in range(3)]


def test_a_loudness_target_off_the_master_is_refused(demo_project):
    s = Session(demo_project)
    s.add_stem_device("stem_b", "limiter")
    with pytest.raises(ValueError, match="master"):
        s.set_master_device_param("limiter", 0, "target_lufs_i", -10.0, target="stem_b")


def test_an_unheard_mix_says_the_loudness_target_can_be_missed(demo_project):
    from orchestrator import daw_bridge
    s = Session(demo_project)
    s.add_master_device("limiter")
    lim = next(e for e in s._state["master_chain"] if e["type"] == "limiter")
    lim["target_lufs_i"] = -12.0
    s._persist()
    edit = {"tracks": [{"name": "Keys", "track_kind": "midi", "clips": [{}]}]}
    out = daw_bridge.with_blind_spots("analyze_session", {}, {"lufs_i": -14.0}, edit, demo_project)
    assert any(b["what"] == "loudness_target" for b in out["not_heard"])
