"""The delay's audit, 2026-10-18 (the device-by-device audit) - the offline engine and the export.

The native half is `--delay-audit`. Here:
  * the offline delay holds its time to the table (20..2000 ms) as every door does - it took the
    stored number raw, so a 5 s time rendered a 5 s echo the DAW's 2 s ring can never play;
  * the reference monitor's delay is held the same way;
  * the export's tail cap covers the longest tail the table allows (2000 ms at feedback 0.85 is 43
    repeats, 86 s) - at 30 s the export cut it.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from mix_engine import dsp, reference_monitor
from mix_engine.session import Session

SR = 8000
REPO = Path(__file__).resolve().parents[2]


def _first_echo_s(render, ms: float) -> float:
    x = np.zeros((SR * 6, 2))
    x[0] = 1.0
    out = render(x, SR, {"delay_ms": ms, "feedback": 0.0, "tone": 1.0})
    return float(np.argmax(np.abs(out[:, 0]))) / SR


def test_the_offline_delay_holds_its_time_to_the_table():
    assert abs(_first_echo_s(dsp._return_delay, 5000.0) - 2.0) < 0.01
    assert abs(_first_echo_s(dsp._return_delay, 5.0) - 0.02) < 0.002


def test_the_reference_monitors_delay_is_held_the_same_way():
    assert abs(_first_echo_s(reference_monitor._return_delay, 5000.0) - 2.0) < 0.01


def test_the_exports_tail_cap_covers_the_longest_delay_the_table_allows():
    params = Session.DEVICE_PARAMS["return_delay"]
    longest_ms, max_fb = params["delay_ms"][2], params["feedback"][2]
    repeats = math.ceil(60.0 / (-20.0 * math.log10(max_fb)))
    need_s = repeats * longest_ms / 1000.0
    text = (REPO / "daw/Source/DawSession.h").read_text(encoding="utf-8")
    cap = float(re.search(r"inline double exportTailSeconds \(te::Edit& edit, double capS = ([0-9.]+)\)", text).group(1))
    assert cap >= need_s, (cap, need_s)
