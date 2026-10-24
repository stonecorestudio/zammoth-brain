"""The saturation's audit, 2026-10-17 (the device-by-device audit) - the offline engine and the text.

The native half is `--sat-audit` (restart from rest, the dry aligned at every mix, meters at rest,
each curve against values pinned from `dsp._waveshape`). Here:
  * every value is held to the table and the oversample snapped down to 1/2/4/8, as the native's
    parse does - a mix outside 0..1 or an oversample of 3 rendered otherwise offline;
  * no text calls tube's NEGATIVE half the hard one: it is the softer, louder one (atan), and the
    tool text says its negative peaks come out louder and it leaves a small DC offset.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mix_engine import dsp

ROOT = Path(__file__).resolve().parents[2]


def _x():
    t = np.arange(4800) / 48000.0
    return np.stack([0.4 * np.sin(2 * np.pi * 220 * t)] * 2, axis=1)


def test_every_value_is_held_to_the_table():
    x = _x()
    at_end = dsp._saturate(x, {"drive_db": 12.0, "output_db": 12.0, "mix": 1.0, "curve": 2}, 48000)
    past = dsp._saturate(x, {"drive_db": 30.0, "output_db": 30.0, "mix": 1.7, "curve": 2}, 48000)
    assert np.max(np.abs(np.asarray(past) - np.asarray(at_end))) < 1e-9


def test_an_oversample_between_factors_snaps_down():
    x = _x()
    two = dsp._saturate(x, {"drive_db": 12.0, "oversample": 2, "curve": 3}, 48000)
    three = dsp._saturate(x, {"drive_db": 12.0, "oversample": 3, "curve": 3}, 48000)
    assert np.max(np.abs(np.asarray(three) - np.asarray(two))) < 1e-9


def test_no_text_calls_tubes_negative_half_the_hard_one():
    for rel in ("server/mix_engine/dsp.py", "server/mix_engine/reference_monitor.py"):
        assert "+ soft / − hard" not in (ROOT / rel).read_text(), rel
    for rel in ("orchestrator/prompts.py", "orchestrator/chat.py", "orchestrator/agents.py"):
        assert "DC offset" in (ROOT / rel).read_text(), rel
