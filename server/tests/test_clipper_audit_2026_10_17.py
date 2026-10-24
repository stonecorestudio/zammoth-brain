"""The clipper's audit, 2026-10-17 (the device-by-device audit) - the offline engine and the text.

The native half is `--clipper-audit`. Here:
  * a threshold or an output trim stored past the table plays at the table's end in both engines
    (the native parse took them raw);
  * no text the agent reads calls the oversampled clipper "true-peak-safe" or says a ceiling holds
    after it on any chain: its decimation filter rings past the threshold (a drum train at 4x came
    out ~3 dB over), and on a stem or bus no limiter follows it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mix_engine import dsp

ROOT = Path(__file__).resolve().parents[2]


def test_a_value_past_the_table_plays_at_its_end():
    t = np.arange(4800) / 48000.0
    x = np.stack([0.5 * np.sin(2 * np.pi * 440 * t)] * 2, axis=1)
    at_end = dsp._clip(x, {"threshold_db": -12.0, "hardness": 1.0, "output_db": 12.0})
    past = dsp._clip(x, {"threshold_db": -30.0, "hardness": 1.0, "output_db": 30.0})
    assert np.max(np.abs(np.asarray(past) - np.asarray(at_end))) < 1e-9


def test_no_text_calls_the_oversampled_clipper_a_ceiling():
    for rel in ("server/mix_engine/models.py", "orchestrator/prompts.py", "orchestrator/chat.py",
                "orchestrator/agents.py", "server/mix_engine/dsp.py"):
        text = (ROOT / rel).read_text()
        assert "true-peak-safe" not in text, rel
        assert "It sits last in whichever chain it is on" not in text.replace("\\\n", ""), rel
