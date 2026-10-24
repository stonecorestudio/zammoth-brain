"""The limiter's inter-sample detector is ONE table in two engines (reader E4·R7, batch T).

The native limiter reads a peak through a 4x polyphase FIR (`kTpPhase`, 33 taps a phase,
native/Source/MonitorChain.cpp); the offline limiter kept the Catmull-Rom cubic the native
replaced on 2026-09-08. The cubic under-reads content near the top of the band, so the offline
brickwall let a peak through and `_master_limiter`'s whole-file trim then turned the entire
master down - a level the DAW's export never has.

`mix_engine/tp_detector.py` holds the Python copy. A second copy is a second thing to drift, so
this re-parses the C++ rows and requires them to BE the Python table (the device clamps' rule,
test_device_clamp_parity.py).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).resolve().parents[2] / "native" / "Source" / "MonitorChain.cpp"


def _cpp_table() -> tuple[np.ndarray, int]:
    text = SOURCE.read_text(encoding="utf-8")
    start = text.index("static constexpr float kTpPhase[3][kTpTapsN] = {")
    body = text[start:text.index("};", start)]
    rows = re.findall(r"\{([^{}]+)\}", body[body.index("{") + 1:])
    table = np.array([[float(v.rstrip("f")) for v in re.findall(r"[-+]?\d+\.\d+f", row)]
                      for row in rows])
    centre = int(re.search(r"static constexpr int kTpCentre = (\d+);", text).group(1))
    return table, centre


def test_the_python_table_is_the_native_table():
    from mix_engine import tp_detector

    table, centre = _cpp_table()
    assert table.shape == (3, 33), table.shape            # control: the parse found the rows
    assert tp_detector.TP_PHASES.shape == table.shape
    assert np.array_equal(tp_detector.TP_PHASES, table)
    assert tp_detector.TP_CENTRE == centre


def test_the_parse_reads_the_real_rows():
    """Control for the pin above: the rows are the filter the header describes - each phase sums
    to one, and the middle phase is symmetric."""
    table, _ = _cpp_table()
    assert np.allclose(table.sum(axis=1), 1.0, atol=1e-4)
    assert np.allclose(table[1][:32], table[1][:32][::-1], atol=1e-8)
