"""The compressor characters' measured static laws (2026-10-25).

The reduction a character asks for, by threshold, ratio and level over the threshold, for the three
characters whose law is not the textbook one - SMOOTH, GLUE and VOCAL. (CLEAN, PUNCH and MASTERING
are textbook: `-(over)(1 - 1/ratio)` above the threshold, nothing below.) Measured as a black box on
a reference compressor: a square wave held at each level, the settled gain read off, on the grid
below, every number rounded to 0.01 dB (`comp_tables_data.TABLES`), and checked at random points off
the grid to 0.04-0.13 dB RMS. The record is docs/10-daw.md.

`native/Source/CompTables.h` is this table, written by `python -m mix_engine.comp_tables
--write-header` and held to it by test_comp_characters.py: change the numbers here, never there.

`static_row(character, threshold_db, ratio, knee_db)` blends the grid rows around a setting into ONE
row over the level axis (the threshold held to -60..0, the ratio read as 1 - 1/ratio and held to
1..10, the knee held to 0..72 where the character has a knee axis); `MonitorChain` blends the same
row once a block. Both then read a sample's level
off that row (`row_gain_db`): linear between grid points, the last step's slope past the top, the
first value held below the bottom.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .comp_tables_data import CLEAN_KNEE, KNEE_TABLES, TABLES

THRESHOLDS_DB = (-60.0, -50.0, -40.0, -30.0, -20.0, -12.0, -6.0, 0.0)
RATIOS = (1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0)
OVERS_DB = tuple(float(v) for v in range(-36, 37, 2))   # the level over the threshold
OVER_STEP_DB = 2.0
TABLE_ORDER = ("smooth", "glue", "vocal")                # kTable's first index
# The KNEE axis (2026-10-25): SMOOTH's and GLUE's laws change with the knee, measured at these widths
# (KNEE_TABLES[name][i] is KNEES_DB[i + 1]; the 0 dB plane is TABLES[name]); VOCAL's does not (the
# reference ignores the knee there, as it ignores the ratio). Read linearly between the widths.
KNEES_DB = (0.0, 3.0, 6.0, 12.0, 24.0, 48.0, 72.0)
KNEE_ORDER = ("smooth", "glue")                          # kKneeTable's first index
CLEAN_KNEE_OVERS_DB = tuple(float(v) for v in range(-40, 41, 2))   # CLEAN_KNEE's level axis


def _cell(axis: tuple, v: float) -> tuple[int, float]:
    """The grid cell a value sits in and how far across it (held at the ends)."""
    v = min(max(v, axis[0]), axis[-1])
    k = 0
    while k < len(axis) - 2 and v >= axis[k + 1]:
        k += 1
    return k, (v - axis[k]) / (axis[k + 1] - axis[k])


def _plane(character: str, knee_index: int):
    return TABLES[character] if knee_index == 0 else KNEE_TABLES[character][knee_index - 1]


def _plane_row(t, threshold_db: float, ratio: float) -> np.ndarray:
    i, fi = _cell(THRESHOLDS_DB, float(threshold_db))
    j, fj = _cell(tuple(1.0 - 1.0 / r for r in RATIOS), 1.0 - 1.0 / max(float(ratio), 1.0))
    def r(a: int, b: int) -> np.ndarray:
        return np.asarray(t[a][b], dtype=np.float64)
    return ((1.0 - fi) * ((1.0 - fj) * r(i, j) + fj * r(i, j + 1))
            + fi * ((1.0 - fj) * r(i + 1, j) + fj * r(i + 1, j + 1)))


def static_row(character: str, threshold_db: float, ratio: float, knee_db: float = 0.0) -> np.ndarray:
    """The table's row over OVERS_DB at this threshold, ratio and knee (bilinear in the threshold
    and in 1 - 1/ratio, then linear in the knee where the character has a knee axis)."""
    if character not in KNEE_TABLES:
        return _plane_row(TABLES[character], threshold_db, ratio)
    k, fk = _cell(KNEES_DB, float(knee_db))
    lo = _plane_row(_plane(character, k), threshold_db, ratio)
    if fk <= 0.0:
        return lo
    return (1.0 - fk) * lo + fk * _plane_row(_plane(character, k + 1), threshold_db, ratio)


def clean_knee(over_db: np.ndarray, knee_db: float) -> np.ndarray:
    """CLEAN's knee: the level over the threshold its slope acts on (>= 0), linear in the knee
    between KNEES_DB and in the level between CLEAN_KNEE_OVERS_DB (the top step's slope above, 0
    below). `MonitorChain::compCleanKnee` takes the same steps."""
    over = np.asarray(over_db, dtype=np.float64)
    k, fk = _cell(KNEES_DB, float(knee_db))
    row = (1.0 - fk) * np.asarray(CLEAN_KNEE[k], dtype=np.float64) + fk * np.asarray(CLEAN_KNEE[k + 1], dtype=np.float64)
    lo, step = CLEAN_KNEE_OVERS_DB[0], 2.0
    pos = (over - lo) / step
    i = np.clip(np.floor(pos), 0, len(CLEAN_KNEE_OVERS_DB) - 2).astype(np.int64)
    f = np.where(pos < 0.0, 0.0, pos - i)
    return np.maximum(row[i] + f * (row[i + 1] - row[i]), 0.0)


def row_gain_db(row: np.ndarray, over_db: np.ndarray) -> np.ndarray:
    """Each level's reduction off a blended row (`MonitorChain::compRowGainDb` takes the same
    steps): linear between grid points, the top step's slope above, the first value below."""
    over = np.asarray(over_db, dtype=np.float64)
    pos = (over - OVERS_DB[0]) / OVER_STEP_DB
    k = np.clip(np.floor(pos), 0, len(OVERS_DB) - 2).astype(np.int64)
    f = np.where(pos < 0.0, 0.0, pos - k)          # below the grid: the first value, held
    y = row[k] + f * (row[k + 1] - row[k])         # above the top: the last step carried on
    return np.minimum(y, 0.0)


def header_text() -> str:
    """native/Source/CompTables.h, from TABLES."""
    def nums(vals, fmt):
        return ", ".join((fmt % v) + "f" for v in vals)
    out = ["#pragma once",
           "// GENERATED by `python -m mix_engine.comp_tables --write-header` from",
           "// server/mix_engine/comp_tables_data.py - the compressor characters' measured static laws",
           "// (comp_tables.py says how they were made and read). Do not edit: change the Python table and",
           "// write this again; test_comp_characters.py holds the two equal.",
           "",
           "namespace comptables",
           "{",
           "    static constexpr int kNumThr = %d, kNumRatio = %d, kNumOver = %d;"
           % (len(THRESHOLDS_DB), len(RATIOS), len(OVERS_DB)),
           "    static constexpr float kThr[kNumThr] = { %s };" % nums(THRESHOLDS_DB, "%.1f"),
           "    static constexpr float kRatio[kNumRatio] = { %s };" % nums(RATIOS, "%.2f"),
           "    static constexpr float kOverLo = %.1ff, kOverStep = %.1ff;" % (OVERS_DB[0], OVER_STEP_DB),
           "    static constexpr int kNumKnee = %d;" % len(KNEES_DB),
           "    static constexpr float kKnee[kNumKnee] = { %s };" % nums(KNEES_DB, "%.1f"),
           "    // [table: 0 SMOOTH, 1 GLUE, 2 VOCAL][threshold][ratio][over] - the knee-0 planes",
           "    static constexpr float kTable[3][kNumThr][kNumRatio][kNumOver] = {"]
    for name in TABLE_ORDER:
        out.append("        {   // " + name)
        for plane in TABLES[name]:
            out.append("            {")
            for row in plane:
                out.append("                { %s }," % nums(row, "%.2f"))
            out.append("            },")
        out.append("        },")
    out += ["    };",
            "    // [knee table: 0 SMOOTH, 1 GLUE][knee: kKnee[1..]][threshold][ratio][over]",
            "    static constexpr float kKneeTable[2][kNumKnee - 1][kNumThr][kNumRatio][kNumOver] = {"]
    for name in KNEE_ORDER:
        out.append("        {   // " + name)
        for knee_plane in KNEE_TABLES[name]:
            out.append("            {")
            for plane in knee_plane:
                out.append("                {")
                for row in plane:
                    out.append("                    { %s }," % nums(row, "%.2f"))
                out.append("                },")
            out.append("            },")
        out.append("        },")
    out += ["    };",
            "    // CLEAN's knee: the level over the threshold its slope acts on, [knee: kKnee][over]",
            "    static constexpr float kCleanKneeOverLo = %.1ff;" % CLEAN_KNEE_OVERS_DB[0],
            "    static constexpr int kNumCleanKneeOver = %d;" % len(CLEAN_KNEE_OVERS_DB),
            "    static constexpr float kCleanKnee[kNumKnee][kNumCleanKneeOver] = {"]
    for row in CLEAN_KNEE:
        out.append("        { %s }," % ", ".join(("%.3f" % v) + "f" for v in row))
    out += ["    };", "}", ""]
    return "\n".join(out)


if __name__ == "__main__" and "--write-header" in sys.argv:
    path = Path(__file__).resolve().parents[2] / "native" / "Source" / "CompTables.h"
    path.write_text(header_text())
    print("wrote", path)
