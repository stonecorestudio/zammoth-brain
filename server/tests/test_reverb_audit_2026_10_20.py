"""The Reverb audit (2026-10-20, the device-by-device audit's twenty-second device).

The law was already one law (offline and native null at float32 rounding). What this file holds:
  - the offline reverb and the reference monitor hold a stored value to the table, as the native
    and every door do (a stored 1000 ms pre-delay rendered 1 s here and 250 ms in the DAW, and a
    NEGATIVE one raised in both - every render and analysis of that target failed);
  - the texts say what the device does (a bypass silences the tail; the tone is one digital pole);
  - the linear oracle profile carries a reverb insert (coverage).
The native half - the pre-delay ring, the mix glide, the faces - is `--reverb-audit`
(daw/Source/ReverbAuditLegs.h).
"""
from __future__ import annotations

import pathlib

import numpy as np

from mix_engine import dsp
from mix_engine import reference_monitor as rm

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _render(**kw):
    sr = 48000
    hit = np.zeros((sr * 3, 2))
    hit[0] = 1.0
    entry = {"type": "return_reverb", "decay_s": 1.8, "pre_delay_ms": 20.0, "tone": 0.5, "mix": 1.0}
    entry.update(kw)
    return dsp._wet_device(hit, sr, entry), rm._return_reverb(hit, float(sr), entry)


def test_every_parameter_is_held_to_its_table_offline():
    for wild, held in (({"pre_delay_ms": 1000.0}, {"pre_delay_ms": 250.0}),
                       ({"pre_delay_ms": -10.0}, {"pre_delay_ms": 0.0}),
                       ({"decay_s": 20.0}, {"decay_s": 12.0}),
                       ({"decay_s": 0.01}, {"decay_s": 0.2}),
                       ({"tone": 3.0}, {"tone": 1.0})):
        a, ra = _render(**wild)
        b, rb = _render(**held)
        assert np.array_equal(a, b), wild
        assert np.array_equal(ra, rb), wild


def test_the_texts_say_what_the_reverb_does():
    faces = (ROOT / "native" / "Source" / "DeviceGraphs.h").read_text(encoding="utf-8")
    manual = (ROOT / "docs" / "17-user-manual.md").read_text(encoding="utf-8")
    rev = faces[faces.index("class RevFaceGraph"):]
    rev = rev[:rev.index("\n};\n")]
    assert "the tail is left in place" not in rev and "the tail is left in place" not in manual
    assert "1.2 to 18 kHz" not in rev and "TONE cutoff (1.2" not in manual
    assert "kRevWetOnsetDb" not in faces   # a hit's ring is measured, not the hit less 8 dB


def test_the_linear_oracle_carries_a_reverb_insert():
    oracle = (ROOT / "orchestrator" / "daw_export_oracle.py").read_text(encoding="utf-8")
    assert "def _reverb_insert(" in oracle and "_reverb_insert(session)" in oracle
