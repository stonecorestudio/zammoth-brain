"""The Stereo Width audit (2026-10-21, the device-by-device audit's twenty-fifth device).

The law was already one law (mid/side, mono sum kept). What this file holds, the brain's half:
  - a track's or bus's Width has an on/off of its own in the brain (`width_on`): a bypassed Width box
    keeps its value, the effective chain carries it disabled, the offline render plays 1.0 - and a
    width set with no width_on switches it on (a hand on the width is heard);
  - a stored width outside 0..2 is held to the table offline and in the reference monitor.
The native half (the glide, the rest, the strip, device.enable, the insert order) is `--width-audit`.
"""
from __future__ import annotations

import numpy as np

from mix_engine import dsp
from mix_engine import reference_monitor as rm
from mix_engine.session import Session, effective_chain


def test_a_bypassed_strip_width_keeps_its_value_and_plays_unity(demo_project):
    s = Session(demo_project)
    stem = next(iter(s.stems))
    s.user_edit(stem, {"width": 0.4})
    s.user_edit(stem, {"width_on": False})
    st = s.stems[stem]
    assert st["width"] == 0.4 and st["width_on"] is False
    widths = [e for e in effective_chain(st) if e["type"] == "width"]
    assert widths == [{"type": "width", "width": 0.4, "enabled": False}]
    off, _ = dsp.render_stem(s, stem)
    s.user_edit(stem, {"width": 1.0})
    s.user_edit(stem, {"width_on": True})
    flat, _ = dsp.render_stem(s, stem)
    assert np.array_equal(off, flat)            # bypassed = width 1.0
    # a hand on the width switches it on
    s.user_edit(stem, {"width_on": False})
    s.user_edit(stem, {"width": 0.6})
    assert s.stems[stem]["width_on"] is True and s.stems[stem]["width"] == 0.6


def test_the_co_edit_carries_width_on():
    from orchestrator import daw_bridge
    import inspect
    src = inspect.getsource(daw_bridge._apply_one_coedit)
    assert '"width_on" in q' in src and 'boolp("width_on")' in src


def test_a_stored_width_is_held_to_the_table():
    x = np.random.default_rng(1).standard_normal((4800, 2))
    assert np.array_equal(dsp._width(x, 5.0), dsp._width(x, 2.0))
    assert np.array_equal(dsp._width(x, -1.0), dsp._width(x, 0.0))
    assert np.array_equal(rm._width(x, 5.0), rm._width(x, 2.0))
