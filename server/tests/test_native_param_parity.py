"""The native knob table and the engine's clamps agree on every shared parameter.

`native/Source/DeviceSpecs.h` is the single source for what a knob, an automation
lane, a macro or a MIDI CC can reach in the DAW and the VST; `Session.DEVICE_PARAMS`
is what the engine clamps a co-edit to. Until 2026-09-01 nothing compared the two
(`test_param_range_parity` reads only the Python side), and the `gain` device
advertised -12..+12 while the engine allowed -60..+12 — undrawable fade-outs, a
macro spending its travel in a quarter of the range (sweep P1/P2). A free text
parse; no build.
"""
from __future__ import annotations

import re
from pathlib import Path

from mix_engine.session import Session

REPO = Path(__file__).resolve().parents[2]
SPECS = REPO / "native" / "Source" / "DeviceSpecs.h"

# (device, field) pairs whose ranges differ ON PURPOSE — each with the reason.
ALLOWED = {
    # the engine keeps the agent-facing model bound; the UI knob is the user's wider hand
}


def _native_ranges() -> dict[tuple[str, str], tuple[float, float]]:
    text = SPECS.read_text(encoding="utf-8")
    fn = text[text.index("deviceParamSpecs"):]
    out: dict[tuple[str, str], tuple[float, float]] = {}
    for m in re.finditer(r'if \(type == "([a-z_]+)"\)(.*?)return \{(.*?)\};', fn, re.S):
        device, body = m.group(1), m.group(3)
        for spec in re.finditer(r'\{\s*"[^"]*",\s*"([a-z_0-9]+)",\s*(-?[0-9.]+),\s*(-?[0-9.]+)', body):
            out[(device, spec.group(1))] = (float(spec.group(2)), float(spec.group(3)))
    assert len(out) > 60, f"parsed only {len(out)} native knobs — the parse is broken"
    return out


def test_every_shared_knob_has_the_same_range_on_both_sides():
    native = _native_ranges()
    engine = Session.DEVICE_PARAMS
    compared, mismatched = 0, []
    for (device, field), (lo, hi) in sorted(native.items()):
        spec = engine.get(device, {}).get(field)
        if spec is None or (device, field) in ALLOWED:
            continue
        _path, elo, ehi = spec
        compared += 1
        if abs(elo - lo) > 1e-9 or abs(ehi - hi) > 1e-9:
            mismatched.append(f"{device}.{field}: native [{lo}, {hi}] vs engine [{elo}, {ehi}]")
    assert compared >= 40, f"only {compared} knobs are shared — did a table get renamed?"
    assert mismatched == [], "\n".join(mismatched)
