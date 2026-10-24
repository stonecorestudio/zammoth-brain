"""The instant-add entry must equal what the brain reconciles to.

Adding a device in the DAW is a two-step handshake: `defaultDeviceEntry()`
(native/Source/DeviceSpecs.h) seeds an entry the in-process MonitorChain plays
IMMEDIATELY, then the co-edit reaches the brain and `Session._ADD_*_DEVICE`
builds the authoritative entry the chain reconciles to. When the two disagree,
the user watches a knob move on its own a moment after the add — and hears the
track change with it. Four types had drifted (mb_comp -18/-24, deess -24/-20,
transient 0/2, limiter target_lufs_i -14/unset).

There is no way to call the C++ from pytest, so this parses the one function
that holds those literals. That is deliberate: the parse is what makes the
C++ side of the contract testable at all, and a parse failure fails the test
rather than silently passing.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from mix_engine.session import Session

SPECS = (Path(__file__).resolve().parents[2]
         / "native" / "Source" / "DeviceSpecs.h")


def _default_device_entries() -> dict[str, dict[str, float]]:
    """Parse `defaultDeviceEntry`'s per-type `p ("key", value)` seeds.

    Only the numeric `p (...)` writes are read — the `o->setProperty` booleans
    and strings (curve, style, ir, band, key_stem_id) are not part of the
    numeric drift this guards.
    """
    src = SPECS.read_text()
    start = src.index("inline juce::var defaultDeviceEntry")
    body = src[start:]
    out: dict[str, dict[str, float]] = {}
    # Each branch is `if/else if (type == "x") { ... }` up to the next branch.
    for m in re.finditer(r'type == "(\w+)"\)\s*\{(.*?)(?=\n\s*(?:else if|return e;))',
                         body, re.S):
        kind, chunk = m.group(1), m.group(2)
        params: dict[str, float] = {}
        for pm in re.finditer(r'p \("([\w.]+)",\s*(-?[\d.]+)\)', chunk):
            params[pm.group(1)] = float(pm.group(2))
        # mb_comp / ott seed their bands through a `for (auto* b : {...})` loop
        # with `juce::String (b) + "key"`; expand it so the bands are comparable.
        loop = re.search(r'for \(auto\* b : \{([^}]*)\}\)\s*\{(.*?)\}', chunk, re.S)
        if loop:
            prefixes = re.findall(r'"(\w+_)"', loop.group(1))
            for lm in re.finditer(r'juce::String \(b\) \+ "(\w+)",\s*(-?[\d.]+)',
                                  loop.group(2)):
                for pre in prefixes:
                    params[pre + lm.group(1)] = float(lm.group(2))
        out[kind] = params
    return out


def _brain_entry(kind: str, table: dict) -> dict | None:
    """The entry ONE brain constructor table reconciles a fresh add to.

    Taking the table explicitly (not `master or stem`) is the point: the old
    fallback meant `_ADD_STEM_DEVICE` was never independently checked for any
    kind present in both tables, so a stem-only default drift would ship unseen.
    """
    ctor = table.get(kind)
    if ctor is None:
        return None
    entry = ctor().model_dump(exclude_none=True)
    flat: dict[str, float] = {}
    for key, val in entry.items():
        if isinstance(val, dict):                      # low/mid/high, mid/side
            for sub, sval in val.items():
                if isinstance(sval, (int, float)) and not isinstance(sval, bool):
                    flat[f"{key}_{sub}"] = float(sval)
        elif isinstance(val, (int, float)) and not isinstance(val, bool):
            flat[key] = float(val)
    return flat


def test_devicespecs_parses():
    """A parse regression must fail loudly, not quietly pass everything."""
    entries = _default_device_entries()
    assert SPECS.is_file(), SPECS
    for expected in ("comp", "limiter", "deess", "transient", "mb_comp", "gate"):
        assert expected in entries, f"{expected} missing — parser out of date"
    assert entries["comp"]["threshold_db"] == pytest.approx(-18.0)


@pytest.mark.parametrize("scope", ["master", "stem"])
@pytest.mark.parametrize("kind", ["comp", "limiter", "deess", "transient",
                                  "mb_comp", "gate", "dynamic_eq", "tilt",
                                  "clip", "saturation", "overdrive", "imager",
                                  "utility", "band_width", "duck", "ott",
                                  "chorus", "flanger", "auto_filter"])
def test_add_defaults_match_brain(kind, scope):
    """Every numeric seed the native shares with the brain must agree — against
    BOTH constructor tables, since a track add and a master add reconcile
    through different ctors and either one can drift alone."""
    table = (Session._ADD_MASTER_DEVICE if scope == "master"
             else Session._ADD_STEM_DEVICE)
    native = _default_device_entries().get(kind)
    brain = _brain_entry(kind, table)
    if native is None or brain is None:
        pytest.skip(f"{kind} has no {scope} add path")
    shared = set(native) & set(brain)
    assert shared, f"{kind}: no comparable params — check the parser"
    mismatched = {k: (native[k], brain[k]) for k in sorted(shared)
                  if native[k] != pytest.approx(brain[k])}
    assert not mismatched, (
        f"{kind}: defaultDeviceEntry (DeviceSpecs.h) disagrees with the brain's "
        f"_ADD_*_DEVICE constructor — the knob will jump after an add. "
        f"{{param: (native, brain)}} = {mismatched}")


def test_limiter_seeds_a_loudness_target():
    """The Loud knob's value must exist on the brain side, not fall back to the
    dock's range MINIMUM (-24 LUFS) the moment the chain reconciles."""
    assert _brain_entry("limiter",
                        Session._ADD_MASTER_DEVICE)["target_lufs_i"] == pytest.approx(-14.0)


def test_a_multiband_comp_seeds_its_crossovers_on_both_sides():
    """A dock-added mb_comp carried no crossovers natively while the brain seeds 150 / 2000:
    the shared-keys comparison above could not see a key on one side only (audit 03 F5)."""
    native = _default_device_entries().get("mb_comp") or {}
    brain = _brain_entry("mb_comp", Session._ADD_MASTER_DEVICE) or {}
    assert "crossover_lo_hz" in brain   # the brain seeds them: the check below is not vacuous
    for key in ("crossover_lo_hz", "crossover_hi_hz"):
        if key in brain:
            assert key in native and native[key] == pytest.approx(brain[key]), key
