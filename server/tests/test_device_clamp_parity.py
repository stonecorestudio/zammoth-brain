"""The C++ half of "advertised == applied", for device parameters.

`test_param_range_parity.py` pins the PYTHON half: every co-edit parameter the
engine publishes stores exactly its declared bound when you push past it. The
2026-09-08 device audit found the other half missing — `device.set`, the agent's
knob command and the busiest write path in the DAW, applied no bound at all.
The brain's mirror clamped on persist, so the monitor and the bounce ran values
`DEVICE_PARAMS` forbids until the next chain push overwrote them. Four of the
audit's nine agents found it independently, in four different device families:

  * `utility.balance = 50` gave the left channel a gain of -49 (+33.8 dB,
    polarity inverted);
  * `comp.makeup_db = 40` sat 28 dB away from what session.json held;
  * `mb_comp.low_makeup_db = 60` became a raw 1000x band gain;
  * and a NEGATIVE time constant drove the one-pole coefficient to 1.021 > 1,
    diverging the detector to 7.5e18 in ~42 ms — an inf on the audio thread
    that persisted across blocks.

The fix gave the DAW its own copy of the table (`kDeviceParamClamps` in
native/Source/DeviceSpecs.h) because the audio thread cannot call into Python.
A second copy is a second thing to drift, so this test is the thing that stops
it drifting: it re-parses the C++ rows and requires them to BE `DEVICE_PARAMS`.

`--device-clamp` (the DAW console harness) proves the clamp is applied at both
write doors. This proves the numbers it applies are the right ones. Neither is
sufficient alone, and this half is free and runs in CI.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from mix_engine.reference_monitor import UTILITY_TOGGLES
from mix_engine.session import AUTO_FILTER_SNAPPED, RETURN_DB_RANGE, SEND_DB_RANGE, Session

HEADER = Path(__file__).resolve().parents[2] / "native" / "Source" / "DeviceSpecs.h"

# A `gain` chain entry is the fader/trim rather than a device, so Python bounds it through
# USER_EDIT_FIELDS instead of DEVICE_PARAMS — but it IS a co-edit target (the master volume
# lane writes it), so the C++ table has to carry it and this test has to expect it.
_EXTRA = {("gain", "gain_db"): Session.USER_EDIT_FIELDS["gain_db"],
          # A SEND is likewise not a device: `send#i.send_db` / `.return_db` are ride targets the
          # engine advertises and writeLiveParam applies on the audio thread, but Python bounds
          # them in set_send / set_return_level rather than DEVICE_PARAMS. They had no C++ row at
          # all until 2026-09-10, so a modifier could drive ~1.0e5 linear onto a send bus.
          ("send", "send_db"): SEND_DB_RANGE,
          ("send", "return_db"): RETURN_DB_RANGE}

_ROW = re.compile(
    r'\{\s*"([a-z_0-9]+)"\s*,\s*"([a-z_0-9]+)"\s*,\s*'
    r'(-?[\d.]+)f\s*,\s*(-?[\d.]+)f\s*,\s*Snap::(\w+)\s*\}'
)


def _cpp_table() -> dict[tuple[str, str], tuple[float, float, str]]:
    """The rows of kDeviceParamClamps[], as {(device, param): (lo, hi, snap)}."""
    src = HEADER.read_text()
    start = src.index("inline constexpr ParamClamp kDeviceParamClamps[] = {")
    body = src[start : src.index("\n};", start)]
    rows = {
        (m.group(1), m.group(2)): (float(m.group(3)), float(m.group(4)), m.group(5))
        for m in _ROW.finditer(body)
    }
    assert rows, "parsed no rows out of kDeviceParamClamps — has the table's shape changed?"
    return rows


def _expected_snap(path: str, dev: str = "") -> str:
    """The snap `set_master_device_param` applies to this path, by its own branch order."""
    if path in ("ir", "style"):   # a limiter style is an index (2026-10-17)
        return "index"
    if dev == "utility" and path in UTILITY_TOGGLES:   # a Utility's switch: 0 or 1 (2026-10-21)
        return "index"
    if dev == "auto_filter" and path in AUTO_FILTER_SNAPPED:   # Phase 63: 12 / 24, the rest indices
        return "filterSlope" if path == "slope_db_oct" else "index"
    if path.endswith("slope_db_oct"):
        return "slope"
    if path.endswith("oversample"):
        return "oversample"
    if path.endswith("voices"):
        return "voices"
    return "none"


def _python_table() -> dict[tuple[str, str], tuple[float, float, str]]:
    out = {
        (dev, param): (float(lo), float(hi), _expected_snap(path, dev))
        for dev, params in Session.DEVICE_PARAMS.items()
        for param, (path, lo, hi) in params.items()
    }
    out.update({k: (float(lo), float(hi), "none") for k, (lo, hi) in _EXTRA.items()})
    return out


def test_cpp_table_covers_exactly_the_python_table():
    """No row may exist on one side only — either way round is a live bug.

    A param in Python but not C++ is an unbounded write door (the audit's finding).
    A param in C++ but not Python is a target `device.set` accepts and the brain
    then refuses, which is the same disagreement pointing the other way.
    """
    cpp, py = _cpp_table(), _python_table()
    missing = sorted(py.keys() - cpp.keys())
    extra = sorted(cpp.keys() - py.keys())
    assert not missing, f"in session.py but not DeviceSpecs.h (unbounded in the DAW): {missing}"
    assert not extra, f"in DeviceSpecs.h but not session.py (the brain will refuse these): {extra}"


@pytest.mark.parametrize("key", sorted(_python_table()))
def test_each_row_carries_the_same_bound_and_the_same_snap(key):
    cpp, py = _cpp_table(), _python_table()
    assert key in cpp, f"{key} has no clamp row in DeviceSpecs.h"
    c_lo, c_hi, c_snap = cpp[key]
    p_lo, p_hi, p_snap = py[key]
    dev, param = key
    assert (c_lo, c_hi) == (p_lo, p_hi), (
        f"{dev}.{param}: DeviceSpecs.h says {c_lo}..{c_hi}, session.py says {p_lo}..{p_hi}"
    )
    assert c_snap == p_snap, (
        f"{dev}.{param}: DeviceSpecs.h snaps as {c_snap!r}, session.py as {p_snap!r} — "
        "a discrete param clamped as continuous stores a value no engine can run"
    )


def test_the_parse_is_not_vacuous():
    """Negative control: a table this test cannot read must not read as an empty agreement.

    Written after the audit found gates all over this repo that passed while measuring
    nothing. If the regex stopped matching, every assertion above would compare {} to {}
    and pass — so pin the row count and one known row.
    """
    cpp = _cpp_table()
    assert len(cpp) == len(_python_table()) >= 168
    assert cpp[("comp", "makeup_db")] == (0.0, 12.0, "none")
    assert cpp[("eq", "slope_db_oct")][2] == "slope"
    assert cpp[("chorus", "voices")][2] == "voices"


def test_coedit_singleton_list_matches_the_native_mirror():
    """`isCoeditSingletonType` in DeviceSpecs.h must BE `_NATIVE_SINGLETON_KINDS`.

    The dock's `+` menu applied no capacity rule at all — it offered a second compressor,
    committed the illegal entry to the live chain, and only then let the brain refuse it, so
    the monitor ran a device the render would never have (dock audit 2026-09-10). Refusing at
    the dock's own door needs the list in C++, and a second copy of a list is a second thing
    to drift, so this pins it the way the clamp table above is pinned.

    Note it is deliberately NOT `isSingleSlotDeviceType`, which is "every native type except
    the EQ-bank ones" and therefore counts `dynamic_eq` and `band_width` as singletons when
    Python does not.
    """
    from mix_engine.session import _NATIVE_SINGLETON_KINDS

    block = HEADER.read_text().split("inline bool isCoeditSingletonType", 1)[1]
    block = block.split("}", 1)[0]
    native = set(re.findall(r'"([a-z_0-9]+)"', block))
    assert native == set(_NATIVE_SINGLETON_KINDS), (
        f"only in C++: {sorted(native - set(_NATIVE_SINGLETON_KINDS))}; "
        f"only in Python: {sorted(set(_NATIVE_SINGLETON_KINDS) - native)}")
