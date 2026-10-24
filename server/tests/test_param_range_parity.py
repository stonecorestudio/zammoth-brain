"""What a control ADVERTISES is what it APPLIES.

The 2026-08-16 review found four places where a value passed validation,
reported success, and silently landed somewhere else — the validation layer
and the engine clamp disagreeing about one parameter's range:

  * `vcaLevelOk` accepted -60..+6 while `setVcaDb` clamped to -24..+24, so a
    -50 dB request replied ok() and became -24 (daw/Source/DawSession.h).
  * `set_master_param("gain")` hardcoded -24..12 while `USER_EDIT_FIELDS`
    allows -60..12, so the same field had a different floor depending on
    whether you touched the fader or the device-dock tile.
  * The comp and gate graphs dragged over their own plot axis rather than the
    parameter's range (native/Source/DeviceGraphs.h).

This pins the Python half of that class mechanically, over every co-edit
parameter the engine publishes: push past each declared bound and require the
stored value to be exactly that bound. It deliberately does NOT assert the
tables sit inside the agent's Action models — the co-edit channel is allowed
to be wider on purpose, and says so in place (dynamic_eq's freq_hz spans the
full audible range while the agent action stays 40-16k; its gain_db allows a
boost while the agent's stays a cut). Advertised == applied is the invariant;
"same as the agent" is not.
"""
from __future__ import annotations

import pytest

from mix_engine.session import Session

# Snapped to a discrete set (or mapped to a NAME) rather than clamped to a
# continuum — a different contract, covered by their own tests.
_SNAPPED = {"slope_db_oct", "oversample", "mode", "ir"}


def _place(s: Session, device_type: str, stem: str):
    """Add the device wherever it is legal and return THAT target, else None.

    Returning the target matters: a space device lands on a return chain, not
    on the stem or the master, and editing it needs the chain it actually
    went to.
    """
    for target, add in (("master", lambda: s.add_master_device(device_type)),
                        (stem, lambda: s.add_stem_device(stem, device_type))):
        try:
            add()
            return target
        except Exception:                     # unsupported on that chain
            continue
    return None


def test_user_gain_clamp_matches_the_declared_field_range(demo_project):
    """The device-dock Gain tile and the channel fader share one floor."""
    s = Session(demo_project)
    lo, hi = s.USER_EDIT_FIELDS["gain_db"]
    assert s.set_master_param("gain", 0, lo - 100.0)["value"] == pytest.approx(lo)
    assert s.set_master_param("gain", 0, hi + 100.0)["value"] == pytest.approx(hi)


def test_master_param_clamps_match_their_declared_ranges(demo_project):
    """Every MASTER_PARAM primary knob clamps to the range it publishes."""
    s = Session(demo_project)
    stem = next(iter(s._state["stems"]))
    checked = 0
    for device_type, (_path, lo, hi) in s.MASTER_PARAM.items():
        if device_type == "gain":
            continue
        target = _place(s, device_type, stem)
        if target is None:
            continue
        assert s.set_master_param(device_type, 0, lo - 1000.0,
                                  target=target)["value"] == pytest.approx(lo, abs=0.05)
        assert s.set_master_param(device_type, 0, hi + 1000.0,
                                  target=target)["value"] == pytest.approx(hi, abs=0.05)
        checked += 1
    assert checked >= 4, "the sweep stopped finding devices — it is proving nothing"


def test_device_param_clamps_match_their_declared_ranges(demo_project):
    """The same, over the full per-device param tables (the detail popups)."""
    s = Session(demo_project)
    stem = next(iter(s._state["stems"]))
    checked = 0
    for device_type, specs in s.DEVICE_PARAMS.items():
        target = _place(s, device_type, stem)
        if target is None:
            continue
        for param, (path, lo, hi) in specs.items():
            if path.split(".")[-1] in _SNAPPED:
                continue
            try:
                low = s.set_master_device_param(device_type, 0, param,
                                                lo - 1000.0, target=target)
                high = s.set_master_device_param(device_type, 0, param,
                                                 hi + 1000.0, target=target)
            except Exception:                 # param not editable on this chain
                continue
            assert low["value"] == pytest.approx(lo, abs=0.05), (
                f"{device_type}.{param}: advertises {lo} as its floor, applied {low['value']}")
            assert high["value"] == pytest.approx(hi, abs=0.05), (
                f"{device_type}.{param}: advertises {hi} as its ceiling, applied {high['value']}")
            checked += 1
    assert checked >= 40, f"only {checked} params swept — the enumeration broke"
