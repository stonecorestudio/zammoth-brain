"""The Tilt EQ's audit, 2026-10-17 (the device-by-device audit).

The native half is `--tilt-audit`. Here:
  * a chain runs at most FOUR tilts - the native keeps a record of four (`kMaxTilt`); a fifth
    played but could not be ridden, and a ride on an earlier tilt's output dropped its trim. The
    brain refuses the fifth at the number the native holds (and the dock, `capacityRefusal`);
  * the face draws at the node's rate (a source pin: it drew at 48 kHz, 1 dB off at 96 kHz).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from mix_engine.session import _NATIVE_SLOT_KINDS, Session

ROOT = Path(__file__).resolve().parents[2]


def test_the_brain_caps_tilts_at_the_natives_count():
    header = (ROOT / "native" / "Source" / "MonitorChain.h").read_text()
    k = int(re.search(r"static constexpr int kMaxTilt\s*=\s*(\d+)", header).group(1))
    assert _NATIVE_SLOT_KINDS["tilt"] == k
    dock = (ROOT / "daw" / "Source" / "DeviceChainView.h").read_text()
    assert 'type == "tilt" && have + arriving > MonitorChain::kMaxTilt' in dock


def test_a_fifth_tilt_is_refused(demo_project):
    session = Session(demo_project)
    for _ in range(4):
        session.add_stem_device("stem_a", "tilt")
    with pytest.raises(ValueError, match="tilt"):
        session.add_stem_device("stem_a", "tilt")


def test_the_tilt_face_draws_at_the_nodes_rate():
    graphs = (ROOT / "native" / "Source" / "DeviceGraphs.h").read_text()
    tilt = graphs[graphs.index("class TiltGraph : public Component"):]
    tilt = tilt[:tilt.index("\nclass ")]
    assert "48000.0" not in tilt.replace("double sampleRate = 48000.0;", "")
    dock = (ROOT / "daw" / "Source" / "DeviceChainView.h").read_text()
    i = dock.index("static_cast<devicegraphs::TiltGraph*> (graph.get());")
    assert "g->setSampleRate (owner.host.getMonitor().getSampleRate());" in dock[i:i + 300]
