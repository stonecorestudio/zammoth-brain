"""The de-esser's audit, 2026-10-17 (the device-by-device audit) - the offline engine.

Since 2026-10-27 the de-esser is `deess_law` (fitted to a reference de-esser; no ratio, attack or
release), held to the native chain by `test_deess_law.py` + `--deess-law`; the old law's two tests
here (its transcription, and its attack knob) went with it. What stays: four de-essers a track. The
native half is `--deess-audit`.
"""
from __future__ import annotations



def test_a_track_takes_four_de_essers_and_refuses_a_fifth(demo_project):
    """James, 2026-10-17: up to four de-essers per track (two-band de-essing) - it was one."""
    import pytest

    from mix_engine.session import Session
    session = Session(demo_project)
    for _ in range(4):
        session.add_stem_device("stem_b", "deess")
    assert sum(e["type"] == "deess" for e in session.stems["stem_b"]["chain"]) == 4
    with pytest.raises(ValueError, match="4 deess"):
        session.add_stem_device("stem_b", "deess")


def test_the_agents_whole_file_de_esser_adjusts_the_last_one_and_leaves_the_others(demo_project):
    from mix_engine.models import DeessAction
    from mix_engine.session import Session
    session = Session(demo_project)
    session.add_stem_device("stem_b", "deess")
    session.add_stem_device("stem_b", "deess")
    session.stems["stem_b"]["chain"][-1]["freq_hz"] = 4000.0        # the user's second one, at the harshness
    session._mutate(DeessAction(stem_id="stem_b", threshold_db=-33.0, freq_hz=8000.0))
    deess = [e for e in session.stems["stem_b"]["chain"] if e["type"] == "deess"]
    assert len(deess) == 2                                           # it took both away before
    assert sorted(e["freq_hz"] for e in deess) != [8000.0, 8000.0]
    assert any(e["freq_hz"] == 8000.0 and e["threshold_db"] == -33.0 for e in deess)
