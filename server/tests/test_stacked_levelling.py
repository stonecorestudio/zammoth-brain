"""One levelling device a track from the engineer (2026-11-02).

The re-mix of the hip hop session, after `against_raw` told the engineer that compressors in
series add up, still left the rap with two whole-file compressors taking 12.4 dB between them:
a note is read and can be weighed away. A proposal that would leave a track (or the master) with
two whole-file levelling devices (comp, mb_comp, ott) is refused here; the engineer adjusts the
one there, or takes one off. Time-ranged compressors keep their own rule (comp_stacking), and the
user may still stack what they like by hand.
"""
from __future__ import annotations

from mix_engine.models import ProposalRequest
from mix_engine.session import Session


def _req(*actions: dict) -> ProposalRequest:
    return ProposalRequest.model_validate({
        "rationale": "levelling", "actions": list(actions),
        "predicted_impact": [{"metric": "stem.crest_db", "predicted_delta": -1.0, "unit": "dB"}]})


def _comp(stem: str, **kw) -> dict:
    return {"type": "comp", "stem_id": stem, "threshold_db": -20.0, "ratio": 3.0,
            "attack_ms": 10.0, "release_ms": 120.0, "makeup_db": 0.0, **kw}


def _stem(s: Session) -> str:
    return next(iter(s._state["stems"]))


def _stacked(res) -> list:
    return [v for v in res.violations if v.rule == "levelling_stacked"]


def test_a_second_compressor_beside_the_one_there_is_refused(demo_project):
    s = Session(demo_project)
    stem = _stem(s)
    s.add_stem_device(stem, "comp")
    res = s.propose(_req(_comp(stem, threshold_db=-14.0, ratio=4.0, instance="new")))
    assert not res.accepted
    said = _stacked(res)
    assert said and "adjust the one there" in said[0].message, res.violations


def test_a_multiband_or_ott_beside_a_compressor_is_refused(demo_project):
    s = Session(demo_project)
    stem = _stem(s)
    s.add_stem_device(stem, "comp")
    band = {"threshold_db": -20.0, "ratio": 2.0, "makeup_db": 0.0, "up": False}
    for move in ({"type": "mb_comp", "stem_id": stem, "low": band, "mid": band, "high": band},
                 {"type": "ott", "stem_id": stem, "depth": 0.3, "in_gain_db": 0.0}):
        res = s.propose(_req(move))
        assert _stacked(res), (move["type"], res.violations)


def test_two_compressors_in_one_proposal_are_refused(demo_project):
    s = Session(demo_project)
    stem = _stem(s)
    res = s.propose(_req(_comp(stem), _comp(stem, instance="new")))
    assert _stacked(res), res.violations


def test_the_one_compressor_is_still_adjusted_by_proposing_it_again(demo_project):
    s = Session(demo_project)
    stem = _stem(s)
    s.add_stem_device(stem, "comp")
    res = s.propose(_req(_comp(stem, threshold_db=-24.0)))
    assert res.accepted, res.violations


def test_one_compressor_on_a_bare_track_and_one_on_another_are_taken(demo_project):
    s = Session(demo_project)
    a, b = list(s._state["stems"])[:2]
    res = s.propose(_req(_comp(a), _comp(b)))
    assert res.accepted, res.violations


def test_a_time_ranged_compressor_keeps_its_own_rule(demo_project):
    s = Session(demo_project)
    stem = _stem(s)
    res = s.propose(_req(_comp(stem, time_range_s=[0.0, 1.0]), _comp(stem, time_range_s=[2.0, 3.0])))
    assert not _stacked(res), res.violations


def test_a_compressor_on_an_audio_track_by_command_is_refused(tmp_path):
    """`device.add` is for a MIDI or instrument track: on an audio stem it skipped these rules
    and filed the device as the user's, which the hip hop re-mix did for its gain trims."""
    import numpy as np
    import soundfile as sf
    from orchestrator.turn_host import checked_command
    project = tmp_path / "song"
    project.mkdir()
    sf.write(project / "vox.wav", np.zeros((4800, 2)), 48000)
    Session(project)._persist()
    for kind in ("comp", "mb_comp", "ott"):
        refused, _ = checked_command(project, "device.add", {"track": "vox", "type": kind}, None)
        assert refused and refused["ok"] is False and "propose" in refused["error"], (kind, refused)
    # a trim, a polarity flip, and a compressor on a track that is no stem go through
    assert checked_command(project, "device.add", {"track": "vox", "type": "utility"}, None)[0] is None
    assert checked_command(project, "device.add", {"track": "synth", "type": "comp"}, None)[0] is None
