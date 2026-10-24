"""Phase 55, step 2d: three things on the chat turn's path that were wrong.

- Three helpers that only READ the session's levels opened it with the constructor that
  writes: session.json was rewritten and synced at the start of every agent turn.
- The brain's render cache was believed because of call order. An in-turn mix move, an undo
  or an import that did not clear it left the old limiter make-up in it, and the next
  audio-neutral change (an empty group, a rename) pushed that to the live graph.
- A live-edit snapshot that carries the "this predates your last command" note was re-sent
  whole (24,000 characters) where the difference would do.
"""
from __future__ import annotations

import json

import orchestrator.daw_bridge as B
import orchestrator.daw_server as S
from mix_engine import dsp
from mix_engine.session import Session
from orchestrator.chat import _LIVE_HEAD, _live_refresh
from tests.test_phase54_render_once import _project, kept  # noqa: F401


def test_reading_the_levels_writes_nothing(tmp_path, monkeypatch):
    proj = _project(tmp_path)
    def file() -> tuple:
        stat = (proj / "session.json").stat()
        return stat.st_ino, stat.st_mtime_ns     # a save renames a new file into place
    before = file()
    snapshot = B.level_snapshot(proj)
    control = ("set_stem", "bass", "gain_db")
    assert control in snapshot, f"vacuity: the snapshot must hold the fader: {sorted(snapshot)[:4]}"
    B.level_values(proj, [control])
    B.level_inverses(proj, ["/api/set_stem?stem=bass&gain_db=-4.0"])
    assert file() == before, "a read of the levels rewrote session.json"


def test_an_audio_neutral_change_does_not_answer_from_another_mixs_numbers(tmp_path, kept):
    proj = _project(tmp_path)
    brain = S.Brain(proj)
    first = B.chains_with_readout(proj, cache=brain.render_cache)["readout"]
    # an in-turn mix move the cache was not told about: the bass 12 dB down
    Session(proj).user_edit("bass", {"gain_db": -12.0})
    masters = dsp.RENDER_COUNTS["master"]
    out = brain.group_stems(["vocals", "keys"])
    assert dsp.RENDER_COUNTS["master"] == masters + 1, "the stale numbers were served"
    assert out["readout"]["lufs_i"] != first["lufs_i"]
    assert out["readout"] == B.session_readout(proj, {})


def test_consecutive_neutral_changes_still_render_nothing(tmp_path, kept):
    """Control: the reason the light path exists (the multi-second hang at every group)."""
    proj = _project(tmp_path)
    brain = S.Brain(proj)
    B.chains_with_readout(proj, cache=brain.render_cache)
    masters = dsp.RENDER_COUNTS["master"]
    made = brain.group_stems(["vocals", "keys"])
    brain.rename_bus(made["bus_id"], "leads")
    brain.rename_stem("bass", "sub bass")
    assert dsp.RENDER_COUNTS["master"] == masters, "an audio-neutral change rendered the song"


def test_a_snapshot_labelled_stale_is_still_sent_as_the_difference():
    tracks = [{"name": f"track {i}", "gain_db": 0.0, "clips": [{"start_s": float(i)}]}
              for i in range(40)]
    old = {"seq": 4, "tempo_bpm": 120.0, "tracks": tracks}
    new = {**old, "seq": 5, "tempo_bpm": 147.0}
    last = _LIVE_HEAD + json.dumps(old)
    plain = _live_refresh(last, _LIVE_HEAD + json.dumps(new))
    stale = _live_refresh(last, B.STALE_PICTURE_NOTE + _LIVE_HEAD + json.dumps(new))
    assert "CHANGED since the last snapshot" in plain and len(plain) < 0.5 * len(last)
    assert "CHANGED since the last snapshot" in stale, "the whole snapshot was sent again"
    assert len(stale) < 0.6 * len(last)
    assert stale.startswith(B.STALE_PICTURE_NOTE.strip()), "and it still says it predates the command"
