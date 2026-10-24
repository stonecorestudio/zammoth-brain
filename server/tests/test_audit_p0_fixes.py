"""Gates on the reader P0 fixes from the 2026-10-02 full audit
(docs/audits/full-2026-10-02/readers/P0-fix-plan.md). The C++ ones are console legs; these are the
brain's.

S4·R8  an import never writes over another track's audio: a renamed track keeps its original file,
       and a deleted track's file must survive for its clips + the undo of the delete. Until the fix,
       importing a file whose stem id named such a file copied over it, registered nothing, and
       answered ok.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

from mix_engine.session import Session
from orchestrator.daw_bridge import apply_coedits, import_stem, rename_stem


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _foreign_wav(folder: Path, name: str, freq_hz: float) -> Path:
    """A different song's file that happens to carry the same name."""
    folder.mkdir(parents=True, exist_ok=True)
    t = np.arange(48000) / 48000.0
    path = folder / name
    sf.write(str(path), (0.25 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32), 48000)
    return path


# ---- S4·R8 -------------------------------------------------------------------------------------
def test_import_after_a_rename_keeps_the_renamed_tracks_audio(demo_project, tmp_path):
    original = _digest(demo_project / "stem_a.wav")
    assert rename_stem(demo_project, "stem_a", "vox")["renamed"]
    foreign = _foreign_wav(tmp_path / "other_song", "stem_a.wav", 997.0)

    res = import_stem(demo_project, str(foreign), stem_id="stem_a")

    assert res["ok"] and res["stem_id"] == "stem_a"
    assert res["wav"] != "stem_a.wav", "the import wrote the renamed track's file"
    assert _digest(demo_project / "stem_a.wav") == original, "vox's audio was replaced"
    assert _digest(demo_project / res["wav"]) == _digest(foreign)
    for _ in range(2):                                   # and it survives reopening, twice
        s = Session(demo_project)
        assert s.stems["vox"]["path"] == "stem_a.wav"
        assert s.stems["stem_a"]["path"] == res["wav"]
        assert not s.stems["stem_a"].get("excluded") and not s.stems["vox"].get("excluded")
        assert set(s.stems) == {"vox", "stem_a", "stem_b"}, "no ghost stem from either file"


def test_import_after_a_delete_keeps_the_deleted_tracks_audio(demo_project, tmp_path):
    original = _digest(demo_project / "stem_b.wav")
    assert apply_coedits(demo_project, [
        "http://127.0.0.1:1/api/set_stem?stem=stem_b&excluded=true"])["failed"] == 0
    foreign = _foreign_wav(tmp_path / "other_song", "stem_b.wav", 431.0)

    res = import_stem(demo_project, str(foreign), stem_id="stem_b")

    assert res["ok"] and res["reset"], "the reused name starts a fresh record"
    assert res["wav"] != "stem_b.wav"
    assert _digest(demo_project / "stem_b.wav") == original, "the undo of the delete would play the new file"
    for _ in range(2):
        s = Session(demo_project)
        assert s.stems["stem_b"]["path"] == res["wav"] and not s.stems["stem_b"].get("excluded")
        assert set(s.stems) == {"stem_a", "stem_b"}


def test_a_retired_file_is_neither_overwritten_nor_readopted(demo_project, tmp_path):
    # rename + re-drop retires stem_a.wav (the renamed track's old audio, which its undo plays)
    assert rename_stem(demo_project, "stem_a", "lead")["renamed"]
    assert import_stem(demo_project, str(demo_project / "stem_b.wav"), stem_id="lead")["ok"]
    old = _digest(demo_project / "stem_a.wav")
    assert "stem_a.wav" in Session(demo_project)._state["retired_wavs"]
    foreign = _foreign_wav(tmp_path / "other_song", "stem_a.wav", 613.0)

    res = import_stem(demo_project, str(foreign), stem_id="stem_a")

    assert res["ok"] and res["wav"] != "stem_a.wav"
    assert _digest(demo_project / "stem_a.wav") == old
    for _ in range(2):
        s = Session(demo_project)
        assert s.stems["stem_a"]["path"] == res["wav"], "discovery pulled the record onto the retired file"
        assert s.stems["lead"]["path"] == "lead.wav"


def test_an_ordinary_import_still_writes_its_own_name(demo_project, tmp_path):
    # control: a free name writes <stem_id>.wav as before, and so does the SAME audio again.
    # (Until reader D4·R1 this also pinned "a live stem re-importing OTHER audio onto itself
    # replaces its file" - which is that finding: the stem's earlier clips, a clip stacked on
    # another lane and the undo of the drop all still play the file. Other audio now takes the
    # next name and the old file stays; test_audit_p1_fixes has the case.)
    fresh = _foreign_wav(tmp_path / "other_song", "loop.wav", 220.0)
    res = import_stem(demo_project, str(fresh))
    assert res["ok"] and res["stem_id"] == "loop" and res["wav"] == "loop.wav"
    res = import_stem(demo_project, str(fresh), stem_id="loop")
    assert res["wav"] == "loop.wav" and _digest(demo_project / "loop.wav") == _digest(fresh)
    assert Session(demo_project).stems["loop"]["path"] == "loop.wav"
    again = _foreign_wav(tmp_path / "newer_take", "loop.wav", 330.0)
    res = import_stem(demo_project, str(again), stem_id="loop")
    assert res["wav"] == "loop 2.wav" and _digest(demo_project / "loop.wav") == _digest(fresh)
    assert Session(demo_project).stems["loop"]["path"] == "loop 2.wav"


def test_a_stem_whose_file_is_gone_takes_its_own_name_back(demo_project, tmp_path):
    # control for the rule above: a missing stem's own name protects nothing, so a relink's
    # register writes <stem_id>.wav and revives the record
    assert "stem_b" in Session(demo_project).stems       # registered, then its file goes
    (demo_project / "stem_b.wav").unlink()
    assert Session(demo_project).stems["stem_b"].get("missing")
    found = _foreign_wav(tmp_path / "moved", "stem_b.wav", 523.0)
    res = import_stem(demo_project, str(found), stem_id="stem_b")
    assert res["ok"] and res["wav"] == "stem_b.wav"
    s = Session(demo_project)
    assert s.stems["stem_b"]["path"] == "stem_b.wav" and not s.stems["stem_b"].get("excluded")
