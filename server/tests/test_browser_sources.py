"""The browser's places, as the engineer reaches them (2026-11-11; James: "can we add splice inside
the daw?"). A sample service's desktop app keeps what the user bought in a local folder, packs
deep; the browser adds that folder as a place by itself, and the engineer's `list_files` can name
the browser's places and search one through its subfolders."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from orchestrator import chat
from orchestrator import daw_bridge as B

REPO = Path(__file__).resolve().parents[2]


def _library(root: Path) -> Path:
    lib = root / "Splice"
    kicks = lib / "sounds" / "packs" / "Dusty Tapes" / "Drums" / "Kicks"
    kicks.mkdir(parents=True)
    (kicks / "DT_kick_soft_01.wav").write_bytes(b"")
    (kicks / "DT_kick_hard_02.wav").write_bytes(b"")
    snares = lib / "sounds" / "packs" / "Dusty Tapes" / "Drums" / "Snares"
    snares.mkdir(parents=True)
    (snares / "DT_snare_01.wav").write_bytes(b"")
    (snares / "notes.txt").write_text("not audio")
    hidden = lib / ".cache"
    hidden.mkdir()
    (hidden / "kick_ghost.wav").write_bytes(b"")
    return lib


def test_a_deep_search_finds_a_file_folders_down_by_words_of_its_path(tmp_path):
    lib = _library(tmp_path)
    got = B.folder_listing(str(lib), "dusty kick", deep=True)
    assert got["ok"] and got["deep"] is True
    assert got["audio"] == ["sounds/packs/Dusty Tapes/Drums/Kicks/DT_kick_hard_02.wav",
                            "sounds/packs/Dusty Tapes/Drums/Kicks/DT_kick_soft_01.wav"]
    assert got["audio_count"] == 2
    # a word that is only a FOLDER's still finds the files under it; a hidden folder is not walked
    assert B.folder_listing(str(lib), "snares", deep=True)["audio"] == [
        "sounds/packs/Dusty Tapes/Drums/Snares/DT_snare_01.wav"]
    assert not any("ghost" in a for a in B.folder_listing(str(lib), "kick", deep=True)["audio"])
    # the plain listing is as it was: one folder, names only
    plain = B.folder_listing(str(lib))
    assert plain["folders"] == ["sounds"] and plain["audio"] == [] and "deep" not in plain


def test_a_deep_search_is_bounded_and_says_when_it_stopped(tmp_path, monkeypatch):
    lib = _library(tmp_path)
    monkeypatch.setattr(B, "DEEP_MAX_ENTRIES", 2)
    got = B.folder_listing(str(lib), "kick", deep=True)
    assert got["ok"] and "cut" in got


def test_a_deep_search_needs_words_and_never_walks_the_home_folder(tmp_path, monkeypatch):
    lib = _library(tmp_path)
    no_words = B.folder_listing(str(lib), "", deep=True)
    assert not no_words["ok"] and "query" in no_words["error"]
    # the home folder holds other apps' data: a walk of it raises a privacy prompt
    monkeypatch.setenv("HOME", str(tmp_path))
    for target in (str(tmp_path), "~", str(tmp_path.parent)):
        refused = B.folder_listing(target, "kick", deep=True)
        assert not refused["ok"] and "places" in refused["error"], target
    assert B.folder_listing(str(lib), "kick", deep=True)["ok"]


def test_places_names_the_browsers_folders_home_as_tilde(tmp_path, monkeypatch):
    lib = _library(tmp_path)
    data = tmp_path / "appdata"
    data.mkdir()
    (data / "browser_places.txt").write_text(f"{lib}\n{tmp_path / 'gone'}\n\n{lib}\n")
    monkeypatch.setattr(B, "BROWSER_DIR", data)
    monkeypatch.setenv("HOME", str(tmp_path))
    got = B.folder_listing("places")
    assert got == {"ok": True, "places": ["~/Splice"]}
    # no file yet (a brain before the browser's first run) is no places, not an error
    monkeypatch.setattr(B, "BROWSER_DIR", tmp_path / "nowhere")
    assert B.folder_listing("places") == {"ok": True, "places": []}


def test_the_tool_offers_both_and_the_turn_passes_deep_through(tmp_path):
    tool = next(d for d in chat.daw_tool_defs(False) if d["name"] == "list_files")
    props = tool["input_schema"]["properties"]
    assert props["deep"]["type"] == "boolean"
    assert "places" in props["path"]["description"]
    assert "Splice" in tool["description"]

    import anyio
    seen: list = []

    class Tools:
        async def list_files(self, path, query="", deep=False):
            seen.append((path, query, deep))
            return {"ok": True}

    async def call(args):
        return await chat._execute_tool(Tools(), "list_files", args)

    try:
        anyio.run(call, {"path": "~/Splice", "query": "kick", "deep": True})
    except TypeError:
        pytest.fail("the chat's router does not hand `deep` to list_files")
    assert seen == [("~/Splice", "kick", True)]


def test_the_browser_looks_for_the_same_folder_the_tool_names():
    """The place the browser adds by itself and the one the tool's text names are one folder."""
    cpp = (REPO / "daw/Source/FileBrowserPanel.h").read_text()
    body = cpp[cpp.index("sampleSourceCandidates()"):][:1500]
    assert re.search(r'getChildFile \("Splice"\)', body)
    assert "browser_sources_offered.txt" in cpp
    assert "ZAMMOTH_OLD_SOURCES" not in cpp   # the red run's switch is gone
