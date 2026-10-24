"""Reader P3s, batch M25 - the boundary on the app side: OSC, diagnostics, the updater,
preferences, recording (2026-10-07).

The behaviour is driven by the console legs in `daw/Source/AuditReaderP3LegsM25.h`. This file
holds what a leg cannot: rules about the SOURCE that a later edit could quietly undo, the words
a user reads, and the one table a saved file depends on.

    python server/tests/test_reader_p3_m25.py --append     # after ADDING a command id (never edits one)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "daw" / "Source"
IDS_FILE = Path(__file__).resolve().parent / "data" / "command_ids.json"


def _read(name: str) -> str:
    return (SRC / name).read_text(encoding="utf-8")


def _product(text: str) -> str:
    """App.cpp without the console legs at its end."""
    cut = text.find("#ifdef MIXAGENT_DAW_CONSOLE\n#include")
    return text if cut < 0 else text[:cut]


# ---------------------------------------------------------------------------------------------
# E9.R11 c - the numeric command ids are a saved file's format


def command_ids(header: str | None = None) -> dict[str, int]:
    """`enum CommandID` of DawCommands.h as {name: value}, counted the way the compiler does."""
    text = header if header is not None else _read("DawCommands.h")
    start = text.index("enum CommandID\n{") + len("enum CommandID\n{")
    body = text[start: text.index("\n};", start)]
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    body = "\n".join(line.split("//", 1)[0] for line in body.splitlines())
    out: dict[str, int] = {}
    nxt = 0
    for raw in body.split(","):
        entry = raw.strip()
        if not entry:
            continue
        m = re.fullmatch(r"([A-Za-z_]\w*)(?:\s*=\s*(0[xX][0-9a-fA-F]+|\d+))?", entry)
        assert m, f"an enum entry this parser does not read: {entry!r}"
        if m.group(2) is not None:
            nxt = int(m.group(2), 0)
        out[m.group(1)] = nxt
        nxt += 1
    return out


def test_the_parser_counts_as_the_compiler_does():
    """The control for the pin below: explicit values restart the count, comments are not entries."""
    sample = "enum CommandID\n{\n    a = 0x10,\n    b,   // args: { x, y }\n    c,\n    d = 40, e\n};"
    assert command_ids(sample) == {"a": 0x10, "b": 0x11, "c": 0x12, "d": 40, "e": 41}
    moved = "enum CommandID\n{\n    a = 0x10,\n    inserted,\n    b,\n    c,\n    d = 40, e\n};"
    assert command_ids(moved)["b"] != command_ids(sample)["b"], "an insertion mid-enum moves what follows"


def test_no_command_id_ever_moves():
    """E9.R11 c. key-mappings.xml stores a rebind under the command's NUMERIC id
    (`KeyPressMappingSet::createXml`), so an id that moves re-points every user's saved chord at
    another command after an update - silently. "Append-only, never renumbered" was a comment.
    Every id ever shipped is pinned here; a new command is appended (`--append`), never inserted."""
    pinned = json.loads(IDS_FILE.read_text(encoding="utf-8"))
    live = command_ids()
    moved = {k: (pinned[k], live[k]) for k in pinned if k in live and live[k] != pinned[k]}
    assert not moved, f"command ids MOVED (a saved shortcut now names another command): {moved}"
    gone = sorted(k for k in pinned if k not in live)
    assert not gone, f"command ids removed from the enum (leave a tombstone, the id is in users' files): {gone}"
    new = sorted(k for k in live if k not in pinned)
    assert not new, ("new command ids are not pinned yet - run "
                     f"`python server/tests/test_reader_p3_m25.py --append`: {new}")
    assert len(set(live.values())) == len(live), "two commands share an id"


# ---------------------------------------------------------------------------------------------
# S9.R13 - the developer's smokes


def test_a_smoke_hook_is_read_only_through_the_helper():
    """S9.R13. Five `DAW_SMOKE_*` hooks edit the open project a few seconds after launch and were
    read straight from the environment in the shipped app. `smokeEnv` (appboundary::devHook)
    answers nothing in an installed copy; a bare read would bring the hook back."""
    app = _product(_read("App.cpp"))
    bare = re.findall(r'getEnvironmentVariable \("DAW_SMOKE_\w+"', app)
    assert not bare, f"a DAW_SMOKE hook is read past smokeEnv: {bare}"
    assert len(re.findall(r'smokeEnv \("DAW_SMOKE_\w+"\)', app)) >= 6
    boundary = _read("AppBoundary.h")
    assert "if (isInstalledPath (appPath)) return {};" in boundary


# ---------------------------------------------------------------------------------------------
# D1.R8 c - one rule for "is this a take"


def test_is_this_a_take_is_asked_in_one_place():
    """D1.R8 c. Six sites compared the file's PARENT folder name with "recordings"; a track whose
    name holds a `/` records into a folder of its own under it, and none of the six saw its
    takes. The rule is appboundary::isTakeFile."""
    pattern = re.compile(r'getFileName\(\)\s*[!=]=\s*"recordings"')
    for name in ("App.cpp", "DawSession.h", "EditSnapshot.h", "ArrangeView.h"):
        text = _read(name)
        text = _product(text) if name == "App.cpp" else text
        assert not pattern.search(text), f"{name} asks the parent folder's name itself"
    assert len(pattern.findall(_read("AppBoundary.h"))) == 1


# ---------------------------------------------------------------------------------------------
# S4.R15 / S9.R8 - the OSC door


def test_the_osc_door_and_the_brain_door_share_the_confirm_rule():
    """S4.R15 a. The brain's door asked for the user's yes on three things, the OSC door copied
    one. Both now call needsUserYes; the per-call confirms are written once."""
    app = _product(_read("App.cpp"))
    assert app.count("dawcmd::agentConfirmFor (command, args)") == 1, "the per-call confirm is written at more than one door"
    assert "const bool needsConfirm = needsUserYes (*spec, command, args);" in app
    door = app[app.index("    juce::String oscRefusal ("):]
    door = door[: door.index("\n    }\n")]
    assert "needsUserYes (*spec, command, args)" in door and "dawcmd::startsEngineer (command)" in door
    wiring = app[app.index("oscBridge->onCommand = "):]
    wiring = wiring[: wiring.index("return oscBridge->start")]
    assert "c->oscRefusal (command, args)" in wiring and "spec->confirm" not in wiring


def test_the_keys_that_start_the_engineer():
    cmds = _read("DawCommands.h")
    body = cmds[cmds.index("inline bool startsEngineer"):]
    body = body[: body.index("\n}\n")]
    for key in ("agent.run", "agent.auto_mix", "agent.advise"):
        assert f'"{key}"' in body


# ---------------------------------------------------------------------------------------------
# E9.R11 b / d - the menu's chords


def test_no_view_menu_item_prints_a_fixed_chord():
    """E9.R11 b. Two View menu items carried their shortcut as text, so a rebind left the menu
    showing the old chord; they were built from the registry into the title. Since the real-use QA
    of 2026-10-22 no View item puts a key in its title at all (In/Out, Track Delay and Time Ruler
    still printed fixed ones): an item that is a DawComponent command carries that command's
    manager, so the menu prints the registry's chord in its own column, as the Edit menu does."""
    app = _product(_read("App.cpp"))
    assert 'Search Commands\\xe2\\x80\\xa6  (' not in app, "Search Commands prints a fixed chord again"
    assert 'Computer MIDI Keyboard\\t' not in app, "Computer MIDI Keyboard prints a fixed chord again"
    view = app[app.index("topLevelMenuIndex == 2"):app.index("topLevelMenuIndex == 3")]
    assert "(\\xe2\\x8c" not in view, "a View item prints its key as text again"
    assert "item.commandManager = &daw->commandsForMenu();" in view
    for command in ("commandPalette", "viewComputerKeyboard", "viewIoSection", "viewTrackDelay", "viewTimeRuler"):
        assert f"viewItem (m, dawcmd::{command}," in view, command
    assert "the menu catches" not in app, "the comment that had the key order backwards is back"


# ---------------------------------------------------------------------------------------------
# words a user reads


def test_the_controller_tooltip_names_what_the_menu_offers():
    """E9.R13. The tooltip promised HUI and generic MIDI controllers; the engine's surfaces are
    Mackie Control-protocol ones."""
    prefs = _read("PrefsPanel.h")
    tip = prefs[prefs.index('"Add a control surface\\n'):]
    tip = tip[: tip.index("kRowH")]
    assert "HUI" not in tip and "generic MIDI" not in tip
    for name in ("MCU", "Icon Pro G2", "Tranzport", "AlphaTrack", "ReMOTE SL"):
        assert name in tip


def test_the_latency_hint_says_the_drivers_figure_is_already_applied():
    """D1.R9. The row put the driver's round trip beside the field as if it were what to type;
    the engine already takes it off every take, so typing it moved every take early."""
    prefs = _read("PrefsPanel.h")
    row = prefs[prefs.index('new Row ("Latency offset"'):]
    row = row[: row.index("kRowH")]
    assert "already compensated" in row and "driver reports" not in row
    assert "nudges recorded audio" not in row
    manual = (ROOT / "docs" / "17-user-manual.md").read_text(encoding="utf-8")
    assert "trimming from a real number" not in manual
    assert "already compensated" in manual or "already taken off" in manual


def test_the_manual_says_what_osc_does():
    """S9.R8 + E9.R12. The manual's OSC section: the engineer is not started over OSC, a
    confirm-gated command is refused (not asked), play toggles, tracks count from 0, bundles."""
    manual = (ROOT / "docs" / "17-user-manual.md").read_text(encoding="utf-8")
    osc = manual[manual.index("## OSC remote control"):]
    osc = osc[: osc.index("\n# ")]
    assert "agent.run" in osc and "Starting the engineer" in osc
    assert "simply refused" in osc, "the manual says again that OSC is treated 'same as the agent' (the agent is ASKED)"
    assert "0-based" in osc or "counts from 0" in osc
    assert "toggle" in osc.lower()
    assert "bundle" in osc.lower()


def test_the_docs_say_where_takes_are_written():
    """D1.R8 a (James, 2026-10-06: "correct the words"). Takes are written to
    <project>/Recorded/recordings/; three documents said <project>/recordings/."""
    manual = (ROOT / "docs" / "17-user-manual.md").read_text(encoding="utf-8")
    claude = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert "Recorded/recordings/" in manual and "Recorded/recordings/" in claude
    assert "takes to `recordings/`" not in claude
    assert not re.search(r"(?<!Recorded/)`recordings/` \(raw takes\)", manual)
    session = _read("DawSession.h")
    assert 'kTakeFilePattern = "%projectdir%/recordings/%track%_%take%"' in session, \
        "the take mask moved: the documents name the folder it writes"


# ---------------------------------------------------------------------------------------------
# S7.R12 / R13 - the updater


def test_the_updater_checks_the_offered_build_outside_the_team_block():
    """S7.R13 b. The rollback guard runs on EVERY channel: it sits before `if [ -n "$want_team" ]`,
    and reads the bundle's CFBundleVersion against the offered build."""
    up = _read("Updater.h")
    script = up[up.index('R"sh('): up.index(')sh"')]
    guard = script.index('[ "$got_build" = "$build" ]')
    assert guard < script.index('if [ -n "$want_team" ]; then')
    assert "Print :CFBundleVersion" in script
    assert 'certificate leaf[subject.OU] = \\"$want_team\\"' in script
    assert script.index("*[!A-Za-z0-9]*") < script.index("certificate leaf[subject.OU]"), \
        "the team reaches the requirement string before it is checked to be a team id"


def test_a_signed_build_asks_for_its_own_team_first():
    """S7.R12 (James, 2026-10-06). expectedTeamFor answers the running app's own team before it
    looks at the channel or the environment."""
    up = _read("Updater.h")
    fn = up[up.index("inline juce::String expectedTeamFor"):]
    fn = fn[: fn.index("\n}\n")]
    lines = [ln.strip() for ln in fn.splitlines() if ln.strip() and not ln.strip().startswith("//")]
    body = lines[lines.index("{") + 1:]
    assert body[0] == "if (ownTeam.isNotEmpty()) return ownTeam;", body[0]


def test_no_red_run_switch_ships():
    """The batch's red run kept each old path behind a switch (M25Red.h). None of it ships."""
    assert not (SRC / "M25Red.h").exists()
    for path in list(SRC.glob("*.h")) + list(SRC.glob("*.cpp")) + list(SRC.glob("*.mm")):
        assert "m25red" not in path.read_text(encoding="utf-8", errors="replace"), path.name


if __name__ == "__main__":
    if "--append" in sys.argv:
        pinned = json.loads(IDS_FILE.read_text(encoding="utf-8")) if IDS_FILE.exists() else {}
        live = command_ids()
        added = {k: v for k, v in live.items() if k not in pinned}
        clash = {k: (pinned[k], live[k]) for k in pinned if k in live and pinned[k] != live[k]}
        if clash:
            sys.exit(f"refusing to append: ids MOVED {clash} - put the enum back, a saved shortcut depends on them")
        pinned.update(added)
        IDS_FILE.write_text(json.dumps(pinned, indent=0, sort_keys=False) + "\n", encoding="utf-8")
        print(f"pinned {len(added)} new id(s); {len(pinned)} in all")
