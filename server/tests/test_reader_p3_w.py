"""Reader P3s, batch W - words only: the manual and the docs (2026-10-07).

Five findings, each a sentence the code no longer backs. A sentence cannot be driven, so each
test here reads the FACT out of the source and then the sentence that describes it: when the
code moves, the sentence that went stale is named by the test that fails.

Record: docs/audits/full-2026-10-02/readers/P3-fixes.md, batch W.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANUAL = (ROOT / "docs" / "17-user-manual.md").read_text(encoding="utf-8")
SESSION_H = (ROOT / "daw" / "Source" / "DawSession.h").read_text(encoding="utf-8")
COMMANDS_H = (ROOT / "daw" / "Source" / "DawCommands.h").read_text(encoding="utf-8")
SNAPSHOT_H = (ROOT / "daw" / "Source" / "EditSnapshot.h").read_text(encoding="utf-8")
APP = (ROOT / "daw" / "Source" / "App.cpp").read_text(encoding="utf-8")
PREFS_H = (ROOT / "daw" / "Source" / "PrefsPanel.h").read_text(encoding="utf-8")
BRAIN = (ROOT / "orchestrator" / "daw_server.py").read_text(encoding="utf-8")
API_KEY = (ROOT / "orchestrator" / "api_key.py").read_text(encoding="utf-8")


def _flat(text: str) -> str:
    """One logical line: the manual wraps at 100 columns, and a sentence is matched whole."""
    return re.sub(r"\s+", " ", text)


def _joined(source: str) -> str:
    """Adjacent string literals read as the one string the compiler makes of them."""
    return re.sub(r'"\s*\n\s*"', "", source)


def _section(heading: str, whole: bool = True) -> str:
    """The manual's text under `heading` (a whole line): to the next heading of the same or a
    higher level, or - `whole=False` - to the next heading of any level."""
    m = re.search(r"^%s$" % re.escape(heading), MANUAL, re.M)
    assert m, f"the manual has no heading {heading!r}"
    level = len(heading) - len(heading.lstrip("#")) if whole else 6
    nxt = re.compile(r"^#{1,%d} " % level, re.M).search(MANUAL, m.end())
    return _flat(MANUAL[m.start(): nxt.start() if nxt else len(MANUAL)])


# ---------------------------------------------------------------- S1.R24

def _stem_patterns() -> set[str]:
    body = SESSION_H[SESSION_H.index("inline juce::StringArray loadProjectStems ("):]
    body = body[: body.index("\n}\n")]
    return set(re.findall(r'findChildFiles \(juce::File::findFiles, false, "([^"]+)"\)', body))


def test_the_manual_names_the_files_a_folder_opens_as_tracks():
    """S1.R24. `loadProjectStems` lists `*.wav` and nothing else, so a folder of .aif / .flac /
    .mp3 opens with no tracks. The manual said "a folder of loose audio files imports as
    tracks". It names the extension, and says how the other formats come in."""
    assert _stem_patterns() == {"*.wav"}, \
        "loadProjectStems lists other files now: the manual's Open sentence must follow"
    projects = _section("## Projects")
    assert "loose audio files imports as tracks" not in projects
    assert re.search(r"folder of loose `\.wav` files", projects), \
        "the Open bullet does not say which files open as tracks"
    opened = projects[projects.index("**Open** (⌘O)"): projects.index("**Save** (⌘S)")]
    for fmt in ("aiff", "flac", "mp3"):
        assert fmt in opened, f"the Open bullet does not say what happens to a folder's {fmt} files"


# ---------------------------------------------------------------- D4.R16

def test_the_manual_does_not_deny_a_feature_it_documents():
    """D4.R16. *What Zammoth is not (yet)* said DAWproject interchange is not available; File >
    Import DAWproject is in the menu, the manual and the catalog. Only the export is absent."""
    assert '"project.import_dawproject"' in COMMANDS_H, "the import command left the catalog"
    assert '"project.export_dawproject"' not in COMMANDS_H, \
        "DAWproject export exists now: *What Zammoth is not (yet)* must stop denying it"
    is_not = _section("### What Zammoth is not (yet)")
    assert "DAWproject interchange" not in is_not
    assert "DAWproject *export*" in is_not or "DAWproject export" in is_not
    assert "Import DAWproject" in is_not, "the sentence that denies the export does not say the import is there"


# ---------------------------------------------------------------- S8.R11

DOCS_THAT_NAME_THE_DOOR = ("CLAUDE.md", "docs/05-roadmap.md", "docs/14-open-beta-checklist.md")


def test_the_docs_name_a_licence_door_that_exists():
    """S8.R11 b. Four doc lines named the brain's licence door as `_route`. No such function was
    ever in the brain: the door is `_license_refusal` (and `_auto_mix_refusal`), in
    orchestrator/daw_server.py."""
    for door in ("_license_refusal", "_auto_mix_refusal"):
        assert re.search(rf"^async def {door}\(", BRAIN, re.M), f"{door} left daw_server.py"
    for rel in DOCS_THAT_NAME_THE_DOOR:
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert not re.search(r"`_route`", text), f"{rel} names `_route`, which does not exist"
        assert "`_license_refusal`" in text, f"{rel} no longer names the brain's licence door"


RUN_COST_FILES = {
    "daw/Source/PrefsPanel.h": PREFS_H,
    "orchestrator/api_key.py": API_KEY,
    "daw/Source/App.cpp": APP,
    "docs/17-user-manual.md": MANUAL,
}


def test_a_run_cost_is_said_of_the_default_model():
    """S8.R11 c. "About a dollar a run" is the measured cost on Opus 5 ($1.28), which was the
    default when it was measured. The picker beside the Preferences note offers other models,
    so the figure is said of the default model wherever it is said - in the app's own words as
    in the manual. (Since 2026-10-08 the default is Opus 5.5, on which a run has not been
    measured: the same tokens are $1.02 at its list price. Re-measure before trusting it.)"""
    seen = 0
    for rel, text in RUN_COST_FILES.items():
        flat = _flat(_joined(text))
        for m in re.finditer(r"dollar a run", flat):
            seen += 1
            after = flat[m.end(): m.end() + 40]
            assert after.lstrip(" *").startswith("on the default model"), \
                f"{rel}: \"a dollar a run\" is not said of the default model: ...{flat[m.start() - 40: m.end() + 40]}..."
    assert seen >= 5, f"the scan found only {seen} of the sentences it is there to read"


# ---------------------------------------------------------------- S8.R13

def test_the_manual_says_the_agent_picks_are_one_setting():
    """S8.R13 (James, 2026-10-06: "one setting; correct the manual"). M14 made the model, the
    effort and Ask / Agent one preference for every open project: `enterForeground` runs
    `applyLivePrefs`, which hands them to the composer. The manual said, twice, that each open
    project keeps its own."""
    body = APP[APP.index("    void applyLivePrefs()"):]
    body = body[: body.index("\n    }\n")]
    assert "engineer->applyLlmPrefs (p.agentModel, p.agentEffort, p.agentMode)" in body, \
        "applyLivePrefs no longer hands the agent picks to the composer: is the pick per project again?"
    enter = APP[APP.index("    void enterForeground()"):]
    assert "applyLivePrefs()" in enter[: enter.index("\n    }\n")]
    flat = _flat(MANUAL)
    assert "each open project keeps its own" not in flat
    for heading in ("### The model", "## Agent"):
        assert "one setting for every open project" in _section(heading), \
            f"{heading}: does not say the picks are one setting for every open project"


# ---------------------------------------------------------------- S9.R11

# What the snapshot carries besides the conversation and the measurements, and the words the
# manual owes for it.
CARRIED = {
    '"available_plugins"': "plug-ins installed",
    '"audio_inputs"': "audio and MIDI devices",
    '"midi_inputs"': "audio and MIDI devices",
    '"sample_path"': "file paths",
    'getSourceFile().getFullPathName()': "file paths",
}


def test_the_privacy_sentences_name_what_a_turn_carries():
    """S9.R11 a (James, 2026-10-06: "name what is sent - names, paths with home as `~`, device
    names - never audio"). The manual said "the only thing sent to the cloud is the conversation
    and those measurements"; a turn also carries a description of the project. Both places that
    say what is sent say all of it."""
    for field in CARRIED:
        assert field in SNAPSHOT_H, f"the snapshot no longer carries {field}: the privacy sentences may say too much"
    intro = _section("## 1. What Zammoth is", whole=False)
    posture = _section("## Privacy posture")
    assert "The only thing sent" not in intro
    for where, text in (("What Zammoth is", intro), ("Privacy posture", posture)):
        for words in sorted(set(CARRIED.values())):
            assert words in text, f"{where}: does not say a turn carries the {words}"
        assert "`~`" in text, f"{where}: does not say the home folder is written as `~`"
        assert re.search(r"never (any )?audio|[Nn]o audio", text), f"{where}: does not say audio is never sent"
    # The requests go to Anthropic on the user's key (Phase 53): no "service" of ours receives them.
    assert "The service receives" not in posture


def test_the_manual_says_the_first_split_downloads_its_model():
    """S9.R11 b (closed by Py8; pinned here with its other half). "Local AI separation (no
    upload)" is true of the audio and was silent on the one-time model fetch."""
    split = _section("## Splitting a track into stems (AI)")
    assert "downloads the separation model" in split and "huggingface.co" in split
    assert "Your audio never leaves the computer" in split
