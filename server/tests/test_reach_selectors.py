"""Only the tests and legs a change reaches (James, 2026-10-11).

"can you make it so this repo doesnt do test suites that are unnecessary to what is being fixed
or added, only run tests the diff reaches and touches". Two selectors answer that from the
diff's own lines: `server/tests/owed_by_diff.py --tests` (one pytest test at a time) and
`daw/scripts/legs_for.py` (console legs). The file-by-file answer the first one used to give
was 2,240 tests for one commit, because half the suite names `App.cpp` or the manual somewhere.

Every case here runs the selectors on a small tree made in the test - a wrong selection is
seen as a wrong selection, not as whatever the real suite holds this week - except the last
three, which pin the written rule to the tools.
"""
from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SKILL = REPO / ".claude" / "skills" / "quality-gates" / "SKILL.md"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------------------------
# pytest: one test at a time
# ---------------------------------------------------------------------------------------------

SAMPLE = '''
    from pathlib import Path
    import re
    from mix_engine import analysis
    from orchestrator.operate_eval import CORPUS as OPERATE

    REPO = Path(__file__).resolve().parents[2]
    SRC = REPO / "daw" / "Source"
    APP = (SRC / "App.cpp").read_text()


    def _in_app(needle):
        return needle in APP


    def test_pins_a_sentence():
        assert _in_app("queueAfterTurn (args)")


    def test_pins_by_regex():
        assert re.search(r"case\\s+dawcmd::clipWarp\\s*:.*?break;", APP, re.S)


    def test_reads_another_file():
        assert (SRC / "DawSession.h").read_text()


    def test_walks_the_folder():
        for source in SRC.glob("*.h"):
            assert "\\t" not in source.read_text()


    def test_names_a_symbol(monkeypatch):
        monkeypatch.setattr(analysis, "TEMPO_METER_MARGIN", 9.0)


    def test_names_the_corpus():
        assert OPERATE


    def test_same_word_another_module():
        from orchestrator import qa_eval
        assert qa_eval.CORPUS


    def test_names_nothing():
        assert 1 + 1 == 2


    class TestGrouped:
        def test_reads_the_manual(self):
            manual = (REPO / "docs" / "17-user-manual.md").read_text()
            assert "What tempo is the file in?" in manual
'''


@pytest.fixture
def owed(tmp_path, monkeypatch):
    """The test selector, pointed at one synthetic test module (the repository stays the real
    one: the paths that module names must exist)."""
    module = _load(REPO / "server" / "tests" / "owed_by_diff.py", "owed_by_diff_under_test")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_sample.py").write_text(textwrap.dedent(SAMPLE), encoding="utf-8")
    monkeypatch.setattr(module, "TESTS", tests)
    return module


def _change(module, text: str, lines=(10,)):
    return module.Change(frozenset(lines), text, " ".join(text.split()))


def _picked(module, changes) -> dict[str, str]:
    chosen, _notes = module.owed_tests(changes)
    return {test_id.split("::", 1)[-1]: why for test_id, why in chosen.items()}


def test_a_name_too_long_to_be_a_file_is_not_a_path(owed):
    """A sentence of a test's text is tried as a path like any string; one past the file system's
    name limit made `Path.exists()` RAISE (ENAMETOOLONG) and the whole selector crash."""
    sentence = "The transitive rule " + "x" * 300 + " (a docstring, not a file)"
    assert owed._repo_path(sentence) is None
    assert owed._repo_path("daw/" + sentence) is None          # a real top-level folder, then the sentence
    assert owed._repo_path("nul\0in it") is None              # a NUL is a ValueError, not an OSError
    assert owed._repo_path("server/tests/owed_by_diff.py") == "server/tests/owed_by_diff.py"   # control


def test_a_line_no_test_looks_for_reaches_only_the_folder_guard(owed):
    picked = _picked(owed, {"daw/Source/ArrangeView.h": _change(owed, "    int qqqUnrelated = 1;")})
    assert set(picked) == {"test_walks_the_folder"}, picked
    assert picked["test_walks_the_folder"].startswith("walks daw/Source")
    assert not _picked(owed, {"daw/Source/App.cpp": _change(owed, "    int qqqUnrelated = 1;")})


def test_a_pinned_sentence_on_a_changed_line_reaches_its_test(owed):
    picked = _picked(owed, {"daw/Source/App.cpp": _change(
        owed, "    if (inOwnTurn()) return queueAfterTurn (args);")})
    assert "test_pins_a_sentence" in picked and "test_pins_by_regex" not in picked, picked
    assert "queueAfterTurn" in picked["test_pins_a_sentence"]


def test_a_block_long_regex_is_reached_through_the_name_in_it(owed):
    """The regex spans a whole handler; the diff shows eight lines of it. What it names does."""
    picked = _picked(owed, {"daw/Source/App.cpp": _change(
        owed, "            case dawcmd::clipWarp:\n                {\n                    auto x = 1;")})
    assert "test_pins_by_regex" in picked and "test_pins_a_sentence" not in picked, picked


def test_a_test_with_nothing_to_judge_by_is_owed(owed):
    picked = _picked(owed, {"daw/Source/DawSession.h": _change(owed, "inline int anything() { return 1; }")})
    assert "test_reads_another_file" in picked, picked
    assert "nothing to judge" in picked["test_reads_another_file"]
    assert "test_pins_a_sentence" not in picked, "it opens App.cpp, not the file that changed"


def test_a_walk_takes_only_what_its_pattern_takes_and_a_new_file_only_who_names_it(owed):
    """`SRC.glob ("*.h")` is no guard over a `.cpp`; and a file that did not exist is owed by
    the tests that walk to it or name it, never by one that runs another file of its folder."""
    cpp = _picked(owed, {"daw/Source/Other.cpp": _change(owed, "int x;")})
    header = _picked(owed, {"daw/Source/Other.h": _change(owed, "int x;")})
    assert not cpp and set(header) == {"test_walks_the_folder"}, (cpp, header)
    added = _picked(owed, {"daw/Source/BrandNew.h": owed.Change(None, "int x;", "int x;", True)})
    assert set(added) == {"test_walks_the_folder"}, added


def test_a_new_or_deleted_file_reaches_everything_that_names_it(owed):
    picked = _picked(owed, {"docs/17-user-manual.md": owed.Change(None, "", "")})
    assert set(picked) == {"TestGrouped::test_reads_the_manual"}, picked


def test_a_manual_change_reaches_the_test_of_that_sentence_only(owed):
    there = _picked(owed, {"docs/17-user-manual.md": _change(owed, "### What tempo is the file in?\n\nA clip")})
    elsewhere = _picked(owed, {"docs/17-user-manual.md": _change(owed, "The Loop Deck has eight scenes.")})
    assert set(there) == {"TestGrouped::test_reads_the_manual"} and not elsewhere, (there, elsewhere)


def _line_of(relative: str, pattern: str) -> int:
    for number, line in enumerate((REPO / relative).read_text(encoding="utf-8").splitlines(), 1):
        if re.search(pattern, line):
            return number
    raise AssertionError(f"{pattern} is not in {relative}")


def test_a_changed_constant_reaches_the_test_that_names_it_with_its_module(owed):
    at = _line_of("server/mix_engine/analysis.py", r"^TEMPO_METER_MARGIN\s*=")
    picked = _picked(owed, {"server/mix_engine/analysis.py": _change(owed, "TEMPO_METER_MARGIN = 0.2", (at,))})
    assert set(picked) == {"test_names_a_symbol"}, picked


def test_a_word_five_modules_use_is_tied_to_its_own(owed):
    at = _line_of("orchestrator/operate_eval.py", r"^CORPUS\b.*=")
    picked = _picked(owed, {"orchestrator/operate_eval.py": _change(owed, "    OpCase (...),", (at + 3,))})
    assert set(picked) == {"test_names_the_corpus"}, picked


def test_a_table_is_named_and_a_function_is_followed_one_step(owed):
    """A prompt or a corpus reaches the tests that name it; every function that reads it is not
    a test of its wording. A function reaches its callers in the file, once."""
    corpus = _line_of("orchestrator/operate_eval.py", r"^CORPUS\b.*=")
    assert owed.changed_symbols("orchestrator/operate_eval.py", {corpus + 3}) == {"CORPUS"}
    period = _line_of("server/mix_engine/analysis.py", r"^def _tempo_period\(")
    reached = owed.changed_symbols("server/mix_engine/analysis.py", {period + 2})
    assert {"_tempo_period", "_tempo"} <= reached and "analyze_session" not in reached, reached
    assert owed.changed_symbols("server/mix_engine/analysis.py", {period + 2}, depth=0) == {"_tempo_period"}


def test_a_changed_test_file_is_run_whole_and_the_harness_owes_everything(owed):
    picked, _ = owed.owed_tests({"server/tests/test_sample.py": owed.Change(None, "", "")})
    assert list(picked) == ["tests/test_sample.py"]
    everything, notes = owed.owed_tests({"server/tests/conftest.py": _change(owed, "x = 1")})
    assert list(everything) == ["tests/test_sample.py"] and "harness" in notes[0]


def test_the_diff_is_read_with_its_neighbourhood(tmp_path, monkeypatch):
    """Changed lines by number, the text either side of them (a pinned sentence wraps), a new
    file whole."""
    module = _load(REPO / "server" / "tests" / "owed_by_diff.py", "owed_by_diff_in_a_scratch_repo")
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
                       cwd=repo, check=True, capture_output=True)

    git("init", "-q")
    (repo / "a.txt").write_text("".join(f"line {n}\n" for n in range(1, 31)))
    git("add", "a.txt")
    git("commit", "-q", "-m", "first")
    (repo / "a.txt").write_text("".join(("moved\n" if n == 12 else f"line {n}\n") for n in range(1, 31)))
    (repo / "new.txt").write_text("brand new\n")
    monkeypatch.setattr(module, "REPO", repo)
    changes = module.diff_of("HEAD")
    assert changes["a.txt"].lines == frozenset({12})
    assert "moved" in changes["a.txt"].text and "line 12" in changes["a.txt"].text
    assert "line 5" in changes["a.txt"].text and "line 2 " not in changes["a.txt"].text + " "
    assert changes["new.txt"].lines is None and "brand new" in changes["new.txt"].text


# ---------------------------------------------------------------------------------------------
# console legs
# ---------------------------------------------------------------------------------------------

MAIN_CPP = '''
    static int runOldStyle (te::Edit& edit)
    {
        // a brace in a comment } and one in a string: "}"
        const char* text = "{ not a block";
        return dawsession::setClipWarp (edit) ? 0 : 1;
    }

    int main (int argc, char** argv)
    {
        if (argc > 1)
        {
            const juce::String flag (argv[1]);
            if (flag == "--new-style") return dawapp::runNewStyle (project);
            if (flag == "--wide-a") return dawapp::runWideA (project);
            if (flag == "--wide-b") return dawapp::runWideB (project);
        }
        bool oldStyle = false, inlineTest = false;
        for (auto& a : args)
        {
            if (a == "--old-style") oldStyle = true;
            else if (a == "--inline") inlineTest = true;
            else if (a == "--play") play = true;
        }
        if (oldStyle)   // no project needed
            return runOldStyle (edit);
        if (inlineTest)
        {
            auto reply = call ("track.set_gain", args);
            return reply.isVoid() ? 1 : 0;
        }
        return 0;
    }
'''

LEGS_H = '''
    namespace dawapp
    {
        int runNewStyle (const juce::File&)
        {
            auto r = d.invokeCommand ("clip.warp", args);
            return agentWarpClip (track, 0, true) ? 0 : 1;
        }

        int runWideA (const juce::File&) { return common() + helperEverybodyUses(); }
        int runWideB (const juce::File&) { return common() + helperEverybodyUses(); }
    }
'''

APP_CPP = '''
    juce::var DawComponent::invokeCommand (int id, const juce::var& args,
                                           bool forAgent)
    {
        if (auto bad = dawcmd::badNumericArg (args))
            return err (bad);
        switch (id)
        {
            case dawcmd::setStemGain:
            {
                const double db = (double) args["gain_db"];
                return setGain (db);
            }
            case dawcmd::clipWarp:
            {
                const bool on = args.getProperty ("on", true);
                return warp (on);
            }
        }
        return {};
    }

    inline void setClipWarp (te::Edit& edit, te::AudioClipBase& acb,
                             bool on, double sourceBpm = 0.0)
    {
        if (on)
            acb.setAutoTempo (true);
        helperEverybodyUses();
    }

    void helperEverybodyUses()
    {
        ++count;
    }
'''

COMMANDS_H = '''
    static const Row rows[] = {
        { setStemGain,            "track.set_gain",         "Set Track Gain",      "Track",     0, 0, true  },
        { clipWarp,               "clip.warp",              "Warp Clip to Tempo",  "Clip",      0, 0, true  },
    };
    static const Schema schemas[] = {
        { "track.set_gain", "{ track, gain_db: -60..+6 }" },
        { "clip.warp", "{ track, clip, on?, source_bpm?: 20-300 }" },
    };
'''

CATALOGUE = '''
    "$A" --new-style test_projects/suno_demo   # clip.warp with a stated tempo
    "$A" --old-style test_projects/suno_demo   # the stretcher keeps time over a long file
    "$A" --inline    test_projects/suno_demo   # the fader stops at +6
'''


@pytest.fixture
def legs_for(tmp_path, monkeypatch):
    module = _load(REPO / "daw" / "scripts" / "legs_for.py", "legs_for_under_test")
    repo = tmp_path / "repo"
    source = repo / "daw" / "Source"
    source.mkdir(parents=True)
    (repo / "native" / "Source").mkdir(parents=True)
    (repo / ".claude" / "skills" / "daw").mkdir(parents=True)
    for name, body in (("Main.cpp", MAIN_CPP), ("SampleLegs.h", LEGS_H), ("App.cpp", APP_CPP),
                       ("DawCommands.h", COMMANDS_H)):
        (source / name).write_text(textwrap.dedent(body), encoding="utf-8")
    (repo / ".claude" / "skills" / "daw" / "SKILL.md").write_text(textwrap.dedent(CATALOGUE), encoding="utf-8")
    monkeypatch.setattr(module, "REPO", repo)
    monkeypatch.setattr(module, "TOO_WIDE", 1)
    return module


def _app_line(module, needle: str) -> int:
    for number, line in enumerate(module._read("daw/Source/App.cpp").splitlines(), 1):
        if needle in line:
            return number
    raise AssertionError(needle)


def test_every_way_a_flag_reaches_its_code_is_indexed(legs_for):
    index = legs_for.legs()
    assert set(index) == {"--new-style", "--wide-a", "--wide-b", "--old-style", "--inline"}, set(index)
    assert "agentWarpClip" in index["--new-style"].text
    assert "setClipWarp" in index["--old-style"].text, "a flag variable leads to its `if`, and that to its function"
    assert "{ not a block" in index["--old-style"].text and "int main" not in index["--old-style"].text
    assert '"track.set_gain"' in index["--inline"].text and "return 0;\n}" not in index["--inline"].text
    assert index["--new-style"].catalogue.startswith("test_projects")


def test_the_line_names_its_function_and_under_a_case_its_command(legs_for):
    warp = legs_for.names_at("daw/Source/App.cpp", {_app_line(legs_for, 'args.getProperty ("on"')})
    assert warp == {"clip.warp"}, "the command, not the function every command passes"
    door = legs_for.names_at("daw/Source/App.cpp", {_app_line(legs_for, "return err (bad)")})
    assert door == {"invokeCommand"}, door
    wrapped = legs_for.names_at("daw/Source/App.cpp", {_app_line(legs_for, "acb.setAutoTempo")})
    assert wrapped == {"setClipWarp"}, "a signature that wraps is still the function"
    row = legs_for.names_at("daw/Source/DawCommands.h", {
        next(n for n, line in enumerate(legs_for._read("daw/Source/DawCommands.h").splitlines(), 1)
             if "source_bpm" in line)})
    assert row == {"clip.warp"}, row


@pytest.mark.parametrize("line,name", [
    ("inline void setClipWarp (te::Edit& edit, te::AudioClipBase& acb,", "setClipWarp"),
    ("    juce::var DawComponent::invokeCommand (int id, const juce::var& args,", "invokeCommand"),
    ("    void helperEverybodyUses()", "helperEverybodyUses"),
    ("        if (auto bad = dawcmd::badNumericArg (args))", None),
    ("            return setGain (db);", None),
    ("        auto reply = call (\"track.set_gain\",", None),
    ("        acb.setOffset (srcStartBefore * beats,", None),
    ("    for (auto* t : te::getAudioTracks (edit))", None),
])
def test_a_definition_is_told_from_a_call(legs_for, line, name):
    assert legs_for._function_named(line) == name


def test_a_change_reaches_the_legs_that_name_what_changed(legs_for, monkeypatch):
    def reached(app_needle: str) -> tuple[dict, list]:
        monkeypatch.setattr(legs_for, "changed_lines",
                            lambda base="HEAD": {"daw/Source/App.cpp": {_app_line(legs_for, app_needle)}})
        return legs_for.select([], [], "HEAD")

    chosen, _ = reached('args.getProperty ("on"')
    assert set(chosen) == {"--new-style"}, chosen                # by the command key, in its code
    chosen, _ = reached("acb.setAutoTempo")
    assert set(chosen) == {"--old-style"}, chosen                # by the function
    chosen, notes = reached("++count")
    assert not chosen and any("helperEverybodyUses: not followed" in n for n in notes), (chosen, notes)
    chosen, notes = reached("return err (bad)")
    assert not chosen and any("invokeCommand: not followed" in n for n in notes), (chosen, notes)


def test_a_changed_leg_is_owed_and_a_name_or_a_word_can_be_given(legs_for, monkeypatch):
    main = legs_for._read("daw/Source/Main.cpp").splitlines()
    inside = next(n for n, line in enumerate(main, 1) if "not a block" in line)
    monkeypatch.setattr(legs_for, "changed_lines", lambda base="HEAD": {"daw/Source/Main.cpp": {inside}})
    chosen, _ = legs_for.select([], [], "HEAD")
    assert chosen == {"--old-style": "the diff changed this leg"}, chosen

    chosen, _ = legs_for.select(["track.set_gain"], [], None)
    assert set(chosen) == {"--inline"}, chosen
    chosen, _ = legs_for.select(["helperEverybodyUses"], [], None)
    assert set(chosen) == {"--wide-a", "--wide-b"}, "a name that was ASKED for is followed, however wide"
    chosen, _ = legs_for.select([], ["stretcher", "fader"], None)
    assert set(chosen) == {"--old-style", "--inline"}, chosen
    chosen, notes = legs_for.select([], ["nonesuch"], None)
    assert not chosen and "nonesuch" in notes[0]


# ---------------------------------------------------------------------------------------------
# the written rule and the tools say the same
# ---------------------------------------------------------------------------------------------

def test_the_smoke_set_is_one_list():
    """The skill's section 2 and the tool's `SMOKE` are the same twelve, and each is a leg."""
    module = _load(REPO / "daw" / "scripts" / "legs_for.py", "legs_for_real")
    skill = SKILL.read_text(encoding="utf-8")
    section = skill[skill.index("**2. The smoke set**"):skill.index("**3. The whole suites")]
    listed = section[section.index("pinned against this section"):section.index("```bash")]
    assert tuple(re.findall(r"--[a-z][a-z0-9-]*", listed)) == module.SMOKE, listed
    flags = set(re.findall(r'"(--[a-z][a-z0-9-]*)"', (REPO / "daw" / "Source" / "Main.cpp").read_text(encoding="utf-8")))
    assert set(module.SMOKE) <= flags, sorted(set(module.SMOKE) - flags)


def test_the_rule_is_where_a_new_session_reads_it():
    claude = (REPO / "CLAUDE.md").read_text(encoding="utf-8")
    skill = SKILL.read_text(encoding="utf-8")
    daw = (REPO / ".claude" / "skills" / "daw" / "SKILL.md").read_text(encoding="utf-8")
    for text, where in ((claude, "CLAUDE.md"), (skill, "the quality-gates skill")):
        assert "owed_by_diff.py --tests" in text, where
        assert "legs_for.py" in text, where
    assert "legs_for.py" in daw
    # the two rules this replaced: a whole run as a per-change default
    assert "the whole server pytest, once" not in skill
    assert "each one owes the full sweep" not in skill


def test_the_selectors_answer_for_this_tree():
    """Both run on the repository as it is: a syntax error in either is a red here, and the
    leg index holds every flag `Main.cpp` dispatches."""
    owed_tool = _load(REPO / "server" / "tests" / "owed_by_diff.py", "owed_by_diff_real")
    chosen, _ = owed_tool.owed_tests({"daw/Source/App.cpp": owed_tool.Change(
        frozenset({10}), "int qqqUnrelated = 1;", "int qqqUnrelated = 1;")})
    naming = [name for name, paths in owed_tool.suite_reads().items()
              if any(p in ("daw/Source", "daw/Source/App.cpp") for p in paths)]
    assert 0 < len({i.split("::")[0] for i in chosen}) < len(naming) / 2, (len(chosen), len(naming))

    legs_tool = _load(REPO / "daw" / "scripts" / "legs_for.py", "legs_for_real_index")
    index = legs_tool.legs()
    with_code = [flag for flag, leg in index.items() if leg.text.strip()]
    assert len(index) > 500 and len(with_code) > 0.95 * len(index), (len(index), len(with_code))
    assert set(legs_tool.SMOKE) <= set(index)
