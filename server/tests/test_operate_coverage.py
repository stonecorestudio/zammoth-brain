"""The operate corpus must grade every agent-invocable command key.

The total-agent-control invariant (docs/10-daw.md) says every feature is
agent-operable; this is its eval half, enforced in code: a command key the
registry ships with no case whose expect/sequence grades it is a feature the
agent operates blind — the 2026-08-19 audit found 195 of them, discovered only
by counting. The corpus (orchestrator/operate_eval.py) parses the registry
(daw/Source/DawCommands.h) at import, so a NEW command fails this test until
it gets a case — the same day it lands, not at the next audit.

Free: no LLM anywhere in this path (the corpus and registry are data).
"""
from __future__ import annotations


def test_every_agent_invocable_command_key_is_graded_by_a_case():
    from orchestrator.operate_eval import coverage_gaps

    gaps = coverage_gaps()
    assert gaps == [], (
        f"{len(gaps)} agent-invocable command keys have no eval trail — add "
        f"an OpCase (or a COVERAGE_EXEMPT entry WITH a reason) for: {gaps}")


def test_the_registry_parse_stays_plausible():
    # The coverage gate is only as strong as the parse feeding it: a regex
    # that silently found half the table would "pass" the test above with
    # half the invariant. The known-size floor makes that loud.
    from orchestrator.operate_eval import OPERATE_COMMANDS, _registry_commands

    rows = _registry_commands()
    assert len(rows) >= 267, f"registry shrank? parsed {len(rows)} keys"
    assert len(OPERATE_COMMANDS) >= 238
    user_only = {key for key, uo, _ in rows if uo}
    # NOTHING is user-only (2026-10-04, James: "give agent full capabilities").
    # The loop was the canonical user-only key, then the viewport and the UI
    # doors, then - the same day - the keyboard's own keys (playhead nudges,
    # Select Up / Down, the Loop Deck's loop-length keys, Launch Selected
    # Slot). The flag and its refusal remain; this pins the set EMPTY, with
    # MixAgentDAW --commands and --agent-view, so withholding a command from
    # the agent is a decision somebody makes here on purpose.
    assert user_only == set()
    assert len(OPERATE_COMMANDS) == len(rows)


def test_a_transport_error_is_not_a_model_failure():
    """The 2026-08-20 poisoned baseline, pinned.

    One of three runs hit a subscription usage limit; every exception in it
    was graded `wrong`, so the recorded baseline claimed 63.7% and 245
    phantom regressions. An error must be its own verdict, must leave the
    score's denominator, and must never classify a case.
    """
    from orchestrator.operate_eval import OpCase, Outcome, tally

    case = OpCase("stop playback", ("transport.stop",))
    outcomes = [Outcome(case, "correct", "ok"),
                Outcome(case, "error", "RateLimitError: 429")]
    result = tally(outcomes)

    assert result["counts"]["error"] == 1
    assert result["counts"]["wrong"] == 0
    # 1 correct of 1 GRADED case, not 50% of two.
    assert result["score"] == 100.0
    assert result["errors"] == 1


def test_a_poisoned_sweep_refuses_to_become_a_baseline():
    from orchestrator.operate_eval import _MAX_RUN_ERRORS, _refuse_if_poisoned

    clean = {"errors": [0, 1, 0], "scores": [86.5, 87.0, 86.1]}
    poisoned = {"errors": [0, _MAX_RUN_ERRORS + 1, 0], "scores": [86.5, 3.0, 86.1]}

    assert _refuse_if_poisoned(clean, "--check") == 0
    assert _refuse_if_poisoned(poisoned, "--check") == 2


def test_no_command_lies_about_taking_arguments():
    """A command whose handler reads args must DECLARE that it takes them.

    Found 2026-08-20 by the operate sweep: track.solo's handler reads
    {track, on?}, but its Spec said wantsArgs=false and it published no
    schema — so the catalog told the agent "no arguments", the agent spent a
    track.select first to make the command apply to the right track, and the
    eval scored the workaround as over-reach. Five commands had the defect.

    The C++ `--commands` self-test already catches the inverse (wantsArgs
    with no schema); it cannot catch this one, because App.cpp's handlers are
    not compiled into the console harness. So it lives here, in the language
    that can read both files.
    """
    import re
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    header = (repo / "daw/Source/DawCommands.h").read_text()
    app = (repo / "daw/Source/App.cpp").read_text()

    handlers = app.find("juce::var invokeCommand (int id, const juce::var& args)")
    assert handlers > 0, "invokeCommand moved: this test reads the handlers from it"
    in_table, liars = False, []
    for line in header.splitlines():
        if "static const std::vector<Spec> table" in line:
            in_table = True
            continue
        if in_table and line.strip().startswith("};"):
            break
        if not in_table:
            continue
        spec = re.match(
            r'\s*\{\s*(\w+),\s*"([a-z_0-9.]+)"\s*,\s*"[^"]*",\s*"[^"]+",\s*(.*?)\s*\},?'
            r'\s*(?://.*)?$', line)
        if not spec:
            continue
        flags = re.findall(r"\b(?:true|false)\b", spec.group(3))
        if flags and flags[0] == "true":
            continue          # already declares that it takes args
        # The HANDLER's label, not the first one in the file (reader E7·R16): `commandEnabled`
        # sits above `invokeCommand` and has a label for agent.run / agent.cancel / agent.undo,
        # so for those three this read the enable switch - which reads no arguments - and
        # agent.undo, whose handler reads `gesture` twice, passed as honest.
        label = f"case dawcmd::{spec.group(1)}:"
        start = app.find(label, handlers)
        if start < 0:
            continue
        # ...and past the labels that share its body (`case a: case b:`), to the code
        at = start + len(label)
        while (more := re.match(r"\s*case dawcmd::\w+:", app[at:])) is not None:
            at += more.end()
        nxt = app.find("case dawcmd::", at)
        body = app[at:nxt if nxt > 0 else at + 1200]
        if re.search(r'\bsArg\s*\(|args\.hasProperty|args\.getProperty', body):
            liars.append(spec.group(2))

    assert liars == [], (
        f"these commands read arguments but declare wantsArgs=false, so the "
        f"agent is told they take none: {liars}")
