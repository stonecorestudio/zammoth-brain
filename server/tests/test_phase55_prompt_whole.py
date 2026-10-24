"""Phase 55, step 5: the DAW engineer's prompt written as ONE document, beside the one that
ships (`orchestrator/prompts.py`).

The prompt that ships is the archived web workspace's prompt + "What changes in the DAW" +
a table of swapped sentences. The whole one says the same things once. It is NOT the
default: `MIX_AGENT_PROMPT=whole` selects it in a source checkout, and it replaces the
default only after a billed comparison. Until then these hold the two together."""
from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

from orchestrator import chat, eval_provenance, prompts

REPO = Path(__file__).resolve().parents[2]
WHOLE = prompts.DAW_PROMPT_WHOLE
SHIPS = chat.DAW_SYSTEM_PROMPT


# ---------------------------------------------------------------- which one runs

def test_the_default_is_the_prompt_that_ships(monkeypatch):
    monkeypatch.delenv(chat.PROMPT_ENV, raising=False)
    assert chat.daw_prompt() is SHIPS
    monkeypatch.setenv(chat.PROMPT_ENV, "something-else")
    assert chat.daw_prompt() is SHIPS


def test_the_variable_selects_the_whole_prompt_in_a_source_checkout(monkeypatch):
    monkeypatch.setenv(chat.PROMPT_ENV, "whole")
    assert chat.daw_prompt() is WHOLE and WHOLE != SHIPS


def test_a_shipped_brain_never_reads_the_variable(monkeypatch):
    monkeypatch.setenv(chat.PROMPT_ENV, "whole")
    monkeypatch.setattr(chat, "is_frozen", lambda: True)
    assert chat.daw_prompt() is SHIPS


def test_everyone_who_builds_a_daw_session_asks_the_one_selector():
    """The brain and the three evals: with a constant imported at one of them, the variable
    would move some of the comparison and not the rest."""
    for name in ("daw_bridge.py", "understanding.py", "qa_eval.py", "operate_eval.py",
                 "eval_provenance.py", "turn_host.py"):
        source = (REPO / "orchestrator" / name).read_text()
        assert "DAW_SYSTEM_PROMPT" not in source, name
    for name in ("daw_bridge.py", "understanding.py", "qa_eval.py", "operate_eval.py"):
        assert "ChatSession(system=daw_prompt()," in (REPO / "orchestrator" / name).read_text(), name


def test_the_eval_stamp_moves_with_the_selection(monkeypatch):
    monkeypatch.delenv(chat.PROMPT_ENV, raising=False)
    default = eval_provenance.prompt_fingerprint()
    monkeypatch.setenv(chat.PROMPT_ENV, "whole")
    assert eval_provenance.prompt_fingerprint() != default, \
        "a checkpoint taken on one prompt would be resumed on the other"


def test_a_session_on_the_whole_prompt_sends_the_reply_note():
    assert chat.is_daw_prompt(WHOLE) and chat.is_daw_prompt(SHIPS)
    assert not chat.is_daw_prompt(chat.SYSTEM_PROMPT)
    assert chat.ChatSession(client=object(), system=WHOLE)._reply_note == chat.REPLY_NOTE


# ---------------------------------------------------------------- the same substance

def _ticked(text: str) -> set[str]:
    return set(re.findall(r"`([^`\n]+)`", text))


def test_both_prompts_name_the_same_tools_keys_and_fields():
    """Everything in backticks: tool names, command keys with their grammar, state fields,
    argument names, literal values."""
    assert _ticked(SHIPS) - _ticked(WHOLE) == set(), "the whole prompt lost a name"
    assert _ticked(WHOLE) - _ticked(SHIPS) == set(), "the whole prompt names something new"


def test_both_prompts_state_the_same_ranges_and_action_types():
    ranges = lambda t: sorted(re.findall(r"-?\d+(?:\.\d+)?\.\.-?\d+(?:\.\d+)?", t))   # noqa: E731
    assert ranges(SHIPS) == ranges(WHOLE)
    types = lambda t: sorted(re.findall(r'type:"(\w+)"', t))                           # noqa: E731
    assert types(SHIPS) == types(WHOLE) and len(set(types(WHOLE))) > 30


def test_the_whole_prompt_is_not_longer():
    assert len(WHOLE.split()) <= len(SHIPS.split())
    assert len(WHOLE.split()) > 0.9 * len(SHIPS.split()), "a section was dropped"


def test_every_contract_bullet_of_the_grammar_is_carried_word_for_word():
    """The action grammar is a tool contract: one bullet per action type, identical in both
    (the two routing bullets are reworded, and say so)."""
    bullets = [line for line in SHIPS.split("\n") if re.match(r'- \w+: \{type:"', line)]
    assert len(bullets) > 25
    reworded = ("- group:", "- return_delay:")
    for bullet in bullets:
        if bullet.startswith(reworded):
            continue
        assert bullet in WHOLE, bullet[:50]


# ---------------------------------------------------------------- what is different, on purpose

SECTIONS = ("Ground rules:", "A command, a mix move, or a run:",
            "The catalog and the live session:", "Craft:", "Reading the audio:",
            "Action grammar for `propose`", "Metric grammar", "Capabilities beyond the single move:",
            "Asking and planning:", "Scope and boundaries:", "How you write to the user:")


def test_it_is_one_document_in_one_order():
    at = [WHOLE.index("\n" + s) for s in SECTIONS]
    assert at == sorted(at)
    assert WHOLE.startswith("You are the resident mix engineer inside a native DAW")
    for gone in ("What changes", "stem-mixing workspace", "headless audio engine",
                 "You cannot change audio state directly"):
        assert gone not in WHOLE, gone
        assert gone in SHIPS, f"control: the prompt that ships no longer says {gone!r}"


def test_a_level_is_a_command_and_is_said_once():
    """The prompt that ships lists "pull the guitars back" under `propose` and, later, says a
    fader or a level is a command."""
    doors = WHOLE.split("A command, a mix move, or a run:")[1].split("The catalog and the live session:")[0]
    command, propose = doors.split("- **`propose`")
    assert '`track.set_gain` ("pull the guitars back")' in " ".join(command.split())
    assert "pull the guitars back" not in propose
    assert WHOLE.count("pull the guitars back") == 1
    ships = " ".join(SHIPS.split("Route by what the user actually wants")[1]
                     .split("`daw_command`")[0].split())
    assert "pull the guitars back" in ships and "`propose`" in ships       # control: the contradiction
    for door in ("`daw_command`", "`propose`", "`start_run`", "`auto_mix`", "A question"):
        assert door in doors, door


def test_routing_in_chat_is_said_as_commands_in_the_grammar_too():
    flat = " ".join(WHOLE.split())
    assert "in chat you route with commands (above)" in flat
    assert "You own routing" not in WHOLE and "You own routing" in SHIPS
    assert "`group.create`, `route.stem`, `bus.route`, `return.add` and `send.set`" in flat


def test_undo_that_names_both_undos():
    rule = next(line for line in WHOLE.split("\n") if line.startswith("- **Do it; do not describe it.**"))
    assert "`undo_proposal` for a mix move" in rule and "`edit.undo` for a command" in rule


def test_the_three_lines_about_what_the_model_gets_wrong_say_the_behaviour_wanted():
    for gone in ("under-reach", "you will forget you have", "most likely to miss",
                 "the failure it turns into", "Nothing in the catalog is withheld"):
        assert gone not in " ".join(WHOLE.split()), gone
        assert gone in " ".join(SHIPS.split()), f"control: {gone!r}"
    assert "fire the scene or the slot rather than describing it" in " ".join(WHOLE.split())


# ---------------------------------------------------------------- every pin on the one that ships

PINS = {
    "test_user_qa_2026_10_22": ("test_the_daw_prompt_does_not_say_measuring_a_reference_loads_it",),
    "test_phase54_prompt_audit": ("test_the_daw_prompt_names_no_tool_the_daw_chat_lacks",
                                  "test_the_daw_prompt_names_only_context_its_context_carries",
                                  "test_the_daw_prompt_does_not_say_what_shared_control_reverses",
                                  "test_no_model_facing_text_carries_a_date_or_a_phase_number"),
    "test_knowledge_pack": ("test_the_daw_prompt_names_the_capabilities_the_agent_under_reaches_for",
                            "test_each_named_capability_carries_a_trigger_not_just_a_name",
                            "test_every_named_command_actually_exists_in_the_catalog"),
    "test_after_turn_queue": ("test_the_prompt_names_every_command_that_waits_and_what_can_be_read_now",),
    "test_reader_p3_py8": ("test_the_daw_prompt_names_no_gating_argument",
                           "test_the_daw_prompt_does_not_say_a_runs_routing_is_invisible"),
    "test_daw_chat": ("test_the_daw_prompt_asks_for_explanation_and_noticing",),
    "test_qa_leftovers_2026_10_09": ("test_the_prompt_says_where_the_key_is_and_that_none_is_none",
                                     "test_the_prompt_says_a_pan_is_quoted_as_the_header_shows_it"),
    "test_showcase2_rehearsal_fixes": ("test_the_agent_is_told_what_the_two_fields_are",),
    "test_awareness_automix": ("test_the_prompt_names_device_enable_as_the_native_bypass",),
    "test_phase54_reply_contract": ("test_the_reply_contract_is_the_last_section_of_the_daw_prompt",
                                    "test_the_rules_that_asked_for_more_words_were_reworded_not_left_to_compete",
                                    "test_the_daw_prompt_does_not_promise_approve_and_reject_cards",
                                    "test_the_prompt_asks_for_independent_calls_in_one_response",
                                    "test_the_daw_session_carries_the_note_and_the_offline_one_does_not",
                                    "test_a_bullet_is_one_line"),
    "test_phase60_sound_design": ("test_the_tool_and_both_prompts_say_how_a_kind_is_given",),
}


@pytest.mark.parametrize("module,name", [(m, n) for m, names in PINS.items() for n in names])
def test_a_pin_on_the_prompt_that_ships_holds_on_the_whole_one(module, name, monkeypatch):
    """Each test that reads `DAW_SYSTEM_PROMPT` is run again with the whole prompt in its
    place - the module's own copy of the name too, where it imported one."""
    mod = importlib.import_module(f"tests.{module}")
    monkeypatch.setattr(chat, "DAW_SYSTEM_PROMPT", WHOLE)
    if hasattr(mod, "DAW_SYSTEM_PROMPT"):
        monkeypatch.setattr(mod, "DAW_SYSTEM_PROMPT", WHOLE)
    getattr(mod, name)()


def test_the_pin_list_is_every_argument_free_test_that_reads_the_prompt():
    """A new pin on the shipping prompt joins the list, or the whole prompt can drift from it."""
    import ast
    left_out = {"test_every_swap_replaces_a_sentence_the_base_prompt_really_says",   # about the base
                "test_the_control_the_offline_prompt_still_names_its_own_tools"}
    found: dict[str, set[str]] = {}
    for path in sorted((REPO / "server" / "tests").glob("test_*.py")):
        if path.name == Path(__file__).name:
            continue
        source = path.read_text()
        if "DAW_SYSTEM_PROMPT" not in source:
            continue
        helpers = {n.name for n in ast.parse(source).body
                   if isinstance(n, ast.FunctionDef) and not n.name.startswith("test_")
                   and "DAW_SYSTEM_PROMPT" in (ast.get_source_segment(source, n) or "")}
        for node in ast.parse(source).body:
            if not (isinstance(node, ast.FunctionDef) and node.name.startswith("test_")):
                continue
            body = ast.get_source_segment(source, node) or ""
            reads = "DAW_SYSTEM_PROMPT" in body or any(h + "(" in body for h in helpers)
            if reads and not node.args.args and not node.decorator_list and node.name not in left_out:
                found.setdefault(path.stem, set()).add(node.name)
    listed = {m: set(names) for m, names in PINS.items()}
    assert found == listed, {m: sorted(found.get(m, set()) ^ listed.get(m, set()))
                             for m in set(found) | set(listed)
                             if found.get(m, set()) != listed.get(m, set())}
