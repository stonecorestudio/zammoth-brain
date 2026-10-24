"""Phase 54, step 6: what the prompt audit of 2026-10-12 fixed, held in place.

The audit (`docs/audits/prompt-2026-10-12/`) read every word the engineer is sent. Most of its
findings are judgement and can only be checked by a billed eval; these three are facts about
the repository, so a test reads them:

- the DAW's prompt told the model to use five tools the DAW's chat is never given (they are
  `daw_command` keys there), and named two context fields its context does not carry;
- two sentences shared control reverses were still said to the DAW's model;
- model-facing text carried dates, phase numbers and the history of its own fixes, which the
  model can do nothing with.
"""
from __future__ import annotations

import re

from orchestrator import agents, chat


def _names(text: str) -> set[str]:
    return set(re.findall(r"`([a-z_]+)`", text)) | set(re.findall(r"\b([a-z]+_[a-z_]+)\b", text))


def _gateway_tools() -> set[str]:
    return {d["name"] for d in (chat._READ_TOOL_DEFS + chat._WORKSPACE_TOOL_DEFS
                                + chat._IDLE_TOOL_DEFS + chat._DURING_RUN_TOOL_DEFS)}


def _daw_tools() -> set[str]:
    """What the DAW's chat is given. Only the offline gateway passes `during_run`, so the
    DAW's during-run set (`file_instruction`, `stop_run`) is never the model's there."""
    from pathlib import Path
    bridge = (Path(__file__).resolve().parents[2] / "orchestrator" / "daw_bridge.py").read_text()
    assert "during_run" not in bridge, "the DAW now passes during_run: widen this set"
    return {d["name"] for d in chat.daw_tool_defs(False)}


def test_the_daw_prompt_names_no_tool_the_daw_chat_lacks():
    missing = (_names(chat.DAW_SYSTEM_PROMPT) & _gateway_tools()) - _daw_tools()
    assert not missing, f"the DAW prompt tells the model to use tools it is not given: {sorted(missing)}"


def test_the_control_the_offline_prompt_still_names_its_own_tools():
    """The swap is the DAW's alone: the offline workspace has those tools and is told so."""
    named = _names(chat.SYSTEM_PROMPT) & _gateway_tools()
    assert {"transport", "select_stem", "solo_stem", "list_files", "import_sample"} <= named
    assert {"transport", "select_stem"} - _daw_tools(), "the control needs a tool the DAW lacks"


def test_the_daw_prompt_names_only_context_its_context_carries():
    prompt = chat.DAW_SYSTEM_PROMPT
    for gone in ("agent_mode", "workspace.selection", "the gateway tools", "web workspace"):
        assert gone not in prompt, gone
    assert "daw_edit_state.selection" in prompt


def test_the_daw_prompt_does_not_say_what_shared_control_reverses():
    prompt = chat.DAW_SYSTEM_PROMPT
    assert "manual channel is only" not in prompt
    assert "not a user routing surface" not in prompt and "agent-owned returns" not in prompt
    # ...and the offline engine, where both are still true, still says them
    assert "manual channel is only" in chat.SYSTEM_PROMPT


def test_every_swap_replaces_a_sentence_the_base_prompt_really_says():
    for base, daw in chat._DAW_SWAPS:
        assert chat.SYSTEM_PROMPT.count(base) == 1, base[:60]
        assert base not in chat.DAW_SYSTEM_PROMPT
        assert not daw or daw in chat.DAW_SYSTEM_PROMPT


def _model_facing() -> dict[str, str]:
    out = {"the DAW prompt": chat.DAW_SYSTEM_PROMPT, "the offline prompt": chat.SYSTEM_PROMPT,
           "the planner prompt": agents.SYSTEM_PROMPT, "the reply note": chat.REPLY_NOTE}
    for d in chat.daw_tool_defs(False) + chat.daw_tool_defs(True) + agents.PLANNER_TOOL_DEFS:
        params = (d.get("input_schema", {}).get("properties") or {}).values()
        out[f"tool {d['name']}"] = " ".join(
            [d.get("description", "")]
            + [str(v.get("description", "")) for v in params if isinstance(v, dict)])
    return out


def test_no_model_facing_text_carries_a_date_or_a_phase_number():
    """A date or "Phase 43" is the repository's history. The model was never shown the
    version before, so "since 2026-09-08 a bus too" only implies a rule it cannot see."""
    found = {}
    for where, text in _model_facing().items():
        hits = re.findall(r"20\d\d-\d\d-\d\d|\bPhase \d+", text)
        if hits:
            found[where] = hits
    assert not found, found
