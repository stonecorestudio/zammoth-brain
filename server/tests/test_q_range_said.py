"""An EQ band's Q above the engine's 8 (2026-10-16, a chat turn on a drum loop).

The agent notched an 11.4 kHz ring with a Q past 8, and `propose_actions` refused it with
pydantic's dump: "134 validation errors for propose_actionsArguments", every action type's
mismatch, in the feed as `[TextContent(type='text', text=...)]`. Two causes, one test each:

- the chat's engine call (`turn_host.LiveTurn.engine_call`) put the MCP content list into its
  error as a Python repr, where the run loop's `tool_error_text` words it as the one problem;
- no prompt said what `q` takes for `eq` or `dynamic_eq` - each other bounded field is said
  (`output_db?: -12..12`, overdrive's `q?: 0.3..8`) - so the model learnt the cap by failing.
"""
from __future__ import annotations

import re
from types import SimpleNamespace

import anyio
import pytest
from pydantic import BaseModel, ValidationError

from mix_engine.models import DynamicEqAction, EqBand, ProposalRequest
from orchestrator import agents, chat, prompts, turn_host


class _Arguments(BaseModel):
    req: ProposalRequest


def _q_too_high_dump() -> str:
    proposal = {"rationale": "notch the ring",
                "predicted_impact": [{"metric": "lufs_i", "predicted_delta": 0.0, "unit": "LUFS"}],
                "actions": [{"type": "eq", "stem_id": "drums",
                             "band": {"freq_hz": 11400.0, "gain_db": 0.0, "q": 20.0,
                                      "shape": "notch"}}]}
    with pytest.raises(ValidationError) as caught:
        _Arguments.model_validate({"req": proposal})
    return f"Error executing tool propose_actions: {caught.value}"


class _Text:
    def __init__(self, text: str) -> None:
        self.type, self.text = "text", text

    def __repr__(self) -> str:                      # as the MCP SDK's TextContent prints
        return f"TextContent(type='text', text={self.text!r})"


class _Engine:
    def __init__(self, text: str) -> None:
        self.text = text

    async def call_tool(self, name, args):
        return SimpleNamespace(isError=True, content=[_Text(self.text)], structuredContent=None)


def test_the_chats_engine_call_words_a_refusal_as_the_one_problem():
    host = SimpleNamespace(engine=_Engine(_q_too_high_dump()), edit_state=None, project_dir=None)

    async def call():
        await turn_host.LiveTurn.engine_call(host, "propose_actions", {"req": {}})

    with pytest.raises(RuntimeError) as caught:
        anyio.run(call)
    said = str(caught.value)
    for noise in ("TextContent", "validation error", "Arguments", "[type=", "function-after"):
        assert noise not in said, f"{noise!r} in: {said}"
    assert said.startswith("engine tool 'propose_actions' failed: "), said
    assert "q" in said and "8" in said, said


def _range(model, field: str) -> str:
    meta = model.model_fields[field].metadata
    lo = next(m.ge for m in meta if hasattr(m, "ge"))
    hi = next(m.le for m in meta if hasattr(m, "le"))
    return f"q: {lo:g}..{hi:g}"


def _entry(text: str, kind: str) -> str:
    found = re.search(rf"(?m)^- {kind}: \{{.*?(?=^- |\Z)", text, re.S)
    assert found, f"no '- {kind}:' entry"
    return found.group(0)


@pytest.mark.parametrize("name, text", [
    ("prompts.DAW_PROMPT_WHOLE", prompts.DAW_PROMPT_WHOLE),
    ("chat.SYSTEM_PROMPT", chat.SYSTEM_PROMPT),
    ("agents.INVESTIGATE_INSTRUCTIONS", agents.INVESTIGATE_INSTRUCTIONS),
])
def test_every_prompt_says_the_q_an_eq_takes(name: str, text: str):
    assert _range(EqBand, "q") in _entry(text, "eq"), name
    assert _range(DynamicEqAction, "q") in _entry(text, "dynamic_eq"), name
