"""What the prompts tell the engineer about a wet device's `tone` and an hpf's cap is what the
validator and the guardrails hold.

Two real mix turns sent a negative `tone` for a "dark" reverb (every prompt listed `tone` with no
range, beside a signed `tilt_db`) and an hpf at 150-450 Hz (no prompt the DAW chat reads named the
120 Hz cap). Each refusal cost a model round.
"""
from __future__ import annotations

import re

import pytest

from mix_engine.guardrails import LIMITS
from mix_engine.models import ReturnConvolutionAction, ReturnDelayAction, ReturnReverbAction
from orchestrator import agents, chat, prompts

PROMPTS = {
    "chat": chat.SYSTEM_PROMPT,
    "daw": chat.DAW_SYSTEM_PROMPT,
    "daw_whole": prompts.DAW_PROMPT_WHOLE,
    "planner": agents.INVESTIGATE_INSTRUCTIONS,
}

MODELS = {
    "return_reverb": ReturnReverbAction,
    "return_convolution": ReturnConvolutionAction,
    "return_delay": ReturnDelayAction,
}


def _bounds(model) -> tuple[float, float]:
    meta = model.model_fields["tone"].metadata
    lo = next(m.ge for m in meta if hasattr(m, "ge"))
    hi = next(m.le for m in meta if hasattr(m, "le"))
    return float(lo), float(hi)


def _entry(text: str, type_: str) -> str:
    """The grammar entry `{type:"<type_>", ... }` (the braces' contents)."""
    m = re.search(r'\{type:"' + type_ + r'",(.*?)\}', text, re.S)
    assert m, f"no grammar entry for {type_}"
    return m.group(1)


@pytest.mark.parametrize("name", sorted(PROMPTS))
@pytest.mark.parametrize("type_", sorted(MODELS))
def test_every_wet_tone_is_given_the_models_range(name: str, type_: str) -> None:
    lo, hi = _bounds(MODELS[type_])
    entry = _entry(PROMPTS[name], type_)
    m = re.search(r"tone:\s*(-?[\d.]+)\.\.(-?[\d.]+)", entry)
    assert m, f"{name}: {type_}'s tone has no range: {entry!r}"
    assert (float(m.group(1)), float(m.group(2))) == (lo, hi)


@pytest.mark.parametrize("name", sorted(PROMPTS))
def test_a_negative_tone_is_refused_so_the_prompt_must_say_dark_is_low(name: str) -> None:
    with pytest.raises(Exception):
        ReturnReverbAction(return_id="rv", tone=-0.3)
    assert "0 dark" in _entry(PROMPTS[name], "return_reverb")


@pytest.mark.parametrize("name", sorted(PROMPTS))
def test_the_hpf_cap_is_named_with_the_guardrails_numbers(name: str) -> None:
    text = PROMPTS[name]
    m = re.search(r"hpf's freq_hz is capped at (\d+) Hz \((\d+) Hz on a\s+stem tagged bass\)", text)
    assert m, f"{name}: the hpf cap is not stated"
    assert float(m.group(1)) == LIMITS["hpf_max_hz_default"]
    assert float(m.group(2)) == LIMITS["hpf_max_hz_bass_tagged"]
