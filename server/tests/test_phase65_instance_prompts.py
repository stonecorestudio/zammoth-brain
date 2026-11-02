"""Phase 65 step 4 (2026-10-24): what the agent READS about a stack of devices says what the code
does. A whole-file proposal of a stackable type (`guardrails.INSTANCE_KINDS`) names which device it
changes with `instance` (N, as `comp#N` counts them) or adds one (`"new"`); the old sentences
told the model a chain carries one of each and that a channel with two could only be left alone.
"""
from __future__ import annotations

import pytest

from mix_engine import models
from mix_engine.guardrails import INSTANCE_KINDS
from orchestrator import agents, chat, prompts


def _prompts() -> dict[str, str]:
    return {"chat (DAW)": chat.daw_prompt(), "whole": prompts.DAW_PROMPT_WHOLE,
            "planner": agents.SYSTEM_PROMPT}


#: What the old grammar said and the code no longer does.
STALE = ("change one of them with device.set comp#i instead",
         "is refused (rule instance_which):\n  leave them",
         "one whole-file comp per stem",
         "One whole-file comp per stem",
         "one whole-file comp (and one mb_comp) per stem",
         "never try to stack",
         "not on master; one per stem (a new one replaces it)",
         "One per stem, a new one replaces it")


@pytest.mark.parametrize("name", ["chat (DAW)", "whole", "planner"])
def test_every_prompt_teaches_instance(name):
    text = _prompts()[name]
    assert 'instance: "new"' in text, name
    assert "instance: N" in text, name
    assert "instance_which" in text, name


@pytest.mark.parametrize("name", ["chat (DAW)", "whole", "planner"])
def test_no_prompt_says_one_per_chain_of_a_stackable_type(name):
    text = _prompts()[name]
    flat = " ".join(text.split())
    for stale in STALE:
        assert " ".join(stale.split()) not in flat, (name, stale)


def test_the_instance_field_is_described_on_every_stackable_action():
    by_type = {}
    for member in models.Action.__args__ if hasattr(models, "Action") else ():
        by_type[member.model_fields["type"].default] = member
    if not by_type:
        for name in dir(models):
            cls = getattr(models, name)
            if isinstance(cls, type) and issubclass(cls, models.BaseModel) \
                    and "type" in getattr(cls, "model_fields", {}):
                by_type.setdefault(cls.model_fields["type"].default, cls)
    for kind in INSTANCE_KINDS:
        cls = by_type[kind]
        field = cls.model_fields.get("instance")
        assert field is not None, kind
        assert field.description and "new" in field.description, kind


def test_the_utility_no_longer_says_it_is_one_per_chain():
    assert "One\n    per chain" not in (models.UtilityAction.__doc__ or "")
    assert "single Utility step" not in (models.UtilityAction.__doc__ or "")


# --- the eval's fake app takes a second compressor and refuses a second Width, as the app does --

import re
from pathlib import Path

from mix_engine.session import CHAIN_MAX_ENTRIES
from orchestrator import command_door, operate_eval

REPO = Path(__file__).resolve().parents[2]


def test_the_transcribed_singletons_are_the_apps():
    src = (REPO / "native/Source/DeviceSpecs.h").read_text()
    listed = re.search(r"kSingletons\[\]\s*=\s*\{([^}]*)\}", src).group(1)
    assert tuple(re.findall(r'"([a-z_]+)"', listed)) == command_door.SINGLETON_DEVICE_TYPES
    assert command_door.CHAIN_MAX_DEVICES == CHAIN_MAX_ENTRIES
    for kind, label in command_door.DEVICE_LABELS.items():
        assert f'if (t == "{kind}") return "{label}";' in src


def test_the_transcribed_words_are_the_apps():
    src = (REPO / "daw/Source/DeviceChainView.h").read_text()
    body = src[src.index("static juce::String capacityRefusal"):]
    body = body[:body.index("\n    }\n")]
    assert '" already has an " + deviceLabel (type)' in body
    assert r'" \xe2\x80\x94 one per chain"' in body
    assert r'" \xe2\x80\x94 a chain holds "' in body and '"; remove one first"' in body


def _add(track: str, kind: str) -> dict:
    case = operate_eval.OpCase("add a device", ("device.add",))
    rec: list = []
    return operate_eval.fake_daw(case, "device.add", {"track": track, "type": kind},
                                 confirm_granted=False,
                                 record=lambda k, step, st: rec.append(k) or st)


def test_the_fake_app_takes_a_second_compressor():
    assert _add("lead_vox", "comp").get("ok") is True     # lead_vox already holds one


def test_the_fake_app_refuses_a_second_width():
    assert command_door.device_add_refusal("device.add", {"track": "x", "type": "width"},
                                           ["eq", "width"]) == \
        "x already has an Imager — one per chain"
    assert command_door.device_add_refusal("device.add", {"track": "x", "type": "comp"},
                                           ["comp", "comp"]) is None
    assert command_door.device_add_refusal("device.add", {"track": "x", "type": "eq"},
                                           ["eq"] * 128).startswith("x already holds 128 devices")
