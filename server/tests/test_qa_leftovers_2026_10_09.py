"""What the user QA of 2026-10-08 still had open on 2026-10-09 that needed nobody but the
engineer's builder - the brain's half.

The DAW's half is driven by console legs (daw/Source/QaLeftoverLegs.h) and pinned by
test_qa_leftovers_2026_10_09_cpp.py.

Record: docs/audits/user-qa-2026-10-08/QA-log.md, *The leftovers*.
"""
from __future__ import annotations

from orchestrator import chat
from orchestrator import daw_bridge as B


# ---- a routing row's DETAILS (O21) --------------------------------------------------------------

def test_a_routing_command_hands_the_daw_what_was_asked_beside_the_payload():
    """The feed's row for a brain-performed routing command described the PAYLOAD - the whole
    resolved mix, every track's chain - because the payload was all the DAW was handed. The
    command's own arguments travel with it now, under `asked`."""
    sent = []

    def fake_invoke(command, args):
        sent.append((command, args))
        return {"ok": True}

    ops = {"send.set": lambda a: {"chains": {"vocals": {"entries": []}}, "send_returns": {"returns": [{"id": "verb"}]}}}
    asked = {"source": "vocals", "return": "verb", "gain_db": -12.0}
    out = B.perform_routing("send.set", dict(asked), ops, fake_invoke)
    assert out["ok"] is True
    command, payload = sent[0]
    assert command == "routing.apply"
    assert payload["asked"] == asked
    assert payload["command"] == "send.set" and "chains" in payload     # the payload is still whole


def test_asked_is_a_copy_the_op_cannot_change():
    sent = []

    def op(args):
        args["source"] = "changed by the op"
        return {"chains": {}}

    B.perform_routing("send.set", {"source": "vocals"}, {"send.set": op}, lambda c, a: sent.append((c, a)) or {"ok": True})
    assert sent[0][1]["asked"] == {"source": "vocals"}


def test_a_routing_command_with_no_arguments_says_so():
    sent = []
    B.perform_routing("return.add", None, {"return.add": lambda a: {"chains": {}}},
                      lambda c, a: sent.append((c, a)) or {"ok": True})
    assert sent[0][1]["asked"] == {}


# ---- the prompt: the key, and a pan as the header shows it (O2, O19) -----------------------------

def test_the_prompt_says_where_the_key_is_and_that_none_is_none():
    prompt = chat.DAW_SYSTEM_PROMPT
    assert "daw_edit_state.key" in prompt
    for word in ("`detected`", "`set`", "`none`"):
        assert word in prompt, word
    assert "never fill in C major" in prompt


def test_the_prompt_says_a_pan_is_quoted_as_the_header_shows_it():
    prompt = chat.DAW_SYSTEM_PROMPT
    assert "`pan_shown`" in prompt and "`shown`" in prompt
    assert "`20L`" in prompt and "pan x 50" in prompt
