"""Awareness and Auto Mix — the brain half of the 2026-09-10 agent audit's last P1s.

- P1-18: one hosted synth's parameter list and automation addresses pushed every track out
  of the agent's context (0/14 tracks with a 2000-param synth).
- P1-20: the prompt said no command bypasses a native device; `device.enable` does.
- P1-21: the composer's effort reached the brain only through a notify that a fresh or
  respawned brain never received, so every turn ran HIGH under a composer showing LOW.
- P1-23: Auto Mix bundles the guardrails refused vanished ("vocal-forward" never lifted
  the vocal, and the feed said "Auto Mix applied").
- P1-24: the refinement read a preset name as a brief ("wide pop" → pop tonal balance), and
  master match-EQs stacked past the boost limit, each judged alone.

(P1-22 — a chat-started run's feed and cost — is in test_daw_chat.py, next to the
`scripted` fixture it needs.)
"""
from __future__ import annotations

import asyncio
import inspect
import json

import pytest

from orchestrator import automix, daw_bridge
from orchestrator import chat as chatmod
from orchestrator.edit_state import compact_edit_state


# ------------------------------------------------------------------ P1-18

def _snapshot(plugin_params: int) -> dict:
    tracks = [{"name": f"Audio {i}", "gain_db": 0.0, "clips": [{"start_s": 0, "length_s": 8}]}
              for i in range(14)]
    tracks[0]["plugins"] = [{
        "name": "BigSynth", "param_count": plugin_params,
        "params": [{"index": i, "name": f"Param {i}", "value": 0.5} for i in range(24)],
        "automation_targets": [f"plugin#0.p{i}" for i in range(plugin_params)],
    }]
    return {"tracks": tracks, "tempo_bpm": 120}


def test_a_big_hosted_synth_does_not_push_the_tracks_out():
    text = compact_edit_state(_snapshot(2000), 24000)
    snap = json.loads(text)
    assert len(snap.get("tracks") or []) == 14, "every track stays in the agent's view"
    assert "plugin_params" in snap.get("omitted", [])
    assert "tracks" not in snap.get("omitted", [])


def test_a_small_plugin_keeps_its_params():
    snap = json.loads(compact_edit_state(_snapshot(24), 24000))
    assert snap["tracks"][0]["plugins"][0]["params"], "nothing is dropped when it fits"


# ------------------------------------------------------------------ P1-20

def test_the_prompt_names_device_enable_as_the_native_bypass():
    prompt = chatmod.DAW_SYSTEM_PROMPT
    assert "device.enable {track, device, on:false}" in prompt
    assert "no\ncommand that removes or bypasses a NATIVE" not in prompt
    assert "command that removes or bypasses a NATIVE chain device" not in prompt


# ------------------------------------------------------------------ P1-21

def test_each_chat_request_carries_the_composers_effort(monkeypatch):
    monkeypatch.delenv("MIX_AGENT_EFFORT", raising=False)
    session = chatmod.ChatSession(client=object())
    model = session.settings["model"]
    daw_bridge._apply_turn_llm(session, {"model": model, "reasoning": "low"})
    assert session.settings == {"model": model, "reasoning": "low"}
    daw_bridge._apply_turn_llm(session, {"model": model, "reasoning": "not-a-level"})
    assert session.settings["reasoning"] == "low", "an invalid value is ignored, not raised"
    assert "llm" in inspect.signature(daw_bridge.chat_turn).parameters


# ------------------------------------------------------------------ P1-23

class _FakeEngine:
    """propose_actions refuses any bundle whose rationale says so."""

    def __init__(self):
        self.calls: list[str] = []
        self._n = 0

    async def call(self, name: str, args: dict | None = None):
        self.calls.append(name)
        if name == "propose_actions":
            bundle = (args or {})["req"]
            if "REFUSE" in bundle["rationale"]:
                return {"accepted": False, "violations": [
                    {"rule": "gain_per_stem_per_pass_db",
                     "message": "cumulative gain on 'vocals' this pass would be +7.5 dB"}]}
            self._n += 1
            return {"accepted": True, "proposal_id": f"p{self._n}"}
        if name == "apply_proposal":
            return {"ok": True}
        return {}


def _bundle(rationale: str, **extra) -> dict:
    return {"rationale": rationale, "actions": [], "predicted_impact": [], **extra}


def test_a_refused_auto_mix_bundle_is_reported_not_swallowed():
    engine, events, rejected = _FakeEngine(), [], []

    async def emit(e):
        events.append(e)

    asyncio.run(automix._apply_bundles(
        engine, emit, [_bundle("phrases"), _bundle("preset vocal-forward REFUSE")],
        rejected=rejected))
    assert len(rejected) == 1 and "+7.5 dB" in rejected[0]
    assert any(e.get("type") == "automix_stage" and e.get("stage") == "refused"
               for e in events), "the refusal reaches the DAW's stage feed"


def test_the_preset_bundle_gets_its_own_pass():
    engine = _FakeEngine()

    async def emit(_e):
        return None

    asyncio.run(automix._apply_bundles(
        engine, emit, [_bundle("phrases"), _bundle("preset", _own_pass=True)]))
    proposes = [i for i, c in enumerate(engine.calls) if c == "propose_actions"]
    assert "advance_pass" in engine.calls[proposes[0]:proposes[1]], \
        "a fresh per-pass wallet before the preset, so the phrase rides can't use it up"


def test_build_bundles_marks_the_preset_for_its_own_pass():
    src = inspect.getsource(automix.build_bundles)
    assert '"_own_pass"' in src


# ------------------------------------------------------------------ P1-24

@pytest.mark.parametrize("brief, expected", [
    ("wide pop", "master to -12 LUFS"),
    ("club loud", "master to -8 LUFS"),
    ("vocal-forward", ""),                               # no loudness goal → bootstrap only
    ("master for streaming", "master for streaming"),    # no preset → unchanged
])
def test_the_refinement_aims_at_loudness_not_the_preset_name(brief, expected):
    assert automix.refinement_brief(brief) == expected


def test_a_brief_with_both_keeps_its_own_loudness_words():
    out = automix.refinement_brief("master for streaming, warm master")
    assert "streaming" in out and "warm" not in out


def test_the_match_eq_why_line_quotes_the_gain_that_applies():
    """The ledger read "high_mid +16.6 dB" over a +6 dB move, and named a reference that
    did not exist. (Whether the +6 dB limit should bind the master's running TOTAL is a
    design call: binding it dropped the rule benchmark's pop-master closure 0.95 -> 0.42.)"""
    from orchestrator.agents import PassContext, RuleAgent, TargetState

    target = TargetState(metric="band_energy.high_mid", unit="dB", current=-30.0,
                         distance=16.0, goal=-14.0)
    ctx = PassContext(pass_number=1, metrics={}, targets=[target], stem_chains={})
    proposal = RuleAgent()._band_match_proposal(ctx)
    boost = proposal["actions"][0]["band_gains_db"]["high_mid"]
    assert 0 < boost <= ctx.limits["eq_boost_max_db"]
    assert f"high_mid {boost:+.1f} dB (gap +16.0)" in proposal["rationale"]
    assert "reference" not in proposal["rationale"]
