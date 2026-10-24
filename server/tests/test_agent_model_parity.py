"""The DAW's model table IS the brain's (2026-09-22; the 5.5 family alone since 2026-10-08).

The model list once lived in six places and drifted: the VST's "Titan" was Opus 4.8 while the
DAW's was Opus 5, and Preferences hard-coded four labels over a one-entry list so a pick silently
reverted (memory `model-beast-names`). Since the picker came back there are exactly two copies — `orchestrator/llm_models.py` (what the brain validates and
prices) and `native/Source/AgentModels.h` (what the DAW draws and persists) — and this test
re-parses the C++ rows and requires them to BE the Python ones: ids, labels, families, effort
sets, order, and both defaults. Free, and in CI. The same pattern as test_device_clamp_parity.py.

It also pins the three DAW consumers to that header, so no seventh copy can grow back: Prefs.h
validates against it, and neither Preferences nor the composer names a model id or an effort
list of its own.
"""
from __future__ import annotations

import re
from pathlib import Path

from orchestrator import llm_models

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "native" / "Source" / "AgentModels.h"


def _rows() -> list[tuple[str, str, str, tuple[str, ...]]]:
    text = HEADER.read_text()
    block = text.split("static const Model rows[] = {", 1)[1].split("};", 1)[0]
    rows = re.findall(r'\{\s*"([^"]+)",\s*"([^"]+)",\s*"([^"]+)",\s*"([^"]*)"\s*\}', block)
    assert rows, "AgentModels.h rows could not be read"
    return [(i, lab, fam, tuple(e.split())) for i, lab, fam, e in rows]


def test_the_daw_table_is_the_brain_table():
    assert _rows() == [(m.id, m.label, m.family, m.efforts) for m in llm_models.MODELS]


def test_the_picker_is_the_5_5_family_and_nothing_else():
    """James, 2026-10-08: "remove all models that are not in the 5.5 family ... keep opus 5.5,
    sonnet 5.5 and the new haiku 5.5 ... no fable yet"."""
    assert [m.id for m in llm_models.MODELS] == [
        "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
    assert llm_models.DEFAULT_MODEL == "claude-opus-5-5"
    for gone in ("claude-fable-5-1", "claude-fable-5", "claude-opus-5", "claude-opus-4-8",
                 "claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-5", "claude-sonnet-4-6",
                 "claude-haiku-4-5"):
        assert not llm_models.is_known(gone), gone
        # ...and each is still PRICED: an old baseline or an env override may name it, and an
        # id nobody priced is billed at the dearest rate
        assert llm_models._rates(gone) != llm_models._UNKNOWN_RATES or gone.startswith("claude-fable")
    # every row: the whole ladder, a 1M window, thinking bound to the conversation
    for m in llm_models.MODELS:
        assert m.efforts == llm_models.EFFORT_LEVELS and m.preserved_thinking
        assert (m.context_tokens, m.max_output_tokens) == (1_000_000, 128_000)


def test_a_newer_model_is_not_a_snapshot_of_an_older_one():
    """`claude-opus-5-5` begins with `claude-opus-5-`. Only an 8-digit date makes a snapshot -
    on the picker's rows and on the price-only ones, which now hold Opus 5 and Sonnet 5."""
    assert llm_models._rates("claude-opus-5-5") == (4.0, 20.0, 0.05)
    assert llm_models._rates("claude-sonnet-5-5") == (2.0, 10.0, 0.05)
    assert llm_models._rates("claude-opus-5") == (5.0, 25.0, 0.1)
    assert llm_models._rates("claude-opus-4-5-20251101") == (5.0, 25.0, 0.1)
    assert llm_models.spec("claude-opus-5-5-20260922").id == "claude-opus-5-5"
    for stranger in ("claude-opus-5-6", "claude-fable-5-5", "claude-sonnet-5-9"):
        assert llm_models.spec(stranger) is None
        assert llm_models._rates(stranger) == llm_models._UNKNOWN_RATES


def test_haiku_5_5_is_priced_by_the_length_of_each_prompt():
    """$0.10 / $0.50 up to 100,000 prompt tokens, five times every rate over it (Anthropic's
    pricing page, read 2026-10-08). The prompt is the fresh input plus what was read from and
    written to the cache - and it is ONE request's."""
    cost = llm_models.cost_usd
    haiku = "claude-haiku-5-5"
    assert cost(haiku, 1_000_000 // 10, 0) == 0.01                       # exactly 100,000: the low price
    assert round(cost(haiku, 100_001, 0), 6) == round(100_001 * 0.50 / 1e6, 6)
    assert cost(haiku, 0, 1_000_000) == 0.50
    # cache reads and writes count toward the length, and are raised with the rest
    low = cost(haiku, 10_000, 1_000, cache_read_tokens=80_000, cache_write_tokens=10_000)
    high = cost(haiku, 10_000, 1_000, cache_read_tokens=80_000, cache_write_tokens=10_001)
    assert round(low, 6) == round((10_000 * 0.10 + 80_000 * 0.01 + 10_000 * 0.125 + 1_000 * 0.50) / 1e6, 6)
    assert 4.9 < high / low < 5.1
    # ten rounds of 30k are ten short prompts, not one long one
    assert round(sum(cost(haiku, 30_000, 500) for _ in range(10)), 6) == round(cost(haiku, 30_000, 500) * 10, 6)
    assert cost(haiku, 300_000, 5_000) > 4.9 * sum(cost(haiku, 30_000, 500) for _ in range(10))
    # no other row has a second price
    for m in llm_models.MODELS:
        if m.id != haiku:
            assert cost(m.id, 900_000, 0) == 9 * cost(m.id, 100_000, 0)


def test_the_defaults_and_the_ladder_agree():
    text = HEADER.read_text()
    assert f'kDefaultModel  = "{llm_models.DEFAULT_MODEL}"' in text
    assert f'kDefaultEffort = "{llm_models.DEFAULT_EFFORT}"' in text
    ladder = re.search(r"allEfforts\(\)\s*\{\s*return\s*\{([^}]*)\}", text)
    assert ladder and tuple(re.findall(r'"([a-z]+)"', ladder.group(1))) == llm_models.EFFORT_LEVELS


def test_the_fast_output_models_and_price_agree():
    """Phase 54: the Fast output row is offered for the models the brain may send
    `speed: "fast"` for, and the price the app states is the brain's multiplier."""
    import re
    text = HEADER.read_text()
    ids = re.search(r'kFastModels = "([^"]*)"', text).group(1).split()
    assert ids == [m.id for m in llm_models.MODELS if m.fast] and ids, ids
    factor = float(re.search(r"kFastPriceX = ([0-9.]+)", text).group(1))
    assert {m.fast_x for m in llm_models.MODELS if m.fast} == {factor}
    # every request that carries the effort carries the switch (the app's five send sites)
    app = (ROOT / "daw" / "Source" / "App.cpp").read_text()
    assert app.count('setProperty ("effort"') == app.count('setProperty ("fast"') == 5


def test_no_daw_file_keeps_its_own_model_or_effort_list():
    prefs = (ROOT / "daw" / "Source" / "Prefs.h").read_text()
    panel = (ROOT / "daw" / "Source" / "PrefsPanel.h").read_text()
    engineer = (ROOT / "native" / "Source" / "EngineerPanel.h").read_text()
    for name, text in (("Prefs.h", prefs), ("PrefsPanel.h", panel), ("EngineerPanel.h", engineer)):
        assert "AgentModels.h" in text, f"{name} must read the model table from AgentModels.h"
        ids = set(re.findall(r'"(claude-[a-z0-9-]+)"', text))
        assert not ids, f"{name} names model ids of its own: {sorted(ids)}"
    for name, text in (("Prefs.h", prefs), ("PrefsPanel.h", panel)):
        assert not re.search(r'\{\s*"low",\s*"medium",\s*"high"', text), \
            f"{name} keeps its own effort list"


# ------------------------------------------ `--verify` reads what the real SDK hands back

class _Models:
    """`client.models.retrieve` answering with the REAL SDK type (anthropic.types.ModelInfo):
    its `capabilities` is a pydantic model, not a dict — the first `_verify` subscripted it and
    would have crashed on its first model the one time James ran it."""

    def __init__(self, override=None, missing=()):
        self.override = override or {}
        self.missing = set(missing)
        self.models = self

    def retrieve(self, model_id):
        from anthropic.types import ModelInfo
        if model_id in self.missing:
            raise LookupError(f"model: {model_id}")
        m = llm_models.spec(model_id)
        levels = {e: {"supported": e in m.efforts} for e in llm_models.EFFORT_LEVELS}
        if "xhigh" not in m.efforts:
            levels["xhigh"] = None          # the API sends null, not {supported: false}
        caps = {"batch": {"supported": True}, "citations": {"supported": True},
                "code_execution": {"supported": True},
                "context_management": {"supported": True},
                "effort": {"supported": bool(m.efforts), **levels},
                "image_input": {"supported": True}, "pdf_input": {"supported": True},
                "structured_outputs": {"supported": True},
                "thinking": {"supported": True, "types": {
                    "adaptive": {"supported": bool(m.efforts)},
                    "enabled": {"supported": not m.efforts}}}}
        for path, value in self.override.get(model_id, {}).items():
            node = caps
            *head, leaf = path.split(".")
            for key in head:
                node = node[key]
            node[leaf] = value
        return ModelInfo.model_validate({
            "id": model_id, "type": "model", "display_name": m.label,
            "created_at": "2026-01-01T00:00:00Z", "max_input_tokens": m.context_tokens,
            "max_tokens": m.max_output_tokens, "capabilities": caps})


def test_verify_passes_a_live_catalog_that_matches_the_table():
    assert llm_models._verify(_Models()) == []


def test_verify_names_every_kind_of_drift():
    client = _Models(override={
        "claude-opus-5-5": {"effort.xhigh": None},                    # the API dropped a level
        "claude-sonnet-5-5": {"structured_outputs.supported": False},  # _parse would 400
        "claude-haiku-5-5": {"thinking.types.adaptive.supported": False}})
    problems = llm_models._verify(client)
    assert len(problems) == 3, problems
    joined = "\n".join(problems)
    assert "claude-opus-5-5: effort levels" in joined
    assert "claude-sonnet-5-5: no structured outputs" in joined
    assert "claude-haiku-5-5: adaptive thinking False" in joined
    problems = llm_models._verify(_Models(missing={"claude-haiku-5-5"}))   # key can't see it
    assert len(problems) == 1 and "claude-haiku-5-5: not retrievable" in problems[0]
