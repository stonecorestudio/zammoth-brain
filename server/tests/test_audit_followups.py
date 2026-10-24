"""The four items the 2026-08-05 agent audit left open, reported closed once
before they were (see the audit memo). Each test pins one.

The fifth — commands lying with ok() — is C++ and is pinned by the build itself:
`startAction` is `[[nodiscard]]`, so a handler that dispatches and reports success
without checking will not compile silently.
"""
import inspect

from orchestrator import chat, daw_bridge


def test_xhigh_is_a_reachable_reasoning_level():
    """The API exposes an effort rung between high and max; the enum simply
    omitted it, so no picker could reach it."""
    assert "xhigh" in chat.REASONING_LEVELS
    assert chat.REASONING_LEVELS.index("high") < chat.REASONING_LEVELS.index("xhigh")
    assert chat.REASONING_LEVELS.index("xhigh") < chat.REASONING_LEVELS.index("max")
    chat.validate_chat_settings(chat.MODEL, "xhigh")   # must not raise


def test_the_daw_effort_picker_offers_only_levels_the_brain_accepts():
    """The DAW keeps its own copy of the ladder (the picker in EngineerPanel.h, whose rows carry
    the descriptions), so a desync is invisible until someone picks a level the brain then
    refuses.

    Since 2026-09-22 (James: "add all models and effort levels") the DAW offers the WHOLE API
    ladder — it had offered a low/medium/high subset (the three Phase 51 measured) since
    2026-08-22. So the invariant is equality with the brain's ladder, in order, and Prefs.h's
    clamp list comes from the shared model table (AgentModels.h), itself pinned to
    llm_models.py by test_agent_model_parity.py."""
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    panel = (root / "native" / "Source" / "EngineerPanel.h").read_text()
    block = panel.split("static const EffortLevel levels[] = {", 1)[1].split("};", 1)[0]
    picker = re.findall(r'\{\s*"([a-z]+)"', block)
    assert picker, "the DAW effort picker's ladder could not be read"
    assert picker == chat.REASONING_LEVELS, f"the DAW ladder {picker} != the brain's"
    for level in picker:
        chat.validate_chat_settings(chat.MODEL, level)   # must not raise

    prefs = (root / "daw" / "Source" / "Prefs.h").read_text()
    body = prefs.split("inline juce::StringArray knownEfforts()", 1)[1].split("}", 1)[0]
    assert "agentmodels::allEfforts()" in body, "knownEfforts must come from AgentModels.h"


def test_import_sample_is_wired_to_a_real_implementation():
    """It was advertised in the chat prompt and bound to `unsupported`, so the
    agent was offered a tool that always refused."""
    from orchestrator import turn_host       # the turn's tools live there since Phase 55
    source = inspect.getsource(turn_host.LiveTurn.tools)
    assert "import_sample=self.do_import_sample" in source
    assert "import_sample=self.unsupported" not in source


def test_import_sample_does_not_double_register():
    """`track.add` already places the clip AND registers the stem with the brain.
    Calling import_stem here as well would register the same file twice.

    The routing moved into `_import_sample_step` (2026-08-16), so that both the
    chat tool and the Ask-mode plan replay run ONE implementation — the tool
    body now just gates on the recorder and delegates.
    """
    body = inspect.getsource(daw_bridge._import_sample_step)
    assert '"track.add"' in body
    assert "import_stem" in body            # ...but only for the no-DAW fallback
    assert "daw_invoke is None" in body

    from orchestrator import turn_host
    tool = inspect.getsource(turn_host.LiveTurn.do_import_sample)
    assert "_import_sample_step" in tool     # delegates, never re-implements
    assert "recorder is not None" in tool    # and honours the Ask-mode gate


def test_the_readout_paths_accept_the_render_cache():
    """Each of these rendered the whole master unconditionally. The brain already
    keeps a per-edit render cache; they just weren't given it."""
    for fn in (daw_bridge.chat_reply, daw_bridge.reference_match,
               daw_bridge.transpose_to_target_key):
        assert "cache" in inspect.signature(fn).parameters, fn.__name__


def test_no_readout_call_forgets_the_cache():
    """A bare `session_readout(project_dir)` is the bug this closes — it forces a
    full master render (and the O(n^2) masking sweep behind it)."""
    from pathlib import Path
    source = (Path(daw_bridge.__file__)).read_text()
    assert "session_readout(project_dir)" not in source, \
        "an uncached session_readout call came back"
