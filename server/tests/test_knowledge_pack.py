"""The shared knowledge block (orchestrator/knowledge.py).

The point of these tests is the anti-drift property: the block is DERIVED from
the engine's own constants, so a band edge or loudness destination that changes
in the engine cannot leave the agent believing the old one. Everything else is a
cheap guard on the music theory (which is arithmetic, so it can be asserted) and
on the block actually reaching every prompt.
"""
import pytest

from orchestrator.knowledge import (KNOWLEDGE_BLOCK, SYMPTOMS, knowledge_block,
                                    note_hz, note_name)
from orchestrator.engine_models import BAND_EDGES_HZ, BAND_LABELS


# ---------- anti-drift: derived from the engine, not copied ----------

def test_every_engine_band_appears_with_its_real_edges():
    block = knowledge_block()
    for label, lo, hi in zip(BAND_LABELS, BAND_EDGES_HZ[:-1], BAND_EDGES_HZ[1:]):
        assert label in block, f"band {label} missing from the knowledge block"
        # the edge is rendered either in Hz or kHz form, but it is rendered
        span = f"{lo}-{hi} Hz" if hi < 1000 else f"{lo / 1000:g}-{hi / 1000:g} kHz"
        assert span in block, f"band {label} does not carry its real edges ({span})"


def test_band_table_tracks_a_changed_edge(monkeypatch):
    # If the engine moves a band edge, the rendered text must move with it.
    import orchestrator.knowledge_content as knowledge

    monkeypatch.setattr(knowledge, "BAND_EDGES_HZ",
                        [20, 70, 150, 400, 1000, 2000, 4000, 8000, 20000])
    assert "20-70 Hz" in knowledge.knowledge_block()


def test_loudness_destinations_come_from_the_engine_table():
    from orchestrator.targets import LOUDNESS_GOALS_LUFS

    block = knowledge_block()
    for goal in set(LOUDNESS_GOALS_LUFS.values()):
        assert f"{goal:g} LUFS" in block


# ---------- music theory is arithmetic; assert it ----------

@pytest.mark.parametrize("midi,name,hz", [
    (69, "A4", 440.0),      # concert pitch, the definition
    (28, "E1", 41.20),      # low E on a bass/guitar
    (60, "C4", 261.63),     # middle C
    (24, "C1", 32.70),
])
def test_note_frequencies(midi, name, hz):
    assert note_name(midi) == name
    assert note_hz(midi) == pytest.approx(hz, abs=0.01)


def test_octave_doubles_frequency():
    assert note_hz(48) == pytest.approx(note_hz(36) * 2.0)


def test_root_frequencies_are_rendered_for_the_low_octaves():
    block = knowledge_block()
    # the two octaves where key and EQ actually interact
    assert "E1 41.2" in block and "A1 55.0" in block


# ---------- the vocabulary itself ----------

def test_symptom_table_is_well_formed():
    assert len(SYMPTOMS) >= 35, "the producer vocabulary is the point — keep it wide"
    for entry in SYMPTOMS:
        term, cause, tool = entry
        assert term and cause and tool, f"incomplete symptom row: {entry}"


@pytest.mark.parametrize("term", [
    "boxy", "muddy", "harsh", "sibilant", "buried", "squashed",
    "no crack", "collapses in mono", "brittle", "honky",
])
def test_common_producer_words_are_covered(term):
    assert any(term in row[0] for row in SYMPTOMS), f"no entry for {term!r}"


def test_block_is_cached_as_a_stable_string():
    # Every prompt embeds the same object so the prompt cache actually hits.
    assert KNOWLEDGE_BLOCK == knowledge_block()
    assert len(KNOWLEDGE_BLOCK) > 5000


# ---------- it must actually reach the prompts ----------

def test_knowledge_reaches_every_agent_prompt(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    from orchestrator.agents import ClaudeAgent

    agent = ClaudeAgent(model="claude-sonnet-5-5")
    for system in (agent._plan_system, agent._investigate_system):
        assert any(block["text"] == KNOWLEDGE_BLOCK for block in system), \
            "an agent system prompt is missing the knowledge block"
        # cheapest cache prefix = the most stable text first
        assert system[0]["text"] == KNOWLEDGE_BLOCK


def test_chat_prompt_carries_knowledge():
    # chat.py builds its system list inline per turn; assert the import wiring
    # rather than reaching into a live turn.
    import orchestrator.chat as chat

    # chat reads the block through the LOADER at call time (2026-08-08 — the
    # book is served, not shipped); same text, different plumbing.
    assert chat._knowledge.get_block() == KNOWLEDGE_BLOCK


# ---------- shipped verticals must be NAMED, not just wired (2026-08-05 audit) ----------

def test_the_daw_prompt_names_the_capabilities_the_agent_under_reaches_for():
    """A command the prompts never mention is a command the agent never uses — the
    repo has hit this before (agent-tool-knowledge-in-prompts). These verticals were
    all built and documented as agent-operable while appearing ZERO times in either
    prompt: VCA groups in particular are called agent-operable in docs/10-daw.md.

    The catalog now carries every command's grammar, so this is not about
    discoverability — it is about giving the agent a reason to reach for a
    capability it would otherwise route around."""
    from orchestrator.chat import DAW_SYSTEM_PROMPT

    for key in ("vca.", "chord.set", "key.set", "sampler.set", "sampler.slice",
                "instrument.load_sample", "midi.quantize", "midi.arpeggiate",
                "track.set_arp", "time.insert", "clip.crop", "clip.consolidate",
                "track.bounce", "track.set_input"):
        assert key in DAW_SYSTEM_PROMPT, f"{key} is built but the agent is never told"


def test_each_named_capability_carries_a_trigger_not_just_a_name():
    """Naming a command without saying WHEN to use it doesn't move behaviour;
    prescriptive 'use it when…' framing is what gives lift. Checked per bullet
    rather than against fixed phrases, so rewording the prompt doesn't break it."""
    from orchestrator.chat import DAW_SYSTEM_PROMPT

    import re
    # The capability notes, in either form of the DAW prompt (`prompts.DAW_PROMPT_WHOLE` heads
    # them "beyond the single move"): from their heading to the next top-level section.
    block = re.split(r"Capabilities (?:you will under-reach for|beyond the single move)",
                     DAW_SYSTEM_PROMPT)[1]
    block = re.split(r"\n- \*\*|\nAsking and planning:", block)[0]
    bullets = [b for b in block.split("  * ")[1:] if b.strip()]
    assert len(bullets) >= 6, f"expected one bullet per capability family, got {len(bullets)}"

    triggers = ("use it when", "instead of", "rather than", "before you",
                "when the ask", "not the whole")
    for b in bullets:
        low = b.lower()
        assert any(t in low for t in triggers), \
            f"capability named with no trigger for when to reach for it: {b[:60]!r}"


def test_every_named_command_actually_exists_in_the_catalog():
    """A prompt naming a command that isn't real teaches the agent to invent keys —
    the exact failure the catalog fix was for. Cross-check against DawCommands.h.

    Only dotted names whose PREFIX is a real command category are checked, so
    context paths like `daw_edit_state.selection` aren't mistaken for commands."""
    import re
    from pathlib import Path
    from orchestrator.chat import DAW_SYSTEM_PROMPT

    header = Path(__file__).resolve().parents[2] / "daw" / "Source" / "DawCommands.h"
    real = set(re.findall(r'\{\s*\w+,\s*"([a-z_]+\.[a-z_]+)"', header.read_text()))
    assert real, "could not parse the command table"
    categories = {k.split(".")[0] for k in real}

    named = set(re.findall(r'`([a-z_]+\.[a-z_]+)`', DAW_SYSTEM_PROMPT))
    checkable = {k for k in named if k.split(".")[0] in categories}
    assert checkable, "no commands named in the DAW prompt at all"
    unreal = checkable - real
    assert not unreal, f"prompt names commands that do not exist: {sorted(unreal)}"


# ---------- the device catalog's SHAPE (2026-09-06 review) ----------
#
# The catalog is a long backslash-continued string, and a stray trailing `\` on the last line of
# one bullet silently swallows the NEXT bullet into it. That is what happened to `- overdrive:`
# when the chorus landed: it stopped being a list item in the chat agent's device list, which is
# the surface the agent reads to decide what a device is for. Nothing caught it because the text
# was still present and every keyword still matched.

def _catalog_bullets(blob: str) -> list[str]:
    """The device bullets, as the model sees them: lines beginning `- <type>: {type:"…"`."""
    import re
    return re.findall(r'^- ([a-z_]+): \{type:"([a-z_]+)"', blob, re.M)


@pytest.mark.parametrize("module_name,const", [("orchestrator.chat", "SYSTEM_PROMPT"),
                                               ("orchestrator.agents", "INVESTIGATE_INSTRUCTIONS")])
def test_every_device_bullet_starts_its_own_line(module_name, const):
    import importlib
    blob = getattr(importlib.import_module(module_name), const)
    bullets = _catalog_bullets(blob)
    assert len(bullets) >= 10, f"{const}: only {len(bullets)} device bullets parsed — the catalog's shape changed"
    for label, typ in bullets:
        assert label == typ, f"{const}: bullet '- {label}:' declares type \"{typ}\""
    # …and every type the engine offers on a stem must BE one of those bullets. A swallowed
    # bullet disappears from this list while its words stay in the blob.
    for typ in ("chorus", "overdrive", "saturation", "comp", "eq"):
        assert typ in [t for _, t in bullets], (
            f"{const}: '{typ}' is not a bullet of its own — it was swallowed into the previous one "
            "(look for a stray trailing backslash on the line above it)"
        )


# ---------- the guardrails table is the agent's bounds, written down (2026-09-06 review) ----------
#
# `docs/03-mcp-tools.md` carries the one human-readable enumeration of what the agent may do, and
# its own closing line promises "tests can assert against them". Nothing did: the table was prose
# with no keys in it, so five limits — chorus_mix_max, overdrive_drive_max_db, ott_depth_max,
# clip_threshold_min_db, gate_threshold_max_db — were simply absent and could not be noticed.

def _guardrail_doc_keys() -> set[str]:
    import pathlib, re
    from orchestrator import knowledge
    root = pathlib.Path(knowledge.__file__).resolve().parents[1]
    doc = (root / "docs" / "03-mcp-tools.md").read_text()
    i = doc.index("| Rule | `guardrails.py` key | Limit |")
    j = doc.index("Limits live in one place", i)
    return set(re.findall(r"\|\s*`([a-z0-9_]+)`\s*\|", doc[i:j]))


def test_every_guardrail_has_a_row():
    from mix_engine.guardrails import LIMITS
    doc_keys = _guardrail_doc_keys()
    missing = sorted(set(LIMITS) - doc_keys)
    assert not missing, (
        f"{len(missing)} guardrail(s) the agent is bound by have no row in docs/03-mcp-tools.md: "
        f"{missing}"
    )


def test_the_table_invents_no_guardrail():
    """The other direction: a row for a limit that no longer exists is just as misleading."""
    from mix_engine.guardrails import LIMITS
    extra = sorted(_guardrail_doc_keys() - set(LIMITS))
    assert not extra, f"docs/03-mcp-tools.md documents limits that are not in LIMITS: {extra}"
