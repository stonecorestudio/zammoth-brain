"""Space devices (reverb / delay / IR reverb) are FREE INSERTS, not return-only.

2026-07-20, James: "like ableton, i need to be able to freely place these audio
effects on tracks ... i dont want restrictions". Before this, `return_reverb` /
`return_delay` / `return_convolution` existed ONLY inside an FX return: they were
absent from _STEM_KINDS/_BUS_KINDS/_MASTER_KINDS (so effective_chain silently
dropped them), `_device_ctor` refused them off a return, dsp.py dispatched them
only in `_apply_return_chain`, and the DAW dock's addableTypesFor listed them for
`return:` targets alone.

The one thing that genuinely made them return-only was that they are WET-ONLY
processors — correct on a return (the dry lives on the source tracks), useless as
an insert. So the fix is a dry/wet `mix`, defaulting to 1.0 so every pre-existing
session renders bit-identically.

Covers: the render on stem/bus/master, the blend math, the return form unchanged,
the user + agent add paths, the native slot cap, and the agent's wet guardrail.
"""
import numpy as np
import pytest
import soundfile as sf
from mcp.shared.memory import create_connected_server_and_client_session

from mix_engine import dsp
from mix_engine import reference_monitor
from mix_engine.mcp_server import create_server
from mix_engine.session import (BUS_ACTION_TYPES, MASTER_ACTION_TYPES,
                                SUPPORTED_ACTION_TYPES, Session, effective_chain,
                                effective_bus_chain, effective_master_chain)

from test_phase1_acceptance import call

SR = 48000
WET_TYPES = ("return_reverb", "return_delay", "return_convolution")


def engine(project):
    return create_connected_server_and_client_session(create_server(project))


def _project(tmp_path, name: str, stems=("vox", "gtr")) -> "object":
    project = tmp_path / name
    project.mkdir()
    t = np.arange(int(1.0 * SR)) / SR
    for i, stem in enumerate(stems):
        sf.write(project / f"{stem}.wav",
                 (0.3 * np.sin(2 * np.pi * (440.0 + 110.0 * i) * t)).astype(np.float32), SR)
    return project


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio))))


# ---------------------------------------------------------------- the render


@pytest.mark.parametrize("wet_type", WET_TYPES)
def test_space_device_renders_as_a_stem_insert(tmp_path, wet_type):
    """The core of the ask: a reverb/delay/IR reverb dropped on a TRACK renders.
    Before, effective_chain dropped the entry and the stem came back untouched."""
    session = Session(_project(tmp_path, f"insert-{wet_type}"))
    dry, _ = dsp.render_stem(session, "vox")

    session.add_stem_device("vox", wet_type)
    assert any(e["type"] == wet_type for e in effective_chain(session.stems["vox"])), \
        "the entry must survive effective_chain, not be filtered out as an unknown kind"

    dsp.RENDER_CACHE.clear()
    wet, _ = dsp.render_stem(session, "vox")
    assert wet.shape == dry.shape                      # an insert never changes length
    assert not np.allclose(wet, dry), "the device rendered nothing — it was dropped again"


def test_space_device_renders_on_bus_and_master(tmp_path):
    """Every target, not just stems (the palette is unified — device freedom)."""
    project = _project(tmp_path, "insert-targets")
    session = Session(project)
    session.add_stem_device("vox", "return_delay")     # prove the stem path too
    base, _ = dsp.render_master(session)

    session = Session(project)
    session.add_master_device("return_reverb")
    assert any(e["type"] == "return_reverb" for e in effective_master_chain(session._state))
    dsp.RENDER_CACHE.clear()
    mastered, _ = dsp.render_master(session)
    assert not np.allclose(mastered, base), "a master-bus reverb rendered nothing"


def test_space_device_renders_on_a_group_bus(tmp_path):
    project = _project(tmp_path, "insert-bus")
    session = Session(project)
    # route both stems into a bus, then insert a reverb ON the bus
    session._state["buses"]["glue"] = {"gain_db": 0.0, "chain": [], "pending": [],
                                       "sends": {}}
    for stem in ("vox", "gtr"):
        session.stems[stem]["bus"] = "glue"
    session._persist()
    dsp.RENDER_CACHE.clear()
    base, _ = dsp.render_bus(session, "glue")

    session.add_stem_device("glue", "return_reverb")
    assert any(e["type"] == "return_reverb" for e in effective_bus_chain(session.buses["glue"]))
    dsp.RENDER_CACHE.clear()
    wet, _ = dsp.render_bus(session, "glue")
    assert not np.allclose(wet, base), "a bus reverb rendered nothing"


def test_stem_space_device_keeps_its_exact_list_position(tmp_path):
    """A wet insert before saturation must not be regrouped after it by dtype."""
    session = Session(_project(tmp_path, "wet-list-order"))
    session.add_stem_device("vox", "return_reverb")
    session.add_stem_device("vox", "saturation")
    reverb = next(e for e in session.stems["vox"]["chain"]
                  if e["type"] == "return_reverb")
    saturation = next(e for e in session.stems["vox"]["chain"]
                      if e["type"] == "saturation")
    reverb.update({"mix": 0.5, "decay_s": 2.5, "pre_delay_ms": 0.0})
    saturation.update({"drive_db": 12.0, "output_db": 0.0, "mix": 1.0})
    session.stems["vox"]["chain"] = [reverb, saturation]
    dsp.RENDER_CACHE.clear()
    wet_then_sat, _ = dsp.render_stem(session, "vox")

    session.stems["vox"]["chain"] = [saturation, reverb]
    dsp.RENDER_CACHE.clear()
    sat_then_wet, _ = dsp.render_stem(session, "vox")

    assert np.max(np.abs(wet_then_sat - sat_then_wet)) > 1e-4


# ---------------------------------------------------------------- the blend


def test_mix_is_the_dry_wet_and_defaults_to_wet_only():
    """`mix` is what makes the device usable off a return — and its 1.0 default is
    what keeps every pre-2026-07-20 return bit-identical."""
    t = np.arange(SR) / SR
    dry = (0.4 * np.sin(2 * np.pi * 220.0 * t))[:, np.newaxis].astype(np.float64)
    entry = {"type": "return_reverb", "decay_s": 2.0, "pre_delay_ms": 0.0, "tone": 0.5}

    full_wet = dsp._wet_device(dry, SR, entry)                       # no key at all
    assert np.array_equal(full_wet, dsp._wet_device(dry, SR, {**entry, "mix": 1.0}))

    bypassed = dsp._wet_device(dry, SR, {**entry, "mix": 0.0})
    assert np.array_equal(bypassed, dry), "mix 0 must return the dry array bit-exactly"

    half = dsp._wet_device(dry, SR, {**entry, "mix": 0.25})
    assert np.allclose(half, full_wet * 0.25 + dry * 0.75)           # a real crossfade
    assert _rms(half) != pytest.approx(_rms(full_wet), rel=1e-3)


def test_reference_monitor_convolution_is_audible_and_matches_offline():
    """The monitor oracle must not turn convolution into dry passthrough."""
    dry = np.zeros((SR, 1), dtype=np.float64)
    dry[0, 0] = 1.0
    entry = {"type": "return_convolution", "ir": "room", "decay_s": 0.4,
             "pre_delay_ms": 0.0, "tone": 0.5, "mix": 1.0}
    reference = reference_monitor._wet_device(dry, SR, entry)
    offline = dsp._wet_device(dry, SR, entry)
    assert not np.array_equal(reference, dry)
    assert np.array_equal(reference, offline)


def test_returns_still_render_wet_only(tmp_path):
    """The return placement is untouched: no `mix` written, so the wet replaces."""
    project = _project(tmp_path, "return-form")
    session = Session(project)
    rid = session.add_return()["return_id"]
    session.add_stem_device(f"return:{rid}", "return_reverb")
    entry = next(e for e in session.returns[rid]["chain"] if e["type"] == "return_reverb")
    assert entry["mix"] == 1.0, "a device added to a RETURN must stay 100% wet"

    # ...while the very same device added to a track lands blended
    session.add_stem_device("vox", "return_reverb")
    insert = next(e for e in session.stems["vox"]["chain"] if e["type"] == "return_reverb")
    assert insert["mix"] == Session.INSERT_WET_MIX < 1.0


# ---------------------------------------------------------------- the palettes


@pytest.mark.parametrize("wet_type", WET_TYPES)
def test_user_can_add_a_space_device_to_every_target(tmp_path, wet_type):
    session = Session(_project(tmp_path, f"palette-{wet_type}"))
    rid = session.add_return()["return_id"]
    for target in ("vox", f"return:{rid}"):
        assert session.add_stem_device(target, wet_type)["type"] == wet_type
    assert session.add_master_device(wet_type)["type"] == wet_type


@pytest.mark.parametrize("wet_type", WET_TYPES)
def test_agent_may_target_a_stem_bus_or_master(wet_type):
    """The total-agent-control invariant: no placement the user has that the agent
    doesn't (docs/10-daw.md). The proposal grammar has to allow all three."""
    assert wet_type in SUPPORTED_ACTION_TYPES
    assert wet_type in MASTER_ACTION_TYPES
    assert wet_type in BUS_ACTION_TYPES


def test_native_slot_budget_is_enforced_honestly(tmp_path):
    """MonitorChain hosts kMaxWetInserts space devices per chain; a 4th would parse into
    nothing and render offline-only. Refuse it with a real message, don't drop it."""
    session = Session(_project(tmp_path, "slot-cap"))
    for wet_type in WET_TYPES:
        session.add_stem_device("vox", wet_type)
    with pytest.raises(ValueError, match="space devices"):
        session.add_stem_device("vox", "return_reverb")
    with pytest.raises(ValueError, match="space devices"):
        for wet_type in WET_TYPES + ("return_reverb",):
            session.add_master_device(wet_type)

    rid = session.add_return()["return_id"]
    for wet_type in WET_TYPES:
        session.add_stem_device(f"return:{rid}", wet_type)
    with pytest.raises(ValueError, match="space devices"):
        session.add_stem_device(f"return:{rid}", "return_delay")


# ---------------------------------------------------------------- the agent path


@pytest.mark.anyio
async def test_agent_proposes_an_insert_and_undo_restores(tmp_path):
    project = _project(tmp_path, "agent-insert")
    async with engine(project) as client:
        proposal = await call(client, "propose_actions", {"req": {
            "rationale": "slap delay that IS the vocal sound",
            "predicted_impact": [],
            "actions": [{"type": "return_delay", "stem_id": "vox",
                         "delay_ms": 180.0, "feedback": 0.2, "tone": 0.5, "mix": 0.3}],
        }})
        assert proposal["accepted"], proposal["violations"]
        pid = proposal["proposal_id"]
        assert (await call(client, "apply_proposal", {"proposal_id": pid}))["ok"]

        session = Session(project)
        assert any(e["type"] == "return_delay" for e in session.stems["vox"]["chain"])
        assert "returns" not in session._state or not session._state["returns"], \
            "an INSERT must not conjure an FX return"

        assert (await call(client, "undo_proposal", {"proposal_id": pid}))["ok"]
        assert not Session(project).stems["vox"]["chain"], "undo must restore the chain"


@pytest.mark.anyio
async def test_agent_wet_cap_binds_the_agent_not_the_user(tmp_path):
    """Guardrails bind the AGENT: it may blend a space device in, not drown the
    source. The user's own knob still goes to 100% (shared control)."""
    project = _project(tmp_path, "wet-cap")
    async with engine(project) as client:
        proposal = await call(client, "propose_actions", {"req": {
            "rationale": "drown it", "predicted_impact": [],
            "actions": [{"type": "return_reverb", "stem_id": "vox",
                         "decay_s": 2.0, "pre_delay_ms": 10.0, "tone": 0.5, "mix": 1.0}],
        }})
        assert not proposal["accepted"]
        assert any(v["rule"] == "wet_insert_mix_max" for v in proposal["violations"])

        # the SAME 100% wet is fine on a RETURN — that is what a return is for
        ok = await call(client, "propose_actions", {"req": {
            "rationale": "shared room", "predicted_impact": [],
            "actions": [{"type": "return_reverb", "return_id": "room",
                         "decay_s": 2.0, "pre_delay_ms": 10.0, "tone": 0.5}],
        }})
        assert ok["accepted"], ok["violations"]

    # and the user's direct co-edit is uncapped
    session = Session(project)
    session.add_stem_device("vox", "return_reverb")
    session.set_master_device_param("return_reverb", 0, "mix", 1.0, target="vox")
    entry = next(e for e in session.stems["vox"]["chain"] if e["type"] == "return_reverb")
    assert entry["mix"] == 1.0


def test_action_needs_exactly_one_target():
    """Naming both targets is ambiguous (which chain owns it?); naming neither has
    no target at all — the model refuses both rather than guessing."""
    from mix_engine.models import ReturnReverbAction

    ReturnReverbAction(return_id="room")                 # the return form
    ReturnReverbAction(stem_id="vox")                    # the insert form
    for kwargs in ({}, {"return_id": "room", "stem_id": "vox"}):
        with pytest.raises(ValueError, match="exactly one"):
            ReturnReverbAction(**kwargs)


# ---------------------------------------------------------------- native parity


@pytest.mark.parametrize("wet_type", WET_TYPES)
def test_live_monitor_carries_the_mix(wet_type):
    """A device the live monitor can't translate renders offline-only (the
    daw-device-freedom trap): the type must be in MONITOR_TYPES *and* the resolver
    must forward `mix`, or the insert would be silent in the DAW."""
    from mix_engine.monitor_chain import MONITOR_TYPES, monitor_chain_params

    assert wet_type in MONITOR_TYPES
    entry = {"type": wet_type, "decay_s": 2.0, "pre_delay_ms": 10.0, "tone": 0.5,
             "delay_ms": 250.0, "feedback": 0.25, "mix": 0.4}
    params = monitor_chain_params([{"type": "gain", "gain_db": 0.0}, entry])
    emitted = next(e for e in params["entries"] if e["type"] == wet_type)
    assert emitted["mix"] == 0.4
