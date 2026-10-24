"""The agent's prompts must state the bounds the ENGINE actually enforces.

A range written by hand into a prompt is a claim about `models.py` + `guardrails.py`,
and nothing was checking it. The 2026-09-08 device audit found several that had drifted:
`agents.py` invited multiband saturation band drives to 24 dB that `saturation_drive_max_db`
refuses at 12; both prompts offered a Utility `balance` of ±1 that `utility_balance_max`
caps at ±0.5 (0 on the master); both offered a `send` up to 0 dB that `return_send_max_db`
holds below −3; and both called the clipper MASTER-ONLY while a stem clipper is accepted
and rendered.

Every one of those is the same failure: the agent is told a number it will be refused for
using (or denied one it is allowed). The corrections are today's; THIS is what stops it
recurring — the numbers are parsed back out of the prompt text and compared against the
tables they came from, so moving a bound in `LIMITS` or a `Field(...)` without touching the
prompt fails here instead of quietly misleading the model.

The true bound for a given knob is the SCHEMA bound narrowed by whatever guardrail binds
the agent (guardrails bind the agent, never the user's hand knob — so the prompt's number
is routinely tighter than the model's, and that is correct).
"""
import re
import typing

import pytest
from pydantic import BaseModel

from mix_engine import models as M
from mix_engine.guardrails import LIMITS
from mix_engine.models import ProposalRequest
from mix_engine.session import SUPPORTED_ACTION_TYPES, Session


# ---------- the two prompt surfaces under test ----------

def _prompts() -> dict[str, str]:
    from orchestrator.agents import INVESTIGATE_INSTRUCTIONS
    from orchestrator.chat import SYSTEM_PROMPT
    return {"chat.SYSTEM_PROMPT": SYSTEM_PROMPT,
            "agents.INVESTIGATE_INSTRUCTIONS": INVESTIGATE_INSTRUCTIONS}


PROMPT_NAMES = sorted(_prompts())


# ---------- the sources of truth ----------

ACTION_MODELS: dict[str, type[BaseModel]] = {}
for _m in typing.get_args(M.Action):
    _f = _m.model_fields.get("type")
    if _f is not None:
        ACTION_MODELS[typing.get_args(_f.annotation)[0]] = _m

# Sub-object knobs (`low/mid/high:{…}`, `mid/side:{…}`) are a different namespace from the
# flat ones — the multiband saturator's band drive runs to 24 while its flat drive stops at
# 12 — so the two scopes are parsed and resolved separately. That distinction IS the F5 bug.
BAND_FIELDS = ("low", "mid", "high", "side")

# Guardrails that NARROW a schema bound for the agent. Keyed by (action type, field);
# the value says how the limit applies and which `LIMITS` key carries it, so a limit that
# moves — or is renamed away — breaks this test rather than the agent's next proposal.
#   "hi"  -> upper bound becomes min(schema_hi, LIMITS[key])
#   "lo"  -> lower bound becomes max(schema_lo, LIMITS[key])
#   "sym" -> both, at ±LIMITS[key]
NARROWED_BY_GUARDRAIL: dict[tuple[str, str], tuple[str, str]] = {
    ("comp", "ratio"):                  ("hi", "comp_ratio_max"),
    ("deess", "ratio"):                 ("hi", "comp_ratio_max"),        # guardrails.py:550
    ("dynamic_eq", "ratio"):            ("hi", "comp_ratio_max"),
    ("dynamic_eq", "gain_db"):          ("lo", "dynamic_eq_cut_min_db"),
    ("saturation", "drive_db"):         ("hi", "saturation_drive_max_db"),
    ("overdrive", "drive_db"):          ("hi", "overdrive_drive_max_db"),
    ("ott", "depth"):                   ("hi", "ott_depth_max"),
    ("chorus", "mix"):                  ("hi", "chorus_mix_max"),
    ("flanger", "mix"):                 ("hi", "flanger_mix_max"),
    ("flanger", "feedback"):            ("sym", "flanger_feedback_max"),
    ("auto_filter", "resonance"):       ("hi", "auto_filter_resonance_max"),         # Phase 63
    ("auto_filter", "lfo_amount_oct"):  ("hi", "auto_filter_lfo_amount_max_oct"),
    ("auto_filter", "env_amount_oct"):  ("sym", "auto_filter_env_amount_max_oct"),
    ("tilt", "tilt_db"):                ("sym", "tilt_max_db"),
    ("clip", "threshold_db"):           ("lo", "clip_threshold_min_db"),
    ("gate", "threshold_db"):           ("hi", "gate_threshold_max_db"),
    ("utility", "balance"):             ("sym", "utility_balance_max"),
    ("duck", "depth_db"):               ("hi", "duck_depth_max_db"),
    ("tune", "cents"):                  ("sym", "tune_cents_max"),
    ("transpose", "semitones"):         ("sym", "transpose_semitones_max"),
    ("spectral_denoise", "reduction_db"): ("hi", "spectral_denoise_reduction_max_db"),
    ("send", "gain_db"):                ("hi", "return_send_max_db"),
    ("return_reverb", "mix"):           ("hi", "wet_insert_mix_max"),
    ("return_delay", "mix"):            ("hi", "wet_insert_mix_max"),
    ("return_convolution", "mix"):      ("hi", "wet_insert_mix_max"),
    ("return_reverb", "decay_s"):       ("hi", "return_reverb_decay_max_s"),
    ("return_convolution", "decay_s"):  ("hi", "return_convolution_decay_max_s"),
    ("return_delay", "feedback"):       ("hi", "return_delay_feedback_max"),
    # The multiband saturator's per-band drive obeys the same ceiling as the flat one
    # (`eff_drive = max(flat, *bands)`, guardrails.py:415).
    ("SatBand", "drive_db"):            ("hi", "saturation_drive_max_db"),
}


def _sub_model(model: type[BaseModel], field: str) -> type[BaseModel] | None:
    spec = model.model_fields.get(field)
    if spec is None:
        return None
    for arg in typing.get_args(spec.annotation) or (spec.annotation,):
        if isinstance(arg, type) and issubclass(arg, BaseModel):
            return arg
    return None


def _schema_bounds(model: type[BaseModel], field: str) -> tuple[float, float] | None:
    spec = model.model_fields.get(field)
    if spec is None:
        return None
    lo = hi = None
    for meta in spec.metadata:
        for attr in ("ge", "gt"):
            if getattr(meta, attr, None) is not None:
                lo = float(getattr(meta, attr))
        for attr in ("le", "lt"):
            if getattr(meta, attr, None) is not None:
                hi = float(getattr(meta, attr))
    return None if lo is None or hi is None else (lo, hi)


def _agent_bounds(model: type[BaseModel], field: str) -> tuple[float, float] | None:
    """What the agent may actually propose: the schema bound, narrowed by any guardrail."""
    schema = _schema_bounds(model, field)
    if schema is None:
        return None
    lo, hi = schema
    rule = NARROWED_BY_GUARDRAIL.get((_type_of(model), field))
    if rule is None:
        return (lo, hi)
    how, key = rule
    assert key in LIMITS, (
        f"this test maps {_type_of(model)}.{field} to a guardrail {key!r} that no longer "
        "exists in LIMITS — fix the map (and check the prompt still states the real bound)")
    cap = float(LIMITS[key])
    if how == "hi":
        hi = min(hi, cap)
    elif how == "lo":
        lo = max(lo, cap)
    else:
        lo, hi = max(lo, -cap), min(hi, cap)
    return (lo, hi)


def _type_of(model: type[BaseModel]) -> str:
    spec = model.model_fields.get("type")
    return typing.get_args(spec.annotation)[0] if spec is not None else model.__name__


# ---------- parsing the prompt ----------

_NUM = r"-?\d+(?:\.\d+)?"
_CLAIM = re.compile(rf"\b([a-z_]+)\??:\s*({_NUM})\.\.({_NUM})")
_BAND_BLOCK = re.compile(r"\b(?:low/mid/high|low|mid|high|side)\??:\s*\{([^}]*)\}")
_BULLET = re.compile(r'^- ([a-z_]+): \{type:"[a-z_]+"', re.M)


def _bullets(blob: str) -> list[tuple[str, str]]:
    """(label, body) per device bullet — the body runs to the next bullet."""
    starts = [(m.group(1), m.start()) for m in _BULLET.finditer(blob)]
    return [(label, blob[start:(starts[i + 1][1] if i + 1 < len(starts) else len(blob))])
            for i, (label, start) in enumerate(starts)]


def _models_named_in(body: str) -> list[type[BaseModel]]:
    """Some bullets carry siblings (`pan` + `width`, the three return effects + `send`),
    so a claim is checked against every action the bullet actually declares."""
    return [ACTION_MODELS[t] for t in dict.fromkeys(re.findall(r'\{type:"([a-z_]+)"', body))
            if t in ACTION_MODELS]


def prompt_range_mismatches(blob: str) -> list[str]:
    """Every `param: lo..hi` in the blob that disagrees with what the engine allows."""
    bad: list[str] = []
    for label, body in _bullets(blob):
        body = body.replace("\\\n", " ")
        flat_models = _models_named_in(body)
        if not flat_models:
            continue
        band_models = [sub for m in flat_models for f in BAND_FIELDS
                       if (sub := _sub_model(m, f)) is not None]
        scopes = (("", _BAND_BLOCK.sub(" ", body), flat_models),
                  (" band", " ".join(_BAND_BLOCK.findall(body)), band_models))
        for scope, text, pool in scopes:
            for claim in _CLAIM.finditer(text):
                param, lo, hi = claim.group(1), float(claim.group(2)), float(claim.group(3))
                allowed = [b for m in pool if (b := _agent_bounds(m, param)) is not None]
                if not allowed:
                    continue          # not a knob of this device (prose that reads like one)
                if (lo, hi) not in allowed:
                    bad.append(
                        f"{label}{scope}.{param}: prompt says {lo:g}..{hi:g}, engine allows "
                        + " or ".join(f"{a:g}..{b:g}" for a, b in allowed))
    return bad


# ---------- the gate ----------

@pytest.mark.parametrize("name", PROMPT_NAMES)
def test_every_range_the_prompt_states_is_the_range_the_engine_allows(name):
    mismatches = prompt_range_mismatches(_prompts()[name])
    assert not mismatches, (
        f"{name} states {len(mismatches)} range(s) the engine does not honour — the agent is "
        "being invited to propose values that will be refused (or denied ones it may use):\n  "
        + "\n  ".join(mismatches))


@pytest.mark.parametrize("name", PROMPT_NAMES)
def test_the_check_is_not_vacuous(name):
    """It must PARSE a real number of claims — a regex that quietly matches nothing
    would let every one of the drifts above through while looking green."""
    blob = _prompts()[name]
    parsed = sum(len(_CLAIM.findall(body)) for _, body in _bullets(blob))
    assert parsed >= 40, f"{name}: only {parsed} numeric range claims parsed out of the catalog"


# ---------- negative controls: prove the gate fires ----------

@pytest.mark.parametrize("name", PROMPT_NAMES)
def test_a_wrong_number_in_the_prompt_is_caught(name):
    """Hand-doctor one claim; the checker must see it."""
    doctored = _prompts()[name].replace("depth: 0..0.25", "depth: 0..1", 1)
    assert doctored != _prompts()[name], "the OTT depth claim moved — retarget this control"
    mismatches = prompt_range_mismatches(doctored)
    assert any("ott.depth" in m for m in mismatches), \
        f"a prompt claiming OTT depth 0..1 (the guardrail stops at 0.25) went unnoticed: {mismatches}"


@pytest.mark.parametrize("name", PROMPT_NAMES)
def test_a_guardrail_that_moves_without_the_prompt_is_caught(name, monkeypatch):
    """The other direction, and the one that matters most: the prompt is untouched and
    the LIMIT moves under it."""
    monkeypatch.setitem(LIMITS, "saturation_drive_max_db", 9.0)
    mismatches = prompt_range_mismatches(_prompts()[name])
    assert any("saturation band.drive_db" in m for m in mismatches), \
        f"saturation_drive_max_db moved to 9 dB and the prompt still said 12: {mismatches}"


# ---------- the non-numeric claims from the same audit ----------

@pytest.mark.parametrize("name", PROMPT_NAMES)
def test_every_action_type_the_engine_supports_is_named_in_the_prompt(name):
    """`spectrogram` was the one member of SUPPORTED_ACTION_TYPES neither prompt named —
    a device the agent could add, could see in a chain, and had no word for."""
    named = set(re.findall(r'\{type:"([a-z_]+)"', _prompts()[name]))
    missing = sorted(SUPPORTED_ACTION_TYPES - named)
    assert not missing, f"{name} never names {missing} — built, addable, and invisible to the model"


def _clip_bullet(blob: str) -> str:
    for label, body in _bullets(blob):
        if label == "clip":
            return body.replace("\\\n", " ")
    raise AssertionError("no clip bullet in this prompt")


def _accepts(session: Session, **action) -> bool:
    return session.propose(ProposalRequest(rationale="placement probe for the prompt gate",
                                           predicted_impact=[], actions=[action])).accepted


@pytest.mark.parametrize("name", PROMPT_NAMES)
def test_the_clip_bullets_placement_claim_matches_what_the_session_accepts(name, demo_project):
    """Both prompts called the clipper MASTER-ONLY. It is not: a stem clipper is accepted
    and rendered (Phase 43 device freedom), and since 2026-09-08 a BUS one is too (`clip`
    joined `BUS_ACTION_TYPES`; it had been in `_BUS_KINDS` and the bus renderer all along,
    so only the proposal path lagged). That hole is closed, so the bullet must no longer
    warn the bus form is refused. Asserted against the real Session rather than a table,
    so it stays true whichever layer moves."""
    body = _clip_bullet(_prompts()[name])
    session = Session(demo_project)
    stem_ok = _accepts(session, type="clip", stem_id="stem_a", threshold_db=-3.0, hardness=0.5)
    grouped = session.propose(ProposalRequest(
        rationale="a bus to probe the clipper's placement against",
        predicted_impact=[],
        actions=[{"type": "group", "bus_id": "drums", "members": ["stem_a", "stem_b"]}]))
    assert grouped.accepted, "could not make a bus to probe against"
    session.apply(grouped.proposal_id)
    bus_ok = _accepts(session, type="clip", stem_id="drums", threshold_db=-3.0, hardness=0.5)

    says_master_only = "master-only" in body.lower()
    assert says_master_only is not stem_ok, (
        f"{name}: the session {'accepts' if stem_ok else 'refuses'} a clip on a stem, and the "
        f"prompt {'says' if says_master_only else 'does not say'} MASTER-ONLY")
    says_bus_refused = "refused" in body.lower()
    assert says_bus_refused is not bus_ok, (
        f"{name}: the session {'accepts' if bus_ok else 'refuses'} a clip on a bus, and the "
        f"prompt {'warns it is refused' if says_bus_refused else 'does not warn'} — if a bus "
        "clipper now works, delete that sentence")


def _master_floor_segment(blob: str) -> str:
    i = blob.find("master_width_min")
    assert i >= 0, "neither the floor nor its rule name is anywhere in this prompt"
    return blob[max(0, i - 600):i + 600]


@pytest.mark.parametrize("name", PROMPT_NAMES)
def test_the_master_width_floor_is_stated_with_its_real_numbers(name):
    """`master_width_min` (2026-09-08) refuses an agent narrowing the MASTER below half
    the side, exempting bands that end in the lows — while every width bullet still said
    plain `0..2`. Both numbers are pinned, next to the rule name, so moving either without
    the prompt fails here."""
    from mix_engine.guardrails import _MASTER_MONO_LOWS_MAX_HZ

    segment = _master_floor_segment(_prompts()[name])
    floor = f"{LIMITS['master_width_min']:g}"
    assert floor in segment, f"{name}: the master width floor is not stated as {floor}"
    lows = f"{_MASTER_MONO_LOWS_MAX_HZ:g} Hz"
    assert lows in segment, f"{name}: the lows exemption is not stated as {lows}"


@pytest.mark.parametrize("name", PROMPT_NAMES)
def test_the_master_width_floor_check_fires_when_the_limit_moves(name, monkeypatch):
    monkeypatch.setitem(LIMITS, "master_width_min", 0.65)
    with pytest.raises(AssertionError):
        test_the_master_width_floor_is_stated_with_its_real_numbers(name)


_SLOT_CAPS = {
    "comps_per_stem_max": ("comp", re.compile(r"at most (\d+) comps")),
    "deessers_per_stem_max": ("deess", re.compile(r"at most (\d+) de-essers")),
}


@pytest.mark.parametrize("rule", sorted(_SLOT_CAPS))
@pytest.mark.parametrize("name", PROMPT_NAMES)
def test_the_dynamics_slot_caps_are_stated_with_their_real_counts(name, rule):
    """`comps_per_stem_max` / `deessers_per_stem_max` (2026-09-08) bound how many comp or
    de-ess entries one chain may carry — the native engine has only that many slots. The
    prompts stated the ranged-overlap rule but no COUNT, so Phase 35 phrase compression
    (several non-overlapping ranged entries per stem, accumulating across passes) could
    only discover the cap by being refused. The number is pinned next to its rule name, so
    moving the limit without the prompt fails here.

    Prose, not a `lo..hi` range, so the parser above cannot see it: matched by phrase and
    compared to `LIMITS`, never a bare `in` on the digit (a lone "4" is all over these
    bullets — `4:1`, `0.4-0.6` — and would pass while measuring nothing)."""
    _device, pattern = _SLOT_CAPS[rule]
    blob = _prompts()[name]
    assert rule in blob, f"{name}: rule {rule} is not cited anywhere in this prompt"
    stated = pattern.search(blob)
    assert stated, f"{name}: no '{pattern.pattern}' claim to check against {rule}"
    assert int(stated.group(1)) == LIMITS[rule], (
        f"{name}: the prompt states {stated.group(1)} where {rule} allows {LIMITS[rule]}")


@pytest.mark.parametrize("rule", sorted(_SLOT_CAPS))
@pytest.mark.parametrize("name", PROMPT_NAMES)
def test_the_slot_cap_check_fires_when_the_limit_moves(name, rule, monkeypatch):
    monkeypatch.setitem(LIMITS, rule, LIMITS[rule] + 1)
    with pytest.raises(AssertionError):
        test_the_dynamics_slot_caps_are_stated_with_their_real_counts(name, rule)


@pytest.mark.parametrize("name", PROMPT_NAMES)
def test_the_one_channel_polarity_flip_is_marked_a_stem_repair(name, demo_project):
    """`master_polarity_split` refuses invert_left/right on the master; the utility bullet
    described the move with no target caveat."""
    session = Session(demo_project)
    assert not _accepts(session, type="utility", stem_id="master", invert_left=True), \
        "the master now takes a one-channel flip — drop the caveat from both prompts"
    assert _accepts(session, type="utility", stem_id="stem_a", invert_left=True), \
        "a stem no longer takes a one-channel flip — the bullet needs rewriting"
    assert "master_polarity_split" in _prompts()[name], \
        f"{name}: offers invert_left/right without saying the master refuses it"


@pytest.mark.parametrize("name", PROMPT_NAMES)
def test_band_width_free_edges_are_offered(name, demo_project):
    """`BandWidthAction` grew `lo_hz`/`hi_hz` in 2026-07-19; the run agent's bullet still
    said 'one fixed labeled band only', so it had to round a real request to a named band."""
    assert _accepts(Session(demo_project), type="band_width", stem_id="stem_a", band="mid",
                    width=0.6, lo_hz=700.0, hi_hz=1200.0), \
        "the engine no longer takes free band edges — update the prompt, then this test"
    body = next(b for label, b in _bullets(_prompts()[name]) if label == "band_width")
    assert "lo_hz" in body and "hi_hz" in body, \
        f"{name}: band_width takes free edges the prompt never mentions"
