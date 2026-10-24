"""Phase 39 — intent understanding: tag-aware stem resolution, tone/space
intent targets, plain-English brief read-back, and notes/taste folded into the
free-form brief compiler. All deterministic — no LLM."""
import pytest

from orchestrator.agents import compile_brief_prompt
from orchestrator.targets import interpret_brief, targets_from_brief

# A small "analysis" dump with opaque stem ids — the realistic case the tag
# resolution exists for (auto-tagged, but named track_NN, not "vocals").
ANALYSIS = {
    "stems": ["track_01", "track_03"],
    "band_energy": {"sub": -10.0, "bass": -7.0, "low_mid": -8.0, "mid": -8.0,
                    "high_mid": -9.0, "presence": -10.0, "brilliance": -12.0,
                    "air": -15.0},
    "stereo_correlation": 0.70,
    "crest_factor_db": 11.0,
    "harshness_db": -18.0,
}
TAGS = {"track_01": ["vocal"], "track_03": ["kick"]}


# ---------- A: resolve stems by tag ----------

def test_clarity_resolves_stem_by_tag():
    # "the vocal" names no stem id, but track_01 is tagged vocal
    targets = targets_from_brief("make the vocal clearer", ANALYSIS["stems"],
                                 ANALYSIS, TAGS)
    masking = [t for t in targets if t.metric.startswith("masking(")]
    assert masking, "expected masking targets for a clarity ask"
    assert all(t.metric.startswith("masking(track_01,") for t in masking)


def test_clarity_without_tags_still_needs_an_id():
    # the old behavior is the fallback: no tags + opaque ids => raises
    with pytest.raises(ValueError):
        targets_from_brief("make the vocal clearer", ANALYSIS["stems"], ANALYSIS)


def test_literal_id_still_wins_over_tag():
    targets = targets_from_brief("track_01 clarity", ANALYSIS["stems"],
                                 ANALYSIS, TAGS)
    assert any(t.metric.startswith("masking(track_01,") for t in targets)


def test_word_boundary_avoids_false_tag_match():
    tags = {"track_05": ["cymbal"]}
    # 'what' must not resolve via the cymbal synonym 'hat'
    with pytest.raises(ValueError):
        targets_from_brief("what part needs clarity", ["track_05"], {}, tags)


# ---------- B: tone / space intents ----------

def test_warm_intent_tilts_bands_relative_to_current():
    targets = {t.metric: t for t in targets_from_brief(
        "make it warmer", ANALYSIS["stems"], ANALYSIS)}
    assert targets["band_energy.low_mid"].goal == pytest.approx(-6.5)  # -8 + 1.5
    assert targets["band_energy.air"].goal == pytest.approx(-16.0)     # -15 - 1
    assert "stereo_correlation" not in targets


def test_wide_intent_lowers_correlation():
    (target,) = [t for t in targets_from_brief(
        "make it wider", ANALYSIS["stems"], ANALYSIS)
        if t.metric == "stereo_correlation"]
    assert target.goal == pytest.approx(0.55)  # 0.70 - 0.15


def test_punchy_intent_sets_a_crest_floor():
    (target,) = [t for t in targets_from_brief(
        "punchier please", ANALYSIS["stems"], ANALYSIS)
        if t.metric == "crest_factor_db"]
    assert target.min_value == pytest.approx(11.5)  # 11.0 + 0.5


def test_tone_intent_needs_analysis_to_anchor():
    # no analysis => no tone target => an otherwise-unparseable brief raises
    with pytest.raises(ValueError):
        targets_from_brief("make it warmer", ANALYSIS["stems"])


def test_genre_owns_its_bands_over_a_tone_nudge():
    # a genre profile already claims every band; warm must not duplicate one
    targets = targets_from_brief("warm pop master", ANALYSIS["stems"], ANALYSIS)
    metrics = [t.metric for t in targets]
    assert metrics.count("band_energy.low_mid") == 1


# ---------- D: plain-English brief read-back ----------

def test_interpret_brief_reads_back_in_english():
    phrases = interpret_brief(
        "master for streaming, warmer, vocal up front", ANALYSIS["stems"], TAGS)
    joined = " | ".join(phrases)
    assert "-14 LUFS" in joined
    assert "warmer" in joined
    assert "track_01 up front" in joined  # resolved by the vocal tag


def test_interpret_brief_empty_when_nothing_recognized():
    assert interpret_brief("do the thing", ANALYSIS["stems"]) == []


# ---------- C: notes + taste in the compiler prompt ----------

def test_compile_brief_prompt_includes_notes_and_taste():
    prompt = compile_brief_prompt(
        "make it pop", ANALYSIS,
        notes=["keep the vocal bright"],
        taste=[{"kind": "eq", "approved": 4, "rejected": 0}])
    assert "keep the vocal bright" in prompt
    assert "project_notes" in prompt
    assert "user_taste_decisions" in prompt
    assert "Honor project_notes" in prompt


def test_compile_brief_prompt_omits_absent_extras():
    prompt = compile_brief_prompt("make it pop", ANALYSIS)
    assert "project_notes" not in prompt
    assert "user_taste_decisions" not in prompt
    assert "Honor project_notes" not in prompt
