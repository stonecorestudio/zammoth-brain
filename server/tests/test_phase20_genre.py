"""Phase 20 — genre tonal-balance curves: a genre word in the brief compiles
to band + loudness targets without a reference file (docs/05 Phase 20)."""
import pytest

from orchestrator.targets import (GENRE_BAND_TOLERANCE_DB, GENRE_PROFILES,
                                  targets_from_brief)

STEMS = ["stem_a", "stem_b"]


def test_pop_master_compiles_to_curve_and_loudness():
    targets = {t.metric: t for t in targets_from_brief("pop master", STEMS)}
    assert targets["lufs_i"].goal == GENRE_PROFILES["pop"]["lufs_i"]
    assert targets["true_peak_dbtp"].max_value == -1.0
    for band, value in GENRE_PROFILES["pop"]["band_energy"].items():
        target = targets[f"band_energy.{band}"]
        assert target.goal == value
        assert target.tolerance == GENRE_BAND_TOLERANCE_DB


def test_explicit_loudness_beats_the_genre_opinion():
    targets = {t.metric: t for t in
               targets_from_brief("master this rock track for streaming",
                                  STEMS)}
    assert targets["lufs_i"].goal == -14.0  # streaming, not rock's -9.5
    assert "band_energy.low_mid" in targets  # the curve still applies


def test_genre_synonyms_resolve():
    targets = targets_from_brief("give it a trap master", STEMS)
    by_metric = {t.metric: t for t in targets}
    assert by_metric["lufs_i"].goal == GENRE_PROFILES["hip hop"]["lufs_i"]


def test_no_genre_means_no_curve():
    targets = targets_from_brief("master for streaming", STEMS)
    assert {t.metric for t in targets} == {"lufs_i", "true_peak_dbtp",
                                           "plr_db"}


def test_genreless_brief_still_raises():
    with pytest.raises(ValueError):
        targets_from_brief("make it sparkle", STEMS)


# ---------- whole-word matching (2026-08-05 audit) ----------

# The destination and genre tables were matched as bare substrings, so ordinary
# words silently compiled a full master the user never asked for: "adjust" ->
# 'dj' -> a −8 LUFS club master; "streamline" -> 'stream' -> −14 LUFS; "the bass
# pops" -> 'pop' -> −10 LUFS + 8 band targets. Worse than a wrong answer, because
# `loop.py` prefers this table over the LLM brief compiler — a confident wrong
# reading beat an honest "I don't know".
@pytest.mark.parametrize("brief", [
    "adjust the vocal a bit",              # a(dj)ust
    "make the guitar adjacent to the vocal",
    "streamline the intro",                # (stream)line
    "the bass pops too much",              # (pop)s
    "wrap it up",                          # w(rap)
])
def test_a_word_containing_a_keyword_is_not_a_mastering_brief(brief):
    """These must fall through to the LLM compiler, not resolve to a master."""
    with pytest.raises(ValueError):
        targets_from_brief(brief, STEMS)


@pytest.mark.parametrize("brief,expected_lufs", [
    ("master for streaming", -14.0),
    ("master this for spotify", -14.0),
    ("make it club ready", -8.0),
    ("master for the dj set", -8.0),
    ("master for broadcast", -23.0),
])
def test_real_destinations_still_resolve(brief, expected_lufs):
    targets = {t.metric: t for t in targets_from_brief(brief, STEMS)}
    assert targets["lufs_i"].goal == expected_lufs


@pytest.mark.parametrize("brief,genre", [
    ("mix it like a pop record", "pop"),
    ("give it a trap master", "hip hop"),
    ("a house master", "edm"),
    ("acoustic master", "acoustic"),
])
def test_real_genres_still_resolve(brief, genre):
    targets = {t.metric: t for t in targets_from_brief(brief, STEMS)}
    assert targets["lufs_i"].goal == GENRE_PROFILES[genre]["lufs_i"]
