"""Instrument-role resolution (2026-07-30).

`classify_stem` only ever emits six tags, so before this a stem literally NAMED
"snare_top" was invisible to a request that said "snare": the tag path had no
snare, and the id path required the whole id to appear in the sentence. These
tests pin the fix — and, just as importantly, the false positives it must NOT
introduce, because a wrong stem match is worse than falling through.
"""
import pytest

from orchestrator.targets import (INSTRUMENT_WORDS, TAG_SYNONYMS, resolve_stem,
                                  _roles_named, _roles_of_stem)

STEMS = ["snare_top", "lead_vox", "gtr_l", "kick_in", "track_01", "piano_di"]
TAGS = {"lead_vox": ["vocal"], "kick_in": ["kick"]}


# ---------- the hole this closes ----------

@pytest.mark.parametrize("brief,expected", [
    ("the snare has no crack", "snare_top"),     # no classifier tag for snare
    ("pull the gtr back", "gtr_l"),              # abbreviation in the id
    ("the guitars sound boxy", "gtr_l"),         # plural word -> abbreviated id
    ("the piano is muddy", "piano_di"),          # no classifier tag for piano
    ("the vox is buried", "lead_vox"),
    ("more kick", "kick_in"),
])
def test_instrument_named_in_the_brief_resolves(brief, expected):
    assert resolve_stem(brief, STEMS, TAGS) == expected


def test_name_beats_the_classifier_tag():
    # the user typed the track name; the tag is a spectral guess. Name wins.
    stems = ["snare_top", "track_09"]
    tags = {"track_09": ["vocal"], "snare_top": ["vocal"]}   # misclassified
    assert resolve_stem("the snare needs more crack", stems, tags) == "snare_top"


# ---------- false positives it must not introduce ----------

@pytest.mark.parametrize("brief", [
    "which track is loudest",        # "track" is a noise token
    "what part needs clarity",       # "hat" must not match inside "what"
    "oh that is better",             # "oh" is an overhead id, not prose
    "can you undo that",
])
def test_no_false_positive(brief):
    assert resolve_stem(brief, STEMS, TAGS) is None


def test_ambiguous_words_do_not_hijack_a_request():
    # "ride the vocal up" is a fader move, not the ride cymbal
    assert resolve_stem("ride the vocal up", STEMS, TAGS) == "lead_vox"
    assert "cymbal" not in _roles_named("ride the vocal up")


def test_symptom_words_are_not_instrument_names():
    # these live in the knowledge pack's symptom table; matching them as
    # instruments would resolve the wrong stem
    for word in ("snap", "impact", "sweep"):
        assert not any(word in words for words in INSTRUMENT_WORDS.values()), \
            f"{word!r} is a symptom, not an instrument"


# ---------- the two vocabularies cannot drift ----------

def test_tag_synonyms_are_derived_from_the_shared_vocabulary():
    for tag, words in TAG_SYNONYMS.items():
        assert words is INSTRUMENT_WORDS[tag], \
            "TAG_SYNONYMS must be derived from INSTRUMENT_WORDS, not copied"


def test_classifier_tags_all_have_vocabulary():
    # the six tags analysis.classify_stem can emit (full_mix is not addressable
    # by name, so it is deliberately absent)
    for tag in ("vocal", "bass", "kick", "cymbal", "pad"):
        assert tag in INSTRUMENT_WORDS


# ---------- id tokenisation ----------

@pytest.mark.parametrize("stem_id,role", [
    ("snare_top", "snare"), ("SNARE 2", "snare"), ("gtr_l", "guitar"),
    ("lead_vox", "vocal"), ("kick_in", "kick"), ("bass drum 1", "kick"),
    ("Hi-Hat", "hat"), ("808", "bass"), ("OH_L", "cymbal"),
])
def test_stem_name_implies_its_role(stem_id, role):
    assert role in _roles_of_stem(stem_id)


def test_opaque_ids_imply_nothing():
    for stem_id in ("track_01", "audio_07", "stem_3", "untitled"):
        assert _roles_of_stem(stem_id) == set()


# ---------- the classifier reads the NAME first (2026-08-21) ----------
#
# A plucked/synth bass clears the kick rule's thresholds easily (low-dominated,
# and 30 notes a minute is already transient_density >= 0.5), so a stem plainly
# named "bass" used to be tagged `kick` — and Auto Mix's TAG_BUS_FAMILY routes
# every `kick` into a bus called "drums". James watched a bass land there.

from mix_engine.analysis import classify_stem
from mix_engine.instruments import tag_from_name


@pytest.mark.parametrize("stem_id,tag", [
    ("bass", "bass"), ("Bass 2", "bass"), ("808 bass", "bass"),
    ("sub bass", "bass"), ("bassline", "bass"),
    ("bass drum", "kick"), ("kick_01", "kick"), ("BD", "kick"),
    ("hihat", "cymbal"), ("OH_L", "cymbal"), ("crash", "cymbal"),
    ("lead_vox", "vocal"), ("pad warm", "pad"),
])
def test_a_name_that_says_the_instrument_wins(stem_id, tag):
    assert tag_from_name(stem_id) == tag


@pytest.mark.parametrize("stem_id", [
    "Audio 1", "track_01", "amen_break", "untitled", "bassoon", "brass",
    "guitar", "snare_top", "drums", "",
])
def test_a_name_that_doesnt_say_stays_silent(stem_id):
    # No classifier tag of its own (snare/guitar/drums) or nothing recognisable:
    # the name must not speak, so the acoustics still decide.
    assert tag_from_name(stem_id) is None


def _stem(stem_id, **over):
    """A low-dominated, punchy stem — the shape that trips the kick rule."""
    from mix_engine.models import StemAnalysis
    fields = dict(stem_id=stem_id, lufs_s=-18.0, peak_dbfs=-3.0,
                  band_energy={"sub": -6.0, "bass": -6.0, "low_mid": -30.0,
                               "mid": -30.0, "high_mid": -36.0,
                               "presence": -40.0, "brilliance": -46.0,
                               "air": -50.0},
                  transient_density=2.0, transient_energy_ratio=0.4,
                  dynamic_spread_db=8.0, silent=False, applied_chain=[])
    fields.update(over)
    return StemAnalysis(**fields)


def test_a_plucked_bass_is_not_tagged_kick():
    assert classify_stem(_stem("bass")) == "bass"          # the name overrules
    assert classify_stem(_stem("Bass Guitar")) == "bass"
    assert classify_stem(_stem("bass drum")) == "kick"     # …and knows a kick
    assert classify_stem(_stem("kick_in")) == "kick"


def test_an_opaque_name_still_falls_through_to_the_metrics():
    assert classify_stem(_stem("track_01")) == "kick"      # punchy + low = kick
    assert classify_stem(_stem("track_01", transient_density=0.1,
                               transient_energy_ratio=0.05)) == "bass"


def test_a_silent_stem_stays_untagged_whatever_its_name():
    assert classify_stem(_stem("bass", silent=True)) is None
