"""The instrument vocabulary: the words people use for instruments, and the
words that show up in stem NAMES, mapped to canonical roles.

It lives in the engine because BOTH halves need it and the engine may never
import the orchestrator (separation.py's rule): `analysis.classify_stem` reads
a stem's name here before it reads its acoustics, and `orchestrator.targets`
resolves the user's words and the stem's id through the same table. One
vocabulary, imported in both directions of the product — a second copy is how
"bass drum" ends up meaning one thing to the classifier and another to the
brief compiler.
"""
from __future__ import annotations

import re
from typing import Optional


# The words people use for instruments, mapped to canonical roles. One
# vocabulary serves BOTH directions (2026-07-30):
#
#   * the user's words  ("pull the gtr back")
#   * the stem's own id ("gtr_l", "snare_top", "lead_vox")
#
# which is the fix for a real hole: `classify_stem` only ever emits six tags
# (kick/bass/cymbal/vocal/pad/full_mix), so a stem literally NAMED "snare_top"
# used to be invisible to a request that said "snare" — the tag path had no
# snare, and the id path required the whole id to appear in the sentence.
# Matching canonical ROLES on both sides closes it without loosening anything:
# an unknown word still resolves to nothing.
#
# Deliberately excludes genuinely ambiguous words — "lead" (vocal or synth?),
# "top" / "in" / "out" (mic positions, not identity). A wrong stem match is
# worse than falling through.
INSTRUMENT_WORDS: dict[str, tuple[str, ...]] = {
    "vocal": ("vocal", "vocals", "vox", "voc", "vcl", "voice", "singer",
              "lead vocal", "leadvox"),
    "backing_vocal": ("backing", "backings", "bv", "bvs", "harmony",
                      "harmonies", "adlib", "adlibs", "double", "doubles"),
    "kick": ("kick", "kik", "kck", "bd", "bassdrum", "bass drum", "kick drum"),
    "snare": ("snare", "snr", "sd", "rimshot", "rim"),
    "clap": ("clap", "claps"),
    "hat": ("hat", "hats", "hihat", "hihats", "hi-hat", "hi hat", "hh"),
    "cymbal": ("cymbal", "cymbals", "crash", "crashes", "ride", "splash",
               "overhead", "overheads", "oh"),
    # NB: "snap"/"impact"/"sweep" are deliberately absent — they are symptom
    # words ("no snap", "the drop needs more impact"), not instrument names.
    "tom": ("tom", "toms", "rack tom", "floor tom"),
    "room": ("room", "rooms", "ambience", "ambient mic"),
    "drums": ("drum", "drums", "kit", "drumkit", "beat", "percussion", "perc"),
    "bass": ("bass", "basses", "bassline", "sub bass", "subbass", "sub", "808",
             "808s"),
    "guitar": ("guitar", "guitars", "gtr", "gtrs", "electric guitar",
               "acoustic guitar", "riff", "riffs"),
    "piano": ("piano", "pno", "keys", "keyboard", "rhodes", "wurli",
              "wurlitzer", "organ"),
    "synth": ("synth", "synths", "arp", "arps", "pluck", "plucks"),
    "pad": ("pad", "pads"),
    "strings": ("strings", "violin", "violins", "viola", "cello", "orchestra",
                "orchestral"),
    "horns": ("horn", "horns", "brass", "sax", "saxophone", "trumpet",
              "trombone"),
    "fx": ("fx", "riser", "risers", "whoosh", "foley"),
}

# Words that are fine INSIDE a stem id but must not match in free prose:
# "oh that's better" is not a request about the overheads, and "ride the vocal
# up" is a fader move, not the ride cymbal. Asymmetric on purpose.
_TEXT_AMBIGUOUS = frozenset({"oh", "ride", "rim", "double", "doubles"})

# The subset that maps onto tags `analysis.classify_stem` actually emits —
# derived, so the two vocabularies can never drift apart. Roles with no
# corresponding tag (snare, guitar, piano, …) still resolve by NAME above.
CLASSIFIER_TAGS = ("vocal", "bass", "kick", "cymbal", "pad")
TAG_SYNONYMS: dict[str, tuple[str, ...]] = {
    tag: INSTRUMENT_WORDS[tag] for tag in CLASSIFIER_TAGS
}

# Tokens that show up in stem ids but say nothing about what the stem IS.
# Without this, "which track is loudest" would resolve to "track_01".
_ID_NOISE_TOKENS = frozenset({
    "track", "tracks", "stem", "stems", "audio", "wav", "aiff", "flac", "mp3",
    "mix", "bounce", "print", "take", "takes", "file", "new", "untitled",
    "channel", "chan", "ch", "final", "copy", "edit", "comp", "bus", "group",
    "l", "r", "lr", "left", "right", "top", "bottom", "in", "out", "up", "dn",
    "mic", "di", "amp", "close", "far", "a", "b", "c", "d", "the", "and",
})

def roles_of_stem(stem_id: str) -> set[str]:
    """Roles implied by a stem's OWN name ('snare_top' -> {'snare'}). This is
    the strongest evidence available in the DAW, where track names are typed by
    the user rather than guessed by the six-way spectral classifier."""
    tokens = [t for t in re.split(r"[^a-z0-9]+", (stem_id or "").lower()) if t]
    meaningful = {t for t in tokens if t not in _ID_NOISE_TOKENS}
    phrase = " ".join(tokens)
    roles = set()
    for role, words in INSTRUMENT_WORDS.items():
        for word in words:
            if (word in meaningful) or (" " in word and word in phrase):
                roles.add(role)
                break
    return roles


# Roles that ARE one of the classifier's tags, most specific first. A "bass
# drum" reads as {kick, bass, drums}, and it is a kick — so kick is consulted
# before bass, and roles with no tag of their own (snare, guitar, drums, …)
# simply leave the name silent and let the acoustics decide.
_ROLE_TAG_ORDER: tuple[tuple[str, str], ...] = (
    ("kick", "kick"), ("hat", "cymbal"), ("cymbal", "cymbal"),
    ("vocal", "vocal"), ("backing_vocal", "vocal"),
    ("bass", "bass"), ("pad", "pad"),
)


def tag_from_name(stem_id: Optional[str]) -> Optional[str]:
    """The classifier tag a stem's NAME states outright, or None when the name
    doesn't say. Naming a track is a user signal about the instrument, the same
    kind as a user tag — see `analysis.classify_stem`, which trusts it over the
    metric rules."""
    if not stem_id:
        return None
    roles = roles_of_stem(stem_id)
    for role, tag in _ROLE_TAG_ORDER:
        if role in roles:
            return tag
    return None
