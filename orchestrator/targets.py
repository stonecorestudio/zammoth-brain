"""Brief -> metric-target translation. This table is product opinion (docs/04):
it will be tuned constantly. Keep entries dumb and explicit.

A Target is satisfied when distance() == 0. Metric names use the engine's
metric grammar (see server/mix_engine/impact.py): "lufs_i",
"masking(a,b).worst_band_excess_db", "stem(x).lufs_s", ...
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Optional

from .engine_models import (BAND_LABELS, CLASSIFIER_TAGS, INSTRUMENT_WORDS,
                            MASKING_METRIC_RE, STEM_METRIC_RE, TAG_SYNONYMS,
                            _TEXT_AMBIGUOUS, roles_of_stem)


class BriefNotUnderstood(ValueError):
    """The brief is not a mixing request (a greeting, a question, gibberish) — no targets
    derivable, and the LLM brief compiler returned nothing either. A ValueError subclass so
    existing `except ValueError` fallbacks still catch it; callers that want to chat instead of
    crash (the DAW Engineer panel) catch this specific type. See orchestrator/daw_bridge.py."""


@dataclass(frozen=True)
class Target:
    metric: str
    unit: str
    goal: Optional[float] = None       # aim for goal ± tolerance...
    tolerance: float = 0.0
    max_value: Optional[float] = None  # ...or stay at/below a ceiling...
    min_value: Optional[float] = None  # ...or at/above a floor (2026-06:
    #                                    the PLR punch guard below)

    def distance(self, value: Optional[float]) -> float:
        """How far from satisfied, in the metric's own units. 0 = met."""
        if value is None:
            return float("inf")
        if self.max_value is not None:
            return max(0.0, value - self.max_value)
        if self.min_value is not None:
            return max(0.0, self.min_value - value)
        return max(0.0, abs(value - self.goal) - self.tolerance)


# Phase 14: how far a "less harsh" / "less muddy" brief asks the composite
# to drop, relative to where the mix currently sits.
COMPOSITE_DROP_DB = 2.0

# 2026-06 review: the punch guard. Every loudness ask carries a PLR floor
# alongside the true-peak ceiling, so crushing the mix into the target now
# registers as an unmet target instead of silently passing. 4 dB only flags
# genuinely squashed masters — a clean -8 LUFS club master sits near 7.
PLR_FLOOR_DB = 4.0


#: What the master limiter can be aimed at (`models.LimiterAction.target_lufs_i`; pinned
#: against it by test_reader_p3_py8.py). The limiter is the only processor allowed to push
#: loudness, so a LUFS goal outside this range is a goal nothing can reach.
LIMITER_LUFS_RANGE = (-24.0, -6.0)


def clamp_loudness_goal(goal: float) -> tuple[float, str]:
    """A loudness goal held to the limiter's range, and the sentence that says so ("" when
    it was inside). Reader E6·R10: the rule agent clamped its LIMITER to the rail and said
    nothing while the TARGET stayed where it was asked - "master to -30 LUFS" was aimed at
    -24, never met, and the run ended "nothing to propose", the words for a fine mix; the
    LLM planner was billed passes toward the same unreachable number. Held here, where a
    target is compiled, the goal is one the run can meet and the user is told what it aimed at."""
    low, high = LIMITER_LUFS_RANGE
    if goal < low:
        return low, (f"{goal:g} LUFS is below the limiter's {low:g} LUFS floor - "
                     f"aimed at {low:g} LUFS instead")
    if goal > high:
        return high, (f"{goal:g} LUFS is above the limiter's {high:g} LUFS ceiling - "
                      f"aimed at {high:g} LUFS instead")
    return goal, ""


def hold_loudness_to_limiter(targets: list[Target]) -> tuple[list[Target], list[str]]:
    """Every compiled ``lufs_i`` goal through `clamp_loudness_goal`, whatever compiled it (a
    reference's own loudness is a goal too). Returns the targets and what was said."""
    held: list[Target] = []
    notes: list[str] = []
    for target in targets:
        if target.metric == "lufs_i" and target.goal is not None:
            goal, note = clamp_loudness_goal(float(target.goal))
            if note:
                target = replace(target, goal=goal)
                notes.append(note)
        held.append(target)
    return held, notes


def brief_notes(brief: str) -> list[str]:
    """What the deterministic compile changed about the brief as asked, for the run to say.
    Today: an explicit "-X LUFS" outside the limiter's range (`targets_from_brief` compiles
    it at the rail)."""
    asked = _explicit_lufs(brief.lower())
    if asked is None:
        return []
    _goal, note = clamp_loudness_goal(asked)
    return [note] if note else []


def _loudness_targets(goal_lufs: float) -> list[Target]:
    """The trio every loudness goal implies: the LUFS goal, the -1.0 dBTP
    ceiling (Phase 6), and the PLR punch floor (2026-06 review)."""
    return [
        Target(metric="lufs_i", unit="LUFS", goal=goal_lufs, tolerance=0.5),
        Target(metric="true_peak_dbtp", unit="dBTP", max_value=-1.0),
        Target(metric="plr_db", unit="dB", min_value=PLR_FLOOR_DB),
    ]


# Cues that ask for one stem to sit clearer in the mix (a masking target).
# Phase 39 widened these past "clarity" to the phrasings people actually use.
CLARITY_CUES = ("clarity", "clearer", "cut through", "up front", "sit louder")


# A QUESTION about the mix is not an instruction to mix it. Two signals, because
# neither alone is enough (both measured, 2026-08-05):
#   * a trailing "?" — the strongest signal there is, and it catches the ones a
#     leading-word rule misses entirely ("is the mix too muddy?", "do you think
#     this is too harsh?", both of which used to compile and EQ the master);
#   * a leading wh-word, for the unpunctuated "what does boxy mean".
# The leading-word rule is deliberately NARROW. Bare "when"/"which"/"how" open real
# instructions ("when the chorus hits make it warmer", "how about a bit less mud"),
# and on the no-LLM path — the signed-out user this module exists to serve —
# refusing those ships nothing at all. So it fires only on the interrogative frames
# that cannot be an instruction: a wh-word followed by a verb ("what does…",
# "why is…", "how do…"). Deliberately NOT "how about" / "what if" — those are
# suggestions wearing a question's clothes ("how about a bit less mud" is a request),
# and an unpunctuated one should compile. With a "?" the rule above catches them.
_QUESTION_RE = re.compile(
    r"\s*(?:what|why|how|where|when|which|who)\s+"
    r"(?:is|are|was|were|does|do|did|should|would|will|can)\b", re.IGNORECASE)

# ...and the "?" rule needs the SAME narrowing, for the same reason (2026-08-05 review).
# As written it fired on ANY trailing "?", so "make the vocal brighter?" and "can you
# make it warmer?" were refused — while the comment above was busy promising that "can
# you make it warmer" must still compile. Punctuation was deciding, and on the no-LLM
# path a refusal ships nothing at all.
#
# Two narrowings, in order:
#   * REQUEST FRAMES are requests whatever punctuation follows ("can you …", "how
#     about …", "please …"), so a "?" after one is not a question;
#   * otherwise a "?" only counts when the sentence is SHAPED like a question. English
#     fronts a wh-word or an auxiliary to ask one ("is the mix too muddy?", "do you
#     think this is harsh?"). An imperative with a question mark on the end is still an
#     imperative, and imperatives are what this compiler exists to read.
_REQUEST_FRAME_RE = re.compile(
    r"\s*(?:(?:can|could|would|will)\s+(?:you|we)|how\s+about|what\s+if|please)\b",
    re.IGNORECASE)
_INTERROGATIVE_RE = re.compile(
    r"\s*(?:what|why|how|where|when|which|who|whose"
    r"|is|are|was|were|am|do|does|did|have|has|had"
    r"|can|could|should|would|will|shall|may|might)\b", re.IGNORECASE)


def _is_question(text: str) -> bool:
    if _REQUEST_FRAME_RE.match(text):
        return False
    if _QUESTION_RE.match(text):
        return True
    return (text.rstrip().endswith("?")
            and bool(_INTERROGATIVE_RE.match(text)))


def _stem_band_ceilings(stem_id: str, bands: tuple[str, ...],
                        stem_analysis: Optional[dict]) -> list[Target]:
    """Per-stem band ceilings: "take THIS instrument's share of these bands down".

    Needs that stem's current band energy as the anchor, which the caller supplies
    (this function stays pure — the free understanding gate runs it with fixtures and
    no engine). Without an anchor we emit NOTHING rather than guessing: the brief then
    falls through to the LLM compiler, which is the honest outcome and mirrors how the
    session composites already skip when their analysis field is missing."""
    bands_now = (stem_analysis or {}).get("band_energy") or {}
    out: list[Target] = []
    for band in bands:
        current = bands_now.get(band)
        if current is None:
            continue
        out.append(Target(metric=f"stem({stem_id}).band_energy.{band}", unit="dB",
                          max_value=round(current - COMPOSITE_DROP_DB, 1)))
    return out


def targets_from_brief(brief: str, stems: list[str],
                       analysis: Optional[dict] = None,
                       stem_tags: Optional[dict[str, list[str]]] = None,
                       stem_analysis: Optional[dict] = None,
                       ) -> list[Target]:
    """Translate a user brief into metric goals, once, at session start.
    `analysis` (the current SessionAnalysis dump) enables relative goals —
    the Phase 14 composite asks ("less harsh") and the Phase 39 tone/space
    intents ("warmer", "wider") are nudges on the current value. `stem_tags`
    (Phase 39) lets the brief name an instrument by role ("the vocal") instead
    of by stem id."""
    text = brief.lower()
    analysis = analysis or {}
    targets: list[Target] = []

    # A QUESTION is not a mix instruction. "what does boxy mean" used to match the
    # composite table on the word "boxy" and compile into a master-wide mud target —
    # the user asked what a word meant and got their mix EQ'd. Leading wh-word only:
    # "can you make it warmer" is a request, not a question, and must still compile.
    if _is_question(text):
        raise BriefNotUnderstood(
            f"{brief!r} reads as a question, not a mix instruction")

    # Resolve ONCE, up front. The composite and tone rules below used to ignore the
    # named instrument entirely, so "the piano is muddy" became a session-wide mud
    # goal — understood perfectly, applied to the whole mix. Same resolver the
    # understanding eval scores against, deliberately: a weaker one here would drift
    # from the eval's definition of "named a stem".
    named = resolve_stem(text, stems, stem_tags)
    if named is not None and _PROTECTS_A_TRACK.search(text):
        # "less boxy, but don't touch the vocals" used to scope the cut ONTO the vocals (E6·R2).
        raise BriefNotUnderstood(
            f"{brief!r} names '{named}' as something to leave alone — a keyword table cannot "
            "tell the protected track from the one to work on")

    if (goal := _loudness_goal(text)) is not None:
        targets += _loudness_targets(goal)

    if any(cue in text for cue in CLARITY_CUES):
        stem = named   # already resolved above
        if stem is None:
            raise ValueError(
                f"brief asks for clarity but names no known stem; stems: {stems}")
        for other in stems:
            if other != stem:
                targets.append(Target(
                    metric=f"masking({stem},{other}).worst_band_excess_db",
                    unit="dB", max_value=3.0))

    # Phase 14 composites: the words users actually use, measured (skill v5)
    # Composites, scoped. mud_db is band_energy[low_mid] and harshness_db is
    # presence+brilliance (mix-analysis skill), so the per-stem form of each is a
    # ceiling on that stem's own band share. There is no per-stem mud_db/harshness_db
    # to aim at. With no stem named, the session goal is still right.
    if "harsh" in text:
        if named is not None:
            targets += _stem_band_ceilings(named, ("presence", "brilliance"),
                                           stem_analysis)
        elif analysis.get("harshness_db") is not None:
            targets.append(Target(
                metric="harshness_db", unit="dB",
                max_value=round(analysis["harshness_db"] - COMPOSITE_DROP_DB, 1)))
    if "mud" in text or "boxy" in text:
        if named is not None:
            targets += _stem_band_ceilings(named, ("low_mid",), stem_analysis)
        elif analysis.get("mud_db") is not None:
            targets.append(Target(
                metric="mud_db", unit="dB",
                max_value=round(analysis["mud_db"] - COMPOSITE_DROP_DB, 1)))

    # Phase 20: genre tonal-balance curves — "pop master" works without a
    # reference file. An explicit loudness ask in the brief wins over the
    # genre's loudness opinion (the block above already added it).
    if genre := _genre_of(text):
        profile = GENRE_PROFILES[genre]
        if _loudness_goal(text) is None:
            targets += _loudness_targets(profile["lufs_i"])
        for band, value in profile["band_energy"].items():
            targets.append(Target(metric=f"band_energy.{band}", unit="dB",
                                  goal=value,
                                  tolerance=GENRE_BAND_TOLERANCE_DB))

    # Phase 39: tone/space intents (warm/bright/dark/wide/narrow/punchy) as
    # measurable session targets layered on the current balance. Needs the
    # analysis to anchor (like the composites); a metric an earlier rule
    # already owns (e.g. a genre band) is left alone.
    if analysis:
        for intent in _tone_intents(text):
            targets += _tone_targets(intent, analysis,
                                     {t.metric for t in targets},
                                     named=named, stem_analysis=stem_analysis)

    if not targets:
        raise BriefNotUnderstood(
            f"no targets derivable from brief {brief!r}; known patterns: "
            "'master for streaming/spotify/club', 'master to -X LUFS', "
            "'<stem> clarity', 'less harsh', 'less muddy', "
            "'warmer/brighter/darker/wider/punchier', or a genre "
            f"({', '.join(GENRE_PROFILES)})")
    return targets


# Loudness goals per destination (Phase 6, docs/05). Product opinion — every
# loudness goal is paired with the −1.0 dBTP true-peak ceiling above.
LOUDNESS_GOALS_LUFS = {
    "spotify": -14.0,
    "youtube": -14.0,
    "streaming": -14.0,
    "stream": -14.0,
    "apple": -16.0,
    "club": -8.0,
    "dj": -8.0,
    "broadcast": -23.0,
    "ebu": -23.0,
}

_EXPLICIT_LUFS_RE = re.compile(r"(-\d+(?:\.\d+)?)\s*lufs")


def _explicit_lufs(text: str) -> Optional[float]:
    """The "-X LUFS" the brief says, as said."""
    if match := _EXPLICIT_LUFS_RE.search(text.replace("−", "-")):
        return float(match.group(1))
    return None


def _loudness_goal(text: str) -> Optional[float]:
    """An explicit '-X LUFS' in the brief wins; otherwise the destination table.

    Whole-word only (`_mentions`): as a substring, 'dj' fires inside "a(dj)ust"
    and 'stream' inside "(stream)line" — which silently turned "adjust the vocal
    a bit" into a −8 LUFS club master. A destination is a word the user said,
    never a fragment of another one.
    """
    if (asked := _explicit_lufs(text)) is not None:
        return clamp_loudness_goal(asked)[0]     # ...and `brief_notes` says when it moved
    for keyword, goal in LOUDNESS_GOALS_LUFS.items():
        if _mentions(text, (keyword,)):
            return goal
    return None


# ---------- genre tonal-balance profiles (Phase 20, docs/05) ----------

# Product opinion, dumb and explicit like the loudness table: a finished
# master's relative band energies (the band_energy definition — dB of band
# power relative to full-band power, so comparable across renders) plus the
# style's loudness goal. Tuned by ear-of-the-house; revise freely.
GENRE_BAND_TOLERANCE_DB = 1.5

GENRE_PROFILES: dict[str, dict] = {
    "pop": {"lufs_i": -10.0, "band_energy": {
        "sub": -10.0, "bass": -6.6, "low_mid": -7.4, "mid": -8.0,
        "high_mid": -9.2, "presence": -10.0, "brilliance": -12.2,
        "air": -15.2}},
    "rock": {"lufs_i": -9.5, "band_energy": {
        "sub": -12.2, "bass": -7.4, "low_mid": -6.6, "mid": -7.0,
        "high_mid": -8.9, "presence": -10.0, "brilliance": -12.2,
        "air": -17.0}},
    "hip hop": {"lufs_i": -9.0, "band_energy": {
        "sub": -7.0, "bass": -5.5, "low_mid": -8.5, "mid": -9.2,
        "high_mid": -10.5, "presence": -11.0, "brilliance": -13.0,
        "air": -17.0}},
    "edm": {"lufs_i": -8.0, "band_energy": {
        "sub": -7.4, "bass": -6.2, "low_mid": -9.2, "mid": -9.6,
        "high_mid": -10.0, "presence": -10.0, "brilliance": -11.0,
        "air": -14.0}},
    "acoustic": {"lufs_i": -14.0, "band_energy": {
        "sub": -14.0, "bass": -8.0, "low_mid": -7.0, "mid": -6.6,
        "high_mid": -8.5, "presence": -10.0, "brilliance": -12.2,
        "air": -15.2}},
}

# Brief words that select each profile. First match wins, in table order.
GENRE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "pop": ("pop",),
    "rock": ("rock",),
    "hip hop": ("hip hop", "hip-hop", "hiphop", "rap", "trap"),
    "edm": ("edm", "house", "techno", "electronic", "dance"),
    "acoustic": ("acoustic", "folk", "singer-songwriter"),
}


def _genre_of(text: str) -> Optional[str]:
    """Whole-word only, for the same reason as `_loudness_goal`: as substrings
    'pop' fires inside "the bass (pop)s too much" and 'rap' inside "(wrap) it
    up", each dragging in a full 8-band master the user never asked for."""
    for genre, keywords in GENRE_KEYWORDS.items():
        if _mentions(text, keywords):
            return genre
    return None


# ---------- tone / space intents (Phase 39, intent understanding) ----------

# The natural-language asks that used to fall straight through to the LLM
# compiler, mapped to measurable session targets layered on the current mix.
# Shared by runs and chat so "warmer" / "wider" / "punchier" mean the same
# everywhere, not just at the Auto Mix button. Product opinion — tune freely.
TONE_INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "warm": ("warm", "warmer", "warmth"),
    "bright": ("bright", "brighter", "crisp", "crispy", "airy", "air"),
    "dark": ("dark", "darker", "duller", "mellow", "smoother"),
    "wide": ("wide", "wider", "widen", "spacious"),
    "narrow": ("narrow", "narrower", "mono", "centered", "focused"),
    "punchy": ("punchy", "punch", "punchier", "snappy"),
}
# Relative band-energy nudges (dB vs the current balance) for the tonal intents.
TONE_INTENT_BANDS: dict[str, dict[str, float]] = {
    "warm": {"low_mid": 1.5, "brilliance": -1.0, "air": -1.0},
    "bright": {"presence": 1.0, "brilliance": 1.5, "air": 1.5},
    "dark": {"presence": -1.0, "brilliance": -1.5, "air": -1.5},
    # "too warm" / "less warm": warm, the other way (no keyword of its own — see _tone_intents)
    "cool": {"low_mid": -1.5, "brilliance": 1.0, "air": 1.0},
}
# What a tone word means when the brief says it has TOO MUCH of it ("too bright", "less air",
# "not so wide"): the opposite intent. Punch has no measurable opposite here, so "too punchy"
# compiles to nothing and falls to the LLM compiler like any brief this table cannot read.
TONE_INTENT_OPPOSITE: dict[str, Optional[str]] = {
    "bright": "dark", "dark": "bright", "wide": "narrow", "narrow": "wide",
    "warm": "cool", "punchy": None,
}
TONE_INTENT_BAND_TOLERANCE_DB = 1.0
# A width intent steers stereo correlation (lower = wider), bounded so "wider"
# never asks for a mono-incompatible mix and "narrower" never demands dead mono.
WIDTH_INTENT_CORR_STEP = 0.15
WIDTH_INTENT_CORR_FLOOR = 0.2
WIDTH_INTENT_CORR_CEIL = 0.95
WIDTH_INTENT_CORR_TOLERANCE = 0.1
# Punch is mostly a per-stem/transient thing; the honest master-level proxy is
# macro-dynamics — ask the crest factor not to collapse from where it is.
PUNCH_CREST_FLOOR_STEP_DB = 0.5

# Plain-English read-back per intent (Phase 39 D — see interpret_brief).
TONE_INTENT_PHRASES: dict[str, str] = {
    "warm": "warmer (fuller low-mids, softer top)",
    "bright": "brighter (more presence and air)",
    "dark": "darker, smoother top end",
    "wide": "wider stereo image",
    "narrow": "tighter, more centered image",
    "punchy": "punchier (keep the transients)",
    "cool": "less warm (leaner low-mids, a little more top)",
}


# A qualifier in front of a tone word that says the mix has TOO MUCH of it...
_TONE_EXCESS = re.compile(
    r"(?:\btoo\b|\boverly\b|\bless\b|\bnot\s+(?:so|as|that)\b)(?:\s+(?:much|of|the|a))*\s*$")
# ...and one that only says the word is NOT wanted, without saying which way to go ("it's not
# bright" is a complaint that wants more; "don't make it bright" wants no more): undecidable here.
_TONE_UNSURE = re.compile(
    r"(?:\bnot\b|n[’']t\b|\bno\b|\bnever\b|\bwithout\b)(?:\s+\w+){0,2}\s*$")
# The same letters in another sense: a punch-in is a recording move, air conditioning is a noise.
_TONE_OTHER_SENSE = {"punch": re.compile(r"^[\s-]*(?:in|out|ins|outs)\b"),
                     "air": re.compile(r"^\s*(?:con\b|conditioning|conditioner)")}


def _tone_intents(text: str) -> list[str]:
    """Tone/space intents named in the brief, in table order (Phase 39) — read WITH the word in
    front of them (full audit 2026-10-02, E6·R2). The table used to take the adjective as the
    direction whatever qualified it, so "it's too bright", "less bright" and "not so bright" all
    compiled to BRIGHTER targets: the rule engineer added +1.5 dB of brilliance and air and
    reported the targets met, and an LLM planner was judged against goals opposite to its brief.

    "too / overly / less / not so X" is the opposite intent; "not X enough" is still X; a bare
    negation ("it's not bright", "don't make it bright") cannot be decided by a table and raises
    BriefNotUnderstood so the LLM compiler (or the no-key reply) takes the brief instead."""
    found: list[str] = []
    for intent, words in TONE_INTENT_KEYWORDS.items():
        for word in words:
            for m in re.finditer(rf"\b{re.escape(word)}\b", text):
                before, after = text[:m.start()], text[m.end():]
                other = _TONE_OTHER_SENSE.get(word)
                if other is not None and other.match(after):
                    continue
                reading: Optional[str] = intent
                if re.match(r"\s*enough\b", after):
                    reading = intent                       # "not bright enough" asks for brighter
                elif _TONE_EXCESS.search(before):
                    reading = TONE_INTENT_OPPOSITE.get(intent)
                elif _TONE_UNSURE.search(before):
                    raise BriefNotUnderstood(
                        f"the brief says what the mix should NOT be ({word!r}) without saying "
                        "which way to take it — that needs a reader, not a keyword table")
                if reading is not None and reading not in found:
                    found.append(reading)
    for intent in found:
        if TONE_INTENT_OPPOSITE.get(intent) in found:
            raise BriefNotUnderstood(
                f"the brief asks for {intent!r} and its opposite — not a single direction")
    order = list(TONE_INTENT_KEYWORDS) + ["cool"]
    return sorted(found, key=order.index)


# "…but don't touch the vocals": a stem the brief names in order to PROTECT it. The stem
# resolver would otherwise read it as the brief's scope and aim the move at that very track.
_PROTECTS_A_TRACK = re.compile(
    r"\b(?:don[’']?t|do\s+not|never|without)\s+(?:touch|chang|mov|alter|affect)\w*\b"
    r"|\bleave\b[^.;]*\balone\b|\bhands\s+off\b|\bexcept(?:\s+for)?\s+the\b")


def _tone_targets(intent: str, analysis: dict, claimed: set[str],
                  named: Optional[str] = None,
                  stem_analysis: Optional[dict] = None) -> list[Target]:
    """Measurable targets for one tone/space intent, anchored on the current
    analysis. Skips any metric an earlier rule already owns.

    When the brief NAMED a stem, the band nudges apply to that stem's own bands
    instead of the master's — "give the vocal more air" is about the vocal. The
    space intents (wide/narrow/punchy) have no per-stem form here, so a named stem
    means they are SKIPPED rather than silently fired at the master: "make the kick
    punchier" must not become a master crest goal."""
    targets: list[Target] = []
    scoped = named is not None
    bands = ((stem_analysis or {}).get("band_energy") or {}) if scoped \
        else (analysis.get("band_energy") or {})
    prefix = f"stem({named})." if scoped else ""
    for band, nudge in TONE_INTENT_BANDS.get(intent, {}).items():
        metric = f"{prefix}band_energy.{band}"
        if metric in claimed or bands.get(band) is None:
            continue
        targets.append(Target(metric=metric, unit="dB",
                              goal=round(bands[band] + nudge, 1),
                              tolerance=TONE_INTENT_BAND_TOLERANCE_DB))
    if scoped:
        return targets   # no per-stem form for the space intents below
    if intent in ("wide", "narrow"):
        corr = analysis.get("stereo_correlation")
        if corr is not None and "stereo_correlation" not in claimed:
            goal = (max(WIDTH_INTENT_CORR_FLOOR,
                        round(corr - WIDTH_INTENT_CORR_STEP, 2))
                    if intent == "wide" else
                    min(WIDTH_INTENT_CORR_CEIL,
                        round(corr + WIDTH_INTENT_CORR_STEP, 2)))
            targets.append(Target(metric="stereo_correlation", unit="corr",
                                  goal=goal,
                                  tolerance=WIDTH_INTENT_CORR_TOLERANCE))
    if intent == "punchy":
        crest = analysis.get("crest_factor_db")
        if crest is not None and "crest_factor_db" not in claimed:
            targets.append(Target(
                metric="crest_factor_db", unit="dB",
                min_value=round(crest + PUNCH_CREST_FLOOR_STEP_DB, 1)))
    return targets


def interpret_brief(brief: str, stems: list[str],
                    stem_tags: Optional[dict[str, list[str]]] = None
                    ) -> list[str]:
    """Plain-English read-back of how the deterministic table reads a brief
    (Phase 39 D), surfaced at run start so the user sees the interpretation,
    not just metric names. Mirrors targets_from_brief's matching; best-effort
    (an LLM-compiled brief that matches nothing here falls back to the metric
    list in the loop)."""
    text = brief.lower()
    phrases: list[str] = []
    if (goal := _loudness_goal(text)) is not None:
        phrases.append(f"master to {goal:g} LUFS, peaks under -1 dBTP")
    if any(cue in text for cue in CLARITY_CUES):
        stem = resolve_stem(text, stems, stem_tags)
        phrases.append(f"{stem} up front and clear" if stem
                       else "the named stem up front and clear")
    if "harsh" in text:
        phrases.append("less harsh (2-5 kHz)")
    if "mud" in text or "boxy" in text:
        phrases.append("less mud / boxiness (low-mids)")
    if genre := _genre_of(text):
        phrases.append(f"{genre} tonal balance")
    try:
        phrases += [TONE_INTENT_PHRASES[intent] for intent in _tone_intents(text)]
    except BriefNotUnderstood:
        pass          # the read-back says nothing about a tone it could not read
    if specs := reference_specs_from_brief(brief):
        names = ", ".join(spec["path"].rsplit("/", 1)[-1] for spec in specs)
        phrases.append(f"match the reference ({names})")
    return phrases


# ---------- reference matching (Phase 7, docs/05) ----------

# Bands the reference barely holds can't be matching goals — same idea as the
# masking band floor, applied to the reference's relative band energy.
REFERENCE_BAND_FLOOR_DB = -30.0
REFERENCE_BAND_TOLERANCE_DB = 1.0
REFERENCE_LUFS_TOLERANCE = 1.0

# A path in quotes may hold spaces ("~/Music/My Refs/Song One.wav" - the reference window's Match
# names its reference that way, real use 2026-10-22); a bare one ends at the first space.
_AUDIO_EXT = r"\.(?:wav|mp3|flac|aiff?|ogg)"
_REFERENCE_RE = re.compile(
    r"(?:match|reference|like)\s+(?:'([^'\n]+" + _AUDIO_EXT + r")'|\"([^\"\n]+" + _AUDIO_EXT
    + r")\"|['\"]?([^'\"\s]+" + _AUDIO_EXT + r"))",
    re.IGNORECASE)
_REFERENCE_PATH_RE = re.compile(
    r"'([^'\n]+" + _AUDIO_EXT + r")'|\"([^\"\n]+" + _AUDIO_EXT + r")\"|['\"]?([^'\"\s,;]+"
    + _AUDIO_EXT + r")",
    re.IGNORECASE)
SECTION_BAND_METRIC_RE = re.compile(
    r"^section_band_energy\.(\d+)\.([^.]+)$")


def reference_path_from_brief(brief: str) -> Optional[str]:
    """'match reference.wav' / 'sit like refs/mix.mp3' -> the audio path
    (any import-boundary format, Phase 26 — analyze_reference reads them
    all directly)."""
    match = _REFERENCE_RE.search(brief)
    return next((g for g in match.groups() if g), None) if match else None


def reference_specs_from_brief(brief: str) -> list[dict]:
    """Phase 36: a reference brief may name several files. Weight/role syntax
    stays explicit through analyze_reference_set; free-form briefs get equal
    weights with all roles so old language keeps working."""
    if not re.search(r"\b(match|reference|like)\b", brief, re.IGNORECASE):
        return []
    specs = []
    seen = set()
    for match in _REFERENCE_PATH_RE.finditer(brief):
        path = next(g for g in match.groups() if g)
        if path in seen:
            continue
        seen.add(path)
        specs.append({"path": path, "weight": 1.0,
                      "roles": ["tonality", "width", "loudness"]})
    return specs


def reference_targets(reference: dict) -> list[Target]:
    """Targets that steer the mix-vs-reference *delta* toward zero. band_energy
    is relative to each render's own power (mix-analysis skill), so goal =
    the reference's per-band value, measured against the mix's own."""
    targets: list[Target] = []
    if reference.get("lufs_i") is not None:
        targets.append(Target(metric="lufs_i", unit="LUFS",
                              goal=reference["lufs_i"],
                              tolerance=REFERENCE_LUFS_TOLERANCE))
        targets.append(Target(metric="true_peak_dbtp", unit="dBTP", max_value=-1.0))
        # the punch guard rides along like any other loudness ask (2026-06)
        targets.append(Target(metric="plr_db", unit="dB",
                              min_value=PLR_FLOOR_DB))
    for band, value in (reference.get("band_energy") or {}).items():
        if value > REFERENCE_BAND_FLOOR_DB:
            targets.append(Target(metric=f"band_energy.{band}", unit="dB",
                                  goal=value,
                                  tolerance=REFERENCE_BAND_TOLERANCE_DB))
    if not targets:
        raise ValueError("reference analysis holds no usable targets "
                         "(silent reference?)")
    return targets


def _section_band_targets(reference: dict) -> list[Target]:
    targets: list[Target] = []
    for section in reference.get("section_targets") or []:
        mix_index = section["mix_index"]
        for band, value in (section.get("band_energy") or {}).items():
            if value > REFERENCE_BAND_FLOOR_DB:
                targets.append(Target(
                    metric=f"section_band_energy.{mix_index}.{band}",
                    unit="dB",
                    goal=value,
                    tolerance=REFERENCE_BAND_TOLERANCE_DB))
    return targets


def reference_set_targets(reference: dict) -> list[Target]:
    """Phase 36 full reference-set targets. Section-aware tonality wins when
    available; loudness still comes from the weighted loudness-role aggregate."""
    targets: list[Target] = []
    if reference.get("lufs_i") is not None:
        targets.append(Target(metric="lufs_i", unit="LUFS",
                              goal=reference["lufs_i"],
                              tolerance=REFERENCE_LUFS_TOLERANCE))
        targets.append(Target(metric="true_peak_dbtp", unit="dBTP", max_value=-1.0))
        targets.append(Target(metric="plr_db", unit="dB",
                              min_value=PLR_FLOOR_DB))
    targets += _section_band_targets(reference)
    if not any(t.metric.startswith("section_band_energy.") for t in targets):
        for band, value in (reference.get("band_energy") or {}).items():
            if value > REFERENCE_BAND_FLOOR_DB:
                targets.append(Target(metric=f"band_energy.{band}", unit="dB",
                                      goal=value,
                                      tolerance=REFERENCE_BAND_TOLERANCE_DB))
    if not targets:
        raise ValueError("reference set holds no usable targets "
                         "(silent references or empty roles?)")
    return targets


def reference_mix_targets(reference: dict) -> list[Target]:
    """Mix-scope reference targets (Phase 25): tonal character only. A mix
    sits nowhere near release loudness, so LUFS/true-peak/PLR are mastering
    concerns — they enter in the master stage, never here. Band goals are
    measured on the pre-master render (analyze_session(pre_master=true))."""
    targets = [Target(metric=f"band_energy.{band}", unit="dB", goal=value,
                      tolerance=REFERENCE_BAND_TOLERANCE_DB)
               for band, value in (reference.get("band_energy") or {}).items()
               if value > REFERENCE_BAND_FLOOR_DB]
    if not targets:
        raise ValueError("reference analysis holds no usable mix targets "
                         "(silent reference?)")
    return targets


def reference_set_mix_targets(reference: dict) -> list[Target]:
    """Mix-scope reference-set targets: tonal only, measured pre-master. Prefer
    paired section metrics; fall back to full-song aggregate tonal balance."""
    targets = _section_band_targets(reference)
    if not targets:
        targets = [Target(metric=f"band_energy.{band}", unit="dB", goal=value,
                          tolerance=REFERENCE_BAND_TOLERANCE_DB)
                   for band, value in (reference.get("band_energy") or {}).items()
                   if value > REFERENCE_BAND_FLOOR_DB]
    if not targets:
        raise ValueError("reference set holds no usable mix targets "
                         "(silent references or empty tonality roles?)")
    return targets


# ---------- LLM-compiled briefs (Phase 11) ----------

# Per-stem fields the loop can measure via analyze_stem (2026-07-30). Until this
# existed the compiler could only aim at SESSION-level numbers, so every
# per-instrument ask had to be flattened into a master-wide goal — the single
# biggest source of "it understood me and then did it to the whole mix".
STEM_TARGET_FIELDS = {
    "lufs_s", "peak_dbfs", "dynamic_spread_db", "transient_density",
    "transient_energy_ratio", "sibilance_density", "tuning_offset_cents",
    "tuning_coherence",
}

# What the pass loop can actually measure (EngineClient.measure_all): session
# scalars, relative band energies, masking pairs, and stem(<id>).<field>.
SESSION_TARGET_METRICS = {
    "lufs_i", "lufs_s_max", "lufs_range", "true_peak_dbtp",
    "stereo_correlation", "spectral_tilt_db_per_oct", "crest_factor_db",
    # Phase 14: composites + mastering QC are session scalars too
    "harshness_db", "mud_db", "clarity_index", "plr_db", "psr_db",
}
MAX_COMPILED_TARGETS = 6


def compiled_targets(raw: list[dict], stems: list[str],
                     warnings: Optional[list[str]] = None) -> list[Target]:
    """Validate compiler output into Targets (Phase 11). The LLM compiler is
    a convenience; this validation is the contract. One malformed entry is
    DROPPED (with a warning the caller can surface), never fatal — a single
    bad model emission used to raise here and kill the whole run
    (transient_energy_ratio with neither goal nor max, 2026-08-19; the
    Phase 50 bug class). Only a compile with ZERO usable targets raises."""
    targets: list[Target] = []
    for i, entry in enumerate(raw[:MAX_COMPILED_TARGETS]):
        metric = str(entry.get("metric", ""))
        goal, max_value = entry.get("goal"), entry.get("max_value")
        problem = None
        if not _metric_is_measurable(metric, stems):
            problem = f"unknown or unmeasurable metric {metric!r}"
        elif (goal is None) == (max_value is None):
            problem = "exactly one of goal / max_value is required"
        if problem is not None:
            if warnings is not None:
                warnings.append(
                    f"compiled target {i} ({metric}) dropped: {problem}")
            continue
        if metric == "lufs_i" and goal is not None:
            goal, note = clamp_loudness_goal(float(goal))    # the LLM's compile, same rail
            if note and warnings is not None:
                warnings.append(note)
        targets.append(Target(
            metric=metric,
            unit=str(entry.get("unit", "dB")),
            goal=None if goal is None else float(goal),
            tolerance=max(0.0, float(entry.get("tolerance", 0.5) or 0.0)),
            max_value=None if max_value is None else float(max_value)))
    if not targets:
        raise ValueError("brief compiler returned no usable targets")
    return targets


def _metric_is_measurable(metric: str, stems: list[str]) -> bool:
    if metric in SESSION_TARGET_METRICS:
        return True
    if metric.startswith("band_energy."):
        return metric.removeprefix("band_energy.") in BAND_LABELS
    if match := STEM_METRIC_RE.match(metric):
        stem_id, field = (g.strip() for g in match.groups())
        if stem_id not in stems:
            return False
        if field.startswith("band_energy."):
            return field.removeprefix("band_energy.") in BAND_LABELS
        return field in STEM_TARGET_FIELDS
    if match := MASKING_METRIC_RE.match(metric):
        stem_a, stem_b, field = (g.strip() for g in match.groups())
        return (stem_a in stems and stem_b in stems and stem_a != stem_b
                and (field == "worst_band_excess_db" or field in BAND_LABELS))
    return False


# The instrument vocabulary now lives in the ENGINE (mix_engine.instruments) —
# `analysis.classify_stem` reads a stem's NAME through the same table before it
# reads its acoustics, and the engine may not import the orchestrator. Imported,
# never copied, so the two halves can't drift (docs/02-architecture.md's rule for
# every contract object). `_roles_of_stem` keeps its private name here: it is the
# spelling this module, `arrangement.py` and the role-vocabulary tests use.
_roles_of_stem = roles_of_stem


def _mentions(text: str, words) -> bool:
    """Whole-word match for any of `words` in `text` — so 'hat' never fires on
    'what'. Used by the Phase 39 tag and tone-intent matching."""
    return any(re.search(rf"\b{re.escape(w)}\b", text) for w in words)


def _roles_named(text: str) -> set[str]:
    """Canonical instrument roles the text names ('pull the gtr back' ->
    {'guitar'}). Words in `_TEXT_AMBIGUOUS` never match here."""
    return {role for role, words in INSTRUMENT_WORDS.items()
            if _mentions(text, tuple(w for w in words
                                     if w not in _TEXT_AMBIGUOUS))}


def _stem_by_name(text: str, stems) -> Optional[str]:
    """First stem (sorted) whose own name shares a role with the brief."""
    wanted = _roles_named(text)
    if not wanted:
        return None
    for stem_id in sorted(stems):
        if wanted & _roles_of_stem(stem_id):
            return stem_id
    return None


def _stem_for_tag(text: str, stems,
                  stem_tags: Optional[dict[str, list[str]]]) -> Optional[str]:
    """First stem (sorted) whose tag the brief names by role (Phase 39)."""
    if not stem_tags:
        return None
    wanted = {tag for tag, words in TAG_SYNONYMS.items()
              if _mentions(text, tuple(w for w in words
                                       if w not in _TEXT_AMBIGUOUS))}
    if not wanted:
        return None
    for stem_id in sorted(stems):
        if wanted & set(stem_tags.get(stem_id) or []):
            return stem_id
    return None


# PUBLIC (2026-08-05): callers resolve the named stem BEFORE compiling, so they can
# fetch that stem's analysis and hand it to `targets_from_brief` (which stays pure).
# One resolver for the compiler, the callers and the understanding eval — a second
# one would drift from the eval's definition of "the brief named a stem".
def resolve_stem(text: str, stems,
                  stem_tags: Optional[dict[str, list[str]]] = None
                  ) -> Optional[str]:
    """Match 'stem A' / 'stem_a' / a role word ('vocals', 'kick', 'snare') to a
    stem id, in descending order of evidence quality:

    1. a literal stem id in the text — unambiguous, always wins;
    2. the stem's OWN name ("snare_top" for "the snare has no crack") — the
       user typed it, so it beats a classifier guess;
    3. the engine's auto-tags, for genuinely opaque ids ("track_03").

    Step 2 is 2026-07-30: without it, a session whose tracks are named after
    their instruments could not resolve any instrument the six-tag classifier
    does not emit — snare, guitar, piano, strings, horns and the rest."""
    # Lowercase HERE, not at the caller: [^a-z0-9] treats an uppercase letter as a
    # separator, so "make stem A louder" normalized to "make_stem__louder" and the
    # docstring's own example stopped resolving.
    normalized = re.sub(r"[^a-z0-9]+", "_", text.lower())
    for stem_id in stems:
        # WHOLE-word, not substring (2026-08-05). A bare `in` test let a stem id
        # called "a" match the "a" inside "make it less harsh", so a session with
        # short ids scoped every composite ask to a random stem. Both ends are
        # normalized the same way, so a multi-word id ("snare_top") still matches
        # "the snare top is dull".
        token = re.sub(r"[^a-z0-9]+", "_", stem_id.lower()).strip("_")
        # `s?` tolerates the plural the user actually types ("the pianos are muddy",
        # "more kicks") without reopening the substring hole — the boundary is still
        # required on both ends.
        if token and re.search(rf"(?:^|_){re.escape(token)}s?(?:_|$)", normalized):
            return stem_id
    return (_stem_by_name(text, stems)
            or _stem_for_tag(text, stems, stem_tags))
