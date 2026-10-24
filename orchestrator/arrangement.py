"""Natural-language song arrangement planning for the native DAW.

Arrangement requests are edit operations, not mix targets.  They therefore bypass the
ordinary brief -> measurable-metric compiler and resolve to one atomic
``arrangement.build`` DAW command.  The native command owns mutation + undo; this module
only turns the user's intent and the live ``daw_edit_state`` snapshot into a validated,
compact plan.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


class ArrangementError(ValueError):
    """The request is an arrangement ask, but the live edit cannot satisfy it."""


_STRUCTURE_PHRASES = (
    "structure this song",
    "structure the song",
    "structure these stems",
    "build a song from",
    "build the song from",
)
_QUESTION_PREFIXES = ("what is", "what's", "how does", "how do", "can you explain")


def is_arrangement_brief(brief: str) -> bool:
    """Return whether ``brief`` explicitly asks to build/rebuild song structure.

    Questions *about* arrangement remain conversational; imperative arrangement asks go
    through the edit command even when no LLM backend is configured.
    """
    text = " ".join(str(brief).lower().replace("-", " ").split())
    if not text or any(text.startswith(prefix) for prefix in _QUESTION_PREFIXES):
        return False
    if any(phrase in text for phrase in _STRUCTURE_PHRASES):
        return True
    words = set(text.replace("?", "").replace(",", "").split())
    if words.intersection({"arrange", "rearrange"}):
        return True
    request_verbs = {"build", "create", "design", "generate", "make", "plan"}
    return bool(words.intersection(request_verbs)) and (
        "arrangement" in words or "layout" in words or "lay out" in text
    )


def _session_tags(session_state: Mapping[str, Any] | None) -> dict[str, list[str]]:
    stems = (session_state or {}).get("stems") or {}
    return {
        str(stem): [str(tag).lower() for tag in (data.get("tags") or [])]
        for stem, data in stems.items()
        if isinstance(data, Mapping)
    }


# The arrangement taxonomy is deliberately coarser than the instrument
# vocabulary — a layout needs to know "is this the beat, the low end, the hook,
# or everything else", not which drum. Derived from the shared vocabulary
# (2026-07-30) so a new instrument word only has to be added in one place.
_ROLE_GROUPS: dict[str, frozenset[str]] = {
    "drums": frozenset({"kick", "snare", "clap", "hat", "cymbal", "tom",
                        "room", "drums"}),
    "bass": frozenset({"bass"}),
    "lead": frozenset({"vocal", "backing_vocal"}),
}
# Words that mark a track as the hook without naming an instrument. "lead" is
# excluded from the instrument vocabulary (lead vocal or lead synth?) but in a
# track NAME it is unambiguous enough for a layout decision.
_LEAD_NAME_WORDS = ("lead", "melody", "topline", "hook")


def _track_role(name: str, tags: list[str]) -> str:
    from .targets import _roles_of_stem

    words = " ".join([name.lower(), *[t.lower() for t in tags]])
    roles = _roles_of_stem(words)
    for group, members in _ROLE_GROUPS.items():
        if roles & members:
            return group
    if any(k in words for k in _LEAD_NAME_WORDS):
        return "lead"
    return "music"


def _arrangeable_tracks(edit_state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The tracks ``arrangement.build`` arranges: a regular track with audio clips (its file is
    its source), or one with MIDI clips and no audio clip (its PART as it plays is its source -
    Phase 61; until then a song the agent had composed could not be arranged at all). A muted
    clip plays nothing and is not a source (the build reads the clips that play, Q10): a track
    whose clips are all muted - a split's source, muted when its stems land - is left out, as
    the app refuses a plan that names it (the pre-merge review of 2026-10-23)."""
    tracks: list[dict[str, Any]] = []
    for raw in edit_state.get("tracks") or []:
        if not isinstance(raw, Mapping):
            continue
        name = str(raw.get("name") or "")
        kind = str(raw.get("kind") or "stem")
        if (not name or name == "master" or name.startswith(("bus:", "return:", "vca:"))
                or kind in {"bus", "return", "master", "vca"}):
            continue
        clips = [c for c in (raw.get("clips") or []) if isinstance(c, Mapping)]
        audio = [c for c in clips if c.get("source") and not c.get("muted")
                 and float(c.get("length_s") or 0.0) > 0.05]
        midi = [c for c in clips if c.get("kind") == "midi" and not c.get("muted")
                and float(c.get("length_s") or 0.0) > 0.05]
        if audio:
            tracks.append({"name": name, "clips": audio, "midi": False})
        elif midi:
            tracks.append({"name": name, "clips": midi, "midi": True})
    return tracks


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def build_arrangement_plan(
    brief: str,
    edit_state: Mapping[str, Any] | None,
    session_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an atomic DAW arrangement plan from the live edit snapshot.

    The first cut deliberately uses a deterministic producer template.  It keeps aligned
    stems aligned, advances through long-form source material in order, and naturally loops
    short clips when the native builder reaches their end - an audio file, or a MIDI track's
    part (Phase 61: a composed four-bar loop repeats through every section it plays in).  Four-bar source phrases yield a
    conventional 52-bar lo-fi form without guessing at sample arrays or touching the audio
    thread.
    """
    if not is_arrangement_brief(brief):
        raise ArrangementError("not an arrangement request")
    if not edit_state:
        raise ArrangementError("the DAW has not supplied its live edit state yet")

    tracks = _arrangeable_tracks(edit_state)
    if not tracks:
        raise ArrangementError("there are no audio or MIDI clips on regular tracks to arrange")

    tempo = edit_state.get("tempo") or {}
    try:
        bpm = float(tempo.get("bpm") or 120.0)
    except (TypeError, ValueError):
        bpm = 120.0
    bpm = min(300.0, max(20.0, bpm))

    numerator, denominator = 4, 4
    time_sigs = tempo.get("time_sigs") or []
    if time_sigs and isinstance(time_sigs[0], Mapping):
        try:
            numerator = min(16, max(1, int(time_sigs[0].get("numerator") or 4)))
            denominator = min(32, max(1, int(time_sigs[0].get("denominator") or 4)))
        except (TypeError, ValueError):
            numerator, denominator = 4, 4

    phrase_bars = 4
    phrase_s = 60.0 / bpm * numerator * (4.0 / denominator) * phrase_bars
    names = [t["name"] for t in tracks]
    tags = _session_tags(session_state)
    roles = {name: _track_role(name, tags.get(name, [])) for name in names}

    drums = [n for n in names if roles[n] == "drums"]
    bass = [n for n in names if roles[n] == "bass"]
    leads = [n for n in names if roles[n] == "lead"]
    music = [n for n in names if roles[n] == "music"]
    tonal = music or leads or bass or names[:1]

    # Sparse bookends + breakdown, full hooks, and a slightly reduced second verse when
    # multiple texture tracks are available.  With opaque track names the stable input order
    # still produces a valid, deterministic arrangement.
    intro = _unique(tonal[:2])
    verse = _unique(drums + bass + leads + music)
    verse_alt = list(verse)
    if len(music) > 1 and music[-1] in verse_alt:
        verse_alt.remove(music[-1])
    bridge = _unique(leads + music) or intro
    outro = _unique(music[:2] or leads[:1] or tonal[:1])
    full = list(names)

    normalized = " ".join(str(brief).lower().replace("-", " ").split())
    lofi = any(cue in normalized for cue in ("lofi", "lo fi", "chillhop", "chill hop"))
    if lofi:
        form = [
            ("Intro", 1, intro),
            ("Verse 1", 2, verse),
            ("Hook", 2, full),
            ("Verse 2", 2, verse_alt),
            ("Hook 2", 2, full),
            ("Bridge", 1, bridge),
            ("Final Hook", 2, full),
            ("Outro", 1, outro),
        ]
        style = "lo-fi"
    else:
        form = [
            ("Intro", 1, intro),
            ("Verse 1", 2, verse),
            ("Pre-Chorus", 1, bridge),
            ("Chorus", 2, full),
            ("Verse 2", 2, verse_alt),
            ("Chorus 2", 2, full),
            ("Bridge", 2, bridge),
            ("Final Chorus", 2, full),
            ("Outro", 1, outro),
        ]
        style = "song"

    sections: list[dict[str, Any]] = []
    source_cursor_s = 0.0
    total_bars = 0
    for name, phrases, active in form:
        length_s = phrase_s * phrases
        sections.append({
            "name": name,
            "length_s": round(length_s, 6),
            "source_start_s": round(source_cursor_s, 6),
            "tracks": active or names[:1],
        })
        source_cursor_s += length_s
        total_bars += phrase_bars * phrases

    return {
        "replace": True,
        "style": style,
        "bpm": round(bpm, 3),
        "total_bars": total_bars,
        "sections": sections,
    }


def arrangement_summary(plan: Mapping[str, Any], result: Mapping[str, Any] | None = None,
                        *, preview: bool = False) -> str:
    sections = plan.get("sections") or []
    names = [str(s.get("name")) for s in sections if isinstance(s, Mapping)]
    prefix = "I’d build" if preview else "Built"
    detail = (f"{len(names)} labeled sections — {' → '.join(names)} — "
              f"{int(plan.get('total_bars') or 0)} bars at "
              f"{float(plan.get('bpm') or 120.0):g} BPM")
    if result and result.get("clips") is not None:
        midi = int(result.get("midi_clips") or 0)
        audio = int(result["clips"]) - midi
        placed = [f"{n} {what} clip{'s' if n != 1 else ''}"
                  for n, what in ((audio, "audio"), (midi, "MIDI")) if n]
        detail += f", with {' and '.join(placed) or 'no clips'} placed"
    left = [e for e in ((result or {}).get("left_out") or []) if isinstance(e, Mapping)]
    if left and not preview:
        detail += ("; notes no section plays were cleared from "
                   + ", ".join(f"{e.get('track')} ({int(e.get('notes') or 0)})" for e in left))
    suffix = ". One Undo reverts the whole arrangement." if not preview else "."
    return f"{prefix} {detail}{suffix}"


def apply_arrangement(
    brief: str,
    edit_state: Mapping[str, Any] | None,
    daw_invoke: Callable[[str, dict[str, Any]], Mapping[str, Any]] | None,
    session_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Plan + apply one arrangement through the DAW's atomic command seam."""
    if daw_invoke is None:
        raise ArrangementError("the live DAW command channel is not connected")
    plan = build_arrangement_plan(brief, edit_state, session_state)
    result = dict(daw_invoke("arrangement.build", plan) or {})
    if not result.get("ok"):
        raise ArrangementError(str(result.get("error") or "the DAW rejected the arrangement"))
    return {"plan": plan, "result": result,
            "message": arrangement_summary(plan, result)}
