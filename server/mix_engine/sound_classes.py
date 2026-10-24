"""What a KIND of sound usually measures as (Phase 60).

`sample_profile` says what one sound is. This says where that reading sits against the kind it
is meant to be - a kick, a closed hat, a pad - as ISSUES: plain sentences, each naming the
figure, what it reads and what the kind usually reads. They are a guide for somebody designing
a sound by measurement, never a verdict: a kick that is deliberately a 300 Hz knock is still a
kick, and the caller is told so (`NOT_A_VERDICT`).

The kinds are the sample packs' (`samplepacks::kinds()`), so a sound kept in a pack is read
against the kind it is filed under. The ranges were sized on three things (the table is in
`.claude/skills/mix-analysis/SKILL.md` ▸ *One sound, against its kind*):

  * the stock drum kit's fifteen one-shots - each must read clean against its own kind;
  * 59 recorded hits cut from a labelled pack of solo kick, snare and hat loops (dark, lo-fi
    material: the other end of the range from the stock kit's bright synthesis);
  * WOOLLY renders: the factory bank, and INIT - one saw - which must NOT read clean against
    any drum, bass, pluck or pad kind.

Pure: a profile dict in, a list out. Change a range and `SOUND_CLASSES_VERSION` moves.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

SOUND_CLASSES_VERSION = 2          # 2: the kind "808" is "boom" (the old word is still taken)

NOT_A_VERDICT = ("where this sound sits outside what its kind usually measures as - a guide, "
                 "not a verdict: a deliberate choice may ignore one")

# band_share_pct's labels, low to high: sub 20-60, bass 60-150, low_mid 150-400, mid 400-1k,
# high_mid 1-2k, presence 2-4k, brilliance 4-8k, air 8-20k
_DERIVED = {
    "sub_bass_pct": ("sub", "bass"),                       # under 150 Hz
    "low_pct": ("sub", "bass", "low_mid"),                 # under 400 Hz
    "high_pct": ("presence", "brilliance", "air"),         # over 2 kHz
    "top_pct": ("brilliance", "air"),                      # over 4 kHz
}

_WORDS = {
    "attack_ms": ("attack", "ms"), "decay_ms": ("decay (the time to fall 20 dB)", "ms"),
    "sounding_s": ("length", "s"), "centroid_hz": ("centroid", "Hz"),
    "pitch_hz": ("pitch", "Hz"), "noisiness": ("noisiness", ""),
    "pitch_drop_semitones": ("pitch drop", "st"), "pitch_settle_ms": ("pitch settle time", "ms"),
    "sub_bass_pct": ("share of its energy under 150 Hz", "%"),
    "low_pct": ("share of its energy under 400 Hz", "%"),
    "high_pct": ("share of its energy over 2 kHz", "%"),
    "top_pct": ("share of its energy over 4 kHz", "%"),
    "end_level_dbfs": ("level at the end of the file", "dBFS"),
    "width_db": ("side level against the mid", "dB"),
    "crest_db": ("peak over its average level", "dB"),
}


@dataclass(frozen=True)
class Want:
    """One figure of a kind: `field` reads `lo`..`hi` (either may be open). `absent` says what a
    figure that could not be read means: "over" (it never happened: a decay that was never
    reached is longer than any limit), "issue" (the kind needs it: a bass with no pitch), or
    "skip"."""
    field: str
    lo: Optional[float] = None
    hi: Optional[float] = None
    why: str = ""
    absent: str = "skip"


def _w(field: str, lo: Optional[float], hi: Optional[float], why: str, absent: str = "skip") -> Want:
    return Want(field, lo, hi, why, absent)


# ---- what every drum hit shares
_HIT = (_w("attack_ms", None, 25.0, "a drum hit starts at once"),)

CLASSES: dict[str, tuple[Want, ...]] = {
    "kick": _HIT + (
        _w("sub_bass_pct", 45.0, None, "a kick's weight is under 150 Hz"),
        _w("centroid_hz", None, 400.0, "a kick is a low sound, with at most a click on top"),
        _w("pitch_hz", 30.0, 130.0, "a kick's body settles on a low note", "skip"),
        _w("decay_ms", 30.0, 900.0, "a kick is a hit, not a held note", "over"),
        _w("noisiness", None, 0.92, "a kick has a tone under its click"),
    ),
    "snare": _HIT + (
        _w("noisiness", 0.15, None, "a snare has noise in it (the wires)"),
        _w("centroid_hz", 400.0, 11000.0, "a snare sits in the mids and highs"),
        _w("sub_bass_pct", None, 30.0, "a snare leaves the bottom to the kick"),
        _w("decay_ms", 15.0, 500.0, "a snare is short", "over"),
    ),
    "clap": _HIT + (
        _w("noisiness", 0.45, None, "a clap is noise"),
        _w("centroid_hz", 700.0, 10000.0, "a clap sits in the upper mids"),
        _w("low_pct", None, 25.0, "a clap has no low end"),
        _w("decay_ms", 30.0, 600.0, "a clap is short", "over"),
    ),
    "rim": _HIT + (
        _w("decay_ms", None, 150.0, "a rim shot is a tick", "over"),
        _w("centroid_hz", 350.0, 7000.0, "a rim shot is a mid-range knock"),
        _w("sub_bass_pct", None, 20.0, "a rim shot has no low end"),
    ),
    "hat": _HIT + (
        _w("noisiness", 0.35, None, "a hat is noise or a dense metallic cluster, not a note"),
        _w("centroid_hz", 1000.0, None, "a hat is a high sound"),
        _w("low_pct", None, 15.0, "a hat has nothing under 400 Hz"),
        _w("decay_ms", None, 160.0, "a closed hat is very short", "over"),
    ),
    "open_hat": _HIT + (
        _w("noisiness", 0.35, None, "a hat is noise or a dense metallic cluster, not a note"),
        _w("centroid_hz", 1000.0, None, "a hat is a high sound"),
        _w("low_pct", None, 15.0, "a hat has nothing under 400 Hz"),
        _w("decay_ms", 160.0, 1500.0, "an open hat rings longer than a closed one, and still ends", "over"),
    ),
    "tom": _HIT + (
        _w("pitch_hz", 55.0, 450.0, "a tom is a pitched drum", "issue"),
        _w("noisiness", None, 0.6, "a tom is mostly tone"),
        _w("decay_ms", 80.0, 1000.0, "a tom rings briefly", "over"),
    ),
    "crash": (
        _w("attack_ms", None, 40.0, "a crash starts at once"),
        _w("noisiness", 0.45, None, "a cymbal is noise or a dense metallic cluster"),
        _w("centroid_hz", 2500.0, None, "a cymbal is a high sound"),
        _w("decay_ms", 500.0, None, "a crash rings on"),
    ),
    "ride": (
        _w("attack_ms", None, 40.0, "a ride starts at once"),
        _w("centroid_hz", 2000.0, None, "a cymbal is a high sound"),
        _w("low_pct", None, 20.0, "a cymbal has nothing under 400 Hz"),
        _w("decay_ms", 250.0, None, "a ride rings on"),
    ),
    "perc": _HIT + (
        _w("decay_ms", None, 1000.0, "a percussion hit ends", "over"),
    ),
    "bass": (
        _w("pitch_hz", 28.0, 270.0, "a bass plays low notes: audition it on one", "issue"),
        _w("low_pct", 40.0, None, "a bass carries its weight under 400 Hz"),
        _w("attack_ms", None, 80.0, "a bass note speaks quickly"),
    ),
    "sub": (
        _w("pitch_hz", 28.0, 110.0, "a sub plays very low notes: audition it on one", "issue"),
        # (a plain saw on a low C reads 76 % under 150 Hz with a 241 Hz centroid: a sub is not that)
        _w("sub_bass_pct", 90.0, None, "a sub is almost all under 150 Hz"),
        _w("centroid_hz", None, 150.0, "a sub has little above its fundamental"),
        _w("noisiness", None, 0.35, "a sub is a clean tone"),
    ),
    "boom": (
        _w("pitch_hz", 28.0, 120.0, "a boom bass is pitched and low", "issue"),
        _w("sub_bass_pct", 80.0, None, "a boom bass's weight is under 150 Hz"),
        _w("centroid_hz", None, 200.0, "a boom bass is close to a sine: a little drive, not a bright wave"),
        _w("attack_ms", None, 25.0, "a boom bass hits at once"),
        _w("decay_ms", 250.0, None, "a boom bass rings on after the hit"),
    ),
    "lead": (
        _w("pitch_hz", 100.0, None, "a lead plays a clear note", "issue"),
        _w("attack_ms", None, 120.0, "a lead speaks quickly"),
        _w("centroid_hz", 350.0, None, "a lead is bright enough to sit on top"),
        # (not decay_ms: a held sound's loudest sample can be anywhere in the hold, so its
        #  "decay" is the time from an arbitrary moment to the note's end - a bare pulse read
        #  21 ms on one render and 210 ms on the next)
        _w("crest_db", None, 16.0, "a lead holds its level while the key is down"),
    ),
    "pad": (
        _w("attack_ms", 30.0, None, "a pad fades in"),
        _w("sounding_s", 1.0, None, "a pad is a long sound: audition it on a long note"),
        _w("decay_ms", 150.0, None, "a pad lets go slowly"),
    ),
    "pluck": (
        _w("pitch_hz", 60.0, None, "a pluck plays a clear note", "issue"),
        _w("attack_ms", None, 20.0, "a pluck starts at once"),
        _w("decay_ms", 20.0, 900.0, "a pluck dies away by itself", "over"),
        _w("crest_db", 12.0, None, "a pluck is a peak that dies away, not a held note"),
    ),
    "keys": (
        _w("pitch_hz", 60.0, None, "a key sound plays a clear note", "issue"),
        _w("attack_ms", None, 40.0, "a key sound speaks at once"),
        _w("sounding_s", 0.2, None, "a key sound rings for a moment at least"),
    ),
    "chord": (
        _w("sounding_s", 0.25, None, "a chord shot is long enough to hear its notes"),
    ),
    "riser": (
        _w("attack_ms", 300.0, None, "a riser builds: its level keeps growing"),
        _w("sounding_s", 1.0, None, "a riser takes its time"),
    ),
    "impact": (
        _w("attack_ms", None, 30.0, "an impact hits at once"),
        _w("decay_ms", 250.0, None, "an impact rings on after the hit"),
    ),
    "fx": (),
    "drums": (),                # a whole beat: a loop, read by its length and seam
}

#: The kinds whose sound should be the note that was played (an octave either way is a choice).
IN_TUNE_KINDS = frozenset({"bass", "sub", "boom", "lead", "pluck", "keys"})
#: What people call a kind that the kinds table files under its own name.
ALIASES = {"808": "boom"}
IN_TUNE_CENTS = 70.0

END_LEVEL_DBFS = -30.0           # a file that ends louder than this is cut while it sounds
DC_OFFSET_PCT = 2.0
LEADING_SILENCE_MS = 30.0


def kinds() -> tuple[str, ...]:
    return tuple(CLASSES)


def canonical(kind: str) -> str:
    """A kind as the table names it: lower case, and a word people use for one (`808`) as
    the kind it is."""
    key = str(kind).strip().lower()
    return ALIASES.get(key, key)


def figure(profile: dict, field: str) -> Optional[float]:
    """A profile's figure by name, the derived band sums included."""
    if field in _DERIVED:
        shares = profile.get("band_share_pct")
        if not isinstance(shares, dict):
            return None
        return round(float(sum(float(shares.get(b, 0.0)) for b in _DERIVED[field])), 1)
    value = profile.get(field)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _num(value: float, unit: str) -> str:
    text = f"{value:.0f}" if abs(value) >= 100 or float(value).is_integer() else f"{value:.1f}"
    if unit == "":
        text = f"{value:.2f}"
    return f"{text} {unit}".strip() if unit != "%" else f"{text} %"


def _range_words(lo: Optional[float], hi: Optional[float], unit: str) -> str:
    if lo is not None and hi is not None:
        return f"{_num(lo, unit)} to {_num(hi, unit)}"
    if lo is not None:
        return f"{_num(lo, unit)} or more"
    return f"{_num(hi, unit)} or less"       # type: ignore[arg-type]


def _issue(field: str, value: Optional[float], want: Want, says: str) -> dict:
    out: dict = {"field": field, "says": says}
    if value is not None:
        out["value"] = value
    if want.lo is not None:
        out["min"] = want.lo
    if want.hi is not None:
        out["max"] = want.hi
    return out


def issues(profile: dict, kind: str, played_midi: Optional[int] = None) -> list[dict]:
    """Where `profile` (a `SampleProfile` as a dict) sits outside what `kind` usually measures
    as. `played_midi` is the one note the sound was auditioned on, when there was one. An
    unknown kind raises: the caller names kinds from `kinds()`."""
    key = canonical(kind)
    if key not in CLASSES:
        raise ValueError(f"no sound kind '{kind}' (kinds: {', '.join(CLASSES)})")
    if profile.get("silent"):
        return [{"field": "silent", "says": "the file is silent"}]
    out: list[dict] = []
    is_loop = profile.get("kind", "one_shot") == "loop"
    if is_loop:
        out.extend(_loop_issues(profile))
    # (a loop is several notes: a kind's one-note figures - its attack, its pitch - are not its)
    for want in () if is_loop else CLASSES[key]:
        name, unit = _WORDS.get(want.field, (want.field, ""))
        value = figure(profile, want.field)
        usual = _range_words(want.lo, want.hi, unit)
        if value is None:
            if want.absent == "issue":
                out.append(_issue(want.field, None, want,
                                  f"{want.why}: no clear {name} could be read (usually {usual})"))
            elif want.absent == "over" and want.hi is not None:
                out.append(_issue(want.field, None, want,
                                  f"{want.why}: its {name} was never reached in the file "
                                  f"(usually {usual})"))
            continue
        if (want.lo is not None and value < want.lo) or (want.hi is not None and value > want.hi):
            out.append(_issue(want.field, value, want,
                              f"{want.why}: its {name} is {_num(value, unit)} (usually {usual})"))

    # ---- what any kept sound should be, whatever its kind
    if profile.get("over_full_scale") or (profile.get("full_scale_samples") or 0) > 0:
        out.append({"field": "full_scale_samples", "says": "it reaches full scale: it is clipped, or will be when it is written to a file - lower the patch's level"})
    dc = profile.get("dc_offset_pct")
    if isinstance(dc, (int, float)) and dc > DC_OFFSET_PCT:
        out.append({"field": "dc_offset_pct", "value": float(dc), "max": DC_OFFSET_PCT,
                    "says": f"it sits {dc:.1f} % off the zero line: it will thump when it starts and stops"})
    if not is_loop and key != "drums":
        end = profile.get("end_level_dbfs")
        if isinstance(end, (int, float)) and end > END_LEVEL_DBFS:
            out.append({"field": "end_level_dbfs", "value": float(end), "max": END_LEVEL_DBFS,
                        "says": f"the file ends at {end:.0f} dBFS, while the sound is still going: it "
                                "will click where it is cut - shorten the release or give it a longer tail"})
        if key in IN_TUNE_KINDS and played_midi is not None:
            midi, cents = profile.get("pitch_midi"), profile.get("pitch_cents")
            if isinstance(midi, (int, float)) and isinstance(cents, (int, float)):
                off = (float(midi) + float(cents) / 100.0) - float(played_midi)
                from_octave = abs(off - 12.0 * round(off / 12.0)) * 100.0
                if from_octave > IN_TUNE_CENTS:
                    out.append({"field": "pitch_midi", "value": float(midi), "played": int(played_midi),
                                "says": f"played on note {int(played_midi)} it sounds at note "
                                        f"{float(midi) + float(cents) / 100.0:.2f}: it is "
                                        f"{from_octave:.0f} cents from that note in any octave, so it "
                                        "will be out of tune in a song"})
    return out


SEAM_EXCESS_DB = 0.0             # the step from a loop's last sample to its first, against its own steps


def _loop_issues(profile: dict) -> list[dict]:
    loop = profile.get("loop")
    if not isinstance(loop, dict):
        return []
    out = []
    off = loop.get("length_error_samples")
    if isinstance(off, (int, float)) and int(off) != 0:
        out.append({"field": "loop.length_error_samples", "value": float(off),
                    "says": f"the loop is {int(off)} samples off its {loop.get('beats')} beats at "
                            f"{loop.get('bpm')} BPM: it will drift against the grid"})
    seam = loop.get("seam_excess_db")
    if isinstance(seam, (int, float)) and seam > SEAM_EXCESS_DB:
        out.append({"field": "loop.seam_excess_db", "value": float(seam), "max": SEAM_EXCESS_DB,
                    "says": f"the loop's end meets its start with a step {seam:.1f} dB larger than "
                            "any step inside it: it will click each time round"})
    return out


def reading(profile: dict, kind: str, played_midi: Optional[int] = None) -> dict:
    """What a tool result carries beside a profile: the kind it was read against and its
    issues - an empty list is said, so "nothing unusual" is not confused with "not read"."""
    found = issues(profile, kind, played_midi)
    out: dict = {"kind": canonical(kind), "issues": [i["says"] for i in found]}
    if found:
        out["issues_are"] = NOT_A_VERDICT
    return out
