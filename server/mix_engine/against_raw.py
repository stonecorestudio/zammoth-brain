"""What the engineer's processing did, against the audio it started from (2026-11-02).

The site's before-and-after samples were played at matched loudness, and listeners preferred
the befores: the "raw" parts were sample-pack files that were already compressed and bright, and
the engineer stacked three compressors, saturation, +2.5 dB at 3.5 kHz and +4 dB of air on a
vocal that needed none of it. Nothing it read said the chain had left the track brighter and
flatter than the file it was given: every reading was of the track as it now stood.

This reads a track (or the whole master) against its own raw self: the stem's file with no chain
and no rides, and the master against every track summed raw. Every figure here is of a SHAPE, not a
level (energy shares, level spread, crest), so it is the comparison at matched loudness whatever
the faders did. A note is written only where the change is large enough to hear; a track the chain
left close to its file says nothing.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from . import analysis, dsp
from .session import MASTER, Session

#: the bands a share is summed over (BAND_LABELS' order: sub, bass, low_mid, mid, high_mid,
#: presence, brilliance, air)
TOP_BANDS = ("presence", "brilliance", "air")         # over 2 kHz
LOW_BANDS = ("sub", "bass", "low_mid")                # under 400 Hz

#: how far a share has to move before a track is called brighter, darker, fuller or thinner than
#: its file. 3 dB is twice the 1.5 dB just-noticeable shift of a broad tilt.
STEM_SHARE_DB = 3.0
#: the master is held a little tighter: a broad shift of the whole mix is heard sooner, and the
#: site's house master lost 3 dB over 2 kHz and its EDM master 2.4 dB under 400 Hz
MASTER_SHARE_DB = 2.5
#: a share this small on both sides is not a tone the track has: a high-pass taking a hat's
#: lows from 0.01 % to 0.001 % is a 10 dB change nobody hears (the site's hip hop session)
SHARE_FLOOR_DB = -17.0           # 2 % of the energy
#: the level spread (p95 - p10 of the 400 ms frames, `analysis._dynamic_spread_db`) lost before
#: a track is called flattened
SPREAD_LOST_DB = 3.0
#: the crest (peak over RMS) lost before a track's transients are called flattened
CREST_LOST_DB = 3.0
#: the gain reduction a track's compressors take TOGETHER, on average, before it is called
#: heavy - the same 6 dB `analysis.OVER_COMPRESSION_GR_MEAN_DB` holds one compressor to
STACKED_GR_DB = analysis.OVER_COMPRESSION_GR_MEAN_DB
#: the processors whose reduction is levelling (a de-esser, a gate, a duck and a limiter are not)
LEVELLING = analysis.STACKED_LEVELLING


def _share_db(bands: dict[str, float], which: tuple[str, ...]) -> float:
    """The energy share of a group of bands, in dB of the whole (bands are dB of the whole)."""
    power = sum(10.0 ** (bands[b] / 10.0) for b in which)
    return 10.0 * np.log10(max(power, 1e-12))


def shape(audio: np.ndarray, sample_rate: int) -> Optional[dict]:
    """The level-free readings of a piece of audio: its top and low shares (dB of the whole),
    its level spread and its crest. None for silence."""
    audio = analysis._frames_channels(audio)
    if np.max(np.abs(audio)) < analysis.SILENCE_PEAK:
        return None
    bands = analysis._band_energy(audio, sample_rate)
    spread = analysis._dynamic_spread_db(audio, sample_rate)
    return {"top_db": round(float(_share_db(bands, TOP_BANDS)), 1),
            "low_db": round(float(_share_db(bands, LOW_BANDS)), 1),
            "spread_db": None if spread is None else round(float(spread), 1),
            "crest_db": round(float(analysis._crest_factor_db(audio)), 1)}


def stacked_gr_db(telemetry: list[dict]) -> tuple[float, int]:
    """(the mean reduction a chain's levelling processors take together, how many of them did
    any): processors in SERIES add their reduction, so three compressors at 3 dB each are 9 dB
    of levelling, and each one alone looked gentle."""
    entries = [e for e in telemetry or [] if e.get("processor") in LEVELLING]
    return round(sum(float(e["gr_mean_db"]) for e in entries), 1), len(entries)


def _trim(entry: dict) -> Optional[dict]:
    """What of a chain entry is a level and nothing else: a utility's gain, the rest of it
    (width, mono, polarity) at rest. None for any other device."""
    if entry.get("type") != "utility" or not entry.get("gain_db"):
        return None
    return {"type": "utility", "gain_db": entry["gain_db"], "width": 1.0, "mono": 0.0,
            "invert": 0.0, "bass_mono": 0.0, "uid": entry.get("uid")}


def raw_copy(session: Session, stems: Optional[list[str]] = None) -> Session:
    """A scratch copy with the given stems (all of them by default, and then the master too) as
    their files AT THEIR LEVELS: no device, no ride, the width at rest, the fader, the pan and a
    utility's gain trim kept. A level is balance, not processing (2026-11-02): the hip hop
    re-mix lifted a piano at -56 dBFS, alone in its intro and outro, by 16 dB, and against the
    files at unity that read as the master flattened by compression."""
    probe = session.probe_copy()
    for sid in (stems if stems is not None else list(probe.stems)):
        state = probe.stems.get(sid)
        if state is None:
            continue
        state["chain"] = [k for k in map(_trim, state.get("chain") or []) if k]
        state["rides"], state["width"] = [], 1.0
    if stems is None:
        probe._state["master_chain"] = []
    return probe


def _pct(share_db: float) -> str:
    """A share as a percentage a reader can compare: whole above 10 %, one decimal under it."""
    pct = 100.0 * 10.0 ** (share_db / 10.0)
    return f"{pct:.0f}%" if pct >= 10.0 else f"{pct:.1f}%"


def _tone_lines(raw: dict, now: dict, limit: float) -> list[str]:
    lines = []
    for key, up, down, where in (("top_db", "brighter", "darker", "over 2 kHz"),
                                 ("low_db", "fuller", "thinner", "under 400 Hz")):
        moved = now[key] - raw[key]
        if abs(moved) >= limit and max(now[key], raw[key]) >= SHARE_FLOOR_DB:
            lines.append(f"{up if moved > 0 else down} by {abs(moved):.1f} dB {where} "
                         f"({_pct(raw[key])} of its energy there raw, {_pct(now[key])} now)")
    return lines


def _flattening_lines(raw: dict, now: dict, crest: bool) -> list[str]:
    lines = []
    if (raw["spread_db"] is not None and now["spread_db"] is not None
            and raw["spread_db"] - now["spread_db"] >= SPREAD_LOST_DB):
        lines.append(f"its level moves {now['spread_db']:.1f} dB where the raw file moved "
                     f"{raw['spread_db']:.1f} dB")
    if crest and raw["crest_db"] - now["crest_db"] >= CREST_LOST_DB:
        lines.append(f"its peaks stand {now['crest_db']:.1f} dB over its average where the raw "
                     f"file's stood {raw['crest_db']:.1f} dB")
    return lines


def stem_against_raw(session: Session, stem_id: str) -> Optional[dict]:
    """The track as its chain leaves it against its file. None for a track with no chain, a
    silent one, or one the engine does not hold."""
    state = session.stems.get(stem_id)
    if state is None or not state.get("chain"):
        return None
    telemetry: list[dict] = []
    now_audio, sr = dsp.render_stem(session, stem_id, telemetry=telemetry)
    raw_audio, raw_sr = dsp.render_stem(raw_copy(session, [stem_id]), stem_id)
    now, raw = shape(now_audio, sr), shape(raw_audio, raw_sr)
    if now is None or raw is None:
        return None
    gr, levellers = stacked_gr_db(telemetry)
    lines = _tone_lines(raw, now, STEM_SHARE_DB) + _flattening_lines(raw, now, crest=True)
    if gr >= STACKED_GR_DB:
        lines.append(f"its {levellers} compressors take {gr:.1f} dB on average together"
                     if levellers > 1 else f"its compressor takes {gr:.1f} dB on average")
    return {"stem": stem_id, "raw": raw, "now": now, "levelling_gr_db": gr, "lines": lines}


def master_against_raw(session: Session) -> Optional[dict]:
    """The master against every track summed raw at its level, both measured as shapes. The crest is left
    out: a master is limited by design."""
    if not session.stems:
        return None
    now_audio, sr = dsp.render_master(session, report={})
    raw_audio, raw_sr = dsp.render_master(raw_copy(session), report={})
    now, raw = shape(now_audio, sr), shape(raw_audio, raw_sr)
    if now is None or raw is None:
        return None
    lines = _tone_lines(raw, now, MASTER_SHARE_DB) + _flattening_lines(raw, now, crest=False)
    return {"stem": MASTER, "raw": raw, "now": now, "lines": lines}


#: what the engineer is asked to do with a note (the knowledge block says why)
ADVICE = ("Was the raw audio already like that? Keep only the change the request needs, and undo "
          "or ease any move that leaves it worse than the raw audio (a sample-pack or bounced part "
          "often arrives compressed and bright already).")


def note(reading: Optional[dict]) -> Optional[str]:
    """The sentence a reading becomes, or None when it found nothing to hear."""
    if not reading or not reading["lines"]:
        return None
    who = ("the master, against the raw tracks at their levels" if reading["stem"] == MASTER
           else f"{reading['stem']}, against its raw file")
    return f"At matched loudness, {who}: " + "; ".join(reading["lines"]) + ". " + ADVICE


def session_notes(session: Session, master: bool = True) -> list[str]:
    """Every note a session's processing earns: each track with a chain, then the master
    (unless `master` is False: two master renders a run's pass budget has no room for)."""
    notes = [note(stem_against_raw(session, sid)) for sid in session.stems]
    if master and (any(state.get("chain") for state in session.stems.values())
                   or session.master_chain):
        notes.append(note(master_against_raw(session)))
    return [n for n in notes if n]


__all__ = ["shape", "stacked_gr_db", "raw_copy", "stem_against_raw", "master_against_raw",
           "note", "session_notes"]
