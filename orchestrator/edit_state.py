"""Fit the DAW's live edit snapshot into a prompt budget WITHOUT going blind.

The snapshot (`daw/Source/EditSnapshot.h`) writes `tracks` first and everything
session-global after it — tempo, transport, undo, groups, VCAs, markers,
sections, scenes, the library, the plugin list, and second to last `selection`.
Until 2026-09-01 every consumer cut it with `text[:budget]`, so on any real
session (one MIDI clip publishes up to 512 notes, ~46 kB alone) the agent lost
exactly the fields the prompt tells it to rely on, was handed unparseable JSON,
and never learned anything was missing. The eval fixtures are small, so every
gate stayed green (sweep P1).

This drops the known-unbounded, least-load-bearing fields first and says so:
the result is always valid JSON, the session-global tail always survives, and
an `omitted` list names what went.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

# Cheapest information first. Each level is applied cumulatively until the
# text fits. Before the track list itself goes, each track is THINNED to the row
# that names it (`track_detail`): the last resort still empties the list. A rack is
# thinned before any clip list or track row goes (`racks`, reader C4·R11): it is
# detail about ONE device group, and `read_track rack:<name>` brings it back.
DROP_ORDER = ("clip_notes", "device_params", "plugin_params", "instrument_params",
              "available_plugins", "library", "racks", "clips", "track_detail", "tracks")

#: The live meter readings a device row carries (`MonitorChainPlugin::getDeviceLevelsVar`).
#: They differ in every snapshot taken while audio plays, and are not a change to the edit.
METER_FIELDS = ("in_db", "out_db", "delta_db", "out_peak_db", "gr_db", "corr", "corr_bands",
                "tracks_sum_db", "tracks_cancel",
                # 2026-10-17 (the device audits): a multiband comp's / OTT's per-band reading, the
                # transient shaper's onset amount and its shaping
                "gr_bands_db", "transient", "shape_db", "move_db", "band_level_db",
                "band_levels_db",
                # Phase 63: where an Auto Filter's cutoff is, and its envelope's reading
                "cutoff_now_hz", "env_now")


def _dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _drop(snap: dict, level: str) -> None:
    tracks = snap.get("tracks") or []
    if level == "clip_notes":
        for t in tracks:
            for c in t.get("clips") or []:
                if isinstance(c, dict) and "notes" in c:
                    c["notes_omitted"] = len(c.pop("notes") or [])
                # ...and each controller's points with them (P2 batch T, reader C4·R13): a
                # dense CC or pitch-bend lane is the same kind of bulk, and must not be what
                # pushes a track out. `count` stays; `read_track <track>` reads them back.
                for cc in (c.get("controllers") or []) if isinstance(c, dict) else []:
                    if isinstance(cc, dict) and "points" in cc:
                        cc["points_omitted"] = len(cc.pop("points") or [])
    elif level == "device_params":
        # A rack's parameters sit one level down, under its chains (EditSnapshot.h `rackVar`:
        # racks[].chains[].devices[].params - a native device's whole chain entry, a hosted
        # plug-in's first twelve parameters). The tier used to look for `racks[].devices`, which
        # the app never writes, so it was recorded as applied while removing nothing and a
        # project with a few racks lost every track before it lost one rack parameter (full
        # audit 2026-10-02, reader C4·R11).
        chains = [c for r in (snap.get("racks") or []) if isinstance(r, dict)
                  for c in (r.get("chains") or [])]
        for holder in [*tracks, *(snap.get("groups") or []), *chains]:
            for d in (holder.get("devices") or []) if isinstance(holder, dict) else []:
                if isinstance(d, dict) and "params" in d:
                    d["params_omitted"] = True
                    d.pop("params", None)
    elif level == "plugin_params":
        # Hosted plug-ins' parameter lists, automation addresses and program names
        # (2026-09-10 agent audit P1-18): one big synth's pushed every track out of the
        # agent's context, because no tier dropped them before `tracks` itself went.
        for t in tracks:
            for plug in (t.get("plugins") or []) if isinstance(t, dict) else []:
                if isinstance(plug, dict):
                    for key in ("params", "automation_targets", "programs"):
                        if key in plug:
                            plug.pop(key, None)
                            plug[f"{key}_omitted"] = True
    elif level == "instrument_params":
        for t in tracks:
            if "instrument_params" in t:
                t["instrument_params_omitted"] = True
                t.pop("instrument_params", None)
            # …and the synth's switches (wave shapes, FX power) with them: one `read_track`
            # brings both back, and the marker above says so.
            if isinstance(t, dict):
                t.pop("synth", None)
    elif level in ("available_plugins", "library"):
        if level in snap:
            gone = snap.pop(level) or []
            # The library is an object - {name, item_count, items} - and its length is its
            # three keys: 137 samples were reported as `library_omitted: 3` (C4·R11).
            if isinstance(gone, dict):
                gone = gone.get("items") or []
            snap[f"{level}_omitted"] = len(gone)
    elif level == "racks":
        for r in snap.get("racks") or []:
            _thin_rack(r)
    elif level == "clips":
        for t in tracks:
            if "clips" in t:
                t["clips_omitted"] = len(t.pop("clips") or [])
    elif level == "track_detail":
        # The master's and the returns' rows are channels like any other (2026-10-03): thinned
        # with the tracks, and one `read_track` away like them.
        for t in [*tracks, snap.get("master"), *(snap.get("returns") or [])]:
            _thin_track(t)
    elif level == "tracks":
        snap["tracks_omitted"] = len(snap.pop("tracks") or [])


def _thin_track(t: Any) -> None:
    """One track, cut to the row that NAMES it (full audit 2026-10-02, reader C4·R3). The tier
    after `clips` used to be `tracks` — the whole list, all or nothing — and a track row is big
    on its own (devices with live levels, automatable targets, a synth's parameters and mod
    targets), so a 30-track mix with three devices a track arrived as `tracks_omitted: 30` in a
    seventh of the budget: no names, no mute/solo, no routing. "Bring the snare up" had no
    track called Snare to find. Every scalar the row carries stays (index, name, kind, mute,
    solo, output, fader…), its devices shrink to their types, and `detail_omitted` says the
    rest is one `read_track` away."""
    if not isinstance(t, dict) or t.get("detail_omitted"):
        return
    devices = [str(d.get("type") or d.get("name") or "?") for d in (t.get("devices") or [])
               if isinstance(d, dict)]
    plugins = [str(p.get("name") or "?") for p in (t.get("plugins") or []) if isinstance(p, dict)]
    for key in [k for k, v in t.items() if isinstance(v, (dict, list))]:
        if key == "clips":
            t["clips_omitted"] = len(t.get("clips") or [])
        del t[key]
    if devices:
        t["device_types"] = devices
    if plugins:
        t["plugin_names"] = plugins
    t["detail_omitted"] = True


def _thin_rack(r: Any) -> None:
    """One rack, cut to what names it and its parts (reader C4·R11): its scalars, each chain's
    name and each macro's name. `racks` was in no tier at all, so six racks were 30 000
    characters after every tier of a 12 000 budget. `detail_omitted` says the device rows,
    their addresses and the macro mappings are one `read_track rack:<name>` away."""
    if not isinstance(r, dict) or r.get("detail_omitted"):
        return
    chains = [str(c.get("name") or "?") for c in (r.get("chains") or []) if isinstance(c, dict)]
    macros = [str(m.get("name") or "?") for m in (r.get("macros") or []) if isinstance(m, dict)]
    r.pop("chains", None)
    r.pop("macros", None)
    if chains:
        r["chain_names"] = chains
    if macros:
        r["macro_names"] = macros
    r["detail_omitted"] = True


def redact_home_text(text: str, home: str | None = None) -> str:
    """One string with the home directory written as `~` — a bare path, or paths inside JSON or
    prose (a tool result is `json.dumps` text). The follow-ups' item 4 applies it to every tool
    result in `chat.execute_tool`, so nothing the machine composes for the model carries it."""
    home = (home if home is not None else os.path.expanduser("~")).rstrip("/")
    if not text or not home or home == "~":
        return text
    if text == home:
        return "~"
    # ANCHORED to a path boundary (2026-09-11 pre-merge review, C9). A bare replace matched the
    # home path anywhere inside a longer one, so a sample on a backup or clone volume that mirrors
    # the home layout came out MANGLED and unopenable — /Volumes/Backup/Users/parr/Music/x.wav
    # became /Volumes/Backup~/Music/x.wav, and nothing downstream can undo that (expanduser leaves
    # it alone, and dawcmd::expandHome only expands a LEADING ~). The home only starts a path at
    # the beginning of the string or after a character no path segment can end with.
    return re.sub(r'(?<![A-Za-z0-9_.\-/])' + re.escape(home) + r'(?=[/"\s]|$)', "~", text)


def redact_home(obj: Any, home: str | None = None) -> Any:
    """`obj` with the user's home directory written as `~` (2026-09-10 audit P3, 06·F8).

    The snapshot carries absolute clip-source and sample paths (`/Users/<name>/…`), and
    every turn sends it to the model — to the control plane and Anthropic on cloud — so the
    macOS username and folder layout left the machine in normal use. `~/…` still names the
    file for the agent, and both readers of a path it hands back expand it: JUCE's File
    (parseAbsolutePath) and the brain's Path.expanduser()."""
    home = (home if home is not None else os.path.expanduser("~")).rstrip("/")
    if not home or home == "~":
        return obj

    def walk(v: Any) -> Any:
        if isinstance(v, str):
            return redact_home_text(v, home)
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        if isinstance(v, list):
            return [walk(x) for x in v]
        return v
    return walk(obj)


def live_edit(edit_state):
    """The edit as it is NOW. `edit_state` reaches the brain's functions either as a snapshot
    (a dict, captured once) or as a getter of the latest one - a chat turn is handed the
    getter, so its later rounds see the edit as its own commands left it. Whoever READS the
    edit asks this first; whoever only passes it on hands the getter on untouched.

    QA 2026-10-08 (D21): a mix run started from chat was passed the turn's getter and its
    planner read the getter as the snapshot - `'function' object has no attribute 'items'`
    on the first pass of every chat-started run."""
    return edit_state() if callable(edit_state) else edit_state


def compact_edit_state(edit_state: dict, budget: int,
                       exclude: tuple[str, ...] = ("commands", "daw_commands")) -> str:
    """Compact JSON for `edit_state` under `budget` characters.

    Never a character-level cut: fields are dropped in DROP_ORDER until the
    text fits, each drop recorded in `omitted`. The command catalog keys are
    excluded up front (they live in the cached half of the prompt). Home paths
    are written as `~` (redact_home): this text only ever goes to a model."""
    # `seq` is the snapshot's own number (the DAW stamps every push; a command reply names the
    # one it follows). It is the brain's to compare, and a number that moves on every push
    # would make an unchanged edit read as changed in every round's delta.
    snap = json.loads(_dumps(redact_home(
        {k: v for k, v in edit_state.items() if k not in exclude and k != "seq"})))
    text = _dumps(snap)
    if len(text) <= budget:
        return text
    omitted: list[str] = []
    for level in DROP_ORDER:
        if level == "clips":
            # Track by track, biggest first, the selected/focused tracks last (2026-09-10
            # audit P2, 04·F13): one tier used to drop EVERY track's clips at once, and on
            # a 40-track mix "trim the snare clip in the bridge" had no clip index anywhere.
            # What goes is named per track (`clips_omitted`); read_track brings one back.
            snap["omitted"] = omitted + ["clips (some tracks)"]
            for t in _clip_drop_order(snap):
                t["clips_omitted"] = len(t.pop("clips") or [])
                text = _dumps(snap)
                if len(text) <= budget:
                    return text
        if level == "track_detail":
            # Row by row, like the clips: unselected tracks first, the biggest rows first, and
            # stop as soon as it fits — the tracks the user is on keep their whole entry.
            snap["omitted"] = omitted + ["track_detail (some tracks)"]
            for t in _detail_drop_order(snap):
                _thin_track(t)
                text = _dumps(snap)
                if len(text) <= budget:
                    return text
        _drop(snap, level)
        omitted.append(level)
        snap["omitted"] = omitted
        text = _dumps(snap)
        if len(text) <= budget:
            return text
    return text   # even the bare tail overflowed — still valid JSON, still labelled


def _clip_drop_order(snap: dict) -> list[dict]:
    """Tracks carrying clips, in the order their clips go: unselected before selected, then
    the biggest clip lists first — so the tracks the user is working on keep theirs."""
    sel = snap.get("selection") or {}
    keep = {str(x) for x in (sel.get("tracks") or [])}
    if sel.get("focused_track"):
        keep.add(str(sel["focused_track"]))
    if isinstance(sel.get("clip"), dict) and sel["clip"].get("track"):
        keep.add(str(sel["clip"]["track"]))
    tracks = [t for t in (snap.get("tracks") or [])
              if isinstance(t, dict) and t.get("clips")]
    return sorted(tracks, key=lambda t: (str(t.get("name", "")) in keep,
                                         -len(_dumps(t.get("clips")))))


def _detail_drop_order(snap: dict) -> list[dict]:
    """Every track row, in the order it is thinned: unselected before selected, biggest first."""
    sel = snap.get("selection") or {}
    keep = {str(x) for x in (sel.get("tracks") or [])}
    if sel.get("focused_track"):
        keep.add(str(sel["focused_track"]))
    if isinstance(sel.get("clip"), dict) and sel["clip"].get("track"):
        keep.add(str(sel["clip"]["track"]))
    tracks = [t for t in (snap.get("tracks") or []) if isinstance(t, dict)]
    return sorted(tracks, key=lambda t: (str(t.get("name", "")) in keep, -len(_dumps(t))))


def _track_key(t: Any, i: int) -> str:
    if isinstance(t, dict):
        for k in ("id", "name"):
            if t.get(k) not in (None, ""):
                return f"{k}:{t[k]}"
    return f"#{i}"


def _without_meters(value: Any) -> Any:
    """`value` with the live meter readings taken off every device row, for COMPARING two
    snapshots. Tracks, the master, the returns and the groups all carry such rows."""
    if isinstance(value, dict):
        return {k: ([{f: x for f, x in row.items() if f not in METER_FIELDS}
                     if isinstance(row, dict) else row for row in v]
                    if k == "devices" and isinstance(v, list) else _without_meters(v))
                for k, v in value.items()}
    if isinstance(value, list):
        return [_without_meters(v) for v in value]
    return value


def _thinned_copy(t: dict) -> dict:
    copy = json.loads(_dumps(t))
    _thin_track(copy)
    return copy


def edit_state_delta(old: dict, new: dict) -> dict | None:
    """What changed between two snapshots, or None when nothing did.

    The subscription transport cannot replace the live block between rounds (one fixed
    system prompt per turn), so each command's result used to carry a FULL refreshed
    snapshot — a 40-command build carried 40 copies, ~240k tokens at the 24k cap, past
    the context window (2026-09-10 audit P2, 01·F7). This carries only the difference:
    every changed session-global field whole, and per track the whole track entry when
    it changed (its clip indexes are then current), plus added and removed tracks. The
    track ORDER is included whenever it moved, since commands address clips by index.

    Both sides are COMPACTED snapshots, so three differences are the budget's and not the
    edit's (full audit 2026-10-02, reader C4·R12), and the note this builds is stored in the
    conversation for good on the append-only models:

    * the track list no longer fits (`tracks_omitted`). That used to read `tracks_removed:
      [every track]`, `track_order: []` under "CHANGED since the last snapshot you saw" - one
      `track.add` at the cliff told the model the session had been emptied. It says the list
      is not shown, and that the tracks are still there.
    * a row was thinned to fit (`detail_omitted`) with nothing it still shows changed. It is
      named in `tracks_detail_not_shown` - what the model holds for it may be out of date -
      instead of being sent again.
    * the live meter readings (METER_FIELDS) moved. They differ in every snapshot taken while
      audio plays, so each command's note carried every device-bearing track whole: ten
      commands over a rolling transport were 105 000 characters of notes against 9 900
      stopped. A row that changed for a real reason still goes whole, meters as they are."""
    delta: dict = {}
    for key in sorted(set(old) | set(new)):
        if key == "tracks":
            continue
        if _without_meters(old.get(key)) != _without_meters(new.get(key)):
            delta[key] = new.get(key, None)
    if "tracks" not in new and new.get("tracks_omitted"):
        if "tracks" in old or old.get("tracks_omitted") != new.get("tracks_omitted"):
            delta["tracks_not_shown"] = (
                "the track list no longer fits this note - every track is still in the "
                "project (tracks_omitted is how many); read_track <name> reads one")
        return delta or None
    old_t = {_track_key(t, i): t for i, t in enumerate(old.get("tracks") or [])}
    new_list = new.get("tracks") or []
    new_t = {_track_key(t, i): t for i, t in enumerate(new_list)}
    changed, thinned = [], []
    for k, t in new_t.items():
        was = old_t.get(k)
        if was is None or _without_meters(was) == _without_meters(t):
            continue
        if (isinstance(t, dict) and isinstance(was, dict) and t.get("detail_omitted")
                and not was.get("detail_omitted") and _thinned_copy(was) == t):
            thinned.append(str(t.get("name", k)))
            continue
        changed.append(t)
    added = [t for k, t in new_t.items() if k not in old_t]
    removed = [k for k in old_t if k not in new_t]
    if changed:
        delta["tracks_changed"] = changed
    if thinned:
        delta["tracks_detail_not_shown"] = thinned
    if added:
        delta["tracks_added"] = added
    if removed:
        delta["tracks_removed"] = removed
    if list(old_t) != list(new_t):
        delta["track_order"] = list(new_t)
    return delta or None
