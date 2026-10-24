"""The DAW's command door, as the brain knows it.

Two things live here, and they are different kinds of code.

**The brain's own half (product).** `resolve_path_args` reads a relative path the agent hands a
command against the project folder (James, 2026-10-05, reader E7·R2). The app's door
(`dawcmd::badPathArg`) still refuses a relative path, for the reason it always did: `juce::File`
resolves one against wherever the app was launched from. So "save samples/riser.wav into the
library" - the path as the user typed it - was refused unless the model rebuilt the absolute path
from the snapshot's project row. The brain knows the project folder; it now says so before the
door is reached, at the two places a MODEL's arguments enter the bridge (the live tool and an
approved plan's command step - the same two `brain_only_refusal` guards).

**The door's refusals, transcribed (evals only).** `bad_path_arg`, `bad_numeric_arg` and
`master_safety_refusal` are Python twins of what the app's choke point refuses before a handler
runs. Nothing in the product calls them: the app is the door. They exist so the operate eval's
fake app (`operate_eval.fake_daw`) answers a reach the way the real one does - it used to answer
ok to everything but a confirm, so the gate rewarded a relative path, a non-number, and switching
the master's safety limiter off (readers E7·R1 / E7·R2). `server/tests/
test_command_door_parity.py` pins every name list and every refusal's wording against
`DawCommands.h`; change the door there and that test says what to change here.

Two refusals the choke point makes are NOT transcribed, because the eval's fixture can never
meet them: a device behind the master's limiter (`agentPostLimiterRefusal` - the fixture
publishes no master chain) and a playhead move under a rolling take (`agentTakeRefusal` - the
fixture never records).
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional

#: `dawcmd::isPathArgName` - the four names a path argument has in the catalog.
PATH_ARG_NAMES = ("path", "dir", "into", "file")

#: A path argument under another name, by command: `slot.set_clip`'s `source` is "midi" or an
#: audio file, and its handler (not the door) refuses a relative one.
EXTRA_PATH_ARGS: dict[str, tuple[str, ...]] = {"slot.set_clip": ("source",)}

#: Where the app writes an export it is given no file for (`bounces/export.<ext>`).
EXPORT_FOLDER = "bounces"

#: `dawcmd::isNumericArgName`: a unit suffix, or one of the unit-less names.
NUMERIC_SUFFIXES = ("_s", "_db", "_ms", "_hz", "_beat", "_beats", "_cents", "_dbtp", "_lufs",
                    "_per_beat", "_per_chord", "_bpm")
NUMERIC_NAMES = ("bpm", "semitones", "curve", "amount", "strength", "pan", "width", "numerator",
                 "denominator", "velocity", "bars", "cents", "octave", "octaves", "gate",
                 "depth", "center", "phase", "tune", "sensitivity", "swing", "probability",
                 "humanize", "after_loops")
#: `dawcmd::isNumericListName`
NUMERIC_LIST_NAMES = ("steps", "at", "gates", "probabilities")

#: `dawcmd::kMaxTimelineS` (48 hours: the engine's maximum edit length)
MAX_TIMELINE_S = 48.0 * 60.0 * 60.0
_INT32_MAX = 2147483647.0

#: `dawcmd::agentMasterSafetyRefusal`'s one sentence, and the four commands it reads.
MASTER_SAFETY_REFUSAL = ("the master safety limiter stays on for the engineer "
                         "— only you can switch it off (⌥⌘L)")
MASTER_SAFETY_COMMANDS = ("master.limiter", "device.enable", "device.remove", "device.move_to_track")


# ---- the brain's half ---------------------------------------------------------------------

def _open_project_dir(name: str, projects) -> Optional[str]:
    for row in projects or ():
        if isinstance(row, dict) and str(row.get("name") or "") == name and row.get("dir"):
            return str(row["dir"])
    return None


def _same_folder(a: Path, b: Path) -> bool:
    """Are these one folder? By identity where both exist (a spelling proves nothing: on this
    file system `/tmp` is `/private/tmp`), else by their normalised text."""
    try:
        if a.exists() and b.exists():
            return os.path.samefile(a, b)
    except OSError:
        pass
    return os.path.normpath(str(a)) == os.path.normpath(str(b))


def export_redirect_note(command: str, before, after) -> str:
    """What to tell the agent when `resolve_path_args` moved an export it had named with a full
    path: where the file is being written, and why. Empty when nothing was moved - and for a
    bare name, which has always gone to `bounces/` and is not news."""
    if command != "session.export" or not isinstance(before, dict) or not isinstance(after, dict):
        return ""
    asked, sent = before.get("file"), after.get("file")
    if not isinstance(asked, str) or not isinstance(sent, str) or asked == sent:
        return ""
    if not (asked.startswith("~") or Path(asked).is_absolute()):
        return ""
    return (f"written to {EXPORT_FOLDER}/{Path(sent).name}, not to the top of the project folder: "
            "exports go in bounces/, and a wav at the top of a project is read as a stem")


SCENE_ARG_NAMES = ("scene", "dest_scene")


def resolve_scene_names(args, scenes):
    """(`args` with a scene named by its NAME sent as its index, or a refusal).

    A scene has a name (`scene.rename`, `scene.from_sections`) and the app takes it by index:
    `scene.launch {scene: "intro"}` was refused, and the engineer looked the number up in
    `scenes[]` (user QA 2026-10-22, Q9). A word under `scene` / `dest_scene` is matched against
    the snapshot's `scenes[].name`, case and spaces aside; one match is sent as its index, a name
    two scenes share is refused naming both, a name none has is refused naming the ones there are.
    A number, or a word of digits, is the app's to judge (`bad_scalar_arg`). The caller's dict is
    never written to."""
    if not isinstance(args, dict):
        return args, None
    out = None
    for key in SCENE_ARG_NAMES:
        value = args.get(key)
        if not isinstance(value, str) or not value.strip() or value.strip().lstrip("-").isdigit():
            continue
        want = " ".join(value.split()).casefold()
        rows = [r for r in (scenes or []) if isinstance(r, dict)]
        hits = [r.get("index") for r in rows
                if isinstance(r.get("name"), str) and " ".join(r["name"].split()).casefold() == want]
        if len(hits) == 1 and isinstance(hits[0], int):
            if out is None:
                out = dict(args)
            out[key] = hits[0]
            continue
        if len(hits) > 1:
            return args, (f"{key}: {len(hits)} scenes are called \"{value}\" (indexes "
                          + ", ".join(str(h) for h in hits) + ") - name one by its index")
        named = [f"{r.get('index')} \"{r['name']}\"" for r in rows
                 if isinstance(r.get("name"), str) and r["name"].strip()]
        return args, (f"{key}: no scene is called \"{value}\""
                      + (" - the scenes are " + ", ".join(named) if named else " - no scene has a name (scenes[] in the snapshot)"))
    return (out if out is not None else args), None


def resolve_path_args(project_dir, command: str, args, projects=None):
    """`args` with every relative path argument made absolute against `project_dir`.

    Returns `args` itself when nothing changes, else a shallow copy - the caller's dict is
    never written to (a recorded plan and the model's own tool call both keep what they held).

    * A path that is already absolute, or starts with `~` (the app expands it), is left alone -
      except `session.export`'s `file` when it sits DIRECTLY in the project folder: that goes to
      `bounces/` too (`export_redirect_note` says so to the agent).
    * `session.export`'s `file`, given as a bare file name, goes to `bounces/` - where an export
      with no `file` goes. Only a stem is written at the top of the project folder; a path that
      names its own folder is taken as written.
    * `dir` always names ANOTHER project, and projects live side by side. A bare name an open
      project has (`projects`, the snapshot's rows) is that project's folder; otherwise the
      path is read against this project's folder like the rest - unless nothing is there and
      a folder of that name sits BESIDE this project, which is then the one meant ("what's
      in older_song?").
    * Anything that is not a non-empty string is the door's to judge, not ours.
    """
    if not isinstance(args, dict) or project_dir is None:
        return args
    base = Path(project_dir).expanduser()
    out = None
    for key in (*PATH_ARG_NAMES, *EXTRA_PATH_ARGS.get(command, ())):
        value = args.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        if key not in PATH_ARG_NAMES and value.strip().lower() == "midi":
            continue
        if value.startswith("~") or Path(value).is_absolute():
            # ...but an EXPORT is never written at the top of the project folder, where a wav is
            # read as a stem (user QA 2026-10-08, D6: "call it smoke" came as
            # <project>/smoke.wav, so the bare-name rule below never saw it).
            if command == "session.export" and key == "file":
                given = Path(value).expanduser()
                if _same_folder(given.parent, base):
                    if out is None:
                        out = dict(args)
                    out[key] = str(base / EXPORT_FOLDER / given.name)
            continue
        rel = Path(value)
        if key == "dir" and len(rel.parts) == 1 \
                and (known := _open_project_dir(value, projects)) is not None:
            resolved = known
        elif key == "dir" and not (base / rel).is_dir() and (base.parent / rel).is_dir():
            resolved = str(base.parent / rel)
        elif command == "session.export" and key == "file" and len(rel.parts) == 1:
            resolved = str(base / EXPORT_FOLDER / rel)
        else:
            resolved = str(base / rel)
        if out is None:
            out = dict(args)
        out[key] = resolved
    return args if out is None else out


# ---- the door's refusals, transcribed (evals only) ----------------------------------------

def _is_number(x) -> bool:
    # juce::var: a JSON true/false is a bool, not an int - isInt() and isDouble() are both false
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _show(d: float) -> str:
    """A number the way the app's message shows one, near enough for a reader: no trailing .0."""
    return f"{d:.15g}"


def is_numeric_arg_name(key: str) -> bool:
    return key in NUMERIC_NAMES or any(key.endswith(s) for s in NUMERIC_SUFFIXES)


def implausible_number(x) -> bool:
    """`dawcmd::implausibleNumber`: any NUMBER that is not finite or is past 32 bits."""
    return _is_number(x) and (not math.isfinite(float(x)) or abs(float(x)) > _INT32_MAX)


def bad_numeric_arg(value, device_fields: bool = False) -> Optional[str]:
    """`dawcmd::badNumericArg`: the reason an argument is not the finite number its name says
    it is, else None. Nested objects and lists are walked, as they are there.

    One difference, deliberate: a numeric STRING the app would read by its longest numeric
    prefix ("1e", "--5") is refused here. No model sends one, and guessing JUCE's parse of it
    would be a second thing to keep in step.

    `params` holds a DEVICE's own fields (device.add) or targets (device.set), not the command's
    arguments: the name rule is not applied inside it - saturation's `curve` is a name there.
    """
    if isinstance(value, (list, tuple)):
        for e in value:
            if implausible_number(e):
                return "a number in a list is not a plausible value"
            if (why := bad_numeric_arg(e, device_fields)) is not None:
                return why
        return None
    if not isinstance(value, dict):
        return None
    for key, x in value.items():
        key = str(key)
        if isinstance(x, (list, tuple)) and key in NUMERIC_LIST_NAMES:
            for e in x:
                if not _is_number(e):
                    return f"{key} must be a list of numbers"
                if not math.isfinite(float(e)):
                    return f"{key} must be a list of finite numbers"
        if isinstance(x, (list, tuple, dict)):
            if (why := bad_numeric_arg(x, device_fields or key == "params")) is not None:
                return why
            continue
        if implausible_number(x):
            return f"{key} {_show(float(x))} is not a plausible value"
        if device_fields or not is_numeric_arg_name(key) or x is None:
            continue
        if isinstance(x, str):
            s = x.strip()
            if not s or any(ch not in "0123456789+-.eE" for ch in s):
                return f'{key} must be a number, got "{s[:40]}"'
            try:
                d = float(s)
            except ValueError:
                return f'{key} must be a number, got "{s[:40]}"'
        elif not _is_number(x):
            return f"{key} must be a number"
        else:
            d = float(x)
        if not math.isfinite(d):
            return f"{key} must be a finite number"
        if abs(d) > 1.0e9:
            return f"{key} {_show(d)} is not a plausible value"
        if key.endswith("_s") and abs(d) > MAX_TIMELINE_S:
            return (f"{key} {_show(d)} is outside the timeline (a project is at most "
                    f"{int(MAX_TIMELINE_S)} s long)")
    return None


#: `dawcmd::isScalarArgName`: the arguments that are ONE value everywhere in the catalog.
SCALAR_ARG_NAMES = ("track", "stem", "source", "return", "bus", "rack", "id", "name", "new_name",
                    "clip", "scene", "slot", "marker", "index", "take", "chain", "occ", "note",
                    "on", "muted", "enabled", "solo", "exclusive", "pre_fader", "copy", "register",
                    "replace", "kind", "type", "format", "mode", "target", "from", "to",
                    "amount", "bars", "query", "scope", "tape", "tapes", "cards", "field", "print",
                    "preset", "category", "description", "overwrite", "clear_routes",
                    "pack", "sound", "instrument", "zip")


#: `dawcmd::isIndexArgName`: read with `(int)` by every handler, and never a name.
INDEX_ARG_NAMES = ("clip", "scene", "index", "slot", "take", "chain", "occ")


def bad_scalar_arg(args) -> Optional[str]:
    """`dawcmd::badScalarArg`: a list or an object where a single value belongs is refused -
    it used to be read as nothing, as zero or as the text "[Array]", and the command ran -
    and so is a word (or a true / false) where an index belongs: `(int)` of a word is 0."""
    if not isinstance(args, dict):
        return None
    for key, x in args.items():
        key = str(key)
        if key in SCALAR_ARG_NAMES and isinstance(x, (list, dict)):
            return f"{key} must be a single value, not a list or an object"
        if key in INDEX_ARG_NAMES and isinstance(x, str):
            s = x.strip()
            if not s or not all(ch in "0123456789" for ch in s):
                return f'{key} is an index - a whole number from the snapshot - not "{s[:40]}"'
        elif key in INDEX_ARG_NAMES and isinstance(x, bool):
            return f"{key} is an index - a whole number from the snapshot"
    return None


def bad_path_arg(args) -> Optional[str]:
    """`dawcmd::badPathArg`: a path argument that is present and non-empty is an absolute path
    (`~/...` counts: the app expands it before the door)."""
    if not isinstance(args, dict):
        return None
    for key, x in args.items():
        key = str(key)
        if key not in PATH_ARG_NAMES or x is None:
            continue
        if not isinstance(x, str):
            return f"{key} must be a file path"
        if x and not (x.startswith("~") or Path(x).is_absolute()):
            return f'{key} must be an absolute path, got "{x[:60]}"'
    return None


def _truthy(x) -> bool:
    """`(bool) juce::var`, for what a model sends: a JSON bool or number, or "true" / "1"."""
    if isinstance(x, str):
        return x.strip().lower() in ("true", "1", "yes", "on")
    return bool(x)


def master_safety_refusal(command: str, args, master_limiter_on: bool) -> Optional[str]:
    """`dawcmd::agentMasterSafetyRefusal`: the agent may not switch the master's safety limiter
    off, bypass it, or move it off the master. None lets the command through."""
    args = args if isinstance(args, dict) else {}

    def text(key: str) -> str:
        return str(args.get(key) or "").strip()

    if command == "master.limiter":
        turns_on = _truthy(args["on"]) if "on" in args else not master_limiter_on   # omitted = toggle
        return None if turns_on else MASTER_SAFETY_REFUSAL
    if (command == "device.enable" and text("track") == "master"
            and text("device").split("#", 1)[0] == "limiter"):
        return None if ("on" in args and _truthy(args["on"])) else MASTER_SAFETY_REFUSAL
    if (command == "device.remove" and text("track") == "master"
            and text("device").split("#", 1)[0] == "limiter"):
        return MASTER_SAFETY_REFUSAL
    if (command == "device.move_to_track" and text("type") == "limiter"
            and not _truthy(args.get("copy"))):
        if text("from") in ("master", ""):       # an omitted `from` is the dock, which may be the master
            return MASTER_SAFETY_REFUSAL
    return None
