"""`orchestrator/command_door.py` says what the app's command door says (reader E7·R2, 2026-10-06).

The operate eval's fake app refuses what the real choke point refuses. For that to stay true the
Python transcription has to follow `daw/Source/DawCommands.h` - its name lists, its bounds and
the words of each refusal - and the order the fake app applies them in has to be the order
`App.cpp` does. This file reads the C++ and fails when one moves without the other.

(The behaviour tests - what each refusal does to a verdict - are in test_reader_p2_py6.py.)
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from orchestrator import command_door as D
from orchestrator import operate_eval as O

ROOT = Path(__file__).resolve().parents[2]
COMMANDS_H = (ROOT / "daw" / "Source" / "DawCommands.h").read_text(encoding="utf-8")
APP = (ROOT / "daw" / "Source" / "App.cpp").read_text(encoding="utf-8")


def _body(signature: str) -> str:
    """The text of the inline function whose definition starts with `signature`."""
    start = COMMANDS_H.index(signature)
    return COMMANDS_H[start: COMMANDS_H.index("\n}\n", start)]


def _strings(text: str) -> list[str]:
    return re.findall(r'"((?:[^"\\]|\\.)*)"', text)


def _utf8(literal: str) -> str:
    """A C++ literal with \\xNN byte escapes, as the text `juce::String::fromUTF8` makes of it."""
    return literal.encode("latin-1").decode("unicode_escape").encode("latin-1").decode("utf-8")


# ---- the name lists ---------------------------------------------------------------------------

def test_the_path_argument_names_are_the_doors():
    assert tuple(_strings(_body("inline bool isPathArgName"))) == D.PATH_ARG_NAMES


def test_the_numeric_argument_names_are_the_doors():
    body = _body("inline bool isNumericArgName")
    suffixes = re.search(r"suffixes\[\]\s*=\s*\{([^}]*)\}", body).group(1)
    names = re.search(r"names\[\]\s*=\s*\{([^}]*)\}", body).group(1)
    assert tuple(_strings(suffixes)) == D.NUMERIC_SUFFIXES
    assert tuple(_strings(names)) == D.NUMERIC_NAMES
    assert tuple(_strings(_body("inline bool isNumericListName"))) == D.NUMERIC_LIST_NAMES


def test_the_single_value_argument_names_are_the_doors():
    """Phase 56: a list where one value belongs. The names, and what a list under one says."""
    body = _body("inline bool isScalarArgName")
    assert tuple(_strings(re.search(r"names\[\]\s*=\s*\{([^}]*)\}", body).group(1))) == D.SCALAR_ARG_NAMES
    assert '" must be a single value, not a list or an object"' in _body("inline bool badScalarArg")
    assert D.bad_scalar_arg({"index": []}) == "index must be a single value, not a list or an object"
    assert D.bad_scalar_arg({"track": {"name": "x"}}) and D.bad_scalar_arg({"on": [True]})
    # controls: one value passes; an argument that IS a list is not this door's
    assert D.bad_scalar_arg({"track": "bass", "index": 2, "on": False, "name": ""}) is None
    assert D.bad_scalar_arg({"stems": ["drums", "bass"], "points": [{"t_s": 1}]}) is None
    assert not set(D.SCALAR_ARG_NAMES) & set(D.NUMERIC_LIST_NAMES)
    # ...and a word where an index belongs: (int) of a word is 0, and scene 0 was deleted
    assert tuple(_strings(re.search(r"names\[\]\s*=\s*\{([^}]*)\}",
                                    _body("inline bool isIndexArgName")).group(1))) == D.INDEX_ARG_NAMES
    assert '" is an index - a whole number from the snapshot' in _body("inline bool badScalarArg")
    assert D.bad_scalar_arg({"scene": "first"}) == \
        'scene is an index - a whole number from the snapshot - not "first"'
    assert D.bad_scalar_arg({"clip": ""}) and D.bad_scalar_arg({"index": True})
    assert D.bad_scalar_arg({"scene": "2", "clip": 0, "index": 3.0, "take": None}) is None
    assert set(D.INDEX_ARG_NAMES) <= set(D.SCALAR_ARG_NAMES) | {"take", "chain", "occ", "slot"}


def test_the_timeline_and_integer_bounds_are_the_doors():
    m = re.search(r"kMaxTimelineS\s*=\s*([\d.]+)\s*\*\s*([\d.]+)\s*\*\s*([\d.]+)\s*;", COMMANDS_H)
    assert float(m.group(1)) * float(m.group(2)) * float(m.group(3)) == D.MAX_TIMELINE_S
    assert "std::abs (d) > 2147483647.0" in _body("inline bool implausibleNumber")
    assert D.implausible_number(2147483648) and not D.implausible_number(2147483647)
    assert "std::abs (d) > 1.0e9" in _body("inline bool badNumericArg")
    assert D.bad_numeric_arg({"gain_db": 1.0e9}) is None
    assert D.bad_numeric_arg({"gain_db": 1.1e9}) is not None


def test_the_one_path_outside_the_four_names_is_the_slots_source():
    """`slot.set_clip {source: "midi" | <audio path>}`: its handler refuses a relative one, so
    the brain resolves it like the rest. No other grammar in the catalog names a path there."""
    extra = {key: ("source",) for key, grammar in O._ARGS.items()
             if re.search(r"\bsource\??\s*:\s*[^,}]*\bpath\b", grammar)}
    assert extra == D.EXTRA_PATH_ARGS
    session = (ROOT / "daw" / "Source" / "DawSession.h").read_text(encoding="utf-8")
    assert "if (! juce::File::isAbsolutePath (source)) return nullptr;" in session


def test_an_export_with_no_file_goes_to_the_folder_a_bare_name_is_sent_to():
    assert f'project.getChildFile ("{D.EXPORT_FOLDER}")' in APP


# ---- the refusals' own words ------------------------------------------------------------------

def test_the_master_safety_sentence_is_one_wording():
    body = _body("inline juce::String agentMasterSafetyRefusal")
    no = body[body.index("const auto no ="): body.index(";", body.index("const auto no ="))]
    assert _utf8("".join(_strings(no))) == D.MASTER_SAFETY_REFUSAL


def test_the_master_safety_rule_reads_the_same_three_commands():
    body = _body("inline juce::String agentMasterSafetyRefusal")
    assert tuple(re.findall(r'command == "([a-z_.]+)"', body)) == D.MASTER_SAFETY_COMMANDS
    for needle in ('"on"', '"track"', '"device"', '"type"', '"copy"', '"from"', '"limiter"',
                   '"master"'):
        assert needle in body, needle


@pytest.mark.parametrize("command, args, limiter_on, refused", [
    ("master.limiter", {"on": False}, True, True),
    ("master.limiter", {"on": True}, False, False),
    ("master.limiter", {}, True, True),                    # omitted = toggle: it is on, so off
    ("master.limiter", {}, False, False),
    ("device.enable", {"track": "master", "device": "limiter#0", "on": False}, True, True),
    ("device.enable", {"track": "master", "device": "limiter"}, True, True),
    ("device.enable", {"track": "master", "device": "limiter", "on": True}, True, False),
    ("device.enable", {"track": "master", "device": "eq", "on": False}, True, False),
    ("device.enable", {"track": "bass", "device": "limiter", "on": False}, True, False),
    ("device.move_to_track", {"type": "limiter", "from": "master", "to": "bass"}, True, True),
    ("device.move_to_track", {"type": "limiter", "to": "bass"}, True, True),
    ("device.move_to_track", {"type": "limiter", "from": "master", "to": "bass", "copy": True},
     True, False),
    ("device.move_to_track", {"type": "limiter", "from": "bass", "to": "master"}, True, False),
    ("device.move_to_track", {"type": "comp", "from": "master", "to": "bass"}, True, False),
    ("device.set", {"track": "master", "params": {"limiter.ceiling_dbtp": -1}}, True, False),
])
def test_the_master_safety_rule_case_by_case(command, args, limiter_on, refused):
    """The cases `--agent-guard` drives on the C++ function, on the transcription."""
    assert (D.master_safety_refusal(command, args, limiter_on) is not None) is refused


#: a refusal the transcription makes -> the words of it that the C++ source must also hold
WORDED = [
    (D.bad_path_arg({"path": 5}), " must be a file path"),
    (D.bad_path_arg({"file": "bounces/x.wav"}), " must be an absolute path, got \\\""),
    (D.bad_numeric_arg({"gain_db": "loud"}), " must be a number, got \\\""),
    (D.bad_numeric_arg({"gain_db": {"x": 1}}), None),          # an object is walked, not judged
    (D.bad_numeric_arg({"gain_db": True}), " must be a number"),
    (D.bad_numeric_arg({"gain_db": float("inf")}), " is not a plausible value"),
    (D.bad_numeric_arg({"clip": 1e12}), " is not a plausible value"),
    (D.bad_numeric_arg({"start_s": 200000}), " is outside the timeline (a project is at most "),
    (D.bad_numeric_arg({"steps": [1, "x"]}), " must be a list of numbers"),
    (D.bad_numeric_arg({"steps": [1, float("nan")]}), " must be a list of finite numbers"),
    (D.bad_numeric_arg([1, 1e12]), "a number in a list is not a plausible value"),
]


@pytest.mark.parametrize("said, words", WORDED)
def test_each_refusal_uses_the_doors_words(said, words):
    if words is None:
        assert said is None
        return
    assert said is not None
    assert words.replace('\\"', '"') in said, said
    assert words in COMMANDS_H, f"the door no longer says {words!r}"


def test_what_the_door_lets_through_the_transcription_lets_through():
    fine = {"track": "bass", "gain_db": -3.5, "bpm": "128", "start_s": 0, "clip": 2, "pan": -1,
            "value": "tape_stop", "scale": "minor", "points": [{"t_s": 1.5, "value": 0.2}],
            "steps": [0, 0.5, 1], "name": None, "depth": None}
    assert D.bad_numeric_arg(fine) is None
    assert D.bad_path_arg({"path": "/a/b.wav", "dir": "~/Music/x", "into": "", "file": None,
                           "source": "relative/is/not/the/doors.wav"}) is None


# ---- the fake app applies them in the app's order --------------------------------------------

def test_the_choke_point_refuses_in_the_order_the_fake_app_does():
    start = APP.index("const auto* spec = dawcmd::specForId (id);")
    choke = APP[start: APP.index("auto perform = [safe, id, command, args, reply, reqId]", start)]
    order = [choke.index('"unknown command: "'), choke.index("' is a user-only control"),
             choke.index("dawcmd::agentMasterSafetyRefusal")]
    assert order == sorted(order)
    confirm = APP.index("' needs the user to confirm it", start)
    assert confirm > start + order[-1], "the confirm gate comes after the master safety rule"
    invoke = APP.index("juce::var invokeCommand (int id, const juce::var& args)")
    door = [APP.index("dawcmd::badPathArg (args, why)", invoke),
            APP.index("dawcmd::badNumericArg (args, why)", invoke),
            APP.index("dawcmd::badScalarArg (args, why)", invoke)]
    assert door == sorted(door), "the path door, then the numeric one, then the single-value one"
    assert door[1] - invoke < 4000, "both doors sit at the top of invokeCommand, before any case"

    fake = inspect.getsource(O.fake_daw)
    code = fake[fake.index("from .daw_bridge import"):]
    mine = [code.index(needle) for needle in (
        "brain_only_refusal(command)", "command_door.resolve_path_args(",
        "implausible_integer(sent)", "unknown command:", "is a user-only control",
        "command_door.master_safety_refusal(", "needs the user to confirm it",
        "command_door.bad_path_arg(", "command_door.bad_numeric_arg(",
        "command_door.bad_scalar_arg(", "case.fail_with")]
    assert mine == sorted(mine), "fake_daw no longer refuses in the app's order"


def test_the_two_refusals_not_transcribed_are_still_the_only_two():
    """`command_door`'s docstring says which of the choke point's refusals the fake app leaves
    out, and why. A new `dawcmd::agent…Refusal` at the choke point is a new line there."""
    start = APP.index("const auto* spec = dawcmd::specForId (id);")
    choke = APP[start: APP.index("auto perform = [safe, id, command, args, reply, reqId]", start)]
    assert sorted(set(re.findall(r"dawcmd::(agent\w+Refusal)", choke))) == [
        "agentMasterSafetyRefusal", "agentPostLimiterRefusal", "agentTakeRefusal"]
    for name in ("agentPostLimiterRefusal", "agentTakeRefusal"):
        assert name in (D.__doc__ or "")
