"""Every argument a command handler reads as a NUMBER is one the numeric door knows by name.

`dawcmd::badNumericArg` (DawCommands.h) runs at the top of `invokeCommand` and refuses an
argument that is not a finite number - for the arguments whose NAME says they are numbers
(`isNumericArgName`: a unit suffix, or a short list of unit-less names). A handler that does
`(double) args.getProperty ("depth", ...)` on a name the door does not know reads the STRING
"nan" as a NaN: `modifier.add` put one on the audio thread that way, and `sampler.set_sound`,
`sampler.slice`, `midi.quantize` and `step.set_cell` put one in the project (reader P2s C2.R11
and C3.R7, 2026-10-05).

So: every `(double)` / `(float)` read of a named argument in the command handlers is either a
name the door knows, or one of the exemptions below - each of which names where its number is
checked instead.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
COMMANDS = REPO / "daw" / "Source" / "DawCommands.h"
HANDLERS = [REPO / "daw" / "Source" / "App.cpp", REPO / "daw" / "Source" / "DawSession.h"]

# name -> where its finiteness is checked instead of at the door
EXEMPT = {
    # device.set / sampler.set take an enum NAME under `value`; instrument.set_param refuses a
    # non-finite one itself, and sampler.set checks it with numArg for every numeric param.
    "value": "the handler (instrumentSetParam's isfinite, samplerSet's numArg)",
}

_READ = re.compile(r'\((?:double|float)\)\s*(?:\(double\)\s*)?args(?:\.getProperty\s*\(\s*"(\w+)"|\s*\[\s*"(\w+)"\s*\])')


def _door():
    text = COMMANDS.read_text(encoding="utf-8")
    body = text[text.index("inline bool isNumericArgName"):]
    body = body[: body.index("\n}\n")]
    suffixes = re.search(r"suffixes\[\]\s*=\s*\{([^}]*)\}", body).group(1)
    names = re.search(r"names\[\]\s*=\s*\{([^}]*)\}", body).group(1)
    return (re.findall(r'"([^"]+)"', suffixes), set(re.findall(r'"([^"]+)"', names)))


def _known(name: str, suffixes, names) -> bool:
    return name in names or any(name.endswith(s) for s in suffixes)


def test_the_door_parse_finds_the_door():
    suffixes, names = _door()
    assert "_db" in suffixes and "_s" in suffixes
    assert {"bpm", "pan", "depth", "tune"} <= names


def test_every_numeric_read_of_an_argument_is_a_name_the_door_knows():
    suffixes, names = _door()
    unknown: dict[str, list[str]] = {}
    seen = 0
    for path in HANDLERS:
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for m in _READ.finditer(line):
                seen += 1
                name = m.group(1) or m.group(2)
                if name in EXEMPT or _known(name, suffixes, names):
                    continue
                unknown.setdefault(name, []).append(f"{path.name}:{n}")
    assert seen > 60, f"the scan read only {seen} numeric argument reads - the pattern no longer matches the handlers"
    assert unknown == {}, (
        "a handler reads these arguments as numbers and the numeric door does not know their names, "
        "so the string \"nan\" reaches the handler as a NaN. Name the argument with its unit, add it "
        "to isNumericArgName, or check it in the handler with dawcmd::numArg and list it in EXEMPT: "
        f"{unknown}")


def test_an_exemption_is_still_needed():
    """An exemption nobody reads any more is a hole waiting for the next reader."""
    text = "".join(p.read_text(encoding="utf-8") for p in HANDLERS)
    for name in EXEMPT:
        assert re.search(r'args(?:\.getProperty\s*\(\s*"%s"|\s*\[\s*"%s"\s*\])' % (name, name), text), name
