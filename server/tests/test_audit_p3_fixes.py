"""Gates on the P3 fixes from the 2026-10-02 full audit (docs/audits/full-2026-10-02/01-findings.md).

The C++ halves are console legs (--command-door, --slot-proxy, --prefs, --fresh-undo). These are
the two things a leg cannot hold: a source rule and the brain's half of F18.

F13  the dock icon's NSImage is released (the .mm files are compiled without ARC)
F14  no signature accumulator multiplies a signed integer in place (sighash:: wraps instead)
F18  the brain refuses an integer the DAW's JSON parser would wrap
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from orchestrator.daw_server import implausible_integer

REPO = Path(__file__).resolve().parents[2]
CPP = sorted(p for d in ("daw/Source", "native/Source") for p in (REPO / d).iterdir()
             if p.suffix in (".h", ".cpp", ".mm"))


# ---- F14 ---------------------------------------------------------------------------------------
_ACCUMULATE = re.compile(r"\b(\w+)\s*=\s*\1\s*\*\s*\d+\b")


def test_no_signed_hash_accumulator():
    """`sig = sig * 31 + x` on a signed integer is undefined once it overflows, and a signature
    overflows by design. UBSan found eleven; sighash::mix does the same sum on unsigned."""
    assert len(CPP) > 60
    bad = [f"{p.relative_to(REPO)}:{i}: {line.strip()[:90]}"
           for p in CPP for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1)
           if _ACCUMULATE.search(line)]
    assert bad == [], "use sighash::mix (native/Source/SigHash.h):\n" + "\n".join(bad)


def test_the_accumulator_pattern_still_finds_one():
    """Vacuity: the pattern recognises the shape it is there to refuse."""
    assert _ACCUMULATE.search("        sig = sig * 31 + (juce::int64) x;")
    assert _ACCUMULATE.search("h = h * 131 + c;")
    assert not _ACCUMULATE.search("sig = sighash::mix (sig, 31, x);")
    assert not _ACCUMULATE.search("w = h * 2;")


def test_sighash_is_used():
    uses = sum(p.read_text(errors="replace").count("sighash::mix") for p in CPP)
    assert uses >= 20, uses


# ---- F13 ---------------------------------------------------------------------------------------
def test_an_alloc_in_a_non_arc_file_is_released():
    """No `-fobjc-arc` in this build, so every [[X alloc] init...] owns a +1 that somebody has to
    give back in the same function. setDockIcon did not (clang's osx.cocoa.RetainCount). That
    function left with the light/dark Dock swap (2026-10-04) and no .mm allocates today, so this
    holds the rule for the next one that does."""
    cmake = (REPO / "daw/CMakeLists.txt").read_text()
    assert "-fobjc-arc" not in cmake, "ARC is on now - this test, and the manual releases, can go"
    for p in CPP:
        if p.suffix != ".mm":
            continue
        text = p.read_text(errors="replace")
        for m in re.finditer(r"(\w+)\s*=\s*\[\[\w+ alloc\]", text):
            var = m.group(1)
            body = text[m.end():]
            body = body[: body.find("\n}\n") if "\n}\n" in body else len(body)]
            assert re.search(rf"\[\s*{var}\s+(auto)?release\s*\]", body) or "autorelease]" in text[m.start():m.start() + 200], \
                f"{p.name}: `{var}` is alloc'ed and never released"


# ---- F18 ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("args", [
    {"scene": 2 ** 63},                     # wraps to the other sign in juce::JSON
    {"scene": 2 ** 64 + 1},                 # wraps to 1 - the DAW could never tell
    {"clip": -(2 ** 40)},
    {"track": "vocals", "notes": [[60, 0.0, 1.0, 2 ** 31]]},
    {"points": [{"t_s": 1.0, "index": 10 ** 30}]},
])
def test_an_integer_the_daw_cannot_hold_is_refused(args):
    why = implausible_integer(args)
    assert why and "not a plausible value" in why


@pytest.mark.parametrize("args", [
    None, {}, {"scene": 0}, {"clip": 2 ** 31 - 1}, {"on": True}, {"gain_db": -6.5},
    {"name": "9223372036854775808"},        # a NAME may be all digits
    {"notes": [[60, 0.0, 1.0, 100]]},
])
def test_ordinary_arguments_pass(args):
    assert implausible_integer(args) is None


def test_the_one_sender_checks_it():
    src = (REPO / "orchestrator/daw_server.py").read_text()
    body = src[src.index("    def daw_invoke(command: str"):]
    body = body[: body.index("req_id = next(_DAW_REQ_IDS)")]
    assert "implausible_integer(args)" in body, "daw_invoke must refuse before it serialises"
