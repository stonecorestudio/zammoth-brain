"""Gates on the P2 fixes from the 2026-10-02 full audit (docs/audits/full-2026-10-02/01-findings.md).

F5  the brain writes STRICT json lines (juce::JSON::parse refuses NaN / Infinity / U+0000 / a lone
    surrogate, and the DAW drops a line it cannot parse)
F6  a co-edit number that is present and not a finite number is REFUSED, not turned into a rail
F7  start_run's pass limit reaches the run in the DAW; `gating` is not offered there
F8  CI's flag pattern finds every console leg in Main.cpp
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from orchestrator import daw_bridge
from orchestrator.chat import daw_tool_defs
from orchestrator.daw_server import wire_line

REPO = Path(__file__).resolve().parents[2]


# ---- F5 ----------------------------------------------------------------------------------------
def _strict(line: bytes):
    def refuse(token):
        raise AssertionError(f"non-standard token {token!r} on the wire")
    return json.loads(line.decode(), parse_constant=refuse)


@pytest.mark.parametrize("payload, expect", [
    ({"lufs": float("-inf"), "peaks": [float("nan"), float("inf"), -3.5]}, {"lufs": None, "peaks": [None, None, -3.5]}),
    ({"text": "a\x00b"}, {"text": "ab"}),
    ({"text": "x\ud83cy"}, {"text": "x�y"}),                      # a lone surrogate
    ({"text": "\U0001f39b ok", "n": 1.25}, {"text": "\U0001f39b ok", "n": 1.25}),   # a real emoji survives
    ({"text": "\\u0000 as typed"}, {"text": "\\u0000 as typed"}),      # literal backslash-u is just text
    ({"nested": [{"v": float("nan")}], "k\x00": 1}, {"nested": [{"v": None}], "k": 1}),
])
def test_the_brain_writes_strict_json_lines(payload, expect):
    line = wire_line(payload)
    assert line.endswith(b"\n") and line.count(b"\n") == 1
    assert _strict(line) == expect
    text = line.decode()
    # the four things juce::JSON::parse refuses a whole document for
    assert not re.search(r"\bNaN\b|Infinity", text)
    assert "\\u0000" not in text.replace("\\\\u0000", "")
    assert not re.search(r"\\ud[89ab][0-9a-f]{2}(?!\\ud[c-f])", text, re.I)


def test_wire_line_control_would_fail_with_plain_dumps():
    """The vacuity guard: plain json.dumps really does emit what the DAW refuses."""
    assert "NaN" in json.dumps({"a": float("nan")}) and "\\u0000" in json.dumps({"t": "a\x00b"})


# ---- F6 ----------------------------------------------------------------------------------------
def _state(project: Path) -> dict:
    s = json.loads((project / "session.json").read_text())
    s.pop("coedit_log", None)
    return s


@pytest.fixture()
def seeded(demo_project: Path) -> Path:
    from mix_engine.session import Session
    s = Session(demo_project)
    s.group_stems(sorted(s.stems))
    s.add_return("reverb")
    r = daw_bridge.apply_coedits(demo_project, [
        "/api/add_device?device_type=comp", "/api/add_device?device_type=eq&stem=stem_a"])
    assert r["applied"] == 2
    return demo_project


# (An EMPTY value is not in this list: parse_qs drops a blank parameter, so `gain_db=` is an
# absent key, which takes its default - the documented behaviour, not a refusal.)
BAD = ["nan", "NaN", "inf", "-inf", "Infinity", "1e999", "abc"]


@pytest.mark.parametrize("url", [
    "/api/set_stem?stem=stem_a&gain_db={v}",
    "/api/set_stem?stem=stem_a&pan={v}",
    "/api/set_stem?stem=stem_a&width={v}",
    "/api/chain_param?device_type=comp&occurrence=0&value={v}",
    "/api/device_param?device_type=comp&occurrence=0&param=threshold_db&value={v}",
    "/api/device_param?device_type=comp&occurrence={v}&param=ratio&value=4",
    "/api/device_param?device_type=eq&occurrence=0&param=gain_db&value={v}&stem=stem_a",
    "/api/set_return_level?return=RET&gain_db={v}",
    "/api/set_send?source=stem_a&return=RET&gain_db={v}",
    "/api/device_range?device_type=comp&occurrence=0&start_s={v}&end_s=2",
    "/api/device_band?occurrence=0&stem=stem_a&lo_hz={v}&hi_hz=4000",
])
def test_a_coedit_number_that_is_not_a_finite_number_is_refused(seeded: Path, url: str):
    ret = next(iter(json.loads((seeded / "session.json").read_text())["returns"]))
    for bad in BAD:
        before = _state(seeded)
        reply = daw_bridge.apply_coedits(seeded, [url.replace("RET", ret).format(v=bad)])
        assert reply["failed"] == 1 and reply["applied"] == 0, (url, bad, reply)
        assert reply["errors"], "a refusal says why"
        assert _state(seeded) == before, f"{url} with {bad!r} moved the session"


def test_a_real_number_still_lands_and_an_absent_one_takes_its_default(seeded: Path):
    ret = next(iter(json.loads((seeded / "session.json").read_text())["returns"]))
    r = daw_bridge.apply_coedits(seeded, [
        "/api/set_stem?stem=stem_a&gain_db=-4.5",
        "/api/device_param?device_type=comp&occurrence=0&param=threshold_db&value=-21",
        f"/api/set_return_level?return={ret}&gain_db=-3",
        "/api/device_param?device_type=comp&param=ratio&value=3",       # occurrence absent -> 0
    ])
    assert r["applied"] == 4 and r["failed"] == 0, r
    s = _state(seeded)
    assert s["stems"]["stem_a"]["gain_db"] == -4.5
    comp = next(e for e in s["master_chain"] if e["type"] == "comp")
    assert comp["threshold_db"] == -21 and comp["ratio"] == 3
    assert s["returns"][ret]["gain_db"] == -3


# ---- F7 ----------------------------------------------------------------------------------------
def test_the_pass_limit_reaches_the_run(demo_project: Path, tmp_path: Path):
    import shutil
    other = tmp_path / "second"
    shutil.copytree(demo_project, other)
    full = daw_bridge.run_agent_pass(demo_project, "master for streaming")
    one = daw_bridge.run_agent_pass(other, "master for streaming", max_passes=1)
    assert full["passes"] >= 2, "vacuity: the unlimited run must take more than one pass"
    assert one["passes"] == 1


def test_the_pass_limit_can_only_lower_the_budget(demo_project: Path, monkeypatch):
    from orchestrator import loop
    seen = {}

    def fake_run_session(project_dir, brief, **kw):
        seen.update(kw)
        raise RuntimeError("stop here")
    monkeypatch.setattr(loop, "run_session", fake_run_session)
    for asked, want in ((None, loop.MAX_PASSES), (0, loop.MAX_PASSES), (1, 1), (3, 3), (999, loop.MAX_PASSES), (-4, 1)):
        with pytest.raises(RuntimeError):
            daw_bridge.run_agent_pass(demo_project, "master for streaming", max_passes=asked)
        assert seen["max_passes"] == want, (asked, seen["max_passes"])


def test_the_daw_does_not_offer_gating():
    for during_run in (False, True):
        for d in daw_tool_defs(during_run):
            if d["name"] == "start_run":
                props = d["input_schema"]["properties"]
                assert "gating" not in props and "max_passes" in props


# ---- F8 ----------------------------------------------------------------------------------------
def test_ci_runs_every_flag():
    yml = (REPO / ".github/workflows/daw.yml").read_text()
    m = re.search(r"""for f in \$\(grep -o(E?) '([^']+)' daw/Source/Main\.cpp""", yml)
    assert m, "the console self-test loop moved - update this test with it"
    main = (REPO / "daw/Source/Main.cpp").read_text()
    ci = set(re.findall(m.group(2), main))
    every = set(re.findall(r'"--[a-z][a-z0-9-]*"', main))
    assert len(every) > 150
    assert every - ci == set(), f"CI's pattern cannot match: {sorted(every - ci)}"
