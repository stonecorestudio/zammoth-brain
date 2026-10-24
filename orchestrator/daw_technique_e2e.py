"""Live-DAW technique E2E (Phase 48 Track 3 — docs/05-roadmap.md).

Track 1 (`technique_eval`) proves the agent executes techniques well on the
OFFLINE Python engine. This proves the same techniques survive the whole live
stack: a natural-language brief goes over the **NDJSON socket** to the real
brain, the brain resolves a mix, the **native DAW** (`MixAgentDAW`) applies it
to the Tracktion graph and **bounces** it — and we assert three things per brief:

  * RESOLVED — the brain put the right device/routing in the staged
    ``session.json`` (the run mutates it on disk). This is Track 1's PLAN
    layer, but reached THROUGH the live socket + brain, not an in-process call.
  * REALIZED — the DAW dumped an ``EditSnapshot`` with the expected tracks
    (a group bus is a real folder track), and rendered per-stem wavs through
    the native chains.
  * AUDIBLE — the named stem's native bounce measurably differs from its raw
    input (the device did something), and the master bounce moved. Track 1
    owns exact-magnitude grading on the controlled fixture; here the fixture is
    real music (suno_demo — the stems the console DAW loads), so the honest
    E2E assertion is "the technique is audible", not "it moved band X by Y dB".

Two tiers, mirroring the other billed evals:

  * ``--check`` gates the DETERMINISTIC case only (the rule agent over the
    socket — free, no LLM, reproducible): brief in → brain resolved → DAW
    applied → bounce moved. A red here is the live stack broken.
  * the technique briefs need the real agent, so they run under ``--llm``
    (subscription by policy, like ``technique_eval``) and are reported, not
    gated — the SDK loop is non-deterministic.

    python -m orchestrator.daw_technique_e2e --list          # cases (free)
    python -m orchestrator.daw_technique_e2e --check         # det gate (free)
    python -m orchestrator.daw_technique_e2e --llm           # billed report
    python -m orchestrator.daw_technique_e2e --llm --case deess_vocals

Exits 0 PASS / 1 FAIL / 2 SKIP (DAW binary not built — build it with the `daw`
skill) / 3 NOTHING RAN (no case in the selection, or `--check` over a selection with no
deterministic case). 3 is not a skip: a caller that maps 2 to "skipped, fine" must go
red on it (reader E7·R17).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf

_REPO = Path(__file__).resolve().parents[0].parent
_PROJECT = _REPO / "test_projects" / "suno_demo"
_DAW_BIN = (_REPO / "daw" / "build.noindex" / "MixAgentDAW_artefacts"
            / "Release" / "MixAgentDAW")

#: "There is a binary and nothing was run against it." It shared exit code 2 with "there is
#: no binary" until 2026-10-07, and daw.yml maps 2 to `e2e SKIPPED`, exit 0 - in a step that
#: only runs after a successful build. Removing the deterministic case, or flipping it to
#: `llm=True`, would have turned the gate into a green no-op.
EXIT_NOTHING_RAN = 3

#: How much a processed stem must diverge from its raw input to count as
#: "the device is audible" — an RMS-band delta in dB, small but unambiguous.
AUDIBLE_DB = 0.3


# --------------------------------------------------------------------------
# session.json / snapshot readers (the brain's dict-or-list shapes)
# --------------------------------------------------------------------------

def _stems(state: dict) -> dict:
    st = state.get("stems", {})
    return st if isinstance(st, dict) else {s["stem_id"]: s for s in st}


def _chain_types(state: dict, stem: str) -> list[str]:
    s = _stems(state).get(stem, {})
    return [d.get("type") for d in s.get("chain", [])]


def _returns(state: dict) -> dict:
    r = state.get("returns", {})
    return r if isinstance(r, dict) else {x.get("id", x.get("return_id")): x for x in r}


def _buses(state: dict) -> dict:
    b = state.get("buses", {}) or state.get("busses", {})
    return b if isinstance(b, dict) else {x.get("id", x.get("bus_id")): x for x in b}


def _bus_members(state: dict, bus_id: str) -> list[str]:
    """Membership is derived — a stem belongs to a bus when its `bus` field
    names it (session.py bus_members); the bus dict carries no member list."""
    return [sid for sid, s in _stems(state).items() if s.get("bus") == bus_id]


def _send_sources(state: dict) -> dict[str, dict]:
    """{stem_id: {return_id: gain_db}} from the raw session (sends live on stems)."""
    return {sid: s.get("sends") for sid, s in _stems(state).items() if s.get("sends")}


# --------------------------------------------------------------------------
# audio measurement — band-limited RMS, deterministic
# --------------------------------------------------------------------------

def _mono(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), always_2d=True)
    return x.mean(axis=1).astype(np.float64), sr


def _band_rms_db(x: np.ndarray, sr: int, lo: float, hi: float) -> float:
    from scipy.signal import butter, sosfilt
    ny = sr / 2.0
    lo, hi = max(1.0, lo), min(hi, ny * 0.999)
    sos = butter(4, [lo / ny, hi / ny], btype="band", output="sos")
    y = sosfilt(sos, x)
    return float(20.0 * np.log10(max(np.sqrt(np.mean(y ** 2)), 1e-9)))


def _rms_db(x: np.ndarray) -> float:
    return float(20.0 * np.log10(max(np.sqrt(np.mean(x ** 2)), 1e-9)))


# --------------------------------------------------------------------------
# the cases
# --------------------------------------------------------------------------

@dataclass
class E2ECase:
    key: str
    brief: str
    #: rule agent (deterministic, free) or the LLM (billed)
    llm: bool = True
    #: (stem, {action types, any one}) that must appear in the resolved session
    resolved_device: Optional[tuple] = None
    #: the named stem must feed at least one return in the resolved session
    resolved_send_from: str = ""
    #: a bus must exist whose members include all of these
    resolved_bus_members: tuple[str, ...] = ()
    #: the stem whose native bounce must differ from its raw input (audible)
    audible_stem: str = ""
    #: assert on the MASTER bounce instead (dry vs mixed RMS). A send/return
    #: technique is INVISIBLE in a solo-stem render by design: native_returns
    #: sends the stem chain back DRY and the wet lives on the return track,
    #: which only sums into the master (found 2026-08-18 — the reverb case
    #: resolved perfectly and scored 0.14 dB on the solo stem).
    audible_master: bool = False
    #: extra note for --list
    note: str = ""


CASES: list[E2ECase] = [
    # -- the deterministic gate: the whole live stack, no LLM --
    E2ECase("master_streaming_rule", "master for streaming", llm=False,
            audible_master=True,
            note="rule agent over the socket: brain resolved -> DAW applied -> "
                 "the MASTER bounce moved. (It was asserted on the solo vocals "
                 "render until 2026-08-19 — which only ever passed because of "
                 "accumulated fixture junk: a master-only rule mix is invisible "
                 "on a masterOff stem render. The harness-measures-itself "
                 "class, again.)"),
    # -- billed technique briefs (real agent, reported) --
    E2ECase("deess_vocals", "de-ess the vocals, they're harsh on the esses",
            resolved_device=("vocals", ("deess", "dynamic_eq")),
            audible_stem="vocals",
            note="a device technique: the brain must resolve a de-esser on the "
                 "vocals and the DAW must render it audibly"),
    E2ECase("saturate_bass", "warm up the bass with some tape saturation",
            resolved_device=("bass", ("saturation",)),
            audible_stem="bass"),
    E2ECase("reverb_send_vocals",
            "give the vocals a sense of space with a reverb send",
            resolved_send_from="vocals", audible_master=True,
            note="a routing technique: a return + a send from vocals; audible "
                 "on the MASTER (the wet lives on the return track, so a solo "
                 "stem render cannot hear it)"),
    E2ECase("glue_rhythm",
            "glue the drums and bass together on a bus with gentle compression",
            resolved_bus_members=("drums", "bass"),
            note="a bus technique: the group bus is realized (a folder track "
                 "in the snapshot) with both members"),
]


# --------------------------------------------------------------------------
# driving one case through the real DAW
# --------------------------------------------------------------------------

@dataclass
class Result:
    case: E2ECase
    passed: bool
    checks: list[tuple[str, bool, str]] = field(default_factory=list)
    error: str = ""


def _run_case(case: E2ECase, daw_bin: Path, timeout_s: int = 1800) -> Result:
    checks: list[tuple[str, bool, str]] = []

    def chk(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    with tempfile.TemporaryDirectory() as td:
        proj = Path(td) / "proj"
        shutil.copytree(_PROJECT, proj)
        # drop the reference master.wav so it is not treated as a fifth stem
        try:
            st = json.loads((proj / "session.json").read_text())
            if isinstance(st.get("stems"), dict):
                st["stems"].pop("master", None)
                (proj / "session.json").write_text(json.dumps(st))
        except Exception:
            pass
        (proj / "master.wav").unlink(missing_ok=True)

        snap = Path(td) / "snap.json"
        stems = Path(td) / "stems"
        env = {**os.environ,
               "MIXAGENT_BRAIN_BRIEF": case.brief,
               "MIXAGENT_BRAIN_SNAPSHOT": str(snap),
               "MIXAGENT_BRAIN_STEMS": str(stems),
               # isolate the smoke's dry/mixed master bounces where we can read
               # them (consoleTempDir = $TMPDIR/MixAgentDAW-console)
               "TMPDIR": td}
        mode = "--brain-llm" if case.llm else "--brain"
        try:
            res = subprocess.run([str(daw_bin), mode, str(proj)], env=env,
                                 capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return Result(case, False, error=f"DAW timed out after {timeout_s}s")

        pass_line = "PASS - LIVE brain bridge" in res.stdout
        chk("live socket run", pass_line,
            "" if pass_line else res.stdout.strip().splitlines()[-1:] and
            res.stdout.strip().splitlines()[-1])
        if not pass_line:
            return Result(case, False, checks, error="live run did not PASS")

        # Phase 49: the DAW smoke stages its own temp copy of the project and
        # the brain mutates THAT (never the fixture) — it prints the staged
        # path as BRAIN_PROJECT=. Read the resolved session from there.
        staged = proj
        for line in res.stdout.splitlines():
            if line.startswith("BRAIN_PROJECT="):
                staged = Path(line.split("=", 1)[1].strip())
                break
        try:
            state = json.loads((staged / "session.json").read_text())
        except Exception as exc:
            return Result(case, False, checks, error=f"no session.json: {exc}")

        # RESOLVED — the brain put the right thing in the session
        if case.resolved_device:
            stem, types = case.resolved_device
            got = _chain_types(state, stem)
            chk(f"resolved {'/'.join(types)} on {stem}",
                any(t in types for t in got), f"chain={got}")
        if case.resolved_send_from:
            sends = _send_sources(state)
            rets = _returns(state)
            # on a miss, say what the agent did INSTEAD — the named stem's chain
            # and the run's stop_reason/moves (from the DAW's captured stdout) —
            # so a red case is diagnosable without a second billed run.
            why = (f"returns={list(rets)} sends={list(sends)} "
                   f"{case.resolved_send_from}.chain="
                   f"{_chain_types(state, case.resolved_send_from)}")
            for line in reversed(res.stdout.splitlines()):
                if '"stop_reason"' in line:
                    why += f" | {line.strip()[:120]}"
                    break
            chk(f"resolved a return + send from {case.resolved_send_from}",
                bool(rets) and case.resolved_send_from in sends, why)
        if case.resolved_bus_members:
            buses = _buses(state)
            membership = {bid: _bus_members(state, bid) for bid in buses}
            hit = any(all(m in members for m in case.resolved_bus_members)
                      for members in membership.values())
            chk(f"resolved a bus with {', '.join(case.resolved_bus_members)}",
                hit, f"membership={membership}")
            # REALIZED — the bus is a real track in the DAW snapshot (a folder
            # track, or simply a track beyond the four stems)
            if snap.is_file():
                snap_j = json.loads(snap.read_text())
                kinds = [t.get("kind", "") for t in snap_j.get("tracks", [])]
                chk("realized the bus as a snapshot track",
                    any(k in ("bus", "folder", "group") for k in kinds)
                    or len(snap_j.get("tracks", [])) > 4,
                    f"tracks={len(snap_j.get('tracks', []))} kinds={sorted(set(kinds))}")

        # AUDIBLE (master) — dry vs brain-mixed master bounce, full-band RMS
        if case.audible_master:
            console = Path(td) / "MixAgentDAW-console"
            dry, mixed = console / "MixAgentDAW_dry.wav", console / "MixAgentDAW_brain.wav"
            if dry.is_file() and mixed.is_file():
                dx, _ = _mono(dry)
                mx, _ = _mono(mixed)
                n = min(len(dx), len(mx))
                # compare on the longer tail too: a reverb adds energy where the
                # dry mix decays, so equal-length full RMS still catches it
                moved = abs(_rms_db(mx[:n]) - _rms_db(dx[:n]))
                chk("master bounce is audibly changed", moved >= AUDIBLE_DB,
                    f"|ΔRMS| {moved:.2f} dB dry->mixed")
            else:
                chk("master bounces present", False,
                    f"dry={dry.is_file()} mixed={mixed.is_file()}")

        # AUDIBLE — the processed stem differs from its raw input
        if case.audible_stem:
            raw = proj / f"{case.audible_stem}.wav"
            proc = stems / f"{case.audible_stem}.wav"
            if raw.is_file() and proc.is_file():
                rx, rsr = _mono(raw)
                px, psr = _mono(proc)
                n = min(len(rx), len(px))
                d_full = abs(_rms_db(px[:n]) - _rms_db(rx[:n]))
                # widest informative band the device could move
                d_band = abs(_band_rms_db(px[:n], psr, 2000, min(12000, psr / 2))
                             - _band_rms_db(rx[:n], rsr, 2000, min(12000, rsr / 2)))
                moved = max(d_full, d_band)
                chk(f"{case.audible_stem} bounce is audibly changed",
                    moved >= AUDIBLE_DB, f"max |ΔRMS| {moved:.2f} dB "
                    f"(full {d_full:.2f}, 2-12k {d_band:.2f})")
            else:
                chk(f"{case.audible_stem} bounce present", False,
                    f"raw={raw.is_file()} processed={proc.is_file()}")

    passed = all(ok for _, ok, _ in checks) and bool(checks)
    return Result(case, passed, checks)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _print(result: Result) -> None:
    mark = "PASS" if result.passed else "FAIL"
    print(f"  {result.case.key:24s} {mark}")
    for name, ok, detail in result.checks:
        print(f"      {'ok ' if ok else 'XX '} {name}"
              + (f"   [{detail}]" if detail and not ok else ""))
    if result.error:
        print(f"      error: {result.error}")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Live-DAW technique E2E (Phase 48 Track 3)")
    ap.add_argument("--list", action="store_true", help="print the cases (free)")
    ap.add_argument("--llm", action="store_true",
                    help="run the billed LLM technique briefs (subscription by "
                         "policy); without it, only the deterministic case runs")
    ap.add_argument("--case", action="append", help="run only this case key")
    ap.add_argument("--check", action="store_true",
                    help="gate: the deterministic case must PASS (exit 1 if not)")
    ap.add_argument("--daw-bin", type=Path, default=_DAW_BIN)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args(argv)

    if args.list:
        for c in CASES:
            tier = "LLM " if c.llm else "rule"
            print(f"  [{tier}] {c.key:24s} {c.brief!r}")
            if c.note:
                print(f"         {c.note}")
        print(f"\n{len(CASES)} cases "
              f"({sum(1 for c in CASES if not c.llm)} deterministic, "
              f"{sum(1 for c in CASES if c.llm)} billed)")
        return 0

    if not args.daw_bin.exists():
        print(f"[e2e] SKIP — DAW binary not built: {args.daw_bin}\n"
              f"      build it with the `daw` skill "
              f"(cmake --build daw/build.noindex --target MixAgentDAW)",
              file=sys.stderr)
        return 2

    keys = set(args.case) if args.case else None
    cases = [c for c in CASES if (keys and c.key in keys)
             or (not keys and (args.llm or not c.llm))]
    if not cases:
        print("no cases selected (billed briefs need --llm)", file=sys.stderr)
        return EXIT_NOTHING_RAN
    if args.check and all(c.llm for c in cases):
        # Said BEFORE anything runs: the selection holds only billed briefs, so the gate
        # would spend them and then have nothing to judge (it used to, and answered 2).
        print("\nFAIL — --check needs at least one deterministic case in the selection; "
              "none would run", file=sys.stderr)
        return EXIT_NOTHING_RAN

    if args.llm or any(c.llm for c in cases):
        os.environ.setdefault("MIX_AGENT_LLM_BACKEND", "subscription")

    print(f"[e2e] {len(cases)} case(s) through {args.daw_bin.name} "
          f"(backend={os.environ.get('MIX_AGENT_LLM_BACKEND', 'rule/none')})")
    results = [_run_case(c, args.daw_bin, args.timeout) for c in cases]
    print()
    for r in results:
        _print(r)

    det = [r for r in results if not r.case.llm]
    passed = sum(1 for r in results if r.passed)
    print(f"\n{passed}/{len(results)} cases passed")

    if args.check:
        if not det:
            # `--check --case <llm-case>` used to print PASS having run no gated case at all
            # (an empty failure list is not a passing gate) — 2026-09-01 sweep.
            print("\nFAIL — --check needs at least one deterministic case in the selection; "
                  "none ran", file=sys.stderr)
            return EXIT_NOTHING_RAN
        det_fail = [r for r in det if not r.passed]
        if det_fail:
            print("\nFAIL — the deterministic live-stack gate is red: "
                  + ", ".join(r.case.key for r in det_fail), file=sys.stderr)
            return 1
        print("PASS — the deterministic live stack works end to end")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
