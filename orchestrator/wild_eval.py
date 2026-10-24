"""Wild-corpus property harness (Phase 50 — docs/05-roadmap.md).

Every project in `wild_fixture.PROJECTS` is driven through the whole offline
product path, and each leg asserts a PROPERTY of the product, not a tuned
number:

  stage     — the corpus is synthesised into the temp work folder and each project is
              copied from there; nothing under test_projects/ is read or written (it was
              rmtree'd and rewritten on every run until 2026-10-07 - reader E7·R14).
  discover  — `Session()` adopts exactly the stems on disk - less the files the
              project says must be LEFT OUT and named (master_trap's master.wav) -
              or raises the project's defined error; anything else is a fail.
  gestures  — user co-edits over the real tool surface (`set_stem`: gain,
              pan, mute, width — the offline user channel). They must LAND: the
              session holds each value, and the gestured stems' own renders moved.
  baseline  — R1 = a full master render. Non-silent.
  run       — the deterministic RuleAgent mixes "master for streaming".
              At least one proposal must APPLY (a mix run that does nothing
              on a valid project is a finding).
  audible   — R2 (post-run render) differs from R1.
  export    — the engine's own `export_mix` prints the master; the file read back is
              the post-run render held to full scale (never wrapped, so non-silent
              as the render is), and its `clipped` flag is true exactly when the render
              passed full scale.
  undo      — every applied proposal undone in reverse order; R3 is
              BIT-IDENTICAL to R1 (the engine's undo guarantee, property-
              tested against hostile projects — user gestures survive).
  bounded   — the whole pipeline stays under the project's time budget.

Free and deterministic: no LLM anywhere (RuleAgent only), so this can run in
CI and gate. Exit 0 = every project green; 1 = failures (each printed with
its leg and detail); the corpus is the source of truth for expectations.

    python -m orchestrator.wild_eval             # the full sweep
    python -m orchestrator.wild_eval --project unicode_names
    python -m orchestrator.wild_eval --list
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import anyio
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "server"))

from .wild_fixture import PROJECTS, WildSpec, generate  # noqa: E402

#: seconds per project before the pipeline counts as hung
TIME_BUDGET_S = {"long_6min": 600.0, "hires_96k_f32": 420.0,
                 "many_stems_30": 420.0}
DEFAULT_BUDGET_S = 240.0

#: the scripted user gestures (indices into the project's sorted stem ids)
GESTURES = ((0, {"gain_db": 2.0}), (1, {"pan": -0.4}),
            (1, {"width": 1.2}), (0, {"muted": False}))


@dataclass
class Leg:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class ProjectResult:
    name: str
    legs: list[Leg] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.legs) and all(l.ok for l in self.legs)


def _render(proj: Path) -> tuple[np.ndarray, int]:
    """A fresh Session each time — renders exactly what session.json says."""
    from mix_engine import dsp
    from mix_engine.session import Session
    audio, sr = dsp.render_master(Session(proj))
    return np.asarray(audio, dtype=np.float64), sr


def _rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(max(float(np.sqrt(np.mean(x ** 2))), 1e-12)))


async def _gestures(proj: Path, stem_ids: list[str]) -> None:
    from orchestrator.benchmark import in_memory_engine
    from orchestrator.loop import EngineClient
    async with in_memory_engine(proj) as client:
        engine = EngineClient(client)
        for idx, changes in GESTURES:
            await engine.call("set_stem",
                              {"stem_id": stem_ids[idx % len(stem_ids)],
                               **changes})


def _stem_renders(proj: Path, stem_ids: list[str]) -> dict[str, np.ndarray]:
    """The gestured stems' own renders, from what session.json says now."""
    from mix_engine import dsp
    from mix_engine.session import Session
    session = Session(proj)
    return {sid: np.asarray(dsp.render_stem(session, sid)[0], dtype=np.float64)
            for sid in stem_ids}


def _gestures_landed(proj: Path, stem_ids: list[str],
                     before: dict[str, np.ndarray]) -> str:
    """"" when every scripted gesture is in the session and audible, else what is missing.

    The leg used to be `did not raise` (reader E7·R14): with `Session.user_edit` a no-op
    every leg stayed green, the undo leg included - it compares against a baseline rendered
    AFTER the gestures, so "user gestures survive" held only if they had landed."""
    from mix_engine.session import Session
    stems = Session(proj).stems
    for idx, changes in GESTURES:
        sid = stem_ids[idx % len(stem_ids)]
        for name, want in changes.items():
            got = stems[sid].get(name)
            if got != want:
                return f"{sid}.{name} is {got!r}, the gesture set {want!r}"
    after = _stem_renders(proj, list(before))
    for sid, was in before.items():
        if _rms_db(was) <= -70.0:
            continue                      # a silent stem cannot show a fader move
        if after[sid].shape == was.shape and np.array_equal(after[sid], was):
            return f"{sid} renders exactly as before its gestures"
    return ""


async def _export(proj: Path, out_dir: Path) -> dict:
    from orchestrator.benchmark import in_memory_engine
    from orchestrator.loop import EngineClient
    async with in_memory_engine(proj) as client:
        return await EngineClient(client).call(
            "export_mix", {"req": {"out_dir": str(out_dir), "format": "wav24"}})


#: 24-bit PCM's top code, as the engine holds a render to it (mcp_server.PCM_FULL_SCALE).
_PCM24_FULL_SCALE = 8388607 / 8388608


def _export_verdict(result: dict, render: np.ndarray, sr: int) -> tuple[bool, str]:
    """Read the master the ENGINE wrote back and hold it to the render it was made from."""
    import soundfile as sf
    master = next((f for f in result.get("files", []) if f.get("kind") == "master"), None)
    if master is None:
        return False, f"export_mix named no master: {result}"
    path = Path(master["path"])
    audio, file_sr = sf.read(str(path), dtype="float64", always_2d=True)
    want = np.clip(render if render.ndim == 2 else render[:, None], -1.0, _PCM24_FULL_SCALE)
    name = path.name
    if file_sr != sr or audio.shape != want.shape:
        return False, f"{name}: {audio.shape} @ {file_sr}, the render is {want.shape} @ {sr}"
    # (non-silent follows: the render is, by the `audible` leg, and the file is the render)
    worst = float(np.max(np.abs(audio - want)))
    if worst > 2.0 ** -22:                # one 24-bit step is 2^-23
        return False, f"{name}: differs from the render held to full scale by {worst:.2e}"
    hot = float(np.max(np.abs(render))) > _PCM24_FULL_SCALE
    if bool(master.get("clipped")) != hot:
        return False, f"{name}: clipped={master.get('clipped')} but the render " \
                      f"{'passed' if hot else 'stayed under'} full scale"
    return True, f"{name}: {path.stat().st_size} bytes, rms {_rms_db(audio):.1f} dBFS" \
                 + (", clipped and said so" if hot else "")


async def _undo_all(proj: Path) -> int:
    from orchestrator.benchmark import in_memory_engine
    from orchestrator.loop import EngineClient
    state = json.loads((proj / "session.json").read_text())
    applied = [p["proposal_id"] for p in state.get("proposals", [])
               if p.get("status") == "applied"]
    async with in_memory_engine(proj) as client:
        engine = EngineClient(client)
        for pid in reversed(applied):
            await engine.call("undo_proposal", {"proposal_id": pid})
    return len(applied)


def run_project(spec: WildSpec, work: Path, source: Path) -> ProjectResult:
    """`source` holds the generated corpus (a folder per project) - the run's own copy."""
    from mix_engine.session import Session
    from orchestrator.agents import RuleAgent
    from orchestrator.benchmark import in_memory_engine
    from orchestrator.loop import run_session

    r = ProjectResult(spec.name)
    leg = lambda n, ok, d="": r.legs.append(Leg(n, bool(ok), d))  # noqa: E731
    started = time.monotonic()
    budget = TIME_BUDGET_S.get(spec.name, DEFAULT_BUDGET_S)

    proj = work / spec.name
    shutil.copytree(source / spec.name, proj)

    # -- discover ----------------------------------------------------------
    expected = sorted(s.name for s in spec.stems)
    try:
        session = Session(proj)
    except Exception as exc:
        if spec.expect_error and spec.expect_error in str(exc) \
                and not isinstance(exc, (MemoryError, OSError)) \
                and type(exc) is not Exception:
            leg("discover", True, f"raised the defined error: {exc}")
            return r   # the project's whole story IS the clean refusal
        leg("discover", False,
            f"{type(exc).__name__}: {str(exc)[:160]}\n{traceback.format_exc(limit=3)}")
        return r
    if spec.expect_error:
        leg("discover", False,
            f"expected the defined error ({spec.expect_error!r}), got none")
        return r
    left_out = sorted(spec.left_out)
    expected = sorted(n for n in expected if f"{n}.wav" not in left_out)
    got = sorted(session.stems.keys())
    named = sorted(getattr(session, "left_out_wavs", []))
    found = got == expected and named == left_out
    leg("discover", found,
        "" if found else f"expected {expected} with {left_out} left out, "
                         f"got {got} with {named} left out")
    if not found:
        return r

    # -- gestures + baseline ----------------------------------------------
    try:
        gestured = sorted({got[idx % len(got)] for idx, _ in GESTURES})
        ungestured = _stem_renders(proj, gestured)
        anyio.run(_gestures, proj, got)
        missing = _gestures_landed(proj, got, ungestured)
        leg("gestures", not missing, missing)
        if missing:
            return r
    except Exception as exc:
        leg("gestures", False, f"{type(exc).__name__}: {str(exc)[:200]}")
        return r
    try:
        r1, sr1 = _render(proj)
        leg("baseline", _rms_db(r1) > -70.0,
            f"R1 rms {_rms_db(r1):.1f} dBFS @ {sr1}")
    except Exception as exc:
        leg("baseline", False, f"{type(exc).__name__}: {str(exc)[:200]}")
        return r

    # -- the rule run -------------------------------------------------------
    try:
        report = run_session(proj, "master for streaming", gating="auto",
                             agent=RuleAgent(), max_passes=2,
                             connect=in_memory_engine)
        applied = report.applied_proposal_ids
        leg("run", len(applied) >= 1,
            f"stop={report.stop_reason} applied={len(applied)}"
            + (f" warnings={report.warnings[:1]}" if report.warnings else ""))
    except Exception as exc:
        leg("run", False,
            f"{type(exc).__name__}: {str(exc)[:200]}\n{traceback.format_exc(limit=3)}")
        return r

    # -- audible + export ---------------------------------------------------
    try:
        r2, _ = _render(proj)
        n = min(len(r1), len(r2))
        changed = not np.array_equal(r1[:n], r2[:n])
        leg("audible", changed and _rms_db(r2) > -70.0,
            f"R2 rms {_rms_db(r2):.1f} dBFS, changed={changed}")
    except Exception as exc:
        leg("audible", False, f"{type(exc).__name__}: {str(exc)[:200]}")
        return r
    try:
        # The ENGINE's export, into a folder outside the project. This leg used to
        # sf.write the harness's own array and check the file's size (reader E7·R14): it
        # tested soundfile, and pathological_audio's clipping stem never met the path that
        # holds a hot render to full scale before the PCM conversion.
        exported = anyio.run(_export, proj, work / f"{spec.name}_export")
        leg("export", *_export_verdict(exported, r2, sr1))
    except Exception as exc:
        leg("export", False, f"{type(exc).__name__}: {str(exc)[:200]}")
        return r

    # -- undo: bit-identical restore ---------------------------------------
    try:
        undone = anyio.run(_undo_all, proj)
        r3, _ = _render(proj)
        identical = r3.shape == r1.shape and np.array_equal(r3, r1)
        leg("undo", identical,
            f"undid {undone}; restore {'bit-identical' if identical else 'DIVERGED'}"
            + ("" if identical else
               f" (max |Δ| {float(np.max(np.abs(r3[:min(len(r3), len(r1))] - r1[:min(len(r3), len(r1))]))):.2e},"
               f" shapes {r3.shape} vs {r1.shape})"))
    except Exception as exc:
        leg("undo", False, f"{type(exc).__name__}: {str(exc)[:200]}")

    took = time.monotonic() - started
    leg("bounded", took <= budget, f"{took:.1f}s of {budget:.0f}s")
    return r


_DAW_BIN = _REPO / "daw" / "build.noindex" / "MixAgentDAW_artefacts" / "Release" / "MixAgentDAW"


def run_daw_leg(spec: WildSpec, work: Path, source: Path) -> ProjectResult:
    """Criterion 5: the console DAW loads the project and bounces it without
    a crash. Runs on a fresh staged copy (the binary must not mutate the
    corpus). A non-zero exit, a crash, or a FAIL line is a finding."""
    import subprocess
    r = ProjectResult(spec.name)
    if spec.expect_error:
        r.legs.append(Leg("daw", True, "skipped: the defined-error project"))
        return r
    proj = work / f"daw_{spec.name}"
    shutil.copytree(source / spec.name, proj)
    try:
        res = subprocess.run([str(_DAW_BIN), str(proj)], capture_output=True,
                             text=True, timeout=300)
    except subprocess.TimeoutExpired:
        r.legs.append(Leg("daw", False, "console render smoke timed out (300s)"))
        return r
    verdicts = [l for l in res.stdout.splitlines() if "PASS" in l or "FAIL" in l
                or "ERROR" in l]
    # The docstring promised "a FAIL line is a finding", but the check used to look at the
    # last two verdict lines only and never at FAIL at all — a leg that printed FAIL and then
    # PASS, or FAIL early in a multi-verdict bounce, read as green (2026-09-01 sweep, P1). Same
    # three-part rule as CI: exit 0, a PASS, and no FAIL/ERROR anywhere in the output.
    failed = [l for l in verdicts if "FAIL" in l or "ERROR" in l]
    ok = res.returncode == 0 and any("PASS" in l for l in verdicts) and not failed
    shown = (failed or verdicts)[-2:]
    r.legs.append(Leg("daw", ok, f"exit={res.returncode} " + " | ".join(shown)))
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--project", action="append")
    ap.add_argument("--keep", action="store_true",
                    help="keep the work dir for post-mortem")
    ap.add_argument("--daw", action="store_true",
                    help="also drive each project through the console DAW "
                         "binary (load -> bounce; needs the daw build)")
    args = ap.parse_args(argv)

    specs = [s for s in PROJECTS
             if not args.project or s.name in set(args.project)]
    if args.list:
        for s in specs:
            print(f"  {s.name:20s} {len(s.stems):2d} stems"
                  + (f"  EXPECT ERROR: {s.expect_error}" if s.expect_error else ""))
        return 0

    work = Path(tempfile.mkdtemp(prefix="mix-agent-wild-"))
    # The corpus is made HERE, in the work folder - synthesis is deterministic. The repo's
    # test_projects/wild is written only by `python -m orchestrator.wild_fixture` itself.
    source = work / "_corpus"
    generate(source, only={s.name for s in specs})
    print(f"[wild] {len(specs)} project(s), work dir {work}")
    results = []
    if args.daw and not _DAW_BIN.exists():
        # Silently dropping the leg printed the same "N/N green" as a real sweep on a machine
        # that had not built the DAW (2026-09-01 sweep). Exit 2 = SKIP, as the sibling
        # harnesses (daw_export_oracle, daw_technique_e2e) already do.
        print(f"[wild] SKIP — --daw asked for the DAW leg but the binary is not built: {_DAW_BIN}",
              file=sys.stderr)
        return 2
    for spec in specs:
        res = run_project(spec, work, source)
        if args.daw:
            res.legs.extend(run_daw_leg(spec, work, source).legs)
        results.append(res)
        mark = "PASS" if res.passed else "FAIL"
        print(f"  {spec.name:20s} {mark}")
        for l in res.legs:
            print(f"      {'ok ' if l.ok else 'XX '} {l.name:9s} {l.detail}"
                  .rstrip())
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)

    failed = [r for r in results if not r.passed]
    print(f"\n{len(results) - len(failed)}/{len(results)} projects green")
    if failed:
        print("FAILURES: " + ", ".join(r.name for r in failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
