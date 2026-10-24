"""The sound bench: render WOOLLY sounds in the real app and read them (Phase 60).

    python -m orchestrator.sound_bench spec.json            # a table, one line a sound
    python -m orchestrator.sound_bench --map                # the synth's knobs, ranges and INIT values
    python -m orchestrator.sound_bench --bank               # every factory preset, against its class

The instrument the factory bank and the sound-design eval's goldens were made with. It hands a
list of sounds - `sound.render`'s own grammar: `{name, preset | patch, notes?, bpm?, tail_s?}` -
to the console binary's `--sound-bench`, which renders them through the real command on the
real component (a private engine, nothing kept), then measures each file with
`sample_profile` and, when a sound says what it is meant to be (`kind`), holds it to that
class's targets (`mix_engine.sound_classes`).

Free: no model, no network. Needs the console built (exit 2 when it is not).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[1]
BINARY = REPO / "daw" / "build.noindex" / "MixAgentDAW_artefacts" / "Release" / "MixAgentDAW"
STEMS = REPO / "test_projects" / "suno_demo"
EXIT_NOT_BUILT = 2

# Fields of a bench sound that are the bench's own, not sound.render's.
OWN_FIELDS = ("kind", "expect")


class BenchError(RuntimeError):
    pass


def built() -> bool:
    return BINARY.exists()


def _engine():
    server = str(REPO / "server")
    if server not in sys.path:
        sys.path.insert(0, server)
    import soundfile as sf
    from mix_engine import sound_classes
    from mix_engine.sample_profile import sample_profile
    return sf, sample_profile, sound_classes


def run_console(sounds: list[dict], out_dir: Path, timeout_s: float = 900.0) -> dict:
    """The console's half: the files and `bench.json` in `out_dir`. Returns the parsed json."""
    if not built():
        raise BenchError(f"the console is not built ({BINARY})")
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = out_dir / "spec.json"
    spec.write_text(json.dumps({"sounds": [{k: v for k, v in s.items() if k not in OWN_FIELDS}
                                           for s in sounds]}))
    env = dict(os.environ, ZAM_BENCH_SPEC=str(spec), ZAM_BENCH_OUT=str(out_dir))
    done = subprocess.run([str(BINARY), "--sound-bench", str(STEMS)], cwd=str(REPO), env=env,
                          capture_output=True, text=True, timeout=timeout_s)
    result = out_dir / "bench.json"
    if done.returncode != 0 or not result.exists():
        raise BenchError(f"the bench exited {done.returncode}: {done.stdout[-600:]} {done.stderr[-300:]}")
    return json.loads(result.read_text())


def measure(sounds: list[dict], rows: list[dict]) -> list[dict]:
    """Each row with its `profile` (a dict) and, for a sound with a `kind`, its `issues`."""
    sf, sample_profile, sound_classes = _engine()
    out = []
    for asked, row in zip(sounds, rows):
        row = dict(row)
        path = row.get("file")
        if path and os.path.exists(path):
            audio, rate = sf.read(path, always_2d=True, dtype="float32")
            loop = asked.get("loop_beats")
            profile = sample_profile(audio, rate, kind="loop" if loop else "one_shot",
                                     bpm=asked.get("bpm") if loop else None, beats=loop)
            row["profile"] = profile.model_dump(exclude_none=True)
            kind = asked.get("kind")
            if kind:
                row["kind"] = kind
                row["issues"] = sound_classes.issues(row["profile"], kind,
                                                     played_midi=_played(asked))
        out.append(row)
    return out


def _played(asked: dict) -> Optional[int]:
    notes = asked.get("notes")
    if not notes:
        return 48                                   # sound.render's own default: one C3
    return int(notes[0]["pitch"]) if len(notes) == 1 else None


def render(sounds: list[dict], out_dir: Optional[Path] = None, keep: bool = False) -> list[dict]:
    """Render and measure. With no `out_dir` the files go to a temp folder that is removed
    (unless `keep`), and each row's `file` is then gone: read the profile."""
    own = out_dir is None
    folder = Path(tempfile.mkdtemp(prefix="sound-bench-")) if own else out_dir
    try:
        rows = measure(sounds, run_console(sounds, folder)["sounds"])
    finally:
        if own and not keep:
            shutil.rmtree(folder, ignore_errors=True)
    return rows


def synth_map() -> dict:
    folder = Path(tempfile.mkdtemp(prefix="sound-bench-"))
    try:
        return run_console([], folder)["synth_map"]
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def line(row: dict) -> str:
    if row.get("error"):
        return f"{row.get('name', '?'):<22} REFUSED  {row['error'][:150]}"
    p: dict[str, Any] = row.get("profile") or {}
    bands = sorted((p.get("band_share_pct") or {}).items(), key=lambda kv: -kv[1])[:2]
    pitch = p.get("pitch_note") or "-"
    if p.get("pitch_drop_semitones") is not None:
        pitch += f" drop {p['pitch_drop_semitones']} in {p.get('pitch_settle_ms')}ms"
    text = (f"{row.get('name', '?'):<22} {p.get('sounding_s', 0):5.2f}s att {p.get('attack_ms')!s:>6} "
            f"dec {p.get('decay_ms')!s:>7} cen {p.get('centroid_hz')!s:>7} {pitch:<24} "
            f"noisy {p.get('noisiness')!s:>4} peak {p.get('sample_peak_dbfs')!s:>6} "
            f"lufs {p.get('loudness_lufs')!s:>6} w {p.get('width_db')!s:>6} "
            + " ".join(f"{k} {v:.0f}" for k, v in bands))
    for issue in row.get("issues") or []:
        text += f"\n{'':<22}   ! {issue['says']}"
    return text


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("spec", nargs="?", help="a json file: {sounds: [...]} or a bare list")
    parser.add_argument("--map", action="store_true", help="print the synth's knobs and ranges")
    parser.add_argument("--bank", action="store_true", help="render every factory preset against its class")
    parser.add_argument("--out", help="keep the files in this folder")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if not built():
        print(f"sound_bench: the console is not built ({BINARY})")
        return EXIT_NOT_BUILT
    if args.map:
        print(json.dumps(synth_map(), indent=1))
        return 0
    if args.bank:
        from orchestrator import sound_bank
        sounds = sound_bank.bench_sounds()
    elif args.spec:
        loaded = json.loads(Path(args.spec).read_text())
        sounds = loaded["sounds"] if isinstance(loaded, dict) else loaded
    else:
        parser.error("give a spec file, --map or --bank")
    rows = render(sounds, Path(args.out).resolve() if args.out else None)
    if args.json:
        print(json.dumps(rows, indent=1))
    else:
        for row in rows:
            print(line(row))
    return 1 if any(r.get("error") for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
