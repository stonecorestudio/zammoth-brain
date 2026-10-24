"""WOOLLY's factory bank, measured (Phase 60).

The bank is fifty-eight patch documents in `daw/Source/WoollyPresets.h`. Each names the KIND of
sound it is and the key it is written to be played on; this module renders every one of them
there (the sound bench: the real app, a private engine) and reads it against its kind
(`mix_engine.sound_classes`).

    python -m orchestrator.sound_bench --bank               # the table
    python -m orchestrator.sound_bank --check                # every preset clean against its kind
    python -m orchestrator.sound_bank --update-census        # ...and write what each measured as
    python -m orchestrator.sound_bank --audition <folder>    # the WAVs, to listen to

The census (`orchestrator/sound_bank_census.json`) is what each preset measured as when the bank
was last signed off. A pytest holds it to the header's names and re-reads every stored profile
against the class ranges, so a range that would turn a factory preset into an "issue" fails
without the app. Numbers are not ears: the census says each preset sits where its kind usually
sits - whether it sounds GOOD is a person's call (`--audition`).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

from orchestrator import sound_bench

REPO = Path(__file__).resolve().parents[1]
HEADER = REPO / "daw" / "Source" / "WoollyPresets.h"
CENSUS = Path(__file__).with_name("sound_bank_census.json")
BPM = 120


def header_names() -> list[str]:
    """The factory presets' names, read from the C++ table (no app needed)."""
    text = HEADER.read_text()
    table = text[text.index("inline const std::vector<Factory>& factory()"):text.index("inline const char* initRoutes()")]
    return re.findall(r'^\s*\{ "([A-Z0-9 ]+)", "[A-Z]*", "', table, flags=re.M)


def bank() -> list[dict]:
    """The bank as the app holds it: name, category, description, kind, note, beats, tags, patch."""
    folder = Path(tempfile.mkdtemp(prefix="sound-bank-"))
    try:
        return sound_bench.run_console([], folder)["factory_bank"]
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def bench_sounds(presets: Optional[list[dict]] = None) -> list[dict]:
    """Every preset that names a kind, as a bench sound: played on its own key, by NAME - so
    what is measured is the preset the app loads, routes and all."""
    out = []
    for p in presets if presets is not None else bank():
        if not p.get("kind"):
            continue                                  # INIT: a starting point, not a kind of sound
        out.append({"name": p["name"], "preset": p["name"], "kind": p["kind"], "bpm": BPM,
                    "notes": [{"pitch": int(p["note"]), "length_beats": float(p["beats"]), "velocity": 100}]})
    return out


def measure(out_dir: Optional[Path] = None) -> tuple[list[dict], list[dict]]:
    presets = bank()
    return presets, sound_bench.render(bench_sounds(presets), out_dir)


def census_of(presets: list[dict], rows: list[dict]) -> dict:
    sys.path.insert(0, str(REPO / "server"))
    from mix_engine import sound_classes
    from mix_engine.models import SAMPLE_PROFILE_VERSION
    by_name = {r["name"]: r for r in rows}
    out = {}
    for p in presets:
        row = by_name.get(p["name"])
        entry = {"category": p["category"], "kind": p["kind"], "note": p["note"], "beats": p["beats"],
                 "tags": p["tags"]}
        if row is not None:
            entry["profile"] = {k: v for k, v in (row.get("profile") or {}).items() if k != "path"}
            entry["issues"] = [i["says"] for i in row.get("issues") or []]
            if row.get("error"):
                entry["error"] = row["error"]
        out[p["name"]] = entry
    return {"sound_classes_version": sound_classes.SOUND_CLASSES_VERSION,
            "sample_profile_version": SAMPLE_PROFILE_VERSION, "bpm": BPM, "presets": out}


def problems(census: dict) -> list[str]:
    """What a census says is wrong: a preset that did not render, or reads outside its kind."""
    out = []
    for name, entry in census["presets"].items():
        if not entry.get("kind"):
            continue
        if entry.get("error") or "profile" not in entry:
            out.append(f"{name}: did not render ({entry.get('error', 'no file')})")
        for issue in entry.get("issues") or []:
            out.append(f"{name} ({entry['kind']}): {issue}")
    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if a preset reads outside its kind")
    parser.add_argument("--update-census", action="store_true", help="write the census (only when clean)")
    parser.add_argument("--audition", metavar="FOLDER", help="keep every preset's WAV in this folder")
    args = parser.parse_args(argv)
    if not sound_bench.built():
        print(f"sound_bank: the console is not built ({sound_bench.BINARY})")
        return sound_bench.EXIT_NOT_BUILT
    out_dir = Path(args.audition).expanduser().resolve() if args.audition else None
    presets, rows = measure(out_dir)
    census = census_of(presets, rows)
    for row in rows:
        print(sound_bench.line(row))
    wrong = problems(census)
    print(f"\n{len(rows)} presets measured, {len(wrong)} problems")
    for line in wrong:
        print("  " + line)
    if out_dir is not None:
        for row in rows:                               # name the files for a person
            src = Path(row.get("file", ""))
            if src.exists():
                entry = census["presets"][row["name"]]
                src.rename(out_dir / f"{entry['category']} - {row['name']}.wav")
        for extra in ("spec.json", "bench.json"):
            (out_dir / extra).unlink(missing_ok=True)
        print(f"the files are in {out_dir}")
    if args.update_census:
        if wrong:
            print("the census was NOT written: fix the problems, or the class ranges, first")
            return 1
        CENSUS.write_text(json.dumps(census, indent=1, sort_keys=True) + "\n")
        print(f"wrote {CENSUS.name}")
    return 1 if (wrong and (args.check or args.update_census)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
