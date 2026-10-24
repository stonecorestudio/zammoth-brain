#!/usr/bin/env python3
"""What third-party Python is INSIDE the frozen brain, and under which licences.

The brain is one PyInstaller bundle: this program's own modules (GPLv3), the Python runtime,
and about seventy third-party distributions. Until 2026-10-07 `daw/THIRD_PARTY_NOTICES.md`
named two of them in passing and the bundle carried no licence text for any (reader S7.R15).

The spec calls `for_analysis` once PyInstaller knows exactly what it will freeze, and bundles
the two files this writes at the archive root (`_internal/`):

  THIRD_PARTY_PYTHON.txt   every bundled distribution - name, version, licence - and the
                           licence files its wheel ships, then the Python runtime's own
  python_bundle.json       the same list as data: what `verify_bundled_brain.py` reads to
                           refuse a brain that carries test tooling or something it cannot name

So the notices describe the freeze they travel with: nothing is listed by hand, and a new
dependency is in the file the first time it is frozen.

    bundle_notices.py --venv           # print what the CURRENT interpreter would attribute
                                       # for a list of module names on stdin (debugging)

Stdlib only (it runs inside the PyInstaller spec, with the build interpreter).
"""
from __future__ import annotations

import json
import re
import sys
import sysconfig
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Iterable

NOTICES_NAME = "THIRD_PARTY_PYTHON.txt"
MANIFEST_NAME = "python_bundle.json"

#: Distributions that are developer or test tooling. None may be in a shipped brain: the spec
#: excludes their packages, and `verify_bundled_brain.py` refuses a bundle whose manifest names
#: one. (Names as `importlib.metadata` normalises them: lower case, `-`.)
DEV_ONLY = frozenset({"pytest", "hypothesis", "mypy", "fastapi", "pluggy", "iniconfig",
                      "claude-agent-sdk", "uvicorn", "websockets", "pyinstaller"})

#: A file in a wheel that carries licence or attribution text.
_LICENCE_FILE = re.compile(
    r"^(LICEN[CS]E|LICEN[CS]ES|COPYING|COPYRIGHT|NOTICE|AUTHORS|PATENTS|"
    r"ThirdPartyNotices|THIRD[-_ ]?PARTY)", re.IGNORECASE)
_TEXT_SUFFIX = {"", ".txt", ".md", ".rst", ".apache", ".bsd", ".mit", ".psf", ".apache2"}
#: Longest single licence file taken whole; a longer one is cut and says so.
_MAX_FILE_CHARS = 600_000


def normal(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass
class Dist:
    name: str
    version: str
    licence: str
    home: str = ""
    modules: int = 0
    files: int = 0
    texts: list[tuple[str, str]] = field(default_factory=list)   # (file name, text)

    def row(self) -> dict:
        return {"name": self.name, "version": self.version, "licence": self.licence,
                "home": self.home, "modules": self.modules, "files": self.files,
                "licence_files": [n for n, _ in self.texts]}


def _licence_of(md) -> str:
    """The licence a distribution DECLARES: its SPDX expression, else a short `License` field,
    else its trove classifiers. A long `License` field is the text itself, not a name."""
    expr = (md.get("License-Expression") or "").strip()
    if expr:
        return expr
    short = (md.get("License") or "").strip()
    if short and len(short) <= 80 and "\n" not in short and short.upper() != "UNKNOWN":
        return short
    trove = [c.split("::")[-1].strip() for c in (md.get_all("Classifier") or [])
             if c.startswith("License ::")]
    trove = [t for t in trove if t and t != "OSI Approved"]
    if trove:
        return " / ".join(dict.fromkeys(trove))
    return "see the licence text below" if short else ""


def _home_of(md) -> str:
    for line in md.get_all("Project-URL") or []:
        label, _, url = line.partition(",")
        if label.strip().lower() in ("homepage", "home", "source", "repository", "source code"):
            return url.strip()
    return (md.get("Home-page") or "").strip()


def _licence_texts(dist) -> list[tuple[str, str]]:
    """Every licence / notice file the wheel ships: the ones its metadata names, and any file
    named like one in its dist-info or at the top of one of its packages."""
    declared = set(dist.metadata.get_all("License-File") or [])
    found: dict[str, Path] = {}
    for f in dist.files or []:
        parts = f.parts
        if not parts:
            continue
        if parts[0].endswith(".dist-info"):
            inside = "/".join(parts[1:])
            named = inside in declared or inside.removeprefix("licenses/") in declared
            if not (named or _LICENCE_FILE.match(parts[-1])):
                continue
            label = inside
        else:
            if len(parts) > 2 or not _LICENCE_FILE.match(parts[-1]):
                continue
            if Path(parts[-1]).suffix.lower() not in _TEXT_SUFFIX:
                continue
            label = "/".join(parts)
        found.setdefault(label, Path(str(dist.locate_file(f))))
    out = []
    for label in sorted(found):
        path = found[label]
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not text.strip():
            continue
        if len(text) > _MAX_FILE_CHARS:
            text = text[:_MAX_FILE_CHARS] + f"\n[... cut at {_MAX_FILE_CHARS} characters ...]\n"
        out.append((label, text))
    return out


def _site_roots() -> list[Path]:
    roots = {Path(p).resolve() for p in sys.path if p and Path(p).name == "site-packages"}
    for key in ("purelib", "platlib"):
        roots.add(Path(sysconfig.get_path(key)).resolve())
    return sorted(roots)


def _top_under(path: Path, roots: list[Path]) -> str | None:
    """The top-level name a file belongs to, when it sits under a site-packages folder."""
    try:
        resolved = path.resolve()
    except OSError:
        return None
    for root in roots:
        try:
            rel = resolved.relative_to(root)
        except ValueError:
            continue
        head = rel.parts[0]
        if head.endswith((".dist-info", ".egg-info")):
            return None                              # metadata: belongs to its distribution anyway
        return head.split(".")[0] if len(rel.parts) == 1 else head
    return None


def attribute(modules: Iterable[tuple[str, str]], files: Iterable[str],
              own_roots: Iterable[Path] = ()) -> dict:
    """Sort what a freeze holds into: third-party distributions, the Python runtime, this
    program's own modules, and what cannot be named.

    `modules` are (dotted name, source path) pairs; `files` are the source paths of bundled
    binaries and data files. Returns {"dists": [Dist], "runtime_modules": n,
    "runtime_files": [names], "own_modules": n, "unattributed": [names]}."""
    roots = _site_roots()
    stdlib = Path(sysconfig.get_path("stdlib")).resolve()
    base = Path(sys.base_prefix).resolve()
    own = [Path(p).resolve() for p in own_roots]
    by_top = metadata.packages_distributions()

    counts: dict[str, list[int]] = {}
    unattributed: set[str] = set()
    runtime_modules = own_modules = 0
    runtime_files: set[str] = set()

    def credit(top: str, slot: int, shown: str) -> None:
        names = by_top.get(top) or by_top.get(top.replace("-", "_"))
        if not names:
            unattributed.add(shown)
            return
        for name in dict.fromkeys(normal(n) for n in names):
            counts.setdefault(name, [0, 0])[slot] += 1

    for dotted, src in modules:
        path = Path(src) if src else None
        top = dotted.split(".")[0]
        if path is not None:
            try:
                resolved = path.resolve()
            except OSError:
                resolved = path
            under = _top_under(path, roots)       # first: a venv can sit inside an own root
            if under is not None:
                credit(under, 0, top)
                continue
            if any(resolved.is_relative_to(o) for o in own):
                own_modules += 1
                continue
            if resolved.is_relative_to(stdlib) or resolved.is_relative_to(base):
                runtime_modules += 1
                continue
        if top in sys.stdlib_module_names or top in by_top:
            if top in by_top and top not in sys.stdlib_module_names:
                credit(top, 0, top)
            else:
                runtime_modules += 1
        else:
            unattributed.add(top)

    for src in files:
        path = Path(src)
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        under = _top_under(path, roots)
        if under is not None:
            credit(under, 1, f"{under} ({path.name})")
        elif any(resolved.is_relative_to(o) for o in own):
            continue
        elif resolved.is_relative_to(base):
            # the interpreter's own extension modules are the runtime; what is worth naming
            # is a LIBRARY its build brought along (OpenSSL, ncurses, zstd)
            if path.suffix == ".dylib":
                runtime_files.add(path.name)
        else:
            unattributed.add(str(path))

    dists = []
    for name in sorted(counts):
        try:
            d = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            unattributed.add(name)
            continue
        md = d.metadata
        dists.append(Dist(name=md.get("Name") or name, version=d.version, licence=_licence_of(md),
                          home=_home_of(md), modules=counts[name][0], files=counts[name][1],
                          texts=_licence_texts(d)))
    return {"dists": dists, "runtime_modules": runtime_modules,
            "runtime_files": sorted(runtime_files), "own_modules": own_modules,
            "unattributed": sorted(unattributed)}


#: Libraries the runtime's build brings (`runtime_files`) whose licence text Python's own
#: license.rst carries: OpenSSL.
_RUNTIME_TEXT_IN_LICENSE_RST = {"libcrypto", "libssl"}

#: ...and those it does not, with their text here. zstd 1.5.7's LICENSE, verbatim from
#: https://github.com/facebook/zstd/blob/v1.5.7/LICENSE (zstd is dual-licensed BSD OR GPLv2;
#: the brain takes it under the BSD terms). Python 3.14's installer bundles libzstd for
#: `compression.zstd` and its license.rst names only the pyzstd bindings (docs/14 section 4.3).
_RUNTIME_LIBRARY_TEXTS = {
    "libzstd": ("Zstandard (libzstd), BSD licence - LICENSE", """\
BSD License

For Zstandard software

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither the name Facebook, nor Meta, nor the names of its contributors may
   be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""),
}


def _library(file_name: str) -> str:
    """`libzstd.1.dylib` -> `libzstd`."""
    return file_name.split(".", 1)[0]


def _runtime_without_text(runtime_files: Iterable[str]) -> list[str]:
    """The runtime's libraries for which no licence text is carried. Not empty = refused."""
    covered = _RUNTIME_TEXT_IN_LICENSE_RST | set(_RUNTIME_LIBRARY_TEXTS)
    return sorted(f for f in runtime_files if _library(f) not in covered)


def _runtime_texts(runtime_files: Iterable[str] = ()) -> list[tuple[str, str]]:
    """The interpreter's licence, and - where the installer shipped it - its own list of the
    software it incorporates (OpenSSL, libffi, expat, zlib, ...); then the text of each library
    the runtime brought that the list leaves out (`_RUNTIME_LIBRARY_TEXTS`)."""
    out = []
    stdlib = Path(sysconfig.get_path("stdlib"))
    for label, path in (
            ("LICENSE.txt", stdlib / "LICENSE.txt"),
            ("Licenses and Acknowledgements for Incorporated Software (license.rst)",
             Path(sys.base_prefix) / "Resources" / "English.lproj" / "Documentation"
             / "_sources" / "license.rst.txt")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if text.strip():
            out.append((label, text))
    for lib in sorted({_library(f) for f in runtime_files} & set(_RUNTIME_LIBRARY_TEXTS)):
        out.append(_RUNTIME_LIBRARY_TEXTS[lib])
    return out


def manifest(found: dict) -> dict:
    dists = found["dists"]
    return {
        "python": sys.version.split()[0],
        "distributions": [d.row() for d in dists],
        "runtime_modules": found["runtime_modules"],
        "runtime_files": found["runtime_files"],
        "runtime_without_text": _runtime_without_text(found["runtime_files"]),
        "own_modules": found["own_modules"],
        "unattributed": found["unattributed"],
        "no_licence_named": [d.name for d in dists if not d.licence],
        "no_licence_text": [d.name for d in dists if not d.texts],
        "dev_only": sorted(d.name for d in dists if normal(d.name) in DEV_ONLY),
    }


_RULE = "=" * 78


def render(found: dict) -> str:
    dists: list[Dist] = found["dists"]
    lines = [
        "Third-party Python in the Zammoth brain",
        _RULE,
        "",
        "The brain (Contents/Resources/mixagent_daw_brain) is a separate program from the",
        "Zammoth app, distributed under the GNU General Public License v3; its source is",
        "published with it. It is built from its own modules, the Python runtime, and the",
        "third-party distributions below. This file was written by the build that froze",
        "them (daw/sidecar/bundle_notices.py), from what that build actually contains.",
        "",
        f"Python runtime: {sys.version.split()[0]} (Python Software Foundation License)",
        f"Distributions: {len(dists)}",
        "",
        f"{'Distribution':<32}{'Version':<16}Licence",
        f"{'-' * 30:<32}{'-' * 14:<16}{'-' * 28}",
    ]
    for d in dists:
        lines.append(f"{d.name:<32}{d.version:<16}{d.licence or '(not declared: see its text below)'}")
    if found["runtime_files"]:
        lines += ["", "Libraries that came with the Python runtime's own build:",
                  "  " + ", ".join(found["runtime_files"])]
    lines += ["", ""]
    for d in dists:
        lines += [_RULE, f"{d.name} {d.version}", f"Licence: {d.licence or 'see below'}"]
        if d.home:
            lines.append(d.home)
        lines.append(_RULE)
        if not d.texts:
            lines += ["", "(this distribution's wheel ships no licence file; the licence above is",
                      " the one its metadata declares)", ""]
        for label, text in d.texts:
            lines += ["", f"--- {label} ---", "", text.rstrip(), ""]
    lines += [_RULE, f"The Python runtime {sys.version.split()[0]}", _RULE]
    for label, text in _runtime_texts(found["runtime_files"]):
        lines += ["", f"--- {label} ---", "", text.rstrip(), ""]
    return "\n".join(lines) + "\n"


def write(found: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    notices, data = out_dir / NOTICES_NAME, out_dir / MANIFEST_NAME
    notices.write_text(render(found), encoding="utf-8")
    data.write_text(json.dumps(manifest(found), indent=1) + "\n", encoding="utf-8")
    return notices, data


def for_analysis(analysis, out_dir: Path, own_roots: Iterable[Path]) -> list[tuple[str, str, str]]:
    """Called from the spec, after Analysis: write both files for what `analysis` will freeze,
    and return the TOC entries that put them at the archive root."""
    modules = [(name, src) for name, src, _kind in analysis.pure]
    files = [src for _dest, src, _kind in list(analysis.binaries) + list(analysis.datas)]
    notices, data = write(attribute(modules, files, own_roots), Path(out_dir))
    return [(NOTICES_NAME, str(notices), "DATA"), (MANIFEST_NAME, str(data), "DATA")]


def problems(data: dict) -> list[str]:
    """Why a bundle with this manifest must not ship. Empty = fine."""
    out = []
    if data.get("dev_only"):
        out.append("it carries developer / test tooling: " + ", ".join(data["dev_only"]))
    if data.get("unattributed"):
        out.append("it carries code no installed distribution accounts for: "
                   + ", ".join(data["unattributed"][:12]))
    if data.get("no_licence_named"):
        out.append("no licence is declared for: " + ", ".join(data["no_licence_named"]))
    if data.get("runtime_without_text"):
        out.append("the Python runtime brings a library with no licence text in the notices "
                   "(add it to bundle_notices._RUNTIME_LIBRARY_TEXTS): "
                   + ", ".join(data["runtime_without_text"]))
    if not data.get("distributions"):
        out.append("the manifest lists no distributions at all")
    return out


def main() -> int:
    if "--venv" in sys.argv:
        names = [line.strip() for line in sys.stdin if line.strip()]
        found = attribute([(n, "") for n in names], [])
        print(json.dumps(manifest(found), indent=1))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
