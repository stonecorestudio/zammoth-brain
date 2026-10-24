#!/usr/bin/env python3
"""Which test files does a diff owe? The tests that NAME the paths it touches.

The suite reads far outside `server/` and `orchestrator/`: the user manual, the tool docs, the
skills, the packaging scripts, the brain's freeze recipe, the workflows, the control plane. A
change to one of those can turn a test red, and nothing ran it: the path filter of the (now
switched-off) engine workflow listed two folders, and the proportionate-testing table said a
docs change owes "nothing" (reader S7.R11).

    python server/tests/owed_by_diff.py docs/17-user-manual.md daw/sidecar/stamp.sh
    python server/tests/owed_by_diff.py $(git diff --name-only main...)
    python server/tests/owed_by_diff.py --reads            # every outside path the suite names

It prints the test files to hand to pytest. The answer is read from each test module's own
text - the literal paths it builds - so a new test that reads a new file is in it by itself.
It is a floor, not a proof: a path a test assembles at run time is not seen. (The C++ sources
have their own, coarser mark: `pytest -m reads_cpp`.)

THE TESTS A CHANGE REACHES (2026-10-11, James: "only run tests the diff reaches and touches"):

    python server/tests/owed_by_diff.py --tests --run      # the working tree against HEAD, run
    python server/tests/owed_by_diff.py --tests --why      # ...listed, each with its reason
    python server/tests/owed_by_diff.py --tests --base main
    python server/tests/owed_by_diff.py --tests daw/Source # only this part of the diff

The file-by-file answer above is too wide for a hub: CLAUDE.md, the DAW record, the manual and
the two big C++ files are named somewhere in half the suite, so one commit "owed" 2,240 tests
and nine minutes; the tests that could move were 408, in one. `--tests` answers one TEST at a
time, from the diff's own lines. A test is its own source plus every helper, constant and
fixture of its module (and conftest) it uses. It is owed when:

  - it names a changed file, and something it LOOKS FOR in it - a string, a regex, a name inside
    a regex - is on a changed line or within `CONTEXT` lines of one (a test that looks for
    nothing readable is owed; so is one that names a file the diff added or removed);
  - it RUNS what it names (a script, a brain: `subprocess`, `importlib`) - then any change to
    that file, or to a file of a folder it names, is owed;
  - it WALKS a folder (`glob`, `os.walk`, `git ls-files`) the changed file is in, and its
    pattern takes that file: such a test is a guard over every file there, a new one included;
  - it names a changed SYMBOL of a changed Python source file together with its module
    (`analysis._tempo`, `setattr (analysis, "MARGIN", ...)`, a name imported from it): the
    function, method, class or constant the changed lines are in and (`--depth`, default 1)
    what uses it in the same file. A prompt or a table of `DATA_LINES` lines or more is named
    and not followed;
  - it is in a test file the diff changed or added (the whole file). `conftest.py` changed =
    the whole suite.

Still a floor, and the notes on stderr say where: a test that reaches a function only through
another module; a symbol too common a word to search for; a needle a test builds at run time.
Add those by hand and say what was left out - the whole suite belongs to a merge to `main`.
"""
from __future__ import annotations

import ast
import fnmatch
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

REPO = Path(__file__).resolve().parents[2]
TESTS = Path(__file__).resolve().parent

#: `/ "a" / "b"` after a base name, and a quoted relative path with at least one slash.
_CHAIN = re.compile(r"""((?:/\s*["'][A-Za-z0-9_.\-]+["']\s*)+)""")
_SEGMENT = re.compile(r"""["']([A-Za-z0-9_.\-]+)["']""")
_LITERAL = re.compile(r"""(?<![A-Za-z0-9_/.\-])((?:\.?[A-Za-z0-9_\-][A-Za-z0-9_.\-]*/)+[A-Za-z0-9_.\-]+)""")


@lru_cache(maxsize=1)
def top_level() -> frozenset[str]:
    """The repository's own top-level entries (tracked ones: a build folder is not a source)."""
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True)
    names = {line.split("/", 1)[0] for line in out.stdout.splitlines() if line}
    if not names:                                    # not a checkout (an exported tree)
        names = {p.name for p in REPO.iterdir()}
    return frozenset(names)


def _on_disk(path: Path, kind: str = "exists") -> bool:
    """`path.exists()` / `.is_file()` / `.is_dir()` for a path built from a test's TEXT, where any
    string is tried: a sentence past the file system's name limit makes the stat raise
    (ENAMETOOLONG), and a NUL in it is a ValueError. Neither is a path."""
    try:
        return getattr(path, kind)()
    except (OSError, ValueError):
        return False


def _existing(candidate: str) -> str | None:
    """The longest prefix of `candidate` that is a real path in the repository."""
    parts = [p for p in candidate.split("/") if p]
    if not parts or parts[0] not in top_level() or ".." in parts:
        return None
    for n in range(len(parts), 0, -1):
        rel = "/".join(parts[:n])
        if _on_disk(REPO / rel):
            return rel if n > 1 or _on_disk(REPO / rel, "is_file") else None
    return None


def reads(test_file: Path) -> set[str]:
    """The repository paths a test module names, relative to the repository root."""
    try:
        text = test_file.read_text(encoding="utf-8")
    except OSError:
        return set()
    chains = ["/".join(_SEGMENT.findall(m.group(1))) for m in _CHAIN.finditer(text)]
    # `SIDECAR = REPO / "daw" / "sidecar"` ... `SIDECAR / "stamp.sh"` is seen as the FOLDER: a
    # module that names a folder is owed by any change under it, which is the safe reading.
    return {p for p in (_existing(c) for c in chains + _LITERAL.findall(text)) if p}


@lru_cache(maxsize=1)
def suite_reads() -> dict[str, frozenset[str]]:
    return {t.name: frozenset(reads(t)) for t in sorted(TESTS.glob("test_*.py"))}


def owed(changed: list[str]) -> list[str]:
    """The test files that name a changed path, a folder it is in, or something under it."""
    changed = [c.strip().strip("/") for c in changed if c.strip()]
    out = []
    for name, paths in suite_reads().items():
        if any(c == p or c.startswith(p + "/") or p.startswith(c + "/")
               for c in changed for p in paths):
            out.append(name)
    return out


# --------------------------------------------------------------------------------------------
# One test at a time (`--tests`)
# --------------------------------------------------------------------------------------------

#: The Python the product is made of: a changed line in one of these is asked for its symbol.
SOURCE_ROOTS = ("server/mix_engine/", "orchestrator/", "backend/app/")
#: Names too common to mean anything as a word in a test's text.
_VAGUE = frozenset({"main", "run", "get", "set", "read", "write", "load", "save", "apply", "reset",
                    "update", "render", "start", "stop", "open", "close", "check", "build", "name"})
#: A test that RUNS what it names (a script, a brain) is owed by any change to it: what it looks
#: for in the output says nothing about which lines matter.
_RUNS = re.compile(r"\bsubprocess\b|\bPopen\b|\brunpy\b|\bimportlib\b|spec_from_file_location|\bexec\(|os\.system")
#: A test that WALKS a folder is a guard over every file in it, a new one included.
_WALKS = re.compile(r"\.r?glob\(|\.iterdir\(|os\.walk\(|os\.listdir\(|os\.scandir\(|ls-files")
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@ ?(.*)$")
_REGEXISH = re.compile(r"\\|\.\*|\.\+|\[|\(\?")
_FRAGMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:(?:::|\.|->)[A-Za-z_][A-Za-z0-9_]*)*")
_NOT_A_NEEDLE = frozenset({"utf-8", "utf8", "ignore", "replace", "__main__", "store_true"})
#: A data constant (a prompt, a table) this long reaches the tests that NAME it, not its users.
DATA_LINES = 4
#: Lines of the file either side of a change that count as "what changed": a pinned sentence
#: wraps, and a pinned block is longer than the one line that moved in it.
CONTEXT = 8


class Change(NamedTuple):
    lines: frozenset[int] | None        # in the file as it is now; None = the whole file
    raw: str                            # added, removed and neighbouring lines, hunk headers
    text: str                           # the same, white space collapsed
    new: bool = False                   # the file did not exist: nothing can depend on it unnamed


def _change(lines: set[int] | None, raw: list[str], new: bool = False) -> Change:
    joined = "\n".join(raw)
    return Change(None if lines is None else frozenset(lines), joined, " ".join(joined.split()), new)


def diff_of(base: str = "HEAD", context: int = CONTEXT) -> dict[str, Change]:
    """What changed between `base` and the working tree, untracked files included."""
    out: dict[str, Change] = {}
    diff = subprocess.run(["git", "diff", f"-U{context}", "--no-color", "--no-renames", base, "--"],
                          cwd=REPO, capture_output=True, text=True, errors="replace").stdout
    path, old, lines, raw, whole, at = None, None, set(), [], False, 0

    def close() -> None:
        if path or old:
            out[path or old] = _change(None if whole or not path else lines, raw, new=bool(path) and old is None)

    for line in diff.splitlines():
        if line.startswith("diff --git "):
            close()
            path, old, lines, raw, whole, at = None, None, set(), [], False, 0
        elif line.startswith("--- "):
            old = None if line.endswith("/dev/null") else line[6:]
            whole = whole or old is None
        elif line.startswith("+++ "):
            path = None if line.endswith("/dev/null") else line[6:]
            whole = whole or path is None
        elif line.startswith("Binary files "):
            whole = True
            names = re.match(r"Binary files (?:a/)?(.+?) and (?:b/)?(.+?) differ$", line)
            if names:
                path = names.group(2) if names.group(2) != "/dev/null" else names.group(1)
        elif (m := _HUNK.match(line)):
            at = int(m.group(1))
            raw.append(m.group(2))
        elif line.startswith("+"):
            lines.add(at)
            raw.append(line[1:])
            at += 1
        elif line.startswith("-"):
            lines.add(at)
            raw.append(line[1:])
        elif line.startswith(" "):
            raw.append(line[1:])
            at += 1
    close()
    new = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"], cwd=REPO,
                         capture_output=True, text=True).stdout
    for name in filter(None, (n.strip() for n in new.splitlines())):
        try:
            body = (REPO / name).read_text(encoding="utf-8", errors="replace")[:2_000_000]
        except OSError:
            body = ""
        out[name] = _change(None, body.splitlines(), new=True)
    return out


def changed_lines(base: str = "HEAD") -> dict[str, frozenset[int] | None]:
    return {path: change.lines for path, change in diff_of(base).items()}


def _span(node: ast.AST) -> tuple[int, int]:
    first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    return first, getattr(node, "end_lineno", node.lineno)


def _names_in(node: ast.AST) -> set[str]:
    found: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            found.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            found.add(sub.attr)
        elif isinstance(sub, ast.arg):
            found.add(sub.arg)
    return found


def _strings_in(node: ast.AST) -> set[str]:
    return {sub.value for sub in ast.walk(node)
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str)}


def _assigned(node: ast.AST) -> list[str]:
    if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        return []
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return [sub.id for target in targets for sub in ast.walk(target) if isinstance(sub, ast.Name)]


def _definitions(tree: ast.Module) -> list[tuple[str, ast.AST]]:
    """(name, node) for what a module defines: functions, classes, their methods, constants."""
    out: list[tuple[str, ast.AST]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append((node.name, node))
        elif isinstance(node, ast.ClassDef):
            out.append((node.name, node))
            out.extend((sub.name, sub) for sub in node.body
                       if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)))
        else:
            out.extend((name, node) for name in _assigned(node))
    return out


def changed_symbols(path: str, lines: frozenset[int] | set[int] | None, depth: int = 1) -> set[str]:
    """What the changed lines of a Python source file DEFINE, and (`depth` rounds) what uses that
    in the same file. A data constant of `DATA_LINES` lines or more - a prompt, a table - is
    named and not followed: every function that reads a prompt is not a test of its wording.
    A whole new file answers with nothing: its own tests name its path."""
    if lines is None:
        return set()
    try:
        tree = ast.parse((REPO / path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return set()
    defs = _definitions(tree)
    hit: set[str] = set()
    follow: set[str] = set()
    for name, node in defs:
        first, last = _span(node)
        if isinstance(node, ast.ClassDef):           # its methods answer for their own lines
            own = {n for n in lines if first <= n <= last}
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    a, b = _span(sub)
                    own -= set(range(a, b + 1))
            if own:
                hit.add(name)
                follow.add(name)
        elif any(first <= n <= last for n in lines):
            hit.add(name)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or last - first + 1 < DATA_LINES:
                follow.add(name)
    for _ in range(max(depth, 0)):
        users = {name for name, node in defs
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _names_in(node) & follow}
        if users <= hit:
            break
        hit |= users
        follow = users
    return {name for name in hit if not (name.startswith("__") and name.endswith("__"))}


class _Closure(NamedTuple):
    text: str
    strings: frozenset[str]


def _table(tree: ast.Module, lines: list[str]) -> dict[str, tuple[str, set[str], set[str]]]:
    def source(node: ast.AST) -> str:
        first, last = _span(node)
        return "\n".join(lines[first - 1:last])
    out: dict[str, tuple[str, set[str], set[str]]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[node.name] = (source(node), _names_in(node), _strings_in(node))
        else:
            for name in _assigned(node):
                out[name] = (source(node), _names_in(node), _strings_in(node))
    return out


@lru_cache(maxsize=None)
def _conftest_table() -> dict[str, tuple[str, set[str], set[str]]]:
    try:
        text = (TESTS / "conftest.py").read_text(encoding="utf-8")
        return _table(ast.parse(text), text.splitlines())
    except (OSError, SyntaxError):
        return {}


@lru_cache(maxsize=None)
def _closures(test_file: Path) -> dict[str, _Closure]:
    try:
        text = test_file.read_text(encoding="utf-8")
        tree = ast.parse(text)
    except (OSError, SyntaxError):
        return {}
    lines = text.splitlines()
    defs = _table(tree, lines)

    def source(node: ast.AST) -> str:
        first, last = _span(node)
        return "\n".join(lines[first - 1:last])

    always_text: list[str] = []                      # import-time statements that define nothing
    always_strings: set[str] = set()
    tests: list[tuple[str, ast.AST]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test"):
                tests.append((node.name, node))
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("Test"):
                tests.extend((f"{node.name}::{sub.name}", sub) for sub in node.body
                             if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
                             and sub.name.startswith("test"))
        elif isinstance(node, ast.ImportFrom) and node.module and (
                TESTS / (node.module.removeprefix("tests.") + ".py")).is_file():
            for alias in node.names:                 # a helper from a sibling module: its file
                where = f'"server/tests/{node.module.removeprefix("tests.")}.py"'
                defs.setdefault(alias.asname or alias.name, (where, set(), set()))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if (TESTS / (alias.name.removeprefix("tests.") + ".py")).is_file():
                    where = f'"server/tests/{alias.name.removeprefix("tests.")}.py"'
                    defs.setdefault(alias.asname or alias.name.split(".")[-1], (where, set(), set()))
        elif not _assigned(node) and not isinstance(node, ast.ImportFrom) and not (
                isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)):
            always_text.append(source(node))
            always_strings |= _strings_in(node)
    shared = _conftest_table()
    out: dict[str, _Closure] = {}
    for test_id, node in tests:
        parts, strings = [source(node)] + always_text, _strings_in(node) | always_strings
        seen, queue = set(), list(_names_in(node))
        if "::" in test_id:                          # a method: its class's fixtures and helpers
            queue.append(test_id.split("::")[0])
        while queue:
            name = queue.pop()
            if name in seen:
                continue
            seen.add(name)
            found = defs.get(name) or shared.get(name)
            if found:
                parts.append(found[0])
                strings |= found[2]
                queue.extend(found[1] - seen)
        out[test_id] = _Closure("\n".join(parts), frozenset(strings))
    return out


def test_closures(test_file: Path) -> dict[str, str]:
    """Each test of a module -> its own source plus that of every helper, constant and fixture
    of the module (and conftest) it uses, however many steps away. Keyed by its pytest id tail
    (`test_x`, `TestThing::test_x`)."""
    return {test_id: closure.text for test_id, closure in _closures(test_file).items()}


test_closures.__test__ = False                       # a helper, not a test (pytest would collect it)


@lru_cache(maxsize=None)
def _module_prefixes(test_file: Path) -> tuple[tuple[str, str], ...]:
    """`SRC = REPO / "daw" / "Source"` -> ("SRC", "daw/Source"): a module's own folder names."""
    try:
        text = test_file.read_text(encoding="utf-8")
    except OSError:
        return ()
    out: dict[str, str] = {}
    for m in re.finditer(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(?::[^=\n]+)?=\s*[^\n]*?" + _CHAIN.pattern, text, re.M):
        rel = _repo_path("/".join(_SEGMENT.findall(m.group(2))))
        if rel and (REPO / rel).is_dir():
            out[m.group(1)] = rel
    return tuple(out.items())


def _repo_path(candidate: str) -> str | None:
    """`_existing`, and a path a test builds from its own folder (`HERE / "data" / "x.json"`)."""
    found = _existing(candidate)
    if found is None and candidate and ".." not in candidate.split("/") and _on_disk(TESTS / candidate):
        found = f"server/tests/{candidate}"
    return found


def _paths_named(closure: _Closure, prefixes: tuple[tuple[str, str], ...]) -> set[str]:
    text = closure.text
    chains = ["/".join(_SEGMENT.findall(m.group(1))) for m in _CHAIN.finditer(text)]
    found = {p for p in (_repo_path(c) for c in chains + _LITERAL.findall(text)) if p}
    for var, base in prefixes:
        if not re.search(r"\b" + re.escape(var) + r"\b", text):
            continue
        found.add(base)
        for m in re.finditer(r"\b" + re.escape(var) + r"\b\s*" + _CHAIN.pattern, text):
            deeper = _existing(base + "/" + "/".join(_SEGMENT.findall(m.group(1))))
            if deeper:
                found.add(deeper)
    for base in [p for p in found if (REPO / p).is_dir()]:       # `read(SRC, "App.cpp")`
        for literal in closure.strings:
            if 2 < len(literal) < 120 and "\n" not in literal and not literal.startswith(("/", ".")):
                if _on_disk(REPO / base / literal, "is_file"):
                    found.add(f"{base}/{literal}")
    return found


def _needle_hit(closure: _Closure, named: set[str], change: Change) -> str | None:
    """What a test LOOKS FOR in a file it reads, that the changed lines hold: the reason the
    change can move the test. "" = the test gives nothing to judge by (it is owed)."""
    segments = {part for path in named for part in path.split("/")}
    needles = []
    for literal in closure.strings:
        flat = " ".join(literal.split())
        if len(flat) < 4 or flat in _NOT_A_NEEDLE or flat in segments or _repo_path(flat.strip("/")):
            continue
        needles.append((literal, flat))
    if not needles:
        return ""
    for literal, flat in needles:
        if flat in change.text:
            return flat
    for literal, flat in needles:
        if not _REGEXISH.search(literal) or len(literal) > 400:
            continue
        try:
            if re.search(literal, change.raw) or re.search(literal, change.raw, re.S | re.M):
                return flat
        except (re.error, RecursionError, OverflowError):
            pass
        for piece in _FRAGMENT.findall(literal):                 # a block-long regex: its names
            specific = "::" in piece or "." in piece or "_" in piece.strip("_") or (
                piece[1:] != piece[1:].lower() and piece != piece.upper())
            if len(piece) >= 8 and specific and piece in change.text:
                return piece
    return None


def _glob_takes(closure: _Closure, path: str) -> bool:
    """Would the patterns a walking test globs with take this file? (`"*.patch"` does not take
    the README beside the patches.) A walk with no pattern to read takes everything."""
    patterns = [s.split("/")[-1] for s in closure.strings
                if ("*" in s or "?" in s) and len(s) < 60 and not re.search(r"[\s\\()|^$+]", s)]
    name = path.rsplit("/", 1)[-1]
    return not patterns or any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def _why_by_path(closure: _Closure, named: set[str], changes: dict[str, Change]) -> str | None:
    runs, walks = bool(_RUNS.search(closure.text)), bool(_WALKS.search(closure.text))
    for path, change in changes.items():
        inside_tests = path.startswith("server/tests/")
        for p in named:
            if p.startswith(path + "/"):
                return f"names {p}, under {path}"
            if path == p:
                exact = True
            elif path.startswith(p + "/"):
                exact = False
            else:
                continue
            if not exact:
                if walks and _glob_takes(closure, path):
                    return f"walks {p} ({path} changed)"
                if change.new or not runs:
                    continue                         # it opens, or runs, other files of that folder
                return f"runs something in {p} ({path} changed)"
            if change.lines is None:
                return f"names {p} ({path} is new or gone)"
            if runs or inside_tests:
                return f"runs or loads {path}"
            hit = _needle_hit(closure, named, change)
            if hit == "":
                return f"reads {path} (nothing to judge its reach by)"
            if hit:
                return f"looks in {path} for {hit[:60]!r}"
    return None


@lru_cache(maxsize=None)
def _imports(test_file: Path) -> tuple[dict[str, set[str]], dict[str, tuple[str, str]]]:
    """A test module's top-level imports: ({source module: the local words for it},
    {local name: (source module, the name it has there)})."""
    aliases: dict[str, set[str]] = {}
    imported: dict[str, tuple[str, str]] = {}
    try:
        tree = ast.parse(test_file.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return aliases, imported
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases.setdefault(alias.name.split(".")[-1], set()).add(alias.asname or alias.name.split(".")[-1])
        elif isinstance(node, ast.ImportFrom) and node.module:
            module = node.module.split(".")[-1]
            for alias in node.names:
                aliases.setdefault(alias.name, set()).add(alias.asname or alias.name)   # `from pkg import mod`
                imported[alias.asname or alias.name] = (module, alias.name)
    return aliases, imported


def _why_by_symbol(text: str, symbols: dict[str, set[str]], aliases: dict[str, set[str]],
                   imported: dict[str, tuple[str, str]]) -> str | None:
    """A test names a changed symbol only WITH its module: `analysis._tempo`, a name imported
    from it, or the module and the name in one test (`setattr (analysis, "MARGIN", ...)`).
    `CORPUS` alone is five modules' corpus."""
    for module, names in symbols.items():
        words = aliases.get(module, set()) | {module}
        if any(re.search(r"\b" + re.escape(w) + r"\b", text) for w in words):
            if (m := re.search(r"\b(" + "|".join(map(re.escape, sorted(names))) + r")\b", text)):
                return f"names {module}.{m.group(1)}"
    for local, (module, original) in imported.items():
        if original in symbols.get(module, ()) and re.search(r"\b" + re.escape(local) + r"\b", text):
            return f"names {module}.{original}"
    return None


def owed_tests(changes: dict[str, Change], depth: int = 1) -> tuple[dict[str, str], list[str]]:
    """({pytest id: why}, notes) for a set of changes, one test at a time."""
    notes: list[str] = []
    harness = [c for c in changes if c in ("server/tests/conftest.py", "server/pytest.ini",
                                           "server/pyproject.toml", "server/setup.cfg")]
    if harness:
        notes.append(f"{harness[0]} changed - the harness itself: the whole suite is owed")
        return {f"tests/{t.name}": "the harness changed" for t in sorted(TESTS.glob("test_*.py"))}, notes
    symbols: dict[str, set[str]] = {}                # source module name -> its changed symbols
    for path in sorted(changes):
        if not (path.endswith(".py") and path.startswith(SOURCE_ROOTS)) or changes[path].lines is None:
            continue
        names = changed_symbols(path, changes[path].lines, depth)
        vague = sorted(n for n in names if n in _VAGUE or len(n) < 4)
        names -= set(vague)
        if vague:
            notes.append(f"{path}: {', '.join(vague)} - too common a word to search tests for; "
                         f"pick those tests by hand")
        if names:
            module = Path(path).stem if Path(path).stem != "__init__" else Path(path).parent.name
            symbols.setdefault(module, set()).update(names)
            notes.append(f"{path}: {', '.join(sorted(names))}")
        elif not vague:
            notes.append(f"{path}: the changed lines are in no function or constant (imports, "
                         f"module text) - any test that imports it shows a break")
    out: dict[str, str] = {}
    for test_file in sorted(TESTS.glob("test_*.py")):
        rel, file_id = f"server/tests/{test_file.name}", f"tests/{test_file.name}"
        if rel in changes:
            out[file_id] = "the diff changed this test file"
            continue
        closures = _closures(test_file)
        if not closures:
            continue
        prefixes = _module_prefixes(test_file)
        aliases, imported = _imports(test_file)
        hit: dict[str, str] = {}
        for test_id, closure in closures.items():
            why = _why_by_path(closure, _paths_named(closure, prefixes), changes)
            if why is None and symbols:
                why = _why_by_symbol(closure.text, symbols, aliases, imported)
            if why:
                hit[test_id] = why
        if hit and len(hit) == len(closures):
            reasons = sorted(set(hit.values()))
            out[file_id] = reasons[0] + (f" (+{len(reasons) - 1} more reasons)" if len(reasons) > 1 else "")
        else:
            out.update({f"{file_id}::{test_id}": why for test_id, why in hit.items()})
    return out, notes


def _tests_mode(argv: list[str]) -> int:
    base, depth, run, why, only = "HEAD", 1, False, False, []
    it = iter(argv)
    for arg in it:
        if arg == "--base":
            base = next(it, "HEAD")
        elif arg == "--depth":
            depth = int(next(it, "1"))
        elif arg == "--run":
            run = True
        elif arg == "--why":
            why = True
        else:
            only.append(arg.strip().strip("/"))
    changes = diff_of(base)
    if only:                                         # only these, of what the diff holds
        changes = {p: c for p, c in changes.items() if any(p == o or p.startswith(o + "/") for o in only)}
    owed_now, notes = owed_tests(changes, depth)
    files = {i.split("::")[0] for i in owed_now}
    total = len(list(TESTS.glob("test_*.py")))
    print(f"# {len(changes)} changed paths reach {len(owed_now)} selections in {len(files)} of "
          f"{total} test files (against {base})", file=sys.stderr)
    for note in notes:
        print(f"# {note}", file=sys.stderr)
    if run:
        if not owed_now:
            print("# nothing to run", file=sys.stderr)
            return 0
        return subprocess.call([sys.executable, "-m", "pytest", "-q", "--maxfail=20",
                                "-p", "no:cacheprovider", *owed_now], cwd=TESTS.parent)
    for test_id, reason in owed_now.items():
        print(f"{test_id}    # {reason}" if why else test_id)
    return 0


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 2
    if argv[0] == "--tests":
        return _tests_mode(argv[1:])
    if argv[0] == "--reads":
        own = ("server/", "orchestrator/", "test_projects/")
        seen = sorted({p for paths in suite_reads().values() for p in paths
                       if not p.startswith(own)})
        print("\n".join(seen))
        return 0
    for name in owed(argv):
        print(f"tests/{name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
