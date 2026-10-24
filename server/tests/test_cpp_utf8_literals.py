"""No C++ string literal may reach a `juce::String` as Latin-1.

`juce::String (const char*)` does not decode UTF-8. It reads each byte as one
character (`CharPointer_ASCII`), so a narrow literal holding an em dash — three
bytes, E2 80 94 — arrives as three characters: `â`, `€` and an invisible
control code. Found on screen 2026-10-01: every Auto Mix that carried a loudness
target (the default one) printed `auto mix â€ vocal-forward mix, master for
streaming` in the engineer feed and titled the chat session with it.

It was not one line. The same conversion is behind every implicit route a
literal takes into a String — an argument (`setStatus ("…")`), a `return` from a
function that returns String, `"…" + name`, `s << "…"`, a `juce::var` — and 137
literals in the product took it (plus 118 labels and fixtures in the console
harness): status lines, tooltips, menu items, device-face captions, and the
`err (…)` text of about fifty command refusals, which is returned to the AGENT as
JSON, so the model read the junk too. Two `const char*` tables were read the same
way: the command catalog handed its labels and 20 arg schemas to `juce::var` raw.
Writing the bytes as escapes (`"\\xe2\\x80\\x94"`) changes nothing — the bytes
are the same — which is how most of them survived a grep for non-ASCII source
characters.

What is correct (and what this file accepts):

  * `juce::String::fromUTF8 ("…")` / `juce::CharPointer_UTF8 ("…")`, or a local
    lambda that only forwards to `fromUTF8`;
  * a `u8"…"` literal (C++20 `char8_t`, which `juce::String` decodes as UTF-8);
  * bytes that never become a Latin-1 String: a direct `std::cout` / `std::cerr`
    operand, or an argument to a local lambda taking `const char*` whose body only
    streams it (the console harness's `check (cond, "…")` labels) or decodes it
    (`setTooltip (juce::String::fromUTF8 (tip))`);
  * a `const char*` TABLE whose reader decodes it — these are listed in
    `RAW_BYTE_TABLES` with the decode site, and the test checks that site is
    still there, because a table is only as correct as the one line reading it.

`juce::StringRef` (what `contains` / `startsWith` / `replace` take) does decode
UTF-8, but nothing here can see a parameter's type, so a literal passed to one is
flagged like any other; wrap it.

The scanner is a lexer, not a grep: it knows comments, raw strings, adjacent
literal concatenation and call nesting, so a `fromUTF8 (` four lines above its
literal counts and a `// —` comment does not. `test_scanner_*` below feed it the
bug's own shapes, so it cannot quietly stop finding them.
"""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ROOTS = (REPO / "daw" / "Source", REPO / "native" / "Source")
EXTS = {".h", ".hpp", ".cpp", ".mm"}

# Calls that decode their argument as UTF-8.
DECODERS = {"fromUTF8", "CharPointer_UTF8"}


@dataclass(frozen=True)
class RawTable:
    """A `const char*` table that holds UTF-8 bytes and is decoded by its reader."""

    file: str        # repo-relative path of the table
    head: str        # regex matched against the declaration enclosing the literal
    decoded_by: tuple[tuple[str, str], ...]   # (repo-relative file, source text that decodes it)
    why: str


# Each entry is a promise that the bytes are decoded where they are read. The
# `decoded_by` text must exist verbatim in the named file or the test fails.
RAW_BYTE_TABLES: tuple[RawTable, ...] = (
    RawTable("daw/Source/DawCommands.h",
             r"static const std::vector<Spec> table =$",
             (("daw/Source/DawCommands.h",
               "juce::String displayName() const   { return juce::String::fromUTF8 (nameUtf8); }"),),
             "the command table's labels; `nameUtf8` is only ever read through displayName()"),
    RawTable("daw/Source/DawCommands.h",
             r"static const std::map<std::string, const char\*> table =$",
             (("daw/Source/DawCommands.h",
               'o->setProperty ("args", juce::String::fromUTF8 (a));'),),
             "the arg schemas the agent's catalog carries"),
    RawTable("daw/Source/FileBrowserPanel.h",
             r"static const char\* placeholders\[\] =$",
             (("daw/Source/FileBrowserPanel.h", "juce::String::fromUTF8 (placeholders["),),
             "the per-tab search placeholders"),
    RawTable("daw/Source/LauncherPanel.h",
             r"static const char\* const hk\[3\] =$",
             (("daw/Source/LauncherPanel.h", "juce::String::fromUTF8 (hk[i])"),),
             "the pad hover keys"),
    RawTable("daw/Source/PrefsPanel.h",
             r"static const char\* tips\[\] =$",
             (("daw/Source/PrefsPanel.h", "juce::String::fromUTF8 (categoryTip (i))"),),
             "the Preferences sidebar tooltips"),
    RawTable("native/Source/DeviceGraphs.h",
             r"static const char\* caps(Mini|Full)\[\] =$",
             (("native/Source/DeviceGraphs.h",
               "String::fromUTF8 (mini ? capsMini[i] : capsFull[i])"),),
             "the delay / IR-reverb face cell captions"),
    RawTable("native/Source/DeviceGraphs.h",
             r"static const char\* labs\[5\] =$",
             (("native/Source/DeviceGraphs.h", "String (CharPointer_UTF8 (labs[i]))"),),
             "the utility face's unhosted toggle labels"),
    RawTable("native/Source/DeviceSpecs.h",
             r"std::vector<ToggleSpec> deviceToggleSpecs \(",
             (("daw/Source/DeviceChainView.h", "juce::String::fromUTF8 (ts.label)"),),
             "the device toggle labels (polarity is a slashed O)"),
    RawTable("native/Source/EngineerPanel.h",
             r"static const std::pair<const char\*, const char\*> targets\[\] =$",
             (("native/Source/EngineerPanel.h", "juce::String::fromUTF8 (targets[i].first)"),),
             "the Auto Mix menu's loudness targets"),
)

# Single literals that are correct for a reason the rules above cannot see:
# (repo-relative file, text the literal contains, why).
ALLOWED_LITERALS: tuple[tuple[str, str, str], ...] = (
    ("daw/Source/CrashReporter.h", r'"Zammoth \xe2\x80\x94 crash report\n"',
     "putStr() write()s the bytes to the report's fd from the signal handler; no String exists"),
)


# --------------------------------------------------------------------------- lexer

_TOKEN = re.compile(
    r"""
      //[^\n]*(?:\\\n[^\n]*)*                                   # line comment
    | /\*.*?\*/                                                 # block comment
    | (?P<rprefix>(?:u8|u|U|L)?R)"(?P<delim>[^()\\\s"]{0,16})\((?P<rbody>.*?)\)(?P=delim)"
    | (?P<prefix>@|u8|u|U|L)?"(?P<body>(?:[^"\\\n]|\\.)*)"      # string literal
    | (?:u8|u|U|L)?'(?:[^'\\\n]|\\.)+'                          # character literal
    """,
    re.S | re.X,
)
_HEX = re.compile(r"[0-9a-fA-F]+")
_OCT = re.compile(r"[0-7]{1,3}")


@dataclass
class Literal:
    start: int       # offset of the first character of the token (its prefix, if any)
    end: int         # one past the closing quote
    prefix: str
    body: str
    raw: bool

    @property
    def high(self) -> bool:
        """Do the literal's BYTES include one >= 0x80 — written raw or as an escape?"""
        b = self.body
        if not b.isascii():
            return True
        if self.raw or "\\" not in b:
            return False
        i, n = 0, len(b)
        while i < n:
            if b[i] != "\\":
                i += 1
                continue
            c = b[i + 1] if i + 1 < n else ""
            if c == "x":
                m = _HEX.match(b, i + 2)
                if m and int(m.group(0), 16) >= 0x80:
                    return True
                i = m.end() if m else i + 2
            elif c in "01234567":
                m = _OCT.match(b, i + 1)
                if int(m.group(0), 8) >= 0x80:
                    return True
                i = m.end()
            elif c in "uU":
                w = 4 if c == "u" else 8
                digits = b[i + 2:i + 2 + w]
                if len(digits) == w and int(digits, 16) >= 0x80:
                    return True
                i += 2 + w
            else:
                i += 2
        return False


@dataclass
class Run:
    """Adjacent literals the compiler concatenates into one string."""

    lits: list[Literal]

    @property
    def start(self) -> int:
        return self.lits[0].start

    @property
    def end(self) -> int:
        return self.lits[-1].end

    @property
    def high(self) -> bool:
        return any(lit.high for lit in self.lits)


def _blank(text: str) -> str:
    return re.sub(r"[^\n]", " ", text)


def lex(src: str) -> tuple[list[Literal], str]:
    """The string literals of `src`, and a MASK of it: the same length, with comments,
    literal bodies and character literals blanked, so brackets and operators found in
    the mask are code."""
    lits: list[Literal] = []
    out: list[str] = []
    last = 0
    for m in _TOKEN.finditer(src):
        out.append(src[last:m.start()])
        tok = m.group(0)
        if m.group("rbody") is not None:
            lits.append(Literal(m.start(), m.end(), m.group("rprefix"), m.group("rbody"), True))
            out.append(m.group("rprefix") + '"' + _blank(tok[len(m.group("rprefix")) + 1:-1]) + '"')
        elif m.group("body") is not None:
            prefix = m.group("prefix") or ""
            lits.append(Literal(m.start(), m.end(), prefix, m.group("body"), False))
            out.append(prefix + '"' + _blank(m.group("body")) + '"')
        else:
            out.append(_blank(tok))
        last = m.end()
    out.append(src[last:])
    return lits, "".join(out)


def runs_of(lits: list[Literal], mask: str) -> list[Run]:
    runs: list[Run] = []
    for lit in lits:
        if runs and not mask[runs[-1].end:lit.start].strip():
            runs[-1].lits.append(lit)
        else:
            runs.append(Run([lit]))
    return runs


# --------------------------------------------------------------------------- context

_CALLEE = re.compile(r"([A-Za-z_]\w*(?:\s*(?:::|\.|->)\s*~?[A-Za-z_]\w*)*)\s*$")
_LAMBDA = re.compile(
    r"\bauto\s+(\w+)\s*=\s*\[[^\]]*\]\s*\(([^()]*)\)\s*(?:mutable\s*)?(?:->\s*[\w:<>]+\s*)?\{"
)
_FORWARDER = re.compile(
    r"\bauto\s+(\w+)\s*=\s*\[\]\s*\(\s*const\s+char\s*\*\s*(\w+)\s*\)\s*"
    r"\{\s*return\s+(?:juce::)?String::fromUTF8\s*\(\s*(\w+)\s*\)\s*;\s*\}"
)
_CONTROL = {"if", "for", "while", "switch", "return", "catch", "sizeof", "decltype", "alignof"}


@dataclass
class _Lambda:
    name: str
    at: int
    params: list[str]
    body: str


@dataclass
class Context:
    src: str
    mask: str
    runs: list[Run]
    stacks: dict[int, list[tuple[str, int]]] = field(default_factory=dict)
    lambdas: list[_Lambda] = field(default_factory=list)
    forwarders: set[str] = field(default_factory=set)
    line_starts: list[int] = field(default_factory=list)

    def line_of(self, pos: int) -> int:
        return bisect.bisect_right(self.line_starts, pos)

    def prev_tok(self, pos: int) -> str:
        k = pos - 1
        while k >= 0 and self.mask[k].isspace():
            k -= 1
        if k < 0:
            return ""
        return "<<" if self.mask[k - 1:k + 1] == "<<" else self.mask[k]

    def next_tok(self, pos: int) -> str:
        k, n = pos, len(self.mask)
        while k < n and self.mask[k].isspace():
            k += 1
        if k >= n:
            return ""
        return "<<" if self.mask[k:k + 2] == "<<" else self.mask[k]

    def callee(self, open_paren: int) -> str:
        m = _CALLEE.search(self.mask, max(0, open_paren - 240), open_paren)
        if not m:
            return ""
        name = re.sub(r"\s+", "", m.group(1))
        return "" if name in _CONTROL else name


def context_for(src: str) -> Context:
    lits, mask = lex(src)
    ctx = Context(src, mask, runs_of(lits, mask))
    ctx.line_starts = [0] + [m.end() for m in re.finditer(r"\n", src)]

    # One pass over the brackets, taking a copy of the open-bracket stack at each run
    # that needs judging (innermost bracket last).
    wanted = sorted(r.start for r in ctx.runs if r.high)
    w = 0
    stack: list[tuple[str, int]] = []
    pairs = {")": "(", "}": "{"}
    for m in re.finditer(r"[(){}]", mask):
        while w < len(wanted) and wanted[w] < m.start():
            ctx.stacks[wanted[w]] = list(stack)
            w += 1
        ch = m.group(0)
        if ch in "({":
            stack.append((ch, m.start()))
        elif stack and stack[-1][0] == pairs[ch]:
            stack.pop()
    for pos in wanted[w:]:
        ctx.stacks[pos] = list(stack)

    for m in _LAMBDA.finditer(mask):
        depth, k = 1, m.end()
        while k < len(mask) and depth:
            depth += {"{": 1, "}": -1}.get(mask[k], 0)
            k += 1
        ctx.lambdas.append(_Lambda(m.group(1), m.start(),
                                   [p.strip() for p in m.group(2).split(",")], mask[m.end():k]))
    for m in _FORWARDER.finditer(mask):
        if m.group(2) == m.group(3):
            ctx.forwarders.add(m.group(1))
    return ctx


def _decoded(run: Run, ctx: Context) -> bool:
    """Is the innermost real call around the run one that decodes UTF-8?"""
    for ch, pos in reversed(ctx.stacks[run.start]):
        if ch != "(":
            return False
        name = ctx.callee(pos)
        if not name:
            continue                      # a bare ( … ) — keep looking outward
        last = re.split(r"::|\.|->", name)[-1]
        return last in DECODERS or name in ctx.forwarders
    return False


def _statement_head(ctx: Context, pos: int, floor: int) -> str:
    """The code between the start of the statement holding `pos` and `pos` itself, looking
    no further back than `floor` (the inside of the enclosing brace)."""
    k = max(ctx.mask.rfind(c, floor, pos) for c in ";{}")
    return ctx.mask[max(k, floor - 1) + 1:pos]


def _ostream_operand(run: Run, ctx: Context) -> bool:
    """A direct operand of a std::cout / std::cerr chain: its bytes go to the stream as
    they are. `std::cout << name + "…"` is NOT one — `+` binds first, and that is a
    juce::String concatenation."""
    stack = ctx.stacks[run.start]
    prev, nxt = ctx.prev_tok(run.start), ctx.next_tok(run.end)
    if stack and stack[-1][0] == "(":
        # `<< (ok ? "…" : "…")`: a bare parenthesis that is itself the operand
        _, paren = stack[-1]
        if ctx.callee(paren) or ctx.prev_tok(paren) != "<<":
            return False
        if prev not in ("?", ":") or nxt not in (":", ")"):
            return False
        if len(stack) > 1 and stack[-2][0] != "{":
            return False
        head = _statement_head(ctx, paren, stack[-2][1] + 1 if len(stack) > 1 else 0)
    else:
        if prev != "<<" or nxt not in ("<<", ";"):
            return False
        head = _statement_head(ctx, run.start, stack[-1][1] + 1 if stack else 0)
    return re.search(r"\bstd::(cout|cerr)\s*<<", head) is not None


_DECODE_BEFORE = re.compile(r"\b(?:fromUTF8|CharPointer_UTF8)\s*\(\s*$")


def _raw_bytes_lambda_arg(run: Run, ctx: Context) -> bool:
    """A whole argument to a local lambda whose parameter there is `const char*`, where the
    lambda's body does only two things with it: stream it (`<< what`) or decode it
    (`fromUTF8 (tip)`). Anything else — `juce::String (what)`, `*why = reason`, handing it
    on — and the literal is judged like any other."""
    stack = ctx.stacks[run.start]
    if not stack or stack[-1][0] != "(":
        return False
    open_paren = stack[-1][1]
    name = ctx.callee(open_paren)
    if not re.fullmatch(r"\w+", name or ""):
        return False
    if ctx.prev_tok(run.start) not in ("(", ",") or ctx.next_tok(run.end) not in (",", ")"):
        return False
    index = depth = 0
    for ch in ctx.mask[open_paren + 1:run.start]:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            index += 1
    defs = [d for d in ctx.lambdas if d.name == name and d.at < open_paren]
    if not defs or index >= len(defs[-1].params):
        return False
    param, body = defs[-1].params[index], defs[-1].body
    if not re.search(r"\bconst\s+char\s*\*", param):
        return False
    pname = re.findall(r"\w+", param.split("=")[0])[-1]
    # (a member of the same name — `c.label = …` — is not a use of the parameter)
    uses = list(re.finditer(rf"(?<!\.)(?<!->)\b{re.escape(pname)}\b", body))

    def harmless(u: re.Match[str]) -> bool:
        before = body[:u.start()].rstrip()
        return before.endswith("<<") or (
            _DECODE_BEFORE.search(body[:u.start()]) is not None
            and body[u.end():].lstrip().startswith(")"))

    return bool(uses) and all(harmless(u) for u in uses)


def _brace_heads(run: Run, ctx: Context) -> list[str]:
    heads = []
    for ch, pos in ctx.stacks[run.start]:
        if ch == "{":
            k = max(ctx.mask.rfind(c, 0, pos) for c in ";{}")
            heads.append(re.sub(r"\s+", " ", ctx.mask[k + 1:pos]).strip())
    return heads


@dataclass
class Violation:
    file: str
    line: int
    text: str
    run: Run

    def __str__(self) -> str:
        return f"{self.file}:{self.line}: {self.text}"


def violations_in(src: str, rel: str = "<snippet>") -> list[Violation]:
    ctx = context_for(src)
    found: list[Violation] = []
    for run in ctx.runs:
        if not run.high:
            continue
        if any(lit.prefix.startswith("@") for lit in run.lits):
            continue                      # an Objective-C NSString literal
        if all(lit.prefix.startswith("u8") for lit in run.lits if lit.high):
            continue                      # char8_t: juce::String decodes it as UTF-8
        if _decoded(run, ctx) or _ostream_operand(run, ctx) or _raw_bytes_lambda_arg(run, ctx):
            continue
        heads = _brace_heads(run, ctx)
        if any(t.file == rel and any(re.search(t.head, h) for h in heads)
               for t in RAW_BYTE_TABLES):
            continue
        text = src[run.start:run.end]
        if any(f == rel and needle in text for f, needle, _ in ALLOWED_LITERALS):
            continue
        shown = re.sub(r"\s+", " ", text)
        found.append(Violation(rel, ctx.line_of(run.start),
                               shown if len(shown) <= 110 else shown[:107] + "...", run))
    return found


def source_files() -> list[Path]:
    return sorted(p for root in ROOTS for p in root.rglob("*") if p.suffix in EXTS)


# --------------------------------------------------------------------------- the gate

def test_no_utf8_literal_reaches_a_juce_string_as_latin1():
    found: list[Violation] = []
    for path in source_files():
        found += violations_in(path.read_text(encoding="utf-8"),
                               path.relative_to(REPO).as_posix())
    assert not found, (
        f"{len(found)} string literal(s) carry non-ASCII bytes outside a UTF-8 decoder. "
        "juce::String (const char*) reads them as Latin-1, so an em dash reaches the "
        "screen (and the agent) as three junk characters. Write "
        'juce::String::fromUTF8 ("...\\xe2\\x80\\x94...") instead:\n  '
        + "\n  ".join(str(v) for v in found)
    )


# A character encoded TWICE is valid UTF-8, sits happily inside `fromUTF8 (...)`, and passes
# everything above: an em dash saved through a Latin-1 round trip is `Ã¢` + two C1 control
# codes, and decodes to exactly that. Two refusals the agent reads carried one (pre-merge
# review 2026-10-04). C1 controls (U+0080-U+009F) have no business in a source file, as
# characters or as the escaped bytes that spell them, so their presence is the signature;
# the cp1252 spelling of the same accident starts `â€`.
_C1 = re.compile("[\u0080-\u009f]")
_C1_ESCAPED = re.compile(r"\\xc2\\x[89][0-9a-f]", re.I)
_CP1252 = "\u00e2\u20ac"


def double_encoded_lines(text: str) -> list[int]:
    return [n for n, line in enumerate(text.split("\n"), 1)
            if _C1.search(line) or _C1_ESCAPED.search(line) or _CP1252 in line]


def test_no_character_is_encoded_twice():
    found = [f"{path.relative_to(REPO).as_posix()}:{n}"
             for path in source_files()
             for n in double_encoded_lines(path.read_text(encoding="utf-8"))]
    assert not found, (
        f"{len(found)} line(s) hold a double-encoded character (an em dash that reads "
        "'a-circumflex' plus two invisible control codes). Re-type it as "
        '"\\xe2\\x80\\x94" inside fromUTF8:\n  ' + "\n  ".join(found))


@pytest.mark.parametrize("line", [
    'err (juce::String::fromUTF8 ("rendering \u00e2\u0080\u0094 try again"));',
    'err (juce::String::fromUTF8 ("rendering \\xc3\\xa2\\xc2\\x80\\xc2\\x94 try again"));',
    'setStatus (juce::String::fromUTF8 ("\u00e2\u20ac\u201d runs later"));',
])
def test_the_double_encoding_scan_flags_the_bug(line: str):
    assert double_encoded_lines(line) == [1]


def test_the_double_encoding_scan_accepts_a_real_em_dash():
    assert double_encoded_lines('juce::String::fromUTF8 ("a \\xe2\\x80\\x94 b") // \u2014 \u00b7 \u2026') == []


def test_raw_byte_tables_are_still_decoded_where_they_are_read():
    """A `const char*` table is exempt only because its reader decodes it. If that line
    goes, the table is the bug again — so each entry names the line, and it must exist."""
    for table in RAW_BYTE_TABLES:
        src = (REPO / table.file).read_text(encoding="utf-8")
        ctx = context_for(src)
        used = any(re.search(table.head, h)
                   for run in ctx.runs if run.high for h in _brace_heads(run, ctx))
        assert used, f"{table.file}: no non-ASCII literal sits under /{table.head}/ any more — drop the entry"
        for file, needle in table.decoded_by:
            assert needle in (REPO / file).read_text(encoding="utf-8"), (
                f"{table.file} /{table.head}/ is exempt because {file} decodes it with "
                f"`{needle}`, and that text is gone. Decode the table's strings as UTF-8 "
                "where they are read, then update this entry.")
    for file, needle, _ in ALLOWED_LITERALS:
        assert needle in (REPO / file).read_text(encoding="utf-8"), (
            f"{file}: the allowed literal `{needle}` is gone — drop the entry")


# --------------------------------------------------------------------------- the scanner's own controls

DASH = "—"

# The bug, in every shape it shipped in. Each must be flagged.
BROKEN = [
    f'void f() {{ setStatus ("auto mix {DASH} " + brief); }}',                 # the Auto Mix line
    'void f() { return err ("busy \\xe2\\x80\\x94 try again in a moment"); }',  # escaped: same bytes
    f'juce::String why() {{ return "turn Warp on first {DASH} a groove rides its warp"; }}',
    f'void f() {{ status += " {DASH} and applied"; }}',
    f'void f() {{ s << " \\xc2\\xb7 MPE"; }}',
    f'void f() {{ m.addItem (2, "Export\\xe2\\x80\\xa6"); }}',
    f'void f() {{ juce::String a ("x {DASH} y"); }}',
    f'void f() {{ t.setTooltip (label + " {DASH} "\n                 "second half"); }}',  # concatenated run
    f'void f() {{ t.setTooltip ("first half "\n                 "{DASH} second half"); }}',
    f'void f() {{ std::cout << name + " {DASH} not an ostream operand"; }}',   # + binds before <<
    f'void f() {{ x = ok ? juce::String ("fine") : "broke {DASH} " + why; }}',
    'void f() { g.drawText ("RT60 " "\\xc2\\xb7" " TRUE", r, c); }',
    'void f() { label.setText ("\\342\\200\\224", juce::dontSendNotification); }',  # octal
    'void f() { label.setText ("\\u2014", juce::dontSendNotification); }',          # universal name
    f'void f() {{ juce::String::fromUTF8 ("ok"); say (juce::String ("x {DASH} y")); }}',
    f'void f() {{ auto check = [&ok] (bool c, const juce::String& what) {{ std::cout << what; }};\n'
    f'           check (true, "label {DASH} becomes a String"); }}',
    f'void f() {{ auto note = [&] (const char* what) {{ feed.add (juce::String (what)); }};\n'
    f'           note ("the lambda converts it {DASH} still the bug"); }}',
    f'static const char* tips[] = {{ "Appearance {DASH} an unregistered table" }};',
    f'void f() {{ auto fail = [why] (const char* reason) {{ *why = reason; return juce::File(); }};\n'
    f'           return fail ("that file exists {DASH} assigned through the pointer"); }}',
    f'void f() {{ auto tip = [this] (const char* cap, const char* text) {{ k.caption = cap; k.setTooltip (juce::String::fromUTF8 (text)); }};\n'
    f'           tip ("CUT {DASH} the caption is NOT decoded", "plain"); }}',
]

# Correct code. None may be flagged.
SOUND = [
    f'void f() {{ setStatus (juce::String::fromUTF8 ("auto mix {DASH} ") + brief); }}',
    'void f() { setStatus (juce::String::fromUTF8 ("auto mix \\xe2\\x80\\x94 ") + brief); }',
    f'void f() {{ t.setTooltip (juce::String::fromUTF8 (\n      "a tip that starts on the next line {DASH} "\n      "and continues")); }}',
    f'void f() {{ x = juce::String::fromUTF8 (ok ? "yes {DASH}" : "no {DASH}"); }}',
    f'void f() {{ x = juce::String (juce::CharPointer_UTF8 ("x {DASH} y")); }}',
    f'void f() {{ x = u8"x {DASH} y"; }}',
    f'void f() {{ const auto u8 = [] (const char* t) {{ return juce::String::fromUTF8 (t); }};\n'
    f'           x = u8 ("x {DASH} y") + name; }}',
    f'void f() {{ std::cout << "[test] a {DASH} b" << n << " c {DASH} d" << std::endl; }}',
    f'void f() {{ if (bad) std::cerr << "dropped {DASH} busy\\n"; }}',
    f'void f() {{ std::cout << "x " << (ok ? "PASS {DASH} fine" : "FAIL {DASH} not") << std::endl; }}',
    f'void f() {{ auto check = [&ok] (bool c, const char* what) {{ if (! c) std::cout << "FAIL: " << what << std::endl; }};\n'
    f'           check (n == 2, "two clips {DASH} one per lane"); }}',
    f'void f() {{ auto tip = [this] (const char* cap, const char* text) {{ k.caption = cap; k.setTooltip (juce::String::fromUTF8 (text)); }};\n'
    f'           tip ("CUT", "Cutoff {DASH} decoded by the lambda"); }}',
    f'// a comment may say anything {DASH} it is not a string\nvoid f() {{ setStatus ("plain"); }}',
    f'/* "quoted {DASH} inside a block comment" */ void f() {{}}',
    'void f() { setStatus ("ASCII only, with an escape\\n and a \\"quote\\""); }',
    f'void f() {{ NSString* s = @"Objective-C {DASH} an NSString literal"; }}',
    'void f() { char c = \'"\'; setStatus ("after a quote character literal"); }',
]


@pytest.mark.parametrize("snippet", BROKEN)
def test_scanner_flags_the_bug(snippet: str):
    assert violations_in(snippet), f"the scanner missed the bug in:\n{snippet}"


@pytest.mark.parametrize("snippet", SOUND)
def test_scanner_accepts_correct_code(snippet: str):
    found = violations_in(snippet)
    assert not found, f"the scanner flagged correct code:\n{snippet}\n{found[0]}"


def test_scanner_reports_the_line_of_a_literal_below_its_call():
    src = 'void f()\n{\n    setStatus (name +\n               " \\xe2\\x80\\x94 gone");\n}\n'
    (only,) = violations_in(src, "x.h")
    assert (only.file, only.line) == ("x.h", 4)
