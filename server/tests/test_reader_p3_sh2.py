"""Reader P3s, batch Sh2 (2026-10-07): packaging, scripts, the workflow files.

S7.R11  the engine workflow's path filter missed files its own tests pin
S7.R15  the frozen brain shipped test and developer tooling, and no notice named its Python
S7.R16  the scratch-project launcher's unsafe-directory guard was defeated by a trailing slash
S9.R14  the pre-commit secret guard missed four bare token shapes

Everything here is a script or a build recipe, so every test RUNS it (in a scratch tree, with a
stand-in for what would do harm) or reads the file it pins. Nothing here freezes a brain: what
only a freeze can prove is `verify_bundled_brain.py`'s, and the record says what it saw.
"""
from __future__ import annotations

import fnmatch
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SIDECAR = REPO / "daw" / "sidecar"
SPEC = SIDECAR / "mixagent_daw_brain.spec"


def _exe(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip())
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ====================================================================================
# S7.R16 - the launcher wipes a scratch project, and only that
# ====================================================================================

REFUSAL = "Refusing to wipe"


class Launcher:
    """A scratch copy of the launcher in a scratch 'repository', a scratch HOME, and an `rm`
    that only writes down what it was asked to remove - so the UNFIXED script can be shown an
    unsafe folder without losing one."""

    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.repo = tmp / "work" / "repo"
        self.home = tmp / "people" / "home"
        (self.repo / "daw").mkdir(parents=True)
        (self.repo / "daw" / "notes").mkdir()
        (self.repo / "daw" / "notes" / "work.txt").write_text("uncommitted work\n")
        (self.home / "Documents").mkdir(parents=True)
        (self.home / "Documents" / "letter.txt").write_text("dear\n")
        shutil.copy(REPO / "daw" / "fresh.sh", self.repo / "daw" / "fresh.sh")
        self.log = tmp / "rm.log"
        _exe(tmp / "bin" / "rm", f"""
            #!/bin/bash
            printf '%s\\n' "$*" >> "{self.log}"
        """)

    def run(self, fresh_dir: str, real_rm: bool = False, *args: str) -> subprocess.CompletedProcess:
        env = {k: v for k, v in os.environ.items() if not k.startswith("DAW_FRESH")}
        scratch = self.tmp / "work" / "tmpdir"
        scratch.mkdir(exist_ok=True)
        env.update(HOME=str(self.home), DAW_FRESH_DIR=fresh_dir, DAW_FRESH_PREPARE_ONLY="1",
                   TMPDIR=str(scratch) + "/")
        if not real_rm:
            env["PATH"] = f"{self.tmp / 'bin'}:{env['PATH']}"
        return subprocess.run(["bash", str(self.repo / "daw" / "fresh.sh"), *args],
                              capture_output=True, text=True, env=env, timeout=60)

    def asked_to_remove(self) -> str:
        return self.log.read_text() if self.log.exists() else ""

    def project(self, name: str = "scratch") -> Path:
        """A folder that looks like what the app leaves: a Zammoth project."""
        folder = self.tmp / "work" / name
        (folder / "Recorded").mkdir(parents=True)
        (folder / "library.tracktion").write_text("<PROJECT/>\n")
        (folder / "session.tracktionedit").write_text("<EDIT/>\n")
        (folder / "old.wav").write_bytes(b"RIFF")
        return folder


@pytest.fixture()
def launcher(tmp_path: Path) -> Launcher:
    return Launcher(tmp_path.resolve())


IS_REPO = "the repository is that folder, or is inside it"
IS_HOME = "your home folder is that folder, or is inside it"
TOO_HIGH = "too close to the root of the disk"


def _unsafe(launcher: Launcher) -> dict[str, tuple[str, str]]:
    """Each unsafe folder, and the reason the launcher must give for it: the rules overlap (a
    folder of files is refused for its contents wherever it is), so a test that read only
    "refused" could not tell which rule did it - and a rule taken out would not show."""
    repo, home = str(launcher.repo), str(launcher.home)
    link = launcher.tmp / "work" / "link-to-repo"
    link.symlink_to(launcher.repo)
    music = launcher.tmp / "work" / "music"
    music.mkdir()
    (music / "song.aif").write_bytes(b"FORM")
    (launcher.tmp / "work" / "a-file").write_text("x\n")
    return {
        "the repository with a trailing slash": (repo + "/", IS_REPO),
        "the repository with two": (repo + "//", IS_REPO),
        "home with a trailing slash": (home + "/", IS_HOME),
        "the folder the repository is in": (str(launcher.repo.parent), IS_REPO),
        "the folder home is in, with a slash": (str(launcher.home.parent) + "/", IS_HOME),
        "a folder inside the repository": (repo + "/daw", "it is inside the repository"),
        "a symlink to the repository": (str(link), IS_REPO),
        "the repository through a ..": (repo + "/daw/..", "no . or .. in it"),
        "a folder directly in home": (home + "/Documents", "directly in your home folder"),
        "/tmp with a trailing slash": ("/tmp/", TOO_HIGH),
        "the root, twice": ("//", "that is the root of the disk"),
        "/private/tmp": ("/private/tmp", TOO_HIGH),
        "the temporary folder itself": (str(launcher.tmp / "work" / "tmpdir") + "/",
                                        "the temporary folder itself"),
        "a file": (str(launcher.tmp / "work" / "a-file"), "it is a file, not a folder"),
        "a folder of someone's files that is not a project":
            (str(music), "it holds files and is not a Zammoth project"),
    }


@pytest.mark.parametrize("which", [
    "the repository with a trailing slash", "the repository with two",
    "home with a trailing slash", "the folder the repository is in",
    "the folder home is in, with a slash", "a folder inside the repository",
    "a symlink to the repository", "the repository through a ..",
    "a folder directly in home", "/tmp with a trailing slash", "the root, twice",
    "/private/tmp", "the temporary folder itself", "a file",
    "a folder of someone's files that is not a project"])
def test_the_launcher_refuses_to_wipe(launcher: Launcher, which: str):
    """`DAW_FRESH_DIR=<repo>/` - what tab completion writes - passed the guard, which compared
    the raw string, and the next line was `rm -rf`: the repository, uncommitted work and the
    engine clone's un-exported patches with it."""
    target, reason = _unsafe(launcher)[which]
    out = launcher.run(target)
    assert launcher.asked_to_remove() == "", \
        f"{which}: the launcher went on to `rm {launcher.asked_to_remove().strip()}`"
    assert out.returncode == 2 and REFUSAL in out.stderr, (out.returncode, out.stderr[-400:])
    assert reason in out.stderr, f"{which}: refused, but for another reason: {out.stderr.strip()}"


def test_the_launcher_refuses_the_repository_spelled_in_another_case(launcher: Launcher):
    other = str(launcher.repo).replace("/work/repo", "/WORK/Repo")
    if not Path(other).exists():
        pytest.skip("a case-sensitive volume: the other spelling is another folder")
    out = launcher.run(other)
    assert launcher.asked_to_remove() == "" and out.returncode == 2 and IS_REPO in out.stderr


def test_a_folder_inside_the_repository_in_another_case_and_not_there_yet(launcher: Launcher):
    """`mkdir -p` would make it inside the repository: found by walking up to a folder that IS."""
    other = str(launcher.repo).replace("/work/repo", "/WORK/Repo") + "/daw/new/scratch"
    if not Path(str(launcher.repo).replace("/work/repo", "/WORK/Repo")).exists():
        other = str(launcher.repo) + "/daw/new/scratch"
    out = launcher.run(other)
    assert out.returncode == 2 and "it is inside the repository" in out.stderr, out.stderr
    assert launcher.asked_to_remove() == ""


def test_the_plain_refusals_still_hold(launcher: Launcher):
    """Controls: what the old guard did refuse, it still does."""
    for target in (str(launcher.repo), str(launcher.home), "/", "/tmp"):
        out = launcher.run(target)
        assert out.returncode == 2 and REFUSAL in out.stderr, target
    out = launcher.run("relative/folder")
    assert out.returncode == 2 and "absolute" in out.stderr
    assert launcher.asked_to_remove() == ""


@pytest.mark.parametrize("spelling", ["{p}", "{p}/", "{p}//"])
def test_the_launcher_still_wipes_a_scratch_project(launcher: Launcher, spelling: str):
    """...and a guard that refused everything would pass every test above."""
    project = launcher.project()
    out = launcher.run(spelling.format(p=project), True)
    assert out.returncode == 0, out.stderr[-400:]
    assert project.is_dir() and list(project.iterdir()) == []
    assert (launcher.repo / "daw" / "notes" / "work.txt").is_file()


def test_the_launcher_makes_a_folder_that_is_not_there_and_takes_an_empty_one(launcher: Launcher):
    fresh = launcher.tmp / "work" / "not-yet" / "project"
    out = launcher.run(str(fresh) + "/", True)
    assert out.returncode == 0 and fresh.is_dir(), out.stderr[-400:]
    out = launcher.run(str(fresh), True)                      # now there, and empty
    assert out.returncode == 0 and list(fresh.iterdir()) == [], out.stderr[-400:]


def test_the_demo_stems_alone_are_a_folder_the_launcher_made(launcher: Launcher):
    """`--demo --prepare-only` leaves four stems and no project file; the next run takes it."""
    fresh = launcher.tmp / "work" / "demo"
    fresh.mkdir()
    for name in ("vocals.wav", "drums.wav"):
        (fresh / name).write_bytes(b"RIFF")
    out = launcher.run(str(fresh), True)
    assert out.returncode == 0 and list(fresh.iterdir()) == [], out.stderr[-400:]
    (fresh / "vocals.wav").write_bytes(b"RIFF")
    (fresh / "my-song.wav").write_bytes(b"RIFF")              # control: not only the stems
    out = launcher.run(str(fresh))
    assert out.returncode == 2 and REFUSAL in out.stderr and launcher.asked_to_remove() == ""


def test_a_symlink_to_a_scratch_project_wipes_the_project_it_points_at(launcher: Launcher):
    project = launcher.project("real")
    link = launcher.tmp / "work" / "link"
    link.symlink_to(project)
    out = launcher.run(str(link) + "/", True)
    assert out.returncode == 0, out.stderr[-400:]
    assert project.is_dir() and list(project.iterdir()) == []


def test_the_default_scratch_folder_is_one_the_guard_accepts():
    """`dawdev` with no variable set must keep working: the default passes its own guard."""
    text = (REPO / "daw" / "fresh.sh").read_text()
    assert 'FRESH="${DAW_FRESH_DIR:-/tmp/daw-fresh}"' in text


# ====================================================================================
# S9.R14 - the secret guard knows the token shapes this repository's services issue
# ====================================================================================

@pytest.fixture(scope="module")
def guard():
    return _load("check_secrets_sh2", REPO / "scripts" / "check-secrets.py")


def _run(*parts: str) -> str:
    """A synthetic value assembled at run time: no token-shaped literal lives in this file."""
    return "".join(parts)


_BODY = "aB3dE5gH7jK9mN1pQ2rS4tU6vW8xY0zA"            # 32 mixed characters, not a credential


def _tokens() -> dict[str, str]:
    jwt = _run("eyJ", _BODY[:20], ".", "eyJ", _BODY[:24], ".", _BODY[:22])
    return {
        "a Stripe webhook secret": _run("wh", "sec_", _BODY),
        "a Resend key": _run("r", "e_", _BODY[:8], "_", _BODY[:24]),
        # short and url-safe: what the bare shape (forty characters, no underscore) misses
        "a Fly token with its scheme": _run("Fly", "V1 ", "fm2", "_", _BODY[:14], "_", _BODY[:12]),
        "a bare Fly token": _run("fm2", "_", _BODY, "+/", _BODY),
        "a Fly org token": _run("fo1", "_", _BODY, _BODY),
        "a JWT": jwt,
        "a Supabase secret key": _run("sb", "_secret_", _BODY),
    }


@pytest.mark.parametrize("which", ["a Stripe webhook secret", "a Resend key",
                                   "a Fly token with its scheme", "a bare Fly token",
                                   "a Fly org token", "a JWT", "a Supabase secret key"])
@pytest.mark.parametrize("line", ['value = "{t}"', "curl -H 'Authorization: Bearer {t}'",
                                  "- paste {t} into the dashboard"])
def test_the_guard_flags_a_bare_token(guard, which: str, line: str):
    """A webhook secret, a mail key, a deploy token or a service JWT pasted into a .py, a .md
    or a test - not on a `SECRET=` line - passed the hook: only eight provider shapes were
    known, and none of these four services' was one."""
    problems = guard.scan_text("notes.md", line.format(t=_tokens()[which]))
    assert problems, f"{which} passed the guard on: {line}"


def test_the_guard_flags_a_jwt_on_a_secret_named_line(guard):
    """The reader's case: `SUPABASE_SERVICE_ROLE_KEY=<jwt>` passed too - the dots fail the
    name=value rule's own idea of what a secret looks like."""
    line = "SUPABASE_SERVICE_ROLE_KEY=" + _tokens()["a JWT"]
    assert guard.scan_text(".env.production", line)


@pytest.mark.parametrize("line", [
    "re_compile_flags = re.IGNORECASE | re.MULTILINE",
    "from re import compile as re_compile_patterns_for_everything",
    "re_exported_names_thatarequitelongindeedforaname = 1",
    "re_patterns_thatmatchsomethingverylongindeedhere = build()",
    "fm2_Level_of_the_second_modulator_in_this_patch_and_more = 0.5",
    "fm2_operator_ratio_for_the_second_carrier_envelope = 2.0",
    "fm1_modulation_index, fm2_modulation_index = 1.0, 2.0",
    "the header is `eyJhbGciOi...` and three parts follow",
    "whsec_ is the prefix Stripe gives a webhook secret",
    "set STRIPE_WEBHOOK_SECRET to the value that starts whsec_",
    "sb_secret_key_name = 'SUPABASE_SECRET'",
    "sb_secret_key_environment_variable_name = 'SUPABASE_SECRET'",
    "token = jwt.encode(claims, key).split('.')",
])
def test_the_guard_does_not_cry_wolf(guard, line: str):
    """A hook that fires on ordinary identifiers is `--no-verify`d for good, which its own
    comments call worse than no hook: the new shapes are narrow."""
    assert guard.scan_text("module.py", line) == [], line


def test_the_marker_still_lets_a_line_through(guard):
    line = f'FIXTURE = "{_tokens()["a JWT"]}"  # {guard.ALLOW_MARKER}'
    assert guard.scan_text("test_x.py", line) == []


def test_the_shapes_it_already_knew_are_still_flagged(guard):
    """Controls: the eight it had."""
    for value in (_run("sk-", "ant-", _BODY), _run("sk", "_live_", _BODY),
                  _run("gh", "p_", _BODY), _run("AK", "IA", "ABCDEFGHIJKLMNOP")):
        assert guard.scan_text("a.py", f"x = '{value}'"), value


def test_the_tracked_tree_is_clean_under_the_guard():
    """`--all` over what is tracked: a new shape that fires on the repository's own files would
    block every commit that touches them."""
    out = subprocess.run([sys.executable, str(REPO / "scripts" / "check-secrets.py"), "--all"],
                         cwd=REPO, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-3000:]


# ====================================================================================
# S7.R11 - which tests a diff owes, and the workflow's path filter
# ====================================================================================

@pytest.fixture(scope="module")
def owed_by_diff():
    return _load("owed_by_diff_sh2", Path(__file__).resolve().parent / "owed_by_diff.py")


def test_a_change_outside_the_python_folders_owes_the_tests_that_read_it(owed_by_diff):
    """The suite reads the manual, the tool docs, the freeze recipe and the launch checklist.
    A change to one of them reached no gate: the workflow's filter did not list them, and the
    proportionate-testing table said a docs change owes nothing."""
    owed = owed_by_diff.owed
    assert "test_api_key_license.py" in owed(["docs/17-user-manual.md"])
    assert "test_knowledge_pack.py" in owed(["docs/03-mcp-tools.md"])
    assert "test_brain_stamp.py" in owed(["daw/sidecar/stamp.sh"])
    assert "test_bundle_specs.py" in owed(["daw/sidecar/mixagent_daw_brain.spec"])
    assert "test_reader_p3_sh2.py" in owed(["scripts/check-secrets.py"])
    assert "test_reader_p3_sh2.py" in owed(["daw/fresh.sh"])
    # named as `SCRIPTS / "notarize.sh"`: the module names the folder, so it is owed
    assert "test_reader_p2_sh1.py" in owed(["daw/scripts/notarize.sh"])


def test_a_change_no_test_reads_owes_none(owed_by_diff):
    """Controls: the answer is not "everything"."""
    assert owed_by_diff.owed(["ui-mockups/index.html"]) == []
    assert owed_by_diff.owed(["no/such/file.txt"]) == []
    assert "test_api_key_license.py" not in owed_by_diff.owed(["daw/sidecar/stamp.sh"])


def _filter_lists() -> dict[str, list[str]]:
    """The engine workflow's two path lists. They are commented out (the workflow is manual-only
    and disabled on GitHub), so they are read as comments."""
    lists: dict[str, list[str]] = {}
    current = None
    for raw in (REPO / ".github" / "workflows" / "engine.yml").read_text().splitlines():
        line = raw.strip().lstrip("#").strip()
        if line in ("push:", "pull_request:"):
            current = lists.setdefault(line.rstrip(":"), [])
        elif current is not None:
            m = re.fullmatch(r'-\s*"([^"]+)"', line)
            if m:
                current.append(m.group(1))
            elif raw.strip() and not raw.strip().startswith("#"):
                current = None
    return lists


def _covered(path: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        if pattern.endswith("/**"):
            if path == pattern[:-3] or path.startswith(pattern[:-2]):
                return True
        elif fnmatch.fnmatchcase(path, pattern):
            return True
    return False


def test_the_workflows_path_filter_covers_what_its_suite_reads(owed_by_diff):
    """If the workflow is ever switched back on, a push that edits a file a test pins must run
    the suite. The filter is checked against what the tests NAME, so it cannot drift again."""
    lists = _filter_lists()
    assert set(lists) == {"push", "pull_request"} and lists["push"], lists
    assert lists["push"] == lists["pull_request"], "the two triggers filter differently"
    tracked = set(subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True,
                                 text=True).stdout.splitlines())
    if not tracked:
        pytest.skip("not a git checkout")
    folders = {str(Path(f).parent) for f in tracked}
    for f in list(folders):
        folders.update(str(p) for p in Path(f).parents)
    missed = {}
    for test, paths in owed_by_diff.suite_reads().items():
        for path in paths:
            if path not in tracked and path not in folders:
                continue                             # a build folder, an untracked clone
            probe = path if path in tracked else path + "/x"
            if not _covered(probe, lists["push"]):
                missed.setdefault(path, test)
    assert not missed, f"a push that edits these runs nothing, and a test reads them: {missed}"


def test_the_path_filter_is_still_a_filter():
    """Control for the test above: the fix is not "run on everything"."""
    patterns = _filter_lists()["push"]
    assert not _covered("ui-mockups/index.html", patterns)
    assert not _covered("site-concept/index.html", patterns)


def test_the_gates_skill_says_what_a_docs_change_owes():
    text = (REPO / ".claude" / "skills" / "quality-gates" / "SKILL.md").read_text()
    assert "owed_by_diff.py" in text
    row = next(line for line in text.splitlines() if line.startswith("| docs, skills"))
    assert "nothing" not in row.split("|")[2], row


# ====================================================================================
# S7.R15 a - the frozen brain carries no test or developer tooling
# ====================================================================================

DEV_PACKAGES = ("pytest", "_pytest", "hypothesis", "mypy", "fastapi")
#: What a reader's sketch also listed, and the brain IMPORTS: FastMCP's server module imports
#: starlette at its top, its logging imports rich (which imports pygments), and the bridge
#: imports the benchmark's in-memory engine at five sites.
NEEDED = ("starlette", "rich", "pygments", "orchestrator.benchmark", "anyio", "mcp")


def _excludes() -> list[str]:
    m = re.search(r"excludes\s*=\s*\[(.*?)\]", SPEC.read_text(encoding="utf-8"), re.S)
    assert m, "no excludes=[...] list in the spec"
    return re.findall(r'"([^"]+)"', m.group(1))


@pytest.mark.parametrize("package", DEV_PACKAGES)
def test_the_spec_keeps_dev_tooling_out_of_the_brain(package: str):
    """The archive of the shipped brain held _pytest (71 modules), hypothesis (89), mypy and
    fastapi (38): each is imported by some library under a guard PyInstaller follows (anyio's
    pytest plugin, pydantic's hypothesis and mypy plugins, the hub's optional OAuth routes),
    and none by the brain."""
    assert package in _excludes(), f"{package} is frozen into the shipped brain"


@pytest.mark.parametrize("package", NEEDED)
def test_the_spec_does_not_exclude_what_the_brain_imports(package: str):
    assert package not in _excludes(), f"{package} is imported by the running brain"


_SESSION_PROBE = textwrap.dedent("""
    import importlib.abc
    import sys

    BLOCKED = set(sys.argv[1].split(","))


    class NotInThisBundle(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name in BLOCKED or name.split(".")[0] in BLOCKED:
                raise ImportError(f"not in a shipped brain: {name}")


    sys.meta_path.insert(0, NotInThisBundle())
    import orchestrator.daw_server                      # what the DAW starts
    import onnxruntime, demucs_onnx, huggingface_hub, soxr   # the spec's lazy imports
    from huggingface_hub import hf_hub_download        # the separator's download call
    from orchestrator.daw_server import Brain

    from pathlib import Path
    brain = Brain(Path(sys.argv[2]))
    boot = brain.boot_sync()
    mix = brain.auto_mix()              # the in-memory engine (mcp, anyio, pydantic) + a render
    print("SESSION", "chains" in boot, len(mix.get("moves") or []), bool(mix.get("readout")))
    print("LOADED", ",".join(sorted(m for m in BLOCKED if m in sys.modules)))
""")


def _run_session(blocked: list[str], project: Path) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items()
           if k not in ("MIX_AGENT_CLOUD_URL", "MIX_AGENT_LLM_BACKEND")}
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), str(REPO / "server")])
    env["ANTHROPIC_API_KEY"] = "placeholder-not-a-key"
    env["MIX_AGENT_DATA_DIR"] = str(project.parent / "data")
    return subprocess.run([sys.executable, "-c", _SESSION_PROBE, ",".join(blocked), str(project)],
                          cwd=str(REPO), env=env, capture_output=True, text=True, timeout=300)


@pytest.mark.slow
def test_a_brain_session_needs_nothing_the_spec_excludes(demo_project: Path):
    """Every package the spec leaves out is made unimportable, and then a brain is started,
    opens a project, mixes it through the in-memory engine and renders. An exclude that the
    running brain needs fails here, in seconds, and not as a brain that dies in a user's app."""
    excluded = [e for e in _excludes() if e != "orchestrator.knowledge_content"]
    out = _run_session(excluded, demo_project)
    assert out.returncode == 0, out.stderr[-3000:]
    seen = dict(line.split(" ", 1) for line in out.stdout.splitlines() if " " in line)
    booted, moves, mixed = seen["SESSION"].split()
    assert booted == "True" and int(moves) >= 1 and mixed == "True", seen


@pytest.mark.slow
def test_the_session_probe_fails_when_something_needed_is_left_out(demo_project: Path):
    """Control: the reader's wider list. With starlette out, the brain does not start."""
    out = _run_session(["starlette"], demo_project)
    assert out.returncode != 0 and "not in a shipped brain: starlette" in out.stderr


# ====================================================================================
# S7.R15 b - the brain says what third-party Python it is made of
# ====================================================================================

@pytest.fixture(scope="module")
def notices():
    return _load("bundle_notices_sh2", SIDECAR / "bundle_notices.py")


@pytest.fixture(scope="module")
def verifier():
    return _load("verify_bundled_brain_sh2", SIDECAR / "verify_bundled_brain.py")


def _site(module: str) -> str:
    return importlib.util.find_spec(module).origin


def _found(notices, modules: list[str], files: list[str] = ()):
    pairs = [(m, _site(m)) for m in modules]
    return notices.attribute(pairs, list(files), [REPO / "orchestrator", REPO / "server"])


def test_every_frozen_distribution_is_named_with_its_licence_and_its_text(notices):
    """The notices named numpy and onnxruntime once, in passing, and none of the rest: scipy
    (with libgfortran and libquadmath), cryptography's OpenSSL, certifi under the MPL, the
    Python runtime. Now the freeze writes the list from what it holds."""
    found = _found(notices, ["numpy", "scipy", "onnxruntime", "cryptography", "certifi",
                             "pedalboard", "anthropic", "soxr", "mix_engine", "json"])
    by_name = {notices.normal(d.name): d for d in found["dists"]}
    assert set(by_name) >= {"numpy", "scipy", "onnxruntime", "cryptography", "certifi",
                            "pedalboard", "anthropic", "soxr"}
    for name, dist in by_name.items():
        assert dist.version and dist.licence, f"{name}: no version or no licence declared"
        assert dist.texts, f"{name}: no licence text taken from its wheel"
    assert "MPL" in by_name["certifi"].licence
    assert "GPL" in by_name["pedalboard"].licence.upper() or "GNU" in by_name["pedalboard"].licence
    assert found["own_modules"] == 1 and found["runtime_modules"] == 1      # mix_engine, json
    assert found["unattributed"] == []
    text = notices.render(found)
    assert "numpy" in text and "Redistribution and use in source and binary forms" in text
    assert "PYTHON SOFTWARE FOUNDATION LICENSE" in text, "the runtime's own licence is missing"
    assert by_name["onnxruntime"].version in text


def test_a_venv_inside_the_repository_is_not_this_programs_own_code(notices):
    """`server/.venv` sits under `server/`, an own root: checked in the wrong order, every
    third-party module counted as the brain's own and the notices listed three distributions."""
    found = _found(notices, ["numpy", "mix_engine"])
    assert [notices.normal(d.name) for d in found["dists"]] == ["numpy"]
    assert found["own_modules"] == 1


def test_a_binary_from_a_wheel_is_credited_to_that_wheel(notices):
    import scipy
    libs = sorted((Path(scipy.__file__).parent / ".dylibs").glob("*.dylib"))
    if not libs:
        pytest.skip("this platform's scipy wheel bundles no libraries")
    found = notices.attribute([], [str(libs[0])])
    assert [notices.normal(d.name) for d in found["dists"]] == ["scipy"]
    assert found["dists"][0].files == 1


def test_what_no_distribution_accounts_for_is_said_and_stops_the_package(notices, tmp_path):
    stray = tmp_path / "vendored_thing.py"
    stray.write_text("x = 1\n")
    found = notices.attribute([("vendored_thing", str(stray))], [str(tmp_path / "lib.dylib")])
    data = notices.manifest(found)
    assert "vendored_thing" in data["unattributed"]
    assert any("no installed distribution accounts for" in p for p in notices.problems(data))


def test_dev_tooling_in_a_manifest_stops_the_package(notices):
    found = _found(notices, ["numpy", "pytest", "hypothesis"])
    data = notices.manifest(found)
    assert data["dev_only"] == ["hypothesis", "pytest"]
    assert any("developer / test tooling" in p for p in notices.problems(data))
    clean = notices.manifest(_found(notices, ["numpy"]))
    assert notices.problems(clean) == []


def test_the_spec_writes_the_notices_from_its_own_analysis():
    """Not from a list kept by hand: the spec hands PyInstaller's Analysis to the generator and
    bundles the two files it writes."""
    text = SPEC.read_text()
    assert "bundle_notices" in text and "for_analysis(a," in text.replace(" (", "(")
    assert text.index("for_analysis") > text.index("a = Analysis("), "before the analysis exists"
    assert text.index("for_analysis") < text.index("COLLECT("), "after the files were collected"


def _bundle(tmp: Path, notices, modules: list[str] | None) -> Path:
    """A stand-in onedir brain: the folder, `_internal/`, and (unless None) the two files."""
    brain = tmp / "mixagent_daw_brain" / "mixagent_daw_brain"
    (brain.parent / "_internal").mkdir(parents=True)
    brain.write_text("#!/bin/sh\n")
    if modules is not None:
        notices.write(_found(notices, modules), brain.parent / "_internal")
    return brain


def test_the_packaging_gate_refuses_a_brain_with_no_notices(verifier, notices, tmp_path):
    brain = _bundle(tmp_path, notices, None)
    with pytest.raises(verifier.GateFailure, match="says nothing about the third-party Python"):
        verifier.check_contents(brain)


def test_the_packaging_gate_refuses_a_brain_that_carries_test_tooling(verifier, notices, tmp_path):
    brain = _bundle(tmp_path, notices, ["numpy", "pytest"])
    with pytest.raises(verifier.GateFailure, match="developer / test tooling: pytest"):
        verifier.check_contents(brain)


def test_the_packaging_gate_refuses_notices_that_do_not_match_the_manifest(verifier, notices,
                                                                          tmp_path):
    brain = _bundle(tmp_path, notices, ["numpy", "scipy"])
    text = brain.parent / "_internal" / notices.NOTICES_NAME
    text.write_text(text.read_text().replace("scipy", "s-c-i-p-y"))
    with pytest.raises(verifier.GateFailure, match="scipy"):
        verifier.check_contents(brain)


def test_the_packaging_gate_passes_a_brain_that_names_what_it_holds(verifier, notices, tmp_path):
    brain = _bundle(tmp_path, notices, ["numpy", "scipy", "anthropic"])
    assert verifier.check_contents(brain) == 3


def test_the_gate_runs_the_contents_check_before_it_passes_a_brain(verifier):
    import inspect
    body = inspect.getsource(verifier.verify)
    assert "check_contents(brain)" in body.replace(" (", "(")
    assert body.index("check_contents") < body.index("bundled brain OK")


def test_the_package_ships_the_python_notices_beside_the_others():
    """In the app they ride inside the brain's folder; the disk image carries a copy beside
    `Third-Party Notices.md`, where a person looks."""
    text = (REPO / "daw" / "scripts" / "package.sh").read_text()
    assert "THIRD_PARTY_PYTHON.txt" in text
    assert re.search(r'cp[^\n]*THIRD_PARTY_PYTHON\.txt[^\n]*\$STAGE/', text)


def test_the_notices_and_the_launch_list_name_the_python_contents():
    notes = (REPO / "daw" / "THIRD_PARTY_NOTICES.md").read_text()
    assert "THIRD_PARTY_PYTHON.txt" in notes and "bundle_notices.py" in notes
    launch = (REPO / "docs" / "14-open-beta-checklist.md").read_text()
    section = launch[launch.index("## 4. Licence obligations"):launch.index("## 5. ")]
    assert "THIRD_PARTY_PYTHON.txt" in section
