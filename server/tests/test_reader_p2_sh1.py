"""Reader P2s, batch Sh1 (2026-10-06): packaging, CI, the engine clone.

S7.R7  publish_update.sh published an ad-hoc, un-notarized build to a production channel
S7.R8  the bundle declared no minimum macOS, and the brain inside cannot start before 14.0
S7.R9  the engine was an untracked clone at an unrecorded commit
S7.R10 daw.yml could not build on a clean runner, and its loop stopped at the first failing leg

Everything here is a script or a workflow file, so every test RUNS the script (in a scratch tree,
with stand-ins for the tools it calls) or reads the file it pins. Nothing here needs a Developer
ID, a network or a DAW build: what only those can prove is listed in P2-fixes.md as a limit.
"""
from __future__ import annotations

import os
import plistlib
import re
import shutil
import stat
import struct
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "daw" / "scripts"
PATCHES = REPO / "daw" / "patches"
ENGINE = REPO / "daw" / "third_party" / "tracktion_engine"


def _exe(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip())
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


# ---- S7.R7: the publish refuses what no client could install ----------------------------------
DEV_ID = "Developer ID Application: Test Person (TEAM123456)"


def _publish_tree(tmp: Path, identities: str = "") -> dict:
    """A scratch copy of the publish script beside a stand-in package.sh that only leaves a mark,
    and a stand-in `security` that lists `identities`. Returns the env to run it with."""
    scripts = tmp / "daw" / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy(SCRIPTS / "publish_update.sh", scripts / "publish_update.sh")
    (tmp / "daw" / "CMakeLists.txt").write_text("project(MixAgentDAW VERSION 0.1.0 LANGUAGES CXX)\n")
    _exe(scripts / "package.sh", f"""
        #!/bin/bash
        echo "sign=${{SIGN_ID:-unset}} notary=${{NOTARY_PROFILE:-unset}} dmg=${{SKIP_DMG:-unset}}" > "{tmp}/PACKAGED"
        exit 0
    """)
    listing = identities + f"     {len(identities.splitlines())} valid identities found\n"
    (tmp / "identities.txt").write_text(listing)
    _exe(tmp / "bin" / "security", f"""
        #!/bin/bash
        cat "{tmp}/identities.txt"
    """)
    env = {k: v for k, v in os.environ.items()
           if k not in ("SIGN_ID", "NOTARY_PROFILE", "BASE_URL", "CHANNEL_DIR", "SKIP_DMG")}
    env.update(HOME=str(tmp), PATH=f"{tmp / 'bin'}:{env['PATH']}", ZAMMOTH_BUILD="4242",
               CHANNEL_DIR=str(tmp / "channel"))
    return env


def _publish(tmp: Path, env: dict, **extra) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(tmp / "daw/scripts/publish_update.sh"), "notes"],
                          env={**env, **extra}, capture_output=True, text=True, timeout=60)


def _refused(tmp: Path, r: subprocess.CompletedProcess, *words: str) -> None:
    assert r.returncode != 0, r.stdout + r.stderr
    assert not (tmp / "PACKAGED").exists(), "the refusal came after packaging had started"
    assert not (tmp / "channel" / "appcast.json").exists()
    assert "refusing to publish" in r.stderr, r.stderr
    for w in words:
        assert w in r.stderr, (w, r.stderr)


@pytest.mark.parametrize("sign", [None, "-", ""])
def test_a_production_publish_without_an_identity_is_refused_before_packaging(tmp_path, sign):
    env = _publish_tree(tmp_path)
    extra = {"BASE_URL": "https://updates.example.invalid"}
    if sign is not None:
        extra["SIGN_ID"] = sign
    _refused(tmp_path, _publish(tmp_path, env, **extra), "SIGN_ID is not set")


def test_a_production_publish_without_a_notary_profile_is_refused(tmp_path):
    env = _publish_tree(tmp_path, f'  1) 0123456789ABCDEF0123456789ABCDEF01234567 "{DEV_ID}"\n')
    r = _publish(tmp_path, env, BASE_URL="https://updates.example.invalid", SIGN_ID=DEV_ID)
    _refused(tmp_path, r, "NOTARY_PROFILE is not set")


def test_a_production_publish_with_an_identity_the_keychain_does_not_hold_is_refused(tmp_path):
    env = _publish_tree(tmp_path)                       # the keychain lists nothing
    r = _publish(tmp_path, env, BASE_URL="https://updates.example.invalid", SIGN_ID=DEV_ID,
                 NOTARY_PROFILE="zammoth")
    _refused(tmp_path, r, "names no signing identity in this keychain")


def test_a_development_certificate_is_not_a_distribution_identity(tmp_path):
    dev = "Apple Development: Test Person (ABCDE12345)"
    env = _publish_tree(tmp_path, f'  1) 0123456789ABCDEF0123456789ABCDEF01234567 "{dev}"\n')
    r = _publish(tmp_path, env, BASE_URL="https://updates.example.invalid", SIGN_ID=dev,
                 NOTARY_PROFILE="zammoth")
    _refused(tmp_path, r, "is not a Developer ID Application identity")


def test_plain_http_is_a_production_channel_too(tmp_path):
    env = _publish_tree(tmp_path)
    _refused(tmp_path, _publish(tmp_path, env, BASE_URL="http://updates.example.invalid"),
             "SIGN_ID is not set")


def test_a_signed_production_publish_reaches_packaging_with_the_identity(tmp_path):
    """The control: the refusals are about the identity, not about https."""
    env = _publish_tree(tmp_path, f'  1) 0123456789ABCDEF0123456789ABCDEF01234567 "{DEV_ID}"\n')
    _publish(tmp_path, env, BASE_URL="https://updates.example.invalid", SIGN_ID=DEV_ID,
             NOTARY_PROFILE="zammoth")
    mark = tmp_path / "PACKAGED"
    assert mark.exists(), "a publish with an identity and a profile never reached package.sh"
    assert f"sign={DEV_ID} notary=zammoth" in mark.read_text()


def test_the_local_channel_still_publishes_ad_hoc(tmp_path):
    """The control the dev loop lives on: a file:// channel needs no identity."""
    env = _publish_tree(tmp_path)
    r = _publish(tmp_path, env)
    assert (tmp_path / "PACKAGED").exists(), r.stdout + r.stderr
    assert "refusing to publish" not in r.stderr


# ---- S7.R7: notarization is judged on what Apple answered -------------------------------------
def _notary_env(tmp: Path, verdict: str, staple: str = "ok", tool_rc: int = 0) -> dict:
    _exe(tmp / "bin" / "xcrun", f"""
        #!/bin/bash
        echo "$*" >> "{tmp}/xcrun.calls"
        case "$1" in
          notarytool)
            echo "Conducting pre-submission checks for $3 and initiating connection to the Apple notary service..."
            echo "  id: 2efe2717-52ef-43a5-96dc-0797e4ca1041"
            echo "Processing complete"
            echo "  id: 2efe2717-52ef-43a5-96dc-0797e4ca1041"
            echo "  status: {verdict}"
            exit {tool_rc} ;;
          stapler)
            [ "{staple}" = "ok" ] || {{ echo "The staple and validate action failed! Error 65."; exit 65; }}
            echo "The $2 action worked!" ;;
        esac
    """)
    return {**os.environ, "PATH": f"{tmp / 'bin'}:{os.environ['PATH']}"}


def _notarize(tmp: Path, env: dict, thing: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", 'set -euo pipefail; . "$0"; notarize_and_staple zammoth "$1"; echo NOTARIZED-OK',
         str(SCRIPTS / "notarize.sh"), str(thing)],
        env=env, capture_output=True, text=True, timeout=60)


def _calls(tmp: Path) -> str:
    f = tmp / "xcrun.calls"
    return f.read_text() if f.exists() else ""


def test_an_accepted_submission_is_stapled_and_validated(tmp_path):
    dmg = tmp_path / "Zammoth.dmg"
    dmg.write_bytes(b"x")
    r = _notarize(tmp_path, _notary_env(tmp_path, "Accepted"), dmg)
    assert r.returncode == 0 and "NOTARIZED-OK" in r.stdout, r.stdout + r.stderr
    calls = _calls(tmp_path)
    assert f"notarytool submit {dmg} --keychain-profile zammoth --wait" in calls
    assert f"stapler staple {dmg}" in calls and f"stapler validate {dmg}" in calls


def test_a_submission_apple_rejects_is_not_stapled_even_when_the_tool_exits_zero(tmp_path):
    """`notarytool submit --wait` exits 0 for a submission it processed and REJECTED."""
    dmg = tmp_path / "Zammoth.dmg"
    dmg.write_bytes(b"x")
    r = _notarize(tmp_path, _notary_env(tmp_path, "Invalid"), dmg)
    assert r.returncode != 0 and "NOTARIZED-OK" not in r.stdout, r.stdout + r.stderr
    assert "stapler" not in _calls(tmp_path)
    assert "notarytool log 2efe2717-52ef-43a5-96dc-0797e4ca1041" in r.stderr, r.stderr


def test_a_notary_tool_failure_stops_the_package(tmp_path):
    dmg = tmp_path / "Zammoth.dmg"
    dmg.write_bytes(b"x")
    r = _notarize(tmp_path, _notary_env(tmp_path, "Accepted", tool_rc=1), dmg)
    assert r.returncode != 0 and "stapler" not in _calls(tmp_path)


def test_a_ticket_that_will_not_staple_stops_the_package(tmp_path):
    dmg = tmp_path / "Zammoth.dmg"
    dmg.write_bytes(b"x")
    r = _notarize(tmp_path, _notary_env(tmp_path, "Accepted", staple="fail"), dmg)
    assert r.returncode != 0 and "NOTARIZED-OK" not in r.stdout


@pytest.mark.skipif(sys.platform != "darwin", reason="an .app is submitted as a ditto zip")
def test_an_app_is_submitted_as_a_zip_and_the_app_itself_is_stapled(tmp_path):
    app = tmp_path / "Zammoth.app"
    (app / "Contents").mkdir(parents=True)
    (app / "Contents" / "Info.plist").write_text("x")
    r = _notarize(tmp_path, _notary_env(tmp_path, "Accepted"), app)
    assert r.returncode == 0, r.stdout + r.stderr
    calls = _calls(tmp_path)
    submitted = re.search(r"notarytool submit (\S+)", calls).group(1)
    assert submitted.endswith("Zammoth.zip") and submitted != str(app)
    assert f"stapler staple {app}" in calls


def test_package_notarizes_the_app_before_the_zip_only_exit():
    """publish_update.sh packages with SKIP_DMG=1; the notarization used to sit after that exit."""
    sh = (SCRIPTS / "package.sh").read_text()
    verify = sh.index("codesign --verify --deep --strict")
    app_notary = sh.index('notarize_and_staple "$NOTARY_PROFILE" "$APP"')
    skip_exit = sh.index('if [ "${SKIP_DMG:-0}" = "1" ]; then')
    assert verify < app_notary < skip_exit
    assert 'notarize_and_staple "$NOTARY_PROFILE" "$DMG"' in sh[skip_exit:]
    assert "notarytool submit" not in sh, "package.sh judges a submission itself again - use notarize.sh"


def test_every_written_production_recipe_names_the_identity():
    """The four places that tell someone how to publish to production (the reader's list)."""
    for rel in ("docs/14-open-beta-checklist.md", "backend/README.md", "docs/10-daw.md",
                ".claude/skills/backend-ops/SKILL.md"):
        text = (REPO / rel).read_text()
        hits = [m.start() for m in re.finditer(r"BASE_URL=<?https", text)]
        assert hits, f"{rel} no longer shows the production publish - update this test with it"
        for at in hits:
            near = text[max(0, at - 400): at + 400]
            assert "SIGN_ID" in near and "NOTARY_PROFILE" in near, f"{rel}: a production publish with no identity"


# ---- S7.R8: the floor is in the bundle, and nothing in the bundle is above it -----------------
def _macho(minos: tuple[int, int, int], legacy: bool = False) -> bytes:
    enc = (minos[0] << 16) | (minos[1] << 8) | minos[2]
    if legacy:                                           # LC_VERSION_MIN_MACOSX
        cmd = struct.pack("<IIII", 0x24, 16, enc, enc)
    else:                                                # LC_BUILD_VERSION, platform 1 = macOS
        cmd = struct.pack("<IIIIII", 0x32, 24, 1, enc, enc, 0)
    seg = struct.pack("<II", 0x19, 8)                    # some other command in front
    cmds = seg + cmd
    head = struct.pack("<IiiIIIII", 0xFEEDFACF, 0x0100000C, 0, 2, 2, len(cmds), 0, 0)
    return head + cmds


def _fat(*slices: bytes) -> bytes:
    out = struct.pack(">II", 0xCAFEBABE, len(slices))
    at = 8 + 20 * len(slices)
    body = b""
    for s in slices:
        out += struct.pack(">iiIII", 0x0100000C, 0, at + len(body), len(s), 0)
        body += s
    return out + body


def _bundle(tmp: Path, floor: str | None, files: dict[str, bytes]) -> Path:
    app = tmp / "Zammoth.app"
    (app / "Contents" / "MacOS").mkdir(parents=True)
    info = {"CFBundleName": "Zammoth"}
    if floor is not None:
        info["LSMinimumSystemVersion"] = floor
    (app / "Contents" / "Info.plist").write_bytes(plistlib.dumps(info))
    for rel, data in files.items():
        p = app / "Contents" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return app


def _floor(app: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPTS / "check_macos_floor.py"), str(app)],
                          capture_output=True, text=True, timeout=60)


def test_a_bundle_whose_binaries_sit_at_or_under_its_floor_passes(tmp_path):
    app = _bundle(tmp_path, "14.0", {
        "MacOS/Zammoth": _macho((11, 0, 0)),
        "Resources/brain/_internal/numpy/_core/_multiarray_umath.so": _macho((14, 0, 0)),
        "Resources/brain/_internal/old.dylib": _macho((10, 13, 0), legacy=True),
        "Resources/notes.md": b"not a binary",
        "Resources/short": b"\xcf",
    })
    r = _floor(app)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "3 Mach-O" in r.stdout and "14.0" in r.stdout


def test_a_binary_above_the_floor_is_named(tmp_path):
    app = _bundle(tmp_path, "14.0", {
        "MacOS/Zammoth": _macho((11, 0, 0)),
        "Resources/brain/_internal/onnxruntime/capi/libonnxruntime.dylib": _macho((15, 2, 0)),
    })
    r = _floor(app)
    assert r.returncode != 0
    assert "libonnxruntime.dylib" in r.stderr and "15.2" in r.stderr and "Zammoth\n" not in r.stderr


def test_a_slice_inside_a_universal_binary_counts(tmp_path):
    app = _bundle(tmp_path, "14.0", {"MacOS/Zammoth": _fat(_macho((11, 0, 0)), _macho((15, 0, 0)))})
    r = _floor(app)
    assert r.returncode != 0 and "15.0" in r.stderr


def test_a_bundle_that_states_no_floor_fails(tmp_path):
    app = _bundle(tmp_path, None, {"MacOS/Zammoth": _macho((11, 0, 0))})
    r = _floor(app)
    assert r.returncode != 0 and "LSMinimumSystemVersion" in r.stderr


def test_the_app_target_writes_the_floor_into_its_plist():
    cmake = (REPO / "daw" / "CMakeLists.txt").read_text()
    app = cmake[cmake.index("juce_add_gui_app(MixAgentDAWApp"):]
    app = app[: app.index("target_sources(MixAgentDAWApp")]
    m = re.search(r"<key>LSMinimumSystemVersion</key><string>([0-9.]+)</string>", app)
    assert m, "the app's Info.plist states no minimum macOS"
    assert m.group(1) == "14.0"


def test_package_checks_the_floor_before_anything_is_signed_or_zipped():
    sh = (SCRIPTS / "package.sh").read_text()
    assert sh.index("check_macos_floor.py") < sh.index('echo "==> codesign')


def test_the_wheels_that_set_the_floor_cannot_move_unnoticed():
    """numpy, scipy and onnxruntime carry the bundle's highest minimum (macOS 14.0 builds). An
    open requirement let any re-freeze pick a wheel built for a later system."""
    reqs = (REPO / "server" / "requirements.txt").read_text()
    for name in ("numpy", "scipy", "onnxruntime"):
        line = next(l for l in reqs.splitlines() if re.match(rf"{name}\b", l))
        spec = line.split("#")[0]
        assert re.search(r"<\s*\d", spec) or "==" in spec, f"{name} has no upper bound: {line!r}"


def test_the_floor_is_said_where_a_user_reads_it():
    assert "macOS 14" in (REPO / "docs" / "17-user-manual.md").read_text()
    assert "macOS 14" in (REPO / "docs" / "14-open-beta-checklist.md").read_text()


# ---- S7.R9: the engine can be made again from what the repository holds -----------------------
def _engine_base() -> dict:
    out = {}
    for line in (PATCHES / "ENGINE_BASE").read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def test_the_engine_base_is_recorded_as_whole_commit_hashes():
    base = _engine_base()
    assert base["TRACKTION_URL"].startswith("https://")
    for key in ("TRACKTION_COMMIT", "JUCE_COMMIT"):
        assert re.fullmatch(r"[0-9a-f]{40}", base[key]), (key, base[key])


def test_the_patches_are_one_unbroken_series():
    """`git am` applies the glob in name order; a gap or a doubled number is a patch that was
    renumbered by hand, and the series no longer says what order the clone was built in."""
    nums = sorted(int(p.name[:4]) for p in PATCHES.glob("[0-9][0-9][0-9][0-9]-*.patch"))
    assert nums and nums == list(range(1, len(nums) + 1)), nums
    readme = (PATCHES / "README.md").read_text()
    assert "fetch_engine.sh" in readme and "ENGINE_BASE" in readme


def test_the_written_setup_is_the_script():
    readme = (REPO / "daw" / "README.md").read_text()
    assert "daw/scripts/fetch_engine.sh" in readme
    assert "git clone --depth 1 https://github.com/Tracktion/tracktion_engine.git" not in readme, \
        "daw/README.md still clones whatever upstream's head is today"


def _check(engine: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPTS / "fetch_engine.sh"), "--check"],
                          env={**os.environ, "ENGINE_DIR": str(engine)},
                          capture_output=True, text=True, timeout=300)


needs_clone = pytest.mark.skipif(not (ENGINE / ".git").exists(),
                                 reason="no engine clone on this machine (CI's Linux job has none)")


@needs_clone
def test_the_clone_on_this_machine_is_the_recorded_engine():
    r = _check(ENGINE)
    assert r.returncode == 0, r.stdout + r.stderr
    n = len(list(PATCHES.glob("[0-9][0-9][0-9][0-9]-*.patch")))
    assert f"{n} of {n} patches" in r.stdout, r.stdout


@pytest.fixture(scope="module")
def engine_copy(tmp_path_factory):
    """A second clone of the engine as it stands here (no JUCE in it), for the checks' controls."""
    if not (ENGINE / ".git").exists():
        pytest.skip("no engine clone on this machine")
    dst = tmp_path_factory.mktemp("engine") / "tracktion_engine"
    subprocess.run(["git", "clone", "-q", "--no-hardlinks", f"file://{ENGINE}", str(dst)],
                   check=True, capture_output=True, timeout=300)
    return dst


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
                           *args], check=True, capture_output=True, text=True, timeout=120).stdout


def test_a_clone_missing_its_last_patch_fails_the_check(engine_copy):
    head = _git(engine_copy, "rev-parse", "HEAD").strip()
    try:
        _git(engine_copy, "reset", "-q", "--hard", "HEAD~1")
        r = _check(engine_copy)
        assert r.returncode != 0 and "base" in r.stderr, r.stdout + r.stderr
    finally:
        _git(engine_copy, "reset", "-q", "--hard", head)


def test_a_commit_that_differs_from_its_patch_file_fails_the_check(engine_copy):
    head = _git(engine_copy, "rev-parse", "HEAD").strip()
    try:
        f = engine_copy / "README.md"
        f.write_text(f.read_text() + "\nedited in the clone, never exported\n")
        _git(engine_copy, "commit", "-q", "-a", "--amend", "--no-edit")
        r = _check(engine_copy)
        last = sorted(PATCHES.glob("[0-9][0-9][0-9][0-9]-*.patch"))[-1].name
        assert r.returncode != 0 and last in r.stderr, r.stdout + r.stderr
    finally:
        _git(engine_copy, "reset", "-q", "--hard", head)


def test_an_edit_that_was_never_committed_fails_the_check(engine_copy):
    f = engine_copy / "README.md"
    was = f.read_text()
    try:
        f.write_text(was + "\nleft in the working tree\n")
        r = _check(engine_copy)
        assert r.returncode != 0 and "uncommitted" in r.stderr, r.stdout + r.stderr
    finally:
        f.write_text(was)


def test_a_clone_without_the_recorded_juce_fails_the_check(engine_copy):
    r = _check(engine_copy)                              # the copy has no submodule checked out
    assert r.returncode != 0 and "modules/juce) is not checked out" in r.stderr, r.stdout + r.stderr


def test_the_script_never_replaces_a_clone_that_is_not_the_recorded_engine(engine_copy):
    """A clone that fails the check may hold a patch nobody has exported yet."""
    head = _git(engine_copy, "rev-parse", "HEAD").strip()
    r = subprocess.run(["bash", str(SCRIPTS / "fetch_engine.sh")],
                       env={**os.environ, "ENGINE_DIR": str(engine_copy)},
                       capture_output=True, text=True, timeout=300)
    assert r.returncode != 0 and "may hold a patch that was never exported" in r.stderr, r.stdout + r.stderr
    assert _git(engine_copy, "rev-parse", "HEAD").strip() == head


# ---- S7.R10: the workflow can reach its own gate, and the gate names what failed --------------
YML = REPO / ".github" / "workflows" / "daw.yml"


def _steps() -> list[dict]:
    """daw.yml's steps: name, uses, if, the `with:` block's text and the `run:` script as the
    runner would hand it to bash. Read by indentation, so the test needs no YAML package."""
    text = YML.read_text()
    body = text[text.index("    steps:\n") + len("    steps:\n"):]
    out: list[dict] = []
    for chunk in re.split(r"(?m)^      - ", body)[1:]:
        lines = chunk.split("\n")
        step: dict = {"text": chunk}
        for key in ("name", "uses", "if"):
            m = re.search(rf"(?m)^(?:        )?{key}: (.*)$", chunk)
            if m and (key != "name" or not m.group(1).startswith("${{")):
                step[key] = m.group(1).strip()
        if "run: |" in chunk:
            at = next(i for i, l in enumerate(lines) if l.strip() == "run: |")
            block = []
            for l in lines[at + 1:]:
                if l.strip() and not l.startswith("          "):
                    break
                block.append(l[10:])
            step["run"] = "\n".join(block).rstrip() + "\n"
        else:
            m = re.search(r"(?m)^        run: (.+)$", chunk)
            if m:
                step["run"] = m.group(1).strip() + "\n"
        out.append(step)
    assert len(out) >= 8, "daw.yml's steps were not read - the workflow moved"
    return out


def _step(name: str) -> str:
    return next(s for s in _steps() if s.get("name") == name)["run"]


def test_the_workflow_fetches_the_engine_the_repository_records():
    steps = _steps()
    checkout = next(s for s in steps if s.get("uses", "").startswith("actions/checkout"))
    assert "submodules" not in checkout["text"], "this repository has no submodule"
    runs = [s.get("run", "") for s in steps]
    fetch = next(i for i, r in enumerate(runs) if "daw/scripts/fetch_engine.sh" in r)
    build = next(i for i, r in enumerate(runs) if "cmake -B" in r)
    assert fetch < build


def test_the_workflow_builds_the_venv_the_console_looks_for():
    """The console harness starts the brain with <repo>/server/.venv/bin/python; the job used to
    install into the runner's own interpreter, and only the engine's requirements."""
    assert "PY: ${{ github.workspace }}/server/.venv/bin/python" in YML.read_text()
    runs = "\n".join(s.get("run", "") for s in _steps())
    assert "python -m venv server/.venv" in runs
    install = next(l.strip() for l in runs.splitlines() if "pip install -r" in l)
    assert install.startswith('"$PY" -m pip install')
    assert "server/requirements.txt" in install and "orchestrator/requirements.txt" in install
    for line in runs.splitlines():
        s = line.strip()
        if s == "python -m venv server/.venv":
            continue
        assert not re.search(r"(^|; |&& |if |\$\()(python3?|pip3?) ", s), f"a step runs outside the venv: {s!r}"


def _run_console_step(tmp: Path) -> subprocess.CompletedProcess:
    """Run the workflow's own console step, under the shell GitHub runs it in, against a stand-in
    binary: --good passes, --skips skips, --bad exits 7, --liar prints PASS and then a FAIL,
    --dies prints PASS and exits non-zero, --hangs never returns."""
    (tmp / "daw" / "Source").mkdir(parents=True)
    (tmp / "daw" / "Source" / "Main.cpp").write_text(
        'a ("--good"); a ("--bad"); a ("--skips"); a ("--liar"); a ("--dies"); a ("--hangs");\n'
        'a ("--agent"); a ("--early");\n'
        'a ("--play"); a ("--brain-llm"); a ("--selfcrash"); a ("--export-oracle");\n')
    _exe(tmp / "daw/build.noindex/MixAgentDAW_artefacts/Release/MixAgentDAW", """
        #!/bin/bash
        case "$1" in
          --good)  echo "[MixAgentDAW] good: PASS" ;;
          --skips) echo "[MixAgentDAW] no audio device: SKIP" ;;
          --bad)   echo "leg-output-of-bad"; exit 7 ;;
          --liar)  echo "[MixAgentDAW] one: PASS"; echo "[MixAgentDAW] two: FAIL" ;;
          --dies)  echo "[MixAgentDAW] one: PASS"; exit 137 ;;
          --agent) [ -f "$2/daw_chains.json" ] && [ -f "$2/stem.wav" ] && [ "$2" != "test_projects/suno_demo" ] \\
                     && echo "[MixAgentDAW] agent: PASS" || { echo "no chains in $2"; exit 8; } ;;
          --early) echo "[MixAgentDAW]   XX  the-check-that-failed-early"; for i in $(seq 1 40); do echo "line $i"; done; exit 5 ;;
          --hangs) sleep 30 ;;
          *)       echo "this flag is excluded and must not run: $1"; exit 99 ;;
        esac
    """)
    (tmp / "test_projects" / "suno_demo").mkdir(parents=True)
    (tmp / "test_projects" / "suno_demo" / "stem.wav").write_bytes(b"RIFF")
    # "$PY -m orchestrator.daw_bridge <project> --brief ..." - the stand-in writes the file the
    # real one writes.
    py = _exe(tmp / "py", """
        #!/bin/bash
        [ "$2" = "orchestrator.daw_bridge" ] && echo "{}" > "$3/daw_chains.json"
        exit 0
    """)
    script = tmp / "step.sh"
    script.write_text(_step("Console self-tests"))
    # GitHub's default for a `run:` with no `shell:` is `bash -e {0}`.
    return subprocess.run(["bash", "-e", str(script)], cwd=tmp, capture_output=True, text=True,
                          timeout=120, env={**os.environ, "LEG_TIMEOUT_S": "3", "PY": str(py)})


@pytest.fixture(scope="module")
def console_step(tmp_path_factory):
    """ONE run of the step, for the four tests that read it: (the finished process, the tree
    it ran in). Each run waits out the hanging leg's alarm, and the four ran it four times."""
    tmp = tmp_path_factory.mktemp("console-step")
    return _run_console_step(tmp), tmp


def test_a_failing_leg_is_named_and_the_loop_goes_on(console_step):
    r, _tmp = console_step
    out = r.stdout
    assert "PASS  --good" in out and "SKIP  --skips" in out, out + r.stderr
    for f in ("--bad", "--liar", "--dies", "--early"):
        assert f"FAIL  {f}" in out, out + r.stderr
    assert "leg-output-of-bad" in out, "the failing leg's own output was not shown"
    assert "must not run" not in out
    assert "--- 5 failing ---" in out, out
    assert r.returncode != 0


def test_the_agent_leg_is_given_the_chains_the_brain_writes(console_step):
    """--agent reads a git-ignored file only a brain run makes; a clean runner has none."""
    r, tmp = console_step
    out = r.stdout
    assert "PASS  --agent" in out, out
    assert not (tmp / "test_projects/suno_demo/daw_chains.json").exists(), \
        "the chains were written into the project every other leg opens"


def test_a_failed_check_is_shown_wherever_in_the_output_it_is(console_step):
    out = console_step[0].stdout
    assert "the-check-that-failed-early" in out, out


def test_every_generated_fixture_exists_before_the_tree_is_listed():
    run = _step("Generate the demo stems")
    listing = run.index("tree_before.txt")
    for made in ("suno_demo/generate_stems.py", "two_stem_demo three_section_demo technique_demo",
                 "orchestrator.wild_fixture"):
        assert made in run and run.index(made) < listing, made
    for d in ("two_stem_demo", "three_section_demo", "technique_demo"):
        assert (REPO / "test_projects" / d / "generate_stems.py").exists(), d


def test_a_leg_that_never_returns_is_stopped_and_named(console_step):
    out = console_step[0].stdout
    assert re.search(r"FAIL  --hangs.*timed out", out), out


def test_the_later_gates_run_even_when_a_leg_failed():
    """One dispatch must say everything: the pytest, the oracle, the E2E and the tree check used
    to be skipped whenever one console leg was red."""
    steps = _steps()
    names = [s.get("name") for s in steps]
    after = steps[names.index("Console self-tests") + 1:]
    assert len(after) >= 4
    for s in after:
        assert "cancelled()" in s.get("if", ""), f"{s.get('name')} is skipped after a red leg"


def test_the_built_app_states_its_floor_on_the_runner():
    runs = "\n".join(s.get("run", "") for s in _steps())
    assert "LSMinimumSystemVersion" in runs
