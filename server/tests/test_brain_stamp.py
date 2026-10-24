"""The shipped brain must be the brain its sources build (2026-09-22 launch review, R1).

daw/scripts/package.sh reused daw/sidecar/dist/mixagent_daw_brain whenever the file existed, so
builds 1397 and the first 1402 shipped a brain frozen on 2026-08-15 inside a September app: its
handshake had no auth token, the app refused it, and the published engineer could not connect.
Three pieces close it, and each is driven here:

- daw/sidecar/stamp.sh — a stamp of everything the brain is frozen from (package.sh rebuilds on
  a mismatch instead of on a missing file);
- daw_server.brain_stamp() — the frozen brain reports the stamp build.sh bundled, in the hello;
- daw/sidecar/verify_bundled_brain.py — the packaging gate that RUNS the binary inside the app.
  Its negative controls are an August-shaped brain (no token) and a brain from other sources.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from orchestrator import daw_server
from orchestrator.daw_server import Hub, _serve_client, brain_stamp

REPO = Path(__file__).resolve().parents[2]
SIDECAR = REPO / "daw" / "sidecar"


# ------------------------------------------------------------------ stamp.sh

def _stamp_tree(tmp_path: Path) -> Path:
    """A miniature repo stamp.sh can run in: it derives the repo from its own location."""
    root = tmp_path / "repo"
    (root / "daw" / "sidecar").mkdir(parents=True)
    shutil.copy(SIDECAR / "stamp.sh", root / "daw" / "sidecar" / "stamp.sh")
    (root / "daw" / "sidecar" / "brain_entry.py").write_text("entry\n")
    (root / "daw" / "sidecar" / "mixagent_daw_brain.spec").write_text("spec\n")
    (root / "orchestrator").mkdir()
    (root / "orchestrator" / "chat.py").write_text("MODEL = 'a'\n")
    (root / "server" / "mix_engine" / "models").mkdir(parents=True)
    (root / "server" / "mix_engine" / "session.py").write_text("x = 1\n")
    (root / "server" / "mix_engine" / "models" / "sep.onnx").write_bytes(b"\x00\x01")
    return root


def _stamp(root: Path, python: str = sys.executable) -> str:
    out = subprocess.run(["bash", str(root / "daw" / "sidecar" / "stamp.sh")],
                         capture_output=True, text=True, check=True,
                         env={**os.environ, "PYTHON": python})
    return out.stdout.strip()


def test_stamp_moves_with_the_notices_generator_the_spec_runs(tmp_path):
    """The spec RUNS daw/sidecar/bundle_notices.py and bundles what it writes (2026-10-07): a
    change to it is a different bundle. The stamp named two files of the recipe by hand, so a
    third was outside it and package.sh would have reused the old brain."""
    root = _stamp_tree(tmp_path)
    before = _stamp(root)
    (root / "daw" / "sidecar" / "bundle_notices.py").write_text("NOTICES = 1\n")
    after = _stamp(root)
    assert after != before, "a new recipe module must change the stamp"
    (root / "daw" / "sidecar" / "bundle_notices.py").write_text("NOTICES = 2\n")
    assert _stamp(root) != after
    # ...and what is not the recipe does not: the build's own output folders sit beside it
    (root / "daw" / "sidecar" / "dist").mkdir()
    (root / "daw" / "sidecar" / "dist" / "left_over.py").write_text("x\n")
    assert _stamp(root) != before and _stamp(root) == _stamp(root)
    settled = _stamp(root)
    (root / "daw" / "sidecar" / "dist" / "left_over.py").write_text("y\n")
    assert _stamp(root) == settled, "a file under dist/ is an output, not a source"


def test_stamp_moves_with_every_input_the_freeze_reads(tmp_path):
    root = _stamp_tree(tmp_path)
    base = _stamp(root)
    assert len(base) == 16 and all(c in "0123456789abcdef" for c in base)
    assert _stamp(root) == base, "the stamp must be deterministic"

    # A brain source edit, a new module, a bundled model, the spec: each is a different brain.
    for path, text in [("orchestrator/chat.py", "MODEL = 'b'\n"),
                       ("orchestrator/new_module.py", "y = 2\n"),
                       ("server/mix_engine/models/sep.onnx", "\x00\x02"),
                       ("daw/sidecar/mixagent_daw_brain.spec", "spec 2\n")]:
        before = _stamp(root)
        (root / path).write_text(text)
        assert _stamp(root) != before, f"editing {path} must change the stamp"

    # Bytecode caches are not sources: they change on every import and must not force rebuilds.
    before = _stamp(root)
    (root / "orchestrator" / "__pycache__").mkdir()
    (root / "orchestrator" / "__pycache__" / "chat.cpython-314.pyc").write_bytes(b"cache")
    assert _stamp(root) == before


def test_stamp_moves_with_the_interpreters_packages(tmp_path):
    """A dependency upgrade changes the binary with no repo file moving — the freeze of
    `pip freeze` is in the stamp. Proven with a fake interpreter whose freeze differs."""
    root = _stamp_tree(tmp_path)
    fakes = []
    for version in ("1.0", "2.0"):
        fake = tmp_path / f"python-{version}"
        fake.write_text(f"#!/bin/sh\necho anthropic=={version}\n")
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
        fakes.append(str(fake))
    assert _stamp(root, fakes[0]) != _stamp(root, fakes[1])


def test_stamp_fails_loudly_when_the_freeze_cannot_be_read(tmp_path):
    """A broken interpreter must not yield a stamp computed from sources alone."""
    root = _stamp_tree(tmp_path)
    run = subprocess.run(["bash", str(root / "daw" / "sidecar" / "stamp.sh")],
                         capture_output=True, text=True,
                         env={**os.environ, "PYTHON": str(tmp_path / "no-such-python")})
    assert run.returncode != 0


# ------------------------------------------------------------------ brain_stamp() + hello

def test_brain_stamp_reports_the_bundled_file(tmp_path, monkeypatch):
    assert brain_stamp() == "dev"            # running from source: nothing to compare
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert brain_stamp() == "unstamped"      # frozen without build.sh
    (tmp_path / daw_server.BRAIN_STAMP_FILE).write_text("0123456789abcdef\n")
    assert brain_stamp() == "0123456789abcdef"
    (tmp_path / daw_server.BRAIN_STAMP_FILE).write_text("\n")
    assert brain_stamp() == "unstamped"


def test_hello_carries_the_brain_stamp(tmp_path, monkeypatch):
    # env key: the hello's key + licence reads stay off the keychain and the network
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-placeholder-not-a-key")
    monkeypatch.delenv("MIX_AGENT_LLM_BACKEND", raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    (tmp_path / daw_server.BRAIN_STAMP_FILE).write_text("feedfacecafebeef\n")
    proj = tmp_path / "proj"
    proj.mkdir()

    async def scenario():
        hub = Hub(proj)
        server = await asyncio.start_server(lambda r, w: _serve_client(r, w, hub), "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            try:
                writer.write((json.dumps({"id": 1, "method": "auth",
                                          "params": {"token": hub.token}}) + "\n"
                              + json.dumps({"id": 2, "method": "hello", "params": {}}) + "\n")
                             .encode())
                await writer.drain()
                replies = {}
                while 2 not in replies:
                    line = await asyncio.wait_for(reader.readline(), 20)
                    msg = json.loads(line)
                    if "id" in msg:
                        replies[msg["id"]] = msg
                return replies[2]
            finally:
                writer.close()
        finally:
            server.close()
            await server.wait_closed()

    hello = asyncio.run(scenario())
    assert hello["result"]["ok"] is True
    assert hello["result"]["brain_stamp"] == "feedfacecafebeef"


# ------------------------------------------------------------------ the packaging gate

def _executable(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def _real_brain(tmp_path: Path, stamp: str, delay_s: float = 0.0, notices: bool = True) -> Path:
    """The REAL daw_server behind an executable, frozen-shaped: `sys._MEIPASS` points at a dir
    holding the stamp, exactly where the PyInstaller archive root would be. The gate runs it in
    a cleared environment, so the wrapper carries its own paths. `delay_s` stalls every start
    before the server comes up — the shape of the onefile's per-launch extraction."""
    meipass = tmp_path / "meipass"
    meipass.mkdir()
    (meipass / "brain_stamp.txt").write_text(stamp + "\n")
    if notices:
        # ...and what a freeze writes beside it since 2026-10-07: the bundle's own list of the
        # third-party Python it holds (bundle_notices.py), which the gate reads back.
        import importlib.util
        spec = importlib.util.spec_from_file_location("bundle_notices_for_stamp",
                                                      SIDECAR / "bundle_notices.py")
        bundle_notices = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = bundle_notices
        spec.loader.exec_module(bundle_notices)
        found = bundle_notices.attribute(
            [("numpy", importlib.util.find_spec("numpy").origin)], [])
        bundle_notices.write(found, tmp_path / "_internal")
    return _executable(tmp_path / "brain", textwrap.dedent(f"""\
        #!{sys.executable}
        import sys, time
        time.sleep({delay_s})
        sys.path[:0] = [{str(REPO)!r}, {str(REPO / 'server')!r}]
        sys._MEIPASS = {str(meipass)!r}
        from orchestrator.daw_server import main
        main()
        """))


def _gate(brain: Path, stamp: str, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SIDECAR / "verify_bundled_brain.py"), str(brain),
                           "--stamp", stamp, "--timeout", "90", *extra],
                          capture_output=True, text=True, timeout=180)


def test_gate_passes_the_current_brain_built_from_these_sources(tmp_path):
    run = _gate(_real_brain(tmp_path, "0123456789abcdef"), "0123456789abcdef")
    assert run.returncode == 0, run.stderr
    assert "bundled brain OK: stamp 0123456789abcdef" in run.stdout


def test_gate_takes_a_relative_path_to_the_brain(tmp_path):
    """The gate starts the brain from a scratch folder; handed `daw/sidecar/dist/...` it said
    "No such file or directory" about a brain that was there."""
    brain = _real_brain(tmp_path, "0123456789abcdef")
    run = subprocess.run([sys.executable, str(SIDECAR / "verify_bundled_brain.py"), brain.name,
                          "--stamp", "0123456789abcdef", "--timeout", "90"],
                         capture_output=True, text=True, timeout=180, cwd=tmp_path)
    assert run.returncode == 0, run.stderr


def test_gate_refuses_a_brain_that_does_not_say_what_it_is_made_of(tmp_path):
    """A brain that starts, authenticates and reports the right stamp, frozen by a recipe that
    wrote no list of its third-party contents (every build before 2026-10-07)."""
    run = _gate(_real_brain(tmp_path, "0123456789abcdef", notices=False), "0123456789abcdef")
    assert run.returncode != 0
    assert "says nothing about the third-party Python inside it" in run.stderr


def test_gate_refuses_a_brain_built_from_other_sources(tmp_path):
    run = _gate(_real_brain(tmp_path, "0123456789abcdef"), "fedcba9876543210")
    assert run.returncode != 0
    assert "built from other sources" in run.stderr


def test_gate_refuses_the_august_brain_with_no_handshake_token(tmp_path):
    """The 1397 shape: a brain that listens but prints `MIXAGENT_LISTENING port=N` without the
    per-spawn token the app now requires."""
    brain = _executable(tmp_path / "brain", textwrap.dedent(f"""\
        #!{sys.executable}
        import socket, time
        s = socket.socket(); s.bind(("127.0.0.1", 0)); s.listen()
        print(f"MIXAGENT_LISTENING port={{s.getsockname()[1]}}", flush=True)
        time.sleep(60)
        """))
    run = _gate(brain, "0123456789abcdef")
    assert run.returncode != 0
    assert "no auth token" in run.stderr


def test_gate_refuses_a_brain_that_dies_before_its_handshake(tmp_path):
    brain = _executable(tmp_path / "brain", f"#!{sys.executable}\nraise SystemExit(3)\n")
    run = _gate(brain, "0123456789abcdef")
    assert run.returncode != 0
    assert "exited (code 3) before its handshake" in run.stderr


def test_gate_refuses_a_brain_that_is_slow_on_every_launch(tmp_path):
    """The onefile shape (launch review R2): it re-extracted ~240 MB on EVERY launch, so the
    second start was as slow as the first. The gate's warm leg must see that."""
    run = _gate(_real_brain(tmp_path, "0123456789abcdef", delay_s=2.0), "0123456789abcdef",
                "--warm-max", "1")
    assert run.returncode != 0
    assert "second start took" in run.stderr and "onefile" in run.stderr


def test_gate_refuses_a_first_start_beyond_the_apps_wait(tmp_path):
    run = _gate(_real_brain(tmp_path, "0123456789abcdef", delay_s=2.0), "0123456789abcdef",
                "--cold-max", "1")
    assert run.returncode != 0
    assert "first start took" in run.stderr


def test_gate_limit_is_the_apps_handshake_wait():
    """The gate's cold-start ceiling IS the app's wait (BrainClient.h kHandshakeTimeoutMs): if one
    moves alone, the gate passes a build the app would give up on, or fails one it would start."""
    import importlib.util
    import re
    spec = importlib.util.spec_from_file_location("verify_bundled_brain",
                                                  SIDECAR / "verify_bundled_brain.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    header = (REPO / "daw" / "Source" / "BrainClient.h").read_text()
    m = re.search(r"kHandshakeTimeoutMs\s*=\s*(\d+)", header)
    assert m, "BrainClient.h no longer declares kHandshakeTimeoutMs"
    assert int(m.group(1)) == int(gate.APP_HANDSHAKE_WAIT_S * 1000)


def test_the_brain_spec_builds_onedir():
    """ONEDIR, never onefile (launch review R2): the onefile brain decompressed ~240 MB into
    $TMPDIR on every launch — 20-34 s to the handshake — and leaked it on a force-kill."""
    text = (SIDECAR / "mixagent_daw_brain.spec").read_text()
    code = "\n".join(l.split("#")[0] for l in text.splitlines())   # comments don't count
    assert "exclude_binaries=True" in code, "the EXE must leave binaries to COLLECT (onedir)"
    assert "COLLECT(" in code, "the spec must COLLECT a onedir folder"
