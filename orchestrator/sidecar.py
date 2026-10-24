"""Desktop sidecar entry point (Phase 4 — docs/07-packaging.md).

The Tauri shell spawns this process on app launch and kills it on quit. It is
the same gateway as Phase 3 with one packaging-specific behavior: by default it
binds an *ephemeral* port on 127.0.0.1 and reports the chosen port to the shell
with a single machine-readable line on stdout:

    MIXAGENT_LISTENING port=<port>

The socket is bound and listening before the line is printed, so the shell may
connect as soon as it reads it. Everything after the handshake is ordinary
gateway traffic (websocket + /api), identical to the Phase 3 local web app.

Run from source: `python -m orchestrator.sidecar [project_dir] [--port N]`
Frozen (PyInstaller): the binary's argv is the same; see plugin/sidecar/.
"""
from __future__ import annotations

import argparse
import os
import shutil
import socket
import sys
import threading
import time
from pathlib import Path

from .agents import RuleAgent
from .gateway import HOST, create_app

HANDSHAKE_PREFIX = "MIXAGENT_LISTENING port="
SUPERVISED_ENV = "MIXAGENT_SUPERVISED"


def _exit_on_stdin_eof() -> None:
    """Die with the shell. The Tauri shell holds our stdin pipe and sets
    MIXAGENT_SUPERVISED=1; EOF means the shell is gone — including crashes,
    where it never gets to send a kill (docs/07: no orphaned engines).
    Terminal and pytest runs don't set the env var and are unaffected."""
    def watch() -> None:
        try:
            while sys.stdin.buffer.read(4096):
                pass
        except Exception:
            pass
        os._exit(0)

    threading.Thread(target=watch, daemon=True).start()


def _exit_when_orphaned() -> None:
    """Die when our parent does — the orphan-proof companion to stdin-EOF.

    A plugin host (VST/AU) does not hold our stdin the way the Tauri shell does
    (juce::ChildProcess can't pass the child's stdin through), and it may tear the
    plugin down by killing the host/sandbox process outright, so our launcher's
    child->kill() never runs and we'd be orphaned (reparented to launchd) — a
    leaked port + ~200 MB per reload. getppid() only changes when the original
    parent dies, so watch it and self-exit. No-op for terminal/pytest runs (they
    don't set MIXAGENT_SUPERVISED)."""
    original = os.getppid()

    def watch() -> None:
        while os.getppid() == original:
            time.sleep(1.0)
        os._exit(0)

    threading.Thread(target=watch, daemon=True).start()


def _exit_when_heartbeat_stale() -> None:
    """Die when the launcher stops refreshing our heartbeat file.

    Covers the leak the parent-pid watch can't: a host destroys the plugin but
    stays alive, and the launcher's child->kill() never reaches us (seen in
    Ableton) — we're left parented to a live host, so getppid() never changes.
    SidecarManager rewrites MIXAGENT_HEARTBEAT every ~1.5s while it owns us and
    stops + deletes it on teardown; a stale (or deleted) file means we exit.

    The staleness window is generous (15 s, vs the ~1.5 s write cadence): under
    DAW load / App Nap the plugin's heartbeat thread can be starved for a few
    seconds without the plugin being gone, and a false self-exit there is exactly
    the "sidecar died, nothing works" failure. A genuinely destroyed plugin still
    deletes the file (immediate exit) or stops writing (gone within 15 s); the
    plugin-side supervisor respawns anything that dies anyway."""
    hb = os.environ.get("MIXAGENT_HEARTBEAT")
    if not hb:
        return

    def watch() -> None:
        while True:
            try:
                stale = time.time() - os.path.getmtime(hb) > 15.0
            except OSError:
                stale = True  # deleted on teardown
            if stale:
                os._exit(0)
            time.sleep(2.0)

    threading.Thread(target=watch, daemon=True).start()


def _clear_project(project_dir: str) -> None:
    """Blank a plugin project on a fresh spawn so a new VST insert starts empty
    (like any other plugin), not with the last session's stuff. The project folder
    is shared + persistent; drop ALL stem wavs, the session, and any renders/runs.

    Phase 40 (per-track instances): clears EVERY stem wav, not just captures — the
    project is populated by each insert's live capture on play, so a fresh spawn
    must start truly empty (otherwise leftover stems get mixed instead of, or
    alongside, the actual captured tracks). Proper per-instance state saved with
    the DAW project is the longer fix."""
    p = Path(project_dir)
    if not p.is_dir():
        return
    for f in (*p.glob("*.wav"), p / "session.json", p / ".reference_cache.json"):
        try:
            f.unlink()
        except OSError:
            pass
    for sub in ("bounces", "runs"):
        shutil.rmtree(p / sub, ignore_errors=True)


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Mix agent desktop sidecar")
    parser.add_argument("project_dir", nargs="?", default=None,
                        help="project folder to open at startup (optional)")
    parser.add_argument("--port", type=int, default=0,
                        help="port on 127.0.0.1 (default 0 = ephemeral)")
    parser.add_argument("--rule-agent", action="store_true",
                        help="use the deterministic no-LLM agent (smoke tests)")
    parser.add_argument("--fresh", action="store_true",
                        help="blank the project on start — a plugin insert is a "
                             "clean slate, not the last session")
    args = parser.parse_args(argv)

    # The desktop (Tauri) shell holds our stdin and sets MIXAGENT_SUPERVISED=1, so
    # stdin-EOF means it's gone. A plugin host does NOT give us a held stdin (we'd
    # read EOF immediately), so the plugin arms the parent-pid + heartbeat watches
    # (MIXAGENT_HEARTBEAT) instead and must NOT trip the stdin watch.
    supervised = os.environ.get(SUPERVISED_ENV) == "1"
    has_heartbeat = bool(os.environ.get("MIXAGENT_HEARTBEAT"))
    if supervised:
        _exit_on_stdin_eof()
    if supervised or has_heartbeat:
        _exit_when_orphaned()
    if has_heartbeat:
        _exit_when_heartbeat_stale()

    if args.fresh and args.project_dir:
        _clear_project(args.project_dir)

    factory = RuleAgent if args.rule_agent else None
    app = create_app(args.project_dir, agent_factory=factory)

    # Bind ourselves (never uvicorn) so the real port is known before serving
    # and the handshake can't race the shell's first connection attempt.
    sock = socket.create_server((HOST, args.port))
    port = sock.getsockname()[1]
    print(f"{HANDSHAKE_PREFIX}{port}", flush=True)

    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    server.run(sockets=[sock])


if __name__ == "__main__":
    main()
    sys.exit(0)
