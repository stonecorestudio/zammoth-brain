"""PyInstaller entry point for the DAW brain (docs/10-daw.md, Milestone B).

The native DAW spawns this as a child process — the NDJSON socket server
(orchestrator/daw_server.py) that runs the agent / analysis / proposals the
graph can't run on the audio thread. Frozen, it needs no system Python or venv,
so the shipped .app is self-contained. Argv is identical to
`python orchestrator/daw_server.py <project_dir>`.

Source-tree imports resolve through pathex in mixagent_daw_brain.spec; in the
frozen binary everything is collected.
"""
import sys

if __name__ == "__main__":
    # What this brain's keychain items are bound to (docs/14 section 6): a dummy item written,
    # read back and its access list printed, through the same Security.framework calls that hold
    # the user's key. Never part of a normal start.
    if len(sys.argv) > 1 and sys.argv[1] == "--keychain-check":
        from orchestrator.keychain_framework import self_check
        sys.exit(self_check(sys.argv[2:]))
    # The stem split's worker is this same binary (mix_engine/separation.py worker_argv): a
    # frozen brain has no interpreter to hand a module to. Read before the server is imported -
    # a worker needs none of it, and must print no handshake.
    if len(sys.argv) > 1 and sys.argv[1] == "--separation-worker":
        from mix_engine.separation_worker import main as worker_main
        sys.exit(worker_main(sys.argv[2:]))
    from orchestrator.daw_server import main
    main()
