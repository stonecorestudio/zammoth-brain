"""The audio engine as the brain reaches it: an in-memory MCP session on a project folder.

The DAW's brain opens one of these for the length of a chat turn, a mix run, an Auto Mix pass
or a stem's registration, and calls the engine's tools through it. It used to be defined in
the eval harness (`orchestrator/benchmark.py`), so the shipped brain imported an eval module
to do its ordinary work (Phase 55 audit). The benchmark and the evals import it from here.
"""
from __future__ import annotations

from .engine_models import LIMITS  # noqa: F401  (bootstraps server/ onto sys.path)

from mcp.shared.memory import create_connected_server_and_client_session
from mix_engine.mcp_server import create_server


def in_memory_engine(project_dir):
    """An async context manager: the engine's MCP server on `project_dir`, and a client
    session connected to it in this process (no subprocess, no socket)."""
    return create_connected_server_and_client_session(create_server(project_dir))
