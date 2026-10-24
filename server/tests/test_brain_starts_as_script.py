"""The brain starts the way the DAW starts it from a source checkout: as a SCRIPT, by path.

`juce::ChildProcess` is handed `<python> <repo>/orchestrator/daw_server.py <project>` - no `-m`,
so the module has no parent package and a relative import raises at the line it sits on. The
top of the file handles that (an absolute import of everything, with the repo root put on
sys.path); an import written later inside a function has to be absolute too.

Twice now it was not: `validate_chat_settings` (2026-09-22) and, in batch Py5 (2026-10-06),
`from .llm_backend import drop_rerouting_env` as the FIRST line `main()` runs - every source
brain died before its handshake. The frozen brain imports the module as a package and never
saw it, the pytest imports it as a package and never saw it, and a Python-only batch runs no
console leg. The first dispatch of daw.yml (batch Sh1) did: five brain legs red on a clean
runner. These two tests are that gate in the suite every push runs.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SERVER = REPO / "orchestrator" / "daw_server.py"


def test_no_relative_import_outside_the_package_branch():
    """Every relative import in the brain's entry module sits in the top-level `else:` of the
    `if __package__ in (None, "")` bootstrap - the one place that only runs inside a package."""
    tree = ast.parse(SERVER.read_text())
    bootstrap = next(n for n in tree.body if isinstance(n, ast.If)
                     and "__package__" in ast.unparse(n.test))
    allowed = {id(n) for stmt in bootstrap.orelse for n in ast.walk(stmt)}
    stray = [f"line {n.lineno}: {ast.unparse(n)[:70]}" for n in ast.walk(tree)
             if isinstance(n, ast.ImportFrom) and n.level > 0 and id(n) not in allowed]
    assert not stray, "a relative import the script entry cannot resolve: " + "; ".join(stray)


def test_the_brain_started_by_path_reaches_its_handshake(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "MIX_AGENT_"))}
    env["MIX_AGENT_DATA_DIR"] = str(tmp_path / "data")
    out = tmp_path / "brain.out"
    with open(out, "w") as f:
        p = subprocess.Popen([sys.executable, str(SERVER), str(tmp_path)], cwd=tmp_path,
                             stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, env=env)
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            text = out.read_text()
            if "MIXAGENT_LISTENING port=" in text or p.poll() is not None:
                break
            time.sleep(0.1)
        text = out.read_text()
        assert "MIXAGENT_LISTENING port=" in text, \
            f"the brain never handshook (exit {p.poll()}):\n{text[-1500:]}"
    finally:
        p.kill()
        p.wait(timeout=10)
