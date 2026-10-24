"""Both PyInstaller bundles keep the knowledge book out of the binary.

`orchestrator/knowledge.py` states the contract: `knowledge_content.py` is
PRIVATE — excluded from the public brain-source cut and from the PyInstaller
sidecar bundle; the shipped brain fetches it from the control plane per
session. The DAW brain spec has excluded it since 2026-08-08; the VST sidecar
spec did not, so every shipped VST3 carried the whole book in its PYZ archive
until the 2026-09-01 sweep (P1). Two specs, one rule — this keeps them from
drifting again. Free: a text parse, no build.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SPECS = [REPO / "plugin" / "sidecar" / "mixagent_sidecar.spec",
         REPO / "daw" / "sidecar" / "mixagent_daw_brain.spec"]


def _excludes(spec: Path) -> list[str]:
    text = spec.read_text(encoding="utf-8")
    m = re.search(r"excludes\s*=\s*\[(.*?)\]", text, re.S)
    assert m, f"{spec.name}: no excludes=[...] list"
    return re.findall(r'"([^"]+)"', m.group(1))


@pytest.mark.parametrize("spec", SPECS, ids=lambda p: p.name)
def test_the_knowledge_book_is_excluded_from_the_bundle(spec):
    assert spec.is_file(), spec
    assert "orchestrator.knowledge_content" in _excludes(spec), (
        f"{spec.name} would bundle the private knowledge book — add "
        f'"orchestrator.knowledge_content" to its excludes')
