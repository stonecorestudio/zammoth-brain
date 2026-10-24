"""The published brain source never carries the knowledge book (2026-10-20, James: "make sure
knowledge_content.py never goes into the published brain source").

`scripts/brain_source_cut.py` is the one way the GPL brain source is made (docs/14 section 4):
it leaves the book's module out AND reads every file that goes out for the book's own text.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import brain_source_cut as CUT  # noqa: E402

BOOK = (REPO / "orchestrator" / "knowledge_content.py").read_text(encoding="utf-8")


def test_the_cut_of_this_tree_is_written_without_the_book(tmp_path):
    out = tmp_path / "brain"
    assert CUT.cut(out, "HEAD") == []
    assert not (out / "orchestrator" / "knowledge_content.py").exists()
    for kept in ("orchestrator/knowledge.py", "orchestrator/knowledge_deep.py",
                 "orchestrator/license_state.py", "server/mix_engine/session.py",
                 "daw/sidecar/mixagent_daw_brain.spec"):
        assert (out / kept).is_file(), kept           # the loaders and the recipe ARE published
    assert not (out / "backend").exists() and not (out / ".git").exists()


def test_a_copied_paragraph_of_the_book_stops_the_cut(tmp_path, monkeypatch):
    """The module left out is not enough: its text pasted into a test or a prompt goes out
    too. Re-wrapped, it is still caught; and the cut writes nothing."""
    longest = max(CUT.fingerprints(BOOK), key=len)
    words = longest.split(" ")
    rewrapped = "\n    ".join(" ".join(words[i:i + 5]) for i in range(0, len(words), 5))
    real = CUT.collect
    monkeypatch.setattr(CUT, "collect", lambda ref: {
        **real(ref), "server/tests/test_pasted.py": f'NOTE = """{rewrapped}"""\n'.encode()})
    out = tmp_path / "brain"
    problems = CUT.cut(out, "HEAD")
    assert problems and "server/tests/test_pasted.py" in problems[0], problems
    assert not out.exists() or not any(out.iterdir()), "a refused cut wrote files"


def test_the_cut_refuses_a_folder_that_holds_something(tmp_path):
    (tmp_path / "x").write_text("old")
    assert CUT.cut(tmp_path, "HEAD") == [f"{tmp_path} is not an empty folder"]


def test_the_book_has_text_to_look_for_and_the_bundle_agrees():
    assert len(CUT.fingerprints(BOOK)) >= 100, "the check would be looking for almost nothing"
    assert "orchestrator/knowledge_content.py" in CUT.PRIVATE
    spec = (REPO / "daw" / "sidecar" / "mixagent_daw_brain.spec").read_text(encoding="utf-8")
    assert '"orchestrator.knowledge_content"' in spec     # the binary leaves out what the source does


def test_the_command_says_why_it_refused(tmp_path, capsys):
    (tmp_path / "x").write_text("old")
    assert CUT.main([str(tmp_path)]) == 1
    assert "NOT written" in capsys.readouterr().err


GPL3_SHA256 = "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986"   # gnu.org's gpl-3.0.txt


def test_the_cut_carries_its_licence_a_readme_and_what_it_was_cut_from(tmp_path):
    """A GPL program's published source carries the licence it is under, and says which
    shipped build it is the source of (docs/14 section 4: "Add the GPLv3 text (COPYING) - this
    repo has none")."""
    import hashlib
    import subprocess
    out = tmp_path / "brain"
    assert CUT.cut(out, "HEAD") == []
    assert hashlib.sha256((out / "COPYING").read_bytes()).hexdigest() == GPL3_SHA256
    readme = (out / "README.md").read_text(encoding="utf-8")
    for said in ("GNU General Public License", "daw/sidecar/build.sh", "knowledge_content.py",
                 "SOURCE_VERSION"):
        assert said in readme, said
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    build = subprocess.run(["git", "-C", str(REPO), "rev-list", "--count", "HEAD"],
                           capture_output=True, text=True, check=True).stdout.strip()
    version = (out / "SOURCE_VERSION").read_text(encoding="utf-8")
    assert f"commit {head}" in version and f"build {build}" in version
