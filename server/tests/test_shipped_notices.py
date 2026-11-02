"""What a user is handed as the third-party notices (2026-11-02, the public launch: James found
our licensing checklist in the disk image - removed libraries and their prices, a to-do list,
build commands). The shipped notice is daw/THIRD_PARTY_NOTICES_PUBLIC.txt; the checklist,
daw/THIRD_PARTY_NOTICES.md, stays in the repo and is copied into no app and no DMG."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PACKAGE = (REPO / "daw" / "scripts" / "package.sh").read_text()
CMAKE = (REPO / "daw" / "CMakeLists.txt").read_text()
APP = (REPO / "daw" / "Source" / "App.cpp").read_text()
PUBLIC = " ".join((REPO / "daw" / "THIRD_PARTY_NOTICES_PUBLIC.txt").read_text().split())
LICENCES = " ".join((REPO / "daw" / "THIRD_PARTY_LICENSES_APP.txt").read_text().split())


def _copies(text: str, name: str) -> bool:
    return any(name in line and ("cp " in line or "ditto" in line)
               for line in text.splitlines() if not line.lstrip().startswith("#"))


def test_the_checklist_is_copied_into_no_app_and_no_dmg():
    assert not _copies(PACKAGE, "THIRD_PARTY_NOTICES.md")
    assert not _copies(CMAKE, "THIRD_PARTY_NOTICES.md")
    assert '"Third-Party Notices.md"' not in PACKAGE


def test_the_public_notice_is_what_ships_and_what_help_opens():
    assert 'cp -f "$DAW/THIRD_PARTY_NOTICES_PUBLIC.txt" "$APP/Contents/Resources/Third-Party Notices.txt"' in PACKAGE
    assert 'cp "$DAW/THIRD_PARTY_NOTICES_PUBLIC.txt" "$STAGE/Third-Party Notices.txt"' in PACKAGE
    assert '"Contents/Resources/Third-Party Notices.txt"' in APP
    assert "THIRD_PARTY_NOTICES_PUBLIC.txt" in CMAKE


def test_the_public_notice_holds_no_notes_of_ours():
    for word in ("DONE", "checklist", "do not re-add", "Breakfast", "£", "knowledge", "2026-",
                 "package.sh", "CMake", "James", "`"):
        assert word not in PUBLIC, word
    for need in ("Powered by Tracktion Engine", "github.com/stonecorestudio/zammoth-brain",
                 "Lesser General Public License 2.1", "support@zammoth.com"):
        assert need in PUBLIC, need


def test_the_licences_file_uses_the_licences_own_words():
    assert "Made with Tracktion Engine" not in LICENCES and "Powered by Tracktion Engine" in LICENCES
