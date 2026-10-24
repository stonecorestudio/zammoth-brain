"""The licence texts of what is compiled into the app travel with it (docs/14 section 4.3).

`daw/scripts/app_notices.py` lists the third-party code JUCE and Tracktion Engine compile into the
Zammoth binary, and the font it embeds, and writes their licence texts to
daw/THIRD_PARTY_LICENSES_APP.txt. The build puts that file in Contents/Resources/Licenses/, the
package refuses a bundle without it, and the DMG carries a copy. `--check` fails when the file is
stale or the engine's 3rd_party folders hold a library nobody has accounted for.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "daw" / "scripts" / "app_notices.py"
ENGINE = REPO / "daw" / "third_party" / "tracktion_engine" / "modules"
BUNDLED = "Zammoth app third-party licences.txt"


@pytest.mark.skipif(not ENGINE.is_dir(), reason="no engine clone (daw/scripts/fetch_engine.sh)")
def test_the_app_licence_file_is_current_and_every_engine_library_is_accounted_for():
    r = subprocess.run([sys.executable, str(SCRIPT), "--check"], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stderr


def test_the_file_carries_the_statements_its_licences_require():
    text = (REPO / "daw" / "THIRD_PARTY_LICENSES_APP.txt").read_text(encoding="utf-8")
    # IJG: "the accompanying documentation must state that ..."
    assert "this software is based in part on the work of\nthe Independent JPEG Group" in text
    # the OFL travels with the font; the VST3 SDK and every MIT text keep their notice
    assert "SIL OPEN FONT LICENSE Version 1.1" in text
    assert "Steinberg Media Technologies" in text
    assert "LEGAL ISSUES\n============" in text           # the section, not the table of contents


def test_the_build_puts_it_in_the_bundle_beside_soundtouchs():
    cmake = (REPO / "daw" / "CMakeLists.txt").read_text()
    assert "THIRD_PARTY_LICENSES_APP.txt" in cmake and BUNDLED in cmake


def test_the_package_refuses_a_bundle_without_it_and_the_dmg_carries_it():
    pkg = (REPO / "daw" / "scripts" / "package.sh").read_text()
    assert f'Licenses/{BUNDLED}' in pkg
    assert "app_notices.py\" --check" in pkg
