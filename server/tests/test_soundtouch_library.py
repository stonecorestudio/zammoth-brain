"""SoundTouch is a library the user can replace (2026-10-21, docs/14 section 4).

SoundTouch is LGPL 2.1 and the app is closed. It ships as Contents/Frameworks/
libzammoth_soundtouch.dylib, built by daw/soundtouch/zammoth_soundtouch.cpp from the engine's
own copy, with its licence and complete source in Contents/Resources/Licenses/SoundTouch;
the engine leaves the sources out of its unity build (engine patch 0056). These pin the pieces
that must agree, and read a built bundle when there is one.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DAW = REPO / "daw"
ENGINE = DAW / "third_party/tracktion_engine/modules/tracktion_engine"
UNITY = ENGINE / "tracktion_engine_timestretch.cpp"
LIBRARY_UNIT = DAW / "soundtouch/zammoth_soundtouch.cpp"
APP = DAW / "build.noindex/MixAgentDAWApp_artefacts/Release/Zammoth.app"


def _soundtouch_sources(text: str) -> list[str]:
    return re.findall(r'#include\s+"(?:3rd_party/soundtouch/source/SoundTouch/)?(\w+\.cpp)"', text)


def test_the_library_compiles_the_files_the_engine_leaves_out_in_the_same_order():
    """The library's names are the engine's only because it compiles the same files, in the
    same order, inside the same namespace: a SoundTouch file the engine gains must be in both."""
    if not UNITY.is_file():
        pytest.skip("no engine clone here (daw/scripts/fetch_engine.sh makes one)")
    unity = UNITY.read_text()
    shared = unity[unity.index("#if ! TRACKTION_SOUNDTOUCH_SHARED_LIBRARY"):
                   unity.index("#endif   // ! TRACKTION_SOUNDTOUCH_SHARED_LIBRARY")]
    library = LIBRARY_UNIT.read_text()
    assert _soundtouch_sources(shared) == _soundtouch_sources(library)
    assert len(_soundtouch_sources(library)) == 14
    opens = library.index("\nnamespace tracktion { inline namespace engine\n")   # the line, not the comment
    # every system header comes before the namespace opens
    head, body = library[:opens], library[opens:]
    assert "#include <" in head and "#include <" not in body


def test_the_build_links_the_library_and_the_package_checks_and_signs_it():
    cmake = (DAW / "CMakeLists.txt").read_text()
    assert "add_library(zammoth_soundtouch SHARED soundtouch/zammoth_soundtouch.cpp)" in cmake
    assert "TRACKTION_SOUNDTOUCH_SHARED_LIBRARY=1" in cmake
    libs = cmake[cmake.index("set(MIXAGENT_DAW_LIBS"):cmake.index("function(mixagent_daw_target")]
    assert "zammoth_soundtouch" in libs
    assert 'INSTALL_RPATH "@executable_path/../Frameworks"' in cmake

    package = (DAW / "scripts/package.sh").read_text()
    assert "check_soundtouch_library.py" in package
    keep = package[package.index('KEEP="'):package.index('"', package.index('KEEP="') + 6)]
    assert "Licenses" in keep, "the sweep of Resources would remove the licence and the source"
    sign_lib = package.index('-name "*.dylib"')
    assert sign_lib < package.index('codesign "${SIGN_APP[@]}" "$APP"'), "inside-out: the library first"
    assert package.index("check_soundtouch_library.py") < package.index('echo "==> codesign')


def test_the_engine_patch_is_recorded():
    patches = sorted(p.name for p in (DAW / "patches").glob("0056-*.patch"))
    assert patches, "engine patch 0056 (SoundTouch as a shared library) is not exported"
    text = (DAW / "patches" / patches[0]).read_text()
    assert "TRACKTION_SOUNDTOUCH_SHARED_LIBRARY" in text and "delaySamplesConvolved" in text


def test_what_travels_with_the_library():
    lgpl = (DAW / "soundtouch/COPYING.LGPL-2.1").read_text()
    assert "GNU LESSER GENERAL PUBLIC LICENSE" in lgpl and "Version 2.1, February 1999" in lgpl
    readme = (DAW / "soundtouch/README.txt").read_text()
    for said in ("libzammoth_soundtouch.dylib", "ZAMMOTH PATCH", "-install_name @rpath/libzammoth_soundtouch.dylib",
                 "codesign --force --deep --sign -", "tracktion::engine"):
        assert said in readme, said


@pytest.mark.skipif(sys.platform != "darwin", reason="reads a Mach-O bundle")
def test_a_built_app_holds_soundtouch_as_a_library():
    if not (APP / "Contents/MacOS").is_dir():
        pytest.skip("the app is not built here")
    out = subprocess.run([sys.executable, str(DAW / "scripts/check_soundtouch_library.py"), str(APP),
                          str(ENGINE / "3rd_party/soundtouch")], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
