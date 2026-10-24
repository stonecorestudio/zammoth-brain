"""The libraries the Python runtime's own build brings into the frozen brain carry licence text.

`daw/sidecar/bundle_notices.py` writes THIRD_PARTY_PYTHON.txt. OpenSSL's text (libssl, libcrypto)
is in Python's own license.rst; zstd's is not (docs/14 section 4.3, rechecked 2026-10-21), so the
notices carry it themselves. A runtime library that neither covers is a reason the packaging gate
refuses the brain (`problems`), so a new Python that brings one cannot ship without its text.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _notices():
    spec = importlib.util.spec_from_file_location(
        "bundle_notices_runtime_test", REPO / "daw" / "sidecar" / "bundle_notices.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _found(runtime_files):
    return {"dists": [], "runtime_modules": 0, "runtime_files": runtime_files,
            "own_modules": 0, "unattributed": []}


def test_a_brain_with_libzstd_carries_zstds_bsd_text():
    bn = _notices()
    text = bn.render(_found(["libcrypto.3.dylib", "libssl.3.dylib", "libzstd.1.dylib"]))
    assert "For Zstandard software" in text
    assert "Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved." in text
    assert "Neither the name Facebook, nor Meta, nor the names of its contributors" in text


def test_a_brain_without_libzstd_does_not_claim_it():
    assert "For Zstandard software" not in _notices().render(_found(["libssl.3.dylib"]))


def test_every_runtime_library_of_todays_brain_has_a_text():
    bn = _notices()
    data = bn.manifest(_found(["libcrypto.3.dylib", "libssl.3.dylib", "libzstd.1.dylib"]))
    assert data["runtime_without_text"] == []
    assert bn.problems({**data, "distributions": [{"name": "numpy"}]}) == []


def test_a_runtime_library_nobody_has_a_text_for_stops_the_package():
    bn = _notices()
    data = bn.manifest(_found(["libzstd.1.dylib", "libmpdec.4.dylib"]))
    assert data["runtime_without_text"] == ["libmpdec.4.dylib"]
    found = bn.problems({**data, "distributions": [{"name": "numpy"}]})
    assert any("libmpdec.4.dylib" in p for p in found), found


def test_an_older_manifest_without_the_field_is_not_refused_for_it():
    bn = _notices()
    assert bn.problems({"distributions": [{"name": "numpy"}]}) == []
