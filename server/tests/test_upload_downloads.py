"""daw/scripts/upload_downloads.sh - the step that puts a published build on zammoth.com/download/.

Run in a scratch tree with stand-ins for `fly` (a local folder plays the site's volume), `curl`,
`hdiutil` and `xcrun`; `ditto`, `shasum` and PlistBuddy are the real ones. Each refusal is read by
its own sentence, and a refusal must come before anything is uploaded.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "daw" / "scripts" / "upload_downloads.sh"
SITE = "https://zammoth.com/download"

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="ditto and PlistBuddy are macOS tools")

FLY = r"""#!/bin/bash
# Stand-in fly: the volume is $FAKE_VOL. Every call is logged.
echo "fly $*" >> "$FAKE_LOG"
if [ "$1 $2" = "ssh console" ]; then
  cmd="${@: -1}"
  cmd="${cmd//\/downloads/$FAKE_VOL}"
  cmd="${cmd/#sha256sum/shasum -a 256}"
  sh -c "$cmd"
elif [ "$1 $2 $3" = "ssh sftp put" ]; then
  src="${@: -2:1}"; dst="${@: -1}"
  cp "$src" "${dst/\/downloads/$FAKE_VOL}"
  [ -n "${FAKE_CORRUPT:-}" ] && printf x >> "${dst/\/downloads/$FAKE_VOL}"
  true
fi
"""
CURL = '#!/bin/bash\necho "curl $*" >> "$FAKE_LOG"\nprintf 200\n'
XCRUN = '#!/bin/bash\necho "xcrun $*" >> "$FAKE_LOG"\nexit "${FAKE_STAPLE_RC:-0}"\n'
HDIUTIL = r"""#!/bin/bash
echo "hdiutil $*" >> "$FAKE_LOG"
if [ "$1" = attach ]; then
  for ((i = 1; i <= $#; i++)); do [ "${!i}" = "-mountpoint" ] && { j=$((i + 1)); mp="${!j}"; }; done
  cp -R "$FAKE_DMG_CONTENT/." "$mp/"
fi
"""


def _app(at: Path, build: int) -> Path:
    plist = at / "Zammoth.app" / "Contents" / "Info.plist"
    plist.parent.mkdir(parents=True)
    plist.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<plist version="1.0"><dict>'
        f"<key>CFBundleVersion</key><string>{build}</string></dict></plist>\n")
    return at / "Zammoth.app"


def _tree(tmp: Path, build: int = 4242, zip_build: int | None = None, dmg_build: int | None = None,
          url: str | None = None) -> dict:
    channel = tmp / "channel"
    channel.mkdir()
    app = _app(tmp / "zipsrc", zip_build or build)
    zipf = channel / f"Zammoth-{build}.zip"
    subprocess.run(["/usr/bin/ditto", "-c", "-k", "--keepParent", str(app), str(zipf)], check=True)
    sha = subprocess.run(["shasum", "-a", "256", str(zipf)], capture_output=True, text=True,
                         check=True).stdout.split()[0]
    (channel / "appcast.json").write_text(json.dumps(
        {"build": build, "url": url or f"{SITE}/Zammoth-{build}.zip", "sha256": sha}))
    _app(tmp / "dmgcontent", dmg_build or build)
    dmg = tmp / "Zammoth.dmg"
    dmg.write_bytes(b"a disk image")
    bin_ = tmp / "bin"
    bin_.mkdir()
    for name, body in {"fly": FLY, "curl": CURL, "xcrun": XCRUN, "hdiutil": HDIUTIL}.items():
        (bin_ / name).write_text(body)
        (bin_ / name).chmod(0o755)
    vol = tmp / "volume"
    vol.mkdir()
    return {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "CHANNEL_DIR": str(channel),
            "DMG": str(dmg), "FAKE_VOL": str(vol), "FAKE_LOG": str(tmp / "calls.log"),
            "FAKE_DMG_CONTENT": str(tmp / "dmgcontent")}


def _run(env: dict, build: str = "4242") -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT), build], env=env, capture_output=True, text=True,
                          timeout=60)


def _calls(env: dict) -> str:
    log = Path(env["FAKE_LOG"])
    return log.read_text() if log.exists() else ""


def test_a_checked_build_is_uploaded_under_three_names_and_read_back(tmp_path):
    env = _tree(tmp_path)
    r = _run(env)
    assert r.returncode == 0, r.stderr
    vol = Path(env["FAKE_VOL"])
    assert sorted(p.name for p in vol.iterdir()) == ["Zammoth-4242.dmg", "Zammoth-4242.zip", "Zammoth.dmg"]
    assert (vol / "Zammoth.dmg").read_bytes() == Path(env["DMG"]).read_bytes()
    assert "sftp put" in _calls(env)
    assert "fly secrets set DAW_APPCAST_JSON" in r.stdout      # said, never run
    assert "fly secrets" not in _calls(env)


def test_an_appcast_that_points_elsewhere_is_refused_before_any_upload(tmp_path):
    env = _tree(tmp_path, url="https://example.com/Zammoth-4242.zip")
    r = _run(env)
    assert r.returncode == 3
    assert "the appcast does not describe this zip at this address" in r.stderr
    assert "fly " not in _calls(env)


def test_a_dmg_of_another_build_is_refused_before_any_upload(tmp_path):
    env = _tree(tmp_path, dmg_build=4241)
    r = _run(env)
    assert r.returncode == 3
    assert "the DMG holds build 4241" in r.stderr
    assert "fly " not in _calls(env)


def test_a_zip_holding_another_build_is_refused(tmp_path):
    env = _tree(tmp_path, zip_build=4241)
    r = _run(env)
    assert r.returncode == 3
    assert "the app in the zip is build 4241" in r.stderr
    assert "fly " not in _calls(env)


def test_a_build_without_a_notarization_ticket_is_refused(tmp_path):
    env = _tree(tmp_path)
    env["FAKE_STAPLE_RC"] = "65"
    r = _run(env)
    assert r.returncode == 3
    assert "the app in the zip carries no notarization ticket" in r.stderr
    assert "fly " not in _calls(env)


def test_a_file_that_arrives_changed_on_the_server_fails_the_read_back(tmp_path):
    env = _tree(tmp_path)
    env["FAKE_CORRUPT"] = "1"
    r = _run(env)
    assert r.returncode == 1
    assert "Zammoth-4242.zip on the server is not the file uploaded" in r.stderr


def test_the_build_number_is_required(tmp_path):
    env = _tree(tmp_path)
    r = _run(env, build="")
    assert r.returncode == 3
    assert "give the build number" in r.stderr
