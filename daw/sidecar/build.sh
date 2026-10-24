#!/usr/bin/env bash
# Build the self-contained DAW brain binary (docs/10-daw.md, Milestone B).
# The native DAW spawns this instead of `python orchestrator/daw_server.py`, so
# the shipped .app needs no system Python / venv.
#
#   daw/sidecar/build.sh                 # uses server/.venv
#   PYTHON=/path/to/python build.sh      # override interpreter
#
# Output: daw/sidecar/dist/mixagent_daw_brain/ — a ONEDIR folder: the executable of the same
# name + _internal/ (2026-09-22: the onefile re-extracted ~240 MB on every launch, 20-34 s to
# the handshake). The C++ resolveBrainCommand finds it via MIXAGENT_DAW_BRAIN (the folder or
# its executable); package.sh copies the folder into the .app's Contents/Resources/. Plus
# dist/mixagent_daw_brain.stamp — the source stamp
# (stamp.sh) it was frozen from. The same stamp is bundled INSIDE the binary, where
# `hello` reports it, so package.sh can prove the brain it ships is this one.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PYTHON="${PYTHON:-$REPO/server/.venv/bin/python}"
BIN_DIR="$HERE/dist/mixagent_daw_brain"
BIN="$BIN_DIR/mixagent_daw_brain"
STAMP_FILE="$BIN_DIR.stamp"

if ! "$PYTHON" -c "import PyInstaller" 2>/dev/null; then
  echo "PyInstaller missing — installing into $($PYTHON -c 'import sys; print(sys.prefix)')"
  "$PYTHON" -m pip install --quiet pyinstaller
fi

# A failed or interrupted build must never leave a binary that LOOKS current: drop the old
# stamp before anything else, and write the new one only once the binary is proven built.
rm -f "$STAMP_FILE"
# …and the old output, whichever shape it has: a onefile FILE at this path would block the
# onedir FOLDER of the same name.
rm -rf "$BIN_DIR"
STAMP="$(PYTHON="$PYTHON" "$HERE/stamp.sh")"
mkdir -p "$HERE/stamp"
printf '%s\n' "$STAMP" > "$HERE/stamp/brain_stamp.txt"   # bundled by the spec (datas)

cd "$HERE"
"$PYTHON" -m PyInstaller --noconfirm --clean \
  --workpath build --distpath dist mixagent_daw_brain.spec

if [ ! -x "$BIN" ]; then
  echo "build failed: $BIN not produced" >&2
  exit 1
fi
# Another session editing the brain mid-freeze would give a binary of NO single source state,
# stamped as the one it started from. Re-stamp and refuse on any change.
AFTER="$(PYTHON="$PYTHON" "$HERE/stamp.sh")"
if [ "$AFTER" != "$STAMP" ]; then
  echo "the brain's sources changed during the build ($STAMP -> $AFTER) — rerun it" >&2
  exit 1
fi
printf '%s\n' "$STAMP" > "$STAMP_FILE"
echo "DAW brain: $BIN (stamp $STAMP)"
echo "point the DAW at it with:  export MIXAGENT_DAW_BRAIN=\"$BIN_DIR\""
