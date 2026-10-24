#!/usr/bin/env bash
# Print the DAW brain's SOURCE STAMP: 16 hex chars that change whenever anything the frozen
# brain is built from changes — the engine + orchestrator Python, the bundled models, this
# directory's entry point + spec, and the interpreter's installed packages.
#
# Why (2026-09-22 launch review, R1): package.sh reused daw/sidecar/dist/mixagent_daw_brain
# whenever the file EXISTED, so builds 1397 and the first 1402 shipped a brain frozen on
# 2026-08-15 inside a September app. Its handshake carried no auth token and the app refused
# it, so the published engineer could not connect at all. Existence is not freshness; this
# stamp is. build.sh writes it next to the binary and into it (`hello` reports it), and
# package.sh rebuilds on any mismatch and checks the bundled binary reports the same one.
#
#   daw/sidecar/stamp.sh                  # uses server/.venv, like build.sh
#   PYTHON=/path/to/python stamp.sh       # the interpreter the brain is frozen with
#
# Deliberately coarse: it hashes every orchestrator/*.py, not only the modules the freeze
# pulls in, so an eval-harness edit costs one unnecessary two-minute rebuild. A stale brain
# costs a release.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PYTHON="${PYTHON:-$REPO/server/.venv/bin/python}"
# A RELATIVE interpreter path names a file relative to the CALLER's directory; resolve it
# before the cd below would silently re-point it (a bare name like python3 stays a PATH lookup).
case "$PYTHON" in
  /*) ;;
  */*) PYTHON="$(cd "$(dirname "$PYTHON")" && pwd)/$(basename "$PYTHON")" ;;
esac

cd "$REPO"
{
  # *.py alone keeps bytecode out (__pycache__ holds only .pyc, which changes on every import)
  find orchestrator server/mix_engine -type f \
       \( -name '*.py' -o -path 'server/mix_engine/models/*' \) -print0 \
    | LC_ALL=C sort -z | xargs -0 shasum -a 256
  # The freeze recipe: the entry point, the spec, and what the spec RUNS (bundle_notices.py
  # writes two files into the bundle). Every *.py here, so a new helper is in by itself.
  find daw/sidecar -maxdepth 1 -type f \( -name '*.py' -o -name '*.spec' \) -print0 \
    | LC_ALL=C sort -z | xargs -0 shasum -a 256
  # The dependencies ARE part of the brain: an upgraded anthropic/numpy in the venv changes
  # the binary without a single repo file moving.
  "$PYTHON" -m pip freeze --all
} | shasum -a 256 | cut -c1-16
