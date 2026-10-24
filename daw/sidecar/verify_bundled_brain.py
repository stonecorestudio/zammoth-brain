#!/usr/bin/env python3
"""Packaging gate: the brain INSIDE the app must be the brain these sources build, and it must
start inside the app's handshake wait.

    verify_bundled_brain.py <.../Zammoth.app/Contents/Resources/mixagent_daw_brain/mixagent_daw_brain>
                            --stamp <hex>

Launches that exact binary the way the app does (an empty project dir, port 0), in a scratch
HOME/TMPDIR with a clean environment, TWICE, and checks what the app itself needs from it:

  1. the handshake line carries the per-spawn auth token (BrainClient::launch refuses a brain
     without one — "the brain's handshake carried no auth token");
  2. `auth` then `hello` round-trip, and the hello carries the fields the app reads
     (`license`, `api_key`, `spend_is_estimate`, `brain_stamp`);
  3. the hello's `brain_stamp` equals `--stamp`, the stamp of the sources being packaged;
  4. the FIRST start (freshly written files: macOS checks each new library once, the cost the
     first launch after an install or an update pays) fits inside the app's handshake wait;
  5. the SECOND start is fast — the steady state every later launch gets. A slow second start
     means the brain redoes work on every launch, which is what the onefile build did.

  6. the bundle names what it is made of (`check_contents`): its manifest lists every
     third-party distribution with a licence, the notices file carries each one, and none of
     them is test or developer tooling.

Why (2026-09-22 launch review): package.sh reused a brain binary whenever one existed, so builds
1397 and the first 1402 shipped a brain frozen on 2026-08-15 that the app refused to talk to (R1);
and the onefile brain decompressed ~240 MB on every launch, 20-34 s against a 25 s wait (R2).
Nothing in the pipeline ever RAN the bundled binary, so neither was seen before a user would.

Stdlib only. Touches neither the keychain nor the network: ANTHROPIC_API_KEY is set to a
placeholder, so the brain reads the key from the environment (no `security` call) and the
licence gate stays off (no licence-service call). Exit 0 = ship it; anything else prints why.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))      # bundle_notices, beside this file

#: The app's handshake wait — daw/Source/BrainClient.h `kHandshakeTimeoutMs` (pinned equal by
#: server/tests/test_brain_stamp.py). A first start slower than this never connects.
APP_HANDSHAKE_WAIT_S = 90.0
#: A warm start is ~0.8 s for the onedir brain; the onefile one was 20-34 s EVERY launch.
WARM_MAX_S = 10.0
HANDSHAKE = re.compile(r"MIXAGENT_LISTENING port=(\d+)(?: token=(\S+))?")
#: The hello fields App.cpp reads (ensureBrain). A brain without them predates Phase 53.
HELLO_FIELDS = ("license", "api_key", "spend_is_estimate", "brain_stamp")


class GateFailure(Exception):
    pass


def check_contents(brain: Path) -> int:
    """What the bundle says it is made of (2026-10-07, reader S7.R15). The freeze writes
    `python_bundle.json` and `THIRD_PARTY_PYTHON.txt` at its archive root from PyInstaller's
    own analysis (bundle_notices.py); a brain without them, or one whose manifest names test
    tooling or code no distribution accounts for, is not shipped. Returns how many third-party
    distributions it names."""
    import bundle_notices as notices
    root = brain.parent / "_internal"
    manifest, text = root / notices.MANIFEST_NAME, root / notices.NOTICES_NAME
    if not manifest.is_file() or not text.is_file():
        raise GateFailure("the bundled brain says nothing about the third-party Python inside "
                          f"it: no {notices.MANIFEST_NAME} / {notices.NOTICES_NAME} in "
                          f"{root} - was it frozen by an older recipe? (REBUILD_SIDECAR=1)")
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GateFailure(f"{manifest.name} is unreadable: {exc}") from None
    problems = notices.problems(data)
    if problems:
        raise GateFailure("the bundled brain must not ship: " + "; ".join(problems))
    body = text.read_text(encoding="utf-8", errors="replace")
    unnamed = [d["name"] for d in data["distributions"]
               if f"\n{d['name']} {d['version']}\n" not in body]
    if unnamed:
        raise GateFailure(f"{text.name} does not name what the manifest lists: "
                          + ", ".join(unnamed[:8]))
    return len(data["distributions"])


def _tail(log: Path) -> str:
    if not log.is_file():
        return ""
    lines = log.read_text(errors="replace").splitlines()[-15:]
    return ("\n    --- brain stderr (tail) ---\n" + "\n".join("    " + l for l in lines)) if lines else ""


def _request(sock: socket.socket, buf: bytearray, obj: dict, deadline: float) -> dict:
    """Send one request line; return the reply carrying its id (events are skipped)."""
    sock.sendall((json.dumps(obj) + "\n").encode())
    while True:
        while b"\n" not in buf:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"no reply to {obj.get('method')!r}")
            sock.settimeout(left)
            chunk = sock.recv(65536)
            if not chunk:
                raise ConnectionError(f"the brain closed the connection during {obj.get('method')!r}")
            buf += chunk
        line, _, rest = bytes(buf).partition(b"\n")
        buf[:] = rest
        reply = json.loads(line)
        if reply.get("id") == obj.get("id"):
            return reply


def _stop(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def _start_once(brain: Path, work: Path, env: dict, timeout_s: float, label: str):
    """Launch the brain; return (proc, port, token, seconds to the handshake). The caller stops it."""
    out, err = work / f"{label}.stdout", work / f"{label}.stderr"
    started = time.monotonic()
    with out.open("w") as so, err.open("w") as se:
        proc = subprocess.Popen([str(brain), str(work / "project"), "--port", "0"],
                                stdout=so, stderr=se, stdin=subprocess.DEVNULL, env=env, cwd=work)
    while time.monotonic() - started < timeout_s:
        m = HANDSHAKE.search(out.read_text(errors="replace"))
        if m:
            return proc, int(m.group(1)), m.group(2), time.monotonic() - started
        if proc.poll() is not None:
            raise GateFailure(f"the brain exited (code {proc.returncode}) before its handshake"
                              + _tail(err))
        time.sleep(0.05)
    _stop(proc)
    raise GateFailure(f"no handshake within {timeout_s:.0f} s ({label} start)" + _tail(err))


def verify(brain: Path, stamp: str, timeout_s: float, cold_max: float, warm_max: float) -> int:
    brain = brain.resolve()      # it is started from a scratch folder: a relative path named nothing there
    if not brain.is_file():
        print(f"    BUNDLED BRAIN GATE FAILED: no brain executable at {brain}", file=sys.stderr)
        return 1
    work = Path(tempfile.mkdtemp(prefix="zammoth-brain-gate-"))
    (work / "project").mkdir()
    (work / "home").mkdir()
    env = {"HOME": str(work / "home"), "PATH": "/usr/bin:/bin", "TMPDIR": str(work) + "/",
           "LANG": "en_US.UTF-8",
           "MIX_AGENT_DATA_DIR": str(work / "appdata"),
           # a placeholder: the brain takes the env path, so neither the keychain nor the
           # licence service is consulted (see the module docstring)
           "ANTHROPIC_API_KEY": "packaging-gate-placeholder-not-a-key"}
    try:
        # --- the first start: freshly written files, the full protocol ---------------------
        proc, port, token, cold_s = _start_once(brain, work, env, timeout_s, "cold")
        try:
            if not token:
                raise GateFailure("the handshake carries no auth token — this brain predates "
                                  "2026-09-10, and the app refuses it (\"the brain's handshake "
                                  "carried no auth token\")")
            deadline = time.monotonic() + 60.0
            with socket.create_connection(("127.0.0.1", port), timeout=10.0) as sock:
                buf = bytearray()
                auth = _request(sock, buf, {"id": 1, "method": "auth",
                                            "params": {"token": token}}, deadline)
                if not (auth.get("result") or {}).get("ok"):
                    raise GateFailure(f"auth refused: {auth}")
                hello = _request(sock, buf, {"id": 2, "method": "hello", "params": {}}, deadline)
        finally:
            _stop(proc)
        result = hello.get("result") or {}
        if not result.get("ok"):
            raise GateFailure(f"hello failed: {hello}")
        missing = [f for f in HELLO_FIELDS if f not in result]
        if missing:
            raise GateFailure(f"the hello lacks {missing} — an older brain than this app speaks to")
        got = str(result.get("brain_stamp"))
        if got != stamp:
            raise GateFailure(f"the bundled brain was built from other sources: it reports stamp "
                              f"{got!r}, the sources being packaged are {stamp!r}")
        if cold_s > cold_max:
            raise GateFailure(f"the first start took {cold_s:.1f} s; the app waits {cold_max:.0f} s "
                              "for the handshake, so the first launch after an install would never "
                              "reach its engineer")
        # --- the second start: what every later launch pays ------------------------------
        proc, _port, token2, warm_s = _start_once(brain, work, env, timeout_s, "warm")
        _stop(proc)
        if not token2:
            raise GateFailure("the second start's handshake carries no auth token")
        if warm_s > warm_max:
            raise GateFailure(f"the second start took {warm_s:.1f} s (limit {warm_max:.0f} s): the "
                              "brain redoes work on EVERY launch — is it a onefile build again? "
                              "(onefile re-extracted ~240 MB per launch, 20-34 s)")
        named = check_contents(brain)
        print(f"    bundled brain OK: stamp {got}, auth token present, hello ok; handshake "
              f"{cold_s:.1f} s first start (app waits {cold_max:.0f} s), {warm_s:.1f} s after; "
              f"{named} third-party distributions named with their licences, no test tooling")
        return 0
    except GateFailure as exc:
        print(f"    BUNDLED BRAIN GATE FAILED: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TimeoutError, ConnectionError) as exc:
        print(f"    BUNDLED BRAIN GATE FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("brain", type=Path)
    ap.add_argument("--stamp", required=True, help="stamp.sh's output for the packaged sources")
    ap.add_argument("--timeout", type=float, default=240.0,
                    help="seconds to wait for a handshake before giving up on the check itself")
    ap.add_argument("--cold-max", type=float, default=APP_HANDSHAKE_WAIT_S,
                    help="the first start must handshake within this (the app's wait)")
    ap.add_argument("--warm-max", type=float, default=WARM_MAX_S,
                    help="the second start must handshake within this")
    args = ap.parse_args()
    return verify(args.brain, args.stamp.strip(), args.timeout, args.cold_max, args.warm_max)


if __name__ == "__main__":
    sys.exit(main())
