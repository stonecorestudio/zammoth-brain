"""The brain's NDJSON port runs nothing for a caller that has not authenticated
(2026-09-10 agent audit P0-2).

Before: `_serve_client` needed no hello or token and SKIPPED a malformed line, so a web
page's fetch() POST — its HTTP header lines skipped as malformed, its body lines
executed — could make the attached DAW run commands, start a billed run, re-point the
brain or rewrite session.json. Now the first line must be
{"method": "auth", "params": {"token": <the per-spawn secret>}}; anything else closes
the connection having executed nothing.

The refusal legs deliberately never read `hub.token`: against the pre-fix server they
run to their assertions and FAIL there (the body lines execute), instead of erroring on
a missing attribute — so they discriminate the gate itself, not the new API."""
import asyncio
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from mix_engine.session import Session
from orchestrator.daw_server import Hub, _serve_client

SET_GAIN = "http://127.0.0.1:8765/api/set_stem?stem=vox&gain_db=-5.0"


def _project(tmp_path: Path) -> Path:
    proj = tmp_path / "proj"
    proj.mkdir()
    sr = 44100
    t = np.arange(sr) / sr
    sf.write(proj / "vox.wav", (0.2 * np.sin(2 * np.pi * 220 * t)).astype("float32"), sr)
    Session(proj)   # materialise session.json with the stem
    return proj


def _gain(proj: Path) -> float:
    return json.loads((proj / "session.json").read_text())["stems"]["vox"].get("gain_db", 0.0)


def _line(obj: dict) -> str:
    return json.dumps(obj) + "\n"


async def _drain(reader, timeout: float = 3.0) -> list:
    """Every line the server sends until it closes the connection. A connection still
    open after `timeout` ends the list with the marker "<still open>"."""
    lines = []
    while True:
        try:
            line = await asyncio.wait_for(reader.readline(), timeout)
        except asyncio.TimeoutError:
            return lines + ["<still open>"]
        if not line:
            return lines
        lines.append(json.loads(line))


def _exchange(proj: Path, payload, *, with_token=None) -> list:
    """Open one connection, write `payload` (str, or a callable taking the hub's token),
    and return everything the server said before closing it."""
    async def scenario():
        hub = Hub(proj)
        server = await asyncio.start_server(
            lambda r, w: _serve_client(r, w, hub), "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            try:
                data = payload(hub.token) if callable(payload) else payload
                writer.write(data.encode())
                await writer.drain()
                return await _drain(reader, timeout=2.0 if with_token else 3.0)
            finally:
                # always: Server.wait_closed() waits for every live connection, so a leg
                # that raised with its socket open would hang the suite instead of failing
                writer.close()
        finally:
            server.close()
            await server.wait_closed()
    return asyncio.run(scenario())


def test_a_browser_shaped_post_executes_nothing(tmp_path):
    proj = _project(tmp_path)
    body = (_line({"id": 101, "method": "coedit", "params": {"urls": [SET_GAIN]}})
            + _line({"id": 102, "method": "get_chains"}))
    http = ("POST / HTTP/1.1\r\nHost: 127.0.0.1\r\nOrigin: https://evil.example\r\n"
            "Content-Type: text/plain;charset=UTF-8\r\n"
            f"Content-Length: {len(body)}\r\n\r\n" + body)
    out = _exchange(proj, http)
    assert out == [], f"an unauthenticated HTTP caller got answers: {out}"
    assert _gain(proj) == 0.0, "the POST body's coedit ran and wrote session.json"


def test_a_request_before_auth_is_refused(tmp_path):
    proj = _project(tmp_path)
    out = _exchange(proj, _line({"id": 1, "method": "coedit", "params": {"urls": [SET_GAIN]}})
                    + _line({"id": 2, "method": "get_chains"}))
    assert out == [], f"a request without auth was answered: {out}"
    assert _gain(proj) == 0.0


def test_a_wrong_token_is_refused_and_closed(tmp_path):
    proj = _project(tmp_path)
    out = _exchange(proj, _line({"id": 1, "method": "auth", "params": {"token": "not-it"}})
                    + _line({"id": 2, "method": "coedit", "params": {"urls": [SET_GAIN]}}))
    assert out == [{"id": 1, "error": "auth failed"}], out
    assert _gain(proj) == 0.0


def test_the_token_opens_the_connection(tmp_path):
    # the positive control: the same coedit, behind a valid auth, runs
    proj = _project(tmp_path)
    out = _exchange(proj, lambda token: (
        _line({"id": 1, "method": "auth", "params": {"token": token}})
        + _line({"id": 2, "method": "coedit", "params": {"urls": [SET_GAIN]}})),
        with_token=True)
    assert out[0] == {"id": 1, "result": {"ok": True}}, out
    assert out[1].get("id") == 2 and "error" not in out[1], out
    assert _gain(proj) == -5.0


def test_every_hub_mints_its_own_token(tmp_path):
    proj = _project(tmp_path)
    a, b = Hub(proj), Hub(proj)
    assert len(a.token) >= 32 and len(b.token) >= 32
    assert a.token != b.token
