"""Phase 54, step 1: the brain renders once per change.

Measured 2026-10-12 on the second showcase's rehearsal: the DAW's brain rendered and
re-measured the whole mix before every chat message, after every reply, and after every
command whose reply carried a co-edit - a clip edit included, because the app reports the
stem's new clip windows as an engine fact. Forty clip commands were forty full renders; that
was most of each 17 s round of a 23-minute arrange turn. Every analysis tool rendered from the
files again, because the stem cache the engine has had since Phase 10 was only ever switched
on by the old gateway.

Now: a master render is kept under the session's render fingerprint (`dsp.MASTER_MEMO`), with
what was measured on it; a fact report is mirrored and starts no measurement; and the brain
switches the caches on when it starts. Off, as everywhere else in this suite, nothing changes.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from mix_engine import dsp
from mix_engine.session import RENDER_BOOKKEEPING, Session
from orchestrator import chat as chatmod
from orchestrator import daw_bridge as B
from orchestrator import daw_server as S

REPO = Path(__file__).resolve().parents[2]


def _project(tmp_path: Path) -> Path:
    proj = tmp_path / "proj"
    proj.mkdir()
    sr = 44100
    t = np.arange(2 * sr) / sr
    for name, hz, amp in (("vocals", 220, 0.2), ("bass", 55, 0.3), ("keys", 440, 0.1)):
        sf.write(proj / f"{name}.wav", (amp * np.sin(2 * np.pi * hz * t)).astype("float32"), sr)
    Session(proj)
    return proj


@pytest.fixture
def kept():
    """The caches on, as the DAW's brain runs (conftest switches them off again)."""
    dsp.configure_render_cache(512)
    yield
    dsp.configure_render_cache(0)
    dsp.RENDER_CACHE.clear()


def _masters() -> int:
    return dsp.RENDER_COUNTS["master"]


# ---------------------------------------------------------------- the fingerprint

def test_the_fingerprint_moves_with_everything_but_the_bookkeeping(tmp_path):
    proj = _project(tmp_path)
    base = Session(proj, read_only=True).render_fingerprint()
    assert Session(proj, read_only=True).render_fingerprint() == base, "the same state, twice"

    def with_state(change) -> str:
        session = Session(proj, read_only=True)
        change(session._state)
        return session.render_fingerprint()

    for key in sorted(RENDER_BOOKKEEPING):           # the ledger, the counters, the journal
        assert with_state(lambda st: st.__setitem__(key, ["moved"])) == base, key
    for key in Session(proj, read_only=True)._state:  # ...and every other key a session holds
        if key not in RENDER_BOOKKEEPING:
            assert with_state(lambda st: st.__setitem__(key, "moved")) != base, key
    assert with_state(lambda st: st.__setitem__("a_key_nobody_listed", 1)) != base, \
        "a state key added later is fingerprinted until it is listed as bookkeeping"
    for field, value in (("gain_db", -3.0), ("pan", 0.5), ("muted", True), ("excluded", True),
                         ("chain", [{"type": "eq"}]), ("clips", [[0.0, 0.5, 1.0]])):
        assert with_state(lambda st: st["stems"]["bass"].__setitem__(field, value)) != base, field


def test_the_fingerprint_moves_when_a_stems_file_does(tmp_path):
    proj = _project(tmp_path)
    base = Session(proj, read_only=True).render_fingerprint()
    sr = 44100
    sf.write(proj / "bass.wav", np.zeros(sr, dtype="float32"), sr)   # another length
    assert Session(proj, read_only=True).render_fingerprint() != base


def test_no_bookkeeping_key_is_read_by_the_render():
    """The list is a claim about dsp.py: none of these keys may appear in it."""
    source = (REPO / "server" / "mix_engine" / "dsp.py").read_text()
    read = [key for key in RENDER_BOOKKEEPING
            if re.search(rf"""["']{key}["']|session\.{key}\b|\._state\b.*{key}""", source)]
    assert read == [], f"the render reads {read}: take it out of RENDER_BOOKKEEPING"


# ---------------------------------------------------------------- the memo

def test_an_unchanged_session_is_rendered_once_and_a_changed_one_again(tmp_path, kept):
    proj = _project(tmp_path)
    before = _masters()
    first, sr = dsp.render_master(Session(proj, read_only=True))
    again, _ = dsp.render_master(Session(proj, read_only=True))
    assert _masters() - before == 1 and again is first
    assert not first.flags.writeable, "a kept render cannot be edited in place"
    with pytest.raises(ValueError):
        first[0, 0] = 1.0

    stems_before = dsp.RENDER_COUNTS["stem"]
    moved = Session(proj, read_only=True)
    moved._state["stems"]["bass"]["gain_db"] = -6.0
    quieter, _ = dsp.render_master(moved)
    assert _masters() - before == 2 and dsp.RENDER_COUNTS["stem"] - stems_before == 1, \
        "one fader: one stem and the master, not the session"
    assert not np.array_equal(quieter, first)


def test_a_kept_render_is_the_render(tmp_path, kept):
    """Bit for bit, with its report and both telemetries, and a range cut from it."""
    proj = _project(tmp_path)
    session = Session(proj)
    session.user_edit("master", {"gain_db": -2.0})
    dsp.configure_render_cache(0)
    plain_report, plain_tel, plain_stems = {}, [], {}
    plain, sr = dsp.render_master(Session(proj, read_only=True), report=plain_report,
                                  telemetry=plain_tel, stem_telemetry=plain_stems)
    plain_cut, _ = dsp.render_master(Session(proj, read_only=True), range_s=(0.5, 1.25))
    plain_pre, _ = dsp.render_master(Session(proj, read_only=True), pre_master=True)

    dsp.configure_render_cache(512)
    before = _masters()
    for _ in range(2):                       # a miss, then a hit
        report, tel, stems = {}, [], {}
        audio, rate = dsp.render_master(Session(proj, read_only=True), report=report,
                                        telemetry=tel, stem_telemetry=stems)
        assert rate == sr and np.array_equal(audio, plain)
        assert report == plain_report and tel == plain_tel and stems == plain_stems
    cut, _ = dsp.render_master(Session(proj, read_only=True), range_s=(0.5, 1.25))
    assert np.array_equal(cut, plain_cut)
    assert _masters() - before == 1, "three asks of one state: one render"
    pre, _ = dsp.render_master(Session(proj, read_only=True), pre_master=True)
    assert np.array_equal(pre, plain_pre) and _masters() - before == 2

    with pytest.raises(ValueError, match="outside"):
        dsp.render_master(Session(proj, read_only=True), range_s=(9.0, 10.0))


def test_what_is_measured_on_a_render_is_kept_with_it(tmp_path, kept):
    proj = _project(tmp_path)
    calls = []

    def measure():
        calls.append(1)
        return {"lufs_i": -14.0, "bands": [1, 2]}
    session = Session(proj, read_only=True)
    assert dsp.kept_with_master(session, "x", measure) == {"lufs_i": -14.0, "bands": [1, 2]}
    assert len(calls) == 1, "no kept render yet: it is just computed"
    dsp.render_master(session)
    first = dsp.kept_with_master(Session(proj, read_only=True), "x", measure)
    first["bands"].append(3)                 # the caller's own copy
    second = dsp.kept_with_master(Session(proj, read_only=True), "x", measure)
    assert second == {"lufs_i": -14.0, "bands": [1, 2]} and len(calls) == 2


def test_with_the_caches_off_nothing_is_kept(tmp_path):
    proj = _project(tmp_path)
    before = _masters()
    a, _ = dsp.render_master(Session(proj, read_only=True))
    b, _ = dsp.render_master(Session(proj, read_only=True))
    assert _masters() - before == 2 and a is not b and a.flags.writeable


# ---------------------------------------------------------------- the bridge

def test_the_readout_is_rendered_once_while_nothing_changes(tmp_path, kept):
    proj = _project(tmp_path)
    cache: dict = {}
    before = _masters()
    first = B.session_readout(proj, cache)
    assert B.session_readout(proj, cache) == first
    assert B.chains_with_readout(proj, cache=cache)["readout"] == first
    assert _masters() - before == 1, "the turn's start, its tail and the chains: one render"
    Session(proj).user_edit("bass", {"gain_db": -9.0})
    assert B.session_readout(proj, cache) != first and _masters() - before == 2


def test_the_master_analysis_is_kept_and_carries_the_current_pass_number(tmp_path, kept):
    import anyio
    from mcp.shared.memory import create_connected_server_and_client_session

    from mix_engine.mcp_server import create_server
    proj = _project(tmp_path)

    async def ask() -> dict:
        async with create_connected_server_and_client_session(
                create_server(proj)._mcp_server) as engine:
            out = await engine.call_tool("analyze_session", {})
            payload = out.structuredContent
            return payload.get("result", payload)
    before = _masters()
    first = anyio.run(ask)
    state = json.loads((proj / "session.json").read_text())
    state["pass_number"] = 7                 # bookkeeping: not a reason to render again
    (proj / "session.json").write_text(json.dumps(state))
    second = anyio.run(ask)
    assert _masters() - before == 1
    assert second["pass_number"] == 7 and first["pass_number"] != 7
    assert second["lufs_i"] == first["lufs_i"]


# ---------------------------------------------------------------- a fact report is not a move

def test_the_fact_report_fields_are_the_apps():
    """`coedit_is_fact_report` is App.cpp's `coeditIsFactReport`, field for field."""
    source = (REPO / "daw" / "Source" / "App.cpp").read_text()
    body = source[source.index("static bool coeditIsFactReport"):]
    body = body[:body.index("return true;")]
    assert set(re.findall(r'name != "([a-z_]+)"', body)) == set(B.FACT_REPORT_FIELDS)


def test_a_fact_report_is_told_from_a_move():
    assert B.coedit_is_fact_report("/api/set_stem?stem=bass&clips=0.0:1.0:2.0:0.0:0.0:0.0")
    assert B.coedit_is_fact_report("/api/set_stem?stem=a%23b&vca_db=-3&delay_ms=4&frozen=")
    assert not B.coedit_is_fact_report("/api/set_stem?stem=bass&gain_db=-3")
    assert not B.coedit_is_fact_report("/api/set_stem?stem=bass&clips=-&muted=1")
    assert not B.coedit_is_fact_report("/api/device_param?stem=bass&device=comp&value=2")
    assert not B.coedit_is_fact_report("/api/set_send?source=bass&return=r1&gain_db=-9")
    assert not B.coedit_is_fact_report({"batch": "b1"})


class _Scripted:
    """Stands in for ChatSession: fixed tool calls through the real execute_tool."""

    def __init__(self, script):
        from orchestrator.turn_timing import TurnTimer
        self.script, self.history, self.usage, self._model, self.log = list(script), [], {}, "", []
        self.timing = TurnTimer()
        self.note = ""
    asks_after_calls = True

    def set_settings(self, model, reasoning):
        pass

    async def turn(self, text, *, tools, context, emit, should_cancel=None,
                   live_context=None, during_run=False):
        self.history.append({"role": "user", "content": text})
        for name, args in self.script:
            out = await chatmod.execute_tool(tools, name, args)
            self.log.append((name, out.text, out.error))
        if self.asks_after_calls and getattr(tools, "after_calls", None) is not None:
            self.note = await tools.after_calls()   # the script is one response (Phase 55)
        await emit({"type": "chat_message", "text": "ok"})


def _turn(proj: Path, script, coedits, monkeypatch, brain=None):
    monkeypatch.setenv("MIX_AGENT_DATA_DIR", str(proj.parent / "data"))
    brain = brain or S.Brain(proj)

    def invoke(command, args=None, *a, **kw):
        out = {"ok": True}
        if command in coedits:
            out["coedits"] = list(coedits[command])
        return out
    brain.daw_invoke = invoke
    session = _Scripted(script)
    monkeypatch.setitem(B._CHAT_SESSIONS, B._session_key(proj, ""), session)
    before = _masters()
    out = B.chat_turn(proj, "scripted", mode="agent", use_llm=False,
                      edit_state=lambda: brain.latest_edit,
                      daw_invoke=brain.counting_invoke(), cache=brain.render_cache,
                      routing_ops=brain.routing_ops(),
                      mirror_coedits=brain._mirror_coedits,
                      register_stem=brain._register_stem)
    return session, out, _masters() - before, brain


FACT = "/api/set_stem?stem=bass&clip_at_s=0.0&clip_from_s=0.25&clip_for_s=1.0"
MOVE = "/api/set_stem?stem=bass&gain_db=-4.0"


def test_six_clip_commands_start_no_measurement(tmp_path, monkeypatch, kept):
    proj = _project(tmp_path)
    script = [("daw_command", {"command": "clip.trim", "args": {"track": "bass", "clip": 0}})] * 6
    session, out, masters, _ = _turn(proj, script, {"clip.trim": [FACT]}, monkeypatch)
    assert all(not error for _, _, error in session.log)
    assert all("measured" not in json.loads(text) for _, text, _ in session.log)
    assert masters == 1, "the turn's own context; none of the six reports"
    assert Session(proj, read_only=True).stems["bass"].get("clip"), "the report WAS mirrored"
    assert out["timing"]["renders"]["master"] == 1 and "measure" not in out["timing"]["marks_s"]


def test_a_fader_move_is_measured_once_and_only_its_stem_is_rendered(tmp_path, monkeypatch, kept):
    proj = _project(tmp_path)
    B.session_readout(proj, {})              # the session as the DAW's brain holds it: rendered
    stems_before = dsp.RENDER_COUNTS["stem"]
    script = [("daw_command", {"command": "track.set_gain", "args": {"track": "bass"}})]
    session, out, masters, _ = _turn(proj, script, {"track.set_gain": [MOVE]}, monkeypatch)
    assert session.note.startswith("measured after these commands"), \
        "the agent sees what its move did (after the response's calls, since Phase 55)"
    assert "measured" not in json.loads(session.log[0][1])
    assert masters == 1 and dsp.RENDER_COUNTS["stem"] - stems_before == 1
    assert out["timing"]["mark_counts"]["measure"] == 1


def test_a_turn_that_changes_nothing_renders_nothing(tmp_path, monkeypatch, kept):
    proj = _project(tmp_path)
    brain = S.Brain(proj)
    B.session_readout(proj, brain.render_cache)
    script = [("daw_command", {"command": "transport.seek", "args": {"position_s": 4.0}})]
    _, out, masters, _ = _turn(proj, script, {}, monkeypatch, brain=brain)
    assert masters == 0 and out["timing"]["renders"] == {"master": 0, "stem": 0}


def test_the_brain_switches_the_caches_on_and_a_setting_can_switch_them_off(monkeypatch):
    monkeypatch.delenv("MIX_AGENT_RENDER_CACHE_MB", raising=False)
    assert 256 <= S.render_cache_mb() <= 4096
    monkeypatch.setenv("MIX_AGENT_RENDER_CACHE_MB", "0")
    assert S.render_cache_mb() == 0
    monkeypatch.setenv("MIX_AGENT_RENDER_CACHE_MB", "1500")
    S._keep_renders()
    assert dsp.RENDER_CACHE.enabled and dsp.MASTER_MEMO.enabled
    assert dsp.RENDER_CACHE.budget_bytes == 1500 * 1024 * 1024
    main = S.__dict__["main"]
    import inspect
    assert "_keep_renders()" in inspect.getsource(main), "the brain's entry point switches them on"
