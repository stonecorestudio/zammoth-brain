"""The agent's senses (2026-09-03): ask_user
with one-click options, set_plan in the feed and in context, a session digest that
says where we left off, and a measured delta after the agent's last turn."""
from __future__ import annotations

import inspect
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from orchestrator import chat as chatmod
from orchestrator import daw_bridge as B
from orchestrator import daw_server as S
from orchestrator import operate_eval as O

CATALOG = [
    {"key": "clip.fade", "name": "Fade Clip", "category": "Clip", "wants_args": True,
     "args": "{ track, clip, fade_in_s?, fade_out_s? }"},
    {"key": "clip.split", "name": "Split Clip", "category": "Clip", "wants_args": True,
     "args": "{ track, clip, at_s }"},
    {"key": "transport.loop", "name": "Loop", "category": "Transport", "user_only": True},
    {"key": "project.close", "name": "Close Project", "category": "Project",
     "needs_confirm": True},
    {"key": "rack.plugin_add", "name": "Add Plug-in to Rack", "category": "Rack",
     "wants_args": True, "args": "{ rack, chain, name }"},
]


# ------------------------------------------------------------ catalog index ----

def test_the_catalog_is_sent_whole_and_the_compact_experiment_is_gone(monkeypatch):
    """`MIX_AGENT_CATALOG=compact` (keys by family + a `find_command` tool) was built in
    2026-09 and measured worse: an extra round cost more than the tokens it saved, and it lost
    a case. It stayed behind an environment variable nothing set; removed in Phase 55."""
    monkeypatch.setenv("MIX_AGENT_CATALOG", "compact")
    full = chatmod.render_catalog(CATALOG)
    for e in CATALOG:
        assert e["key"] in full and str(e.get("args", "")) in full
    names = {t["name"] for t in chatmod.daw_tool_defs(False)}
    assert {"ask_user", "set_plan"} <= names and "find_command" not in names
    assert not hasattr(chatmod, "find_command") and not hasattr(chatmod, "catalog_mode")
    assert "ask_user" in chatmod.DAW_SYSTEM_PROMPT and "set_plan" in chatmod.DAW_SYSTEM_PROMPT


# ------------------------------------------------------------ ask + plan ----

class _Sub:
    """Stands in for the SDK: runs the two sense tools the way the model would."""
    async def chat_turn(self, *, system, prompt, tool_defs, execute, on_delta, on_tool,
                        max_turns, effort=None, should_stop=None, max_tokens=None, timer=None):
        self.plan_reply = (await execute("set_plan", {"steps": ["comp the vocal", "tune it",
                                                                "build a reverb send"],
                                                      "current": 0})).text
        self.ask_reply = (await execute("ask_user", {"question": "Which vocal clip?",
                                                     "options": ["the first", "the second"]})).text
        self.bad = await execute("ask_user", {"question": "x", "options": ["only one"]})
        return "done", {}


def _session(monkeypatch):
    monkeypatch.setenv("MIX_AGENT_LLM_BACKEND", "subscription")
    s = chatmod.ChatSession(client=object(), system=chatmod.DAW_SYSTEM_PROMPT,
                            tool_defs_for=chatmod.daw_tool_defs)
    s._sub = _Sub()
    return s


def _tools(**over):
    async def unsupported(*_a, **_k):
        return "unsupported"
    base = dict(engine_call=unsupported, propose=unsupported, start_run=unsupported,
                stop_run=unsupported, auto_mix=unsupported, file_instruction=unsupported,
                save_note=unsupported, workspace_command=unsupported, list_files=unsupported,
                import_sample=unsupported, undo_proposal=unsupported, render=unsupported,
                daw_command=unsupported)
    return chatmod.ChatTools(**{**base, **over})


def test_ask_and_plan_emit_feed_events_and_the_plan_rides_the_live_context(monkeypatch):
    import anyio
    session = _session(monkeypatch)
    events = []

    async def emit(e):
        events.append(e)

    anyio.run(lambda: session.turn("do the vocal", tools=_tools(), context="CTX", emit=emit,
                                   live_context=lambda: "LIVE"))
    kinds = [e["type"] for e in events]
    assert "chat_plan" in kinds and "chat_question" in kinds
    plan = next(e for e in events if e["type"] == "chat_plan")
    assert plan["steps"][1] == "tune it" and plan["current"] == 0 and plan["done"] == []
    q = next(e for e in events if e["type"] == "chat_question")
    assert q["options"] == ["the first", "the second"]
    assert session.plan["steps"][0] == "comp the vocal"
    assert "Your plan (set_plan" in session._with_plan("LIVE") and "[current] comp the vocal" in session._with_plan("LIVE")
    assert "end your turn" in session._sub.ask_reply
    assert session._sub.bad.error, "one option is not a question with options"
    assert "[current] comp the vocal" in session._sub.plan_reply


def test_a_host_hook_still_runs_after_the_event(monkeypatch):
    import anyio
    session = _session(monkeypatch)
    seen = []

    async def hook(question, options):
        seen.append((question, options))

    async def emit(_e):
        return None

    anyio.run(lambda: session.turn("x", tools=_tools(ask_user=hook), context="CTX", emit=emit))
    assert seen == [("Which vocal clip?", ["the first", "the second"])]


def test_the_operate_grader_counts_an_ask_step_as_asking():
    case = next(c for c in O.CORPUS if c.wants_ask)
    steps = [{"kind": "ask", "payload": {"question": "which?", "options": ["a", "b"]}}]
    verdict, detail, _ = O.grade(case, steps, "")
    assert verdict == "correct" and "ask_user" in detail


def test_the_evals_wire_the_senses():
    from orchestrator import qa_eval as Q
    for mod in (O, Q):
        src = inspect.getsource(mod.evaluate)
        assert "ask_user=do_ask" in src and "set_plan=do_plan" in src, mod.__name__


# ------------------------------------------------------------ session digest ----

def test_the_digest_shows_earlier_sessions_only_and_stays_bounded(tmp_path):
    project = tmp_path
    for i in range(3):
        B.record_session_digest(project, session_id="s", user=f"ask {i}", reply="ok.",
                                commands=["track.set_gain", "track.set_gain"])
    # nothing from THIS process shows (its turns are already in the model's history)
    assert B.session_digest_part(project) == ""
    later = datetime.now(timezone.utc) + timedelta(seconds=1)
    part = B.session_digest_part(project, before=later)
    assert "Where we left off" in part and "'ask 2' -> track.set_gain" in part
    assert part.count("track.set_gain") == 3, "commands are de-duplicated per turn"
    assert "Your last reply then: 'ok.'" in part
    B.record_session_digest(project, session_id="s", user="which one?", reply="The first or the second?",
                            commands=[])
    part = B.session_digest_part(project, before=datetime.now(timezone.utc) + timedelta(seconds=1))
    assert "open question" in part
    for i in range(B.SESSION_DIGEST_KEEP + 20):
        B.record_session_digest(project, session_id="s", user=f"n{i}", reply="", commands=[])
    lines = (project / "runs" / B.SESSION_DIGEST_FILE).read_text().splitlines()
    assert len(lines) == B.SESSION_DIGEST_KEEP


def test_the_stable_context_carries_the_digest_and_the_delta(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "_PROCESS_STARTED", datetime.now(timezone.utc) + timedelta(days=1))
    B.record_session_digest(tmp_path, session_id="", user="pull the vocal down", reply="Done.",
                            commands=["track.set_gain"])
    B.write_turn_delta(tmp_path, {**B.turn_delta({"lufs_i": -14.2}, {"lufs_i": -13.6}),
                                  "commands": ["track.set_gain"]})
    ctx = B._chat_context_stable(tmp_path, "agent", {}, {"commands": CATALOG})
    assert "Where we left off" in ctx and "pull the vocal down" in ctx
    assert "Measured effect of your last turn" in ctx and "LUFS -14.2 -> -13.6" in ctx
    assert ctx.index("Where we left off") < ctx.index("Commands available now")


# ------------------------------------------------------------ turn delta ----

def test_turn_delta_names_what_moved_and_nothing_else():
    before = {"lufs_i": -14.2, "true_peak_dbtp": -1.0, "key": "A minor", "tempo_bpm": 120.0,
              "mud_db": -3.1, "silent": False}
    after = {"lufs_i": -13.6, "true_peak_dbtp": -1.0, "key": "A minor", "tempo_bpm": 120.0,
             "mud_db": -4.0, "silent": False}
    d = B.turn_delta(before, after)
    assert d["changed"] and "LUFS -14.2 -> -13.6" in d["text"] and "mud -3.1 -> -4.0" in d["text"]
    assert "true peak" not in d["text"] and "tempo" not in d["text"]
    # the row: biggest mover first, short labels
    assert d["was"] == "MUD -3.1 · LUFS -14.2" and d["now"] == "MUD -4.0 · LUFS -13.6"
    # three movers: the sentence names all three, the one-line row the two largest
    three = B.turn_delta({"lufs_i": -14.8, "spectral_tilt_db_per_oct": -1.2, "true_peak_dbtp": -0.6},
                         {"lufs_i": -14.6, "spectral_tilt_db_per_oct": -0.9, "true_peak_dbtp": 0.6})
    assert three["text"].count("->") == 3 and "tilt dB/oct" in three["text"]
    assert three["was"] == "TP -0.6 · TILT -1.2" and three["now"] == "TP 0.6 · TILT -0.9"
    assert B.turn_delta(before, before) == {"changed": False, "was": "", "now": "", "text": ""}


def test_a_stale_or_unchanged_delta_is_not_shown(tmp_path):
    B.write_turn_delta(tmp_path, B.turn_delta({"lufs_i": -14.0}, {"lufs_i": -14.0}))
    assert B.turn_delta_part(tmp_path) == ""
    B.write_turn_delta(tmp_path, B.turn_delta({"lufs_i": -14.0}, {"lufs_i": -12.0}))
    assert "LUFS -14.0 -> -12.0" in B.turn_delta_part(tmp_path)
    later = datetime.now(timezone.utc) + timedelta(seconds=B.TURN_DELTA_FRESH_S + 5)
    assert B.turn_delta_part(tmp_path, now=later) == ""


def test_the_delta_settles_at_chat_done_never_on_a_later_coedit(tmp_path):
    """Every turn settles at chat_done (2026-09-10 audit P2, 02·F12). Settling on "the
    next co-edit batch" reported a fader the USER pulled minutes after an arrangement-only
    turn as the agent's measured effect. Driven, not grepped: park a pending delta, send
    a co-edit that moves the loudness, and nothing may call it the turn's effect."""
    import shutil
    proj = tmp_path / "p"
    shutil.copytree(Path(__file__).resolve().parents[2] / "test_projects" / "suno_demo", proj,
                    ignore=shutil.ignore_patterns("bounces", "runs", ".edits"))
    # Discover the stems HERE (session.json is gitignored, so a fresh checkout has none, and
    # Brain.stems() reads that file rather than discovering). Without this the test rode a
    # session.json left behind by earlier local runs: green on every dev machine, IndexError on
    # CI's first checkout — invisible until engine.yml could collect again (2026-09-11).
    from mix_engine.session import Session
    Session(proj)
    brain = S.Brain(proj)
    brain._pending_turn_delta = {"before": {"lufs_i": -12.2}, "commands": ["clip.delete"],
                                 "settle": "chat_done", "ts": B._now_iso()}
    stems = sorted(brain.stems())
    assert stems, "the fixture must discover stems (it carries no session.json in a fresh clone)"
    stem = stems[0]
    out = brain.coedit([f"http://x/api/set_stem?stem={stem}&gain_db=-9.0"])
    assert out.get("applied") == 1
    assert "turn_delta" not in out, "a later user co-edit is not the agent's effect"
    assert B.turn_delta_part(proj) == ""
    # a pending delta must not follow the brain into another edit either
    brain._pending_turn_delta = {"before": {"lufs_i": -12.2}, "commands": ["x"]}
    brain.open_edit(str(proj / ".edits" / "e2"))
    assert brain._pending_turn_delta is None
    assert "chat_question" in inspect.getsource(chatmod.ChatSession._bind_turn_tools)


def test_a_created_return_is_named_in_its_reply_and_still_counts_as_a_write():
    """The fixture snapshot is static: without the id in return.add's reply the agent
    added a return it could not see and undid it to retry."""
    reply = O.CANNED_WRITE_REPLIES["return.add"]({"kind": "reverb"})
    assert reply["return_id"] == "rv"   # Session.add_return mints dly / cv / rv / fx
    assert "return.add" not in O.CANNED_READ_REPLIES, "a write must not become a read"


# ------------------------------------------------------------ in-turn routing ----

def test_the_brain_performs_routing_mid_turn_and_the_daw_only_realizes_it():
    """return.add / send.set inside a chat turn came back "routing is busy" — the DAW
    asked the brain, whose socket thread was running the turn. Now the brain does the
    op and the DAW gets `routing.apply` with the resolved result."""
    sent = []

    def fake_invoke(command, args):
        sent.append((command, args))
        return {"ok": True}

    ops = {"return.add": lambda a: {"return_id": "rv", "kind": a.get("kind"),
                                    "chains": {"big": True}, "readout": {"lufs_i": -14.0},
                                    "send_returns": {"returns": [{"id": "rv"}]}}}
    out = B.perform_routing("return.add", {"kind": "reverb"}, ops, fake_invoke)
    assert out == {"ok": True, "return_id": "rv", "kind": "reverb",
                   "send_returns": {"returns": [{"id": "rv"}]}}, out
    assert sent[0][0] == "routing.apply" and sent[0][1]["chains"] == {"big": True}
    assert sent[0][1]["command"] == "return.add"
    # not a routing key -> None, the caller sends it to the DAW as usual
    assert B.perform_routing("clip.split", {}, ops, fake_invoke) is None
    # a brain refusal reaches the model as the tool error, no realize request
    ops["send.set"] = lambda a: (_ for _ in ()).throw(ValueError("no such return"))
    out = B.perform_routing("send.set", {"source": "v", "return": "x"}, ops, fake_invoke)
    assert out["ok"] is False and "no such return" in out["error"] and len(sent) == 1


def _const_strings(code) -> set:
    """Every string constant in `code`, INCLUDING the ones nested in a constant key tuple or an
    inner code object. Reading only top-level `co_consts` is an interpreter-version claim, not a
    claim about the table: CPython 3.12 compiles this dict literal with BUILD_CONST_KEY_MAP, so
    the keys live in a tuple const and the bare-string check found none — red on CI's 3.12 and
    green on the dev venv's 3.14 for four months, unseen because engine CI never collected
    (2026-09-11)."""
    import types
    out: set = set()
    for c in code.co_consts:
        if isinstance(c, str):
            out.add(c)
        elif isinstance(c, tuple):
            out.update(x for x in c if isinstance(x, str))
        elif isinstance(c, types.CodeType):
            out |= _const_strings(c)
    return out


def test_routing_ops_cover_every_brain_realized_routing_key():
    keys = _const_strings(S.Brain.__dict__["routing_ops"].__code__)
    for k in ("return.add", "return.remove", "return.set_prefader", "send.set", "route.stem",
              "group.create", "bus.remove", "bus.route", "duck.set_key"):
        assert k in keys, k
    assert "routing_ops=self.routing_ops()" in inspect.getsource(S.Brain.chat)
    # the names the delta paths call must be IMPORTED, not just mentioned — a NameError in
    # a try/except printed "turn delta skipped" on every turn of the first live smoke
    for name in ("turn_delta", "write_turn_delta", "session_readout"):
        assert getattr(S, name) is getattr(B, name), name
    assert '"turn_delta"' in inspect.getsource(S.Brain.chat), "the proposal path settles its delta at chat_done"


# ------------------------------------------------------------ A/B preference -> taste ----

def test_ask_preference_offers_two_fixed_options_and_records_the_click_as_taste(tmp_path, monkeypatch):
    import anyio
    from orchestrator.calibration import load_decisions
    session = _session(monkeypatch)

    class _PrefSub:
        async def chat_turn(self, *, system, prompt, tool_defs, execute, on_delta, on_tool,
                            max_turns, effort=None, should_stop=None, max_tokens=None, timer=None):
            self.reply = (await execute("ask_preference", {"about": "the vocal reverb"})).text
            return "done", {}
    session._sub = _PrefSub()
    events, offered = [], []

    async def emit(e):
        events.append(e)

    async def hook(about, move):
        offered.append(about)

    anyio.run(lambda: session.turn("x", tools=_tools(ask_preference=hook), context="CTX", emit=emit))
    q = next(e for e in events if e["type"] == "chat_question")
    assert q["preference"] is True and q["options"] == list(chatmod.PREFERENCE_OPTIONS)
    assert offered == ["the vocal reverb"] and "END YOUR TURN" in chatmod._ASK_PREFERENCE_TOOL_DEF["description"]

    # the brain side: the offer is parked, the click is recorded, then consumed
    B.offer_preference(tmp_path, about="the vocal reverb",
                       move={"command": "track.set_gain", "alone": True})
    assert B.record_preference_answer(tmp_path, "something else") is None, "an ordinary message is not an answer"
    assert (tmp_path / "runs" / B.PENDING_PREFERENCE_FILE).is_file()
    out = B.record_preference_answer(tmp_path, chatmod.PREFERENCE_OPTIONS[1])
    assert out["choice"] == "B" and out["recorded"] >= 1
    rows = load_decisions(tmp_path / "runs")
    assert rows and rows[-1]["decision"] == "approved" and "track.set_gain" in rows[-1]["action_key"]
    assert not (tmp_path / "runs" / B.PENDING_PREFERENCE_FILE).exists(), "consumed once answered"
    assert B.record_preference_answer(tmp_path, chatmod.PREFERENCE_OPTIONS[0]) is None, "no pending offer"
    B.offer_preference(tmp_path, about="the drums level",
                       move={"command": "track.set_gain", "alone": True})
    out = B.record_preference_answer(tmp_path, chatmod.PREFERENCE_OPTIONS[0])
    assert out["choice"] == "A" and load_decisions(tmp_path / "runs")[-1]["decision"] == "rejected"


def test_the_daw_reply_co_edits_are_mirrored_and_measured_mid_turn():
    """The command reply carries the co-edits it queued; the brain mirrors them and
    measures before the next round — and a fully mirrored turn settles at chat_done."""
    from orchestrator import turn_host       # the turn's tools live there since Phase 55
    src = inspect.getsource(turn_host.LiveTurn)
    assert 'res.pop("coedits"' in src and "mirror_coedits" in src and '"chat_measured"' in src
    assert '"settle": "chat_done"' in src and '"settle": "coedit"' not in src
    assert "_mirror_coedits" in inspect.getsource(S.Brain.chat)
    assert 'pending.get("settle") == "chat_done"' in inspect.getsource(S.Brain.chat)
    assert "chat_measured" in inspect.getsource(S._do_chat)


def test_a_fader_only_turn_still_settles_its_delta_for_the_next_turn(tmp_path):
    """A daw_command-only turn records no gesture, so nothing re-rendered at chat_done;
    the mid-turn mirror left the readout in the cache and the settle must use it."""
    pending = {"before": {"lufs_i": -12.2, "true_peak_dbtp": 2.0}, "commands": ["track.set_gain"],
               "settle": "chat_done", "measured_midturn": True}
    out = B.settle_turn_delta(tmp_path, pending, {"lufs_i": -12.4, "true_peak_dbtp": -0.3})
    assert out is None, "the per-command row already said it — no second feed row"
    assert "true peak dBTP 2.0 -> -0.3" in B.turn_delta_part(tmp_path)
    pending["measured_midturn"] = False
    out = B.settle_turn_delta(tmp_path, pending, {"lufs_i": -12.4, "true_peak_dbtp": -0.3})
    assert out and out["changed"]
    assert B.settle_turn_delta(tmp_path, None, {}) is None
    assert S.settle_turn_delta is B.settle_turn_delta
    src = inspect.getsource(S.Brain.chat)
    assert "settle_turn_delta" in src and "session_readout(self.project_dir, self.render_cache)" in src
