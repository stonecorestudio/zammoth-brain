"""Talking to the engineer WHILE a run is in flight (2026-08-05).

The brain used to refuse `chat` outright during a run ("a run is already in
progress"), so "how's it going?" was unanswerable by construction. A question is
now a NOTIFY (`chat_async`) whose answer comes back as a uniquely-named event.

Three things here are load-bearing and each has a test:

* the reply event name must be UNIQUE — `BrainClient::waitForEvent` matches on
  event NAME ONLY, so a reply called `chat_done`/`run_failed` would satisfy the
  RunThread's parked wait and end the real run early;
* the status task must live OUTSIDE the `tasks` set — the run guards read
  `any(not t.done() for t in tasks)`, so parking one there refuses the user's
  next run, and `cancels` iteration would inflate Cancel's reported count;
* it must answer without rendering or touching session.json.
"""
import inspect
import json

from orchestrator import daw_bridge, daw_server


def test_the_reply_event_name_is_unique():
    """The whole design rests on this. `chat_done` and `run_failed` are what the
    DAW's RunThread is parked on."""
    src = inspect.getsource(daw_server._do_chat_status)
    assert '"event": "chat_async_done"' in src
    assert '"chat_done"' not in src and "'chat_done'" not in src
    assert '"run_failed"' not in src and "'run_failed'" not in src
    # ...including on the failure paths, which is where the mistake is easy: the licence
    # door (2026-09-22, now the task's first step), a failed answer, and the answer itself
    assert src.count('"event": "chat_async_done"') == 3


def test_the_status_task_is_not_in_the_run_task_set():
    src = inspect.getsource(daw_server._serve_client)
    branch = src[src.index('elif method == "chat_async"'):]
    branch = branch[:branch.index("else:")]
    # Comments here EXPLAIN why these are absent, so they must not be searched —
    # otherwise the test passes on the prose and never looks at the code.
    code = "\n".join(ln for ln in branch.splitlines()
                     if not ln.strip().startswith("#"))
    assert "status_tasks.add(st)" in code
    # `status_tasks.add` legitimately contains "tasks.add" — check the bare form.
    assert "  tasks.add(" not in code, "a status turn must not claim the pass slot"
    assert "cancels[" not in code, "a status turn must not register a cancel token"
    assert "run_active" not in code, "a status turn must not claim the pass slot"


def test_the_status_task_is_still_cancelled_on_teardown():
    """Outside `tasks` must not mean leaked — and the RUN tasks are deliberately NOT
    cancelled (2026-09-01 sweep, P2): cancelling a coroutine parked in to_thread does
    not stop its worker, but its done-callbacks fired at once and released
    brain.run_active under a thread still writing session.json."""
    src = inspect.getsource(daw_server._serve_client)
    assert "for task in status_tasks:" in src
    assert "for task in tasks | status_tasks:" not in src


def test_the_status_reply_never_renders_or_mutates():
    """`chat_reply` calls session_readout (a full master bounce) — mid-run that
    contends with the run's own renders. And touching ChatSession/begin_agent_turn
    would corrupt the in-flight gesture's undo + taste state."""
    # Strip the docstring: it EXPLAINS why these are absent, so a naive substring
    # search over the whole source matches the prose and never tests the code.
    src = inspect.getsource(daw_bridge.chat_status_reply)
    body = src[src.index('"""', src.index('"""') + 3) + 3:]
    for forbidden in ("session_readout", "_chat_session", "begin_agent_turn",
                      "mutate_lock", "in_memory_engine", "ChatTools"):
        assert forbidden not in body, f"the status turn must not touch {forbidden}"


def test_it_answers_instantly_when_signed_out():
    """A question must never hang; with no model there is nothing to wait for."""
    reply = daw_bridge.chat_status_reply("how's it going", {"kind": "run"})
    assert reply and "run is still going" in reply


def test_the_prompt_forbids_claiming_a_change(monkeypatch):
    """No tools are bound, so it CANNOT change anything — but it must also not say
    that it did."""
    seen = {}

    class FakeAgent:
        def chat(self, message, context=None):
            seen["context"] = context
            return "two passes in, tightening the low end"

    monkeypatch.setattr("orchestrator.llm_backend.llm_available", lambda: True)
    monkeypatch.setattr("orchestrator.agents.ClaudeAgent", FakeAgent)
    live = {"kind": "run", "brief": "warm master", "pass_number": 2,
            "max_passes": 5, "started_at": 0.0, "moves": [{"stem": "bass"}]}
    reply = daw_bridge.chat_status_reply("how's it going", live,
                                         readout={"lufs_i": -14.0})
    assert reply == "two passes in, tightening the low end"
    ctx = seen["context"]
    assert "NO TOOLS" in ctx and "cannot change anything" in ctx
    assert "never claim to have made one" in ctx
    assert "warm master" in ctx and "-14" in ctx


def test_a_run_that_already_ended_is_not_answered_as_if_it_were_running(monkeypatch):
    """`live_run` is cleared the instant a run finishes, so an empty one means the run
    ENDED between the question being sent and this turn running (2026-08-05 review).
    Answering anyway opened the prompt with "a mix run is in flight" and promised to act
    "once this pass finishes" — on a pass already over, in a state where the ordinary
    path could just have done it."""
    called = []

    class FakeAgent:
        def chat(self, message, context=None):
            called.append(context)
            return "should never be reached"

    monkeypatch.setattr("orchestrator.llm_backend.llm_available", lambda: True)
    monkeypatch.setattr("orchestrator.agents.ClaudeAgent", FakeAgent)

    for finished in (None, {}):
        reply = daw_bridge.chat_status_reply("make the vocal brighter", finished)
        assert "just finished" in reply and "ask me again" in reply.lower()
        assert "in flight" not in reply
    assert not called, "a finished run must cost no LLM turn at all"


def test_the_status_turn_names_the_right_brief_for_every_kind():
    """`chat` carries `message` and an approved plan carries `plan` — reading
    params["brief"] for those left the status turn describing the run it was asked
    about as having no brief at all."""
    src = inspect.getsource(daw_server._serve_client)
    assert '"kind": method, "run_id": run_id, "brief": text' in src, \
        "the chat handler must publish the message it was given"
    assert '"brief": "(approved plan)"' in src
    assert src.count('"brief": str(params.get("brief", ""))') == 3, \
        "only run + auto_mix + advise genuinely take a `brief` param"


def test_the_run_publishes_its_progress_for_the_answer():
    """The status turn reads live_run instead of the session, so the run has to
    keep it current."""
    src = inspect.getsource(daw_server._do_run)
    assert "brain.live_run" in src
    assert 'del live["moves"][:-24]' in src, "the move list must stay bounded"


def test_live_run_is_claimed_and_cleared_together_with_run_active():
    src = inspect.getsource(daw_server._serve_client)
    assert src.count('brain.live_run = {"kind"') == src.count("brain.run_active = True")
    assert src.count('setattr(b, "live_run", None)') == \
           src.count('setattr(b, "run_active", False)')


# ---------- end to end, over a real socket ----------

def test_a_question_lands_while_a_run_is_in_flight(tmp_path):
    """The whole point, exercised against a live daw_server: start a rule-agent run,
    ask a question mid-flight, and get `chat_async_done` back WITHOUT disturbing the
    run — which must still report `run_done`, and must not be refused afterwards."""
    import re
    import socket
    import subprocess
    import sys
    import numpy as np
    import soundfile as sf

    project = tmp_path / "midrun"
    project.mkdir()
    sr = 44100
    t = np.arange(sr) / sr
    sf.write(project / "a.wav", (0.2 * np.sin(2 * np.pi * 220 * t)).astype("float32"), sr)
    sf.write(project / "b.wav", (0.1 * np.sin(2 * np.pi * 440 * t)).astype("float32"), sr)

    repo = __import__("pathlib").Path(__file__).resolve().parents[2]
    proc = subprocess.Popen(
        [sys.executable, "-m", "orchestrator.daw_server", str(project)],
        cwd=str(repo), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        port = token = None
        for _ in range(400):
            line = proc.stdout.readline()
            if not line:
                break
            if "MIXAGENT_LISTENING port=" in line:
                # the handshake carries the per-spawn secret (2026-09-10 agent audit P0-2)
                m = re.search(r"port=(\d+) token=(\S+)", line)
                assert m, f"the handshake carries no token: {line!r}"
                port, token = int(m.group(1)), m.group(2)
                break
        assert port, "brain never announced a port"

        sock = socket.create_connection(("127.0.0.1", port), timeout=120)
        f = sock.makefile("rw")

        def send(obj):
            f.write(json.dumps(obj) + "\n")
            f.flush()

        send({"id": 0, "method": "auth", "params": {"token": token}})
        assert json.loads(f.readline()).get("result", {}).get("ok"), "the brain refused its own token"
        send({"id": 1, "method": "hello", "params": {"role": "daw"}})
        f.readline()
        send({"id": 2, "method": "run",
              "params": {"brief": "master to -14 LUFS", "agent": "rule"}})
        # the run's ack, then ask mid-flight
        send({"id": 3, "method": "chat_async",
              "params": {"chat_id": 77, "message": "how's it going?"}})

        answered = run_done = False
        for _ in range(600):
            line = f.readline()
            if not line:
                break
            msg = json.loads(line)
            if msg.get("event") == "chat_async_done":
                assert msg["chat_id"] == 77
                assert msg.get("reply") or msg.get("error")
                answered = True
            if msg.get("event") == "run_done":
                run_done = True
            if answered and run_done:
                break

        assert answered, "the mid-run question was never answered"
        assert run_done, "the run did not finish"

        # the status turn must not have claimed the pass slot
        send({"id": 4, "method": "run",
              "params": {"brief": "master to -14 LUFS", "agent": "rule"}})
        for _ in range(50):
            msg = json.loads(f.readline())
            if msg.get("id") == 4:
                assert "error" not in msg, f"the next run was refused: {msg}"
                break
    finally:
        proc.terminate()
        proc.wait(timeout=30)
