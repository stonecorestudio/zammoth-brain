"""The chat agent's `import_sample` tool: one implementation, correctly awaited.

Two defects here (2026-08-16 review), which had to be fixed together:

  * `daw_invoke` is SYNCHRONOUS everywhere in daw_bridge (daw_server hands over
    `counting_invoke`, a blocking call on the reply future). `do_import_sample`
    alone awaited it directly, so it raised "'dict' object can't be awaited" the
    moment a DAW was attached — i.e. always, in the real product.
  * It never checked the Ask-mode plan recorder that its five sibling mutators
    all check, so an approval-gated turn really imported the file while the turn
    narrated "recorded — will apply once you approve".

Fixing only the first would have ACTIVATED the second.
"""
from __future__ import annotations

import anyio
import pytest

from orchestrator import daw_bridge


def test_import_step_calls_a_synchronous_daw_invoke(tmp_path):
    """The DAW path must call daw_invoke, not await it."""
    wav = tmp_path / "loop.wav"
    wav.write_bytes(b"RIFF....WAVEfmt ")     # only has to exist
    seen: list[tuple] = []

    def sync_invoke(command, args, *a, **kw):   # exactly daw_server's shape
        seen.append((command, args))
        return {"ok": True, "track": "loop"}

    out = anyio.run(daw_bridge._import_sample_step, tmp_path, str(wav),
                    sync_invoke)
    assert out["ok"] is True
    # `register: false` — the brain registers the stem itself, in-turn (below)
    assert seen == [("track.add", {"path": str(wav), "register": False})]


def test_import_step_reports_a_missing_file(tmp_path):
    out = anyio.run(daw_bridge._import_sample_step, tmp_path,
                    str(tmp_path / "nope.wav"), None)
    assert out["ok"] is False and "no such file" in out["error"]


def test_ask_mode_records_the_import_instead_of_running_it():
    """The recorder step must be replayable: apply_plan dispatches on `kind`,
    so an unknown kind would be approved and then silently do nothing."""
    rec = daw_bridge._PlanRecorder()
    note = rec.record("import", {"path": "/tmp/loop.wav"}, "recorded")
    assert note == "recorded"
    assert rec.steps == [{"kind": "import", "payload": {"path": "/tmp/loop.wav"}}]
    # apply_plan has a branch for it (guards against record/replay drift)
    import inspect
    assert 'kind == "import"' in inspect.getsource(daw_bridge.apply_plan)


# ---- registration happens INSIDE the agent's turn (2026-09-04) ----
# James: "i asked agent to put a gate on the track and it said it did but the gate
# is not there." The DAW's background `register` cannot run while the turn holds
# the RunThread, so the agent's next call addressed a track the brain did not
# hold — or a deleted namesake's soft-excluded ghost stem, which propose accepted
# and the deferred registration's reset then wiped. The step now asks the DAW for
# `register: false`, registers the track it names before returning, and only then
# mirrors the co-edits the reply carried.

def _step(tmp_path, reply, register=None, mirror=None):
    wav = tmp_path / "loop.wav"
    wav.write_bytes(b"RIFF....WAVEfmt ")
    log: list = []

    def sync_invoke(command, args, *a, **kw):
        log.append(("invoke", command, dict(args)))
        return dict(reply)

    def register_stem(path, stem_id, bus):
        log.append(("register", path, stem_id, bus))
        return register if register is not None else {"ok": True, "stem_id": stem_id,
                                                      "wav": f"{stem_id}.wav"}

    def mirror_coedits(urls):
        log.append(("mirror", list(urls)))
        return {"ok": True}

    out = anyio.run(lambda: daw_bridge._import_sample_step(
        tmp_path, str(wav), sync_invoke, register_stem=register_stem,
        mirror_coedits=mirror_coedits if mirror is not False else None))
    return out, log, wav


def test_import_step_registers_the_track_before_returning(tmp_path):
    out, log, wav = _step(tmp_path, {"ok": True, "track": "loop", "registered": False,
                                     "coedits": ["/api/set_stem?stem=loop&gain_db=-3.00"]})
    assert out["ok"] is True and out["stem_id"] == "loop" and out["registered"] is True
    kinds = [e[0] for e in log]
    # invoke → register → mirror: the stem exists BEFORE the co-edits that name it land
    assert kinds == ["invoke", "register", "mirror"], log
    assert log[1] == ("register", str(wav), "loop", None)
    assert log[2] == ("mirror", ["/api/set_stem?stem=loop&gain_db=-3.00"])
    assert "coedits" not in out   # consumed, not handed back to the agent


def test_import_step_leaves_registration_to_a_daw_that_kept_it(tmp_path):
    """A fill/replace or a stacked clip: the DAW's own register action owns the
    group + copy_stem_state carry, and says `registered: true`. Registering here
    as well would register the same file twice (and reset the record)."""
    for reply in ({"ok": True, "track": "loop", "registered": True},
                  {"ok": True, "track": "loop", "registered": False, "stack": True},
                  {"ok": True}):   # an older DAW that says nothing keeps its own register
        out, log, _ = _step(tmp_path, reply)
        assert out["ok"] is True
        assert [e[0] for e in log] == ["invoke"], (reply, log)


def test_import_step_reports_a_failed_registration_by_track(tmp_path):
    out, log, _ = _step(tmp_path, {"ok": True, "track": "loop", "registered": False},
                        register={"ok": False, "error": "unsupported audio file: loop.wav"})
    assert out["ok"] is False and out["track"] == "loop"
    assert "loop" in out["error"] and "unsupported audio file" in out["error"]


def test_import_step_does_not_register_when_the_daw_refused(tmp_path):
    out, log, _ = _step(tmp_path, {"ok": False, "error": "couldn't add loop.wav"})
    assert out["ok"] is False and [e[0] for e in log] == ["invoke"]


def test_import_step_default_registrar_is_import_stem(demo_project):
    """Without daw_server's locked registrar (tests, a bare bridge) the step still
    registers — with the module's own import_stem — so a DAW reply handing the
    stem back never leaves it unregistered."""
    import json
    external = next(demo_project.glob("*.wav"))
    calls = []

    def sync_invoke(command, args, *a, **kw):
        calls.append(command)
        return {"ok": True, "track": "fresh lane", "registered": False}

    out = anyio.run(daw_bridge._import_sample_step, demo_project, str(external),
                    sync_invoke)
    assert out["ok"] is True and out["stem_id"] == "fresh lane"
    stems = json.loads((demo_project / "session.json").read_text())["stems"]
    assert "fresh lane" in stems and stems["fresh lane"]["excluded"] is False


def test_the_ghost_stem_scenario_end_to_end(demo_project):
    """The exact 2026-09-04 failure, against a real session: a track named X was
    deleted (its stem soft-excluded, chain kept for undo) and the agent re-adds
    the same file as X in one turn, then gates it.

    OLD order — gate the ghost, register later: the gate landed on the excluded
    record and the registration's reset erased it (the DAW never showed a gate).
    NEW order — register in-turn, then gate: the stem is live and keeps the gate.
    And the engine now refuses the ghost move outright, so even a user drop that
    lands mid-turn cannot be silently mixed into nothing."""
    import json
    from mix_engine.models import ProposalRequest
    from mix_engine.session import Session

    ghost = sorted(Session(demo_project).stems)[0]
    wav = demo_project / f"{ghost}.wav"
    assert wav.is_file()
    sess = Session(demo_project)
    sess.user_edit(ghost, {"excluded": True})     # the DAW's delete-track rule

    def gate(stem):
        return ProposalRequest(rationale="demo gate", predicted_impact=[], actions=[{
            "type": "gate", "stem_id": stem, "threshold_db": -24.0, "ratio": 8.0,
            "attack_ms": 1.0, "release_ms": 120.0, "hold_ms": 80.0, "range_db": -20.0}])

    # (defence) the ghost refuses a mix move…
    refused = Session(demo_project).propose(gate(ghost))
    assert not refused.accepted and refused.violations[0].rule == "excluded_stem"

    # …and the in-turn registration makes it a live stem the gate can land on
    def sync_invoke(command, args, *a, **kw):
        assert command == "track.add" and args["register"] is False
        return {"ok": True, "track": ghost, "registered": False}

    out = anyio.run(daw_bridge._import_sample_step, demo_project, str(wav), sync_invoke)
    assert out["ok"] is True and out["stem_id"] == ghost
    live = Session(demo_project)
    assert live.stems[ghost]["excluded"] is False and live.stems[ghost]["chain"] == []
    res = live.propose(gate(ghost))
    assert res.accepted, res.violations
    assert live.apply(res.proposal_id).ok
    chain = json.loads((demo_project / "session.json").read_text())["stems"][ghost]["chain"]
    assert [e["type"] for e in chain] == ["gate"] and chain[0]["threshold_db"] == -24.0


# --- the DAW's co-edit queue is DRAINED by the reply, so it must never be dropped ------------
# App.cpp SWAPS coeditQueue into every command reply: the queue is emptied whether or not the
# brain does anything with the batch, and flushCoedits' "a failed flush must not LOSE the moves"
# re-queue does not cover a reply-carried one. Both failure returns used to drop it, so a fader
# the user nudged just before a failed track.add was gone from session.json for ever
# (2026-09-05 review).

def _step_invoke(tmp_path, invoke, **kw):
    wav = tmp_path / "loop.wav"
    wav.write_bytes(b"RIFF....WAVEfmt ")
    return anyio.run(lambda: daw_bridge._import_sample_step(
        tmp_path, str(wav), invoke, **kw)), wav


def test_coedits_are_mirrored_when_the_daw_REFUSES_the_add(tmp_path):
    seen: list[list] = []
    out, _ = _step_invoke(tmp_path,
                   lambda c, a, *x, **k: {"ok": False, "error": "unreadable",
                                          "coedits": ["/api/set_stem?stem=bass&gain_db=-2"]},
                   mirror_coedits=lambda urls: seen.append(list(urls)))
    assert out["ok"] is False
    assert seen == [["/api/set_stem?stem=bass&gain_db=-2"]], "the user's move was dropped"
    assert out.pop("_coedits_mirrored") is True
    assert "coedits" not in out          # consumed, not handed to the model


def test_coedits_are_mirrored_when_REGISTRATION_fails(tmp_path):
    seen: list[list] = []
    out, _ = _step_invoke(tmp_path,
                   lambda c, a, *x, **k: {"ok": True, "track": "loop", "registered": False,
                                          "coedits": ["/api/set_stem?stem=bass&gain_db=-2"]},
                   register_stem=lambda *a, **k: {"ok": False, "error": "no decoder"},
                   mirror_coedits=lambda urls: seen.append(list(urls)))
    assert out["ok"] is False and "could not " in out["error"]
    assert seen == [["/api/set_stem?stem=bass&gain_db=-2"]], "the user's move was dropped"
    assert out.pop("_coedits_mirrored") is True


def test_a_raising_registrar_is_reported_not_propagated(tmp_path):
    """import_stem RAISES on a file that will not decode (only the .wav branch pre-validates
    with sf.info), so the carefully worded "the clip is on track X but the brain could not
    register it" message was never produced — and the co-edits went with the exception."""
    seen: list[list] = []

    def boom(*a, **k):
        raise RuntimeError("libsndfile: format not recognised")

    out, _ = _step_invoke(tmp_path,
                   lambda c, a, *x, **k: {"ok": True, "track": "loop", "registered": False,
                                          "coedits": ["/api/set_stem?stem=bass&gain_db=-2"]},
                   register_stem=boom,
                   mirror_coedits=lambda urls: seen.append(list(urls)))
    assert out["ok"] is False
    assert "could not" in out["error"] and "format not recognised" in out["error"]
    assert seen == [["/api/set_stem?stem=bass&gain_db=-2"]]


def test_a_failed_mirror_is_reported_as_NOT_mirrored(tmp_path):
    """do_daw_command recorded track.add as `mirrored` unconditionally, so a mirror that threw
    still settled the turn's delta against a brain that never received the moves."""
    def boom(urls):
        raise RuntimeError("engine busy")

    out, _ = _step_invoke(tmp_path,
                   lambda c, a, *x, **k: {"ok": True, "track": "loop",
                                          "coedits": ["/api/set_stem?stem=bass&gain_db=-2"]},
                   mirror_coedits=boom)
    assert out["ok"] is True                      # the add itself stands
    assert out.pop("_coedits_mirrored") is False  # …but the turn must not claim it synced


def test_no_coedits_is_not_a_mirror(tmp_path):
    out, _ = _step_invoke(tmp_path, lambda c, a, *x, **k: {"ok": True, "track": "loop"},
                   mirror_coedits=lambda urls: None)
    assert out["ok"] is True and out.pop("_coedits_mirrored") is False
