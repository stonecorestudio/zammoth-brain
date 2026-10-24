"""Phase 55, step 6: a print can be measured in the turn that made it.

`track.bounce` makes a new stem. The app registered it with the brain in the background, and
that cannot run while the agent's turn holds the app's run thread - so a turn that printed a
part and then asked for the print's numbers addressed a stem the brain did not hold (the
second showcase's rehearsal: "a print is measured in the turn after the one that made it").
The brain now asks the app for `register: false`, as it does for `track.add`,
`track.duplicate` and `library.import`, and registers the file the reply names before the
tool call returns. The app's half is the console leg `--bounce-register`."""
from __future__ import annotations

from pathlib import Path

import anyio

from orchestrator import turn_host

REPO = Path(__file__).resolve().parents[2]


def _bounce(tmp_path, reply, *, args=None, register_raises=None):
    log: list = []

    def daw_invoke(command, call_args, *a, **kw):
        log.append(("invoke", command, dict(call_args)))
        return dict(reply)

    def register_stem(path, stem_id, bus):
        log.append(("register", path, stem_id, bus))
        if register_raises:
            raise register_raises
        return {"ok": True, "stem_id": stem_id, "wav": f"{stem_id}.wav"}

    def mirror(urls):
        log.append(("mirror", list(urls)))

    async def go():
        return await turn_host.registering_command(
            tmp_path, "track.bounce", args if args is not None else {"track": "bass"},
            daw_invoke=daw_invoke, register_stem=register_stem, mirror_coedits=mirror)

    return anyio.run(go), log


def test_a_bounce_is_registered_before_the_tool_call_returns(tmp_path):
    printed = tmp_path / "bass bounce.wav"
    printed.write_bytes(b"RIFF....WAVEfmt ")
    out, log = _bounce(tmp_path, {"ok": True, "track": "bass bounce", "registered": False,
                                  "file": str(printed), "coedits": ["/set_stem?stem=bass&gain_db=-3"]})
    assert out is not None, "track.bounce went the plain way: nothing registers its print in the turn"
    assert log[0] == ("invoke", "track.bounce", {"track": "bass", "register": False})
    assert ("register", str(printed), "bass bounce", None) in log
    # the stem exists before the moves the reply carried are mirrored onto the session
    assert [e[0] for e in log] == ["invoke", "register", "mirror"]
    assert out["ok"] is True and out["registered"] is True and out["stem_id"] == "bass bounce"
    assert "_coedits_mirrored" not in out and "coedits" not in out


def test_the_agents_own_arguments_reach_the_app_and_register_is_the_brains(tmp_path):
    printed = tmp_path / "p.wav"
    printed.write_bytes(b"RIFF")
    _, log = _bounce(tmp_path, {"ok": True, "track": "p", "registered": False, "file": str(printed)},
                     args={"track": "bass", "start_s": 4.0, "end_s": 12.0, "register": True})
    assert log[0] == ("invoke", "track.bounce",
                      {"track": "bass", "start_s": 4.0, "end_s": 12.0, "register": False})


def test_a_refused_bounce_registers_nothing_and_still_mirrors(tmp_path):
    out, log = _bounce(tmp_path, {"ok": False, "error": "nothing to bounce",
                                  "coedits": ["/set_stem?stem=bass&muted=1"]})
    assert out["ok"] is False and out["error"] == "nothing to bounce"
    assert [e[0] for e in log] == ["invoke", "mirror"]


def test_an_app_that_kept_the_registration_is_not_registered_twice(tmp_path):
    """An app build from before this change answers with no `registered` field: it queued its
    own registration, and the brain must not make a second stem of the same file."""
    out, log = _bounce(tmp_path, {"ok": True, "track": "bass bounce"})
    assert out["ok"] is True and not any(e[0] == "register" for e in log)


def test_a_print_the_brain_cannot_register_is_said(tmp_path):
    printed = tmp_path / "p.wav"
    printed.write_bytes(b"RIFF")
    out, _ = _bounce(tmp_path, {"ok": True, "track": "p", "registered": False, "file": str(printed)},
                     register_raises=RuntimeError("could not decode"))
    assert out["ok"] is False and out["track"] == "p"
    assert "could not register it as a stem" in out["error"] and "could not decode" in out["error"]


def test_the_other_three_doors_are_as_they_were(tmp_path):
    seen: list = []

    def daw_invoke(command, call_args, *a, **kw):
        seen.append((command, dict(call_args)))
        return {"ok": True, "track": "x", "registered": True}

    async def go(command, args):
        return await turn_host.registering_command(tmp_path, command, args, daw_invoke=daw_invoke,
                                                   register_stem=None, mirror_coedits=None)

    assert anyio.run(go, "library.import", {"item": "Kick"})["ok"] is True
    assert seen[-1] == ("library.import", {"item": "Kick", "register": False})
    assert anyio.run(go, "track.set_gain", {"track": "x", "gain_db": -3}) is None
    assert anyio.run(go, "library.import", {"item": "Kick", "track": "x", "scene": 0}) is None


def test_the_app_hands_the_print_back_and_a_leg_drives_it():
    app = (REPO / "daw" / "Source" / "App.cpp").read_text()
    handler = app.split("case dawcmd::bounceToNewTrack:")[1].split("case dawcmd::commandPalette:")[0]
    assert 'args.getProperty ("register", true)' in handler
    assert 'o->setProperty ("file", printed.getFullPathName())' in handler
    assert "if (printed.existsAsFile() && ! handBack)" in handler
    legs = (REPO / "daw" / "Source" / "AuditReaderP2Legs.h").read_text()
    assert "int runBounceRegister" in legs and '{ "register", false }' in legs
    assert 'flag == "--bounce-register"' in (REPO / "daw" / "Source" / "Main.cpp").read_text()
    assert "--bounce-register" in (REPO / ".claude" / "skills" / "daw" / "SKILL.md").read_text()


# ---------------------------------------------------------------- a print is not a reason to render

def test_printing_several_tracks_does_not_render_the_mix_before_each(demo_project, tmp_path,
                                                                    monkeypatch):
    """The timed run-through, 2026-10-12: eight bounces took 114 s where they had taken 19.
    Before a command that renders, the brain hands the app this gesture's mix if the app's is
    out of date - and a print registered in the turn made it look out of date, so every
    bounce after the first waited for a full render of the mix and a chain push. The app made
    that track itself and holds every chain there is; a one-track render needs nothing more.
    The EXPORT still gets its push: the master's limiter makeup follows from the whole mix."""
    import shutil

    from tests.test_midturn_export import _DawEnd, _Scripted, _export
    from mix_engine.session import Session
    from orchestrator import daw_bridge as B
    from orchestrator import daw_server as S

    Session(demo_project)
    brain = S.Brain(demo_project)
    stem_wav = next(p for p in sorted(demo_project.glob("*.wav")))

    class _PrintingDaw(_DawEnd):
        prints = 0

        def __call__(self, command, args=None, *a, **k):
            if command == "track.bounce" and isinstance(args, dict) and args.get("register") is False:
                self.log.append((command, {"asked": dict(args)}))
                type(self).prints += 1
                printed = tmp_path / f"print {self.prints}.wav"
                shutil.copyfile(stem_wav, printed)
                return {"ok": True, "track": f"print {self.prints}", "registered": False,
                        "file": str(printed)}
            return super().__call__(command, args, *a, **k)

    end = _PrintingDaw(brain)
    brain.daw_invoke = end
    bounce = lambda track: ("daw_command", {"command": "track.bounce", "args": {"track": track}})  # noqa: E731
    script = [bounce("stem_a"), bounce("stem_b"), bounce("stem_a"), _export(tmp_path)]
    monkeypatch.setitem(B._CHAT_SESSIONS, B._session_key(demo_project, ""), _Scripted(script))
    brain.chat("print three, then export", mode="agent", use_llm=True)

    pushes = [held["sync"] for c, held in end.log if c == "routing.apply" and held]
    assert len(end.seen("track.bounce")) == 3, end.log
    assert {"print 1", "print 2", "print 3"} <= set(Session(demo_project, read_only=True)._state["stems"])
    assert "track.bounce" not in pushes, f"a print made the brain re-render the mix before a bounce: {pushes}"
    assert pushes == ["session.export"], pushes       # the control: the export is still handed the mix
