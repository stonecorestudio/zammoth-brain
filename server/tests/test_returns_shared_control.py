"""Sends/returns shared control + awareness (DAW parity, docs/10-daw.md).

Keeps the proven per-track-wet realization (send_return_entries) and adds:
- analyze_return: the agent's ears on a send/return bus (every active source's
  send summed through the return's wet chain), mirroring analyze_bus.
- Session.set_send: a DIRECT co-edit (no proposal lifecycle) so the user + the
  agent both dial a send level live — the dry/wet balance is shared control.
"""
import numpy as np
import pytest
import soundfile as sf
from mcp.shared.memory import create_connected_server_and_client_session

from mix_engine import dsp
from mix_engine.mcp_server import create_server
from mix_engine.session import Session

from test_phase1_acceptance import call

SR = 48000


def engine(project):
    return create_connected_server_and_client_session(create_server(project))


async def submit(client, actions: list[dict], rationale: str = "returns") -> str:
    proposal = await call(client, "propose_actions", {"req": {
        "rationale": rationale, "predicted_impact": [], "actions": actions,
    }})
    assert proposal["accepted"], proposal["violations"]
    pid = proposal["proposal_id"]
    assert (await call(client, "apply_proposal", {"proposal_id": pid}))["ok"]
    return pid


def _write(project, name: str, audio: np.ndarray) -> None:
    sf.write(project / f"{name}.wav", audio.astype(np.float32), SR)


@pytest.mark.anyio
async def test_analyze_return_hears_the_wet(tmp_path):
    project = tmp_path / "ret-analyze"
    project.mkdir()
    t = np.arange(int(1.0 * SR)) / SR
    _write(project, "vox", 0.3 * np.sin(2 * np.pi * 440.0 * t))

    async with engine(project) as client:
        await submit(client, [
            {"type": "return_reverb", "return_id": "room",
             "decay_s": 1.2, "pre_delay_ms": 10.0, "tone": 0.5},
            {"type": "send", "stem_id": "vox", "return_id": "room", "gain_db": -6.0},
        ], "send vox to room")

        ret = await call(client, "analyze_return", {"return_id": "room"})
        assert ret["stem_id"] == "room"
        assert ret["lufs_s"] is not None                              # the wet is audible
        assert any(e["type"] == "return_reverb" for e in ret["applied_chain"])

        # an unknown return is an error, not a silent empty analysis
        with pytest.raises(Exception):
            await call(client, "analyze_return", {"return_id": "ghost"})


@pytest.mark.anyio
async def test_set_send_shared_control_round_trip(tmp_path):
    project = tmp_path / "ret-setsend"
    project.mkdir()
    t = np.arange(int(1.0 * SR)) / SR
    _write(project, "vox", 0.3 * np.sin(2 * np.pi * 440.0 * t))

    # the agent creates the return (reverb/delay stay agent-owned)…
    async with engine(project) as client:
        await submit(client, [
            {"type": "return_reverb", "return_id": "room",
             "decay_s": 1.0, "pre_delay_ms": 8.0, "tone": 0.5},
        ], "create a room return")
    dry, _ = dsp.render_master(Session(project))            # return exists, no send yet

    # …the user (or agent) dials the send live — direct, no proposal lifecycle.
    res = Session(project).set_send("vox", "room", -6.0)
    assert res == {"source": "vox", "return": "room", "gain_db": -6.0}
    assert Session(project).stems["vox"]["sends"]["room"] == -6.0

    wet, _ = dsp.render_master(Session(project))
    assert not np.array_equal(wet, dry)                     # the send is now in the mix

    # a cut to ≤ -60 dB disconnects the send → back to the dry render, bit-identical
    Session(project).set_send("vox", "room", -60.0)
    assert "room" not in Session(project).stems["vox"].get("sends", {})
    cut, _ = dsp.render_master(Session(project))
    assert np.array_equal(cut, dry)

    # guards: unknown return / unknown source are rejected (never a silent no-op)
    with pytest.raises(ValueError):
        Session(project).set_send("vox", "ghost", -6.0)
    with pytest.raises(ValueError):
        Session(project).set_send("ghost", "room", -6.0)


def test_add_return_caps_at_twelve(tmp_path):
    """Ableton parity: at most 12 send/return tracks. The first 12 add_return calls succeed; the 13th is
    refused with an error (audio/MIDI tracks themselves stay unlimited — this cap is returns-only)."""
    from mix_engine.session import MAX_RETURNS
    assert MAX_RETURNS == 12
    project = tmp_path / "ret-cap"
    project.mkdir()
    t = np.arange(int(0.5 * SR)) / SR
    _write(project, "vox", 0.3 * np.sin(2 * np.pi * 440.0 * t))

    s = Session(project)
    for _ in range(MAX_RETURNS):
        assert s.add_return()["ok"] is True
    over = s.add_return()                              # the 13th
    assert over.get("ok") is False
    assert "limit" in over.get("error", "").lower()
    assert len(s.returns) == MAX_RETURNS              # not created


def test_remove_return_clears_it_and_its_sends(tmp_path):
    """Session.remove_return (the inverse of add_return): deletes the return AND clears any sends to it,
    so no dangling sends remain (the DAW's realizeReturns then drops the orphaned return:<id> track)."""
    project = tmp_path / "ret-remove"
    project.mkdir()
    t = np.arange(int(1.0 * SR)) / SR
    _write(project, "vox", 0.3 * np.sin(2 * np.pi * 440.0 * t))

    s = Session(project)
    rid = s.add_return("reverb")["return_id"]
    assert "stem_id" not in s.returns[rid]["chain"][0]
    assert "return_id" not in s.returns[rid]["chain"][0]
    s.set_send("vox", rid, -6.0)
    assert rid in Session(project).returns
    assert Session(project).stems["vox"]["sends"].get(rid) == -6.0

    res = Session(project).remove_return(rid)
    assert res == {"ok": True, "return_id": rid}
    after = Session(project)
    assert rid not in after.returns                              # the return is gone
    assert rid not in (after.stems["vox"].get("sends") or {})    # and the send to it is cleared

    with pytest.raises(ValueError):                              # unknown return → error, not a no-op
        Session(project).remove_return("ghost")


@pytest.mark.anyio
async def test_chains_with_readout_surfaces_returns_and_sends(tmp_path):
    """The DAW's send UI reads the returns + per-source send levels the brain surfaces in
    chains_with_readout (returns live in session-state, not the Tracktion edit)."""
    from orchestrator.daw_bridge import chains_with_readout

    project = tmp_path / "ret-topo"
    project.mkdir()
    t = np.arange(int(1.0 * SR)) / SR
    _write(project, "vox", 0.3 * np.sin(2 * np.pi * 440.0 * t))
    async with engine(project) as client:
        await submit(client, [
            {"type": "return_reverb", "return_id": "room",
             "decay_s": 1.0, "pre_delay_ms": 8.0, "tone": 0.5},
            {"type": "send", "stem_id": "vox", "return_id": "room", "gain_db": -8.0},
        ], "send vox to room")

    sr = chains_with_readout(project)["send_returns"]
    assert [r["id"] for r in sr["returns"]] == ["room"]
    assert sr["returns"][0]["kind"] == "return_reverb"
    assert sr["sends"]["vox"] == {"room": -8.0}


# --- Blank-slate FX returns: a return is a bus fed by sends, with a FREE device chain ---

def _rms_d(a) -> float:
    return float(np.sqrt(np.mean(np.square(a))))


def test_add_return_is_blank_then_takes_any_device(tmp_path):
    """A new return is BLANK (no forced reverb/delay); the user adds ANY effect to it freely, in any
    order (eq → reverb → comp), via add_stem_device on the return:<id> target — like a stem/bus."""
    project = tmp_path / "ret-blank"
    project.mkdir()
    t = np.arange(int(0.5 * SR)) / SR
    _write(project, "vox", 0.3 * np.sin(2 * np.pi * 440.0 * t))

    s = Session(project)
    res = s.add_return()                       # blank by default
    rid = res["return_id"]
    assert res["kind"] == "blank"
    assert Session(project).returns[rid]["chain"] == []     # truly empty
    Session(project).set_send("vox", rid, 0.0)

    for dev in ("eq", "return_reverb", "comp"):             # free order, full palette
        Session(project).add_stem_device(f"return:{rid}", dev)
    chain = Session(project).returns[rid]["chain"]
    assert [e["type"] for e in chain] == ["eq", "return_reverb", "comp"]   # LIST order

    from mix_engine import dsp
    wet, _ = dsp.render_return(Session(project), rid)
    assert _rms_d(wet) > 1e-4                                # the free chain renders

    # remove + reorder work on the return too
    Session(project).remove_stem_device(f"return:{rid}", "eq", 0)
    Session(project).set_stem_device_order(f"return:{rid}", ["comp", "return_reverb"])
    assert [e["type"] for e in Session(project).returns[rid]["chain"]] == ["comp", "return_reverb"]


def test_comp_only_return_is_parallel_compression(tmp_path):
    """A return with NO wet effect (just a compressor) passes its PROCESSED input — parallel/NY
    compression. (The old model returned silence unless a reverb/delay was present.)"""
    project = tmp_path / "ret-parallel"
    project.mkdir()
    t = np.arange(int(0.5 * SR)) / SR
    _write(project, "drums", 0.4 * np.sin(2 * np.pi * 110.0 * t))

    s = Session(project)
    rid = s.add_return()["return_id"]
    Session(project).set_send("drums", rid, 0.0)
    Session(project).add_stem_device(f"return:{rid}", "comp")

    from mix_engine import dsp
    wet, _ = dsp.render_return(Session(project), rid)
    assert _rms_d(wet) > 1e-3                                # the compressed input is audible (not silence)


# --- Ableton-style rebuild (2026-06): pre/post tap, cascading, persisted return level ---

def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(a))))


def test_set_return_level_and_prefader_persist(tmp_path):
    """set_return_level + set_return_prefader are direct co-edits (shared control): they persist,
    clamp the level like a channel fader, and reject an unknown return — never a silent no-op."""
    project = tmp_path / "ret-level"
    project.mkdir()
    t = np.arange(int(0.5 * SR)) / SR
    _write(project, "vox", 0.3 * np.sin(2 * np.pi * 440.0 * t))

    rid = Session(project).add_return("reverb")["return_id"]
    assert Session(project).set_return_level(rid, 3.0) == {"return": rid, "gain_db": 3.0}
    assert Session(project).returns[rid]["gain_db"] == 3.0
    assert Session(project).set_return_level(rid, 99.0)["gain_db"] == 6.0      # clamp high (+6)
    assert Session(project).set_return_level(rid, -99.0)["gain_db"] == -60.0   # clamp low
    with pytest.raises(ValueError):
        Session(project).set_return_level("ghost", 0.0)

    assert Session(project).set_return_prefader(rid, True) == {"return": rid, "pre_fader": True}
    assert Session(project).returns[rid]["pre_fader"] is True
    assert Session(project).set_return_prefader(rid, False)["pre_fader"] is False
    with pytest.raises(ValueError):
        Session(project).set_return_prefader("ghost", True)


def test_prefader_send_is_fader_independent(tmp_path):
    """A POST-fader send scales with the source's fader; a PRE-fader send taps the dry pre-channel
    source, independent of the fader (Option A). render_return reflects the tap point."""
    project = tmp_path / "ret-prefader"
    project.mkdir()
    t = np.arange(int(0.5 * SR)) / SR
    _write(project, "vox", 0.3 * np.sin(2 * np.pi * 440.0 * t))

    rid = Session(project).add_return("reverb")["return_id"]
    Session(project).set_send("vox", rid, 0.0)                  # full send into the return
    base = _rms(dsp.render_return(Session(project), rid)[0])    # source fader at 0, post (default)

    Session(project).user_edit("vox", {"gain_db": -40.0})       # pull the source fader right down
    post = _rms(dsp.render_return(Session(project), rid)[0])    # post: follows the fader → tiny
    assert post < 0.1 * base

    Session(project).set_return_prefader(rid, True)
    pre = _rms(dsp.render_return(Session(project), rid)[0])     # pre: ignores the fader → ~ base
    assert pre == pytest.approx(base, rel=0.02)


def test_cascading_send_forward_only(tmp_path):
    """A return may send to another return (cascading), but only FORWARD — to a return created
    after it — so it can't feed back. The cascade is audible in the master deliverable."""
    project = tmp_path / "ret-cascade"
    project.mkdir()
    t = np.arange(int(0.5 * SR)) / SR
    _write(project, "vox", 0.3 * np.sin(2 * np.pi * 440.0 * t))

    s = Session(project)
    a = s.add_return("reverb")["return_id"]     # created first
    b = s.add_return("delay")["return_id"]      # created second (later)

    res = Session(project).set_send(f"return:{a}", b, -6.0)   # forward a -> b: accepted
    assert res["source"] == a and res["return"] == b
    assert Session(project).returns[a]["sends"][b] == -6.0
    with pytest.raises(ValueError):                          # backward b -> a would feed back
        Session(project).set_send(f"return:{b}", a, -6.0)
    with pytest.raises(ValueError):                          # a return can't send to itself
        Session(project).set_send(f"return:{a}", a, -6.0)

    Session(project).set_send("vox", a, -3.0)               # feed a so it + the cascade are audible
    casc, _ = dsp.render_master(Session(project))
    Session(project).set_send(f"return:{a}", b, -60.0)      # cut the cascade → b loses its only input
    nocasc, _ = dsp.render_master(Session(project))
    assert not np.array_equal(casc, nocasc)                 # the cascade (a -> b's wet) was in the mix


def test_topology_carries_prefader_level_and_cascade(tmp_path):
    """returns_topology surfaces each return's pre_fader + gain_db and a cascading return source
    (keyed "return:<id>") so the DAW + the agent see the full send graph."""
    from orchestrator.daw_bridge import chains_with_readout

    project = tmp_path / "ret-topo2"
    project.mkdir()
    t = np.arange(int(0.5 * SR)) / SR
    _write(project, "vox", 0.3 * np.sin(2 * np.pi * 440.0 * t))

    s = Session(project)
    a = s.add_return("reverb")["return_id"]
    b = s.add_return("delay")["return_id"]
    Session(project).set_send("vox", a, -8.0)
    Session(project).set_send(f"return:{a}", b, -10.0)   # cascade a -> b
    Session(project).set_return_prefader(a, True)
    Session(project).set_return_level(a, -3.0)

    sr = chains_with_readout(project)["send_returns"]
    rmap = {r["id"]: r for r in sr["returns"]}
    assert rmap[a]["pre_fader"] is True
    assert rmap[a]["gain_db"] == -3.0
    assert rmap[b]["pre_fader"] is False
    assert sr["sends"]["vox"][a] == -8.0
    assert sr["sends"][f"return:{a}"][b] == -10.0


def test_rename_return_carries_every_reference_with_it(tmp_path):
    """A return's id IS its displayed name — the DAW paints it on the return strip, the pinned
    head card and every send knob's chip — so renaming one has to MOVE the id, and the id is a
    reference. Every sender's `sends` dict keys it (stems, buses and earlier returns alike), the
    proposal ledger names it as `return_id`, a device on the return names it as `stem_id` =
    `return:<id>`, the inverse snapshot is keyed by that same decorated form, and a journaled
    co-edit carries it BOTH ways (`stem`/`source` decorated, `return` bare). A reference left
    behind is not a stale label: it is a send that has silently stopped feeding the return.
    (rename_bus's twin — 2026-09-07.)"""
    from mix_engine.models import ProposalRequest

    project = tmp_path / "ret-rename"
    project.mkdir()
    t = np.arange(int(1.0 * SR)) / SR
    _write(project, "vox", 0.3 * np.sin(2 * np.pi * 440.0 * t))
    _write(project, "drums", 0.3 * np.sin(2 * np.pi * 110.0 * t))

    s = Session(project)
    rv = s.add_return("reverb")["return_id"]            # first…
    dly = s.add_return("delay")["return_id"]            # …then a later one (the cascade order)
    s = Session(project)
    s.group_stems(["vox", "drums"])
    s = Session(project)
    bus = next(iter(s.buses))
    s.set_send("vox", rv, -6.0)
    s.set_send("drums", rv, -12.0)
    s.set_send(bus, rv, -9.0)                            # a BUS sends to it too
    s.set_send(f"return:{rv}", dly, -8.0)                # …and it cascades into the later return
    s.set_return_level(rv, -2.0)
    s.set_return_prefader(rv, True)
    s = Session(project)
    prop = s.propose(ProposalRequest(
        rationale="space",
        predicted_impact=[{"metric": "master.lufs_i", "predicted_delta": 0.2, "unit": "LUFS"}],
        actions=[{"type": "send", "stem_id": "vox", "return_id": rv, "gain_db": -6.0},
                 {"type": "return_reverb", "return_id": rv, "decay_s": 2.4,
                  "pre_delay_ms": 12.0, "tone": 0.4},
                 {"type": "return_delay", "return_id": "hall", "delay_ms": 300.0,
                  "feedback": 0.3, "mix": 0.5}]))
    assert prop.accepted, prop.violations
    assert s.apply(prop.proposal_id).ok
    s = Session(project)
    s.add_stem_device(f"return:{rv}", "eq")              # a USER co-edit device on the return
    s = Session(project)
    s._state.setdefault("coedit_log", []).extend([
        {"url": f"http://127.0.0.1:1/api/set_send?source=vox&return={rv}&gain_db=-6"},
        {"url": f"http://127.0.0.1:1/api/set_return_level?return={rv}&gain_db=-2"},
        {"url": f"http://127.0.0.1:1/api/add_device?stem=return:{rv}&type=eq"},
    ])
    s._persist()

    order_before = list(Session(project).returns)
    res = Session(project).rename_return(rv, "Plate")
    assert res["renamed"] and res["return_id"] == "Plate" and res["was"] == rv

    s2 = Session(project)
    assert "Plate" in s2.returns and rv not in s2.returns
    assert s2.returns["Plate"]["gain_db"] == -2.0 and s2.returns["Plate"]["pre_fader"] is True
    assert [e["type"] for e in s2.returns["Plate"]["chain"]] == ["return_reverb", "eq"]
    assert s2.returns["Plate"]["chain"][0]["decay_s"] == 2.4, "the return kept its own chain state"
    # every sender follows — a stem, a group bus, and the return's own cascade
    assert s2.stems["vox"]["sends"] == {"Plate": -6.0}
    assert s2.stems["drums"]["sends"] == {"Plate": -12.0}
    assert s2.buses[bus]["sends"] == {"Plate": -9.0}
    assert s2.returns["Plate"]["sends"] == {dly: -8.0}
    # …and the return keeps its PLACE. This one is load-bearing, not cosmetic: the cascade rule
    # ("a return may only send to a LATER return") is an index comparison over the insertion
    # order, so a rename that popped and re-added would move Plate PAST dly and silently
    # legalise the feedback path set_send refuses below.
    assert [("Plate" if r == rv else r) for r in order_before] == list(s2.returns)
    with pytest.raises(ValueError):
        s2.set_send(f"return:{dly}", "Plate", -8.0)          # still backward → still refused
    assert Session(project).set_send("return:Plate", dly, -9.0)["gain_db"] == -9.0

    record = s2._state["proposals"][0]
    assert "return:Plate" in record["inverse"] and f"return:{rv}" not in record["inverse"]
    assert {a.get("return_id") for a in record["actions"]} == {"Plate", "hall"}
    # …and a return the LEDGER created: undo deletes those by id, so a rename left behind here
    # makes the undo miss the return it minted and leave it in the mix for ever.
    assert record["created_returns"] == ["hall"]
    assert Session(project).rename_return("hall", "Tail")["renamed"]
    assert Session(project)._state["proposals"][0]["created_returns"] == ["Tail"]
    urls = [e["url"] for e in s2._state["coedit_log"]]
    assert all(f"return%3A{rv}" not in u and f"return={rv}" not in u for u in urls), urls
    assert sum("Plate" in u for u in urls) == 3, urls   # source=… / return=… / stem=return:… all moved

    # the namespace is shared with stems/buses/the master, and ':' is the prefix separator
    for taken in ("master", "vox", bus, dly, "half:baked"):
        with pytest.raises(ValueError):
            Session(project).rename_return("Plate", taken)
    with pytest.raises(ValueError):
        Session(project).rename_return("ghost", "Whatever")
    # a no-op rename is not an error, and the DAW may hand either spelling of the id
    assert Session(project).rename_return("return:Plate", "Plate")["renamed"] is False
    assert Session(project).rename_return("return:Plate", "return:Hall")["renamed"] is True
    assert "Hall" in Session(project).returns


def test_rename_return_moves_the_id_inside_inverse_snapshots(tmp_path):
    """A proposal's inverse snapshot is a DEEP COPY of each holder, so it carries that holder's
    `sends` dict keyed by return id. `rename_return` re-keyed only the snapshot's OUTER key, so
    undoing a proposal restored a send pointing at a return that no longer exists — and
    `dsp.add_sends` skips an unknown id WITHOUT error, so the send went silent with no message.
    `rename_stem` walks `inverse.values()` for exactly this reason (2026-09-08 pre-merge review).
    """
    from mix_engine.models import ProposalRequest
    project = tmp_path / "ret-rename-undo"
    project.mkdir()
    t = np.arange(int(0.5 * SR)) / SR
    _write(project, "vox", 0.3 * np.sin(2 * np.pi * 440.0 * t))

    s = Session(project)
    added = s.add_return("reverb")
    assert added["ok"], added
    ret = added["return_id"]
    s.set_send("vox", ret, -12.0)

    prop = s.propose(ProposalRequest(
        rationale="touch the stem so its sends are snapshotted",
        predicted_impact=[{"metric": "master.lufs_i", "predicted_delta": 0.2, "unit": "LUFS"}],
        actions=[{"type": "gain", "stem_id": "vox", "delta_db": -1.0}]))
    assert prop.accepted and s.apply(prop.proposal_id).ok

    snap = s._state["proposals"][-1].get("inverse") or {}
    assert any(isinstance(v, dict) and ret in (v.get("sends") or {}) for v in snap.values()), \
        "the fixture never snapshotted a send — this test would prove nothing"

    assert s.rename_return(ret, "verb")["renamed"]
    assert "verb" in (s._state["stems"]["vox"]["sends"] or {})

    assert s.undo(prop.proposal_id).ok
    restored = s._state["stems"]["vox"]["sends"] or {}
    assert ret not in restored, \
        "undo restored the OLD return id — dsp.add_sends drops it silently and the send is dead"
    assert restored.get("verb") == -12.0, restored
    assert "verb" in s._state["returns"]
