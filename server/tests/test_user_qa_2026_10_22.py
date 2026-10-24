"""Real-use QA, 2026-10-22 (docs/audits/user-qa-2026-10-22/QA-log.md).

Each test was written from a defect seen in the running app and seen red on the code that
had it, before the fix.
"""
from pathlib import Path

from orchestrator import automix, daw_bridge


def _flat(stems):
    return {sid: {"lufs_s": -20.0, "band_energy": {"sub": -30.0},
                  "dynamic_spread_db": 5.0} for sid in stems}


def test_auto_mix_keeps_a_track_in_the_group_the_user_put_it_in():
    """Q24: the user had grouped Drums, Bass and Music as Rhythm; Auto Mix laid its own
    "drums" family group and moved Drums out of Rhythm into it. A track that is already in a
    group stays there: the families are made of the loose tracks only."""
    tags = {"kik": ["kick"], "hat": ["cymbal"], "lead": ["vocal"], "bv": ["vocal"]}
    bundles = automix.build_bundles(tags, _flat(tags), [], {"lufs_i": -18.0},
                                    goal_lufs=None, grouped={"kik"})
    groups = {a["bus_id"]: a["members"] for b in bundles for a in b["actions"]
              if a["type"] == "group"}
    assert all("kik" not in m for m in groups.values()), groups
    assert "drums" not in groups, "a family of one loose track is no group"
    assert groups.get("vox") == ["bv", "lead"], "the loose families are still grouped"


def test_auto_mix_leaves_a_muted_track_alone(demo_project: Path):
    """Q25: Auto Mix put an EQ on the muted source of a split, and phrase rides and two
    de-essers on a muted print - moves on tracks that do not play, and their levels in the
    pack it balances against. A muted track is left as it is, and the reply says so."""
    from mix_engine.session import Session

    Session(demo_project).user_edit("stem_b", {"muted": True})
    notes: list = []
    daw_bridge.auto_mix_pass(demo_project, "master for streaming", notes=notes)
    applied = daw_bridge.applied_proposals_since(demo_project, 0)
    targets = {a.get("stem_id") for r in applied for a in (r.get("actions") or [])}
    members = {m for r in applied for a in (r.get("actions") or [])
               for m in (a.get("members") or [])}
    assert "stem_b" not in targets | members, targets | members
    assert any("stem_b" in n and "muted" in n for n in notes), notes


def test_the_daw_prompt_does_not_say_measuring_a_reference_loads_it():
    """Q27: asked to load Future.wav as the reference, the engineer ran `analyze_reference`
    and said "Future is now your reference"; the project held no reference. The shared base
    prompt says that tool "doubles as setting the workspace's reference track" - true in the
    old gateway, false in the DAW, where `agent.reference` loads one. Both DAW prompts must say
    which does what."""
    from orchestrator import chat, prompts

    for name, text in (("DAW_SYSTEM_PROMPT", chat.DAW_SYSTEM_PROMPT),
                       ("DAW_PROMPT_WHOLE", prompts.DAW_PROMPT_WHOLE)):
        assert "doubles as setting" not in text, name
        assert "agent.reference" in text, name


def test_a_reference_brief_keeps_a_quoted_path_with_spaces():
    """Q29: the reference window's Match handed the run "Match the reference 'Future.wav'", the
    file's NAME, and the run looked for it in the project folder and failed. The card now names
    the reference by its path - and a path a user keeps references in can have spaces, which the
    brief's parser cut at the first one."""
    from orchestrator.targets import reference_path_from_brief, reference_specs_from_brief

    brief = ("Match the reference '~/Music/My Refs/Song One.wav'. Target loudness -11.4 LUFS "
             "integrated. Shape the tonal balance toward it: less air (-6.6 dB).")
    assert reference_path_from_brief(brief) == "~/Music/My Refs/Song One.wav"
    assert [s["path"] for s in reference_specs_from_brief(brief)] == ["~/Music/My Refs/Song One.wav"]
    # the plain forms still read as before
    assert reference_path_from_brief("match refs/target.wav") == "refs/target.wav"
    assert [s["path"] for s in reference_specs_from_brief("match a.wav and b.mp3")] == ["a.wav", "b.mp3"]


def test_the_reference_card_names_the_reference_by_its_path():
    """Q29, the card's half (read in the C++: no leg drives the card's brief)."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "daw" / "Source" / "ReferenceCard.h").read_text()
    compose = src[src.index("void composeBrief()"):]
    compose = compose[:compose.index("\n    }\n")]
    assert "Match the reference '\" + name" not in compose, "the brief names the file, not where it is"
    assert "refPathForBrief" in compose


def test_what_the_snapshot_drops_can_be_read_back():
    """Q16: on a 7-track beat the live edit no longer fit, `available_plugins` was dropped to
    `available_plugins_omitted: 5`, and nothing read it back - the engineer guessed a plug-in's
    name four ways and never tried the one that was installed. The DROP_ORDER rule: what a tier
    drops, a tool reads back. `read_track plugins:` (and `plugins:<words>`), and `library:`."""
    import json as _json
    from orchestrator.daw_bridge import read_track_entry
    from orchestrator.edit_state import compact_edit_state

    plugins = [{"name": "Serum 2", "format": "VST3", "instrument": True},
               {"name": "Serum 2 FX", "format": "AudioUnit"},
               {"name": "AUNBandEQ", "format": "AudioUnit"}]
    library = {"name": "s0", "item_count": 2,
               "items": [{"name": "Kick 01", "type": "wave"}, {"name": "Main", "type": "edit"}]}
    edit = {"tracks": [{"name": f"T{i}", "clips": [{"start_s": j} for j in range(40)]}
                       for i in range(30)],
            "available_plugins": plugins, "library": library}
    shown = _json.loads(compact_edit_state(edit, 2000))
    assert shown.get("available_plugins_omitted") == 3 and "available_plugins" not in shown
    got = read_track_entry(edit, "plugins:")
    assert got["ok"] and [p["name"] for p in got["available_plugins"]] == ["Serum 2", "Serum 2 FX", "AUNBandEQ"]
    hit = read_track_entry(edit, "plugins:serum fx")
    assert [p["name"] for p in hit["available_plugins"]] == ["Serum 2 FX"], hit
    lib = read_track_entry(edit, "library:kick")
    assert lib["ok"] and [i["name"] for i in lib["library"]] == ["Kick 01"], lib


def test_the_engineer_can_look_in_a_folder(tmp_path):
    """Q2: asked to use "the Snares And Claps one" from a folder of one-shots, the engineer
    guessed seven file names with analyze_sample and gave up - the DAW's chat had no folder
    listing (`list_files` was left out as "a daw_command there", and no command lists files).
    It has one now: the folder's subfolders and audio files, searchable by words."""
    from orchestrator import chat
    from orchestrator.daw_bridge import folder_listing

    (tmp_path / "PMIT_Snares And Claps_02.wav").write_bytes(b"")
    (tmp_path / "PMIT_Kick_01.wav").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")
    (tmp_path / "Loops").mkdir()
    got = folder_listing(str(tmp_path))
    assert got["ok"] and got["audio"] == ["PMIT_Kick_01.wav", "PMIT_Snares And Claps_02.wav"]
    assert got["folders"] == ["Loops"]
    hit = folder_listing(str(tmp_path), "snares claps")
    assert hit["audio"] == ["PMIT_Snares And Claps_02.wav"], hit
    assert not folder_listing(str(tmp_path / "nope"))["ok"]
    assert "list_files" in {d["name"] for d in chat.daw_tool_defs(False)}
