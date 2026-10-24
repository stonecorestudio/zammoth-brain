"""Phase 59 - sample packs: the brain's half, and the pins that hold the two sides together.

The app's half (the finishing function, the pack commands, the kit and instrument builders) is
proven by the console legs `--sample-finish`, `--sample-packs` and `--pack-build`. Here:

* the two pack commands that render are MEASURED like a render - the finished file's profile
  rides the same tool result - and waited for like one;
* the kinds a sound can be are ONE list at three places (the app's table, the command's
  grammar, both prompts);
* the single-value argument names, the confirm flag, and the rule that a pack is written
  whole and removed to the Trash.
"""
from __future__ import annotations

import re
from pathlib import Path

import anyio
import numpy as np
import pytest
import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
SR = 44100
PACK_KEYS = ("pack.create", "pack.add", "pack.multisample", "pack.list", "pack.read",
             "pack.remove", "pack.finish", "pack.build_kit", "pack.build_instrument")


def _hit(freq: float = 60.0, seconds: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.8 * np.exp(-t / 0.08) * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _packs_source() -> str:
    return (REPO / "daw/Source/SamplePacks.h").read_text()


def _app_kinds() -> list[str]:
    """The ids of `samplepacks::kinds()`, in the table's order."""
    src = _packs_source()
    table = src[src.index("inline const std::vector<Kind>& kinds()"):]
    table = table[:table.index("return all;")]
    return re.findall(r'\{\s*"([a-z0-9_]+)",\s*"[A-Za-z ]+",', table)


# ----------------------------------------------------------------- the same result, measured

def test_the_pack_commands_that_render_are_measured_and_waited_for(tmp_path):
    from orchestrator import daw_bridge, daw_server, turn_host

    kick = tmp_path / "Kick_Deep.wav"
    sf.write(kick, _hit(), SR, subtype="PCM_24")

    async def go():
        added = await turn_host.measured_command(
            "pack.add", {"ok": True, "sounds": [{"name": "Deep", "file": str(kick)}]})
        built = await turn_host.measured_command(
            "pack.build_kit", {"ok": True, "sounds": [{"name": "Deep", "file": str(kick)}]})
        return added, built
    added, built = anyio.run(go)
    assert "attack_ms" in added["sounds"][0]["profile"]
    assert "profile" not in built["sounds"][0]          # a kit is not a render: nothing to read
    assert daw_bridge.MEASURED_COMMANDS == {"sound.render", "pack.add", "pack.multisample"}
    for key in ("pack.add", "pack.multisample"):
        assert daw_server.reply_wait_s(key, False) == daw_server.RENDER_REPLY_TIMEOUT_S
    assert daw_server.reply_wait_s("pack.read", False) != daw_server.RENDER_REPLY_TIMEOUT_S


def test_a_finished_loop_is_read_as_a_loop(tmp_path):
    from orchestrator import daw_bridge

    beats, bpm = 4, 120.0
    n = int(round(beats * 60.0 / bpm * SR))
    t = np.arange(n) / SR
    loop = tmp_path / "Bass_Loop_Night_Am_120bpm.wav"
    sf.write(loop, (0.5 * np.sin(2 * np.pi * 110.0 * t)).astype(np.float32), SR, subtype="PCM_24")
    res = daw_bridge.attach_sound_profiles(
        {"ok": True, "sounds": [{"name": "Night", "file": str(loop), "loop_beats": beats, "bpm": bpm}]})
    profile = res["sounds"][0]["profile"]
    # the loop's own reading (its length against its beats, its seam) is there only when the
    # file was read AS a loop of those beats
    assert isinstance(profile.get("loop"), dict) and profile["loop"], sorted(profile)
    one_shot = daw_bridge.attach_sound_profiles({"ok": True, "sounds": [{"name": "Night", "file": str(loop)}]})
    assert "loop" not in one_shot["sounds"][0]["profile"]


def test_a_multisample_carries_three_profiles_not_sixteen(tmp_path):
    from orchestrator import daw_bridge

    zones = []
    for i in range(13):
        f = tmp_path / f"Glass_{i}.wav"
        sf.write(f, _hit(110.0 * 2 ** (i / 3.0)), SR, subtype="PCM_24")
        zones.append({"file": str(f), "root_note": 36 + 4 * i})
    res = daw_bridge.attach_sound_profiles({"ok": True, "instrument": "Glass", "zones": zones})
    read = [i for i, z in enumerate(res["zones"]) if "profile" in z]
    assert read == [0, 6, 12]
    assert not any("profile_error" in z for z in res["zones"])
    # one zone is one profile; a reply with neither list is handed back as it came
    one = daw_bridge.attach_sound_profiles({"ok": True, "zones": [dict(zones[0])]})
    assert "profile" in one["zones"][0]
    plain = {"ok": True, "pack": "Night Tools"}
    assert daw_bridge.attach_sound_profiles(plain) is plain


def test_a_file_that_cannot_be_read_does_not_fail_the_add(tmp_path):
    from orchestrator import daw_bridge

    res = daw_bridge.attach_sound_profiles(
        {"ok": True, "sounds": [{"name": "Gone", "file": str(tmp_path / "not there.wav")}]})
    assert res["ok"] is True and "profile_error" in res["sounds"][0]


# ----------------------------------------------------------------- one list of kinds, three places

def test_the_kinds_are_one_list_in_the_table_the_grammar_and_both_prompts():
    from orchestrator import chat, prompts
    from orchestrator.operate_eval import _arg_schemas

    kinds = _app_kinds()
    assert len(kinds) == 22 and len(set(kinds)) == 22, kinds
    grammar = _arg_schemas()["pack.add"]
    said = re.search(r"kind: ([a-z0-9_|]+)", grammar)
    assert said and said.group(1).split("|") == kinds, said
    for text in (chat.daw_prompt(), prompts.DAW_PROMPT_WHOLE):
        para = text[text.index("**Sample packs**"):]
        para = para[:para.index("\n  * **", 10)]
        for kind in kinds:
            assert f"`{kind}`" in para, kind
        for key in PACK_KEYS:
            assert f"`{key}`" in para, key
        assert "never paths" in para


def test_every_drum_kind_has_a_pad_and_no_two_share_one():
    src = _packs_source()
    table = src[src.index("inline const std::vector<Kind>& kinds()"):]
    table = table[:table.index("return all;")]
    rows = re.findall(r'\{\s*"([a-z0-9_]+)",\s*"[A-Za-z ]+",\s*"[A-Za-z0-9_]+",\s*"([a-z]+)",\s*(-?\d+),', table)
    assert len(rows) == 22, rows
    pads = {kind: int(note) for kind, family, note in rows if family == "drums" and int(note) >= 0}
    assert set(pads) == {"kick", "snare", "clap", "rim", "hat", "open_hat", "tom", "crash", "ride", "perc"}
    assert len(set(pads.values())) == len(pads) and all(36 <= n <= 51 for n in pads.values()), pads
    # ...and nothing that is not a drum is given a pad by its kind
    assert all(int(note) < 0 for kind, family, note in rows if family != "drums")


# ----------------------------------------------------------------- the doors

def test_the_pack_arguments_that_are_one_value_are_refused_as_lists():
    from orchestrator import command_door

    for name in ("pack", "sound", "instrument", "zip"):
        assert name in command_door.SCALAR_ARG_NAMES
        assert command_door.bad_scalar_arg({name: ["x"]})
    assert "sounds" not in command_door.SCALAR_ARG_NAMES          # a list, at both commands that take it
    assert not command_door.bad_scalar_arg({"pack": "Night Tools", "sounds": ["Deep"]})


def test_a_removal_asks_the_user_and_an_add_over_a_sound_asks_per_call():
    cmds = (REPO / "daw/Source/DawCommands.h").read_text()
    row = next(line for line in cmds.splitlines() if '"pack.remove"' in line and "packRemove" in line)
    assert re.search(r"true,\s*false,\s*true\s*\}", row), row            # wantsArgs, not userOnly, CONFIRM
    for key in PACK_KEYS:
        if key != "pack.remove":
            line = next(l for l in cmds.splitlines() if f'"{key}"' in l and l.lstrip().startswith("{ pack"))
            assert not re.search(r"true,\s*false,\s*true\s*\}", line), line
    app = (REPO / "daw/Source/App.cpp").read_text()
    yes = app[app.index("static bool needsUserYes"):]
    yes = yes[:yes.index("\n    }") + 6]
    assert 'command == "pack.add" || command == "pack.multisample"' in yes and "samplepacks::addWouldReplace (args)" in yes


def test_a_pack_command_leaves_a_folder_that_is_not_a_project_alone_unless_it_makes_a_track():
    cmds = (REPO / "daw/Source/DawCommands.h").read_text()
    looks = cmds[cmds.index("inline bool looksOnly"):]
    looks = looks[:looks.index("return false;")]
    for key in PACK_KEYS:
        makes_a_track = key in ("pack.build_kit", "pack.build_instrument")
        assert (f'"{key}"' in looks) != makes_a_track, key


def test_a_pack_is_written_whole_and_removed_to_the_trash():
    src = _packs_source()
    assert "replaceWithText" not in src and "replaceWithData" not in src
    write = src[src.index("inline bool writeManifest"):]
    assert "wholefile::write (manifestFile (packDir)" in write[:write.index("\n    }")]
    gone = src[src.index("inline bool removeFromDisk"):]
    gone = gone[:gone.index("\n    }")]
    assert "moveToTrash()" in gone and "trashDeletesForConsole()" in gone
    # the finished file goes through a temp beside its target, never straight onto it
    finish = (REPO / "daw/Source/SampleFinish.h").read_text()
    wav = finish[finish.index("inline bool writeWav24"):]
    wav = wav[:wav.index("\n    }")]
    assert "juce::TemporaryFile temp (target" in wav and "overwriteTargetFileWithTemporary()" in wav
    # ...and a sound of a call is moved into place only once every sound of it is written
    add = src[src.index("inline juce::var finishAdd"):]
    add = add[:add.index("inline juce::String mergeAdd")]
    assert add.index("if (cancelled.load()) { takeBack(); return {}; }") < add.index("e.incoming.moveFileTo (e.target)")


def test_no_path_comes_from_the_caller():
    """A pack's files are named from a kind and a name. No pack command takes a path, and the
    command door's own path names are not among any pack command's arguments."""
    from orchestrator import command_door
    from orchestrator.chat_rig import _arg_names
    from orchestrator.operate_eval import _arg_schemas

    schemas = _arg_schemas()
    for key in PACK_KEYS:
        names = set(_arg_names(schemas[key]))
        assert not names & set(command_door.PATH_ARG_NAMES), (key, names)
    src = _packs_source()
    assert "isAbsolutePath (over)" in src                    # the one path read is the harness's own override
    plan = src[src.index("inline juce::String planAdd"):]
    plan = plan[:plan.index("inline bool addWouldReplace")]
    assert "juce::File (" not in plan and "getChildFile (e.relFile)" in plan


@pytest.mark.parametrize("key", PACK_KEYS)
def test_every_pack_command_is_graded_swept_and_pinned(key):
    import json

    from orchestrator import operate_eval

    assert any(key in case.expect for case in operate_eval.CORPUS), key
    ids = json.loads((REPO / "server/tests/data/command_ids.json").read_text())
    enum = "pack" + "".join(part.capitalize() for part in key.split(".", 1)[1].split("_"))
    assert enum in ids, enum
    census = json.loads((REPO / "server/tests/data/command_sweep_census.json").read_text())
    assert f"{key} bare" in census, key
