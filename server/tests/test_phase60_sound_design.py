"""Phase 60 - the engineer as a sound designer that can be measured.

What the app does (a synth's delay and reverb tail in a render, the factory bank) is proven by
the console leg `--factory-bank`; the eval and the showcase prove their own graders through the
real app (`sound_eval --self-test`, `pack_showcase --self-test`: both need the console built
and are run by hand, like the chat rig). Here, with no app:

* `sound_classes` - what a KIND of sound usually measures as - on signals made in the test,
  each answer known first;
* the brain attaches that reading to a render, a pack sound and `analyze_sample`;
* the factory bank's census: the header's names, and every preset still clean against the
  ranges as they stand;
* the eval's recorded answers, re-graded: the golden passes, INIT fails, the mutant fails on
  the figure its case names;
* the showcase's graders on fabricated feeds and packs;
* the pins that hold the pieces together (the three knowledge topics, the tool definitions,
  the C++ doors).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from mcp.shared.memory import create_connected_server_and_client_session

from mix_engine import sound_classes as sc
from mix_engine.mcp_server import create_server
from mix_engine.models import SampleProfile
from mix_engine.sample_profile import sample_profile
from test_phase1_acceptance import call

REPO = Path(__file__).resolve().parents[2]
SR = 44100


def _text(rel: str) -> str:
    return (REPO / rel).read_text()


def _t(seconds: float) -> np.ndarray:
    return np.arange(int(seconds * SR)) / SR


def _profile(audio: np.ndarray, **kw) -> dict:
    return sample_profile(audio.astype(np.float32), SR, **kw).model_dump(exclude_none=True)


def _kick() -> np.ndarray:
    t = _t(0.6)
    phase = 2 * np.pi * np.cumsum(50 + 150 * np.exp(-t / 0.040)) / SR
    return 0.9 * np.sin(phase) * np.exp(-t / 0.12)


def _hat(decay: float = 0.012, seconds: float = 0.12) -> np.ndarray:
    rng = np.random.default_rng(7)
    noise = rng.uniform(-1.0, 1.0, int(seconds * SR))
    bright = np.diff(np.diff(noise, prepend=0.0), prepend=0.0)          # +12 dB an octave: nothing low
    bright /= np.max(np.abs(bright))
    return 0.8 * bright * np.exp(-_t(seconds) / decay)


def _saw(hz: float, seconds: float = 0.6) -> np.ndarray:
    """A held saw with a short fade at each end: what INIT gives for one note."""
    t = _t(seconds)
    env = np.minimum(1.0, t / 0.002) * np.minimum(1.0, (seconds - t) / 0.03)
    return 0.6 * (2.0 * ((t * hz) % 1.0) - 1.0) * env


def _sine(hz: float, seconds: float = 2.0, fade: float = 0.05) -> np.ndarray:
    t = _t(seconds)
    return 0.5 * np.sin(2 * np.pi * hz * t) * np.minimum(1.0, t / 0.005) * np.minimum(1.0, (seconds - t) / fade)


def _fields(issues: list[dict]) -> set[str]:
    return {i["field"] for i in issues}


# ---------------------------------------------------------------- a sound against its kind

def test_a_kick_made_here_is_clean_as_a_kick_and_not_as_anything_high():
    p = _profile(_kick())
    assert sc.issues(p, "kick") == []
    assert {"centroid_hz", "low_pct"} <= _fields(sc.issues(p, "hat"))
    assert "noisiness" in _fields(sc.issues(p, "snare"))
    assert "attack_ms" in _fields(sc.issues(p, "pad"))


def test_a_noise_tick_is_a_closed_hat_and_a_longer_one_an_open_hat():
    tick, ring = _profile(_hat()), _profile(_hat(decay=0.12, seconds=0.9))
    assert sc.issues(tick, "hat") == []
    assert _fields(sc.issues(tick, "open_hat")) == {"decay_ms"}        # too short to be an open one
    assert sc.issues(ring, "open_hat") == []
    assert _fields(sc.issues(ring, "hat")) == {"decay_ms"}             # ...and too long to be a closed one
    assert "sub_bass_pct" in _fields(sc.issues(tick, "kick"))


@pytest.mark.parametrize("kind", ["kick", "snare", "clap", "hat", "open_hat", "tom", "crash",
                                  "sub", "boom", "pluck", "pad", "riser", "impact"])
def test_one_held_saw_is_none_of_the_kinds_the_eval_asks_for(kind):
    """The premise of the sound-design eval: INIT - one saw - kept as a kick, a hat, a pad ...
    must draw an issue. (As a lead or a key sound it need not: a saw IS one.)"""
    low = kind in ("sub", "boom")
    p = _profile(_saw(65.4 if low else 261.6, 1.2 if low else 0.6))
    assert sc.issues(p, kind, played_midi=36 if low else 60), kind


def test_a_sub_is_a_low_sine_and_a_low_saw_is_not():
    assert sc.issues(_profile(_sine(65.4)), "sub", played_midi=36) == []
    said = _fields(sc.issues(_profile(_saw(65.4, 1.2)), "sub", played_midi=36))
    assert said & {"sub_bass_pct", "centroid_hz"}, said


def test_a_tuned_kind_is_held_to_the_key_it_was_played_on_in_any_octave():
    p = _profile(_sine(65.4))                                          # a C
    assert sc.issues(p, "bass", played_midi=36) == []
    assert sc.issues(p, "bass", played_midi=48) == []                  # an octave away is a choice
    off = sc.issues(p, "bass", played_midi=41)                         # played on F, sounds a C
    assert _fields(off) == {"pitch_midi"} and "out of tune" in off[0]["says"]
    assert sc.issues(p, "bass") == []                                  # no key given: nothing claimed
    kick = _profile(_kick())
    assert "pitch_midi" not in _fields(sc.issues(kick, "kick", played_midi=60))   # a drum is not held to its key


def test_a_held_sounds_decay_is_not_read():
    """A held sound's loudest sample can be anywhere in the hold, so `decay_ms` is the time
    from an arbitrary moment to the note's end: a bare pulse read 21 ms on one render and
    210 ms on the next. The held kinds read the crest instead."""
    t = _t(0.5)
    wave = np.where((t * 523.0) % 1.0 < 0.25, 1.5, -0.5)               # a quarter-width pulse, no offset
    pulse = 0.3 * wave * np.minimum(1.0, (0.5 - t) / 0.1) * np.minimum(1.0, t / 0.001)
    p = _profile(pulse)
    assert sc.issues(p, "lead", played_midi=72) == []
    for early in (5.0, 21.0, 400.0):
        assert sc.issues(dict(p, decay_ms=early), "lead", played_midi=72) == []
    assert not any(w.field == "decay_ms" for w in sc.CLASSES["lead"] + sc.CLASSES["keys"])
    assert "crest_db" in _fields(sc.issues(dict(p, crest_db=22.0), "lead", played_midi=72))


def test_what_any_kept_sound_should_be():
    cut = _profile(0.5 * np.sin(2 * np.pi * 220 * _t(0.3)))            # stops dead at full level
    assert "end_level_dbfs" in _fields(sc.issues(cut, "fx"))
    clipped = _profile(np.clip(1.6 * np.sin(2 * np.pi * 220 * _t(0.3)) * np.minimum(1.0, (0.3 - _t(0.3)) / 0.05), -1.0, 1.0))
    assert "full_scale_samples" in _fields(sc.issues(clipped, "fx"))
    offset = _profile((0.3 * np.sin(2 * np.pi * 220 * _t(0.3)) + 0.1) * np.minimum(1.0, (0.3 - _t(0.3)) / 0.05))
    assert "dc_offset_pct" in _fields(sc.issues(offset, "fx"))
    assert sc.issues({"silent": True}, "kick") == [{"field": "silent", "says": "the file is silent"}]
    assert sc.issues(_profile(_sine(440.0, 0.5)), "fx") == []


def test_a_loop_is_read_by_its_length_and_seam_and_not_by_one_notes_figures():
    beats, bpm = 4.0, 120.0
    n = int(round(beats * 60.0 / bpm * SR))
    loop = 0.5 * np.sin(2 * np.pi * 110.0 * np.arange(n) / SR)         # 220 whole cycles: the seam is clean
    good = _profile(loop, kind="loop", bpm=bpm, beats=beats)
    assert sc.issues(good, "bass") == []                               # no "attack", no "pitch" asked of a loop
    short = _profile(loop[:-100], kind="loop", bpm=bpm, beats=beats)
    assert any(i["field"] == "loop.length_error_samples" for i in sc.issues(short, "bass"))
    stepped = _profile(loop[:n - n // 440], kind="loop", bpm=bpm, beats=beats)   # cut mid-cycle
    assert {"loop.length_error_samples", "loop.seam_excess_db"} & _fields(sc.issues(stepped, "bass"))


def test_the_kinds_are_the_sample_packs_kinds_and_an_unknown_one_is_refused():
    src = _text("daw/Source/SamplePacks.h")
    table = src[src.index("inline const std::vector<Kind>& kinds()"):]
    app = re.findall(r'\{\s*"([a-z0-9_]+)",\s*"[A-Za-z ]+",', table[:table.index("return all;")])
    assert sorted(sc.kinds()) == sorted(app)
    with pytest.raises(ValueError, match="no sound kind 'cymbal'"):
        sc.issues({}, "cymbal")


def test_a_reading_says_nothing_unusual_as_an_empty_list_and_is_never_a_verdict():
    clean = sc.reading(_profile(_kick()), "Kick")
    assert clean == {"kind": "kick", "issues": []}
    flagged = sc.reading(_profile(_saw(261.6)), "kick", 60)
    assert flagged["issues"] and all(isinstance(s, str) for s in flagged["issues"])
    assert "not a verdict" in flagged["issues_are"]
    for kind, wants in sc.CLASSES.items():
        for w in wants:
            assert w.why and w.field in sc._WORDS and (w.lo is not None or w.hi is not None), (kind, w)


# ---------------------------------------------------------------- the brain says it in the same result

def test_a_render_and_a_pack_sound_that_name_their_kind_come_back_with_issues(tmp_path):
    from orchestrator import daw_bridge as B

    kick, saw = tmp_path / "kick.wav", tmp_path / "saw.wav"
    sf.write(kick, _kick().astype(np.float32), SR, subtype="FLOAT")
    sf.write(saw, _saw(261.6).astype(np.float32), SR, subtype="FLOAT")
    res = B.attach_sound_profiles({"ok": True, "sounds": [
        {"name": "a", "kind": "kick", "file": str(kick), "root_note": 60},
        {"name": "b", "kind": "kick", "file": str(saw), "root_note": 60},
        {"name": "c", "file": str(saw)},                               # no kind: nothing is claimed
        {"name": "d", "kind": "gong", "file": str(saw)}]})             # a kind the classes do not know
    a, b, c, d = res["sounds"]
    assert a["issues"] == [] and "issues_are" not in a
    assert any("under 150 Hz" in s for s in b["issues"]) and "not a verdict" in b["issues_are"]
    assert "issues" not in c and "issues" not in d and d["kind"] == "gong"
    assert all("profile" in s for s in res["sounds"])
    json.dumps(res)


@pytest.mark.anyio
async def test_analyze_sample_reads_a_file_against_a_kind(demo_project, tmp_path):
    kick = tmp_path / "kick.wav"
    sf.write(kick, _kick().astype(np.float32), SR, subtype="FLOAT")
    async with create_connected_server_and_client_session(create_server(demo_project)) as client:
        plain = await call(client, "analyze_sample", {"path": str(kick)})
        as_kick = await call(client, "analyze_sample", {"path": str(kick), "sound_kind": "kick"})
        as_hat = await call(client, "analyze_sample", {"path": str(kick), "sound_kind": "hat", "played_note": 60})
        with pytest.raises(Exception, match="no sound kind"):
            await call(client, "analyze_sample", {"path": str(kick), "sound_kind": "gong"})
        with pytest.raises(Exception, match="played_note"):
            await call(client, "analyze_sample", {"path": str(kick), "sound_kind": "kick", "played_note": 300})
    for got in (plain, as_kick, as_hat):
        SampleProfile.model_validate(got)
    assert plain.get("issues") is None and plain.get("sound_kind") is None
    assert as_kick["sound_kind"] == "kick" and as_kick["issues"] == []
    assert as_hat["issues"] and "not a verdict" in as_hat["issues_are"]
    assert as_hat["kind"] == "one_shot"                                 # `kind` is still one_shot | loop


def test_the_tool_and_both_prompts_say_how_a_kind_is_given():
    from orchestrator import agents, chat, prompts

    for defs in (chat._READ_TOOL_DEFS, agents.PLANNER_TOOL_DEFS):
        tool = next(t for t in defs if t["name"] == "analyze_sample")
        assert {"sound_kind", "played_note"} <= set(tool["input_schema"]["properties"])
        assert "sound_kind" in tool["description"]
    for text in (chat.DAW_SYSTEM_PROMPT, prompts.DAW_PROMPT_WHOLE):
        assert "`issues`" in text and "not that it is good" in text
        assert "starting points across DRUMS" in text
        for topic in ("woolly-synth", "drum-synthesis", "sample-packs"):
            assert topic in text


def test_sound_render_takes_a_kind_and_says_it_back():
    factory = _text("daw/Source/SoundFactory.h")
    assert re.search(r'fields \{[^}]*"kind"', factory)
    assert 's->setProperty ("kind", p.kind)' in factory and 's->setProperty ("root_note", p.notes[0].pitch)' in factory
    assert "ctx.kindId = [] (const juce::String& k) { auto* kind = samplepacks::kindNamed (k);" in _text("daw/Source/App.cpp")
    grammar = _text("daw/Source/DawCommands.h")
    line = grammar[grammar.index('{ "sound.render",'):grammar.index('{ "pack.create",')]
    assert "kind?" in line and "issues" in line


# ---------------------------------------------------------------- a synth's tail

def test_a_render_is_sized_by_the_synths_own_delay_and_reverb():
    """A render is sized by what the patch's delay and reverb add (`--factory-bank` (t1)-(t3)
    measure it). Since engine patch 0052 (2026-10-17, the WOOLLY audit) the ONE formula is the
    engine's own - `FourOscPlugin::tailSecondsAt`, behind `getTailLength` - because the engine's
    FREEZE asks getTailLength too and cut a patch's echoes; the app's `synthTailSeconds` delegates."""
    session = _text("daw/Source/DawSession.h")
    assert "inline double synthTailSeconds (te::FourOscPlugin& fo, double bpm)" in session
    assert "return fo.tailSecondsAt (bpm);" in session
    tail = session[session.index("inline double exportTailSeconds"):]
    tail = tail[:tail.index("return juce::jmin (tailS, capS);")]
    assert "p->getTailLength()" in tail
    patch = _text("daw/patches/0052-Zammoth-patch-4OSC-a-note-released-before-its-retrig.patch")
    assert "double FourOscPlugin::tailSecondsAt (double bpm) const" in patch
    assert "return tailSecondsAt (edit.tempoSequence.getBpmAt" in patch


# ---------------------------------------------------------------- the factory bank

def _census() -> dict:
    return json.loads(_text("orchestrator/sound_bank_census.json"))


def test_the_census_is_the_headers_bank():
    from orchestrator import sound_bank

    names = sound_bank.header_names()
    census = _census()
    assert len(names) == len(set(names)) and 48 <= len(names) <= 64, len(names)
    assert names[0] == "INIT" and names[1] == "SOFT KEYS"              # the face's arrows step from INIT to this
    assert sorted(census["presets"]) == sorted(names)
    assert census["sound_classes_version"] == sc.SOUND_CLASSES_VERSION


def test_every_factory_preset_still_reads_clean_against_its_kind():
    """The profiles are the ones measured when the bank was signed off; the RANGES are today's.
    A range that would turn a factory preset into an issue fails here, without the app."""
    from orchestrator import sound_bank

    census = _census()
    assert sound_bank.problems(census) == []
    wrong = []
    for name, entry in census["presets"].items():
        if name == "INIT":
            assert not entry["kind"]
            continue
        assert entry["kind"] in sc.CLASSES, (name, entry["kind"])
        for issue in sc.issues(entry["profile"], entry["kind"], entry["note"]):
            wrong.append(f"{name} ({entry['kind']}): {issue['says']}")
    assert wrong == []


def test_the_bank_is_spread_over_the_panes_categories_and_tagged_from_one_vocabulary():
    header = _text("daw/Source/WoollyPresets.h")
    words = header[header.index("static const juce::StringArray words {"):]
    vocabulary = set(re.findall(r'"([a-z0-9]+)"', words[:words.index("};")]))
    stock = set(re.findall(r'"([A-Z]+)"', re.search(r"stockCategories\(\) \{ return \{([^}]*)\}", header).group(1)))
    census = _census()["presets"]
    per: dict[str, int] = {}
    for name, entry in census.items():
        if name == "INIT":
            continue
        assert entry["category"] in stock, name
        per[entry["category"]] = per.get(entry["category"], 0) + 1
        assert 2 <= len(entry["tags"]) <= 6 and set(entry["tags"]) <= vocabulary, (name, entry["tags"])
    assert set(per) == stock and min(per.values()) >= 4, per
    drums = [e for e in census.values() if e["category"] == "DRUMS"]
    assert all(e["note"] == 60 for e in drums)                          # a drum preset is written for C3
    assert {e["kind"] for e in drums} >= {"kick", "snare", "clap", "hat", "open_hat", "tom", "crash"}


def test_a_factory_presets_routes_are_added_to_inits_own():
    header = _text("daw/Source/WoollyPresets.h")
    assert "return factoryPatch (factory()[(size_t) i]);" in header
    init = header[header.index("inline const char* initRoutes()"):header.index("inline juce::var factoryPatch")]
    assert len(re.findall(r'"source"', init)) == 10
    table = header[header.index("inline const std::vector<Factory>& factory()"):header.index("inline const char* initRoutes()")]
    assert '"cc11"' not in table and '"cc74"' not in table              # no preset restates them


# ---------------------------------------------------------------- the eval

def test_the_evals_corpus_is_briefs_in_a_users_words():
    from orchestrator import chat_rig as R
    from orchestrator import sound_eval as E

    assert len(E.CASES) >= 12 and len({c.key for c in E.CASES}) == len(E.CASES)
    assert len({c.pack for c in E.CASES}) == len(E.CASES)
    assert len({c.kind for c in E.CASES}) >= 10
    keys = set(R.sweep_plan())
    for case in E.CASES:
        assert case.kind in sc.CLASSES
        words = set(re.findall(r"[a-z_]+\.[a-z_]+", case.prompt()))
        assert not words & keys, (case.key, words & keys)
        assert not re.search(r"\b(preset|patch|kind|loop_beats|voice\.phase|ampDecay|tune1)\b", case.prompt()), case.key
        assert case.mutant != case.golden and case.mutant_turns_red
        assert case.pack in case.prompt()


def test_the_recorded_answers_grade_as_the_self_test_left_them():
    """`sound_eval --self-test --record` wrote what each case's scripted answers measured as in
    the real app. Re-graded here: the golden passes, INIT fails, the mutant fails on the figure
    its case names - so a change to a grader or a range that breaks one is seen without the app."""
    from orchestrator import sound_eval as E

    recorded = json.loads(_text("orchestrator/sound_eval_goldens.json"))["cases"]
    assert sorted(recorded) == sorted(c.key for c in E.CASES)
    for case in E.CASES:
        got = {v: E.outcome_of(case, recorded[case.key][v]["row"], recorded[case.key][v]["profile"])
               for v in ("golden", "init", "mutant")}
        assert got["golden"] == [], (case.key, got["golden"])
        assert got["init"], case.key
        assert any(case.mutant_turns_red.lower() in f.lower() for f in got["mutant"]), (case.key, got["mutant"])


def test_the_plan_layer_wants_a_sound_heard_before_it_is_kept_and_the_project_left_alone():
    from orchestrator import sound_eval as E

    def rows(*keys, failed=()):
        return [{"key": k, "failed": k in failed} for k in keys]
    assert E.plan_failures(rows("pack.create", "sound.render", "pack.add"), []) == []
    assert E.plan_failures(rows("pack.create", "sound.render", "sound.render", "pack.add"), ["INPUTDEVICE x"]) == []
    assert "no pack.add" in E.plan_failures(rows("sound.render"), [])[0]
    assert "no pack.add" in E.plan_failures(rows("sound.render", "pack.add", failed=("pack.add",)), [])[0]
    assert "without being heard" in E.plan_failures(rows("pack.create", "pack.add", "sound.render"), [])[0]
    assert any("removed" in f for f in E.plan_failures(rows("sound.render", "pack.add", "pack.remove"), []))
    assert any("project was changed" in f for f in E.plan_failures(rows("sound.render", "pack.add"), ["TRACK added"]))
    assert len(E.plan_failures([], [])) == 1


def test_an_ask_of_the_brief_reads_one_figure():
    from orchestrator import sound_eval as E

    kick = _profile(_kick())
    assert E.Ask("pitch_hz", None, 62.0, "deep").failure(kick, sc) is None
    assert "deep" in E.Ask("pitch_hz", None, 40.0, "deep").failure(kick, sc)
    on = E.Ask("pitch_midi", says="on its note", pitch_class=int(kick["pitch_midi"]) % 12)
    assert on.failure(kick, sc) is None
    assert E.Ask("pitch_midi", says="on E", pitch_class=(int(kick["pitch_midi"]) + 3) % 12).failure(kick, sc)
    assert "no clear pitch" in E.Ask("pitch_midi", says="on E", pitch_class=4).failure(_profile(_hat()), sc)
    held = dict(kick)
    held.pop("decay_ms", None)
    assert "never fell" in E.Ask("decay_ms", None, 100.0, "short").failure(held, sc)   # never reached is over a ceiling
    assert E.Ask("decay_ms", 100.0, None, "long").failure(held, sc) is None             # ...and under no floor


def test_a_live_run_must_be_asked_for_twice_and_a_check_needs_repeats(capsys):
    from orchestrator import sound_eval as E

    assert E.main(["--live", "api"]) == 1
    assert "BILLED" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        E.main(["--live", "api", "--billed", "--check"])
    assert E.main(["--list"]) == 0


# ---------------------------------------------------------------- the showcase

def _feed(drop: tuple[str, ...] = (), failed: tuple[str, ...] = ()) -> list[dict]:
    from orchestrator import pack_showcase as P

    turns = []
    for stage in P.STAGES:
        commands = [r.any_of[0] for r in stage.requires if r.any_of[0] not in drop]
        turns.append({"user": stage.prompt, "commands": commands,
                      "failed": [k for k in commands if k in failed]})
    return turns


def test_the_showcases_prompts_are_a_users_words_and_its_commands_layer_reads_the_feed():
    from orchestrator import chat_rig as R
    from orchestrator import pack_showcase as P

    keys = set(R.sweep_plan())
    for stage in P.STAGES:
        assert not set(re.findall(r"[a-z_]+\.[a-z_]+", stage.prompt)) & keys, stage.key
        assert all(k in keys for r in stage.requires for k in r.any_of), stage.key
    assert all(c.ok for c in P.grade_commands(_feed()))
    lost = [c for c in P.grade_commands(_feed(drop=("pack.multisample",))) if not c.ok]
    assert [c.label for c in lost] == ["5 An instrument: the multisample"]
    refused = [c for c in P.grade_commands(_feed(failed=("pack.add",))) if not c.ok]
    assert len(refused) == 4 and all("kept" in c.label for c in refused)      # one failure cancels one call
    assert all(not c.ok for c in P.grade_commands([]))
    typed = _feed()
    typed[0]["user"] = typed[0]["user"].replace("call it", "name it") + "  "    # a retyped prompt still finds its stage
    assert all(c.ok for c in P.grade_commands(typed))


def _fake_pack(root: Path) -> Path:
    """A pack on disk that the FILES layer should pass: the manifest and empty files."""
    from orchestrator import pack_showcase as P

    folder = root / P.PACK
    sounds = []

    def add(kind: str, name: str, rel: str, **more):
        (folder / rel).parent.mkdir(parents=True, exist_ok=True)
        (folder / rel).write_bytes(b"")
        preset = f"Presets/{Path(rel).stem}.woolly"
        (folder / preset).parent.mkdir(parents=True, exist_ok=True)
        (folder / preset).write_text("{}")
        sounds.append({"name": name, "kind": kind, "file": rel, "loop": False, "preset": preset, **more})
    for i, (kind, n) in enumerate(P.ONE_SHOTS.items()):
        for j in range(n):
            add(kind, f"S{i}{j}", f"One Shots/X/{kind}_{j}.wav", **({"root_note": 36, "root": "C1"} if kind in P.ON_C else {}))
    for kind, beats in P.LOOPS.items():
        add(kind, f"L{kind}", f"Loops/X/{kind}.wav", loop=True, beats=beats, bpm=P.BPM, key="Am")
    zones = []
    for root_note in range(48, 85, 6):
        rel = f"Instruments/Pad/Pad_{root_note}.wav"
        (folder / rel).parent.mkdir(parents=True, exist_ok=True)
        (folder / rel).write_bytes(b"")
        zones.append({"file": rel, "root_note": root_note, "root": str(root_note)})
    (folder / "pack.json").write_text(json.dumps({
        "name": P.PACK, "sounds": sounds, "instruments": [{"name": "Pad", "zones": zones}],
        "finished": "2026-10-13T00:00:00"}))
    return folder


def test_the_files_layer_counts_what_the_pack_holds(tmp_path):
    from orchestrator import pack_showcase as P

    folder = _fake_pack(tmp_path)
    assert all(c.ok for c in P.grade_files(P.find_pack(tmp_path))), [c.line() for c in P.grade_files(P.find_pack(tmp_path)) if not c.ok]
    doc = json.loads((folder / "pack.json").read_text())

    def broken(change) -> list[str]:
        copy = json.loads(json.dumps(doc))
        change(copy)
        (folder / "pack.json").write_text(json.dumps(copy))
        return [c.label for c in P.grade_files(P.find_pack(tmp_path)) if not c.ok]
    assert broken(lambda d: d.pop("finished")) == ["the pack is finished"]
    assert broken(lambda d: d["sounds"].pop(0)) == ["one-shots: 2 x kick"]
    assert broken(lambda d: d["instruments"][0]["zones"].__delitem__(slice(2, None))) == [
        f"an instrument of {P.MIN_ZONES}+ zones across {P.MIN_SPAN}+ semitones"]
    assert broken(lambda d: [r.update(bpm=100) for r in d["sounds"] if r.get("loop")]) == [
        "a 4-beat bass loop at 120 BPM in A minor", "a 8-beat pluck loop at 120 BPM in A minor"]
    assert broken(lambda d: [r.update(root_note=41) for r in d["sounds"] if r["kind"] == "sub"]) == ["the tuned one-shots are on C"]
    assert [c.ok for c in P.grade_files(None)] == [False]
    ok, checks = P.verify(tmp_path / "no-project", tmp_path / "no-packs", [], quiet=True)
    assert not ok and {c.layer for c in checks if not c.ok} == {"commands", "files", "audio", "project"}


# ---------------------------------------------------------------- the knowledge topics

def test_the_three_topics_are_in_the_book_and_carry_no_history():
    from orchestrator import knowledge_deep

    for topic in ("woolly-synth", "drum-synthesis", "sample-packs"):
        body = knowledge_deep.lookup(topic)
        assert topic in knowledge_deep.TOPICS and len(body) > 2500, topic
        assert not re.search(r"\b20\d\d-\d\d-\d\d\b|\bPhase \d+\b", body), topic       # a model never saw "before"
    synth = knowledge_deep.lookup("woolly-synth")
    assert "MIDI note number" in synth and "What it cannot do" in synth
    drums = knowledge_deep.lookup("drum-synthesis")
    for word in ("Kick", "Snare", "Clap", "Hats", "808", "What these are not"):
        assert word in drums
    assert "What a pack here cannot hold" in knowledge_deep.lookup("sample-packs")


def test_what_the_drum_topic_says_a_kick_and_a_hat_are_is_what_the_factory_bank_holds():
    """The recipes were measured from the bank's own presets: the topic and the header must not
    drift apart on the figures a model will copy."""
    from orchestrator import knowledge_deep

    header = _text("daw/Source/WoollyPresets.h")
    drums = knowledge_deep.lookup("drum-synthesis")

    def knobs(name: str) -> dict:
        block = header[header.index('{ "%s", "DRUMS"' % name):]
        block = block[:block.index('"drum,')]
        return json.loads("{" + re.search(r'"knobs":\s*\{(.*?)\}', block, flags=re.S).group(1) + "}")
    deep, punch, hat, open_hat = knobs("KICK DEEP"), knobs("KICK PUNCH"), knobs("HAT CLOSED"), knobs("HAT OPEN")
    assert -31 <= deep["tune1"] <= -27 and -31 <= punch["tune1"] <= -27 and "`tune1` -27..-31" in drums
    assert punch["ampDecay"] == 0.2 and deep["ampDecay"] == 0.45 and "`ampDecay` 0.2 falls" in drums and "0.45 in about 200 ms" in drums
    assert (hat["tune1"], hat["tune2"], hat["tune3"], hat["tune4"]) == (4, 10, 12, 15) and "`tune` 4, 10 (+40 cents)" in drums
    assert hat["filterFreq"] == 117 and "high-pass at note 117" in drums
    assert hat["ampDecay"] == 0.05 and open_hat["ampDecay"] == 0.6 and "`ampDecay` 0.05 is a closed hat" in drums


def test_the_operate_evals_fixture_lists_the_bank_as_the_app_lists_it():
    """The user's own presets first, then the factory bank in the face's order - what the
    snapshot's `synth_presets` holds since the bank grew past fifty."""
    from orchestrator import operate_eval, sound_bank

    listed = operate_eval.OPERATE_EDIT["synth_presets"]["list"]
    names = sound_bank.header_names()
    own = [n for n in listed if "/" in n]
    assert listed == own + names and own
    app = _text("daw/Source/App.cpp")
    block = app[app.index("// The user's own first: the factory names are the same on every machine"):]
    assert "for (const bool user : { true, false })" in block[:700] and "rows.size() < 96" in block[:700]


def test_the_skills_table_of_ranges_is_the_codes():
    """The mix-analysis skill prints every kind's ranges; a range that moves in the code moves
    there (and `SOUND_CLASSES_VERSION` with it)."""
    skill = _text(".claude/skills/mix-analysis/SKILL.md")
    section = skill[skill.index("## One sound, against its kind"):]
    for kind, wants in sc.CLASSES.items():
        cells = "; ".join(f"`{w.field}` {sc._range_words(w.lo, w.hi, sc._WORDS[w.field][1])}" for w in wants)
        assert f"| `{kind}` | {cells if cells else '(none of its own)'} |" in section, kind


def test_the_word_people_use_for_a_boom_bass_is_taken_as_that_kind():
    """The kind is `boom` (the app, the classes, the manual); "808" - what a producer calls
    it - is taken at every door and answered under the kind's own name."""
    boom = _profile(_sine(55.0, 1.5, fade=0.6))
    assert sc.canonical(" 808 ") == "boom" and "808" not in sc.CLASSES and "boom" in sc.CLASSES
    assert sc.issues(boom, "808", played_midi=33) == sc.issues(boom, "boom", played_midi=33)
    assert sc.reading(boom, "808", 33)["kind"] == "boom"
    packs = _text("daw/Source/SamplePacks.h")
    assert 'id.trim().equalsIgnoreCase ("808") ? juce::String ("boom")' in packs
    assert '{ "boom",     "Bass",       "Boom",     "bass",' in packs
    for text in (_text("docs/17-user-manual.md"),):
        bank = text[text.index("**The factory bank** is"):text.index("### Rendering a sound without touching the project")]
        assert "808" not in bank and "BOOM BASS" in bank and "GROWL" not in bank


def test_the_manual_says_how_many_presets_the_bank_holds():
    from orchestrator import sound_bank

    words = {57: "fifty-seven", 58: "fifty-eight", 59: "fifty-nine", 60: "sixty", 61: "sixty-one"}
    names = sound_bank.header_names()
    manual = _text("docs/17-user-manual.md")
    assert f"**The factory bank** is {words[len(names)]} presets" in manual
    listed = manual[manual.index("- **KEYS** — SOFT KEYS"):manual.index("A **DRUMS** preset is a drum made by the synth")]
    on_the_page = [n.strip() for item in listed.split("- **")[1:]
                   for n in " ".join(item.split("—", 1)[1].split()).split(",") if n.strip()]
    assert sorted(on_the_page) == sorted(n for n in names if n != "INIT")
