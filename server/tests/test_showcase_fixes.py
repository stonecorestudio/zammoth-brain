"""Gates for the brain's half of the 2026-10-03 showcase session's fixes.

The session drove the real app from an empty project through MIDI and a mix, on camera, and
the agent hit five things no fixture had: it could not duck an instrument track, could not read
the master it had just automated, and measured its own work with a render that hears neither
automation nor MIDI - without anything saying so. The DAW's halves are console legs
(daw/Source/ShowcaseFixLegs.h); the record is docs/10-daw.md, "The showcase session's findings".

Each test drives the function the product calls and carries its own control.
"""
from __future__ import annotations

import json

from mix_engine.session import Session


class _FakeDaw:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, command, args=None, *a, **kw):
        self.calls.append(command)
        return {"ok": True}


# ---------- duck.set_key on a track the session holds no stem for ----------

def test_a_duck_on_a_track_with_no_stem_is_the_daws_to_make(demo_project):
    """"Duck the pad from the kick": the brain performed `duck.set_key` itself, and
    `Session.set_duck` refused a MIDI track ("unknown stem 'Pad'") - the agent then faked the
    duck with a tempo-synced LFO. The brain now declines the call for a target it holds no stem
    for, and the command goes to the DAW's own handler (--duck-midi); a stem is still its own."""
    from orchestrator import daw_server as server
    from orchestrator.daw_bridge import perform_routing
    brain = server.Brain(demo_project)
    daw = _FakeDaw()
    brain.daw_invoke = daw
    ops = brain.routing_ops()

    assert "Pad" not in Session(demo_project).stems
    out = perform_routing("duck.set_key", {"stem": "Pad", "key": "stem_a"}, ops, daw)
    assert out is None, "the caller sends a command the brain declines to the DAW"
    assert "routing.apply" not in daw.calls

    # control: a stem the session holds is performed here, realized through routing.apply
    done = perform_routing("duck.set_key", {"stem": "stem_b", "key": "stem_a"}, ops, daw)
    assert done is not None and done["ok"] is True and "routing.apply" in daw.calls
    duck = next(e for e in Session(demo_project)._state["stems"]["stem_b"]["chain"]
                if e["type"] == "duck")
    assert duck["key_stem_id"] == "stem_a"


def test_key_stem_id_is_read_as_the_key(demo_project):
    """The `duck` ACTION in a proposal spells its trigger `key_stem_id`; the COMMAND spells it
    `key`. The command's door read only `key`, and a key that is not read is "none" - which
    CLEARS the trigger. So the natural slip answered ok and keyed nothing."""
    from orchestrator import daw_server as server
    from orchestrator.daw_bridge import perform_routing
    brain = server.Brain(demo_project)
    daw = _FakeDaw()
    brain.daw_invoke = daw
    done = perform_routing("duck.set_key", {"stem": "stem_b", "key_stem_id": "stem_a"},
                           brain.routing_ops(), daw)
    assert done is not None and done["ok"] is True
    ducks = [e for e in Session(demo_project)._state["stems"]["stem_b"]["chain"]
             if e["type"] == "duck"]
    assert len(ducks) == 1 and ducks[0]["key_stem_id"] == "stem_a"


# ---------- read_track: the master and the returns ----------

def _edit() -> dict:
    return {
        "tracks": [
            {"index": 0, "name": "Kick", "kind": "stem", "track_kind": "audio",
             "clips": [{"index": 0, "start_s": 0.0}]},
            {"index": 1, "name": "Pad", "kind": "stem", "track_kind": "midi",
             "clips": [{"index": 0, "start_s": 22.6}],
             "modifiers": [{"target": "gain.gain_db", "type": "lfo"}],
             "plugins": [{"name": "4OSC", "instrument": True, "enabled": True}]},
            {"index": 2, "name": "bus:bus1", "kind": "bus"},
        ],
        "master": {"name": "master", "kind": "master", "gain_db": 0.0, "device_count": 2,
                   "devices": [{"type": "eq"}, {"type": "limiter"}],
                   "automation": [{"param": "eq#0.freq_hz", "num_points": 6, "device": True,
                                   "points": [{"t_s": 0, "value": 700}]}],
                   "automatable_targets": ["eq#0.freq_hz", "eq#0.gain_db"]},
        "returns": [{"name": "return:rv", "kind": "return", "gain_db": -3.0,
                     "devices": [{"type": "return_reverb"}]}],
    }


def test_read_track_answers_for_the_master_and_a_return():
    """The agent wrote a filter sweep onto the master and `read_track {"track": "master"}`
    answered "no track named 'master'": the lookup searched `tracks` only, and the master and
    the returns are published beside it."""
    from orchestrator.daw_bridge import read_track_entry
    edit = _edit()
    master = read_track_entry(edit, "master")
    assert master["ok"] is True and master["track"]["kind"] == "master"
    assert master["track"]["automation"][0]["param"] == "eq#0.freq_hz"
    ret = read_track_entry(edit, "return:rv")
    assert ret["ok"] is True and ret["track"]["kind"] == "return"
    assert read_track_entry(edit, "MASTER")["ok"] is True          # case-insensitive, like a track

    # control: a name that is no channel is still refused - and the list it offers has them all
    nope = read_track_entry(edit, "Mastre")
    assert nope["ok"] is False
    assert {"Kick", "bus:bus1", "master", "return:rv"} <= set(nope["tracks"])
    # control: an edit with no such rows (an older DAW build) answers as it did
    old = {"tracks": edit["tracks"]}
    assert read_track_entry(old, "master")["ok"] is False
    assert read_track_entry(old, "Kick")["ok"] is True


def test_the_new_rows_fit_the_prompt_budget_like_a_track():
    """`master` and `returns` sit outside `tracks`, so the tier that thins every track to the
    row that names it has to thin them too - or a session too big for its tracks' detail would
    still carry the master's in full. And a synth's switches go with its params."""
    from orchestrator.edit_state import _drop
    edit = _edit()
    edit["tracks"][1]["instrument_params"] = [{"name": "Level 1", "value": -6}]
    edit["tracks"][1]["synth"] = {"osc1.wave": "sin"}
    _drop(edit, "instrument_params")
    assert "synth" not in edit["tracks"][1]
    assert edit["tracks"][1]["instrument_params_omitted"] is True
    _drop(edit, "track_detail")
    assert edit["master"]["detail_omitted"] is True and "automation" not in edit["master"]
    assert edit["master"]["device_types"] == ["eq", "limiter"]      # still names what is there
    assert edit["returns"][0]["detail_omitted"] is True
    assert edit["master"]["name"] == "master" and edit["master"]["gain_db"] == 0.0


# ---------- what the measurements do not hear ----------

def test_render_names_what_it_did_not_hear():
    """The agent automated a master low-pass open over the intro, and its own render measured
    the whole song with the filter shut; it balanced three synths "by judgement" because no
    tool could read them. Both are properties of the offline engine - it renders the stems'
    chains, not the DAW's graph - and nothing said so. `render` now carries them."""
    from orchestrator.daw_bridge import render_blind_spots
    spots = {s["what"]: s for s in render_blind_spots(_edit())}
    assert spots["midi_tracks"]["tracks"] == ["Pad"]
    assert "track.bounce" in spots["midi_tracks"]["to_measure"]
    assert spots["automation"]["lanes"] == ["master: eq#0.freq_hz"]
    assert spots["modulators"]["targets"] == ["Pad: gain.gain_db"]
    assert "hosted_plugins" not in spots            # the synth itself is the midi_tracks entry

    # a hosted effect, and a fader lane on a stem, are named too
    edit = _edit()
    edit["tracks"][0]["plugins"] = [{"name": "Pro-Q 4", "kind": "external", "enabled": True},
                                    {"name": "Off Plug", "kind": "external", "enabled": False}]
    edit["tracks"][0]["automation"] = [{"param": "gain.gain_db", "num_points": 2}]
    spots = {s["what"]: s for s in render_blind_spots(edit)}
    assert spots["hosted_plugins"]["plugins"] == ["Kick: Pro-Q 4"]   # a bypassed one is not heard anyway
    assert "Kick: gain.gain_db" in spots["automation"]["lanes"]

    # controls: a muted MIDI track and one with no clips are not in the mix the user hears;
    # stems and chains alone leave nothing unheard, and the field is then absent
    quiet = _edit()
    quiet["tracks"][1]["muted"] = True
    assert "midi_tracks" not in {s["what"] for s in render_blind_spots(quiet)}
    empty = _edit()
    empty["tracks"][1]["clips"] = []
    assert "midi_tracks" not in {s["what"] for s in render_blind_spots(empty)}
    assert render_blind_spots({"tracks": [{"name": "Kick", "track_kind": "audio",
                                           "clips": [{"index": 0}]}]}) == []
    assert render_blind_spots(None) == []


def test_the_knob_commands_door_leads_to_the_one_switch_table():
    """Two sessions fixed WOOLLY's switches on the same day, each with its own table (reader
    D6.R9: `synth{}` + `instrument.set_patch`; the showcase branch: `instrument_switches` set
    by `instrument.set_param`). Merged 2026-10-04 onto ONE: the knob command keeps its door
    (`synthswitch::fieldFor` - the spellings a caller reaches for), and every name it can
    answer is a field of `synthSwitchFields`. test_eval_fixture_shape.py holds the eval
    fixture's `synth{}` to the same list."""
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / "daw" / "Source"
    text = (root / "DawSession.h").read_text(encoding="utf-8")
    assert text.count("inline juce::var synthSwitchesVar (te::FourOscPlugin& fo)") == 1
    fields = set(re.findall(r'"([a-z0-9]+\.[a-z]+)"',
                            text.split("inline juce::StringArray synthSwitchFields()")[1].split("}")[1]))
    assert {"osc1.wave", "filter.type", "voice.mode", "fx.reverb", "delay.beats"} <= fields
    door = text.split("namespace synthswitch")[1].split("//== Track KIND")[0]
    for name in re.findall(r'return "([a-z]+\.[a-z]+)";', door):                 # whole fields
        assert name in fields, name
    for group, key in re.findall(r'return "(osc|lfo)" \+ [^;]+ \+ "(\.[a-z]+)";', door):   # composed ones
        assert f"{group}1{key}" in fields, group + key
    assert len(re.findall(r'return "', door)) >= 20
    for gone in ("synthSwitchIds", "struct Ref", "inline std::vector<Ref> all"):
        assert gone not in text, gone
    assert "instrument_switches" not in (root / "EditSnapshot.h").read_text(encoding="utf-8")
    app = (root / "App.cpp").read_text(encoding="utf-8")
    assert "dawsession::synthswitch::fieldFor (want)" in app
    assert app.count("dawsession::synthSetSwitches (*fo, set, um") >= 2   # set_param's check and its write
    # ...and set_patch reaches the same table through the patch document (Phase 57): its check and its write
    assert app.count("dawsession::synthApplyPatch (*fo, juce::var (doc.get()), um, how") >= 2


def test_analyzing_a_midi_track_says_why_there_is_no_stem():
    """`analyze_stem {"stem_id": "Pad"}` answered the engine's `no such stem: Pad; stems:
    [...]` for a track the agent had just made. The reason and the way round it now stand in
    for that error - only when the name really is a track in the DAW."""
    from orchestrator.daw_bridge import stemless_track_hint
    edit = _edit()
    hint = stemless_track_hint(edit, {"stem_id": "Pad"})
    assert hint and "MIDI" in hint and "track.bounce" in hint
    edit["tracks"].append({"index": 3, "name": "Audio 4", "track_kind": "audio", "clips": []})
    lane = stemless_track_hint(edit, {"stem_id": "Audio 4"})
    assert lane and "no audio" in lane and "track.bounce" not in lane
    # controls: a name that is no track keeps the engine's own error; no edit, no hint
    assert stemless_track_hint(edit, {"stem_id": "Padd"}) is None
    assert stemless_track_hint(None, {"stem_id": "Pad"}) is None
    assert stemless_track_hint(edit, {}) is None


def test_the_prompt_teaches_the_switches_and_the_blind_spots():
    """The knowledge has to reach the model: the catalog's argument note is C++ (--commands
    reads it), the rest is the chat prompt."""
    from orchestrator import chat
    text = json.dumps([chat.SYSTEM_PROMPT, getattr(chat, "DAW_OPERATING_NOTES", "")])
    source = open(chat.__file__, encoding="utf-8").read()
    for needle in ("`synth`", "osc1.wave", "instrument.set_patch", "not_heard", "track.bounce"):
        assert needle in source, needle
    assert "`master`" in source and "return:<id>" in source     # read_track's own description
    assert text                                                   # the module imports and builds
