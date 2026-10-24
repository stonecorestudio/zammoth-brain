"""What the mashup showcase's rehearsal found (2026-10-11): two released songs, split, lined up,
arranged, mixed by the agent in the real app. The brain's half of each fix is pinned here.

1. A proposal of eleven actions was refused with the request model's own dump in the feed
   ("1 validation error for propose_actionsArguments ... input_value=[{'type': 'comp', ...").
   The limit is said in a sentence that tells the agent what to do, and no validation dump
   reaches the agent or the feed, whatever was refused.
"""
from __future__ import annotations

import re

import pytest
from pydantic import BaseModel, ValidationError

from mix_engine import guardrails
from mix_engine import models as engine_models
from mix_engine.models import ProbeRequest, ProposalRequest
from orchestrator.loop import tool_error_text


class _Text:
    """What an MCP tool failure hands the loop: content items with a `.text`."""
    def __init__(self, text: str) -> None:
        self.text = text


class _ToolArguments(BaseModel):
    """How the MCP server holds a tool's one argument: under `req`, so every path in a
    refusal starts with it (the feed showed `propose_actionsArguments` / `req.actions`)."""
    req: ProposalRequest


def _proposal(actions: list[dict]) -> dict:
    return {"rationale": "level the vocal",
            "predicted_impact": [{"metric": "lufs_i", "predicted_delta": -0.5, "unit": "LUFS"}],
            "actions": actions}


def _gain(delta_db: float = -0.5) -> dict:
    return {"type": "gain", "stem_id": "vocals", "delta_db": delta_db}


def _refused(actions: list[dict]) -> str:
    """The text of the failed engine tool as the loop words it for the agent and the feed."""
    with pytest.raises(ValidationError) as caught:
        _ToolArguments.model_validate({"req": _proposal(actions)})
    return tool_error_text("propose_actions",
                           [_Text(f"Error executing tool propose_actions: {caught.value}")])


# ---------------------------------------------------------------------------------------------
# 1 - too many actions in one proposal
# ---------------------------------------------------------------------------------------------

def test_the_action_limit_is_one_number():
    assert engine_models.MAX_ACTIONS_PER_REQUEST == guardrails.LIMITS["actions_per_proposal"]
    for model in (ProposalRequest, ProbeRequest):
        bound = [m for m in model.model_fields["actions"].metadata if hasattr(m, "max_length")]
        assert bound and bound[0].max_length == engine_models.MAX_ACTIONS_PER_REQUEST


def test_eleven_actions_are_refused_in_a_sentence_that_says_what_to_do():
    said = _refused([_gain() for _ in range(11)])
    assert "at most 8 actions" in said and "11" in said and "split" in said, said
    assert "List should have at most" not in said, said


def test_eight_actions_are_a_proposal():
    assert len(ProposalRequest.model_validate(_proposal([_gain() for _ in range(8)])).actions) == 8


@pytest.mark.parametrize("actions", [
    [_gain() for _ in range(11)],                                        # too many
    [{"type": "gain", "stem_id": "vocals", "gain_db": -1.0}],            # a field the model lacks
    [{"stem_id": "vocals", "delta_db": -1.0}],                           # no type
    [{"type": "gain", "stem_id": "vocals", "delta_db": "loud"}],         # a value of the wrong kind
    [_gain(), {"type": "eq", "stem_id": "vocals"}],                      # a required field missing
])
def test_no_validation_dump_reaches_the_agent_or_the_feed(actions: list[dict]):
    said = _refused(actions)
    for noise in ("validation error", "input_value", "input_type", "errors.pydantic.dev",
                  "[type=", "Value error,", "function-after", "Arguments", "req."):
        assert noise not in said, f"{noise!r} in: {said}"
    assert said.startswith("engine tool 'propose_actions' failed: ") and len(said) < 600, said


def test_a_refusal_still_names_what_it_refused():
    said = _refused([{"type": "gain", "stem_id": "vocals", "gain_db": -1.0}])
    assert "unknown field 'gain_db'" in said and "delta_db" in said, said
    # one wrong value is reported by every action type there is (67 problems): the agent is
    # told the one that is about the action it wrote
    said = _refused([{"type": "gain", "stem_id": "vocals", "delta_db": "loud"}])
    assert "action 0, delta_db" in said and "valid number" in said, said
    assert "Input should be 'eq'" not in said and "more)" not in said, said
    said = _refused([_gain(), {"type": "eq", "stem_id": "vocals"}])
    assert "action 1, band" in said and "required" in said.lower(), said


def test_a_failure_that_is_not_a_validation_dump_is_worded_as_it_was():
    said = tool_error_text("render", [_Text("Error executing tool render: stem 'kick' has no audio")])
    assert said == "engine tool 'render' failed: stem 'kick' has no audio"


# ---------------------------------------------------------------------------------------------
# 2 - one track's own tempo (`analyze_tempo`)
#
# The agent had no way to ask what tempo ONE track is at: every grid the engine reads is the
# master's. With two songs in the project it muted each in turn and read the master.
# ---------------------------------------------------------------------------------------------

import numpy as np
import soundfile as sf
from mcp.shared.memory import create_connected_server_and_client_session

from mix_engine import analysis
from mix_engine.mcp_server import create_server
from mix_engine.models import ActivityAnalysis, TempoAnalysis
from test_phase1_acceptance import call
from test_tempo_long_take import _song

SR = 44100


def test_one_tracks_tempo_and_where_its_beat_falls():
    got = analysis.stem_tempo(_song(124.0, 30.0, SR, intro_s=0.0)[:, np.newaxis], SR, "drums")
    assert isinstance(got, TempoAnalysis) and got.stem_id == "drums" and got.reason is None
    assert abs(got.tempo_bpm - 124.0) <= 0.02 and got.tempo_confidence >= 0.5, got
    period = 60.0 / 124.0
    assert abs((got.beat_offset_s + period / 2) % period - period / 2) <= 0.003, got


@pytest.mark.parametrize("what", ["a steady tone", "a tone with a slow tremolo", "noise"])
def test_a_part_with_no_pulse_has_no_tempo_and_says_so(what: str):
    """A plain sine read 172 BPM at confidence 0.45 (its period beating against the tempo
    envelope's hop) and a tremolo'd one 186, with a "beat offset" of twelve seconds."""
    t = np.arange(SR * 12) / SR
    part = {"a steady tone": 0.2 * np.sin(2 * np.pi * 220.0 * t),
            "a tone with a slow tremolo": 0.2 * np.sin(2 * np.pi * 220.0 * t)
                                          * (0.6 + 0.4 * np.sin(2 * np.pi * 0.11 * t)),
            "noise": 0.1 * np.random.default_rng(1).standard_normal(t.size)}[what]
    assert analysis._tempo(part, SR) is None
    got = analysis.stem_tempo(part[:, np.newaxis], SR, "pad")
    assert (got.tempo_bpm, got.tempo_confidence, got.beat_offset_s) == (None, None, None)
    assert got.reason and "pulse" in got.reason


def test_a_beat_offset_is_always_inside_the_first_beat():
    for bpm, dur in ((72.0, 12.0), (120.0, 10.0), (124.0, 30.0)):
        tempo = analysis._tempo(_song(bpm, dur, SR, intro_s=3.0), SR)
        assert tempo is not None and 0.0 <= tempo[2] < 60.0 / tempo[0], (bpm, tempo)


def _two_song_project(folder) -> None:
    """Two rhythmic stems at DIFFERENT tempi: a master's grid cannot be both."""
    sf.write(folder / "song_a.wav", 0.5 * _song(124.0, 24.0, SR, intro_s=0.0, strays=0.0), SR)
    sf.write(folder / "song_b.wav", 0.5 * _song(100.0, 24.0, SR, intro_s=0.0, strays=0.0, seed=3), SR)


@pytest.mark.anyio
async def test_analyze_tempo_reads_each_track_on_its_own(tmp_path):
    _two_song_project(tmp_path)
    async with create_connected_server_and_client_session(create_server(tmp_path)) as client:
        a = await call(client, "analyze_tempo", {"stem_id": "song_a"})
        b = await call(client, "analyze_tempo", {"stem_id": "song_b"})
    assert abs(a["tempo_bpm"] - 124.0) <= 0.05 and abs(b["tempo_bpm"] - 100.0) <= 0.05, (a, b)
    assert a["stem_id"] == "song_a" and a["reason"] is None


@pytest.mark.anyio
async def test_analyze_tempo_round_trips_on_the_demo(demo_project):
    """The demo's stems are a tone and noise: no tempo, and the reply says so."""
    async with create_connected_server_and_client_session(create_server(demo_project)) as client:
        got = await call(client, "analyze_tempo", {"stem_id": "stem_a"})
    assert got["stem_id"] == "stem_a" and got["tempo_bpm"] is None and got["reason"]
    TempoAnalysis.model_validate(got)


# ---------------------------------------------------------------------------------------------
# 3 - where each part comes in and drops out, bar by bar (`analyze_activity`)
#
# Building the arrangement, the agent read a song's sections every 8 bars from bar 1; the song
# has a one-bar lead-in, so every section it took was a bar early. Nothing told it, to the bar,
# where the drums drop out and come back.
# ---------------------------------------------------------------------------------------------

def _kit_by_bars(plan: str, bpm: float = 120.0, bar1_s: float = 0.25, sr: int = SR) -> np.ndarray:
    """One bar per character: K = kick on every beat, h = a quiet hat on every beat, . = nothing."""
    beat = 60.0 / bpm
    x = np.zeros(int((bar1_s + len(plan) * 4 * beat + 0.5) * sr))
    t = np.arange(int(0.25 * sr)) / sr
    kick = 0.8 * np.sin(2 * np.pi * 60.0 * t) * np.exp(-t / 0.08)
    rng = np.random.default_rng(5)
    # a hat fill under a break: about 20 dB under the kit's full bars, as on a record
    hat = 0.25 * rng.standard_normal(int(0.06 * sr)) * np.exp(-np.arange(int(0.06 * sr)) / (0.02 * sr))
    for bar, what in enumerate(plan):
        hit = {"K": kick, "h": hat}.get(what)
        if hit is None:
            continue
        for b in range(4):
            at = int(round((bar1_s + (bar * 4 + b) * beat) * sr))
            x[at:at + len(hit)] += hit
    return x


def _spans(result: ActivityAnalysis, stem_id: str) -> list[tuple[str, int, int]]:
    stem = next(s for s in result.stems if s.stem_id == stem_id)
    return [(s.state, s.from_bar, s.to_bar) for s in stem.spans]


def test_where_a_part_plays_is_told_to_the_bar():
    drums = _kit_by_bars("KKKK....hhhhKKKK")
    got = analysis.analyze_activity({"drums": drums[:, np.newaxis]}, SR, bpm=120.0, bar1_s=0.25)
    assert _spans(got, "drums") == [("in", 1, 4), ("out", 5, 8), ("low", 9, 12), ("in", 13, 16)]
    assert got.bpm == 120.0 and got.bar1_s == 0.25 and got.bars == 16


def test_a_section_that_starts_one_bar_off_the_eight_bar_lattice_is_found_where_it_is():
    """The rehearsal's defect: a one-bar lead-in, then the drums out for bars 18-25 and back
    at 26. Read every 8 bars from bar 1 that is "17 and 25"."""
    plan = "." + "K" * 16 + "h" * 8 + "K" * 8          # bars 2-17 in, 18-25 low, 26-33 in
    got = analysis.analyze_activity({"drums": _kit_by_bars(plan)[:, np.newaxis]}, SR,
                                    bpm=120.0, bar1_s=0.25)
    assert _spans(got, "drums") == [("out", 1, 1), ("in", 2, 17), ("low", 18, 25), ("in", 26, 33)]


def test_activity_is_each_parts_own_and_a_silent_part_is_out_throughout():
    drums = _kit_by_bars("KKKKKKKK")
    late = _kit_by_bars("....KKKK")
    silent = np.zeros_like(drums)
    got = analysis.analyze_activity({"a": drums[:, np.newaxis], "b": late[:, np.newaxis],
                                     "c": silent[:, np.newaxis]}, SR, bpm=120.0, bar1_s=0.25)
    assert _spans(got, "a") == [("in", 1, 8)]
    assert _spans(got, "b") == [("out", 1, 4), ("in", 5, 8)]
    assert _spans(got, "c") == [("out", 1, 8)]


def test_activity_needs_a_grid_and_says_so_without_one():
    got = analysis.analyze_activity({"a": _kit_by_bars("KKKK")[:, np.newaxis]}, SR, bpm=None, bar1_s=0.0)
    assert got.stems == [] and got.reason and "tempo" in got.reason


@pytest.mark.anyio
async def test_analyze_activity_round_trips_and_takes_the_callers_grid(tmp_path, demo_project):
    sf.write(tmp_path / "drums.wav", _kit_by_bars("." + "K" * 8 + "h" * 4 + "K" * 4), SR)
    async with create_connected_server_and_client_session(create_server(tmp_path)) as client:
        got = await call(client, "analyze_activity", {"bpm": 120.0, "bar1_s": 0.25})
    ActivityAnalysis.model_validate(got)
    spans = [(s["state"], s["from_bar"], s["to_bar"]) for s in got["stems"][0]["spans"]]
    assert got["stems"][0]["stem_id"] == "drums"
    assert spans == [("out", 1, 1), ("in", 2, 9), ("low", 10, 13), ("in", 14, 17)], spans
    # the demo (a tone and noise, no pulse): with no grid given and none detected it says so
    async with create_connected_server_and_client_session(create_server(demo_project)) as client:
        none = await call(client, "analyze_activity", {})
    assert none["stems"] == [] and none["reason"]


# ---------------------------------------------------------------------------------------------
# 4 - a track of several clips is measured AS ARRANGED
#
# A stem was one file held where its ONE clip plays; a track of several clips was the file
# whole, from 0:00. Every lane of the mashup was several clips, so the agent's loudness, peak
# and every per-part number were of eight full-length stems summed - not of the mashup - and
# after it moved a clip its measurement "had not updated". The DAW now reports each plain clip
# of the stem's own file as a window (`clips`), with its clip gain and fades.
# ---------------------------------------------------------------------------------------------

from mix_engine import dsp
from mix_engine.session import Session, format_clip_windows, normalize_clip_windows
from orchestrator import daw_bridge


def _wire(*windows: tuple) -> str:
    """The fact as the DAW sends it: at:from:for:gain:fade_in:fade_out, one clip after another."""
    return ";".join(":".join(f"{v:.4f}" if i != 3 else f"{v:.2f}" for i, v in enumerate(w))
                    for w in windows)


def _ramp_project(folder, seconds: float = 8.0) -> np.ndarray:
    """One stem whose sample VALUE says where in the file it is: 0.1 at 1 s, 0.8 at 8 s."""
    audio = (np.arange(int(seconds * SR)) / SR / 10.0).astype(np.float64)
    sf.write(folder / "part.wav", audio, SR, subtype="FLOAT")
    return audio


def test_the_wire_form_is_read_bounded_and_canonical():
    got = normalize_clip_windows(_wire((0.0, 4.0, 2.0, 0.0, 0.0, 0.0), (2.0, 0.0, 2.0, -6.0, 0.015, 0.015)))
    assert got == [{"at_s": 0.0, "from_s": 4.0, "for_s": 2.0, "gain_db": 0.0, "fade_in_s": 0.0, "fade_out_s": 0.0},
                   {"at_s": 2.0, "from_s": 0.0, "for_s": 2.0, "gain_db": -6.0, "fade_in_s": 0.015, "fade_out_s": 0.015}]
    assert normalize_clip_windows("-") is None and normalize_clip_windows("") is None
    assert normalize_clip_windows(None) is None
    # anything that is not six finite numbers a clip is not a report: nothing is half-kept
    for bad in ("1:2:3", "a:b:c:d:e:f", "0:0:2:0:0:0;nan:0:1:0:0:0", "0:0:0:0:0:0", "0:0:-1:0:0:0"):
        assert normalize_clip_windows(bad) is None, bad
    # a fade is held inside its clip, a gain inside the clip-gain range, a list inside its cap
    tight = normalize_clip_windows(_wire((0.0, 0.0, 1.0, 99.0, 5.0, 5.0)))[0]
    assert tight["gain_db"] <= 24.0 and tight["fade_in_s"] + tight["fade_out_s"] <= 1.0 + 1e-9
    many = ";".join(_wire((float(i), 0.0, 0.5, 0.0, 0.0, 0.0)) for i in range(5000))
    assert normalize_clip_windows(many) is None           # past the cap: not held, never truncated


def test_a_stem_of_several_clips_renders_as_it_is_arranged(tmp_path):
    ramp = _ramp_project(tmp_path)
    session = Session(tmp_path)
    # the file's seconds 4-6 play first, then - after a second of nothing - its seconds 0-2, 6 dB down
    session.user_edit("part", {"clips": _wire((0.0, 4.0, 2.0, 0.0, 0.0, 0.0), (3.0, 0.0, 2.0, -6.0, 0.0, 0.0))})
    audio, rate = dsp.render_stem(Session(tmp_path), "part")
    mono = audio[:, 0]
    want = np.zeros(5 * SR)
    want[: 2 * SR] = ramp[4 * SR: 6 * SR]
    want[3 * SR: 5 * SR] = ramp[: 2 * SR] * 10 ** (-6.0 / 20.0)
    assert rate == SR and mono.shape[0] == want.shape[0], mono.shape
    assert np.max(np.abs(mono - want)) < 1e-6
    assert np.array_equal(audio[:, 0], audio[:, 1])                      # a mono file plays from both


def test_a_clips_fades_are_in_the_render(tmp_path):
    sf.write(tmp_path / "part.wav", np.full(4 * SR, 0.5), SR, subtype="FLOAT")
    session = Session(tmp_path)
    session.user_edit("part", {"clips": _wire((1.0, 0.0, 2.0, 0.0, 0.5, 0.25))})
    mono = dsp.render_stem(Session(tmp_path), "part")[0][:, 0]
    assert abs(mono[int(1.25 * SR)] - 0.25) < 1e-3            # half-way up a half-second fade in
    assert abs(mono[int(2.0 * SR)] - 0.5) < 1e-6              # the body
    assert abs(mono[int(2.875 * SR)] - 0.25) < 1e-3           # half-way down a quarter-second fade out
    assert np.all(mono[:SR] == 0.0) and mono.shape[0] == 3 * SR


def test_the_single_window_and_no_window_paths_are_what_they_were(tmp_path):
    ramp = _ramp_project(tmp_path)
    whole = dsp.render_stem(Session(tmp_path), "part")[0]
    assert np.max(np.abs(whole[:, 0] - ramp)) < 1e-6
    session = Session(tmp_path)
    session.user_edit("part", {"clip_at_s": 2.0, "clip_from_s": 1.0, "clip_for_s": 3.0})
    one = dsp.render_stem(Session(tmp_path), "part")[0][:, 0]
    assert one.shape[0] == 5 * SR and np.max(np.abs(one[2 * SR:] - ramp[SR: 4 * SR])) < 1e-6
    assert "clips" not in Session(tmp_path).stems["part"]


def test_one_fact_or_the_other_never_both(tmp_path):
    _ramp_project(tmp_path)
    session = Session(tmp_path)
    session.user_edit("part", {"clip_at_s": 2.0, "clip_from_s": 1.0, "clip_for_s": 3.0})
    session.user_edit("part", {"clips": _wire((0.0, 4.0, 2.0, 0.0, 0.0, 0.0), (3.0, 0.0, 2.0, 0.0, 0.0, 0.0)),
                               "clip_at_s": 0.0, "clip_from_s": 0.0, "clip_for_s": 0.0})
    stem = Session(tmp_path).stems["part"]
    assert len(stem["clips"]) == 2 and "clip" not in stem
    # the track goes back to one plain clip: the DAW says "no list" and the one window
    session = Session(tmp_path)
    session.user_edit("part", {"clips": "-", "clip_at_s": 2.0, "clip_from_s": 1.0, "clip_for_s": 3.0})
    stem = Session(tmp_path).stems["part"]
    assert "clips" not in stem and stem["clip"] == {"at_s": 2.0, "from_s": 1.0, "for_s": 3.0}


def test_where_the_stem_ends_is_where_its_last_clip_ends(tmp_path):
    _ramp_project(tmp_path)
    session = Session(tmp_path)
    assert abs(session.stem_end_s("part") - 8.0) < 1e-6
    session.user_edit("part", {"clips": _wire((0.0, 4.0, 2.0, 0.0, 0.0, 0.0), (10.0, 0.0, 2.5, 0.0, 0.0, 0.0))})
    assert abs(Session(tmp_path).stem_end_s("part") - 12.5) < 1e-6


def test_the_reply_tells_the_daw_what_the_brain_holds(tmp_path):
    _ramp_project(tmp_path)
    session = Session(tmp_path)
    sent = _wire((0.0, 4.0, 2.0, 0.0, 0.0, 0.0), (3.0, 0.0, 2.0, -6.0, 0.015, 0.015))
    session.user_edit("part", {"clips": sent})
    row = daw_bridge.strip_state(Session(tmp_path)._state)["part"]
    assert row["clips"] == sent                    # the same text: the DAW compares it as it is
    session = Session(tmp_path)
    session.user_edit("part", {"clips": "-"})
    assert daw_bridge.strip_state(Session(tmp_path)._state)["part"]["clips"] == ""
    assert daw_bridge.arranged_held(tmp_path) == set()
    session = Session(tmp_path)
    session.user_edit("part", {"clips": sent})
    assert daw_bridge.arranged_held(tmp_path) == {"part"}
    assert daw_bridge.arranged_held(tmp_path / "no-such-project") == set()


def test_the_arrangement_is_what_the_agent_measures(tmp_path):
    """The rehearsal in miniature: a song whose drums sit out bars 5-8. Arranged with that
    break FIRST, the bar map is the arrangement's - it used to be the file's."""
    sf.write(tmp_path / "drums.wav", _kit_by_bars("KKKK....KKKK", bar1_s=0.0), SR)
    session = Session(tmp_path)
    bar = 2.0                                                  # 120 BPM
    session.user_edit("drums", {"clips": _wire((0.0, 4 * bar, 4 * bar, 0.0, 0.0, 0.0),    # the break
                                               (4 * bar, 0.0, 4 * bar, 0.0, 0.0, 0.0))})  # then bars 1-4
    audio, rate = dsp.render_stem(Session(tmp_path), "drums")
    got = analysis.analyze_activity({"drums": audio}, rate, bpm=120.0, bar1_s=0.0)
    assert _spans(got, "drums") == [("out", 1, 4), ("in", 5, 8)]


def _edit_track(name: str, clips: list[dict]) -> dict:
    return {"name": name, "track_kind": "audio", "muted": False, "clips": clips}


def test_a_track_the_brain_holds_as_arranged_is_no_blind_spot():
    moved = [{"kind": "audio", "source": "a.wav", "start_s": 0.0, "offset_s": 8.0},
             {"kind": "audio", "source": "a.wav", "start_s": 8.0, "offset_s": 0.0}]
    edit = {"tracks": [_edit_track("held", moved), _edit_track("not_held", moved),
                       _edit_track("warped", [{"kind": "audio", "source": "b.wav", "warp": True}])]}
    named = lambda spots: next((s["tracks"] for s in spots if s["what"] == "arranged_audio"), [])
    assert named(daw_bridge.render_blind_spots(edit)) == ["held", "not_held", "warped"]
    assert named(daw_bridge.render_blind_spots(edit, arranged_held={"held"})) == ["not_held", "warped"]


# ---------------------------------------------------------------------------------------------
# 5 - the app's half, as far as it can be read (the legs drive it: --stem-clip-windows, --key-shown)
# ---------------------------------------------------------------------------------------------

from pathlib import Path

from mix_engine import session as engine_session

_REPO = Path(__file__).resolve().parents[2]
_APP = (_REPO / "daw" / "Source" / "App.cpp").read_text(encoding="utf-8")
_DAWSESSION = (_REPO / "daw" / "Source" / "DawSession.h").read_text(encoding="utf-8")


def test_both_ends_cap_the_list_at_the_same_number_and_clamp_the_gain_alike():
    cap = re.search(r"constexpr int kMaxStemClipWindows = (\d+);", _DAWSESSION)
    assert cap and int(cap.group(1)) == engine_session.CLIP_WINDOWS_MAX
    lo, hi = engine_session.CLIP_GAIN_DB_RANGE
    assert f"juce::jlimit ({lo}, {hi}, w.gain)" in _DAWSESSION
    assert f'numArg (args, "gain_db", {lo}, {hi}, g, why)' in _APP          # clip.gain's own bounds


def test_every_field_the_fact_report_sends_is_known_as_a_fact():
    """A field `noteEngineFacts` sends and `coeditIsFactReport` does not know makes a queued
    report read as a user's move, which holds a reply's pan and mute back."""
    body = _APP[_APP.index("int noteEngineFacts()"):_APP.index("int noteEngineFactsForConsole()")]
    sent = set(re.findall(r'"[?&](\w+)="', body))
    known_at = _APP.index("static bool coeditIsFactReport")
    known = set(re.findall(r'name != "(\w+)"', _APP[known_at:known_at + 900]))
    assert "clips" in sent and sent <= known, sorted(sent - known)
    assert "clips" in engine_session.ENGINE_FACT_FIELDS


def test_the_shown_key_is_one_rule_at_every_surface():
    """The cell, the agent's snapshot, the browser's chip and the console line ask one function;
    only the transposing pick still asks whether a song key was measured."""
    assert "bool detectedKeyIsShown() const" in _APP
    lines = [line for line in _APP.splitlines() if "detectedKey.isNotEmpty() ? detectedKey" in line]
    assert lines == [], lines


def test_the_red_runs_switch_is_gone():
    for text in (_APP, _DAWSESSION):
        assert "showcaseOld" not in text and "ZAMMOTH_SHOWCASE_OLD" not in text


def test_the_brains_own_door_takes_the_list_the_app_sends(tmp_path):
    """Found by reopening the rehearsal project on the fixed build: the app reported the list,
    the leg (a stand-in brain) and the tests above (the session, called directly) were green,
    and the brain's co-edit handler - which copies a fixed set of fields out of the report -
    dropped it. This goes through that handler, with the report as the app writes it."""
    from urllib.parse import quote

    _ramp_project(tmp_path)
    Session(tmp_path)
    sent = _wire((0.0, 4.0, 2.0, 0.0, 0.0, 0.0), (3.0, 0.0, 2.0, -6.0, 0.015, 0.015))
    facts = ("/api/set_stem?stem=part&delay_ms=0.00&vca_db=0.00&output=master"
             "&clip_at_s=0.0000&clip_from_s=0.0000&clip_for_s=0.0000&frozen=%3Alive&clips=")
    daw_bridge.apply_coedits(tmp_path, [facts + quote(sent, safe="")])
    stem = Session(tmp_path).stems["part"]
    assert format_clip_windows(stem.get("clips")) == sent and "clip" not in stem, stem
    daw_bridge.apply_coedits(tmp_path, [facts + "-"])
    assert "clips" not in Session(tmp_path).stems["part"]


def test_every_fact_the_app_reports_is_read_by_the_brains_door():
    """The same class, pinned: each field `noteEngineFacts` sends is one the handler reads."""
    body = _APP[_APP.index("int noteEngineFacts()"):_APP.index("int noteEngineFactsForConsole()")]
    sent = set(re.findall(r'"[?&](\w+)="', body)) - {"stem"}
    bridge = (_REPO / "orchestrator" / "daw_bridge.py").read_text(encoding="utf-8")
    handler = bridge[bridge.index("# What only the engine knows (D2·R3 / D2·R4)"):bridge.index('elif endpoint == "restore_routing"')]
    read = set(re.findall(r'"(\w+)" in q', handler)) | set(re.findall(r'"(clip_\w+)"', handler))
    assert sent <= read, sorted(sent - read)
