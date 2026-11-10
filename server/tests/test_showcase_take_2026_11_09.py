"""What James's showcase take of 2026-11-09 found (a liquid drum-and-bass song made in the real
app by the engineer, recorded; read back from the recording frame by frame).

The audio and paint halves are the console leg `--waveform-zoom-stable`. These are the pins a test
can read: the temporary switches of that leg's red run are gone, the painter that re-cut a waveform
per pixel column is gone, the engineer is told what the take showed it getting wrong, and the two
prompts say it alike.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from orchestrator import chat, operate_eval, prompts

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "daw" / "Source"

pytestmark = pytest.mark.reads_cpp


def _text(name: str) -> str:
    return (SOURCE / name).read_text(encoding="utf-8")


def _code(text: str) -> str:
    """The text with // comments and /* */ blocks taken out."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


# ------------------------------------------------------------------ the red run's switches
def test_the_red_run_switches_of_the_zoom_leg_are_gone():
    for path in SOURCE.rglob("*"):
        if path.suffix not in (".h", ".cpp", ".mm"):
            continue
        body = path.read_text(encoding="utf-8", errors="replace")
        for name in ("ZAMMOTH_OLD_WAVE", "ZAMMOTH_OLD_NAME", "oldColumnsForRedRun", "oldNameForRedRun"):
            assert name not in body, f"{path.name} still holds the red run's switch {name}"


# ------------------------------------------------------------------ the waveform
def test_a_thumbnail_waveform_is_never_one_line_a_pixel_column():
    """The envelope is area-sampled on bins that belong to the file. The painter it replaced drew
    one vertical line a column, each column's window cut from the view's own start - the picture
    was re-cut at every zoom step."""
    code = _code(_text("WaveformPaint.h"))
    start = code.index("inline void drawMonoEnvelope")
    end = code.index("namespace clippaint")
    assert "drawVerticalLine" not in code[start:end]
    assert "drawEnvelope (" in code[start:end]


def test_the_arrangement_hands_the_painter_its_clip_cache_and_the_thumbnails_cell():
    code = _code(_text("ArrangeView.h"))
    call = code[code.index("wavepaint::drawMonoEnvelope (g, *cb->thumb"):]
    call = call[:call.index(";")]
    assert "&cb->waveBins" in call and "256.0 / sr" in call


def test_a_clip_name_starts_where_the_clip_comes_into_view():
    code = _code(_text("ArrangeView.h"))
    strip = code[code.index("void paintHeaderStrip"):]
    strip = strip[:strip.index("\n    }\n")]
    assert "clipNameInset (r)" in strip
    paint = _code(_text("WaveformPaint.h"))
    assert "withTrimmedLeft (juce::jmax (0, leftInset))" in paint


def test_a_section_name_starts_where_the_section_comes_into_view():
    code = _code(_text("ArrangeView.h"))
    assert "const auto inView = r.withLeft (juce::jmax (r.getX(), ss.getX()));" in code
    assert "g.drawText (name, inView.reduced (5, 0)" in code


# ------------------------------------------------------------------ the feed's working line
def test_the_working_line_is_handed_the_turns_phase():
    code = _text("App.cpp")   # (read whole: a "/*" inside one of its string literals defeats a comment strip)
    assert "engineer->setRunStatus (turnBusy ? workingLineText() : juce::String())" in code
    assert "engineer->setRunStatus (turnBusy ? runStatus" not in code
    # a chat turn names its phase, and so does a measure or an export inside it
    assert re.search(r'setPhaseStatus \(chatMode == "ask"', code)
    assert re.search(r'setPhaseStatus \(juce::String::fromUTF8 \(measureOnly', code)


# ------------------------------------------------------------------ the engineer
def _both_prompts():
    return {"ships": chat.DAW_SYSTEM_PROMPT, "whole": prompts.DAW_PROMPT_WHOLE}


@pytest.mark.parametrize("which", ["ships", "whole"])
def test_the_engineer_is_told_the_users_mutes_and_solos_are_decisions(which):
    text = " ".join(_both_prompts()[which].split())
    assert "The user's mutes and solos are decisions." in text
    assert "only un-mute a track or clear a solo that you set yourself in this conversation" in text
    assert "the mix as it sounds NOW" in text
    assert "list EVERY muted and soloed track in the question" in text


@pytest.mark.parametrize("which", ["ships", "whole"])
def test_the_engineer_is_told_an_eq_moves_whole(which):
    text = " ".join(_both_prompts()[which].split())
    assert "moves whole: one call, no `occ`, takes every band" in text


def test_the_two_prompts_say_both_alike():
    def bullet(text: str) -> str:
        start = text.index("  * **The user's mutes and solos are decisions.**")
        return text[start:text.index("  * **A file at another tempo", start)]
    assert bullet(chat.DAW_SYSTEM_PROMPT) == bullet(prompts.DAW_PROMPT_WHOLE)


def test_the_command_says_an_eq_moves_whole_and_answers_how_many_bands():
    grammar = _text("DawCommands.h")
    line = next(l for l in grammar.splitlines() if l.lstrip().startswith('{ "device.move_to_track"'))
    assert "moves WHOLE" in line and "bands_moved" in line
    code = _code(_text("App.cpp"))
    handler = code[code.index("case dawcmd::deviceMoveToTrack:"):]
    handler = handler[:handler.index("case dawcmd::deviceAdd:")]
    assert '"bands_moved"' in handler
    mover = code[code.index("juce::String moveDeviceToTrack"):]
    mover = mover[:mover.index("juce::var groupSelectedDevices")]
    assert "one call takes every band" in mover
    assert "the fader is not a device that moves" in mover


def test_the_eval_grades_both():
    by_text = {c.text: c for c in operate_eval.CORPUS}
    eq = by_text["move lead_vox's EQ over to the bvs track"]
    assert eq.max_commands == 1 and eq.expect == ("device.move_to_track",)
    rows = [d for t in eq.context()["tracks"] if t.get("name") == "lead_vox" for d in t["devices"]]
    assert [d["type"] for d in rows].count("eq") == 2, "the case shows the engineer two eq rows"
    keep = by_text["print the mix as it sounds right now to a wav called rough"]
    assert {"track.set_mute", "track.mute_toggle", "track.solo"} <= set(keep.forbid)
    assert next(t for t in keep.context()["tracks"] if t.get("name") == "lead_vox")["muted"] is True
