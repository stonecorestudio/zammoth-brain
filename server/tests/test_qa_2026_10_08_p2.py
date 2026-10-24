"""User-style QA of 2026-10-08, the P2 batch - the brain's half.

D4. `runs/session_digest.jsonl` kept the first paragraph of a turn's WHOLE reply. A turn that
uses tools speaks more than once - a line of intent before the tools, the answer after them -
and the reply is those parts joined, so the digest of "is anything too loud?" was "The project
has changed since last time ... I'll measure them before I say anything." The next launch's
"Where we left off" then quoted that back to the engineer as "Your last reply then".

Record: docs/audits/user-qa-2026-10-08/QA-log.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from orchestrator import daw_bridge


@pytest.fixture
def quiet_project(tmp_path: Path) -> Path:
    sr = 48000
    t = np.arange(sr * 2) / sr
    for name, f in (("stem_a", 220.0), ("stem_b", 440.0)):
        sf.write(tmp_path / f"{name}.wav", 0.1 * np.sin(2 * np.pi * f * t), sr)
    return tmp_path


INTENT = ("The project has changed since last time: there are four stems on the timeline now. "
          "I'll measure them before I say anything.")
ANSWER = ("Yes - the bass is about 10 dB hot against the rest, and the master limiter is "
          "working 54 % of the time.")


class SpeaksTwice:
    """The chat model's stand-in: a line of intent, a tool, then the answer - the shape of
    every turn that measures before it speaks."""

    def __init__(self, parts):
        self._parts = parts

    async def turn(self, message, *, tools, context, emit, during_run=False,
                   should_cancel=None, live_context=None):
        for text in self._parts:
            await emit({"type": "chat_message", "text": text})


def _digest(project: Path) -> list[dict]:
    path = project / "runs" / daw_bridge.SESSION_DIGEST_FILE
    return [json.loads(line) for line in path.read_text().splitlines()]


def _turn(project: Path, parts) -> dict:
    daw_bridge._CHAT_SESSIONS[str(project)] = SpeaksTwice(parts)
    try:
        return daw_bridge.chat_turn(project, "is anything too loud?", mode="agent",
                                    use_llm=True, edit_state=lambda: {})
    finally:
        daw_bridge._CHAT_SESSIONS.clear()


def test_the_digest_of_a_turn_that_spoke_twice_is_its_answer(quiet_project):
    out = _turn(quiet_project, [INTENT, ANSWER + "\n\nWant me to pull the bass down?"])
    assert out["reply"].startswith(INTENT), "the feed's reply is still the whole turn"
    kept = _digest(quiet_project)[-1]
    assert kept["reply"] == ANSWER, "the digest kept the opening line, not the answer"
    assert kept["asked"] is True


def test_where_we_left_off_quotes_the_answer(quiet_project):
    from datetime import datetime, timedelta, timezone
    _turn(quiet_project, [INTENT, ANSWER])
    part = daw_bridge.session_digest_part(
        quiet_project, before=datetime.now(timezone.utc) + timedelta(seconds=5))
    assert "Your last reply then:" in part
    assert "10 dB hot" in part and "before I say anything" not in part


def test_a_turn_that_spoke_once_keeps_its_first_paragraph(quiet_project):
    _turn(quiet_project, ["Nothing is too loud.\n\nThe master peaks at -3 dBFS."])
    assert _digest(quiet_project)[-1]["reply"] == "Nothing is too loud."


def test_a_lead_in_line_takes_the_paragraph_it_introduces(quiet_project):
    """An answer that opens "Here is what I measured:" says nothing by its first paragraph."""
    _turn(quiet_project, [INTENT, "Here is what I measured:\n\nBass -8 LUFS, keys -19 LUFS."])
    assert _digest(quiet_project)[-1]["reply"] == (
        "Here is what I measured: Bass -8 LUFS, keys -19 LUFS.")


def test_blank_parts_after_the_answer_do_not_empty_the_digest(quiet_project):
    _turn(quiet_project, [INTENT, ANSWER, "  \n"])
    assert _digest(quiet_project)[-1]["reply"] == ANSWER


def test_the_helper_alone():
    assert daw_bridge.digest_reply([]) == ""
    assert daw_bridge.digest_reply(["", " "]) == ""
    assert daw_bridge.digest_reply(["one\n\ntwo"]) == "one"
    assert daw_bridge.digest_reply(["intent", "answer\n\nmore"]) == "answer"
    long = "x" * 400
    assert len(daw_bridge.digest_reply(["intent", long])) == 240


# --------------------------------------------------------------------------------------------
# D28. The limiter's detector read a peak at FOUR points a sample. A crest that falls between two
# of them was under-read by up to a quarter of a dB on content rich in highs, so the limiter let
# it through: the QA pass's composed song exported at -0.81 dBTP under a -1 dBTP ceiling. Both
# detectors - the native's and this transcription of it - now put a parabola through the three
# points round every local maximum. Same pulses, same bound, on both sides
# (MixAgentDAW --qa-limiter-dense holds the native limiter itself).

def _pulse(offset: float, n: int = 257) -> np.ndarray:
    """A pulse band-limited to nine tenths of the band (21.6 kHz at 48 k) whose crest is 1.0,
    `offset` of a sample after the centre sample. Not the full band: no interpolator of the
    detector's length is flat to the very top, and that roll-off is another matter."""
    k = np.arange(n) - n // 2
    x = 0.9 * np.sinc(0.9 * (k - offset)) * np.blackman(n)
    # the window is centred on the sample, not on the crest: put the crest back at 1.0
    fine = np.linspace(-1.5, 1.5, 6001)
    crest = max(abs(float(np.sum(x * np.sinc(t - k)))) for t in fine)
    return (x / crest)[:, np.newaxis]


# 0.25 and 0.5 fall ON two of the detector's points: the control. The others fall between.
@pytest.mark.parametrize("offset", [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.875])
def test_the_detector_reads_a_crest_between_its_points(offset):
    from mix_engine.tp_detector import intersample_peak
    read = float(intersample_peak(_pulse(offset)).max())
    error_db = 20.0 * np.log10(read)
    assert abs(error_db) <= 0.05, f"a crest {offset} of a sample in reads {error_db:+.3f} dB"


def test_the_detector_never_reads_less_than_the_sample():
    from mix_engine.tp_detector import intersample_peak
    rng = np.random.default_rng(3)
    x = rng.standard_normal((4000, 2))
    assert np.all(intersample_peak(x) >= np.abs(x) - 1e-12)


def test_the_detector_of_silence_is_silence():
    from mix_engine.tp_detector import intersample_peak
    assert float(intersample_peak(np.zeros((64, 2))).max()) == 0.0
    assert intersample_peak(np.zeros((0, 2))).shape == (0, 2)
