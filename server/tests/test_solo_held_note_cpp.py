"""A silenced track keeps playing, silently (2026-10-17) - what a leg cannot hold, pinned.

James: "when turning solo button on a track that isnt solo while app is playing, it waits until
the next note hits to hear audio". The engine skipped a muted or soloed-out track (its clip MIDI
cleared, its synth not run) unless `shouldProcessMutedTracks()` said otherwise, and our hook said so
only while a return was soloed. The behaviour is driven by the console leg `--solo-held-note`
(daw/Source/SoloResumeLegs.h, live: the instrument node's meter). Here: the hook is true in every
live graph and false only inside a render, the render bracket counts, and a silenced node's
meter feed reads what is heard of it.

Record: docs/10-daw.md, *A silenced track keeps playing, silently*.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "daw" / "Source"
SESSION_H = (SRC / "DawSession.h").read_text(encoding="utf-8")
NODE_H = (SRC / "MonitorChainPlugin.h").read_text(encoding="utf-8")
MAIN = (SRC / "Main.cpp").read_text(encoding="utf-8")
APP = (SRC / "App.cpp").read_text(encoding="utf-8")

pytestmark = pytest.mark.reads_cpp


def _body(source: str, start: str, end: str) -> str:
    i = source.index(start)
    return source[i: source.index(end, i)]


def test_a_silenced_track_is_processed_in_every_live_graph():
    body = _body(SESSION_H, "bool shouldProcessMutedTracks() override", "\n    }\n")
    # true unless a render is being built - or a return is soloed, which a render still needs
    assert "rendersRunning.load" in body and "== 0" in body, body
    assert "returnSoloed.load" in body, body
    assert "std::atomic<int> rendersRunning" in SESSION_H


def test_every_render_the_app_brackets_is_counted_and_let_go():
    body = _body(SESSION_H, "struct ScopedDetachedRender", "\n};\n")
    up = body.index("rendersRunning.fetch_add")
    down = body.index("rendersRunning.fetch_sub")
    # counted before the render's graph exists (the constructor's first lines), let go before the
    # live graph is allocated again (that graph processes silenced tracks)
    assert up < body.index("MessageManagerLock mml"), "count before anything else in the constructor"
    assert down < body.index("ensureContextAllocated"), "let go before the live context comes back"
    # renderEditToFile is the door every app render passes, and it holds the bracket
    door = _body(SESSION_H, "inline bool renderEditToFile (te::Engine& engine, te::Edit& edit, const juce::File& out,\n                              const RenderOptions& opt)\n{", "\n}\n")
    assert "ScopedDetachedRender detached (engine, edit);" in door


def test_a_silenced_nodes_meter_reads_nothing():
    assert re.search(r"std::atomic<Track\*> audibleSource", NODE_H)
    assert "audibleSource.store (ownerAudioTrack" in NODE_H          # initialise
    assert "audibleSource.store (nullptr" in NODE_H                  # deinitialise
    feed = _body(NODE_H, "if (auto* t = audibleSource.load", "meterFeed.push (view")
    assert "shouldBePlayed()" in feed and "meterGainL = meterGainR = 0.0f" in feed, feed


def test_the_leg_is_registered():
    assert 'flag == "--solo-held-note"' in MAIN and "runSoloHeldNote" in MAIN
    assert '#include "SoloResumeLegs.h"' in APP
