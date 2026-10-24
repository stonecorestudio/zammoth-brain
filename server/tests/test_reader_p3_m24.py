"""Reader P3s, batch M24 - the engineer panel, the wire, lifetimes, the licence on screen
(2026-10-07).

The behaviour is driven by the console legs in `daw/Source/AuditReaderP3LegsM24.h`. This file
holds what a leg cannot: rules about the SOURCE that a later edit could quietly undo.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = (ROOT / "daw" / "Source" / "App.cpp").read_text(encoding="utf-8")
PANEL = (ROOT / "native" / "Source" / "EngineerPanel.h").read_text(encoding="utf-8")
CLIENT = (ROOT / "daw" / "Source" / "BrainClient.h").read_text(encoding="utf-8")


def _product(text: str) -> str:
    """App.cpp without the console legs at its end."""
    cut = text.find("#ifdef MIXAGENT_DAW_CONSOLE\n#include")
    return text if cut < 0 else text[:cut]


def test_the_run_thread_never_writes_the_pending_action():
    """C5.R8. `pendingAction` is a juce::String the 30 Hz timer reads (`agentTurnBusy`) and the
    command handlers read; the RunThread assigned it inside performAction for a queued routing
    job. A ref-counted string written on one thread while another copies it is a read of a freed
    holder. performAction works on a LOCAL (`action`), and the member has one writer:
    startAction, on the message thread."""
    body = APP[APP.index("    void performAction()"):]
    body = body[: body.index("\n    }\n")]
    assert "pendingAction = " not in body, "performAction assigns the member the message thread reads"
    assert re.search(r"auto action = pendingAction;|juce::String action \(pendingAction\);|juce::String action = pendingAction;", body), \
        "performAction takes the action into a local at its top"
    assert "pendingAction ==" not in body and "pendingAction !=" not in body, \
        "a branch of performAction still tests the member instead of the local"


def test_no_worker_thread_is_stopped_with_a_kill():
    """C5.R6. `Thread::stopThread (ms)` KILLS a thread that has not ended by then (a deferred
    pthread_cancel) and answers false; every caller ignored the answer and went on to free what
    the thread was inside. A worker is joined with `joinWorker`, which never kills, and a
    project whose worker is still running is set aside (`DawComponent::retire`)."""
    product = _product(APP)
    for thread in ("runThread", "exportThread"):
        assert f"{thread}.stopThread (" not in product, f"{thread}.stopThread kills on a timeout - use joinWorker"
    assert "static bool joinWorker (juce::Thread&" in product


def test_every_open_project_leaves_through_retire():
    """C5.R6. The two places the app drops a DawComponent hand it to `retire`, which destroys it
    only once its workers have ended."""
    product = _product(APP)
    assert "openProjects.clear();" not in product, "shutdown destroys projects without asking their workers"
    erases = [m.start() for m in re.finditer(r"openProjects\.erase \(", product)]
    assert erases, "the reload path no longer erases - re-point this test"
    for at in erases:
        assert "DawComponent::retire (" in product[at - 400: at], "an open project is erased without retire"


def test_what_counts_as_a_failure_is_said_by_the_door_not_the_sentence():
    """E8.R10. The panel's comment claimed one definition shared with App.cpp's setStatus, and
    the two word lists differed. A status that came through errorLine / midTurnStatus is marked
    where it is made; the word test stays only for lines that carry the warning sign or come
    from an old file."""
    assert "has one definition" not in PANEL
    block = PANEL[PANEL.index("static juce::var blockToVar (const BlockType& b)"):]
    block = block[: block.index("auto* o = new juce::DynamicObject();")]
    assert "b.transient" in block, "blockToVar no longer skips a status by the door it came through"


def test_a_new_connection_forgets_the_last_ones_half_character():
    """S3.R12 a. `utf8Pending` holds the lead bytes of a character whose tail has not arrived;
    both doors that start a connection cleared `inbuf` and left it."""
    for door in ("bool connectOnly (int port", "bool launch (const juce::StringArray& program"):
        body = CLIENT[CLIENT.index(door):][:900]
        assert "utf8Pending.clear();" in body, f"{door.split('(')[0].strip()} keeps the old connection's pending bytes"
