"""User QA 2026-10-08, the manual's sentences the app did not match (and the two leads).

Each test reads the FACT out of the source and then the sentence that describes it, as
test_reader_p3_w.py does: when the code moves, the sentence that went stale is named by the
test that fails.

Record: docs/audits/user-qa-2026-10-08/QA-log.md, *The two leads, and the manual's sentences*.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANUAL = (ROOT / "docs" / "17-user-manual.md").read_text(encoding="utf-8")
SESSION_H = (ROOT / "daw" / "Source" / "DawSession.h").read_text(encoding="utf-8")
COMMANDS_H = (ROOT / "daw" / "Source" / "DawCommands.h").read_text(encoding="utf-8")
DOCK_H = (ROOT / "daw" / "Source" / "DeviceChainView.h").read_text(encoding="utf-8")
LAYOUT_H = (ROOT / "daw" / "Source" / "LaneLayout.h").read_text(encoding="utf-8")
HEADS_H = (ROOT / "daw" / "Source" / "TrackHeadColumn.h").read_text(encoding="utf-8")
CHAIN_H = (ROOT / "native" / "Source" / "MonitorChain.h").read_text(encoding="utf-8")
GRAPHS_H = (ROOT / "native" / "Source" / "DeviceGraphs.h").read_text(encoding="utf-8")
GUARDRAILS = (ROOT / "server" / "mix_engine" / "guardrails.py").read_text(encoding="utf-8")
AUTOMIX = (ROOT / "orchestrator" / "automix.py").read_text(encoding="utf-8")

pytestmark = __import__("pytest").mark.reads_cpp


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


FLAT = _flat(MANUAL)


def _addable(after: str) -> set[str]:
    """The device types one branch of `addableTypesFor` returns: the first brace list after `after`."""
    body = DOCK_H[DOCK_H.index("static juce::StringArray addableTypesFor"):] if "static juce::StringArray addableTypesFor" in DOCK_H \
        else DOCK_H[DOCK_H.index("addableTypesFor ("):]
    body = body[body.index(after):]
    return set(re.findall(r'"([a-z_]+)"', body[body.index("{"): body.index("}")]))


def test_the_eq_holds_as_many_bands_as_the_manual_says():
    bank = int(re.search(r"kMaxBands\s*=\s*(\d+)", CHAIN_H).group(1))
    brain = int(re.search(r'"eq_bands_max":\s*(\d+)', GUARDRAILS).group(1))
    assert bank == brain == 32
    assert f"entry (up to {bank})" in FLAT
    assert "up to 24" not in FLAT


def test_the_agents_fader_bounds_are_the_faders():
    lo = re.search(r"kVcaMinDb\s*=\s*(-?[\d.]+)", COMMANDS_H).group(1)
    hi = re.search(r"kVcaMaxDb\s*=\s*(-?[\d.]+)", COMMANDS_H).group(1)
    assert (float(lo), float(hi)) == (-60.0, 6.0)
    assert re.search(r"FADER_DB_RANGE\s*=\s*\(-60(\.0)?,\s*6(\.0)?\)", GUARDRAILS)
    level = FLAT[FLAT.index("## Level bounds"):][:400]
    assert "−60…+6 dB" in level and "+24" not in level and "−96" not in level


def test_h_fits_down_to_the_floor_the_manual_names():
    floor = int(re.search(r"kFitMin\s*=\s*(\d+)", LAYOUT_H).group(1))
    assert f"never made shorter than {floor} px" in FLAT


def test_the_lane_height_keys_are_the_ones_bound():
    for key in ("view.track_height_inc", "view.track_height_dec"):
        assert key in COMMANDS_H
    row = re.search(r'"view\.track_height_inc",[^\n]*', COMMANDS_H).group(0)
    assert "'='" in row, "the grow key moved: the manual says ⌥="
    assert "⌥+ / ⌥−" not in MANUAL and "**⌥= / ⌥−**" in MANUAL


def test_appendix_a_lists_the_four_bindings_it_left_out():
    appendix = MANUAL[MANUAL.index("# Appendix A"):]
    appendix = appendix[: appendix.index("\n# ", 10)]
    for key, char, chord in (("record.retrospective", "'R'", "⌥⌘R"), ("view.computer_keyboard", "'K'", "⌘K"),
                             ("view.session_rail", "'S'", "⌥⌘S"), ("view.toggle_engineer", "'E'", "⌥⌘E")):
        row = re.search(r'"%s",[^\n]*' % re.escape(key), COMMANDS_H).group(0)
        assert char in row and "commandModifier" in row, f"{key} is no longer bound to {chord}"
        assert f"| {chord} |" in appendix, f"Appendix A does not list {chord}"


def test_width_and_duck_go_where_the_manual_says():
    master, ret, stem = _addable('selectedName == "master"'), _addable('startsWith ("return:")'), _addable("juce::StringArray t {")
    assert "width" in master and "width" in stem and "width" not in ret
    assert "duck" in stem and "duck" not in master and "duck" not in ret
    assert 't.removeString ("duck")' in DOCK_H            # a group's list
    assert "Width** (the device) is Master-only" not in MANUAL
    assert "**Stereo Width** goes on the Master, a track or a group" in FLAT
    assert "Master-only devices refuse" not in FLAT and "master-only Width device" not in FLAT


def test_the_add_tile_trails_the_boxes_and_its_menu_is_by_family():
    assert "addDeviceTypesByFamily" in DOCK_H
    tab = FLAT[FLAT.index('**The identity tab ("+")**'):][:500]
    assert "after the last device box" in tab and "grouped by family" in tab
    assert "leading tile" not in tab and "alphabetical menu of every device" not in tab


def test_the_fader_leads_the_chain_in_every_sentence():
    assert "Hosted plug-ins sit at the end of the chain, before the fader" not in FLAT
    assert "then the channel fader — so a hosted compressor" not in FLAT
    assert "The channel fader leads the chain" in FLAT
    assert "The **gain** stage that leads every chain is the channel fader itself" in FLAT


def test_the_comp_scope_command_is_said_to_flip_nothing_on_this_face():
    assert "The R15 face shows BOTH plots at" in GRAPHS_H
    assert "plug-in's rack view" not in FLAT
    assert "`view.comp_scope` has nothing to flip" in FLAT


def test_spectral_denoise_is_said_to_be_left_out():
    assert "left_out" in AUTOMIX
    assert "runs inside the engineer's AI-cleanup pass" not in FLAT


def test_the_head_send_knob_resets_on_a_double_click():
    assert "k->setDoubleClickReturnValue (true, -60.0);" in HEADS_H
    assert "double-click types a value; ⌥-double-click turns the send off" not in FLAT


def test_a_vca_is_built_in_the_tracks_own_order():
    body = SESSION_H[SESSION_H.index("inline te::FolderTrack* createVca ("):]
    body = body[: body.index("\n}\n")]
    loop = body[body.index("juce::Array<te::AudioTrack*> members;"):]
    assert loop.lstrip().split("\n")[1].strip().startswith("for (auto* at : te::getAudioTracks (edit))"), \
        "createVca collects its members in the order they were NAMED again: a selection clicked bottom-first swaps them"
    assert "The members keep the order they had among themselves" in FLAT


def test_the_manual_says_where_a_capture_over_existing_clips_lands():
    assert "return std::ceil ((contentEnd - earliest) / bar) * bar;" in SESSION_H
    assert "moved later by whole bars until it is clear of every such clip" in FLAT


def test_the_small_words():
    assert "status bar" not in MANUAL
    assert "Read/Touch/Latch/Write mode zoo" not in FLAT
    assert "**OPUS 5** — the model word" not in MANUAL and "every current model, grouped by family" not in FLAT
    assert "a confirmed agent action" not in FLAT
    assert "bottom-left of the arrangement" not in FLAT
    assert "unclickable" not in FLAT
    assert "2026-08-17" not in MANUAL.split("\n")[5]
