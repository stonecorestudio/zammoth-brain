"""Phase 36 — reference matching v2: weighted sets + section targets."""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from mcp.shared.memory import create_connected_server_and_client_session

from mix_engine import analysis
from mix_engine.mcp_server import create_server
from orchestrator.agents import PassContext, RuleAgent, TargetState
from orchestrator.loop import EngineClient
from orchestrator.targets import (REFERENCE_BAND_TOLERANCE_DB,
                                  reference_set_mix_targets,
                                  reference_specs_from_brief)

from test_phase1_acceptance import call

SR = 48_000


def _tone_ref(duration_s: float = 10.0, high_gain: float = 1.0,
              level: float = 0.2) -> np.ndarray:
    t = np.arange(int(duration_s * SR)) / SR
    audio = (np.sin(2 * np.pi * 220.0 * t)
             + high_gain * np.sin(2 * np.pi * 5000.0 * t))
    return (level * audio / np.max(np.abs(audio))).astype(np.float32)


def _two_section(high_second: bool = True) -> np.ndarray:
    t = np.arange(int(16.0 * SR)) / SR
    low = 0.12 * np.sin(2 * np.pi * 220.0 * t)
    high = 0.45 * np.sin(2 * np.pi * 5000.0 * t)
    split = int(8.0 * SR)
    first, second = (low, high) if high_second else (high, low)
    return np.concatenate([first[:split], second[split:]]).astype(np.float32)


def _write_ref(project: Path, name: str, audio: np.ndarray) -> None:
    (project / "refs").mkdir(exist_ok=True)
    sf.write(project / "refs" / name, audio, SR)


@pytest.mark.anyio
async def test_reference_set_weighted_roles_and_cache(tmp_path):
    project = tmp_path / "weighted_refs"
    project.mkdir()
    _write_ref(project, "dark.wav", _tone_ref(high_gain=0.2, level=0.2))
    _write_ref(project, "bright.wav", _tone_ref(high_gain=2.0, level=0.35))

    async with create_connected_server_and_client_session(
            create_server(project)) as client:
        dark = await call(client, "analyze_reference",
                          {"path": "refs/dark.wav"})
        bright = await call(client, "analyze_reference",
                            {"path": "refs/bright.wav"})
        result = await call(client, "analyze_reference_set", {"references": [
            {"path": "refs/dark.wav", "weight": 3.0, "roles": ["tonality"]},
            {"path": "refs/bright.wav", "weight": 1.0,
             "roles": ["tonality", "loudness"]},
        ]})
        again = await call(client, "analyze_reference_set", {"references": [
            {"path": "refs/dark.wav", "weight": 3.0, "roles": ["tonality"]},
            {"path": "refs/bright.wav", "weight": 1.0,
             "roles": ["tonality", "loudness"]},
        ]})

    expected = round((3 * dark["band_energy"]["brilliance"]
                      + bright["band_energy"]["brilliance"]) / 4, 1)
    assert result["band_energy"]["brilliance"] == expected
    assert result["lufs_i"] == bright["lufs_i"]
    assert result == again
    cache = json.loads((project / ".reference_cache.json").read_text())
    assert len(cache["entries"]) == 2


@pytest.mark.anyio
async def test_reference_set_pairs_loud_sections(tmp_path):
    project = tmp_path / "section_refs"
    project.mkdir()
    _write_ref(project, "ref_a.wav", _two_section(high_second=True))
    _write_ref(project, "ref_b.wav", _two_section(high_second=False))
    mix_sections = analysis.analyze_sections(_two_section(high_second=True),
                                             SR, {})

    async with create_connected_server_and_client_session(
            create_server(project)) as client:
        result = await call(client, "analyze_reference_set", {
            "references": [
                {"path": "refs/ref_a.wav", "weight": 2.0,
                 "roles": ["tonality"]},
                {"path": "refs/ref_b.wav", "weight": 1.0,
                 "roles": ["tonality"]},
            ],
            "mix_sections": mix_sections.model_dump(),
        })

    assert result["section_fallback"] is False
    target = result["section_targets"][0]
    # mix_index is the loudness rank now (0 = loudest) — stable across renders;
    # it should still describe the actual loudest mix section.
    loud = max(mix_sections.sections, key=lambda s: s.lufs_i or -120.0)
    assert target["mix_index"] == 0
    assert target["mix_start_s"] == loud.start_s
    assert target["confidence"] == 1.0
    assert {c["path"] for c in target["references"]} == {
        "refs/ref_a.wav", "refs/ref_b.wav"}
    assert target["band_energy"]["brilliance"] > -30.0


@pytest.mark.anyio
async def test_section_band_metric_measures_from_loop(tmp_path):
    project = tmp_path / "section_metric"
    project.mkdir()
    sf.write(project / "song.wav", _two_section(high_second=True), SR)

    async with create_connected_server_and_client_session(
            create_server(project)) as client:
        sections = await call(client, "analyze_sections")
        loud_index = max(enumerate(sections["sections"]),
                         key=lambda item: item[1]["lufs_i"] or -120.0)[0]
        # rank 0 = loudest section; resolution re-ranks the live sections, so a
        # pinned rank tracks the loudest section even if its position moves.
        metric = "section_band_energy.0.brilliance"
        values = await EngineClient(client).measure_all([metric])

    assert values[metric] == sections["sections"][loud_index][
        "band_energy"]["brilliance"]


def test_reference_set_targets_and_rule_agent_section_match_eq():
    specs = reference_specs_from_brief(
        "match refs/a.wav plus refs/b.mp3 against the chorus")
    assert [s["path"] for s in specs] == ["refs/a.wav", "refs/b.mp3"]

    targets = reference_set_mix_targets({
        "section_targets": [{
            "mix_index": 1,
            "band_energy": {"brilliance": -8.0},
        }],
        "band_energy": {"brilliance": -20.0},
    })
    assert targets[0].metric == "section_band_energy.1.brilliance"
    assert targets[0].tolerance == REFERENCE_BAND_TOLERANCE_DB

    agent = RuleAgent()
    proposal = agent.propose(PassContext(
        pass_number=1,
        metrics={},
        targets=[TargetState(metric=targets[0].metric, unit="dB",
                             current=-14.0, goal=-8.0, distance=5.0)],
        stem_chains={},
    ))[0]

    assert proposal["actions"] == [{
        "type": "match_eq",
        "stem_id": "master",
        "band_gains_db": {"brilliance": 6.0},
    }]
    assert proposal["predicted_impact"][0]["metric"] == (
        "section_band_energy.1.brilliance")


def test_match_eq_boost_caps_resonant_reference_bands():
    agent = RuleAgent()
    proposal = agent.propose(PassContext(
        pass_number=1,
        metrics={"harshness_db": -6.0,
                 "resonances": [{"freq_hz": 5200, "prominence_db": 10.0,
                                  "q": 8.0}]},
        targets=[TargetState(metric="band_energy.brilliance", unit="dB",
                             current=-16.0, goal=-8.0, distance=7.0)],
        stem_chains={},
    ))[0]

    assert proposal["actions"][0]["band_gains_db"]["brilliance"] == 2.0
