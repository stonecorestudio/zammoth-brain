"""Phase 55, step 2c: what is measured on a kept render is kept, and a stem that is not in
the mix is not rendered for a read about the mix.

Read in the code and counted on the timed run-through: `analyze_chords`, `analyze_timing`
and `analyze_activity` each ran the song's tempo (and key) estimator over the whole master on
every call, although the master itself was kept; and `describe_region` rendered every stem in
the session - muted and deleted ones included - to ask who masks the one it was asked about
(the arrange turn's 18 stem renders were ten muted originals being resampled), measuring each
pair with twelve block-FFT passes where two are distinct.
"""
from __future__ import annotations

import json
from pathlib import Path

import anyio
import numpy as np
import soundfile as sf
from mcp.shared.memory import create_connected_server_and_client_session

from mix_engine import analysis, dsp
from mix_engine.mcp_server import create_server
from mix_engine.session import Session
from tests.test_phase54_render_once import kept  # noqa: F401

SR = 44100


def _project(tmp_path: Path) -> Path:
    """A click (so the song has a tempo), a pad, and two tones a noise bed buries."""
    proj = tmp_path / "proj"
    proj.mkdir()
    n = 6 * SR
    t = np.arange(n) / SR
    click = np.zeros(n, dtype=np.float32)
    for beat in range(12):                       # 120 BPM
        at = int(beat * 0.5 * SR)
        click[at:at + 400] = 0.6 * np.hanning(400)
    rng = np.random.default_rng(7)
    stems = {
        "drums": click,
        "pad": (0.2 * (np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 277 * t))).astype(np.float32),
        "vocal": (0.01 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32),
        "noise": (0.5 * rng.standard_normal(n)).astype(np.float32),
        "old_take": (0.5 * rng.standard_normal(n)).astype(np.float32),
        "deleted": (0.5 * rng.standard_normal(n)).astype(np.float32),
    }
    for name, audio in stems.items():
        sf.write(proj / f"{name}.wav", audio, SR)
    Session(proj)
    state = json.loads((proj / "session.json").read_text())
    state["stems"]["old_take"]["muted"] = True
    state["stems"]["deleted"]["excluded"] = True
    (proj / "session.json").write_text(json.dumps(state))
    return proj


def _call(proj: Path, calls: list[tuple[str, dict]]) -> list[dict]:
    async def ask() -> list[dict]:
        out = []
        async with create_connected_server_and_client_session(
                create_server(proj)._mcp_server) as engine:
            for name, args in calls:
                res = await engine.call_tool(name, args)
                assert not res.isError, res.content
                payload = res.structuredContent
                out.append(payload.get("result", payload))
        return out
    return anyio.run(ask)


def _count(monkeypatch, name: str) -> list[int]:
    calls = [0]
    real = getattr(analysis, name)

    def counting(*a, **k):
        calls[0] += 1
        return real(*a, **k)
    monkeypatch.setattr(analysis, name, counting)
    return calls


def test_the_songs_tempo_and_key_are_measured_once_for_a_kept_master(tmp_path, monkeypatch, kept):
    proj = _project(tmp_path)
    tempo, key = _count(monkeypatch, "song_tempo"), _count(monkeypatch, "song_key")
    first, second, timing, activity = _call(proj, [
        ("analyze_chords", {"stem_id": "pad"}), ("analyze_chords", {"stem_id": "pad"}),
        ("analyze_timing", {"stem_id": "drums"}), ("analyze_activity", {"stem_ids": ["drums"]})])
    assert tempo[0] == 1 and key[0] == 1, f"tempo x{tempo[0]}, key x{key[0]} on one kept master"
    assert second == first, "the kept figures are the measured ones"
    assert activity["bpm"] and timing is not None


def test_with_the_caches_off_every_read_measures_as_before(tmp_path, monkeypatch):
    """Control: the tests and the benchmark run with the caches off, and count."""
    proj = _project(tmp_path)
    tempo = _count(monkeypatch, "song_tempo")
    _call(proj, [("analyze_chords", {"stem_id": "pad"}), ("analyze_chords", {"stem_id": "pad"})])
    assert tempo[0] == 2


def test_describe_region_asks_only_the_stems_that_sound(tmp_path, kept):
    proj = _project(tmp_path)
    before = dsp.RENDER_COUNTS["stem"]
    (region,) = _call(proj, [("describe_region", {"stem_id": "vocal", "start_s": 1.0, "end_s": 5.0})])
    rendered = dsp.RENDER_COUNTS["stem"] - before
    assert rendered == 4, f"{rendered} stems rendered: the four in the mix, not all six"
    named = {m["stem_id"] for m in region["masked_by"]} | {m["stem_id"] for m in region["masks"]}
    assert "noise" in named, f"vacuity: the live masker must be found: {region['masked_by']}"
    assert not named & {"old_take", "deleted"}, f"a stem that does not sound was reported: {named}"


def test_a_muted_stem_can_still_be_described_itself(tmp_path, kept):
    proj = _project(tmp_path)
    (region,) = _call(proj, [("describe_region", {"stem_id": "old_take", "start_s": 1.0, "end_s": 5.0})])
    assert region["stem_id"] == "old_take" and region["lufs_i"] is not None


def test_sections_report_the_activity_of_the_stems_in_the_mix(tmp_path, kept):
    proj = _project(tmp_path)
    (sections,) = _call(proj, [("analyze_sections", {})])
    stems = set(sections["sections"][0]["stem_activity"])
    assert stems == {"drums", "pad", "vocal", "noise"}


def _reference_maskers(seg, sample_rate, stem_id, others):
    """`_region_maskers` as it was: `analyze_masking` in each direction."""
    masked_by, masks = [], []
    for other_id, other in sorted(others.items()):
        other = analysis._frames_channels(other)
        n = min(seg.shape[0], other.shape[0])
        if n < int(0.4 * sample_rate) or np.max(np.abs(other[:n])) < analysis.SILENCE_PEAK:
            continue
        a, b = seg[:n], other[:n]
        for first, second, first_id, second_id, into in ((a, b, stem_id, other_id, masked_by),
                                                         (b, a, other_id, stem_id, masks)):
            try:
                found = analysis.analyze_masking(first, second, sample_rate,
                                                 stem_a=first_id, stem_b=second_id)
            except ValueError:
                continue
            if found.worst_band_excess_db >= analysis.REGION_MASK_FLOOR_DB:
                into.append((other_id, found.worst_band,
                             analysis._round_db(found.worst_band_excess_db)))
    return (sorted(masked_by, key=lambda m: -m[2]), sorted(masks, key=lambda m: -m[2]))


def test_the_regions_maskers_are_the_ones_analyze_masking_finds(monkeypatch):
    n = 3 * SR
    t = np.arange(n) / SR
    rng = np.random.default_rng(3)
    seg = analysis._frames_channels((0.02 * np.sin(2 * np.pi * 900 * t)).astype(np.float32))
    others = {
        "noise": (0.4 * rng.standard_normal((n, 2))).astype(np.float32),
        "bass": (0.5 * np.sin(2 * np.pi * 60 * t)).astype(np.float32),
        "short": (0.3 * rng.standard_normal(SR // 10)).astype(np.float32),
        "silent": np.zeros(n, dtype=np.float32),
        "dc": np.full(n, 0.2, dtype=np.float32),
        "late": np.concatenate([np.zeros(2 * SR), 0.4 * rng.standard_normal(SR)]).astype(np.float32),
    }
    want = _reference_maskers(seg, SR, "vocal", others)
    built = _count(monkeypatch, "_block_band_power")
    got = analysis._region_maskers(seg, SR, "vocal", others)
    flat = tuple([(m.stem_id, m.worst_band, m.excess_db) for m in side] for side in got)
    assert flat == want and (want[0] or want[1]), "vacuity: something must mask something"
    assert built[0] <= 2 * len(others), f"{built[0]} block-FFT passes for {len(others)} pairs"
