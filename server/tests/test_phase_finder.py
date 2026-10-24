"""The phase finder (`analyze_phase`, 2026-10-13): can the agent FIND a phase problem, tell where
it comes from, and is the move it is handed the one that fixes it.

Every case is built so the answer is known before it is read: one side flipped, a Haas delay, two
stems in opposite polarity, one stem a few milliseconds early, a pad widened past mono-safe by its
own strip width - and the controls, a wide stem that is not a problem and a clean mix, which must
read clean. The rule's thresholds were sized on released records (mix-analysis skill ▸ *Phase*)."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
from mcp.shared.memory import create_connected_server_and_client_session

from mix_engine import dsp, phase
from mix_engine.mcp_server import create_server
from mix_engine.models import ProposalRequest
from mix_engine.session import Session
from test_phase1_acceptance import call

SR = 48000
SECONDS = 12.0


def _noise(seed: int, lo_hz: float, hi_hz: float, seconds: float = SECONDS) -> np.ndarray:
    """Band-limited noise (FFT brick-wall), unit RMS - deterministic."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    spec = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    spec[(freqs < lo_hz) | (freqs >= hi_hz)] = 0.0
    x = np.fft.irfft(spec, n=n)
    return x / np.sqrt(np.mean(x * x))


def _delayed(x: np.ndarray, ms: float) -> np.ndarray:
    d = int(round(ms * SR / 1000.0))
    return np.concatenate([np.zeros(d), x[:len(x) - d]])


def _write(folder, name: str, left: np.ndarray, right: np.ndarray, level: float = 0.15) -> None:
    sf.write(folder / f"{name}.wav", (level * np.stack([left, right], axis=1)).astype(np.float32),
             SR, subtype="FLOAT")


def _pad(seed: int, corr: float) -> tuple[np.ndarray, np.ndarray]:
    """A wide 'pad': two channels sharing `corr` of their power (a decorrelated mid band)."""
    common, own_l, own_r = (_noise(seed + k, 200.0, 4000.0) for k in range(3))
    a, b = np.sqrt(corr), np.sqrt(1.0 - corr)
    return a * common + b * own_l, a * common + b * own_r


def _mono_bed(folder) -> None:
    """A mono kick-and-bass bed: what keeps a real master's lows and centre correlated."""
    low = _noise(90, 40.0, 180.0)
    _write(folder, "bed", low, low, level=0.2)


def _reading(folder, stem_ids=None):
    return phase.phase_reading(Session(folder), stem_ids)


def _apply(session: Session, actions: list[dict]) -> None:
    """The agent's door: a proposal under the guardrails, applied."""
    req = ProposalRequest.model_validate({"rationale": "phase", "predicted_impact": [],
                                          "actions": actions})
    proposal = session.propose(req)
    assert proposal.accepted, proposal.violations
    session.apply(proposal.proposal_id)


def _kinds(reading, target):
    return {(i.kind, i.band) for i in reading.issues if i.target == target}


# ---------------------------------------------------------------- the measurements themselves

def test_the_band_reading_is_plus_one_for_mono_zero_for_unrelated_and_minus_one_anti_phase():
    x = _noise(1, 300.0, 3000.0)
    y = _noise(2, 300.0, 3000.0)
    mono = phase.band_reading(phase.WindowSums(np.stack([x, x], 1), SR))
    loose = phase.band_reading(phase.WindowSums(np.stack([x, y], 1), SR))
    anti = phase.band_reading(phase.WindowSums(np.stack([x, -x], 1), SR))
    for band in ("mid", "high_mid"):
        assert mono[band].corr == 1.0 and mono[band].below_zero_pct == 0.0 and mono[band].mono_db == 0.0
        assert abs(loose[band].corr) < 0.05 and loose[band].mono_db == pytest.approx(-3.0, abs=0.2)
        # unrelated channels are below 0 about half of the windows - which is why a WIDE source
        # is not, on its own, an issue (the lamps light, the record still folds)
        assert 30.0 < loose[band].below_zero_pct < 70.0
        assert anti[band].corr == -1.0 and anti[band].below_zero_pct == 100.0
        assert anti[band].mono_db == phase.PHASE_MONO_DB_FLOOR


def test_a_band_with_nothing_in_it_is_not_judged():
    x = _noise(3, 300.0, 1000.0)
    reading = phase.band_reading(phase.WindowSums(np.stack([x, x], 1), SR))
    assert set(reading) <= {"low_mid", "mid"}, reading.keys()


def test_mono_safe_width_is_the_width_at_which_the_stem_stops_being_too_wide():
    left, right = _pad(10, 0.4)
    sums = phase.WindowSums(phase._stereo(dsp._width(np.stack([left, right], 1), 1.6)), SR)
    safe = phase.mono_safe_width(sums, 1.6)
    assert safe is not None and safe < 1.6
    # ...exact: rendered AT that width the stem passes, a step wider it does not
    at = phase.WindowSums(dsp._width(np.stack([left, right], 1), safe), SR)
    assert phase.mono_safe_width(at, safe) is None
    wider = round(safe + phase.PHASE_WIDTH_STEP, 2)
    over = phase.WindowSums(dsp._width(np.stack([left, right], 1), wider), SR)
    assert phase.mono_safe_width(over, wider) is not None
    # and a stem already inside it gets no number
    assert phase.mono_safe_width(phase.WindowSums(np.stack([left, right], 1), SR), 1.0) is None


def _stretches(anti_share: float, seed: int) -> np.ndarray:
    """A master-like band: mono for most of the song, ANTI-phase for `anti_share` of it, in
    stretches of a few seconds - its whole-song correlation stays well above 0."""
    x = _noise(seed, 300.0, 3000.0)
    n = x.size
    sign = np.ones(n)
    block = 4 * SR
    for start in range(0, n, block):
        sign[start:start + int(block * anti_share)] = -1.0
    return np.stack([x, sign * x], axis=1)


def test_the_master_rule_is_twenty_percent_of_the_time_below_zero():
    """Released masters read at most 15.5 % below 0 in a band; this mix's read 25 %. The rule
    sits between: 30 % of the time anti-phase is an issue, 12 % is not, though both are well
    above 0 over the song."""
    often = phase.band_reading(phase.WindowSums(_stretches(0.30, 100), SR))
    rarely = phase.band_reading(phase.WindowSums(_stretches(0.12, 101), SR))
    for reading in (often, rarely):
        assert all(b.corr > 0.3 for b in reading.values())
    assert all(25.0 < b.below_zero_pct < 35.0 for b in often.values())
    assert all(8.0 < b.below_zero_pct < 16.0 for b in rarely.values())
    assert phase.master_at_risk(_stretches(0.30, 100), SR)
    assert phase.master_at_risk(_stretches(0.12, 101), SR) == []


def test_mono_safe_width_counts_the_time_below_zero_not_only_the_song():
    """Half the song wide, half mono: over the song it reads well above +0.2, but it is below 0
    for a quarter of its time - the width is set by the time share."""
    x = _noise(102, 300.0, 3000.0)
    left, right = _pad(103, 0.0)
    half = x.size // 2
    audio = np.stack([np.concatenate([x[:half], left[half:]]),
                      np.concatenate([x[:half], right[half:]])], axis=1)
    sums = phase.WindowSums(audio, SR)
    reading = phase.band_reading(sums)
    assert all(b.corr > phase.STEM_SAFE_CORRELATION for b in reading.values())
    assert any(b.below_zero_pct >= phase.STEM_SAFE_TIME_PCT for b in reading.values())
    safe = phase.mono_safe_width(sums, 1.0)
    assert safe is not None and safe < 1.0
    at = phase.band_reading(phase.WindowSums(dsp._width(audio, safe), SR))
    assert all(b.below_zero_pct < phase.STEM_SAFE_TIME_PCT for b in at.values())
    wider = phase.band_reading(phase.WindowSums(dsp._width(audio, round(safe + 0.05, 2)), SR))
    assert any(b.below_zero_pct >= phase.STEM_SAFE_TIME_PCT for b in wider.values())


def test_a_haas_delay_between_the_sides_is_found_with_its_side_and_length():
    x = _noise(4, 100.0, 8000.0)
    late_right = phase.channel_delay(phase.WindowSums(np.stack([x, _delayed(x, 12.0)], 1), SR))
    assert late_right is not None
    assert late_right[0] == pytest.approx(-12.0, abs=0.05) and late_right[1] > 0.9
    late_left = phase.channel_delay(phase.WindowSums(np.stack([_delayed(x, 6.0), x], 1), SR))
    assert late_left is not None and late_left[0] == pytest.approx(6.0, abs=0.05)
    # a wide source lines up at no lag, and a mono one lines up where it is
    left, right = _pad(11, 0.3)
    assert phase.channel_delay(phase.WindowSums(np.stack([left, right], 1), SR)) is None
    assert phase.channel_delay(phase.WindowSums(np.stack([x, x], 1), SR)) is None


# ---------------------------------------------------------------- finding it in a mix

def test_a_stem_with_one_side_upside_down_is_found_and_the_flip_named(tmp_path):
    _mono_bed(tmp_path)
    x = _noise(5, 80.0, 6000.0)
    _write(tmp_path, "guitar", x, -x)
    reading = _reading(tmp_path)
    flipped = [i for i in reading.issues if i.target == "guitar"]
    assert [i.kind for i in flipped] == ["inverted_channel"]
    assert flipped[0].cause == "file"
    assert "utility invert_" in flipped[0].fix and "guitar" in flipped[0].fix
    assert not reading.clean


def test_the_flip_named_keeps_the_stem_in_polarity_with_the_rest_of_the_mix(tmp_path):
    """Flipping the right makes the centre (L - R) / 2, flipping the left its negative: the one
    named must leave the stem's centre IN phase with the bed it doubles."""
    low = _noise(91, 60.0, 300.0)
    _write(tmp_path, "bed", low, low, level=0.2)
    # a doubled part whose LEFT is the bed's polarity and whose right was flipped
    _write(tmp_path, "double", low, -low, level=0.1)
    fix = next(i.fix for i in _reading(tmp_path).issues if i.target == "double")
    assert "invert_right" in fix
    _write(tmp_path, "double", -low, low, level=0.1)
    fix = next(i.fix for i in _reading(tmp_path).issues if i.target == "double")
    assert "invert_left" in fix


def test_a_pad_widened_past_mono_safe_is_found_on_the_master_and_traced_to_its_strip_width(tmp_path):
    _mono_bed(tmp_path)
    left, right = _pad(20, 0.45)
    _write(tmp_path, "pad", left, right, level=0.2)
    session = Session(tmp_path)
    assert _reading(tmp_path).clean          # as it arrives, the mix folds fine
    session.user_edit("pad", {"width": 1.8})
    reading = _reading(tmp_path)
    master = [i for i in reading.issues if i.target == "master"]
    assert master and all(i.kind == "cancels_in_mono" and i.cause == "mix" for i in master)
    assert all(i.from_stems[:1] == ["pad"] for i in master)
    assert "strip width 1.80 ->" in master[0].fix and "chain takes" in master[0].fix
    pad = next(t for t in reading.targets if t.target == "pad")
    assert pad.width == 1.8 and pad.mono_safe_width is not None and pad.mono_safe_width < 1.8
    assert all(b.file_corr is not None for b in pad.bands.values())
    # ...and the move it names is the move that fixes it
    session = Session(tmp_path)
    session.user_edit("pad", {"width": pad.mono_safe_width})
    assert _reading(tmp_path).clean


def test_a_wide_stem_on_its_own_is_not_an_issue(tmp_path):
    """Released synth stems sit below 0 for 40-57 % of their time and the records fold fine: a
    wide stem the master carries is a choice, not a fault."""
    _mono_bed(tmp_path)
    vocal = _noise(30, 200.0, 5000.0)
    _write(tmp_path, "vocal", vocal, vocal, level=0.25)
    left, right = _pad(31, 0.05)
    _write(tmp_path, "pad", left, right, level=0.06)
    reading = _reading(tmp_path)
    pad = next(t for t in reading.targets if t.target == "pad")
    assert max(b.below_zero_pct for b in pad.bands.values()) > 30.0
    assert reading.clean, reading.issues


def test_the_masters_own_widening_is_blamed_on_the_master_chain(tmp_path):
    _mono_bed(tmp_path)
    left, right = _pad(40, 0.5)
    _write(tmp_path, "pad", left, right, level=0.2)
    _apply(Session(tmp_path), [{"type": "imager", "stem_id": "master", "width_mid": 2.0,
                                "width_high": 2.0}])
    reading = _reading(tmp_path)
    master = [i for i in reading.issues if i.target == "master"]
    assert master and {i.cause for i in master} == {"master_chain"}
    assert "imager" in master[0].fix


def test_two_stems_in_opposite_polarity_are_a_conflict_with_a_flip(tmp_path):
    kick = _noise(50, 40.0, 120.0)
    _write(tmp_path, "kick", kick, kick, level=0.3)
    _write(tmp_path, "sub", -kick, -kick, level=0.2)
    reading = _reading(tmp_path)
    assert len(reading.conflicts) >= 1
    c = reading.conflicts[0]
    assert sorted(c.stems) == ["kick", "sub"] and c.corr < -0.9 and c.sum_db < -6.0
    assert c.lag_ms is None and "flip sub's polarity" in c.fix   # the quieter one
    # ...and flipping it, as named, ends it
    _apply(Session(tmp_path), [{"type": "utility", "stem_id": "sub", "invert": True}])
    assert not _reading(tmp_path).conflicts


def test_a_stem_a_few_milliseconds_early_is_a_timing_conflict_not_a_flip(tmp_path):
    bass = _noise(60, 60.0, 150.0)
    # 5 ms: half a period at 100 Hz, so the bass band cancels where both play
    _write(tmp_path, "bass_di", bass, bass, level=0.25)
    _write(tmp_path, "bass_amp", _delayed(bass, 5.0), _delayed(bass, 5.0), level=0.25)
    reading = _reading(tmp_path)
    timing = [c for c in reading.conflicts if c.lag_ms is not None]
    assert timing, reading.conflicts
    c = timing[0]
    assert c.stems == ["bass_amp", "bass_di"]
    # stems[1] (bass_di) is the EARLY one: positive lag
    assert c.lag_ms == pytest.approx(5.0, abs=0.1) and c.corr_at_lag > 0.9
    assert "bass_di arrives 5.0 ms early" in c.fix and "track.set_delay" in c.fix


def test_unrelated_stems_in_the_same_band_are_not_a_conflict(tmp_path):
    _write(tmp_path, "kick", _noise(70, 40.0, 150.0), _noise(70, 40.0, 150.0), level=0.3)
    _write(tmp_path, "bass", _noise(71, 40.0, 150.0), _noise(71, 40.0, 150.0), level=0.3)
    assert not _reading(tmp_path).conflicts


def test_naming_stems_reads_only_those_and_the_master(tmp_path):
    _mono_bed(tmp_path)
    x = _noise(80, 300.0, 3000.0)
    _write(tmp_path, "lead", x, x)
    reading = _reading(tmp_path, ["lead"])
    assert [t.target for t in reading.targets] == ["lead"]
    assert reading.master is not None and "only the channels named" in reading.note
    with pytest.raises(KeyError):
        _reading(tmp_path, ["nobody"])


# ---------------------------------------------------------------- the render's flag + the tool

def test_the_render_flag_is_the_masters_rule(tmp_path):
    left, right = _pad(81, 0.0)
    master_like = np.stack([left, right], 1)
    assert phase.master_at_risk(master_like, SR)          # unrelated sides: below 0 half the time
    x = _noise(82, 80.0, 8000.0)
    assert phase.master_at_risk(np.stack([x, -x], 1), SR) == ["inverted"]
    assert phase.master_at_risk(np.stack([x, x], 1), SR) == []


@pytest.mark.anyio
async def test_analyze_phase_roundtrips_on_the_demo_project(demo_project):
    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        reading = await call(client, "analyze_phase", {})
        assert reading["master"]["target"] == "master"
        assert {t["target"] for t in reading["targets"]} == {"stem_a", "stem_b"}
        assert reading["window_s"] == pytest.approx(phase.window_seconds(
            Session(demo_project).project_sample_rate()), abs=0.001)
        assert isinstance(reading["clean"], bool)
        named = await call(client, "analyze_phase", {"stem_ids": ["stem_a"]})
        assert [t["target"] for t in named["targets"]] == ["stem_a"]


# ---------------------------------------------------------------- a stereo move comes back measured

def _engine_call(client):
    async def call_engine(name: str, args: dict):
        result = await client.call_tool(name, args or {})
        assert not result.isError, result.content
        payload = result.structuredContent
        return payload["result"] if isinstance(payload, dict) and set(payload) == {"result"} else payload
    return call_engine


@pytest.mark.anyio
async def test_a_width_move_comes_back_with_what_it_did_to_the_phase(tmp_path):
    """The agent widened a pad to 1.40 to clear masking and nothing it read said what that did
    (2026-10-13). A stereo move's result now carries the phase check of what it touched; a move
    that is not a stereo move carries none."""
    from orchestrator.turn_host import propose_and_apply, stereo_targets
    _mono_bed(tmp_path)
    left, right = _pad(20, 0.45)
    _write(tmp_path, "pad", left, right, level=0.2)
    req = {"rationale": "out of the centre", "predicted_impact": [],
           "actions": [{"type": "width", "stem_id": "pad", "width": 1.8}]}
    assert stereo_targets(req) == ["pad"]
    async with create_connected_server_and_client_session(create_server(tmp_path)) as client:
        engine_call = _engine_call(client)
        pid, _, applied = await propose_and_apply(engine_call, None, req)
        assert pid and applied["ok"]
        check = applied["phase"]
        assert check["clean"] is False
        assert "pad" in check["corr_by_band"]
        master = [i for i in check["issues"] if i["target"] == "master"]
        assert master and master[0]["from_stems"][0] == "pad"
        assert "strip width 1.80 ->" in master[0]["fix"]
        # the move it names, made, reads clean
        safe = float(master[0]["fix"].split("strip width 1.80 -> ")[1].split()[0])
        _, _, fixed = await propose_and_apply(engine_call, None, {
            "rationale": "back to mono-safe", "predicted_impact": [],
            "actions": [{"type": "width", "stem_id": "pad", "width": safe}]})
        assert fixed["phase"]["clean"] is True
        # a level move is not measured for phase
        _, _, level = await propose_and_apply(engine_call, None, {
            "rationale": "down", "predicted_impact": [],
            "actions": [{"type": "gain", "stem_id": "pad", "delta_db": -1.0}]})
        assert "phase" not in level


def test_the_stereo_moves_are_the_ones_that_move_the_stereo_picture():
    from orchestrator.turn_host import stereo_targets
    moves = [{"type": "imager", "stem_id": "master", "width_high": 1.2},
             {"type": "eq", "stem_id": "vox", "channel": "side", "band": {}},
             {"type": "eq", "stem_id": "gtr", "band": {}},
             {"type": "band_width", "stem_id": "keys", "band": "mid", "width": 1.3},
             {"type": "comp", "stem_id": "drums"},
             {"type": "utility", "stem_id": "bass", "invert": True}]
    assert stereo_targets({"actions": moves}) == ["master", "vox", "keys", "bass"]


# ---------------------------------------------------------------- the flip the engineer reaches for

def test_invert_on_a_utility_the_dock_made_flips_both_channels(tmp_path):
    """A utility the dock adds carries `invert_left` / `invert_right` as false, and those win over
    `invert` in both engines - so setting `invert` flipped nothing that renders (James's phase test,
    2026-10-13: the polarity flip the engineer reached for was silent). `invert` is both flags now."""
    from mix_engine import dsp
    left, right = _pad(30, 0.5)
    _write(tmp_path, "keys", left, right)
    s = Session(tmp_path)
    before, _ = dsp.render_stem(s, "keys")
    s.add_stem_device("keys", "utility")
    for flag in ("invert_left", "invert_right"):   # what the dock's add persists
        s.set_master_device_param("utility", 0, flag, 0.0, target="keys")
    unflipped, _ = dsp.render_stem(s, "keys")
    assert np.allclose(unflipped, before, atol=1e-6)   # control: a utility at rest changes nothing
    s.set_master_device_param("utility", 0, "invert", 1.0, target="keys")
    u = [e for e in s._state["stems"]["keys"]["chain"] if e["type"] == "utility"][0]
    assert u["invert_left"] == u["invert_right"] == 1.0
    after, _ = dsp.render_stem(s, "keys")
    assert np.max(np.abs(after + before)) < 1e-6 * np.max(np.abs(before)) + 1e-9
    s.set_master_device_param("utility", 0, "invert", 0.0, target="keys")
    back, _ = dsp.render_stem(s, "keys")
    assert np.allclose(back, before, atol=1e-6)


def test_a_mix_move_on_an_instrument_track_is_refused_with_the_door_that_reaches_it():
    """"unknown stem 'Phase B'; stems: []" was all the engineer read when it tried to flip an
    instrument track's polarity with a proposal, and it concluded the fix could not be made. The
    refusal now names `device.add` on that track."""
    from orchestrator.turn_host import with_stemless_hint
    edit = {"tracks": [{"index": 0, "name": "Phase B", "track_kind": "midi", "clips": []}]}
    req = {"actions": [{"type": "gain", "stem_id": "bass", "delta_db": -1.0},
                       {"type": "utility", "stem_id": "Phase B", "invert": True}]}
    refused = {"accepted": False, "violations": [
        {"action_index": 1, "rule": "unknown_stem", "detail": "unknown stem 'Phase B'; stems: []"}]}
    hinted = with_stemless_hint(refused, req, edit)
    assert "device.add" in hinted["hint"] and "'Phase B'" in hinted["hint"]
    assert '"utility"' in hinted["hint"] and '"invert": true' in hinted["hint"]
    assert hinted["violations"] == refused["violations"]
    # controls: another rule, a name that is no track, no edit - the refusal as it was
    other = {"accepted": False, "violations": [{"action_index": 1, "rule": "gain_per_stem"}]}
    assert with_stemless_hint(other, req, edit) == other
    stray = {**req, "actions": [{"type": "utility", "stem_id": "Nobody", "invert": True}]}
    nobody = {"accepted": False, "violations": [{"action_index": 0, "rule": "unknown_stem"}]}
    assert with_stemless_hint(nobody, stray, edit) == nobody
    assert with_stemless_hint(refused, req, None) == refused
