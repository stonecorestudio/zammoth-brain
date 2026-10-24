"""`describe_region` (2026-09-03): the agent's read of ONE PART of a stem."""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
from mcp.shared.memory import create_connected_server_and_client_session

from mix_engine import analysis
from mix_engine.mcp_server import create_server
from mix_engine.models import ANALYSIS_VERSION

from test_phase1_acceptance import call

SR = 44100


def _tone(freq_hz: float, seconds: float, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq_hz * t)).astype(np.float64)


def test_a_region_is_measured_on_its_own_slice_not_the_whole_file():
    """Quiet first half, loud second half: the two regions must read differently,
    and the envelope inside the loud half must be flat."""
    audio = np.concatenate([_tone(220, 4.0, 0.02), _tone(220, 4.0, 0.4)])
    quiet = analysis.describe_region(audio, SR, "bass", 0.0, 4.0)
    loud = analysis.describe_region(audio, SR, "bass", 4.0, 8.0)
    assert not quiet.silent and not loud.silent
    assert loud.lufs_s - quiet.lufs_s > 20
    assert loud.start_s == 4.0 and loud.end_s == 8.0
    assert 1 <= len(loud.lufs_envelope) <= analysis.REGION_ENVELOPE_POINTS
    values = [p.lufs_s for p in loud.lufs_envelope]
    assert max(values) - min(values) < 1.0
    assert loud.lufs_envelope[0].t_s == 4.0


def test_a_fade_shows_in_the_envelope_and_a_brightening_in_the_tilt_trend():
    n = int(6.0 * SR)
    t = np.arange(n) / SR
    ramp = np.linspace(0.05, 0.5, n)
    # a low tone fading up while a high tone fades in only in the second half
    audio = ramp * np.sin(2 * np.pi * 110 * t)
    audio[n // 2:] += 0.3 * np.sin(2 * np.pi * 6000 * t[n // 2:])
    r = analysis.describe_region(audio, SR, "keys", 0.0, 6.0)
    env = [p.lufs_s for p in r.lufs_envelope]
    assert env[-1] > env[0] + 6, env
    assert r.tilt_trend_db_per_oct is not None and r.tilt_trend_db_per_oct > 0.5


def test_masking_is_reported_both_ways_only_where_the_other_stem_is_loud():
    this = _tone(300, 6.0, 0.1)                       # low-mid, modest
    masker = np.concatenate([np.zeros(int(3.0 * SR)), _tone(300, 3.0, 0.6)])
    early = analysis.describe_region(this, SR, "gtr", 0.0, 3.0, others={"synth": masker})
    late = analysis.describe_region(this, SR, "gtr", 3.0, 6.0, others={"synth": masker})
    assert early.masked_by == [], "nothing is playing over it yet"
    assert late.masked_by and late.masked_by[0].stem_id == "synth"
    assert late.masked_by[0].excess_db >= analysis.REGION_MASK_FLOOR_DB
    # and the loud one reports that it masks the quiet one
    mirror = analysis.describe_region(masker, SR, "synth", 3.0, 6.0, others={"gtr": this})
    assert mirror.masks and mirror.masks[0].stem_id == "gtr"


def test_a_silent_region_says_so_and_carries_no_metrics():
    audio = np.concatenate([np.zeros(int(2.0 * SR)), _tone(440, 2.0)])
    r = analysis.describe_region(audio, SR, "vox", 0.0, 2.0)
    assert r.silent and r.lufs_s is None and r.lufs_envelope == []


@pytest.mark.anyio
async def test_describe_region_round_trips_over_mcp(demo_project):
    async with create_connected_server_and_client_session(
            create_server(demo_project)) as client:
        result = await call(client, "describe_region",
                            {"stem_id": "stem_a", "start_s": 1.0, "end_s": 3.0})
        with pytest.raises(Exception):
            await call(client, "describe_region",
                       {"stem_id": "stem_a", "start_s": 1.0, "end_s": 1.2})
    assert result["stem_id"] == "stem_a"
    assert result["analysis_version"] == ANALYSIS_VERSION
    assert result["start_s"] == 1.0 and result["end_s"] == 3.0
    assert "masked_by" in result and "band_energy" in result
