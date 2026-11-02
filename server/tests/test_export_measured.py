"""An export's reply carries what the FILE measures (2026-11-02, before/after round 5).

The engineer mastered to a -1 dBTP brief with the limiter's ceiling at exactly -1.0, exported,
and reported "-1.0 dB true peak": the file read -0.97 on an 8x true-peak meter. The export's
reply gave only its length and sample peak, so the figure the engineer said was its own. The
brain now reads the exported file before the model sees the reply: its integrated loudness and
4x true peak (`measured`), and `over_ceiling` when the true peak is over -1 dBTP."""
from __future__ import annotations

import anyio
import numpy as np
import soundfile as sf

from orchestrator import turn_host

SR = 48000


def _sine_between_samples(path, true_peak_db: float) -> None:
    """A sine at a quarter of the sample rate, 45 degrees off the grid: every sample sits 3 dB
    under the waveform's peak, so the sample peak and the true peak are 3 dB apart."""
    t = np.arange(4 * SR) / SR
    x = 10 ** (true_peak_db / 20.0) * np.sin(2 * np.pi * (SR / 4) * t + np.pi / 4)
    sf.write(path, np.stack([x, x], axis=1).astype(np.float32), SR, subtype="FLOAT")


def _measured(path, project):
    reply = {"ok": True, "file": str(path), "format": "wav"}
    return anyio.run(turn_host.measured_command, "session.export", reply, project)


def test_an_export_over_the_ceiling_says_so_with_the_files_true_peak(tmp_path):
    path = tmp_path / "bounces" / "song-after.wav"
    path.parent.mkdir()
    _sine_between_samples(path, -0.9)
    res = _measured(path, tmp_path)
    assert abs(res["measured"]["true_peak_dbtp"] - (-0.9)) < 0.15, res["measured"]
    assert isinstance(res["measured"]["lufs_i"], float)
    assert "over_ceiling" in res and "-1 dBTP" in res["over_ceiling"]


def test_an_export_under_the_ceiling_carries_its_figures_and_no_warning(tmp_path):
    path = tmp_path / "song.wav"
    _sine_between_samples(path, -1.9)
    res = _measured(path, tmp_path)
    assert abs(res["measured"]["true_peak_dbtp"] - (-1.9)) < 0.15, res["measured"]
    assert "over_ceiling" not in res


def test_a_refused_export_or_a_missing_file_is_left_as_it_came(tmp_path):
    refused = {"ok": False, "error": "declined at the approval box"}
    assert anyio.run(turn_host.measured_command, "session.export", dict(refused), tmp_path) == refused
    gone = {"ok": True, "file": str(tmp_path / "nowhere.wav")}
    res = anyio.run(turn_host.measured_command, "session.export", dict(gone), tmp_path)
    assert "measured" not in res and "measure_error" in res
