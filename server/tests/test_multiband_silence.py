"""The multiband devices must write nothing into a silent stretch before their content - audit
`docs/audits/03-multiband.md` F1, and its end.

THE BUG (2026-09-08). Their offline band split was a hard RECTANGULAR bin mask on the whole-file
rFFT: each band ALONE carried tens of dB into a digitally-silent lead-in (a zero-phase brickwall
spreads content acausally; an N-point transform of a whole take is circular), cancelling only
while the bands took the SAME gain - which is the whole purpose of a multiband device. A tapered,
zero-padded FFT split (`_smooth_fft_bands`) took that to below -120 dBFS.

THE END (2026-10-17, the multiband compressor, OTT and multiband saturation audits). The native
split those devices by the COMPLEMENT `mid = dry - low - high`, which is not a band (driving the
low band moved an 80 Hz tone into the "mid"), and offline by the FFT split. Both engines now run
the REAL 3-way LR4 split (`dsp._lr4_three_way`) - real bands, causal - so every one of them writes
exactly nothing before its first sample of content (`test_the_device_writes_nothing_before_its_content`).
The FFT splits, their legs and their controls went with them: nothing calls one any more.
"""
import math

import numpy as np
import pytest
import soundfile as sf

from mix_engine import dsp
from mix_engine.models import ProposalRequest
from mix_engine.session import Session

SR = 48000
LEAD_S = 1.0                 # bit-exact digital silence at the head of the stem


def _db(v: float) -> float:
    return 20.0 * math.log10(v) if v > 0.0 else -600.0


def _peak_db(a) -> float:
    a = np.asarray(a, dtype=np.float64)
    return _db(float(np.max(np.abs(a)))) if a.size else -600.0


def _rms_db(a) -> float:
    return _db(float(np.sqrt(np.mean(np.asarray(a, dtype=np.float64) ** 2))))


def _ms(v: float) -> int:
    return int(round(v / 1000.0 * SR))


def _silent_lead_stem(seconds: float = 6.0, fade_ms: float = 0.0, seed: int = 7):
    """(frames, channels) float32: `LEAD_S` of BIT-EXACT zero, then music that runs
    to the last sample — a non-zero final sample is what the circular wrap folds
    onto sample 0. `fade_ms` softens the onset; 0 is a hard splice, the worst case."""
    rng = np.random.default_rng(seed)
    n, k = int(seconds * SR), int(LEAD_S * SR)
    t = np.arange(n - k) / SR
    music = (0.35 * np.sin(2 * np.pi * 70.0 * t)         # low band
             + 0.25 * np.sin(2 * np.pi * 700.0 * t)      # mid band
             + 0.18 * np.sin(2 * np.pi * 5200.0 * t)     # high band
             + 0.05 * rng.standard_normal(n - k))        # broadband, non-zero AT the onset
    music = music * (0.6 + 0.4 * np.sin(2 * np.pi * 0.7 * t))
    x = np.zeros((n, 2), dtype=np.float32)
    x[k:, 0] = music.astype(np.float32)
    x[k:, 1] = (music * 0.9).astype(np.float32)
    if fade_ms:
        m = _ms(fade_ms)
        x[k:k + m] *= np.linspace(0.0, 1.0, m, endpoint=False)[:, None].astype(np.float32)
    assert np.all(x[:k] == 0.0), "the lead-in must be bit-exact zero"
    assert abs(float(x[-1, 0])) > 1e-3, "the take must end on a non-zero sample"
    return x, k


def _channels_first(x_fr):
    return np.ascontiguousarray(np.asarray(x_fr).T)


def _ott(**over) -> dict:
    e = {"type": "ott", "depth": 0.35, "time": 1.0,
         "low": {"up": 1.0, "down": 1.0, "gain_db": 0.0},
         "mid": {"up": 1.0, "down": 1.0, "gain_db": 0.0},
         "high": {"up": 1.0, "down": 1.0, "gain_db": 0.0}}
    e.update(over)
    return e


def _mb(low: dict, **over) -> dict:
    gentle = {"threshold_db": 0.0, "ratio": 1.0}
    e = {"type": "mb_comp", "low": low, "mid": dict(gentle), "high": dict(gentle),
         "attack_ms": 5.0, "release_ms": 100.0}
    e.update(over)
    return e


def _mbsat_entry(drive_db: float = 18.0) -> dict:
    return {"low": {"drive_db": drive_db}, "mid": {"drive_db": 0.0},
            "high": {"drive_db": 0.0},
            "crossover_lo_hz": 150.0, "crossover_hi_hz": 2000.0}


BOOM = {"threshold_db": -45.0, "ratio": 8.0}   # the Phase 48 "multiband boom control" shape

# The devices that split this way are held to the causal split's claim below: the multiband
# compressor left the FFT split for the DAW's real 3-way LR4 on 2026-10-17 and OTT the same
# day. Each case is (label, render), render taking (channels, frames) and an entry override.
CAUSAL_CASES = [
    ("mb_comp low -45/r8 (Phase 48 boom control)",
     lambda a, **o: dsp._multiband_comp(a, SR, _mb(dict(BOOM), **o))),
    ("mb_comp low -45/r8 +6 dB makeup",
     lambda a, **o: dsp._multiband_comp(a, SR, _mb(dict(BOOM, makeup_db=6.0), **o))),
    ("mb_comp low -24/r2 +6 dB makeup",
     lambda a, **o: dsp._multiband_comp(
         a, SR, _mb({"threshold_db": -24.0, "ratio": 2.0, "makeup_db": 6.0}, **o))),
    ("mb_comp high up -6/r2",
     lambda a, **o: dsp._multiband_comp(a, SR, _mb({"threshold_db": 0.0, "ratio": 1.0},
                                                    high={"threshold_db": -6.0, "ratio": 2.0,
                                                          "up": True}, **o))),
    ("ott depth 0.35 (shipped add default)", lambda a, **o: dsp._ott(a, SR, _ott(**o))),
    ("ott depth 0.80 (agent cap)", lambda a, **o: dsp._ott(a, SR, _ott(depth=0.8, **o))),
    ("mb_sat low +18 dB", lambda a, **o: np.ascontiguousarray(
        dsp._multiband_saturate(np.ascontiguousarray(a.T).astype(np.float64), {**_mbsat_entry(18.0), **o}, SR).T)),
]


def _unity(label: str):
    """The case's device doing nothing - the negative control's reference."""
    if label.startswith("ott"):
        return lambda a, **o: dsp._ott(a, SR, _ott(depth=0.0, **o))
    if label.startswith("mb_sat"):
        return lambda a, **o: np.ascontiguousarray(
            dsp._multiband_saturate(np.ascontiguousarray(a.T).astype(np.float64), {**_mbsat_entry(0.0), **o}, SR).T)
    return lambda a, **o: dsp._multiband_comp(a, SR, _mb({"threshold_db": 0.0, "ratio": 1.0}, **o))




# ------------------------------------------------- 1. the silence stays silent --

@pytest.mark.parametrize("label,render", CAUSAL_CASES, ids=[c[0] for c in CAUSAL_CASES])
@pytest.mark.parametrize("lo_hz,hi_hz", [(150.0, 2000.0), (30.0, 200.0), (500.0, 750.0),
                                         (4000.0, 16000.0)])
def test_the_device_writes_nothing_before_its_content(label, render, lo_hz, hi_hz):
    """The causal split's claim, at the default and at moved crossovers (Phase 44): every
    sample of the bit-exact silent lead-in is still bit-exact zero - not -120 dBFS, zero."""
    x_fr, k = _silent_lead_stem()
    cx = {"crossover_lo_hz": lo_hz, "crossover_hi_hz": hi_hz}
    out = np.asarray(render(_channels_first(x_fr), **cx))
    assert np.all(out[:, :k] == 0.0), (
        f"{label} at {lo_hz:.0f}/{hi_hz:.0f} Hz: {_peak_db(out[:, :k]):.1f} dBFS before the "
        f"first sample of content")
    # NEGATIVE CONTROL - the device does something to the music, or the silence is vacuous.
    # At the default crossovers only: at 30 Hz the low band of this music is near empty.
    if (lo_hz, hi_hz) == (150.0, 2000.0):
        unity = np.asarray(_unity(label)(_channels_first(x_fr), **cx))
        assert _rms_db(out[:, k:] - unity[:, k:]) > -35.0, f"{label}: the device did nothing"


def test_the_silence_stays_silent_end_to_end_through_render_stem(tmp_path):
    """The audit's own path: a real project, a real proposal, `dsp.render_stem`."""
    x_fr, k = _silent_lead_stem()
    project = tmp_path / "silent_lead"
    project.mkdir()
    sf.write(project / "lead.wav", x_fr, SR)

    session = Session(project)
    result = session.propose(ProposalRequest(
        rationale="audit 03-multiband F1", predicted_impact=[],
        actions=[{"type": "ott", "stem_id": "lead", "depth": 0.25}]))   # the run's cap (2026-10-05)
    assert result.accepted, result.violations
    assert session.apply(result.proposal_id).ok
    out, sr = dsp.render_stem(session, "lead")
    assert sr == SR
    rendered = _channels_first(out)                       # render_stem returns (frames, ch)
    assert np.all(rendered[:, :k] == 0.0), (
        f"render_stem put {_peak_db(rendered[:, :k]):.1f} dBFS into a bit-exact silent lead-in")
    # NEGATIVE CONTROL - the render path runs the device: the MUSIC is changed by the OTT.
    music = slice(k, rendered.shape[1])
    dry, wet = _rms_db(_channels_first(x_fr)[:, music]), _rms_db(rendered[:, music])
    assert abs(wet - dry) > 1.0, (
        f"the OTT moved the music by only {wet - dry:+.2f} dB RMS - the render path is "
        f"not actually running the device, so the silence leg is vacuous")


def test_multiband_saturation_still_saturates():
    """Non-vacuity for the other device: driving the low band must change the audio."""
    rng = np.random.default_rng(12)
    n = SR
    x = np.stack([rng.standard_normal(n) * 0.2, rng.standard_normal(n) * 0.2], axis=1)
    out = dsp._multiband_saturate(x.copy(), _mbsat_entry(18.0), SR)
    assert _peak_db(np.asarray(out) - x) > -30.0, "an 18 dB low-band drive must audibly change the signal"
