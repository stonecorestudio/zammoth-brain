"""The algorithmic reverb's law (2026-10-27): fitted to a reference reverb - `reverb_law`.

Held here:
  * the reference's MEASURED behaviour at its default space, which the eight-line network before this
    did not have (the red half): its decay by octave, its onset, the colour of its tail, no audible
    modes, and what its brightness does to the top;
  * the two offline engines run the same law, a mono input gives a mono output, the knobs are held to
    the table, ReverbLaw.h is the file this module writes, and the old network's numbers are gone.

ONE fixture, `data/reverb_law_parity.json`, carries Python's answers on a signal both engines make from
one formula; the console leg `--reverb-law` asserts MonitorChain renders them. A law change on purpose:
`python server/tests/test_reverb_law.py --write`, then the leg.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))

from mix_engine import dsp, reference_monitor, reverb_law as rl  # noqa: E402

SR = 48000
FIXTURE = HERE / "data" / "reverb_law_parity.json"


def _ir(decay: float = 2.5, tone: float = 0.5, pre: float = 0.0, secs: float = 8.0) -> np.ndarray:
    x = np.zeros((int(secs * SR), 2))
    x[0] = 1.0
    return dsp._return_reverb(x, SR, {"decay_s": decay, "tone": tone, "pre_delay_ms": pre})


def _band(y: np.ndarray, f: float) -> np.ndarray:
    from scipy.signal import butter, sosfilt
    return sosfilt(butter(4, [f / math.sqrt(2), min(f * math.sqrt(2), SR * 0.475)], "bandpass", fs=SR, output="sos"), y, axis=0)


def _t60(y: np.ndarray, f: float) -> float:
    e = np.sum(_band(y, f) ** 2, axis=1)
    edc = 10 * np.log10(np.cumsum(e[::-1])[::-1] / e.sum() + 1e-30)
    t = np.arange(len(edc)) / SR
    i0, i1 = int(np.argmax(edc <= -5)), int(np.argmax(edc <= -35))
    return -60.0 / np.polyfit(t[i0:i1], edc[i0:i1], 1)[0]


def _octave_db(y: np.ndarray) -> np.ndarray:
    v = np.array([10 * np.log10(np.sum(_band(y, f) ** 2) + 1e-30) for f in OCT])
    return v - v[OCT.index(1000)]


OCT = [125, 500, 1000, 2000, 4000, 8000, 16000]
# ------------------------------------------------ the reference's behaviour (the red half)
# Its default space (2.5 s), measured on impulses: T60 by octave, the octave energy of its tail
# relative to 1 kHz, its onset. And what its brightness does to the top third-octave of the tail.
MEASURED_T60 = {125: 2.73, 500: 2.35, 1000: 2.15, 2000: 2.04, 4000: 1.93, 8000: 1.80, 16000: 1.75}
MEASURED_OCT = [-4.8, -3.1, 0.0, 4.2, 5.9, 3.7, -5.1]
MEASURED_ONSET_MS = 6.7
MEASURED_TOP_DB = {0.0: -30.3, 0.25: -14.8, 0.75: 10.1, 1.0: 10.1}   # 12.9 kHz third, against tone 0.5


def test_the_decay_by_band_is_the_references():
    """Within 12 % to 8 kHz. The 16 kHz octave decays sooner (the swinging cubic read takes ~1.5 dB a
    pass there): a named limit, within 30 % - the reverb that was heard and picked had it."""
    y = _ir()
    for f, want in MEASURED_T60.items():
        got = _t60(y, f)
        assert abs(got - want) < (0.30 if f >= 16000 else 0.12) * want, (f, got, want)


@pytest.mark.parametrize("decay", [0.6, 1.0, 1.8, 4.0, 8.0])
def test_the_tail_rings_as_long_as_its_law_says_at_every_decay(decay):
    """The rendered T60 follows `reverb_law.decays` across the knob, not at the default alone (the
    pre-merge review of 2026-10-28: the deleted FDN test held 0.6 / 1.8 / 4.0, its replacement 2.5
    only). Measured 1 kHz within 3 % of the mid decay, 125 Hz of the low one, up to 8 s."""
    y = _ir(decay=decay, secs=max(4.0, decay * 2.5))
    low, mid, _ = rl.decays(decay, 0.5)
    assert abs(_t60(y, 1000) - mid) < 0.05 * mid, (decay, _t60(y, 1000), mid)
    assert abs(_t60(y, 125) - low) < 0.06 * low, (decay, _t60(y, 125), low)


def test_the_tails_colour_is_the_references():
    got = _octave_db(_ir())
    assert np.max(np.abs(got - np.array(MEASURED_OCT))) < 1.5, np.round(got, 2)


def test_it_starts_where_the_reference_starts():
    y = _ir()
    e = np.sum(y ** 2, axis=1)
    onset_ms = 1000.0 * np.argmax(e >= e.max() * 1e-3) / SR
    assert abs(onset_ms - MEASURED_ONSET_MS) < 2.0, onset_ms
    # ...and builds without a gap: every 5 ms of the first 60 is within 20 dB of the loudest
    w = int(0.005 * SR)
    win = np.array([np.sum(e[i:i + w]) for i in range(int(0.010 * SR), int(0.060 * SR), w)])
    assert 10 * np.log10(win.min() / win.max()) > -20.0


def test_the_tail_has_no_audible_modes():
    """The narrow spectrum of the tail stands no further over its median than a noise's does: the
    eight-line network rang at its modes (23 dB at 4 kHz, the reference 12)."""
    y = _ir(secs=4.0)
    worst = []
    for fc in (250, 500, 1000, 2000, 4000):
        vals = []
        for a in (0.5, 1.0, 1.5, 2.0):
            for c in (0, 1):
                z = y[int(a * SR):int((a + 0.5) * SR), c]
                S = np.abs(np.fft.rfft(z * np.hanning(len(z)))) ** 2
                f = np.fft.rfftfreq(len(z), 1 / SR)
                m = (f > fc / 1.41) & (f < fc * 1.41)
                vals.append(10 * np.log10(S[m].max() / np.median(S[m])))
        worst.append(np.mean(vals))
    assert max(worst) < 15.0, np.round(worst, 1)


def test_tone_is_the_references_brightness():
    def third(y, c=12900.0):
        Y = np.abs(np.fft.rfft(y[:, 0])) ** 2 + np.abs(np.fft.rfft(y[:, 1])) ** 2
        f = np.fft.rfftfreq(len(y), 1 / SR)
        return 10 * np.log10(np.mean(Y[(f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))]))
    flat = third(_ir(tone=0.5))
    for tone, want in MEASURED_TOP_DB.items():
        got = third(_ir(tone=tone)) - flat
        assert abs(got - want) < 3.0, (tone, got, want)


def test_left_and_right_are_decorrelated():
    y = _ir(secs=3.0)
    corr = float(np.corrcoef(y[SR // 10:, 0], y[SR // 10:, 1])[0, 1])
    assert abs(corr) < 0.2, corr


# ------------------------------------------------ the law, one in both offline engines
def test_the_reference_monitor_is_the_same_law():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((SR, 2)) * 0.1
    e = {"decay_s": 2.0, "tone": 0.4, "pre_delay_ms": 30.0}
    assert np.array_equal(dsp._return_reverb(x, SR, e), reference_monitor._return_reverb(x, float(SR), e))


def test_a_mono_input_gives_a_mono_output():
    x = np.zeros((SR, 1)); x[0, 0] = 1.0
    y = dsp._return_reverb(x, SR, {"decay_s": 1.0, "tone": 0.5})
    assert y.shape == x.shape and np.abs(y).max() > 0.0


def test_the_knobs_are_held_to_the_table():
    x = np.zeros((SR, 2)); x[0] = 1.0
    assert np.array_equal(rl.apply(x, SR, {"decay_s": 40.0, "tone": 3.0, "pre_delay_ms": 900.0}),
                          rl.apply(x, SR, {"decay_s": 12.0, "tone": 1.0, "pre_delay_ms": 250.0}))


def test_the_header_is_the_one_this_module_writes():
    assert (REPO / "native/Source/ReverbLaw.h").read_text() == rl.header_text()


def test_the_old_network_is_gone():
    for name in ("REVERB_LINE_MS", "REVERB_AP_MS", "REVERB_WET_SCALE", "reverb_damping_alpha", "_hadamard8"):
        assert not hasattr(dsp, name), name
    hdr = (REPO / "native/Source/MonitorChain.h").read_text()
    for name in ("kRevLineMs", "kRevApMs", "kRevWetScaleTimesRoot8", "revLineFb", "reverbToneAlpha"):
        assert name not in hdr, name


# ------------------------------------------------ the fixture the native leg reads
def signal(n: int, sr: float) -> np.ndarray:
    """(2, n): what --reverb-law makes from the same formula (rvl::signal)."""
    i = np.arange(n, dtype=np.float64)
    out = []
    for c in (0, 1):
        ph = 0.0 if c == 0 else 0.7
        hit = 0.6 * np.exp(-np.mod(i, 9000.0) / 300.0) * np.sin(2 * math.pi * 520.0 * i / sr + ph)
        tone = 0.15 * np.sin(2 * math.pi * 3100.0 * i / sr + 2.0 * ph) * (i < n * 0.4)
        out.append(hit + tone)
    return np.array(out, dtype=np.float32)


CASES = [(48000, {"decay_s": 2.5, "tone": 0.5, "pre_delay_ms": 0.0}),
         (48000, {"decay_s": 0.6, "tone": 0.0, "pre_delay_ms": 30.0}),
         (48000, {"decay_s": 8.0, "tone": 1.0, "pre_delay_ms": 120.0}),
         (44100, {"decay_s": 1.8, "tone": 0.3, "pre_delay_ms": 20.0})]
N = 72000


def _fixture() -> dict:
    idx = list(range(0, N, 97))
    cases = []
    for sr, e in CASES:
        x = signal(N, sr)
        y = rl.apply(x.T.astype(np.float64), sr, e)
        entry = {"type": "return_reverb", "mix": 1.0, **e}
        cases.append({"sr": sr, "entry": entry, "left": [float(y[k, 0]) for k in idx], "right": [float(y[k, 1]) for k in idx]})
    return {"n": N, "indices": idx, "cases": cases}


def test_the_fixture_is_this_law():
    want = _fixture()
    got = json.loads(FIXTURE.read_text())
    assert got["n"] == want["n"] and got["indices"] == want["indices"]
    for a, b in zip(got["cases"], want["cases"]):
        assert a["entry"] == b["entry"] and a["sr"] == b["sr"]
        assert np.allclose(a["left"], b["left"], atol=1e-9) and np.allclose(a["right"], b["right"], atol=1e-9)


if __name__ == "__main__":
    if "--write" in sys.argv:
        FIXTURE.write_text(json.dumps(_fixture()))
        print(f"wrote {FIXTURE}")
