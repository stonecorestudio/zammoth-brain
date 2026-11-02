"""The delay's law (2026-10-27): fitted to a reference delay - `delay_law`.

Held here:
  * the reference's MEASURED behaviour, which the delay before this did not have (the red half): what
    one pass through the loop does by band (a 200 Hz high-pass and a 24 dB/oct low-pass at the tone's
    corner), the first echo's level, feedback as the linear gain of a pass, the right side 1 % later,
    and a slow wow on the time;
  * the two offline engines run the same law, a mono input gives a mono output, the knobs are held to
    the table, DelayLaw.h is the file this module writes, the old loop filter is gone.

ONE fixture, `data/delay_law_parity.json`, carries Python's answers on a signal both engines make from
one formula; the console leg `--delay-law` asserts MonitorChain renders them. A law change on purpose:
`python server/tests/test_delay_law.py --write`, then the leg.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))

from mix_engine import dsp, reference_monitor  # noqa: E402

SR = 48000
FIXTURE = HERE / "data" / "delay_law_parity.json"


def _impulse(seconds: float = 2.0) -> np.ndarray:
    x = np.zeros((int(seconds * SR), 2))
    x[0] = 1.0
    return x


def _spec(y: np.ndarray, k: int, d: int, ch: int = 0) -> tuple[np.ndarray, np.ndarray]:
    w = int(0.14 * SR)
    seg = y[k * d - w:k * d + w, ch]
    return np.fft.rfftfreq(65536, 1 / SR), 20 * np.log10(np.abs(np.fft.rfft(seg, 65536)) + 1e-30)


# ------------------------------------------------ the reference's behaviour (the red half)
# One pass through its loop at its default (tone in the middle), dB by frequency, read off echo 2
# against echo 1 at 50 % feedback; and its first echo at 1 kHz.
MEASURED_PASS = {100: -11.9, 200: -3.3, 1000: -0.1, 5000: -6.1, 8000: -19.5}
MEASURED_FIRST_ECHO_1K = -3.1


def test_one_pass_is_the_references():
    y = dsp._return_delay(_impulse(), SR, {"delay_ms": 300.0, "feedback": 0.5, "tone": 0.5})
    f, e1 = _spec(y, 1, int(0.3 * SR))
    _, e2 = _spec(y, 2, int(0.3 * SR))
    h = e2 - e1 - 20 * math.log10(0.5)
    for fc, want in MEASURED_PASS.items():
        got = h[np.argmin(abs(f - fc))]
        assert abs(got - want) < 1.0, (fc, got, want)


def test_the_first_echo_is_the_references():
    y = dsp._return_delay(_impulse(), SR, {"delay_ms": 300.0, "feedback": 0.0, "tone": 0.5})
    f, e1 = _spec(y, 1, int(0.3 * SR))
    assert abs(e1[np.argmin(abs(f - 1000))] - MEASURED_FIRST_ECHO_1K) < 0.3


def test_feedback_is_the_gain_of_a_pass():
    for fb in (0.2, 0.5, 0.8):
        y = dsp._return_delay(_impulse(), SR, {"delay_ms": 200.0, "feedback": fb, "tone": 1.0})
        f, e2 = _spec(y, 2, int(0.2 * SR))
        _, e3 = _spec(y, 3, int(0.2 * SR))
        k = np.argmin(abs(f - 1000))
        assert abs((e3[k] - e2[k]) - 20 * math.log10(fb)) < 0.5, fb


def test_the_right_side_is_one_percent_later():
    y = dsp._return_delay(_impulse(), SR, {"delay_ms": 350.0, "feedback": 0.0, "tone": 1.0})
    tl = np.argmax(np.abs(y[:, 0])) / SR * 1000.0
    tr = np.argmax(np.abs(y[:, 1])) / SR * 1000.0
    assert abs((tr - tl) - 3.5) < 0.5, (tl, tr)


def test_the_time_wows():
    """A click every 50 ms, its echo's place: it swings about a millisecond and a half either way, slowly."""
    n, per = SR * 8, SR // 20
    x = np.zeros((n, 2)); x[::per] = 1.0
    y = dsp._return_delay(x, SR, {"delay_ms": 350.0, "feedback": 0.0, "tone": 1.0})
    d = int(0.35 * SR)
    off = []
    for k in range(1, n // per - 8):
        a = k * per + d - 200
        off.append(np.argmax(np.abs(y[a:a + 400, 0])) - 200)
    swing = (max(off) - min(off)) / SR * 1000.0
    assert 2.0 < swing < 3.5, swing


# ------------------------------------------------ the law, one in both offline engines
def test_the_reference_monitor_is_the_same_law():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((SR, 2)) * 0.1
    for pp in (0.0, 1.0):
        e = {"delay_ms": 130.0, "feedback": 0.6, "tone": 0.3, "ping_pong": pp}
        assert np.array_equal(dsp._return_delay(x, SR, e), reference_monitor._return_delay(x, float(SR), e))


def test_ping_pong_keeps_both_sides_on_the_same_time():
    y = dsp._return_delay(_impulse(), SR, {"delay_ms": 300.0, "feedback": 0.6, "tone": 0.8, "ping_pong": 1.0})
    tl = np.argmax(np.abs(y[: int(0.45 * SR), 0])) / SR * 1000.0
    tr = np.argmax(np.abs(y[int(0.45 * SR): int(0.75 * SR), 1])) / SR * 1000.0 + 450.0
    assert abs(tl - 300.0) < 2.0 and abs(tr - 600.0) < 3.0, (tl, tr)


def test_the_knobs_are_held_to_the_table():
    x = _impulse(1.0)
    from mix_engine import delay_law as dl
    assert np.array_equal(dl.apply(x, SR, {"delay_ms": 9000.0, "feedback": 3.0, "tone": 7.0}),
                          dl.apply(x, SR, {"delay_ms": 2000.0, "feedback": 0.85, "tone": 1.0}))


def test_the_header_is_the_one_this_module_writes():
    from mix_engine import delay_law as dl
    assert (REPO / "native/Source/DelayLaw.h").read_text() == dl.header_text()


def test_the_old_loop_filter_is_gone():
    assert not hasattr(dsp, "DELAY_LOOP_HP_HZ") and not hasattr(dsp, "_DelayLoopFilter")
    hdr = (REPO / "native/Source/MonitorChain.h").read_text()
    for name in ("kDelayLoopHpHz", "delayToneAlpha", "delayCutAlpha", "srDelayTone"):
        assert name not in hdr, name


# ------------------------------------------------ the fixture the native leg reads
def signal(n: int, sr: float) -> np.ndarray:
    """(2, n): what --delay-law makes from the same formula (dlyl::signal)."""
    i = np.arange(n, dtype=np.float64)
    out = []
    for c in (0, 1):
        ph = 0.0 if c == 0 else 0.7
        hit = 0.6 * np.exp(-np.mod(i, 7000.0) / 250.0) * np.sin(2 * math.pi * 440.0 * i / sr + ph)
        tone = 0.12 * np.sin(2 * math.pi * 2500.0 * i / sr + 2.0 * ph) * (i < n * 0.4)
        out.append(hit + tone)
    return np.array(out, dtype=np.float32)


CASES = [(48000, {"delay_ms": 350.0, "feedback": 0.35, "tone": 0.5, "ping_pong": 0.0}),
         (48000, {"delay_ms": 120.0, "feedback": 0.7, "tone": 0.15, "ping_pong": 1.0}),
         (48000, {"delay_ms": 37.0, "feedback": 0.85, "tone": 1.0, "ping_pong": 0.0}),
         (44100, {"delay_ms": 260.0, "feedback": 0.5, "tone": 0.7, "ping_pong": 0.0})]
N = 72000


def _fixture() -> dict:
    from mix_engine import delay_law as dl
    idx = list(range(0, N, 97))
    cases = []
    for sr, e in CASES:
        y = dl.apply(signal(N, sr).T.astype(np.float64), sr, e)
        cases.append({"sr": sr, "entry": {"type": "return_delay", "mix": 1.0, **e},
                      "left": [float(y[k, 0]) for k in idx], "right": [float(y[k, 1]) for k in idx]})
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
