"""The delay's tone is IN its feedback loop, and it can ping-pong (2026-10-25, James's A/B).

It was a comb with one tone filter after it: every echo came back as bright as the first and the
low end piled up under the repeats. Held here: (1) each echo is darker than the one before; (2) a
low tone dies faster in the loop than a mid one (the loop's low cut); (3) `ping_pong` alternates
the echoes left and right; (4) the reference monitor is the same function; (5) the switch is a
parameter of the device at the model and the session's table. Since 2026-10-27 the loop is
delay_law's (test_delay_law.py holds its numbers, and that the right side runs 1 % later).
"""
from __future__ import annotations


import numpy as np

from mix_engine import dsp, models, reference_monitor, session

SR = 48000


def _burst(seconds: float = 3.0) -> np.ndarray:
    rng = np.random.default_rng(3)
    x = np.zeros((int(seconds * SR), 2))
    b = rng.standard_normal(int(0.02 * SR))
    x[:len(b), 0] = b
    x[:len(b), 1] = b
    return x


def _echo(y: np.ndarray, k: int, d: int, ch: int) -> np.ndarray:
    return y[k * d:k * d + int(0.05 * SR), ch]


def _centroid(v: np.ndarray) -> float:
    s = np.abs(np.fft.rfft(v))
    f = np.fft.rfftfreq(len(v), 1.0 / SR)
    return float((s * f).sum() / s.sum())


def test_each_echo_is_darker_than_the_last():
    e = {"delay_ms": 300.0, "feedback": 0.7, "tone": 0.5}
    d = int(0.3 * SR)
    y = dsp._return_delay(_burst(), SR, e)
    c = [_centroid(_echo(y, k, d, 0)) for k in range(1, 6)]
    assert all(b < a * 0.97 for a, b in zip(c, c[1:])), c


def test_the_low_end_dies_faster_in_the_loop():
    t = np.arange(int(0.05 * SR)) / SR

    def tail_drop(freq: float) -> float:
        x = np.zeros((int(2.0 * SR), 2))
        x[:len(t), :] = np.sin(2 * np.pi * freq * t)[:, None]
        y = dsp._return_delay(x, SR, {"delay_ms": 200.0, "feedback": 0.8, "tone": 1.0})
        d = int(0.2 * SR)
        rms = lambda v: np.sqrt((v ** 2).mean())
        return 20 * np.log10(rms(_echo(y, 6, d, 0)) / rms(_echo(y, 1, d, 0)))

    assert tail_drop(40.0) < tail_drop(1000.0) - 6.0, (tail_drop(40.0), tail_drop(1000.0))


def test_ping_pong_alternates_the_sides():
    e = {"delay_ms": 300.0, "feedback": 0.6, "tone": 0.8, "ping_pong": 1.0}
    d = int(0.3 * SR)
    y = dsp._return_delay(_burst(), SR, e)
    rms = lambda v: float(np.sqrt((v ** 2).mean()))
    for k in range(1, 5):
        here, there = (0, 1) if k % 2 else (1, 0)
        assert rms(_echo(y, k, d, there)) < 1e-6 * rms(_echo(y, k, d, here)), k


def test_a_mono_input_gives_a_mono_output_either_way():
    x = _burst()[:, :1]
    for pp in (0.0, 1.0):
        y = dsp._return_delay(x, SR, {"delay_ms": 250.0, "feedback": 0.5, "tone": 0.4, "ping_pong": pp})
        assert y.shape == x.shape and np.abs(y).max() > 0.0


def test_the_reference_monitor_is_the_same_law():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((SR, 2)) * 0.1
    for pp in (0.0, 1.0):
        e = {"delay_ms": 130.0, "feedback": 0.6, "tone": 0.3, "ping_pong": pp}
        assert np.array_equal(dsp._return_delay(x, SR, e),
                              reference_monitor._return_delay(x, float(SR), e))


def test_ping_pong_is_a_parameter_of_the_device():
    a = models.ReturnDelayAction(return_id="r", ping_pong=True)
    assert a.ping_pong is True and models.ReturnDelayAction(return_id="r").ping_pong is False
    assert session.Session.DEVICE_PARAMS["return_delay"]["ping_pong"][1:] == (0.0, 1.0)
