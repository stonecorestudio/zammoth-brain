"""The limiter's inter-sample peak detector - the native limiter's, transcribed (reader E4·R7).

The DAW's limiter reads a peak through a 4x polyphase FIR: a Blackman-windowed sinc,
``firwin(129, 1/4) * 4``, split into four phases of 33 taps. Phase 0 is an impulse (the sample
itself); the three below are the inter-sample points. It replaced a 4-point Catmull-Rom cubic on
2026-09-08 because the cubic under-reads content near the top of the band by up to a decibel, and
a detector that under-reads is a ceiling that is not honoured.

The offline limiter kept the cubic. Its brickwall therefore let those peaks through, and
``dsp._master_limiter`` - which measures the contractual true peak afterwards and trims the WHOLE
FILE by the excess - turned every limited master down by 1-2 dB that the DAW's own export never
loses: the number the agent measured was not the number the user bounced.

ONE table: ``TP_PHASES`` is ``kTpPhase`` in native/Source/MonitorChain.cpp, value for value, and
``server/tests/test_tp_detector_parity.py`` re-parses the C++ and fails when they part. Change
both or neither.
"""
from __future__ import annotations

import numpy as np

#: The sample the estimate is centred on, counted back from the newest tap (``kTpCentre``).
TP_CENTRE = 16

#: The three inter-sample phases (``kTpPhase``), 33 taps each.
TP_PHASES = np.array([
    [-0.00000310, +0.00008451, -0.00030703, +0.00073470, -0.00146308, +0.00262214,
     -0.00437952, +0.00694728, -0.01059806, +0.01570326, -0.02282188, +0.03291466,
     -0.04791543, +0.07256290, -0.12252078, +0.29744496, +0.89941944, -0.17566267,
     +0.09230624, -0.05850804, +0.03961030, -0.02740966, +0.01896425, -0.01293992,
     +0.00861708, -0.00554678, +0.00341395, -0.00197923, +0.00105402, -0.00049003,
     +0.00017483, -0.00002903, +0.00000000],
    [-0.00001788, +0.00017666, -0.00055374, +0.00125036, -0.00241349, +0.00423973,
     -0.00698065, +0.01095458, -0.01657513, +0.02441782, -0.03537565, +0.05104157,
     -0.07476693, +0.11527890, -0.20477498, +0.63410291, +0.63410291, -0.20477498,
     +0.11527890, -0.07476693, +0.05104157, -0.03537565, +0.02441782, -0.01657513,
     +0.01095458, -0.00698065, +0.00423973, -0.00241349, +0.00125036, -0.00055374,
     +0.00017666, -0.00001788, +0.00000000],
    [-0.00002903, +0.00017483, -0.00049003, +0.00105402, -0.00197923, +0.00341395,
     -0.00554678, +0.00861708, -0.01293992, +0.01896425, -0.02740966, +0.03961030,
     -0.05850804, +0.09230624, -0.17566267, +0.89941944, +0.29744496, -0.12252078,
     +0.07256290, -0.04791543, +0.03291466, -0.02282188, +0.01570326, -0.01059806,
     +0.00694728, -0.00437952, +0.00262214, -0.00146308, +0.00073470, -0.00030703,
     +0.00008451, -0.00000310, +0.00000000],
], dtype=np.float64)


def intersample_peak(audio: np.ndarray) -> np.ndarray:
    """Per frame and channel, the peak around the sample: the sample's own magnitude, the three
    inter-sample estimates after it, and the crest between any two of those four points.
    ``(frames, channels)`` in, the same shape out.

    The native computes ``sum_k kTpPhase[p][k] * h[newest - k]`` and pairs it with the sample
    ``TP_CENTRE`` behind the newest; offline there is no latency to absorb, so the same sums are
    simply aligned on that sample. Silence is assumed outside the file, as the native's history
    starts at zero.

    THE CREST BETWEEN THE POINTS (user QA 2026-10-08, D28; ``tpStep`` in MonitorChain.cpp). Four
    points a sample are not the peak: a crest midway between two of them is under-read by up to
    0.2 dB on content rich in highs, and the limiter let that through. Round every point that
    is a local maximum of |x| the vertex of the parabola through it and its two neighbours is
    taken - across the sample boundary too. It never reads less than the largest of the four.
    """
    audio = np.asarray(audio, dtype=np.float64)
    peak = np.abs(audio)
    frames = audio.shape[0]
    if frames == 0:
        return peak
    for ch in range(audio.shape[1]):
        # the 4x-rate points in time order: the sample, then +1/4, +1/2, +3/4 of a sample
        points = np.empty((frames, 4))
        points[:, 0] = audio[:, ch]
        for p, phase in enumerate(TP_PHASES):
            points[:, p + 1] = np.convolve(audio[:, ch], phase)[TP_CENTRE:TP_CENTRE + frames]
        seq = points.reshape(-1)
        mag = np.abs(seq)
        est = mag.copy()
        lo, mid, hi = mag[:-2], mag[1:-1], mag[2:]
        crest = ((mid >= lo) & (mid >= hi)
                 & (seq[1:-1] * seq[:-2] > 0.0) & (seq[1:-1] * seq[2:] > 0.0))
        a = 0.5 * (lo + hi) - mid
        b = 0.5 * (hi - lo)
        bends = crest & (a < 0.0)
        est[1:-1][bends] = mid[bends] - (b[bends] ** 2) / (4.0 * a[bends])
        np.maximum(peak[:, ch], est.reshape(frames, 4).max(axis=1), out=peak[:, ch])
    return peak
