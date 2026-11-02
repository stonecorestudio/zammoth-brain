"""The Duck's law (2026-10-28): a drawn ducking curve, restarted by the song's beat or by the hits
of a trigger track - fitted to a reference curve-ducker measured as a black box (James's pick, B).

Measured (headless, and in the DAW with the reference keyed off a track; the session's scratchpad
has the probes):

* GAIN = 1 - mix x (1 - curve(phase)), exactly. A curve stretches with its LENGTH (1/8, 1/4, 1/2,
  1/1 of a whole note: 0.5, 1, 2, 4 quarters) and both channels get the same gain.
* SYNC: the phase is the song's quarter-note position over the length - the curve loops, phase 0 on
  the grid - from the tempo map (`tempo_map.TempoMap`, the Auto Filter's clock).
* TRACK (the reference's audio trigger): a hit on the trigger track restarts the curve, which plays
  ONCE and stops at its END marker, holding that value until the next hit (curve 1 stops at 63/64,
  so its dip before the next beat is never played). The gain eases from where it is to the curve's
  first value over GLIDE_MS (a square), ending on the hit. The detector: the trigger high-passed
  (Butterworth 2nd order at DET_HP_HZ), the louder channel's |x|, a peak follower (instant attack,
  DET_FAST_MS release) against a one-pole of it (DET_SLOW_MS): a hit is the follower over DET_THR_DB
  and DET_RISE_DB over the slow level - the moment it rises into that; it re-arms once the follower is back under DET_RISE_DB x
  DET_REARM over it, and never within DET_HOLD_MS of the last hit. The curve then starts where the
  running peak of the high-passed key first comes within PLACE_DB of its peak in the next
  PLACE_MS. On the reference's probes this agrees on ~107 of ~110 hits (it differs at three edges:
  an abrupt 40 Hz start, a 1 kHz burst at exactly -20 dBFS, a 150 ms swell); on a real kick loop
  each curve starts within 0.24 ms of the reference's (0.7 at worst), and the ducked loop nulls
  38 dB under what the reference changed. A slow synthetic attack (5-100 ms) starts up to ~4 ms
  early. There is no sensitivity knob, as in the reference: a trigger that never peaks near -20 dBFS
  above ~160 Hz never fires.
* The reference looks PLACE_MS + GLIDE_MS (16 ms) ahead. Here nothing does: the offline render sees
  the whole key, and the DAW's node plays the hits the brain finds with `onsets` on the trigger's
  audio (daw_bridge.attach_duck_hits) by its playhead - a sidechain reaches a node delayed by whatever
  latency the node reports, so a look-ahead cancelled itself (the export oracle, 2026-10-28). No Duck
  reports latency.
* BAND: only a one-pole low band (bilinear, no pre-warp) at band_hz is ducked; the rest is the input
  minus that band, so at full gain the two sum to the input exactly.

The curves are our own breakpoint list (phase, gain, bend): y = y0 + (y1 - y0) s^bend between two
points, fitted to each measured curve within 0.004 of gain. 1-8 duck, 9-10 sub-bass (slower
returns), 11-12 trim a kick's tail, 13-14 reverse ("suck"), 15-16 rhythm.
`DuckLaw.h` is written from this file (`python -m mix_engine.duck_law --write-header`).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

CURVES = (
    ((0, 0.12987, 1), (0.00024, 0.12285, 1), (0.00049, 0.11714, 0.5535), (0.00464, 0.0853, 0.8073), (0.02881, 0.05821, 0.8134), (0.07812, 0.01905, 3.3426), (0.09863, 0.03003, 1.3067), (0.13696, 0.13399, 1.0471), (0.15991, 0.22613, 0.9657), (0.25732, 0.64161, 0.9486), (0.29077, 0.76073, 0.8555), (0.34375, 0.89831, 0.9795), (0.34937, 0.90887, 0.7073), (0.42163, 0.99005, 0.4837), (0.46899, 1, 54.598), (0.98999, 1, 0.9869), (0.99902, 0.18003, 0.7502), (1, 0.12987, 1)),
    ((0, 0.13749, 0.5067), (0.00269, 0.0893, 0.7239), (0.05542, 0.01434, 6.7341), (0.06689, 0.01969, 1.2478), (0.09375, 0.10103, 1.0295), (0.1189, 0.20225, 1.001), (0.14014, 0.29133, 0.994), (0.15845, 0.36738, 0.9942), (0.17017, 0.4151, 0.9083), (0.26538, 0.7533, 0.9109), (0.30542, 0.85581, 0.8111), (0.3689, 0.95633, 0.8486), (0.40137, 0.98341, 0.9887), (0.40308, 0.98444, 0.9882), (0.40479, 0.98544, 0.5585), (0.45752, 0.9999, 0.1121), (0.99487, 1, 1), (0.99512, 0.97774, 0.9995), (0.99927, 0.221, 1), (0.99951, 0.18181, 0.7137), (1, 0.13749, 1)),
    ((0, 0.39216, 1), (0.00098, 0.31373, 0.9448), (0.00391, 0.08538, 0.8279), (0.00464, 0.05047, 0.6033), (0.00708, 0.01074, 0.1433), (0.08838, 0, 2.7573), (0.21948, 0.11547, 1.2555), (0.33252, 0.61756, 0.966), (0.35522, 0.732, 0.8364), (0.39966, 0.89589, 0.9762), (0.4043, 0.90762, 0.6994), (0.45703, 0.9856, 0.7168), (0.5, 1, 54.598), (0.99243, 0.99998, 1.0001), (1, 0.39216, 1)),
    ((0, 0.39216, 1), (0.00073, 0.33333, 0.9591), (0.00342, 0.12173, 0.8021), (0.00415, 0.08804, 0.6453), (0.00732, 0.03895, 0.9497), (0.00781, 0.03553, 0.9534), (0.0083, 0.03255, 0.4991), (0.02563, 0.00299, 0.3626), (0.05811, 0, 2.7887), (0.24878, 0.10947, 1.245), (0.29248, 0.21647, 1.1123), (0.31934, 0.32527, 1.0379), (0.37402, 0.62186, 0.9569), (0.38965, 0.69928, 0.9029), (0.41455, 0.79736, 0.7985), (0.46265, 0.90765, 0.9749), (0.46851, 0.91643, 0.6683), (0.5625, 0.99065, 0.4728), (0.63501, 1, 54.598), (0.99243, 0.99998, 1.0001), (1, 0.39216, 1)),
    ((0, 0.2439, 1), (0.00024, 0.21951, 1), (0.00049, 0.19512, 0.8466), (0.00146, 0.10931, 0.8302), (0.00195, 0.08709, 0.7026), (0.00391, 0.04904, 0.5302), (0.01538, 0.00844, 0.3461), (0.05518, 0, 2.6728), (0.18457, 0.03802, 1.1826), (0.21851, 0.07413, 1.0782), (0.23291, 0.09637, 1.0584), (0.24292, 0.1154, 1.4185), (0.2876, 0.27647, 1.104), (0.31982, 0.51853, 0.9473), (0.33203, 0.60749, 0.8704), (0.35547, 0.73269, 0.8193), (0.39062, 0.84042, 0.8997), (0.41309, 0.88138, 0.742), (0.50098, 0.96455, 0.811), (0.55835, 0.98761, 0.47), (0.67627, 1, 54.598), (0.99243, 0.99998, 1.0001), (1, 0.2439, 1)),
    ((0, 0.06204, 1), (0.00024, 0.05399, 0.5419), (0.00464, 0.01023, 0.6111), (0.0105, 0.00152, 0.0183), (0.16187, 0.00084, 2.1179), (0.22925, 0.03851, 1.4431), (0.26953, 0.13577, 1.1153), (0.27905, 0.18216, 1.1339), (0.29028, 0.26199, 1.1112), (0.3208, 0.619, 0.8909), (0.33398, 0.74903, 0.9798), (0.33594, 0.7642, 0.7615), (0.36328, 0.89699, 0.9694), (0.36694, 0.90761, 0.9722), (0.37036, 0.91663, 0.6603), (0.42529, 0.99052, 0.4714), (0.46948, 1, 54.598), (0.99243, 0.99997, 0.9951), (0.99902, 0.13039, 0.6686), (1, 0.06204, 1)),
    ((0, 0.06996, 0.5644), (0.00269, 0.02061, 0.7007), (0.0061, 0.00698, 0.3741), (0.02075, 0, 2.5983), (0.09424, 0.06529, 1.276), (0.12817, 0.18409, 1.0747), (0.1438, 0.27249, 0.9826), (0.19214, 0.5918, 0.945), (0.20947, 0.68701, 0.8647), (0.24292, 0.82114, 0.7746), (0.29858, 0.93468, 0.8531), (0.33276, 0.96948, 0.5517), (0.41895, 0.99972, 0.1224), (0.99487, 1, 1), (0.99512, 0.97561, 1), (0.99731, 0.53659, 0.998), (0.99927, 0.14666, 0.5934), (1, 0.06996, 1)),
    ((0, 0.0661, 0.4883), (0.00342, 0.0144, 0.4627), (0.00854, 0.00741, 1.7052), (0.05322, 0.05666, 1.2607), (0.073, 0.12008, 1.05), (0.07666, 0.13805, 1.3848), (0.09302, 0.27855, 1.0554), (0.10913, 0.56813, 0.8944), (0.11694, 0.67725, 0.7942), (0.13501, 0.82106, 0.9683), (0.1377, 0.83471, 0.9658), (0.14062, 0.84808, 0.968), (0.14355, 0.86008, 0.7704), (0.17676, 0.94336, 0.8608), (0.19824, 0.97064, 0.5352), (0.26709, 0.99979, 0.1117), (0.99487, 1, 1), (0.99512, 0.97561, 1), (0.99731, 0.53659, 0.9984), (0.99927, 0.1466, 0.603), (1, 0.0661, 1)),
    ((0, 0, 3.9934), (0.36182, 0.10127, 1.2511), (0.41406, 0.18625, 1.0886), (0.43066, 0.23005, 1.0654), (0.44214, 0.26828, 1.0339), (0.44775, 0.29008, 1.2624), (0.47632, 0.45522, 1.1181), (0.48438, 0.53545, 1.2604), (0.49292, 0.67635, 1.3536), (0.5, 1, 46.7614), (0.92603, 0.90932, 1.3587), (0.93677, 0.76336, 1.1026), (0.94116, 0.65418, 1.0364), (0.9436, 0.5798, 1.0141), (0.94531, 0.52357, 0.9177), (0.95361, 0.26392, 0.9118), (0.95703, 0.18833, 0.7922), (0.96533, 0.08107, 0.7456), (0.97656, 0.0188, 0.7757), (0.98315, 0.0045, 0.2962), (1, 0, 1)),
    ((0, 0, 54.598), (0.47046, 0.06573, 1.3783), (0.47803, 0.26829, 1), (0.4812, 0.37398, 1), (0.48755, 0.58537, 1), (0.5, 1, 17.0313), (0.85278, 0.89339, 1.0615), (0.85791, 0.86825, 1.0527), (0.86206, 0.844, 1.0461), (0.86548, 0.82073, 1.247), (0.90332, 0.34947, 0.9249), (0.9126, 0.23262, 0.8379), (0.92578, 0.12227, 0.9067), (0.93335, 0.08251, 0.9751), (0.9353, 0.07414, 0.6334), (0.96777, 0.00472, 0.2994), (1, 0, 1)),
    ((0, 1, 38.6321), (0.17822, 0.99431, 1.9109), (0.18872, 0.95673, 1.1929), (0.19336, 0.90389, 1.0215), (0.2063, 0.69922, 0.9766), (0.24805, 0.03763, 0.6065), (0.25317, 0.00158, 0.1051), (0.99976, -0.00038, 1), (1, 1, 1)),
    ((0, 1, 30.7737), (0.18164, 0.99215, 1.3236), (0.21436, 0.90742, 1.0079), (0.24292, 0.81022, 0.9879), (0.27124, 0.71506, 0.9874), (0.29175, 0.64894, 0.9868), (0.31055, 0.5908, 0.9905), (0.323, 0.55368, 0.8881), (0.43994, 0.26451, 0.9114), (0.49585, 0.16797, 0.7803), (0.61328, 0.05031, 0.6011), (0.76733, 0.00173, 0.2915), (0.99976, -0.00038, 1), (1, 1, 1)),
    ((0, 0.95573, 0.5966), (0.00757, 0.99143, 0.6154), (0.0166, 0.9988, 0.1541), (0.16064, 1, 3.7165), (0.53442, 0.87745, 1.2692), (0.59058, 0.76904, 1.133), (0.61719, 0.67936, 1.0556), (0.62769, 0.63258, 1.0448), (0.6355, 0.59192, 1.2287), (0.6626, 0.38623, 1.088), (0.67017, 0.29512, 1.0778), (0.6875, 0, 3.9258), (0.9043, 0.11041, 1.0942), (0.91919, 0.14592, 1.287), (0.95312, 0.28607, 1.0843), (0.96289, 0.35259, 1.1113), (0.97363, 0.45058, 1.1699), (0.98511, 0.60737, 1.1862), (0.99731, 0.91106, 0.7966), (1, 0.95573, 1)),
    ((0, 0.95573, 0.5893), (0.00781, 0.99185, 0.6017), (0.01733, 0.99901, 0.0919), (0.23242, 0.99981, 2.2861), (0.32031, 0.93576, 1.1394), (0.34009, 0.89201, 1.0662), (0.34937, 0.86427, 1.297), (0.38501, 0.68098, 1.0767), (0.39624, 0.58713, 1.1051), (0.41113, 0.42321, 1.1057), (0.4375, 0, 3.9615), (0.81982, 0.08545, 1.1317), (0.85913, 0.12795, 1.099), (0.88403, 0.1658, 1.0509), (0.89575, 0.18822, 1.3292), (0.95215, 0.36967, 1.1226), (0.96777, 0.46243, 1.2414), (0.98486, 0.62907, 1.0299), (0.98633, 0.65048, 1.2524), (0.99731, 0.91106, 0.7966), (1, 0.95573, 1)),
    ((0, 0, 8.6824), (0.55859, 0.08314, 1.1316), (0.59253, 0.13032, 1.1016), (0.61597, 0.17627, 1.0726), (0.63306, 0.21996, 1.0522), (0.646, 0.25998, 1.0396), (0.65601, 0.29566, 1.0317), (0.66406, 0.32774, 1.1565), (0.69971, 0.51596, 1.055), (0.71191, 0.60369, 1.1035), (0.75244, 1, 1.0001), (0.7876, 0.83838, 0.9694), (0.93481, 0.18256, 0.9408), (0.9668, 0.06157, 0.7788), (0.98999, 0.00249, 0.2983), (1, 0, 1)),
    ((0, 0.07333, 0.443), (0.00513, 0.04175, 1.9305), (0.01196, 0.05127, 1.0345), (0.09619, 0.34237, 0.9707), (0.1311, 0.45529, 0.933), (0.17456, 0.57642, 0.763), (0.22266, 0.66204, 0.5412), (0.23389, 0.66665, 2.2954), (0.26392, 0.60197, 1.0862), (0.26807, 0.57677, 1.3403), (0.27979, 0.45007, 1.1196), (0.28271, 0.39094, 1.1956), (0.28564, 0.29886, 1.1844), (0.28857, 0.11518, 0.8032), (0.28931, 0.08586, 0.6202), (0.29346, 0.03469, 0.4988), (0.3125, 0.00337, 0.3774), (0.3479, 0, 2.6925), (0.62256, 0.16977, 1.0403), (0.63867, 0.20012, 1.2005), (0.77393, 0.59141, 1.0049), (0.8186, 0.75618, 0.907), (0.86646, 0.91229, 0.7323), (0.91089, 0.99236, 0.4593), (0.93555, 1, 43.0944), (0.98975, 0.96612, 1.3629), (0.99072, 0.91877, 1.0311), (0.99731, 0.26829, 1), (0.99756, 0.2439, 0.8814), (0.99902, 0.1089, 0.7171), (1, 0.07333, 1)),
)
# where each curve stops when a hit restarts it (TRACK); SYNC loops the whole cycle
END_PHASE = (63 / 64,) * 8 + (0.895, 0.79, 63 / 64, 63 / 64, 1.0, 1.0, 1.0, 63 / 64)
CURVE_NAMES = ("Duck 1", "Duck 2", "Duck 3", "Duck 4", "Duck 5", "Duck 6", "Duck 7", "Duck 8",
               "Sub 1", "Sub 2", "Trim 1", "Trim 2", "Reverse 1", "Reverse 2", "Rhythm 1", "Rhythm 2")
NUM_CURVES = len(CURVES)
DEFAULT_CURVE = 1                      # 1-based, as the face numbers them

LENGTH_QUARTERS = (0.5, 1.0, 2.0, 4.0)
LENGTH_NAMES = ("1/8", "1/4", "1/2", "1/1")
DEFAULT_LENGTH = 1                     # 1/4

TRIGGERS = ("sync", "track")           # the stored words, by index
DEFAULT_TRIGGER = 0

BAND_MIN_HZ, BAND_MAX_HZ, DEFAULT_BAND_HZ = 20.0, 5120.0, 320.0

GLIDE_MS = 6.0
DET_HP_HZ = 160.0
DET_THR_DB = -20.0
DET_RISE_DB = 6.0
DET_REARM = 0.3
DET_FAST_MS = 10.0
DET_SLOW_MS = 20.0
DET_HOLD_MS = 32.0
PLACE_DB = 8.0
PLACE_MS = 10.0


def _index(v, names, default: int) -> int:
    """A stored index or its word (case-insensitive) - an unknown one is the default."""
    if isinstance(v, str):
        w = v.strip().lower()
        for i, nm in enumerate(names):
            if nm.lower() == w:
                return i
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        return default
    return int(min(len(names) - 1, max(0, round(f))))


def trigger_index(v) -> int:
    return _index(v, TRIGGERS, DEFAULT_TRIGGER)


def length_index(v) -> int:
    return _index(v, LENGTH_NAMES, DEFAULT_LENGTH)


def curve_number(v) -> int:
    """1..16 - a number, or a name ("Sub 1")."""
    if isinstance(v, str) and not v.strip().lstrip("-").replace(".", "", 1).isdigit():
        return _index(v, CURVE_NAMES, DEFAULT_CURVE - 1) + 1
    try:
        f = float(v)
    except (TypeError, ValueError):
        return DEFAULT_CURVE
    return int(min(NUM_CURVES, max(1, round(f)))) if math.isfinite(f) else DEFAULT_CURVE


def values(entry: dict) -> dict:
    """An entry's knobs, held to their ranges (the native `parse` holds them the same way)."""
    mix = float(entry.get("mix", 1.0))
    band_hz = float(entry.get("band_hz", DEFAULT_BAND_HZ))
    return {
        "trigger": trigger_index(entry.get("trigger", DEFAULT_TRIGGER)),
        "curve": curve_number(entry.get("curve", DEFAULT_CURVE)),
        "length": length_index(entry.get("length", DEFAULT_LENGTH)),
        "mix": min(1.0, max(0.0, mix)) if math.isfinite(mix) else 1.0,
        "band": bool(entry.get("band", False)),
        "band_hz": (min(BAND_MAX_HZ, max(BAND_MIN_HZ, band_hz))
                    if math.isfinite(band_hz) else DEFAULT_BAND_HZ),
    }


def from_legacy(entry: dict) -> dict:
    """A Duck saved before the curve law (threshold / depth / attack / release, keyed by a track):
    read once as a TRACK duck on curve 1 at 1/4, its depth as the mix that cuts as far
    (mix = 1 - 10^(-depth/20)); threshold, attack and release have no place in the law."""
    out = {k: v for k, v in entry.items()
           if k not in ("threshold_db", "depth_db", "attack_ms", "release_ms")}
    depth = float(entry.get("depth_db", 6.0))
    depth = depth if math.isfinite(depth) else 6.0
    out.update(trigger=1, curve=DEFAULT_CURVE, length=DEFAULT_LENGTH,
               mix=round(min(1.0, max(0.0, 1.0 - 10.0 ** (-depth / 20.0))), 4),
               band=False, band_hz=DEFAULT_BAND_HZ)
    return out


def is_legacy(entry: dict) -> bool:
    return "curve" not in entry and any(
        k in entry for k in ("threshold_db", "depth_db", "attack_ms", "release_ms"))


def convert_legacy(obj) -> int:
    """Walk anything read from an old file and convert every Duck in it in place (a dict of type
    "duck" holding the old knobs): the session's chains, its undo snapshots, its recorded actions.
    Returns how many it converted (session STATE_VERSION 5)."""
    n = 0
    if isinstance(obj, dict):
        if obj.get("type") == "duck" and is_legacy(obj):
            conv = from_legacy(obj)
            obj.clear()
            obj.update(conv)
            n += 1
        for v in obj.values():
            if isinstance(v, (dict, list)):
                n += convert_legacy(v)
    elif isinstance(obj, list):
        for v in obj:
            n += convert_legacy(v)
    return n


def curve_at(n: int, phase) -> np.ndarray:
    """Curve `n` (1-based) at `phase` (0..1, held there)."""
    pts = CURVES[n - 1]
    ph = np.clip(np.asarray(phase, dtype=np.float64), 0.0, 1.0)
    out = np.empty_like(ph)
    for k in range(len(pts) - 1):
        x0, y0, e = pts[k]
        x1, y1, _ = pts[k + 1]
        m = (ph >= x0) & (ph <= x1) if k == len(pts) - 2 else (ph >= x0) & (ph < x1)
        s = (ph[m] - x0) / (x1 - x0)
        out[m] = y0 + (y1 - y0) * s ** e
    return out


def hp_coeffs(sr: float) -> tuple:
    """The detector's 2nd-order Butterworth high-pass (bilinear, pre-warped) - b0 b1 b2 a1 a2."""
    k = math.tan(math.pi * DET_HP_HZ / sr)
    norm = 1.0 / (1.0 + math.sqrt(2.0) * k + k * k)
    return (norm, -2.0 * norm, norm, 2.0 * (k * k - 1.0) * norm,
            (1.0 - math.sqrt(2.0) * k + k * k) * norm)


def band_coeffs(sr: float, hz: float) -> tuple:
    """The band split's one-pole low-pass (bilinear, no pre-warp): y = b (x + x1) - a1 y1."""
    kk = 2.0 * math.pi * hz / (2.0 * sr)
    return kk / (1.0 + kk), (kk - 1.0) / (1.0 + kk)


def _biquad(x: np.ndarray, c: tuple) -> np.ndarray:
    from scipy.signal import lfilter
    b0, b1, b2, a1, a2 = c
    return lfilter([b0, b1, b2], [1.0, a1, a2], x, axis=0)


def key_level(key: np.ndarray, sr: float) -> np.ndarray:
    """The detector's input: the louder channel of the high-passed trigger, |x|."""
    k = np.asarray(key, dtype=np.float64)
    if k.ndim == 1:
        k = k[:, np.newaxis]
    return np.max(np.abs(_biquad(k, hp_coeffs(sr))), axis=1)


def onsets(key: np.ndarray, sr: float) -> np.ndarray:
    """Where each hit starts its curve (sample indices), from the trigger's audio."""
    a = key_level(key, sr)
    n = a.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    # fast[i] = max(a[i], kf fast[i-1]): a running max in the log domain
    c = -1.0 / (DET_FAST_MS * 1e-3 * sr)
    idx = np.arange(n, dtype=np.float64)
    fast = np.exp(np.maximum.accumulate(np.log(np.maximum(a, 1e-30)) - idx * c) + idx * c)
    from scipy.signal import lfilter
    ks = math.exp(-1.0 / (DET_SLOW_MS * 1e-3 * sr))
    slow = lfilter([1.0 - ks], [1.0, -ks], fast, zi=[ks * 1e-9])[0]
    lt, r = 10.0 ** (DET_THR_DB / 20.0), 10.0 ** (DET_RISE_DB / 20.0)
    ra = 10.0 ** (DET_RISE_DB * DET_REARM / 20.0)
    on = (fast > lt) & (fast > r * slow)
    fire = np.flatnonzero(on & ~np.r_[False, on[:-1]])        # a hit is a RISE into the condition
    rearm = np.flatnonzero(fast < ra * slow)
    hold = int(round(DET_HOLD_MS * 1e-3 * sr))
    win = max(1, int(round(PLACE_MS * 1e-3 * sr)))
    thr = 10.0 ** (-PLACE_DB / 20.0)
    out = []
    pos = 0
    while True:
        j = np.searchsorted(fire, pos)
        if j >= len(fire):
            break
        d = int(fire[j])
        w = np.maximum.accumulate(a[d:d + win])
        out.append(d + int(np.argmax(w >= w[-1] * thr)))
        r_i = np.searchsorted(rearm, d + 1)
        if r_i >= len(rearm):
            break
        pos = max(int(rearm[r_i]), d + hold)
    return np.asarray(out, dtype=np.int64)


def quarters(n: int, sr: float, tempo=None, offset: int = 0) -> np.ndarray:
    """The song's quarter-note position at each sample (the Auto Filter's clock)."""
    if tempo is None:
        from .tempo_map import TempoMap
        tempo = TempoMap.from_fact(None)
    t = np.arange(offset, offset + n, dtype=np.float64) / float(sr)
    return np.asarray(tempo.ppq_at(t), dtype=np.float64)


def curve_gain(n: int, sr: float, v: dict, q: np.ndarray, hits=None) -> np.ndarray:
    """The curve's gain before the mix: SYNC from the clock `q`; TRACK from `hits` (None or empty:
    unity - no trigger heard)."""
    lq = LENGTH_QUARTERS[v["length"]]
    cn = v["curve"]
    if v["trigger"] == 0:
        return curve_at(cn, np.mod(q / lq, 1.0))
    g = np.ones(n, dtype=np.float64)
    if hits is None or len(hits) == 0:
        return g
    end = END_PHASE[cn - 1]
    hold = float(curve_at(cn, np.array([end]))[0])
    c0 = float(curve_at(cn, np.array([0.0]))[0])
    G = int(round(GLIDE_MS * 1e-3 * sr))
    hits = [int(h) for h in hits if 0 <= h < n]
    for k, s in enumerate(hits):
        nxt = hits[k + 1] if k + 1 < len(hits) else n
        ph = (q[s:nxt] - q[s]) / lq
        seg = curve_at(cn, np.minimum(ph, end))
        g[s:nxt] = np.where(ph < end, seg, hold)
        a = max(0, s - G)
        if s > a:
            v0 = g[a]
            t = (np.arange(a, s) - a + 1) / (s - a)
            g[a:s] = v0 + (c0 - v0) * t * t
    return g


def gain(n: int, sr: float, entry: dict, tempo=None, key=None, offset: int = 0) -> np.ndarray:
    """The Duck's gain at each sample (mix applied): `key` = the trigger's audio on this stem's
    timeline (TRACK), None when no trigger is heard."""
    v = values(entry)
    q = quarters(n, sr, tempo, offset)
    hits = onsets(key, sr) if (v["trigger"] == 1 and key is not None) else None
    g = curve_gain(n, sr, v, q, hits)
    return 1.0 - v["mix"] * (1.0 - g)


def apply(audio: np.ndarray, sr: float, entry: dict, tempo=None, key=None,
          offset: int = 0) -> np.ndarray:
    """(frames, channels) -> the ducked audio; the band split ducks only the low band."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    v = values(entry)
    if v["trigger"] == 1 and key is None:
        return x
    g = gain(x.shape[0], sr, entry, tempo, key, offset)[:, np.newaxis]
    if not v["band"]:
        return x * g
    from scipy.signal import lfilter
    b, a1 = band_coeffs(sr, v["band_hz"])
    low = lfilter([b, b], [1.0, a1], x, axis=0)
    return (x - low) + low * g


def latency_samples(entry: dict, sr: float) -> int:
    """What the native delays its audio by: nothing, in either trigger mode (see the module's note)."""
    return 0


HEADER = Path(__file__).resolve().parents[2] / "native" / "Source" / "DuckLaw.h"


def header_text() -> str:
    def f(v) -> str:
        return repr(float(v)) + "f"
    starts, pts = [], []
    for c in CURVES:
        starts.append(len(pts))
        pts.extend(c)
    starts.append(len(pts))
    def d(v) -> str:
        return repr(float(v))
    rows = ",\n".join("        { %s, %s, %s }" % (d(x), d(y), d(e)) for x, y, e in pts)
    L = [
        "#pragma once",
        "// GENERATED by `python -m mix_engine.duck_law --write-header` from server/mix_engine/duck_law.py -",
        "// the Duck's law (that file says how it was measured and what each constant does). Do not edit:",
        "// change the Python table and write this again; test_duck_law.py holds the two equal.",
        "",
        "namespace ducklaw",
        "{",
        f"    static constexpr int kNumCurves = {NUM_CURVES}, kNumPoints = {len(pts)};",
        "    static constexpr int kCurveStart[kNumCurves + 1] = { " + ", ".join(str(s) for s in starts) + " };",
        "    static constexpr double kPoints[kNumPoints][3] = {   // phase, gain, bend",
        rows,
        "    };",
        "    static constexpr double kEndPhase[kNumCurves] = { " + ", ".join(d(e) for e in END_PHASE) + " };",
        "    static constexpr const char* kCurveName[kNumCurves] = { " + ", ".join(f'"{n}"' for n in CURVE_NAMES) + " };",
        "    static constexpr int kNumLengths = 4;",
        "    static constexpr double kLengthQuarters[kNumLengths] = { " + ", ".join(d(x) for x in LENGTH_QUARTERS) + " };",
        "    static constexpr const char* kLengthName[kNumLengths] = { " + ", ".join(f'"{n}"' for n in LENGTH_NAMES) + " };",
        f"    static constexpr int kDefaultCurve = {DEFAULT_CURVE}, kDefaultLength = {DEFAULT_LENGTH}, kDefaultTrigger = {DEFAULT_TRIGGER};",
        f"    static constexpr float kBandMinHz = {f(BAND_MIN_HZ)}, kBandMaxHz = {f(BAND_MAX_HZ)}, kDefaultBandHz = {f(DEFAULT_BAND_HZ)};",
        f"    static constexpr double kGlideMs = {GLIDE_MS!r}, kDetHpHz = {DET_HP_HZ!r}, kDetThrDb = {DET_THR_DB!r}, kDetRiseDb = {DET_RISE_DB!r};",
        f"    static constexpr double kDetRearm = {DET_REARM!r}, kDetFastMs = {DET_FAST_MS!r}, kDetSlowMs = {DET_SLOW_MS!r}, kDetHoldMs = {DET_HOLD_MS!r};",
        f"    static constexpr double kPlaceDb = {PLACE_DB!r}, kPlaceMs = {PLACE_MS!r};",
        "}",
        "",
    ]
    return "\n".join(L)


if __name__ == "__main__":
    if "--write-header" in sys.argv:
        HEADER.write_text(header_text())
        print(f"wrote {HEADER}")
