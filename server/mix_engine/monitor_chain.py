"""Phase 40 (native VST) — the monitor-chain param contract.

Translate an ``effective_master_chain`` (``session.effective_master_chain``) into
the flat, fully-resolved parameter set the native plugin's real-time MonitorChain
consumes. The native engine must read the *same numbers* the offline engine
renders, so the live monitor and the deliverable agree (the determinism reset is
"native engine vs the Python oracle", see docs/09-vst.md / the Phase 40 plan).

This module is pure state -> params; it imports no DSP. The numpy reference that
proves the params render identically lives in ``reference_monitor.py``; the C++
``MonitorChain`` is a transcription of that reference.

Step 1 mirrors the mastering core (gain / EQ / comp / limiter) in real time.
Everything else in the chain (transpose, match-EQ FIR, dynamic EQ, multiband
comp) stays offline ("print-only") until Step 2 ports it natively; those types
are reported separately so the UI can flag "heard in the printed master".
"""
from __future__ import annotations

import math
from typing import Optional

from .models import BAND_EDGES_HZ, BAND_LABELS
from .reference_monitor import auto_filter_values   # Phase 63: the Auto Filter as the law reads it

# Actions the native real-time monitor mirrors. gain/eq/limiter/saturation are
# bit-matched to the engine (saturation is a memoryless waveshaper); comp +
# dynamic_eq + mb_comp + match_eq are real-time approximations of the offline
# processors (the offline export stays faithful).
MONITOR_TYPES = frozenset({"gain", "eq", "comp", "limiter", "dynamic_eq",
                           "mb_comp", "match_eq", "saturation", "width", "utility",
                           "deess", "transient", "band_width", "imager", "duck", "clip", "gate",
                           "tilt", "ott", "overdrive",   # 2026-07-07
                           "chorus", "flanger",   # 2026-09-06: modulation family
                           "auto_filter",         # Phase 63: the family's third
                           # blank-return wet devices (rendered at their chain position on a return)
                           "return_reverb", "return_delay", "return_convolution"})
TILT_Q = 0.5   # gentle shelf slope for the one-knob tilt EQ (mirror dsp.TILT_Q)


def _range(entry: dict) -> dict:
    """The entry's `time_range_s`, passed through to the native monitor.

    The real-time chain gates a ranged device on the playhead (MonitorChain::inRange),
    so a region edit is confined in the monitor AND in the DAW's bounce, the way the
    offline render already confines it.
    """
    rng = entry.get("time_range_s")
    if not rng or len(rng) != 2:
        return {}
    return {"time_range_s": [float(rng[0]), float(rng[1])]}
# The agent's gain rides (a ranged `gain` move). They are not chain DEVICES — no tile, no
# parameters to edit — so they travel beside `entries`, as `rides`, and the native chain runs
# them as one stage after its devices: `dsp._ride_envelope`'s law, sample for sample
# (MonitorChain::applyRides). Until the 2026-10-02 full audit (reader S4·R29) a ride was
# `print_only`: Auto Mix's "level vocal phrases" was in session.json, the ledger and every number
# the agent measured, and in neither the speakers nor the export. The native chain holds this
# many; any past it stay print_only, so the snapshot still says what is not heard.
MAX_NATIVE_RIDES = 64
# Chain entries the real-time monitor cannot yet reproduce — rendered offline.
PRINT_ONLY_TYPES = frozenset({"transpose"})
# Chain entries that are heard NOWHERE — pure analyzers. They are not `print_only`,
# whose documented meaning is "the monitor cannot reproduce this, you hear it in the
# printed master": the spectrogram is a render no-op in BOTH engines, so reporting it
# as print_only told the reader the export would sound different. It does not.
INERT_TYPES = frozenset({"spectrogram"})



def dock_can_name(entry_type) -> bool:
    """True for a chain entry the NATIVE chain holds as an entry - so the dock draws it and can
    name it in an order. Everything else is pushed as `print_only` (a `tune`, a `transpose`,
    the offline-only `spectral_denoise`): in session.json, in no node's chain. The one question
    `session._dock_order` asks (reader E2·R4); `monitor_chain_params` below is what it answers
    for, and `test_reader_p2_py2` pins the two together."""
    return entry_type in MONITOR_TYPES or entry_type in INERT_TYPES


# "send_return" is a synthetic entry the gateway injects (it is not an engine
# action `type`): one per send on the monitored source, carrying the send level
# and its target return's resolved chain (Phase 38 reverb/delay). The native
# monitor adds the parallel wet locally; because the return chain is linear, the
# DAW-summed per-instance wet equals the offline shared return (see the fidelity
# test). Bit-exact tier, like saturation/width.

# Match-EQ band geometric centers (Hz). The live monitor approximates the engine's offline
# linear-phase FIR as one broad peaking filter per band (zero latency, min-phase).
#
# The old comment here said "the export keeps the linear-phase FIR". In the DAW that is FALSE —
# the export is a bounce of the native graph, so this bank IS the deliverable — and the two were
# far apart: feeding the raw band gains straight to the bells measured up to 8.4 dB from the FIR,
# because adjacent 0.7-Q bells overlap heavily and their dB contributions add. A "-3 dB bass"
# correction landed as -4.8 dB here and +0.7 dB offline (2026-09-08 device audit, 04-F4).
#
# So the bank is FITTED to the FIR's own magnitude response instead (`_fit_match_eq_gains`).
_BAND_CENTER_HZ = {b: round(math.sqrt(lo * hi), 1) for b, lo, hi
                   in zip(BAND_LABELS, BAND_EDGES_HZ[:-1], BAND_EDGES_HZ[1:])}
MATCH_EQ_Q = 0.7
# The fit's DEFAULT reference rate + grid. Fitting at a fixed 48 kHz is as good as fitting at the
# project's own rate NEAR it (44.1 kHz: 1.42 dB vs 1.37 dB worst case; 96 kHz: 0.47 vs 0.64, i.e.
# better) — but only near it. Below ~32 kHz the two disagree about which bands EXIST:
# `dsp.match_eq_fir` breaks out of its band loop at Nyquist, so on a 22.05 kHz project (an
# ordinary decoded mp3) the offline FIR drops the air band entirely while a 48 kHz fit still
# includes it — measured 3.68 dB of divergence against 1.20 dB for the raw gains, i.e. the fit
# made it WORSE, and in the half of the parity triangle that IS the deliverable in the DAW.
# So the rate is threaded through after all; 48 kHz is only the fallback for callers that have
# no project (2026-09-08 pre-merge review).
_MATCH_FIT_SR = 48000
_MATCH_FIT_LO_HZ, _MATCH_FIT_HI_HZ, _MATCH_FIT_POINTS = 30.0, 16000.0, 400


def _fit_match_eq_gains(band_gains_db: dict, sample_rate: float = _MATCH_FIT_SR) -> dict:
    """Bell gains whose SUMMED response follows the offline FIR, rather than the raw band
    gains — which the bells overshoot because they overlap.

    Least squares over a log-spaced grid, not a solve at the eight band centres: pinning the
    centres exactly still left 4.6 dB between the centres (and made the widest correction
    WORSE), because eight 0.7-Q bells cannot follow a frequency-sampled FIR point for point.
    Fitting the whole curve measured 5.5 -> 1.1 dB on an ordinary correction, 8.4 -> 2.7 dB on
    the widest legal one, and 2.7 -> 0.4 dB on a single band.

    Deterministic (fixed grid, fixed rate, no randomness), and cheap: an 8-column lstsq over
    400 points, once per chain resolve. Falls back to the raw gains if anything goes wrong —
    an approximation that is merely imperfect beats a chain that fails to resolve.
    """
    # OUTSIDE the try, deliberately: the except below falls back to the raw gains, so a bad rate
    # would silently un-fit every match_eq in the session and only the export oracle would notice.
    # It did — `session.project_sample_rate` is a METHOD, and passing it uncalled degraded the
    # whole contract without a word (2026-09-08 pre-merge review). A wrong type raises here, at
    # the caller, instead.
    sr = float(sample_rate) if sample_rate else float(_MATCH_FIT_SR)
    if not (1000.0 <= sr <= 768000.0):
        raise ValueError(f"_fit_match_eq_gains: implausible sample_rate {sr!r}")
    try:
        import numpy as np

        from .dsp import match_eq_fir
        from .reference_monitor import biquad_coeffs

        # …and the grid stops below THIS rate's Nyquist, or the fit is chasing a response the
        # FIR it is approximating does not have up there.
        hi_hz = min(_MATCH_FIT_HI_HZ, sr * 0.45)
        grid = np.geomspace(_MATCH_FIT_LO_HZ, hi_hz, _MATCH_FIT_POINTS)
        omega = 2.0 * np.pi * grid / sr
        z = np.exp(-1j * omega)

        def bell_db(centre_hz: float) -> np.ndarray:
            b, a = biquad_coeffs("peak", sr, min(centre_hz, sr * 0.45), MATCH_EQ_Q, 1.0)
            h = (b[0] + b[1] * z + b[2] * z ** 2) / (a[0] + a[1] * z + a[2] * z ** 2)
            return 20.0 * np.log10(np.abs(h))

        labels = list(BAND_LABELS)
        basis = np.column_stack([bell_db(_BAND_CENTER_HZ[b]) for b in labels])
        taps = match_eq_fir(band_gains_db, sr)
        spec = np.fft.rfft(taps, n=8192)
        fir_freqs = np.fft.rfftfreq(8192, 1.0 / sr)
        target = np.interp(grid, fir_freqs,
                           20.0 * np.log10(np.abs(spec) + 1e-12))
        fitted, *_ = np.linalg.lstsq(basis, target, rcond=None)
        if not np.all(np.isfinite(fitted)):
            return dict(band_gains_db)
        return {b: float(g) for b, g in zip(labels, fitted)}
    except Exception:   # noqa: BLE001 — a resolvable chain matters more than an exact fit
        return dict(band_gains_db)


def _mb_band(bd: dict) -> dict:
    return {"threshold_db": float(bd["threshold_db"]),
            "ratio": float(bd["ratio"]),
            "makeup_db": float(bd.get("makeup_db", 0.0)),
            "up": bool(bd.get("up", False))}   # upward-compression mode (boost quiet)


def monitor_chain_params(chain: list[dict],
                         limiter_makeup_db: float = 0.0,
                         limiter_pre_lufs_i: Optional[float] = None,
                         sample_rate: Optional[float] = None) -> dict:
    """Resolve ``chain`` (the output of ``effective_master_chain``) into the
    native monitor contract::

        {"entries": [ {type, ...params}, ... ],   # in signal order, live subset
         "rides": [ {start_s, end_s, delta_db} ], # gain rides: one stage after the devices
         "print_only": ["transpose", ...],        # types deferred to the render
         "tune_cents": -15.0,                     # summed enabled tune (stems) …
         "transpose_semitones": 2.0}              # … / transpose (master): the VST
        # hears these only in the printed master; the DAW realizes them natively as
        # fractional clip pitch (dawsession::reconcileAgentClipPitch), so they are
        # audible live + in the bounce there. Always present (0.0 = none/removed —
        # the zero is what un-tunes clips when the action is undone).

    ``limiter_makeup_db`` is the makeup the offline limiter computed at render
    time (``dsp._master_limiter`` -> the render ``report``). The real-time
    limiter applies it as a static pre-gain, so it never has to predict LUFS on
    the audio thread; it then brickwalls at ``ceiling_dbtp``.

    ``limiter_pre_lufs_i`` is the master's measured *pre-limiter* integrated LUFS
    (also from the render report). Because the target only scales the makeup and
    not the pre-limiter signal, the makeup is ``target - pre_lufs`` — resolving it
    here lets the live monitor track the "Loud" knob immediately, without waiting
    for a re-render to recompute the makeup.
    """
    entries: list[dict] = []
    rides: list[dict] = []
    print_only: list[str] = []
    tune_cents = 0.0
    transpose_st = 0.0
    for e in chain:
        t = e["type"]
        en = bool(e.get("enabled", True))    # Phase 40: per-device bypass — the
        # native monitor skips a disabled device's DSP but still shows its tile.
        if t == "gain_ride":
            if len(rides) < MAX_NATIVE_RIDES:
                rides.append({"start_s": float(e["start_s"]), "end_s": float(e["end_s"]),
                              "delta_db": float(e["delta_db"])})
            else:
                print_only.append(t)
        elif t == "gain":
            entries.append({"type": "gain", "gain_db": float(e["gain_db"]),
                            "enabled": en})
        elif t == "eq":
            b = e["band"]
            entries.append({"type": "eq", "shape": b["shape"],
                            "freq_hz": float(b["freq_hz"]),
                            "gain_db": float(b["gain_db"]),
                            "q": float(b["q"]),
                            # Phase 46: always emit (default 6) so the native parse
                            # is unambiguous — meaningful only for shape hpf/lpf.
                            "slope_db_oct": int(b.get("slope_db_oct", 6) or 6),
                            "channel": e.get("channel", "stereo"),  # Phase 41 M/S, Phase 46 L/R
                            # 2026-07-12: the EQ device's output trim (summed across
                            # entries → one post-bank gain in the native monitor).
                            "output_db": float(e.get("output_db", 0.0) or 0.0),
                            # Phase 33 region edit. Dropping it made a ranged band play
                            # (and, in the DAW, BOUNCE) across the whole song while the
                            # offline render confined it to the range.
                            **_range(e),
                            "enabled": en})
        elif t == "comp":
            comp_e = {"type": "comp",
                      "threshold_db": float(e["threshold_db"]),
                      "ratio": float(e["ratio"]),
                      "attack_ms": float(e["attack_ms"]),
                      "release_ms": float(e["release_ms"]),
                      "makeup_db": float(e.get("makeup_db", 0.0)),
                      "mix": float(e.get("mix", 1.0)),
                      "channel": e.get("channel", "stereo"),  # Phase 41 M/S
                      **_range(e),
                      "enabled": en}
            # True M/S comp: independent mid/side param sets (the native engine runs both).
            for sc in ("mid", "side"):
                sub = e.get(sc)
                if isinstance(sub, dict):
                    comp_e[sc] = {k: float(sub[k]) for k in
                                  ("threshold_db", "ratio", "attack_ms",
                                   "release_ms", "makeup_db", "mix") if k in sub}
            entries.append(comp_e)
        elif t == "dynamic_eq":
            entries.append({"type": "dynamic_eq",
                            "freq_hz": float(e["freq_hz"]),
                            "q": float(e["q"]),
                            "threshold_db": float(e["threshold_db"]),
                            "ratio": float(e["ratio"]),
                            "attack_ms": float(e.get("attack_ms", 10.0)),
                            "release_ms": float(e.get("release_ms", 120.0)),
                            "gain_db": float(e["gain_db"]),
                            **_range(e), "enabled": en})
        elif t == "mb_comp":
            entries.append({"type": "mb_comp",
                            "low": _mb_band(e["low"]),
                            "mid": _mb_band(e["mid"]),
                            "high": _mb_band(e["high"]),
                            "attack_ms": float(e.get("attack_ms", 10.0)),
                            "release_ms": float(e.get("release_ms", 150.0)),
                            # user-movable crossovers (default dsp.MB_CROSSOVERS_HZ) — must
                            # round-trip to the native editor or a drag snaps back on the poll
                            "crossover_lo_hz": float(e.get("crossover_lo_hz", 150.0)),
                            "crossover_hi_hz": float(e.get("crossover_hi_hz", 2000.0)),
                            "enabled": en})
        elif t == "ott":
            # OTT-style 3-band up/down comp (dsp._ott). LR4 split + per-sample
            # envelope natively — perceptual tier like mb_comp; the offline FFT
            # split stays the deliverable.
            def _ott_band(bd: dict) -> dict:
                bd = bd or {}
                return {"up": float(bd.get("up", 1.0)),
                        "down": float(bd.get("down", 1.0)),
                        "gain_db": float(bd.get("gain_db", 0.0))}
            entries.append({"type": "ott",
                            "depth": float(e.get("depth", 0.5)),
                            "time": float(e.get("time", 1.0)),
                            "in_gain_db": float(e.get("in_gain_db", 0.0)),
                            "out_gain_db": float(e.get("out_gain_db", 0.0)),
                            "low": _ott_band(e.get("low")),
                            "mid": _ott_band(e.get("mid")),
                            "high": _ott_band(e.get("high")),
                            "crossover_lo_hz": float(e.get("crossover_lo_hz", 150.0)),
                            "crossover_hi_hz": float(e.get("crossover_hi_hz", 2000.0)),
                            "enabled": en})
        elif t == "overdrive":
            # Drive-pedal distortion (dsp._overdrive): band-pass pre-filter +
            # exponential soft-clip + one-pole tone tilt — the same biquad family
            # natively, so it sits in the near-bit-exact tier (like the de-ess split).
            entries.append({"type": "overdrive",
                            "drive_db": float(e.get("drive_db", 12.0)),
                            "filter_on": bool(e.get("filter_on", False)),
                            "freq_hz": float(e.get("freq_hz", 800.0)),
                            "q": float(e.get("q", 0.7)),
                            "tone": float(e.get("tone", 0.5)),
                            "output_db": float(e.get("output_db", 0.0)),
                            "mix": float(e.get("mix", 1.0)),
                            "oversample": int(e.get("oversample", 1) or 1),
                            "enabled": en})
        elif t == "chorus":
            # Modulation (dsp._chorus): a delay swept by an LFO whose phase comes from the
            # HOST TIMELINE, so the native monitor lands on the same point of the sweep as
            # the offline render rather than wherever its own accumulator had drifted to.
            entries.append({"type": "chorus",
                            "rate_hz": float(e.get("rate_hz", 0.8)),
                            "depth": float(e.get("depth", 0.25)),
                            "delay_ms": float(e.get("delay_ms", 20.0)),
                            "mix": float(e.get("mix", 0.5)),
                            "voices": int(e.get("voices", 1) or 1),
                            "stereo_deg": float(e.get("stereo_deg", 90.0)),
                            "enabled": en})
        elif t == "flanger":
            # Modulation (dsp._flanger): the chorus's swept read plus a return path, so the
            # native side runs a real per-sample recursion and this side runs it in blocks.
            # Same host-timeline phase, same Lagrange stencil — the feedback is the only
            # structural difference, and its SIGN is a parameter, not a flag.
            entries.append({"type": "flanger",
                            "rate_hz": float(e.get("rate_hz", 0.5)),
                            "depth": float(e.get("depth", 0.6)),
                            "delay_ms": float(e.get("delay_ms", 3.0)),
                            "feedback": float(e.get("feedback", 0.0)),
                            "mix": float(e.get("mix", 0.5)),
                            "stereo_deg": float(e.get("stereo_deg", 0.0)),
                            "enabled": en})
        elif t == "auto_filter":
            # Phase 63 (dsp._auto_filter): every parameter as the law reads it
            # (`auto_filter_values` - held, snapped, the note's length in quarters beside its
            # index) plus the trigger. The native chain sweeps the same filter per sample.
            entries.append({"type": "auto_filter", **auto_filter_values(e),
                            "key_stem_id": e.get("key_stem_id") or "", "enabled": en})
        elif t == "match_eq":
            # FITTED, not raw: the bells overlap, so their dB contributions add and the raw
            # gains overshoot the FIR the offline engine renders — and in the DAW this bank is
            # the deliverable. A band the request omits can still need a non-zero bell, to
            # cancel its neighbours' bleed, so the fit runs over all eight.
            gains = _fit_match_eq_gains(e.get("band_gains_db", {}), sample_rate or _MATCH_FIT_SR)
            bands = [{"freq_hz": _BAND_CENTER_HZ[b], "gain_db": float(g),
                      "q": MATCH_EQ_Q}
                     for b, g in gains.items()
                     if b in _BAND_CENTER_HZ and abs(float(g)) > 0.01]
            if bands:
                entries.append({"type": "match_eq", "bands": bands,
                                "enabled": en})
        elif t == "tilt":
            # One-knob tilt EQ (dsp._tilt) — two shelves at the pivot. Bit-exact:
            # the native engine builds the two shelf biquads from these params.
            entries.append({"type": "tilt",
                            "tilt_db": float(e["tilt_db"]),
                            "pivot_hz": float(e["pivot_hz"]),
                            "q": TILT_Q,
                            "output_db": float(e.get("output_db", 0.0) or 0.0),  # 2026-07-12 post trim
                            "enabled": en})
        elif t == "saturation":
            # Memoryless waveshaper (dsp._saturate). Single-band is bit-exact native (nulls
            # like gain/EQ); the Phase-45 multiband form (movable crossovers + per-band
            # drive/output/mix/curve + global mix) is a real-time approximation like mb_comp.
            def _ci(c):   # curve name or int -> int index (0 tanh/1 warm/2 tube/3 hard)
                if isinstance(c, str):
                    return {"tanh": 0, "warm": 1, "tube": 2, "hard": 3}.get(c, 0)
                return 0 if c is None else int(c)
            sat = {"type": "saturation",
                   "drive_db": float(e["drive_db"]),
                   "output_db": float(e.get("output_db", 0.0)),
                   "curve": _ci(e.get("curve", 0)),
                   "mix": float(e.get("mix", 1.0)),          # dry/wet (single-band + multiband)
                   # single-band only: the multiband path is not oversampled (2026-09-06)
                   "oversample": int(e.get("oversample", 1) or 1),
                   "enabled": en}
            if isinstance(e.get("low"), dict):     # multiband saturator (top-level low/mid/high)
                for b in ("low", "mid", "high"):
                    bd = e.get(b) or {}
                    # a partial band inherits the entry's flat drive — exactly what dsp.py
                    # does, and what its comment claimed this already did (2026-09-01 sweep)
                    sat[b] = {"drive_db": float(bd.get("drive_db", e.get("drive_db", 0.0))),
                              "output_db": float(bd.get("output_db", 0.0)),
                              "mix": float(bd.get("mix", 1.0)),
                              "curve": None if bd.get("curve") is None else _ci(bd.get("curve"))}
                sat["crossover_lo_hz"] = float(e.get("crossover_lo_hz", 150.0))
                sat["crossover_hi_hz"] = float(e.get("crossover_hi_hz", 2000.0))
                sat["mix"] = float(e.get("mix", 1.0))
            entries.append(sat)
        elif t == "clip":
            # Pre-limiter clipper (dsp._clip) — bit-exact native at oversample 1
            # (memoryless, like saturation); v2's oversampled path is the perceptual
            # tier (juce::dsp::Oversampling vs resample_poly). Master-only, sits
            # just before the limiter.
            entries.append({"type": "clip",
                            "threshold_db": float(e["threshold_db"]),
                            "hardness": float(e.get("hardness", 0.5)),
                            "output_db": float(e.get("output_db", 0.0)),
                            "oversample": int(e.get("oversample", 1) or 1),
                            "enabled": en})
        elif t == "width":
            # User channel-strip width (stem/bus M/S) — memoryless, bit-exact;
            # finally audible live (the user's own width move), not print-only.
            entries.append({"type": "width", "width": float(e["width"]),
                            "enabled": en})
        elif t == "utility":
            # Ableton-style Utility (dsp._utility / MonitorChain::applyUtility) — v2:
            # DC filter + gain trim + per-channel polarity + width/mono + bass mono +
            # balance, in that order. The per-channel flips are RESOLVED here
            # (invert_left/right default to the legacy both-channel invert) so the
            # native reads plain bools.
            legacy = bool(e.get("invert", False))
            fl = e.get("invert_left", None)
            fr = e.get("invert_right", None)
            entries.append({"type": "utility",
                            "gain_db": float(e.get("gain_db", 0.0)),
                            "width": float(e.get("width", 1.0)),
                            "mono": bool(e.get("mono", False)),
                            "invert_left": legacy if fl is None else bool(fl),
                            "invert_right": legacy if fr is None else bool(fr),
                            "bass_mono": bool(e.get("bass_mono", False)),
                            "bass_mono_hz": float(e.get("bass_mono_hz", 120.0)),
                            "dc_filter": bool(e.get("dc_filter", False)),
                            "balance": float(e.get("balance", 0.0)), "enabled": en})
        elif t == "deess":
            # Split-band de-esser (dsp._deess) — real-time approximation (1st-order
            # split + the comp envelope), like comp/dyn-EQ. Stems/buses only. v2
            # carries the freed dynamics + range floor + wide-band mode.
            entries.append({"type": "deess",
                            "threshold_db": float(e["threshold_db"]),
                            "freq_hz": float(e["freq_hz"]),
                            "ratio": float(e.get("ratio", 4.0)),
                            "attack_ms": float(e.get("attack_ms", 1.0)),
                            "release_ms": float(e.get("release_ms", 50.0)),
                            "range_db": float(e.get("range_db", -80.0)),
                            "wide": bool(e.get("wide", False)),
                            **_range(e), "enabled": en})
        elif t == "gate":
            # Downward gate/expander (dsp._gate) — real-time approximation
            # (per-sample downward envelope), like comp. Stems/buses only. v2
            # carries hold/range/hysteresis + the detector-only sidechain filters.
            entries.append({"type": "gate",
                            "threshold_db": float(e["threshold_db"]),
                            "ratio": float(e["ratio"]),
                            "attack_ms": float(e.get("attack_ms", 1.0)),
                            "release_ms": float(e.get("release_ms", 100.0)),
                            "hold_ms": float(e.get("hold_ms", 0.0)),
                            "range_db": float(e.get("range_db", -80.0)),
                            "hysteresis_db": float(e.get("hysteresis_db", 0.0)),
                            "sc_hpf_hz": float(e.get("sc_hpf_hz", 0.0)),
                            "sc_lpf_hz": float(e.get("sc_lpf_hz", 20000.0)),
                            **_range(e), "enabled": en})
        elif t == "transient":
            # Attack/sustain shaper (dsp._transient) — real-time approximation
            # (block-envelope follower). Stems/buses only. v2 carries the detector
            # speed scale + the post-shaper output trim.
            entries.append({"type": "transient",
                            "attack_db": float(e["attack_db"]),
                            "sustain_db": float(e["sustain_db"]),
                            "speed": float(e.get("speed", 1.0) or 1.0),
                            "output_db": float(e.get("output_db", 0.0)),
                            "enabled": en})
        elif t == "duck":
            # Sidechain duck (dsp._duck) — real-time approximation keyed by a sidechain
            # input: the native monitor reads the key live, so the contract carries the
            # amount/timing plus `key_stem_id`. In the VST that id is advisory (the user
            # routes the host sidechain); in the DAW it is authoritative — wireDuckSidechains
            # (daw/Source/App.cpp) routes that key TRACK so the agent's duck is audible
            # (docs/10-daw.md Tier-1).
            entries.append({"type": "duck",
                            "threshold_db": float(e["threshold_db"]),
                            "depth_db": float(e["depth_db"]),
                            "attack_ms": float(e.get("attack_ms", 5.0)),
                            "release_ms": float(e.get("release_ms", 150.0)),
                            "key_stem_id": e.get("key_stem_id"), "enabled": en})
        elif t == "band_width":
            # Per-band stereo width (dsp._band_width_scale) — the band's EDGES are what
            # both engines split at (one LR8 law since 2026-10-21, the band width audit:
            # bandwidthspec::edges reads lo_hz/hi_hz first); the label's centre + Q ride
            # along for older readers. Stems/buses only. Custom `lo_hz`/`hi_hz` on the
            # entry (the DAW's free band scrub) override the label.
            b = e.get("band")
            custom = e.get("lo_hz") and e.get("hi_hz")
            if custom or b in _BAND_CENTER_HZ:
                if custom:
                    lo, hi = float(e["lo_hz"]), float(e["hi_hz"])
                    fc = round((lo * hi) ** 0.5, 1)
                else:
                    i = BAND_LABELS.index(b)
                    lo, hi = BAND_EDGES_HZ[i], BAND_EDGES_HZ[i + 1]
                    fc = _BAND_CENTER_HZ[b]
                q = fc / max(hi - lo, 1.0)
                entries.append({"type": "band_width",
                                "freq_hz": fc, "q": round(q, 4),
                                # `band` + the edges: the native splits at the edges
                                # (bandwidthspec::edges) and the dock draws them.
                                "band": "custom" if custom else b,
                                "lo_hz": lo, "hi_hz": hi,
                                "width": float(e["width"]), "enabled": en})
        elif t == "imager":
            # 3-band stereo imager (dsp._imager) — the native monitor splits with
            # an LR4 crossover (like mb_comp) vs the engine's zero-phase FFT split;
            # per-band M/S width. Stems/buses/master.
            entries.append({"type": "imager",
                            "width_low": float(e.get("width_low", 1.0)),
                            "width_mid": float(e.get("width_mid", 1.0)),
                            "width_high": float(e.get("width_high", 1.0)),
                            "crossover_lo_hz": float(e.get("crossover_lo_hz", 150.0)),
                            "crossover_hi_hz": float(e.get("crossover_hi_hz", 2000.0)),
                            "enabled": en})
        elif t == "return_delay":
            # A SPACE device at its chain position (the native MonitorChain parses it into a
            # send slot + a LinStep::SendFx step). Same flat keys the send_return return_entries
            # carry, PLUS `mix` — the dry/wet that lets it live on a track and not only on a
            # return (2026-07-20). Absent/1.0 = wet only, the return behaviour.
            entries.append({"type": "return_delay",
                            "delay_ms": float(e["delay_ms"]),
                            "feedback": float(e["feedback"]),
                            "tone": float(e["tone"]),
                            "mix": float(e.get("mix", 1.0)), "enabled": en})
        elif t == "return_reverb":
            entries.append({"type": "return_reverb",
                            "decay_s": float(e["decay_s"]),
                            "pre_delay_ms": float(e["pre_delay_ms"]),
                            "tone": float(e["tone"]),
                            "mix": float(e.get("mix", 1.0)), "enabled": en})
        elif t == "return_convolution":
            entries.append({"type": "return_convolution",
                            "ir": str(e.get("ir", "hall")),
                            "decay_s": float(e["decay_s"]),
                            "pre_delay_ms": float(e["pre_delay_ms"]),
                            "tone": float(e["tone"]),
                            "mix": float(e.get("mix", 1.0)), "enabled": en})
        elif t == "send_return":
            # Synthetic (gateway-injected): a send + its return's wet chain. Resolve
            # the return entries to flat params the native delay-line module reads.
            ret = []
            for re in e["return"]:
                rt = re["type"]
                if rt == "gain":
                    ret.append({"type": "gain", "gain_db": float(re["gain_db"])})
                elif rt == "return_delay":
                    ret.append({"type": "return_delay",
                                "delay_ms": float(re["delay_ms"]),
                                "feedback": float(re["feedback"]),
                                "tone": float(re["tone"])})
                elif rt == "return_reverb":
                    ret.append({"type": "return_reverb",
                                "decay_s": float(re["decay_s"]),
                                "pre_delay_ms": float(re["pre_delay_ms"]),
                                "tone": float(re["tone"])})
                elif rt == "return_convolution":
                    ret.append({"type": "return_convolution",
                                "ir": str(re.get("ir", "hall")),
                                "decay_s": float(re["decay_s"]),
                                "pre_delay_ms": float(re["pre_delay_ms"]),
                                "tone": float(re["tone"])})
            if any(r["type"] in ("return_delay", "return_reverb", "return_convolution") for r in ret):
                entries.append({"type": "send_return",
                                "send_db": float(e["send_db"]),
                                "return_entries": ret, "enabled": en})
        elif t == "limiter":
            target = e.get("target_lufs_i")
            makeup = float(limiter_makeup_db)
            # The makeup only depends on the (target-independent) pre-limiter
            # loudness, so resolve it from the cached pre-limiter LUFS instead of
            # the last render's makeup — the live monitor then tracks the Loud knob
            # without waiting for a re-render.
            if target is not None and limiter_pre_lufs_i is not None:
                makeup = float(target) - float(limiter_pre_lufs_i)
            lim = {"type": "limiter",
                   "ceiling_dbtp": float(e["ceiling_dbtp"]),
                   "makeup_db": makeup,
                   # Advanced (native VST): recovery / lookahead ramp / stereo-link / style.
                   "release_ms": float(e.get("release_ms", 50.0)),
                   "lookahead_ms": float(e.get("lookahead_ms", 3.0)),
                   "channel_link": float(e.get("channel_link", 1.0)),
                   "style": int(e.get("style", 0)),
                   "enabled": en}
            # Carry the pre-limiter LUFS whenever we have it (even with no target yet)
            # so the native editor can resolve makeup = target - pre_lufs locally the
            # FIRST time the Loud knob is turned, not one render later.
            if limiter_pre_lufs_i is not None:
                lim["pre_limiter_lufs_i"] = float(limiter_pre_lufs_i)
            # Round-trip the target too, else the Loud knob snaps back on the next poll.
            if target is not None:
                lim["target_lufs_i"] = float(target)
            entries.append(lim)
        elif t == "tune":
            # Phase 31 whole-stem tune (cents): no real-time chain node — but the
            # DAW consumes the summed value below as clip pitch, so only ENABLED
            # entries count (a bypassed tune must un-tune the clips).
            if en:
                tune_cents += float(e.get("cents", 0.0))
            print_only.append(t)
        elif t == "transpose":
            # Phase 32 master key transpose (semitones) — same split as tune.
            if en:
                transpose_st += float(e.get("semitones", 0.0))
            print_only.append(t)
        elif t in INERT_TYPES:
            # An analyzer: no DSP in either engine, so neither live nor print - but it IS a
            # device the user placed, and the DAW applies a pushed chain wholesale (full audit
            # 2026-10-02, reader E2·R3). Left out of `entries`, a Spectrogram added in the dock
            # was removed by the next push - every reopen, send move and agent move - while the
            # brain still held it, so adding it again stacked a second, invisible one. The native
            # walk passes a type it has no DSP for straight through (`--spectrogram`).
            entries.append({"type": t, "enabled": en})
        else:
            print_only.append(t)
    return {"entries": entries, "rides": rides, "print_only": print_only,
            "tune_cents": round(tune_cents, 1),
            "transpose_semitones": round(transpose_st, 2)}
