"""Build Pedalboard chains from session state and render bounces.

Conventions live in .claude/skills/audio-dsp/SKILL.md — read it before editing.
Key invariants: chains are derived from state (never mutated live), fixed processor
order, deterministic output, float64 summing, NaN renders raise.
"""
from __future__ import annotations

import contextlib
import copy
import json
import math
import threading
from collections import OrderedDict
from typing import Optional

from pathlib import Path

import numpy as np
import soundfile as sf
from pedalboard import (
    Compressor,
    Gain,
    HighpassFilter,
    HighShelfFilter,
    LowpassFilter,
    LowShelfFilter,
    PeakFilter,
    Pedalboard,
    PitchShift,
)

from . import sat_characters as _satc   # the saturation characters (2026-10-26)
from . import od_characters as _odc     # the overdrive's characters (2026-10-26)
from . import stereoize_law             # the Imager's stereoize (2026-10-28)
from .tp_detector import TP_CENTRE, intersample_peak   # the limiter's detector: the native's table (reader E4·R7)
from .models import BAND_EDGES_HZ, BAND_LABELS, COMP_STYLES
from .reference_monitor import biquad_coeffs, slope_cascade_coeffs
from .reference_monitor import (auto_filter_modulation, auto_filter_outputs,   # Phase 63: one law
                                auto_filter_qs, auto_filter_values)
from .session import (Session, effective_bus_chain, effective_chain,
                      effective_return_chain)

# v1 de-esser: split-band compression with fixed dynamics (see audio-dsp skill).

# Gain rides (Phase 5) + ranged processors (Phase 15): fixed raised-cosine
# fades at the range edges.
RIDE_FADE_S = 0.05

# Match-EQ (Phase 15): linear-phase FIR length. 4097 taps gives ~12 Hz
# frequency resolution at 48 kHz — enough to honor sub-band corrections.
MATCH_EQ_TAPS = 4097

# Multiband compressor (Phase 15): default crossovers (audio-dsp skill); the user
# co-edit moves them per entry. MB_MAX_BOOST_DB caps a band's UPWARD-compression boost
# (per-band `up` mode lifts quiet passages) so silence can't blow up the noise floor.
MB_CROSSOVERS_HZ = (150.0, 2000.0)
MB_MAX_BOOST_DB = 12.0

# OTT's law (the split, detector, makeup, both sides and their caps) is `ott_law` (2026-10-28).

# Overdrive (2026-07-07): post-shaper tone-tilt pivot (one-pole LP blend).
OD_TONE_HZ = 3000.0
# Overdrive voicing (2026-10-24, James: "they sound awful"). The lows are driven OD_EMPHASIS_DB
# less than the rest (a low shelf cut before the clipper, the same shelf boosted after it - the
# pair cancels exactly where nothing clips), the curve is biased by OD_BIAS so it clips one side
# sooner (even harmonics), a two-pole roll-off at OD_FIZZ_HZ takes the fizz, and a one-pole at
# SHAPER_DC_HZ takes the DC the bias makes. MonitorChain.h holds the same five (kOd*), pinned.
OD_EMPHASIS_HZ = 300.0
OD_EMPHASIS_DB = 9.0
OD_BIAS = 0.2
OD_FIZZ_HZ = 6500.0
SHAPER_DC_HZ = 10.0
# What a new overdrive / saturation runs its shaper at (2026-10-24): at 1x a hard drive's
# harmonics folded back as aliasing nearly as loud as the note. An entry's own `oversample` wins.
OD_OVERSAMPLE_DEFAULT = 8
SAT_OVERSAMPLE_DEFAULT = 4

# Phase 7 envelope followers (transient shaper, sidechain duck) work on 5 ms
# RMS blocks with asymmetric one-pole smoothing — deterministic and fast.
ENV_BLOCK_S = 0.005
ENV_FLOOR_DB = -120.0
TRANSIENT_FAST_ATTACK_MS = 1.0
TRANSIENT_FAST_RELEASE_MS = 20.0
TRANSIENT_SLOW_ATTACK_MS = 30.0
TRANSIENT_SLOW_RELEASE_MS = 300.0
TRANSIENT_FULL_DB = 6.0  # fast env this far above slow = fully transient
TRANSIENT_HOLD_MS = 10.0  # the detector's peak hold (James, 2026-10-17): a held note does not ripple it
TRANSIENT_DEADBAND_DB = 1.0  # fast over slow by less than this is no transient

# Master limiter (Phase 6): a numpy windowed brickwall, NOT Pedalboard's
# Limiter — that one applies internal makeup gain, which breaks LUFS
# targeting. Fixed lookahead/smoothing; audio-dsp skill documents the design.
LIMITER_LOOKAHEAD_S = 0.005
LIMITER_SMOOTH_S = 0.005

# Render-work observability (Phase 10): incremented per actual render (cache
# hits don't count). Tests assert on these; nothing in the engine reads them.
RENDER_COUNTS = {"master": 0, "stem": 0, "bus": 0, "return": 0}


class _StemRenderCache:
    """Cross-render memo of rendered stems, keyed on everything that shapes a
    stem's output (file mtime, gain/pan/width, chain, rides, automation, duck
    keys' state, project rate). With it, re-rendering the mix after one stem
    changed re-renders *that* stem — the rest return their cached audio, so a
    knob commit's catch-up render costs one stem, not the whole session.

    Off by default (budget 0): engine-pure runs, tests and the benchmark gate
    keep exact render counts. The gateway enables it per process
    (configure_render_cache; MIX_AGENT_RENDER_CACHE_MB overrides the budget).

    Bit-identical by construction: hits return the exact float64 arrays a
    fresh render produced — callers never mutate rendered audio (the same
    contract render_stem's per-call memo relies on). Telemetry is cached
    beside the audio so gain-reduction stats survive hits. Bounded LRU by
    bytes; eviction only ever costs a re-render. Thread-safe: renders run on
    worker threads (frozen state copies), so get/put may race."""

    def __init__(self) -> None:
        self.budget_bytes = 0
        self._entries: OrderedDict[str, tuple[np.ndarray, int, list]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.budget_bytes > 0

    def configure(self, budget_mb: int) -> None:
        with self._lock:
            self.budget_bytes = max(0, budget_mb) * 1024 * 1024
            self._evict()

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0
        MASTER_MEMO.clear()   # one switch, one clear: a master kept past its stems is no use

    def get(self, key: str) -> Optional[tuple[np.ndarray, int, list]]:
        with self._lock:
            hit = self._entries.get(key)
            if hit is not None:
                self._entries.move_to_end(key)
            return hit

    def put(self, key: str, audio: np.ndarray, sample_rate: int,
            telemetry: list) -> None:
        if not self.enabled or audio.nbytes > self.budget_bytes:
            return
        with self._lock:
            if key not in self._entries:
                self._entries[key] = (audio, sample_rate, telemetry)
                self._bytes += audio.nbytes
                self._evict()

    def _evict(self) -> None:  # caller holds the lock
        while self._bytes > self.budget_bytes and self._entries:
            _, (audio, _, _) = self._entries.popitem(last=False)
            self._bytes -= audio.nbytes


class _MasterMemo:
    """The last few MASTER renders, kept whole (Phase 54, 2026-10-12).

    The stem cache above makes a re-render cost one stem; the master stage itself (the sum,
    the busses, the limiter) and whatever is then measured on it were still paid on every
    call - 3.6 s on a two-minute song with every stem cached. The DAW's brain asked for it
    before every chat message, after every reply, after every command that carried a
    co-edit, and once per analysis tool, whether or not anything had changed.

    Keyed on `Session.render_fingerprint()` (everything the render reads) and `pre_master`.
    A hit returns the same float64 array a fresh render produced, marked READ-ONLY: a caller
    that edits rendered audio in place fails loudly instead of handing the next caller its
    edit. The report and both telemetries ride with it, and so does anything computed FROM
    the render (`kept_with_master`: the top bar's readout, the full master analysis).

    On with the stem cache (`configure_render_cache`), off with it: engine-pure runs, tests
    and the benchmark keep exact render counts."""

    ENTRIES = 3

    def __init__(self) -> None:
        self._entries: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return RENDER_CACHE.enabled

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            hit = self._entries.get(key)
            if hit is not None:
                self._entries.move_to_end(key)
            return hit

    def put(self, key: str, entry: dict) -> Optional[dict]:
        """Keep `entry` and hand back the kept one - or None when one master render is more
        than the whole budget, which then means what it says: nothing that size is held."""
        if entry["audio"].nbytes > RENDER_CACHE.budget_bytes:
            return None
        with self._lock:
            kept = self._entries.setdefault(key, entry)
            self._entries.move_to_end(key)
            while len(self._entries) > self.ENTRIES:
                self._entries.popitem(last=False)
            return kept


MASTER_MEMO = _MasterMemo()
RENDER_CACHE = _StemRenderCache()


def configure_render_cache(budget_mb: int) -> None:
    """Enable (or disable, 0) the persistent stem-render cache, and the master memo with it.
    The gateway calls this at project open and the DAW's brain when it starts; standalone
    engine use leaves it off."""
    RENDER_CACHE.configure(budget_mb)
    if not RENDER_CACHE.enabled:
        MASTER_MEMO.clear()


def _master_memo_key(session: Session, pre_master: bool) -> Optional[str]:
    """What a kept master render is filed under, or None when it cannot be said (the memo is
    off, or `session` is a stand-in with no fingerprint)."""
    if not MASTER_MEMO.enabled:
        return None
    fingerprint = getattr(session, "render_fingerprint", None)
    if fingerprint is None:
        return None
    try:
        return f"{fingerprint()}|{int(bool(pre_master))}"
    except Exception:  # noqa: BLE001 - a session that cannot be fingerprinted is just rendered
        return None


def kept_with_master(session: Session, name: str, compute, pre_master: bool = False):
    """A value computed from the master render - the top bar's readout, the whole master
    analysis - kept beside the render it was computed from, so asking twice costs once.
    Call it right after `render_master`; with the memo off, or no kept render under this
    state, it is `compute()`. What comes back is the caller's own copy."""
    if has_native_mix(session):
        return compute()      # of the app's mix, which nothing keeps (`render_master`)
    key = _master_memo_key(session, pre_master)
    entry = MASTER_MEMO.get(key) if key is not None else None
    if entry is None:
        return compute()
    derived = entry["derived"]
    if name not in derived:
        derived[name] = compute()
    return copy.deepcopy(derived[name])


def _stem_cache_key(session: Session, stem_id: str,
                    exclude_ducks: bool) -> str:
    """Everything render_stem's output depends on. Duck-key stems' state is
    folded in (one level — key renders are duck-free by construction); their
    full chain over-keys slightly, which costs misses, never wrong audio."""
    def fingerprint(sid: str) -> dict:
        stem = session.stems[sid]
        path = session.project_dir / stem["path"]
        return {"path": stem["path"], "mtime_ns": path.stat().st_mtime_ns,
                "gain_db": stem["gain_db"], "pan": stem["pan"],
                "width": stem["width"], "width_on": stem.get("width_on", True), "chain": stem["chain"],
                "stereoize": stem.get("stereoize", False), "stereoize_ms": stem.get("stereoize_ms", 6.0),
                "rides": stem["rides"],
                "capture_start_s": stem.get("capture_start_s", 0.0),   # duck alignment (2026-09-01)
                "vca_db": stem.get("vca_db", 0.0),       # the engine's facts (D2·R4)
                "delay_ms": stem.get("delay_ms", 0.0),
                "clip": stem.get("clip"),                # where the file plays (D4·R8)
                "clips": stem.get("clips"),              # ...as several clips (2026-10-11)
                "automation": session.automation_active(sid)}

    duck_keys = sorted({entry["key_stem_id"]
                        for entry in session.stems[stem_id]["chain"]
                        if entry["type"] in ("duck", "auto_filter") and not exclude_ducks
                        and _duck_key_heard(session, entry.get("key_stem_id"))})
    # A synced Auto Filter here or on a trigger reads the song's tempo map (Phase 63), and so does
    # every Duck (2026-10-28: its curve runs on the song's quarters, synced or triggered).
    synced = any((e.get("type") == "auto_filter" and e.get("lfo_sync"))
                 or (e.get("type") == "duck" and sid == stem_id)
                 for sid in [stem_id, *duck_keys] for e in session.stems[sid]["chain"])
    return json.dumps({
        "project": str(session.project_dir),
        "stem_id": stem_id,
        "exclude_ducks": exclude_ducks,
        "rate": session.project_sample_rate(),
        "stem": fingerprint(stem_id),
        "ducks": {key: fingerprint(key) for key in duck_keys},
        "tempo": session.tempo_map if synced else None,
    }, sort_keys=True)

# Gain-reduction telemetry (Phase 20, mix-analysis skill v6): dry-vs-wet on
# the 5 ms block envelopes, over blocks where the dry input is active.
GR_ACTIVE_GATE_DBFS = -60.0
GR_COUNT_MIN_DB = 0.5


def _comp_makeup_credit_db(entry: dict) -> float:
    """What a comp's makeup adds to its OUTPUT level when it reduces nothing: the makeup through
    the mix (`mix * 10^(mk/20) + 1 - mix`), of the half the comp is set to on a Mid / Side comp.
    The full makeup was credited, so a parallel comp at mix 0.5 + 12 dB that never compressed read
    4 dB of reduction 100 % of the time and `over_compression` fired (audit 2026-10-17)."""
    ch = entry.get("channel", "stereo")
    half = entry.get(ch) if ch in ("mid", "side") else None
    p = {**entry, **half} if isinstance(half, dict) else entry
    mk = float(p.get("makeup_db", 0.0) or 0.0)
    mix = min(1.0, max(0.0, float(p.get("mix", 1.0) if p.get("mix") is not None else 1.0)))
    return float(20.0 * np.log10(max(1e-9, mix * 10.0 ** (mk / 20.0) + 1.0 - mix)))


def _gr_telemetry(processor: str, dry: np.ndarray, wet: np.ndarray,
                  sample_rate: int, makeup_db: float = 0.0) -> Optional[dict]:
    """Gain-reduction stats for one dynamics processor (Phase 20): block
    envelopes of dry vs wet (frames, channels) audio; `makeup_db` is added
    back so telemetry reports *reduction*, not net level change. None when
    the dry signal never rises above the gate."""
    dry_db = _block_env_db(dry, sample_rate)
    wet_db = _block_env_db(wet, sample_rate)
    n = min(dry_db.size, wet_db.size)
    active = dry_db[:n] > GR_ACTIVE_GATE_DBFS
    if not active.any():
        return None
    reduction = np.maximum(dry_db[:n][active] - wet_db[:n][active] + makeup_db,
                           0.0)
    return {"processor": processor,
            "gr_mean_db": round(float(reduction.mean()), 1),
            "gr_max_db": round(float(reduction.max()), 1),
            "gr_active_pct": round(
                100.0 * float((reduction > GR_COUNT_MIN_DB).mean()), 1)}


def _rbj_band(audio: np.ndarray, sample_rate: int, band: dict) -> np.ndarray:
    """One EQ band's filter - peak, shelves, the 1st-order hpf/lpf - on (channels, frames) audio,
    in float64 with the native's own coefficients (`reference_monitor.biquad_coeffs`, which
    clamps freq and q as `makeBiquad` does). Since 2026-10-17 (the EQ audit) the native bank runs
    in double too, so the two engines run one filter: through Pedalboard (JUCE's float maths) a
    30 Hz shelf at 96 kHz was 6 dB off its design while the DAW's was 2 dB off the other way."""
    from scipy.signal import lfilter
    b, a = biquad_coeffs(band["shape"], sample_rate, float(band["freq_hz"]),
                         float(band.get("q", 0.707)), float(band.get("gain_db", 0.0)))
    out = lfilter(b, a, np.asarray(audio, dtype=np.float64), axis=-1)
    return out.astype(np.asarray(audio).dtype, copy=False)


def _eq_band(audio: np.ndarray, sample_rate: int, band: dict) -> np.ndarray:
    """Apply one EQ band to (channels, frames) audio: the EQ's law (`eq_law`, 2026-10-27 - a
    reference EQ's shapes and Q, measured, as magnitude-matched biquads in float64; the native
    bank runs the same sections, `native/Source/EqLaw.h`). Every shape, every slope."""
    from scipy.signal import lfilter
    from .eq_law import band_sections
    out = np.asarray(audio, dtype=np.float64)
    for b, a in band_sections(band["shape"], sample_rate, float(band["freq_hz"]),
                              float(band.get("q", 1.0)), float(band.get("gain_db", 0.0)),
                              int(band.get("slope_db_oct", 6) or 6)):
        out = lfilter(b, a, out, axis=-1)
    return out.astype(np.asarray(audio).dtype, copy=False)


def gate_gain_db(peak: np.ndarray, sample_rate: float, threshold_db: float, ratio: float,
                 attack_ms: float, release_ms: float, hold_ms: float = 0.0,
                 range_db: float = -80.0, hysteresis_db: float = 0.0,
                 character: int = 0) -> np.ndarray:
    """THE GATE'S LAW (v4, 2026-10-27: four characters fitted to a reference gate) - the gain in dB
    for a detector `peak` (the linked |x| after the user's sidechain filters and the character's
    own detector EQ, `gate_characters.detector_eq`). `gate_characters.gain_db` is the law and says
    what each stage does; `MonitorChain::applyGate` runs it per sample."""
    from . import gate_characters
    return gate_characters.gain_db(peak, sample_rate, threshold_db, ratio, attack_ms, release_ms,
                                   hold_ms, range_db, hysteresis_db, character)


def _gate(audio: np.ndarray, sample_rate: int, entry: dict) -> np.ndarray:
    """Downward gate/expander (Phase 41; v2 adds hold/range/hysteresis + detector
    filters) on (channels, frames) float32. Detector: optional sidechain HP/LP
    (Butterworth 2nd-order, DETECTOR ONLY — the audio path is untouched), then a
    peak follower on the linked per-channel |x| (since 2026-10-17: the DAW's law). Gain law: below the effective
    threshold the gain drops by (ratio−1) dB per dB under, clamped at `range_db`;
    while open the CLOSE threshold sits `hysteresis_db` lower (no flutter);
    `hold_ms` freezes the envelope before the release starts; asymmetric one-pole
    attack/release, per sample. Deterministic — the one law (`gate_gain_db`) that
    MonitorChain::applyGate runs."""
    ratio = float(entry["ratio"])
    if ratio <= 1.0:
        return audio
    x = audio.astype(np.float64)
    det = x
    sc_hpf_hz = float(entry.get("sc_hpf_hz", 0.0))
    sc_lpf_hz = float(entry.get("sc_lpf_hz", 20000.0))
    if sc_hpf_hz >= 20.0 or sc_lpf_hz < 20000.0:
        # The same 2nd-order biquads the native monitor builds — q=0 is the
        # fixed-Butterworth-Q sentinel in both biquad makers (detector parity).
        from scipy.signal import lfilter
        nyq = sample_rate / 2.0
        if sc_hpf_hz >= 20.0:
            b, a = biquad_coeffs("hpf2", sample_rate, min(sc_hpf_hz, 0.9 * nyq),
                                 0.0, 0.0)
            det = lfilter(b, a, det, axis=1)
        if sc_lpf_hz < 20000.0:
            b, a = biquad_coeffs("lpf2", sample_rate, min(sc_lpf_hz, 0.9 * nyq),
                                 0.0, 0.0)
            det = lfilter(b, a, det, axis=1)
    # THE DAW'S LAW (gate_gain_db = MonitorChain::applyGate). The character's own detector EQ
    # comes after the user's filters, as natively.
    from . import gate_characters
    character = gate_characters.character_index(entry.get("character", gate_characters.DEFAULT))
    det = gate_characters.detector_eq(det, sample_rate, character, axis=1)
    gain_db = gate_gain_db(np.max(np.abs(det), axis=0), sample_rate,
                           float(entry["threshold_db"]), ratio,
                           float(entry["attack_ms"]), float(entry["release_ms"]),
                           float(entry.get("hold_ms", 0.0)), float(entry.get("range_db", -80.0)),
                           float(entry.get("hysteresis_db", 0.0)), character)
    gain = 10.0 ** (gain_db / 20.0)
    return (x * gain[np.newaxis, :]).astype(np.float32)


def _lr4_split(audio: np.ndarray, sample_rate: int, freq_hz: float) -> tuple:
    """Linkwitz-Riley 4th-order split of (channels, frames) audio -> (low, high).

    Two cascaded Butterworth 2nd-order sections per leg (`biquad_coeffs` with q == 0,
    the fixed LR building block the mb-comp / OTT / imager crossovers already use), so
    low + high is magnitude-FLAT (an allpass) rather than the 1st-order pair's exact
    sum. That trade buys the isolation a de-esser needs: at a 6 kHz split an 8 kHz ess
    leaks -13.5 dB through the low leg instead of -4.7 dB, and the DETECTOR's high band
    reads 1 kHz vocal body at -64 dB instead of -16 dB.
    """
    from scipy.signal import lfilter
    fc = min(max(float(freq_hz), 2.0), sample_rate * 0.49)
    bl, al = biquad_coeffs("lpf2", sample_rate, fc, 0.0, 0.0)
    bh, ah = biquad_coeffs("hpf2", sample_rate, fc, 0.0, 0.0)
    x = np.asarray(audio, dtype=np.float64)
    low = lfilter(bl, al, lfilter(bl, al, x, axis=1), axis=1)
    high = lfilter(bh, ah, lfilter(bh, ah, x, axis=1), axis=1)
    return low.astype(audio.dtype, copy=False), high.astype(audio.dtype, copy=False)


def _deess(audio: np.ndarray, sample_rate: int, entry: dict) -> np.ndarray:
    """The de-esser on (channels, frames) float32: `deess_law.apply` - the DAW's law
    (`MonitorChain::applyDeess`), fitted to a reference de-esser (2026-10-27; the module says how).
    No ratio, attack or release: a RANGE reached at 0 dBFS on the detector, a band FREQ..TOP, a
    MODE (allround / vocal), split (a linear-phase shelf) or WIDE, a 12 ms lookahead - aligned
    here, the latency the native chain reports."""
    from . import deess_law
    return deess_law.apply(audio, sample_rate, entry)


def _pan(audio: np.ndarray, pan: float) -> np.ndarray:
    """Balance-law pan on (frames, channels) float64: unity at center, the
    far channel attenuates linearly. Mono upmixes to stereo first."""
    if pan == 0.0:
        return audio
    if audio.shape[1] == 1:
        audio = np.tile(audio, (1, 2))
    out = audio.copy()
    out[:, 0] *= min(1.0, 1.0 - pan)
    out[:, 1] *= min(1.0, 1.0 + pan)
    return out


ENV_BLOCK_MIN_S = 0.00025   # 0.25 ms floor (12 samples @ 48 k) — bounds cost, keeps RMS meaningful


def _env_block_s(*time_ms: float) -> float:
    """The envelope block for a device whose FASTEST time constant is `time_ms`.

    A time constant shorter than the block cannot be represented at block rate:
    `_smooth_env_db` clamped it to the block (`max(attack_ms, block_ms)`), so every
    `attack_ms` in [0.1, 5.0] rendered BIT-IDENTICALLY offline — the whole fast-gate
    region, the de-esser's 1 ms default, and the transient shaper's `speed` — while the
    native monitor, which is per-sample, honoured all of them. The knob moved what you
    heard live and nothing in the deliverable.

    Halving the block for a faster constant lets the offline express it (and converges it
    toward the native's per-sample detector). A constant at or above ENV_BLOCK_S keeps the
    5 ms block, so every render whose times are 5 ms or slower is bit-identical to before.
    """
    fastest = min((float(t) for t in time_ms if t and float(t) > 0.0),
                  default=ENV_BLOCK_S * 1000.0)
    if fastest >= ENV_BLOCK_S * 1000.0:
        return ENV_BLOCK_S
    return max(ENV_BLOCK_MIN_S, fastest / 1000.0 / 2.0)


def _block_env_db(audio: np.ndarray, sample_rate: int,
                  block_s: float = ENV_BLOCK_S) -> np.ndarray:
    """RMS per non-overlapping block (5 ms unless a faster time constant needs
    finer resolution — see `_env_block_s`), in dBFS: each CHANNEL's, linked by the louder.

    It was the RMS of the channel mean (reader E4·R8, extending 02·F3): content that is out
    of phase between the channels cancelled before it was measured. A dynamic EQ never cut a
    wide synth's anti-phase band, a duck never followed such a key, and OTT - which levels
    toward a target - read silence and BOOSTED it by its full range; the DAW's devices follow
    the linked per-channel peak and acted at full depth, so what the agent measured after such
    a move was not the deliverable. On two equal channels (every mono file since E4·R4) the
    linked reading is the old one exactly; on one channel alone it is 6 dB up, where the
    native reads it."""
    block = max(1, int(block_s * sample_rate))
    n_blocks = -(-audio.shape[0] // block)  # ceil: the tail counts too
    mean_square = np.zeros(n_blocks)
    padded = np.zeros(n_blocks * block)
    for channel in range(audio.shape[1]):   # one at a time, summed as the mono reading was:
        padded[: audio.shape[0]] = audio[:, channel]   # equal channels give its very bits
        np.maximum(mean_square, np.mean(padded.reshape(n_blocks, block) ** 2, axis=1),
                   out=mean_square)
    rms = np.sqrt(mean_square)
    return 20.0 * np.log10(np.maximum(rms, 10.0 ** (ENV_FLOOR_DB / 20.0)))


def _smooth_env_db(env_db: np.ndarray, sample_rate: int,
                   attack_ms: float, release_ms: float,
                   block_s: float = ENV_BLOCK_S) -> np.ndarray:
    """Asymmetric one-pole over the block series: fast toward rising levels
    (attack), slow toward falling (release). Plain loop over the block series —
    deterministic, and cheap at this block rate. `block_s` must be the block the
    caller measured the envelope at, else the time constants are mis-scaled."""
    if len(env_db) == 0:
        return np.empty_like(env_db)
    block_ms = block_s * 1000.0
    up = float(np.exp(-block_ms / max(attack_ms, block_ms)))
    down = float(np.exp(-block_ms / max(release_ms, block_ms)))
    out = np.empty_like(env_db)
    level = env_db[0]
    for i, value in enumerate(env_db):
        coef = up if value > level else down
        level = coef * level + (1.0 - coef) * value
        out[i] = level
    return out


_BALLISTICS_OFFSET = 1.0      # keeps the follower's input above its threshold, and nonzero
_BALLISTICS_THRESHOLD = 0.5   # below the offset: the gain law is always in its invertible branch


def _gr_ballistics_reference(target_db: np.ndarray, sample_rate: float,
                             attack_ms: float, release_ms: float) -> np.ndarray:
    """The native compressor's ballistics, one sample at a time (float64): a
    one-pole on the gain reduction in dB that moves toward MORE reduction with
    the attack coefficient and back with the release one, each a literal time
    constant (`exp(-1 / (sr * ms / 1000))`), starting from no reduction. A
    transcription of `MonitorChain::processOneComp` / `mbBandGain` and of
    `reference_monitor._compress`. About 1.3 µs a sample — the reference
    `_gr_ballistics_db` is pinned against, not something a render calls."""
    a_att = math.exp(-1000.0 / (sample_rate * attack_ms)) if attack_ms > 0.0 else 0.0
    a_rel = math.exp(-1000.0 / (sample_rate * release_ms)) if release_ms > 0.0 else 0.0
    out = np.empty(len(target_db), dtype=np.float64)
    env = 0.0
    for i, target in enumerate(np.asarray(target_db, dtype=np.float64)):
        coef = a_att if target < env else a_rel
        env = coef * env + (1.0 - coef) * target
        out[i] = env
    return out


def _gr_ballistics_db(target_db: np.ndarray, sample_rate: float,
                      attack_ms: float, release_ms: float) -> np.ndarray:
    """`_gr_ballistics_reference` at compiled speed (four minutes in ~0.1 s).

    The recursion branches on its own state, so numpy cannot vectorise it and a
    Python loop costs ~15 s per compressor on a four-minute stem. The engine has
    no compiled code of its own — but Pedalboard's `Compressor` (JUCE
    `dsp::Compressor`) runs exactly this recursion inside it: its level follower
    is `y = x + cte * (y - x)` with `cte = attack if x > y else release`, and
    `cte = exp(-2π·1000 / (sr · ms))`. So the reduction (as a positive number,
    plus an offset) is fed THROUGH that follower with its times scaled by 2π,
    and the follower's state is read back out of the output by inverting the
    gain law (`out = in · (y / thr)^(1/ratio − 1)`, ratio 2 → `y = thr · (in /
    out)²`). A branching one-pole is shift-equivariant, so the offset comes off
    again; the lead-in settles the follower on the offset, which is the
    recursion's "starts from no reduction".

    Float32 inside, like the native: the two agree to ~4e-3 dB on the slowest
    release, which is the size of the native's own float rounding. Pinned by
    `test_comp_native_law.py` against the loop above, so a Pedalboard that
    changes its compressor turns a test red instead of moving every render."""
    target = np.asarray(target_db, dtype=np.float64)
    n = target.shape[0]
    if n == 0 or not np.any(target < 0.0):
        return np.zeros(n, dtype=np.float64)            # never above threshold
    a_att = math.exp(-1000.0 / (sample_rate * attack_ms)) if attack_ms > 0.0 else 0.0
    lead_in = int(math.ceil(18.0 / max(1.0 - a_att, 1.0e-6))) + 16
    fed = np.full(lead_in + n, _BALLISTICS_OFFSET, dtype=np.float32)
    fed[lead_in:] = (_BALLISTICS_OFFSET - target).astype(np.float32)
    follower = Compressor(
        threshold_db=20.0 * math.log10(_BALLISTICS_THRESHOLD), ratio=2.0,
        attack_ms=2.0 * math.pi * max(float(attack_ms), 0.0),
        release_ms=2.0 * math.pi * max(float(release_ms), 0.0))
    out = Pedalboard([follower])(fed[np.newaxis, :], float(sample_rate))[0]
    ratio = fed[lead_in:].astype(np.float64) / out[lead_in:].astype(np.float64)
    return _BALLISTICS_OFFSET - _BALLISTICS_THRESHOLD * ratio * ratio


def _native_comp(audio: np.ndarray, sample_rate: int, threshold_db: float,
                 ratio: float, attack_ms: float, release_ms: float,
                 makeup_db: float = 0.0) -> np.ndarray:
    """The multiband compressor's band law (`MonitorChain::mbBandGain`; the `comp` device ran it
    too until 2026-10-25 and now runs `_character_comp`), on
    (channels, frames): a feed-forward PEAK detector LINKED across the channels
    (one gain for all of them — the stereo image holds still), hard knee, and
    the ballistics on the gain reduction in dB. Returns the wet signal, makeup
    included, in the input's dtype.

    This replaced Pedalboard's compressor here (2026-10-03, reader E4·R1), whose
    `attack_ms` is a time constant of ms/2π on the LEVEL, per channel: a "10 ms /
    100 ms" comp clamped in ~1 ms and let go in ~22 offline while the monitor and
    the bounce took 10 and 100, so on drums the deliverable sat up to 4.7 dB
    above what the agent had measured — and a hard-panned hit pulled only its
    own channel down. A signal that never crosses the threshold (and a ratio of
    1) comes back untouched, bit for bit."""
    slope = 1.0 - 1.0 / max(float(ratio), 1.0)
    makeup = 10.0 ** (float(makeup_db) / 20.0)
    detector = np.max(np.abs(audio), axis=0).astype(np.float64)
    over = 20.0 * np.log10(np.maximum(detector, 1.0e-12)) - float(threshold_db)
    if slope <= 0.0 or not np.any(over > 0.0):
        return audio if makeup == 1.0 else (audio * audio.dtype.type(makeup))
    target = np.where(over > 0.0, -over * slope, 0.0)
    env = _gr_ballistics_db(target, sample_rate, attack_ms, release_ms)
    gain = (10.0 ** (env / 20.0)) * makeup
    return (audio * gain[np.newaxis, :]).astype(audio.dtype)


# The compressor's CHARACTERS (2026-10-25, the black-box pass against a reference compressor; James
# picked CLEAN, then PUNCH, SMOOTH, GLUE, VOCAL and MASTERING). Each is a row of one law, fitted to
# the reference's measured gain on step tests and a dry vocal (docs/10-daw.md has the record). Every
# stage is a fixed-coefficient branching one-pole, so `_gr_ballistics_db` runs each at compiled
# speed - a law whose coefficient depends on its own state cannot be run fast here (that is why
# CLEAN's first fit, whose attack slowed on small gaps, was refitted into this form):
#   fast stage:  two followers of the static target, both releasing over release / RELEASE_DIV -
#                one attacking over attack x ATTACK_K, one ATTACK_SLOW_X times slower (or faster);
#                the reduction is the slower one, held to within ATTACK_GAP_DB of the quicker. With a
#                slow follower and a gap of about a dB a jump is taken fast down to its last dB and
#                that last dB slowly, so the ripple of a held note is not chased.
#   slow stage:  the BODY follows the fast stage, charging over BODY_ATTACK_MS and letting go over
#                BODY_RELEASE_MS + BODY_RELEASE_K x release.
#   gain:        the deeper of the two - a short hit lets go at the fast stage's speed, a long
#                squash at the body's.
# A stage runs on the reduction in dB, or (LIN_*) on the linear gain minus one - a release on the
# linear gain is faster in dB the deeper it starts, as a reference's MASTERING style measured.
# The static target is the textbook law, or (STATIC) one of comp_tables' measured laws.
# MonitorChain::processOneComp (kCompCharacters in MonitorChain.h) runs the same law per sample and
# reference_monitor._compress is its float64 loop; test_comp_characters.py holds the tables equal
# and server/tests/data/comp_clean_parity.json the samples.
COMP_DEFAULT_STYLE = 0   # names: models.COMP_STYLES
#: index -> (attack_k, attack_slow_x, attack_gap_db, release_div, body_attack_ms, body_release_ms,
#:           body_release_k, lin_fast, lin_body, static, hold_ms) - static: None textbook (with
#:           the quadratic knee), "clean_knee" textbook with CLEAN's measured knee, else a
#:           comp_tables name; hold_ms: a PEAK-HOLD detector in front (instant up, a straight line
#:           down in dB over hold_ms; 0 = the raw peak) - MASTERING's: the reference reads a sine
#:           at its peak with no ripple at all, which only a held detector does
COMP_CHARACTERS = {
    0: (0.333, 6.73, 0.956, 2.36, 2172.0, 42.6, 0.211, 0, 0, "clean_knee", 0.0),   # CLEAN
    1: (0.860, 0.464, 2.44, 2.11, 0.01, 7.92, 0.0, 0, 1, None, 0.0),           # PUNCH
    2: (0.294, 50.2, 0.0, 6.24, 117.0, 0.01, 0.686, 1, 1, "smooth", 0.0),      # SMOOTH
    3: (0.192, 5.36, 0.855, 4.27, 21.6, 0.597, 0.0229, 0, 1, "glue", 0.0),     # GLUE
    4: (5.17, 0.0961, 0.0, 2.66, 136.0, 101.0, 0.0, 1, 0, "vocal", 0.0),       # VOCAL
    5: (0.405, 1.735, 1.069, 2.349, 27.0, 410.7, 0.0, 1, 1, None, 27.1),       # MASTERING
}


def comp_style(entry: dict) -> int:
    """A stored comp entry's character index (absent = CLEAN), held to the table."""
    try:
        st = int(entry.get("style", COMP_DEFAULT_STYLE) or 0)
    except (TypeError, ValueError):
        st = COMP_DEFAULT_STYLE
    return min(len(COMP_STYLES) - 1, max(0, st))


def _lin_db(x: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(1.0 + x, 1.0e-9))


def comp_character_gain_db(target_db: np.ndarray, sample_rate: float, attack_ms: float,
                           release_ms: float, style: int = 0) -> np.ndarray:
    """A character's reduction (dB, <= 0) for a static target (dB, <= 0), from rest."""
    ka, kq, gap, kf, sa, s0, s1, lin_fast, lin_body, _, _ = COMP_CHARACTERS[style]
    target = np.asarray(target_db, dtype=np.float64)
    fast_release = float(release_ms) / kf
    attack = float(attack_ms) * ka
    fed = (10.0 ** (target / 20.0) - 1.0) if lin_fast else target
    held = _gr_ballistics_db(fed, sample_rate, attack * kq, fast_release)
    quick = _gr_ballistics_db(fed, sample_rate, attack, fast_release)
    if lin_fast:
        held, quick = _lin_db(held), _lin_db(quick)
    fast = np.minimum(held, quick + gap)
    fed = (10.0 ** (fast / 20.0) - 1.0) if lin_body else fast
    body = _gr_ballistics_db(fed, sample_rate, sa, s0 + s1 * float(release_ms))
    if lin_body:
        body = _lin_db(body)
    return np.minimum(fast, body)


def comp_detector_db(peak: np.ndarray, sample_rate: float, style: int = 0) -> np.ndarray:
    """The level a character's static law reads (dB): the linked peak, or - with a hold - the
    recursion `det = max(|x|, det * a)` from rest, in closed form as the gate's detector is:
    det_db[i] = i*L + cummax(lx - i*L)."""
    lx = 20.0 * np.log10(np.maximum(np.asarray(peak, dtype=np.float64), 1.0e-12))
    hold_ms = COMP_CHARACTERS[style][10]
    if hold_ms <= 0.0 or lx.shape[0] == 0:
        return lx
    step_db = 20.0 * math.log10(math.exp(-1.0 / (sample_rate * hold_ms / 1000.0)))
    j = np.arange(lx.shape[0], dtype=np.float64)
    return j * step_db + np.maximum.accumulate(lx - j * step_db)


def comp_static_db(over_db: np.ndarray, threshold_db: float, ratio: float,
                   style: int = 0, knee_db: float = 0.0) -> np.ndarray:
    """The reduction a character asks for at each level over the threshold (dB, <= 0). The KNEE
    (2026-10-25, measured): on a textbook character the exact quadratic knee W wide, centred on the
    threshold - the hard curve averaged over a W-wide window of level, which is what the reference
    does there; CLEAN its own measured knee (comp_tables.clean_knee: the reference's default style
    bends otherwise); a table character reads it off its table's knee axis (VOCAL has none)."""
    over = np.asarray(over_db, dtype=np.float64)
    table = COMP_CHARACTERS[style][9]
    from . import comp_tables
    if table is None or table == "clean_knee":
        slope = 1.0 - 1.0 / max(float(ratio), 1.0)
        w = max(float(knee_db), 0.0)
        if w <= 0.0:
            return np.where(over > 0.0, -over * slope, 0.0)
        if table == "clean_knee":
            return -comp_tables.clean_knee(over, w) * slope
        knee = np.where(over <= -w / 2.0, 0.0,
                        np.where(over >= w / 2.0, over, (over + w / 2.0) ** 2 / (2.0 * w)))
        return -knee * slope
    return comp_tables.row_gain_db(comp_tables.static_row(table, threshold_db, ratio, knee_db), over)


def _character_comp(audio: np.ndarray, sample_rate: int, threshold_db: float,
                    ratio: float, attack_ms: float, release_ms: float,
                    makeup_db: float = 0.0, style: int = 0, knee_db: float = 0.0) -> np.ndarray:
    """The compressor device (`comp`), on (channels, frames): a PEAK detector linked across the
    channels, the character's static law and its ballistics. Returns the wet signal, makeup
    included, in the input's dtype; a signal the law asks nothing of comes back untouched, bit for
    bit - and so does a ratio of 1 in EVERY character (the knob's bottom is off: VOCAL's measured
    law ignores the ratio, and the Mid / Side half nobody picked is seeded at 1:1 to pass through)."""
    makeup = 10.0 ** (float(makeup_db) / 20.0)
    if float(ratio) <= 1.0:
        return audio if makeup == 1.0 else (audio * audio.dtype.type(makeup))
    detector = np.max(np.abs(audio), axis=0).astype(np.float64)
    over = comp_detector_db(detector, sample_rate, style) - float(threshold_db)
    target = comp_static_db(over, threshold_db, ratio, style, knee_db)
    if not np.any(target < 0.0):
        return audio if makeup == 1.0 else (audio * audio.dtype.type(makeup))
    env = comp_character_gain_db(target, sample_rate, attack_ms, release_ms, style)
    gain = (10.0 ** (env / 20.0)) * makeup
    return (audio * gain[np.newaxis, :]).astype(audio.dtype)


def _block_gain_to_frames(gain_db: np.ndarray, n_frames: int,
                          sample_rate: int,
                          block_s: float = ENV_BLOCK_S) -> np.ndarray:
    """Expand a per-block gain curve (dB) to per-frame linear gain, INTERPOLATED
    between block centres.

    This used to be a `np.repeat` zero-order hold, which wrote a 5 ms gain
    STAIRCASE into the deliverable: a gate opening moved the gain up to ~18 dB
    (and, on a quiet floor, far more) in a single sample, putting a step
    discontinuity in the audio that measured +22 dB of energy above 2 kHz
    against the same curve interpolated — an audible click on every gate
    opening, in the exported file only. The native monitor is per-sample and
    smooth, so the monitor sounded clean and the render did not.

    Interpolating between block CENTRES (not edges) keeps the envelope's timing:
    the block's gain still lands in the middle of the block it was measured
    over, and `np.interp` holds the first/last value at the edges rather than
    extrapolating.
    """
    block = max(1, int(block_s * sample_rate))
    gain_lin = 10.0 ** (np.asarray(gain_db, dtype=np.float64) / 20.0)
    if gain_lin.size == 0:
        return np.ones(n_frames, dtype=np.float64)
    if gain_lin.size == 1:
        return np.full(n_frames, gain_lin[0], dtype=np.float64)
    centres = np.arange(gain_lin.size) * block + (block - 1) / 2.0
    return np.interp(np.arange(n_frames, dtype=np.float64), centres, gain_lin)


def _block_peak_db(audio: np.ndarray, sample_rate: int,
                   block_s: float = ENV_BLOCK_S) -> np.ndarray:
    """Channel-linked PEAK per non-overlapping 5 ms block, in dBFS, on
    (channels, frames). The dynamics-detector convention the native monitor's
    per-sample peak detectors approximate (gate / de-ess) — peaks, not RMS,
    so a hit registers within its first block."""
    block = max(1, int(block_s * sample_rate))
    n_frames = audio.shape[1]
    n_blocks = -(-n_frames // block)
    padded = np.zeros((audio.shape[0], n_blocks * block))
    padded[:, :n_frames] = np.abs(audio)
    peak = padded.reshape(audio.shape[0], n_blocks, block).max(axis=2).max(axis=0)
    return 20.0 * np.log10(np.maximum(peak, 1e-12))


# Saturation models (Phase 45) — all MEMORYLESS + deterministic, each normalized so the
# small-signal slope is 1 (drive 0 = unity) and the output is bounded. Since 2026-10-26 the four are
# CHARACTERS fitted to a reference saturator (`sat_characters`: soft / tape / tube / heavy under the
# stored words tanh / warm / tube / hard); tape and tube add a linear tone after the curve. Shared by
# the single-band and the multiband (Saturn-style) saturator.
SAT_TYPES = _satc.NAMES   # curve index 0..3 -> name


def _curve_index(curve) -> int:
    """Saturation curve as an int index (0 tanh/1 warm/2 tube/3 hard); accepts the name too."""
    return _satc.curve_index(curve)


def _waveshape(x: np.ndarray, drive_db: float, curve=0) -> np.ndarray:
    """One character's curve at `drive_db` (sat_characters.shape), bounded, its level law after it. The
    tape and tube tones are NOT in here - they are linear, and run after it (`_satc.apply_tone`)."""
    return _satc.shape(x, drive_db, curve)

def _oversampled(x, os_n: int, shape):
    """Run a MEMORYLESS shaper at ``os_n``x (2/4/8) the sample rate and come back down.

    A hard nonlinearity generates harmonics above Nyquist; at the base rate those fold
    back into the audible band as INHARMONIC aliasing (a 9 kHz tone driven hard puts
    energy at 3 kHz, which no amount of EQ can remove afterwards). Shaping at Nx moves
    the fold point Nx higher and the decimation filter discards what is left.
    ``resample_poly`` is deterministic and linear-phase (delay-compensated), so this is
    reproducible render to render. ``os_n`` 1 returns the exact elementwise path, so
    every render made before this option existed is bit-unchanged. Mirrors ``_clip``'s
    v2 path; the native monitor uses ``juce::dsp::Oversampling`` (a different filter),
    so an oversampled render is oracle-TOLERANCE, not bit-exact — see docs/10-daw.md.
    """
    if os_n <= 1:
        return shape(x)
    from scipy.signal import resample_poly
    a = np.asarray(x, dtype=np.float64)
    up = resample_poly(a, os_n, 1, axis=0)
    return resample_poly(shape(up), 1, os_n, axis=0)[: a.shape[0]]


def _saturate(audio: np.ndarray, entry: dict, sample_rate: int = 48000) -> np.ndarray:
    """Deterministic waveshaper saturator (audio-dsp skill). Single-band when the entry
    has no `bands`; otherwise a 3-band (Saturn-style) saturator (needs sample_rate for the
    FFT split). `output_db` trims the output post-shaper; `type` selects the curve.
    `oversample` (2/4/8) runs the curve at N x the sample rate so its harmonics stop
    folding back as aliasing (2026-09-06); the default is SAT_OVERSAMPLE_DEFAULT (2026-10-24,
    it was 1), and 1 is the original elementwise path. The multiband form runs its whole split
    at that rate."""
    if isinstance(entry.get("low"), dict):    # multiband: top-level low/mid/high (like mb_comp)
        return _multiband_saturate(audio, entry, sample_rate)
    # held to the table and the oversample snapped down to 1/2/4/8, as the native's parse does
    # (2026-10-17, the saturation audit: a mix outside 0..1 or an oversample of 3 rendered otherwise)
    os_req = int(entry.get("oversample", SAT_OVERSAMPLE_DEFAULT) or 1)
    os_n = 8 if os_req >= 8 else 4 if os_req >= 4 else 2 if os_req >= 2 else 1
    drive = float(np.clip(entry.get("drive_db", _satc.DRIVE_DEFAULT_DB), *_satc.DRIVE_RANGE_DB))
    curve = entry.get("curve", 0)
    out = _oversampled(audio, os_n, lambda v: _waveshape(v, drive, curve))
    # the character's tone (tape / tube), linear, at the base rate the shaper came back down to
    out = _satc.apply_tone(out, curve, sample_rate, axis=0)
    output_db = float(np.clip(entry.get("output_db", 0.0), -12.0, 12.0))
    if output_db:
        out = out * 10.0 ** (output_db / 20.0)
    mix = float(np.clip(entry.get("mix", 1.0), 0.0, 1.0))   # dry/wet (parallel saturation); 1.0 = fully wet
    if mix < 1.0:
        out = out * mix + audio * (1.0 - mix)
    return out


def mb_sat_oversample(entry: dict) -> int:
    """The factor a multiband saturation runs at: its `oversample` snapped to 1/2/4/8, never below
    SAT_OVERSAMPLE_DEFAULT (2026-10-24)."""
    os_req = int(entry.get("oversample", SAT_OVERSAMPLE_DEFAULT) or 1)
    os_n = 8 if os_req >= 8 else 4 if os_req >= 4 else 2 if os_req >= 2 else 1
    return max(os_n, SAT_OVERSAMPLE_DEFAULT)


def _multiband_saturate(audio: np.ndarray, entry: dict, sample_rate: int) -> np.ndarray:
    """3-band saturator (Phase 45): the real 3-way LR4 split at movable crossovers (since 2026-10-17),
    a per-band waveshaper (drive / output / mix / optional per-band type), summed back,
    then a global dry/wet mix. (frames, channels) float64 in/out. Each band is shaped by
    its OWN drive; per-band mix < 1 keeps that band partly dry (parallel saturation)."""
    # _saturate works on (frames, channels) float64; the split works on (channels, frames).
    x = np.ascontiguousarray(audio.T)            # (channels, frames)
    frames = x.shape[-1]
    # THE WHOLE DEVICE RUNS AT `oversample` x (2026-10-24): the split and the three shapers. At 1x
    # a hard-driven high band folded its harmonics back as aliasing nearly as loud as the band.
    # Never below SAT_OVERSAMPLE_DEFAULT: the form has no OS control on its face, and every one
    # made before this stored 1 (MonitorChain's parse holds the same floor).
    os_n = mb_sat_oversample(entry)
    if os_n > 1:
        from scipy.signal import resample_poly
        x = resample_poly(x, os_n, 1, axis=-1)
    sr = sample_rate * os_n
    default_lo, default_hi = MB_CROSSOVERS_HZ
    lo_x = float(np.clip(entry.get("crossover_lo_hz", default_lo), 30.0, 4000.0))
    hi_x = float(np.clip(entry.get("crossover_hi_hz", default_hi), 200.0, 16000.0))
    if hi_x <= lo_x:
        hi_x = lo_x * 1.5
    gcurve = entry.get("curve", 0)
    out = np.zeros(x.shape, dtype=np.float64)
    # A band sub-object may be PARTIAL or absent: `set_master_device_param` creates one band
    # on first nested write, and the native monitor already tolerates that
    # (`monitor_chain_params` fills the gaps). Read the same way here — a missing band
    # inherits the entry's flat drive, exactly what `set_sat_multiband` seeds it from — so a
    # half-written entry renders like the monitor instead of raising KeyError mid-render.
    flat_drive = float(entry.get("drive_db", _satc.DRIVE_DEFAULT_DB))   # the native's default too (the multiband saturation audit)
    # The same tapered + zero-padded split mb_comp and OTT moved to on 2026-09-08 (audit
    # 03-F1). The hard rectangular mask this replaces summed back perfectly but each band
    # ALONE carried audible content inside bit-exact digital silence, and the leakage only
    # cancels while the bands take the SAME gain — which a multiband saturator never does.
    # THE DAW'S SPLIT since 2026-10-17 (the multiband saturation audit): the real 3-way LR4
    # (`_lr4_three_way`, the multiband comp's and OTT's), in float64 as the native runs it in double.
    # It was the FFT partition above here against a native COMPLEMENT (mid = dry - low - high).
    split = _lr4_three_way(x, sr, lo_x, hi_x)
    for (name, _lo, _hi), band in zip((("low", 0.0, lo_x), ("mid", lo_x, hi_x),
                                       ("high", hi_x, None)), split):
        p = entry.get(name) or {}
        pc = p.get("curve")                                      # per-band curve, else the global one
        drive = float(np.clip(p.get("drive_db", flat_drive), *_satc.BAND_DRIVE_RANGE_DB))   # held as the native's parse holds it
        bc = gcurve if pc is None else pc
        # the band's character, then its tone at the rate the band runs at (the oversampled one)
        shaped = _satc.apply_tone(_waveshape(band, drive, bc), bc, sr, axis=-1)
        if p.get("output_db"):
            shaped = shaped * 10.0 ** (float(np.clip(p["output_db"], -12.0, 12.0)) / 20.0)
        mix = float(np.clip(p.get("mix", 1.0), 0.0, 1.0))
        out += shaped * mix + band * (1.0 - mix)                 # per-band parallel mix
    if os_n > 1:
        from scipy.signal import resample_poly
        out = resample_poly(out, 1, os_n, axis=-1)[:, :frames]
    # NO DC blocker after the sum (2026-10-26, measured on the reference's multiband form: it has
    # none): the one asymmetric curve, the tube, takes its own DC out in its tone, band by band.
    wet = out.T                                                  # (frames, channels)
    gmix = float(np.clip(entry.get("mix", 1.0), 0.0, 1.0))
    return wet * gmix + audio * (1.0 - gmix) if gmix < 1.0 else wet


# --------------------------------------------------------------------------- modulation
# The modulation family's shared core (2026-09-06). A chorus/flanger is a delay line whose
# length is swept by an LFO; the sweep is what makes it, and the two things that decide
# whether it survives the parity triangle are the LFO's PHASE SOURCE and the fractional-delay
# INTERPOLATION.
#
# PHASE comes from the HOST TIMELINE (sample index / sample_rate), never from a free-running
# accumulator. That is what lets the native monitor and this renderer agree: MonitorChain gets
# `playheadSamples` and derives the same t, so the same sample of the same take always sees the
# same point of the sweep — the contract `time_range_s` already relies on. A free accumulator
# would drift with block size and make every export unreproducible.
#
# INTERPOLATION is JUCE's `Lagrange3rd` stencil, reproduced here EXACTLY so the native
# `juce::dsp::DelayLine<float, Lagrange3rd>` and this function are the same filter rather than
# two plausible ones. Linear interpolation was rejected: on a swept delay its high-frequency
# loss varies WITH the sweep, which is audible as a dull wobble.
#
# Reproducing it EXACTLY means reproducing `updateInternalVariables()`, not just the four taps
# and the basis. JUCE SHIFTS THE STENCIL BACK ONE SAMPLE before reading:
#
#     if (delayFrac < 2.0 && delayInt >= 1) { delayFrac++; delayInt--; }
#
# `setDelay` always leaves `delayFrac` in [0,1), so that branch fires for EVERY delay >= 1 —
# it is the normal path, not an edge case. The taps therefore sit at d-1, d, d+1, d+2 and the
# cubic is evaluated at f+1, i.e. in the CENTRE interval of the stencil rather than at its
# edge. Cubic Lagrange is not shift-invariant across stencil windows, so getting this wrong
# is a different filter at the same nominal delay — and it agrees EXACTLY at f == 0, which is
# the one case a naive test exercises. Measured before this shift was mirrored here: the two
# stencils differed by -22.9 dB (7.2 % of the wet) on a 9 kHz input, and by 0.0 at depth 0.
#
# THIS APPLIES TO EVERY MODULATION DEVICE ADDED LATER (flanger, phaser): if it reads a
# fractional delay through `juce::dsp::DelayLine<..., Lagrange3rd>` natively, its Python
# mirror must carry the same shift, and its parity test must use a NON-ZERO fractional part.
CHORUS_MIN_DELAY_SAMPLES = 4.0    # the Lagrange stencil reaches d+3; keep it inside the line
# The native line is sized once at prepare() for the family's worst case; the offline
# renderer clamps to the SAME ceiling (in seconds, not samples — the line is sized from the
# sample rate) so a hostile delay_ms can never read past its end on either engine.
MODULATION_LINE_SECONDS = 0.105


def _lfo_phase(n: int, sample_rate: int, rate_hz: float, phase_offset: float,
               offset: int = 0) -> np.ndarray:
    """The family's LFO, sampled on the host timeline. `phase_offset` is in CYCLES (0..1);
    `offset` is the absolute sample index this slice starts at, so a chunked caller stays on
    the same timeline the unchunked one rides."""
    t = np.arange(offset, offset + n, dtype=np.float64) / float(sample_rate)
    return 2.0 * np.pi * (rate_hz * t + phase_offset)


def _lagrange3_taps(x: np.ndarray, delay: np.ndarray, offset: int = 0) -> np.ndarray:
    """Read `x` (frames,) at a per-sample fractional `delay` behind each index, with JUCE's
    Lagrange3rd stencil. Out-of-range reads (the first samples, where the line is not yet
    full) come back as silence, exactly like a freshly cleared delay line.

    `delay` may cover only a SLICE of `x`, starting at absolute index `offset`; the gather still
    reads the whole of `x`, so a chunked caller sees exactly what an unchunked one would. That is
    what lets `_chorus` bound its temporaries without changing a sample (see the note there)."""
    n = x.shape[0]
    m = delay.shape[0]
    idx = np.arange(offset, offset + m, dtype=np.float64)
    d_int = np.floor(delay).astype(np.int64)
    f = (delay - d_int).astype(np.float64)
    # JUCE's `updateInternalVariables()`, mirrored: the stencil shifts back one sample so the
    # cubic is evaluated in its CENTRE interval. `f` is in [0,1) here, so the `delayFrac < 2.0`
    # half of JUCE's condition is always true and only `delayInt >= 1` needs testing.
    shift = d_int >= 1
    d_int = d_int - shift
    f = f + shift
    # v1..v4 sit at delays d_int, d_int+1, d_int+2, d_int+3 (v2 is the OLDER neighbour)
    base = idx.astype(np.int64) - d_int
    taps = []
    for k in range(4):
        j = base - k
        taps.append(np.where(j >= 0, x[np.clip(j, 0, n - 1)], 0.0))
    v1, v2, v3, v4 = taps
    d1, d2, d3 = f - 1.0, f - 2.0, f - 3.0
    c1 = -d1 * d2 * d3 / 6.0
    c2 = d2 * d3 * 0.5
    c3 = -d1 * d3 * 0.5
    c4 = d1 * d2 / 6.0
    return v1 * c1 + f * (v2 * c2 + v3 * c3 + v4 * c4)


def _chorus(audio: np.ndarray, sample_rate: int, entry: dict) -> np.ndarray:
    """Chorus (2026-10-28: fitted to a reference two-line chorus, with its knob set): MODE I / II /
    I+II, STEREO, MIX, VOLUME - `chorus_law.apply`, which says how it was measured. It reads its
    lines through `_lagrange3_taps` on the host timeline, as the flanger does."""
    from .chorus_law import apply
    return apply(audio, sample_rate, entry)


FLANGER_MIN_DELAY_SAMPLES = 4.0   # the same stencil floor the chorus keeps, for the same reason


def _flanger(audio: np.ndarray, sample_rate: int, entry: dict) -> np.ndarray:
    """Flanger (2026-09-06) — the modulation family's second device, on (frames, channels)
    float64.

    A chorus with a RETURN PATH. One voice reads a delay swept around `delay_ms` by `depth`
    (`delay_ms * (1 + depth*sin)` — the family's sweep, and the right channel offset by
    `stereo_deg`), but what the line recirculates is the input PLUS `feedback` of the line's
    own output. That return is the whole difference between the two devices: it turns the
    comb's gentle dips into resonant teeth, and the SIGN of the feedback decides which tooth
    you get. Positive resonates the peaks at k/delay, so the first NOTCH sits between them at
    1/(2*delay); negative puts the notches on k/delay itself and drops the whole picture an
    octave. Sweeping that comb is the jet-plane sound; the chorus, with no return at all, is
    the thickener.

    THE RECURSION, and why this function is shaped the way it is. With feedback the output
    stops being a function of the input alone:

        wet[n] = read(buf, d[n])              # the swept fractional read
        buf[n] = x[n] + feedback * wet[n]     # what recirculates

    `buf[n]` depends on `buf` around `n - d[n]`, so this can NOT be the single vectorised
    gather `_chorus` is. It can still be done in BLOCKS: every read lands at least `di`
    samples back (`di` is the stencil's integer part, >= 3 under the 4-sample floor), so a
    block of `di` output samples reads nothing but already-final values and the whole block
    goes in one gather. The block length is therefore the LOCAL delay, recomputed as the
    sweep moves — not the sweep's global minimum, which would pin every block to the bottom
    of the trough and cost 8x (measured on a 3-minute stereo stem at delay 3 ms / depth 0.8:
    1.8 s adaptive vs 14.5 s pinned; the pathological 0.5 ms / depth 0.9 is 14.9 s adaptive
    and minutes pinned). The tap indices and the four Lagrange coefficients are built ONCE
    per outer block with the out-of-range mask folded into the coefficients, so the inner
    loop is four takes and four multiply-adds and nothing else — that fold is worth 8x on its
    own, and it is what makes a per-block recursion affordable in numpy at all.

    With `feedback` at 0 there is no recursion: the line carries the dry input, the block
    loop collapses to one gather, and this costs exactly what a one-voice chorus costs."""
    x = np.ascontiguousarray(np.asarray(audio, dtype=np.float64))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    n, nch = x.shape
    if n == 0:
        return x
    # held to the native parse's range (MonitorChain.cpp, "flanger"; audit 2026-10-17)
    rate = float(np.clip(entry.get("rate_hz", 0.5), 0.01, 10.0))
    depth = float(np.clip(entry.get("depth", 0.6), 0.0, 1.0))
    centre = float(np.clip(entry.get("delay_ms", 3.0), 0.5, 20.0))
    fb = float(np.clip(entry.get("feedback", 0.0), -0.95, 0.95))
    mix = float(np.clip(entry.get("mix", 0.5), 0.0, 1.0))
    spread = float(np.clip(entry.get("stereo_deg", 0.0), 0.0, 180.0)) / 360.0

    max_d = min((centre * (1.0 + depth) / 1000.0) * sample_rate + 4.0,
                float(int(MODULATION_LINE_SECONDS * sample_rate) + 8 - 1))
    # `buf` is the delay line's contents. With no feedback it IS the dry input (never
    # written), so the fast path reads x directly rather than copying it.
    buf = x if fb == 0.0 else x.copy()
    bflat = buf.reshape(-1)
    wet = np.empty_like(x)
    wflat = wet.reshape(-1)
    xflat = x.reshape(-1)
    ar = np.arange(nch, dtype=np.int64)

    # OUTER blocks bound the temporaries, exactly as `_chorus` does and for the same measured
    # reason: four index arrays plus four coefficient arrays at full length is ~128 bytes per
    # frame per channel (1.1 GB on a 3-minute stereo stem). Per outer block it is ~67 MB
    # whatever the stem's length. This changes no sample: every index is absolute.
    outer = 1 << 19
    for bs in range(0, n, outer):
        be = min(bs + outer, n)
        m = be - bs
        # The family's LFO, on the host timeline (see `_lfo_phase`). One voice, so the
        # stereo offset is the plain cycle fraction — the chorus divides by its voice count
        # because a uniform lattice is invariant under a shift of its own spacing, and a
        # single voice has no lattice to be invariant under.
        ph = np.stack([_lfo_phase(m, sample_rate, rate, spread if ch == 1 else 0.0, offset=bs)
                       for ch in range(nch)], axis=1)
        d = np.clip(centre * (1.0 + depth * np.sin(ph)) / 1000.0 * sample_rate,
                    FLANGER_MIN_DELAY_SAMPLES, max_d)
        di = np.floor(d).astype(np.int64)
        f = d - di
        # JUCE's `updateInternalVariables()` stencil shift — the note above `_lagrange3_taps`
        # explains why omitting it is a different filter at the same nominal delay.
        sh = di >= 1
        di = di - sh
        f = f + sh
        d1, d2, d3 = f - 1.0, f - 2.0, f - 3.0
        coef = (-d1 * d2 * d3 / 6.0, f * d2 * d3 * 0.5, -f * d1 * d3 * 0.5, f * d1 * d2 / 6.0)
        base = np.arange(bs, be, dtype=np.int64)[:, None] - di    # absolute frame of tap 0
        # Flat (frame*nch + channel) indices, and the coefficients with the "before the start
        # of the signal" mask already folded in — a read off the front returns silence, like a
        # freshly cleared delay line, without a np.where in the inner loop.
        jidx, jcof = [], []
        for k in range(4):
            j = base - k
            jidx.append((np.clip(j, 0, n - 1) * nch + ar).reshape(-1))
            jcof.append((coef[k] * (j >= 0)).reshape(-1))

        if fb == 0.0:
            acc = np.take(bflat, jidx[0]) * jcof[0]
            for k in (1, 2, 3):
                acc += np.take(bflat, jidx[k]) * jcof[k]
            wflat[bs * nch:be * nch] = acc
            continue

        # A block [s, e) is safe when every read lands strictly before s. The newest tap of
        # frame s+j sits at (s+j) - di[s+j], so the block is valid while j < di[s+j] — i.e.
        # its length is bounded by the SMALLEST integer delay inside it. `w` is a lookahead
        # window: the minimum over a window at least as long as the block it authorises is a
        # valid bound for that block, which is what makes one np.min per block enough.
        dimin = di.min(axis=1)
        s = 0
        while s < m:
            w = min(m - s, max(4, int(dimin[s]) * 2 + 8))
            c = max(1, min(int(dimin[s:s + w].min()), w))
            e = min(s + c, m)
            a0, a1 = (bs + s) * nch, (bs + e) * nch
            g0, g1 = s * nch, e * nch
            acc = np.take(bflat, jidx[0][g0:g1]) * jcof[0][g0:g1]
            for k in (1, 2, 3):
                acc += np.take(bflat, jidx[k][g0:g1]) * jcof[k][g0:g1]
            wflat[a0:a1] = acc
            bflat[a0:a1] = xflat[a0:a1] + fb * acc
            s = e
    return wet * mix + x * (1.0 - mix) if mix < 1.0 else wet


AUTO_FILTER_BLOCK = 512   # the block-parallel walk's block length (any length gives the same samples)


def _svf_sweep(x: np.ndarray, g: np.ndarray, k: float, outputs: tuple,
               block: int = AUTO_FILTER_BLOCK) -> np.ndarray:
    """One TPT state-variable section whose `g` moves EVERY SAMPLE, on (frames, channels)
    float64 - exactly the recursion `reference_monitor._auto_filter` walks sample by sample,
    walked block-parallel so a whole song costs seconds, not minutes.

    The section is a linear time-varying system on its two integrator states s = (ic1, ic2):

        s[n+1] = A[n] s[n] + b[n] x[n],   y[n] = C[n] s[n] + D[n] x[n]

    with A, b, C, D functions of g[n] alone. Cut the signal into blocks of `block` samples and
    lay every block of every channel side by side as a column: one pass over the positions in a
    block, vectorised across ALL the columns at once, gives each block's zero-state end state z
    and its transition product P = A[last]...A[first]. The true state at each block's start
    then follows from a short serial scan over the blocks (S[m+1] = P[m] S[m] + z[m], from rest),
    and a second pass from those states produces the output. Both passes are `block` numpy steps
    over (blocks x channels)-long arrays; the scan is one multiply-add per block. The output is
    the same recursion to rounding (the scan sums in another order), which the oracle null holds.
    A per-sample Python loop was not an option for a song (`_gr_ballistics_db`'s rule)."""
    n, nch = x.shape
    K = int(block)
    M = -(-n // K)
    pad = M * K - n

    def cols(a: np.ndarray, mode: str) -> np.ndarray:
        a = np.pad(a, ((0, pad), (0, 0)), mode=mode) if pad else a
        return np.ascontiguousarray(a.reshape(M, K, nch).transpose(1, 0, 2).reshape(K, M * nch))

    xc = cols(x, "constant")
    gc = cols(g, "edge")
    a1 = 1.0 / (1.0 + gc * (gc + k))
    a2 = gc * a1
    a3 = gc * a2
    A11 = 2.0 * a1 - 1.0
    A12 = -2.0 * a2
    A22 = 1.0 - 2.0 * a3
    B2 = 2.0 * a3
    # pass 1: zero-state end states and transition products, every block at once
    z1 = np.zeros(M * nch)
    z2 = np.zeros(M * nch)
    p11 = np.ones(M * nch)
    p12 = np.zeros(M * nch)
    p21 = np.zeros(M * nch)
    p22 = np.ones(M * nch)
    for j in range(K):
        a11, a12, a21, a22 = A11[j], A12[j], -A12[j], A22[j]
        xj = xc[j]
        z1, z2 = a11 * z1 + a12 * z2 - a12 * xj, a21 * z1 + a22 * z2 + B2[j] * xj
        p11, p12, p21, p22 = (a11 * p11 + a12 * p21, a11 * p12 + a12 * p22,
                              a21 * p11 + a22 * p21, a21 * p12 + a22 * p22)
    # the scan: each block's starting state, per channel, from rest
    s1 = np.zeros(M * nch)
    s2 = np.zeros(M * nch)
    P = np.stack([p11, p12, p21, p22, z1, z2], axis=1).reshape(M, nch, 6).tolist()
    s1v = s1.reshape(M, nch)
    s2v = s2.reshape(M, nch)
    for ch in range(nch):
        c1 = c2 = 0.0
        for m in range(M):
            s1v[m, ch] = c1
            s2v[m, ch] = c2
            q11, q12, q21, q22, e1, e2 = P[m][ch]
            c1, c2 = q11 * c1 + q12 * c2 + e1, q21 * c1 + q22 * c2 + e2
    # pass 2: the output from the true starting states
    oc1, oc2, od = outputs
    y = np.empty_like(xc)
    for j in range(K):
        a11, a12, a21, a22 = A11[j], A12[j], -A12[j], A22[j]
        xj = xc[j]
        aa1, aa2, aa3 = a1[j], a2[j], a3[j]
        v1 = aa1 * s1 - aa2 * s2 + aa2 * xj
        v2 = aa2 * s1 + (1.0 - aa3) * s2 + aa3 * xj
        y[j] = oc1 * v1 + oc2 * v2 + od * xj
        s1, s2 = a11 * s1 + a12 * s2 - a12 * xj, a21 * s1 + a22 * s2 + B2[j] * xj
    return y.reshape(K, M, nch).transpose(1, 0, 2).reshape(M * K, nch)[:n]


def _key_level(key_audio: np.ndarray, n: int, sample_rate: int,
               key_offset_s: float = 0.0) -> np.ndarray:
    """A trigger's per-sample level (the louder channel's |x|), laid on the keyed stem's
    timeline: `key_offset_s` = the key stem's `capture_start_s` minus this one's. The Duck's
    detector and the Auto Filter's envelope read the trigger through this one function."""
    key = np.asarray(key_audio, dtype=np.float32)
    level = np.max(np.abs(key), axis=1) if key.ndim == 2 else np.abs(key)
    shift = int(round(float(key_offset_s) * sample_rate))
    if shift > 0:      # the key starts later than the keyed stem: silence before it
        level = np.concatenate([np.zeros(shift, dtype=np.float32), level])
    elif shift < 0:    # the key started earlier: its first samples precede this stem
        level = level[-shift:]
    lev = np.zeros(n, dtype=np.float32)
    m = min(n, level.shape[0])
    lev[:m] = level[:m]
    return lev


def _auto_filter(audio: np.ndarray, sample_rate: int, entry: dict, tempo=None,
                 key_level: Optional[np.ndarray] = None) -> np.ndarray:
    """Auto Filter (Phase 63) on (frames, channels) float64: the law in `reference_monitor`
    (the cutoff from the LFO, the envelope and their octaves; the TPT state-variable filter),
    the recursion walked block-parallel (`_svf_sweep`). `tempo` = the song's `TempoMap` for a
    synced LFO (None = 4/4 at 120); `key_level` = the trigger's per-sample level (`_key_level`)
    when the entry is keyed and the trigger is heard, else the envelope follows the input."""
    x = np.ascontiguousarray(np.asarray(audio, dtype=np.float64))
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if x.shape[0] == 0:
        return x
    v = auto_filter_values(entry)
    key = None if key_level is None else np.asarray(key_level, dtype=np.float64)
    fc = auto_filter_modulation(x, sample_rate, v, key=key, tempo=tempo,
                                follower=_rising_follower)
    g = np.tan(np.pi * fc / float(sample_rate))
    y = x
    for q in auto_filter_qs(v["filter_type"], v["slope_db_oct"], v["resonance"]):
        k = 1.0 / q
        y = _svf_sweep(y, g, k, auto_filter_outputs(v["filter_type"], k))
    mix = v["mix"]
    return y * mix + x * (1.0 - mix) if mix < 1.0 else y


def _tempo_of(session: Session):
    """The song's `TempoMap` for a synced Auto Filter, from the engine fact the DAW sent (None =
    the DAW has sent none: 4/4 at 120, `auto_filter_cycles`' own default)."""
    from .tempo_map import TempoMap
    fact = session.tempo_map
    return TempoMap.from_fact(fact) if fact else None


def od_shape(v, k: float):
    """The overdrive's clipper: the exponential soft-clip sign(u)(1 - e^-|u|) at u = k·v + OD_BIAS,
    the bias's own output taken back out and the slope at 0 scaled to 1 - bounded, zero at zero,
    unity small-signal gain, and one side clipping sooner than the other (even harmonics)."""
    b = OD_BIAS
    u = k * np.asarray(v, dtype=np.float64) + b
    f0 = 1.0 - np.exp(-b)
    return (np.sign(u) * (1.0 - np.exp(-np.abs(u))) - f0) * np.exp(b) / k


def _overdrive(audio: np.ndarray, sample_rate: int, entry: dict) -> np.ndarray:
    """Overdrive distortion (2026-07-07; voiced 2026-10-24) on (frames, channels) float64: an
    optional resonant band-pass pre-filter (`filter_on`; the JUCE-transcribed
    `biquad_coeffs("bandpass")`, so the native transcription is coefficient-identical), a low
    shelf at OD_EMPHASIS_HZ cutting OD_EMPHASIS_DB (the lows clip less), the biased exponential
    soft-clipper `od_shape` at k = 10^(drive/20), run at `oversample` x (default
    OD_OVERSAMPLE_DEFAULT), the shelf given back, a two-pole roll-off at OD_FIZZ_HZ, a DC blocker
    at SHAPER_DC_HZ, a one-pole `tone` tilt around OD_TONE_HZ (0 dark / 0.5 = exact passthrough /
    1 bright), a post `output_db` trim, then dry/wet `mix` against the UNfiltered input.
    Deterministic (fresh filter state per render)."""
    from scipy.signal import lfilter
    # Every value held to the table as the native's parse holds it (2026-10-17, the overdrive audit)
    x = audio
    if entry.get("filter_on", False):
        freq = float(np.clip(entry.get("freq_hz", 800.0), 40.0, min(12000.0, 0.45 * sample_rate)))
        b, a = biquad_coeffs("bandpass", sample_rate, freq,
                             float(np.clip(entry.get("q", 0.7), 0.3, 8.0)), 0.0)
        x = lfilter(b, a, x, axis=0)
    drive_db = float(np.clip(entry.get("drive_db", 12.0), 0.0, 36.0))
    # Only the SHAPER is oversampled: the filters are linear and alias nothing, and running
    # them at the base rate keeps them coefficient-identical to the native transcription
    # (the whole point of biquad_coeffs).
    os_req = int(entry.get("oversample", OD_OVERSAMPLE_DEFAULT) or 1)
    os_n = 8 if os_req >= 8 else 4 if os_req >= 4 else 2 if os_req >= 2 else 1
    character = _odc.character_index(entry.get("character", _odc.DEFAULT))
    if character == 0:
        # the PEDAL, as it always was
        k = 10.0 ** (drive_db / 20.0)
        x = lfilter(*biquad_coeffs("low_shelf", sample_rate, OD_EMPHASIS_HZ, 0.707, -OD_EMPHASIS_DB), x, axis=0)
        shaped = _oversampled(x, os_n, lambda v: od_shape(v, k))
        shaped = lfilter(*biquad_coeffs("low_shelf", sample_rate, OD_EMPHASIS_HZ, 0.707, OD_EMPHASIS_DB), shaped, axis=0)
        shaped = lfilter(*biquad_coeffs("lpf2", sample_rate, OD_FIZZ_HZ, 0.707, 0.0), shaped, axis=0)
        shaped = lfilter(*biquad_coeffs("hpf", sample_rate, SHAPER_DC_HZ, 0.707, 0.0), shaped, axis=0)
    else:
        # an AMP (od_characters, 2026-10-26): its pre filters, its curve at s, its post filters,
        # its cabinet's reflections, its trim - the post high-passes take the curve's DC out
        for kind, f, q, g in _odc.pre_sections(character, sample_rate):
            x = lfilter(*biquad_coeffs(kind, sample_rate, f, q, g), x, axis=0)
        amp = _odc.AMPS[character]
        s = 10.0 ** ((drive_db + amp["s0_db"]) / 20.0)
        lev = 10.0 ** (_odc.level_db(drive_db, character) / 20.0)   # its level law (2026-10-26)
        shaped = _oversampled(x, os_n, lambda v: _odc.amp_curve(v, s, character) * lev)
        for kind, f, q, g in _odc.post_sections(character, sample_rate):
            shaped = lfilter(*biquad_coeffs(kind, sample_rate, f, q, g), shaped, axis=0)
        shaped = _odc.cab(shaped, character, sample_rate, axis=0) * 10.0 ** (_odc.TRIMS_DB[character] / 20.0)
    tone = float(np.clip(entry.get("tone", 0.5), 0.0, 1.0))
    if tone != 0.5:
        lp = _one_pole_lp(shaped, sample_rate, OD_TONE_HZ)
        shaped = lp + (2.0 * tone) * (shaped - lp)
    output_db = float(np.clip(entry.get("output_db", 0.0), -24.0, 12.0))
    if output_db:
        shaped = shaped * 10.0 ** (output_db / 20.0)
    mix = float(np.clip(entry.get("mix", 1.0), 0.0, 1.0))
    return shaped * mix + audio * (1.0 - mix) if mix < 1.0 else shaped


TILT_Q = 0.5   # gentle (non-resonant) shelf slope for the one-knob tilt EQ


def _tilt(audio: np.ndarray, sample_rate: int, entry: dict) -> np.ndarray:
    """One-knob tilt EQ (Phase 41): a low shelf at −tilt and a high shelf at +tilt,
    both at `pivot_hz`, so the spectrum tilts around the pivot (positive = brighter).
    Two standard shelf biquads via `_eq_plugin` — bit-exact in the native monitor.
    2026-07-12: plus the device's `output_db` post trim (gain staging on the device)."""
    tilt = entry["tilt_db"]
    pivot = entry["pivot_hz"]
    low = {"shape": "low_shelf", "freq_hz": pivot, "gain_db": -tilt, "q": TILT_Q}
    high = {"shape": "high_shelf", "freq_hz": pivot, "gain_db": tilt, "q": TILT_Q}
    out = _rbj_band(_rbj_band(audio, sample_rate, low), sample_rate, high)
    out_db = entry.get("output_db", 0.0)
    if out_db:
        out = (out * np.float32(10.0 ** (out_db / 20.0))).astype(np.float32)
    return out


def _eq_entry(audio: np.ndarray, sample_rate: int, entry: dict) -> np.ndarray:
    """One eq entry: the band filter on its channel scope, then the entry's
    `output_db` trim applied FULL-signal (2026-07-12 gain staging — the EQ device's
    Output; the device's audible trim is the sum across its entries, which is exact
    because the whole bank is linear). The trim is full-stereo even for a mid/side/
    left/right band — a scalar gain commutes with any per-channel linear filter."""
    wet = _channel_apply(audio, entry.get("channel", "stereo"),
                         lambda a: _eq_band(a, sample_rate, entry["band"]))
    out_db = entry.get("output_db", 0.0)
    if out_db:
        wet = (wet * np.float32(10.0 ** (out_db / 20.0))).astype(np.float32)
    return wet


def _clip(audio: np.ndarray, entry: dict) -> np.ndarray:
    """The clipper (2026-10-28, `clip_law`: fitted to a reference clipper) on (frames, channels):
    GAIN, then a quadratic knee of SOFTNESS centred on the CLIP ceiling, each channel on its own; at
    OVERSAMPLE 2 the knee runs at 2x (resample_poly) and the output is held under 0 dBFS. An entry
    saved with the old knobs (threshold / hardness / output) is read through `clip_law.from_legacy`.
    Not a ceiling below full scale when oversampled: the resampling rings past CLIP, and on the
    master the limiter after it holds the true peak."""
    from .clip_law import apply
    return apply(audio, entry)


def _transient(audio: np.ndarray, sample_rate: int, entry: dict) -> np.ndarray:
    """Attack/sustain shaper (v2 2026-07-05 adds speed + output trim) - since 2026-10-17 the
    DAW's law (`MonitorChain::applyTransient`): a fast and a slow follower of the linked
    per-sample peak held over TRANSIENT_HOLD_MS (a running maximum), each LINEAR from rest
    (`_rising_follower`: attack when it rises), converted to dB after; transientness = how far
    the fast sits above the slow, past TRANSIENT_DEADBAND_DB, 0..1 up to TRANSIENT_FULL_DB; the per-sample gain blends attack_db and sustain_db, then `output_db`.
    `speed` divides the detector time constants. Parameters are held to the table, as the
    native's parse holds them. (frames, channels) in and out. (Until 2026-10-17: a 5 ms block
    RMS smoothed in dB here, against a per-sample follower smoothed in dB there.)"""
    speed = float(np.clip(entry.get("speed", 1.0) or 1.0, 0.25, 4.0))
    attack_db = float(np.clip(entry.get("attack_db", 0.0), -6.0, 6.0))
    sustain_db = float(np.clip(entry.get("sustain_db", 0.0), -6.0, 6.0))
    output_db = float(np.clip(entry.get("output_db", 0.0), -12.0, 12.0))
    x = np.asarray(audio)
    peak = (np.max(np.abs(x), axis=1) if x.ndim == 2 else np.abs(x)).astype(np.float64)
    hold = int(sample_rate * TRANSIENT_HOLD_MS / 1000.0)
    if hold > 0:   # the running maximum over [i - hold, i], from rest (the gate's convention)
        from scipy.ndimage import maximum_filter1d
        padded = np.concatenate([np.zeros(hold), peak])
        peak = maximum_filter1d(padded, size=hold + 1, origin=hold // 2)[hold:]
    fast = _rising_follower(peak, sample_rate, TRANSIENT_FAST_ATTACK_MS / speed,
                            TRANSIENT_FAST_RELEASE_MS / speed)
    slow = _rising_follower(peak, sample_rate, TRANSIENT_SLOW_ATTACK_MS / speed,
                            TRANSIENT_SLOW_RELEASE_MS / speed)
    db = lambda v: 20.0 * np.log10(np.maximum(v, 1.0e-6))   # noqa: E731
    transientness = np.clip((db(fast) - db(slow) - TRANSIENT_DEADBAND_DB)
                            / (TRANSIENT_FULL_DB - TRANSIENT_DEADBAND_DB), 0.0, 1.0)
    gain_db = attack_db * transientness + sustain_db * (1.0 - transientness) + output_db
    gain = 10.0 ** (gain_db / 20.0)
    return x * (gain[:, np.newaxis] if x.ndim == 2 else gain)


def _duck_key_heard(session, key) -> bool:
    """Is a duck's trigger a stem this render can hear? Not a deleted (excluded) one - the DAW
    has no such track to key from, and the render still ducked under it (2026-10-17, the duck
    audit) - and not a DAW-only trigger (an instrument track: `Session.set_duck`'s
    `key_may_be_track`). The render and its fingerprint ask the same question."""
    rec = session.stems.get(key) if key else None
    return rec is not None and not rec.get("excluded")


def _key_audio(key_audio: np.ndarray, n: int, sample_rate: int,
               key_offset_s: float = 0.0) -> np.ndarray:
    """A trigger's audio (frames, channels) laid on the keyed stem's timeline, as `_key_level`
    lays its level: `key_offset_s` = the key stem's `capture_start_s` minus this one's."""
    key = np.asarray(key_audio, dtype=np.float64)
    if key.ndim == 1:
        key = key[:, np.newaxis]
    shift = int(round(float(key_offset_s) * sample_rate))
    if shift > 0:
        key = np.concatenate([np.zeros((shift, key.shape[1])), key])
    elif shift < 0:
        key = key[-shift:]
    out = np.zeros((n, key.shape[1]), dtype=np.float64)
    m = min(n, key.shape[0])
    out[:m] = key[:m]
    return out


def _duck(audio: np.ndarray, sample_rate: int, entry: dict, tempo=None,
          key_audio: Optional[np.ndarray] = None, key_offset_s: float = 0.0) -> np.ndarray:
    """Duck (2026-10-28): the law in `duck_law` - a drawn curve on the song's grid (SYNC) or
    restarted by each hit of the trigger track (TRACK; `key_audio` = the trigger rendered without
    its own keyed devices, None when no trigger is heard: the duck then rests at unity), scaled by
    the mix, the low band alone with BAND. The native runs the same law (`MonitorChain::applyDuck`).
    (Until 2026-10-28: full depth_db while the key was over threshold_db, attack / release one-poles.)"""
    from . import duck_law
    key = None if key_audio is None else _key_audio(key_audio, audio.shape[0], sample_rate,
                                                    key_offset_s)
    return duck_law.apply(audio, sample_rate, entry, tempo=tempo, key=key).astype(audio.dtype)


def _dynamic_eq(audio: np.ndarray, sample_rate: int, entry: dict) -> np.ndarray:
    """Dynamic EQ (Phase 38) on (channels, frames) float32: the law fitted to a reference EQ's dynamic
    band (`dyneq_law`, 2026-10-27; the native runs the same, `MonitorChain::applyDynamicEq`). The
    SIGN of gain_db (the RANGE) sets direction: negative = downward cut, positive = upward boost (the
    user co-edit can choose either; the agent stays a cut). 0 is off, bit for bit. (Until 2026-10-27:
    a ratio over the threshold, and the band's delta added back as x + band x (g - 1).)"""
    from .dyneq_law import apply
    return apply(audio, sample_rate, entry)


def _range_shape(n_frames: int, sample_rate: int, start_s: float,
                 end_s: float) -> np.ndarray:
    """0→1 envelope over [start, end] with RIDE_FADE_S raised-cosine edges
    (compressed to half the range when the range is shorter). Shared by gain
    rides (Phase 5) and ranged processors (Phase 15). Pure numpy —
    deterministic by construction."""
    shape = np.zeros(n_frames, dtype=np.float64)
    start = int(start_s * sample_rate)
    end = min(int(end_s * sample_rate), n_frames)
    if start >= n_frames or end <= start:
        return shape  # range entirely past the end of this stem's audio
    segment = np.ones(end - start, dtype=np.float64)
    fade = min(int(RIDE_FADE_S * sample_rate), (end - start) // 2)
    if fade > 0:
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade) / fade)  # 0 -> 1
        segment[:fade] = ramp
        segment[-fade:] = ramp[::-1]
    shape[start:end] = segment
    return shape


def _ride_envelope(n_frames: int, sample_rate: int, rides: list[dict]) -> np.ndarray:
    """Float64 gain envelope for the stem's rides: unity outside each range,
    10^(delta/20) inside (fades per _range_shape). Overlapping rides multiply."""
    envelope = np.ones(n_frames, dtype=np.float64)
    for ride in rides:
        gain = 10.0 ** (ride["delta_db"] / 20.0)
        envelope *= 1.0 + (gain - 1.0) * _range_shape(
            n_frames, sample_rate, ride["start_s"], ride["end_s"])
    return envelope


def _range_blend(dry: np.ndarray, wet: np.ndarray, sample_rate: int,
                 range_s) -> np.ndarray:
    """Phase 15 ranged processing on (channels, frames) float32: the wet
    signal inside [start, end] (raised-cosine crossfade at the edges), the
    dry signal everywhere else. range_s None = fully wet (whole-file)."""
    if range_s is None:
        return wet
    envelope = _range_shape(dry.shape[1], sample_rate,
                            range_s[0], range_s[1]).astype(np.float32)
    return dry + (wet - dry) * envelope[np.newaxis, :]


def _ms_apply(audio: np.ndarray, channel: str, fn) -> np.ndarray:
    """Apply `fn` ((channels,frames)->(channels,frames)) to ONLY the mid or side of a
    stereo signal, then recombine (Phase 41 M/S mode for eq/comp). `channel`=="stereo"
    or a mono input runs `fn` directly. Encode/decode is linear + exact, so an M/S eq
    stays as faithful as the stereo path; the target channel is processed as a 2-channel
    mono-dup so a stereo-linked `fn` (the comp) behaves."""
    if channel == "stereo" or audio.shape[0] != 2:
        return fn(audio)
    mid = (audio[0] + audio[1]) * 0.5
    side = (audio[0] - audio[1]) * 0.5
    target = mid if channel == "mid" else side
    proc = fn(np.ascontiguousarray(np.stack([target, target])))[0]
    if channel == "mid":
        mid = proc
    else:
        side = proc
    return np.ascontiguousarray(np.stack([mid + side, mid - side]))


def _channel_apply(audio: np.ndarray, channel: str, fn) -> np.ndarray:
    """Dispatch `fn` ((channels,frames)->(channels,frames)) by `channel` (Phase 46
    adds left/right to the Phase 41 stereo/mid/side set, used by the eq call
    sites). stereo/mid/side delegate to `_ms_apply` (linear M/S encode/decode);
    left/right process ONLY that stereo channel (index 0 = left, 1 = right)
    in place — no encode/decode, so the other channel stays bit-identical. Mono
    audio (or any non-stereo channel count) degrades left/right to processing
    the (only) channel directly, same as `_ms_apply`'s mono no-op."""
    if channel in ("left", "right"):
        if audio.shape[0] != 2:
            return fn(audio)
        idx = 0 if channel == "left" else 1
        out = np.array(audio, copy=True)
        out[idx:idx + 1] = fn(audio[idx:idx + 1])
        return out
    return _ms_apply(audio, channel, fn)


def _width(audio: np.ndarray, width: float, stereoize: bool = False,
           stereoize_ms: float = 6.0, sample_rate: float = 48000.0) -> np.ndarray:
    """The Imager: mid/side width scaling on stereo, plus STEREOIZE above unity (side made from
    the mid through `stereoize_law`'s velvet taps - the mid is never touched). No-op on mono (no
    side signal). Held to 0..2, as the native holds it (the stereo width audit, SW7: a stored 5
    played 5, a -1 swapped L and R)."""
    width = min(2.0, max(0.0, float(width))) if math.isfinite(float(width)) else 1.0
    if width == 1.0 or audio.shape[1] != 2:
        return audio
    if stereoize and width > 1.0:
        return stereoize_law.apply(audio, sample_rate, width, True, stereoize_ms)
    mid = (audio[:, 0] + audio[:, 1]) / 2.0
    side = (audio[:, 0] - audio[:, 1]) / 2.0 * width
    return np.stack([mid + side, mid - side], axis=1)


def _width_entry(audio: np.ndarray, entry: dict, sample_rate: float) -> np.ndarray:
    """A `width` chain entry (a bus strip's, the master's): its width and its stereoize."""
    return _width(audio, entry["width"], bool(entry.get("stereoize", False)),
                  float(entry.get("stereoize_ms", 6.0)), sample_rate)


def match_eq_fir(band_gains_db: dict, sample_rate: int) -> np.ndarray:
    """The match-EQ's linear-phase FIR taps: frequency-sampled at the band geometric
    centers. Extracted from `_match_eq` (2026-09-08) so `monitor_chain` can fit the live
    monitor's peak bank to THIS response rather than to the raw band gains — the two were
    up to 8.4 dB apart, and in the DAW the export is a bounce of the monitor, so that gap
    was in the deliverable (device audit 04-F4). One source of truth for the target."""
    from scipy.signal import firwin2

    nyquist = sample_rate / 2.0
    gains_db = {band: band_gains_db.get(band, 0.0) for band in BAND_LABELS}
    points, amplitudes = [], []
    for band, lo_hz, hi_hz in zip(BAND_LABELS, BAND_EDGES_HZ[:-1], BAND_EDGES_HZ[1:]):
        center_hz = math.sqrt(lo_hz * hi_hz)
        if center_hz >= nyquist:
            break
        points.append(center_hz / nyquist)
        amplitudes.append(10.0 ** (gains_db[band] / 20.0))
    freqs = [0.0] + points + [1.0]
    amps = [amplitudes[0]] + amplitudes + [amplitudes[-1]]
    return firwin2(MATCH_EQ_TAPS, freqs, amps)


def _match_eq(audio: np.ndarray, sample_rate: int, entry: dict) -> np.ndarray:
    """Match-EQ (Phase 15): one linear-phase FIR built from the entry's
    per-band corrections — frequency-sampled at the band geometric centers,
    delay-compensated so stems stay time-aligned. (channels, frames) float32
    in/out; coefficients depend only on the entry + sample rate, so renders
    are deterministic from state alone (audio-dsp skill)."""
    from scipy.signal import fftconvolve

    fir = match_eq_fir(entry["band_gains_db"], sample_rate)

    delay = (MATCH_EQ_TAPS - 1) // 2
    n_frames = audio.shape[1]
    convolved = fftconvolve(audio.astype(np.float64), fir[np.newaxis, :],
                            axes=1)
    return convolved[:, delay:delay + n_frames].astype(np.float32)


def _rising_follower(x: np.ndarray, sample_rate: float, attack_ms: float,
                     release_ms: float) -> np.ndarray:
    """`env = c·env + (1-c)·x`, c the attack coefficient where x is ABOVE the envelope and the
    release one where it is not, from 0 - the native's linear level follower (mbBandGainUp), in
    float64. `_gr_ballistics_db` runs the same recursion in float32 around an offset of 1, which
    leaves a quiet level (1e-4) a few significant digits: 0.5 % off exactly where an upward band
    acts. Here the branch is taken from that estimate, the recursion is solved exactly with it
    (a time-varying one-pole in closed form, block by block), and the branch is re-taken from
    the exact envelope until it stands."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n == 0:
        return x.copy()
    a_att = math.exp(-1000.0 / (sample_rate * max(attack_ms, 1e-6)))
    a_rel = math.exp(-1000.0 / (sample_rate * max(release_ms, 1e-6)))
    env = -_gr_ballistics_db(-x, sample_rate, attack_ms, release_ms)   # the estimate
    slowest = max(min(a_att, a_rel), 1e-300)
    block = int(max(16, min(4096, 600.0 / max(-math.log(slowest), 1e-9))))   # cumprod stays > 1e-260
    for _ in range(8):
        prev = np.concatenate([[0.0], env[:-1]])
        c = np.where(x > prev, a_att, a_rel)
        d = (1.0 - c) * x
        exact = np.empty(n)
        start = 0.0
        for b0 in range(0, n, block):
            cb, db = c[b0:b0 + block], d[b0:b0 + block]
            pc = np.cumprod(cb)
            exact[b0:b0 + block] = pc * (start + np.cumsum(db / pc))
            start = exact[b0 + len(cb) - 1]
        if np.array_equal(np.where(x > np.concatenate([[0.0], exact[:-1]]), a_att, a_rel), c):
            return exact
        env = exact
    return env


def _lr4_three_way(audio: np.ndarray, sample_rate: int, lo_hz: float, hi_hz: float) -> tuple:
    """The REAL 3-way Linkwitz-Riley split the DAW runs (the imager's and, since 2026-10-17, the
    multiband compressor's), on (channels, frames): low = AP@hi(LP@lo x), rest = HP@lo x,
    mid = LP@hi rest, high = HP@hi rest - each leg its Butterworth section (q=0, the native's
    sentinel) run twice; the low through the hi pair summed is the allpass that keeps the three
    magnitude-flat at unity. Real bands, never a complement."""
    from scipy.signal import lfilter

    def lr4(x, shape, f):
        b, a = biquad_coeffs(shape, sample_rate, f, 0.0, 0.0)
        return lfilter(b, a, lfilter(b, a, x, axis=-1), axis=-1)
    x = np.asarray(audio, dtype=np.float64)
    low_raw = lr4(x, "lpf2", lo_hz)
    rest = lr4(x, "hpf2", lo_hz)
    low = lr4(low_raw, "lpf2", hi_hz) + lr4(low_raw, "hpf2", hi_hz)
    return low, lr4(rest, "lpf2", hi_hz), lr4(rest, "hpf2", hi_hz)


def _multiband_comp(audio: np.ndarray, sample_rate: int, entry: dict,
                    telemetry: Optional[list] = None) -> np.ndarray:
    """Three-band compressor (Phase 15) - since 2026-10-17 the DAW's law
    (`MonitorChain::applyMultibandComp`): the real 3-way LR4 split (`_lr4_three_way`),
    each band's linked peak through the native down (`_native_comp`) or up law,
    per-band makeup, summed back. (channels, frames) float32 in/out. With
    `telemetry` (Phase 20) the pre-makeup band sum is also accumulated so
    GR is measured as reduction, per-band makeup excluded. Crossovers default
    to MB_CROSSOVERS_HZ but are user-movable per entry (the native editor drags
    them to widen / narrow each band) — clamped, and ordered so mid stays
    non-empty."""
    default_lo, default_hi = MB_CROSSOVERS_HZ
    lo_x = float(np.clip(entry.get("crossover_lo_hz", default_lo), 30.0, 4000.0))
    hi_x = float(np.clip(entry.get("crossover_hi_hz", default_hi), 200.0, 16000.0))
    if hi_x <= lo_x:                       # keep the mid band non-empty (the UI clamps too)
        hi_x = lo_x * 1.5
    out = np.zeros(audio.shape, dtype=np.float64)
    pre_makeup = np.zeros(audio.shape, dtype=np.float64) \
        if telemetry is not None else None
    # THE DAW'S SPLIT (2026-10-17, the multiband audit): the real 3-way LR4 the native runs
    # (`MonitorChain::applyMultibandComp`) - it was a tapered FFT partition here against a
    # complement there, and the export's lows got LOUDER where this render's got quieter.
    for name, band64 in zip(("low", "mid", "high"),
                            _lr4_three_way(audio, sample_rate, lo_x, hi_x)):
        params = entry[name]
        band = np.ascontiguousarray(band64.astype(np.float32))
        if params.get("up"):
            # Upward (`mbBandGainUp`, transcribed): the band's linked per-sample peak smoothed
            # LINEARLY (attack when it rises, release when it falls - from rest, 0), and below the
            # threshold a boost of under·(1 - 1/ratio), capped at MB_MAX_BOOST_DB. It read a 5 ms
            # block RMS smoothed in dB: 3.8 dB louder than the DAW on drums (the multiband audit).
            peak = np.max(np.abs(band.astype(np.float64)), axis=0)
            env = _rising_follower(peak, sample_rate, float(entry["attack_ms"]),
                                   float(entry["release_ms"]))
            level_db = 20.0 * np.log10(np.maximum(env, 1.0e-6))
            under = float(params["threshold_db"]) - level_db
            boost_db = np.where(under > 0.0,
                                np.minimum(MB_MAX_BOOST_DB,
                                           under * (1.0 - 1.0 / max(float(params["ratio"]), 1.0))),
                                0.0)
            compressed = (band * (10.0 ** (boost_db / 20.0))[np.newaxis, :]).astype(np.float32)
        else:
            # The native band law (`mbBandGain`): a linked peak detector on the band,
            # ballistics on the reduction in dB — see `_native_comp`.
            compressed = _native_comp(band, sample_rate, params["threshold_db"],
                                      params["ratio"], entry["attack_ms"],
                                      entry["release_ms"])
        mk = float(params.get("makeup_db", 0.0) or 0.0)
        boosted = compressed * np.float32(10.0 ** (mk / 20.0)) if mk else compressed
        out += boosted.astype(np.float64)
        if pre_makeup is not None:
            pre_makeup += compressed.astype(np.float64)
    if telemetry is not None:
        stats = _gr_telemetry("mb_comp", audio.T, pre_makeup.T, sample_rate)
        if stats is not None:
            telemetry.append(stats)
    return out.astype(np.float32)


def _ott(audio: np.ndarray, sample_rate: int, entry: dict,
         telemetry: Optional[list] = None) -> np.ndarray:
    """OTT, three-band upward + downward compression - since 2026-10-28 the law fitted to a reference
    OTT (`ott_law`, which says how it was measured; `MonitorChain::applyOtt` plays the same law): the
    DAW's LR4 split at 88.3 Hz / 2.5 kHz, each band's linked RMS detector, a fixed makeup, an upward
    side below one threshold and a downward side above another, scaled by `depth`. `time` scales the
    release; `in_gain_db` / `out_gain_db` bracket the device; per-band `gain_db` is a post gain
    (makeup - excluded from telemetry). (channels, frames) float32 in/out, deterministic. (Until
    2026-10-28: one fixed -30 dB target, a peak follower, 12 / 120 ms, crossovers at 150 Hz / 2 kHz.)"""
    from .ott_law import process as ott_process
    pre = [] if telemetry is not None else None
    out = ott_process(audio.astype(np.float64), sample_rate, entry, telemetry_pre=pre)
    if telemetry is not None:
        from .ott_law import values as ott_values
        g = ott_values(entry)["in_gain_db"]
        x = audio.astype(np.float64) * (10.0 ** (g / 20.0) if g else 1.0)
        stats = _gr_telemetry("ott", x.T, pre[0].T, sample_rate)
        if stats is not None:
            telemetry.append(stats)
    return out.astype(np.float32)


def _one_pole_lp(audio: np.ndarray, sample_rate: int, freq_hz: float) -> np.ndarray:
    """First-order low-pass on (frames, channels) float64 — the SAME JUCE/Pedalboard
    one-pole the de-ess split uses (biquad_coeffs 'lpf'), so the native transcription
    is coefficient-identical. LP + (x − LP) is exactly complementary."""
    from scipy.signal import lfilter
    b, a = biquad_coeffs("lpf", sample_rate, freq_hz, 0.707, 0.0)
    return lfilter(b, a, audio, axis=0)


def _utility(audio: np.ndarray, sample_rate: int, entry: dict) -> np.ndarray:
    """Ableton-style Utility (device freedom; v2 2026-07-05 = Live parity): DC filter -> gain trim
    -> per-channel polarity -> stereo width (mono = width 0) -> bass mono -> balance, on (frames,
    channels) float64. ONE law with `reference_monitor._utility` (this calls it) and
    MonitorChain::applyUtility: the values held by `reference_monitor.utility_values`, Bass Mono a
    Linkwitz-Riley 8th-order split at bass_mono_hz (the Utility audit, 2026-10-21 - James picked
    the band width's law; it was a 6 dB/oct one-pole). All-default is a pass-through."""
    from .reference_monitor import _utility as one_law
    return one_law(audio, sample_rate, entry)


def _tone_filter(audio: np.ndarray, sample_rate: int, tone: float) -> np.ndarray:
    """One-pole wet-return tone control on (frames, channels): tone=0 is dark,
    tone=1 is mostly open. Deterministic, no lookahead or state across calls."""
    if audio.shape[0] == 0:
        return audio
    from scipy.signal import lfilter
    cutoff_hz = 1200.0 + np.clip(tone, 0.0, 1.0) * 16800.0
    alpha = float(np.exp(-2.0 * np.pi * cutoff_hz / sample_rate))
    feed = 1.0 - alpha
    # y[n] = feed·x[n] + alpha·y[n-1], seeded so y[0] == x[0] (the return opens
    # without a click). A 2-tap IIR is exactly what lfilter is for — O(n) and
    # no Python loop, unlike the long delay-line combs above.
    x = audio.astype(np.float64)
    out, _ = lfilter([feed], [1.0, -alpha], x, axis=0, zi=alpha * x[:1])
    return out


# The delay is mix_engine.delay_law (2026-10-27): fitted to a reference delay - every pass through a
# 200 Hz high-pass and a Linkwitz-Riley low-pass at the tone's corner, a slow wow on the time, the right
# side 1 % longer. It replaced a one-pole tone and a 100 Hz low cut in the loop (2026-10-25), which was a
# tone filter after a comb before that. The native chain runs the same law (DelayLaw.h, generated), held
# together by test_delay_law.py and --delay-law.
def _return_delay(audio: np.ndarray, sample_rate: int, entry: dict) -> np.ndarray:
    """Wet-only feedback delay - delay_law.apply. One law with MonitorChain::applySendReturns
    (reference_monitor calls this)."""
    from . import delay_law
    return delay_law.apply(audio, sample_rate, entry)


# The algorithmic reverb is mix_engine.reverb_law (2026-10-27): a sixteen-line modulated feedback-delay
# network fitted to a reference reverb - its decay by band, its build, its colour and its brightness.
# It replaced an eight-line network (2026-10-24) that replaced four parallel combs. The native chain
# runs the same law (ReverbLaw.h, generated), held together by test_reverb_law.py and --reverb-law.
def _return_reverb(audio: np.ndarray, sample_rate: int, entry: dict) -> np.ndarray:
    """Wet-only reverb on (frames, channels) - reverb_law.apply. One law with MonitorChain::applySendReturns
    (reference_monitor calls this)."""
    from . import reverb_law
    return reverb_law.apply(audio, sample_rate, entry)


# Procedural impulse responses for the convolution reverb (Tier-1). A deterministic decaying-noise
# IR per space, made the same way by all three engines that make it - here, the reference monitor
# and the native chain (MonitorChain::proceduralIr) - sample for sample, so the export oracle can
# compare the reverb itself (2026-10-19, the IR reverb audit: the noise was numpy's normal noise
# here and juce::Random's uniform noise there, two different filters up to 3.9 dB apart per band).
_IR_SEEDS = {"room": 0x12001, "hall": 0x12002, "plate": 0x12003}
_IR_DECAY_SCALE = {"room": 0.55, "hall": 1.0, "plate": 0.75}   # per-space RT60 multiplier
# A room's three early reflections: (ms, gain). Each spike's ENERGY stands this much over one
# millisecond of the tail around it (gain^2 x lift^2 against sr/1000 samples of unit-variance
# noise) - +3.6 / +1.1 / -1.1 dB, which is what the face draws. They were one noise sample's size
# (0.6 / 0.45 / 0.35 against a unit sigma): nothing anyone could hear.
_IR_EARLY = ((7.0, 0.6), (11.0, 0.45), (19.0, 0.35))
_IR_EARLY_LIFT = 10.0 ** (8.0 / 20.0)
# Output level after unit-L2 (energy) normalization of the NOISE, before the tone filter. It MUST
# be L2, not RMS: convolution gain scales with the IR's L2 norm, and L2 = rms * sqrt(n) where n =
# rt60 * sample_rate. And it must come BEFORE the tone filter: normalised after it, every bit of
# top the filter took away was put back across the whole band, so TONE was a level knob (Tone 0
# lifted the lows by 10 dB) and the level moved with the sample rate (2026-10-19). The scale keeps
# the default space (Tone 0.5) as loud as it was at 48 kHz, music-weighted.
_IR_WET_SCALE = 0.34


def _ir_noise(seed: int, n: int) -> np.ndarray:
    """Unit-variance uniform noise from a counter hash (lowbias32 of the sample index plus the
    seed) - exact in uint32 arithmetic, so MonitorChain::irNoise computes the same numbers."""
    i = np.arange(n, dtype=np.uint32)
    with np.errstate(over="ignore"):
        h = i * np.uint32(0x9E3779B1) + np.uint32(seed)
        h ^= h >> np.uint32(16)
        h *= np.uint32(0x7FEB352D)
        h ^= h >> np.uint32(15)
        h *= np.uint32(0x846CA68B)
        h ^= h >> np.uint32(16)
    u = (h >> np.uint32(8)).astype(np.float64) * (1.0 / 16777216.0)
    return (2.0 * u - 1.0) * np.sqrt(3.0)


def _ir_params(entry: dict) -> tuple[str, float, float, float]:
    """The four IR parameters held to the table (decay 0.2..8 s, tone 0..1, pre 0..250 ms) - every
    door holds them there, and the DAW cannot play anything else."""
    mode = str(entry.get("ir", "hall"))
    if mode not in _IR_SEEDS:
        mode = "hall"
    decay = float(np.clip(float(entry.get("decay_s", 2.0)), 0.2, 8.0))
    tone = float(np.clip(float(entry.get("tone", 0.5)), 0.0, 1.0))
    pre = float(np.clip(float(entry.get("pre_delay_ms", 0.0)), 0.0, 250.0))
    return mode, decay, tone, pre


def _round_half_up(x: float) -> int:
    return int(np.floor(x + 0.5))   # = std::llround for x >= 0


def _procedural_ir(mode: str, decay_s: float, tone: float, sample_rate: int) -> np.ndarray:
    """A deterministic decaying-noise impulse response (mono). `mode` shapes the space
    (room = three discrete early reflections; hall = a 20 ms onset build-up; plate = dense from
    sample 0); `decay_s` x the space's scale is the RT60; `tone` is a one-pole shelf on the whole
    tail (0 dark .. 1 bright) - the highs are quieter from the first millisecond, they do not die
    faster, and the lows do not move."""
    mode, decay_s, tone, _ = _ir_params({"ir": mode, "decay_s": decay_s, "tone": tone})
    rt60 = max(0.2, decay_s * _IR_DECAY_SCALE[mode])
    n = max(8, _round_half_up(min(rt60, 8.0) * sample_rate))
    t = np.arange(n) / sample_rate
    ir = _ir_noise(_IR_SEEDS[mode], n) * (10.0 ** (-3.0 * t / rt60))   # -60 dB at rt60
    if mode == "room":
        lift = _IR_EARLY_LIFT * np.sqrt(sample_rate / 1000.0)
        for d_ms, g in _IR_EARLY:
            k = _round_half_up(d_ms * sample_rate / 1000.0)
            if k < n:
                ir[k] += g * lift
    elif mode == "hall":
        ramp = _round_half_up(0.02 * sample_rate)   # 20 ms onset build-up (a soft attack)
        ir[:ramp] *= np.arange(ramp) / ramp
    l2 = float(np.sqrt(np.sum(ir ** 2)))   # NOT rms - see _IR_WET_SCALE
    if l2 > 1e-12:
        ir = ir / l2 * _IR_WET_SCALE
    return _tone_filter(ir[:, np.newaxis], sample_rate, tone)[:, 0]   # the shelf, last


def _convolve_ir(audio: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """Convolve (frames,) or (frames, channels) audio with a mono IR along time, trimmed back to
    the input length (the tail past the buffer drops, like the comb reverb)."""
    from scipy.signal import fftconvolve
    n = audio.shape[0]
    if audio.ndim == 1:
        return fftconvolve(audio.astype(np.float64), ir)[:n]
    out = np.zeros_like(audio, dtype=np.float64)
    for c in range(audio.shape[1]):
        out[:, c] = fftconvolve(audio[:, c].astype(np.float64), ir)[:n]
    return out


def _return_convolution(audio: np.ndarray, sample_rate: int, entry: dict) -> np.ndarray:
    """Wet-only convolution (IR) reverb for an agent-owned return (Tier-1): the summed sends,
    pre-delayed, convolved with a procedural room/hall/plate impulse. The native engine mirrors
    this with juce::dsp::Convolution on the same IR."""
    n_frames = audio.shape[0]
    if n_frames == 0:
        return np.zeros_like(audio, dtype=np.float64)
    mode, decay_s, tone, pre_ms = _ir_params(entry)
    pre_delay = _round_half_up(pre_ms * sample_rate / 1000.0)   # rounded as the native rounds it
    fed = np.zeros_like(audio, dtype=np.float64)
    if pre_delay < n_frames:
        fed[pre_delay:] = audio[: n_frames - pre_delay]
    return _convolve_ir(fed, _procedural_ir(mode, decay_s, tone, sample_rate))


_RETURN_WET_KINDS = ("return_delay", "return_reverb", "return_convolution")


def _wet_device(dry: np.ndarray, sample_rate: int, entry: dict) -> np.ndarray:
    """One SPACE device — reverb / delay / IR reverb — on (frames, channels) float64.

    The same processor serves both placements (2026-07-20, "no restrictions"): an FX
    RETURN and a plain INSERT on a track / bus / master. `mix` is the dry/wet balance
    and is what separates them — 1.0 (the default, and what every return uses, since a
    return's dry already lives on the source tracks) returns the wet array untouched, so
    every pre-2026-07-20 session renders BIT-IDENTICALLY; below 1.0 it crossfades, which
    is what makes the device usable as an insert the way Live's Reverb/Delay are.
    Mirrored natively by MonitorChain::applySendReturns."""
    kind = entry["type"]
    if kind == "return_delay":
        wet = _return_delay(dry, sample_rate, entry)
    elif kind == "return_reverb":
        wet = _return_reverb(dry, sample_rate, entry)
    else:
        wet = _return_convolution(dry, sample_rate, entry)
    return _comp_mix(dry.astype(np.float64), wet, float(entry.get("mix", 1.0)))


def _apply_return_chain(audio: np.ndarray, sample_rate: int,
                        chain: list[dict], tempo=None) -> np.ndarray:
    """Render one return bus from its summed sends through a FREE device chain (Ableton-style — a
    return is a bus fed by sends). Devices apply SERIALLY in list order: the normal effects (eq/comp/
    sat/gate/…) reuse the bus per-entry applier; reverb/delay/convolution are WET-REPLACING serial
    steps (on a return the whole signal is the effect — the dry already lives on the source tracks).
    A return with no wet effect (e.g. a lone compressor → parallel/NY comp) passes its processed
    input. Order is honoured both ways — eq→reverb shapes the input, reverb→eq tames the tail."""
    wet = audio.astype(np.float64)
    for entry in chain:
        if not entry.get("enabled", True):   # Phase 40 per-device bypass on returns too
            continue
        kind = entry["type"]
        if kind in _RETURN_WET_KINDS:
            wet = _wet_device(wet, sample_rate, entry)   # mix defaults 1.0 → wet-replacing, as before
        else:                                   # gain + any normal device — shared bus applier (DRY)
            wet = _apply_bus_chain(wet, sample_rate, [entry], tempo=tempo)
    return wet


def _comp_mix(dry: np.ndarray, wet: np.ndarray, mix: float) -> np.ndarray:
    """Parallel-compression blend (Phase 23): mix·wet + (1−mix)·dry,
    applied post-makeup, before any time-range crossfade. mix=1 returns the
    wet array untouched — bit-identical to the pre-Phase-23 path — and
    mix=0 returns the dry array bit-exactly. Blend math stays in the
    inputs' dtype (float32 in the stem/bus stage, float64 on the master)."""
    if mix >= 1.0:
        return wet
    if mix <= 0.0:
        return dry
    if dry.dtype == np.float32:
        blend = np.float32(mix)
        return wet * blend + dry * (np.float32(1.0) - blend)
    return wet * mix + dry * (1.0 - mix)


def _ms_apply_both(audio: np.ndarray, fn_mid, fn_side) -> np.ndarray:
    """True M/S (native parity): process the mid with `fn_mid` and the side with
    `fn_side` INDEPENDENTLY, then recombine. Same exact M/S encode/decode as
    _ms_apply; mono falls back to fn_mid (no side)."""
    if audio.shape[0] != 2:
        return fn_mid(audio)
    mid = (audio[0] + audio[1]) * 0.5
    side = (audio[0] - audio[1]) * 0.5
    mid = fn_mid(np.ascontiguousarray(np.stack([mid, mid])))[0]
    side = fn_side(np.ascontiguousarray(np.stack([side, side])))[0]
    return np.ascontiguousarray(np.stack([mid + side, mid - side]))


def _apply_comp(audio: np.ndarray, sample_rate: int, entry: dict) -> np.ndarray:
    """Compressor on (channels, frames) float32 — stereo / single-channel M/S
    (Phase 41) via _ms_apply, OR true M/S (independent mid + side param sets) when
    the entry carries `mid`/`side` sub-objects. The law is the native one
    (`_character_comp`, the entry's `style`): the M/S paths hand it the target as two identical channels,
    so its linked detector reads |mid| or |side| exactly as `processOneComp` does."""
    def comp_fn(params):
        def _fn(a):
            wet = _character_comp(a, sample_rate, params["threshold_db"], params["ratio"],
                                  params["attack_ms"], params["release_ms"],
                                  params.get("makeup_db") or 0.0, comp_style(params),
                                  float(params.get("knee_db") or 0.0))
            return _comp_mix(a, wet, params.get("mix", 1.0))
        return _fn
    channel = entry.get("channel", "stereo")
    mid_s, side_s = entry.get("mid"), entry.get("side")
    if channel in ("mid", "side") and (isinstance(mid_s, dict) or isinstance(side_s, dict)):
        # One rule for a half that is not stored, in both engines (MonitorChain's comp parse): the
        # knobs as they stand, and the half the comp is NOT set to passes through (1:1, no makeup)
        # - what a flip seeds. A
        # lone `side` (a knob's first turn on a side-only comp) used to be ignored here.
        def half(name: str, sub) -> dict:
            if isinstance(sub, dict):
                return {**entry, **sub}       # nested params, flat entry as the fallback
            return {**entry} if name == channel else {**entry, "ratio": 1.0, "makeup_db": 0.0}
        return _ms_apply_both(audio, comp_fn(half("mid", mid_s)), comp_fn(half("side", side_s)))
    return _ms_apply(audio, channel, comp_fn(entry))


def _spectral_denoise(audio: np.ndarray, sample_rate: int, entry: dict) -> np.ndarray:
    """Phase 29 spectral-gate de-noise / de-bleed for separated stems, on
    (channels, frames) float32 (the float32-stage convention). Per
    (channel, frequency) bin: estimate a noise floor as a low percentile of the
    bin's magnitude over time, then attenuate bins sitting within
    `sensitivity_db` of that floor by up to `reduction_db` (smoothstep
    transition), phase taken from the input. Deterministic — percentile +
    scipy STFT/ISTFT, no randomness (audio-dsp skill: no noise/dither in a
    DSP path)."""
    from scipy.signal import istft, stft
    reduction_db = float(entry["reduction_db"])
    sensitivity_db = float(entry.get("sensitivity_db", 6.0))
    floor_gain = 10.0 ** (-reduction_db / 20.0)
    x = audio.astype(np.float64)
    n_frames = x.shape[1]
    nperseg = min(2048, n_frames) if n_frames else 2048
    noverlap = nperseg * 3 // 4  # 75% hann satisfies COLA -> clean reconstruction
    _, _, Z = stft(x, fs=sample_rate, nperseg=nperseg, noverlap=noverlap,
                   axis=-1, boundary="zeros", padded=True)  # (channels, bins, times)
    mag = np.abs(Z)
    noise = np.maximum(np.percentile(mag, 25.0, axis=-1, keepdims=True), 1e-9)
    # How far each T-F bin sits above its own quiet-floor estimate.
    excess_db = 20.0 * np.log10(np.maximum(mag, 1e-12) / noise)
    # Keep bins more than `sensitivity_db` above that floor; gate the rest toward
    # floor_gain through a 6 dB smoothstep transition. Stationary broadband noise
    # sits only a few dB above its temporal 25th percentile, so a few dB of
    # sensitivity floors it, while loud, time-varying content stands well clear
    # and is kept. (Known soft spot: a sustained pure tone that never gaps reads
    # as its own floor — cap reduction_db and let the artifact metrics gate where
    # this runs; see the mix-analysis skill failure-modes.)
    transition_db = 6.0
    u = np.clip((excess_db - (sensitivity_db - transition_db / 2.0)) / transition_db,
                0.0, 1.0)
    keep = u * u * (3.0 - 2.0 * u)  # smoothstep
    gain = floor_gain + (1.0 - floor_gain) * keep
    _, y = istft(Z * gain, fs=sample_rate, nperseg=nperseg, noverlap=noverlap,
                 boundary=True, time_axis=-1, freq_axis=-2)
    y = np.atleast_2d(y)
    if y.shape[-1] >= n_frames:
        y = y[..., :n_frames]
    else:
        y = np.pad(y, [(0, 0)] * (y.ndim - 1) + [(0, n_frames - y.shape[-1])])
    return np.ascontiguousarray(y).astype(np.float32)


def _pitch_shift(audio: np.ndarray, sample_rate: int, cents: float) -> np.ndarray:
    """Phase 31 uniform tuning correction: shift the whole stem by `cents`
    (±100) via Pedalboard's Rubber Band PitchShift, on (channels, frames)
    float32. Deterministic — the board is built fresh per render like every
    other entry, so no internal state carries; verified bit-identical across
    renders (audio-dsp skill: no randomness in a DSP path). Rubber Band can
    return a frame count off by its block size, so truncate / zero-pad back to
    the input length, keeping summing + undo bit-identical (the _spectral_denoise
    length guard)."""
    n_frames = audio.shape[-1]
    y = Pedalboard([PitchShift(semitones=cents / 100.0)])(audio, sample_rate)
    if y.shape[-1] >= n_frames:
        y = y[..., :n_frames]
    else:
        y = np.pad(y, [(0, 0)] * (y.ndim - 1) + [(0, n_frames - y.shape[-1])])
    return np.ascontiguousarray(y).astype(np.float32)


def _stem_file(session: Session, stem_id: str,
               _cache: dict) -> tuple[np.ndarray, int, bool]:
    """A stem's FILE as its track plays it - read, mended, two channels, at the project's
    rate, placed where its clips play - before anything is fed into the track or runs on it.
    `(audio float32 (frames, 2), sample_rate, mended)`: `mended` is "this read has something
    to say about its file" (a non-finite sample dropped, a short file padded - reported
    through `_cache`). `render_stem`'s first step, and what `analyze_phase` compares a
    channel with to say whether its chain or its file is the cause (2026-10-13)."""
    stem = session.stems[stem_id]
    audio, sample_rate = sf.read(session.project_dir / stem["path"],
                                 dtype="float32", always_2d=True)
    # A non-finite sample in the FILE is dropped, as the DAW's node drops one coming in
    # (reader E5·R8; batch M4 natively). A NaN used to make every render of the project raise
    # without naming the file, and an inf went through to the master's metrics and, by way of
    # a measured delta, into the cross-project calibration store. The render says which stem
    # it mended (render_master: report["non_finite_stems"]); such a render is not cached (nor
    # is a padded one, below: `mended` is "this render has something to say about its file"), so
    # every later one says it again.
    mended = False
    if not np.isfinite(audio).all():
        bad = ~np.isfinite(audio)
        audio[bad] = 0.0
        _cache.setdefault(_NON_FINITE, {})[stem_id] = int(bad.sum())
        mended = True
    # A mono file plays from BOTH speakers (reader E4·R4): the DAW's node runs every track at
    # two channels, so side = 0.5 (L - R) is silence and left / right are real channels. Read
    # as ONE channel, every channel-scoped device fell through and did something else - a side
    # comp compressed the whole stem (+8.8 dB by its make-up), `invert_left` did nothing where
    # the DAW plays the stem anti-phase, a reverb came back mono. Everything downstream is now
    # at least two channels, the master of an all-mono project included.
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        # A surround file on a stereo track (reader E5·R10): the first two channels - the
        # front pair of every WAV layout - are what the track's two-channel node is handed.
        # Passed through whole, a 5.1 stem made every readout of the project raise
        # ("Audio must have five channels or less") until the track was deleted.
        audio = np.ascontiguousarray(audio[:, :2])
    target_rate = session.project_sample_rate()
    if sample_rate != target_rate:
        # Phase 15: mixed-SR projects resample up to the project rate at
        # read time (resample_poly — deterministic, never downsamples).
        from scipy.signal import resample_poly
        divisor = math.gcd(target_rate, sample_rate)
        audio = resample_poly(audio.astype(np.float64),
                              target_rate // divisor,
                              sample_rate // divisor,
                              axis=0).astype(np.float32)
        sample_rate = target_rate
    # A file of a few samples (reader D4·R13) is padded with silence to STEM_MIN_FRAMES: two
    # frames by two channels is an array the float32 stages cannot tell channels from frames
    # in, and one such file in the folder made every render of the project raise. The render
    # names it (render_master: report["short_stems"]); `import_stem` refuses one outright.
    # Like a mended stem it is kept OUT of the persistent cache (pre-merge review, 2026-10-07):
    # the name is written here, where the file is read, and a cached render is never read
    # here again - with the gateway's cache on, only the first render of a project said it.
    if audio.shape[0] < STEM_MIN_FRAMES:
        _cache.setdefault(_TOO_SHORT, {})[stem_id] = int(audio.shape[0])
        mended = True
        audio = np.concatenate([audio, np.zeros((STEM_MIN_FRAMES - audio.shape[0],
                                                 audio.shape[1]), dtype=audio.dtype)])
    # The file as the track's ONE clip plays it (D4·R8): before anything is fed into the track
    # or runs on it, so a ranged move, a ride and a feeder all meet the audio on the timeline.
    audio = _placed(audio, sample_rate, stem.get("clip"), stem.get("clips"))
    return audio, sample_rate, mended


def render_stem(session: Session, stem_id: str, exclude_ducks: bool = False,
                _cache: Optional[dict] = None,
                telemetry: Optional[list] = None,
                _pre_pan: bool = False) -> tuple[np.ndarray, int]:
    """Render one stem through its applied chain.

    Returns (audio float64, shape (frames, channels), sample_rate).
    Processor order is exactly ``session.effective_chain`` list order. Processors
    backed by Pedalboard use channel-major float32 at their boundary; numpy space,
    nonlinear, dynamics and spatial devices use frame-major float64. Switching
    representations between adjacent entries is exact for float32 values and is
    what lets a wet device genuinely sit before or after an EQ. Ranged float32
    entries crossfade in via ``_range_blend``. Gain rides, pan and channel-strip
    width remain trailing scalars rather than chain inserts.
    `exclude_ducks=True` renders the stem without its duck entries — used
    for duck key signals, which breaks key cycles by design.
    `_cache` memoizes per (stem_id, exclude_ducks) within one render call
    (Phase 10): duck keys shared by several stems render once. Callers must
    not mutate returned audio (render_master sums into its own buffer).
    `telemetry` (Phase 20): a list that collects gain-reduction stats for
    this stem's dynamics entries (never for recursive duck-key renders).
    Pure observation — the rendered audio is identical with or without it.
    Phase 27: rides and time-ranged entries render only while automation is
    active for this stem (session.automation_active) — off = bypassed dry.
    """
    # `_pre_pan` (reader P1 D2.R2): the channel as it stands in front of its pan stage — what a
    # pre-fader send taps natively (the AuxSend sits directly before Tracktion's VolumeAndPan).
    # Only _source_pre_fader asks, and only for a panned stem; never persisted.
    cache_key = (stem_id, exclude_ducks, "prepan") if _pre_pan else (stem_id, exclude_ducks)
    if _cache is not None and cache_key in _cache:
        return _cache[cache_key]
    if _cache is None:
        _cache = {}
    persist_key = None
    feeders = session.stem_feeders(stem_id)   # stems whose Audio To is this one (D2·R4)
    if RENDER_CACHE.enabled and not _pre_pan and not feeders:
        persist_key = _stem_cache_key(session, stem_id, exclude_ducks)
        hit = RENDER_CACHE.get(persist_key)
        if hit is not None:
            audio, sample_rate, cached_telemetry = hit
            if telemetry is not None:
                telemetry.extend(cached_telemetry)
            _cache[cache_key] = (audio, sample_rate)
            return audio, sample_rate
    # With the persistent cache on, GR telemetry is always collected, so a
    # cache hit on a later render can still report this stem's dynamics.
    collect = [] if persist_key is not None else telemetry
    RENDER_COUNTS["stem"] += 1
    stem = session.stems[stem_id]
    audio, sample_rate, mended = _stem_file(session, stem_id, _cache)
    # Audio To ▸ this track (D2·R4): another track's finished output is this one's INPUT, in
    # front of its devices — as in the engine, where the feeder's output is routed to this
    # track. `feeding` breaks a cycle the engine refuses and a hand-edited session may hold.
    if feeders:
        feeding = _cache.setdefault("__feeding__", set())
        if stem_id not in feeding:
            feeding.add(stem_id)
            try:
                for feeder in feeders:
                    if feeder in feeding:
                        continue
                    fed, fed_rate = render_stem(session, feeder, _cache=_cache)
                    if fed_rate == sample_rate:
                        audio = _sum_frames(audio, fed).astype(np.float32)
            finally:
                feeding.discard(stem_id)
    audio = np.ascontiguousarray(audio.T)  # pedalboard wants (channels, frames)

    # Phase 27: with automation off (global or this stem), section moves —
    # gain rides and time-ranged entries — render bypassed (dry). The
    # entries stay in state and in the chain; only their render is gated,
    # so toggling back on is bit-identical to never having toggled.
    automation_on = session.automation_active(stem_id)

    # One list-order walk across BOTH processor representations. Keeping one loop
    # is important: two type-grouped loops silently turn [reverb, EQ] into
    # [EQ, reverb]. A float32 result is exactly representable as float64, so the
    # canonical all-Pedalboard prefix remains sample-for-sample unchanged.
    chain = effective_chain(stem)
    out = audio.T.astype(np.float64)
    for entry in chain:
        if out.size == 0:   # a 0-frame stem / range: seven devices raised on an empty buffer (2026-09-01)
            break
        if not entry.get("enabled", True):   # Phase 40 per-device bypass — parity with
            continue                          # the master chain + live monitor; params
            # are untouched so re-enabling is bit-identical to never toggling.
        if entry.get("time_range_s") is not None and not automation_on:
            continue
        dry32 = None
        wet32 = None
        t = entry["type"]
        if t in ("gain", "tune", "eq", "tilt", "match_eq", "dynamic_eq",
                 "gate", "deess", "comp", "mb_comp", "ott", "spectral_denoise"):
            dry32 = np.ascontiguousarray(out.T.astype(np.float32))
        if t == "gain":
            wet32 = Pedalboard([Gain(gain_db=entry["gain_db"])])(dry32, sample_rate)
        elif t == "tune":  # Phase 31: uniform tuning correction
            wet32 = _pitch_shift(dry32, sample_rate, entry["cents"])
        elif t == "eq":
            wet32 = _eq_entry(dry32, sample_rate, entry)   # band filter + output_db trim
        elif t == "tilt":  # Phase 41: one-knob tilt EQ (two shelves)
            wet32 = _tilt(dry32, sample_rate, entry)
        elif t == "match_eq":
            wet32 = _match_eq(dry32, sample_rate, entry)
        elif t == "dynamic_eq":
            wet32 = _dynamic_eq(dry32, sample_rate, entry)
        elif t == "gate":  # Phase 41: downward gate/expander (pre-comp)
            wet32 = _gate(dry32, sample_rate, entry)
        elif t == "deess":
            wet32 = _deess(dry32, sample_rate, entry)
        elif t == "comp":
            wet32 = _apply_comp(dry32, sample_rate, entry)   # stereo / M/S (incl. true mid+side)
        elif t == "mb_comp":
            wet32 = _multiband_comp(dry32, sample_rate, entry,
                                  telemetry=collect)
        elif t == "ott":   # 3-band upward+downward comp (2026-07-07)
            wet32 = _ott(dry32, sample_rate, entry, telemetry=collect)
        elif t == "spectral_denoise":  # Phase 29 repair (whole-file)
            wet32 = _spectral_denoise(dry32, sample_rate, entry)
        if wet32 is not None:
            blended = _range_blend(dry32, wet32, sample_rate,
                                   entry.get("time_range_s"))
            # Phase 20: GR measured on what is audible (post range blend);
            # comp makeup added back so the stat is reduction, not net level.
            if collect is not None and t in ("comp", "deess"):
                stats = _gr_telemetry(
                    t, dry32.T, blended.T, sample_rate,
                    makeup_db=(_comp_makeup_credit_db(entry) if t == "comp"
                               else entry.get("makeup_db", 0.0)))
                if stats is not None:
                    collect.append(stats)
            out = blended.T.astype(np.float64)
        elif t == "saturation":
            out = _saturate(out, entry, sample_rate)
        elif t == "overdrive":   # drive-pedal distortion (2026-07-07)
            out = _overdrive(out, sample_rate, entry)
        elif t == "chorus":      # modulation family (2026-09-06)
            out = _chorus(out, sample_rate, entry)
        elif t == "flanger":     # modulation family (2026-09-06)
            out = _flanger(out, sample_rate, entry)
        elif t == "auto_filter":  # Phase 63: the modulation family's third
            # Keyed, its envelope reads the TRIGGER's level - rendered the way a Duck's key is
            # (without its own keyed devices, so no render loops) - and a trigger this render
            # cannot hear (an instrument track, a deleted one), or a key render's own keyed
            # device, rests: the native's sidechain is silent there too. Unkeyed: its own input.
            key = entry.get("key_stem_id") or ""
            key_level = None
            if key:
                key_level = np.zeros(out.shape[0], dtype=np.float32)
                if not exclude_ducks and _duck_key_heard(session, key):
                    key_audio, _ = render_stem(session, key, exclude_ducks=True, _cache=_cache)
                    key_level = _key_level(
                        key_audio, out.shape[0], sample_rate,
                        float(session.stems[key].get("capture_start_s", 0.0))
                        - float(session.stems[stem_id].get("capture_start_s", 0.0)))
            out = _auto_filter(out, sample_rate, entry, tempo=_tempo_of(session),
                               key_level=key_level)
        elif t == "transient":
            out = _transient(out, sample_rate, entry)
        elif t == "duck" and not exclude_ducks:
            # SYNC needs no trigger. TRACK reads the trigger rendered without its own keyed devices;
            # a trigger this render cannot hear (none picked, an instrument track, a deleted one)
            # leaves the duck at unity - the native's sidechain is silent there too.
            from .duck_law import trigger_index
            key_audio, key_off = None, 0.0
            if trigger_index(entry.get("trigger", 0)) == 1 and _duck_key_heard(session, entry.get("key_stem_id")):
                key_audio, _ = render_stem(session, entry["key_stem_id"],
                                           exclude_ducks=True, _cache=_cache)
                key_off = (float(session.stems[entry["key_stem_id"]].get("capture_start_s", 0.0))
                           - float(session.stems[stem_id].get("capture_start_s", 0.0)))
            ducked = _duck(out, sample_rate, entry, tempo=_tempo_of(session),
                           key_audio=key_audio, key_offset_s=key_off)
            if collect is not None:
                stats = _gr_telemetry("duck", out, ducked, sample_rate)
                if stats is not None:
                    collect.append(stats)
            out = ducked
        elif t == "utility":   # Ableton Utility: gain/width/mono/phase, freely placeable
            out = _utility(out, sample_rate, entry)
        elif t == "clip":      # Phase 43 device freedom: peak clipper on a stem
            out = _clip(out, entry)
        elif t == "limiter":   # peak ceiling on a stem (brickwall — no loudness makeup)
            out = _brickwall(out, sample_rate, entry)
        elif t in _RETURN_WET_KINDS:
            # Space devices as plain INSERTS (2026-07-20): reverb / delay / IR reverb
            # on a track, dry/wet blended. Off a return they are ordinary inserts.
            out = _wet_device(out, sample_rate, entry)
    if stem["rides"] and automation_on:
        out = out * _ride_envelope(out.shape[0], sample_rate, stem["rides"])[:, np.newaxis]
    # WIDTH then PAN — the order the deliverable runs. Natively the MonitorChain (which carries
    # the width stage) is the FIRST plugin on the track and Tracktion's VolumeAndPan is the LAST,
    # so the native graph is width -> pan. The two do not commute (pan scales L and R by different
    # factors; width is an M/S scale of their sum and difference), so rendering pan -> width here
    # put the offline master and the DAW's bounce ~9 dB apart on a hard-panned stereo channel
    # (2026-09-05 review). The pan LAW is matched from the other side: daw/patches/0007 adds
    # PanLawBalance so the native gains equal _pan's.
    if stem.get("width_on", True):   # a bypassed strip Width plays 1.0 (the stereo width audit, SW1)
        out = _width(out, stem["width"], bool(stem.get("stereoize", False)),
                     float(stem.get("stereoize_ms", 6.0)), sample_rate)
    # The delay the track's node RUNS (its dial summed along its path, plus the compensation a
    # negative dial forces on everyone — the DAW works it out and reports it, D2·R4). Last in
    # the node, so in front of the pan stage and inside a pre-fader tap.
    out = _delayed(out, stem.get("delay_ms", 0.0), sample_rate)
    if not _pre_pan:
        out = _pan(out, stem["pan"])
        # The VCA over this track: the engine sums it into the track's VolumeAndPan, after the
        # devices and the pre-fader sends, without moving the track's own fader (D2·R4).
        vca_db = float(stem.get("vca_db", 0.0) or 0.0)
        if vca_db != 0.0:
            out = out * (10.0 ** (vca_db / 20.0))
    _cache[cache_key] = (out, sample_rate)
    if persist_key is not None:
        if not mended:
            RENDER_CACHE.put(persist_key, out, sample_rate, collect)
        if telemetry is not None:
            telemetry.extend(collect)
    return out, sample_rate


#: render memo key: {stem_id: samples dropped} for stems whose file held NaN / inf (E5·R8)
_NON_FINITE = "__non_finite__"
_TOO_SHORT = "__too_short__"
# The shortest audio a stem's render carries, at the project's rate: ~1.3 ms at 48 k. Above
# anything a one-shot would be, and long enough that no stage sees a degenerate array.
STEM_MIN_FRAMES = 64


def _silenced_by_destination(session: Session) -> tuple[set[str], set[str]]:
    """``(stems, groups)`` that are silent in the mixdown because what they play INTO is muted
    (reader D2·R6). Natively a track is "muted by destination" - ``AudioTrack::isMuted (true)``
    follows its output - and an AuxSend on a track that is not audible returns 0, so a member of
    a muted group, a group under a muted parent and a track whose Audio To is a muted track all
    feed their returns nothing. Offline each one's sends were tapped before any mute was asked:
    the agent, and the limiter's make-up, heard reverb the user had muted. A muted group is in
    the set itself; a muted stem is not (it is never rendered into the mix at all)."""
    buses = session.buses
    dead_buses: set[str] = set()
    for bus_id in buses:
        seen: set[str] = set()
        cursor = bus_id
        while cursor in buses and cursor not in seen:
            seen.add(cursor)
            if buses[cursor].get("muted"):
                dead_buses.add(bus_id)
                break
            cursor = buses[cursor].get("output")

    def dead(stem_id: str, seen: frozenset) -> bool:
        if stem_id in seen:                      # a routing cycle a hand-edited session may hold
            return False
        kind, dest = session.stem_output(stem_id)
        if kind == "track":
            return bool(session.stems[dest].get("muted")) or dead(dest, seen | {stem_id})
        return session.stem_bus(stem_id) in dead_buses

    return {stem_id for stem_id in session.stems if dead(stem_id, frozenset())}, dead_buses


LIMITER_DELAY_S = 0.003        # MonitorChain::lookaheadSamples: the fixed delay (= the reported latency)
LIMITER_ATTACK_CONSTANTS = 6.0 # kLimAttackConstants: the fast stage's ramp is finished across the window
LIMITER_FLOOR_HALF = 2         # kLimFloorHalf: the gain floor's half-width (samples)
LIMITER_SMOOTH_HALF = 2        # kLimSmoothHalf: the gain's min-then-average half-width (samples)


def limiter_window(sample_rate: int, lookahead_ms: float) -> int:
    """The lookahead window in samples (`MonitorChain::limWindow`): the knob, at least one sample,
    and no longer than the fixed delay leaves after the detector's own lag (TP_CENTRE) and the
    smoothing's reach (2 x LIMITER_SMOOTH_HALF) - ~2.5 ms."""
    ring = max(1, int(sample_rate * LIMITER_DELAY_S))
    return min(max(1, round(sample_rate * lookahead_ms / 1000.0)),
               max(1, ring - TP_CENTRE - 2 * LIMITER_SMOOTH_HALF))


def limiter_stages(audio: np.ndarray, sample_rate: int, entry: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """THE DAW'S LIMITER LAW (`MonitorChain::applyLookaheadLimiter`, v3 2026-10-25: two stages) on
    (frames, channels) float64 that already carries its makeup. Returns ``(gain, slow, target)``,
    each (frames, channels), aligned on the input (the native's fixed delay is its reported
    latency, compensated). Per output sample m:

    * the TARGET is ceiling / peak where the native's own 4x detector (`tp_detector`, the same
      table) reads over, blended between each channel's and the loudest's by `channel_link`;
    * the WINDOW is its minimum over [m, m + L], L = `limiter_window` - the lookahead knob is how
      early a peak starts being pulled down (it was the whole fixed delay, whatever the knob said);
    * the FAST stage follows the window: attack across L (six constants), release the character's
      fast release - it catches each peak and lets go in milliseconds;
    * the SLOW stage follows the window's minimum over the last H samples (the character's hold:
      steady through a low note): the character's attack, release `release_ms` - the held level;
    * the gain is the lower of the two, never above the least target within LIMITER_FLOOR_HALF of
      the sample (the floor that holds a crest between samples);
    * and it is SMOOTHED without being raised: the minimum over 2h + 1 samples, then the average
      over 2h + 1 (h = LIMITER_SMOOTH_HALF), so it is never above what it was at any sample. A
      0.5 ms window ramps the fast stage within a few samples of a crest, and that steep a change
      sharpened the crest between samples - up to 0.03 dB past the ceiling on a song at +18 dB
      (an exact FFT meter; 0.007 after). The old law's 3 ms ramp was gentle enough not to.

    The characters (models.LIMITER_CHARACTERS) were sized by measurement against a reference
    limiter; the record is docs/10-daw.md. `slow` is what the face draws as the BODY."""
    from scipy.ndimage import minimum_filter1d
    from .models import LIMITER_CHARACTERS, limiter_knobs

    x = np.asarray(audio, dtype=np.float64)
    n, nch = x.shape
    ceiling = 10.0 ** (float(entry["ceiling_dbtp"]) / 20.0)
    rel_ms, look_ms, style = limiter_knobs(entry)
    fast_rel_ms, slow_att_ms, hold_ms = LIMITER_CHARACTERS[style]
    rel_ms = max(1.0, rel_ms)
    L = limiter_window(sample_rate, look_ms)
    H = int(round(sample_rate * hold_ms / 1000.0))
    link = min(1.0, max(0.0, float(entry.get("channel_link", 1.0))))
    peak = intersample_peak(x)                                       # (frames, channels)
    own = np.where(peak > ceiling, ceiling / np.maximum(peak, 1e-30), 1.0)
    loud = np.max(peak, axis=1, keepdims=True)
    linked = np.where(loud > ceiling, ceiling / np.maximum(loud, 1e-30), 1.0)
    target = own * (1.0 - link) + linked * link                      # aligned on the sample
    gain = np.ones_like(x)
    slow_out = np.ones_like(x)
    if n == 0 or float(np.max(peak, initial=0.0)) <= ceiling:
        return gain, slow_out, target                                # nothing over: no reduction
    atk_ms = L * 1000.0 / sample_rate / LIMITER_ATTACK_CONSTANTS
    h = LIMITER_SMOOTH_HALF
    lead, tail = L + 2 * h, L + 2 * h          # the stages run from 2h + L samples before the first output
    for ch in range(nch):
        t_ext = np.concatenate([np.ones(lead), target[:, ch], np.ones(tail)])
        fwd = minimum_filter1d(t_ext, size=L + 1, origin=-((L + 1) // 2), mode="constant", cval=1.0)
        fast = 1.0 + _gr_ballistics_db(fwd - 1.0, sample_rate, atk_ms, fast_rel_ms)
        held = (minimum_filter1d(fwd, size=H + 1, origin=H // 2, mode="constant", cval=1.0)
                if H > 0 else fwd)
        slow = 1.0 + _gr_ballistics_db(held - 1.0, sample_rate, slow_att_ms, rel_ms)
        floor = minimum_filter1d(t_ext, size=2 * LIMITER_FLOOR_HALF + 1, mode="constant", cval=1.0)
        raw = np.minimum(np.minimum(fast, slow), floor)
        # min, then average, over 2h + 1 - at output sample m these read raw[m - 2h .. m + 2h]
        lo = lead - 2 * h
        mf = minimum_filter1d(raw[lo:lead + n + 2 * h], size=2 * h + 1, mode="constant", cval=1.0)
        sm = np.convolve(mf, np.ones(2 * h + 1) / (2 * h + 1), mode="valid")   # aligned: sm[k] <-> m = k
        gain[:, ch] = sm[h:h + n]
        slow_out[:, ch] = slow[lead:lead + n]
    return gain, slow_out, target


def _brickwall(audio: np.ndarray, sample_rate: int, entry: dict) -> np.ndarray:
    """The DAW's limiter on (frames, channels): the GAIN knob (`gain_db`, the drive into it - on
    the master it adds to the LOUD target's make-up, applied before this), then `limiter_stages`'
    gain, then the ceiling clamp (the backstop the native applies to every sample)."""
    x = np.asarray(audio, dtype=np.float64)
    drive_db = float(entry.get("gain_db", 0.0) or 0.0)
    if drive_db:
        x = x * 10.0 ** (min(24.0, max(0.0, drive_db)) / 20.0)
    ceiling = 10.0 ** (float(entry["ceiling_dbtp"]) / 20.0)
    gain, _, _ = limiter_stages(x, sample_rate, entry)
    return np.clip(x * gain, -ceiling, ceiling)


def _master_limiter(audio: np.ndarray, sample_rate: int, entry: dict,
                    report: Optional[dict],
                    telemetry: Optional[list] = None) -> np.ndarray:
    """The only processor allowed to push loudness (Phase 6, docs/05):
    1. optional makeup toward target_lufs_i (measured with the engine's own
       contractual LUFS — deterministic: same input, same measurement);
    2. Pedalboard brickwall at the ceiling (sample peaks);
    3. deterministic trim if 4x-oversampled *true* peak still exceeds the
       ceiling — the ≤ −1.0 dBTP guarantee lives here, not in a flag."""
    from . import analysis

    makeup_db = 0.0
    pre_lufs = None
    target = entry.get("target_lufs_i")
    if target is not None:
        import pyloudnorm as pyln
        pre_lufs = analysis._integrated_lufs(pyln.Meter(sample_rate), audio)
        if pre_lufs is not None:
            makeup_db = target - pre_lufs
            audio = audio * 10.0 ** (makeup_db / 20.0)

    pre_limit = audio  # post-makeup input: GR is what the brickwall+trim took
    limiter_input_lufs = None
    if report is not None:
        import pyloudnorm as pyln
        limiter_input_lufs = analysis._integrated_lufs(
            pyln.Meter(sample_rate), pre_limit)
    audio = _brickwall(audio, sample_rate, entry)

    pre_trim_true_peak = analysis._true_peak_dbtp(audio)
    true_peak_overshoot_db = max(0.0, pre_trim_true_peak
                                 - entry["ceiling_dbtp"])
    trim_db = min(0.0, entry["ceiling_dbtp"] - pre_trim_true_peak)
    if trim_db < 0.0:
        audio = audio * 10.0 ** (trim_db / 20.0)
    stats = None
    if telemetry is not None:
        stats = _gr_telemetry("limiter", pre_limit, audio, sample_rate)
        if stats is not None:
            telemetry.append(stats)
    if report is not None:
        if stats is None:
            stats = _gr_telemetry("limiter", pre_limit, audio, sample_rate)
        import pyloudnorm as pyln
        post_lufs = analysis._integrated_lufs(pyln.Meter(sample_rate), audio)
        final_true_peak = analysis._true_peak_dbtp(audio)
        tp_margin = entry["ceiling_dbtp"] - final_true_peak
        report.update(pre_limiter_lufs_i=(round(pre_lufs, 1)
                                          if pre_lufs is not None else None),
                      limiter_input_lufs_i=(round(limiter_input_lufs, 1)
                                            if limiter_input_lufs is not None
                                            else None),
                      post_limiter_lufs_i=(round(post_lufs, 1)
                                           if post_lufs is not None else None),
                      limiter_makeup_db=round(makeup_db, 1),
                      limiter_trim_db=round(trim_db, 1),
                      limiter_gr_mean_db=(stats or {}).get("gr_mean_db", 0.0),
                      limiter_gr_max_db=(stats or {}).get("gr_max_db", 0.0),
                      limiter_gr_active_pct=(stats or {}).get(
                          "gr_active_pct", 0.0),
                      true_peak_overshoot_db=round(true_peak_overshoot_db, 1),
                      true_peak_margin_db=round(tp_margin, 1))
    return audio


def _apply_master_chain(mix: np.ndarray, sample_rate: int, chain: list[dict],
                        report: Optional[dict],
                        telemetry: Optional[list] = None, tempo=None) -> np.ndarray:
    """Master-bus processing in fixed order transpose -> eq -> match-eq ->
    dynamic-eq -> bus comp -> multiband comp -> saturation -> width -> limiter
    (session.effective_master_chain provides
    exactly that order; the gain entry is applied by render_master before
    this runs). Master entries are always whole-file — ranged actions are
    rejected at propose time."""
    for entry in chain:
        if not entry.get("enabled", True):   # Phase 40: user per-device bypass —
            continue                          # skipped here and in the live monitor;
            # bit-identical when re-enabled (the entry's params are untouched).
        if entry["type"] == "transpose":  # Phase 32: whole-song key change —
            # first in the chain (effective_master_chain), so eq/comp/limiter
            # operate on the transposed mix and the ceiling stays guaranteed.
            dry32 = np.ascontiguousarray(mix.T.astype(np.float32))
            mix = _pitch_shift(dry32, sample_rate,
                               entry["semitones"] * 100).T.astype(np.float64)
        elif entry["type"] in ("eq", "comp"):
            dry32 = np.ascontiguousarray(mix.T.astype(np.float32))
            if entry["type"] == "eq":   # Phase 41/46: optional M/S or L/R channel on the master eq
                processed = _eq_entry(dry32, sample_rate, entry)   # + output_db trim
            else:                       # stereo / M/S comp (incl. true mid+side) — native parity
                processed = _apply_comp(dry32, sample_rate, entry)
            if telemetry is not None and entry["type"] == "comp":
                stats = _gr_telemetry("comp", mix, processed.T, sample_rate,
                                      makeup_db=entry.get("makeup_db", 0.0))
                if stats is not None:
                    telemetry.append(stats)
            mix = processed.T.astype(np.float64)
        elif entry["type"] == "tilt":  # Phase 41: one-knob tilt EQ (two shelves)
            mix = _tilt(np.ascontiguousarray(mix.T.astype(np.float32)),
                        sample_rate, entry).T.astype(np.float64)
        elif entry["type"] == "match_eq":
            mix = _match_eq(np.ascontiguousarray(mix.T.astype(np.float32)),
                            sample_rate, entry).T.astype(np.float64)
        elif entry["type"] == "dynamic_eq":
            mix = _dynamic_eq(np.ascontiguousarray(mix.T.astype(np.float32)),
                              sample_rate, entry).T.astype(np.float64)
        elif entry["type"] == "mb_comp":
            mix = _multiband_comp(
                np.ascontiguousarray(mix.T.astype(np.float32)),
                sample_rate, entry, telemetry=telemetry).T.astype(np.float64)
        elif entry["type"] == "ott":   # 3-band up/down comp on the master (user-add)
            mix = _ott(np.ascontiguousarray(mix.T.astype(np.float32)),
                       sample_rate, entry, telemetry=telemetry).T.astype(np.float64)
        elif entry["type"] == "saturation":   # Phase 40: master-bus coloration
            mix = _saturate(mix, entry, sample_rate)   # (user co-edit; float64 in place)
        elif entry["type"] == "overdrive":    # drive-pedal distortion (float64 in place)
            mix = _overdrive(mix, sample_rate, entry)
        elif entry["type"] == "chorus":       # modulation family (float64 in place)
            mix = _chorus(mix, sample_rate, entry)
        elif entry["type"] == "flanger":      # modulation family (float64 in place)
            mix = _flanger(mix, sample_rate, entry)
        elif entry["type"] == "auto_filter":  # Phase 63 (on the master: its own input drives it)
            mix = _auto_filter(mix, sample_rate, entry, tempo=tempo)
        elif entry["type"] == "deess":   # Phase 43 device freedom: de-ess on the master too
            dry32 = np.ascontiguousarray(mix.T.astype(np.float32))
            wet = _deess(dry32, sample_rate, entry)
            if telemetry is not None:
                stats = _gr_telemetry("deess", dry32.T, wet.T, sample_rate)
                if stats is not None:
                    telemetry.append(stats)
            mix = wet.T.astype(np.float64)
        elif entry["type"] == "gate":    # downward gate/expander on the master
            wet = _gate(np.ascontiguousarray(mix.T.astype(np.float32)),
                        sample_rate, entry)
            mix = wet.T.astype(np.float64)
        elif entry["type"] == "transient":   # transient shaper on the master
            mix = _transient(mix, sample_rate, entry)
        elif entry["type"] == "width":         # Phase 40: master-bus M/S width
            mix = _width_entry(mix, entry, sample_rate)
        elif entry["type"] == "utility":       # Ableton Utility: gain/width/mono/phase, freely placeable
            mix = _utility(mix, sample_rate, entry)         # (float64 (frames, channels) in place — linear)
        elif entry["type"] == "clip":          # Phase 41: pre-limiter peak clipper
            mix = _clip(mix, entry)            # (memoryless; shave peaks before limiting)
        elif entry["type"] == "limiter":
            mix = _master_limiter(mix, sample_rate, entry, report,
                                  telemetry=telemetry)
        elif entry["type"] in _RETURN_WET_KINDS:   # space devices as master inserts (2026-07-20)
            mix = _wet_device(mix, sample_rate, entry)
    return mix


def _apply_bus_chain(mix: np.ndarray, sample_rate: int, chain: list[dict],
                     telemetry: Optional[list] = None, tempo=None) -> np.ndarray:
    """Group-bus processing (Phase 23) on the float64 member sum, in
    effective_bus_chain order: gain -> eq -> match-eq -> dynamic-eq -> de-ess
    -> comp -> multiband comp -> saturation -> transient -> band width -> pan
    -> width (the last two are the strip's scalars, as on a stem). Float32
    processors cross the same boundary as the master chain; saturation /
    transient / band width run float64 like the stem stage. Bus entries
    are always whole-file (ranged actions are rejected at propose time)."""
    for entry in chain:
        if not entry.get("enabled", True):   # Phase 40 per-device bypass — parity with
            continue                          # the master chain, live monitor, and stems.
        kind = entry["type"]
        if kind == "gain":
            if entry["gain_db"]:
                mix = mix * 10.0 ** (entry["gain_db"] / 20.0)
        elif kind in ("eq", "deess", "comp"):
            dry32 = np.ascontiguousarray(mix.T.astype(np.float32))
            if kind == "eq":
                wet = _eq_entry(dry32, sample_rate, entry)   # + output_db trim
            elif kind == "deess":
                wet = _deess(dry32, sample_rate, entry)
            else:                       # stereo / M/S comp (incl. true mid+side) — native parity
                wet = _apply_comp(dry32, sample_rate, entry)
            if telemetry is not None and kind in ("comp", "deess"):
                stats = _gr_telemetry(kind, dry32.T, wet.T, sample_rate,
                                      makeup_db=entry.get("makeup_db", 0.0))
                if stats is not None:
                    telemetry.append(stats)
            mix = wet.T.astype(np.float64)
        elif kind == "tilt":  # Phase 41: one-knob tilt EQ on the bus
            mix = _tilt(np.ascontiguousarray(mix.T.astype(np.float32)),
                        sample_rate, entry).T.astype(np.float64)
        elif kind == "gate":  # Phase 41: downward gate/expander on the bus
            wet = _gate(np.ascontiguousarray(mix.T.astype(np.float32)),
                        sample_rate, entry)
            mix = wet.T.astype(np.float64)
        elif kind == "match_eq":
            mix = _match_eq(np.ascontiguousarray(mix.T.astype(np.float32)),
                            sample_rate, entry).T.astype(np.float64)
        elif kind == "dynamic_eq":
            mix = _dynamic_eq(np.ascontiguousarray(mix.T.astype(np.float32)),
                              sample_rate, entry).T.astype(np.float64)
        elif kind == "mb_comp":
            mix = _multiband_comp(
                np.ascontiguousarray(mix.T.astype(np.float32)),
                sample_rate, entry, telemetry=telemetry).T.astype(np.float64)
        elif kind == "ott":       # 3-band upward+downward comp on the bus
            mix = _ott(np.ascontiguousarray(mix.T.astype(np.float32)),
                       sample_rate, entry, telemetry=telemetry).T.astype(np.float64)
        elif kind == "saturation":
            mix = _saturate(mix, entry, sample_rate)
        elif kind == "overdrive": # drive-pedal distortion (float64 in place)
            mix = _overdrive(mix, sample_rate, entry)
        elif kind == "chorus":    # modulation family (float64 in place)
            mix = _chorus(mix, sample_rate, entry)
        elif kind == "flanger":   # modulation family (float64 in place)
            mix = _flanger(mix, sample_rate, entry)
        elif kind == "auto_filter":   # Phase 63 (on a group or a return: its own input drives it)
            mix = _auto_filter(mix, sample_rate, entry, tempo=tempo)
        elif kind == "transient":
            mix = _transient(mix, sample_rate, entry)
        elif kind == "utility":   # Ableton Utility: gain/width/mono/phase, freely placeable
            mix = _utility(mix, sample_rate, entry)
        elif kind == "clip":      # Phase 43 device freedom: peak clipper on a bus
            mix = _clip(mix, entry)
        elif kind == "limiter":   # peak ceiling on a bus (brickwall — no loudness makeup)
            mix = _brickwall(mix, sample_rate, entry)
        elif kind in _RETURN_WET_KINDS:   # space devices as bus inserts (2026-07-20)
            mix = _wet_device(mix, sample_rate, entry)
        elif kind == "pan":       # the bus strip's pan (trailing scalar, as on a stem)
            mix = _pan(mix, entry["pan"])
        elif kind == "width":     # the bus strip's M/S width (trailing scalar, as on a stem)
            mix = _width_entry(mix, entry, sample_rate)
    return mix


def render_bus(session: Session, bus_id: str,
               telemetry: Optional[list] = None,
               _cache: Optional[dict] = None,
               dry: bool = False) -> tuple[np.ndarray, int]:
    """Render one group bus alone (Phase 23): active member stems through
    their chains, summed in float64 in sorted order, then the bus chain —
    what analyze_bus listens to. Raises for an unknown bus or one with no
    active members. ``dry=True`` returns the bus INPUT — the summed member
    chains BEFORE the bus chain/gain — the tap point for a PRE-FADER send off
    the bus: the bus AFTER its devices, BEFORE its fader and pan (reader P1 D2.R2 — it was the
    bus INPUT, in front of the whole chain)."""
    bus = session.buses[bus_id]  # KeyError for unknown busses
    members = [stem_id for stem_id in session.bus_members(bus_id)
               if not session.stems[stem_id]["muted"]
               and not session.stems[stem_id]["excluded"]]
    if not members:
        raise ValueError(f"bus '{bus_id}' has no active member stems")
    RENDER_COUNTS["bus"] += 1
    cache: dict = {} if _cache is None else _cache
    rendered: list[np.ndarray] = []
    sample_rate: Optional[int] = None
    for stem_id in members:
        audio, sr = render_stem(session, stem_id, _cache=cache)
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise ValueError(f"sample-rate mismatch: '{stem_id}' is {sr} Hz, "
                             f"expected {sample_rate}")
        rendered.append(audio)
    frames = max(a.shape[0] for a in rendered)
    channels = max(a.shape[1] for a in rendered)
    out = np.zeros((frames, channels), dtype=np.float64)
    for audio in rendered:
        if audio.shape[1] == 1 and channels > 1:
            audio = np.tile(audio, (1, channels))
        out[: audio.shape[0], : audio.shape[1]] += audio
    tempo = _tempo_of(session)
    if dry:   # a PRE-FADER send off the bus: post its devices, pre its fader and pan (D2.R2)
        return _bus_pre_fader(out, sample_rate, bus, tempo=tempo), sample_rate
    return _apply_bus_chain(out, sample_rate, effective_bus_chain(bus),
                            telemetry=telemetry, tempo=tempo), sample_rate


def _bus_pre_fader(bus_input: np.ndarray, sample_rate: int, bus: dict,
                   post: Optional[np.ndarray] = None, tempo=None) -> np.ndarray:
    """A group bus's pre-fader tap: its chain without the trailing pan, the fader divided back
    out (see ``_unfaded``). ``post`` = the bus's finished output when the caller already has it
    — an unpanned bus needs no second pass."""
    chain = effective_bus_chain(bus)
    if post is None or any(e["type"] == "pan" for e in chain):
        post = _apply_bus_chain(bus_input, sample_rate,
                                [e for e in chain if e["type"] != "pan"], tempo=tempo)
    return _unfaded(post, bus["gain_db"])


def _arranged(audio: np.ndarray, sample_rate: int, clips: list) -> np.ndarray:
    """A stem's file as SEVERAL clips play it (session ``clips``, 2026-10-11): a timeline of
    silence with each window laid on it at ``at_s`` - the file from ``from_s`` for ``for_s`` -
    times its clip gain, under a linear fade in and out. Where two windows meet the same frame
    the one that starts later is what plays (a lane does not overlap its clips; the DAW reports
    no list for one that does). Never empty."""
    frames, channels = audio.shape
    end = max(int(round((float(w["at_s"]) + float(w["for_s"])) * sample_rate)) for w in clips)
    out = np.zeros((max(1, end), channels), dtype=audio.dtype)
    for w in sorted(clips, key=lambda w: float(w["at_s"])):
        first = min(frames, int(round(float(w["from_s"]) * sample_rate)))
        piece = audio[first: min(frames, first + int(round(float(w["for_s"]) * sample_rate)))]
        if piece.shape[0] == 0:
            continue                                   # a window past the file's end plays nothing
        piece = piece * (10.0 ** (float(w.get("gain_db", 0.0) or 0.0) / 20.0))
        n = piece.shape[0]
        fade_in = min(n, int(round(float(w.get("fade_in_s", 0.0) or 0.0) * sample_rate)))
        fade_out = min(n - fade_in, int(round(float(w.get("fade_out_s", 0.0) or 0.0) * sample_rate)))
        if fade_in > 0:
            piece[:fade_in] *= (np.arange(fade_in) / fade_in)[:, np.newaxis]
        if fade_out > 0:
            piece[n - fade_out:] *= (np.arange(fade_out, 0, -1) / fade_out)[:, np.newaxis]
        at = int(round(float(w["at_s"]) * sample_rate))
        out[at: at + n] = piece[: max(0, out.shape[0] - at)]
    return out


def _placed(audio: np.ndarray, sample_rate: int, clip, clips=None) -> np.ndarray:
    """A stem's file as its clip plays it (session ``clip``: readers D4·R8 / D1·R2): the
    window that starts ``from_s`` into the file and lasts ``for_s`` (0 = to its end), with
    ``at_s`` of silence in front. No window hands back the same array, so every session
    without one renders bit-identically. Never empty: a window past the file's end is one
    frame of silence, like a clip trimmed to nothing. A track of SEVERAL clips is `_arranged`."""
    if clips:
        return _arranged(audio, sample_rate, clips)
    if not clip:
        return audio
    frames = audio.shape[0]
    first = min(frames, int(round(float(clip.get("from_s", 0.0) or 0.0) * sample_rate)))
    length = float(clip.get("for_s", 0.0) or 0.0)
    last = frames if length <= 0.0 else min(frames, first + int(round(length * sample_rate)))
    window = audio[first:last]
    lead = int(round(float(clip.get("at_s", 0.0) or 0.0) * sample_rate))
    if window.shape[0] == 0:
        window = np.zeros((1, audio.shape[1]), dtype=audio.dtype)
    if lead <= 0:
        return np.ascontiguousarray(window)
    return np.concatenate([np.zeros((lead, audio.shape[1]), dtype=audio.dtype), window])


def _delayed(audio: np.ndarray, delay_ms: float, sample_rate: int) -> np.ndarray:
    """Shift (frames, channels) audio later by ``delay_ms`` — a Track Delay as the engine
    realizes it (always >= 0: a negative dial is everyone else's delay). The length is kept;
    0 hands back the same array, so a session with no delay renders bit-identically."""
    shift = int(round(float(delay_ms or 0.0) * sample_rate / 1000.0))
    if shift <= 0 or audio.shape[0] == 0:
        return audio
    out = np.zeros_like(audio)
    if shift < audio.shape[0]:
        out[shift:] = audio[:-shift]
    return out


def _sum_frames(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a + b for two (frames, channels) arrays of any lengths / mono-or-stereo widths."""
    frames = max(a.shape[0], b.shape[0])
    channels = max(a.shape[1], b.shape[1])
    out = np.zeros((frames, channels), dtype=np.float64)
    for part in (a, b):
        if part.shape[1] == 1 and channels > 1:
            part = np.tile(part, (1, channels))
        out[: part.shape[0], : part.shape[1]] += part
    return out


def _unfaded(audio: np.ndarray, gain_db: float) -> np.ndarray:
    """Take a channel's fader back out of its processed signal. The fader is the chain's FIRST
    stage in both engines (``effective_chain`` synthesizes the leading ``gain``; the native node
    runs it first), so everything after the chain is post-fader — and a PRE-fader tap is that
    signal with the fader's gain divided out, which is what the native send does per block
    (engine patch 0020, ``DawEngineBehaviour::getAuxSendGainTrim``). Exact for a linear chain;
    with dynamics after the fader both engines carry the same remainder."""
    return audio / (10.0 ** (round(float(gain_db), 1) / 20.0))


def _source_pre_fader(session: Session, stem_id: str,
                      _cache: Optional[dict] = None) -> tuple[np.ndarray, int]:
    """The tap point of a PRE-FADER send (reader P1 D2.R2): the stem AFTER its devices, rides
    and width, BEFORE its fader and its pan — "post-devices, pre-fader", as the manual promises
    and as the native send taps (directly before the track's pan stage, the fader taken back
    out). It used to be the raw file (``_source_dry``): the agent's return and master numbers
    heard a reverb fed by the un-de-essed, un-trimmed source while the DAW fed it the processed
    one, following the fader. An unpanned stem reuses its cached render."""
    stem = session.stems[stem_id]
    if stem["pan"] == 0.0 and not stem.get("vca_db"):   # nothing in the pan stage to step in front of
        audio, sample_rate = render_stem(session, stem_id, _cache=_cache)
    else:
        audio, sample_rate = render_stem(session, stem_id, _cache=_cache, _pre_pan=True)
    return _unfaded(audio, stem["gain_db"]), sample_rate


def render_return(session: Session, return_id: str,
                  _cache: Optional[dict] = None) -> tuple[np.ndarray, int]:
    """Render one send/return bus alone (Phase 38 + 2026-06 Ableton rebuild): every active source's
    send into it, summed in float64, then the return's wet-only chain — what analyze_return listens
    to. A PRE-FADER return taps each source after its devices with the fader taken back out
    (``_source_pre_fader``); a POST-FADER return
    (default) taps the processed source. Cascading: an upstream return feeding this one contributes
    its wet (forward-only, so the recursion can't loop). Mirrors render_master, so the measured wet
    matches the deliverable. Raises for an unknown return or one with no active sends."""
    if return_id not in session.returns:
        raise KeyError(return_id)
    cache: dict = {} if _cache is None else _cache
    pre = bool(session.returns[return_id].get("pre_fader"))
    contributions: list[tuple[np.ndarray, float]] = []  # (source audio, send gain_db)
    sample_rate: Optional[int] = None
    dead_stems, dead_buses = _silenced_by_destination(session)   # D2·R6: as render_master

    def consider(audio: np.ndarray, sr: int, gain_db: Optional[float]) -> None:
        nonlocal sample_rate
        if gain_db is None or gain_db <= -60.0:
            return
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise ValueError(f"sample-rate mismatch in return '{return_id}'")
        contributions.append((audio, float(gain_db)))

    for stem_id in sorted(session.stems):
        stem = session.stems[stem_id]
        gain_db = (stem.get("sends") or {}).get(return_id)   # 0 dB is a valid full send → test presence
        if stem["muted"] or stem["excluded"] or gain_db is None or stem_id in dead_stems:
            continue
        audio, sr = (_source_pre_fader(session, stem_id, _cache=cache) if pre
                     else render_stem(session, stem_id, _cache=cache))
        consider(audio, sr, gain_db)
    for bus_id in sorted(session.buses):
        gain_db = (session.buses[bus_id].get("sends") or {}).get(return_id)
        if gain_db is None or bus_id in dead_buses:
            continue
        try:
            audio, sr = render_bus(session, bus_id, _cache=cache, dry=pre)
        except ValueError:
            continue   # a bus with no active members contributes nothing
        consider(audio, sr, gain_db)
    for rid in session.returns:   # cascading: an upstream return feeding this one (forward-only)
        if rid == return_id:
            continue
        gain_db = (session.returns[rid].get("sends") or {}).get(return_id)
        if gain_db is None or gain_db <= -60.0:
            continue
        try:
            audio, sr = render_return(session, rid, _cache=cache)   # the upstream wet (post)
        except ValueError:
            continue   # an upstream return with no active sends contributes nothing
        consider(audio, sr, gain_db)

    if not contributions:
        raise ValueError(f"return '{return_id}' has no active sends")
    frames = max(a.shape[0] for a, _ in contributions)
    channels = max(a.shape[1] for a, _ in contributions)
    inp = np.zeros((frames, channels), dtype=np.float64)
    for audio, gain_db in contributions:
        if audio.shape[1] == 1 and channels > 1:
            audio = np.tile(audio, (1, channels))
        send = audio * (10.0 ** (gain_db / 20.0))
        inp[: send.shape[0], : send.shape[1]] += send
    RENDER_COUNTS["return"] += 1
    out = _apply_return_chain(inp, sample_rate,
                              effective_return_chain(session.returns[return_id]),
                              tempo=_tempo_of(session))
    out = _delayed(out, session.returns[return_id].get("delay_ms", 0.0), sample_rate)   # its own dial (D2·R4)
    # the strip's pan, as the master sum applies it; `muted` is a mixdown filter and is NOT
    # applied here — analyze_return still hears a muted return, as analyze_bus hears a muted group
    out = _pan(out, float(session.returns[return_id].get("pan", 0.0) or 0.0))
    return out, sample_rate


#: THE MIX AS THE APP PLAYED IT, in place of this render for the length of a read (user QA
#: 2026-10-22, Q13). The brain's render hears no instrument track (and no automation lane, hosted
#: plug-in...: `daw_bridge.render_blind_spots`), so on a beat made in the app every reading of the
#: master was a reading of silence and every level the engineer set was a guess. `turn_host` puts
#: the app's own render of the mix, cut at the master's last limiter (mix.measure), here around a
#: read of a mix its render cannot hear; `render_master` then runs the master's last limiter - and
#: anything after it - on that, and keeps nothing. Keyed by the project folder, and good only for
#: the session state it was measured at (its render fingerprint): this is the PROCESS's, so a
#: render on another thread meanwhile - a co-edit's `chains_with_readout` - reads it too, and
#: after a change it was the app's mix of the state BEFORE that change (the pre-merge review of
#: 2026-10-23). A session whose fingerprint differs renders its own.
NATIVE_MIX: dict = {}


@contextlib.contextmanager
def native_mix(project_dir, audio: np.ndarray, sample_rate: int, fp: Optional[str] = None):
    """`audio` (frames x 2, the mix at the master limiter's input) is the master's input for the
    renders of this block, for the session in `project_dir` while its render fingerprint is
    `fp` (the state the app measured; None = the state on disk as the block opens)."""
    key = str(Path(project_dir).resolve())
    if fp is None:
        fp = Session(project_dir, read_only=True).render_fingerprint()
    held = NATIVE_MIX.get(key)
    NATIVE_MIX[key] = (audio, int(sample_rate), fp)
    try:
        yield
    finally:
        if held is None:
            NATIVE_MIX.pop(key, None)
        else:
            NATIVE_MIX[key] = held


def _native_for(session: Session) -> Optional[tuple]:
    """(audio, rate) of the app's mix standing in for this session's render, or None - none is
    held for its project, or the session is not at the state it was measured at."""
    if not NATIVE_MIX:
        return None
    entry = NATIVE_MIX.get(str(Path(session.project_dir).resolve()))
    if entry is None:
        return None
    audio, rate, fp = entry
    try:
        if session.render_fingerprint() != fp:
            return None
    except Exception:  # noqa: BLE001 - a session that cannot say what it is renders its own
        return None
    return audio, rate


def has_native_mix(session: Session) -> bool:
    """Is the app's own mix standing in for this session's render right now (`native_mix`)?"""
    return _native_for(session) is not None


def _master_from_native(session: Session, native: tuple, report: Optional[dict],
                        telemetry: Optional[list], pre_master: bool = False) -> tuple[np.ndarray, int]:
    """The master from the app's mix. `pre_master` (the mix as it enters mastering) is the
    app's mix as it is: cut at the last limiter, so the master's devices AHEAD of that limiter
    are in it - the nearest the app's render comes, and said where it is used."""
    audio, sample_rate = native
    mix = np.asarray(audio, dtype=np.float64)
    if mix.ndim == 1:
        mix = np.stack([mix, mix], axis=1)
    elif mix.shape[1] == 1:
        mix = np.repeat(mix, 2, axis=1)
    else:
        mix = mix[:, :2].copy()
    chain = [e for e in session.master_chain if e["type"] != "gain"]
    last = max((i for i, e in enumerate(chain) if e["type"] == "limiter"), default=None)
    tail = chain[last:] if last is not None and not pre_master else []
    if tail:
        mix = _apply_master_chain(mix, sample_rate, tail, report, telemetry, tempo=_tempo_of(session))
    if report is not None:
        report["native_mix"] = True
    return mix, sample_rate


def render_master(session: Session,
                  range_s: Optional[tuple[float, float]] = None,
                  report: Optional[dict] = None,
                  telemetry: Optional[list] = None,
                  stem_telemetry: Optional[dict] = None,
                  pre_master: bool = False,
                  _cache: Optional[dict] = None) -> tuple[np.ndarray, int]:
    """Render the master bounce. Returns (audio_ndarray, sample_rate) - see
    `_render_master_whole` for what a render is. This is its door: the whole render is
    served from `MASTER_MEMO` when the session's render fingerprint has not moved (the DAW's
    brain; off elsewhere), and `range_s` is cut from it here, so six sections measured on the
    master are one render. The array handed back is never to be edited in place."""
    native = _native_for(session)
    key = None if native is not None else _master_memo_key(session, pre_master)
    entry = MASTER_MEMO.get(key) if key is not None else None
    if entry is None:
        if native is not None:   # the mix as the APP played it (Q13): never kept, read for this call
            mix, sample_rate = _master_from_native(session, native, report, telemetry, pre_master)
        elif key is None:
            mix, sample_rate = _render_master_whole(session, report, telemetry, stem_telemetry,
                                                    pre_master, _cache)
        else:
            # Collected whether or not this caller asked: the next one may (as the stem cache
            # collects its telemetry). Pure observation - the audio is the same either way.
            kept_report, kept_tel, kept_stem_tel = {}, [], {}
            mix, sample_rate = _render_master_whole(session, kept_report, kept_tel,
                                                    kept_stem_tel, pre_master, _cache)
            if np.isfinite(mix).all():      # a render that raises below is never kept
                entry = MASTER_MEMO.put(key, {
                    "audio": mix, "sample_rate": sample_rate, "report": kept_report,
                    "telemetry": kept_tel, "stem_telemetry": kept_stem_tel, "derived": {}})
            if entry is not None:
                entry["audio"].flags.writeable = False
            else:
                _hand_back(report, telemetry, stem_telemetry, kept_report, kept_tel, kept_stem_tel)
    if entry is not None:
        mix, sample_rate = entry["audio"], entry["sample_rate"]
        _hand_back(report, telemetry, stem_telemetry, entry["report"], entry["telemetry"],
                   entry["stem_telemetry"])

    if range_s is not None:
        start_s, end_s = range_s
        # A range past the render used to slice an EMPTY buffer whose analysis raised
        # (or, for a negative start, a NEGATIVE index — the wrong region, silently).
        # Refuse it with a message the caller can act on (2026-09-01 sweep).
        length_s = mix.shape[0] / float(sample_rate)
        if start_s < 0.0 or end_s <= start_s or start_s >= length_s:
            raise ValueError(f"range_s {(start_s, end_s)} is outside the {length_s:.2f} s render")
        mix = mix[int(start_s * sample_rate): int(min(end_s, length_s) * sample_rate)]

    # `not isfinite`, not `isnan` (the lead filed with E5·R8): an inf a device made passed the
    # NaN test and gave NaN master metrics.
    if not np.isfinite(mix).all():
        raise FloatingPointError("render produced non-finite samples (NaN or inf)")
    return mix, sample_rate


def _hand_back(report: Optional[dict], telemetry: Optional[list], stem_telemetry: Optional[dict],
               kept_report: dict, kept_tel: list, kept_stem_tel: dict) -> None:
    """Give a caller its own copies of what a kept master render observed."""
    if report is not None:
        report.update(copy.deepcopy(kept_report))
    if telemetry is not None:
        telemetry.extend(copy.deepcopy(kept_tel))
    if stem_telemetry is not None:
        stem_telemetry.update(copy.deepcopy(kept_stem_tel))


def _render_master_whole(session: Session,
                         report: Optional[dict] = None,
                         telemetry: Optional[list] = None,
                         stem_telemetry: Optional[dict] = None,
                         pre_master: bool = False,
                         _cache: Optional[dict] = None) -> tuple[np.ndarray, int]:
    """The whole master bounce, rendered. Returns (audio_ndarray, sample_rate).

    `pre_master=True` (Phase 25) renders the mix as it enters mastering:
    stems -> busses -> sum -> master fader, master chain bypassed — the
    measurement point for mix-scope reference matching.

    Sum of stems rendered through their chains, then master gain, then the
    master chain (eq -> bus comp -> limiter, Phase 6). Stems are summed in
    float64 in sorted-id order; shorter stems are zero-padded, mono stems are
    upmixed if any stem is stereo. When `report` is a dict it is filled with
    the limiter's MasterReport fields. `telemetry` (Phase 20) collects the
    master chain's gain-reduction stats; `stem_telemetry` (2026-06 review),
    a dict, additionally collects each stem's chain stats under its id —
    same render, no extra cost — so the over-compression flag can see stem
    dynamics without re-rendering every stem. `_cache` is render_stem's memo
    dict: pass one in to share the per-stem renders with the caller
    (analyze_sections needs the same stems again for activity) — entries
    must never be mutated.
    """
    if not session.stems:
        raise ValueError(f"no stems in project: {session.project_dir}")
    RENDER_COUNTS["master"] += 1
    tempo = _tempo_of(session)   # a synced Auto Filter on a group, a return or the master (Phase 63)
    # Muted/excluded stems stay out of the mix (Phase 9). With nothing active
    # the master is silence at the longest stem's length.
    stem_ids = sorted(s for s, state in session.stems.items()
                      if not state["muted"] and not state["excluded"])
    if not stem_ids:
        info = sf.info(str(session.project_dir
                           / next(iter(session.stems.values()))["path"]))
        return np.zeros((info.frames, max(2, info.channels)), dtype=np.float64), info.samplerate

    rendered: list[np.ndarray] = []
    sample_rate: Optional[int] = None
    # duck keys shared across stems render once (Phase 10)
    cache: dict = {} if _cache is None else _cache
    for stem_id in stem_ids:
        per_stem: Optional[list] = [] if stem_telemetry is not None else None
        audio, sr = render_stem(session, stem_id, _cache=cache,
                                telemetry=per_stem)
        if per_stem:
            stem_telemetry[stem_id] = per_stem
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise ValueError(f"sample-rate mismatch: '{stem_id}' is {sr} Hz, expected {sample_rate}")
        rendered.append(audio)

    frames = max(a.shape[0] for a in rendered)
    channels = max(a.shape[1] for a in rendered)
    # A panned BUS upmixes a mono group to stereo (_pan does, exactly as it does for a panned
    # stem — whose render is already stereo by the time this max runs). The bus stage runs
    # AFTER these buffers are sized, so an all-mono project with a panned group has to be
    # stereo from here or the bus sum won't broadcast. No existing session is affected: a bus
    # had no pan at all before 2026-09-05. Width needs nothing — _width no-ops on mono.
    if any(b.get("pan", 0.0) != 0.0 and not b.get("muted")
           for b in session.buses.values()):
        channels = max(channels, 2)
    # ...and a panned RETURN does the same to its wet (2026-10-03, below).
    if any(r.get("pan", 0.0) != 0.0 and not r.get("muted")
           for r in session.returns.values()):
        channels = max(channels, 2)

    def fit(audio: np.ndarray) -> np.ndarray:
        if audio.shape[1] == 1 and channels > 1:
            audio = np.tile(audio, (1, channels))
        return audio

    # Returns sum in INSERTION order — the canonical order (matches the native busNumber
    # assignment + the cascade forward-only rule); iterating session.returns preserves it.
    return_inputs = {
        return_id: np.zeros((frames, channels), dtype=np.float64)
        for return_id in session.returns
    }
    dry_by_id: dict[str, np.ndarray] = {}   # lazily-built pre-fader taps (post-devices, fader out)

    def has_prefader(sends: dict) -> bool:
        return any(session.returns.get(rid, {}).get("pre_fader")
                   for rid, g in (sends or {}).items() if g > -60.0)

    def add_sends(post_audio: np.ndarray, sends: dict[str, float],
                  pre_audio: Optional[np.ndarray] = None) -> None:
        """Tap one source into its returns. A PRE-FADER return taps ``pre_audio`` (the source
        post-devices, pre-fader); a POST-FADER return (default) taps ``post_audio`` (the
        processed source). ``pre_audio`` None (e.g. a cascading return source — a return is
        wet-only) ⇒ always post."""
        if not return_inputs:
            return
        for return_id, gain_db in sorted((sends or {}).items()):
            if return_id not in return_inputs or gain_db <= -60.0:
                continue
            src = (pre_audio if (pre_audio is not None
                                 and session.returns[return_id].get("pre_fader"))
                   else post_audio)
            audio = fit(src)
            send = audio * (10.0 ** (gain_db / 20.0))
            return_inputs[return_id][: send.shape[0], : send.shape[1]] += send

    # Phase 23 bus stage: stems sum per assigned bus (sorted bus order),
    # each bus applies gain + chain, then busses + unrouted stems sum into
    # the master. Phase 38 sends tap post-stem and post-bus sources into
    # wet-only returns, which sum into the mix before the master fader/chain.
    # 2026-07-03 (nested bus routing): a bus may itself route its OWN output
    # into ANOTHER bus (buses[bus_id]["output"]) instead of straight to the
    # master — a group's own output routing, distinct from route_stem. Buses
    # are processed in topological order (a child bus's WET sum lands in its
    # parent's input before the parent applies its own chain) so a parent-bus
    # device audibly affects everything summed into it, nested or not. With no
    # busses/nested-routing this reduces to the earlier sum in the same
    # float64 addition order — bit-identical. Stale/orphan routing (a bus id
    # that no longer exists, either as membership or as an output target)
    # falls back to the direct master sum.
    by_id = dict(zip(stem_ids, rendered))
    # Summed into a group, or nowhere dry at all (D2·R3 / D2·R4): a Sends Only track (or one on
    # a hardware output) reaches the master through its sends alone, and a track whose Audio
    # To is another track is already inside that track's render (render_stem's feeders).
    routed = {stem_id for stem_id in stem_ids
              if session.stem_bus(stem_id) is not None
              or session.stem_output(stem_id)[0] in ("sends_only", "track")}
    mix = np.zeros((frames, channels), dtype=np.float64)
    # Muted by destination (D2·R6): nothing out of the sends of what plays into a muted group or
    # a muted track, at any depth.
    dead_stems, dead_buses = _silenced_by_destination(session)
    for stem_id in stem_ids:
        if stem_id in dead_stems:
            continue
        sends = session.stems[stem_id].get("sends", {})
        pre = None
        if has_prefader(sends):   # only load the dry stem when a pre-fader send needs it
            if stem_id not in dry_by_id:
                dry_by_id[stem_id], _ = _source_pre_fader(session, stem_id, _cache=cache)
            pre = dry_by_id[stem_id]
        add_sends(by_id[stem_id], sends, pre)

    bus_ids = sorted(session.buses)
    members_of: dict[str, list[str]] = {
        bus_id: [stem_id for stem_id in stem_ids
                if session.stem_bus(stem_id) == bus_id]
        for bus_id in bus_ids}
    # children_of[bid] = the OTHER buses whose output routes into bid — used both to order
    # processing (a bus is only summed once every bus feeding it is done) and to decide whether
    # an otherwise-memberless bus is still "in use" (a pure submix collector).
    children_of: dict[str, list[str]] = {bus_id: [] for bus_id in bus_ids}
    for bus_id in bus_ids:
        out = session.buses[bus_id].get("output")
        if out in children_of:
            children_of[out].append(bus_id)
    order: list[str] = []
    visited: set = set()
    visiting: set = set()

    def _visit(bid: str) -> None:
        if bid in visited or bid in visiting:   # a cycle should never reach here (route_bus
            return                              # guards it) — defensively break instead of looping
        visiting.add(bid)
        for child in children_of.get(bid, []):
            _visit(child)
        visiting.discard(bid)
        visited.add(bid)
        order.append(bid)

    for bus_id in bus_ids:
        _visit(bus_id)

    bus_input: dict[str, np.ndarray] = {
        bus_id: np.zeros((frames, channels), dtype=np.float64) for bus_id in bus_ids}
    for bus_id in bus_ids:
        for stem_id in members_of[bus_id]:
            audio = fit(by_id[stem_id])
            bus_input[bus_id][: audio.shape[0], : audio.shape[1]] += audio

    for bus_id in order:
        if not members_of[bus_id] and not children_of[bus_id]:
            continue   # never in use (no stem members, nothing routes into it) — skip, bit-identical
        if bus_id in dead_buses:
            # A muted group is silent (2026-09-05, the bus strip's [M]): nothing into the master,
            # nothing out of its sends, and nothing into a parent bus — so muting a parent
            # silences the whole subtree, which is what the live graph does. Muting is a mixdown
            # filter here exactly as it is for a stem; render_bus (analyze_bus) still hears it.
            # `dead_buses` holds the muted groups AND every group under one (D2·R6): a child was
            # processed, and its own sends tapped, before its parent's mute was reached.
            continue
        bus_sum = bus_input[bus_id]
        bus_sends = session.buses[bus_id].get("sends", {})
        bus_in = bus_sum.copy() if has_prefader(bus_sends) else None
        per_bus: Optional[list] = [] if stem_telemetry is not None else None
        bus_sum = _apply_bus_chain(
            bus_sum, sample_rate,
            effective_bus_chain(session.buses[bus_id]), telemetry=per_bus, tempo=tempo)
        if per_bus:
            stem_telemetry[bus_id] = per_bus
        # post-devices, pre-fader (D2.R2) — was the bus INPUT, in front of its whole chain
        bus_pre = (_bus_pre_fader(bus_in, sample_rate, session.buses[bus_id], post=bus_sum,
                                  tempo=tempo)
                   if bus_in is not None else None)
        add_sends(bus_sum, bus_sends, bus_pre)
        output = session.buses[bus_id].get("output")
        if output in bus_input:   # nested: sum into the PARENT bus's input, before its own chain
            bus_input[output][: bus_sum.shape[0], : bus_sum.shape[1]] += bus_sum
        else:                     # no output set, or an orphan target -> straight to the master
            mix += bus_sum
    for stem_id in stem_ids:
        if stem_id in routed:
            continue
        audio = fit(by_id[stem_id])
        mix[: audio.shape[0], : audio.shape[1]] += audio
    # Returns in insertion order: each return's wet sums into the mix AND (cascading) feeds its
    # own sends into LATER returns' inputs — forward-only, so a downstream return (not yet
    # processed) picks it up. A return source is wet-only, so it always taps post (pre_audio None).
    #
    # A return has a strip too (full audit 2026-10-02, reader S4·R7): MUTED, it is silent in the
    # mixdown — nothing into the master and nothing out of its own sends, as a muted return
    # track is in the live graph (the same mixdown filter a muted stem or group gets;
    # render_return still hears it). PANNED, its wet is panned with the stem's balance law —
    # BEFORE its own sends tap it, as for a stem and a group: natively a return's cascading
    # send sits at the END of its track's plug-in list (dawsession::realizeReturns: "ANY return
    # source … taps at the end of the chain"), after Tracktion's VolumeAndPan. Absent keys are
    # centred and un-muted, so a stored session renders as before.
    for return_id in return_inputs:
        ret = session.returns[return_id]
        if ret.get("muted"):
            continue
        wet = _apply_return_chain(
            return_inputs[return_id], sample_rate,
            effective_return_chain(ret), tempo=tempo)
        wet = _delayed(wet, ret.get("delay_ms", 0.0), sample_rate)   # the return's own dial (D2·R4)
        wet = _pan(wet, float(ret.get("pan", 0.0) or 0.0))   # 0.0 hands back the same array
        add_sends(wet, ret.get("sends", {}))
        mix += wet

    mix *= 10.0 ** (session.master_gain_db / 20.0)
    # Phase 25: pre_master stops here — the mix as it enters mastering
    # (fader applied, chain bypassed), the comparison point for mix-scope
    # reference matching. Same float64 sum, so it stays deterministic.
    chain = [] if pre_master else [e for e in session.master_chain
                                   if e["type"] != "gain"]
    if chain:
        mix = _apply_master_chain(mix, sample_rate, chain, report, telemetry, tempo=tempo)

    if report is not None and cache.get(_NON_FINITE):
        report["non_finite_stems"] = sorted(cache[_NON_FINITE])
    if report is not None and cache.get(_TOO_SHORT):
        report["short_stems"] = sorted(cache[_TOO_SHORT])
    return mix, sample_rate
