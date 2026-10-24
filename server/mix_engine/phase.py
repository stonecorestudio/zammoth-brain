"""The stereo phase reading (`analyze_phase`, 2026-10-13; mix-analysis skill ▸ *Phase*).

One measurement for the agent's ears and the device faces' lamps: a band's L/R correlation over
windows of about 0.3 s. The DAW's imager, width, utility and band-width faces light PHASE while
that reading is below 0 (their meters integrate over the same 0.3 s); `below_zero_pct` here is
the share of a band's sounding time it is - how much of the song such a lamp would be lit.

What is an ISSUE is judged where it matters, and was sized on released records (four mastered
songs and their separated stems, 2026-10-13):

- the MASTER is the judge of width: a band cancels when it is below 0 over the song, or below 0
  for 20 % of its sounding time or more (`MASTER_CANCEL_TIME_PCT`). The released masters read at
  most 15.5 % and never below +0.28 over a song. Its issue names the stems the band's side comes
  from and the move on each.
- a STEM (or group) is judged only for defects: one side upside down (`inverted`), the two sides
  being the same sound a few milliseconds apart (`channel_delay`: a Haas widener combs when
  folded to mono), a band anti-phase as a whole (corr <= -0.2: it folds to mono quieter than two
  unrelated channels would). A wide stem is not an issue: released synth stems sit below 0 for
  40-57 % of their time and the records fold to mono fine;
- two STEMS that cancel each other in the lows (`conflicts`): opposite polarity, or one a few
  milliseconds early - measured on their centre signals at their mix levels.

`mono_safe_width` is exact for the strip width: it scales the side, and a window's correlation is
below 0 exactly when its mid power is below its side power, so the answer is read off the
windows, not searched for by rendering. Deterministic, numpy only - the agent reads numbers."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .models import (BAND_EDGES_HZ, BAND_LABELS, PhaseBand, PhaseConflict, PhaseIssue,
                     PhaseTarget)

PHASE_WINDOW_S = 0.3              # the window a correlation is read over - the lamps' integration
PHASE_CANCEL_CORRELATION = 0.0    # a window below this cancels in mono (the lamps' PHASE)
MASTER_CANCEL_TIME_PCT = 20.0     # a master band below 0 this much of its sounding time cancels
STEM_CANCEL_CORRELATION = -0.2    # a stem band this anti-phase over the song cancels
STEM_SAFE_CORRELATION = 0.2       # mono_safe_width: every band at least this over the song...
STEM_SAFE_TIME_PCT = 20.0         # ...and below 0 less than this of its time
PHASE_BAND_FLOOR_DB = -30.0       # a band this far under the channel's total is not judged
PHASE_WINDOW_FLOOR_DB = -20.0     # a window this far under the band's mean is not "sounding"
PHASE_INVERTED_CORRELATION = -0.5 # broadband AND lowest band at or below this = one side flipped
PHASE_MONO_DB_FLOOR = -40.0
PHASE_WIDTH_STEP = 0.05           # mono_safe_width is said on this grid, rounded DOWN
CHAIN_MADE_DROP = 0.15            # a band this much less correlated than its input: the chain
CHANNEL_DELAY_PEAK = 0.7          # the two sides line up at least this well at some lag...
CHANNEL_DELAY_AT_ZERO = 0.3       # ...and at most this well where they are
CHANNEL_DELAY_MIN_MS = 0.2
CHANNEL_DELAY_MAX_MS = 40.0
PHASE_CONFLICT_BANDS = ("sub", "bass", "low_mid")
PHASE_CONFLICT_CORRELATION = -0.3
PHASE_CONFLICT_SUM_DB = -1.0
PHASE_CONFLICT_MIN_WINDOWS = 8    # ~1.4 s of overlap at least
PHASE_CONFLICT_MIN_OVERLAP_PCT = 5.0
PHASE_CONFLICT_LEVEL_SPAN_DB = 10.0
PHASE_LAG_MAX_MS = 10.0           # a timing fix is looked for within this
PHASE_LAG_MIN_MS = 0.25           # ...and is not a timing fix below this
PHASE_LAG_MIN_GAIN = 0.1          # ...and must line up this much better than a flip would
BASS_MONO_MAX_HZ = 500.0          # utility bass_mono_hz's ceiling
SIDE_SHARE_MIN = 0.1              # a stem carrying less of a band's side than this is not named
_TINY = 1.0e-20

_CONFLICT_TOP_HZ = BAND_EDGES_HZ[BAND_LABELS.index(PHASE_CONFLICT_BANDS[-1]) + 1]


def window_size(sample_rate: int) -> int:
    """The FFT length a window is: the next power of two at or above PHASE_WINDOW_S."""
    return 1 << max(8, int(math.ceil(math.log2(PHASE_WINDOW_S * sample_rate))))


def window_seconds(sample_rate: int) -> float:
    return round(window_size(sample_rate) / float(sample_rate), 3)


def _stereo(audio: np.ndarray) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, np.newaxis]
    if a.shape[1] == 1:
        a = np.repeat(a, 2, axis=1)
    return a[:, :2]


class WindowSums:
    """Per window and band: the sums of |L|^2, |R|^2 and Re(L R*) over the band's bins - what
    the band figures are made of - and, over the whole song, the cross-spectrum of the two sides
    (`sxy`, `sxx`, `syy`: what a delay between them is read from). `mid` (when asked for) is the
    complex spectrum of the centre signal (L + R) / 2 under the conflict bands' top, per window."""

    def __init__(self, audio: np.ndarray, sample_rate: int, keep_mid: bool = False) -> None:
        x = _stereo(audio)
        nfft = window_size(sample_rate)
        hop = nfft // 2
        n = x.shape[0]
        frames = 1 if n <= nfft else 1 + int(math.ceil((n - nfft) / hop))
        need = nfft + (frames - 1) * hop
        if need > n:
            x = np.concatenate([x, np.zeros((need - n, 2))])
        from scipy.signal import get_window
        window = get_window("hann", nfft)
        freqs = np.fft.rfftfreq(nfft, 1.0 / sample_rate)
        self.sample_rate, self.nfft, self.hop, self.frames = sample_rate, nfft, hop, frames
        bands = [((freqs >= lo) & (freqs < hi)) for lo, hi
                 in zip(BAND_EDGES_HZ[:-1], BAND_EDGES_HZ[1:])]
        audible = (freqs >= BAND_EDGES_HZ[0]) & (freqs < BAND_EDGES_HZ[-1])
        self.ll = np.zeros((frames, len(BAND_LABELS)))
        self.rr = np.zeros_like(self.ll)
        self.lr = np.zeros_like(self.ll)
        self.sxy = np.zeros(freqs.size, dtype=np.complex128)
        self.sxx = np.zeros(freqs.size)
        self.syy = np.zeros(freqs.size)
        mid_bins = (freqs >= BAND_EDGES_HZ[0]) & (freqs < _CONFLICT_TOP_HZ)
        self.mid_index = np.flatnonzero(mid_bins)
        self.mid = (np.zeros((frames, self.mid_index.size), dtype=np.complex64)
                    if keep_mid else None)
        views = [np.lib.stride_tricks.sliding_window_view(x[:, ch], nfft)[::hop]
                 for ch in (0, 1)]
        block = 16
        for start in range(0, frames, block):
            stop = min(frames, start + block)
            spec_l = np.fft.rfft(views[0][start:stop] * window, axis=1)
            spec_r = np.fft.rfft(views[1][start:stop] * window, axis=1)
            pl, pr = np.abs(spec_l) ** 2, np.abs(spec_r) ** 2
            cross = spec_l * np.conj(spec_r)
            real = np.real(cross)
            for b, bins in enumerate(bands):
                if bins.any():
                    self.ll[start:stop, b] = pl[:, bins].sum(axis=1)
                    self.rr[start:stop, b] = pr[:, bins].sum(axis=1)
                    self.lr[start:stop, b] = real[:, bins].sum(axis=1)
            self.sxy[audible] += cross[:, audible].sum(axis=0)
            self.sxx[audible] += pl[:, audible].sum(axis=0)
            self.syy[audible] += pr[:, audible].sum(axis=0)
            if keep_mid:
                self.mid[start:stop] = (0.5 * (spec_l[:, mid_bins] + spec_r[:, mid_bins])
                                        ).astype(np.complex64)


def _corr(lr, ll, rr):
    """L/R correlation from the sums: 0 where one side holds (next to) nothing - a channel
    playing from one speaker folds to mono 3 dB down, as two unrelated channels do."""
    lr, ll, rr = (np.asarray(v, dtype=np.float64) for v in (lr, ll, rr))
    denom = np.sqrt(ll * rr)
    scale = np.maximum(ll + rr, _TINY)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denom > 1.0e-9 * scale, lr / np.maximum(denom, _TINY), 0.0)
    return np.clip(out, -1.0, 1.0)


def _mono_db(ll: float, rr: float, lr: float) -> float:
    stereo = 2.0 * (ll + rr)
    mono = ll + rr + 2.0 * lr
    if stereo <= _TINY:
        return 0.0
    return round(max(PHASE_MONO_DB_FLOOR, 10.0 * math.log10(max(mono, _TINY) / stereo)), 1)


def _sounding(power: np.ndarray) -> np.ndarray:
    """The windows a band SOUNDS in: within PHASE_WINDOW_FLOOR_DB of its mean window."""
    mean = float(power.mean()) if power.size else 0.0
    return power > max(mean * 10.0 ** (PHASE_WINDOW_FLOOR_DB / 10.0), _TINY)


def energetic_bands(sums: WindowSums) -> list[int]:
    power = (sums.ll + sums.rr).sum(axis=0)
    total = float(power.sum())
    if total <= _TINY:
        return []
    floor = total * 10.0 ** (PHASE_BAND_FLOOR_DB / 10.0)
    return [b for b in range(len(BAND_LABELS)) if power[b] > floor]


def broadband_corr(sums: WindowSums) -> Optional[float]:
    ll, rr, lr = (float(a.sum()) for a in (sums.ll, sums.rr, sums.lr))
    if ll + rr <= _TINY:
        return None
    return round(float(_corr(lr, ll, rr)), 2)


def _band(sums: WindowSums, b: int) -> tuple[float, float, float, float]:
    """(whole-song correlation, % of sounding windows below 0, 5th-percentile window, mono dB)."""
    ll, rr, lr = sums.ll[:, b], sums.rr[:, b], sums.lr[:, b]
    whole = float(_corr(lr.sum(), ll.sum(), rr.sum()))
    live = _sounding(ll + rr)
    window_corr = _corr(lr[live], ll[live], rr[live]) if live.any() else np.array([whole])
    return (whole,
            100.0 * float(np.mean(window_corr < PHASE_CANCEL_CORRELATION)),
            float(np.percentile(window_corr, 5)),
            _mono_db(float(ll.sum()), float(rr.sum()), float(lr.sum())))


def band_reading(sums: WindowSums, file_sums: Optional[WindowSums] = None
                 ) -> dict[str, PhaseBand]:
    """Every band that carries the channel's sound, read; `file_sums` (its input) adds each
    band's `file_corr` where the input holds anything there."""
    out: dict[str, PhaseBand] = {}
    for b in energetic_bands(sums):
        whole, below, worst, mono = _band(sums, b)
        file_corr = None
        if file_sums is not None and (file_sums.ll[:, b].sum() + file_sums.rr[:, b].sum()) > _TINY:
            file_corr = round(float(_corr(file_sums.lr[:, b].sum(), file_sums.ll[:, b].sum(),
                                          file_sums.rr[:, b].sum())), 2)
        out[BAND_LABELS[b]] = PhaseBand(corr=round(whole, 2), below_zero_pct=round(below, 1),
                                        worst=round(worst, 2), mono_db=mono, file_corr=file_corr)
    return out


def master_cancels(band: PhaseBand) -> bool:
    """A master band cancels: below 0 over the song, or for MASTER_CANCEL_TIME_PCT of it."""
    return (band.corr < PHASE_CANCEL_CORRELATION
            or band.below_zero_pct >= MASTER_CANCEL_TIME_PCT)


def stem_cancels(band: PhaseBand) -> bool:
    """A stem or group band cancels: anti-phase as a whole."""
    return band.corr <= STEM_CANCEL_CORRELATION


def inverted(corr: Optional[float], bands: dict[str, PhaseBand]) -> bool:
    """One side upside down: the whole channel AND its lowest sounding band anti-phase - real
    stereo is never anti-phase at the bottom (a wide source is uncorrelated there, not negative)."""
    if corr is None or corr > PHASE_INVERTED_CORRELATION or not bands:
        return False
    lowest = next(bands[label] for label in BAND_LABELS if label in bands)
    return lowest.corr <= PHASE_INVERTED_CORRELATION


def channel_delay(sums: WindowSums) -> Optional[tuple[float, float]]:
    """(lag_ms, how well the sides line up there) when the two sides are the same sound a few
    milliseconds apart - a Haas widener, a delay on one side. Positive lag: the LEFT plays later
    than the right. None when they are not (a wide source does not line up at any lag)."""
    if not sums.sxx.any() or not sums.syy.any():
        return None
    rate, nfft = sums.sample_rate, sums.nfft
    r = np.fft.irfft(sums.sxy, n=nfft)
    norm = math.sqrt(max(float(np.fft.irfft(sums.sxx, n=nfft)[0]), _TINY)
                     * max(float(np.fft.irfft(sums.syy, n=nfft)[0]), _TINY))
    reach = min(nfft // 2 - 1, int(round(CHANNEL_DELAY_MAX_MS * rate / 1000.0)))
    lags = np.concatenate([np.arange(0, reach + 1), np.arange(-reach, 0)])
    values = np.concatenate([r[:reach + 1], r[-reach:]]) / norm
    k = int(np.argmax(values))
    lag_ms = 1000.0 * float(lags[k]) / rate
    if (values[k] >= CHANNEL_DELAY_PEAK and values[0] <= CHANNEL_DELAY_AT_ZERO
            and abs(lag_ms) >= CHANNEL_DELAY_MIN_MS):
        return round(lag_ms, 2), round(float(values[k]), 2)
    return None


def mono_safe_width(sums: WindowSums, width: float) -> Optional[float]:
    """The widest strip width (on the PHASE_WIDTH_STEP grid, rounded down, at most `width`) at
    which every sounding band of this stem reads STEM_SAFE_CORRELATION or more over the song and
    is below 0 less than STEM_SAFE_TIME_PCT of its time. `sums` are of the channel in front of
    its pan, at strip width `width`. None when the stem already does."""
    bands = energetic_bands(sums)
    if width <= 0.0 or not bands:
        return None
    mid = (sums.ll + sums.rr + 2.0 * sums.lr) / 4.0
    side = (sums.ll + sums.rr - 2.0 * sums.lr) / 4.0
    target = (1.0 - STEM_SAFE_CORRELATION) / (1.0 + STEM_SAFE_CORRELATION)
    live = [_sounding(sums.ll[:, b] + sums.rr[:, b]) for b in bands]

    def wide(scale: float) -> bool:
        squared = scale * scale
        for b, on in zip(bands, live):
            m, s = mid[:, b], side[:, b] * squared
            # corr ~= (M - S) / (M + S) >= safe  <=>  S <= M * (1 - safe) / (1 + safe)
            if s.sum() > m.sum() * target:
                return True
            if on.any() and 100.0 * float(np.mean(m[on] < s[on])) >= STEM_SAFE_TIME_PCT:
                return True
        return False

    if not wide(1.0):
        return None
    for k in range(int(math.floor(width / PHASE_WIDTH_STEP + 1e-9)), -1, -1):
        w = k * PHASE_WIDTH_STEP
        if not wide(w / width):
            return round(w, 2)
    return 0.0


def stereo_devices(chain: list[dict]) -> list[str]:
    """The devices on a chain that change its stereo picture, said the way the agent sets them -
    what a fix names when a chain made a band cancel."""
    out: list[str] = []
    for entry in chain:
        if not entry.get("enabled", True):
            continue
        kind = entry.get("type")
        if kind == "imager":
            moved = [f"{k} {float(entry[k]):.2f}" for k in ("width_low", "width_mid", "width_high")
                     if k in entry and float(entry[k]) > 1.0]
            if moved:
                out.append("imager " + ", ".join(moved))
        elif kind == "band_width" and float(entry.get("width", 1.0)) > 1.0:
            out.append(f"band_width {entry.get('band', '')} {float(entry['width']):.2f}".replace("  ", " "))
        elif kind == "utility":
            left, right = entry.get("invert_left"), entry.get("invert_right")
            left = bool(entry.get("invert", False)) if left is None else bool(left)
            right = bool(entry.get("invert", False)) if right is None else bool(right)
            if left != right:
                out.append("utility " + ("invert_left" if left else "invert_right"))
            if float(entry.get("width", 1.0)) > 1.0:
                out.append(f"utility width {float(entry['width']):.2f}")
        elif kind in ("chorus", "flanger"):
            out.append(kind)
        elif kind == "auto_filter" and float(entry.get("lfo_stereo_deg", 0.0) or 0.0) > 0.0:
            # Phase 63: an offset sweep filters the two sides differently
            out.append(f"auto_filter lfo_stereo_deg {float(entry['lfo_stereo_deg']):.0f}")
        elif kind == "eq" and entry.get("channel") in ("side", "left", "right"):
            out.append(f"eq on the {entry['channel']}")
    return out


def _band_top_hz(label: str) -> float:
    return float(BAND_EDGES_HZ[BAND_LABELS.index(label) + 1])


def stem_move(row: PhaseTarget, label: str, devices: list[str]) -> str:
    """What to do on one stem so it stops pulling `label` below 0: its own widening first (the
    strip width, the devices on its chain), else narrow the band where it is wide in its file."""
    band = row.bands.get(label)
    made = (band is not None and band.file_corr is not None
            and band.file_corr - band.corr >= CHAIN_MADE_DROP)
    parts: list[str] = []
    if made and row.width is not None and row.width > 1.0:
        parts.append(f"strip width {row.width:.2f} -> {row.mono_safe_width:.2f} or less"
                     if row.mono_safe_width is not None else
                     f"strip width {row.width:.2f} -> 1.00")
    if made and devices:
        parts.append("narrow " + ", ".join(devices))
    if not parts:
        top = _band_top_hz(label)
        if top <= BASS_MONO_MAX_HZ:
            parts.append(f"utility bass_mono at {max(top, 120.0):.0f} Hz")
        else:
            parts.append(f"an imager or band_width below 1.0 on {label}")
        if row.mono_safe_width is not None:
            parts.append(f"strip width {row.mono_safe_width:.2f} or less")
    said = f"{row.target}: " + " or ".join(parts)
    if band is not None and band.file_corr is not None:
        said += (f" (its chain takes {label} from {band.file_corr:+.2f} in its file to "
                 f"{band.corr:+.2f})" if made else
                 f" (wide in its file: {band.file_corr:+.2f} in {label})")
    return said


def stem_issues(row: PhaseTarget, *, file_corr: Optional[float] = None,
                file_bands: Optional[dict[str, PhaseBand]] = None,
                delay: Optional[tuple[float, float]] = None, devices: Optional[list[str]] = None,
                flip_side: str = "invert_right") -> list[PhaseIssue]:
    """A stem's or group's own defects (the module docstring): one side flipped, a delay between
    its sides, a band anti-phase as a whole - each with its cause and the move."""
    devices = devices or []
    target, kind = row.target, row.kind
    if inverted(row.corr, row.bands):
        made = file_corr is not None and file_corr > PHASE_INVERTED_CORRELATION
        flips = [d for d in devices if "invert" in d]
        fix = (f"a device on {target} flips one side ({', '.join(flips) or 'its chain'}) - "
               "switch that invert off" if made else
               f"flip one side back: utility {flip_side}: true on {target}")
        return [PhaseIssue(target=target, kind="inverted_channel", cause="chain" if made else "file",
                           corr=row.corr if row.corr is not None else -1.0, fix=fix)]
    out: list[PhaseIssue] = []
    if delay is not None:
        lag, _ = delay
        later = "left" if lag > 0 else "right"
        out.append(PhaseIssue(
            target=target, kind="channel_delay", cause="file", corr=row.corr or 0.0,
            lag_ms=abs(lag),
            fix=(f"its {later} side is the same sound {abs(lag):.1f} ms late: folded to mono it "
                 "combs, and no device here lines the two sides up - leave it if the master's "
                 "reading is clean; otherwise narrow it (strip width) so the stereo and mono "
                 "versions match, or ask the user for a version without the delay")))
    for label, band in row.bands.items():
        if not stem_cancels(band):
            continue
        before = (file_bands or {}).get(label)
        cause = "chain" if before is not None and not stem_cancels(before) else "file"
        if label in PHASE_CONFLICT_BANDS:
            fix = (f"{target}'s {label} is anti-phase: fold it to mono, utility bass_mono at "
                   f"{max(_band_top_hz(label), 120.0):.0f} Hz")
        else:
            fix = stem_move(row, label, devices)
        if cause == "chain" and devices:
            fix = f"its chain makes it ({', '.join(devices)}): narrow that; or " + fix
        out.append(PhaseIssue(target=target, band=label, kind="cancels_in_mono", cause=cause,
                              corr=band.corr, below_zero_pct=band.below_zero_pct,
                              mono_db=band.mono_db, fix=fix))
    return out


def side_shares(stem_sums: dict[str, WindowSums], label: str,
                entering: Optional[WindowSums] = None) -> list[tuple[str, float]]:
    """The stems whose side signal fills `label`, as (stem, share of the band's side), widest
    first, those under SIDE_SHARE_MIN left out - who a master band's width comes from. The share
    is of the side of the mix going into the master (`entering`) when it is given, so a reading
    of one named stem does not hand it the whole band."""
    b = BAND_LABELS.index(label)

    def side_of(s: WindowSums) -> float:
        return float(((s.ll[:, b] + s.rr[:, b] - 2.0 * s.lr[:, b]) / 4.0).sum())

    side = {sid: side_of(s) for sid, s in stem_sums.items()}
    total = side_of(entering) if entering is not None else sum(v for v in side.values() if v > 0)
    if total <= _TINY:
        return []
    return sorted(((sid, v / total) for sid, v in side.items() if v / total >= SIDE_SHARE_MIN),
                  key=lambda item: (-item[1], item[0]))


def master_issues(row: PhaseTarget, entering: dict[str, PhaseBand], entering_corr: Optional[float],
                  devices: list[str], from_stems: dict[str, list[str]],
                  moves: dict[tuple[str, str], str]) -> list[PhaseIssue]:
    """The master's issues: one side flipped, or a band that cancels (MASTER_CANCEL_TIME_PCT) -
    from its own devices, or from the mix going in (then the stems it comes from, and the move on
    each: `moves[(stem, band)]`)."""
    if inverted(row.corr, row.bands):
        mix = inverted(entering_corr, entering)
        fix = ("a stem or group plays one side upside down - flip it back on THAT channel (its "
               "own issue says which); a one-channel flip on the master is refused" if mix else
               "a device on the master flips one side - switch its invert_left / invert_right off")
        return [PhaseIssue(target="master", kind="inverted_channel",
                           cause="mix" if mix else "master_chain",
                           corr=row.corr if row.corr is not None else -1.0,
                           from_stems=from_stems.get("*", []), fix=fix)]
    out: list[PhaseIssue] = []
    for label, band in row.bands.items():
        if not master_cancels(band):
            continue
        went_in = entering.get(label)
        if went_in is not None and not master_cancels(went_in):
            stems: list[str] = []
            fix = (f"the master's devices make {label} cancel - the mix going in does not "
                   f"({went_in.corr:+.2f}, below 0 {went_in.below_zero_pct:.0f} % of the time): "
                   f"bring {', '.join(devices) or 'the master widening'} back toward 1.00 there")
            cause = "master_chain"
        else:
            stems = from_stems.get(label, [])
            said = [moves[(s, label)] for s in stems if (s, label) in moves]
            fix = ((f"the mix going in already cancels in {label}; it comes from "
                    + "; ".join(said) + ". Fix the first, then read again")
                   if said else
                   f"the mix going in already cancels in {label} - the stems' readings say where")
            fix += " - never by narrowing the whole master"
            cause = "mix"
        out.append(PhaseIssue(target="master", band=label, kind="cancels_in_mono", cause=cause,
                              corr=band.corr, below_zero_pct=band.below_zero_pct,
                              mono_db=band.mono_db, from_stems=stems, fix=fix))
    return out


def master_at_risk(audio: np.ndarray, sample_rate: int) -> list[str]:
    """The master's bands that cancel ("inverted" when one side is upside down) - the render's
    `mono_compatibility_risk` flag, on the same rule `analyze_phase` judges the master by."""
    sums = WindowSums(audio, sample_rate)
    bands = band_reading(sums)
    if inverted(broadband_corr(sums), bands):
        return ["inverted"]
    return [label for label, band in bands.items() if master_cancels(band)]


def conflicts(stem_sums: dict[str, WindowSums]) -> list[PhaseConflict]:
    """Pairs of stems whose centre signals cancel each other in a low band (PhaseConflict)."""
    ids = sorted(sid for sid, s in stem_sums.items() if s.mid is not None)
    if len(ids) < 2:
        return []
    first = stem_sums[ids[0]]
    frames = min(stem_sums[sid].frames for sid in ids)
    freqs = np.fft.rfftfreq(first.nfft, 1.0 / first.sample_rate)[first.mid_index]
    span = 10.0 ** (-PHASE_CONFLICT_LEVEL_SPAN_DB / 10.0)
    out: list[PhaseConflict] = []
    for label in PHASE_CONFLICT_BANDS:
        b = BAND_LABELS.index(label)
        bins = (freqs >= BAND_EDGES_HZ[b]) & (freqs < BAND_EDGES_HZ[b + 1])
        if not bins.any():
            continue
        spec = np.stack([stem_sums[sid].mid[:frames][:, bins] for sid in ids]
                        ).astype(np.complex128)                       # stems x frames x bins
        power = np.sum(np.abs(spec) ** 2, axis=2)                      # stems x frames
        sounding = np.stack([_sounding(p) for p in power])
        cross = np.real(np.einsum("ifk,jfk->ijf", spec, np.conj(spec)))   # stems x stems x frames
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pi, pj = power[i], power[j]
                both = sounding[i] & sounding[j] & (np.minimum(pi, pj) >= span * np.maximum(pi, pj))
                count = int(both.sum())
                if (count < PHASE_CONFLICT_MIN_WINDOWS
                        or 100.0 * count / frames < PHASE_CONFLICT_MIN_OVERLAP_PCT):
                    continue
                si, sj = float(pi[both].sum()), float(pj[both].sum())
                sx = float(cross[i, j][both].sum())
                corr = sx / max(math.sqrt(si * sj), _TINY)
                sum_db = 10.0 * math.log10(max(si + sj + 2.0 * sx, _TINY) / max(si + sj, _TINY))
                if corr > PHASE_CONFLICT_CORRELATION or sum_db > PHASE_CONFLICT_SUM_DB:
                    continue
                lag_ms, at_lag = _best_lag(spec[i][both], spec[j][both], first.mid_index[bins],
                                           first)
                a, c = ids[i], ids[j]
                if lag_ms is not None:
                    early, late = (c, a) if lag_ms > 0 else (a, c)
                    fix = (f"{early} arrives {abs(lag_ms):.1f} ms early in {label}: delay it "
                           f"{abs(lag_ms):.1f} ms more (track.set_delay on {early}: its delay_ms "
                           f"+ {abs(lag_ms):.1f}) to line it up with {late} - a polarity flip "
                           "would only half-fix it")
                else:
                    quieter = a if si < sj else c
                    fix = (f"opposite polarity in {label}: flip {quieter}'s polarity "
                           f"(utility invert: true on {quieter}), then read again")
                out.append(PhaseConflict(
                    stems=[a, c], band=label, corr=round(corr, 2), sum_db=round(sum_db, 1),
                    overlap_pct=round(100.0 * count / frames, 1),
                    lag_ms=lag_ms, corr_at_lag=at_lag, fix=fix))
    out.sort(key=lambda conflict: conflict.sum_db)
    return out


def _best_lag(spec_a: np.ndarray, spec_b: np.ndarray, index: np.ndarray, sums: WindowSums
              ) -> tuple[Optional[float], Optional[float]]:
    """Where `b` lines up with `a` best within PHASE_LAG_MAX_MS, from their band-limited cross-
    spectrum (summed over the windows both sound in; `index` = those bins in the full spectrum).
    A positive peak at lag d means b is a's sound d ms EARLY. (None, None) when flipping one of
    them at zero lag does as well."""
    nfft, rate = sums.nfft, sums.sample_rate
    full = np.zeros(nfft // 2 + 1, dtype=np.complex128)
    auto_a, auto_b = np.zeros_like(full), np.zeros_like(full)
    full[index] = np.sum(spec_a * np.conj(spec_b), axis=0)
    auto_a[index] = np.sum(np.abs(spec_a) ** 2, axis=0)
    auto_b[index] = np.sum(np.abs(spec_b) ** 2, axis=0)
    r = np.fft.irfft(full, n=nfft)
    norm = math.sqrt(max(float(np.fft.irfft(auto_a, n=nfft)[0]), _TINY)
                     * max(float(np.fft.irfft(auto_b, n=nfft)[0]), _TINY))
    reach = int(round(PHASE_LAG_MAX_MS * rate / 1000.0))
    lags = np.concatenate([np.arange(0, reach + 1), np.arange(-reach, 0)])
    values = np.concatenate([r[:reach + 1], r[-reach:]]) / norm
    at_zero = float(values[0])
    k = int(np.argmax(values))
    best, lag_ms = float(values[k]), 1000.0 * float(lags[k]) / rate
    if (abs(lag_ms) >= PHASE_LAG_MIN_MS and best >= -PHASE_CONFLICT_CORRELATION
            and best >= abs(at_zero) + PHASE_LAG_MIN_GAIN):
        return round(lag_ms, 2), round(best, 2)
    return None, None


def _flip_side(audio: np.ndarray, sample_rate: int, others: list[WindowSums]) -> str:
    """Which side to flip back on a channel that plays one side upside down: the one that leaves
    its centre in the SAME polarity as the rest of the mix in the low bands (flipping the right
    makes the centre (L - R) / 2, flipping the left its negative). The right when it cannot tell."""
    a = _stereo(audio)
    side = (a[:, 0] - a[:, 1]) / 2.0
    mine = WindowSums(np.stack([side, side], axis=1), sample_rate, keep_mid=True)
    rest = [o for o in others if o.mid is not None]
    if not rest:
        return "invert_right"
    frames = min([mine.frames] + [o.frames for o in rest])
    total = sum(o.mid[:frames].astype(np.complex128) for o in rest)
    mid = mine.mid[:frames].astype(np.complex128)
    cross = float(np.real(np.sum(mid * np.conj(total))))
    norm = math.sqrt(float(np.sum(np.abs(mid) ** 2)) * float(np.sum(np.abs(total) ** 2)))
    if norm <= _TINY or abs(cross) / norm < 0.1:
        return "invert_right"
    return "invert_right" if cross > 0 else "invert_left"


# ---------------------------------------------------------------- the session's reading

def phase_reading(session, stem_ids: Optional[list[str]] = None):
    """`analyze_phase`: the master, every stem and group in the mix (or only the stems and groups
    `stem_ids` names), what cancels and why, and the stems that cancel each other."""
    from . import dsp
    from .models import PhaseAnalysis

    asked = list(stem_ids or [])
    bus_ids = [name for name in asked if name in session.buses and name not in session.stems]
    names = [name for name in asked if name not in bus_ids] if stem_ids else sorted(
        sid for sid, state in session.stems.items()
        if not state["muted"] and not state["excluded"])
    for name in names:
        session.applied_chain(name)  # raises KeyError for unknown stems
    if not stem_ids:
        bus_ids = [bus_id for bus_id, bus in sorted(session.buses.items()) if not bus.get("muted")]
    if not names and not bus_ids:
        return PhaseAnalysis(window_s=PHASE_WINDOW_S, clean=True, note="no stems in the mix")
    cache: dict = {}
    rows: dict[str, PhaseTarget] = {}
    stem_sums: dict[str, WindowSums] = {}
    rate = session.project_sample_rate()
    for sid in names:
        audio, rate = dsp.render_stem(session, sid, _cache=cache)
        stem_sums[sid] = WindowSums(audio, rate, keep_mid=True)
        rows[sid] = PhaseTarget(target=sid, kind="stem", corr=broadband_corr(stem_sums[sid]),
                                bands=band_reading(stem_sums[sid]))
    mix, rate = dsp.render_master(session, _cache=cache)
    entering, _ = dsp.render_master(session, pre_master=True, _cache=cache)
    entering_sums = WindowSums(entering, rate)
    entering_bands = band_reading(entering_sums)
    master = PhaseTarget(target="master", kind="master")
    master_sums = WindowSums(mix, rate)
    master.corr, master.bands = broadband_corr(master_sums), band_reading(master_sums)

    # Who each cancelling master band comes from: the stems that carry its side.
    from_stems: dict[str, list[str]] = {}
    for label, band in master.bands.items():
        if master_cancels(band):
            from_stems[label] = [sid for sid, _ in side_shares(stem_sums, label,
                                                                entering_sums)][:3]
    named = {s for stems in from_stems.values() for s in stems}

    issues: list[PhaseIssue] = []
    moves: dict[tuple[str, str], str] = {}
    for sid in names:
        row, sums, state = rows[sid], stem_sums[sid], session.stems[sid]
        delay = channel_delay(sums)
        own = (inverted(row.corr, row.bands) or delay is not None
               or any(stem_cancels(b) for b in row.bands.values()))
        devices = stereo_devices(session.applied_chain(sid))
        file_corr, file_bands = None, None
        if own or sid in named:
            file_audio, _, _ = dsp._stem_file(session, sid, {})
            file_sums = WindowSums(file_audio, rate)
            file_corr, file_bands = broadband_corr(file_sums), band_reading(file_sums)
            row.bands = band_reading(sums, file_sums)
            row.width = round(float(state.get("width", 1.0)), 2)
            width_sums = sums
            if float(state.get("pan", 0.0)) != 0.0:
                # The strip width runs BEFORE the pan: read the channel in front of it.
                width_sums = WindowSums(dsp.render_stem(session, sid, _cache={}, _pre_pan=True)[0],
                                        rate)
            row.mono_safe_width = mono_safe_width(width_sums, row.width)
        for label, stems in from_stems.items():
            if sid in stems:
                moves[(sid, label)] = stem_move(row, label, devices)
        if own:
            flip = "invert_right"
            if inverted(row.corr, row.bands):
                flip = _flip_side(dsp.render_stem(session, sid, _cache=cache)[0], rate,
                                  [s for k, s in stem_sums.items() if k != sid])
            issues += stem_issues(row, file_corr=file_corr, file_bands=file_bands, delay=delay,
                                  devices=devices, flip_side=flip)
    targets = [rows[sid] for sid in names]
    for bus_id in bus_ids:
        try:
            audio, _ = dsp.render_bus(session, bus_id, _cache=cache)
        except (KeyError, ValueError):
            continue
        sums = WindowSums(audio, rate)
        row = PhaseTarget(target=bus_id, kind="bus", corr=broadband_corr(sums),
                          bands=band_reading(sums))
        delay = channel_delay(sums)
        if (inverted(row.corr, row.bands) or delay is not None
                or any(stem_cancels(b) for b in row.bands.values())):
            dry_sums = WindowSums(dsp.render_bus(session, bus_id, _cache=cache, dry=True)[0], rate)
            row.bands = band_reading(sums, dry_sums)
            issues += stem_issues(row, file_corr=broadband_corr(dry_sums),
                                  file_bands=band_reading(dry_sums), delay=delay,
                                  devices=stereo_devices(session.applied_bus_chain(bus_id)))
        targets.append(row)
    from_stems["*"] = [i.target for i in issues if i.kind == "inverted_channel"]
    issues = master_issues(master, entering_bands, broadband_corr(entering_sums),
                           stereo_devices(session.master_chain), from_stems, moves) + issues
    found = conflicts(stem_sums)
    notes = []
    if stem_ids:
        notes.append("only the channels named were read (the master always is)")
    if session.returns:
        notes.append("returns (reverb / delay wets) are wide by design and are not judged on "
                     "their own - they are in the master's reading")
    return PhaseAnalysis(window_s=window_seconds(rate), clean=not issues and not found,
                         master=master, targets=targets, issues=issues, conflicts=found,
                         note="; ".join(notes) or None)
