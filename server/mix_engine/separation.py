"""Stem separation at the import boundary (Phase 29).

A **user** import-time operation — not an agent capability and not an MCP tool
(see the hard invariants in CLAUDE.md and docs/05-roadmap.md Phase 29). A single
mixed track is separated once, on entry, into named stem wavs; from there the
engine treats them as ordinary stems and the render path is untouched. So
separation sits *outside* the bit-identical **render** invariant — but the
pipeline is itself deterministic: same input + same model => identical stems on
re-run (we pin onnxruntime to a single thread + the CPU provider for exactly
this reason).

Model (since 2026-08-13): **htdemucs v4** via the ``demucs-onnx`` package
(MIT code + MIT weights, parity-verified ONNX exports of Meta's MIT
Hybrid-Transformer Demucs; pure numpy + onnxruntime, **no torch** — keeps the
sidecar PyInstaller-bundleable per docs/07-packaging.md). Real 4-stem
(vocals/drums/bass/other) with 2-stem derived as ``accompaniment = drums +
bass + other``; weights download on first use into ``<data_dir>/models/demucs``.
This closed the Phase 29 "4-stem open" gap and audibly upgraded the split.

The **legacy Spleeter** path (k2-fsa / sherpa-onnx conversions, one UNet per
source, MIT, trained on Deezer's own licensed catalog) is kept behind
``kind="onnx"`` for envs that bundle its models. Its inference pipeline here
reproduces sherpa-onnx's ``scripts/spleeter/separate.py`` in numpy/scipy: STFT
(n_fft 4096 / hop 1024 / periodic Hann), keep the low 1024 freq bins, run each
source UNet over 512-frame segments, form the ratio mask
``(Sᵢ² + ε/2) / (Σⱼ Sⱼ² + ε)`` (scale-invariant), apply it to the mix's complex
STFT (bins ≥ 1024 zeroed, as Spleeter does) and ISTFT back.

Audio convention matches the rest of the engine (audio-dsp skill): float32,
shape ``(frames, channels)``. ``onnxruntime`` is imported **lazily** (inside the
ONNX separator), so this module — and the test suite — import without it; tests
drive the dependency-free ``FakeSeparator``.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.signal import resample_poly

# Source-name layouts. Order is the canonical stem order for each checkpoint.
FOUR_STEM: tuple[str, ...] = ("vocals", "drums", "bass", "other")
TWO_STEM: tuple[str, ...] = ("vocals", "accompaniment")

# Spleeter's native training/operating sample rate. Inputs at other rates are
# resampled deterministically (resample_poly, the Phase 15 path) for inference,
# then the separated stems are resampled back to the caller's rate.
MODEL_SAMPLE_RATE = 44100

# STFT geometry Spleeter operates on (samples @ 44.1 kHz) + the model's fixed
# spectrogram crop (sherpa-onnx separate.py): keep the low 1024 freq bins, feed
# 512-frame time segments. Bins above 1024 (~11 kHz) are zeroed in the output.
_STFT_FRAME = 4096
_STFT_HOP = 1024
_SPLEETER_BINS = 1024
_SPLEETER_SEG = 512
_FULL_BINS = _STFT_FRAME // 2 + 1   # 2049 one-sided bins
_MASK_EPS = 1e-10


def stem_names(stems: int) -> tuple[str, ...]:
    """Canonical source names for a 2- or 4-stem split."""
    if stems == 4:
        return FOUR_STEM
    if stems == 2:
        return TWO_STEM
    raise ValueError(f"stems must be 2 or 4, got {stems}")


def _as_2d_float32(audio: np.ndarray) -> np.ndarray:
    """Engine convention: float32, shape (frames, channels). 1-D -> (N, 1)."""
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim != 2:
        raise ValueError(f"expected (frames, channels) audio, got shape {arr.shape}")
    return arr


@runtime_checkable
class Separator(Protocol):
    """Audio -> named stems. Pure (no file I/O, no session coupling): the
    gateway owns writing + registering the stem wavs (Phase 29 import boundary).
    Implementations MUST be deterministic for a fixed input."""

    stems: int
    names: tuple[str, ...]

    def separate(self, audio: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
        ...


# --------------------------------------------------------------------------- #
# Deterministic, dependency-free separator                                    #
# --------------------------------------------------------------------------- #

def _complementary_bands(audio: np.ndarray, sample_rate: int, n: int) -> list[np.ndarray]:
    """Split ``audio`` (frames, channels) into ``n`` contiguous rFFT frequency
    bands whose sum reconstructs the input to float tolerance. Partitioning the
    bins (each bin in exactly one band) makes the split exactly complementary —
    the same idea the multiband split uses (audio-dsp skill), so sum-back holds
    and the result is deterministic."""
    frames = audio.shape[0]
    spec = np.fft.rfft(audio, axis=0)
    n_bins = spec.shape[0]
    # Log-spaced bin edges from ~20 Hz to Nyquist; band 0 also carries DC..edge0.
    nyquist = sample_rate / 2.0
    lo = max(20.0, sample_rate / frames) if frames else 20.0
    edges_hz = np.geomspace(lo, nyquist, n + 1)
    bin_hz = np.fft.rfftfreq(frames, d=1.0 / sample_rate)
    bands: list[np.ndarray] = []
    assigned = np.zeros(n_bins, dtype=bool)
    for b in range(n):
        mask = (bin_hz >= edges_hz[b]) & (bin_hz < edges_hz[b + 1])
        if b == 0:
            mask |= bin_hz < edges_hz[0]          # fold DC..lo into the first band
        if b == n - 1:
            mask |= bin_hz >= edges_hz[n]          # fold Nyquist into the last band
        mask &= ~assigned                          # never double-count a bin
        assigned |= mask
        banded = np.where(mask[:, None], spec, 0.0)
        bands.append(np.fft.irfft(banded, n=frames, axis=0).astype(np.float32))
    return bands


class FakeSeparator:
    """Deterministic, dependency-free separator for tests and as a graceful
    dev fallback when the ONNX model isn't bundled. It splits the spectrum into
    complementary bands mapped (low->high) onto the stem names — *not* musically
    meaningful, but it is deterministic, returns the correct named stems, and
    sums back to the input, which is exactly what the import/render/undo
    plumbing tests need. Never ships as the product separator."""

    def __init__(self, stems: int = 4) -> None:
        self.stems = stems
        self.names = stem_names(stems)

    def separate(self, audio: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
        audio = _as_2d_float32(audio)
        bands = _complementary_bands(audio, sample_rate, len(self.names))
        return {name: band for name, band in zip(self.names, bands)}


# --------------------------------------------------------------------------- #
# Spleeter ONNX separator (the product path)                                  #
# --------------------------------------------------------------------------- #

class ModelNotAvailable(RuntimeError):
    """Raised when the requested separation model (or onnxruntime) is missing —
    the gateway turns this into a clear user-facing message and the opt-in
    real-model test skips on it."""


@dataclass(frozen=True)
class _ModelSpec:
    checkpoint: str                       # "2stems" | "4stems"
    names: tuple[str, ...]
    files: dict[str, str]                 # source name -> onnx filename (one UNet/source)
    sha256: dict[str, str] = field(default_factory=dict)  # source -> hash ("" => skip)


_MODELS: dict[int, _ModelSpec] = {
    2: _ModelSpec("2stems", TWO_STEM, {
        "vocals": "spleeter_2stems_vocals.onnx",
        "accompaniment": "spleeter_2stems_accompaniment.onnx",
    }, sha256={
        # k2-fsa/sherpa-onnx Spleeter 2stems conversion (MIT). Pinned 2026-06-13.
        "vocals": "bdc16ab6bf6117ddd4842c19e80e40e2be188fc555295064d424616b0224ac97",
        "accompaniment": "671ace17acd3720674a2bc14de32ac6292453dec20d9eb0ba4255d4ad8e3d8c0",
    }),
    4: _ModelSpec("4stems", FOUR_STEM, {
        "vocals": "spleeter_4stems_vocals.onnx",
        "drums": "spleeter_4stems_drums.onnx",
        "bass": "spleeter_4stems_bass.onnx",
        "other": "spleeter_4stems_other.onnx",
    }),
}


def _default_data_dir() -> Path:
    """App-support dir for models/state — mirrors orchestrator.calibration.
    data_dir() (the engine must not import the orchestrator). MIX_AGENT_DATA_DIR
    overrides it (tests, the desktop shell, a post-ship model drop-in)."""
    if override := os.environ.get("MIX_AGENT_DATA_DIR"):
        return Path(override)
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "mix-agent"
    return Path.home() / ".mix-agent"


def _model_search_dirs() -> list[Path]:
    """Where bundled/exported models may live: the app-support data dir wins (so
    a user/dev can drop in or update a model without rebuilding the .app — even
    when MIX_AGENT_DATA_DIR is unset, e.g. the packaged sidecar), then the
    in-package bundle dir."""
    return [_default_data_dir() / "models",
            Path(__file__).resolve().parent / "models"]


def resolve_model_files(stems: int) -> dict[str, Path]:
    """Resolve every per-source ONNX file for a checkpoint (sherpa-onnx Spleeter
    is one UNet per source). Searches MIX_AGENT_DATA_DIR/models then the
    in-package bundle. Raises ModelNotAvailable listing what is missing."""
    spec = _MODELS[stems]
    dirs = _model_search_dirs()
    found: dict[str, Path] = {}
    missing: list[str] = []
    for src in spec.names:
        for d in dirs:
            cand = d / spec.files[src]
            if cand.is_file():
                found[src] = cand
                break
        else:
            missing.append(spec.files[src])
    if missing:
        raise ModelNotAvailable(
            f"Spleeter {spec.checkpoint} model file(s) {missing} not found in "
            f"{[str(d) for d in dirs]} — see server/mix_engine/models/README.md "
            "(place them in the bundle dir or set MIX_AGENT_DATA_DIR).")
    return found


def _verify_checksum(path: Path, expected_sha256: str) -> None:
    if not expected_sha256 or expected_sha256.startswith("SHA_"):
        return  # not pinned yet; resolution suffices
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != expected_sha256:
        raise ModelNotAvailable(
            f"checksum mismatch for {path.name}: expected {expected_sha256}, "
            f"got {h.hexdigest()}")


def _hann() -> np.ndarray:
    from scipy.signal.windows import hann
    return hann(_STFT_FRAME, sym=False).astype(np.float64)  # periodic = torch hann_window(periodic=True)


def _stft(audio: np.ndarray) -> np.ndarray:
    """(frames, 2) -> (2, 2049, T) complex. Manual STFT in the **Spleeter/torch
    convention the model was trained on**: periodic Hann, frame 4096, hop 1024,
    **raw (un-normalized) rFFT**. This scale matters: scipy.signal.stft divides
    the magnitude by win.sum() (≈2048×), which starves the UNet and produces a
    thin, bleedy mask — the model needs torch-scale magnitudes. End-padded by one
    frame (as sherpa-onnx separate.py does); center=False framing."""
    win = _hann()
    x = np.ascontiguousarray(audio.T.astype(np.float64))      # (2, frames)
    x = np.pad(x, ((0, 0), (0, _STFT_FRAME)))
    n_frames = 1 + max(0, x.shape[1] - _STFT_FRAME) // _STFT_HOP
    out = np.empty((x.shape[0], _FULL_BINS, n_frames), dtype=np.complex128)
    for i in range(n_frames):
        s = i * _STFT_HOP
        out[:, :, i] = np.fft.rfft(x[:, s:s + _STFT_FRAME] * win, axis=1)
    return out


def _istft(spec: np.ndarray, n_frames: int) -> np.ndarray:
    """(2, 2049, T) complex -> (n_frames, 2) float32. Weighted overlap-add
    (synthesis Hann + divide by Σ win²) — the proper inverse of _stft; an
    unmodified STFT reconstructs to unity, so no torch-style 2/3 fudge."""
    win = _hann()
    n_ch, _, n_seg = spec.shape
    length = (n_seg - 1) * _STFT_HOP + _STFT_FRAME
    acc = np.zeros((n_ch, length), dtype=np.float64)
    wsum = np.zeros(length, dtype=np.float64)
    win_sq = win ** 2
    for i in range(n_seg):
        s = i * _STFT_HOP
        acc[:, s:s + _STFT_FRAME] += np.fft.irfft(spec[:, :, i], n=_STFT_FRAME, axis=1) * win
        wsum[s:s + _STFT_FRAME] += win_sq
    out = (acc / np.maximum(wsum, 1e-8)).T.astype(np.float32)   # (length, 2)
    if out.shape[0] < n_frames:
        out = np.pad(out, ((0, n_frames - out.shape[0]), (0, 0)))
    return out[:n_frames]


class SpleeterOnnxSeparator:
    """Spleeter via onnxruntime (sherpa-onnx conversion) — the shipping
    separator. One UNet per source predicts that source's magnitude spectrogram;
    the cross-source ratio mask is applied to the mix's complex STFT and
    inverted. onnxruntime is imported lazily and pinned single-threaded + CPU so
    a separation is deterministic."""

    in_child = True     # minutes of inference: run where a Stop can kill it (separate_in_child)

    def __init__(self, stems: int = 4, verify_checksum: bool = True) -> None:
        self.stems = stems
        self._spec = _MODELS[stems]
        self.names = self._spec.names
        self._paths = resolve_model_files(stems)        # raises ModelNotAvailable
        if verify_checksum:
            for src, p in self._paths.items():
                _verify_checksum(p, self._spec.sha256.get(src, ""))
        self._sessions = None  # lazily created {source: onnxruntime.InferenceSession}

    def _ensure_sessions(self):
        if self._sessions is None:
            try:
                import onnxruntime as ort
            except ImportError as exc:  # pragma: no cover - env without onnxruntime
                raise ModelNotAvailable(
                    "onnxruntime is not installed; it ships with the engine "
                    "sidecar (requirements.txt)") from exc
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1   # determinism: no thread-order nondeterminism
            opts.inter_op_num_threads = 1
            self._sessions = {
                src: ort.InferenceSession(str(p), sess_options=opts,
                                          providers=["CPUExecutionProvider"])
                for src, p in self._paths.items()}
        return self._sessions

    def separate(self, audio: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
        audio = _as_2d_float32(audio)
        n_in = audio.shape[0]
        work = _resample(audio, sample_rate, MODEL_SAMPLE_RATE)
        if work.shape[1] == 1:                 # the model expects stereo
            work = np.repeat(work, 2, axis=1)
        elif work.shape[1] > 2:
            work = work[:, :2]

        Z = _stft(work)                                  # (2, 2049, T) complex
        T = Z.shape[2]
        mag = np.abs(Z[:, :_SPLEETER_BINS, :])           # (2, 1024, T)
        x = np.transpose(mag, (0, 2, 1))                 # (2, T, 1024)
        pad = (-T) % _SPLEETER_SEG
        if pad:
            x = np.pad(x, ((0, 0), (0, pad), (0, 0)))
        n_splits = x.shape[1] // _SPLEETER_SEG
        x = x.reshape(2, n_splits, _SPLEETER_SEG, _SPLEETER_BINS).astype(np.float32)

        sessions = self._ensure_sessions()
        in_names = {src: s.get_inputs()[0].name for src, s in sessions.items()}
        # Per-segment inference bounds memory on full songs (the tensor is
        # 2·n_splits·512·1024 floats otherwise).
        specs = {src: np.empty((2, n_splits, _SPLEETER_SEG, _SPLEETER_BINS),
                               dtype=np.float64) for src in self.names}
        for k in range(n_splits):
            seg = x[:, k:k + 1, :, :]
            for src, sess in sessions.items():
                specs[src][:, k:k + 1, :, :] = sess.run(None, {in_names[src]: seg})[0]

        sum_sq = sum(s ** 2 for s in specs.values()) + _MASK_EPS
        out: dict[str, np.ndarray] = {}
        for src in self.names:
            mask = (specs[src] ** 2 + _MASK_EPS / 2.0) / sum_sq        # (2,n_splits,512,1024)
            m = mask.reshape(2, n_splits * _SPLEETER_SEG, _SPLEETER_BINS)[:, :T, :]
            m = np.transpose(m, (0, 2, 1))                             # (2, 1024, T)
            full = np.zeros((2, _FULL_BINS, T), dtype=np.float64)
            full[:, :_SPLEETER_BINS, :] = m                           # bins ≥1024 -> 0
            wav = _istft(Z * full, work.shape[0])                     # (work_frames, 2)
            wav = _resample(wav, MODEL_SAMPLE_RATE, sample_rate)
            if wav.shape[0] < n_in:
                wav = np.pad(wav, ((0, n_in - wav.shape[0]), (0, 0)))
            out[src] = wav[:n_in].astype(np.float32)
        return out


def _resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio
    g = np.gcd(int(sr_in), int(sr_out))
    return resample_poly(audio, sr_out // g, sr_in // g, axis=0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Demucs ONNX separator (the product path since 2026-08-13)                    #
# --------------------------------------------------------------------------- #

_hub_offline_lock = threading.Lock()
_hub_offline_depth = 0
_hub_offline_was = False


@contextlib.contextmanager
def _hub_offline():
    """Hugging Face's own offline switch, held for the calls inside.

    A model that is on disk is resolved from disk (reader D4·R10). ``hf_hub_download`` asks the
    Hub for the file's metadata on EVERY call before it looks at its cache - a HEAD request
    per resolve path, and a GET the library adds about its caller - so every split said, to a
    third party, that this machine was splitting a track just now, and waited on the network
    before it fell back to the copy it already had. The switch is the library's constant, read
    at call time (the environment variable is read once, at import: setting it here would be
    too late); with it on the library raises before a request is made and resolves from the
    cache. Counted, because two projects' brains can split at once in one process."""
    global _hub_offline_depth, _hub_offline_was
    try:
        from huggingface_hub import constants
    except ImportError:                 # no hub client = nothing that could ask
        yield
        return
    with _hub_offline_lock:
        if _hub_offline_depth == 0:
            _hub_offline_was = bool(constants.HF_HUB_OFFLINE)
            constants.HF_HUB_OFFLINE = True
        _hub_offline_depth += 1
    try:
        yield
    finally:
        with _hub_offline_lock:
            _hub_offline_depth -= 1
            if _hub_offline_depth == 0:
                constants.HF_HUB_OFFLINE = _hub_offline_was


class DemucsOnnxSeparator:
    """htdemucs v4 via the ``demucs-onnx`` package (MIT code + MIT weights;
    pure numpy + onnxruntime, no torch) — the shipping separator, replacing
    Spleeter. Default model is single-file ``htdemucs`` (one ~166 MB fp16
    download, all 4 sources in one inference); ``MIX_AGENT_DEMUCS_MODEL`` can
    select ``htdemucs_ft`` (specialist bag, best quality, 4 downloads).

    Weights download on first use from Hugging Face into
    ``<data_dir>/models/demucs`` — so construction stays cheap (the gateway
    calls the factory on the event loop) and the download happens inside
    ``separate()``, which the gateway already runs off-loop. That first use is
    the ONLY time the network is asked: once the weights are on disk every
    resolve runs under ``_hub_offline`` and sends nothing. A resolve/download
    failure raises ModelNotAvailable, which the gateway turns into its clean
    "no_model" message. Inference is pinned to the CPU execution provider with
    fixed chunking, so a separation is deterministic for a fixed input.

    2-stem mode is derived from the same 4-source inference:
    ``accompaniment = drums + bass + other`` (the standard karaoke complement).
    """

    in_child = True     # minutes of inference: run where a Stop can kill it (separate_in_child)

    def __init__(self, stems: int = 4, model: str | None = None) -> None:
        self.stems = stems
        self.names = stem_names(stems)
        self._model = model or os.environ.get("MIX_AGENT_DEMUCS_MODEL", "htdemucs")
        self._precision = os.environ.get("MIX_AGENT_DEMUCS_PRECISION", "fp16weights")
        try:
            import demucs_onnx  # noqa: F401 — availability check only
        except ImportError as exc:
            raise ModelNotAvailable(
                "demucs-onnx is not installed; it ships with the engine "
                "sidecar (requirements.txt)") from exc

    @staticmethod
    def cache_dir() -> Path:
        return _default_data_dir() / "models" / "demucs"

    def models_cached(self) -> bool:
        """True when the weights are already on disk — callers use this to
        announce the first-use download as its OWN progress stage (a bare
        "separating…" while 166 MB quietly pulls reads as a hang)."""
        cache = self.cache_dir()
        if not cache.is_dir():
            return False
        need = 4 if self._model == "htdemucs_ft" else 1
        return sum(1 for _ in cache.rglob("*.onnx")) >= need

    def download_detail(self) -> str:
        """Human line for the download stage (sizes: fp16 ≈166 MB per file)."""
        n = 4 if self._model == "htdemucs_ft" else 1
        mb = n * (166 if self._precision == "fp16weights" else 316)
        return f"AI model {self._model} — about {mb} MB, first use only"

    def fetch_models(self) -> None:
        """Resolve (download on first use) the ONNX weights. Network/cache
        failures become ModelNotAvailable — never a hard crash."""
        from demucs_onnx import inference as dx

        def resolve() -> None:
            if self._model == "htdemucs_ft":
                for src in FOUR_STEM:
                    dx.download_stem_model(src, cache_dir=self.cache_dir(),
                                           precision=self._precision)
            else:
                dx.download_single_model(self._model, cache_dir=self.cache_dir(),
                                         precision=self._precision)

        if self.models_cached():
            try:
                with _hub_offline():
                    resolve()
                return
            except Exception:  # noqa: BLE001 - files there but not a whole cache: ask online
                pass
        try:
            resolve()
        except Exception as exc:
            raise ModelNotAvailable(
                f"htdemucs model {self._model!r} could not be resolved "
                f"(first use needs a network connection to download ~166 MB "
                f"into {self.cache_dir()}): {exc}") from exc

    def separate(self, audio: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
        import tempfile

        import demucs_onnx
        import soundfile as sf

        audio = _as_2d_float32(audio)
        n_in = audio.shape[0]
        self.fetch_models()   # no-op when cached; callers may pre-fetch to stage it
        # The package's public API is path-in / arrays-out; a temp wav keeps us
        # on the supported surface (its loader owns mono->stereo + resampling).
        with tempfile.TemporaryDirectory(prefix="mix-agent-demucs-") as td:
            tmp = Path(td) / "input.wav"
            sf.write(str(tmp), audio, sample_rate, subtype="FLOAT")
            # ...which resolves the model AGAIN itself. It is on disk by now (fetch_models
            # above returned), so that second resolve asks nothing either.
            with _hub_offline():
                raw = demucs_onnx.separate(
                    str(tmp), model=self._model, providers="cpu",
                    precision=self._precision, cache_dir=self.cache_dir(),
                    progress=False)
        # raw: {source: (channels, samples)} at the caller's native rate.
        def to_engine(arr: np.ndarray) -> np.ndarray:
            wav = np.asarray(arr, dtype=np.float32).T          # (frames, ch)
            if wav.shape[0] < n_in:
                wav = np.pad(wav, ((0, n_in - wav.shape[0]), (0, 0)))
            return wav[:n_in]

        if self.stems == 2:
            acc = sum(np.asarray(raw[s], dtype=np.float32)
                      for s in ("drums", "bass", "other"))
            return {"vocals": to_engine(raw["vocals"]), "accompaniment": to_engine(acc)}
        return {name: to_engine(raw[name]) for name in self.names}


def get_separator(stems: int = 4, kind: str | None = None) -> Separator:
    """Factory. ``kind`` defaults to ``MIX_AGENT_SEPARATOR`` env (or "demucs"):
    "demucs" is the product path (htdemucs v4 via demucs-onnx), "onnx" keeps
    the legacy Spleeter path (needs its bundled per-source UNets), "fake"
    forces the deterministic stand-in (tests set this, and it is the dev
    fallback when no model is available)."""
    kind = (kind or os.environ.get("MIX_AGENT_SEPARATOR", "demucs")).lower()
    if kind == "fake":
        return FakeSeparator(stems)
    if kind == "demucs":
        return DemucsOnnxSeparator(stems)
    if kind in ("onnx", "spleeter"):
        return SpleeterOnnxSeparator(stems)
    raise ValueError(
        f"unknown separator kind {kind!r} (use 'demucs', 'onnx', or 'fake')")


# --------------------------------------------------------------------------- #
# The split in a process of its own (P3 batch T2, reader S3·R10)              #
# --------------------------------------------------------------------------- #

class SplitCancelled(Exception):
    """The caller's `cancelled()` came true: the worker was killed and its files removed."""


def worker_argv() -> tuple[list[str], dict]:
    """How this brain starts `separation_worker`, and the environment it runs in. A frozen
    brain has no interpreter to hand a module to: it starts ITSELF with a flag its entry point
    reads before it imports anything else (daw/sidecar/brain_entry.py)."""
    import sys

    env = dict(os.environ)
    if getattr(sys, "frozen", False):
        return [sys.executable, "--separation-worker"], env
    root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return [sys.executable, "-m", "mix_engine.separation_worker"], env


def separate_in_child(src: Path, stems: int, cancelled=None,
                      on_separating=None) -> tuple[dict[str, np.ndarray], int]:
    """Run the split of ``src`` in `separation_worker` and return ``(stems, sample_rate)``.

    ``cancelled()`` is asked ten times a second; true, the worker is KILLED - whatever the model
    was doing - and ``SplitCancelled`` is raised. ``on_separating()`` is called once, when the
    worker says the model is on disk and the inference has begun (or when it ends, if that was
    never seen). ``ModelNotAvailable`` passes through as it does in-process; a worker that
    ends without an answer is a ``RuntimeError`` with its exit code."""
    import json
    import shutil
    import subprocess
    import tempfile
    import time

    from .separation_worker import DONE, STAGE_SEPARATING

    out = Path(tempfile.mkdtemp(prefix="zammoth-split-"))
    argv, env = worker_argv()
    proc = None
    try:
        # stdout is the brain's handshake line's channel: the worker gets none. Its stderr is
        # the brain's (brain.log).
        proc = subprocess.Popen([*argv, str(src), str(out), str(int(stems))], env=env,
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        announced = False
        while proc.poll() is None:
            if cancelled is not None and cancelled():
                raise SplitCancelled()
            if not announced and (out / STAGE_SEPARATING).exists():
                announced = True
                if on_separating:
                    on_separating()
            time.sleep(0.1)
        if cancelled is not None and cancelled():
            raise SplitCancelled()
        try:
            answer = json.loads((out / DONE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise RuntimeError(
                f"the stem split stopped unexpectedly (exit {proc.returncode})") from None
        if not answer.get("ok"):
            if answer.get("kind") == "no_model":
                raise ModelNotAvailable(str(answer.get("message", "")))
            raise RuntimeError(str(answer.get("message") or "the stem split failed"))
        if not announced and on_separating:
            on_separating()
        result = {name: np.load(str(out / f"{name}.npy")) for name in stem_names(stems)}
        return result, int(answer["sample_rate"])
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=10)
            except Exception:  # noqa: BLE001 - killed; nothing more to do about it here
                pass
        shutil.rmtree(out, ignore_errors=True)
