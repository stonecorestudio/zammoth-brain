"""Grade a FINISHED mix session on what a listener notices first (the mixing audit of 2026-11-10).

Free - no model. It reads a project the engineer has already mixed (its stems, its
`session.json`, its exported master) and checks three end states the synthetic gates never saw:

1. **the lead against the rest** - the processed lead stem's loudness against everything else,
   over the stretches where the lead plays (median of 3 s windows). Sized on seven real
   sessions of 2026-11-02: the mixes that passed by ear read -3.5 LU or better, the ones with
   a drowned vocal -5.4 to -6.9;
2. **width against the raw session** - the export is not narrower than the raw stems summed
   (asked for "wider", two of those sessions came back 3 dB narrower);
3. **the delivery** - loudness within 0.5 LU of what was asked, true peak under the ceiling.

The meter here shares nothing with `mix_engine.analysis` (pyloudnorm + scipy), on purpose.

    python -m orchestrator.mix_session_check verify --project <dir> --export <wav> --lead <stem>
                                                    [--lufs -14] [--ceiling -1] [--raw <dir>]
    python -m orchestrator.mix_session_check --self-test

Exit 0 = every check passed, 1 = a check failed, 2 = nothing to run on.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

LEAD_UNDER_MAX_LU = 4.5      # the lead may sit this far under the rest, no further
WIDTH_LOSS_MAX_DB = 1.5      # the export's side-to-mid may be this far under the raw sum's
LOUDNESS_TOLERANCE_LU = 0.5
CEILING_TOLERANCE_DB = 0.05
WINDOW_S = 3.0
LEAD_FLOOR_LUFS = -40.0      # a window where the lead is under this is not "playing"


def _stereo(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    return np.repeat(x, 2, axis=1) if x.shape[1] == 1 else x[:, :2]


def _read(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf
    audio, rate = sf.read(str(path), always_2d=True, dtype="float64")
    return _stereo(audio), rate


def lufs(audio: np.ndarray, rate: int) -> float:
    import pyloudnorm as pyln
    return float(pyln.Meter(rate).integrated_loudness(_stereo(audio)))


def true_peak_dbtp(audio: np.ndarray, oversample: int = 8) -> float:
    from scipy import signal
    peak = max(float(np.max(np.abs(signal.resample_poly(_stereo(audio)[:, c], oversample, 1))))
               for c in range(2))
    return 20.0 * np.log10(max(peak, 1e-12))


def side_minus_mid_db(audio: np.ndarray) -> float:
    x = _stereo(audio)
    mid, side = (x[:, 0] + x[:, 1]) / 2.0, (x[:, 0] - x[:, 1]) / 2.0
    return float(10.0 * np.log10(max(np.mean(side ** 2), 1e-20) / max(np.mean(mid ** 2), 1e-20)))


def lead_against_rest(lead: np.ndarray, rest: np.ndarray, rate: int) -> Optional[float]:
    """Median over the windows where the lead plays of (lead LUFS - rest LUFS); None when the
    lead never plays against anything."""
    import pyloudnorm as pyln
    meter = pyln.Meter(rate)
    lead, rest = _stereo(lead), _stereo(rest)
    n, window, hop = min(len(lead), len(rest)), int(WINDOW_S * rate), rate
    gaps = []
    for start in range(0, n - window, hop):
        a = meter.integrated_loudness(lead[start:start + window])
        b = meter.integrated_loudness(rest[start:start + window])
        if a > LEAD_FLOOR_LUFS and b > -60.0:
            gaps.append(a - b)
    return float(np.median(gaps)) if gaps else None


def _processed_stems(project: Path) -> tuple[dict[str, np.ndarray], int]:
    """Every stem in the mix through its own chain, fader and pan, by the offline engine."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
    from mix_engine import dsp
    from mix_engine.session import Session
    session = Session(project, read_only=True)
    parts, rate = {}, 0
    for stem_id, stem in session.stems.items():
        if stem.get("excluded") or stem.get("muted"):
            continue
        audio, rate = dsp.render_stem(session, stem_id, _cache={})[:2]
        parts[stem_id] = _stereo(audio)
    return parts, rate


def _raw_sum(folder: Path) -> Optional[np.ndarray]:
    total = None
    for wav in sorted(p for p in folder.iterdir() if p.suffix.lower() == ".wav"):
        audio, _ = _read(wav)
        if total is None:
            total = audio.copy()
        else:
            n = max(len(total), len(audio))
            total = np.pad(total, ((0, n - len(total)), (0, 0)))
            total[:len(audio)] += audio
    return total


def check(project: Path, export: Path, *, lead: Optional[str] = None,
          asked_lufs: Optional[float] = None, ceiling_dbtp: Optional[float] = None,
          raw: Optional[Path] = None) -> tuple[list[str], dict]:
    """(failures, figures). A failure is a sentence; no failures = the session passes."""
    failures: list[str] = []
    figures: dict = {}
    audio, rate = _read(export)
    figures["lufs_i"] = round(lufs(audio, rate), 2)
    figures["true_peak_dbtp"] = round(float(true_peak_dbtp(audio)), 2)
    if asked_lufs is not None and abs(figures["lufs_i"] - asked_lufs) > LOUDNESS_TOLERANCE_LU:
        failures.append(f"loudness {figures['lufs_i']:.1f} LUFS, asked {asked_lufs:.1f} "
                        f"(want within {LOUDNESS_TOLERANCE_LU})")
    if ceiling_dbtp is not None and figures["true_peak_dbtp"] > ceiling_dbtp + CEILING_TOLERANCE_DB:
        failures.append(f"true peak {figures['true_peak_dbtp']:.2f} dBTP is over the "
                        f"{ceiling_dbtp:.1f} ceiling")
    raw_sum = _raw_sum(raw or project)
    if raw_sum is not None:
        figures["width_db"] = round(side_minus_mid_db(audio), 1)
        figures["raw_width_db"] = round(side_minus_mid_db(raw_sum), 1)
        if figures["width_db"] < figures["raw_width_db"] - WIDTH_LOSS_MAX_DB:
            failures.append(f"the mix is narrower than the raw session: side-to-mid "
                            f"{figures['width_db']:.1f} dB against {figures['raw_width_db']:.1f} "
                            f"(want no more than {WIDTH_LOSS_MAX_DB} dB under)")
    if lead is not None:
        parts, stem_rate = _processed_stems(project)
        if lead not in parts:
            failures.append(f"no stem '{lead}' in the mix; stems: {sorted(parts)}")
        else:
            n = min(len(v) for v in parts.values())
            rest = sum(v[:n] for k, v in parts.items() if k != lead)
            gap = lead_against_rest(parts[lead][:n], rest, stem_rate)
            figures["lead_minus_rest_lu"] = None if gap is None else round(gap, 1)
            if gap is None:
                failures.append(f"'{lead}' never plays against the rest: nothing to judge")
            elif gap < -LEAD_UNDER_MAX_LU:
                failures.append(f"'{lead}' sits {-gap:.1f} LU under the rest of the mix "
                                f"(want no more than {LEAD_UNDER_MAX_LU}): the lead is buried")
    return failures, figures


# ------------------------------------------------------------------ the checks, proven free

def _fixture(folder: Path, *, lead_db: float = 0.0) -> tuple[Path, np.ndarray, int]:
    """A three-stem project (a mono lead in bursts, a mono beat, a wide pad) and its plain sum."""
    import soundfile as sf
    from scipy import signal
    rate, seconds = 44100, 16
    rng = np.random.default_rng(11)
    t = np.arange(rate * seconds) / rate

    def band(lo, hi, seed):
        sos = signal.butter(4, [lo, hi], "bandpass", fs=rate, output="sos")
        return signal.sosfilt(sos, np.random.default_rng(seed).standard_normal(len(t)))
    gate = (np.sin(2 * np.pi * 0.25 * t) > -0.3).astype(float)
    lead = band(300, 3500, 1) * gate * 0.12 * 10 ** (lead_db / 20.0)
    beat = band(40, 9000, 2) * (0.5 + 0.5 * (np.sin(2 * np.pi * 2.0 * t) > 0.6)) * 0.08
    pad_l, pad_r = band(200, 6000, 3) * 0.06, band(200, 6000, 4) * 0.06
    stems = {"lead": np.stack([lead, lead], 1), "beat": np.stack([beat, beat], 1),
             "pad": np.stack([pad_l, pad_r], 1)}
    folder.mkdir(parents=True, exist_ok=True)
    for name, audio in stems.items():
        sf.write(str(folder / f"{name}.wav"), audio, rate, subtype="PCM_24")
    return folder, sum(stems.values()), rate


def self_test() -> list[str]:
    """Every check passes a good session and fails the one mutation aimed at it. Returns the
    problems found in the CHECKER (empty = proven)."""
    import soundfile as sf
    problems: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        project, mix, rate = _fixture(tmp / "good")
        mix = mix * 10 ** ((-14.0 - lufs(mix, rate)) / 20.0)
        export = tmp / "good.wav"
        sf.write(str(export), mix, rate, subtype="FLOAT")
        ceiling = true_peak_dbtp(mix) + 0.5

        def run(proj=project, exp=export, **kw):
            return check(proj, exp, lead="lead", asked_lufs=-14.0, ceiling_dbtp=ceiling, **kw)[0]
        if (got := run()):
            problems.append(f"a good session fails: {got}")
        buried, _, _ = _fixture(tmp / "buried", lead_db=-9.0)
        if not any("buried" in f for f in run(proj=buried, raw=project)):
            problems.append("a lead 9 dB down is not called buried")
        narrow = tmp / "narrow.wav"
        mono = mix.mean(axis=1)
        sf.write(str(narrow), np.stack([mono, mono], 1), rate, subtype="FLOAT")
        if not any("narrower" in f for f in run(exp=narrow)):
            problems.append("a mono export is not called narrower than the raw session")
        loud = tmp / "loud.wav"
        sf.write(str(loud), mix * 10 ** (3.0 / 20.0), rate, subtype="FLOAT")
        got = run(exp=loud)
        if not any("loudness" in f for f in got) or not any("true peak" in f for f in got):
            problems.append(f"an export 3 dB hot passes loudness or the ceiling: {got}")
    return problems


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m orchestrator.mix_session_check",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("command", nargs="?", choices=["verify"])
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--project")
    parser.add_argument("--export")
    parser.add_argument("--lead")
    parser.add_argument("--raw")
    parser.add_argument("--lufs", type=float)
    parser.add_argument("--ceiling", type=float)
    args = parser.parse_args(argv)
    if args.self_test:
        problems = self_test()
        print("\n".join(problems) or "mix_session_check: the checks are proven "
              "(a good session passes; a buried lead, a narrowed mix and a hot export fail)")
        return 1 if problems else 0
    if args.command != "verify" or not args.project or not args.export:
        parser.print_usage()
        return 2
    project, export = Path(args.project).expanduser(), Path(args.export).expanduser()
    if not project.is_dir() or not export.is_file():
        print(f"nothing to run on: {project} / {export}")
        return 2
    failures, figures = check(project, export, lead=args.lead, asked_lufs=args.lufs,
                              ceiling_dbtp=args.ceiling,
                              raw=Path(args.raw).expanduser() if args.raw else None)
    print(f"{project.name}: {figures}")
    for failure in failures:
        print(f"  FAILS  {failure}")
    if not failures:
        print("  passes")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
