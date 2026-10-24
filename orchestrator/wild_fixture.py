"""Wild-corpus generator (Phase 50 — docs/05-roadmap.md).

Deterministic projects shaped like the messy real world our fixtures
are not: mixed sample rates and bit depths, mono files, thirty stems, a
six-minute song, unicode / spaced / parenthesised stem names, an
already-clipping stem, a silent stem, DC offset, a near-Nyquist tone, stems
of different lengths in one folder — and one project whose `master.wav`
must produce the engine's DEFINED error (discovery reserves the name), not
a crash.

Like `technique_fixture`, this module is the single source of truth: the
synthesis AND the per-project expectations (`WildSpec.expect_error`,
`stems`) that `wild_eval` asserts against. Nothing large is committed —
audio regenerates deterministically (seeded per project). THIS module's command
line writes it into `test_projects/wild/<name>/` (for the DAW's console legs and a
look by hand); `wild_eval` and `speed_baseline` call `generate()` into their own
work folder and never touch that one (2026-10-07).

    python -m orchestrator.wild_fixture           # (re)generate the corpus
    python -m orchestrator.wild_fixture --list    # print the specs
"""
from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

_REPO = Path(__file__).resolve().parents[1]
WILD_DIR = _REPO / "test_projects" / "wild"

_SUBTYPE = {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}


@dataclass(frozen=True)
class StemSpec:
    name: str                 # filename without .wav (the stem id)
    kind: str                 # tone | noise | beat | silent | clip | dc | nyquist
    sr: int = 48000
    bits: int = 24            # 16 | 24 | 32 (32 = float)
    channels: int = 2
    dur_s: float = 20.0
    freq_hz: float = 220.0    # tone/nyquist fundamental
    #: output scale. Every project must genuinely NEED the "master for
    #: streaming" mix: the first sweep synthesized two projects at ~-14.5
    #: dBFS — already at the streaming target — and the rule agent's honest
    #: "nothing to propose" failed the run-must-mix property. The corpus is
    #: the source of truth, so quiet projects are made quiet HERE.
    level: float = 1.0
    #: the file's extension as written. A hardware recorder writes `TRACK01.WAV`; the DAW lists
    #: `*.wav` without regard to case and the brain listed it with (reader S1·R17).
    ext: str = ".wav"


@dataclass(frozen=True)
class WildSpec:
    name: str
    stems: tuple[StemSpec, ...]
    #: substring of the exception a Session() on this project MUST raise
    #: (None = must construct cleanly)
    expect_error: str | None = None
    #: files in the folder that discovery must LEAVE OUT, and name (`Session.left_out_wavs`):
    #: the folder opens with the rest. A `StemSpec` whose file is listed here is on disk and
    #: is not a stem.
    left_out: tuple[str, ...] = ()
    note: str = ""


def _t(sr: int, dur_s: float) -> np.ndarray:
    return np.arange(int(round(sr * dur_s))) / sr


def _synth(spec: StemSpec, rng: np.random.Generator) -> np.ndarray:
    t = _t(spec.sr, spec.dur_s)
    if spec.kind == "silent":
        y = np.zeros_like(t)
    elif spec.kind == "tone":
        # phrased tone with a slow amplitude envelope, so runs have dynamics
        env = 0.5 + 0.5 * np.sin(2 * np.pi * t / max(4.0, spec.dur_s / 4))
        y = 0.25 * env * np.sin(2 * np.pi * spec.freq_hz * t)
    elif spec.kind == "noise":
        y = 0.15 * rng.standard_normal(len(t))
        # band-limit crudely so it isn't full-spectrum harshness
        y = np.convolve(y, np.ones(8) / 8, mode="same")
    elif spec.kind == "beat":
        y = np.zeros_like(t)
        period = int(spec.sr * 0.5)
        for i in range(0, len(t), period):
            n = min(int(spec.sr * 0.09), len(t) - i)
            y[i:i + n] += 0.8 * np.exp(-np.linspace(0, 9, n)) \
                * np.sin(2 * np.pi * 70.0 * _t(spec.sr, n / spec.sr))
    elif spec.kind == "clip":
        # deliberately OVER full scale in float; int formats will saturate
        y = 1.4 * np.sin(2 * np.pi * spec.freq_hz * t)
    elif spec.kind == "dc":
        y = 0.3 + 0.1 * np.sin(2 * np.pi * spec.freq_hz * t)
    elif spec.kind == "nyquist":
        y = 0.2 * np.sin(2 * np.pi * (spec.sr * 0.47) * t)
    else:  # pragma: no cover - spec typo guard
        raise ValueError(f"unknown stem kind {spec.kind!r}")
    y = y * spec.level
    if spec.channels == 2:
        # slight decorrelation so stereo isn't two identical channels
        other = np.roll(y, int(spec.sr * 0.004))
        return np.stack([y, other], axis=1)
    return y[:, None]


PROJECTS: tuple[WildSpec, ...] = (
    WildSpec("tiny_two_30s", (
        StemSpec("voice", "tone", dur_s=30, freq_hz=330),
        StemSpec("beat", "beat", dur_s=30),
    ), note="the floor: two stereo 48k/24 stems, 30 s"),
    WildSpec("lofi_16_441", (
        StemSpec("keys", "tone", sr=44100, bits=16, freq_hz=262),
        StemSpec("bass", "tone", sr=44100, bits=16, freq_hz=55),
        StemSpec("drums", "beat", sr=44100, bits=16),
    ), note="16-bit 44.1 kHz — the bounce-from-2009 project"),
    WildSpec("hires_96k_f32", (
        StemSpec("lead", "tone", sr=96000, bits=32, dur_s=45, freq_hz=440, level=0.35),
        StemSpec("pad", "noise", sr=96000, bits=32, dur_s=45, level=0.35),
        StemSpec("bass", "tone", sr=96000, bits=32, dur_s=45, freq_hz=41, level=0.35),
        StemSpec("perc", "beat", sr=96000, bits=32, dur_s=45, level=0.35),
    ), note="96 kHz float — the audiophile session (quiet: the mix must have work)"),
    WildSpec("mono_only", (
        StemSpec("vox", "tone", channels=1, freq_hz=294, level=0.35),
        StemSpec("gtr", "noise", channels=1, level=0.35),
        StemSpec("bass", "tone", channels=1, freq_hz=49, level=0.35),
        StemSpec("kick", "beat", channels=1, level=0.35),
    ), note="every stem mono — width/imaging code paths on 1-ch input "
            "(quiet: the mix must have work)"),
    WildSpec("many_stems_30", tuple(
        StemSpec(f"layer_{i:02d}",
                 ("tone", "noise", "beat")[i % 3],
                 bits=16, channels=1, dur_s=12,
                 freq_hz=110.0 * (1 + i % 8), level=0.12)
        for i in range(30)
    ), note="thirty stems at sane per-layer levels — session-scale state, "
            "guardrail budgets. (level 0.12: thirty FULL-scale layers summed "
            "to +9 dBFS over, clipping the 24-bit console bounce so hard the "
            "render smoke's -6 dB device check had nothing to measure — that "
            "is not real-world-shaped, and the clipping case has its own "
            "project.)"),
    WildSpec("long_6min", (
        StemSpec("song", "tone", sr=44100, bits=16, channels=1,
                 dur_s=360, freq_hz=196),
        StemSpec("pulse", "beat", sr=44100, bits=16, channels=1, dur_s=360),
    ), note="six minutes — analysis windows, render time, memory"),
    WildSpec("unicode_names", (
        StemSpec("Vökal Läad", "tone", freq_hz=349),
        StemSpec("ドラム", "beat"),
        StemSpec("bass gitarre (DI)", "tone", freq_hz=62),
        StemSpec("hi—hat & perc", "noise"),
    ), note="unicode, spaces, parens, an em-dash and an ampersand in stem ids"),
    WildSpec("master_trap", (
        StemSpec("voice", "tone", freq_hz=330),
        StemSpec("beat", "beat"),
        StemSpec("master", "noise"),
    ), left_out=("master.wav",),
       note="master.wav (the reference mix, usually) is LEFT OUT and named - the folder opens "
            "with the other stems, in both engines; it never drives the master chain "
            "(the suno_demo oracle trap) and never refuses the project (reader S1.R18)"),
    WildSpec("pathological_audio", (
        StemSpec("hot", "clip", freq_hz=220),
        StemSpec("nothing", "silent"),
        StemSpec("leaning", "dc", freq_hz=110),
        StemSpec("edge", "nyquist"),
    ), note="clipping, silence, DC offset, near-Nyquist — analyzer poison"),
    WildSpec("mixed_rates", (
        StemSpec("a_441", "tone", sr=44100, freq_hz=262),
        StemSpec("b_48k", "tone", sr=48000, freq_hz=330),
        StemSpec("c_96k", "beat", sr=96000),
    ), note="three sample rates in ONE project folder"),
    WildSpec("mixed_lengths", (
        StemSpec("short", "beat", dur_s=10),
        StemSpec("medium", "tone", dur_s=30, freq_hz=220),
        StemSpec("long", "noise", dur_s=60),
    ), note="stems of different lengths — sum length, ranged windows, tails"),
    WildSpec("recorder_upper_case", (
        StemSpec("TRACK01", "tone", freq_hz=247, level=0.35, ext=".WAV"),
        StemSpec("TRACK02", "beat", level=0.35, ext=".WAV"),
        StemSpec("Track03", "noise", level=0.35, ext=".Wav"),
    ), note="a hardware recorder's folder: the extension in upper and mixed case "
            "(the lanes played in the DAW and the brain held no stem - reader S1·R17)"),
)


def generate(out_dir: Path = WILD_DIR, only: set[str] | None = None) -> list[Path]:
    made = []
    for i, spec in enumerate(PROJECTS):
        if only and spec.name not in only:
            continue
        proj = out_dir / spec.name
        if proj.exists():
            shutil.rmtree(proj)
        proj.mkdir(parents=True)
        rng = np.random.default_rng(1000 + i)
        for stem in spec.stems:
            audio = _synth(stem, rng)
            sf.write(str(proj / f"{stem.name}{stem.ext}"), audio, stem.sr,
                     subtype=_SUBTYPE[stem.bits], format="WAV")
        made.append(proj)
    return made


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", type=Path, default=WILD_DIR)
    ap.add_argument("--project", action="append")
    args = ap.parse_args(argv)
    if args.list:
        for s in PROJECTS:
            rates = sorted({st.sr for st in s.stems})
            bits = sorted({st.bits for st in s.stems})
            print(f"  {s.name:20s} {len(s.stems):2d} stems  sr={rates} bits={bits}"
                  + (f"  EXPECT ERROR: {s.expect_error}" if s.expect_error else ""))
            if s.note:
                print(f"      {s.note}")
        return 0
    made = generate(args.out, set(args.project) if args.project else None)
    for p in made:
        wavs = [f for f in p.iterdir() if f.suffix.lower() == ".wav"]
        n = len(wavs)
        mb = sum(f.stat().st_size for f in wavs) / 1e6
        print(f"  {p.name:20s} {n:2d} wavs  {mb:6.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
