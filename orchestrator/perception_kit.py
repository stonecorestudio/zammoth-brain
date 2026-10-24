"""Perception rating kit (2026-09-02 eval audit).

`perception_eval.py` (Phase 20) measures how well the composite metrics —
harshness_db, mud_db, clarity_index — track HUMAN judgement, by Spearman rank
correlation against a CSV of rated clips. It has never been fed a real
rating: the only CSV it ever saw was one a test wrote about its own
construction. So the constants behind the three numbers the agent reasons
with have never been checked against an ear.

This kit removes every excuse but the listening. From one project it renders
a short excerpt of the master and twenty loudness-matched variants with KNOWN
manipulations (presence boosts and cuts for harshness, low-mid boosts and cuts
for mud, tilt combinations for clarity, plus two level/width distractors that
should NOT move the ratings), shuffles them blind, and writes:

    <out>/clip_NN.wav        the clips, in a seeded random order
    <out>/ratings.csv        the template perception_eval reads: path,harshness,mud,clarity
    <out>/key.json           clip -> manipulation (do not read before rating)
    <out>/README.txt         how to rate

Rate each clip 1-7 on each quality (higher = more of it), leave a cell blank
when unsure, then:

    python -m orchestrator.perception_eval <out>/ratings.csv

Every variant is matched to the base clip's integrated loudness, so a rating
is about timbre, not level. Deterministic: the same project and seed give
byte-identical clips.

What the metrics themselves said about the kit on the demo (2026-09-02, before
any human heard it): `mud_db` follows the low-mid moves cleanly (Spearman 0.94
against the intended directions); `clarity_index` follows its tilt pairs
(0.66); `harshness_db` follows presence boosts and cuts but ALSO swings with
the low-mid moves — a -6 dB low-mid cut reads ~3 dB harsher than the base —
because it measures presence relative to the mids (0.55). The level and width
distractors leave all three untouched. Whether that relativity is how ears
hear harshness is the question the ratings answer.

    python -m orchestrator.perception_kit test_projects/suno_demo --out ~/Desktop/rating_kit
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "server"))

EXCERPT_S = 20.0
RATING_SCALE = "1-7"
#: every clip is matched to this loudness — quiet enough that a +6 dB band never clips
KIT_LUFS = -20.0
SEED = 2026_09_02


@dataclass(frozen=True)
class Variant:
    name: str
    #: (freq_hz, gain_db, q) peaking bands applied in order; shelves use q <= 0
    bands: tuple[tuple[float, float, float], ...]
    #: what the manipulation SHOULD move, for the kit's own self-test
    expects: dict
    #: a level or width change that should move nothing (rating sanity)
    distractor: bool = False
    width: float = 1.0


def variants() -> list[Variant]:
    out = [Variant("base", (), {})]
    for g in (-6.0, -3.0, 3.0, 6.0):
        out.append(Variant(f"presence_{g:+.0f}dB_3k", ((3000.0, g, 1.0),),
                           {"harshness": g}))
    out.append(Variant("presence_+4dB_5k", ((5000.0, 4.0, 1.0),), {"harshness": 4.0}))
    out.append(Variant("presence_-4dB_5k", ((5000.0, -4.0, 1.0),), {"harshness": -4.0}))
    for g in (-6.0, -3.0, 3.0, 6.0):
        out.append(Variant(f"lowmid_{g:+.0f}dB_250", ((250.0, g, 0.8),), {"mud": g}))
    out.append(Variant("lowmid_+4dB_400", ((400.0, 4.0, 0.8),), {"mud": 4.0}))
    # clarity: carve the low mids AND lift presence (clearer), or the reverse (duller)
    for g in (3.0, 6.0):
        out.append(Variant(f"clarity_+{g:.0f}", ((300.0, -g, 0.8), (3500.0, g * 0.5, 1.0)),
                           {"clarity": g, "mud": -g}))
        out.append(Variant(f"clarity_-{g:.0f}", ((300.0, g, 0.8), (3500.0, -g * 0.5, 1.0)),
                           {"clarity": -g, "mud": g}))
    out.append(Variant("mid_+4dB_1k", ((1000.0, 4.0, 1.0),), {}))
    out.append(Variant("harsh_and_muddy_+4", ((3000.0, 4.0, 1.0), (250.0, 4.0, 0.8)),
                       {"harshness": 4.0, "mud": 4.0}))
    out.append(Variant("distractor_level_-3dB", (), {}, distractor=True))
    out.append(Variant("distractor_width_1.3", (), {}, distractor=True, width=1.3))
    assert len(out) == 20, len(out)
    return out


def _excerpt(audio: np.ndarray, sample_rate: int, seconds: float = EXCERPT_S) -> np.ndarray:
    """The loudest `seconds` window (by short-term RMS on a 1 s grid) — the
    part of the song a rating should be about, not the intro."""
    n = int(seconds * sample_rate)
    if audio.shape[0] <= n:
        return audio
    hop = sample_rate
    best, best_rms = 0, -1.0
    for start in range(0, audio.shape[0] - n + 1, hop):
        rms = float(np.sqrt(np.mean(audio[start:start + n] ** 2)))
        if rms > best_rms:
            best, best_rms = start, rms
    return audio[best:best + n]


def _process(audio: np.ndarray, sample_rate: int, variant: Variant) -> np.ndarray:
    from pedalboard import Pedalboard, PeakFilter

    out = audio.astype(np.float32)
    if variant.bands:
        board = Pedalboard([PeakFilter(cutoff_frequency_hz=f, gain_db=g, q=q)
                            for f, g, q in variant.bands])
        out = board(out.T, sample_rate).T
    if variant.width != 1.0 and out.shape[1] == 2:
        mid = 0.5 * (out[:, 0] + out[:, 1])
        side = 0.5 * (out[:, 0] - out[:, 1]) * variant.width
        out = np.stack([mid + side, mid - side], axis=1)
    return out


def _match_loudness(audio: np.ndarray, sample_rate: int, target_lufs: float) -> np.ndarray:
    import pyloudnorm

    meter = pyloudnorm.Meter(sample_rate)
    lufs = meter.integrated_loudness(audio)
    if not np.isfinite(lufs):
        return audio
    gain = 10.0 ** ((target_lufs - lufs) / 20.0)
    out = audio * gain
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out.astype(np.float32)


def build(project_dir: Path, out_dir: Path, seed: int = SEED) -> dict:
    """Render the kit. Returns the key (clip -> manipulation)."""
    import pyloudnorm
    import soundfile as sf

    from mix_engine import dsp
    from mix_engine.session import Session

    out_dir.mkdir(parents=True, exist_ok=True)
    master, sample_rate = dsp.render_master(Session(project_dir, read_only=True))
    base = _excerpt(np.asarray(master, dtype=np.float32), sample_rate)
    # Headroom first: a +6 dB band on a loud master would peak past full scale when
    # matched back to the master's loudness, and the peak clamp would leave it
    # quieter than its siblings — a level cue in a kit built to remove level cues.
    target = KIT_LUFS
    base = _match_loudness(base, sample_rate, target)
    order = list(variants())
    random.Random(seed).shuffle(order)
    key: dict = {"project": str(project_dir), "excerpt_s": EXCERPT_S,
                 "target_lufs": round(target, 2), "seed": seed, "clips": {}}
    rows = []
    for i, variant in enumerate(order, 1):
        name = f"clip_{i:02d}.wav"
        audio = _process(base, sample_rate, variant)
        if variant.name == "distractor_level_-3dB":
            audio = (audio * 10.0 ** (-3.0 / 20.0)).astype(np.float32)   # the ONE unmatched clip
        else:
            audio = _match_loudness(audio, sample_rate, target)
        sf.write(out_dir / name, audio, sample_rate, subtype="PCM_24")
        key["clips"][name] = {"manipulation": variant.name, "bands": list(variant.bands),
                              "expects": variant.expects, "distractor": variant.distractor}
        rows.append({"path": name, "harshness": "", "mud": "", "clarity": ""})
    with (out_dir / "ratings.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "harshness", "mud", "clarity"])
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "key.json").write_text(json.dumps(key, indent=2))
    (out_dir / "README.txt").write_text(
        f"Perception rating kit — {len(order)} clips, {EXCERPT_S:.0f} s each, loudness-matched.\n\n"
        f"Listen to each clip and rate it {RATING_SCALE} on each column of ratings.csv:\n"
        "  harshness  how harsh / fatiguing the upper mids feel (7 = very harsh)\n"
        "  mud        how muddy / boxy the low mids feel (7 = very muddy)\n"
        "  clarity    how clear and separated the mix feels (7 = very clear)\n"
        "Leave a cell blank when unsure. Rate blind — do not open key.json first.\n\n"
        "Then run:  python -m orchestrator.perception_eval ratings.csv\n"
        "It prints the Spearman rank correlation of each composite metric against your\n"
        "ratings; the constants in server/mix_engine/analysis.py are right when those are high.\n")
    return key


def self_rate(key: dict) -> list[dict]:
    """Ratings a listener WOULD give if the manipulations do what they say —
    the kit's own pipeline check (the harness must correlate with these), never
    a substitute for a person."""
    rows = []
    for name, info in key["clips"].items():
        e = info["expects"]
        rows.append({"path": name,
                     "harshness": round(4.0 + e.get("harshness", 0.0) / 2.0, 1),
                     "mud": round(4.0 + e.get("mud", 0.0) / 2.0, 1),
                     "clarity": round(4.0 + e.get("clarity", 0.0) / 2.0, 1)})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    key = build(args.project, args.out, args.seed)
    print(f"wrote {len(key['clips'])} clips + ratings.csv + README.txt to {args.out} "
          f"(matched to {key['target_lufs']} LUFS; key.json holds the answers — rate first)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
