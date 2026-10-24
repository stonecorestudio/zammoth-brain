"""Technique evals (Phase 48 — docs/05-roadmap.md): does the agent execute
advanced mixing/mastering techniques WELL, judged on the re-rendered audio?

Ten cases against the purpose-built `technique_demo` fixture
(`orchestrator/technique_fixture.py` — synthesis + every constant the graders
read). Each case is a real `run_session` LLM run, graded by two deterministic
layers (no LLM judge anywhere):

- PLAN — the applied actions name the right tool with sane params (comp
  `mix` < 1 for parallel, a duck keyed by the named stem, band-selective
  dynamics for a time-varying boom, imager/band-width for mono-safe lows,
  a send/return or low-mix short reverb for space).
- OUTCOME — windowed, band-limited measurements on the re-rendered
  stems/master (ducking depth inside known lead phrases, swell-window boom
  reduction with the constant top preserved, sibilance-window HF drop with
  the melody untouched, LUFS/true-peak/PLR bands, low-band correlation kept
  mono while high-band side energy rises, added harmonics at known bins).

Billing policy (James, 2026-08-17): billed modes default
MIX_AGENT_LLM_BACKEND to `subscription` (the DAW's own dev transport) and
refuse `api` without --allow-api; the model pins to the product's default
(claude-opus-5-5 since 2026-10-08; every recorded run before that is Opus 5's). The SDK
loop is non-deterministic, so `--repeat N --check` gates on STABLE failures
only (a case fails the gate when every repeat fails) with absolute outcome
thresholds — no baseline diff.

    python -m orchestrator.technique_eval --list        # the corpus (free)
    python -m orchestrator.technique_eval --self-test   # grader validity (free)
    python -m orchestrator.technique_eval               # full run — BILLED
    python -m orchestrator.technique_eval --case duck_pads_under_lead  # one case
    python -m orchestrator.technique_eval --repeat 3 --check           # the gate

Grader validity (`--self-test`, free, no LLM): every case ships a golden
action script; the graders must pass the golden AND fail the untouched
project on both layers — a red eval therefore indicts the agent, not the
grader. Run it before merging changes to this file or the fixture.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import anyio
import numpy as np
import pyloudnorm
from scipy.signal import butter, resample_poly, sosfilt

from .benchmark import in_memory_engine
from .loop import EngineClient, run_session
from .technique_fixture import (BAND_BASS_TOP, BAND_BOOM, BAND_MELODY,
                                BAND_PAD_HIGH, BAND_PAD_LOW, BAND_PAD_MID,
                                BAND_SIB, BAND_TAIL, BASS_H3_HZ, BASS_STEADY,
                                BASS_SWELLS, HARSH_HZ, LEAD_AIR_HZ, LEAD_GAPS,
                                LEAD_LOUD_IDX, LEAD_PHRASES, PAD_LOW_HZ,
                                SIB_DUR_S, SIBILANT_S, STEM_NAMES,
                                TAIL_WINDOWS, write_fixture)

from mix_engine import dsp
from mix_engine.session import Session

PROJECT = "technique_demo"
DEFAULT_MODEL = "claude-opus-5-5"


# --------------------------------------------------------------------------
# Measurement windows (derived once from the fixture constants)
# --------------------------------------------------------------------------

def _inner(windows, margin_s: float = 0.3):
    return tuple((s + margin_s, e - margin_s) for s, e in windows)


LOUD_INNER = _inner(tuple(LEAD_PHRASES[i] for i in LEAD_LOUD_IDX))
QUIET_INNER = _inner(tuple(w for i, w in enumerate(LEAD_PHRASES)
                           if i not in LEAD_LOUD_IDX))
ALL_INNER = _inner(LEAD_PHRASES)
SIB_WINDOWS = tuple((s, s + SIB_DUR_S) for s in SIBILANT_S)
SWELL_INNER = _inner(BASS_SWELLS, margin_s=0.15)


# --------------------------------------------------------------------------
# Signal measurements (numpy only — deterministic)
# --------------------------------------------------------------------------

def _mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim == 2 else x


def _db(v: float) -> float:
    return 20.0 * float(np.log10(max(v, 1e-12)))


def _rms_db(x: np.ndarray) -> float:
    return _db(float(np.sqrt(np.mean(np.square(x)))))


def _peak_db(x: np.ndarray) -> float:
    return _db(float(np.abs(x).max()))


def _band(x: np.ndarray, sr: int, lo_hz: float, hi_hz: float,
          order: int = 6) -> np.ndarray:
    sos = butter(order, [lo_hz, hi_hz], btype="bandpass", fs=sr, output="sos")
    return sosfilt(sos, x)


def _slice_cat(x: np.ndarray, sr: int, windows) -> np.ndarray:
    parts = [x[int(s * sr):int(e * sr)] for s, e in windows]
    return np.concatenate([p for p in parts if len(p)])


def _windows_rms_db(x: np.ndarray, sr: int, windows,
                    band: Optional[tuple[float, float]] = None) -> float:
    if band is not None:
        x = _band(x, sr, *band)
    return _rms_db(_slice_cat(x, sr, windows))


def _tone_db(x: np.ndarray, sr: int, windows, freqs_hz,
             half_width_hz: float = 10.0) -> float:
    """Mean level (dBFS-ish sine amplitude) of the strongest bin near any of
    `freqs_hz`, averaged across the windows. Exact-bin measurement for the
    fixture's pure-sine components (leakage floor is far below any signal)."""
    if np.isscalar(freqs_hz):
        freqs_hz = (float(freqs_hz),)
    powers = []
    for start, end in windows:
        seg = x[int(start * sr):int(end * sr)]
        if len(seg) < sr // 10:
            continue
        win = np.hanning(len(seg))
        spec = np.abs(np.fft.rfft(seg * win)) * 2.0 / win.sum()
        bins = np.fft.rfftfreq(len(seg), 1.0 / sr)
        best = 0.0
        for f in freqs_hz:
            mask = (bins >= f - half_width_hz) & (bins <= f + half_width_hz)
            if mask.any():
                best = max(best, float(spec[mask].max()))
        powers.append(best ** 2)
    return _db(float(np.sqrt(np.mean(powers))) if powers else 0.0)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def measure(project: Path) -> dict:
    """Every scalar any case's outcome layer reads, from one render set."""
    sess = Session(project)
    master, sr = dsp.render_master(sess)
    stems = {}
    for sid in STEM_NAMES:
        audio, _ = dsp.render_stem(sess, sid)
        stems[sid] = audio

    drums = _mono(stems["drums"])
    bass = _mono(stems["bassline"])
    lead = _mono(stems["lead"])
    pads = stems["pads"]
    pads_mono = _mono(pads)
    master_mono = _mono(master)

    # Rhythm sum: the glue case — through the bus chain when one exists.
    bus_id = next((b for b in getattr(sess, "buses", {})
                   if {"drums", "bassline"} <= set(sess.bus_members(b))), None)
    if bus_id is not None:
        rhythm = _mono(dsp.render_bus(sess, bus_id)[0])
    else:
        rhythm = drums + bass

    lufs = float(pyloudnorm.Meter(sr).integrated_loudness(
        master if master.ndim == 2 else master[:, None]))
    tp = _peak_db(resample_poly(master, 4, 1, axis=0))

    pads_l, pads_r = (pads[:, 0], pads[:, 1]) if pads.ndim == 2 \
        else (pads_mono, pads_mono)
    low_l = _band(pads_l, sr, *BAND_PAD_LOW, order=4)
    low_r = _band(pads_r, sr, *BAND_PAD_LOW, order=4)
    side = (pads_l - pads_r) / 2.0
    mid = (pads_l + pads_r) / 2.0

    return {
        # every stem's own full-band level: what the collateral check reads (E7·R11)
        "stem_rms_db": {sid: _rms_db(_mono(stems[sid])) for sid in STEM_NAMES},
        "drums_rms_db": _rms_db(drums),
        "drums_peak_db": _peak_db(drums),
        "drums_crest_db": _peak_db(drums) - _rms_db(drums),
        "pads_phrase_rms_db": _windows_rms_db(pads_mono, sr, LOUD_INNER),
        "pads_gap_rms_db": _windows_rms_db(pads_mono, sr, LEAD_GAPS),
        "pads_low_corr": _corr(low_l, low_r),
        "pads_high_side_mid_db": (
            _rms_db(_band(side, sr, *BAND_PAD_HIGH))
            - _rms_db(_band(mid, sr, *BAND_PAD_HIGH))),
        "pads_mid_rms_db": _rms_db(_band(pads_mono, sr, *BAND_PAD_MID,
                                         order=4)),
        "lead_sib_hf_db": _windows_rms_db(lead, sr, SIB_WINDOWS,
                                          band=BAND_SIB),
        "lead_melody_db": _windows_rms_db(lead, sr, ALL_INNER,
                                          band=BAND_MELODY),
        "lead_harsh_loud_db": _tone_db(lead, sr, LOUD_INNER, HARSH_HZ),
        "lead_harsh_quiet_db": _tone_db(lead, sr, QUIET_INNER, HARSH_HZ),
        "lead_air_db": _tone_db(lead, sr, ALL_INNER, LEAD_AIR_HZ),
        "bass_boom_swell_db": _windows_rms_db(bass, sr, SWELL_INNER,
                                              band=BAND_BOOM),
        "bass_boom_steady_db": _windows_rms_db(bass, sr, BASS_STEADY,
                                               band=BAND_BOOM),
        "bass_top_swell_db": _windows_rms_db(bass, sr, SWELL_INNER,
                                             band=BAND_BASS_TOP),
        "bass_h_db": _tone_db(bass, sr, BASS_STEADY,
                              (PAD_LOW_HZ, BASS_H3_HZ)),
        "bass_peak_db": _peak_db(bass),
        "bass_rms_db": _rms_db(bass),
        "master_tail_db": _windows_rms_db(master_mono, sr, TAIL_WINDOWS,
                                          band=BAND_TAIL),
        "master_lead_active_db": _windows_rms_db(master_mono, sr, ALL_INNER,
                                                 band=BAND_TAIL),
        "master_lufs_i": lufs,
        "master_tp_dbtp": tp,
        "master_plr_db": tp - lufs,
        "rhythm_crest_db": _peak_db(rhythm) - _rms_db(rhythm),
    }


# --------------------------------------------------------------------------
# The cases: brief + PLAN grader + OUTCOME grader + golden action script
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TechCase:
    key: str
    technique: str
    brief: str
    plan: Callable[[list[dict]], list[str]]
    outcome: Callable[[dict, dict], list[str]]
    golden: tuple[dict, ...]
    # The plausible WRONG move for this brief — the static / full-band / fader cheat a
    # lazy engineer reaches for. `--self-test` applies it and requires BOTH layers to
    # reject it: until 2026-09-01 the free gate only proved the graders fail a no-op,
    # so the fixture's whole premise ("time-varying defects, so static cheats measurably
    # fail") was asserted by nothing (sweep B18).
    cheat: tuple[dict, ...] = ()
    # The stems whose OWN render this technique processes. Every other stem is a bystander
    # and must come out where it went in (`collateral_failures`). Empty where the technique
    # works on a bus, a return or the master: no stem's own render moves there.
    moves: tuple[str, ...] = ()


#: How far a stem the technique does not touch may move, full-band RMS, before the outcome
#: layer calls it collateral damage. Phase 48's iteration 1 failed on exactly this ("source
#: levels drifting -3..-7 dB") and the graders could not see it on any stem but the one the
#: brief named: nine of ten cases passed their golden PLUS -6 dB on an unrelated stem
#: (reader E7·R11). 1.5 dB is the bound the per-case graders already hold their own stem to.
COLLATERAL_TOLERANCE_DB = 1.5


def collateral_failures(case: "TechCase", b: dict, a: dict) -> list[str]:
    """Every stem outside `case.moves` whose level left where it was."""
    fails = []
    before, after = b.get("stem_rms_db") or {}, a.get("stem_rms_db") or {}
    for stem in STEM_NAMES:
        if stem in case.moves or stem not in before or stem not in after:
            continue
        moved = after[stem] - before[stem]
        if abs(moved) > COLLATERAL_TOLERANCE_DB:
            fails.append(f"collateral: '{stem}' moved {moved:+.1f} dB and the brief never "
                         f"asked for it (want +-{COLLATERAL_TOLERANCE_DB:g}; this technique "
                         f"works on {list(case.moves) or 'no stem of its own'})")
    return fails


def outcome_failures(case: "TechCase", b: dict, a: dict) -> list[str]:
    """The OUTCOME layer: the case's own grader, then the bystanders - one check for every
    case, run by the billed eval and the self-test alike."""
    return case.outcome(b, a) + collateral_failures(case, b, a)


def bystander(case: "TechCase") -> str:
    """A stem this case's golden never names: the one the self-test pulls down 6 dB."""
    named = {v for action in case.golden for k, v in action.items()
             if k in ("stem_id", "key_stem_id") and isinstance(v, str)}
    named |= {m for action in case.golden for m in (action.get("members") or [])}
    return next(s for s in STEM_NAMES if s not in named and s not in case.moves)


def _plan_parallel(actions: list[dict]) -> list[str]:
    if any(a.get("type") == "comp" and a.get("stem_id") == "drums"
           and 0.0 < a.get("mix", 1.0) <= 0.8 for a in actions):
        return []
    return ["no parallel compression: want a comp on 'drums' with "
            "0 < mix <= 0.8 (mix is the wet/dry blend; 1.0 is a plain insert)"]


def _out_parallel(b: dict, a: dict) -> list[str]:
    # The engine's comp `mix` is a wet/dry CROSSFADE (Phase 23), so parallel
    # thickening reads as: crest down (body raised toward the peaks), level
    # roughly held or up, peaks not up. Additive-NY "RMS jumps" intuition
    # does not apply here.
    fails = []
    d_crest = a["drums_crest_db"] - b["drums_crest_db"]
    if d_crest < -6.0:
        fails.append(f"drums crest fell {d_crest:+.1f} dB (punch lost — "
                     "that is a slam, not a blend)")
    if d_crest > -0.5:
        fails.append(f"drums crest moved {d_crest:+.1f} dB "
                     "(no audible thickening; want <= -0.5)")
    d_rms = a["drums_rms_db"] - b["drums_rms_db"]
    if not -1.0 <= d_rms <= 6.0:
        fails.append(f"drums RMS moved {d_rms:+.1f} dB (want -1..+6 — "
                     "the blend should hold or lift the level)")
    if a["drums_peak_db"] - b["drums_peak_db"] > 1.0:
        fails.append("drums peak grew more than 1 dB")
    return fails


def _plan_duck(actions: list[dict]) -> list[str]:
    if any(a.get("type") == "duck" and a.get("stem_id") == "pads"
           and a.get("key_stem_id") == "lead"
           and a.get("depth_db", 0.0) >= 2.0 for a in actions):
        return []
    return ["no sidechain duck on 'pads' keyed by 'lead' "
            "(type 'duck', depth_db >= 2)"]


def _out_duck(b: dict, a: dict) -> list[str]:
    fails = []
    depth = ((a["pads_gap_rms_db"] - a["pads_phrase_rms_db"])
             - (b["pads_gap_rms_db"] - b["pads_phrase_rms_db"]))
    if depth < 2.0:
        fails.append(f"ducking depth {depth:+.1f} dB during lead phrases "
                     "(want >= 2)")
    if a["pads_gap_rms_db"] < b["pads_gap_rms_db"] - 1.0:
        fails.append("pads dropped in the gaps too — that is a fader move, "
                     "not a duck")
    return fails


def _plan_deess(actions: list[dict]) -> list[str]:
    if any((a.get("type") == "deess" and a.get("stem_id") == "lead"
            and 3500.0 <= a.get("freq_hz", 0.0) <= 10000.0)
           or (a.get("type") == "dynamic_eq" and a.get("stem_id") == "lead"
               and a.get("freq_hz", 0.0) >= 4000.0
               and a.get("gain_db", 0.0) < 0.0) for a in actions):
        return []
    return ["no de-esser on 'lead' (deess at 3.5-10 kHz, or a dynamic_eq "
            "cut >= 4 kHz)"]


def _out_deess(b: dict, a: dict) -> list[str]:
    fails = []
    drop = b["lead_sib_hf_db"] - a["lead_sib_hf_db"]
    if drop < 3.0:
        fails.append(f"sibilance bursts only came down {drop:.1f} dB "
                     "(want >= 3)")
    d_mel = a["lead_melody_db"] - b["lead_melody_db"]
    if abs(d_mel) > 1.0:
        fails.append(f"melody band moved {d_mel:+.1f} dB (want +-1: "
                     "de-ess the esses, not the melody)")
    d_air = a["lead_air_db"] - b["lead_air_db"]
    if abs(d_air) > 1.0:
        fails.append(f"the constant 4 kHz air moved {d_air:+.1f} dB (want +-1: "
                     "a static high cut dulls everything, a de-esser acts only "
                     "on the esses)")
    return fails


def _plan_boom(actions: list[dict]) -> list[str]:
    if any((a.get("type") == "mb_comp" and a.get("stem_id") == "bassline"
            and (a.get("low") or {}).get("ratio", 1.0) >= 1.5
            and a.get("crossover_lo_hz", 150.0) <= 250.0)
           or (a.get("type") == "dynamic_eq"
               and a.get("stem_id") == "bassline"
               and a.get("freq_hz", 0.0) <= 150.0
               and a.get("gain_db", 0.0) < 0.0) for a in actions):
        return []
    return ["no band-selective dynamics on 'bassline' (mb_comp with an "
            "active low band, or a low dynamic_eq cut) — the boom is "
            "time-varying, a static EQ cut is the wrong tool"]


def _out_boom(b: dict, a: dict) -> list[str]:
    fails = []
    drop = b["bass_boom_swell_db"] - a["bass_boom_swell_db"]
    if drop < 2.5:
        fails.append(f"swell-window boom only came down {drop:.1f} dB "
                     "(want >= 2.5)")
    d_steady = a["bass_boom_steady_db"] - b["bass_boom_steady_db"]
    if abs(d_steady) > 1.5:
        fails.append(f"steady low end moved {d_steady:+.1f} dB (want +-1.5: "
                     "tame the swells, keep the foundation)")
    d_top = a["bass_top_swell_db"] - b["bass_top_swell_db"]
    if abs(d_top) > 1.5:
        fails.append(f"the constant 220/440 top moved {d_top:+.1f} dB during "
                     "swells (want +-1.5: full-band compression pumps it)")
    return fails


def _plan_master(actions: list[dict]) -> list[str]:
    # The brief says "around -14 LUFS, keep it dynamic": a limiter driven to -9 is the
    # loudness-war move, not the plan (2026-09-01 sweep B18 — the cheat that exposed the
    # ceiling-only check). A target within -16..-12 (or none: the engine's default) passes.
    if any(a.get("type") == "limiter" and a.get("stem_id") == "master"
           and a.get("ceiling_dbtp", 0.0) <= -0.9
           and (a.get("target_lufs_i") is None
                or -16.0 <= a.get("target_lufs_i") <= -12.0) for a in actions):
        return []
    return ["no master limiter with a ceiling at/below -1 dBTP and a streaming "
            "target (-16..-12 LUFS) — a limiter slammed to -9 is not 'dynamic'"]


def _out_master(b: dict, a: dict) -> list[str]:
    fails = []
    if not -15.0 <= a["master_lufs_i"] <= -13.0:
        fails.append(f"master landed at {a['master_lufs_i']:.1f} LUFS "
                     "(want -15..-13 for streaming)")
    if a["master_tp_dbtp"] > -0.9:
        fails.append(f"true peak {a['master_tp_dbtp']:.1f} dBTP "
                     "(want <= -1)")
    if a["master_plr_db"] < 6.0:
        fails.append(f"PLR {a['master_plr_db']:.1f} dB (want >= 6: "
                     "over-limited)")
    return fails


def _plan_width(actions: list[dict]) -> list[str]:
    imager_ok = any(a.get("type") == "imager" and a.get("stem_id") == "pads"
                    and a.get("width_high", 1.0) >= 1.1
                    and a.get("width_low", 1.0) <= 1.0 for a in actions)
    bw_ok = any(a.get("type") == "band_width" and a.get("stem_id") == "pads"
                and a.get("band") in ("high_mid", "presence", "brilliance",
                                      "air")
                and a.get("width", 1.0) >= 1.1 for a in actions)
    low_guard = any(
        (a.get("type") == "utility" and a.get("stem_id") == "pads"
         and a.get("bass_mono"))
        or (a.get("type") == "imager" and a.get("stem_id") == "pads"
            and a.get("width_low", 1.0) < 1.0)
        or (a.get("type") == "band_width" and a.get("stem_id") == "pads"
            and a.get("band") in ("sub", "bass")
            and a.get("width", 1.0) < 1.0) for a in actions)
    naive = any(a.get("type") == "width" and a.get("stem_id") == "pads"
                and a.get("width", 1.0) > 1.05 for a in actions)
    if imager_ok or bw_ok or (naive and low_guard):
        return []
    return ["no band-selective width on 'pads': want an imager "
            "(width_high > 1, width_low <= 1), a high band_width, or a "
            "full-band widen WITH a low-end guard (bass_mono / narrowed "
            "lows) — naked full-band width loosens the low end"]


def _out_width(b: dict, a: dict) -> list[str]:
    fails = []
    if a["pads_low_corr"] < b["pads_low_corr"] - 0.02:
        fails.append(f"pad low-band correlation fell "
                     f"{b['pads_low_corr']:.2f} -> {a['pads_low_corr']:.2f} "
                     "(the lows got looser, not mono-safer)")
    lift = a["pads_high_side_mid_db"] - b["pads_high_side_mid_db"]
    if lift < 2.0:
        fails.append(f"high-band side energy only rose {lift:+.1f} dB "
                     "(want >= 2: it should actually get wider)")
    d_mid = a["pads_mid_rms_db"] - b["pads_mid_rms_db"]
    if abs(d_mid) > 1.5:
        fails.append(f"pad mids moved {d_mid:+.1f} dB (want +-1.5)")
    return fails


def _plan_space(actions: list[dict]) -> list[str]:
    for a in actions:
        if (a.get("type") in ("return_reverb", "return_convolution")
                and a.get("return_id") and a.get("decay_s", 1.8) <= 2.0):
            if any(s.get("type") == "send" and s.get("stem_id") == "lead"
                   and s.get("return_id") == a["return_id"]
                   for s in actions):
                return []
    # The low-mix INSERT idiom was accepted here until 2026-08-18, but the
    # outcome layer can never pass it (mix 0.35 lifts the tail +0.56 dB vs
    # the +1 bar) — a plan grader that green-lights a certain outcome failure
    # is a trap, and two of three iteration-4 runs walked into it. A space
    # for a stem is a send/return; the golden is one; the prompt says so.
    return ["no send/return space on 'lead': want a reverb return "
            "(decay <= 2 s) fed by a send from 'lead' — a low-mix insert "
            "reverb cannot lift the tail audibly, and a wet one washes the "
            "lead out"]


def _out_space(b: dict, a: dict) -> list[str]:
    fails = []
    lift = a["master_tail_db"] - b["master_tail_db"]
    if lift < 1.0:
        fails.append(f"post-phrase tail only rose {lift:+.1f} dB "
                     "(want >= 1: no audible space was added)")
    d_lead = a["master_lead_active_db"] - b["master_lead_active_db"]
    if abs(d_lead) > 1.5:
        fails.append(f"lead level moved {d_lead:+.1f} dB while singing "
                     "(want +-1.5: keep it upfront)")
    return fails


def _plan_glue(actions: list[dict]) -> list[str]:
    grp = next((a for a in actions if a.get("type") == "group"
                and {"drums", "bassline"} <= set(a.get("members") or [])),
               None)
    if grp is None:
        return ["no group action making a bus with drums + bassline"]
    # mix >= 0.3: both glue idioms are valid — the classic insert (mix ~1)
    # and the blended bus comp SYSTEM_PROMPT itself teaches (mix 0.3-0.5).
    # Iteration 2: the grader used to demand mix >= 0.9 while the prompt
    # taught 0.4 — a grader-vs-prompt disagreement, not model error.
    if any(a.get("type") == "comp" and a.get("stem_id") == grp.get("bus_id")
           and 1.2 <= a.get("ratio", 0.0) <= 4.0
           and a.get("mix", 1.0) >= 0.3 for a in actions):
        return []
    return [f"no gentle glue comp (ratio 1.2-4, mix >= 0.3) on bus "
            f"'{grp.get('bus_id')}'"]


def _out_glue(b: dict, a: dict) -> list[str]:
    fails = []
    drop = b["rhythm_crest_db"] - a["rhythm_crest_db"]
    if not 0.4 <= drop <= 5.0:
        fails.append(f"rhythm-bus crest moved {-drop:+.1f} dB "
                     "(want a 0.4-5 dB reduction: glue, not a slam)")
    d_lufs = a["master_lufs_i"] - b["master_lufs_i"]
    if abs(d_lufs) > 1.5:
        fails.append(f"master loudness moved {d_lufs:+.1f} LU "
                     "(glue should not be a loudness move)")
    return fails


def _plan_dynharsh(actions: list[dict]) -> list[str]:
    if any((a.get("type") == "dynamic_eq" and a.get("stem_id") == "lead"
            and 2000.0 <= a.get("freq_hz", 0.0) <= 6000.0
            and a.get("gain_db", 0.0) < 0.0)
           or (a.get("type") == "deess" and a.get("stem_id") == "lead"
               and 2000.0 <= a.get("freq_hz", 0.0) <= 4500.0)
           or (a.get("type") == "mb_comp" and a.get("stem_id") == "lead"
               and (a.get("high") or {}).get("ratio", 1.0) >= 1.5)
           for a in actions):
        return []
    return ["no threshold-gated presence control on 'lead' (a dynamic_eq "
            "cut at 2-6 kHz, a 2-4.5 kHz de-esser, or an mb_comp high "
            "band) — a static EQ cut dulls the quiet phrases too"]


def _out_dynharsh(b: dict, a: dict) -> list[str]:
    fails = []
    drop = b["lead_harsh_loud_db"] - a["lead_harsh_loud_db"]
    if drop < 3.0:
        fails.append(f"loud-phrase harshness only came down {drop:.1f} dB "
                     "(want >= 3)")
    d_quiet = a["lead_harsh_quiet_db"] - b["lead_harsh_quiet_db"]
    if abs(d_quiet) > 1.5:
        fails.append(f"quiet-phrase presence moved {d_quiet:+.1f} dB "
                     "(want +-1.5: the fix must be threshold-gated, "
                     "not a static cut)")
    return fails


def _plan_sat(actions: list[dict]) -> list[str]:
    if any(a.get("type") == "saturation" and a.get("stem_id") == "bassline"
           for a in actions):
        return []
    return ["no saturation on 'bassline' (tape warmth is the saturation "
            "device — overdrive is the aggressive tool)"]


def _out_sat(b: dict, a: dict) -> list[str]:
    fails = []
    lift = a["bass_h_db"] - b["bass_h_db"]
    if lift < 10.0:
        fails.append(f"added harmonics only rose {lift:+.1f} dB at "
                     "110/165 Hz (want >= 10: no audible warmth)")
    if a["bass_peak_db"] - b["bass_peak_db"] > 0.5:
        fails.append("bass peak level grew (saturation should compress "
                     "peaks, not push them)")
    d_rms = a["bass_rms_db"] - b["bass_rms_db"]
    if abs(d_rms) > 2.0:
        fails.append(f"bass level moved {d_rms:+.1f} dB (want +-2)")
    return fails


CASES: tuple[TechCase, ...] = (
    TechCase(
        "parallel_comp_drums", "parallel (NY) compression",
        "add parallel compression to the drums to thicken them up, "
        "but keep the punch",
        _plan_parallel, _out_parallel,
        # The golden was -26 / 4:1 / 5 ms / 150 ms / mix 0.5 until 2026-10-03, and it thickened
        # only in the OFFLINE render: that compressor's "5 ms" was a time constant of 0.8 ms
        # (reader E4·R1). On the law the DAW runs - which the offline engine now runs too - the
        # same plan RAISES the drums' crest 0.8 dB: at 5 ms the hit is through before the comp
        # closes. Measured over the fixture (1,440 settings at makeup 6): attack <= 1 ms passes
        # 60-78 %, 2 ms 15 %, 3 ms none; release 20-60 ms beats 150; threshold -20..-24.
        ({"type": "comp", "stem_id": "drums", "threshold_db": -22.0,
          "ratio": 4.0, "attack_ms": 1.0, "release_ms": 60.0,
          "makeup_db": 6.0, "mix": 0.6},),   # makeup 6 = the guardrail cap
        cheat=({"type": "gain", "stem_id": "drums", "delta_db": 3.0},),
        moves=("drums",)),
    TechCase(
        "duck_pads_under_lead", "sidechain ducking",
        "duck the pads under the lead so the lead always stays on top",
        _plan_duck, _out_duck,
        ({"type": "duck", "stem_id": "pads", "key_stem_id": "lead",
          "threshold_db": -35.0, "depth_db": 6.0, "attack_ms": 5.0,
          "release_ms": 150.0},),
        cheat=({"type": "gain", "stem_id": "pads", "delta_db": -6.0},),
        moves=("pads",)),
    TechCase(
        "deess_lead", "de-essing",
        "the lead has piercing sibilance - de-ess it without dulling "
        "the melody",
        _plan_deess, _out_deess,
        ({"type": "deess", "stem_id": "lead", "threshold_db": -48.0,
          "freq_hz": 5000.0, "ratio": 8.0},),
        cheat=({"type": "eq", "stem_id": "lead", "band": {"shape": "high_shelf", "freq_hz": 4500.0, "gain_db": -9.0, "q": 0.7}},),
        moves=("lead",)),
    TechCase(
        "tame_boom_bassline", "multiband / dynamic low control",
        "the bassline gets boomy when it swells - tame the boom without "
        "losing the steady low end or dulling its top",
        _plan_boom, _out_boom,
        ({"type": "dynamic_eq", "stem_id": "bassline", "freq_hz": 55.0,
          "gain_db": -8.0, "q": 1.0, "threshold_db": -24.0, "ratio": 6.0,
          "attack_ms": 10.0, "release_ms": 120.0},),
        cheat=({"type": "eq", "stem_id": "bassline", "band": {"shape": "peak", "freq_hz": 55.0, "gain_db": -8.0, "q": 1.0}},),
        moves=("bassline",)),
    TechCase(
        "master_streaming", "mastering chain",
        "master this for streaming: around -14 LUFS, true peak at -1, "
        "keep it clean and dynamic",
        _plan_master, _out_master,
        ({"type": "limiter", "stem_id": "master", "ceiling_dbtp": -1.0,
          "target_lufs_i": -14.0},),
        # "slam it loud": the ceiling is legal (the guardrail admits it) but the limiter is
        # driven to -9 LUFS — over-limited, out of the streaming window, PLR collapses
        cheat=({"type": "limiter", "stem_id": "master", "ceiling_dbtp": -1.0, "target_lufs_i": -9.0},)),
    TechCase(
        "widen_pads_mono_lows", "band-limited stereo width",
        "make the pads wider and more spacious, but keep the low end "
        "mono and club-safe",
        _plan_width, _out_width,
        ({"type": "imager", "stem_id": "pads", "width_low": 0.5,
          "width_mid": 1.0, "width_high": 1.6, "crossover_lo_hz": 150.0,
          "crossover_hi_hz": 2000.0},),
        cheat=({"type": "width", "stem_id": "pads", "width": 1.6},),
        moves=("pads",)),
    TechCase(
        "vocal_space_send", "send/return ambience",
        "give the lead a sense of space with a short room reverb, but "
        "keep it upfront - not washed out",
        _plan_space, _out_space,
        ({"type": "return_reverb", "return_id": "space", "decay_s": 1.4,
          "pre_delay_ms": 20.0, "tone": 0.5, "mix": 1.0},
         {"type": "send", "stem_id": "lead", "return_id": "space",
          "gain_db": -4.0}),
        # the low-mix INSERT idiom (a space device on the stem itself, no return, no send)
        cheat=({"type": "return_reverb", "stem_id": "lead", "decay_s": 1.4, "pre_delay_ms": 20.0, "tone": 0.5, "mix": 0.35},)),
    TechCase(
        "glue_rhythm_section", "bus glue compression",
        "group the drums and bassline into a rhythm bus and glue them "
        "with gentle bus compression",
        _plan_glue, _out_glue,
        ({"type": "group", "bus_id": "rhythm",
          "members": ["drums", "bassline"]},
         # 10 ms / 250 ms until 2026-10-03 - which, on the true law, lets every hit through
         # and RAISES the bus's crest 0.7 dB (see the parallel golden above). The crest falls
         # for attack 1-3 ms and rises from 10 ms up; 5 ms is the turning point.
         {"type": "comp", "stem_id": "rhythm", "threshold_db": -22.0,
          "ratio": 2.0, "attack_ms": 2.0, "release_ms": 150.0,
          "makeup_db": 0.0, "mix": 1.0}),
        cheat=({"type": "group", "bus_id": "rhythm", "members": ["drums", "bassline"]}, {"type": "comp", "stem_id": "rhythm", "threshold_db": -36.0, "ratio": 10.0, "attack_ms": 1.0, "release_ms": 50.0, "makeup_db": 0.0, "mix": 1.0})),
    TechCase(
        "dynamic_harsh_lead", "dynamic EQ",
        "the lead turns harsh when it gets loud - tame that harshness "
        "without dulling the quiet phrases",
        _plan_dynharsh, _out_dynharsh,
        ({"type": "dynamic_eq", "stem_id": "lead", "freq_hz": 3200.0,
          "gain_db": -9.0, "q": 2.0, "threshold_db": -33.0, "ratio": 6.0,
          "attack_ms": 5.0, "release_ms": 120.0},),
        cheat=({"type": "eq", "stem_id": "lead", "band": {"shape": "peak", "freq_hz": 3200.0, "gain_db": -9.0, "q": 2.0}},),
        moves=("lead",)),
    TechCase(
        "warm_saturation_bassline", "harmonic saturation",
        "add some tape-style warmth to the bassline - harmonic richness, "
        "but don't push its peak level",
        _plan_sat, _out_sat,
        ({"type": "saturation", "stem_id": "bassline", "drive_db": 9.0,
          "curve": "warm"},),
        cheat=({"type": "gain", "stem_id": "bassline", "delta_db": 3.0},),
        moves=("bassline",)),
)


# --------------------------------------------------------------------------
# Golden application + self-test (free — grader validity)
# --------------------------------------------------------------------------

async def _apply_golden(project: Path, actions: tuple[dict, ...]) -> None:
    async with in_memory_engine(project) as client:
        engine = EngineClient(client)
        for action in actions:   # one proposal per action: order guaranteed
            res = await engine.call("propose_actions", {"req": {
                "rationale": "technique-eval golden script",
                "predicted_impact": [],
                "actions": [action]}})
            if not res.get("accepted"):
                raise RuntimeError(
                    f"golden action rejected: {res.get('violations')}")
            await engine.call("apply_proposal",
                              {"proposal_id": res["proposal_id"]})


def fresh_fixture(work: Path) -> Path:
    """The fixture, synthesised into the WORK folder: deterministic and < 1 s, so it is made
    for every run and can never be stale against technique_fixture.py.

    It was written into `test_projects/technique_demo` until 2026-10-07 - the repo's own
    folder, by every self-test and every billed run - which is why no test could run the
    self-test (reader E7·R12) and why two runs side by side shared one set of files."""
    source = work / "_fixture"
    write_fixture(source)
    return source


def stage(source: Path, work_dir: Path) -> Path:
    """A throwaway project: the fixture's wavs and nothing else."""
    project = work_dir / PROJECT
    project.mkdir(parents=True)
    for wav in sorted(source.glob("*.wav")):
        shutil.copy(wav, project / wav.name)
    return project


def self_test(keys: Optional[set[str]] = None) -> int:
    """Prove the graders: the golden passes, the untouched project fails, the cheat is
    rejected, and the golden with a bystander pulled down fails — every case. No LLM, free."""
    work = Path(tempfile.mkdtemp(prefix="mix-agent-technique-selftest-"))
    source = fresh_fixture(work)
    problems: list[str] = []
    for case in CASES:
        if keys and case.key not in keys:
            continue
        project = stage(source, work / case.key)
        before = measure(project)
        if not case.outcome(before, before):
            problems.append(f"{case.key}: outcome layer passes the "
                            "untouched project (grader has no teeth)")
        if not case.plan([]):
            problems.append(f"{case.key}: plan layer passes an empty "
                            "action list")
        plan_fails = case.plan([dict(a) for a in case.golden])
        if plan_fails:
            problems.append(f"{case.key}: golden fails the plan layer: "
                            f"{plan_fails}")
        try:
            anyio.run(_apply_golden, project, case.golden)
        except Exception as exc:
            problems.append(f"{case.key}: golden failed to apply: {exc}")
            continue
        outcome_fails = outcome_failures(case, before, measure(project))
        if outcome_fails:
            problems.append(f"{case.key}: golden fails the outcome layer: "
                            f"{outcome_fails}")
        # …and the golden PLUS a stem nobody mentioned pulled down 6 dB must fail: the
        # technique was done and something else was damaged (E7·R11).
        victim = bystander(case)
        damaged = stage(source, work / (case.key + "-collateral"))
        try:
            anyio.run(_apply_golden, damaged, case.golden + (
                {"type": "gain", "stem_id": victim, "delta_db": -6.0},))
        except Exception as exc:
            problems.append(f"{case.key}: golden + collateral failed to apply: {exc}")
        else:
            if not collateral_failures(case, before, measure(damaged)):
                problems.append(f"{case.key}: outcome layer ACCEPTS the golden plus -6 dB "
                                f"on '{victim}' — collateral damage is invisible here")
        # …and the CHEAT — the plausible wrong move — must be rejected by both
        # layers, on a fresh copy (the golden is already applied above).
        if case.cheat:
            if not case.plan([dict(a) for a in case.cheat]):
                problems.append(f"{case.key}: plan layer ACCEPTS the cheat "
                                f"{[a['type'] for a in case.cheat]}")
            cheated = stage(source, work / (case.key + "-cheat"))
            try:
                anyio.run(_apply_golden, cheated, case.cheat)
            except Exception as exc:
                problems.append(f"{case.key}: cheat failed to apply: {exc}")
            else:
                if not case.outcome(before, measure(cheated)):
                    problems.append(f"{case.key}: outcome layer ACCEPTS the "
                                    f"cheat {[a['type'] for a in case.cheat]} "
                                    "— the fixture's time-varying defect is "
                                    "not doing its job here")
        else:
            problems.append(f"{case.key}: no cheat script — every case ships one")
        status = "ok" if not any(p.startswith(case.key + ":")
                                 for p in problems) else "FAIL"
        print(f"  {case.key:<28} {status}")
    shutil.rmtree(work, ignore_errors=True)
    if problems:
        print("\nSELF-TEST FAILURES:\n" + "\n".join(f"- {p}"
                                                    for p in problems))
        return 1
    print("\nself-test: every golden passes, every untouched project fails, "
          "every cheat is rejected by both layers, every golden with a bystander "
          "pulled down fails — graders are valid")
    return 0


# --------------------------------------------------------------------------
# The billed LLM run
# --------------------------------------------------------------------------

def _applied_actions(project: Path) -> list[dict]:
    state = json.loads((project / "session.json").read_text())
    return [action for p in state.get("proposals", [])
            if p.get("status") == "applied"
            for action in p.get("actions", [])]


def run_case_llm(case: TechCase, work_dir: Path, max_passes: int,
                 source: Path) -> dict:
    from .agents import ClaudeAgent
    project = stage(source, work_dir)
    before = measure(project)
    saved = os.environ.get("MIX_AGENT_DATA_DIR")
    os.environ["MIX_AGENT_DATA_DIR"] = str(work_dir / "appdata")
    try:
        report = run_session(project, case.brief, gating="auto",
                             agent=ClaudeAgent(), max_passes=max_passes,
                             connect=in_memory_engine)
        stop_reason, passes = report.stop_reason, report.passes_run
        # A pass that raised is swallowed by the loop into warnings +
        # stop_reason "error" (loop.py), so run_session RETURNS normally.
        # Grading that would measure the UNTOUCHED project and print it as a
        # technique verdict — a dead transport read as "no master limiter"
        # on 2026-08-18. A run that never reached the agent is not a
        # measurement of the agent.
        error = (f"run stopped: {'; '.join(report.warnings) or 'unknown'}"
                 if stop_reason == "error" else None)
    except Exception as exc:   # a failed run is a failed repeat, not a crash
        error, stop_reason, passes = f"{type(exc).__name__}: {exc}", None, 0
    finally:
        if saved is None:
            os.environ.pop("MIX_AGENT_DATA_DIR", None)
        else:
            os.environ["MIX_AGENT_DATA_DIR"] = saved
    actions = _applied_actions(project)
    plan_fails = (case.plan(actions) if error is None
                  else [f"run failed: {error}"])
    outcome_fails = (outcome_failures(case, before, measure(project))
                     if error is None else ["run failed"])
    return {
        "plan_failures": plan_fails,
        "outcome_failures": outcome_fails,
        "passed": not plan_fails and not outcome_fails,
        "applied_actions": len(actions),
        "stop_reason": stop_reason,
        "passes_run": passes,
        "error": error,
    }


def run_eval(cases: tuple[TechCase, ...], repeat: int,
             max_passes: int) -> dict:
    work = Path(tempfile.mkdtemp(prefix="mix-agent-technique-"))
    source = fresh_fixture(work)
    results = []
    for case in cases:
        repeats = []
        for r in range(repeat):
            repeats.append(run_case_llm(case, work / f"{case.key}_{r}",
                                        max_passes, source))
            mark = "pass" if repeats[-1]["passed"] else "FAIL"
            print(f"  {case.key:<28} repeat {r + 1}/{repeat}: {mark}")
        passed_n = sum(1 for r in repeats if r["passed"])
        # Every repeat erroring is a case the gate never tested — label it
        # unmeasured (still gates red, but as missing data, not as the agent
        # missing the technique; glue_rhythm_section, 2026-08-18).
        unmeasured = all(r["error"] is not None for r in repeats)
        results.append({
            "key": case.key,
            "technique": case.technique,
            "brief": case.brief,
            "passed": passed_n,
            "repeats": repeat,
            "stable_failure": passed_n == 0,
            "unmeasured": unmeasured,
            "runs": repeats,
        })
    measured = sum(1 for c in results for r in c["runs"]
                   if r["error"] is None)
    summary = {
        "cases": len(results),
        "runs": len(results) * repeat,
        "not_measured": len(results) * repeat - measured,
        "stable_failures": sum(1 for c in results if c["stable_failure"]),
        "flaky": sum(1 for c in results
                     if 0 < c["passed"] < c["repeats"]),
        "clean_passes": sum(1 for c in results
                            if c["passed"] == c["repeats"]),
    }
    return {"model": os.environ.get("MIX_AGENT_MODEL", DEFAULT_MODEL),
            "backend": os.environ.get("MIX_AGENT_LLM_BACKEND", "api"),
            "cases": results, "summary": summary}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Mix-agent technique evals (Phase 48)")
    parser.add_argument("--list", action="store_true",
                        help="print the corpus and exit (free)")
    parser.add_argument("--self-test", action="store_true",
                        help="grader validity: goldens pass, untouched "
                             "fails (free, no LLM)")
    parser.add_argument("--case", action="append", default=None,
                        help="run only this case key (repeatable)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="repeats per case (the SDK loop is "
                             "non-deterministic)")
    parser.add_argument("--max-passes", type=int, default=3)
    parser.add_argument("--check", action="store_true",
                        help="exit 1 on any STABLE failure (all repeats); needs "
                             "--repeat 2 or more")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--allow-api", action="store_true",
                        help="permit the metered api backend (policy: "
                             "this suite runs on the subscription)")
    parser.add_argument("--out", type=Path, default=None,
                        help="write the JSON report here")
    args = parser.parse_args(argv)

    keys = set(args.case) if args.case else None
    unknown = (keys or set()) - {c.key for c in CASES}
    if unknown:
        parser.error(f"unknown case keys: {sorted(unknown)} "
                     f"(see --list)")

    if args.list:
        for case in CASES:
            print(f"{case.key:<28} [{case.technique}] {case.brief}")
        return 0
    if args.self_test:
        return self_test(keys)
    # "Fails on EVERY repeat" needs more than one repeat (reader E7·R12): with the default
    # of 1, a single unlucky run of a non-deterministic agent was a stable failure and the
    # gate went red on it. operate_eval's gate refuses the same shape. Said before a cent
    # is spent.
    if args.check and args.repeat < 2:
        parser.error("--check gates on STABLE failures - a case that fails every repeat - "
                     "so it needs --repeat 2 or more (the documented gate is --repeat 3)")

    backend = os.environ.get("MIX_AGENT_LLM_BACKEND", "").strip()
    if not backend:
        backend = "subscription"
        os.environ["MIX_AGENT_LLM_BACKEND"] = backend
    if backend == "api" and not args.allow_api:
        print("technique_eval runs on the subscription backend by policy "
              "(Phase 48). Set MIX_AGENT_LLM_BACKEND=subscription (and "
              "CLAUDE_CODE_OAUTH_TOKEN), or pass --allow-api to bill an "
              "API key deliberately.", file=sys.stderr)
        return 2
    os.environ["MIX_AGENT_MODEL"] = args.model

    cases = tuple(c for c in CASES if keys is None or c.key in keys)
    print(f"technique_eval: {len(cases)} case(s) x {args.repeat} repeat(s) "
          f"on backend={backend} model={args.model} — BILLED")
    report = run_eval(cases, args.repeat, args.max_passes)
    text = json.dumps(report, indent=2)
    if args.out is not None:
        args.out.write_text(text + "\n")

    print()
    for case in report["cases"]:
        status = ("UNMEASURED" if case.get("unmeasured") else
                  "PASS" if case["passed"] == case["repeats"] else
                  "FLAKY" if case["passed"] else "FAIL")
        print(f"  {case['key']:<28} {status} "
              f"({case['passed']}/{case['repeats']})")
        for run in case["runs"]:
            for f in run["plan_failures"] + run["outcome_failures"]:
                print(f"      - {f}")
    s = report["summary"]
    print(f"\n{s['clean_passes']} clean / {s['flaky']} flaky / "
          f"{s['stable_failures']} stable failures of {s['cases']} cases")

    # Every run erroring is a broken transport, not a 0/N capability score.
    # Reporting it as the latter is how an expired token reads as "the agent
    # regressed" (2026-08-18). Distinct exit code: this is not a gate result.
    if s["not_measured"] == s["runs"]:
        print(f"\nNOT A MEASUREMENT: all {s['runs']} run(s) errored before "
              f"reaching the agent — the scores above are graded against "
              f"UNTOUCHED audio. Check the transport (backend={report['backend']}) "
              f"before reading this as agent behaviour.", file=sys.stderr)
        return 2
    if s["not_measured"]:
        print(f"\nWARNING: {s['not_measured']}/{s['runs']} run(s) errored "
              f"before reaching the agent; those repeats measure nothing.",
              file=sys.stderr)

    if args.check and s["stable_failures"]:
        unmeasured = [c["key"] for c in report["cases"] if c.get("unmeasured")]
        if unmeasured:
            print(f"\nUNMEASURED (every repeat errored — the gate never "
                  f"tested these): {', '.join(unmeasured)}", file=sys.stderr)
        print("\nSTABLE FAILURES — the agent missed the technique on "
              "every repeat", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
