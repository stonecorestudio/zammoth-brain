"""DAW export oracle (docs/10-daw.md Milestone D — "a project bounce matches the live master
within tolerance").

The native DAW export is a bounce of the live Tracktion graph running our native ``MonitorChain``
nodes. The deliverable's reference is the Python offline engine (``dsp.render_master`` / Pedalboard).
Phase 40 validated each native device against the offline engine *per device* (the
``reference_monitor`` null tests). This closes the loop end-to-end: it renders the SAME controlled
mix two ways — offline (the reference) and through the real DAW export path
(``MixAgentDAW --export-oracle``) — and asserts they agree on loudness, peak, and spectral shape
within tolerance.

We use a CONTROLLED master chain (gain + EQ + brickwall limiter — the tight-null device tier) rather
than a variable agent mix, so the test isolates the export PATH fidelity (not agent-mix variance);
the looser dynamics devices are covered per-device by ``test_phase40_monitor_fidelity``.

Usage::

    python -m orchestrator.daw_export_oracle [project_dir] [--daw-bin PATH]

Three profiles (``--profile``): ``linear`` (the path gate), ``nonlinear`` (the approximation tier
end to end), and ``instrument`` (2026-09-02 — the MIDI half: the console SYNTHESIZES a sampler track
from a fixed MIDI figure; phase 1 renders it dry in isolation and that render is registered as the stem
the offline engine mixes; phase 2 bounces the mix with the synth LIVE through its chain, so the synth,
its native devices, the sum and the master are compared against the offline reference of the same
signal). ``all`` runs the three.

Exits 0 on PASS, 1 on FAIL, 2 if the DAW binary isn't built (SKIP — build it with the ``daw`` skill).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_PROJECT = _REPO / "test_projects" / "suno_demo"
_DEFAULT_DAW_BIN = _REPO / "daw" / "build.noindex" / "MixAgentDAW_artefacts" / "Release" / "MixAgentDAW"

# Tolerances: the master bounce matches the offline deliverable within these. The controlled mix
# below uses the LINEAR bit-exact device tier (gain/EQ/width — Phase 40's tight-null devices),
# rendered at the SAME rate. Comparison is on the MONO downmix with timing-invariant metrics
# (integrated LUFS + a Welch-averaged octave-band spectrum), because Tracktion PDC + the chain's
# always-on lookahead delay line leave a sub-sample time offset that brittle sample/peak metrics
# (and nonlinear sat/clip/limiter) would amplify — those approximation devices are validated
# per-device by test_phase40_monitor_fidelity, not here. This gate is on the export PATH
# (sum / routing / gain-stage / linear DSP / file-write).
# Measured on the demo, 2026-10-06 (deterministic; batch T, reader E4·R5): linear dLUFS 0.18 dB,
# band corr 1.0000, max band d 0.06 dB, true peak d 0.23 dB; instrument 0.16 / 1.0000 / 0.10 /
# 0.18; stereo 0.00 on every axis. The loudness and peak tolerances are ~2.5x what is measured.
# (LUFS_TOL_DB was 1.5 against a "~1.1 dB float32 native vs float64 Pedalboard" offset that has
# not been true since the offline engine rendered mono stems as two channels and read peaks
# through the native limiter's detector: three times the largest figure this profile shows is
# a gate that passes a 1 dB level fault.)
LUFS_TOL_DB = 0.5      # integrated loudness (mono) — the deliverable's loudness within tolerance
BAND_CORR_MIN = 0.99   # octave-band spectral-shape correlation — tonal balance matches
BAND_TOL_DB = 2.5      # max per-octave-band level difference (normalised shape)
TP_TOL_DB = 0.6        # contractual true peak of the full render (compare(): "THE PEAK")
LEN_TOL_S = 0.1        # export length vs the reference — the metrics above cannot see a truncated bounce

# The SIDE axis (2026-09-08). Every metric above is computed on the MONO downmix, and the 2026-09-08
# device audit found the consequence: the whole stereo family — width, band_width, imager, utility —
# is mono-sum-preserving to 0.000000000 dB by construction, so NO divergence in any of them could
# ever register here. The audit measured `band_width` at width 0 removing 6.89 dB of a band's side
# natively where the offline reference removes all of it, and no gate in the repo could see it.
# `_controlled_chain` even carries a comment recording that a width stage was removed from this
# profile in 2026-09-01 precisely because nothing could measure it — the axis was known missing.
# So: L-R gets the same THREE timing-invariant metrics the mono sum gets. Side is a DIFFERENCE of two
# nearly-equal signals when material is centred, so it is only meaningful when there is real side
# content — SIDE_FLOOR_DBFS gates the axis in, and the stereo profile generates material that
# clears it by a wide margin rather than hoping the project has some.
SIDE_FLOOR_DBFS = -60.0   # below this the side is numerical noise and the axis is skipped
SIDE_LUFS_TOL_DB = 1.5    # side-channel loudness (same budget as the mono sum)
SIDE_BAND_CORR_MIN = 0.99 # side-channel octave-band shape (correlation: is the CURVE the same)
SIDE_BAND_TOL_DB = 2.5    # …and its worst per-band error. Correlation alone is blind to a
                          # divergence that keeps the shape and shifts one band's balance — which
                          # is the band_width / imager failure mode this axis exists for. The mono
                          # sum has always had both; the side had only the correlation until the
                          # 2026-09-08 pre-merge review.

# Two PROFILES (2026-09-01, sweep T11 — the oracle covered the linear tier only while the
# invariant was stated without that qualifier). `linear` is the path gate above. `nonlinear`
# runs the approximation tier END TO END through the real export — a stem comp + saturation, a
# master comp + clipper + limiter — with tolerances DERIVED from measurement (the per-sample
# native detectors vs the offline 5 ms block envelopes are a perceptual tier, never a null):
# measured on the demo 2026-09-01 — dLUFS 0.59 dB, octave-band corr 0.9994, max band d 0.47 dB —
# and set at ~2.5-3x the observed deltas, so a real divergence (a device not applied natively, a
# wrong makeup, a limiter that never engages: any of those is several dB) still trips it while the
# known engine difference does not.
PROFILES = {
    "linear":    {"lufs_tol_db": LUFS_TOL_DB, "band_corr_min": BAND_CORR_MIN, "band_tol_db": BAND_TOL_DB},
    # Tightened 2026-10-03 (reader E4·R1). The 1.5 dB / 0.995 / 1.5 dB above were 2.5-3x a delta
    # most of which was ONE disagreement: the offline compressor was Pedalboard's (attack a time
    # constant of ms/2π, on the level, per channel) against the native's linked, literal-ms law.
    # With the offline engine on the native law the same profile measures dLUFS 0.16 dB, band
    # corr 0.9997, max band d 0.40 dB; with Pedalboard's put back it measures 0.59 / 0.9988 /
    # 0.69 and FAILS these - which makes this profile the cross-engine pin for the compressor's
    # timing on the real binary (its stem comp and its master comp both engage on the demo).
    #
    # 2026-10-06 (batch Py3, reader E4·R4): the offline render of a mono stem is two channels
    # now, as the DAW's is, and the side axis is judged here for the first time.
    #
    # 2026-10-06 (batch T, readers E4·R7 + E4·R5): the offline limiter reads peaks through the
    # native detector (no whole-file trim any more), and the stem dynamics sit on the HOTTEST
    # stem - the drums, where the stem comp takes 4.0 dB; on the bass it took nothing, so the
    # "its stem comp and its master comp both engage" above was false for the stem until now
    # (run_oracle asserts it: _stem_dynamics_engage). Measured: dLUFS 0.04 dB, band corr
    # 1.0000, max band d 0.11 dB, true peak d 0.04 dB, side d 0.55 dB.
    "nonlinear": {"lufs_tol_db": 0.3, "band_corr_min": 0.999, "band_tol_db": 0.5},
    # The synth signal is the SAME render on both sides (phase 1's dry bounce is the reference's stem),
    # so the instrument profile is a linear-tier path gate too: the linear tolerances apply.
    "instrument": {"lufs_tol_db": LUFS_TOL_DB, "band_corr_min": BAND_CORR_MIN, "band_tol_db": BAND_TOL_DB},
    # The stereo axis (2026-09-08). Linear devices on generated STEREO material — the demo stems are
    # all 1-channel, which is the other half of why this axis never existed. Same linear tolerances,
    # plus the side metrics above.
    "stereo": {"lufs_tol_db": LUFS_TOL_DB, "band_corr_min": BAND_CORR_MIN, "band_tol_db": BAND_TOL_DB},
}

# The stereo fixture: two generated stems with REAL decorrelated side content, so L-R is a signal
# in its own right rather than the small difference of two large centred ones (which would make the
# side metrics a float32-vs-float64 noise measurement).
STEREO_SECONDS = 6.0
STEREO_SR = 48000

ORACLE_INSTRUMENT = "Synth"   # the track name the console synthesizes (phase 1 + 2 must agree)


def _instrument_chain(session, inst: str) -> None:
    """The linear profile's mix, plus a level + a presence band on the SYNTH track's own
    MonitorChain — the devices the DAW's device dock puts after an instrument. STRONG moves on
    purpose: a -3 dB / +2 dB version measured 0.95 dB LUFS and 1.34 dB of band delta when the
    chain was withheld natively — inside the tolerances, i.e. a gate that could not see its own
    subject (the 2026-09-02 probe). run_oracle asserts the negative control below."""
    _controlled_chain(session)
    rec = session._state["stems"][inst]
    rec["gain_db"] = -9.0
    rec["chain"].append({"type": "eq", "stem_id": inst,
                         "band": {"shape": "peak", "freq_hz": 2500.0, "gain_db": 6.0, "q": 1.0}})


def _write_stereo_fixture(proj) -> None:
    """Generate the `stereo` profile's project: two stems whose L and R share a centred core and
    carry INDEPENDENT side content. Deterministic (fixed seed) — the oracle is a gate, not a
    sampler. Written into a bare directory with no session.json, so stem discovery picks them up
    the way it does for any real project."""
    import soundfile as sf
    rng = np.random.default_rng(20260908)
    n = int(STEREO_SR * STEREO_SECONDS)
    t = np.arange(n) / STEREO_SR
    for name, centre_amp, side_amp, f0 in (("wide_pad", 0.10, 0.30, 110.0),
                                           ("centre_lead", 0.30, 0.06, 330.0)):
        centre = centre_amp * np.sin(2.0 * np.pi * f0 * t)
        left  = centre + side_amp * rng.standard_normal(n) * 0.3
        right = centre + side_amp * rng.standard_normal(n) * 0.3
        sf.write(str(proj / f"{name}.wav"),
                 np.stack([left, right], axis=1).astype(np.float32), STEREO_SR)


def _stereo_chain(session, *, widths: bool = True) -> None:
    """Linear STEREO devices, one per side of the routing: a stem `width` and a master `utility`
    (gain + width). Both are exactly mono-sum-preserving, which is the point — under the mono-only
    comparison this chain is INVISIBLE, and the side metrics are what see it.

    `widths=False` builds the SAME chain with every stereo scale left at unity — the negative
    control. It has to be the same chain, because the control's job is to differ from the real one
    ONLY in the stereo image: the first version rendered a BARE session, so it diverged on the mono
    axis through the gains alone and would have reported "the reference without the width devices
    DIVERGES (as it must)" with the side comparison deleted outright (2026-09-08 pre-merge review).

    `band_width` is here since 2026-10-21 (the band width audit closed 07-stereo-utility.md F4: both
    engines run one LR8 split; it took the band's side 6.9 dB down natively where the offline FFT
    mask took all of it), on the stem that carries the image, 300 Hz - 3 kHz narrowed at width 0 (a wide band, so the old
    law's 6.9 dB is a divergence this axis cannot miss).
    The `imager` is here since 2026-10-21 too (the imager audit closed F4 for it: both engines run the
    same LR8 split): on the master, the lows folded and the highs widened."""
    # A stem's width is the CHANNEL-STRIP SCALAR, not a chain insert: "width" is not in
    # session._STEM_KINDS, so `chain.append({"type": "width", ...})` was filtered straight back out
    # by effective_chain and never rendered. Only the master utility was ever under test, and the
    # docstring's "one per side of the routing" was false (2026-09-08 pre-merge review).
    # It also has to land on the stem that HAS a side: stems sort to
    # ["centre_lead", "wide_pad"], and centre_lead's side_amp is 0.06 — widening it 1.6x moves the
    # summed side by ~0.25 dB, six times under SIDE_LUFS_TOL_DB. wide_pad carries the image.
    # Both scales NARROW, so they compound instead of cancelling. Measured against the honest
    # control: the original 1.6 (widen the pad) x 0.7 (narrow the master) nearly cancelled on the
    # stem that carries the image and moved the summed side by 0.92 dB — UNDER SIDE_LUFS_TOL_DB,
    # so the axis could not see its own devices. 0.4 x 0.6 moves it 11.87 dB while the mono sum
    # is unchanged to 0.000 dB, which is the property being asserted.
    stems = session._state["stems"]
    for name, gain_db, width in (("centre_lead", -2.0, 1.0), ("wide_pad", 0.0, 0.4)):
        if name in stems:
            stems[name]["gain_db"] = gain_db
            stems[name]["width"] = width if widths else 1.0
    if "wide_pad" in stems:
        stems["wide_pad"].setdefault("chain", []).append(
            {"type": "band_width", "stem_id": "wide_pad", "band": "custom", "lo_hz": 300.0, "hi_hz": 3000.0,
             "width": 0.0 if widths else 1.0})
    # EVERY stage of the Utility is under test (2026-10-21, the Utility audit; 07-F5): until then only
    # its gain and width were, and its DC block, one-channel polarity, Bass Mono and balance were
    # compared native against offline nowhere. One channel flipped on the pad's companion (a STEM
    # repair - the agent may not do it on the master), and on the master the DC block, an LR8 Bass
    # Mono at 150 Hz and a balance. With `widths` off they rest at unity (the DC block stays: it is
    # no stereo scale), so the negative control below still has every stereo stage to lose.
    if "centre_lead" in stems:
        stems["centre_lead"].setdefault("chain", []).append(
            {"type": "utility", "stem_id": "centre_lead", "invert_left": bool(widths)})
    session._state["master_gain_db"] = -3.0
    session._state["master_chain"] = [
        {"type": "utility", "stem_id": "master", "gain_db": -1.0,
         "width": 0.6 if widths else 1.0, "dc_filter": True,
         "bass_mono": bool(widths), "bass_mono_hz": 400.0, "balance": 0.25 if widths else 0.0},
        {"type": "imager", "stem_id": "master", "width_low": 0.0 if widths else 1.0, "width_mid": 1.0,
         "width_high": 1.6 if widths else 1.0, "crossover_lo_hz": 250.0, "crossover_hi_hz": 3000.0},
    ]


DRY_RMS_FLOOR_DBFS = -40.0   # a synth that rendered (near) silence would make the profile vacuous


def _delay_insert(session) -> None:
    """A DELAY insert (2026-10-18, the delay audit): no profile carried a space device. Linear, so
    the linear profile carries it - and only that one: the instrument profile's export keeps the
    echoes ringing past the song (8.12 s against the reference's 8.0, over the length tolerance),
    which is the export being right about a tail the offline render does not have. Coverage: a
    whole-master comparison cannot see the ring's handling of a time move (`--delay-audit` is)."""
    stems = list(session._state["stems"])
    if stems:
        first = stems[0]
        session._state["stems"][first]["chain"].append(
            {"type": "return_delay", "stem_id": first, "delay_ms": 375.0, "feedback": 0.4,
             "tone": 0.6, "mix": 0.4})


def _ir_insert(session) -> None:
    """An IR REVERB insert (2026-10-19, the IR reverb audit): the two engines' impulses were two
    different noises (numpy's normal noise, juce::Random's uniform noise - up to 3.9 dB apart per
    band) and no profile carried the device. They are one impulse now, sample for sample, so the
    linear profile compares the reverb itself. A room, so its early reflections are in it; short,
    so the export's tail stays inside the length tolerance."""
    stems = list(session._state["stems"])
    if len(stems) > 1:
        second = stems[1]
        session._state["stems"][second]["chain"].append(
            {"type": "return_convolution", "stem_id": second, "ir": "room", "decay_s": 1.0,
             "pre_delay_ms": 12.0, "tone": 0.4, "mix": 0.35})


def _reverb_insert(session) -> None:
    """A REVERB insert (2026-10-20, the reverb audit): the delay and the IR reverb each had one and
    the algorithmic reverb, the third space device, had none. Coverage, like theirs: a whole-master
    comparison cannot see a pre-delay ride (`--reverb-audit` is that); the law nulls at float32."""
    stems = list(session._state["stems"])
    if len(stems) > 2:
        third = stems[2]
        session._state["stems"][third]["chain"].append(
            {"type": "return_reverb", "stem_id": third, "decay_s": 1.2, "pre_delay_ms": 15.0,
             "tone": 0.6, "mix": 0.3})


def _controlled_chain(session) -> None:
    """A fixed, reproducible LINEAR bit-exact mix: a stem gain + high-shelf, and a master gain + two
    EQ bands — all memoryless/linear, so the offline reference and the native export agree
    closely (no nonlinear or time-varying device to amplify the sub-sample latency offset)."""
    stems = list(session._state["stems"])
    if stems:
        first = stems[0]
        session._state["stems"][first]["gain_db"] = -2.0
        session._state["stems"][first]["chain"].append(
            # `output_db` on purpose (2026-09-08): NO fixture in this file carried one, so the
            # native bank's trim handling was never compared against either Python engine. In a
            # single-Eq-run chain like this profile's it cannot diverge — the nonlinear profile
            # straddles it across a compressor, which is where it could.
            {"type": "eq", "stem_id": first, "output_db": 4.0,
             "band": {"shape": "high_shelf", "freq_hz": 8000.0, "gain_db": -3.0, "q": 0.7}})
        # A gain RIDE (2026-10-03, full audit reader S4·R29). Until then a ride was `print_only`:
        # the offline master carried it and the DAW's monitor and bounce did not, and no profile
        # here had one, so this oracle could not have noticed. It is a native stage now
        # (MonitorChain::applyRides, the same raised-cosine law). On the LOUDEST stem and large,
        # so the whole-master loudness can see it: MEASURED with the native stage taken out
        # (monitor_chain.MAX_NATIVE_RIDES = 0, the ride back in print_only) this profile
        # DIVERGES — dLUFS 3.28 dB against a 1.5 dB tolerance — and with it in, 0.18 dB.
        loud = "vocals" if "vocals" in stems else stems[-1]
        session._state["stems"][loud].setdefault("rides", []).append(
            {"start_s": 1.0, "end_s": 7.0, "delta_db": -9.0})
        # A RANGED eq with an output trim (2026-10-17, the EQ audit): no profile carried one. The
        # native switched a ranged band in at a block edge and played its TRIM over the whole song
        # (offline the trim is inside the range blend) - on the loudest stem, so the master's
        # loudness sees a trim heard outside its range.
        session._state["stems"][loud]["chain"].append(
            {"type": "eq", "stem_id": loud, "output_db": -6.0, "time_range_s": [2.0, 4.0],
             "band": {"shape": "peak", "freq_hz": 3000.0, "gain_db": 6.0, "q": 1.0}})
        # A TILT (2026-10-17, the tilt audit): no profile carried one. Off the default pivot, with a trim.
        session._state["stems"][first]["chain"].append(
            {"type": "tilt", "stem_id": first, "tilt_db": 4.0, "pivot_hz": 2500.0, "output_db": -2.0})
    session._state["master_gain_db"] = -3.0
    session._state["master_chain"] = [
        # A match_eq (2026-09-08): no profile in this file carried one, so the offline FIR and
        # the native bell bank that approximates it had never been compared by any gate — and
        # in the DAW that bank IS the deliverable.
        #
        # MEASURED, like the output_db straddle below: this does NOT gate the fit. With the fit
        # reverted to the raw band gains the profile still PASSES (band corr 1.0000 -> 0.9974,
        # max delta 0.06 -> 0.73 dB, both inside tolerance). A whole-master octave-band
        # correlation is simply too coarse to see a device-level divergence — the same reason
        # the trim straddle is coverage rather than a gate. test_match_eq_fit.py is the gate;
        # this is coverage, so the device is compared between the engines at all.
        {"type": "match_eq", "stem_id": "master",
         "band_gains_db": {"sub": -1.5, "bass": -2.0, "low_mid": -1.0,
                           "mid": 0.5, "high_mid": 1.0, "presence": 1.5,
                           "brilliance": 2.0, "air": 2.5}},
        {"type": "eq", "stem_id": "master",
         "band": {"shape": "low_shelf", "freq_hz": 120.0, "gain_db": 2.0, "q": 0.7}},
        {"type": "eq", "stem_id": "master",
         "band": {"shape": "peak", "freq_hz": 3000.0, "gain_db": -3.0, "q": 1.5}},
        # No `width` here (removed 2026-09-01): the comparison is on the MONO downmix and width
        # scales only the side signal (L'+R' == L+R), and the demo stems are 1-channel anyway —
        # so a width stage was in the chain and "covered" while nothing could ever measure it.
        # Native width has NO oracle coverage; it is a per-device null test's job.
    ]


def _nonlinear_chain(session) -> None:
    """The approximation tier, end to end: the same stem gain + shelf as the linear profile,
    plus a stem compressor and tape saturation, and a master compressor -> clipper -> limiter.
    Every device here is one Phase 40 null-tested per-device in numpy; this is the first time
    the C++ runs them in one graph against the offline render of the same session."""
    _controlled_chain(session)
    stems = list(session._state["stems"])
    if stems:
        # The HOTTEST stem carries the stem dynamics (P2 batch T, reader E4·R5). They sat on
        # stems[0] - the demo's bass, whose peak is -28 dBFS: under the comp's -24 dB threshold
        # and far below anything a 6 dB drive bends, so this profile's stem comp and stem
        # saturation never did anything and "the approximation tier, end to end" was its
        # master devices alone. run_oracle asserts that both engage (_stem_dynamics_engage).
        first = _hottest_stem(session)
        session._state["stems"][first]["chain"].extend([
            {"type": "comp", "stem_id": first, "threshold_db": -24.0, "ratio": 3.0,
             "attack_ms": 10.0, "release_ms": 120.0, "makeup_db": 2.0, "mix": 1.0},
            # oversample 4 WITH mix < 1 (2026-09-06 review): the pair that combed, and the tier
            # no profile here used to reach — every saturation in this file ran fully wet at the
            # base rate. It is here so the two engines are at least COMPARED on this path.
            #
            # It is NOT the gate for the dry alignment, and must not be cited as one. Measured
            # with the alignment removed: band corr 0.9984 -> 0.9980 and max band delta
            # 0.96 -> 1.12 dB, both still inside this profile's tolerance, so the oracle PASSES
            # on the comb. One stem's dry/wet comb simply does not move a whole-master octave-band
            # correlation far enough. The real gate is --shaper-alias, which asserts at mix 0 that
            # the output is the input delayed by EXACTLY the declared latency (bit-exact, with a
            # one-sample-off control) — that is the assertion that fails when this regresses.
            {"type": "saturation", "stem_id": first, "drive_db": 6.0, "curve": "warm",
             "oversample": 4, "mix": 0.5},
            # 2026-09-06: a TIME-VARYING device. Everything else in this profile is static, so
            # nothing here could catch the failure a modulation device actually has — the two
            # engines landing on DIFFERENT points of the LFO sweep. Both derive the phase from
            # the host timeline; if that ever diverges, the band correlation falls here.
            {"type": "chorus", "stem_id": first, "rate_hz": 1.0, "depth": 0.3,
             "delay_ms": 18.0, "mix": 0.4, "voices": 2, "stereo_deg": 90.0},
            # …and a time-varying device with a FEEDBACK LOOP (2026-09-06). The chorus proves
            # the two engines agree on the sweep; only this proves they agree on a RECURSION.
            # The native side runs it per sample, the offline side in blocks that track the
            # sweep — the one device in this profile where the two engines are not even running
            # the same SHAPE of computation, so a divergence in either surfaces here rather than
            # in a unit test that only ever calls one of them. feedback is POSITIVE on purpose:
            # negative at a half blend is nearly an allpass, and a leg that passes by having no
            # audible effect to disagree about is not a leg.
            {"type": "flanger", "stem_id": first, "rate_hz": 0.4, "depth": 0.6,
             "delay_ms": 2.5, "feedback": 0.5, "mix": 0.4, "stereo_deg": 90.0},
            # A SECOND EQ, after the nonlinear devices, so a trim STRADDLES a compressor — the
            # shape that diverged before 2026-09-08 (every eq/tilt `output_db` folded into one
            # scalar paid after the LAST native Eq run, putting a trim on the far side of a comp
            # both Python engines feed it INTO). Only a STEM chain can express it: the master
            # chain is reordered into a fixed device order by effective_master_chain.
            #
            # MEASURED: this profile does NOT gate that bug. Reverting the fix and re-running it
            # still PASSES (dLUFS 0.28, band corr 0.9988) — one stem's trim reordering, diluted
            # into a four-stem master and measured as integrated LUFS plus an octave-band shape,
            # lands well inside the tolerance. `--eq-trim` is the gate; this is COVERAGE, so that
            # a trim is at least compared between the engines at all, which it never was.
            {"type": "eq", "stem_id": first, "output_db": -2.0,
             "band": {"shape": "peak", "freq_hz": 2000.0, "gain_db": 2.5, "q": 1.2}},
            # A DE-ESSER (2026-10-17, the de-esser audit): no profile carried one, and the offline
            # law read a block peak that de-essed several dB harder than the DAW plays. Its
            # engagement is asserted with the comp's (_stem_dynamics_engage). LAST: ahead of the
            # saturation it took enough top end that the saturation no longer engaged.
            {"type": "deess", "stem_id": first, "freq_hz": 5000.0, "threshold_db": -40.0,
             "ratio": 4.0, "attack_ms": 1.0, "release_ms": 60.0, "range_db": -12.0},
            # A LIMITER WORKING HARD at a slow release (2026-10-17, the limiter audit): the two
            # engines' limiters were two laws - read ~3 dB apart at release 300 - and the master's
            # limiter here barely limits. +18 dB into a -6 dBTP stem ceiling takes ~8 dB off the hits.
            {"type": "utility", "stem_id": first, "gain_db": 18.0},
            {"type": "limiter", "stem_id": first, "ceiling_dbtp": -6.0, "release_ms": 300.0,
             "lookahead_ms": 3.0},
            # A GATE (2026-10-17, the gate audit): no profile carried one, and the two engines'
            # detectors read up to 10 dB apart (the native's raw |x| shut inside each cycle).
            {"type": "gate", "stem_id": first, "threshold_db": -30.0, "ratio": 4.0,
             "attack_ms": 1.0, "release_ms": 80.0, "hold_ms": 20.0, "range_db": -20.0,
             "hysteresis_db": 6.0},
        ])
        # A MULTIBAND COMP (2026-10-17, the multiband audit) on ANOTHER stem: no profile carried
        # one, the DAW's complement split made a compressed low band LOUDER in the export while
        # every number here read it quieter, and the upward law was another law. Not on `first`:
        # its chorus and flanger already differ between the engines (31 % RMS on the stem, inside
        # loudness and band tolerances), and an upward band and a hard limiter after them turned
        # that into a peak difference that is not the multiband's. Low band down, high band up.
        other = next((sid for sid in stems if sid != first), None)
        if other is not None:
            session._state["stems"][other]["chain"].append(
                {"type": "mb_comp", "stem_id": other, "attack_ms": 5.0, "release_ms": 150.0,
                 "crossover_lo_hz": 120.0, "crossover_hi_hz": 3000.0,
                 "low": {"threshold_db": -40.0, "ratio": 6.0, "makeup_db": 0.0},
                 "mid": {"threshold_db": 0.0, "ratio": 1.0, "makeup_db": 0.0},
                 "high": {"threshold_db": -50.0, "ratio": 2.0, "makeup_db": 0.0, "up": True}})
            # ...and OTT after it (2026-10-17, the OTT audit): no profile carried one either, and
            # the DAW split it by the same complement. Full depth on the LOW band alone - where a
            # band that is not a band shows: at the default depth on every band the old split
            # read 0.42 dB of band difference against the 0.5 tolerance, and passed.
            session._state["stems"][other]["chain"].append(
                {"type": "ott", "stem_id": other, "depth": 1.0, "time": 1.0,
                 "in_gain_db": 0.0, "out_gain_db": 0.0,
                 "low": {"up": 1.0, "down": 1.0, "gain_db": 0.0},
                 "mid": {"up": 0.0, "down": 0.0, "gain_db": 0.0},
                 "high": {"up": 0.0, "down": 0.0, "gain_db": 0.0},
                 "crossover_lo_hz": 150.0, "crossover_hi_hz": 2000.0})
        # AN OVERDRIVE (2026-10-17, the overdrive audit): no profile carried one. Oversampled, the
        # band-pass on, half wet - every part of its law - after the transient shaper on the third stem.
        # A TRANSIENT SHAPER (2026-10-17, the transient audit) on a third stem: no profile carried
        # one, and the two engines followed the level by different laws. This case shows it engages
        # and the engines agree (0.13 dB band difference); it does NOT fail on the old law (0.42, the
        # most of four settings tried - a shaper's effect is in time and this compares the mix's
        # bands over the file, after a limiter that takes a level back). The law's proof is
        # test_transient_audit_2026_10_17.py (a per-sample transcription) and --transient-audit.
        third = next((sid for sid in stems if sid not in (first, other)), None)
        if third is not None:
            session._state["stems"][third]["chain"].append(
                {"type": "transient", "stem_id": third, "attack_db": 6.0, "sustain_db": -4.0,
                 "speed": 1.0, "output_db": 0.0})
            session._state["stems"][third]["chain"].append(
                {"type": "overdrive", "stem_id": third, "drive_db": 18.0, "filter_on": True,
                 "freq_hz": 1500.0, "q": 1.0, "tone": 0.7, "output_db": 3.0, "mix": 0.5, "oversample": 4})
            # AN AUTO FILTER (Phase 63), keyed from `other`: the LFO on the timeline, the stereo
            # offset, the envelope on a TRIGGER rendered as the Duck's key is (the oracle's edit now
            # wires sidechains: dawsession::wireSidechainsByName). `other` because its chain already
            # agrees between the engines - a trigger carrying a difference would carry it into the
            # envelope. The per-sample law is --auto-filter's (the fixture, ~1e-8); this is the
            # graph, the timeline and the trigger, compared end to end.
            if other is not None:
                session._state["stems"][third]["chain"].append(
                    {"type": "auto_filter", "stem_id": third, "filter_type": 0, "slope_db_oct": 24,
                     "cutoff_hz": 1200.0, "resonance": 0.25, "lfo_amount_oct": 1.5, "lfo_shape": 0,
                     "lfo_rate_hz": 0.5, "lfo_stereo_deg": 90.0, "env_amount_oct": 1.5,
                     "env_attack_ms": 5.0, "env_release_ms": 150.0, "key_stem_id": other, "mix": 1.0})
        # A STEEP CUT ON THE MID and a TILT SHELF (2026-10-17, the EQ audit) on the fourth stem: the
        # profiles built peaks and shelves only - no sloped cut, no M/S band, no tilt_shelf.
        fourth = next((sid for sid in stems if sid not in (first, other, third)), None)
        if fourth is not None:
            session._state["stems"][fourth]["chain"].extend([
                {"type": "eq", "stem_id": fourth, "channel": "mid",
                 "band": {"shape": "hpf", "freq_hz": 180.0, "gain_db": 0.0, "q": 0.707, "slope_db_oct": 48}},
                {"type": "eq", "stem_id": fourth,
                 "band": {"shape": "tilt_shelf", "freq_hz": 1200.0, "gain_db": 5.0, "q": 0.7}},
                # TWO DYNAMIC EQ BANDS (2026-10-17, the dynamic EQ audit): no profile carried one, the
                # engines ran different bands and detectors, and every entry folded into the first
                # one's step and range. The second is ranged: the first must not take its range.
                {"type": "dynamic_eq", "stem_id": fourth, "freq_hz": 2500.0, "q": 1.5,
                 "threshold_db": -45.0, "ratio": 4.0, "attack_ms": 5.0, "release_ms": 100.0, "gain_db": -9.0},
                {"type": "dynamic_eq", "stem_id": fourth, "freq_hz": 300.0, "q": 1.0, "time_range_s": [2.0, 5.0],
                 "threshold_db": -45.0, "ratio": 4.0, "attack_ms": 5.0, "release_ms": 100.0, "gain_db": -9.0},
                # A MULTIBAND SATURATION (2026-10-17, the multiband saturation audit): no profile carried
                # one, and the engines split it differently (the native by the complement). The low band
                # driven, the mid and high dry - where the complement put a low tone into the "mid".
                {"type": "saturation", "stem_id": fourth, "drive_db": 3.0, "curve": 0, "mix": 1.0,
                 "crossover_lo_hz": 200.0, "crossover_hi_hz": 3000.0,
                 "low": {"drive_db": 18.0, "output_db": 0.0, "mix": 1.0},
                 "mid": {"drive_db": 0.0, "output_db": 0.0, "mix": 0.0},
                 "high": {"drive_db": 0.0, "output_db": 0.0, "mix": 0.0}}])
    session._state["master_chain"].extend([
        {"type": "comp", "stem_id": "master", "threshold_db": -18.0, "ratio": 2.0,
         "attack_ms": 20.0, "release_ms": 200.0, "makeup_db": 0.0, "mix": 1.0},
        # OVERSAMPLED, and low enough to bite (2026-10-17, the clipper audit): at -1 dB / 1x the
        # oversampled path - the two engines' different filters and the latency - had no case.
        {"type": "clip", "stem_id": "master", "threshold_db": -10.0, "hardness": 0.5, "oversample": 4},
        # release 300 (2026-10-17, the limiter audit): at the default 50 the two engines' limiters
        # - then two different laws - read within 0.04 dB; at 300 they read ~3 dB apart.
        {"type": "limiter", "stem_id": "master", "ceiling_dbtp": -1.0, "target_lufs_i": -14.0,
         "release_ms": 300.0},
    ])
    return first if stems else None


def _hottest_stem(session) -> str:
    """The stem whose FILE peaks highest - the one a threshold and a drive are sure to reach."""
    import soundfile as sf

    def peak(stem_id: str) -> float:
        rec = session._state["stems"][stem_id]
        try:
            audio, _ = sf.read(str(Path(session.project_dir) / str(rec.get("path") or f"{stem_id}.wav")),
                               always_2d=True)
        except Exception:  # noqa: BLE001 - an unreadable stem is not the hottest
            return 0.0
        return float(np.max(np.abs(audio))) if audio.size else 0.0

    stems = list(session._state["stems"])
    return max(stems, key=lambda stem_id: (peak(stem_id), -stems.index(stem_id)))


#: What the stem dynamics must DO on the fixture for the nonlinear profile to mean anything:
#: the compressor's mean gain reduction, and how far the saturation moves the signal.
STEM_COMP_GR_MIN_DB = 1.0
STEM_SAT_CHANGE_MIN_DBFS = -50.0
STEM_DEESS_CHANGE_MIN_DBFS = -60.0
STEM_GATE_CHANGE_MIN_DBFS = -60.0


STEM_MBCOMP_CHANGE_MIN_DBFS = -40.0


def _master_device_engages(session, kind: str) -> float:
    """The RMS in dBFS of what the master's `kind` changes on the offline master render - its
    vacuity guard (the clipper audit). -inf when the master carries none."""
    import copy

    from mix_engine import dsp

    if not any(e.get("type") == kind for e in session._state.get("master_chain", [])):
        return float("-inf")

    def rendered(drop: bool) -> np.ndarray:
        probe = copy.copy(session)
        probe._state = copy.deepcopy(session._state)
        if drop:
            probe._state["master_chain"] = [e for e in probe._state["master_chain"] if e.get("type") != kind]
        audio, _ = dsp.render_master(probe, report={})
        return np.asarray(audio, dtype=np.float64)

    return _rms_db(rendered(False) - rendered(True))


def _device_engages(session, kind: str, match=None) -> tuple[str | None, float]:
    """(the stem that carries the profile's `kind` - mb_comp, ott, transient -, the RMS in dBFS of what it
    changes on the offline render) - its vacuity guard. None and -inf when no stem carries one."""
    import copy

    from mix_engine import dsp

    hit = (lambda e: e.get("type") == kind and (match is None or match(e)))
    stem = next((sid for sid, rec in session._state["stems"].items()
                 if any(hit(e) for e in rec.get("chain", []))), None)
    if stem is None:
        return None, float("-inf")

    def rendered(drop: bool) -> np.ndarray:
        probe = copy.copy(session)
        probe._state = copy.deepcopy(session._state)
        if drop:
            chain = probe._state["stems"][stem]["chain"]
            chain[:] = [e for e in chain if not hit(e)]
        audio, _ = dsp.render_stem(probe, stem)
        return np.asarray(audio, dtype=np.float64)

    return stem, _rms_db(rendered(False) - rendered(True))


def _stem_dynamics_engage(session, stem: str) -> tuple[float, float, float, float]:
    """(the stem comp's gain reduction in dB, the RMS of what the saturation changes in dBFS),
    measured on the offline render of `stem` - the vacuity guard for the nonlinear profile's
    stem tier. A device that leaves its input alone agrees with itself in any engine."""
    import copy

    from mix_engine import dsp

    def rendered(edit) -> np.ndarray:
        probe = copy.copy(session)
        probe._state = copy.deepcopy(session._state)
        edit(probe._state["stems"][stem]["chain"])
        audio, _ = dsp.render_stem(probe, stem)
        return np.asarray(audio, dtype=np.float64)

    def keep(chain, kinds):                      # the chain up to and including the last of `kinds`
        last = max((i for i, e in enumerate(chain) if e.get("type") in kinds), default=-1)
        del chain[last + 1:]

    def comp_in(chain):
        keep(chain, ("comp",))
        chain[-1]["makeup_db"] = 0.0

    def comp_out(chain):
        keep(chain, ("comp",))
        chain.pop()

    def sat_in(chain):
        keep(chain, ("saturation",))
        chain[-1]["mix"] = 1.0

    def sat_out(chain):
        keep(chain, ("saturation",))
        chain.pop()

    def deess_in(chain):
        keep(chain, ("deess",))

    def deess_out(chain):
        keep(chain, ("deess",))
        chain.pop()

    gr_db = _rms_db(rendered(comp_out)) - _rms_db(rendered(comp_in))
    wet, dry = rendered(sat_in), rendered(sat_out)
    n = min(len(wet), len(dry))
    de_wet, de_dry = rendered(deess_in), rendered(deess_out)
    m = min(len(de_wet), len(de_dry))

    def gate_in(chain):
        keep(chain, ("gate",))

    def gate_out(chain):
        keep(chain, ("gate",))
        chain.pop()

    g_wet, g_dry = rendered(gate_in), rendered(gate_out)
    g = min(len(g_wet), len(g_dry))
    return (float(gr_db), float(_rms_db(wet[:n] - dry[:n])), float(_rms_db(de_wet[:m] - de_dry[:m])),
            float(_rms_db(g_wet[:g] - g_dry[:g])))


def _read_mono(path: Path) -> tuple[np.ndarray, int]:
    """Read a wav and downmix to mono. The two engines used to differ in channel LAYOUT (the
    offline sum of mono stems was 1-channel until batch Py3; the DAW export is dual-mono stereo),
    which alone was a ~3 dB BS.1770 channel-count artifact — comparing the mono downmix removed
    it, and the per-stem content is what we actually want to compare."""
    import soundfile as sf
    audio, sr = sf.read(str(path), always_2d=True)
    return np.asarray(audio, dtype=np.float64).mean(axis=1), int(sr)


def _mid_side(full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(mono sum, side) from a render of either shape. A 1-channel render has no side at all, and
    returns an explicit zero rather than a near-zero, so the caller's floor check is unambiguous."""
    a = np.asarray(full, dtype=np.float64)
    if a.ndim == 1 or a.shape[1] < 2:
        mono = a if a.ndim == 1 else a[:, 0]
        return mono, np.zeros_like(mono)
    return a.mean(axis=1), a[:, 0] - a[:, 1]


def _read_stereo(path) -> tuple[np.ndarray, np.ndarray, int]:
    import soundfile as sf
    audio, sr = sf.read(str(path), always_2d=True)
    mono, side = _mid_side(audio)
    return mono, side, int(sr)


def _as_frames(full: np.ndarray) -> np.ndarray:
    a = np.asarray(full, dtype=np.float64)
    return a[:, np.newaxis] if a.ndim == 1 else a


def _rms_db(x: np.ndarray) -> float:
    return 20.0 * np.log10(float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2))) + 1e-12)


def _lufs(mono: np.ndarray, sr: int) -> float:
    import pyloudnorm as pyln
    return float(pyln.Meter(sr).integrated_loudness(mono))


def _band_shape_db(mono: np.ndarray, sr: int, n: int = 10) -> np.ndarray:
    """Normalised octave-band spectrum (dB, peak-band = 0) from a Welch-averaged PSD — averaging
    many short windows smooths the comb-ripple a sub-sample time offset would otherwise put in a
    single full-length FFT, so the shape reflects real tonal balance, not alignment."""
    from scipy.signal import welch
    freqs, psd = welch(mono, fs=sr, nperseg=min(4096, len(mono)))
    edges = np.logspace(np.log10(30.0), np.log10(min(16000.0, sr / 2 * 0.9)), n + 1)
    bands = np.array([
        np.sqrt(np.mean(psd[(freqs >= edges[i]) & (freqs < edges[i + 1])]))
        if np.any((freqs >= edges[i]) & (freqs < edges[i + 1])) else 0.0
        for i in range(n)])
    return 10.0 * np.log10(bands / (np.max(bands) or 1e-12) + 1e-12)


def compare(ref_mono: np.ndarray, ref_sr: int,
            daw_mono: np.ndarray, daw_sr: int, profile: str = "linear",
            ref_side: np.ndarray | None = None,
            daw_side: np.ndarray | None = None,
            ref_stereo: bool = False,
            ref_tp_dbtp: float | None = None,
            daw_tp_dbtp: float | None = None) -> tuple[bool, list[str]]:
    """The verdict on two mono renders, as (ok, report lines).

    Rate and LENGTH are judged first. The tolerant metrics below are deliberately
    timing-invariant, which makes them length-invariant too: a bounce that stopped at
    the last clip, dropped a reverb tail, or lost most of the song to a render-length
    off-by-one sailed straight through them (2026-09-01 sweep: the demo master with
    75 % of its tail cut gave dLUFS 0.02 dB, band corr 0.999 -> PASS). Until then the
    two lengths were only printed. LEN_TOL_S leaves room for a block of padding or a
    lookahead tail, not for a missing bar."""
    tol = PROFILES[profile]
    lufs_tol, corr_min, band_tol = tol["lufs_tol_db"], tol["band_corr_min"], tol["band_tol_db"]
    lines = [f"[oracle] [{profile}] offline (ref) vs DAW export @ {ref_sr}/{daw_sr} Hz, "
             f"{len(ref_mono) / ref_sr:.3f} s vs {len(daw_mono) / max(int(daw_sr), 1):.3f} s"]
    if int(daw_sr) != int(ref_sr):
        lines.append(f"[oracle] FAIL - DAW export rate {daw_sr} Hz != reference {ref_sr} Hz")
        return False, lines
    d_len_s = abs(len(daw_mono) - len(ref_mono)) / ref_sr
    if d_len_s > LEN_TOL_S:
        lines.append(f"[oracle] FAIL - DAW export length differs from the reference by "
                     f"{d_len_s:.3f} s (tol {LEN_TOL_S} s): a truncated or over-long bounce")
        return False, lines

    # loudness + spectral shape on the MONO downmix (channel- + timing-tolerant)
    lufs_ref, lufs_daw = _lufs(ref_mono, ref_sr), _lufs(daw_mono, daw_sr)
    shape_ref, shape_daw = _band_shape_db(ref_mono, ref_sr), _band_shape_db(daw_mono, daw_sr)
    d_lufs = abs(lufs_ref - lufs_daw)
    d_band = float(np.max(np.abs(shape_ref - shape_daw)))
    band_corr = float(np.corrcoef(shape_ref, shape_daw)[0, 1])
    ok = bool(d_lufs <= lufs_tol and band_corr >= corr_min and d_band <= band_tol)
    lines.append(f"[oracle]   LUFS  {lufs_ref:6.2f} vs {lufs_daw:6.2f}  -> dLUFS {d_lufs:.2f} dB "
                 f"(tol {lufs_tol})")
    lines.append(f"[oracle]   octave-band corr {band_corr:.4f} (min {corr_min}), "
                 f"max d {d_band:.2f} dB (tol {band_tol})")
    # THE PEAK (reader E4·R5): this module's first line and the console's own note both said
    # the two renders are compared on "loudness, peak and spectral shape", and no peak was
    # read. The contractual true peak of each FULL render (analysis._true_peak_dbtp: 4x
    # oversampled, timing-invariant), within the profile's `tp_tol_db`. A limiter that lets a
    # peak through in one engine and not the other is a deliverable past its ceiling, which no
    # loudness or band figure shows.
    if ref_tp_dbtp is not None and daw_tp_dbtp is not None:
        tp_tol = tol.get("tp_tol_db", TP_TOL_DB)
        d_tp = abs(ref_tp_dbtp - daw_tp_dbtp)
        ok = ok and bool(d_tp <= tp_tol)
        lines.append(f"[oracle]   true peak {ref_tp_dbtp:6.2f} vs {daw_tp_dbtp:6.2f} dBTP "
                     f"-> d {d_tp:.2f} dB (tol {tp_tol})")
    # THE SIDE AXIS. Everything above is the mono sum, and the whole stereo device family is
    # mono-sum-preserving by construction — so without this, a width/imager/band_width/utility
    # divergence of any size passes. Skipped, loudly, when the material has no side to measure:
    # a silent side would otherwise "agree" perfectly and read as coverage.
    if ref_side is not None and daw_side is not None:
        ref_side_db, daw_side_db = _rms_db(ref_side), _rms_db(daw_side)
        if not ref_stereo:
            # The offline render of a project whose stems are all 1-channel is itself 1-CHANNEL, so
            # it cannot carry a side at all, while the DAW's dual-mono path can. Judging the two
            # against each other here would fail the profile for a channel-LAYOUT difference rather
            # than an export-path fault. Reported rather than skipped silently, because the number
            # is worth seeing: on the nonlinear profile the DAW's side measures ~-37 dBFS RMS —
            # the chorus/flanger `stereo_deg` spread existing in the DAW's deliverable and not in
            # the offline engine's render of the same session (2026-09-08, worth its own finding).
            lines.append(f"[oracle]   side: n/a — the reference render is 1-channel (mono project); "
                         f"the DAW's dual-mono path measured {daw_side_db:.1f} dBFS RMS of side, "
                         f"which this comparison cannot judge")
        elif ref_side_db < SIDE_FLOOR_DBFS and daw_side_db < SIDE_FLOOR_DBFS:
            lines.append(f"[oracle]   side: n/a — no side content "
                         f"({ref_side_db:.1f} / {daw_side_db:.1f} dBFS RMS, floor {SIDE_FLOOR_DBFS})")
        elif (ref_side_db < SIDE_FLOOR_DBFS) != (daw_side_db < SIDE_FLOOR_DBFS):
            # Both engines started stereo and one lost its image — a collapsed or invented stereo
            # field, which is exactly the failure this axis exists for.
            lines.append(f"[oracle]   side  {ref_side_db:6.2f} vs {daw_side_db:6.2f} dBFS RMS")
            lines.append("[oracle] FAIL - one engine's render has a stereo image and the other's does not")
            ok = False
        else:
            s_lufs_ref, s_lufs_daw = _lufs(ref_side, ref_sr), _lufs(daw_side, daw_sr)
            s_shape_ref, s_shape_daw = _band_shape_db(ref_side, ref_sr), _band_shape_db(daw_side, daw_sr)
            s_corr = float(np.corrcoef(s_shape_ref, s_shape_daw)[0, 1])
            s_band = float(np.max(np.abs(s_shape_ref - s_shape_daw)))
            # SIDE_FLOOR_DBFS is an unweighted RMS gate; `_lufs` is K-weighted with pyloudnorm's
            # own -70 LUFS absolute gate. There is a band between them — a sub-heavy side above
            # -60 dBFS RMS whose K-weighted loudness is -inf on BOTH sides — where `abs(-inf -
            # -inf)` is nan, and `nan <= tol` is False. That failed the profile on two renders
            # that agree bit-for-bit (2026-09-08 pre-merge review). Fall back to the unweighted
            # RMS difference, which is defined everywhere this axis engages.
            if np.isfinite(s_lufs_ref) and np.isfinite(s_lufs_daw):
                d_s_lufs, level_what = abs(s_lufs_ref - s_lufs_daw), "LUFS"
                l_ref, l_daw = s_lufs_ref, s_lufs_daw
            else:
                d_s_lufs, level_what = abs(ref_side_db - daw_side_db), "RMS (below the LUFS gate)"
                l_ref, l_daw = ref_side_db, daw_side_db
            side_ok = bool(d_s_lufs <= SIDE_LUFS_TOL_DB and s_corr >= SIDE_BAND_CORR_MIN
                           and s_band <= SIDE_BAND_TOL_DB)
            ok = ok and side_ok
            lines.append(f"[oracle]   side {level_what} {l_ref:6.2f} vs {l_daw:6.2f} "
                         f"-> d {d_s_lufs:.2f} dB (tol {SIDE_LUFS_TOL_DB}); "
                         f"side band corr {s_corr:.4f} (min {SIDE_BAND_CORR_MIN}), "
                         f"max d {s_band:.2f} dB (tol {SIDE_BAND_TOL_DB})")
    lines.append(f"[oracle] {'PASS' if ok else 'FAIL'} - [{profile}] DAW export "
                 f"{'matches' if ok else 'DIVERGES FROM'} the offline master within tolerance")
    return ok, lines


def run_oracle(project: Path, daw_bin: Path, profile: str = "linear") -> int:
    if not daw_bin.exists():
        print(f"[oracle] SKIP — DAW binary not built: {daw_bin}\n"
              f"         build it with the `daw` skill (cmake --build daw/build.noindex --target MixAgentDAW)",
              file=sys.stderr)
        return 2

    sys.path.insert(0, str(_REPO / "server"))
    import soundfile as sf

    from mix_engine import dsp
    from mix_engine.session import Session, MASTER
    from orchestrator.daw_bridge import resolve_chains

    with tempfile.TemporaryDirectory() as td:
        proj = Path(td) / "proj"
        if profile == "stereo":
            # The demo's stems are all 1-CHANNEL, so no copy of it can exercise a stereo device.
            # A bare directory + generated stems is the fixture; discovery finds them the same way
            # it finds any project's (Phase 50 made the console's loader discovery-based too).
            proj.mkdir(parents=True)
            _write_stereo_fixture(proj)
        else:
            shutil.copytree(project, proj)

        extra_env: dict[str, str] = {}
        if profile == "instrument":
            # PHASE 1: the console synthesizes the instrument track and renders it DRY, in
            # isolation, at the project's rate. That render IS the stem the offline engine mixes
            # (discovery picks it up from the project root), so both engines see the same synth
            # signal and the comparison is on the path — the track's devices, the sum, the master.
            wavs = sorted(proj.glob("*.wav"))
            sr_env = str(int(sf.info(str(wavs[0])).samplerate)) if wavs else "0"
            dry = proj / f"{ORACLE_INSTRUMENT}.wav"
            env1 = {**os.environ, "MIXAGENT_ORACLE_INSTRUMENT": ORACLE_INSTRUMENT,
                    "MIXAGENT_ORACLE_DRY_OUT": str(dry), "MIXAGENT_ORACLE_SR": sr_env}
            res = subprocess.run([str(daw_bin), "--export-oracle", str(proj)],
                                 env=env1, capture_output=True, text=True)
            if res.returncode != 0 or not dry.exists():
                print(res.stdout, file=sys.stderr)
                print(res.stderr, file=sys.stderr)
                print(f"[oracle] FAIL — the DAW did not render the instrument track dry to {dry}",
                      file=sys.stderr)
                return 1
            extra_env = {"MIXAGENT_ORACLE_INSTRUMENT": ORACLE_INSTRUMENT}

        session = Session(proj)
        # The native console export (dawsession::loadProjectStems) loads only the four demo
        # stems vocals/drums/bass/other. The demo's session.json ALSO registers the reference
        # `master.wav` AS a stem (a name colliding with the reserved master bus) — so the offline
        # `render_master` would sum the full mix a SECOND time, ~5 dB hotter than the native, and
        # the path comparison would be invalid (apples-to-oranges stem sets). Drop it so both
        # engines mix the SAME stems. (Real projects don't name a stem "master"; this is demo data.)
        session._state["stems"].pop(MASTER, None)
        if profile == "instrument":
            if ORACLE_INSTRUMENT not in session._state["stems"]:
                print(f"[oracle] FAIL — the dry render was not discovered as stem {ORACLE_INSTRUMENT!r}",
                      file=sys.stderr)
                return 1
            # VACUITY GUARDS (the repo's rule: a gate must be able to see its own subject).
            # (1) the synth actually rendered — a silent dry wav would match a silent live synth.
            x, _ = sf.read(str(proj / f"{ORACLE_INSTRUMENT}.wav"), dtype="float64", always_2d=True)
            dry_rms_db = 20.0 * np.log10(float(np.sqrt(np.mean(x ** 2))) + 1e-12)
            if dry_rms_db < DRY_RMS_FLOOR_DBFS:
                print(f"[oracle] FAIL — the instrument's dry render is silent ({dry_rms_db:.1f} dBFS RMS)",
                      file=sys.stderr)
                return 1
            # (2) the synth's chain is LOUD enough to be seen: the reference WITHOUT it must
            # diverge from the reference WITH it beyond this profile's tolerances, or a native
            # graph that never applied the chain would pass (measured 2026-09-02: -3 dB / +2 dB
            # slipped under; -9 dB / +6 dB does not).
            probe = Session(proj)
            probe._state["stems"].pop(MASTER, None)
            _controlled_chain(probe)
            without_full, without_sr = dsp.render_master(probe, report={})
            without_mono = (np.asarray(without_full, dtype=np.float64).mean(axis=1)
                            if without_full.ndim == 2 else np.asarray(without_full, dtype=np.float64))
            _instrument_chain(session, ORACLE_INSTRUMENT)
        elif profile == "stereo":
            # VACUITY GUARD, the same shape the instrument profile uses: the reference WITHOUT the
            # stereo devices must FAIL this profile's own gate. If it passed, a native graph that
            # dropped every width stage would pass too — which is precisely the state this axis was
            # added to end, since the mono metrics cannot see a width device at all.
            probe = Session(proj)
            probe._state["stems"].pop(MASTER, None)
            _stereo_chain(probe, widths=False)   # …the SAME chain, stereo scales at unity
            without_full, without_sr = dsp.render_master(probe, report={})
            without_mono, without_side = _mid_side(without_full)
            _stereo_chain(session)
        elif profile == "nonlinear":
            hot = _nonlinear_chain(session)
            # VACUITY GUARD (reader E4·R5): the stem tier has to DO something on this fixture.
            gr_db, sat_dbfs, deess_dbfs, gate_dbfs = _stem_dynamics_engage(session, hot)
            if (gr_db < STEM_COMP_GR_MIN_DB or sat_dbfs < STEM_SAT_CHANGE_MIN_DBFS
                    or deess_dbfs < STEM_DEESS_CHANGE_MIN_DBFS or gate_dbfs < STEM_GATE_CHANGE_MIN_DBFS):
                print(f"[oracle] FAIL - negative control: on stem {hot!r} the stem compressor takes "
                      f"{gr_db:.2f} dB (min {STEM_COMP_GR_MIN_DB}) and the saturation changes "
                      f"{sat_dbfs:.1f} dBFS RMS (min {STEM_SAT_CHANGE_MIN_DBFS}) and the de-esser "
                      f"{deess_dbfs:.1f} dBFS RMS (min {STEM_DEESS_CHANGE_MIN_DBFS}) and the gate "
                      f"{gate_dbfs:.1f} dBFS RMS (min {STEM_GATE_CHANGE_MIN_DBFS}): a device that "
                      f"does nothing agrees in any engine, so this profile would not be "
                      f"checking the stem tier it names", file=sys.stderr)
                return 1
            print(f"[oracle] [nonlinear] negative control: on stem {hot!r} the stem compressor "
                  f"takes {gr_db:.2f} dB, the saturation changes {sat_dbfs:.1f} dBFS RMS and the "
                  f"de-esser {deess_dbfs:.1f} dBFS RMS and the gate {gate_dbfs:.1f} (all engage, as they must)")
            clip_dbfs = _master_device_engages(session, "clip")
            if clip_dbfs < STEM_MBCOMP_CHANGE_MIN_DBFS:
                print(f"[oracle] FAIL - negative control: the master clipper changes {clip_dbfs:.1f} dBFS "
                      f"RMS (min {STEM_MBCOMP_CHANGE_MIN_DBFS}): a device that does nothing agrees in any "
                      f"engine", file=sys.stderr)
                return 1
            print(f"[oracle] [nonlinear] negative control: the master clipper changes {clip_dbfs:.1f} "
                  f"dBFS RMS (it engages, as it must)")
            mbs_stem, mbs_dbfs = _device_engages(session, "saturation", match=lambda e: isinstance(e.get("low"), dict))
            if mbs_stem is not None and mbs_dbfs < STEM_MBCOMP_CHANGE_MIN_DBFS:
                print(f"[oracle] FAIL - negative control: the multiband saturation on stem {mbs_stem!r} changes "
                      f"{mbs_dbfs:.1f} dBFS RMS (min {STEM_MBCOMP_CHANGE_MIN_DBFS})", file=sys.stderr)
                return 1
            if mbs_stem is not None:
                print(f"[oracle] [nonlinear] negative control: the multiband saturation on stem {mbs_stem!r} "
                      f"changes {mbs_dbfs:.1f} dBFS RMS (it engages, as it must)")
            for kind, name in (("mb_comp", "multiband comp"), ("ott", "OTT"),
                               ("transient", "transient shaper"), ("dynamic_eq", "dynamic EQ"),
                               ("overdrive", "overdrive"), ("auto_filter", "auto filter")):
                band_stem, band_dbfs = _device_engages(session, kind)
                if band_stem is not None and band_dbfs < STEM_MBCOMP_CHANGE_MIN_DBFS:
                    print(f"[oracle] FAIL - negative control: the {name} on stem {band_stem!r} "
                          f"changes {band_dbfs:.1f} dBFS RMS (min {STEM_MBCOMP_CHANGE_MIN_DBFS}): a "
                          f"device that does nothing agrees in any engine", file=sys.stderr)
                    return 1
                if band_stem is not None:
                    print(f"[oracle] [nonlinear] negative control: the {name} on stem "
                          f"{band_stem!r} changes {band_dbfs:.1f} dBFS RMS (it engages, as it must)")
        else:
            _controlled_chain(session)
            _delay_insert(session)
            _ir_insert(session)
            _reverb_insert(session)

        # offline render — the reference deliverable (renders the in-memory mutated session)
        report: dict = {}
        ref_full, ref_sr = dsp.render_master(session, report=report)
        ref_mono, ref_side = _mid_side(ref_full)
        if profile == "stereo":
            side_db = _rms_db(ref_side)
            if side_db < SIDE_FLOOR_DBFS:
                print(f"[oracle] FAIL — the stereo fixture has no side content ({side_db:.1f} dBFS "
                      f"RMS): the axis this profile exists for would be skipped", file=sys.stderr)
                return 1
            vac_ok, _l = compare(ref_mono, ref_sr, without_mono, without_sr, profile,
                                 ref_side, without_side, ref_stereo=True)
            if vac_ok:
                print("[oracle] FAIL — negative control: the reference WITHOUT the stereo devices "
                      "passes the gate, so this profile cannot see what it exists to check",
                      file=sys.stderr)
                return 1
            print(f"[oracle] [stereo] negative control: side {side_db:.1f} dBFS RMS; the reference "
                  f"without the width devices DIVERGES (as it must)")

        if profile == "instrument":
            vac_ok, _vac_lines = compare(ref_mono, ref_sr, without_mono, without_sr, profile)
            if vac_ok:
                print("[oracle] FAIL — negative control: the reference WITHOUT the synth's chain passes the "
                      "gate, so this profile cannot see the chain it exists to check", file=sys.stderr)
                return 1
            print(f"[oracle] [instrument] negative control: dry synth {dry_rms_db:.1f} dBFS RMS; the "
                  f"reference without the synth's chain DIVERGES (as it must)")

        # resolve the same mix into the DAW's monitor-chain contracts + the loudness-correct makeup
        makeup_db = float(report.get("limiter_makeup_db", 0.0) or 0.0)
        chains = resolve_chains(session._state, makeup_db, report.get("pre_limiter_lufs_i"),
                                sample_rate=session.project_sample_rate())
        (proj / "daw_chains.json").write_text(json.dumps(chains))
        # The offline reference applies `target_lufs_i - pre_lufs` of makeup at RENDER time; the
        # DAW can only match it if that resolved value reached the chain we just pushed. On the
        # demo this term is 4.8 dB against a 1.5 dB tolerance (2026-09-08 audit, 01-dynamics-core
        # F5), so the leg could not distinguish "the makeup plumbing broke" from "pass" — it was
        # passing on a term the profile never states. Assert the plumbing, so this leg gates the
        # export path rather than depending on something outside it.
        if abs(makeup_db) > 0.05:
            # chains is {stem_id: {"entries": [...]}} — the monitor contract's shape.
            carried = [e.get("makeup_db")
                       for chain in chains.values() if isinstance(chain, dict)
                       for e in (chain.get("entries") or []) if isinstance(e, dict)
                       and e.get("type") == "limiter"]
            if not carried or not any(abs(float(v or 0.0) - makeup_db) < 0.05 for v in carried):
                print(f"[oracle] FAIL — the offline reference applied {makeup_db:+.2f} dB of limiter "
                      f"makeup and the chain pushed to the DAW carries {carried!r}. The two engines "
                      f"would be compared at different loudness through a term this profile never "
                      f"states.", file=sys.stderr)
                return 1
            print(f"[oracle] [{profile}] limiter makeup {makeup_db:+.2f} dB resolved into the DAW chain")

        # render through the real DAW export path — at the SAME rate as the offline reference, so
        # the comparison isn't muddied by a 48k<->44.1k resample. (Instrument profile: PHASE 2 —
        # the console drops the audio track discovery made from the dry wav and plays the synth live.)
        out = proj / "daw_oracle.wav"
        env = {**os.environ, "MIXAGENT_ORACLE_OUT": str(out),
               "MIXAGENT_ORACLE_SR": str(ref_sr), **extra_env}
        res = subprocess.run([str(daw_bin), "--export-oracle", str(proj)],
                             env=env, capture_output=True, text=True)
        if res.returncode != 0 or not out.exists():
            print(res.stdout, file=sys.stderr)
            print(res.stderr, file=sys.stderr)
            print(f"[oracle] FAIL — DAW export did not produce {out}", file=sys.stderr)
            return 1
        daw_mono, daw_side, daw_sr = _read_stereo(out)
        ref_is_stereo = bool(np.asarray(ref_full).ndim == 2 and np.asarray(ref_full).shape[1] >= 2)
        from mix_engine import analysis
        daw_full, _ = sf.read(str(out), always_2d=True)
        ok, lines = compare(ref_mono, ref_sr, daw_mono, daw_sr, profile, ref_side, daw_side,
                            ref_stereo=ref_is_stereo,
                            ref_tp_dbtp=float(analysis._true_peak_dbtp(_as_frames(ref_full))),
                            daw_tp_dbtp=float(analysis._true_peak_dbtp(_as_frames(daw_full))))
        for line in lines:
            print(f"[{profile}] {line}")
        return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="DAW export oracle (Milestone D)")
    ap.add_argument("project", nargs="?", default=str(_DEFAULT_PROJECT))
    ap.add_argument("--daw-bin", default=str(_DEFAULT_DAW_BIN))
    ap.add_argument("--profile", choices=sorted(PROFILES) + ["all"], default="all",
                    help="linear = the path gate; nonlinear = the approximation tier end to end "
                         "(derived tolerances); instrument = the MIDI half (a synthesized sampler "
                         "track, dry render vs live); stereo = the SIDE axis on generated stereo "
                         "material (the mono downmix cannot see a width device at all); "
                         "all = every profile (default)")
    args = ap.parse_args()
    profiles = sorted(PROFILES) if args.profile == "all" else [args.profile]
    worst = 0
    for prof in profiles:
        rc = run_oracle(Path(args.project), Path(args.daw_bin), prof)
        if rc == 2:
            return 2
        worst = max(worst, rc)
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
