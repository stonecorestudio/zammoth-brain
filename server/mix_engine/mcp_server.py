"""MCP server entry point. Thin layer: register tools, delegate, return models.

Tool surface contract: docs/03-mcp-tools.md (+ models.py as executable twin).
Run over stdio in dev: `python -m mix_engine.mcp_server <project_dir>`.

Full Phase 2 surface: analyze_session, analyze_stem, analyze_masking,
get_proposals, propose_actions, apply_proposal, undo_proposal,
render_and_analyze, advance_pass.
"""
from __future__ import annotations

import argparse
import functools
import inspect
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
from anyio import to_thread
from mcp.server.fastmcp import FastMCP
from scipy.io import wavfile

from . import analysis, dsp, impact, phase
from .guardrails import LIMITS
from .session import effective_return_chain
from .models import (SampleProfile, ANALYSIS_VERSION, ActivityAnalysis, ApplyResult, ChordAnalysis,
                     NoteAnalysis,
                     ExportedFile, ExportRequest, ExportResult, MaskingAnalysis, PhaseAnalysis,
                     MasterDeviceState, MasterReport, PhraseAnalysis,
                     ProbeRequest, ProbeResult,
                     ProposalRequest, ProposalResult, ProposalSummary,
                     ReferenceAnalysis, ReferenceSetAnalysis,
                     ReferenceSetMember, ReferenceSpec, RenderResult,
                     SectionAnalysis, SeparationReport, SessionAnalysis,
                     SessionState, StemAnalysis, StemCaptureState, StemUserState,
                     TempoAnalysis, TimingAnalysis,
                     RegionDescription)
from .models import _check_time_range
from .session import BOUNCE_DIRNAME, Session


def create_server(project_dir: Path | str) -> FastMCP:
    session = Session(Path(project_dir))
    mcp = FastMCP("mix-engine")

    # Every tool first re-reads session.json if another Session wrote it (2026-09-10 agent
    # audit P0-1). One engine lives for a whole chat turn, mix run, Auto Mix or gateway
    # session, while routing ops, the co-edit mirror, stem registration, undo and nested
    # runs each write through their own fresh Session, so without this the next
    # propose/apply here persisted a stale copy over all of them. Wrapping `mcp.tool`
    # itself (not each tool) means a tool added later cannot forget to sync.
    register_tool = mcp.tool

    def synced_tool(*dargs, **dkw):
        register = register_tool(*dargs, **dkw)

        def decorate(fn):
            if inspect.iscoroutinefunction(fn):
                @functools.wraps(fn)
                async def synced(*args, **kwargs):
                    session.sync_from_disk()
                    return await fn(*args, **kwargs)
            else:
                @functools.wraps(fn)
                def synced(*args, **kwargs):
                    session.sync_from_disk()
                    return fn(*args, **kwargs)
            return register(synced)
        return decorate

    mcp.tool = synced_tool

    # Rendering tools are async shells over sync bodies run via
    # to_thread.run_sync: the gateway shares this event loop, and a sync
    # tool would hold the websocket and every HTTP endpoint hostage for the
    # length of a render. Concurrency contract (EngineClient enforces the
    # lanes): sync tools — all mutations — run on the event loop itself and
    # are serialized by the gateway's main engine lock; threaded tools that
    # outlive a lock (render_and_analyze, analyze_stem, analyze_sections)
    # take a frozen state copy *on the loop* first, so the live Session is
    # never read from a thread while a mutation runs.

    @mcp.tool()
    async def analyze_session(pre_master: bool = False) -> SessionAnalysis:
        """Master-bus metrics for the current session state (in-memory render;
        no files written). `pre_master=True` (Phase 25) measures the mix as it
        enters mastering — master chain bypassed — for mix-scope reference
        matching."""
        return await to_thread.run_sync(_analyze_session, pre_master)

    def _analyze_session(pre_master: bool = False) -> SessionAnalysis:
        telemetry: list = []  # Phase 20: master-chain gain reduction
        audio, sample_rate = dsp.render_master(session, telemetry=telemetry,
                                               pre_master=pre_master)
        # Kept with the render it was measured on (Phase 54): "measure first, even when you
        # remember the numbers" is the prompt's rule, and the second ask used to cost a whole
        # render and analysis again.
        result = dsp.kept_with_master(
            session, "analyze_master",
            lambda: analysis.analyze_master(audio, sample_rate,
                                            stems=sorted(session.stems),
                                            pass_number=session.pass_number,
                                            master_dynamics=telemetry or None),
            pre_master=pre_master)
        result.pass_number = session.pass_number   # a counter, and not part of what was kept
        result.pre_master = pre_master
        return result

    @mcp.tool()
    async def analyze_stem(stem_id: str) -> StemAnalysis:
        """Per-stem metrics, measured after the stem's applied chain."""
        # Frozen copy (taken on the loop, atomic wrt mutations): read-only
        # over its own state, so the gateway runs this lock-free — snapshot
        # refreshes never make a fader or device op wait.
        return await to_thread.run_sync(_analyze_stem, session.probe_copy(),
                                        stem_id)

    def _analyze_stem(session: Session, stem_id: str) -> StemAnalysis:
        chain = session.applied_chain(stem_id)  # raises KeyError for unknown stems
        telemetry: list = []  # Phase 20: this chain's gain reduction
        audio, sample_rate = dsp.render_stem(session, stem_id,
                                             telemetry=telemetry)
        return analysis.analyze_stem(audio, sample_rate, stem_id=stem_id,
                                     applied_chain=chain,
                                     dynamics=telemetry or None)

    def _in_the_mix(session: Session) -> list[str]:
        """The stems the master is made of: not muted, not excluded - `dsp.render_master`'s
        own rule. What a read asks "the other stems" or "each stem of the mix" is these."""
        return sorted(sid for sid, state in session.stems.items()
                      if not state["muted"] and not state["excluded"])

    def _song_tempo(session: Session):
        """The song's tempo, measured on the master and KEPT with that render (Phase 55):
        `analyze_chords`, `analyze_timing` and `analyze_activity` each ran the estimator over
        the whole master on every call - most of a chord read's 2.7 s."""
        master, rate = dsp.render_master(session)
        return dsp.kept_with_master(session, "song_tempo",
                                    lambda: analysis.song_tempo(master, rate))

    def _song_key(session: Session):
        master, rate = dsp.render_master(session)
        return dsp.kept_with_master(session, "song_key",
                                    lambda: analysis.song_key(master, rate))

    @mcp.tool()
    async def describe_region(stem_id: str, start_s: float, end_s: float,
                              pre_master: bool = False) -> RegionDescription:
        """A compact perceptual read of ONE stem - or of the MIX, `stem_id: "master"` - over
        [start_s, end_s): the window's integrated loudness (`lufs_i`), its loudest short-term
        window + envelope, peak and true peak, dynamics, spectrum + how it moves, transients,
        the harshness/mud/clarity composites, and for a stem which other stems mask it (or it
        masks) in that window. The way to look at 'this part' (a selection, a marker span, a
        bar range, one section of the song) instead of the whole file. `master` is what leaves
        the master chain; `pre_master: true` (master only) is the mix as it enters mastering,
        analyze_session's measurement point. Read-only."""
        _check_time_range((start_s, end_s))
        if pre_master and stem_id != "master":
            raise ValueError("pre_master is the master's alone: a stem has no master chain "
                             f"in front of it (stem_id was {stem_id!r})")
        return await to_thread.run_sync(_describe_region, session.probe_copy(),
                                        stem_id, start_s, end_s, pre_master)

    def _describe_region(session: Session, stem_id: str, start_s: float,
                         end_s: float, pre_master: bool = False) -> RegionDescription:
        cache: dict = {}
        if stem_id == "master":
            # A BAR RANGE OF THE MIX (2026-10-11, the mashup showcase's mastering turn): the
            # agent was asked for the master section by section and this refused "master", so
            # it added up each stem's loudness instead - about 1 dB off, and two sections in
            # the wrong order. No stem can be called master (`reserved_stem_name`). The mix
            # masks nothing and is masked by nothing: it is everything.
            audio, sample_rate = dsp.render_master(session, _cache=cache,
                                                   pre_master=pre_master)
            return analysis.describe_region(audio, sample_rate, stem_id="master",
                                            start_s=start_s, end_s=end_s, others={})
        session.applied_chain(stem_id)  # raises KeyError for unknown stems
        audio, sample_rate = dsp.render_stem(session, stem_id, _cache=cache)
        # Who masks it is asked of the stems that SOUND (Phase 55): every stem was rendered,
        # muted and deleted ones included - in a session of eight prints beside ten muted
        # originals, each call rendered (and resampled) the ten for nothing.
        others = {sid: dsp.render_stem(session, sid, _cache=cache)[0]
                  for sid in _in_the_mix(session) if sid != stem_id}
        return analysis.describe_region(audio, sample_rate, stem_id=stem_id,
                                        start_s=start_s, end_s=end_s,
                                        others=others)

    @mcp.tool()
    async def analyze_phrases(stem_id: str) -> PhraseAnalysis:
        """Phase 35: local level outliers and sibilant bursts for existing
        ranged gain/de-ess proposals. Read-only; no audio or mutation."""
        return await to_thread.run_sync(_analyze_phrases, session.probe_copy(),
                                        stem_id)

    def _analyze_phrases(session: Session, stem_id: str) -> PhraseAnalysis:
        session.applied_chain(stem_id)  # raises KeyError for unknown stems
        audio, sample_rate = dsp.render_stem(session, stem_id)
        return analysis.analyze_phrases(audio, sample_rate, stem_id=stem_id)

    @mcp.tool()
    async def analyze_notes(stem_id: str) -> NoteAnalysis:
        """Per-note intonation for a MONOPHONIC stem (2026-08-08): every held
        note with its cents error, worst deviation first. Lead vocal, bass, or
        a single-line lead only — on polyphonic material pitch tracking locks
        onto one voice and reports a `reason` instead. Read-only."""
        return await to_thread.run_sync(_analyze_notes, session.probe_copy(),
                                        stem_id)

    def _analyze_notes(session: Session, stem_id: str) -> NoteAnalysis:
        session.applied_chain(stem_id)  # raises KeyError for unknown stems
        audio, sample_rate = dsp.render_stem(session, stem_id)
        return analysis.analyze_notes(audio, sample_rate, stem_id=stem_id)

    @mcp.tool()
    async def analyze_chords(stem_id: str) -> ChordAnalysis:
        """Chords DETECTED from one stem's audio (2026-08-08): major/minor
        triads and sevenths per beat, merged into spans, scored against the
        song's detected key. Run it on a HARMONIC stem (piano, guitar, pad) —
        a full mix or a drum stem smears the chroma. Distinct from the DAW's
        authored chord_progression; never present one as the other."""
        return await to_thread.run_sync(_analyze_chords, session.probe_copy(),
                                        stem_id)

    def _analyze_chords(session: Session, stem_id: str) -> ChordAnalysis:
        session.applied_chain(stem_id)  # raises KeyError for unknown stems
        tempo, key = _song_tempo(session), _song_key(session)
        audio, sample_rate = dsp.render_stem(session, stem_id)
        return analysis.analyze_chords(
            audio, sample_rate, stem_id=stem_id,
            bpm=tempo[0] if tempo else None,
            beat_offset_s=tempo[2] if tempo else 0.0, key=key)

    @mcp.tool()
    async def analyze_timing(stem_id: str) -> TimingAnalysis:
        """Onset timing against the SONG's grid (2026-08-08): is this part
        rushing or dragging, how tight is it, and is it swung. The grid comes
        from the master's tempo estimate, not the stem's — the question is
        whether the part sits with the record. Read-only."""
        return await to_thread.run_sync(_analyze_timing, session.probe_copy(),
                                        stem_id)

    def _analyze_timing(session: Session, stem_id: str) -> TimingAnalysis:
        session.applied_chain(stem_id)  # raises KeyError for unknown stems
        tempo = _song_tempo(session)
        audio, sample_rate = dsp.render_stem(session, stem_id)
        return analysis.analyze_timing(
            audio, sample_rate, stem_id=stem_id,
            bpm=tempo[0] if tempo else None,
            beat_offset_s=tempo[2] if tempo else 0.0)

    @mcp.tool()
    async def analyze_tempo(stem_id: str) -> TempoAnalysis:
        """ONE stem's own tempo and where its beat falls on the timeline (2026-10-11): the
        song's estimator run on that stem alone, as it plays. Every other grid here is the
        master's - with two songs in a project that is neither song's. Nulls with a `reason`
        when the part has no steady pulse. Which beat is bar 1 is not said. Read-only."""
        return await to_thread.run_sync(_analyze_tempo, session.probe_copy(), stem_id)

    def _analyze_tempo(session: Session, stem_id: str) -> TempoAnalysis:
        session.applied_chain(stem_id)  # raises KeyError for unknown stems
        audio, sample_rate = dsp.render_stem(session, stem_id)
        return analysis.stem_tempo(audio, sample_rate, stem_id)

    @mcp.tool()
    async def analyze_activity(stem_ids: Optional[list[str]] = None,
                               bpm: Optional[float] = None, bar1_s: float = 0.0,
                               beats_per_bar: int = 4) -> ActivityAnalysis:
        """Where each stem comes in and drops out, to the BAR (2026-10-11): per stem, runs
        of whole bars that are `in`, `low` (a fill under a break, a tail) or `out`. Give the
        grid - `bpm` and `bar1_s`, where bar 1 starts on the timeline - or the master's
        detected tempo is used with bar 1 at its first beat (the downbeat is then a guess).
        `stem_ids` omitted = every active stem. Use it before choosing which bars of a song
        are its verse, chorus or drop: a section starts where a part enters, and that is not
        always a multiple of eight bars from bar 1. Read-only."""
        return await to_thread.run_sync(_analyze_activity, session.probe_copy(), stem_ids,
                                        bpm, bar1_s, beats_per_bar)

    def _analyze_activity(session: Session, stem_ids: Optional[list[str]],
                          bpm: Optional[float], bar1_s: float,
                          beats_per_bar: int) -> ActivityAnalysis:
        names = list(stem_ids) if stem_ids else [
            name for name, stem in sorted(session.stems.items()) if not stem.get("excluded")]
        for name in names:
            session.applied_chain(name)  # raises KeyError for unknown stems
        if not bpm:
            tempo = _song_tempo(session)
            if tempo is None:
                return analysis.analyze_activity({}, dsp.render_master(session)[1], bpm=None)
            bpm, bar1_s = tempo[0], tempo[2]
        cache: dict = {}
        rendered = {name: dsp.render_stem(session, name, _cache=cache) for name in names}
        rate = next(iter(rendered.values()))[1] if rendered else 48000
        return analysis.analyze_activity({name: audio for name, (audio, _) in rendered.items()},
                                         rate, bpm=bpm, bar1_s=bar1_s,
                                         beats_per_bar=beats_per_bar)

    @mcp.tool()
    async def analyze_bus(bus_id: str) -> StemAnalysis:
        """Per-bus metrics (Phase 23): the group's active members summed
        through the bus chain — StemAnalysis-shaped, stem_id = the bus id."""
        return await to_thread.run_sync(_analyze_bus, bus_id)

    def _analyze_bus(bus_id: str) -> StemAnalysis:
        chain = session.applied_bus_chain(bus_id)  # KeyError for unknown busses
        telemetry: list = []
        audio, sample_rate = dsp.render_bus(session, bus_id,
                                            telemetry=telemetry)
        return analysis.analyze_stem(audio, sample_rate, stem_id=bus_id,
                                     applied_chain=chain,
                                     dynamics=telemetry or None)

    @mcp.tool()
    async def analyze_return(return_id: str) -> StemAnalysis:
        """Per-return metrics (Phase 38 send/return): every active source's send
        into the return, summed through its wet chain — StemAnalysis-shaped,
        stem_id = the return id. The agent's ears on its own reverb/delay returns
        (so it can judge + balance the wet against the dry mix)."""
        return await to_thread.run_sync(_analyze_return, return_id)

    def _analyze_return(return_id: str) -> StemAnalysis:
        if return_id not in session.returns:
            raise KeyError(return_id)
        chain = effective_return_chain(session.returns[return_id])
        audio, sample_rate = dsp.render_return(session, return_id)
        return analysis.analyze_stem(audio, sample_rate, stem_id=return_id,
                                     applied_chain=chain)

    @mcp.tool()
    async def analyze_masking(stem_a: str, stem_b: str) -> MaskingAnalysis:
        """Per-band excess of stem_b's energy over stem_a's where stem_a is active.
        Positive = stem_a is being masked in that band."""
        return await to_thread.run_sync(_analyze_masking, stem_a, stem_b)

    def _analyze_masking(stem_a: str, stem_b: str) -> MaskingAnalysis:
        cache: dict = {}  # shared duck keys render once across the pair
        audio_a, sample_rate = dsp.render_stem(session, stem_a, _cache=cache)
        audio_b, _ = dsp.render_stem(session, stem_b, _cache=cache)
        return analysis.analyze_masking(audio_a, audio_b, sample_rate,
                                        stem_a=stem_a, stem_b=stem_b)

    @mcp.tool()
    async def analyze_phase(stem_ids: Optional[list[str]] = None) -> PhaseAnalysis:
        """The mix's stereo PHASE in one read (2026-10-13): what cancels when the song is
        folded to mono, on which channel, why, and the move that fixes it. Per channel (every
        stem in the mix, every group, the master): each band's L/R correlation (+1 mono, 0 two
        unrelated channels, -1 anti-phase), the share of its ~0.3 s windows below 0 - what the
        DAW's PHASE lamps light on - and how much it loses folded to mono. `issues`: the bands
        that cancel (below 0 over the song, or for 20 % of its time or more) and a channel with one
        side upside down, each with its cause (the file, the channel's own chain, the master's
        devices, the mix) and a fix. `conflicts`: pairs of stems that cancel EACH OTHER in
        sub / bass / low_mid (opposite polarity, or one a few ms early). `clean` = nothing
        cancels. `stem_ids` limits the channels read to the stems and groups it names; the
        master is always read. Read-only."""
        return await to_thread.run_sync(_analyze_phase, session.probe_copy(), stem_ids)

    def _analyze_phase(session: Session, stem_ids: Optional[list[str]]) -> PhaseAnalysis:
        return phase.phase_reading(session, stem_ids)

    @mcp.tool()
    async def analyze_sections(pre_master: bool = False) -> SectionAnalysis:
        """Deterministic section detection on the current master render: labeled
        time ranges with per-section loudness/band energy and per-stem activity,
        plus the tempo estimate (tempo_bpm / tempo_confidence / beat_offset_s)
        and 4/4 bar annotations when the grid is confident (Phase 20).
        Use this before proposing time-ranged moves (gain rides). `pre_master`
        matches analyze_session's Phase 25 reference-scope measurement point."""
        # Frozen copy, like analyze_stem: renders every stem, so it must
        # never hold user mutations hostage.
        return await to_thread.run_sync(_analyze_sections,
                                        session.probe_copy(), pre_master)

    def _analyze_sections(session: Session,
                          pre_master: bool = False) -> SectionAnalysis:
        # One shared memo: the master render and the per-stem activity
        # signals are the same renders — without it every stem rendered twice.
        cache: dict = {}
        audio, sample_rate = dsp.render_master(
            session, _cache=cache, pre_master=pre_master)
        # ...of the stems in the mix: a muted or deleted one is not in this master (Phase 55).
        stem_audio = {stem_id: dsp.render_stem(session, stem_id, _cache=cache)[0]
                      for stem_id in _in_the_mix(session)}
        return analysis.analyze_sections(audio, sample_rate, stem_audio)

    @mcp.tool()
    async def analyze_separation() -> SeparationReport:
        """Phase 29 separation 'double check': per-stem artifact metrics
        (musical-noise shimmer, transient smear, spectral holes, inter-stem
        leakage) + the loudness spread across the current stems — what the
        AI-cleanup preset reads to decide repair. Read-only; measured on each
        stem's rendered output, so re-running after a repair reflects it.
        Only the stems the app itself split are measured; the others are listed
        under `not_separated` (on ordinary material these metrics read
        sibilance, sustain and playing in time as artifacts)."""
        return await to_thread.run_sync(_analyze_separation, session.probe_copy())

    def _analyze_separation(session: Session) -> SeparationReport:
        cache: dict = {}
        stem_audio: dict = {}
        sample_rate = session.project_sample_rate()
        split = session.separated_stems()
        for stem_id in split:
            audio, sample_rate = dsp.render_stem(session, stem_id, _cache=cache)
            stem_audio[stem_id] = audio
        report = analysis.analyze_separation(stem_audio, sample_rate)
        report.not_separated = sorted(
            stem_id for stem_id, stem in session.stems.items()
            if not stem.get("excluded") and stem_id not in split)
        return report

    @mcp.tool()
    async def analyze_reference(path: str, start_s: Optional[float] = None,
                                end_s: Optional[float] = None) -> ReferenceAnalysis:
        """Profile a reference track ("sit like this record"): loudness,
        dynamics, tonal balance, per-band width - and the file's sample peak and
        true peak, so it is also how a file on disk is MEASURED (an export just
        written to bounces/). `start_s` + `end_s` profile ONE STRETCH of the file
        (one section of an exported song; `window_s` says which) - the file holds
        what the DAW played, automation included, where a reading of `master` is
        the offline engine's. Cached next to the session (a stretch is not); the
        reference audio never enters the mix path."""
        if (start_s is None) != (end_s is None):
            raise ValueError("a stretch of the file needs both start_s and end_s "
                             "(leave both out for the whole file)")
        if start_s is not None:
            _check_time_range((start_s, end_s))
            return await to_thread.run_sync(_analyze_reference_window, path,
                                            float(start_s), float(end_s))
        return await to_thread.run_sync(_analyze_reference, path)

    @mcp.tool()
    async def analyze_sample(path: str, kind: str = "one_shot", bpm: Optional[float] = None,
                             beats: Optional[float] = None,
                             sound_kind: Optional[str] = None,
                             played_note: Optional[int] = None) -> SampleProfile:
        """ONE SOUND, measured - a one-shot or a loop, at any length from a few milliseconds
        up (the mix tools need 0.4 s or more and read nothing useful of a drum hit). How fast
        it starts (`attack_ms`), how long it rings (`decay_ms`, `sounding_s`), silence in
        front of it, the level it ends at (a cut that clicks), peaks and a loudness that
        answers on a 20 ms file, where its energy is (`centroid_hz`, `band_share_pct`), its
        pitch as a note and - when the pitch MOVES, a kick's drop - where it started and how
        long it took to settle, whether it is a tone or noise, DC, width. `kind: "loop"` with
        `bpm` and `beats` also holds the file to that length and reads its seam.
        `sound_kind` (a sample-pack kind: kick, snare, hat, boom, bass, pad, pluck ...) also
        reads it against what that kind usually measures as and answers `issues` - plain
        sentences, a guide and not a verdict; `played_note` is the key it was played on, for
        the kinds that should sound at that note. Read-only; any audio file the brain can
        open."""
        if kind not in ("one_shot", "loop"):
            raise ValueError('kind is "one_shot" or "loop"')
        if (bpm is None) != (beats is None):
            raise ValueError("a loop's length is read against bpm AND beats: give both, or neither")
        if sound_kind is not None:
            from . import sound_classes as sc

            if sc.canonical(sound_kind) not in sc.CLASSES:
                raise ValueError(f"no sound kind '{sound_kind}' (kinds: {', '.join(sc.CLASSES)})")
        if played_note is not None and not 0 <= int(played_note) <= 127:
            raise ValueError("played_note is a MIDI note number, 0-127 (60 is middle C)")
        profile = await to_thread.run_sync(_analyze_sample, path, kind, bpm, beats)
        if sound_kind is None:
            return profile
        from . import sound_classes as sc

        read = sc.reading(profile.model_dump(), sound_kind, played_note)
        return profile.model_copy(update={"sound_kind": read["kind"], "issues": read["issues"],
                                          "issues_are": read.get("issues_are")})

    def _analyze_sample(path: str, kind: str, bpm: Optional[float],
                        beats: Optional[float]) -> SampleProfile:
        import soundfile as sf

        from . import sample_profile as sp

        sample_path = _reference_path(path)
        info = sf.info(str(sample_path))
        if info.frames / float(info.samplerate) > sp.MAX_SECONDS:
            raise ValueError(f"the file is {round(info.frames / float(info.samplerate), 1)} s long: "
                             f"analyze_sample reads one sound, up to {int(sp.MAX_SECONDS)} s "
                             "(a longer file is analyze_reference's)")
        audio, sample_rate = sf.read(sample_path, dtype="float32", always_2d=True)
        return sp.sample_profile(audio, sample_rate, kind=kind, bpm=bpm, beats=beats, path=path)

    def _analyze_reference_window(path: str, start_s: float,
                                  end_s: float) -> ReferenceAnalysis:
        """One stretch of a file, read from the file (only those frames) and never cached:
        the whole-file profile is the cache's, and a window served as it would be a lie."""
        import soundfile as sf

        ref_path = _reference_path(path)
        info = sf.info(str(ref_path))
        length_s = info.frames / float(info.samplerate)
        if start_s >= length_s:
            raise ValueError(f"the file is {round(length_s, 2)} s long: a stretch starting "
                             f"at {start_s} s is past its end")
        first = int(round(start_s * info.samplerate))
        last = min(info.frames, int(round(end_s * info.samplerate)))
        audio, sample_rate = sf.read(ref_path, start=first, stop=last,
                                     dtype="float32", always_2d=True)
        result = analysis.analyze_reference(audio, sample_rate, path=path)
        return result.model_copy(update={
            "window_s": (round(first / sample_rate, 3), round(last / sample_rate, 3))})

    def _reference_path(path: str) -> Path:
        # `~/…` is the form the agent HOLDS (full audit 2026-10-02, reader C4·R9): the brain
        # writes the home directory as `~` in everything a model reads, and promises every
        # reader of a path it hands back expands it. This one did not - `Path("~/x")` is not
        # absolute, so it was looked for as a folder called `~` inside the project.
        ref_path = Path(path).expanduser()
        if not ref_path.is_absolute():
            ref_path = session.project_dir / ref_path
        if not ref_path.is_file():
            raise FileNotFoundError(f"reference not found: {ref_path}")
        return ref_path

    def _reference_cache_entries(cache_path: Path) -> list[dict]:
        if not cache_path.exists():
            return []
        # A cache that cannot be read is no cache (reader E5·R10): the file was written in
        # place, so a crash mid-write left it cut short - and every retry of the tool raised
        # on it until someone deleted a hidden file.
        try:
            cached = json.loads(cache_path.read_text())
        except (OSError, ValueError):
            return []
        if not isinstance(cached, dict):
            return []
        if isinstance(cached.get("entries"), list):
            return [e for e in cached["entries"] if isinstance(e, dict)]
        # Back-compat with the Phase 7 single-entry cache shape.
        if cached.get("key") is not None and cached.get("result") is not None:
            return [cached]
        return []

    def _write_reference_cache(cache_path: Path, entry: dict) -> None:
        entries = [e for e in _reference_cache_entries(cache_path)
                   if e.get("key") != entry["key"]]
        entries.append(entry)
        staged = cache_path.with_name(cache_path.name + ".tmp")   # whole, or not at all
        staged.write_text(json.dumps({"entries": entries}, indent=2))
        os.replace(staged, cache_path)

    def _reference_key(ref_path: Path) -> dict:
        return {"path": str(ref_path.resolve()),
                "mtime_ns": ref_path.stat().st_mtime_ns,
                "analysis_version": ANALYSIS_VERSION}

    def _load_reference(path: str, include_sections: bool = False
                        ) -> tuple[ReferenceAnalysis, Optional[SectionAnalysis]]:
        import soundfile as sf

        ref_path = _reference_path(path)
        cache_path = session.project_dir / ".reference_cache.json"
        key = _reference_key(ref_path)
        for cached in _reference_cache_entries(cache_path):
            if cached.get("key") == key:
                try:
                    result = ReferenceAnalysis.model_validate(cached["result"])
                    sections = (SectionAnalysis.model_validate(cached["sections"])
                                if cached.get("sections") is not None else None)
                except (KeyError, ValueError):            # an entry that does not hold a result
                    break
                if include_sections and sections is None:
                    break
                # ...nor one written before the profile held the file's peaks (2026-10-11):
                # validated, it would answer "true peak: none" for a file that has one.
                if "true_peak_dbtp" not in cached["result"]:
                    break
                return result, sections

        audio, sample_rate = sf.read(ref_path, dtype="float32", always_2d=True)
        result = analysis.analyze_reference(audio, sample_rate, path=path)
        sections = (analysis.analyze_sections(audio, sample_rate, {})
                    if include_sections else None)
        entry = {"key": key, "result": result.model_dump()}
        if sections is not None:
            entry["sections"] = sections.model_dump()
        _write_reference_cache(cache_path, entry)
        return result, sections

    def _analyze_reference(path: str) -> ReferenceAnalysis:
        return _load_reference(path)[0]

    @mcp.tool()
    async def analyze_reference_set(
            references: list[ReferenceSpec],
            mix_sections: Optional[SectionAnalysis] = None
            ) -> ReferenceSetAnalysis:
        """Phase 36: weighted multi-reference profile. Roles decide aggregate
        targets (tonality / width / loudness); optional mix_sections enables
        section-aware tonal targets. Read-only — references never enter the mix
        path."""
        return await to_thread.run_sync(_analyze_reference_set,
                                        references, mix_sections)

    def _analyze_reference_set(
            references: list[ReferenceSpec],
            mix_sections: Optional[SectionAnalysis] = None
            ) -> ReferenceSetAnalysis:
        members: list[ReferenceSetMember] = []
        for spec in references:
            ref, sections = _load_reference(spec.path, include_sections=True)
            members.append(ReferenceSetMember(
                spec=spec, analysis=ref, sections=sections))
        return analysis.analyze_reference_set(members, mix_sections)

    @mcp.tool()
    def get_state() -> SessionState:
        """Current session state — stems' chains, gains, rides, tags and the
        master chain, straight from session state. Cheap: no rendering, no
        analysis. Use this between renders instead of analyze_stem when only
        the applied state is needed."""
        return session.get_state()

    @mcp.tool()
    def refresh_stems(separated: Optional[list[str]] = None) -> list[str]:
        """Re-discover stems on disk (Phase 10): picks up wav files added to
        the project folder after open. Returns the current stem ids.
        `separated` names the wav files a stem split just wrote, so the
        separation check knows which stems it applies to."""
        return session.refresh_stems(separated)

    @mcp.tool()
    def reset_stem(stem_id: str, only_if_excluded: bool = False) -> dict:
        """Restore a stem's record to a fresh registration (defaults, path kept).
        The import boundary passes only_if_excluded=True when re-registering an
        id: a soft-deleted (excluded) stem must not hand its stale chain +
        excluded flag to a new track that reuses the name."""
        return session.reset_stem(stem_id, only_if_excluded)

    @mcp.tool()
    async def auto_tag_stems(
            stem_ids: Optional[list[str]] = None) -> dict[str, list[str]]:
        """Classify untagged stems (Phase 14): deterministic — the stem's NAME
        first (the shared instrument vocabulary), then rules over the per-stem
        metrics (kick / bass / vocal / pad / cymbal / full_mix —
        contract in the mix-analysis skill). Stems that already carry tags
        are never touched: user tags win. `stem_ids` narrows the candidates
        (the gateway passes the just-imported stem so unclassifiable older
        stems aren't re-analyzed on every import); default is all stems.
        Returns stem_id -> tags for every stem. Tags drive context-dependent
        guardrails (a "bass" tag tightens the HPF limit) and the agent's
        reasoning."""
        return await to_thread.run_sync(_auto_tag_stems, stem_ids)

    def _auto_tag_stems(stem_ids: Optional[list[str]]) -> dict[str, list[str]]:
        for stem_id, stem in sorted(session.stems.items()):
            if stem_ids is not None and stem_id not in stem_ids:
                continue
            if stem["tags"]:
                continue  # user (or an earlier pass) already tagged it
            audio, sample_rate = dsp.render_stem(session, stem_id)
            stem_analysis = analysis.analyze_stem(
                audio, sample_rate, stem_id=stem_id,
                applied_chain=session.applied_chain(stem_id))
            tag = analysis.classify_stem(stem_analysis)
            if tag is not None:
                session.set_stem_tags(stem_id, [tag])
        return {stem_id: list(stem["tags"])
                for stem_id, stem in sorted(session.stems.items())}

    @mcp.tool()
    def get_proposals(status: Optional[str] = None) -> list[ProposalSummary]:
        """Proposal history, optionally filtered by status (proposed/applied/undone/rejected)."""
        return session.get_proposals(status)

    @mcp.tool()
    async def probe_actions(req: ProbeRequest) -> ProbeResult:
        """What-if sandbox (Phase 12): apply actions to a scratch copy of
        session state, measure the named metrics before/after over an
        optional [start_s, end_s] range, discard everything. No proposal
        record, no guardrails — nothing persists. Deterministic: a probe's
        `after` equals the post-apply measurement of the same actions on the
        same range. Probe candidate moves here before committing to a
        proposal."""
        return await to_thread.run_sync(_probe_actions, req)

    def _probe_actions(req: ProbeRequest) -> ProbeResult:
        violations = session.validate_action_targets(req.actions)
        if violations:
            raise ValueError("; ".join(v.message for v in violations))
        started = time.perf_counter()
        measurements = impact.probe(session, req.actions, req.metrics,
                                    range_s=req.range_s)
        return ProbeResult(
            measurements=measurements,
            render_ms=int((time.perf_counter() - started) * 1000))

    @mcp.tool()
    def propose_actions(req: ProposalRequest) -> ProposalResult:
        """Validate a proposal against guardrails. Accepted proposals get an id; rejected
        ones return machine-readable violations to revise against."""
        return session.propose(req)

    @mcp.tool()
    def set_stem(stem_id: str,
                 gain_db: Optional[float] = None,
                 pan: Optional[float] = None,
                 width: Optional[float] = None,
                 muted: Optional[bool] = None,
                 excluded: Optional[bool] = None,
                 automation_enabled: Optional[bool] = None) -> StemUserState:
        """USER channel (docs/03, Phase 9): direct absolute setters for the
        human's own moves — fader, pan, mute, "delete", and the Phase 27
        automation switch (this stem's section moves; on "master" it is the
        session-global switch, Ableton-style). Immediate, no proposals, no
        guardrails. The planning agent must never call this. `stem_id` may
        also name a group bus (Phase 23) — fader, pan, width and mute, the
        group strip's own scalars (2026-09-05); `excluded` is stem-only (a bus
        is dissolved with remove_bus) and automation is per-stem."""
        changes = {k: v for k, v in dict(
            gain_db=gain_db, pan=pan, width=width, muted=muted,
            excluded=excluded, automation_enabled=automation_enabled).items()
                   if v is not None}
        if not changes:
            raise ValueError("set_stem called with nothing to change")
        return StemUserState(**session.user_edit(stem_id, changes))

    @mcp.tool()
    def set_stem_capture_start(stem_id: str, capture_start_s: float) -> StemCaptureState:
        """CAPTURE bookkeeping (Phase 41, gateway-only): record a stem's host-
        timeline offset (seconds) so a sidechain duck reads its key stem's envelope
        at the right place — even when the key and the ducked track were captured in
        separate play-throughs. Not an audio mutation and not a user/agent move; the
        gateway calls it at import with the offset the plugin reported. The planning
        agent has no reason to call it."""
        return StemCaptureState(
            stem_id=stem_id,
            capture_start_s=session.set_stem_capture_start(stem_id, capture_start_s))

    @mcp.tool()
    def set_master_param(device_type: str, occurrence: int = 0,
                         value: float = 0.0, target: str = "master") -> MasterDeviceState:
        """USER co-edit channel (Phase 40, native VST only): set a chain device's
        primary control — eq gain, dyn-eq depth, comp ratio, limiter ceiling, or
        gain — directly. Immediate, no proposals, no guardrails, clamped. `target`
        is master (default) or a stem/bus id (a per-track insert edits its own
        stem). The planning agent must NEVER call this (a user channel, like
        set_stem)."""
        return MasterDeviceState(
            **session.set_master_param(device_type, occurrence, value, target))

    @mcp.tool()
    def add_master_device(device_type: str) -> MasterDeviceState:
        """USER co-edit channel (Phase 40, native VST only): append a device
        (eq / comp / limiter / dynamic_eq) to the master chain with neutral
        defaults. Immediate, no proposals, no guardrails. The planning agent must
        NEVER call this (a user channel, like set_stem)."""
        d = session.add_master_device(device_type)
        return MasterDeviceState(type=d["type"], occurrence=d["occurrence"])

    @mcp.tool()
    def remove_master_device(device_type: str, occurrence: int = 0) -> MasterDeviceState:
        """USER co-edit channel (Phase 40, native VST only): remove the Nth
        `device_type` from the master chain. Immediate, no proposals. The planning
        agent must NEVER call this (a user channel)."""
        d = session.remove_master_device(device_type, occurrence)
        return MasterDeviceState(type=d["type"], occurrence=d["occurrence"])

    @mcp.tool()
    def set_master_device_order(order: list[str]) -> dict:
        """USER co-edit channel (Phase 42, native VST only): set the master device
        render order (a list of device-type modules left-to-right) from the native
        drag-to-reorder. The limiter is always rendered last (true-peak ceiling).
        Immediate, no proposals. The planning agent must NEVER call this."""
        return session.set_master_device_order(order)

    @mcp.tool()
    def add_stem_device(target: str, device_type: str) -> MasterDeviceState:
        """USER co-edit channel (Phase 40, per-track insert): append a stem-only
        COLOR device (transient / deess / band_width — the processors the master
        can't host) to stem `target`'s chain with neutral defaults. Immediate, no
        proposals. The planning agent must NEVER call this (a user channel)."""
        d = session.add_stem_device(target, device_type)
        return MasterDeviceState(type=d["type"], occurrence=d["occurrence"])

    @mcp.tool()
    def remove_stem_device(target: str, device_type: str,
                           occurrence: int = 0) -> MasterDeviceState:
        """USER co-edit channel (Phase 40, per-track insert): remove the Nth
        `device_type` from stem `target`'s chain. Immediate, no proposals. The
        planning agent must NEVER call this (a user channel)."""
        d = session.remove_stem_device(target, device_type, occurrence)
        return MasterDeviceState(type=d["type"], occurrence=d["occurrence"])

    @mcp.tool()
    def set_master_device_param(device_type: str, occurrence: int = 0,
                                param: str = "", value: float = 0.0,
                                target: str = "master") -> MasterDeviceState:
        """USER co-edit channel (Phase 40, native VST only): set any editable param
        on a device (the device detail popup) — eq/dyn-eq freq/q/threshold, comp
        times/makeup/mix, limiter ceiling. Immediate, clamped, no proposals.
        `target` = master or a stem/bus id (per-track insert). The planning agent
        must NEVER call this (a user channel)."""
        d = session.set_master_device_param(device_type, occurrence, param, value, target)
        return MasterDeviceState(type=d["type"], occurrence=d["occurrence"],
                                 value=d["value"])

    @mcp.tool()
    def set_eq_shape(occurrence: int = 0, shape: str = "peak",
                     target: str = "master") -> MasterDeviceState:
        """USER co-edit channel (Phase 40, native VST only): change an EQ band's
        filter shape (peak / low_shelf / high_shelf / hpf / lpf / notch / bandpass /
        tilt_shelf) from the detail popup. `target` = master or a stem/bus id.
        Agent must NEVER call this."""
        session.set_eq_shape(occurrence, shape, target)
        return MasterDeviceState(type="eq", occurrence=occurrence)

    @mcp.tool()
    def set_device_channel(device_type: str = "eq", occurrence: int = 0,
                           channel: str = "stereo", target: str = "master") -> MasterDeviceState:
        """USER co-edit channel (Phase 41, native VST only): set an eq/comp device's
        channel mode (stereo / mid / side / left / right) from the device UI.
        `target` = master or a stem/bus id. The planning agent sets `channel` in its
        proposal instead — it must NEVER call this user channel."""
        session.set_device_channel(device_type, occurrence, channel, target)
        return MasterDeviceState(type=device_type, occurrence=occurrence)

    @mcp.tool()
    def set_device_band(occurrence: int = 0, band: str = "presence",
                        lo_hz: float | None = None, hi_hz: float | None = None,
                        target: str = "master") -> MasterDeviceState:
        """USER co-edit channel (2026-07-05): pick a band_width device's band LABEL
        (sub / bass / low_mid / mid / high_mid / presence / brilliance / air) from
        the device UI — or (2026-07-19) explicit lo_hz/hi_hz to place the band
        freely (they override the label). `target` = master or a stem/bus id. The
        planning agent sets `band` in its proposal instead — it must NEVER call
        this user channel."""
        session.set_device_band(occurrence, band, lo_hz=lo_hz, hi_hz=hi_hz,
                                target=target)
        return MasterDeviceState(type="band_width", occurrence=occurrence)

    @mcp.tool()
    def set_sat_multiband(occurrence: int = 0, on: bool = True,
                          target: str = "master") -> MasterDeviceState:
        """USER co-edit (Phase 45, native VST only): toggle a saturation device between
        single-band and the 3-band (Saturn) form — the native MB button. Seeds/clears
        low/mid/high. `target` = master or a stem/bus id. The agent sets the bands in
        its proposal instead — it must NEVER call this user channel."""
        session.set_sat_multiband(occurrence, on, target)
        return MasterDeviceState(type="saturation", occurrence=occurrence)

    @mcp.tool()
    def set_master_device_enabled(device_type: str, occurrence: int = 0,
                                  enabled: bool = True,
                                  target: str = "master") -> MasterDeviceState:
        """USER co-edit channel (Phase 40, native VST only): bypass / re-enable a
        device (the tile on/off). Render + live monitor both skip a disabled device.
        `target` = master or a stem/bus id. Agent must NEVER call this."""
        session.set_master_device_enabled(device_type, occurrence, enabled, target)
        return MasterDeviceState(type=device_type, occurrence=occurrence)

    # Phase 28: busses are no longer a user channel. Routing is the agent's —
    # it creates, routes and glues busses through proposals (the group/route
    # actions), so there is no create_bus/remove_bus/route_stem tool. The
    # user's manual moves are volume/pan/mute/solo; analyze_bus stays as ears.

    @mcp.tool()
    def apply_proposal(proposal_id: str) -> ApplyResult:
        """Atomically apply an accepted proposal. The only way audio state changes.
        Pre-apply values of the predicted metrics are captured so measured impact
        can be reported after the next render."""
        return impact.apply_with_baseline(session, proposal_id)

    @mcp.tool()
    def undo_proposal(proposal_id: str) -> ApplyResult:
        """Revert an applied proposal as one unit."""
        return session.undo(proposal_id)

    @mcp.tool()
    async def render_and_analyze(range_s: Optional[tuple[float, float]] = None) -> RenderResult:
        """Bounce the master (optionally a [start, end] time range), write it under
        bounces/, and return fresh master-bus metrics."""
        # Freeze state here on the event loop — sync (mutation) tools also
        # run on the loop, so the copy is atomic — and render entirely from
        # the copy. The gateway gives renders their own lock lane, so user
        # mutations land in milliseconds mid-render instead of queueing
        # behind it; the frozen copy is what makes that safe.
        frozen = session.probe_copy()
        result, measured = await to_thread.run_sync(_render_and_analyze,
                                                    frozen, range_s)
        # Measured impacts were computed against the frozen state; record
        # them on the live session back on the loop (atomic again).
        for proposal_id, measurements in measured:
            session.record_measured(proposal_id, measurements)
        return result

    def _render_and_analyze(session: Session,
                            range_s: Optional[tuple[float, float]],
                            ) -> tuple[RenderResult, list]:
        started = time.perf_counter()
        report: dict = {}
        telemetry: list = []  # Phase 20: master-chain gain reduction
        stem_telemetry: dict = {}  # 2026-06 review: per-stem GR, same render
        audio, sample_rate = dsp.render_master(session, range_s=range_s,
                                               report=report,
                                               telemetry=telemetry,
                                               stem_telemetry=stem_telemetry)
        bounce_dir = session.project_dir / BOUNCE_DIRNAME
        bounce_dir.mkdir(exist_ok=True)
        bounce_path = bounce_dir / "master.wav"
        if range_s is None and bounce_path.exists():
            # Phase 16: keep the previous full bounce for hold-to-compare
            # A/B in the transport (and probe auditioning). Hardlink, not
            # copyfile — the bounce runs ~25 MB/min and this ran every
            # render. master.wav stays present throughout (the gateway may
            # be streaming it); the replace below only detaches the name,
            # and bounces are written tmp+replace, never in place, so the
            # shared inode is immutable.
            import os
            prev_path = bounce_dir / "master.prev.wav"
            prev_tmp = bounce_dir / "master.prev.wav.tmp"
            prev_tmp.unlink(missing_ok=True)
            try:
                os.link(bounce_path, prev_tmp)
                prev_tmp.replace(prev_path)
            except OSError:  # filesystem without hardlinks
                import shutil
                shutil.copyfile(bounce_path, prev_path)
        # scipy, not soundfile: libsndfile stamps float WAVs with a PEAK chunk
        # containing wall-clock time, which breaks bit-identical renders.
        # tmp + replace: the gateway loop stays live during renders now, so
        # /api/audio/bounce must never stream a half-written file.
        bounce_tmp = bounce_path.with_suffix(".wav.tmp")
        wavfile.write(bounce_tmp, sample_rate, audio.astype(np.float32))
        bounce_tmp.replace(bounce_path)
        result = analysis.analyze_master(audio, sample_rate,
                                         stems=sorted(session.stems),
                                         pass_number=session.pass_number,
                                         master_dynamics=telemetry or None)
        # A range render's analysis is not the full master — only seed the
        # measured-impact resolver with whole-file analyses. Measure here
        # (frozen state, render thread); the async shell records on the
        # live session.
        measured = impact.measure_impacts(
            session, master_analysis=result if range_s is None else None)
        flags = []
        if (result.true_peak_dbtp is not None
                and result.true_peak_dbtp > LIMITS["master_true_peak_max_dbtp"]):
            flags.append("master_true_peak_exceeded")
        # Phase 14 QC: a band that cancels on mono fold-down. Since 2026-10-13 on the rule
        # `analyze_phase` judges the master by (phase.master_at_risk: below 0 over the song, or
        # for 20 % of its ~0.3 s windows; one side upside down) - the old whole-song -0.3 let a
        # master the DAW's PHASE lamps flashed on pass. A full render only: a range is not the song.
        if range_s is None and phase.master_at_risk(audio, sample_rate):
            flags.append("mono_compatibility_risk")
        # 2026-06 review: a comp (stem or bus) squashing its source is a
        # mistake the loudness/band metrics alone won't surface — flag it
        # from the render's own GR telemetry, like the true-peak rule.
        if analysis.over_compressed(telemetry, stem_telemetry):
            flags.append("over_compression")
        # Phase 34: limiter-specific QC. The limiter is exempt from
        # over_compression because loud masters can legitimately hit it; this
        # flag only fires when limiter GR and low PSR/PLR agree.
        if analysis.over_limited(telemetry, result.psr_db, result.plr_db):
            flags.append("over_limiting")
        # Phase 31: the song sits off concert pitch, or its intonation is
        # smeared — surfaced like the other post-render flags so the agent and
        # UI can speak up. Descriptive: the agent corrects a stem, not the mix.
        if (result.tuning_offset_cents is not None
                and (abs(result.tuning_offset_cents) >= analysis.TUNING_FLAG_CENTS
                     or (result.tuning_coherence is not None
                         and result.tuning_coherence
                         < analysis.TUNING_COHERENCE_FLOOR))):
            flags.append("out_of_tune")
        # A stem's FILE held NaN / inf samples: they were dropped, as the DAW's node drops
        # them, and the render says which stem rather than measure it silently (E5·R8).
        mended = report.pop("non_finite_stems", [])
        if mended:
            flags.append("non_finite_input")
        return RenderResult(
            bounce_path=str(bounce_path.relative_to(session.project_dir)),
            render_ms=int((time.perf_counter() - started) * 1000),
            analysis=result,
            master=MasterReport(**report) if report else None,
            flags=flags, non_finite_stems=mended), measured

    @mcp.tool()
    async def export_mix(req: ExportRequest) -> ExportResult:
        """USER channel (Phase 24; flac/mp3 in Phase 26): print
        deliverables — the master (and, with include_stems, each active
        stem through its applied chain) in a folder outside the project.
        wav24/wav16 integer PCM, wav32f float, flac (24-bit lossless) or
        mp3 (320 kbps CBR). No session-state mutation, no guardrails; the
        planning agent must never call this."""
        return await to_thread.run_sync(_export_mix, req)

    # Integer full scale per PCM format: the float render is clipped to
    # [-1.0, max] before soundfile converts, so libsndfile can never
    # wrap-around on inter-stem sums hotter than full scale.
    PCM_FULL_SCALE = {"wav16": 32767 / 32768, "wav24": 8388607 / 8388608,
                      "flac": 8388607 / 8388608}
    PCM_SUBTYPE = {"wav16": "PCM_16", "wav24": "PCM_24", "flac": "PCM_24"}
    EXPORT_EXT = {"flac": ".flac", "mp3": ".mp3"}  # everything else is .wav
    MP3_MAX_SAMPLE_RATE = 48000  # libsndfile's lame stops here

    def _export_mix(req: ExportRequest) -> ExportResult:
        started = time.perf_counter()
        out_dir = Path(req.out_dir).expanduser().resolve()
        project = session.project_dir.resolve()
        if out_dir == project or project in out_dir.parents:
            raise ValueError(
                "export folder must be outside the project folder — "
                "exported audio there would be re-imported as stems")
        out_dir.mkdir(parents=True, exist_ok=True)

        name = session.project_dir.name
        audio, sample_rate = dsp.render_master(session)
        if req.format == "mp3" and sample_rate > MP3_MAX_SAMPLE_RATE:
            raise ValueError(
                f"mp3 export tops out at 48 kHz and this session renders "
                f"at {sample_rate / 1000:g} kHz — export wav or flac "
                "instead (no resampling, files ship at the session rate)")
        ext = EXPORT_EXT.get(req.format, ".wav")
        renders: list[tuple[Path, str, Optional[str], np.ndarray]] = [
            (out_dir / f"{name} - Master{ext}", "master", None, audio)]
        if req.include_stems:
            stems_dir = out_dir / f"{name} - Stems"
            stems_dir.mkdir(exist_ok=True)
            cache: dict = {}  # shared duck keys render once across stems
            for stem_id, state in sorted(session.stems.items()):
                if state["muted"] or state["excluded"]:
                    continue
                stem_audio, _ = dsp.render_stem(session, stem_id, _cache=cache)
                renders.append((stems_dir / f"{stem_id}{ext}", "stem",
                                stem_id, stem_audio))

        files = [_write_export(path, kind, stem_id, data, sample_rate,
                               req.format)
                 for path, kind, stem_id, data in renders]
        return ExportResult(
            out_dir=str(out_dir), format=req.format,
            sample_rate_hz=sample_rate,
            render_ms=int((time.perf_counter() - started) * 1000),
            files=files)

    def _write_export(path: Path, kind: str, stem_id: Optional[str],
                      audio: np.ndarray, sample_rate: int,
                      format: str) -> ExportedFile:
        import soundfile as sf

        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if format == "wav32f":
            # scipy, not soundfile: same PEAK-chunk rule as the bounce —
            # this file is bit-identical to bounces/master.wav.
            wavfile.write(path, sample_rate, audio.astype(np.float32))
            clipped = False
        elif format == "mp3":
            clipped = peak > 1.0
            # compression_level 0.0 under CONSTANT = lame's top CBR rung,
            # 320 kbps (probed against libsndfile 1.2.2)
            sf.write(str(path), np.clip(audio, -1.0, 1.0), sample_rate,
                     subtype="MPEG_LAYER_III", bitrate_mode="CONSTANT",
                     compression_level=0.0)
        else:
            full_scale = PCM_FULL_SCALE[format]
            clipped = peak > full_scale
            sf.write(str(path), np.clip(audio, -1.0, full_scale),
                     sample_rate, subtype=PCM_SUBTYPE[format])
        return ExportedFile(
            path=str(path), kind=kind, stem_id=stem_id,
            duration_s=round(audio.shape[0] / sample_rate, 2),
            peak_dbfs=round(20 * float(np.log10(peak)), 1) if peak > 0 else None,
            clipped=clipped)

    @mcp.tool()
    def advance_pass() -> dict:
        """Start the next pass: resets the per-pass gain budget and proposal count.
        Called by the orchestrator between passes."""
        return {"pass_number": session.advance_pass()}

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(description="Mix engine MCP server (stdio)")
    parser.add_argument("project_dir", help="project folder containing stems (*.wav)")
    args = parser.parse_args()
    create_server(args.project_dir).run(transport="stdio")


if __name__ == "__main__":
    main()
