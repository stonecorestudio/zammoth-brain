"""Prediction accountability: resolve metric names to values, capture pre-apply
baselines, and fill measured_impact after the next render.

Metric name grammar (the same names agents use in `predicted_impact`):
    lufs_i | lufs_s_max | true_peak_dbtp | crest_factor_db
    | stereo_correlation | spectral_tilt_db_per_oct
    band_energy.<band>
    section_band_energy.<index>.<band>
    stem(<id>).lufs_s | stem(<id>).peak_dbfs | stem(<id>).dynamic_spread_db
    | stem(<id>).band_energy.<band>
    masking(<a>,<b>).worst_band_excess_db | masking(<a>,<b>).<band>

Unknown names resolve to None (recorded, never fatal): a bad metric name should
cost the agent its accountability data, not crash an apply.
"""
from __future__ import annotations

import math
import re
from typing import Optional

from . import analysis, dsp
from .models import MetricMeasurement, ProbeMeasurement
from .session import Session

_STEM_RE = re.compile(r"^stem\(([^)]+)\)\.(.+)$")
_MASKING_RE = re.compile(r"^masking\(([^,)]+),\s*([^)]+)\)\.(.+)$")
_SECTION_BAND_RE = re.compile(r"^section_band_energy\.(\d+)\.([^.]+)$")


class _Resolver:
    """Computes metric values for the current session state, rendering each
    needed source (master / stem / stem pair) at most once. With `range_s`
    every measurement is confined to that time slice (Phase 12 probes)."""

    def __init__(self, session: Session,
                 range_s: Optional[tuple[float, float]] = None):
        self.session = session
        self.range_s = range_s
        self._master = None
        self._sections = None
        self._stems: dict[str, object] = {}
        self._masking: dict[tuple[str, str], object] = {}

    def _slice(self, audio, sample_rate: int):
        if self.range_s is None:
            return audio
        start_s, end_s = self.range_s
        return audio[int(start_s * sample_rate):int(end_s * sample_rate)]

    def value(self, metric: str) -> Optional[float]:
        try:
            return self._value(metric)
        except Exception:
            return None

    def _value(self, metric: str) -> Optional[float]:
        if match := _STEM_RE.match(metric):
            stem_id, field = match.groups()
            return _field(self._stem_analysis(stem_id), field)
        if match := _MASKING_RE.match(metric):
            stem_a, stem_b, field = match.groups()
            masking = self._masking_analysis(stem_a, stem_b)
            if field == "worst_band_excess_db":
                return masking.worst_band_excess_db
            if field == "spatial_release_db":  # Phase 20: pan moves register
                return masking.spatial_release_db
            return masking.per_band_excess_db.get(field)
        if match := _SECTION_BAND_RE.match(metric):
            index, band = match.groups()
            sections = self._section_analysis().sections
            # Index is a loudness rank (0 = loudest), stable across renders —
            # see analysis._weighted_section_band_energy.
            ranked = sorted(
                sections,
                key=lambda s: s.lufs_i if s.lufs_i is not None else -120.0,
                reverse=True)
            idx = int(index)
            if idx >= len(ranked):
                return None
            return (ranked[idx].band_energy or {}).get(band)
        return _field(self._master_analysis(), metric)

    def _master_analysis(self):
        if self._master is None:
            audio, sample_rate = dsp.render_master(self.session,
                                                   range_s=self.range_s)
            self._master = analysis.analyze_master(
                audio, sample_rate, stems=sorted(self.session.stems),
                pass_number=self.session.pass_number)
        return self._master

    def _section_analysis(self):
        if self._sections is None:
            audio, sample_rate = dsp.render_master(self.session,
                                                   range_s=self.range_s)
            self._sections = analysis.analyze_sections(audio, sample_rate, {})
        return self._sections

    def _stem_analysis(self, stem_id: str):
        if stem_id not in self._stems:
            audio, sample_rate = dsp.render_stem(self.session, stem_id)
            audio = self._slice(audio, sample_rate)
            self._stems[stem_id] = analysis.analyze_stem(
                audio, sample_rate, stem_id=stem_id,
                applied_chain=self.session.applied_chain(stem_id))
        return self._stems[stem_id]

    def _masking_analysis(self, stem_a: str, stem_b: str):
        key = (stem_a, stem_b)
        if key not in self._masking:
            audio_a, sample_rate = dsp.render_stem(self.session, stem_a)
            audio_b, _ = dsp.render_stem(self.session, stem_b)
            self._masking[key] = analysis.analyze_masking(
                self._slice(audio_a, sample_rate),
                self._slice(audio_b, sample_rate),
                sample_rate, stem_a=stem_a, stem_b=stem_b)
        return self._masking[key]


def _field(model, field: str) -> Optional[float]:
    """A metric's value, or None when there is none - and a non-finite number is none (reader
    E5·R8): a NaN here became a `measured_delta` of NaN on the proposal, and from there a
    running sum in the cross-project calibration store that never recovers."""
    if field.startswith("band_energy."):
        bands = model.band_energy or {}
        value = bands.get(field.split(".", 1)[1])
    else:
        value = getattr(model, field, None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def compute_metrics(session: Session, names: list[str]) -> dict[str, Optional[float]]:
    resolver = _Resolver(session)
    return {name: resolver.value(name) for name in names}


def probe(session: Session, actions: list, metrics: list[str],
          range_s: Optional[tuple[float, float]] = None,
          ) -> list[ProbeMeasurement]:
    """What-if measurement (Phase 12): named metrics before/after `actions`,
    applied to a scratch copy of session state — the real session is never
    touched and nothing persists. Deterministic by construction: the scratch
    copy mutates through the same Session._mutate path apply() uses, so a
    probe's `after` equals the post-apply measurement on the same range."""
    before = _Resolver(session, range_s=range_s)
    scratch = session.probe_copy()
    for action in actions:
        scratch._mutate(action)
    after = _Resolver(scratch, range_s=range_s)
    measurements = []
    for name in metrics:
        value_before, value_after = before.value(name), after.value(name)
        measurements.append(ProbeMeasurement(
            metric=name, before=value_before, after=value_after,
            delta=(None if value_before is None or value_after is None
                   else round(value_after - value_before, 2))))
    return measurements


def apply_with_baseline(session: Session, proposal_id: str):
    """Apply a proposal, capturing pre-apply values of its predicted metrics so
    measured_impact can be computed at the next render."""
    names = _predicted_metric_names(session, proposal_id)
    baseline = compute_metrics(session, names) if names else {}
    result = session.apply(proposal_id)
    if result.ok and baseline:
        session.set_baseline(proposal_id, baseline)
    return result


def measure_impacts(session: Session, master_analysis=None,
                    ) -> list[tuple[str, list[MetricMeasurement]]]:
    """The measurement half of fill_measured_impacts: resolve every applied-
    awaiting-measurement record to its per-metric deltas, without writing
    anything back. Safe to run against a frozen state copy in a render
    thread — the caller records the results on the live session.

    One resolver serves every awaiting record, and `master_analysis` (the
    SessionAnalysis the caller just computed from its own render) seeds it so
    master metrics don't trigger a second render (Phase 10)."""
    records = session.applied_awaiting_measurement()
    if not records:
        return []
    resolver = _Resolver(session)
    if master_analysis is not None:
        resolver._master = master_analysis
    measured = []
    for record in records:
        current = {name: resolver.value(name) for name in record["baseline"]}
        measurements = []
        for prediction in record["predicted_impact"]:
            name = prediction["metric"]
            before, after = record["baseline"].get(name), current.get(name)
            if before is None or after is None:
                continue
            measurements.append(MetricMeasurement(
                metric=name, measured_delta=round(after - before, 2),
                unit=prediction["unit"]))
        measured.append((record["proposal_id"], measurements))
    return measured


def fill_measured_impacts(session: Session, master_analysis=None) -> None:
    """Called after a render: fill measured_impact on applied proposals that
    still lack it, as (current value - pre-apply baseline) per predicted
    metric."""
    for proposal_id, measurements in measure_impacts(session, master_analysis):
        session.record_measured(proposal_id, measurements)


def _predicted_metric_names(session: Session, proposal_id: str) -> list[str]:
    record = session._require_proposal(proposal_id)
    return [p["metric"] for p in record["predicted_impact"]]
