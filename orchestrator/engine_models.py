"""Import the engine's contract objects (models + limits) for orchestrator use.

The tool/proposal schemas come from the engine, never duplicated here
(docs/02-architecture.md). The engine package lives under server/; when the
orchestrator runs from the repo root (CLI), add server/ to the path first.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    from mix_engine.guardrails import LIMITS
    from mix_engine.impact import _MASKING_RE as MASKING_METRIC_RE
    from mix_engine.impact import _STEM_RE as STEM_METRIC_RE
    from mix_engine.instruments import (CLASSIFIER_TAGS, INSTRUMENT_WORDS,
                                        TAG_SYNONYMS, _ID_NOISE_TOKENS,
                                        _TEXT_AMBIGUOUS, roles_of_stem)
    from mix_engine.models import BAND_EDGES_HZ, BAND_LABELS, ProposalRequest
except ImportError:  # running from repo root: put server/ on the path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
    from mix_engine.guardrails import LIMITS
    from mix_engine.impact import _MASKING_RE as MASKING_METRIC_RE
    from mix_engine.impact import _STEM_RE as STEM_METRIC_RE
    from mix_engine.instruments import (CLASSIFIER_TAGS, INSTRUMENT_WORDS,
                                        TAG_SYNONYMS, _ID_NOISE_TOKENS,
                                        _TEXT_AMBIGUOUS, roles_of_stem)
    from mix_engine.models import BAND_EDGES_HZ, BAND_LABELS, ProposalRequest

__all__ = ["BAND_EDGES_HZ", "BAND_LABELS", "CLASSIFIER_TAGS", "INSTRUMENT_WORDS",
           "LIMITS", "MASKING_METRIC_RE", "STEM_METRIC_RE", "TAG_SYNONYMS",
           "ProposalRequest", "_ID_NOISE_TOKENS", "_TEXT_AMBIGUOUS",
           "roles_of_stem"]
