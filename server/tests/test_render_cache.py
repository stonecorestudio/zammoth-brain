"""Persistent stem-render cache: re-rendering the mix after one stem changed
re-renders that stem only — the rest serve cached audio, bit-identically.
Off by default (engine-pure runs keep exact render counts); the gateway
enables it per process. Telemetry rides the cache so gain-reduction stats
survive hits; the byte budget bounds memory and eviction only costs a
re-render."""
import numpy as np

from mix_engine import dsp
from mix_engine.session import Session

COMP = {"type": "comp", "threshold_db": -35.0, "ratio": 4.0,
        "attack_ms": 5.0, "release_ms": 100.0, "makeup_db": 0.0}


def _counts() -> dict:
    return dict(dsp.RENDER_COUNTS)


def _reset_counts() -> None:
    dsp.RENDER_COUNTS.update(master=0, stem=0, bus=0)


def test_cached_rerender_is_bit_identical_and_skips_stems(demo_project):
    session = Session(demo_project)

    dsp.configure_render_cache(0)  # cold reference, cache off
    reference, _ = dsp.render_master(session)

    dsp.configure_render_cache(256)
    _reset_counts()
    first, _ = dsp.render_master(session)  # populates the cache
    assert _counts()["stem"] == 2
    second, _ = dsp.render_master(session)  # serves it
    assert _counts()["stem"] == 2  # no stem re-rendered
    assert np.array_equal(first, reference)
    assert np.array_equal(second, reference)


def test_changed_stem_rerenders_alone(demo_project):
    session = Session(demo_project)
    dsp.configure_render_cache(256)
    dsp.render_master(session)

    session.user_edit("stem_a", {"gain_db": -3.0})
    _reset_counts()
    dsp.render_master(session)
    assert _counts()["stem"] == 1  # stem_b came from the cache

    # the cached path stays bit-identical to a cold render of the same state
    dsp.configure_render_cache(0)
    dsp.RENDER_CACHE.clear()
    cold, _ = dsp.render_master(session)
    dsp.configure_render_cache(256)
    warm, _ = dsp.render_master(session)
    assert np.array_equal(warm, cold)


def test_telemetry_survives_cache_hits(demo_project):
    session = Session(demo_project)
    session.stems["stem_a"]["chain"].append(dict(COMP))
    dsp.configure_render_cache(256)

    first: dict = {}
    dsp.render_master(session, stem_telemetry=first)
    assert any(s["processor"] == "comp" for s in first.get("stem_a", []))

    _reset_counts()
    second: dict = {}
    dsp.render_master(session, stem_telemetry=second)
    assert _counts()["stem"] == 0  # full cache hit
    assert second.get("stem_a") == first.get("stem_a")


def test_budget_bounds_and_eviction_recovers(demo_project):
    session = Session(demo_project)
    # a budget too small for any stem: nothing caches, renders stay correct
    dsp.configure_render_cache(0)
    reference, _ = dsp.render_master(session)
    dsp.RENDER_CACHE.configure(1)
    dsp.RENDER_CACHE.budget_bytes = 1024  # smaller than any rendered stem
    _reset_counts()
    dsp.render_master(session)
    out, _ = dsp.render_master(session)
    assert _counts()["stem"] == 4  # both renders did real work
    assert dsp.RENDER_CACHE._bytes == 0
    assert np.array_equal(out, reference)
