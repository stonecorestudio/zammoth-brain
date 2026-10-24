"""Stopping Auto Mix (2026-08-05 audit).

Auto Mix applies a chain of bundles through the proposal lifecycle. The only way
to stop it was `task.cancel()`, which raises CancelledError somewhere inside the
bundle loop — so `_apply_bundles`' return value never reached the caller, and the
ids of everything ALREADY APPLIED were lost. The gateway then reported
"applied: 0" while N bundles sat in the session outside the one-click-undo set:
changes the user could hear, that the button built to take them back could not
reach. On the DAW path there was no cancel token at all, so Stop did nothing.

Two properties here: a stop lands on a bundle boundary, and whatever landed
before it stays undoable.
"""
import anyio
import pytest

from orchestrator import automix


class FakeEngine:
    """Records the lifecycle calls a bundle makes, and accepts everything."""

    def __init__(self):
        self.calls: list[str] = []
        self.applied: list[str] = []
        self._n = 0

    async def call(self, name: str, args: dict | None = None):
        self.calls.append(name)
        if name == "propose_actions":
            self._n += 1
            return {"accepted": True, "proposal_id": f"p{self._n}"}
        if name == "apply_proposal":
            self.applied.append(args["proposal_id"])
            return {"ok": True}
        return {}

    async def render_and_analyze(self):
        self.calls.append("render_and_analyze")
        return {"analysis": {}, "flags": [], "master": {}, "bounce_path": ""}


def _bundle(i: int) -> dict:
    return {"rationale": f"move {i}", "predicted_impact": [],
            "actions": [{"type": "gain", "stem_id": "s", "delta_db": -1.0}]}


def _run(coro_factory):
    events: list[dict] = []

    async def emit(event: dict) -> None:
        events.append(event)

    result = anyio.run(lambda: coro_factory(emit))
    return result, events


def test_a_cancel_stops_at_the_next_bundle_boundary():
    engine = FakeEngine()
    bundles = [_bundle(i) for i in range(6)]
    stop_after = 2

    def should_cancel():
        return len(engine.applied) >= stop_after

    applied, events = _run(lambda emit: automix._apply_bundles(
        engine, emit, bundles, should_cancel=should_cancel))

    assert len(applied) == stop_after, "must stop, and not mid-bundle"
    assert engine.applied == applied
    assert any(e.get("stage") == "cancelled" for e in events), \
        "the user is told it stopped"


def test_what_landed_before_the_stop_is_still_undoable():
    """The regression that mattered: the caller's undo set must hold the ids even
    though the loop never returned normally."""
    engine = FakeEngine()
    undo_set: list[str] = []

    def should_cancel():
        return len(undo_set) >= 2

    _run(lambda emit: automix._apply_bundles(
        engine, emit, [_bundle(i) for i in range(6)],
        should_cancel=should_cancel, applied=undo_set))

    assert undo_set == engine.applied
    assert len(undo_set) == 2, "the caller holds exactly what was applied"


def test_the_caller_keeps_the_ids_even_if_the_loop_never_returns():
    """A hard cancel (CancelledError mid-loop) is the original failure. The list is
    the caller's, so it survives the exception that discards the return value."""
    engine = FakeEngine()
    undo_set: list[str] = []

    def should_cancel():
        if len(undo_set) >= 3:
            raise anyio.get_cancelled_exc_class()()
        return False

    with pytest.raises(BaseException):
        _run(lambda emit: automix._apply_bundles(
            engine, emit, [_bundle(i) for i in range(6)],
            should_cancel=should_cancel, applied=undo_set))

    assert len(undo_set) == 3, \
        "ids applied before a hard cancel must still reach the undo set"
    assert undo_set == engine.applied


def test_no_cancel_token_applies_everything():
    engine = FakeEngine()
    applied, _events = _run(lambda emit: automix._apply_bundles(
        engine, emit, [_bundle(i) for i in range(4)]))
    assert len(applied) == 4


def test_a_cancelled_apply_bundles_never_starts_a_bundle():
    engine = FakeEngine()

    applied, _events = _run(lambda emit: automix._apply_bundles(
        engine, emit, [_bundle(0)], should_cancel=lambda: True))

    assert applied == []
    assert "propose_actions" not in engine.calls, "cancelled before the first bundle"


def test_a_cancelled_bootstrap_skips_the_render():
    """The render is the most expensive step; a stop should not pay for it.

    Drives `bootstrap` itself — the function that owns the render-skip logic. The
    old version of this test called `_apply_bundles`, which renders on no path, so
    deleting the guard in `bootstrap` left it green (2026-09-01 sweep)."""
    engine = FakeEngine()
    applied: list[str] = []

    _result, events = _run(lambda emit: automix.bootstrap(
        engine, "quick balance", emit, should_cancel=lambda: True, applied=applied))

    assert applied == []
    assert "render_and_analyze" not in engine.calls
    assert "propose_actions" not in engine.calls
