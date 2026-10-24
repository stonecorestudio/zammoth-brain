"""A run stops once the composed EQ rail is spent (audit follow-ups 2026-09-11, item 1).

Measured on the rule benchmark's pop master before this: the rail was spent after pass 4, and
passes 5 and 6 re-applied an identical -10 LUFS limiter for a 0.3 LU residual — the run could
only end at "pass budget exhausted". On the LLM agent each of those is a paid planner round.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from orchestrator import benchmark as bm
from orchestrator.agents import PassContext, RuleAgent, TargetState


def rail_spent(*a):   # imported per call: the agent and run legs must fail on ASSERTIONS pre-fix
    from orchestrator.loop import rail_spent as fn
    return fn(*a)


SPENT = {"boost": 0.0, "cut": -12.0}
ROOM = {"boost": 4.0, "cut": -12.0}
LIMITER = {"type": "limiter", "stem_id": "master", "ceiling_dbtp": -1.0, "target_lufs_i": -10.0}


def _t(metric, current, goal):
    return TargetState(metric=metric, unit="dB", current=current, distance=abs(goal - current),
                       goal=goal)


def test_saturated_boosts_and_an_aimed_limiter_are_spent():
    states = [_t("band_energy.air", -20.0, -12.0), _t("lufs_i", -10.3, -10.0)]
    spent = rail_spent(states, {"master": {"air": SPENT}}, [LIMITER])
    assert spent and "air 8.0 dB under" in spent[0]


def test_anything_still_movable_keeps_the_run_alive():
    air = _t("band_energy.air", -20.0, -12.0)
    assert rail_spent([air], {"master": {"air": ROOM}}, []) is None, "room left"
    assert rail_spent([air, _t("lufs_i", -14.0, -10.0)], {"master": {"air": SPENT}}, []) is None, \
        "loudness with no limiter aimed at it can still move"
    masking = TargetState(metric="masking(a, b).worst_band_excess_db", unit="dB", current=4.0,
                          distance=2.0, goal=2.0)
    assert rail_spent([air, masking], {"master": {"air": SPENT}}, []) is None, "a non-EQ target"
    cut = _t("band_energy.mid", -2.0, -9.0)
    assert rail_spent([cut], {"master": {"mid": SPENT}}, []) is None, "a CUT with cut room left"
    assert rail_spent([cut], {"master": {"mid": {"boost": 6.0, "cut": 0.0}}}, [])


def test_a_stem_scoped_band_reads_its_own_target():
    states = [_t("stem(piano).band_energy.low_mid", -6.0, -12.0)]
    assert rail_spent(states, {"piano": {"low_mid": {"boost": 6.0, "cut": 0.0}}}, [])
    assert rail_spent(states, {"master": {"low_mid": {"boost": 6.0, "cut": 0.0}}}, []) is None


def test_the_rule_agent_does_not_repeat_the_limiter_it_already_placed():
    target = _t("lufs_i", -10.3, -10.0)

    def ctx(chain):
        return PassContext(pass_number=5, metrics={"stems": []}, brief="pop master",
                           targets=[target], stem_chains={}, master_chain=chain)
    assert not [p for p in RuleAgent().propose(ctx([LIMITER]))
                if p["actions"][0]["type"] == "limiter"]
    assert [p for p in RuleAgent().propose(ctx([]))
            if p["actions"][0]["type"] == "limiter"], "control: no limiter yet → it proposes one"


def test_pop_master_stops_at_the_rail_instead_of_the_pass_budget():
    case = next(c for c in bm.BENCHMARK_CASES if c.brief == "pop master")
    res = bm.run_case(case, RuleAgent, Path(tempfile.mkdtemp(prefix="rail-stop-")))
    assert res["stop_reason"] == "EQ rail reached", res["stop_reason"]
    assert res["passes_run"] < 6, res["passes_run"]
    assert res["mean_closure"] >= 0.69, res["mean_closure"]   # the honest closure under the rail


def test_a_run_still_short_of_the_loudness_goal_is_not_spent():
    """C5 (2026-09-11 pre-merge review): the limiter aiming at the goal was taken as
    "settled" whatever the DISTANCE, so a run 3-6 LU short stopped saying "EQ rail reached"
    — the agent reporting a rail it had not reached. Only the limiter's own residual counts."""
    air = _t("band_energy.air", -20.0, -12.0)
    spent_air = {"master": {"air": SPENT}}
    for short in (3.0, 6.0):
        states = [air, _t("lufs_i", -10.0 - short, -10.0)]
        assert rail_spent(states, spent_air, [LIMITER]) is None, \
            f"{short} LU short of the goal is a move the run can still make"
    # the residual the rule exists for is still settled (control: the 0.3 LU case above)
    assert rail_spent([air, _t("lufs_i", -10.3, -10.0)], spent_air, [LIMITER])
    # …and the boundary is the residual, not the aim
    from orchestrator.loop import LIMITER_RESIDUAL_LU as LIM
    assert rail_spent([air, _t("lufs_i", -10.0 - LIM, -10.0)], spent_air, [LIMITER]), "at the rail"
    assert rail_spent([air, _t("lufs_i", -10.0 - LIM - 0.2, -10.0)], spent_air, [LIMITER]) is None
