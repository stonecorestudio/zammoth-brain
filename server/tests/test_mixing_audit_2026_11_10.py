"""The mixing audit of 2026-11-10 (docs/audits/mixing-2026-11-10/findings.md): each test was
seen red on the tree the audit read."""
from mix_engine.session import Session

from test_audit_p1_fixes import A, B, _apply, _propose, _rules


def _ride(delta: float, rng, stem: str = "stem_b") -> dict:
    return {"type": "gain", "stem_id": stem, "delta_db": delta, "time_range_s": list(rng)}


# ---------- M4: two cuts in different parts of the song are not one cut ----------

def test_cuts_in_two_separate_ranges_are_charged_by_what_plays_at_once(demo_project):
    """'Kick down 4 dB everywhere except the drop' is two ranges; the level is -4 at any
    moment. The wallet summed them to -8 and refused (the EDM session, 2026-11-02)."""
    session = Session(demo_project)
    both = _propose(session, [_ride(-4.0, A), _ride(-4.0, B)])
    assert both.accepted, _rules(both)


def test_cuts_on_the_same_range_still_add_up(demo_project):
    session = Session(demo_project)
    stacked = _propose(session, [_ride(-4.0, A), _ride(-4.0, A)])
    assert not stacked.accepted and "gain_per_stem_per_pass_db" in _rules(stacked)


def test_ranges_that_overlap_add_up_where_they_overlap(demo_project):
    session = Session(demo_project)
    a0, a1 = A
    inside = [a0 + (a1 - a0) / 4.0, a1 + 1.0]            # starts inside A
    over = _propose(session, [_ride(-4.0, A), _ride(-4.0, inside)])
    assert not over.accepted and "gain_per_stem_per_pass_db" in _rules(over)


def test_a_second_proposal_sees_the_first_ones_ranges(demo_project):
    session = Session(demo_project)
    _apply(session, [_ride(-4.0, A)])
    _apply(session, [_ride(-4.0, B)])                      # elsewhere: fits
    again = _propose(session, [_ride(-4.0, A)])            # on A again: -8 there
    assert not again.accepted and "gain_per_stem_per_pass_db" in _rules(again)


def test_a_ranged_cut_is_still_no_credit_for_a_lift(demo_project):
    """D9.R4 stays shut: one-sided, whatever the ranges."""
    session = Session(demo_project)
    walk = _propose(session, [_ride(-6.0, B), _ride(12.0, A)])
    assert not walk.accepted and "gain_per_stem_per_pass_db" in _rules(walk)


# ---------- M6: the brief compiler reads ordinary requests as what they say ----------

from orchestrator import automix as _automix, targets as _targets   # noqa: E402

_STEMS = ["vocal", "snare", "kick", "bass", "guitar"]


def _reads(brief: str) -> list[str]:
    try:
        return list(_targets.interpret_brief(brief, _STEMS))
    except _targets.BriefNotUnderstood:
        return []                                      # left to a reader: nothing compiled


def test_a_genre_word_used_as_a_verb_or_on_an_instrument_is_no_genre():
    assert "pop tonal balance" not in _reads("make the snare pop")
    assert "hip hop tonal balance" not in _reads("the rap vocal is harsh")
    assert "hip hop tonal balance" not in _reads("make the rap the star")
    assert "acoustic tonal balance" not in _reads("the acoustic guitar is muddy")
    assert "edm tonal balance" not in _reads("the electronic drums are too loud")
    assert "rock tonal balance" not in _reads("make the chorus rock harder")


def test_a_genre_named_as_a_genre_still_is_one():
    assert "pop tonal balance" in _reads("pop master")
    assert "pop tonal balance" in _reads("master it as a pop song")
    assert "hip hop tonal balance" in _reads("make it sound like a hip hop record")
    assert "hip hop tonal balance" in _reads("trap banger, loud")
    assert "hip hop tonal balance" in _reads("a rap record")
    assert "edm tonal balance" in _reads("master for edm")
    assert "acoustic tonal balance" in _reads("acoustic folk master")
    assert "rock tonal balance" in _reads("rock mix")


def test_checking_mono_is_not_asking_for_narrow():
    assert "tighter, more centered image" not in _reads("make sure it works in mono")
    assert "tighter, more centered image" not in _reads("keep it mono compatible")
    assert "tighter, more centered image" in _reads("make it narrower")


def test_a_tone_word_on_one_section_is_not_the_whole_song():
    assert "wider stereo image" not in _reads("wider chorus")
    assert "wider stereo image" not in _reads("make the hook wider")
    assert "wider stereo image" in _reads("make it wider")


def test_keep_the_punch_is_not_more_punch():
    reads = _reads("warmer but keep the punch")
    assert any(r.startswith("warmer") for r in reads) and not any(r.startswith("punchier") for r in reads)
    assert any(r.startswith("punchier") for r in _reads("make it punchier"))


def test_auto_mix_does_not_master_warm_for_a_warmer_vocal():
    assert _automix.intent_from_brief("make the vocal warmer") is None
    assert _automix.intent_from_brief("warmer guitar please") is None
    assert _automix.intent_from_brief("warm master") == "warm-master"
    assert _automix.intent_from_brief("make it warmer") == "warm-master"


# ---------- M2: wide is not cancelling ----------

import numpy as np   # noqa: E402

from mix_engine import phase as _phase   # noqa: E402

_SR = 44100


def _mix_with_section(section) -> np.ndarray:
    """20 s of a centred sound with a 12 s stretch (60 % of it) that `section` makes stereo."""
    rng = np.random.default_rng(7)
    centre = rng.standard_normal(20 * _SR).astype(np.float32) * 0.1
    audio = np.stack([centre, centre], axis=1)
    a, b = 6 * _SR, 18 * _SR
    audio[a:b] = section(rng, b - a)
    return audio


def _flagged(audio) -> list[str]:
    bands = _phase.band_reading(_phase.WindowSums(audio, _SR))
    return [label for label, band in bands.items() if _phase.master_cancels(band)]


def test_a_wide_section_of_the_master_is_not_called_cancelling():
    """A double-tracked hook: two unrelated sides for most of the song. It loses 3 dB in
    mono there, as every wide part does, and nothing cancels. The rule counted the time its
    correlation dipped under 0 and told the engineer to narrow it (hip-hop, 2026-11-02)."""
    def double(rng, n):
        return np.stack([rng.standard_normal(n), rng.standard_normal(n)], axis=1).astype(np.float32) * 0.1
    assert _flagged(_mix_with_section(double)) == []


def test_an_anti_phase_section_of_the_master_still_is():
    def flipped(rng, n):
        side = rng.standard_normal(n).astype(np.float32) * 0.1
        return np.stack([side, -side], axis=1)
    assert _flagged(_mix_with_section(flipped))


def test_the_mono_safe_width_stops_where_the_stem_stops_cancelling():
    """A part whose sides are mildly anti-phase (-0.4) was sent to half its width - the width
    at which it reads +0.2. Where it stops cancelling (0) is about two thirds."""
    rng = np.random.default_rng(3)
    mid = rng.standard_normal(10 * _SR) * 0.1
    side = rng.standard_normal(10 * _SR) * 0.1 * np.sqrt(1.4 / 0.6)      # corr = (M-S)/(M+S) = -0.4
    audio = np.stack([mid + side, mid - side], axis=1).astype(np.float32)
    width = _phase.mono_safe_width(_phase.WindowSums(audio, _SR), 1.0)
    assert width is not None and 0.6 <= width <= 0.7, width


# ---------- M5: Auto Mix on a real session ----------

def _stem(lufs_s: float, peak: float, spread: float = 6.0, **more) -> dict:
    return {"lufs_s": lufs_s, "true_peak_dbtp": peak, "dynamic_spread_db": spread,
            "band_energy": {"sub": -40.0}, "chain": [], **more}


def _actions(bundles) -> list[dict]:
    return [a for b in bundles for a in b["actions"]]


def _masking(a: str, b: str, band: str, excess: float) -> dict:
    return {"stem_a": a, "stem_b": b, "worst_band": band, "worst_band_excess_db": excess}


def test_auto_mix_never_carves_a_lead_or_the_low_end_in_its_own_band():
    """On a real rap session it cut the lead 4 dB at 2.8 kHz and the 808 4 dB at 244 Hz: the
    louder stem of a pair is 'the masker' in the band where it is loudest - its own."""
    tags = {"rap": ["vocal"], "piano": [], "kick": ["kick"], "sub": ["bass"]}
    stems = {k: _stem(-18.0, -6.0) for k in tags}
    maskings = [_masking("piano", "rap", "high_mid", 9.0), _masking("kick", "sub", "low_mid", 9.0),
                _masking("sub", "kick", "bass", 9.0), _masking("rap", "piano", "mid", 9.0)]
    cuts = {(a["stem_id"], round(a["band"]["freq_hz"])) for a in _actions(
        _automix.build_bundles(tags, stems, maskings, {"lufs_i": -18.0}, goal_lufs=None))
        if a["type"] == "eq" and a["band"].get("gain_db", 0) < 0}
    assert not any(stem in ("rap", "kick", "sub") for stem, _ in cuts), cuts
    assert any(stem == "piano" for stem, _ in cuts), cuts      # control: the bed is still carved


def test_auto_mix_keeps_the_lead_vocals_own_compressor_when_vocals_are_grouped():
    tags = {"rap": ["vocal"], "hook": ["vocal"], "kick": ["kick"], "hats": ["cymbal"]}
    stems = {"rap": _stem(-18.0, -6.0, spread=19.0), "hook": _stem(-18.0, -6.0),
             "kick": _stem(-18.0, -6.0), "hats": _stem(-18.0, -6.0)}
    comps = [a for a in _actions(_automix.build_bundles(tags, stems, [], {"lufs_i": -18.0},
                                                         goal_lufs=None))
             if a["type"] == "comp" and a.get("stem_id") == "rap"]
    assert comps, "the take that swings 19 dB has no compressor of its own"


def test_auto_mix_sets_a_drum_compressor_by_its_peaks():
    """A clap's peaks sit 12-15 dB over its loudness: a threshold 4 dB under the loudness is
    ~17 dB under the hit, at 3:1 and 2 ms - the hit is flattened."""
    tags = {"clap": [], "pad": ["pad"]}
    stems = {"clap": _stem(-30.0, -15.0, spread=20.0, transient_energy_ratio=0.5,
                           transient_density=2.0), "pad": _stem(-18.0, -8.0)}
    comp = next(a for a in _actions(_automix.build_bundles(tags, stems, [], {"lufs_i": -18.0},
                                                            goal_lufs=None))
                if a["type"] == "comp" and a["stem_id"] == "clap")
    assert comp["threshold_db"] >= -15.0 - 8.0 - 1e-6, comp


def test_auto_mix_says_when_a_stem_is_further_out_than_it_may_move():
    tags = {"hats": ["cymbal"], "a": [], "b": [], "c": []}
    stems = {"hats": _stem(-60.0, -45.0), "a": _stem(-18.0, -6.0), "b": _stem(-18.0, -6.0),
             "c": _stem(-18.0, -6.0)}
    said = " ".join(b["rationale"] for b in _automix.build_bundles(
        tags, stems, [], {"lufs_i": -18.0}, goal_lufs=None))
    assert "hats" in said and "clip gain" in said, said


# ---------- M1 / M7: what both prompts say ----------

def test_both_daw_prompts_carry_the_audits_rules():
    from orchestrator import chat, prompts
    for text in (chat.DAW_SYSTEM_PROMPT, prompts.DAW_PROMPT_WHOLE):
        flat = " ".join(text.split())
        assert "Carve the masker, don't just push the maskee up" not in flat
        assert "Faders before processors" in flat and "a fader is the first fix for a buried lead" in flat
        assert "in tune with each other" in flat
        assert "never narrowed below its file's width without asking" in flat
        assert "about -10 LUFS or louder" in flat
        assert "what you heard" not in flat and "re-listen" not in flat
        assert "serial compression" not in flat          # the DAW refuses a second whole-file comp


def test_a_lone_clip_with_a_gain_is_reported_as_a_list_of_one():
    """M3, the C++ half read as text (the leg is --print-facts (d)): `stemClipWindows` no longer
    answers nothing for one clip, and the single window is dropped when the list is sent."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / "daw" / "Source"
    session = (root / "DawSession.h").read_text()
    assert "if (waveClips < 2 || ws.empty()" not in session
    assert "waveClips < 2 && std::abs (ws[0].gain) < 0.005" in session
    assert "ZAM_TMP_OLD_ONECLIP" not in session
    assert "if (f.clips.isNotEmpty()) f.clip = {};" in (root / "App.cpp").read_text()


# ---------- the second batch (2026-11-10): what a mix RUN may do to meet a number ----------

from orchestrator import agents as _agents, loop as _loop   # noqa: E402


def _state(metric, current, distance, goal=None, max_value=None, unit="dB"):
    return _agents.TargetState(metric=metric, unit=unit, current=current, distance=distance,
                               goal=goal, max_value=max_value)


def _ctx(targets, pass_number=1, engine_call=None):
    return _agents.PassContext(pass_number=pass_number, metrics={}, targets=targets,
                               stem_chains={}, engine_call=engine_call)


def test_a_clearer_vocal_is_not_measured_against_the_low_end():
    tags = {"vocal": ["vocal"], "kick": ["kick"], "bass": ["bass"], "guitar": []}
    metrics = [t.metric for t in _targets.targets_from_brief(
        "make the vocal clearer", list(tags), stem_tags=tags)]
    assert metrics == ["masking(vocal,guitar).worst_band_excess_db"], metrics


def test_a_masker_is_not_pulled_down_without_end_for_a_target_out_of_reach():
    agent = _agents.RuleAgent()
    total = 0.0
    for n in range(1, 6):
        target = _state("masking(vocal,guitar).worst_band_excess_db", 23.0, 20.0, max_value=3.0)
        for proposal in agent.propose(_ctx([target], n)):
            total += sum(a["delta_db"] for a in proposal["actions"] if a["type"] == "gain")
    assert -6.0 - 1e-9 <= total < 0.0, total


def test_a_probe_that_cannot_meet_the_target_takes_the_small_cut_not_the_big_one():
    def engine_call(name, args):
        return {"metrics": {"masking(vocal,guitar).worst_band_excess_db": 15.0}}
    agent = _agents.ProbingRuleAgent()
    agent._probe = lambda ctx, actions, metrics: {metrics[0]: 15.0}
    target = _state("masking(vocal,guitar).worst_band_excess_db", 23.0, 20.0, max_value=3.0)
    proposal = agent._masking_proposal(_ctx([target], engine_call=engine_call), target)
    assert proposal["actions"][0]["delta_db"] == -2.0, proposal["actions"]


def test_band_matching_does_not_invent_content_the_mix_does_not_have():
    """A voice-and-guitar mix has no sub; the acoustic curve asked for +12 dB of 20-60 Hz."""
    agent = _agents.RuleAgent()
    far = _state("band_energy.sub", -28.0, 14.0, goal=-14.0)
    near = _state("band_energy.mid", -11.0, 5.0, goal=-6.0)
    proposal = agent._band_match_proposal(_ctx([far, near]))
    gains = proposal["actions"][0]["band_gains_db"]
    assert "sub" not in gains and gains["mid"] > 0.0, gains
    assert "sub" in proposal["rationale"]                 # and it says what it left


def test_a_run_says_what_it_cost_the_parts_nobody_asked_about():
    before = {"stems": [{"stem_id": "kick", "gain_db": 0.0}, {"stem_id": "vocal", "gain_db": 0.0}]}
    after = {"stems": [{"stem_id": "kick", "gain_db": -8.0}, {"stem_id": "vocal", "gain_db": 1.0}]}
    notes = _loop.collateral_notes(before, after, {"crest_factor_db": 14.0},
                                   {"crest_factor_db": 10.0}, loudness_asked=False)
    assert any("kick" in n and "8" in n for n in notes), notes
    assert any("crest" in n or "punch" in n for n in notes), notes
    assert not any("vocal" in n for n in notes)
    # control: a loudness brief expects the crest to fall, and small moves say nothing
    assert _loop.collateral_notes(before, before, {"crest_factor_db": 14.0},
                                  {"crest_factor_db": 10.0}, loudness_asked=True) == []


def _limiter(gr_mean, gr_max, active=50.0):
    return [{"processor": "limiter", "gr_mean_db": gr_mean, "gr_max_db": gr_max,
             "gr_active_pct": active}]


def test_a_loud_goal_is_not_over_limiting_for_being_loud():
    """A -8 LUFS master at -1 dBTP has a PLR of 7 by arithmetic: the flag fired on every club,
    EDM and hip-hop goal the product itself sets, and the prompt said to back off."""
    from mix_engine import analysis
    assert not analysis.over_limited(_limiter(1.5, 4.5), psr_db=7.0, plr_db=7.0, target_lufs_i=-8.0)
    # ...unless the limiter is really working hard for it
    assert analysis.over_limited(_limiter(3.5, 9.0), psr_db=6.0, plr_db=7.0, target_lufs_i=-8.0)
    # control: with no loud goal the rule is the one it was
    assert analysis.over_limited(_limiter(1.5, 4.5), psr_db=7.0, plr_db=7.0)
    assert analysis.over_limited(_limiter(1.5, 4.5), psr_db=7.0, plr_db=7.0, target_lufs_i=-14.0)


def test_a_loudness_push_too_far_for_a_limiter_is_said():
    agent = _agents.RuleAgent()
    far = agent._loudness_proposal(_ctx([]), _state("lufs_i", -22.0, 14.0, goal=-8.0, unit="LUFS"))
    near = agent._loudness_proposal(_ctx([]), _state("lufs_i", -17.0, 3.0, goal=-14.0, unit="LUFS"))
    assert "into the limiter" in far["rationale"] and "into the limiter" not in near["rationale"]


# ---------- the second batch: Auto Mix ----------

def _master_actions(bundles):
    return [a for a in _actions(bundles) if a.get("stem_id") == "master"]


def test_auto_mix_trims_a_mix_that_sums_over_full_scale_before_the_limiter():
    """Loud sample-pack stems sum to +5 dBTP; with no loudness asked the ceiling limiter took
    all of it as reduction."""
    tags = {"a": [], "b": []}
    stems = {k: _stem(-10.0, -1.0) for k in tags}
    hot = _automix.build_bundles(tags, stems, [], {"lufs_i": -6.0, "true_peak_dbtp": 5.0},
                                 goal_lufs=None)
    trims = [a for a in _master_actions(hot) if a["type"] == "gain"]
    assert trims and -6.0 <= trims[0]["delta_db"] <= -5.0, _master_actions(hot)
    assert [a["type"] for a in _master_actions(hot)][-1] == "limiter"
    calm = _automix.build_bundles(tags, stems, [], {"lufs_i": -18.0, "true_peak_dbtp": -4.0},
                                  goal_lufs=None)
    assert not [a for a in _master_actions(calm) if a["type"] == "gain"]


def test_vocal_forward_lifts_the_lead_not_the_first_name_in_the_alphabet():
    tags = {"adlibs": ["vocal"], "bv": ["vocal"], "lead_vox": ["vocal"]}
    stems = {"adlibs": _stem(-26.0, -12.0), "bv": _stem(-24.0, -12.0),
             "lead_vox": _stem(-16.0, -6.0)}
    actions, _ = _automix._intent_actions("vocal-forward", tags, stems)
    assert [a["stem_id"] for a, _ in actions if a["type"] == "gain"] == ["lead_vox"]


def test_wide_pop_widens_what_can_be_wide_and_leaves_the_lead_in_the_middle():
    tags = {"lead_vox": ["vocal"], "bv": ["vocal"], "pad": ["pad"]}
    stems = {"lead_vox": _stem(-16.0, -6.0, stereo_correlation=1.0),
             "bv": _stem(-24.0, -12.0, stereo_correlation=1.0),
             "pad": _stem(-20.0, -10.0, stereo_correlation=0.4)}
    actions, _ = _automix._intent_actions("wide-pop", tags, stems)
    widths = {a["stem_id"]: a for a, _ in actions if a["type"] == "width"}
    assert "lead_vox" not in widths and set(widths) == {"bv", "pad"}
    assert widths["bv"].get("stereoize") is True         # a mono part has no side to scale
    assert not widths["pad"].get("stereoize")


def test_auto_mix_does_not_high_pass_a_whole_mix_or_a_part_with_real_lows():
    tags = {"beat": ["full_mix"], "organ": ["pad"], "airpad": ["pad"]}
    stems = {"beat": {**_stem(-14.0, -3.0), "band_energy": {"sub": -8.0, "bass": -6.0}},
             "organ": {**_stem(-18.0, -8.0), "band_energy": {"sub": -14.0, "bass": -7.0}},
             "airpad": {**_stem(-18.0, -8.0), "band_energy": {"sub": -22.0, "bass": -30.0}}}
    hpf = {a["stem_id"] for a in _actions(_automix.build_bundles(
        tags, stems, [], {"lufs_i": -18.0}, goal_lufs=None))
        if a["type"] == "eq" and a["band"].get("shape") == "hpf"}
    assert hpf == {"airpad"}, hpf


def test_hats_may_be_high_passed_where_engineers_do(demo_project):
    session = Session(demo_project)
    session.user_edit("stem_a", {"tags": ["cymbal"]})

    def hpf(stem, hz):
        return _propose(session, [{"type": "eq", "stem_id": stem,
                                   "band": {"freq_hz": hz, "gain_db": 0.0, "q": 0.7, "shape": "hpf"}}])
    assert hpf("stem_a", 400.0).accepted                      # hats
    assert not hpf("stem_b", 400.0).accepted                  # control: an untagged part is held at 120


def test_cleanup_does_not_glue_bass_drums_and_vocals_on_one_bus():
    def report(*stems):
        return {"per_stem": [{"stem_id": s, "transient_smear": 0.6} for s in stems]}
    mixed = _automix.build_cleanup_bundles(
        report("bass", "drums", "vocals"), {"bass": ["bass"], "drums": ["kick"], "vocals": ["vocal"]})
    assert not [a for a in _actions(mixed) if a["type"] == "group"], _actions(mixed)
    one = _automix.build_cleanup_bundles(
        report("lead", "bv"), {"lead": ["vocal"], "bv": ["vocal"]})
    assert [a for a in _actions(one) if a["type"] == "group"]      # control: one family is glued


def test_the_second_batch_of_prompt_wording():
    from orchestrator import chat, prompts
    for text in (chat.DAW_SYSTEM_PROMPT, prompts.DAW_PROMPT_WHOLE):
        flat = " ".join(text.split())
        assert "independent L/R (wider)" not in flat and "about two octaves" not in flat
        assert "0.62 to 0.41" not in flat
        assert "no tool here reads them" in flat and "not each band" in flat
        assert "500 Hz on a stem tagged cymbal" in flat


# ---------- the third batch: what the second left ----------

def _stem_reading(lufs, presence, brilliance, low_mid=-12.0):
    return {"lufs_s": lufs, "band_energy": {"presence": presence, "brilliance": brilliance,
                                            "low_mid": low_mid}}


def _composite(readings, metric="harshness_db"):
    agent = _agents.RuleAgent()
    ctx = _ctx([], engine_call=(lambda name, args: readings[args["stem_id"]]) if readings else None)
    ctx.stem_chains = {k: [] for k in (readings or {"a": 0, "b": 0})}
    return agent._composite_proposal(ctx, _state(metric, -6.0, 2.0, max_value=-8.0))["actions"][0]


def test_a_harsh_mix_is_cut_on_the_part_that_carries_it():
    """'Less harsh' cut 2-4 kHz on the whole master, pass after pass - the vocal's words and the
    guitar's bite went with the hats that were the problem."""
    cut = _composite({"hats": _stem_reading(-14.0, -3.0, -4.0),
                      "vocal": _stem_reading(-16.0, -14.0, -20.0),
                      "bass": _stem_reading(-12.0, -30.0, -40.0)})
    assert cut["stem_id"] == "hats" and cut["band"]["gain_db"] < 0, cut


def test_a_harshness_nobody_owns_is_still_cut_on_the_master():
    even = _composite({"a": _stem_reading(-14.0, -8.0, -10.0), "b": _stem_reading(-14.0, -8.0, -10.0)})
    assert even["stem_id"] == "master"
    assert _composite(None)["stem_id"] == "master"            # no way to look: as before


def test_a_run_says_when_the_limiter_is_doing_the_work():
    quiet = {"stems": []}
    notes = _loop.collateral_notes(quiet, quiet, {}, {"limiter_gr_max_db": 9.0, "lufs_i": -8.0},
                                   loudness_asked=True)
    assert any("limiter" in n and "9" in n for n in notes), notes
    assert _loop.collateral_notes(quiet, quiet, {}, {"limiter_gr_max_db": 2.0},
                                  loudness_asked=True) == []


def test_the_real_session_check_is_proven_on_its_own_mutations():
    """`python -m orchestrator.mix_session_check --self-test`: a good session passes; a buried
    lead, a narrowed mix and a hot export each fail the check aimed at them."""
    from orchestrator import mix_session_check
    assert mix_session_check.self_test() == []
