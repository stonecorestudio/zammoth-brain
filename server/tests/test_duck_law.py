"""The Duck's law (2026-10-28, duck_law): a drawn curve fitted to a reference curve-ducker measured
as a black box. These hold the law to what was MEASURED (the numbers below are the reference's, read
off its own output), not to itself:

  * the native header is the table (`DuckLaw.h` is written from it);
  * gain = 1 - mix x (1 - curve) exactly; curve 1 starts at 0.130 and dips to ~0.02;
  * SYNC: the curve on the song's quarters, a 1/4 curve is one beat long, a tempo map moves it;
  * TRACK: a hit restarts the curve, it plays once to its END marker and holds; a 6 ms ease into it;
  * the detector fires on the reference's classes (a kick at -14 dBFS yes, -18 no; a 40 Hz burst
    at 0 dBFS no, 50 Hz yes; 1 kHz at -18 yes; a 150 ms swell no; a re-hit 20 ms later no, 60 ms yes)
    and places a kick's curve on its first sample;
  * BAND: only a one-pole low band ducks, the two halves sum to the input exactly;
  * an old Duck (threshold / depth / attack / release) is read once; a request naming them is refused.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mix_engine import duck_law as D  # noqa: E402

SR = 48000
REPO = Path(__file__).resolve().parents[2]


class _Bpm:
    def __init__(self, bpm: float, first_beat_s: float = 0.0):
        self.bpm, self.b0 = bpm, first_beat_s

    def ppq_at(self, t):
        return (np.asarray(t, dtype=np.float64) - self.b0) * self.bpm / 60.0


def _kick(n: int, at_s, db: float = 0.0, attack_ms: float = 0.0) -> np.ndarray:
    k = np.zeros((n, 2))
    tt = np.arange(int(0.3 * SR)) / SR
    env = np.exp(-tt / 0.12)
    if attack_ms:
        env *= np.minimum(tt / (attack_ms / 1000.0), 1.0)
    hit = 10 ** (db / 20) * np.sin(2 * np.pi * np.cumsum(50 + 100 * np.exp(-tt / 0.03)) / SR) * env
    for a in np.atleast_1d(at_s):
        i = int(a * SR); m = min(hit.size, n - i)
        k[i:i + m] += hit[:m, None]
    return k


def _burst(n: int, at_s: float, hz: float, db: float, attack_ms: float = 0.0) -> np.ndarray:
    k = np.zeros((n, 2))
    tt = np.arange(int(0.15 * SR)) / SR        # the reference's probes: 150 ms, a 10 ms fade-out
    env = np.minimum(tt / (attack_ms / 1000.0), 1.0) if attack_ms else np.ones_like(tt)
    env[-int(0.01 * SR):] *= np.linspace(1.0, 0.0, int(0.01 * SR))
    x = 10 ** (db / 20) * np.sin(2 * np.pi * hz * tt) * env
    i = int(at_s * SR)
    k[i:i + x.size] = x[:, None]
    return k


def _fires(key) -> np.ndarray:
    return D.onsets(key, SR) / SR


def test_the_native_header_is_the_table():
    assert (REPO / "native/Source/DuckLaw.h").read_text() == D.header_text()


def test_the_mix_law_and_curve_one_are_the_reference_s():
    ph = np.linspace(0, 1, 2001)
    c = D.curve_at(1, ph)
    assert c[0] == pytest.approx(0.130, abs=0.002)            # measured 0.1299 at the beat
    assert c.min() == pytest.approx(0.0185, abs=0.004)        # measured 0.0185
    assert D.curve_at(1, [0.5])[0] == pytest.approx(1.0, abs=0.004)
    q = np.arange(SR) / SR * 2.0                              # 120 BPM
    for mix in (0.25, 0.5, 0.75):
        g = D.gain(SR, SR, {"trigger": 0, "curve": 1, "length": 1, "mix": mix}, tempo=_Bpm(120))
        raw = D.curve_at(1, np.mod(q, 1.0))
        assert np.max(np.abs(g - (1 - mix * (1 - raw)))) < 1e-12


def test_sync_runs_on_the_song_s_quarters():
    g = D.gain(2 * SR, SR, {"trigger": 0, "curve": 1, "length": 1, "mix": 1.0}, tempo=_Bpm(120))
    beat = SR // 2                                            # 1/4 at 120 BPM = 0.5 s
    assert np.max(np.abs(g[:beat] - g[beat:2 * beat])) < 1e-9
    g8 = D.gain(SR, SR, {"trigger": 0, "curve": 1, "length": "1/8", "mix": 1.0}, tempo=_Bpm(120))
    assert np.max(np.abs(g8[:SR // 4] - g8[SR // 4:SR // 2])) < 1e-9
    moved = D.gain(SR, SR, {"trigger": 0, "curve": 1, "length": 1, "mix": 1.0}, tempo=_Bpm(120, 0.1))
    i = int(0.1 * SR)
    assert moved[i] == pytest.approx(D.curve_at(1, [0.0])[0], abs=1e-6)   # phase 0 on the first beat


def test_track_plays_once_holds_at_its_end_and_eases_in():
    n = 3 * SR
    key = _kick(n, 1.0)
    hits = D.onsets(key, SR)
    assert len(hits) == 1 and 0 <= hits[0] - SR <= int(0.0005 * SR)   # on the kick (measured: +3 samples)
    g = D.gain(n, SR, {"trigger": 1, "curve": 1, "length": 1, "mix": 1.0}, tempo=_Bpm(120), key=key)
    s = hits[0]
    G = int(round(D.GLIDE_MS * 1e-3 * SR))
    assert np.all(g[:s - G] == 1.0)                           # nothing before the ease
    assert g[s] == pytest.approx(D.curve_at(1, [0.0])[0], abs=1e-9)
    assert g[s - G // 2] == pytest.approx(1 + (g[s] - 1) * ((G // 2 + 1) / G) ** 2, abs=1e-6)
    end = int(s + D.END_PHASE[0] * SR / 2)
    held = D.curve_at(1, [D.END_PHASE[0]])[0]
    assert np.all(np.abs(g[end + 2:] - held) < 1e-12)         # holds at the End marker, no pre-dip
    # curve 10 stops at 0.79 and holds its value there (measured: the reference's End marker)
    g10 = D.gain(n, SR, {"trigger": 1, "curve": 10, "length": 1, "mix": 1.0}, tempo=_Bpm(120), key=key)
    assert g10[-1] == pytest.approx(D.curve_at(10, [0.79])[0], abs=1e-9)


@pytest.mark.parametrize("db,fires", [(0, True), (-12, True), (-14, True), (-18, False), (-30, False)])
def test_a_kick_fires_at_the_reference_s_level(db, fires):
    assert (len(_fires(_kick(SR, 0.3, db))) == 1) is fires


# (40 Hz at 0 dBFS: the reference holds back and the law fires - one of its three named edges)
@pytest.mark.parametrize("hz,db,fires", [(30, 0, False), (50, 0, True), (100, -10, True),
                                         (100, -16, False), (1000, -18, True), (5000, -16, True)])
def test_the_detector_hears_less_of_the_lows(hz, db, fires):
    assert (len(_fires(_burst(SR, 0.3, hz, db))) >= 1) is fires


def test_a_slow_rise_and_a_held_note_do_not_fire():
    # (a 150 Hz swell over 150 ms: the reference holds back and the law fires - a named edge)
    assert len(_fires(_burst(SR, 0.2, 60, -1, attack_ms=100))) == 0
    n = 2 * SR
    key = _kick(n, 0.2) + _burst(n, 0.35, 50, -3)               # a bass note held after the kick
    assert len(_fires(key)) == 1


@pytest.mark.parametrize("gap_s,count", [(0.02, 1), (0.06, 2), (0.12, 2)])
def test_a_re_hit_inside_the_hold_is_one_hit(gap_s, count):
    assert len(_fires(_kick(SR, [0.2, 0.2 + gap_s]))) == count


def test_the_band_split_ducks_the_low_band_alone_and_sums_back():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((SR, 2)) * 0.1
    e = {"trigger": 0, "curve": 10, "length": 1, "mix": 0.0, "band": True, "band_hz": 300.0}
    assert np.max(np.abs(D.apply(x, SR, e, tempo=_Bpm(120)) - x)) < 1e-12   # mix 0: the input
    e = dict(e, mix=1.0)
    y = D.apply(x, SR, e, tempo=_Bpm(120))
    shut = slice(int(0.05 * SR), int(0.2 * SR))                # curve 10 is 0 there
    from scipy.signal import lfilter
    b, a1 = D.band_coeffs(SR, 300.0)
    high = x - lfilter([b, b], [1.0, a1], x, axis=0)
    assert np.max(np.abs(y[shut] - high[shut])) < 1e-6         # curve 10 is ~1e-6 there, not 0
    k = 2 * np.pi * 300.0 / (2 * SR)
    assert b == pytest.approx(k / (1 + k))                    # bilinear, unwarped (measured 1e-7)


def test_an_old_duck_is_read_once_and_a_request_naming_its_knobs_is_refused():
    old = {"type": "duck", "key_stem_id": "kick", "threshold_db": -30.0, "depth_db": 6.0,
           "attack_ms": 5.0, "release_ms": 150.0}
    new = D.from_legacy(old)
    assert new["trigger"] == 1 and new["curve"] == 1 and new["length"] == 1
    assert new["mix"] == pytest.approx(1 - 10 ** (-6 / 20), abs=1e-3)
    assert not any(k in new for k in ("threshold_db", "depth_db", "attack_ms", "release_ms"))
    from mix_engine.models import ProposalRequest, SidechainDuckAction
    stored = SidechainDuckAction.model_validate({**old, "stem_id": "bass"}, context={"stored": True})
    assert stored.trigger == 1 and stored.mix == pytest.approx(new["mix"])
    with pytest.raises(Exception, match="threshold_db"):
        ProposalRequest(rationale="x", predicted_impact=[], actions=[{**old, "stem_id": "bass"}])


def test_an_old_session_s_ducks_are_converted_at_load(demo_project):
    import json
    from mix_engine.session import Session
    s = Session(demo_project)
    stem = next(iter(s.stems))
    state = json.loads((demo_project / "session.json").read_text())
    state["state_version"] = 4
    state["stems"][stem]["chain"] = [{"type": "duck", "key_stem_id": "", "threshold_db": -30.0,
                                      "depth_db": 12.0, "attack_ms": 5.0, "release_ms": 150.0}]
    (demo_project / "session.json").write_text(json.dumps(state))
    duck = [e for e in Session(demo_project).stems[stem]["chain"] if e["type"] == "duck"][0]
    assert duck["trigger"] == 1 and duck["mix"] == pytest.approx(0.749, abs=1e-3)
    assert "threshold_db" not in duck


def test_words_are_taken_for_the_index_knobs():
    from mix_engine.models import SidechainDuckAction
    a = SidechainDuckAction(stem_id="bass", key_stem_id="kick", trigger="track", length="1/2",
                            curve="Sub 1")
    assert (a.trigger, a.length, a.curve) == (1, 2, 9)
    with pytest.raises(Exception):
        SidechainDuckAction(stem_id="bass", trigger="sidechain")


def test_the_agent_may_cut_twelve_db_at_most():
    from mix_engine.guardrails import validate_proposal
    from mix_engine.models import ProposalRequest
    req = ProposalRequest(rationale="x", predicted_impact=[], actions=[
        {"type": "duck", "stem_id": "bass", "key_stem_id": "kick", "trigger": "track", "mix": 0.9},
        {"type": "duck", "stem_id": "pad", "trigger": "track", "mix": 0.5}])
    rules = [v.rule for v in validate_proposal(req, {})]
    assert "duck_depth_max_db" in rules and "duck_track_needs_key" in rules


# ------------------------------------------------------------------ the native parity fixture
# ONE fixture, `server/tests/data/duck_law_parity.json`, carries Python's answers (the left channel)
# on a signal and a key both engines make from one formula; the console leg `--duck-law` asserts
# MonitorChain renders them (shifted by its look-ahead). A law change on purpose:
# `python server/tests/test_duck_law.py --write`, then the leg.
FIX_SR = 24000
FIX_N = 12000
FIXTURE = Path(__file__).resolve().parent / "data" / "duck_law_parity.json"
FIX_KICKS = (0.05, 0.2, 0.218, 0.35)          # the third inside the hold: one hit
FIX_CASES = [
    {"type": "duck", "trigger": 0, "curve": 1, "length": 1, "mix": 1.0},
    {"type": "duck", "trigger": 0, "curve": 15, "length": 0, "mix": 0.8, "band": True, "band_hz": 200.0},
    {"type": "duck", "trigger": 1, "curve": 1, "length": 1, "mix": 1.0, "key_stem_id": "k"},
    {"type": "duck", "trigger": 1, "curve": 10, "length": 0, "mix": 0.7, "band": True, "band_hz": 300.0,
     "key_stem_id": "k"},
    {"type": "duck", "threshold_db": -30.0, "depth_db": 9.0, "attack_ms": 5.0, "release_ms": 150.0,
     "key_stem_id": "k"},                     # saved before the law: read as from_legacy reads it
]


def fixture_signal(sr: int = FIX_SR, n: int = FIX_N):
    """left = 0.5 sin(2 pi 220 t)(0.6 + 0.4 sin(2 pi 3 t)); right = 0.4 sin(2 pi 330 t + 0.4);
    key = kicks at FIX_KICKS: 0.9 sin(2 pi (50 u + 3 (1 - e^-u/0.03))) e^-u/0.12 for u < 0.2 s."""
    t = np.arange(n) / sr
    x = np.stack([0.5 * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)),
                  0.4 * np.sin(2 * np.pi * 330 * t + 0.4)], axis=1).astype(np.float32).astype(np.float64)
    k = np.zeros(n)
    for at in FIX_KICKS:
        i0 = int(round(at * sr))
        u = np.arange(min(int(round(0.2 * sr)), n - i0)) / sr
        k[i0:i0 + u.size] += 0.9 * np.sin(2 * np.pi * (50 * u + 3 * (1 - np.exp(-u / 0.03)))) * np.exp(-u / 0.12)
    k = k.astype(np.float32).astype(np.float64)
    return x, np.stack([k, k], axis=1)


def _fixture_hits(entry: dict, sr: int = FIX_SR, n: int = FIX_N) -> dict:
    """What the brain attaches to a TRACK duck for the DAW (daw_bridge.attach_duck_hits): the hits
    in seconds and the song's quarter there (no tempo map: 120, so quarters = 2 t)."""
    e = D.from_legacy(entry) if D.is_legacy(entry) else entry
    if D.values(e)["trigger"] != 1:
        return {}
    _, key = fixture_signal(sr, n)
    secs = [float(i) / sr for i in D.onsets(key, sr)]
    return {"hits_s": secs, "hits_q": [2.0 * t for t in secs]}


def _fixture_out(entry: dict, sr: int = FIX_SR, n: int = FIX_N) -> list:
    x, key = fixture_signal(sr, n)
    e = D.from_legacy(entry) if D.is_legacy(entry) else entry
    y = D.apply(x, sr, e, tempo=None, key=key if D.values(e)["trigger"] == 1 else None)
    return [round(float(v), 7) for v in y[:, 0]]


#: ...and two cases at 48 kHz (the review of 2026-10-28: every case was at one rate, so nothing held the
#: law's dependence on it - the detector's high-pass, the hold, the ease). Each names its rate and length.
FIX_RATE_CASES = [(48000, 24000, FIX_CASES[1]), (48000, 24000, FIX_CASES[2])]


def _rate_case(sr: int, n: int, e: dict) -> dict:
    return {"entry": e, "sr": sr, "n": n, "out": _fixture_out(e, sr, n), **_fixture_hits(e, sr, n)}


def test_the_parity_fixture_is_python_s_answer():
    data = json.loads(FIXTURE.read_text())
    assert data["sr"] == FIX_SR and data["n"] == FIX_N and list(data["kicks"]) == list(FIX_KICKS)
    base = [c for c in data["cases"] if "sr" not in c]
    assert [c["entry"] for c in base] == FIX_CASES
    for c in data["cases"]:
        sr, n = c.get("sr", FIX_SR), c.get("n", FIX_N)
        assert np.max(np.abs(np.array(c["out"]) - np.array(_fixture_out(c["entry"], sr, n)))) < 2e-7
        assert {k: c[k] for k in ("hits_s", "hits_q") if k in c} == _fixture_hits(c["entry"], sr, n)
    assert [(c["sr"], c["n"], c["entry"]) for c in data["cases"] if "sr" in c] == FIX_RATE_CASES


if __name__ == "__main__" and "--write" in sys.argv:
    FIXTURE.write_text(json.dumps({"sr": FIX_SR, "n": FIX_N, "kicks": list(FIX_KICKS),
                                   "signal": fixture_signal.__doc__,
                                   "cases": [{"entry": e, "out": _fixture_out(e), **_fixture_hits(e)}
                                             for e in FIX_CASES]
                                            + [_rate_case(sr, n, e) for sr, n, e in FIX_RATE_CASES]}))
    print(f"wrote {FIXTURE}")


def test_the_brain_hands_a_track_duck_its_triggers_hits(tmp_path):
    """The DAW's node plays a TRACK duck's hits by the playhead (no latency: a sidechain arrives delayed
    by whatever a node reports, so a look-ahead cancelled itself - the export oracle, 2026-10-28). The
    brain finds them where the offline render does: duck_law.onsets on the trigger, rendered without
    its own keyed devices. SYNC, and a trigger the brain cannot hear, get none."""
    import soundfile as sf
    from mix_engine import dsp
    from mix_engine.session import Session
    from orchestrator.daw_bridge import resolve_chains
    project = tmp_path / "hits"
    project.mkdir()
    kick = _kick(SR, [0.1, 0.35, 0.6])[:, 0]
    sf.write(project / "kick.wav", kick.astype(np.float32), SR)
    sf.write(project / "bass.wav", (0.2 * np.sin(2 * np.pi * 55 * np.arange(SR) / SR)).astype(np.float32), SR)
    s = Session(project)
    s.set_duck("bass", "kick")
    s.stems["bass"]["chain"].append({"type": "duck", "trigger": 0, "key_stem_id": "kick"})
    chains = resolve_chains(s._state, 0.0, None, sample_rate=s.project_sample_rate(), session=s)
    track, sync = [e for e in chains["bass"]["entries"] if e["type"] == "duck"]
    key, _ = dsp.render_stem(s, "kick", exclude_ducks=True)
    want = [i / SR for i in D.onsets(dsp._key_audio(key, len(key), SR), SR)]
    assert len(want) == 3 and track["hits_s"] == pytest.approx(want, abs=1e-12)
    assert track["hits_q"] == pytest.approx([2 * t for t in want], abs=1e-9)   # no tempo map: 120
    assert "hits_s" not in sync
    s.user_edit("kick", {"excluded": True})                   # a trigger the render cannot hear
    chains = resolve_chains(s._state, 0.0, None, sample_rate=s.project_sample_rate(), session=s)
    assert all("hits_s" not in e for e in chains["bass"]["entries"] if e["type"] == "duck")
