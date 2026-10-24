"""Reader P3s, batch M22 - device faces and the native chain (2026-10-07).

The behaviour is driven by the console legs in `daw/Source/AuditReaderP3LegsM22.h`. This file
holds what a leg cannot: rules about the SOURCE that a later edit could quietly undo - above
all C6.R12, whose harm is a WAIT on the audio thread, which no leg can see.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NODE = (ROOT / "daw" / "Source" / "MonitorChainPlugin.h").read_text(encoding="utf-8")
SIMPLER = (ROOT / "daw" / "Source" / "SimplerPlugin.h").read_text(encoding="utf-8")
CHAIN_H = (ROOT / "native" / "Source" / "MonitorChain.h").read_text(encoding="utf-8")
CHAIN = (ROOT / "native" / "Source" / "MonitorChain.cpp").read_text(encoding="utf-8")
GRAPHS = (ROOT / "native" / "Source" / "DeviceGraphs.h").read_text(encoding="utf-8")
APP = (ROOT / "daw" / "Source" / "App.cpp").read_text(encoding="utf-8")
MANUAL = (ROOT / "docs" / "17-user-manual.md").read_text(encoding="utf-8")


def _body(text: str, start: str, end: str = "\n        }\n") -> str:
    at = text.index(start)
    return text[at: text.index(end, at)]


def _code(text: str) -> str:
    """The text without its // comments (a rule about calls is not a rule about prose)."""
    return "\n".join(line.split("//")[0] for line in text.splitlines())


def test_the_red_run_switch_is_gone():
    """The batch was shown red and green on one build through a temporary switch
    (ZAMMOTH_M22_OLD, native/Source/M22Old.h): every fixed part kept its old path behind it.
    Nothing of it may be left in the tree."""
    assert not (ROOT / "native" / "Source" / "M22Old.h").exists(), "the temporary header is still there"
    for folder in ("daw/Source", "native/Source"):
        for path in sorted((ROOT / folder).glob("*")):
            if path.suffix not in (".h", ".cpp", ".mm"):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            assert "m22old" not in text and "M22Old" not in text and "ZAMMOTH_M22_OLD" not in text, \
                f"{path.name} still takes the batch's old path behind the switch"


def test_parameter_changed_takes_no_mutex_to_ask_which_thread_it_is():
    """C6.R12 a. DeviceScalarParameter::parameterChanged runs on the AUDIO thread for a rack
    macro or a modifier. juce::MessageManager::existsAndIsCurrentThread() locks a std::mutex
    (messageThreadIdMutex, no priority inheritance) on every call, and it was called twice per
    delivery. The node compares thread ids against one read once."""
    body = _code(_body(NODE, "void parameterChanged (float newValue, bool byAutomation) override", "\n    public:\n"))
    assert "existsAndIsCurrentThread" not in body and "isThisTheMessageThread" not in body, \
        "parameterChanged asks JUCE which thread it is on: that call locks a mutex"
    assert body.count("onMessageThread()") >= 2, "both thread tests go through the node's own compare"
    helper = _code(_body(NODE, "static bool onMessageThread() noexcept", "\n    }\n"))
    assert "getCurrentThreadId()" in helper and "MessageManager" not in helper, \
        "onMessageThread is a thread-id compare, nothing that can lock"
    ctor = _body(NODE, "explicit MonitorChainPlugin (PluginCreationInfo info)", "\n    }\n")
    assert "messageThreadIdOnce()" in ctor, \
        "the id is read once in the constructor, so its first read is never on the audio thread"


def test_the_rt_comment_no_longer_claims_a_pure_atomic_store():
    """C6.R12 b. The comment said "the RT path is now a pure atomic store ... no lock, no alloc,
    no post" while the engine's own caller posts an AsyncUpdater message one frame up. Our half
    is lock-free; the comment says whose the post is."""
    body = _body(NODE, "void parameterChanged (float newValue, bool byAutomation) override", "\n    public:\n")
    assert "the RT path is now a pure atomic store, and" not in body
    assert "triggerAsyncUpdate" in body and "engine patch" in body, \
        "the comment names the engine's post and where removing it would belong"


def test_the_simpler_takes_its_block_lock_only_when_the_slice_table_changed():
    """C6.R12 c. syncAtomicsFromCached runs on every value-tree change (every knob tick). It
    took `lock` - the CriticalSection applyToBuffer holds for a whole block - each time, to copy
    a slice table that was the same. The lock is inside the "it differs" branch."""
    body = _code(_body(SIMPLER, "    void syncAtomicsFromCached()", "\n    }\n"))
    lock_at = body.index("const juce::ScopedLock sl (lock);")
    guard = body.rfind("if (! same)", 0, lock_at)
    assert guard >= 0 and "}" not in body[guard:lock_at], \
        "the block lock is taken outside the table-changed branch"
    assert body.count("ScopedLock") == 1


def test_a_synced_modifier_counts_in_the_tempo_maps_beats():
    """S2.R18. evalModifier was handed `t` and one cached tempo and used t * bpm / 60 for a
    synced rate: an export ran every synced modifier at the tempo under the PARKED playhead.
    It is handed the block's BEAT, read from the tempo map's RT-safe cursor."""
    sig = re.search(r"static float evalModifier \(Modifier& m, double t, double (\w+), float env01, double blockSeconds\)", NODE)
    assert sig and sig.group(1) == "beats", "evalModifier takes the block's beat"
    body = _code(_body(NODE, "static float evalModifier (Modifier& m, double t, double beats", "\n    Modifier* modifierFor"))
    assert "bpm" not in body, "a synced branch still multiplies seconds by a tempo"
    assert body.count("beats / m.rateBeats") == 3, "LFO, random and step all count in beats"
    block = _code(_body(NODE, "    void applyToBuffer (const PluginRenderContext& rc) override", "\n        juce::AudioBuffer<float> view"))
    assert "tempoPos->set" in block and "tempoPos->getBeats()" in block
    init = _code(_body(NODE, "    void initialise (const PluginInitialisationInfo& info) override", "\n    void deinitialise"))
    assert "tempo::Sequence::Position" in init


def test_the_simplers_synced_lfo_reads_the_blocks_tempo():
    """The lead with S2.R18: SimplerPlugin read its own cachedBpm for the synced LFO's rate."""
    block = _code(_body(SIMPLER, "    void applyToBuffer (const PluginRenderContext& fc) override", "\n        // deterministic LFO evaluation"))
    assert "tempoPos->set (fc.editTime.getStart())" in block and "tempoPos->getTempo()" in block


def test_a_render_is_told_it_is_one_and_every_audition_asks():
    """C6.R10. What is held under a mouse to listen is not played into a render: the node hands
    the chain rc.isRendering each block, and the four auditions read the block's flag."""
    block = _code(_body(NODE, "    void applyToBuffer (const PluginRenderContext& rc) override", "\n        preFaderTrim->store"))
    assert block.index("chain.setOfflineRender (rc.isRendering);") < block.index("chain.process (sub"), \
        "the chain is told before the block is processed"
    code = _code(CHAIN)
    assert "auditions = ! offlineRender.load" in code
    for needle in ("proc && auditions && soloOn.load", "c.delta && auditions", "de.listen && auditions",
                   "active.wid.listenMid && auditions", "active.wid.listenSide && auditions"):
        assert needle in code, f"an audition no longer asks whether this block is a render: {needle}"
    # ...and nothing reads the raw flags past those lines
    assert len(re.findall(r"\bc\.delta\b", code)) == 1 and len(re.findall(r"\bde\.listen\b", code)) == 1
    setter = _code(_body(CHAIN, "void MonitorChain::setBandSolo (bool on, double freqHz, double q)", "\n}\n"))
    assert "soloAppliedF" not in setter, "the UI thread writes a float the audio thread owns"


def test_a_new_dsp_memory_is_cleared_in_both_resets():
    """M4's rule, for this batch's two memories: the clipper's true-peak history and the solo's
    edge flag are cleared in prepare() AND in resetDspState()."""
    prepare = _body(CHAIN, "void MonitorChain::prepare (double sampleRate", "\n}\n")
    reset = _body(CHAIN, "void MonitorChain::resetDspState()", "\n}\n")
    for part in (prepare, reset):
        assert "std::memset (clipTpHist, 0, sizeof (clipTpHist));" in part
        assert "soloWasOn = false;" in part


def test_prepare_allocates_no_wet_ring_and_no_convolver_by_itself():
    """C6.R11. prepare() used to setSize three ring buffers for all fifteen wet slots and the
    constructor built fifteen convolvers and a loader queue (a thread). Rings and convolvers are
    made for the slots that hold a device, where the chain is applied."""
    code = _code(CHAIN)
    assert "srDelayRing" not in code and "srPreRing" not in code and "srCombRing" not in code
    assert "kMaxSends * kMaxChans" not in code, "a ring is sized for every slot again"
    ctor = CHAIN[CHAIN.index("MonitorChain::MonitorChain() : Thread"):].split("\n", 1)[0]
    assert ctor.rstrip().endswith("{}"), "the constructor builds something again (a convolver, the loader queue)"
    assert "std::unique_ptr<juce::dsp::ConvolutionMessageQueue> convQueue;" in CHAIN_H
    apply = _code(_body(CHAIN, "    // What this chain needs BUILT and the chain does not hold yet", "\n    if (beforeFirstBlock)"))
    made, lock, adopt = apply.index("buildMissingRings (ns, ringsMade);"), apply.index("ScopedLockType sl (stateLock);"), apply.index("adoptRings (ringsMade);")
    assert made < lock < adopt, "rings are built before the lock and taken in under it"
    assert apply.index("convMade[s] = makeConvolver();") < lock < apply.index("convolvers[s] = std::move (convMade[s]);")
    # ...and the slot's spare (the IR audit's ring-out, 2026-10-19), the same way
    assert apply.index("spareMade[s] = makeConvolver();") < lock < apply.index("convSpare[s]  = std::move (spareMade[s]);")
    # the two live-param branches no longer carry a copy of the ring's length
    assert "(sr * 2.0) + 2" not in code and "(sr * 0.25) + 2" not in code
    assert "delayRingLen (sr)" in code and "preRingLen (sr)" in code


def test_one_thread_drains_every_nodes_meters():
    """C6.R11 c. MeterFeed was a juce::Thread: one per node."""
    feed = _body(NODE, "    struct MeterFeed\n", "\n    };\n")
    assert "juce::Thread" not in _code(feed) and "startThread" not in feed
    assert "juce::SharedResourcePointer<MeterDrain> drain;" in feed
    assert "struct MeterDrain : private juce::Thread" in NODE


def test_the_face_and_the_agent_share_one_band_tilt():
    """E3.R10. Two least-squares fits under one word. Both readers call LiveMeters'."""
    cmd = _body(APP, "            case dawcmd::trackSpectrogram:", "\n            default: break;")
    assert "LiveMeters::bandTiltDbPerOct (bandMean)" in cmd and '"band_tilt_db_per_oct"' in cmd
    assert "tiltNum" not in cmd and "tiltDen" not in cmd, "track.spectrogram fits its own line again"
    dock = (ROOT / "daw" / "Source" / "DeviceChainView.h").read_text(encoding="utf-8")
    assert "LiveMeters::bandTiltOfRows (sgBands.data(), rows, kSgTiltRows, means)" in dock
    assert '"BAND TILT"' in GRAPHS and '"band tilt "' in GRAPHS
    assert "the same number the agent gets from track.spectrogram" not in GRAPHS
    assert "BAND TILT" in MANUAL


def test_the_overdrive_face_takes_the_nodes_rate():
    """E3.R6. odTiltDb / odBpDb drew analog prototypes; they evaluate the digital filters the
    chain builds, at the node's sample rate."""
    assert re.search(r"inline float odTiltDb \(double f, float tone, double sr\)", GRAPHS)
    assert re.search(r"inline float odBpDb \(double f, float f0, float q, double sr\)", GRAPHS)
    assert not re.search(r"odTiltDb \([^()]*\(float\) tone\)\)", GRAPHS), "a call site draws the tilt without a rate"
    dock = (ROOT / "daw" / "Source" / "DeviceChainView.h").read_text(encoding="utf-8")
    assert "g->setSampleRate (m.getSampleRate());" in dock


def test_the_words_s2_r17_found_are_gone():
    """S2.R17. Four sentences the code no longer keeps."""
    assert "leave a little space after the last clip" not in MANUAL
    assert "The bounce currently ends at the" not in MANUAL
    strip = (ROOT / "daw" / "Source" / "ChannelStrip.h").read_text(encoding="utf-8")
    assert "the render is playback-driven" not in strip
    assert "the engineer's export is still rendering" not in APP
