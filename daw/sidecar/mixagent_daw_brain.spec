# PyInstaller spec for the Mix Agent DAW brain (docs/10-daw.md, Milestone B).
# One self-contained binary: engine + orchestrator + the NDJSON socket server
# (orchestrator/daw_server.py), no system Python. Built by build.sh, which drops
# the output next to the DAW app (the C++ resolveBrainCommand finds it there, or
# via MIXAGENT_DAW_BRAIN).
#
# Differs from plugin/sidecar (the gateway): the DAW talks a raw asyncio socket,
# NOT uvicorn/websockets over HTTP, and the UI is native — so no uvicorn/websockets
# hidden imports and no web/dist data bundle. The heavy engine deps + the offline
# render path (Pedalboard, soundfile, pyloudnorm, numpy/scipy) and onnxruntime
# (lazy, separation) are the same.
from pathlib import Path

REPO = Path(SPECPATH).resolve().parents[1]
# The source stamp build.sh wrote for THIS freeze (stamp.sh). Bundled at the archive root,
# where daw_server.brain_stamp() reads it and `hello` reports it — package.sh then proves the
# binary inside the app is the one built from the current sources (2026-09-22 launch review,
# R1). A freeze run without build.sh has no stamp and the brain says "unstamped", which the
# packaging gate refuses.
STAMP = Path(SPECPATH).resolve() / "stamp" / "brain_stamp.txt"

a = Analysis(
    ["brain_entry.py"],
    pathex=[str(REPO), str(REPO / "server")],
    binaries=[],
    datas=[
        # Bundle the Spleeter ONNX models so the .app separates offline.
        # MIX_AGENT_DATA_DIR/models (or the app-support dir) still wins, for a
        # post-ship model update without a rebuild.
        (str(REPO / "server" / "mix_engine" / "models"), "mix_engine/models"),
    ] + ([(str(STAMP), ".")] if STAMP.is_file() else []),
    hiddenimports=[
        # onnxruntime is imported lazily in separation.py, so static analysis
        # misses it — pull it in explicitly (its hook then bundles the native libs).
        "onnxruntime",
        # Same lazy-import story for the htdemucs separator (2026-08-13): the
        # product path since the demucs-onnx swap; weights download at runtime,
        # so only the (small) package + its hub client ride in the binary.
        "demucs_onnx",
        "huggingface_hub",
        "soxr",
    ],
    excludes=[
        # The knowledge BOOK never ships in the binary (2026-08-08, "keep the
        # knowledge book server-side"): the shipped sidecar fetches it from the
        # control plane per session (orchestrator/knowledge.py), and GPLv3
        # source obligations cover what is IN the binary — so the private
        # content module stays out of both the bundle and the public repo cut.
        "orchestrator.knowledge_content",
        # In requirements but never imported by the engine; excluding them keeps
        # numba/llvmlite out of the bundle (mirrors plugin/sidecar).
        "librosa",
        "numba",
        "llvmlite",
        "matplotlib",
        "tkinter",
        "IPython",
        "PIL",
        # The native DAW has no webview + no HTTP gateway — drop the web server stack.
        "uvicorn",
        "websockets",
        # Dev-only subscription backend (claude-agent-sdk); production uses the API.
        "claude_agent_sdk",
        # Test and typing tools (2026-10-07, reader S7.R15). No module of the brain imports
        # them; PyInstaller followed a guarded import inside a library to each one - anyio's
        # pytest plugin and fsspec's conftest (pytest), pydantic's hypothesis and mypy plugins,
        # the hub's optional OAuth routes (fastapi) - and froze 71 + 89 + 38 modules and mypy's
        # compiled extension into the shipped binary. NOT in this list, though a reader's
        # sketch had them: starlette (mcp's FastMCP server imports it at module top), rich and
        # pygments (its logging), orchestrator.benchmark (the brain imported its in-memory
        # engine from it until that moved to orchestrator/engine_client.py; an exclude is still
        # owed its two proofs before it is added). test_reader_p3_sh2.py starts a brain session with every name below made
        # unimportable, so an exclude the brain needs fails in the pytest, not in the app.
        "pytest",
        "_pytest",
        "hypothesis",
        "mypy",
        "fastapi",
    ],
    noarchive=False,
)

# What third-party Python this freeze holds, and under which licences: written from the
# analysis itself (bundle_notices.py), never from a list kept by hand, and bundled at the
# archive root beside the stamp. verify_bundled_brain.py reads the manifest back and refuses a
# brain that carries test tooling or code no distribution accounts for.
import sys
sys.path.insert(0, str(Path(SPECPATH).resolve()))
from bundle_notices import for_analysis
a.datas += for_analysis(a, Path(SPECPATH).resolve() / "stamp",
                        own_roots=[REPO / "orchestrator", REPO / "server", Path(SPECPATH).resolve()])

pyz = PYZ(a.pure)

# ONEDIR, not onefile (2026-09-22 launch review, R2). The onefile binary decompressed ~240 MB
# into $TMPDIR on EVERY launch: 20-34 s to the handshake against the app's 25 s wait, so a
# first launch could time out, and a force-killed brain left the 240 MB behind in the temp dir.
# The same brain unfrozen handshakes in 0.6 s — the time was all extraction. Onedir ships the
# files already unpacked: dist/mixagent_daw_brain/{mixagent_daw_brain, _internal/}, which
# package.sh copies whole into the app and signs file by file (notarization wants every Mach-O
# signed, and a onefile archive hid its libraries from both).
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="mixagent_daw_brain",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,          # prints the MIXAGENT_LISTENING handshake on stdout
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="mixagent_daw_brain",
)
