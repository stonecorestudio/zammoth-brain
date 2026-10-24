import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = REPO_ROOT / "test_projects" / "two_stem_demo"
THREE_SECTION_DIR = REPO_ROOT / "test_projects" / "three_section_demo"
SUNO_DIR = REPO_ROOT / "test_projects" / "suno_demo"

# the orchestrator package lives at the repo root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---- Run what the diff can reach (2026-10-04; the rule is the quality-gates skill's first section) ----
#
# DISCONTINUED PRODUCT LINES ARE NOT GATES. The VST/AU plug-in was discontinued 2026-09-07 and the
# Tauri desktop shell archived 2026-08-18; test_phase39_vst.py and test_phase4_sidecar*.py (40 tests
# of their gateway surface and sidecar process) ran in every pytest and CI job until they were
# deleted on 2026-10-04 (James). They are in git history; do not write new ones.

# A C++-ONLY CHANGE CAN ONLY MOVE THE TESTS THAT READ THE C++ (30 of 150 files when this was written):
#     pytest -m reads_cpp
# The mark is derived from each test module's own text, so a new source-scan test is in it by itself.
_CPP_SOURCE = re.compile(r'daw/Source|native/Source|"Source"|App\.cpp|DawCommands\.h|DawSession\.h')
_reads_cpp: dict[str, bool] = {}


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "reads_cpp: the module reads the DAW's C++ sources - the only tests a C++-only change can move")
    config.addinivalue_line(
        "markers", "slow: renders audio for half a minute or more - leave out of a proportionate run with -m 'not slow'")


def pytest_collection_modifyitems(config, items):
    for item in items:
        path = str(item.fspath)
        if path not in _reads_cpp:
            try:
                _reads_cpp[path] = bool(_CPP_SOURCE.search(Path(path).read_text(encoding="utf-8")))
            except OSError:
                _reads_cpp[path] = False
        if _reads_cpp[path]:
            item.add_marker(pytest.mark.reads_cpp)


# ---- The whole suite in parallel: `pytest -n auto` (pytest-xdist) ----
#
# Measured on the build Mac (4 performance + 4 efficiency cores, 16 GB), the whole suite:
# one process 12 min, 4 workers 5.6, 6 workers about 5, 8 workers 6.5. The work is one core's
# worth of numpy per test and several hundred MB per worker, so a worker on every logical
# core is slower than one on each fast core and half the slow ones - which is what `auto`
# means here. `-n <number>` is taken as given.

def _fast_and_slow_cores() -> tuple[int, int] | None:
    """(performance, efficiency) cores on Apple silicon, else None."""
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu",
                              "hw.perflevel1.physicalcpu"],
                             capture_output=True, text=True, timeout=5, check=False).stdout.split()
        return int(out[0]), int(out[1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


@pytest.hookimpl(optionalhook=True)          # the hook is xdist's: absent without it
def pytest_xdist_auto_num_workers(config):
    cores = _fast_and_slow_cores()
    if cores is None:
        return None                          # xdist's own count
    return max(1, cores[0] + cores[1] // 2)


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config):
    """`-n` with no `--dist` shares the tests out by work stealing. xdist's default deals them
    in fixed batches, and here a handful of tests are a minute each among four thousand that
    are milliseconds: a worker dealt two of the long ones was still running when the others
    had nothing left (6 workers: 5 min 57 s dealt; 4 min 49 s and 5 min 17 s stolen). A `--dist` on the
    command line is taken as given."""
    if getattr(config.option, "numprocesses", None) and getattr(config.option, "dist", "no") == "no":
        config.option.dist = "worksteal"


@pytest.fixture()
def demo_project(tmp_path: Path) -> Path:
    """A throwaway copy of the two-stem demo project (stems are gitignored,
    so generate them on first use)."""
    if not (DEMO_DIR / "stem_a.wav").exists():
        subprocess.run([sys.executable, str(DEMO_DIR / "generate_stems.py")], check=True)
    project = tmp_path / "two_stem_demo"
    project.mkdir()
    for wav in DEMO_DIR.glob("*.wav"):
        shutil.copy(wav, project / wav.name)
    return project


@pytest.fixture()
def three_section_project(tmp_path: Path) -> Path:
    """A throwaway copy of the three-section demo (Phase 5: sections + rides)."""
    if not (THREE_SECTION_DIR / "stem_a.wav").exists():
        subprocess.run([sys.executable,
                        str(THREE_SECTION_DIR / "generate_stems.py")], check=True)
    project = tmp_path / "three_section_demo"
    project.mkdir()
    for wav in THREE_SECTION_DIR.glob("*.wav"):
        shutil.copy(wav, project / wav.name)
    return project


@pytest.fixture()
def suno_project(tmp_path: Path) -> Path:
    """Phase 29: a synthetic AI-separated stem set with injected artifacts
    (HF shimmer, transients, a wide loudness spread)."""
    if not (SUNO_DIR / "vocals.wav").exists():
        subprocess.run([sys.executable, str(SUNO_DIR / "generate_stems.py")],
                       check=True)
    project = tmp_path / "suno_demo"
    project.mkdir()
    for wav in SUNO_DIR.glob("*.wav"):
        shutil.copy(wav, project / wav.name)
    return project


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path: Path, monkeypatch):
    """Phase 13: the cross-project calibration store lives in app support;
    tests must never read or write the real one."""
    monkeypatch.setenv("MIX_AGENT_DATA_DIR", str(tmp_path / "appdata"))


@pytest.fixture(autouse=True)
def trusted_test_licence_key(monkeypatch):
    """The brain believes only a signed licence answer (2026-10-20): every test trusts the test
    key in `licence_signing.py`, and a test that stages an answer signs it there."""
    import licence_signing
    monkeypatch.setenv("MIX_AGENT_LICENSE_KEYS", licence_signing.KEYS_ENV_VALUE)


@pytest.fixture(autouse=True)
def no_login_keychain(monkeypatch):
    """No test reads or writes this machine's login keychain (reader S8·R15).

    Two doors led there. `api_key._store` defaulted to the real keychain, so every test that
    reached `api_key()` / `api_key_state()` / `gate_applies()` ran `security
    find-generic-password` against the developer's own - and `llm_available() is False` failed
    on a Mac that holds a key. And anything that builds a real `Keychain` (a `DeviceLogin`
    given none) reached the tool directly. So: the key store is a memory one for every test,
    and `device_auth._security` - the one call to the tool - is answered "not found" here.
    A test that FAKES `subprocess.run` (the keychain's own tests) is let through to its fake.
    """
    from orchestrator import api_key, device_auth

    real_run = subprocess.run
    real_security = device_auth._security

    def guard(argv, **kwargs):
        if subprocess.run is not real_run:      # the test put a fake tool there: use it
            return real_security(argv, **kwargs)
        guard.calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 44, "", "")    # 44 = item not found

    guard.calls = []
    guard.is_test_guard = True
    monkeypatch.setattr(device_auth, "_security", guard)
    # ...and the third door (2026-10-23): a brain signed with a Developer ID keeps its secrets
    # through Security.framework, on the login keychain, with no `security` call to guard. A
    # test that plays a frozen brain would have the brain ask its OWN signature - which in a
    # test is Python's, signed by a real team. So no test brain is ever a signed one; the tests
    # of the framework path hand it a throwaway keychain themselves (test_signed_keychain.py).
    monkeypatch.setattr(device_auth, "_signed_team", None)
    api_key.use_store(api_key.memory_store())
    yield
    api_key.use_store(None)


@pytest.fixture(autouse=True, scope="session")
def test_client_speaks_as_this_machine():
    """Starlette's TestClient says `Host: testserver`; the gateway answers only a request
    addressed to this machine (reader S9·R16). The suite's client says 127.0.0.1 - the
    product's host list never learns a test's name."""
    try:
        from starlette import testclient
    except ImportError:
        yield
        return
    from urllib.parse import urljoin

    original = testclient.TestClient.__init__
    original_ws = testclient.TestClient.websocket_connect

    def init(self, app, base_url="http://127.0.0.1:8765", *args, **kwargs):
        original(self, app, base_url, *args, **kwargs)

    def websocket_connect(self, url, *args, **kwargs):
        # Starlette joins the path onto a hard-coded "ws://testserver"; an absolute URL wins
        return original_ws(self, urljoin("ws://127.0.0.1:8765", url), *args, **kwargs)

    testclient.TestClient.__init__ = init
    testclient.TestClient.websocket_connect = websocket_connect
    yield
    testclient.TestClient.__init__ = original
    testclient.TestClient.websocket_connect = original_ws


@pytest.fixture(autouse=True)
def reset_render_cache():
    """The persistent stem-render cache is process-global state the gateway
    enables at project open — reset it around every test so engine-pure
    tests keep their exact render counts and nothing leaks across tests."""
    from mix_engine import dsp

    dsp.configure_render_cache(0)
    dsp.RENDER_CACHE.clear()
    yield
    dsp.configure_render_cache(0)
    dsp.RENDER_CACHE.clear()


@pytest.fixture
def odd_ladders(monkeypatch):
    """Two models the picker no longer has: one with no `xhigh`, one that takes no effort at
    all. Every 5.5 model takes the whole ladder (2026-10-08), so the clamp in
    `llm_models.effective_effort` and the "no thinking config" branch are idle on the real
    table; they stay in the code for the next row that needs them, and these rows keep them
    tested. Returns (no_xhigh_id, no_effort_id)."""
    from orchestrator import llm_models
    rows = (llm_models.ModelSpec("claude-test-no-xhigh", "No X-High", "Test", 3.0, 15.0,
                                 efforts=("low", "medium", "high", "max")),
            llm_models.ModelSpec("claude-test-no-effort", "No effort", "Test", 1.0, 5.0,
                                 context_tokens=200_000, max_output_tokens=64_000, efforts=()))
    monkeypatch.setattr(llm_models, "MODELS", llm_models.MODELS + rows)
    monkeypatch.setattr(llm_models, "_BY_ID", {**llm_models._BY_ID, **{m.id: m for m in rows}})
    return rows[0].id, rows[1].id
