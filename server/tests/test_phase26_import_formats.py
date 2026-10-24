"""Phase 26 (docs/05-roadmap.md): broader import formats.

The import boundary accepts mp3/flac/aiff/ogg alongside wav and decodes each
non-wav exactly once into a `<name>.wav` project copy — the engine stays
wav-only, so render determinism never meets a lossy decoder. Listings,
previews, folder adoption at open, and the reference brief regex follow.
"""
import json

import numpy as np
import pytest
import soundfile as sf
from starlette.testclient import TestClient

from orchestrator.agents import RuleAgent
from orchestrator.chat import ScriptedChat
from orchestrator.gateway import adopt_foreign_audio, create_app
from orchestrator.targets import reference_path_from_brief

FORMATS = ["mp3", "flac", "aiff", "ogg"]


@pytest.fixture()
def client(demo_project):
    app = create_app(demo_project, agent_factory=RuleAgent,
                     chat_factory=ScriptedChat)
    with TestClient(app) as test_client:
        yield test_client


def write_audio(folder, name, seconds=1.0, sr=44100):
    """A real encoded file of the requested format (libsndfile writes all
    five), not a renamed wav — the decode path must earn the test."""
    folder.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * sr)) / sr
    tone = 0.3 * np.sin(2 * np.pi * 330.0 * t)
    sf.write(folder / name, tone.astype(np.float32), sr)
    return folder / name


def project_stems(project) -> dict:
    return json.loads((project / "session.json").read_text())["stems"]


def test_fs_lists_every_supported_format(client, tmp_path):
    folder = tmp_path / "samples"
    for fmt in FORMATS:
        write_audio(folder, f"loop_{fmt}.{fmt}")
    write_audio(folder, "loop_wav.wav")
    (folder / "notes.txt").write_text("not audio")
    (folder / ".hidden.mp3").write_bytes(b"ID3")
    body = client.get("/api/fs", params={"path": str(folder)}).json()
    assert body["audio"] == sorted(
        [f"loop_{fmt}.{fmt}" for fmt in FORMATS] + ["loop_wav.wav"])
    assert body["audio_count"] == 5


@pytest.mark.parametrize("fmt", FORMATS)
def test_import_by_path_decodes_to_a_wav_stem(client, demo_project,
                                              tmp_path, fmt):
    sample = write_audio(tmp_path / "samples", f"pad_{fmt}.{fmt}")
    response = client.post("/api/import", params={"path": str(sample)})
    assert response.status_code == 200, response.text
    assert response.json()["stem_id"] == f"pad_{fmt}"

    wav = demo_project / f"pad_{fmt}.wav"
    assert wav.exists()
    info = sf.info(str(wav))
    assert info.samplerate == 44100
    # mp3 may pad the tail with encoder delay; never shorter than the source
    assert info.frames >= 44100
    assert f"pad_{fmt}" in project_stems(demo_project)
    # the source format never enters the project folder
    assert not (demo_project / sample.name).exists()

    # a converted stem collides like any other
    duplicate = client.post("/api/import", params={"path": str(sample)})
    assert duplicate.status_code == 409
    assert f"pad_{fmt}.wav" in duplicate.text


def test_drop_put_decodes_flac_and_cleans_up(client, demo_project, tmp_path):
    source = write_audio(tmp_path, "drop_kit.flac")
    response = client.put("/api/stems/drop_kit.flac",
                          content=source.read_bytes())
    assert response.status_code == 200, response.text
    assert response.json()["stem_id"] == "drop_kit"
    assert (demo_project / "drop_kit.wav").exists()
    assert "drop_kit" in project_stems(demo_project)
    # no temp dotfile and no flac left behind
    assert list(demo_project.glob(".import-*")) == []
    assert list(demo_project.glob("*.flac")) == []


def test_drop_put_rejects_junk(client, demo_project):
    # unsupported extension
    assert client.put("/api/stems/tune.xyz",
                      content=b"RIFF1234").status_code == 400
    # wrong magic for the claimed format
    response = client.put("/api/stems/tune.mp3", content=b"hello world")
    assert response.status_code == 400
    # right magic, undecodable body
    response = client.put("/api/stems/tune.flac",
                          content=b"fLaC" + b"\x00" * 64)
    assert response.status_code == 400
    assert "could not decode" in response.text
    assert list(demo_project.glob(".import-*")) == []
    assert not (demo_project / "tune.wav").exists()


def test_open_project_adopts_a_folder_of_mp3s(tmp_path):
    folder = tmp_path / "mp3_session"
    for name in ("kick.mp3", "bass.mp3"):
        write_audio(folder, name)
    app = create_app(folder, agent_factory=RuleAgent,
                     chat_factory=ScriptedChat)
    with TestClient(app):
        stems = project_stems(folder)
    assert set(stems) == {"kick", "bass"}
    assert (folder / "kick.wav").exists() and (folder / "bass.wav").exists()
    # sources stay untouched, and a second adoption pass is a no-op
    assert (folder / "kick.mp3").exists()
    assert adopt_foreign_audio(folder) == []


def test_adoption_never_overwrites_an_existing_wav(tmp_path):
    folder = tmp_path / "mixed"
    write_audio(folder, "kick.wav")
    before = (folder / "kick.wav").read_bytes()
    write_audio(folder, "kick.mp3")
    write_audio(folder, "snare.mp3")
    assert adopt_foreign_audio(folder) == ["snare.wav"]
    assert (folder / "kick.wav").read_bytes() == before


def test_preview_endpoints_serve_an_mp3(client, tmp_path):
    sample = write_audio(tmp_path / "samples", "audition.mp3")
    peaks = client.get("/api/preview/peaks", params={"path": str(sample)})
    assert peaks.status_code == 200
    assert peaks.json()["duration_s"] > 0.9
    audio = client.get("/api/preview/audio", params={"path": str(sample)})
    assert audio.status_code == 200
    assert audio.content.startswith(b"RIFF")  # decoded for the webview

    bad = client.get("/api/preview/audio",
                     params={"path": str(tmp_path / "samples" / "x.txt")})
    assert bad.status_code == 400


def test_reference_brief_regex_matches_every_format():
    for ext in ("wav", "mp3", "flac", "aiff", "aif", "ogg"):
        brief = f"make it sit like refs/target.{ext}"
        assert reference_path_from_brief(brief) == f"refs/target.{ext}"
    assert reference_path_from_brief("match the energy of the chorus") is None
    assert reference_path_from_brief("like refs/notes.txt") is None
