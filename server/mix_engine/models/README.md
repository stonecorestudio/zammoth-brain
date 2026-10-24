# Separation models (Phase 29)

> **2026-08-13: this dir is now the LEGACY path.** The product separator is
> htdemucs v4 via the `demucs-onnx` package (`kind="demucs"`, the default) —
> its weights are NOT placed here; they download on first use into
> `<data_dir>/models/demucs` (huggingface_hub cache, ~166 MB fp16). The
> Spleeter files below remain supported behind `get_separator(kind="onnx")`.

The Spleeter ONNX models the stem separator loads at runtime. The wired models
are the **k2-fsa / sherpa-onnx** Spleeter conversion — **one UNet per source**
(MIT; Spleeter's weights are MIT, trained on Deezer's own non-MUSDB catalog).

Resolution order (`mix_engine.separation.resolve_model_files`):

1. `$MIX_AGENT_DATA_DIR/models/` — dev / user drop-in (wins).
2. this in-package dir — what `daw/sidecar/build.sh` bundles into the DAW's brain.

Per-source filenames (not committed — see `.gitignore`):

- **2-stem (live):** `spleeter_2stems_vocals.onnx`, `spleeter_2stems_accompaniment.onnx`
- **4-stem (pending a clean ONNX):** `spleeter_4stems_{vocals,drums,bass,other}.onnx`

To place one (no TF needed — onnxruntime only):

```bash
cd server
# download each source .onnx from the sherpa-onnx HF repo
#   huggingface.co/csukuangfj/sherpa-onnx-spleeter-2stems  (vocals.onnx, accompaniment.onnx)
# then verify + place + print the sha256 to pin in separation._MODELS:
python -m scripts.export_spleeter_onnx install vocals.onnx --stems 2 --source vocals
python -m scripts.export_spleeter_onnx install accompaniment.onnx --stems 2 --source accompaniment
```

Converting Spleeter→ONNX yourself needs TensorFlow, which won't run on Apple
Silicon (no arm64 wheel; Rosetta lacks AVX) — do it on a Linux/Intel box (see
`scripts/export_spleeter_onnx.py` `export` + `docs/07-packaging.md`).
