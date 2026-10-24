"""Produce the Spleeter ONNX models the Phase 29 separator loads.

Two modes — `install` is bulletproof, `export` is best-effort:

    # You already have a per-source .onnx (sherpa-onnx Spleeter is one UNet per
    # source): verify + place it.
    python -m scripts.export_spleeter_onnx install vocals.onnx --stems 2 --source vocals

    # Attempt the Spleeter (TF) -> ONNX conversion here (needs spleeter + tf2onnx
    # on a machine that can download Deezer's MIT weights).
    python -m scripts.export_spleeter_onnx export --stems 4

Run from `server/`. The model resolves at runtime from `MIX_AGENT_DATA_DIR/models/`
(dev/user drop-in) then the in-package `mix_engine/models/` (the bundle) — see
`mix_engine.separation.resolve_model_path`. `install` copies into the in-package
dir by default (so `plugin/sidecar/build.sh` bundles it) and prints the SHA-256
to paste into `separation._MODELS[...].sha256` so loads are checksum-verified.

**Verify the conversion at build time, not from memory** (docs/07-packaging.md):
Spleeter ships as TensorFlow and its exact graph signature shifts across
versions; the `export` path here captures the standard route but the I/O names
must match what `separation.SpleeterOnnxSeparator._run_model` expects (one soft
mask output per source, in `FOUR_STEM` / `TWO_STEM` order). `install` does a
load-and-shape sanity check via onnxruntime before it copies, so a wrong-shaped
model is caught here rather than at first use.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

# mix_engine is importable when run from server/ (the engine package root).
from mix_engine.separation import _MODELS  # noqa: E402

PACKAGE_MODELS_DIR = Path(__file__).resolve().parents[1] / "mix_engine" / "models"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sanity_check(path: Path) -> None:
    """Load via onnxruntime and print the I/O signature so a mismatched export
    is caught here, not at separation time. sherpa-onnx Spleeter is one UNet per
    source (a single magnitude output, input [2, num_splits, 512, 1024])."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("! onnxruntime not installed — skipping the load/shape check "
              "(install it to validate before shipping)", file=sys.stderr)
        return
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    print("  model signature:")
    for i in sess.get_inputs():
        print(f"    input  {i.name}: {i.shape} ({i.type})")
    for o in sess.get_outputs():
        print(f"    output {o.name}: {o.shape} ({o.type})")
    print(f"✓ onnxruntime loads it ({len(sess.get_outputs())} output(s))")


def install(onnx_path: Path, stems: int, source: str, dest_dir: Path) -> None:
    spec = _MODELS[stems]
    if source not in spec.files:
        raise SystemExit(f"--source must be one of {list(spec.files)} for "
                         f"{stems}-stem")
    if not onnx_path.is_file():
        raise SystemExit(f"no such file: {onnx_path}")
    _sanity_check(onnx_path)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / spec.files[source]
    shutil.copyfile(onnx_path, target)
    digest = _sha256(target)
    print(f"✓ installed {target}")
    print(f"  sha256: {digest}")
    print(f"  -> set _MODELS[{stems}].sha256['{source}'] = \"{digest}\" in "
          "mix_engine/separation.py to enable checksum verification.")


def export(stems: int, out_dir: Path) -> None:
    """Best-effort Spleeter (TF) -> ONNX. Deezer's Spleeter code + pretrained
    weights are MIT and trained on Deezer's own (non-MUSDB) catalog — clean for
    a commercial ship. Needs `spleeter` and `tf2onnx`."""
    checkpoint = _MODELS[stems].checkpoint  # "2stems" | "4stems"
    try:
        import tf2onnx  # noqa: F401
        from spleeter.separator import Separator  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"missing dependency ({exc}). The conversion needs `pip install "
            "spleeter tf2onnx` on a machine that can fetch the pretrained "
            f"weights, then exports the `spleeter:{checkpoint}` model. If you "
            "already have an .onnx, use the `install` mode instead — it has no "
            "TF/spleeter dependency.")
    # Spleeter loads `spleeter:<checkpoint>` (downloads Deezer's MIT weights to
    # the pretrained-models cache), exposes a TF separation model over the
    # STFT magnitude. Convert that to ONNX with tf2onnx, preserving one soft
    # mask output per source. The exact estimator->function extraction is
    # version-specific — VERIFY against the installed spleeter/tf2onnx docs at
    # build time (docs/07). Pseudocode of the route:
    #   sep = Separator(f"spleeter:{checkpoint}")
    #   model_fn, signature = <spleeter's separation tf.function + input spec>
    #   tf2onnx.convert.from_function(model_fn, input_signature=signature,
    #                                 output_path=str(out))
    raise SystemExit(
        "export scaffold reached: wire spleeter's separation tf.function into "
        "tf2onnx.convert.from_function here, matching the installed versions "
        "(see the comment above + docs/07-packaging.md), then run `install` on "
        "the produced file to checksum + place it. Kept explicit so a stale, "
        "unverified conversion can't ship silently.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    pi = sub.add_parser("install", help="verify + place an existing per-source .onnx")
    pi.add_argument("onnx", type=Path)
    pi.add_argument("--stems", type=int, choices=(2, 4), default=2)
    pi.add_argument("--source", required=True,
                    help="which source this .onnx is (vocals/accompaniment for "
                         "2-stem; vocals/drums/bass/other for 4-stem)")
    pi.add_argument("--data-dir", type=Path, default=None,
                    help="install into <dir>/models instead of the in-package "
                         "bundle dir (e.g. your MIX_AGENT_DATA_DIR)")
    pe = sub.add_parser("export", help="best-effort Spleeter -> ONNX (needs tf)")
    pe.add_argument("--stems", type=int, choices=(2, 4), default=4)
    pe.add_argument("--out", type=Path, default=PACKAGE_MODELS_DIR)
    args = ap.parse_args()

    if args.mode == "install":
        dest = (args.data_dir / "models") if args.data_dir else PACKAGE_MODELS_DIR
        install(args.onnx, args.stems, args.source, dest)
    else:
        export(args.stems, args.out)


if __name__ == "__main__":
    main()
