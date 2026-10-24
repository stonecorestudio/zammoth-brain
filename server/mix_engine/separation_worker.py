"""The stem split's own process (P3 batch T2, reader S3·R10; James, 2026-10-06).

The model's inference is one call that nothing can interrupt. Run inside the brain it made
Stop a wait of minutes, and the app's answer to a Stop that does not land is to kill the
brain every open project shares. Run here, Stop is a kill of this process and nothing else.

    python -m mix_engine.separation_worker <source audio> <out dir> <stems>
    mixagent_daw_brain --separation-worker <source audio> <out dir> <stems>     (frozen)

It writes into <out dir>, which its caller made and removes:
    stage.separating   an empty file, once the model is on disk and the inference starts
    <stem>.npy         float32 (frames, channels), one per stem
    done.json          LAST: {ok: true, sample_rate} or {ok: false, kind, message}
Nothing is written into the project: `daw_bridge.separate_track` names and places the stems.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

STAGE_SEPARATING = "stage.separating"
DONE = "done.json"


def exit_when_orphaned(every_s: float = 1.0) -> None:
    """End this process when the one that started it has gone. A brain that is killed
    mid-split (a force-stop, a crash) would otherwise leave minutes of inference running with
    nobody to hand the stems to."""
    parent = os.getppid()

    def watch() -> None:
        # (adopted by the system already = the starter went before we could note who it was)
        while parent != 1 and os.getppid() == parent:
            time.sleep(every_s)
        print("[separation_worker] the brain that started this split has gone - stopping",
              file=sys.stderr, flush=True)
        os._exit(3)

    threading.Thread(target=watch, name="split-orphan-watch", daemon=True).start()


def _answer(out: Path, payload: dict) -> None:
    tmp = out / (DONE + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(out / DONE)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: separation_worker <source audio> <out dir> <stems>", file=sys.stderr)
        return 2
    src, out, stems = Path(argv[0]), Path(argv[1]), int(argv[2])
    exit_when_orphaned()

    import numpy as np
    import soundfile as sf

    from mix_engine.separation import ModelNotAvailable, get_separator, stem_names

    try:
        audio, sample_rate = sf.read(str(src), dtype="float32", always_2d=True)
        sep = get_separator(stems)
        if hasattr(sep, "fetch_models"):
            sep.fetch_models()                  # the first-use download; a no-op when cached
        (out / STAGE_SEPARATING).touch()
        result = sep.separate(audio, sample_rate)
        for name in stem_names(stems):
            np.save(str(out / f"{name}.npy"), np.asarray(result[name], dtype=np.float32))
        _answer(out, {"ok": True, "sample_rate": int(sample_rate)})
    except ModelNotAvailable as exc:
        _answer(out, {"ok": False, "kind": "no_model", "message": str(exc)})
    except Exception as exc:  # noqa: BLE001 - whatever it is, it is the answer
        _answer(out, {"ok": False, "kind": "error", "message": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
