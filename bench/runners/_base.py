"""Shared segment loop for every runner (runs inside the model's own env under envs/).

A runner module defines `load(device) -> engine` and `synth(engine, row) -> (audio, rate)` and
calls `main(load, synth)`. The driver (../render.py) starts it as

    envs/<model>/bin/python runners/run_<model>.py <work.jsonl> <out_dir>

Every row is one segment of an engine script (paths in it are relative to the repo root) and is
saved as <out_dir>/<id>.wav (an id may contain "/"). A segment whose wav already exists is
skipped, so a killed run resumes. One JSON line per event goes to stdout: {"event": "loaded" |
"segment" | "error" | "done", ...} (the driver ignores other lines); the model's own prints go to
stderr. Only numpy + soundfile here.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # the repo root


def clip_path(row: dict) -> str:
    return str(ROOT / row["voice_clip"])


def ref_text(row: dict) -> str:
    if row.get("ref_text"):
        return row["ref_text"]
    path = ROOT / Path(row["voice_clip"]).with_suffix(".txt")
    return path.read_text(encoding="utf-8").strip()


_OUT = sys.stdout  # events only; main() points sys.stdout (the models' prints) at stderr


def _emit(**event) -> None:
    _OUT.write(json.dumps(event) + "\n")
    _OUT.flush()


def _peak_gb(engine=None) -> float | None:
    # A runner whose model lives in another process (a server) gives its own reading.
    if isinstance(engine, dict) and callable(engine.get("peak_gb")):
        return engine["peak_gb"]()
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1e9, 2)
    except Exception:  # noqa: BLE001 - torch-less runners (onnx) just don't report it
        return None
    return None


def main(load, synth) -> None:
    import numpy as np
    import soundfile as sf

    sys.stdout = sys.stderr
    work, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in work.read_text(encoding="utf-8").splitlines() if x.strip()]
    todo = [r for r in rows if not (out_dir / f"{r['id']}.wav").exists()]
    if not todo:
        _emit(event="done", rendered=0)
        return
    device = os.environ.get("VT_DEVICE", "cuda")
    started = time.perf_counter()
    engine = load(device)
    _emit(event="loaded", load_s=round(time.perf_counter() - started, 1), peak_gb=_peak_gb(engine))
    rendered = 0
    for row in todo:
        t0 = time.perf_counter()
        try:
            audio, rate = synth(engine, row)
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            if audio.size == 0 or not np.isfinite(audio).all():
                raise RuntimeError(f"bad audio ({audio.size} samples)")
            final = out_dir / f"{row['id']}.wav"
            final.parent.mkdir(parents=True, exist_ok=True)
            tmp = final.with_name(final.stem + ".tmp.wav")
            sf.write(str(tmp), audio, rate)
            os.replace(tmp, final)
            rendered += 1
            _emit(
                event="segment",
                id=row["id"],
                compute_s=round(time.perf_counter() - t0, 2),
                audio_s=round(audio.size / rate, 2),
                peak_gb=_peak_gb(engine),
            )
        except Exception as exc:  # noqa: BLE001 - one bad segment must not stop the model
            traceback.print_exc()
            _emit(event="error", id=row["id"], error=f"{type(exc).__name__}: {exc}"[:500])
    _emit(event="done", rendered=rendered, peak_gb=_peak_gb(engine))
