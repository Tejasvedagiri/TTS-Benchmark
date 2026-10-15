"""Assembly: a test's `segments/<id>.wav` -> `listen/<model>/<test>.wav` (+ `<test>-scene-0N.wav`
when the test has more than one scene).

Pauses: 0.25 s between segments of a line, 0.55 s between lines, 1.4 s between scenes. A
segment with no audio (a sound-only line the model has no tag for) adds nothing, not even its
pause. Peak-normalised to 0.9 over the whole script, so scenes keep their relative loudness.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

PAUSE_SEGMENT = 0.25
PAUSE_LINE = 0.55
PAUSE_SCENE = 1.4


def _mono(path: Path) -> tuple[np.ndarray, int]:
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return audio.mean(axis=1), rate


def _resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or audio.size == 0:
        return audio
    n = int(round(audio.size * dst / src))
    return np.interp(np.linspace(0, audio.size - 1, n), np.arange(audio.size), audio).astype(
        np.float32
    )


def assemble(segments_dir: Path, rows: list[dict], out_dir: Path, name: str) -> dict:
    """Returns {"scenes": [paths], "full": path, "audio_s": seconds, "missing": [ids]}."""
    rate = None
    scenes: dict[int, list[list[np.ndarray]]] = {}  # scene -> lines -> segment audio
    last_line = None
    missing = []
    for row in rows:
        path = segments_dir / f"{row['id']}.wav"
        if not path.exists():
            if not row.get("empty"):
                missing.append(row["id"])
            continue
        audio, sr = _mono(path)
        rate = rate or sr
        audio = _resample(audio, sr, rate)
        lines = scenes.setdefault(row["scene"], [])
        if row["line"] != last_line:
            lines.append([])
            last_line = row["line"]
        lines[-1].append(audio)
    if rate is None:
        return {"scenes": [], "full": None, "audio_s": 0.0, "missing": missing}

    def join(parts: list[np.ndarray], pause: float) -> np.ndarray:
        out: list[np.ndarray] = []
        for part in parts:
            if out:
                out.append(np.zeros(int(pause * rate), np.float32))
            out.append(part)
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    scene_audio = {
        n: join([join(segs, PAUSE_SEGMENT) for segs in lines], PAUSE_LINE)
        for n, lines in sorted(scenes.items())
    }
    full = join(list(scene_audio.values()), PAUSE_SCENE)
    gain = 0.9 / (float(np.max(np.abs(full))) or 1.0)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_paths = []
    if len(scene_audio) > 1:
        for n, audio in scene_audio.items():
            path = out_dir / f"{name}-scene-{n + 1:02d}.wav"
            sf.write(str(path), audio * gain, rate)
            scene_paths.append(str(path))
    full_path = out_dir / f"{name}.wav"
    sf.write(str(full_path), full * gain, rate)
    return {
        "scenes": scene_paths,
        "full": str(full_path),
        "audio_s": round(full.size / rate, 1),
        "missing": missing,
    }
