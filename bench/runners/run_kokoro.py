"""Kokoro-82M baseline (preset voices, no cloning, no tags). Each speaker gets a preset of
their gender (_presets.py): the same one their generated reference clip was made with."""

from _base import main
from _presets import preset


def load(device):
    from kokoro import KPipeline

    return KPipeline(lang_code="a", device=device)


def synth(pipe, row):
    import numpy as np

    parts = [
        np.asarray(audio, dtype=np.float32)
        for _, _, audio in pipe(row["text"], voice=preset("kokoro", row))
    ]
    return np.concatenate(parts), 24000


if __name__ == "__main__":
    main(load, synth)
