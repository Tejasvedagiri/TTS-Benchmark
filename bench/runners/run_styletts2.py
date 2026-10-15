"""StyleTTS 2 (the `styletts2` PyPI wrapper of yl4579/StyleTTS2, LibriTTS multi-speaker model):
clones from the clip; style_scale -> embedding_scale. 24 kHz."""

import os

import _espeak  # noqa: F401 - before styletts2 (phonemizer)

# The package's official LibriTTS checkpoint is a full pickle; torch >= 2.6 loads weights only
# by default and refuses it. Trusted source (the styletts2 package's own download).
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
from _base import clip_path, main


def load(device):
    import nltk

    # its text splitter needs punkt_tab; kept with every other download ($HF_HOME)
    data = os.path.join(os.environ["HF_HOME"], "nltk_data")
    nltk.data.path.insert(0, data)
    nltk.download("punkt_tab", download_dir=data, quiet=True)
    from styletts2 import tts

    return {"tts": tts.StyleTTS2(), "styles": {}}


def synth(e, row):
    style = e["styles"].get(row["voice"])
    if style is None:
        style = e["styles"][row["voice"]] = e["tts"].compute_style(clip_path(row))
    audio = e["tts"].inference(
        row["text"], ref_s=style, embedding_scale=row.get("style_scale", 1.0)
    )
    return audio, 24000


if __name__ == "__main__":
    main(load, synth)
