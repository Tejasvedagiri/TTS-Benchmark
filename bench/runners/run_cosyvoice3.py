"""Fun-CosyVoice 3 (FunAudioLLM/Fun-CosyVoice3-0.5B-2512): clones from the clip; the story
model's free-text `description` goes in as CosyVoice 3's instruction ("You are a helpful
assistant. <description><|endofprompt|>", inference_instruct2), else plain zero-shot with the
clip's transcript. The repo's own code from envs/cosyvoice3/src (+ its Matcha-TTS submodule),
non-streaming, as upstream example.py. 24 kHz."""

import sys

from _base import ROOT, clip_path, main, ref_text

SRC = ROOT / "envs" / "cosyvoice3" / "src"
sys.path[:0] = [str(SRC), str(SRC / "third_party" / "Matcha-TTS")]
REPO = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
SYSTEM = "You are a helpful assistant."


def load(device):
    from cosyvoice.cli.cosyvoice import AutoModel
    from huggingface_hub import snapshot_download

    return AutoModel(model_dir=snapshot_download(REPO))


def synth(model, row):
    import numpy as np
    import torch

    torch.manual_seed(1234)
    if row.get("description"):
        parts = model.inference_instruct2(
            row["text"],
            f"{SYSTEM} {row['description']}<|endofprompt|>",
            clip_path(row),
            stream=False,
        )
    else:
        parts = model.inference_zero_shot(
            row["text"], f"{SYSTEM}<|endofprompt|>{ref_text(row)}", clip_path(row), stream=False
        )
    audio = [p["tts_speech"].float().cpu().numpy().reshape(-1) for p in parts]
    return np.concatenate(audio), model.sample_rate


if __name__ == "__main__":
    main(load, synth)
