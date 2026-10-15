"""Miso TTS 8B (MisoLabs/MisoTTS, Sesame CSM architecture): clip + transcript as a speaker-0
context segment, then the segment as speaker 0 (the repo's run_misotts.py flow). The repo's own
code from envs/miso_tts/src, bf16. Its tokenizer is meta-llama/Llama-3.2-1B, gated: the same
tokenizer is read from the ungated unsloth/Llama-3.2-1B mirror instead. 24 kHz."""

import os
import sys

from _base import ROOT, clip_path, main, ref_text

os.environ["NO_TORCH_COMPILE"] = "1"  # as run_misotts.py
sys.path.insert(0, str(ROOT / "envs" / "miso_tts" / "src"))
TOKENIZER_MIRROR = "unsloth/Llama-3.2-1B"


def load(device):
    import generator
    from transformers import AutoTokenizer

    original = AutoTokenizer.from_pretrained

    def from_mirror(name, *args, **kwargs):
        if name == "meta-llama/Llama-3.2-1B":
            name = TOKENIZER_MIRROR
        return original(name, *args, **kwargs)

    generator.AutoTokenizer.from_pretrained = from_mirror
    return {"gen": generator.load_miso_8b(device), "ctx": {}}


def synth(e, row):
    import torch
    import torchaudio
    from generator import Segment

    gen = e["gen"]
    ctx = e["ctx"].get(row["voice"])
    if ctx is None:
        audio, rate = torchaudio.load(clip_path(row))
        audio = torchaudio.functional.resample(audio.mean(dim=0), rate, gen.sample_rate)
        ctx = e["ctx"][row["voice"]] = Segment(text=ref_text(row), speaker=0, audio=audio)
    torch.manual_seed(1234)
    audio = gen.generate(text=row["text"], speaker=0, context=[ctx], max_audio_length_ms=30_000)
    return audio.float().cpu().numpy(), gen.sample_rate


if __name__ == "__main__":
    main(load, synth)
