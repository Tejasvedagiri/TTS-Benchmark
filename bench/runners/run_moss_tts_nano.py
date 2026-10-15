"""MOSS-TTS-Nano (OpenMOSS-Team/MOSS-TTS-Nano-100M): clip only ("voice_clone" mode), no tags.
The checkpoint's own remote code (`model.inference`), with upstream infer.py's defaults: bf16,
sdpa, sampling on (text 1.0 / 1.0 / 50, audio 0.8 / 0.95 / 25, repetition 1.2), 375 frames,
75 text tokens per chunk. No WeTextProcessing (pynini): our text needs no normalisation.
inference() writes a file. 48 kHz."""

import tempfile
from pathlib import Path

from _base import clip_path, main

REPO = "OpenMOSS-Team/MOSS-TTS-Nano-100M"
SAMPLING = {
    "text_temperature": 1.0,
    "text_top_p": 1.0,
    "text_top_k": 50,
    "audio_temperature": 0.8,
    "audio_top_p": 0.95,
    "audio_top_k": 25,
    "audio_repetition_penalty": 1.2,
}


def load(device):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(REPO, trust_remote_code=True)
    model.to(device=device, dtype=torch.bfloat16)
    model._set_attention_implementation("sdpa")
    model.eval()
    return {"model": model, "device": torch.device(device)}


def synth(e, row):
    import soundfile as sf
    import torch

    torch.manual_seed(1234)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.wav"
        result = e["model"].inference(
            text=row["text"],
            output_audio_path=str(out),
            mode="voice_clone",
            prompt_audio_path=clip_path(row),
            audio_tokenizer_type="moss-audio-tokenizer-nano",
            audio_tokenizer_pretrained_name_or_path="OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
            device=e["device"],
            max_new_frames=375,
            voice_clone_max_text_tokens=75,
            voice_clone_max_memory_per_sample_gb=1.0,
            do_sample=True,
            use_kv_cache=True,
            **SAMPLING,
        )
        audio, rate = sf.read(str(result.get("audio_path") or out), dtype="float32", always_2d=True)
    return audio.mean(axis=1), rate


if __name__ == "__main__":
    main(load, synth)
