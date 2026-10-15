"""Qwen3-TTS 1.7B Base: clip + transcript, no tags. Calls as qwen-tts's README."""

from _base import clip_path, main, ref_text

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
SEED = 1234


def load(device):
    import torch
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        MODEL, device_map=device, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    return {"model": model, "prompts": {}}


def synth(engine, row):
    import torch

    prompt = engine["prompts"].get(row["voice"])
    if prompt is None:
        prompt = engine["model"].create_voice_clone_prompt(
            ref_audio=clip_path(row), ref_text=ref_text(row), x_vector_only_mode=False
        )
        engine["prompts"][row["voice"]] = prompt
    torch.manual_seed(SEED)
    wavs, rate = engine["model"].generate_voice_clone(
        text=row["text"], language="English", voice_clone_prompt=prompt
    )
    return wavs[0], int(rate)


if __name__ == "__main__":
    main(load, synth)
