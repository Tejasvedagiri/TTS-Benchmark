"""MOSS-TTS v1.5 (OpenMOSS-Team/MOSS-TTS-v1.5, 8B): clip as reference, [pause Ns] inline.
The model card's generation flow (trust_remote_code), bf16."""

import _npp  # noqa: F401 - before torchcodec
from _base import clip_path, main

REPO = "OpenMOSS-Team/MOSS-TTS-v1.5"


def load(device):
    import torch
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(REPO, trust_remote_code=True)
    processor.audio_tokenizer = processor.audio_tokenizer.to(device)
    model = AutoModel.from_pretrained(
        REPO, trust_remote_code=True, attn_implementation="sdpa", torch_dtype=torch.bfloat16
    ).to(device)
    return {"proc": processor, "model": model, "device": device}


def synth(e, row):
    import torch

    proc = e["proc"]
    message = proc.build_user_message(
        text=row["text"], reference=[clip_path(row)], language="English"
    )
    batch = proc([[message]], mode="generation")
    torch.manual_seed(1234)
    with torch.inference_mode():
        out = e["model"].generate(
            input_ids=batch["input_ids"].to(e["device"]),
            attention_mask=batch["attention_mask"].to(e["device"]),
            max_new_tokens=4096,
        )
    audio = next(iter(proc.decode(out))).audio_codes_list[0]
    return audio.float().cpu().numpy(), int(proc.model_config.sampling_rate)


if __name__ == "__main__":
    main(load, synth)
