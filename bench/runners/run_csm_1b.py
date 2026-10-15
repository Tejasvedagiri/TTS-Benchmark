"""Sesame CSM-1B (sesame/csm-1b, gated): clip + transcript as a speaker-0 context turn, then the
segment as the next speaker-0 turn. Transformers' native CSM (CsmForConditionalGeneration), not
the repo's generator.py, which also needs the gated meta-llama/Llama-3.2-1B tokenizer. 24 kHz."""

from _base import clip_path, main, ref_text

REPO = "sesame/csm-1b"


def load(device):
    from transformers import AutoProcessor, CsmForConditionalGeneration

    processor = AutoProcessor.from_pretrained(REPO)
    model = CsmForConditionalGeneration.from_pretrained(REPO, device_map=device)
    return {"proc": processor, "model": model, "device": device, "clips": {}}


def synth(e, row):
    import soundfile as sf
    import torch

    clip = e["clips"].get(row["voice"])
    if clip is None:
        audio, rate = sf.read(clip_path(row), dtype="float32", always_2d=True)
        if rate != 24000:
            raise RuntimeError(f"clip is {rate} Hz; CSM wants 24 kHz")
        clip = e["clips"][row["voice"]] = audio.mean(axis=1)
    conversation = [
        {
            "role": "0",
            "content": [{"type": "text", "text": ref_text(row)}, {"type": "audio", "path": clip}],
        },
        {"role": "0", "content": [{"type": "text", "text": row["text"]}]},
    ]
    inputs = e["proc"].apply_chat_template(conversation, tokenize=True, return_dict=True)
    torch.manual_seed(1234)
    out = e["model"].generate(**inputs.to(e["device"]), output_audio=True)
    return out[0].float().cpu().numpy(), 24000


if __name__ == "__main__":
    main(load, synth)
