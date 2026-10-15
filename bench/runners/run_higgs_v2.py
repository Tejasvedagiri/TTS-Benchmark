"""Higgs TTS 2 (bosonai/higgs-tts-2-3b-base): clone by in-context example, as the model card's
Transformers voice-cloning snippet: the clip's transcript as a user turn, the clip as the
assistant's audio, then the segment as the next user turn. Greedy (do_sample=False, as the
card). 24 kHz."""

import tempfile
from pathlib import Path

from _base import clip_path, main, ref_text

REPO = "bosonai/higgs-tts-2-3b-base"
SYSTEM = "Generate audio following instruction."
SCENE = "Audio is recorded from a quiet room."


def load(device):
    import torch
    from transformers import AutoProcessor, HiggsAudioV2ForConditionalGeneration

    # the processor holds the audio tokenizer model: on the GPU too, as the card's device_map
    processor = AutoProcessor.from_pretrained(REPO, device_map=device)
    model = HiggsAudioV2ForConditionalGeneration.from_pretrained(
        REPO, device_map=device, dtype=torch.bfloat16
    )
    return {"proc": processor, "model": model}


def synth(e, row):
    import soundfile as sf

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "scene", "content": [{"type": "text", "text": SCENE}]},
        {"role": "user", "content": [{"type": "text", "text": ref_text(row)}]},
        {"role": "assistant", "content": [{"type": "audio", "path": clip_path(row)}]},
        {"role": "user", "content": [{"type": "text", "text": row["text"]}]},
    ]
    proc, model = e["proc"], e["model"]
    inputs = proc.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        sampling_rate=24000,
        return_tensors="pt",
    ).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=2000, do_sample=False)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.wav"
        proc.save_audio(proc.batch_decode(outputs), str(out))
        audio, rate = sf.read(str(out), dtype="float32", always_2d=True)
    return audio.mean(axis=1), rate


if __name__ == "__main__":
    main(load, synth)
