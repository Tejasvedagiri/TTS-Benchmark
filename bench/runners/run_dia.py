"""Dia 1.6B (nari-labs/Dia-1.6B-0626): voice cloning by prefixing the clip's transcript as an
[S1] turn and passing the clip as audio_prompt (upstream example/voice_clone.py); the reply is
only the new text. Tags like (laughs) inline; the [S1] prefix is already in the text."""

from _base import clip_path, main, ref_text

MODEL = "nari-labs/Dia-1.6B-0626"


def load(device):
    from dia.model import Dia

    return Dia.from_pretrained(MODEL, compute_dtype="float16", device=device)


def synth(model, row):
    import torch

    torch.manual_seed(1234)
    text = f"[S1] {ref_text(row)} {row['text']}"
    audio = model.generate(
        text, audio_prompt=clip_path(row), use_torch_compile=False, verbose=False
    )
    return audio, 44100


if __name__ == "__main__":
    main(load, synth)
