"""OmniVoice (k2-fsa/OmniVoice): clip + transcript, [laughter] / [sigh] inline. 24 kHz."""

from _base import clip_path, main, ref_text


def load(device):
    import torch
    from omnivoice import OmniVoice

    return OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device, dtype=torch.float16)


def synth(model, row):
    audio = model.generate(text=row["text"], ref_audio=clip_path(row), ref_text=ref_text(row))
    return audio[0], 24000


if __name__ == "__main__":
    main(load, synth)
