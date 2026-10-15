"""Kitten TTS 2 (KittenML/kitten-tts-2, 1.7B): clones from the clip (5-30 s; the model card's
API takes no transcript). 24 kHz."""

from _base import clip_path, main


def load(device):
    from kittenml import KittenTTS

    return KittenTTS("KittenML/kitten-tts-2")


def synth(model, row):
    audio = model.generate(row["text"], reference=clip_path(row))
    return audio, model.sample_rate


if __name__ == "__main__":
    main(load, synth)
