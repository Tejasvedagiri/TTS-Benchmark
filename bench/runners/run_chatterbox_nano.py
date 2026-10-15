"""Chatterbox-Nano (ResembleAI/chatterbox-nano): clip only, tags inline (git chatterbox only)."""

from _base import clip_path, main


def load(device):
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    return ChatterboxTurboTTS.from_pretrained(device=device, nano=True)


def synth(model, row):
    wav = model.generate(row["text"], audio_prompt_path=clip_path(row))
    return wav.squeeze().detach().cpu().numpy(), model.sr


if __name__ == "__main__":
    main(load, synth)
