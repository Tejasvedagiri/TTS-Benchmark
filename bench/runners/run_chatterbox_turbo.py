"""Chatterbox-Turbo (ResembleAI/chatterbox-turbo): clip only, tags inline in the text."""

from _base import clip_path, main


def load(device):
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    return ChatterboxTurboTTS.from_pretrained(device=device)


def synth(model, row):
    wav = model.generate(row["text"], audio_prompt_path=clip_path(row))
    return wav.squeeze().detach().cpu().numpy(), model.sr


if __name__ == "__main__":
    main(load, synth)
