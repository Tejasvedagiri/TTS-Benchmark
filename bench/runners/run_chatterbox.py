"""Chatterbox (ResembleAI/chatterbox, original): clip + exaggeration / cfg_weight knobs."""

from _base import clip_path, main


def load(device):
    from chatterbox.tts import ChatterboxTTS

    return ChatterboxTTS.from_pretrained(device=device)


def synth(model, row):
    wav = model.generate(
        row["text"],
        audio_prompt_path=clip_path(row),
        exaggeration=row.get("exaggeration", 0.5),
        cfg_weight=row.get("cfg_weight", 0.5),
    )
    return wav.squeeze().detach().cpu().numpy(), model.sr


if __name__ == "__main__":
    main(load, synth)
