"""VoxCPM2 (openbmb/VoxCPM2): clip (+ its transcript, which upstream takes as prompt_text), and
the story model's tone written as a "(... tone)" prefix."""

from _base import clip_path, main, ref_text


def load(device):
    from voxcpm import VoxCPM

    return VoxCPM.from_pretrained("openbmb/VoxCPM2")


def synth(model, row):
    text = row["text"]
    if row.get("tone_prefix"):
        text = f"{row['tone_prefix']} {text}"
    wav = model.generate(text=text, prompt_wav_path=clip_path(row), prompt_text=ref_text(row))
    return wav, model.tts_model.sample_rate


if __name__ == "__main__":
    main(load, synth)
