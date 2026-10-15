"""F5-TTS v1 (SWivid/F5-TTS): clip + transcript, no tags."""

from _base import clip_path, main, ref_text


def load(device):
    from f5_tts.api import F5TTS

    return F5TTS(device=device)


def synth(model, row):
    wav, rate, _ = model.infer(
        ref_file=clip_path(row),
        ref_text=ref_text(row),
        gen_text=row["text"],
        seed=1234,
        remove_silence=False,
    )
    return wav, rate


if __name__ == "__main__":
    main(load, synth)
