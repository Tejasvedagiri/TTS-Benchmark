"""Pocket-TTS (kyutai/pocket-tts, gated): clones from the clip (one voice state per speaker, as
the README advises), no tags. Built for CPU (no device argument; Spec.device "cpu"); moved to
the GPU only when VT_DEVICE says cuda."""

from _base import clip_path, main


def load(device):
    from pocket_tts import TTSModel

    model = TTSModel.load_model()
    if device != "cpu":
        model.to(device)
    return {"model": model, "states": {}}


def synth(e, row):
    model = e["model"]
    state = e["states"].get(row["voice"])
    if state is None:
        state = e["states"][row["voice"]] = model.get_state_for_audio_prompt(clip_path(row))
    audio = model.generate_audio(state, row["text"])
    return audio.float().cpu().numpy(), model.sample_rate


if __name__ == "__main__":
    main(load, synth)
