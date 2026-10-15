"""Zonos v0.1 transformer (Zyphra/Zonos-v0.1-transformer): clones from the clip's speaker
embedding; the story model's 8 emotion weights go in as `emotion`, its speaking_rate (a
multiplier) scales Zonos's default 15 phonemes/min. pitch_shift has no Zonos knob (all 0 here).
README's Python example; torch.compile off (no mamba, so no CUDA graphs either). 44.1 kHz."""

import sys

import _espeak  # noqa: F401 - before zonos (phonemizer)
from _base import ROOT, clip_path, main

sys.path.insert(0, str(ROOT / "envs" / "zonos" / "src"))

REPO = "Zyphra/Zonos-v0.1-transformer"
DEFAULT_RATE = 15.0


def load(device):
    from zonos.model import Zonos

    return {"model": Zonos.from_pretrained(REPO, device=device), "spk": {}, "device": device}


def synth(e, row):
    import torch
    import torchaudio
    from zonos.conditioning import make_cond_dict

    model = e["model"]
    spk = e["spk"].get(row["voice"])
    if spk is None:
        # Built on the CPU: under its torch.device("cuda") context torchaudio 2.8's mel
        # filterbank mixes devices. Once per speaker, so the CPU costs nothing here.
        if model.spk_clone_model is None:
            from zonos.speaker_cloning import SpeakerEmbeddingLDA

            model.spk_clone_model = SpeakerEmbeddingLDA(device="cpu")
        wav, sr = torchaudio.load(clip_path(row))
        spk = model.make_speaker_embedding(wav, sr).to(e["device"])
        e["spk"][row["voice"]] = spk
    kwargs = {}
    if row.get("emotion_vector"):
        kwargs["emotion"] = [float(x) for x in row["emotion_vector"]]
    cond = make_cond_dict(
        text=row["text"],
        speaker=spk,
        language="en-us",
        speaking_rate=DEFAULT_RATE * float(row.get("speaking_rate", 1.0)),
        device=e["device"],
        **kwargs,
    )
    torch.manual_seed(1234)
    codes = model.generate(
        model.prepare_conditioning(cond), progress_bar=False, disable_torch_compile=True
    )
    audio = model.autoencoder.decode(codes).cpu()
    return audio[0].float().numpy().reshape(-1), model.autoencoder.sampling_rate


if __name__ == "__main__":
    main(load, synth)
