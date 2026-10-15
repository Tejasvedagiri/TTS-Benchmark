"""Voxtral 4B TTS (mistralai/Voxtral-4B-TTS-2603) baseline: presets only. The open checkpoint
has no voice-encoder weights, so vLLM-Omni refuses ref_audio ("requires the full Voxtral
checkpoint"); each speaker gets one fixed preset of their gender instead (_presets.py). Served
by vLLM-Omni (~27 GB on a 32 GB card per its recipe). Peak memory = nvidia-smi (includes the
desktop's ~1 GB). 24 kHz."""

from _base import main
from _presets import preset
from _vllm_omni import speech, start

REPO = "mistralai/Voxtral-4B-TTS-2603"


def load(device):
    return start(REPO)


def synth(e, row):
    return speech(
        e,
        {
            "input": row["text"],
            "voice": preset("voxtral_tts", row),
            "language": "English",
            "seed": 1234,
        },
    )


if __name__ == "__main__":
    main(load, synth)
