"""Fish Audio Drama 3 (fishaudio/drama-3-int4: Fish's int4 build of the 35B MoE, 3B active,
~21 GB): clip + transcript as a reference, free-form [sighs] / [laughs] point tags inline.
Served by SGLang-Omni (`sgl-omni serve`, its Drama 3 cookbook). Peak memory = nvidia-smi
(includes the desktop's ~1 GB). 44.1 kHz."""

from _base import clip_path, main, ref_text
from _vllm_omni import speech, start

REPO = "fishaudio/drama-3-int4"


def load(device):
    # 32 GB card: the vocoder's default 0.10 share holds 624 frames, fewer than the engine's
    # 2048-frame cap; 600 is Fish's own per-chunk default (~28 s, longer than any line here)
    # and one request at a time here: 16 parallel slots' hybrid-attention state didn't fit
    return start(
        REPO,
        (
            "--tts_engine.factory.max_new_tokens",
            "600",
            "--tts_engine.factory.max_running_requests",
            "4",
        ),
        server="sglang",
    )


def synth(e, row):
    # no "seed": SGLang-Omni's Drama 3 rejects it (HTTP 400 "seed is not supported")
    return speech(
        e,
        {
            "input": row["text"],
            "references": [{"audio_path": clip_path(row), "text": ref_text(row)}],
        },
    )


if __name__ == "__main__":
    main(load, synth)
