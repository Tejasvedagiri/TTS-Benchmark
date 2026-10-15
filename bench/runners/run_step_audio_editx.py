"""Step-Audio-EditX (stepfun-ai/Step-Audio-EditX, 3B, vLLM): clip + transcript. A segment with
paralinguistic tags is made in upstream's two steps (tts_infer.py's infer_test): clone the
plain words (the tags, exact strings from the card's list, taken out), then a "paralinguistic"
edit of that audio with the tagged text. Untagged segments are the clone alone. The repo's own
code and lock (envs/step_audio_editx/src, `uv sync --frozen`). 24 kHz."""

import os
import sys
import tempfile
from pathlib import Path

from _base import ROOT, clip_path, main, ref_text

os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")  # as tts_infer.py
SRC = ROOT / "envs" / "step_audio_editx" / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "tool"))
TAGS = None


def load(device):
    global TAGS
    from adapters.registry import ALL_MODELS
    from huggingface_hub import snapshot_download
    from tokenizer import StepAudioTokenizer
    from tts import StepAudioTTS

    TAGS = sorted(ALL_MODELS["step_audio_editx"].tags, key=len, reverse=True)
    tokenizer = StepAudioTokenizer(
        snapshot_download("stepfun-ai/Step-Audio-Tokenizer"), model_source="local"
    )
    os.chdir(SRC)  # upstream resolves some relative paths (config/) from the repo root
    return StepAudioTTS(
        snapshot_download("stepfun-ai/Step-Audio-EditX"),
        tokenizer,
        model_source="local",
        gpu_memory_utilization=0.6,
        max_model_len=8192,
        max_num_seqs=1,
        cosyvoice_dtype="bfloat16",
    )


def synth(model, row):
    import soundfile as sf

    text = row["text"]
    plain = text
    for tag in TAGS:
        plain = plain.replace(tag, " ")
    plain = " ".join(plain.split())
    if not plain:
        raise RuntimeError("sound-only segment: nothing to clone before the edit")
    audio, rate = model.clone(
        prompt_wav_path=clip_path(row), prompt_text=ref_text(row), target_text=plain
    )
    if plain != " ".join(text.split()):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "clone.wav"
            sf.write(str(first), audio.float().cpu().numpy().reshape(-1), rate)
            audio, rate = model.edit(
                prompt_wav_path=str(first),
                prompt_text=plain,
                edit_type="paralinguistic",
                target_text=text,
            )
    return audio.float().cpu().numpy().reshape(-1), rate


if __name__ == "__main__":
    main(load, synth)
