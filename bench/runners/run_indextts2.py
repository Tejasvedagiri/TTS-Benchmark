"""IndexTTS-2 (IndexTeam/IndexTTS-2): clip only; the story model's emotion text steers delivery
(use_emo_text + emo_text, emo_alpha 0.6 as upstream's example). infer() writes a file."""

import tempfile
from pathlib import Path

from _base import clip_path, main


def load(device):
    from huggingface_hub import snapshot_download
    from indextts.infer_v2 import IndexTTS2

    ckpt = Path(snapshot_download("IndexTeam/IndexTTS-2"))
    return IndexTTS2(
        cfg_path=str(ckpt / "config.yaml"),
        model_dir=str(ckpt),
        use_fp16=True,
        use_cuda_kernel=False,
        use_deepspeed=False,
    )


def synth(tts, row):
    import soundfile as sf

    kwargs = {}
    if row.get("emotion_text"):
        kwargs = {"use_emo_text": True, "emo_text": row["emotion_text"], "emo_alpha": 0.6}
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.wav"
        tts.infer(
            spk_audio_prompt=clip_path(row),
            text=row["text"],
            output_path=str(out),
            use_random=False,
            verbose=False,
            **kwargs,
        )
        audio, rate = sf.read(str(out), dtype="float32")
    return audio, rate


if __name__ == "__main__":
    main(load, synth)
