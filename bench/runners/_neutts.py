"""NeuTTS Air / Nano (neuphonic/neutts-*): clip + transcript (required), no tags. 24 kHz."""

import _espeak  # noqa: F401 - before neutts (phonemizer)
from _base import clip_path, main, ref_text


def run(backbone: str) -> None:
    def load(device):
        from neutts import NeuTTS

        tts = NeuTTS(
            backbone_repo=backbone,
            backbone_device=device,
            codec_repo="neuphonic/neucodec",
            codec_device=device,
        )
        return {"tts": tts, "refs": {}}

    def synth(e, row):
        codes = e["refs"].get(row["voice"])
        if codes is None:
            codes = e["refs"][row["voice"]] = e["tts"].encode_reference(clip_path(row))
        return e["tts"].infer(row["text"], codes, ref_text(row)), 24000

    main(load, synth)
