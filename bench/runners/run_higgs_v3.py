"""Higgs TTS 3 (bosonai/higgs-tts-3-4b, formerly higgs-audio-v3-tts-4b): clip + transcript as
ref_audio / ref_text, served by vLLM-Omni. Control tokens (<|emotion:…|>, <|sfx:…|>) are inline
in the text; the card's `emotion` field goes first as <|emotion:NAME|> when it is one of the
model's 21 emotions (anything else, e.g. "neutral", is left out). Peak memory = nvidia-smi
(includes the desktop's ~1 GB). 24 kHz."""

from _base import clip_path, main, ref_text
from _vllm_omni import data_url, speech, start

REPO = "bosonai/higgs-tts-3-4b"
EMOTIONS = {
    "elation",
    "amusement",
    "enthusiasm",
    "determination",
    "pride",
    "contentment",
    "affection",
    "relief",
    "contemplation",
    "confusion",
    "surprise",
    "awe",
    "longing",
    "arousal",
    "anger",
    "fear",
    "disgust",
    "bitterness",
    "sadness",
    "shame",
    "helplessness",
}


def load(device):
    # the deploy config splits the GPU between its two stages; a global utilization flag gave
    # stage 0 85% and OOM'd the codec. Context = the model's trained 8192 (config says 65536).
    engine = start(REPO, ("--max-model-len", "8192"))
    engine["refs"] = {}
    return engine


def synth(e, row):
    ref = e["refs"].get(row["voice"])
    if ref is None:
        ref = e["refs"][row["voice"]] = data_url(clip_path(row))
    text = row["text"]
    if row.get("emotion") in EMOTIONS:
        text = f"<|emotion:{row['emotion']}|>{text}"
    return speech(e, {"input": text, "ref_audio": ref, "ref_text": ref_text(row), "seed": 1234})


if __name__ == "__main__":
    main(load, synth)
