"""Fish Audio S2 Pro (fishaudio/s2-pro): clip + transcript as an in-context reference, free-form
"[verb]" tags inline. The repo's own ModelManager (what tools/api_server.py runs: S2 model +
codec.pth, warm-up), from a checkout in envs/fish_s2/src; one ServeTTSRequest per segment. No
torch.compile. 44.1 kHz."""

import sys
from pathlib import Path

from _base import ROOT, clip_path, main, ref_text

SRC = ROOT / "envs" / "fish_s2" / "src"
sys.path.insert(0, str(SRC))
REPO = "fishaudio/s2-pro"


def load(device):
    from huggingface_hub import snapshot_download
    from tools.server.model_manager import ModelManager

    ckpt = snapshot_download(REPO)
    manager = ModelManager(
        mode="tts",
        device=device,
        half=False,
        compile=False,
        llama_checkpoint_path=ckpt,
        decoder_checkpoint_path=None,
        decoder_config_name="modded_dac_vq",
    )
    return {"engine": manager.tts_inference_engine, "refs": {}}


def synth(e, row):
    from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

    ref = e["refs"].get(row["voice"])
    if ref is None:
        ref = e["refs"][row["voice"]] = ServeReferenceAudio(
            audio=Path(clip_path(row)).read_bytes(), text=ref_text(row)
        )
    request = ServeTTSRequest(text=row["text"], references=[ref], seed=1234, format="wav")
    final = None
    for result in e["engine"].inference(request):
        if result.code == "error":
            raise RuntimeError(str(result.error))
        if result.code == "final":
            final = result.audio
    if final is None:
        raise RuntimeError("no final audio")
    rate, audio = final
    return audio, rate


if __name__ == "__main__":
    main(load, synth)
