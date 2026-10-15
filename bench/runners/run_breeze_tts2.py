"""Breeze-TTS-2: clip + transcript + instruction, open (tag) sounds. Runs the official
inference code from the vendor/breeze-tts submodule: cfg 4.0 with an instruction, 1.0 without;
BREEZE_TTS_FAST off (plain eager runtime, no compile)."""

import sys
from pathlib import Path

from _base import ROOT, clip_path, main, ref_text

CODE = ROOT / "vendor" / "breeze-tts"
MODEL = "BreezeBlue/Breeze-TTS-2"
SEED = 1234


def load(device):
    sys.path.insert(0, str(CODE))
    from breeze_infer.runtime import load_runtime, update_generation_config_for_breeze
    from huggingface_hub import snapshot_download
    from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig

    tokenizer, model, audio_tokenizer = load_runtime(
        # breeze's runtime wants an indexed device ("cuda" fails: "Expected ... index")
        Path(snapshot_download(MODEL)),
        device="cuda:0" if device == "cuda" else device,
        attn_implementation="eager",
    )
    update_generation_config_for_breeze(model)
    runtime = FastBreezeStreamingRuntime(
        model,
        audio_tokenizer,
        FastStreamingConfig(max_new_tokens=1500, max_seq_len=2048, fast_all=False),
        tokenizer=tokenizer,
    )
    return {"tok": tokenizer, "model": model, "audio_tok": audio_tokenizer, "rt": runtime, "n": 0}


def synth(e, row):
    from breeze_infer.runtime import set_all_seeds
    from breeze_infer.templates import get_template, prepare_inputs, select_template_name

    e["n"] += 1
    request = {
        "id": f"seg-{e['n']}",
        "text": row["text"],
        "speaker": "S0",
        "ref_audio_path": clip_path(row),
        "ref_text": ref_text(row),
    }
    if row.get("instruction"):
        request["instruction"] = row["instruction"]
    cfg = 4.0 if row.get("instruction") else 1.0
    template = get_template(select_template_name(request))
    set_all_seeds(SEED)
    inputs = prepare_inputs(
        e["tok"],
        e["audio_tok"],
        e["model"],
        [request],
        template,
        guidance_scale=cfg,
        guidance_scale_ref=None,
        guidance_scale_ins=None,
    )
    import numpy as np

    chunks = [
        np.asarray(c.audio, dtype=np.float32).reshape(-1)
        for c in e["rt"].iter_audio_chunks(inputs, request_id=request["id"], seed=SEED)
    ]
    return np.concatenate(chunks), int(e["rt"].sample_rate)


if __name__ == "__main__":
    main(load, synth)
