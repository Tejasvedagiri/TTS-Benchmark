"""T5Gemma-TTS 2b-2b (Aratako/T5Gemma-TTS-2b-2b, 5.3B, gated): clip + transcript, no tags. The
repo's inference_commandline_hf.py split into load-once / per-segment, same defaults (top_k 30,
top_p 0.9, temperature 0.8, stop_repetition 3, duration estimated from the text and the clip).
Code from envs/t5gemma_tts/src. Sample rate = its XCodec2's."""

import sys

from _base import ROOT, clip_path, main, ref_text

sys.path.insert(0, str(ROOT / "envs" / "t5gemma_tts" / "src"))
REPO = "Aratako/T5Gemma-TTS-2b-2b"


def load(device):
    import torch
    from data.tokenizer import AudioTokenizer
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model = (
        AutoModelForSeq2SeqLM.from_pretrained(REPO, trust_remote_code=True, dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    cfg = model.config
    name = getattr(cfg, "text_tokenizer_name", None) or getattr(cfg, "t5gemma_model_name", None)
    audio_tok = AudioTokenizer(
        backend="xcodec2", model_name=getattr(cfg, "xcodec2_model_name", None)
    )
    return {
        "model": model,
        "cfg": cfg,
        "text_tok": AutoTokenizer.from_pretrained(name),
        "audio_tok": audio_tok,
        "device": device,
    }


def synth(e, row):
    from duration_estimator import estimate_duration
    from inference_commandline_hf import seed_everything
    from inference_tts_utils import (
        get_audio_info,
        get_sample_rate,
        inference_one_sample,
        normalize_text_with_lang,
    )

    seed_everything(1234)
    clip = clip_path(row)
    text, lang = normalize_text_with_lang(row["text"], None)
    prefix, _ = normalize_text_with_lang(ref_text(row), lang)
    rate = e["audio_tok"].sample_rate
    decode = {
        "top_k": 30,
        "top_p": 0.9,
        "min_p": 0,
        "temperature": 0.8,
        "stop_repetition": 3,
        "codec_audio_sr": rate,
        "codec_sr": getattr(e["cfg"], "encodec_sr", 50),
        "silence_tokens": [],
        "sample_batch_size": 1,
    }
    _, gen = inference_one_sample(
        model=e["model"],
        model_args=e["cfg"],
        text_tokenizer=e["text_tok"],
        audio_tokenizer=e["audio_tok"],
        audio_fn=clip,
        target_text=text,
        lang=lang,
        device=e["device"],
        decode_config=decode,
        prompt_end_frame=int(100 * get_sample_rate(get_audio_info(clip))),
        target_generation_length=estimate_duration(
            target_text=text,
            reference_speech=clip,
            reference_transcript=prefix,
            target_lang=lang,
            reference_lang=lang,
        ),
        prefix_transcript=prefix,
        multi_trial=[],
        repeat_prompt=0,
    )
    return gen[0].float().cpu().numpy().reshape(-1), rate


if __name__ == "__main__":
    main(load, synth)
