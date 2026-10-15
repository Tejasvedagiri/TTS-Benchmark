"""Llasa-8B (HKUSTAudio/Llasa-8B): clones by continuation, as the model card's "speech prompt"
example: transcript + segment text as the text, the clip's XCodec2 codes as the start of the
speech, and only the newly generated codes decoded. XCodec2 is 16 kHz only, so the 24 kHz clip
is resampled first. bf16 (the card loads fp32: 32 GB). 16 kHz."""

from _base import clip_path, main, ref_text

REPO = "HKUSTAudio/Llasa-8B"
CODEC = "HKUSTAudio/xcodec2"


def load(device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from xcodec2.modeling_xcodec2 import XCodec2Model

    tok = AutoTokenizer.from_pretrained(REPO)
    model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.bfloat16).eval().to(device)
    codec = XCodec2Model.from_pretrained(CODEC).eval().to(device)
    return {"tok": tok, "model": model, "codec": codec, "device": device, "codes": {}}


def _clip_codes(e, row):
    import numpy as np
    import soundfile as sf
    import torch

    codes = e["codes"].get(row["voice"])
    if codes is None:
        audio, rate = sf.read(clip_path(row), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        n = int(round(audio.size * 16000 / rate))
        audio = np.interp(np.linspace(0, audio.size - 1, n), np.arange(audio.size), audio)
        wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            codes = e["codes"][row["voice"]] = e["codec"].encode_code(input_waveform=wav)[0, 0, :]
    return codes


def synth(e, row):
    import torch

    tok = e["tok"]
    prefix = "".join(f"<|s_{int(c)}|>" for c in _clip_codes(e, row))
    text = f"<|TEXT_UNDERSTANDING_START|>{ref_text(row)} {row['text']}<|TEXT_UNDERSTANDING_END|>"
    chat = [
        {"role": "user", "content": "Convert the text to speech:" + text},
        {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>" + prefix},
    ]
    ids = tok.apply_chat_template(
        chat, tokenize=True, return_tensors="pt", continue_final_message=True
    ).to(e["device"])
    end = tok.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
    torch.manual_seed(1234)
    with torch.no_grad():
        out = e["model"].generate(
            ids, max_length=2048, eos_token_id=end, do_sample=True, top_p=1, temperature=0.8
        )
    new = out[0][ids.shape[1] :]
    if new.numel() and new[-1] == end:
        new = new[:-1]
    speech = []
    for token in tok.batch_decode(new, skip_special_tokens=True):
        if token.startswith("<|s_") and token.endswith("|>"):
            speech.append(int(token[4:-2]))
    if not speech:
        raise RuntimeError("no speech tokens generated")
    with torch.no_grad():
        wav = e["codec"].decode_code(torch.tensor(speech, device=e["device"])[None, None, :])
    return wav[0, 0, :].float().cpu().numpy(), 16000


if __name__ == "__main__":
    main(load, synth)
