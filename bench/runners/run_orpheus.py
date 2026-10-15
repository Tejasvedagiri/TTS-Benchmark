"""Orpheus 3B (canopylabs/orpheus-3b-0.1-ft, gated) baseline: preset voices, no cloning, its 8
tags (<laugh> <sigh> ...) inline. Transformers + SNAC like run_maya1.py (the same Llama + SNAC
token scheme), not the orpheus-speech package, which pins a vLLM too old for the 5090. Prompt
"<voice>: <text>" between the start/end tokens, sampling and decoding as in the model's
official Colab. Each speaker gets a preset of their gender (_presets.py). 24 kHz."""

from _base import main
from _presets import preset

REPO = "canopylabs/orpheus-3b-0.1-ft"
START, END_TEXT, END_HUMAN = 128259, 128009, 128260
CODE_START, CODE_END, CODE_OFFSET = 128257, 128258, 128266


def load(device):
    import torch
    from snac import SNAC
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        REPO, torch_dtype=torch.bfloat16, device_map=device
    )
    tok = AutoTokenizer.from_pretrained(REPO)
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)
    return {"model": model, "tok": tok, "snac": snac, "device": device}


def synth(e, row):
    import torch

    ids = e["tok"](f"{preset('orpheus', row)}: {row['text']}", return_tensors="pt").input_ids
    ids = torch.cat([torch.tensor([[START]]), ids, torch.tensor([[END_TEXT, END_HUMAN]])], dim=1)
    ids = ids.to(e["device"])
    torch.manual_seed(1234)
    with torch.inference_mode():
        out = e["model"].generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.1,
            eos_token_id=CODE_END,
            pad_token_id=CODE_END,
        )
    return decode(e, out[0, ids.shape[1] :].tolist()), 24000


def decode(e, gen):
    """Generated ids -> 24 kHz audio (the Colab's SNAC unpacking), shared with run_orpheus_clone."""
    import torch

    if CODE_START in gen:  # keep what follows the last start-of-audio token
        gen = gen[len(gen) - gen[::-1].index(CODE_START) :]
    codes = [t - CODE_OFFSET for t in gen if t >= CODE_OFFSET]
    l1, l2, l3 = [], [], []
    for i in range(len(codes) // 7):
        s = [c % 4096 for c in codes[i * 7 : (i + 1) * 7]]
        l1.append(s[0])
        l2 += [s[1], s[4]]
        l3 += [s[2], s[3], s[5], s[6]]
    if not l1:
        raise RuntimeError("no audio codes generated")
    levels = [
        torch.tensor(x, dtype=torch.long, device=e["device"]).unsqueeze(0) for x in (l1, l2, l3)
    ]
    with torch.inference_mode():
        audio = e["snac"].decode(levels)[0, 0]
    return audio.float().cpu().numpy()


if __name__ == "__main__":
    main(load, synth)
