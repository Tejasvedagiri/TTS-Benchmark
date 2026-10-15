"""Maya1 (maya-research/maya1) baseline: no cloning; the voice is a written description.
Each speaker keeps one fixed identity (their description in cast.json, written by the LLM), and
the LLM's per-line description is added as delivery, so voices don't drift line to line. Prompt
format, sampling and SNAC decoding as in the model card's Transformers example."""

from _base import main

CODE_START, CODE_END, CODE_OFFSET = 128257, 128258, 128266
SNAC_MIN, SNAC_MAX = 128266, 156937
SOH, EOH, SOA, TEXT_EOT = 128259, 128260, 128261, 128009


def load(device):
    import torch
    from snac import SNAC
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        "maya-research/maya1", torch_dtype=torch.bfloat16, device_map=device
    )
    tok = AutoTokenizer.from_pretrained("maya-research/maya1")
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)
    return {"model": model, "tok": tok, "snac": snac, "device": device}


def synth(e, row):
    import torch

    tok = e["tok"]
    description = row["voice_description"]
    if row.get("description"):
        description += f" Delivery: {row['description']}."
    prompt = (
        tok.decode([SOH])
        + tok.bos_token
        + f'<description="{description}"> {row["text"]}'
        + tok.decode([TEXT_EOT])
        + tok.decode([EOH])
        + tok.decode([SOA])
        + tok.decode([CODE_START])
    )
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(e["device"])
    torch.manual_seed(1234)
    with torch.inference_mode():
        out = e["model"].generate(
            **inputs,
            max_new_tokens=2048,
            min_new_tokens=28,
            temperature=0.4,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            eos_token_id=CODE_END,
            pad_token_id=tok.pad_token_id,
        )
    ids = out[0, inputs["input_ids"].shape[1] :].tolist()
    ids = ids[: ids.index(CODE_END)] if CODE_END in ids else ids
    codes = [t for t in ids if SNAC_MIN <= t <= SNAC_MAX]
    l1, l2, l3 = [], [], []
    for i in range(len(codes) // 7):
        s = [(c - CODE_OFFSET) % 4096 for c in codes[i * 7 : (i + 1) * 7]]
        l1.append(s[0])
        l2 += [s[1], s[4]]
        l3 += [s[2], s[3], s[5], s[6]]
    levels = [
        torch.tensor(x, dtype=torch.long, device=e["device"]).unsqueeze(0) for x in (l1, l2, l3)
    ]
    with torch.inference_mode():
        audio = e["snac"].decoder(e["snac"].quantizer.from_codes(levels))[0, 0]
    return audio.float().cpu().numpy()[2048:], 24000


if __name__ == "__main__":
    main(load, synth)
