"""Orpheus 3B pretrained (canopylabs/orpheus-3b-0.1-pretrained): zero-shot cloning by in-context
example, per the Orpheus README ("generate speech conditioned on one or more existing
text-speech pairs in the prompt"; the model wasn't trained for cloning, so one 7-8 s pair per
speaker is the weak case). Prompt = [clip transcript turn + its SNAC codes], then the segment's
text turn, then generate; token layout from orpheus_tts/engine_class.py. Orpheus's tags are kept
in the text (the finetuned model's script) to hear whether the base model performs them. SNAC
encode/decode and sampling as run_orpheus.py. 24 kHz."""

from _base import clip_path, main, ref_text
from run_orpheus import CODE_END, CODE_OFFSET, CODE_START, END_HUMAN, END_TEXT, START, decode

REPO = "canopylabs/orpheus-3b-0.1-pretrained"
START_AI, END_AI = 128261, 128262


def load(device):
    import torch
    from snac import SNAC
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        REPO, torch_dtype=torch.bfloat16, device_map=device
    )
    tok = AutoTokenizer.from_pretrained(REPO)
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)
    return {"model": model, "tok": tok, "snac": snac, "device": device, "refs": {}}


def _clip_tokens(e, row) -> list[int]:
    """The clip as Orpheus audio tokens: 7 per SNAC frame, each code offset by its position."""
    import soundfile as sf
    import torch

    tokens = e["refs"].get(row["voice"])
    if tokens is None:
        audio, rate = sf.read(clip_path(row), dtype="float32", always_2d=True)
        if rate != 24000:
            raise RuntimeError(f"clip is {rate} Hz; SNAC wants 24 kHz")
        wav = torch.from_numpy(audio.mean(axis=1)).to(e["device"])[None, None, :]
        with torch.inference_mode():
            l1, l2, l3 = (c[0].tolist() for c in e["snac"].encode(wav))
        tokens = []
        for i in range(len(l1)):
            frame = [
                l1[i],
                l2[2 * i],
                l3[4 * i],
                l3[4 * i + 1],
                l2[2 * i + 1],
                l3[4 * i + 2],
                l3[4 * i + 3],
            ]
            tokens += [c + CODE_OFFSET + k * 4096 for k, c in enumerate(frame)]
        e["refs"][row["voice"]] = tokens
    return tokens


def _turn(tok, text: str) -> list[int]:
    return [START, *tok(text).input_ids, END_TEXT, END_HUMAN, START_AI, CODE_START]


def synth(e, row):
    import torch

    tok = e["tok"]
    prompt = [
        *_turn(tok, ref_text(row)),
        *_clip_tokens(e, row),
        CODE_END,
        END_AI,
        *_turn(tok, row["text"]),
    ]
    ids = torch.tensor([prompt], device=e["device"])
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
    gen = [CODE_START, *out[0, ids.shape[1] :].tolist()]  # decode() keeps what follows it
    return decode(e, gen), 24000


if __name__ == "__main__":
    main(load, synth)
