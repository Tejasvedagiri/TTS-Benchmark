"""Step 2: the LLM writes each TTS model's version of the labelled script, in that model's own
dialect (its sound tags, emotion fields, style knobs), from the model's card (registry.py).

Output per model: `output/<test>/<model>/<test>.engine.txt` (to read) and `.engine.jsonl` (what
the renderer reads). Every reply is checked in code: the words must be the original words (minus
the spans the labels say are sounds), tags only from the model's list, fields in range. A
segment that fails is asked again alone, then falls back to plain text with its tags dropped.
Models with no tags and no free-text fields need no LLM call at all. Results are cached by
segment + card + LLM model in `engine-cache.json`.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

from .llm import LLM, parse_json
from .registry import ModelCard
from .script import Line, Segment, scenes, segment_id

SCENE_MAX_TOKENS = 8000  # hidden reasoning counts against this
RETRY_MAX_TOKENS = 6000

# A sample of the tag shape for open-vocabulary models (the card lists no tags to look at).
OPEN_TAG_SAMPLE = {
    "echo_tts": "(laughs)",
    "fish_s2": "[laughing]",
    "breeze_tts2": "(laugh)",
    "drama3": "[laughs]",
}


def _tag_pattern(card: ModelCard) -> str | None:
    sample = None
    if isinstance(card.tags, list) and card.tags:
        sample = card.tags[0]
    elif card.tags == "open":
        sample = OPEN_TAG_SAMPLE.get(card.id)
    if not sample:
        return None
    if sample.startswith("<|"):
        return r"<\|[^|]*\|>"
    if sample.startswith("["):
        return r"\[[^\]]*\]"
    if sample.startswith("("):
        return r"\([^)]*\)"
    if sample.startswith("<"):
        return r"<[^>]*>"
    return None


def _words_of(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def _without_dropped_vocals(seg: Segment, drop=None) -> str:
    text = seg.text
    for vocal in sorted(drop if drop is not None else seg.vocals, key=lambda v: -len(v.text)):
        if not vocal.keep:
            text = text.replace(vocal.text, " ")
    return text


def _optional_words(seg: Segment) -> list[list[str]]:
    """Every word list with each keep=false span either removed or kept (for cards with
    sound_words_after_tag)."""
    drop = [v for v in seg.vocals if not v.keep]
    options = []
    for mask in range(2 ** min(len(drop), 6)):
        chosen = [v for i, v in enumerate(drop) if mask >> i & 1]
        options.append(_words_of(_without_dropped_vocals(seg, chosen)))
    return options


def _strip_tags(text: str, card: ModelCard) -> tuple[str, set[str]]:
    """(text with recognized tags removed, every tag-shaped token found)."""
    pattern = _tag_pattern(card)
    if pattern is None:
        return text, set()
    found = set(re.findall(pattern, text))
    if isinstance(card.tags, list):
        stripped = re.sub(pattern, lambda m: " " if m.group(0) in card.tags else m.group(0), text)
    else:  # open vocabulary: anything tag-shaped is a tag
        stripped = re.sub(pattern, " ", text)
    return stripped, found


_RANGE_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)")


def _validate_fields(card: ModelCard, result: dict) -> list[str]:
    errors = []
    for name, kind, _desc in card.style_keys:
        if name not in result:
            continue
        val = result[name]
        match = _RANGE_RE.search(kind)
        if match and "float" in kind:
            lo, hi = float(match.group(1)), float(match.group(2))
            if "floats" in kind:
                ok = isinstance(val, list) and all(
                    isinstance(x, (int, float)) and lo <= x <= hi for x in val
                )
            else:
                ok = isinstance(val, (int, float)) and lo <= val <= hi
            if not ok:
                errors.append(f"{name}: expected {kind}, got {val!r}")
        elif "string" in kind and not isinstance(val, str):
            errors.append(f"{name}: expected string, got {val!r}")
    return errors


def validate_item(seg: Segment, card: ModelCard, item: dict | None) -> list[str]:
    if item is None:
        return ["no result"]
    if not isinstance(item.get("text"), str):
        return ["missing/non-string text"]
    stripped, found = _strip_tags(item["text"], card)
    got, want = _words_of(stripped), _words_of(_without_dropped_vocals(seg))
    errors = []
    if got != want and not (card.sound_words_after_tag and got in _optional_words(seg)):
        errors.append(f"words differ: got {got} want {want}")
    if isinstance(card.tags, list):
        unknown = found - set(card.tags)
        if unknown:
            errors.append(f"tags not in {card.id}'s list: {sorted(unknown)}")
    errors.extend(_validate_fields(card, item))
    return errors


def fallback_result(seg: Segment) -> dict:
    return {"text": re.sub(r"\s+", " ", _without_dropped_vocals(seg)).strip()}


# --- knobs for no-tag models, mapped in code from the labels ------------------------------


def _chatterbox_knobs(seg: Segment) -> dict:
    return {
        "exaggeration": round(0.3 + 0.7 * seg.intensity, 2),
        "cfg_weight": 0.3 if seg.pace == "slow" or seg.delivery in ("soft", "whisper") else 0.5,
    }


_ZONOS_EMOTION_IDX = {
    "happy": 0, "amused": 0, "excited": 0, "tender": 0, "confident": 0,
    "sad": 1, "wistful": 1, "sleepy": 1,
    "disgusted": 2,
    "afraid": 3, "nervous": 3, "tense": 3, "pained": 3,
    "surprised": 4, "shocked": 4,
    "angry": 5, "exasperated": 5, "sarcastic": 5,
}  # fmt: skip


def _zonos_knobs(seg: Segment) -> dict:
    vec = [0.0] * 8
    idx = _ZONOS_EMOTION_IDX.get(seg.emotion)
    if idx is None:
        vec[7] = 1.0
    else:
        vec[idx] = min(1.0, 0.4 + seg.intensity)
        vec[7] = max(0.0, 1.0 - vec[idx])
    return {
        "emotion_vector": [round(x, 2) for x in vec],
        "speaking_rate": {"slow": 0.85, "fast": 1.15}.get(seg.pace, 1.0),
        "pitch_shift": 0.0,
    }


def _styletts2_knobs(seg: Segment) -> dict:
    return {"style_scale": round(0.5 + 0.5 * seg.intensity, 2)}


NO_TAG_STYLE_FN = {
    "chatterbox": _chatterbox_knobs,
    "zonos": _zonos_knobs,
    "styletts2": _styletts2_knobs,
}


def needs_llm(card: ModelCard) -> bool:
    """Tags to place, or free-text fields no code mapping fills."""
    return card.tags is not None or bool(card.style_keys and card.id not in NO_TAG_STYLE_FN)


# --- prompts -------------------------------------------------------------------------------

SYSTEM_PROMPT = """You convert labelled story segments into the exact text a specific \
text-to-speech model should receive. You never invent new sounds or emotions: every \
segment already says what sound or delivery it needs; your only job is writing that in \
the target model's own syntax. A tag is never a word a voice would read aloud -- it \
either becomes the model's own symbol for that sound, or (if the model can't perform it) \
the span is dropped. Reply with ONLY JSON, no prose, no markdown fences."""


def _card_description(card: ModelCard) -> str:
    lines = [f"Model: {card.id} ({card.repo}, {card.size})"]
    if card.tags is None:
        lines.append("Tags: none. This model cannot perform sounds -- never add a tag.")
    elif card.tags == "open":
        lines.append(f"Tags: open vocabulary. {card.tag_notes}")
    else:
        lines.append(f"Tags (use ONLY these, exact spelling): {', '.join(card.tags)}")
        if card.tag_notes:
            lines.append(f"Tag notes: {card.tag_notes}")
    for name, kind, desc in card.style_keys:
        lines.append(f'Field "{name}" ({kind}): {desc}')
    return "\n".join(lines)


def _segment_payload(sid: str, seg: Segment) -> dict:
    return {
        "id": sid,
        "voice": seg.voice,
        "mode": seg.mode,
        "text": seg.text,
        "emotion": seg.emotion,
        "intensity": seg.intensity,
        "delivery": seg.delivery,
        "pace": seg.pace,
        "sounds": [{"kind": s.kind, "at": s.at} for s in seg.sounds],
        "vocals": [{"text": v.text, "sound": v.sound, "keep": v.keep} for v in seg.vocals],
    }


def _extra_fields(card: ModelCard) -> str:
    names = ", ".join(f'"{n}"' for n, _, _ in card.style_keys)
    return f", plus {names}" if names else ""


def build_scene_prompt(card: ModelCard, items: list[tuple[str, Segment]]) -> str:
    """One call per scene: a thinking model's reasoning is paid once for the whole scene."""
    payload = [_segment_payload(sid, seg) for sid, seg in items]
    return f"""{_card_description(card)}

Segments, in order:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Return a JSON array, one object per segment, SAME order, each object: \
{{"id": "...", "text": "..."{_extra_fields(card)}, "note": "..."}}.
Rules:
- "text" must contain every word of the segment's original "text", in order, except any \
"vocals" span with keep=false (that span is replaced by the sound, so its words are gone) \
-- a span with keep=true keeps its words and only gets the tag placed just before them.
- "sounds" (if any) are extra sounds at the very start or end of the segment, not inside the text.
- Use intensity/delivery/pace/emotion only to choose which tag or field value fits; never echo \
those labels back as text.
- "note": a short string ONLY when you had to drop or approximate something; otherwise "".
- Omit any field this model doesn't have."""


def build_retry_prompt(card: ModelCard, sid: str, seg: Segment, errors: list[str]) -> str:
    return f"""{_card_description(card)}

Your previous answer for this one segment had problems: {"; ".join(errors)}

Segment:
{json.dumps(_segment_payload(sid, seg), ensure_ascii=False, indent=2)}

Return a JSON object (not an array): \
{{"id": "{sid}", "text": "..."{_extra_fields(card)}, "note": "..."}}, fixing those problems. \
Same rules as before: every original word must appear (except a dropped vocals span), tags \
only from this model's own list, fields in range."""


# --- driver --------------------------------------------------------------------------------


def _cache_key(card: ModelCard, seg: Segment, llm: LLM) -> str:
    card_key = repr((card.tags, card.tag_notes, card.style_keys, SYSTEM_PROMPT))
    seg_key = json.dumps(_segment_payload("", seg), sort_keys=True)
    digest = hashlib.sha1(f"{card_key}|{seg_key}".encode()).hexdigest()[:20]
    return f"{digest}|{llm.model}"


def write_for_model(
    card: ModelCard, lines: list[Line], llm: LLM, model_dir: Path, log
) -> dict[str, dict]:
    model_dir.mkdir(parents=True, exist_ok=True)
    cache_path = model_dir / "engine-cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    results: dict[str, dict] = {}
    for scene_no, scene_lines in enumerate(scenes(lines)):
        items = [
            (segment_id(ln, i), seg) for ln in scene_lines for i, seg in enumerate(ln.segments)
        ]
        if not needs_llm(card):
            style_fn = NO_TAG_STYLE_FN.get(card.id)
            for sid, seg in items:
                results[sid] = {**fallback_result(seg), **(style_fn(seg) if style_fn else {})}
            continue
        to_call = []
        for sid, seg in items:
            key = _cache_key(card, seg, llm)
            if key in cache:
                results[sid] = cache[key]
            else:
                to_call.append((sid, seg, key))
        if not to_call:
            continue
        log(f"  {card.id}: scene {scene_no + 1}, {len(to_call)} segments -> LLM")
        started = time.perf_counter()
        try:
            raw = llm.complete(
                SYSTEM_PROMPT,
                build_scene_prompt(card, [(s, g) for s, g, _ in to_call]),
                max_tokens=SCENE_MAX_TOKENS,
            )
            replies = {
                r["id"]: r for r in parse_json(raw, list) if isinstance(r, dict) and "id" in r
            }
        except Exception as exc:  # noqa: BLE001 - each segment is asked alone below
            log(f"    scene call failed ({exc}); asking each segment on its own")
            replies = {}
        log(f"    replied in {time.perf_counter() - started:.0f}s")
        for sid, seg, key in to_call:
            item = replies.get(sid)
            errors = validate_item(seg, card, item)
            if errors:
                try:
                    item = parse_json(
                        llm.complete(
                            SYSTEM_PROMPT,
                            build_retry_prompt(card, sid, seg, errors),
                            max_tokens=RETRY_MAX_TOKENS,
                        )
                    )
                    errors = validate_item(seg, card, item)
                except Exception as exc:  # noqa: BLE001
                    errors = [str(exc)]
            if errors:
                final = fallback_result(seg)
                final["note"] = f"fallback (plain text, tags dropped): {'; '.join(errors)}"
                log(f"    {sid}: plain text ({errors[0][:120]})")
            else:
                final = {**item, "note": item.get("note", "")}
            results[sid] = final
            cache[key] = final
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return results


def write_engine_script(
    card: ModelCard,
    lines: list[Line],
    results: dict[str, dict],
    voices: dict[str, dict],
    model_dir: Path,
    test: str,
) -> list[dict]:
    """`<test>.engine.txt` + `.engine.jsonl`; returns the rows."""
    txt = [f"# {test}.engine.txt -- {card.id}: {card.repo} ({card.size}, {card.licence})", ""]
    rows = []
    for ln in lines:
        for i, seg in enumerate(ln.segments):
            sid = segment_id(ln, i)
            result = results.get(sid, fallback_result(seg))
            voice = voices[seg.voice]
            row = {
                "id": sid,
                "line": ln.index,
                "scene": ln.scene,
                "voice": seg.voice,
                "voice_clip": voice["clip"],
                "voice_gender": voice["gender"],
                "voice_slot": voice["slot"],
                "voice_description": voice["description"],
                "text": card.speaker_prefix + result.get("text", ""),
            }
            if card.needs_transcript:
                row["ref_text"] = voice["transcript"]
            extra = []
            for name, _kind, _desc in card.style_keys:
                if name in result:
                    row[name] = result[name]
                    extra.append(f"{name}: {result[name]}")
            if result.get("note"):
                row["note"] = result["note"]
                extra.append(f"note: {result['note']}")
            rows.append(row)
            tail = f"  || {' | '.join(extra)}" if extra else ""
            txt.append(f"[{seg.voice} · {seg.mode}] {row['text']}{tail}")
        txt.append("")
    (model_dir / f"{test}.engine.txt").write_text("\n".join(txt), encoding="utf-8")
    with (model_dir / f"{test}.engine.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows
