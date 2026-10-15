"""Step 1: the LLM labels the script (who reads each piece, emotion, delivery, pace, and which
spans spell out a sound), and describes the cast (for the preset-voice models).

The parser has already split every line (script.py); the LLM only labels the pieces. Its
answer is checked line by line: a line whose labels don't fit (wrong count, a vocals span that
isn't in the text, ...) falls back to plain labels (`script.default_line`) and is logged.
Results are cached by line text + LLM model in `direction-cache.json`, so an edit to the script
only asks again for the lines that changed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .llm import LLM, parse_json
from .script import (
    KIND_ASIDE,
    MODE_INNER,
    MODE_NARRATION,
    MODE_SPEECH,
    NARRATOR,
    Line,
    RawLine,
    Segment,
    Sound,
    Vocal,
    default_line,
    is_narrator,
    segment_from_dict,
)

EMOTIONS = (
    "neutral", "happy", "amused", "excited", "tender", "confident", "serious", "wistful", "sad",
    "nervous", "afraid", "tense", "angry", "exasperated", "sarcastic", "disgusted", "surprised",
    "shocked", "pained", "sleepy",
)  # fmt: skip
DELIVERIES = ("normal", "soft", "whisper", "loud", "shout")
PACES = ("slow", "normal", "fast")
# Examples offered to the LLM, not a limit: each engine's script step decides what it can do.
SOUNDS = (
    "laugh", "chuckle", "giggle", "sigh", "gasp", "groan", "moan", "cough", "clears throat",
    "sneeze", "hiccup", "burp", "gulp", "sniffle", "snort", "whimper", "sob", "cry", "pant",
    "hiss", "grunt", "scoff", "yawn", "scream", "slurp",
)  # fmt: skip
GENDERS = ("female", "male")
AGES = ("child", "adult", "older")
MAX_VOCAL_CHARS = 60
BATCH_LINES = 12
CONTEXT_LINES = 4
MAX_TOKENS = 8000

SYSTEM = (
    "You are the voice director of an audio drama. For every numbered segment you decide how it "
    "is performed. Respond with a single JSON object and nothing else."
)

RULES = f"""For every segment give: reader, emotion, intensity (0-1), delivery, pace, and optional
sounds (non-verbal noises like a sigh or laugh, at the start or end of the segment; sparingly).
Segments marked "aside" are *starred* text: decide whether the character says it as an inner
thought ("inner") or the Narrator reads it as description ("narration"). Third-person actions
("He raises an eyebrow.") are narration; first-person thoughts are inner. Other segments are
read as written: reader "speech" (or "narration" on a Narrator line).
Pain ("Ow!", "Ouch", "Hot, hot, hot!") is emotion "pained", not "surprised"; drowsy mumbling
is "sleepy".
Sounds belong to whoever makes them: narration never gets sounds or vocals. When an aside
describes a sound the speaker makes ("A long, loud slurp.", "She laughs."), keep the aside as
narration and also give the speaker's own speech segment next to it that sound (at "start" if
the aside comes before the speech, "end" if after).

Vocals: a voice engine reads letters aloud, so a sound spelled out as letters comes out garbled.
List every span of a segment that spells out a non-verbal sound instead of words, copied exactly
(same letters, case and punctuation), with the sound it is: "Ha! Ha ha ha!" -> laugh, "Hee
hee!" -> laugh, "Hnnngh" -> groan, "Hhhh!" -> sigh, "Achoo!" -> sneeze, "Hic!" -> hiccup,
"Hah-- hah--" (out of breath) -> pant, "Sss" (through the teeth) -> hiss, "Urrrp" -> burp,
"Shhhlrrp" -> slurp. A hum of tasting or delight, long or short ("Mmmmm", "Mmh!", "Mm.",
"Mmm-mm-MMM!") or a pleased "Ohhh" -> moan (a groan is pain or effort); "Mm-hmm" (yes) and
"Mm-mm" (no) are words; a satisfied "Ahhh" -> sigh; a drowsy "Mmmhh..." -> moan. A noise with
no name gets "none". Words used as a reflex noise count too: a yelp at a burn or a stubbed toe
("Ooh!", "Ah!", "Eek!", "Ow!") -> gasp or whimper; cries of pain that are still words ("Hot,
hot, hot, hot!", "Ow, ow, ow!") -> keep: true and the sound that goes with them (hiss, whimper,
groan), so the words are said through it. keep: true means the words stay and the sound comes
just before them; otherwise the sound replaces the span. Words said as words ("Whoa", "No",
"Uh-oh", "Psst", "Shh") are not vocals. Most segments have none.

Allowed emotions: {", ".join(EMOTIONS)}
Allowed deliveries: {", ".join(DELIVERIES)}; paces: {", ".join(PACES)}
Sounds (in "sounds" and "vocals"): any non-verbal sound in 1-3 plain words, e.g.
{", ".join(SOUNDS)}.

Return JSON: {{"lines": [{{"line": <line number>, "segments": [{{"n": 1, "reader": "...",
"emotion": "...", "intensity": 0.5, "delivery": "...", "pace": "...",
"sounds": [{{"kind": "sigh", "at": "end"}}],
"vocals": [{{"text": "<exact span>", "sound": "...", "keep": false}}]}}]}}]}}"""


def _block(raw: RawLine) -> str:
    out = [f"Line {raw.index + 1} ({raw.speaker}):"]
    for n, piece in enumerate(raw.pieces, start=1):
        kind = (
            "aside"
            if piece.kind == KIND_ASIDE
            else ("narration" if is_narrator(raw.speaker) else "speech")
        )
        out.append(f"  {n}. [{kind}] {piece.text}")
    return "\n".join(out)


def _prompt(batch: list[RawLine], context: list[RawLine], cast: str) -> str:
    ctx = "\n".join(f"{c.speaker}: {c.text}" for c in context) or "(start of the script)"
    blocks = "\n\n".join(_block(raw) for raw in batch)
    return f"""Cast: {cast}

Script so far (context only):
{ctx}

Direct these lines:

{blocks}

{RULES}"""


def _pick(value, allowed: tuple[str, ...], default: str) -> str:
    value = str(value or "").strip().lower()
    return value if value in allowed else default


def _sound_name(value) -> str | None:
    name = " ".join(str(value or "").strip().lower().split())
    return name if name and len(name.split()) <= 3 and len(name) <= 30 else None


def apply_labels(raw: RawLine, labels: list) -> Line | None:
    """The labelled line, or None when the labels don't fit the parsed pieces."""
    by_n = {}
    for label in labels:
        if isinstance(label, dict):
            try:
                by_n[int(label.get("n"))] = label
            except (TypeError, ValueError):
                return None
    if set(by_n) != set(range(1, len(raw.pieces) + 1)):
        return None
    segments = []
    for n, piece in enumerate(raw.pieces, start=1):
        label = by_n[n]
        if is_narrator(raw.speaker):
            voice, mode = NARRATOR, MODE_NARRATION
        elif piece.kind == KIND_ASIDE:
            inner = str(label.get("reader", "")).strip().lower() == MODE_INNER
            voice, mode = (raw.speaker, MODE_INNER) if inner else (NARRATOR, MODE_NARRATION)
        else:
            voice, mode = raw.speaker, MODE_SPEECH
        try:
            intensity = min(1.0, max(0.0, float(label.get("intensity", 0.5))))
        except (TypeError, ValueError):
            intensity = 0.5
        sounds = []
        for s in label.get("sounds") or []:
            if isinstance(s, dict) and (kind := _sound_name(s.get("kind"))):
                sounds.append(Sound(kind, "end" if s.get("at") == "end" else "start"))
        vocals = []
        for v in label.get("vocals") or []:
            if not isinstance(v, dict):
                return None
            span = str(v.get("text", ""))
            if not span.strip() or span not in piece.text or len(span) > MAX_VOCAL_CHARS:
                return None  # the one thing that must be exact: the span is in the text
            vocals.append(Vocal(span, _sound_name(v.get("sound")) or "none", bool(v.get("keep"))))
        segments.append(
            Segment(
                voice=voice,
                mode=mode,
                text=piece.text,
                emotion=_pick(label.get("emotion"), EMOTIONS, "neutral"),
                intensity=intensity,
                delivery=_pick(
                    label.get("delivery"), DELIVERIES, "soft" if mode == MODE_INNER else "normal"
                ),
                pace=_pick(label.get("pace"), PACES, "normal"),
                sounds=tuple(sounds),
                vocals=tuple(vocals),
            )
        )
    return Line(raw.index, raw.scene, raw.speaker, tuple(segments), "llm", raw.key)


def _line_from_cache(obj: dict, raw: RawLine) -> Line:
    return Line(
        raw.index,
        raw.scene,
        raw.speaker,
        tuple(segment_from_dict(s) for s in obj["segments"]),
        obj.get("source", "llm"),
        raw.key,
    )


def _prompt_version() -> str:
    return hashlib.sha1(f"{SYSTEM}|{RULES}".encode()).hexdigest()[:8]


def direct(raws: list[RawLine], llm: LLM, cast: dict, cache_path: Path, log) -> list[Line]:
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    version = _prompt_version()
    cast_text = "; ".join(f"{name} ({c['gender']}, {c['age']})" for name, c in cast.items())
    done: dict[int, Line] = {}
    todo = []
    for raw in raws:
        hit = cache.get(f"{raw.key}|{llm.model}|{version}")
        if hit:
            done[raw.index] = _line_from_cache(hit, raw)
        else:
            todo.append(raw)
    if todo:
        log(f"labelling {len(todo)} lines with {llm.describe()}")

    def ask(batch: list[RawLine]) -> list[RawLine]:
        """Labels a batch; returns the lines it didn't get right."""
        context = [r for r in raws if r.index < batch[0].index][-CONTEXT_LINES:]
        try:
            reply = parse_json(
                llm.complete(SYSTEM, _prompt(batch, context, cast_text), max_tokens=MAX_TOKENS)
            )
            got = {
                int(x.get("line", 0)) - 1: x.get("segments") or []
                for x in reply.get("lines", [])
                if isinstance(x, dict)
            }
        except Exception as exc:  # noqa: BLE001 - unreachable, bad JSON, ...: retried / defaults
            log(f"  labelling call failed: {exc}")
            got = {}
        missed = []
        for raw in batch:
            line = apply_labels(raw, got.get(raw.index, []))
            if line is None:
                missed.append(raw)
                continue
            done[raw.index] = line
            cache[f"{raw.key}|{llm.model}|{version}"] = asdict(line)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        return missed

    for start in range(0, len(todo), BATCH_LINES):
        missed = ask(todo[start : start + BATCH_LINES])
        for raw in missed:  # once more, alone
            if ask([raw]):
                log(f"  line {raw.index + 1}: labels didn't fit twice; plain labels used")
                done[raw.index] = default_line(raw)
    return [done[r.index] for r in raws]


# --- cast --------------------------------------------------------------------------------

CAST_SYSTEM = "You cast voices for an audio drama. Respond with a single JSON object only."


def cast_voices(raws: list[RawLine], names: list[str], llm: LLM, path: Path, log) -> dict:
    """{speaker: {gender, age, description}}: which preset voice fits, for models that can't
    clone, and the voice description for models that take one. Saved in cast.json; edit it to
    change a voice, delete it to ask again."""
    if path.exists():
        cast = json.loads(path.read_text(encoding="utf-8"))
        if set(cast) == set(names):
            return cast
    samples = {name: [] for name in names}
    for raw in raws:
        if raw.speaker in samples and len(samples[raw.speaker]) < 4:
            samples[raw.speaker].append(raw.text)
    prompt = (
        "For each speaker, give gender (female or male), age (child, adult or older) and a "
        "one-sentence description of their voice (pitch, warmth, energy), from their lines.\n\n"
        + "\n".join(
            f"{name}: " + " | ".join(lines or ["(narrates)"]) for name, lines in samples.items()
        )
        + '\n\nReturn JSON: {"speakers": {"<name>": {"gender": "...", "age": "...", '
        '"description": "..."}}}'
    )
    try:
        got = parse_json(llm.complete(CAST_SYSTEM, prompt, max_tokens=MAX_TOKENS)).get("speakers")
        got = got if isinstance(got, dict) else {}
    except Exception as exc:  # noqa: BLE001 - defaults below
        log(f"  cast call failed ({exc}); default voices")
        got = {}
    cast = {}
    for i, name in enumerate(names):
        c = got.get(name) if isinstance(got.get(name), dict) else {}
        cast[name] = {
            "gender": _pick(c.get("gender"), GENDERS, GENDERS[i % 2]),
            "age": _pick(c.get("age"), AGES, "adult"),
            "description": str(c.get("description") or "A clear, natural voice.").strip(),
        }
    path.write_text(json.dumps(cast, ensure_ascii=False, indent=1), encoding="utf-8")
    return cast


def voice_slots(cast: dict) -> dict[str, int]:
    """Each speaker's index among speakers of the same gender: which preset they get."""
    count = {g: 0 for g in GENDERS}
    slots = {}
    for name, c in cast.items():
        slots[name] = count[c["gender"]]
        count[c["gender"]] += 1
    return slots
