"""The test script: parse `input/<test>.txt` into lines and segments, and the labelled form
(`direction.jsonl`) that every later step reads.

Format, one line per spoken line, blank lines between scenes, `#` for comments:

    Narrator: Saturday, one in the afternoon. Two bowls of soup sit on the table.
    Mom: *She closes her eyes and sips.* "Mmmmmm. Mmm. Wow. This... is amazing."
    Leo: Ten out of ten!

Quoted text ("..." or “...”) is the speaker's speech, *starred* text is an aside (read by the
Narrator, or by the speaker as a thought: the labelling step decides), and bare text is speech
(narration on a Narrator line). Splitting is done here, in code; the LLM only labels the pieces.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

NARRATOR = "Narrator"
MODE_SPEECH = "speech"
MODE_INNER = "inner"
MODE_NARRATION = "narration"
KIND_SPEECH = "speech"
KIND_ASIDE = "aside"

_PIECE = re.compile(r"\*([^*]+)\*|“([^”]+)”|\"([^\"]+)\"")
_SPEAKER = re.compile(r"^\s*([^:\"“*]{1,40}?)\s*:\s*(.*)$")
_WORDY = re.compile(r"[A-Za-z0-9]")


class ScriptError(ValueError):
    pass


@dataclass(frozen=True)
class Vocal:
    text: str  # exact span of the segment's text that spells out a sound
    sound: str  # what it is ("laugh", "moan", ...) or "none"
    keep: bool  # True: the words stay and the sound comes just before them


@dataclass(frozen=True)
class Sound:
    kind: str
    at: str  # "start" | "end"


@dataclass(frozen=True)
class Segment:
    voice: str
    mode: str
    text: str
    emotion: str = "neutral"
    intensity: float = 0.5
    delivery: str = "normal"
    pace: str = "normal"
    sounds: tuple[Sound, ...] = ()
    vocals: tuple[Vocal, ...] = ()


@dataclass(frozen=True)
class Piece:
    """One unlabelled part of a line, as the parser split it."""

    kind: str  # KIND_SPEECH | KIND_ASIDE
    text: str


@dataclass(frozen=True)
class RawLine:
    index: int  # 0-based among the script's spoken lines
    scene: int  # 0-based
    speaker: str
    text: str  # the line as written, after "Speaker:"
    pieces: tuple[Piece, ...]

    @property
    def key(self) -> str:
        return hashlib.sha1(f"{self.speaker}: {self.text}".encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Line:
    index: int
    scene: int
    speaker: str
    segments: tuple[Segment, ...]
    source: str = "llm"  # "llm" | "default" (labelling failed: plain labels)
    key: str = field(default="", compare=False)


def is_narrator(speaker: str) -> bool:
    return speaker.strip().lower() == NARRATOR.lower()


def split_pieces(text: str) -> list[Piece]:
    pieces: list[Piece] = []

    def bare(chunk: str) -> None:
        chunk = chunk.strip()
        if _WORDY.search(chunk):
            pieces.append(Piece(KIND_SPEECH, chunk))

    pos = 0
    for m in _PIECE.finditer(text):
        bare(text[pos : m.start()])
        if m.group(1) is not None:
            if _WORDY.search(m.group(1)):
                pieces.append(Piece(KIND_ASIDE, m.group(1).strip()))
        else:
            spoken = (m.group(2) or m.group(3)).strip()
            if spoken:
                pieces.append(Piece(KIND_SPEECH, spoken))
        pos = m.end()
    bare(text[pos:])
    return pieces


def parse_script(path: Path) -> list[RawLine]:
    """Every spoken line of the file; a blank line (or several) starts a new scene."""
    lines: list[RawLine] = []
    scene, pending_break = 0, False
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            pending_break = bool(lines)
            continue
        if stripped.startswith("#"):
            continue
        if pending_break:
            scene += 1
            pending_break = False
        m = _SPEAKER.match(stripped)
        speaker, text = (m.group(1).strip(), m.group(2)) if m else (NARRATOR, stripped)
        if is_narrator(speaker):
            speaker = NARRATOR
        pieces = split_pieces(text)
        if not pieces:
            raise ScriptError(f"{path.name}:{number}: nothing to say in {stripped!r}")
        lines.append(RawLine(len(lines), scene, speaker, text, tuple(pieces)))
    if not lines:
        raise ScriptError(f"{path} has no lines")
    return lines


def speakers(lines: list[RawLine]) -> list[str]:
    """Every voice in order of first appearance, the Narrator first when anyone narrates."""
    seen: list[str] = []
    narrates = False
    for ln in lines:
        narrates = (
            narrates or is_narrator(ln.speaker) or any(p.kind == KIND_ASIDE for p in ln.pieces)
        )
        if not is_narrator(ln.speaker) and ln.speaker not in seen:
            seen.append(ln.speaker)
    return [NARRATOR, *seen] if narrates else seen


def default_line(raw: RawLine) -> Line:
    """Plain labels, used when the LLM's labels for a line don't pass the checks."""
    segments = []
    for piece in raw.pieces:
        if is_narrator(raw.speaker) or piece.kind == KIND_ASIDE:
            segments.append(Segment(NARRATOR, MODE_NARRATION, piece.text, intensity=0.3))
        else:
            segments.append(Segment(raw.speaker, MODE_SPEECH, piece.text))
    return Line(raw.index, raw.scene, raw.speaker, tuple(segments), "default", raw.key)


def scenes(lines: list[Line]) -> list[list[Line]]:
    out: dict[int, list[Line]] = {}
    for ln in lines:
        out.setdefault(ln.scene, []).append(ln)
    return [out[k] for k in sorted(out)]


def segment_id(line: Line, i: int) -> str:
    return f"{line.index:03d}-{i}"


# --- direction.jsonl ---------------------------------------------------------------------


def write_direction(path: Path, lines: list[Line]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for ln in lines:
            fh.write(json.dumps(asdict(ln), ensure_ascii=False) + "\n")


def segment_from_dict(raw: dict) -> Segment:
    return Segment(
        voice=raw["voice"],
        mode=raw["mode"],
        text=raw["text"],
        emotion=raw.get("emotion", "neutral"),
        intensity=float(raw.get("intensity", 0.5)),
        delivery=raw.get("delivery", "normal"),
        pace=raw.get("pace", "normal"),
        sounds=tuple(Sound(s["kind"], s["at"]) for s in raw.get("sounds", [])),
        vocals=tuple(
            Vocal(v["text"], v["sound"], bool(v.get("keep"))) for v in raw.get("vocals", [])
        ),
    )


def read_direction(path: Path) -> list[Line]:
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            obj = json.loads(raw)
            lines.append(
                Line(
                    obj["index"],
                    obj["scene"],
                    obj["speaker"],
                    tuple(segment_from_dict(s) for s in obj["segments"]),
                    obj.get("source", "llm"),
                    obj.get("key", ""),
                )
            )
    return sorted(lines, key=lambda ln: ln.index)
