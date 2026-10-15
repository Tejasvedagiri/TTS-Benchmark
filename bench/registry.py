"""Model cards: one entry per TTS model, data only.

A card holds what the LLM needs to write that model's version of a segment (tag syntax and
vocabulary, style knobs, whether it takes the clip's transcript) and what the report shows
(repo, size, licence). Install pins live in render.py (SPECS), the inference code in
runners/run_<id>.py.

`tags`: None = the model can't perform sounds (sound spans are dropped); a list = its closed
tag vocabulary, in its own syntax ("[laugh]", "(laughs)", "<laugh>"); "open" = free-form tags
described in `tag_notes`.

`style_keys`: extra per-segment fields the runner reads, as (name, kind, description) for the
LLM's instructions.

`ref`: "wav" (clip only), "wav+txt" (clip + exact transcript) or "clone" (the repo's term).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelCard:
    id: str
    repo: str
    size: str
    licence: str
    ref: str  # "wav" | "wav+txt" | "clone"
    tags: list[str] | str | None  # None = no tags at all, "open" = open vocabulary
    tag_notes: str = ""
    style_keys: tuple[tuple[str, str, str], ...] = field(default_factory=tuple)
    needs_transcript: bool = False
    baseline: bool = False  # preset or described voices, no cloning
    # Fixed text put before every segment by code (e.g. Dia's "[S1] " speaker label): structure,
    # not a sound, so the story model never sees or writes it.
    speaker_prefix: str = ""
    # The model wants a sound tag followed by the sound written out (Higgs: "<|sfx:laughter|>
    # Haha"), so a keep=false vocals span may stay in the text (Phase 1 accepts both).
    sound_words_after_tag: bool = False


# --- Main comparison: zero-shot cloning models -----------------------------------
CLONING_MODELS: list[ModelCard] = [
    ModelCard(
        id="omnivoice",
        repo="k2-fsa/OmniVoice",
        size="~1B",
        licence="Apache code / NC weights",
        ref="wav+txt",
        needs_transcript=True,
        tags=["[laughter]", "[sigh]"],
        tag_notes="Unreliable: can garble or drop words around a tag. Use sparingly.",
    ),
    ModelCard(
        id="indextts2",
        repo="IndexTeam/IndexTTS-2",
        size="1.5B",
        licence="Apache 2.0",
        ref="wav",
        tags=None,
        tag_notes="No inline tags. Emotion is a separate control: either an emotion vector "
        "(8 weights) or free emotion text describing the delivery.",
        style_keys=(
            (
                "emotion_text",
                "string",
                "a short phrase describing the emotional delivery, used "
                "instead of a vector, e.g. 'pained, hissing through her teeth'",
            ),
        ),
    ),
    ModelCard(
        id="f5tts",
        repo="SWivid/F5-TTS",
        size="330M",
        licence="CC-BY-NC",
        ref="wav+txt",
        needs_transcript=True,
        tags=None,
        tag_notes="No tags, no style knobs. Weak prosody.",
    ),
    ModelCard(
        id="moss_tts",
        repo="OpenMOSS-Team/MOSS-TTS(-v1.5)",
        size="8B",
        licence="Apache 2.0",
        ref="clone",
        tags=["[pause 3.2s]"],
        tag_notes="Only a pause tag (any duration in seconds); no sound/emotion tags.",
    ),
    ModelCard(
        id="qwen3_tts",
        repo="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        size="1.7B",
        licence="Apache 2.0",
        ref="wav+txt",
        needs_transcript=True,
        tags=None,
        tag_notes="No tags.",
    ),
    ModelCard(
        id="fish_s2",
        repo="fishaudio/s2-pro",
        size="4B",
        licence="research NC",
        ref="clone",
        tags="open",
        tag_notes='Free-form "[verb]" tags, e.g. "[laughing] [sigh] [whisper]"; '
        "not an enumerated list, so propose natural verbs/gerunds only.",
    ),
    ModelCard(
        id="pocket_tts",
        repo="kyutai/pocket-tts",
        size="100M",
        licence="Apache 2.0",
        ref="wav",
        tags=None,
        tag_notes="No tags. Gated repo: needs `hf auth login`.",
    ),
    ModelCard(
        id="zonos",
        repo="Zyphra/Zonos-v0.1-transformer",
        size="1.6B",
        licence="Apache 2.0",
        ref="clone",
        tags=None,
        tag_notes="No inline tags; emotion is 8 continuous weights, plus speaking rate and "
        "pitch controls.",
        style_keys=(
            (
                "emotion_vector",
                "8 floats 0-1",
                "happiness, sadness, disgust, fear, surprise, "
                "anger, other, neutral -- weights summing loosely to 1",
            ),
            ("speaking_rate", "float", "words/sec multiplier, default ~1.0"),
            ("pitch_shift", "float semitones", "default 0"),
        ),
    ),
    ModelCard(
        id="voxcpm2",
        repo="openbmb/VoxCPM2",
        size="2B",
        licence="Apache 2.0",
        ref="wav",
        tags=None,
        tag_notes='No inline tags; a free-text "tone" prefix, e.g. "(cheerful tone)" ahead of '
        "the line, sets overall delivery for that line (not per-word).",
        style_keys=(
            (
                "tone_prefix",
                "string",
                "short parenthetical tone description prepended to the "
                "whole line, e.g. '(cheerful tone)' or '(pained, hissing)'",
            ),
        ),
    ),
    ModelCard(
        id="chatterbox_turbo",
        repo="ResembleAI/chatterbox-turbo",
        size="744M",
        licence="MIT",
        ref="wav",
        tags=[
            "[laugh]",
            "[chuckle]",
            "[cough]",
            "[sigh]",
            "[gasp]",
            "[sniffle]",
            "[groan]",
            "[yawn]",
            "[clear throat]",
            "[clears throat]",
            "[exhale]",
            "[gulp]",
            "[hesitates]",
            "[sneeze]",
            "[swallow]",
            "[chuckles]",
            "[laughs]",
            "[sighs]",
            "[gasps]",
        ],
    ),
    ModelCard(
        id="chatterbox_nano",
        repo="ResembleAI/chatterbox-nano",
        size="110M",
        licence="MIT",
        ref="wav",
        tags=[
            "[laugh]",
            "[sigh]",
            "[chuckle]",
            "[cough]",
            "[gasp]",
            "[groan]",
            "[yawn]",
            "[sniffle]",
            "[clear throat]",
            "[exhale]",
            "[advertisement]",
            "[happy]",
            "[sad]",
            "[angry]",
            "[whisper]",
        ],
        tag_notes="Sound tags plus a few emotion/style tags ([happy], [whisper], ...).",
    ),
    ModelCard(
        id="moss_tts_nano",
        repo="OpenMOSS-Team/MOSS-TTS-Nano",
        size="100M",
        licence="Apache 2.0",
        ref="wav",
        tags=None,
        tag_notes="No tags.",
    ),
    ModelCard(
        id="chatterbox",
        repo="ResembleAI/chatterbox",
        size="1.2B",
        licence="MIT",
        ref="wav",
        tags=None,
        tag_notes="No inline tags; two continuous knobs instead.",
        style_keys=(
            ("exaggeration", "float 0-2", "default ~0.5; higher = more exaggerated delivery"),
            ("cfg_weight", "float 0-1", "default ~0.5; lower = slower/more deliberate pacing"),
        ),
    ),
    ModelCard(
        id="dia",
        repo="nari-labs/Dia-1.6B-0626",
        size="1.6B",
        licence="Apache 2.0",
        ref="clone",
        tags=[
            "(laughs)",
            "(sighs)",
            "(gasps)",
            "(coughs)",
            "(clears throat)",
            "(humming)",
            "(sneezes)",
            "(chuckles)",
            "(whistles)",
        ],
        tag_notes="Sound tags in parentheses, inline where the sound happens.",
        speaker_prefix="[S1] ",
    ),
    ModelCard(
        id="csm_1b",
        repo="sesame/csm-1b",
        size="1B",
        licence="Apache 2.0",
        ref="wav+txt",
        needs_transcript=True,
        tags=None,
        tag_notes="No tags. Gated repo: needs manual approval on HF.",
    ),
    ModelCard(
        id="step_audio_editx",
        repo="stepfun-ai/Step-Audio-EditX",
        size="3B",
        licence="Apache 2.0",
        ref="clone",
        tags=[
            "[sigh]",
            "[inhale]",
            "[exhale]",
            "[laugh]",
            "[chuckle]",
            "[giggle]",
            "[snort]",
            "[cough]",
            "[clears throat]",
            "[Breathing]",
            "[Laughter]",
            "[Uhm]",
            "[Sigh]",
            "[Surprise-oh]",
            "[Surprise-ah]",
            "[Surprise-wa]",
            "[Confirmation-en]",
            "[Question-ei]",
            "[Dissatisfaction-hnn]",
        ],
        tag_notes="Paralinguistic tags from the README (2026-10-14: sigh, inhale, laugh, "
        "chuckle, exhale, clears throat, snort, giggle, cough) and the older set in app.py "
        "(Breathing, Laughter, Uhm, Sigh, Surprise-oh/ah/wa, Confirmation-en, Question-ei, "
        "Dissatisfaction-hnn). Inline where the sound happens; anything else is dropped.",
    ),
    ModelCard(
        id="neutts_air",
        repo="neuphonic/neutts-air",
        size="748M",
        licence="Apache 2.0",
        ref="wav+txt",
        needs_transcript=True,
        tags=None,
        tag_notes="No tags.",
    ),
    ModelCard(
        id="neutts_nano",
        repo="neuphonic/neutts-air-nano-q4-gguf",
        size="229M",
        licence="Apache 2.0",
        ref="wav+txt",
        needs_transcript=True,
        tags=None,
        tag_notes="No tags.",
    ),
    ModelCard(
        id="styletts2",
        repo="yl4579/StyleTTS2",
        size="148M",
        licence="MIT",
        ref="clone",
        tags=None,
        tag_notes="No inline tags; style is a continuous scale, not named emotions.",
        style_keys=(("style_scale", "float", "overall style-vector interpolation strength"),),
    ),
    ModelCard(
        id="cosyvoice3",
        repo="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        size="0.5B",
        licence="Apache 2.0",
        ref="clone",
        tags=None,
        tag_notes="No inline tags; a free-text natural-language description of the voice/style.",
        style_keys=(("description", "string", "free natural-language style description"),),
    ),
    ModelCard(
        id="higgs_v3",
        repo="bosonai/higgs-tts-3-4b",
        size="4B",
        licence="research NC",
        ref="clone",
        tags=[
            "<|sfx:cough|>",
            "<|sfx:laughter|>",
            "<|sfx:crying|>",
            "<|sfx:screaming|>",
            "<|sfx:burping|>",
            "<|sfx:humming|>",
            "<|sfx:sigh|>",
            "<|sfx:sniff|>",
            "<|sfx:sneeze|>",
            "<|style:whispering|>",
            "<|style:shouting|>",
        ],
        tag_notes="From the model card (2026-10-14): the 9 <|sfx:...|> tokens go inline exactly "
        "where the sound happens, and each MUST be followed by its written sound, so for a "
        "vocals span this model can perform, put the tag just before the span and KEEP the "
        "span's letters (e.g. <|sfx:laughter|>Ha! Ha ha ha!) even when keep=false. A span with "
        "no fitting sfx token is dropped as usual (keep=false) or left as words (keep=true). "
        "<|style:whispering|> / <|style:shouting|> go at the very start of the text, only for "
        "whisper / shout delivery.",
        style_keys=(
            (
                "emotion",
                "string",
                "one of elation, amusement, enthusiasm, "
                "determination, pride, contentment, affection, relief, contemplation, "
                "confusion, surprise, awe, longing, arousal, anger, fear, disgust, "
                "bitterness, sadness, shame, helplessness; omit for neutral delivery",
            ),
        ),
        sound_words_after_tag=True,
    ),
    ModelCard(
        id="breeze_tts2",
        repo="BreezeBlue/Breeze-TTS-2",
        size="3.5B",
        licence="research NC",
        ref="wav+txt",
        needs_transcript=True,
        tags="open",
        tag_notes="Open vocabulary '(name)' tags; card examples: (laugh) (cough) (clears "
        "throat) (sigh). Known NOT performed (read aloud or silent): sniff, sniffle, inhale, "
        "gulp, lip smack, click tongue -- drop these spans instead of tagging them. Also takes "
        "a free natural-language instruction (tone/pace), separate from the tag.",
        style_keys=(
            (
                "instruction",
                "string",
                "plain-English direction, e.g. 'Speak loudly with a "
                "pained tone.' (see breeze.py's breeze_instruction for the shape)",
            ),
        ),
    ),
    # --- Added 2026-10-14 (user: "is there any 5B or 7B?"): the larger open models on HF.
    # Parameter counts are HF's safetensors totals (codec / audio adapters included).
    ModelCard(
        id="drama3",
        repo="fishaudio/drama-3-int4",
        size="36B MoE (int4, ~21 GB)",
        licence="Fish Audio (research NC)",
        ref="wav+txt",
        needs_transcript=True,
        tags="open",
        tag_notes="Fish Audio Drama 3 (released 2026-10-14): free-form point tags in square "
        "brackets, written where the sound happens, e.g. [sighs] [laughs] [gasps] [moans] "
        "[groans]; natural third-person verbs, one sound per tag.",
    ),
    ModelCard(
        id="miso_tts",
        repo="MisoLabs/MisoTTS",
        size="8B",
        licence="other (Miso Labs)",
        ref="wav+txt",
        needs_transcript=True,
        tags=None,
        tag_notes="No tags. Sesame CSM architecture with an 8B Llama backbone.",
    ),
    ModelCard(
        id="llasa_8b",
        repo="HKUSTAudio/Llasa-8B",
        size="8B",
        licence="CC-BY-NC",
        ref="wav+txt",
        needs_transcript=True,
        tags=None,
        tag_notes="No tags. Llama + XCodec2 speech tokens; clones by continuation.",
    ),
    ModelCard(
        id="orpheus_clone",
        repo="canopylabs/orpheus-3b-0.1-pretrained",
        size="3B",
        licence="Apache 2.0",
        ref="wav+txt",
        needs_transcript=True,
        tags=[
            "<laugh>",
            "<chuckle>",
            "<sigh>",
            "<cough>",
            "<sniffle>",
            "<groan>",
            "<yawn>",
            "<gasp>",
        ],
        tag_notes="Orpheus's pretrained base, cloning by in-context example (README: not "
        "trained for cloning). Tags are the finetuned model's; whether the base performs them "
        "is what the listening checks. Engine script = orpheus's (same tags).",
    ),
    ModelCard(
        id="t5gemma_tts",
        repo="Aratako/T5Gemma-TTS-2b-2b",
        size="5.3B",
        licence="CC-BY-NC + Gemma terms",
        ref="wav+txt",
        needs_transcript=True,
        tags=None,
        tag_notes="No tags. T5Gemma encoder-decoder + XCodec2; English, Chinese, "
        "Japanese; duration control (estimated from the text here).",
    ),
    ModelCard(
        id="higgs_v2",
        repo="bosonai/higgs-tts-2-3b-base",
        size="3B (5.8B with audio adapter)",
        licence="research NC",
        ref="wav+txt",
        needs_transcript=True,
        tags=None,
        tag_notes="No inline tags documented on the card; expressiveness comes "
        "from the text. Native in transformers (HiggsAudioV2).",
    ),
    ModelCard(
        id="kitten_tts2",
        repo="KittenML/kitten-tts-2",
        size="1.7B",
        licence="Apache 2.0",
        ref="wav",
        tags=None,
        tag_notes="No tags.",
    ),
]

# --- Baselines: preset voices, no cloning ----------------------------------------
BASELINE_MODELS: list[ModelCard] = [
    ModelCard(
        id="kokoro",
        repo="hexgrad/Kokoro-82M",
        size="82M",
        licence="Apache 2.0",
        ref="wav",
        baseline=True,
        tags=None,
        tag_notes='Card says "cannot laugh": drop all vocals spans.',
    ),
    ModelCard(
        id="maya1",
        repo="maya-research/maya1",
        size="3B",
        licence="Apache 2.0",
        ref="wav",
        baseline=True,
        tags=["<laugh>", "<sigh>", "<whisper>", "<angry>"],
        tag_notes="20+ tags per the card; the 4 above are confirmed, the rest to confirm on "
        'install. The voice description goes ONLY in the "description" field, never in '
        '"text" (code wraps it as <description="..."> when rendering).',
        style_keys=(
            (
                "description",
                "string",
                "plain voice/character description, no tag "
                "syntax, e.g. warm, tired mother in her 30s",
            ),
        ),
    ),
    ModelCard(
        id="orpheus",
        repo="canopylabs/orpheus-3b-0.1-ft",
        size="3B",
        licence="Apache 2.0",
        ref="wav",
        baseline=True,
        tags=[
            "<laugh>",
            "<chuckle>",
            "<sigh>",
            "<cough>",
            "<sniffle>",
            "<groan>",
            "<yawn>",
            "<gasp>",
        ],
        tag_notes="The finetune's documented tag set (served via vLLM). No emotion tags: "
        "emotion comes from the words themselves.",
    ),
    ModelCard(
        id="voxtral_tts",
        repo="mistralai/Voxtral-4B-TTS-2603",
        size="4B",
        licence="CC-BY-NC",
        ref="wav",
        baseline=True,
        tags=None,
        tag_notes="Follows the reference's delivery rather than tags; Mac path (4-bit MLX) is "
        "presets only, no cloning.",
    ),
]

ALL_MODELS: dict[str, ModelCard] = {m.id: m for m in (*CLONING_MODELS, *BASELINE_MODELS)}
