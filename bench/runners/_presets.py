"""Preset voices for the models that can't clone (and for the Kokoro clips made for speakers
with no reference clip). Each speaker has a gender and a slot (their index among speakers of
that gender, from cast.json), so every speaker gets a different preset and keeps it."""

PRESETS = {
    "kokoro": {
        "female": [
            "af_heart",
            "af_bella",
            "af_nicole",
            "af_sarah",
            "af_aoede",
            "af_kore",
            "af_nova",
            "af_sky",
        ],
        "male": [
            "am_fenrir",
            "am_michael",
            "am_puck",
            "am_echo",
            "am_eric",
            "am_liam",
            "am_onyx",
            "am_adam",
        ],
    },
    "orpheus": {
        "female": ["tara", "leah", "jess", "mia", "zoe"],
        "male": ["zac", "leo", "dan"],
    },
    "voxtral_tts": {
        "female": ["neutral_female", "cheerful_female", "casual_female"],
        "male": ["casual_male", "neutral_male"],
    },
}


def preset(model: str, row: dict) -> str:
    voices = PRESETS[model][row.get("voice_gender", "female")]
    return voices[int(row.get("voice_slot", 0)) % len(voices)]
