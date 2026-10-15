"""Point phonemizer at the espeak-ng library bundled in the espeakng-loader wheel (no system
espeak on Windows). Import before the model package."""

import os

import espeakng_loader

os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", espeakng_loader.get_library_path())
os.environ.setdefault("ESPEAK_DATA_PATH", espeakng_loader.get_data_path())
try:
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    EspeakWrapper.set_library(espeakng_loader.get_library_path())
except ImportError:
    pass
