"""Talker patches + Python wrapper for the megakernel-backed Qwen3-TTS talker."""

from .talker_constants import (
    CODEBOOKS_PER_FRAME,
    CODEC_VOCAB_SIZE,
    HEAD_DIM,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    MAX_SEQ_LEN,
    NUM_LAYERS,
    SAMPLE_RATE,
    SAMPLES_PER_TALKER_STEP,
    TALKER_HZ,
)
from .talker_model import TalkerDecoder, TalkerStep, build_rope_tables, load_talker_weights

__all__ = [
    "TalkerDecoder",
    "TalkerStep",
    "build_rope_tables",
    "load_talker_weights",
    "CODEBOOKS_PER_FRAME",
    "CODEC_VOCAB_SIZE",
    "HEAD_DIM",
    "HIDDEN_SIZE",
    "INTERMEDIATE_SIZE",
    "MAX_SEQ_LEN",
    "NUM_LAYERS",
    "SAMPLE_RATE",
    "SAMPLES_PER_TALKER_STEP",
    "TALKER_HZ",
]
