"""Shapes and special tokens for Qwen3-TTS-12Hz-0.6B-Base talker.

All values verified against:
  - ``Qwen/Qwen3-TTS-12Hz-0.6B-Base/config.json`` on HuggingFace
  - ``QwenLM/Qwen3-TTS/qwen_tts/core/models/modeling_qwen3_tts.py``

(See ``refs/qwen3_tts/`` for the local snapshot.)
"""

# --- Talker transformer shape (matches Qwen3-0.6B text) ---------------------
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3072
NUM_LAYERS = 28
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128

# Talker config sets max_position_embeddings = 32768. We keep the megakernel's
# 2048 cap because (a) a single TTS utterance never gets near 32k frames,
# (b) larger MAX_SEQ_LEN inflates the KV cache.
MAX_SEQ_LEN = 2048

# --- LM head ----------------------------------------------------------------

# Talker output vocab. 3072 entries cover codec tokens [0..2047] + special
# control tokens [2148..2157] + language/speaker IDs. Source: talker_config.vocab_size.
CODEC_VOCAB_SIZE = 3072

# --- Embedding tables -------------------------------------------------------

# Text embedding is 2048-dim; the talker has a separate `text_projection`
# (a Resize MLP) that projects 2048 -> 1024. Source: talker_config.text_hidden_size.
TEXT_VOCAB_SIZE = 151_936
TEXT_HIDDEN_SIZE = 2048

# --- RoPE -------------------------------------------------------------------

# rope_theta from talker_config. We can keep MRoPE info around for clarity,
# but: during pure-TTS inference, `get_rope_index` returns position_ids of
# shape [3, B, T] with ALL THREE axes equal — so MRoPE-interleaved collapses
# to plain 1D RoPE. The megakernel's existing single-axis RoPE math works
# unchanged; we just feed it cos/sin tables built with rope_theta=1e6.
ROPE_THETA = 1_000_000.0
MROPE_SECTION = (24, 20, 20)  # informational only — see build_rope_tables()

# --- Codec special tokens (from talker_config) ------------------------------

CODEC_BOS_ID = 2149
CODEC_EOS_ID = 2150          # the talker's EOS during autoregressive decode
CODEC_PAD_ID = 2148
CODEC_THINK_ID = 2154
CODEC_NOTHINK_ID = 2155
CODEC_THINK_BOS_ID = 2156
CODEC_THINK_EOS_ID = 2157

# --- Text-side TTS special tokens (from top-level config) -------------------

TTS_BOS_TOKEN_ID = 151_672
TTS_EOS_TOKEN_ID = 151_673
TTS_PAD_TOKEN_ID = 151_671
IM_END_TOKEN_ID = 151_645
IM_START_TOKEN_ID = 151_644
ASSISTANT_TOKEN_ID = 77_091

# --- Codec language IDs (from talker_config.codec_language_id) --------------

CODEC_LANG = {
    "english": 2050,
    "spanish": 2054,
    "chinese": 2055,
    "japanese": 2058,
    "korean": 2064,
    "french": 2061,
    "german": 2053,
    "russian": 2069,
    "italian": 2070,
    "portuguese": 2071,
}

# --- Audio / streaming ------------------------------------------------------

SAMPLE_RATE = 24_000        # speaker_encoder_config.sample_rate
TALKER_HZ = 12.5             # one autoregressive step per 80 ms of audio
SAMPLES_PER_TALKER_STEP = int(SAMPLE_RATE / TALKER_HZ)  # 1920
CODEBOOKS_PER_FRAME = 16     # talker emits codebook 0; Code Predictor emits 1..15

# --- Derived shapes for megakernel scratch ----------------------------------

Q_SIZE = NUM_Q_HEADS * HEAD_DIM   # 2048
KV_SIZE = NUM_KV_HEADS * HEAD_DIM  # 1024
