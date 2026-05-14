"""TalkerDecoder — drives the (patched) megakernel for Qwen3-TTS's talker.

Talker layout (verified against ``QwenLM/Qwen3-TTS`` modeling code):

  Top-level model: Qwen3TTSForConditionalGeneration
    .talker  : Qwen3TTSTalkerForConditionalGeneration   (base_model_prefix="talker")
        .model       : Qwen3TTSTalkerModel              (base_model_prefix="talker.model")
            .layers       : ModuleList[28 × Qwen3TTSTalkerDecoderLayer]
            .norm         : Qwen3TTSRMSNorm
            .codec_embedding   : nn.Embedding(3072, 1024)
            .text_embedding    : nn.Embedding(151936, 2048)
            .rotary_emb        : Qwen3TTSTalkerRotaryEmbedding
        .text_projection  : Qwen3TTSTalkerResizeMLP (2048 -> 1024)
        .codec_head       : nn.Linear(1024, 3072, bias=False)     ← the LM head
        .code_predictor   : Qwen3TTSTalkerCodePredictorModelForConditionalGeneration
    .speaker_encoder : Qwen3TTSSpeakerEncoder
    .speech_tokenizer: Qwen3TTSTokenizer  (codec→audio decoder, loaded post-from_pretrained)

Per-talker-layer state-dict keys (with prefix ``talker.model.layers.{i}.``):
    input_layernorm.weight                  (1024,)
    self_attn.q_proj.weight                 (2048, 1024)
    self_attn.k_proj.weight                 (1024, 1024)
    self_attn.v_proj.weight                 (1024, 1024)
    self_attn.q_norm.weight                 (128,)   ← head_dim only
    self_attn.k_norm.weight                 (128,)   ← head_dim only
    self_attn.o_proj.weight                 (1024, 2048)
    post_attention_layernorm.weight         (1024,)
    mlp.gate_proj.weight                    (3072, 1024)
    mlp.up_proj.weight                      (3072, 1024)
    mlp.down_proj.weight                    (1024, 3072)

This matches the megakernel's ``LDGLayerWeights`` struct exactly.

MRoPE note: ``get_rope_index`` returns ``position_ids`` of shape ``[3, B, T]``
where all 3 axes are equal during pure-TTS inference. The interleaved-rope
``apply_multimodal_rotary_pos_emb`` then collapses to plain 1D RoPE on the
temporal axis. So we precompute cos/sin tables exactly like the original
megakernel did for Qwen3-0.6B text, but with ``rope_theta=1e6`` instead of
``10000``. The kernel's RoPE math is unchanged.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any, Callable

import torch

from .talker_constants import (
    CODEC_EOS_ID,
    CODEC_VOCAB_SIZE,
    HEAD_DIM,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    KV_SIZE,
    MAX_SEQ_LEN,
    NUM_KV_HEADS,
    NUM_LAYERS,
    Q_SIZE,
    ROPE_THETA,
)

# Trigger the JIT compile of the megakernel extension before resolving the op.
# Without this, ``torch.ops.qwen_megakernel_C.decode`` is None on first import
# (the op registry is only populated when the extension is loaded).
import qwen_megakernel  # noqa: F401 — side-effect: load the C++ extension

_decode = torch.ops.qwen_megakernel_C.decode


@dataclass
class TalkerStep:
    """One step's worth of decoded state."""

    codec_token: int  # argmax codec id from the LM head (talker emits codebook 0)
    hidden: torch.Tensor  # [HIDDEN_SIZE] bf16 — post-final-RMSNorm hidden (for Code Predictor)


def build_rope_tables(
    head_dim: int = HEAD_DIM,
    max_seq_len: int = MAX_SEQ_LEN,
    rope_theta: float = ROPE_THETA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plain 1D RoPE cos/sin tables, sized for the megakernel.

    Why not multi-axis: the talker's ``get_rope_index`` sets all three MRoPE
    axes to the same position vector during TTS. The interleaved-rope helper
    then writes the temporal cos into x_t, then "overwrites" height/width
    stride-3 entries with values that are identical to temporal. Net result:
    plain 1D RoPE.

    Layout matches qwen_megakernel.model.load_weights: shape
    ``[max_seq_len, head_dim]`` bf16 on cuda, with the second half a repeat
    of the first (so the kernel's `cos_table[position, :head_dim]` read works).
    """
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )  # [head_dim // 2]
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # [max_seq_len, head_dim // 2]
    cos = torch.cos(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
    sin = torch.sin(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
    return cos, sin


def _pack_layer_weights(layer_weights: list[torch.Tensor]) -> torch.Tensor:
    """Pack 11-tensor-per-layer flat list into a device blob of LDGLayerWeights structs."""
    ptr_size = 8
    n_ptrs = 11
    buf = bytearray(NUM_LAYERS * n_ptrs * ptr_size)
    for i in range(NUM_LAYERS):
        for j in range(n_ptrs):
            ptr = layer_weights[i * n_ptrs + j].data_ptr()
            struct.pack_into("Q", buf, (i * n_ptrs + j) * ptr_size, ptr)
    return torch.frombuffer(buf, dtype=torch.uint8).cuda()


def load_talker_weights(
    hf_model: Any,
    *,
    verbose: bool = True,
) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    """Extract the 28 talker layers + codec head + final norm from an HF Qwen3-TTS model.

    Walks the live module tree (not the flat state-dict) so we can grab ``data_ptr()``
    of the actual model parameters; this means the kernel reads from the SAME
    GPU memory as the HF model, no extra copy.

    Args:
        hf_model: A loaded ``Qwen3TTSForConditionalGeneration`` instance on cuda.
        verbose: Print a summary if True.

    Returns:
        A dict with keys ``layer_weights`` (list of 28*11 tensors),
        ``final_norm_weight``, ``lm_head_weight``, ``cos_table``, ``sin_table``.
    """
    talker = hf_model.talker          # Qwen3TTSTalkerForConditionalGeneration
    model = talker.model              # Qwen3TTSTalkerModel
    if verbose:
        print(f"[talker] loaded model.num_layers={len(model.layers)} "
              f"hidden={HIDDEN_SIZE} vocab={CODEC_VOCAB_SIZE}")
    assert len(model.layers) == NUM_LAYERS, (
        f"talker has {len(model.layers)} layers, expected {NUM_LAYERS}"
    )

    layer_weights: list[torch.Tensor] = []
    for i, layer in enumerate(model.layers):
        attn = layer.self_attn
        mlp = layer.mlp
        layer_weights.extend(
            t.to(torch.bfloat16).contiguous()
            for t in (
                layer.input_layernorm.weight,
                attn.q_proj.weight,
                attn.k_proj.weight,
                attn.v_proj.weight,
                attn.q_norm.weight,
                attn.k_norm.weight,
                attn.o_proj.weight,
                layer.post_attention_layernorm.weight,
                mlp.gate_proj.weight,
                mlp.up_proj.weight,
                mlp.down_proj.weight,
            )
        )

    final_norm = model.norm.weight.to(torch.bfloat16).contiguous()
    lm_head = talker.codec_head.weight.to(torch.bfloat16).contiguous()
    assert lm_head.shape == (CODEC_VOCAB_SIZE, HIDDEN_SIZE), (
        f"codec_head is {tuple(lm_head.shape)}, expected ({CODEC_VOCAB_SIZE}, {HIDDEN_SIZE})"
    )

    cos_table, sin_table = build_rope_tables()

    return dict(
        layer_weights=layer_weights,
        final_norm_weight=final_norm,
        lm_head_weight=lm_head,
        cos_table=cos_table,
        sin_table=sin_table,
    )


class TalkerDecoder:
    """One-instance-per-process talker driver.

    The trick: the patched megakernel still expects an ``embed_weight`` tensor
    and an ``input_token_id``. We construct a 1-row "fake embedding table"
    (``self._fake_embed``, shape ``[HIDDEN_SIZE]``), copy our pre-composed
    hidden state into it each step, and pass ``input_token_id = 0``. The
    kernel's ``embed_row = embed_weight + 0 * HIDDEN_SIZE`` lookup then
    returns our hidden state directly. No kernel source code change.

    Usage::

        td = TalkerDecoder(weights, compose_embed_fn=my_embed)
        td.reset()
        for frame_idx in range(max_steps):
            step = td.step(frame_idx)
            if td.is_eos(step.codec_token):
                break
            # step.codec_token  -> Code Predictor + speech_tokenizer -> audio

    Args:
        weights: Output of :func:`load_talker_weights`.
        compose_embed_fn: Callable ``(*, frame_idx, position, prev_codec_token) -> torch.Tensor``
            returning the per-step composite hidden state — shape
            ``[HIDDEN_SIZE]`` bf16 on cuda. The Pipecat TTS service owns the
            implementation (it needs access to the HF model's embedding
            tables, Code Predictor, etc.).
    """

    def __init__(
        self,
        weights: dict[str, torch.Tensor | list[torch.Tensor]],
        *,
        compose_embed_fn: Callable[..., torch.Tensor],
    ) -> None:
        self._compose = compose_embed_fn
        self._weights = weights
        self._final_norm = weights["final_norm_weight"]
        self._lm_head = weights["lm_head_weight"]
        self._cos = weights["cos_table"]
        self._sin = weights["sin_table"]
        self._layer_weights_packed = _pack_layer_weights(weights["layer_weights"])

        self._attn_scale = 1.0 / math.sqrt(HEAD_DIM)
        self._position = 0

        # 1-row fake embed table — see class docstring.
        self._fake_embed = torch.zeros(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

        self._k_cache = torch.zeros(
            NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM,
            dtype=torch.bfloat16, device="cuda",
        )
        self._v_cache = torch.zeros_like(self._k_cache)

        f32 = dict(dtype=torch.float32, device="cuda")
        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        self._hidden = torch.empty(HIDDEN_SIZE, **bf16)
        self._act = torch.empty(HIDDEN_SIZE, **f32)
        self._res = torch.empty(HIDDEN_SIZE, **f32)
        self._q = torch.empty(Q_SIZE, **f32)
        self._k = torch.empty(KV_SIZE, **f32)
        self._v = torch.empty(KV_SIZE, **f32)
        self._attn_out = torch.empty(Q_SIZE, **f32)
        self._mlp_inter = torch.empty(INTERMEDIATE_SIZE, **f32)
        self._norm_out = torch.empty(HIDDEN_SIZE, **f32)

        # 4096 here is just an upper bound on LDG_LM_NUM_BLOCKS; with the
        # patched build.py we use 64 blocks for codec_vocab=3072.
        self._bmax_vals = torch.empty(4096, **f32)
        self._bmax_idxs = torch.empty(4096, dtype=torch.int32, device="cuda")
        self._out_token = torch.empty(1, dtype=torch.int32, device="cuda")

    def reset(self) -> None:
        self._position = 0
        self._k_cache.zero_()
        self._v_cache.zero_()

    @property
    def position(self) -> int:
        return self._position

    def step_with_hidden(self, hidden_in: torch.Tensor) -> int:
        """Drive one decode step with a caller-supplied hidden state.

        Used during prefill (when the caller already has the composite
        embedding tensor and just wants the kernel to advance the KV cache
        + emit a codec token).

        Args:
            hidden_in: [HIDDEN_SIZE] bf16 cuda tensor.

        Returns:
            Codec token id (int) emitted by the kernel for this position.
        """
        assert hidden_in.shape == (HIDDEN_SIZE,) and hidden_in.dtype == torch.bfloat16
        self._fake_embed.copy_(hidden_in)

        _decode(
            self._out_token,
            0,  # input_token_id — fake_embed row 0 = our hidden_in
            self._fake_embed,
            self._layer_weights_packed,
            self._final_norm,
            self._lm_head,
            self._cos,
            self._sin,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._act,
            self._res,
            self._q,
            self._k,
            self._v,
            self._attn_out,
            self._mlp_inter,
            self._norm_out,
            self._bmax_vals,
            self._bmax_idxs,
            NUM_LAYERS,
            self._position,
            MAX_SEQ_LEN,
            self._attn_scale,
        )
        self._position += 1
        return int(self._out_token.item())

    def step(self, frame_idx: int) -> TalkerStep:
        """Run one autoregressive talker decode step.

        Calls the caller's ``compose_embed_fn`` to build the composite hidden,
        feeds it to the kernel, returns the codec token plus the
        post-final-RMSNorm hidden state (needed by the Code Predictor).
        """
        hidden_in = self._compose(
            frame_idx=frame_idx,
            position=self._position,
            prev_codec_token=int(self._out_token.item()) if self._position > 0 else None,
        )
        codec_token = self.step_with_hidden(hidden_in)
        # `g_normalized` (kernel's final-RMSNorm output, fp32) lives in
        # self._norm_out. Cast to bf16 for the Code Predictor.
        return TalkerStep(codec_token=codec_token, hidden=self._norm_out.to(torch.bfloat16))

    @staticmethod
    def is_eos(codec_token: int) -> bool:
        return codec_token == CODEC_EOS_ID
