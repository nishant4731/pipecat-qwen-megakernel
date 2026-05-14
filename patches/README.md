# Megakernel patches for Qwen3-TTS talker

This directory contains everything we add to a vanilla clone of
[`AlpinDale/qwen_megakernel`](https://github.com/AlpinDale/qwen_megakernel) to make
it serve as the autoregressive decode backend for the **Talker** of
`Qwen/Qwen3-TTS-12Hz-0.6B-Base`.

The talker is shape-identical to `Qwen3-0.6B` (hidden 1024, 28 layers, GQA 16/8, head_dim 128, MLP 3072). Three things differ:

| Difference | How we handle it |
|---|---|
| **LM head: 3,072 codec vocab** (not 151,936 text vocab) | build-time `LDG_VOCAB_SIZE` macro |
| **Composite input embedding** (text + speaker + ref-audio + codec-history + code-group) | computed in PyTorch on GPU each step into a **single-row "fake embed table"** that the kernel reads via the existing `embed_weight + token_id * HIDDEN_SIZE` lookup (`token_id` is always 0). No kernel code change needed. |
| **MRoPE sections `[24, 20, 20]`** with `rope_theta = 1e6` | precomputed cos/sin tables built in `talker_model.py` — height/width sections collapse to identity for pure-TTS use, so the kernel's RoPE math is unchanged |

We do NOT touch:

- the 128×512 block scheme
- the MLP prefetch partition (intermediate/hidden ratio is identical: 3072/1024 = 3)
- the KV-cache layout, `MAX_SEQ_LEN`
- the attention math
- the `LDGLayerWeights` struct (talker has q_norm / k_norm, same as 0.6B text)

## Files

| File | Purpose |
|---|---|
| `apply_patches.py` | Idempotent script that mutates a cloned `qwen_megakernel/` tree in-place: opens up `LDG_VOCAB_SIZE` as a `-D` macro and adjusts the LM-head block partition for small codec vocabs. Two minimal string-anchored edits, total ~10 lines changed across `kernel.cu` and `build.py`. |
| `talker_constants.py` | Shapes (vocab=3,072, codec frame rate, MRoPE sections) from `Qwen/Qwen3-TTS-12Hz-0.6B-Base/config.json`. |
| `talker_model.py` | `TalkerDecoder` — owns the talker weight loader, the composite embed, the MRoPE-aware cos/sin tables, the 1-row fake embed table, and a `step(text_token, codec_token, frame_idx)` method that drives the (unmodified) kernel one talker step at a time. |

## Why no traditional `.patch` file?

A line-number diff would break if Alpin pushes a single commit to the upstream
repo. `apply_patches.py` does the same edits anchored on string patterns —
robust to upstream drift, and easy to re-run.

## How to apply

```bash
# inside qwen_megakernel/ working tree on the 5090 box
python /path/to/this/patches/apply_patches.py /path/to/qwen_megakernel
```

The script prints exactly what it changed. Idempotent — running twice is a no-op.

## RoPE detail (Patch C explained)

The talker uses MRoPE with `mrope_section=[24, 20, 20]` and `rope_theta=1e6`. MRoPE is multi-axis (temporal, height, width) and was introduced for Qwen2.5-Omni's vision+audio joint position handling. For pure TTS, height and width positions are always 0 — so the rotation in those dimensions collapses to identity (cos=1, sin=0).

This means the kernel's RoPE math doesn't need to change at all. We just build cos/sin tables whose entries past the temporal section (dims 24..63 of the half-table) are 1 / 0 respectively. See `talker_model.py::build_mrope_tables`.

If a future Qwen-TTS variant actually drives the height/width axes during TTS, we'd need to encode those axes into the position passed to the kernel and switch to three cos/sin tables. Out of scope here.

## Validation gate (on the 5090, before running Pipecat)

1. Run stock HF `Qwen3-TTS-12Hz-0.6B-Base` end-to-end on a fixed prompt — save the codec-token stream and waveform.
2. Run the patched megakernel-talker + stock Code Predictor + stock Vocoder on the same prompt.
3. Codec tokens should match the reference 1-for-1 (argmax is deterministic in bf16 modulo numerical drift; first divergence point pinpoints which patch is wrong).
4. Waveforms should be perceptually identical and within tight L2 of the reference.
