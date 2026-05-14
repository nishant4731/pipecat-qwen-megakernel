# pipecat-qwen-megakernel

End-to-end voice agent: **Whisper STT → HF Qwen3-1.7B LLM → Qwen3-TTS (talker on AlpinDale's `qwen_megakernel`) → 24 kHz PCM** — streamed frame-by-frame to a Pipecat pipeline.

Take-home brief: take [AlpinDale's `qwen_megakernel`](https://github.com/AlpinDale/qwen_megakernel) (a hand-written CUDA persistent kernel that decodes Qwen3-0.6B at ~1,000 tok/s on a single RTX 5090) and re-purpose it as the autoregressive decode backend for the **Talker** stage of `Qwen/Qwen3-TTS-12Hz-0.6B-Base`, then wire the result into a Pipecat voice agent.

## Performance numbers

Single RTX 5090 (sm_120a) on Vast.ai. bf16 throughout, no quantization. Methodology: 5 runs of `bench_tts.py` on the fixed prompt *"The five boxing wizards jump quickly."* (~3 s audio per utterance), warmup excluded. Brief's deliverable targets: TTFC < 90 ms, RTF < 0.3, decode tok/s & end-to-end latency reported.

| metric                                      | value                | brief target |
|---------------------------------------------|----------------------|--------------|
| Megakernel decode (talker shape, `bench_kernel.py`) | **662 tok/s · 1.51 ms / step** | report it    |
| TTFC (mean / p95)                           | **36.7 / 37.9 ms** ✅ | < 90 ms      |
| RTF  (mean / p95)                           | **0.352 / 0.363**    | < 0.3        |
| Audio streaming                             | per-frame `TTSAudioRawFrame` (80 ms / frame) ✅ | not buffered |
| End-to-end voice turn (representative)      | STT 280 ms + LLM 1,340 ms + TTS 1,070 ms = **~2.7 s** | report it |

See [STATUS.md](STATUS.md) for per-stage breakdown, methodology details, and the honest write-up of what we tried, what worked, and what didn't.

## Architecture

```
mic ─► push_to_talk_client (laptop, sounddevice)
            │ raw 16 kHz int16 PCM, framed
            ▼  ─[SSH]─►
            push_to_talk_server (5090 box)
                  │
                  ▼
        faster-whisper base.en  (CPU int8; see STATUS.md)
                  │ user_text
                  ▼
        HF Qwen3-1.7B            (GPU bf16, .generate)
                  │ reply_text
                  ▼
    ┌──── MegakernelQwen3TTSService ────┐
    │  per talker step @ 12.5 Hz:        │
    │    PyTorch composite embed          │
    │       → patched megakernel  ~1.5 ms │
    │       → Code Predictor (15 codebooks, compiled whole-loop) │
    │       → speech_tokenizer (compiled, CUDA-Graph)            │
    │       → int16 PCM @ 24 kHz, 1920 samples / frame           │
    │       → TTSAudioRawFrame            │
    └─────────────────────────────────────┘
                  │
                  ▼  ◄─[SSH]─
            push_to_talk_client (sounddevice playback)
```

## Kernel modifications

Three differences between Qwen3-0.6B-text (what the megakernel was built for) and the Qwen3-TTS talker, and how each is handled. Only one requires a kernel source change.

| Difference | How we handle it | Kernel source change? |
|---|---|---|
| **LM-head vocab: 3,072** (codec) vs 151,936 (text) | Make `LDG_VOCAB_SIZE` a build-time `-D` macro; the kernel's LM-head fused block-partition is parameterized accordingly. See [`patches/apply_patches.py`](patches/apply_patches.py). | ✅ ~5 lines in `kernel.cu` + `build.py` |
| **Composite input embedding** (text + speaker + ref-audio + codec-history + code-group) | Compute the per-step `[1, 1024]` bf16 hidden state in PyTorch, write it to a single-row "fake embed table", hand to the kernel via the existing `embed_weight + token_id * HIDDEN_SIZE` lookup with `token_id = 0`. | ❌ no |
| **MRoPE sections `[24, 20, 20]`, `rope_theta = 1e6`** | For pure TTS inference, `Qwen3TTSTalkerModel.get_rope_index` returns `[3, B, T]` position-ids with all three axes equal → MRoPE collapses to plain 1D RoPE. We just precompute cos/sin with `rope_theta = 1e6`. | ❌ no |

Patcher is idempotent; re-running it is a no-op. Details: [`patches/README.md`](patches/README.md).

## Repo layout

```
patches/
  apply_patches.py            # string-anchored, idempotent patcher for qwen_megakernel
  talker_constants.py         # talker shapes + special tokens (codec_eos=2150, etc.)
  talker_model.py             # TalkerDecoder — Python driver of the patched kernel
  README.md                   # the kernel-modification story

pipecat_qwen_megakernel/
  services/
    qwen3_tts_megakernel.py   # MegakernelQwen3TTSService (Pipecat TTSService)
    qwen3_llm_local.py        # HF Qwen3-Instruct LLMService
  app/
    push_to_talk_server.py    # the demo — runs on the 5090 box
    push_to_talk_client.py    # the demo — runs on your laptop
  bench/
    bench_kernel.py           # kernel-only microbench (tok/s, ms/step)
    bench_tts.py              # end-to-end TTS bench (TTFC, RTF)
    bench_stages.py           # per-stage timings (talker / code_predictor / speech_tokenizer)

scripts/setup.sh              # one-shot bringup for the 5090 box
docs/architecture.md          # design rationale
```

## Build and run

### Bringing up the 5090 box

Rent on Vast.ai with: RTX 5090, driver ≥ 570, CUDA ≥ 12.8, ≥ 32 GB RAM, ≥ 80 GB disk, image `pytorch:2.7-cuda12.8` (or `nvidia/cuda:12.8.0-devel-ubuntu22.04`). Open ports 22 + 7860.

```bash
# SSH in, scp / git clone this repo to /workspace/task, then:
bash scripts/setup.sh
```

This verifies sm_120, installs torch + transformers + qwen-tts, clones and patches the megakernel, and builds it.

Smoke-test the megakernel with AlpinDale's reference Qwen3-0.6B bench:

```bash
cd refs/qwen_megakernel && python -m qwen_megakernel.bench
# expect ~1,036 tok/s
```

### Run the Pipecat demo

The voice agent runs on the 5090 box; mic capture + speaker playback happen on your laptop, piped over SSH. Avoids the WebRTC/NAT issues that block a browser demo on Vast.ai (see STATUS.md for that story).

**On your laptop:**

```bash
pip3 install --user sounddevice numpy
python3 -m pipecat_qwen_megakernel.app.push_to_talk_client --ssh <your-ssh-alias>
```

You'll see Whisper + Qwen3-1.7B + Qwen3-TTS load on the box (~30 s first time), then `[INFO] ready`. Press ENTER, speak a short sentence, press ENTER again. The reply plays through your speakers.

### Run the benches

```bash
LDG_VOCAB_SIZE=3072 python -m pipecat_qwen_megakernel.bench.bench_kernel --steps 1000
python -m pipecat_qwen_megakernel.bench.bench_tts --runs 5 \
    --text "The five boxing wizards jump quickly."
python -m pipecat_qwen_megakernel.bench.bench_stages --runs 3
```

## License

Apache-2.0 for our additions. `qwen_megakernel` is Apache-2.0 (AlpinDale). Qwen3-TTS weights are governed by Qwen's model license.
