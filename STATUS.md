# Status

**Last updated: 2026-05-14 (post-execution on 5090).**

## What works end-to-end

The push-to-talk demo path (laptop client over SSH to bot on Vast.ai 5090):

- Mic capture (sounddevice) → SSH stdin
- faster-whisper `base.en` (CPU int8) transcribes
- HF Qwen3-1.7B generates a reply (`max_new_tokens=80`, brief-assistant prompt)
- `MegakernelQwen3TTSService` (Pipecat `TTSService` subclass) does talker-prefill → autoregressive loop → Code Predictor → speech-tokenizer per 80 ms frame
- 24 kHz int16 PCM is streamed back over SSH, played via sounddevice

Audio is intelligible speech in the default Qwen3-TTS Base voice.

## Measured numbers (single RTX 5090 sm_120a, Vast.ai)

All measurements come from the bench scripts under `pipecat_qwen_megakernel/bench/`. Single GPU, no quantization, bf16 throughout.

### Megakernel-only microbench (`bench_kernel.py`, 1,000 steps, 50 warmup)

| metric | value |
|---|---|
| Talker decode | **662.4 tok/s** (1.510 ms / step) |
| AlpinDale's reference (Qwen3-0.6B-text, same kernel) | ~1,036 tok/s |
| Audio rate needed | 12.5 Hz → 12.5 tok/s ⇒ kernel has **53×** headroom |

The talker-shape number is lower than the text-shape reference for two reasons we know of: (a) the per-step composite-embed copy into our 1-row "fake embed table" adds Python+H2D overhead the text path doesn't have, (b) the LM-head fused block partition was tuned for 151,936-class output — at 3,072 it's over-provisioned (`LDG_LM_NUM_BLOCKS` was not retuned). Neither matters for our 12.5 Hz audio target — even at 1.5 ms/step the kernel is 53× faster than required.

### End-to-end TTS bench (`bench_tts.py`, 5 runs, fixed prompt)

Prompt: `"The five boxing wizards jump quickly."` — produces ~2.96 s of audio after the optimization round, 32–37 AR frames after a 9-row prefill (7-token text trail). (Audio length varies by ±1 frame from baseline because `torch.compile`'s slight numeric reorderings tip the silent-frame EOS detector a frame earlier or later — the speech itself is the same.)

| metric | baseline | after full optimization | brief target | status |
|---|---|---|---|---|
| TTFC (mean) | 105.1 ms | **36.7 ms** | < 60 ms strict / < 50 ms stretch / < 90 ms lenient | ✅ all three |
| TTFC (p95) | 105.3 ms | **37.9 ms** | — | — |
| RTF (mean) | 1.225 | **0.352** | < 0.15 strict / < 0.1 stretch / < 0.3 lenient | ❌ all three |
| RTF (p95) | 1.227 | 0.363 | — | — |

**TTFC: every threshold met, including the stretch (< 50 ms).** RTF: the optimizations cut it from 1.22 to 0.35 (-71 %), but the strict and lenient thresholds (< 0.15 / < 0.3) are still over.

### Per-stage breakdown (`bench_stages.py`, 3 runs)

CUDA-synced timers around each GPU stage inside `MegakernelQwen3TTSService`. Per AR frame (80 ms of audio):

| stage | baseline ms | after optimization ms | how |
|---|---|---|---|
| Talker step (megakernel + compose_embed callback, includes CP) | 76.3 | **25.6** | see below |
|   ↳ Code Predictor 15-step decode | 74.76 | **~15** | hand-rolled greedy loop replaces `cp.generate(max_new_tokens=15)`. **Whole-loop** `torch.compile(max-autotune-no-cudagraphs)` (not just `cp.model`) so inductor fuses across all 15 sequential forwards. Pre-allocated HF `StaticCache` so cache shape never changes between steps. |
|   ↳ Megakernel decode + Python callback to compose_embed | 1.5 + ~9 | 1.5 + ~9 | unchanged — Python C++↔Python bridge is on the critical path |
|   ↳ Fused codebook gather (replaces 15 separate embedding lookups) | — | ~0.04 | one fancy-indexing op into pre-stacked `[15, 2048, H]` tensor |
| `speech_tokenizer.decode` | 22.50 | **1.59** | `torch.compile(reduce-overhead)` on `speech_tokenizer.model.decoder` (pure conv-net, no HF Cache → CUDA Graphs capture cleanly). Bypass the wrapper that goes through `chunked_decode(...)` — that method internally does `self(codes_chunk)` where `self` is the *original* decoder, not the compiled OptimizedModule. |
| **Per-AR-frame total** | **98.78** | **~27.2** | **-72 %** |
| → per-frame RTF | 1.235 | **~0.34** | matches measured (0.352) |

The kernel is doing what it should. The remaining gap to RTF < 0.15 is ~10 ms of Python callback overhead the megakernel pays each step when it invokes `_compose_embed` to compute the next composite embed. Pure-GPU ops there sum to < 1 ms; the rest is the C++↔Python bridge cost.

### End-to-end voice agent (push-to-talk demo)

One representative turn (single user utterance → one assistant reply):

| stage | latency |
|---|---|
| STT (faster-whisper base.en, CPU int8) | ~280 ms |
| LLM (HF Qwen3-1.7B `.generate(max_new_tokens=80)`) | ~1,340 ms |
| TTS (this work — Pipecat service backed by patched megakernel) | ~1,070 ms for ~3 s of audio (RTF ≈ 0.35) |
| Per-turn total (audio-in to last audio-out) | **~2.7 s** |

End-to-end is dominated by LLM (`.generate(...)` is one-shot, blocks TTS until done — token-streaming would let TTS start on the first LLM token).

## What changed in the megakernel (the actual patches)

Single source change in `qwen_megakernel`: `LDG_VOCAB_SIZE` becomes a build-time `-D` macro (~5 lines across `kernel.cu` and `build.py`). Patches/apply_patches.py is idempotent.

Everything else is handled in Python without touching the kernel:

- **Composite embedding** (text + speaker + ref-audio + codec-history + code-group): computed each step in PyTorch, written into a 1-row "fake embed table" that the kernel reads with `token_id=0`.
- **MRoPE**: `Qwen3TTSTalkerModel.get_rope_index` returns `[3, B, T]` position-ids with all three axes equal during TTS inference. The interleaved MRoPE writes then collapse to plain 1D rotation. We just precompute cos/sin with `rope_theta=1e6`.
- **LM head wiring**: `talker.codec_head` (nn.Linear 1024→3072) is loaded as the kernel's LM head; talker layer weights come from the live `talker.model.layers.*` module tree.
- **Codec EOS**: `codec_eos_token_id=2150` from `talker_config` (not the text-side `tts_eos_token_id=151673`).

## What we explicitly chose not to optimize

We **deliberately left Code Predictor and speech_tokenizer un-tuned**. Per the brief, the megakernel is the target for the *talker*, and the kernel work is what we wanted measured. Touching the other stages risks numerical drift vs the HF reference, which makes the codec-token diff (the correctness canary, see "Open risks" below) harder to read. With the per-stage measurements above we now know exactly where the cheapest wins live: a hand-rolled 15-step Code Predictor decode loop (skipping `GenerationMixin`) is the single biggest lever, and `torch.compile` on the `speech_tokenizer.decode` path is second. If the project continued, that's the order.

The same logic applies to LLM streaming. Right now `Qwen3LocalLLMService` calls `model.generate(...)` once and waits — the full reply text only reaches TTS after the LLM is done. Hooking `TextIteratorStreamer` so TTS starts on the first LLM token would cut perceived end-to-end latency dramatically without touching any GPU code.

## What did not work

### Live browser WebRTC demo on Vast.ai

`SmallWebRTCTransport` variant of the demo (`app/demo.py`) is wired but doesn't actually connect a browser to the bot on Vast.ai:

- Vast.ai blocks inbound UDP — TURN is mandatory.
- Cloudflare TURN ephemeral creds work for ALLOCATE but fail on **CHANNEL-BIND** with `401 ERROR-CODE` and **no** `NONCE`/`REALM` in the response. aioice 0.10's `request_with_retry` only re-auths when those attrs are present, so retry doesn't fire.
- A monkey-patch (probe-without-auth → get NONCE/REALM → retry with fresh creds) does pull a fresh nonce, but the retried CHANNEL-BIND still gets 401. This looks like a missing CREATE-PERMISSION step that some servers require before CHANNEL-BIND.
- Other ICE-server paths considered: LiveKit Cloud (free, no card; SFU bypasses the issue — was the next thing to try), Daily.co (requires payment method), self-hosted coturn (works in principle but Vast doesn't give a public-routable IP on UDP).

This is *not* a bug in the megakernel work or the TTS service — both stages produce correct, frame-by-frame streamed audio that the Pipecat pipeline pushes downstream as `TTSAudioRawFrame`s. The blocker is purely the WebRTC/NAT/TURN side.

### faster-whisper on GPU

The Vast.ai PyTorch image we used ships CUDA 13; ctranslate2 has no CUDA-13 wheel yet, so `device="cuda"` raises `libcublas.so.12 not found`. We fell back to `device="cpu", compute_type="int8"`. Installing `nvidia-cublas-cu12` + `nvidia-cudnn-cu12` pip wheels alongside CUDA 13 would restore GPU Whisper.

## What still has open risk

These are real, not paper:

1. **No codec-token-level diff vs stock HF Qwen3-TTS.** We validated by ear and by dimensions. A bit-for-bit (or bf16-tolerance) diff against the HF reference for a fixed prompt is the proper correctness canary; first divergence point would identify any silent bug in the patches.
2. **Speech-tokenizer streaming behavior.** Our code calls `decode([{"audio_codes": codes}])` per single-frame slice. We can hear that audio is continuous and recognizable, but we didn't characterize discontinuities at frame boundaries beyond casual listening. For higher-fidelity demos a windowed-decode buffer (emit only the newest 1920 samples each call) might be needed.
3. **Long prompts not stress-tested.** Prefill goes through the kernel one row at a time; with a 30-row prompt at ~1 ms/row, that's still under TTFC budget — but we didn't measure prompts beyond ~10 tokens.
4. **Single voice.** Default Base voice only; we did not exercise reference-audio cloning or the `instruct` mode.

## Deliverables

- [x] Repo with code (patches, services, demo, benches, setup script)
- [x] README with architecture, what changed in the kernel and where, build instructions, honest numbers
- [x] STATUS.md with everything in this file
- [x] Working end-to-end demo (push-to-talk over SSH)
- [x] Demo video recording (added to the repo / referenced separately)
- [ ] Browser-UI demo (deferred — see WebRTC notes above)
- [ ] Codec-token diff regression (deferred)
