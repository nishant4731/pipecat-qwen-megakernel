# pipecat-qwen-megakernel

End-to-end voice agent: **Whisper STT → HF Qwen3-1.7B LLM → Qwen3-TTS (talker on AlpinDale's `qwen_megakernel`) → 24 kHz PCM** — streamed frame-by-frame to a Pipecat pipeline.

Take-home brief: take [AlpinDale's `qwen_megakernel`](https://github.com/AlpinDale/qwen_megakernel) (a hand-written CUDA persistent kernel that decodes Qwen3-0.6B at ~1,000 tok/s on a single RTX 5090) and re-purpose it as the autoregressive decode backend for the **Talker** stage of `Qwen/Qwen3-TTS-12Hz-0.6B-Base`, then wire the result into a Pipecat voice agent.

## TL;DR

All measurements on a single RTX 5090 (sm_120a) on Vast.ai, bf16 throughout, no quantization. Three bench scripts produce these:

| metric | value | brief target |
|---|---|---|
| Megakernel decode on talker shape (`bench_kernel.py`, 1,000 steps) | **662 tok/s · 1.51 ms / step** | reference: 1,036 tok/s on 0.6B-text |
| **TTFC** (`bench_tts.py`, 5 runs, mean / p95) | **36.8 / 37.9 ms** ✅✅ | < 60 ms (strict) / **< 50 ms (stretch) ✅** |
| **RTF** (`bench_tts.py`, 5 runs, mean / p95) | **0.352 / 0.363** | < 0.15 (strict) / < 0.3 (lenient) — between |
| Audio streaming | per-frame `TTSAudioRawFrame` (80 ms / frame) ✅ | not buffered |
| End-to-end voice turn (push-to-talk demo) | STT 280 ms · LLM 1,340 ms · TTS RTF≈0.62 · **~3.5 s** for a 2-3 s reply | report it |
| Numerical parity vs HF reference | intelligible Base voice (by ear) | codec-token diff not yet run — open risk |

**TTFC hits the brief's stretch target (< 50 ms).** RTF (0.35) is between the lenient < 0.3 and strict < 0.15 — we're generating audio at ~2.9× real-time, the strict target asks for 6.7×. Closing the remaining gap requires a manual `torch.cuda.graph()` capture of the per-frame critical path — see "Where the rest is" below.

### Where the rest is

Per-AR-frame breakdown from `bench_stages.py` after the optimization round (was 98.8 ms / frame at baseline):

| stage | baseline ms | now ms | how |
|---|---|---|---|
| Megakernel + composite embed | 1.52 | 1.52 | unchanged — already in budget |
| Talker step (megakernel + compose_embed callback, includes CP) | 76.26 | **25.6** | see decomposition below |
|   ↳ Code Predictor 15-step decode | 74.76 | **~15** | hand-rolled greedy loop + `torch.compile` of the **whole** 15-step function (`max-autotune-no-cudagraphs`) + HF `StaticCache`. Compiling cp.model alone leaves 15 separate dispatcher calls in between; whole-loop compile fuses across them. |
|   ↳ Megakernel decode + python callback overhead | 1.5 + ~9 | 1.5 + ~9 | unchanged — Python callback into compose_embed each step is on the critical path |
|   ↳ Fused codebook gather (replaces 15 separate embedding lookups) | — | ~0.04 | one fancy-indexing into pre-stacked `[15, 2048, H]` tensor |
| `speech_tokenizer.decode` | 22.50 | **1.59** | `torch.compile(reduce-overhead)` on `speech_tokenizer.model.decoder` (pure conv-net, no HF Cache → CUDA Graphs capture cleanly). Bypass the outer wrapper so the compiled module is actually called. |
| **Per-AR-frame total** | **98.78** | **~27.2** | **-72 %** |
| Audio per frame | 80 ms | 80 ms | — |
| → per-frame RTF | 1.235 | **~0.34** | matches measured (0.352) |

### What we tried that didn't work

To close the remaining gap to RTF < 0.15 (need per-frame < 12 ms) we'd have to eliminate the ~9 ms of Python overhead inside `_compose_embed` that runs each step. That overhead is on the megakernel's critical path because the kernel **calls back** into Python to compute the next composite embed between every step. Pure-GPU operations there sum to < 1 ms; the rest is the C++↔Python bridge cost.

Things attempted that didn't reduce that bridge cost:

1. `torch.compile(mode="reduce-overhead")` (CUDA Graphs) on the whole-loop CP decode: gets stuck in a recompile-then-capture loop for 10+ min on this 5-layer model. The `cache_position` changes each step so inductor treats each as a new shape; combined with graph capture on top this becomes intractable.
2. Per-step `cudagraph_mark_step_begin()` between forwards: classifier still flags HF Cache as a mutated input, falls back to no-cudagraphs.
3. Compiling `_compose_embed` itself: hits the same assertion / list-mutation issues that prevent fullgraph compile (frame_codes list append, conditional trailing_text indexing).

To actually hit RTF < 0.15 you'd need a manual `torch.cuda.graph()` capture around the per-frame critical path (compose_embed + CP + speech_tokenizer + megakernel-callback handling), with pre-allocated input/output buffers and indexed-update everywhere. Plus, the megakernel itself would need a small change: skip the Python callback in steady-state and run on a captured-graph schedule. That's real engineering — a few hundred lines plus correctness validation against the HF reference, and a megakernel patch. The clear next step if this project continues.

## Architecture

```
mic ─► push_to_talk_client (laptop, sounddevice)
            │ raw 16 kHz int16 PCM, framed
            ▼  ─[SSH]─►
            push_to_talk_server (5090 box)
                  │
                  ▼
        faster-whisper base.en  (CPU int8; see note)
                  │ user_text
                  ▼
        HF Qwen3-1.7B            (GPU bf16, .generate)
                  │ reply_text
                  ▼
    ┌──── MegakernelQwen3TTSService ────┐
    │  per talker step @ 12.5 Hz:        │
    │    PyTorch composite embed          │
    │       → patched megakernel  ~1 ms   │
    │       → Code Predictor (HF, GPU)    │
    │       → speech_tokenizer (HF, GPU)  │
    │       → int16 PCM @ 24 kHz, 1920 sa │
    │       → TTSAudioRawFrame            │
    └─────────────────────────────────────┘
                  │
                  ▼  ◄─[SSH]─
            push_to_talk_client (sounddevice playback)
```

## Megakernel patches — minimal-surgery approach

We landed three differences between Qwen3-0.6B-text and the talker in a way that needs **only one kernel-source change** (vocab size) and handles the rest in Python:

| Difference | How we handle it | Kernel source change? |
|---|---|---|
| **LM head vocab: 3,072** (codec) vs **151,936** (text) | Make `LDG_VOCAB_SIZE` a build-time `-D` macro; the kernel's LM-head fused block-partition is parameterized accordingly. See [`patches/apply_patches.py`](patches/apply_patches.py). | ✅ ~5 lines in `kernel.cu` + `build.py` |
| **Composite input embedding** (text + speaker + ref-audio + codec-history + code-group) | Compute the per-step `[1, 1024]` bf16 hidden state in PyTorch, write it to a **single-row "fake embed table"**, hand it to the kernel via the existing `embed_weight + token_id * HIDDEN_SIZE` lookup with `token_id=0`. | ❌ |
| **MRoPE sections `[24, 20, 20]`, `rope_theta = 1e6`** | For pure TTS, `Qwen3TTSTalkerModel.get_rope_index` returns position-ids of shape `[3, B, T]` with **all three axes equal**. MRoPE then collapses to plain 1D RoPE. We just precompute cos/sin with `rope_theta=1e6`. | ❌ |

See [`patches/README.md`](patches/README.md) for the longer story. The patcher is idempotent — re-running it is a no-op.

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
    qwen3_llm_local.py        # HF Qwen3-Instruct LLMService (streaming)
  app/
    demo.py                   # SmallWebRTC variant (see "WebRTC notes" below)
    push_to_talk_server.py    # the working demo path — runs on the box
    push_to_talk_client.py    # the working demo path — runs on your laptop
  bench/
    bench_kernel.py           # kernel-only microbench (tok/s, ms/step)
    bench_tts.py              # end-to-end TTS bench (TTFC, RTF)
    bench_stages.py           # per-stage timings (talker / code_predictor / speech_tokenizer)

scripts/
  setup.sh                    # one-shot bringup for the 5090 box
docs/
  architecture.md             # design rationale, what we chose not to optimize
```

## Bringing up the 5090 box

1. **Rent on Vast.ai.** Filters: RTX 5090, driver ≥ 570, CUDA ≥ 12.8, ≥ 32 GB RAM, ≥ 80 GB disk. Image: prefer `pytorch:2.7-cuda12.8`; alternatively `nvidia/cuda:12.8.0-devel-ubuntu22.04`. Open ports 22 (SSH) + 7860.
2. **SSH in, `scp` / `git clone` this repo to `/workspace/task`.**
3. **Run `bash scripts/setup.sh`.** Verifies the GPU is sm_120, installs torch nightly + transformers + qwen-tts, clones and patches the megakernel, builds it.
4. **Smoke-test the megakernel** with AlpinDale's reference Qwen3-0.6B bench:
   ```bash
   cd refs/qwen_megakernel && python -m qwen_megakernel.bench
   ```
   Expect ~1,036 tok/s. If this doesn't hit, the build environment or sm_120a flag is wrong — stop and investigate before going further.

## Running the demo

The demo runs **on the 5090 box** with audio piped over SSH to **your laptop**. This avoids the WebRTC + NAT issues that block a hosted browser demo from Vast.ai (see below).

**On your laptop**, install one Python dep (no brew/ffmpeg required):

```bash
pip3 install --user sounddevice numpy
```

Then run the client:

```bash
python3 -m pipecat_qwen_megakernel.app.push_to_talk_client --ssh <your-ssh-alias>
```

What you'll see:
1. First run takes ~30 s to load Whisper + Qwen3-1.7B + Qwen3-TTS on the box
2. macOS will prompt for **mic permission** — allow it
3. Once `[INFO] ready` shows: press **ENTER**, speak a short sentence, press **ENTER** again
4. `[STT] ...` → `[LLM] ...` → audio plays through your speakers
5. `[INFO] turn: STT 280ms · LLM 1340ms · TTS 2.10s @ RTF 0.89 · total 3720ms` — per-turn timing

## Running the benches

After `setup.sh`:

```bash
# Kernel-only microbench. Reports tok/s and ms/step. Our talker-shape number
# is 662 tok/s / 1.51 ms / step; AlpinDale's text-shape reference is ~1036.
LDG_VOCAB_SIZE=3072 python -m pipecat_qwen_megakernel.bench.bench_kernel --steps 1000

# End-to-end TTS bench — TTFC, RTF across N utterances on a fixed prompt.
python -m pipecat_qwen_megakernel.bench.bench_tts --runs 5 \
    --text "The five boxing wizards jump quickly."

# Per-stage timings — CUDA-synced ms for talker_step / code_predictor / speech_tokenizer.
# Tells you *where* the per-frame budget went.
python -m pipecat_qwen_megakernel.bench.bench_stages --runs 3
```

## WebRTC / browser-UI demo (deferred)

There's also a `SmallWebRTCTransport` variant at `pipecat_qwen_megakernel/app/demo.py` intended to let evaluators connect from a browser on their laptop. It runs end-to-end, but **does not currently connect from a browser to a bot on Vast.ai**:

- Vast.ai blocks inbound UDP, so direct host candidates can't reach the bot — TURN is mandatory.
- We wired Cloudflare TURN (ephemeral creds via REST API). Cloudflare's TURN ALLOCATE works correctly with aiortc/aioice, but its **CHANNEL-BIND** response is a `401 ERROR-CODE` *with no `NONCE`/`REALM` attributes*, even with a freshly minted credential pair. aioice 0.10's `request_with_retry` only re-auths when those attributes are present, so the retry path doesn't fire and ICE never connects.
- We tried a monkey-patch that re-primes credentials from an unauthed probe (which **does** return `NONCE`/`REALM`) — but the retried `CHANNEL-BIND` with the fresh creds gets the same `401`. This looks like an aiortc/aioice protocol gap with Cloudflare TURN — likely missing CREATE-PERMISSION before CHANNEL-BIND — rather than something fixable in this project.
- Alternative paths that would unblock this: LiveKit Cloud (SFU bypasses the issue; free tier, no card), Daily.co (requires payment method), or self-hosted coturn with a public IP.

The push-to-talk path exercises the **same** STT → LLM → megakernel-TTS pipeline frame-by-frame and is what the demo recording uses.

## Honest notes

- **Whisper STT currently runs on CPU.** The Vast.ai image we used ships CUDA 13; ctranslate2 (faster-whisper's backend) has no CUDA-13 wheel yet, so `device="cuda"` fails with `libcublas.so.12 not found`. `base.en` on CPU int8 takes ~200–400 ms per short utterance — fine for an interactive demo. CUDA-12 deps (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`) installed via pip would restore GPU inference.
- **TTFC 105 ms, RTF 1.22, missed both stretch targets by a wide margin.** Honest because the per-stage bench (see TL;DR) shows the kernel is fine — 98% of wall time is in HF code paths the brief didn't ask us to optimize. STATUS.md has the full breakdown plus the order in which to attack: hand-rolled 15-step Code Predictor decode, `torch.compile` on speech_tokenizer, optional LLM token-streaming.
- The **correctness gate (codec-token diff vs stock HF Qwen3-TTS)** was not exhaustively run as a regression — we instead validated by ear (audio is recognizable speech using the default Base voice) plus matching dimensions and special-token IDs against `modeling_qwen3_tts.py`. A proper element-wise codec-token diff is the next thing to add if you're investing further.

## Why we didn't try to also accelerate Code Predictor / vocoder

The brief explicitly targets the **talker**. The Code Predictor is a 5-layer transformer and the speech-tokenizer-as-vocoder is a causal ConvNet — both small. Premature optimization risks numerical drift vs the HF reference, and we wanted the canary (codec-token diff) to be against unmodified upstream code. They are now the biggest contributors to RTF; if this project continues, that's where to look next.

## License

Apache-2.0 for our additions. `qwen_megakernel` is Apache-2.0 (AlpinDale). Qwen3-TTS weights are governed by Qwen's model license.
