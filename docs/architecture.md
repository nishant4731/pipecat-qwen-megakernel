# Architecture decisions

This document captures the *why* behind each non-obvious choice. The README has the *what*.

## 1. We do NOT modify the kernel source for the composite embedding (Patch B)

The talker's input is a sum of five embedding components (text, speaker, ref-audio, codec-history, code-group). Implementing all five inside `kernel.cu` would mean:

- A bunch of new kernel parameters (5 weight pointers + per-component lookups).
- Tighter constraints on launch occupancy (more shared/L1 pressure).
- Numerical-parity hell when reconciling against the stock HF forward.

Instead, we exploit a property of the existing kernel: the first layer reads `embed_row = embed_weight + input_token_id * HIDDEN_SIZE`. If we hand the kernel a 1-row "fake" embedding table containing our pre-composed hidden state, and pass `input_token_id = 0`, the lookup returns our composite hidden directly. The kernel is *unchanged*.

Cost: a single 1 KB `cuda::memcpy` per talker step, which is invisible against the ~1 ms decode budget.

## 2. We do NOT modify the kernel source for MRoPE (Patch C)

MRoPE was introduced for Qwen2.5-Omni so that one model could handle text + audio + image tokens whose positions live on different axes (temporal, height, width). For *pure-TTS* inference, the talker only produces audio tokens — height and width position indices are always 0. That means cosine/sine in those sections evaluate to 1 and 0, i.e. **identity rotation**.

So MRoPE-with-mostly-zeros looks exactly like single-axis RoPE if we precompute the cos/sin tables that way. We do that in `patches/talker_model.py::build_mrope_tables` — set `freqs[:, sections[0]:] = 0` before `cos`/`sin`. The kernel's existing RoPE math is unchanged.

This is the riskiest place in the design: if a future Qwen-TTS variant *does* drive height/width during TTS, this assumption breaks. We document this as a fast-path and leave a clear extension point.

## 3. We DO change `LDG_VOCAB_SIZE` (Patch A)

The codec vocab is 3,072. The text vocab is 151,936. The LM-head kernel iterates over the full vocab to compute partial-argmax, and the block partition (`LDG_LM_NUM_BLOCKS = 1280`) is tuned for ~152 k rows. Leaving these unchanged with a 3,072-row LM head would over-provision 50× and waste SM cycles.

So we make `LDG_VOCAB_SIZE` a `-D` macro (build-time) and pick a saner default for `LDG_LM_NUM_BLOCKS` when the vocab is small (`64` blocks for vocab < 16k; `1280` otherwise). Both stay env-overridable.

## 4. Why we use `step()` instead of `generate_nosync` for streaming

The kernel ships a `generate_nosync` entrypoint that queues N decode + LM-head + position-update launches in one Python call, then `cudaStreamSynchronize` at the end. It's optimized for batch throughput — *zero* CPU↔GPU coordination cost per token, hitting the 1,036 tok/s number.

That mode is **wrong for our use case**: we *want* per-step coordination, because each codec token has to flow through the Code Predictor + Vocoder *before* we can emit audio, and the next step's composite embed depends on the previous codec token. So we use the synchronous `step(token_id)` path in a Python loop. Cost: one `cudaStreamSynchronize` per step (~10 μs), negligible against ~1 ms decode.

At 12.5 Hz this Python loop runs 12.5 times per second of audio. Even if the loop itself adds 100 μs of Python overhead per step, that's 1.25 ms per second of audio — invisible in the RTF budget.

## 5. Why a thread + asyncio.Queue rather than direct async generation

CUDA calls are synchronous from Python's perspective and *don't* yield the event loop. Doing the decode loop on the asyncio thread would block every other Pipecat processor (STT, transport, VAD) for the duration of an utterance. So we run the CUDA loop in a `asyncio.to_thread`-spawned worker thread that pushes byte chunks onto an `asyncio.Queue`. The async iterator on the main loop just `await`s the queue and yields `TTSAudioRawFrame`s.

This is the same pattern as Pipecat's Piper TTS service (`pipecat/services/piper/tts.py` lines 151–160).

## 6. Why a self-hosted WebRTC transport instead of LocalAudioTransport

The compute lives on a Vast.ai instance with no microphone or speakers. `LocalAudioTransport` opens local PyAudio — there's nothing to open. Options:

- **`SmallWebRTCTransport`** (our choice): Pipecat-bundled, self-hosted, talks to a browser client. No third-party signup, no API key. The transport serves a static HTML page; the user hits it from their laptop.
- **`DailyTransport`**: more polished, but requires a Daily account and an API key. Free tier is generous but the brief asked for local-only.
- **File I/O**: feasible but loses "live round-trip" from the brief.

## 7. Why we keep the Code Predictor + Vocoder in stock PyTorch

The take-home is explicit: optimize the **talker**, not the codebook generator. At 12.5 Hz, both downstream stages run easily inside the per-frame budget on a 5090. Hand-porting them to CUDA would:

- Risk numerical drift vs. the reference waveform.
- Double the surface area of "things to validate against HF."
- Eat hours that should go into actually getting the pipeline to work and benched.

If measurement shows one of them dominating the per-frame budget, the first move is `torch.compile`, not a hand kernel.

## 8. Why a local Qwen3-Instruct LLM rather than a hosted one

User pushed back on API costs; the box has spare GPU memory anyway (1.7B Qwen3-Instruct in bf16 is ~3.5 GB, plus ~1.2 GB talker, plus ~50 MB Whisper-base — fits in well under half of the 5090's 32 GB).

We run it via HF `generate` with `TextIteratorStreamer` in a worker thread — simple and avoids the complexity of standing up vLLM or Ollama as a sidecar.

## 9. Validation strategy

The single most informative check is: **do the codec tokens match the stock HF reference?** Argmax in bf16 is deterministic modulo tiny numerical drift, so the patched megakernel and the HF forward should produce nearly-identical token streams for the same prompt. We do this before any benchmarking — if codec tokens diverge after token N, that pinpoints which patch (A, B, or C) is wrong, at which layer.

Waveforms are compared by ear *and* by L2 distance, but the codec-token diff is the primary canary.

## 10. Out of scope

- Quantization. Brief says no.
- Multi-utterance batching. Voice agents don't have a batch dimension.
- KV-cache sharing across utterances. Each utterance starts fresh; the cost is amortized.
- Production-grade hardening (multi-process serving, graceful shutdown beyond basic asyncio cancellation, retry/backoff).
