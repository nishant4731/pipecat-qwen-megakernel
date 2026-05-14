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

Methodology: `python -m pipecat_qwen_megakernel.bench.bench_tts --runs 5 --text "The five boxing wizards jump quickly."` (audio = 3.04 s / utterance, AR_steps = 37, prefill rows = 9, text trail = 7). Mean / p95 / min / max over 5 runs:

| metric | mean | p95 | min | max |
|---|---|---|---|---|
| Megakernel talker decode | ~1 ms / step | — | — | — |
| TTFC | **105.1 ms** | 105.3 | 104.9 | 105.3 |
| RTF | **1.225** | 1.227 | 1.224 | 1.227 |
| Wall per utterance | 3.72 s | 3.73 | 3.72 | 3.73 |

Targets from the brief were TTFC < 60 ms and RTF < 0.15 — we missed both, by a lot. The megakernel itself is well inside budget. The dominant per-frame costs are upstream-side: `code_predictor.generate(max_new_tokens=15, ...)` per talker frame (Python `.generate()` overhead) and `speech_tokenizer.decode([{"audio_codes": codes}])` for a single-frame slice. Each full AR step costs **~98 ms wall** for 80 ms of audio → RTF > 1. See "Where the budget went" below.

## What changed in the megakernel (the actual patches)

Single source change in `qwen_megakernel`: `LDG_VOCAB_SIZE` becomes a build-time `-D` macro (~5 lines across `kernel.cu` and `build.py`). Patches/apply_patches.py is idempotent.

Everything else is handled in Python without touching the kernel:

- **Composite embedding** (text + speaker + ref-audio + codec-history + code-group): computed each step in PyTorch, written into a 1-row "fake embed table" that the kernel reads with `token_id=0`.
- **MRoPE**: `Qwen3TTSTalkerModel.get_rope_index` returns `[3, B, T]` position-ids with all three axes equal during TTS inference. The interleaved MRoPE writes then collapse to plain 1D rotation. We just precompute cos/sin with `rope_theta=1e6`.
- **LM head wiring**: `talker.codec_head` (nn.Linear 1024→3072) is loaded as the kernel's LM head; talker layer weights come from the live `talker.model.layers.*` module tree.
- **Codec EOS**: `codec_eos_token_id=2150` from `talker_config` (not the text-side `tts_eos_token_id=151673`).

## Where the budget went (and what we didn't optimize)

| stage | per-frame (~80 ms audio) | notes |
|---|---|---|
| talker megakernel step | ~1 ms | in budget |
| code_predictor.generate | dominant | un-tuned HF `.generate(max_new_tokens=15)`; manual decode loop would help |
| speech_tokenizer.decode | dominant | called per single-frame slice; batching N frames per call (with a small extra delay) trades RTF for TTFC |
| LLM | not per-frame | one-shot `.generate(max_new_tokens=80)` blocks the LLM stage; streaming token-by-token would let TTS start earlier |

We **explicitly chose not to optimize Code Predictor or speech tokenizer**. The brief targets the talker; touching the others risks numerical drift vs the HF reference and the codec-token diff (the correctness canary) becomes harder to read. They are the biggest remaining levers if the project continues.

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
