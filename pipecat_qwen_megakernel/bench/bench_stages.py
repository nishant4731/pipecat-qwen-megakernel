"""Per-stage TTS timing bench.

Wraps the three GPU stages inside MegakernelQwen3TTSService with
``time.perf_counter()`` + ``torch.cuda.synchronize()`` and reports
mean/p95 ms per AR frame for:

  * talker_step        — patched megakernel step + composite embed
  * code_predictor     — HF code_predictor.generate(max_new_tokens=15)
  * speech_tokenizer   — self._hf.speech_tokenizer.decode([{...}])

Run after bench_tts.py — the overall TTFC/RTF totals come from there;
this one tells you where the budget went.

Usage::

    python -m pipecat_qwen_megakernel.bench.bench_stages \\
        --runs 3 --text "The five boxing wizards jump quickly."
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import torch
from loguru import logger

from pipecat.frames.frames import TTSAudioRawFrame

from patches.talker_constants import SAMPLE_RATE
from pipecat_qwen_megakernel.services.qwen3_tts_megakernel import (
    MegakernelQwen3TTSService,
)


def _summary(name: str, xs: list[float]) -> str:
    if not xs:
        return f"{name:<22} n=0"
    return (
        f"{name:<22} mean={statistics.mean(xs):7.2f}  "
        f"p50={statistics.median(xs):7.2f}  "
        f"p95={sorted(xs)[max(int(len(xs)*0.95)-1, 0)]:7.2f}  "
        f"min={min(xs):7.2f}  max={max(xs):7.2f}  n={len(xs)}"
    )


def _install_timers(svc: MegakernelQwen3TTSService):
    """Monkey-patch the three stages to record CUDA-synced wall ms per call."""
    talker_ms: list[float] = []
    code_pred_ms: list[float] = []
    decode_ms: list[float] = []

    orig_step = svc._talker.step
    orig_code_pred = svc._code_predictor.generate
    orig_decode = svc._decode_frame_to_audio

    def timed_step(*a, **kw):
        torch.cuda.synchronize()
        t = time.perf_counter()
        r = orig_step(*a, **kw)
        torch.cuda.synchronize()
        talker_ms.append((time.perf_counter() - t) * 1000.0)
        return r

    def timed_code_pred(*a, **kw):
        torch.cuda.synchronize()
        t = time.perf_counter()
        r = orig_code_pred(*a, **kw)
        torch.cuda.synchronize()
        code_pred_ms.append((time.perf_counter() - t) * 1000.0)
        return r

    def timed_decode(*a, **kw):
        torch.cuda.synchronize()
        t = time.perf_counter()
        r = orig_decode(*a, **kw)
        torch.cuda.synchronize()
        decode_ms.append((time.perf_counter() - t) * 1000.0)
        return r

    svc._talker.step = timed_step
    svc._code_predictor.generate = timed_code_pred
    svc._decode_frame_to_audio = timed_decode

    return talker_ms, code_pred_ms, decode_ms


async def _init_service(svc):
    svc._sample_rate = SAMPLE_RATE
    svc._sample_rate_was_set = True


async def _drain(svc, text: str) -> tuple[float, float]:
    """Run one synthesis; return (wall_s, audio_s)."""
    t0 = time.perf_counter()
    samples = 0
    async for frame in svc.run_tts(text, context_id="bench_stages"):
        if isinstance(frame, TTSAudioRawFrame):
            samples += len(frame.audio) // 2
    wall = time.perf_counter() - t0
    audio = samples / SAMPLE_RATE
    return wall, audio


async def main_async(args):
    logger.info(f"Loading TTS model: {args.model}")
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
    from transformers import AutoTokenizer

    hf_tts = Qwen3TTSForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda",
    )
    hf_tts.eval()
    tts_tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    svc = MegakernelQwen3TTSService(
        hf_model=hf_tts,
        tokenizer=tts_tokenizer,
        speaker=args.speaker,
    )
    await _init_service(svc)

    talker_ms, code_pred_ms, decode_ms = _install_timers(svc)

    # Warm-up (don't include).
    logger.info("Warm-up …")
    _ = await _drain(svc, args.text)
    talker_ms.clear(); code_pred_ms.clear(); decode_ms.clear()

    walls: list[float] = []
    audios: list[float] = []
    for i in range(args.runs):
        wall, audio = await _drain(svc, args.text)
        walls.append(wall)
        audios.append(audio)
        logger.info(f"[run {i+1}/{args.runs}] wall={wall:.2f}s audio={audio:.2f}s rtf={wall/audio:.3f}")

    print("\n========== Per-stage timings (ms / call) ==========")
    print(f"model:     {args.model}")
    print(f"runs:      {args.runs}")
    print(f"prompt:    {args.text!r}")
    print(f"AR frames totalled: talker={len(talker_ms)} code_pred={len(code_pred_ms)} decode={len(decode_ms)}")
    print()
    # code_predictor.generate is called inside _talker.step (via _compose_embed),
    # so talker_step time double-counts code_predictor time. Subtract pairwise
    # to isolate "talker_step excluding code_predictor".
    # We have len(code_pred_ms) = len(talker_ms) + 1 (one extra call for the
    # first-frame priming path before the AR loop). Align lists from the right.
    n = min(len(talker_ms), len(code_pred_ms))
    talker_ex_cp_ms = [
        max(0.0, t - c) for t, c in zip(talker_ms[-n:], code_pred_ms[-n:])
    ]
    print(_summary("talker_step (raw)", talker_ms))
    print(_summary("  └ code_predictor", code_pred_ms))
    print(_summary("  └ talker_ex_cp",   talker_ex_cp_ms))
    print(_summary("speech_tokenizer",   decode_ms))
    print()
    if talker_ex_cp_ms and code_pred_ms and decode_ms:
        mean_ar = (
            statistics.mean(talker_ex_cp_ms)
            + statistics.mean(code_pred_ms)
            + statistics.mean(decode_ms)
        )
        print(f"per-AR-frame total (sum of means): {mean_ar:.2f} ms")
        print(f"audio per AR frame: 80 ms")
        print(f"=> per-frame RTF: {mean_ar/80.0:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument(
        "--text",
        default="The five boxing wizards jump quickly.",
    )
    ap.add_argument("--speaker", default=None)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
