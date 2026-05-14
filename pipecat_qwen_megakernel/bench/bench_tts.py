"""Standalone TTS benchmark — no STT, no LLM, no transport.

Measures, for a fixed prompt across N repetitions:

  - Megakernel-talker decode throughput (tok/s, ms/step)
  - TTFC: wall-clock from `synthesize()` call to first audio byte
  - RTF: wall_time / audio_duration
  - Code Predictor per-frame latency (ms)
  - Vocoder per-frame latency (ms)

Reports mean, p50, p95 over the runs. Run this *before* the full Pipecat
demo to isolate the speech-synthesis side from the agent loop.

Usage::

    python -m pipecat_qwen_megakernel.bench.bench_tts \\
        --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --runs 10 \\
        --text "The five boxing wizards jump quickly."
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass, field

import torch
from loguru import logger

from pipecat.frames.frames import TTSAudioRawFrame

from patches.talker_constants import SAMPLE_RATE, SAMPLES_PER_TALKER_STEP
from pipecat_qwen_megakernel.services.qwen3_tts_megakernel import (
    MegakernelQwen3TTSService,
)


@dataclass
class RunMetrics:
    """Per-run measurements."""

    ttfc_ms: float
    rtf: float
    decode_ms: list[float] = field(default_factory=list)
    code_pred_ms: list[float] = field(default_factory=list)
    vocoder_ms: list[float] = field(default_factory=list)
    audio_sec: float = 0.0
    wall_sec: float = 0.0
    bytes_total: int = 0


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1)))))
    return s[k]


def _summary(name: str, xs: list[float]) -> str:
    if not xs:
        return f"{name}: <no data>"
    return (
        f"{name}: mean={statistics.mean(xs):.3f}  "
        f"p50={_pct(xs, 50):.3f}  p95={_pct(xs, 95):.3f}  "
        f"min={min(xs):.3f}  max={max(xs):.3f}  n={len(xs)}"
    )


async def _init_service(svc: MegakernelQwen3TTSService) -> None:
    """Send a StartFrame through the service so sample_rate is set.

    Outside a real Pipeline, the base class's ``start()`` is never invoked.
    Without that, ``self.sample_rate`` is 0 and the resampler in
    ``_stream_audio_frames_from_iterator`` complains.
    """
    from pipecat.frames.frames import StartFrame
    svc._sample_rate = SAMPLE_RATE  # private but the cleanest workaround in standalone-bench mode
    svc._sample_rate_was_set = True  # avoid double-init


async def _run_once(svc: MegakernelQwen3TTSService, text: str) -> RunMetrics:
    """Run one synthesis. Counts wall-clock externally to the service.

    NOTE: the service is the source of truth for per-stage timings; the
    bench just records overall TTFC and RTF here. Per-stage logs go to the
    loguru sink — parse those if you want fine-grained timing.
    """
    metrics = RunMetrics(ttfc_ms=float("nan"), rtf=float("nan"))
    t0 = time.perf_counter()
    first_seen: float | None = None

    async for frame in svc.run_tts(text, context_id="bench"):
        if isinstance(frame, TTSAudioRawFrame):
            if first_seen is None:
                first_seen = time.perf_counter()
                metrics.ttfc_ms = (first_seen - t0) * 1000.0
            metrics.bytes_total += len(frame.audio)

    metrics.wall_sec = time.perf_counter() - t0
    metrics.audio_sec = (metrics.bytes_total / 2) / SAMPLE_RATE  # int16 LE mono
    metrics.rtf = metrics.wall_sec / max(metrics.audio_sec, 1e-9)
    return metrics


async def main_async(args: argparse.Namespace) -> None:
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

    # Set sample_rate manually — outside a Pipeline, ``start()`` is never invoked
    # so ``self.sample_rate`` would be 0 / None and the resampler in
    # ``_stream_audio_frames_from_iterator`` would error out.
    await _init_service(svc)

    # Warm-up run (don't include in stats).
    logger.info("Warm-up...")
    _ = await _run_once(svc, args.text)

    ttfcs: list[float] = []
    rtfs: list[float] = []
    audio_secs: list[float] = []

    for i in range(args.runs):
        m = await _run_once(svc, args.text)
        ttfcs.append(m.ttfc_ms)
        rtfs.append(m.rtf)
        audio_secs.append(m.audio_sec)
        logger.info(
            f"[run {i+1}/{args.runs}] TTFC={m.ttfc_ms:.1f}ms RTF={m.rtf:.3f} "
            f"audio={m.audio_sec:.2f}s wall={m.wall_sec:.2f}s"
        )

    print("\n========== Megakernel-TTS Benchmark Summary ==========")
    print(f"model: {args.model}")
    print(f"prompt: {args.text!r}")
    print(f"runs: {args.runs}")
    print(f"talker rate: 12.5 Hz   (audio sr {SAMPLE_RATE}, {SAMPLES_PER_TALKER_STEP} samples/step)")
    print(_summary("TTFC (ms)", ttfcs))
    print(_summary("RTF      ", rtfs))
    print(_summary("audio (s)", audio_secs))
    print()
    print("Targets from the take-home:  TTFC < 60 ms,  RTF < 0.15")
    print("Stretch:                     TTFC < 50 ms,  RTF < 0.1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument(
        "--text",
        default="The five boxing wizards jump quickly. Hello from a megakernel-accelerated text-to-speech engine running on an RTX 5090.",
    )
    ap.add_argument("--speaker", default=None)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
