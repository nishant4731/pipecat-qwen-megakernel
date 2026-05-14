"""Megakernel-only microbenchmark.

Runs N decode steps on the patched kernel with talker weights but a dummy
composite-embed (zeros). Reports tok/s and ms/step.

This is the kernel-only number you'd compare against AlpinDale's 1036 tok/s
on Qwen3-0.6B. Expectation: roughly identical, since the only kernel-level
change in our build is a smaller LM head (3072 vs 151936) which is *faster*,
not slower.

Usage::

    python -m pipecat_qwen_megakernel.bench.bench_kernel --steps 1000
"""

from __future__ import annotations

import argparse
import time

import torch
from loguru import logger

from patches.talker_constants import HIDDEN_SIZE
from patches.talker_model import TalkerDecoder, load_talker_weights


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=50)
    args = ap.parse_args()

    logger.info(f"Loading talker weights from {args.model}")
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration

    hf = Qwen3TTSForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda",
    )
    weights = load_talker_weights(hf, verbose=True)

    dummy_hidden = torch.zeros(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    td = TalkerDecoder(weights, compose_embed_fn=lambda **kw: dummy_hidden)
    td.reset()

    # Warmup.
    for i in range(args.warmup):
        td.step(frame_idx=i)
    torch.cuda.synchronize()

    td.reset()
    t0 = time.perf_counter()
    for i in range(args.steps):
        td.step(frame_idx=i)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    toks_per_sec = args.steps / dt
    ms_per_tok = (dt / args.steps) * 1000.0
    print()
    print("========== Megakernel kernel-only microbench ==========")
    print(f"steps:     {args.steps}")
    print(f"wall:      {dt*1000:.1f} ms")
    print(f"tok/s:     {toks_per_sec:.1f}")
    print(f"ms/tok:    {ms_per_tok:.3f}")
    print()
    print("AlpinDale's reference (Qwen3-0.6B text):  ~1036 tok/s, 0.99 ms/tok")


if __name__ == "__main__":
    main()
