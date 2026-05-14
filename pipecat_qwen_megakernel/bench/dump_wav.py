"""Synthesize one utterance through MegakernelQwen3TTSService and save as WAV.

Used to inspect audio quality. We run the synthesis with a low max_steps cap
so we don't wait 60 s every time the talker fails to emit EOS.

Usage::
    python -m pipecat_qwen_megakernel.bench.dump_wav \
        --text "Hello world." --out /tmp/dump.wav --max-steps 80
"""

from __future__ import annotations

import argparse
import asyncio

import numpy as np
import soundfile as sf
import torch
from loguru import logger

from pipecat.frames.frames import TTSAudioRawFrame

from patches.talker_constants import SAMPLE_RATE
from pipecat_qwen_megakernel.services.qwen3_tts_megakernel import (
    MegakernelQwen3TTSService,
)


async def main_async(args: argparse.Namespace) -> None:
    logger.info(f"Loading {args.model}")
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
    from transformers import AutoTokenizer

    hf = Qwen3TTSForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda",
    )
    hf.eval()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    svc = MegakernelQwen3TTSService(
        hf_model=hf, tokenizer=tok, max_steps=args.max_steps,
    )
    svc._sample_rate = SAMPLE_RATE  # bench-mode init

    pcm_chunks: list[bytes] = []
    async for frame in svc.run_tts(args.text, context_id="dump"):
        if isinstance(frame, TTSAudioRawFrame):
            pcm_chunks.append(frame.audio)

    pcm = b"".join(pcm_chunks)
    arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0
    sf.write(args.out, arr, SAMPLE_RATE)
    logger.info(
        f"wrote {args.out}: {len(arr)/SAMPLE_RATE:.2f}s audio, "
        f"int16 range [{arr.min():.3f}, {arr.max():.3f}], "
        f"rms={float(np.sqrt((arr**2).mean())):.3f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    ap.add_argument("--text", default="Hello world, this is a test.")
    ap.add_argument("--out", default="/tmp/dump.wav")
    ap.add_argument("--max-steps", type=int, default=80, help="Cap talker steps (12.5 Hz -> 80 = 6.4 s)")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
