"""End-to-end Pipecat voice agent demo using Daily.co's hosted WebRTC SFU.

Why Daily instead of SmallWebRTCTransport: aiortc (the WebRTC engine inside
``SmallWebRTCTransport``) has known issues with NAT traversal and TURN servers.
Daily.co provides a hosted SFU + TURN that just works — browser ↔ Daily ↔ bot
all flow through their servers, so no port mapping or TURN signups are needed.

Pipeline (identical to the SmallWebRTC version):

    browser-mic ─[Daily WebRTC]─►  DailyTransport.input
                                          │
                                          ▼
                              SileroVADAnalyzer
                                          │
                                          ▼
                              WhisperSTTService (faster-whisper, GPU)
                                          │
                                          ▼
                            Qwen3LocalLLMService (HF Qwen3-1.7B, GPU)
                                          │
                                          ▼
                  MegakernelQwen3TTSService  ← the talker on the megakernel
                                          │
                                          ▼
                          DailyTransport.output ─[Daily WebRTC]─► browser
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

import torch
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.daily.transport import DailyParams, DailyTransport

from pipecat_qwen_megakernel.services.qwen3_llm_local import (
    Qwen3LocalLLMService,
    Qwen3LocalLLMSettings,
)
from pipecat_qwen_megakernel.services.qwen3_tts_megakernel import (
    MegakernelQwen3TTSService,
)


# Lazy-load TTS once per process.
_hf_tts = None
_tts_tokenizer = None


def _load_tts_once(model_name: str):
    global _hf_tts, _tts_tokenizer
    if _hf_tts is not None:
        return _hf_tts, _tts_tokenizer
    logger.info(f"Loading TTS model: {model_name}")
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
    from transformers import AutoTokenizer
    _hf_tts = Qwen3TTSForConditionalGeneration.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda",
    )
    _hf_tts.eval()
    _tts_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return _hf_tts, _tts_tokenizer


async def run_bot(room_url: str, token: str | None) -> None:
    tts_model_name = os.getenv("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    llm_model_name = os.getenv("QWEN_LLM_MODEL", "Qwen/Qwen3-1.7B")
    whisper_model = os.getenv("WHISPER_MODEL", "base.en")

    logger.info(f"Bot joining Daily room: {room_url}")
    logger.info(
        f"TTS={tts_model_name} LLM={llm_model_name} STT=Whisper({whisper_model})"
    )

    hf_tts, tts_tokenizer = _load_tts_once(tts_model_name)

    transport = DailyTransport(
        room_url=room_url,
        token=token,
        bot_name="megakernel-qwen-tts",
        params=DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_out_sample_rate=24_000,
            vad_analyzer=SileroVADAnalyzer(),
            transcription_enabled=False,  # we use Whisper locally
        ),
    )

    stt = WhisperSTTService(model=whisper_model, device="cuda")

    llm = Qwen3LocalLLMService(
        settings=Qwen3LocalLLMSettings(model_name=llm_model_name),
    )

    tts = MegakernelQwen3TTSService(
        hf_model=hf_tts,
        tokenizer=tts_tokenizer,
        speaker=os.getenv("QWEN_TTS_SPEAKER", None),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            audio_in_sample_rate=16_000,
            audio_out_sample_rate=24_000,
        ),
    )

    @transport.event_handler("on_first_participant_joined")
    async def _on_joined(transport, participant):
        logger.info(f"Participant joined: {participant['id']}")
        await transport.capture_participant_transcription(participant["id"])
        context.add_message({
            "role": "developer",
            "content": "Please greet the user briefly and ask how you can help.",
        })
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_participant_left")
    async def _on_left(transport, participant, reason):
        logger.info(f"Participant left: {participant['id']} ({reason})")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=True)
    await runner.run(task)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", required=True, help="Daily room URL")
    ap.add_argument("--token", default=None, help="Daily room token (optional, for private rooms)")
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    asyncio.run(run_bot(args.room, args.token))


if __name__ == "__main__":
    main()
