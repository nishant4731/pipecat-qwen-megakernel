"""End-to-end Pipecat voice agent demo using LiveKit Cloud's hosted SFU.

LiveKit Cloud's free tier has no card requirement and handles all the WebRTC
NAT/firewall complexity for us — browser ↔ LiveKit ↔ bot.

Pipeline (identical to other variants):

    browser-mic ─[LiveKit WebRTC]─►  LiveKitTransport.input
                                            │
                                            ▼
                                  SileroVADAnalyzer
                                            │
                                            ▼
                                  WhisperSTTService (GPU)
                                            │
                                            ▼
                              Qwen3LocalLLMService (GPU)
                                            │
                                            ▼
                  MegakernelQwen3TTSService  ← the talker on the megakernel
                                            │
                                            ▼
                        LiveKitTransport.output ─[LiveKit WebRTC]─► browser
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
from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport

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


def _mint_token(api_key: str, api_secret: str, room_name: str, identity: str) -> str:
    """Mint a short-lived LiveKit access token (JWT) for a given identity + room."""
    from livekit import api  # type: ignore
    token = (
        api.AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )
    return token


async def run_bot(ws_url: str, room_name: str, api_key: str, api_secret: str) -> None:
    tts_model_name = os.getenv("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    llm_model_name = os.getenv("QWEN_LLM_MODEL", "Qwen/Qwen3-1.7B")
    whisper_model = os.getenv("WHISPER_MODEL", "base.en")

    logger.info(f"Bot joining LiveKit room: {room_name} at {ws_url}")
    logger.info(
        f"TTS={tts_model_name} LLM={llm_model_name} STT=Whisper({whisper_model})"
    )

    bot_token = _mint_token(api_key, api_secret, room_name, "megakernel-bot")

    hf_tts, tts_tokenizer = _load_tts_once(tts_model_name)

    transport = LiveKitTransport(
        url=ws_url,
        token=bot_token,
        room_name=room_name,
        params=LiveKitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_out_sample_rate=24_000,
            vad_analyzer=SileroVADAnalyzer(),
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

    @transport.event_handler("on_participant_connected")
    async def _on_joined(transport, participant_id):
        logger.info(f"Participant joined: {participant_id}")
        context.add_message({
            "role": "developer",
            "content": "Please greet the user briefly and ask how you can help.",
        })
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_participant_disconnected")
    async def _on_left(transport, participant_id):
        logger.info(f"Participant left: {participant_id}")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=True)
    await runner.run(task)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws-url", default=os.getenv("LIVEKIT_URL"), required=False)
    ap.add_argument("--api-key", default=os.getenv("LIVEKIT_API_KEY"), required=False)
    ap.add_argument("--api-secret", default=os.getenv("LIVEKIT_API_SECRET"), required=False)
    ap.add_argument("--room", default=os.getenv("LIVEKIT_ROOM", "megakernel-demo"))
    args = ap.parse_args()

    missing = [k for k, v in dict(
        ws_url=args.ws_url, api_key=args.api_key, api_secret=args.api_secret
    ).items() if not v]
    if missing:
        print(f"ERROR: missing required args: {missing}", file=sys.stderr)
        print("Set --ws-url, --api-key, --api-secret or env LIVEKIT_URL/LIVEKIT_API_KEY/LIVEKIT_API_SECRET", file=sys.stderr)
        sys.exit(1)

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Print a clickable user URL (LiveKit Meet UI with our credentials).
    user_token = _mint_token(args.api_key, args.api_secret, args.room, "user")
    livekit_meet = (
        "https://meet.livekit.io/custom?"
        f"liveKitUrl={args.ws_url}&"
        f"token={user_token}"
    )
    print(
        "\n=== Open this URL in your browser to join the bot ===\n"
        f"{livekit_meet}\n"
        "===\n",
        file=sys.stderr,
    )

    asyncio.run(run_bot(args.ws_url, args.room, args.api_key, args.api_secret))


if __name__ == "__main__":
    main()
