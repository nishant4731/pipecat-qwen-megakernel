"""A thin local LLM service for Pipecat that wraps an HF Qwen3-Instruct model.

This sidesteps having to run a separate OpenAI-compatible server (vLLM /
TGI / Ollama) just to talk to an in-process model. The trade-off: we don't get
the heavy batching/scheduling of vLLM — but for a single-user voice demo with
a 1.7B model, in-process HF is fine.

Pattern follows pipecat's ``LLMService`` base; emits streaming text frames
that the TTS service consumes.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

import torch
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings


@dataclass
class Qwen3LocalLLMSettings:
    """Configuration for the local Qwen3 LLM service."""

    model_name: str = "Qwen/Qwen3-1.7B"
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    system_instruction: str = (
        "You are a helpful assistant in a voice conversation. "
        "Your responses will be spoken aloud, so avoid emojis, "
        "bullet points, or other formatting that can't be spoken. "
        "Respond in a creative, helpful, and brief way."
    )


class Qwen3LocalLLMService(LLMService):
    """Streaming local Qwen3-Instruct LLM service.

    Runs the model in a background thread; streams generated tokens as
    ``LLMTextFrame`` (which is the same frame the OpenAI streaming service
    emits — so the downstream TTS can consume either interchangeably).
    """

    def __init__(
        self,
        *,
        settings: Qwen3LocalLLMSettings | None = None,
        **kwargs,
    ) -> None:
        self._cfg = settings or Qwen3LocalLLMSettings()
        # Pipecat's LLMService validates that ALL LLMSettings fields are
        # initialized (no NOT_GIVEN sentinels). Build the full settings object
        # explicitly with None for fields we don't expose locally.
        super().__init__(
            settings=LLMSettings(
                model=self._cfg.model_name,
                extra=None,
                system_instruction=self._cfg.system_instruction,
                temperature=self._cfg.temperature,
                max_tokens=self._cfg.max_new_tokens,
                top_p=self._cfg.top_p,
                top_k=None,
                frequency_penalty=None,
                presence_penalty=None,
                seed=None,
                filter_incomplete_user_turns=None,
                user_turn_completion_config=None,
            ),
            **kwargs,
        )

        logger.info(f"Loading local Qwen3 LLM: {self._cfg.model_name}")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self._cfg.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._cfg.model_name,
            dtype=torch.bfloat16,
            device_map="cuda",
        )
        self._model.eval()

    # ----------------------------------------------------------- frame handling

    def can_generate_metrics(self) -> bool:
        return True

    async def _process_context(self, context: LLMContext) -> None:
        await self.push_frame(LLMFullResponseStartFrame())
        await self.start_ttfb_metrics()

        try:
            async for chunk in self._stream_completion(context):
                await self.stop_ttfb_metrics()
                if chunk:
                    await self.push_frame(LLMTextFrame(chunk))
        except Exception as e:
            logger.exception(f"{self}: LLM generation failed: {e}")
            await self.push_error(f"Qwen3 local LLM error: {e}")
        finally:
            await self.push_frame(LLMFullResponseEndFrame())

    # ----------------------------------------------------------- generation

    def _format_chat(self, context: LLMContext) -> str:
        """Render the LLMContext into a Qwen3 ChatML prompt string."""
        # LLMContext exposes ``messages`` as a list of dicts {role, content}.
        msgs: list[dict[str, str]] = list(context.messages)
        # Inject our system instruction if the user hasn't already.
        if not msgs or msgs[0].get("role") != "system":
            msgs.insert(0, {"role": "system", "content": self._cfg.system_instruction})
        return self._tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    async def _stream_completion(self, context: LLMContext) -> AsyncGenerator[str, None]:
        """Stream Qwen3 generation as text chunks.

        Uses ``transformers.TextIteratorStreamer`` from a worker thread.
        """
        from transformers import TextIteratorStreamer

        prompt_str = self._format_chat(context)
        inputs = self._tok(prompt_str, return_tensors="pt").to("cuda")

        streamer = TextIteratorStreamer(
            self._tok, skip_prompt=True, skip_special_tokens=True
        )

        def _generate() -> None:
            with torch.inference_mode():
                self._model.generate(
                    **inputs,
                    max_new_tokens=self._cfg.max_new_tokens,
                    do_sample=True,
                    temperature=self._cfg.temperature,
                    top_p=self._cfg.top_p,
                    repetition_penalty=self._cfg.repetition_penalty,
                    streamer=streamer,
                    pad_token_id=self._tok.eos_token_id,
                )

        gen_thread = threading.Thread(target=_generate, daemon=True)
        gen_thread.start()

        loop = asyncio.get_running_loop()

        def _next_chunk() -> str | None:
            try:
                return next(streamer)
            except StopIteration:
                return None

        while True:
            chunk = await loop.run_in_executor(None, _next_chunk)
            if chunk is None:
                return
            yield chunk

    # ----------------------------------------------------------- frame router

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Same pattern as ``BaseOpenAILLMService.process_frame`` — react to
        ``LLMContextFrame`` by invoking the model."""
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            try:
                await self.start_processing_metrics()
                await self._process_context(frame.context)
            except Exception as e:
                await self.push_error(error_msg=f"Qwen3 local LLM error: {e}", exception=e)
            finally:
                await self.stop_processing_metrics()
        else:
            await self.push_frame(frame, direction)
