"""Pipecat TTSService backed by the patched qwen_megakernel.

Per-utterance flow inside ``run_tts``::

    text (from LLM)
       │
       ▼
    1. Build talker prompt embeddings using HF's own helpers
       (text_projection, codec_embedding, speaker_encoder, tts_pad/bos/eos):
         talker_input_embeds  : [B=1, T_prefill, 1024]
         trailing_text_hidden : [B=1, T_text, 1024]
         tts_pad_embed        : [1, 1, 1024]
       (this is exactly what Qwen3TTSForConditionalGeneration.generate does
        for the assistant turn — we replicate it.)
       │
       ▼
    2. Prefill: feed each row of talker_input_embeds into the megakernel
       as a single autoregressive step. Last codec token from prefill is
       the "kick-off" for autoregressive decoding.
       │
       ▼
    3. Autoregressive loop, one frame per 80 ms of audio:
         a. Talker megakernel step → codec_token_0 (codebook 0).
            ↳ also returns post-RMSNorm hidden state.
         b. Code Predictor (HF) consumes (past_talker_hidden + codec_token_0_emb)
            and generates 15 residual codec tokens (codebooks 1..15).
         c. Composite embed for NEXT step:
              sum over i of codebook_i_emb
              + trailing_text_hidden[step]  (or tts_pad_embed if past text)
         d. Speech tokenizer (HF) decodes the 16-codebook frame to 1920
            int16 samples @ 24 kHz; push as TTSAudioRawFrame.
       Stop when codec_token_0 == CODEC_EOS_ID or max_steps reached.

All GPU work happens in a worker thread. The asyncio side just pulls from
an asyncio.Queue and yields frames.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

from patches.talker_constants import (
    CODEBOOKS_PER_FRAME,
    CODEC_BOS_ID,
    CODEC_LANG,
    CODEC_NOTHINK_ID,
    CODEC_PAD_ID,
    CODEC_THINK_BOS_ID,
    CODEC_THINK_EOS_ID,
    CODEC_THINK_ID,
    HIDDEN_SIZE,
    SAMPLE_RATE,
    SAMPLES_PER_TALKER_STEP,
    TALKER_HZ,
    TTS_BOS_TOKEN_ID,
    TTS_EOS_TOKEN_ID,
    TTS_PAD_TOKEN_ID,
)
from patches.talker_model import TalkerDecoder, load_talker_weights


@dataclass
class MegakernelQwen3TTSSettings(TTSSettings):
    """Settings for the megakernel-backed Qwen3-TTS service."""


class MegakernelQwen3TTSService(TTSService):
    """Streaming Qwen3-TTS with a megakernel-accelerated Talker.

    Args:
        hf_model: A loaded ``Qwen3TTSForConditionalGeneration`` on cuda
            (use ``Qwen3TTSForConditionalGeneration.from_pretrained(...)``,
            which loads the speech tokenizer too).
        tokenizer: The text tokenizer (loaded via ``AutoTokenizer.from_pretrained``).
        language: Language for codec-language tag (e.g. ``"english"``).
            Pass ``"auto"`` to skip the language tag (no-think mode).
        speaker: Optional named speaker (must be in the model's ``spk_id``
            table). For Base models this stays ``None`` and the model uses
            its default voice. Voice-cloning is intentionally out of scope.
        max_steps: Hard cap on talker steps per utterance (default 750 = 60 s).
        settings: Optional runtime settings.
        **kwargs: Forwarded to ``TTSService``.
    """

    Settings = MegakernelQwen3TTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        hf_model: Any,
        tokenizer: Any,
        language: str = "english",
        speaker: str | None = None,
        max_steps: int = 750,
        settings: Settings | None = None,
        **kwargs,
    ) -> None:
        default_settings = self.Settings(model=None, voice=speaker, language=None)
        if settings is not None:
            default_settings.apply_update(settings)
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            sample_rate=SAMPLE_RATE,
            settings=default_settings,
            **kwargs,
        )

        self._hf = hf_model
        self._tok = tokenizer
        self._language = language.lower()
        self._speaker = speaker
        self._max_steps = max_steps
        self._device = next(hf_model.parameters()).device
        self._dtype = next(hf_model.parameters()).dtype

        # talker / code-predictor / speech-tokenizer handles
        self._talker_hf = hf_model.talker
        self._code_predictor = hf_model.talker.code_predictor
        self._speech_tokenizer = hf_model.speech_tokenizer

        # torch.compile on Code Predictor's per-step model.
        # We tried mode="reduce-overhead" (CUDA Graphs) but HF's DynamicCache
        # mutates K/V tensors via torch.cat each step, which torch.compile
        # detects as an in-place input mutation and either (a) silently falls
        # back to no-cudagraphs or (b) gets stuck in a recompile loop trying
        # to capture each cache state. max-autotune-no-cudagraphs is the safe
        # sweet spot: kernel fusion + autotune wins without the graph capture
        # complications. Closing the remaining ~45 ms CP-loop gap would need
        # a custom static KV cache + manual torch.cuda.graph() capture (out
        # of scope here).
        try:
            self._code_predictor.model = torch.compile(
                self._code_predictor.model,
                mode="max-autotune-no-cudagraphs",
                dynamic=False,
                fullgraph=False,
            )
            logger.info("torch.compile(max-autotune-no-cudagraphs) applied to code_predictor.model")
        except Exception as e:
            logger.warning(f"torch.compile on code_predictor.model failed: {e}")

        # The vocoder (speech_tokenizer.model.decoder) is a causal ConvNet that
        # runs once per AR frame with stable single-frame input shape [1, 16, 1].
        # It has NO HF Cache (pure conv), so ``mode="reduce-overhead"`` is safe
        # here — CUDA Graphs cut the decode call from ~20 ms to ~1 ms.
        # The caller must wrap each invocation with cudagraph_mark_step_begin()
        # (handled in _decode_frame_to_audio).
        try:
            self._speech_tokenizer.model.decoder = torch.compile(
                self._speech_tokenizer.model.decoder,
                mode="reduce-overhead", dynamic=False, fullgraph=False,
            )
            logger.info("torch.compile(reduce-overhead) applied to speech_tokenizer.model.decoder")
        except Exception as e:
            logger.warning(f"torch.compile on speech_tokenizer.model.decoder failed: {e}")
        if self._speech_tokenizer is None:
            raise RuntimeError(
                "hf_model.speech_tokenizer is None — load the model via "
                "Qwen3TTSForConditionalGeneration.from_pretrained(...), not AutoModel."
            )

        # Build the megakernel-backed talker. ``compose_embed_fn`` is wired
        # to ``self._compose_embed`` which has access to per-utterance state
        # (trailing_text_hidden, prev_codes, etc.). Per-utterance state is
        # stashed on self._utt — see _run_synthesis_blocking.
        logger.debug("Loading talker weights into megakernel...")
        self._talker_weights = load_talker_weights(hf_model, verbose=True)
        self._talker = TalkerDecoder(
            self._talker_weights, compose_embed_fn=self._compose_embed,
        )
        self._utt: _UtteranceState | None = None  # set per-utterance

        # Serialize utterances — talker has a single KV cache.
        self._state_lock = threading.Lock()

        logger.info(
            f"{self}: ready. talker_hz={TALKER_HZ}, sr={SAMPLE_RATE}, "
            f"samples/step={SAMPLES_PER_TALKER_STEP}, codebooks/frame={CODEBOOKS_PER_FRAME}, "
            f"language={self._language}"
        )

    def can_generate_metrics(self) -> bool:
        return True

    # ---------------------------------------------------------------- prefill

    def _build_prefill(self, text: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the talker prefill embeddings for one utterance.

        Mirrors the relevant slice of ``Qwen3TTSForConditionalGeneration.generate``
        (the part that constructs ``talker_input_embeds`` etc.) — but only for
        the case we care about: ``non_streaming_mode=False``, ``voice_clone_prompt=None``.

        Returns:
            (talker_input_embeds, trailing_text_hidden, tts_pad_embed):
                talker_input_embeds  : [1, T_prefill, HIDDEN_SIZE] bf16
                trailing_text_hidden : [1, T_text_trail, HIDDEN_SIZE] bf16
                tts_pad_embed        : [1, 1, HIDDEN_SIZE] bf16
        """
        talker = self._talker_hf
        config = self._hf.config
        tcfg = config.talker_config
        device, dtype = self._device, self._dtype

        # ChatML prompt for the assistant turn. The HF code expects 'role tokens'
        # at positions [0:3] (i.e. "<|im_start|>assistant\n") and the actual
        # text starting at position 3. We mirror that exactly.
        # The assistant prompt prefix is: "<|im_start|>assistant\n"
        prompt_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = self._tok(prompt_text, return_tensors="pt").input_ids.to(device)

        # tts_bos / tts_eos / tts_pad embeddings (text-side, projected to 1024).
        tts_special = torch.tensor(
            [[TTS_BOS_TOKEN_ID, TTS_EOS_TOKEN_ID, TTS_PAD_TOKEN_ID]],
            device=device, dtype=input_ids.dtype,
        )
        tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(
            talker.get_text_embeddings()(tts_special)
        ).chunk(3, dim=1)

        # Codec language tag prefix.
        if self._language == "auto" or self._language not in CODEC_LANG:
            codec_prefill_ids = [[CODEC_NOTHINK_ID, CODEC_THINK_BOS_ID, CODEC_THINK_EOS_ID]]
        else:
            language_id = CODEC_LANG[self._language]
            codec_prefill_ids = [[
                CODEC_THINK_ID, CODEC_THINK_BOS_ID, language_id, CODEC_THINK_EOS_ID,
            ]]
        codec_input_embedding_0 = talker.get_input_embeddings()(
            torch.tensor(codec_prefill_ids, device=device, dtype=input_ids.dtype)
        )

        codec_input_embedding_1 = talker.get_input_embeddings()(
            torch.tensor([[CODEC_PAD_ID, CODEC_BOS_ID]], device=device, dtype=input_ids.dtype)
        )

        # Named speaker — for the Base model, this is None (use default voice).
        speaker_embed = None
        if self._speaker:
            if self._speaker.lower() not in tcfg.spk_id:
                raise ValueError(f"Speaker {self._speaker!r} not in tcfg.spk_id")
            spk_id = tcfg.spk_id[self._speaker.lower()]
            speaker_embed = talker.get_input_embeddings()(
                torch.tensor(spk_id, device=device, dtype=input_ids.dtype)
            ).view(1, 1, -1)

        if speaker_embed is None:
            codec_input_embedding = torch.cat(
                [codec_input_embedding_0, codec_input_embedding_1], dim=1,
            )
        else:
            codec_input_embedding = torch.cat(
                [codec_input_embedding_0, speaker_embed, codec_input_embedding_1], dim=1,
            )

        # role tokens (positions 0..2 = "<|im_start|>assistant\n")
        role_embed = talker.text_projection(talker.get_text_embeddings()(input_ids[:, :3]))

        # tts_pad * (k-1) + tts_bos, summed with codec_input_embedding[:-1]
        pad_padding = tts_pad_embed.expand(-1, codec_input_embedding.shape[1] - 2, -1)
        intermediate = torch.cat((pad_padding, tts_bos_embed), dim=1) + codec_input_embedding[:, :-1]

        talker_input_embed = torch.cat((role_embed, intermediate), dim=1)

        # tts_text_first_token + codec_input_embedding[-1:]
        first_text_embed = talker.text_projection(
            talker.get_text_embeddings()(input_ids[:, 3:4])
        )
        talker_input_embed = torch.cat(
            [talker_input_embed, first_text_embed + codec_input_embedding[:, -1:]], dim=1,
        )

        # trailing_text_hidden: text from position 4 to -5, plus tts_eos_embed.
        # Position layout: ids[0..2] role, ids[3] first content tok, ids[4..-5] middle,
        # ids[-5..] terminator. (See HF code for the slicing.)
        if input_ids.shape[1] > 9:
            trailing_text_hidden = torch.cat(
                (
                    talker.text_projection(talker.get_text_embeddings()(input_ids[:, 4:-5])),
                    tts_eos_embed,
                ),
                dim=1,
            )
        else:
            # Very short prompt — no middle text, just eos.
            trailing_text_hidden = tts_eos_embed

        return (
            talker_input_embed.to(torch.bfloat16),
            trailing_text_hidden.to(torch.bfloat16),
            tts_pad_embed.to(torch.bfloat16),
        )

    # -------------------------------------------------------- code-predictor

    def _code_predictor_decode_15(self, cp_inputs: torch.Tensor) -> torch.Tensor:
        """Hand-rolled greedy 15-step decode of the Code Predictor.

        Replaces ``self._code_predictor.generate(do_sample=False, max_new_tokens=15)``.
        Saves the HF GenerationMixin per-step Python overhead; the inner
        ``cp.model`` is torch.compile'd (max-autotune-no-cudagraphs) so the
        per-step kernel cost is autotuned.

        Args:
            cp_inputs: ``[1, 2, H]`` bf16 — ``cat(past_talker_hidden, last_id_hidden)``.

        Returns:
            ``[1, 15]`` int64 — the residual codebook ids for codebooks 1..15
            (codebook 0 is the talker's argmax-from-megakernel; passed in via
            ``last_id_hidden``).
        """
        cp = self._code_predictor
        # --- prefill on the 2-row input ---
        proj = cp.small_to_mtp_projection(cp_inputs)
        out = cp.model(
            inputs_embeds=proj,
            use_cache=True,
            return_dict=True,
        )
        logits = cp.lm_head[0](out.last_hidden_state[:, -1:, :])  # [1, 1, V]
        next_id = logits.argmax(dim=-1)  # [1, 1]
        past_kv = out.past_key_values

        sub_codes = [next_id]
        # --- 14 more greedy steps, generation_steps = 1..14 ---
        for gen_step in range(1, CODEBOOKS_PER_FRAME - 1):
            emb_in = cp.model.get_input_embeddings()[gen_step - 1](next_id)
            proj = cp.small_to_mtp_projection(emb_in)
            out = cp.model(
                inputs_embeds=proj,
                past_key_values=past_kv,
                use_cache=True,
                return_dict=True,
            )
            logits = cp.lm_head[gen_step](out.last_hidden_state[:, -1:, :])
            next_id = logits.argmax(dim=-1)
            past_kv = out.past_key_values
            sub_codes.append(next_id)
        return torch.cat(sub_codes, dim=-1)  # [1, 15]

    # -------------------------------------------------------- composite embed

    def _compose_embed(
        self,
        *,
        frame_idx: int,
        position: int,
        prev_codec_token: int | None,
    ) -> torch.Tensor:
        """Build the composite hidden state for the NEXT talker step.

        Called by :class:`TalkerDecoder` between successive ``step()`` calls.
        Replicates ``Qwen3TTSTalkerForConditionalGeneration.forward`` lines
        1668-1692 — the generation-step branch.

        Returns ``[HIDDEN_SIZE]`` bf16 cuda tensor.
        """
        u = self._utt
        assert u is not None, "_compose_embed called outside an utterance"

        # We need a previous codec token to run the Code Predictor. On the
        # very first autoregressive step (right after prefill), the kernel
        # emitted prev_codec_token in step `position-1` (the last prefill row).
        assert prev_codec_token is not None
        prev_codec_t = torch.tensor([[prev_codec_token]], device=self._device, dtype=torch.long)

        with torch.inference_mode():
            # 1) codec_embedding for talker's previous codec token (group 0).
            last_id_hidden = self._talker_hf.get_input_embeddings()(prev_codec_t)  # [1, 1, H]

            # 2) Code Predictor: 15 residual codebook tokens.
            #
            # The HF forward concatenates past_talker_hidden with last_id_hidden:
            #   inputs_embeds = torch.cat((past_hidden, last_id_hidden), dim=1)
            # past_hidden is the talker's post-RMSNorm hidden from the
            # PREVIOUS step. We stashed it on the utterance state.
            past_hidden = u.past_hidden  # [1, 1, H]
            cp_inputs = torch.cat((past_hidden, last_id_hidden), dim=1)
            sub_codes = self._code_predictor_decode_15(cp_inputs)  # [1, 15] residual codebook ids

            # Stash the per-frame [1, 16] codes for the speech tokenizer.
            u.frame_codes.append(
                torch.cat(
                    (torch.tensor([[prev_codec_token]], device=self._device, dtype=torch.long),
                     sub_codes), dim=-1,
                )
            )

            # 3) Compose embedding:
            #      sum over i of codebook_i_emb + (trailing_text or tts_pad).
            codec_hiddens = [last_id_hidden]
            for i in range(CODEBOOKS_PER_FRAME - 1):
                emb_i = self._code_predictor.get_input_embeddings()[i](sub_codes[..., i:i+1])
                codec_hiddens.append(emb_i)
            codec_hiddens_cat = torch.cat(codec_hiddens, dim=1)
            inputs_embeds = codec_hiddens_cat.sum(1, keepdim=True)  # [1, 1, H]

            # frame_idx is 0-indexed for the autoregressive part. trailing_text_hidden
            # has length T_text — index frame_idx if in range, else use pad.
            if frame_idx < u.trailing_text_hidden.shape[1]:
                inputs_embeds = inputs_embeds + u.trailing_text_hidden[:, frame_idx].unsqueeze(1)
            else:
                inputs_embeds = inputs_embeds + u.tts_pad_embed

        # [1, 1, H] -> [H]
        return inputs_embeds.view(HIDDEN_SIZE).to(torch.bfloat16).contiguous()

    # ---------------------------------------------------------- synthesis

    def _run_synthesis_blocking(
        self,
        text: str,
        out_queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
        cancel: threading.Event,
    ) -> None:
        try:
            with self._state_lock, torch.inference_mode():
                self._talker.reset()

                # Prefill embeddings.
                talker_in, trail, pad_emb = self._build_prefill(text)
                t_prefill = talker_in.shape[1]
                logger.debug(f"{self}: prefill rows={t_prefill}, text_trail={trail.shape[1]}")

                # Per-utterance state for _compose_embed.
                self._utt = _UtteranceState(
                    trailing_text_hidden=trail,
                    tts_pad_embed=pad_emb,
                    past_hidden=torch.zeros(1, 1, HIDDEN_SIZE, dtype=torch.bfloat16, device=self._device),
                )

                t0 = time.perf_counter()
                first_byte_t: float | None = None
                total_samples = 0

                # ---- PREFILL ---------------------------------------------------
                # Feed each prefill row to the kernel. The last row's output
                # codec_token is the kick-off for autoregressive decoding.
                last_codec_token = None
                last_hidden_bf16: torch.Tensor | None = None
                for j in range(t_prefill):
                    if cancel.is_set():
                        return
                    row = talker_in[0, j].contiguous()
                    last_codec_token = self._talker.step_with_hidden(row)
                    last_hidden_bf16 = self._talker._norm_out.to(torch.bfloat16)
                # past_hidden for the first AR _compose call.
                assert last_hidden_bf16 is not None
                self._utt.past_hidden = last_hidden_bf16.view(1, 1, HIDDEN_SIZE)

                # The kernel's last-prefill output IS the first codec token (group 0).
                # We need to package the full 16-codebook frame for the speech tokenizer.
                # For the first frame, we run the Code Predictor manually right here:
                cp_inputs = torch.cat(
                    (
                        self._utt.past_hidden,
                        self._talker_hf.get_input_embeddings()(
                            torch.tensor([[last_codec_token]], device=self._device, dtype=torch.long)
                        ),
                    ),
                    dim=1,
                )
                sub_codes_first = self._code_predictor_decode_15(cp_inputs)  # [1, 15]
                first_frame_codes = torch.cat(
                    (
                        torch.tensor([[last_codec_token]], device=self._device, dtype=torch.long),
                        sub_codes_first,
                    ),
                    dim=-1,
                )  # [1, 16]
                # Emit audio for this frame.
                audio_bytes = self._decode_frame_to_audio(first_frame_codes)
                if first_byte_t is None:
                    first_byte_t = time.perf_counter()
                    logger.debug(f"{self}: first chunk after {(first_byte_t - t0)*1000:.1f}ms")
                asyncio.run_coroutine_threadsafe(out_queue.put(audio_bytes), loop)
                total_samples += SAMPLES_PER_TALKER_STEP

                # ---- AUTOREGRESSIVE LOOP -------------------------------------
                # Frame counter for trailing_text_hidden indexing. Starts at 0
                # because the first frame above used the prefill tail; we now
                # consume one trailing-text position per AR step.
                frame_idx = 0
                _token_trace: list[int] = [last_codec_token]
                # Silent-frame EOS detection: the talker doesn't always emit
                # CODEC_EOS_ID=2150 cleanly when speech ends — it sometimes
                # continues with tokens that decode to silence. Stop after
                # `_max_silent_frames` consecutive silent frames (RMS < threshold).
                _silent_run = 0
                _max_silent_frames = 4   # 4 * 80ms = 320ms of trailing silence
                _silence_rms_threshold = 0.005  # int16-normalized
                while frame_idx < self._max_steps:
                    if cancel.is_set():
                        break
                    step = self._talker.step(frame_idx=frame_idx)
                    _token_trace.append(step.codec_token)
                    if frame_idx < 8:
                        logger.debug(
                            f"{self}: frame_idx={frame_idx} codec_token={step.codec_token}"
                        )
                    if TalkerDecoder.is_eos(step.codec_token):
                        logger.info(
                            f"{self}: codec EOS at frame {frame_idx}. "
                            f"Token trace: {_token_trace[:20]}"
                        )
                        break
                    # Refresh past_hidden for the next compose call.
                    self._utt.past_hidden = step.hidden.view(1, 1, HIDDEN_SIZE)

                    # Pull the 16-codebook frame compose_embed just produced.
                    frame_codes = self._utt.frame_codes[-1]
                    audio_bytes = self._decode_frame_to_audio(frame_codes)

                    # Silent-frame counter — quick RMS check on this frame.
                    audio_arr = np.frombuffer(audio_bytes, dtype=np.int16)
                    frame_rms = float(np.sqrt(
                        (audio_arr.astype(np.float32) / 32767.0) ** 2 + 1e-12
                    ).mean())
                    if frame_rms < _silence_rms_threshold:
                        _silent_run += 1
                    else:
                        _silent_run = 0

                    asyncio.run_coroutine_threadsafe(out_queue.put(audio_bytes), loop)
                    total_samples += SAMPLES_PER_TALKER_STEP
                    frame_idx += 1

                    if _silent_run >= _max_silent_frames:
                        logger.info(
                            f"{self}: silent-frame EOS at frame {frame_idx} "
                            f"(after {_silent_run} consecutive silent frames)."
                        )
                        break

                wall = time.perf_counter() - t0
                audio_sec = total_samples / SAMPLE_RATE if total_samples else 1e-9
                logger.info(
                    f"{self}: utterance complete. "
                    f"AR_steps={frame_idx} audio_s={audio_sec:.2f} "
                    f"wall_s={wall:.2f} rtf={wall/audio_sec:.3f}"
                )

        except Exception as e:
            logger.exception(f"{self}: synthesis worker failed: {e}")
            asyncio.run_coroutine_threadsafe(out_queue.put(("error", str(e))), loop)
        finally:
            asyncio.run_coroutine_threadsafe(out_queue.put(None), loop)
            self._utt = None

    def _decode_frame_to_audio(self, frame_codes: torch.Tensor) -> bytes:
        """Run the speech_tokenizer decoder for one talker frame.

        ``frame_codes`` shape: ``[1, 16]`` long (T=1, Q=16).

        We bypass ``speech_tokenizer.decode([{audio_codes: ...}])`` because that
        wrapper goes through ``decoder.chunked_decode(...)`` which internally
        calls ``self(codes_chunk)`` — and ``self`` there is the ORIGINAL decoder
        instance, so the compiled wrapper we installed at ``decoder`` never gets
        hit. Calling ``self._speech_tokenizer.model.decoder(...)`` directly
        respects the compile and saves ~20 ms per call (CUDA-Graph-captured
        path: ~1 ms vs ~20 ms eager).
        """
        # codes shape that the decoder.forward expects: (B, Q, T) = (1, 16, 1)
        codes_btq = frame_codes.view(1, 1, CODEBOOKS_PER_FRAME)
        codes_bqt = codes_btq.transpose(1, 2)  # (1, 16, 1)

        # Mark CUDA-Graph step boundary so previous frame's captured outputs
        # are recycled cleanly before the next replay.
        torch.compiler.cudagraph_mark_step_begin()
        out = self._speech_tokenizer.model.decoder(codes_bqt)  # (1, 1, samples)
        wav = out.squeeze().to(torch.float32).detach().cpu().numpy()
        wav = np.clip(wav, -1.0, 1.0)
        audio_int16 = (wav * 32767.0).astype(np.int16)
        return audio_int16.tobytes()

    # --------------------------------------------------------------- run_tts

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: TTS [{text}]")

        loop = asyncio.get_running_loop()
        out_queue: asyncio.Queue = asyncio.Queue()
        cancel = threading.Event()

        # In a real Pipecat Pipeline, TaskManager wires up cancellation; fall
        # back to plain asyncio when running outside a pipeline (e.g. the bench).
        worker_coro = asyncio.to_thread(
            self._run_synthesis_blocking, text, out_queue, loop, cancel
        )
        try:
            worker_done = self.create_task(
                worker_coro, name=f"megakernel-tts-worker-{context_id}"
            )
        except Exception:
            worker_done = asyncio.create_task(
                worker_coro, name=f"megakernel-tts-worker-{context_id}"
            )

        async def byte_iterator() -> AsyncIterator[bytes]:
            while True:
                item = await out_queue.get()
                if item is None:
                    return
                if isinstance(item, tuple) and item and item[0] == "error":
                    raise RuntimeError(item[1])
                yield item

        try:
            await self.start_tts_usage_metrics(text)
            async for frame in self._stream_audio_frames_from_iterator(
                byte_iterator(),
                in_sample_rate=SAMPLE_RATE,
                context_id=context_id,
            ):
                await self.stop_ttfb_metrics()
                yield frame
        except Exception as e:
            logger.error(f"{self}: {e}")
            yield ErrorFrame(error=f"Megakernel TTS error: {e}")
        finally:
            cancel.set()
            try:
                await worker_done
            except Exception:
                pass
            await self.stop_ttfb_metrics()
            logger.debug(f"{self}: finished TTS [{text}]")


# ---------------------------------------------------------------------------
# Per-utterance state — separate dataclass so we don't accidentally leak it
# across utterances. _compose_embed reads/writes via self._utt.
# ---------------------------------------------------------------------------


@dataclass
class _UtteranceState:
    trailing_text_hidden: torch.Tensor  # [1, T_text, HIDDEN_SIZE] bf16
    tts_pad_embed: torch.Tensor          # [1, 1, HIDDEN_SIZE] bf16
    past_hidden: torch.Tensor            # [1, 1, HIDDEN_SIZE] bf16 — talker hidden from prev step

    # Each AR step's full 16-codebook frame, accumulated by _compose_embed
    # and consumed by _decode_frame_to_audio. We only ever read the last entry.
    frame_codes: list[torch.Tensor] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.frame_codes is None:
            self.frame_codes = []
