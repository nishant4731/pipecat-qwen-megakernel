"""Push-to-talk server: receives raw PCM audio on stdin, streams synthesis PCM on stdout.

Runs on the 5090 box. The client (push_to_talk_client.py on your laptop) records
audio from the local mic, scp's nothing — it pipes raw audio bytes over an SSH
ExecChannel directly into this process's stdin. We run the FULL pipeline on those
bytes:

    raw 16 kHz PCM stdin
       → faster-whisper (transcribe)
       → Qwen3-1.7B (generate response text, streamed)
       → MegakernelQwen3TTSService (synthesize, streamed)
       → raw 24 kHz int16 PCM stdout

The client plays stdout via ``ffplay`` in real time. No WebRTC, no NAT, no TURN —
just a pure TCP-over-SSH pipe that always works.

Framing on stdin/stdout::

    \\x01\\x00\\x00\\x00       MAGIC (4 bytes)
    <4-byte little-endian len> N
    <N bytes payload>          either int16 PCM or a JSON header/control msg
"""

from __future__ import annotations

import asyncio
import json
import os
import struct
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
from loguru import logger


# Make loguru go to stderr only — stdout is the audio stream.
logger.remove()
logger.add(sys.stderr, level="INFO")


MAGIC_AUDIO = 0x01  # payload: int16 PCM
MAGIC_TEXT = 0x02   # payload: utf-8 JSON {"kind": "stt"|"llm"|"info", "text": "..."}


def _send_frame(out, magic: int, payload: bytes) -> None:
    out.write(struct.pack("<BI", magic, len(payload)))
    out.write(payload)
    out.flush()


def _read_frame(inp):
    hdr = inp.read(5)
    if len(hdr) < 5:
        return None, None
    magic, n = struct.unpack("<BI", hdr)
    body = inp.read(n)
    return magic, body


async def main() -> None:
    text_in = os.environ.get("PUSH_TO_TALK_TEXT", "")

    sample_rate_in = int(os.environ.get("PTT_SAMPLE_RATE_IN", "16000"))
    logger.info(f"push-to-talk server starting; expecting {sample_rate_in} Hz int16 mono on stdin")

    # ---- load TTS ------------------------------------------------------------
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
    from transformers import AutoTokenizer
    logger.info("loading TTS model …")
    hf_tts = Qwen3TTSForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        dtype=torch.bfloat16, device_map="cuda",
    )
    hf_tts.eval()
    tts_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base", trust_remote_code=True)
    logger.info("TTS loaded")

    # ---- talker megakernel ---------------------------------------------------
    from pipecat_qwen_megakernel.services.qwen3_tts_megakernel import (
        MegakernelQwen3TTSService,
    )
    from patches.talker_constants import SAMPLE_RATE
    tts = MegakernelQwen3TTSService(hf_model=hf_tts, tokenizer=tts_tokenizer)
    tts._sample_rate = SAMPLE_RATE

    # ---- LLM (HF Qwen3-1.7B) ------------------------------------------------
    from transformers import AutoModelForCausalLM, TextIteratorStreamer
    logger.info("loading LLM …")
    llm_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    llm = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B", dtype=torch.bfloat16, device_map="cuda",
    )
    llm.eval()
    logger.info("LLM loaded")

    # ---- Whisper STT --------------------------------------------------------
    from faster_whisper import WhisperModel
    logger.info("loading Whisper …")
    # NOTE: ctranslate2 has no CUDA 13 wheel yet; the box runs CUDA 13 so we
    # run Whisper on CPU (int8). base.en CPU @ int8 is ~200-400 ms for short
    # utterances — fine for an interactive demo.
    stt = WhisperModel("base.en", device="cpu", compute_type="int8")
    logger.info("Whisper loaded (cpu, int8)")

    logger.info("READY — reading audio frames from stdin")
    _send_frame(sys.stdout.buffer, MAGIC_TEXT, json.dumps({"kind": "info", "text": "ready"}).encode())

    inp = sys.stdin.buffer
    out = sys.stdout.buffer

    while True:
        # Drain one user utterance (raw 16 kHz int16 PCM) until we get an empty
        # frame (which signals end-of-utterance).
        chunks: list[bytes] = []
        while True:
            magic, body = _read_frame(inp)
            if magic is None:
                logger.info("stdin closed; exiting")
                return
            if magic != MAGIC_AUDIO:
                continue
            if len(body) == 0:
                break  # end of utterance
            chunks.append(body)

        if not chunks:
            continue

        t0 = time.perf_counter()
        pcm = b"".join(chunks)
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0

        # 1) STT
        t_stt0 = time.perf_counter()
        segments, _ = stt.transcribe(audio, beam_size=1, language="en")
        user_text = " ".join(s.text.strip() for s in segments).strip()
        t_stt = time.perf_counter() - t_stt0
        logger.info(f"STT [{t_stt*1000:.0f} ms]: {user_text!r}")
        _send_frame(out, MAGIC_TEXT, json.dumps({"kind": "stt", "text": user_text}).encode())

        if not user_text:
            continue

        # 2) LLM
        t_llm0 = time.perf_counter()
        prompt = llm_tok.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful voice assistant. Be brief — respond in 1-2 short sentences. Do not use markdown or emoji."},
                {"role": "user", "content": user_text},
            ],
            tokenize=False, add_generation_prompt=True,
        )
        ids = llm_tok(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out_ids = llm.generate(
                **ids, max_new_tokens=80, do_sample=True, top_p=0.9, temperature=0.7,
                pad_token_id=llm_tok.eos_token_id,
            )
        reply_ids = out_ids[0, ids.input_ids.shape[1]:]
        reply_text = llm_tok.decode(reply_ids, skip_special_tokens=True).strip()
        t_llm = time.perf_counter() - t_llm0
        logger.info(f"LLM [{t_llm*1000:.0f} ms]: {reply_text!r}")
        _send_frame(out, MAGIC_TEXT, json.dumps({"kind": "llm", "text": reply_text}).encode())

        # 3) TTS → stream audio frames
        t_tts0 = time.perf_counter()
        first_chunk_t = None
        total_audio_bytes = 0
        from pipecat.frames.frames import TTSAudioRawFrame

        async for frame in tts.run_tts(reply_text, context_id="ptt"):
            if isinstance(frame, TTSAudioRawFrame):
                if first_chunk_t is None:
                    first_chunk_t = time.perf_counter()
                    ttfc = (first_chunk_t - t_tts0) * 1000.0
                    logger.info(f"TTS first chunk at {ttfc:.0f} ms")
                _send_frame(out, MAGIC_AUDIO, frame.audio)
                total_audio_bytes += len(frame.audio)

        # end-of-utterance marker
        _send_frame(out, MAGIC_AUDIO, b"")

        wall_total = time.perf_counter() - t0
        audio_secs = total_audio_bytes / 2 / SAMPLE_RATE if total_audio_bytes else 0
        rtf = (time.perf_counter() - t_tts0) / max(audio_secs, 1e-9)
        logger.info(
            f"turn complete: stt={t_stt*1000:.0f}ms llm={t_llm*1000:.0f}ms "
            f"tts_audio={audio_secs:.2f}s rtf={rtf:.3f} total={wall_total*1000:.0f}ms"
        )
        _send_frame(out, MAGIC_TEXT, json.dumps({
            "kind": "info",
            "text": f"turn: STT {int(t_stt*1000)}ms · LLM {int(t_llm*1000)}ms · "
                    f"TTS {audio_secs:.2f}s @ RTF {rtf:.3f} · total {int(wall_total*1000)}ms"
        }).encode())


if __name__ == "__main__":
    asyncio.run(main())
