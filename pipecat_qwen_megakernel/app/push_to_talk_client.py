"""Push-to-talk client: runs on your laptop, talks to the box over SSH.

Usage::

    python3 -m pipecat_qwen_megakernel.app.push_to_talk_client --ssh vast5090

Mic capture + speaker playback use the ``sounddevice`` Python package (pure pip,
no brew/ffmpeg required). The script SSH-execs ``push_to_talk_server`` on the
5090 box; stdin/stdout is a framed audio + JSON pipe.

Press ENTER to start talking, ENTER again to stop. Server runs STT → LLM → TTS
and streams 24 kHz audio back, which we play in real time.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import struct
import subprocess
import sys
import threading
import time

import numpy as np
import sounddevice as sd


MAGIC_AUDIO = 0x01
MAGIC_TEXT = 0x02
SAMPLE_RATE_IN = 16000  # we send 16 kHz to Whisper
SAMPLE_RATE_OUT = 24000  # box sends 24 kHz back
FRAME_SAMPLES_IN = 1600  # 100 ms @ 16 kHz


def _send_frame(stream, magic: int, payload: bytes) -> None:
    stream.write(struct.pack("<BI", magic, len(payload)))
    stream.write(payload)
    stream.flush()


def _read_frame(stream):
    hdr = stream.read(5)
    if len(hdr) < 5:
        return None, None
    magic, n = struct.unpack("<BI", hdr)
    body = b""
    while len(body) < n:
        chunk = stream.read(n - len(body))
        if not chunk:
            return None, None
        body += chunk
    return magic, body


def _record_loop(out_pipe, stop_event: threading.Event) -> None:
    """Stream mic audio (16 kHz int16 mono) into out_pipe until stop_event is set."""
    def callback(indata, frames, time_info, status):
        if stop_event.is_set():
            raise sd.CallbackStop()
        try:
            _send_frame(out_pipe, MAGIC_AUDIO, bytes(indata))
        except (BrokenPipeError, OSError):
            raise sd.CallbackStop()

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE_IN,
        channels=1,
        dtype="int16",
        blocksize=FRAME_SAMPLES_IN,
        callback=callback,
    ):
        while not stop_event.is_set():
            time.sleep(0.05)


def _player_loop(audio_q: "queue.Queue[bytes]", stop_event: threading.Event) -> None:
    """Pull 24 kHz int16 PCM chunks from audio_q and play them via sounddevice."""
    stream = sd.RawOutputStream(
        samplerate=SAMPLE_RATE_OUT,
        channels=1,
        dtype="int16",
        blocksize=0,
    )
    stream.start()
    try:
        while not stop_event.is_set():
            try:
                chunk = audio_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if chunk is None:
                continue
            stream.write(chunk)
    finally:
        stream.stop()
        stream.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssh", default="vast5090", help="SSH host alias from ~/.ssh/config")
    ap.add_argument(
        "--server-cmd",
        default=(
            "export HF_HOME=/dev/shm/hfcache HUGGINGFACE_HUB_CACHE=/dev/shm/hfcache "
            "LDG_VOCAB_SIZE=3072 PYTHONUNBUFFERED=1 ; "
            "cd /workspace/task && "
            "python -m pipecat_qwen_megakernel.app.push_to_talk_server"
        ),
        help="Command to run on the box (must reach push_to_talk_server)",
    )
    args = ap.parse_args()

    print(f"connecting to {args.ssh} ...", file=sys.stderr)
    ssh = subprocess.Popen(
        ["ssh", args.ssh, args.server_cmd],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
        bufsize=0,
    )

    audio_q: "queue.Queue[bytes]" = queue.Queue()
    stop_player = threading.Event()
    threading.Thread(
        target=_player_loop, args=(audio_q, stop_player), daemon=True,
    ).start()

    def reader_loop():
        while True:
            magic, body = _read_frame(ssh.stdout)
            if magic is None:
                print("\n[server disconnected]", file=sys.stderr)
                stop_player.set()
                return
            if magic == MAGIC_TEXT:
                try:
                    msg = json.loads(body.decode())
                    kind = msg.get("kind", "?")
                    text = msg.get("text", "")
                    print(f"\n[{kind.upper():4}] {text}", file=sys.stderr)
                except Exception:
                    print(f"\n[??] {body[:120]}", file=sys.stderr)
            elif magic == MAGIC_AUDIO:
                if body:
                    audio_q.put(body)

    threading.Thread(target=reader_loop, daemon=True).start()

    print("\nwaiting for server to finish model loading…", file=sys.stderr)
    time.sleep(2)
    print("\n=== Push-to-talk ready ===", file=sys.stderr)
    print("Press ENTER to start talking, ENTER again to stop.", file=sys.stderr)
    print("Ctrl-C to quit.\n", file=sys.stderr)

    while True:
        try:
            input("[press ENTER to start]")
        except (EOFError, KeyboardInterrupt):
            break
        print("  recording… (press ENTER again to stop)", file=sys.stderr)
        stop_rec = threading.Event()
        rec_thread = threading.Thread(
            target=_record_loop, args=(ssh.stdin, stop_rec), daemon=True,
        )
        rec_thread.start()
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            stop_rec.set()
            break
        stop_rec.set()
        rec_thread.join(timeout=2)
        try:
            _send_frame(ssh.stdin, MAGIC_AUDIO, b"")  # end-of-utterance marker
        except (BrokenPipeError, OSError):
            break
        print("  …processing", file=sys.stderr)

    ssh.terminate()
    stop_player.set()


if __name__ == "__main__":
    main()
