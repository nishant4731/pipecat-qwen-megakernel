#!/usr/bin/env bash
# Setup the rented Vast.ai RTX 5090 instance for this project.
#
# Run inside the instance, from the directory where this repo lives.
# Tested on `nvcr.io/nvidia/pytorch:26.01-py3` (NGC) which ships with
# PyTorch 2.10.0a0 + CUDA 13.1 + Python 3.12 — already compatible with sm_120.
# We do NOT install a torch nightly here; the NGC torch already works.
#
#   bash scripts/setup.sh
#
# Idempotent — re-running is safe.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---- HF cache lives on tmpfs (RAM-backed) because the container overlay is
# tight (16 GB). With 1 TB host RAM there's plenty of room. ---------------
HF_CACHE="${HF_CACHE:-/dev/shm/hfcache}"
mkdir -p "$HF_CACHE"
export HF_HOME="$HF_CACHE"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
echo "==> HF cache → $HF_CACHE (tmpfs)"
echo "    Set the same env in your shell before running benches/demo:"
echo "      export HF_HOME=$HF_CACHE"

echo "==> sanity: confirm we are on a Blackwell GPU"
if ! command -v nvidia-smi >/dev/null; then
  echo "ERROR: nvidia-smi not on PATH"; exit 1
fi
nvidia-smi -L
python -c "
import torch
cc = torch.cuda.get_device_capability()
print(f'torch={torch.__version__}  cuda={torch.version.cuda}  cc={cc}')
assert cc == (12, 0), f'expected sm_120 (Blackwell / RTX 5090), got {cc}'
print('OK — sm_120 detected')
"

echo "==> apt deps (audio + git already there in NGC; just ffmpeg + libsndfile for audio I/O)"
DEBIAN_FRONTEND=noninteractive apt-get update -y -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends -qq \
  ffmpeg libsndfile1 portaudio19-dev ninja-build

echo "==> clone qwen_megakernel and apply patches"
mkdir -p third_party
if [[ ! -d third_party/qwen_megakernel ]]; then
  git clone --depth=1 https://github.com/AlpinDale/qwen_megakernel third_party/qwen_megakernel
fi
python patches/apply_patches.py third_party/qwen_megakernel

echo "==> install patched megakernel (editable, --no-build-isolation to reuse NGC torch)"
pip install --no-cache-dir --no-build-isolation -e third_party/qwen_megakernel

echo "==> clone QwenLM/Qwen3-TTS (modeling code + speech_tokenizer)"
if [[ ! -d third_party/Qwen3-TTS ]]; then
  git clone --depth=1 https://github.com/QwenLM/Qwen3-TTS third_party/Qwen3-TTS
fi
pip install --no-cache-dir --no-build-isolation -e third_party/Qwen3-TTS

echo "==> reinstall torch + torchaudio for cu130 (NGC bundle's torchaudio breaks when qwen-tts pulls torchaudio 2.11)"
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu130 \
  torch==2.10.0 torchaudio==2.10.0

echo "==> install pipecat + STT/VAD/WebRTC extras"
# --ignore-installed cryptography sidesteps debian-managed cryptography (no RECORD file).
pip install --no-cache-dir --upgrade-strategy only-if-needed --ignore-installed cryptography \
  "pipecat-ai[silero,webrtc,whisper]>=0.0.104" \
  "transformers==4.57.3" "huggingface-hub>=0.34,<1.0" \
  faster-whisper>=1.0.3 accelerate safetensors \
  numpy soundfile loguru python-dotenv websockets aiortc

echo "==> install our package"
pip install --no-cache-dir --no-build-isolation -e .

echo "==> generate self-signed TLS cert for HTTPS (browsers require HTTPS for getUserMedia)"
if [[ ! -f /workspace/cert.pem ]]; then
  openssl req -x509 -newkey rsa:2048 -keyout /workspace/key.pem -out /workspace/cert.pem \
    -sha256 -days 365 -nodes -subj "/CN=megakernel-demo" >/dev/null 2>&1
  echo "   wrote /workspace/{cert,key}.pem"
fi

echo "==> sanity: AlpinDale's reference Qwen3-0.6B bench (Gate 1)"
echo "    NOTE: this downloads ~1.2 GB to $HF_CACHE."
python -m qwen_megakernel.bench

echo ""
echo "==> setup complete. Next:"
echo "  Gate 2:  python -m pipecat_qwen_megakernel.bench.bench_kernel"
echo "  Gate 3:  (correctness diff — manual)"
echo "  Gate 4:  python -m pipecat_qwen_megakernel.bench.bench_tts --runs 5"
echo "  Demo:    python -m pipecat_qwen_megakernel.app.demo --port 7860"
echo "           SSH-forward 7860 to your laptop:  ssh vast5090 -L 7860:localhost:7860"
echo "           then open http://localhost:7860/"
