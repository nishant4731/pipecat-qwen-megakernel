"""End-to-end Pipecat voice agent demo.

Behind the scenes:
- aiortc by default binds ICE host candidates to a random ephemeral UDP port,
  which doesn't survive Vast.ai's port mapping (only port 7860/udp is exposed).
  We monkey-patch aioice to bind to UDP 7860 specifically.
- We inject Google STUN servers into the ICE config so aiortc discovers the
  box's public IP via the NAT-mapped UDP port (Vast maps 7860/udp -> 43062/udp).
- We serve the signaling HTTPS over TLS so the browser will grant mic permission
  on a remote (non-localhost) origin. Self-signed cert is at /workspace/cert.pem.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# (1) aioice port-pin — patch aioice.ice.Connection.gather_candidates so the
#     UDP host candidate binds to container port 7860 (which Vast's docker
#     -p 7860:7860/udp maps to host port 43062, the pre-allocated port).
# ---------------------------------------------------------------------------
import os as _os

from loguru import logger as _logger

_PINNED_ICE_UDP_PORT = int(_os.environ.get("QWEN_DEMO_ICE_UDP_PORT", "7860"))


def _install_ice_port_pin() -> None:
    """Patch aioice's gather_host_candidates inner loop to bind to port 7860 once."""
    import aioice.ice as _ice
    # The actual binding happens inside Connection.connect() which iterates
    # addresses and calls loop.create_datagram_endpoint(local_addr=(addr, 0)).
    # We patch the running event loop's method *only when called via aioice*.
    # Simpler: patch the unbound method by re-defining a wrapper at the
    # Connection class level.

    if getattr(_ice.Connection, "_qwen_patched", False):
        return

    orig_connect = _ice.Connection.connect

    async def patched_connect(self, *args, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        orig = loop.create_datagram_endpoint
        used = {"first": False}

        async def cde(*a, **kw):
            local_addr = kw.get("local_addr")
            if (not used["first"] and local_addr
                    and local_addr[1] == 0
                    and local_addr[0] not in ("127.0.0.1", "::1")):
                pinned = (local_addr[0], _PINNED_ICE_UDP_PORT)
                _logger.info(f"[ICE port pin] trying {pinned}")
                kw["local_addr"] = pinned
                try:
                    r = await orig(*a, **kw)
                    used["first"] = True
                    _logger.info(f"[ICE port pin] bound to {pinned} OK")
                    return r
                except OSError as e:
                    _logger.warning(f"[ICE port pin] failed: {e}, falling back to ephemeral")
                    kw["local_addr"] = local_addr
            return await orig(*a, **kw)

        loop.create_datagram_endpoint = cde
        try:
            return await orig_connect(self, *args, **kwargs)
        finally:
            loop.create_datagram_endpoint = orig

    _ice.Connection.connect = patched_connect
    _ice.Connection._qwen_patched = True
    _logger.info("[ICE port pin] aioice.Connection.connect patched")


_install_ice_port_pin()


# ---------------------------------------------------------------------------
# (1b) aioice TURN nonce-refresh patch
# Cloudflare's TURN returns 401 on CHANNEL_BIND without a fresh NONCE attribute
# (the server expects the client to re-allocate on stale nonce). aioice 0.10's
# request_with_retry skips the retry path when NONCE is absent. Patch:
#   - On the first 401, log all response attribute keys.
#   - If NONCE+REALM are present: existing retry works (delegate to original).
#   - If only ERROR-CODE 401 is present: clear cached nonce, send the request
#     once with NO auth (which forces Cloudflare to send NONCE+REALM in the 401),
#     pick those up, and retry with fresh creds. This mirrors the long-term
#     STUN-auth dance from RFC 5389 §10.2.
# ---------------------------------------------------------------------------
def _install_aioice_turn_nonce_refresh() -> None:
    import aioice.turn as _turn
    import aioice.stun as _stun

    if getattr(_turn.TurnClientMixin, "_qwen_nonce_patched", False):
        return

    orig_request_with_retry = _turn.TurnClientMixin.request_with_retry

    async def patched(self, request):
        try:
            return await orig_request_with_retry(self, request)
        except _stun.TransactionFailed as e:
            attrs = e.response.attributes
            try:
                err = attrs["ERROR-CODE"][0]
            except Exception:
                err = None
            _logger.warning(
                f"[aioice turn] {request.message_method} failed err={err} "
                f"attrs={list(attrs.keys())}"
            )
            # Only attempt our fallback for 401 without NONCE/REALM
            if err == 401 and ("NONCE" not in attrs or "REALM" not in attrs):
                # Send same request with NO auth to coax a proper 401 with NONCE
                probe = _stun.Message(
                    message_method=request.message_method,
                    message_class=_stun.Class.REQUEST,
                )
                # copy non-auth attributes (e.g. XOR-PEER-ADDRESS, CHANNEL-NUMBER)
                for k, v in request.attributes.items():
                    if k in ("USERNAME", "NONCE", "REALM", "MESSAGE-INTEGRITY"):
                        continue
                    probe.attributes[k] = v
                # Save current auth state and clear it for the probe
                saved_nonce = getattr(self, "nonce", None)
                saved_realm = getattr(self, "realm", None)
                saved_key = getattr(self, "integrity_key", None)
                self.integrity_key = None
                try:
                    await self.request(probe)
                    # request() raises TransactionFailed for non-2xx, but
                    # request_with_retry inside it might catch.
                    _logger.warning("[aioice turn] probe unexpectedly succeeded")
                except _stun.TransactionFailed as pe:
                    pattrs = pe.response.attributes
                    if "NONCE" in pattrs and "REALM" in pattrs:
                        from aioice.turn import make_integrity_key
                        self.nonce = pattrs["NONCE"]
                        self.realm = pattrs["REALM"]
                        self.integrity_key = make_integrity_key(
                            self.username, self.realm, self.password
                        )
                        request.transaction_id = _stun.random_transaction_id()
                        # Strip stale auth attrs so __add_authentication re-adds fresh
                        for k in ("USERNAME", "NONCE", "REALM", "MESSAGE-INTEGRITY"):
                            request.attributes.pop(k, None)
                        _logger.info("[aioice turn] re-primed nonce, retrying")
                        return await self.request(request)
                    else:
                        _logger.warning(
                            f"[aioice turn] probe attrs={list(pattrs.keys())}"
                        )
                # Restore auth state for any future requests
                self.nonce = saved_nonce
                self.realm = saved_realm
                self.integrity_key = saved_key
            raise

    _turn.TurnClientMixin.request_with_retry = patched
    _turn.TurnClientMixin._qwen_nonce_patched = True
    _logger.info("[aioice turn] nonce-refresh patch installed")


_install_aioice_turn_nonce_refresh()


# ---------------------------------------------------------------------------
# (2) ICE-server monkey-patch — must happen BEFORE pipecat.runner.run is imported.
# Pipecat 1.1's runner constructs SmallWebRTCRequestHandler without passing
# ice_servers; we wrap its __init__ to splice in a default ICE config.
# ---------------------------------------------------------------------------
from aiortc import RTCIceServer as _RTCIceServer

from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCRequestHandler as _SmallWebRTCRequestHandler,
)

# Cloudflare TURN — fetched at startup. The aiortc side uses these directly;
# the browser side gets them via the /ice-servers endpoint we add below.
def _fetch_cloudflare_ice_servers() -> list[dict]:
    """Fetch ephemeral ICE credentials from Cloudflare TURN.

    Falls back to Google STUN if no Cloudflare key is configured.
    """
    key_id = _os.environ.get("QWEN_TURN_KEY_ID")
    api_tok = _os.environ.get("QWEN_TURN_API_TOKEN")
    if not (key_id and api_tok):
        return [{"urls": "stun:stun.l.google.com:19302"}]
    import urllib.request
    import urllib.error
    req = urllib.request.Request(
        f"https://rtc.live.cloudflare.com/v1/turn/keys/{key_id}/credentials/generate",
        data=b'{"ttl": 3600}',
        headers={
            "Authorization": f"Bearer {api_tok}",
            "Content-Type": "application/json",
            "User-Agent": "curl/8.5.0",
            "Accept": "*/*",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return [data["iceServers"]]
    except Exception as e:
        import sys
        print(f"[WARN] Cloudflare TURN fetch failed: {e}", file=sys.stderr)
        return [{"urls": "stun:stun.l.google.com:19302"}]


import json
_CF_ICE = _fetch_cloudflare_ice_servers()
print(f"[ICE servers] {json.dumps(_CF_ICE)[:240]}", flush=True)

# aiortc side: Cloudflare TURN UDP only (no TCP transport).
# aioice has a known nonce-caching bug with TURN-over-TCP (401 on CHANNEL-BIND).
# The UDP variant works fine because aiortc / aioice handle UDP-TURN much
# more cleanly than TCP-TURN. The box CAN reach Cloudflare TURN over UDP
# (Vast.ai allows outbound UDP fine; only inbound is blocked).
def _aiortc_ice_servers(cf_servers):
    out = []
    for srv in cf_servers:
        urls = srv.get("urls")
        if isinstance(urls, str):
            urls = [urls]
        # Filter to STUN + TURN-UDP only — skip TCP/TLS variants that hit aioice bugs
        filtered = [u for u in urls if "transport=tcp" not in u and not u.startswith("turns:")]
        if not filtered:
            continue
        out.append(_RTCIceServer(
            urls=filtered,
            username=srv.get("username"),
            credential=srv.get("credential"),
        ))
    return out


_DEFAULT_ICE_SERVERS = _aiortc_ice_servers(_CF_ICE)
if not _DEFAULT_ICE_SERVERS:
    _DEFAULT_ICE_SERVERS = [_RTCIceServer(urls="stun:stun.l.google.com:19302")]
print(f"[aiortc ICE] {[s.urls for s in _DEFAULT_ICE_SERVERS]}", flush=True)

_orig_handler_init = _SmallWebRTCRequestHandler.__init__


def _patched_handler_init(self, ice_servers=None, *args, **kwargs):
    if ice_servers is None:
        ice_servers = _DEFAULT_ICE_SERVERS
    _orig_handler_init(self, ice_servers=ice_servers, *args, **kwargs)


_SmallWebRTCRequestHandler.__init__ = _patched_handler_init


# (2b) Inject iceConfig into Pipecat's offer response so the BROWSER picks up
# the same ICE config. The prebuilt UI's JS reads `V.iceConfig.iceServers`
# from the answer and applies it to its RTCPeerConnection.
from pipecat.transports.smallwebrtc.connection import (
    SmallWebRTCConnection as _SmallWebRTCConnection,
)

_orig_get_answer = _SmallWebRTCConnection.get_answer


def _patched_get_answer(self):
    answer = _orig_get_answer(self)
    if answer is None:
        return None
    # Convert our internal _CF_ICE (list of dicts) to what the JS expects.
    answer["iceConfig"] = {"iceServers": _CF_ICE}
    return answer


_SmallWebRTCConnection.get_answer = _patched_get_answer
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# (3) Inject TLS args into Pipecat runner's uvicorn.run call.
# Browsers require HTTPS for getUserMedia (mic) from non-localhost origins.
# ---------------------------------------------------------------------------
import uvicorn as _uvicorn

_orig_uvicorn_run = _uvicorn.run


def _patched_uvicorn_run(app, **kwargs):
    cert = _os.environ.get("QWEN_DEMO_TLS_CERT", "/workspace/cert.pem")
    key = _os.environ.get("QWEN_DEMO_TLS_KEY", "/workspace/key.pem")
    if _os.path.exists(cert) and _os.path.exists(key):
        kwargs.setdefault("ssl_keyfile", key)
        kwargs.setdefault("ssl_certfile", cert)
    return _orig_uvicorn_run(app, **kwargs)


_uvicorn.run = _patched_uvicorn_run
# ---------------------------------------------------------------------------

import os

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
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.base_transport import BaseTransport, TransportParams

from pipecat_qwen_megakernel.services.qwen3_llm_local import (
    Qwen3LocalLLMService,
    Qwen3LocalLLMSettings,
)
from pipecat_qwen_megakernel.services.qwen3_tts_megakernel import (
    MegakernelQwen3TTSService,
)


# Lazy globals so we only pay model-load cost once per process even if multiple
# WebRTC clients connect. Loaded on first ``bot()`` invocation.
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


# Transport parameter factories. Pipecat's runner picks the right one based
# on ``-t <transport>``. For our demo only ``webrtc`` matters.
transport_params = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_out_sample_rate=24_000,    # match Qwen3-TTS native rate
        vad_analyzer=SileroVADAnalyzer(),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    tts_model_name = os.getenv("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    llm_model_name = os.getenv("QWEN_LLM_MODEL", "Qwen/Qwen3-1.7B")
    whisper_model = os.getenv("WHISPER_MODEL", "base.en")

    logger.info(
        f"Bot starting — TTS={tts_model_name} LLM={llm_model_name} STT=Whisper({whisper_model})"
    )

    hf_tts, tts_tokenizer = _load_tts_once(tts_model_name)

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
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def _on_connected(_t, _client):
        logger.info("Client connected — kicking off greeting")
        context.add_message(
            {
                "role": "developer",
                "content": "Please greet the user in one short sentence and ask how you can help.",
            }
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnected(_t, _client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments) -> None:
    """Main bot entry point — used by Pipecat's runner CLI."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main as _runner_main

    _runner_main()
