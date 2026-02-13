"""
Lightweight FastAPI server exposing an OpenAI-compatible /v1/audio/speech endpoint for streaming TTS audio.

- Accepts POST requests with JSON: {"input": "text to synthesize"}
- Streams PCM audio bytes as response (audio/wav)
- Uses a BatchScheduler to batch concurrent requests into a single forward pass
- On-demand voice clone loading with per-process caching

Configuration via environment variables:
    TTS_NUM_WORKERS: Number of uvicorn worker processes (used by entrypoint.sh)
    TTS_VOICE_META_DIR: Directory for voice metadata JSON files (default: /app/voices)
    TTS_API_KEY: Optional API key for Bearer auth on /v1/* endpoints (unset = auth disabled)
    TTS_MAX_BATCH_SIZE: Maximum batch size for concurrent requests (default: 8)
    TTS_BATCH_WAIT_MS: Max milliseconds to wait for batch to fill (default: 50)
"""

import asyncio
import hmac
import json
import os
import queue
import threading
import traceback
import numpy as np
from fastapi import Depends, FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from qwen_tts import Qwen3TTSModel
import time
import torch
from typing import Optional
from dataclasses import dataclass

_SENTINEL = object()


# ---------------------------------------------------------------------------
# App and model initialization
# ---------------------------------------------------------------------------

app = FastAPI()

DEFAULT_VOICE_CLONE_REF_PATH = "eesha_voice_cloning.wav"
DEFAULT_TEXT = "Hello. This is an audio recording that's at least 5 seconds long. How are you doing today? Bye!"

VOICE_META_DIR = os.environ.get("TTS_VOICE_META_DIR", "/app/voices")
API_KEY = os.environ.get("TTS_API_KEY")
MAX_BATCH_SIZE = int(os.environ.get("TTS_MAX_BATCH_SIZE", "8"))
BATCH_WAIT_S = int(os.environ.get("TTS_BATCH_WAIT_MS", "50")) / 1000.0
os.makedirs(VOICE_META_DIR, exist_ok=True)

print(f"[INIT] Loading model (PID={os.getpid()})...", flush=True)

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
model.enable_streaming_optimizations(
    decode_window_frames=80,
    use_compile=True,
    use_cuda_graphs=False,
    compile_mode="max-autotune-no-cudagraphs",
    use_fast_codebook=True,
    compile_codebook_predictor=True,
    compile_talker=True,
)

# In-process voice clone cache: filepath → VoiceClonePromptItem
VOICE_CLONE_CACHE = {}

# Pre-build default voice at startup
VOICE_CLONE_CACHE[DEFAULT_VOICE_CLONE_REF_PATH] = model.create_voice_clone_prompt(
    ref_audio=DEFAULT_VOICE_CLONE_REF_PATH,
    ref_text=DEFAULT_TEXT,
)

print(f"[INIT] Model ready (PID={os.getpid()}).", flush=True)


# ---------------------------------------------------------------------------
# BatchScheduler — collects concurrent requests and processes as one batch
# ---------------------------------------------------------------------------

@dataclass
class _BatchItem:
    """One pending request waiting to be batched."""
    text: str
    language: str
    voice_clone_prompt: object  # VoiceClonePromptItem
    output_queue: queue.Queue
    stop_event: threading.Event
    start_time: float


class BatchScheduler:
    """
    Collects incoming TTS requests and dispatches them as batched forward passes.

    Instead of N independent workers each doing batch_size=1, a single scheduler
    batches up to max_batch_size requests into one forward pass, converting
    matrix-vector ops into matrix-matrix ops for much higher GPU utilization.
    """

    def __init__(self, model, max_batch_size: int = 8, max_wait_s: float = 0.05):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_s = max_wait_s

        self._pending: asyncio.Queue[_BatchItem] = asyncio.Queue()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._started = False

    async def start(self):
        """Start the scheduler background loop. Must be called from async context."""
        if self._started:
            return
        self._loop = asyncio.get_running_loop()
        self._started = True
        asyncio.create_task(self._scheduler_loop())
        print(
            f"[SCHED] BatchScheduler started: max_batch={self.max_batch_size}, "
            f"wait={self.max_wait_s*1000:.0f}ms",
            flush=True,
        )

    async def submit(self, text: str, language: str, voice_clone_prompt) -> tuple[queue.Queue, threading.Event]:
        """
        Submit a request for batched generation.

        Returns (output_queue, stop_event). The caller consumes PCM bytes from
        the queue until _SENTINEL is received. Setting stop_event cancels this
        item without affecting others in the batch.
        """
        if not self._started:
            await self.start()

        q = queue.Queue(maxsize=16)
        stop_event = threading.Event()
        item = _BatchItem(
            text=text,
            language=language,
            voice_clone_prompt=voice_clone_prompt,
            output_queue=q,
            stop_event=stop_event,
            start_time=time.time(),
        )
        await self._pending.put(item)
        return q, stop_event

    async def _scheduler_loop(self):
        """Background loop: collect requests, dispatch batches."""
        while True:
            # Wait for at least one request
            first = await self._pending.get()
            batch = [first]

            # Collect more requests up to max_batch_size or max_wait_s
            deadline = time.monotonic() + self.max_wait_s
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(
                        self._pending.get(), timeout=remaining
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            print(
                f"[SCHED] Dispatching batch of {len(batch)} request(s)",
                flush=True,
            )

            # Run generation in thread pool to avoid blocking the event loop
            loop = self._loop
            await loop.run_in_executor(None, self._generate_batch, batch)

    def _generate_batch(self, batch: list[_BatchItem]):
        """Run batched generation in a thread. Dispatches PCM chunks to per-item queues."""
        texts = [item.text for item in batch]
        language = batch[0].language  # All items use the same language
        voice_clone_prompts = [item.voice_clone_prompt for item in batch]
        stop_events = [item.stop_event for item in batch]
        ttfb_printed = [False] * len(batch)
        sentinel_sent = [False] * len(batch)

        def _send_sentinel(idx: int):
            if sentinel_sent[idx]:
                return
            sentinel_sent[idx] = True
            item = batch[idx]
            gen_time = time.time() - item.start_time
            print(
                f"[GEN_DONE] PID={os.getpid()} {gen_time:.3f}s input: {item.text[:60]}",
                flush=True,
            )
            try:
                item.output_queue.put(_SENTINEL, timeout=1)
            except queue.Full:
                pass

        try:
            if len(batch) == 1:
                # Single request: use the original unbatched path for efficiency
                item = batch[0]
                try:
                    for chunk, sr in self.model.stream_generate_voice_clone(
                        text=item.text,
                        language=item.language,
                        voice_clone_prompt=item.voice_clone_prompt,
                        emit_every_frames=4,
                        decode_window_frames=80,
                        overlap_samples=512,
                    ):
                        if item.stop_event.is_set():
                            print(
                                f"[CANCEL] PID={os.getpid()} client disconnected: {item.text[:60]}",
                                flush=True,
                            )
                            break
                        if not ttfb_printed[0]:
                            ttfb = time.time() - item.start_time
                            print(
                                f"[TTFB] PID={os.getpid()} {ttfb:.3f}s input: {item.text[:60]}",
                                flush=True,
                            )
                            ttfb_printed[0] = True
                        chunk_int16 = np.clip(chunk, -1.0, 1.0)
                        chunk_bytes = (chunk_int16 * 32767.0).astype(np.int16).tobytes()
                        try:
                            item.output_queue.put(chunk_bytes, timeout=5)
                        except queue.Full:
                            print(
                                f"[CANCEL] PID={os.getpid()} queue full: {item.text[:60]}",
                                flush=True,
                            )
                            break
                finally:
                    _send_sentinel(0)
            else:
                # Batched generation
                finished = [False] * len(batch)
                try:
                    for results in self.model.batched_stream_generate_voice_clone(
                        texts=texts,
                        language=language,
                        voice_clone_prompts=voice_clone_prompts,
                        stop_events=stop_events,
                        emit_every_frames=4,
                        decode_window_frames=80,
                        overlap_samples=512,
                    ):
                        for i, result in enumerate(results):
                            if finished[i]:
                                continue
                            if result is None:
                                continue
                            chunk, sr = result
                            item = batch[i]
                            if not ttfb_printed[i]:
                                ttfb = time.time() - item.start_time
                                print(
                                    f"[TTFB] PID={os.getpid()} {ttfb:.3f}s input: {item.text[:60]}",
                                    flush=True,
                                )
                                ttfb_printed[i] = True
                            chunk_int16 = np.clip(chunk, -1.0, 1.0)
                            chunk_bytes = (chunk_int16 * 32767.0).astype(np.int16).tobytes()
                            try:
                                item.output_queue.put(chunk_bytes, timeout=5)
                            except queue.Full:
                                print(
                                    f"[CANCEL] PID={os.getpid()} queue full: {item.text[:60]}",
                                    flush=True,
                                )
                                item.stop_event.set()

                        # Send sentinel immediately for items that just finished
                        # (EOS or cancelled) so clients don't wait for the whole batch
                        for i, item in enumerate(batch):
                            if not finished[i] and item.stop_event.is_set():
                                finished[i] = True
                                _send_sentinel(i)
                except Exception as e:
                    print(f"[ERROR] PID={os.getpid()} Batch generation error: {e}", flush=True)
                    traceback.print_exc()
                finally:
                    for i in range(len(batch)):
                        _send_sentinel(i)

        except Exception as e:
            print(f"[ERROR] PID={os.getpid()} Fatal batch error: {e}", flush=True)
            traceback.print_exc()
            for i in range(len(batch)):
                _send_sentinel(i)


scheduler = BatchScheduler(model, max_batch_size=MAX_BATCH_SIZE, max_wait_s=BATCH_WAIT_S)


# ---------------------------------------------------------------------------
# Auth dependency — enabled only when TTS_API_KEY is set
# ---------------------------------------------------------------------------

async def verify_api_key(request: Request):
    if not API_KEY:
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing API key")
    token = auth[7:]
    if not hmac.compare_digest(token, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")


# ---------------------------------------------------------------------------
# On-demand voice loading
# ---------------------------------------------------------------------------

def _get_voice_clone_prompt(filepath: str):
    """Load voice clone prompt from cache or disk metadata. Returns None if not found."""
    if filepath in VOICE_CLONE_CACHE:
        return VOICE_CLONE_CACHE[filepath]

    basename = os.path.basename(filepath)
    meta_path = os.path.join(VOICE_META_DIR, f"{basename}.json")
    if not os.path.exists(meta_path):
        return None

    with open(meta_path, "r") as f:
        meta = json.load(f)

    prompt = model.create_voice_clone_prompt(meta["ref_audio"], meta["ref_text"])
    VOICE_CLONE_CACHE[filepath] = prompt
    return prompt


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SpeechRequest(BaseModel):
    input: str
    cloning_audio_filename: Optional[str] = None


class AddVoiceRequest(BaseModel):
    ref_audio_filename: str
    ref_text: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/v1/add_voice", dependencies=[Depends(verify_api_key)])
async def add_voice(body: AddVoiceRequest):
    """
    Register a voice for cloning. The .wav file must already exist in /app.
    Metadata is persisted to disk so other workers can load it on-demand.
    """
    filepath = f"/app/{body.ref_audio_filename}"
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"File not found: {filepath}")

    try:
        # Persist metadata to disk for cross-worker discovery
        # Use basename to match _get_voice_clone_prompt's lookup key
        meta_path = os.path.join(VOICE_META_DIR, f"{os.path.basename(body.ref_audio_filename)}.json")
        with open(meta_path, "w") as f:
            json.dump({"ref_audio": filepath, "ref_text": body.ref_text}, f)

        # Eagerly cache in THIS worker
        prompt = model.create_voice_clone_prompt(filepath, body.ref_text)
        VOICE_CLONE_CACHE[filepath] = prompt

        return {"status": "success", "message": f"Voice added: {body.ref_audio_filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/audio/speech", dependencies=[Depends(verify_api_key)])
async def speech_endpoint(request: Request, body: SpeechRequest):
    text = body.input
    language = "English"

    voice_clone_prompt = VOICE_CLONE_CACHE[DEFAULT_VOICE_CLONE_REF_PATH]
    if body.cloning_audio_filename:
        print(f"[INFO] Using cloning audio: {body.cloning_audio_filename}", flush=True)
        loaded = _get_voice_clone_prompt(body.cloning_audio_filename)
        if loaded is not None:
            voice_clone_prompt = loaded

    print(
        f"[REQ] /v1/audio/speech PID={os.getpid()} input: {text[:60]}",
        flush=True,
    )

    output_queue, stop_event = await scheduler.submit(text, language, voice_clone_prompt)

    def audio_stream():
        try:
            while True:
                try:
                    item = output_queue.get(timeout=30)
                except queue.Empty:
                    print(
                        f"[TIMEOUT] PID={os.getpid()} no data for 30s, aborting: {text[:60]}",
                        flush=True,
                    )
                    break
                if item is _SENTINEL:
                    break
                yield item
        finally:
            stop_event.set()

    headers = {
        "Content-Type": "audio/L16; rate=24000",
        "Transfer-Encoding": "chunked",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "X-Content-Type-Options": "nosniff",
    }
    return StreamingResponse(audio_stream(), headers=headers, media_type="audio/L16")


@app.get("/")
def root():
    return {
        "message": "Qwen3-TTS Streaming FastAPI server. POST /v1/audio/speech /v1/add_voice",
        "pid": os.getpid(),
    }
