# Qwen3-TTS Streaming

Streaming inference implementation for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) that the official repo doesn't provide.

The official team mentions "Extreme Low-Latency Streaming Generation" in their paper and marketing, but the actual streaming code was never released - they point users to vLLM-Omni, which still doesn't support online serving.

This fork adds real streaming generation directly to the `qwen-tts` package.

## What's Added

- `stream_generate_pcm()` - real-time PCM audio streaming
- `stream_generate_voice_clone()` - streaming with voice cloning

## Benchmark (A100, H100)
Will add my own benchmarking here later.

## Usage

See examples/
- [test_streaming_optimized.py](https://github.com/dffdeeq/Qwen3-TTS-streaming/blob/main/examples/test_streaming_optimized.py)
- [test_optimized_no_streaming.py](https://github.com/dffdeeq/Qwen3-TTS-streaming/blob/main/examples/test_optimized_no_streaming.py)

## Running the FastAPI Server

The server exposes an OpenAI-compatible `POST /v1/audio/speech` endpoint that streams PCM audio. It uses nginx + supervisord to load-balance across multiple uvicorn workers, each with its own model instance.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_NUM_WORKERS` | `2` | Number of uvicorn worker processes. Set to `auto` to match the number of NVIDIA MIG instances. |
| `TTS_VOICE_META_DIR` | `/app/voices` (Docker), `./voices` (local) | Directory for voice metadata JSON files |
| `TTS_API_KEY` | (empty) | Optional API key for Bearer auth on `/v1/*` endpoints. Unset/empty disables auth. |

### Option A: Docker

Uses `entrypoint.sh`, which writes configs to `/etc/` and assumes the `/app` working directory.

```bash
docker build --network=host -t qwen3-tts-server .
docker run --gpus all -p 8000:8000 -v $(pwd):/app -v $(pwd)/hf_cache:/root/.cache/huggingface --network=host qwen3-tts-server
```

MIG is also supported in Docker — pass `TTS_NUM_WORKERS=auto` and expose MIG devices via `--gpus`:

```bash
docker run --gpus all -e TTS_NUM_WORKERS=auto -p 8000:8000 -v $(pwd):/app -v $(pwd)/hf_cache:/root/.cache/huggingface --network=host qwen3-tts-server
```

### Option A2: Docker Compose

Wraps the same Docker image with auto-restart, health checks, and env var configuration in one command.

```bash
docker compose up -d --build
```

Configure via environment variables (inline or in a `.env` file):

```bash
TTS_NUM_WORKERS=4 TTS_API_KEY=my-secret docker compose up -d --build
```

Or create a `.env` file in the project root:

```env
TTS_NUM_WORKERS=2
TTS_API_KEY=my-secret
TTS_MAX_BATCH_SIZE=16
TTS_BATCH_WAIT_MS=100
```

Then just `docker compose up -d --build`.

Useful commands:

```bash
docker compose logs -f       # tail logs
docker compose down          # stop and remove container
docker compose restart       # restart container
```

The container will automatically restart if it crashes, gets OOM-killed, or the host reboots. Individual workers also auto-restart via supervisord if they crash.

MIG is also supported — set `TTS_NUM_WORKERS=auto`:

```bash
TTS_NUM_WORKERS=auto docker compose up -d --build
```

### Option B: Directly on a VM (no Docker, no sudo)

Uses `entrypoint_local.sh`, which generates nginx and supervisord configs in a local `.run/` directory and does not require root.

Prerequisites: `nginx`, `supervisord`, `uvicorn`, and Python dependencies installed.

```bash
pip install -e .
TTS_NUM_WORKERS=2 ./entrypoint_local.sh
```

#### NVIDIA MIG support

On GPUs that support Multi-Instance GPU (e.g. A100), you can partition the GPU into isolated slices and run one worker per slice. Create the MIG instances first, then use `auto`:

```bash
# Create 7 MIG instances (1g.10gb each on an A100 80GB)
sudo nvidia-smi mig -cgi 19,19,19,19,19,19,19 -C

# Start with one worker per MIG slice (auto-detected)
TTS_NUM_WORKERS=auto ./entrypoint_local.sh
```

Each worker gets its own `CUDA_VISIBLE_DEVICES` set to the MIG device UUID, providing hardware-level memory and compute isolation without MPS.

### Testing

Stream audio and play it locally:
```bash
curl -N -s -X POST http://localhost:8000/v1/audio/speech -H "Content-Type: application/json" -d '{"input": "Hello, this is a test."}' | ffplay -nodisp -autoexit -f s16le -ar 24000 -ch_layout mono -
```

If `TTS_API_KEY` is set, add `-H "Authorization: Bearer <your-key>"`.

Run a concurrency test (default 2 concurrent requests):
```bash
./concurrency_test.sh                            # 2 concurrent requests to localhost:8000
./concurrency_test.sh http://<host>:8000 4       # custom host, 4 requests
```

Adding a new voice cloning audio:
```bash
curl -N -X POST "http://<host>:8000/v1/add_voice" -F "file=@../output_voice_design.wav" -F "ref_text=Its in the top drawer. Wait, its empty? No way, thats impossible, Im sure I put it there"
```
All workers will pick up new cloning audios added within 5 seconds and begin their prewarm, this may take up to 1min.

Then, to use a specific voice, the cloning_audio_filename field:
```bash
curl -N -s -X POST http://<host>:8000/v1/audio/speech -H "Content-Type: application/json" -d '{"input": "옛날에 큰 호랑이 한 마리가 숲  속에 살았다. 어느 날 호랑이는 배가 고파서 마을로 갔다. 마을 옆 밭에 소 한 마리가 서 있었다.", "language_id": "ko", "cloning_audio_filename": "output_voice_design.wav"}' | ffplay -nodisp -autoexit -f s16le -ar 24000 -ch_layout mono -
```

## Installation (python 3.12)

> Note: torch versions differ between Linux/Windows due to available flash_attn prebuilt wheels.

### 1. Install SOX

**Linux:**
```bash
sudo apt install sox libsox-fmt-all
```

**Windows:**
```bash
# Download from https://sourceforge.net/projects/sox/ and add to PATH !!
```

### 2. Create environment
```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
```

### 3. Install dependencies

**Linux:**
```bash
pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.6.8/flash_attn-2.8.3%2Bcu130torch2.9-cp312-cp312-linux_x86_64.whl
```

**Windows:**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.12/flash_attn-2.8.3%2Bcu130torch2.10-cp312-cp312-win_amd64.whl
pip install -U "triton-windows<3.7"
```

### 4. Install package
```bash
git clone https://github.com/dffdeeq/Qwen3-TTS-streaming.git
cd Qwen3-TTS-streaming
pip install -e .
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `emit_every_frames` | 4 | Emit audio every N frames (~0.33s at 12Hz) |
| `decode_window_frames` | 80 | Decoder context window |

## Why This Exists

From official Qwen3-TTS README:
> Now only offline inference is supported. Online serving will be supported later.

This fork provides streaming now, without waiting for vLLM-Omni updates.

---

Based on [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
