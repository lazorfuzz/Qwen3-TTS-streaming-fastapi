import time
import numpy as np
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel


def log_time(start, operation):
    elapsed = time.time() - start
    print(f"[{elapsed:.2f}s] {operation}")
    return time.time()


total_start = time.time()

start = time.time()
clone_model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
start = log_time(start, "Load Base model")

# for real speedup, use vLLM for LM inference (or SGlang probably)
# torch.compile doesn't help much for autoregressive generation due to dynamic shapes ( I think but idk )

ref_audio_path = "neurona-10sec (3).wav"
ref_text = (
    "реф текст"
)

voice_clone_prompt = clone_model.create_voice_clone_prompt(
    ref_audio=ref_audio_path,
    ref_text=ref_text,
)
start = log_time(start, "Create voice clone prompt")

# Test sentence
test_text = "Привет всем! Я того всё ебала, что за новый голос тут на обзоре у вилсакома? А? Так он мне понравился. Ганс оф буллщит."

# ============== Standard generation ==============
print("\n--- Standard generation ---")
start = time.time()
wavs, sr = clone_model.generate_voice_clone(
    text=test_text,
    language="Russian",
    voice_clone_prompt=voice_clone_prompt,
)
standard_time = time.time() - start
print(f"[{standard_time:.2f}s] Standard generate ({len(test_text)} chars)")
sf.write("clone_standard.wav", wavs[0], sr)

# ============== Streaming generation ==============
print("\n--- Streaming generation ---")
start = time.time()
chunks = []
first_chunk_time = None
chunk_count = 0

for chunk, chunk_sr in clone_model.stream_generate_voice_clone(
    text=test_text,
    language="Russian",
    voice_clone_prompt=voice_clone_prompt,
    emit_every_frames=8,
    decode_window_frames=80,
    overlap_samples=512,
):
    chunk_count += 1
    chunks.append(chunk)
    if first_chunk_time is None:
        first_chunk_time = time.time() - start
        print(f"[{first_chunk_time:.2f}s] First chunk received ({len(chunk)} samples)")

streaming_time = time.time() - start
print(f"[{streaming_time:.2f}s] Streaming complete ({chunk_count} chunks)")

# concatenate
final_audio = np.concatenate(chunks)
sf.write("clone_streaming.wav", final_audio, chunk_sr)

print(f"\n--- Summary ---")
print(f"Standard generation: {standard_time:.2f}s")
print(f"Streaming first chunk: {first_chunk_time:.2f}s")
print(f"Streaming total: {streaming_time:.2f}s")
print(f"Latency improvement: {standard_time - first_chunk_time:.2f}s faster to first audio")

print(f"\n[{time.time() - total_start:.2f}s] TOTAL")