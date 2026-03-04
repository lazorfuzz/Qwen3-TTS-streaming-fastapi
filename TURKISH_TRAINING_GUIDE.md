# Turkish Language Training for Qwen3-TTS

## Overview

This guide covers finetuning Qwen3-TTS to speak Turkish, a language not in the original 10-language set. Two files were modified from the upstream repo (`dataset.py` and `sft_12hz.py`) to support injecting a language token during training — without this, the model wouldn't learn to associate a specific token with Turkish, and inference with `language="turkish"` wouldn't work.

## Architecture Context

Qwen3-TTS has three components:
- **Encoder** (MimiModel): Generic audio codec, compresses audio → discrete codes. Language-agnostic — no changes needed.
- **Decoder**: Reconstructs waveform from codes. Also language-agnostic — no changes needed.
- **Talker** (transformer LM): Maps text → audio codes. This is where language knowledge lives. This is what gets finetuned.

The talker uses a two-channel input at each position:
- **Text channel**: Tokenized text (from Qwen2 tokenizer)
- **Codec channel**: Audio codec tokens + special prefix tokens (language, speaker, think/nothink)

## What Was Changed

### 1. `finetuning/dataset.py`

**Problem**: The original `collate_fn` hardcoded a "nothink" codec prefix with no language token:
```
[nothink, think_bos, think_eos, speaker_slot, pad]  (5 tokens, positions 3-7)
```
But at inference time, specifying a language uses a "think" prefix with a language token:
```
[think, think_bos, language_id, think_eos]  (4 tokens)
```
This mismatch means the model never sees a language token during training.

**Fix**: Added `language_id` parameter to `TTSDataset`. When set, `collate_fn` uses:
```
[think, think_bos, language_id, think_eos, speaker_slot, pad]  (6 tokens, positions 3-8)
```
This adds 1 extra position, so all subsequent positions shift by 1:
- Prefix offset `o` = 9 (was 8)
- Speaker embedding position `spk_pos` = 7 (was 6)
- All text/codec positions use `o` instead of hardcoded `8`

When `language_id=None`, behavior is identical to the original code (backward compatible).

The batch now also returns `spk_pos` so the training script knows where to inject the speaker embedding.

### 2. `finetuning/sft_12hz.py`

**Changes**:
- Added `--language` arg (e.g., `"turkish"`) and `--language_id` arg (e.g., `2072`)
- Resolves the language_id: checks model config first, falls back to `--language_id`
- Passes `language_id` to `TTSDataset`
- Uses dynamic `spk_pos` from batch instead of hardcoded `6` for speaker embedding injection
- On checkpoint save, adds the new language to `codec_language_id` in config.json

## Training Data

**Source**: Google FLEURS Turkish subset (`google/fleurs`, `tr_tr` split)

**Location** (relative to repo root): `../fleurs-turkish/`

```
../fleurs-turkish/
├── train/                  # 2526 wav files (extracted from train.tar.gz)
├── train.tsv               # FLEURS metadata
├── train.tar.gz            # original archive (can delete)
├── convert_to_jsonl.py     # TSV → JSONL conversion script
└── train_raw.jsonl         # 1753 female-filtered entries, ready for prepare_data.py
```

**Data format** — `train_raw.jsonl` has one JSON object per line:
```json
{"audio": "train/1424211997093929997.wav", "text": "Proje tabanl\u0131 \u00f6\u011frenme...", "ref_audio": "train/7412286569331940983.wav"}
```

The FLEURS TSV columns are: `sentence_id | wav_filename | original_text | normalized_text | char_tokenization | num_samples | gender`. The conversion script (`convert_to_jsonl.py`) filters to `--gender FEMALE` (1753 clips, ~5.6 hours) and picks one random wav as `ref_audio` for all entries.

**Note**: FLEURS is multi-speaker (no speaker IDs in TSV). The finetuning pipeline's single-speaker design means only one speaker embedding gets baked into the saved model. For language learning this is fine — the model learns Turkish pronunciation from all speakers during training; the output voice is just whoever's ref_audio gets saved.

## Full Training Commands (on Linux GPU machine)

### Setup
```bash
git clone https://github.com/QwenLM/Qwen3-TTS.git
cd Qwen3-TTS
pip install qwen-tts
pip install flash-attn --no-build-isolation
```

Apply the changes to `finetuning/dataset.py` and `finetuning/sft_12hz.py` as described above (the modified files are in the repo on the `training_pipeline` branch).

### Step 1: Encode audio → codec tokens
```bash
cd finetuning

python prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl ../fleurs-turkish/train_raw.jsonl \
  --output_jsonl ../fleurs-turkish/train_with_codes.jsonl
```

### Step 2: Finetune
```bash
python sft_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path output \
  --train_jsonl ../fleurs-turkish/train_with_codes.jsonl \
  --batch_size 2 \
  --lr 2e-5 \
  --num_epochs 3 \
  --speaker_name turkish_speaker \
  --language turkish \
  --language_id 2072
```

Use `Qwen/Qwen3-TTS-12Hz-0.6B-Base` instead if you have limited VRAM.

### Step 3: Inference
```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

tts = Qwen3TTSModel.from_pretrained(
    "output/checkpoint-epoch-2",
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

wavs, sr = tts.generate_custom_voice(
    text="Merhaba, bugun hava cok guzel.",
    speaker="turkish_speaker",
    language="turkish",
)
sf.write("output.wav", wavs[0], sr)
```

## Language Token ID

Turkish uses codec token ID `2072`. Existing language IDs in the base model:

| Language | ID |
|----------|-----|
| English | 2050 |
| German | 2053 |
| Spanish | 2054 |
| Chinese | 2055 |
| Japanese | 2058 |
| French | 2061 |
| Korean | 2064 |
| Russian | 2069 |
| Italian | 2070 |
| Portuguese | 2071 |
| **Turkish** | **2072** (new) |

## Improving Training Quality

The first run (3 epochs, lr=2e-5, all layers unfrozen) produced garbled output with loss plateauing around 10-11. Below are concrete improvements to try, roughly ordered by expected impact.

### 1. More Epochs, Lower Learning Rate

The model needs more passes over the data to learn a new language. 3 epochs on 1753 samples is too few. A lower learning rate prevents "catastrophic forgetting" — where the model overwrites its existing audio generation ability while trying to learn Turkish.

```bash
python sft_12hz.py \
  --init_model_path ~/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path output \
  --train_jsonl ~/fleurs-turkish/train_with_codes.jsonl \
  --batch_size 2 --lr 5e-6 --num_epochs 15 \
  --speaker_name turkish_speaker \
  --language turkish --language_id 2072
```

- `--lr 5e-6` (was `2e-5`) — 4x smaller, less destructive to pretrained weights
- `--num_epochs 15` (was `3`) — more time to learn the language

No additional data needed. The model just loops over the same 1753 samples more times.

### 2. Freeze Lower Transformer Layers

Currently all talker parameters are updated, which risks destroying the model's core audio generation capability. Freezing the lower transformer layers preserves the model's foundational knowledge (attention patterns, audio structure) while letting the upper layers adapt to Turkish.

Add this to `sft_12hz.py` after model loading, before the optimizer:

```python
# Freeze lower N layers of the talker (keep top layers trainable)
FREEZE_LAYERS_BELOW = 20  # freeze layers 0-19, train layers 20+

for name, param in qwen3tts.model.named_parameters():
    param.requires_grad = True  # default: trainable

for i in range(FREEZE_LAYERS_BELOW):
    for param in qwen3tts.model.talker.model.layers[i].parameters():
        param.requires_grad = False
```

The 1.7B model has 28 transformer layers. Freezing the bottom 20 and training the top 8 is a reasonable starting point. Adjust `FREEZE_LAYERS_BELOW` based on results — fewer frozen layers = more capacity to learn, more frozen = safer against forgetting.

### 3. Use the 0.6B Model

Smaller models adapt faster with limited data. The 0.6B model has fewer parameters, so the same 1753 samples represent a higher data-to-parameter ratio. It may overfit more readily, but for language adaptation that can actually help — the model memorizes Turkish phonetic patterns more aggressively.

```bash
python sft_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --output_model_path output_0.6b \
  ...
```

Requires less VRAM too (~8-10GB vs ~20-30GB for 1.7B). Can run on a T4 or even the A10-8Q.

### 4. Single-Speaker Data

FLEURS is multi-speaker with no speaker IDs in the metadata. The current setup uses audio from many different speakers but a single `ref_audio` for the speaker embedding. This creates a mismatch: the speaker embedding says "sound like speaker A" but the target audio alternates between speakers A, B, C, etc. The model gets conflicting signals about what the output should sound like.

To fix this, cluster the FLEURS audio by speaker using a speaker embedding model (e.g., `speechbrain/spkrec-ecapa-voxceleb`):

1. Extract speaker embeddings for all 1753 wav files
2. Cluster with k-means or agglomerative clustering (try k=5-10)
3. Pick the largest single-speaker cluster
4. Rebuild `train_raw.jsonl` with only that cluster's entries
5. Set `ref_audio` to a file from that same cluster

This gives the model consistent voice targets during training. The tradeoff is less data (maybe 300-500 samples from one speaker instead of 1753 from many), so you'll need even more epochs.

### 5. Get More Turkish Audio Data

5.6 hours may not be enough to teach a new language. Other sources of Turkish audio:

- **Common Voice** (`mozilla-foundation/common_voice_17_0`, `tr` split) — crowd-sourced, many speakers, ~100+ hours
- **LibriVox Turkish** — public domain audiobooks, fewer speakers, longer utterances
- **YouTube with subtitles** — can be scraped, but quality varies and requires alignment

More data won't help if the fundamental approach (single-speaker pipeline, no layer freezing) isn't working, so fix those first.

### Recommended Experiment Order

1. Lower lr + more epochs (easiest, no code changes)
2. Freeze lower layers (small code change in sft_12hz.py)
3. Try 0.6B model (just change the model path)
4. Speaker clustering (requires additional tooling)
5. More data (requires data collection pipeline)

Compare checkpoints from different epochs — earlier epochs may sound better if later ones overfit. Generate the same test sentence from each checkpoint and listen.

## Expectations

This is experimental — the model was not pretrained on Turkish. Results depend on:
- How well the text encoder handles Turkish characters (ğ, ş, ı, ç exist in German/French so partial overlap)
- Whether 5.6 hours is enough data for language adaptation
- How different Turkish phonology is from the 10 supported languages

The codec encoder/decoder are language-agnostic (pure audio compression), so they work on Turkish audio without modification.
