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

## Expectations

This is experimental — the model was not pretrained on Turkish. Results depend on:
- How well the text encoder handles Turkish characters (ğ, ş, ı, ç exist in German/French so partial overlap)
- Whether 5.6 hours is enough data for language adaptation
- How different Turkish phonology is from the 10 supported languages

The codec encoder/decoder are language-agnostic (pure audio compression), so they work on Turkish audio without modification.
