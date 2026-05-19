# AGENTS.md — kokoro-french

Fine-tuning recipe for [Kokoro-82M](https://github.com/hexgrad/kokoro) (StyleTTS 2-based) for French TTS. Forked from [semidark/kikiri-tts](https://github.com/semidark/kikiri-tts).

- **Python:** 3.10–3.14
- **Package manager:** `uv`

## Setup

```bash
uv sync
```

## CLI

```bash
uv run kokoro --text "Bonjour le monde" -o out.wav -l fr --voice <voice_name>
```

## Project Structure

```
kokoro/kokoro/       # Inference package (KModel, KPipeline)
kokoro/              # Upstream kokoro submodule (demo, examples, tests)
misaki/              # G2P phonemizer (bundled, not submodule)
StyleTTS2/           # Patched training code (submodule: semidark/StyleTTS2)
scripts/             # prepare_dataset.py, prepare_training.py, extract_voicepack.py, test_inference.py
configs/             # config_french_ft.yml, config_german_ft.yml
training/            # Training metadata (lists, symbols, OOD texts)
docs/                # TRAINING_GUIDE.md, TROUBLESHOOTING.md, ARCHITECTURE.md
```

## Training Pipeline

`Dataset prep → Weight conversion → Stage 1 → Stage 2 → Voicepack extraction → KModel inference`

### Training scripts

- `StyleTTS2/train_first.py` — Stage 1: decoder + alignment
- `StyleTTS2/train_second.py` — Stage 2: prosody predictor
- `StyleTTS2/train_finetune.py` — Full fine-tuning
- `StyleTTS2/train_finetune_accelerate.py` — Accelerate-based fine-tuning
- `scripts/extract_voicepack.py` — Extract `.pt` voicepack from checkpoint
- `scripts/test_inference.py` — Convert checkpoint + run inference tests

### Stage 2 config requirements

```yaml
joint_epoch: 3                  # Start adversarial training (not 999)
lambda_slm: 1.0                 # Enable SLM adversarial loss
second_stage_load_pretrained: false  # Load from first_stage.pth
```

## Technical Details

- **Sample rate:** 24000 Hz
- **Max phoneme length:** 510 tokens
- **Voice files:** `.pt`, shape `[510, 1, 256]` (float32)
