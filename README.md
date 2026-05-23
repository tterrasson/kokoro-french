# kokoro-french

Fine-tuning recipe for [Kokoro-82M](https://github.com/hexgrad/kokoro) (StyleTTS 2-based) for French TTS.

Forked from [semidark/kikiri-tts](https://github.com/semidark/kikiri-tts).

## Quick Start

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
# System dependencies
# Ubuntu/Debian: sudo apt-get install espeak-ng libsndfile1
# macOS:         brew install espeak-ng libsndfile

# Install Python dependencies
uv sync
```

## Training Pipeline

```
  Raw audio → Segment → Transcribe
                      → Filter → Cluster → Format
                      → Prepare splits → Convert weights
                      → Precompute → Verify
                      → Stage 1 (Decoder + Alignment)
                      → Stage 2 (Prosody Predictor)
                      → Export checkpoint → Extract voicepack
                      → Inference
```

### 1. Segment raw audio

Drop your raw audio files into the `raw/` directory, then split long recordings into short segments:

```bash
uv run python scripts/prepare_dataset.py segment --input-dir raw/
```

### 2. Transcribe segments

Generate IPA phoneme transcriptions using Whisper (resumable — safe to run overnight):

```bash
uv run python scripts/prepare_dataset.py transcribe
```

### 3. Filter

Remove low-quality segments (noise, wrong language, too short/long):

```bash
uv run python scripts/prepare_dataset.py filter
```

### 4. Cluster speakers

Group utterances into distinct speaker clusters using voice embeddings:

```bash
uv run python scripts/prepare_dataset.py cluster
```

Or specify a target number of speakers:

```bash
uv run python scripts/prepare_dataset.py cluster --n-speakers 5
```

### 5. Format and rename speakers

Convert audio to 24 kHz WAV, generate IPA phonemes, and assign speaker names:

```bash
uv run python scripts/prepare_dataset.py format --rename-speakers d_speaker0=ff_XXXX d_speaker1=dm_YYYY
```

### 6. Prepare training data

Generate train/validation splits and precompute mel spectrograms and F0 contours:

```bash
uv run python scripts/prepare_training.py prepare
```

Convert Kokoro HuggingFace weights to StyleTTS2 checkpoint format (downloads from HuggingFace, requires internet):

```bash
uv run python scripts/prepare_training.py convert-weights
```

Precompute mel spectrograms and F0 contours (optional but saves GPU time during training):

```bash
uv run python scripts/prepare_training.py precompute
```

Verify data integrity before training (check missing WAVs, unknown phonemes, etc.):

```bash
uv run python scripts/prepare_training.py verify
```

### 7. Train

#### Stage 1 — Decoder + Alignment

Trains the vocoder decoder and monotonic aligner. Produces a checkpoint with a working style encoder.

```bash
cd StyleTTS2
uv run accelerate launch train_first.py --config_path ../configs/config_french_ft.yml
```

#### Stage 2 — Prosody Predictor

Trains duration, pitch (F0), and energy predictors on top of the Stage 1 decoder.

```bash
uv run accelerate launch train_second.py --config_path ../configs/config_french_ft.yml
```

**Important config settings** (in `config_french_ft.yml`):

```yaml
second_stage_load_pretrained: false  # Load from first_stage.pth, not from scratch
joint_epoch: 3                       # Start adversarial training at epoch 3
lambda_slm: 1.0                      # Enable SLM adversarial loss
```

### 8. Export checkpoint

Convert the trained checkpoint to Kokoro KModel format:

```bash
uv run python scripts/export_checkpoint.py \
    --checkpoint StyleTTS2/logs/kokoro_french/epoch_2nd_00001.pth \
    --output voices/kokoro_french.pth
```

### 9. Extract voicepacks (optional)

Extract per-speaker voicepacks from a checkpoint.

> **Note:** Always pass `--style-encoder-model` pointing to the Stage 1 checkpoint. Stage 2 can degrade the style encoder (spectral norm drift), so load the timbre encoder from Stage 1 and the prosody predictor from Stage 2.

```bash
uv run python scripts/extract_voicepack.py \
    --model StyleTTS2/logs/kokoro_french/epoch_2nd_00001.pth \
    --style-encoder-model StyleTTS2/logs/kokoro_french/epoch_1st_00002.pth \
    --audio-dir dataset/audio/ff_XXXX \
    --output voices/ff_XXXX.pt
```

Add `--device cpu` to run on CPU (slower but works while the GPU is busy training).

### 10. Inference

```bash
uv run kokoro --text "Bonjour le monde" -o out.wav -l fr --voice <voice_name>
```

## Technical Details

- **Sample rate:** 24 000 Hz
- **Max phoneme length:** 510 tokens
- **Voicepack format:** `.pt`, shape `[510, 1, 256]` (float32)

## Repository Layout

```
kokoro/kokoro/       # Inference package (KModel, KPipeline)
kokoro/              # Upstream kokoro submodule
misaki/              # G2P phonemizer (bundled)
StyleTTS2/           # Patched training code (submodule)
scripts/             # Dataset prep, checkpoint export, voicepack extraction
configs/             # Training config(s)
training/            # Training metadata (lists, symbols, OOD texts)
docs/                # Troubleshooting, architecture notes
```

## License

Apache License 2.0 — see `LICENSE`.
