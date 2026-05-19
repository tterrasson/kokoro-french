# kokoro-français

Training recipe for fine-tuning [Kokoro-82M](https://github.com/hexgrad/kokoro) for French with a patched [StyleTTS2](https://github.com/yl4579/StyleTTS2).

Forked from [semidark/kikiri-tts](https://github.com/semidark/kikiri-tts) and adapted for French.

## What This Is

- A reproducible fine-tuning workflow (dataset prep -> Stage 1 -> Stage 2 -> voicepack extraction)
- Original scripts for data preparation and checkpoint/voicepack conversion
- A patched `StyleTTS2/` submodule with the fixes required for stable Stage 2 training

## What This Is Not

- Not a general-purpose Kokoro replacement repository
- Not a bundled upstream mirror of `demo/`, `examples/`, `kokoro.js/`, or `tests/`
- Not a redistributable training dataset

## Start Here

### I want to train my own French voice

Start with `docs/TRAINING_GUIDE.md`.

### I am debugging training failures

Go to `docs/TROUBLESHOOTING.md`.

### I want architecture details and compatibility notes

See `docs/ARCHITECTURE.md`.

## Status

The end-to-end pipeline is working:

`Dataset preparation -> Weight conversion -> Stage 1 -> Stage 2 -> Voicepack extraction -> KModel inference`

## Quick Setup

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get install espeak-ng libsndfile1

# macOS
brew install espeak-ng libsndfile
```

### Clone

```bash
git clone https://github.com/tterrasson/kokoro-french
cd kokoro-french
uv sync
```

## Repository Layout

```text
kokoro/              # Kokoro fork submodule (contains the `kokoro/` Python package)
StyleTTS2/           # Patched training code (git submodule: semidark/StyleTTS2)
scripts/             # Dataset prep, voicepack extraction, inference testing
configs/             # Training config(s)
docs/                # Training guide, troubleshooting, architecture notes
training/            # Local training artifacts metadata (audio excluded)
```

## Contributing

Contributions are welcome, especially:

- Reproducible runs on public datasets
- Training stability and quality improvements
- French dataset contributions

## Attribution

See `NOTICE` for upstream attribution and license details.

## License

Apache License 2.0 — see `LICENSE`.
