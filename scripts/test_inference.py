#!/usr/bin/env python3
"""
Kokoro German: Test Inference
==============================
Tests the fine-tuned Kokoro model with a German phonetic test set.

Usage:
    # Convert checkpoint + run inference
    uv run python scripts/test_inference.py \
        --checkpoint StyleTTS2/logs/kokoro_german/epoch_1st_00002.pth \
        --voicepack voices/dm_daniel_epoch3.pt \
        --output-dir test_output/epoch3

    # Use a previously converted model
    uv run python scripts/test_inference.py \
        --model voices/kokoro_german_epoch3.pth \
        --voicepack voices/dm_daniel_epoch3.pt

    # Run on CPU
    uv run python scripts/test_inference.py \
        --checkpoint StyleTTS2/logs/kokoro_german/epoch_1st_00002.pth \
        --voicepack voices/dm_daniel_epoch3.pt \
        --device cpu
"""

import argparse
import sys
from pathlib import Path

# Prefer the kokoro submodule over any pip-installed kokoro package
_repo_root = Path(__file__).resolve().parents[1]
_kokoro_submodule = _repo_root / "kokoro"
if _kokoro_submodule.exists() and str(_kokoro_submodule) not in sys.path:
    sys.path.insert(0, str(_kokoro_submodule))

# Standard German phonetic test set — covers all major pronunciation challenges
TEST_SENTENCES = [
    "La République française est un État démocratique et laïque.",
    "Demain il pleuvra, n'oubliez pas de prendre votre parapluie.",
    "Pourriez-vous m'indiquer le chemin de la gare, s'il vous plaît ?",
    "Cette tâche nécessite une attention particulière et beaucoup de soin.",
    "Pourriez-vous m'expliquer comment fonctionne cet appareil ?",
    "La bibliothèque est ouverte du lundi au vendredi de huit heures à vingt heures.",
    "Veuillez excuser le retard, le trafic était particulièrement difficile aujourd'hui.",
    "Avez-vous déjà entendu l'Orchestre de Paris en concert ?",
    "Le médecin recommande de marcher au moins trente minutes par jour.",
]


def convert_checkpoint(checkpoint_path: str, output_path: str) -> str:
    """Convert a StyleTTS2 Stage 2 checkpoint to Kokoro KModel format.

    Extracts the inference components (bert, bert_encoder, predictor,
    text_encoder, decoder, and diffusion when present) from the training checkpoint. All state dict
    keys must have the 'module.' prefix for KModel's loading fallback
    to work correctly.

    Requires that training was done with the new parametrizations API
    (torch.nn.utils.parametrizations.weight_norm/spectral_norm) so the
    state dict keys are natively compatible with Kokoro's KModel.
    """
    import torch

    print(f"Converting checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    net = ckpt["net"]

    def ensure_module_prefix(state_dict):
        """Ensure all keys have 'module.' prefix for KModel compatibility."""
        return {
            ("module." + k if not k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }

    kokoro_weights = {}
    for key in ["bert", "bert_encoder", "predictor", "text_encoder", "decoder", "diffusion"]:
        if key in net:
            kokoro_weights[key] = ensure_module_prefix(net[key])
            print(f"  {key}: {len(kokoro_weights[key])} keys")
        else:
            print(f"  WARNING: '{key}' not found in checkpoint")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(kokoro_weights, str(output))
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"  Saved Kokoro-format weights: {output} ({size_mb:.1f} MB)")
    return str(output)


def run_inference(
    model_path: str,
    voicepack_path: str,
    config_path: str,
    output_dir: str,
    device: str = "auto",
    diffusion_steps: int = 0,
    diffusion_seed: int | None = None,
    diffusion_alpha: float = 0.1,
    diffusion_beta: float = 0.5,
    embedding_scale: float = 1.0,
):
    """Run inference on the German test set."""
    import torch
    import soundfile as sf
    from kokoro import KModel, KPipeline

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model with our fine-tuned weights and config
    print(f"Loading model from: {model_path}")
    print(f"  Config: {config_path}")
    kmodel = KModel(repo_id="hexgrad/Kokoro-82M", config=config_path, model=model_path)
    kmodel = kmodel.to(device).eval()

    # Create pipeline with French lang_code
    pipeline = KPipeline(lang_code="f", repo_id="hexgrad/Kokoro-82M", model=kmodel)

    # Load voicepack
    print(f"Loading voicepack: {voicepack_path}")
    voice = torch.load(voicepack_path, map_location="cpu", weights_only=True)

    # Create output directory
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Generate audio for each test sentence
    print(f"\nGenerating {len(TEST_SENTENCES)} test sentences...\n")
    for i, text in enumerate(TEST_SENTENCES):
        print(f"[{i + 1}/{len(TEST_SENTENCES)}] {text[:60]}...")
        try:
            generator = pipeline(
                text,
                voice=voice,
                speed=1,
                diffusion_steps=diffusion_steps,
                diffusion_seed=diffusion_seed,
                diffusion_alpha=diffusion_alpha,
                diffusion_beta=diffusion_beta,
                embedding_scale=embedding_scale,
            )
            all_audio = []
            for _, ps, audio in generator:
                assert ps is not None
                print(f"  phonemes: {ps[:60]}...")
                all_audio.append(audio)

            if all_audio:
                import numpy as np

                combined = np.concatenate(all_audio)
                wav_path = out / f"test_{i + 1:02d}.wav"
                sf.write(str(wav_path), combined, 24000)
                duration = len(combined) / 24000
                print(f"  saved: {wav_path} ({duration:.1f}s)")
            else:
                print("  WARNING: No audio generated")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone! Test audio saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Test fine-tuned Kokoro French model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--checkpoint",
        help="Path to StyleTTS2 checkpoint (.pth) — will be converted automatically",
    )
    group.add_argument(
        "--model",
        help="Path to already-converted Kokoro-format weights (.pth)",
    )
    parser.add_argument(
        "--voicepack",
        required=True,
        help="Path to voicepack (.pt)",
    )
    parser.add_argument(
        "--config",
        default="training/config.json",
        help="Path to Kokoro config.json",
    )
    parser.add_argument(
        "--output-dir",
        default="test_output/",
        help="Directory to save generated WAV files",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run on (default: auto)",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=0,
        help="Enable diffusion style sampling with this many denoising steps (0 = deterministic voicepack)",
    )
    parser.add_argument(
        "--diffusion-seed",
        type=int,
        default=None,
        help="Seed for repeatable diffusion style sampling",
    )
    parser.add_argument(
        "--diffusion-alpha",
        type=float,
        default=0.1,
        help="Blend for acoustic/timbre style when diffusion is enabled",
    )
    parser.add_argument(
        "--diffusion-beta",
        type=float,
        default=0.5,
        help="Blend for prosodic style when diffusion is enabled",
    )
    parser.add_argument(
        "--embedding-scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale for diffusion style sampling",
    )

    args = parser.parse_args()

    # Convert checkpoint if needed
    if args.checkpoint:
        model_path = convert_checkpoint(
            args.checkpoint,
            str(Path(args.output_dir) / "kokoro_french_converted.pth"),
        )
    else:
        model_path = args.model

    run_inference(
        model_path=model_path,
        voicepack_path=args.voicepack,
        config_path=args.config,
        output_dir=args.output_dir,
        device=args.device,
        diffusion_steps=args.diffusion_steps,
        diffusion_seed=args.diffusion_seed,
        diffusion_alpha=args.diffusion_alpha,
        diffusion_beta=args.diffusion_beta,
        embedding_scale=args.embedding_scale,
    )


if __name__ == "__main__":
    main()
