#!/usr/bin/env python3
"""
Kokoro French: Export Checkpoint
==================================
Converts a StyleTTS2 Stage 2 training checkpoint to Kokoro KModel format.

Usage:
    uv run python scripts/export_checkpoint.py \
        --checkpoint StyleTTS2/logs/kokoro_french/epoch_1st_00002.pth \
        --output voices/kokoro_french_epoch2.pth
"""

import argparse
from pathlib import Path


def export_checkpoint(checkpoint_path: str, output_path: str) -> str:
    """Convert a StyleTTS2 Stage 2 checkpoint to Kokoro KModel format.

    Extracts the 5 inference components (bert, bert_encoder, predictor,
    text_encoder, decoder) from the training checkpoint. All state dict
    keys must have the 'module.' prefix for KModel's loading fallback
    to work correctly.
    """
    import torch

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    net = ckpt["net"]

    def ensure_module_prefix(state_dict):
        return {
            ("module." + k if not k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }

    kokoro_weights = {}
    for key in ["bert", "bert_encoder", "predictor", "text_encoder", "decoder"]:
        if key in net:
            kokoro_weights[key] = ensure_module_prefix(net[key])
            print(f"  {key}: {len(kokoro_weights[key])} keys")
        else:
            print(f"  WARNING: '{key}' not found in checkpoint")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(kokoro_weights, str(output))
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"Saved: {output} ({size_mb:.1f} MB)")
    return str(output)


def main():
    parser = argparse.ArgumentParser(
        description="Export StyleTTS2 checkpoint to Kokoro KModel format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to StyleTTS2 training checkpoint (.pth)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the Kokoro-format weights (.pth)",
    )

    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
