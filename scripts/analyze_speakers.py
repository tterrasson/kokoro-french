#!/usr/bin/env python3
"""
Standalone speaker count analyzer — works on raw, unsegmented audio files.

Decodes each file once at 16 kHz, extracts N evenly-spaced windows, and embeds
each window independently. All window embeddings are clustered globally, which
correctly handles files that contain multiple speakers.

Usage:
    uv run python scripts/analyze_speakers.py raw/
    uv run python scripts/analyze_speakers.py raw/ --n-speakers 6
    uv run python scripts/analyze_speakers.py raw/ --windows 30 --window-s 8
    uv run python scripts/analyze_speakers.py raw/ --plot
"""

import argparse
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
from resemblyzer import VoiceEncoder
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.metrics import silhouette_score
from tqdm import tqdm

AUDIO_EXTS  = {".opus", ".ogg", ".mp3", ".wav", ".flac", ".m4a"}
SAMPLE_RATE = 16_000
RNG_SEED    = 42
MIN_RMS     = 0.005   # skip near-silent windows


# ── Audio decoding ────────────────────────────────────────────────────────────

def decode_mono16k(path: Path) -> np.ndarray | None:
    """Decode entire file to 16 kHz mono float32 in one ffmpeg call."""
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"],
        capture_output=True,
    )
    if r.returncode != 0 or not r.stdout:
        return None
    return np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def extract_windows(
    audio: np.ndarray, n_windows: int, window_s: float, rng: random.Random
) -> list[np.ndarray]:
    """Return up to n_windows evenly-spaced non-silent windows."""
    win_len = int(window_s * SAMPLE_RATE)
    total   = len(audio)

    if total < win_len:
        return [audio] if np.sqrt(np.mean(audio ** 2)) > MIN_RMS else []

    # Evenly distribute window start positions across the file, with small jitter
    positions = np.linspace(0, total - win_len, n_windows, dtype=int)
    jitter = win_len // 10
    windows = []
    for pos in positions:
        start = int(np.clip(pos + rng.randint(-jitter, jitter), 0, total - win_len))
        chunk = audio[start : start + win_len]
        if np.sqrt(np.mean(chunk ** 2)) > MIN_RMS:
            windows.append(chunk)
    return windows


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_files(
    files: list[Path],
    encoder: VoiceEncoder,
    n_windows: int,
    window_s: float,
    rng: random.Random,
) -> tuple[np.ndarray, list[str]]:
    """
    Returns:
        embeddings  — (M, 256) array, one row per window
        window_labels — list of length M with the source filename for each window
    """
    all_embs: list[np.ndarray] = []
    all_labels: list[str] = []

    for f in tqdm(files, desc="Decoding + embedding", unit="file"):
        audio = decode_mono16k(f)
        if audio is None:
            tqdm.write(f"  skip (decode error): {f.name}")
            continue

        windows = extract_windows(audio, n_windows, window_s, rng)
        if not windows:
            tqdm.write(f"  skip (silent): {f.name}")
            continue

        for chunk in windows:
            try:
                emb = encoder.embed_utterance(chunk)
                all_embs.append(emb / (np.linalg.norm(emb) + 1e-8))
                all_labels.append(f.name)
            except Exception:
                pass

        tqdm.write(f"  {f.name}  [{len(windows)} windows]")

    if not all_embs:
        return np.empty((0, 256)), []
    return np.array(all_embs, dtype=np.float32), all_labels


# ── Clustering ────────────────────────────────────────────────────────────────

def ward_linkage(embeddings: np.ndarray):
    return linkage(embeddings, method="ward")


def largest_gap_n(Z: np.ndarray, min_n: int = 2, max_n: int = 40) -> tuple[int, list[tuple[int, float]]]:
    """
    Find best cluster count using the largest-gap heuristic on Ward merge distances.
    For k clusters: gap = Z[n-k, 2] - Z[n-k-1, 2]  (height gain at that cut).
    A large gap means k clusters are naturally well-separated.
    """
    n = len(Z) + 1
    gaps = []
    for k in range(min_n, min(max_n + 1, n)):
        gap = float(Z[n - k, 2] - Z[n - k - 1, 2])
        gaps.append((k, gap))
    if not gaps:
        return min_n, []
    ranked = sorted(gaps, key=lambda x: -x[1])
    best_k = ranked[0][0]
    return best_k, ranked


def silhouette_top(embeddings: np.ndarray, Z: np.ndarray, candidates: list[int]) -> list[tuple[int, float]]:
    """Compute silhouette scores for a limited set of candidate N values."""
    scores = []
    n = len(Z) + 1
    for k in tqdm(candidates, desc="Silhouette check", unit="n"):
        if k >= n:
            continue
        labels = fcluster(Z, k, criterion="maxclust") - 1
        try:
            scores.append((k, float(silhouette_score(embeddings, labels))))
        except Exception:
            pass
    return sorted(scores, key=lambda x: -x[1])


# ── Dendrogram ────────────────────────────────────────────────────────────────

def plot_dendrogram(Z: np.ndarray, truncate: int = 60):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skip (pip install matplotlib)")
        return

    fig, ax = plt.subplots(figsize=(16, 6))
    dendrogram(
        Z, ax=ax,
        truncate_mode="lastp", p=truncate,
        show_leaf_counts=True, no_labels=True,
        color_threshold=0,
    )
    ax.set_title(
        f"Ward dendrogram (last {truncate} merges) — "
        "the tallest vertical gaps indicate natural speaker boundaries"
    )
    ax.set_ylabel("Ward distance")
    plt.tight_layout()
    plt.show()


# ── Report ────────────────────────────────────────────────────────────────────

def report(
    embeddings: np.ndarray,
    Z: np.ndarray,
    window_labels: list[str],
    n: int,
):
    labels = fcluster(Z, n, criterion="maxclust") - 1  # 0-indexed

    # Per-cluster: count, file distribution, representative window
    from collections import Counter
    print(f"\n{'─' * 65}")
    print(f"  {n} distinct speaker(s)  |  {len(embeddings)} windows embedded")
    print(f"{'─' * 65}")

    for lbl in sorted(set(labels)):
        mask = labels == lbl
        cluster_embs   = embeddings[mask]
        cluster_files  = [window_labels[i] for i, m in enumerate(mask) if m]

        # Most common file in this cluster
        dominant_file = Counter(cluster_files).most_common(1)[0][0]
        n_files = len(set(cluster_files))

        # Window count and %
        count = int(mask.sum())
        pct   = count / len(labels) * 100

        print(f"  speaker {lbl:2d}  {count:4d} windows ({pct:5.1f}%)  across {n_files} file(s)")
        print(f"    dominant file: {dominant_file}")
    print(f"{'─' * 65}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Estimate distinct speaker count from raw unsegmented audio.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("audio_dir", type=Path, help="Directory of audio files")
    parser.add_argument(
        "--n-speakers", type=int, default=None, metavar="N",
        help="Force cluster count (skip auto-detection)",
    )
    parser.add_argument(
        "--windows", type=int, default=20, metavar="N",
        help="Windows sampled per file, evenly spaced (default: 20)",
    )
    parser.add_argument(
        "--window-s", type=float, default=10.0, metavar="SEC",
        help="Duration of each window in seconds (default: 10)",
    )
    parser.add_argument(
        "--max-gap-n", type=int, default=40, metavar="N",
        help="Max cluster count tested in largest-gap search (default: 40)",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Show Ward dendrogram after clustering",
    )
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    args = parser.parse_args()

    if not args.audio_dir.is_dir():
        print(f"Not a directory: {args.audio_dir}")
        sys.exit(1)

    files = sorted(p for p in args.audio_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
    if not files:
        print(f"No audio files found in {args.audio_dir}")
        sys.exit(1)

    print(f"Found {len(files):,} file(s)")
    print(f"Strategy: {args.windows} windows × {args.window_s}s per file, embedded individually")
    print(f"  → up to {len(files) * args.windows:,} embeddings, 1 ffmpeg decode per file\n")

    rng = random.Random(args.seed)

    print("Loading resemblyzer VoiceEncoder...")
    encoder = VoiceEncoder()

    embeddings, window_labels = embed_files(files, encoder, args.windows, args.window_s, rng)

    if len(embeddings) < 2:
        print("Not enough embeddings to cluster.")
        sys.exit(1)

    print(f"\n{len(embeddings)} total window embeddings ready")

    # ── Build Ward linkage once, reuse for everything ─────────────────────────
    print("Computing Ward linkage...")
    Z = ward_linkage(embeddings)

    if args.plot:
        plot_dendrogram(Z)

    # ── Determine N ───────────────────────────────────────────────────────────
    if args.n_speakers is not None:
        n = args.n_speakers
        print(f"\nForced cluster count: {n}")
    else:
        best_gap_n, gap_ranking = largest_gap_n(Z, max_n=args.max_gap_n)

        print("\nLargest-gap candidates (Ward distance jump — larger = more natural boundary):")
        for rank, (k, gap) in enumerate(gap_ranking[:10]):
            marker = " ← recommended" if rank == 0 else ""
            print(f"  n={k:3d}  gap={gap:.4f}{marker}")

        # Cross-check top-5 gap candidates with silhouette
        top_candidates = [k for k, _ in gap_ranking[:5]]
        sil_scores = silhouette_top(embeddings, Z, top_candidates)
        if sil_scores:
            print("\nSilhouette cross-check on top-5 candidates:")
            for k, s in sil_scores:
                print(f"  n={k:3d}  silhouette={s:.4f}")

        n = best_gap_n
        print(f"\nAuto-selected: {n} speaker(s)")
        print("Override with --n-speakers N if you know the real count.")

    report(embeddings, Z, window_labels, n)
    print("\nTip: use --plot to see the dendrogram and pick the cut visually.")


if __name__ == "__main__":
    main()
