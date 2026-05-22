#!/usr/bin/env python3
"""
Kokoro French Training Dataset Pipeline
========================================
Processes cached Polly MP3s into a clean French TTS training dataset.

Usage:
    # Step 0 (optional): Cut long recordings into short segments
    uv run python scripts/prepare_dataset.py segment --input-dir raw/

    # Step 1: Transcribe all MP3s with openai-whisper (resumable, run overnight)
    uv run python scripts/prepare_dataset.py transcribe

    # Step 2: Filter by language, duration, quality
    uv run python scripts/prepare_dataset.py filter

    # Step 3: Cluster speakers (find distinct Polly voices)
    uv run python scripts/prepare_dataset.py cluster

    # Step 4: Convert audio + generate IPA + write final dataset
    uv run python scripts/prepare_dataset.py format

    # Quick sanity check on a few files before committing to the full run
    uv run python scripts/prepare_dataset.py transcribe --sample 20

    # Print stats at any stage
    uv run python scripts/prepare_dataset.py stats
"""

import argparse
import json
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import whisper
from misaki import espeak
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────────

RAW_DIR = Path("./raw")   # input dir for long raw recordings (step 0)
CACHE_DIR = Path("./cache")
DATASET_DIR = Path("./dataset")
AUDIO_DIR = DATASET_DIR / "audio"
TRANSCRIPTIONS_FILE = DATASET_DIR / "transcriptions.jsonl"
FILTERED_FILE = DATASET_DIR / "filtered.jsonl"
SPEAKERS_FILE = DATASET_DIR / "speakers.jsonl"
METADATA_FILE = DATASET_DIR / "metadata.csv"
PHONEMES_FILE = DATASET_DIR / "phonemes.csv"
STATS_FILE = DATASET_DIR / "stats.json"

# ── Filtering thresholds ─────────────────────────────────────────────────────

MIN_DURATION_S = 2.0
MAX_DURATION_S = 30.0
MIN_AVG_LOGPROB = -1.0  # Whisper confidence: closer to 0 is better
MAX_NO_SPEECH_PROB = 0.5  # Reject segments that are likely silence/noise
MIN_WORDS = 3  # Minimum words in transcription
TARGET_LANGUAGE = "fr"  # ISO 639-1 French
MIN_SNR_DB = 22.0         # estimated SNR; below this → continuous background hum / bad mic

# ── Segmentation ─────────────────────────────────────────────────────────────

SILENCE_DB = -35.0        # dB below which audio is considered silence
SILENCE_MIN_S = 0.15      # shortest pause detected (soft boundary: breath, comma)
SEG_HARD_BOUNDARY_S = 0.4 # pauses this long are treated as inter-sentence boundaries
SEG_MIN_S = 2.0           # shortest output segment kept
SEG_MAX_S = 30.0          # hard ceiling for any output segment
SEG_TARGET_MIN_S = 5.0    # random target range lower bound (for length variety)
SEG_TARGET_MAX_S = 28.0   # random target range upper bound
SEG_TARGET_MODE_S = 18.0  # triangular distribution mode — peak probability (biases toward longer clips)

# ── Loudness normalisation (EBU R128) ────────────────────────────────────────

LOUDNORM_TARGET_I = -16.0   # integrated loudness target (LUFS)
LOUDNORM_TARGET_TP = -1.5   # true-peak ceiling (dBTP) — prevents clipping
LOUDNORM_TARGET_LRA = 11.0  # loudness range target (LU) — preserves dynamics

# ── Light denoising (applied before loudnorm when --no-denoise is not set) ───

DENOISE_HIGHPASS_HZ = 60    # cut sub-bass hum / 50–60 Hz electrical noise; 60 Hz preserves male vocal fundamentals (80–130 Hz)
DENOISE_AFFTDN_NF = -30     # afftdn noise floor estimate (dB) — lower = more conservative; -30 is gentle
# NOTE: afftdn estimates noise from the first frames of each clip. On pre-segmented audio that starts
# directly with speech, it mistakes voiced frames for noise and attenuates speech harmonics (musical
# noise). Only enable --denoise when the source has audible background hum (not for clean Polly output).

# ── Whisper model ─────────────────────────────────────────────────────────────

WHISPER_MODEL = "turbo"

# ── Speaker clustering ───────────────────────────────────────────────────────

N_SPEAKER_CLUSTERS = None  # None = auto-detect via DBSCAN


# ─────────────────────────────────────────────────────────────────────────────
# Step 0: Segment (cut long recordings into prosody-respecting short clips)
# ─────────────────────────────────────────────────────────────────────────────

_AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus"}


def cmd_segment(
    input_dir: Path,
    output_dir: Path,
    silence_db: float = SILENCE_DB,
    silence_min_s: float = SILENCE_MIN_S,
    rng_seed: int = 42,
    normalize: bool = True,
    denoise: bool = False,
):
    """Cut long audio files into prosody-respecting segments (varied 2–30 s)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(rng_seed)

    input_files = sorted(p for p in input_dir.glob("**/*") if p.suffix.lower() in _AUDIO_EXTS)
    if not input_files:
        print(f"No audio files found in {input_dir}")
        return

    print(f"Found {len(input_files)} file(s) in {input_dir}")
    print(f"Silence threshold: {silence_db:.0f} dB  |  min pause: {silence_min_s}s  |  sentence boundary: ≥{SEG_HARD_BOUNDARY_S}s")
    print(f"Segment range: {SEG_MIN_S}–{SEG_MAX_S}s  |  target variety: {SEG_TARGET_MIN_S}–{SEG_TARGET_MAX_S}s")
    if normalize:
        print(f"Loudness norm: EBU R128  I={LOUDNORM_TARGET_I} LUFS  TP={LOUDNORM_TARGET_TP} dBTP  LRA={LOUDNORM_TARGET_LRA} LU")
    if normalize and denoise:
        print(f"Denoising: highpass={DENOISE_HIGHPASS_HZ} Hz  +  afftdn nf={DENOISE_AFFTDN_NF} dB")

    total_segs = 0
    total_dur = 0.0
    skipped_files = 0
    total_hard_cuts = 0

    for src in tqdm(input_files, desc="Segmenting", unit="file"):
        duration = _get_duration(src)
        if duration < SEG_MIN_S:
            tqdm.write(f"  Skip {src.name}: too short ({duration:.1f}s)")
            skipped_files += 1
            continue

        silences = _detect_silences(src, silence_db, silence_min_s)
        cut_points, hard_cuts = _compute_cut_points(silences, duration, rng)
        total_hard_cuts += hard_cuts

        stem = src.stem
        seg_start = 0.0
        for idx, seg_end in enumerate(cut_points):
            seg_dur = seg_end - seg_start
            if seg_dur < SEG_MIN_S:
                seg_start = seg_end
                continue
            out_path = output_dir / f"{stem}_{idx:04d}.flac"
            if not out_path.exists():
                _cut_segment(src, seg_start, seg_end, out_path, normalize=normalize, denoise=denoise)
            total_segs += 1
            total_dur += seg_dur
            seg_start = seg_end

        # Trailing remainder
        remainder = duration - seg_start
        if remainder >= SEG_MIN_S:
            out_path = output_dir / f"{stem}_{len(cut_points):04d}.flac"
            if not out_path.exists():
                _cut_segment(src, seg_start, duration, out_path, normalize=normalize, denoise=denoise)
            total_segs += 1
            total_dur += remainder

    print(f"\nSegments created : {total_segs:,}")
    print(f"Total duration   : {total_dur / 60:.1f} min  ({total_dur / 3600:.2f}h)")
    if skipped_files:
        print(f"Files skipped    : {skipped_files}")
    if total_hard_cuts:
        pct = total_hard_cuts / max(total_segs, 1) * 100
        print(f"Hard cuts (no silence found) : {total_hard_cuts} ({pct:.1f}%) — consider lowering --silence-db or --silence-min")
    print(f"Output dir       : {output_dir}")


def _detect_silences(path: Path, noise_db: float, min_dur: float) -> list[tuple[float, float]]:
    """Return (silence_start, silence_end) pairs via ffmpeg silencedetect."""
    result = subprocess.run(
        [
            "ffmpeg", "-i", str(path),
            "-af", f"silencedetect=noise={noise_db:.0f}dB:d={min_dur}",
            "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
    )
    output = result.stderr  # ffmpeg writes filter output to stderr
    starts = [float(m) for m in re.findall(r"silence_start: ([0-9.e+-]+)", output)]
    ends   = [float(m) for m in re.findall(r"silence_end: ([0-9.e+-]+)", output)]
    return list(zip(starts, ends))


def _compute_cut_points(
    silences: list[tuple[float, float]],
    total_duration: float,
    rng: random.Random,
) -> tuple[list[float], int]:
    """Greedy segmentation using silence midpoints as prosodic cut candidates.

    Silences are split into two tiers:
    - Hard boundaries (≥ SEG_HARD_BOUNDARY_S): likely inter-sentence pauses, preferred.
    - Soft boundaries (< SEG_HARD_BOUNDARY_S): breaths, commas — used as fallback.

    Each segment gets a randomly sampled target length for natural length variety.
    """
    hard = sorted((s + e) / 2.0 for s, e in silences if (e - s) >= SEG_HARD_BOUNDARY_S)
    soft = sorted((s + e) / 2.0 for s, e in silences if (e - s) < SEG_HARD_BOUNDARY_S)

    cut_points: list[float] = []
    hard_cuts = 0
    seg_start = 0.0

    while seg_start < total_duration - SEG_MIN_S:
        # Triangular distribution: allows short clips but peaks at SEG_TARGET_MODE_S
        target_end = seg_start + rng.triangular(SEG_TARGET_MIN_S, SEG_TARGET_MAX_S, SEG_TARGET_MODE_S)

        window_lo = seg_start + SEG_MIN_S
        window_hi = min(seg_start + SEG_MAX_S, total_duration)

        valid_hard = [c for c in hard if window_lo <= c <= window_hi]
        valid_soft = [c for c in soft if window_lo <= c <= window_hi]

        if valid_hard:
            cut = min(valid_hard, key=lambda c: abs(c - target_end))
        elif valid_soft:
            cut = min(valid_soft, key=lambda c: abs(c - target_end))
        else:
            cut = window_hi
            hard_cuts += 1

        cut_points.append(cut)
        seg_start = cut

    return cut_points, hard_cuts


def _cut_segment(
    src: Path,
    start: float,
    end: float,
    dst: Path,
    normalize: bool = True,
    denoise: bool = False,
):
    """Extract [start, end] seconds from src into dst (FLAC when dst.suffix == .flac).

    With normalize=True applies EBU R128 loudnorm, optionally preceded by light denoising:
      highpass=f=60  →  removes sub-bass hum / 50–60 Hz electrical noise
      afftdn=nf=-30  →  conservative adaptive FFT denoising (only with denoise=True)
      loudnorm       →  EBU R128 integrated loudness normalisation (linear gain, no DRC)

    With denoise=False (default), only loudnorm is applied — safe for clean Polly output.
    With normalize=False, stream copy is used (fastest, no audio modification).
    """
    if normalize:
        loudnorm = (
            f"loudnorm=I={LOUDNORM_TARGET_I}:TP={LOUDNORM_TARGET_TP}"
            f":LRA={LOUDNORM_TARGET_LRA}:linear=true:print_format=none"
        )
        if denoise:
            af = (
                f"highpass=f={DENOISE_HIGHPASS_HZ},"
                f"afftdn=nf={DENOISE_AFFTDN_NF},"
                + loudnorm
            )
        else:
            af = loudnorm
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-to", f"{end:.3f}",
            "-i", str(src),
            "-af", af,
            str(dst),
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(src),
            "-ss", f"{start:.3f}",
            "-to", f"{end:.3f}",
            "-c", "copy",
            str(dst),
        ]
    subprocess.run(cmd, capture_output=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Transcribe
# ─────────────────────────────────────────────────────────────────────────────


def cmd_transcribe(sample: int | None, model_name: str = WHISPER_MODEL, device: str = "auto"):
    """Transcribe all MP3s using openai-whisper. Resumable."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # Load already-processed hashes for resumability
    done = set()
    if TRANSCRIPTIONS_FILE.exists():
        with open(TRANSCRIPTIONS_FILE) as f:
            for line in f:
                entry = json.loads(line)
                done.add(entry["hash"])

    # Collect all audio files (FLAC from segment step, or raw MP3 from Polly cache)
    all_files = sorted(p for p in CACHE_DIR.glob("**/*") if p.suffix.lower() in _AUDIO_EXTS)
    if sample:
        all_files = all_files[:sample]

    pending = [f for f in all_files if f.stem not in done]
    print(
        f"Total MP3s: {len(all_files)}  |  Already done: {len(done)}  |  Pending: {len(pending)}"
    )

    if not pending:
        print("Nothing to do.")
        return

    print(f"Loading model: {model_name} | device: {device}")
    model = whisper.load_model(model_name, device=device)

    errors = 0
    with open(TRANSCRIPTIONS_FILE, "a") as out:
        for mp3 in tqdm(pending, unit="file", desc="Transcribing"):
            entry = _transcribe_file(model, mp3)
            if entry is None:
                errors += 1
            else:
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                out.flush()

    print(f"\nDone. Errors: {errors}")
    _print_transcription_stats()


def _transcribe_file(model, mp3: Path) -> dict | None:
    try:
        result = model.transcribe(
            str(mp3),
            language=None,
            task="transcribe",
            word_timestamps=False,
            fp16=True,
            condition_on_previous_text=False,
        )
        segments = result.get("segments", [])
        avg_logprob = (
            sum(s["avg_logprob"] for s in segments) / len(segments)
            if segments
            else -9.0
        )
        no_speech_prob = (
            sum(s["no_speech_prob"] for s in segments) / len(segments)
            if segments
            else 1.0
        )
        return {
            "hash": mp3.stem,
            "path": str(mp3),
            "duration": round(_get_duration(mp3), 3),
            "language": result.get("language", "unknown"),
            "text": result.get("text", "").strip(),
            "avg_logprob": round(avg_logprob, 4),
            "no_speech_prob": round(no_speech_prob, 4),
            "n_segments": len(segments),
        }
    except Exception as e:
        tqdm.write(f"ERROR {mp3.name}: {e}")
        return None


def _estimate_snr_db(path: Path, frame_ms: int = 20) -> float:
    """Estimate SNR via 10th-/90th-percentile frame-RMS ratio.

    Decodes audio to raw 16 kHz mono PCM, splits into short frames, then
    computes SNR = 20 * log10(speech_rms / noise_rms) where:
      - noise_rms   = 10th-percentile frame RMS  (the persistent noise floor)
      - speech_rms  = 90th-percentile frame RMS  (the voiced speech level)

    A clean recording has SNR > 25 dB. A recording with continuous background
    hum or a poorly-set mic gain typically falls below 20 dB.

    Returns -inf on decode failure, +inf for clips too short to assess.
    """
    result = subprocess.run(
        ["ffmpeg", "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
        capture_output=True,
    )
    if result.returncode != 0 or not result.stdout:
        return float("-inf")

    samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    frame_len = 16000 * frame_ms // 1000
    n_frames = len(samples) // frame_len
    if n_frames < 2:
        return float("inf")

    frames = samples[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    rms = rms[rms > 0]
    if len(rms) < 2:
        return float("inf")

    noise_floor = np.percentile(rms, 10)
    speech_level = np.percentile(rms, 90)
    if noise_floor <= 0:
        return float("inf")

    return float(20.0 * np.log10(speech_level / noise_floor))


def _get_duration(path: Path) -> float:
    """Fast duration extraction via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def _print_transcription_stats():
    if not TRANSCRIPTIONS_FILE.exists():
        return
    total = done_fr = done_en = done_other = 0
    total_duration = 0.0
    with open(TRANSCRIPTIONS_FILE) as f:
        for line in f:
            e = json.loads(line)
            total += 1
            total_duration += e.get("duration", 0)
            lang = e.get("language", "")
            if lang == "fr":
                done_fr += 1
            elif lang == "en":
                done_en += 1
            else:
                done_other += 1
    print("\nTranscription stats:")
    print(f"  Total files   : {total:,}")
    print(f"  Total duration: {total_duration / 3600:.1f}h")
    print(f"  French (fr)   : {done_fr:,}  ({done_fr / total * 100:.1f}%)")
    print(f"  English (en)  : {done_en:,}  ({done_en / total * 100:.1f}%)")
    print(f"  Other         : {done_other:,}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Filter
# ─────────────────────────────────────────────────────────────────────────────


def cmd_filter(check_snr: bool = False):
    """Filter transcriptions by language, duration, and quality."""
    if not TRANSCRIPTIONS_FILE.exists():
        print("No transcriptions.jsonl found. Run: transcribe first.")
        sys.exit(1)

    entries = []
    with open(TRANSCRIPTIONS_FILE) as f:
        for line in f:
            entries.append(json.loads(line))

    print(f"Input: {len(entries):,} entries")
    if not check_snr:
        print("SNR check skipped (use --snr to enable)")

    reasons = {
        "wrong_language": 0,
        "too_short": 0,
        "too_long": 0,
        "low_confidence": 0,
        "high_no_speech": 0,
        "too_few_words": 0,
        "low_snr": 0,
    }
    kept = []

    desc = "Filtering"
    if check_snr:
        print("SNR check reads audio files — may take a while...")
    for e in tqdm(entries, desc=desc, unit="file"):
        if e.get("language") != TARGET_LANGUAGE:
            reasons["wrong_language"] += 1
            continue
        dur = e.get("duration", 0)
        if dur < MIN_DURATION_S:
            reasons["too_short"] += 1
            continue
        if dur > MAX_DURATION_S:
            reasons["too_long"] += 1
            continue
        if e.get("avg_logprob", -9.0) < MIN_AVG_LOGPROB:
            reasons["low_confidence"] += 1
            continue
        if e.get("no_speech_prob", 1.0) > MAX_NO_SPEECH_PROB:
            reasons["high_no_speech"] += 1
            continue
        text = e.get("text", "")
        if len(text.split()) < MIN_WORDS:
            reasons["too_few_words"] += 1
            continue
        if check_snr:
            snr = _estimate_snr_db(Path(e["path"]))
            if snr < MIN_SNR_DB:
                reasons["low_snr"] += 1
                continue
        kept.append(e)

    total_duration = sum(e.get("duration", 0) for e in kept)
    print(f"\nKept: {len(kept):,}  ({total_duration / 3600:.1f}h of audio)")
    print("Dropped:")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        if count:
            print(f"  {reason:<20}: {count:,}")

    with open(FILTERED_FILE, "w") as f:
        for e in kept:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"\nWrote {FILTERED_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Cluster speakers
# ─────────────────────────────────────────────────────────────────────────────


def cmd_cluster(n_speakers=None, min_speakers=2, distance_threshold=None):
    """Extract speaker embeddings and cluster into distinct Polly voices."""
    if not FILTERED_FILE.exists():
        print("No filtered.jsonl found. Run: filter first.")
        sys.exit(1)

    entries = []
    with open(FILTERED_FILE) as f:
        for line in f:
            entries.append(json.loads(line))

    print(f"Loading {len(entries):,} filtered entries for speaker clustering...")

    MAX_EMBED = 3000
    rng = random.Random(42)
    sample = rng.sample(entries, min(MAX_EMBED, len(entries)))

    print(f"Extracting speaker embeddings from {len(sample):,} files...")
    print("Loading resemblyzer VoiceEncoder...")

    try:
        encoder = VoiceEncoder()
    except Exception as e:
        print(f"Failed to load resemblyzer: {e}")
        print('Falling back to single-speaker labeling (all files -> "d_speaker0")')
        _write_speakers_single(entries)
        return

    embeddings = []
    valid_entries = []
    errors = 0

    for entry in tqdm(sample, desc="Embedding"):
        try:
            wav = preprocess_wav(entry["path"])
            emb = encoder.embed_utterance(wav)
            embeddings.append(emb)
            valid_entries.append(entry)
        except Exception:
            errors += 1

    if errors:
        print(f"Embedding errors: {errors}")

    if not embeddings:
        print("No embeddings extracted. Falling back to single-speaker.")
        _write_speakers_single(entries)
        return

    embeddings = np.array(embeddings, dtype=np.float32)

    print(f"Clustering {len(embeddings)} embeddings...")
    n_speakers, labels = _cluster_embeddings(embeddings, n_speakers=n_speakers, min_speakers=min_speakers, distance_threshold=distance_threshold)
    print(f"Detected {n_speakers} distinct speaker(s)")

    path_to_speaker: dict[str, str] = {}
    for entry, label in zip(valid_entries, labels):
        path_to_speaker[entry["path"]] = f"d_speaker{label}"

    centroids = {}
    for label in range(n_speakers):
        mask = labels == label
        centroids[label] = embeddings[mask].mean(axis=0)

    print(f"Assigning speakers to all {len(entries):,} files...")
    print("(For unsampled files, we embed and find nearest centroid)")

    unsampled = [e for e in entries if e["path"] not in path_to_speaker]
    centroid_matrix = np.array([centroids[i] for i in range(n_speakers)])

    for entry in tqdm(unsampled, desc="Assigning"):
        try:
            wav = preprocess_wav(entry["path"])
            emb = encoder.embed_utterance(wav)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            dists = np.dot(centroid_matrix, emb)
            best = int(np.argmax(dists))
            path_to_speaker[entry["path"]] = f"d_speaker{best}"
        except Exception:
            path_to_speaker[entry["path"]] = "d_speaker0"

    with open(SPEAKERS_FILE, "w") as f:
        for entry in entries:
            entry["speaker"] = path_to_speaker.get(entry["path"], "d_speaker0")
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    speaker_counts = Counter(path_to_speaker.values())

    # Pick one representative sample per speaker (closest to centroid)
    speaker_sample: dict[str, str] = {}
    for label in range(n_speakers):
        spk = f"d_speaker{label}"
        mask = labels == label
        if not mask.any():
            continue
        centroid = centroids[label]
        spk_embs = embeddings[mask]
        spk_entries = [e for e, lbl in zip(valid_entries, labels) if lbl == label]
        norms = spk_embs / (np.linalg.norm(spk_embs, axis=1, keepdims=True) + 1e-8)
        c_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
        best_idx = int(np.argmax(norms @ c_norm))
        speaker_sample[spk] = spk_entries[best_idx]["path"]

    print("\nSpeaker distribution:")
    for spk, count in sorted(speaker_counts.items()):
        dur = sum(
            e["duration"] for e in entries if path_to_speaker.get(e["path"]) == spk
        )
        sample_path = speaker_sample.get(spk, "n/a")
        print(f"  {spk}: {count:,} files  ({dur / 3600:.1f}h)")
        print(f"    sample → {sample_path}")

    print(f"\nWrote {SPEAKERS_FILE}")
    print("\nNext: listen to each sample above to verify gender,")
    print("then re-run format with --rename-speakers, e.g.:")
    rename_example = "  python prepare_dataset.py format --rename-speakers " + " ".join(
        f"{spk}=df_name" if i == 0 else f"{spk}=dm_name"
        for i, spk in enumerate(sorted(speaker_counts))
    )
    print(rename_example)


def _cluster_embeddings(embeddings, n_speakers=None, min_speakers=2, distance_threshold=None):
    """Auto-cluster speaker embeddings. Returns (n_speakers, labels array).

    Auto-detection uses Ward linkage + largest-gap heuristic:
    for each candidate k, gap = Z[n-k, 2] - Z[n-k-1, 2] measures how much
    Ward distance is gained by merging the k-th cluster pair. A large gap
    means k clusters are naturally well-separated in embedding space.
    This is more reliable than silhouette for Polly voices, which can have
    uneven cluster sizes and subtle inter-voice similarity.
    """
    if n_speakers is not None:
        clustering = AgglomerativeClustering(n_clusters=n_speakers)
        labels = clustering.fit_predict(embeddings)
        print(f"Using forced cluster count: {n_speakers}")
        return n_speakers, labels

    if distance_threshold is not None:
        clustering = AgglomerativeClustering(
            n_clusters=None, distance_threshold=distance_threshold, linkage="ward"
        )
        labels = clustering.fit_predict(embeddings)
        n = len(set(labels))
        print(f"Distance threshold {distance_threshold} → {n} cluster(s)")
        return n, labels

    Z = linkage(embeddings, method="ward")
    n_pts = len(embeddings)
    max_search = min(50, n_pts - 1)
    start = max(2, min_speakers)

    if max_search < start:
        return 1, np.zeros(n_pts, dtype=int)

    gaps = [
        (k, float(Z[n_pts - k, 2] - Z[n_pts - k - 1, 2]))
        for k in range(start, max_search + 1)
    ]
    ranked = sorted(gaps, key=lambda x: -x[1])
    best_n = ranked[0][0]

    print("Top cluster candidates (Ward gap — larger = more natural boundary):")
    for k, gap in ranked[:10]:
        marker = " ← recommended" if k == best_n else ""
        print(f"  n={k:3d}  gap={gap:.4f}{marker}")
    print("Tip: use --n-speakers to override if the result looks wrong.")

    labels = fcluster(Z, best_n, criterion="maxclust") - 1  # 0-indexed
    return best_n, labels


def _write_speakers_single(entries):
    """Fallback: label all entries as a single speaker."""
    with open(SPEAKERS_FILE, "w") as f:
        for entry in entries:
            entry["speaker"] = "d_speaker0"
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Wrote {SPEAKERS_FILE} (single speaker fallback)")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3b: Drop speakers
# ─────────────────────────────────────────────────────────────────────────────


def cmd_drop(speakers_to_drop: list[str]):
    """Remove entries for specific speaker IDs from speakers.jsonl."""
    if not SPEAKERS_FILE.exists():
        print("No speakers.jsonl found. Run: cluster first.")
        sys.exit(1)

    entries = []
    with open(SPEAKERS_FILE) as f:
        for line in f:
            entries.append(json.loads(line))

    print(f"Input: {len(entries):,} entries")
    print(f"Dropping speakers: {speakers_to_drop}")

    kept = [e for e in entries if e.get("speaker") not in speakers_to_drop]
    dropped = len(entries) - len(kept)
    dropped_duration = sum(
        e["duration"] for e in entries if e.get("speaker") in speakers_to_drop
    )
    kept_duration = sum(e["duration"] for e in kept)

    print(f"Dropped: {dropped:,} files  ({dropped_duration / 3600:.1f}h)")
    print(f"Kept   : {len(kept):,} files  ({kept_duration / 3600:.1f}h)")

    with open(SPEAKERS_FILE, "w") as f:
        for e in kept:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"\nUpdated {SPEAKERS_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Format (convert audio + IPA + write final dataset)
# ─────────────────────────────────────────────────────────────────────────────


def cmd_format(rename_speakers: list[str] | None):
    """Convert MP3→WAV, generate IPA phonemes, write final dataset."""
    if not SPEAKERS_FILE.exists():
        print("No speakers.jsonl found. Run: cluster first.")
        sys.exit(1)

    entries = []
    with open(SPEAKERS_FILE) as f:
        for line in f:
            entries.append(json.loads(line))

    rename_map = {}
    if rename_speakers:
        for pair in rename_speakers:
            old, new = pair.split("=", 1)
            rename_map[old.strip()] = new.strip()
    if rename_map:
        print(f"Applying speaker renames: {rename_map}")
        for e in entries:
            e["speaker"] = rename_map.get(e["speaker"], e["speaker"])

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    speakers = sorted(set(e["speaker"] for e in entries))
    for spk in speakers:
        (AUDIO_DIR / spk).mkdir(parents=True, exist_ok=True)

    print(f"Converting {len(entries):,} MP3s to 24kHz mono WAV...")
    errors = 0
    skipped = 0
    converted = 0

    for entry in tqdm(entries, desc="Converting"):
        spk = entry["speaker"]
        wav_path = AUDIO_DIR / spk / f"{entry['hash']}.wav"

        if wav_path.exists():
            skipped += 1
            entry["wav_path"] = str(wav_path)
            continue

        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", entry["path"],
                "-af", "aresample=resampler=soxr:precision=28",
                "-ac", "1",
                "-ar", "24000",
                "-sample_fmt", "s16",
                str(wav_path),
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            errors += 1
            continue

        entry["wav_path"] = str(wav_path)
        converted += 1

    print(f"Converted: {converted}  Skipped (exists): {skipped}  Errors: {errors}")

    print("Generating IPA phonemes via misaki (espeak-ng French G2P)...")
    g2p = espeak.EspeakG2P(language="fr-fr")

    PHONEME_FIXUPS: dict[str, str] = {}

    metadata_rows = []
    phoneme_rows = []
    ipa_errors = 0

    for entry in tqdm(entries, desc="G2P"):
        if "wav_path" not in entry:
            continue
        text = entry["text"]
        spk = entry["speaker"]
        wav_name = f"{entry['speaker']}/{entry['hash']}.wav"

        try:
            phonemes, _ = g2p(text)
            for old, new in PHONEME_FIXUPS.items():
                phonemes = phonemes.replace(old, new)
        except Exception:
            ipa_errors += 1
            phonemes = ""

        metadata_rows.append(f"{wav_name}|{text}|{spk}")
        phoneme_rows.append(f"{wav_name}|{phonemes}")

    if ipa_errors:
        print(f"IPA generation errors: {ipa_errors}")

    with open(METADATA_FILE, "w") as f:
        f.write("filename|text|speaker\n")
        f.write("\n".join(metadata_rows) + "\n")

    with open(PHONEMES_FILE, "w") as f:
        f.write("filename|ipa\n")
        f.write("\n".join(phoneme_rows) + "\n")

    total_duration = sum(e["duration"] for e in entries if "wav_path" in e)
    speaker_stats = {}
    for e in entries:
        if "wav_path" not in e:
            continue
        spk = e["speaker"]
        if spk not in speaker_stats:
            speaker_stats[spk] = {"files": 0, "duration_s": 0.0}
        speaker_stats[spk]["files"] += 1
        speaker_stats[spk]["duration_s"] += e["duration"]

    stats = {
        "total_files": len(metadata_rows),
        "total_duration_h": round(total_duration / 3600, 2),
        "speakers": {
            spk: {
                "files": v["files"],
                "duration_h": round(v["duration_s"] / 3600, 2),
            }
            for spk, v in sorted(speaker_stats.items())
        },
    }
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    print("\nDataset ready:")
    print(f"  Files   : {stats['total_files']:,}")
    print(f"  Duration: {stats['total_duration_h']}h")
    print(f"  Speakers: {len(speaker_stats)}")
    print(f"  metadata.csv  -> {METADATA_FILE}")
    print(f"  phonemes.csv  -> {PHONEMES_FILE}")
    print(f"  stats.json    -> {STATS_FILE}")
    print(f"  audio/        -> {AUDIO_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────────────────────────────────────


def cmd_stats():
    """Print statistics for each stage that has been completed."""
    if TRANSCRIPTIONS_FILE.exists():
        print("=== Transcriptions ===")
        _print_transcription_stats()

    if FILTERED_FILE.exists():
        with open(FILTERED_FILE) as f:
            entries = [json.loads(line) for line in f]
        dur = sum(e.get("duration", 0) for e in entries)
        print("\n=== Filtered ===")
        print(f"  Files   : {len(entries):,}")
        print(f"  Duration: {dur / 3600:.1f}h")

    if SPEAKERS_FILE.exists():
        with open(SPEAKERS_FILE) as f:
            entries = [json.loads(line) for line in f]
        counts = Counter(e.get("speaker", "?") for e in entries)
        print("\n=== Speakers ===")
        for spk, cnt in sorted(counts.items()):
            dur = sum(e["duration"] for e in entries if e.get("speaker") == spk)
            print(f"  {spk}: {cnt:,} files  ({dur / 3600:.1f}h)")

    if STATS_FILE.exists():
        with open(STATS_FILE) as f:
            stats = json.load(f)
        print("\n=== Final Dataset ===")
        print(f"  Files   : {stats['total_files']:,}")
        print(f"  Duration: {stats['total_duration_h']}h")
        for spk, v in stats.get("speakers", {}).items():
            print(f"  {spk}: {v['files']:,} files  ({v['duration_h']}h)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Kokoro French TTS training dataset pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # segment
    p_segment = subparsers.add_parser(
        "segment",
        help="Cut long recordings into prosody-respecting segments (step 0, before transcribe)",
    )
    p_segment.add_argument(
        "--input-dir",
        type=Path,
        default=RAW_DIR,
        metavar="DIR",
        help=f"Directory of long source audio files (default: {RAW_DIR})",
    )
    p_segment.add_argument(
        "--output-dir",
        type=Path,
        default=CACHE_DIR,
        metavar="DIR",
        help=f"Where to write short segments (default: {CACHE_DIR})",
    )
    p_segment.add_argument(
        "--silence-db",
        type=float,
        default=SILENCE_DB,
        metavar="DB",
        help=f"Silence threshold in dB (default: {SILENCE_DB})",
    )
    p_segment.add_argument(
        "--silence-min",
        type=float,
        default=SILENCE_MIN_S,
        metavar="SEC",
        help=f"Minimum pause duration to treat as a boundary (default: {SILENCE_MIN_S}s)",
    )
    p_segment.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for target-length sampling (default: 42)",
    )
    p_segment.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip EBU R128 loudness normalization (uses stream copy, faster but no volume correction)",
    )
    p_segment.add_argument(
        "--denoise",
        action="store_true",
        help="Enable light denoising (highpass + afftdn); only applies when normalizing",
    )

    # transcribe
    p_transcribe = subparsers.add_parser(
        "transcribe", help="Transcribe MP3s with openai-whisper"
    )
    p_transcribe.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Only process first N files (for testing)",
    )
    p_transcribe.add_argument(
        "--model",
        default=WHISPER_MODEL,
        help=f"Whisper model name (default: {WHISPER_MODEL})",
    )
    p_transcribe.add_argument(
        "--device",
        default="auto",
        help="Device for Whisper inference: auto, cpu, cuda, cuda:0, mps, … (default: auto)",
    )

    # filter
    p_filter = subparsers.add_parser("filter", help="Filter by language, duration, quality")
    p_filter.add_argument(
        "--snr",
        action="store_true",
        help="Enable SNR background-noise check (reads all audio files — slow)",
    )

    # cluster
    p_cluster = subparsers.add_parser(
        "cluster", help="Cluster speakers using ECAPA-TDNN embeddings"
    )
    p_cluster.add_argument(
        "--n-speakers", type=int, default=None, metavar="N",
        help="Force exact number of speaker clusters (skips auto-detection)"
    )
    p_cluster.add_argument(
        "--min-speakers", type=int, default=2, metavar="N",
        help="Minimum number of clusters to consider during auto-detection (default: 2)"
    )
    p_cluster.add_argument(
        "--distance-threshold", type=float, default=None, metavar="D",
        help="Ward linkage distance cutoff — lower = more clusters (e.g. 0.4–0.8). Skips silhouette search."
    )

    # drop
    p_drop = subparsers.add_parser(
        "drop", help="Remove entries for specific speaker IDs from speakers.jsonl"
    )
    p_drop.add_argument(
        "speakers",
        nargs="+",
        metavar="SPEAKER_ID",
        help="Speaker IDs to drop (e.g. d_speaker0)",
    )

    # format
    p_format = subparsers.add_parser(
        "format", help="Convert audio, generate IPA, write final dataset"
    )
    p_format.add_argument(
        "--rename-speakers",
        nargs="+",
        metavar="OLD=NEW",
        help="Rename speaker IDs (e.g. d_speaker0=df_camille d_speaker1=dm_pierre)",
    )

    # stats
    subparsers.add_parser("stats", help="Print statistics for completed stages")

    args = parser.parse_args()

    if args.command == "segment":
        cmd_segment(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            silence_db=args.silence_db,
            silence_min_s=args.silence_min,
            rng_seed=args.seed,
            normalize=not args.no_normalize,
            denoise=args.denoise,
        )
    elif args.command == "transcribe":
        cmd_transcribe(sample=args.sample, model_name=args.model, device=args.device)
    elif args.command == "filter":
        cmd_filter(check_snr=args.snr)
    elif args.command == "cluster":
        cmd_cluster(n_speakers=args.n_speakers, min_speakers=args.min_speakers, distance_threshold=args.distance_threshold)
    elif args.command == "drop":
        cmd_drop(speakers_to_drop=args.speakers)
    elif args.command == "format":
        cmd_format(rename_speakers=args.rename_speakers)
    elif args.command == "stats":
        cmd_stats()


if __name__ == "__main__":
    main()
