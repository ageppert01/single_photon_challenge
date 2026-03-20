"""
Preprocess the full Single Photon Challenge dataset.

Downloads compressed chunks from the UW-Madison S3 bucket one at a time,
preprocesses each photoncube into a measurement PNG using the naive-sum
pipeline (average, response inversion, sRGB tonemap), copies the
ground-truth target PNG alongside it, then deletes the raw chunk.

Requirements:
    - aws-cli  (https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html)
    - 7z       (apt install p7zip-full)

Usage:
    python scripts/preprocess_full_dataset.py --output-dir ./preprocessed
    python scripts/preprocess_full_dataset.py --output-dir ./preprocessed --split test
    python scripts/preprocess_full_dataset.py --output-dir ./preprocessed --num-frames 64

Output structure:
    output_dir/
      metadata.json          # preprocessing parameters
      train/
        <scene>/<frame>_measurement.png
        <scene>/<frame>_target.png
      test/
        <scene>/<frame>_measurement.png
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Allow imports of photoncube_preprocess:
#   - In HTCondor sandbox: all transferred files are in the working directory
#   - Running locally:     photoncube_preprocess.py lives in ../diffusion/
sys.path.insert(0, ".")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffusion"))

import numpy as np
from PIL import Image

from photoncube_preprocess import (
    load_photoncube,
    naive_sum_preprocess,
    spc_avg_to_rgb,
    linearrgb_to_srgb,
)


S3_BUCKET = "s3://public-datasets/challenges/reconstruction"
S3_ENDPOINT = "https://web.s3.wisc.edu"


def run_cmd(cmd: list[str], description: str) -> None:
    print(f"  {description}")
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def list_s3_chunks(split: str | None = None) -> list[str]:
    """List zip files available in the S3 bucket."""
    cmd = [
        "aws", "s3", "ls",
        "--recursive", S3_BUCKET,
        "--endpoint-url", S3_ENDPOINT,
        "--no-sign-request",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        raise RuntimeError("Failed to list S3 bucket contents.")

    chunks = []
    for line in result.stdout.strip().split("\n"):
        parts = line.split()
        if len(parts) >= 4 and parts[-1].endswith(".zip"):
            key = parts[-1]
            if split is None:
                chunks.append(key)
            elif key.startswith(f"{split}"):
                chunks.append(key)
            # Also handle paths like challenges/reconstruction/train_001.zip
            elif f"/{split}" in key:
                chunks.append(key)

    return sorted(chunks)


def download_chunk(s3_key: str, download_dir: Path, max_retries: int = 5) -> Path:
    """Download a single chunk from S3, with retries for transient failures."""
    local_path = download_dir / Path(s3_key).name
    cmd = [
        "aws", "s3", "cp",
        f"{S3_BUCKET}/{Path(s3_key).name}" if "/" not in s3_key else f"s3://public-datasets/{s3_key}",
        str(local_path),
        "--endpoint-url", S3_ENDPOINT,
        "--no-sign-request",
        "--cli-connect-timeout", "120",
        "--cli-read-timeout", "120",
    ]

    for attempt in range(1, max_retries + 1):
        print(f"  Downloading {s3_key} (attempt {attempt}/{max_retries})")
        print(f"  $ {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return local_path

        print(f"  STDERR: {result.stderr.strip()}")
        if attempt < max_retries:
            wait = 30 * attempt  # 30s, 60s, 90s, 120s, 150s
            print(f"  Retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError(
        f"Download failed after {max_retries} attempts: {' '.join(cmd)}"
    )


def extract_chunk(zip_path: Path, extract_dir: Path) -> None:
    """Extract a zip file using 7z (handles LZMA compression)."""
    cmd = ["7z", "x", str(zip_path), f"-o{extract_dir}", "-y"]
    run_cmd(cmd, f"Extracting {zip_path.name}")


def preprocess_photoncube_to_png(
    npy_path: Path,
    output_path: Path,
    num_frames: int,
    invert_response: bool,
    invert_factor: float,
    tonemap: bool,
) -> None:
    """Preprocess a single photoncube .npy file and save as uint8 PNG."""
    photoncube = load_photoncube(str(npy_path))
    image = naive_sum_preprocess(photoncube, num_frames=num_frames)

    if invert_response:
        image = spc_avg_to_rgb(image, factor=invert_factor)

    if tonemap:
        image = linearrgb_to_srgb(image)

    image = np.clip(image, 0.0, 1.0)
    image_uint8 = (image * 255).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_uint8).save(output_path)


def process_extracted_chunk(
    extract_dir: Path,
    output_dir: Path,
    num_frames: int,
    invert_response: bool,
    invert_factor: float,
    tonemap: bool,
) -> int:
    """Process all photoncube/target pairs in an extracted chunk directory.

    Returns the number of samples processed.
    """
    count = 0
    npy_files = sorted(extract_dir.rglob("*.npy"))

    for npy_path in npy_files:
        # Preserve the relative path structure (e.g. train/scene/frame)
        rel_path = npy_path.relative_to(extract_dir)
        stem = rel_path.with_suffix("")

        # Save preprocessed measurement
        measurement_out = output_dir / f"{stem}_measurement.png"
        try:
            preprocess_photoncube_to_png(
                npy_path, measurement_out,
                num_frames, invert_response, invert_factor, tonemap,
            )
        except Exception as e:
            print(f"  WARNING: Failed to process {npy_path}: {e}")
            continue

        # Copy ground-truth target if it exists (test set won't have these)
        gt_path = npy_path.with_suffix(".png")
        if gt_path.exists():
            target_out = output_dir / f"{stem}_target.png"
            target_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(gt_path, target_out)

        count += 1
        if count % 50 == 0:
            print(f"  Processed {count} samples...")

    return count


def save_metadata(output_dir: Path, args: argparse.Namespace) -> None:
    """Save preprocessing parameters so they can be reproduced."""
    metadata = {
        "source": "Single Photon Challenge reconstruction dataset",
        "source_url": "https://singlephotonchallenge.com/download",
        "num_frames": args.num_frames,
        "invert_response": args.invert_response,
        "invert_factor": args.invert_factor,
        "tonemap": args.tonemap,
        "split": args.split or "all",
        "notes": (
            "Measurements are preprocessed from raw photoncubes using: "
            "naive sum averaging, SPC response inversion, and sRGB tonemapping. "
            "Saved as uint8 PNGs. Targets are copied from original ground-truth PNGs."
        ),
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess the full Single Photon Challenge dataset."
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to save preprocessed PNG pairs.",
    )
    parser.add_argument(
        "--scratch-dir", type=str, default=None,
        help="Temporary directory for downloads/extraction. "
             "Defaults to <output-dir>/_scratch.",
    )
    parser.add_argument(
        "--split", type=str, default=None, choices=["train", "test"],
        help="Only process this split. Default: process both.",
    )
    parser.add_argument(
        "--num-frames", type=int, default=16,
        help="Number of binary frames to average (default: 16).",
    )
    parser.add_argument(
        "--invert-response", action="store_true", default=True,
        help="Apply SPC response inversion (default: True).",
    )
    parser.add_argument(
        "--no-invert-response", action="store_false", dest="invert_response",
        help="Disable SPC response inversion.",
    )
    parser.add_argument(
        "--invert-factor", type=float, default=0.5,
        help="Factor for SPC response inversion (default: 0.5).",
    )
    parser.add_argument(
        "--tonemap", action="store_true", default=True,
        help="Apply linear RGB to sRGB tonemapping (default: True).",
    )
    parser.add_argument(
        "--no-tonemap", action="store_false", dest="tonemap",
        help="Disable sRGB tonemapping.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    scratch_dir = Path(args.scratch_dir) if args.scratch_dir else output_dir / "_scratch"

    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    # Check dependencies
    for tool in ["aws", "7z"]:
        if shutil.which(tool) is None:
            print(f"ERROR: '{tool}' not found on PATH. Please install it first.")
            sys.exit(1)

    # List available chunks
    print(f"Listing S3 chunks (split={args.split or 'all'})...")
    chunks = list_s3_chunks(split=args.split)
    if not chunks:
        print("ERROR: No chunks found. Check your network connection and aws-cli setup.")
        sys.exit(1)
    print(f"Found {len(chunks)} chunk(s) to process.")

    # Resume support: skip chunks already processed in a previous run
    completed_file = output_dir / "completed_chunks.txt"
    completed = set()
    if completed_file.exists():
        completed = set(completed_file.read_text().strip().splitlines())
        print(f"Resuming: {len(completed)} chunk(s) already processed, "
              f"{len(chunks) - len(completed)} remaining.\n")
    else:
        print()

    total_samples = 0

    for i, chunk_key in enumerate(chunks):
        if chunk_key in completed:
            print(f"[{i+1}/{len(chunks)}] Skipping (already done): {chunk_key}")
            continue

        print(f"[{i+1}/{len(chunks)}] Processing chunk: {chunk_key}")

        # Download
        zip_path = download_chunk(chunk_key, scratch_dir)

        # Extract
        extract_dir = scratch_dir / "extracted"
        extract_dir.mkdir(exist_ok=True)
        extract_chunk(zip_path, extract_dir)

        # Preprocess
        count = process_extracted_chunk(
            extract_dir, output_dir,
            args.num_frames, args.invert_response,
            args.invert_factor, args.tonemap,
        )
        total_samples += count
        print(f"  Done: {count} samples from this chunk.\n")

        # Clean up raw data
        zip_path.unlink(missing_ok=True)
        shutil.rmtree(extract_dir, ignore_errors=True)

        # Record this chunk as completed
        with open(completed_file, "a") as f:
            f.write(chunk_key + "\n")

    # Save metadata
    save_metadata(output_dir, args)

    # Clean up scratch directory if empty
    try:
        scratch_dir.rmdir()
    except OSError:
        pass

    print(f"\nFinished! {total_samples} total samples saved to {output_dir}")


if __name__ == "__main__":
    main()