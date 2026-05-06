"""
Preprocess the full Single Photon Challenge dataset.

Downloads compressed chunks from the UW-Madison S3 bucket one at a time,
preprocesses each photoncube into a measurement PNG using the adaptive
similarity-flow-sum pipeline (block-wise scale+translation registration,
optional dense optical flow refinement, SPC response inversion, sRGB
tonemap), copies the ground-truth target PNG alongside it, then deletes
the raw chunk.

Requirements:
    - aws-cli  (https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html)

Usage:
    python scripts/preprocess_full_dataset.py --output-dir ./preprocessed
    python scripts/preprocess_full_dataset.py --output-dir ./preprocessed --split test
    python scripts/preprocess_full_dataset.py --output-dir ./preprocessed --K 128

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
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

# Allow imports of photoncube_preprocess:
#   - In HTCondor sandbox: all transferred files are in the working directory
#   - Running locally:     photoncube_preprocess.py lives in ../diffusion/
sys.path.insert(0, ".")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffusion"))

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.ndimage import shift
from skimage.registration import phase_cross_correlation, optical_flow_tvl1
from skimage.transform import warp

from photoncube_preprocess import (
    spc_avg_to_rgb,
    linearrgb_to_srgb,
)


S3_BUCKET = "s3://public-datasets/challenges/reconstruction"
S3_ENDPOINT = "https://web.s3.wisc.edu"


# ---------------------------------------------------------------------------
# Adaptive similarity-flow-sum helpers
# ---------------------------------------------------------------------------

def rgb_to_gray(img):
    """Convert H x W x 3 image to grayscale using standard luminance weights."""
    if img.ndim == 2:
        return img
    return (0.2989 * img[..., 0] +
            0.5870 * img[..., 1] +
            0.1140 * img[..., 2]).astype(np.float32)


def zoom_about_center(img, scale, order=1):
    """Zoom image about its center and return same output size."""
    if abs(scale - 1.0) < 1e-6:
        return img.astype(np.float32).copy()

    H, W = img.shape[:2]
    is_color = (img.ndim == 3)
    zoom_factors = (scale, scale, 1.0) if is_color else (scale, scale)

    z = ndi.zoom(img.astype(np.float32), zoom_factors, order=order)
    out = np.zeros_like(img, dtype=np.float32)
    Hz, Wz = z.shape[:2]

    if scale >= 1.0:
        y0 = (Hz - H) // 2
        x0 = (Wz - W) // 2
        out = z[y0:y0 + H, x0:x0 + W].astype(np.float32)
    else:
        y0 = (H - Hz) // 2
        x0 = (W - Wz) // 2
        if is_color:
            out[y0:y0 + Hz, x0:x0 + Wz, :] = z
        else:
            out[y0:y0 + Hz, x0:x0 + Wz] = z

    return out


def shift_image(img, shift_yx, order=1):
    """Shift image by (dy, dx). Same size output."""
    shift_yx = np.asarray(shift_yx, dtype=np.float32)
    if img.ndim == 3:
        shift_full = (shift_yx[0], shift_yx[1], 0.0)
    else:
        shift_full = (shift_yx[0], shift_yx[1])
    return ndi.shift(img.astype(np.float32), shift=shift_full,
                     order=order, mode="constant", cval=0.0)


def warp_with_flow(img, flow_vu, order=1):
    """Warp image using dense flow (2, H, W) from optical_flow_tvl1."""
    H, W = img.shape[:2]
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    coords = np.array([rr + flow_vu[0], cc + flow_vu[1]], dtype=np.float32)

    if img.ndim == 2:
        return warp(img.astype(np.float32), coords, order=order,
                    mode="constant", cval=0.0, preserve_range=True).astype(np.float32)

    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        out[..., c] = warp(img[..., c].astype(np.float32), coords, order=order,
                           mode="constant", cval=0.0, preserve_range=True)
    return out


def estimate_global_scale_translation(
    ref_gray, mov_gray,
    scale_candidates=(0.90, 0.94, 0.98, 1.00, 1.02, 1.06, 1.10),
    upsample_factor=10,
):
    """Search over scale candidates + phase-correlation translation."""
    H, W = ref_gray.shape
    best = None

    for s in scale_candidates:
        mov_s = zoom_about_center(mov_gray, s, order=1)
        shift_yx, _, _ = phase_cross_correlation(ref_gray, mov_s,
                                                  upsample_factor=upsample_factor)
        mov_st = shift_image(mov_s, shift_yx, order=1)

        valid = shift_image(np.ones((H, W), dtype=np.float32), shift_yx, order=0)
        valid = (valid > 0.5).astype(np.float32)

        denom = max(valid.sum(), 1.0)
        mse = (((ref_gray - mov_st) ** 2) * valid).sum() / denom

        candidate = (s, np.array(shift_yx, dtype=np.float32), float(mse), valid)
        if best is None or candidate[2] < best[2]:
            best = candidate

    return best


def make_nonoverlapping_registration_blocks(frames, block_size):
    """Sum consecutive non-overlapping blocks for robust registration."""
    T = frames.shape[0]
    num_blocks = T // block_size
    usable_T = num_blocks * block_size
    frames = frames[-usable_T:]

    reg_blocks = []
    block_ranges = []
    for b in range(num_blocks):
        start = b * block_size
        end = start + block_size
        reg_blocks.append(frames[start:end].sum(axis=0).astype(np.float32))
        block_ranges.append((start, end))

    return np.stack(reg_blocks, axis=0), block_ranges, frames


def stage1_adaptive_similarity_flow_sum(
    frames,
    K=256,
    reg_block_size=8,
    scale_candidates=(0.90, 0.94, 0.98, 1.00, 1.02, 1.06, 1.10),
    overlap_threshold=0.45,
    max_global_mse=None,
    use_dense_flow=True,
    flow_attachment=15,
    flow_tightness=0.3,
    num_warp=5,
):
    """
    Adaptive Stage 1: scale+translation registration per block with optional
    dense optical-flow refinement, then SPC response inversion and sRGB tonemap.

    Returns a float32 image in [0, 1] (display-ready sRGB).
    """
    if K is not None:
        frames = frames[-K:]

    T, H, W, C = frames.shape
    if T < reg_block_size:
        raise ValueError("Need at least reg_block_size frames.")

    reg_blocks, block_ranges, frames = make_nonoverlapping_registration_blocks(
        frames, block_size=reg_block_size,
    )
    T = frames.shape[0]
    num_blocks = reg_blocks.shape[0]

    ref_idx = num_blocks - 1
    ref_block = reg_blocks[ref_idx]
    ref_gray = rgb_to_gray(ref_block)

    sum_image = np.zeros((H, W, C), dtype=np.float32)
    count_image = np.zeros((H, W), dtype=np.float32)

    # Always include reference block
    ref_start, ref_end = block_ranges[ref_idx]
    for t in range(ref_start, ref_end):
        sum_image += frames[t].astype(np.float32)
        count_image += 1.0

    for b in range(num_blocks - 2, -1, -1):
        mov_block = reg_blocks[b]
        mov_gray = rgb_to_gray(mov_block)

        scale, shift_yx, global_mse, valid_mask = estimate_global_scale_translation(
            ref_gray, mov_gray,
            scale_candidates=scale_candidates,
            upsample_factor=10,
        )

        overlap = valid_mask.mean()
        if overlap < overlap_threshold:
            break
        if max_global_mse is not None and global_mse > max_global_mse:
            break

        # Optional local refinement
        flow_vu = None
        if use_dense_flow:
            mov_block_sim = shift_image(
                zoom_about_center(mov_block, scale, order=1), shift_yx, order=1,
            )
            mov_block_sim_gray = rgb_to_gray(mov_block_sim)
            flow_vu = optical_flow_tvl1(
                ref_gray, mov_block_sim_gray,
                attachment=flow_attachment,
                tightness=flow_tightness,
                num_warp=num_warp,
            )

        start, end = block_ranges[b]
        for t in range(start, end):
            fr = frames[t].astype(np.float32)
            warped = zoom_about_center(fr, scale, order=1)
            warped = shift_image(warped, shift_yx, order=1)

            w = np.ones((H, W), dtype=np.float32)
            w = zoom_about_center(w, scale, order=0)
            w = shift_image(w, shift_yx, order=0)
            w = (w > 0.5).astype(np.float32)

            if flow_vu is not None:
                warped = warp_with_flow(warped, flow_vu, order=1)
                w = warp_with_flow(w, flow_vu, order=0)
                w = (w > 0.5).astype(np.float32)

            sum_image += warped * w[..., None]
            count_image += w

    stage1_avg = sum_image / np.maximum(count_image[..., None], 1e-8)

    # SPC response inversion + sRGB tonemap (same as before)
    out = spc_avg_to_rgb(stage1_avg, factor=0.5)
    out = linearrgb_to_srgb(out)
    out = np.clip(out, 0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# Pipeline infrastructure (unchanged except preprocess_photoncube_to_png)
# ---------------------------------------------------------------------------

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
            wait = 30 * attempt
            print(f"  Retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError(
        f"Download failed after {max_retries} attempts: {' '.join(cmd)}"
    )


def extract_chunk(zip_path: Path, extract_dir: Path) -> None:
    """Extract a zip file. Tries Python zipfile first, falls back to 7z."""
    print(f"  Extracting {zip_path.name}")

    # Try Python's built-in zipfile (supports LZMA since Python 3.3)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        return
    except Exception as e:
        print(f"  WARNING: Python zipfile failed ({e}), trying 7z...")

    # Fall back to 7z / 7zz if available
    sz = shutil.which("7z") or shutil.which("7zz")
    if sz is None:
        raise RuntimeError(
            f"Cannot extract {zip_path.name}: Python zipfile failed (likely "
            "missing LZMA support) and 7z is not installed. Install p7zip-full "
            "or ensure Python has lzma support."
        )
    cmd = [sz, "x", str(zip_path), f"-o{extract_dir}", "-y"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"7z extraction failed: {result.stderr}")


def preprocess_photoncube_to_png(
    npy_path: Path,
    output_path: Path,
    K: int,
    reg_block_size: int,
    scale_candidates: tuple[float, ...],
    overlap_threshold: float,
    max_global_mse: float | None,
    use_dense_flow: bool,
    flow_attachment: int,
    flow_tightness: float,
    num_warp: int,
) -> None:
    """Preprocess a single photoncube .npy file and save as uint8 PNG."""
    pc = np.load(str(npy_path), mmap_mode="r")
    frames = np.unpackbits(pc[-K:], axis=2)

    image = stage1_adaptive_similarity_flow_sum(
        frames,
        K=K,
        reg_block_size=reg_block_size,
        scale_candidates=scale_candidates,
        overlap_threshold=overlap_threshold,
        max_global_mse=max_global_mse,
        use_dense_flow=use_dense_flow,
        flow_attachment=flow_attachment,
        flow_tightness=flow_tightness,
        num_warp=num_warp,
    )

    image_uint8 = (image * 255).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_uint8).save(output_path)


def _process_single_npy(args_tuple):
    """Worker function for parallel processing (must be top-level for pickling)."""
    (npy_path, extract_dir, output_dir,
     K, reg_block_size, scale_candidates,
     overlap_threshold, max_global_mse,
     use_dense_flow, flow_attachment,
     flow_tightness, num_warp) = args_tuple

    npy_path = Path(npy_path)
    extract_dir = Path(extract_dir)
    output_dir = Path(output_dir)
    rel_path = npy_path.relative_to(extract_dir)
    stem = rel_path.with_suffix("")

    measurement_out = output_dir / f"{stem}_measurement.png"
    try:
        preprocess_photoncube_to_png(
            npy_path, measurement_out,
            K, reg_block_size, scale_candidates,
            overlap_threshold, max_global_mse,
            use_dense_flow, flow_attachment,
            flow_tightness, num_warp,
        )
    except Exception as e:
        print(f"  WARNING: Failed to process {npy_path}: {e}")
        return 0

    gt_path = npy_path.with_suffix(".png")
    if gt_path.exists():
        target_out = output_dir / f"{stem}_target.png"
        target_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(gt_path, target_out)

    return 1


def process_extracted_chunk(
    extract_dir: Path,
    output_dir: Path,
    K: int,
    reg_block_size: int,
    scale_candidates: tuple[float, ...],
    overlap_threshold: float,
    max_global_mse: float | None,
    use_dense_flow: bool,
    flow_attachment: int,
    flow_tightness: float,
    num_warp: int,
    num_workers: int = 1,
) -> int:
    """Process all photoncube/target pairs in an extracted chunk directory."""
    npy_files = sorted(extract_dir.rglob("*.npy"))
    if not npy_files:
        return 0

    work_args = [
        (str(npy_path), str(extract_dir), str(output_dir),
         K, reg_block_size, scale_candidates,
         overlap_threshold, max_global_mse,
         use_dense_flow, flow_attachment,
         flow_tightness, num_warp)
        for npy_path in npy_files
    ]

    count = 0
    if num_workers <= 1:
        for args_tuple in work_args:
            count += _process_single_npy(args_tuple)
            if count % 50 == 0 and count > 0:
                print(f"  Processed {count} samples...")
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as pool:
            for result in pool.map(_process_single_npy, work_args):
                count += result
                if count % 50 == 0 and count > 0:
                    print(f"  Processed {count} samples...")

    return count


def save_metadata(output_dir: Path, args: argparse.Namespace) -> None:
    """Save preprocessing parameters so they can be reproduced."""
    metadata = {
        "source": "Single Photon Challenge reconstruction dataset",
        "source_url": "https://singlephotonchallenge.com/download",
        "algorithm": "adaptive_similarity_flow_sum",
        "K": args.K,
        "reg_block_size": args.reg_block_size,
        "scale_candidates": list(args.scale_candidates),
        "overlap_threshold": args.overlap_threshold,
        "max_global_mse": args.max_global_mse,
        "use_dense_flow": args.use_dense_flow,
        "flow_attachment": args.flow_attachment,
        "flow_tightness": args.flow_tightness,
        "num_warp": args.num_warp,
        "invert_response": True,
        "invert_factor": 0.5,
        "tonemap": True,
        "split": args.split or "all",
        "notes": (
            "Measurements are preprocessed from raw photoncubes using: "
            "adaptive block-wise scale+translation registration with optional "
            "dense optical-flow refinement, followed by SPC response inversion "
            "and sRGB tonemapping. Saved as uint8 PNGs. Targets are copied "
            "from original ground-truth PNGs."
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
    # --- Algorithm parameters ---
    parser.add_argument(
        "--K", type=int, default=256,
        help="Number of last binary frames to use (default: 256).",
    )
    parser.add_argument(
        "--reg-block-size", type=int, default=8,
        help="Registration block size (default: 8).",
    )
    parser.add_argument(
        "--scale-candidates", type=float, nargs="+",
        default=[0.90, 0.94, 0.98, 1.00, 1.02, 1.06, 1.10],
        help="Scale candidates for global registration "
             "(default: 0.90 0.94 0.98 1.00 1.02 1.06 1.10).",
    )
    parser.add_argument(
        "--overlap-threshold", type=float, default=0.45,
        help="Minimum valid-pixel overlap to accept a block (default: 0.45).",
    )
    parser.add_argument(
        "--max-global-mse", type=float, default=None,
        help="Maximum global MSE to accept a block (default: None = no limit).",
    )
    parser.add_argument(
        "--use-dense-flow", action="store_true", default=True,
        help="Enable dense optical-flow refinement (default: True).",
    )
    parser.add_argument(
        "--no-dense-flow", action="store_false", dest="use_dense_flow",
        help="Disable dense optical-flow refinement.",
    )
    parser.add_argument(
        "--flow-attachment", type=int, default=15,
        help="TVL1 optical flow attachment parameter (default: 15).",
    )
    parser.add_argument(
        "--flow-tightness", type=float, default=0.3,
        help="TVL1 optical flow tightness parameter (default: 0.3).",
    )
    parser.add_argument(
        "--num-warp", type=int, default=5,
        help="TVL1 optical flow num_warp parameter (default: 5).",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="Parallel workers for processing samples within each chunk. "
             "0 = use all available CPUs (default: 0).",
    )
    args = parser.parse_args()

    # Convert list to tuple for consistency
    args.scale_candidates = tuple(args.scale_candidates)

    if args.num_workers <= 0:
        args.num_workers = os.cpu_count() or 1

    output_dir = Path(args.output_dir)
    scratch_dir = Path(args.scratch_dir) if args.scratch_dir else output_dir / "_scratch"

    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    for tool in ["aws"]:
        if shutil.which(tool) is None:
            print(f"ERROR: '{tool}' not found on PATH. Please install it first.")
            sys.exit(1)

    print(f"Listing S3 chunks (split={args.split or 'all'})...")
    chunks = list_s3_chunks(split=args.split)
    if not chunks:
        print("ERROR: No chunks found. Check your network connection and aws-cli setup.")
        sys.exit(1)
    print(f"Found {len(chunks)} chunk(s) to process.")
    print(f"Using {args.num_workers} parallel worker(s).")

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

        zip_path = download_chunk(chunk_key, scratch_dir)

        extract_dir = scratch_dir / "extracted"
        extract_dir.mkdir(exist_ok=True)
        extract_chunk(zip_path, extract_dir)

        count = process_extracted_chunk(
            extract_dir, output_dir,
            args.K, args.reg_block_size,
            args.scale_candidates, args.overlap_threshold,
            args.max_global_mse, args.use_dense_flow,
            args.flow_attachment, args.flow_tightness,
            args.num_warp, args.num_workers,
        )
        total_samples += count
        print(f"  Done: {count} samples from this chunk.\n")

        zip_path.unlink(missing_ok=True)
        shutil.rmtree(extract_dir, ignore_errors=True)

        with open(completed_file, "a") as f:
            f.write(chunk_key + "\n")

    save_metadata(output_dir, args)

    try:
        scratch_dir.rmdir()
    except OSError:
        pass

    print(f"\nFinished! {total_samples} total samples saved to {output_dir}")


if __name__ == "__main__":
    main()