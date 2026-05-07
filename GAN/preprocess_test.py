"""
Preprocess unlabeled SPAD test data: extract test_0..test_4.zip, compute
naive-sum images at all 7 budgets, and save as PNGs. No ground truth.

Set DATASETS_DIR to the directory containing your test_0..test_4.zip files.

Output: processed/test/{scene}/{id}/naivesum_B*.png
"""

import os
import glob
import zipfile

import numpy as np
from PIL import Image
from tqdm import tqdm

# Set this to the directory containing your test zip files (test_0.zip .. test_4.zip)
DATASETS_DIR = "data/raw"
EXTRACT_DIR  = "data/raw/extracted/test"
TEST_OUT     = "processed/test"
BUDGETS      = [16, 32, 64, 128, 256, 512, 1024]
NUM_TEST_ZIPS = 5

os.makedirs(EXTRACT_DIR, exist_ok=True)
os.makedirs(TEST_OUT,    exist_ok=True)


def compute_naive_sums(npy_raw, budgets):
    """npy_raw is (1024, 800, 100, 3) uint8 bit-packed; returns {budget: (800,800,3) uint8}."""
    bits = np.unpackbits(npy_raw, axis=2)  # (1024, 800, 800, 3)
    results = {}
    for b in budgets:
        naive = np.clip(
            bits[-b:].sum(axis=0).astype(np.float32) / b * 255, 0, 255
        ).astype(np.uint8)
        results[b] = naive
    return results


def process_sample(npy_path, out_dir):
    expected = [os.path.join(out_dir, f"naivesum_B{b:04d}.png") for b in BUDGETS]
    if all(os.path.exists(p) for p in expected):
        return False  # already done

    os.makedirs(out_dir, exist_ok=True)
    npy_raw = np.load(npy_path)
    for b, img in compute_naive_sums(npy_raw, BUDGETS).items():
        Image.fromarray(img).save(os.path.join(out_dir, f"naivesum_B{b:04d}.png"))
    return True


def extract_zips():
    """Extract test_0..test_4.zip into EXTRACT_DIR if not already done."""
    for i in range(NUM_TEST_ZIPS):
        zip_path = os.path.join(DATASETS_DIR, f"test_{i}.zip")
        if not os.path.exists(zip_path):
            print(f"WARNING: {zip_path} not found, skipping")
            continue
        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path) as z:
            for member in z.infolist():
                # zip paths are test/scene/id.npy — strip the leading "test/" prefix
                rel = os.path.relpath(member.filename, "test")
                out_path = os.path.join(EXTRACT_DIR, rel)
                if member.is_dir():
                    os.makedirs(out_path, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    if not os.path.exists(out_path):
                        with z.open(member) as src, open(out_path, "wb") as dst:
                            dst.write(src.read())


def main():
    extract_zips()

    scenes = sorted(os.listdir(EXTRACT_DIR))
    print(f"\nFound {len(scenes)} test scenes: {scenes}")

    total_processed = total_skipped = 0
    for scene in tqdm(scenes, desc="Scenes"):
        scene_dir = os.path.join(EXTRACT_DIR, scene)
        npy_files = sorted(glob.glob(os.path.join(scene_dir, "*.npy")))
        for npy_path in npy_files:
            id_str  = os.path.splitext(os.path.basename(npy_path))[0]
            out_dir = os.path.join(TEST_OUT, scene, id_str)
            if process_sample(npy_path, out_dir):
                total_processed += 1
            else:
                total_skipped += 1

    print(f"\nDone. processed={total_processed}, skipped={total_skipped}")
    print(f"Test naive sums → {TEST_OUT}")


if __name__ == "__main__":
    main()
