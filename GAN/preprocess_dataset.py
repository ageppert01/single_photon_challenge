"""
Preprocess SPAD dataset: read bit-packed NPY files, compute naive-sum images
at all 7 budgets, and save as PNGs.

Set EXTRACTED_DIR to the path of your extracted training data:
  {EXTRACTED_DIR}/{scene}/{id}.npy + {id}.png

Output structure:
  processed/{train,test}/{scene}/{id}/
      ground_truth.png
      naivesum_B0016.png  ...  naivesum_B1024.png
"""

import os
import glob

import numpy as np
from PIL import Image
from tqdm import tqdm

# Set this to the directory containing your extracted training scenes
EXTRACTED_DIR = "data/raw/train"
TRAIN_OUT     = "processed/train"
TEST_OUT      = "processed/test"
BUDGETS       = [16, 32, 64, 128, 256, 512, 1024]
NUM_TEST_SCENES = 0   # 0 = all training scenes go to TRAIN_OUT (test set is separate/unlabeled)

os.makedirs(TRAIN_OUT, exist_ok=True)
os.makedirs(TEST_OUT,  exist_ok=True)


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


def process_sample(npy_path, gt_path, out_dir):
    expected = [os.path.join(out_dir, f"naivesum_B{b:04d}.png") for b in BUDGETS]
    expected.append(os.path.join(out_dir, "ground_truth.png"))
    if all(os.path.exists(p) for p in expected):
        return False  # already done

    os.makedirs(out_dir, exist_ok=True)

    npy_raw = np.load(npy_path)
    naive_sums = compute_naive_sums(npy_raw, BUDGETS)
    for b, img in naive_sums.items():
        Image.fromarray(img).save(os.path.join(out_dir, f"naivesum_B{b:04d}.png"))

    gt = Image.open(gt_path).convert("RGB")
    gt.save(os.path.join(out_dir, "ground_truth.png"))
    return True


def main():
    all_scenes = sorted(os.listdir(EXTRACTED_DIR))
    test_scenes = set(all_scenes[-NUM_TEST_SCENES:]) if NUM_TEST_SCENES > 0 else set()

    print(f"Scenes: {len(all_scenes) - len(test_scenes)} train, {len(test_scenes)} test")
    print(f"Test scenes: {sorted(test_scenes)}")
    print()

    total_processed = total_skipped = 0

    for scene in tqdm(sorted(all_scenes), desc="Scenes"):
        scene_dir = os.path.join(EXTRACTED_DIR, scene)
        out_root  = TEST_OUT if scene in test_scenes else TRAIN_OUT

        npy_files = sorted(glob.glob(os.path.join(scene_dir, "*.npy")))
        for npy_path in npy_files:
            id_str  = os.path.splitext(os.path.basename(npy_path))[0]
            gt_path = npy_path.replace(".npy", ".png")
            if not os.path.exists(gt_path):
                continue
            out_dir = os.path.join(out_root, scene, id_str)
            if process_sample(npy_path, gt_path, out_dir):
                total_processed += 1
            else:
                total_skipped += 1

    print()
    print(f"Done. processed={total_processed}, skipped={total_skipped}")
    print(f"Train → {TRAIN_OUT}")
    print(f"Test  → {TEST_OUT}")


if __name__ == "__main__":
    main()
