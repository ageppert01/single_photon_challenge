"""
Run inference on the preprocessed test set using a trained checkpoint.
Saves predicted images to an output directory.

Usage:
  python infer.py --checkpoint checkpoints/epoch_050.pt
  python infer.py --checkpoint checkpoints/epoch_050.pt --out_dir results/epoch_050
"""

import os
import glob
import argparse
import numpy as np
from PIL import Image

import torch

from train_gan import UNetGenerator, BUDGETS, DEVICE, _run_generator

TEST_ROOT  = "processed/test"
DEFAULT_OUT = "results"


def infer(checkpoint_path, test_root, out_dir, budgets):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    G = UNetGenerator().to(DEVICE)
    G.load_state_dict(ckpt["G_state"])
    G.eval()
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded checkpoint: {checkpoint_path} (epoch {epoch})")
    print(f"Output dir: {out_dir}")

    scene_dirs = sorted(glob.glob(os.path.join(test_root, "*", "*")))
    print(f"Found {len(scene_dirs)} test samples across "
          f"{len(set(d.split('/')[-2] for d in scene_dirs))} scenes")

    with torch.no_grad():
        for scene_dir in scene_dirs:
            scene = scene_dir.split("/")[-2]
            sid   = scene_dir.split("/")[-1]
            for b in budgets:
                budget_str = f"B{b:04d}"
                noisy_path = os.path.join(scene_dir, f"naivesum_{budget_str}.png")
                if not os.path.exists(noisy_path):
                    continue

                noisy_np = np.array(Image.open(noisy_path)).astype(np.float32) / 255.0
                out_np   = _run_generator(G, noisy_np, DEVICE)

                out_path = os.path.join(out_dir, budget_str, scene, f"{sid}_pred.png")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                Image.fromarray(out_np).save(out_path)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--out_dir", default=None, help="Output directory")
    parser.add_argument("--test_root", default=TEST_ROOT)
    args = parser.parse_args()

    out_dir = args.out_dir
    if out_dir is None:
        epoch_tag = os.path.splitext(os.path.basename(args.checkpoint))[0]
        out_dir = os.path.join(DEFAULT_OUT, epoch_tag)

    infer(args.checkpoint, args.test_root, out_dir, BUDGETS)
