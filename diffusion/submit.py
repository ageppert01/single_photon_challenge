"""
Generate competition submission zip.

Produces a zip file with structure:
    <SCENE-NAME>/<FRAME-IDX>.png

matching the test set layout as required by singlephotonchallenge.com.

Usage:
    python submit.py                        # default settings
    python submit.py --best                 # use best checkpoint
    python submit.py --steps 20             # override DDIM steps
    python submit.py --output submission.zip
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from config import (
    SD_PALETTE_MODEL_CONFIG,
    SD_PALETTE_SAMPLE_CONFIG,
    FULL_DATASET_CONFIG,
    sd_palette_checkpoint_dir,
    sd_palette_best_checkpoint_dir,
)
from dataset import PreprocessedPairedDataset
from sd_utils import load_palette_sd, sd_palette_inference


def parse_args():
    parser = argparse.ArgumentParser(description="Generate competition submission")
    parser.add_argument(
        "--best", action="store_true",
        help="Use best validation checkpoint",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help=f"DDIM steps (default: {SD_PALETTE_SAMPLE_CONFIG['num_steps']})",
    )
    parser.add_argument(
        "--eta", type=float, default=None,
        help=f"DDIM eta (default: {SD_PALETTE_SAMPLE_CONFIG['eta']})",
    )
    parser.add_argument(
        "--output", type=str, default="submission.zip",
        help="Output zip file path (default: submission.zip)",
    )
    parser.add_argument(
        "--split", type=str, default="test",
        help="Dataset split to run on (default: test)",
    )
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────
    model_id = SD_PALETTE_MODEL_CONFIG["sd_model_id"]
    use_qvae = SD_PALETTE_MODEL_CONFIG.get("use_gqir_qvae", False)

    if args.best:
        ckpt_dir = sd_palette_best_checkpoint_dir()
        print("Using best validation checkpoint")
    else:
        ckpt_dir = sd_palette_checkpoint_dir()

    meas_vae, vae, unet, null_embeds = load_palette_sd(
        model_id, ckpt_dir, device, use_gqir_qvae=use_qvae,
    )
    unet.eval()
    print(f"Model loaded (qVAE: {meas_vae is not None})")

    num_steps = args.steps or SD_PALETTE_SAMPLE_CONFIG["num_steps"]
    eta = args.eta if args.eta is not None else SD_PALETTE_SAMPLE_CONFIG["eta"]
    print(f"DDIM: {num_steps} steps, eta={eta}")

    # ── Load test dataset ─────────────────────────────────────────────────
    dataset = PreprocessedPairedDataset(
        source=FULL_DATASET_CONFIG["dataset_source"],
        local_dir=FULL_DATASET_CONFIG.get("dataset_local_dir"),
        hf_repo=FULL_DATASET_CONFIG.get("dataset_hf_repo"),
        hf_revision=FULL_DATASET_CONFIG.get("dataset_hf_revision"),
        split=args.split,
    )

    # ── Generate predictions and write zip ────────────────────────────────
    print(f"Generating {len(dataset)} predictions → {args.output}")

    with ZipFile(args.output, "w") as zipf:
        for idx in tqdm(range(len(dataset)), desc="Inference"):
            measurement, _ = dataset[idx]
            measurement = measurement.unsqueeze(0).to(device, dtype=torch.float16)

            restored = sd_palette_inference(
                unet=unet,
                meas_vae=meas_vae,
                vae=vae,
                null_embeds=null_embeds,
                measurement=measurement,
                model_id=model_id,
                device=device,
                num_steps=num_steps,
                eta=eta,
            )

            # Convert [-1, 1] → [0, 255] uint8
            img = ((restored[0].float().cpu() + 1) / 2).clamp(0, 1)
            img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            # Derive submission path: scene/frame.png
            meas_path = dataset.measurements[idx]
            rel_path = meas_path.relative_to(dataset.root_dir)
            submit_name = str(rel_path).replace("_measurement.png", ".png")

            # Write PNG into zip
            pil_img = Image.fromarray(img)
            with zipf.open(submit_name, "w") as f:
                pil_img.save(f, format="PNG")

            if idx < 3 or (idx + 1) % 50 == 0:
                print(f"  [{idx:4d}] → {submit_name}")

    print(f"\nSubmission saved: {args.output}")
    print(f"Total images: {len(dataset)}")


if __name__ == "__main__":
    main()