"""
Palette-style conditional restoration.

Loads a trained conditional UNet, feeds each measurement through the
Palette DDPM/DDIM sampler, and saves:
  - Side-by-side comparison images (measurement | restored | target)
  - Per-image PSNR in a CSV file

Usage:
    python sample_palette.py
"""

import os
import csv

import torch
from tqdm import tqdm

from model import UNet
from diffusion import LinearNoiseScheduler, sample_palette_ddpm, sample_palette_ddim
from dataset import get_restoration_dataloader
from config import (
    palette_checkpoint_path,
    PALETTE_MODEL_CONFIG,
    DIFFUSION_CONFIG,
    RESTORATION_DATA_CONFIG,
    PALETTE_SAMPLE_CONFIG,
)
from utils import save_comparison


def psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    """PSNR between two tensors in [-1, 1]."""
    mse = torch.mean((x - y) ** 2)
    return (-10 * torch.log10(mse + 1e-8)).item()


def run_palette():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load the conditional model
    model = UNet(PALETTE_MODEL_CONFIG).to(device)
    ckpt_path = palette_checkpoint_path()
    print(f"Loading checkpoint: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Load paired data
    dataloader = get_restoration_dataloader(RESTORATION_DATA_CONFIG)

    # Noise scheduler
    scheduler = LinearNoiseScheduler(**DIFFUSION_CONFIG)

    # Sampling config
    sampler_type = PALETTE_SAMPLE_CONFIG["sampler"]
    num_steps = PALETTE_SAMPLE_CONFIG["num_steps"]
    eta = PALETTE_SAMPLE_CONFIG["eta"]
    out_dir = PALETTE_SAMPLE_CONFIG["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    print(f"Sampler: {sampler_type}, steps: {num_steps}, eta: {eta}")

    metrics_file = os.path.join(out_dir, "metrics.csv")
    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "psnr"])

        for i, (measurement, target) in enumerate(tqdm(dataloader)):
            measurement = measurement.to(device)

            # Run conditional sampling
            if sampler_type == "ddpm":
                restored = sample_palette_ddpm(
                    model=model,
                    scheduler=scheduler,
                    measurement=measurement,
                    device=device,
                )
            else:
                restored = sample_palette_ddim(
                    model=model,
                    scheduler=scheduler,
                    measurement=measurement,
                    device=device,
                    num_inference_steps=num_steps,
                    eta=eta,
                )

            # Save results
            if target is not None:
                target = target.to(device)

                save_comparison(
                    measurement,
                    restored,
                    target,
                    os.path.join(out_dir, f"{i:04d}_comparison.png"),
                )

                p = psnr(restored, target)
                writer.writerow([i, f"{p:.4f}"])
                print(f"  [{i}] PSNR: {p:.2f} dB")
            else:
                # No target available (test split)
                from torchvision.utils import save_image

                save_image(
                    (restored + 1) / 2,
                    os.path.join(out_dir, f"{i:04d}_restored.png"),
                )
                writer.writerow([i, "N/A"])

    print(f"\nResults saved to {out_dir}/")
    print(f"Metrics saved to {metrics_file}")


if __name__ == "__main__":
    run_palette()