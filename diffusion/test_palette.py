"""Quick single-image test for the Palette conditional sampler."""

import torch
from model import UNet
from diffusion import LinearNoiseScheduler, sample_palette_ddim
from dataset import get_restoration_dataloader
from config import (
    palette_checkpoint_path,
    PALETTE_MODEL_CONFIG,
    DIFFUSION_CONFIG,
    RESTORATION_DATA_CONFIG,
)
from utils import save_comparison


def test_one():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load conditional model
    model = UNet(PALETTE_MODEL_CONFIG).to(device)
    model.load_state_dict(
        torch.load(palette_checkpoint_path(), map_location=device)
    )
    model.eval()
    print("Palette model loaded")

    # Load one sample
    dataloader = get_restoration_dataloader(RESTORATION_DATA_CONFIG)
    measurement, target = next(iter(dataloader))
    measurement = measurement.to(device)
    target = target.to(device)
    print(f"Measurement range: [{measurement.min():.2f}, {measurement.max():.2f}]")
    print(f"Target range:      [{target.min():.2f}, {target.max():.2f}]")

    # Run Palette DDIM with fewer steps for a quick test
    scheduler = LinearNoiseScheduler(**DIFFUSION_CONFIG)

    num_steps = 50  # fast test; bump to 250+ for quality
    print(f"Running Palette DDIM with {num_steps} steps...")
    restored = sample_palette_ddim(
        model=model,
        scheduler=scheduler,
        measurement=measurement,
        device=device,
        num_inference_steps=num_steps,
        eta=0.0,
    )

    print(f"Restored range:    [{restored.min():.2f}, {restored.max():.2f}]")

    # PSNR
    mse = torch.mean((restored - target) ** 2)
    psnr = -10 * torch.log10(mse + 1e-8)
    print(f"PSNR: {psnr.item():.2f} dB")

    # Save comparison: measurement | restored | target
    out_path = "single_photon_palette/palette_test.png"
    import os
    os.makedirs("single_photon_palette", exist_ok=True)
    save_comparison(measurement, restored, target, out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    test_one()