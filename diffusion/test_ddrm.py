"""Quick single-image test for the DDRM sampler."""

import torch
from model import UNet
from diffusion import LinearNoiseScheduler
from dataset import get_restoration_dataloader
from config import checkpoint_path, MODEL_CONFIG, DIFFUSION_CONFIG, RESTORATION_DATA_CONFIG, DDRM_CONFIG
from ddrm import DDRMSampler
from utils import save_comparison


def test_one():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = UNet(MODEL_CONFIG).to(device)
    model.load_state_dict(torch.load(checkpoint_path(), map_location=device))
    model.eval()
    print("Model loaded")

    # Load one sample
    dataloader = get_restoration_dataloader(RESTORATION_DATA_CONFIG)
    measurement, target = next(iter(dataloader))
    measurement = measurement.to(device)
    target = target.to(device)
    print(f"Measurement range: [{measurement.min():.2f}, {measurement.max():.2f}]")
    print(f"Target range:      [{target.min():.2f}, {target.max():.2f}]")

    # Run DDRM with fewer steps for a quick test
    scheduler = LinearNoiseScheduler(**DIFFUSION_CONFIG)
    sampler = DDRMSampler(
        model=model,
        scheduler=scheduler,
        device=device,
        observation_sigma=DDRM_CONFIG["observation_sigma"],
    )

    num_steps = 1000  # fast test; bump to 250-1000 for quality
    print(f"Running DDRM with {num_steps} steps...")
    restored = sampler.sample(measurement, num_steps)

    print(f"Restored range:    [{restored.min():.2f}, {restored.max():.2f}]")

    # Save comparison: measurement | restored | target
    save_comparison(measurement, restored, target, "single_photon_ground_truth_diffusion/ddrm_test.png")
    print("Saved single_photon_ground_truth_diffusion/ddrm_test.png")


if __name__ == "__main__":
    test_one()