import torch
import os
from tqdm import tqdm
import csv

from model import UNet
from diffusion import LinearNoiseScheduler
from dataset import get_single_photon_restoration_dataloader
from config import checkpoint_path, MODEL_CONFIG, DIFFUSION_CONFIG, DDRM_CONFIG, RESTORATION_DATA_CONFIG
from ddrm import DDRMSampler
from utils import save_image_grid, save_comparison


def psnr(x, y):
    mse = torch.mean((x - y) ** 2)
    return -10 * torch.log10(mse + 1e-8)


def run_ddrm():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = get_single_photon_restoration_dataloader(RESTORATION_DATA_CONFIG)

    model = UNet(MODEL_CONFIG).to(device)
    model.load_state_dict(torch.load(checkpoint_path(), map_location=device))
    model.eval()

    scheduler = LinearNoiseScheduler(**DIFFUSION_CONFIG)

    sampler = DDRMSampler(
        model=model,
        scheduler=scheduler,
        device=device,
        observation_sigma=DDRM_CONFIG["observation_sigma"]
    )

    out_dir = DDRM_CONFIG["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    metrics_file = os.path.join(DDRM_CONFIG["output_dir"], "metrics.csv")
    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "psnr"])

        for i, (measurement, target) in enumerate(tqdm(dataloader)):
            measurement = measurement.to(device)
            target = target.to(device)

            restored = sampler.sample(measurement, DDRM_CONFIG["num_steps"])

            save_comparison(
                measurement,
                restored,
                target,
                f"{DDRM_CONFIG['output_dir']}/{i}_comparison.png"
            )

            writer.writerow([i, psnr(restored, target).item()])


if __name__ == "__main__":
    run_ddrm()