from __future__ import annotations

import math

from config import DEVICE, DIFFUSION_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, checkpoint_path, generated_samples_path
from diffusion import LinearNoiseScheduler, sample_ddpm, sample_ddim
from model import Unet
from utils import load_checkpoint, save_image_grid


def main() -> None:
    scheduler = LinearNoiseScheduler(**DIFFUSION_CONFIG)
    model = Unet(MODEL_CONFIG)
    model = load_checkpoint(model, checkpoint_path(), DEVICE)

    num_samples = 16
    generated_samples = sample_ddpm(
        model=model,
        scheduler=scheduler,
        num_samples=num_samples,
        image_size=MODEL_CONFIG["im_size"],
        channels=MODEL_CONFIG["im_channels"],
        device=DEVICE,
    )

    save_image_grid(
        generated_samples,
        output_path=generated_samples_path(),
        nrow=int(math.sqrt(num_samples)),
    )

    print(f"Generated samples saved to {generated_samples_path('ddpm')}")

    num_samples = 16
    generated_samples = sample_ddim(
        model=model,
        scheduler=scheduler,
        num_samples=num_samples,
        image_size=MODEL_CONFIG["im_size"],
        channels=MODEL_CONFIG["im_channels"],
        device=DEVICE,
        num_inference_steps=250,
        eta=0.0
    )

    save_image_grid(
        generated_samples,
        output_path=generated_samples_path(),
        nrow=int(math.sqrt(num_samples)),
    )

    print(f"Generated samples saved to {generated_samples_path('ddim')}")


if __name__ == "__main__":
    main()
