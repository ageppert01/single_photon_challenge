from __future__ import annotations

import math

from config import DEVICE, DIFFUSION_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, checkpoint_path, generated_samples_path
from diffusion import LinearNoiseScheduler, ddpm_sample
from model import Unet
from utils import load_checkpoint, save_image_grid


def main() -> None:
    scheduler = LinearNoiseScheduler(**DIFFUSION_CONFIG)
    model = Unet(MODEL_CONFIG)
    model = load_checkpoint(model, checkpoint_path(), DEVICE)

    num_samples = 16
    generated_samples = ddpm_sample(
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

    print(f"Generated samples saved to {generated_samples_path()}")


if __name__ == "__main__":
    main()
