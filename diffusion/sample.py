from __future__ import annotations

import math

from config import (
    DEVICE,
    DIFFUSION_CONFIG,
    MODEL_CONFIG,
    TRAIN_CONFIG,
    checkpoint_path,
    generated_samples_path,
)
from diffusion import LinearNoiseScheduler, sample_ddpm, sample_ddim
from model import UNet
from utils import load_checkpoint, save_image_grid


def main() -> None:

    scheduler = LinearNoiseScheduler(**DIFFUSION_CONFIG)

    model = UNet(MODEL_CONFIG)
    model = load_checkpoint(model, checkpoint_path(), DEVICE)

    num_samples = TRAIN_CONFIG["num_generated_samples"]

    images = sample_ddpm(
        model,
        scheduler,
        num_samples,
        MODEL_CONFIG["im_size"],
        MODEL_CONFIG["im_channels"],
        DEVICE,
    )

    save_image_grid(
        images,
        generated_samples_path("ddpm"),
        nrow=max(1, int(math.sqrt(num_samples))),
    )

    images = sample_ddim(
        model,
        scheduler,
        num_samples,
        MODEL_CONFIG["im_size"],
        MODEL_CONFIG["im_channels"],
        DEVICE,
    )

    save_image_grid(
        images,
        generated_samples_path("ddim"),
        nrow=max(1, int(math.sqrt(num_samples))),
    )


if __name__ == "__main__":
    main()