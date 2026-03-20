from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from config import DEVICE, DIFFUSION_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, checkpoint_path
from dataset import get_training_dataloader
from diffusion import LinearNoiseScheduler
from model import UNet
from utils import ensure_dir, save_checkpoint, seed_everything


def _extract_target(batch, dataset_mode: str) -> torch.Tensor:
    """
    Normalise what the dataloader returns into a single target tensor.

    - "sample" mode: batch is already a tensor (ground-truth only).
    - "full" mode: batch is (measurement, target); we train on target.
    """
    if dataset_mode == "sample":
        return batch
    else:
        _measurement, target = batch
        return target


def train(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    scheduler: LinearNoiseScheduler,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    num_epochs: int,
    device: torch.device,
    save_path: str,
    dataset_mode: str,
) -> None:

    model.train()

    scaler = GradScaler(enabled=TRAIN_CONFIG["use_amp"])
    accumulation = TRAIN_CONFIG["gradient_accumulation"]

    for epoch in range(num_epochs):

        epoch_loss = 0.0

        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(dataloader)):

            images = _extract_target(batch, dataset_mode).to(device)

            noise = torch.randn_like(images)

            timesteps = torch.randint(
                0,
                scheduler.num_timesteps,
                (images.shape[0],),
                device=device,
            )

            with autocast(enabled=TRAIN_CONFIG["use_amp"]):

                noisy_images = scheduler.add_noise(images, noise, timesteps)

                pred_noise = model(noisy_images, timesteps)

                loss = criterion(pred_noise, noise)
                loss = loss / accumulation

            scaler.scale(loss).backward()

            if (step + 1) % accumulation == 0:

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad()

            epoch_loss += loss.item() * accumulation

        if len(dataloader) % accumulation != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()

        epoch_loss /= len(dataloader)

        print(f"Epoch {epoch+1} | Loss {epoch_loss:.4f}")

        save_checkpoint(model, save_path)


def main() -> None:

    seed_everything(TRAIN_CONFIG["seed"])

    ensure_dir(TRAIN_CONFIG["task_name"])

    scheduler = LinearNoiseScheduler(**DIFFUSION_CONFIG)

    dataloader = get_training_dataloader(TRAIN_CONFIG)

    model = UNet(MODEL_CONFIG).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=TRAIN_CONFIG["lr"])

    criterion = nn.MSELoss()

    train(
        model=model,
        dataloader=dataloader,
        scheduler=scheduler,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=TRAIN_CONFIG["num_epochs"],
        device=DEVICE,
        save_path=checkpoint_path(),
        dataset_mode=TRAIN_CONFIG["dataset_mode"],
    )


if __name__ == "__main__":
    main()