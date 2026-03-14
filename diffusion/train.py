from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from config import DEVICE, DIFFUSION_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, checkpoint_path
from dataset import get_mnist_dataloader, get_single_photon_dataloader
from diffusion import LinearNoiseScheduler
from model import Unet
from utils import ensure_dir, save_checkpoint, seed_everything


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    scheduler: LinearNoiseScheduler,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    num_epochs: int,
    device: torch.device,
    save_path: str,
) -> None:
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            optimizer.zero_grad()

            images = batch.to(device)

            noise = torch.randn_like(images, device=device)
            timesteps = torch.randint(0, scheduler.num_timesteps, (images.shape[0],), device=device)

            noisy_images = scheduler.add_noise(images, noise, timesteps)
            predicted_noise = model(noisy_images, timesteps)

            loss = criterion(predicted_noise, noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
        save_checkpoint(model, save_path)


def main() -> None:
    seed_everything(TRAIN_CONFIG["seed"])
    ensure_dir(TRAIN_CONFIG["task_name"])

    scheduler = LinearNoiseScheduler(**DIFFUSION_CONFIG)

    dataloader = get_single_photon_dataloader(
        batch_size=TRAIN_CONFIG["batch_size"],
        data_dir=TRAIN_CONFIG["data_dir"],
        shuffle=True,
        num_workers=TRAIN_CONFIG["num_workers"],
    )

    model = Unet(MODEL_CONFIG).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=TRAIN_CONFIG["lr"])
    criterion = nn.MSELoss()

    train(
        model=model,
        train_loader=dataloader,
        scheduler=scheduler,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=TRAIN_CONFIG["num_epochs"],
        device=DEVICE,
        save_path=checkpoint_path(),
    )


if __name__ == "__main__":
    main()