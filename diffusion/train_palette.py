"""
Palette-style conditional diffusion training.

Key differences from unconditional train.py:
  1. Uses paired (measurement, target) data exclusively.
  2. At each step: diffuses the *target*, concatenates the measurement
     as a condition, and trains the model to predict the added noise.
  3. Can warm-start from a pretrained unconditional DDPM checkpoint
     by zero-initializing the new condition channels in init_conv.

The training objective is identical to standard DDPM:
    L = E_{t, x0, eps} [ || eps - eps_theta(x_t, t, measurement) ||^2 ]

where x_t = sqrt(alpha_bar_t) * target + sqrt(1 - alpha_bar_t) * eps,
and the model receives cat(x_t, measurement) as input.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from config import (
    DEVICE,
    DIFFUSION_CONFIG,
    PALETTE_MODEL_CONFIG,
    PALETTE_TRAIN_CONFIG,
    palette_checkpoint_path,
)
from dataset import get_training_dataset
from diffusion import LinearNoiseScheduler
from model import UNet, load_unconditional_into_conditional
from utils import ensure_dir, save_checkpoint, seed_everything


# ── Distributed helpers (same as train.py) ───────────────────────────────────


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def _is_main_process() -> bool:
    if _is_distributed():
        return dist.get_rank() == 0
    return True


def _setup_distributed() -> torch.device:
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = _local_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = DEVICE
    return device


def _cleanup_distributed() -> None:
    if _is_distributed():
        dist.destroy_process_group()


# ── Training ─────────────────────────────────────────────────────────────────


def train_palette(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    sampler: DistributedSampler | None,
    scheduler: LinearNoiseScheduler,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    num_epochs: int,
    device: torch.device,
    save_path: str,
) -> None:

    model.train()

    scaler = GradScaler(enabled=PALETTE_TRAIN_CONFIG["use_amp"])
    accumulation = PALETTE_TRAIN_CONFIG["gradient_accumulation"]

    for epoch in range(num_epochs):

        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(dataloader, disable=not _is_main_process())

        for step, batch in enumerate(pbar):

            # Paired data: (measurement, target), both in [-1, 1]
            measurement, target = batch
            measurement = measurement.to(device)
            target = target.to(device)

            # Skip samples without targets (test split edge case)
            if target is None:
                continue

            noise = torch.randn_like(target)

            timesteps = torch.randint(
                0,
                scheduler.num_timesteps,
                (target.shape[0],),
                device=device,
            )

            with autocast(enabled=PALETTE_TRAIN_CONFIG["use_amp"]):

                # Diffuse the target image
                noisy_target = scheduler.add_noise(target, noise, timesteps)

                # Palette: model takes (noisy_target, t, condition=measurement)
                pred_noise = model(noisy_target, timesteps, condition=measurement)

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

        # Flush remaining accumulated gradients
        if len(dataloader) % accumulation != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()

        epoch_loss /= max(len(dataloader), 1)

        if _is_main_process():
            print(f"Epoch {epoch+1}/{num_epochs} | Loss {epoch_loss:.6f}")

            raw_model = model.module if _is_distributed() else model
            save_checkpoint(raw_model, save_path)


def main() -> None:

    device = _setup_distributed()

    seed_everything(PALETTE_TRAIN_CONFIG["seed"])

    task_dir = PALETTE_TRAIN_CONFIG["task_name"]
    if _is_main_process():
        ensure_dir(task_dir)

    if _is_distributed():
        dist.barrier()

    scheduler = LinearNoiseScheduler(**DIFFUSION_CONFIG)

    # Build paired dataset — Palette always needs (measurement, target) pairs
    dataset = get_training_dataset(PALETTE_TRAIN_CONFIG)

    sampler = None
    shuffle = True

    if _is_distributed():
        sampler = DistributedSampler(dataset, shuffle=True)
        shuffle = False

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=PALETTE_TRAIN_CONFIG["batch_size"],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=PALETTE_TRAIN_CONFIG["num_workers"],
        pin_memory=True,
    )

    # Build conditional UNet
    model = UNet(PALETTE_MODEL_CONFIG).to(device)

    # Optionally warm-start from the unconditional checkpoint
    if PALETTE_TRAIN_CONFIG["init_from_unconditional"]:
        ckpt_path = PALETTE_TRAIN_CONFIG["unconditional_ckpt"]
        if os.path.isfile(ckpt_path):
            model = load_unconditional_into_conditional(model, ckpt_path, device)
        else:
            if _is_main_process():
                print(
                    f"WARNING: unconditional checkpoint not found at {ckpt_path}. "
                    f"Training conditional model from scratch."
                )

    # Optionally resume from a previous Palette checkpoint
    palette_ckpt = palette_checkpoint_path()
    if os.path.isfile(palette_ckpt):
        if _is_main_process():
            print(f"Resuming from Palette checkpoint: {palette_ckpt}")
        state = torch.load(palette_ckpt, map_location=device)
        model.load_state_dict(state)

    if _is_distributed():
        model = DDP(model, device_ids=[_local_rank()])

    optimizer = Adam(model.parameters(), lr=PALETTE_TRAIN_CONFIG["lr"])

    criterion = nn.MSELoss()

    train_palette(
        model=model,
        dataloader=dataloader,
        sampler=sampler,
        scheduler=scheduler,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=PALETTE_TRAIN_CONFIG["num_epochs"],
        device=device,
        save_path=palette_checkpoint_path(),
    )

    _cleanup_distributed()


if __name__ == "__main__":
    main()