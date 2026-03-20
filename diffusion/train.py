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

from config import DEVICE, DIFFUSION_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, checkpoint_path
from dataset import get_training_dataloader, get_training_dataset
from diffusion import LinearNoiseScheduler
from model import UNet
from utils import ensure_dir, save_checkpoint, seed_everything


# ── Distributed helpers ──────────────────────────────────────────────────────


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def _is_main_process() -> bool:
    if _is_distributed():
        return dist.get_rank() == 0
    return True


def _setup_distributed() -> torch.device:
    """
    Initialize the process group and return the device for this rank.

    When launched with `torchrun`, LOCAL_RANK and RANK are set automatically.
    When run as a plain `python train.py`, this falls back to single-GPU.
    """
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
    sampler: DistributedSampler | None,
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

        # Ensure different shuffling per epoch in distributed mode
        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(dataloader, disable=not _is_main_process())

        for step, batch in enumerate(pbar):

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

        # Flush remaining accumulated gradients
        if len(dataloader) % accumulation != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()

        epoch_loss /= len(dataloader)

        if _is_main_process():
            print(f"Epoch {epoch+1} | Loss {epoch_loss:.4f}")

            # Unwrap DDP to save just the underlying model weights
            raw_model = model.module if _is_distributed() else model
            save_checkpoint(raw_model, save_path)


def main() -> None:

    device = _setup_distributed()

    seed_everything(TRAIN_CONFIG["seed"])

    if _is_main_process():
        ensure_dir(TRAIN_CONFIG["task_name"])

    # Synchronise so rank 0 creates the directory before others proceed
    if _is_distributed():
        dist.barrier()

    scheduler = LinearNoiseScheduler(**DIFFUSION_CONFIG)

    # Build dataset, then wrap in DistributedSampler if needed
    dataset = get_training_dataset(TRAIN_CONFIG)

    sampler = None
    shuffle = True

    if _is_distributed():
        sampler = DistributedSampler(dataset, shuffle=True)
        shuffle = False  # sampler handles shuffling

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=TRAIN_CONFIG["num_workers"],
        pin_memory=True,
    )

    model = UNet(MODEL_CONFIG).to(device)

    if _is_distributed():
        model = DDP(model, device_ids=[_local_rank()])

    optimizer = Adam(model.parameters(), lr=TRAIN_CONFIG["lr"])

    criterion = nn.MSELoss()

    train(
        model=model,
        dataloader=dataloader,
        sampler=sampler,
        scheduler=scheduler,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=TRAIN_CONFIG["num_epochs"],
        device=device,
        save_path=checkpoint_path(),
        dataset_mode=TRAIN_CONFIG["dataset_mode"],
    )

    _cleanup_distributed()


if __name__ == "__main__":
    main()