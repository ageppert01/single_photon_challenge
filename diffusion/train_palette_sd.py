"""
Palette-style conditional diffusion in Stable Diffusion's latent space.

Supports multi-GPU training via DDP (launch with torchrun).

Pipeline per training step:
  1. Encode measurement -> z_meas  (frozen VAE, deterministic)
  2. Encode target     -> z_target (frozen VAE, sampled)
  3. Add noise to z_target at random timestep
  4. UNet predicts noise from cat(z_noisy, z_meas) + null text embedding
  5. MSE loss between predicted and true noise

Only the LoRA adapters and the expanded conv_in are trainable.
Everything else (VAE, text encoder, base UNet weights) is frozen.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from config import (
    DEVICE,
    SD_PALETTE_TRAIN_CONFIG,
    SD_PALETTE_MODEL_CONFIG,
    FULL_DATASET_CONFIG,
    sd_palette_checkpoint_dir,
)
from dataset import get_training_dataset
from sd_utils import (
    load_sd_components,
    save_palette_sd,
    load_palette_sd,
    encode_to_latent,
)
from utils import ensure_dir, seed_everything


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


def main() -> None:

    device = _setup_distributed()
    cfg = SD_PALETTE_TRAIN_CONFIG
    seed_everything(cfg["seed"])

    task_dir = cfg["task_name"]
    if _is_main_process():
        ensure_dir(task_dir)

    if _is_distributed():
        dist.barrier()

    # ── Load SD components ────────────────────────────────────────────────
    vae, unet, null_embeds, noise_scheduler = load_sd_components(
        model_id=SD_PALETTE_MODEL_CONFIG["sd_model_id"],
        device=device,
        lora_rank=SD_PALETTE_MODEL_CONFIG["lora_rank"],
        lora_alpha=SD_PALETTE_MODEL_CONFIG["lora_alpha"],
        dtype=torch.float16,
    )

    # Upcast trainable params (LoRA + conv_in) to fp32 for stable training
    for p in unet.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    # ── Dataset with optional DistributedSampler ──────────────────────────
    ds_cfg = {
        "dataset_mode": "full",
        **{k: v for k, v in FULL_DATASET_CONFIG.items()},
        "batch_size": cfg["batch_size"],
        "num_workers": cfg["num_workers"],
    }
    dataset = get_training_dataset(ds_cfg)

    sampler = None
    shuffle = True
    if _is_distributed():
        sampler = DistributedSampler(dataset, shuffle=True)
        shuffle = False

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    # ── Optionally resume ─────────────────────────────────────────────────
    save_dir = sd_palette_checkpoint_dir()
    start_epoch = 0

    resume_path = os.path.join(save_dir, "training_state.pth")
    if os.path.isfile(resume_path):
        if _is_main_process():
            print(f"Resuming from {resume_path}")
        state = torch.load(resume_path, map_location="cpu")
        start_epoch = state["epoch"] + 1

        _vae, unet, _null = load_palette_sd(
            SD_PALETTE_MODEL_CONFIG["sd_model_id"],
            save_dir, device,
        )
        unet.train()

    # ── Wrap in DDP ───────────────────────────────────────────────────────
    if _is_distributed():
        unet = DDP(unet, device_ids=[_local_rank()], find_unused_parameters=True)

    # ── Optimizer ─────────────────────────────────────────────────────────
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg["lr"], weight_decay=1e-2)

    if os.path.isfile(resume_path):
        state = torch.load(resume_path, map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        if _is_main_process():
            print(f"  Resuming from epoch {start_epoch}")

    scaler = GradScaler(enabled=cfg["use_amp"])
    accumulation = cfg["gradient_accumulation"]

    # ── Training loop ─────────────────────────────────────────────────────

    num_train_timesteps = noise_scheduler.config.num_train_timesteps

    for epoch in range(start_epoch, cfg["num_epochs"]):

        if sampler is not None:
            sampler.set_epoch(epoch)

        unet.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(dataloader, disable=not _is_main_process(),
                    desc=f"Epoch {epoch+1}/{cfg['num_epochs']}")

        for step, batch in enumerate(pbar):

            measurement, target = batch

            if target is None:
                continue

            measurement = measurement.to(device, dtype=torch.float16)
            target = target.to(device, dtype=torch.float16)

            # ── Encode to latent space ────────────────────────────────
            with torch.no_grad():
                z_meas = encode_to_latent(vae, measurement, deterministic=True)
                z_target = encode_to_latent(vae, target, deterministic=False)

            # ── Forward diffusion on target latent ────────────────────
            noise = torch.randn_like(z_target)
            timesteps = torch.randint(
                0, num_train_timesteps, (z_target.shape[0],), device=device,
            ).long()

            z_noisy = noise_scheduler.add_noise(z_target, noise, timesteps)

            # ── Palette: concatenate condition ────────────────────────
            z_input = torch.cat([z_noisy, z_meas], dim=1)

            # ── Predict noise ─────────────────────────────────────────
            with autocast(enabled=cfg["use_amp"]):
                encoder_hidden_states = null_embeds.expand(
                    z_input.shape[0], -1, -1
                )

                # Handle DDP-wrapped vs bare model
                model_fn = unet.module if _is_distributed() else unet
                noise_pred = model_fn(
                    z_input,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

                loss = F.mse_loss(noise_pred, noise)
                loss = loss / accumulation

            scaler.scale(loss).backward()

            if (step + 1) % accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * accumulation
            pbar.set_postfix(loss=f"{loss.item() * accumulation:.6f}")

        # Flush remaining gradients
        if len(dataloader) % accumulation != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        epoch_loss /= max(len(dataloader), 1)

        if _is_main_process():
            print(f"Epoch {epoch+1}/{cfg['num_epochs']} | Loss {epoch_loss:.6f}")

            raw_unet = unet.module if _is_distributed() else unet
            save_palette_sd(raw_unet, save_dir)
            torch.save(
                {"epoch": epoch, "optimizer": optimizer.state_dict()},
                os.path.join(save_dir, "training_state.pth"),
            )

    _cleanup_distributed()
    if _is_main_process():
        print("Training complete.")


if __name__ == "__main__":
    main()