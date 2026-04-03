"""
Palette-style conditional diffusion in Stable Diffusion latent space.

Unified version:
  - SD 1.5 (epsilon-prediction) or SD 2.1 (v-prediction): auto-detected
  - Optional gQIR qVAE for measurement encoding: controlled by config
  - Selective unfreezing, DDP, LR warmup + cosine decay
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
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
    encode_to_latent,
    encode_measurement,
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


# ── LR schedule ──────────────────────────────────────────────────────────────


def _get_cosine_schedule_with_warmup(
    optimizer, warmup_steps, total_steps, min_lr_ratio=0.1,
) -> LambdaLR:
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


# ── Training ─────────────────────────────────────────────────────────────────


def main() -> None:

    device = _setup_distributed()
    cfg = SD_PALETTE_TRAIN_CONFIG
    model_cfg = SD_PALETTE_MODEL_CONFIG
    seed_everything(cfg["seed"])

    task_dir = cfg["task_name"]
    if _is_main_process():
        ensure_dir(task_dir)

    if _is_distributed():
        dist.barrier()

    # ── Load SD components ────────────────────────────────────────────────
    meas_vae, vae, unet, null_embeds, noise_scheduler = load_sd_components(
        model_id=model_cfg["sd_model_id"],
        device=device,
        use_gqir_qvae=model_cfg.get("use_gqir_qvae", False),
        dtype=torch.float16,
    )

    # Upcast trainable params to fp32
    for p in unet.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    # ── Dataset ───────────────────────────────────────────────────────────
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
    global_step = 0

    resume_path = os.path.join(save_dir, "training_state.pth")
    if os.path.isfile(resume_path):
        if _is_main_process():
            print(f"Resuming from {resume_path}")
        resume_state = torch.load(resume_path, map_location="cpu")
        start_epoch = resume_state["epoch"] + 1
        global_step = resume_state.get("global_step", 0)

        ckpt_path = os.path.join(save_dir, "unet.pth")
        state = torch.load(ckpt_path, map_location="cpu")
        unet.load_state_dict(state)
        del state
        for p in unet.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        unet.train()

    # ── Wrap in DDP ───────────────────────────────────────────────────────
    if _is_distributed():
        unet = DDP(unet, device_ids=[_local_rank()], find_unused_parameters=False)

    # ── Optimizer + scheduler ─────────────────────────────────────────────
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg["lr"], weight_decay=1e-2)

    accumulation = cfg["gradient_accumulation"]
    steps_per_epoch = len(dataloader) // accumulation
    total_steps = steps_per_epoch * cfg["num_epochs"]
    warmup_steps = cfg.get("warmup_steps", 50)

    scheduler = _get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if os.path.isfile(resume_path):
        optimizer.load_state_dict(resume_state["optimizer"])
        if "scheduler" in resume_state:
            scheduler.load_state_dict(resume_state["scheduler"])
        for _ in range(global_step):
            scheduler.step()
        if _is_main_process():
            print(f"  Resumed from epoch {start_epoch}, step {global_step}")

    scaler = GradScaler(enabled=cfg["use_amp"])

    # ── Training loop ─────────────────────────────────────────────────────

    num_train_timesteps = noise_scheduler.config.num_train_timesteps
    is_v_prediction = noise_scheduler.config.prediction_type == "v_prediction"

    if _is_main_process():
        print(f"  v-prediction: {is_v_prediction}")
        print(f"  Using gQIR qVAE: {meas_vae is not None}")
        print(f"  Steps/epoch: {steps_per_epoch}, Total steps: {total_steps}")

    for epoch in range(start_epoch, cfg["num_epochs"]):

        if sampler is not None:
            sampler.set_epoch(epoch)

        unet.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(dataloader, disable=not _is_main_process(),
                    desc=f"Epoch {epoch+1}/{cfg['num_epochs']}")

        for step, batch in enumerate(pbar):

            measurement_batch, target = batch
            if target is None:
                continue

            measurement_batch = measurement_batch.to(device, dtype=torch.float16)
            target = target.to(device, dtype=torch.float16)

            # ── Encode to latent space ────────────────────────────────
            with torch.no_grad():
                z_meas = encode_measurement(meas_vae, vae, measurement_batch)
                z_target = encode_to_latent(vae, target, deterministic=False)

            # ── Forward diffusion ─────────────────────────────────────
            noise = torch.randn_like(z_target)
            timesteps = torch.randint(
                0, num_train_timesteps, (z_target.shape[0],), device=device,
            ).long()

            z_noisy = noise_scheduler.add_noise(z_target, noise, timesteps)

            # ── Palette: concatenate condition ────────────────────────
            z_input = torch.cat([z_noisy, z_meas], dim=1)

            # ── Compute target (auto v-prediction or epsilon) ─────────
            if is_v_prediction:
                target_pred = noise_scheduler.get_velocity(z_target, noise, timesteps)
            else:
                target_pred = noise

            # ── Predict ───────────────────────────────────────────────
            with autocast(enabled=cfg["use_amp"]):
                encoder_hidden_states = null_embeds.expand(
                    z_input.shape[0], -1, -1
                )

                model_pred = unet(
                    z_input,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

                loss = F.mse_loss(model_pred, target_pred)
                loss = loss / accumulation

            scaler.scale(loss).backward()

            if (step + 1) % accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            epoch_loss += loss.item() * accumulation
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{loss.item() * accumulation:.6f}", lr=f"{current_lr:.2e}")

        # Flush remaining gradients
        if len(dataloader) % accumulation != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

        epoch_loss /= max(len(dataloader), 1)

        if _is_main_process():
            print(
                f"Epoch {epoch+1}/{cfg['num_epochs']} | "
                f"Loss {epoch_loss:.6f} | "
                f"LR {current_lr:.2e} | "
                f"Step {global_step}"
            )

            raw_unet = unet.module if _is_distributed() else unet
            save_palette_sd(raw_unet, save_dir)
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                os.path.join(save_dir, "training_state.pth"),
            )

    _cleanup_distributed()
    if _is_main_process():
        print("Training complete.")


if __name__ == "__main__":
    main()