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
import statistics

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from diffusers import DDIMScheduler

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
    decode_from_latent,
)
from eval_single import eval_image_pair
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


# ── Validation ───────────────────────────────────────────────────────────────


def _compute_composite(psnr: float, ms_ssim: float, lpips: float) -> float:
    """
    Composite validation metric (higher = better).

    Normalizes PSNR to ~[0,1] range so all three terms contribute roughly
    equally: PSNR/40 + MS-SSIM - LPIPS.

    Example: 31 dB PSNR, 0.92 MS-SSIM, 0.12 LPIPS → 1.575
    """
    return psnr / 40.0 + ms_ssim - lpips


@torch.no_grad()
def _validate(
    unet: torch.nn.Module,
    meas_vae,
    vae,
    null_embeds: torch.Tensor,
    val_dataloader: torch.utils.data.DataLoader,
    model_id: str,
    device: torch.device,
    num_steps: int = 20,
):
    """
    Run DDIM inference on the validation set and compute metrics.

    Uses fewer DDIM steps than full evaluation for speed — the relative
    ranking between checkpoints is stable even at 20 steps.

    Returns:
        (mean_psnr, mean_ms_ssim, mean_lpips, composite)
    """
    unet.eval()

    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    scheduler.set_timesteps(num_steps, device=device)

    all_psnr, all_msssim, all_lpips = [], [], []

    for measurement, target in tqdm(val_dataloader, desc="Validation"):
        if target is None:
            continue

        measurement = measurement.to(device, dtype=torch.float16)
        z_meas = encode_measurement(meas_vae, vae, measurement)

        # Reverse diffusion from noise
        z = torch.randn_like(z_meas)
        enc_hidden = null_embeds.expand(z.shape[0], -1, -1)

        for t in scheduler.timesteps:
            z_input = torch.cat([z, z_meas], dim=1)
            with autocast(enabled=True):
                pred = unet(
                    z_input,
                    t.unsqueeze(0).expand(z.shape[0]),
                    encoder_hidden_states=enc_hidden,
                ).sample
            z = scheduler.step(pred, t, z, eta=0.0).prev_sample

        restored = decode_from_latent(vae, z, original_size=(800, 800))

        # Convert [-1, 1] → [0, 1] for metrics
        gt_eval = ((target[0].float() + 1) / 2).clamp(0, 1)
        pred_eval = ((restored[0].float() + 1) / 2).clamp(0, 1)

        psnr_val, msssim_val, lpips_val = eval_image_pair(
            gt_eval, pred_eval, device=device,
        )
        all_psnr.append(psnr_val)
        all_msssim.append(msssim_val)
        all_lpips.append(lpips_val)

    n = len(all_psnr)
    mean_psnr = statistics.mean(all_psnr)
    mean_msssim = statistics.mean(all_msssim)
    mean_lpips = statistics.mean(all_lpips)
    composite = _compute_composite(mean_psnr, mean_msssim, mean_lpips)

    print(f"\n{'='*60}")
    print(f"  Validation ({n} images, {num_steps} DDIM steps)")
    print(f"{'='*60}")
    print(f"  Mean PSNR:    {mean_psnr:.4f} dB")
    print(f"  Mean MS-SSIM: {mean_msssim:.6f}")
    print(f"  Mean LPIPS:   {mean_lpips:.6f}")
    print(f"  Composite:    {composite:.6f}")
    print(f"{'='*60}\n")

    return mean_psnr, mean_msssim, mean_lpips, composite


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
    full_dataset = get_training_dataset(ds_cfg)

    # ── Train / validation split ──────────────────────────────────────────
    val_size = cfg.get("val_size", 185)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )

    if _is_main_process():
        print(f"  Dataset split: {train_size} train, {val_size} val")

    sampler = None
    shuffle = True
    if _is_distributed():
        sampler = DistributedSampler(train_dataset, shuffle=True)
        shuffle = False

    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    # Validation dataloader (main process only, batch_size=1)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["num_workers"],
    )

    # ── Optionally resume ─────────────────────────────────────────────────
    save_dir = sd_palette_checkpoint_dir()
    start_epoch = 0
    global_step = 0
    best_composite = -float("inf")
    patience_counter = 0

    resume_path = os.path.join(save_dir, "training_state.pth")
    if os.path.isfile(resume_path):
        if _is_main_process():
            print(f"Resuming from {resume_path}")
        resume_state = torch.load(resume_path, map_location="cpu")
        start_epoch = resume_state["epoch"] + 1
        global_step = resume_state.get("global_step", 0)
        best_composite = resume_state.get("best_composite", -float("inf"))
        patience_counter = resume_state.get("patience_counter", 0)

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

            # ── Validation ────────────────────────────────────────────
            val_every = cfg.get("val_every_epochs", 100)
            is_val_epoch = (epoch + 1) % val_every == 0 or (epoch + 1) == cfg["num_epochs"]

            if is_val_epoch:
                val_psnr, val_msssim, val_lpips, composite = _validate(
                    unet=raw_unet,
                    meas_vae=meas_vae,
                    vae=vae,
                    null_embeds=null_embeds,
                    val_dataloader=val_dataloader,
                    model_id=model_cfg["sd_model_id"],
                    device=device,
                    num_steps=cfg.get("val_num_steps", 20),
                )

                if composite > best_composite:
                    best_composite = composite
                    patience_counter = 0
                    # Save best checkpoint separately
                    save_palette_sd(raw_unet, save_dir + "_best")
                    torch.save(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "composite": composite,
                            "psnr": val_psnr,
                            "ms_ssim": val_msssim,
                            "lpips": val_lpips,
                        },
                        os.path.join(save_dir + "_best", "best_metrics.pth"),
                    )
                    print(f"  >>> New best composite: {composite:.6f} (saved)")
                else:
                    patience_counter += 1
                    print(
                        f"  No improvement ({patience_counter}/"
                        f"{cfg.get('early_stopping_patience', 5)}). "
                        f"Best: {best_composite:.6f}"
                    )

            # ── Save training state ───────────────────────────────────
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_composite": best_composite,
                    "patience_counter": patience_counter,
                },
                os.path.join(save_dir, "training_state.pth"),
            )

        # ── Early stopping (DDP-safe: broadcast decision to all ranks) ──
        patience_limit = cfg.get("early_stopping_patience", 5)
        should_stop = False

        if _is_main_process():
            should_stop = patience_counter >= patience_limit

        if _is_distributed():
            stop_flag = torch.tensor(
                [1 if should_stop else 0], device=device, dtype=torch.long,
            )
            dist.broadcast(stop_flag, src=0)
            should_stop = stop_flag.item() == 1

        if should_stop:
            val_every = cfg.get("val_every_epochs", 100)
            if _is_main_process():
                print(
                    f"\nEarly stopping: no improvement for "
                    f"{patience_limit} validation rounds "
                    f"({patience_limit * val_every} epochs).\n"
                    f"Best composite: {best_composite:.6f}\n"
                    f"Best checkpoint: {save_dir}_best/"
                )
            break

    _cleanup_distributed()
    if _is_main_process():
        print("Training complete.")
        print(f"Best checkpoint: {save_dir}_best/")
        print(f"Best composite:  {best_composite:.6f}")


if __name__ == "__main__":
    main()