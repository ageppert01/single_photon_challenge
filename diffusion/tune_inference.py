"""
Inference-time hyperparameter sweep.

Loads the trained model once, then evaluates every combination of:
  - Scheduler: DDIM, DPM-Solver++, Euler
  - Steps:     configurable grid
  - Eta:       DDIM stochasticity (only applies to DDIM)
  - Avg:       multi-sample averaging (only useful when eta > 0)

on a small validation subset.  Results are written to a CSV sorted by
the composite metric (PSNR/40 + MS-SSIM − LPIPS).

Usage:
    python tune_inference.py                      # defaults
    python tune_inference.py --num-images 30      # more images (slower)
    python tune_inference.py --best               # use best checkpoint
    python tune_inference.py --quick               # minimal 5-image smoke test
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import statistics
import time

import torch
from torch.cuda.amp import autocast
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

from config import (
    FULL_DATASET_CONFIG,
    SD_PALETTE_MODEL_CONFIG,
    sd_palette_checkpoint_dir,
    sd_palette_best_checkpoint_dir,
)
from dataset import get_training_dataset
from eval_single import eval_image_pair
from sd_utils import (
    load_palette_sd,
    encode_measurement,
    decode_from_latent,
)
from utils import seed_everything


# ── Schedulers ────────────────────────────────────────────────────────────────

from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
)

SCHEDULER_REGISTRY = {
    "ddim": DDIMScheduler,
    "dpm++": DPMSolverMultistepScheduler,
    "euler": EulerDiscreteScheduler,
}


# ── Inference with pluggable scheduler ────────────────────────────────────────


@torch.no_grad()
def inference_with_scheduler(
    unet,
    meas_vae,
    vae,
    null_embeds,
    z_meas,
    scheduler,
    device,
    eta=0.0,
):
    """
    Run reverse diffusion with a pre-configured scheduler.

    Unlike sd_palette_inference, this takes z_meas (already encoded)
    and a scheduler object, avoiding redundant encoding and scheduler
    construction across sweep iterations.
    """
    z = torch.randn_like(z_meas)
    encoder_hidden_states = null_embeds.expand(z.shape[0], -1, -1)

    for t in scheduler.timesteps:
        z_input = torch.cat([z, z_meas], dim=1)
        with autocast(enabled=True):
            pred = unet(
                z_input,
                t.unsqueeze(0).expand(z.shape[0]),
                encoder_hidden_states=encoder_hidden_states,
            ).sample

        step_kwargs = {}
        # Only DDIMScheduler accepts eta
        if isinstance(scheduler, DDIMScheduler):
            step_kwargs["eta"] = eta

        z = scheduler.step(pred, t, z, **step_kwargs).prev_sample

    return decode_from_latent(vae, z, original_size=(800, 800))


# ── Composite metric ─────────────────────────────────────────────────────────


def composite(psnr, ms_ssim, lpips):
    return psnr / 40.0 + ms_ssim - lpips


# ── Main ──────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Inference hyperparameter sweep")
    p.add_argument("--best", action="store_true", help="Use best checkpoint")
    p.add_argument(
        "--num-images", type=int, default=20,
        help="Number of validation images to evaluate (default: 20)",
    )
    p.add_argument("--quick", action="store_true", help="Smoke test: 5 images, tiny grid")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output", type=str, default="tune_results.csv",
        help="Output CSV path (default: tune_results.csv)",
    )
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Define sweep grid ─────────────────────────────────────────────────
    if args.quick:
        steps_grid = [20, 50]
        eta_grid = [0.0, 0.3]
        avg_grid = [1]
        sched_grid = ["ddim"]
        args.num_images = min(args.num_images, 5)
    else:
        steps_grid = [10, 15, 20, 25, 30, 50]
        eta_grid = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
        avg_grid = [1, 2, 4]
        sched_grid = ["ddim", "dpm++", "euler"]

    # Build configs: for non-DDIM schedulers, eta is ignored (fixed 0);
    # for deterministic (eta=0), avg > 1 is pointless.
    configs = []
    for sched_name in sched_grid:
        for steps in steps_grid:
            if sched_name == "ddim":
                for eta in eta_grid:
                    if eta == 0.0:
                        configs.append((sched_name, steps, 0.0, 1))
                    else:
                        for avg in avg_grid:
                            configs.append((sched_name, steps, eta, avg))
            else:
                # Non-DDIM: eta not applicable, deterministic → avg=1
                configs.append((sched_name, steps, 0.0, 1))

    print(f"Sweep: {len(configs)} configurations × {args.num_images} images")

    # ── Load model (once) ─────────────────────────────────────────────────
    model_id = SD_PALETTE_MODEL_CONFIG["sd_model_id"]
    use_qvae = SD_PALETTE_MODEL_CONFIG.get("use_gqir_qvae", False)
    ckpt_dir = sd_palette_best_checkpoint_dir() if args.best else sd_palette_checkpoint_dir()

    meas_vae, vae, unet, null_embeds = load_palette_sd(
        model_id, ckpt_dir, device, use_gqir_qvae=use_qvae,
    )
    unet.eval()
    print(f"Model loaded from {ckpt_dir}")

    # ── Load validation subset ────────────────────────────────────────────
    ds_cfg = {**FULL_DATASET_CONFIG, "batch_size": 1, "num_workers": 2}
    full_dataset = get_training_dataset(ds_cfg)

    val_size = 185
    train_size = len(full_dataset) - val_size
    _, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    # Take a reproducible subset
    num_eval = min(args.num_images, len(val_dataset))
    eval_subset, _ = random_split(
        val_dataset,
        [num_eval, len(val_dataset) - num_eval],
        generator=torch.Generator().manual_seed(args.seed + 1),
    )
    eval_loader = DataLoader(eval_subset, batch_size=1, shuffle=False, num_workers=2)

    # ── Pre-encode all measurements (shared across configs) ───────────────
    print(f"Pre-encoding {num_eval} measurements ...")
    encoded_data = []
    for measurement, target in eval_loader:
        if target is None:
            continue
        measurement = measurement.to(device, dtype=torch.float16)
        z_meas = encode_measurement(meas_vae, vae, measurement)
        gt = ((target[0].float() + 1) / 2).clamp(0, 1)
        encoded_data.append((z_meas, gt))
    print(f"  {len(encoded_data)} images encoded.")

    # ── Sweep ─────────────────────────────────────────────────────────────
    results = []

    for cfg_idx, (sched_name, steps, eta, avg) in enumerate(configs):
        label = f"{sched_name}/steps={steps}/eta={eta}/avg={avg}"
        print(f"\n[{cfg_idx+1}/{len(configs)}] {label}")

        # Build scheduler
        sched_cls = SCHEDULER_REGISTRY[sched_name]
        scheduler = sched_cls.from_pretrained(model_id, subfolder="scheduler")
        scheduler.set_timesteps(steps, device=device)

        all_psnr, all_msssim, all_lpips = [], [], []
        t0 = time.time()

        for z_meas, gt in tqdm(encoded_data, desc=f"  {label}", leave=False):
            accum = None
            for _ in range(avg):
                restored = inference_with_scheduler(
                    unet, meas_vae, vae, null_embeds,
                    z_meas, scheduler, device, eta=eta,
                )
                sample = restored.float()
                accum = sample if accum is None else accum + sample

            pred = (accum / avg).clamp(-1, 1)
            pred_eval = ((pred[0] + 1) / 2).clamp(0, 1)

            psnr_v, msssim_v, lpips_v = eval_image_pair(
                gt, pred_eval, device=device,
            )
            all_psnr.append(psnr_v)
            all_msssim.append(msssim_v)
            all_lpips.append(lpips_v)

        elapsed = time.time() - t0
        mean_p = statistics.mean(all_psnr)
        mean_m = statistics.mean(all_msssim)
        mean_l = statistics.mean(all_lpips)
        comp = composite(mean_p, mean_m, mean_l)

        # Percentiles (5% low = worst 5%)
        sorted_psnr = sorted(all_psnr)
        n = len(sorted_psnr)
        p5_idx = max(0, int(n * 0.05) - 1)
        p1_idx = 0
        p5_psnr = sorted_psnr[p5_idx] if n > 1 else sorted_psnr[0]
        p1_psnr = sorted_psnr[p1_idx]

        row = {
            "scheduler": sched_name,
            "steps": steps,
            "eta": eta,
            "avg_samples": avg,
            "psnr": mean_p,
            "ms_ssim": mean_m,
            "lpips": mean_l,
            "composite": comp,
            "psnr_p5": p5_psnr,
            "psnr_p1": p1_psnr,
            "time_sec": elapsed,
        }
        results.append(row)

        print(
            f"  PSNR {mean_p:.2f} | MS-SSIM {mean_m:.4f} | "
            f"LPIPS {mean_l:.4f} | Composite {comp:.4f} | "
            f"{elapsed:.1f}s"
        )

    # ── Sort and write CSV ────────────────────────────────────────────────
    results.sort(key=lambda r: r["composite"], reverse=True)

    fieldnames = [
        "scheduler", "steps", "eta", "avg_samples",
        "psnr", "ms_ssim", "lpips", "composite",
        "psnr_p5", "psnr_p1", "time_sec",
    ]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                k: f"{v:.4f}" if isinstance(v, float) else v
                for k, v in r.items()
            })

    # ── Print top results ─────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  TOP 10 CONFIGURATIONS (by composite, {num_eval} images)")
    print(f"{'='*80}")
    print(f"  {'Scheduler':<8} {'Steps':>5} {'Eta':>5} {'Avg':>4}  "
          f"{'PSNR':>7} {'MS-SSIM':>8} {'LPIPS':>7} {'Comp':>7} {'Time':>6}")
    print(f"  {'-'*72}")
    for r in results[:10]:
        print(
            f"  {r['scheduler']:<8} {r['steps']:>5} {r['eta']:>5.1f} "
            f"{r['avg_samples']:>4}  {r['psnr']:>7.2f} {r['ms_ssim']:>8.4f} "
            f"{r['lpips']:>7.4f} {r['composite']:>7.4f} {r['time_sec']:>5.1f}s"
        )
    print(f"{'='*80}")
    print(f"\nFull results: {args.output}")

    # ── Print recommended config ──────────────────────────────────────────
    best = results[0]
    print(f"\nRecommended config:")
    print(f"  Scheduler:   {best['scheduler']}")
    print(f"  Steps:       {best['steps']}")
    print(f"  Eta:         {best['eta']}")
    print(f"  Avg samples: {best['avg_samples']}")
    print(f"  Composite:   {best['composite']:.4f}")


if __name__ == "__main__":
    main()