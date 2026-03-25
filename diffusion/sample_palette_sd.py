"""
Palette-SD restoration with full metric evaluation.

For each image:
  1. Encode measurement -> latent, run conditional DDIM, decode to pixels
  2. Evaluate against ground truth using eval_single.eval_image_pair():
     PSNR (dB), MS-SSIM, LPIPS
  3. Save per-image metrics CSV and comparison images
  4. Print aggregate statistics at the end

The eval_image_pair function expects images as 3D tensors (no batch dim)
in [0,1] float, CHW or HWC format.  Our diffusion outputs are [-1,1]
with batch dim, so we convert via (x+1)/2 and squeeze.
"""

from __future__ import annotations

import os
import csv
import statistics

import torch
from tqdm import tqdm
from diffusers import DDIMScheduler

from config import (
    SD_PALETTE_MODEL_CONFIG,
    SD_PALETTE_SAMPLE_CONFIG,
    RESTORATION_DATA_CONFIG,
    sd_palette_checkpoint_dir,
)
from dataset import get_restoration_dataloader
from sd_utils import load_palette_sd, encode_to_latent, decode_from_latent
from eval_single import eval_image_pair
from utils import save_comparison


def _to_eval_format(x: torch.Tensor) -> torch.Tensor:
    """
    Convert a single image from diffusion format to eval_single format.

    Input:  (B, 3, H, W) in [-1, 1], float16 or float32
    Output: (3, H, W) in [0, 1], float32, no batch dim
    """
    img = x[0].float()           # remove batch dim, ensure float32
    img = (img + 1.0) / 2.0      # [-1,1] -> [0,1]
    img = img.clamp(0.0, 1.0)
    return img


@torch.no_grad()
def run_sd_palette():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_id = SD_PALETTE_MODEL_CONFIG["sd_model_id"]
    ckpt_dir = sd_palette_checkpoint_dir()

    # ── Load trained model ────────────────────────────────────────────────
    vae, unet, null_embeds = load_palette_sd(model_id, ckpt_dir, device)
    unet.eval()
    print("Palette-SD model loaded.")

    # ── DDIM scheduler ────────────────────────────────────────────────────
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    num_steps = SD_PALETTE_SAMPLE_CONFIG["num_steps"]
    eta = SD_PALETTE_SAMPLE_CONFIG["eta"]
    scheduler.set_timesteps(num_steps, device=device)
    print(f"DDIM: {num_steps} steps, eta={eta}")

    # ── Data ──────────────────────────────────────────────────────────────
    dataloader = get_restoration_dataloader(RESTORATION_DATA_CONFIG)

    out_dir = SD_PALETTE_SAMPLE_CONFIG["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # ── Sampling + evaluation loop ────────────────────────────────────────
    all_psnr = []
    all_msssim = []
    all_lpips = []

    metrics_file = os.path.join(out_dir, "metrics.csv")

    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "psnr", "ms_ssim", "lpips"])

        for i, (measurement, target) in enumerate(tqdm(dataloader)):

            measurement = measurement.to(device, dtype=torch.float16)

            # ── Encode measurement ────────────────────────────────────
            z_meas = encode_to_latent(vae, measurement, deterministic=True)

            # ── Reverse diffusion ─────────────────────────────────────
            z = torch.randn_like(z_meas)
            encoder_hidden_states = null_embeds.expand(z.shape[0], -1, -1)

            for t in scheduler.timesteps:
                z_input = torch.cat([z, z_meas], dim=1)
                noise_pred = unet(
                    z_input,
                    t.unsqueeze(0).expand(z.shape[0]),
                    encoder_hidden_states=encoder_hidden_states,
                ).sample
                z = scheduler.step(noise_pred, t, z, eta=eta).prev_sample

            # ── Decode to pixel space ─────────────────────────────────
            restored = decode_from_latent(vae, z, original_size=(800, 800))

            # ── Evaluate metrics ──────────────────────────────────────
            if target is not None:
                target = target.to(device)

                # Convert from [-1,1] with batch to [0,1] without batch
                gt_eval = _to_eval_format(target)
                pred_eval = _to_eval_format(restored)

                psnr_val, msssim_val, lpips_val = eval_image_pair(
                    gt_eval, pred_eval, device=device,
                )

                all_psnr.append(psnr_val)
                all_msssim.append(msssim_val)
                all_lpips.append(lpips_val)

                writer.writerow([
                    i,
                    f"{psnr_val:.4f}",
                    f"{msssim_val:.6f}",
                    f"{lpips_val:.6f}",
                ])

                # Save comparison image
                save_comparison(
                    measurement.float(),
                    restored.float(),
                    target.float(),
                    os.path.join(out_dir, f"{i:04d}_comparison.png"),
                )

                if i < 5 or i % 100 == 0:
                    print(
                        f"  [{i:4d}] PSNR: {psnr_val:.2f} dB | "
                        f"MS-SSIM: {msssim_val:.4f} | "
                        f"LPIPS: {lpips_val:.4f}"
                    )
            else:
                from torchvision.utils import save_image
                save_image(
                    (restored.float() + 1) / 2,
                    os.path.join(out_dir, f"{i:04d}_restored.png"),
                )
                writer.writerow([i, "N/A", "N/A", "N/A"])

    # ── Print summary ─────────────────────────────────────────────────────
    n = len(all_psnr)
    if n > 0:
        print(f"\n{'='*60}")
        print(f"  Palette-SD Results ({n} images)")
        print(f"{'='*60}")
        print(f"  Mean PSNR:    {statistics.mean(all_psnr):.4f} dB")
        print(f"  Mean MS-SSIM: {statistics.mean(all_msssim):.6f}")
        print(f"  Mean LPIPS:   {statistics.mean(all_lpips):.6f}")
        print(f"{'='*60}")
        print(f"  Median PSNR:    {statistics.median(all_psnr):.4f} dB")
        print(f"  Median MS-SSIM: {statistics.median(all_msssim):.6f}")
        print(f"  Median LPIPS:   {statistics.median(all_lpips):.6f}")
        print(f"{'='*60}")
        if n > 1:
            print(f"  Std PSNR:    {statistics.stdev(all_psnr):.4f}")
            print(f"  Std MS-SSIM: {statistics.stdev(all_msssim):.6f}")
            print(f"  Std LPIPS:   {statistics.stdev(all_lpips):.6f}")
        print(f"  Min PSNR:  {min(all_psnr):.4f} | Max PSNR: {max(all_psnr):.4f}")
        print(f"{'='*60}\n")

    print(f"Per-image metrics: {metrics_file}")
    print(f"Comparison images: {out_dir}/")


if __name__ == "__main__":
    run_sd_palette()