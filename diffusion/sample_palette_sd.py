"""Palette-SD restoration with full metric evaluation."""

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
from sd_utils import load_palette_sd, encode_measurement, decode_from_latent
from eval_single import eval_image_pair
from utils import save_comparison


def _to_eval_format(x: torch.Tensor) -> torch.Tensor:
    img = x[0].float()
    img = (img + 1.0) / 2.0
    img = img.clamp(0.0, 1.0)
    return img


@torch.no_grad()
def run_sd_palette():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_id = SD_PALETTE_MODEL_CONFIG["sd_model_id"]
    use_qvae = SD_PALETTE_MODEL_CONFIG.get("use_gqir_qvae", False)
    ckpt_dir = sd_palette_checkpoint_dir()

    meas_vae, vae, unet, null_embeds = load_palette_sd(
        model_id, ckpt_dir, device, use_gqir_qvae=use_qvae,
    )
    unet.eval()
    print(f"Palette-SD model loaded (qVAE: {meas_vae is not None}).")

    # DDIM (prediction type auto-detected)
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    num_steps = SD_PALETTE_SAMPLE_CONFIG["num_steps"]
    eta = SD_PALETTE_SAMPLE_CONFIG["eta"]
    scheduler.set_timesteps(num_steps, device=device)
    print(f"DDIM: {num_steps} steps, eta={eta}, prediction={scheduler.config.prediction_type}")

    dataloader = get_restoration_dataloader(RESTORATION_DATA_CONFIG)

    out_dir = SD_PALETTE_SAMPLE_CONFIG["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    all_psnr = []
    all_msssim = []
    all_lpips = []

    metrics_file = os.path.join(out_dir, "metrics.csv")

    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "psnr", "ms_ssim", "lpips"])

        for i, (measurement, target) in enumerate(tqdm(dataloader)):

            measurement = measurement.to(device, dtype=torch.float16)

            # Encode measurement (qVAE if available, else standard VAE)
            z_meas = encode_measurement(meas_vae, vae, measurement)

            # Reverse diffusion
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

            # Decode with standard VAE
            restored = decode_from_latent(vae, z, original_size=(800, 800))

            # Evaluate
            if target is not None:
                target = target.to(device)

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

    n = len(all_psnr)
    if n > 0:
        print(f"\n{'='*60}")
        print(f"  Palette-SD Results ({n} images)")
        print(f"{'='*60}")
        print(f"  Mean PSNR:    {statistics.mean(all_psnr):.4f} dB")
        print(f"  Mean MS-SSIM: {statistics.mean(all_msssim):.6f}")
        print(f"  Mean LPIPS:   {statistics.mean(all_lpips):.6f}")
        print(f"{'='*60}")
        if n > 1:
            print(f"  Median PSNR:    {statistics.median(all_psnr):.4f} dB")
            print(f"  Std PSNR:       {statistics.stdev(all_psnr):.4f}")
            print(f"  Min PSNR:       {min(all_psnr):.4f} | Max: {max(all_psnr):.4f}")
        print(f"{'='*60}\n")

    print(f"Per-image metrics: {metrics_file}")
    print(f"Comparison images: {out_dir}/")


if __name__ == "__main__":
    run_sd_palette()