"""
Palette-SD restoration and evaluation.

Replaces both test_palette_sd.py and sample_palette_sd.py.

Usage:
    python evaluate.py --quick              # 1 image, fast sanity check
    python evaluate.py                      # all images, full metrics
    python evaluate.py --max-images 50      # first 50 images
    python evaluate.py --best               # use best checkpoint
    python evaluate.py --steps 100          # override DDIM steps
    python evaluate.py --avg-samples 4      # average 4 runs per image
"""

from __future__ import annotations

import argparse
import os
import csv
import statistics

import torch
from tqdm import tqdm

from config import (
    SD_PALETTE_MODEL_CONFIG,
    SD_PALETTE_SAMPLE_CONFIG,
    RESTORATION_DATA_CONFIG,
    sd_palette_checkpoint_dir,
    sd_palette_best_checkpoint_dir,
)
from dataset import get_restoration_dataloader
from sd_utils import load_palette_sd, sd_palette_inference
from eval_single import eval_image_pair
from utils import save_comparison


def _to_eval_format(x: torch.Tensor) -> torch.Tensor:
    """Convert [-1, 1] NCHW tensor to [0, 1] CHW (first image in batch)."""
    return ((x[0].float() + 1.0) / 2.0).clamp(0.0, 1.0)


def parse_args():
    parser = argparse.ArgumentParser(description="Palette-SD restoration & evaluation")
    parser.add_argument(
        "--quick", action="store_true",
        help="Run on a single image for a fast sanity check",
    )
    parser.add_argument(
        "--max-images", type=int, default=None,
        help="Maximum number of images to process (default: all)",
    )
    parser.add_argument(
        "--best", action="store_true",
        help="Use best validation checkpoint instead of latest",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help=f"DDIM steps (default: {SD_PALETTE_SAMPLE_CONFIG['num_steps']})",
    )
    parser.add_argument(
        "--eta", type=float, default=None,
        help=f"DDIM eta (default: {SD_PALETTE_SAMPLE_CONFIG['eta']})",
    )
    parser.add_argument(
        "--avg-samples", type=int, default=1,
        help="Number of inference runs to average per image (default: 1)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: from config)",
    )
    return parser.parse_args()


@torch.no_grad()
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_id = SD_PALETTE_MODEL_CONFIG["sd_model_id"]
    use_qvae = SD_PALETTE_MODEL_CONFIG.get("use_gqir_qvae", False)

    if args.best:
        ckpt_dir = sd_palette_best_checkpoint_dir()
        print("Using best validation checkpoint")
    else:
        ckpt_dir = sd_palette_checkpoint_dir()

    meas_vae, vae, unet, null_embeds = load_palette_sd(
        model_id, ckpt_dir, device, use_gqir_qvae=use_qvae,
    )
    unet.eval()
    print(f"Palette-SD model loaded (qVAE: {meas_vae is not None}).")

    num_steps = args.steps or SD_PALETTE_SAMPLE_CONFIG["num_steps"]
    eta = args.eta if args.eta is not None else SD_PALETTE_SAMPLE_CONFIG["eta"]
    avg_samples = args.avg_samples
    max_images = 1 if args.quick else args.max_images

    print(f"DDIM: {num_steps} steps, eta={eta}, avg_samples={avg_samples}")
    if max_images is not None:
        print(f"Processing up to {max_images} image(s)")

    dataloader = get_restoration_dataloader(RESTORATION_DATA_CONFIG)

    out_dir = args.output_dir or SD_PALETTE_SAMPLE_CONFIG["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    all_psnr = []
    all_msssim = []
    all_lpips = []

    metrics_file = os.path.join(out_dir, "metrics.csv")

    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "psnr", "ms_ssim", "lpips"])

        for i, (measurement, target) in enumerate(tqdm(dataloader)):
            if max_images is not None and i >= max_images:
                break

            measurement = measurement.to(device, dtype=torch.float16)

            # ── Multi-sample averaging ────────────────────────────────
            accum = None
            for s in range(avg_samples):
                sample = sd_palette_inference(
                    unet=unet,
                    meas_vae=meas_vae,
                    vae=vae,
                    null_embeds=null_embeds,
                    measurement=measurement,
                    model_id=model_id,
                    device=device,
                    num_steps=num_steps,
                    eta=eta,
                )
                if accum is None:
                    accum = sample.float()
                else:
                    accum += sample.float()

            restored = (accum / avg_samples).clamp(-1, 1)

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
        print(f"  Palette-SD Results ({n} images, {avg_samples} avg samples)")
        print(f"{'='*60}")
        print(f"  Mean PSNR:    {statistics.mean(all_psnr):.4f} dB")
        print(f"  Mean MS-SSIM: {statistics.mean(all_msssim):.6f}")
        print(f"  Mean LPIPS:   {statistics.mean(all_lpips):.6f}")
        if n > 1:
            print(f"  Median PSNR:  {statistics.median(all_psnr):.4f} dB")
            print(f"  Std PSNR:     {statistics.stdev(all_psnr):.4f}")
            print(f"  Min PSNR:     {min(all_psnr):.4f} | Max: {max(all_psnr):.4f}")
        print(f"{'='*60}\n")

    print(f"Per-image metrics: {metrics_file}")
    print(f"Comparison images: {out_dir}/")


if __name__ == "__main__":
    run(parse_args())