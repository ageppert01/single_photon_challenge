"""Quick single-image test for Palette-SD with full metrics."""

import os
import torch

from config import SD_PALETTE_MODEL_CONFIG, RESTORATION_DATA_CONFIG, sd_palette_checkpoint_dir
from dataset import get_restoration_dataloader
from sd_utils import load_palette_sd, sd_palette_inference
from eval_single import eval_image_pair
from utils import save_comparison


@torch.no_grad()
def test_one():
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

    # Load one sample
    dataloader = get_restoration_dataloader(RESTORATION_DATA_CONFIG)
    measurement, target = next(iter(dataloader))
    measurement = measurement.to(device, dtype=torch.float16)
    target = target.to(device)
    print(f"Measurement range: [{measurement.min():.2f}, {measurement.max():.2f}]")
    print(f"Target range:      [{target.min():.2f}, {target.max():.2f}]")

    # Run inference
    num_steps = 50
    print(f"Running DDIM with {num_steps} steps...")
    restored = sd_palette_inference(
        unet=unet,
        meas_vae=meas_vae,
        vae=vae,
        null_embeds=null_embeds,
        measurement=measurement,
        model_id=model_id,
        device=device,
        num_steps=num_steps,
    )
    print(f"Restored range:    [{restored.min():.2f}, {restored.max():.2f}]")

    # Metrics
    gt_eval = ((target[0].float() + 1) / 2).clamp(0, 1)
    pred_eval = ((restored[0].float() + 1) / 2).clamp(0, 1)

    psnr_val, msssim_val, lpips_val = eval_image_pair(gt_eval, pred_eval, device=device)
    print(f"PSNR:    {psnr_val:.2f} dB")
    print(f"MS-SSIM: {msssim_val:.4f}")
    print(f"LPIPS:   {lpips_val:.4f}")

    out_dir = "single_photon_palette_sd"
    os.makedirs(out_dir, exist_ok=True)
    save_comparison(measurement.float(), restored.float(), target.float(),
                    os.path.join(out_dir, "sd_palette_test.png"))
    print(f"Saved {out_dir}/sd_palette_test.png")


if __name__ == "__main__":
    test_one()