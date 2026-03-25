"""Quick single-image test for Palette-SD with full metrics."""

import os
import torch
from diffusers import DDIMScheduler

from config import SD_PALETTE_MODEL_CONFIG, RESTORATION_DATA_CONFIG, sd_palette_checkpoint_dir
from dataset import get_restoration_dataloader
from sd_utils import load_palette_sd, encode_to_latent, decode_from_latent
from eval_single import eval_image_pair
from utils import save_comparison


@torch.no_grad()
def test_one():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_id = SD_PALETTE_MODEL_CONFIG["sd_model_id"]
    ckpt_dir = sd_palette_checkpoint_dir()

    vae, unet, null_embeds = load_palette_sd(model_id, ckpt_dir, device)
    unet.eval()
    print("Palette-SD model loaded.")

    # Load one sample
    dataloader = get_restoration_dataloader(RESTORATION_DATA_CONFIG)
    measurement, target = next(iter(dataloader))
    measurement = measurement.to(device, dtype=torch.float16)
    target = target.to(device)
    print(f"Measurement range: [{measurement.min():.2f}, {measurement.max():.2f}]")
    print(f"Target range:      [{target.min():.2f}, {target.max():.2f}]")

    # Encode measurement
    z_meas = encode_to_latent(vae, measurement, deterministic=True)
    print(f"Latent shape: {z_meas.shape}")

    # DDIM with few steps for fast test
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    num_steps = 50
    scheduler.set_timesteps(num_steps, device=device)
    print(f"Running DDIM with {num_steps} steps...")

    z = torch.randn_like(z_meas)
    encoder_hidden_states = null_embeds.expand(z.shape[0], -1, -1)

    for t in scheduler.timesteps:
        z_input = torch.cat([z, z_meas], dim=1)
        noise_pred = unet(
            z_input,
            t.unsqueeze(0).expand(z.shape[0]),
            encoder_hidden_states=encoder_hidden_states,
        ).sample
        z = scheduler.step(noise_pred, t, z, eta=0.0).prev_sample

    restored = decode_from_latent(vae, z, original_size=(800, 800))
    print(f"Restored range:    [{restored.min():.2f}, {restored.max():.2f}]")

    # Convert to [0,1] for eval_single (remove batch dim)
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