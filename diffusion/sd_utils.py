"""
Shared utilities for Stable Diffusion latent Palette.

Unified version supporting both:
  - SD 1.5 (epsilon-prediction, standard VAE for all encoding)
  - SD 2.1 (v-prediction, gQIR qVAE for measurement encoding)

The prediction type and qVAE usage are auto-detected from the config
and model, so the training/inference scripts don't need to branch.
"""

from __future__ import annotations

import os
import re
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from torch.cuda.amp import autocast
from transformers import CLIPTextModel, CLIPTokenizer


# ── Constants ────────────────────────────────────────────────────────────────

SD_VAE_FACTOR = 8
SD_UNET_ALIGN = 8


# ── Padding ──────────────────────────────────────────────────────────────────


def _compute_pad(size: int) -> int:
    latent = size // SD_VAE_FACTOR
    remainder = latent % SD_UNET_ALIGN
    if remainder == 0:
        return 0
    return (SD_UNET_ALIGN - remainder) * SD_VAE_FACTOR


def pad_for_sd(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    _, _, h, w = x.shape
    pad_h = _compute_pad(h)
    pad_w = _compute_pad(w)
    if pad_h == 0 and pad_w == 0:
        return x, (h, w)
    return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect"), (h, w)


def unpad(x: torch.Tensor, original_size: Tuple[int, int]) -> torch.Tensor:
    h, w = original_size
    return x[:, :, :h, :w]


# ── Conv_in expansion ───────────────────────────────────────────────────────


def _expand_conv_in(unet: UNet2DConditionModel) -> None:
    old = unet.conv_in
    new = nn.Conv2d(
        8, old.out_channels,
        kernel_size=old.kernel_size, stride=old.stride, padding=old.padding,
    )
    new.weight.data.zero_()
    new.weight.data[:, :4] = old.weight.data
    new.bias.data = old.bias.data.clone()
    unet.conv_in = new


# ── Selective unfreezing ─────────────────────────────────────────────────────


def _selective_unfreeze(unet: UNet2DConditionModel) -> None:
    unet.requires_grad_(False)
    for name, param in unet.named_parameters():
        if any(k in name for k in [
            "conv_in", "mid_block", "up_blocks", "conv_norm_out", "conv_out",
        ]):
            param.requires_grad = True


# ── LDM → Diffusers VAE key conversion (for gQIR checkpoints) ───────────────

NUM_DECODER_UP_BLOCKS = 4


def convert_ldm_vae_keys(ldm_state_dict: dict) -> dict:
    """Convert gQIR qVAE checkpoint (LDM format) to diffusers format."""
    new_state = {}

    for key, value in ldm_state_dict.items():
        new_key = key

        # Encoder down blocks
        new_key = re.sub(
            r"encoder\.down\.(\d+)\.block\.(\d+)",
            r"encoder.down_blocks.\1.resnets.\2", new_key,
        )
        new_key = re.sub(
            r"encoder\.down\.(\d+)\.downsample",
            r"encoder.down_blocks.\1.downsamplers.0", new_key,
        )

        # Decoder up blocks (REVERSED: LDM up.0 = diffusers up_blocks.3)
        def reverse_up_idx(m):
            new_idx = NUM_DECODER_UP_BLOCKS - 1 - int(m.group(1))
            return f"decoder.up_blocks.{new_idx}.{m.group(2)}"

        new_key = re.sub(r"decoder\.up\.(\d+)\.(.*)", reverse_up_idx, new_key)
        new_key = re.sub(
            r"(decoder\.up_blocks\.\d+)\.block\.(\d+)",
            r"\1.resnets.\2", new_key,
        )
        new_key = re.sub(
            r"(decoder\.up_blocks\.\d+)\.upsample\.",
            r"\1.upsamplers.0.", new_key,
        )

        # Mid blocks
        new_key = re.sub(
            r"(encoder|decoder)\.mid\.block_(\d+)",
            lambda m: f"{m.group(1)}.mid_block.resnets.{int(m.group(2)) - 1}",
            new_key,
        )
        new_key = re.sub(
            r"(encoder|decoder)\.mid\.attn_1",
            r"\1.mid_block.attentions.0", new_key,
        )

        # Attention layers
        new_key = re.sub(r"\.q\.", ".to_q.", new_key)
        new_key = re.sub(r"\.k\.", ".to_k.", new_key)
        new_key = re.sub(r"\.v\.", ".to_v.", new_key)
        new_key = re.sub(r"\.proj_out\.", ".to_out.0.", new_key)
        new_key = re.sub(r"(attentions\.\d+)\.norm\.", r"\1.group_norm.", new_key)

        # norm_out → conv_norm_out
        new_key = re.sub(
            r"(encoder|decoder)\.norm_out\.",
            r"\1.conv_norm_out.", new_key,
        )

        # nin_shortcut → conv_shortcut
        new_key = new_key.replace("nin_shortcut", "conv_shortcut")

        # Squeeze attention Conv2d(1,1) → Linear
        if any(a in new_key for a in [".to_q.", ".to_k.", ".to_v.", ".to_out.0."]):
            if value.ndim == 4 and value.shape[2:] == (1, 1):
                value = value.squeeze(-1).squeeze(-1)

        new_state[new_key] = value

    return new_state


# ── Load gQIR qVAE ──────────────────────────────────────────────────────────


def _load_gqir_qvae(
    base_model_id: str,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> AutoencoderKL:
    """Download the gQIR 1-bit qVAE and load into a diffusers AutoencoderKL."""
    from huggingface_hub import hf_hub_download

    print("  Loading gQIR 1-bit qVAE ...")
    ckpt_path = hf_hub_download(repo_id="aRy4n/gQIR", filename="1-bit/1965000.pt")

    qvae = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae", torch_dtype=dtype)

    ldm_state = torch.load(ckpt_path, map_location="cpu")
    diffusers_state = convert_ldm_vae_keys(ldm_state)
    qvae.load_state_dict(diffusers_state, strict=True)

    qvae.to(device)
    qvae.eval()
    qvae.requires_grad_(False)
    print("  gQIR qVAE loaded.")
    return qvae


# ── Load / save ──────────────────────────────────────────────────────────────


def load_sd_components(
    model_id: str,
    device: torch.device,
    use_gqir_qvae: bool = False,
    dtype: torch.dtype = torch.float16,
):
    """
    Load all SD components for Palette training.

    Returns:
        meas_vae:        gQIR qVAE for measurements (or None if not using)
        vae:             Standard VAE for targets + decoding (frozen)
        unet:            UNet with expanded conv_in, decoder unfrozen
        null_embeds:     Null text embedding
        noise_scheduler: DDPMScheduler (prediction type from model config)
    """
    print(f"Loading Stable Diffusion from {model_id} ...")

    # ── Standard VAE (frozen) ──
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
    vae.to(device)
    vae.eval()
    vae.requires_grad_(False)

    # ── Optional gQIR qVAE for measurements ──
    meas_vae = None
    if use_gqir_qvae:
        meas_vae = _load_gqir_qvae(model_id, device, dtype)
    else:
        print("  Using standard VAE for measurements (no gQIR qVAE)")

    # ── UNet ──
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=dtype,
    )
    _expand_conv_in(unet)
    unet.enable_gradient_checkpointing()
    _selective_unfreeze(unet)
    unet.to(device)

    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet.parameters())
    print(f"  UNet: {trainable:,} trainable / {total:,} total parameters")

    # ── Null text embedding ──
    null_embeds = _compute_null_embedding(model_id, device, dtype)

    # ── Noise scheduler ──
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    print(f"  Prediction type: {noise_scheduler.config.prediction_type}")

    print("  SD components loaded.")
    return meas_vae, vae, unet, null_embeds, noise_scheduler


def _compute_null_embedding(
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=dtype,
    )
    text_encoder.to(device)
    text_encoder.eval()

    with torch.no_grad():
        null_ids = tokenizer(
            "",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(device)
        null_embeds = text_encoder(null_ids)[0]

    print(f"  Null embedding shape: {null_embeds.shape}")
    del text_encoder, tokenizer
    torch.cuda.empty_cache()
    return null_embeds


def save_palette_sd(unet, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    torch.save(unet.state_dict(), os.path.join(save_dir, "unet.pth"))
    print(f"  Saved Palette-SD checkpoint to {save_dir}")


def load_palette_sd(
    model_id: str,
    save_dir: str,
    device: torch.device,
    use_gqir_qvae: bool = False,
    dtype: torch.dtype = torch.float16,
):
    """
    Load a trained Palette-SD model for inference.

    Returns:
        meas_vae (or None), vae, unet, null_embeds
    """
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
    vae.to(device)
    vae.eval()
    vae.requires_grad_(False)

    meas_vae = None
    if use_gqir_qvae:
        meas_vae = _load_gqir_qvae(model_id, device, dtype)

    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=dtype,
    )
    _expand_conv_in(unet)

    ckpt_path = os.path.join(save_dir, "unet.pth")
    state = torch.load(ckpt_path, map_location="cpu")
    unet.load_state_dict(state)
    unet.to(device=device, dtype=dtype)
    unet.eval()

    null_embeds = _compute_null_embedding(model_id, device, dtype)

    return meas_vae, vae, unet, null_embeds


# ── Encode / decode helpers ──────────────────────────────────────────────────


@torch.no_grad()
def encode_to_latent(
    vae: AutoencoderKL,
    images: torch.Tensor,
    deterministic: bool = False,
) -> torch.Tensor:
    x, _orig_size = pad_for_sd(images)
    posterior = vae.encode(x).latent_dist
    z = posterior.mode() if deterministic else posterior.sample()
    return z * vae.config.scaling_factor


@torch.no_grad()
def encode_measurement(
    meas_vae: Optional[AutoencoderKL],
    vae: AutoencoderKL,
    measurement: torch.Tensor,
) -> torch.Tensor:
    """Encode measurement using qVAE if available, otherwise standard VAE."""
    encoder = meas_vae if meas_vae is not None else vae
    return encode_to_latent(encoder, measurement, deterministic=True)


@torch.no_grad()
def decode_from_latent(
    vae: AutoencoderKL,
    latent: torch.Tensor,
    original_size: Tuple[int, int] = (800, 800),
) -> torch.Tensor:
    x = vae.decode(latent / vae.config.scaling_factor).sample
    x = unpad(x, original_size)
    return x.clamp(-1, 1)


# ── Inference ────────────────────────────────────────────────────────────────


@torch.no_grad()
def sd_palette_inference(
    unet: UNet2DConditionModel,
    meas_vae: Optional[AutoencoderKL],
    vae: AutoencoderKL,
    null_embeds: torch.Tensor,
    measurement: torch.Tensor,
    model_id: str,
    device: torch.device,
    num_steps: int = 20,
    eta: float = 0.0,
) -> torch.Tensor:
    """
    Run Palette-SD DDIM inference on a single measurement batch.

    Handles encoding, the full reverse diffusion loop (with autocast for
    mixed-precision safety), and decoding back to pixel space.

    Args:
        unet:         Palette UNet (8-channel conv_in: noisy + condition)
        meas_vae:     gQIR qVAE for measurements (or None to use standard VAE)
        vae:          Standard SD VAE for decoding
        null_embeds:  Null text embedding
        measurement:  Input measurement in [-1, 1], shape (B, 3, H, W), fp16
        model_id:     HuggingFace model ID (for loading the scheduler config)
        device:       Torch device
        num_steps:    Number of DDIM denoising steps
        eta:          DDIM stochasticity (0 = deterministic)

    Returns:
        Restored image in [-1, 1], shape (B, 3, 800, 800)
    """
    # Encode measurement to latent space
    z_meas = encode_measurement(meas_vae, vae, measurement)

    # Set up DDIM scheduler (prediction type auto-detected from config)
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    scheduler.set_timesteps(num_steps, device=device)

    # Start from pure noise
    z = torch.randn_like(z_meas)
    encoder_hidden_states = null_embeds.expand(z.shape[0], -1, -1)

    # Reverse diffusion
    for t in scheduler.timesteps:
        z_input = torch.cat([z, z_meas], dim=1)
        with autocast(enabled=True):
            noise_pred = unet(
                z_input,
                t.unsqueeze(0).expand(z.shape[0]),
                encoder_hidden_states=encoder_hidden_states,
            ).sample
        z = scheduler.step(noise_pred, t, z, eta=eta).prev_sample

    # Decode back to pixel space
    return decode_from_latent(vae, z, original_size=(800, 800))