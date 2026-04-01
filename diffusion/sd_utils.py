"""
Shared utilities for Stable Diffusion latent Palette.

Handles:
  - Loading SD components (VAE, UNet, null text embeddings)
  - Expanding UNet conv_in for Palette conditioning (4ch -> 8ch)
  - Selectively unfreezing conv_in + mid_block + decoder for training
  - Padding images to SD-compatible resolutions
  - Encoding / decoding through the frozen VAE
  - Saving / loading Palette-SD checkpoints

V2: Replaced LoRA-only fine-tuning with selective unfreezing.
    The LoRA approach failed because gradients could not flow back
    through the frozen encoder to update the zero-initialized conv_in
    condition channels. Now conv_in, mid_block, and the full decoder
    are trainable, giving ~340M params and a direct gradient path
    from the loss to the conditioning input.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer


# ── Constants ────────────────────────────────────────────────────────────────

SD_VAE_FACTOR = 8       # VAE spatial downsampling factor
SD_UNET_ALIGN = 8       # latent must be divisible by 2^(num_down_blocks)


# ── Padding ──────────────────────────────────────────────────────────────────


def _compute_pad(size: int) -> int:
    """Pixels to add so that size / VAE_FACTOR is divisible by UNET_ALIGN."""
    latent = size // SD_VAE_FACTOR
    remainder = latent % SD_UNET_ALIGN
    if remainder == 0:
        return 0
    return (SD_UNET_ALIGN - remainder) * SD_VAE_FACTOR


def pad_for_sd(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Reflection-pad (B,C,H,W) so H and W are SD-compatible.

    For 800x800 images: pads to 832x832 (latent 104x104).

    Returns:
        (padded_tensor, original_size)
    """
    _, _, h, w = x.shape
    pad_h = _compute_pad(h)
    pad_w = _compute_pad(w)
    if pad_h == 0 and pad_w == 0:
        return x, (h, w)
    return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect"), (h, w)


def unpad(x: torch.Tensor, original_size: Tuple[int, int]) -> torch.Tensor:
    """Crop decoded output back to the original spatial size."""
    h, w = original_size
    return x[:, :, :h, :w]


# ── Conv_in expansion ───────────────────────────────────────────────────────


def _expand_conv_in(unet: UNet2DConditionModel) -> None:
    """
    Expand UNet's conv_in from 4 -> 8 input channels in-place.

    Copies pretrained weights to the first 4 channels and zero-inits
    the new 4 channels (condition path).
    """
    old = unet.conv_in
    new = nn.Conv2d(
        8,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
    )
    new.weight.data.zero_()
    new.weight.data[:, :4] = old.weight.data
    new.bias.data = old.bias.data.clone()
    unet.conv_in = new


# ── Selective unfreezing ─────────────────────────────────────────────────────


def _selective_unfreeze(unet: UNet2DConditionModel) -> None:
    """
    Freeze encoder, unfreeze conv_in + mid_block + decoder + output.

    This gives a direct gradient path from the MSE loss through the
    decoder and mid_block back to conv_in, so the zero-initialized
    condition channels can actually learn.

    The encoder (down_blocks) stays frozen to preserve pretrained
    feature extraction. Cross-attention layers in up_blocks are also
    unfrozen, allowing the model to adapt how it uses the null
    text embedding context.

    Typical param count: ~340M trainable / ~860M total.
    """
    # First freeze everything
    unet.requires_grad_(False)

    # Unfreeze: conv_in, mid_block, up_blocks (decoder), output layers
    for name, param in unet.named_parameters():
        if any(k in name for k in [
            "conv_in",
            "mid_block",
            "up_blocks",
            "conv_norm_out",
            "conv_out",
        ]):
            param.requires_grad = True


# ── Load / save ──────────────────────────────────────────────────────────────


def load_sd_components(
    model_id: str,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
):
    """
    Load and prepare all SD components for Palette training.

    Returns:
        vae:             Frozen AutoencoderKL
        unet:            UNet with expanded conv_in, decoder unfrozen
        null_embeds:     Pre-computed null text embedding for cross-attn
        noise_scheduler: DDPMScheduler matching SD's pretrained schedule
    """
    print(f"Loading Stable Diffusion from {model_id} ...")

    # ── VAE (frozen) ──
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
    vae.to(device)
    vae.eval()
    vae.requires_grad_(False)

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

    # ── Null text embedding (computed once, then discard text encoder) ──
    null_embeds = _compute_null_embedding(model_id, device, dtype)

    # ── Noise scheduler ──
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    print("  SD components loaded.")
    return vae, unet, null_embeds, noise_scheduler


def _compute_null_embedding(
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the null text embedding and discard the text encoder."""
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
        null_embeds = text_encoder(null_ids)[0]  # (1, 77, hidden_dim)

    del text_encoder, tokenizer
    torch.cuda.empty_cache()
    return null_embeds


def save_palette_sd(unet, save_dir: str) -> None:
    """
    Save the full UNet state dict (trainable + frozen weights).

    We save everything rather than just trainable params to keep
    the loading logic simple and avoid key-matching issues.
    The checkpoint is ~1.7 GB in fp16.
    """
    os.makedirs(save_dir, exist_ok=True)
    torch.save(
        unet.state_dict(),
        os.path.join(save_dir, "unet.pth"),
    )
    print(f"  Saved Palette-SD checkpoint to {save_dir}")


def load_palette_sd(
    model_id: str,
    save_dir: str,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
):
    """
    Load a trained Palette-SD model.

    Returns:
        vae, unet, null_embeds   (ready for inference)
    """
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
    vae.to(device)
    vae.eval()
    vae.requires_grad_(False)

    # Build UNet shell with expanded conv_in, then load trained weights
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

    return vae, unet, null_embeds


# ── Encode / decode helpers ──────────────────────────────────────────────────


@torch.no_grad()
def encode_to_latent(
    vae: AutoencoderKL,
    images: torch.Tensor,
    deterministic: bool = False,
) -> torch.Tensor:
    """
    Encode (B,3,H,W) images in [-1,1] to scaled latents.

    Handles padding internally.  Returns (B,4,Hl,Wl) latents.
    """
    x, _orig_size = pad_for_sd(images)
    posterior = vae.encode(x).latent_dist
    z = posterior.mode() if deterministic else posterior.sample()
    return z * vae.config.scaling_factor


@torch.no_grad()
def decode_from_latent(
    vae: AutoencoderKL,
    latent: torch.Tensor,
    original_size: Tuple[int, int] = (800, 800),
) -> torch.Tensor:
    """
    Decode scaled latents to (B,3,H,W) images in [-1,1].

    Handles un-padding to original_size.
    """
    x = vae.decode(latent / vae.config.scaling_factor).sample
    x = unpad(x, original_size)
    return x.clamp(-1, 1)