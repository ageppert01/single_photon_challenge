"""
Shared utilities for Stable Diffusion latent Palette.

Handles:
  - Loading SD components (VAE, UNet, null text embeddings)
  - Expanding UNet conv_in for Palette conditioning (4ch -> 8ch)
  - Applying LoRA to the UNet for efficient fine-tuning
  - Padding images to SD-compatible resolutions
  - Encoding / decoding through the frozen VAE
  - Saving / loading Palette-SD checkpoints
"""

from __future__ import annotations

import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, PeftModel


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


def _get_conv_in(unet) -> nn.Conv2d:
    """
    Get the conv_in layer from a UNet that may be wrapped in PeftModel.

    PeftModel attribute paths vary across peft versions:
      - peft >= 0.10: unet.base_model.model.conv_in
      - peft < 0.10:  unet.base_model.conv_in
      - bare UNet:    unet.conv_in

    This helper tries all paths to be version-safe.
    """
    # Direct attribute (bare UNet or some peft versions)
    if hasattr(unet, "conv_in") and isinstance(unet.conv_in, nn.Conv2d):
        return unet.conv_in

    # peft >= 0.10 path
    if hasattr(unet, "base_model"):
        base = unet.base_model
        if hasattr(base, "model") and hasattr(base.model, "conv_in"):
            return base.model.conv_in
        if hasattr(base, "conv_in"):
            return base.conv_in

    raise AttributeError(
        f"Cannot find conv_in on {type(unet).__name__}. "
        f"Attributes: {[a for a in dir(unet) if not a.startswith('_')]}"
    )


# ── LoRA ─────────────────────────────────────────────────────────────────────


def _apply_lora(unet: UNet2DConditionModel, rank: int, alpha: int) -> PeftModel:
    """Wrap UNet with LoRA adapters on attention layers."""
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",
        ],
        lora_dropout=0.0,
    )
    return get_peft_model(unet, config)


# ── Load / save ──────────────────────────────────────────────────────────────


def load_sd_components(
    model_id: str,
    device: torch.device,
    lora_rank: int = 64,
    lora_alpha: int = 64,
    dtype: torch.dtype = torch.float16,
):
    """
    Load and prepare all SD components for Palette training.

    Returns:
        vae:             Frozen AutoencoderKL
        unet:            UNet with expanded conv_in + LoRA (trainable)
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

    # Gradient checkpointing MUST be enabled before LoRA wrapping,
    # because PeftModel may not expose this method.
    unet.enable_gradient_checkpointing()

    unet = _apply_lora(unet, lora_rank, lora_alpha)
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
    Save LoRA weights + modified conv_in.

    Works whether unet is a bare PeftModel or DDP-unwrapped PeftModel.
    """
    os.makedirs(save_dir, exist_ok=True)

    unet.save_pretrained(os.path.join(save_dir, "lora"))

    conv_in = _get_conv_in(unet)
    torch.save(
        conv_in.state_dict(),
        os.path.join(save_dir, "conv_in.pth"),
    )
    print(f"  Saved Palette-SD checkpoint to {save_dir}")


def load_palette_sd(
    model_id: str,
    save_dir: str,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
):
    """
    Load a trained Palette-SD model (LoRA + conv_in).

    Returns:
        vae, unet, null_embeds   (ready for inference)
    """
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
    vae.to(device)
    vae.eval()
    vae.requires_grad_(False)

    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=dtype,
    )
    _expand_conv_in(unet)

    conv_in_path = os.path.join(save_dir, "conv_in.pth")
    unet.conv_in.load_state_dict(torch.load(conv_in_path, map_location="cpu"))

    lora_path = os.path.join(save_dir, "lora")
    unet = PeftModel.from_pretrained(unet, lora_path)
    unet.to(device)
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