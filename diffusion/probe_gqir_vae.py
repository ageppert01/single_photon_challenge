"""
Probe script v3: convert gQIR qVAE (LDM format) to diffusers and test.

Fixes from v2:
  1. Decoder up_blocks indices are reversed (LDM up.0 = highest res = diffusers up_blocks.3)
  2. Attention q/k/v/proj_out weights: squeeze (512,512,1,1) → (512,512)
  3. encoder.norm_out → encoder.conv_norm_out (v2 only handled decoder)
"""

import os
import re
import torch


QVAE_PATH = None
SD21_MODEL_ID = "ByteDance/sd2.1-base-zsnr-laionaes5"
NUM_DECODER_UP_BLOCKS = 4  # SD VAE has 4 up blocks in decoder


def convert_ldm_vae_keys(ldm_state_dict: dict) -> dict:
    """Convert LDM-format VAE state dict to diffusers format."""
    new_state = {}

    for key, value in ldm_state_dict.items():
        new_key = key

        # ── Encoder down blocks ──
        new_key = re.sub(
            r"encoder\.down\.(\d+)\.block\.(\d+)",
            r"encoder.down_blocks.\1.resnets.\2",
            new_key,
        )
        new_key = re.sub(
            r"encoder\.down\.(\d+)\.downsample",
            r"encoder.down_blocks.\1.downsamplers.0",
            new_key,
        )

        # ── Decoder up blocks (REVERSED indexing) ──
        # LDM decoder.up.0 = highest resolution = diffusers up_blocks.3
        def reverse_up_idx(m):
            old_idx = int(m.group(1))
            new_idx = NUM_DECODER_UP_BLOCKS - 1 - old_idx
            rest = m.group(2)  # "block.N" or "upsample"
            return f"decoder.up_blocks.{new_idx}.{rest}"

        new_key = re.sub(
            r"decoder\.up\.(\d+)\.(.*)",
            reverse_up_idx,
            new_key,
        )

        # Now convert block.N → resnets.N and upsample → upsamplers.0
        new_key = re.sub(
            r"(decoder\.up_blocks\.\d+)\.block\.(\d+)",
            r"\1.resnets.\2",
            new_key,
        )
        new_key = re.sub(
            r"(decoder\.up_blocks\.\d+)\.upsample\.",
            r"\1.upsamplers.0.",
            new_key,
        )

        # ── Mid blocks (encoder + decoder) ──
        new_key = re.sub(
            r"(encoder|decoder)\.mid\.block_(\d+)",
            lambda m: f"{m.group(1)}.mid_block.resnets.{int(m.group(2)) - 1}",
            new_key,
        )
        new_key = re.sub(
            r"(encoder|decoder)\.mid\.attn_1",
            r"\1.mid_block.attentions.0",
            new_key,
        )

        # ── Attention layers ──
        new_key = re.sub(r"\.q\.", ".to_q.", new_key)
        new_key = re.sub(r"\.k\.", ".to_k.", new_key)
        new_key = re.sub(r"\.v\.", ".to_v.", new_key)
        new_key = re.sub(r"\.proj_out\.", ".to_out.0.", new_key)
        new_key = re.sub(
            r"(attentions\.\d+)\.norm\.",
            r"\1.group_norm.",
            new_key,
        )

        # ── norm_out → conv_norm_out (both encoder and decoder) ──
        new_key = re.sub(
            r"(encoder|decoder)\.norm_out\.",
            r"\1.conv_norm_out.",
            new_key,
        )

        # ── nin_shortcut → conv_shortcut ──
        new_key = new_key.replace("nin_shortcut", "conv_shortcut")

        # ── Squeeze attention Conv2d weights to Linear ──
        if any(attn_key in new_key for attn_key in [".to_q.", ".to_k.", ".to_v.", ".to_out.0."]):
            if value.ndim == 4 and value.shape[2:] == (1, 1):
                value = value.squeeze(-1).squeeze(-1)

        new_state[new_key] = value

    return new_state


def download_checkpoint():
    global QVAE_PATH
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id="aRy4n/gQIR", filename="1-bit/1965000.pt")
    QVAE_PATH = path
    print(f"qVAE checkpoint: {QVAE_PATH}")


def test_conversion():
    print(f"\n{'='*60}")
    print("LOADING AND CONVERTING")
    print(f"{'='*60}")

    ckpt = torch.load(QVAE_PATH, map_location="cpu")
    print(f"Checkpoint: {len(ckpt)} keys (LDM format)")

    converted = convert_ldm_vae_keys(ckpt)
    print(f"Converted:  {len(converted)} keys (diffusers format)")

    print(f"\n{'='*60}")
    print("LOADING INTO DIFFUSERS VAE")
    print(f"{'='*60}")

    from diffusers import AutoencoderKL

    print(f"Loading base VAE from {SD21_MODEL_ID} ...")
    vae = AutoencoderKL.from_pretrained(SD21_MODEL_ID, subfolder="vae")
    vae_keys = set(vae.state_dict().keys())
    conv_key_set = set(converted.keys())

    matching = vae_keys & conv_key_set
    missing = vae_keys - conv_key_set
    extra = conv_key_set - vae_keys

    print(f"\n  Matching: {len(matching)} / {len(vae_keys)}")
    print(f"  Missing:  {len(missing)}")
    print(f"  Extra:    {len(extra)}")

    if missing:
        print("\n  Missing keys:")
        for k in sorted(missing):
            print(f"    {k}")
    if extra:
        print("\n  Extra keys:")
        for k in sorted(extra):
            print(f"    {k}")

    # Check shapes
    shape_mismatches = []
    vae_state = vae.state_dict()
    for k in matching:
        if vae_state[k].shape != converted[k].shape:
            shape_mismatches.append((k, converted[k].shape, vae_state[k].shape))
    if shape_mismatches:
        print(f"\n  Shape mismatches ({len(shape_mismatches)}):")
        for k, ckpt_shape, vae_shape in shape_mismatches[:5]:
            print(f"    {k}: ckpt {ckpt_shape} vs vae {vae_shape}")

    # Try loading
    try:
        vae.load_state_dict(converted, strict=True)
        print("\n  >>> STRICT load succeeded!")
    except RuntimeError as e:
        msg = str(e)
        # Count errors
        missing_count = msg.count("Missing key")
        size_count = msg.count("size mismatch")
        print(f"\n  Strict failed: {missing_count} missing, {size_count} size mismatches")
        try:
            vae.load_state_dict(converted, strict=False)
            print("  >>> LOOSE load succeeded")
        except RuntimeError as e2:
            print(f"  Loose also failed: {str(e2)[:200]}")
            return

    # Functional test
    print(f"\n{'='*60}")
    print("ENCODE / DECODE TEST")
    print(f"{'='*60}")

    vae.eval()
    x = torch.randn(1, 3, 256, 256).clamp(-1, 1)

    with torch.no_grad():
        latent = vae.encode(x).latent_dist.mode()
        decoded = vae.decode(latent).sample

    print(f"  Input:   {x.shape}")
    print(f"  Latent:  {latent.shape}")
    print(f"  Decoded: {decoded.shape}")
    print(f"  Latent range:  [{latent.min():.3f}, {latent.max():.3f}]")
    print(f"  Decoded range: [{decoded.min():.3f}, {decoded.max():.3f}]")
    print(f"  >>> Encode/decode functional")


if __name__ == "__main__":
    download_checkpoint()
    test_conversion()
    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")