"""
Run Stage 1 RNN on test .npy files (no GT) and save output .png images.

Keeps the same folder structure as the test set (e.g. basement/000000.png)
so you can zip the output for submission or inspection.

Usage:
  python -m stage1.run_inference \
    --input_dir /path/to/test \
    --output_dir /path/to/test_output \
    --ckpt /path/to/stage1_epoch9.pt \
    --scale 0.25
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from stage1.dataloader import load_photoncube, downsample_frames
from stage1.model import Stage1RNN


def parse_args():
    p = argparse.ArgumentParser(description="Run Stage 1 RNN on test .npy, save .png")
    p.add_argument("--input_dir", type=str, required=True, help="Root folder containing scene/*.npy (e.g. test/)")
    p.add_argument("--output_dir", type=str, required=True, help="Where to save scene/*.png (same structure)")
    p.add_argument("--ckpt", type=str, required=True, help="Path to stage1 checkpoint .pt")
    p.add_argument("--scale", type=float, default=0.25, help="Must match training")
    p.add_argument("--chunk_size", type=int, default=64)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def run_inference():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # Find all .npy under input_dir
    npy_files = sorted(input_root.glob("**/*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files under {input_root}")

    # Load model
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    scale = ckpt.get("scale", args.scale)
    model = Stage1RNN(in_channels=3, hidden_channels=64, chunk_size=args.chunk_size, use_decoder=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    try:
        import imageio.v3 as imageio
    except ImportError:
        from PIL import Image
        imageio = None

    for i, npy_path in enumerate(npy_files):
        rel = npy_path.relative_to(input_root)
        out_path = output_root / rel.with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Stream load: same as dataloader, no GT
        pc = np.load(npy_path, mmap_mode="r")
        num_frames = pc.shape[0]

        def chunk_iter():
            for start in range(0, num_frames, args.chunk_size):
                slab = pc[start : start + args.chunk_size]
                unpacked = np.unpackbits(slab, axis=2)
                ch = downsample_frames(unpacked, scale=scale)
                t = torch.from_numpy(ch).permute(0, 3, 1, 2).to(device).float()
                yield t

        with torch.no_grad():
            h, c, decoded = model.forward_chunked(chunk_iter(), h=None, c=None)
        decoded = decoded.squeeze(0)
        decoded_up = F.interpolate(
            decoded.unsqueeze(0), size=(800, 800), mode="bilinear", align_corners=False
        ).squeeze(0)
        out_img = (decoded_up.permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)

        if imageio is not None:
            imageio.imwrite(out_path, out_img)
        else:
            Image.fromarray(out_img).save(out_path)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"Saved {i + 1}/{len(npy_files)}: {out_path}")

    print(f"Done. Saved {len(npy_files)} images to {output_root}")


if __name__ == "__main__":
    run_inference()
