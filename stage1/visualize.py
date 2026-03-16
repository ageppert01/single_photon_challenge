"""
Save comparison images: Naive sum | RNN | Ground truth (side by side).

Usage:
  python -m stage1.visualize --data_root /path/to/data --ckpt path/to/stage1_epoch9.pt --scale 0.25 --max_samples 6 --out_dir comparison_images
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from stage1.dataloader import (
    PhotonCubeDataset,
    load_photoncube,
    load_gt_image,
    naive_sum,
    downsample_frames,
)
from stage1.model import Stage1RNN


def parse_args():
    p = argparse.ArgumentParser(description="Save Naive vs RNN vs GT comparison images")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--scale", type=float, default=0.25)
    p.add_argument("--chunk_size", type=int, default=64)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max_samples", type=int, default=6, help="Number of comparison images to save")
    p.add_argument("--out_dir", type=str, default="stage1_comparisons", help="Where to save PNGs")
    p.add_argument("--display_size", type=int, default=400, help="Width/height of each panel in the saved image")
    return p.parse_args()


def make_comparison_image(
    naive_img: np.ndarray,
    rnn_img: np.ndarray,
    gt_img: np.ndarray,
    display_size: int,
    psnr_naive: float,
    psnr_rnn: float,
    scene_name: str,
) -> np.ndarray:
    """Stack three images side by side with labels. All inputs (H,W,3) uint8."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        # Fallback: resize with torch, concat, no text
        def resize_np(x: np.ndarray, size: int) -> np.ndarray:
            t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
            return (t.squeeze(0).permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
        n = resize_np(naive_img, display_size)
        r = resize_np(rnn_img, display_size)
        g = resize_np(gt_img, display_size)
        return np.concatenate([n, r, g], axis=1)

    def resize(img: np.ndarray, size: int) -> Image.Image:
        return Image.fromarray(img).resize((size, size), Image.Resampling.LANCZOS)

    label_h = 32
    panel_w = panel_h = display_size
    total_w = panel_w * 3
    total_h = label_h + panel_h

    canvas = Image.new("RGB", (total_w, total_h), (40, 40, 40))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    # Panels: Naive | RNN | GT
    n_pil = resize(naive_img, display_size)
    r_pil = resize(rnn_img, display_size)
    g_pil = resize(gt_img, display_size)
    canvas.paste(n_pil, (0, label_h))
    canvas.paste(r_pil, (panel_w, label_h))
    canvas.paste(g_pil, (panel_w * 2, label_h))

    # Labels
    draw.text((panel_w // 2 - 40, 6), f"Naive sum (PSNR {psnr_naive:.1f})", fill=(255, 255, 255), font=font)
    draw.text((panel_w + panel_w // 2 - 30, 6), f"RNN (PSNR {psnr_rnn:.1f})", fill=(255, 255, 255), font=font)
    draw.text((panel_w * 2 + panel_w // 2 - 35, 6), "Ground truth", fill=(255, 255, 255), font=font)
    draw.text((10, total_h - 20), scene_name, fill=(180, 180, 180), font=font)

    return np.array(canvas)


def run_visualize():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    scale = ckpt.get("scale", args.scale)
    model = Stage1RNN(in_channels=3, hidden_channels=64, chunk_size=args.chunk_size, use_decoder=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    dataset = PhotonCubeDataset(root=args.data_root, split=args.split)
    if len(dataset) == 0:
        raise FileNotFoundError(f"No samples under {Path(args.data_root)/args.split}")

    # PSNR helper (simple)
    def psnr(gt: np.ndarray, pred: np.ndarray) -> float:
        gt = gt.astype(np.float32) / 255.0
        pred = np.clip(pred.astype(np.float32) / 255.0, 0, 1)
        mse = np.mean((gt - pred) ** 2)
        if mse == 0:
            return float("inf")
        import math
        return 10.0 * math.log10(1.0 / mse)

    try:
        import imageio.v3 as imageio
    except ImportError:
        from PIL import Image
        imageio = None

    for idx in range(min(args.max_samples, len(dataset))):
        npy_path, gt_path = dataset[idx]
        scene_name = npy_path.parent.name

        frames = load_photoncube(npy_path, mmap=False)
        frames_ds = downsample_frames(frames, scale=scale)
        gt = load_gt_image(gt_path)
        gt_uint8 = (np.clip(gt.astype(np.float32) / 255.0, 0, 1) * 255).astype(np.uint8)

        naive_img = naive_sum(frames, num_frames=1024, to_uint8=False)
        naive_img = (np.clip(naive_img, 0, 1) * 255).astype(np.uint8)

        def chunk_iter():
            for start in range(0, frames_ds.shape[0], args.chunk_size):
                ch = frames_ds[start : start + args.chunk_size]
                yield torch.from_numpy(ch).permute(0, 3, 1, 2).to(device).float()

        with torch.no_grad():
            h, c, decoded = model.forward_chunked(chunk_iter(), h=None, c=None)
        decoded = decoded.squeeze(0)
        decoded_up = F.interpolate(
            decoded.unsqueeze(0), size=(800, 800), mode="bilinear", align_corners=False
        ).squeeze(0)
        rnn_img = (decoded_up.permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)

        psnr_n = psnr(gt_uint8, naive_img)
        psnr_r = psnr(gt_uint8, rnn_img)

        comparison = make_comparison_image(
            naive_img, rnn_img, gt_uint8,
            display_size=args.display_size,
            psnr_naive=psnr_n,
            psnr_rnn=psnr_r,
            scene_name=scene_name,
        )

        out_path = out_dir / f"comparison_{idx}_{scene_name}.png"
        if imageio is not None:
            imageio.imwrite(out_path, comparison)
        else:
            Image.fromarray(comparison).save(out_path)
        print(f"Saved {out_path}")

    print(f"Done. Saved {min(args.max_samples, len(dataset))} comparison images to {out_dir}")


if __name__ == "__main__":
    run_visualize()
