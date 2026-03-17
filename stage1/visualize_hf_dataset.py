"""
Visualize on Hugging Face dataset (single-photon-full): one pair at a time.

Downloads one .npy + .png pair at a time to work_dir, runs Stage 1 RNN + naive sum,
saves comparison image to results_dir, then deletes the pair. Repeats for each folder:
top 10 pairs per folder (in sequence). No need to download the full dataset.

Dataset: https://huggingface.co/datasets/ishitakakkar-10/single-photon-full (train/)

Usage:
  python -m stage1.visualize_hf_dataset \
    --repo_id ishitakakkar-10/single-photon-full \
    --work_dir /content/drive/MyDrive/test_hf \
    --results_dir /content/drive/MyDrive/stage1_comparison_hf \
    --ckpt /content/drive/MyDrive/single_photon_checkpoints/stage1_epoch9.pt \
    --scale 0.25 \
    --top_k 10
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from stage1.dataloader import load_photoncube, load_gt_image, naive_sum, downsample_frames
from stage1.model import Stage1RNN


def parse_args():
    p = argparse.ArgumentParser(description="Visualize HF single-photon-full: 1 pair at a time")
    p.add_argument("--repo_id", type=str, default="ishitakakkar-10/single-photon-full")
    p.add_argument("--train_subdir", type=str, default="train")
    p.add_argument("--work_dir", type=str, default="/content/drive/MyDrive/test_hf")
    p.add_argument("--results_dir", type=str, default="/content/drive/MyDrive/stage1_comparison_hf")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--scale", type=float, default=0.25)
    p.add_argument("--chunk_size", type=int, default=64)
    p.add_argument("--top_k", type=int, default=10, help="Top K pairs per folder (in sequence)")
    p.add_argument("--max_folders", type=int, default=None, help="Cap number of folders (default: all)")
    p.add_argument("--display_size", type=int, default=400)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def _psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    gt = gt.astype(np.float32) / 255.0
    pred = np.clip(pred.astype(np.float32) / 255.0, 0, 1)
    mse = np.mean((gt - pred) ** 2)
    return float("inf") if mse == 0 else (10.0 * math.log10(1.0 / mse))


def _make_comparison(naive_img: np.ndarray, rnn_img: np.ndarray, gt_img: np.ndarray,
                     display_size: int, psnr_naive: float, psnr_rnn: float, scene_name: str) -> np.ndarray:
    """Side-by-side Naive | RNN | GT. All (H,W,3) uint8."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        def resize_np(x: np.ndarray, size: int) -> np.ndarray:
            t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
            return (t.squeeze(0).permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
        n = resize_np(naive_img, display_size)
        r = resize_np(rnn_img, display_size)
        g = resize_np(gt_img, display_size)
        return np.concatenate([n, r, g], axis=1)

    def resize(img: np.ndarray, size: int):
        return Image.fromarray(img).resize((size, size), Image.Resampling.LANCZOS)

    label_h, panel = 32, display_size
    canvas = Image.new("RGB", (panel * 3, label_h + panel), (40, 40, 40))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    canvas.paste(resize(naive_img, panel), (0, label_h))
    canvas.paste(resize(rnn_img, panel), (panel, label_h))
    canvas.paste(resize(gt_img, panel), (panel * 2, label_h))
    draw.text((panel // 2 - 40, 6), f"Naive (PSNR {psnr_naive:.1f})", fill=(255, 255, 255), font=font)
    draw.text((panel + panel // 2 - 30, 6), f"RNN (PSNR {psnr_rnn:.1f})", fill=(255, 255, 255), font=font)
    draw.text((panel * 2 + panel // 2 - 35, 6), "Ground truth", fill=(255, 255, 255), font=font)
    draw.text((10, label_h + panel - 20), scene_name, fill=(180, 180, 180), font=font)
    return np.array(canvas)


def main():
    args = parse_args()
    try:
        from huggingface_hub import list_repo_files, hf_hub_download
    except ImportError:
        raise ImportError("Install huggingface_hub: pip install huggingface_hub")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    work_dir = Path(args.work_dir)
    results_dir = Path(args.results_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # List all files in repo (train/)
    prefix = args.train_subdir.rstrip("/") + "/"
    all_files = list_repo_files(args.repo_id, repo_type="dataset")
    train_files = [f for f in all_files if f.startswith(prefix) and "/" in f[len(prefix):]]

    # Group by folder: folder_name -> list of (npy_path, png_path) in order
    folders = defaultdict(list)
    for f in train_files:
        rest = f[len(prefix):]
        parts = rest.split("/")
        if len(parts) != 2:
            continue
        folder_name, fname = parts
        if fname.endswith(".npy"):
            png_path = prefix + folder_name + "/" + fname.replace(".npy", ".png")
            if png_path in all_files:
                folders[folder_name].append((f, png_path))

    # Sort pairs per folder by filename so "top 10 in sequence" = first 10
    for folder_name in folders:
        pairs = folders[folder_name]
        pairs.sort(key=lambda x: x[0])
        folders[folder_name] = pairs[: args.top_k]

    folder_names = sorted(folders.keys())
    if args.max_folders is not None:
        folder_names = folder_names[: args.max_folders]
    if not folder_names:
        raise RuntimeError(f"No folders with .npy+.png pairs under {prefix}")

    # Load model once
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

    total = 0
    for folder_name in folder_names:
        pairs = folders[folder_name]
        out_folder = results_dir / folder_name
        out_folder.mkdir(parents=True, exist_ok=True)

        for i, (npy_rel, png_rel) in enumerate(pairs):
            # Download one pair to work_dir (creates work_dir/train/folder/xxx.npy and .png)
            npy_path = hf_hub_download(
                args.repo_id, npy_rel, repo_type="dataset", local_dir=work_dir, local_dir_use_symlinks=False
            )
            png_path = hf_hub_download(
                args.repo_id, png_rel, repo_type="dataset", local_dir=work_dir, local_dir_use_symlinks=False
            )
            npy_path = Path(npy_path)
            png_path = Path(png_path)

            try:
                frames = load_photoncube(npy_path, mmap=False)
                frames_ds = downsample_frames(frames, scale=scale)
                gt = load_gt_image(png_path)
                gt_uint8 = (np.clip(gt.astype(np.float32) / 255.0, 0, 1) * 255).astype(np.uint8)

                naive_img = (np.clip(naive_sum(frames, num_frames=1024, to_uint8=False), 0, 1) * 255).astype(np.uint8)

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

                psnr_n = _psnr(gt_uint8, naive_img)
                psnr_r = _psnr(gt_uint8, rnn_img)
                comp = _make_comparison(
                    naive_img, rnn_img, gt_uint8,
                    args.display_size, psnr_n, psnr_r,
                    f"{folder_name} #{i}",
                )
                out_path = out_folder / f"comparison_{i:02d}.png"
                if imageio is not None:
                    imageio.imwrite(out_path, comp)
                else:
                    Image.fromarray(comp).save(out_path)
                total += 1
                print(f"Saved {total}: {out_path}")
            finally:
                # Remove downloaded pair to free space
                if npy_path.exists():
                    npy_path.unlink()
                if png_path.exists():
                    png_path.unlink()

    # Remove empty dirs under work_dir if any
    for d in sorted(work_dir.rglob("*"), key=lambda x: len(x.parts), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    print(f"Done. {total} comparison images in {results_dir}")

if __name__ == "__main__":
    main()
