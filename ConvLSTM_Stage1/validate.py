"""
Validation: compare Stage 1 RNN output vs naive sum vs GT.

Computes PSNR, MS-SSIM, LPIPS for (RNN vs GT) and (naive sum vs GT) and prints
a comparison. Uses eval_single.eval_image_pair for metrics.

Usage:
  python -m stage1.validate --data_root /path/to/data --ckpt stage1_checkpoints/stage1_epoch10.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Add repo root for eval_single
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from eval_single import eval_image_pair

from stage1.dataloader import (
    PhotonCubeDataset,
    load_photoncube,
    load_gt_image,
    naive_sum,
    downsample_frames,
)
from stage1.model import Stage1RNN


def parse_args():
    p = argparse.ArgumentParser(description="Validate Stage 1: RNN vs naive sum vs GT")
    p.add_argument("--data_root", type=str, required=True, help="Root dir with train/ or val/")
    p.add_argument("--split", type=str, default="train", help="train or val")
    p.add_argument("--ckpt", type=str, required=True, help="Path to stage1 checkpoint .pt")
    p.add_argument("--scale", type=float, default=0.25, help="Must match training")
    p.add_argument("--chunk_size", type=int, default=64)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max_samples", type=int, default=None, help="Cap number of samples (e.g. 5 for quick run)")
    p.add_argument("--naive_frames", type=int, default=1024, help="Number of frames for naive sum (default all)")
    p.add_argument("--no_lpips", action="store_true", help="Skip LPIPS (e.g. if download fails due to SSL)")
    return p.parse_args()


def _psnr_mssim_only(gt_uint8: np.ndarray, pred_uint8: np.ndarray, device: torch.device) -> tuple[float, float]:
    """Compute PSNR and MS-SSIM only (no LPIPS). Both (800,800,3) uint8."""
    import math
    from pytorch_msssim import ms_ssim
    gt = torch.from_numpy(gt_uint8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    pr = torch.from_numpy(pred_uint8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    mse = torch.mean((gt - pr) ** 2).item()
    psnr = float("inf") if mse == 0 else (10.0 * math.log10(1.0 / mse))
    mssim = float(ms_ssim(pr, gt, data_range=1.0, size_average=True).item())
    return psnr, mssim


def run_validation():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

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

    n = 0
    rnn_psnr, rnn_mssim, rnn_lpips = [], [], []
    naive_psnr, naive_mssim, naive_lpips = [], [], []

    for idx in range(len(dataset)):
        if args.max_samples is not None and n >= args.max_samples:
            break
        npy_path, gt_path = dataset[idx]

        # Load cube and GT
        frames = load_photoncube(npy_path, mmap=False)
        frames_ds = downsample_frames(frames, scale=scale)
        gt = load_gt_image(gt_path)
        gt_float = (gt.astype(np.float32) / 255.0).clip(0, 1)

        # Naive sum (full res): last N frames, then scale to [0,1]
        naive_img = naive_sum(frames, num_frames=args.naive_frames, to_uint8=False)
        naive_img = (np.clip(naive_img, 0, 1) * 255).astype(np.uint8)  # 800x800x3 uint8 for eval

        # RNN: chunked forward
        def chunk_iter():
            chunk_sz = args.chunk_size
            for start in range(0, frames_ds.shape[0], chunk_sz):
                ch = frames_ds[start : start + chunk_sz]
                yield torch.from_numpy(ch).permute(0, 3, 1, 2).to(device).float()

        with torch.no_grad():
            h, c, decoded = model.forward_chunked(chunk_iter(), h=None, c=None)
        # decoded: (1, 3, H_ds, W_ds)
        decoded = decoded.squeeze(0)
        decoded_up = F.interpolate(
            decoded.unsqueeze(0), size=(800, 800), mode="bilinear", align_corners=False
        ).squeeze(0)
        rnn_img = (decoded_up.permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)

        # Metrics vs GT (eval_image_pair expects 800x800x3)
        gt_uint8 = (gt_float * 255).astype(np.uint8)
        if not args.no_lpips:
            try:
                psnr_r, mssim_r, lpips_r = eval_image_pair(gt_uint8, rnn_img, device=device)
                psnr_n, mssim_n, lpips_n = eval_image_pair(gt_uint8, naive_img, device=device)
            except Exception:
                psnr_r, mssim_r = _psnr_mssim_only(gt_uint8, rnn_img, device)
                psnr_n, mssim_n = _psnr_mssim_only(gt_uint8, naive_img, device)
                lpips_r = lpips_n = None
        else:
            psnr_r, mssim_r = _psnr_mssim_only(gt_uint8, rnn_img, device)
            psnr_n, mssim_n = _psnr_mssim_only(gt_uint8, naive_img, device)
            lpips_r = lpips_n = None

        rnn_psnr.append(psnr_r)
        rnn_mssim.append(mssim_r)
        rnn_lpips.append(lpips_r)
        naive_psnr.append(psnr_n)
        naive_mssim.append(mssim_n)
        naive_lpips.append(lpips_n)
        n += 1

    # Aggregate
    def mean(x, skip_none=True):
        x = [v for v in x if v is not None] if skip_none else x
        return sum(x) / len(x) if x else 0.0

    print(f"Validated {n} samples (scale={scale}, chunk_size={args.chunk_size})")
    print("-" * 60)
    print(f"{'Method':<12}  {'PSNR (dB)':>12}  {'MS-SSIM':>12}  {'LPIPS':>12}")
    print("-" * 60)
    lpips_rnn_str = f"{mean(rnn_lpips):.6f}" if any(rnn_lpips) else "N/A"
    lpips_naive_str = f"{mean(naive_lpips):.6f}" if any(naive_lpips) else "N/A"
    print(f"{'RNN':<12}  {mean(rnn_psnr):>12.4f}  {mean(rnn_mssim):>12.6f}  {lpips_rnn_str:>12}")
    print(f"{'Naive sum':<12}  {mean(naive_psnr):>12.4f}  {mean(naive_mssim):>12.6f}  {lpips_naive_str:>12}")
    print("-" * 60)
    print("(PSNR/MS-SSIM: higher is better; LPIPS: lower is better)")


if __name__ == "__main__":
    run_validation()
