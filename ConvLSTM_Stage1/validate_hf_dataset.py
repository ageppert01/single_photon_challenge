"""
Validate Stage 1 on a Hugging Face dataset: stream one .npy+.png pair at a time,
same metrics as stage1.validate (PSNR, MS-SSIM, LPIPS). No full local copy needed.

Usage:
  python -m stage1.validate_hf_dataset \
    --repo_id ishitakakkar-10/single-photon-full \
    --work_dir /content/drive/MyDrive/test_hf \
    --ckpt /path/to/stage1_epoch3.pt \
    --scale 0.5 \
    --top_k 10
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from eval_single import eval_image_pair

from stage1.dataloader import load_gt_image, load_photoncube, naive_sum, downsample_frames
from stage1.model import Stage1RNN


def parse_args():
    p = argparse.ArgumentParser(description="Validate Stage 1 on HF dataset (stream pairs)")
    p.add_argument("--repo_id", type=str, default="ishitakakkar-10/single-photon-full")
    p.add_argument("--train_subdir", type=str, default="train")
    p.add_argument("--work_dir", type=str, required=True, help="Temp dir for one pair at a time")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--scale", type=float, default=0.25)
    p.add_argument("--chunk_size", type=int, default=64)
    p.add_argument("--top_k", type=int, default=10, help="First K pairs per scene folder (sorted by name)")
    p.add_argument("--max_folders", type=int, default=None, help="Cap number of scene folders")
    p.add_argument("--max_samples", type=int, default=None, help="Cap total pairs evaluated (across folders)")
    p.add_argument("--naive_frames", type=int, default=1024)
    p.add_argument("--no_lpips", action="store_true")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def _psnr_mssim_only(gt_uint8: np.ndarray, pred_uint8: np.ndarray, device: torch.device) -> tuple[float, float]:
    import math

    from pytorch_msssim import ms_ssim

    gt = torch.from_numpy(gt_uint8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    pr = torch.from_numpy(pred_uint8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    mse = torch.mean((gt - pr) ** 2).item()
    psnr = float("inf") if mse == 0 else (10.0 * math.log10(1.0 / mse))
    mssim = float(ms_ssim(pr, gt, data_range=1.0, size_average=True).item())
    return psnr, mssim


def _collect_hf_pairs(repo_id: str, train_subdir: str, top_k: int) -> dict[str, list[tuple[str, str]]]:
    from huggingface_hub import list_repo_files

    prefix = train_subdir.rstrip("/") + "/"
    all_files = list_repo_files(repo_id, repo_type="dataset")
    train_files = [f for f in all_files if f.startswith(prefix) and "/" in f[len(prefix) :]]

    folders: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for f in train_files:
        rest = f[len(prefix) :]
        parts = rest.split("/")
        if len(parts) != 2:
            continue
        folder_name, fname = parts
        if fname.endswith(".npy"):
            png_path = prefix + folder_name + "/" + fname.replace(".npy", ".png")
            if png_path in all_files:
                folders[folder_name].append((f, png_path))

    for folder_name in folders:
        pairs = sorted(folders[folder_name], key=lambda x: x[0])[:top_k]
        folders[folder_name] = pairs
    return dict(folders)


def main():
    args = parse_args()
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("Install huggingface_hub: pip install huggingface_hub")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    folders_map = _collect_hf_pairs(args.repo_id, args.train_subdir, args.top_k)
    folder_names = sorted(folders_map.keys())
    if args.max_folders is not None:
        folder_names = folder_names[: args.max_folders]
    if not folder_names:
        raise RuntimeError(f"No .npy+.png pairs under {args.train_subdir}/")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    scale = ckpt.get("scale", args.scale)
    model = Stage1RNN(in_channels=3, hidden_channels=64, chunk_size=args.chunk_size, use_decoder=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    rnn_psnr, rnn_mssim, rnn_lpips = [], [], []
    naive_psnr, naive_mssim, naive_lpips = [], [], []
    n = 0

    for folder_name in folder_names:
        for npy_rel, png_rel in folders_map[folder_name]:
            if args.max_samples is not None and n >= args.max_samples:
                break
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
                gt_float = (gt.astype(np.float32) / 255.0).clip(0, 1)

                naive_img = naive_sum(frames, num_frames=args.naive_frames, to_uint8=False)
                naive_img = (np.clip(naive_img, 0, 1) * 255).astype(np.uint8)

                def chunk_iter():
                    chunk_sz = args.chunk_size
                    for start in range(0, frames_ds.shape[0], chunk_sz):
                        ch = frames_ds[start : start + chunk_sz]
                        yield torch.from_numpy(ch).permute(0, 3, 1, 2).to(device).float()

                with torch.no_grad():
                    _h, _c, decoded = model.forward_chunked(chunk_iter(), h=None, c=None)
                decoded = decoded.squeeze(0)
                decoded_up = F.interpolate(
                    decoded.unsqueeze(0), size=(800, 800), mode="bilinear", align_corners=False
                ).squeeze(0)
                rnn_img = (decoded_up.permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)

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
                print(f"  [{n}] {folder_name} PSNR RNN {psnr_r:.2f} dB", flush=True)
            finally:
                if npy_path.exists():
                    npy_path.unlink()
                if png_path.exists():
                    png_path.unlink()

        if args.max_samples is not None and n >= args.max_samples:
            break

    for d in sorted(work_dir.rglob("*"), key=lambda x: len(x.parts), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()

    def mean(x, skip_none=True):
        x = [v for v in x if v is not None] if skip_none else x
        return sum(x) / len(x) if x else 0.0

    print(f"\nValidated {n} HF samples (repo={args.repo_id}, scale={scale}, chunk_size={args.chunk_size})")
    print("-" * 60)
    print(f"{'Method':<12}  {'PSNR (dB)':>12}  {'MS-SSIM':>12}  {'LPIPS':>12}")
    print("-" * 60)
    lpips_rnn_str = f"{mean(rnn_lpips):.6f}" if any(v is not None for v in rnn_lpips) else "N/A"
    lpips_naive_str = f"{mean(naive_lpips):.6f}" if any(v is not None for v in naive_lpips) else "N/A"
    print(f"{'RNN':<12}  {mean(rnn_psnr):>12.4f}  {mean(rnn_mssim):>12.6f}  {lpips_rnn_str:>12}")
    print(f"{'Naive sum':<12}  {mean(naive_psnr):>12.4f}  {mean(naive_mssim):>12.6f}  {lpips_naive_str:>12}")
    print("-" * 60)
    print("(PSNR/MS-SSIM: higher is better; LPIPS: lower is better)")


if __name__ == "__main__":
    main()
