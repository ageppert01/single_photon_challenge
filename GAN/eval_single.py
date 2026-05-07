"""
Evaluate a (ground truth, predicted) image pair with PSNR, MS-SSIM, and LPIPS.

Dependencies: pip install pytorch-msssim lpips
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np
import torch

from pytorch_msssim import ms_ssim as _ms_ssim
import lpips as _lpips

_ArrayLike = Union[np.ndarray, torch.Tensor]
_LPIPS_CACHE = {}


def _to_float01_hwc(img: _ArrayLike, name: str) -> torch.Tensor:
    """Convert numpy/torch HWC or CHW uint8/float image to float32 HWC in [0, 1]."""
    if isinstance(img, np.ndarray):
        t = torch.from_numpy(img)
    elif torch.is_tensor(img):
        t = img.detach()
    else:
        raise TypeError(f"{name} must be a numpy.ndarray or torch.Tensor, got {type(img)}")

    t = t.to("cpu")

    if t.ndim != 3:
        raise ValueError(f"{name} must be 3D (HWC or CHW), got shape {tuple(t.shape)}")

    if t.shape[-1] == 3:
        hwc = t
    elif t.shape[0] == 3:
        hwc = t.permute(1, 2, 0)
    else:
        raise ValueError(f"{name} must have 3 channels in HWC or CHW, got shape {tuple(t.shape)}")

    if hwc.shape[2] != 3:
        raise ValueError(f"{name} must have 3 channels, got shape {tuple(hwc.shape)}")
    if hwc.shape[0] < 160 or hwc.shape[1] < 160:
        raise ValueError(f"{name} must be at least 160x160 for MS-SSIM, got {tuple(hwc.shape)}")

    if hwc.dtype == torch.uint8:
        hwc = hwc.to(torch.float32) / 255.0
    else:
        hwc = hwc.to(torch.float32)
        # Float values > 1.5 are assumed to be in [0, 255] range.
        if float(hwc.max().item()) > 1.5:
            hwc = hwc / 255.0

    return torch.clamp(hwc, 0.0, 1.0)


def _psnr_from_float01(gt: torch.Tensor, pred: torch.Tensor) -> float:
    mse = torch.mean((gt - pred) ** 2).item()
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def _get_lpips_model(device: torch.device, net: str = "alex") -> torch.nn.Module:
    key = (str(device), net)
    if key not in _LPIPS_CACHE:
        model = _lpips.LPIPS(net=net).to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _LPIPS_CACHE[key] = model
    return _LPIPS_CACHE[key]


@torch.no_grad()
def eval_image_pair(
    gt_image: _ArrayLike,
    noisy_image: _ArrayLike,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[float, float, float]:
    """
    Returns (psnr_db, ms_ssim_value, lpips_value) for a gt/predicted image pair.
    Inputs: numpy or torch, HWC or CHW, uint8 [0,255] or float [0,1].
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    gt_hwc   = _to_float01_hwc(gt_image,    "gt_image")
    pred_hwc = _to_float01_hwc(noisy_image, "noisy_image")

    psnr_db = _psnr_from_float01(gt_hwc, pred_hwc)

    gt_nchw = gt_hwc.permute(2, 0, 1).unsqueeze(0).to(device)
    pr_nchw = pred_hwc.permute(2, 0, 1).unsqueeze(0).to(device)

    ms_ssim_val = float(_ms_ssim(pr_nchw, gt_nchw, data_range=1.0, size_average=True).item())

    lpips_model = _get_lpips_model(device=device, net="alex")
    lpips_val   = float(lpips_model(pr_nchw * 2 - 1, gt_nchw * 2 - 1).mean().item())

    return psnr_db, ms_ssim_val, lpips_val


if __name__ == "__main__":
    gt    = (np.random.rand(800, 800, 3) * 255).astype(np.uint8)
    noisy = np.clip(gt.astype(np.int16) + np.random.randint(-10, 11, gt.shape), 0, 255).astype(np.uint8)
    psnr, msssim, lp = eval_image_pair(gt, noisy)
    print(f"PSNR: {psnr:.4f} dB  MS-SSIM: {msssim:.6f}  LPIPS: {lp:.6f}")
