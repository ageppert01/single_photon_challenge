# eval_single.py
# Required deps:
#   pip install pytorch-msssim lpips
#
# Purpose:
#   Evaluate a single (gt, noisy/restored) image pair with:
#     - PSNR (dB)
#     - MS-SSIM
#     - LPIPS
#
# Design notes for reuseability:
#   - Inputs can be numpy or torch, HWC or CHW, uint8 or float.
#   - All normalization/shape validation is centralized in _to_float01_hwc().
#   - LPIPS network construction is cached to avoid expensive reinitialization.

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np
import torch

from pytorch_msssim import ms_ssim as _ms_ssim
import lpips as _lpips

# Public-facing functions accept either numpy arrays or torch tensors.
_ArrayLike = Union[np.ndarray, torch.Tensor]

# Cache LPIPS model(s) by (device, net) so repeated calls are fast.
# Keying by device string is simple and avoids storing device objects as keys.
_LPIPS_CACHE = {}


def _to_float01_hwc(img: _ArrayLike, name: str) -> torch.Tensor:
    """
    Convert an input image to torch.float32 HWC format in [0, 1] on CPU.

    Why this helper exists:
      - Keeps input normalization rules in one place.
      - Makes eval_image_pair() easier to read.
      - Enforces consistent interpretation of dtype/range/layout.

    Accepts:
      - numpy HWC or CHW
      - torch HWC or CHW
      - uint8 [0,255] or float
        * Float is assumed to be [0,1] unless values look like [0,255].

    Returns:
      torch.Tensor of shape (800, 800, 3), dtype float32, values clipped to [0,1].
    """
    # Convert numpy -> torch; detach torch to ensure no autograd history leaks in.
    if isinstance(img, np.ndarray):
        t = torch.from_numpy(img)
    elif torch.is_tensor(img):
        t = img.detach()
    else:
        raise TypeError(f"{name} must be a numpy.ndarray or torch.Tensor, got {type(img)}")

    # Normalize/validate on CPU for predictability; later we move to device.
    t = t.to("cpu")

    # Require a single image (no batch dimension).
    if t.ndim != 3:
        raise ValueError(f"{name} must be 3D (HWC or CHW), got shape {tuple(t.shape)}")

    # Accept both HWC and CHW. Convert to HWC internally.
    if t.shape[-1] == 3:  # HWC
        hwc = t
    elif t.shape[0] == 3:  # CHW -> HWC
        hwc = t.permute(1, 2, 0)
    else:
        raise ValueError(f"{name} must have 3 channels in HWC or CHW, got shape {tuple(t.shape)}")

    # Hard constraint from your task: inputs must be exactly 800x800x3.
    # If you later want arbitrary sizes, change this check and adjust MS-SSIM settings if needed.
    if hwc.shape[0] != 800 or hwc.shape[1] != 800 or hwc.shape[2] != 3:
        raise ValueError(f"{name} must be 800x800x3, got shape {tuple(hwc.shape)}")

    # Convert dtype + normalize into [0,1].
    # - uint8 is unambiguous: scale by 255.
    # - floats: assume already [0,1] unless values exceed a small threshold.
    if hwc.dtype == torch.uint8:
        hwc = hwc.to(torch.float32) / 255.0
    else:
        hwc = hwc.to(torch.float32)
        # Conservative heuristic: if max > ~1, treat as [0,255] and rescale.
        # This prevents silent misuse when users pass float images in 0..255.
        mx = float(hwc.max().item()) if hwc.numel() else 0.0
        if mx > 1.5:
            hwc = hwc / 255.0

    # Clamp to [0,1] so metric functions behave sensibly on slight overshoots.
    # If you want strict validation instead, replace with a range check + error.
    hwc = torch.clamp(hwc, 0.0, 1.0)
    return hwc


def _psnr_from_float01(gt: torch.Tensor, pred: torch.Tensor) -> float:
    """
    Compute PSNR in dB for float images in [0,1].

    PSNR = 10 * log10(MAX^2 / MSE), with MAX=1 here.
    """
    mse = torch.mean((gt - pred) ** 2).item()
    if mse == 0.0:
        # Identical images -> infinite PSNR by definition.
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def _get_lpips_model(device: torch.device, net: str = "alex") -> torch.nn.Module:
    """
    Build (or reuse) a cached LPIPS model on a given device. Default is "alex",
    but it is unclear which model challenge uses. 

    Rationale:
      - LPIPS model initialization is relatively expensive.
      - Caching avoids repeated weight loads when scoring many pairs.
    """
    key = (str(device), net)
    if key in _LPIPS_CACHE:
        return _LPIPS_CACHE[key]
    if _lpips is None:  # pragma: no cover
        raise ImportError(
            "lpips is not installed. Install with: pip install lpips"
        ) from _LPIPS_IMPORT_ERROR
    
    # LPIPS internally builds a backbone (alex/vgg/squeeze) and loads weights.
    model = _lpips.LPIPS(net=net).to(device)
    model.eval()

    # Safety: ensure inference-only mode even if user enables grads elsewhere.
    for p in model.parameters():
        p.requires_grad_(False)
    _LPIPS_CACHE[key] = model
    return model


@torch.no_grad()
def eval_image_pair(
    gt_image: _ArrayLike,
    noisy_image: _ArrayLike,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[float, float, float]:
    """
    Evaluate a single pair of images (ground truth and noisy/estimated).

    Inputs:
      gt_image:
        - shape (800,800,3) HWC or (3,800,800) CHW
        - numpy or torch
        - uint8 in [0,255] or float in [0,1] (or float in [0,255], auto-detected)
      noisy_image: same rules as gt_image
      device:
        - torch device for MS-SSIM / LPIPS computation (e.g., "cuda", "cpu")
        - if None, uses CUDA if available else CPU

    Returns:
      (psnr_db, ms_ssim_value, lpips_value)

    Metric conventions:
      - PSNR: computed in [0,1] domain (MAX=1)
      - MS-SSIM: data_range=1.0 because inputs are [0,1]
      - LPIPS: requires inputs in [-1,1], so we map via x*2-1
    """
    # Choose device once; keep PSNR computation on CPU since it’s lightweight.
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Normalize + validate shapes/ranges early so later code can assume invariants.
    gt_hwc = _to_float01_hwc(gt_image, "gt_image")
    pred_hwc = _to_float01_hwc(noisy_image, "noisy_image")

    # PSNR on CPU tensors (HWC) in [0,1].
    psnr_db = _psnr_from_float01(gt_hwc, pred_hwc)

    # Most torch vision metrics expect NCHW.
    # Add batch dim: (1,3,800,800).
    gt_nchw = gt_hwc.permute(2, 0, 1).unsqueeze(0).to(device)   # 1x3x800x800
    pr_nchw = pred_hwc.permute(2, 0, 1).unsqueeze(0).to(device)

    # MS-SSIM dependency check here (not at import) keeps module importable
    # even if user only wants PSNR.
    if _ms_ssim is None:  # pragma: no cover
        raise ImportError(
            "pytorch-msssim is not installed. Install with: pip install pytorch-msssim"
        ) from _MS_SSIM_IMPORT_ERROR

    # MS-SSIM: higher is better; 1.0 means identical.
    ms_ssim_val = float(_ms_ssim(pr_nchw, gt_nchw, data_range=1.0, size_average=True).item())

    # LPIPS: lower is better; 0 means perceptually identical.
    # LPIPS expects normalized inputs in [-1,1].
    lpips_model = _get_lpips_model(device=device, net="alex")
    gt_lp = gt_nchw * 2.0 - 1.0
    pr_lp = pr_nchw * 2.0 - 1.0
    lpips_val = float(lpips_model(pr_lp, gt_lp).mean().item())

    return psnr_db, ms_ssim_val, lpips_val


if __name__ == "__main__":  # pragma: no cover
    # Minimal smoke test to verify imports + metric execution paths.
    # Uses synthetic noise so the numbers are stable-ish and non-trivial.
    gt = (np.random.rand(800, 800, 3) * 255).astype(np.uint8)
    noisy = np.clip(gt.astype(np.int16) + np.random.randint(-10, 11, gt.shape), 0, 255).astype(np.uint8)
    psnr, msssim, lp = eval_image_pair(gt, noisy)
    print(f"PSNR: {psnr:.4f} dB")
    print(f"MS-SSIM: {msssim:.6f}")
    print(f"LPIPS: {lp:.6f}")