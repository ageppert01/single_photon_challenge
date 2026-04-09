from __future__ import annotations

import numpy as np


def load_photoncube(path: str) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def spc_avg_to_rgb(avg_image: np.ndarray, factor: float = 0.5) -> np.ndarray:
    """Invert the SPC non-linear response to recover linear RGB flux.

    The SPC detection probability relates to flux as:
        p = 1 - exp(-flux * factor)
    Inverting gives:
        flux = -log(1 - p) / factor

    Args:
        avg_image: Averaged binary frames in [0, 1].
        factor: Dead-time / exposure parameter (default 0.5).
    """
    eps = 1e-7
    p = np.clip(avg_image, 0.0, 1.0 - eps)
    return -np.log(1.0 - p) / factor


def linearrgb_to_srgb(image: np.ndarray) -> np.ndarray:
    """Standard linear RGB to sRGB gamma curve."""
    out = np.where(
        image <= 0.0031308,
        12.92 * image,
        1.055 * np.power(np.clip(image, 0.0031308, None), 1.0 / 2.4) - 0.055,
    )
    return np.clip(out, 0.0, 1.0)


def naive_sum_preprocess(
    photoncube: np.ndarray,
    num_frames: int = 16,
) -> np.ndarray:
    """Average the last num_frames binary frames from a photoncube.

    Returns values in [0, 1] representing detection probability per pixel.
    """
    frames = photoncube[-num_frames:]
    frames = np.unpackbits(frames, axis=2)
    image = frames.sum(axis=0) / num_frames
    return image.astype(np.float32)


def photoncube_file_to_tensor(
    path: str,
    num_frames: int = 16,
    invert_response: bool = True,
    invert_factor: float = 0.5,
    tonemap: bool = True,
):
    """Load a photoncube .npy file and convert to a [-1, 1] tensor.

    Pipeline (matching the Single Photon Challenge FAQ naive sum):
        1. Average last num_frames binary frames -> detection probability [0, 1]
        2. (optional) Invert SPC response -> linear RGB flux
        3. (optional) Apply sRGB tonemap
        4. Normalize to [-1, 1]
    """
    import torch
    photoncube = load_photoncube(path)

    image = naive_sum_preprocess(photoncube, num_frames=num_frames)

    if invert_response:
        image = spc_avg_to_rgb(image, factor=invert_factor)

    if tonemap:
        image = linearrgb_to_srgb(image)

    # Normalize to [-1, 1]
    image = np.clip(image, 0.0, 1.0)
    image = (image - 0.5) * 2.0

    tensor = torch.from_numpy(image).permute(2, 0, 1)  # (C, H, W)
    return tensor